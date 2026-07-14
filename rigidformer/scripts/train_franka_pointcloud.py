#!/usr/bin/env python3
"""Train RigidFormer on calibrated Franka cube-drop pointcloud trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rigidformer import Rigidformer
from rigidformer.franka_pointcloud import (
    FrankaPointCloudRigidDataset,
    franka_pointcloud_collate_fn,
    parse_stride_choices,
)
from rigidformer.franka_pointcloud_render import (
    franka_pointcloud_episode_indices,
    render_franka_pointcloud_rollout,
)


def _default_hidden_layers(object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if cross_depth == 1:
        return (object_depth,)
    return tuple(int(i * object_depth / (cross_depth - 1)) for i in range(cross_depth))


def _parse_hidden_layers(value: str | None, object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if value is None or value == "":
        return _default_hidden_layers(object_depth, cross_depth)
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(layers) != cross_depth:
        raise ValueError("--object-hidden-layers length must match --anchor-cross-attn-depth")
    return layers


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _batch_to_model_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "delta_times": batch["delta_times"],
        "vertex_properties": batch["vertex_properties"],
        "object_pos": batch["object_pos"],
        "object_pos_prev": batch["object_pos_prev"],
        "object_pos_next": batch["object_pos_next"],
        "object_first_frame_pos": batch["object_first_frame_pos"],
        "object_lens": batch["object_lens"],
        "object_point_lens": batch["object_point_lens"],
        "loss_object_mask": batch["loss_object_mask"],
    }


def _vertex_property_summary(input_file: Path) -> dict[str, Any]:
    with h5py.File(input_file, "r", locking=False) as file:
        data = file["data"]
        if "vertex_properties" not in data:
            return {"vertex_properties_present": False}

        props = np.asarray(data["vertex_properties"], dtype=np.float32)
        config = json.loads(str(file.attrs.get("rigidformer_pointcloud_config", "{}")))
        object_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in np.asarray(data.get("object_names", []))
        ]

    flat = props.reshape(-1, props.shape[-1])
    summary: dict[str, Any] = {
        "vertex_properties_present": True,
        "vertex_properties_shape": list(props.shape),
        "vertex_properties_semantics": config.get("vertex_properties_semantics", "unknown"),
        "physics_properties": bool(config.get("physics_properties", False)),
        "vertex_properties_min": flat.min(axis=0).tolist(),
        "vertex_properties_max": flat.max(axis=0).tolist(),
        "vertex_properties_mean": flat.mean(axis=0).tolist(),
    }

    if props.ndim == 3 and props.shape[1] == len(object_names):
        per_object: dict[str, Any] = {}
        for object_index, object_name in enumerate(object_names):
            values = props[:, object_index, :]
            per_object[object_name] = {
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
                "mean": values.mean(axis=0).tolist(),
            }
        summary["vertex_properties_per_object"] = per_object
    else:
        summary["vertex_properties_values"] = props.tolist()

    return summary


def _z_rotation_matrix(angle_degrees: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    angle = torch.tensor(math.radians(angle_degrees), device=device, dtype=dtype)
    cos = angle.cos()
    sin = angle.sin()
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    return torch.stack(
        (
            torch.stack((cos, -sin, zero)),
            torch.stack((sin, cos, zero)),
            torch.stack((zero, zero, one)),
        )
    )


def _maybe_apply_rotation_augmentation(batch: dict[str, Any], args: argparse.Namespace) -> int | None:
    if args.rotation_augment_prob <= 0.0 or random.random() >= args.rotation_augment_prob:
        return None
    angle_choices = list(range(args.rotation_augment_step_deg, 360, args.rotation_augment_step_deg))
    angle_degrees = random.choice(angle_choices)
    pos = batch["object_pos"]
    rotation = _z_rotation_matrix(angle_degrees, device=pos.device, dtype=pos.dtype)
    for key in ("object_pos_prev", "object_pos", "object_pos_next", "object_first_frame_pos"):
        batch[key] = batch[key] @ rotation.T
    return angle_degrees


def _make_loader(input_file: Path, split: str, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    dataset = FrankaPointCloudRigidDataset(
        input_file,
        split=split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_points=args.max_points,
        min_stride=args.min_stride,
        max_stride=args.max_stride,
        stride_choices=args.parsed_stride_choices,
        sample_count=args.sample_count if split == "train" else args.val_sample_count,
        seed=args.seed,
        random_points=not args.deterministic_points,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=franka_pointcloud_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def _render_episode_indices(args: argparse.Namespace) -> list[int]:
    if args.render_episode_index is not None:
        start = int(args.render_episode_index)
        stop = start + int(args.render_num_episodes)
        all_indices = franka_pointcloud_episode_indices(
            args.input_file,
            split="all",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        if stop > len(all_indices):
            raise IndexError(
                f"render-episode-index range [{start}, {stop}) out of range "
                f"for dataset with {len(all_indices)} episodes"
            )
        return list(range(start, stop))

    indices = franka_pointcloud_episode_indices(
        args.input_file,
        split=args.render_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    stop = int(args.render_sample_index) + int(args.render_num_episodes)
    if stop > len(indices):
        raise IndexError(
            f"render sample range [{args.render_sample_index}, {stop}) out of range "
            f"for split {args.render_split!r} with {len(indices)} episodes"
        )

    return [int(index) for index in indices[int(args.render_sample_index) : stop]]


def _render_training_rollout(
    model: Rigidformer,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    *,
    epoch: int,
    global_step: int,
    wandb_run: Any | None,
) -> dict[str, Any]:
    episode_indices = _render_episode_indices(args)
    output_dir = args.output_dir / "renders" / f"epoch_{epoch + 1:04d}"
    render_max_points = args.render_max_points or args.max_points
    all_metrics: list[dict[str, Any]] = []

    was_training = model.training
    model.eval()
    try:
        for render_slot, episode_index in enumerate(episode_indices):
            episode_output_dir = (
                output_dir
                if len(episode_indices) == 1
                else output_dir / f"episode_{render_slot:02d}_idx_{episode_index:06d}"
            )
            metrics = render_franka_pointcloud_rollout(
                model,
                args.input_file,
                episode_output_dir,
                episode_index=episode_index,
                start_frame=args.render_start_frame,
                steps=args.render_steps,
                stride=args.render_stride,
                max_points=render_max_points,
                random_points=args.render_random_points,
                seed=args.seed + render_slot,
                video_fps=args.render_video_fps,
                teacher_force_context=args.render_teacher_force_context,
                device=device,
                use_amp=use_amp,
                show_progress=not args.disable_render_tqdm,
            )
            metrics.update(
                {
                    "event": "render",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "input_file": str(args.input_file),
                    "render_slot": render_slot,
                    "render_count": len(episode_indices),
                }
            )
            all_metrics.append(metrics)
            print(json.dumps(metrics), flush=True)
    finally:
        model.train(was_training)

    summary = {
        "event": "render_summary",
        "epoch": epoch + 1,
        "global_step": global_step,
        "input_file": str(args.input_file),
        "render_count": len(all_metrics),
        "episode_indices": episode_indices,
        "rmse_mean_rollout_only": float(np.nanmean([m["rmse_mean_rollout_only"] for m in all_metrics])),
        "rmse_final": float(np.nanmean([m["rmse_final"] for m in all_metrics])),
        "pose_position_rmse_rollout": float(np.nanmean([m["pose_position_rmse_rollout"] for m in all_metrics])),
        "pose_orientation_rmse_deg_rollout": float(
            np.nanmean([m["pose_orientation_rmse_deg_rollout"] for m in all_metrics])
        ),
    }
    print(json.dumps(summary), flush=True)

    if wandb_run is not None:
        import wandb

        payload: dict[str, Any] = {
            "global_step": global_step,
            "render/epoch": epoch + 1,
            "render/count": len(all_metrics),
            "render/rmse_mean_rollout_only": summary["rmse_mean_rollout_only"],
            "render/rmse_final": summary["rmse_final"],
            "render/pose_position_rmse_rollout": summary["pose_position_rmse_rollout"],
            "render/pose_orientation_rmse_deg_rollout": summary["pose_orientation_rmse_deg_rollout"],
        }
        for render_slot, metrics in enumerate(all_metrics):
            prefix = "render" if len(all_metrics) == 1 else f"render/episode_{render_slot:02d}"
            payload.update(
                {
                    f"{prefix}/episode_index": metrics["episode_index"],
                    f"{prefix}/rmse_mean_all_frames": metrics["rmse_mean_all_frames"],
                    f"{prefix}/rmse_mean_rollout_only": metrics["rmse_mean_rollout_only"],
                    f"{prefix}/rmse_final": metrics["rmse_final"],
                    f"{prefix}/rmse_all_objects_mean": metrics["rmse_all_objects_mean"],
                    f"{prefix}/rmse_all_objects_final": metrics["rmse_all_objects_final"],
                    f"{prefix}/pose_position_rmse_rollout": metrics["pose_position_rmse_rollout"],
                    f"{prefix}/pose_orientation_rmse_deg_rollout": metrics["pose_orientation_rmse_deg_rollout"],
                    f"{prefix}/final_overlay": wandb.Image(metrics["final_overlay"]),
                    f"{prefix}/rmse_plot": wandb.Image(metrics["rmse_plot"]),
                    f"{prefix}/pose_error_plot": wandb.Image(metrics["pose_error_plot"]),
                    f"{prefix}/rollout": wandb.Video(metrics["video"], fps=args.render_video_fps, format="mp4"),
                }
            )
        wandb_run.log(payload)

    return summary


def _learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    *,
    base_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        if warmup_steps == 1:
            warmup_progress = 1.0
        else:
            warmup_progress = step / (warmup_steps - 1)
        return base_lr * (0.1 + 0.9 * warmup_progress)

    if total_steps <= warmup_steps:
        return base_lr

    decay_steps = max(total_steps - warmup_steps - 1, 1)
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _masked_point_rmse(
    pred_next: torch.Tensor,
    gt_next: torch.Tensor,
    point_lens: torch.Tensor,
    object_lens: torch.Tensor,
    loss_object_mask: torch.Tensor,
) -> float:
    sq_errors: list[torch.Tensor] = []
    batch, max_objects = pred_next.shape[:2]
    for batch_index in range(batch):
        num_objects = int(object_lens[batch_index].item())
        for object_index in range(num_objects):
            if object_index >= max_objects or not bool(loss_object_mask[batch_index, object_index].item()):
                continue
            num_points = int(point_lens[batch_index, object_index].item())
            diff = pred_next[batch_index, object_index, :num_points] - gt_next[batch_index, object_index, :num_points]
            sq_errors.append(diff.square().reshape(-1))
    if not sq_errors:
        return float("nan")
    return float(torch.cat(sq_errors).mean().sqrt().detach().cpu())


@torch.no_grad()
def _evaluate(
    model: Rigidformer,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    acc_losses: list[float] = []
    pos_losses: list[float] = []
    point_rmses: list[float] = []

    for step, batch in enumerate(loader):
        if steps > 0 and step >= steps:
            break
        batch = _to_device(batch, device)
        model_kwargs = _batch_to_model_kwargs(batch)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss, breakdown, intermediates = model(**model_kwargs, return_intermediates=True)
            pred = model(
                **{key: value for key, value in model_kwargs.items() if key != "object_pos_next"},
                anchor_indices=intermediates.anchor_indices,
            )

        losses.append(float(loss.detach().cpu()))
        acc_losses.append(float(breakdown.acceleration.detach().cpu()))
        pos_losses.append(float(breakdown.position.detach().cpu()))
        point_rmses.append(
            _masked_point_rmse(
                pred.object_pos_next.float(),
                batch["object_pos_next"].float(),
                batch["object_point_lens"],
                batch["object_lens"],
                batch["loss_object_mask"],
            )
        )

    model.train()
    return {
        "loss": float(np.nanmean(losses)) if losses else float("nan"),
        "acceleration_loss": float(np.nanmean(acc_losses)) if acc_losses else float("nan"),
        "position_loss": float(np.nanmean(pos_losses)) if pos_losses else float("nan"),
        "point_rmse": float(np.nanmean(point_rmses)) if point_rmses else float("nan"),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: Rigidformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="Calibrated HDF5 from calibrate_franka_pointcloud.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rigidformer-franka-pointcloud"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means use the whole train split.")
    parser.add_argument("--sample-count", type=int, default=None, help="Limit train samples for debugging.")
    parser.add_argument("--val-sample-count", type=int, default=None, help="Limit validation samples for debugging.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs. 0 disables validation.")
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="rigidformer", help="WandB project name.")
    parser.add_argument("--wandb-entity", type=str, default="tjrcjf410-seoul-national-university", help="Optional WandB entity/team.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional WandB run name.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional WandB group.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="WandB mode.",
    )
    parser.add_argument("--wandb-watch", action="store_true", help="Log model gradients/parameters to WandB.")
    parser.add_argument("--wandb-log-checkpoints", action="store_true", help="Upload latest checkpoints as WandB artifacts.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoints every N epochs; final is always saved.")
    parser.add_argument("--render-every", type=int, default=1, help="Render a validation rollout every N epochs. Set 0 to disable.")
    parser.add_argument("--render-split", type=str, default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--render-sample-index", type=int, default=0, help="Episode index within --render-split.")
    parser.add_argument("--render-num-episodes", type=int, default=4, help="Number of consecutive episodes to render.")
    parser.add_argument("--render-episode-index", type=int, default=None, help="Absolute episode index. Overrides --render-sample-index.")
    parser.add_argument("--render-start-frame", type=int, default=0)
    parser.add_argument("--render-steps", type=int, default=62)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--render-max-points", type=int, default=1024, help="Points per object for render rollouts. Set 0 to use --max-points.")
    parser.add_argument("--render-random-points", action="store_true")
    parser.add_argument("--render-video-fps", type=int, default=12)
    parser.add_argument(
        "--render-teacher-force-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ground-truth future positions for unsupervised/context objects such as the gripper.",
    )
    parser.add_argument("--disable-render-tqdm", action="store_true")
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--min-stride", type=int, default=1)
    parser.add_argument("--max-stride", type=int, default=4)
    parser.add_argument(
        "--stride-choices",
        type=str,
        default="1,2,4",
        help="Comma-separated temporal strides. Empty string falls back to --min-stride/--max-stride.",
    )
    parser.add_argument("--deterministic-points", action="store_true")
    parser.add_argument("--rotation-augment-prob", type=float, default=0.5)
    parser.add_argument("--rotation-augment-step-deg", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--paper-architecture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vertex-feature-dim", type=int, default=256)
    parser.add_argument("--avp-dim", type=int, default=128)
    parser.add_argument("--paper-pointnet-level-dim", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--object-self-attn-depth", type=int, default=2)
    parser.add_argument("--anchor-cross-attn-depth", type=int, default=2)
    parser.add_argument(
        "--object-hidden-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices. Defaults to an even spread.",
    )
    parser.add_argument("--anchor-self-attn", action="store_true")
    parser.add_argument("--use-platonic-transformer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if not args.input_file.is_file():
        raise FileNotFoundError(args.input_file)
    if not 0.0 < args.train_ratio <= 1.0:
        raise ValueError("Expected 0 < --train-ratio <= 1")
    if not 0.0 <= args.val_ratio <= 1.0:
        raise ValueError("Expected 0 <= --val-ratio <= 1")
    if args.train_ratio + args.val_ratio > 1.0 + 1e-6:
        raise ValueError("Expected --train-ratio + --val-ratio <= 1")
    if args.max_points < 4:
        raise ValueError("Expected --max-points >= 4")
    if args.rotation_augment_step_deg <= 0 or args.rotation_augment_step_deg >= 360:
        raise ValueError("Expected 0 < --rotation-augment-step-deg < 360")
    if not 0.0 <= args.rotation_augment_prob <= 1.0:
        raise ValueError("Expected 0 <= --rotation-augment-prob <= 1")
    if args.lr <= 0.0 or args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError("Expected 0 <= --min-lr <= --lr and --lr > 0")
    if args.save_every < 0:
        raise ValueError("Expected --save-every >= 0")
    if args.render_every < 0:
        raise ValueError("Expected --render-every >= 0")
    if args.render_sample_index < 0:
        raise ValueError("Expected --render-sample-index >= 0")
    if args.render_num_episodes < 1:
        raise ValueError("Expected --render-num-episodes >= 1")
    if args.render_episode_index is not None and args.render_episode_index < 0:
        raise ValueError("Expected --render-episode-index >= 0")
    if args.render_start_frame < 0:
        raise ValueError("Expected --render-start-frame >= 0")
    if args.render_steps < 1:
        raise ValueError("Expected --render-steps >= 1")
    if args.render_stride < 1:
        raise ValueError("Expected --render-stride >= 1")
    if args.render_max_points < 0:
        raise ValueError("Expected --render-max-points >= 0")
    if args.render_max_points not in (0, None) and args.render_max_points < 4:
        raise ValueError("Expected --render-max-points to be 0 or at least 4")

    args.parsed_stride_choices = parse_stride_choices(args.stride_choices)

    _set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = _make_loader(args.input_file, "train", args, shuffle=True)
    val_loader = None
    if args.val_every > 0 and args.val_ratio > 0.0:
        val_loader = _make_loader(args.input_file, "val", args, shuffle=False)

    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    warmup_steps = args.warmup_steps or int(round(args.warmup_epochs * steps_per_epoch))
    args.effective_warmup_steps = warmup_steps
    vertex_property_summary = _vertex_property_summary(args.input_file)

    config = _jsonable_args(args)
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    startup_metrics = {
        "event": "startup",
        "input_file": str(args.input_file),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "amp": use_amp,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "stride_choices": args.parsed_stride_choices,
        "warmup_steps": warmup_steps,
        "epochs": args.epochs,
        **vertex_property_summary,
    }
    print(json.dumps(startup_metrics), flush=True)

    object_hidden_layers = _parse_hidden_layers(
        args.object_hidden_layers,
        args.object_self_attn_depth,
        args.anchor_cross_attn_depth,
    )

    model = Rigidformer(
        dim=args.dim,
        dim_head=args.dim_head,
        heads=args.heads,
        object_self_attn_depth=args.object_self_attn_depth,
        anchor_cross_attn_depth=args.anchor_cross_attn_depth,
        object_hidden_layers=object_hidden_layers,
        anchor_self_attn=args.anchor_self_attn,
        num_anchors=args.num_anchors,
        vertex_properties_dim=3,
        use_platonic_transformer=args.use_platonic_transformer,
        paper_architecture=args.paper_architecture,
        vertex_feature_dim=args.vertex_feature_dim,
        avp_dim=args.avp_dim,
        paper_pointnet_level_dim=args.paper_pointnet_level_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    wandb_run = None
    wandb_module = None
    if args.wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError("Install wandb or run without `--wandb`. Try: pip install wandb") from exc

        wandb_run = wandb_module.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config=config | startup_metrics,
            dir=str(args.output_dir),
        )
        wandb_module.define_metric("global_step")
        wandb_module.define_metric("train/*", step_metric="global_step")
        wandb_module.define_metric("val/*", step_metric="global_step")
        wandb_module.define_metric("render/*", step_metric="global_step")
        wandb_module.define_metric("epoch/*", step_metric="global_step")

        if args.wandb_watch:
            wandb_module.watch(
                model,
                log="all",
                log_freq=max(args.log_every, 1),
            )

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"[INFO] resumed: {args.resume}")

    total_steps = max(1, args.epochs * steps_per_epoch)
    model.train()

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            total=steps_per_epoch,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=args.disable_tqdm,
        )
        running_loss = 0.0
        running_acc = 0.0
        running_pos = 0.0
        seen = 0

        for step, batch in enumerate(progress):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break

            batch = _to_device(batch, device)
            rotation_aug_angle_deg = _maybe_apply_rotation_augmentation(batch, args)

            lr = _learning_rate(global_step, total_steps, warmup_steps, base_lr=args.lr, min_lr=args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, breakdown = model(**_batch_to_model_kwargs(batch))

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            seen += 1
            loss_value = float(loss.detach().cpu())
            acc_value = float(breakdown.acceleration.detach().cpu())
            pos_value = float(breakdown.position.detach().cpu())
            running_loss += loss_value
            running_acc += acc_value
            running_pos += pos_value

            step_metrics = {
                "loss": running_loss / seen,
                "acc": running_acc / seen,
                "pos": running_pos / seen,
                "lr": optimizer.param_groups[0]["lr"],
                "rotation_aug_applied": int(rotation_aug_angle_deg is not None),
            }
            progress.set_postfix(
                {
                    "loss": f"{step_metrics['loss']:.5f}",
                    "acc": f"{step_metrics['acc']:.5f}",
                    "pos": f"{step_metrics['pos']:.5f}",
                    "lr": f"{step_metrics['lr']:.2e}",
                }
            )

            if args.log_every > 0 and (global_step == 1 or global_step % args.log_every == 0):
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "global_step": global_step,
                            "rotation_aug_angle_deg": rotation_aug_angle_deg,
                            **step_metrics,
                        }
                    ),
                    flush=True,
                )

            if wandb_run is not None:
                wandb_metrics = {
                    "global_step": global_step,
                    "train/loss": step_metrics["loss"],
                    "train/acceleration_loss": step_metrics["acc"],
                    "train/position_loss": step_metrics["pos"],
                    "train/lr": step_metrics["lr"],
                    "train/rotation_aug_applied": step_metrics["rotation_aug_applied"],
                    "train/epoch": epoch + 1,
                }
                if rotation_aug_angle_deg is not None:
                    wandb_metrics["train/rotation_aug_angle_deg"] = rotation_aug_angle_deg
                wandb_run.log(wandb_metrics)

        metrics: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(seen, 1),
            "train_acceleration_loss": running_acc / max(seen, 1),
            "train_position_loss": running_pos / max(seen, 1),
        }

        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            val_metrics = _evaluate(model, val_loader, device, args.val_steps, use_amp)
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            print(json.dumps({"event": "validation", **metrics}), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "global_step": global_step,
                        "val/loss": val_metrics["loss"],
                        "val/acceleration_loss": val_metrics["acceleration_loss"],
                        "val/position_loss": val_metrics["position_loss"],
                        "val/point_rmse": val_metrics["point_rmse"],
                        "epoch/index": epoch + 1,
                    }
                )

        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        is_final_epoch = epoch + 1 == args.epochs
        should_save_checkpoint = is_final_epoch or (args.save_every > 0 and (epoch + 1) % args.save_every == 0)
        if should_save_checkpoint:
            latest_checkpoint_path = args.output_dir / "latest.pt"
            epoch_checkpoint_path = args.output_dir / f"epoch_{epoch + 1:04d}.pt"
            _save_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                args=args,
            )
            _save_checkpoint(
                epoch_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                args=args,
            )
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "path": str(latest_checkpoint_path),
                        "epoch_path": str(epoch_checkpoint_path),
                        **metrics,
                    }
                ),
                flush=True,
            )

            if args.wandb_log_checkpoints and wandb_run is not None:
                assert wandb_module is not None
                artifact = wandb_module.Artifact(
                    f"{wandb_run.id}-checkpoint",
                    type="model",
                    metadata=metrics,
                )
                artifact.add_file(str(latest_checkpoint_path))
                wandb_run.log_artifact(
                    artifact,
                    aliases=["latest", f"epoch-{epoch + 1}"],
                )

        if args.render_every > 0 and (epoch + 1) % args.render_every == 0:
            _render_training_rollout(
                model,
                args,
                device,
                use_amp,
                epoch=epoch,
                global_step=global_step,
                wandb_run=wandb_run,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "global_step": global_step,
                    "epoch/train_loss": metrics["train_loss"],
                    "epoch/train_acceleration_loss": metrics["train_acceleration_loss"],
                    "epoch/train_position_loss": metrics["train_position_loss"],
                    "epoch/index": epoch + 1,
                }
            )

    print(f"[INFO] training complete: {args.output_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
