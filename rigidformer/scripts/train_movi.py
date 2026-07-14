from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rigidformer import Rigidformer
from rigidformer.movi import MoviMetadataDataset, movi_collate_fn, movi_sample_paths
from rigidformer.movi_render import render_movi_rollout
from rigidformer.pose_metrics import pose_errors_from_point_trajectories, summarize_pose_errors


def _default_hidden_layers(object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if cross_depth == 1:
        return (object_depth,)

    return tuple(int(i * object_depth / (cross_depth - 1)) for i in range(cross_depth))


def _parse_hidden_layers(value: str | None, object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if value is None or value == "":
        return _default_hidden_layers(object_depth, cross_depth)

    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(layers) != cross_depth:
        raise ValueError("object_hidden_layers length must match anchor_cross_attn_depth")

    return layers


def _parse_stride_choices(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None

    choices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not choices:
        return None

    if any(choice < 1 for choice in choices):
        raise ValueError("All --stride-choices values must be >= 1")

    return choices


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
        moved[key] = value.to(device, non_blocking = True) if torch.is_tensor(value) else value
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
    }


def _z_rotation_matrix(angle_degrees: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    angle = torch.tensor(math.radians(angle_degrees), device = device, dtype = dtype)
    cos = angle.cos()
    sin = angle.sin()
    zero = torch.zeros((), device = device, dtype = dtype)
    one = torch.ones((), device = device, dtype = dtype)

    return torch.stack((
        torch.stack((cos, -sin, zero)),
        torch.stack((sin, cos, zero)),
        torch.stack((zero, zero, one)),
    ))


def _maybe_apply_rotation_augmentation(batch: dict[str, Any], args: argparse.Namespace) -> int | None:
    if args.rotation_augment_prob <= 0. or random.random() >= args.rotation_augment_prob:
        return None

    angle_choices = list(range(args.rotation_augment_step_deg, 360, args.rotation_augment_step_deg))
    angle_degrees = random.choice(angle_choices)
    pos = batch["object_pos"]
    rotation = _z_rotation_matrix(angle_degrees, device = pos.device, dtype = pos.dtype)

    for key in ("object_pos_prev", "object_pos", "object_pos_next", "object_first_frame_pos"):
        batch[key] = batch[key] @ rotation.T

    return angle_degrees


def _make_loader(
    dataset_dir: Path,
    objects_dir: Path,
    split: str,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = MoviMetadataDataset(
        dataset_dir,
        objects_dir,
        split = split,
        max_points = args.max_points,
        min_stride = args.min_stride,
        max_stride = args.max_stride,
        stride_choices = args.parsed_stride_choices,
        sample_count = args.sample_count if split == "train" else args.val_sample_count,
        seed = args.seed,
        random_points = not args.deterministic_points,
        object_permutation_prob = args.object_permutation_prob if split == "train" else 0.,
    )

    return DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = shuffle,
        num_workers = args.num_workers,
        collate_fn = movi_collate_fn,
        pin_memory = torch.cuda.is_available(),
        persistent_workers = args.num_workers > 0,
    )


def _render_sample_path(args: argparse.Namespace) -> Path:
    if args.render_sample_id is not None:
        return args.dataset_dir / args.render_sample_id

    paths = movi_sample_paths(args.dataset_dir, split = args.render_split, seed = args.seed)
    if args.render_sample_index >= len(paths):
        raise IndexError(
            f"render-sample-index {args.render_sample_index} out of range "
            f"for split {args.render_split!r} with {len(paths)} samples"
        )

    return paths[args.render_sample_index]


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
    sample_path = _render_sample_path(args)
    output_dir = args.output_dir / "renders" / f"epoch_{epoch + 1:04d}"
    render_max_points = args.render_max_points or args.max_points

    was_training = model.training
    model.eval()
    try:
        metrics = render_movi_rollout(
            model,
            sample_path,
            args.objects_dir,
            output_dir,
            start_frame = args.render_start_frame,
            steps = args.render_steps,
            stride = args.render_stride,
            max_points = render_max_points,
            random_points = args.render_random_points,
            seed = args.seed,
            video_fps = args.render_video_fps,
            device = device,
            use_amp = use_amp,
            show_progress = not args.disable_render_tqdm,
        )
    finally:
        model.train(was_training)

    metrics.update({
        "event": "render",
        "epoch": epoch + 1,
        "global_step": global_step,
        "sample_path": str(sample_path),
    })

    print(json.dumps(metrics), flush = True)

    if wandb_run is not None:
        import wandb

        wandb_run.log({
            "global_step": global_step,
            "render/epoch": epoch + 1,
            "render/rmse_mean_all_frames": metrics["rmse_mean_all_frames"],
            "render/rmse_mean_rollout_only": metrics["rmse_mean_rollout_only"],
            "render/rmse_final": metrics["rmse_final"],
            "render/pose_position_rmse_rollout": metrics["pose_position_rmse_rollout"],
            "render/pose_orientation_rmse_deg_rollout": metrics["pose_orientation_rmse_deg_rollout"],
            "render/final_overlay": wandb.Image(metrics["final_overlay"]),
            "render/rmse_plot": wandb.Image(metrics["rmse_plot"]),
            "render/pose_error_plot": wandb.Image(metrics["pose_error_plot"]),
            "render/rollout": wandb.Video(metrics["video"], fps = args.render_video_fps, format = "mp4"),
        })

    return metrics


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
            warmup_progress = 1.
        else:
            warmup_progress = step / (warmup_steps - 1)

        return base_lr * (0.1 + 0.9 * warmup_progress)

    if total_steps <= warmup_steps:
        return base_lr

    decay_steps = max(total_steps - warmup_steps - 1, 1)
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.), 1.)
    cosine = 0.5 * (1. + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


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
    point_rmse: list[float] = []
    position_rmse: list[float] = []
    orientation_rmse_deg: list[float] = []

    for step, batch in enumerate(loader):
        if steps > 0 and step >= steps:
            break

        batch = _to_device(batch, device)
        model_kwargs = _batch_to_model_kwargs(batch)

        with torch.amp.autocast(device_type = device.type, enabled = use_amp):
            loss, _, intermediates = model(**model_kwargs, return_intermediates = True)
            pred = model(
                **{key: value for key, value in model_kwargs.items() if key != "object_pos_next"},
                anchor_indices = intermediates.anchor_indices,
            )

        losses.append(float(loss.detach().cpu()))

        gt_next = batch["object_pos_next"].detach().cpu().numpy()
        pred_next = pred.object_pos_next.detach().cpu().float().numpy()
        reference = batch["object_first_frame_pos"].detach().cpu().numpy()
        point_lens = batch["object_point_lens"].detach().cpu().numpy()
        object_lens = batch["object_lens"].detach().cpu().numpy()

        for batch_index in range(gt_next.shape[0]):
            num_objects = int(object_lens[batch_index])
            squared_errors = []

            for object_index in range(num_objects):
                num_points = int(point_lens[batch_index, object_index])
                diff = pred_next[batch_index, object_index, :num_points] - gt_next[batch_index, object_index, :num_points]
                squared_errors.append(diff.reshape(-1, 3))

            all_errors = np.concatenate(squared_errors, axis = 0)
            point_rmse.append(float(np.sqrt(np.mean(all_errors ** 2))))

            pose_errors = pose_errors_from_point_trajectories(
                reference[batch_index, :num_objects],
                gt_next[batch_index:batch_index + 1, :num_objects],
                pred_next[batch_index:batch_index + 1, :num_objects],
                point_lens[batch_index, :num_objects],
                num_objects,
            )
            pose_summary = summarize_pose_errors(
                pose_errors["position_error"],
                pose_errors["orientation_error_rad"],
                rollout_start_index = 0,
            )
            position_rmse.append(pose_summary["position_rmse_rollout"])
            orientation_rmse_deg.append(pose_summary["orientation_rmse_deg_rollout"])

    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "point_rmse": float(np.mean(point_rmse)) if point_rmse else float("nan"),
        "position_rmse": float(np.mean(position_rmse)) if position_rmse else float("nan"),
        "orientation_rmse_deg": float(np.mean(orientation_rmse_deg)) if orientation_rmse_deg else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "Train Rigidformer on HOPNet/RigidFormer MOVi metadata.")
    parser.add_argument("--dataset-dir", type = Path, default = Path("data/movi/MOVi-spheres"))
    parser.add_argument("--objects-dir", type = Path, default = Path("data/movi_objects/objects"))
    parser.add_argument("--output-dir", type = Path, default = Path("runs/rigidformer-movi-aug"))
    parser.add_argument("--epochs", type = int, default = 300)
    parser.add_argument("--batch-size", type = int, default = 10)
    parser.add_argument("--num-workers", type = int, default = 2)
    parser.add_argument("--steps-per-epoch", type = int, default = 0, help = "0 means use the whole split.")
    parser.add_argument("--sample-count", type = int, default = None, help = "Limit train scenes for debugging.")
    parser.add_argument("--val-sample-count", type = int, default = None, help = "Limit validation scenes for debugging.")
    parser.add_argument("--val-every", type = int, default = 10)
    parser.add_argument("--val-steps", type = int, default = 20)
    parser.add_argument("--log-every", type = int, default = 10, help = "Print a flushed scalar log every N optimizer steps.")
    parser.add_argument("--disable-tqdm", action = "store_true", help = "Disable progress bars and use only line logs.")
    parser.add_argument("--wandb", action = "store_true", help = "Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type = str, default = "rigidformer", help = "WandB project name.")
    parser.add_argument("--wandb-entity", type = str, default = "tjrcjf410-seoul-national-university", help = "Optional WandB entity/team.")
    parser.add_argument("--wandb-run-name", type = str, default = None, help = "Optional WandB run name.")
    parser.add_argument("--wandb-group", type = str, default = None, help = "Optional WandB group.")
    parser.add_argument("--wandb-mode", type = str, default = "online", choices = ("online", "offline", "disabled"), help = "WandB mode.")
    parser.add_argument("--wandb-watch", action = "store_true", help = "Log model gradients/parameters to WandB.")
    parser.add_argument("--wandb-log-checkpoints", action = "store_true", help = "Upload latest checkpoints as WandB artifacts.")
    parser.add_argument("--save-every", type = int, default = 50, help = "Save checkpoints every N epochs. Set 0 to save only the final epoch.")
    parser.add_argument("--render-every", type = int, default = 10, help = "Render a rollout every N epochs. Set 0 to disable.")
    parser.add_argument("--render-split", type = str, default = "val", choices = ("train", "val", "test", "all"))
    parser.add_argument("--render-sample-index", type = int, default = 0)
    parser.add_argument("--render-sample-id", type = str, default = None)
    parser.add_argument("--render-start-frame", type = int, default = 0)
    parser.add_argument("--render-steps", type = int, default = 60)
    parser.add_argument("--render-stride", type = int, default = 4)
    parser.add_argument("--render-max-points", type = int, default = 256, help = "Points per object for render/eval rollouts. Set 0 to use --max-points.")
    parser.add_argument("--render-random-points", action = "store_true")
    parser.add_argument("--render-video-fps", type = int, default = 12)
    parser.add_argument("--disable-render-tqdm", action = "store_true")
    parser.add_argument("--max-points", type = int, default = 1200)
    parser.add_argument("--min-stride", type = int, default = 1)
    parser.add_argument("--max-stride", type = int, default = 4)
    parser.add_argument("--stride-choices", type = str, default = "1,5,10", help = "Comma-separated temporal strides sampled during train/val. Empty string falls back to --min-stride/--max-stride.")
    parser.add_argument("--deterministic-points", action = "store_true")
    parser.add_argument("--rotation-augment-prob", type = float, default = 0.5, help = "Training-only probability for one Z-axis batch rotation.")
    parser.add_argument("--rotation-augment-step-deg", type = int, default = 5, help = "Discrete yaw step size in degrees, giving choices step..355.")
    parser.add_argument("--object-permutation-prob", type = float, default = 0.5, help = "Training-only probability for randomly permuting object order per sample.")
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--min-lr", type = float, default = 1e-6)
    parser.add_argument("--weight-decay", type = float, default = 0.01)
    parser.add_argument("--warmup-epochs", type = float, default = 10., help = "Linear warmup duration in epochs. Starts at 10% of --lr.")
    parser.add_argument("--warmup-steps", type = int, default = 0, help = "Override warmup steps. 0 uses --warmup-epochs * steps_per_epoch.")
    parser.add_argument("--grad-clip", type = float, default = 1.)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action = "store_true")
    parser.add_argument("--paper-architecture", action = argparse.BooleanOptionalAction, default = True, help = "Use paper-style PointNet/AVP/multi-scale anchor predictor.")
    parser.add_argument("--vertex-feature-dim", type = int, default = 1024)
    parser.add_argument("--avp-dim", type = int, default = 256)
    parser.add_argument("--paper-pointnet-level-dim", type = int, default = 1024)
    parser.add_argument("--dim", type = int, default = 768)
    parser.add_argument("--dim-head", type = int, default = 128)
    parser.add_argument("--heads", type = int, default = 6)
    parser.add_argument("--num-anchors", type = int, default = 4)
    parser.add_argument("--object-self-attn-depth", type = int, default = 4)
    parser.add_argument("--anchor-cross-attn-depth", type = int, default = 4)
    parser.add_argument("--object-hidden-layers", type = str, default = None, help = "Comma-separated layer indices. Defaults to an even spread.")
    parser.add_argument("--anchor-self-attn", action = "store_true")
    parser.add_argument("--use-platonic-transformer", action = "store_true")
    parser.add_argument("--resume", type = Path, default = None)
    args = parser.parse_args()

    if not 0. <= args.rotation_augment_prob <= 1.:
        raise ValueError("Expected 0 <= --rotation-augment-prob <= 1")

    if not 0. <= args.object_permutation_prob <= 1.:
        raise ValueError("Expected 0 <= --object-permutation-prob <= 1")

    if args.rotation_augment_step_deg <= 0 or args.rotation_augment_step_deg >= 360:
        raise ValueError("Expected 0 < --rotation-augment-step-deg < 360")

    args.parsed_stride_choices = _parse_stride_choices(args.stride_choices)

    if args.parsed_stride_choices is None and args.min_stride < 1:
        raise ValueError("Expected --min-stride >= 1")

    if args.parsed_stride_choices is None and args.max_stride < args.min_stride:
        raise ValueError("Expected --max-stride >= --min-stride")

    if args.lr <= 0.:
        raise ValueError("Expected --lr > 0")

    if args.min_lr < 0.:
        raise ValueError("Expected --min-lr >= 0")

    if args.min_lr > args.lr:
        raise ValueError("Expected --min-lr <= --lr")

    if args.warmup_epochs < 0.:
        raise ValueError("Expected --warmup-epochs >= 0")

    if args.warmup_steps < 0:
        raise ValueError("Expected --warmup-steps >= 0")

    if args.vertex_feature_dim < 1:
        raise ValueError("Expected --vertex-feature-dim >= 1")

    if args.avp_dim < 1:
        raise ValueError("Expected --avp-dim >= 1")

    if args.paper_pointnet_level_dim < 1:
        raise ValueError("Expected --paper-pointnet-level-dim >= 1")

    if args.save_every < 0:
        raise ValueError("Expected --save-every >= 0")

    if args.render_every < 0:
        raise ValueError("Expected --render-every >= 0")

    if args.render_sample_index < 0:
        raise ValueError("Expected --render-sample-index >= 0")

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

    _set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    args.output_dir.mkdir(parents = True, exist_ok = True)

    train_loader = _make_loader(args.dataset_dir, args.objects_dir, "train", args, shuffle = True)
    val_loader = None
    if args.val_every > 0:
        val_loader = _make_loader(args.dataset_dir, args.objects_dir, "val", args, shuffle = False)

    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    warmup_steps = args.warmup_steps or int(round(args.warmup_epochs * steps_per_epoch))
    args.effective_warmup_steps = warmup_steps

    startup_metrics = {
        "event": "startup",
        "dataset_dir": str(args.dataset_dir),
        "objects_dir": str(args.objects_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "amp": use_amp,
        "train_scenes": len(train_loader.dataset),
        "val_scenes": len(val_loader.dataset) if val_loader is not None else 0,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "stride_choices": args.parsed_stride_choices,
        "warmup_steps": warmup_steps,
        "warmup_epochs": args.warmup_epochs,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "epochs": args.epochs,
    }

    config = _jsonable_args(args)
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent = 2)

    print(
        json.dumps(startup_metrics),
        flush = True,
    )

    object_hidden_layers = _parse_hidden_layers(
        args.object_hidden_layers,
        args.object_self_attn_depth,
        args.anchor_cross_attn_depth,
    )

    model = Rigidformer(
        dim = args.dim,
        dim_head = args.dim_head,
        heads = args.heads,
        object_self_attn_depth = args.object_self_attn_depth,
        anchor_cross_attn_depth = args.anchor_cross_attn_depth,
        object_hidden_layers = object_hidden_layers,
        anchor_self_attn = args.anchor_self_attn,
        num_anchors = args.num_anchors,
        vertex_properties_dim = 3,
        use_platonic_transformer = args.use_platonic_transformer,
        paper_architecture = args.paper_architecture,
        vertex_feature_dim = args.vertex_feature_dim,
        avp_dim = args.avp_dim,
        paper_pointnet_level_dim = args.paper_pointnet_level_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled = use_amp)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Install wandb or run without `--wandb`.") from exc

        wandb_run = wandb.init(
            project = args.wandb_project,
            entity = args.wandb_entity,
            name = args.wandb_run_name,
            group = args.wandb_group,
            mode = args.wandb_mode,
            config = config | startup_metrics,
            dir = str(args.output_dir),
        )
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric = "global_step")
        wandb.define_metric("val/*", step_metric = "global_step")
        wandb.define_metric("epoch/*", step_metric = "global_step")

        if args.wandb_watch:
            wandb.watch(
                model,
                log = "all",
                log_freq = max(args.log_every, 1),
            )

    start_epoch = 0
    global_step = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location = "cpu", weights_only = False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"resumed: {args.resume}")

    total_steps = max(1, args.epochs * steps_per_epoch)

    model.train()

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(
            train_loader,
            desc = f"epoch {epoch + 1}/{args.epochs}",
            total = steps_per_epoch,
            dynamic_ncols = True,
            file = sys.stdout,
            disable = args.disable_tqdm,
        )
        running_loss = 0.
        seen = 0

        for step, batch in enumerate(progress):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break

            batch = _to_device(batch, device)
            rotation_aug_angle_deg = _maybe_apply_rotation_augmentation(batch, args)

            lr = _learning_rate(
                global_step,
                total_steps,
                warmup_steps,
                base_lr = args.lr,
                min_lr = args.min_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none = True)

            with torch.amp.autocast(device_type = device.type, enabled = use_amp):
                loss, breakdown = model(**_batch_to_model_kwargs(batch))

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            seen += 1
            running_loss += float(loss.detach().cpu())

            step_metrics = {
                "loss": running_loss / seen,
                "acc": float(breakdown.acceleration.detach().cpu()),
                "pos": float(breakdown.position.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "rotation_aug_applied": int(rotation_aug_angle_deg is not None),
            }

            progress.set_postfix({
                "loss": f"{step_metrics['loss']:.4f}",
                "acc": f"{step_metrics['acc']:.4f}",
                "pos": f"{step_metrics['pos']:.4f}",
                "lr": f"{step_metrics['lr']:.2e}",
            })

            if args.log_every > 0 and (global_step == 1 or global_step % args.log_every == 0):
                print(
                    json.dumps({
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "global_step": global_step,
                        "rotation_aug_angle_deg": rotation_aug_angle_deg,
                        **step_metrics,
                    }),
                    flush = True,
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

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(seen, 1),
        }

        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            val_metrics = _evaluate(model, val_loader, device, args.val_steps, use_amp)
            metrics.update({
                "val_loss": val_metrics["loss"],
                "val_point_rmse": val_metrics["point_rmse"],
                "val_position_rmse": val_metrics["position_rmse"],
                "val_orientation_rmse_deg": val_metrics["orientation_rmse_deg"],
            })
            print(json.dumps({"event": "validation", **metrics}), flush = True)
            if wandb_run is not None:
                wandb_run.log({
                    "global_step": global_step,
                    "val/loss": metrics["val_loss"],
                    "val/point_rmse": metrics["val_point_rmse"],
                    "val/position_rmse": metrics["val_position_rmse"],
                    "val/orientation_rmse_deg": metrics["val_orientation_rmse_deg"],
                    "epoch/index": epoch + 1,
                })

        is_final_epoch = epoch + 1 == args.epochs
        should_save_checkpoint = is_final_epoch or (args.save_every > 0 and (epoch + 1) % args.save_every == 0)
        metrics["checkpoint_saved"] = should_save_checkpoint

        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        if should_save_checkpoint:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "metrics": metrics,
                "args": vars(args),
            }

            latest_checkpoint_path = args.output_dir / "latest.pt"
            epoch_checkpoint_path = args.output_dir / f"epoch_{epoch + 1:04d}.pt"

            torch.save(checkpoint, latest_checkpoint_path)
            torch.save(checkpoint, epoch_checkpoint_path)

            print(json.dumps({
                "event": "checkpoint",
                "path": str(latest_checkpoint_path),
                "epoch_path": str(epoch_checkpoint_path),
                **metrics,
            }), flush = True)

        if args.render_every > 0 and (epoch + 1) % args.render_every == 0:
            _render_training_rollout(
                model,
                args,
                device,
                use_amp,
                epoch = epoch,
                global_step = global_step,
                wandb_run = wandb_run,
            )

        if wandb_run is not None:
            wandb_run.log({
                "global_step": global_step,
                "epoch/train_loss": metrics["train_loss"],
                "epoch/index": epoch + 1,
            })

            if args.wandb_log_checkpoints and should_save_checkpoint:
                artifact = wandb.Artifact(
                    f"{wandb_run.id}-checkpoint",
                    type = "model",
                    metadata = metrics,
                )
                artifact.add_file(str(args.output_dir / "latest.pt"))
                wandb_run.log_artifact(
                    artifact,
                    aliases = ["latest", f"epoch-{epoch + 1}"],
                )

    print(f"training complete: {args.output_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
