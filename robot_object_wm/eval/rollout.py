from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from robot_object_wm.data import object_mlp_dataset
from robot_object_wm.eval.episode import (
    load_episode_data,
    load_episodes,
    per_horizon_episode_errors,
    rollout_episode,
    rollout_metrics,
)
from robot_object_wm.eval.plots import (
    save_evaluation_artifacts,
    save_metrics_json,
    save_per_horizon_csv,
    save_prediction_error_curves,
    save_trajectory_plot,
)
from robot_object_wm.models.utils import FrankaForwardKinematics
from robot_object_wm.models.rwm import RWMEnsemble
from robot_object_wm.training.train import MODEL_NAME, RWMConfig, TrainConfig, build_model, parse_args

EvalSplit = Literal["all", "train", "val"]


@dataclass
class RolloutResult:
    pred_states: np.ndarray
    gt_states: np.ndarray
    failed_step: int | None = None
    failure_reason: str | None = None


@dataclass
class LoadedWorldModel:
    checkpoint_path: str
    model: torch.nn.Module
    config: TrainConfig
    rwm_config: RWMConfig | None
    architecture: str
    layout: dict[str, Any]
    data_meta: dict[str, Any]
    epoch: int | None


@dataclass
class EvaluationSummary:
    checkpoint_path: str
    architecture: str
    split: EvalSplit
    dataset_paths: list[str]
    n_samples: int
    n_batches: int
    failed_samples: int
    metrics: dict[str, float]
    per_horizon: dict[str, np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "architecture": self.architecture,
            "split": self.split,
            "dataset_paths": self.dataset_paths,
            "n_samples": self.n_samples,
            "n_batches": self.n_batches,
            "failed_samples": self.failed_samples,
            "metrics": self.metrics,
            "per_horizon": {key: value.tolist() for key, value in self.per_horizon.items()},
        }


def euclidean_position_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1)


def rollout_mse(pred_states: np.ndarray, gt_states: np.ndarray) -> float:
    return float(np.mean((pred_states - gt_states) ** 2))


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device | str | None = None,
    *,
    strict: bool = True,
) -> LoadedWorldModel:
    resolved_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(resolved_path, map_location=torch_device, weights_only=False)
    cfg = config_from_checkpoint(checkpoint)
    rwm_cfg = rwm_config_from_checkpoint(checkpoint)
    model = build_model(cfg, device=torch_device, rwm_cfg=rwm_cfg).to(torch_device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    except RuntimeError as exc:
        raise RuntimeError("Checkpoint state_dict is not compatible with the current WMDynamics model.") from exc
    model.eval()
    return LoadedWorldModel(
        checkpoint_path=resolved_path,
        model=model,
        config=cfg,
        rwm_config=rwm_cfg,
        architecture=str(checkpoint.get("architecture", MODEL_NAME)),
        layout=dict(checkpoint.get("layout", {})),
        data_meta=dict(checkpoint.get("data_meta", {})),
        epoch=checkpoint.get("epoch"),
    )


def config_from_checkpoint(checkpoint: dict[str, Any]) -> TrainConfig:
    saved = dict(checkpoint.get("config", {}))
    defaults = asdict(parse_args([]))
    legacy_aliases = {
        "object_mlp_hidden_dim": "object_hidden_dim",
        "object_mlp_depth": "object_depth",
    }
    for old_key, new_key in legacy_aliases.items():
        if old_key in saved and new_key not in saved:
            saved[new_key] = saved[old_key]

    valid_fields = {field.name for field in fields(TrainConfig)}
    merged = defaults | {key: value for key, value in saved.items() if key in valid_fields}
    return TrainConfig(**merged)


def rwm_config_from_checkpoint(checkpoint: dict[str, Any]) -> RWMConfig | None:
    saved = checkpoint.get("rwm_config")
    if not isinstance(saved, dict):
        return None
    valid_fields = {field.name for field in fields(RWMConfig)}
    values = asdict(RWMConfig()) | {key: value for key, value in saved.items() if key in valid_fields}
    return RWMConfig(**values)


def make_eval_loader(
    cfg: TrainConfig,
    *,
    dataset_file: str | None = None,
    dataset_dir: str | None = None,
    split: EvalSplit = "val",
    batch_size: int | None = None,
    num_workers: int = 0,
    max_episodes: int | None = None,
) -> tuple[DataLoader, dict[str, Any]]:
    hdf5_paths = _resolve_hdf5_paths(
        dataset_file=dataset_file if dataset_file is not None else cfg.dataset_file,
        dataset_dir=dataset_dir if dataset_dir is not None else cfg.dataset_dir,
    )
    episode_refs = object_mlp_dataset.load_all_episode_refs(hdf5_paths)
    if split == "all":
        refs = episode_refs
    else:
        train_refs, val_refs = object_mlp_dataset.split_episode_refs(episode_refs, cfg.train_split, cfg.seed)
        refs = train_refs if split == "train" else val_refs
    if max_episodes is not None:
        refs = refs[: max(0, max_episodes)]
    if not refs:
        raise RuntimeError(f"No episodes selected for eval split '{split}'.")

    layout, dataset = _build_eval_dataset(cfg, refs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    meta = {
        "hdf5_paths": hdf5_paths,
        "split": split,
        "n_episodes_total": len(episode_refs),
        "n_eval_episodes": len(refs),
        "n_eval_samples": len(dataset),
        "layout": layout.to_dict(),
    }
    return loader, meta


def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    dataset_file: str | None = None,
    dataset_dir: str | None = None,
    split: EvalSplit = "val",
    batch_size: int | None = None,
    max_batches: int | None = None,
    max_episodes: int | None = None,
    num_workers: int = 0,
    device: torch.device | str | None = None,
) -> EvaluationSummary:
    loaded = load_checkpoint_model(checkpoint_path, device=device)
    loader, meta = make_eval_loader(
        loaded.config,
        dataset_file=dataset_file,
        dataset_dir=dataset_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        max_episodes=max_episodes,
    )
    summary = evaluate_loader(
        loaded.model,
        loader,
        loaded.config,
        device=next(loaded.model.parameters()).device,
        max_batches=max_batches,
    )
    summary.checkpoint_path = loaded.checkpoint_path
    summary.architecture = loaded.architecture
    summary.split = split
    summary.dataset_paths = list(meta["hdf5_paths"])
    return summary


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    *,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> EvaluationSummary:
    torch_device = torch.device(device or next(model.parameters()).device)
    model.eval()
    totals: dict[str, float] = {}
    horizon_totals: dict[str, np.ndarray] = {}
    n_samples = 0
    n_batches = 0
    failed_samples = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            pred = predict_batch(model, batch, torch_device)
            target = batch["future_states"].to(torch_device)
            finite_mask = torch.isfinite(pred).flatten(1).all(dim=1) & torch.isfinite(target).flatten(1).all(dim=1)
            failed_samples += int((~finite_mask).sum().item())
            if not torch.any(finite_mask):
                n_batches += 1
                continue
            pred = pred[finite_mask]
            target = target[finite_mask]
            batch_samples = int(pred.shape[0])
            _add_weighted(totals, _batch_metrics(pred, target, cfg.robot_dof), batch_samples)
            _add_horizon(horizon_totals, _horizon_metrics(pred, target, cfg.robot_dof), batch_samples)
            n_samples += batch_samples
            n_batches += 1

    if n_samples == 0:
        raise RuntimeError("No finite samples were evaluated.")
    metrics = {key: value / n_samples for key, value in totals.items()}
    per_horizon = {key: value / n_samples for key, value in horizon_totals.items()}
    return EvaluationSummary(
        checkpoint_path="",
        architecture=MODEL_NAME,
        split="all",
        dataset_paths=[],
        n_samples=n_samples,
        n_batches=n_batches,
        failed_samples=failed_samples,
        metrics=metrics,
        per_horizon=per_horizon,
    )


def predict_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(model, RWMEnsemble) or getattr(model, "model_type", None) == "rwm":
        return predict_rwm_batch(model, batch, device)

    history_states = batch["history_states"].to(device)
    future_torques = batch["future_torques"].to(device)
    object_context = batch["object_context"].to(device)
    history_torques = batch["history_torques"].to(device) if getattr(model, "use_context_encoder", False) else None
    return model(
        history_states,
        future_torques,
        object_context,
        history_torques=history_torques,
        return_aux=False,
    )


def predict_rwm_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    history_states = batch["history_states"].to(device)
    action_type = str(getattr(model, "action_type", "torque")).strip().lower()
    if action_type == "policy":
        history_actions = batch["history_actions"].to(device)
        future_actions = batch["future_actions"].to(device)
        first_actions = torch.cat([history_actions, future_actions[:, :1]], dim=1)
    elif action_type == "torque":
        first_actions = batch["history_torques"].to(device)
        future_actions = batch["future_torques"].to(device)
    else:
        raise ValueError(f"RWM action_type must be 'policy' or 'torque', got {action_type!r}.")

    pred_steps: list[torch.Tensor] = []
    try:
        if hasattr(model, "reset"):
            model.reset()
        x_state = history_states
        for step in range(future_actions.shape[1]):
            x_action = first_actions if step == 0 else future_actions[:, step : step + 1]
            pred_state, _aleatoric, _epistemic = model(x_state, x_action)
            pred_steps.append(pred_state)
            x_state = pred_state.unsqueeze(1)
    finally:
        if hasattr(model, "reset"):
            model.reset()

    if not pred_steps:
        return history_states.new_empty((history_states.shape[0], 0, history_states.shape[-1]))
    return torch.stack(pred_steps, dim=1)


def _resolve_hdf5_paths(*, dataset_file: str | None, dataset_dir: str) -> list[str]:
    if dataset_file:
        path = os.path.abspath(dataset_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return [path]
    root = os.path.abspath(dataset_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    paths = object_mlp_dataset.discover_hdf5_files(root)
    if not paths:
        raise FileNotFoundError(f"No *.hdf5 files found in {root}")
    return paths


def _build_eval_dataset(cfg: TrainConfig, refs: list[Any]):
    layout = object_mlp_dataset.RobotObjectWMStateLayout(
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
    )
    dataset = object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=refs,
        history_len=cfg.history_len,
        rollout_horizon=cfg.rollout_horizon,
        dt=cfg.dt,
        torque_key=cfg.torque_key,
        filter_pre_contact=cfg.filter_pre_contact,
        object_displacement_threshold=cfg.object_displacement_threshold,
        object_velocity_threshold=cfg.object_velocity_threshold,
        contact_consecutive_steps=cfg.contact_consecutive_steps,
        contact_settle_steps=cfg.contact_settle_steps,
        subtract_env_origin=cfg.subtract_env_origin,
        layout=layout,
    )
    return layout, dataset


def _batch_metrics(pred: torch.Tensor, target: torch.Tensor, robot_dof: int) -> dict[str, float]:
    offset = 2 * robot_dof
    object_pos_error = torch.linalg.norm(pred[..., offset : offset + 3] - target[..., offset : offset + 3], dim=-1)
    final_object_pos_error = object_pos_error[:, -1]
    return {
        "rollout_mse": float(torch.mean((pred - target).square()).item()),
        "q_mse": float(torch.mean((pred[..., :robot_dof] - target[..., :robot_dof]).square()).item()),
        "dq_mse": float(torch.mean((pred[..., robot_dof:offset] - target[..., robot_dof:offset]).square()).item()),
        "object_pos_mse": float(torch.mean((pred[..., offset : offset + 3] - target[..., offset : offset + 3]).square()).item()),
        "object_quat_mse": float(torch.mean((pred[..., offset + 3 : offset + 7] - target[..., offset + 3 : offset + 7]).square()).item()),
        "object_lin_vel_mse": float(torch.mean((pred[..., offset + 7 : offset + 10] - target[..., offset + 7 : offset + 10]).square()).item()),
        "object_ang_vel_mse": float(torch.mean((pred[..., offset + 10 : offset + 13] - target[..., offset + 10 : offset + 13]).square()).item()),
        "object_pos_error_mean": float(object_pos_error.mean().item()),
        "object_pos_error_final": float(final_object_pos_error.mean().item()),
    }


def _horizon_metrics(pred: torch.Tensor, target: torch.Tensor, robot_dof: int) -> dict[str, np.ndarray]:
    offset = 2 * robot_dof
    object_pos_error = torch.linalg.norm(pred[..., offset : offset + 3] - target[..., offset : offset + 3], dim=-1)
    batch = pred.shape[0]
    return {
        "object_pos_error": object_pos_error.sum(dim=0).detach().cpu().numpy(),
        "rollout_mse": torch.mean((pred - target).square(), dim=(0, 2)).detach().cpu().numpy() * batch,
        "q_mse": torch.mean((pred[..., :robot_dof] - target[..., :robot_dof]).square(), dim=(0, 2)).detach().cpu().numpy()
        * batch,
        "dq_mse": torch.mean((pred[..., robot_dof:offset] - target[..., robot_dof:offset]).square(), dim=(0, 2)).detach().cpu().numpy()
        * batch,
        "object_pos_mse": torch.mean((pred[..., offset : offset + 3] - target[..., offset : offset + 3]).square(), dim=(0, 2))
        .detach()
        .cpu()
        .numpy()
        * batch,
        "object_quat_mse": torch.mean((pred[..., offset + 3 : offset + 7] - target[..., offset + 3 : offset + 7]).square(), dim=(0, 2))
        .detach()
        .cpu()
        .numpy()
        * batch,
        "object_lin_vel_mse": torch.mean((pred[..., offset + 7 : offset + 10] - target[..., offset + 7 : offset + 10]).square(), dim=(0, 2))
        .detach()
        .cpu()
        .numpy()
        * batch,
        "object_ang_vel_mse": torch.mean((pred[..., offset + 10 : offset + 13] - target[..., offset + 10 : offset + 13]).square(), dim=(0, 2))
        .detach()
        .cpu()
        .numpy()
        * batch,
    }


def _add_weighted(totals: dict[str, float], metrics: dict[str, float], weight: int) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + value * weight


def _add_horizon(totals: dict[str, np.ndarray], metrics: dict[str, np.ndarray], weight: int) -> None:
    del weight
    for key, value in metrics.items():
        if key not in totals:
            totals[key] = np.zeros_like(value, dtype=np.float64)
        totals[key] += value


def parse_eval_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate WMDynamics checkpoints and write plots/metrics.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--split", choices=("all", "train", "val"), default="val")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs/wm_dynamics")

    episode = parser.add_argument_group("single episode rollout plot")
    episode.add_argument("--episode_plot", action="store_true", default=False)
    episode.add_argument("--episode_index", type=int, default=0)
    episode.add_argument("--episode_name", type=str, default=None)
    episode.add_argument("--start_t", type=int, default=25)
    episode.add_argument("--rollout_steps", type=int, default=10)
    episode.add_argument("--target", type=str, default="gripper")

    pred = parser.add_argument_group("per-time prediction metrics")
    pred.add_argument("--prediction_metrics", action="store_true", default=False)
    pred.add_argument("--pred_horizon", type=int, default=10)
    pred.add_argument("--prediction_max_episodes", type=int, default=0)

    video = parser.add_argument_group("video")
    video.add_argument("--video", action="store_true", default=False)
    video.add_argument("--video_output", type=str, default=None)
    video.add_argument("--fps", type=int, default=25)
    video.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = parse_eval_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    loaded = load_checkpoint_model(args.checkpoint, device=args.device)
    loader, meta = make_eval_loader(
        loaded.config,
        dataset_file=args.dataset_file,
        dataset_dir=args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_episodes=args.max_episodes,
    )
    summary = evaluate_loader(
        loaded.model,
        loader,
        loaded.config,
        device=next(loaded.model.parameters()).device,
        max_batches=args.max_batches,
    )
    summary.checkpoint_path = loaded.checkpoint_path
    summary.architecture = loaded.architecture
    summary.split = args.split
    summary.dataset_paths = list(meta["hdf5_paths"])
    written = save_evaluation_artifacts(summary, args.output_dir)

    dataset_file = args.dataset_file or (summary.dataset_paths[0] if summary.dataset_paths else None)
    if dataset_file is None and (args.episode_plot or args.prediction_metrics or args.video):
        raise ValueError("--dataset_file or --dataset_dir with at least one HDF5 file is required for episode outputs.")

    if args.episode_plot:
        written.update(_write_episode_plot(args, loaded, dataset_file))
    if args.prediction_metrics:
        written.update(_write_prediction_metrics(args, loaded, dataset_file))
    if args.video:
        from robot_object_wm.eval.animation import render_checkpoint_animation

        output = args.video_output or os.path.join(args.output_dir, "episode_animation.mp4")
        written["video"] = render_checkpoint_animation(
            model=loaded.model,
            cfg=loaded.config,
            dataset_file=dataset_file,
            output_path=output,
            episode_index=args.episode_index,
            episode_name=args.episode_name,
            pred_horizon=args.pred_horizon,
            fps=args.fps,
            max_frames=args.max_frames,
            device=next(loaded.model.parameters()).device,
            target=args.target,
        )

    print("===== WMDynamics Evaluation =====")
    print(f"checkpoint: {summary.checkpoint_path}")
    print(f"split: {summary.split}")
    print(f"n_samples: {summary.n_samples}")
    print(f"rollout_mse: {summary.metrics['rollout_mse']:.8f}")
    print(f"object_pos_error_mean: {summary.metrics['object_pos_error_mean']:.8f}")
    for key, path in written.items():
        print(f"{key}: {path}")
    return written


def _write_episode_plot(args, loaded: LoadedWorldModel, dataset_file: str) -> dict[str, str]:
    cfg = loaded.config
    device = next(loaded.model.parameters()).device
    episode = load_episode_data(
        dataset_file,
        episode_index=args.episode_index,
        episode_name=args.episode_name,
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
        torque_key=cfg.torque_key,
        subtract_env_origin=cfg.subtract_env_origin,
        max_frames=args.max_frames,
        dt=cfg.dt,
    )
    start_t = max(cfg.history_len - 1, args.start_t)
    steps = episode.T - start_t - 1 if args.rollout_steps <= 0 else args.rollout_steps
    rollout = rollout_episode(
        loaded.model,
        episode,
        start_t=start_t,
        rollout_steps=steps,
        history_len=cfg.history_len,
        device=device,
    )
    if rollout.steps == 0:
        metrics_path = save_metrics_json(
            {
                "checkpoint": loaded.checkpoint_path,
                "dataset_file": dataset_file,
                "episode": episode.name,
                "start_t": start_t,
                "requested_rollout_steps": steps,
                "failed_step": rollout.failed_step,
                "failure_reason": rollout.failure_reason,
                "metrics": {"evaluated_rollout_steps": 0},
            },
            os.path.join(args.output_dir, "episode_metrics.json"),
        )
        return {"episode_metrics_json": metrics_path}
    fk = FrankaForwardKinematics(robot_dof=cfg.robot_dof, tool_z_offset=cfg.tool_z_offset).to(device)
    fk.eval()
    metrics, cart = rollout_metrics(
        rollout,
        episode,
        robot_dof=cfg.robot_dof,
        target=args.target,
        fk=fk,
        device=device,
    )
    time = (np.arange(rollout.steps, dtype=np.float32) + start_t + 1) * cfg.dt
    plot_path, plot_metrics = save_trajectory_plot(
        real_episode_target=cart["real_episode_target"],
        real_episode_object=cart["real_episode_object"],
        gt_target=cart["gt_target"],
        pred_target=cart["pred_target"],
        gt_object=cart["gt_object"],
        pred_object=cart["pred_object"],
        time=time,
        output_path=os.path.join(args.output_dir, "episode_trajectory.png"),
        episode_name=episode.name,
        target_name=args.target,
        start_t=start_t,
    )
    payload = {
        "checkpoint": loaded.checkpoint_path,
        "dataset_file": dataset_file,
        "episode": episode.name,
        "start_t": start_t,
        "requested_rollout_steps": steps,
        "failed_step": rollout.failed_step,
        "failure_reason": rollout.failure_reason,
        "metrics": metrics | plot_metrics,
    }
    metrics_path = save_metrics_json(payload, os.path.join(args.output_dir, "episode_metrics.json"))
    return {"episode_plot": plot_path, "episode_metrics_json": metrics_path}


def _write_prediction_metrics(args, loaded: LoadedWorldModel, dataset_file: str) -> dict[str, str]:
    cfg = loaded.config
    device = next(loaded.model.parameters()).device
    episodes = load_episodes(
        dataset_file,
        max_episodes=args.prediction_max_episodes,
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
        torque_key=cfg.torque_key,
        subtract_env_origin=cfg.subtract_env_origin,
        dt=cfg.dt,
    )
    fk = FrankaForwardKinematics(robot_dof=cfg.robot_dof, tool_z_offset=cfg.tool_z_offset).to(device)
    fk.eval()
    per_horizon = per_horizon_episode_errors(
        loaded.model,
        episodes,
        history_len=cfg.history_len,
        pred_horizon=args.pred_horizon,
        robot_dof=cfg.robot_dof,
        target=args.target,
        fk=fk,
        device=device,
    )
    csv_path = save_per_horizon_csv(per_horizon, os.path.join(args.output_dir, "prediction_metrics_per_horizon.csv"))
    json_path = save_metrics_json(
        {
            "checkpoint": loaded.checkpoint_path,
            "dataset_file": dataset_file,
            "n_episodes": len(episodes),
            "pred_horizon": args.pred_horizon,
            "per_horizon": per_horizon,
        },
        os.path.join(args.output_dir, "prediction_metrics.json"),
    )
    plot_path = save_prediction_error_curves(
        per_horizon,
        os.path.join(args.output_dir, "prediction_error_curves.png"),
        title=f"{MODEL_NAME} {args.pred_horizon}-step prediction error",
    )
    return {
        "prediction_metrics_csv": csv_path,
        "prediction_metrics_json": json_path,
        "prediction_error_plot": plot_path,
    }


if __name__ == "__main__":
    main()
