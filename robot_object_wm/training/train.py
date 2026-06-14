from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the training environment.
    yaml = None

from robot_object_wm.data import object_mlp_dataset
from robot_object_wm.models.world_model import (
    WMDynamicsConfig,
    build_wm_dynamics,
    context_kl_loss,
    weighted_rollout_mse,
)
from robot_object_wm.models.whole_dynamics import WholeWMDynamicsConfig, build_whole_wm_dynamics
from robot_object_wm.training.checkpoint import checkpoint_and_log_epoch, start_training_run
from robot_object_wm.training.losses import MetricAverager
from robot_object_wm.training.helper import add_wandb_args, set_seed

MODEL_NAME = "WMDynamics"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "train_config.yaml"


@dataclass
class TrainConfig:
    dataset_dir: str
    dataset_file: str | None
    model_type: str
    history_len: int
    rollout_horizon: int
    robot_dof: int
    action_dim: int
    torque_dim: int
    torque_key: str
    hidden_dim: int
    object_hidden_dim: int
    object_depth: int
    tool_z_offset: float
    dt: float
    ode_solver: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    q_weight: float
    dq_weight: float
    object_pos_weight: float
    object_quat_weight: float
    object_lin_vel_weight: float
    object_ang_vel_weight: float
    use_context_encoder: bool
    latent_dim: int
    context_encoder_hidden_dim: int
    context_encoder_depth: int
    delan_use_film: bool
    delan_film_depth: int
    lambda_context_kl: float
    train_split: float
    seed: int
    num_workers: int
    output_dir: str
    run_name: str | None
    wandb_project_name: str
    wandb_entity: str
    wandb_name: str | None
    wandb_mode: str
    wandb_artifacts: bool
    log_every_batches: int
    train_batch_fraction: float
    train_subsample_fraction: float
    val_subsample_fraction: float
    filter_pre_contact: bool
    object_displacement_threshold: float
    object_velocity_threshold: float
    contact_consecutive_steps: int
    contact_settle_steps: int
    subtract_env_origin: bool
    eval_after_train: bool
    eval_fail_on_error: bool
    eval_output_dir: str
    eval_split: str
    eval_batch_size: int | None
    eval_max_batches: int | None
    eval_max_episodes: int | None
    eval_num_workers: int
    eval_episode_plot: bool
    eval_episode_index: int
    eval_episode_name: str | None
    eval_start_t: int
    eval_rollout_steps: int
    eval_target: str
    eval_prediction_metrics: bool
    eval_pred_horizon: int
    eval_prediction_max_episodes: int
    eval_video: bool
    eval_fps: int
    eval_max_frames: int


def _load_config_file(path: str) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for TrainConfig YAML. Install it with `pip install pyyaml`.")

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Train config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(values).__name__}.")

    valid_keys = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(values) - valid_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown train config key(s) in {config_path}: {joined}")
    for key in ("dataset_dir", "dataset_file", "output_dir"):
        value = values.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            values[key] = str((config_path.parent / value).resolve())
    return values


def _require_config_keys(values: dict[str, Any], path: str | Path) -> None:
    required = {field.name for field in fields(TrainConfig)}
    missing = sorted(required - set(values))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing train config key(s) in {path}: {joined}")


def _defaults_from_config(argv: list[str] | None) -> tuple[dict[str, Any], str | None]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    known, _ = config_parser.parse_known_args(argv)

    default_values = _load_config_file(str(DEFAULT_CONFIG_PATH))
    _require_config_keys(default_values, DEFAULT_CONFIG_PATH)
    defaults = dict(default_values)
    config_path = str(Path(known.config).expanduser()) if known.config else str(DEFAULT_CONFIG_PATH)
    if Path(config_path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        defaults.update(_load_config_file(config_path))
        _require_config_keys(defaults, config_path)
    return defaults, config_path


def _build_parser(defaults: dict[str, Any], config_path: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DeLaN + object-MLP WMDynamics model.")
    parser.add_argument("--config", type=str, default=config_path, help="Optional YAML file with TrainConfig values.")
    parser.add_argument("--dataset_dir", type=str, default=defaults["dataset_dir"])
    parser.add_argument("--dataset_file", type=str, default=defaults["dataset_file"])
    parser.add_argument("--model_type", type=str, default=defaults["model_type"], choices=("split", "whole"))
    parser.add_argument("--history_len", type=int, default=defaults["history_len"])
    parser.add_argument("--rollout_horizon", type=int, default=defaults["rollout_horizon"])
    parser.add_argument("--robot_dof", type=int, default=defaults["robot_dof"])
    parser.add_argument("--action_dim", type=int, default=defaults["action_dim"])
    parser.add_argument("--torque_dim", type=int, default=defaults["torque_dim"])
    parser.add_argument(
        "--torque_key",
        type=str,
        default=defaults["torque_key"],
        choices=["applied_torque", "computed_torque"],
    )
    parser.add_argument("--hidden_dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument(
        "--object_hidden_dim",
        "--object_mlp_hidden_dim",
        dest="object_hidden_dim",
        type=int,
        default=defaults["object_hidden_dim"],
    )
    parser.add_argument(
        "--object_depth",
        "--object_mlp_depth",
        dest="object_depth",
        type=int,
        default=defaults["object_depth"],
    )
    parser.add_argument("--tool_z_offset", type=float, default=defaults["tool_z_offset"])
    parser.add_argument("--dt", type=float, default=defaults["dt"])
    parser.add_argument("--ode_solver", type=str, default=defaults["ode_solver"], choices=("euler", "rk4"))
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])

    rollout = parser.add_argument_group("rollout loss")
    rollout.add_argument("--q_weight", type=float, default=defaults["q_weight"])
    rollout.add_argument("--dq_weight", type=float, default=defaults["dq_weight"])
    rollout.add_argument("--object_pos_weight", type=float, default=defaults["object_pos_weight"])
    rollout.add_argument("--object_quat_weight", type=float, default=defaults["object_quat_weight"])
    rollout.add_argument("--object_lin_vel_weight", type=float, default=defaults["object_lin_vel_weight"])
    rollout.add_argument("--object_ang_vel_weight", type=float, default=defaults["object_ang_vel_weight"])

    context = parser.add_argument_group("context encoder and DeLaN FiLM")
    context.add_argument("--use_context_encoder", dest="use_context_encoder", action="store_true", default=defaults["use_context_encoder"])
    context.add_argument("--no_use_context_encoder", dest="use_context_encoder", action="store_false")
    context.add_argument("--latent_dim", type=int, default=defaults["latent_dim"])
    context.add_argument("--context_encoder_hidden_dim", type=int, default=defaults["context_encoder_hidden_dim"])
    context.add_argument("--context_encoder_depth", type=int, default=defaults["context_encoder_depth"])
    context.add_argument("--delan_use_film", dest="delan_use_film", action="store_true", default=defaults["delan_use_film"])
    context.add_argument("--no_delan_use_film", dest="delan_use_film", action="store_false")
    context.add_argument("--delan_film_depth", type=int, default=defaults["delan_film_depth"])
    context.add_argument("--lambda_context_kl", type=float, default=defaults["lambda_context_kl"])

    train = parser.add_argument_group("training")
    train.add_argument("--train_split", type=float, default=defaults["train_split"])
    train.add_argument("--seed", type=int, default=defaults["seed"])
    train.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    train.add_argument("--output_dir", type=str, default=defaults["output_dir"])
    train.add_argument("--run_name", type=str, default=defaults["run_name"])
    train.add_argument("--log_every_batches", type=int, default=defaults["log_every_batches"])
    train.add_argument("--train_batch_fraction", type=float, default=defaults["train_batch_fraction"])
    train.add_argument("--train_subsample_fraction", type=float, default=defaults["train_subsample_fraction"])
    train.add_argument("--val_subsample_fraction", type=float, default=defaults["val_subsample_fraction"])

    data = parser.add_argument_group("data filtering")
    data.add_argument("--filter_pre_contact", dest="filter_pre_contact", action="store_true", default=defaults["filter_pre_contact"])
    data.add_argument("--no_filter_pre_contact", dest="filter_pre_contact", action="store_false")
    data.add_argument("--object_displacement_threshold", type=float, default=defaults["object_displacement_threshold"])
    data.add_argument("--object_velocity_threshold", type=float, default=defaults["object_velocity_threshold"])
    data.add_argument("--contact_consecutive_steps", type=int, default=defaults["contact_consecutive_steps"])
    data.add_argument("--contact_settle_steps", type=int, default=defaults["contact_settle_steps"])
    data.add_argument("--subtract_env_origin", dest="subtract_env_origin", action="store_true", default=defaults["subtract_env_origin"])
    data.add_argument("--no_subtract_env_origin", dest="subtract_env_origin", action="store_false")

    eval_group = parser.add_argument_group("post-training evaluation")
    eval_group.add_argument("--eval_after_train", dest="eval_after_train", action="store_true", default=defaults["eval_after_train"])
    eval_group.add_argument("--no_eval_after_train", dest="eval_after_train", action="store_false")
    eval_group.add_argument("--eval_fail_on_error", dest="eval_fail_on_error", action="store_true", default=defaults["eval_fail_on_error"])
    eval_group.add_argument("--no_eval_fail_on_error", dest="eval_fail_on_error", action="store_false")
    eval_group.add_argument("--eval_output_dir", type=str, default=defaults["eval_output_dir"])
    eval_group.add_argument("--eval_split", type=str, default=defaults["eval_split"], choices=("all", "train", "val"))
    eval_group.add_argument("--eval_batch_size", type=int, default=defaults["eval_batch_size"])
    eval_group.add_argument("--eval_max_batches", type=int, default=defaults["eval_max_batches"])
    eval_group.add_argument("--eval_max_episodes", type=int, default=defaults["eval_max_episodes"])
    eval_group.add_argument("--eval_num_workers", type=int, default=defaults["eval_num_workers"])
    eval_group.add_argument("--eval_episode_plot", dest="eval_episode_plot", action="store_true", default=defaults["eval_episode_plot"])
    eval_group.add_argument("--no_eval_episode_plot", dest="eval_episode_plot", action="store_false")
    eval_group.add_argument("--eval_episode_index", type=int, default=defaults["eval_episode_index"])
    eval_group.add_argument("--eval_episode_name", type=str, default=defaults["eval_episode_name"])
    eval_group.add_argument("--eval_start_t", type=int, default=defaults["eval_start_t"])
    eval_group.add_argument("--eval_rollout_steps", type=int, default=defaults["eval_rollout_steps"])
    eval_group.add_argument("--eval_target", type=str, default=defaults["eval_target"])
    eval_group.add_argument(
        "--eval_prediction_metrics",
        dest="eval_prediction_metrics",
        action="store_true",
        default=defaults["eval_prediction_metrics"],
    )
    eval_group.add_argument("--no_eval_prediction_metrics", dest="eval_prediction_metrics", action="store_false")
    eval_group.add_argument("--eval_pred_horizon", type=int, default=defaults["eval_pred_horizon"])
    eval_group.add_argument("--eval_prediction_max_episodes", type=int, default=defaults["eval_prediction_max_episodes"])
    eval_group.add_argument("--eval_video", dest="eval_video", action="store_true", default=defaults["eval_video"])
    eval_group.add_argument("--no_eval_video", dest="eval_video", action="store_false")
    eval_group.add_argument("--eval_fps", type=int, default=defaults["eval_fps"])
    eval_group.add_argument("--eval_max_frames", type=int, default=defaults["eval_max_frames"])

    add_wandb_args(parser, defaults)
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    defaults, config_path = _defaults_from_config(argv)
    parser = _build_parser(defaults, config_path)
    values = vars(parser.parse_args(argv))
    values.pop("config", None)
    return TrainConfig(**values)


def resolve_hdf5_paths(cfg: TrainConfig) -> list[str]:
    if cfg.dataset_file:
        path = os.path.abspath(cfg.dataset_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return [path]
    dataset_dir = os.path.abspath(cfg.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    paths = object_mlp_dataset.discover_hdf5_files(dataset_dir)
    if not paths:
        raise FileNotFoundError(f"No *.hdf5 files found in {dataset_dir}")
    return paths


def _subsample_refs(refs: list[Any], fraction: float, seed: int, kind: str) -> list[Any]:
    if fraction >= 1.0:
        return refs
    if fraction <= 0.0:
        raise ValueError(f"--{kind}_subsample_fraction must be > 0; got {fraction}")
    n = max(1, int(round(len(refs) * fraction)))
    if n >= len(refs):
        return refs
    return random.Random(seed).sample(refs, n)


def make_dataloaders(cfg: TrainConfig):
    hdf5_paths = resolve_hdf5_paths(cfg)
    episode_refs = object_mlp_dataset.load_all_episode_refs(hdf5_paths)
    train_refs, val_refs = object_mlp_dataset.split_episode_refs(episode_refs, cfg.train_split, cfg.seed)
    n_train_before, n_val_before = len(train_refs), len(val_refs)
    train_refs = _subsample_refs(train_refs, cfg.train_subsample_fraction, cfg.seed + 1, "train")
    val_refs = _subsample_refs(val_refs, cfg.val_subsample_fraction, cfg.seed + 2, "val")

    layout = object_mlp_dataset.RobotObjectWMStateLayout(
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
    )
    dataset_kwargs = {
        "history_len": cfg.history_len,
        "rollout_horizon": cfg.rollout_horizon,
        "dt": cfg.dt,
        "torque_key": cfg.torque_key,
        "filter_pre_contact": cfg.filter_pre_contact,
        "object_displacement_threshold": cfg.object_displacement_threshold,
        "object_velocity_threshold": cfg.object_velocity_threshold,
        "contact_consecutive_steps": cfg.contact_consecutive_steps,
        "contact_settle_steps": cfg.contact_settle_steps,
        "subtract_env_origin": cfg.subtract_env_origin,
        "layout": layout,
    }
    train_dataset = object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=train_refs,
        **dataset_kwargs,
    )
    val_dataset = object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=val_refs,
        **dataset_kwargs,
    )
    meta = {
        "hdf5_paths": hdf5_paths,
        "n_episodes_total": len(episode_refs),
        "n_train_episodes_before_subsample": n_train_before,
        "n_val_episodes_before_subsample": n_val_before,
        "n_train_episodes": len(train_refs),
        "n_val_episodes": len(val_refs),
        "train_subsample_fraction": cfg.train_subsample_fraction,
        "val_subsample_fraction": cfg.val_subsample_fraction,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "torque_key": cfg.torque_key,
        "train_episodes_with_contact": train_dataset.episodes_with_contact,
        "val_episodes_with_contact": val_dataset.episodes_with_contact,
        "train_skipped_short_episodes": train_dataset.skipped_short_episodes,
        "val_skipped_short_episodes": val_dataset.skipped_short_episodes,
        "train_skipped_missing_torque_episodes": train_dataset.skipped_missing_torque_episodes,
        "val_skipped_missing_torque_episodes": val_dataset.skipped_missing_torque_episodes,
        "train_skipped_missing_object_dynamics_episodes": train_dataset.skipped_missing_object_dynamics_episodes,
        "val_skipped_missing_object_dynamics_episodes": val_dataset.skipped_missing_object_dynamics_episodes,
        "train_skipped_contact_filtered_windows": train_dataset.skipped_contact_filtered_windows,
        "val_skipped_contact_filtered_windows": val_dataset.skipped_contact_filtered_windows,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, layout, meta


def _max_batches(loader: DataLoader, fraction: float) -> int | None:
    if fraction >= 1.0:
        return None
    if fraction <= 0.0:
        raise ValueError("--train_batch_fraction must be > 0")
    return max(1, int(round(len(loader) * fraction)))


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.model_type == "whole":
        whole_cfg = WholeWMDynamicsConfig(
            robot_dof=cfg.robot_dof,
            torque_dim=cfg.torque_dim,
            hidden_dim=cfg.hidden_dim,
            dt=cfg.dt,
            ode_solver=cfg.ode_solver,
            history_len=cfg.history_len,
            use_context_encoder=cfg.use_context_encoder,
            latent_dim=cfg.latent_dim,
            context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
            context_encoder_depth=cfg.context_encoder_depth,
            delan_use_film=cfg.delan_use_film,
            delan_film_depth=cfg.delan_film_depth,
        )
        return build_whole_wm_dynamics(whole_cfg)
    if cfg.model_type != "split":
        raise ValueError(f"Unknown model_type: {cfg.model_type}")
    model_cfg = WMDynamicsConfig(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        hidden_dim=cfg.hidden_dim,
        object_hidden_dim=cfg.object_hidden_dim,
        object_depth=cfg.object_depth,
        dt=cfg.dt,
        ode_solver=cfg.ode_solver,
        tool_z_offset=cfg.tool_z_offset,
        history_len=cfg.history_len,
        use_context_encoder=cfg.use_context_encoder,
        latent_dim=cfg.latent_dim,
        context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
        context_encoder_depth=cfg.context_encoder_depth,
        delan_use_film=cfg.delan_use_film,
        delan_film_depth=cfg.delan_film_depth,
    )
    return build_wm_dynamics(model_cfg)


def _forward_model(model, batch, device: torch.device, *, return_context: bool = False):
    history_states = batch["history_states"].to(device)
    future_torques = batch["future_torques"].to(device)
    object_context = batch["object_context"].to(device)
    history_torques = batch["history_torques"].to(device) if getattr(model, "use_context_encoder", False) else None
    return model(
        history_states,
        future_torques,
        object_context,
        history_torques=history_torques,
        return_context=return_context,
    )


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().detach().cpu().item())


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach()
    finite = torch.isfinite(value)
    nan_count = int(torch.isnan(value).sum().detach().cpu().item())
    inf_count = int(torch.isinf(value).sum().detach().cpu().item())
    finite_count = int(finite.sum().detach().cpu().item())
    total = value.numel()
    if finite_count > 0:
        finite_values = value[finite]
        min_value = float(finite_values.min().detach().cpu().item())
        max_value = float(finite_values.max().detach().cpu().item())
        abs_max = float(finite_values.abs().max().detach().cpu().item())
    else:
        min_value = max_value = abs_max = float("nan")
    return (
        f"{name}: shape={tuple(value.shape)} finite={finite_count}/{total} "
        f"nan={nan_count} inf={inf_count} min={min_value:.6g} max={max_value:.6g} abs_max={abs_max:.6g}"
    )


def _raise_nonfinite_step(phase: str, batch_idx: int, tensors: dict[str, torch.Tensor]) -> None:
    summaries = [_tensor_summary(name, tensor) for name, tensor in tensors.items()]
    joined = "\n  ".join(summaries)
    raise FloatingPointError(f"Non-finite value during {phase} batch {batch_idx}:\n  {joined}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    averager = MetricAverager()
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    phase = "train" if is_train else "val"

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        future_torques = batch["future_torques"].to(device)
        future_states = batch["future_states"].to(device)

        with torch.set_grad_enabled(is_train):
            if cfg.use_context_encoder:
                pred_future, context = _forward_model(model, batch, device, return_context=True)
            else:
                pred_future = _forward_model(model, batch, device)
                context = None
            rollout_loss, rollout_metrics = weighted_rollout_mse(
                pred=pred_future,
                target=future_states,
                robot_dof=cfg.robot_dof,
                q_weight=cfg.q_weight,
                dq_weight=cfg.dq_weight,
                object_pos_weight=cfg.object_pos_weight,
                object_quat_weight=cfg.object_quat_weight,
                object_lin_vel_weight=cfg.object_lin_vel_weight,
                object_ang_vel_weight=cfg.object_ang_vel_weight,
            )
            if context is not None:
                context_loss, context_metrics = context_kl_loss(context, cfg.lambda_context_kl)
            else:
                context_loss = rollout_loss.new_zeros(())
                context_metrics = {}
            loss = rollout_loss + context_loss
            if not _is_finite(loss):
                debug_tensors = {
                    "history_states": batch["history_states"].to(device),
                    "history_torques": batch["history_torques"].to(device),
                    "future_torques": future_torques,
                    "future_states": future_states,
                    "pred_future": pred_future,
                    "rollout_loss": rollout_loss,
                    "context_loss": context_loss,
                    "loss": loss,
                }
                if context is not None:
                    debug_tensors.update(context.as_aux())
                debug_tensors.update(rollout_metrics)
                debug_tensors.update(context_metrics)
                _raise_nonfinite_step(phase, batch_idx, debug_tensors)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not _is_finite(grad_norm):
                _raise_nonfinite_step(phase, batch_idx, {"loss": loss, "grad_norm": grad_norm})
            optimizer.step()

        metrics = _step_metrics(
            loss=loss,
            rollout_loss=rollout_loss,
            pred_future=pred_future,
            future_states=future_states,
            future_torques=future_torques,
            robot_dof=cfg.robot_dof,
        )
        metrics.update(rollout_metrics)
        if context_metrics:
            metrics.update(context_metrics)
            metrics["context_kl_loss"] = context_loss.detach()
        averager.update(metrics)
        averager.step()

        if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
            print(f"  [{phase}] batch {batch_idx}/{num_batches} loss={loss.item():.6f}")

    return averager.means()


def _step_metrics(
    *,
    loss,
    rollout_loss,
    pred_future,
    future_states,
    future_torques,
    robot_dof: int,
) -> dict[str, torch.Tensor]:
    robot_state_dim = 2 * robot_dof
    one_step_q = torch.nn.functional.mse_loss(pred_future[:, 0, :robot_dof], future_states[:, 0, :robot_dof])
    one_step_object_pos = torch.nn.functional.mse_loss(
        pred_future[:, 0, robot_state_dim : robot_state_dim + 3],
        future_states[:, 0, robot_state_dim : robot_state_dim + 3],
    )
    final_step_q = torch.nn.functional.mse_loss(pred_future[:, -1, :robot_dof], future_states[:, -1, :robot_dof])
    final_step_object_pos = torch.nn.functional.mse_loss(
        pred_future[:, -1, robot_state_dim : robot_state_dim + 3],
        future_states[:, -1, robot_state_dim : robot_state_dim + 3],
    )
    metrics = {
        "loss": loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "one_step_q_mse": one_step_q.detach(),
        "one_step_object_pos_mse": one_step_object_pos.detach(),
        "final_step_q_mse": final_step_q.detach(),
        "final_step_object_pos_mse": final_step_object_pos.detach(),
        "torque_abs_mean": future_torques.abs().mean().detach(),
    }
    return metrics


def _print_run_header(cfg: TrainConfig, device: torch.device, output_dir: str, meta: dict[str, Any]) -> None:
    print("Training config:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print("Dataset meta:")
    for key, value in meta.items():
        print(f"  {key}: {value}")


def _optional_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _post_training_eval_output_dir(cfg: TrainConfig, run_output_dir: str) -> str:
    if os.path.isabs(cfg.eval_output_dir):
        return cfg.eval_output_dir
    return os.path.join(run_output_dir, cfg.eval_output_dir)


def _metric_key(value: str) -> str:
    return value.replace(os.sep, "/").replace(" ", "_")


def _flatten_eval_metrics(prefix: str, value: Any, metrics: dict[str, float]) -> None:
    if isinstance(value, bool):
        metrics[prefix] = float(value)
    elif isinstance(value, int | float):
        metrics[prefix] = float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}/{_metric_key(str(key))}" if prefix else _metric_key(str(key))
            _flatten_eval_metrics(next_prefix, item, metrics)
    elif isinstance(value, list) and value and all(isinstance(item, int | float) for item in value):
        metrics[f"{prefix}/mean"] = float(sum(value) / len(value))
        metrics[f"{prefix}/final"] = float(value[-1])


def _read_eval_metrics(output_dir: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    json_specs = {
        "summary.json": "summary",
        "episode_metrics.json": "episode",
        "prediction_metrics.json": "prediction",
    }
    for filename, prefix in json_specs.items():
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            _flatten_eval_metrics(prefix, payload, metrics)
        except Exception as exc:
            print(f"[WARN] Could not read eval metrics from {path}: {exc}")
    return metrics


def _collect_eval_media(output_dir: str, written: dict[str, str] | None = None) -> dict[str, str]:
    media_exts = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".m4v", ".gif"}
    media: dict[str, str] = {}
    seen_paths: set[str] = set()
    for key, path in (written or {}).items():
        if os.path.isfile(path) and Path(path).suffix.lower() in media_exts:
            seen_paths.add(os.path.abspath(path))
            media[_metric_key(str(key))] = path
    if not os.path.isdir(output_dir):
        return media
    for root, _, files in os.walk(output_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if Path(path).suffix.lower() not in media_exts:
                continue
            abs_path = os.path.abspath(path)
            if abs_path in seen_paths:
                continue
            rel = os.path.relpath(path, output_dir)
            key = _metric_key(os.path.splitext(rel)[0])
            media.setdefault(key, path)
            seen_paths.add(abs_path)
    return media


def run_post_training_evaluation(cfg: TrainConfig, run) -> None:
    if not cfg.eval_after_train:
        return

    output_dir = _post_training_eval_output_dir(cfg, run.output_dir)
    argv = [
        "--checkpoint",
        run.best_path,
        "--output_dir",
        output_dir,
        "--split",
        cfg.eval_split,
        "--num_workers",
        str(cfg.eval_num_workers),
        "--episode_index",
        str(cfg.eval_episode_index),
        "--start_t",
        str(cfg.eval_start_t),
        "--rollout_steps",
        str(cfg.eval_rollout_steps),
        "--target",
        cfg.eval_target,
        "--pred_horizon",
        str(cfg.eval_pred_horizon),
        "--prediction_max_episodes",
        str(cfg.eval_prediction_max_episodes),
        "--fps",
        str(cfg.eval_fps),
        "--max_frames",
        str(cfg.eval_max_frames),
    ]
    if cfg.dataset_file:
        argv.extend(["--dataset_file", cfg.dataset_file])
    else:
        argv.extend(["--dataset_dir", cfg.dataset_dir])
    _optional_arg(argv, "--batch_size", cfg.eval_batch_size)
    _optional_arg(argv, "--max_batches", cfg.eval_max_batches)
    _optional_arg(argv, "--max_episodes", cfg.eval_max_episodes)
    _optional_arg(argv, "--episode_name", cfg.eval_episode_name)
    if cfg.eval_episode_plot:
        argv.append("--episode_plot")
    if cfg.eval_prediction_metrics:
        argv.append("--prediction_metrics")
    if cfg.eval_video:
        argv.append("--video")

    print("===== Post-training evaluation =====")
    print(f"Best checkpoint: {run.best_path}")
    print(f"Eval output dir: {output_dir}")
    try:
        from robot_object_wm.eval.rollout import main as eval_main

        written = eval_main(argv)
        metrics = _read_eval_metrics(output_dir)
        media = _collect_eval_media(output_dir, written)
        if not media:
            print(f"[WARN] No eval media files found under {output_dir}.")
        run.log_evaluation(output_dir, metrics=metrics, media=media, fps=cfg.eval_fps)
    except Exception as exc:
        if cfg.eval_fail_on_error:
            raise
        print(f"[WARN] Post-training evaluation failed: {exc}")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    set_seed(cfg.seed)
    run = start_training_run(cfg, MODEL_NAME, asdict(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, layout, meta = make_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    _print_run_header(cfg, device, run.output_dir, meta)
    run.update_config({"architecture": MODEL_NAME, "data_meta": meta})

    best_val = float("inf")
    max_train_batches = _max_batches(train_loader, cfg.train_batch_fraction)

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, max_train_batches)
            val_metrics = run_epoch(model, val_loader, None, device, cfg, None)
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_rollout={val_metrics['rollout_loss']:.6f}"
            )
            best_val = checkpoint_and_log_epoch(
                run=run,
                epoch=epoch,
                checkpoint={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "architecture": MODEL_NAME,
                    "config": asdict(cfg),
                    "layout": layout.to_dict(),
                    "data_meta": meta,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=best_val,
            )
        print(f"Best checkpoint: {run.best_path}")
        print(f"Last checkpoint: {run.last_path}")
        print(f"W&B run: {run.wandb_url or cfg.wandb_mode}")
        run_post_training_evaluation(cfg, run)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
