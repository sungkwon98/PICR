from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

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
from robot_object_wm.models.rwm import RWMEnsemble
from robot_object_wm.models.whole_dynamics import WholeWMDynamicsConfig, build_whole_wm_dynamics
from robot_object_wm.training.checkpoint import checkpoint_and_log_epoch, start_training_run
from robot_object_wm.training.losses import MetricAverager
from robot_object_wm.training.helper import set_seed

MODEL_NAME = "WMDynamics"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "train_config.yaml"
DEFAULT_RWM_CONFIG_PATH = PACKAGE_ROOT / "configs" / "rwm_config.yaml"
NONE_STRINGS = {"", "none", "null", "nil"}
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
FIELD_CHOICES = {
    "model_type": ("split", "whole", "rwm"),
    "action_type": ("policy", "torque", "Torque"),
    "torque_key": ("applied_torque", "computed_torque"),
    "ode_solver": ("euler", "rk4"),
    "wandb_mode": ("online", "offline", "disabled"),
    "eval_split": ("all", "train", "val"),
}
FIELD_ALIASES = {
    "object_hidden_dim": ("object_mlp_hidden_dim",),
    "object_depth": ("object_mlp_depth",),
}


@dataclass
class TrainConfig:
    dataset_dir: str
    dataset_file: str | None
    model_type: str
    rwm_config: str | None
    history_len: int
    rollout_horizon: int
    robot_dof: int
    action_dim: int
    torque_dim: int
    action_type: str
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


@dataclass
class RWMConfig:
    architecture: str = "RNN"
    cell: str = "GRU"
    ensemble_size: int = 5
    history_horizon: int = 32
    forecast_horizon: int = 8
    rnn_num_layers: int = 2
    rnn_hidden_size: int = 256
    state_mean_head: list[int] = field(default_factory=lambda: [128])
    state_logstd_head: list[int] | None = field(default_factory=lambda: [128])
    state_loss_weight: float = 1.0
    sequence_loss_weight: float = 1.0
    bound_loss_weight: float = 0.01
    kl_loss_weight: float = 1.0
    bootstrap: bool = False


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
    for key in ("dataset_dir", "dataset_file", "output_dir", "rwm_config"):
        value = values.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            values[key] = str((config_path.parent / value).resolve())
    return values


def load_rwm_config(path: str | None) -> RWMConfig:
    if yaml is None:
        raise ImportError("PyYAML is required for RWMConfig YAML. Install it with `pip install pyyaml`.")

    config_path = Path(path).expanduser() if path else DEFAULT_RWM_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"RWM config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(loaded).__name__}.")

    valid_keys = {field.name for field in fields(RWMConfig)}
    unknown = sorted(set(loaded) - valid_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown RWM config key(s) in {config_path}: {joined}")

    values = asdict(RWMConfig()) | loaded
    cfg = RWMConfig(**values)
    _validate_rwm_config(cfg, config_path)
    return cfg


def _validate_rwm_config(cfg: RWMConfig, path: Path) -> None:
    if cfg.architecture.strip().lower() != "rnn":
        raise ValueError(f"{path}: only architecture: RNN is currently wired for training.")
    if cfg.cell.strip().lower() not in {"gru", "lstm"}:
        raise ValueError(f"{path}: cell must be GRU or LSTM.")
    if cfg.ensemble_size < 1:
        raise ValueError(f"{path}: ensemble_size must be >= 1.")
    if cfg.history_horizon < 1 or cfg.forecast_horizon < 1:
        raise ValueError(f"{path}: history_horizon and forecast_horizon must be >= 1.")
    if not cfg.state_mean_head:
        raise ValueError(f"{path}: state_mean_head must contain at least one hidden size.")
    if cfg.state_logstd_head is not None and not cfg.state_logstd_head:
        raise ValueError(f"{path}: state_logstd_head must be null or contain at least one hidden size.")


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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_STRINGS:
        return True
    if normalized in FALSE_STRINGS:
        return False
    expected = ", ".join(sorted(TRUE_STRINGS | FALSE_STRINGS))
    raise argparse.ArgumentTypeError(f"expected a boolean value ({expected}), got {value!r}")


def _allows_none(annotation: Any) -> bool:
    return type(None) in get_args(annotation)


def _without_none(annotation: Any) -> Any:
    all_args = get_args(annotation)
    if not all_args:
        return annotation
    args = tuple(arg for arg in all_args if arg is not type(None))
    if not args:
        return str
    if len(args) == 1:
        return args[0]
    return annotation


def _value_parser(name: str, annotation: Any):
    allow_none = _allows_none(annotation)
    value_type = _without_none(annotation) if allow_none else annotation
    origin = get_origin(value_type)
    if origin is not None:
        value_type = origin

    if value_type is bool:
        parser = _parse_bool
    elif value_type is int:
        parser = int
    elif value_type is float:
        parser = float
    elif value_type is str:
        parser = str
    else:
        raise TypeError(f"Unsupported TrainConfig field type for {name}: {annotation!r}")

    if not allow_none:
        return parser

    def parse_optional(value: str) -> Any:
        if str(value).strip().lower() in NONE_STRINGS:
            return None
        return parser(value)

    return parse_optional


def _option_strings(name: str, *, negative: bool = False) -> list[str]:
    option_names = [name, *FIELD_ALIASES.get(name, ())]
    flags: list[str] = []
    for option_name in option_names:
        cli_name = f"no_{option_name}" if negative else option_name
        flags.append(f"--{cli_name}")
        hyphen_name = cli_name.replace("_", "-")
        if hyphen_name != cli_name:
            flags.append(f"--{hyphen_name}")
    return flags


def _build_parser(defaults: dict[str, Any], config_path: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DeLaN + object-MLP WMDynamics model.")
    parser.add_argument("--config", type=str, default=config_path, help="Optional YAML file with TrainConfig values.")
    overrides = parser.add_argument_group("TrainConfig overrides")
    type_hints = get_type_hints(TrainConfig)
    for config_field in fields(TrainConfig):
        name = config_field.name
        annotation = type_hints[name]
        default = defaults[name]
        kwargs = {
            "dest": name,
            "default": default,
            "help": f"Override TrainConfig.{name} (default: %(default)s).",
        }
        if name in FIELD_CHOICES:
            kwargs["choices"] = FIELD_CHOICES[name]

        if _without_none(annotation) is bool:
            overrides.add_argument(
                *_option_strings(name),
                nargs="?",
                const=True,
                type=_parse_bool,
                metavar="{true,false}",
                **kwargs,
            )
            overrides.add_argument(
                *_option_strings(name, negative=True),
                dest=name,
                action="store_false",
                help=f"Set TrainConfig.{name}=False.",
            )
        else:
            overrides.add_argument(*_option_strings(name), type=_value_parser(name, annotation), **kwargs)
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    defaults, config_path = _defaults_from_config(argv)
    parser = _build_parser(defaults, config_path)
    values = vars(parser.parse_args(argv))
    values.pop("config", None)
    return TrainConfig(**values)


def _normalize_action_type(action_type: str) -> str:
    normalized = action_type.strip().lower()
    if normalized not in {"policy", "torque"}:
        raise ValueError(f"action_type must be 'policy' or 'torque', got {action_type!r}.")
    return normalized


def _model_name(cfg: TrainConfig) -> str:
    return "RWMEnsemble" if cfg.model_type == "rwm" else MODEL_NAME


def _apply_rwm_horizons(cfg: TrainConfig, rwm_cfg: RWMConfig | None) -> None:
    if rwm_cfg is None:
        return
    cfg.history_len = rwm_cfg.history_horizon
    cfg.rollout_horizon = rwm_cfg.forecast_horizon


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
        "action_type": cfg.action_type,
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


def _rwm_action_dim(cfg: TrainConfig) -> int:
    return cfg.action_dim if _normalize_action_type(cfg.action_type) == "policy" else cfg.torque_dim


def _rwm_architecture_config(cfg: RWMConfig) -> dict[str, Any]:
    return {
        "type": cfg.architecture.strip().lower(),
        "rnn_type": cfg.cell.strip().lower(),
        "rnn_num_layers": cfg.rnn_num_layers,
        "rnn_hidden_size": cfg.rnn_hidden_size,
        "state_mean_shape": list(cfg.state_mean_head),
        "state_logstd_shape": None if cfg.state_logstd_head is None else list(cfg.state_logstd_head),
    }


def build_model(
    cfg: TrainConfig,
    *,
    device: torch.device | str | None = None,
    rwm_cfg: RWMConfig | None = None,
) -> torch.nn.Module:
    if cfg.model_type == "rwm":
        rwm_cfg = rwm_cfg or load_rwm_config(cfg.rwm_config)
        torch_device = torch.device(device or "cpu")
        layout = object_mlp_dataset.RobotObjectWMStateLayout(
            robot_dof=cfg.robot_dof,
            action_dim=cfg.action_dim,
            torque_dim=cfg.torque_dim,
        )
        model = RWMEnsemble(
            state_dim=layout.state_dim,
            action_dim=_rwm_action_dim(cfg),
            device=str(torch_device),
            ensemble_size=rwm_cfg.ensemble_size,
            history_horizon=rwm_cfg.history_horizon,
            architecture_config=_rwm_architecture_config(rwm_cfg),
        )
        model.model_type = "rwm"
        model.action_type = _normalize_action_type(cfg.action_type)
        return model
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


def _rwm_sequence_batch(
    batch: dict[str, Any],
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_states = batch["history_states"].to(device)
    future_states = batch["future_states"].to(device)
    state_batch = torch.cat([history_states, future_states], dim=1)

    action_type = _normalize_action_type(cfg.action_type)
    if action_type == "policy":
        history_actions = batch["history_actions"].to(device)
        future_actions = batch["future_actions"].to(device)
        pad = history_actions.new_zeros(history_actions.shape[0], 1, history_actions.shape[-1])
        action_batch = torch.cat([pad, history_actions, future_actions], dim=1)
        action_values = torch.cat([history_actions, future_actions], dim=1)
    else:
        history_torques = batch["history_torques"].to(device)
        future_torques = batch["future_torques"].to(device)
        pad = history_torques.new_zeros(history_torques.shape[0], 1, history_torques.shape[-1])
        action_batch = torch.cat([pad, history_torques, future_torques[:, 1:]], dim=1)
        action_values = torch.cat([history_torques, future_torques[:, 1:]], dim=1)

    if action_batch.shape[1] != state_batch.shape[1]:
        raise ValueError(
            "RWM state/action sequence length mismatch: "
            f"states={state_batch.shape[1]} actions={action_batch.shape[1]}."
        )
    return state_batch, action_batch, action_values


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


def run_rwm_epoch(
    model: RWMEnsemble,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
    rwm_cfg: RWMConfig,
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

        state_batch, action_batch, action_values = _rwm_sequence_batch(batch, device, cfg)
        if hasattr(model, "reset"):
            model.reset()

        with torch.set_grad_enabled(is_train):
            state_loss, sequence_loss, bound_loss, kl_loss = model.compute_loss(
                state_batch,
                action_batch,
                bootstrap=rwm_cfg.bootstrap,
            )
            loss = (
                rwm_cfg.state_loss_weight * state_loss
                + rwm_cfg.sequence_loss_weight * sequence_loss
                + rwm_cfg.bound_loss_weight * bound_loss
                + rwm_cfg.kl_loss_weight * kl_loss
            )
            if not _is_finite(loss):
                _raise_nonfinite_step(
                    phase,
                    batch_idx,
                    {
                        "state_batch": state_batch,
                        "action_batch": action_batch,
                        "state_loss": state_loss,
                        "sequence_loss": sequence_loss,
                        "bound_loss": bound_loss,
                        "kl_loss": kl_loss,
                        "loss": loss,
                    },
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not _is_finite(grad_norm):
                _raise_nonfinite_step(phase, batch_idx, {"loss": loss, "grad_norm": grad_norm})
            optimizer.step()

        metrics = {
            "loss": loss.detach(),
            "rollout_loss": state_loss.detach(),
            "rwm_state_loss": state_loss.detach(),
            "rwm_sequence_loss": sequence_loss.detach(),
            "rwm_bound_loss": bound_loss.detach(),
            "rwm_kl_loss": kl_loss.detach(),
            "action_abs_mean": action_values.abs().mean().detach(),
        }
        averager.update(metrics)
        averager.step()

        if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
            print(f"  [{phase}] batch {batch_idx}/{num_batches} loss={loss.item():.6f}")

    if hasattr(model, "reset"):
        model.reset()
    return averager.means()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
    max_batches: int | None,
    rwm_cfg: RWMConfig | None = None,
) -> dict[str, float]:
    if cfg.model_type == "rwm":
        if rwm_cfg is None:
            raise ValueError("rwm_cfg is required when model_type='rwm'.")
        return run_rwm_epoch(model, loader, optimizer, device, cfg, rwm_cfg, max_batches)

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


def _print_run_header(
    cfg: TrainConfig,
    device: torch.device,
    output_dir: str,
    meta: dict[str, Any],
    *,
    model_name: str = MODEL_NAME,
    rwm_cfg: RWMConfig | None = None,
) -> None:
    print("Training config:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    if rwm_cfg is not None:
        print("RWM config:")
        for key, value in asdict(rwm_cfg).items():
            print(f"  {key}: {value}")
    print(f"Model: {model_name}")
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
    cfg.action_type = _normalize_action_type(cfg.action_type)
    rwm_cfg = load_rwm_config(cfg.rwm_config) if cfg.model_type == "rwm" else None
    _apply_rwm_horizons(cfg, rwm_cfg)
    model_name = _model_name(cfg)
    run_config = asdict(cfg)
    if rwm_cfg is not None:
        run_config["rwm_config_values"] = asdict(rwm_cfg)

    set_seed(cfg.seed)
    run = start_training_run(cfg, model_name, run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, layout, meta = make_dataloaders(cfg)
    model = build_model(cfg, device=device, rwm_cfg=rwm_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    _print_run_header(cfg, device, run.output_dir, meta, model_name=model_name, rwm_cfg=rwm_cfg)
    run.update_config({"architecture": model_name, "data_meta": meta})

    best_val = float("inf")
    max_train_batches = _max_batches(train_loader, cfg.train_batch_fraction)

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, max_train_batches, rwm_cfg)
            val_metrics = run_epoch(model, val_loader, None, device, cfg, None, rwm_cfg)
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
                    "architecture": model_name,
                    "config": asdict(cfg),
                    "rwm_config": asdict(rwm_cfg) if rwm_cfg is not None else None,
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
