"""Train the robot+object MCGDF world model.

Highlights vs. the robot-only MCGDF trainer:

* loads object state, per-episode object context, and per-episode residual
  baseline (Approach A of mismatch #2);
* passes ``object_context`` and ``residual_baseline`` to the model and loss;
* supervises object linear/angular acceleration (mismatch #7 dropped);
* regularizes the contact wrench magnitude;
* otherwise mirrors the robot-only trainer's structure for ``H, c, g``,
  damping/friction supervision, and the residual head.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    RobotObjectMCGDFRolloutDataset,
    RobotObjectMCGDFStateLayout,
    discover_hdf5_files,
    load_all_episode_refs,
    split_episode_refs,
)
from models import (
    MultiStepRobotObjectMCGDFWorldModel,
    auxiliary_regularization,
    damping_friction_supervised_loss,
    info_nce_soft,
    kl_divergence_standard_normal,
    residual_supervised_loss,
    supervised_object_dynamics_loss,
    supervised_robot_dynamics_loss,
    weighted_rollout_mse,
)


@dataclass
class TrainConfig:
    dataset_dir: str
    dataset_file: str | None
    history_len: int
    rollout_horizon: int
    robot_dof: int
    action_dim: int
    torque_dim: int
    object_context_dim: int
    torque_key: str
    friction_key: str
    hidden_dim: int
    residual_hidden_dim: int
    residual_depth: int
    contact_hidden_dim: int
    contact_depth: int
    tool_z_offset: float
    dt: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    # Rollout weights.
    q_weight: float
    dq_weight: float
    object_pos_weight: float
    object_quat_weight: float
    object_lin_vel_weight: float
    object_ang_vel_weight: float
    # Supervised robot dynamics weights.
    lambda_mass_matrix: float
    lambda_inertial: float
    lambda_coriolis_gt: float
    lambda_gravity: float
    lambda_qdd: float
    lambda_inverse_dynamics: float
    # Supervised object dynamics weights.
    lambda_object_lin_acc: float
    lambda_object_ang_acc: float
    # Regularization weights.
    lambda_coriolis: float
    lambda_residual: float
    lambda_contact: float
    lambda_residual_supervised: float
    lambda_damping: float
    lambda_friction: float
    lambda_weight_l2: float
    # Joint params behaviour.
    learn_damping_friction: bool
    init_damping: float
    init_friction: float
    omit_damping: bool
    friction_eps: float
    # Context encoder / FiLM / contrastive (optional, off by default).
    use_context_encoder: bool
    latent_dim: int
    context_encoder_hidden_dim: int
    context_encoder_depth: int
    # Optional FiLM modulation of the DeLaN heads (H, c, g).  Off by
    # default so the existing setting is preserved.
    delan_use_film: bool
    delan_film_depth: int
    beta_kl: float
    lambda_nce: float
    nce_temperature: float
    nce_context_similarity_sigma: float
    nce_negative_weight: float
    lambda_ctx_regression: float
    # Training scaffolding.
    train_split: float
    seed: int
    num_workers: int
    output_dir: str
    run_name: str | None
    log_every_batches: int
    train_batch_fraction: float
    train_subsample_fraction: float
    val_subsample_fraction: float
    # Filtering.
    filter_pre_contact: bool
    object_displacement_threshold: float
    object_velocity_threshold: float
    contact_consecutive_steps: int
    contact_settle_steps: int
    baseline_settle_steps: int
    subtract_env_origin: bool


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the robot+object MCGDF world model.")
    parser.add_argument("--dataset_dir", type=str, default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/datasets")
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--rollout_horizon", type=int, default=3)
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--action_dim", type=int, default=8)
    parser.add_argument("--torque_dim", type=int, default=9)
    parser.add_argument("--object_context_dim", type=int, default=13)
    parser.add_argument("--torque_key", type=str, default="applied_torque",
                        choices=["applied_torque", "computed_torque"])
    parser.add_argument("--friction_key", type=str, default="joint_dynamic_friction_coeff",
                        choices=["joint_dynamic_friction_coeff", "joint_friction_coeff"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--residual_hidden_dim", type=int, default=128)
    parser.add_argument("--residual_depth", type=int, default=2)
    parser.add_argument("--contact_hidden_dim", type=int, default=128)
    parser.add_argument("--contact_depth", type=int, default=3)
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # Rollout weights.
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--dq_weight", type=float, default=0.2)
    parser.add_argument("--object_pos_weight", type=float, default=5.0)
    parser.add_argument("--object_quat_weight", type=float, default=0.5)
    parser.add_argument("--object_lin_vel_weight", type=float, default=1.0)
    parser.add_argument("--object_ang_vel_weight", type=float, default=0.5)

    # Robot dynamics supervision.
    parser.add_argument("--lambda_mass_matrix", type=float, default=0.00)
    parser.add_argument("--lambda_inertial", type=float, default=0.00)
    parser.add_argument("--lambda_coriolis_gt", type=float, default=0.00)
    parser.add_argument("--lambda_gravity", type=float, default=0.00)
    parser.add_argument("--lambda_qdd", type=float, default=0.0)
    parser.add_argument("--lambda_inverse_dynamics", type=float, default=0.0)

    # Object dynamics supervision.
    parser.add_argument("--lambda_object_lin_acc", type=float, default=0.01)
    parser.add_argument("--lambda_object_ang_acc", type=float, default=0.001)

    # Regularization.
    parser.add_argument("--lambda_coriolis", type=float, default=0.0)
    parser.add_argument("--lambda_residual", type=float, default=1e-4)
    parser.add_argument("--lambda_contact", type=float, default=1e-5)
    parser.add_argument("--lambda_residual_supervised", type=float, default=0.01)
    parser.add_argument("--lambda_damping", type=float, default=0.0)
    parser.add_argument("--lambda_friction", type=float, default=0.0)
    parser.add_argument("--lambda_weight_l2", type=float, default=1e-6)

    # Joint params.
    parser.add_argument("--learn_damping_friction", action="store_true", default=False)
    parser.add_argument("--init_damping", type=float, default=0.0)
    parser.add_argument("--init_friction", type=float, default=0.0)
    parser.add_argument(
        "--omit_damping", action="store_true", default=True,
        help="Drop the explicit d*qdot term from the robot ODE; recommended for Franka Lift.",
    )
    parser.add_argument("--friction_eps", type=float, default=1e-3)

    # ----- Context encoder / FiLM / contrastive (Ver1-style, optional).
    parser.add_argument(
        "--use_context_encoder", action="store_true", default=True,
        help="Enable the variational ContextEncoder + FiLM modulation of the "
             "residual and contact heads, plus KL and InfoNCE losses.",
    )
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--context_encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--context_encoder_depth", type=int, default=2)
    parser.add_argument("--beta_kl", type=float, default=1e-3,
                        help="Weight of KL(q(z|h) || N(0, I)); Ver1 default.")
    parser.add_argument("--lambda_nce", type=float, default=0.1,
                        help="Weight of the soft-weighted InfoNCE on z.")
    parser.add_argument("--nce_temperature", type=float, default=0.1)
    parser.add_argument("--nce_context_similarity_sigma", type=float, default=1.0,
                        help="Gaussian kernel sigma over standardized context-target distance.")
    parser.add_argument("--nce_negative_weight", type=float, default=1.0)
    parser.add_argument("--lambda_ctx_regression", type=float, default=0.0,
                        help="Weight of the auxiliary context-regression head loss.")
    parser.add_argument(
        "--delan_use_film", action="store_true", default=False,
        help="Modulate the DeLaN heads (H, c, g) with the latent z via FiLM. "
             "Off by default; requires --use_context_encoder.",
    )
    parser.add_argument(
        "--delan_film_depth", type=int, default=2,
        help="Number of FiLM blocks per DeLaN head when --delan_use_film is set.",
    )

    # Scaffolding.
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs_mcgdf")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_every_batches", type=int, default=200)
    parser.add_argument(
        "--train_batch_fraction", type=float, default=0.1,
        help="Fraction of batches consumed per epoch (uses all DATA but stops "
             "early in each epoch when < 1.0). Independent of --train_subsample_fraction.",
    )
    parser.add_argument(
        "--train_subsample_fraction", type=float, default=0.2,
        help="Randomly subsample this fraction of TRAIN EPISODES at load time. "
             "Reduces memory and per-epoch wall-clock proportionally. "
             "Defaults to 1.0 (use all train episodes).",
    )
    parser.add_argument(
        "--val_subsample_fraction", type=float, default=1.0,
        help="Same as --train_subsample_fraction but for the validation split. "
             "Defaults to 1.0 (use all val episodes).",
    )
    parser.add_argument(
        "--filter_pre_contact", action="store_true", default=False,
        help="If set, drop rollout windows whose horizon crosses contact onset.",
    )
    parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
    parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
    parser.add_argument("--contact_consecutive_steps", type=int, default=3)
    parser.add_argument("--contact_settle_steps", type=int, default=5)
    parser.add_argument("--baseline_settle_steps", type=int, default=5)
    parser.add_argument(
        "--no_subtract_env_origin", dest="subtract_env_origin", action="store_false",
        default=True,
        help="Disable subtracting env_origin from object position (use raw world frame).",
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_hdf5_paths(cfg: TrainConfig) -> list[str]:
    if cfg.dataset_file:
        path = os.path.abspath(cfg.dataset_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return [path]
    dataset_dir = os.path.abspath(cfg.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    paths = discover_hdf5_files(dataset_dir)
    if not paths:
        raise FileNotFoundError(f"No *.hdf5 files found in {dataset_dir}")
    return paths


def _subsample_refs(refs: list, fraction: float, seed: int, kind: str) -> list:
    """Randomly take ``fraction`` of ``refs``.  Returns the original list
    when ``fraction >= 1.0`` (the default).  Always keeps at least one ref.
    """
    if fraction >= 1.0:
        return refs
    if fraction <= 0.0:
        raise ValueError(f"--{kind}_subsample_fraction must be > 0; got {fraction}")
    n = max(1, int(round(len(refs) * fraction)))
    if n >= len(refs):
        return refs
    rng = random.Random(seed)
    return rng.sample(refs, n)


def make_dataloaders(cfg: TrainConfig):
    hdf5_paths = resolve_hdf5_paths(cfg)
    episode_refs = load_all_episode_refs(hdf5_paths)
    train_refs, val_refs = split_episode_refs(episode_refs, cfg.train_split, cfg.seed)
    n_train_before, n_val_before = len(train_refs), len(val_refs)
    # Subsample after the train/val split so the split is reproducible across
    # different subsample fractions sharing the same --seed.
    train_refs = _subsample_refs(train_refs, cfg.train_subsample_fraction, cfg.seed + 1, "train")
    val_refs = _subsample_refs(val_refs, cfg.val_subsample_fraction, cfg.seed + 2, "val")
    layout = RobotObjectMCGDFStateLayout(
        robot_dof=cfg.robot_dof, action_dim=cfg.action_dim, torque_dim=cfg.torque_dim,
    )
    kwargs = dict(
        history_len=cfg.history_len,
        rollout_horizon=cfg.rollout_horizon,
        dt=cfg.dt,
        torque_key=cfg.torque_key,
        friction_key=cfg.friction_key,
        friction_eps=cfg.friction_eps,
        require_joint_params=True,
        filter_pre_contact=cfg.filter_pre_contact,
        object_displacement_threshold=cfg.object_displacement_threshold,
        object_velocity_threshold=cfg.object_velocity_threshold,
        contact_consecutive_steps=cfg.contact_consecutive_steps,
        contact_settle_steps=cfg.contact_settle_steps,
        baseline_settle_steps=cfg.baseline_settle_steps,
        subtract_env_origin=cfg.subtract_env_origin,
        layout=layout,
    )
    train_dataset = RobotObjectMCGDFRolloutDataset(episode_refs=train_refs, **kwargs)
    val_dataset = RobotObjectMCGDFRolloutDataset(episode_refs=val_refs, **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
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
        "friction_key": cfg.friction_key,
        "context_target_dim_train": train_dataset.context_target_dim,
        "context_target_dim_val": val_dataset.context_target_dim,
        "train_episodes_with_contact": train_dataset.episodes_with_contact,
        "val_episodes_with_contact": val_dataset.episodes_with_contact,
        "train_skipped_short_episodes": train_dataset.skipped_short_episodes,
        "val_skipped_short_episodes": val_dataset.skipped_short_episodes,
        "train_skipped_missing_torque_episodes": train_dataset.skipped_missing_torque_episodes,
        "val_skipped_missing_torque_episodes": val_dataset.skipped_missing_torque_episodes,
        "train_skipped_missing_robot_dynamics_episodes": train_dataset.skipped_missing_robot_dynamics_episodes,
        "val_skipped_missing_robot_dynamics_episodes": val_dataset.skipped_missing_robot_dynamics_episodes,
        "train_skipped_missing_object_dynamics_episodes": train_dataset.skipped_missing_object_dynamics_episodes,
        "val_skipped_missing_object_dynamics_episodes": val_dataset.skipped_missing_object_dynamics_episodes,
        "train_skipped_missing_joint_params_episodes": train_dataset.skipped_missing_joint_params_episodes,
        "val_skipped_missing_joint_params_episodes": val_dataset.skipped_missing_joint_params_episodes,
        "train_skipped_contact_filtered_windows": train_dataset.skipped_contact_filtered_windows,
        "val_skipped_contact_filtered_windows": val_dataset.skipped_contact_filtered_windows,
    }
    return train_loader, val_loader, layout, meta


def _max_batches(loader, fraction: float):
    if fraction >= 1.0:
        return None
    if fraction <= 0.0:
        raise ValueError("--train_batch_fraction must be > 0")
    return max(1, int(round(len(loader) * fraction)))


def move_dict_to_device(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in d.items()}


def network_weight_l2(model: torch.nn.Module, lambda_weight_l2: float):
    first = next(model.parameters())
    raw_l2 = torch.zeros((), device=first.device, dtype=first.dtype)
    if lambda_weight_l2 <= 0.0:
        return raw_l2, raw_l2.detach()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim > 1 and "weight" in name:
            raw_l2 = raw_l2 + param.pow(2).sum()
    return lambda_weight_l2 * raw_l2, raw_l2.detach()


def run_epoch(model, loader, optimizer, device, cfg: TrainConfig, max_batches):
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {
        "loss": 0.0, "rollout_loss": 0.0,
        "robot_dyn_loss": 0.0, "object_dyn_loss": 0.0,
        "dfr_sup_loss": 0.0, "res_sup_loss": 0.0,
        "reg_loss": 0.0, "weight_l2_loss": 0.0, "weight_l2_norm_sq": 0.0,
        "q_mse": 0.0, "dq_mse": 0.0,
        "object_pos_mse": 0.0, "object_quat_mse": 0.0,
        "object_lin_vel_mse": 0.0, "object_ang_vel_mse": 0.0,
        "one_step_q_mse": 0.0, "one_step_object_pos_mse": 0.0,
        "final_step_q_mse": 0.0, "final_step_object_pos_mse": 0.0,
        "coriolis_reg": 0.0, "residual_reg": 0.0, "contact_reg": 0.0,
        "torque_abs_mean": 0.0, "residual_abs_mean": 0.0,
        "contact_wrench_abs_mean": 0.0,
        "damping_abs_mean": 0.0, "friction_abs_mean": 0.0,
        "mass_matrix_mse": 0.0, "inertial_mse": 0.0,
        "coriolis_gt_mse": 0.0, "gravity_mse": 0.0,
        "qdd_mse": 0.0, "inverse_dynamics_mse": 0.0,
        "damping_mse": 0.0, "friction_mse": 0.0,
        "object_lin_acc_mse": 0.0, "object_ang_acc_mse": 0.0,
        "residual_sup_mse": 0.0,
        # Context encoder metrics (always present so the schema stays
        # uniform; remain at 0 when --use_context_encoder is off).
        "kl_loss": 0.0, "nce_loss": 0.0, "ctx_regression_loss": 0.0,
        "z_norm_mean": 0.0,
    }
    total_batches = 0
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    phase = "train" if is_train else "val"

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        history_states = batch["history_states"].to(device)
        future_torques = batch["future_torques"].to(device)
        future_states = batch["future_states"].to(device)
        object_context = batch["object_context"].to(device)
        residual_baseline = batch["residual_baseline"].to(device)
        future_robot_dynamics = move_dict_to_device(batch["future_robot_dynamics"], device)
        future_object_dynamics = move_dict_to_device(batch["future_object_dynamics"], device)
        joint_params = {
            "damping": batch["joint_params"]["damping"].to(device),
            "friction": batch["joint_params"]["friction"].to(device),
        }
        # New fields used only when --use_context_encoder.  Older HDF5s may
        # have an empty context_target; that's handled by info_nce_soft.
        history_torques = batch["history_torques"].to(device) if cfg.use_context_encoder else None
        context_target = batch["context_target"].to(device) if cfg.use_context_encoder else None

        with torch.set_grad_enabled(is_train):
            if cfg.use_context_encoder:
                pred_future, aux_list, context_outputs = model(
                    history_states, future_torques, joint_params, object_context,
                    history_torques=history_torques, return_aux=True, return_context=True,
                )
            else:
                pred_future, aux_list = model(
                    history_states, future_torques, joint_params, object_context, return_aux=True,
                )
                context_outputs = None
            rollout_loss, parts = weighted_rollout_mse(
                pred=pred_future, target=future_states, robot_dof=cfg.robot_dof,
                q_weight=cfg.q_weight, dq_weight=cfg.dq_weight,
                object_pos_weight=cfg.object_pos_weight,
                object_quat_weight=cfg.object_quat_weight,
                object_lin_vel_weight=cfg.object_lin_vel_weight,
                object_ang_vel_weight=cfg.object_ang_vel_weight,
            )
            robot_dyn_loss, robot_dyn_metrics = supervised_robot_dynamics_loss(
                aux_list=aux_list, future_robot_dynamics=future_robot_dynamics,
                lambda_mass_matrix=cfg.lambda_mass_matrix,
                lambda_inertial=cfg.lambda_inertial,
                lambda_coriolis_gt=cfg.lambda_coriolis_gt,
                lambda_gravity=cfg.lambda_gravity,
                lambda_qdd=cfg.lambda_qdd,
                lambda_inverse_dynamics=cfg.lambda_inverse_dynamics,
            )
            object_dyn_loss, object_dyn_metrics = supervised_object_dynamics_loss(
                aux_list=aux_list, future_object_dynamics=future_object_dynamics,
                lambda_object_lin_acc=cfg.lambda_object_lin_acc,
                lambda_object_ang_acc=cfg.lambda_object_ang_acc,
            )
            dfr_sup_loss, dfr_sup_metrics = damping_friction_supervised_loss(
                aux_list=aux_list, joint_params=joint_params,
                lambda_damping=cfg.lambda_damping, lambda_friction=cfg.lambda_friction,
            )
            res_sup_loss, res_sup_metrics = residual_supervised_loss(
                aux_list=aux_list, future_robot_dynamics=future_robot_dynamics,
                future_torques=future_torques, future_states=future_states,
                joint_params=joint_params, residual_baseline=residual_baseline,
                omit_damping=cfg.omit_damping, friction_eps=cfg.friction_eps,
                robot_dof=cfg.robot_dof,
                lambda_residual_supervised=cfg.lambda_residual_supervised,
            )
            reg_loss, regs = auxiliary_regularization(
                aux_list,
                lambda_coriolis=cfg.lambda_coriolis,
                lambda_residual=cfg.lambda_residual,
                lambda_contact=cfg.lambda_contact,
            )
            weight_l2_loss, weight_l2_norm_sq = network_weight_l2(model, cfg.lambda_weight_l2)

            # Context encoder losses (zero when --use_context_encoder is off).
            zero = torch.zeros((), device=device, dtype=pred_future.dtype)
            kl_loss = zero
            nce_loss = zero
            ctx_reg_loss = zero
            z_norm_mean = zero
            if context_outputs is not None and context_outputs["z"] is not None:
                mu = context_outputs["mu"]
                logvar = context_outputs["logvar"]
                z = context_outputs["z"]
                hat_xi = context_outputs["hat_xi"]
                if mu is not None and logvar is not None:
                    kl_loss = kl_divergence_standard_normal(mu, logvar)
                if context_target is not None and context_target.shape[-1] > 0:
                    nce_loss = info_nce_soft(
                        embedding=z,
                        context_value=context_target,
                        temperature=cfg.nce_temperature,
                        context_similarity_sigma=cfg.nce_context_similarity_sigma,
                        nce_negative_weight=cfg.nce_negative_weight,
                    )
                if (
                    hat_xi is not None
                    and context_target is not None
                    and context_target.shape[-1] == hat_xi.shape[-1]
                    and context_target.shape[-1] > 0
                ):
                    ctx_reg_loss = torch.nn.functional.mse_loss(hat_xi, context_target)
                z_norm_mean = z.norm(dim=-1).mean().detach()

            loss = (
                rollout_loss
                + robot_dyn_loss + object_dyn_loss
                + dfr_sup_loss + res_sup_loss
                + reg_loss + weight_l2_loss
                + cfg.beta_kl * kl_loss
                + cfg.lambda_nce * nce_loss
                + cfg.lambda_ctx_regression * ctx_reg_loss
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        robot_dof = cfg.robot_dof
        offset = 2 * robot_dof
        one_step_q = torch.nn.functional.mse_loss(
            pred_future[:, 0, :robot_dof], future_states[:, 0, :robot_dof]
        )
        one_step_op = torch.nn.functional.mse_loss(
            pred_future[:, 0, offset : offset + 3], future_states[:, 0, offset : offset + 3]
        )
        final_q = torch.nn.functional.mse_loss(
            pred_future[:, -1, :robot_dof], future_states[:, -1, :robot_dof]
        )
        final_op = torch.nn.functional.mse_loss(
            pred_future[:, -1, offset : offset + 3], future_states[:, -1, offset : offset + 3]
        )

        residual_stack = torch.stack([aux["residual_torque"] for aux in aux_list], dim=1)
        contact_stack = torch.stack([aux["contact_wrench"] for aux in aux_list], dim=1)
        damping_stack = torch.stack([aux["damping_coeff"] for aux in aux_list], dim=1)
        friction_stack = torch.stack([aux["friction_coeff"] for aux in aux_list], dim=1)

        totals["loss"] += float(loss.item())
        totals["rollout_loss"] += float(rollout_loss.item())
        totals["robot_dyn_loss"] += float(robot_dyn_loss.item())
        totals["object_dyn_loss"] += float(object_dyn_loss.item())
        totals["dfr_sup_loss"] += float(dfr_sup_loss.item())
        totals["res_sup_loss"] += float(res_sup_loss.item())
        totals["reg_loss"] += float(reg_loss.item())
        totals["weight_l2_loss"] += float(weight_l2_loss.item())
        totals["weight_l2_norm_sq"] += float(weight_l2_norm_sq.item())
        for key, value in parts.items():
            totals[key] += float(value.item())
        totals["one_step_q_mse"] += float(one_step_q.item())
        totals["one_step_object_pos_mse"] += float(one_step_op.item())
        totals["final_step_q_mse"] += float(final_q.item())
        totals["final_step_object_pos_mse"] += float(final_op.item())
        totals["coriolis_reg"] += float(regs["coriolis_reg"].item())
        totals["residual_reg"] += float(regs["residual_reg"].item())
        totals["contact_reg"] += float(regs["contact_reg"].item())
        totals["torque_abs_mean"] += float(future_torques.abs().mean().item())
        totals["residual_abs_mean"] += float(residual_stack.abs().mean().item())
        totals["contact_wrench_abs_mean"] += float(contact_stack.abs().mean().item())
        totals["damping_abs_mean"] += float(damping_stack.abs().mean().item())
        totals["friction_abs_mean"] += float(friction_stack.abs().mean().item())
        for key, value in robot_dyn_metrics.items():
            totals[key] += float(value.item())
        for key, value in object_dyn_metrics.items():
            totals[key] += float(value.item())
        for key, value in dfr_sup_metrics.items():
            totals[key] += float(value.item())
        for key, value in res_sup_metrics.items():
            totals[key] += float(value.item())
        totals["kl_loss"] += float(kl_loss.item())
        totals["nce_loss"] += float(nce_loss.item())
        totals["ctx_regression_loss"] += float(ctx_reg_loss.item())
        totals["z_norm_mean"] += float(z_norm_mean.item()) if torch.is_tensor(z_norm_mean) else float(z_norm_mean)
        total_batches += 1

        if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
            print(f"  [{phase}] batch {batch_idx}/{num_batches} loss={loss.item():.6f}")

    return {key: value / max(1, total_batches) for key, value in totals.items()}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    run_name = cfg.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=False)
    tb_dir = os.path.join(run_output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, layout, meta = make_dataloaders(cfg)
    # Each history step contributes ``state_dim + torque_dim`` features to the
    # encoder.  Used only when --use_context_encoder is on.
    history_step_dim = layout.state_dim + cfg.torque_dim if cfg.use_context_encoder else 0
    context_target_dim = meta.get("context_target_dim_train", 0) if cfg.use_context_encoder else 0
    model = MultiStepRobotObjectMCGDFWorldModel(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        object_context_dim=cfg.object_context_dim,
        hidden_dim=cfg.hidden_dim,
        residual_hidden_dim=cfg.residual_hidden_dim,
        residual_depth=cfg.residual_depth,
        contact_hidden_dim=cfg.contact_hidden_dim,
        contact_depth=cfg.contact_depth,
        dt=cfg.dt,
        friction_eps=cfg.friction_eps,
        learn_damping_friction=cfg.learn_damping_friction,
        init_damping=cfg.init_damping,
        init_friction=cfg.init_friction,
        omit_damping=cfg.omit_damping,
        tool_z_offset=cfg.tool_z_offset,
        use_context_encoder=cfg.use_context_encoder,
        latent_dim=cfg.latent_dim,
        context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
        context_encoder_depth=cfg.context_encoder_depth,
        history_step_dim=history_step_dim,
        history_len=cfg.history_len,
        context_target_dim=context_target_dim,
        delan_use_film=cfg.delan_use_film,
        delan_film_depth=cfg.delan_film_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("Training config:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    print(f"Device: {device}")
    print(f"Output dir: {run_output_dir}")
    print("Dataset meta:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    best_val = float("inf")
    best_path = os.path.join(run_output_dir, "best.pt")
    last_path = os.path.join(run_output_dir, "last.pt")
    max_train_batches = _max_batches(train_loader, cfg.train_batch_fraction)
    max_val_batches = _max_batches(val_loader, cfg.train_batch_fraction)

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, max_train_batches)
        val_metrics = run_epoch(model, val_loader, None, device, cfg, max_val_batches)
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_rollout={val_metrics['rollout_loss']:.6f} "
            f"val_obj_dyn={val_metrics['object_dyn_loss']:.6f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(cfg),
            "layout": layout.to_dict(),
            "data_meta": meta,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, last_path)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, best_path)

    writer.close()
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"TensorBoard log dir: {tb_dir}")


if __name__ == "__main__":
    main()
