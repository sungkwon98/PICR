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
    RobotObjectGTDynamicsRolloutDataset,
    RobotObjectGTDynamicsStateLayout,
    discover_hdf5_files,
    load_all_episode_refs,
    split_episode_refs,
)
from models import (
    MultiStepRobotObjectGTDynamicsWorldModel,
    auxiliary_regularization,
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
    torque_key: str
    hidden_dim: int
    dt: float
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
    lambda_mass_matrix: float
    lambda_inertial: float
    lambda_coriolis_gt: float
    lambda_gravity: float
    lambda_qdd: float
    lambda_inverse_dynamics: float
    lambda_residual_torque: float
    lambda_object_lin_acc: float
    lambda_object_ang_acc: float
    lambda_object_external_force: float
    lambda_coriolis: float
    lambda_residual_l2: float
    lambda_weight_l2: float
    train_split: float
    seed: int
    num_workers: int
    output_dir: str
    run_name: str | None
    log_every_batches: int
    train_batch_fraction: float
    filter_pre_contact: bool
    object_displacement_threshold: float
    object_velocity_threshold: float
    contact_consecutive_steps: int
    contact_settle_steps: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train hybrid robot-object DeLaN GT-dynamics world model.")
    parser.add_argument("--dataset_dir", type=str, default="../../../../reinforcement_learning/skrl/datasets")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../../../../reinforcement_learning/skrl/datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5",
    )
    parser.add_argument("--history_len", type=int, default=5, help="K+1 states in history.")
    parser.add_argument("--rollout_horizon", type=int, default=3)
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--action_dim", type=int, default=8)
    parser.add_argument("--torque_dim", type=int, default=9)
    parser.add_argument("--torque_key", type=str, default="applied_torque", choices=["applied_torque", "computed_torque"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--dq_weight", type=float, default=0.2)
    parser.add_argument("--object_pos_weight", type=float, default=5.0)
    parser.add_argument("--object_quat_weight", type=float, default=0.5)
    parser.add_argument("--object_lin_vel_weight", type=float, default=1.0)
    parser.add_argument("--object_ang_vel_weight", type=float, default=0.5)
    parser.add_argument("--lambda_mass_matrix", type=float, default=0.01)
    parser.add_argument("--lambda_inertial", type=float, default=0.01)
    parser.add_argument("--lambda_coriolis_gt", type=float, default=0.01)
    parser.add_argument("--lambda_gravity", type=float, default=0.01)
    parser.add_argument("--lambda_qdd", type=float, default=0.0)
    parser.add_argument("--lambda_inverse_dynamics", type=float, default=0.0)
    parser.add_argument("--lambda_residual_torque", type=float, default=0.01)
    parser.add_argument("--lambda_object_lin_acc", type=float, default=0.01)
    parser.add_argument("--lambda_object_ang_acc", type=float, default=0.001)
    parser.add_argument("--lambda_object_external_force", type=float, default=0.01)
    parser.add_argument("--lambda_coriolis", type=float, default=0.0)
    parser.add_argument("--lambda_residual_l2", type=float, default=1e-5)
    parser.add_argument("--lambda_weight_l2", type=float, default=1e-6)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs_robot_object_gt_dynamics")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_every_batches", type=int, default=200)
    parser.add_argument("--train_batch_fraction", type=float, default=0.1)
    parser.add_argument(
        "--filter_pre_contact",
        action="store_true",
        default=False,
        help="Only keep windows whose full rollout ends before detected object motion/contact.",
    )
    parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
    parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
    parser.add_argument("--contact_consecutive_steps", type=int, default=3)
    parser.add_argument("--contact_settle_steps", type=int, default=5)
    return TrainConfig(**vars(parser.parse_args()))


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


def make_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, RobotObjectGTDynamicsStateLayout, dict[str, object]]:
    hdf5_paths = resolve_hdf5_paths(cfg)
    episode_refs = load_all_episode_refs(hdf5_paths)
    train_refs, val_refs = split_episode_refs(episode_refs, cfg.train_split, cfg.seed)
    layout = RobotObjectGTDynamicsStateLayout(
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
    )
    kwargs = dict(
        history_len=cfg.history_len,
        rollout_horizon=cfg.rollout_horizon,
        dt=cfg.dt,
        torque_key=cfg.torque_key,
        filter_pre_contact=cfg.filter_pre_contact,
        object_displacement_threshold=cfg.object_displacement_threshold,
        object_velocity_threshold=cfg.object_velocity_threshold,
        contact_consecutive_steps=cfg.contact_consecutive_steps,
        contact_settle_steps=cfg.contact_settle_steps,
        layout=layout,
    )
    train_dataset = RobotObjectGTDynamicsRolloutDataset(episode_refs=train_refs, **kwargs)
    val_dataset = RobotObjectGTDynamicsRolloutDataset(episode_refs=val_refs, **kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    meta = {
        "hdf5_paths": hdf5_paths,
        "n_episodes_total": len(episode_refs),
        "n_train_episodes": len(train_refs),
        "n_val_episodes": len(val_refs),
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "torque_key": cfg.torque_key,
        "train_skipped_short_episodes": train_dataset.skipped_short_episodes,
        "val_skipped_short_episodes": val_dataset.skipped_short_episodes,
        "train_skipped_missing_torque_episodes": train_dataset.skipped_missing_torque_episodes,
        "val_skipped_missing_torque_episodes": val_dataset.skipped_missing_torque_episodes,
        "train_skipped_missing_robot_dynamics_episodes": train_dataset.skipped_missing_robot_dynamics_episodes,
        "val_skipped_missing_robot_dynamics_episodes": val_dataset.skipped_missing_robot_dynamics_episodes,
        "train_skipped_missing_object_dynamics_episodes": train_dataset.skipped_missing_object_dynamics_episodes,
        "val_skipped_missing_object_dynamics_episodes": val_dataset.skipped_missing_object_dynamics_episodes,
        "train_skipped_contact_filtered_windows": train_dataset.skipped_contact_filtered_windows,
        "val_skipped_contact_filtered_windows": val_dataset.skipped_contact_filtered_windows,
    }
    return train_loader, val_loader, layout, meta


def _max_batches(loader: DataLoader, fraction: float) -> int | None:
    if fraction >= 1.0:
        return None
    if fraction <= 0.0:
        raise ValueError("--train_batch_fraction must be > 0")
    return max(1, int(round(len(loader) * fraction)))


def move_dict_to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def network_weight_l2(model: torch.nn.Module, lambda_weight_l2: float) -> tuple[torch.Tensor, torch.Tensor]:
    first_param = next(model.parameters())
    raw_l2 = torch.zeros((), device=first_param.device, dtype=first_param.dtype)
    if lambda_weight_l2 <= 0.0:
        return raw_l2, raw_l2.detach()
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim > 1 and "weight" in name:
            raw_l2 = raw_l2 + param.pow(2).sum()
    return lambda_weight_l2 * raw_l2, raw_l2.detach()


def _initial_totals() -> dict[str, float]:
    keys = [
        "loss",
        "rollout_loss",
        "robot_dyn_loss",
        "object_dyn_loss",
        "reg_loss",
        "weight_l2_loss",
        "weight_l2_norm_sq",
        "q_mse",
        "dq_mse",
        "object_pos_mse",
        "object_quat_mse",
        "object_lin_vel_mse",
        "object_ang_vel_mse",
        "one_step_q_mse",
        "one_step_object_pos_mse",
        "final_step_q_mse",
        "final_step_object_pos_mse",
        "coriolis_reg",
        "residual_torque_reg",
        "torque_abs_mean",
        "residual_torque_abs_mean",
        "mass_matrix_mse",
        "inertial_mse",
        "coriolis_gt_mse",
        "gravity_mse",
        "qdd_mse",
        "inverse_dynamics_mse",
        "residual_torque_mse",
        "object_lin_acc_mse",
        "object_ang_acc_mse",
        "object_external_force_mse",
    ]
    return {key: 0.0 for key in keys}


def run_epoch(
    model: MultiStepRobotObjectGTDynamicsWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = _initial_totals()
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
        future_robot_dynamics = move_dict_to_device(batch["future_robot_dynamics"], device)
        future_object_dynamics = move_dict_to_device(batch["future_object_dynamics"], device)

        with torch.set_grad_enabled(is_train):
            pred_future, aux_list = model(history_states, future_torques, object_context, return_aux=True)
            rollout_loss, rollout_parts = weighted_rollout_mse(
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
            robot_dyn_loss, robot_dyn_metrics = supervised_robot_dynamics_loss(
                aux_list=aux_list,
                future_robot_dynamics=future_robot_dynamics,
                future_torques=future_torques,
                lambda_mass_matrix=cfg.lambda_mass_matrix,
                lambda_inertial=cfg.lambda_inertial,
                lambda_coriolis_gt=cfg.lambda_coriolis_gt,
                lambda_gravity=cfg.lambda_gravity,
                lambda_qdd=cfg.lambda_qdd,
                lambda_inverse_dynamics=cfg.lambda_inverse_dynamics,
                lambda_residual_torque=cfg.lambda_residual_torque,
            )
            object_dyn_loss, object_dyn_metrics = supervised_object_dynamics_loss(
                aux_list=aux_list,
                future_object_dynamics=future_object_dynamics,
                lambda_object_lin_acc=cfg.lambda_object_lin_acc,
                lambda_object_ang_acc=cfg.lambda_object_ang_acc,
                lambda_object_external_force=cfg.lambda_object_external_force,
            )
            reg_loss, regs = auxiliary_regularization(
                aux_list,
                lambda_coriolis=cfg.lambda_coriolis,
                lambda_residual_l2=cfg.lambda_residual_l2,
            )
            weight_l2_loss, weight_l2_norm_sq = network_weight_l2(model, cfg.lambda_weight_l2)
            loss = rollout_loss + robot_dyn_loss + object_dyn_loss + reg_loss + weight_l2_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        robot_dof = cfg.robot_dof
        robot_state_dim = 2 * robot_dof
        one_step_q = torch.nn.functional.mse_loss(pred_future[:, 0, :robot_dof], future_states[:, 0, :robot_dof])
        one_step_object_pos = torch.nn.functional.mse_loss(
            pred_future[:, 0, robot_state_dim : robot_state_dim + 3],
            future_states[:, 0, robot_state_dim : robot_state_dim + 3],
        )
        final_q = torch.nn.functional.mse_loss(pred_future[:, -1, :robot_dof], future_states[:, -1, :robot_dof])
        final_object_pos = torch.nn.functional.mse_loss(
            pred_future[:, -1, robot_state_dim : robot_state_dim + 3],
            future_states[:, -1, robot_state_dim : robot_state_dim + 3],
        )

        totals["loss"] += float(loss.item())
        totals["rollout_loss"] += float(rollout_loss.item())
        totals["robot_dyn_loss"] += float(robot_dyn_loss.item())
        totals["object_dyn_loss"] += float(object_dyn_loss.item())
        totals["reg_loss"] += float(reg_loss.item())
        totals["weight_l2_loss"] += float(weight_l2_loss.item())
        totals["weight_l2_norm_sq"] += float(weight_l2_norm_sq.item())
        totals["one_step_q_mse"] += float(one_step_q.item())
        totals["one_step_object_pos_mse"] += float(one_step_object_pos.item())
        totals["final_step_q_mse"] += float(final_q.item())
        totals["final_step_object_pos_mse"] += float(final_object_pos.item())
        totals["torque_abs_mean"] += float(future_torques.abs().mean().item())
        totals["residual_torque_abs_mean"] += float(torch.stack([aux["tau_residual"].abs().mean() for aux in aux_list]).mean().item())
        for metrics in (rollout_parts, robot_dyn_metrics, object_dyn_metrics, regs):
            for key, value in metrics.items():
                totals[key] += float(value.item())
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
    model = MultiStepRobotObjectGTDynamicsWorldModel(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        object_context_dim=layout.object_context_dim,
        hidden_dim=cfg.hidden_dim,
        dt=cfg.dt,
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
            f"val_robot_dyn={val_metrics['robot_dyn_loss']:.6f} "
            f"val_object_dyn={val_metrics['object_dyn_loss']:.6f}"
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
