from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_utils import (
    first_object_motion_timestep,
    franka_joint_and_gripper_positions,
    load_checkpoint_model,
    load_episode_arrays,
    load_episode_robot_joint_positions,
    parse_robot_target_name,
    resolve_episode_name,
    rollout_stepwise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Context DeLaN robot/object rollout trajectory.")
    parser.add_argument("--checkpoint", type=str,
    default='./outputs_context_delan/run_20260514_092259/best.pt')
    parser.add_argument("--dataset_file", type=str, 
    default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/Robot_and_Object/GT_dynamics/datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument("--start_t", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=35, help="Future rollout steps. 0 means rest of episode.")
    parser.add_argument("--target", type=str, default="gripper", help="Robot body to plot: 0, 1, ..., 7, or gripper.")
    parser.add_argument("--torque_key", type=str, default=None, choices=[None, "applied_torque", "computed_torque"])
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--output_name", type=str, default="context_delan_robot_object_trajectory.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
    parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
    parser.add_argument("--contact_consecutive_steps", type=int, default=3)
    parser.add_argument("--contact_settle_steps", type=int, default=5)
    return parser.parse_args()


def set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.05, 0.5 * float(np.max(maxs - mins)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_trajectory(
    real_episode_target: np.ndarray,
    real_episode_object: np.ndarray,
    gt_target: np.ndarray,
    pred_target: np.ndarray,
    gt_object: np.ndarray,
    pred_object: np.ndarray,
    time: np.ndarray,
    contact_t: int | None,
    dt: float,
    output_path: str,
    episode_name: str,
    target_name: str,
    start_t: int,
) -> tuple[float, float]:
    target_error = np.linalg.norm(pred_target - gt_target, axis=-1)
    object_error = np.linalg.norm(pred_object - gt_object, axis=-1)
    all_points = np.concatenate([real_episode_target, pred_target, real_episode_object, pred_object], axis=0)

    fig = plt.figure(figsize=(18, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.75, 1.0])
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    lines = []
    labels = []

    def add(handle, label: str) -> None:
        lines.append(handle)
        labels.append(label)

    add(
        ax3d.plot(real_episode_target[:, 0], real_episode_target[:, 1], real_episode_target[:, 2], color="black", linewidth=1.8)[0],
        f"Real full episode {target_name}",
    )
    add(
        ax3d.plot(pred_target[:, 0], pred_target[:, 1], pred_target[:, 2], color="tab:red", linestyle="--", linewidth=1.8)[0],
        f"Pred {target_name}",
    )
    add(
        ax3d.plot(real_episode_object[:, 0], real_episode_object[:, 1], real_episode_object[:, 2], color="tab:blue", linewidth=1.8)[0],
        "Real full episode object",
    )
    add(
        ax3d.plot(pred_object[:, 0], pred_object[:, 1], pred_object[:, 2], color="tab:orange", linestyle="--", linewidth=1.8)[0],
        "Pred object",
    )
    add(ax3d.scatter(*real_episode_target[0], color="black", s=40, marker="o"), f"Real episode start {target_name}")
    add(ax3d.scatter(*real_episode_object[0], color="tab:blue", s=40, marker="o"), "Real episode start object")
    add(ax3d.scatter(*gt_target[0], color="tab:purple", s=45, marker="o"), f"Real at prediction start {target_name}")
    add(ax3d.scatter(*gt_object[0], color="tab:cyan", s=45, marker="o"), "Real at prediction start object")
    add(ax3d.scatter(*pred_target[-1], color="tab:red", s=50, marker="x"), f"Pred {target_name} end")
    add(ax3d.scatter(*pred_object[-1], color="tab:orange", s=50, marker="x"), "Pred object end")
    if contact_t is not None and 0 <= contact_t < real_episode_object.shape[0]:
        add(
            ax3d.scatter(*real_episode_object[contact_t], color="tab:green", s=90, marker="*"),
            f"First contact proxy t={contact_t}",
        )
    ax3d.set_title(f"Context DeLaN {target_name.title()}/Object ({episode_name}, start_t={start_t})")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    set_axes_equal(ax3d, all_points)

    ax_leg = fig.add_subplot(grid[0, 1])
    ax_leg.axis("off")
    ax_leg.legend(lines, labels, loc="center left", frameon=True, fontsize=13, markerscale=1.8)

    ax_err = fig.add_subplot(grid[0, 2])
    ax_err.plot(time, target_error, color="tab:red", label=f"{target_name} error")
    ax_err.plot(time, object_error, color="tab:blue", label="object error")
    if contact_t is not None:
        contact_time = contact_t * dt
        if time[0] <= contact_time <= time[-1]:
            ax_err.axvline(contact_time, color="tab:green", linestyle=":", linewidth=1.8, label="First contact proxy")
    ax_err.set_title("Euclidean Position Error")
    ax_err.set_xlabel("episode time [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return float(np.sqrt(np.mean(target_error**2))), float(np.sqrt(np.mean(object_error**2)))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, layout, checkpoint_path = load_checkpoint_model(args.checkpoint, device)
    episode_name = resolve_episode_name(args.dataset_file, args.episode_index, args.episode_name)
    torque_key = args.torque_key or str(cfg["torque_key"])
    states, torques, _, _, _ = load_episode_arrays(args.dataset_file, episode_name, layout, torque_key)
    absolute_joint_pos = load_episode_robot_joint_positions(args.dataset_file, episode_name, layout.robot_dof)
    t_count = min(states.shape[0], torques.shape[0], absolute_joint_pos.shape[0])
    states = states[:t_count]
    torques = torques[:t_count]
    absolute_joint_pos = absolute_joint_pos[:t_count]

    history_len = int(cfg["history_len"])
    start_t = history_len - 1 if args.start_t is None else args.start_t
    if start_t < history_len - 1:
        raise ValueError(f"start_t must be >= history_len - 1 ({history_len - 1}).")
    if start_t >= states.shape[0] - 1:
        raise ValueError(f"start_t must leave at least one future step. Episode length is {states.shape[0]}.")

    max_available_steps = states.shape[0] - start_t - 1
    rollout_steps = max_available_steps if args.rollout_steps <= 0 else min(args.rollout_steps, max_available_steps)
    history_states = torch.from_numpy(states[start_t - history_len + 1 : start_t + 1]).unsqueeze(0).to(device)
    history_torques = torch.from_numpy(torques[start_t - history_len + 1 : start_t + 1]).unsqueeze(0).to(device)
    future_torques = torch.from_numpy(torques[start_t : start_t + rollout_steps]).unsqueeze(0).to(device)
    pred_future, _, z, failed_step, failure_reason = rollout_stepwise(model, history_states, history_torques, future_torques)

    actual_steps = pred_future.shape[0]
    gt_future = states[start_t + 1 : start_t + actual_steps + 1]
    target_name = parse_robot_target_name(args.target)
    gt_future_absolute_q = absolute_joint_pos[start_t + 1 : start_t + actual_steps + 1]
    relative_to_absolute_offset = absolute_joint_pos[start_t] - states[start_t, : layout.robot_dof]
    pred_absolute_q = pred_future[:, : layout.robot_dof] + relative_to_absolute_offset
    real_episode_trajectories = franka_joint_and_gripper_positions(absolute_joint_pos, args.tool_z_offset)
    gt_trajectories = franka_joint_and_gripper_positions(gt_future_absolute_q, args.tool_z_offset)
    pred_trajectories = franka_joint_and_gripper_positions(pred_absolute_q, args.tool_z_offset)
    robot_state_dim = layout.robot_state_dim
    contact_t = first_object_motion_timestep(
        object_pos=states[:, robot_state_dim : robot_state_dim + 3],
        dt=float(cfg["dt"]),
        displacement_threshold=args.object_displacement_threshold,
        velocity_threshold=args.object_velocity_threshold,
        consecutive_steps=args.contact_consecutive_steps,
        settle_steps=args.contact_settle_steps,
    )
    time = (np.arange(actual_steps, dtype=np.float32) + start_t + 1) * float(cfg["dt"])
    output_path = os.path.join(args.output_dir, args.output_name)
    target_rmse, object_rmse = plot_trajectory(
        real_episode_target=real_episode_trajectories[target_name],
        real_episode_object=states[:, robot_state_dim : robot_state_dim + 3],
        gt_target=gt_trajectories[target_name],
        pred_target=pred_trajectories[target_name],
        gt_object=gt_future[:, robot_state_dim : robot_state_dim + 3],
        pred_object=pred_future[:, robot_state_dim : robot_state_dim + 3],
        time=time,
        contact_t=contact_t,
        dt=float(cfg["dt"]),
        output_path=output_path,
        episode_name=episode_name,
        target_name=target_name,
        start_t=start_t,
    )

    print("===== Context DeLaN Robot/Object Trajectory Plot =====")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset_file: {args.dataset_file}")
    print(f"episode: {episode_name}")
    print(f"context_encoder: {cfg['context_encoder']}")
    print(f"latent_abs_mean: {float(z.abs().mean()):.8f}")
    print(f"target: {target_name}")
    print(f"torque_key: {torque_key}")
    print(f"start_t: {start_t}")
    print(f"first_contact_proxy_timestep: {contact_t}")
    print(f"requested_rollout_steps: {rollout_steps}")
    print(f"evaluated_rollout_steps: {actual_steps}")
    if failed_step is not None:
        print(f"rollout_stopped_at_step: {failed_step}")
        print(f"rollout_stop_reason: {failure_reason}")
    print(f"{target_name.replace(' ', '_')}_position_rmse_m: {target_rmse:.8f}")
    print(f"object_position_rmse_m: {object_rmse:.8f}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
