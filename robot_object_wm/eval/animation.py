from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from robot_object_wm.data.dataset import h5_open
from robot_object_wm.eval.episode import (
    EpisodeData,
    EpisodeRollout,
    compute_joint_positions,
    load_episode_data,
    precompute_prediction_trajectories,
    rollout_episode,
)
from robot_object_wm.eval.rollout import load_checkpoint_model
from robot_object_wm.models.utils import FrankaForwardKinematics

_CUBE_EDGES = np.asarray(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
    dtype=np.int64,
)
_CUBE_CORNERS_LOCAL = np.asarray(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)


@dataclass
class CollisionOverlay:
    pair_names: list[str]
    collision: np.ndarray
    distance: np.ndarray | None = None
    signed_distance: np.ndarray | None = None
    nearest_points: np.ndarray | None = None
    source_file: str = ""


def render_checkpoint_animation(
    *,
    model: torch.nn.Module,
    cfg,
    dataset_file: str,
    output_path: str,
    episode_index: int = 0,
    episode_name: str | None = None,
    pred_horizon: int = 10,
    start_t: int | None = None,
    rollout_steps: int | None = None,
    render_mode: str = "rollout",
    fps: int = 25,
    max_frames: int = 0,
    device: torch.device | str | None = None,
    target: str = "gripper",
    cube_size: float = 0.04,
    show_trails: bool = True,
    show_gt_future: bool = True,
    show_collision_info: bool = True,
    collision_group: str = "privileged_collision",
    collision_dataset_file: str | None = None,
) -> str:
    torch_device = torch.device(device or next(model.parameters()).device)
    episode = load_episode_data(
        dataset_file,
        episode_index=episode_index,
        episode_name=episode_name,
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
        torque_key=cfg.torque_key,
        subtract_env_origin=cfg.subtract_env_origin,
        max_frames=max_frames,
        dt=cfg.dt,
        state_prediction_mode=getattr(cfg, "state_prediction_mode", "full"),
        privileged_collision_observation=cfg.privileged_collision_observation,
        privileged_collision_group=cfg.privileged_collision_group,
        privileged_collision_pairs=cfg.privileged_collision_pairs,
    )
    fk = FrankaForwardKinematics(robot_dof=cfg.robot_dof, tool_z_offset=cfg.tool_z_offset).to(torch_device)
    fk.eval()
    collision_overlay = (
        load_collision_overlay(
            collision_dataset_file or dataset_file,
            episode.name,
            max_frames=episode.T,
            group_name=collision_group,
            fallback_dataset_file=dataset_file,
        )
        if show_collision_info
        else None
    )
    normalized_mode = str(render_mode).strip().lower()
    if normalized_mode not in {"rollout", "sliding"}:
        raise ValueError("render_mode must be 'rollout' or 'sliding'.")
    if normalized_mode == "rollout":
        start = max(int(cfg.history_len) - 1, int(start_t if start_t is not None else int(cfg.history_len) - 1))
        if start >= episode.T - 1:
            raise ValueError(f"start_t={start} leaves no rollout room for episode length T={episode.T}.")
        steps = int(rollout_steps or 0)
        if steps <= 0:
            steps = episode.T - start - 1
        rollout = rollout_episode(
            model,
            episode,
            start_t=start,
            rollout_steps=steps,
            history_len=cfg.history_len,
            device=torch_device,
        )
        return render_rollout_comparison_animation(
            episode=episode,
            rollout=rollout,
            fk=fk,
            device=torch_device,
            output_path=output_path,
            fps=fps,
            cube_size=cube_size,
            show_trails=show_trails,
            target=target,
            collision_overlay=collision_overlay,
        )

    pred_target_per_t, pred_object_per_t = precompute_prediction_trajectories(
        model,
        episode,
        history_len=cfg.history_len,
        pred_horizon=pred_horizon,
        robot_dof=cfg.robot_dof,
        target=target,
        fk=fk,
        device=torch_device,
    )
    return render_animation(
        episode=episode,
        fk=fk,
        device=torch_device,
        output_path=output_path,
        fps=fps,
        cube_size=cube_size,
        show_trails=show_trails,
        pred_target_per_t=pred_target_per_t,
        pred_object_per_t=pred_object_per_t,
        pred_horizon=pred_horizon,
        target=target,
        show_gt_future=show_gt_future,
        collision_overlay=collision_overlay,
    )


def render_dataset_animation(
    *,
    dataset_file: str,
    output_path: str,
    episode_index: int = 0,
    episode_name: str | None = None,
    robot_dof: int = 9,
    action_dim: int = 8,
    torque_dim: int = 9,
    torque_key: str = "applied_torque",
    subtract_env_origin: bool = True,
    dt: float = 0.02,
    tool_z_offset: float = 0.1034,
    fps: int = 25,
    max_frames: int = 0,
    cube_size: float = 0.04,
    show_trails: bool = True,
    show_gt_future: bool = True,
    show_collision_info: bool = True,
    collision_group: str = "privileged_collision",
    collision_dataset_file: str | None = None,
    device: torch.device | str = "cpu",
) -> str:
    torch_device = torch.device(device)
    episode = load_episode_data(
        dataset_file,
        episode_index=episode_index,
        episode_name=episode_name,
        robot_dof=robot_dof,
        action_dim=action_dim,
        torque_dim=torque_dim,
        torque_key=torque_key,
        subtract_env_origin=subtract_env_origin,
        max_frames=max_frames,
        dt=dt,
    )
    fk = FrankaForwardKinematics(robot_dof=robot_dof, tool_z_offset=tool_z_offset).to(torch_device)
    fk.eval()
    collision_overlay = (
        load_collision_overlay(
            collision_dataset_file or dataset_file,
            episode.name,
            max_frames=episode.T,
            group_name=collision_group,
            fallback_dataset_file=dataset_file,
        )
        if show_collision_info
        else None
    )
    return render_animation(
        episode=episode,
        fk=fk,
        device=torch_device,
        output_path=output_path,
        fps=fps,
        cube_size=cube_size,
        show_trails=show_trails,
        show_gt_future=show_gt_future,
        collision_overlay=collision_overlay,
    )


def _precompute_gt_future_trajectories(
    joints_3d: np.ndarray,
    object_pos: np.ndarray,
    horizon: int,
    target: str = "gripper",
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    T = int(joints_3d.shape[0])
    gt_target: list[np.ndarray | None] = [None] * T
    gt_object: list[np.ndarray | None] = [None] * T
    if horizon <= 0:
        return gt_target, gt_object
    target_pos = _target_positions_from_joints(joints_3d, target)
    for t in range(T - 1):
        n_steps = min(horizon, T - 1 - t)
        if n_steps <= 0:
            continue
        gt_target[t] = target_pos[t + 1 : t + 1 + n_steps].astype(np.float32, copy=True)
        gt_object[t] = object_pos[t + 1 : t + 1 + n_steps].astype(np.float32, copy=True)
    return gt_target, gt_object


def _target_positions_from_joints(joints_3d: np.ndarray, target: str) -> np.ndarray:
    target = target.lower()
    if target == "gripper":
        return joints_3d[:, -1, :]
    try:
        idx = int(target)
    except ValueError as exc:
        raise ValueError("target must be 'gripper' or an integer joint index.") from exc
    if idx < 0 or idx >= joints_3d.shape[1]:
        raise ValueError(f"target index must be in [0, {joints_3d.shape[1]}); got {idx}.")
    return joints_3d[:, idx, :]


def render_rollout_comparison_animation(
    *,
    episode: EpisodeData,
    rollout: EpisodeRollout,
    fk: FrankaForwardKinematics,
    device: torch.device,
    output_path: str,
    fps: int = 25,
    cube_size: float = 0.04,
    show_trails: bool = True,
    target: str = "gripper",
    collision_overlay: CollisionOverlay | None = None,
) -> str:
    """Render one open-loop rollout against the GT episode.

    Unlike the older sliding-horizon animation, this runs the model once from
    ``rollout.start_t`` and visualizes the resulting long prediction sequence.
    """

    if rollout.steps <= 0:
        raise ValueError(f"Rollout produced no frames: {rollout.failure_reason or 'unknown failure'}")

    layout = episode.state_layout
    start_t = int(rollout.start_t)
    frame_indices = np.arange(start_t + 1, start_t + 1 + rollout.steps, dtype=np.int64)

    q_offset = episode.joint_pos_abs[start_t] - episode.state[start_t, layout.robot_q_slice]
    pred_q_abs = rollout.pred_states[:, layout.robot_q_slice] + q_offset[None, :]
    gt_q_abs = episode.joint_pos_abs[frame_indices]
    pred_joints = compute_joint_positions(pred_q_abs.astype(np.float32), fk, device)
    gt_joints = compute_joint_positions(gt_q_abs.astype(np.float32), fk, device)

    pred_object_pos = rollout.pred_states[:, layout.object_pos_slice]
    gt_object_pos = rollout.gt_states[:, layout.object_pos_slice]
    pred_object_quat = rollout.pred_states[:, layout.object_quat_slice]
    gt_object_quat = rollout.gt_states[:, layout.object_quat_slice]

    pred_target = _target_positions_from_joints(pred_joints, target)
    gt_target = _target_positions_from_joints(gt_joints, target)
    target_error = np.linalg.norm(pred_target - gt_target, axis=-1)
    object_error = np.linalg.norm(pred_object_pos - gt_object_pos, axis=-1)

    output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    base = Path(output)
    metrics_path = base.with_suffix(".metrics.json")
    trajectory_path = base.with_suffix(".npz")
    metrics = {
        "episode": episode.name,
        "start_t": start_t,
        "steps": int(rollout.steps),
        "failed_step": rollout.failed_step,
        "failure_reason": rollout.failure_reason,
        f"{target}_position_rmse_m": float(np.sqrt(np.mean(target_error**2))),
        f"{target}_position_error_final_m": float(target_error[-1]),
        f"{target}_position_error_max_m": float(np.max(target_error)),
        "object_position_rmse_m": float(np.sqrt(np.mean(object_error**2))),
        "object_position_error_final_m": float(object_error[-1]),
        "object_position_error_max_m": float(np.max(object_error)),
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    np.savez_compressed(
        trajectory_path,
        frame_indices=frame_indices,
        gt_q_abs=gt_q_abs,
        pred_q_abs=pred_q_abs,
        gt_joints=gt_joints,
        pred_joints=pred_joints,
        gt_object_pos=gt_object_pos,
        pred_object_pos=pred_object_pos,
        gt_object_quat=gt_object_quat,
        pred_object_quat=pred_object_quat,
        target_error=target_error,
        object_error=object_error,
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    _set_bounds_from_points(
        ax,
        np.concatenate(
            [
                gt_joints.reshape(-1, 3),
                pred_joints.reshape(-1, 3),
                gt_object_pos,
                pred_object_pos,
                episode.object_pos,
            ],
            axis=0,
        ),
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, alpha=0.2)
    ax.view_init(elev=20.0, azim=-60.0)

    ax.plot(
        episode.object_pos[:, 0],
        episode.object_pos[:, 1],
        episode.object_pos[:, 2],
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        alpha=0.35,
        label="Full GT object path",
    )

    gt_joint_scatter, gt_links = _make_robot_artists(ax, gt_joints[0], color="tab:blue", label="GT robot")
    pred_joint_scatter, pred_links = _make_robot_artists(ax, pred_joints[0], color="tab:orange", label="Pred robot")
    gt_cube_lines = _make_cube_artists(ax, gt_object_pos[0], gt_object_quat[0], cube_size, color="tab:green", label="GT cube")
    pred_cube_lines = _make_cube_artists(
        ax,
        pred_object_pos[0],
        pred_object_quat[0],
        cube_size,
        color="tab:purple",
        label="Pred cube",
        linestyle="--",
    )

    nan_coord = np.asarray([np.nan], dtype=np.float32)
    nan_xyz = (nan_coord, nan_coord, nan_coord)
    contact_marker = ax.scatter(
        [np.nan],
        [np.nan],
        [np.nan],
        s=240,
        marker="*",
        c="tab:red",
        edgecolors="black",
        linewidths=0.8,
        depthshade=False,
        label="GT collision",
        zorder=8,
    )
    collision_points_a = ax.scatter(
        [np.nan],
        [np.nan],
        [np.nan],
        s=70,
        marker="o",
        c="tab:cyan",
        edgecolors="black",
        linewidths=0.5,
        depthshade=False,
        label="collision point A",
        zorder=9,
    )
    collision_points_b = ax.scatter(
        [np.nan],
        [np.nan],
        [np.nan],
        s=70,
        marker="X",
        c="tab:purple",
        edgecolors="black",
        linewidths=0.5,
        depthshade=False,
        label="collision point B",
        zorder=9,
    )

    gt_gripper_trail = gt_cube_trail = pred_gripper_trail = pred_cube_trail = None
    if show_trails:
        gt_gripper_trail = ax.plot([], [], [], color="tab:blue", linewidth=1.0, alpha=0.55)[0]
        gt_cube_trail = ax.plot([], [], [], color="tab:green", linewidth=1.0, alpha=0.55)[0]
        pred_gripper_trail = ax.plot([], [], [], color="tab:orange", linewidth=1.0, alpha=0.75)[0]
        pred_cube_trail = ax.plot([], [], [], color="tab:purple", linewidth=1.0, alpha=0.75)[0]

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=12)
    error_text = ax.text2D(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75", "boxstyle": "round,pad=0.35"},
    )
    collision_text = ax.text2D(
        0.02,
        0.82,
        "",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "0.75", "boxstyle": "round,pad=0.25"},
    )
    ax.set_title(f"Open-loop rollout comparison | Episode '{episode.name}'")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    def update(local_frame: int):
        gt_frame = int(frame_indices[local_frame])
        contact_summary = _collision_summary(collision_overlay, gt_frame)

        _update_robot_artists(gt_joint_scatter, gt_links, gt_joints[local_frame], color="tab:blue")
        _update_robot_artists(pred_joint_scatter, pred_links, pred_joints[local_frame], color="tab:orange")
        _update_cube_artists(gt_cube_lines, gt_object_pos[local_frame], gt_object_quat[local_frame], cube_size, color="tab:green")
        pred_cube_color = "tab:red" if object_error[local_frame] == np.max(object_error[: local_frame + 1]) else "tab:purple"
        _update_cube_artists(
            pred_cube_lines,
            pred_object_pos[local_frame],
            pred_object_quat[local_frame],
            cube_size,
            color=pred_cube_color,
        )

        if contact_summary["active"]:
            contact_marker._offsets3d = (
                gt_object_pos[local_frame : local_frame + 1, 0],
                gt_object_pos[local_frame : local_frame + 1, 1],
                gt_object_pos[local_frame : local_frame + 1, 2],
            )
        else:
            contact_marker._offsets3d = nan_xyz
        points_a, points_b = _collision_points_for_frame(collision_overlay, gt_frame)
        collision_points_a._offsets3d = (points_a[:, 0], points_a[:, 1], points_a[:, 2]) if points_a.size else nan_xyz
        collision_points_b._offsets3d = (points_b[:, 0], points_b[:, 1], points_b[:, 2]) if points_b.size else nan_xyz

        if show_trails:
            assert gt_gripper_trail is not None and gt_cube_trail is not None
            assert pred_gripper_trail is not None and pred_cube_trail is not None
            gt_gripper_trail.set_data_3d(
                gt_joints[: local_frame + 1, -1, 0],
                gt_joints[: local_frame + 1, -1, 1],
                gt_joints[: local_frame + 1, -1, 2],
            )
            gt_cube_trail.set_data_3d(
                gt_object_pos[: local_frame + 1, 0],
                gt_object_pos[: local_frame + 1, 1],
                gt_object_pos[: local_frame + 1, 2],
            )
            pred_gripper_trail.set_data_3d(
                pred_joints[: local_frame + 1, -1, 0],
                pred_joints[: local_frame + 1, -1, 1],
                pred_joints[: local_frame + 1, -1, 2],
            )
            pred_cube_trail.set_data_3d(
                pred_object_pos[: local_frame + 1, 0],
                pred_object_pos[: local_frame + 1, 1],
                pred_object_pos[: local_frame + 1, 2],
            )

        time_text.set_text(
            f"dataset step {gt_frame} | rollout {local_frame + 1}/{rollout.steps} | t={gt_frame * episode.dt:.2f}s"
        )
        error_text.set_text(
            f"{target} err={target_error[local_frame]:.4f} m  "
            f"object err={object_error[local_frame]:.4f} m\n"
            f"{target} RMSE={metrics[f'{target}_position_rmse_m']:.4f} m  "
            f"object RMSE={metrics['object_position_rmse_m']:.4f} m"
        )
        collision_text.set_text(contact_summary["text"])
        return ()

    anim = FuncAnimation(fig, update, frames=rollout.steps, interval=1000.0 / max(1, fps), blit=False)
    actual_path = output
    ext = Path(output).suffix.lower()
    if ext in (".mp4", ".mov", ".m4v"):
        try:
            anim.save(output, writer=FFMpegWriter(fps=fps, bitrate=2600), dpi=120)
        except Exception as exc:
            gif_path = str(Path(output).with_suffix(".gif"))
            print(f"[WARN] FFMpeg writer failed ({exc}); falling back to GIF at {gif_path}.")
            anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
            actual_path = gif_path
    else:
        anim.save(output, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    print(f"Saved rollout metrics: {metrics_path}")
    print(f"Saved rollout trajectory: {trajectory_path}")
    return actual_path


def render_animation(
    *,
    episode: EpisodeData,
    fk: FrankaForwardKinematics,
    device: torch.device,
    output_path: str,
    fps: int = 25,
    cube_size: float = 0.04,
    show_trails: bool = True,
    pred_target_per_t: list[np.ndarray | None] | None = None,
    pred_object_per_t: list[np.ndarray | None] | None = None,
    pred_horizon: int = 10,
    target: str = "gripper",
    show_gt_future: bool = True,
    collision_overlay: CollisionOverlay | None = None,
) -> str:
    joints_3d = compute_joint_positions(episode.joint_pos_abs, fk, device)
    object_pos = episode.object_pos
    object_quat = episode.object_quat
    target_pos = _target_positions_from_joints(joints_3d, target)
    show_pred = pred_target_per_t is not None and pred_object_per_t is not None
    gt_target_per_t, gt_object_per_t = _precompute_gt_future_trajectories(
        joints_3d,
        object_pos,
        pred_horizon if show_gt_future else 0,
        target=target,
    )
    show_gt_future = show_gt_future and any(item is not None for item in gt_target_per_t)
    T = episode.T

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    _set_bounds(
        ax,
        joints_3d,
        object_pos,
        pred_target_per_t,
        pred_object_per_t,
        gt_target_per_t if show_gt_future else None,
        gt_object_per_t if show_gt_future else None,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, alpha=0.2)
    ax.view_init(elev=20.0, azim=-60.0)

    joint_xyz = joints_3d[0]
    joint_scatter = ax.scatter(
        joint_xyz[:, 0],
        joint_xyz[:, 1],
        joint_xyz[:, 2],
        s=180,
        c="tab:blue",
        edgecolors="black",
        linewidths=0.8,
        depthshade=True,
        zorder=5,
    )
    link_lines = []
    for idx in range(joint_xyz.shape[0] - 1):
        line = ax.plot(
            [joint_xyz[idx, 0], joint_xyz[idx + 1, 0]],
            [joint_xyz[idx, 1], joint_xyz[idx + 1, 1]],
            [joint_xyz[idx, 2], joint_xyz[idx + 1, 2]],
            color="black",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=4,
        )[0]
        link_lines.append(line)

    corners0 = cube_corners(object_pos[0], object_quat[0], cube_size)
    cube_lines = []
    for edge in _CUBE_EDGES:
        i, j = int(edge[0]), int(edge[1])
        cube_lines.append(
            ax.plot(
                [corners0[i, 0], corners0[j, 0]],
                [corners0[i, 1], corners0[j, 1]],
                [corners0[i, 2], corners0[j, 2]],
                color="tab:red",
                linewidth=2.0,
                zorder=3,
            )[0]
        )

    contact_marker = ax.scatter(
        [np.nan],
        [np.nan],
        [np.nan],
        s=260,
        marker="*",
        c="tab:orange",
        edgecolors="black",
        linewidths=0.8,
        depthshade=False,
        label="Active collision",
        zorder=7,
    )
    collision_points_a = ax.scatter(
        [np.nan],
        [np.nan],
        [np.nan],
        s=80,
        marker="o",
        c="tab:cyan",
        edgecolors="black",
        linewidths=0.6,
        depthshade=False,
        label="Collision point A",
        zorder=8,
    )
    collision_points_b = ax.scatter(
        [np.nan],
        [np.nan],
        [np.nan],
        s=80,
        marker="X",
        c="tab:purple",
        edgecolors="black",
        linewidths=0.6,
        depthshade=False,
        label="Collision point B",
        zorder=8,
    )

    gripper_trail = cube_trail = None
    if show_trails:
        gripper_trail = ax.plot([], [], [], color="tab:blue", linewidth=1.0, alpha=0.55, zorder=2)[0]
        cube_trail = ax.plot([], [], [], color="tab:red", linewidth=1.0, alpha=0.55, zorder=2)[0]

    ax.plot(
        target_pos[:, 0],
        target_pos[:, 1],
        target_pos[:, 2],
        color="tab:cyan",
        linestyle=":",
        linewidth=1.4,
        alpha=0.45,
        label=f"Dataset {target} path",
        zorder=1.5,
    )
    ax.plot(
        object_pos[:, 0],
        object_pos[:, 1],
        object_pos[:, 2],
        color="tab:green",
        linestyle=":",
        linewidth=1.4,
        alpha=0.45,
        label="Dataset object path",
        zorder=1.5,
    )

    nan_coord = np.asarray([np.nan], dtype=np.float32)
    nan_xyz = (nan_coord, nan_coord, nan_coord)
    pred_target_line = pred_object_line = None
    gt_target_line = gt_object_line = None
    if show_pred:
        pred_target_line = ax.plot(
            *nan_xyz,
            color="tab:orange",
            linestyle="--",
            linewidth=2.0,
            label=f"Pred {target} (N={pred_horizon})",
        )[0]
        pred_object_line = ax.plot(
            *nan_xyz,
            color="tab:purple",
            linestyle="--",
            linewidth=2.0,
            label=f"Pred object (N={pred_horizon})",
        )[0]
    if show_gt_future:
        gt_target_line = ax.plot(
            *nan_xyz,
            color="tab:cyan",
            linestyle="-",
            linewidth=2.0,
            label=f"GT {target} future (N={pred_horizon})",
        )[0]
        gt_object_line = ax.plot(
            *nan_xyz,
            color="tab:green",
            linestyle="-",
            linewidth=2.0,
            label=f"GT object future (N={pred_horizon})",
        )[0]
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=12)
    collision_text = ax.text2D(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75", "boxstyle": "round,pad=0.35"},
    )
    ax.set_title(f"Episode '{episode.name}'")

    def update(frame: int):
        contact_summary = _collision_summary(collision_overlay, frame)
        cube_color = _cube_collision_color(contact_summary)
        cube_width = 3.2 if contact_summary["active"] else 2.0
        joint_scatter._offsets3d = (
            joints_3d[frame, :, 0],
            joints_3d[frame, :, 1],
            joints_3d[frame, :, 2],
        )
        for idx, line in enumerate(link_lines):
            line.set_color("tab:orange" if contact_summary["gripper_active"] and idx >= len(link_lines) - 2 else "black")
            line.set_data_3d(
                [joints_3d[frame, idx, 0], joints_3d[frame, idx + 1, 0]],
                [joints_3d[frame, idx, 1], joints_3d[frame, idx + 1, 1]],
                [joints_3d[frame, idx, 2], joints_3d[frame, idx + 1, 2]],
            )
        corners = cube_corners(object_pos[frame], object_quat[frame], cube_size)
        for line, edge in zip(cube_lines, _CUBE_EDGES):
            i, j = int(edge[0]), int(edge[1])
            line.set_color(cube_color)
            line.set_linewidth(cube_width)
            line.set_data_3d(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
            )
        if contact_summary["active"]:
            contact_marker._offsets3d = (
                object_pos[frame : frame + 1, 0],
                object_pos[frame : frame + 1, 1],
                object_pos[frame : frame + 1, 2],
            )
        else:
            contact_marker._offsets3d = nan_xyz
        points_a, points_b = _collision_points_for_frame(collision_overlay, frame)
        if points_a.size > 0:
            collision_points_a._offsets3d = (points_a[:, 0], points_a[:, 1], points_a[:, 2])
        else:
            collision_points_a._offsets3d = nan_xyz
        if points_b.size > 0:
            collision_points_b._offsets3d = (points_b[:, 0], points_b[:, 1], points_b[:, 2])
        else:
            collision_points_b._offsets3d = nan_xyz
        if gripper_trail is not None and cube_trail is not None:
            gripper_trail.set_data_3d(joints_3d[: frame + 1, -1, 0], joints_3d[: frame + 1, -1, 1], joints_3d[: frame + 1, -1, 2])
            cube_trail.set_data_3d(object_pos[: frame + 1, 0], object_pos[: frame + 1, 1], object_pos[: frame + 1, 2])
        if pred_target_line is not None and pred_object_line is not None:
            pred_target = pred_target_per_t[frame] if frame < len(pred_target_per_t) else None
            pred_object = pred_object_per_t[frame] if frame < len(pred_object_per_t) else None
            if pred_target is not None and pred_object is not None:
                pred_target_line.set_data_3d(pred_target[:, 0], pred_target[:, 1], pred_target[:, 2])
                pred_object_line.set_data_3d(pred_object[:, 0], pred_object[:, 1], pred_object[:, 2])
            else:
                pred_target_line.set_data_3d(*nan_xyz)
                pred_object_line.set_data_3d(*nan_xyz)
        if gt_target_line is not None and gt_object_line is not None:
            gt_target = gt_target_per_t[frame] if frame < len(gt_target_per_t) else None
            gt_object = gt_object_per_t[frame] if frame < len(gt_object_per_t) else None
            if gt_target is not None and gt_object is not None:
                gt_target_line.set_data_3d(gt_target[:, 0], gt_target[:, 1], gt_target[:, 2])
                gt_object_line.set_data_3d(gt_object[:, 0], gt_object[:, 1], gt_object[:, 2])
            else:
                gt_target_line.set_data_3d(*nan_xyz)
                gt_object_line.set_data_3d(*nan_xyz)
        time_text.set_text(f"t = {frame * episode.dt:.2f} s   step {frame + 1}/{T}")
        collision_text.set_text(contact_summary["text"])
        return ()

    anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / max(1, fps), blit=False)
    output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    actual_path = output
    ext = Path(output).suffix.lower()
    if ext in (".mp4", ".mov", ".m4v"):
        try:
            anim.save(output, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=120)
        except Exception as exc:
            gif_path = str(Path(output).with_suffix(".gif"))
            print(f"[WARN] FFMpeg writer failed ({exc}); falling back to GIF at {gif_path}.")
            anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
            actual_path = gif_path
    else:
        anim.save(output, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return actual_path


def save_point_animation(points: np.ndarray, output_path: str, fps: int = 20) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(points[0:1, 0], points[0:1, 1], points[0:1, 2])
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float((maxs - mins).max()) * 0.5, 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    def update(frame: int):
        scatter._offsets3d = (points[frame : frame + 1, 0], points[frame : frame + 1, 1], points[frame : frame + 1, 2])
        return ()

    anim = FuncAnimation(fig, update, frames=points.shape[0], interval=1000.0 / max(1, fps), blit=False)
    path = str(Path(output_path))
    anim.save(path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return path


def _make_robot_artists(ax, joints: np.ndarray, *, color: str, label: str):
    scatter = ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        s=120,
        c=color,
        edgecolors="black",
        linewidths=0.6,
        depthshade=True,
        label=label,
        zorder=5,
    )
    links = []
    for idx in range(joints.shape[0] - 1):
        links.append(
            ax.plot(
                [joints[idx, 0], joints[idx + 1, 0]],
                [joints[idx, 1], joints[idx + 1, 1]],
                [joints[idx, 2], joints[idx + 1, 2]],
                color=color,
                linewidth=2.4,
                alpha=0.9,
                solid_capstyle="round",
                zorder=4,
            )[0]
        )
    return scatter, links


def _update_robot_artists(scatter, links: list, joints: np.ndarray, *, color: str) -> None:
    scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
    for idx, line in enumerate(links):
        line.set_color(color)
        line.set_data_3d(
            [joints[idx, 0], joints[idx + 1, 0]],
            [joints[idx, 1], joints[idx + 1, 1]],
            [joints[idx, 2], joints[idx + 1, 2]],
        )


def _make_cube_artists(
    ax,
    center: np.ndarray,
    quat: np.ndarray,
    size: float,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> list:
    corners = cube_corners(center, quat, size)
    lines = []
    for edge_index, edge in enumerate(_CUBE_EDGES):
        i, j = int(edge[0]), int(edge[1])
        lines.append(
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label if edge_index == 0 else None,
                zorder=3,
            )[0]
        )
    return lines


def _update_cube_artists(
    lines: list,
    center: np.ndarray,
    quat: np.ndarray,
    size: float,
    *,
    color: str,
) -> None:
    corners = cube_corners(center, quat, size)
    for line, edge in zip(lines, _CUBE_EDGES):
        i, j = int(edge[0]), int(edge[1])
        line.set_color(color)
        line.set_data_3d(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
        )


def cube_corners(center: np.ndarray, quat: np.ndarray, size: float) -> np.ndarray:
    half = size / 2.0
    return (_CUBE_CORNERS_LOCAL * half) @ _quat_to_rotation_matrix_np(quat).T + center[None, :]


def collision_augmented_path(dataset_file: str) -> str:
    path = Path(dataset_file)
    if path.stem.endswith("_collision_augmented"):
        return str(path)
    return str(path.with_name(f"{path.stem}_collision_augmented{path.suffix}"))


def _decode_hdf5_strings(values: np.ndarray) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _candidate_collision_files(dataset_file: str, fallback_dataset_file: str | None = None) -> list[str]:
    candidates: list[str] = []
    for path in (dataset_file, fallback_dataset_file):
        if not path:
            continue
        for candidate in (path, collision_augmented_path(path)):
            candidate = os.path.abspath(candidate)
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def load_collision_overlay(
    dataset_file: str,
    episode_name: str,
    *,
    max_frames: int = 0,
    group_name: str = "privileged_collision",
    fallback_dataset_file: str | None = None,
) -> CollisionOverlay | None:
    for path in _candidate_collision_files(dataset_file, fallback_dataset_file):
        if not os.path.isfile(path):
            continue
        try:
            with h5_open(path) as file:
                group_path = f"data/{episode_name}/{group_name}"
                if group_path not in file:
                    continue
                group = file[group_path]
                pair_names = _decode_hdf5_strings(np.asarray(group["pair_names"]))
                collision = np.asarray(group["collision"], dtype=bool)
                distance = np.asarray(group["distance"], dtype=np.float32) if "distance" in group else None
                signed_distance = (
                    np.asarray(group["signed_distance"], dtype=np.float32) if "signed_distance" in group else None
                )
                nearest_points = (
                    np.asarray(group["nearest_points"], dtype=np.float32) if "nearest_points" in group else None
                )
        except Exception as exc:
            print(f"[WARN] Could not read collision info from {path}: {exc}")
            continue

        if max_frames > 0:
            collision = collision[:max_frames]
            if distance is not None:
                distance = distance[:max_frames]
            if signed_distance is not None:
                signed_distance = signed_distance[:max_frames]
            if nearest_points is not None:
                nearest_points = nearest_points[:max_frames]
        print(f"Loaded collision overlay: {path}::{group_path}")
        return CollisionOverlay(
            pair_names=pair_names,
            collision=collision,
            distance=distance,
            signed_distance=signed_distance,
            nearest_points=nearest_points,
            source_file=path,
        )
    return None


def _collision_summary(overlay: CollisionOverlay | None, frame: int) -> dict[str, object]:
    if overlay is None:
        return {
            "active": False,
            "gripper_active": False,
            "ground_active": False,
            "text": "collision: unavailable",
        }
    if frame >= overlay.collision.shape[0]:
        return {
            "active": False,
            "gripper_active": False,
            "ground_active": False,
            "text": "collision: out of range",
        }

    active_indices = np.flatnonzero(overlay.collision[frame])
    active_names = [overlay.pair_names[int(idx)] for idx in active_indices]
    gripper_active = any(name in {"object_left_finger", "object_right_finger", "object_gripper"} for name in active_names)
    ground_active = any(name in {"object_ground", "left_finger_ground", "right_finger_ground", "gripper_ground"} for name in active_names)

    if active_names:
        parts = []
        for idx, name in zip(active_indices[:4], active_names[:4]):
            if overlay.distance is not None and frame < overlay.distance.shape[0]:
                parts.append(f"{name}:{overlay.distance[frame, int(idx)]:.3f}m")
            else:
                parts.append(name)
        if len(active_names) > 4:
            parts.append(f"+{len(active_names) - 4} more")
        text = "collision: " + ", ".join(parts)
    else:
        if overlay.distance is not None and overlay.distance.shape[1] > 0:
            finite = overlay.distance[frame][np.isfinite(overlay.distance[frame])]
            text = f"collision: none  nearest={float(np.min(finite)):.3f}m" if finite.size else "collision: none"
        else:
            text = "collision: none"

    return {
        "active": bool(active_names),
        "gripper_active": gripper_active,
        "ground_active": ground_active,
        "text": text,
    }


def _collision_points_for_frame(overlay: CollisionOverlay | None, frame: int) -> tuple[np.ndarray, np.ndarray]:
    empty = np.zeros((0, 3), dtype=np.float32)
    if overlay is None or overlay.nearest_points is None:
        return empty, empty
    if frame >= overlay.collision.shape[0] or frame >= overlay.nearest_points.shape[0]:
        return empty, empty

    active_indices = np.flatnonzero(overlay.collision[frame])
    if active_indices.size == 0:
        return empty, empty

    points = overlay.nearest_points[frame, active_indices]
    if points.ndim != 3 or points.shape[-2:] != (2, 3):
        return empty, empty

    finite = np.isfinite(points).all(axis=-1)
    points_a = points[:, 0, :][finite[:, 0]]
    points_b = points[:, 1, :][finite[:, 1]]
    return points_a.astype(np.float32, copy=False), points_b.astype(np.float32, copy=False)


def _cube_collision_color(summary: dict[str, object]) -> str:
    if bool(summary["gripper_active"]):
        return "tab:orange"
    if bool(summary["ground_active"]):
        return "tab:red"
    return "tab:red"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render WMDynamics dataset/prediction animation.")
    parser.add_argument("--dataset_file", type=str, default="./dataset/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5")
    parser.add_argument("--checkpoint", type=str, default="./outputs_wm_dynamics/run_20260622_100149/best.pt")
    parser.add_argument("--pointcloud_file", type=str, default=None)
    parser.add_argument(
        "--hybrid_rollout_feedback_mode",
        choices=("robot_native", "rigidformer_pose"),
        default=None,
    )
    parser.add_argument(
        "--hybrid_gripper_pointcloud_mode",
        choices=("gt", "predicted_fk"),
        default=None,
    )
    parser.add_argument("--output", type=str, default="./eval_outputs/wm_dynamics_episode.mp4")
    parser.add_argument("--episode_index", type=int, default=3)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--action_dim", type=int, default=8)
    parser.add_argument("--torque_dim", type=int, default=9)
    parser.add_argument("--torque_key", choices=("applied_torque", "computed_torque"), default="applied_torque")
    parser.add_argument("--no_subtract_env_origin", dest="subtract_env_origin", action="store_false", default=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument(
        "--render_mode",
        choices=("rollout", "sliding"),
        default="rollout",
        help="rollout renders one long open-loop rollout; sliding renders the older per-frame short-horizon preview.",
    )
    parser.add_argument("--start_t", type=int, default=25, help="Dataset frame to start the long rollout animation.")
    parser.add_argument("--rollout_steps", type=int, default=0, help="Long-rollout steps. 0 means until episode end.")
    parser.add_argument("--pred_horizon", type=int, default=10)
    parser.add_argument("--target", type=str, default="gripper")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--cube_size", type=float, default=0.04)
    parser.add_argument("--no_trails", dest="show_trails", action="store_false", default=True)
    parser.add_argument("--no_gt_future", dest="show_gt_future", action="store_false", default=False)
    collision = parser.add_argument_group("collision overlay")
    collision.add_argument("--collision_info", dest="show_collision_info", action="store_true", default=True)
    collision.add_argument("--no_collision_info", dest="show_collision_info", action="store_false")
    collision.add_argument("--collision_group", type=str, default="privileged_collision")
    collision.add_argument("--collision_dataset_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.checkpoint:
        loaded = load_checkpoint_model(args.checkpoint, device=args.device)
        if args.pointcloud_file is not None:
            loaded.config.pointcloud_file = os.path.abspath(args.pointcloud_file)
            if hasattr(loaded.model, "pointcloud_file"):
                loaded.model.pointcloud_file = loaded.config.pointcloud_file
        if args.hybrid_rollout_feedback_mode is not None:
            loaded.config.hybrid_rollout_feedback_mode = args.hybrid_rollout_feedback_mode
            if hasattr(loaded.model, "feedback_mode"):
                loaded.model.feedback_mode = args.hybrid_rollout_feedback_mode
        if args.hybrid_gripper_pointcloud_mode is not None:
            loaded.config.hybrid_gripper_pointcloud_mode = args.hybrid_gripper_pointcloud_mode
            if hasattr(loaded.model, "gripper_pointcloud_mode"):
                loaded.model.gripper_pointcloud_mode = args.hybrid_gripper_pointcloud_mode
        written = render_checkpoint_animation(
            model=loaded.model,
            cfg=loaded.config,
            dataset_file=args.dataset_file,
            output_path=args.output,
            episode_index=args.episode_index,
            episode_name=args.episode_name,
            pred_horizon=args.pred_horizon,
            start_t=args.start_t,
            rollout_steps=args.rollout_steps,
            render_mode=args.render_mode,
            fps=args.fps,
            max_frames=args.max_frames,
            device=next(loaded.model.parameters()).device,
            target=args.target,
            cube_size=args.cube_size,
            show_trails=args.show_trails,
            show_gt_future=args.show_gt_future,
            show_collision_info=args.show_collision_info,
            collision_group=args.collision_group,
            collision_dataset_file=args.collision_dataset_file,
        )
    else:
        written = render_dataset_animation(
            dataset_file=args.dataset_file,
            output_path=args.output,
            episode_index=args.episode_index,
            episode_name=args.episode_name,
            robot_dof=args.robot_dof,
            action_dim=args.action_dim,
            torque_dim=args.torque_dim,
            torque_key=args.torque_key,
            subtract_env_origin=args.subtract_env_origin,
            dt=args.dt,
            tool_z_offset=args.tool_z_offset,
            fps=args.fps,
            max_frames=args.max_frames,
            cube_size=args.cube_size,
            show_trails=args.show_trails,
            show_gt_future=args.show_gt_future,
            show_collision_info=args.show_collision_info,
            collision_group=args.collision_group,
            collision_dataset_file=args.collision_dataset_file,
            device=args.device,
        )
    print(f"Saved: {written}")


def _quat_to_rotation_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = quat / max(np.linalg.norm(quat), 1.0e-8)
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _set_bounds(
    ax,
    joints_3d: np.ndarray,
    object_pos: np.ndarray,
    pred_target_per_t: list[np.ndarray | None] | None,
    pred_object_per_t: list[np.ndarray | None] | None,
    gt_target_per_t: list[np.ndarray | None] | None = None,
    gt_object_per_t: list[np.ndarray | None] | None = None,
) -> None:
    points = [joints_3d.reshape(-1, 3), object_pos]
    for values in (pred_target_per_t, pred_object_per_t, gt_target_per_t, gt_object_per_t):
        if values is None:
            continue
        points.extend(arr for arr in values if arr is not None and arr.size > 0)
    all_points = np.concatenate(points, axis=0)
    mins = all_points.min(axis=0) - 0.05
    maxs = all_points.max(axis=0) + 0.05
    center = 0.5 * (mins + maxs)
    half = max(float((maxs - mins).max()) * 0.5, 0.25)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _set_bounds_from_points(ax, points: np.ndarray) -> None:
    finite = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = finite[np.isfinite(finite).all(axis=-1)]
    if finite.size == 0:
        finite = np.zeros((1, 3), dtype=np.float32)
    mins = finite.min(axis=0) - 0.05
    maxs = finite.max(axis=0) + 0.05
    center = 0.5 * (mins + maxs)
    half = max(float((maxs - mins).max()) * 0.5, 0.25)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


if __name__ == "__main__":
    main()
