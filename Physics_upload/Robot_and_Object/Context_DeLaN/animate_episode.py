"""Animate one (or all) episode(s) of the Context DeLaN robot+object dataset.

Mirrors ``Robot_and_Object/No_object_Euler/animate_episode.py`` so the two
models can be compared side by side.  At every frame this produces an MP4
(or GIF fallback) that shows:

* Eight spherical markers at the world-frame positions of the Franka joints
  (base + 7 arm joints + gripper tip) computed via the analytic FK helper
  in ``eval_utils.franka_joint_and_gripper_positions``.
* Solid lines joining consecutive joints (the link skeleton).
* A small wire-frame cube at the manipulated object's recorded position and
  orientation, drawn in env-local frame.
* Optional past-trajectory trails (gripper + cube).
* Optional ``--pred_horizon``-step predicted trajectory (dashed) overlaid
  with the recorded ``--pred_horizon``-step ground-truth trajectory (solid),
  both refreshed at every frame so they evolve as the animation plays.
* Optional right-side time-series panel showing per-frame model signals
  (predicted object lin/ang acc magnitude, residual torque magnitude,
  and latent ``||z||``) — see ``--show_signal_panel``.
* Fixed world-frame axis ranges so framing is consistent across episodes.
* ``--all_episodes`` mode that loops over every episode in the dataset
  and saves one MP4 per episode.

Usage:

    python animate_episode.py \\
        --dataset_file ./datasets/<...>.hdf5 \\
        --checkpoint ./outputs_context_delan/run_<X>/best.pt \\
        --episode_index 4 --output ./eval_outputs/animation.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter  # noqa: E402

# 3-D projection registration (side effect of importing).
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

from eval_utils import (  # noqa: E402
    franka_joint_and_gripper_positions,
    load_checkpoint_model,
    load_episode_arrays,
    load_episode_robot_joint_positions,
    resolve_episode_name,
    episode_names as _episode_names,
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_file", type=str,
        default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep.hdf5",
        help="HDF5 dataset with one or more recorded episodes.",
    )
    parser.add_argument("--episode_index", type=int, default=4)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument(
        "--output", type=str,
        default="./eval_outputs/context_delan_episode_animation_light.mp4",
        help="Output .mp4 (preferred, needs ffmpeg) or .gif.  When "
             "--all_episodes is set this is used as a template: the episode "
             "name is inserted into the file stem.",
    )
    parser.add_argument(
        "--all_episodes", action="store_true", default=False,
        help="Render one animation per episode in the dataset file.  When "
             "set, --episode_index / --episode_name are ignored.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=0,
        help="Cap the number of episodes rendered when --all_episodes is "
             "set (0 = no cap).  Ignored otherwise.",
    )

    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--joint_marker_size", type=float, default=240.0)
    parser.add_argument("--link_linewidth", type=float, default=4.0)
    parser.add_argument("--cube_size", type=float, default=0.04)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument("--show_trails", action="store_true", default=True)
    parser.add_argument("--no_trails", dest="show_trails", action="store_false")
    parser.add_argument(
        "--no_subtract_env_origin", dest="subtract_env_origin",
        action="store_false", default=True,
        help="Match the training default: subtract env_origin from object position.",
    )
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--dpi", type=int, default=120)

    # Fixed world-frame axis ranges (meters).
    parser.add_argument("--xlim", type=float, nargs=2, default=[-0.2, 0.8],
                        metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", type=float, nargs=2, default=[-0.4, 0.4],
                        metavar=("YMIN", "YMAX"))
    parser.add_argument("--zlim", type=float, nargs=2, default=[-0.1, 0.8],
                        metavar=("ZMIN", "ZMAX"))

    # ----- Predicted-trajectory overlay (requires a checkpoint).
    parser.add_argument(
        "--checkpoint", type=str, 
        default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/Robot_and_Object/Context_DeLaN/outputs_context_delan/run_20260514_092259/best.pt",
        help="Optional path to best.pt/last.pt.  When given, the animation "
             "overlays an N-step predicted trajectory at every frame.",
    )
    parser.add_argument("--pred_horizon", type=int, default=10,
                        help="Number of predicted future steps to draw at "
                             "each frame.  Ignored when --checkpoint is None.")
    parser.add_argument(
        "--torque_key", type=str, default=None,
        choices=[None, "applied_torque", "computed_torque"],
        help="Recorded torque field used as the model's per-step input.  "
             "Default: read from the checkpoint config.",
    )

    parser.add_argument("--pred_gripper_color", type=str, default="tab:orange")
    parser.add_argument("--pred_cube_color", type=str, default="tab:purple")
    parser.add_argument("--pred_linewidth", type=float, default=2.0)

    # ----- GT future trajectory overlay (on by default).
    parser.add_argument("--show_gt_future", action="store_true", default=True)
    parser.add_argument("--no_gt_future", dest="show_gt_future",
                        action="store_false")
    parser.add_argument("--gt_future_gripper_color", type=str, default="tab:cyan")
    parser.add_argument("--gt_future_cube_color", type=str, default="tab:green")
    parser.add_argument("--gt_future_linewidth", type=float, default=2.0)

    # ----- Signal panel.
    parser.add_argument(
        "--show_signal_panel", action="store_true", default=False,
        help="Add a right-side time-series panel showing the model's per-"
             "frame outputs.  Requires --checkpoint.",
    )
    parser.add_argument("--signal_linewidth", type=float, default=1.5)
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Dataset / FK helpers
# ----------------------------------------------------------------------------

def _h5_open(path: str):
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def load_episode_for_animation(
    dataset_file: str,
    episode_name: str,
    robot_dof: int,
    subtract_env_origin: bool,
    max_frames: int,
) -> dict:
    """Load the visualisation arrays (absolute joint pos for FK,
    env-local object pose) for one episode."""
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with _h5_open(dataset_file) as f:
        ep = f["data"][episode_name]
        try:
            joint_pos_abs = np.asarray(
                ep["states"]["articulation"]["robot"]["joint_position"],
                dtype=np.float32,
            )[:, :robot_dof]
            joint_pos_source = "states/articulation/robot/joint_position (absolute)"
        except KeyError:
            joint_pos_abs = np.asarray(ep["obs"]["joint_pos"], dtype=np.float32)[:, :robot_dof]
            joint_pos_source = "obs/joint_pos (relative; FK may render wrong configuration)"
        root_pose = np.asarray(
            ep["states"]["rigid_object"]["object"]["root_pose"], dtype=np.float32,
        )
        object_pos_w = root_pose[:, :3]
        object_quat = root_pose[:, 3:7]
        if subtract_env_origin and "initial_state" in ep:
            try:
                origin = np.asarray(
                    ep["initial_state"]["articulation"]["robot"]["root_pose"],
                    dtype=np.float32,
                )[0, :3]
            except KeyError:
                origin = np.zeros(3, dtype=np.float32)
        else:
            origin = np.zeros(3, dtype=np.float32)
        object_pos = object_pos_w - origin[None, :]

    T = min(joint_pos_abs.shape[0], object_pos.shape[0], object_quat.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)
    return {
        "name": episode_name,
        "T": T,
        "joint_pos": joint_pos_abs[:T],   # absolute angles for FK
        "joint_pos_source": joint_pos_source,
        "object_pos": object_pos[:T],
        "object_quat": object_quat[:T],
    }


def compute_joint_positions(joint_pos_abs: np.ndarray, tool_z_offset: float) -> np.ndarray:
    """Run the analytic Franka FK chain and stack into (T, 9, 3).

    Index 0 = base; indices 1..7 = arm joints 1..7; index 8 = gripper tip.
    Uses the same FK helper as ``eval_utils.franka_joint_and_gripper_positions``.
    """
    traj = franka_joint_and_gripper_positions(joint_pos_abs, tool_z_offset=tool_z_offset)
    ordered = [traj["joint 0"]]
    for i in range(1, 8):
        ordered.append(traj[f"joint {i}"])
    ordered.append(traj["gripper"])
    return np.stack(ordered, axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# Cube wire-frame
# ----------------------------------------------------------------------------

_CUBE_EDGES = np.asarray([
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
], dtype=np.int64)

_CUBE_CORNERS_LOCAL = np.asarray([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=np.float32)


def _quat_to_rotation_matrix_np(quat: np.ndarray) -> np.ndarray:
    """wxyz unit quaternion -> 3x3 rotation matrix (numpy)."""
    q = quat / max(np.linalg.norm(quat), 1.0e-8)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def cube_corners(center: np.ndarray, quat: np.ndarray, size: float) -> np.ndarray:
    half = size / 2.0
    corners_local = _CUBE_CORNERS_LOCAL * half
    R = _quat_to_rotation_matrix_np(quat)
    return (corners_local @ R.T) + center[None, :]


# ----------------------------------------------------------------------------
# Rollouts and GT future trajectories
# ----------------------------------------------------------------------------

def _precompute_gt_future_trajectories(
    joints_3d: np.ndarray,
    object_pos: np.ndarray,
    T: int,
    horizon: int,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Slice the recorded next-``horizon``-step gripper and cube positions."""
    gt_gripper: list[np.ndarray | None] = [None] * T
    gt_cube: list[np.ndarray | None] = [None] * T
    if horizon <= 0:
        return gt_gripper, gt_cube
    gripper_xyz = joints_3d[:, -1, :]
    for t in range(T - 1):
        n_steps = min(horizon, T - 1 - t)
        if n_steps <= 0:
            continue
        gt_gripper[t] = gripper_xyz[t + 1 : t + 1 + n_steps, :].astype(np.float32, copy=True)
        gt_cube[t] = object_pos[t + 1 : t + 1 + n_steps, :].astype(np.float32, copy=True)
    return gt_gripper, gt_cube


def precompute_pred_trajectories(
    model,
    layout,
    states: np.ndarray,
    torques: np.ndarray,
    joint_pos_abs: np.ndarray,
    pred_horizon: int,
    history_len: int,
    device: torch.device,
    tool_z_offset: float,
) -> tuple[
    list[np.ndarray | None],
    list[np.ndarray | None],
    np.ndarray,  # obj_lin_acc_per_t  (T, 3)
    np.ndarray,  # obj_ang_acc_per_t  (T, 3)
    np.ndarray,  # residual_torque_per_t (T, robot_dof)
    np.ndarray,  # z_norm_per_t  (T,)
]:
    """For every frame ``t``, roll the Context DeLaN model out for up to
    ``pred_horizon`` steps and return:

      * per-frame predicted gripper-tip positions (list of (n, 3) or None),
      * per-frame predicted cube-center positions (same shape),
      * per-frame predicted object linear acceleration ``aux["object_lin_acc"]``
        at h=0 of each rollout, shape ``(T, 3)`` NaN-padded,
      * per-frame predicted object angular acceleration ``aux["object_ang_acc"]``
        at h=0, shape ``(T, 3)`` NaN-padded,
      * per-frame residual torque ``aux["tau_residual"]`` at h=0,
        shape ``(T, robot_dof)`` NaN-padded,
      * per-frame latent ||z||, shape ``(T,)`` NaN-padded.
    """
    robot_dof = layout.robot_dof
    robot_state_dim = layout.robot_state_dim
    T = min(states.shape[0], joint_pos_abs.shape[0])
    obj_pos_start = robot_state_dim
    obj_pos_end = robot_state_dim + 3

    base_t = history_len - 1
    rel_to_abs_offset = (
        joint_pos_abs[base_t] - states[base_t, :robot_dof]
    ).astype(np.float32)

    pred_gripper_per_t: list[np.ndarray | None] = [None] * T
    pred_cube_per_t: list[np.ndarray | None] = [None] * T
    obj_lin_acc_per_t = np.full((T, 3), np.nan, dtype=np.float32)
    obj_ang_acc_per_t = np.full((T, 3), np.nan, dtype=np.float32)
    residual_torque_per_t = np.full((T, robot_dof), np.nan, dtype=np.float32)
    z_norm_per_t = np.full((T,), np.nan, dtype=np.float32)

    with torch.inference_mode():
        for t in range(base_t, T - 1):
            n_steps = min(pred_horizon, T - 1 - t)
            if n_steps <= 0:
                continue
            hist_lo = t - history_len + 1
            history_states = torch.from_numpy(
                states[hist_lo : t + 1][None, ...]
            ).to(device)
            history_torques = torch.from_numpy(
                torques[hist_lo : t + 1][None, ...]
            ).to(device)
            try:
                z = model.encode_context(history_states, history_torques)
            except RuntimeError:
                continue
            z_norm_per_t[t] = float(torch.linalg.norm(z[0]).item())

            state = history_states[:, -1]
            pred_q_rel_list: list[np.ndarray] = []
            pred_obj_list: list[np.ndarray] = []
            for h in range(n_steps):
                if not torch.isfinite(state).all():
                    break
                torque_h = torch.from_numpy(
                    torques[t + h][None, ...]
                ).to(device)
                if not torch.isfinite(torque_h).all():
                    break
                try:
                    state, aux_step = model.dynamics(state, torque_h, z)
                except RuntimeError:
                    break
                if not torch.isfinite(state).all():
                    break
                if h == 0:
                    obj_lin_acc_per_t[t] = (
                        aux_step["object_lin_acc"][0].detach().cpu().numpy().astype(np.float32)
                    )
                    obj_ang_acc_per_t[t] = (
                        aux_step["object_ang_acc"][0].detach().cpu().numpy().astype(np.float32)
                    )
                    residual_torque_per_t[t] = (
                        aux_step["tau_residual"][0].detach().cpu().numpy().astype(np.float32)
                    )
                pred_q_rel_list.append(
                    state[0, :robot_dof].detach().cpu().numpy().copy()
                )
                pred_obj_list.append(
                    state[0, obj_pos_start:obj_pos_end].detach().cpu().numpy().copy()
                )

            if not pred_q_rel_list:
                continue
            pred_q_rel = np.stack(pred_q_rel_list, axis=0)
            pred_obj = np.stack(pred_obj_list, axis=0)
            pred_q_abs = pred_q_rel + rel_to_abs_offset[None, :]
            # FK on the predicted (absolute) joints.
            traj = franka_joint_and_gripper_positions(pred_q_abs, tool_z_offset=tool_z_offset)
            pred_gripper_per_t[t] = traj["gripper"].astype(np.float32)
            pred_cube_per_t[t] = pred_obj.astype(np.float32)

    return (
        pred_gripper_per_t,
        pred_cube_per_t,
        obj_lin_acc_per_t,
        obj_ang_acc_per_t,
        residual_torque_per_t,
        z_norm_per_t,
    )


# ----------------------------------------------------------------------------
# Render one episode
# ----------------------------------------------------------------------------

def render_animation(
    args: argparse.Namespace,
    episode: dict,
    pred_gripper_per_t: list[np.ndarray | None] | None = None,
    pred_cube_per_t: list[np.ndarray | None] | None = None,
    obj_lin_acc_per_t: np.ndarray | None = None,
    obj_ang_acc_per_t: np.ndarray | None = None,
    residual_torque_per_t: np.ndarray | None = None,
    z_norm_per_t: np.ndarray | None = None,
) -> str:
    joints_3d = compute_joint_positions(episode["joint_pos"], args.tool_z_offset)
    object_pos = episode["object_pos"]
    object_quat = episode["object_quat"]
    T = episode["T"]
    show_pred = pred_gripper_per_t is not None and pred_cube_per_t is not None
    show_gt_future = bool(getattr(args, "show_gt_future", False))
    gt_horizon = int(getattr(args, "pred_horizon", 10))
    if show_gt_future:
        gt_gripper_per_t, gt_cube_per_t = _precompute_gt_future_trajectories(
            joints_3d, object_pos, T, gt_horizon,
        )
    else:
        gt_gripper_per_t = None
        gt_cube_per_t = None
    show_signal_panel = (
        bool(getattr(args, "show_signal_panel", False))
        and obj_lin_acc_per_t is not None
        and obj_ang_acc_per_t is not None
    )

    if show_signal_panel:
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.18)
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        gs_right = gs[0, 1].subgridspec(3, 1, hspace=0.45)
        ax_acc = fig.add_subplot(gs_right[0])
        ax_res = fig.add_subplot(gs_right[1], sharex=ax_acc)
        ax_z = fig.add_subplot(gs_right[2], sharex=ax_acc)
    else:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax_acc = ax_res = ax_z = None
    ax.view_init(elev=args.elev, azim=args.azim)

    # Fixed axis ranges
    xmin, xmax = float(args.xlim[0]), float(args.xlim[1])
    ymin, ymax = float(args.ylim[0]), float(args.ylim[1])
    zmin, zmax = float(args.zlim[0]), float(args.zlim[1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect(((xmax - xmin), (ymax - ymin), (zmax - zmin)))
    ax.grid(True, alpha=0.2)

    joint_xyz = joints_3d[0]
    joint_scatter = ax.scatter(
        joint_xyz[:, 0], joint_xyz[:, 1], joint_xyz[:, 2],
        s=args.joint_marker_size, c="tab:blue", edgecolors="black",
        linewidths=1.0, depthshade=True, zorder=5,
    )

    link_lines = []
    for i in range(joint_xyz.shape[0] - 1):
        line, = ax.plot(
            [joint_xyz[i, 0], joint_xyz[i + 1, 0]],
            [joint_xyz[i, 1], joint_xyz[i + 1, 1]],
            [joint_xyz[i, 2], joint_xyz[i + 1, 2]],
            color="black", linewidth=args.link_linewidth,
            solid_capstyle="round", zorder=4,
        )
        link_lines.append(line)

    corners0 = cube_corners(object_pos[0], object_quat[0], args.cube_size)
    cube_lines = []
    for edge in _CUBE_EDGES:
        i, j = int(edge[0]), int(edge[1])
        line, = ax.plot(
            [corners0[i, 0], corners0[j, 0]],
            [corners0[i, 1], corners0[j, 1]],
            [corners0[i, 2], corners0[j, 2]],
            color="tab:red", linewidth=2.0, zorder=3,
        )
        cube_lines.append(line)

    if args.show_trails:
        gripper_trail, = ax.plot([], [], [], color="tab:blue",
                                 linewidth=1.0, alpha=0.55, zorder=2)
        cube_trail, = ax.plot([], [], [], color="tab:red",
                              linewidth=1.0, alpha=0.55, zorder=2)
    else:
        gripper_trail = None
        cube_trail = None

    nan_xyz = ([np.nan], [np.nan], [np.nan])
    if show_pred:
        pred_gripper_line, = ax.plot(
            *nan_xyz, color=args.pred_gripper_color, linestyle="--",
            linewidth=args.pred_linewidth, zorder=2.5,
            label=f"Pred gripper (N={args.pred_horizon})",
        )
        pred_cube_line, = ax.plot(
            *nan_xyz, color=args.pred_cube_color, linestyle="--",
            linewidth=args.pred_linewidth, zorder=2.5,
            label=f"Pred cube (N={args.pred_horizon})",
        )
    else:
        pred_gripper_line = None
        pred_cube_line = None
    if show_gt_future:
        gt_gripper_line, = ax.plot(
            *nan_xyz, color=args.gt_future_gripper_color, linestyle="-",
            linewidth=args.gt_future_linewidth, zorder=2.6,
            label=f"GT gripper future (N={gt_horizon})",
        )
        gt_cube_line, = ax.plot(
            *nan_xyz, color=args.gt_future_cube_color, linestyle="-",
            linewidth=args.gt_future_linewidth, zorder=2.6,
            label=f"GT cube future (N={gt_horizon})",
        )
    else:
        gt_gripper_line = None
        gt_cube_line = None
    if show_pred or show_gt_future:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # ----- Signal panel artists ------------------------------------------
    sig_lw = float(getattr(args, "signal_linewidth", 1.5))
    time_axis_sec: np.ndarray | None = None
    acc_lines: list = []
    res_lines: list = []
    z_lines: list = []
    sig_cursor_acc = sig_cursor_res = sig_cursor_z = None
    if show_signal_panel:
        time_axis_sec = np.arange(T, dtype=np.float32) * float(args.dt)
        # Object-acc subplot.
        lin_acc_mag = np.linalg.norm(obj_lin_acc_per_t, axis=1)
        ang_acc_mag = np.linalg.norm(obj_ang_acc_per_t, axis=1)
        ln_lin, = ax_acc.plot([], [], color="tab:blue",
                              linewidth=sig_lw,
                              label=r"$\|\dot v^o\|$ [m/s²]")
        ln_ang, = ax_acc.plot([], [], color="tab:red",
                              linewidth=sig_lw,
                              label=r"$\|\dot \omega^o\|$ [rad/s²]")
        acc_lines = [(ln_lin, lin_acc_mag), (ln_ang, ang_acc_mag)]
        finite = np.concatenate([lin_acc_mag, ang_acc_mag])
        y_top = float(np.nanmax(finite)) * 1.15 if np.any(np.isfinite(finite)) else 1.0
        y_top = max(y_top, 1e-3)
        ax_acc.set_xlim(time_axis_sec[0], time_axis_sec[-1])
        ax_acc.set_ylim(0.0, y_top)
        ax_acc.set_ylabel("magnitude")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_acc.set_title("Predicted object acceleration", fontsize=10)
        plt.setp(ax_acc.get_xticklabels(), visible=False)

        # Residual torque subplot.
        res_mag = np.linalg.norm(residual_torque_per_t, axis=1)
        ln_res, = ax_res.plot([], [], color="tab:purple",
                              linewidth=sig_lw,
                              label=r"$\|\tau_{\mathrm{res}}\|$ [N·m]")
        res_lines = [(ln_res, res_mag)]
        y_top_res = float(np.nanmax(res_mag)) * 1.15 if np.any(np.isfinite(res_mag)) else 1.0
        y_top_res = max(y_top_res, 1e-3)
        ax_res.set_xlim(time_axis_sec[0], time_axis_sec[-1])
        ax_res.set_ylim(0.0, y_top_res)
        ax_res.set_ylabel("[N·m]")
        ax_res.grid(True, alpha=0.3)
        ax_res.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_res.set_title("Residual torque magnitude", fontsize=10)
        plt.setp(ax_res.get_xticklabels(), visible=False)

        # Latent ||z|| subplot.
        ln_z, = ax_z.plot([], [], color="tab:olive",
                          linewidth=sig_lw,
                          label=r"$\|z\|$")
        z_lines = [(ln_z, z_norm_per_t)]
        y_top_z = float(np.nanmax(z_norm_per_t)) * 1.15 if np.any(np.isfinite(z_norm_per_t)) else 1.0
        y_top_z = max(y_top_z, 1e-3)
        ax_z.set_xlim(time_axis_sec[0], time_axis_sec[-1])
        ax_z.set_ylim(0.0, y_top_z)
        ax_z.set_xlabel("episode time [s]")
        ax_z.set_ylabel(r"$\|z\|$")
        ax_z.grid(True, alpha=0.3)
        ax_z.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_z.set_title("Latent context norm", fontsize=10)

        sig_cursor_acc = ax_acc.axvline(time_axis_sec[0], color="black",
                                        linestyle=":", linewidth=1.0, alpha=0.6)
        sig_cursor_res = ax_res.axvline(time_axis_sec[0], color="black",
                                        linestyle=":", linewidth=1.0, alpha=0.6)
        sig_cursor_z = ax_z.axvline(time_axis_sec[0], color="black",
                                    linestyle=":", linewidth=1.0, alpha=0.6)

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=12)
    ax.set_title(
        f"Episode '{episode['name']}' from {os.path.basename(args.dataset_file)}",
        fontsize=11,
    )

    def update(frame: int):
        # Joints
        joint_scatter._offsets3d = (
            joints_3d[frame, :, 0],
            joints_3d[frame, :, 1],
            joints_3d[frame, :, 2],
        )
        # Links
        for i, line in enumerate(link_lines):
            line.set_data_3d(
                [joints_3d[frame, i, 0], joints_3d[frame, i + 1, 0]],
                [joints_3d[frame, i, 1], joints_3d[frame, i + 1, 1]],
                [joints_3d[frame, i, 2], joints_3d[frame, i + 1, 2]],
            )
        # Cube
        corners = cube_corners(object_pos[frame], object_quat[frame], args.cube_size)
        for line, edge in zip(cube_lines, _CUBE_EDGES):
            i, j = int(edge[0]), int(edge[1])
            line.set_data_3d(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
            )
        # Trails
        if gripper_trail is not None:
            gripper_trail.set_data_3d(
                joints_3d[: frame + 1, -1, 0],
                joints_3d[: frame + 1, -1, 1],
                joints_3d[: frame + 1, -1, 2],
            )
            cube_trail.set_data_3d(
                object_pos[: frame + 1, 0],
                object_pos[: frame + 1, 1],
                object_pos[: frame + 1, 2],
            )
        # Predicted future
        if pred_gripper_line is not None:
            g = pred_gripper_per_t[frame] if frame < len(pred_gripper_per_t) else None
            c = pred_cube_per_t[frame] if frame < len(pred_cube_per_t) else None
            if g is not None and c is not None:
                pred_gripper_line.set_data_3d(g[:, 0], g[:, 1], g[:, 2])
                pred_cube_line.set_data_3d(c[:, 0], c[:, 1], c[:, 2])
            else:
                pred_gripper_line.set_data_3d(*nan_xyz)
                pred_cube_line.set_data_3d(*nan_xyz)
        # GT future
        if gt_gripper_line is not None:
            gg = gt_gripper_per_t[frame] if frame < len(gt_gripper_per_t) else None
            gc = gt_cube_per_t[frame] if frame < len(gt_cube_per_t) else None
            if gg is not None and gc is not None:
                gt_gripper_line.set_data_3d(gg[:, 0], gg[:, 1], gg[:, 2])
                gt_cube_line.set_data_3d(gc[:, 0], gc[:, 1], gc[:, 2])
            else:
                gt_gripper_line.set_data_3d(*nan_xyz)
                gt_cube_line.set_data_3d(*nan_xyz)
        # Signal panel
        if show_signal_panel and time_axis_sec is not None:
            up_to = frame + 1
            x_now = time_axis_sec[:up_to]
            for line, y in acc_lines:
                line.set_data(x_now, y[:up_to])
            for line, y in res_lines:
                line.set_data(x_now, y[:up_to])
            for line, y in z_lines:
                line.set_data(x_now, y[:up_to])
            for cur in (sig_cursor_acc, sig_cursor_res, sig_cursor_z):
                if cur is not None:
                    cur.set_xdata([float(time_axis_sec[frame])] * 2)
        time_text.set_text(
            f"t = {frame * args.dt:.2f} s   step {frame + 1}/{T}"
        )
        return ()

    anim = FuncAnimation(
        fig, update, frames=T, interval=1000.0 / max(1, args.fps), blit=False,
    )

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    ext = Path(output).suffix.lower()
    actual_path = output
    if ext in (".mp4", ".mov", ".m4v"):
        try:
            writer = FFMpegWriter(fps=args.fps, bitrate=2400)
            anim.save(output, writer=writer, dpi=args.dpi)
        except Exception as exc:
            gif_path = str(Path(output).with_suffix(".gif"))
            print(f"[WARN] FFMpeg writer failed ({exc}); falling back to GIF at {gif_path}.")
            anim.save(gif_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
            actual_path = gif_path
    else:
        anim.save(output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    return actual_path


# ----------------------------------------------------------------------------
# Per-episode dispatch + main
# ----------------------------------------------------------------------------

def _per_episode_output_path(base_output: str, ep_name: str) -> str:
    p = Path(base_output)
    return str(p.with_name(f"{p.stem}_{ep_name}{p.suffix}"))


def render_single_episode(
    args: argparse.Namespace,
    ep_name_override: str | None,
    output_path: str,
    model,
    layout,
    cfg: dict | None,
    ckpt_path: str | None,
    device: torch.device,
) -> tuple[str, dict, dict | None]:
    args_ep = argparse.Namespace(**vars(args))
    if ep_name_override is not None:
        args_ep.episode_name = ep_name_override
    args_ep.output = output_path

    # Resolve which episode to use for the visualisation.
    ep_name = resolve_episode_name(
        args_ep.dataset_file, args_ep.episode_index, args_ep.episode_name,
    )
    episode = load_episode_for_animation(
        args_ep.dataset_file, ep_name, args_ep.robot_dof,
        args_ep.subtract_env_origin, args_ep.max_frames,
    )

    pred_gripper_per_t = None
    pred_cube_per_t = None
    obj_lin_acc_per_t = None
    obj_ang_acc_per_t = None
    residual_torque_per_t = None
    z_norm_per_t = None
    pred_info: dict | None = None
    if model is not None:
        if args_ep.pred_horizon <= 0:
            raise ValueError("--pred_horizon must be a positive integer.")
        torque_key = args_ep.torque_key or str(cfg["torque_key"])
        states, torques, _phys_ctx, _robot_dyn, _obj_dyn = load_episode_arrays(
            args_ep.dataset_file, ep_name, layout, torque_key,
        )
        absolute_joint_pos = load_episode_robot_joint_positions(
            args_ep.dataset_file, ep_name, layout.robot_dof,
        )
        T_min = min(states.shape[0], torques.shape[0], absolute_joint_pos.shape[0])
        states = states[:T_min]
        torques = torques[:T_min]
        absolute_joint_pos = absolute_joint_pos[:T_min]
        history_len = int(cfg["history_len"])
        (
            pred_gripper_per_t,
            pred_cube_per_t,
            obj_lin_acc_per_t,
            obj_ang_acc_per_t,
            residual_torque_per_t,
            z_norm_per_t,
        ) = precompute_pred_trajectories(
            model=model, layout=layout,
            states=states, torques=torques, joint_pos_abs=absolute_joint_pos,
            pred_horizon=args_ep.pred_horizon, history_len=history_len,
            device=device, tool_z_offset=args_ep.tool_z_offset,
        )
        n_valid = sum(1 for arr in pred_gripper_per_t if arr is not None)
        pred_info = {
            "checkpoint": ckpt_path,
            "history_len": history_len,
            "pred_horizon": args_ep.pred_horizon,
            "context_encoder": str(cfg.get("context_encoder", "?")),
            "latent_dim": int(cfg.get("latent_dim", -1)),
            "first_pred_frame": history_len - 1,
            "n_valid_pred_frames": n_valid,
            "n_animation_frames": min(episode["T"], states.shape[0]),
        }

    written = render_animation(
        args_ep, episode,
        pred_gripper_per_t=pred_gripper_per_t,
        pred_cube_per_t=pred_cube_per_t,
        obj_lin_acc_per_t=obj_lin_acc_per_t,
        obj_ang_acc_per_t=obj_ang_acc_per_t,
        residual_torque_per_t=residual_torque_per_t,
        z_norm_per_t=z_norm_per_t,
    )
    episode_meta = {
        "name": episode["name"],
        "T": episode["T"],
        "joint_pos_source": episode["joint_pos_source"],
    }
    return written, episode_meta, pred_info


def main() -> None:
    args = parse_args()

    model = None
    layout = None
    cfg = None
    ckpt_path = None
    device = torch.device(args.device)
    if args.checkpoint is not None:
        model, cfg, layout, ckpt_path = load_checkpoint_model(args.checkpoint, device)

    if args.all_episodes:
        all_names = _episode_names(args.dataset_file)
        # Sort by trailing integer for stable ordering across runs.
        def _key(name: str):
            parts = name.rsplit("_", 1)
            return (int(parts[1]), name) if len(parts) == 2 and parts[1].isdigit() else (10**9, name)
        all_names = sorted(all_names, key=_key)
        if args.max_episodes > 0:
            all_names = all_names[: args.max_episodes]
        print(f"===== Context DeLaN Episode Animation (all_episodes, "
              f"{len(all_names)} episode(s)) =====")
        print(f"dataset_file: {args.dataset_file}")
        print(f"fps: {args.fps}   cube_size: {args.cube_size} m")
        for i, ep_name in enumerate(all_names, start=1):
            output_path = _per_episode_output_path(args.output, ep_name)
            print(f"[{i:>4}/{len(all_names)}] {ep_name} -> {output_path}")
            written, meta, pred_info = render_single_episode(
                args=args, ep_name_override=ep_name, output_path=output_path,
                model=model, layout=layout, cfg=cfg, ckpt_path=ckpt_path,
                device=device,
            )
            print(f"        joint_pos_source: {meta['joint_pos_source']}")
            print(f"        steps: {meta['T']}  duration: {meta['T'] * args.dt:.2f} s")
            if pred_info is not None:
                print(f"        encoder={pred_info['context_encoder']}  "
                      f"latent_dim={pred_info['latent_dim']}  "
                      f"pred_horizon={pred_info['pred_horizon']}  "
                      f"first_pred_frame={pred_info['first_pred_frame']}  "
                      f"n_valid={pred_info['n_valid_pred_frames']}")
            print(f"        saved: {written}")
        return

    written, meta, pred_info = render_single_episode(
        args=args, ep_name_override=None, output_path=args.output,
        model=model, layout=layout, cfg=cfg, ckpt_path=ckpt_path, device=device,
    )
    print("===== Context DeLaN Episode Animation =====")
    print(f"dataset_file: {args.dataset_file}")
    print(f"episode: {meta['name']}")
    print(f"joint_pos_source: {meta['joint_pos_source']}")
    print(f"steps: {meta['T']}  duration: {meta['T'] * args.dt:.2f} s")
    if pred_info is not None:
        print("---- Predicted-trajectory overlay ----")
        for key, value in pred_info.items():
            print(f"  {key}: {value}")
        print("--------------------------------------")
    print(f"Saved: {written}")


if __name__ == "__main__":
    main()
