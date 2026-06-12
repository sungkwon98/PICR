"""Animate one episode of the robot+object MCGDF dataset.

Produces an MP4 (or GIF as fallback) that shows:

* Eight spherical markers at the world-frame positions of the Franka joints
  -- the base (joint 0), the seven arm joints (1-7), and the gripper tip --
  all computed via the model's own ``FrankaForwardKinematics`` chain so the
  visualisation matches what the model "sees".
* Solid straight lines joining consecutive joints (the link skeleton).
* A small wire-frame cube at the manipulated object's recorded position and
  orientation, drawn in env-local frame to match the rest of the MCGDF
  pipeline.
* Optional gripper and cube trails showing the path travelled so far.

This is a *dataset* visualisation: no checkpoint is loaded, no neural network
is queried.  Forward kinematics is the only piece taken from ``models.py``
because it encodes the URDF chain we want.

Usage:

    python animate_episode.py \\
        --dataset_file ./datasets/<...>.hdf5 \\
        --episode_index 0 --output ./eval_outputs/episode_0.mp4
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

from models import FrankaForwardKinematics, MultiStepRobotObjectMCGDFWorldModel  # noqa: E402


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_file", type=str, 
        default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_heavy_context_10ep.hdf5",
        help="HDF5 dataset with one or more recorded episodes.",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument(
        "--output", type=str,
        default="./eval_outputs/mcgdf_no_GT_episode_animation_heavy.mp4",
        help="Output .mp4 (preferred, needs ffmpeg) or .gif.  When "
             "--all_episodes is set this is used as a template: the episode "
             "name is inserted into the file stem.",
    )
    parser.add_argument(
        "--all_episodes", action="store_true", default=False,
        help="Render one animation per episode in the dataset file.  When "
             "set, --episode_index / --episode_name are ignored and each "
             "episode is saved as <stem>_<episode_name>.<ext> under the "
             "directory of --output.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=20,
        help="Cap the number of episodes rendered when --all_episodes is "
             "set (0 = no cap).  Ignored otherwise.",
    )
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=25,
                        help="Playback frames-per-second of the rendered video.")
    parser.add_argument("--joint_marker_size", type=float, default=240.0,
                        help="Matplotlib scatter ``s`` for the joint spheres.")
    parser.add_argument("--link_linewidth", type=float, default=4.0,
                        help="Width of the link segments connecting joints.")
    parser.add_argument("--cube_size", type=float, default=0.04,
                        help="Edge length of the manipulated-object wire-frame cube (m).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device used to run FrankaForwardKinematics (cpu is plenty).")
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument("--show_trails", action="store_true", default=True,
                        help="Draw the gripper and cube trails up to the current frame.")
    parser.add_argument("--no_trails", dest="show_trails", action="store_false")
    parser.add_argument(
        "--no_subtract_env_origin", dest="subtract_env_origin", action="store_false",
        default=True,
        help="Match the MCGDF training default: subtract env_origin from object position.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=0,
        help="Cap the animation at this many frames (0 = full episode).",
    )
    parser.add_argument(
        "--elev", type=float, default=20.0,
        help="3-D view elevation angle (degrees).",
    )
    parser.add_argument(
        "--azim", type=float, default=-60.0,
        help="3-D view azimuth angle (degrees).",
    )
    parser.add_argument("--dpi", type=int, default=120)
    # Fixed world-frame axis ranges (meters).  Defaults bracket the Franka
    # Lift workspace; pass three explicit numbers to override per-axis.
    parser.add_argument(
        "--xlim", type=float, nargs=2, default=[-0.2, 0.8],
        metavar=("XMIN", "XMAX"),
        help="World-frame x-axis range [m].",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, default=[-0.4, 0.4],
        metavar=("YMIN", "YMAX"),
        help="World-frame y-axis range [m].",
    )
    parser.add_argument(
        "--zlim", type=float, nargs=2, default=[-0.1, 0.8],
        metavar=("ZMIN", "ZMAX"),
        help="World-frame z-axis range [m].",
    )
    # ----- Optional predicted-trajectory overlay.  When --checkpoint is set,
    # the script rolls out the MCGDF model for --pred_horizon steps at every
    # frame ``t >= history_len - 1`` and draws the resulting gripper-tip and
    # cube-center trajectories as dashed lines.  Defaults are tuned so that
    # leaving --checkpoint unset reproduces the original animation exactly.
    parser.add_argument(
        "--checkpoint", type=str, 
        default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/Robot_and_Object/MCGDF/outputs_mcgdf/run_20260608_112215/best.pt",
        help="Optional path to best.pt/last.pt.  When given, the animation "
             "overlays an N-step predicted trajectory at every frame.",
    )
    parser.add_argument(
        "--pred_horizon", type=int, default=10,
        help="Number of predicted future steps to draw at each frame (the "
             "'N' in the spec).  Ignored when --checkpoint is not set.",
    )
    parser.add_argument(
        "--torque_key", type=str, default="applied_torque",
        choices=["applied_torque", "computed_torque"],
        help="Recorded torque field used as the model's per-step input.",
    )
    parser.add_argument(
        "--friction_key", type=str, default="joint_dynamic_friction_coeff",
        choices=["joint_dynamic_friction_coeff", "joint_friction_coeff"],
    )
    parser.add_argument(
        "--deterministic_context", action="store_true", default=True,
        help="When the checkpoint uses --use_context_encoder, sample z = mu.",
    )
    parser.add_argument(
        "--sample_context", dest="deterministic_context", action="store_false",
        help="Sample z ~ q(z|h) instead of using mu.",
    )
    parser.add_argument(
        "--pred_gripper_color", type=str, default="tab:orange",
        help="Color of the predicted gripper-tip trajectory line.",
    )
    parser.add_argument(
        "--pred_cube_color", type=str, default="tab:purple",
        help="Color of the predicted cube-center trajectory line.",
    )
    parser.add_argument(
        "--pred_linewidth", type=float, default=2.0,
        help="Linewidth of the predicted trajectory lines.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------

def _episode_name(file: h5py.File, episode_index: int, episode_name: str | None) -> str:
    names = list(file["data"].keys())

    def _key(name: str):
        parts = name.rsplit("_", 1)
        return (int(parts[1]), name) if len(parts) == 2 and parts[1].isdigit() else (10**9, name)

    names.sort(key=_key)
    if episode_name is not None:
        if episode_name not in file["data"]:
            raise KeyError(f"Episode '{episode_name}' not found in dataset.")
        return episode_name
    if episode_index < 0 or episode_index >= len(names):
        raise IndexError(f"episode_index {episode_index} out of [0, {len(names)})")
    return names[episode_index]


def load_episode(args: argparse.Namespace) -> dict:
    """Load the per-step robot joint angles and object pose for one episode.

    Uses ``states/articulation/robot/joint_position`` (absolute joint angles
    as set by PhysX during simulation) so the Franka kinematic chain renders
    in the correct world configuration.  In the Franka Lift task,
    ``obs/joint_pos`` is the *relative-to-default* observation
    (``mdp.joint_pos_rel``), which is correct for the model but visually
    wrong: feeding it into FK would render a robot at "home plus a tiny
    delta," and the rendered gripper would never touch the recorded cube.
    """
    if not os.path.isfile(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

    with h5py.File(args.dataset_file, "r") as f:
        ep_name = _episode_name(f, args.episode_index, args.episode_name)
        ep = f["data"][ep_name]
        # Prefer the absolute joint position from the ``states`` group.  Fall
        # back to ``obs/joint_pos`` only if it is unavailable, so that older
        # HDF5 files lacking the ``states`` group still produce *some*
        # animation (with the known relative-coordinate caveat).
        try:
            joint_pos_abs = np.asarray(
                ep["states"]["articulation"]["robot"]["joint_position"], dtype=np.float32,
            )[:, : args.robot_dof]
            joint_pos = joint_pos_abs
            joint_pos_source = "states/articulation/robot/joint_position (absolute)"
        except KeyError:
            joint_pos = np.asarray(ep["obs"]["joint_pos"], dtype=np.float32)[:, : args.robot_dof]
            joint_pos_source = "obs/joint_pos (relative; FK may render wrong configuration)"
        states_obj = ep["states"]["rigid_object"]["object"]
        root_pose = np.asarray(states_obj["root_pose"], dtype=np.float32)
        object_pos_w = root_pose[:, :3]
        object_quat = root_pose[:, 3:7]
        if args.subtract_env_origin and "initial_state" in ep:
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

    T = min(joint_pos.shape[0], object_pos.shape[0], object_quat.shape[0])
    if args.max_frames > 0:
        T = min(T, args.max_frames)
    return {
        "name": ep_name,
        "T": T,
        "joint_pos": joint_pos[:T],
        "joint_pos_source": joint_pos_source,
        "object_pos": object_pos[:T],
        "object_quat": object_quat[:T],
    }


# ----------------------------------------------------------------------------
# Forward kinematics → per-frame joint 3-D positions
# ----------------------------------------------------------------------------

def compute_joint_positions(
    joint_pos: np.ndarray, fk: FrankaForwardKinematics, device: torch.device,
) -> np.ndarray:
    """Return joint origins in world frame for every timestep.

    Output shape ``(T, 9, 3)``:
      index 0 = base (the static robot origin),
      indices 1..7 = origins of joints 1..7,
      index 8 = gripper tip.
    """
    q_t = torch.from_numpy(np.asarray(joint_pos, dtype=np.float32)).to(device)
    with torch.no_grad():
        _R_ee, p_ee, _R_origins, p_origins = fk._arm_chain(q_t[:, : fk.NUM_ARM])
        T = q_t.shape[0]
        base = torch.zeros(T, 3, device=device, dtype=q_t.dtype)
        # p_origins is a list of NUM_ARM=7 tensors, each (T, 3).
        joints = torch.stack([base] + list(p_origins) + [p_ee], dim=1)
    return joints.detach().cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------
# Cube wire-frame
# ----------------------------------------------------------------------------

# 12 edges of a unit cube as pairs of corner indices.
_CUBE_EDGES = np.asarray([
    (0, 1), (1, 2), (2, 3), (3, 0),    # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),    # top
    (0, 4), (1, 5), (2, 6), (3, 7),    # vertical edges
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
    """8 cube corners in world frame, shape (8, 3)."""
    half = size / 2.0
    corners_local = _CUBE_CORNERS_LOCAL * half
    R = _quat_to_rotation_matrix_np(quat)
    return (corners_local @ R.T) + center[None, :]


# ----------------------------------------------------------------------------
# Optional: checkpoint loader for the predicted-trajectory overlay
# ----------------------------------------------------------------------------

def load_checkpoint_model(path: str, device: torch.device):
    """Reconstruct ``MultiStepRobotObjectMCGDFWorldModel`` from a checkpoint.

    Mirrors ``plot_multistep_trajectory.load_checkpoint_model`` but kept
    local here so the animation script has no cross-file dependency.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    layout = ckpt.get("layout", {})
    state_dim = int(layout.get("state_dim", 31))
    torque_dim = int(cfg.get("torque_dim", 9))
    history_step_dim = state_dim + torque_dim
    model = MultiStepRobotObjectMCGDFWorldModel(
        robot_dof=int(cfg["robot_dof"]),
        torque_dim=torque_dim,
        object_context_dim=int(cfg["object_context_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        residual_hidden_dim=int(cfg["residual_hidden_dim"]),
        residual_depth=int(cfg["residual_depth"]),
        contact_hidden_dim=int(cfg["contact_hidden_dim"]),
        contact_depth=int(cfg["contact_depth"]),
        dt=float(cfg["dt"]),
        friction_eps=float(cfg["friction_eps"]),
        learn_damping_friction=bool(cfg.get("learn_damping_friction", False)),
        init_damping=float(cfg.get("init_damping", 0.0)),
        init_friction=float(cfg.get("init_friction", 0.0)),
        omit_damping=bool(cfg.get("omit_damping", True)),
        tool_z_offset=float(cfg.get("tool_z_offset", 0.1034)),
        use_context_encoder=bool(cfg.get("use_context_encoder", False)),
        latent_dim=int(cfg.get("latent_dim", 8)),
        context_encoder_hidden_dim=int(cfg.get("context_encoder_hidden_dim", 256)),
        context_encoder_depth=int(cfg.get("context_encoder_depth", 2)),
        history_step_dim=history_step_dim if cfg.get("use_context_encoder", False) else 0,
        history_len=int(cfg.get("history_len", 5)),
        context_target_dim=int(
            ckpt.get("data_meta", {}).get("context_target_dim_train", 0)
        ),
        delan_use_film=bool(cfg.get("delan_use_film", False)),
        delan_film_depth=int(cfg.get("delan_film_depth", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def load_episode_extras(args: argparse.Namespace, episode_name: str) -> dict:
    """Load the per-step inputs the MCGDF model needs (state, torque, ctx, joint params).

    Kept separate from ``load_episode`` so that callers without a checkpoint
    still get the original lightweight animation loader.
    """
    with h5py.File(args.dataset_file, "r") as f:
        ep = f["data"][episode_name]
        obs = ep["obs"]
        states_grp = ep["states"]["rigid_object"]["object"]
        odg = ep["object_dynamics"]
        jp = ep.get("robot_joint_params", None)

        joint_pos_rel = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : args.robot_dof]
        joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : args.robot_dof]
        torque = np.asarray(
            ep["robot_torques"][args.torque_key], dtype=np.float32,
        )[:, : args.robot_dof]
        root_pose = np.asarray(states_grp["root_pose"], dtype=np.float32)
        root_velocity = np.asarray(states_grp["root_velocity"], dtype=np.float32)
        object_pos_w = root_pose[:, :3]
        object_quat = root_pose[:, 3:7]
        object_lin_vel = root_velocity[:, :3]
        object_ang_vel = root_velocity[:, 3:6]

        if args.subtract_env_origin and "initial_state" in ep:
            try:
                origin = np.asarray(
                    ep["initial_state"]["articulation"]["robot"]["root_pose"],
                    dtype=np.float32,
                )[0, :3]
            except KeyError:
                origin = np.zeros(3, dtype=np.float32)
        else:
            origin = np.zeros(3, dtype=np.float32)
        object_pos_local = object_pos_w - origin[None, :]

        mass = np.asarray(odg["mass"], dtype=np.float32).reshape(-1, 1)
        inertia = np.asarray(odg["inertia"], dtype=np.float32).reshape(-1, 9)
        if "material_properties" in odg:
            material = np.asarray(odg["material_properties"], dtype=np.float32).reshape(
                mass.shape[0], -1,
            )[:, :3]
        else:
            material = np.zeros((mass.shape[0], 3), dtype=np.float32)

        if jp is not None and "joint_damping" in jp:
            damping = np.asarray(jp["joint_damping"], dtype=np.float32)[0, : args.robot_dof]
            f_key = args.friction_key if args.friction_key in jp else "joint_friction_coeff"
            friction = np.asarray(jp[f_key], dtype=np.float32)[0, : args.robot_dof]
        else:
            damping = np.zeros(args.robot_dof, dtype=np.float32)
            friction = np.zeros(args.robot_dof, dtype=np.float32)

    T_min = min(
        joint_pos_rel.shape[0], joint_vel.shape[0], torque.shape[0],
        object_pos_local.shape[0], object_quat.shape[0],
        object_lin_vel.shape[0], object_ang_vel.shape[0],
        mass.shape[0], inertia.shape[0], material.shape[0],
    )
    state = np.concatenate(
        [
            joint_pos_rel[:T_min], joint_vel[:T_min],
            object_pos_local[:T_min], object_quat[:T_min],
            object_lin_vel[:T_min], object_ang_vel[:T_min],
        ], axis=-1,
    ).astype(np.float32)
    object_context = np.concatenate(
        [mass[:T_min], inertia[:T_min], material[:T_min]], axis=-1,
    ).astype(np.float32)
    return {
        "T": T_min,
        "state": state,
        "torque": torque[:T_min],
        "object_context": object_context,
        "damping": damping,
        "friction": friction,
    }


def precompute_pred_trajectories(
    model: MultiStepRobotObjectMCGDFWorldModel,
    episode: dict,
    extras: dict,
    history_len: int,
    pred_horizon: int,
    device: torch.device,
    deterministic_context: bool,
    fk: FrankaForwardKinematics,
    robot_dof: int,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """For every frame ``t``, roll the model out for up to ``pred_horizon``
    steps and return the predicted gripper-tip and cube-center positions.

    Element ``t`` of each returned list is ``None`` when prediction is not
    possible at ``t`` (no full history window yet, or no future steps left
    in the episode), and otherwise an ``(n_steps, 3)`` float array of the
    predicted positions for ``t+1, t+2, ..., t+n_steps`` (``n_steps`` may
    be smaller than ``pred_horizon`` near the end of the episode or after
    a divergent rollout step).
    """
    state_arr = extras["state"]
    torque_arr = extras["torque"]
    object_context_arr = extras["object_context"]
    joint_pos_abs = episode["joint_pos"]  # absolute angles for FK
    T = min(state_arr.shape[0], joint_pos_abs.shape[0], episode["T"])

    damping = torch.from_numpy(extras["damping"][None, ...]).to(device)
    friction = torch.from_numpy(extras["friction"][None, ...]).to(device)

    # Constant per-episode offset (HOME_Q on Franka Lift).  Computing it
    # once at the first frame where the model is callable matches the
    # convention used in plot_multistep_trajectory.py.
    base_t = history_len - 1
    rel_to_abs_offset = (
        joint_pos_abs[base_t] - state_arr[base_t, :robot_dof]
    ).astype(np.float32)

    obj_start = 2 * robot_dof
    obj_end = obj_start + 3

    pred_gripper_per_t: list[np.ndarray | None] = [None] * T
    pred_cube_per_t: list[np.ndarray | None] = [None] * T

    with torch.inference_mode():
        for t in range(base_t, T - 1):
            n_steps = min(pred_horizon, T - 1 - t)
            if n_steps <= 0:
                continue

            history_states = torch.from_numpy(
                state_arr[t - history_len + 1 : t + 1][None, ...]
            ).to(device)
            history_torques = torch.from_numpy(
                torque_arr[t - history_len + 1 : t + 1][None, ...]
            ).to(device)
            object_context_t = torch.from_numpy(
                object_context_arr[t][None, ...]
            ).to(device)

            _mu, _logvar, z, _hat_xi = model.encode_context(
                history_states,
                history_torques if model.use_context_encoder else None,
                deterministic_context=deterministic_context,
            )
            state = history_states[:, -1]
            pred_q_rel_list: list[np.ndarray] = []
            pred_obj_list: list[np.ndarray] = []
            for h in range(n_steps):
                if not torch.isfinite(state).all():
                    break
                torque_h = torch.from_numpy(
                    torque_arr[t + h][None, ...]
                ).to(device)
                if not torch.isfinite(torque_h).all():
                    break
                try:
                    state, _aux = model.dynamics(
                        state, torque_h, damping, friction, object_context_t, z=z,
                    )
                except RuntimeError:
                    break
                if not torch.isfinite(state).all():
                    break
                pred_q_rel_list.append(
                    state[0, :robot_dof].detach().cpu().numpy().copy()
                )
                pred_obj_list.append(
                    state[0, obj_start:obj_end].detach().cpu().numpy().copy()
                )

            if not pred_q_rel_list:
                continue

            pred_q_rel = np.stack(pred_q_rel_list, axis=0)
            pred_obj = np.stack(pred_obj_list, axis=0)
            pred_q_abs = pred_q_rel + rel_to_abs_offset[None, :]
            q_torch = torch.from_numpy(pred_q_abs).to(device)
            p_ee, _ = fk(q_torch)
            pred_gripper_per_t[t] = p_ee.detach().cpu().numpy().astype(np.float32)
            pred_cube_per_t[t] = pred_obj.astype(np.float32)

    return pred_gripper_per_t, pred_cube_per_t


# ----------------------------------------------------------------------------
# Build and run the animation
# ----------------------------------------------------------------------------

def build_axis_bounds(
    joints_3d: np.ndarray,
    object_pos: np.ndarray,
    extra_points: list[np.ndarray] | None = None,
    pad: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Cubic bounds so the aspect ratio stays correct across the whole episode."""
    points = [joints_3d.reshape(-1, 3), object_pos]
    if extra_points:
        points.extend(extra_points)
    all_points = np.concatenate(points, axis=0)
    mins = all_points.min(axis=0) - pad
    maxs = all_points.max(axis=0) + pad
    centres = 0.5 * (mins + maxs)
    half = max((maxs - mins).max(), 0.5) / 2.0
    return centres, half


def render_animation(
    args: argparse.Namespace,
    episode: dict,
    pred_gripper_per_t: list[np.ndarray | None] | None = None,
    pred_cube_per_t: list[np.ndarray | None] | None = None,
) -> str:
    """Build the figure, the FuncAnimation, save it, return the actual path written."""
    device = torch.device(args.device)
    fk = FrankaForwardKinematics(
        robot_dof=args.robot_dof, tool_z_offset=args.tool_z_offset,
    ).to(device)
    fk.eval()

    joints_3d = compute_joint_positions(episode["joint_pos"], fk, device)  # (T, 9, 3)
    object_pos = episode["object_pos"]
    object_quat = episode["object_quat"]
    T = episode["T"]
    show_pred = pred_gripper_per_t is not None and pred_cube_per_t is not None

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    extra_points: list[np.ndarray] = []
    if show_pred:
        for arr in pred_gripper_per_t:
            if arr is not None:
                extra_points.append(arr)
        for arr in pred_cube_per_t:
            if arr is not None:
                extra_points.append(arr)
    # Fixed axis ranges from CLI (defaults give the Franka Lift workspace).
    # ``extra_points`` is left unused here so the camera framing stays
    # consistent across episodes when prediction is enabled or disabled.
    del extra_points
    xmin, xmax = float(args.xlim[0]), float(args.xlim[1])
    ymin, ymax = float(args.ylim[0]), float(args.ylim[1])
    zmin, zmax = float(args.zlim[0]), float(args.zlim[1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # Match the visual box aspect to the axis spans so 1 m on x looks like
    # 1 m on y and z (without this, unequal spans get squashed into a cube).
    ax.set_box_aspect(((xmax - xmin), (ymax - ymin), (zmax - zmin)))
    ax.grid(True, alpha=0.2)

    # Joints: spherical markers via 3-D scatter.
    joint_xyz = joints_3d[0]
    joint_scatter = ax.scatter(
        joint_xyz[:, 0], joint_xyz[:, 1], joint_xyz[:, 2],
        s=args.joint_marker_size, c="tab:blue", edgecolors="black",
        linewidths=1.0, depthshade=True, zorder=5,
    )

    # Link segments: 8 solid lines connecting consecutive joints (base -> j1
    # -> ... -> j7 -> gripper).
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

    # Object: wire-frame cube reconstructed every frame from (pos, quat).
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

    # Trails.
    if args.show_trails:
        gripper_trail, = ax.plot([], [], [], color="tab:blue",
                                 linewidth=1.0, alpha=0.55, zorder=2)
        cube_trail, = ax.plot([], [], [], color="tab:red",
                              linewidth=1.0, alpha=0.55, zorder=2)
    else:
        gripper_trail = None
        cube_trail = None

    # Predicted-trajectory lines (drawn only when a checkpoint is supplied).
    # A length-1 NaN segment hides the line until valid data is set.
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
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    else:
        pred_gripper_line = None
        pred_cube_line = None

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=12)
    ax.set_title(
        f"Episode '{episode['name']}' from {os.path.basename(args.dataset_file)}",
        fontsize=11,
    )

    def update(frame: int):
        # Joints (scatter): use the documented private attribute.
        joint_scatter._offsets3d = (
            joints_3d[frame, :, 0],
            joints_3d[frame, :, 1],
            joints_3d[frame, :, 2],
        )
        # Links.
        for i, line in enumerate(link_lines):
            line.set_data_3d(
                [joints_3d[frame, i, 0], joints_3d[frame, i + 1, 0]],
                [joints_3d[frame, i, 1], joints_3d[frame, i + 1, 1]],
                [joints_3d[frame, i, 2], joints_3d[frame, i + 1, 2]],
            )
        # Cube.
        corners = cube_corners(object_pos[frame], object_quat[frame], args.cube_size)
        for line, edge in zip(cube_lines, _CUBE_EDGES):
            i, j = int(edge[0]), int(edge[1])
            line.set_data_3d(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
            )
        # Trails.
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
        # Predicted N-step trajectories: redrawn from scratch every frame so
        # frame ``t`` always shows the rollout produced *at* ``t``, covering
        # ``t+1, ..., t+N``.  Hidden via NaN when no prediction is available
        # at this frame (e.g. before the history window fills).
        if pred_gripper_line is not None:
            g = pred_gripper_per_t[frame] if frame < len(pred_gripper_per_t) else None
            c = pred_cube_per_t[frame] if frame < len(pred_cube_per_t) else None
            if g is not None and c is not None:
                pred_gripper_line.set_data_3d(g[:, 0], g[:, 1], g[:, 2])
                pred_cube_line.set_data_3d(c[:, 0], c[:, 1], c[:, 2])
            else:
                pred_gripper_line.set_data_3d(*nan_xyz)
                pred_cube_line.set_data_3d(*nan_xyz)
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


def list_episode_names(dataset_file: str) -> list[str]:
    """Return every episode name in the HDF5, sorted by trailing integer."""
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with h5py.File(dataset_file, "r") as f:
        names = list(f["data"].keys())

    def _key(name: str):
        parts = name.rsplit("_", 1)
        return (int(parts[1]), name) if len(parts) == 2 and parts[1].isdigit() else (10**9, name)

    names.sort(key=_key)
    return names


def _per_episode_output_path(base_output: str, ep_name: str) -> str:
    """Insert ``_<ep_name>`` into the stem of ``base_output``."""
    p = Path(base_output)
    return str(p.with_name(f"{p.stem}_{ep_name}{p.suffix}"))


def render_single_episode(
    args: argparse.Namespace,
    ep_name_override: str | None,
    output_path: str,
    model: MultiStepRobotObjectMCGDFWorldModel | None,
    ckpt_path: str | None,
    device: torch.device,
    history_len: int,
    fk_pred: FrankaForwardKinematics | None,
    use_context_encoder_flag: bool,
) -> tuple[str, dict, dict | None]:
    """Run the full per-episode flow (load + optional predict + render).

    Returns ``(written_path, episode_meta, pred_info_or_None)``.
    ``ep_name_override`` lets the caller force a specific episode without
    mutating the original ``args``.
    """
    # Shallow-copy so per-iteration overrides don't leak out of the loop.
    args_ep = argparse.Namespace(**vars(args))
    if ep_name_override is not None:
        args_ep.episode_name = ep_name_override
    args_ep.output = output_path

    episode = load_episode(args_ep)

    pred_gripper_per_t: list[np.ndarray | None] | None = None
    pred_cube_per_t: list[np.ndarray | None] | None = None
    pred_info: dict | None = None
    if model is not None:
        if args_ep.pred_horizon <= 0:
            raise ValueError("--pred_horizon must be a positive integer.")
        extras = load_episode_extras(args_ep, episode["name"])
        # Episode length consistency: prefer the shorter of the two loaders.
        T_pred = min(episode["T"], extras["T"])
        pred_gripper_per_t, pred_cube_per_t = precompute_pred_trajectories(
            model=model, episode=episode, extras=extras,
            history_len=history_len, pred_horizon=args_ep.pred_horizon,
            device=device, deterministic_context=args_ep.deterministic_context,
            fk=fk_pred, robot_dof=args_ep.robot_dof,
        )
        n_valid = sum(1 for arr in pred_gripper_per_t if arr is not None)
        pred_info = {
            "checkpoint": ckpt_path,
            "history_len": history_len,
            "pred_horizon": args_ep.pred_horizon,
            "use_context_encoder": use_context_encoder_flag,
            "deterministic_context": bool(args_ep.deterministic_context),
            "first_pred_frame": history_len - 1,
            "n_valid_pred_frames": n_valid,
            "n_animation_frames": T_pred,
        }

    written = render_animation(
        args_ep, episode,
        pred_gripper_per_t=pred_gripper_per_t,
        pred_cube_per_t=pred_cube_per_t,
    )
    episode_meta = {
        "name": episode["name"],
        "T": episode["T"],
        "joint_pos_source": episode["joint_pos_source"],
    }
    return written, episode_meta, pred_info


def main() -> None:
    args = parse_args()

    # Load the checkpoint (and the FK chain used for prediction visualisation)
    # exactly once: when --all_episodes is set we reuse them across every
    # episode in the loop.
    model: MultiStepRobotObjectMCGDFWorldModel | None = None
    ckpt_path: str | None = None
    device = torch.device(args.device)
    history_len = 5
    fk_pred: FrankaForwardKinematics | None = None
    use_context_encoder_flag = False
    if args.checkpoint is not None:
        ckpt_path = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model, cfg = load_checkpoint_model(ckpt_path, device)
        history_len = int(cfg.get("history_len", 5))
        use_context_encoder_flag = bool(cfg.get("use_context_encoder", False))
        if args.pred_horizon <= 0:
            raise ValueError("--pred_horizon must be a positive integer.")
        fk_pred = FrankaForwardKinematics(
            robot_dof=args.robot_dof, tool_z_offset=args.tool_z_offset,
        ).to(device)
        fk_pred.eval()

    if args.all_episodes:
        all_names = list_episode_names(args.dataset_file)
        if args.max_episodes > 0:
            all_names = all_names[: args.max_episodes]
        print(f"===== MCGDF Episode Animation (all_episodes, "
              f"{len(all_names)} episode(s)) =====")
        print(f"dataset_file: {args.dataset_file}")
        print(f"fps: {args.fps}   joint_marker_size: {args.joint_marker_size}   "
              f"link_linewidth: {args.link_linewidth}   cube_size: {args.cube_size} m")
        for i, ep_name in enumerate(all_names, start=1):
            output_path = _per_episode_output_path(args.output, ep_name)
            print(f"[{i:>4}/{len(all_names)}] {ep_name} -> {output_path}")
            written, meta, pred_info = render_single_episode(
                args=args, ep_name_override=ep_name, output_path=output_path,
                model=model, ckpt_path=ckpt_path, device=device,
                history_len=history_len, fk_pred=fk_pred,
                use_context_encoder_flag=use_context_encoder_flag,
            )
            print(f"        joint_pos_source: {meta['joint_pos_source']}")
            print(f"        steps: {meta['T']}  duration: {meta['T'] * args.dt:.2f} s")
            if pred_info is not None:
                print(f"        pred_horizon: {pred_info['pred_horizon']}  "
                      f"first_pred_frame: {pred_info['first_pred_frame']}  "
                      f"n_valid_pred_frames: {pred_info['n_valid_pred_frames']}")
            print(f"        saved: {written}")
        return

    written, meta, pred_info = render_single_episode(
        args=args, ep_name_override=None, output_path=args.output,
        model=model, ckpt_path=ckpt_path, device=device,
        history_len=history_len, fk_pred=fk_pred,
        use_context_encoder_flag=use_context_encoder_flag,
    )
    print("===== MCGDF Episode Animation =====")
    print(f"dataset_file: {args.dataset_file}")
    print(f"episode: {meta['name']}")
    print(f"joint_pos_source: {meta['joint_pos_source']}")
    print(f"steps: {meta['T']}  duration: {meta['T'] * args.dt:.2f} s")
    print(f"fps: {args.fps}   joint_marker_size: {args.joint_marker_size}   "
          f"link_linewidth: {args.link_linewidth}   cube_size: {args.cube_size} m")
    if pred_info is not None:
        print("---- Predicted-trajectory overlay ----")
        for key, value in pred_info.items():
            print(f"  {key}: {value}")
        print("--------------------------------------")
    print(f"Saved: {written}")


if __name__ == "__main__":
    main()
