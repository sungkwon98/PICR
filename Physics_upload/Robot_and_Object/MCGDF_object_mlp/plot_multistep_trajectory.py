"""Open-loop multi-step trajectory evaluation for the robot+object MCGDF model.

Mirrors the structure of
``scripts/world_model/Ver1/plot_multistep_trajectory.py``: load a checkpoint,
pick an episode from an HDF5, roll out the model for ``rollout_steps`` steps
starting at ``start_t``, and produce a 3-D trajectory comparison plus an
error-versus-time figure for the gripper and the cube.

Differences from the Ver1 plotter:

* The MCGDF model has structured state ``(q, qdot, object_pos, object_quat,
  object_lin_vel, object_ang_vel)``; the gripper Cartesian position is
  computed from the predicted ``q`` using the model's own
  ``FrankaForwardKinematics`` so the predicted target line is consistent
  with what the model is forward-integrating.
* The dataset reads object positions in env-local frame (subtracts
  ``initial_state/articulation/robot/root_pose``), matching the convention
  used by the MCGDF dataset class.
* The MCGDF ``MultiStepRobotObjectMCGDFWorldModel`` already performs the
  recursive rollout internally; we just feed it the history window and the
  recorded future torques.

Usage:

    python plot_multistep_trajectory.py \\
        --checkpoint ./outputs_mcgdf/run_XYZ/best.pt \\
        --dataset_file ./datasets/<...>.hdf5 \\
        --episode_index 0 --start_t 50 --rollout_steps 0
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402

# Local imports (this script lives in the same folder).
from models import (  # noqa: E402
    FrankaForwardKinematics,
    MultiStepRobotObjectMCGDFWorldModel,
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-loop multi-step trajectory evaluation for the robot+object MCGDF model."
    )
    parser.add_argument(
        "--checkpoint", type=str, 
        default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/Robot_and_Object/MCGDF/outputs_mcgdf/run_20260605_195456/best.pt",
        help="Path to best.pt/last.pt.  Defaults to the newest "
             "./outputs_mcgdf/**/best.pt under the cwd.",
    )
    parser.add_argument(
        "--dataset_file", type=str, 
        default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_10002ep.hdf5",
        help="HDF5 dataset to draw the evaluation episode from.",
    )
    parser.add_argument("--episode_index", type=int, default=4)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument(
        "--target", type=str, default="gripper",
        help="Robot body to plot: 0,1,...,7 or 'gripper' (default).",
    )
    parser.add_argument("--start_t", type=int, default=25,
                        help="Index at which the prediction begins (>= history_len - 1).")
    parser.add_argument(
        "--rollout_steps", type=int, default=10,
        help="Number of future steps to roll out.  0 means 'rest of episode'.",
    )
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument(
        "--output_name", type=str, default="mcgdf_multistep_trajectory.png",
        help="Output PNG filename inside --output_dir.",
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument(
        "--deterministic_context", action="store_true", default=True,
        help="When the checkpoint uses --use_context_encoder, sample z = mu.",
    )
    parser.add_argument(
        "--sample_context", dest="deterministic_context", action="store_false",
        help="Sample z ~ q(z|h) instead of using mu.",
    )
    parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
    parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
    parser.add_argument("--contact_consecutive_steps", type=int, default=3)
    parser.add_argument("--contact_settle_steps", type=int, default=5)
    parser.add_argument(
        "--torque_key", type=str, default="applied_torque",
        choices=["applied_torque", "computed_torque"],
    )
    parser.add_argument(
        "--friction_key", type=str, default="joint_dynamic_friction_coeff",
        choices=["joint_dynamic_friction_coeff", "joint_friction_coeff"],
    )
    parser.add_argument(
        "--no_subtract_env_origin", dest="subtract_env_origin",
        action="store_false", default=True,
        help="Match the MCGDF training default: subtract env_origin from object position.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------

def resolve_checkpoint_path(path: str | None) -> str:
    if path is not None:
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    candidates = sorted(
        glob.glob(os.path.abspath("./outputs_mcgdf/run_*/best.pt"))
        + glob.glob(os.path.abspath("./outputs_mcgdf/best.pt")),
        key=os.path.getmtime, reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint was provided and no ./outputs_mcgdf/**/best.pt was found."
        )
    return candidates[0]


def load_checkpoint_model(path: str, device: torch.device):
    """Reconstruct MultiStepRobotObjectMCGDFWorldModel from the checkpoint config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    layout = ckpt.get("layout", {})

    # Reconstruct constructor arguments from saved config.  Use defaults for
    # any new flag that older checkpoints don't have.
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
        object_mlp_hidden_dim=int(cfg.get("object_mlp_hidden_dim", 256)),
        object_mlp_depth=int(cfg.get("object_mlp_depth", 3)),
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, layout, path


# ----------------------------------------------------------------------------
# Dataset reading
# ----------------------------------------------------------------------------

def episode_names(hdf5_path: str) -> list[str]:
    with h5py.File(hdf5_path, "r") as file:
        return list(file["data"].keys())


def resolve_episode_name(hdf5_path: str, episode_index: int, episode_name: str | None) -> str:
    names = episode_names(hdf5_path)
    if episode_name is not None:
        if episode_name not in names:
            raise KeyError(f"Episode '{episode_name}' not found.")
        return episode_name
    # Sort by trailing integer like dataset.py does.
    def _key(name: str):
        parts = name.rsplit("_", 1)
        return (int(parts[1]), name) if len(parts) == 2 and parts[1].isdigit() else (10**9, name)
    names.sort(key=_key)
    if episode_index < 0 or episode_index >= len(names):
        raise IndexError(f"episode_index {episode_index} out of [0, {len(names)})")
    return names[episode_index]


def load_episode(
    hdf5_path: str,
    episode_name: str,
    robot_dof: int,
    torque_key: str,
    friction_key: str,
    subtract_env_origin: bool,
) -> dict:
    """Read everything we need for one episode in env-local frame."""
    with h5py.File(hdf5_path, "r") as file:
        ep = file["data"][episode_name]
        obs = ep["obs"]
        states_grp = ep["states"]["rigid_object"]["object"]
        rd = ep["robot_dynamics"]
        odg = ep["object_dynamics"]
        jp = ep.get("robot_joint_params", None)

        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)[:, :robot_dof]
        joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, :robot_dof]
        # Absolute joint angles from ``states/articulation/robot/joint_position``.
        # In the Franka Lift recordings, ``obs/joint_pos`` is the relative-to-default
        # observation (``mdp.joint_pos_rel``); the state vector that the MCGDF
        # model is trained on uses this relative form.  For Cartesian FK we
        # need the absolute angles, hence loading both:
        #   * ``joint_pos`` (relative)        --> state vector for the model
        #   * ``joint_pos_abs`` (absolute)    --> FK inputs for visualisation
        try:
            joint_pos_abs = np.asarray(
                ep["states"]["articulation"]["robot"]["joint_position"], dtype=np.float32,
            )[:, :robot_dof]
        except KeyError:
            # Older HDF5s without the absolute state group: degrade to using
            # the relative obs for FK (the gripper render will be wrong but
            # the rest of the script keeps working).
            joint_pos_abs = joint_pos.copy()
        torque = np.asarray(ep["robot_torques"][torque_key], dtype=np.float32)[:, :robot_dof]
        root_pose = np.asarray(states_grp["root_pose"], dtype=np.float32)
        root_velocity = np.asarray(states_grp["root_velocity"], dtype=np.float32)

        # Object dynamics labels (we use these for plotting GT only).
        object_pos_w = root_pose[:, :3]
        object_quat = root_pose[:, 3:7]
        object_lin_vel = root_velocity[:, :3]
        object_ang_vel = root_velocity[:, 3:6]

        # env_origin from initial_state if available.
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

        mass = np.asarray(odg["mass"], dtype=np.float32).reshape(-1, 1)
        inertia = np.asarray(odg["inertia"], dtype=np.float32).reshape(-1, 9)
        if "material_properties" in odg:
            material = np.asarray(odg["material_properties"], dtype=np.float32).reshape(
                mass.shape[0], -1,
            )[:, :3]
        else:
            material = np.zeros((mass.shape[0], 3), dtype=np.float32)

        if jp is not None and "joint_damping" in jp:
            damping = np.asarray(jp["joint_damping"], dtype=np.float32)[0, :robot_dof]
            f_key = friction_key if friction_key in jp else "joint_friction_coeff"
            friction = np.asarray(jp[f_key], dtype=np.float32)[0, :robot_dof]
        else:
            damping = np.zeros(robot_dof, dtype=np.float32)
            friction = np.zeros(robot_dof, dtype=np.float32)

    T = min(
        joint_pos.shape[0], joint_pos_abs.shape[0], joint_vel.shape[0], torque.shape[0],
        object_pos.shape[0], object_quat.shape[0],
        object_lin_vel.shape[0], object_ang_vel.shape[0],
        mass.shape[0], inertia.shape[0], material.shape[0],
    )
    state = np.concatenate(
        [
            joint_pos[:T], joint_vel[:T],
            object_pos[:T], object_quat[:T], object_lin_vel[:T], object_ang_vel[:T],
        ],
        axis=-1,
    ).astype(np.float32)
    object_context = np.concatenate([mass[:T], inertia[:T], material[:T]], axis=-1).astype(np.float32)
    return {
        "T": T,
        "state": state,
        "joint_pos_abs": joint_pos_abs[:T],  # absolute joint angles for FK visualisation
        "torque": torque[:T],
        "object_context": object_context,
        "damping": damping,
        "friction": friction,
        "object_pos": object_pos[:T],  # env-local for plotting consistency
    }


# ----------------------------------------------------------------------------
# Contact onset proxy + small math helpers
# ----------------------------------------------------------------------------

def estimate_object_velocity(object_pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(object_pos, dtype=np.float32)
    if object_pos.shape[0] <= 1:
        return vel
    vel[0] = (object_pos[1] - object_pos[0]) / dt
    vel[-1] = (object_pos[-1] - object_pos[-2]) / dt
    if object_pos.shape[0] > 2:
        vel[1:-1] = (object_pos[2:] - object_pos[:-2]) / (2.0 * dt)
    return vel.astype(np.float32)


def first_object_motion_timestep(
    object_pos: np.ndarray,
    dt: float,
    displacement_threshold: float,
    velocity_threshold: float,
    consecutive_steps: int,
    settle_steps: int = 5,
) -> int | None:
    if object_pos.shape[0] == 0:
        return None
    baseline_idx = min(max(0, settle_steps), object_pos.shape[0] - 1)
    disp = np.linalg.norm(object_pos - object_pos[baseline_idx : baseline_idx + 1], axis=-1)
    speed = np.linalg.norm(estimate_object_velocity(object_pos, dt), axis=-1)
    moving = (disp > displacement_threshold) | (speed > velocity_threshold)
    moving[:baseline_idx] = False
    if consecutive_steps <= 1:
        idx = np.flatnonzero(moving)
        return int(idx[0]) if idx.size > 0 else None
    run = 0
    for i, flag in enumerate(moving):
        run = run + 1 if bool(flag) else 0
        if run >= consecutive_steps:
            return i - consecutive_steps + 1
    return None


def parse_target_name(target: str) -> str:
    normalized = target.strip().lower()
    if normalized == "gripper":
        return "gripper"
    try:
        idx = int(normalized)
    except ValueError as exc:
        raise ValueError("--target must be one of 0,1,...,7, or 'gripper'") from exc
    if not 0 <= idx <= 7:
        raise ValueError("--target joint index must be in [0, 7] or 'gripper'")
    return f"joint {idx}"


# ----------------------------------------------------------------------------
# Cartesian targets via the model's FK
# ----------------------------------------------------------------------------

def compute_target_points(
    q: np.ndarray, target_name: str, fk: FrankaForwardKinematics, device: torch.device,
) -> np.ndarray:
    """Run the model's FK to get the world-frame Cartesian point of ``target_name``.

    For ``gripper`` we return the gripper-tip position.  For ``joint i`` we
    intercept the FK chain and return the i-th joint origin (joint 0 being the
    base, joint 7 the last revolute joint).
    """
    q_t = torch.from_numpy(np.asarray(q, dtype=np.float32)).to(device)
    if q_t.ndim == 1:
        q_t = q_t.unsqueeze(0)
    with torch.no_grad():
        if target_name == "gripper":
            p_ee, _ = fk(q_t)
            return p_ee.detach().cpu().numpy()
        # Joint origins from the internal arm chain.
        _R_ee, _p_ee, R_origins, p_origins = fk._arm_chain(q_t[:, : fk.NUM_ARM])
        joint_idx = int(target_name.split()[-1])
        if joint_idx == 0:
            base = torch.zeros(q_t.shape[0], 3, device=device, dtype=q_t.dtype)
            return base.detach().cpu().numpy()
        if joint_idx > fk.NUM_ARM:
            raise ValueError(f"joint index {joint_idx} > {fk.NUM_ARM}")
        return p_origins[joint_idx - 1].detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Open-loop rollout using the MCGDF model
# ----------------------------------------------------------------------------

def rollout_open_loop(
    model: MultiStepRobotObjectMCGDFWorldModel,
    episode: dict,
    start_t: int,
    rollout_steps: int,
    history_len: int,
    device: torch.device,
    deterministic_context: bool,
):
    """Step the MCGDF model open-loop, stopping cleanly on divergence.

    Unlike the one-shot ``model(...)`` call, this version uses the inner
    ``model.dynamics`` so that:
      - the linear solve inside the DeLaN step is wrapped in a try/except,
      - every predicted state is checked for non-finite entries,
      - the rollout terminates at the first failing step and returns the
        valid prefix together with a ``failed_step`` / ``failure_reason`` pair
        that the caller can plot and report.

    Returns ``(pred_states [H', state_dim], failed_step, failure_reason)``
    where ``H'`` is the number of steps that completed successfully
    (``failed_step == None`` and ``H' == rollout_steps`` on a clean rollout).
    """
    state_arr = episode["state"]
    torque_arr = episode["torque"]
    object_context_arr = episode["object_context"]

    if start_t < history_len - 1:
        raise ValueError(f"start_t must be >= history_len - 1 ({history_len - 1})")
    if start_t >= state_arr.shape[0] - 1:
        raise ValueError("start_t must leave at least one future step.")

    history_states = torch.from_numpy(
        state_arr[start_t - history_len + 1 : start_t + 1][None, ...]
    ).to(device)
    history_torques = torch.from_numpy(
        torque_arr[start_t - history_len + 1 : start_t + 1][None, ...]
    ).to(device)
    damping = torch.from_numpy(episode["damping"][None, ...]).to(device)
    friction = torch.from_numpy(episode["friction"][None, ...]).to(device)
    # MCGDF context is constant within an episode; use the window-start value.
    object_context_t = torch.from_numpy(object_context_arr[start_t][None, ...]).to(device)

    preds: list[np.ndarray] = []
    failed_step: int | None = None
    failure_reason: str | None = None

    with torch.inference_mode():
        # Encode the latent context once (no-op when the encoder is off).
        _mu, _logvar, z, _hat_xi = model.encode_context(
            history_states,
            history_torques if model.use_context_encoder else None,
            deterministic_context=deterministic_context,
        )
        state = history_states[:, -1]
        for h in range(rollout_steps):
            if not torch.isfinite(state).all():
                failed_step = h
                failure_reason = "non-finite state before model step"
                break
            torque_h = torch.from_numpy(torque_arr[start_t + h][None, ...]).to(device)
            if not torch.isfinite(torque_h).all():
                failed_step = h
                failure_reason = "non-finite recorded torque input"
                break
            try:
                state, _aux = model.dynamics(
                    state, torque_h, damping, friction, object_context_t, z=z,
                )
            except RuntimeError as exc:
                # Catches torch._C._LinAlgError (mass-matrix singular) and any
                # other CUDA/CPU solver failure during a step.
                failed_step = h
                failure_reason = f"dynamics step failed: {exc}"
                break
            if not torch.isfinite(state).all():
                failed_step = h + 1
                failure_reason = "non-finite predicted state"
                break
            preds.append(state.detach().cpu().numpy()[0].copy())

    if not preds:
        raise RuntimeError(
            f"Rollout failed before producing any prediction: {failure_reason}"
        )
    return np.asarray(preds, dtype=np.float32), failed_step, failure_reason


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def set_axes_equal(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins, maxs = points.min(axis=0), points.max(axis=0)
    centres = (mins + maxs) / 2.0
    half = max((maxs - mins).max(), 1.0e-3) / 2.0
    ax.set_xlim(centres[0] - half, centres[0] + half)
    ax.set_ylim(centres[1] - half, centres[1] + half)
    ax.set_zlim(centres[2] - half, centres[2] + half)


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
    all_points = np.concatenate(
        [real_episode_target, pred_target, real_episode_object, pred_object], axis=0,
    )

    fig = plt.figure(figsize=(21, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.1])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(real_episode_target[:, 0], real_episode_target[:, 1], real_episode_target[:, 2],
              color="black", linewidth=1.8, label=f"Real full episode {target_name}")
    ax3d.plot(pred_target[:, 0], pred_target[:, 1], pred_target[:, 2],
              color="tab:red", linestyle="--", linewidth=1.8, label=f"Pred {target_name}")
    ax3d.plot(real_episode_object[:, 0], real_episode_object[:, 1], real_episode_object[:, 2],
              color="tab:blue", linewidth=1.8, label="Real full episode object")
    ax3d.plot(pred_object[:, 0], pred_object[:, 1], pred_object[:, 2],
              color="tab:orange", linestyle="--", linewidth=1.8, label="Pred object")
    ax3d.scatter(*real_episode_target[0], color="black", s=30, marker="o",
                 label=f"Real episode start {target_name}")
    ax3d.scatter(*real_episode_object[0], color="tab:blue", s=30, marker="o",
                 label="Real episode start object")
    ax3d.scatter(*gt_target[0], color="tab:purple", s=35, marker="o",
                 label=f"Real at prediction start {target_name}")
    ax3d.scatter(*gt_object[0], color="tab:cyan", s=35, marker="o",
                 label="Real at prediction start object")
    ax3d.scatter(*pred_target[-1], color="tab:red", s=40, marker="x",
                 label=f"Pred {target_name} end")
    ax3d.scatter(*pred_object[-1], color="tab:orange", s=40, marker="x",
                 label="Pred object end")
    if contact_t is not None and 0 <= contact_t < real_episode_object.shape[0]:
        contact_object = real_episode_object[contact_t]
        ax3d.scatter(*contact_object, color="tab:green", s=80, marker="*",
                     label=f"First contact proxy t={contact_t}")
    ax3d.set_title(f"{target_name.title()}/Object Trajectory ({episode_name}, start_t={start_t})")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    set_axes_equal(ax3d, all_points)

    handles, labels = ax3d.get_legend_handles_labels()
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")
    ax_legend.legend(
        handles, labels, loc="center", frameon=True, fontsize=12,
        markerscale=1.8, handlelength=2.6, borderpad=1.0, labelspacing=0.75,
    )

    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.plot(time, target_error, color="tab:red", label=f"{target_name} error")
    ax_err.plot(time, object_error, color="tab:blue", label="object error")
    if contact_t is not None:
        contact_time = contact_t * dt
        if time[0] <= contact_time <= time[-1]:
            ax_err.axvline(contact_time, color="tab:green", linestyle=":", linewidth=1.8,
                           label="First contact proxy")
        else:
            ax_err.text(
                0.02, 0.95,
                f"contact proxy t={contact_t}\noutside prediction window",
                transform=ax_err.transAxes, va="top", ha="left",
                fontsize=9, color="tab:green",
            )
    ax_err.set_title("Euclidean Position Error")
    ax_err.set_xlabel("episode time [s]")
    ax_err.set_ylabel("Euclidean error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return float(np.sqrt(np.mean(target_error ** 2))), float(np.sqrt(np.mean(object_error ** 2)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, cfg, layout, checkpoint_path = load_checkpoint_model(checkpoint_path, device)
    robot_dof = int(cfg["robot_dof"])
    dt = float(cfg.get("dt", 0.02))
    history_len = int(cfg.get("history_len", 5))

    episode_name = resolve_episode_name(args.dataset_file, args.episode_index, args.episode_name)
    episode = load_episode(
        args.dataset_file, episode_name, robot_dof,
        torque_key=args.torque_key, friction_key=args.friction_key,
        subtract_env_origin=args.subtract_env_origin,
    )
    T = episode["T"]
    start_t = max(history_len - 1, args.start_t)
    rollout_steps = (T - start_t - 1) if args.rollout_steps <= 0 else min(args.rollout_steps, T - start_t - 1)
    if rollout_steps <= 0:
        raise ValueError(f"No rollout room: start_t={start_t}, T={T}.")

    pred_states, failed_step, failure_reason = rollout_open_loop(
        model=model, episode=episode, start_t=start_t,
        rollout_steps=rollout_steps, history_len=history_len,
        device=device, deterministic_context=args.deterministic_context,
    )
    actual_steps = pred_states.shape[0]
    state_gt = episode["state"]
    gt_future = state_gt[start_t + 1 : start_t + actual_steps + 1]

    # Cartesian target via the model's FK (consistent with what the model
    # represents internally).
    fk = FrankaForwardKinematics(robot_dof=robot_dof, tool_z_offset=args.tool_z_offset).to(device)
    fk.eval()
    target_name = parse_target_name(args.target)

    # FK inputs need the *absolute* joint angles.  ``state[:, :robot_dof]`` is
    # the model's relative obs (``mdp.joint_pos_rel``), so we convert by
    # adding the per-frame offset ``joint_pos_abs - state[:, :robot_dof]``
    # (constant across time = home position).
    joint_pos_abs = episode["joint_pos_abs"]
    rel_to_abs_offset = joint_pos_abs[start_t] - state_gt[start_t, :robot_dof]
    real_episode_q_abs = joint_pos_abs
    gt_future_q_abs = joint_pos_abs[start_t + 1 : start_t + actual_steps + 1]
    pred_q_abs = pred_states[:, :robot_dof] + rel_to_abs_offset[None, :]
    real_episode_target = compute_target_points(real_episode_q_abs, target_name, fk, device)
    gt_target = compute_target_points(gt_future_q_abs, target_name, fk, device)
    pred_target = compute_target_points(pred_q_abs, target_name, fk, device)

    # Diagnostic so we can tell whether the rel->abs offset actually changes
    # the FK inputs.  In the Franka Lift task this should be ~HOME_Q
    # ([0, -0.569, 0, -2.81, 0, 3.037, 0.741, 0.04, 0.04]); if it's near zero
    # the dataset is already absolute and the fix is intentionally a no-op.
    print("---- joint-position diagnostic (the FK fix) ----")
    print(f"  joint_pos_abs[start_t]         = {np.array2string(joint_pos_abs[start_t], precision=4)}")
    print(f"  state_gt[start_t, :robot_dof]  = {np.array2string(state_gt[start_t, :robot_dof], precision=4)}")
    print(f"  rel_to_abs_offset (home_q)     = {np.array2string(rel_to_abs_offset, precision=4)}")
    print(f"  |rel_to_abs_offset|_inf        = {float(np.abs(rel_to_abs_offset).max()):.6f}")
    if float(np.abs(rel_to_abs_offset).max()) < 1.0e-3:
        print("  --> obs/joint_pos and states/.../joint_position appear identical "
              "in this dataset; the fix is a no-op for FK rendering.")
    else:
        print("  --> Using absolute joint angles for real/gt/pred FK; the plot "
              "will differ from the obs-only version.")
    print("------------------------------------------------")

    # Object positions: env-local from load_episode for consistency.
    real_episode_object = episode["object_pos"]
    obj_start = 2 * robot_dof
    obj_end = obj_start + 3
    gt_object = gt_future[:, obj_start:obj_end]
    pred_object = pred_states[:, obj_start:obj_end]

    contact_t = first_object_motion_timestep(
        object_pos=real_episode_object, dt=dt,
        displacement_threshold=args.object_displacement_threshold,
        velocity_threshold=args.object_velocity_threshold,
        consecutive_steps=args.contact_consecutive_steps,
        settle_steps=args.contact_settle_steps,
    )

    time = (np.arange(actual_steps, dtype=np.float32) + start_t + 1) * dt
    output_path = os.path.join(args.output_dir, args.output_name)
    target_rmse, object_rmse = plot_trajectory(
        real_episode_target=real_episode_target,
        real_episode_object=real_episode_object,
        gt_target=gt_target, pred_target=pred_target,
        gt_object=gt_object, pred_object=pred_object,
        time=time, contact_t=contact_t, dt=dt,
        output_path=output_path, episode_name=episode_name,
        target_name=target_name, start_t=start_t,
    )

    target_error = np.linalg.norm(pred_target - gt_target, axis=-1)
    object_error = np.linalg.norm(pred_object - gt_object, axis=-1)
    q_mse = float(np.mean((pred_states[:, :robot_dof] - gt_future[:, :robot_dof]) ** 2))
    qdot_mse = float(np.mean(
        (pred_states[:, robot_dof : 2 * robot_dof] - gt_future[:, robot_dof : 2 * robot_dof]) ** 2
    ))
    object_mse = float(np.mean(
        (pred_states[:, obj_start:obj_end] - gt_future[:, obj_start:obj_end]) ** 2
    ))

    print("===== MCGDF (robot+object) Multi-Step Trajectory Evaluation =====")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset_file: {args.dataset_file}")
    print(f"episode: {episode_name}")
    print(f"target: {target_name}")
    print(f"use_context_encoder: {bool(cfg.get('use_context_encoder', False))}")
    print(f"deterministic_context: {args.deterministic_context}")
    print(f"history_len: {history_len}")
    print(f"start_t: {start_t}")
    print(f"first_contact_proxy_timestep: {contact_t}")
    print(f"requested_rollout_steps: {rollout_steps}")
    print(f"evaluated_rollout_steps: {actual_steps}")
    if failed_step is not None:
        print(f"rollout_stopped_at_step: {failed_step}")
        print(f"rollout_stop_reason: {failure_reason}")
    print(f"joint_position_mse: {q_mse:.8f}")
    print(f"joint_velocity_mse: {qdot_mse:.8f}")
    print(f"object_position_mse: {object_mse:.8f}")
    print(f"{target_name.replace(' ', '_')}_position_rmse_m: {target_rmse:.8f}")
    print(f"object_position_rmse_m: {object_rmse:.8f}")
    print(
        "target_position_error_m: "
        f"mean={float(np.mean(target_error)):.8f}, "
        f"std={float(np.std(target_error)):.8f}, "
        f"max={float(np.max(target_error)):.8f}"
    )
    print(
        "object_position_error_m: "
        f"mean={float(np.mean(object_error)):.8f}, "
        f"std={float(np.std(object_error)):.8f}, "
        f"max={float(np.max(object_error)):.8f}"
    )
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
