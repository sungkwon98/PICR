from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import torch

from dataset import RobotObjectGTDynamicsStateLayout, first_object_motion_timestep
from models import MultiStepRobotObjectGTDynamicsWorldModel


def resolve_checkpoint_path(path: str | None) -> str:
    if path is not None:
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    candidates = sorted(
        glob.glob(os.path.abspath("./outputs_robot_object_gt_dynamics/run_*/best.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint was provided and no ./outputs_robot_object_gt_dynamics/run_*/best.pt file was found."
        )
    return candidates[0]


def load_checkpoint_model(
    path: str | None, device: torch.device
) -> tuple[MultiStepRobotObjectGTDynamicsWorldModel, dict, RobotObjectGTDynamicsStateLayout, str]:
    path = resolve_checkpoint_path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    layout_dict = ckpt["layout"]
    layout = RobotObjectGTDynamicsStateLayout(
        robot_dof=int(layout_dict["robot_dof"]),
        action_dim=int(layout_dict["action_dim"]),
        torque_dim=int(layout_dict["torque_dim"]),
    )
    model = MultiStepRobotObjectGTDynamicsWorldModel(
        robot_dof=int(cfg["robot_dof"]),
        torque_dim=int(cfg["torque_dim"]),
        object_context_dim=layout.object_context_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        dt=float(cfg["dt"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, layout, path


def h5_open(path: str):
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def episode_names(hdf5_path: str) -> list[str]:
    with h5_open(hdf5_path) as file:
        return list(file["data"].keys())


def resolve_episode_name(hdf5_path: str, episode_index: int, episode_name: str | None) -> str:
    names = episode_names(hdf5_path)
    if episode_name is not None:
        if episode_name not in names:
            raise KeyError(f"Episode {episode_name!r} not found. Available first episodes: {names[:5]}")
        return episode_name
    if not 0 <= episode_index < len(names):
        raise IndexError(f"episode_index must be in [0, {len(names) - 1}], got {episode_index}")
    return names[episode_index]


def _material_context(object_group, t_count: int, material_dim: int) -> np.ndarray:
    if "material_properties" not in object_group:
        return np.zeros((t_count, material_dim), dtype=np.float32)
    material = np.asarray(object_group["material_properties"], dtype=np.float32)[:t_count]
    material = material.reshape(material.shape[0], -1)
    if material.shape[1] < material_dim:
        pad = np.zeros((material.shape[0], material_dim - material.shape[1]), dtype=np.float32)
        material = np.concatenate([material, pad], axis=-1)
    return material[:, :material_dim]


def load_episode_arrays(
    hdf5_path: str,
    episode_name: str,
    layout: RobotObjectGTDynamicsStateLayout,
    torque_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with h5_open(hdf5_path) as file:
        episode = file["data"][episode_name]
        obs = episode["obs"]
        object_state_group = episode["states"]["rigid_object"]["object"]
        object_dyn_group = episode["object_dynamics"]
        robot_dyn_group = episode["robot_dynamics"]

        q = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : layout.robot_dof]
        dq = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : layout.robot_dof]
        torques = np.asarray(episode["robot_torques"][torque_key], dtype=np.float32)[:, : layout.torque_dim]
        root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
        root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
        object_pos = root_pose[:, :3]
        object_quat = root_pose[:, 3:7]
        object_lin_vel = root_velocity[:, :3]
        object_ang_vel = root_velocity[:, 3:6]

        robot_dynamics = {
            "qdd": np.asarray(robot_dyn_group["qdd"], dtype=np.float32)[:, : layout.robot_dof],
            "mass_matrix": np.asarray(robot_dyn_group["mass_matrix"], dtype=np.float32)[
                :, : layout.robot_dof, : layout.robot_dof
            ],
            "inertial": np.asarray(robot_dyn_group["inertial"], dtype=np.float32)[:, : layout.robot_dof],
            "coriolis": np.asarray(robot_dyn_group["coriolis"], dtype=np.float32)[:, : layout.robot_dof],
            "gravity": np.asarray(robot_dyn_group["gravity"], dtype=np.float32)[:, : layout.robot_dof],
            "inverse_dynamics_tau": np.asarray(robot_dyn_group["inverse_dynamics_tau"], dtype=np.float32)[
                :, : layout.robot_dof
            ],
        }
        object_dynamics = {
            "root_lin_acc_w": np.asarray(object_dyn_group["root_lin_acc_w"], dtype=np.float32),
            "root_ang_acc_w": np.asarray(object_dyn_group["root_ang_acc_w"], dtype=np.float32),
            "external_force_est_w": np.asarray(object_dyn_group["external_force_est_w"], dtype=np.float32),
            "inertial_force_w": np.asarray(object_dyn_group["inertial_force_w"], dtype=np.float32),
            "gravity_force_w": np.asarray(object_dyn_group["gravity_force_w"], dtype=np.float32),
        }
        mass = np.asarray(object_dyn_group["mass"], dtype=np.float32).reshape(-1, layout.object_mass_dim)
        inertia = np.asarray(object_dyn_group["inertia"], dtype=np.float32).reshape(-1, layout.object_inertia_dim)
        material_raw = (
            np.asarray(object_dyn_group["material_properties"], dtype=np.float32)
            if "material_properties" in object_dyn_group
            else None
        )

    t_count = min(
        q.shape[0],
        dq.shape[0],
        torques.shape[0],
        object_pos.shape[0],
        object_quat.shape[0],
        object_lin_vel.shape[0],
        object_ang_vel.shape[0],
        mass.shape[0],
        inertia.shape[0],
        *(value.shape[0] for value in robot_dynamics.values()),
        *(value.shape[0] for value in object_dynamics.values()),
    )
    if material_raw is None:
        material = np.zeros((t_count, layout.object_material_dim), dtype=np.float32)
    else:
        material = material_raw[:t_count].reshape(t_count, -1)
        if material.shape[1] < layout.object_material_dim:
            pad = np.zeros((t_count, layout.object_material_dim - material.shape[1]), dtype=np.float32)
            material = np.concatenate([material, pad], axis=-1)
        material = material[:, : layout.object_material_dim]
    states = np.concatenate(
        [
            q[:t_count],
            dq[:t_count],
            object_pos[:t_count],
            object_quat[:t_count],
            object_lin_vel[:t_count],
            object_ang_vel[:t_count],
        ],
        axis=-1,
    )
    object_context = np.concatenate([mass[:t_count], inertia[:t_count], material], axis=-1)
    robot_dynamics = {key: value[:t_count] for key, value in robot_dynamics.items()}
    object_dynamics = {key: value[:t_count] for key, value in object_dynamics.items()}
    return states, torques[:t_count], object_context, robot_dynamics, object_dynamics


def load_episode_robot_joint_positions(
    hdf5_path: str,
    episode_name: str,
    robot_dof: int,
) -> np.ndarray:
    """Load absolute simulator joint positions for FK visualizations.

    The policy observation ``obs/joint_pos`` is relative to the default joint
    pose in this task, while the FK helper expects absolute Panda joint angles.
    """
    with h5_open(hdf5_path) as file:
        episode = file["data"][episode_name]
        if "states" not in episode:
            raise KeyError("Episode does not contain states/articulation/robot/joint_position")
        joint_pos = np.asarray(episode["states"]["articulation"]["robot"]["joint_position"], dtype=np.float32)
    return joint_pos[:, :robot_dof]


def rollout_stepwise(
    model: MultiStepRobotObjectGTDynamicsWorldModel,
    initial_state: torch.Tensor,
    future_torques: torch.Tensor,
    object_context: torch.Tensor,
) -> tuple[np.ndarray, list[dict[str, torch.Tensor]], int | None, str | None]:
    state = initial_state
    preds: list[torch.Tensor] = []
    aux_list: list[dict[str, torch.Tensor]] = []
    failed_step: int | None = None
    failure_reason: str | None = None
    with torch.inference_mode():
        for step_idx in range(future_torques.shape[1]):
            if not torch.isfinite(state).all():
                failed_step = step_idx
                failure_reason = "non-finite predicted state before dynamics step"
                break
            try:
                state, aux = model.dynamics(state, future_torques[:, step_idx], object_context)
            except RuntimeError as exc:
                failed_step = step_idx
                failure_reason = str(exc).splitlines()[0]
                break
            if not torch.isfinite(state).all():
                failed_step = step_idx + 1
                failure_reason = "non-finite predicted state after dynamics step"
                break
            preds.append(state.detach().cpu())
            aux_list.append({key: value.detach().cpu() for key, value in aux.items()})

    if not preds:
        raise RuntimeError(f"Model rollout failed before producing any prediction: {failure_reason}")
    return torch.cat(preds, dim=0).numpy(), aux_list, failed_step, failure_reason


def stack_aux(aux_list: list[dict[str, torch.Tensor]], key: str) -> np.ndarray:
    return torch.cat([aux[key] for aux in aux_list], dim=0).numpy()


def transform_from_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_z @ rot_y @ rot_x
    transform[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return transform


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return transform


def franka_gripper_positions(q: np.ndarray, tool_z_offset: float = 0.1034) -> np.ndarray:
    return franka_joint_and_gripper_positions(q, tool_z_offset)["gripper"]


def franka_joint_and_gripper_positions(q: np.ndarray, tool_z_offset: float = 0.1034) -> dict[str, np.ndarray]:
    """Compute Cartesian trajectories for Panda joint origins, base, and gripper tip."""
    fixed_transforms = [
        transform_from_xyz_rpy((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
        transform_from_xyz_rpy((0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
        transform_from_xyz_rpy((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0)),
        transform_from_xyz_rpy((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
        transform_from_xyz_rpy((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
        transform_from_xyz_rpy((0.0, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
        transform_from_xyz_rpy((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ]
    hand_transform = transform_from_xyz_rpy((0.0, 0.0, 0.107), (0.0, 0.0, -np.pi / 4.0))
    tool_transform = transform_from_xyz_rpy((0.0, 0.0, tool_z_offset), (0.0, 0.0, 0.0))
    trajectories: dict[str, list[np.ndarray]] = {f"joint {idx}": [] for idx in range(0, 8)}
    trajectories["gripper"] = []
    for q_row in q:
        transform = np.eye(4, dtype=np.float64)
        trajectories["joint 0"].append(transform[:3, 3].copy())
        for joint_idx in range(7):
            transform = transform @ fixed_transforms[joint_idx]
            trajectories[f"joint {joint_idx + 1}"].append(transform[:3, 3].copy())
            transform = transform @ rot_z(float(q_row[joint_idx]))
        gripper_transform = transform @ hand_transform @ tool_transform
        trajectories["gripper"].append(gripper_transform[:3, 3].copy())
    return {name: np.asarray(values, dtype=np.float32) for name, values in trajectories.items()}


def parse_robot_target_name(target: str) -> str:
    normalized = target.strip().lower()
    if normalized == "gripper":
        return "gripper"
    try:
        idx = int(normalized)
    except ValueError as exc:
        raise ValueError("--target must be one of 0, 1, ..., 7, or gripper") from exc
    if idx < 0 or idx > 7:
        raise ValueError("--target joint index must be in [0, 7], or use --target gripper")
    return f"joint {idx}"


__all__ = [
    "RobotObjectGTDynamicsStateLayout",
    "episode_names",
    "first_object_motion_timestep",
    "franka_gripper_positions",
    "franka_joint_and_gripper_positions",
    "load_checkpoint_model",
    "load_episode_arrays",
    "load_episode_robot_joint_positions",
    "parse_robot_target_name",
    "resolve_episode_name",
    "rollout_stepwise",
    "stack_aux",
]
