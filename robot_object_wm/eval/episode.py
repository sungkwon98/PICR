from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch

from robot_object_wm.data.dataset import h5_open
from robot_object_wm.models.utils import FrankaForwardKinematics


@dataclass
class EpisodeData:
    name: str
    state: np.ndarray
    torque: np.ndarray
    object_context: np.ndarray
    joint_pos_abs: np.ndarray
    joint_pos_source: str
    object_pos: np.ndarray
    object_quat: np.ndarray
    dt: float

    @property
    def T(self) -> int:
        return int(self.state.shape[0])


@dataclass
class EpisodeRollout:
    pred_states: np.ndarray
    gt_states: np.ndarray
    start_t: int
    failed_step: int | None = None
    failure_reason: str | None = None

    @property
    def steps(self) -> int:
        return int(self.pred_states.shape[0])


def list_episode_names(dataset_file: str) -> list[str]:
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with h5_open(dataset_file) as file:
        names = list(file["data"].keys())
    names.sort(key=_episode_sort_key)
    return names


def resolve_episode_name(dataset_file: str, episode_index: int = 0, episode_name: str | None = None) -> str:
    names = list_episode_names(dataset_file)
    if episode_name is not None:
        if episode_name not in names:
            raise KeyError(f"Episode '{episode_name}' not found in {dataset_file}.")
        return episode_name
    if episode_index < 0 or episode_index >= len(names):
        raise IndexError(f"episode_index {episode_index} out of [0, {len(names)})")
    return names[episode_index]


def load_episode_data(
    dataset_file: str,
    *,
    episode_index: int = 0,
    episode_name: str | None = None,
    robot_dof: int = 9,
    torque_dim: int = 9,
    torque_key: str = "applied_torque",
    subtract_env_origin: bool = True,
    max_frames: int = 0,
    dt: float = 0.02,
) -> EpisodeData:
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    resolved_name = resolve_episode_name(dataset_file, episode_index, episode_name)

    with h5_open(dataset_file) as file:
        episode = file["data"][resolved_name]
        obs = episode["obs"]
        object_state_group = episode["states"]["rigid_object"]["object"]
        object_dyn_group = episode["object_dynamics"]

        joint_pos_rel = np.asarray(obs["joint_pos"], dtype=np.float32)[:, :robot_dof]
        joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, :robot_dof]
        torques = np.asarray(episode["robot_torques"][torque_key], dtype=np.float32)[:, :torque_dim]

        try:
            joint_pos_abs = np.asarray(
                episode["states"]["articulation"]["robot"]["joint_position"],
                dtype=np.float32,
            )[:, :robot_dof]
            joint_pos_source = "states/articulation/robot/joint_position"
        except KeyError:
            joint_pos_abs = joint_pos_rel
            joint_pos_source = "obs/joint_pos"

        root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
        root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
        object_pos_w = root_pose[:, :3]
        object_quat = root_pose[:, 3:7]
        object_lin_vel = root_velocity[:, :3]
        object_ang_vel = root_velocity[:, 3:6]

        if subtract_env_origin:
            try:
                origin = np.asarray(
                    episode["initial_state"]["articulation"]["robot"]["root_pose"],
                    dtype=np.float32,
                )[0, :3]
            except KeyError:
                origin = np.zeros(3, dtype=np.float32)
        else:
            origin = np.zeros(3, dtype=np.float32)
        object_pos = object_pos_w - origin[None, :]

        mass = np.asarray(object_dyn_group["mass"], dtype=np.float32).reshape(-1, 1)
        inertia = np.asarray(object_dyn_group["inertia"], dtype=np.float32).reshape(-1, 9)
        material = _material_context(object_dyn_group, mass.shape[0])

    T = min(
        joint_pos_rel.shape[0],
        joint_vel.shape[0],
        torques.shape[0],
        joint_pos_abs.shape[0],
        object_pos.shape[0],
        object_quat.shape[0],
        object_lin_vel.shape[0],
        object_ang_vel.shape[0],
        mass.shape[0],
        inertia.shape[0],
        material.shape[0],
    )
    if max_frames > 0:
        T = min(T, max_frames)

    state = np.concatenate(
        [
            joint_pos_rel[:T],
            joint_vel[:T],
            object_pos[:T],
            object_quat[:T],
            object_lin_vel[:T],
            object_ang_vel[:T],
        ],
        axis=-1,
    ).astype(np.float32)
    object_context = np.concatenate([mass[:T], inertia[:T], material[:T]], axis=-1).astype(np.float32)

    return EpisodeData(
        name=resolved_name,
        state=state,
        torque=torques[:T].astype(np.float32),
        object_context=object_context,
        joint_pos_abs=joint_pos_abs[:T].astype(np.float32),
        joint_pos_source=joint_pos_source,
        object_pos=object_pos[:T].astype(np.float32),
        object_quat=object_quat[:T].astype(np.float32),
        dt=dt,
    )


def rollout_episode(
    model: torch.nn.Module,
    episode: EpisodeData,
    *,
    start_t: int,
    rollout_steps: int,
    history_len: int,
    device: torch.device,
) -> EpisodeRollout:
    if start_t < history_len - 1:
        raise ValueError(f"start_t must be >= history_len - 1 ({history_len - 1}).")
    if start_t >= episode.T - 1:
        raise ValueError("start_t must leave at least one future step.")
    rollout_steps = min(rollout_steps, episode.T - start_t - 1)
    if rollout_steps <= 0:
        raise ValueError(f"No rollout room: start_t={start_t}, T={episode.T}.")

    history_states = torch.from_numpy(episode.state[start_t - history_len + 1 : start_t + 1][None]).to(device)
    history_torques = torch.from_numpy(episode.torque[start_t - history_len + 1 : start_t + 1][None]).to(device)
    future_torques = torch.from_numpy(episode.torque[start_t : start_t + rollout_steps][None]).to(device)
    object_context = torch.from_numpy(episode.object_context[start_t : start_t + 1]).to(device)

    failed_step = None
    failure_reason = None
    with torch.inference_mode():
        try:
            pred = model(
                history_states,
                future_torques,
                object_context,
                history_torques=history_torques if getattr(model, "use_context_encoder", False) else None,
                return_aux=False,
            )
        except RuntimeError as exc:
            pred = history_states.new_empty((1, 0, episode.state.shape[-1]))
            failed_step = 0
            failure_reason = f"model rollout failed: {exc}"

    pred_np = pred.detach().cpu().numpy()[0].astype(np.float32)
    if pred_np.size > 0:
        finite_by_step = np.isfinite(pred_np).reshape(pred_np.shape[0], -1).all(axis=1)
        if not bool(finite_by_step.all()):
            first_bad = int(np.flatnonzero(~finite_by_step)[0])
            pred_np = pred_np[:first_bad]
            failed_step = first_bad
            failure_reason = "non-finite predicted state"

    gt_states = episode.state[start_t + 1 : start_t + 1 + pred_np.shape[0]]
    return EpisodeRollout(
        pred_states=pred_np,
        gt_states=gt_states.astype(np.float32),
        start_t=start_t,
        failed_step=failed_step,
        failure_reason=failure_reason,
    )


def compute_joint_positions(
    joint_pos_abs: np.ndarray,
    fk: FrankaForwardKinematics,
    device: torch.device,
) -> np.ndarray:
    q = torch.from_numpy(np.asarray(joint_pos_abs, dtype=np.float32)).to(device)
    with torch.no_grad():
        _R_ee, p_ee, _R_origins, p_origins = fk._arm_chain(q[:, : fk.NUM_ARM])
        batch = q.shape[0]
        base = torch.zeros(batch, 3, device=device, dtype=q.dtype)
        joints = torch.stack([base] + list(p_origins) + [p_ee], dim=1)
    return joints.detach().cpu().numpy().astype(np.float32)


def compute_target_points(
    joint_pos_abs: np.ndarray,
    target: str,
    fk: FrankaForwardKinematics,
    device: torch.device,
) -> np.ndarray:
    target = target.lower()
    joints = compute_joint_positions(joint_pos_abs, fk, device)
    if target == "gripper":
        return joints[:, -1]
    try:
        idx = int(target)
    except ValueError as exc:
        raise ValueError("target must be 'gripper' or an integer joint index.") from exc
    if idx < 0 or idx >= joints.shape[1]:
        raise ValueError(f"target index must be in [0, {joints.shape[1]}); got {idx}.")
    return joints[:, idx]


def rollout_cartesian(
    rollout: EpisodeRollout,
    episode: EpisodeData,
    *,
    robot_dof: int,
    target: str,
    fk: FrankaForwardKinematics,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if rollout.steps == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return {
            "pred_target": empty,
            "gt_target": empty,
            "pred_object": empty,
            "gt_object": empty,
            "real_episode_target": compute_target_points(episode.joint_pos_abs, target, fk, device),
            "real_episode_object": episode.object_pos,
        }

    start_t = rollout.start_t
    offset = episode.joint_pos_abs[start_t] - episode.state[start_t, :robot_dof]
    pred_q_abs = rollout.pred_states[:, :robot_dof] + offset[None, :]
    gt_q_abs = episode.joint_pos_abs[start_t + 1 : start_t + 1 + rollout.steps]
    obj_start = 2 * robot_dof
    return {
        "pred_target": compute_target_points(pred_q_abs, target, fk, device),
        "gt_target": compute_target_points(gt_q_abs, target, fk, device),
        "pred_object": rollout.pred_states[:, obj_start : obj_start + 3],
        "gt_object": rollout.gt_states[:, obj_start : obj_start + 3],
        "real_episode_target": compute_target_points(episode.joint_pos_abs, target, fk, device),
        "real_episode_object": episode.object_pos,
    }


def rollout_metrics(
    rollout: EpisodeRollout,
    episode: EpisodeData,
    *,
    robot_dof: int,
    target: str,
    fk: FrankaForwardKinematics,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cart = rollout_cartesian(
        rollout,
        episode,
        robot_dof=robot_dof,
        target=target,
        fk=fk,
        device=device,
    )
    if rollout.steps == 0:
        return {"evaluated_rollout_steps": 0}, cart

    obj_start = 2 * robot_dof
    pred = rollout.pred_states
    gt = rollout.gt_states
    target_error = np.linalg.norm(cart["pred_target"] - cart["gt_target"], axis=-1)
    object_error = np.linalg.norm(cart["pred_object"] - cart["gt_object"], axis=-1)
    metrics = {
        "evaluated_rollout_steps": int(rollout.steps),
        "joint_position_mse": float(np.mean((pred[:, :robot_dof] - gt[:, :robot_dof]) ** 2)),
        "joint_velocity_mse": float(np.mean((pred[:, robot_dof:obj_start] - gt[:, robot_dof:obj_start]) ** 2)),
        "object_state_mse": float(np.mean((pred[:, obj_start : obj_start + 13] - gt[:, obj_start : obj_start + 13]) ** 2)),
        "object_position_mse": float(np.mean((cart["pred_object"] - cart["gt_object"]) ** 2)),
        f"{target}_position_rmse_m": float(np.sqrt(np.mean(target_error**2))),
        "object_position_rmse_m": float(np.sqrt(np.mean(object_error**2))),
        f"{target}_position_error_mean_m": float(np.mean(target_error)),
        f"{target}_position_error_std_m": float(np.std(target_error)),
        f"{target}_position_error_max_m": float(np.max(target_error)),
        "object_position_error_mean_m": float(np.mean(object_error)),
        "object_position_error_std_m": float(np.std(object_error)),
        "object_position_error_max_m": float(np.max(object_error)),
    }
    return metrics, cart


def precompute_prediction_trajectories(
    model: torch.nn.Module,
    episode: EpisodeData,
    *,
    history_len: int,
    pred_horizon: int,
    robot_dof: int,
    target: str,
    fk: FrankaForwardKinematics,
    device: torch.device,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    pred_target_per_t: list[np.ndarray | None] = [None] * episode.T
    pred_object_per_t: list[np.ndarray | None] = [None] * episode.T
    base_t = history_len - 1

    for t in range(base_t, episode.T - 1):
        steps = min(pred_horizon, episode.T - t - 1)
        if steps <= 0:
            continue
        rollout = rollout_episode(
            model,
            episode,
            start_t=t,
            rollout_steps=steps,
            history_len=history_len,
            device=device,
        )
        if rollout.steps == 0:
            continue
        cart = rollout_cartesian(
            rollout,
            episode,
            robot_dof=robot_dof,
            target=target,
            fk=fk,
            device=device,
        )
        pred_target_per_t[t] = cart["pred_target"]
        pred_object_per_t[t] = cart["pred_object"]

    return pred_target_per_t, pred_object_per_t


def per_horizon_episode_errors(
    model: torch.nn.Module,
    episodes: list[EpisodeData],
    *,
    history_len: int,
    pred_horizon: int,
    robot_dof: int,
    target: str,
    fk: FrankaForwardKinematics,
    device: torch.device,
) -> dict[str, np.ndarray]:
    target_err_by_h: list[list[float]] = [[] for _ in range(pred_horizon)]
    object_err_by_h: list[list[float]] = [[] for _ in range(pred_horizon)]
    q_mse_by_h: list[list[float]] = [[] for _ in range(pred_horizon)]
    dq_mse_by_h: list[list[float]] = [[] for _ in range(pred_horizon)]

    for episode in episodes:
        for t in range(history_len - 1, episode.T - 1):
            steps = min(pred_horizon, episode.T - t - 1)
            rollout = rollout_episode(
                model,
                episode,
                start_t=t,
                rollout_steps=steps,
                history_len=history_len,
                device=device,
            )
            if rollout.steps == 0:
                continue
            cart = rollout_cartesian(
                rollout,
                episode,
                robot_dof=robot_dof,
                target=target,
                fk=fk,
                device=device,
            )
            for h in range(rollout.steps):
                target_err_by_h[h].append(float(np.linalg.norm(cart["pred_target"][h] - cart["gt_target"][h])))
                object_err_by_h[h].append(float(np.linalg.norm(cart["pred_object"][h] - cart["gt_object"][h])))
                q_mse_by_h[h].append(float(np.mean((rollout.pred_states[h, :robot_dof] - rollout.gt_states[h, :robot_dof]) ** 2)))
                qdot = slice(robot_dof, 2 * robot_dof)
                dq_mse_by_h[h].append(float(np.mean((rollout.pred_states[h, qdot] - rollout.gt_states[h, qdot]) ** 2)))

    target_mean, target_std, count = _mean_std_count(target_err_by_h)
    object_mean, object_std, _ = _mean_std_count(object_err_by_h)
    q_mse, _, _ = _mean_std_count(q_mse_by_h)
    dq_mse, _, _ = _mean_std_count(dq_mse_by_h)
    return {
        "horizon": np.arange(1, pred_horizon + 1, dtype=np.int64),
        "target_position_error_mean_m": target_mean,
        "target_position_error_std_m": target_std,
        "object_position_error_mean_m": object_mean,
        "object_position_error_std_m": object_std,
        "joint_position_mse": q_mse,
        "joint_velocity_mse": dq_mse,
        "count": count,
    }


def load_episodes(
    dataset_file: str,
    *,
    max_episodes: int = 0,
    robot_dof: int = 9,
    torque_dim: int = 9,
    torque_key: str = "applied_torque",
    subtract_env_origin: bool = True,
    dt: float = 0.02,
) -> list[EpisodeData]:
    names = list_episode_names(dataset_file)
    if max_episodes > 0:
        names = names[:max_episodes]
    return [
        load_episode_data(
            dataset_file,
            episode_name=name,
            robot_dof=robot_dof,
            torque_dim=torque_dim,
            torque_key=torque_key,
            subtract_env_origin=subtract_env_origin,
            dt=dt,
        )
        for name in names
    ]


def _episode_sort_key(name: str):
    parts = name.rsplit("_", 1)
    return (int(parts[1]), name) if len(parts) == 2 and parts[1].isdigit() else (10**9, name)


def _material_context(object_dyn_group, t_count: int) -> np.ndarray:
    if "material_properties" not in object_dyn_group:
        return np.zeros((t_count, 3), dtype=np.float32)
    material = np.asarray(object_dyn_group["material_properties"], dtype=np.float32)[:t_count]
    material = material.reshape(material.shape[0], -1)
    if material.shape[1] < 3:
        pad = np.zeros((material.shape[0], 3 - material.shape[1]), dtype=np.float32)
        material = np.concatenate([material, pad], axis=-1)
    return material[:, :3].astype(np.float32)


def _mean_std_count(values_by_h: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values_by_h)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for idx, values in enumerate(values_by_h):
        if values:
            arr = np.asarray(values, dtype=np.float64)
            mean[idx] = arr.mean()
            std[idx] = arr.std()
            count[idx] = arr.size
    return mean, std, count
