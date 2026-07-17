from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from typing import Any

import numpy as np
import torch

from robot_object_wm.data import rollout_dataset
from robot_object_wm.data.dataset import h5_open
from robot_object_wm.data.hdf5_schema import RobotObjectStateLayout
from robot_object_wm.models.hybrid import HybridRigidFormerWMDynamics
from robot_object_wm.models.rwm import RWMEnsemble
from robot_object_wm.models.utils import FrankaForwardKinematics


@dataclass
class EpisodeData:
    name: str
    state: np.ndarray
    action: np.ndarray
    torque: np.ndarray
    object_context: np.ndarray
    joint_pos_abs: np.ndarray
    joint_pos_source: str
    object_pos: np.ndarray
    object_quat: np.ndarray
    state_layout: RobotObjectStateLayout
    source_origin: np.ndarray
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
    pred_pointclouds: np.ndarray | None = None

    @property
    def steps(self) -> int:
        return int(self.pred_states.shape[0])


def _empty_rollout(episode: EpisodeData, *, start_t: int, reason: str) -> EpisodeRollout:
    empty = np.empty((0, episode.state.shape[-1]), dtype=np.float32)
    return EpisodeRollout(
        pred_states=empty,
        gt_states=empty,
        start_t=start_t,
        failed_step=0,
        failure_reason=reason,
    )


@dataclass
class EpisodePointCloudData:
    name: str
    object_points: np.ndarray
    vertex_properties: np.ndarray
    object_point_lens: np.ndarray
    loss_object_mask: np.ndarray
    gripper_part_ids: np.ndarray
    control_dt: float

    @property
    def T(self) -> int:
        return int(self.object_points.shape[0])


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


def _decode_hdf5_strings(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_episode_pointcloud_data(
    pointcloud_file: str,
    episode_name: str,
    *,
    max_points: int,
) -> EpisodePointCloudData:
    if not pointcloud_file or not os.path.isfile(pointcloud_file):
        raise FileNotFoundError(f"Pointcloud file not found: {pointcloud_file}")
    with h5_open(pointcloud_file) as file:
        data = file["data"]
        points_ds = data["object_points"]
        if points_ds.ndim != 5 or points_ds.shape[-1] != 3:
            raise ValueError("/data/object_points must have shape (episodes, steps, objects, points, 3).")
        if "episode_names" in data:
            names = _decode_hdf5_strings(data["episode_names"][()])
        else:
            names = [f"demo_{index}" for index in range(points_ds.shape[0])]
        try:
            episode_index = names.index(episode_name)
        except ValueError as exc:
            raise KeyError(f"Episode '{episode_name}' not found in pointcloud file {pointcloud_file}.") from exc
        point_count = min(int(max_points), int(points_ds.shape[-2]))
        object_points = np.asarray(points_ds[episode_index, :, :, :point_count, :], dtype=np.float32)
        object_points = np.nan_to_num(object_points, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if "vertex_properties" in data:
            props_ds = data["vertex_properties"]
            vertex_properties = np.asarray(props_ds if props_ds.ndim == 2 else props_ds[episode_index], dtype=np.float32)
        else:
            vertex_properties = np.zeros((object_points.shape[1], 3), dtype=np.float32)
            if object_points.shape[1] > 0:
                vertex_properties[0] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
            if object_points.shape[1] > 1:
                vertex_properties[1] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if "loss_object_mask" in data:
            mask_ds = data["loss_object_mask"]
            loss_object_mask = np.asarray(mask_ds if mask_ds.ndim == 1 else mask_ds[episode_index], dtype=bool)
        else:
            loss_object_mask = np.ones((object_points.shape[1],), dtype=bool)
            if object_points.shape[1] > 1:
                loss_object_mask[1:] = False
        object_point_lens = np.full((object_points.shape[1],), point_count, dtype=np.int64)
        if "object_point_counts" in data:
            counts = np.asarray(data["object_point_counts"][episode_index, :, : object_points.shape[1]], dtype=np.int64)
            object_point_lens = np.minimum(object_point_lens, counts.min(axis=0).clip(min=0))
        try:
            config = json.loads(str(file.attrs.get("rigidformer_pointcloud_config", "{}")))
        except json.JSONDecodeError:
            config = {}
        gripper_part_ids = rollout_dataset.infer_gripper_part_ids_from_pointcloud_config(
            config,
            selected_points=point_count,
            stored_points=int(points_ds.shape[-2]),
        )
        control_dt = float(config.get("control_dt", file.attrs.get("control_dt", data.attrs.get("control_dt", 0.02))))
    return EpisodePointCloudData(
        name=episode_name,
        object_points=object_points,
        vertex_properties=vertex_properties[: object_points.shape[1]].astype(np.float32),
        object_point_lens=object_point_lens,
        loss_object_mask=loss_object_mask[: object_points.shape[1]],
        gripper_part_ids=gripper_part_ids,
        control_dt=control_dt,
    )


def _cached_episode_pointcloud_data(
    model: torch.nn.Module,
    pointcloud_file: str,
    episode_name: str,
    *,
    max_points: int,
) -> EpisodePointCloudData:
    key = (os.path.abspath(pointcloud_file), episode_name, int(max_points))
    cache = getattr(model, "_hybrid_pointcloud_episode_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(model, "_hybrid_pointcloud_episode_cache", cache)
    if key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value

    value = load_episode_pointcloud_data(pointcloud_file, episode_name, max_points=max_points)
    cache[key] = value
    while len(cache) > 16:
        cache.popitem(last=False)
    return value


def load_episode_data(
    dataset_file: str,
    *,
    episode_index: int = 0,
    episode_name: str | None = None,
    robot_dof: int = 9,
    action_dim: int = 8,
    torque_dim: int = 9,
    torque_key: str = "applied_torque",
    subtract_env_origin: bool = True,
    max_frames: int = 0,
    dt: float = 0.02,
    state_prediction_mode: str = "full",
    state_layout: RobotObjectStateLayout | None = None,
    privileged_collision_observation: int = 0,
    privileged_collision_group: str = "privileged_collision",
    privileged_collision_pairs: str | None = None,
) -> EpisodeData:
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    resolved_name = resolve_episode_name(dataset_file, episode_index, episode_name)
    if state_layout is None:
        collision_dim = rollout_dataset.privileged_collision_observation_dim(
            int(privileged_collision_observation),
            len(rollout_dataset.parse_privileged_collision_pairs(privileged_collision_pairs)),
        )
        state_layout = rollout_dataset.make_robot_object_state_layout(
            robot_dof=robot_dof,
            action_dim=action_dim,
            torque_dim=torque_dim,
            state_prediction_mode=state_prediction_mode,
            privileged_collision_obs_dim=collision_dim,
        )

    with h5_open(dataset_file) as file:
        episode = file["data"][resolved_name]
        obs = episode["obs"]
        object_state_group = episode["states"]["rigid_object"]["object"]
        object_dyn_group = episode["object_dynamics"]

        joint_pos_rel = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : state_layout.robot_dof]
        joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : int(state_layout.joint_vel_dim or 0)]
        actions = np.asarray(episode["actions"], dtype=np.float32)[:, : state_layout.action_dim]
        torques = np.asarray(episode["robot_torques"][torque_key], dtype=np.float32)[:, : state_layout.torque_dim]

        try:
            joint_pos_abs = np.asarray(
                episode["states"]["articulation"]["robot"]["joint_position"],
                dtype=np.float32,
            )[:, : state_layout.robot_dof]
            joint_pos_source = "states/articulation/robot/joint_position"
        except KeyError:
            joint_pos_abs = joint_pos_rel
            joint_pos_source = "obs/joint_pos"

        root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
        root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
        object_pos_w = root_pose[:, : state_layout.object_pos_dim]
        object_quat = root_pose[
            :,
            state_layout.object_pos_dim : state_layout.object_pos_dim + state_layout.object_quat_dim,
        ]
        object_lin_vel = root_velocity[:, : state_layout.object_lin_vel_dim]
        object_ang_vel = root_velocity[
            :,
            state_layout.object_lin_vel_dim : state_layout.object_lin_vel_dim + state_layout.object_ang_vel_dim,
        ]

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
        collision_obs = rollout_dataset.load_privileged_collision_observation(
            episode,
            mode=privileged_collision_observation,
            group_name=privileged_collision_group,
            pair_names=privileged_collision_pairs,
            subtract_origin=origin if subtract_env_origin else None,
        )

        mass = np.asarray(object_dyn_group["mass"], dtype=np.float32).reshape(-1, 1)
        inertia = np.asarray(object_dyn_group["inertia"], dtype=np.float32).reshape(-1, 9)
        material = _material_context(object_dyn_group, mass.shape[0])

    T = min(
        joint_pos_rel.shape[0],
        joint_vel.shape[0],
        actions.shape[0],
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
    if collision_obs is not None:
        T = min(T, collision_obs.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)

    state_parts = [
        joint_pos_rel[:T],
        joint_vel[:T],
        object_pos[:T],
        object_quat[:T],
        object_lin_vel[:T],
        object_ang_vel[:T],
    ]
    if collision_obs is not None:
        state_parts.append(collision_obs[:T])
    state = np.concatenate(state_parts, axis=-1).astype(np.float32)
    object_context = np.concatenate([mass[:T], inertia[:T], material[:T]], axis=-1).astype(np.float32)

    return EpisodeData(
        name=resolved_name,
        state=state,
        action=actions[:T].astype(np.float32),
        torque=torques[:T].astype(np.float32),
        object_context=object_context,
        joint_pos_abs=joint_pos_abs[:T].astype(np.float32),
        joint_pos_source=joint_pos_source,
        object_pos=object_pos[:T].astype(np.float32),
        object_quat=object_quat[:T].astype(np.float32),
        state_layout=state_layout,
        source_origin=origin.astype(np.float32),
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

    if isinstance(model, HybridRigidFormerWMDynamics) or getattr(model, "model_type", None) == "hybrid":
        return _rollout_hybrid_episode(
            model,
            episode,
            start_t=start_t,
            rollout_steps=rollout_steps,
            history_len=history_len,
            device=device,
        )

    if isinstance(model, RWMEnsemble) or getattr(model, "model_type", None) == "rwm":
        return _rollout_rwm_episode(
            model,
            episode,
            start_t=start_t,
            rollout_steps=rollout_steps,
            history_len=history_len,
            device=device,
        )

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


def _pointcloud_batch_for_rollout(
    pc_episode: EpisodePointCloudData,
    episode: EpisodeData,
    *,
    start_t: int,
    rollout_steps: int,
) -> dict[str, torch.Tensor]:
    if start_t < 1:
        raise ValueError("Hybrid RigidFormer rollout requires start_t >= 1 for history-2 pointcloud input.")
    start = int(start_t) - 1
    stop = int(start_t) + int(rollout_steps) + 1
    if stop > pc_episode.T:
        raise ValueError(f"Pointcloud rollout window [{start}, {stop}) exceeds T={pc_episode.T}.")
    sequence = pc_episode.object_points[start:stop]
    first_object_pos_w = episode.object_pos[0] + episode.source_origin
    first_robot_q_obs = episode.state[0, episode.state_layout.robot_q_slice]
    return {
        "pc_delta_times": torch.tensor([pc_episode.control_dt], dtype=torch.float32),
        "pc_vertex_properties": torch.from_numpy(pc_episode.vertex_properties[None]),
        "pc_object_pos_prev": torch.from_numpy(sequence[None, 0]),
        "pc_object_pos": torch.from_numpy(sequence[None, 1]),
        "pc_object_pos_next": torch.from_numpy(sequence[None, 2]),
        "pc_object_pos_rollout": torch.from_numpy(sequence[None]),
        "pc_object_first_frame_pos": torch.from_numpy(pc_episode.object_points[None, 0]),
        "pc_object_point_lens": torch.from_numpy(pc_episode.object_point_lens[None]),
        "pc_object_lens": torch.tensor([pc_episode.object_points.shape[1]], dtype=torch.long),
        "pc_loss_object_mask": torch.from_numpy(pc_episode.loss_object_mask[None]),
        "pc_gripper_part_ids": torch.from_numpy(pc_episode.gripper_part_ids[None]),
        "pc_first_object_pos_w": torch.from_numpy(first_object_pos_w[None].astype(np.float32)),
        "pc_first_object_quat": torch.from_numpy(episode.object_quat[None, 0].astype(np.float32)),
        "pc_env_origin": torch.from_numpy(episode.source_origin[None].astype(np.float32)),
        "pc_first_robot_q_abs": torch.from_numpy(episode.joint_pos_abs[None, 0].astype(np.float32)),
        "pc_first_robot_q_obs": torch.from_numpy(first_robot_q_obs[None].astype(np.float32)),
    }


def _rollout_hybrid_episode(
    model: torch.nn.Module,
    episode: EpisodeData,
    *,
    start_t: int,
    rollout_steps: int,
    history_len: int,
    device: torch.device,
) -> EpisodeRollout:
    pointcloud_file = getattr(model, "pointcloud_file", None)
    if not pointcloud_file:
        raise ValueError("Hybrid episode rollout requires model.pointcloud_file.")
    pc_episode = _cached_episode_pointcloud_data(
        model,
        pointcloud_file,
        episode.name,
        max_points=int(getattr(model, "rigidformer_max_points", 1024)),
    )
    if start_t < 1:
        return _empty_rollout(
            episode,
            start_t=start_t,
            reason="hybrid pointcloud rollout requires start_t >= 1 for history-2 input",
        )
    rollout_steps = min(rollout_steps, pc_episode.T - start_t - 1)
    if rollout_steps <= 0:
        return _empty_rollout(
            episode,
            start_t=start_t,
            reason=f"no hybrid pointcloud rollout room: start_t={start_t}, pointcloud_T={pc_episode.T}",
        )

    history_states = torch.from_numpy(episode.state[start_t - history_len + 1 : start_t + 1][None]).to(device)
    history_torques = torch.from_numpy(episode.torque[start_t - history_len + 1 : start_t + 1][None]).to(device)
    future_torques = torch.from_numpy(episode.torque[start_t : start_t + rollout_steps][None]).to(device)
    pointcloud_batch = _pointcloud_batch_for_rollout(
        pc_episode,
        episode,
        start_t=start_t,
        rollout_steps=rollout_steps,
    )
    kwargs: dict[str, Any] = {
        "history_torques": history_torques,
        "pointcloud_batch": pointcloud_batch,
        "feedback_mode": getattr(model, "feedback_mode", "robot_native"),
    }
    if str(getattr(model, "action_type", "torque")).strip().lower() == "policy":
        kwargs["history_actions"] = torch.from_numpy(episode.action[start_t - history_len + 1 : start_t][None]).to(device)
        kwargs["future_actions"] = torch.from_numpy(episode.action[start_t : start_t + rollout_steps][None]).to(device)

    failed_step = None
    failure_reason = None
    with torch.inference_mode():
        try:
            pred, aux_steps = model(history_states, future_torques, return_aux=True, **kwargs)
        except RuntimeError as exc:
            pred = history_states.new_empty((1, 0, episode.state.shape[-1]))
            aux_steps = []
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

    pred_pointclouds = None
    if aux_steps and all("rigidformer_points" in aux for aux in aux_steps[: pred_np.shape[0]]):
        pred_pointclouds = torch.stack(
            [aux["rigidformer_points"].detach().cpu()[0] for aux in aux_steps[: pred_np.shape[0]]], dim=0
        ).numpy().astype(np.float32, copy=False)

    gt_states = episode.state[start_t + 1 : start_t + 1 + pred_np.shape[0]]
    return EpisodeRollout(
        pred_states=pred_np,
        gt_states=gt_states.astype(np.float32),
        start_t=start_t,
        failed_step=failed_step,
        failure_reason=failure_reason,
        pred_pointclouds=pred_pointclouds,
    )


def _rwm_episode_action_windows(
    model: torch.nn.Module,
    episode: EpisodeData,
    *,
    start_t: int,
    rollout_steps: int,
    history_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_type = str(getattr(model, "action_type", "torque")).strip().lower()
    if action_type == "policy":
        values = episode.action
    elif action_type == "torque":
        values = episode.torque
    else:
        raise ValueError(f"RWM action_type must be 'policy' or 'torque', got {action_type!r}.")
    history = torch.from_numpy(values[start_t - history_len + 1 : start_t + 1][None]).to(device)
    future = torch.from_numpy(values[start_t : start_t + rollout_steps][None]).to(device)
    return history, future


def _rollout_rwm_episode(
    model: torch.nn.Module,
    episode: EpisodeData,
    *,
    start_t: int,
    rollout_steps: int,
    history_len: int,
    device: torch.device,
) -> EpisodeRollout:
    history_states = torch.from_numpy(episode.state[start_t - history_len + 1 : start_t + 1][None]).to(device)
    history_actions, future_actions = _rwm_episode_action_windows(
        model,
        episode,
        start_t=start_t,
        rollout_steps=rollout_steps,
        history_len=history_len,
        device=device,
    )

    failed_step = None
    failure_reason = None
    pred_steps: list[torch.Tensor] = []
    with torch.inference_mode():
        try:
            if hasattr(model, "reset"):
                model.reset()
            x_state = history_states
            for step in range(rollout_steps):
                x_action = history_actions if step == 0 else future_actions[:, step : step + 1]
                pred_state, _aleatoric, _epistemic = model(x_state, x_action)
                pred_steps.append(pred_state)
                x_state = pred_state.unsqueeze(1)
        except RuntimeError as exc:
            failed_step = len(pred_steps)
            failure_reason = f"model rollout failed: {exc}"
        finally:
            if hasattr(model, "reset"):
                model.reset()

    if pred_steps:
        pred = torch.stack(pred_steps, dim=1)
    else:
        pred = history_states.new_empty((1, 0, episode.state.shape[-1]))
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
    layout = episode.state_layout
    offset = episode.joint_pos_abs[start_t] - episode.state[start_t, layout.robot_q_slice]
    pred_q_abs = rollout.pred_states[:, layout.robot_q_slice] + offset[None, :]
    gt_q_abs = episode.joint_pos_abs[start_t + 1 : start_t + 1 + rollout.steps]
    del robot_dof
    return {
        "pred_target": compute_target_points(pred_q_abs, target, fk, device),
        "gt_target": compute_target_points(gt_q_abs, target, fk, device),
        "pred_object": rollout.pred_states[:, layout.object_pos_slice],
        "gt_object": rollout.gt_states[:, layout.object_pos_slice],
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

    layout = episode.state_layout
    pred = rollout.pred_states
    gt = rollout.gt_states
    target_error = np.linalg.norm(cart["pred_target"] - cart["gt_target"], axis=-1)
    object_error = np.linalg.norm(cart["pred_object"] - cart["gt_object"], axis=-1)
    metrics = {
        "evaluated_rollout_steps": int(rollout.steps),
        "joint_position_mse": float(np.mean((pred[:, layout.robot_q_slice] - gt[:, layout.robot_q_slice]) ** 2)),
        "object_state_mse": float(np.mean((pred[:, layout.object_state_slice] - gt[:, layout.object_state_slice]) ** 2)),
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
    if layout.has_joint_vel:
        metrics["joint_velocity_mse"] = float(
            np.mean((pred[:, layout.robot_dq_slice] - gt[:, layout.robot_dq_slice]) ** 2)
        )
    if layout.object_quat_dim > 0:
        metrics["object_quat_mse"] = float(
            np.mean((pred[:, layout.object_quat_slice] - gt[:, layout.object_quat_slice]) ** 2)
        )
    if layout.has_object_lin_vel:
        metrics["object_lin_vel_mse"] = float(
            np.mean((pred[:, layout.object_lin_vel_slice] - gt[:, layout.object_lin_vel_slice]) ** 2)
        )
    if layout.has_object_ang_vel:
        metrics["object_ang_vel_mse"] = float(
            np.mean((pred[:, layout.object_ang_vel_slice] - gt[:, layout.object_ang_vel_slice]) ** 2)
        )
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
                layout = episode.state_layout
                object_err_by_h[h].append(float(np.linalg.norm(cart["pred_object"][h] - cart["gt_object"][h])))
                q_mse_by_h[h].append(
                    float(
                        np.mean(
                            (
                                rollout.pred_states[h, layout.robot_q_slice]
                                - rollout.gt_states[h, layout.robot_q_slice]
                            )
                            ** 2
                        )
                    )
                )
                if layout.has_joint_vel:
                    dq_mse_by_h[h].append(
                        float(
                            np.mean(
                                (
                                    rollout.pred_states[h, layout.robot_dq_slice]
                                    - rollout.gt_states[h, layout.robot_dq_slice]
                                )
                                ** 2
                            )
                        )
                    )

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
    action_dim: int = 8,
    torque_dim: int = 9,
    torque_key: str = "applied_torque",
    subtract_env_origin: bool = True,
    dt: float = 0.02,
    state_prediction_mode: str = "full",
    state_layout: RobotObjectStateLayout | None = None,
    privileged_collision_observation: int = 0,
    privileged_collision_group: str = "privileged_collision",
    privileged_collision_pairs: str | None = None,
) -> list[EpisodeData]:
    names = list_episode_names(dataset_file)
    if max_episodes > 0:
        names = names[:max_episodes]
    return [
        load_episode_data(
            dataset_file,
            episode_name=name,
            robot_dof=robot_dof,
            action_dim=action_dim,
            torque_dim=torque_dim,
            torque_key=torque_key,
            subtract_env_origin=subtract_env_origin,
            dt=dt,
            state_prediction_mode=state_prediction_mode,
            state_layout=state_layout,
            privileged_collision_observation=privileged_collision_observation,
            privileged_collision_group=privileged_collision_group,
            privileged_collision_pairs=privileged_collision_pairs,
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
