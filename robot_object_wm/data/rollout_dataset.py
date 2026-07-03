"""Rollout dataset for the DeLaN + object-MLP world model."""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .contact_labels import first_object_motion_timestep
from .dataset import discover_hdf5_files, h5_open, load_all_episode_refs, load_episode_names, split_episode_refs
from .hdf5_schema import (
    EpisodeRef,
    Hdf5Groups,
    RobotObjectStateLayout,
    parse_privileged_collision_pairs,
    privileged_collision_observation_dim,
)

RobotObjectWMStateLayout = RobotObjectStateLayout


def _decode_hdf5_strings(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_privileged_collision_observation(
    episode,
    *,
    mode: int,
    group_name: str = "privileged_collision",
    pair_names: str | None = None,
    subtract_origin: np.ndarray | None = None,
) -> np.ndarray | None:
    """Load optional privileged collision observation features for one episode."""
    mode = int(mode)
    if mode == 0:
        return None
    selected_pairs = parse_privileged_collision_pairs(pair_names)
    if group_name not in episode:
        raise KeyError(
            f"Episode '{episode.name}' does not contain collision group '{group_name}'. "
            "Use a *_collision_augmented.hdf5 dataset or set privileged_collision_observation=0."
        )
    group = episode[group_name]
    if "pair_names" not in group:
        raise KeyError(f"Collision group '{episode.name}/{group_name}' is missing 'pair_names'.")
    available_pairs = _decode_hdf5_strings(group["pair_names"][()])
    missing_pairs = [name for name in selected_pairs if name not in available_pairs]
    if missing_pairs:
        raise KeyError(
            f"Collision group '{episode.name}/{group_name}' is missing requested pair(s): {missing_pairs}. "
            f"Available pairs: {available_pairs}"
        )
    pair_indices = [available_pairs.index(name) for name in selected_pairs]

    if mode == 1:
        if "collision" not in group:
            raise KeyError(f"Collision group '{episode.name}/{group_name}' is missing 'collision'.")
        values = np.asarray(group["collision"], dtype=np.float32)[:, pair_indices]
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if "signed_distance" not in group:
        raise KeyError(f"Collision group '{episode.name}/{group_name}' is missing 'signed_distance'.")
    signed = np.asarray(group["signed_distance"], dtype=np.float32)[:, pair_indices]
    signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if mode == 2:
        return signed
    if mode != 3:
        raise ValueError("privileged_collision_observation must be 0, 1, 2, or 3.")

    if "nearest_points" not in group:
        raise KeyError(
            f"Collision group '{episode.name}/{group_name}' is missing 'nearest_points'. "
            "Re-run augment_collision_info.py with --nearest_points or choose mode 1/2."
        )
    nearest = np.asarray(group["nearest_points"], dtype=np.float32)[:, pair_indices]
    if subtract_origin is not None:
        nearest = nearest - np.asarray(subtract_origin, dtype=np.float32).reshape(1, 1, 1, 3)
    nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    nearest = nearest.reshape(nearest.shape[0], -1)
    return np.concatenate([signed, nearest], axis=-1).astype(np.float32, copy=False)


class RobotObjectWMRolloutDataset(Dataset):
    """Rollout samples with robot state, object state, torque, and object context."""

    def __init__(
        self,
        episode_refs: List[EpisodeRef],
        history_len: int,
        rollout_horizon: int,
        dt: float,
        torque_key: str = "applied_torque",
        filter_pre_contact: bool = False,
        object_displacement_threshold: float = 0.005,
        object_velocity_threshold: float = 0.02,
        contact_consecutive_steps: int = 3,
        contact_settle_steps: int = 5,
        subtract_env_origin: bool = True,
        layout: RobotObjectWMStateLayout | None = None,
        privileged_collision_observation: int = 0,
        privileged_collision_group: str = "privileged_collision",
        privileged_collision_pairs: str | None = None,
    ) -> None:
        super().__init__()
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be >= 1.")
        self.privileged_collision_observation = int(privileged_collision_observation)
        self.privileged_collision_group = privileged_collision_group
        self.privileged_collision_pairs = parse_privileged_collision_pairs(privileged_collision_pairs)
        expected_collision_dim = privileged_collision_observation_dim(
            self.privileged_collision_observation,
            len(self.privileged_collision_pairs),
        )
        self.history_len = history_len
        self.rollout_horizon = rollout_horizon
        self.dt = dt
        self.torque_key = torque_key
        self.subtract_env_origin = subtract_env_origin
        self.layout = layout or RobotObjectWMStateLayout(privileged_collision_obs_dim=expected_collision_dim)
        if self.layout.privileged_collision_obs_dim != expected_collision_dim:
            raise ValueError(
                "layout.privileged_collision_obs_dim does not match privileged_collision_observation: "
                f"layout={self.layout.privileged_collision_obs_dim}, expected={expected_collision_dim}."
            )

        self.history_states: list[np.ndarray] = []
        self.history_actions: list[np.ndarray] = []
        self.history_torques: list[np.ndarray] = []
        self.future_actions: list[np.ndarray] = []
        self.future_torques: list[np.ndarray] = []
        self.future_states: list[np.ndarray] = []
        self.object_context: list[np.ndarray] = []
        self.contact_t_in_window: list[int] = []
        self.sample_meta: list[tuple[str, str, int, int]] = []

        self.skipped_short_episodes = 0
        self.skipped_missing_torque_episodes = 0
        self.skipped_missing_object_dynamics_episodes = 0
        self.skipped_contact_filtered_windows = 0
        self.episodes_with_contact = 0

        self._build_samples(
            episode_refs=episode_refs,
            filter_pre_contact=filter_pre_contact,
            object_displacement_threshold=object_displacement_threshold,
            object_velocity_threshold=object_velocity_threshold,
            contact_consecutive_steps=contact_consecutive_steps,
            contact_settle_steps=contact_settle_steps,
        )

    def _material_context(self, object_dyn_group, t_count: int) -> np.ndarray:
        if "material_properties" not in object_dyn_group:
            return np.zeros((t_count, self.layout.object_material_dim), dtype=np.float32)
        material = np.asarray(object_dyn_group["material_properties"], dtype=np.float32)[:t_count]
        material = material.reshape(material.shape[0], -1)
        if material.shape[1] < self.layout.object_material_dim:
            pad = np.zeros(
                (material.shape[0], self.layout.object_material_dim - material.shape[1]),
                dtype=np.float32,
            )
            material = np.concatenate([material, pad], axis=-1)
        return material[:, : self.layout.object_material_dim]

    def _env_origin(self, episode) -> np.ndarray:
        try:
            init_group = episode["initial_state"]["articulation"]["robot"]
            if "root_pose" in init_group:
                return np.asarray(init_group["root_pose"], dtype=np.float32)[0, : self.layout.object_pos_dim]
        except KeyError:
            pass
        return np.zeros(self.layout.object_pos_dim, dtype=np.float32)

    def _build_samples(
        self,
        episode_refs: List[EpisodeRef],
        filter_pre_contact: bool,
        object_displacement_threshold: float,
        object_velocity_threshold: float,
        contact_consecutive_steps: int,
        contact_settle_steps: int,
    ) -> None:
        by_file: dict[str, list[str]] = defaultdict(list)
        for path, name in episode_refs:
            by_file[path].append(name)

        k = self.history_len - 1
        h = self.rollout_horizon

        for hdf5_path, names in by_file.items():
            with h5_open(hdf5_path) as file:
                data_group = file[Hdf5Groups.DATA]
                for episode_name in names:
                    episode = data_group[episode_name]
                    obs = episode[Hdf5Groups.OBS]
                    if Hdf5Groups.ROBOT_TORQUES not in episode or self.torque_key not in episode[Hdf5Groups.ROBOT_TORQUES]:
                        self.skipped_missing_torque_episodes += 1
                        continue
                    if Hdf5Groups.OBJECT_DYNAMICS not in episode or Hdf5Groups.STATES not in episode:
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    odg = episode[Hdf5Groups.OBJECT_DYNAMICS]
                    if "mass" not in odg or "inertia" not in odg:
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    object_state_group = episode[Hdf5Groups.STATES]["rigid_object"]["object"]
                    root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
                    root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
                    object_pos_w = root_pose[:, : self.layout.object_pos_dim]
                    object_quat = root_pose[
                        :,
                        self.layout.object_pos_dim : self.layout.object_pos_dim + self.layout.object_quat_dim,
                    ]
                    object_lin_vel = root_velocity[:, : self.layout.object_lin_vel_dim]
                    object_ang_vel = root_velocity[
                        :,
                        self.layout.object_lin_vel_dim : self.layout.object_lin_vel_dim + self.layout.object_ang_vel_dim,
                    ]
                    env_origin = self._env_origin(episode) if self.subtract_env_origin else None
                    object_pos = object_pos_w - env_origin[None, :] if env_origin is not None else object_pos_w
                    collision_obs = load_privileged_collision_observation(
                        episode,
                        mode=self.privileged_collision_observation,
                        group_name=self.privileged_collision_group,
                        pair_names=",".join(self.privileged_collision_pairs),
                        subtract_origin=env_origin,
                    )

                    joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : self.layout.robot_dof]
                    joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : self.layout.robot_dof]
                    actions = np.asarray(episode[Hdf5Groups.ACTIONS], dtype=np.float32)[:, : self.layout.action_dim]
                    torques = np.asarray(episode[Hdf5Groups.ROBOT_TORQUES][self.torque_key], dtype=np.float32)[
                        :, : self.layout.torque_dim
                    ]

                    mass = np.asarray(odg["mass"], dtype=np.float32).reshape(-1, self.layout.object_mass_dim)
                    inertia = np.asarray(odg["inertia"], dtype=np.float32).reshape(-1, self.layout.object_inertia_dim)

                    t_count = min(
                        joint_pos.shape[0],
                        joint_vel.shape[0],
                        actions.shape[0],
                        torques.shape[0],
                        object_pos.shape[0],
                        object_quat.shape[0],
                        object_lin_vel.shape[0],
                        object_ang_vel.shape[0],
                        mass.shape[0],
                        inertia.shape[0],
                    )
                    if collision_obs is not None:
                        t_count = min(t_count, collision_obs.shape[0])
                    if t_count <= k + h:
                        self.skipped_short_episodes += 1
                        continue

                    material = self._material_context(odg, t_count)
                    object_context_series = np.concatenate(
                        [mass[:t_count], inertia[:t_count], material[:t_count]], axis=-1
                    )
                    state_parts = [
                        joint_pos[:t_count],
                        joint_vel[:t_count],
                        object_pos[:t_count],
                        object_quat[:t_count],
                        object_lin_vel[:t_count],
                        object_ang_vel[:t_count],
                    ]
                    if collision_obs is not None:
                        state_parts.append(collision_obs[:t_count])
                    state = np.concatenate(state_parts, axis=-1)
                    actions = actions[:t_count]
                    torques = torques[:t_count]

                    contact_t = first_object_motion_timestep(
                        object_pos=object_pos[:t_count],
                        dt=self.dt,
                        displacement_threshold=object_displacement_threshold,
                        velocity_threshold=object_velocity_threshold,
                        consecutive_steps=contact_consecutive_steps,
                        settle_steps=contact_settle_steps,
                    )
                    if contact_t is not None:
                        self.episodes_with_contact += 1

                    for t in range(k, t_count - h):
                        if filter_pre_contact and contact_t is not None and (t + h) >= contact_t:
                            self.skipped_contact_filtered_windows += 1
                            continue

                        self.history_states.append(state[t - k : t + 1])
                        self.history_actions.append(actions[t - k : t])
                        self.history_torques.append(torques[t - k : t + 1])
                        self.future_actions.append(actions[t : t + h])
                        self.future_torques.append(torques[t : t + h])
                        self.future_states.append(state[t + 1 : t + h + 1])
                        self.object_context.append(object_context_series[t])
                        self.contact_t_in_window.append(self._contact_index(contact_t, t, h))
                        self.sample_meta.append((hdf5_path, episode_name, t - k, t + h))

        if len(self.history_states) == 0:
            raise RuntimeError("No valid robot-object rollout samples were built.")

        self.history_states = np.asarray(self.history_states, dtype=np.float32)
        self.history_actions = np.asarray(self.history_actions, dtype=np.float32)
        self.history_torques = np.asarray(self.history_torques, dtype=np.float32)
        self.future_actions = np.asarray(self.future_actions, dtype=np.float32)
        self.future_torques = np.asarray(self.future_torques, dtype=np.float32)
        self.future_states = np.asarray(self.future_states, dtype=np.float32)
        self.object_context = np.asarray(self.object_context, dtype=np.float32)
        self.contact_t_in_window = np.asarray(self.contact_t_in_window, dtype=np.int64)

    @staticmethod
    def _contact_index(contact_t: int | None, t: int, horizon: int) -> int:
        if contact_t is None:
            return -1
        if contact_t <= t:
            return -2
        if contact_t >= t + horizon + 1:
            return -1
        return int(contact_t - (t + 1))

    def __len__(self) -> int:
        return int(self.history_states.shape[0])

    def __getitem__(self, idx: int):
        return {
            "history_states": torch.from_numpy(self.history_states[idx]),
            "history_actions": torch.from_numpy(self.history_actions[idx]),
            "history_torques": torch.from_numpy(self.history_torques[idx]),
            "future_actions": torch.from_numpy(self.future_actions[idx]),
            "future_torques": torch.from_numpy(self.future_torques[idx]),
            "future_states": torch.from_numpy(self.future_states[idx]),
            "object_context": torch.from_numpy(self.object_context[idx]),
            "contact_t_in_window": int(self.contact_t_in_window[idx]),
        }
