"""Rollout dataset for the DeLaN + object-MLP world model."""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .contact_labels import first_object_motion_timestep
from .dataset import discover_hdf5_files, h5_open, load_all_episode_refs, load_episode_names, split_episode_refs
from .hdf5_schema import EpisodeRef, Hdf5Groups, RobotObjectStateLayout

RobotObjectWMStateLayout = RobotObjectStateLayout


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
    ) -> None:
        super().__init__()
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be >= 1.")
        self.history_len = history_len
        self.rollout_horizon = rollout_horizon
        self.dt = dt
        self.torque_key = torque_key
        self.subtract_env_origin = subtract_env_origin
        self.layout = layout or RobotObjectWMStateLayout()

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
                    object_pos = object_pos_w - self._env_origin(episode)[None, :] if self.subtract_env_origin else object_pos_w

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
                    if t_count <= k + h:
                        self.skipped_short_episodes += 1
                        continue

                    material = self._material_context(odg, t_count)
                    object_context_series = np.concatenate(
                        [mass[:t_count], inertia[:t_count], material[:t_count]], axis=-1
                    )
                    state = np.concatenate(
                        [
                            joint_pos[:t_count],
                            joint_vel[:t_count],
                            object_pos[:t_count],
                            object_quat[:t_count],
                            object_lin_vel[:t_count],
                            object_ang_vel[:t_count],
                        ],
                        axis=-1,
                    )
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
