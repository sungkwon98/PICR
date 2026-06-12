from __future__ import annotations

import glob
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import List, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

EpisodeRef = Tuple[str, str]


@dataclass
class ContextDeLaNStateLayout:
    """Robot-object state layout used by the Context DeLaN model."""

    robot_dof: int = 9
    action_dim: int = 8
    torque_dim: int = 9
    object_pos_dim: int = 3
    object_quat_dim: int = 4
    object_lin_vel_dim: int = 3
    object_ang_vel_dim: int = 3
    object_material_dim: int = 3
    object_inertia_dim: int = 9
    object_mass_dim: int = 1

    @property
    def robot_state_dim(self) -> int:
        return 2 * self.robot_dof

    @property
    def object_state_dim(self) -> int:
        return self.object_pos_dim + self.object_quat_dim + self.object_lin_vel_dim + self.object_ang_vel_dim

    @property
    def state_dim(self) -> int:
        return self.robot_state_dim + self.object_state_dim

    @property
    def physical_context_dim(self) -> int:
        return self.object_mass_dim + self.object_inertia_dim + self.object_material_dim

    def to_dict(self) -> dict[str, int]:
        return asdict(self) | {
            "robot_state_dim": self.robot_state_dim,
            "object_state_dim": self.object_state_dim,
            "state_dim": self.state_dim,
            "physical_context_dim": self.physical_context_dim,
        }


def _h5_open(path: str):
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def discover_hdf5_files(dataset_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(os.path.abspath(dataset_dir), "*.hdf5")))


def load_episode_names(hdf5_path: str) -> List[str]:
    with _h5_open(hdf5_path) as file:
        return list(file["data"].keys())


def load_all_episode_refs(hdf5_paths: List[str]) -> List[EpisodeRef]:
    refs: List[EpisodeRef] = []
    for path in hdf5_paths:
        for name in load_episode_names(path):
            refs.append((path, name))
    return refs


def split_episode_refs(
    episode_refs: List[EpisodeRef], train_split: float, seed: int
) -> tuple[List[EpisodeRef], List[EpisodeRef]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1).")
    refs = episode_refs.copy()
    random.Random(seed).shuffle(refs)
    n_train = int(len(refs) * train_split)
    train_refs = refs[:n_train]
    val_refs = refs[n_train:]
    if not train_refs or not val_refs:
        raise RuntimeError("Episode split produced an empty train or validation set.")
    return train_refs, val_refs


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


class ContextDeLaNRolloutDataset(Dataset):
    """HDF5 rollout windows for context-conditioned robot-object DeLaN training."""

    robot_dynamics_keys = ("qdd", "mass_matrix", "inertial", "coriolis", "gravity", "inverse_dynamics_tau")
    object_dynamics_keys = (
        "root_lin_acc_w",
        "root_ang_acc_w",
        "external_force_est_w",
        "inertial_force_w",
        "gravity_force_w",
    )

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
        layout: ContextDeLaNStateLayout | None = None,
    ) -> None:
        super().__init__()
        if history_len < 2:
            raise ValueError("history_len must be >= 2.")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be >= 1.")
        self.history_len = history_len
        self.rollout_horizon = rollout_horizon
        self.dt = dt
        self.torque_key = torque_key
        self.layout = layout or ContextDeLaNStateLayout()

        self.history_states: list[np.ndarray] = []
        self.history_torques: list[np.ndarray] = []
        self.future_torques: list[np.ndarray] = []
        self.future_states: list[np.ndarray] = []
        self.physical_context: list[np.ndarray] = []
        self.future_robot_dynamics: dict[str, list[np.ndarray]] = {key: [] for key in self.robot_dynamics_keys}
        self.future_object_dynamics: dict[str, list[np.ndarray]] = {key: [] for key in self.object_dynamics_keys}
        self.sample_meta: list[tuple[str, str, int, int]] = []
        self.skipped_short_episodes = 0
        self.skipped_missing_torque_episodes = 0
        self.skipped_missing_robot_dynamics_episodes = 0
        self.skipped_missing_object_dynamics_episodes = 0
        self.skipped_contact_filtered_windows = 0

        self._build_samples(
            episode_refs=episode_refs,
            filter_pre_contact=filter_pre_contact,
            object_displacement_threshold=object_displacement_threshold,
            object_velocity_threshold=object_velocity_threshold,
            contact_consecutive_steps=contact_consecutive_steps,
            contact_settle_steps=contact_settle_steps,
        )

    def _slice_robot_vector(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float32)[:, : self.layout.robot_dof]

    def _slice_robot_matrix(self, array: np.ndarray) -> np.ndarray:
        matrix = np.asarray(array, dtype=np.float32)
        return matrix[:, : self.layout.robot_dof, : self.layout.robot_dof]

    def _material_context(self, object_group, t_count: int) -> np.ndarray:
        if "material_properties" not in object_group:
            return np.zeros((t_count, self.layout.object_material_dim), dtype=np.float32)
        material = np.asarray(object_group["material_properties"], dtype=np.float32)[:t_count].reshape(t_count, -1)
        if material.shape[1] < self.layout.object_material_dim:
            pad = np.zeros((t_count, self.layout.object_material_dim - material.shape[1]), dtype=np.float32)
            material = np.concatenate([material, pad], axis=-1)
        return material[:, : self.layout.object_material_dim]

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
            with _h5_open(hdf5_path) as file:
                data_group = file["data"]
                for episode_name in names:
                    episode = data_group[episode_name]
                    if "robot_torques" not in episode or self.torque_key not in episode["robot_torques"]:
                        self.skipped_missing_torque_episodes += 1
                        continue
                    if "robot_dynamics" not in episode:
                        self.skipped_missing_robot_dynamics_episodes += 1
                        continue
                    if "object_dynamics" not in episode or "states" not in episode:
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    robot_dyn_group = episode["robot_dynamics"]
                    object_dyn_group = episode["object_dynamics"]
                    if any(key not in robot_dyn_group for key in self.robot_dynamics_keys):
                        self.skipped_missing_robot_dynamics_episodes += 1
                        continue
                    if any(key not in object_dyn_group for key in (*self.object_dynamics_keys, "mass", "inertia")):
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    obs = episode["obs"]
                    object_state_group = episode["states"]["rigid_object"]["object"]
                    root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
                    root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
                    q = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : self.layout.robot_dof]
                    dq = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : self.layout.robot_dof]
                    torques = np.asarray(episode["robot_torques"][self.torque_key], dtype=np.float32)[
                        :, : self.layout.torque_dim
                    ]
                    object_pos = root_pose[:, :3]
                    object_quat = root_pose[:, 3:7]
                    object_lin_vel = root_velocity[:, :3]
                    object_ang_vel = root_velocity[:, 3:6]
                    robot_dynamics = {
                        "qdd": self._slice_robot_vector(robot_dyn_group["qdd"]),
                        "mass_matrix": self._slice_robot_matrix(robot_dyn_group["mass_matrix"]),
                        "inertial": self._slice_robot_vector(robot_dyn_group["inertial"]),
                        "coriolis": self._slice_robot_vector(robot_dyn_group["coriolis"]),
                        "gravity": self._slice_robot_vector(robot_dyn_group["gravity"]),
                        "inverse_dynamics_tau": self._slice_robot_vector(robot_dyn_group["inverse_dynamics_tau"]),
                    }
                    object_dynamics = {
                        key: np.asarray(object_dyn_group[key], dtype=np.float32) for key in self.object_dynamics_keys
                    }
                    mass = np.asarray(object_dyn_group["mass"], dtype=np.float32).reshape(-1, self.layout.object_mass_dim)
                    inertia = np.asarray(object_dyn_group["inertia"], dtype=np.float32).reshape(
                        -1, self.layout.object_inertia_dim
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
                    if t_count <= k + h:
                        self.skipped_short_episodes += 1
                        continue

                    material = self._material_context(object_dyn_group, t_count)
                    state = np.concatenate(
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
                    physical_context = np.concatenate([mass[:t_count], inertia[:t_count], material], axis=-1)
                    robot_dynamics = {key: value[:t_count] for key, value in robot_dynamics.items()}
                    object_dynamics = {key: value[:t_count] for key, value in object_dynamics.items()}
                    torques = torques[:t_count]

                    contact_t = None
                    if filter_pre_contact:
                        contact_t = first_object_motion_timestep(
                            object_pos=object_pos[:t_count],
                            dt=self.dt,
                            displacement_threshold=object_displacement_threshold,
                            velocity_threshold=object_velocity_threshold,
                            consecutive_steps=contact_consecutive_steps,
                            settle_steps=contact_settle_steps,
                        )

                    for t in range(k, t_count - h):
                        if contact_t is not None and (t + h) >= contact_t:
                            self.skipped_contact_filtered_windows += 1
                            continue
                        self.history_states.append(state[t - k : t + 1])
                        self.history_torques.append(torques[t - k : t + 1])
                        self.future_torques.append(torques[t : t + h])
                        self.future_states.append(state[t + 1 : t + h + 1])
                        self.physical_context.append(physical_context[t])
                        for key in self.robot_dynamics_keys:
                            self.future_robot_dynamics[key].append(robot_dynamics[key][t : t + h])
                        for key in self.object_dynamics_keys:
                            self.future_object_dynamics[key].append(object_dynamics[key][t : t + h])
                        self.sample_meta.append((hdf5_path, episode_name, t - k, t + h))

        if len(self.history_states) == 0:
            raise RuntimeError("No valid Context DeLaN rollout samples were built.")

        self.history_states = np.asarray(self.history_states, dtype=np.float32)
        self.history_torques = np.asarray(self.history_torques, dtype=np.float32)
        self.future_torques = np.asarray(self.future_torques, dtype=np.float32)
        self.future_states = np.asarray(self.future_states, dtype=np.float32)
        self.physical_context = np.asarray(self.physical_context, dtype=np.float32)
        self.future_robot_dynamics = {
            key: np.asarray(values, dtype=np.float32) for key, values in self.future_robot_dynamics.items()
        }
        self.future_object_dynamics = {
            key: np.asarray(values, dtype=np.float32) for key, values in self.future_object_dynamics.items()
        }

    def __len__(self) -> int:
        return int(self.history_states.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return {
            "history_states": torch.from_numpy(self.history_states[idx]),
            "history_torques": torch.from_numpy(self.history_torques[idx]),
            "future_torques": torch.from_numpy(self.future_torques[idx]),
            "future_states": torch.from_numpy(self.future_states[idx]),
            "physical_context": torch.from_numpy(self.physical_context[idx]),
            "future_robot_dynamics": {
                key: torch.from_numpy(values[idx]) for key, values in self.future_robot_dynamics.items()
            },
            "future_object_dynamics": {
                key: torch.from_numpy(values[idx]) for key, values in self.future_object_dynamics.items()
            },
        }
