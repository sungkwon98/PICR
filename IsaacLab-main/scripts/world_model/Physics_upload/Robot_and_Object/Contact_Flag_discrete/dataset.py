"""Rollout dataset for the robot+object MCGDF world model.

Combines the joint-params and structured-residual machinery from
``../../Robot_only/MCGDF/dataset.py`` with the object-state and per-episode
context machinery from ``../GT_dynamics/dataset.py``.  Adds two pieces:

(1)  Per-episode *pre-contact baseline* of the residual torque used as the
     Approach-A clean-up of the GT residual target discussed in
     ``../../delan_vs_physx_verification.tex`` (Section 4.3, mismatch #2):

         baseline_e = < tau_applied - (tau_ID_GT + d*qdot + f*tanh(qdot/eps)) >_{pre-contact}

     The trainer subtracts ``-baseline_e`` from the per-step residual target to
     isolate the contact-induced contribution, which is then the supervision
     target for the contact-wrench head.

(2)  Optional subtraction of ``env_origin`` from the recorded world-frame
     object position.  The Lift task records ``root_pos_w`` in absolute
     simulator coordinates so positions can be very large when multiple envs
     are spawned at separate origins; the model gets a much cleaner signal in
     env-local frame.  ``env_origin`` is read from
     ``initial_state/articulation/robot/root_pose`` when available; otherwise
     a zero offset is used.
"""

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
class RobotObjectMCGDFStateLayout:
    """State layout combining the 9 DOF robot and the 13-dim free rigid body."""

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
        return (
            self.object_pos_dim
            + self.object_quat_dim
            + self.object_lin_vel_dim
            + self.object_ang_vel_dim
        )

    @property
    def object_context_dim(self) -> int:
        return self.object_mass_dim + self.object_inertia_dim + self.object_material_dim

    @property
    def state_dim(self) -> int:
        return self.robot_state_dim + self.object_state_dim

    def to_dict(self) -> dict[str, int]:
        return asdict(self) | {
            "robot_state_dim": self.robot_state_dim,
            "object_state_dim": self.object_state_dim,
            "object_context_dim": self.object_context_dim,
            "state_dim": self.state_dim,
        }


def discover_hdf5_files(dataset_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(os.path.abspath(dataset_dir), "*.hdf5")))


def load_episode_names(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as file:
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
    rng = random.Random(seed)
    refs = episode_refs.copy()
    rng.shuffle(refs)
    n_train = int(len(refs) * train_split)
    train_refs = refs[:n_train]
    val_refs = refs[n_train:]
    if len(train_refs) == 0 or len(val_refs) == 0:
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
    """Detect contact onset by looking for the first sustained object motion."""
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


def _h5_open(path: str) -> h5py.File:
    return h5py.File(path, "r", swmr=False)


# ---------------------------------------------------------------------------
# Franka FK -- used at dataset build time to compute a per-frame heuristic
# contact label from the gripper-to-cube Euclidean distance.  Adapted from
# ../GT_dynamics/eval_utils.franka_gripper_positions so this module stays
# torch-free at build time.
# ---------------------------------------------------------------------------

_FRANKA_FIXED_DH = (
    ((0.0, 0.0, 0.333),     (0.0,         0.0, 0.0)),
    ((0.0, 0.0, 0.0),       (-np.pi / 2., 0.0, 0.0)),
    ((0.0, -0.316, 0.0),    (np.pi / 2.,  0.0, 0.0)),
    ((0.0825, 0.0, 0.0),    (np.pi / 2.,  0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-np.pi / 2., 0.0, 0.0)),
    ((0.0, 0.0, 0.0),       (np.pi / 2.,  0.0, 0.0)),
    ((0.088, 0.0, 0.0),     (np.pi / 2.,  0.0, 0.0)),
)
_FRANKA_HAND_TRANSLATION = (0.0, 0.0, 0.107)
_FRANKA_HAND_RPY = (0.0, 0.0, -np.pi / 4.0)


def _transform_xyz_rpy(xyz, rpy) -> np.ndarray:
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rot_z @ rot_y @ rot_x
    transform[:3, 3] = xyz
    return transform


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    transform = np.eye(4)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return transform


def franka_gripper_positions(absolute_q: np.ndarray, tool_z_offset: float = 0.1034) -> np.ndarray:
    """Per-frame gripper-tip world position in the robot-base frame.

    ``absolute_q`` is ``(T, >=7)`` of absolute Panda joint angles.  Returns
    ``(T, 3)`` Cartesian positions.  The robot base sits at the env origin
    in the Lift task, so the returned positions live in the same env-local
    frame as the env-origin-subtracted cube position.
    """
    fixed = [_transform_xyz_rpy(xyz, rpy) for xyz, rpy in _FRANKA_FIXED_DH]
    hand_transform = _transform_xyz_rpy(_FRANKA_HAND_TRANSLATION, _FRANKA_HAND_RPY)
    tool_transform = _transform_xyz_rpy((0.0, 0.0, tool_z_offset), (0.0, 0.0, 0.0))
    positions = np.empty((absolute_q.shape[0], 3), dtype=np.float64)
    for t_idx, q_row in enumerate(absolute_q):
        transform = np.eye(4)
        for joint_idx in range(7):
            transform = transform @ fixed[joint_idx]
            transform = transform @ _rot_z(float(q_row[joint_idx]))
        gripper = transform @ hand_transform @ tool_transform
        positions[t_idx] = gripper[:3, 3]
    return positions.astype(np.float32)


class RobotObjectMCGDFRolloutDataset(Dataset):
    """Rollout samples for the robot+object MCGDF world model.

    Each sample contains:
      - ``history_states``       (K+1, state_dim) with robot 18 + object 13.
      - ``future_torques``       (H, torque_dim).
      - ``future_states``        (H, state_dim).
      - ``future_robot_dynamics`` dict with PhysX free-space labels.
      - ``future_object_dynamics`` dict with object kinematic + force labels.
      - ``joint_params``         per-joint damping/friction (constant per episode).
      - ``object_context``       (object_mass, object_inertia, material).
      - ``residual_baseline``    9-vector pre-contact average residual torque,
                                 used to subtract the implicit-PD baseline from
                                 the per-step GT residual (Approach A in the
                                 verification report, mismatch #2).
      - ``contact_t_in_window``  index within [t, t+H) of contact onset; -1 if
                                 contact never happens in the window, -2 if
                                 the window starts already in contact.
    """

    robot_dynamics_keys = (
        "qdd",
        "mass_matrix",
        "inertial",
        "coriolis",
        "gravity",
        "inverse_dynamics_tau",
    )
    object_dynamics_keys = (
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "root_lin_acc_w",
        "root_ang_acc_w",
    )

    # Sub-keys under episode["episode_physics_randomization"] used to build the
    # per-episode context_target.  Matches the layout used by ../../Ver1.  When
    # all entries are missing or zero, the context_target is the zero vector.
    physics_randomization_keys = (
        "object_contact",
        "object_mass",
        "robot_arm_joint_friction",
        "robot_gripper_joint_friction",
        "robot_link_masses",
    )

    def __init__(
        self,
        episode_refs: List[EpisodeRef],
        history_len: int,
        rollout_horizon: int,
        dt: float,
        torque_key: str = "applied_torque",
        friction_key: str = "joint_dynamic_friction_coeff",
        friction_eps: float = 1.0e-3,
        require_joint_params: bool = True,
        filter_pre_contact: bool = False,
        object_displacement_threshold: float = 0.005,
        object_velocity_threshold: float = 0.02,
        contact_consecutive_steps: int = 3,
        contact_settle_steps: int = 5,
        baseline_settle_steps: int = 5,
        subtract_env_origin: bool = True,
        tool_z_offset: float = 0.1034,
        contact_distance_threshold: float = 0.05,
        layout: RobotObjectMCGDFStateLayout | None = None,
    ) -> None:
        super().__init__()
        if history_len < 2:
            raise ValueError("history_len must be >= 2.")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be >= 1.")
        if contact_distance_threshold <= 0.0:
            raise ValueError("contact_distance_threshold must be > 0.")
        self.history_len = history_len
        self.rollout_horizon = rollout_horizon
        self.dt = dt
        self.torque_key = torque_key
        self.friction_key = friction_key
        self.friction_eps = friction_eps
        self.require_joint_params = require_joint_params
        self.baseline_settle_steps = baseline_settle_steps
        self.subtract_env_origin = subtract_env_origin
        self.tool_z_offset = tool_z_offset
        self.contact_distance_threshold = contact_distance_threshold
        self.layout = layout or RobotObjectMCGDFStateLayout()

        # Per-sample storage.
        self.history_states: list[np.ndarray] = []
        self.history_actions: list[np.ndarray] = []
        self.history_torques: list[np.ndarray] = []
        self.future_actions: list[np.ndarray] = []
        self.future_torques: list[np.ndarray] = []
        self.future_states: list[np.ndarray] = []
        self.object_context: list[np.ndarray] = []
        self.context_target: list[np.ndarray] = []
        self.joint_damping: list[np.ndarray] = []
        self.joint_friction: list[np.ndarray] = []
        self.residual_baseline: list[np.ndarray] = []
        self.contact_t_in_window: list[int] = []
        # Per-step binary contact label for the future window, derived from
        # gripper-cube Euclidean distance < ``contact_distance_threshold``.
        self.future_contact_label: list[np.ndarray] = []
        self.future_gripper_object_dist: list[np.ndarray] = []
        self.future_robot_dynamics: dict[str, list[np.ndarray]] = {k: [] for k in self.robot_dynamics_keys}
        self.future_object_dynamics: dict[str, list[np.ndarray]] = {k: [] for k in self.object_dynamics_keys}
        self.sample_meta: list[tuple[str, str, int, int]] = []
        self.context_target_dim: int = 0

        self.skipped_short_episodes = 0
        self.skipped_missing_torque_episodes = 0
        self.skipped_missing_robot_dynamics_episodes = 0
        self.skipped_missing_object_dynamics_episodes = 0
        self.skipped_missing_joint_params_episodes = 0
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

    # ------------------------------------------------------------------ utils
    def _slice_robot_vector(self, array) -> np.ndarray:
        return np.asarray(array, dtype=np.float32)[:, : self.layout.robot_dof]

    def _slice_robot_matrix(self, array) -> np.ndarray:
        matrix = np.asarray(array, dtype=np.float32)
        return matrix[:, : self.layout.robot_dof, : self.layout.robot_dof]

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

    def _read_joint_params(self, episode) -> tuple[np.ndarray, np.ndarray] | None:
        if "robot_joint_params" not in episode:
            return None
        jp = episode["robot_joint_params"]
        if "joint_damping" not in jp:
            return None
        damping = np.asarray(jp["joint_damping"], dtype=np.float32)[0, : self.layout.robot_dof]
        friction_field = self.friction_key
        if friction_field not in jp:
            friction_field = "joint_friction_coeff"
        if friction_field not in jp:
            return None
        friction = np.asarray(jp[friction_field], dtype=np.float32)[0, : self.layout.robot_dof]
        return damping, friction

    def _build_episode_context_target(self, episode) -> np.ndarray:
        """Concatenate ``episode_physics_randomization/*`` into a flat 1-D vector.

        Returns an empty (zero-length) vector when the group is absent so the
        sample loader can still build samples on legacy HDF5 files.  Matches
        the layout used by ``../../Ver1`` so the contrastive loss can be
        targeted against the same physical labels.
        """
        if "episode_physics_randomization" not in episode:
            return np.zeros((0,), dtype=np.float32)
        group = episode["episode_physics_randomization"]
        parts: list[np.ndarray] = []
        for key in self.physics_randomization_keys:
            if key not in group:
                continue
            arr = np.asarray(group[key], dtype=np.float32).reshape(-1)
            parts.append(arr)
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _env_origin(self, episode) -> np.ndarray:
        """Read the env origin (robot base position in world frame) for env-local positions."""
        try:
            init_group = episode["initial_state"]["articulation"]["robot"]
            if "root_pose" in init_group:
                return np.asarray(init_group["root_pose"], dtype=np.float32)[0, : self.layout.object_pos_dim]
        except KeyError:
            pass
        return np.zeros(self.layout.object_pos_dim, dtype=np.float32)

    def _compute_residual_baseline(
        self,
        contact_t: int | None,
        t_count: int,
        tau: np.ndarray,
        inv_dyn_tau: np.ndarray,
        damping: np.ndarray,
        friction: np.ndarray,
        qdot: np.ndarray,
    ) -> np.ndarray:
        """Pre-contact average residual r_theta target.

        From the MCGDF forward equation rearranged:

            r_theta = (M*qdd + c + g + d*qdot + f*tanh(qdot/eps)) - tau_applied

        the per-step "implicit target" for r_theta (Eq.~r-theta-implicit-target
        in the MCGDF description).  The episode-level baseline is the mean of
        this target over the pre-contact window, used as the "non-contact
        offset" subtracted from the per-step target to leave the contact
        contribution for the contact-wrench head.
        """
        end = contact_t if contact_t is not None else t_count
        start = min(max(0, self.baseline_settle_steps), max(0, end - 1))
        if end - start < 2:
            return np.zeros(self.layout.robot_dof, dtype=np.float32)
        qdot_w = qdot[start:end]
        damping_term = damping[None, :] * qdot_w
        friction_term = friction[None, :] * np.tanh(qdot_w / self.friction_eps)
        target_r_theta = (inv_dyn_tau[start:end] + damping_term + friction_term) - tau[start:end]
        return target_r_theta.mean(axis=0).astype(np.float32)

    # ----------------------------------------------------------------- build
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
                    obs = episode["obs"]
                    if "robot_torques" not in episode or self.torque_key not in episode["robot_torques"]:
                        self.skipped_missing_torque_episodes += 1
                        continue
                    if "robot_dynamics" not in episode:
                        self.skipped_missing_robot_dynamics_episodes += 1
                        continue
                    if "object_dynamics" not in episode or "states" not in episode:
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue
                    rdg = episode["robot_dynamics"]
                    odg = episode["object_dynamics"]
                    if any(key not in rdg for key in self.robot_dynamics_keys):
                        self.skipped_missing_robot_dynamics_episodes += 1
                        continue
                    if any(key not in odg for key in self.object_dynamics_keys):
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    jp = self._read_joint_params(episode)
                    if jp is None and self.require_joint_params:
                        self.skipped_missing_joint_params_episodes += 1
                        continue
                    if jp is None:
                        damping = np.zeros(self.layout.robot_dof, dtype=np.float32)
                        friction = np.zeros(self.layout.robot_dof, dtype=np.float32)
                    else:
                        damping, friction = jp

                    object_state_group = episode["states"]["rigid_object"]["object"]
                    root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
                    root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
                    object_pos_w = root_pose[:, : self.layout.object_pos_dim]
                    object_quat = root_pose[:, self.layout.object_pos_dim : self.layout.object_pos_dim + self.layout.object_quat_dim]
                    object_lin_vel = root_velocity[:, : self.layout.object_lin_vel_dim]
                    object_ang_vel = root_velocity[
                        :,
                        self.layout.object_lin_vel_dim : self.layout.object_lin_vel_dim + self.layout.object_ang_vel_dim,
                    ]
                    if self.subtract_env_origin:
                        origin = self._env_origin(episode)
                        object_pos = object_pos_w - origin[None, :]
                    else:
                        object_pos = object_pos_w

                    joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : self.layout.robot_dof]
                    joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : self.layout.robot_dof]
                    # Absolute joint angles for FK; only needed to derive the
                    # heuristic contact label (not consumed by the model).
                    if "states" in episode and "articulation" in episode["states"]:
                        absolute_joint_pos = np.asarray(
                            episode["states"]["articulation"]["robot"]["joint_position"],
                            dtype=np.float32,
                        )[:, : self.layout.robot_dof]
                    else:
                        absolute_joint_pos = joint_pos.copy()
                    actions = np.asarray(episode["actions"], dtype=np.float32)[:, : self.layout.action_dim]
                    torques = np.asarray(episode["robot_torques"][self.torque_key], dtype=np.float32)[
                        :, : self.layout.torque_dim
                    ]

                    robot_dynamics = {
                        "qdd": self._slice_robot_vector(rdg["qdd"]),
                        "mass_matrix": self._slice_robot_matrix(rdg["mass_matrix"]),
                        "inertial": self._slice_robot_vector(rdg["inertial"]),
                        "coriolis": self._slice_robot_vector(rdg["coriolis"]),
                        "gravity": self._slice_robot_vector(rdg["gravity"]),
                        "inverse_dynamics_tau": self._slice_robot_vector(rdg["inverse_dynamics_tau"]),
                    }
                    object_dynamics = {
                        key: np.asarray(odg[key], dtype=np.float32) for key in self.object_dynamics_keys
                    }

                    mass = np.asarray(odg["mass"], dtype=np.float32).reshape(-1, self.layout.object_mass_dim)
                    inertia = np.asarray(odg["inertia"], dtype=np.float32).reshape(-1, self.layout.object_inertia_dim)

                    t_count = min(
                        joint_pos.shape[0],
                        joint_vel.shape[0],
                        absolute_joint_pos.shape[0],
                        actions.shape[0],
                        torques.shape[0],
                        object_pos.shape[0],
                        object_quat.shape[0],
                        object_lin_vel.shape[0],
                        object_ang_vel.shape[0],
                        mass.shape[0],
                        inertia.shape[0],
                        *(v.shape[0] for v in robot_dynamics.values()),
                        *(v.shape[0] for v in object_dynamics.values()),
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
                    robot_dynamics = {key: value[:t_count] for key, value in robot_dynamics.items()}
                    object_dynamics = {key: value[:t_count] for key, value in object_dynamics.items()}

                    contact_t = first_object_motion_timestep(
                        object_pos=object_pos[:t_count],
                        dt=self.dt,
                        displacement_threshold=object_displacement_threshold,
                        velocity_threshold=object_velocity_threshold,
                        consecutive_steps=contact_consecutive_steps,
                        settle_steps=contact_settle_steps,
                    )

                    # Per-frame heuristic contact label: 1 iff the gripper
                    # tip is within ``contact_distance_threshold`` of the
                    # cube centre.  Both endpoints live in the env-local
                    # frame (FK is in the robot-base frame, which equals
                    # the env origin in Lift), so the Euclidean norm is
                    # directly comparable to the threshold.
                    gripper_pos = franka_gripper_positions(
                        absolute_joint_pos[:t_count], self.tool_z_offset,
                    )
                    gripper_object_dist_ep = np.linalg.norm(
                        gripper_pos - object_pos[:t_count], axis=-1,
                    ).astype(np.float32)
                    contact_label_ep = (
                        gripper_object_dist_ep < self.contact_distance_threshold
                    ).astype(np.float32)
                    if contact_t is not None:
                        self.episodes_with_contact += 1

                    residual_baseline = self._compute_residual_baseline(
                        contact_t=contact_t,
                        t_count=t_count,
                        tau=torques,
                        inv_dyn_tau=robot_dynamics["inverse_dynamics_tau"],
                        damping=damping,
                        friction=friction,
                        qdot=joint_vel[:t_count],
                    )

                    episode_context_target = self._build_episode_context_target(episode)
                    if self.context_target_dim == 0 and episode_context_target.size > 0:
                        self.context_target_dim = int(episode_context_target.size)

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
                        self.context_target.append(episode_context_target.copy())
                        self.joint_damping.append(damping)
                        self.joint_friction.append(friction)
                        self.residual_baseline.append(residual_baseline)
                        if contact_t is None:
                            ct_in = -1
                        elif contact_t <= t:
                            ct_in = -2
                        elif contact_t >= t + h + 1:
                            ct_in = -1
                        else:
                            ct_in = int(contact_t - (t + 1))
                        self.contact_t_in_window.append(ct_in)
                        # Future-window contact label and raw distance,
                        # aligned with ``future_states[t+1 : t+h+1]``.
                        self.future_contact_label.append(
                            contact_label_ep[t + 1 : t + h + 1]
                        )
                        self.future_gripper_object_dist.append(
                            gripper_object_dist_ep[t + 1 : t + h + 1]
                        )
                        for key in self.robot_dynamics_keys:
                            self.future_robot_dynamics[key].append(robot_dynamics[key][t : t + h])
                        for key in self.object_dynamics_keys:
                            self.future_object_dynamics[key].append(object_dynamics[key][t : t + h])
                        self.sample_meta.append((hdf5_path, episode_name, t - k, t + h))

        if len(self.history_states) == 0:
            raise RuntimeError("No valid robot-object MCGDF rollout samples were built.")

        self.history_states = np.asarray(self.history_states, dtype=np.float32)
        self.history_actions = np.asarray(self.history_actions, dtype=np.float32)
        self.history_torques = np.asarray(self.history_torques, dtype=np.float32)
        self.future_actions = np.asarray(self.future_actions, dtype=np.float32)
        self.future_torques = np.asarray(self.future_torques, dtype=np.float32)
        self.future_states = np.asarray(self.future_states, dtype=np.float32)
        self.object_context = np.asarray(self.object_context, dtype=np.float32)
        self.joint_damping = np.asarray(self.joint_damping, dtype=np.float32)
        self.joint_friction = np.asarray(self.joint_friction, dtype=np.float32)
        self.residual_baseline = np.asarray(self.residual_baseline, dtype=np.float32)
        self.contact_t_in_window = np.asarray(self.contact_t_in_window, dtype=np.int64)
        self.future_contact_label = np.asarray(self.future_contact_label, dtype=np.float32)
        self.future_gripper_object_dist = np.asarray(self.future_gripper_object_dist, dtype=np.float32)
        # context_target may have variable size; pad-or-stack into a homogeneous array.
        if self.context_target_dim > 0 and len(self.context_target) > 0:
            stacked: list[np.ndarray] = []
            for vec in self.context_target:
                if vec.size < self.context_target_dim:
                    padded = np.zeros((self.context_target_dim,), dtype=np.float32)
                    padded[: vec.size] = vec
                    stacked.append(padded)
                else:
                    stacked.append(vec[: self.context_target_dim].astype(np.float32))
            self.context_target = np.asarray(stacked, dtype=np.float32)
        else:
            self.context_target = np.zeros((len(self.history_states), 0), dtype=np.float32)
        self.future_robot_dynamics = {
            key: np.asarray(vals, dtype=np.float32) for key, vals in self.future_robot_dynamics.items()
        }
        self.future_object_dynamics = {
            key: np.asarray(vals, dtype=np.float32) for key, vals in self.future_object_dynamics.items()
        }

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
            "context_target": torch.from_numpy(self.context_target[idx]),
            "joint_params": {
                "damping": torch.from_numpy(self.joint_damping[idx]),
                "friction": torch.from_numpy(self.joint_friction[idx]),
            },
            "residual_baseline": torch.from_numpy(self.residual_baseline[idx]),
            "contact_t_in_window": int(self.contact_t_in_window[idx]),
            "future_contact_label": torch.from_numpy(self.future_contact_label[idx]),
            "future_gripper_object_dist": torch.from_numpy(self.future_gripper_object_dist[idx]),
            "future_robot_dynamics": {
                key: torch.from_numpy(vals[idx]) for key, vals in self.future_robot_dynamics.items()
            },
            "future_object_dynamics": {
                key: torch.from_numpy(vals[idx]) for key, vals in self.future_object_dynamics.items()
            },
        }
