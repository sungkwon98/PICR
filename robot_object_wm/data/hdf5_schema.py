from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


EpisodeRef = Tuple[str, str]
STATE_PREDICTION_MODE_CHOICES = ("full", "position")
DEFAULT_PRIVILEGED_COLLISION_PAIRS = (
    "object_ground",
    "object_left_finger",
    "object_right_finger",
    "object_gripper",
)
PRIVILEGED_COLLISION_OBSERVATION_CHOICES = (0, 1, 2, 3)


def parse_privileged_collision_pairs(value: str | None) -> tuple[str, ...]:
    if value is None or not str(value).strip() or str(value).strip().lower() == "default":
        return DEFAULT_PRIVILEGED_COLLISION_PAIRS
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def privileged_collision_observation_dim(mode: int, pair_count: int) -> int:
    if mode not in PRIVILEGED_COLLISION_OBSERVATION_CHOICES:
        raise ValueError(f"privileged_collision_observation must be one of {PRIVILEGED_COLLISION_OBSERVATION_CHOICES}.")
    if mode == 0:
        return 0
    if mode in (1, 2):
        return pair_count
    # signed_distance plus nearest_points: P + P * 2 points * xyz.
    return pair_count + pair_count * 2 * 3


def normalize_state_prediction_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in STATE_PREDICTION_MODE_CHOICES:
        raise ValueError(f"state_prediction_mode must be one of {STATE_PREDICTION_MODE_CHOICES}.")
    return normalized


@dataclass(frozen=True)
class RobotObjectStateLayout:
    robot_dof: int = 9
    joint_vel_dim: int | None = None
    action_dim: int = 8
    torque_dim: int = 9
    object_pos_dim: int = 3
    object_quat_dim: int = 4
    object_lin_vel_dim: int = 3
    object_ang_vel_dim: int = 3
    object_material_dim: int = 3
    object_inertia_dim: int = 9
    object_mass_dim: int = 1
    privileged_collision_obs_dim: int = 0

    def __post_init__(self) -> None:
        if self.joint_vel_dim is None:
            object.__setattr__(self, "joint_vel_dim", int(self.robot_dof))

    @property
    def robot_state_dim(self) -> int:
        return self.robot_dof + int(self.joint_vel_dim or 0)

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
    def base_state_dim(self) -> int:
        return self.robot_state_dim + self.object_state_dim

    @property
    def state_dim(self) -> int:
        return self.base_state_dim + self.privileged_collision_obs_dim

    @property
    def robot_q_slice(self) -> slice:
        return slice(0, self.robot_dof)

    @property
    def robot_dq_slice(self) -> slice:
        start = self.robot_dof
        return slice(start, start + int(self.joint_vel_dim or 0))

    @property
    def object_start(self) -> int:
        return self.robot_state_dim

    @property
    def object_pos_slice(self) -> slice:
        start = self.object_start
        return slice(start, start + self.object_pos_dim)

    @property
    def object_quat_slice(self) -> slice:
        start = self.object_pos_slice.stop
        return slice(start, start + self.object_quat_dim)

    @property
    def object_lin_vel_slice(self) -> slice:
        start = self.object_quat_slice.stop
        return slice(start, start + self.object_lin_vel_dim)

    @property
    def object_ang_vel_slice(self) -> slice:
        start = self.object_lin_vel_slice.stop
        return slice(start, start + self.object_ang_vel_dim)

    @property
    def object_state_slice(self) -> slice:
        return slice(self.object_start, self.object_start + self.object_state_dim)

    @property
    def privileged_collision_slice(self) -> slice:
        start = self.object_state_slice.stop
        return slice(start, start + self.privileged_collision_obs_dim)

    @property
    def has_joint_vel(self) -> bool:
        return int(self.joint_vel_dim or 0) > 0

    @property
    def has_object_lin_vel(self) -> bool:
        return self.object_lin_vel_dim > 0

    @property
    def has_object_ang_vel(self) -> bool:
        return self.object_ang_vel_dim > 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self) | {
            "robot_state_dim": self.robot_state_dim,
            "object_state_dim": self.object_state_dim,
            "object_context_dim": self.object_context_dim,
            "base_state_dim": self.base_state_dim,
            "state_dim": self.state_dim,
        }


def make_robot_object_state_layout(
    *,
    robot_dof: int = 9,
    action_dim: int = 8,
    torque_dim: int = 9,
    state_prediction_mode: str = "full",
    privileged_collision_obs_dim: int = 0,
) -> RobotObjectStateLayout:
    mode = normalize_state_prediction_mode(state_prediction_mode)
    if mode == "full":
        return RobotObjectStateLayout(
            robot_dof=robot_dof,
            joint_vel_dim=robot_dof,
            action_dim=action_dim,
            torque_dim=torque_dim,
            object_lin_vel_dim=3,
            object_ang_vel_dim=3,
            privileged_collision_obs_dim=privileged_collision_obs_dim,
        )
    return RobotObjectStateLayout(
        robot_dof=robot_dof,
        joint_vel_dim=0,
        action_dim=action_dim,
        torque_dim=torque_dim,
        object_lin_vel_dim=0,
        object_ang_vel_dim=0,
        privileged_collision_obs_dim=privileged_collision_obs_dim,
    )


class Hdf5Groups:
    DATA = "data"
    OBS = "obs"
    STATES = "states"
    ACTIONS = "actions"
    ROBOT_TORQUES = "robot_torques"
    OBJECT_DYNAMICS = "object_dynamics"
