from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


EpisodeRef = Tuple[str, str]
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


@dataclass(frozen=True)
class RobotObjectStateLayout:
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
    privileged_collision_obs_dim: int = 0

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
    def base_state_dim(self) -> int:
        return self.robot_state_dim + self.object_state_dim

    @property
    def state_dim(self) -> int:
        return self.base_state_dim + self.privileged_collision_obs_dim

    def to_dict(self) -> dict[str, int]:
        return asdict(self) | {
            "robot_state_dim": self.robot_state_dim,
            "object_state_dim": self.object_state_dim,
            "object_context_dim": self.object_context_dim,
            "base_state_dim": self.base_state_dim,
            "state_dim": self.state_dim,
        }


class Hdf5Groups:
    DATA = "data"
    OBS = "obs"
    STATES = "states"
    ACTIONS = "actions"
    ROBOT_TORQUES = "robot_torques"
    OBJECT_DYNAMICS = "object_dynamics"
