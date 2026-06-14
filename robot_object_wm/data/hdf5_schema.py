from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


EpisodeRef = Tuple[str, str]


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


class Hdf5Groups:
    DATA = "data"
    OBS = "obs"
    STATES = "states"
    ACTIONS = "actions"
    ROBOT_TORQUES = "robot_torques"
    OBJECT_DYNAMICS = "object_dynamics"
