from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from .utils import build_mlp, normalize_quat


@dataclass
class ObjectState:
    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor

    @classmethod
    def from_full_state(cls, state: torch.Tensor, robot_dof: int) -> "ObjectState":
        offset = 2 * robot_dof
        return cls(
            pos=state[:, offset : offset + 3],
            quat=normalize_quat(state[:, offset + 3 : offset + 7]),
            lin_vel=state[:, offset + 7 : offset + 10],
            ang_vel=state[:, offset + 10 : offset + 13],
        )

    def as_tensor(self) -> torch.Tensor:
        return torch.cat([self.pos, normalize_quat(self.quat), self.lin_vel, self.ang_vel], dim=-1)


@dataclass
class ObjectStepInput:
    full_state: torch.Tensor
    torque: torch.Tensor
    object_state: ObjectState
    history_states: torch.Tensor
    history_torques: torch.Tensor
    z: torch.Tensor | None = None


@dataclass
class ObjectStepOutput:
    state: ObjectState
    aux: dict[str, torch.Tensor] = field(default_factory=dict)



class ObjectStateMLPDynamics(nn.Module):
    """Pure MLP object dynamics using history, current torque, and latent z."""

    def __init__(
        self,
        robot_dof: int = 9,
        state_dim: int = 31,
        torque_dim: int = 9,
        history_len: int = 1,
        latent_dim: int = 0,
        hidden_dim: int = 256,
        depth: int = 3,
        dt: float = 0.02,
    ) -> None:
        super().__init__()
        del robot_dof
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        self.state_dim = int(state_dim)
        self.torque_dim = int(torque_dim)
        self.history_len = int(history_len)
        self.latent_dim = int(latent_dim)
        self.dt = float(dt)
        input_dim = self.history_len * (self.state_dim + self.torque_dim) + self.torque_dim + self.latent_dim
        self.net = build_mlp(input_dim, hidden_dim, 13, depth=depth)

    def step(self, values: ObjectStepInput) -> ObjectStepOutput:
        self._validate_inputs(values)
        parts = [
            values.history_states.flatten(start_dim=1),
            values.history_torques.flatten(start_dim=1),
            values.torque,
        ]
        if self.latent_dim > 0:
            if values.z is None:
                raise ValueError("ObjectStateMLPDynamics requires latent z.")
            if values.z.shape[-1] != self.latent_dim:
                raise ValueError(f"Expected latent dim {self.latent_dim}, got {values.z.shape[-1]}.")
            parts.append(values.z)

        pred = self.net(torch.cat(parts, dim=-1))
        next_state = ObjectState(
            pos=pred[:, :3],
            quat=normalize_quat(pred[:, 3:7]),
            lin_vel=pred[:, 7:10],
            ang_vel=pred[:, 10:13],
        )
        aux = {
            "object_lin_acc": (next_state.lin_vel - values.object_state.lin_vel) / self.dt,
            "object_ang_acc": (next_state.ang_vel - values.object_state.ang_vel) / self.dt,
        }
        return ObjectStepOutput(state=next_state, aux=aux)

    def _validate_inputs(self, values: ObjectStepInput) -> None:
        if values.full_state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected full_state dim {self.state_dim}, got {values.full_state.shape[-1]}.")
        if values.torque.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {values.torque.shape[-1]}.")
        if values.history_states.shape[1:] != (self.history_len, self.state_dim):
            raise ValueError(
                f"Expected history_states shape (batch, {self.history_len}, {self.state_dim}), "
                f"got {tuple(values.history_states.shape)}."
            )
        if values.history_torques.shape[1:] != (self.history_len, self.torque_dim):
            raise ValueError(
                f"Expected history_torques shape (batch, {self.history_len}, {self.torque_dim}), "
                f"got {tuple(values.history_torques.shape)}."
            )
