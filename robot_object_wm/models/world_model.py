from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .context import (
    ContextEncoder,
    ContextOutput,
    context_info_nce_loss,
    context_kl_loss,
    info_nce_soft,
    reparameterize,
)
from .delan import DeLaNRobotDynamics
from .object_dynamics import ObjectState, ObjectStateMLPDynamics, ObjectStepInput

RobotDynamics = DeLaNRobotDynamics
ObjDynamics = ObjectStateMLPDynamics


@dataclass(frozen=True)
class WMDynamicsConfig:
    robot_dof: int = 9
    torque_dim: int = 9
    hidden_dim: int = 256
    object_hidden_dim: int = 256
    object_depth: int = 3
    dt: float = 0.02
    ode_solver: str = "euler"
    tool_z_offset: float = 0.1034
    history_len: int = 1
    use_context_encoder: bool = False
    latent_dim: int = 8
    context_encoder_hidden_dim: int = 256
    context_encoder_depth: int = 2
    delan_use_film: bool = False
    delan_film_depth: int = 2
    delan_use_history : bool =False 
    privileged_collision_obs_dim: int = 0

    @property
    def object_state_dim(self) -> int:
        return 13 + self.privileged_collision_obs_dim

    @property
    def state_dim(self) -> int:
        return 2 * self.robot_dof + self.object_state_dim


class WMDynamics(nn.Module):
    """Rollout model: DeLaN robot dynamics plus an object-state MLP."""

    def __init__(
        self,
        robot: RobotDynamics,
        obj: ObjDynamics,
        robot_dof: int = 9,
        *,
        history_len: int = 1,
        torque_dim: int = 9,
        use_context_encoder: bool = False,
        latent_dim: int = 8,
        context_encoder_hidden_dim: int = 256,
        context_encoder_depth: int = 2,
        delan_use_film: bool = False,
        privileged_collision_obs_dim: int = 0,
    ) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        self.privileged_collision_obs_dim = int(privileged_collision_obs_dim)
        self.state_dim = 2 * robot_dof + 13 + self.privileged_collision_obs_dim
        self.torque_dim = torque_dim
        self.history_len = history_len
        self.use_context_encoder = bool(use_context_encoder)
        self.delan_use_film = bool(delan_use_film)
        self.latent_dim = latent_dim if self.use_context_encoder else 0
        self.robot = robot
        self.obj = obj
        if self.delan_use_film and not self.use_context_encoder:
            raise ValueError("delan_use_film=True requires use_context_encoder=True.")
        if self.use_context_encoder:
            if history_len <= 0:
                raise ValueError("use_context_encoder=True requires history_len > 0.")
            seq_input_dim = history_len * (self.state_dim + torque_dim)
            self.context_encoder = ContextEncoder(
                seq_input_dim=seq_input_dim,
                hidden_dim=context_encoder_hidden_dim,
                latent_dim=latent_dim,
                depth=context_encoder_depth,
            )
        else:
            self.context_encoder = None

    def forward(
        self,
        history_states: torch.Tensor,
        future_torques: torch.Tensor,
        object_context: torch.Tensor | None = None,
        history_torques: torch.Tensor | None = None,
        deterministic_context: bool | None = None,
        *,
        return_aux: bool = False,
        return_context: bool = False,
    ):
        if history_states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {history_states.shape[-1]}")
        if future_torques.ndim != 3:
            raise ValueError("future_torques must have shape (batch, horizon, torque_dim).")

        state = history_states[:, -1]
        if history_torques is None:
            history_torques = history_states.new_zeros(
                history_states.shape[0],
                history_states.shape[1],
                self.torque_dim,
            )
        context = self.encode_context(
            history_states,
            history_torques,
            deterministic_context=deterministic_context,
        )
        z = context.z
        history_state_window = history_states
        history_torque_window = history_torques
        predictions: list[torch.Tensor] = []
        aux_steps: list[dict[str, torch.Tensor]] = []

        for step in range(future_torques.shape[1]):
            torque = future_torques[:, step]
            q, dq = self._split_robot(state)
            object_state = ObjectState.from_full_state(state, self.robot_dof)

            object_out = self.obj.step(
                ObjectStepInput(
                    full_state=state,
                    torque=torque,
                    object_state=object_state,
                    history_states=history_state_window,
                    history_torques=history_torque_window,
                    z=z,
                )
            )
            x = torch.cat((history_state_window[:, :-1].flatten(start_dim=1),q), dim=1) 
            robot_out = self.robot(x=x, dq=dq, torque=torque, z=z)

            state_parts = [robot_out.next_q, robot_out.next_dq, object_out.state.as_tensor()]
            if self.privileged_collision_obs_dim > 0:
                if object_out.privileged_collision is None:
                    raise RuntimeError("Object dynamics did not return privileged collision state.")
                state_parts.append(object_out.privileged_collision)
            state = torch.cat(state_parts, dim=-1)
            predictions.append(state)
            history_state_window = torch.cat([history_state_window[:, 1:], state[:, None]], dim=1)
            history_torque_window = torch.cat([history_torque_window[:, 1:], torque[:, None]], dim=1)
            if return_aux:
                aux = {}
                aux.update(robot_out.aux)
                aux.update(object_out.aux)
                aux.update(context.as_aux())
                aux_steps.append(aux)

        future_states = torch.stack(predictions, dim=1)
        if return_aux and return_context:
            return future_states, aux_steps, context
        if return_aux:
            return future_states, aux_steps
        if return_context:
            return future_states, context
        return future_states

    def encode_context(
        self,
        history_states: torch.Tensor,
        history_torques: torch.Tensor | None,
        *,
        deterministic_context: bool | None = None,
    ) -> ContextOutput:
        if not self.use_context_encoder:
            return ContextOutput(mu=None, logvar=None, z=None)
        if self.context_encoder is None:
            raise RuntimeError("Context encoder is not initialized.")
        if history_torques is None:
            raise ValueError("use_context_encoder=True requires history_torques.")
        if history_states.shape[1] != self.history_len:
            raise ValueError(f"Expected history_len {self.history_len}, got {history_states.shape[1]}.")
        if history_torques.shape[:2] != history_states.shape[:2]:
            raise ValueError(
                "history_torques must have shape (batch, history_len, torque_dim) matching history_states."
            )
        if history_torques.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {history_torques.shape[-1]}.")
        flat_history = torch.cat([history_states, history_torques], dim=-1).flatten(start_dim=1)
        mu, logvar = self.context_encoder(flat_history)
        if deterministic_context is None:
            deterministic_context = not self.training
        z = mu if deterministic_context else reparameterize(mu, logvar)
        return ContextOutput(mu=mu, logvar=logvar, z=z)

    def _split_robot(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[:, : self.robot_dof], state[:, self.robot_dof : 2 * self.robot_dof]


def build_robot_dynamics(cfg: WMDynamicsConfig) -> RobotDynamics:
    return RobotDynamics(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        hidden_dim=cfg.hidden_dim,
        history_len = cfg.history_len,
        dt=cfg.dt,
        ode_solver=cfg.ode_solver,
        latent_dim=cfg.latent_dim if cfg.delan_use_film else None,
        use_film=cfg.delan_use_film,
        film_depth=cfg.delan_film_depth,
        use_history=cfg.delan_use_history,
        history_state_dim=cfg.state_dim,
    )


def build_obj_dynamics(cfg: WMDynamicsConfig) -> ObjDynamics:
    return ObjDynamics(
        robot_dof=cfg.robot_dof,
        state_dim=cfg.state_dim,
        torque_dim=cfg.torque_dim,
        history_len=cfg.history_len,
        latent_dim=cfg.latent_dim if cfg.use_context_encoder else 0,
        hidden_dim=cfg.object_hidden_dim,
        depth=cfg.object_depth,
        dt=cfg.dt,
    )


def build_wm_dynamics(cfg: WMDynamicsConfig = WMDynamicsConfig()) -> WMDynamics:
    return WMDynamics(
        robot=build_robot_dynamics(cfg),
        obj=build_obj_dynamics(cfg),
        robot_dof=cfg.robot_dof,
        history_len=cfg.history_len,
        torque_dim=cfg.torque_dim,
        use_context_encoder=cfg.use_context_encoder,
        latent_dim=cfg.latent_dim,
        context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
        context_encoder_depth=cfg.context_encoder_depth,
        delan_use_film=cfg.delan_use_film,
        privileged_collision_obs_dim=cfg.privileged_collision_obs_dim,
    )


def weighted_rollout_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    robot_dof: int,
    q_weight: float = 1.0,
    dq_weight: float = 0.2,
    object_pos_weight: float = 5.0,
    object_quat_weight: float = 0.5,
    object_lin_vel_weight: float = 1.0,
    object_ang_vel_weight: float = 0.5,
    privileged_collision_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    offset = 2 * robot_dof
    object_end = offset + 13
    parts = {
        "q_mse": F.mse_loss(pred[..., :robot_dof], target[..., :robot_dof]),
        "dq_mse": F.mse_loss(pred[..., robot_dof:offset], target[..., robot_dof:offset]),
        "object_pos_mse": F.mse_loss(pred[..., offset : offset + 3], target[..., offset : offset + 3]),
        "object_quat_mse": F.mse_loss(pred[..., offset + 3 : offset + 7], target[..., offset + 3 : offset + 7]),
        "object_lin_vel_mse": F.mse_loss(pred[..., offset + 7 : offset + 10], target[..., offset + 7 : offset + 10]),
        "object_ang_vel_mse": F.mse_loss(pred[..., offset + 10 : offset + 13], target[..., offset + 10 : offset + 13]),
    }
    loss = (
        q_weight * parts["q_mse"]
        + dq_weight * parts["dq_mse"]
        + object_pos_weight * parts["object_pos_mse"]
        + object_quat_weight * parts["object_quat_mse"]
        + object_lin_vel_weight * parts["object_lin_vel_mse"]
        + object_ang_vel_weight * parts["object_ang_vel_mse"]
    )
    if pred.shape[-1] > object_end:
        parts["privileged_collision_mse"] = F.mse_loss(pred[..., object_end:], target[..., object_end:])
        loss = loss + privileged_collision_weight * parts["privileged_collision_mse"]
    return loss, parts
