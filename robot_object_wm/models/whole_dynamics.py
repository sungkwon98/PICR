from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .context import ContextEncoder, ContextOutput, reparameterize
from .delan import DeLaNCore, DelanTerms, FiLMLayer, validate_ode_solver
from .utils import build_mlp, integrate_quat, normalize_quat, safe_solve


@dataclass(frozen=True)
class WholeWMDynamicsConfig:
    robot_dof: int = 9
    torque_dim: int = 9
    hidden_dim: int = 256
    dt: float = 0.02
    ode_solver: str = "euler"
    history_len: int = 1
    use_context_encoder: bool = False
    latent_dim: int = 8
    context_encoder_hidden_dim: int = 256
    context_encoder_depth: int = 2
    delan_use_film: bool = False
    delan_film_depth: int = 2
    delan_use_history : bool = False
    mlp_depth: int = 3
    privileged_collision_obs_dim: int = 0

    @property
    def object_state_dim(self) -> int:
        return 13 + self.privileged_collision_obs_dim

    @property
    def object_dof(self) -> int:
        # Object generalized coordinates are pos(3) + rotvec(3).
        return 6

    @property
    def whole_dof(self) -> int:
        return self.robot_dof + self.object_dof

    @property
    def state_dim(self) -> int:
        # Existing dataset/model state:
        # robot_q, robot_dq, object_pos, object_quat, object_lin_vel, object_ang_vel.
        return 2 * self.robot_dof + self.object_state_dim


class WholeDeLaNWMDynamics(nn.Module):
    """One DeLaN over robot + object generalized coordinates.

    Generalized coordinates:
        q_whole  = [robot_q(9), object_pos(3), object_rotvec(3)]
        dq_whole = [robot_dq(9), object_lin_vel(3), object_ang_vel(3)]

    The public state format stays compatible with the existing dataset:
        [robot_q, robot_dq, object_pos, object_quat, object_lin_vel, object_ang_vel]
    """

    def __init__(
        self,
        *,
        robot_dof: int = 9,
        torque_dim: int = 9,
        hidden_dim: int = 256,
        dt: float = 0.02,
        history_len: int = 1,
        ode_solver: str = "euler",
        use_context_encoder: bool = False,
        latent_dim: int = 8,
        context_encoder_hidden_dim: int = 256,
        context_encoder_depth: int = 2,
        delan_use_film: bool = False,
        delan_film_depth: int = 2,
        delan_use_history : bool = False,
        mlp_depth: int = 3,
        privileged_collision_obs_dim: int = 0,
    ) -> None:
        super().__init__()
        if torque_dim != robot_dof:
            raise ValueError("WholeDeLaNWMDynamics expects torque_dim == robot_dof for fixed actuation.")
        self.robot_dof = int(robot_dof)
        self.object_dof = 6
        self.whole_dof = self.robot_dof + self.object_dof
        self.privileged_collision_obs_dim = int(privileged_collision_obs_dim)
        self.base_state_dim = 2 * self.robot_dof + 13
        self.state_dim = self.base_state_dim + self.privileged_collision_obs_dim
        self.torque_dim = int(torque_dim)
        self.history_len = int(history_len)
        self.dt = float(dt)
        self.ode_solver = validate_ode_solver(ode_solver)
        self.use_context_encoder = bool(use_context_encoder)
        self.delan_use_film = bool(delan_use_film)
        self.latent_dim = int(latent_dim) if self.use_context_encoder else 0
        if self.delan_use_film and not self.use_context_encoder:
            raise ValueError("delan_use_film=True requires use_context_encoder=True.")
        self.delan_use_history = delan_use_history
        self.core = DeLaNCore(
            robot_dof=self.whole_dof,
            hidden_dim=hidden_dim,
            history_len=history_len if delan_use_history else 1,
            latent_dim=latent_dim if self.delan_use_film else None,
            use_film=self.delan_use_film,
            film_depth=delan_film_depth,
            whole_dynamics=True,
            history_state_dim=self.state_dim,
        )

        latent_input_dim = self.latent_dim if self.use_context_encoder and not self.delan_use_film else 0
        collision_input_dim = self.history_len * self.state_dim + self.torque_dim + latent_input_dim
        if self.privileged_collision_obs_dim > 0 and self.delan_use_film:
            self.privileged_collision_net = _FiLMStateMLP(
                input_dim=collision_input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.privileged_collision_obs_dim,
                depth=delan_film_depth,
                latent_dim=latent_dim,
            )
        elif self.privileged_collision_obs_dim > 0:
            self.privileged_collision_net = build_mlp(
                collision_input_dim,
                hidden_dim,
                self.privileged_collision_obs_dim,
                depth=mlp_depth,
            )
        else:
            self.privileged_collision_net = None

        if self.use_context_encoder:
            if self.history_len <= 0:
                raise ValueError("use_context_encoder=True requires history_len > 0.")
            seq_input_dim = self.history_len * (self.state_dim + self.torque_dim)
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
        del object_context
        self._validate_rollout_inputs(history_states, future_torques)
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

        predictions: list[torch.Tensor] = []
        aux_steps: list[dict[str, torch.Tensor]] = []

        for step in range(future_torques.shape[1]):
            torque = future_torques[:, step]
            state, aux = self.step(history_states, torque, z=z, return_aux=return_aux)
            history_states = torch.cat([history_states[:, 1:], state[:, None]], axis=1)
            predictions.append(state)
            if return_aux:
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

    def step(
        self,
        history_state: torch.Tensor,
        torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if torque.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {torque.shape[-1]}.")
        state = history_state[:, -1]
        q_whole, dq_whole, quat = self.state_to_generalized(state)
        generalized_force = self.fixed_generalized_force(torque)
        x_whole  = torch.cat((history_state[:, :-1].flatten(start_dim=1),q_whole), dim=1) 
        ddq_whole, terms = self.acceleration(x_whole, dq_whole, generalized_force, z=z)
        next_q_whole, next_dq_whole = self.integrate(
            x_whole,
            dq_whole,
            generalized_force,
            ddq_whole=ddq_whole,
            z=z,
        )
        next_state = self.generalized_to_state(
            next_q_whole=next_q_whole,
            next_dq_whole=next_dq_whole,
            current_quat=quat,
        )
        if self.privileged_collision_obs_dim > 0:
            next_collision = self.predict_privileged_collision(history_state, torque, z=z)
            next_state = torch.cat([next_state, next_collision], dim=-1)

        aux: dict[str, torch.Tensor] = {}
        if return_aux:
            inertial = (terms.H @ ddq_whole.unsqueeze(-1)).squeeze(-1)
            aux.update(terms.as_aux())
            aux.update(
                {
                    "generalized_force": generalized_force,
                    "ddq_whole": ddq_whole,
                    "inertial": inertial,
                    "inverse_dynamics_force": inertial + terms.coriolis + terms.g,
                    "ode_solver": torch.tensor(0 if self.ode_solver == "euler" else 1, device=state.device),
                }
            )
        return next_state, aux

    def predict_privileged_collision(
        self,
        history_state: torch.Tensor,
        torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.privileged_collision_net is None:
            return history_state.new_empty((history_state.shape[0], 0))
        parts = [history_state.flatten(start_dim=1), torque]
        if self.use_context_encoder and not self.delan_use_film:
            if z is None:
                raise ValueError("WholeDeLaNWMDynamics privileged collision head requires latent z.")
            parts.append(z)
        x = torch.cat(parts, dim=-1)
        if self.delan_use_film:
            if z is None:
                raise ValueError("WholeDeLaNWMDynamics FiLM collision head requires latent z.")
            return self.privileged_collision_net(x, z)
        return self.privileged_collision_net(x)

    def fixed_generalized_force(self, torque: torch.Tensor) -> torch.Tensor:
        zeros_object = torque.new_zeros(torque.shape[0], self.object_dof)
        return torch.cat([torque, zeros_object], dim=-1)

    def acceleration(
        self,
        x_whole: torch.Tensor,
        dq_whole: torch.Tensor,
        generalized_force: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DelanTerms]:
        if not self.delan_use_history:
            x_whole = x_whole[:, -self.whole_dof:]
        terms = self.core(x_whole, dq_whole, z=z if self.delan_use_film else None)
        rhs = generalized_force - terms.g - terms.coriolis
        ddq = safe_solve(terms.H, rhs)
        ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3)
        ddq = ddq.clamp(-1e3, 1e3)
        return ddq, terms

    def state_derivative(
        self,
        x_whole: torch.Tensor,
        dq_whole: torch.Tensor,
        generalized_force: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ddq_whole, _terms = self.acceleration(x_whole, dq_whole, generalized_force, z=z)
        return dq_whole, ddq_whole

    def integrate(
        self,
        x_whole: torch.Tensor,
        dq_whole: torch.Tensor,
        generalized_force: torch.Tensor,
        *,
        ddq_whole: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_whole = x_whole[:, -(self.whole_dof):]
        if self.ode_solver == "euler":
            next_dq_whole = dq_whole + ddq_whole * self.dt
            next_q_whole = q_whole + next_dq_whole * self.dt
            return next_q_whole, next_dq_whole
        return self._rk4_integrate(x_whole, dq_whole, generalized_force, z=z)

    def _rk4_integrate(
        self,
        x_whole: torch.Tensor,
        dq_whole: torch.Tensor,
        generalized_force: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.dt
        q_whole = x_whole[:, -(self.whole_dof):]
        k1_q, k1_dq = self.state_derivative(x_whole, dq_whole, generalized_force, z=z)
        new_q = q_whole + 0.5 * dt * k1_q
        new_x= x_whole.clone()
        new_x[...,-self.whole_dof:] = new_q
        k2_q, k2_dq = self.state_derivative(
            new_x,
            dq_whole + 0.5 * dt * k1_dq,
            generalized_force,
            z=z,
        )
        new_q = q_whole + 0.5 * dt * k2_q
        new_x= x_whole.clone()
        new_x[...,-self.whole_dof:] = new_q
        k3_q, k3_dq = self.state_derivative(
            new_x,
            dq_whole + 0.5 * dt * k2_dq,
            generalized_force,
            z=z,
        )
        new_q = q_whole + dt * k3_q
        new_x= x_whole.clone()
        new_x[...,-self.whole_dof:] = new_q
        k4_q, k4_dq = self.state_derivative(
            new_x,
            dq_whole + dt * k3_dq,
            generalized_force,
            z=z,
        )
        next_q_whole = q_whole + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        next_dq_whole = dq_whole + (dt / 6.0) * (k1_dq + 2.0 * k2_dq + 2.0 * k3_dq + k4_dq)
        return next_q_whole, next_dq_whole

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

    def state_to_generalized(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape[-1]}.")
        r = self.robot_dof
        robot_q = state[:, :r]
        robot_dq = state[:, r : 2 * r]
        offset = 2 * r
        object_pos = state[:, offset : offset + 3]
        object_quat = normalize_quat(state[:, offset + 3 : offset + 7])
        object_lin_vel = state[:, offset + 7 : offset + 10]
        object_ang_vel = state[:, offset + 10 : offset + 13]
        object_rotvec = quat_to_rotvec(object_quat)
        q_whole = torch.cat([robot_q, object_pos, object_rotvec], dim=-1)
        dq_whole = torch.cat([robot_dq, object_lin_vel, object_ang_vel], dim=-1)
        return q_whole, dq_whole, object_quat

    def generalized_to_state(
        self,
        *,
        next_q_whole: torch.Tensor,
        next_dq_whole: torch.Tensor,
        current_quat: torch.Tensor,
    ) -> torch.Tensor:
        r = self.robot_dof
        next_robot_q = next_q_whole[:, :r]
        next_object_pos = next_q_whole[:, r : r + 3]
        next_object_rotvec = next_q_whole[:, r + 3 : r + 6]
        next_robot_dq = next_dq_whole[:, :r]
        next_object_lin_vel = next_dq_whole[:, r : r + 3]
        next_object_ang_vel = next_dq_whole[:, r + 3 : r + 6]
        if self.ode_solver == "euler":
            next_object_quat = integrate_quat(current_quat, next_object_ang_vel, self.dt)
        else:
            next_object_quat = rotvec_to_quat(next_object_rotvec)
        return torch.cat(
            [
                next_robot_q,
                next_robot_dq,
                next_object_pos,
                next_object_quat,
                next_object_lin_vel,
                next_object_ang_vel,
            ],
            dim=-1,
        )

    def _validate_rollout_inputs(self, history_states: torch.Tensor, future_torques: torch.Tensor) -> None:
        if history_states.ndim != 3:
            raise ValueError("history_states must have shape (batch, history_len, state_dim).")
        if history_states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {history_states.shape[-1]}.")
        if future_torques.ndim != 3:
            raise ValueError("future_torques must have shape (batch, horizon, torque_dim).")
        if future_torques.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {future_torques.shape[-1]}.")


class _FiLMStateMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("FiLM MLP requires latent_dim > 0.")
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.film_layers = nn.ModuleList([FiLMLayer(hidden_dim, latent_dim) for _ in range(depth)])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for layer in self.film_layers:
            h = layer(h, z)
        return self.output(h)


class WholeMLPWMDynamics(nn.Module):
    """Pure MLP whole-state rollout with the same public contract as WholeDeLaNWMDynamics."""

    def __init__(
        self,
        *,
        robot_dof: int = 9,
        torque_dim: int = 9,
        hidden_dim: int = 256,
        dt: float = 0.02,
        history_len: int = 1,
        ode_solver: str = "euler",
        use_context_encoder: bool = False,
        latent_dim: int = 8,
        context_encoder_hidden_dim: int = 256,
        context_encoder_depth: int = 2,
        delan_use_film: bool = False,
        delan_film_depth: int = 2,
        mlp_depth: int = 3,
        privileged_collision_obs_dim: int = 0,
    ) -> None:
        super().__init__()
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        self.robot_dof = int(robot_dof)
        self.object_dof = 6
        self.whole_dof = self.robot_dof + self.object_dof
        self.privileged_collision_obs_dim = int(privileged_collision_obs_dim)
        self.state_dim = 2 * self.robot_dof + 13 + self.privileged_collision_obs_dim
        self.torque_dim = int(torque_dim)
        self.history_len = int(history_len)
        self.dt = float(dt)
        self.ode_solver = validate_ode_solver(ode_solver)
        self.use_context_encoder = bool(use_context_encoder)
        self.delan_use_film = bool(delan_use_film)
        self.latent_dim = int(latent_dim) if self.use_context_encoder else 0
        self.model_type = "MLP"
        if self.delan_use_film and not self.use_context_encoder:
            raise ValueError("delan_use_film=True requires use_context_encoder=True.")

        latent_input_dim = self.latent_dim if self.use_context_encoder and not self.delan_use_film else 0
        input_dim = self.history_len * self.state_dim + self.torque_dim + latent_input_dim
        if self.delan_use_film:
            self.net = _FiLMStateMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.state_dim,
                depth=delan_film_depth,
                latent_dim=latent_dim,
            )
        else:
            self.net = build_mlp(input_dim, hidden_dim, self.state_dim, depth=mlp_depth)

        if self.use_context_encoder:
            seq_input_dim = self.history_len * (self.state_dim + self.torque_dim)
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
        del object_context
        self._validate_rollout_inputs(history_states, future_torques)
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

        predictions: list[torch.Tensor] = []
        aux_steps: list[dict[str, torch.Tensor]] = []
        for step in range(future_torques.shape[1]):
            torque = future_torques[:, step]
            state, aux = self.step(history_states, torque, z=z, return_aux=return_aux)
            history_states = torch.cat([history_states[:, 1:], state[:, None]], dim=1)
            predictions.append(state)
            if return_aux:
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

    def step(
        self,
        history_states: torch.Tensor,
        torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if torque.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {torque.shape[-1]}.")
        parts = [history_states.flatten(start_dim=1), torque]
        if self.use_context_encoder and not self.delan_use_film:
            if z is None:
                raise ValueError("WholeMLPWMDynamics requires latent z.")
            parts.append(z)
        x = torch.cat(parts, dim=-1)
        if self.delan_use_film and z is None:
            raise ValueError("WholeMLPWMDynamics FiLM requires latent z.")
        raw_state = self.net(x, z) if self.delan_use_film else self.net(x)
        next_state = self._normalize_state(raw_state)
        aux: dict[str, torch.Tensor] = {}
        if return_aux:
            aux["next_state_raw"] = raw_state
        return next_state, aux

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

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        offset = 2 * self.robot_dof
        return torch.cat(
            [
                state[:, : offset + 3],
                normalize_quat(state[:, offset + 3 : offset + 7]),
                state[:, offset + 7 :],
            ],
            dim=-1,
        )

    def _validate_rollout_inputs(self, history_states: torch.Tensor, future_torques: torch.Tensor) -> None:
        if history_states.ndim != 3:
            raise ValueError("history_states must have shape (batch, history_len, state_dim).")
        if history_states.shape[1] != self.history_len:
            raise ValueError(f"Expected history_len {self.history_len}, got {history_states.shape[1]}.")
        if history_states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {history_states.shape[-1]}.")
        if future_torques.ndim != 3:
            raise ValueError("future_torques must have shape (batch, horizon, torque_dim).")
        if future_torques.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {future_torques.shape[-1]}.")


def quat_to_rotvec(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    quat = normalize_quat(quat, eps=eps)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    w = quat[..., :1].clamp(-1.0 + eps, 1.0 - eps)
    xyz = quat[..., 1:]
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, w)
    scale = torch.where(xyz_norm > eps, angle / xyz_norm.clamp_min(eps), 2.0 * torch.ones_like(xyz_norm))
    return xyz * scale


def rotvec_to_quat(rotvec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > eps,
        torch.sin(half_angle) / angle.clamp_min(eps),
        0.5 - angle.square() / 48.0,
    )
    quat = torch.cat([torch.cos(half_angle), scale * rotvec], dim=-1)
    return normalize_quat(quat, eps=eps)


def build_whole_wm_dynamics(cfg: WholeWMDynamicsConfig = WholeWMDynamicsConfig()) -> WholeDeLaNWMDynamics:
    return WholeDeLaNWMDynamics(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        hidden_dim=cfg.hidden_dim,
        dt=cfg.dt,
        ode_solver=cfg.ode_solver,
        history_len=cfg.history_len,
        use_context_encoder=cfg.use_context_encoder,
        latent_dim=cfg.latent_dim,
        context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
        context_encoder_depth=cfg.context_encoder_depth,
        delan_use_film=cfg.delan_use_film,
        delan_film_depth=cfg.delan_film_depth,
        delan_use_history=cfg.delan_use_history,
        mlp_depth=cfg.mlp_depth,
        privileged_collision_obs_dim=cfg.privileged_collision_obs_dim,
    )


def build_whole_mlp_wm_dynamics(cfg: WholeWMDynamicsConfig = WholeWMDynamicsConfig()) -> WholeMLPWMDynamics:
    return WholeMLPWMDynamics(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        hidden_dim=cfg.hidden_dim,
        dt=cfg.dt,
        ode_solver=cfg.ode_solver,
        history_len=cfg.history_len,
        use_context_encoder=cfg.use_context_encoder,
        latent_dim=cfg.latent_dim,
        context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
        context_encoder_depth=cfg.context_encoder_depth,
        delan_use_film=cfg.delan_use_film,
        delan_film_depth=cfg.delan_film_depth,
        mlp_depth=cfg.mlp_depth,
        privileged_collision_obs_dim=cfg.privileged_collision_obs_dim,
    )
