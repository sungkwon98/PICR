from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp, safe_solve

ODE_SOLVERS = ("euler", "rk4")


def validate_ode_solver(value: str) -> str:
    solver = value.lower()
    if solver not in ODE_SOLVERS:
        joined = ", ".join(ODE_SOLVERS)
        raise ValueError(f"ode_solver must be one of ({joined}); got {value!r}.")
    return solver


def safe_cholesky_solve(
    mass_matrix: torch.Tensor,
    rhs: torch.Tensor,
    damping: float = 1.0e-4,
) -> torch.Tensor:
    return safe_solve(mass_matrix, rhs, damping=damping)


@dataclass
class DelanTerms:
    H: torch.Tensor
    L: torch.Tensor
    dL_dq: torch.Tensor
    l_diag: torch.Tensor
    l_offdiag: torch.Tensor
    g: torch.Tensor
    coriolis: torch.Tensor

    def as_aux(self) -> dict[str, torch.Tensor]:
        return {
            "H": self.H,
            "L": self.L,
            "dL_dq": self.dL_dq,
            "l_diag": self.l_diag,
            "l_offdiag": self.l_offdiag,
            "g": self.g,
            "coriolis": self.coriolis,
        }


@dataclass
class RobotDynamicsOutput:
    next_q: torch.Tensor
    next_dq: torch.Tensor
    ddq: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class DerivativeMLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> None:
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        super().__init__(*layers)

    @staticmethod
    def silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    def forward_with_jacobian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, in_dim = x.shape
        h = x
        jac = torch.eye(in_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, in_dim, in_dim)
        for layer in self:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                jac = torch.einsum("oi,bik->bok", layer.weight, jac)
            elif isinstance(layer, nn.SiLU):
                deriv = self.silu_derivative(h)
                h = F.silu(h)
                jac = deriv.unsqueeze(-1) * jac
            else:
                raise TypeError(f"Unsupported derivative layer: {type(layer).__name__}")
        return h, jac


class FiLMLayer(nn.Module):
    """Linear layer modulated by FiLM parameters from latent context z."""

    def __init__(self, hidden_dim: int, latent_dim: int, modulation_limit: float = 0.1) -> None:
        super().__init__()
        self.modulation_limit = float(modulation_limit)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film_gamma = nn.Linear(latent_dim, hidden_dim)
        self.film_beta = nn.Linear(latent_dim, hidden_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        scale = 1.0 + self.modulation_limit * torch.tanh(self.film_gamma(z))
        shift = self.modulation_limit * torch.tanh(self.film_beta(z))
        return F.silu(scale * h + shift)


class FiLMDerivativeMLP(nn.Module):
    """DerivativeMLP replacement with FiLM blocks conditioned on latent z.

    The Jacobian is with respect to x only. z is treated as a fixed per-window
    latent, matching the DeLaN requirement for dL/dq.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Linear(in_dim, hidden_dim)
        self.film_layers = nn.ModuleList([FiLMLayer(hidden_dim, latent_dim) for _ in range(depth)])
        self.out = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.trunk(x))
        for layer in self.film_layers:
            h = layer(h, z)
        return self.out(h)

    def forward_with_jacobian(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, in_dim = x.shape
        pre = self.trunk(x)
        deriv = self.silu_derivative(pre)
        h = F.silu(pre)
        jac = torch.eye(in_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, in_dim, in_dim)
        jac = torch.einsum("oi,bik->bok", self.trunk.weight, jac)
        jac = deriv.unsqueeze(-1) * jac

        for layer in self.film_layers:
            linear_h = layer.linear(h)
            scale = 1.0 + layer.modulation_limit * torch.tanh(layer.film_gamma(z))
            shift = layer.modulation_limit * torch.tanh(layer.film_beta(z))
            pre = scale * linear_h + shift
            h = F.silu(pre)

            jac = torch.einsum("oi,bik->bok", layer.linear.weight, jac)
            jac = scale.unsqueeze(-1) * jac
            jac = self.silu_derivative(pre).unsqueeze(-1) * jac

        out = self.out(h)
        jac = torch.einsum("oi,bik->bok", self.out.weight, jac)
        return out, jac


class DeLaNCore(nn.Module):
    """DeLaN robot core producing H(q), c(q,dq), and g(q)."""

    def __init__(
        self,
        robot_dof: int,
        hidden_dim: int,
        eps: float = 1.0e-4,
        diag_eps: float = 1.0e-3,
        latent_dim: int | None = None,
        use_film: bool = False,
        film_depth: int = 2,
    ) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        self.eps = eps
        self.diag_eps = diag_eps
        n_offdiag = robot_dof * (robot_dof - 1) // 2
        self.use_film = bool(use_film) and latent_dim is not None and latent_dim > 0
        self.latent_dim = int(latent_dim) if self.use_film else None

        if self.use_film:
            self.v_net = FiLMDerivativeMLP(robot_dof, hidden_dim, 1, depth=film_depth, latent_dim=int(latent_dim))
            self.l_diag_net = FiLMDerivativeMLP(
                robot_dof,
                hidden_dim,
                robot_dof,
                depth=film_depth,
                latent_dim=int(latent_dim),
            )
            self.l_offdiag_net = FiLMDerivativeMLP(
                robot_dof,
                hidden_dim,
                n_offdiag,
                depth=film_depth,
                latent_dim=int(latent_dim),
            )
        else:
            self.v_net = DerivativeMLP(robot_dof, hidden_dim, 1, depth=2)
            self.l_diag_net = DerivativeMLP(robot_dof, hidden_dim, robot_dof, depth=2)
            self.l_offdiag_net = DerivativeMLP(robot_dof, hidden_dim, n_offdiag, depth=2)

        offdiag = torch.tril_indices(row=robot_dof, col=robot_dof, offset=-1)
        self.register_buffer("offdiag_row", offdiag[0], persistent=False)
        self.register_buffer("offdiag_col", offdiag[1], persistent=False)
        self.register_buffer("diag_idx", torch.arange(robot_dof), persistent=False)

    def make_l_and_derivatives(self, q: torch.Tensor, z: torch.Tensor | None = None):
        batch = q.shape[0]
        if self.use_film:
            if z is None:
                raise ValueError("DeLaNCore.use_film=True requires latent z.")
            l_diag_raw, dl_diag_raw_dq = self.l_diag_net.forward_with_jacobian(q, z)
            l_offdiag, dl_offdiag_dq = self.l_offdiag_net.forward_with_jacobian(q, z)
        else:
            l_diag_raw, dl_diag_raw_dq = self.l_diag_net.forward_with_jacobian(q)
            l_offdiag, dl_offdiag_dq = self.l_offdiag_net.forward_with_jacobian(q)

        L = q.new_zeros((batch, self.robot_dof, self.robot_dof))
        dL_dq = q.new_zeros((batch, self.robot_dof, self.robot_dof, self.robot_dof))
        L[:, self.offdiag_row, self.offdiag_col] = l_offdiag
        dL_dq[:, self.offdiag_row, self.offdiag_col, :] = dl_offdiag_dq

        l_diag = F.softplus(l_diag_raw) + self.diag_eps
        dl_diag_dq = torch.sigmoid(l_diag_raw).unsqueeze(-1) * dl_diag_raw_dq
        L[:, self.diag_idx, self.diag_idx] = l_diag
        dL_dq[:, self.diag_idx, self.diag_idx, :] = dl_diag_dq
        return L, dL_dq, l_diag, l_offdiag

    def inertia(self, q: torch.Tensor, z: torch.Tensor | None = None):
        L, dL_dq, l_diag, l_offdiag = self.make_l_and_derivatives(q, z=z)
        eye = torch.eye(self.robot_dof, device=q.device, dtype=q.dtype).unsqueeze(0)
        H = L @ L.transpose(-1, -2) + self.eps * eye
        return H, L, dL_dq, l_diag, l_offdiag

    @staticmethod
    def coriolis_centrifugal(L: torch.Tensor, dL_dq: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        dL_dt = torch.einsum("bijk,bk->bij", dL_dq, dq)
        dH_dt = L @ dL_dt.transpose(-1, -2) + dL_dt @ L.transpose(-1, -2)
        dH_dt_dq = torch.einsum("bij,bj->bi", dH_dt, dq)
        dH_dq = torch.einsum("bim,bjmk->bijk", L, dL_dq) + torch.einsum("bimk,bjm->bijk", dL_dq, L)
        kinetic_grad = torch.einsum("bi,bijk,bj->bk", dq, dH_dq, dq)
        return dH_dt_dq - 0.5 * kinetic_grad

    def forward(self, q: torch.Tensor, dq: torch.Tensor, z: torch.Tensor | None = None) -> DelanTerms:
        H, L, dL_dq, l_diag, l_offdiag = self.inertia(q, z=z)
        if self.use_film:
            if z is None:
                raise ValueError("DeLaNCore.use_film=True requires latent z.")
            V, dV_dq = self.v_net.forward_with_jacobian(q, z)
        else:
            V, dV_dq = self.v_net.forward_with_jacobian(q)
        coriolis = self.coriolis_centrifugal(L, dL_dq, dq)
        g= dV_dq.squeeze(1)
        return DelanTerms(H=H, L=L, dL_dq=dL_dq, l_diag=l_diag, l_offdiag=l_offdiag, g=g, coriolis=coriolis)


class DeLaNRobotDynamics(nn.Module):
    """Robot dynamics with no residual, damping/friction, or context branch."""

    def __init__(
        self,
        robot_dof: int = 9,
        torque_dim: int = 9,
        hidden_dim: int = 256,
        dt: float = 0.02,
        latent_dim: int | None = None,
        use_film: bool = False,
        film_depth: int = 2,
        ode_solver: str = "euler",
    ) -> None:
        super().__init__()
        if torque_dim != robot_dof:
            raise ValueError("DeLaNRobotDynamics expects torque_dim == robot_dof.")
        self.robot_dof = robot_dof
        self.torque_dim = torque_dim
        self.dt = dt
        self.ode_solver = validate_ode_solver(ode_solver)
        self.use_film = bool(use_film) and latent_dim is not None and latent_dim > 0
        self.core = DeLaNCore(
            robot_dof=robot_dof,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim if self.use_film else None,
            use_film=self.use_film,
            film_depth=film_depth,
        )

    def forward(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        torque: torch.Tensor,
        external_torque: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> RobotDynamicsOutput:
        external = torch.zeros_like(torque) if external_torque is None else external_torque
        effective_torque = torque + external
        ddq, terms = self.acceleration(q, dq, effective_torque, z=z)
        inertial = (terms.H @ ddq.unsqueeze(-1)).squeeze(-1)
        next_q, next_dq = self.integrate(q, dq, effective_torque, ddq=ddq, z=z)

        aux = terms.as_aux()
        aux.update(
            {
                "torque": torque,
                "external_torque": external,
                "effective_torque": effective_torque,
                "ddq": ddq,
                "inertial": inertial,
                "inverse_dynamics_tau": inertial + terms.coriolis + terms.g,
                "ode_solver": torch.tensor(0 if self.ode_solver == "euler" else 1, device=q.device),
            }
        )
        return RobotDynamicsOutput(next_q=next_q, next_dq=next_dq, ddq=ddq, aux=aux)

    def acceleration(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DelanTerms]:
        terms = self.core(q, dq, z=z if self.use_film else None)
        rhs = effective_torque - terms.g - terms.coriolis
        ddq = safe_solve(terms.H, rhs)
        ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3)
        ddq = ddq.clamp(-1e3, 1e3)
        return ddq, terms

    def state_derivative(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ddq, _terms = self.acceleration(q, dq, effective_torque, z=z)
        return dq, ddq

    def integrate(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        ddq: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ode_solver == "euler":
            next_dq = dq + ddq * self.dt
            next_q = q + next_dq * self.dt
            return next_q, next_dq
        return self._rk4_integrate(q, dq, effective_torque, z=z)

    def _rk4_integrate(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.dt
        k1_q, k1_dq = self.state_derivative(q, dq, effective_torque, z=z)
        k2_q, k2_dq = self.state_derivative(
            q + 0.5 * dt * k1_q,
            dq + 0.5 * dt * k1_dq,
            effective_torque,
            z=z,
        )
        k3_q, k3_dq = self.state_derivative(
            q + 0.5 * dt * k2_q,
            dq + 0.5 * dt * k2_dq,
            effective_torque,
            z=z,
        )
        k4_q, k4_dq = self.state_derivative(
            q + dt * k3_q,
            dq + dt * k3_dq,
            effective_torque,
            z=z,
        )
        next_q = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        next_dq = dq + (dt / 6.0) * (k1_dq + 2.0 * k2_dq + 2.0 * k3_dq + k4_dq)
        return next_q, next_dq


DeLaNStructuredTerms = DeLaNCore
