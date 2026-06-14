from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RobotObjectGTDynamicsStateSpec:
    robot_dof: int = 9
    torque_dim: int = 9
    object_context_dim: int = 13

    @property
    def robot_state_dim(self) -> int:
        return 2 * self.robot_dof

    @property
    def object_state_dim(self) -> int:
        return 13

    @property
    def state_dim(self) -> int:
        return self.robot_state_dim + self.object_state_dim


def mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(depth):
        layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class DerivativeMLP(nn.Sequential):
    """MLP that propagates output derivatives w.r.t. the input."""

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
                act_deriv = self.silu_derivative(h)
                h = F.silu(h)
                jac = act_deriv.unsqueeze(-1) * jac
            else:
                raise TypeError(f"Unsupported derivative layer: {type(layer).__name__}")
        return h, jac


class DeLaNStructuredTerms(nn.Module):
    """Predict robot g(q), L(q), and construct H(q) and c(q,dq)."""

    def __init__(self, robot_dof: int, hidden_dim: int, eps: float = 1.0e-4, diag_eps: float = 1.0e-3) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        self.eps = eps
        self.diag_eps = diag_eps
        n_offdiag = robot_dof * (robot_dof - 1) // 2
        self.g_net = mlp(robot_dof, hidden_dim, robot_dof, depth=2)
        self.l_diag_net = DerivativeMLP(robot_dof, hidden_dim, robot_dof, depth=2)
        self.l_offdiag_net = DerivativeMLP(robot_dof, hidden_dim, n_offdiag, depth=2)
        offdiag = torch.tril_indices(row=robot_dof, col=robot_dof, offset=-1)
        diag = torch.arange(robot_dof)
        self.register_buffer("offdiag_row", offdiag[0], persistent=False)
        self.register_buffer("offdiag_col", offdiag[1], persistent=False)
        self.register_buffer("diag_idx", diag, persistent=False)

    def make_l_and_derivatives(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = q.shape[0]
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

    def inertia(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L, dL_dq, l_diag, l_offdiag = self.make_l_and_derivatives(q)
        H = L @ L.transpose(-1, -2)
        eye = torch.eye(self.robot_dof, device=q.device, dtype=q.dtype).unsqueeze(0)
        return H + self.eps * eye, L, dL_dq, l_diag, l_offdiag

    def coriolis_centrifugal(self, L: torch.Tensor, dL_dq: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        dL_dt = torch.einsum("bijk,bk->bij", dL_dq, dq)
        dH_dt = L @ dL_dt.transpose(-1, -2) + dL_dt @ L.transpose(-1, -2)
        dH_dt_dq = torch.einsum("bij,bj->bi", dH_dt, dq)
        dH_dq = (
            torch.einsum("bim,bjmk->bijk", L, dL_dq)
            + torch.einsum("bimk,bjm->bijk", dL_dq, L)
        )
        kinetic_grad = torch.einsum("bi,bijk,bj->bk", dq, dH_dq, dq)
        return dH_dt_dq - 0.5 * kinetic_grad

    def forward(self, q: torch.Tensor, dq: torch.Tensor) -> dict[str, torch.Tensor]:
        H, L, dL_dq, l_diag, l_offdiag = self.inertia(q)
        g = self.g_net(q)
        coriolis = self.coriolis_centrifugal(L, dL_dq, dq)
        return {
            "H": H,
            "L": L,
            "dL_dq": dL_dq,
            "l_diag": l_diag,
            "l_offdiag": l_offdiag,
            "g": g,
            "coriolis": coriolis,
        }


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def integrate_quat_wxyz(quat: torch.Tensor, ang_vel_w: torch.Tensor, dt: float) -> torch.Tensor:
    omega_quat = torch.cat([torch.zeros_like(ang_vel_w[:, :1]), ang_vel_w], dim=-1)
    quat_dot = 0.5 * quat_multiply_wxyz(omega_quat, quat)
    return normalize_quat(quat + quat_dot * dt)


class RobotObjectGTDynamicsStep(nn.Module):
    """Hybrid one-step model with robot DeLaN terms and learned object/contact coupling."""

    def __init__(
        self,
        robot_dof: int = 9,
        torque_dim: int = 9,
        object_context_dim: int = 13,
        hidden_dim: int = 256,
        dt: float = 0.02,
        inertia_eps: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if torque_dim != robot_dof:
            raise ValueError("This GT-dynamics model expects torque_dim == robot_dof.")
        self.robot_dof = robot_dof
        self.torque_dim = torque_dim
        self.object_context_dim = object_context_dim
        self.object_state_dim = 13
        self.state_dim = 2 * robot_dof + self.object_state_dim
        self.dt = dt
        self.structured_terms = DeLaNStructuredTerms(robot_dof=robot_dof, hidden_dim=hidden_dim, eps=inertia_eps)
        coupling_input_dim = self.state_dim + torque_dim + object_context_dim
        self.residual_torque_net = mlp(coupling_input_dim, hidden_dim, robot_dof, depth=3)
        self.object_acc_net = mlp(coupling_input_dim, hidden_dim, 6, depth=3)

    def split_state(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = state[:, : self.robot_dof]
        dq = state[:, self.robot_dof : 2 * self.robot_dof]
        object_start = 2 * self.robot_dof
        object_pos = state[:, object_start : object_start + 3]
        object_quat = normalize_quat(state[:, object_start + 3 : object_start + 7])
        object_lin_vel = state[:, object_start + 7 : object_start + 10]
        object_ang_vel = state[:, object_start + 10 : object_start + 13]
        return q, dq, object_pos, object_quat, object_lin_vel, object_ang_vel

    def forward(
        self,
        state: torch.Tensor,
        torque: torch.Tensor,
        object_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape[-1]}")
        if torque.shape[-1] != self.torque_dim:
            raise ValueError(f"Expected torque dim {self.torque_dim}, got {torque.shape[-1]}")

        q, dq, object_pos, object_quat, object_lin_vel, object_ang_vel = self.split_state(state)
        coupling_input = torch.cat([state, torque, object_context], dim=-1)
        tau_residual = self.residual_torque_net(coupling_input)
        object_acc = self.object_acc_net(coupling_input)
        object_lin_acc = object_acc[:, :3]
        object_ang_acc = object_acc[:, 3:]

        terms = self.structured_terms(q, dq)
        H = terms["H"]
        g = terms["g"]
        coriolis = terms["coriolis"]

        effective_torque = torque + tau_residual
        rhs = (effective_torque - g - coriolis).unsqueeze(-1)
        ddq = torch.linalg.solve(H, rhs).squeeze(-1)
        inertial = (H @ ddq.unsqueeze(-1)).squeeze(-1)

        dq_next = dq + ddq * self.dt
        q_next = q + dq_next * self.dt
        object_lin_vel_next = object_lin_vel + object_lin_acc * self.dt
        object_ang_vel_next = object_ang_vel + object_ang_acc * self.dt
        object_pos_next = object_pos + object_lin_vel_next * self.dt
        object_quat_next = integrate_quat_wxyz(object_quat, object_ang_vel_next, self.dt)

        next_state = torch.cat(
            [q_next, dq_next, object_pos_next, object_quat_next, object_lin_vel_next, object_ang_vel_next],
            dim=-1,
        )

        mass = object_context[:, :1]
        gravity_acc = torch.zeros_like(object_lin_acc)
        gravity_acc[:, 2] = -9.81
        object_inertial_force = mass * object_lin_acc
        object_gravity_force = mass * gravity_acc
        object_external_force = object_inertial_force - object_gravity_force

        free_inverse_dynamics_tau = inertial + coriolis + g
        aux = {
            "H": H,
            "L": terms["L"],
            "l_diag": terms["l_diag"],
            "l_offdiag": terms["l_offdiag"],
            "g": g,
            "coriolis": coriolis,
            "torque": torque,
            "tau_residual": tau_residual,
            "effective_torque": effective_torque,
            "ddq": ddq,
            "inertial": inertial,
            "inverse_dynamics_tau": free_inverse_dynamics_tau,
            "object_lin_acc": object_lin_acc,
            "object_ang_acc": object_ang_acc,
            "object_inertial_force": object_inertial_force,
            "object_gravity_force": object_gravity_force,
            "object_external_force": object_external_force,
        }
        return next_state, aux


class MultiStepRobotObjectGTDynamicsWorldModel(nn.Module):
    """Recursive open-loop rollout using recorded future torques and fixed object context."""

    def __init__(
        self,
        robot_dof: int = 9,
        torque_dim: int = 9,
        object_context_dim: int = 13,
        hidden_dim: int = 256,
        dt: float = 0.02,
        inertia_eps: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.spec = RobotObjectGTDynamicsStateSpec(
            robot_dof=robot_dof,
            torque_dim=torque_dim,
            object_context_dim=object_context_dim,
        )
        self.dynamics = RobotObjectGTDynamicsStep(
            robot_dof=robot_dof,
            torque_dim=torque_dim,
            object_context_dim=object_context_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            inertia_eps=inertia_eps,
        )

    def forward(
        self,
        history_states: torch.Tensor,
        future_torques: torch.Tensor,
        object_context: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        state = history_states[:, -1]
        preds: list[torch.Tensor] = []
        aux_list: list[dict[str, torch.Tensor]] = []
        for h in range(future_torques.shape[1]):
            state, aux = self.dynamics(state, future_torques[:, h], object_context)
            preds.append(state)
            if return_aux:
                aux_list.append(aux)
        pred_future = torch.stack(preds, dim=1)
        if return_aux:
            return pred_future, aux_list
        return pred_future


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    robot_state_dim = 2 * robot_dof
    q_loss = F.mse_loss(pred[..., :robot_dof], target[..., :robot_dof])
    dq_loss = F.mse_loss(pred[..., robot_dof:robot_state_dim], target[..., robot_dof:robot_state_dim])
    object_pos_loss = F.mse_loss(pred[..., robot_state_dim : robot_state_dim + 3], target[..., robot_state_dim : robot_state_dim + 3])
    pred_quat = normalize_quat(pred[..., robot_state_dim + 3 : robot_state_dim + 7])
    target_quat = normalize_quat(target[..., robot_state_dim + 3 : robot_state_dim + 7])
    object_quat_loss = torch.minimum(
        F.mse_loss(pred_quat, target_quat, reduction="none").mean(dim=-1),
        F.mse_loss(pred_quat, -target_quat, reduction="none").mean(dim=-1),
    ).mean()
    object_lin_vel_loss = F.mse_loss(
        pred[..., robot_state_dim + 7 : robot_state_dim + 10],
        target[..., robot_state_dim + 7 : robot_state_dim + 10],
    )
    object_ang_vel_loss = F.mse_loss(
        pred[..., robot_state_dim + 10 : robot_state_dim + 13],
        target[..., robot_state_dim + 10 : robot_state_dim + 13],
    )
    loss = (
        q_weight * q_loss
        + dq_weight * dq_loss
        + object_pos_weight * object_pos_loss
        + object_quat_weight * object_quat_loss
        + object_lin_vel_weight * object_lin_vel_loss
        + object_ang_vel_weight * object_ang_vel_loss
    )
    return loss, {
        "q_mse": q_loss.detach(),
        "dq_mse": dq_loss.detach(),
        "object_pos_mse": object_pos_loss.detach(),
        "object_quat_mse": object_quat_loss.detach(),
        "object_lin_vel_mse": object_lin_vel_loss.detach(),
        "object_ang_vel_mse": object_ang_vel_loss.detach(),
    }


def stack_aux(aux_list: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([aux[key] for aux in aux_list], dim=1)


def supervised_robot_dynamics_loss(
    aux_list: list[dict[str, torch.Tensor]],
    future_robot_dynamics: dict[str, torch.Tensor],
    future_torques: torch.Tensor,
    lambda_mass_matrix: float = 0.0,
    lambda_inertial: float = 0.0,
    lambda_coriolis_gt: float = 0.0,
    lambda_gravity: float = 0.0,
    lambda_qdd: float = 0.0,
    lambda_inverse_dynamics: float = 0.0,
    lambda_residual_torque: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not aux_list:
        zero = torch.tensor(0.0)
        return zero, {}

    device = aux_list[0]["H"].device
    dtype = aux_list[0]["H"].dtype
    total = torch.zeros((), device=device, dtype=dtype)
    metrics: dict[str, torch.Tensor] = {}

    pred_h = stack_aux(aux_list, "H")
    pred_inertial = stack_aux(aux_list, "inertial")
    pred_coriolis = stack_aux(aux_list, "coriolis")
    pred_gravity = stack_aux(aux_list, "g")
    pred_qdd = stack_aux(aux_list, "ddq")
    pred_inv = stack_aux(aux_list, "inverse_dynamics_tau")
    pred_residual = stack_aux(aux_list, "tau_residual")
    target_residual = future_robot_dynamics["inverse_dynamics_tau"] - future_torques

    terms = [
        ("mass_matrix_mse", lambda_mass_matrix, pred_h, future_robot_dynamics["mass_matrix"]),
        ("inertial_mse", lambda_inertial, pred_inertial, future_robot_dynamics["inertial"]),
        ("coriolis_gt_mse", lambda_coriolis_gt, pred_coriolis, future_robot_dynamics["coriolis"]),
        ("gravity_mse", lambda_gravity, pred_gravity, future_robot_dynamics["gravity"]),
        ("qdd_mse", lambda_qdd, pred_qdd, future_robot_dynamics["qdd"]),
        ("inverse_dynamics_mse", lambda_inverse_dynamics, pred_inv, future_robot_dynamics["inverse_dynamics_tau"]),
        ("residual_torque_mse", lambda_residual_torque, pred_residual, target_residual),
    ]
    for name, weight, pred, target in terms:
        mse = F.mse_loss(pred, target)
        metrics[name] = mse.detach()
        total = total + weight * mse
    return total, metrics


def supervised_object_dynamics_loss(
    aux_list: list[dict[str, torch.Tensor]],
    future_object_dynamics: dict[str, torch.Tensor],
    lambda_object_lin_acc: float = 0.0,
    lambda_object_ang_acc: float = 0.0,
    lambda_object_external_force: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not aux_list:
        zero = torch.tensor(0.0)
        return zero, {}

    device = aux_list[0]["object_lin_acc"].device
    dtype = aux_list[0]["object_lin_acc"].dtype
    total = torch.zeros((), device=device, dtype=dtype)
    metrics: dict[str, torch.Tensor] = {}
    terms = [
        (
            "object_lin_acc_mse",
            lambda_object_lin_acc,
            stack_aux(aux_list, "object_lin_acc"),
            future_object_dynamics["root_lin_acc_w"],
        ),
        (
            "object_ang_acc_mse",
            lambda_object_ang_acc,
            stack_aux(aux_list, "object_ang_acc"),
            future_object_dynamics["root_ang_acc_w"],
        ),
        (
            "object_external_force_mse",
            lambda_object_external_force,
            stack_aux(aux_list, "object_external_force"),
            future_object_dynamics["external_force_est_w"],
        ),
    ]
    for name, weight, pred, target in terms:
        mse = F.mse_loss(pred, target)
        metrics[name] = mse.detach()
        total = total + weight * mse
    return total, metrics


def auxiliary_regularization(
    aux_list: list[dict[str, torch.Tensor]],
    lambda_coriolis: float = 0.0,
    lambda_residual_l2: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not aux_list:
        zero = torch.tensor(0.0)
        return zero, {"coriolis_reg": zero, "residual_torque_reg": zero}
    device = aux_list[0]["coriolis"].device
    dtype = aux_list[0]["coriolis"].dtype
    coriolis_reg = torch.stack([aux["coriolis"].pow(2).mean() for aux in aux_list]).mean()
    residual_reg = torch.stack([aux["tau_residual"].pow(2).mean() for aux in aux_list]).mean()
    total = torch.zeros((), device=device, dtype=dtype)
    total = total + lambda_coriolis * coriolis_reg + lambda_residual_l2 * residual_reg
    return total, {"coriolis_reg": coriolis_reg.detach(), "residual_torque_reg": residual_reg.detach()}
