"""Robot + object MCGDF world model.

Extends the robot-only MCGDF design with a structured contact and object
dynamics block, addressing the mismatches identified in Section 4.3 of
``../delan_vs_physx_verification.tex``:

* (#1, #5, #6) A single learned contact wrench ``F_c in R^6`` couples robot
  and object via the analytic gripper Jacobian (Newton's third law) and the
  object dynamics follow Newton-Euler with the world-frame inertia
  ``I_w(r_o) = R(r_o) I_b R(r_o)^T``.
* (#4) Quaternion integration uses the exponential map
  ``q_{t+1} = q_t * exp(0.5 * omega * dt)``.
* (#7) Only ``v_dot^o`` is supervised on the object linear side; the
  linearly dependent ``F_ext`` target is dropped.
* (#2, Approach A) The implicit residual target for ``r_theta`` is shifted by
  the dataset's pre-contact baseline so the contact-wrench head has a clean
  target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RobotObjectMCGDFStateSpec:
    robot_dof: int = 9
    torque_dim: int = 9
    object_state_dim: int = 13
    object_context_dim: int = 13

    @property
    def state_dim(self) -> int:
        return 2 * self.robot_dof + self.object_state_dim


# ----------------------------------------------------------------------------
# Generic building blocks (reused from robot-only MCGDF)
# ----------------------------------------------------------------------------

def mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(depth):
        layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


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
                act_deriv = self.silu_derivative(h)
                h = F.silu(h)
                jac = act_deriv.unsqueeze(-1) * jac
            else:
                raise TypeError(f"Unsupported derivative layer: {type(layer).__name__}")
        return h, jac


class DeLaNStructuredTerms(nn.Module):
    """DeLaN-structured inertia/Coriolis/gravity block.

    By default (``use_film=False``) the three sub-networks are the original
    plain MLP / DerivativeMLP heads with no context dependence, exactly
    matching the pre-FiLM behaviour.  When ``use_film=True`` and
    ``latent_dim`` is a positive int, ``g_net``/``l_diag_net``/
    ``l_offdiag_net`` are replaced by FiLM-modulated variants conditioned
    on the per-episode latent ``z``; the forward methods then require
    ``z`` to be passed in.
    """

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
        self.use_film = bool(use_film) and (latent_dim is not None) and (latent_dim > 0)
        self.latent_dim = latent_dim if self.use_film else None
        if not self.use_film:
            self.g_net = mlp(robot_dof, hidden_dim, robot_dof, depth=2)
            self.l_diag_net = DerivativeMLP(robot_dof, hidden_dim, robot_dof, depth=2)
            self.l_offdiag_net = DerivativeMLP(robot_dof, hidden_dim, n_offdiag, depth=2)
        else:
            self.g_net = FiLMDerivativeMLP(
                robot_dof, hidden_dim, robot_dof, depth=film_depth, latent_dim=latent_dim,
            )
            self.l_diag_net = FiLMDerivativeMLP(
                robot_dof, hidden_dim, robot_dof, depth=film_depth, latent_dim=latent_dim,
            )
            self.l_offdiag_net = FiLMDerivativeMLP(
                robot_dof, hidden_dim, n_offdiag, depth=film_depth, latent_dim=latent_dim,
            )
        offdiag = torch.tril_indices(row=robot_dof, col=robot_dof, offset=-1)
        diag = torch.arange(robot_dof)
        self.register_buffer("offdiag_row", offdiag[0], persistent=False)
        self.register_buffer("offdiag_col", offdiag[1], persistent=False)
        self.register_buffer("diag_idx", diag, persistent=False)

    def make_l_and_derivatives(self, q, z: torch.Tensor | None = None):
        batch = q.shape[0]
        if self.use_film:
            if z is None:
                raise ValueError("DeLaNStructuredTerms.use_film=True requires latent z.")
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

    def inertia(self, q, z: torch.Tensor | None = None):
        L, dL_dq, l_diag, l_offdiag = self.make_l_and_derivatives(q, z=z)
        H = L @ L.transpose(-1, -2)
        eye = torch.eye(self.robot_dof, device=q.device, dtype=q.dtype).unsqueeze(0)
        return H + self.eps * eye, L, dL_dq, l_diag, l_offdiag

    def coriolis_centrifugal(self, L, dL_dq, dq):
        dL_dt = torch.einsum("bijk,bk->bij", dL_dq, dq)
        dH_dt = L @ dL_dt.transpose(-1, -2) + dL_dt @ L.transpose(-1, -2)
        dH_dt_dq = torch.einsum("bij,bj->bi", dH_dt, dq)
        dH_dq = (
            torch.einsum("bim,bjmk->bijk", L, dL_dq)
            + torch.einsum("bimk,bjm->bijk", dL_dq, L)
        )
        kinetic_grad = torch.einsum("bi,bijk,bj->bk", dq, dH_dq, dq)
        return dH_dt_dq - 0.5 * kinetic_grad

    def forward(self, q, dq, z: torch.Tensor | None = None):
        H, L, dL_dq, l_diag, l_offdiag = self.inertia(q, z=z)
        if self.use_film:
            if z is None:
                raise ValueError("DeLaNStructuredTerms.use_film=True requires latent z.")
            g = self.g_net(q, z)
        else:
            g = self.g_net(q)
        coriolis = self.coriolis_centrifugal(L, dL_dq, dq)
        return {"H": H, "L": L, "dL_dq": dL_dq, "l_diag": l_diag, "l_offdiag": l_offdiag,
                "g": g, "coriolis": coriolis}


class FiLMLayer(nn.Module):
    """A linear layer modulated by FiLM parameters from a latent ``z``.

    Mirrors ``scripts/world_model/Ver1/models.FiLMLayer`` but uses SiLU to
    stay consistent with the rest of the MCGDF code.  Each layer is one
    Linear+FiLM+activation block:

        h = SiLU( gamma(z) * Linear(x) + beta(z) )

    where ``gamma`` and ``beta`` are tiny linear maps from ``z``.
    """

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film_gamma = nn.Linear(latent_dim, hidden_dim)
        self.film_beta = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        gamma = self.film_gamma(z)
        beta = self.film_beta(z)
        return F.silu(gamma * h + beta)


class FiLMDerivativeMLP(nn.Module):
    """FiLM-modulated drop-in replacement for ``DerivativeMLP``.

    Architecture mirrors the FiLM heads used elsewhere (an input projection,
    then ``depth`` FiLM blocks, then an output linear):

        h = SiLU( Linear_in(x) )
        for i in 1..depth:
            h = SiLU( gamma_i(z) * Linear_i(h) + beta_i(z) )
        out = Linear_out(h)

    ``forward_with_jacobian(x, z)`` returns ``(out, dout/dx)`` with ``z``
    treated as fixed w.r.t. ``x``.  This is the DeLaN identity's
    requirement: the inertia heads need ``dL/dq``, and ``z`` is a
    per-episode latent that does not depend on ``q``, so it factors as a
    constant inside the chain rule.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Linear(in_dim, hidden_dim)
        self.film_layers = nn.ModuleList(
            [FiLMLayer(hidden_dim, latent_dim) for _ in range(depth)]
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.trunk(x))
        for layer in self.film_layers:
            h = layer(h, z)
        return self.out(h)

    def forward_with_jacobian(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, in_dim = x.shape
        h = x
        jac = torch.eye(in_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, in_dim, in_dim)

        # Trunk: Linear -> SiLU
        h_pre = self.trunk(h)
        jac = torch.einsum("oi,bik->bok", self.trunk.weight, jac)
        act_deriv = DerivativeMLP.silu_derivative(h_pre)
        h = F.silu(h_pre)
        jac = act_deriv.unsqueeze(-1) * jac

        # FiLM blocks: Linear -> *gamma(z) + beta(z) -> SiLU
        for layer in self.film_layers:
            h_lin = layer.linear(h)
            jac = torch.einsum("oi,bik->bok", layer.linear.weight, jac)
            gamma = layer.film_gamma(z)
            beta = layer.film_beta(z)
            h_pre = gamma * h_lin + beta
            jac = gamma.unsqueeze(-1) * jac
            act_deriv = DerivativeMLP.silu_derivative(h_pre)
            h = F.silu(h_pre)
            jac = act_deriv.unsqueeze(-1) * jac

        # Output linear
        out = self.out(h)
        jac = torch.einsum("oi,bik->bok", self.out.weight, jac)
        return out, jac


class ResidualTorqueHead(nn.Module):
    """Implicit-PD residual head with optional FiLM modulation by ``z``.

    When ``latent_dim`` is ``None`` (the default) the head is the original
    plain MLP from the robot-only MCGDF design.  When ``latent_dim`` is an
    int, the architecture is

        in -> Linear -> SiLU  (trunk)
           -> FiLM_1(z) -> ... -> FiLM_d(z)
           -> Linear -> out

    matching ``scripts/world_model/Ver1`` in structure but with depth equal
    to ``depth``.  ``depth`` keeps the same meaning as in the plain branch.
    """

    def __init__(
        self,
        robot_dof: int,
        torque_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        in_dim = 2 * robot_dof + torque_dim
        self.use_film = latent_dim is not None and latent_dim > 0
        if not self.use_film:
            self.net = mlp(in_dim, hidden_dim, robot_dof, depth=depth)
        else:
            self.trunk = nn.Linear(in_dim, hidden_dim)
            self.film_layers = nn.ModuleList(
                [FiLMLayer(hidden_dim, latent_dim) for _ in range(depth)]
            )
            self.out = nn.Linear(hidden_dim, robot_dof)

    def forward(self, q, dq, torque, z: torch.Tensor | None = None):
        x = torch.cat([q, dq, torque], dim=-1)
        if not self.use_film:
            return self.net(x)
        h = F.silu(self.trunk(x))
        for layer in self.film_layers:
            h = layer(h, z)
        return self.out(h)


class JointDampingFrictionParams(nn.Module):
    def __init__(self, robot_dof: int, learnable: bool = False,
                 init_damping: float = 0.0, init_friction: float = 0.0) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        self.learnable = learnable
        if learnable:
            self.raw_damping = nn.Parameter(torch.full((robot_dof,), float(init_damping)))
            self.raw_friction = nn.Parameter(torch.full((robot_dof,), float(init_friction)))
        else:
            self.register_parameter("raw_damping", None)
            self.register_parameter("raw_friction", None)

    def resolve(self, dataset_damping, dataset_friction):
        if not self.learnable:
            return dataset_damping, dataset_friction
        damping = F.softplus(self.raw_damping).unsqueeze(0).expand_as(dataset_damping)
        friction = F.softplus(self.raw_friction).unsqueeze(0).expand_as(dataset_friction)
        return damping, friction


# ----------------------------------------------------------------------------
# Quaternion utilities (wxyz convention)
# ----------------------------------------------------------------------------

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dim=-1)


def quat_exp_integrate(q: torch.Tensor, ang_vel_w: torch.Tensor, dt: float) -> torch.Tensor:
    """Closed-form exp-map quaternion integration (verification mismatch #4).

    ``q_{t+1} = q_t * exp((1/2) omega dt)``, where the exponential of the pure
    vector quaternion ``[0, v]`` equals ``[cos|v|, (sin|v|/|v|) v]``.
    """
    half_angle = 0.5 * ang_vel_w * dt
    angle = half_angle.norm(dim=-1, keepdim=True)
    small = angle < 1.0e-6
    safe_angle = angle.clamp_min(1.0e-9)
    factor = torch.where(small, 1.0 - (angle ** 2) / 6.0, torch.sin(safe_angle) / safe_angle)
    dq_w = torch.where(small, 1.0 - (angle ** 2) / 2.0, torch.cos(safe_angle))
    dq_xyz = factor * half_angle
    dq = torch.cat([dq_w, dq_xyz], dim=-1)
    return quat_normalize(quat_multiply_wxyz(q, dq))


def quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz unit quaternion to 3x3 rotation matrix, batched (B, 3, 3)."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


# ----------------------------------------------------------------------------
# Franka analytic forward kinematics + geometric Jacobian
# ----------------------------------------------------------------------------

def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _xyz_rpy(xyz, rpy):
    return _rpy_to_rot(*rpy), torch.tensor(list(xyz), dtype=torch.float32)


class FrankaForwardKinematics(nn.Module):
    """Differentiable batched Franka FK and geometric body Jacobian.

    Mirrors the kinematic chain of
    ``../GT_dynamics/eval_utils.franka_joint_and_gripper_positions``.  All
    seven arm joints are revolute about the local z-axis; the two finger
    joints do not move the gripper tip frame, so columns 7-8 of the body
    Jacobian are identically zero.
    """

    NUM_ARM = 7

    def __init__(self, robot_dof: int = 9, tool_z_offset: float = 0.1034) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        link_transforms = [
            _xyz_rpy((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
            _xyz_rpy((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
            _xyz_rpy((0.0, -0.316, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            _xyz_rpy((0.0825, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            _xyz_rpy((-0.0825, 0.384, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
            _xyz_rpy((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
            _xyz_rpy((0.088, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
        ]
        R_fixed = torch.stack([t[0] for t in link_transforms], dim=0)
        p_fixed = torch.stack([t[1] for t in link_transforms], dim=0)
        self.register_buffer("R_fixed", R_fixed, persistent=False)
        self.register_buffer("p_fixed", p_fixed, persistent=False)
        R_hand, p_hand = _xyz_rpy((0.0, 0.0, 0.107), (0.0, 0.0, -math.pi / 4.0))
        R_tool, p_tool = _xyz_rpy((0.0, 0.0, tool_z_offset), (0.0, 0.0, 0.0))
        R_grip = R_hand @ R_tool
        p_grip = R_hand @ p_tool + p_hand
        self.register_buffer("R_grip", R_grip, persistent=False)
        self.register_buffer("p_grip", p_grip, persistent=False)

    def _arm_chain(self, q: torch.Tensor):
        batch = q.shape[0]
        device, dtype = q.device, q.dtype
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, 3, 3).contiguous()
        p = torch.zeros(batch, 3, device=device, dtype=dtype)
        R_joint_origins: list[torch.Tensor] = []
        p_joint_origins: list[torch.Tensor] = []
        for i in range(self.NUM_ARM):
            R_fixed_i = self.R_fixed[i].to(dtype=dtype)
            p_fixed_i = self.p_fixed[i].to(dtype=dtype)
            p = p + torch.einsum("bij,j->bi", R, p_fixed_i)
            R = R @ R_fixed_i
            R_joint_origins.append(R)
            p_joint_origins.append(p)
            cos_q = torch.cos(q[:, i])
            sin_q = torch.sin(q[:, i])
            R_q = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
            R_q[:, 0, 0] = cos_q
            R_q[:, 0, 1] = -sin_q
            R_q[:, 1, 0] = sin_q
            R_q[:, 1, 1] = cos_q
            R_q[:, 2, 2] = 1.0
            R = R @ R_q
        R_grip = self.R_grip.to(dtype=dtype)
        p_grip = self.p_grip.to(dtype=dtype)
        p_ee = p + torch.einsum("bij,j->bi", R, p_grip)
        R_ee = R @ R_grip
        return R_ee, p_ee, R_joint_origins, p_joint_origins

    def forward(self, q: torch.Tensor):
        R_ee, p_ee, _, _ = self._arm_chain(q[:, : self.NUM_ARM])
        return p_ee, R_ee

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        R_ee, p_ee, R_joint_origins, p_joint_origins = self._arm_chain(q[:, : self.NUM_ARM])
        batch = q.shape[0]
        device, dtype = q.device, q.dtype
        J = torch.zeros(batch, 6, self.robot_dof, device=device, dtype=dtype)
        for i in range(self.NUM_ARM):
            R_i = R_joint_origins[i]
            p_i = p_joint_origins[i]
            z_i = R_i[:, :, 2]
            r = p_ee - p_i
            v = torch.cross(z_i, r, dim=-1)
            J[:, :3, i] = v
            J[:, 3:, i] = z_i
        return J


# ----------------------------------------------------------------------------
# Contact wrench head
# ----------------------------------------------------------------------------

class ContactWrenchHead(nn.Module):
    """Predict the gripper-to-cube contact wrench F_c in (B, 6).

    First 3 outputs: linear force on the cube (world frame).
    Last 3 outputs: torque on the cube about its centre of mass (world frame).
    Robot-side reaction torque is then ``tau_contact = -J_c(q)^T F_c``.

    Optional FiLM modulation by a latent context ``z`` (same architecture as
    ``ResidualTorqueHead``).  The known per-episode physical context ``xi``
    remains a concatenated input so ``z`` only needs to absorb residual,
    unrecorded variation.
    """

    def __init__(self, state_dim: int, torque_dim: int, context_dim: int,
                 hidden_dim: int = 128, depth: int = 3,
                 latent_dim: int | None = None) -> None:
        super().__init__()
        in_dim = state_dim + torque_dim + context_dim
        self.use_film = latent_dim is not None and latent_dim > 0
        if not self.use_film:
            self.net = mlp(in_dim, hidden_dim, 6, depth=depth)
        else:
            self.trunk = nn.Linear(in_dim, hidden_dim)
            self.film_layers = nn.ModuleList(
                [FiLMLayer(hidden_dim, latent_dim) for _ in range(depth)]
            )
            self.out = nn.Linear(hidden_dim, 6)

    def forward(self, state, torque, context, z: torch.Tensor | None = None):
        x = torch.cat([state, torque, context], dim=-1)
        if not self.use_film:
            return self.net(x)
        h = F.silu(self.trunk(x))
        for layer in self.film_layers:
            h = layer(h, z)
        return self.out(h)


# ----------------------------------------------------------------------------
# Object next-state head (MCGDF_object_mlp variant)
# ----------------------------------------------------------------------------

class ObjectStateMLP(nn.Module):
    """Predict the next object substate purely from (state, torque, z).

    This is the *whole* object side of the MCGDF_object_mlp variant:
    Newton-Euler is removed entirely and replaced by a single MLP that
    emits the 13-D next object substate directly.

    Architecture (same FiLM pattern as ``ResidualTorqueHead`` and
    ``ContactWrenchHead``):

        in   = cat(state, torque) in R^{state_dim + torque_dim}
        h    = SiLU(Linear(in))
        h    = FiLM_1(h, z) -> ... -> FiLM_d(h, z)
        out  = Linear(h)   in R^{13}

    The output 13 dims are interpreted as
    ``(p^o_next (3), r^o_next_raw (4), v^o_next (3), omega^o_next (3))``
    and the caller normalizes the quaternion component before assembling
    the next state.  When ``latent_dim`` is None or 0 the head falls back
    to a plain MLP (no FiLM, no z input), matching the convention used
    elsewhere in this file.
    """

    def __init__(
        self,
        state_dim: int,
        torque_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.torque_dim = int(torque_dim)
        in_dim = self.state_dim + self.torque_dim
        self.use_film = latent_dim is not None and latent_dim > 0
        if not self.use_film:
            self.net = mlp(in_dim, hidden_dim, 13, depth=depth)
        else:
            self.trunk = nn.Linear(in_dim, hidden_dim)
            self.film_layers = nn.ModuleList(
                [FiLMLayer(hidden_dim, latent_dim) for _ in range(depth)]
            )
            self.out = nn.Linear(hidden_dim, 13)

    def forward(
        self,
        state: torch.Tensor,
        torque: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([state, torque], dim=-1)
        if not self.use_film:
            return self.net(x)
        h = F.silu(self.trunk(x))
        for layer in self.film_layers:
            h = layer(h, z)
        return self.out(h)


# ----------------------------------------------------------------------------
# Variational context encoder + auxiliary heads
# ----------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """Variational encoder ``q_phi(z | history)``.

    Input is the flattened history window of ``(state, torque)`` pairs of
    length ``K+1``; outputs the mean and log-variance of a diagonal Gaussian
    of latent size ``latent_dim``.

    The MLP backbone uses SiLU activations to stay consistent with the rest
    of the MCGDF code, although Ver1 uses ReLU; either choice is fine.
    """

    def __init__(self, seq_input_dim: int, hidden_dim: int, latent_dim: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = seq_input_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, flat_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(flat_seq)
        return self.mu_head(h), self.logvar_head(h)


class ContextRegressionHead(nn.Module):
    """Optional small head ``hat_xi(z)`` against the dataset physics labels."""

    def __init__(self, latent_dim: int, context_target_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(latent_dim, context_target_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(z|x) || N(0, I)) over the batch."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def info_nce_soft(
    embedding: torch.Tensor,
    context_value: torch.Tensor,
    temperature: float = 0.1,
    context_similarity_sigma: float = 1.0,
    nce_negative_weight: float = 1.0,
) -> torch.Tensor:
    """Soft-weighted InfoNCE based on continuous context-distance.

    Verbatim from ``scripts/world_model/Ver1/wm_train.py``.  Returns zero
    when the batch has fewer than two samples or when ``context_value`` has
    zero feature dimension.
    """
    if embedding.shape[0] < 2 or context_value.shape[-1] == 0:
        return torch.zeros((), device=embedding.device, dtype=embedding.dtype)

    z = F.normalize(embedding, dim=-1)
    logits = (z @ z.transpose(0, 1)) / temperature
    bsz = logits.shape[0]
    eye = torch.eye(bsz, dtype=torch.bool, device=logits.device)

    ctx = context_value
    ctx = (ctx - ctx.mean(dim=0, keepdim=True)) / (ctx.std(dim=0, keepdim=True) + 1e-6)
    ctx_dist2 = torch.sum((ctx.unsqueeze(1) - ctx.unsqueeze(0)).pow(2), dim=-1)
    w_pos = torch.exp(-ctx_dist2 / (2.0 * context_similarity_sigma * context_similarity_sigma + 1e-12))
    w_pos = w_pos.masked_fill(eye, 0.0)
    w_neg = (1.0 - w_pos).masked_fill(eye, 0.0)

    losses: list[torch.Tensor] = []
    for i in range(bsz):
        row_logits = logits[i]
        exp_logits = torch.exp(row_logits)
        num = torch.sum(w_pos[i] * exp_logits)
        den = num + nce_negative_weight * torch.sum(w_neg[i] * exp_logits)
        if num.item() <= 0.0 or den.item() <= 0.0:
            continue
        loss_i = -torch.log(num / den)
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.zeros((), device=embedding.device, dtype=embedding.dtype)
    return torch.stack(losses).mean()


# ----------------------------------------------------------------------------
# One-step robot+object MCGDF dynamics
# ----------------------------------------------------------------------------

class RobotObjectMCGDFStep(nn.Module):
    """One step of the robot+object MCGDF forward dynamics."""

    def __init__(
        self,
        robot_dof: int = 9,
        torque_dim: int = 9,
        object_context_dim: int = 13,
        hidden_dim: int = 256,
        residual_hidden_dim: int = 128,
        residual_depth: int = 2,
        contact_hidden_dim: int = 128,
        contact_depth: int = 3,
        object_mlp_hidden_dim: int = 256,
        object_mlp_depth: int = 3,
        dt: float = 0.02,
        inertia_eps: float = 1.0e-4,
        friction_eps: float = 1.0e-3,
        learn_damping_friction: bool = False,
        init_damping: float = 0.0,
        init_friction: float = 0.0,
        omit_damping: bool = True,
        tool_z_offset: float = 0.1034,
        latent_dim: int | None = None,
        delan_use_film: bool = False,
        delan_film_depth: int = 2,
    ) -> None:
        super().__init__()
        if torque_dim != robot_dof:
            raise ValueError("This MCGDF model expects torque_dim == robot_dof.")
        self.robot_dof = robot_dof
        self.torque_dim = torque_dim
        self.object_context_dim = object_context_dim
        self.object_state_dim = 13
        self.state_dim = 2 * robot_dof + self.object_state_dim
        self.dt = dt
        self.friction_eps = friction_eps
        self.omit_damping = omit_damping
        # When ``latent_dim`` is None the residual and contact heads stay in
        # their original (plain MLP) form, so the model behaves exactly like
        # the pre-FiLM MCGDF.
        self.latent_dim = latent_dim
        # ``delan_use_film`` is opt-in: when False the DeLaN heads (H, c, g)
        # remain plain (context-free) exactly as in the pre-FiLM design.
        self.delan_use_film = bool(delan_use_film) and (latent_dim is not None) and (latent_dim > 0)

        self.structured_terms = DeLaNStructuredTerms(
            robot_dof=robot_dof, hidden_dim=hidden_dim, eps=inertia_eps,
            latent_dim=latent_dim if self.delan_use_film else None,
            use_film=self.delan_use_film,
            film_depth=delan_film_depth,
        )
        self.residual_head = ResidualTorqueHead(
            robot_dof=robot_dof, torque_dim=torque_dim,
            hidden_dim=residual_hidden_dim, depth=residual_depth,
            latent_dim=latent_dim,
        )
        self.joint_params = JointDampingFrictionParams(
            robot_dof=robot_dof, learnable=learn_damping_friction,
            init_damping=init_damping, init_friction=init_friction,
        )
        self.fk = FrankaForwardKinematics(robot_dof=robot_dof, tool_z_offset=tool_z_offset)
        self.contact_head = ContactWrenchHead(
            state_dim=self.state_dim, torque_dim=torque_dim, context_dim=object_context_dim,
            hidden_dim=contact_hidden_dim, depth=contact_depth,
            latent_dim=latent_dim,
        )
        # Object-side head: replaces the analytic Newton-Euler integrator
        # in the original MCGDF.  Predicts the full 13-D next object
        # substate directly from (state, torque) with FiLM modulation by
        # the latent z.  See ``ObjectStateMLP``.
        self.object_state_mlp = ObjectStateMLP(
            state_dim=self.state_dim, torque_dim=torque_dim,
            hidden_dim=object_mlp_hidden_dim, depth=object_mlp_depth,
            latent_dim=latent_dim,
        )

    def split_state(self, state: torch.Tensor):
        n = self.robot_dof
        q = state[:, :n]
        dq = state[:, n : 2 * n]
        offset = 2 * n
        object_pos = state[:, offset : offset + 3]
        object_quat = quat_normalize(state[:, offset + 3 : offset + 7])
        object_lin_vel = state[:, offset + 7 : offset + 10]
        object_ang_vel = state[:, offset + 10 : offset + 13]
        return q, dq, object_pos, object_quat, object_lin_vel, object_ang_vel

    def forward(
        self,
        state: torch.Tensor,
        torque: torch.Tensor,
        damping: torch.Tensor,
        friction: torch.Tensor,
        object_context: torch.Tensor,
        z: torch.Tensor | None = None,
    ):
        q, dq, object_pos, object_quat, object_lin_vel, object_ang_vel = self.split_state(state)
        terms = self.structured_terms(q, dq, z=z if self.delan_use_film else None)
        H, g, coriolis = terms["H"], terms["g"], terms["coriolis"]

        damping_eff, friction_eff = self.joint_params.resolve(damping, friction)
        damping_torque = damping_eff * dq
        friction_torque = friction_eff * torch.tanh(dq / self.friction_eps)
        damping_subtract = torch.zeros_like(damping_torque) if self.omit_damping else damping_torque

        # ---- Robot side: unchanged from MCGDF ------------------------------
        # Contact wrench F_c is still predicted so the robot side gets the
        # action-reaction torque tau_contact = -J_c^T F_c via Newton's
        # third law.  F_c does NOT enter the object integrator here -- the
        # object side is purely the MLP below.
        F_c = self.contact_head(state, torque, object_context, z=z)
        J_c = self.fk.jacobian(q)
        tau_contact = -torch.einsum("bji,bj->bi", J_c, F_c)

        residual = self.residual_head(q, dq, torque, z=z)

        effective_torque = torque + residual - damping_subtract - friction_torque + tau_contact
        rhs = (effective_torque - g - coriolis).unsqueeze(-1)
        ddq = torch.linalg.solve(H, rhs).squeeze(-1)
        inertial = (H @ ddq.unsqueeze(-1)).squeeze(-1)

        # Robot kinematic integration (semi-implicit Euler) ------------------
        dq_next = dq + ddq * self.dt
        q_next = q + dq_next * self.dt

        # ---- Object side: 100% MLP (no Newton-Euler, no F=ma) --------------
        # The 13-D output is interpreted as (p_o_next, r_o_next_raw,
        # v_o_next, omega_o_next); the quaternion component is normalized
        # to unit length before assembly.
        object_state_pred = self.object_state_mlp(state, torque, z=z)
        object_pos_next      = object_state_pred[:, 0:3]
        object_quat_next_raw = object_state_pred[:, 3:7]
        object_lin_vel_next  = object_state_pred[:, 7:10]
        object_ang_vel_next  = object_state_pred[:, 10:13]
        object_quat_next = quat_normalize(object_quat_next_raw)

        next_state = torch.cat(
            [q_next, dq_next, object_pos_next, object_quat_next, object_lin_vel_next, object_ang_vel_next], dim=-1
        )

        if self.omit_damping:
            full_inv_dyn_tau = inertial + coriolis + g + friction_torque - residual - tau_contact
        else:
            full_inv_dyn_tau = inertial + coriolis + g + damping_torque + friction_torque - residual - tau_contact

        # Finite-difference accelerations on the predicted next velocities,
        # so existing downstream losses keyed on ``object_lin_acc`` /
        # ``object_ang_acc`` continue to work.  These are diagnostics,
        # NOT physically derived quantities -- the integrator above did
        # not produce them via F = m a.
        lin_acc_fd = (object_lin_vel_next - object_lin_vel) / self.dt
        ang_acc_fd = (object_ang_vel_next - object_ang_vel) / self.dt
        gravity_acc = torch.zeros_like(F_c[:, :3])
        gravity_acc[:, 2] = -9.81
        zero_force = torch.zeros_like(F_c[:, :3])

        aux = {
            "H": H, "L": terms["L"], "l_diag": terms["l_diag"], "l_offdiag": terms["l_offdiag"],
            "g": g, "coriolis": coriolis,
            "torque": torque, "ddq": ddq, "inertial": inertial,
            "inverse_dynamics_tau": inertial + coriolis + g,
            "damping_coeff": damping_eff, "friction_coeff": friction_eff,
            "damping_torque": damping_torque, "damping_torque_applied": damping_subtract,
            "friction_torque": friction_torque, "residual_torque": residual,
            "contact_wrench": F_c, "contact_wrench_lin": F_c[:, :3], "contact_wrench_ang": F_c[:, 3:],
            "tau_contact": tau_contact,
            # Object side: FD diagnostics; not used inside the integrator.
            "object_lin_acc": lin_acc_fd, "object_ang_acc": ang_acc_fd,
            "object_gravity_acc": gravity_acc,
            "object_inertial_force": zero_force,
            "object_gravity_force": zero_force,
            "object_state_pred_raw": object_state_pred,
            "object_quat_next_raw": object_quat_next_raw,
            "full_inverse_dynamics_tau": full_inv_dyn_tau,
        }
        return next_state, aux


class MultiStepRobotObjectMCGDFWorldModel(nn.Module):
    """Recursive open-loop rollout, optionally with a context-encoder latent ``z``.

    When ``use_context_encoder=False`` (default), the model behaves exactly as
    the pre-FiLM MCGDF.  When ``use_context_encoder=True`` and ``latent_dim``
    is a positive integer, a variational context encoder is built; its output
    ``z`` is sampled once per forward pass and broadcast to every rollout
    step, so that the same per-episode context modulates each substep.
    """

    def __init__(
        self,
        robot_dof: int = 9,
        torque_dim: int = 9,
        object_context_dim: int = 13,
        hidden_dim: int = 256,
        residual_hidden_dim: int = 128,
        residual_depth: int = 2,
        contact_hidden_dim: int = 128,
        contact_depth: int = 3,
        object_mlp_hidden_dim: int = 256,
        object_mlp_depth: int = 3,
        dt: float = 0.02,
        inertia_eps: float = 1.0e-4,
        friction_eps: float = 1.0e-3,
        learn_damping_friction: bool = False,
        init_damping: float = 0.0,
        init_friction: float = 0.0,
        omit_damping: bool = True,
        tool_z_offset: float = 0.1034,
        use_context_encoder: bool = False,
        latent_dim: int = 8,
        context_encoder_hidden_dim: int = 256,
        context_encoder_depth: int = 2,
        history_step_dim: int = 0,
        history_len: int = 0,
        context_target_dim: int = 0,
        delan_use_film: bool = False,
        delan_film_depth: int = 2,
    ) -> None:
        super().__init__()
        self.spec = RobotObjectMCGDFStateSpec(
            robot_dof=robot_dof, torque_dim=torque_dim,
            object_state_dim=13, object_context_dim=object_context_dim,
        )
        self.omit_damping = omit_damping
        self.use_context_encoder = bool(use_context_encoder)
        self.latent_dim = latent_dim if self.use_context_encoder else None
        self.history_step_dim = history_step_dim
        self.history_len = history_len
        self.context_target_dim = context_target_dim
        # Opt-in FiLM on the DeLaN (H, c, g) heads.  Requires the context
        # encoder so that there is a latent ``z`` to modulate by.
        if delan_use_film and not self.use_context_encoder:
            raise ValueError(
                "delan_use_film=True requires use_context_encoder=True so that "
                "the DeLaN heads have a latent z to condition on."
            )
        self.delan_use_film = bool(delan_use_film)
        self.delan_film_depth = int(delan_film_depth)

        if self.use_context_encoder:
            if history_step_dim <= 0 or history_len <= 0:
                raise ValueError(
                    "use_context_encoder=True requires history_step_dim and history_len > 0."
                )
            self.context_encoder = ContextEncoder(
                seq_input_dim=history_step_dim * history_len,
                hidden_dim=context_encoder_hidden_dim,
                latent_dim=latent_dim,
                depth=context_encoder_depth,
            )
            if context_target_dim > 0:
                self.context_regression_head = ContextRegressionHead(
                    latent_dim=latent_dim, context_target_dim=context_target_dim
                )
            else:
                self.context_regression_head = None
        else:
            self.context_encoder = None
            self.context_regression_head = None

        self.dynamics = RobotObjectMCGDFStep(
            robot_dof=robot_dof, torque_dim=torque_dim, object_context_dim=object_context_dim,
            hidden_dim=hidden_dim,
            residual_hidden_dim=residual_hidden_dim, residual_depth=residual_depth,
            contact_hidden_dim=contact_hidden_dim, contact_depth=contact_depth,
            object_mlp_hidden_dim=object_mlp_hidden_dim,
            object_mlp_depth=object_mlp_depth,
            dt=dt, inertia_eps=inertia_eps, friction_eps=friction_eps,
            learn_damping_friction=learn_damping_friction,
            init_damping=init_damping, init_friction=init_friction,
            omit_damping=omit_damping, tool_z_offset=tool_z_offset,
            latent_dim=self.latent_dim,
            delan_use_film=self.delan_use_film,
            delan_film_depth=self.delan_film_depth,
        )

    def encode_context(
        self,
        history_states: torch.Tensor,
        history_torques: torch.Tensor | None,
        deterministic_context: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Run the context encoder. Returns (mu, logvar, z, hat_xi) or None tuple."""
        if not self.use_context_encoder:
            return None, None, None, None
        if history_torques is None:
            raise ValueError("use_context_encoder=True requires history_torques in the batch.")
        flat_states = history_states.flatten(start_dim=1)
        flat_torques = history_torques.flatten(start_dim=1)
        flat_seq = torch.cat([flat_states, flat_torques], dim=-1)
        mu, logvar = self.context_encoder(flat_seq)
        z = mu if deterministic_context else reparameterize(mu, logvar)
        hat_xi = self.context_regression_head(z) if self.context_regression_head is not None else None
        return mu, logvar, z, hat_xi

    def forward(
        self,
        history_states: torch.Tensor,
        future_torques: torch.Tensor,
        joint_params: dict[str, torch.Tensor],
        object_context: torch.Tensor,
        history_torques: torch.Tensor | None = None,
        deterministic_context: bool = False,
        return_aux: bool = False,
        return_context: bool = False,
    ):
        """Forward rollout.

        Default return signature preserves the pre-FiLM contract:
          - ``return_aux=False``: ``pred_future``
          - ``return_aux=True``: ``(pred_future, aux_list)``

        When the new context encoder is in use, pass ``return_context=True``
        to additionally receive the variational outputs:
          - ``return_aux=False, return_context=True``: ``(pred_future, context)``
          - ``return_aux=True,  return_context=True``: ``(pred_future, aux_list, context)``

        where ``context`` is a dict ``{"mu", "logvar", "z", "hat_xi"}``.
        """
        state = history_states[:, -1]
        damping = joint_params["damping"]
        friction = joint_params["friction"]
        mu, logvar, z, hat_xi = self.encode_context(
            history_states, history_torques, deterministic_context=deterministic_context
        )
        preds: list[torch.Tensor] = []
        aux_list: list[dict[str, torch.Tensor]] = []
        for h in range(future_torques.shape[1]):
            state, aux = self.dynamics(
                state, future_torques[:, h], damping, friction, object_context, z=z,
            )
            preds.append(state)
            if return_aux:
                aux_list.append(aux)
        pred_future = torch.stack(preds, dim=1)
        context_outputs = {"mu": mu, "logvar": logvar, "z": z, "hat_xi": hat_xi}
        if return_aux and return_context:
            return pred_future, aux_list, context_outputs
        if return_aux:
            return pred_future, aux_list
        if return_context:
            return pred_future, context_outputs
        return pred_future


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------

def stack_aux(aux_list: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([aux[key] for aux in aux_list], dim=1)


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
):
    n = robot_dof
    p, t = pred, target
    q_mse = F.mse_loss(p[..., :n], t[..., :n])
    dq_mse = F.mse_loss(p[..., n : 2 * n], t[..., n : 2 * n])
    offset = 2 * n
    op_mse = F.mse_loss(p[..., offset : offset + 3], t[..., offset : offset + 3])
    qr_pred = p[..., offset + 3 : offset + 7]
    qr_targ = t[..., offset + 3 : offset + 7]
    quat_mse = torch.minimum(
        (qr_pred - qr_targ).pow(2).sum(dim=-1),
        (qr_pred + qr_targ).pow(2).sum(dim=-1),
    ).mean()
    ov_mse = F.mse_loss(p[..., offset + 7 : offset + 10], t[..., offset + 7 : offset + 10])
    ow_mse = F.mse_loss(p[..., offset + 10 : offset + 13], t[..., offset + 10 : offset + 13])
    total = (
        q_weight * q_mse + dq_weight * dq_mse
        + object_pos_weight * op_mse + object_quat_weight * quat_mse
        + object_lin_vel_weight * ov_mse + object_ang_vel_weight * ow_mse
    )
    return total, {
        "q_mse": q_mse.detach(), "dq_mse": dq_mse.detach(),
        "object_pos_mse": op_mse.detach(), "object_quat_mse": quat_mse.detach(),
        "object_lin_vel_mse": ov_mse.detach(), "object_ang_vel_mse": ow_mse.detach(),
    }


def supervised_robot_dynamics_loss(
    aux_list,
    future_robot_dynamics: dict[str, torch.Tensor],
    lambda_mass_matrix: float = 0.0,
    lambda_inertial: float = 0.0,
    lambda_coriolis_gt: float = 0.0,
    lambda_gravity: float = 0.0,
    lambda_qdd: float = 0.0,
    lambda_inverse_dynamics: float = 0.0,
):
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
    terms = [
        ("mass_matrix_mse", lambda_mass_matrix, pred_h, future_robot_dynamics["mass_matrix"]),
        ("inertial_mse", lambda_inertial, pred_inertial, future_robot_dynamics["inertial"]),
        ("coriolis_gt_mse", lambda_coriolis_gt, pred_coriolis, future_robot_dynamics["coriolis"]),
        ("gravity_mse", lambda_gravity, pred_gravity, future_robot_dynamics["gravity"]),
        ("qdd_mse", lambda_qdd, pred_qdd, future_robot_dynamics["qdd"]),
        ("inverse_dynamics_mse", lambda_inverse_dynamics, pred_inv, future_robot_dynamics["inverse_dynamics_tau"]),
    ]
    for name, w, pred, target in terms:
        mse = F.mse_loss(pred, target)
        metrics[name] = mse.detach()
        total = total + w * mse
    return total, metrics


def supervised_object_dynamics_loss(
    aux_list,
    future_object_dynamics: dict[str, torch.Tensor],
    lambda_object_lin_acc: float = 0.0,
    lambda_object_ang_acc: float = 0.0,
):
    """Object acceleration supervision.  The redundant ``F_ext`` target
    (mismatch #7) is intentionally omitted."""
    if not aux_list:
        zero = torch.tensor(0.0)
        return zero, {}
    device = aux_list[0]["object_lin_acc"].device
    dtype = aux_list[0]["object_lin_acc"].dtype
    pred_lin = stack_aux(aux_list, "object_lin_acc")
    pred_ang = stack_aux(aux_list, "object_ang_acc")
    target_lin = future_object_dynamics["root_lin_acc_w"]
    target_ang = future_object_dynamics["root_ang_acc_w"]
    lin_mse = F.mse_loss(pred_lin, target_lin)
    ang_mse = F.mse_loss(pred_ang, target_ang)
    total = torch.zeros((), device=device, dtype=dtype)
    total = total + lambda_object_lin_acc * lin_mse + lambda_object_ang_acc * ang_mse
    return total, {"object_lin_acc_mse": lin_mse.detach(), "object_ang_acc_mse": ang_mse.detach()}


def damping_friction_supervised_loss(
    aux_list,
    joint_params: dict[str, torch.Tensor],
    lambda_damping: float = 0.0,
    lambda_friction: float = 0.0,
):
    if not aux_list:
        zero = torch.tensor(0.0)
        return zero, {"damping_mse": zero, "friction_mse": zero}
    device = aux_list[0]["damping_coeff"].device
    dtype = aux_list[0]["damping_coeff"].dtype
    target_damping = joint_params["damping"].unsqueeze(1).expand_as(stack_aux(aux_list, "damping_coeff"))
    target_friction = joint_params["friction"].unsqueeze(1).expand_as(stack_aux(aux_list, "friction_coeff"))
    pred_damping = stack_aux(aux_list, "damping_coeff")
    pred_friction = stack_aux(aux_list, "friction_coeff")
    damping_mse = F.mse_loss(pred_damping, target_damping)
    friction_mse = F.mse_loss(pred_friction, target_friction)
    total = torch.zeros((), device=device, dtype=dtype)
    total = total + lambda_damping * damping_mse + lambda_friction * friction_mse
    return total, {"damping_mse": damping_mse.detach(), "friction_mse": friction_mse.detach()}


def residual_supervised_loss(
    aux_list,
    future_robot_dynamics: dict[str, torch.Tensor],
    future_torques: torch.Tensor,
    future_states: torch.Tensor,
    joint_params: dict[str, torch.Tensor],
    residual_baseline: torch.Tensor,
    omit_damping: bool,
    friction_eps: float,
    robot_dof: int,
    lambda_residual_supervised: float = 0.0,
):
    """Direct supervision of r_theta with pre-contact baseline removal."""
    if not aux_list or lambda_residual_supervised <= 0.0:
        zero = torch.tensor(0.0)
        return zero, {"residual_sup_mse": zero}
    pred_residual = stack_aux(aux_list, "residual_torque")
    inv_dyn_gt = future_robot_dynamics["inverse_dynamics_tau"]
    qdot = future_states[..., robot_dof : 2 * robot_dof]
    d = joint_params["damping"].unsqueeze(1)
    f = joint_params["friction"].unsqueeze(1)
    d_term = torch.zeros_like(qdot) if omit_damping else d * qdot
    f_term = f * torch.tanh(qdot / friction_eps)
    target = inv_dyn_gt + d_term + f_term - future_torques
    target = target - residual_baseline.unsqueeze(1)
    mse = F.mse_loss(pred_residual, target)
    return lambda_residual_supervised * mse, {"residual_sup_mse": mse.detach()}


def auxiliary_regularization(
    aux_list,
    lambda_coriolis: float = 0.0,
    lambda_residual: float = 0.0,
    lambda_contact: float = 0.0,
):
    if not aux_list:
        zero = torch.tensor(0.0)
        return zero, {"coriolis_reg": zero, "residual_reg": zero, "contact_reg": zero}
    device = aux_list[0]["coriolis"].device
    dtype = aux_list[0]["coriolis"].dtype
    coriolis_reg = torch.stack([aux["coriolis"].pow(2).mean() for aux in aux_list]).mean()
    residual_reg = torch.stack([aux["residual_torque"].pow(2).mean() for aux in aux_list]).mean()
    contact_reg = torch.stack([aux["contact_wrench"].pow(2).mean() for aux in aux_list]).mean()
    total = torch.zeros((), device=device, dtype=dtype)
    total = total + lambda_coriolis * coriolis_reg + lambda_residual * residual_reg + lambda_contact * contact_reg
    return total, {
        "coriolis_reg": coriolis_reg.detach(),
        "residual_reg": residual_reg.detach(),
        "contact_reg": contact_reg.detach(),
    }
