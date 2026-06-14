from __future__ import annotations

import math

import torch
from torch import nn


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = in_dim
    for _ in range(depth):
        layers += [nn.Linear(last_dim, hidden_dim), nn.SiLU()]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


def normalize_quat(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    w, xyz = quat[..., :1], quat[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


def quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
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


def integrate_quat(quat: torch.Tensor, ang_vel: torch.Tensor, dt: float) -> torch.Tensor:
    half_angle = 0.5 * ang_vel * dt
    angle = half_angle.norm(dim=-1, keepdim=True)
    small = angle < 1.0e-6
    safe_angle = angle.clamp_min(1.0e-9)
    scale = torch.where(small, 1.0 - angle.square() / 6.0, torch.sin(safe_angle) / safe_angle)
    delta_w = torch.where(small, 1.0 - angle.square() / 2.0, torch.cos(safe_angle))
    delta = torch.cat([delta_w, scale * half_angle], dim=-1)
    return normalize_quat(quat_mul(quat, delta))


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat(quat)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def safe_solve(matrix: torch.Tensor, rhs: torch.Tensor, damping: float = 1.0e-6) -> torch.Tensor:
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.solve(matrix + damping * eye, rhs.unsqueeze(-1)).squeeze(-1)


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    return _rpy_to_rot(*rpy), torch.tensor(list(xyz), dtype=torch.float32)


def transform_from_xyz_rpy(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    R, p = _xyz_rpy(xyz, rpy)
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = R.to(device=device, dtype=dtype)
    transform[:3, 3] = p.to(device=device, dtype=dtype)
    return transform


def rot_z(theta: torch.Tensor) -> torch.Tensor:
    batch = theta.shape[0]
    transform = torch.eye(4, device=theta.device, dtype=theta.dtype).expand(batch, 4, 4).clone()
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    transform[:, 0, 0] = cos_theta
    transform[:, 0, 1] = -sin_theta
    transform[:, 1, 0] = sin_theta
    transform[:, 1, 1] = cos_theta
    return transform


class FrankaForwardKinematics(nn.Module):
    """Differentiable Franka gripper FK and geometric Jacobian."""

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
        self.register_buffer("R_fixed", torch.stack([item[0] for item in link_transforms]), persistent=False)
        self.register_buffer("p_fixed", torch.stack([item[1] for item in link_transforms]), persistent=False)
        R_hand, p_hand = _xyz_rpy((0.0, 0.0, 0.107), (0.0, 0.0, -math.pi / 4.0))
        R_tool, p_tool = _xyz_rpy((0.0, 0.0, tool_z_offset), (0.0, 0.0, 0.0))
        self.register_buffer("R_grip", R_hand @ R_tool, persistent=False)
        self.register_buffer("p_grip", R_hand @ p_tool + p_hand, persistent=False)

    def _arm_chain(self, q: torch.Tensor):
        batch = q.shape[0]
        device, dtype = q.device, q.dtype
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, 3, 3).contiguous()
        p = torch.zeros(batch, 3, device=device, dtype=dtype)
        R_joint_origins: list[torch.Tensor] = []
        p_joint_origins: list[torch.Tensor] = []

        for idx in range(self.NUM_ARM):
            R_fixed = self.R_fixed[idx].to(device=device, dtype=dtype)
            p_fixed = self.p_fixed[idx].to(device=device, dtype=dtype)
            p = p + torch.einsum("bij,j->bi", R, p_fixed)
            R = R @ R_fixed
            R_joint_origins.append(R)
            p_joint_origins.append(p)

            cos_q = torch.cos(q[:, idx])
            sin_q = torch.sin(q[:, idx])
            R_q = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
            R_q[:, 0, 0] = cos_q
            R_q[:, 0, 1] = -sin_q
            R_q[:, 1, 0] = sin_q
            R_q[:, 1, 1] = cos_q
            R_q[:, 2, 2] = 1.0
            R = R @ R_q

        R_grip = self.R_grip.to(device=device, dtype=dtype)
        p_grip = self.p_grip.to(device=device, dtype=dtype)
        p_ee = p + torch.einsum("bij,j->bi", R, p_grip)
        R_ee = R @ R_grip
        return R_ee, p_ee, R_joint_origins, p_joint_origins

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        R_ee, p_ee, _, _ = self._arm_chain(q[:, : self.NUM_ARM])
        return p_ee, R_ee

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        _, p_ee, R_joint_origins, p_joint_origins = self._arm_chain(q[:, : self.NUM_ARM])
        batch = q.shape[0]
        J = torch.zeros(batch, 6, self.robot_dof, device=q.device, dtype=q.dtype)
        for idx in range(self.NUM_ARM):
            z_axis = R_joint_origins[idx][:, :, 2]
            radius = p_ee - p_joint_origins[idx]
            J[:, :3, idx] = torch.cross(z_axis, radius, dim=-1)
            J[:, 3:, idx] = z_axis
        return J
