from __future__ import annotations

import numpy as np


def estimate_object_velocity(object_pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(object_pos, dtype=np.float32)
    if object_pos.shape[0] <= 1:
        return vel
    vel[0] = (object_pos[1] - object_pos[0]) / dt
    vel[-1] = (object_pos[-1] - object_pos[-2]) / dt
    if object_pos.shape[0] > 2:
        vel[1:-1] = (object_pos[2:] - object_pos[:-2]) / (2.0 * dt)
    return vel.astype(np.float32)


def first_object_motion_timestep(
    object_pos: np.ndarray,
    dt: float,
    displacement_threshold: float,
    velocity_threshold: float,
    consecutive_steps: int,
    settle_steps: int = 5,
) -> int | None:
    if object_pos.shape[0] == 0:
        return None
    baseline_idx = min(max(0, settle_steps), object_pos.shape[0] - 1)
    disp = np.linalg.norm(object_pos - object_pos[baseline_idx : baseline_idx + 1], axis=-1)
    speed = np.linalg.norm(estimate_object_velocity(object_pos, dt), axis=-1)
    moving = (disp > displacement_threshold) | (speed > velocity_threshold)
    moving[:baseline_idx] = False
    if consecutive_steps <= 1:
        idx = np.flatnonzero(moving)
        return int(idx[0]) if idx.size > 0 else None
    run = 0
    for i, flag in enumerate(moving):
        run = run + 1 if bool(flag) else 0
        if run >= consecutive_steps:
            return i - consecutive_steps + 1
    return None


