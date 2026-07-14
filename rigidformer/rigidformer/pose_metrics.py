from __future__ import annotations

from typing import Any

import numpy as np


def estimate_rigid_transform(
    reference_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate row-vector rigid transform target ~= reference @ rotation + translation."""

    reference_points = np.asarray(reference_points, dtype = np.float64)
    target_points = np.asarray(target_points, dtype = np.float64)

    if reference_points.shape != target_points.shape:
        raise ValueError("reference_points and target_points must have matching shapes")

    reference_center = reference_points.mean(axis = 0)
    target_center = target_points.mean(axis = 0)

    reference_centered = reference_points - reference_center
    target_centered = target_points - target_center

    covariance = reference_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.
        rotation = vt.T @ u.T

    translation = target_center - reference_center @ rotation

    return rotation.astype(np.float32), translation.astype(np.float32)


def rotation_angle_error_rad(
    pred_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> float:
    relative = pred_rotation.T @ target_rotation
    cos_angle = (np.trace(relative) - 1.) * 0.5
    cos_angle = np.clip(cos_angle, -1., 1.)
    return float(np.arccos(cos_angle))


def pose_errors_from_point_trajectories(
    reference_points: np.ndarray,
    target_positions: np.ndarray,
    pred_positions: np.ndarray,
    point_lens: np.ndarray,
    object_lens: int,
) -> dict[str, Any]:
    """Compute per-frame/per-object position and orientation errors.

    Shapes:
        reference_points: (objects, points, 3)
        target_positions: (frames, objects, points, 3)
        pred_positions: (frames, objects, points, 3)
        point_lens: (objects,)
    """

    num_frames = target_positions.shape[0]
    position_error = np.full((num_frames, object_lens), np.nan, dtype = np.float32)
    orientation_error_rad = np.full((num_frames, object_lens), np.nan, dtype = np.float32)

    target_translations = np.full((num_frames, object_lens, 3), np.nan, dtype = np.float32)
    pred_translations = np.full((num_frames, object_lens, 3), np.nan, dtype = np.float32)
    target_rotations = np.full((num_frames, object_lens, 3, 3), np.nan, dtype = np.float32)
    pred_rotations = np.full((num_frames, object_lens, 3, 3), np.nan, dtype = np.float32)

    for frame_index in range(num_frames):
        for object_index in range(object_lens):
            point_len = int(point_lens[object_index])

            reference = reference_points[object_index, :point_len]
            target = target_positions[frame_index, object_index, :point_len]
            pred = pred_positions[frame_index, object_index, :point_len]

            target_rotation, target_translation = estimate_rigid_transform(reference, target)
            pred_rotation, pred_translation = estimate_rigid_transform(reference, pred)

            target_rotations[frame_index, object_index] = target_rotation
            pred_rotations[frame_index, object_index] = pred_rotation
            target_translations[frame_index, object_index] = target_translation
            pred_translations[frame_index, object_index] = pred_translation

            position_error[frame_index, object_index] = np.linalg.norm(pred_translation - target_translation)
            orientation_error_rad[frame_index, object_index] = rotation_angle_error_rad(pred_rotation, target_rotation)

    return {
        "position_error": position_error,
        "orientation_error_rad": orientation_error_rad,
        "orientation_error_deg": np.rad2deg(orientation_error_rad),
        "target_translations": target_translations,
        "pred_translations": pred_translations,
        "target_rotations": target_rotations,
        "pred_rotations": pred_rotations,
    }


def summarize_pose_errors(
    position_error: np.ndarray,
    orientation_error_rad: np.ndarray,
    *,
    rollout_start_index: int = 2,
) -> dict[str, Any]:
    rollout_position = position_error[rollout_start_index:]
    rollout_orientation = orientation_error_rad[rollout_start_index:]

    return {
        "position_rmse_all": float(np.sqrt(np.nanmean(position_error ** 2))),
        "orientation_rmse_rad_all": float(np.sqrt(np.nanmean(orientation_error_rad ** 2))),
        "orientation_rmse_deg_all": float(np.rad2deg(np.sqrt(np.nanmean(orientation_error_rad ** 2)))),
        "position_rmse_rollout": float(np.sqrt(np.nanmean(rollout_position ** 2))),
        "orientation_rmse_rad_rollout": float(np.sqrt(np.nanmean(rollout_orientation ** 2))),
        "orientation_rmse_deg_rollout": float(np.rad2deg(np.sqrt(np.nanmean(rollout_orientation ** 2)))),
        "position_rmse_per_object": np.sqrt(np.nanmean(rollout_position ** 2, axis = 0)).astype(np.float32),
        "orientation_rmse_rad_per_object": np.sqrt(np.nanmean(rollout_orientation ** 2, axis = 0)).astype(np.float32),
        "orientation_rmse_deg_per_object": np.rad2deg(np.sqrt(np.nanmean(rollout_orientation ** 2, axis = 0))).astype(np.float32),
    }
