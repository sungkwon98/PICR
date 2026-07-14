#!/usr/bin/env python3
"""Extract cube/gripper pointcloud trajectories from a Franka cube-drop RGB-D HDF5 file.

The collector stores calibrated multi-view RGB-D frames in environment coordinates.
This script unprojects each depth map, transforms the points into the environment
frame, and then keeps points belonging to:

* the cube, by cropping with the recorded cube pose and a known cube side length;
* the gripper, either by provided instance-segmentation ids or by a local crop
  around the initial held-cube position with the cube crop removed.

The output is a compressed ``.npz`` with fixed-size sampled point sets for each
timestep, ready for quick visualization or model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


CAMERA_VIEWS = ("front", "left", "right")


def parse_ids(text: str | None) -> set[int] | None:
    if text is None or text.strip() == "":
        return None
    return {int(part) for part in text.replace(";", ",").split(",") if part.strip()}


def parse_vec3(text: str) -> np.ndarray:
    values = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return np.asarray(values, dtype=np.float32)


def quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def camera_rotation_ros_from_hdf5(camera_group: h5py.Group) -> np.ndarray:
    if "quaternion_ros_convention" in camera_group:
        return quat_to_matrix_wxyz(np.asarray(camera_group["quaternion_ros_convention"]))

    # Backward-compatible fallback for files that only store Isaac Lab's
    # "world" camera convention (+X forward, +Y left, +Z up).
    rotation_world = quat_to_matrix_wxyz(np.asarray(camera_group["quaternion_world_convention"]))
    ros_axes_in_world_camera = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return rotation_world @ ros_axes_in_world_camera


def unproject_depth_to_env(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation_ros: np.ndarray,
    mask: np.ndarray | None = None,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth_2d = np.asarray(depth, dtype=np.float32)
    if depth_2d.ndim == 3:
        depth_2d = depth_2d[..., 0]
    h, w = depth_2d.shape
    valid = np.isfinite(depth_2d) & (depth_2d > 0.0)
    if max_depth is not None and max_depth > 0.0:
        valid &= depth_2d <= float(max_depth)
    if mask is not None:
        valid &= mask
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_2d[rows, cols]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (cols.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points_camera_ros = np.stack((x, y, z), axis=-1).astype(np.float32)
    points_env = points_camera_ros @ camera_rotation_ros.T + camera_position.reshape(1, 3)
    colors = np.asarray(rgb, dtype=np.uint8)[rows, cols, :3]
    return points_env.astype(np.float32), colors


def read_depth_frame(episode: h5py.Group, view: str, timestep: int) -> np.ndarray:
    dataset = episode[f"depth/{view}"]
    depth = np.asarray(dataset[timestep])
    if np.issubdtype(depth.dtype, np.integer):
        scale = float(episode["depth"].attrs.get("scale_to_meters", 0.001))
        depth = depth.astype(np.float32) * scale
    else:
        depth = depth.astype(np.float32, copy=False)
    return depth


def crop_oriented_box(points: np.ndarray, pose_wxyz: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    center = np.asarray(pose_wxyz[:3], dtype=np.float32)
    rotation = quat_to_matrix_wxyz(np.asarray(pose_wxyz[3:7], dtype=np.float32))
    local = (points - center.reshape(1, 3)) @ rotation
    return np.all(np.abs(local) <= half_extent.reshape(1, 3), axis=1)


def crop_axis_aligned(points: np.ndarray, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.all(np.abs(points - center.reshape(1, 3)) <= half_extent.reshape(1, 3), axis=1)


def sample_fixed(
    points: np.ndarray,
    colors: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    actual = int(points.shape[0])
    sampled_points = np.full((count, 3), np.nan, dtype=np.float32)
    sampled_colors = np.zeros((count, 3), dtype=np.uint8)
    if actual == 0 or count <= 0:
        return sampled_points, sampled_colors, actual
    replace = actual < count
    indices = rng.choice(actual, size=count, replace=replace)
    sampled_points[:] = points[indices]
    sampled_colors[:] = colors[indices]
    return sampled_points, sampled_colors, actual


def segmentation_group(episode: h5py.Group) -> h5py.Group | None:
    if "segmentation/instance_id" in episode:
        return episode["segmentation/instance_id"]
    if "segmentation/instance" in episode:
        return episode["segmentation/instance"]
    return None


def print_instance_ids(episode: h5py.Group, views: list[str], timestep: int) -> None:
    seg_group = segmentation_group(episode)
    if seg_group is None:
        print("[INFO] This episode does not contain segmentation masks.")
        return
    summary: dict[str, list[tuple[int, int]]] = {}
    for view in views:
        ids, counts = np.unique(np.asarray(seg_group[view][timestep, ..., 0]), return_counts=True)
        order = np.argsort(counts)[::-1]
        summary[view] = [(int(ids[i]), int(counts[i])) for i in order[:20]]
    print("[INFO] Top instance ids by pixel count:")
    print(json.dumps(summary, indent=2))


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def resolve_episode(data: h5py.Group, episode: str) -> h5py.Group:
    name = episode if episode.startswith("demo_") else f"demo_{int(episode)}"
    if name not in data:
        raise KeyError(f"Episode {name!r} not found")
    return data[name]


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--episode", default="demo_0", help="Episode name or integer index.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--views", default="front,left,right", help="Comma-separated camera views.")
    parser.add_argument("--points-per-part", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube-size", type=float, default=0.055, help="Cube side length in meters.")
    parser.add_argument("--cube-margin", type=float, default=0.012, help="Extra cube crop margin in meters.")
    parser.add_argument(
        "--gripper-crop-half-extent",
        type=parse_vec3,
        default=np.asarray([0.18, 0.18, 0.16], dtype=np.float32),
        help="Axis-aligned gripper fallback crop half extents around the initial cube center.",
    )
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--cube-instance-ids", default=None, help="Optional comma-separated cube instance ids.")
    parser.add_argument("--gripper-instance-ids", default=None, help="Optional comma-separated gripper instance ids.")
    parser.add_argument("--print-instance-ids", action="store_true")
    parser.add_argument("--print-instance-timestep", type=int, default=0)
    parser.add_argument("--save-ply-dir", default=None, help="Optional directory for first/last cube/gripper PLY previews.")
    return parser.parse_args()


def main() -> str:
    args = parse_cli()
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unknown = sorted(set(views) - set(CAMERA_VIEWS))
    if unknown:
        raise ValueError(f"Unknown camera views: {unknown}")
    cube_ids = parse_ids(args.cube_instance_ids)
    gripper_ids = parse_ids(args.gripper_instance_ids)
    rng = np.random.default_rng(int(args.seed))

    with h5py.File(args.input_file, "r", locking=False) as file:
        episode = resolve_episode(file["data"], str(args.episode))
        if "depth" not in episode:
            raise RuntimeError("Input episode has RGB images but no depth group. Recollect with --include-depth.")
        seg_group = segmentation_group(episode)
        if (cube_ids or gripper_ids or args.print_instance_ids) and seg_group is None:
            raise RuntimeError("Instance ids were requested, but this episode has no segmentation group.")

        if args.print_instance_ids:
            print_instance_ids(episode, views, int(args.print_instance_timestep))

        steps = int(episode.attrs["num_samples"])
        points_per_part = int(args.points_per_part)
        cube_points = np.full((steps, points_per_part, 3), np.nan, dtype=np.float32)
        cube_rgb = np.zeros((steps, points_per_part, 3), dtype=np.uint8)
        cube_counts = np.zeros((steps,), dtype=np.int32)
        gripper_points = np.full((steps, points_per_part, 3), np.nan, dtype=np.float32)
        gripper_rgb = np.zeros((steps, points_per_part, 3), dtype=np.uint8)
        gripper_counts = np.zeros((steps,), dtype=np.int32)

        cube_poses = np.asarray(episode["states/rigid_object/object/root_pose"], dtype=np.float32)
        initial_cube_center = np.asarray(episode["initial_state/rigid_object/object/root_pose"][0, :3], dtype=np.float32)
        cube_half_extent = np.full((3,), float(args.cube_size) / 2.0 + float(args.cube_margin), dtype=np.float32)

        camera_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for view in views:
            camera_info = episode[f"camera_info/{view}"]
            camera_cache[view] = (
                np.asarray(camera_info["intrinsic_matrix"], dtype=np.float32),
                np.asarray(camera_info["position"], dtype=np.float32),
                camera_rotation_ros_from_hdf5(camera_info),
            )

        for timestep in range(steps):
            all_points: list[np.ndarray] = []
            all_colors: list[np.ndarray] = []
            cube_points_from_ids: list[np.ndarray] = []
            cube_colors_from_ids: list[np.ndarray] = []
            gripper_points_from_ids: list[np.ndarray] = []
            gripper_colors_from_ids: list[np.ndarray] = []

            for view in views:
                intrinsic, camera_position, camera_rotation = camera_cache[view]
                rgb = np.asarray(episode[f"images/{view}"][timestep])
                depth = read_depth_frame(episode, view, timestep)
                instance = None
                if seg_group is not None:
                    instance = np.asarray(seg_group[view][timestep, ..., 0])

                points, colors = unproject_depth_to_env(
                    depth,
                    rgb,
                    intrinsic,
                    camera_position,
                    camera_rotation,
                    max_depth=float(args.max_depth) if args.max_depth else None,
                )
                all_points.append(points)
                all_colors.append(colors)

                if instance is not None and cube_ids:
                    mask = np.isin(instance, list(cube_ids))
                    points_i, colors_i = unproject_depth_to_env(
                        depth, rgb, intrinsic, camera_position, camera_rotation, mask=mask, max_depth=args.max_depth
                    )
                    cube_points_from_ids.append(points_i)
                    cube_colors_from_ids.append(colors_i)

                if instance is not None and gripper_ids:
                    mask = np.isin(instance, list(gripper_ids))
                    points_i, colors_i = unproject_depth_to_env(
                        depth, rgb, intrinsic, camera_position, camera_rotation, mask=mask, max_depth=args.max_depth
                    )
                    gripper_points_from_ids.append(points_i)
                    gripper_colors_from_ids.append(colors_i)

            merged_points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)
            merged_colors = np.concatenate(all_colors, axis=0) if all_colors else np.empty((0, 3), dtype=np.uint8)

            if cube_points_from_ids:
                cube_raw = np.concatenate(cube_points_from_ids, axis=0)
                cube_color_raw = np.concatenate(cube_colors_from_ids, axis=0)
            else:
                cube_mask = crop_oriented_box(merged_points, cube_poses[timestep], cube_half_extent)
                cube_raw = merged_points[cube_mask]
                cube_color_raw = merged_colors[cube_mask]
            cube_points[timestep], cube_rgb[timestep], cube_counts[timestep] = sample_fixed(
                cube_raw, cube_color_raw, points_per_part, rng
            )

            if gripper_points_from_ids:
                gripper_raw = np.concatenate(gripper_points_from_ids, axis=0)
                gripper_color_raw = np.concatenate(gripper_colors_from_ids, axis=0)
            else:
                gripper_mask = crop_axis_aligned(merged_points, initial_cube_center, args.gripper_crop_half_extent)
                # Remove the current cube so the fallback crop is mostly fingers/palm/wrist.
                gripper_mask &= ~crop_oriented_box(merged_points, cube_poses[timestep], cube_half_extent)
                gripper_raw = merged_points[gripper_mask]
                gripper_color_raw = merged_colors[gripper_mask]
            gripper_points[timestep], gripper_rgb[timestep], gripper_counts[timestep] = sample_fixed(
                gripper_raw, gripper_color_raw, points_per_part, rng
            )

    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        episode=str(args.episode),
        views=np.asarray(views),
        cube_points=cube_points,
        cube_rgb=cube_rgb,
        cube_counts=cube_counts,
        gripper_points=gripper_points,
        gripper_rgb=gripper_rgb,
        gripper_counts=gripper_counts,
        cube_size=np.float32(args.cube_size),
        cube_margin=np.float32(args.cube_margin),
        gripper_crop_half_extent=np.asarray(args.gripper_crop_half_extent, dtype=np.float32),
    )

    if args.save_ply_dir:
        ply_dir = Path(args.save_ply_dir).expanduser().resolve()
        ply_dir.mkdir(parents=True, exist_ok=True)
        for timestep_name, timestep in (("first", 0), ("last", cube_points.shape[0] - 1)):
            write_ply(ply_dir / f"{timestep_name}_cube.ply", cube_points[timestep], cube_rgb[timestep])
            write_ply(ply_dir / f"{timestep_name}_gripper.ply", gripper_points[timestep], gripper_rgb[timestep])

    print(
        f"[INFO] Saved pointcloud trajectory: {output_path} "
        f"(cube median visible points={int(np.median(cube_counts))}, "
        f"gripper median visible points={int(np.median(gripper_counts))})"
    )
    return str(output_path)


if __name__ == "__main__":
    main()
