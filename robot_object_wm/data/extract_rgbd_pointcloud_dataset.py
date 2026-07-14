#!/usr/bin/env python3
"""Batch-extract cube/gripper pointcloud tensors from RGB-D/depth Franka drop datasets.

This writes pointcloud-only HDF5 output:

* ``/data/cube_points``: ``(episodes, steps, points_per_part, 3)``
* ``/data/gripper_points``: ``(episodes, steps, points_per_part, 3)``
* ``/data/cube_counts`` and ``/data/gripper_counts``: visible raw point counts
* ``/data/done``: per-episode extraction completion flag for resume

No RGB images or RGB point colors are written.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import extract_rgbd_pointcloud_trajectory as pc  # noqa: E402


def numeric_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return sys.maxsize, name


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="RGB-D/depth drop HDF5 produced by collect_franka_cube_drop.py.")
    parser.add_argument("--output-file", required=True, help="Pointcloud-only HDF5 output path.")
    parser.add_argument("--views", default="front,left,right")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=0, help="0 means all episodes from episode-start.")
    parser.add_argument("--points-per-part", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube-size", type=float, default=0.055)
    parser.add_argument("--cube-margin", type=float, default=0.012)
    parser.add_argument(
        "--gripper-crop-half-extent",
        type=pc.parse_vec3,
        default=np.asarray([0.18, 0.18, 0.16], dtype=np.float32),
    )
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--cube-instance-ids", default=None)
    parser.add_argument("--gripper-instance-ids", default=None)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=4)
    parser.add_argument("--point-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.compression == "none":
        return {}
    kwargs: dict[str, Any] = {"compression": args.compression, "shuffle": True}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.gzip_level)
    return kwargs


def create_or_open_output(
    path: str,
    args: argparse.Namespace,
    episode_names: list[str],
    steps: int,
    views: list[str],
) -> h5py.File:
    exists = os.path.exists(path)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if exists and args.overwrite:
        os.remove(path)
        exists = False
    if exists and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume or --overwrite: {path}")

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file = h5py.File(path, "r+" if exists else "w")
    if exists:
        data = file["data"]
        expected = {
            "points_per_part": int(args.points_per_part),
            "steps": int(steps),
            "views": views,
            "point_dtype": str(args.point_dtype),
        }
        stored = json.loads(str(file.attrs.get("pointcloud_config", "{}")))
        for key, value in expected.items():
            if stored.get(key) != value:
                file.close()
                raise RuntimeError(f"Cannot resume with different {key}: stored={stored.get(key)!r}, requested={value!r}")
        return file

    data = file.require_group("data")
    point_dtype = np.float16 if str(args.point_dtype) == "float16" else np.float32
    shape = (len(episode_names), int(steps), int(args.points_per_part), 3)
    chunk = (1, int(steps), int(args.points_per_part), 3)
    kwargs = dataset_kwargs(args)
    data.create_dataset("cube_points", shape=shape, dtype=point_dtype, chunks=chunk, **kwargs)
    data.create_dataset("gripper_points", shape=shape, dtype=point_dtype, chunks=chunk, **kwargs)
    data.create_dataset("cube_counts", shape=(len(episode_names), int(steps)), dtype=np.int32, chunks=(1, int(steps)), **kwargs)
    data.create_dataset(
        "gripper_counts", shape=(len(episode_names), int(steps)), dtype=np.int32, chunks=(1, int(steps)), **kwargs
    )
    data.create_dataset("done", data=np.zeros((len(episode_names),), dtype=bool), chunks=(min(len(episode_names), 1024),))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    data.create_dataset("episode_names", data=np.asarray(episode_names, dtype=object), dtype=string_dtype)
    file.attrs["source_hdf5"] = os.path.abspath(args.input_file)
    file.attrs["schema_version"] = "franka_cube_drop_pointcloud_v1"
    file.attrs["pointcloud_config"] = json.dumps(
        {
            "points_per_part": int(args.points_per_part),
            "steps": int(steps),
            "views": views,
            "cube_size": float(args.cube_size),
            "cube_margin": float(args.cube_margin),
            "gripper_crop_half_extent": np.asarray(args.gripper_crop_half_extent, dtype=float).tolist(),
            "max_depth": float(args.max_depth),
            "point_dtype": str(args.point_dtype),
            "cube_instance_ids": args.cube_instance_ids,
            "gripper_instance_ids": args.gripper_instance_ids,
        },
        sort_keys=True,
    )
    file.flush()
    return file


def extract_episode(
    episode: h5py.Group,
    views: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "depth" not in episode:
        raise RuntimeError("Input episode has no depth group. Recollect with --include-depth.")

    cube_ids = pc.parse_ids(args.cube_instance_ids)
    gripper_ids = pc.parse_ids(args.gripper_instance_ids)
    seg_group = pc.segmentation_group(episode)
    if (cube_ids or gripper_ids) and seg_group is None:
        raise RuntimeError("Instance ids were requested, but this episode has no segmentation group.")

    steps = int(episode.attrs["num_samples"])
    n_points = int(args.points_per_part)
    cube_points = np.full((steps, n_points, 3), np.nan, dtype=np.float32)
    gripper_points = np.full((steps, n_points, 3), np.nan, dtype=np.float32)
    cube_counts = np.zeros((steps,), dtype=np.int32)
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
            pc.camera_rotation_ros_from_hdf5(camera_info),
        )

    for timestep in range(steps):
        all_points: list[np.ndarray] = []
        cube_points_from_ids: list[np.ndarray] = []
        gripper_points_from_ids: list[np.ndarray] = []

        for view in views:
            intrinsic, camera_position, camera_rotation = camera_cache[view]
            depth = pc.read_depth_frame(episode, view, timestep)
            h, w = depth.shape[:2]
            dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            instance = None
            if seg_group is not None:
                instance = np.asarray(seg_group[view][timestep, ..., 0])

            points, _ = pc.unproject_depth_to_env(
                depth,
                dummy_rgb,
                intrinsic,
                camera_position,
                camera_rotation,
                max_depth=float(args.max_depth) if args.max_depth else None,
            )
            all_points.append(points)

            if instance is not None and cube_ids:
                points_i, _ = pc.unproject_depth_to_env(
                    depth,
                    dummy_rgb,
                    intrinsic,
                    camera_position,
                    camera_rotation,
                    mask=np.isin(instance, list(cube_ids)),
                    max_depth=args.max_depth,
                )
                cube_points_from_ids.append(points_i)

            if instance is not None and gripper_ids:
                points_i, _ = pc.unproject_depth_to_env(
                    depth,
                    dummy_rgb,
                    intrinsic,
                    camera_position,
                    camera_rotation,
                    mask=np.isin(instance, list(gripper_ids)),
                    max_depth=args.max_depth,
                )
                gripper_points_from_ids.append(points_i)

        merged_points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)

        if cube_points_from_ids:
            cube_raw = np.concatenate(cube_points_from_ids, axis=0)
        else:
            cube_mask = pc.crop_oriented_box(merged_points, cube_poses[timestep], cube_half_extent)
            cube_raw = merged_points[cube_mask]
        cube_colors = np.zeros((cube_raw.shape[0], 3), dtype=np.uint8)
        cube_points[timestep], _, cube_counts[timestep] = pc.sample_fixed(cube_raw, cube_colors, n_points, rng)

        if gripper_points_from_ids:
            gripper_raw = np.concatenate(gripper_points_from_ids, axis=0)
        else:
            gripper_mask = pc.crop_axis_aligned(merged_points, initial_cube_center, args.gripper_crop_half_extent)
            gripper_mask &= ~pc.crop_oriented_box(merged_points, cube_poses[timestep], cube_half_extent)
            gripper_raw = merged_points[gripper_mask]
        gripper_colors = np.zeros((gripper_raw.shape[0], 3), dtype=np.uint8)
        gripper_points[timestep], _, gripper_counts[timestep] = pc.sample_fixed(
            gripper_raw, gripper_colors, n_points, rng
        )

    return cube_points, gripper_points, cube_counts, gripper_counts


def main() -> str:
    args = parse_cli()
    if int(args.points_per_part) <= 0:
        raise ValueError("--points-per-part must be positive.")
    views = [view.strip() for view in str(args.views).split(",") if view.strip()]
    unknown = sorted(set(views) - set(pc.CAMERA_VIEWS))
    if unknown:
        raise ValueError(f"Unknown camera views: {unknown}")

    with h5py.File(args.input_file, "r", locking=False) as source:
        data = source["data"]
        all_names = sorted((name for name in data if name.startswith("demo_")), key=numeric_demo_key)
        start = int(args.episode_start)
        stop = len(all_names) if int(args.num_episodes) <= 0 else min(len(all_names), start + int(args.num_episodes))
        episode_names = all_names[start:stop]
        if not episode_names:
            raise ValueError("No episodes selected.")
        steps = int(data[episode_names[0]].attrs["num_samples"])
        for name in episode_names:
            if int(data[name].attrs["num_samples"]) != steps:
                raise RuntimeError("All selected episodes must have the same number of samples.")

        output = create_or_open_output(os.path.abspath(args.output_file), args, episode_names, steps, views)
        try:
            out = output["data"]
            done = out["done"]
            rng = np.random.default_rng(int(args.seed))
            completed_before = int(np.asarray(done).sum())
            print(
                f"[INFO] Extracting {len(episode_names)} episodes x {steps} steps, "
                f"{args.points_per_part} points/part. Already done: {completed_before}."
            )
            for local_index, name in enumerate(episode_names):
                if bool(done[local_index]):
                    continue
                episode_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.uint32).max) + local_index)
                cube_points, gripper_points, cube_counts, gripper_counts = extract_episode(
                    data[name], views, args, episode_rng
                )
                out["cube_points"][local_index] = cube_points.astype(out["cube_points"].dtype, copy=False)
                out["gripper_points"][local_index] = gripper_points.astype(out["gripper_points"].dtype, copy=False)
                out["cube_counts"][local_index] = cube_counts
                out["gripper_counts"][local_index] = gripper_counts
                done[local_index] = True
                if (local_index + 1) % int(args.progress_every) == 0 or local_index == len(episode_names) - 1:
                    output.flush()
                    print(
                        f"[INFO] {local_index + 1}/{len(episode_names)} episodes; "
                        f"cube median count={int(np.median(cube_counts))}, "
                        f"gripper median count={int(np.median(gripper_counts))}"
                    )
            output.flush()
        finally:
            output.close()

    print(f"[INFO] Pointcloud dataset ready: {os.path.abspath(args.output_file)}")
    return os.path.abspath(args.output_file)


if __name__ == "__main__":
    main()
