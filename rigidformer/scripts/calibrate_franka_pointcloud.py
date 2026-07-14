#!/usr/bin/env python3
"""Prepare Franka cube-drop pointclouds for RigidFormer.

The raw pointcloud extractor writes visible points independently for each frame:

``cube_points`` and ``gripper_points`` with shape ``(episodes, steps, points, 3)``.

RigidFormer trains on point trajectories, so point index ``i`` should represent
the same canonical point across time.  This script produces that calibrated
representation:

``/data/object_points`` with shape ``(episodes, steps, objects=2, points, 3)``.

Default calibration is fast centroid calibration:

* take frame 0 as the canonical visible point set;
* move that same point set by each frame's observed pointcloud centroid.

This gives stable point identities and is cheap enough for the full Franka
dataset.  It intentionally uses gripper as context only by default, while cube
is the supervised object.

For cube rotation, prefer ``--method pose``.  It uses the cube root pose stored
in the source drop HDF5 to move canonical cube points rigidly through time.  The
gripper is still centroid-calibrated because the two fingers open and are not a
single rigid body.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


OBJECT_NAMES = ("cube", "gripper")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Pointcloud-only HDF5 from extract_rgbd_pointcloud_dataset.py.")
    parser.add_argument("--output-file", required=True, help="RigidFormer-calibrated HDF5 output.")
    parser.add_argument(
        "--pose-source-hdf5",
        default=None,
        help="Drop HDF5 containing states/rigid_object/object/root_pose. Defaults to input attrs['source_hdf5'].",
    )
    parser.add_argument("--points", type=int, default=1024, help="Number of canonical points per object.")
    parser.add_argument("--method", choices=("pose", "centroid", "raw"), default="centroid")
    parser.add_argument("--train-object", choices=("cube", "both"), default="cube")
    parser.add_argument("--point-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=4)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=0, help="0 means all episodes from episode-start.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.compression == "none":
        return {}
    kwargs: dict[str, Any] = {"compression": args.compression, "shuffle": True}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.gzip_level)
    return kwargs


def valid_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    mask = np.isfinite(points).all(axis=-1)
    return points[mask]


def robust_centroid(points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    valid = valid_points(points)
    if valid.shape[0] == 0:
        return fallback.astype(np.float32, copy=True)
    return valid.mean(axis=0).astype(np.float32)


def decode_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def canonical_points(object_traj: h5py.Dataset, episode_index: int, points: int) -> tuple[np.ndarray, np.ndarray]:
    steps = object_traj.shape[1]
    canonical = None
    canonical_frame = 0
    for frame in range(steps):
        valid = valid_points(object_traj[episode_index, frame])
        if valid.shape[0] > 0:
            canonical = valid
            canonical_frame = frame
            break
    if canonical is None:
        return np.zeros((points, 3), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    rng = np.random.default_rng(episode_index * 1009 + points)
    replace = canonical.shape[0] < points
    indices = rng.choice(canonical.shape[0], size=points, replace=replace)
    selected = canonical[indices].astype(np.float32)
    center = robust_centroid(object_traj[episode_index, canonical_frame], selected.mean(axis=0))
    return selected, center


def calibrate_object_centroid(object_traj: h5py.Dataset, episode_index: int, points: int) -> tuple[np.ndarray, np.ndarray]:
    steps = object_traj.shape[1]
    canonical, base_center = canonical_points(object_traj, episode_index, points)
    out = np.empty((steps, points, 3), dtype=np.float32)
    counts = np.zeros((steps,), dtype=np.int32)
    last_center = base_center
    for frame in range(steps):
        raw = np.asarray(object_traj[episode_index, frame], dtype=np.float32)
        valid = valid_points(raw)
        counts[frame] = int(valid.shape[0])
        center = valid.mean(axis=0).astype(np.float32) if valid.shape[0] else last_center
        out[frame] = canonical + (center - base_center).reshape(1, 3)
        last_center = center
    return out, counts


def calibrate_object_pose(
    object_traj: h5py.Dataset,
    episode_index: int,
    points: int,
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    steps = object_traj.shape[1]
    if poses.shape[0] < steps or poses.shape[-1] < 7:
        raise ValueError(f"Pose trajectory must have shape (at least {steps}, 7); got {poses.shape}")

    canonical, _ = canonical_points(object_traj, episode_index, points)
    pose0 = np.asarray(poses[0], dtype=np.float32)
    t0 = pose0[:3]
    r0 = quat_wxyz_to_matrix(pose0[3:7])
    local = (canonical - t0.reshape(1, 3)) @ r0

    out = np.empty((steps, points, 3), dtype=np.float32)
    counts = np.zeros((steps,), dtype=np.int32)
    for frame in range(steps):
        raw = np.asarray(object_traj[episode_index, frame], dtype=np.float32)
        counts[frame] = int(valid_points(raw).shape[0])
        pose = np.asarray(poses[frame], dtype=np.float32)
        rotation = quat_wxyz_to_matrix(pose[3:7])
        translation = pose[:3]
        out[frame] = local @ rotation.T + translation.reshape(1, 3)
    return out, counts


def calibrate_object_raw(object_traj: h5py.Dataset, episode_index: int, points: int) -> tuple[np.ndarray, np.ndarray]:
    steps = object_traj.shape[1]
    stored_points = object_traj.shape[2]
    if points > stored_points:
        raise ValueError(f"--points={points} exceeds stored points={stored_points}")
    out = np.asarray(object_traj[episode_index, :, :points], dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.isfinite(np.asarray(object_traj[episode_index], dtype=np.float32)).all(axis=-1).sum(axis=-1).astype(np.int32)
    return out, counts


def create_or_open_output(
    args: argparse.Namespace,
    *,
    episodes: int,
    steps: int,
    points: int,
    source_config: dict[str, Any],
) -> h5py.File:
    path = os.path.abspath(args.output_file)
    exists = os.path.exists(path)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if exists and args.overwrite:
        os.remove(path)
        exists = False
    if exists and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume or --overwrite: {path}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file = h5py.File(path, "r+" if exists else "w")
    if exists:
        data = file["data"]
        if data["object_points"].shape != (episodes, steps, len(OBJECT_NAMES), points, 3):
            file.close()
            raise RuntimeError("Cannot resume: output shape does not match requested conversion.")
        return file

    data = file.require_group("data")
    dtype = np.float16 if args.point_dtype == "float16" else np.float32
    kwargs = dataset_kwargs(args)
    data.create_dataset(
        "object_points",
        shape=(episodes, steps, len(OBJECT_NAMES), points, 3),
        dtype=dtype,
        chunks=(1, steps, len(OBJECT_NAMES), points, 3),
        **kwargs,
    )
    data.create_dataset(
        "object_point_counts",
        shape=(episodes, steps, len(OBJECT_NAMES)),
        dtype=np.int32,
        chunks=(1, steps, len(OBJECT_NAMES)),
        **kwargs,
    )
    data.create_dataset("done", data=np.zeros((episodes,), dtype=bool), chunks=(min(episodes, 1024),))
    data.create_dataset("object_names", data=np.asarray(OBJECT_NAMES, dtype=object), dtype=h5py.string_dtype("utf-8"))
    vertex_properties = np.asarray(
        [
            [1.0, 0.0, 1.0],  # cube: cube type, not gripper, supervised movable
            [0.0, 1.0, 0.0],  # gripper context: not cube, gripper type, context/kinematic
        ],
        dtype=np.float32,
    )
    data.create_dataset("vertex_properties", data=vertex_properties)
    if args.train_object == "cube":
        loss_mask = np.asarray([True, False], dtype=bool)
    else:
        loss_mask = np.asarray([True, True], dtype=bool)
    data.create_dataset("loss_object_mask", data=loss_mask)

    file.attrs["schema_version"] = "franka_rigidformer_pointcloud_v1"
    file.attrs["source_hdf5"] = os.path.abspath(args.input_file)
    file.attrs["rigidformer_pointcloud_config"] = json.dumps(
        {
            "method": args.method,
            "points": points,
            "objects": list(OBJECT_NAMES),
            "train_object": args.train_object,
            "point_dtype": args.point_dtype,
            "pose_source_hdf5": os.path.abspath(args.pose_source_hdf5) if args.pose_source_hdf5 else None,
            "control_dt": float(source_config.get("control_dt", 0.02)),
            "source_config": source_config,
        },
        sort_keys=True,
    )
    file.flush()
    return file


def resolve_pose_source_path(args: argparse.Namespace, source: h5py.File) -> str:
    path = args.pose_source_hdf5 or source.attrs.get("source_hdf5", "")
    path = decode_string(path)
    if not path:
        raise RuntimeError("--method pose needs --pose-source-hdf5 or input attrs['source_hdf5'].")
    return os.path.abspath(os.path.expanduser(path))


def episode_names_from_source(data: h5py.Group, count: int) -> list[str]:
    if "episode_names" in data:
        return [decode_string(name) for name in np.asarray(data["episode_names"])]
    return [f"demo_{index}" for index in range(count)]


def cube_pose_trajectory(pose_source: h5py.File, episode_name: str, steps: int) -> np.ndarray:
    if "data" not in pose_source or episode_name not in pose_source["data"]:
        raise KeyError(f"Episode {episode_name!r} not found in pose source HDF5.")
    episode = pose_source["data"][episode_name]
    if "states/rigid_object/object/root_pose" in episode:
        poses = np.asarray(episode["states/rigid_object/object/root_pose"], dtype=np.float32)
    elif "object_dynamics/root_pos_w" in episode and "object_dynamics/root_quat_w" in episode:
        poses = np.concatenate(
            (
                np.asarray(episode["object_dynamics/root_pos_w"], dtype=np.float32),
                np.asarray(episode["object_dynamics/root_quat_w"], dtype=np.float32),
            ),
            axis=-1,
        )
    else:
        raise KeyError(
            f"Episode {episode_name!r} has no states/rigid_object/object/root_pose "
            "or object_dynamics root pose datasets."
        )
    if poses.shape[0] < steps:
        raise ValueError(f"Episode {episode_name!r} pose length {poses.shape[0]} < pointcloud steps {steps}.")
    return poses[:steps]


def main() -> str:
    args = parse_cli()
    if args.points < 4:
        raise ValueError("--points must be >= 4 for RigidFormer anchor sampling.")
    with h5py.File(args.input_file, "r", locking=False) as source:
        data = source["data"]
        cube = data["cube_points"]
        gripper = data["gripper_points"]
        if cube.shape != gripper.shape or cube.ndim != 4 or cube.shape[-1] != 3:
            raise ValueError("cube_points and gripper_points must both have shape (episodes, steps, points, 3)")
        start = int(args.episode_start)
        total_source_episodes, steps, stored_points, _ = cube.shape
        stop = total_source_episodes if int(args.num_episodes) <= 0 else min(total_source_episodes, start + int(args.num_episodes))
        if start < 0 or start >= stop:
            raise ValueError("No source episodes selected.")
        selected_episodes = list(range(start, stop))
        if args.points > stored_points:
            raise ValueError(f"--points={args.points} exceeds stored points={stored_points}")
        source_config = json.loads(str(source.attrs.get("pointcloud_config", "{}")))
        episode_names = episode_names_from_source(data, total_source_episodes)

        pose_source = None
        if args.method == "pose":
            pose_source_path = resolve_pose_source_path(args, source)
            if not os.path.isfile(pose_source_path):
                raise FileNotFoundError(f"Pose source HDF5 not found: {pose_source_path}")
            args.pose_source_hdf5 = pose_source_path
            pose_source = h5py.File(pose_source_path, "r", locking=False)

        output = create_or_open_output(
            args,
            episodes=len(selected_episodes),
            steps=steps,
            points=int(args.points),
            source_config=source_config,
        )
        try:
            out = output["data"]
            done = out["done"]
            print(
                f"[INFO] Calibrating {len(selected_episodes)} episodes x {steps} steps "
                f"using method={args.method}, points={args.points}."
            )
            for local_index, source_episode_index in enumerate(selected_episodes):
                if bool(done[local_index]):
                    continue
                if args.method == "pose":
                    assert pose_source is not None
                    poses = cube_pose_trajectory(pose_source, episode_names[source_episode_index], steps)
                    cube_out, cube_counts = calibrate_object_pose(cube, source_episode_index, int(args.points), poses)
                    gripper_out, gripper_counts = calibrate_object_centroid(
                        gripper, source_episode_index, int(args.points)
                    )
                elif args.method == "centroid":
                    cube_out, cube_counts = calibrate_object_centroid(cube, source_episode_index, int(args.points))
                    gripper_out, gripper_counts = calibrate_object_centroid(
                        gripper, source_episode_index, int(args.points)
                    )
                else:
                    cube_out, cube_counts = calibrate_object_raw(cube, source_episode_index, int(args.points))
                    gripper_out, gripper_counts = calibrate_object_raw(gripper, source_episode_index, int(args.points))
                stacked = np.stack((cube_out, gripper_out), axis=1)
                out["object_points"][local_index] = stacked.astype(out["object_points"].dtype, copy=False)
                out["object_point_counts"][local_index] = np.stack((cube_counts, gripper_counts), axis=1)
                done[local_index] = True
                if (local_index + 1) % int(args.progress_every) == 0 or local_index == len(selected_episodes) - 1:
                    output.flush()
                    print(f"[INFO] {local_index + 1}/{len(selected_episodes)} episodes calibrated.")
            output.flush()
        finally:
            output.close()
            if pose_source is not None:
                pose_source.close()

    print(f"[INFO] RigidFormer pointcloud dataset ready: {os.path.abspath(args.output_file)}")
    return os.path.abspath(args.output_file)


if __name__ == "__main__":
    main()
