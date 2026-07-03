"""Augment robot-object HDF5 datasets with privileged collision labels.

The augmentation is intentionally scoped to the geometry available in this
world-model dataset:

* object cube OBB from ``states/rigid_object/object/root_pose``;
* left/right Panda finger boxes from Franka FK and the finger collision mesh;
* an analytic horizontal ground/support plane.

The written labels are suitable for privileged training targets and for
reconstructing ``python-fcl`` box collision objects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import h5py
import numpy as np
import torch

from robot_object_wm.models.utils import FrankaForwardKinematics

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


DEFAULT_FINGER_BOX_SIZE = np.asarray([0.02097427, 0.02653603, 0.05371734], dtype=np.float32)
DEFAULT_FINGER_BOX_CENTER = np.asarray([0.00000768, 0.01313537, 0.02699037], dtype=np.float32)
FINGER_JOINT_Z = 0.0584
SCHEMA_VERSION = "1.0"

BASE_PAIR_NAMES = (
    "object_ground",
    "object_left_finger",
    "object_right_finger",
    "object_gripper",
    "left_finger_ground",
    "right_finger_ground",
    "gripper_ground",
    "left_right_finger",
)
DEFAULT_PAIR_NAMES = (
    "object_ground",
    "object_left_finger",
    "object_right_finger",
    "object_gripper",
    # "left_finger_ground",
    # "right_finger_ground",
    # "gripper_ground",
)
OBB_NAMES = ("object_box", "left_finger", "right_finger")


@dataclass(frozen=True)
class FingerBoxSpec:
    size: np.ndarray
    center: np.ndarray
    source: str


@dataclass
class EpisodeCollisionInfo:
    pair_names: list[str]
    collision: np.ndarray
    distance: np.ndarray
    signed_distance: np.ndarray
    nearest_points: np.ndarray | None
    ground_z: float
    obb_center: np.ndarray | None
    obb_rotation: np.ndarray | None
    obb_size: np.ndarray | None


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1.0e-8)
    w, x, y, z = [quat[..., idx] for idx in range(4)]
    matrix = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def obb_corners(center: np.ndarray, rotation: np.ndarray, size: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return center[None, :] + (signs * (np.asarray(size, dtype=np.float64) * 0.5)) @ rotation.T


def ground_signed_distance(center: np.ndarray, rotation: np.ndarray, size: np.ndarray, ground_z: float) -> float:
    return float(np.min(obb_corners(center, rotation, size)[:, 2]) - ground_z)


def infer_cube_size(mass: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    mass0 = float(np.asarray(mass).reshape(-1)[0])
    inertia0 = np.asarray(inertia, dtype=np.float64).reshape(-1, 3, 3)[0]
    diag_mean = float(np.mean(np.diag(inertia0)))
    if mass0 <= 0.0 or diag_mean <= 0.0:
        raise ValueError(f"Cannot infer cube size from mass={mass0} and inertia={inertia0}.")
    side = math.sqrt(6.0 * diag_mean / mass0)
    return np.asarray([side, side, side], dtype=np.float32)


def discover_dataset_files(dataset_files: list[str], dataset_dir: str | None) -> list[str]:
    paths = [os.path.abspath(path) for path in dataset_files]
    if dataset_dir:
        root = os.path.abspath(dataset_dir)
        paths.extend(str(path) for path in sorted(Path(root).glob("*.hdf5")))
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise FileNotFoundError("No HDF5 files were provided or discovered.")
    missing = [path for path in unique if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Dataset file(s) not found: {missing}")
    return unique


def with_output_suffix(path: str, suffix: str, output_dir: str | None) -> str:
    src = Path(path)
    root = Path(output_dir).expanduser().resolve() if output_dir else src.parent
    return str(root / f"{src.stem}{suffix}{src.suffix}")


def prepare_output_files(
    paths: list[str],
    *,
    output_suffix: str,
    output_dir: str | None,
    overwrite: bool,
    dry_run: bool,
) -> list[str]:
    if not output_suffix:
        return paths

    targets: list[str] = []
    for src_path in paths:
        src = os.path.abspath(src_path)
        target = os.path.abspath(with_output_suffix(src, output_suffix, output_dir))
        if src == target:
            raise ValueError(f"Output path would overwrite source: {src}")
        if Path(src).stem.endswith(output_suffix):
            print(f"[SKIP] Source already has output suffix: {src}")
            continue
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        if os.path.exists(target):
            if not overwrite:
                raise FileExistsError(f"Output file already exists: {target}. Use --overwrite to replace it.")
            if not dry_run:
                os.remove(target)
        print(f"[{'DRY COPY' if dry_run else 'COPY'}] {src} -> {target}")
        if dry_run:
            targets.append(src)
        else:
            shutil.copy2(src, target)
            targets.append(target)
    return targets


def resolve_pair_names(values: Iterable[str]) -> list[str]:
    requested: list[str] = []
    for value in values:
        requested.extend(part.strip() for part in value.split(",") if part.strip())
    if not requested:
        requested = ["default"]
    expanded: list[str] = []
    for value in requested:
        if value == "default":
            expanded.extend(DEFAULT_PAIR_NAMES)
        elif value == "all":
            expanded.extend(BASE_PAIR_NAMES)
        elif value in BASE_PAIR_NAMES:
            expanded.append(value)
        else:
            choices = ", ".join(("default", "all", *BASE_PAIR_NAMES))
            raise ValueError(f"Unknown collision pair '{value}'. Choices: {choices}")
    deduped: list[str] = []
    seen = set()
    for name in expanded:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def candidate_finger_mesh_paths() -> list[str]:
    return [
        "/home/sukchul/miniconda3/envs/torch/lib/python3.11/site-packages/mani_skill/assets/robots/panda/franka_description/meshes/collision/finger.stl",
        "/home/sukchul/miniconda3/envs/torch/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/meshes/collision/finger.stl",
        "/home/sukchul/world_model/neural-robot-dynamics/envs/warp_sim_envs/assets/franka_description/meshes/collision/finger.stl",
    ]


def load_finger_box_spec(path: str | None) -> FingerBoxSpec:
    mesh_path = None
    if path and path.lower() != "auto":
        mesh_path = path
    elif path is None or path.lower() == "auto":
        mesh_path = next((candidate for candidate in candidate_finger_mesh_paths() if os.path.isfile(candidate)), None)

    if mesh_path:
        try:
            import trimesh

            mesh = trimesh.load(mesh_path, force="mesh")
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            size = (bounds[1] - bounds[0]).astype(np.float32)
            center = (0.5 * (bounds[0] + bounds[1])).astype(np.float32)
            return FingerBoxSpec(size=size, center=center, source=os.path.abspath(mesh_path))
        except Exception as exc:
            print(f"[WARN] Failed to load finger mesh '{mesh_path}' ({exc}); using measured defaults.")

    return FingerBoxSpec(
        size=DEFAULT_FINGER_BOX_SIZE.copy(),
        center=DEFAULT_FINGER_BOX_CENTER.copy(),
        source="measured_default_franka_finger_collision_aabb",
    )


def parse_float_or_auto(value: str) -> str | float:
    if value.lower() == "auto":
        return "auto"
    return float(value)


def compute_hand_frames(
    joint_position: np.ndarray,
    robot_root_pose: np.ndarray,
    *,
    robot_dof: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if joint_position.shape[1] < 9:
        raise ValueError("Gripper collision requires 9 robot joints, including two Panda finger joints.")

    torch_device = torch.device(device)
    fk = FrankaForwardKinematics(robot_dof=robot_dof, tool_z_offset=0.0).to(torch_device)
    fk.eval()
    with torch.no_grad():
        q = torch.from_numpy(np.asarray(joint_position[:, :robot_dof], dtype=np.float32)).to(torch_device)
        p_hand_base, r_hand_base = fk(q)
    p_hand_base_np = p_hand_base.cpu().numpy().astype(np.float64)
    r_hand_base_np = r_hand_base.cpu().numpy().astype(np.float64)

    r_root = quat_wxyz_to_matrix(robot_root_pose[:, 3:7])
    p_root = robot_root_pose[:, :3].astype(np.float64)
    r_hand_world = np.einsum("tij,tjk->tik", r_root, r_hand_base_np)
    p_hand_world = p_root + np.einsum("tij,tj->ti", r_root, p_hand_base_np)
    return p_hand_world, r_hand_world


def compute_box_transforms(
    episode,
    *,
    robot_dof: int,
    object_size_arg: str | float,
    ground_z_arg: str | float,
    finger_spec: FingerBoxSpec,
    device: str,
) -> tuple[dict[str, np.ndarray], float, np.ndarray, np.ndarray]:
    object_root_pose = np.asarray(episode["states"]["rigid_object"]["object"]["root_pose"], dtype=np.float32)
    joint_position = np.asarray(episode["states"]["articulation"]["robot"]["joint_position"], dtype=np.float32)
    robot_root_pose = np.asarray(episode["states"]["articulation"]["robot"]["root_pose"], dtype=np.float32)

    t_count = min(object_root_pose.shape[0], joint_position.shape[0], robot_root_pose.shape[0])
    object_root_pose = object_root_pose[:t_count]
    joint_position = joint_position[:t_count]
    robot_root_pose = robot_root_pose[:t_count]

    if object_size_arg == "auto":
        object_size = infer_cube_size(episode["object_dynamics"]["mass"], episode["object_dynamics"]["inertia"])
    else:
        object_size = np.asarray([float(object_size_arg)] * 3, dtype=np.float32)

    object_center = object_root_pose[:, :3].astype(np.float64)
    object_rotation = quat_wxyz_to_matrix(object_root_pose[:, 3:7])

    p_hand_world, r_hand_world = compute_hand_frames(
        joint_position,
        robot_root_pose,
        robot_dof=robot_dof,
        device=device,
    )

    q_left = joint_position[:, 7].astype(np.float64)
    q_right = joint_position[:, 8].astype(np.float64)
    left_offset = np.stack([np.zeros_like(q_left), q_left, np.full_like(q_left, FINGER_JOINT_Z)], axis=-1)
    right_offset = np.stack([np.zeros_like(q_right), -q_right, np.full_like(q_right, FINGER_JOINT_Z)], axis=-1)
    r_right_collision = rot_z(math.pi)

    left_rotation = r_hand_world
    left_center = p_hand_world + np.einsum("tij,tj->ti", r_hand_world, left_offset + finger_spec.center)
    right_rotation = np.einsum("tij,jk->tik", r_hand_world, r_right_collision)
    right_center = p_hand_world + np.einsum(
        "tij,tj->ti",
        r_hand_world,
        right_offset + r_right_collision @ finger_spec.center.astype(np.float64),
    )

    if ground_z_arg == "auto":
        ground_z = ground_signed_distance(object_center[0], object_rotation[0], object_size, 0.0)
    else:
        ground_z = float(ground_z_arg)

    transforms = {
        "object_box_center": object_center,
        "object_box_rotation": object_rotation,
        "left_finger_center": left_center,
        "left_finger_rotation": left_rotation,
        "right_finger_center": right_center,
        "right_finger_rotation": right_rotation,
        "joint_position": joint_position,
    }
    return transforms, ground_z, object_size.astype(np.float32), finger_spec.size.astype(np.float32)


def fcl_transform(rotation: np.ndarray, center: np.ndarray):
    import fcl

    return fcl.Transform(
        np.ascontiguousarray(rotation, dtype=np.float64),
        np.ascontiguousarray(center, dtype=np.float64),
    )


def fcl_distance(obj_a, obj_b, request, *, nearest_points: bool) -> tuple[float, np.ndarray | None]:
    import fcl

    result = fcl.DistanceResult()
    raw_distance = float(fcl.distance(obj_a, obj_b, request, result))
    if not nearest_points or raw_distance < 0.0:
        return raw_distance, None
    points = np.asarray(result.nearest_points, dtype=np.float32)
    if points.shape != (2, 3):
        return raw_distance, None
    return raw_distance, points


def compute_episode_collision_info(
    episode,
    *,
    pair_names: list[str],
    robot_dof: int,
    object_size_arg: str | float,
    ground_z_arg: str | float,
    finger_spec: FingerBoxSpec,
    device: str,
    collision_margin: float,
    nearest_points: bool,
    store_obbs: bool,
) -> EpisodeCollisionInfo:
    import fcl

    transforms, ground_z, object_size, finger_size = compute_box_transforms(
        episode,
        robot_dof=robot_dof,
        object_size_arg=object_size_arg,
        ground_z_arg=ground_z_arg,
        finger_spec=finger_spec,
        device=device,
    )
    object_center = transforms["object_box_center"]
    object_rotation = transforms["object_box_rotation"]
    left_center = transforms["left_finger_center"]
    left_rotation = transforms["left_finger_rotation"]
    right_center = transforms["right_finger_center"]
    right_rotation = transforms["right_finger_rotation"]

    t_count = object_center.shape[0]
    p_count = len(pair_names)
    collision = np.zeros((t_count, p_count), dtype=np.bool_)
    distance = np.full((t_count, p_count), np.nan, dtype=np.float32)
    signed_distance = np.full((t_count, p_count), np.nan, dtype=np.float32)
    nearest = np.full((t_count, p_count, 2, 3), np.nan, dtype=np.float32) if nearest_points else None

    object_obj = fcl.CollisionObject(fcl.Box(*object_size.tolist()))
    left_obj = fcl.CollisionObject(fcl.Box(*finger_size.tolist()))
    right_obj = fcl.CollisionObject(fcl.Box(*finger_size.tolist()))
    distance_request = fcl.DistanceRequest(enable_nearest_points=nearest_points)

    need_object_left = any(name in pair_names for name in ("object_left_finger", "object_gripper"))
    need_object_right = any(name in pair_names for name in ("object_right_finger", "object_gripper"))
    need_left_right = "left_right_finger" in pair_names

    for t in range(t_count):
        object_obj.setTransform(fcl_transform(object_rotation[t], object_center[t]))
        left_obj.setTransform(fcl_transform(left_rotation[t], left_center[t]))
        right_obj.setTransform(fcl_transform(right_rotation[t], right_center[t]))

        pair_values: dict[str, tuple[bool, float, float, np.ndarray | None]] = {}

        if "object_ground" in pair_names:
            raw = ground_signed_distance(object_center[t], object_rotation[t], object_size, ground_z)
            pair_values["object_ground"] = (raw <= collision_margin, max(raw, 0.0), raw, None)
        if "left_finger_ground" in pair_names or "gripper_ground" in pair_names:
            raw = ground_signed_distance(left_center[t], left_rotation[t], finger_size, ground_z)
            pair_values["left_finger_ground"] = (raw <= collision_margin, max(raw, 0.0), raw, None)
        if "right_finger_ground" in pair_names or "gripper_ground" in pair_names:
            raw = ground_signed_distance(right_center[t], right_rotation[t], finger_size, ground_z)
            pair_values["right_finger_ground"] = (raw <= collision_margin, max(raw, 0.0), raw, None)

        if need_object_left:
            raw, points = fcl_distance(object_obj, left_obj, distance_request, nearest_points=nearest_points)
            pair_values["object_left_finger"] = (raw <= collision_margin, max(raw, 0.0), raw, points)
        if need_object_right:
            raw, points = fcl_distance(object_obj, right_obj, distance_request, nearest_points=nearest_points)
            pair_values["object_right_finger"] = (raw <= collision_margin, max(raw, 0.0), raw, points)
        if need_left_right:
            raw, points = fcl_distance(left_obj, right_obj, distance_request, nearest_points=nearest_points)
            pair_values["left_right_finger"] = (raw <= collision_margin, max(raw, 0.0), raw, points)

        if "object_gripper" in pair_names:
            left = pair_values["object_left_finger"]
            right = pair_values["object_right_finger"]
            chosen = left if left[1] <= right[1] else right
            pair_values["object_gripper"] = (left[0] or right[0], min(left[1], right[1]), min(left[2], right[2]), chosen[3])
        if "gripper_ground" in pair_names:
            left = pair_values["left_finger_ground"]
            right = pair_values["right_finger_ground"]
            pair_values["gripper_ground"] = (left[0] or right[0], min(left[1], right[1]), min(left[2], right[2]), None)

        for p_idx, name in enumerate(pair_names):
            collides, sep, signed, points = pair_values[name]
            collision[t, p_idx] = collides
            distance[t, p_idx] = sep
            signed_distance[t, p_idx] = signed
            if nearest is not None and points is not None:
                nearest[t, p_idx] = points

    obb_center = obb_rotation = obb_size = None
    if store_obbs:
        obb_center = np.stack([object_center, left_center, right_center], axis=1).astype(np.float32)
        obb_rotation = np.stack([object_rotation, left_rotation, right_rotation], axis=1).astype(np.float32)
        object_sizes = np.broadcast_to(object_size, (t_count, 3))
        finger_sizes = np.broadcast_to(finger_size, (t_count, 3))
        obb_size = np.stack([object_sizes, finger_sizes, finger_sizes], axis=1).astype(np.float32)

    return EpisodeCollisionInfo(
        pair_names=pair_names,
        collision=collision,
        distance=distance,
        signed_distance=signed_distance,
        nearest_points=nearest,
        ground_z=ground_z,
        obb_center=obb_center,
        obb_rotation=obb_rotation,
        obb_size=obb_size,
    )


def dataset_kwargs(compression: str) -> dict:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {"compression": compression, "shuffle": True}


def write_episode_group(
    episode,
    *,
    group_name: str,
    info: EpisodeCollisionInfo,
    finger_spec: FingerBoxSpec,
    object_size_arg: str | float,
    ground_z_arg: str | float,
    robot_dof: int,
    collision_margin: float,
    overwrite: bool,
    compression: str,
) -> None:
    if group_name in episode:
        if not overwrite:
            raise RuntimeError(f"Episode already has group '{group_name}'. Use --overwrite to replace it.")
        del episode[group_name]

    kwargs = dataset_kwargs(compression)
    group = episode.create_group(group_name)
    group.attrs["schema_version"] = SCHEMA_VERSION
    group.attrs["description"] = "Privileged collision labels for object cube, Panda gripper fingers, and ground."
    group.attrs["object_size_mode"] = str(object_size_arg)
    group.attrs["ground_z_mode"] = str(ground_z_arg)
    group.attrs["ground_z"] = float(info.ground_z)
    group.attrs["finger_box_source"] = finger_spec.source
    group.attrs["finger_box_size_xyz"] = finger_spec.size.astype(np.float32)
    group.attrs["finger_box_center_xyz"] = finger_spec.center.astype(np.float32)
    group.attrs["finger_joint_z"] = FINGER_JOINT_Z
    group.attrs["robot_dof"] = int(robot_dof)
    group.attrs["collision_margin"] = float(collision_margin)
    group.attrs["distance_note"] = (
        "distance is nonnegative separation and 0 when colliding; "
        "signed_distance is exact for ground pairs and python-fcl raw distance for box-box pairs."
    )

    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("pair_names", data=np.asarray(info.pair_names, dtype=object), dtype=string_dtype)
    group.create_dataset("collision", data=info.collision, **kwargs)
    group.create_dataset("distance", data=info.distance, **kwargs)
    group.create_dataset("signed_distance", data=info.signed_distance, **kwargs)
    if info.nearest_points is not None:
        group.create_dataset("nearest_points", data=info.nearest_points, **kwargs)
    if info.obb_center is not None and info.obb_rotation is not None and info.obb_size is not None:
        obb_group = group.create_group("obbs")
        obb_group.create_dataset("names", data=np.asarray(OBB_NAMES, dtype=object), dtype=string_dtype)
        obb_group.create_dataset("center", data=info.obb_center, **kwargs)
        obb_group.create_dataset("rotation", data=info.obb_rotation, **kwargs)
        obb_group.create_dataset("size", data=info.obb_size, **kwargs)


def augment_file(
    path: str,
    *,
    group_name: str,
    pair_names: list[str],
    robot_dof: int,
    object_size_arg: str | float,
    ground_z_arg: str | float,
    finger_spec: FingerBoxSpec,
    device: str,
    collision_margin: float,
    nearest_points: bool,
    store_obbs: bool,
    overwrite: bool,
    compression: str,
    episode_name: str | None,
    max_episodes: int,
    dry_run: bool,
) -> int:
    mode = "r" if dry_run else "r+"
    processed = 0
    with h5py.File(path, mode, locking=False) as file:
        names = sorted(file["data"].keys(), key=episode_sort_key)
        if episode_name is not None:
            if episode_name not in file["data"]:
                raise KeyError(f"Episode '{episode_name}' not found in {path}.")
            names = [episode_name]
        if max_episodes > 0:
            names = names[:max_episodes]

        for idx, name in enumerate(names, start=1):
            episode = file["data"][name]
            if dry_run:
                print(f"[DRY] {Path(path).name}:{name} would write group '{group_name}'")
                processed += 1
                continue

            info = compute_episode_collision_info(
                episode,
                pair_names=pair_names,
                robot_dof=robot_dof,
                object_size_arg=object_size_arg,
                ground_z_arg=ground_z_arg,
                finger_spec=finger_spec,
                device=device,
                collision_margin=collision_margin,
                nearest_points=nearest_points,
                store_obbs=store_obbs,
            )
            write_episode_group(
                episode,
                group_name=group_name,
                info=info,
                finger_spec=finger_spec,
                object_size_arg=object_size_arg,
                ground_z_arg=ground_z_arg,
                robot_dof=robot_dof,
                collision_margin=collision_margin,
                overwrite=overwrite,
                compression=compression,
            )
            processed += 1
            if idx == 1 or idx % 100 == 0 or idx == len(names):
                collisions = info.collision.sum(axis=0)
                summary = ", ".join(f"{pair}={int(count)}" for pair, count in zip(pair_names, collisions))
                print(f"[{Path(path).name}] {idx}/{len(names)} {name}: {summary}")
    return processed


def episode_sort_key(name: str):
    parts = name.rsplit("_", 1)
    return (int(parts[1]), name) if len(parts) == 2 and parts[1].isdigit() else (10**12, name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_file",
        action="append",
        default=[],
        help="HDF5 dataset to augment. May be passed multiple times.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./robot_object_wm/dataset",
        help="Directory of *.hdf5 datasets. Use '' to disable directory discovery.",
    )
    parser.add_argument(
        "--output_suffix",
        default="_collision_augmented",
        help="If set, copy each source HDF5 to <stem><suffix>.hdf5 and augment the copy.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory for suffixed output files. Default: same directory as each source file.",
    )
    parser.add_argument("--group_name", default="privileged_collision")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["default"],
        help=(
            "Collision pairs to add. Choices: default, all, "
            + ", ".join(BASE_PAIR_NAMES)
            + ". Comma-separated values are also accepted."
        ),
    )
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--object_size", default="auto", help="'auto' or cube side length in meters.")
    parser.add_argument(
        "--ground_z",
        default="auto",
        help="'auto' infers the support plane from frame-0 object bottom, or pass a fixed z value.",
    )
    parser.add_argument(
        "--finger_mesh",
        default="auto",
        help="Collision finger STL path. 'auto' searches installed Franka assets; '' uses measured defaults.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device used for Franka FK.")
    parser.add_argument("--collision_margin", type=float, default=0.0)
    parser.add_argument("--nearest_points", action="store_true", help="Store FCL nearest points for separated box-box pairs.")
    parser.add_argument("--store_obbs", action="store_true", help="Also store per-frame object/finger OBB transforms.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing augmentation group.")
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--episode_name", default=None, help="Only augment one named episode.")
    parser.add_argument("--max_episodes", type=int, default=0, help="Debug limit per file; 0 means all episodes.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir if args.dataset_dir else None
    paths = discover_dataset_files(args.dataset_file, dataset_dir)
    if args.output_suffix:
        paths = [path for path in paths if not Path(path).stem.endswith(args.output_suffix)]
        paths = prepare_output_files(
            paths,
            output_suffix=args.output_suffix,
            output_dir=args.output_dir or None,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    pair_names = resolve_pair_names(args.pairs)
    object_size_arg = parse_float_or_auto(args.object_size)
    ground_z_arg = parse_float_or_auto(args.ground_z)
    finger_spec = load_finger_box_spec(args.finger_mesh or None)

    print("Collision augmentation")
    print(f"  files: {len(paths)}")
    print(f"  group: {args.group_name}")
    print(f"  pairs: {', '.join(pair_names)}")
    print(f"  object_size: {object_size_arg}")
    print(f"  ground_z: {ground_z_arg}")
    print(f"  finger_box_size: {np.array2string(finger_spec.size, precision=6)}")
    print(f"  finger_box_center: {np.array2string(finger_spec.center, precision=6)}")
    print(f"  finger_box_source: {finger_spec.source}")

    total = 0
    for path in paths:
        total += augment_file(
            path,
            group_name=args.group_name,
            pair_names=pair_names,
            robot_dof=args.robot_dof,
            object_size_arg=object_size_arg,
            ground_z_arg=ground_z_arg,
            finger_spec=finger_spec,
            device=args.device,
            collision_margin=args.collision_margin,
            nearest_points=args.nearest_points,
            store_obbs=args.store_obbs,
            overwrite=args.overwrite,
            compression=args.compression,
            episode_name=args.episode_name,
            max_episodes=args.max_episodes,
            dry_run=args.dry_run,
        )
    print(f"Done. Processed {total} episode(s).")


if __name__ == "__main__":
    main()
