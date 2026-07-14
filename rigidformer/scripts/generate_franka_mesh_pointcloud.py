#!/usr/bin/env python3
"""Generate mesh-accurate Franka cube-drop point trajectories for RigidFormer.

This is the mesh/FK counterpart to the RGB-D pointcloud extraction pipeline.
It reads a Franka drop HDF5 produced by ``collect_franka_cube_drop.py`` and
writes the RigidFormer-ready schema consumed by ``train_franka_pointcloud.py``:

``/data/object_points`` with shape ``(episodes, steps, objects=2, points, 3)``.

The cube points are sampled once in the cube local frame and transformed by the
recorded cube root pose.  The gripper points are sampled once from Panda
``hand.stl`` and ``finger.stl`` meshes and transformed by forward kinematics
from the recorded Franka joint positions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise ImportError("Mesh point generation requires trimesh. Install with: pip install trimesh") from exc


OBJECT_NAMES = ("cube", "gripper")


FRANKA_MESH_DIR_CANDIDATES = (
    Path("/home/sukchul/IsaacLab/scripts/world_model/rigidformer/data/franka_objects/gripper_meshes"),
    Path("/home/sukchul/IsaacLab/scripts/world_model/rigidformer/data/franka_drop_objects/gripper_meshes"),
    Path("/home/sukchul/miniconda3/envs/torch/lib/python3.11/site-packages/isaacsim/exts")
    / "isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/meshes/collision",
    Path("/home/sukchul/miniconda3/envs/torch/lib/python3.11/site-packages/mani_skill/assets/robots/panda")
    / "franka_description/meshes/collision",
    Path("/home/sukchul/world_model/neural-robot-dynamics/envs/warp_sim_envs/assets")
    / "franka_description/meshes/collision",
)

MESH_EXTENSIONS = (".obj", ".stl")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Drop HDF5 produced by collect_franka_cube_drop.py.")
    parser.add_argument("--output-file", required=True, help="RigidFormer-ready mesh pointcloud HDF5.")
    parser.add_argument("--points", type=int, default=1024, help="Canonical points per object.")
    parser.add_argument("--cube-size", type=float, default=0.055, help="Cube side length in meters.")
    parser.add_argument(
        "--cube-obj",
        default=None,
        help="Optional cube mesh path. If omitted, an exact analytic cube surface is sampled.",
    )
    parser.add_argument(
        "--franka-mesh-dir",
        default=None,
        help="Directory containing Panda collision hand/finger meshes as .obj or .stl. Auto-detected if omitted.",
    )
    parser.add_argument("--hand-mesh", default=None, help="Optional explicit Franka/Panda hand mesh path.")
    parser.add_argument("--finger-mesh", default=None, help="Optional explicit Franka/Panda finger mesh path.")
    parser.add_argument("--train-object", choices=("cube", "both"), default="cube")
    parser.add_argument(
        "--physics-properties",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store per-episode vertex_properties as [mass, friction, restitution] from the source/drop HDF5. "
            "If disabled, store the old constant object-tag properties."
        ),
    )
    parser.add_argument(
        "--friction-field",
        choices=("static", "dynamic"),
        default="dynamic",
        help=(
            "Which Isaac material friction coefficient to use as RigidFormer's single friction property. "
            "Isaac material_properties are interpreted as [static_friction, dynamic_friction, restitution]."
        ),
    )
    parser.add_argument(
        "--gripper-properties",
        choices=("zeros", "tag"),
        default="zeros",
        help="Physics/property vector for the gripper object. 'zeros' keeps physics semantics; 'tag' uses the old [0,1,0].",
    )
    parser.add_argument("--point-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=4)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=0, help="0 means all episodes from episode-start.")
    parser.add_argument(
        "--step-policy",
        choices=("strict", "min"),
        default="strict",
        help=(
            "How to handle variable-length source episodes. 'strict' requires equal lengths; "
            "'min' trims every selected episode to the shortest selected length."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def numeric_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return sys.maxsize, name


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.compression == "none":
        return {}
    kwargs: dict[str, Any] = {"compression": args.compression, "shuffle": True}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.gzip_level)
    return kwargs


def resolve_franka_mesh_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if args.hand_mesh or args.finger_mesh:
        if not args.hand_mesh or not args.finger_mesh:
            raise ValueError("--hand-mesh and --finger-mesh must be passed together.")
        hand_path = Path(args.hand_mesh).expanduser().resolve()
        finger_path = Path(args.finger_mesh).expanduser().resolve()
        if not hand_path.is_file():
            raise FileNotFoundError(f"Hand mesh not found: {hand_path}")
        if not finger_path.is_file():
            raise FileNotFoundError(f"Finger mesh not found: {finger_path}")
        return hand_path, finger_path, None

    if args.franka_mesh_dir:
        path = Path(args.franka_mesh_dir).expanduser().resolve()
        hand_path = next((path / f"hand{ext}" for ext in MESH_EXTENSIONS if (path / f"hand{ext}").is_file()), None)
        finger_path = next((path / f"finger{ext}" for ext in MESH_EXTENSIONS if (path / f"finger{ext}").is_file()), None)
        if hand_path is None or finger_path is None:
            raise FileNotFoundError(f"Expected hand/finger meshes as .obj or .stl in {path}")
        return hand_path, finger_path, path

    for path in FRANKA_MESH_DIR_CANDIDATES:
        hand_path = next((path / f"hand{ext}" for ext in MESH_EXTENSIONS if (path / f"hand{ext}").is_file()), None)
        finger_path = next((path / f"finger{ext}" for ext in MESH_EXTENSIONS if (path / f"finger{ext}").is_file()), None)
        if hand_path is not None and finger_path is not None:
            return hand_path, finger_path, path

    raise FileNotFoundError(
        "Could not auto-detect Franka collision meshes. Pass --franka-mesh-dir containing hand/finger .obj or .stl."
    )


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Mesh has no vertices/faces: {path}")
    return loaded


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Mesh surface area is zero or invalid.")
    face_ids = rng.choice(faces.shape[0], size=count, replace=True, p=areas / total)
    selected = tri[face_ids]

    u = rng.random((count, 1))
    v = rng.random((count, 1))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = selected[:, 0] + u * (selected[:, 1] - selected[:, 0]) + v * (selected[:, 2] - selected[:, 0])
    return points.astype(np.float32)


def sample_cube_surface(size: float, count: int, rng: np.random.Generator) -> np.ndarray:
    half = float(size) * 0.5
    face_ids = rng.integers(0, 6, size=count)
    uv = rng.uniform(-half, half, size=(count, 2)).astype(np.float32)
    points = np.empty((count, 3), dtype=np.float32)

    for face in range(6):
        mask = face_ids == face
        if not np.any(mask):
            continue
        sign = 1.0 if face % 2 == 0 else -1.0
        axis = face // 2
        other_axes = [idx for idx in range(3) if idx != axis]
        points[mask, axis] = sign * half
        points[mask, other_axes[0]] = uv[mask, 0]
        points[mask, other_axes[1]] = uv[mask, 1]

    return points


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-9)
    raw = weights / weights.sum() * int(total)
    counts = np.floor(raw).astype(np.int64)
    counts = np.maximum(counts, 1)
    while counts.sum() > total:
        idx = int(np.argmax(counts))
        counts[idx] -= 1
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return counts


def canonical_cube_points(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    if args.cube_obj:
        mesh = load_mesh(args.cube_obj)
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        center = (bounds[0] + bounds[1]) * 0.5
        extent = float(np.max(bounds[1] - bounds[0]))
        if extent <= 0.0:
            raise ValueError(f"Cube mesh has invalid extent: {args.cube_obj}")
        points = sample_mesh_surface(mesh, int(args.points), rng)
        points = (points - center[None, :]) / extent * float(args.cube_size)
        return points.astype(np.float32)

    return sample_cube_surface(float(args.cube_size), int(args.points), rng)


def canonical_gripper_points(
    hand_mesh_path: Path,
    finger_mesh_path: Path,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    hand = load_mesh(hand_mesh_path)
    finger = load_mesh(finger_mesh_path)
    weights = np.asarray([hand.area, finger.area, finger.area], dtype=np.float64)
    counts = allocate_counts(count, weights)

    hand_points = sample_mesh_surface(hand, int(counts[0]), rng)
    left_points = sample_mesh_surface(finger, int(counts[1]), rng)
    right_points = sample_mesh_surface(finger, int(counts[2]), rng)

    # The right finger mesh in the Panda URDF has an extra collision origin
    # rotation of pi about z.
    right_points = transform_points(transform_from_rpy_xyz((0.0, 0.0, math.pi), (0.0, 0.0, 0.0)), right_points)

    points = np.concatenate((hand_points, left_points, right_points), axis=0).astype(np.float32)
    part_ids = np.concatenate(
        (
            np.zeros((hand_points.shape[0],), dtype=np.int8),
            np.ones((left_points.shape[0],), dtype=np.int8),
            np.full((right_points.shape[0],), 2, dtype=np.int8),
        )
    )
    if points.shape[0] != count:
        raise RuntimeError(f"Internal point count mismatch: {points.shape[0]} != {count}")
    return points, part_ids


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float32)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float32)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)


def transform_from_rpy_xyz(rpy: tuple[float, float, float], xyz: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    mat[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return mat


def revolute_z(angle: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot_z(angle)
    return mat


def prismatic(axis: tuple[float, float, float], amount: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = np.asarray(axis, dtype=np.float32) * float(amount)
    return mat


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def transform_from_pose(pose: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quat_wxyz_to_matrix(np.asarray(pose[3:7], dtype=np.float32))
    mat[:3, 3] = np.asarray(pose[:3], dtype=np.float32)
    return mat


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


PANDA_REVOLUTE_ORIGINS = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.333)),
    ((-math.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((math.pi / 2.0, 0.0, 0.0), (0.0, -0.316, 0.0)),
    ((math.pi / 2.0, 0.0, 0.0), (0.0825, 0.0, 0.0)),
    ((-math.pi / 2.0, 0.0, 0.0), (-0.0825, 0.384, 0.0)),
    ((math.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((math.pi / 2.0, 0.0, 0.0), (0.088, 0.0, 0.0)),
)


def panda_gripper_link_transforms(robot_pose: np.ndarray, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
    if joint_pos.shape[0] < 9:
        raise ValueError(f"Expected at least 9 Franka joints, got {joint_pos.shape}")

    transform = transform_from_pose(robot_pose)
    for joint_index, (rpy, xyz) in enumerate(PANDA_REVOLUTE_ORIGINS):
        transform = transform @ transform_from_rpy_xyz(rpy, xyz) @ revolute_z(float(joint_pos[joint_index]))

    link8 = transform @ transform_from_rpy_xyz((0.0, 0.0, 0.0), (0.0, 0.0, 0.107))
    hand = link8 @ transform_from_rpy_xyz((0.0, 0.0, -math.pi / 4.0), (0.0, 0.0, 0.0))
    left = hand @ transform_from_rpy_xyz((0.0, 0.0, 0.0), (0.0, 0.0, 0.0584)) @ prismatic(
        (0.0, 1.0, 0.0), float(joint_pos[7])
    )
    right = hand @ transform_from_rpy_xyz((0.0, 0.0, 0.0), (0.0, 0.0, 0.0584)) @ prismatic(
        (0.0, -1.0, 0.0), float(joint_pos[8])
    )
    return {"hand": hand, "left": left, "right": right}


def gripper_points_at(
    canonical_points: np.ndarray,
    part_ids: np.ndarray,
    robot_pose: np.ndarray,
    joint_pos: np.ndarray,
) -> np.ndarray:
    transforms = panda_gripper_link_transforms(robot_pose, joint_pos)
    out = np.empty_like(canonical_points, dtype=np.float32)
    masks = (part_ids == 0, part_ids == 1, part_ids == 2)
    names = ("hand", "left", "right")
    for mask, name in zip(masks, names):
        if np.any(mask):
            out[mask] = transform_points(transforms[name], canonical_points[mask])
    return out


def episode_names_from_source(data: h5py.Group) -> list[str]:
    return sorted((name for name in data if name.startswith("demo_")), key=numeric_demo_key)


def get_episode_steps(episode: h5py.Group) -> int:
    if "states/rigid_object/object/root_pose" in episode:
        return int(episode["states/rigid_object/object/root_pose"].shape[0])
    if "object_dynamics/root_pos_w" in episode:
        return int(episode["object_dynamics/root_pos_w"].shape[0])
    raise KeyError("Episode has no cube root pose trajectory.")


def cube_pose_trajectory(episode: h5py.Group) -> np.ndarray:
    if "states/rigid_object/object/root_pose" in episode:
        return np.asarray(episode["states/rigid_object/object/root_pose"], dtype=np.float32)
    if "object_dynamics/root_pos_w" in episode and "object_dynamics/root_quat_w" in episode:
        return np.concatenate(
            (
                np.asarray(episode["object_dynamics/root_pos_w"], dtype=np.float32),
                np.asarray(episode["object_dynamics/root_quat_w"], dtype=np.float32),
            ),
            axis=-1,
        )
    raise KeyError("Episode has no cube root pose trajectory.")


def robot_pose_trajectory(episode: h5py.Group, steps: int) -> np.ndarray:
    if "states/articulation/robot/root_pose" in episode:
        return np.asarray(episode["states/articulation/robot/root_pose"][:steps], dtype=np.float32)
    pose = np.zeros((steps, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    return pose


def joint_position_trajectory(episode: h5py.Group, steps: int) -> np.ndarray:
    if "states/articulation/robot/joint_position" not in episode:
        raise KeyError("Episode has no states/articulation/robot/joint_position.")
    return np.asarray(episode["states/articulation/robot/joint_position"][:steps], dtype=np.float32)


def _first_finite_scalar(array: np.ndarray, *, name: str) -> float:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values.")
    return float(finite[0])


def _first_material_triplet(array: np.ndarray) -> np.ndarray:
    material = np.asarray(array, dtype=np.float32)
    if material.size < 3:
        raise ValueError(f"Material properties must contain at least 3 values, got shape {material.shape}.")
    material = material.reshape(-1, 3)
    finite_rows = material[np.isfinite(material).all(axis=1)]
    if finite_rows.size == 0:
        raise ValueError(f"Material properties contain no finite triplet, got shape {material.shape}.")
    return finite_rows[0].astype(np.float32)


def cube_physics_properties(episode: h5py.Group, *, friction_field: str) -> np.ndarray:
    """Return [mass, friction, restitution] for the cube in one episode."""
    if "episode_physics_randomization/object_mass" in episode:
        mass = _first_finite_scalar(episode["episode_physics_randomization/object_mass"], name="object_mass")
    elif "object_dynamics/mass" in episode:
        mass = _first_finite_scalar(episode["object_dynamics/mass"], name="object_dynamics/mass")
    else:
        raise KeyError("Episode has no object mass at episode_physics_randomization/object_mass or object_dynamics/mass.")

    if "episode_physics_randomization/object_contact" in episode:
        material = _first_material_triplet(episode["episode_physics_randomization/object_contact"])
    elif "object_dynamics/material_properties" in episode:
        material = _first_material_triplet(episode["object_dynamics/material_properties"])
    else:
        raise KeyError(
            "Episode has no material properties at episode_physics_randomization/object_contact "
            "or object_dynamics/material_properties."
        )

    friction_index = 0 if friction_field == "static" else 1
    return np.asarray([mass, float(material[friction_index]), float(material[2])], dtype=np.float32)


def gripper_property_vector(args: argparse.Namespace) -> np.ndarray:
    if args.gripper_properties == "tag":
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return np.zeros((3,), dtype=np.float32)


def legacy_tag_properties() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def create_or_open_output(
    args: argparse.Namespace,
    *,
    episode_names: list[str],
    source_episode_steps: np.ndarray,
    steps: int,
    hand_mesh_path: Path,
    finger_mesh_path: Path,
    mesh_dir: Path | None,
) -> h5py.File:
    output_path = Path(args.output_file).expanduser().resolve()
    exists = output_path.exists()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if exists and args.overwrite:
        output_path.unlink()
        exists = False
    if exists and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume or --overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file = h5py.File(output_path, "r+" if exists else "w")
    point_dtype = np.float16 if str(args.point_dtype) == "float16" else np.float32

    if exists:
        points = file["data/object_points"]
        expected_shape = (len(episode_names), int(steps), len(OBJECT_NAMES), int(args.points), 3)
        if points.shape != expected_shape:
            file.close()
            raise RuntimeError(f"Cannot resume with shape {points.shape}; expected {expected_shape}")
        expected_props_shape = (
            (len(episode_names), len(OBJECT_NAMES), 3)
            if bool(args.physics_properties)
            else (len(OBJECT_NAMES), 3)
        )
        current_props_shape = file["data/vertex_properties"].shape if "vertex_properties" in file["data"] else None
        if current_props_shape != expected_props_shape:
            file.close()
            raise RuntimeError(
                "Cannot resume with incompatible /data/vertex_properties shape "
                f"{current_props_shape}; expected {expected_props_shape}."
            )
        return file

    data = file.require_group("data")
    kwargs = dataset_kwargs(args)
    chunk = (1, int(steps), len(OBJECT_NAMES), int(args.points), 3)
    data.create_dataset(
        "object_points",
        shape=(len(episode_names), int(steps), len(OBJECT_NAMES), int(args.points), 3),
        dtype=point_dtype,
        chunks=chunk,
        **kwargs,
    )
    data.create_dataset(
        "object_point_counts",
        shape=(len(episode_names), int(steps), len(OBJECT_NAMES)),
        dtype=np.int32,
        chunks=(1, int(steps), len(OBJECT_NAMES)),
        **kwargs,
    )
    data.create_dataset("done", data=np.zeros((len(episode_names),), dtype=bool), chunks=(min(len(episode_names), 1024),))
    data.create_dataset("object_names", data=np.asarray(OBJECT_NAMES, dtype=object), dtype=h5py.string_dtype("utf-8"))
    data.create_dataset("episode_names", data=np.asarray(episode_names, dtype=object), dtype=h5py.string_dtype("utf-8"))
    data.create_dataset("source_episode_steps", data=np.asarray(source_episode_steps, dtype=np.int32))
    if bool(args.physics_properties):
        data.create_dataset(
            "vertex_properties",
            shape=(len(episode_names), len(OBJECT_NAMES), 3),
            dtype=np.float32,
            chunks=(min(len(episode_names), 1024), len(OBJECT_NAMES), 3),
        )
    else:
        data.create_dataset("vertex_properties", data=legacy_tag_properties())
    if args.train_object == "cube":
        loss_mask = np.asarray([True, False], dtype=bool)
    else:
        loss_mask = np.asarray([True, True], dtype=bool)
    data.create_dataset("loss_object_mask", data=loss_mask)

    source_attrs: dict[str, Any] = {}
    with h5py.File(args.input_file, "r", locking=False) as source:
        for key in ("control_dt", "support_plane_z", "support_plane_frame", "schema_version"):
            if key in source.attrs:
                value = source.attrs[key]
                source_attrs[key] = value.item() if hasattr(value, "item") else value

    control_dt = float(source_attrs.get("control_dt", 0.02))
    file.attrs["schema_version"] = "franka_rigidformer_mesh_pointcloud_v1"
    file.attrs["source_hdf5"] = os.path.abspath(args.input_file)
    file.attrs["rigidformer_pointcloud_config"] = json.dumps(
        {
            "method": "mesh_fk",
            "points": int(args.points),
            "objects": list(OBJECT_NAMES),
            "train_object": args.train_object,
            "point_dtype": args.point_dtype,
            "physics_properties": bool(args.physics_properties),
            "vertex_properties_semantics": (
                f"per-episode [mass, {args.friction_field}_friction, restitution] for cube; "
                f"gripper={args.gripper_properties}"
                if bool(args.physics_properties)
                else "legacy constant object tags"
            ),
            "friction_field": str(args.friction_field),
            "gripper_properties": str(args.gripper_properties),
            "cube_size": float(args.cube_size),
            "cube_obj": os.path.abspath(args.cube_obj) if args.cube_obj else None,
            "franka_mesh_dir": str(mesh_dir) if mesh_dir is not None else None,
            "hand_mesh": str(hand_mesh_path),
            "finger_mesh": str(finger_mesh_path),
            "step_policy": str(args.step_policy),
            "steps": int(steps),
            "source_episode_steps_min": int(np.min(source_episode_steps)),
            "source_episode_steps_max": int(np.max(source_episode_steps)),
            "control_dt": control_dt,
            "source_attrs": source_attrs,
        },
        sort_keys=True,
    )
    file.flush()
    return file


def main() -> str:
    args = parse_cli()
    if int(args.points) < 4:
        raise ValueError("--points must be >= 4 for RigidFormer anchor sampling.")
    hand_mesh_path, finger_mesh_path, mesh_dir = resolve_franka_mesh_paths(args)
    rng = np.random.default_rng(int(args.seed))
    cube_canonical = canonical_cube_points(args, rng)
    gripper_canonical, gripper_part_ids = canonical_gripper_points(
        hand_mesh_path,
        finger_mesh_path,
        int(args.points),
        rng,
    )

    with h5py.File(args.input_file, "r", locking=False) as source:
        data = source["data"]
        all_episode_names = episode_names_from_source(data)
        start = int(args.episode_start)
        stop = len(all_episode_names) if int(args.num_episodes) <= 0 else min(
            len(all_episode_names), start + int(args.num_episodes)
        )
        episode_names = all_episode_names[start:stop]
        if not episode_names:
            raise ValueError("No episodes selected.")
        source_episode_steps = np.asarray([get_episode_steps(data[name]) for name in episode_names], dtype=np.int32)
        if str(args.step_policy) == "strict":
            steps = int(source_episode_steps[0])
            if not np.all(source_episode_steps == steps):
                raise RuntimeError(
                    "All selected episodes must have the same number of samples. "
                    f"Got min={int(source_episode_steps.min())}, max={int(source_episode_steps.max())}; "
                    "rerun with --step-policy min to trim every episode to the shortest selected length."
                )
        else:
            steps = int(source_episode_steps.min())
            print(
                "[INFO] Variable source episode lengths detected; "
                f"using --step-policy min and trimming all selected episodes to {steps} steps "
                f"(source min={int(source_episode_steps.min())}, max={int(source_episode_steps.max())})."
            )

    output = create_or_open_output(
        args,
        episode_names=episode_names,
        source_episode_steps=source_episode_steps,
        steps=steps,
        hand_mesh_path=hand_mesh_path,
        finger_mesh_path=finger_mesh_path,
        mesh_dir=mesh_dir,
    )
    try:
        out = output["data"]
        done = out["done"]
        completed_before = int(np.asarray(done).sum())
        print(
            f"[INFO] Generating mesh/FK point trajectories for {len(episode_names)} episodes x {steps} steps "
            f"at points={args.points}, hand_mesh={hand_mesh_path}, finger_mesh={finger_mesh_path}."
        )
        if completed_before:
            print(f"[INFO] Resume detected: {completed_before}/{len(episode_names)} episodes already complete.")

        with h5py.File(args.input_file, "r", locking=False) as source:
            data = source["data"]
            for local_index, episode_name in enumerate(episode_names):
                if bool(done[local_index]):
                    continue
                episode = data[episode_name]
                cube_poses = cube_pose_trajectory(episode)[:steps]
                robot_poses = robot_pose_trajectory(episode, steps)
                joint_positions = joint_position_trajectory(episode, steps)

                if bool(args.physics_properties):
                    out["vertex_properties"][local_index, 0] = cube_physics_properties(
                        episode,
                        friction_field=str(args.friction_field),
                    )
                    out["vertex_properties"][local_index, 1] = gripper_property_vector(args)

                object_points = np.empty((steps, len(OBJECT_NAMES), int(args.points), 3), dtype=np.float32)
                for frame in range(steps):
                    object_points[frame, 0] = transform_points(transform_from_pose(cube_poses[frame]), cube_canonical)
                    object_points[frame, 1] = gripper_points_at(
                        gripper_canonical,
                        gripper_part_ids,
                        robot_poses[frame],
                        joint_positions[frame],
                    )

                out["object_points"][local_index] = object_points.astype(out["object_points"].dtype, copy=False)
                out["object_point_counts"][local_index] = int(args.points)
                done[local_index] = True
                if (local_index + 1) % int(args.progress_every) == 0 or local_index == len(episode_names) - 1:
                    output.flush()
                    print(f"[INFO] {local_index + 1}/{len(episode_names)} episodes generated.")
        output.flush()
    finally:
        output.close()

    output_path = os.path.abspath(args.output_file)
    print(f"[INFO] Mesh/FK RigidFormer pointcloud dataset ready: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
