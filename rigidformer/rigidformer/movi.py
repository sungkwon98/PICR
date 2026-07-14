from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "MOVi loading requires trimesh. Install it with `pip install trimesh`."
    ) from exc


def _split_sample_paths(
    sample_paths: list[Path],
    split: str,
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[Path]:
    assert split in {"train", "val", "test", "all"}

    if split == "all":
        return sample_paths

    rng = np.random.default_rng(seed)
    indices = np.arange(len(sample_paths))
    rng.shuffle(indices)

    train_end = int(np.ceil(train_ratio * len(indices)))
    val_end = train_end + int(np.ceil(val_ratio * len(indices)))

    if split == "train":
        selected = indices[:train_end]
    elif split == "val":
        selected = indices[train_end:val_end]
    else:
        selected = indices[val_end:]

    return [sample_paths[i] for i in selected]


def movi_sample_paths(
    dataset_dir: str | Path,
    *,
    split: str = "all",
    seed: int = 0,
    sample_count: int | None = None,
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")

    def sample_sort_key(path: Path) -> tuple[bool, int | str]:
        return (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name)

    paths = [
        path
        for path in sorted(dataset_dir.iterdir(), key = sample_sort_key)
        if path.is_dir() and (path / "metadata.json").is_file()
    ]

    paths = _split_sample_paths(paths, split = split, seed = seed)
    return paths[:sample_count] if sample_count is not None else paths


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    loaded = trimesh.load(mesh_path, process = False)

    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate = True)

    vertices = np.asarray(loaded.vertices, dtype = np.float32)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise RuntimeError(f"Unexpected mesh vertices in {mesh_path}")

    return vertices


def _quaternion_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype = np.float32)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    w, x, y, z = quat

    return np.array([
        [1. - 2. * (y * y + z * z), 2. * (x * y - z * w), 2. * (x * z + y * w)],
        [2. * (x * y + z * w), 1. - 2. * (x * x + z * z), 2. * (y * z - x * w)],
        [2. * (x * z - y * w), 2. * (y * z + x * w), 1. - 2. * (x * x + y * y)],
    ], dtype = np.float32)


def _transform_vertices(
    vertices: np.ndarray,
    position: list[float],
    quaternion: list[float],
) -> np.ndarray:
    rot = _quaternion_wxyz_to_matrix(np.asarray(quaternion, dtype = np.float32))
    trans = np.asarray(position, dtype = np.float32)
    return vertices @ rot.T + trans


def _point_indices(
    num_points: int,
    max_points: int,
    *,
    random_points: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points)

    if random_points:
        return np.sort(rng.choice(num_points, max_points, replace = False))

    return np.linspace(0, num_points - 1, max_points).astype(np.int64)


def load_movi_scene_trajectory(
    sample_path: str | Path,
    objects_dir: str | Path,
    *,
    frame_indices: list[int] | np.ndarray,
    max_points: int = 256,
    random_points: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    sample_path = Path(sample_path)
    objects_dir = Path(objects_dir)
    metadata_path = sample_path / "metadata.json"

    if not metadata_path.is_file():
        raise RuntimeError(f"Missing metadata file: {metadata_path}")

    if not objects_dir.is_dir():
        raise RuntimeError(f"Objects directory does not exist: {objects_dir}")

    with metadata_path.open("r") as f:
        metadata = json.load(f)

    objects = metadata["instances"]
    frame_rate = float(metadata["metadata"].get("frame_rate", 240))
    num_frames = min(len(obj["positions"]) for obj in objects)
    frame_indices = [int(frame_index) for frame_index in frame_indices]

    if not frame_indices:
        raise ValueError("frame_indices must not be empty")

    if min(frame_indices) < 0 or max(frame_indices) >= num_frames:
        raise ValueError(f"Frame indices must be within [0, {num_frames - 1}]")

    rng = np.random.default_rng(seed)
    mesh_cache: dict[str, np.ndarray] = {}

    object_positions: list[np.ndarray] = []
    object_first_frame_positions: list[np.ndarray] = []
    vertex_properties: list[np.ndarray] = []
    point_lens: list[int] = []

    for obj in objects:
        shape = obj.get("shape") or obj["asset_id"]
        scale = float(obj.get("size", 1.) or 1.)

        if shape not in mesh_cache:
            mesh_path = objects_dir / shape / "collision_geometry.obj"
            if not mesh_path.is_file():
                raise RuntimeError(f"Missing collision mesh for shape {shape!r}: {mesh_path}")
            mesh_cache[shape] = _load_mesh_vertices(mesh_path)

        base_vertices = mesh_cache[shape] * scale
        indices = _point_indices(
            base_vertices.shape[0],
            max_points,
            random_points = random_points,
            rng = rng,
        )

        selected_vertices = base_vertices[indices]
        point_len = selected_vertices.shape[0]
        point_lens.append(point_len)

        def padded_points_at(frame_index: int) -> np.ndarray:
            transformed = _transform_vertices(
                selected_vertices,
                obj["positions"][frame_index],
                obj["quaternions"][frame_index],
            )

            padded = np.zeros((max_points, 3), dtype = np.float32)
            padded[:point_len] = transformed
            return padded

        object_first_frame_positions.append(padded_points_at(0))
        object_positions.append(np.stack([padded_points_at(frame_index) for frame_index in frame_indices]))

        vertex_properties.append(np.asarray([
            float(obj.get("mass", 0.)),
            float(obj.get("friction", 0.)),
            float(obj.get("restitution", 0.)),
        ], dtype = np.float32))

    positions = np.stack(object_positions, axis = 1)

    return {
        "sample_id": sample_path.name,
        "metadata": metadata,
        "frame_rate": frame_rate,
        "frame_indices": np.asarray(frame_indices, dtype = np.int64),
        "positions": torch.tensor(positions, dtype = torch.float32),
        "first_frame_pos": torch.tensor(np.stack(object_first_frame_positions), dtype = torch.float32),
        "vertex_properties": torch.tensor(np.stack(vertex_properties), dtype = torch.float32),
        "object_point_lens": torch.tensor(point_lens, dtype = torch.long),
        "object_lens": len(objects),
    }


class MoviMetadataDataset(Dataset):
    """HOPNet / RigidFormer-style MOVi metadata dataset.

    The public MOVi benchmark archives store rigid-body trajectories as object
    positions and quaternions in per-scene metadata.json files. This dataset
    reconstructs per-object point trajectories from those poses and the KuBasic
    collision meshes, then emits the tensors expected by Rigidformer.forward.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        objects_dir: str | Path,
        *,
        split: str = "train",
        max_points: int = 256,
        min_stride: int = 1,
        max_stride: int = 1,
        stride_choices: tuple[int, ...] | list[int] | None = None,
        sample_count: int | None = None,
        seed: int = 0,
        random_points: bool = True,
        object_permutation_prob: float = 0.,
    ):
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        self.objects_dir = Path(objects_dir)
        self.max_points = max_points
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.stride_choices = tuple(int(stride) for stride in stride_choices) if stride_choices is not None else None
        self.random_points = random_points
        self.seed = seed
        self.object_permutation_prob = object_permutation_prob

        if max_points < 4:
            raise ValueError("max_points must be at least 4 for anchor sampling")

        if min_stride < 1 or max_stride < min_stride:
            raise ValueError("Expected 1 <= min_stride <= max_stride")

        if self.stride_choices is not None:
            if not self.stride_choices:
                raise ValueError("stride_choices must not be empty")

            if any(stride < 1 for stride in self.stride_choices):
                raise ValueError("All stride choices must be >= 1")

        if not 0. <= object_permutation_prob <= 1.:
            raise ValueError("Expected 0 <= object_permutation_prob <= 1")

        if not self.dataset_dir.is_dir():
            raise RuntimeError(f"Dataset directory does not exist: {self.dataset_dir}")

        if not self.objects_dir.is_dir():
            raise RuntimeError(f"Objects directory does not exist: {self.objects_dir}")

        def sample_sort_key(path: Path) -> tuple[bool, int | str]:
            return (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name)

        sample_paths = [
            path
            for path in sorted(self.dataset_dir.iterdir(), key = sample_sort_key)
            if path.is_dir() and (path / "metadata.json").is_file()
        ]

        if not sample_paths:
            raise RuntimeError(f"No numbered samples with metadata.json found in {self.dataset_dir}")

        self.sample_paths = _split_sample_paths(sample_paths, split = split, seed = seed)

        if sample_count is not None:
            self.sample_paths = self.sample_paths[:sample_count]

        if not self.sample_paths:
            raise RuntimeError(f"Split {split!r} is empty for {self.dataset_dir}")

        self._mesh_vertices: dict[str, np.ndarray] = {}
        self._metadata_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _metadata(self, sample_path: Path) -> dict[str, Any]:
        if sample_path not in self._metadata_cache:
            with (sample_path / "metadata.json").open("r") as f:
                self._metadata_cache[sample_path] = json.load(f)

        return self._metadata_cache[sample_path]

    def _base_vertices(self, shape: str) -> np.ndarray:
        if shape not in self._mesh_vertices:
            mesh_path = self.objects_dir / shape / "collision_geometry.obj"
            if not mesh_path.is_file():
                raise RuntimeError(f"Missing collision mesh for shape {shape!r}: {mesh_path}")
            self._mesh_vertices[shape] = _load_mesh_vertices(mesh_path)

        return self._mesh_vertices[shape]

    def _point_indices(self, num_points: int) -> np.ndarray:
        return _point_indices(
            num_points,
            self.max_points,
            random_points = self.random_points,
            rng = np.random.default_rng(),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample_path = self.sample_paths[index]
        metadata = self._metadata(sample_path)

        objects = metadata["instances"]
        frame_rate = float(metadata["metadata"].get("frame_rate", 240))
        num_frames = min(len(obj["positions"]) for obj in objects)

        if self.stride_choices is not None:
            stride = random.choice(self.stride_choices)
        else:
            stride = random.randint(self.min_stride, self.max_stride)

        if num_frames <= 2 * stride:
            raise RuntimeError(f"Not enough frames in {sample_path} for stride {stride}")

        frame_index = random.randint(stride, num_frames - stride - 1)
        frame_indices = {
            "first": 0,
            "prev": frame_index - stride,
            "current": frame_index,
            "next": frame_index + stride,
        }

        object_pos_prev: list[np.ndarray] = []
        object_pos: list[np.ndarray] = []
        object_pos_next: list[np.ndarray] = []
        object_first_frame_pos: list[np.ndarray] = []
        vertex_properties: list[np.ndarray] = []
        point_lens: list[int] = []

        for obj in objects:
            shape = obj.get("shape") or obj["asset_id"]
            scale = float(obj.get("size", 1.) or 1.)
            base_vertices = self._base_vertices(shape) * scale

            indices = self._point_indices(base_vertices.shape[0])
            selected_vertices = base_vertices[indices]
            point_len = selected_vertices.shape[0]

            def points_at(which: str) -> np.ndarray:
                t = frame_indices[which]
                points = _transform_vertices(
                    selected_vertices,
                    obj["positions"][t],
                    obj["quaternions"][t],
                )

                padded = np.zeros((self.max_points, 3), dtype = np.float32)
                padded[:point_len] = points
                return padded

            object_first_frame_pos.append(points_at("first"))
            object_pos_prev.append(points_at("prev"))
            object_pos.append(points_at("current"))
            object_pos_next.append(points_at("next"))

            vertex_properties.append(np.asarray([
                float(obj.get("mass", 0.)),
                float(obj.get("friction", 0.)),
                float(obj.get("restitution", 0.)),
            ], dtype = np.float32))

            point_lens.append(point_len)

        if len(objects) > 1 and random.random() < self.object_permutation_prob:
            permutation = list(range(len(objects)))
            random.shuffle(permutation)

            object_first_frame_pos = [object_first_frame_pos[i] for i in permutation]
            object_pos_prev = [object_pos_prev[i] for i in permutation]
            object_pos = [object_pos[i] for i in permutation]
            object_pos_next = [object_pos_next[i] for i in permutation]
            vertex_properties = [vertex_properties[i] for i in permutation]
            point_lens = [point_lens[i] for i in permutation]

        return {
            "delta_times": torch.tensor(stride / frame_rate, dtype = torch.float32),
            "vertex_properties": torch.tensor(np.stack(vertex_properties), dtype = torch.float32),
            "object_pos_prev": torch.tensor(np.stack(object_pos_prev), dtype = torch.float32),
            "object_pos": torch.tensor(np.stack(object_pos), dtype = torch.float32),
            "object_pos_next": torch.tensor(np.stack(object_pos_next), dtype = torch.float32),
            "object_first_frame_pos": torch.tensor(np.stack(object_first_frame_pos), dtype = torch.float32),
            "object_point_lens": torch.tensor(point_lens, dtype = torch.long),
            "object_lens": len(objects),
            "sample_id": sample_path.name,
            "frame_index": frame_index,
            "stride": stride,
        }


def movi_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(samples)
    max_objects = max(int(sample["object_lens"]) for sample in samples)
    max_points = samples[0]["object_pos"].shape[-2]

    out = {
        "delta_times": torch.stack([sample["delta_times"] for sample in samples]),
        "vertex_properties": torch.zeros((batch, max_objects, 3), dtype = torch.float32),
        "object_pos_prev": torch.zeros((batch, max_objects, max_points, 3), dtype = torch.float32),
        "object_pos": torch.zeros((batch, max_objects, max_points, 3), dtype = torch.float32),
        "object_pos_next": torch.zeros((batch, max_objects, max_points, 3), dtype = torch.float32),
        "object_first_frame_pos": torch.zeros((batch, max_objects, max_points, 3), dtype = torch.float32),
        "object_point_lens": torch.ones((batch, max_objects), dtype = torch.long),
        "object_lens": torch.tensor([sample["object_lens"] for sample in samples], dtype = torch.long),
        "sample_id": [sample["sample_id"] for sample in samples],
        "frame_index": torch.tensor([sample["frame_index"] for sample in samples], dtype = torch.long),
        "stride": torch.tensor([sample["stride"] for sample in samples], dtype = torch.long),
    }

    for batch_index, sample in enumerate(samples):
        num_objects = int(sample["object_lens"])
        out["vertex_properties"][batch_index, :num_objects] = sample["vertex_properties"]
        out["object_pos_prev"][batch_index, :num_objects] = sample["object_pos_prev"]
        out["object_pos"][batch_index, :num_objects] = sample["object_pos"]
        out["object_pos_next"][batch_index, :num_objects] = sample["object_pos_next"]
        out["object_first_frame_pos"][batch_index, :num_objects] = sample["object_first_frame_pos"]
        out["object_point_lens"][batch_index, :num_objects] = sample["object_point_lens"]

    return out
