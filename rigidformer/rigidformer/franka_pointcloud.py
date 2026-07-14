from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _split_indices(
    count: int,
    *,
    split: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Unknown split: {split!r}")
    indices = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if split == "all":
        return indices
    train_end = int(np.ceil(train_ratio * count))
    val_end = train_end + int(np.ceil(val_ratio * count))
    if split == "train":
        return indices[:train_end]
    if split == "val":
        return indices[train_end:val_end]
    return indices[val_end:]


def parse_stride_choices(value: str | None) -> tuple[int, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    choices = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not choices:
        return None
    if any(choice < 1 for choice in choices):
        raise ValueError("Stride choices must be >= 1")
    return choices


class FrankaPointCloudRigidDataset(Dataset):
    """RigidFormer-ready Franka cube-drop pointcloud dataset.

    Expected HDF5 schema, produced by ``scripts/calibrate_franka_pointcloud.py``:

    ``/data/object_points``: ``(episodes, steps, objects, points, 3)``
    ``/data/loss_object_mask``: ``(objects,)`` or ``(episodes, objects)``
    ``/data/vertex_properties``: ``(objects, 3)`` or ``(episodes, objects, 3)``
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        split: str = "train",
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        max_points: int = 512,
        min_stride: int = 1,
        max_stride: int = 1,
        stride_choices: tuple[int, ...] | list[int] | None = None,
        sample_count: int | None = None,
        seed: int = 0,
        random_points: bool = True,
    ):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.max_points = int(max_points)
        self.min_stride = int(min_stride)
        self.max_stride = int(max_stride)
        self.stride_choices = tuple(int(stride) for stride in stride_choices) if stride_choices is not None else None
        self.sample_count = sample_count
        self.seed = int(seed)
        self.random_points = bool(random_points)
        self._file: h5py.File | None = None

        if self.max_points < 4:
            raise ValueError("max_points must be at least 4 for RigidFormer anchor sampling.")
        if self.min_stride < 1 or self.max_stride < self.min_stride:
            raise ValueError("Expected 1 <= min_stride <= max_stride")
        if self.stride_choices is not None and any(stride < 1 for stride in self.stride_choices):
            raise ValueError("All stride choices must be >= 1")
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(f"Pointcloud HDF5 not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r", locking=False) as file:
            points = file["data/object_points"]
            if points.ndim != 5 or points.shape[-1] != 3:
                raise ValueError("/data/object_points must have shape (episodes, steps, objects, points, 3)")
            self.num_episodes, self.num_steps, self.num_objects, self.stored_points, _ = points.shape
            if self.max_points > self.stored_points:
                raise ValueError(f"max_points={self.max_points} exceeds stored points={self.stored_points}")
            self.object_names = [
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in np.asarray(file["data"].get("object_names", []))
            ]
            self.config = json.loads(str(file.attrs.get("rigidformer_pointcloud_config", "{}")))

        max_possible_stride = max(self.stride_choices) if self.stride_choices is not None else self.max_stride
        if self.num_steps <= 2 * max_possible_stride:
            raise ValueError(
                f"Dataset has only {self.num_steps} steps, not enough for stride {max_possible_stride}."
            )

        self.episode_indices = _split_indices(
            self.num_episodes,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        if self.episode_indices.size == 0:
            raise ValueError(f"Split {split!r} is empty.")

        # A dataset item draws one temporal triple.  This length gives one item
        # per possible center frame by default, but frame/stride are randomized
        # deterministically from the item index.
        windows_per_episode = max(1, self.num_steps - 2 * max_possible_stride)
        default_len = int(self.episode_indices.size * windows_per_episode)
        self.length = int(sample_count) if sample_count is not None else default_len

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r", locking=False)
        return self._file

    def __len__(self) -> int:
        return self.length

    def _point_indices(self, rng: np.random.Generator) -> np.ndarray:
        if self.max_points == self.stored_points:
            return np.arange(self.stored_points, dtype=np.int64)
        if self.random_points:
            return np.sort(rng.choice(self.stored_points, size=self.max_points, replace=False)).astype(np.int64)
        return np.arange(self.max_points, dtype=np.int64)

    def _stride(self, rng: np.random.Generator) -> int:
        if self.stride_choices is not None:
            return int(rng.choice(self.stride_choices))
        return int(rng.integers(self.min_stride, self.max_stride + 1))

    def _vertex_properties(self, episode_index: int) -> np.ndarray:
        data = self.file["data"]
        if "vertex_properties" not in data:
            default_props = np.zeros((self.num_objects, 3), dtype=np.float32)
            if self.num_objects > 0:
                default_props[0] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
            if self.num_objects > 1:
                default_props[1] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            return default_props
        props = data["vertex_properties"]
        if props.ndim == 2:
            return np.asarray(props, dtype=np.float32)
        return np.asarray(props[episode_index], dtype=np.float32)

    def _loss_object_mask(self, episode_index: int) -> np.ndarray:
        data = self.file["data"]
        if "loss_object_mask" not in data:
            mask = np.ones((self.num_objects,), dtype=bool)
            if self.num_objects > 1:
                mask[1:] = False
            return mask
        mask = data["loss_object_mask"]
        if mask.ndim == 1:
            return np.asarray(mask, dtype=bool)
        return np.asarray(mask[episode_index], dtype=bool)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        rng = np.random.default_rng(self.seed + int(index) * 7919)
        episode_index = int(self.episode_indices[int(index) % len(self.episode_indices)])
        stride = self._stride(rng)
        frame_index = int(rng.integers(stride, self.num_steps - stride))
        point_indices = self._point_indices(rng)

        points = self.file["data/object_points"]
        # HDF5 fancy indexing requires strictly increasing indices, so frame
        # lists like [0, 0, 1, 2] fail when the previous frame is also the first
        # frame.  Read the four required frames independently, then point-subset
        # in numpy.
        frame_indices = (0, frame_index - stride, frame_index, frame_index + stride)
        frame_points = np.stack(
            [np.asarray(points[episode_index, frame, :, :, :], dtype=np.float32) for frame in frame_indices],
            axis=0,
        )
        frame_points = frame_points[:, :, point_indices, :]
        frame_points = np.nan_to_num(frame_points, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "delta_times": torch.tensor(float(stride) * float(self.config.get("control_dt", 0.02)), dtype=torch.float32),
            "vertex_properties": torch.tensor(self._vertex_properties(episode_index), dtype=torch.float32),
            "object_first_frame_pos": torch.tensor(frame_points[0], dtype=torch.float32),
            "object_pos_prev": torch.tensor(frame_points[1], dtype=torch.float32),
            "object_pos": torch.tensor(frame_points[2], dtype=torch.float32),
            "object_pos_next": torch.tensor(frame_points[3], dtype=torch.float32),
            "object_point_lens": torch.full((self.num_objects,), self.max_points, dtype=torch.long),
            "object_lens": self.num_objects,
            "loss_object_mask": torch.tensor(self._loss_object_mask(episode_index), dtype=torch.bool),
            "episode_index": episode_index,
            "frame_index": frame_index,
            "stride": stride,
        }


def franka_pointcloud_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(samples)
    max_objects = max(int(sample["object_lens"]) for sample in samples)
    max_points = int(samples[0]["object_pos"].shape[-2])

    out: dict[str, Any] = {
        "delta_times": torch.stack([sample["delta_times"] for sample in samples]),
        "vertex_properties": torch.zeros((batch, max_objects, 3), dtype=torch.float32),
        "object_pos_prev": torch.zeros((batch, max_objects, max_points, 3), dtype=torch.float32),
        "object_pos": torch.zeros((batch, max_objects, max_points, 3), dtype=torch.float32),
        "object_pos_next": torch.zeros((batch, max_objects, max_points, 3), dtype=torch.float32),
        "object_first_frame_pos": torch.zeros((batch, max_objects, max_points, 3), dtype=torch.float32),
        "object_point_lens": torch.ones((batch, max_objects), dtype=torch.long),
        "object_lens": torch.tensor([sample["object_lens"] for sample in samples], dtype=torch.long),
        "loss_object_mask": torch.zeros((batch, max_objects), dtype=torch.bool),
        "episode_index": torch.tensor([sample["episode_index"] for sample in samples], dtype=torch.long),
        "frame_index": torch.tensor([sample["frame_index"] for sample in samples], dtype=torch.long),
        "stride": torch.tensor([sample["stride"] for sample in samples], dtype=torch.long),
    }

    for batch_index, sample in enumerate(samples):
        num_objects = int(sample["object_lens"])
        out["vertex_properties"][batch_index, :num_objects] = sample["vertex_properties"]
        out["object_pos_prev"][batch_index, :num_objects] = sample["object_pos_prev"]
        out["object_pos"][batch_index, :num_objects] = sample["object_pos"]
        out["object_pos_next"][batch_index, :num_objects] = sample["object_pos_next"]
        out["object_first_frame_pos"][batch_index, :num_objects] = sample["object_first_frame_pos"]
        out["object_point_lens"][batch_index, :num_objects] = sample["object_point_lens"]
        out["loss_object_mask"][batch_index, :num_objects] = sample["loss_object_mask"]

    return out
