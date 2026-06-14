from __future__ import annotations

import glob
import os
import random
from typing import List

try:
    import h5py
except ImportError:  # pragma: no cover - depends on local environment.
    h5py = None

from .hdf5_schema import EpisodeRef, Hdf5Groups

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def h5_open(path: str):
    if h5py is None:
        raise RuntimeError("h5py is required to load robot-object world-model datasets.") from None
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def discover_hdf5_files(dataset_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(os.path.abspath(dataset_dir), "*.hdf5")))


def load_episode_names(hdf5_path: str) -> List[str]:
    with h5_open(hdf5_path) as file:
        return list(file[Hdf5Groups.DATA].keys())


def load_all_episode_refs(hdf5_paths: List[str]) -> List[EpisodeRef]:
    refs: List[EpisodeRef] = []
    for path in hdf5_paths:
        for name in load_episode_names(path):
            refs.append((path, name))
    return refs


def split_episode_refs(
    episode_refs: List[EpisodeRef],
    train_split: float,
    seed: int,
) -> tuple[List[EpisodeRef], List[EpisodeRef]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1).")
    refs = episode_refs.copy()
    random.Random(seed).shuffle(refs)
    n_train = int(len(refs) * train_split)
    train_refs = refs[:n_train]
    val_refs = refs[n_train:]
    if not train_refs or not val_refs:
        raise RuntimeError("Episode split produced an empty train or validation set.")
    return train_refs, val_refs
