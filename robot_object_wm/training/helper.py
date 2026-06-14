from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from robot_object_wm.config import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT


def add_wandb_args(parser: argparse.ArgumentParser, defaults: Mapping[str, Any] | None = None) -> None:
    defaults = defaults or {}
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--wandb-project-name",
        type=str,
        default=defaults.get("wandb_project_name", DEFAULT_WANDB_PROJECT),
    )
    group.add_argument("--wandb-entity", type=str, default=defaults.get("wandb_entity", DEFAULT_WANDB_ENTITY))
    group.add_argument(
        "--wandb-name",
        type=str,
        default=defaults.get("wandb_name", None),
        help="Weights & Biases run name. Defaults to '<model>-<run_name>'.",
    )
    group.add_argument(
        "--wandb-mode",
        type=str,
        default=defaults.get("wandb_mode", "online"),
        choices=("online", "offline", "disabled"),
        help="Weights & Biases logging mode.",
    )
    group.add_argument(
        "--wandb-artifacts",
        dest="wandb_artifacts",
        action="store_true",
        default=defaults.get("wandb_artifacts", True),
        help="Log improved best.pt checkpoints as W&B artifacts.",
    )
    group.add_argument(
        "--no-wandb-artifacts",
        dest="wandb_artifacts",
        action="store_false",
        help="Disable logging improved best.pt checkpoints as W&B artifacts.",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_dict_to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}
