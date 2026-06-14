from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Mapping


DEFAULT_WANDB_PROJECT = "world-model"
DEFAULT_WANDB_ENTITY = "tjrcjf410-seoul-national-university"


def default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def to_plain_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        config = asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError(f"Expected dataclass or mapping config, got {type(config).__name__}")

    clean: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[str(key)] = value
        elif isinstance(value, (list, tuple)):
            clean[str(key)] = list(value)
        elif isinstance(value, Mapping):
            clean[str(key)] = to_plain_config(value)
        else:
            clean[str(key)] = str(value)
    return clean


def safe_wandb_artifact_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "world-model"

