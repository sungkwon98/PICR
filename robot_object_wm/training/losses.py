from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass
class MetricAverager:
    """Accumulate scalar metrics and return per-batch means."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: Mapping[str, float | int | torch.Tensor]) -> None:
        for key, value in metrics.items():
            if torch.is_tensor(value):
                scalar = float(value.detach().item())
            else:
                scalar = float(value)
            self.totals[key] = self.totals.get(key, 0.0) + scalar
            self.counts[key] = self.counts.get(key, 0) + 1

    def step(self) -> None:
        self.count += 1

    def means(self) -> dict[str, float]:
        return {key: value / max(1, self.counts.get(key, self.count)) for key, value in self.totals.items()}


def prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}
