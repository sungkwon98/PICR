from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

LOGVAR_MIN = -10.0
LOGVAR_MAX = 5.0
LATENT_MAX_ABS = 10.0


def bound_latent(value: torch.Tensor, max_abs: float = LATENT_MAX_ABS) -> torch.Tensor:
    return max_abs * torch.tanh(value / max_abs)


@dataclass
class ContextOutput:
    mu: torch.Tensor | None
    logvar: torch.Tensor | None
    z: torch.Tensor | None

    def as_aux(self) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        if self.mu is not None:
            values["context_mu"] = self.mu
        if self.logvar is not None:
            values["context_logvar"] = self.logvar
        if self.z is not None:
            values["context_z"] = self.z
        return values


class ContextEncoder(nn.Module):
    """Encode a flattened history window into a latent context z."""

    def __init__(self, seq_input_dim: int, hidden_dim: int, latent_dim: int, depth: int = 2) -> None:
        super().__init__()
        if seq_input_dim <= 0:
            raise ValueError("seq_input_dim must be > 0.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")
        layers: list[nn.Module] = []
        last = seq_input_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, latent_dim)
        self.logvar_head = nn.Linear(last, latent_dim)

    def forward(self, flat_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(flat_history)
        mu = bound_latent(self.mu_head(h))
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    std = torch.exp(0.5 * logvar)
    return bound_latent(mu + torch.randn_like(std) * std)


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    mu = bound_latent(mu)
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
