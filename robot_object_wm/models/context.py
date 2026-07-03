from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

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


def context_kl_loss(context: ContextOutput, weight: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weight <= 0.0 or context.mu is None or context.logvar is None:
        if context.mu is not None:
            zero = context.mu.new_zeros(())
        else:
            zero = torch.zeros(())
        return zero, {}
    kl = kl_divergence_standard_normal(context.mu, context.logvar)
    return weight * kl, {"context_kl": kl.detach()}


def info_nce_soft(
    embedding: torch.Tensor,
    context_value: torch.Tensor,
    *,
    temperature: float = 0.1,
    context_similarity_sigma: float = 1.0,
    nce_negative_weight: float = 1.0,
) -> torch.Tensor:
    if embedding.shape[0] < 2 or context_value.shape[-1] == 0:
        return ValueError("embedding should has [B, Z]")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if context_similarity_sigma <= 0.0:
        raise ValueError("context_similarity_sigma must be > 0.")

    z = F.normalize(embedding, dim=-1)
    logits = (z @ z.transpose(0, 1)) / temperature
    batch_size = logits.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)

    ctx = context_value.to(device=embedding.device, dtype=embedding.dtype)
    ctx = (ctx - ctx.mean(dim=0, keepdim=True)) / (ctx.std(dim=0, keepdim=True) + 1.0e-6)
    ctx_dist2 = torch.sum((ctx.unsqueeze(1) - ctx.unsqueeze(0)).pow(2), dim=-1)
    w_pos = torch.exp(-ctx_dist2 / (2.0 * context_similarity_sigma * context_similarity_sigma + 1.0e-12))
    w_pos = w_pos.masked_fill(eye, 0.0)
    w_neg = (1.0 - w_pos).masked_fill(eye, 0.0)

    # Mask self-similarity before exponentiation.
    logits = logits.masked_fill(eye, float("-inf"))

    exp_logits = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))

    num = torch.sum(w_pos * exp_logits, dim=1)
    den = num + nce_negative_weight * torch.sum(w_neg * exp_logits, dim=1)
    valid = (num > 0.0) & (den > 0.0)
    if not bool(valid.any().detach().cpu().item()):
        return embedding.new_zeros(())
    return -torch.log(num[valid].clamp_min(1.0e-12) / den[valid].clamp_min(1.0e-12)).mean()


def context_info_nce_loss(
    context: ContextOutput,
    context_value: torch.Tensor,
    *,
    weight: float = 0.0,
    temperature: float = 0.1,
    context_similarity_sigma: float = 1.0,
    nce_negative_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weight <= 0.0 or context.z is None:
        if context.z is not None:
            zero = context.z.new_zeros(())
        else:
            zero = context_value.new_zeros(())
        return zero, {}
    nce = info_nce_soft(
        context.z,
        context_value,
        temperature=temperature,
        context_similarity_sigma=context_similarity_sigma,
        nce_negative_weight=nce_negative_weight,
    )
    return weight * nce, {"context_info_nce": nce.detach()}
