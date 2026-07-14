from __future__ import annotations

import torch
from torch import nn, einsum, cat
from torch.nn import Module
from einops import rearrange

class RotaryEmbedding3D(Module):
    def __init__(self, dim, omega = 10000):
        super().__init__()
        rope_dim = (dim // 6) * 6
        assert rope_dim > 0, 'dim must be at least 6 for 3d rope'
        
        inv_freq = omega ** (-torch.arange(0, rope_dim, 6).float() / rope_dim)

        self.register_buffer('inv_freq', inv_freq)
    
    @property
    def device(self):
        return self.inv_freq.device

    def forward(self, pos):
        freqs = einsum('... p, f -> ... p f', pos, self.inv_freq)
        freqs = rearrange(freqs, '... p f -> ... (p f)')
        return cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos_emb, t):
    rope_dim = pos_emb.shape[-1]
    t_rope, t_pass = t[..., :rope_dim], t[..., rope_dim:]
    t_rope = t_rope * pos_emb.cos() + rotate_half(t_rope) * pos_emb.sin()
    return cat((t_rope, t_pass), dim = -1)
