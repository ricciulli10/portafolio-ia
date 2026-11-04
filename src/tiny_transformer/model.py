"""A character-level GPT written from scratch: attention, masking, blocks, sampling.

The only torch layers used are Linear, Embedding, LayerNorm and Dropout, so the attention
arithmetic stays visible instead of hiding in nn.MultiheadAttention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.query_key_value = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.query_key_value(x).chunk(3, dim=-1)
        # (batch, time, channels) -> (batch, heads, time, head_dim)
        q, k, v = (
            t.view(batch, time, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        weights = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # A position may look at itself and at the past, never at the future.
        weights = weights.masked_fill(~self.mask[:, :, :time, :time], float("-inf"))
        weights = self.attn_drop(F.softmax(weights, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, time, channels)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        time = idx.shape[1]
        if time > self.cfg.block_size:
            raise ValueError(f"Sequence of {time} exceeds block_size {self.cfg.block_size}")
        x = self.drop(self.tok(idx) + self.pos(torch.arange(time, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, steps: int, temperature: float = 0.8, top_k: int | None = 20
    ) -> torch.Tensor:
        self.eval()
        for _ in range(steps):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx
