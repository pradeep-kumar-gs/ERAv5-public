"""Minimal decoder-only transformer, nanoGPT-style (design reference: Karpathy's
nanoGPT, MIT licensed -- https://github.com/karpathy/nanoGPT; same lineage as
s6/src/model.py, reimplemented here so the token embedding module is injected
rather than hardcoded -- that's the one thing this experiment needs to swap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 8
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Same shape as s6/src/model.py's GPT, except the token embedding is an
    injected module (dense / Kronecker V1 / Kronecker V2) instead of a bare
    nn.Embedding, so the three arms differ only in that one component.
    """

    def __init__(self, config: GPTConfig, token_embedding: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = token_embedding
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ):
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
            if loss_mask is not None:
                mask = loss_mask.float()
                denom = mask.sum().clamp(min=1.0)
                loss = (nll * mask).sum() / denom
            else:
                loss = nll.mean()
        return logits, loss
