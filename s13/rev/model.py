"""A ~20M-parameter GPT with three ways to move the residual stream between layers.

    standard   p[l+1] = p[l] + f_l(p[l])                       (ordinary pre-norm GPT)
    midpoint   p[l+1] = a*p[l-1] + (1-a)*p[l] + 2h*f_l(p[l])   (a=1 is the course's midpoint rule)
    euler      y1 = x1 + h*Attn_l(x2),  y2 = x2 + h*MLP_l(y1)  (two-stream coupling, as in RevNet / Reformer)

f_l is the whole block's update: attention then feed-forward, f(p) = A(p) + M(p + A(p)).
The euler rule uses the two halves of the block as the two coupled updates.

`reversible=True` runs the stack inside one autograd Function that keeps only the
final states. The backward pass rebuilds every layer's input from its output by
running the rule backwards, then recomputes that layer with gradients on.
`reversible=False` runs the same rule through ordinary autograd, which stores every
activation. The two give the same gradients up to reconstruction error, and the
tests check that.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab: int = 8192
    seq: int = 256
    d: int = 384
    n_head: int = 6
    n_layer: int = 10
    rule: str = "standard"          # standard | midpoint | euler
    h: float = 0.5                  # step size; midpoint with h=0.5, a=1 has the same scale as standard
    blend: float = 1.0              # a in the midpoint rule; 1.0 is the pure midpoint rule
    reversible: bool = False
    loss_chunk: int = 0             # >0: LM head + loss in token chunks, logits recomputed in backward


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.n_head = c.n_head

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c.d)
        self.up = nn.Linear(c.d, 4 * c.d, bias=False)
        self.down = nn.Linear(4 * c.d, c.d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(self.ln(x))))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn, self.mlp = Attn(c), MLP(c)

    def forward(self, p):
        """The block's update f(p), not p + f(p)."""
        a = self.attn(p)
        return a + self.mlp(p + a)


# ---------------------------------------------------------------- the rules, stored activations

def stack_standard(blocks, x, c):
    for b in blocks:
        x = x + b(x)
    return x


def stack_midpoint(blocks, x, c):
    prev, cur = x, x                                   # p[-1] = p[0] = embedding
    for b in blocks:
        prev, cur = cur, c.blend * prev + (1 - c.blend) * cur + 2 * c.h * b(cur)
    return cur


def stack_euler(blocks, x, c):
    x1, x2 = x, x
    for b in blocks:
        x1 = x1 + c.h * b.attn(x2)
        x2 = x2 + c.h * b.mlp(x1)
    return 0.5 * (x1 + x2)


# ---------------------------------------------------------------- the rules, reversible

class _Reversible(torch.autograd.Function):
    """Forward keeps only the final pair of states. Backward walks down the stack,
    rebuilding each input from the output and recomputing that layer with grad on.
    Parameter gradients are accumulated straight into .grad, as RevNet code does."""

    @staticmethod
    def forward(ctx, x, model):
        c, blocks = model.cfg, model.blocks
        ctx.model = model
        ctx.autocast = torch.is_autocast_enabled(x.device.type)
        ctx.dtype = torch.get_autocast_dtype(x.device.type)
        with torch.no_grad():
            if c.rule == "midpoint":
                a, b = x, x
                for blk in blocks:
                    a, b = b, c.blend * a + (1 - c.blend) * b + 2 * c.h * blk(b)
                out = b
            else:                                          # euler coupling
                a, b = x, x
                for blk in blocks:
                    a = a + c.h * blk.attn(b)
                    b = b + c.h * blk.mlp(a)
                out = 0.5 * (a + b)
        ctx.save_for_backward(a, b)                        # the two boundary states, nothing else
        return out

    @staticmethod
    def backward(ctx, g_out):
        model = ctx.model
        c, blocks = model.cfg, model.blocks
        a, b = ctx.saved_tensors
        dev = g_out.device.type
        if c.rule == "midpoint":
            # state (a, b) = (p[l], p[l+1]); grads (ga, gb)
            ga, gb = torch.zeros_like(g_out), g_out
            for blk in reversed(blocks):
                with torch.enable_grad(), torch.autocast(dev, dtype=ctx.dtype, enabled=ctx.autocast):
                    a_ = a.detach().requires_grad_(True)
                    f = blk(a_)
                    torch.autograd.backward(f, 2 * c.h * gb)
                prev = (b - (1 - c.blend) * a - 2 * c.h * f.detach()) / c.blend
                ga, gb = c.blend * gb, ga + (1 - c.blend) * gb + a_.grad
                a, b = prev, a
            model.recon = (a, b)                           # rebuilt p[-1], p[0]: both should equal x
            return ga + gb, None
        # euler: state (x1, x2) after the layer; grads (g1, g2)
        g1, g2 = 0.5 * g_out, 0.5 * g_out
        for blk in reversed(blocks):
            with torch.enable_grad(), torch.autocast(dev, dtype=ctx.dtype, enabled=ctx.autocast):
                y1 = a.detach().requires_grad_(True)
                m = blk.mlp(y1)
                torch.autograd.backward(m, c.h * g2)
            g1 = g1 + y1.grad
            x2 = b - c.h * m.detach()
            with torch.enable_grad(), torch.autocast(dev, dtype=ctx.dtype, enabled=ctx.autocast):
                x2_ = x2.detach().requires_grad_(True)
                at = blk.attn(x2_)
                torch.autograd.backward(at, c.h * g1)
            g2 = g2 + x2_.grad
            x1 = a - c.h * at.detach()
            a, b = x1, x2
        model.recon = (a, b)
        return g1 + g2, None


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.seq, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight                 # tied
        self.recon = None
        self.apply(self._init)
        for n, p in self.named_parameters():               # GPT-2 scaling of the residual writes
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, idx):
        T = idx.shape[1]
        return self.tok(idx) + self.pos(torch.arange(T, device=idx.device))

    def trunk(self, x):
        c = self.cfg
        if c.reversible and c.rule != "standard":
            return _Reversible.apply(x, self)
        return {"standard": stack_standard, "midpoint": stack_midpoint, "euler": stack_euler}[c.rule](self.blocks, x, c)

    def forward(self, idx, targets=None):
        x = self.trunk(self.embed(idx))
        n = self.cfg.loss_chunk
        if targets is not None and n:
            hs, ys = self.ln_f(x).view(-1, x.size(-1)).split(n), targets.view(-1).split(n)
            tot = sum(checkpoint(_chunk_loss, h_, y_, self.head.weight, use_reentrant=False)
                      for h_, y_ in zip(hs, ys))
            return tot / targets.numel()
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


def _chunk_loss(h, y, w):
    return F.cross_entropy((h @ w.t()).float(), y, reduction="sum")
