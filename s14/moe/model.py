"""A ~21M-parameter GPT whose feed-forward block is either dense or a mixture of experts.

Dense FFN:  x -> GELU(x W_up^T) W_down^T, inner width 1,536 (4 x d).
MoE FFN:    y = shared(x) + sum over the top-k experts of g_i * E_i(x)

- Each expert E_i has the same shape as the dense FFN, at its own inner width.
- The router is one d x E matrix, run in fp32. Softmax gives a probability per expert.
- The top-k experts are chosen from probability + bias. The bias is used only for choosing, and moves after
  every step towards an even load: the auxiliary-loss-free method.
- The chosen experts' probabilities are rescaled to sum to 1 and become the weights g_i.
- Routing is dropless: every token reaches every expert it picked, however busy that expert is.
- `selection="sample"` draws the k experts from the router's probabilities without replacement (Gumbel top-k)
  instead of taking the k largest. That gives near-identical clones a chance to receive gradient before the
  router settles.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    n_experts: int = 8
    top_k: int = 2
    expert_width: int = 1536
    shared_width: int = 0          # >0: an always-on shared expert of this inner width
    router_std: float = 0.002      # one tenth of the usual 0.02, as Switch Transformer found stable
    bias_gamma: float = 0.001      # aux-loss-free bias update speed


@dataclass
class GPTConfig:
    vocab: int = 8192
    seq: int = 256
    d: int = 384
    n_head: int = 6
    n_layer: int = 10
    ffn: int = 1536
    moe: MoEConfig = None          # None = dense


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.n_head = c.n_head

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.up = nn.Linear(d, width, bias=False)
        self.down = nn.Linear(width, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    def __init__(self, d, m: MoEConfig):
        super().__init__()
        self.m = m
        E, w = m.n_experts, m.expert_width
        self.router = nn.Linear(d, E, bias=False)
        self.w_up = nn.Parameter(torch.empty(E, w, d))
        self.w_down = nn.Parameter(torch.empty(E, d, w))
        self.shared = MLP(d, m.shared_width) if m.shared_width else None
        self.register_buffer("bias", torch.zeros(E))
        self.selection = "hard"                       # or "sample"
        self.counts = None                            # tokens per expert in the last forward
        self.last_choice = None                       # (N, k) expert ids, kept only when asked
        self.keep_choice = False

    def route(self, h):
        """Returns expert ids (N, k) and their weights (N, k)."""
        logits = h.float() @ self.router.weight.float().t()
        probs = logits.softmax(-1)
        k = self.m.top_k
        if self.selection == "sample" and self.training:
            g = -torch.log(-torch.log(torch.rand_like(probs).clamp_min(1e-20)))
            idx = (probs.clamp_min(1e-20).log() + g).topk(k, dim=-1).indices
        else:
            idx = (probs + self.bias).topk(k, dim=-1).indices
        w = probs.gather(1, idx)
        return idx, w / w.sum(-1, keepdim=True)

    def forward(self, x):
        B, T, D = x.shape
        h = x.reshape(-1, D)
        N, k, E = h.shape[0], self.m.top_k, self.m.n_experts
        idx, w = self.route(h)
        flat_e = idx.flatten()
        order = flat_e.argsort()
        tok = torch.arange(N, device=h.device).repeat_interleave(k)[order]
        wts = w.flatten()[order].to(h.dtype)
        counts = torch.bincount(flat_e, minlength=E)
        self.counts = counts.detach()
        if self.keep_choice:
            self.last_choice = idx.detach()
        # Round each expert's slice up to a multiple of `bucket` with weight-0 rows (token 0). On MPS every new
        # tensor size gets its own cached block, so raw per-step counts grew the driver's memory to 19 GiB in
        # 30 steps; with buckets only a few sizes ever occur.
        bucket = max(64, 4096 // E)                  # 512 rows for 8 experts, 128 for 32
        sizes = [-(-n // bucket) * bucket for n in counts.tolist()]
        pad_start = torch.tensor([0] + sizes[:-1], device=h.device).cumsum(0)
        start = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        dest = torch.arange(N * k, device=h.device) + (pad_start - start)[flat_e[order]]
        tok_p = torch.zeros(sum(sizes), dtype=torch.long, device=h.device).index_copy(0, dest, tok)
        wts_p = torch.zeros(sum(sizes), dtype=h.dtype, device=h.device).index_copy(0, dest, wts)
        out = torch.zeros_like(h)
        s0 = 0
        for e, n in enumerate(sizes):               # dropless: each expert takes all its tokens
            if n:
                t = tok_p[s0:s0 + n]
                y = F.gelu(h[t] @ self.w_up[e].t()) @ self.w_down[e].t()
                out.index_add_(0, t, y * wts_p[s0:s0 + n, None])
            s0 += n
        if self.shared is not None:
            out = out + self.shared(h)
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.attn = Attn(c)
        self.ffn = MoE(c.d, c.moe) if c.moe else MLP(c.d, c.ffn)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.seq, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight") or n.endswith("w_down"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))
            elif n.endswith("w_up"):
                nn.init.normal_(p, 0.0, 0.02)
            elif n.endswith("router.weight"):
                nn.init.normal_(p, 0.0, c.moe.router_std)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_active(self):
        """Weights one token uses: everything except the experts it didn't pick."""
        total = self.n_params()
        for m in self.moe_layers():
            per_expert = m.w_up[0].numel() + m.w_down[0].numel()
            total -= (m.m.n_experts - m.m.top_k) * per_expert
        return total

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
