"""A small character-level GPT, written as plain functions of explicit weights.

The model is split into units: the embeddings, one unit per transformer block,
and the head. A unit is the granularity at which ZeRO-3 gathers and frees
weights, and at which gradients are reduced. Matrix multiplies run in the weight
dtype (bf16 by default). LayerNorm, softmax and the loss run in fp32, the cheap
places where precision matters (course Section 11).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16


@dataclass
class ModelConfig:
    vocab: int = 65
    d: int = 64
    n_layer: int = 4
    n_head: int = 4
    seq: int = 64
    dtype: torch.dtype = BF16


def unit_specs(cfg):
    """[(unit name, [(param name, shape), ...]), ...] in forward order."""
    d, V, T = cfg.d, cfg.vocab, cfg.seq
    units = [("embed", [("tok", (V, d)), ("pos", (T, d))])]
    for i in range(cfg.n_layer):
        units.append((f"block{i}", [
            ("ln1_w", (d,)), ("ln1_b", (d,)),
            ("qkv_w", (3 * d, d)), ("qkv_b", (3 * d,)),
            ("proj_w", (d, d)), ("proj_b", (d,)),
            ("ln2_w", (d,)), ("ln2_b", (d,)),
            ("fc_w", (4 * d, d)), ("fc_b", (4 * d,)),
            ("out_w", (d, 4 * d)), ("out_b", (d,)),
        ]))
    units.append(("head", [("lnf_w", (d,)), ("lnf_b", (d,)), ("head_w", (V, d))]))
    return units


def init_unit(params, gen, n_layer):
    """fp32 initial values for one unit, flattened in spec order."""
    out = []
    for name, shape in params:
        if name.endswith("_b"):
            t = torch.zeros(shape)
        elif name.startswith("ln"):
            t = torch.ones(shape)
        else:
            std = 0.02 / math.sqrt(2 * n_layer) if name in ("proj_w", "out_w") else 0.02
            t = torch.randn(shape, generator=gen) * std
        out.append(t.flatten())
    return torch.cat(out)


def views(flat, params):
    """Name -> shaped view into the unit's flat bf16 buffer. No copies."""
    out, off = {}, 0
    for name, shape in params:
        n = math.prod(shape)
        out[name] = flat[off:off + n].view(shape)
        off += n
    return out


def _ln(x, w, b):
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float()).to(x.dtype)


def unit_forward(name, p, x, cfg, targets=None):
    if name == "embed":
        T = x.shape[1]
        return p["tok"][x] + p["pos"][:T]
    if name == "head":
        logits = _ln(x, p["lnf_w"], p["lnf_b"]) @ p["head_w"].t()
        return F.cross_entropy(logits.float().view(-1, cfg.vocab), targets.view(-1))
    B, T, d = x.shape
    h, hd = cfg.n_head, d // cfg.n_head
    a = _ln(x, p["ln1_w"], p["ln1_b"])
    q, k, v = F.linear(a, p["qkv_w"], p["qkv_b"]).split(d, dim=-1)
    q, k, v = (t.view(B, T, h, hd).transpose(1, 2) for t in (q, k, v))
    scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(hd)
    mask = torch.ones(T, T, dtype=torch.bool).triu(1)
    att = scores.masked_fill(mask, float("-inf")).softmax(-1).to(v.dtype)
    y = (att @ v).transpose(1, 2).reshape(B, T, d)
    x = x + F.linear(y, p["proj_w"], p["proj_b"])
    m = _ln(x, p["ln2_w"], p["ln2_b"])
    return x + F.linear(F.gelu(F.linear(m, p["fc_w"], p["fc_b"])), p["out_w"], p["out_b"])
