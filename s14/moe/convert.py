"""Turning a dense model into a mixture of experts, and a mixture of experts into a bigger one.

Every method copies attention, norms and embeddings unchanged and builds only the feed-forward block anew.

    copy       every expert is the whole dense FFN; only the router is new          (sparse upcycling, 2022)
    drop       every expert is the dense FFN with a fraction r of its neurons
               re-initialised; r = 0.5 was best in the paper                          (drop-upcycling, 2025)
    partition  the dense FFN becomes an always-on shared expert; each routed expert
               takes a random half of its neurons                                     (Lightning LM's dense -> MoE step)
    grow       each expert is cloned `factor` times, router rows tiled with small
               noise, optionally with half of each clone's neurons redrawn            (Lightning LM's 20 -> 460 step)

A "neuron" is one inner unit of the FFN: one row of W_up together with the matching column of W_down.
"""
import copy as _copy
from dataclasses import replace

import torch

from .model import GPT, MLP, MoE, MoEConfig


def _redraw(w_up, w_down, frac, ref_up, ref_down, g):
    """Re-initialise a random `frac` of the neurons, drawing from the statistics of the reference weights,
    as drop-upcycling does."""
    width = w_up.shape[0]
    idx = torch.randperm(width, generator=g)[: int(round(frac * width))]
    w_up[idx] = torch.randn(len(idx), w_up.shape[1], generator=g) * ref_up.std() + ref_up.mean()
    w_down[:, idx] = torch.randn(w_down.shape[0], len(idx), generator=g) * ref_down.std() + ref_down.mean()
    return idx


@torch.no_grad()
def dense_to_moe(dense: GPT, method: str, n_experts=8, top_k=2, r=0.5, seed=0):
    c = dense.cfg
    g = torch.Generator().manual_seed(seed)
    if method in ("copy", "drop"):
        m = MoEConfig(n_experts=n_experts, top_k=top_k, expert_width=c.ffn)
    elif method == "partition":
        m = MoEConfig(n_experts=n_experts, top_k=top_k, expert_width=c.ffn // 2, shared_width=c.ffn)
    else:
        raise ValueError(method)
    moe = GPT(replace(c, moe=m))
    state = {k: v for k, v in dense.state_dict().items() if ".ffn." not in k}
    missing = moe.load_state_dict(state, strict=False).missing_keys
    assert all(".ffn." in k for k in missing), missing
    for blk_d, blk_m in zip(dense.blocks, moe.blocks):
        up, down = blk_d.ffn.up.weight.detach().cpu(), blk_d.ffn.down.weight.detach().cpu()
        f = blk_m.ffn
        for e in range(n_experts):
            if method == "partition":
                idx = torch.randperm(c.ffn, generator=g)[: c.ffn // 2]
                f.w_up[e] = up[idx]
                f.w_down[e] = down[:, idx]
            else:
                f.w_up[e] = up
                f.w_down[e] = down
                if method == "drop":
                    _redraw(f.w_up[e], f.w_down[e], r, up, down, g)
        if method == "partition":
            f.shared.up.weight.copy_(up)
            f.shared.down.weight.copy_(down)
    return moe


@torch.no_grad()
def grow(moe_model: GPT, factor=4, router_noise=1e-3, redraw=0.0, seed=0):
    """E experts -> E * factor. Expert e becomes experts e*factor ... e*factor + factor - 1 (a "clone family").
    Top-k is kept, so the compute per token doesn't change."""
    c = moe_model.cfg
    g = torch.Generator().manual_seed(seed)
    m = replace(c.moe, n_experts=c.moe.n_experts * factor)
    big = GPT(replace(c, moe=m))
    state = {k: v for k, v in moe_model.state_dict().items() if ".ffn." not in k or ".ffn.shared." in k}
    big.load_state_dict(state, strict=False)
    for blk_s, blk_b in zip(moe_model.blocks, big.blocks):
        s, b = blk_s.ffn, blk_b.ffn
        b.w_up.copy_(s.w_up.detach().cpu().repeat_interleave(factor, 0))
        b.w_down.copy_(s.w_down.detach().cpu().repeat_interleave(factor, 0))
        rw = s.router.weight.detach().cpu().repeat_interleave(factor, 0)
        b.router.weight.copy_(rw + router_noise * torch.randn(rw.shape, generator=g))
        b.bias.copy_(s.bias.detach().cpu().repeat_interleave(factor))
        if redraw:
            for e in range(m.n_experts):
                if e % factor:                         # keep one member of each family intact
                    _redraw(b.w_up[e], b.w_down[e], redraw, s.w_up[e // factor].cpu(), s.w_down[e // factor].cpu(), g)
    return big
