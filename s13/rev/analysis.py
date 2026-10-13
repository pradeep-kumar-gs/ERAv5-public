"""Measurements made around the training runs: reconstruction drift, gradient
agreement, memory against depth and sequence length, and per-phase timing."""
import time
from dataclasses import replace

import torch

from .data import Stream
from .model import GPT, GPTConfig
from .train import RunConfig, build, measure_peak, sync, train_step

GiB = 2 ** 30


def load_model(path, cfg: GPTConfig, dev="cpu"):
    m = GPT(cfg)
    if path is not None:
        m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(dev)


def _batch(seq, batch=8, dev="cpu"):
    x, y = Stream(batch, seq, "valid").next()
    return x.to(dev), y.to(dev)


@torch.no_grad()
def reconstruction(model, idx, amp=False):
    """Run the rule forward keeping every state, then backwards from the top two
    states, as the reversible backward pass does. Returns the relative error of
    every rebuilt state, bottom (index 0) to top."""
    c, blocks, dev = model.cfg, model.blocks, idx.device.type
    ctx = torch.autocast(dev, dtype=torch.bfloat16, enabled=amp)
    with ctx:
        x = model.embed(idx)
        if c.rule == "midpoint":
            states = [x, x]
            for blk in blocks:
                states.append(c.blend * states[-2] + (1 - c.blend) * states[-1] + 2 * c.h * blk(states[-1]))
            rec = [None] * len(states)
            rec[-1], rec[-2] = states[-1], states[-2]
            for i in range(len(blocks) - 1, -1, -1):          # rebuild states[i] from states[i+1], states[i+2]
                rec[i] = (rec[i + 2] - (1 - c.blend) * rec[i + 1] - 2 * c.h * blocks[i](rec[i + 1])) / c.blend
            pairs = list(zip(states[1:], rec[1:]))            # p[0] .. p[L]
        elif c.rule == "euler":
            states = [(x, x)]
            for blk in blocks:
                x1, x2 = states[-1]
                x1 = x1 + c.h * blk.attn(x2)
                x2 = x2 + c.h * blk.mlp(x1)
                states.append((x1, x2))
            rec = [None] * len(states)
            rec[-1] = states[-1]
            for i in range(len(blocks) - 1, -1, -1):
                y1, y2 = rec[i + 1]
                x2 = y2 - c.h * blocks[i].mlp(y1)
                x1 = y1 - c.h * blocks[i].attn(x2)
                rec[i] = (x1, x2)
            pairs = [(torch.cat(s), torch.cat(r)) for s, r in zip(states, rec)]
        else:
            raise ValueError("standard has no inverse")
    return [((r.cpu().double() - s.cpu().double()).norm() / s.cpu().double().norm()).item() for s, r in pairs]


def gradient_agreement(model, idx, y, amp=False):
    """Reversible gradients against stored-activation gradients for the same weights."""
    dev = idx.device.type
    out = {}
    for rev in (False, True):
        model.zero_grad(set_to_none=True)
        model.cfg.reversible = rev
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            loss = model(idx, y)
        loss.backward()
        out[rev] = torch.cat([p.grad.flatten().cpu().double() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    a, b = out[False], out[True]
    return dict(rel_err=((a - b).norm() / a.norm()).item(),
                cos=torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def memory_sweep(base: RunConfig, axis, values, rules=(("standard", False), ("midpoint", True), ("euler", True))):
    """Measured step peak (GiB) for each rule as one dimension changes."""
    rows = []
    for v in values:
        row = {axis: v}
        for rule, rev in rules:
            cfg = replace(base, rule=rule, reversible=rev, **{axis: v})
            row[f"{rule}{'_rev' if rev else ''}"] = measure_peak(cfg) / GiB
        rows.append(row)
        print(row, flush=True)
    return rows


def phase_times(cfg: RunConfig, steps=6, dev="mps"):
    """Mean seconds for forward (incl. loss), backward and optimizer step."""
    model, opt = build(cfg, dev)
    s = Stream(cfg.batch, cfg.seq)
    acc = dict(forward=0.0, backward=0.0, optimizer=0.0)
    for i in range(steps + 2):
        x, y = (t.to(dev) for t in s.next())
        sync(dev); t0 = time.perf_counter()
        loss = model(x, y)
        sync(dev); t1 = time.perf_counter()
        loss.backward()
        sync(dev); t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        sync(dev); t3 = time.perf_counter()
        if i >= 2:
            acc["forward"] += t1 - t0; acc["backward"] += t2 - t1; acc["optimizer"] += t3 - t2
    del model, opt
    if dev == "mps":
        torch.mps.empty_cache()
    out = {k: v / steps for k, v in acc.items()}
    out["step"] = sum(out.values())
    out["tok_per_s"] = cfg.batch * cfg.seq / out["step"]
    return out


def throughput(cfg: RunConfig, steps=4, dev="mps"):
    model, opt = build(cfg, dev)
    s = Stream(cfg.batch, cfg.seq)
    for i in range(steps + 1):
        if i == 1:
            sync(dev); t0 = time.perf_counter()
        x, y = (t.to(dev) for t in s.next())
        train_step(model, opt, x, y, cfg, dev)
    sync(dev)
    dt = (time.perf_counter() - t0) / steps
    del model, opt
    if dev == "mps":
        torch.mps.empty_cache()
    return cfg.batch * cfg.seq / dt
