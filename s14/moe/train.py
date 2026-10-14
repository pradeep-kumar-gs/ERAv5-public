"""Training on a shared token axis.

Every run lives on the same 0 to 30M-token axis and the same learning-rate schedule:
- warmup over the first 1M tokens;
- constant until 24M;
- linear decay to 10% at 30M (warmup-stable-decay).

A converted model starts at the token where the dense run was converted, and reads exactly the data the dense
run read from there. It gets a fresh AdamW, because the experts and router are new tensors, and a short re-warmup.
"""
import json
import math
import time
from dataclasses import asdict, dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .data import Stream
from .model import GPT

GiB = 2 ** 30


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def allocated(dev):
    return torch.mps.current_allocated_memory() if dev == "mps" else (torch.cuda.memory_allocated() if dev == "cuda" else 0)


class Peak(TorchDispatchMode):
    """Peak allocated bytes, sampled after every aten op (forward and the autograd thread). From Session 13:
    MPS has no max_memory_allocated, and sampling only at module boundaries misses the loss's transients."""

    def __init__(self, dev):
        super().__init__()
        self.dev, self.peak = dev, 0

    def __enter__(self):
        self.peak = allocated(self.dev)
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, allocated(self.dev))
        return out


@dataclass
class RunConfig:
    name: str
    start_token: int = 0
    end_token: int = 30_000_000
    batch: int = 64
    seq: int = 256
    lr: float = 1.5e-3
    warmup_tokens: int = 1_000_000
    rewarm_tokens: int = 500_000       # for runs that start from a checkpoint with a fresh optimizer
    decay_start: int = 24_000_000
    total_tokens: int = 30_000_000
    min_lr_frac: float = 0.1
    wd: float = 0.1
    clip: float = 1.0
    sample_until_token: int = 0        # MoE: sample experts (Gumbel top-k) until this token, then hard top-k
    checkpoints: tuple = ()            # token marks at which to save weights
    eval_every_tokens: int = 1_500_000
    log_every: int = 50
    seed: int = 0
    notes: dict = field(default_factory=dict)


def lr_at(cfg, tok):
    if tok < cfg.decay_start:
        lr = cfg.lr * min(1.0, (tok + cfg.batch * cfg.seq) / cfg.warmup_tokens)
    else:
        frac = (tok - cfg.decay_start) / (cfg.total_tokens - cfg.decay_start)
        lr = cfg.lr * (1 - (1 - cfg.min_lr_frac) * min(1.0, frac))
    if cfg.start_token:                                    # re-warm a fresh optimizer
        lr *= min(1.0, (tok - cfg.start_token + cfg.batch * cfg.seq) / cfg.rewarm_tokens)
    return lr


def optimizer(model, cfg):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd}, {"params": other, "weight_decay": 0.0}],
                             lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, dev, n=20, seq=256):
    model.eval()
    s = Stream(16, seq, "valid")
    tot = sum(model(*(t.to(dev) for t in s.next())).item() for _ in range(n))
    model.train()
    return tot / n


def step(model, opt, x, y, cfg):
    loss = model(x, y)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():                                  # aux-loss-free balancing: nudge the selection bias
        for m in model.moe_layers():
            load = m.counts.float()
            m.bias += m.m.bias_gamma * torch.sign(load.mean() - load)
    return loss, gnorm


class RoutingStats:
    """Per-layer token counts accumulated over a logging window."""

    def __init__(self, model):
        self.layers = model.moe_layers()
        self.reset()

    def reset(self):
        self.acc = [torch.zeros(m.m.n_experts, device=m.bias.device) for m in self.layers]

    def add(self):
        for a, m in zip(self.acc, self.layers):
            a += m.counts

    def summary(self, family=1):
        out = dict(maxvio=[], dead=[], starved=[], dead_by_family=[])
        for a in self.acc:
            load = a.float().cpu()
            mean = load.mean().item()
            out["maxvio"].append((load.max().item() - mean) / mean)
            out["dead"].append(int((load == 0).sum()))
            out["starved"].append(int((load < 0.1 * mean).sum()))           # under 10% of a fair share
            if family > 1:
                fam = (load.view(-1, family) == 0).sum(1)
                out["dead_by_family"].append(fam.tolist())
        return out


def run(model: GPT, cfg: RunConfig, out=None, dev=None, family=1, quiet=False, results_dir=None):
    dev = dev or device()
    torch.manual_seed(cfg.seed)
    model = model.to(dev)
    opt = optimizer(model, cfg)
    stream = Stream(cfg.batch, cfg.seq, start=cfg.start_token)
    per_step = cfg.batch * cfg.seq
    steps = (cfg.end_token - cfg.start_token) // per_step
    is_moe = bool(model.moe_layers())
    stats = RoutingStats(model) if is_moe else None
    log = dict(config=asdict(cfg), device=dev, n_params=model.n_params(), n_active=model.n_active(),
               moe=asdict(model.cfg.moe) if is_moe else None, steps=steps, tokens_per_step=per_step,
               train=[], evals=[dict(tokens=cfg.start_token, val_loss=evaluate(model, dev))])
    tok, t_train, timed, peak = cfg.start_token, 0.0, 0, 0
    next_eval = cfg.start_token + cfg.eval_every_tokens
    marks = sorted(m for m in cfg.checkpoints if m > cfg.start_token)
    win, nwin, diverged = 0.0, 0, False
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, tok)
        sel = "sample" if tok < cfg.sample_until_token else "hard"
        for m in model.moe_layers():
            m.selection = sel
        x, y = (t.to(dev) for t in stream.next())
        sync(dev)
        if i == 1:
            with Peak(dev) as pk:
                loss, gnorm = step(model, opt, x, y, cfg)
                sync(dev)
            peak = pk.peak
        else:
            t0 = time.perf_counter()
            loss, gnorm = step(model, opt, x, y, cfg)
            sync(dev)
            t_train += time.perf_counter() - t0
            timed += per_step
        tok += per_step
        if dev == "mps" and i % 10 == 9:                     # hand cached blocks back; see MoE.forward
            torch.mps.empty_cache()
        if stats:
            stats.add()
        lv = loss.item()
        win += lv; nwin += 1
        diverged = diverged or not math.isfinite(lv)
        if (i + 1) % cfg.log_every == 0 or i == steps - 1 or diverged:
            row = dict(step=i + 1, tokens=tok, loss=win / nwin, gnorm=gnorm.item(), lr=opt.param_groups[0]["lr"],
                       tok_per_s=timed / max(t_train, 1e-9), selection=sel)
            if stats:
                row["routing"] = stats.summary(family)
                stats.reset()
            log["train"].append(row)
            win, nwin = 0.0, 0
            if not quiet:
                r = row.get("routing")
                extra = f" maxvio {max(r['maxvio']):.2f} dead {sum(r['dead'])}" if r else ""
                print(f"[{cfg.name}] tok {tok/1e6:.1f}M loss {row['loss']:.4f} lr {row['lr']:.2e} "
                      f"{row['tok_per_s']:.0f} tok/s{extra}", flush=True)
        if tok >= next_eval or i == steps - 1 or diverged:
            log["evals"].append(dict(tokens=tok, val_loss=evaluate(model, dev)))
            next_eval += cfg.eval_every_tokens
        while marks and tok >= marks[0]:
            assert tok == marks[0], f"checkpoint mark {marks[0]} is not on a step boundary"
            if results_dir:
                torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                           results_dir / f"{cfg.name}_at{marks[0]}.pt")
            marks.pop(0)
        if diverged:
            break
    log.update(diverged=diverged, tokens_end=tok, train_seconds=t_train, tok_per_s=timed / max(t_train, 1e-9),
               peak_bytes=peak, final_train_loss=log["train"][-1]["loss"],
               final_val_loss=evaluate(model, dev, n=100) if not diverged else float("nan"))
    if out:
        with open(out, "w") as f:
            json.dump(log, f, indent=1)
    return log, model
