"""One training run: fixed token budget, fixed data order, measured loss / speed / memory."""
import json
import math
import time
from dataclasses import asdict, dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .data import Stream
from .model import GPT, GPTConfig


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def allocated(dev):
    if dev == "mps":
        return torch.mps.current_allocated_memory()
    if dev == "cuda":
        return torch.cuda.memory_allocated()
    return 0


class Peak(TorchDispatchMode):
    """Peak allocated bytes over a region, sampled after every aten op.

    MPS has no max_memory_allocated, so the peak has to be sampled. A dispatch mode
    sees every op, including the autograd backward thread's (the mode is part of
    the thread-local state autograd carries over), so it catches transients such as
    the logits, their log-softmax and their gradient. Sampling only at module
    boundaries misses those and under-reports by up to half at large batch.
    MPS allocates on the host when an op is enqueued, so each sample is exact for
    that point. On CUDA the allocator's own peak is used as well.

    It costs a Python call per op, so training runs use it for one step only."""

    def __init__(self, dev):
        super().__init__()
        self.dev, self.peak = dev, 0

    def __enter__(self):
        if self.dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.peak = allocated(self.dev)
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, allocated(self.dev))
        return out

    def value(self):
        return max(self.peak, torch.cuda.max_memory_allocated()) if self.dev == "cuda" else self.peak


@dataclass
class RunConfig:
    name: str = "standard"
    rule: str = "standard"
    reversible: bool = False
    h: float = 0.5
    blend: float = 1.0
    batch: int = 64
    seq: int = 256
    d: int = 384
    n_layer: int = 10
    n_head: int = 6
    tokens: int = 50_000_000
    lr: float = 1.5e-3
    lr_batch: int = 64            # lr is set for this batch; other batches scale it by sqrt(batch / lr_batch)
    lr_cap: float = 6e-3          # ...but never above this
    warmup_tokens: int = 1_000_000
    warmup_min_steps: int = 10
    min_lr_frac: float = 0.1
    wd: float = 0.1
    clip: float = 1.0
    amp: bool = False             # bf16 autocast; off by default because fp32 is faster on MPS
    loss_chunk: int = 0           # >0: compute the LM head + loss in chunks of this many tokens, recomputed in backward
    eval_every_tokens: int = 2_500_000
    eval_batches: int = 20
    seed: int = 0
    log_every: int = 50
    notes: dict = field(default_factory=dict)

    def gpt(self):
        return GPTConfig(seq=self.seq, d=self.d, n_head=self.n_head, n_layer=self.n_layer,
                         rule=self.rule, h=self.h, blend=self.blend, reversible=self.reversible,
                         loss_chunk=self.loss_chunk)


def build(cfg, dev):
    torch.manual_seed(cfg.seed)
    model = GPT(cfg.gpt()).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    other = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd},
                             {"params": other, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    return model, opt


def peak_lr(cfg):
    return min(cfg.lr_cap, cfg.lr * math.sqrt(cfg.batch / cfg.lr_batch))


def lr_at(cfg, seen):
    peak = peak_lr(cfg)
    warm = max(cfg.warmup_tokens, cfg.warmup_min_steps * cfg.batch * cfg.seq)
    if seen < warm:
        return peak * (seen + cfg.batch * cfg.seq) / warm
    frac = (seen - warm) / max(1, cfg.tokens - warm)
    return peak * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


@torch.no_grad()
def evaluate(model, cfg, dev, n=None):
    model.eval()
    s = Stream(16, cfg.seq, "valid")
    tot = 0.0
    n = n or cfg.eval_batches
    for _ in range(n):
        x, y = (t.to(dev) for t in s.next())
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.amp):
            tot += model(x, y).item()
    model.train()
    return tot / n


def train_step(model, opt, x, y, cfg, dev):
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.amp):
        loss = model(x, y)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss, gnorm


def measure_peak(cfg, steps=2, dev=None):
    """Peak allocated bytes of one full training step (weights, gradients, Adam state,
    activations, loss), after a warm-up step has created the Adam state."""
    dev = dev or device()
    model, opt = build(cfg, dev)
    s = Stream(cfg.batch, cfg.seq)
    try:
        for i in range(steps):
            x, y = (t.to(dev) for t in s.next())
            if i < steps - 1:
                train_step(model, opt, x, y, cfg, dev)
                continue
            sync(dev)
            with Peak(dev) as mem:
                train_step(model, opt, x, y, cfg, dev)
                sync(dev)
        return mem.value()
    finally:
        del model, opt
        if dev == "mps":
            torch.mps.empty_cache()


def run(cfg: RunConfig, out=None, dev=None, quiet=False):
    dev = dev or device()
    model, opt = build(cfg, dev)
    peak = 0
    stream = Stream(cfg.batch, cfg.seq)
    per_step = cfg.batch * cfg.seq
    steps = cfg.tokens // per_step
    log = dict(config=asdict(cfg), device=dev, n_params=model.n_params(), steps=steps,
               tokens_per_step=per_step, train=[], evals=[])
    seen, t_train, next_eval = 0, 0.0, cfg.eval_every_tokens
    win_loss, win_n, diverged = 0.0, 0, False
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, seen)
        x, y = (t.to(dev) for t in stream.next())
        sync(dev)
        if step == 1:                                      # measured once, untimed: the sampler is slow
            with Peak(dev) as mem:
                loss, gnorm = train_step(model, opt, x, y, cfg, dev)
                sync(dev)
            peak = mem.value()
        else:
            t0 = time.perf_counter()
            loss, gnorm = train_step(model, opt, x, y, cfg, dev)
            sync(dev)
            t_train += time.perf_counter() - t0
        seen += per_step
        timed = seen - (per_step if step >= 1 else 0)
        win_loss += loss.item()
        win_n += 1
        if not math.isfinite(loss.item()):
            diverged = True
        if (step + 1) % cfg.log_every == 0 or step == steps - 1 or diverged:
            row = dict(step=step + 1, tokens=seen, loss=win_loss / win_n, gnorm=gnorm.item(),
                       lr=opt.param_groups[0]["lr"], tok_per_s=timed / t_train, peak_bytes=peak)
            log["train"].append(row)
            win_loss, win_n = 0.0, 0
            if not quiet:
                print(f"[{cfg.name}] step {row['step']}/{steps} tok {seen/1e6:.1f}M loss {row['loss']:.4f} "
                      f"gnorm {row['gnorm']:.2f} {row['tok_per_s']:.0f} tok/s peak {row['peak_bytes']/2**30:.2f} GiB",
                      flush=True)
        if seen >= next_eval or step == steps - 1 or diverged:
            log["evals"].append(dict(tokens=seen, val_loss=evaluate(model, cfg, dev)))
            next_eval += cfg.eval_every_tokens
        if diverged:
            break
    log.update(diverged=diverged, tokens_seen=seen, train_seconds=t_train,
               tok_per_s=timed / t_train, peak_bytes=peak,
               final_train_loss=log["train"][-1]["loss"],
               final_val_loss=evaluate(model, cfg, dev, n=100) if not diverged else float("nan"))
    if out:
        with open(out, "w") as f:
            json.dump(log, f, indent=1)
    log["model"] = model
    return log
