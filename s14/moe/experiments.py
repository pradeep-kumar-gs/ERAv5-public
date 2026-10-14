"""The whole experiment, in order. Each finished run is saved to results/ and skipped next time.

    python -m moe.experiments                  # everything
    python -m moe.experiments dense convert    # stages: dense | convert | grow | scratch

1. dense      the dense GPT, 0 -> 30M tokens. Weights saved at 15M (the conversion point) and 24M.
2. convert    at 15M, convert the dense checkpoint by copy, drop (r = 0.5) and partition, then train each
              15M -> 30M. The dense run's second half is the "kept dense" baseline on the same tokens.
3. grow       take the best converted MoE at 24M and grow it 8 -> 32 experts by cloning (top-k stays 2).
              Train 24M -> 30M three ways: hard top-k, sampled top-k for the first 1.5M tokens, and hard
              top-k with half of each clone's neurons redrawn. The ungrown MoE's own 24M -> 30M is the baseline.
4. scratch    the copy-shaped MoE (8 experts x 1,536, top-2) from random init, 0 -> 30M.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from .convert import dense_to_moe, grow
from .model import GPT, GPTConfig, MoEConfig
from .train import RunConfig, run

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STEP_TOKENS = 64 * 256
CONVERT_AT = 15_000_000 // STEP_TOKENS * STEP_TOKENS      # 14,991,360: on a step boundary, so a converted run
GROW_AT = 24_000_000 // STEP_TOKENS * STEP_TOKENS         # 23,986,176  reads exactly the dense run's next tokens
METHODS = ("copy", "drop", "partition")
GROW_RUNS = {                      # name: (sample_for_tokens, redraw)
    "grow_hard": (0, 0.0),
    "grow_sample": (1_500_000, 0.0),
    "grow_redraw": (0, 0.5),
}


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _run(model, cfg, family=1):
    RESULTS.mkdir(exist_ok=True)
    if load(cfg.name):
        print(f"[{cfg.name}] cached")
        return load(cfg.name)
    log, model = run(model, cfg, out=RESULTS / f"{cfg.name}.json", family=family, results_dir=RESULTS)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, RESULTS / f"{cfg.name}.pt")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return log


def _weights(model, name):
    model.load_state_dict(torch.load(RESULTS / name, map_location="cpu"))
    return model


def moe_config(method):
    base = GPTConfig()
    if method == "partition":
        return replace(base, moe=MoEConfig(expert_width=base.ffn // 2, shared_width=base.ffn))
    return replace(base, moe=MoEConfig())


def dense():
    torch.manual_seed(0)
    return _run(GPT(GPTConfig()), RunConfig(name="dense", checkpoints=(CONVERT_AT, GROW_AT)))


def convert():
    for method in METHODS:
        src = _weights(GPT(GPTConfig()), f"dense_at{CONVERT_AT}.pt")
        model = dense_to_moe(src, method)
        _run(model, RunConfig(name=f"moe_{method}", start_token=CONVERT_AT, checkpoints=(GROW_AT,)))


def best_at_grow():
    """The converted MoE with the lowest validation loss at the growth point."""
    def val_at(log):
        return min(log["evals"], key=lambda e: abs(e["tokens"] - GROW_AT))["val_loss"]
    return min(METHODS, key=lambda m: val_at(load(f"moe_{m}")))


def grow_runs():
    method = best_at_grow()
    for name, (sample_for, redraw) in GROW_RUNS.items():
        src = _weights(GPT(moe_config(method)), f"moe_{method}_at{GROW_AT}.pt")
        big = grow(src, factor=4, redraw=redraw)
        _run(big, RunConfig(name=name, start_token=GROW_AT, sample_until_token=GROW_AT + sample_for,
                            log_every=25, eval_every_tokens=1_000_000, notes=dict(grown_from=method)), family=4)


def scratch():
    torch.manual_seed(0)
    return _run(GPT(moe_config("copy")), RunConfig(name="moe_scratch"))


if __name__ == "__main__":
    stages = sys.argv[1:] or ["dense", "convert", "grow", "scratch"]
    for s in stages:
        {"dense": dense, "convert": convert, "grow": grow_runs, "scratch": scratch}[s]()
