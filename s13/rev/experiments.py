"""The whole experiment, in order. Each finished run is saved to results/ and skipped next time.

    python -m rev.experiments            # everything
    python -m rev.experiments sweep      # one stage: sweep | main | maxbatch | lrcheck
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from .train import RunConfig, measure_peak, run

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GiB = 2 ** 30
BUDGET = 16 * GiB                 # the memory card we pretend to have: a Colab T4's 16 GB
FIXED_BATCH = 64
SWEEP_TOKENS = 5_000_000

SWEEP = {                         # name: (rule, h, blend)   short runs with stored activations
    "sweep_standard": ("standard", 0.5, 1.0),
    "sweep_midpoint_h0.125": ("midpoint", 0.125, 1.0),
    "sweep_midpoint_h0.25": ("midpoint", 0.25, 1.0),
    "sweep_midpoint_h0.5": ("midpoint", 0.5, 1.0),
    "sweep_midpoint_h1.0": ("midpoint", 1.0, 1.0),
    "sweep_midpoint_h0.25_blend0.5": ("midpoint", 0.25, 0.5),
    "sweep_euler_h0.5": ("euler", 0.5, 1.0),
    "sweep_euler_h1.0": ("euler", 1.0, 1.0),
}


def _path(name):
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"{name}.json"


def load(name):
    p = _path(name)
    return json.loads(p.read_text()) if p.exists() else None


def _run(cfg, save_weights=False):
    if load(cfg.name):
        print(f"[{cfg.name}] cached")
        return load(cfg.name)
    log = run(cfg, out=_path(cfg.name))
    model = log.pop("model")
    if save_weights:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, RESULTS / f"{cfg.name}.pt")
    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    return log


def sweep():
    for name, (rule, h, blend) in SWEEP.items():
        # the stored-activation version of each rule: same maths, faster, so the sweep is cheap
        _run(RunConfig(name=name, rule=rule, h=h, blend=blend, tokens=SWEEP_TOKENS,
                       eval_every_tokens=1_000_000))


def best(rule):
    """The (h, blend) with the lowest sweep validation loss for this rule."""
    rows = [(load(n), h, b) for n, (r, h, b) in SWEEP.items() if r == rule]
    rows = [(log["final_val_loss"], h, b) for log, h, b in rows if log and not log["diverged"]]
    return min(rows)[1:]


def main_runs():
    out = {}
    out["standard"] = _run(RunConfig(name="main_standard", rule="standard", batch=FIXED_BATCH), True)
    for rule in ("midpoint", "euler"):
        h, blend = best(rule)
        out[rule] = _run(RunConfig(name=f"main_{rule}_rev", rule=rule, reversible=True,
                                   h=h, blend=blend, batch=FIXED_BATCH), True)
    return out


def winner():
    """The reversible rule with the lower final validation loss at the fixed batch."""
    logs = {r: load(f"main_{r}_rev") for r in ("midpoint", "euler")}
    return min(logs, key=lambda r: logs[r]["final_val_loss"])


def max_batch(cfg, budget=BUDGET, align=64):
    """Largest batch (a multiple of `align`) whose measured step peak fits `budget`.

    Peak memory is linear in batch, so fit it from two measurements, jump to the
    prediction, then measure and step down until it fits."""
    probes = {}

    def peak(b):
        if b not in probes:
            probes[b] = measure_peak(replace(cfg, batch=b))
            print(f"  batch {b}: {probes[b] / GiB:.2f} GiB", flush=True)
        return probes[b]

    b0, b1 = align, 4 * align
    slope = (peak(b1) - peak(b0)) / (b1 - b0)
    b = int((budget - peak(b0)) / slope + b0) // align * align
    while peak(b) > budget:
        b -= align
    while peak(b + align) <= budget:
        b += align
    return dict(batch=b, peak_bytes=probes[b], probes={str(k): v for k, v in sorted(probes.items())},
                slope_bytes_per_sample=slope, budget_bytes=budget)


def maxbatch():
    path = _path("maxbatch_search")
    if path.exists():
        search = json.loads(path.read_text())
    else:
        rule = winner()
        ref = load(f"main_{rule}_rev")["config"]
        rev = RunConfig(rule=rule, reversible=True, h=ref["h"], blend=ref["blend"])
        search = dict(rule=rule, h=ref["h"], blend=ref["blend"])
        print("max batch, reversible", rule)
        search["reversible"] = max_batch(rev)
        print("max batch, standard")
        search["standard"] = max_batch(RunConfig(rule="standard"))
        print("max batch, same rule with activations stored")
        search["stored_same_rule"] = max_batch(replace(rev, reversible=False))
        path.write_text(json.dumps(search, indent=1))
    b = search["reversible"]["batch"]
    return _run(RunConfig(name=f"maxbatch_{search['rule']}_rev", rule=search["rule"], reversible=True,
                          h=search["h"], blend=search["blend"], batch=b,
                          eval_every_tokens=5_000_000, log_every=5), True)


LR_CHECK_BATCH = 256
LR_CHECK = (1.5e-3, 3e-3, 6e-3)


def lrcheck():
    """Is the max-batch run's loss the batch or the learning rate? Same rule, batch 256 (fits
    without swap), 60 steps, three peak learning rates, cosine over those 60 steps."""
    rule = winner()
    ref = load(f"main_{rule}_rev")["config"]
    for lr in LR_CHECK:
        _run(RunConfig(name=f"lrcheck_{rule}_b{LR_CHECK_BATCH}_lr{lr:g}", rule=rule, reversible=True,
                       h=ref["h"], blend=ref["blend"], batch=LR_CHECK_BATCH, lr=lr, lr_batch=LR_CHECK_BATCH,
                       lr_cap=1.0, tokens=60 * LR_CHECK_BATCH * 256, eval_every_tokens=2_000_000, log_every=5))


if __name__ == "__main__":
    stages = sys.argv[1:] or ["sweep", "main", "maxbatch", "lrcheck"]
    for s in stages:
        {"sweep": sweep, "main": main_runs, "maxbatch": maxbatch, "lrcheck": lrcheck}[s]()
