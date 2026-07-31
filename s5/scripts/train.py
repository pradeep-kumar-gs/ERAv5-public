#!/usr/bin/env python3
"""Toy-scale proxy training run (D37): trains the same ~18M-param model from the
same initialization on hypotheses A/B/C's token streams (built by
proxy_data_mix.py), evaluating each on the SAME shared validation set and the
SAME shared held-out "complex" set (withheld Bespoke-Stratos Band 3 + withheld
LongAlpaca 16k+ documents).

This is a directional, sub-1B toy run on a single M1 Pro (MPS) -- explicitly
NOT the assignment's literal 1B/3B proxy-experiment bar (D37). It answers one
narrow, falsifiable question per hypothesis pair:
  A vs B: does pushing the Indic overlay to its 25% ceiling change validation
          loss on the fixed reference-mixture validation set?
  A vs C: at fixed mixture ratios, does curriculum-ordering (short/simple
          lanes first, long-context/reasoning last) change held-out "complex"
          generalization loss vs fully shuffled ordering?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[2]
PROXY_DIR = ROOT / "s5" / "data" / "proxy"
OUTPUT = ROOT / "s5" / "site" / "data" / "proxy-results.json"

SEED = 1337
BLOCK_SIZE = 256
BATCH_SIZE = 32
STEPS = 600
EVAL_INTERVAL = 100
EVAL_ITERS = 20
LR = 3e-4


def load_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)


def get_batch(data: np.ndarray, device: str, generator: torch.Generator):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model: GPT, data: np.ndarray, device: str, generator: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, device, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(name: str, train_data: np.ndarray, val_data: np.ndarray, heldout_data: np.ndarray, device: str) -> dict:
    torch.manual_seed(SEED)
    model = GPT(GPTConfig(block_size=BLOCK_SIZE)).to(device)
    n_params = model.num_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED)

    curve = []
    start = time.time()
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, device, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % EVAL_INTERVAL == 0 or step == STEPS:
            val_l = eval_loss(model, val_data, device, generator)
            curve.append({"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(val_l, 4)})
            print(f"[{name}] step {step}/{STEPS} train_loss={loss.item():.4f} val_loss={val_l:.4f}")
    elapsed = time.time() - start

    final_val = eval_loss(model, val_data, device, generator)
    final_heldout = eval_loss(model, heldout_data, device, generator)
    return {
        "hypothesis": name,
        "n_params": n_params,
        "train_tokens": int(len(train_data)),
        "steps": STEPS,
        "elapsed_seconds": round(elapsed, 1),
        "final_train_loss": curve[-1]["train_loss"] if curve else None,
        "final_val_loss_shared": round(final_val, 4),
        "final_heldout_complex_loss": round(final_heldout, 4),
        "loss_curve": curve,
    }


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    val_data = load_bin(PROXY_DIR / "val_shared.bin")
    heldout_data = load_bin(PROXY_DIR / "heldout_complex.bin")

    results = {"device": device, "config": {"block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE, "steps": STEPS, "lr": LR}, "hypotheses": {}}
    for name in ["A", "B", "C"]:
        train_data = load_bin(PROXY_DIR / f"train_{name}.bin")
        results["hypotheses"][name] = train_one(name, train_data, val_data, heldout_data, device)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"final_val_loss_shared": v["final_val_loss_shared"], "final_heldout_complex_loss": v["final_heldout_complex_loss"]} for k, v in results["hypotheses"].items()}, indent=2))


if __name__ == "__main__":
    main()
