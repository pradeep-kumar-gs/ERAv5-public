"""Checkpoint = model + optimizer + full RNG state + ledger offsets at the
moment of save. Saving the RNG state (not just model weights) is what makes
deterministic resume possible: reloading it puts the packing RNG back to
exactly the state it was in right after the checkpointed step's batch was
drawn, so the next `rng.randrange(...)` call reproduces the same draw a
reference run would make.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from manifest import sha256_file


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     step: int, run_id: str, py_rng: random.Random,
                     consumption_offset: int, learning_offset: int, extra: dict | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "run_id": run_id,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "py_rng_state": py_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "consumption_offset": consumption_offset,
        "learning_offset": learning_offset,
        "extra": extra or {},
    }
    torch.save(state, path)
    return {
        "path": str(path), "step": step, "run_id": run_id, "sha256": sha256_file(path),
        "consumption_offset": consumption_offset, "learning_offset": learning_offset,
    }


def load_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     py_rng: random.Random) -> dict:
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    py_rng.setstate(state["py_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    return state
