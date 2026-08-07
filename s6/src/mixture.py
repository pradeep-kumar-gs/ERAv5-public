"""Curriculum stages, lane weights, and protected floors.

This is the single source of truth both packing.py (which lane to sample)
and opus.py (stage eligibility, protected-floor tracking) read from -- the
mixture schedule is compiled once and never re-derived ad hoc elsewhere.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import config


@dataclass
class Stage:
    name: str
    start: int
    end: int
    weights: dict[str, float]
    eligible: set[str]

    def floor(self, lane: str) -> float:
        """Protected floor for `lane` in this stage: half its stage weight."""
        if lane != config.PROTECTED_LANE:
            return 0.0
        return self.weights.get(lane, 0.0) * config.PROTECTED_FLOOR_FRACTION


STAGES = [Stage(**s) for s in config.STAGES]


def stage_for_step(step: int) -> Stage:
    for stage in STAGES:
        if stage.start <= step <= stage.end:
            return stage
    raise ValueError(f"step {step} outside any curriculum stage")


def compile_schedule() -> list[dict]:
    """One row per step -- this is what gets written to manifests/mixture_schedule.json
    and is what evidence.py cross-checks planned-vs-actual lane shares against."""
    rows = []
    for step in range(1, config.TOTAL_STEPS + 1):
        stage = stage_for_step(step)
        rows.append({
            "step": step, "stage": stage.name, "weights": stage.weights,
            "eligible_lanes": sorted(stage.eligible),
            "protected_floor": {config.PROTECTED_LANE: stage.floor(config.PROTECTED_LANE)},
        })
    return rows


def sample_lane(step: int, rng: random.Random) -> str:
    stage = stage_for_step(step)
    lanes = list(stage.weights.keys())
    weights = [stage.weights[l] for l in lanes]
    return rng.choices(lanes, weights=weights, k=1)[0]
