from __future__ import annotations

import pytest
import torch

import config
import mixture
from opus import ProtectedFloorTracker


def make_stage(name="mid", start=1, end=10, weights=None, eligible=None):
    weights = weights or {"prose": 0.4, "code": 0.35, "agent": 0.25}
    eligible = eligible if eligible is not None else {"prose", "code", "agent"}
    return mixture.Stage(name=name, start=start, end=end, weights=weights, eligible=eligible)


def _patch_gradients(opus_gate, golden_vec, candidate_vec):
    def fake_probe(batch):
        return golden_vec if batch is opus_gate.golden_batch else candidate_vec

    opus_gate._probe_gradient = fake_probe


def test_defer_for_stage_ineligible_lane(opus_gate):
    stage = make_stage(eligible={"prose", "code"})
    decision = opus_gate.decide({}, "agent", stage)
    assert decision.decision == "DEFER"
    assert "stage_mismatch" in decision.reason


def test_accept_when_gradient_aligned(opus_gate):
    stage = make_stage(weights={"prose": 1.0}, eligible={"prose"})
    _patch_gradients(opus_gate, torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    decision = opus_gate.decide({"x": 1}, "prose", stage)
    assert decision.decision == "ACCEPT"
    assert decision.score == pytest.approx(1.0)


def test_reject_when_gradient_opposed_and_not_protected(opus_gate):
    stage = make_stage(weights={"prose": 1.0}, eligible={"prose"})
    _patch_gradients(opus_gate, torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    decision = opus_gate.decide({"x": 1}, "prose", stage)
    assert decision.decision == "REJECT"
    assert decision.score == pytest.approx(-1.0)


def test_protected_floor_override_forces_accept_below_threshold(opus_gate):
    stage = make_stage(name="mid", start=1, end=2, weights={"code": 1.0}, eligible={"code"})
    _patch_gradients(opus_gate, torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    decision = opus_gate.decide({"x": 1}, config.PROTECTED_LANE, stage)
    assert decision.decision == "PROTECTED_FLOOR_OVERRIDE"
    assert opus_gate.tracker.accepted_count("mid", config.PROTECTED_LANE) == 1


def test_protected_floor_stops_overriding_once_met(opus_gate):
    stage = make_stage(name="mid", start=1, end=2, weights={"code": 1.0}, eligible={"code"})
    _patch_gradients(opus_gate, torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    first = opus_gate.decide({"x": 1}, config.PROTECTED_LANE, stage)
    second = opus_gate.decide({"x": 1}, config.PROTECTED_LANE, stage)
    assert first.decision == "PROTECTED_FLOOR_OVERRIDE"
    assert second.decision == "REJECT"


def test_tracker_only_counts_accepts():
    tracker = ProtectedFloorTracker()
    assert tracker.accepted_count("mid", "code") == 0
    tracker.record_accept("mid", "code")
    tracker.record_accept("mid", "code")
    assert tracker.accepted_count("mid", "code") == 2
    assert tracker.accepted_count("mid", "prose") == 0
