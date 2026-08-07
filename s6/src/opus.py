"""OPUS: the gradient-informed data-selection gate (see the transcript in
`s6/ERA V5 Session...md` for the source description this implements).

For each candidate batch: probe two gradients on the *current* model weights
without ever stepping the optimizer on either --

  - the "golden proxy" gradient: a fixed reference batch of data we want the
    model to behave well on. Computed fresh every call because the model's
    weights (and therefore its gradient) change every step.
  - the candidate gradient: the batch actually on offer.

If the candidate's gradient direction doesn't resemble the golden-proxy
direction, training on it wouldn't move the model toward what we want --
reject. A curriculum-stage-ineligible lane is deferred before any gradient
math happens. A protected lane that hasn't hit its floor yet for the current
stage is forced in regardless of score. Every decision (not just accepts) is
logged -- rejections are the valuable half of this ledger.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import config
from mixture import Stage


@dataclass
class OpusDecision:
    decision: str  # ACCEPT | REJECT | DEFER | PROTECTED_FLOOR_OVERRIDE
    score: float
    reason: str


class ProtectedFloorTracker:
    def __init__(self):
        self._accepted: dict[tuple[str, str], int] = {}

    def record_accept(self, stage_name: str, lane: str) -> None:
        key = (stage_name, lane)
        self._accepted[key] = self._accepted.get(key, 0) + 1

    def accepted_count(self, stage_name: str, lane: str) -> int:
        return self._accepted.get((stage_name, lane), 0)


class OpusGate:
    def __init__(self, model: torch.nn.Module, golden_batch: dict, threshold: float = config.OPUS_ACCEPT_THRESHOLD):
        self.model = model
        self.golden_batch = golden_batch
        self.threshold = threshold
        self.tracker = ProtectedFloorTracker()

    def _probe_gradient(self, batch: dict) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        _, loss = self.model(
            batch["input_ids"], position_ids=batch["position_ids"], attn_mask=batch["attn_mask"],
            targets=batch["targets"], loss_mask=batch["loss_mask"],
        )
        loss.backward()
        grads = [p.grad.detach().reshape(-1) for p in self.model.parameters() if p.grad is not None]
        flat = torch.cat(grads).clone()
        self.model.zero_grad(set_to_none=True)
        return flat

    def decide(self, batch: dict, lane: str, stage: Stage) -> OpusDecision:
        if lane not in stage.eligible:
            return OpusDecision("DEFER", 0.0, f"stage_mismatch: {lane} not eligible during {stage.name}")

        ref_grad = self._probe_gradient(self.golden_batch)
        cand_grad = self._probe_gradient(batch)
        cos = F.cosine_similarity(ref_grad.unsqueeze(0), cand_grad.unsqueeze(0)).item()

        floor_needed = False
        floor_batches = already = None
        if lane == config.PROTECTED_LANE:
            stage_len = stage.end - stage.start + 1
            floor_batches = max(1, round(stage_len * stage.floor(lane)))
            already = self.tracker.accepted_count(stage.name, lane)
            floor_needed = already < floor_batches

        if cos < self.threshold and floor_needed:
            self.tracker.record_accept(stage.name, lane)
            return OpusDecision(
                "PROTECTED_FLOOR_OVERRIDE", cos,
                f"below threshold ({cos:.4f} < {self.threshold}) but protected floor not yet met "
                f"({already}/{floor_batches} accepted this stage)",
            )
        if cos >= self.threshold:
            self.tracker.record_accept(stage.name, lane)
            return OpusDecision("ACCEPT", cos, f"cosine similarity {cos:.4f} >= threshold {self.threshold}")
        return OpusDecision("REJECT", cos, f"cosine similarity {cos:.4f} < threshold {self.threshold}")
