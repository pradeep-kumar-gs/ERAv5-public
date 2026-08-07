"""Training-loop mechanics shared by every lineage in the demo (the main
run, its crash/resume continuation, and the forked branch): sample a lane,
pack a batch, ask OPUS, train if accepted, and append both ledgers. Kept as
one function so all three lineages provably go through identical logic --
nothing lineage-specific is hardcoded here except which model/optimizer/RNG/
ledgers/checkpoint directory get passed in.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch

import checkpoint as checkpoint_mod
import config
import mixture
import packing
from ledger import JsonlLedger
from logbook import Logbook
from manifest import ShardManifest
from opus import OpusGate


def get_device() -> str:
    # Always CPU: MPS's matmul/attention reduction order is not bit-stable
    # across process runs, which breaks the cross-run determinism this demo
    # exists to prove (OPUS's gradient-cosine score, and everything trained
    # on top of it, diverged between two otherwise-identical runs on MPS).
    # The model is tiny, so CPU costs nothing that matters here.
    return "cpu"


def batch_to_tensors(batch: packing.Batch, device: str) -> dict:
    input_ids = torch.tensor(np.stack([s.input_ids for s in batch.sequences]), dtype=torch.long, device=device)
    targets = torch.tensor(np.stack([s.target_ids for s in batch.sequences]), dtype=torch.long, device=device)
    loss_mask = torch.tensor(np.stack([s.loss_mask for s in batch.sequences]), dtype=torch.float32, device=device)
    position_ids = torch.tensor(np.stack([s.position_ids for s in batch.sequences]), dtype=torch.long, device=device)
    attn = np.stack([s.attention_mask() for s in batch.sequences])[:, None, :, :]  # (B,1,T,T)
    attn_mask = torch.tensor(attn, dtype=torch.bool, device=device)
    return {"input_ids": input_ids, "targets": targets, "loss_mask": loss_mask,
            "position_ids": position_ids, "attn_mask": attn_mask}


def build_golden_batch(manifests: dict[str, ShardManifest], shard_tokens: dict[str, np.ndarray],
                        device: str, seed: int = config.SEED + 999) -> dict:
    """Fixed reference batch OPUS scores every candidate against. Built once
    from its own RNG stream so it never interacts with (or is perturbed by)
    the main lineage's packing RNG -- it is never trained on."""
    grng = random.Random(seed)
    batch = packing.build_batch(0, "golden", "prose", manifests["prose"], shard_tokens["prose"], grng)
    return batch_to_tensors(batch, device)


def _update_latest_pointer(checkpoint_dir: Path, info: dict) -> None:
    import json
    (checkpoint_dir / "latest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def assert_trainable(lane: str, manifests: dict[str, ShardManifest]) -> None:
    """The evaluation/validation firewall. Raises if `lane` is tagged eval --
    called on every real training step, and also called directly (against
    the eval lane, on purpose) in run_demo.py to demonstrate the block."""
    if manifests[lane].role == "eval":
        raise RuntimeError(f"firewall breach: attempted to train on eval-tagged lane {lane!r}")


def run_steps(steps: list[int], *, model, optimizer, py_rng: random.Random, opus_gate: OpusGate,
              manifests: dict[str, ShardManifest], shard_tokens: dict[str, np.ndarray], device: str,
              consumption_ledger: JsonlLedger, learning_ledger: JsonlLedger, opus_ledger: JsonlLedger,
              logbook: Logbook, run_id: str, checkpoint_dir: Path | None = None,
              checkpoint_every: int | None = None, checkpoint_at: set[int] | None = None) -> list[dict]:
    results = []
    for step in steps:
        stage = mixture.stage_for_step(step)
        lane = mixture.sample_lane(step, py_rng)
        assert_trainable(lane, manifests)

        batch = packing.build_batch(step, stage.name, lane, manifests[lane], shard_tokens[lane], py_rng)
        tensors = batch_to_tensors(batch, device)
        decision = opus_gate.decide(tensors, lane, stage)
        trained = decision.decision in ("ACCEPT", "PROTECTED_FLOOR_OVERRIDE")

        loss_val = grad_norm = None
        if trained:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(tensors["input_ids"], position_ids=tensors["position_ids"],
                             attn_mask=tensors["attn_mask"], targets=tensors["targets"],
                             loss_mask=tensors["loss_mask"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item()
            optimizer.step()
            loss_val = loss.item()

        segments_per_seq = [[seg.to_dict() for seg in s.segments] for s in batch.sequences]
        consumption_ledger.append({
            "step": step, "run_id": run_id, "timestamp": time.time(),
            "batch_id": batch.batch_id, "batch_hash": batch.batch_hash,
            "lane": lane, "stage": stage.name, "accepted": trained,
            "opus_decision": decision.decision, "opus_score": decision.score, "opus_reason": decision.reason,
            "tokens_consumed": batch.tokens_consumed() if trained else 0,
            "total_slots": batch.total_slots(), "segments": segments_per_seq,
        })
        opus_ledger.append({
            "step": step, "run_id": run_id, "batch_id": batch.batch_id, "lane": lane, "stage": stage.name,
            "decision": decision.decision, "score": decision.score, "reason": decision.reason,
        })
        learning_offset = None
        if trained:
            learning_offset = learning_ledger.append({
                "step": step, "run_id": run_id, "batch_id": batch.batch_id, "lane": lane, "stage": stage.name,
                "loss": loss_val, "grad_norm": grad_norm,
            })

        logbook.event(
            f"step {step} run={run_id} lane={lane} stage={stage.name} opus={decision.decision} "
            f"score={decision.score:.4f} trained={trained}" + (f" loss={loss_val:.4f}" if trained else "")
        )
        results.append({
            "step": step, "batch_id": batch.batch_id, "batch_hash": batch.batch_hash, "lane": lane,
            "stage": stage.name, "decision": decision.decision, "trained": trained, "loss": loss_val,
            "learning_offset": learning_offset,
        })

        if checkpoint_dir is not None:
            due = (checkpoint_every and step % checkpoint_every == 0) or (checkpoint_at and step in checkpoint_at)
            if due:
                ckpt_path = checkpoint_dir / f"checkpoint_step{step:04d}.pt"
                info = checkpoint_mod.save_checkpoint(
                    ckpt_path, model=model, optimizer=optimizer, step=step, run_id=run_id, py_rng=py_rng,
                    consumption_offset=consumption_ledger.offset, learning_offset=learning_ledger.offset,
                )
                logbook.event(f"[PASS] checkpoint_saved step={step} file={ckpt_path.name} sha256={info['sha256'][:12]}")
                _update_latest_pointer(checkpoint_dir, info)
    return results
