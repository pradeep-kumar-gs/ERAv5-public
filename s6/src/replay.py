"""Reconstruct a historical interval purely from the consumption ledger and
the immutable shards -- no model, no RNG, no live training state. This is
the audit path: if a reviewer wants to know exactly what the model saw at
step N, this is what they'd run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import packing
from ledger import JsonlLedger
from manifest import ShardManifest


def replay_interval(consumption_ledger_path: Path, manifests: dict[str, ShardManifest],
                     shard_tokens: dict[str, np.ndarray], start_step: int, end_step: int) -> dict:
    ledger = JsonlLedger(consumption_ledger_path)
    records = ledger.read_by_step_range(start_step, end_step)
    details = []
    for rec in records:
        lane, step, stage = rec["lane"], rec["step"], rec["stage"]
        segments_per_seq = [[packing.Segment.from_dict(s) for s in seq] for seq in rec["segments"]]
        rebuilt = packing.rebuild_batch(step, stage, lane, segments_per_seq, shard_tokens[lane], manifests[lane])
        match = (rebuilt.batch_hash == rec["batch_hash"]) and (rebuilt.batch_id == rec["batch_id"])
        details.append({
            "step": step, "lane": lane, "original_batch_id": rec["batch_id"],
            "replayed_batch_id": rebuilt.batch_id, "original_hash": rec["batch_hash"],
            "replayed_hash": rebuilt.batch_hash, "match": bool(match),
        })
    all_match = bool(details) and all(d["match"] for d in details)
    return {"start_step": start_step, "end_step": end_step, "records_replayed": len(details),
            "all_match": all_match, "details": details}
