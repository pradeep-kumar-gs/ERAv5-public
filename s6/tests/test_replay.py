from __future__ import annotations

import random

import torch

import config
import replay as replay_mod
import train as train_mod
from ledger import JsonlLedger
from logbook import Logbook


def test_replay_reconstructs_identical_batches(manifests, shard_tokens, device, tiny_model, opus_gate, tmp_path):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=config.LR)
    py_rng = random.Random(config.SEED)
    ledger_path = tmp_path / "consumption_ledger.jsonl"
    consumption_ledger = JsonlLedger(ledger_path)
    learning_ledger = JsonlLedger(tmp_path / "learning_ledger.jsonl")
    opus_ledger = JsonlLedger(tmp_path / "opus_audit.jsonl")
    logbook = Logbook(tmp_path / "run.log")

    train_mod.run_steps(
        list(range(1, 6)), model=tiny_model, optimizer=optimizer, py_rng=py_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="test",
    )

    report = replay_mod.replay_interval(ledger_path, manifests, shard_tokens, 2, 4)
    assert report["records_replayed"] == 3
    assert report["all_match"] is True
    for detail in report["details"]:
        assert detail["match"] is True
        assert detail["original_hash"] == detail["replayed_hash"]
        assert detail["original_batch_id"] == detail["replayed_batch_id"]


def test_replay_detects_a_tampered_ledger_record(manifests, shard_tokens, device, tiny_model, opus_gate, tmp_path):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=config.LR)
    py_rng = random.Random(config.SEED)
    ledger_path = tmp_path / "consumption_ledger.jsonl"
    consumption_ledger = JsonlLedger(ledger_path)
    learning_ledger = JsonlLedger(tmp_path / "learning_ledger.jsonl")
    opus_ledger = JsonlLedger(tmp_path / "opus_audit.jsonl")
    logbook = Logbook(tmp_path / "run.log")

    train_mod.run_steps(
        [1, 2, 3], model=tiny_model, optimizer=optimizer, py_rng=py_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="test",
    )

    import json
    lines = ledger_path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["batch_hash"] = "0" * 64
    lines[0] = json.dumps(tampered)
    ledger_path.write_text("\n".join(lines) + "\n")

    report = replay_mod.replay_interval(ledger_path, manifests, shard_tokens, 1, 3)
    assert report["all_match"] is False
    assert report["details"][0]["match"] is False
