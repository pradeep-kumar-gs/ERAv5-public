from __future__ import annotations

import random

import torch

import checkpoint as checkpoint_mod
import config
import train as train_mod
from ledger import JsonlLedger
from logbook import Logbook
from model import GPT, GPTConfig
from opus import OpusGate
from tokenizer import get_tokenizer


def _build_model(device):
    tok = get_tokenizer()
    torch.manual_seed(0)
    gpt_config = GPTConfig(vocab_size=tok.vocab_size, block_size=config.BLOCK_SIZE, n_layer=1, n_head=2, n_embd=32)
    return GPT(gpt_config).to(device), gpt_config


def test_resume_after_simulated_crash_matches_expected_batches(manifests, shard_tokens, device, tmp_path):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model, gpt_config = _build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    py_rng = random.Random(config.SEED)
    golden_batch = train_mod.build_golden_batch(manifests, shard_tokens, device, seed=999)
    opus_gate = OpusGate(model, golden_batch)

    ledgers_dir = tmp_path / "ledgers"
    checkpoints_dir = tmp_path / "checkpoints"
    consumption_ledger = JsonlLedger(ledgers_dir / "consumption_ledger.jsonl")
    learning_ledger = JsonlLedger(ledgers_dir / "learning_ledger.jsonl")
    opus_ledger = JsonlLedger(ledgers_dir / "opus_audit.jsonl")
    logbook = Logbook(tmp_path / "run.log")

    train_mod.run_steps(
        [1, 2, 3], model=model, optimizer=optimizer, py_rng=py_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main", checkpoint_dir=checkpoints_dir, checkpoint_every=3,
    )
    checkpoint_path = checkpoints_dir / "checkpoint_step0003.pt"
    assert checkpoint_path.exists()

    # expected: clone the rng right after the checkpoint, reuse the live model
    expected_rng = random.Random()
    expected_rng.setstate(py_rng.getstate())
    scratch = tmp_path / "scratch"
    expected_results = train_mod.run_steps(
        [4, 5, 6], model=model, optimizer=optimizer, py_rng=expected_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=JsonlLedger(scratch / "consumption_ledger.jsonl"),
        learning_ledger=JsonlLedger(scratch / "learning_ledger.jsonl"),
        opus_ledger=JsonlLedger(scratch / "opus_audit.jsonl"),
        logbook=Logbook(scratch / "run.log"), run_id="main-expected",
    )
    expected = {r["step"]: (r["batch_id"], r["batch_hash"]) for r in expected_results}

    # simulate a crash: drop every in-memory object
    del model, optimizer, py_rng, opus_gate, golden_batch, expected_rng

    model2, _ = _build_model(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.LR)
    py_rng2 = random.Random()
    checkpoint_mod.load_checkpoint(checkpoint_path, model=model2, optimizer=optimizer2, py_rng=py_rng2)
    golden_batch2 = train_mod.build_golden_batch(manifests, shard_tokens, device, seed=999)
    opus_gate2 = OpusGate(model2, golden_batch2)

    resumed_results = train_mod.run_steps(
        [4, 5, 6], model=model2, optimizer=optimizer2, py_rng=py_rng2, opus_gate=opus_gate2,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main",
    )

    for r in resumed_results:
        exp_id, exp_hash = expected[r["step"]]
        assert r["batch_id"] == exp_id
        assert r["batch_hash"] == exp_hash


def test_resume_does_not_skip_or_repeat_ledger_steps(manifests, shard_tokens, device, tmp_path):
    model, _ = _build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    py_rng = random.Random(config.SEED)
    golden_batch = train_mod.build_golden_batch(manifests, shard_tokens, device, seed=999)
    opus_gate = OpusGate(model, golden_batch)

    consumption_ledger = JsonlLedger(tmp_path / "consumption_ledger.jsonl")
    learning_ledger = JsonlLedger(tmp_path / "learning_ledger.jsonl")
    opus_ledger = JsonlLedger(tmp_path / "opus_audit.jsonl")
    logbook = Logbook(tmp_path / "run.log")

    train_mod.run_steps(
        list(range(1, 7)), model=model, optimizer=optimizer, py_rng=py_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main",
    )
    steps = [r["step"] for r in consumption_ledger.read_all()]
    assert steps == list(range(1, 7))
