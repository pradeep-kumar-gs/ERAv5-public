from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import checkpoint as checkpoint_mod  # noqa: E402
import config  # noqa: E402
import mixture as mixture_mod  # noqa: E402
import replay as replay_mod  # noqa: E402
import shards as shards_mod  # noqa: E402
import train as train_mod  # noqa: E402
from ledger import JsonlLedger  # noqa: E402
from logbook import Logbook  # noqa: E402
from manifest import load_all_manifests  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from opus import OpusGate  # noqa: E402
from tokenizer import get_tokenizer  # noqa: E402

train_mod.get_device()  # forces CPU import path; nothing else needed


@pytest.fixture(scope="session")
def corpus_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("s6_corpus")
    shards_dir = root / "shards"
    manifests_dir = root / "manifests"
    shards_mod.build_shards(shards_dir, manifests_dir)
    return shards_dir, manifests_dir


@pytest.fixture(scope="session")
def manifests(corpus_dirs):
    _, manifests_dir = corpus_dirs
    return {m.lane: m for m in load_all_manifests(manifests_dir)}


@pytest.fixture(scope="session")
def shard_tokens(corpus_dirs, manifests):
    shards_dir, _ = corpus_dirs
    return {lane: shards_mod.load_shard_tokens(m, shards_dir) for lane, m in manifests.items()}


@pytest.fixture(scope="session")
def device():
    return train_mod.get_device()


@pytest.fixture
def tiny_model(device):
    tok = get_tokenizer()
    import torch

    torch.manual_seed(0)
    gpt_config = GPTConfig(vocab_size=tok.vocab_size, block_size=config.BLOCK_SIZE, n_layer=1, n_head=2, n_embd=32)
    return GPT(gpt_config).to(device)


@pytest.fixture
def golden_batch(manifests, shard_tokens, device):
    return train_mod.build_golden_batch(manifests, shard_tokens, device, seed=999)


@pytest.fixture
def opus_gate(tiny_model, golden_batch):
    return OpusGate(tiny_model, golden_batch)


@pytest.fixture
def py_rng():
    return random.Random(config.SEED)


def _build_fresh_model(device):
    tok = get_tokenizer()
    torch.manual_seed(0)
    gpt_config = GPTConfig(vocab_size=tok.vocab_size, block_size=config.BLOCK_SIZE, n_layer=1, n_head=2, n_embd=32)
    return GPT(gpt_config).to(device)


@pytest.fixture
def built_artifacts_dir(manifests, shard_tokens, device, corpus_dirs, tmp_path):
    """A real (small-scale) submission_artifacts/ tree, built the same way run_demo.py
    builds the full one -- actual shards/ledgers/checkpoints/reports on disk, not
    fixtures pretending to be one. Evidence-quality tests tamper with copies of this
    to prove evidence.py catches what it claims to catch, instead of trusting it.
    """
    shards_dir_src, manifests_dir_src = corpus_dirs
    artifacts = tmp_path / "artifacts"
    shutil.copytree(shards_dir_src, artifacts / "shards")
    shutil.copytree(manifests_dir_src, artifacts / "manifests")
    schedule = mixture_mod.compile_schedule()
    (artifacts / "manifests" / "mixture_schedule.json").write_text(json.dumps(schedule, indent=2))

    ledgers_dir = artifacts / "ledgers"
    checkpoints_dir = artifacts / "checkpoints"
    logbook = Logbook(artifacts / "run.log")
    logbook.event("run started")

    try:
        train_mod.assert_trainable("eval", manifests)
        logbook.result("eval_shard_blocked", False, "eval lane was not blocked")
    except RuntimeError as exc:
        logbook.result("eval_shard_blocked", True, str(exc))

    model = _build_fresh_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    py_rng = random.Random(config.SEED)
    golden_batch = train_mod.build_golden_batch(manifests, shard_tokens, device, seed=999)
    opus_gate = OpusGate(model, golden_batch)

    consumption_ledger = JsonlLedger(ledgers_dir / "consumption_ledger.jsonl")
    learning_ledger = JsonlLedger(ledgers_dir / "learning_ledger.jsonl")
    opus_ledger = JsonlLedger(ledgers_dir / "opus_audit.jsonl")

    train_mod.run_steps(
        [1, 2, 3], model=model, optimizer=optimizer, py_rng=py_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main", checkpoint_dir=checkpoints_dir, checkpoint_every=3,
    )
    checkpoint_path = checkpoints_dir / "checkpoint_step0003.pt"

    expected_rng = random.Random()
    expected_rng.setstate(py_rng.getstate())
    scratch = tmp_path / "scratch"
    expected_results = train_mod.run_steps(
        [4, 5], model=model, optimizer=optimizer, py_rng=expected_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=JsonlLedger(scratch / "consumption_ledger.jsonl"),
        learning_ledger=JsonlLedger(scratch / "learning_ledger.jsonl"),
        opus_ledger=JsonlLedger(scratch / "opus_audit.jsonl"),
        logbook=Logbook(scratch / "run.log"), run_id="main-expected",
    )
    expected_batches = {r["step"]: (r["batch_id"], r["batch_hash"]) for r in expected_results}
    shutil.rmtree(scratch)

    del model, optimizer, py_rng, opus_gate, golden_batch, expected_rng
    logbook.event("crash simulated")

    model2 = _build_fresh_model(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.LR)
    py_rng2 = random.Random()
    checkpoint_mod.load_checkpoint(checkpoint_path, model=model2, optimizer=optimizer2, py_rng=py_rng2)
    golden_batch2 = train_mod.build_golden_batch(manifests, shard_tokens, device, seed=999)
    opus_gate2 = OpusGate(model2, golden_batch2)

    resumed_results = train_mod.run_steps(
        [4, 5], model=model2, optimizer=optimizer2, py_rng=py_rng2, opus_gate=opus_gate2,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main", checkpoint_dir=checkpoints_dir, checkpoint_every=3,
    )
    resume_details = []
    all_match = True
    for r in resumed_results:
        exp_id, exp_hash = expected_batches.get(r["step"], (None, None))
        match = r["batch_id"] == exp_id and r["batch_hash"] == exp_hash
        all_match = all_match and match
        resume_details.append({
            "step": r["step"], "expected_batch_id": exp_id, "expected_hash": exp_hash,
            "resumed_batch_id": r["batch_id"], "resumed_hash": r["batch_hash"], "match": match,
        })
    resume_report = {"steps_compared": len(resumed_results), "all_match": all_match, "details": resume_details}
    (ledgers_dir / "resume_report.json").write_text(json.dumps(resume_report, indent=2))
    logbook.result("resume_next_batch_matched", all_match, f"steps compared={len(resumed_results)}")

    train_mod.run_steps(
        [6], model=model2, optimizer=optimizer2, py_rng=py_rng2, opus_gate=opus_gate2,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main",
    )
    logbook.event("run resumed")

    replay_report = replay_mod.replay_interval(
        ledgers_dir / "consumption_ledger.jsonl", manifests, shard_tokens, 2, 4,
    )
    (ledgers_dir / "replay_report.json").write_text(json.dumps(replay_report, indent=2))
    logbook.result("replay_hash_matched", replay_report["all_match"], f"records={replay_report['records_replayed']}")
    logbook.event("historical stream replayed")
    logbook.event("audit completed")

    all_records = consumption_ledger.read_all()
    trained = [r for r in all_records if r["accepted"]]
    total_tokens = sum(r["tokens_consumed"] for r in trained)
    total_slots = sum(r["total_slots"] for r in trained)
    packing_utilization = round(total_tokens / total_slots, 4) if total_slots else 0.0
    opus_counts: dict[str, int] = {}
    for r in opus_ledger.read_all():
        opus_counts[r["decision"]] = opus_counts.get(r["decision"], 0) + 1
    performance = {
        "elapsed_seconds": 0.01,
        "total_steps": len(all_records),
        "total_batches_trained": len(trained),
        "tokens_per_second": round(total_slots / 0.01, 2),
        "useful_loss_bearing_tokens_per_second": round(total_tokens / 0.01, 2),
        "packing_utilization": packing_utilization,
        "opus_decision_counts": opus_counts,
        "opus_acceptance_rate": round(len(trained) / len(all_records), 4) if all_records else 0.0,
    }
    (artifacts / "performance.json").write_text(json.dumps(performance, indent=2))
    logbook.event("performance measured")

    return artifacts
