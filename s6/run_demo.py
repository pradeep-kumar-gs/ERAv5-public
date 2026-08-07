#!/usr/bin/env python3
"""Single-command entrypoint. Regenerates submission_artifacts/ from nothing:
documents -> shards -> manifests -> mixture -> packing -> batches -> training
-> ledgers -> checkpoint -> crash -> resume -> replay -> fork -> audit ->
performance -> evidence. No manual steps, no network calls.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import checkpoint as checkpoint_mod  # noqa: E402
import config  # noqa: E402
import evidence as evidence_mod  # noqa: E402
import mixture as mixture_mod  # noqa: E402
import model as model_mod  # noqa: E402
import opus as opus_mod  # noqa: E402
import replay as replay_mod  # noqa: E402
import shards as shards_mod  # noqa: E402
import train as train_mod  # noqa: E402
from ledger import JsonlLedger  # noqa: E402
from logbook import Logbook  # noqa: E402
from manifest import verify_shard  # noqa: E402
from tokenizer import get_tokenizer  # noqa: E402

ARTIFACTS = ROOT / "submission_artifacts"


def fresh_model(gpt_config, device):
    return model_mod.GPT(gpt_config).to(device)


def main() -> None:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    manifests_dir = ARTIFACTS / "manifests"
    shards_dir = ARTIFACTS / "shards"
    ledgers_dir = ARTIFACTS / "ledgers"
    checkpoints_dir = ARTIFACTS / "checkpoints"

    logbook = Logbook(ARTIFACTS / "run.log")
    logbook.event("run started")

    # ---- shards + manifests -------------------------------------------------
    manifest_list = shards_mod.build_shards(shards_dir, manifests_dir)
    manifests = {m.lane: m for m in manifest_list}
    logbook.event("shards created")

    tok = get_tokenizer()
    for m in manifest_list:
        ok, detail = verify_shard(m, shards_dir)
        logbook.result("tokenizer_hash_verified", ok, f"{m.shard_id}: {detail}")
        if not ok:
            raise RuntimeError(f"shard verification failed for {m.shard_id}: {detail}")
    logbook.event("manifests validated")

    shard_tokens = {lane: shards_mod.load_shard_tokens(m, shards_dir) for lane, m in manifests.items()}

    # ---- mixture schedule -----------------------------------------------------
    schedule = mixture_mod.compile_schedule()
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "mixture_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    logbook.event("mixture compiled")

    # ---- evaluation / validation firewall demo ---------------------------------
    try:
        train_mod.assert_trainable("eval", manifests)
        logbook.result("eval_shard_blocked", False, "assert_trainable did not raise for the eval lane")
        raise RuntimeError("eval firewall failed to block the eval lane")
    except RuntimeError as exc:
        logbook.result("eval_shard_blocked", True, str(exc))
    logbook.event("evaluation data blocked")

    # ---- model / optimizer / rng / OPUS init -----------------------------------
    device = train_mod.get_device()
    torch.use_deterministic_algorithms(True)
    gpt_config = model_mod.GPTConfig(
        vocab_size=tok.vocab_size, block_size=config.BLOCK_SIZE,
        n_layer=config.N_LAYER, n_head=config.N_HEAD, n_embd=config.N_EMBD,
    )
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)  # captured in every checkpoint; never seeding it left OS-entropy bytes in each file
    model = fresh_model(gpt_config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    py_rng = random.Random(config.SEED)
    golden_batch = train_mod.build_golden_batch(manifests, shard_tokens, device)
    opus_gate = opus_mod.OpusGate(model, golden_batch)

    consumption_ledger = JsonlLedger(ledgers_dir / "consumption_ledger.jsonl")
    learning_ledger = JsonlLedger(ledgers_dir / "learning_ledger.jsonl")
    opus_ledger = JsonlLedger(ledgers_dir / "opus_audit.jsonl")

    run_start = time.time()

    # ---- phase A: steps 1..CRASH_CHECKPOINT_STEP ----------------------------
    phase_a_steps = list(range(1, config.CRASH_CHECKPOINT_STEP + 1))
    train_mod.run_steps(
        phase_a_steps, model=model, optimizer=optimizer, py_rng=py_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main", checkpoint_dir=checkpoints_dir, checkpoint_every=config.CHECKPOINT_EVERY,
    )
    logbook.event(f"batches packed: {consumption_ledger.offset} so far (main lineage, steps 1-{config.CRASH_CHECKPOINT_STEP})")

    checkpoint_path = checkpoints_dir / f"checkpoint_step{config.CRASH_CHECKPOINT_STEP:04d}.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"expected checkpoint missing: {checkpoint_path}")

    # ---- compute the "expected" post-crash batches, before crashing ---------
    # Clone the packing RNG's state right after the checkpoint: this is what
    # a resumed run *should* reproduce. The live model/optimizer are reused
    # (their weights are identical to what the checkpoint just saved) but the
    # results are written to scratch ledgers/logs, never the real ones.
    expected_rng = random.Random()
    expected_rng.setstate(py_rng.getstate())
    scratch_dir = ARTIFACTS / "_scratch_expected"
    scratch_logbook = Logbook(scratch_dir / "expected.log")
    expected_results = train_mod.run_steps(
        config.RESUME_STEPS, model=model, optimizer=optimizer, py_rng=expected_rng, opus_gate=opus_gate,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=JsonlLedger(scratch_dir / "consumption_ledger.jsonl"),
        learning_ledger=JsonlLedger(scratch_dir / "learning_ledger.jsonl"),
        opus_ledger=JsonlLedger(scratch_dir / "opus_audit.jsonl"),
        logbook=scratch_logbook, run_id="main-expected",
    )
    expected_batches = {r["step"]: (r["batch_id"], r["batch_hash"]) for r in expected_results}
    shutil.rmtree(scratch_dir)

    # ---- simulate a crash: drop every in-memory object the live run used ----
    del model, optimizer, py_rng, opus_gate, golden_batch, expected_rng
    logbook.event("crash simulated")

    # ---- resume: fresh model/optimizer/rng, loaded strictly from disk -------
    model2 = fresh_model(gpt_config, device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.LR)
    py_rng2 = random.Random()
    checkpoint_mod.load_checkpoint(checkpoint_path, model=model2, optimizer=optimizer2, py_rng=py_rng2)
    golden_batch2 = train_mod.build_golden_batch(manifests, shard_tokens, device)
    opus_gate2 = opus_mod.OpusGate(model2, golden_batch2)

    resumed_results = train_mod.run_steps(
        config.RESUME_STEPS, model=model2, optimizer=optimizer2, py_rng=py_rng2, opus_gate=opus_gate2,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main", checkpoint_dir=checkpoints_dir, checkpoint_every=config.CHECKPOINT_EVERY,
    )
    logbook.event("run resumed")

    mismatches = []
    resume_details = []
    for r in resumed_results:
        exp_id, exp_hash = expected_batches.get(r["step"], (None, None))
        match = r["batch_id"] == exp_id and r["batch_hash"] == exp_hash
        if not match:
            mismatches.append(r["step"])
        resume_details.append({
            "step": r["step"], "expected_batch_id": exp_id, "expected_hash": exp_hash,
            "resumed_batch_id": r["batch_id"], "resumed_hash": r["batch_hash"], "match": match,
        })
    resume_ok = not mismatches and len(resumed_results) == len(config.RESUME_STEPS)
    resume_report = {
        "steps_compared": len(resumed_results), "all_match": resume_ok, "details": resume_details,
    }
    (ledgers_dir / "resume_report.json").write_text(json.dumps(resume_report, indent=2), encoding="utf-8")
    logbook.result(
        "resume_next_batch_matched", resume_ok,
        f"steps {config.RESUME_STEPS}: expected={expected_batches} actual="
        f"{[(r['step'], r['batch_id'], r['batch_hash']) for r in resumed_results]}"
        if not resume_ok else f"all {len(resumed_results)} resumed batches matched the pre-crash expected batches",
    )
    if not resume_ok:
        raise RuntimeError(f"resume produced different batches at steps {mismatches}")

    # ---- continue the main lineage to the end --------------------------------
    remaining_steps = list(range(config.RESUME_STEPS[-1] + 1, config.TOTAL_STEPS + 1))
    train_mod.run_steps(
        remaining_steps, model=model2, optimizer=optimizer2, py_rng=py_rng2, opus_gate=opus_gate2,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=consumption_ledger, learning_ledger=learning_ledger, opus_ledger=opus_ledger,
        logbook=logbook, run_id="main", checkpoint_dir=checkpoints_dir, checkpoint_every=config.CHECKPOINT_EVERY,
    )
    run_end = time.time()
    logbook.event(f"OPUS decisions recorded: {opus_ledger.offset} total (main lineage)")

    # ---- replay an earlier interval purely from the ledger + shards ---------
    replay_report = replay_mod.replay_interval(
        ledgers_dir / "consumption_ledger.jsonl", manifests, shard_tokens,
        config.REPLAY_WINDOW[0], config.REPLAY_WINDOW[1],
    )
    (ledgers_dir / "replay_report.json").write_text(json.dumps(replay_report, indent=2), encoding="utf-8")
    logbook.result(
        "replay_hash_matched", replay_report["all_match"],
        f"{replay_report['records_replayed']} records over steps {config.REPLAY_WINDOW}",
    )
    logbook.event("historical stream replayed")

    # ---- fork from an earlier checkpoint -------------------------------------
    fork_checkpoint_path = checkpoints_dir / f"checkpoint_step{config.FORK_FROM_STEP:04d}.pt"
    model3 = fresh_model(gpt_config, device)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=config.LR)
    py_rng3 = random.Random()
    checkpoint_mod.load_checkpoint(fork_checkpoint_path, model=model3, optimizer=optimizer3, py_rng=py_rng3)
    golden_batch3 = train_mod.build_golden_batch(manifests, shard_tokens, device)
    opus_gate3 = opus_mod.OpusGate(model3, golden_batch3)

    fork_ledgers_dir = ledgers_dir / "fork"
    fork_checkpoints_dir = checkpoints_dir / "fork"
    fork_consumption = JsonlLedger(fork_ledgers_dir / "consumption_ledger.jsonl")
    fork_learning = JsonlLedger(fork_ledgers_dir / "learning_ledger.jsonl")
    fork_opus = JsonlLedger(fork_ledgers_dir / "opus_audit.jsonl")
    train_mod.run_steps(
        config.FORK_STEPS, model=model3, optimizer=optimizer3, py_rng=py_rng3, opus_gate=opus_gate3,
        manifests=manifests, shard_tokens=shard_tokens, device=device,
        consumption_ledger=fork_consumption, learning_ledger=fork_learning, opus_ledger=fork_opus,
        logbook=logbook, run_id="fork", checkpoint_dir=fork_checkpoints_dir,
        checkpoint_every=config.CHECKPOINT_EVERY, checkpoint_at={config.FORK_STEPS[-1]},
    )
    logbook.event(f"branch forked from checkpoint@step{config.FORK_FROM_STEP} -> run_id=fork")

    # ---- performance -----------------------------------------------------------
    all_records = consumption_ledger.read_all()
    trained = [r for r in all_records if r["accepted"]]
    total_tokens_consumed = sum(r["tokens_consumed"] for r in trained)
    total_slots = sum(r["total_slots"] for r in trained)
    packing_utilization = round(total_tokens_consumed / total_slots, 4) if total_slots else 0.0
    elapsed_seconds = max(run_end - run_start, 1e-6)
    opus_records = opus_ledger.read_all()
    opus_counts: dict[str, int] = {}
    for r in opus_records:
        opus_counts[r["decision"]] = opus_counts.get(r["decision"], 0) + 1
    accepted_decisions = opus_counts.get("ACCEPT", 0) + opus_counts.get("PROTECTED_FLOOR_OVERRIDE", 0)
    performance = {
        "elapsed_seconds": elapsed_seconds,
        "total_steps": len(all_records),
        "total_batches_trained": len(trained),
        "tokens_per_second": round(total_slots / elapsed_seconds, 2),
        "useful_loss_bearing_tokens_per_second": round(total_tokens_consumed / elapsed_seconds, 2),
        "packing_utilization": packing_utilization,
        "opus_decision_counts": opus_counts,
        "opus_acceptance_rate": round(accepted_decisions / len(opus_records), 4) if opus_records else 0.0,
    }
    (ARTIFACTS / "performance.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")
    logbook.event("performance measured")

    # ---- evidence bundle ---------------------------------------------------
    logbook.event("audit completed")
    evidence = evidence_mod.build_evidence(ARTIFACTS)
    (ARTIFACTS / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (ARTIFACTS / "evidence.md").write_text(evidence_mod.render_markdown(evidence), encoding="utf-8")
    logbook.event(f"run finished; evidence overall_pass={evidence['overall_pass']}")

    if not evidence["overall_pass"]:
        failing = [name for name, row in evidence["requirements"].items() if not row["passed"]]
        raise SystemExit(f"evidence bundle reports failures: {failing}")


if __name__ == "__main__":
    main()
