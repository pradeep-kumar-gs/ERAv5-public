"""Proves evidence.py doesn't rubber-stamp: build a real (small-scale) artifacts
tree via built_artifacts_dir, then tamper it one way at a time and confirm
build_evidence() independently catches each specific failure mode the
assignment calls out (leaked eval batch, repeated/skipped resume, mismatched
replay, hardcoded throughput, etc.) rather than trusting a log line.
"""
from __future__ import annotations

import json

import config
import evidence as evidence_mod


def _rewrite_first_line(path, mutate):
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    mutate(rec)
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")


def test_evidence_passes_on_an_untampered_run(built_artifacts_dir):
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["overall_pass"] is True
    for name, row in ev["requirements"].items():
        assert row["passed"] is True, f"{name}: {row['detail']}"


def test_evidence_catches_a_leaked_eval_batch(built_artifacts_dir):
    path = built_artifacts_dir / "ledgers" / "consumption_ledger.jsonl"
    _rewrite_first_line(path, lambda rec: rec.__setitem__("lane", "eval"))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Evaluation firewall"]["passed"] is False
    assert ev["overall_pass"] is False


def test_evidence_catches_a_corrupted_batch_hash(built_artifacts_dir):
    path = built_artifacts_dir / "ledgers" / "consumption_ledger.jsonl"
    _rewrite_first_line(path, lambda rec: rec.__setitem__("batch_hash", "0" * 64))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Packing correctness"]["passed"] is False


def test_evidence_catches_a_falsified_resume_report(built_artifacts_dir):
    path = built_artifacts_dir / "ledgers" / "resume_report.json"
    report = json.loads(path.read_text())
    report["all_match"] = False
    path.write_text(json.dumps(report))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Crash recovery"]["passed"] is False


def test_evidence_catches_a_missing_resume_report(built_artifacts_dir):
    (built_artifacts_dir / "ledgers" / "resume_report.json").unlink()
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Crash recovery"]["passed"] is False


def test_evidence_catches_a_falsified_replay_report(built_artifacts_dir):
    path = built_artifacts_dir / "ledgers" / "replay_report.json"
    report = json.loads(path.read_text())
    report["all_match"] = False
    path.write_text(json.dumps(report))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Replay"]["passed"] is False


def test_evidence_catches_a_missing_opus_decision(built_artifacts_dir):
    path = built_artifacts_dir / "ledgers" / "opus_audit.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")  # one candidate now has no logged decision
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["OPUS audit trail"]["passed"] is False


def test_evidence_catches_an_orphaned_learning_ledger_row(built_artifacts_dir):
    path = built_artifacts_dir / "ledgers" / "learning_ledger.jsonl"
    _rewrite_first_line(path, lambda rec: rec.__setitem__("batch_id", "does-not-exist"))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Learning trace"]["passed"] is False


def test_evidence_catches_a_protected_floor_breach(built_artifacts_dir):
    path = built_artifacts_dir / "manifests" / "mixture_schedule.json"
    schedule = json.loads(path.read_text())
    for row in schedule:
        row["protected_floor"][config.PROTECTED_LANE] = 0.99
    path.write_text(json.dumps(schedule))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Mixture compliance"]["passed"] is False


def test_evidence_catches_a_falsified_throughput_number(built_artifacts_dir):
    path = built_artifacts_dir / "performance.json"
    perf = json.loads(path.read_text())
    perf["packing_utilization"] = 0.123456
    path.write_text(json.dumps(perf))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Throughput"]["passed"] is False


def test_evidence_catches_a_corrupted_shard_file(built_artifacts_dir):
    path = built_artifacts_dir / "shards" / "shard-prose.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    ev = evidence_mod.build_evidence(built_artifacts_dir)
    assert ev["requirements"]["Tokenizer integrity"]["passed"] is False
