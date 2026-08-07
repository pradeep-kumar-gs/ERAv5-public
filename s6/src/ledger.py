"""Append-only JSONL ledgers with a monotonic `ledger_offset`.

Two ledgers, per the assignment: `consumption_ledger.jsonl` (what was fed to
the model -- batch id, lane, stage, shard/token spans, OPUS decision) and
`learning_ledger.jsonl` (what came back -- loss, grad norm, per-lane
attribution). Checkpoints record the offset into each ledger at save time,
which is what lets resume/replay/fork prove they aren't reading a ledger
that was truncated or reordered out from under them.
"""
from __future__ import annotations

import json
from pathlib import Path


class JsonlLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self._count = self._count_lines()

    def _count_lines(self) -> int:
        with self.path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def append(self, record: dict) -> int:
        offset = self._count
        full = {"ledger_offset": offset, **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(full) + "\n")
        self._count += 1
        return offset

    @property
    def offset(self) -> int:
        """Number of records written so far == offset the next record will get."""
        return self._count

    def read_all(self) -> list[dict]:
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def read_range(self, start_offset: int, end_offset: int) -> list[dict]:
        return [r for r in self.read_all() if start_offset <= r["ledger_offset"] <= end_offset]

    def read_by_step_range(self, start_step: int, end_step: int) -> list[dict]:
        return [r for r in self.read_all() if start_step <= r["step"] <= end_step]
