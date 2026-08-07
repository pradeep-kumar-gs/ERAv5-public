"""Append-only run.log writer. Every important system event goes through
here so run.log is always the single, literal execution record — never
reconstructed after the fact.
"""
from __future__ import annotations

import time
from pathlib import Path


class Logbook:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def _write(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def event(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write(f"[{ts}] {message}")

    def result(self, name: str, passed: bool, detail: str = "") -> None:
        tag = "PASS" if passed else "FAIL"
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        suffix = f" ({detail})" if detail else ""
        self._write(f"[{ts}] [{tag}] {name}{suffix}")
