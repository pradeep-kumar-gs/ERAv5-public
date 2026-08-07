from __future__ import annotations

from ledger import JsonlLedger


def test_append_assigns_monotonic_offsets(tmp_path):
    ledger = JsonlLedger(tmp_path / "l.jsonl")
    assert ledger.offset == 0
    o0 = ledger.append({"step": 1})
    o1 = ledger.append({"step": 2})
    o2 = ledger.append({"step": 3})
    assert (o0, o1, o2) == (0, 1, 2)
    assert ledger.offset == 3
    records = ledger.read_all()
    assert [r["ledger_offset"] for r in records] == [0, 1, 2]


def test_reopen_continues_offset_from_existing_file(tmp_path):
    path = tmp_path / "l.jsonl"
    first = JsonlLedger(path)
    first.append({"step": 1})
    first.append({"step": 2})

    second = JsonlLedger(path)
    assert second.offset == 2
    o = second.append({"step": 3})
    assert o == 2
    assert len(second.read_all()) == 3


def test_read_by_step_range_filters(tmp_path):
    ledger = JsonlLedger(tmp_path / "l.jsonl")
    for step in range(1, 11):
        ledger.append({"step": step})
    rows = ledger.read_by_step_range(4, 6)
    assert [r["step"] for r in rows] == [4, 5, 6]


def test_read_range_filters_by_ledger_offset(tmp_path):
    ledger = JsonlLedger(tmp_path / "l.jsonl")
    for step in range(1, 11):
        ledger.append({"step": step})
    rows = ledger.read_range(2, 4)
    assert [r["ledger_offset"] for r in rows] == [2, 3, 4]
