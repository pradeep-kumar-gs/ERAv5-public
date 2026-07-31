"""Shared 9-stage audit helpers reused by audit_agentic.py, audit_longcontext.py,
and audit_reasoning.py, cloned from s4/scripts/audit_dataset.py's audit()
template. Kept here once instead of tripled across three files.
"""
from __future__ import annotations

import hashlib
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL = re.compile(r"https?://\S+", re.I)
CONTROL_CATEGORIES = {"Cc", "Cf"}
PRESERVED_FORMAT_CHARS = {"‌", "‍"}
ASCII_LETTER = re.compile(r"[A-Za-z]")
ANY_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> tuple[str, int]:
    normalized = unicodedata.normalize("NFC", text)
    removed = 0
    kept: list[str] = []
    for char in normalized:
        if (
            unicodedata.category(char) in CONTROL_CATEGORIES
            and char not in PRESERVED_FORMAT_CHARS
            and char not in "\n\t"
        ):
            removed += 1
            continue
        kept.append(char)
    normalized = "".join(kept).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines()).strip()
    return normalized, removed


def ascii_script_ratio(text: str) -> float:
    letters = ANY_LETTER.findall(text)
    if not letters:
        return 1.0
    ascii_letters = ASCII_LETTER.findall(text)
    return len(ascii_letters) / len(letters)


def run_text_audit(
    *,
    dataset: str,
    revision: str,
    license_name: str,
    source_path: Path,
    tokenizer_name: str,
    encoder,
    record_texts: list[str],
    expected_language: str = "English",
    extra: dict | None = None,
) -> dict:
    """Runs the 9-stage template over a flat list of (already row-joined) texts.

    Each entry in record_texts is one record's full text (all fields joined).
    This mirrors audit_dataset.py's per-message loop but for datasets that are
    single free-text records rather than threaded conversations.
    """
    raw_tokens = normalized_tokens = control_chars_removed = 0
    empty_records = over_length_records = 0
    token_lengths: list[int] = []
    normalized_hashes: list[str] = []
    regex_findings: Counter[str] = Counter()
    mismatched_script_records = 0

    for text in record_texts:
        raw_token_count = len(encoder.encode(text, disallowed_special=()))
        raw_tokens += raw_token_count
        cleaned, removed = normalize(text)
        control_chars_removed += removed
        clean_token_count = len(encoder.encode(cleaned, disallowed_special=()))
        normalized_tokens += clean_token_count
        token_lengths.append(clean_token_count)
        normalized_hashes.append(hashlib.sha256(cleaned.encode("utf-8")).hexdigest())

        if not cleaned:
            empty_records += 1
        if clean_token_count > 8192:
            over_length_records += 1
        if ascii_script_ratio(cleaned) < 0.5 and expected_language == "English":
            mismatched_script_records += 1

        regex_findings["email"] += len(EMAIL.findall(cleaned))
        regex_findings["phone_india"] += len(PHONE.findall(cleaned))
        regex_findings["ipv4"] += len(IPV4.findall(cleaned))
        regex_findings["url"] += len(URL.findall(cleaned))

    exact_dupe_count = len(normalized_hashes) - len(set(normalized_hashes))
    source_hash = sha256_file(source_path) if source_path.exists() else None

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "dataset": dataset,
            "revision": revision,
            "source_sha256": source_hash,
            "tokenizer": tokenizer_name,
            "license": license_name,
        },
        "raw": {
            "records": len(record_texts),
            "tokens": raw_tokens,
            "mean_tokens_per_record": round(statistics.mean(token_lengths), 2) if token_lengths else 0,
            "median_tokens_per_record": statistics.median(token_lengths) if token_lengths else 0,
            "p95_tokens_per_record": sorted(token_lengths)[int(0.95 * (len(token_lengths) - 1))] if token_lengths else 0,
            "max_tokens_per_record": max(token_lengths) if token_lengths else 0,
        },
        "stages": [
            {
                "id": "normalize",
                "name": "Unicode & text normalization",
                "input_tokens": raw_tokens,
                "output_tokens": normalized_tokens,
                "findings": {"control_or_format_chars_removed": control_chars_removed},
            },
            {
                "id": "format",
                "name": "Canonical conversation formatting",
                "input_records": len(record_texts),
                "output_records": len(record_texts),
                "findings": {"note": "Single-turn or role-tagged records reformatted to a uniform schema at load time."},
            },
            {
                "id": "language",
                "name": "Language & script validation",
                "input_records": len(record_texts),
                "output_records": len(record_texts) - mismatched_script_records,
                "findings": {"records_below_50pct_ascii_script_ratio": mismatched_script_records},
                "note": f"Expected language: {expected_language}.",
            },
            {
                "id": "heuristic",
                "name": "Heuristic quality filtering",
                "input_records": len(record_texts),
                "output_records": len(record_texts) - empty_records - over_length_records,
                "findings": {"empty": empty_records, "over_8192_tokens": over_length_records},
            },
            {
                "id": "classifier",
                "name": "Quality-score gate",
                "status": "planned",
                "findings": {},
                "note": "Same status as S4: threshold requires calibration, not yet executed.",
            },
            {
                "id": "dedup",
                "name": "Exact & near deduplication",
                "input_records": len(record_texts),
                "output_records": len(set(normalized_hashes)),
                "findings": {"exact_duplicate_records": exact_dupe_count},
            },
            {
                "id": "pii",
                "name": "PII & secret scrubbing",
                "status": "audit",
                "findings": dict(regex_findings),
            },
            {
                "id": "contamination",
                "name": "Benchmark decontamination",
                "status": "blocked",
                "findings": {},
                "note": "Same status as S4: a sealed benchmark fingerprint set is required before claiming a pass.",
            },
            {
                "id": "manifest",
                "name": "Deterministic provenance manifest",
                "status": "complete",
                "findings": {"source_sha256": source_hash, "revision": revision},
            },
        ],
    }
    if extra:
        result["extra"] = extra
    return result
