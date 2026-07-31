#!/usr/bin/env python3
"""Spot-sample codeparrot/github-code-clean, restricted to permissively licensed
files, to give the Code lane (D36's zero-supply admission) its first real,
measured data point.

This is a SPOT-SAMPLE, not a full-corpus audit: github-code-clean spans 883
parquet shards (~847GB after CodeParrot's quality filtering). Downloading and
auditing all of it is out of scope for this assignment; instead this script
downloads shard 0 (the same single-shard-spot-sample methodology used for
Sangraha in audit_sangraha.py) and applies deterministic SHA-256 rank sampling
restricted to a permissive-license allowlist. Every other shard's numbers stay
an extrapolation from this one shard's measured license mix, honestly labeled
as such -- not a corpus-wide measurement.

Per-file license is a real column in this dataset (`license`), inherited from
each file's origin GitHub repository at BigQuery-export time -- this is the
same per-file license provenance approach StarCoder's training data used, not
an invented signal.

Raw shard is large (~130MB); downloaded to s5/data/raw/github-code/
(gitignored) and never published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow
import tiktoken
from huggingface_hub import hf_hub_download

TOKENIZER = "cl100k_base"
DEFAULT_TARGET = 2_000_000
SHARD_FILE = "data/train-00000-of-00880.parquet"
TOTAL_SHARDS = 883  # codeparrot/github-code-clean's data/ directory file count, checked via HfApi

# Permissive, redistribution-friendly licenses only -- excludes copyleft
# (gpl-*, agpl-3.0, lgpl-*) and weak-copyleft/ambiguous (mpl-2.0, epl-1.0,
# artistic-2.0) licenses, matching the conservative allowlist StarCoder's
# training data used for the same reason: pretraining reuse without
# redistribution/share-alike obligations.
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "cc0-1.0",
    "unlicense",
}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SECRET_LIKE = re.compile(
    r"\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][A-Za-z0-9_\-]{8,}['\"]",
    re.I,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    kept = [
        char
        for char in text
        if not (unicodedata.category(char) in {"Cc", "Cf"} and char not in {"\n", "\t"})
    ]
    return "\n".join(line.rstrip() for line in "".join(kept).splitlines()).strip()


def fetch_shard(local_root: Path) -> Path:
    return Path(
        hf_hub_download(
            "codeparrot/github-code-clean",
            SHARD_FILE,
            repo_type="dataset",
            local_dir=str(local_root),
        )
    )


def rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_pass(path: Path, encoder) -> dict:
    total_tokens = total_rows = 0
    permissive_tokens = permissive_rows = 0
    license_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    permissive_language_counts: Counter[str] = Counter()
    for row in rows(path):
        clean = normalize(str(row.get("code") or ""))
        token_count = len(encoder.encode(clean, disallowed_special=()))
        total_tokens += token_count
        total_rows += 1
        license_counts[row.get("license") or "unknown"] += 1
        language_counts[row.get("language") or "unknown"] += 1
        if row.get("license") in PERMISSIVE_LICENSES:
            permissive_tokens += token_count
            permissive_rows += 1
            permissive_language_counts[row.get("language") or "unknown"] += 1
    return {
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "permissive_rows": permissive_rows,
        "permissive_tokens": permissive_tokens,
        "license_counts": dict(license_counts.most_common()),
        "language_counts": dict(language_counts.most_common(10)),
        "permissive_language_counts": dict(permissive_language_counts.most_common(10)),
    }


def sample_permissive(path: Path, encoder, target: int, available_permissive_tokens: int) -> dict:
    probability = min(1.0, target * 1.08 / max(available_permissive_tokens, 1))
    cutoff = int(probability * (2**64 - 1))
    candidates = []
    lowest = None
    for row in rows(path):
        if row.get("license") not in PERMISSIVE_LICENSES:
            continue
        clean = normalize(str(row.get("code") or ""))
        digest = hashlib.sha256(
            f"github-code:{row.get('repo_name')}:{row.get('path')}:{clean}".encode()
        ).digest()
        rank = int.from_bytes(digest[:8], "big")
        item = (rank, clean, digest.hex(), row.get("language") or "unknown", row.get("license"))
        if lowest is None or rank < lowest[0]:
            lowest = item
        if rank <= cutoff:
            candidates.append(item)
    if not candidates and lowest is not None:
        candidates.append(lowest)
    candidates.sort(key=lambda item: item[0])

    selected_tokens = 0
    selected = []
    for item in candidates:
        count = len(encoder.encode(item[1], disallowed_special=()))
        if selected and selected_tokens + count > target:
            continue
        selected.append((*item, count))
        selected_tokens += count
        if selected_tokens >= target:
            break

    exact_hashes: Counter[str] = Counter()
    emails = secret_like = 0
    manifest_hash = hashlib.sha256()
    selected_language_counts: Counter[str] = Counter()
    for _, clean, digest, language, license_name, token_count in selected:
        exact_hashes[hashlib.sha256(clean.encode()).hexdigest()] += 1
        emails += len(EMAIL.findall(clean))
        secret_like += len(SECRET_LIKE.findall(clean))
        selected_language_counts[language] += 1
        manifest_hash.update(f"{digest}:{token_count}\n".encode())

    return {
        "selected_tokens": selected_tokens,
        "selected_rows": len(selected),
        "selected_language_counts": dict(selected_language_counts.most_common()),
        "exact_duplicate_rows": sum(c - 1 for c in exact_hashes.values() if c > 1),
        "pii_or_secret_hits": {"email": emails, "secret_like_assignment": secret_like},
        "sample_manifest_sha256": manifest_hash.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/github-code"))
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/code-audit.json"))
    args = parser.parse_args()

    encoder = tiktoken.get_encoding(TOKENIZER)
    path = fetch_shard(args.data_dir)
    shard_stats = first_pass(path, encoder)
    sample_stats = sample_permissive(
        path, encoder, args.target_tokens, shard_stats["permissive_tokens"]
    )
    permissive_fraction = shard_stats["permissive_tokens"] / max(shard_stats["total_tokens"], 1)
    permissive_pct = round(permissive_fraction * 100, 1)
    extrapolation = {
        "method": (
            f"shard_stats value x {TOTAL_SHARDS} (total shards in "
            "codeparrot/github-code-clean's data/ directory), assuming roughly "
            "uniform rows-per-shard -- a single-shard order-of-magnitude "
            "estimate, not a measurement."
        ),
        "total_shards": TOTAL_SHARDS,
        "estimated_total_rows": shard_stats["total_rows"] * TOTAL_SHARDS,
        "estimated_total_tokens": shard_stats["total_tokens"] * TOTAL_SHARDS,
        "estimated_permissive_tokens": shard_stats["permissive_tokens"] * TOTAL_SHARDS,
        "consistency_check": (
            "Estimated total rows can be cross-checked against the dataset "
            "card's own arithmetic: codeparrot/github-code cites 115M files, "
            "github-code-clean's README states 3.39M files (2.94%) were "
            "removed by quality filtering, i.e. ~111.6M files expected -- "
            "compare against estimated_total_rows above."
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured-spot-sample",
        "purpose": (
            "Gives the Code lane (D36's zero-supply admission) its first real, "
            "measured data point: a single-shard spot-sample of "
            "codeparrot/github-code-clean, restricted to permissively licensed "
            "files, with the shard's overall permissive-license fraction "
            "reported for extrapolation."
        ),
        "dataset": "codeparrot/github-code-clean",
        "upstream_source": "codeparrot/github-code (BigQuery GitHub public dataset export, query executed 2022-03-16), quality-filtered by CodeParrot",
        "license_note": (
            "Each row carries the license of its origin GitHub repository "
            "(dataset's own 'license' column). This audit reports figures "
            "for the permissive allowlist only "
            f"({sorted(PERMISSIVE_LICENSES)}); copyleft licenses "
            "(gpl-2.0/3.0, agpl-3.0, lgpl-2.1/3.0) and ambiguous ones "
            "(mpl-2.0, epl-1.0, artistic-2.0) are measured but excluded from "
            "the sampled/selected figures below."
        ),
        "scope": (
            "Single shard (train-00000-of-00880.parquet) out of 883 -- NOT the "
            "full corpus. This shard's license mix is reported as the basis "
            "for any full-corpus extrapolation; do not assume it is "
            "representative without a wider multi-shard sample."
        ),
        "tokenizer": TOKENIZER,
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
            "tiktoken": tiktoken.__version__,
        },
        "target_tokens": args.target_tokens,
        "shard_file": SHARD_FILE,
        "shard_sha256": file_sha256(path),
        "shard_stats": shard_stats,
        "permissive_fraction_of_shard_tokens": round(permissive_fraction, 4),
        "full_corpus_extrapolation": extrapolation,
        "sample": sample_stats,
        "caveats": [
            f"This is a single-shard spot-sample, not a full-corpus audit -- do "
            f"not extrapolate this shard's {permissive_pct}% permissive-token "
            f"fraction to the full ~847GB corpus without a wider sample across "
            f"languages and shard-export dates.",
            "License is inherited from the origin repository at BigQuery "
            "export time (2022); it is not re-verified per-file and may be "
            "stale relative to the repository's current license.",
            "Secret/PII scrubbing here is a coarse regex heuristic, not a "
            "production-grade secret scanner -- flagged as 'audit' status, "
            "same as every other S5 lane's pii stage.",
            "Tokenized with cl100k_base for cross-lane audit comparability; "
            "a real Code-lane pretraining run would use a code-aware "
            "tokenizer (e.g. StarCoder's BPE), not this one.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "shard_total_tokens": shard_stats["total_tokens"],
        "permissive_fraction": round(permissive_fraction, 4),
        "selected_tokens": sample_stats["selected_tokens"],
        "estimated_permissive_tokens_full_corpus": extrapolation["estimated_permissive_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
