#!/usr/bin/env python3
"""Spot-sample ai4bharat/sangraha's Hindi verified/unverified/synthetic tiers.

This is a SPOT-SAMPLE, not a full-corpus audit: Sangraha's Hindi verified tier
alone spans ~100 parquet shards (~12.6B tokens total per S3's arXiv:2403.06350
citation). Downloading and auditing all of it is out of scope for this
assignment; instead this script downloads shard 0 of each tier (the only
mirrored-repository-layout shard needed to reach the token target) and applies
the same deterministic SHA-256 rank sampling S4 used for Aya, scoped to that
one shard per tier. Every other language's Sangraha numbers stay cited (not
measured) from S3 -- this script measures Hindi only, honestly labeled.

Raw shards are large (~250-380MB each); they are downloaded to
s5/data/raw/sangraha/ (gitignored) and never published.
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
TIERS = {
    "verified": "verified/hin/data-0.parquet",
    "unverified": "unverified/hin/data-0.parquet",
    "synthetic": "synthetic/hin_Deva/wiki_hin_Deva_0000_of_0063.parquet",
}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    kept = [
        char
        for char in text
        if not (
            unicodedata.category(char) in {"Cc", "Cf"}
            and char not in {"\n", "\t", "‌", "‍"}
        )
    ]
    return "\n".join(line.rstrip() for line in "".join(kept).splitlines()).strip()


def fetch_shard(repo_filename: str, local_root: Path) -> Path:
    return Path(
        hf_hub_download(
            "ai4bharat/sangraha",
            repo_filename,
            repo_type="dataset",
            local_dir=str(local_root),
        )
    )


def rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def first_pass(path: Path, encoder) -> tuple[int, int]:
    total_tokens = total_rows = 0
    for row in rows(path):
        clean = normalize(str(row.get("text") or ""))
        total_tokens += len(encoder.encode(clean, disallowed_special=()))
        total_rows += 1
    return total_tokens, total_rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_tier(tier: str, path: Path, encoder, target: int) -> dict:
    available_tokens, available_rows = first_pass(path, encoder)
    probability = min(1.0, target * 1.08 / max(available_tokens, 1))
    cutoff = int(probability * (2**64 - 1))
    candidates = []
    lowest = None
    for row in rows(path):
        clean = normalize(str(row.get("text") or ""))
        digest = hashlib.sha256(f"sangraha:{tier}:hin:{row.get('doc_id')}:{clean}".encode()).digest()
        rank = int.from_bytes(digest[:8], "big")
        item = (rank, clean, digest.hex())
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

    exact_hashes = Counter()
    controls_removed = emails = phones = 0
    manifest_hash = hashlib.sha256()
    for _, clean, digest, token_count in selected:
        exact_hashes[hashlib.sha256(clean.encode()).hexdigest()] += 1
        emails += len(EMAIL.findall(clean))
        phones += len(PHONE.findall(clean))
        manifest_hash.update(f"{digest}:{token_count}\n".encode())

    return {
        "tier": tier,
        "shard_file": str(path.name),
        "shard_available_tokens": available_tokens,
        "shard_available_rows": available_rows,
        "selected_tokens": selected_tokens,
        "selected_rows": len(selected),
        "exact_duplicate_rows": sum(count - 1 for count in exact_hashes.values() if count > 1),
        "pii_hits": {"email": emails, "phone": phones},
        "sample_manifest_sha256": manifest_hash.hexdigest(),
        "shard_sha256": file_sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/sangraha"))
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/sangraha-hindi-spotsample.json"))
    args = parser.parse_args()

    encoder = tiktoken.get_encoding(TOKENIZER)
    results = {}
    for tier, repo_filename in TIERS.items():
        path = fetch_shard(repo_filename, args.data_dir)
        results[tier] = audit_tier(tier, path, encoder, args.target_tokens)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured-spot-sample",
        "purpose": "Spot-samples ai4bharat/sangraha's Hindi verified/unverified/synthetic tiers to give the S5 Indic mixture spec at least one independently measured verified/synthetic data point, since S3 only ever cited Sangraha's headline arXiv numbers.",
        "dataset": "ai4bharat/sangraha",
        "citation": "Khan et al. 2024, arXiv:2403.06350",
        "scope": "Hindi only, shard 0 of each tier only -- NOT the full corpus. Every other language and every other shard remains a CITED (not measured) number from S3, taken from the paper's headline split (64.306B verified / 24.308B unverified / 162.708B synthetic across 22 languages; Hindi verified cited at 12.617B).",
        "tokenizer": TOKENIZER,
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
            "tiktoken": tiktoken.__version__,
        },
        "target_tokens_per_tier": args.target_tokens,
        "tiers": results,
        "caveats": [
            "This is a single-shard spot-sample per tier, not a full-corpus audit -- do not extrapolate shard-level ratios to the full 12.6B-token verified corpus without a wider sample.",
            "Sangraha's 'verified' label reflects the dataset authors' own source-curation methodology (see paper), not independent re-review by this project.",
            "'type' column present in the verified shard was not used for further sub-tiering here; left for future work.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tiers": {k: v["selected_tokens"] for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
