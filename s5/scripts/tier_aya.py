#!/usr/bin/env python3
"""Tier S4's already-selected Aya sample into verified/unverified/translated.

Reuses S4's exact deterministic two-pass SHA-256 rank sample (same target,
same revision) so this operates on the identical ~10M-token/language sample
S4 already measured, then buckets each selected row by `dataset_name`:

  - "(T)" suffix            -> translated (machine/pipeline-translated from
                                English source corpora, e.g. "Flan-CoT-submix (T)")
  - "Aya-Dataset"            -> verified (human-annotated by native-speaker
                                volunteers; see Singh et al. 2024, "Aya Dataset:
                                An Open-Access Collection for Multilingual
                                Instruction Tuning", arXiv:2402.06619)
  - anything else            -> unverified (native-language web/templated
                                content -- e.g. "Telugu-news-articles",
                                "TamilStories" -- not translated, but no
                                human quality-verification label exists)

No synthetic-labeled rows exist in this Aya split; the synthetic tier is
therefore always 0 here by direct inspection, not by assumption.

Raw records never leave the local data directory; output is aggregate
statistics and hashes only, matching S4's convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow
import tiktoken
from huggingface_hub import hf_hub_download, list_repo_files

AYA_DATASET = "CohereLabs/aya_collection_language_split"
AYA_REVISION = "a3af2fde4b4cb5b2775830b11244a1a20b5f004f"
TOKENIZER = "cl100k_base"
DEFAULT_TARGET = 10_000_000
LANGUAGES = {
    "hin": ("Hindi", "hindi"),
    "tam": ("Tamil", "tamil"),
    "tel": ("Telugu", "telugu"),
    "kan": ("Kannada", "kannada"),
    "mal": ("Malayalam", "malayalam"),
}
VERIFIED_SOURCES = {"Aya-Dataset"}


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


def text_fields(row: dict) -> str:
    return "\n".join(str(row.get(key) or "") for key in ("inputs", "targets")).strip()


def classify_tier(dataset_name: str) -> str:
    name = (dataset_name or "").strip()
    if name in VERIFIED_SOURCES:
        return "verified"
    if name.endswith("(T)"):
        return "translated"
    return "unverified"


def download_language_shards(root: Path, directory: str) -> None:
    remote_files = [
        name
        for name in list_repo_files(AYA_DATASET, repo_type="dataset", revision=AYA_REVISION)
        if name.startswith(f"{directory}/train-") and name.endswith(".parquet")
    ]
    if not remote_files:
        raise SystemExit(f"No Aya train shards found in {AYA_DATASET} for {directory}")
    for name in remote_files:
        hf_hub_download(
            AYA_DATASET,
            name,
            repo_type="dataset",
            revision=AYA_REVISION,
            local_dir=str(root),
        )


def source_files(root: Path, directory: str) -> list[Path]:
    files = sorted((root / directory).glob("train-*.parquet"))
    if not files:
        download_language_shards(root, directory)
        files = sorted((root / directory).glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"No Aya train shards found under {root / directory}")
    return files


def rows(files: list[Path]):
    columns = ["id", "inputs", "targets", "dataset_name", "task_type"]
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()


def first_pass(files: list[Path], encoder) -> tuple[int, int]:
    total_tokens = total_rows = 0
    for row in rows(files):
        clean = normalize(text_fields(row))
        total_tokens += len(encoder.encode(clean, disallowed_special=()))
        total_rows += 1
    return total_tokens, total_rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tier_language(code: str, files: list[Path], encoder, target: int) -> dict:
    name, _ = LANGUAGES[code]
    available_tokens, available_rows = first_pass(files, encoder)
    probability = min(1.0, target * 1.08 / max(available_tokens, 1))
    cutoff = int(probability * (2**64 - 1))
    candidates = []
    lowest = None
    for row in rows(files):
        clean = normalize(text_fields(row))
        digest = hashlib.sha256(f"{AYA_REVISION}:{code}:{row.get('id')}:{clean}".encode()).digest()
        rank = int.from_bytes(digest[:8], "big")
        item = (rank, row, clean, digest.hex())
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
        count = len(encoder.encode(item[2], disallowed_special=()))
        if selected and selected_tokens + count > target:
            continue
        selected.append((*item, count))
        selected_tokens += count
        if selected_tokens >= target:
            break

    tier_tokens = Counter()
    tier_rows = Counter()
    tier_sources = {tier: Counter() for tier in ("verified", "unverified", "translated")}
    manifest_hash = hashlib.sha256()
    for _, row, clean, digest, token_count in selected:
        tier = classify_tier(row.get("dataset_name"))
        tier_tokens[tier] += token_count
        tier_rows[tier] += 1
        tier_sources[tier][str(row.get("dataset_name") or "unknown")] += 1
        manifest_hash.update(f"{digest}:{token_count}:{tier}\n".encode())

    return {
        "code": code,
        "name": name,
        "selected_tokens_total": selected_tokens,
        "selected_rows_total": len(selected),
        "tier_tokens": {
            "verified": tier_tokens["verified"],
            "unverified": tier_tokens["unverified"],
            "translated": tier_tokens["translated"],
            "synthetic": 0,
        },
        "tier_rows": {
            "verified": tier_rows["verified"],
            "unverified": tier_rows["unverified"],
            "translated": tier_rows["translated"],
            "synthetic": 0,
        },
        "tier_top_sources": {
            tier: dict(counter.most_common(5)) for tier, counter in tier_sources.items()
        },
        "sample_manifest_sha256": manifest_hash.hexdigest(),
        "input_files": [{"name": path.name, "sha256": file_sha256(path)} for path in files],
    }


def tier_language_worker(code: str, files: list[Path], target: int) -> dict:
    encoder = tiktoken.get_encoding(TOKENIZER)
    return tier_language(code, files, encoder, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aya-dir", type=Path, default=Path("s5/data/raw/aya"))
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/aya-tiers.json"))
    args = parser.parse_args()
    jobs = [
        (code, source_files(args.aya_dir, directory), args.target_tokens)
        for code, (_, directory) in LANGUAGES.items()
    ]
    try:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(tier_language_worker, *job) for job in jobs]
            results = [future.result() for future in futures]
    except PermissionError:
        results = [tier_language_worker(*job) for job in jobs]

    totals = Counter()
    for item in results:
        for tier, count in item["tier_tokens"].items():
            totals[tier] += count

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured",
        "purpose": "Tiers S4's existing Aya sample into verified/unverified/translated/synthetic for the S5 Indic mixture spec.",
        "dataset": "CohereLabs/aya_collection_language_split",
        "revision": AYA_REVISION,
        "license": "Apache-2.0",
        "tokenizer": TOKENIZER,
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
            "tiktoken": tiktoken.__version__,
        },
        "sampling": "Identical two-pass deterministic SHA-256 rank sample S4 used (same target, same revision) -- this is a re-tiering of S4's sample, not a new sample.",
        "tier_definition": {
            "verified": "dataset_name == 'Aya-Dataset' (human-annotated by native-speaker volunteers, Singh et al. 2024 arXiv:2402.06619)",
            "translated": "dataset_name ends with '(T)' (machine/pipeline-translated from English source corpora)",
            "unverified": "everything else -- native-language web/templated content with no human quality-verification label",
            "synthetic": "no synthetic-labeled rows exist in this Aya split; always 0 here by direct inspection",
        },
        "target_tokens_per_language": args.target_tokens,
        "tier_tokens_total": dict(totals),
        "languages": results,
        "caveats": [
            "This tiers S4's already-selected sample; it does not re-derive available_tokens per tier for the full Aya corpus.",
            "'Verified' here means human-annotated per the Aya paper's own methodology, not independently re-reviewed by this project.",
            "Aya's schema has no synthetic-content label; the synthetic tier for the Indic overlay must come from a different source (see audit_sangraha.py).",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tier_tokens_total": dict(totals)}, indent=2))


if __name__ == "__main__":
    main()
