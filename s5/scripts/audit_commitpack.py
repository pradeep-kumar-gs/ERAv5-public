#!/usr/bin/env python3
"""Spot-sample bigcode/commitpack as a SECOND, independent Code-lane source,
to start narrowing the ~464B-token gap D38/D39 left after auditing
codeparrot/github-code-clean alone (still only ~22.6% of the 600B budget).

CommitPack (BigCode, ungated, license:mit on the dataset card itself) is a
~3.8TB corpus of git commits across 350 languages, each row carrying the
*origin repository's* license in a `license` column -- the same per-file
license-provenance approach github-code-clean uses, so the same permissive
allowlist applies directly.

This is a SPOT-SAMPLE, not a full-corpus audit, restricted to one shard file
each for 5 mainstream languages (python, javascript, java, c, go) out of 350
language directories and up to ~516 shard files per language. Every other
language/shard's numbers stay cited (BigCode's own dataset-card total), not
measured.

Because CommitPack draws from the same public-GitHub universe as
github-code-clean, this script also runs a real (not assumed) overlap check:
it hashes the permissive rows already cached locally from D38/D39's
github-code-clean shard 0 and reports how many of this script's sampled
CommitPack rows are exact content matches -- a lower-bound overlap signal,
not proof of no overlap at full-corpus scale.

Raw shards are large (~2.6GB combined); downloaded to
s5/data/raw/commitpack/ (gitignored) and never published.
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
import tiktoken
from huggingface_hub import hf_hub_download

TOKENIZER = "cl100k_base"
DEFAULT_TARGET = 2_000_000
REPO_ID = "bigcode/commitpack"
REVISION = "5eee2c845bf88dbffcafedb6e80d2a72a43fe575"  # pinned, checked via HfApi
TARGET_LANGUAGES = ["python", "javascript", "java", "c", "go"]
SHARD_TEMPLATE = "data/{lang}/{lang}-0001.jsonl"

# Full data/<lang>/ directory sizes in bytes, from HfApi dataset_info(files_metadata=True)
# on REVISION above -- used only to extrapolate within these 5 sampled languages,
# not the full 350-language corpus.
LANGUAGE_DIR_BYTES = {
    "python": 239_930_000_000,
    "javascript": 269_130_000_000,
    "java": 130_150_000_000,
    "c": 205_700_000_000,
    "go": 88_520_000_000,
}

# Same allowlist as audit_code.py (D38): permissive, redistribution-friendly
# licenses only -- excludes copyleft (gpl-*, agpl-3.0, lgpl-*) and
# weak-copyleft/ambiguous (mpl-2.0, epl-1.0, artistic-2.0) licenses.
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


def fetch_shards(local_root: Path) -> dict[str, Path]:
    paths = {}
    for lang in TARGET_LANGUAGES:
        rel = SHARD_TEMPLATE.format(lang=lang)
        paths[lang] = Path(
            hf_hub_download(
                REPO_ID,
                rel,
                repo_type="dataset",
                revision=REVISION,
                local_dir=str(local_root),
            )
        )
    return paths


def rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_hashes(reference_shard: Path) -> set[str]:
    """SHA-256 of normalized code text for every permissive row in the D38/D39
    github-code-clean shard 0 already cached locally -- used only as a real,
    if partial (1-of-883-shard), cross-source overlap check below."""
    hashes: set[str] = set()
    if not reference_shard.exists():
        return hashes
    parquet = pq.ParquetFile(reference_shard)
    for batch in parquet.iter_batches(batch_size=4096, columns=["code", "license"]):
        for row in batch.to_pylist():
            if row.get("license") not in PERMISSIVE_LICENSES:
                continue
            clean = normalize(str(row.get("code") or ""))
            hashes.add(hashlib.sha256(clean.encode()).hexdigest())
    return hashes


def first_pass(paths: dict[str, Path], encoder) -> dict:
    total_tokens = total_rows = 0
    permissive_tokens = permissive_rows = 0
    license_counts: Counter[str] = Counter()
    per_language: dict[str, dict] = {}
    for lang, path in paths.items():
        lang_total_tokens = lang_total_rows = 0
        lang_permissive_tokens = lang_permissive_rows = 0
        for row in rows(path):
            clean = normalize(str(row.get("new_contents") or ""))
            token_count = len(encoder.encode(clean, disallowed_special=()))
            total_tokens += token_count
            total_rows += 1
            lang_total_tokens += token_count
            lang_total_rows += 1
            license_name = row.get("license") or "unknown"
            license_counts[license_name] += 1
            if license_name in PERMISSIVE_LICENSES:
                permissive_tokens += token_count
                permissive_rows += 1
                lang_permissive_tokens += token_count
                lang_permissive_rows += 1
        per_language[lang] = {
            "total_rows": lang_total_rows,
            "total_tokens": lang_total_tokens,
            "permissive_rows": lang_permissive_rows,
            "permissive_tokens": lang_permissive_tokens,
        }
    return {
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "permissive_rows": permissive_rows,
        "permissive_tokens": permissive_tokens,
        "license_counts": dict(license_counts.most_common()),
        "per_language": per_language,
    }


def sample_permissive(
    paths: dict[str, Path], encoder, target: int, available_permissive_tokens: int, reference_hashes: set[str]
) -> dict:
    probability = min(1.0, target * 1.08 / max(available_permissive_tokens, 1))
    cutoff = int(probability * (2**64 - 1))
    candidates = []
    lowest = None
    for lang, path in paths.items():
        for row in rows(path):
            if row.get("license") not in PERMISSIVE_LICENSES:
                continue
            clean = normalize(str(row.get("new_contents") or ""))
            digest = hashlib.sha256(
                f"commitpack:{row.get('commit')}:{row.get('new_file')}:{clean}".encode()
            ).digest()
            rank = int.from_bytes(digest[:8], "big")
            item = (rank, clean, digest.hex(), lang, row.get("license"))
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
    overlap_with_github_code = 0
    manifest_hash = hashlib.sha256()
    selected_language_counts: Counter[str] = Counter()
    for _, clean, digest, lang, license_name, token_count in selected:
        content_hash = hashlib.sha256(clean.encode()).hexdigest()
        exact_hashes[content_hash] += 1
        if content_hash in reference_hashes:
            overlap_with_github_code += 1
        emails += len(EMAIL.findall(clean))
        secret_like += len(SECRET_LIKE.findall(clean))
        selected_language_counts[lang] += 1
        manifest_hash.update(f"{digest}:{token_count}\n".encode())

    return {
        "selected_tokens": selected_tokens,
        "selected_rows": len(selected),
        "selected_language_counts": dict(selected_language_counts.most_common()),
        "exact_duplicate_rows_within_sample": sum(c - 1 for c in exact_hashes.values() if c > 1),
        "exact_overlap_with_github_code_clean_shard0": overlap_with_github_code,
        "pii_or_secret_hits": {"email": emails, "secret_like_assignment": secret_like},
        "sample_manifest_sha256": manifest_hash.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/commitpack"))
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--reference-shard",
        type=Path,
        default=Path("s5/data/raw/github-code/data/train-00000-of-00880.parquet"),
        help="D38/D39's cached github-code-clean shard 0, used for the overlap check",
    )
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/commitpack-audit.json"))
    args = parser.parse_args()

    encoder = tiktoken.get_encoding(TOKENIZER)
    paths = fetch_shards(args.data_dir)
    shard_stats = first_pass(paths, encoder)
    reference_hashes = load_reference_hashes(args.reference_shard)
    sample_stats = sample_permissive(
        paths, encoder, args.target_tokens, shard_stats["permissive_tokens"], reference_hashes
    )
    permissive_fraction = shard_stats["permissive_tokens"] / max(shard_stats["total_tokens"], 1)
    permissive_pct = round(permissive_fraction * 100, 1)

    sampled_dir_bytes = sum(LANGUAGE_DIR_BYTES[lang] for lang in TARGET_LANGUAGES)
    sampled_shard_bytes = sum(path.stat().st_size for path in paths.values())
    tokens_per_byte = shard_stats["total_tokens"] / max(sampled_shard_bytes, 1)
    extrapolation = {
        "method": (
            "tokens_per_byte (measured on the 5 sampled shards) x full "
            "data/<lang>/ directory byte size for those SAME 5 languages only "
            "-- NOT extrapolated to all 350 CommitPack languages."
        ),
        "languages_covered": TARGET_LANGUAGES,
        "sampled_shard_bytes": sampled_shard_bytes,
        "full_dir_bytes_same_languages": sampled_dir_bytes,
        "tokens_per_byte": round(tokens_per_byte, 6),
        "estimated_total_tokens_same_languages": int(sampled_dir_bytes * tokens_per_byte),
        "estimated_permissive_tokens_same_languages": int(
            sampled_dir_bytes * tokens_per_byte * permissive_fraction
        ),
        "not_covered": (
            "345 other language directories (php, typescript, ruby, c#, rust, "
            "and 340 more, plus non-programming-language dirs like json/xml/"
            "markdown/csv) are entirely unsampled -- their tokens are cited "
            "(BigCode's ~3.8TB dataset-card total) not measured, and excluded "
            "from every figure in this file."
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured-spot-sample",
        "purpose": (
            "Second, independent Code-lane source audit (D40), to start "
            "narrowing the gap D38/D39 left after github-code-clean alone "
            "covered ~22.6% of the 600B-token Code-lane budget. Also runs a "
            "real cross-source overlap check against D38/D39's cached "
            "github-code-clean shard 0, since both datasets draw from the "
            "same public-GitHub universe."
        ),
        "dataset": REPO_ID,
        "dataset_revision": REVISION,
        "upstream_source": (
            "BigCode's CommitPack: git commit history mined across public "
            "GitHub repositories, 350 languages, ~3.8TB raw. Each row is one "
            "commit's before/after file contents plus commit message."
        ),
        "license_note": (
            "Each row carries the license of its origin GitHub repository "
            "(dataset's own 'license' column), same provenance approach as "
            "github-code-clean (D38). This audit reports figures for the "
            "permissive allowlist only "
            f"({sorted(PERMISSIVE_LICENSES)}); copyleft and ambiguous "
            "licenses are measured but excluded from sampled/selected "
            "figures below."
        ),
        "scope": (
            "One shard file each for 5 of 350 language directories "
            f"({TARGET_LANGUAGES}) -- NOT the full corpus. Chosen as "
            "mainstream, high-volume programming languages; excludes "
            "markup/data/doc-heavy directories (json, xml, text, markdown, "
            "csv, yaml, html) that dominate CommitPack's raw byte count but "
            "are not 'software engineering code' by this project's Code-lane "
            "definition."
        ),
        "tokenizer": TOKENIZER,
        "runtime": {
            "python": platform.python_version(),
            "tiktoken": tiktoken.__version__,
        },
        "target_tokens": args.target_tokens,
        "shard_files": {lang: SHARD_TEMPLATE.format(lang=lang) for lang in TARGET_LANGUAGES},
        "shard_sha256": {lang: file_sha256(path) for lang, path in paths.items()},
        "shard_stats": shard_stats,
        "permissive_fraction_of_shard_tokens": round(permissive_fraction, 4),
        "same_language_extrapolation": extrapolation,
        "sample": sample_stats,
        "cross_source_overlap_check": {
            "method": (
                "SHA-256 of normalized code text for every permissive row in "
                "this script's sample, compared against a hash set built from "
                "every permissive row in D38/D39's already-cached "
                "github-code-clean shard 0 (1 of 883 shards)."
            ),
            "reference_shard": str(args.reference_shard),
            "reference_hash_set_size": len(reference_hashes),
            "exact_overlap_rows_found": sample_stats["exact_overlap_with_github_code_clean_shard0"],
            "caveat": (
                "This checks only 1 of github-code-clean's 883 shards against "
                "this script's own sample -- a lower-bound signal from a tiny "
                "fraction of both corpora's file-level intersection, not proof "
                "of the true overlap at full-corpus scale. Near-zero exact "
                "matches here does NOT mean the two corpora's full-scale "
                "permissive-token estimates are additive without double-"
                "counting risk: CommitPack stores per-commit snapshots (often "
                "many per file, at different points in the same file's "
                "history) of the same public repositories github-code-clean "
                "snapshot once, so file-level (not exact-content-level) "
                "overlap between the two corpora is almost certainly "
                "substantial and unmeasured here."
            ),
        },
        "caveats": [
            f"This is a 5-of-350-language spot-sample, not a full-corpus "
            f"audit -- do not extrapolate this sample's {permissive_pct}% "
            f"permissive-token fraction to CommitPack's other 345 language "
            f"directories without a wider sample.",
            "License is inherited from the origin repository at the time "
            "BigCode mined the commit; it is not re-verified per-file and "
            "may be stale relative to the repository's current license.",
            "new_contents (the post-commit file state) is used as the code "
            "text measured here; old_contents (the pre-commit state) is "
            "present in the dataset but not counted, to avoid trivially "
            "double-counting near-identical before/after pairs.",
            "See cross_source_overlap_check above: this second source is "
            "NOT proven independent of github-code-clean (D38/D39) at "
            "full-corpus scale -- combine the two lanes' estimates as an "
            "upper bound, not a safe sum.",
            "Secret/PII scrubbing here is a coarse regex heuristic, not a "
            "production-grade secret scanner, matching every other S5 lane's "
            "pii stage.",
            "Tokenized with cl100k_base for cross-lane audit comparability; "
            "a real Code-lane pretraining run would use a code-aware "
            "tokenizer, not this one.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "shard_total_tokens": shard_stats["total_tokens"],
        "permissive_fraction": round(permissive_fraction, 4),
        "selected_tokens": sample_stats["selected_tokens"],
        "estimated_permissive_tokens_same_languages": extrapolation["estimated_permissive_tokens_same_languages"],
        "exact_overlap_rows_found": sample_stats["exact_overlap_with_github_code_clean_shard0"],
    }, indent=2))


if __name__ == "__main__":
    main()
