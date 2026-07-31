#!/usr/bin/env python3
"""Wide, shallow multi-shard scan of codeparrot/github-code-clean, to replace
D38's single-shard extrapolation with a real aggregate measurement.

audit_code.py did a DEEP single-shard spot-sample: full text tokenization,
content-level dedup and PII/secret heuristics, on shard 0 of 883. That gave a
detailed but single-shard estimate of the permissive-license token fraction
(52.8%), extrapolated x883 to ~126.2B permissive tokens corpus-wide.

This script does a WIDE scan instead: it reads only the `license` and `size`
(byte-count) columns -- no text, no tokenization -- across a deterministic
sample of shards spread evenly across the full 883-shard range. This is cheap
enough to cover 10+ shards and directly measures shard-to-shard variance in
the permissive-license fraction, rather than assuming shard 0 is
representative. Token counts are estimated from measured byte totals using a
tokens-per-byte ratio calibrated against audit_code.py's shard-0 measurement
(exact tiktoken counts), not re-tokenized here.

Raw shards are large (~300-350MB each); downloaded to
s5/data/raw/github-code/ (gitignored) and never published.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow
from huggingface_hub import hf_hub_download

TOTAL_SHARDS = 883
DEFAULT_NUM_SAMPLE_SHARDS = 10
SHARD0_MEASURED_TOKENS = 270_601_959  # audit_code.py's exact tiktoken count for shard 0
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "cc0-1.0",
    "unlicense",
}


def shard_indices(n: int) -> list[int]:
    step = TOTAL_SHARDS / n
    return sorted({int(i * step) for i in range(n)})


def shard_filename(index: int) -> str:
    return f"data/train-{index:05d}-of-00880.parquet"


def scan_shard(local_root: Path, index: int) -> dict:
    path = Path(
        hf_hub_download(
            "codeparrot/github-code-clean",
            shard_filename(index),
            repo_type="dataset",
            local_dir=str(local_root),
        )
    )
    parquet = pq.ParquetFile(path)
    total_rows = total_bytes = permissive_rows = permissive_bytes = 0
    for batch in parquet.iter_batches(batch_size=16384, columns=["license", "size"]):
        d = batch.to_pydict()
        for license_name, size in zip(d["license"], d["size"]):
            total_rows += 1
            total_bytes += size
            if license_name in PERMISSIVE_LICENSES:
                permissive_rows += 1
                permissive_bytes += size
    return {
        "shard_index": index,
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "permissive_rows": permissive_rows,
        "permissive_bytes": permissive_bytes,
        "permissive_byte_fraction": round(permissive_bytes / max(total_bytes, 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/github-code"))
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SAMPLE_SHARDS)
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/code-shards-audit.json"))
    args = parser.parse_args()

    indices = shard_indices(args.num_shards)
    per_shard = [scan_shard(args.data_dir, i) for i in indices]

    # Calibrate a tokens-per-byte ratio from audit_code.py's exact shard-0
    # tiktoken measurement, using this script's own byte count for the same
    # shard (shard index 0 is always included since step starts at 0).
    shard0 = next(s for s in per_shard if s["shard_index"] == 0)
    tokens_per_byte = SHARD0_MEASURED_TOKENS / max(shard0["total_bytes"], 1)

    fractions = [s["permissive_byte_fraction"] for s in per_shard]
    total_bytes_sampled = sum(s["total_bytes"] for s in per_shard)
    permissive_bytes_sampled = sum(s["permissive_bytes"] for s in per_shard)
    aggregate_fraction = permissive_bytes_sampled / max(total_bytes_sampled, 1)

    mean_total_bytes_per_shard = total_bytes_sampled / len(per_shard)
    mean_permissive_bytes_per_shard = permissive_bytes_sampled / len(per_shard)
    estimated_total_tokens = mean_total_bytes_per_shard * TOTAL_SHARDS * tokens_per_byte
    estimated_permissive_tokens = mean_permissive_bytes_per_shard * TOTAL_SHARDS * tokens_per_byte

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured-multi-shard-sample",
        "purpose": (
            "Replaces D38's single-shard extrapolation with a real aggregate "
            "measurement across multiple shards of codeparrot/github-code-clean, "
            "to check whether shard 0's 52.8% permissive-license fraction is "
            "representative or an outlier."
        ),
        "dataset": "codeparrot/github-code-clean",
        "method": (
            "Reads only the 'license' and 'size' (byte count) columns -- no "
            "text, no tokenization -- across a deterministic, evenly spaced "
            "sample of shards. Token counts are estimated from measured byte "
            "totals via a tokens-per-byte ratio calibrated against "
            "audit_code.py's exact tiktoken count for shard 0, not "
            "re-tokenized here."
        ),
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
        },
        "total_shards_in_dataset": TOTAL_SHARDS,
        "sampled_shard_indices": indices,
        "tokens_per_byte_calibration": {
            "source": "audit_code.py shard-0 exact tiktoken count",
            "shard0_measured_tokens": SHARD0_MEASURED_TOKENS,
            "shard0_total_bytes_this_scan": shard0["total_bytes"],
            "tokens_per_byte": round(tokens_per_byte, 6),
        },
        "per_shard": per_shard,
        "permissive_byte_fraction_stats": {
            "mean": round(statistics.mean(fractions), 4),
            "stdev": round(statistics.stdev(fractions), 4) if len(fractions) > 1 else 0,
            "min": round(min(fractions), 4),
            "max": round(max(fractions), 4),
        },
        "aggregate_permissive_byte_fraction": round(aggregate_fraction, 4),
        "full_corpus_estimate": {
            "method": (
                f"mean bytes/shard across {len(per_shard)} sampled shards x "
                f"{TOTAL_SHARDS} total shards x calibrated tokens-per-byte ratio"
            ),
            "estimated_total_tokens": int(estimated_total_tokens),
            "estimated_permissive_tokens": int(estimated_permissive_tokens),
        },
        "caveats": [
            "Byte-to-token calibration uses a single shard's ratio (shard 0); "
            "if code-file size/token density varies by language mix across "
            "shards, this ratio may not hold uniformly.",
            f"{len(per_shard)} of {TOTAL_SHARDS} shards sampled (~{round(100*len(per_shard)/TOTAL_SHARDS, 1)}%) "
            "-- a wider sample would further narrow the variance estimate.",
            "License is inherited from the origin repository at BigQuery "
            "export time (2022); not re-verified per-file.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sampled_shards": indices,
        "permissive_fraction_mean": output["permissive_byte_fraction_stats"]["mean"],
        "permissive_fraction_stdev": output["permissive_byte_fraction_stats"]["stdev"],
        "estimated_permissive_tokens_full_corpus": output["full_corpus_estimate"]["estimated_permissive_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
