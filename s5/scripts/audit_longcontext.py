#!/usr/bin/env python3
"""Audit the long-context lane candidate: Yukang/LongAlpaca-12k.

IMPORTANT licence note: the LongLoRA project's *code* is Apache-2.0, but the
*data* (this dataset) is declared CC BY-NC 4.0 (non-commercial) in the
project's own README/DATA_LICENSE. This is flagged explicitly rather than
carried forward as "Apache-2.0" -- a real foundation-model pretraining run
would need either a licence carve-out/negotiation or a commercially licensed
substitute; for this assignment's audit/spot-check purpose the non-commercial
data licence is acceptable but must not be misrepresented.

Adds one extra stage beyond the shared 9-stage template: a context-length
histogram, since this lane's entire point is document length, not just token
volume.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download

from common_audit import run_text_audit

DATASET = "Yukang/LongAlpaca-12k"
FILENAME = "LongAlpaca-12k.json"
LICENSE = "CC BY-NC 4.0 (data; code is Apache-2.0) -- non-commercial, flagged"
TOKENIZER = "cl100k_base"
HISTOGRAM_BUCKETS = [0, 2_000, 8_000, 16_000, 32_000, 64_000, 1_000_000]


def bucket_label(n: int) -> str:
    for lower, upper in zip(HISTOGRAM_BUCKETS, HISTOGRAM_BUCKETS[1:]):
        if lower <= n < upper:
            return f"{lower}-{upper}"
    return f"{HISTOGRAM_BUCKETS[-1]}+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/longalpaca"))
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/longcontext-audit.json"))
    args = parser.parse_args()

    path = Path(
        hf_hub_download(DATASET, FILENAME, repo_type="dataset", local_dir=str(args.data_dir))
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    texts = [f"{row.get('instruction', '')}\n{row.get('output', '')}" for row in rows]

    encoder = tiktoken.get_encoding(TOKENIZER)
    histogram: Counter[str] = Counter()
    for text in texts:
        histogram[bucket_label(len(encoder.encode(text, disallowed_special=())))] += 1

    result = run_text_audit(
        dataset=DATASET,
        revision="main",
        license_name=LICENSE,
        source_path=path,
        tokenizer_name=TOKENIZER,
        encoder=encoder,
        record_texts=texts,
        expected_language="English",
        extra={
            "role": "long-context lane candidate (D8 staged-extension precedent)",
            "context_length_histogram_tokens": dict(sorted(histogram.items(), key=lambda kv: HISTOGRAM_BUCKETS.index(int(kv[0].split("-")[0].rstrip("+"))))),
            "licence_warning": "Data licence is CC BY-NC 4.0 (non-commercial); code licence (Apache-2.0) does not extend to the data. A real production pretraining run would need a licence carve-out or a commercially licensed substitute.",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["raw"], "histogram": dict(histogram)}, indent=2))


if __name__ == "__main__":
    main()
