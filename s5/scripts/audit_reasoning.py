#!/usr/bin/env python3
"""Audit the reasoning/difficulty-band lane candidate: bespokelabs/Bespoke-Stratos-17k.

Named in S3 as an inventory candidate ("retained as fallback") but never
actually used in S3 or S4 -- this is the first time it's audited. Its
long-chain-of-thought <|begin_of_thought|>...<|end_of_thought|> structure is
exactly what the S5 curriculum's Band 1-3 (short/medium/long CoT) needs real
example rows for, so this script also buckets each record by its
<|begin_of_thought|>...<|end_of_thought|> span length into the same 4 curriculum
bands used in the README, and keeps the shortest/longest examples' token counts
(not raw text) for citation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

from common_audit import run_text_audit

DATASET = "bespokelabs/Bespoke-Stratos-17k"
FILENAME = "data/train-00000-of-00001.parquet"
LICENSE = "Apache-2.0"
TOKENIZER = "cl100k_base"
THOUGHT_SPAN = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.S)

BAND_EDGES = [(0, 100, "Band 0 (direct/no-CoT)"), (100, 500, "Band 1 (short CoT)"),
              (500, 2000, "Band 2 (medium CoT)"), (2000, float("inf"), "Band 3 (long CoT)")]


def band_for(thought_tokens: int) -> str:
    for lower, upper, label in BAND_EDGES:
        if lower <= thought_tokens < upper:
            return label
    return BAND_EDGES[-1][2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/bespoke_stratos"))
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/reasoning-audit.json"))
    args = parser.parse_args()

    path = Path(
        hf_hub_download(DATASET, FILENAME, repo_type="dataset", local_dir=str(args.data_dir))
    )
    rows = pq.read_table(path).to_pylist()

    encoder = tiktoken.get_encoding(TOKENIZER)
    texts = []
    band_counts: Counter[str] = Counter()
    band_token_examples: dict[str, int] = {}
    for row in rows:
        turns = row.get("conversations") or []
        full_text = (row.get("system") or "") + "\n" + "\n".join(
            str(turn.get("value") or "") for turn in turns
        )
        texts.append(full_text)

        thought_texts = []
        for turn in turns:
            thought_texts.extend(THOUGHT_SPAN.findall(str(turn.get("value") or "")))
        thought_tokens = len(encoder.encode("\n".join(thought_texts), disallowed_special=())) if thought_texts else 0
        band = band_for(thought_tokens)
        band_counts[band] += 1
        if band not in band_token_examples or thought_tokens < band_token_examples[band]:
            band_token_examples[band] = thought_tokens

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
            "role": "reasoning/difficulty-band lane candidate (S3 fallback, first used here)",
            "curriculum_band_distribution_by_thought_tokens": dict(band_counts),
            "curriculum_band_representative_thought_token_count": band_token_examples,
            "note": "Bands measured on <|begin_of_thought|>...<|end_of_thought|> span length per record, matching the README's 4-band curriculum definition.",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["raw"], "bands": dict(band_counts)}, indent=2))


if __name__ == "__main__":
    main()
