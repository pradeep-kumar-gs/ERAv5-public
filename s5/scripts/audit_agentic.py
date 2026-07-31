#!/usr/bin/env python3
"""Audit the agentic-lane candidate: glaiveai/glaive-function-calling-v2.

Salesforce/xlam-function-calling-60k (the originally planned candidate) turned
out to be a gated repository (401 Unauthorized without an approved HF access
request) -- swapped to this real, ungated, Apache-2.0 alternative rather than
leaving a broken citation. 112,960 system-prompt + multi-turn tool-calling
conversations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download

from common_audit import run_text_audit

DATASET = "glaiveai/glaive-function-calling-v2"
FILENAME = "glaive-function-calling-v2.json"
LICENSE = "Apache-2.0"
TOKENIZER = "cl100k_base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("s5/data/raw/glaive"))
    parser.add_argument("--output", type=Path, default=Path("s5/site/data/agentic-audit.json"))
    args = parser.parse_args()

    path = Path(
        hf_hub_download(DATASET, FILENAME, repo_type="dataset", local_dir=str(args.data_dir))
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    texts = [f"{row.get('system', '')}\n{row.get('chat', '')}" for row in rows]

    encoder = tiktoken.get_encoding(TOKENIZER)
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
            "role": "agentic pretraining/SFT lane candidate (D11)",
            "note": "system field carries the available-function schema; chat field carries the multi-turn tool-call conversation as a single string with role markers.",
            "swap_reason": "Salesforce/xlam-function-calling-60k is gated (401); this is a real, ungated, appropriately licensed substitute.",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["raw"]}, indent=2))


if __name__ == "__main__":
    main()
