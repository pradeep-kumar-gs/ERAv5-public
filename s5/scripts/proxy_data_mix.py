#!/usr/bin/env python3
"""Build token pools + three competing mixture streams for the toy-scale proxy run.

Uses ONLY already-audited real data (D31/D36: Code and General have no audited
training corpus in this project, so they are excluded rather than faked):

  - indic       : Aya (5 languages, all tiers, D29) + Sangraha-Hindi spot-sample
                  (verified/unverified/synthetic, D30)
  - agentic     : glaive-function-calling-v2 (D31)
  - reasoning   : Bespoke-Stratos-17k, split by curriculum band (D33) -- Band 3
                  (thought span >=2000 tokens) is withheld entirely from every
                  hypothesis's training stream and reserved for the shared
                  held-out "complex" eval set.
  - longcontext : LongAlpaca-12k, split at 16,000 tokens -- documents >=16000
                  tokens are withheld the same way as reasoning's Band 3.

All three hypotheses train on the SAME per-lane token pools and are evaluated
on the SAME fixed validation set and the SAME fixed held-out "complex" set, so
differences in the results table isolate mixture-ratio and ordering effects,
not data differences. Tokenizer is tiktoken's "gpt2" encoding (vocab 50257),
matching nanoGPT convention -- distinct from the cl100k_base encoder used by
the audit scripts.

Token .bin files are written to s5/data/proxy/ (gitignored, local only, same
convention as s5/data/raw/).
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
S5_RAW = ROOT / "s5" / "data" / "raw"
AYA_DIR = S5_RAW / "aya"
OUT_DIR = ROOT / "s5" / "data" / "proxy"
MANIFEST = ROOT / "s5" / "site" / "data" / "proxy-data-manifest.json"

SEED = 1337
RAW_TOKEN_CAP_PER_LANE = 3_000_000
THOUGHT_SPAN = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.S)
LONGCONTEXT_HELDOUT_TOKENS = 16_000
REASONING_HELDOUT_THOUGHT_TOKENS = 2_000

AYA_LANGUAGES = ["hindi", "tamil", "telugu", "kannada", "malayalam"]

# Renormalized shares among the 4 covered lanes only (Code/General excluded, D36).
# Source point-values before renorm: indic 20 (D6/D19), agentic 2.5 (D11),
# longcontext 0.75 + reasoning 0.75 (the ~1.5% long-context/reasoning micro-slice, D3 report).
HYPOTHESIS_A_WEIGHTS = {"indic": 20 / 24, "agentic": 2.5 / 24, "reasoning": 0.75 / 24, "longcontext": 0.75 / 24}
HYPOTHESIS_B_WEIGHTS = {"indic": 25 / 29, "agentic": 2.5 / 29, "reasoning": 0.75 / 29, "longcontext": 0.75 / 29}
HYPOTHESIS_C_WEIGHTS = HYPOTHESIS_A_WEIGHTS  # same ratios as A; only ordering differs (curriculum vs shuffled)

TRAIN_BUDGET_TOKENS = 900_000
VAL_BUDGET_TOKENS = 100_000
CHUNK_SIZE = 256  # matches model block_size


def encoder():
    return tiktoken.get_encoding("gpt2")


def tokenize_records(enc, texts: list[str], cap: int, rng: random.Random) -> list[list[int]]:
    order = list(range(len(texts)))
    rng.shuffle(order)
    out = []
    total = 0
    for idx in order:
        ids = enc.encode(texts[idx], disallowed_special=())
        if not ids:
            continue
        out.append(ids)
        total += len(ids)
        if total >= cap:
            break
    return out


def load_aya_texts() -> list[str]:
    texts = []
    for lang in AYA_LANGUAGES:
        for path in sorted((AYA_DIR / lang).glob("train-*.parquet")):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096, columns=["inputs", "targets"]):
                for row in batch.to_pylist():
                    text = f"{row.get('inputs') or ''}\n{row.get('targets') or ''}".strip()
                    if text:
                        texts.append(text)
    return texts


def load_sangraha_texts() -> list[str]:
    texts = []
    for rel in (
        "sangraha/verified/hin/data-0.parquet",
        "sangraha/unverified/hin/data-0.parquet",
        "sangraha/synthetic/hin_Deva/wiki_hin_Deva_0000_of_0063.parquet",
    ):
        path = S5_RAW / rel
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=["text"]):
            for row in batch.to_pylist():
                text = str(row.get("text") or "").strip()
                if text:
                    texts.append(text)
    return texts


def load_glaive_texts() -> list[str]:
    rows = json.loads((S5_RAW / "glaive" / "glaive-function-calling-v2.json").read_text(encoding="utf-8"))
    return [f"{r.get('system', '')}\n{r.get('chat', '')}".strip() for r in rows]


def load_longalpaca_records() -> list[tuple[str, int]]:
    """Returns (text, token_length) pairs using the gpt2 encoder for the length gate."""
    enc = encoder()
    rows = json.loads((S5_RAW / "longalpaca" / "LongAlpaca-12k.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        text = f"{r.get('instruction', '')}\n{r.get('output', '')}".strip()
        if text:
            out.append((text, len(enc.encode(text, disallowed_special=()))))
    return out


def load_bespoke_records() -> list[tuple[str, int]]:
    """Returns (text, thought_span_tokens) pairs."""
    enc = encoder()
    path = S5_RAW / "bespoke_stratos" / "data" / "train-00000-of-00001.parquet"
    out = []
    for row in pq.read_table(path).to_pylist():
        turns = row.get("conversations") or []
        full_text = (row.get("system") or "") + "\n" + "\n".join(str(t.get("value") or "") for t in turns)
        thought_texts = []
        for turn in turns:
            thought_texts.extend(THOUGHT_SPAN.findall(str(turn.get("value") or "")))
        thought_tokens = len(enc.encode("\n".join(thought_texts), disallowed_special=())) if thought_texts else 0
        if full_text.strip():
            out.append((full_text.strip(), thought_tokens))
    return out


def split_train_val(token_lists: list[list[int]], val_fraction: float, rng: random.Random):
    order = list(range(len(token_lists)))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * val_fraction)) if order else 0
    val_idx = set(order[:n_val])
    train = [token_lists[i] for i in order if i not in val_idx]
    val = [token_lists[i] for i in order if i in val_idx]
    return train, val


def flatten(token_lists: list[list[int]], cap: int | None = None) -> list[int]:
    out: list[int] = []
    for ids in token_lists:
        out.extend(ids)
        if cap is not None and len(out) >= cap:
            return out[:cap]
    return out


def to_chunks(ids: list[int], chunk_size: int = CHUNK_SIZE) -> list[list[int]]:
    return [ids[i:i + chunk_size] for i in range(0, len(ids) - chunk_size, chunk_size)]


def write_bin(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)


def build_lane_pools(rng: random.Random) -> dict:
    enc = encoder()
    print("Loading + tokenizing indic (Aya + Sangraha-Hindi)...")
    indic_texts = load_aya_texts() + load_sangraha_texts()
    indic_tok = tokenize_records(enc, indic_texts, RAW_TOKEN_CAP_PER_LANE, rng)
    indic_train, indic_val = split_train_val(indic_tok, 0.1, rng)

    print("Loading + tokenizing agentic (glaive-function-calling-v2)...")
    agentic_texts = load_glaive_texts()
    agentic_tok = tokenize_records(enc, agentic_texts, RAW_TOKEN_CAP_PER_LANE, rng)
    agentic_train, agentic_val = split_train_val(agentic_tok, 0.1, rng)

    print("Loading + tokenizing reasoning (Bespoke-Stratos-17k), splitting by curriculum band...")
    bespoke = load_bespoke_records()
    reasoning_main = [enc.encode(t, disallowed_special=()) for t, thought in bespoke if thought < REASONING_HELDOUT_THOUGHT_TOKENS]
    reasoning_band3 = [enc.encode(t, disallowed_special=()) for t, thought in bespoke if thought >= REASONING_HELDOUT_THOUGHT_TOKENS]
    rng.shuffle(reasoning_main)
    reasoning_main = reasoning_main[: max(1, RAW_TOKEN_CAP_PER_LANE // 500)]  # cap by record count, cheap
    reasoning_train, reasoning_val = split_train_val(reasoning_main, 0.1, rng)

    print("Loading + tokenizing long-context (LongAlpaca-12k), splitting at 16k tokens...")
    longalpaca = load_longalpaca_records()
    lc_main = [(t, n) for t, n in longalpaca if n < LONGCONTEXT_HELDOUT_TOKENS]
    lc_heldout = [(t, n) for t, n in longalpaca if n >= LONGCONTEXT_HELDOUT_TOKENS]
    rng.shuffle(lc_main)
    lc_main_tok = [enc.encode(t, disallowed_special=()) for t, _ in lc_main]
    lc_heldout_tok = [enc.encode(t, disallowed_special=()) for t, _ in lc_heldout]
    lc_train, lc_val = split_train_val(lc_main_tok, 0.1, rng)

    heldout_complex = reasoning_band3 + lc_heldout_tok
    rng.shuffle(heldout_complex)

    return {
        "indic": {"train": indic_train, "val": indic_val},
        "agentic": {"train": agentic_train, "val": agentic_val},
        "reasoning": {"train": reasoning_train, "val": reasoning_val},
        "longcontext": {"train": lc_train, "val": lc_val},
        "heldout_complex": heldout_complex,
        "counts": {
            "indic_records": {"train": len(indic_train), "val": len(indic_val)},
            "agentic_records": {"train": len(agentic_train), "val": len(agentic_val)},
            "reasoning_records": {"train": len(reasoning_train), "val": len(reasoning_val), "band3_withheld": len(reasoning_band3)},
            "longcontext_records": {"train": len(lc_train), "val": len(lc_val), "long_withheld": len(lc_heldout_tok)},
        },
    }


def build_hypothesis_train(pools: dict, weights: dict, budget: int, curriculum: bool, rng: random.Random) -> list[int]:
    lane_chunks = {}
    for lane, weight in weights.items():
        lane_budget = int(weight * budget)
        ids = flatten(pools[lane]["train"], cap=lane_budget)
        lane_chunks[lane] = to_chunks(ids)

    if not curriculum:
        all_chunks = []
        for lane in weights:
            for c in lane_chunks[lane]:
                all_chunks.append(c)
        rng.shuffle(all_chunks)
    else:
        stage1 = lane_chunks.get("indic", []) + lane_chunks.get("agentic", [])
        stage2 = lane_chunks.get("reasoning", [])
        stage3 = lane_chunks.get("longcontext", [])
        rng.shuffle(stage1)
        rng.shuffle(stage2)
        rng.shuffle(stage3)
        all_chunks = stage1 + stage2 + stage3

    flat: list[int] = []
    for c in all_chunks:
        flat.extend(c)
    return flat


def build_shared_val(pools: dict, budget: int) -> list[int]:
    flat: list[int] = []
    for lane, weight in HYPOTHESIS_A_WEIGHTS.items():
        flat.extend(flatten(pools[lane]["val"], cap=int(weight * budget)))
    return flat


def main() -> None:
    rng = random.Random(SEED)
    pools = build_lane_pools(rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hypotheses = {
        "A": (HYPOTHESIS_A_WEIGHTS, False),
        "B": (HYPOTHESIS_B_WEIGHTS, False),
        "C": (HYPOTHESIS_C_WEIGHTS, True),
    }
    written = {}
    for name, (weights, curriculum) in hypotheses.items():
        ids = build_hypothesis_train(pools, weights, TRAIN_BUDGET_TOKENS, curriculum, random.Random(SEED))
        path = OUT_DIR / f"train_{name}.bin"
        write_bin(path, ids)
        written[name] = {"path": str(path.relative_to(ROOT)), "tokens": len(ids), "weights": weights, "curriculum_ordered": curriculum}

    val_ids = build_shared_val(pools, VAL_BUDGET_TOKENS)
    write_bin(OUT_DIR / "val_shared.bin", val_ids)

    heldout_ids = flatten(pools["heldout_complex"])
    write_bin(OUT_DIR / "heldout_complex.bin", heldout_ids)

    manifest = {
        "seed": SEED,
        "tokenizer": "gpt2 (tiktoken)",
        "vocab_size": 50257,
        "chunk_size": CHUNK_SIZE,
        "train_budget_tokens": TRAIN_BUDGET_TOKENS,
        "val_budget_tokens": VAL_BUDGET_TOKENS,
        "lane_pool_counts": pools["counts"],
        "hypotheses": written,
        "val_shared_tokens": len(val_ids),
        "heldout_complex_tokens": len(heldout_ids),
        "heldout_complex_composition": "Bespoke-Stratos-17k records with thought-span >=2000 tokens (Band 3, D33) + LongAlpaca-12k documents >=16000 tokens (D31) -- never included in any hypothesis's training stream.",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v["tokens"] for k, v in written.items()} | {"val": len(val_ids), "heldout_complex": len(heldout_ids)}, indent=2))


if __name__ == "__main__":
    main()
