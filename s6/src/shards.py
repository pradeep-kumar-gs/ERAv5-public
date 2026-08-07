"""Tokenizes the synthetic corpus into immutable per-lane shards.

One shard file per lane: a single flat int32 token array (`.bin`) plus a
manifest recording every document's token span within it (and, for the
`agent` lane, every turn's span within each document). "Immutable" means:
once written, a shard is only ever read and hash-verified, never edited in
place -- the manifest's sha256 is the contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from corpus import Document, generate_corpus
from manifest import DocSpan, ShardManifest, TurnSpan, sha256_file, write_manifest
from tokenizer import ENCODING_NAME, get_tokenizer

DTYPE = "int32"


def _tokenize_lane(lane: str, docs: list[Document]) -> tuple[np.ndarray, list[DocSpan]]:
    tok = get_tokenizer()
    all_ids: list[int] = []
    doc_spans: list[DocSpan] = []
    cursor = 0
    for doc in docs:
        if doc.turns:
            turn_spans: list[TurnSpan] = []
            doc_start = cursor
            for turn in doc.turns:
                ids = tok.encode(f"[{turn.role}] {turn.text}\n")
                start = cursor
                all_ids.extend(ids)
                cursor += len(ids)
                turn_spans.append(TurnSpan(role=turn.role, start=start - doc_start, end=cursor - doc_start))
            doc_spans.append(DocSpan(doc_id=doc.doc_id, start=doc_start, end=cursor, turns=turn_spans))
        else:
            ids = tok.encode(doc.text)
            start = cursor
            all_ids.extend(ids)
            cursor += len(ids)
            doc_spans.append(DocSpan(doc_id=doc.doc_id, start=start, end=cursor))
    return np.array(all_ids, dtype=np.int32), doc_spans


def build_shards(shards_dir: Path, manifests_dir: Path) -> list[ShardManifest]:
    tok = get_tokenizer()
    corpus = generate_corpus()
    manifests: list[ShardManifest] = []
    shards_dir.mkdir(parents=True, exist_ok=True)
    for lane, docs in corpus.items():
        token_ids, doc_spans = _tokenize_lane(lane, docs)
        shard_id = f"shard-{lane}"
        bin_name = f"{shard_id}.bin"
        bin_path = shards_dir / bin_name
        token_ids.tofile(bin_path)
        manifest = ShardManifest(
            shard_id=shard_id,
            lane=lane,
            role="eval" if lane == "eval" else "train",
            path=bin_name,
            sha256=sha256_file(bin_path),
            token_count=int(len(token_ids)),
            dtype=DTYPE,
            tokenizer_name=ENCODING_NAME,
            tokenizer_hash=tok.hash,
            documents=doc_spans,
        )
        write_manifest(manifest, manifests_dir)
        manifests.append(manifest)
    return manifests


def load_shard_tokens(manifest: ShardManifest, shards_dir: Path) -> np.ndarray:
    return np.fromfile(shards_dir / manifest.path, dtype=np.int32)
