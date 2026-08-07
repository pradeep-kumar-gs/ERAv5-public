"""Shard manifest schema + hash verification.

A manifest is the immutability contract for a shard: it pins the shard's
content hash and the tokenizer hash that produced it. `verify_shard`
re-derives both from the actual files on disk and the live tokenizer, so a
tampered shard or a drifted tokenizer is caught, not assumed away.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tokenizer import get_tokenizer


@dataclass
class TurnSpan:
    role: str
    start: int  # token offset within the document (inclusive)
    end: int    # exclusive


@dataclass
class DocSpan:
    doc_id: str
    start: int  # token offset within the shard (inclusive)
    end: int    # exclusive
    turns: list[TurnSpan] = field(default_factory=list)


@dataclass
class ShardManifest:
    shard_id: str
    lane: str
    role: str  # "train" or "eval" -- eval shards must never reach a training batch
    path: str  # filename of the .bin file, relative to the manifests dir's shards/ sibling
    sha256: str
    token_count: int
    dtype: str
    tokenizer_name: str
    tokenizer_hash: str
    documents: list[DocSpan]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardManifest":
        docs = [
            DocSpan(
                doc_id=doc["doc_id"],
                start=doc["start"],
                end=doc["end"],
                turns=[TurnSpan(**t) for t in doc.get("turns", [])],
            )
            for doc in d["documents"]
        ]
        return cls(
            shard_id=d["shard_id"], lane=d["lane"], role=d["role"], path=d["path"],
            sha256=d["sha256"], token_count=d["token_count"], dtype=d["dtype"],
            tokenizer_name=d["tokenizer_name"], tokenizer_hash=d["tokenizer_hash"],
            documents=docs,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_manifest(manifest: ShardManifest, manifests_dir: Path) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out = manifests_dir / f"{manifest.shard_id}.json"
    out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return out


def read_manifest(path: Path) -> ShardManifest:
    return ShardManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_all_manifests(manifests_dir: Path) -> list[ShardManifest]:
    # "shard-*.json" only -- manifests_dir also holds mixture_schedule.json,
    # which is not a ShardManifest.
    return [read_manifest(p) for p in sorted(manifests_dir.glob("shard-*.json"))]


def verify_shard(manifest: ShardManifest, shards_dir: Path) -> tuple[bool, str]:
    shard_path = shards_dir / manifest.path
    if not shard_path.exists():
        return False, f"shard file missing: {shard_path}"
    actual_hash = sha256_file(shard_path)
    if actual_hash != manifest.sha256:
        return False, f"content hash mismatch: expected {manifest.sha256}, got {actual_hash}"
    tok = get_tokenizer()
    if tok.hash != manifest.tokenizer_hash:
        return False, f"tokenizer hash mismatch: expected {manifest.tokenizer_hash}, got {tok.hash}"
    return True, "ok"
