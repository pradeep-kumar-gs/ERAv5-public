"""Per-lane packing policies.

Two policies:
  - "concat_window" (prose): a single contiguous window sliced straight out
    of the lane's shard, ignoring document boundaries. Standard causal mask,
    sequential position ids. This is the baseline every other policy is
    compared against.
  - "doc_isolated" (code, agent): a packed sequence built by concatenating
    whole (or truncated-to-fit) documents. Attention is block-diagonal --
    positions may only attend within their own document -- and position ids
    reset to 0 at each document's start, so packing multiple short documents
    into one sequence never leaks cross-document context. For `agent`, the
    loss mask additionally zeroes every position whose *target* token
    belongs to a system/user turn, so loss is only charged on assistant
    output.

The critical design point for replay: segment *selection* (which docs, what
slice) is the only place randomness enters. Once a list of Segments exists,
`assemble()` is a pure function of (lane, segments, shard tokens, manifest).
Live training calls choose-then-assemble; replay calls assemble() directly
on segments pulled back out of the consumption ledger -- same function, same
result, which is what makes the replay hash-match provable rather than
asserted.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np

import config
from manifest import DocSpan, ShardManifest

LANE_POLICY = {"prose": "concat_window", "code": "doc_isolated", "agent": "doc_isolated"}


@dataclass
class Segment:
    shard_id: str
    doc_id: str
    token_start: int  # absolute offset in the shard's flat token array
    token_end: int     # exclusive
    seq_start: int      # offset within the assembled (block_size+1)-length window
    seq_end: int         # exclusive

    def to_dict(self) -> dict:
        return dict(shard_id=self.shard_id, doc_id=self.doc_id,
                    token_start=self.token_start, token_end=self.token_end,
                    seq_start=self.seq_start, seq_end=self.seq_end)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**d)


@dataclass
class PackedSequence:
    lane: str
    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray      # (block_size,) 1 = loss-bearing
    position_ids: np.ndarray   # (block_size,)
    seg_ids: np.ndarray        # (block_size,) which segment each position belongs to
    segments: list[Segment] = field(default_factory=list)

    def attention_mask(self) -> np.ndarray:
        t = len(self.seg_ids)
        causal = np.tril(np.ones((t, t), dtype=bool))
        same_seg = self.seg_ids[:, None] == self.seg_ids[None, :]
        return causal & same_seg

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.input_ids.astype(np.int32).tobytes())
        h.update(self.target_ids.astype(np.int32).tobytes())
        h.update(self.loss_mask.astype(np.int8).tobytes())
        h.update(self.position_ids.astype(np.int32).tobytes())
        return h.hexdigest()


def _role_at(doc: DocSpan, abs_pos: int) -> str | None:
    if not doc.turns:
        return None
    offset = abs_pos - doc.start
    for turn in doc.turns:
        if turn.start <= offset < turn.end:
            return turn.role
    return None


def choose_segments_window(manifest: ShardManifest, rng: random.Random, block_size: int) -> list[Segment]:
    needed = block_size + 1
    token_count = manifest.token_count
    if token_count < needed:
        raise ValueError(f"shard {manifest.shard_id} too small to pack ({token_count} < {needed})")
    start = rng.randrange(0, token_count - needed + 1)
    return [Segment(shard_id=manifest.shard_id, doc_id="(window)",
                     token_start=start, token_end=start + needed, seq_start=0, seq_end=needed)]


def choose_segments_isolated(manifest: ShardManifest, rng: random.Random, block_size: int) -> list[Segment]:
    needed = block_size + 1
    docs = manifest.documents
    segments: list[Segment] = []
    seq_pos = 0
    doc_idx = rng.randrange(len(docs))
    guard = 0
    while seq_pos < needed:
        doc = docs[doc_idx % len(docs)]
        doc_len = doc.end - doc.start
        take = min(doc_len, needed - seq_pos)
        segments.append(Segment(shard_id=manifest.shard_id, doc_id=doc.doc_id,
                                 token_start=doc.start, token_end=doc.start + take,
                                 seq_start=seq_pos, seq_end=seq_pos + take))
        seq_pos += take
        doc_idx += 1
        guard += 1
        if guard > 10_000:
            raise RuntimeError(f"shard {manifest.shard_id} could not be packed to {needed} tokens")
    return segments


def choose_segments(lane: str, manifest: ShardManifest, rng: random.Random, block_size: int) -> list[Segment]:
    policy = LANE_POLICY[lane]
    if policy == "concat_window":
        return choose_segments_window(manifest, rng, block_size)
    return choose_segments_isolated(manifest, rng, block_size)


def assemble(lane: str, segments: list[Segment], shard_tokens: np.ndarray,
             manifest: ShardManifest, block_size: int) -> PackedSequence:
    """Pure: same (lane, segments, shard_tokens, manifest) always yields the
    same PackedSequence. Used by both live packing and replay reconstruction.
    """
    needed = block_size + 1
    ids = np.empty(needed, dtype=np.int32)
    seg_ids_full = np.empty(needed, dtype=np.int32)
    roles_full: list[str | None] = [None] * needed
    docs_by_id = {d.doc_id: d for d in manifest.documents}

    for idx, seg in enumerate(segments):
        length = seg.token_end - seg.token_start
        ids[seg.seq_start:seg.seq_end] = shard_tokens[seg.token_start:seg.token_end]
        seg_ids_full[seg.seq_start:seg.seq_end] = idx
        doc = docs_by_id.get(seg.doc_id)
        if doc is not None and doc.turns:
            for pos in range(seg.token_start, seg.token_end):
                roles_full[seg.seq_start + (pos - seg.token_start)] = _role_at(doc, pos)

    if ids.shape[0] != needed:
        raise RuntimeError("assembled window does not match block_size+1")

    input_ids = ids[:block_size]
    target_ids = ids[1:block_size + 1]
    position_ids = np.zeros(block_size, dtype=np.int32)
    for idx in range(len(segments)):
        mask = seg_ids_full[:block_size] == idx
        count = int(mask.sum())
        if count:
            position_ids[mask] = np.arange(count, dtype=np.int32)

    if any(r is not None for r in roles_full):
        loss_mask = np.array(
            [1 if roles_full[i + 1] == "assistant" else 0 for i in range(block_size)], dtype=np.int8
        )
    else:
        loss_mask = np.ones(block_size, dtype=np.int8)

    return PackedSequence(
        lane=lane, input_ids=input_ids, target_ids=target_ids, loss_mask=loss_mask,
        position_ids=position_ids, seg_ids=seg_ids_full[:block_size], segments=segments,
    )


def pack_sequence(lane: str, manifest: ShardManifest, shard_tokens: np.ndarray,
                   rng: random.Random, block_size: int = config.BLOCK_SIZE) -> PackedSequence:
    segments = choose_segments(lane, manifest, rng, block_size)
    return assemble(lane, segments, shard_tokens, manifest, block_size)


@dataclass
class Batch:
    batch_id: str
    batch_hash: str
    step: int
    lane: str
    stage: str
    sequences: list[PackedSequence]

    def tokens_consumed(self) -> int:
        return sum(int(s.loss_mask.sum()) for s in self.sequences)

    def total_slots(self) -> int:
        return sum(len(s.input_ids) for s in self.sequences)


def build_batch(step: int, stage: str, lane: str, manifest: ShardManifest, shard_tokens: np.ndarray,
                 rng: random.Random, batch_size: int = config.BATCH_SIZE,
                 block_size: int = config.BLOCK_SIZE) -> Batch:
    sequences = [pack_sequence(lane, manifest, shard_tokens, rng, block_size) for _ in range(batch_size)]
    h = hashlib.sha256()
    for seq in sequences:
        h.update(seq.content_hash().encode())
    batch_hash = h.hexdigest()
    batch_id = f"{lane}-step{step:04d}-{batch_hash[:12]}"
    return Batch(batch_id=batch_id, batch_hash=batch_hash, step=step, lane=lane, stage=stage, sequences=sequences)


def rebuild_batch(step: int, stage: str, lane: str, segments_per_seq: list[list[Segment]],
                   shard_tokens: np.ndarray, manifest: ShardManifest,
                   block_size: int = config.BLOCK_SIZE) -> Batch:
    """Replay entry point: no RNG, segments come from the consumption ledger."""
    sequences = [assemble(lane, segs, shard_tokens, manifest, block_size) for segs in segments_per_seq]
    h = hashlib.sha256()
    for seq in sequences:
        h.update(seq.content_hash().encode())
    batch_hash = h.hexdigest()
    batch_id = f"{lane}-step{step:04d}-{batch_hash[:12]}"
    return Batch(batch_id=batch_id, batch_hash=batch_hash, step=step, lane=lane, stage=stage, sequences=sequences)
