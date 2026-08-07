from __future__ import annotations

import random

import numpy as np

import config
import packing


def test_prose_window_is_plain_causal_with_full_loss_mask(manifests, shard_tokens):
    rng = random.Random(1)
    batch = packing.build_batch(1, "warmup", "prose", manifests["prose"], shard_tokens["prose"], rng)
    t = config.BLOCK_SIZE
    expected_causal = np.tril(np.ones((t, t), dtype=bool))
    for seq in batch.sequences:
        assert len(seq.segments) == 1
        assert (seq.attention_mask() == expected_causal).all()
        assert list(seq.position_ids) == list(range(t))
        assert (seq.loss_mask == 1).all()


def test_code_doc_isolated_blocks_cross_document_attention(manifests, shard_tokens):
    manifest = manifests["code"]
    rng = random.Random(7)
    segments = packing.choose_segments_isolated(manifest, rng, config.BLOCK_SIZE)
    assert len(segments) >= 2, "code docs are short enough that this seed should span >1 document"
    seq = packing.assemble("code", segments, shard_tokens["code"], manifest, config.BLOCK_SIZE)

    mask = seq.attention_mask()
    first_seg_positions = [i for i, sid in enumerate(seq.seg_ids) if sid == 0]
    second_seg_positions = [i for i, sid in enumerate(seq.seg_ids) if sid == 1]
    i, j = first_seg_positions[0], second_seg_positions[0]
    assert not mask[j, i], "a token in the second document must not attend into the first document"
    assert not mask[i, j], "a token in the first document must not attend into the second document"

    # position ids reset to 0 at the start of every document segment
    for seg in segments:
        if seg.seq_start < config.BLOCK_SIZE:
            assert seq.position_ids[seg.seq_start] == 0


def test_agent_loss_mask_only_on_assistant_targets(manifests, shard_tokens):
    manifest = manifests["agent"]
    rng = random.Random(3)
    segments = packing.choose_segments_isolated(manifest, rng, config.BLOCK_SIZE)
    seq = packing.assemble("agent", segments, shard_tokens["agent"], manifest, config.BLOCK_SIZE)

    docs_by_id = {d.doc_id: d for d in manifest.documents}
    for seg in segments:
        doc = docs_by_id[seg.doc_id]
        for abs_pos in range(seg.token_start, seg.token_end):
            role = packing._role_at(doc, abs_pos)
            seq_pos = seg.seq_start + (abs_pos - seg.token_start)
            target_idx = seq_pos - 1  # loss_mask[k] gates target_ids[k] == ids[k+1]
            if 0 <= target_idx < config.BLOCK_SIZE:
                expected = 1 if role == "assistant" else 0
                assert seq.loss_mask[target_idx] == expected

    assert seq.loss_mask.sum() > 0, "at least one assistant token should be loss-bearing"
    assert seq.loss_mask.sum() < config.BLOCK_SIZE, "system/user tokens must not be loss-bearing"


def test_assemble_is_pure_and_replay_matches_live_pack(manifests, shard_tokens):
    manifest = manifests["code"]
    rng = random.Random(11)
    live = packing.build_batch(5, "mid", "code", manifest, shard_tokens["code"], rng)

    segments_per_seq = [[seg for seg in s.segments] for s in live.sequences]
    replayed_sequences = [
        packing.assemble("code", segs, shard_tokens["code"], manifest, config.BLOCK_SIZE)
        for segs in segments_per_seq
    ]
    for live_seq, replayed_seq in zip(live.sequences, replayed_sequences):
        assert live_seq.content_hash() == replayed_seq.content_hash()


def test_rebuild_batch_matches_original_hash(manifests, shard_tokens):
    manifest = manifests["agent"]
    rng = random.Random(42)
    live = packing.build_batch(9, "mid", "agent", manifest, shard_tokens["agent"], rng)

    segments_per_seq = [[seg.to_dict() for seg in s.segments] for s in live.sequences]
    segments_per_seq = [[packing.Segment.from_dict(d) for d in segs] for segs in segments_per_seq]
    rebuilt = packing.rebuild_batch(9, "mid", "agent", segments_per_seq, shard_tokens["agent"], manifest)

    assert rebuilt.batch_hash == live.batch_hash
    assert rebuilt.batch_id == live.batch_id
