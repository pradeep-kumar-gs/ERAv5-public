from __future__ import annotations

from manifest import verify_shard


def test_all_shards_verify_clean(manifests, corpus_dirs):
    shards_dir, _ = corpus_dirs
    for m in manifests.values():
        ok, detail = verify_shard(m, shards_dir)
        assert ok, detail


def test_verify_shard_detects_content_tamper(manifests, corpus_dirs, tmp_path):
    shards_dir, _ = corpus_dirs
    m = manifests["prose"]
    tampered_dir = tmp_path / "tampered_shards"
    tampered_dir.mkdir()
    original = bytearray((shards_dir / m.path).read_bytes())
    original[0] ^= 0xFF
    (tampered_dir / m.path).write_bytes(bytes(original))

    ok, detail = verify_shard(m, tampered_dir)
    assert not ok
    assert "content hash mismatch" in detail


def test_verify_shard_detects_missing_file(manifests, tmp_path):
    m = manifests["code"]
    ok, detail = verify_shard(m, tmp_path)
    assert not ok
    assert "missing" in detail


def test_verify_shard_detects_tokenizer_hash_drift(manifests, corpus_dirs):
    shards_dir, _ = corpus_dirs
    m = manifests["code"]
    tampered = m.__class__.from_dict({**m.to_dict(), "tokenizer_hash": "not-the-real-hash"})
    ok, detail = verify_shard(tampered, shards_dir)
    assert not ok
    assert "tokenizer hash mismatch" in detail


def test_eval_lane_manifest_role_is_eval(manifests):
    assert manifests["eval"].role == "eval"
    for lane in ("prose", "code", "agent"):
        assert manifests[lane].role == "train"
