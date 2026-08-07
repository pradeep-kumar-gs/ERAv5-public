"""Frozen tokenizer wrapper.

Tokenizer "freezing" here means: pin the encoding name, and record a hash
that changes if the tokenizer's actual encode/decode behavior ever drifts
(e.g. a tiktoken version bump that changes merges). Every manifest stores
this hash; shard/manifest validation re-derives it and compares, which is
what backs the run.log line `[PASS] tokenizer_hash_verified`.
"""
from __future__ import annotations

import hashlib

import tiktoken

ENCODING_NAME = "cl100k_base"
EOS_TOKEN = "<|endoftext|>"

# Fixed canary string: hashing what it encodes to catches any drift in the
# tokenizer's actual behavior, not just its name/version string.
_CANARY = "OPUS governs the mixture; the ledger remembers everything. 42."


class FrozenTokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding(ENCODING_NAME)
        self.eos_id = self._enc.encode(EOS_TOKEN, allowed_special={EOS_TOKEN})[0]
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def hash(self) -> str:
        canary_ids = self._enc.encode(_CANARY)
        payload = f"{ENCODING_NAME}:{self.vocab_size}:{canary_ids}".encode()
        return hashlib.sha256(payload).hexdigest()

    def verify(self, expected_hash: str) -> bool:
        return self.hash == expected_hash


_singleton: FrozenTokenizer | None = None


def get_tokenizer() -> FrozenTokenizer:
    global _singleton
    if _singleton is None:
        _singleton = FrozenTokenizer()
    return _singleton
