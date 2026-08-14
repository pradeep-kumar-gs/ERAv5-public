"""Kronecker Embedding V2: the shipped byte-position codec (Session 7, s7 course
material) plus an appended, frozen "meaning channel" that makes vector arithmetic
on numeral tokens mirror real arithmetic.

Byte codec (V1, reimplemented from the course spec): every token is read as UTF-8
bytes; a 256 (byte value) x pos_dim (byte position) grid is one-hot marked per byte,
flattened to a fixed vector, scaled by 1/sqrt(L), then z-normalised. Nothing in this
codec is learned -- it is a deterministic function of the token string.

Meaning channel (V2, new): if the token parses as a number, append
[sign(n), log(|n|+eps), n/NORM]. This is also deterministic and never trained.
Because addition is exact for the linear term and log-addition is exact for the
magnitude term, vector-adding two numeral codes recovers both the sum (from the
linear channel) and, after exponentiating, the product (from the log channel) --
see algebra_proof.py for the verification. The only trained parameter in either
codec is the single shared Linear projection into d_model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

BYTE_VALUES = 256
POS_DIM = 32
BYTE_CODE_DIM = BYTE_VALUES * POS_DIM  # 8192
MEANING_DIM = 3  # [sign, log(|n|+eps), n/NORM]
NORM = 1000.0
EPS = 1e-6


def byte_kronecker_code(token: str, pos_dim: int = POS_DIM) -> np.ndarray:
    """The course-shipped byte x position Kronecker code, fixed and untrained."""
    byte_seq = token.encode("utf-8")
    length = min(len(byte_seq), pos_dim)
    grid = np.zeros((BYTE_VALUES, pos_dim), dtype=np.float64)
    for p in range(length):
        grid[byte_seq[p], p] = 1.0
    vec = grid.reshape(-1)
    scale = 1.0 / math.sqrt(max(length, 1))
    vec = vec * scale
    std = vec.std()
    if std > 0:
        vec = (vec - vec.mean()) / std
    else:
        vec = vec - vec.mean()
    return vec.astype(np.float32)


def parse_numeral(token: str) -> float | None:
    """Returns the numeric value of a token if it parses as int/float, else None."""
    try:
        return float(int(token))
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return None


def numeral_meaning_code(token: str, norm: float = NORM, eps: float = EPS) -> np.ndarray:
    """[sign, log(|n|+eps), n/norm] for a numeral token; zeros for anything else.

    This is the whole V2 idea: append new, appended dimensions after the existing
    32-position byte slots, one linear and one logarithmic, so that vector addition
    in embedding space mirrors real addition (linear channel) and, after exponentiating,
    real multiplication (log channel). See algebra_proof.py for the exactness proof.
    """
    value = parse_numeral(token)
    if value is None:
        return np.zeros(MEANING_DIM, dtype=np.float32)
    sign = float(np.sign(value))
    log_mag = math.log(abs(value) + eps)
    lin = value / norm
    return np.array([sign, log_mag, lin], dtype=np.float32)


def decode_sum(code_a: np.ndarray, code_b: np.ndarray, norm: float = NORM) -> float:
    """Recover a+b from the linear channel of two meaning codes (exact, no training)."""
    return float((code_a[2] + code_b[2]) * norm)


def decode_diff(code_a: np.ndarray, code_b: np.ndarray, norm: float = NORM) -> float:
    return float((code_a[2] - code_b[2]) * norm)


def decode_product(code_a: np.ndarray, code_b: np.ndarray, eps: float = EPS) -> float:
    """Recover a*b from the log/sign channels of two meaning codes."""
    sign = code_a[0] * code_b[0]
    mag = math.exp(float(code_a[1] + code_b[1])) - eps * eps
    return float(sign * max(mag, 0.0))


def decode_quotient(code_a: np.ndarray, code_b: np.ndarray, eps: float = EPS) -> float:
    """Recover a/b from the log/sign channels of two meaning codes."""
    sign = code_a[0] * code_b[0]
    mag = math.exp(float(code_a[1] - code_b[1]))
    return float(sign * mag)


def full_code(token: str, pos_dim: int = POS_DIM, norm: float = NORM) -> np.ndarray:
    """byte code (8192) concatenated with the appended meaning channel (3) = 8195."""
    return np.concatenate([byte_kronecker_code(token, pos_dim), numeral_meaning_code(token, norm)])


@dataclass
class Vocab:
    tokens: list[str]

    def __post_init__(self) -> None:
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, token: str) -> int:
        return self.token_to_id[token]


def _codes_buffer(vocab: Vocab, code_fn) -> torch.Tensor:
    rows = [code_fn(tok) for tok in vocab.tokens]
    return torch.from_numpy(np.stack(rows, axis=0)).float()


class DenseEmbeddingBaseline(nn.Module):
    """The course's Section 2 dense table: one learned row per token."""

    def __init__(self, vocab: Vocab, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(vocab), d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class KroneckerV1TextOnly(nn.Module):
    """Byte-position codec only (8192-dim), no numeral meaning channel."""

    def __init__(self, vocab: Vocab, d_model: int, pos_dim: int = POS_DIM) -> None:
        super().__init__()
        codes = _codes_buffer(vocab, lambda t: byte_kronecker_code(t, pos_dim))
        self.register_buffer("codes", codes, persistent=False)
        self.proj = nn.Linear(codes.shape[1], d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes[ids])


class KroneckerEmbeddingV2(nn.Module):
    """Byte-position codec (8192) with the appended arithmetic meaning channel (3).

    The raw meaning channel [sign, log(|n|+eps), n/NORM] is exact (see
    algebra_proof.py) but badly scaled for gradient descent: log(|n|+eps) for
    n=0 is log(eps) ~ -13.8, a single-row outlier next to a z-normalised
    8192-dim byte channel and the other ~400 rows' log values in [0, 6]. That
    is purely an optimization-conditioning problem, not a correctness one --
    exp(log(eps)+log(b)) still correctly decodes to ~0 for a=0*b. Fixed
    (non-trainable, vocab-wide) z-normalization of the meaning columns brings
    it onto a comparable scale to the byte channel, mirroring the
    z-normalization byte_kronecker_code already does for its own channel.
    This is a preprocessing rescale only -- algebra_proof.py verifies the raw
    numeral_meaning_code() output directly and is untouched by it.
    """

    def __init__(self, vocab: Vocab, d_model: int, pos_dim: int = POS_DIM, norm: float = NORM) -> None:
        super().__init__()
        codes = _codes_buffer(vocab, lambda t: full_code(t, pos_dim, norm))
        meaning = codes[:, -MEANING_DIM:]
        mean = meaning.mean(dim=0, keepdim=True)
        std = meaning.std(dim=0, keepdim=True).clamp_min(1e-6)
        codes[:, -MEANING_DIM:] = (meaning - mean) / std
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("meaning_mean", mean.squeeze(0), persistent=False)
        self.register_buffer("meaning_std", std.squeeze(0), persistent=False)
        self.proj = nn.Linear(codes.shape[1], d_model)

    def normalized_full_code(self, token: str, pos_dim: int = POS_DIM, norm: float = NORM) -> torch.Tensor:
        """Same transform baked into self.codes at construction time, exposed
        so callers/tests can reproduce a single row's embedding input from the
        raw token string rather than reaching into the codes buffer."""
        raw = torch.from_numpy(full_code(token, pos_dim, norm)).float()
        raw[-MEANING_DIM:] = (raw[-MEANING_DIM:] - self.meaning_mean) / self.meaning_std
        return raw

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes[ids])
