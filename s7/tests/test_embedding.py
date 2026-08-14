from __future__ import annotations

import math

import numpy as np
import torch

from kronecker import (
    BYTE_CODE_DIM,
    EPS,
    MEANING_DIM,
    NORM,
    DenseEmbeddingBaseline,
    KroneckerEmbeddingV2,
    KroneckerV1TextOnly,
    Vocab,
    byte_kronecker_code,
    numeral_meaning_code,
)


def test_byte_code_is_deterministic_and_correctly_shaped():
    a = byte_kronecker_code("137")
    b = byte_kronecker_code("137")
    assert a.shape == (BYTE_CODE_DIM,)
    np.testing.assert_array_equal(a, b)


def test_byte_code_differs_for_different_tokens():
    a = byte_kronecker_code("137")
    b = byte_kronecker_code("138")
    assert not np.array_equal(a, b)


def test_numeral_meaning_code_matches_closed_form():
    code = numeral_meaning_code("9")
    assert code.shape == (MEANING_DIM,)
    assert code[0] == 1.0
    assert math.isclose(code[1], math.log(9 + EPS), rel_tol=1e-6)
    assert math.isclose(code[2], 9 / NORM, rel_tol=1e-6)


def test_numeral_meaning_code_handles_negative_and_non_numeral():
    neg = numeral_meaning_code("-42")
    assert neg[0] == -1.0
    assert math.isclose(neg[2], -42 / NORM, rel_tol=1e-6)

    non_numeral = numeral_meaning_code("+")
    np.testing.assert_array_equal(non_numeral, np.zeros(MEANING_DIM, dtype=np.float32))


def test_distinct_integers_get_distinct_meaning_codes():
    codes = [numeral_meaning_code(str(n)) for n in range(51)]
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            assert not np.array_equal(codes[i], codes[j])


def test_kronecker_v2_forward_matches_manual_projection():
    vocab = Vocab(["0", "1", "2", "+", "="])
    torch.manual_seed(0)
    emb = KroneckerEmbeddingV2(vocab, d_model=8)

    ids = torch.tensor([vocab.encode("2"), vocab.encode("+")])
    out = emb(ids)
    assert out.shape == (2, 8)

    expected_0 = emb.proj(emb.normalized_full_code("2"))
    torch.testing.assert_close(out[0], expected_0)


def test_kronecker_v1_has_no_meaning_channel():
    vocab = Vocab(["9", "18"])
    emb = KroneckerV1TextOnly(vocab, d_model=4)
    assert emb.codes.shape == (2, BYTE_CODE_DIM)


def test_dense_baseline_forward_shape():
    vocab = Vocab(["0", "1", "+", "="])
    emb = DenseEmbeddingBaseline(vocab, d_model=8)
    ids = torch.tensor([[0, 2, 1, 3]])
    out = emb(ids)
    assert out.shape == (1, 4, 8)
