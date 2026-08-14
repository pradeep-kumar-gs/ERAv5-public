from __future__ import annotations

import numpy as np

from task import (
    EQ,
    EVEN,
    MAGNITUDE_HIGH,
    MAGNITUDE_LOW,
    MAX_NUM,
    PLUS,
    build_vocab,
    encode_examples,
    sample_even_even,
    sample_magnitude_ood,
    sample_ood,
)


def test_vocab_token_id_equals_numeric_value():
    vocab = build_vocab()
    assert len(vocab) == MAX_NUM + 1 + 2
    for n in (0, 1, 9, 137, MAX_NUM):
        assert vocab.encode(str(n)) == n
    assert vocab.encode(PLUS) == MAX_NUM + 1
    assert vocab.encode(EQ) == MAX_NUM + 2


def test_sample_even_even_is_always_even():
    rng = np.random.default_rng(0)
    a, b = sample_even_even(rng, 500)
    assert np.all(a % 2 == 0)
    assert np.all(b % 2 == 0)


def test_sample_ood_is_always_odd_plus_odd():
    rng = np.random.default_rng(0)
    a, b = sample_ood(rng, 500)
    assert np.all(a % 2 == 1)
    assert np.all(b % 2 == 1)
    assert np.all((a + b) % 2 == 0)  # result lands in the trained (even) output range


def test_sample_magnitude_ood_operand_never_used_in_training():
    """MAGNITUDE_HIGH must be disjoint from EVEN (the training operand pool),
    and the sum must land inside the trained target range (max training
    sum is 198+198=396), or this split would reintroduce the lm_head
    output-bottleneck confound sample_ood was redesigned to avoid."""
    assert set(MAGNITUDE_HIGH).isdisjoint(EVEN)
    rng = np.random.default_rng(0)
    a, b = sample_magnitude_ood(rng, 500)
    assert np.all(np.isin(a, MAGNITUDE_HIGH))
    assert np.all(np.isin(b, MAGNITUDE_LOW))
    assert set(MAGNITUDE_LOW).issubset(EVEN)
    result = a + b
    assert np.all(result <= 2 * (max(EVEN)))  # <= max training sum
    assert np.all(result % 2 == 0)


def test_train_batch_never_touches_an_odd_token():
    """Structural guarantee behind the OOD experiment: since even+even is
    always even, no odd-numbered token ever appears as input or target when
    training only on sample_even_even() pairs."""
    rng = np.random.default_rng(0)
    vocab = build_vocab()
    a, b = sample_even_even(rng, 2000)
    batch = encode_examples(vocab, a, b)
    for tensor in (batch.input_ids, batch.targets):
        numeric_ids = tensor[tensor <= MAX_NUM]
        assert numeric_ids.numel() > 0
        assert bool((numeric_ids % 2 == 0).all())


def test_loss_mask_only_marks_the_result_position():
    rng = np.random.default_rng(0)
    vocab = build_vocab()
    a, b = sample_even_even(rng, 10)
    batch = encode_examples(vocab, a, b)
    assert (batch.loss_mask[:, :-1] == 0).all()
    assert (batch.loss_mask[:, -1] == 1).all()
