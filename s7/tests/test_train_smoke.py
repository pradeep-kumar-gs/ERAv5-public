from __future__ import annotations

import math

from task import ODD, build_vocab
from train_compare import ARMS, train_one_arm


def test_all_arms_train_a_few_steps_without_nans():
    vocab = build_vocab()
    for arm in ARMS:
        result = train_one_arm(
            arm,
            vocab,
            seed=0,
            n_steps=5,
            batch_size=32,
            eval_every=5,
            eval_n=20,
        )
        assert result["curve"], f"{arm} produced no eval points"
        for point in result["curve"]:
            assert math.isfinite(point["train_loss"])
            assert 0.0 <= point["id_acc"] <= 1.0
            assert 0.0 <= point["ood_acc"] <= 1.0


def test_dense_arm_grad_mass_is_exactly_zero_for_odd_tokens():
    """Direct instrumentation of the course's scatter-add claim: a row that
    never appeared in a batch receives exactly zero gradient. This holds after
    a single step already, independent of how much training happens."""
    vocab = build_vocab()
    result = train_one_arm("dense", vocab, seed=0, n_steps=3, batch_size=32, eval_every=3, eval_n=10)
    grad_mass = result["grad_mass"]
    assert grad_mass is not None
    for odd_value in ODD[:50]:
        assert grad_mass[odd_value] == 0.0
