from __future__ import annotations

import random

import pytest

import config
import mixture
import train as train_mod


def test_assert_trainable_blocks_eval_lane(manifests):
    with pytest.raises(RuntimeError, match="firewall breach"):
        train_mod.assert_trainable("eval", manifests)


def test_assert_trainable_allows_train_lanes(manifests):
    for lane in ("prose", "code", "agent"):
        train_mod.assert_trainable(lane, manifests)  # must not raise


def test_eval_lane_is_never_sampled_by_the_mixture():
    rng = random.Random(0)
    for step in range(1, config.TOTAL_STEPS + 1):
        for _ in range(20):
            lane = mixture.sample_lane(step, rng)
            assert lane != "eval"


def test_eval_lane_is_excluded_from_every_stage_weight_table():
    for stage in mixture.STAGES:
        assert "eval" not in stage.weights
        assert "eval" not in stage.eligible
