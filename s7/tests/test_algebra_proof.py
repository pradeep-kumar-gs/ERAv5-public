from __future__ import annotations

import numpy as np

from algebra_proof import (
    _sample_pairs,
    verify_additive_group,
    verify_multiplicative_group,
    verify_power_bilinear,
)


def test_additive_group_is_exact():
    rng = np.random.default_rng(1)
    pairs = _sample_pairs(rng, 2000)
    result = verify_additive_group(pairs)
    assert result["add_max_rel_err"] < 1e-4
    assert result["sub_max_rel_err"] < 1e-4


def test_multiplicative_group_is_exact():
    rng = np.random.default_rng(2)
    pairs = _sample_pairs(rng, 2000)
    result = verify_multiplicative_group(pairs)
    assert result["mul_max_rel_err"] < 1e-3
    assert result["div_max_rel_err"] < 1e-3


def test_power_needs_bilinear_readout():
    rng = np.random.default_rng(3)
    result = verify_power_bilinear(rng, n=1000)
    # the Kronecker/outer-product readout reconstructs the bilinear term exactly
    assert result["outer_product_max_abs_err"] < 1e-8
    assert result["power_reconstruction_max_rel_err"] < 1e-6
    # no linear map on the plain concatenation gets remotely close
    assert result["best_linear_on_concat_max_abs_err"] > 1.0
    assert result["concat_err_over_outer_err"] > 1e6
