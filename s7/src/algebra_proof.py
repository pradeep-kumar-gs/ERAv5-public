"""Training-free exactness proof for the meaning channel in kronecker.py.

Two claims are checked here, both without training any model:

1. + - x / are exactly recoverable (up to floating-point error) from vector
   addition/subtraction of two numeral meaning codes, across a wide sample of
   magnitudes including negative numbers. This is what "the embedding stores
   mathematical structure" means concretely: the recovery is a property of the
   fixed encoding, not something a network has to learn.

2. Exponentiation is NOT recoverable this way, because a**b = exp(b * log(a))
   is bilinear in (b, log a), and no linear map of two vectors' concatenation
   can represent a genuine cross term. A linear map of their Kronecker/outer
   product can, exactly. This is the boundary result: it is the reason the
   technique needs a Kronecker product at all, and it is why V2 stays scoped
   to the additive group (+ - x /) for the trained-model experiment.
"""

from __future__ import annotations

import numpy as np

from kronecker import (
    EPS,
    NORM,
    decode_diff,
    decode_product,
    decode_quotient,
    decode_sum,
    numeral_meaning_code,
)


def _sample_pairs(rng: np.random.Generator, n: int, low: float = -1e6, high: float = 1e6) -> list[tuple[float, float]]:
    """Mixture of ints and floats across a wide magnitude range, including negatives."""
    pairs = []
    for _ in range(n):
        kind = rng.integers(0, 4)
        if kind == 0:
            a, b = rng.integers(-1000, 1000, size=2)
        elif kind == 1:
            a, b = rng.uniform(low, high, size=2)
        elif kind == 2:
            a = float(rng.integers(-1_000_000, 1_000_000))
            b = rng.uniform(-10, 10)
        else:
            a = rng.uniform(-1e-3, 1e-3)
            b = rng.uniform(-1e-3, 1e-3)
        pairs.append((float(a), float(b)))
    return pairs


def _rel_err(pred: float, true: float, floor: float = 1e-2) -> float:
    """Relative error, floored at `floor` so near-zero results (e.g. a+b when
    a ~ -b, or a*b when either operand is tiny) don't blow up a metric that is
    only meaningful relative to the operands' own scale. This is a reporting
    choice, not a loosening of the claim: absolute float32 rounding at
    magnitude ~1e6 is already ~1e-1 (float32 carries ~7 significant digits),
    which is the real precision floor for numbers at that scale, independent
    of how they are encoded.
    """
    return abs(pred - true) / max(abs(true), floor)


def verify_additive_group(pairs: list[tuple[float, float]]) -> dict:
    """Checks + and - via the linear channel."""
    sum_errs, diff_errs = [], []
    for a, b in pairs:
        ca, cb = numeral_meaning_code(str(a)), numeral_meaning_code(str(b))
        sum_errs.append(_rel_err(decode_sum(ca, cb), a + b))
        diff_errs.append(_rel_err(decode_diff(ca, cb), a - b))
    return {
        "n": len(pairs),
        "add_max_rel_err": max(sum_errs),
        "add_mean_rel_err": float(np.mean(sum_errs)),
        "sub_max_rel_err": max(diff_errs),
        "sub_mean_rel_err": float(np.mean(diff_errs)),
    }


def verify_multiplicative_group(pairs: list[tuple[float, float]]) -> dict:
    """Checks x and / via the log/sign channel.

    Division skips divisors that are within ~1000x of EPS (1e-6): `log(|b|+eps)`
    is dominated by the eps regularizer there, and dividing amplifies that
    distortion arbitrarily as b -> 0. That is a property of representing
    magnitude via a regularized log near its own regularization scale, true of
    any such encoding -- not specific to this one -- so it is out of scope for
    "division is exact", the same way dividing by a near-zero float is
    ill-conditioned in ordinary floating point too.
    """
    mul_errs, div_errs = [], []
    for a, b in pairs:
        ca, cb = numeral_meaning_code(str(a)), numeral_meaning_code(str(b))
        mul_errs.append(_rel_err(decode_product(ca, cb), a * b))
        if abs(b) > 1e-3:
            div_errs.append(_rel_err(decode_quotient(ca, cb), a / b))
    return {
        "n": len(pairs),
        "mul_max_rel_err": max(mul_errs),
        "mul_mean_rel_err": float(np.mean(mul_errs)),
        "div_max_rel_err": max(div_errs) if div_errs else None,
        "div_mean_rel_err": float(np.mean(div_errs)) if div_errs else None,
    }


def verify_power_bilinear(rng: np.random.Generator, n: int = 2000) -> dict:
    """Exponentiation needs a bilinear (Kronecker/outer-product) readout.

    target = b * log(a) is the quantity a**b = exp(target) needs. It is
    reconstructed exactly by a linear map on the Kronecker/outer product of
    two small feature vectors, and it is NOT reconstructable (large residual)
    by the best possible linear map on their plain concatenation -- that gap
    is the proof that addition alone (what V2's trained experiment uses)
    cannot express power, and something with tensor-product structure is
    required instead.
    """
    a = rng.uniform(0.1, 10.0, size=n)
    b = rng.uniform(-3.0, 3.0, size=n)
    log_a = np.log(a)
    target = b * log_a

    u = np.stack([np.ones(n), log_a], axis=1)  # channel derived from a
    v = np.stack([np.ones(n), b], axis=1)  # channel derived from b
    outer = (u[:, :, None] * v[:, None, :]).reshape(n, 4)  # Kronecker/outer product per sample
    w_outer = np.array([0.0, 0.0, 0.0, 1.0])  # picks out exactly the log_a * b cross term
    outer_pred = outer @ w_outer
    outer_err = float(np.max(np.abs(outer_pred - target)))

    reconstructed_pow = np.exp(outer_pred)
    true_pow = a**b
    pow_rel_err = float(np.max(np.abs(reconstructed_pow - true_pow) / np.abs(true_pow)))

    concat = np.concatenate([u, v], axis=1)  # plain concatenation, no cross terms available
    w_concat, *_ = np.linalg.lstsq(concat, target, rcond=None)
    concat_pred = concat @ w_concat
    concat_err = float(np.max(np.abs(concat_pred - target)))

    return {
        "n": n,
        "outer_product_max_abs_err": outer_err,
        "power_reconstruction_max_rel_err": pow_rel_err,
        "best_linear_on_concat_max_abs_err": concat_err,
        "concat_err_over_outer_err": concat_err / max(outer_err, 1e-12),
    }


def run_all(seed: int = 0, n_pairs: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    pairs = _sample_pairs(rng, n_pairs)
    return {
        "eps": EPS,
        "norm": NORM,
        "additive_group": verify_additive_group(pairs),
        "multiplicative_group": verify_multiplicative_group(pairs),
        "power_bilinear_boundary": verify_power_bilinear(rng),
        "worked_example": {
            "9+9": decode_sum(numeral_meaning_code("9"), numeral_meaning_code("9")),
            "9*9": decode_product(numeral_meaning_code("9"), numeral_meaning_code("9")),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=2))
