"""Single-token integer addition, with an even/odd split engineered to reproduce
two claims the course made from the model side (Section 2's scatter-add: "a row
that did not appear in this batch receives exactly zero gradient", Section 11's
wall: "row 7,000 does not exist") on numeral tokens instead of rare words or
out-of-range positions.

Vocabulary: integers 0..MAX_NUM as atomic tokens, plus '+' and '='. Training
only ever uses EVEN operands (0, 2, ..., 198); since even+even is always even,
no ODD token (1, 3, ..., 199) appears anywhere in the training set -- not as
an input, not as a target. The out-of-distribution eval set uses ODD + ODD, so
every OOD input token is one the model never received gradient for under a
dense embedding table (see below for why the result stays even).

Operand range is kept small (0..199) so the combinatorial pair space (100x100)
is actually learnable by a tiny CPU-trained transformer in a few thousand
steps -- the point of the experiment is the *generalization gap* between
embedding schemes, which only means something once the in-distribution task
is close to solved.

OOD pairs are ODD + ODD, not "at least one odd". Reason: the model's output
classifier (lm_head) has exactly the same untrained-row problem as a dense
input embedding, and it applies regardless of which input embedding scheme is
used -- since training targets are always even (even+even=even), lm_head's
rows for odd-valued *results* never receive gradient either. odd+even gives an
odd result, which no arm could predict correctly no matter how good its input
embedding is; that confound would swamp the thing this experiment is actually
trying to isolate. odd+odd gives an even result (already in the trained output
range) while both *inputs* are tokens that never appeared as an input during
training -- so it isolates the input-embedding generalization question from
the separate, output-side bottleneck.

A second, independent OOD axis: magnitude. The odd/even split tests whether an
embedding generalizes to unseen tokens that are *interleaved* with trained
ones (198 is trained, 199 is not, but they're adjacent and share almost all
byte-code structure and both meaning-channel coordinates are nearly
identical). It says little about whether the embedding generalizes to tokens
*larger* than anything trained on -- which is the more direct test of
Problem 1's actual claim (does the meaning channel encode true, extrapolable
magnitude, not just "not-yet-seen-but-nearby"). MAGNITUDE_HIGH (200..298,
even) is never sampled as a training operand (training only ever draws from
EVEN, 0..198) but every value in it *is* reachable as a training sum
(e.g. 100+100=200), so its lm_head output row is trained even though its
input embedding row is not -- avoiding the same output-bottleneck confound
odd+odd was designed to avoid. Pairing it with a small, familiar operand from
MAGNITUDE_LOW (0..98, even, a strict subset of the trained operand pool) keeps
the sum inside the densely-trained target range (max training sum is
198+198=396) while making the single unfamiliar-magnitude operand the only
new variable under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kronecker import Vocab

PLUS = "+"
EQ = "="
NUM_RANGE = 200  # operands drawn from 0..NUM_RANGE-1
MAX_NUM = 2 * (NUM_RANGE - 1)  # covers every even+even and every 0..199 + 0..199 sum
EVEN = list(range(0, NUM_RANGE, 2))
ODD = list(range(1, NUM_RANGE, 2))
MAGNITUDE_LOW = list(range(0, 100, 2))  # familiar operand: subset of EVEN
MAGNITUDE_HIGH = list(range(200, 300, 2))  # never sampled as a training operand


def build_vocab() -> Vocab:
    tokens = [str(n) for n in range(MAX_NUM + 1)] + [PLUS, EQ]
    return Vocab(tokens)


@dataclass
class Batch:
    input_ids: torch.Tensor  # (N, 4) = [A, '+', B, '=']
    targets: torch.Tensor  # (N, 4) shifted by one
    loss_mask: torch.Tensor  # (N, 4), 1 only at the position predicting the result
    a: np.ndarray
    b: np.ndarray
    result: np.ndarray


def encode_examples(vocab: Vocab, a_arr: np.ndarray, b_arr: np.ndarray) -> Batch:
    plus_id = vocab.encode(PLUS)
    eq_id = vocab.encode(EQ)
    result = a_arr + b_arr
    seqs = np.zeros((len(a_arr), 5), dtype=np.int64)
    for i, (a, b, r) in enumerate(zip(a_arr, b_arr, result)):
        seqs[i] = [vocab.encode(str(int(a))), plus_id, vocab.encode(str(int(b))), eq_id, vocab.encode(str(int(r)))]
    input_ids = seqs[:, :-1]
    targets = seqs[:, 1:]
    loss_mask = np.zeros_like(targets)
    loss_mask[:, -1] = 1
    return Batch(
        input_ids=torch.from_numpy(input_ids),
        targets=torch.from_numpy(targets),
        loss_mask=torch.from_numpy(loss_mask),
        a=a_arr,
        b=b_arr,
        result=result,
    )


def sample_even_even(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.choice(EVEN, size=n).astype(np.int64)
    b = rng.choice(EVEN, size=n).astype(np.int64)
    return a, b


def sample_ood(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Both operands odd -> both input tokens are unseen, result is even (in
    the trained output range). See module docstring for why not "any odd"."""
    a = rng.choice(ODD, size=n).astype(np.int64)
    b = rng.choice(ODD, size=n).astype(np.int64)
    return a, b


def sample_magnitude_ood(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """One operand from a magnitude band never sampled as a training input
    (MAGNITUDE_HIGH, 200..298), paired with a small familiar operand
    (MAGNITUDE_LOW, a subset of what training already uses) so the sum stays
    inside the densely-trained target range. See module docstring for why
    this is a different, stronger axis than sample_ood's odd/even split."""
    a = rng.choice(MAGNITUDE_HIGH, size=n).astype(np.int64)
    b = rng.choice(MAGNITUDE_LOW, size=n).astype(np.int64)
    return a, b


def make_splits(
    rng: np.random.Generator,
    vocab: Vocab,
    n_train: int,
    n_eval_id: int,
    n_eval_ood: int,
) -> dict[str, Batch]:
    train_a, train_b = sample_even_even(rng, n_train)
    id_a, id_b = sample_even_even(rng, n_eval_id)
    ood_a, ood_b = sample_ood(rng, n_eval_ood)
    return {
        "train": encode_examples(vocab, train_a, train_b),
        "eval_in_distribution": encode_examples(vocab, id_a, id_b),
        "eval_ood": encode_examples(vocab, ood_a, ood_b),
    }
