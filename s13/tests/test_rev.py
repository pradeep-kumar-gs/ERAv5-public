"""Small CPU checks of the claims the README relies on. Run: pytest tests -q"""
import math

import pytest
import torch

from rev.analysis import gradient_agreement, reconstruction
from rev.data import Stream
from rev.model import GPT, GPTConfig
from rev.train import RunConfig, lr_at, peak_lr

torch.set_num_threads(2)
RULES = [("midpoint", 0.5, 1.0), ("midpoint", 0.25, 0.5), ("euler", 1.0, 1.0)]


def tiny(rule, h=0.5, blend=1.0, layers=6):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab=64, seq=16, d=32, n_head=4, n_layer=layers, rule=rule, h=h, blend=blend)).double()


def data():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 64, (3, 16), generator=g), torch.randint(0, 64, (3, 16), generator=g)


@pytest.mark.parametrize("rule,h,blend", RULES)
def test_reversible_gradients_equal_stored(rule, h, blend):
    m = tiny(rule, h, blend)
    agree = gradient_agreement(m, *data())
    assert agree["rel_err"] < 1e-12 and agree["cos"] > 1 - 1e-12


@pytest.mark.parametrize("rule,h,blend", RULES)
def test_inverse_rebuilds_the_input(rule, h, blend):
    m = tiny(rule, h, blend)
    assert max(reconstruction(m, data()[0])) < 1e-12


def test_reversible_keeps_only_two_states():
    m = tiny("midpoint")
    m.cfg.reversible = True
    idx, y = data()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        m.trunk(m.embed(idx))
    # the reversible Function saves exactly the two boundary states; the embedding adds nothing big
    trunk_saved = [t for t in saved if t.shape == (3, 16, 32)]
    assert len(trunk_saved) == 2


def test_blend_halves_amplify_error():
    """With blend a the inverse divides by a every layer, so float32 rounding grows like (1/a)^L."""
    errs = {}
    for blend in (1.0, 0.5):
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab=64, seq=16, d=32, n_head=4, n_layer=24, rule="midpoint", h=0.25, blend=blend))
        errs[blend] = reconstruction(m, data()[0])[0]
    assert errs[0.5] > 100 * errs[1.0]


def test_chunked_loss_equals_full():
    m = tiny("standard")
    idx, y = data()
    full = m(idx, y)
    m.cfg.loss_chunk = 7
    assert torch.allclose(full, m(idx, y), atol=1e-12)


def test_lr_schedule():
    c = RunConfig(batch=64, tokens=10_000_000)
    assert peak_lr(c) == c.lr
    assert peak_lr(RunConfig(batch=4096)) == c.lr_cap
    warm = c.warmup_tokens
    assert math.isclose(lr_at(c, warm), c.lr)
    assert math.isclose(lr_at(c, c.tokens), c.lr * c.min_lr_frac)


def test_stream_is_contiguous_and_shifted():
    s = Stream(2, 8)
    x1, y1 = s.next()
    x2, _ = s.next()
    assert torch.equal(x1.flatten()[1:], y1.flatten()[:-1])
    assert x2.flatten()[0] == y1.flatten()[-1]
