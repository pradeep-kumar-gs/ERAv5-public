"""Small CPU checks of the claims the README relies on. Run: pytest tests -q"""
import torch

from moe.convert import dense_to_moe, grow
from moe.model import GPT, GPTConfig, MoE, MoEConfig
from moe.train import RunConfig, lr_at

torch.set_num_threads(2)
SMALL = GPTConfig(vocab=64, seq=16, d=32, n_head=4, n_layer=2, ffn=128)


def dense():
    torch.manual_seed(0)
    return GPT(SMALL).double()


def data(n=3):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 64, (n, 16), generator=g), torch.randint(0, 64, (n, 16), generator=g)


def test_copy_upcycling_is_exactly_the_dense_model():
    """Identical experts with weights renormalised to 1: whatever the router picks, the output is the dense FFN."""
    d = dense()
    m = dense_to_moe(d, "copy").double()
    idx, y = data()
    with torch.no_grad():
        assert torch.allclose(d(idx, y), m(idx, y), atol=1e-12)


def test_drop_and_partition_change_the_function():
    d = dense()
    idx, y = data()
    with torch.no_grad():
        base = d(idx, y)
        for method in ("drop", "partition"):
            assert not torch.allclose(base, dense_to_moe(d, method).double()(idx, y), atol=1e-6)


def test_drop_redraws_half_the_neurons_of_each_expert():
    d = dense()
    m = dense_to_moe(d, "drop", r=0.5)
    up = d.blocks[0].ffn.up.weight
    for e in range(8):
        same = (m.blocks[0].ffn.w_up[e].double() == up).all(dim=1).sum().item()
        assert same == SMALL.ffn // 2


def test_partition_keeps_dense_as_shared_and_halves_experts():
    d = dense()
    m = dense_to_moe(d, "partition")
    f = m.blocks[0].ffn
    assert torch.equal(f.shared.up.weight.double(), d.blocks[0].ffn.up.weight)
    assert f.w_up.shape == (8, SMALL.ffn // 2, SMALL.d)


def test_active_params_are_equal_across_methods():
    d = dense()
    actives = {meth: dense_to_moe(d, meth).n_active() for meth in ("copy", "drop", "partition")}
    assert len(set(actives.values())) == 1


def test_grow_tiles_experts_and_router_into_clone_families():
    d = dense()
    m = dense_to_moe(d, "drop").double()
    g = grow(m, factor=4, router_noise=0.0).double()
    big, small = g.blocks[0].ffn, m.blocks[0].ffn
    assert big.w_up.shape[0] == 32
    for e in range(32):                               # expert e is a clone of expert e // 4
        assert torch.equal(big.w_up[e], small.w_up[e // 4])
        assert torch.equal(big.router.weight[e], small.router.weight[e // 4])


def test_routing_is_dropless_and_weights_sum_to_one():
    torch.manual_seed(0)
    layer = MoE(32, MoEConfig(n_experts=8, top_k=2, expert_width=16))
    torch.nn.init.normal_(layer.router.weight, 0, 1.0)
    torch.nn.init.normal_(layer.w_up, 0, 0.1)
    torch.nn.init.normal_(layer.w_down, 0, 0.1)
    x = torch.randn(4, 50, 32)
    idx, w = layer.route(x.reshape(-1, 32))
    assert torch.allclose(w.sum(-1), torch.ones(200))
    layer(x)
    assert layer.counts.sum().item() == 200 * 2                 # every (token, choice) was processed


def test_bias_steers_selection_but_not_weights():
    torch.manual_seed(0)
    layer = MoE(32, MoEConfig(n_experts=8, top_k=2, expert_width=16))
    h = torch.randn(100, 32)
    idx0, _ = layer.route(h)
    layer.bias[idx0[:, 0].mode().values] = -1.0               # punish the favourite
    idx1, w1 = layer.route(h)
    assert not torch.equal(idx0, idx1)
    probs = (h @ layer.router.weight.t()).softmax(-1)
    raw = probs.gather(1, idx1)
    assert torch.allclose(w1, raw / raw.sum(-1, keepdim=True))  # weights come from the raw scores


def test_sampled_selection_reaches_every_clone():
    torch.manual_seed(0)
    layer = MoE(32, MoEConfig(n_experts=8, top_k=2, expert_width=16))
    layer.router.weight.data = layer.router.weight.data[:2].repeat(4, 1)   # 4 identical pairs
    h = torch.randn(4000, 32)
    layer.selection = "hard"
    hard = set(layer.route(h)[0].flatten().tolist())
    layer.selection = "sample"
    layer.train()
    sampled = set(layer.route(h)[0].flatten().tolist())
    assert len(sampled) == 8 and len(hard) < 8


def test_lr_schedule_rewarms_after_conversion():
    c = RunConfig(name="x")
    assert lr_at(c, 10_000_000) == c.lr
    conv = RunConfig(name="y", start_token=10_000_000)
    assert lr_at(conv, 10_000_000) < 0.05 * c.lr
    assert lr_at(conv, 10_600_000) == c.lr
    assert abs(lr_at(c, c.total_tokens) - c.lr * c.min_lr_frac) < 1e-12
