"""Trains the three embedding arms (dense table / Kronecker V1 byte-only /
Kronecker V2 byte+meaning) on the even-only addition task from task.py, and logs
what the algebra proof alone cannot show: whether a network reading through each
embedding actually learns to exploit the structure that's there.

Dense is the Section 2 baseline: a row per token, gradient only for rows that
appeared. Kronecker V1 shares a projection across all tokens but only sees
spelling. Kronecker V2 shares a projection and sees both spelling and the
appended arithmetic meaning channel. All three see identical (A, '+', B, '=')
sequences and identical optimizer settings; the only thing that differs is
what tok_emb is made of.
"""

from __future__ import annotations

import numpy as np
import torch

from kronecker import DenseEmbeddingBaseline, KroneckerEmbeddingV2, KroneckerV1TextOnly, Vocab
from model import GPT, GPTConfig
from task import build_vocab, encode_examples, sample_even_even, sample_magnitude_ood, sample_ood

ARMS = ("dense", "kron_v1", "kron_v2")


def make_embedding(arm: str, vocab: Vocab, d_model: int) -> torch.nn.Module:
    if arm == "dense":
        return DenseEmbeddingBaseline(vocab, d_model)
    if arm == "kron_v1":
        return KroneckerV1TextOnly(vocab, d_model)
    if arm == "kron_v2":
        return KroneckerEmbeddingV2(vocab, d_model)
    raise ValueError(f"unknown arm {arm!r}")


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits[:, -1, :].argmax(dim=-1)
    return (pred == target[:, -1]).float().mean().item()


def train_one_arm(
    arm: str,
    vocab: Vocab,
    seed: int,
    n_steps: int = 600,
    batch_size: int = 256,
    eval_every: int = 50,
    eval_n: int = 1000,
    lr: float = 3e-3,
    d_model: int = 64,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    config = GPTConfig(vocab_size=len(vocab), block_size=4, n_layer=2, n_head=2, n_embd=d_model)
    model = GPT(config, make_embedding(arm, vocab, d_model)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    eval_id = encode_examples(vocab, *sample_even_even(rng, eval_n))
    eval_ood = encode_examples(vocab, *sample_ood(rng, eval_n))
    eval_mag = encode_examples(vocab, *sample_magnitude_ood(rng, eval_n))

    grad_mass = np.zeros(len(vocab), dtype=np.float64) if arm == "dense" else None

    curve = []
    last_loss = None
    for step in range(1, n_steps + 1):
        batch = encode_examples(vocab, *sample_even_even(rng, batch_size))
        model.train()
        _, loss = model(batch.input_ids.to(device), batch.targets.to(device), batch.loss_mask.to(device))
        opt.zero_grad()
        loss.backward()

        if grad_mass is not None:
            g = model.tok_emb.emb.weight.grad
            if g is not None:
                grad_mass += g.detach().pow(2).sum(dim=1).sqrt().cpu().numpy()

        opt.step()
        last_loss = loss.item()

        if step % eval_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                id_logits, _ = model(eval_id.input_ids.to(device))
                ood_logits, _ = model(eval_ood.input_ids.to(device))
                mag_logits, _ = model(eval_mag.input_ids.to(device))
            curve.append(
                {
                    "step": step,
                    "train_loss": last_loss,
                    "id_acc": _accuracy(id_logits, eval_id.targets.to(device)),
                    "ood_acc": _accuracy(ood_logits, eval_ood.targets.to(device)),
                    "magnitude_ood_acc": _accuracy(mag_logits, eval_mag.targets.to(device)),
                }
            )

    return {
        "arm": arm,
        "seed": seed,
        "curve": curve,
        "grad_mass": grad_mass.tolist() if grad_mass is not None else None,
        "num_params": model.num_params(),
    }


def run_comparison(seeds: tuple[int, ...] = (0, 1, 2), **kwargs) -> tuple[Vocab, dict[str, list[dict]]]:
    vocab = build_vocab()
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            results[arm].append(train_one_arm(arm, vocab, seed, **kwargs))
    return vocab, results


if __name__ == "__main__":
    import json

    _, res = run_comparison(seeds=(0,), n_steps=300)
    print(json.dumps({arm: res[arm][0]["curve"][-1] for arm in ARMS}, indent=2))
