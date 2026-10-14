"""What the experts learned, measured on validation text.

The course's claim (Section 9): experts specialise by kind of token (punctuation, function words, names,
numbers), not by subject. TinyStories has no subjects to speak of, so only the first half can be tested here.
"""
import math
import string
from collections import Counter

import numpy as np
import torch
from tokenizers import Tokenizer

from .data import TOKENIZER, Stream

FUNCTION_WORDS = set("""a an the and or but so then because if when while as of to in on at by for with from into
up out over under about after before he she it they we you i him her them us me his its their our your my was were is
are be been had has have do did not no that this there what who which said""".split())
CATEGORIES = ["punctuation", "newline/end", "number", "capitalised word", "function word", "other word"]


def token_categories():
    tok = Tokenizer.from_file(str(TOKENIZER))
    eot = tok.token_to_id("<|endoftext|>")
    cats = []
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i])
        w = s.strip()
        if i == eot or (s and not w and "\n" in s):
            cats.append(1)
        elif not w or all(ch in string.punctuation for ch in w):
            cats.append(0)
        elif any(ch.isdigit() for ch in w):
            cats.append(2)
        elif w.lower() in FUNCTION_WORDS:
            cats.append(4)
        elif w[0].isupper():
            cats.append(3)
        else:
            cats.append(5)
    return np.array(cats), tok


@torch.no_grad()
def route_validation(model, dev, n_batches=8, batch=16, seq=256):
    """Returns tokens (M,) and each layer's top-1 expert (L, M) and top-k experts (L, M, k)."""
    model.eval()
    layers = model.moe_layers()
    for m in layers:
        m.keep_choice = True
    s = Stream(batch, seq, "valid")
    toks, choices = [], [[] for _ in layers]
    for _ in range(n_batches):
        x, _ = s.next()
        model(x.to(dev))
        toks.append(x.flatten())
        for c, m in zip(choices, layers):
            c.append(m.last_choice.cpu())
    for m in layers:
        m.keep_choice, m.last_choice = False, None
    model.train()
    return torch.cat(toks).numpy(), torch.stack([torch.cat(c) for c in choices]).numpy()


def mutual_information(a, b):
    """I(a; b) in bits for two integer arrays."""
    joint = Counter(zip(a.tolist(), b.tolist()))
    n = len(a)
    pa, pb = Counter(a.tolist()), Counter(b.tolist())
    return sum(c / n * math.log2((c / n) / (pa[x] / n * pb[y] / n)) for (x, y), c in joint.items())


def specialisation(tokens, choices, cats, seed=0):
    """Per layer:
    - mi_type: how many bits the top-1 expert tells you about the token's type, against the same experts with
      the tokens shuffled (mi_shuffled, the chance level);
    - h_type: the type's entropy, the most mi_type could be;
    - same_next: how often token t+1 picks the same top-1 expert as token t, against chance (sum of squared loads).
    """
    rng = np.random.default_rng(seed)
    types = cats[tokens]
    pt = np.bincount(types, minlength=len(CATEGORIES)) / len(types)
    h_type = -sum(p * math.log2(p) for p in pt if p > 0)
    rows = []
    for layer, ch in enumerate(choices):
        top1 = ch[:, 0]
        load = np.bincount(top1) / len(top1)
        rows.append(dict(layer=layer, mi_type=mutual_information(types, top1),
                         mi_shuffled=mutual_information(rng.permutation(types), top1), h_type=h_type,
                         same_next=float((top1[1:] == top1[:-1]).mean()), same_next_chance=float((load ** 2).sum())))
    return rows


def type_mix(tokens, choices, cats, layer):
    """Share of each token type among the tokens each expert receives (top-1), one row per expert."""
    types = cats[tokens]
    top1 = choices[layer][:, 0]
    E = int(choices.max()) + 1
    out = np.zeros((E, len(CATEGORIES)))
    for e in range(E):
        t = types[top1 == e]
        if len(t):
            out[e] = np.bincount(t, minlength=len(CATEGORIES)) / len(t)
    return out


def top_tokens(tokens, choices, tok, layer, k=8, min_count=20):
    """For each expert, the tokens most over-represented in its share (lift over their overall frequency)."""
    top1 = choices[layer][:, 0]
    overall = Counter(tokens.tolist())
    n = len(tokens)
    out = {}
    for e in range(int(choices.max()) + 1):
        mine = Counter(tokens[top1 == e].tolist())
        m = sum(mine.values())
        if not m:
            out[e] = []
            continue
        lift = [(c / m) / (overall[t] / n) for t, c in mine.items() if c >= min_count]
        ids = [t for t, c in mine.items() if c >= min_count]
        best = sorted(zip(lift, ids), reverse=True)[:k]
        out[e] = [(tok.decode([t]), round(l, 1)) for l, t in best]
    return out


@torch.no_grad()
def expert_similarity(model):
    """Mean pairwise cosine similarity between the experts' flattened weights, per layer."""
    rows = []
    for f in model.moe_layers():
        w = torch.cat([f.w_up.flatten(1), f.w_down.transpose(1, 2).flatten(1)], dim=1).float().cpu()
        w = torch.nn.functional.normalize(w, dim=1)
        sim = w @ w.t()
        E = sim.shape[0]
        rows.append(((sim.sum() - E) / (E * (E - 1))).item())
    return rows
