"""Builds s14_dense_to_moe.ipynb. Execute it with:

    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=s14-moe s14_dense_to_moe.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Session 14: train a dense model, convert it into a mixture of experts, keep training

**Assignment:** *Train a Linear model and convert that into an MoE! Your call on model size and data, but must show
they continue to train and reduce loss!*

"Linear model" in class means the ordinary dense feed-forward block that every token passes through in full.
The plan:

1. Train a ~21M-parameter dense GPT on TinyStories.
2. At 15M tokens, convert its feed-forward blocks into mixtures of experts, three different ways.
3. Keep training every converted model to 30M tokens.
4. Compare against the dense model trained on the same tokens, and against the same MoE trained from scratch.

On top of the assignment:
- a second growth step, 8 → 32 experts by cloning, which reproduces the "collapse along clone families" failure
  the course describes, and tests its fix;
- a look at what the experts specialise in.

Everything ran on one Apple M1 Pro (MPS, fp32). The training runs are made by `python -m moe.experiments`, which
saves every run to `results/`. This notebook reads those runs and makes every other measurement live.

| piece | where |
|---|---|
| data (TinyStories, 8,192-token BPE, same as Session 13) | `moe/data.py` |
| model: dense FFN or MoE (fp32 router, bias balancing, dropless, hard or sampled top-k) | `moe/model.py` |
| dense → MoE (copy, drop, partition) and MoE → bigger MoE (grow) | `moe/convert.py` |
| training on a shared token axis, routing statistics | `moe/train.py` |
| run order | `moe/experiments.py` |
| specialisation measurements | `moe/analysis.py` |
| tests | `tests/test_moe.py` |
""")

code(r"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from moe.model import GPT, GPTConfig
from moe.convert import dense_to_moe, grow
from moe.data import Stream
from moe import analysis as A
from moe.experiments import CONVERT_AT, GROW_AT, METHODS, GROW_RUNS, moe_config, best_at_grow, load as R

GiB = 2 ** 30
RES = ROOT / "results"
ART = ROOT / "submission_artifacts"; ART.mkdir(exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NUM = {}
print("torch", torch.__version__, "| device", DEV, "| conversion at", f"{CONVERT_AT:,}", "| growth at", f"{GROW_AT:,}")
""")

md(r"""
## 1. The models

**The dense model.**
- d = 384, 10 layers, 6 heads, sequence 256.
- An 8,192-token vocabulary with a tied head.
- A GELU feed-forward block of inner width 1,536 (4 × d).

**The mixture of experts** replaces that block in **every** layer and keeps everything else. All three conversions
give each token the same work, about twice the dense FFN:

| method | routed experts | each expert's width | shared expert | top-k | width each token uses |
|---|---|---|---|---|---|
| copy | 8 | 1,536 (a full copy) | – | 2 | 2 × 1,536 = 3,072 |
| drop (r = 0.5) | 8 | 1,536 (copy, half the neurons re-initialised) | – | 2 | 3,072 |
| partition | 8 | 768 (a random half of the neurons) | the dense FFN, 1,536 | 2 | 1,536 + 2 × 768 = 3,072 |

Top-1 with full copies would use the same width as dense, but it has a problem: the single chosen weight is
renormalised to exactly 1, so the router gets no gradient. Top-2 is the smallest choice for which routing is learned.
""")

code(r"""
rows = {}
for name, model in [("dense", GPT(GPTConfig())), ("copy / drop", GPT(moe_config("copy"))),
                    ("partition", GPT(moe_config("partition"))), (f"grown 32 experts (from {best_at_grow()})", grow(GPT(moe_config(best_at_grow())), 4))]:
    rows[name] = dict(total=model.n_params(), active=model.n_active())
    print(f"{name:18s} total {model.n_params()/1e6:7.1f}M   active per token {model.n_active()/1e6:5.1f}M   "
          f"training FLOPs per token ~ 6 x active = {6*model.n_active()/1e6:5.0f} MFLOP")
    del model
NUM["params"] = rows
""")

md(r"""
## 2. What each conversion does to the model at the moment of conversion

The same trained dense checkpoint (at 15M tokens) is converted each way, and every result is scored on the same
validation batches:

- **Copy** should give exactly the dense model's loss. Every expert is identical, and the two chosen weights are
  renormalised to sum to 1, so whatever the new router picks, the output is the dense FFN's output.
- **Drop** and **partition** change the function, so they start from a worse loss and have to recover it.
""")

code(r"""
dense15 = GPT(GPTConfig()); dense15.load_state_dict(torch.load(RES / f"dense_at{CONVERT_AT}.pt", map_location="cpu"))
s = Stream(16, 256, "valid"); batches = [s.next() for _ in range(20)]
def val(model):
    model = model.to(DEV).eval()
    with torch.no_grad():
        return float(np.mean([model(x.to(DEV), y.to(DEV)).item() for x, y in batches]))
at_conv = {"dense": val(dense15)}
for m in METHODS:
    at_conv[m] = val(dense_to_moe(dense15.cpu(), m))
for k, v in at_conv.items():
    print(f"{k:10s} val loss at conversion {v:.4f}   ({v - at_conv['dense']:+.4f} vs dense)")
NUM["val_at_conversion"] = at_conv
""")

md(r"""
## 3. Do they continue to train and reduce loss?

The three converted models train from 15M to 30M tokens. The dense run carries on over the same tokens, on the same
schedule: warmup, constant, then decay to 10% over the last 6M. That is the fair comparison: same data, same
learning rate.

Each converted model gets a fresh AdamW and a 0.5M-token re-warmup, because its experts and router are new tensors.
""")

code(r"""
D = R("dense"); RUNS = {m: R(f"moe_{m}") for m in METHODS}
def ev_at(log, t):
    return min(log["evals"], key=lambda e: abs(e["tokens"] - t))
print(f"{'model':10s} {'val @ conv':>10s} {'val @ +1.5M':>11s} {'val @ 24M':>10s} {'final val':>10s} {'vs dense':>9s} "
      f"{'tok/s':>7s} {'peak GiB':>9s}")
NUM["main"] = {}
dfinal = D["final_val_loss"]
for name, log in [("dense", D)] + list(RUNS.items()):
    first = log["evals"][0] if name != "dense" else ev_at(D, CONVERT_AT)
    r = dict(val_at_conv=first["val_loss"], val_after_1_5M=ev_at(log, CONVERT_AT + 1_500_000)["val_loss"],
             val_24M=ev_at(log, GROW_AT)["val_loss"], final_val=log["final_val_loss"], final_train=log["final_train_loss"],
             tok_s=log["tok_per_s"], peak_gib=log["peak_bytes"] / GiB, total=log["n_params"], active=log["n_active"],
             minutes=log["train_seconds"] / 60)
    NUM["main"][name] = r
    print(f"{name:10s} {r['val_at_conv']:10.4f} {r['val_after_1_5M']:11.4f} {r['val_24M']:10.4f} {r['final_val']:10.4f} "
          f"{r['final_val']-dfinal:+9.4f} {r['tok_s']:7.0f} {r['peak_gib']:9.2f}")
print()
for name, log in RUNS.items():
    vals = [e["val_loss"] for e in log["evals"]]
    drops = sum(b < a for a, b in zip(vals, vals[1:]))
    print(f"{name}: validation loss fell at {drops} of {len(vals)-1} evaluations after conversion; "
          f"{vals[0]:.3f} -> {vals[-1]:.3f}")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ev = D["evals"]
for a in ax:
    a.plot([e["tokens"]/1e6 for e in ev], [e["val_loss"] for e in ev], "k-o", ms=3, label="dense (kept dense)")
    for name, log in RUNS.items():
        e2 = log["evals"]
        a.plot([e["tokens"]/1e6 for e in e2], [e["val_loss"] for e in e2], "-o", ms=3, label=f"MoE by {name}")
    a.axvline(CONVERT_AT/1e6, color="grey", ls=":"); a.grid(alpha=.3); a.set_xlabel("tokens (M)"); a.set_ylabel("validation loss")
ax[0].set(ylim=(1.4, 3.0), title="whole run"); ax[0].legend()
ax[1].set(xlim=(CONVERT_AT/1e6 - 1, 30.5), ylim=(1.68, 2.62), title="after conversion (dotted line)")
fig.tight_layout(); fig.savefig(ART / "loss_after_conversion.png", dpi=130); plt.show()
""")

md(r"""
## 4. Routing health after conversion

Every converted model uses the auxiliary-loss-free bias method: after each step, each expert's selection bias
moves ±0.001 towards an even load. Over each 50-step window the log records two things:
- **MaxVio**: (busiest expert's load − mean) / mean. 0 means a perfectly even load, 1 means twice the average.
- **Dead experts**: experts that got no tokens at all in the window.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
NUM["routing"] = {}
for name, log in list(RUNS.items()) + [("scratch", R("moe_scratch"))]:
    tr = log["train"]
    t = [r["tokens"]/1e6 for r in tr]
    mv = [np.mean(r["routing"]["maxvio"]) for r in tr]
    dead = [sum(r["routing"]["dead"]) for r in tr]
    ax[0].plot(t, mv, label=name); ax[1].plot(t, dead, label=name)
    NUM["routing"][name] = dict(maxvio_first=mv[0], maxvio_last=mv[-1], maxvio_max=max(mv), dead_max=max(dead), dead_last=dead[-1])
    print(f"{name:10s} mean MaxVio first window {mv[0]:.2f}, last {mv[-1]:.2f}, worst {max(mv):.2f};  "
          f"dead experts (of 80) worst {max(dead)}, last {dead[-1]}")
ax[0].set(xlabel="tokens (M)", ylabel="MaxVio, mean over layers", title="load imbalance"); ax[0].set_yscale("log")
ax[1].set(xlabel="tokens (M)", ylabel="experts with no tokens in the window (of 80)", title="dead experts")
for a in ax: a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ART / "routing_health.png", dpi=130); plt.show()
""")

md(r"""
## 5. Upcycled against trained from scratch

The same MoE shape (8 experts × 1,536, top-2) also trained from random weights for the full 30M tokens. There are
two fair ways to compare:
- **Same tokens:** every model sees 30M tokens.
- **Same compute:** training FLOPs ≈ 6 × active parameters × tokens. The upcycled models spent the first 15M tokens
  as the cheaper dense model.
""")

code(r"""
SC = R("moe_scratch")
dense_act, moe_act = NUM["params"]["dense"]["active"], NUM["params"]["copy / drop"]["active"]
def flops(log, dense_until=0):
    out = []
    for e in log["evals"]:
        t = e["tokens"]
        f = 6 * dense_act * min(t, dense_until) + 6 * moe_act * max(0, t - dense_until) if dense_until else 6 * (dense_act if log["moe"] is None else moe_act) * t
        out.append(f / 1e15)
    return out
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
curves = [("dense", D, 0)] + [(f"upcycled: {m}", RUNS[m], CONVERT_AT) for m in METHODS] + [("MoE from scratch", SC, 0)]
for name, log, du in curves:
    e = log["evals"]
    if du:
        e = [x for x in D["evals"] if x["tokens"] <= CONVERT_AT] + e
        f = [6 * dense_act * x["tokens"] / 1e15 for x in D["evals"] if x["tokens"] <= CONVERT_AT] + flops(log, du)
    else:
        f = flops(log)
    ax[0].plot([x["tokens"]/1e6 for x in e], [x["val_loss"] for x in e], "-o", ms=2, label=name)
    ax[1].plot(f, [x["val_loss"] for x in e], "-o", ms=2, label=name)
ax[0].set(xlabel="tokens (M)", ylabel="validation loss", ylim=(1.4, 3.0), title="same tokens")
ax[1].set(xlabel="training PFLOP (6 x active x tokens)", ylabel="validation loss", ylim=(1.4, 3.0), title="same compute")
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ART / "upcycled_vs_scratch.png", dpi=130); plt.show()
tot_up = (6 * dense_act * CONVERT_AT + 6 * moe_act * (30_000_000 - CONVERT_AT)) / 1e15
tot_sc = 6 * moe_act * SC["tokens_end"] / 1e15
NUM["scratch"] = dict(final_val=SC["final_val_loss"], pflop=tot_sc, upcycled_pflop=tot_up, tok_s=SC["tok_per_s"], minutes=SC["train_seconds"]/60)
best = min(METHODS, key=lambda m: RUNS[m]["final_val_loss"])
print(f"MoE from scratch: final val {SC['final_val_loss']:.4f} for {tot_sc:.2f} PFLOP")
print(f"best upcycled ({best}): final val {RUNS[best]['final_val_loss']:.4f} for {tot_up:.2f} PFLOP")
print(f"dense: final val {D['final_val_loss']:.4f} for {6*dense_act*D['tokens_end']/1e15:.2f} PFLOP")
""")

md(r"""
## 6. Growing again: 8 → 32 experts by cloning, and the clone-family collapse

The course describes a failure from Lightning LM's 20 → 460 expert growth. Each expert was cloned 23 times, the
router rows were tiled with small noise, and hard top-k was used. The dead experts went 27 → 38 → 122 → 154 → 168
of 460 within a few hundred steps. The router can't tell near-identical clones apart, so each clone family settles on
a few members and the rest never receive a token. The fix was sampled ("probabilistic") selection for an early
window, so every clone gets some gradient before the router settles.

Here the best converted MoE at 24M tokens is grown 8 → 32 experts:
- each expert is cloned 4 times, giving 8 families of 4;
- the router rows are tiled, with noise of std 1e-3;
- top-k stays 2, so the compute per token doesn't change.

It then trains 24M → 30M three ways. The ungrown MoE's own 24M → 30M is the baseline.

| run | selection | clones |
|---|---|---|
| grow_hard | hard top-2 throughout | exact copies |
| grow_sample | sampled top-2 for the first 1.5M tokens, then hard | exact copies |
| grow_redraw | hard top-2 throughout | 3 of each family's 4 have half their neurons re-initialised |
""")

code(r"""
src = best_at_grow(); BASE = RUNS[src]
G = {n: R(n) for n in GROW_RUNS}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
NUM["grow"] = dict(from_method=src)
for name, log in G.items():
    tr = log["train"]
    t = [r["tokens"]/1e6 for r in tr]; dead = [sum(r["routing"]["dead"]) for r in tr]
    ax[0].plot(t, dead, "-o", ms=2, label=name)
    e = log["evals"]; ax[1].plot([x["tokens"]/1e6 for x in e], [x["val_loss"] for x in e], "-o", ms=3, label=name)
    fam = np.array(tr[-1]["routing"]["dead_by_family"])           # (layers, 8 families): dead clones per family
    NUM["grow"][name] = dict(dead_max=max(dead), dead_last=dead[-1], final_val=log["final_val_loss"],
                             families_all_alive=int((fam == 0).sum()), families_one_left=int((fam == 3).sum()),
                             dead_curve=list(zip(t, dead)))
    print(f"{name:12s} dead experts (of 320): worst {max(dead)}, last window {dead[-1]};  families (of 80) with all 4 "
          f"clones alive {int((fam == 0).sum())}, with only 1 alive {int((fam == 3).sum())};  final val {log['final_val_loss']:.4f}")
eb = [x for x in BASE["evals"] if x["tokens"] >= GROW_AT - 1]
ax[1].plot([x["tokens"]/1e6 for x in eb], [x["val_loss"] for x in eb], "k--o", ms=3, label=f"not grown ({src}, 8 experts)")
NUM["grow"]["baseline_final_val"] = BASE["final_val_loss"]
print(f"{'not grown':12s} final val {BASE['final_val_loss']:.4f}")
ax[0].set(xlabel="tokens (M)", ylabel="experts with no tokens in the window (of 320)", title="dead experts after growth")
ax[1].set(xlabel="tokens (M)", ylabel="validation loss", title="loss after growth")
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ART / "clone_growth.png", dpi=130); plt.show()
""")

md(r"""
## 7. What did the experts learn?

The course's claim (Section 9) is that experts specialise by kind of token (punctuation, function words, names,
numbers), not by subject. TinyStories is one genre with no subjects, so only the token-type half can be tested here.

Every validation token is put in one of six types: punctuation, newline/end of text, number, capitalised word,
function word, other word. Three measures:
- **Token type.** How many bits a token's top-1 expert tells you about its type. Chance is measured by shuffling the
  types; the ceiling is the type's own entropy.
- **Neighbours.** How often the next token picks the same top-1 expert. The course quotes Mixtral at 24–28% against
  12.5% chance.
- **Expert similarity.** The mean cosine similarity between experts' weights. Copied experts start at exactly 1.
""")

code(r"""
cats, tok = A.token_categories()
NUM["specialisation"] = {}
spec_models = {f"upcycled ({best})": (moe_config(best), RES / f"moe_{best}.pt"), "from scratch": (moe_config("copy"), RES / "moe_scratch.pt")}
SPEC = {}
for name, (cfg, path) in spec_models.items():
    m = GPT(cfg); m.load_state_dict(torch.load(path, map_location="cpu")); m = m.to(DEV)
    toks, ch = A.route_validation(m, DEV)
    rows = A.specialisation(toks, ch, cats)
    sim = A.expert_similarity(m)
    SPEC[name] = (toks, ch)
    NUM["specialisation"][name] = dict(rows=rows, expert_cosine=sim)
    print(f"--- {name}")
    print(f"{'layer':>5s} {'type info (bits)':>16s} {'shuffled':>9s} {'max':>5s} {'same expert next':>17s} {'chance':>7s} {'expert cosine':>14s}")
    for r, c in zip(rows, sim):
        print(f"{r['layer']:5d} {r['mi_type']:16.3f} {r['mi_shuffled']:9.3f} {r['h_type']:5.2f} {r['same_next']:17.1%} {r['same_next_chance']:7.1%} {c:14.3f}")
    del m
""")

code(r"""
name = f"upcycled ({best})"; toks, ch = SPEC[name]
layer = int(np.argmax([r["mi_type"] for r in NUM["specialisation"][name]["rows"]]))
mix = A.type_mix(toks, ch, cats, layer)
overall = np.bincount(cats[toks], minlength=len(A.CATEGORIES)) / len(toks)
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(np.divide(mix, overall, out=np.ones_like(mix), where=overall > 0), cmap="RdBu_r", vmin=0, vmax=2, aspect="auto")
ax.set(xticks=range(len(A.CATEGORIES)), xticklabels=A.CATEGORIES, ylabel="expert",
       title=f"{name}, layer {layer}: share of each token type in an expert's tokens / overall share")
plt.setp(ax.get_xticklabels(), rotation=20); fig.colorbar(im, label="1 = no preference")
fig.tight_layout(); fig.savefig(ART / "expert_token_types.png", dpi=130); plt.show()
print(f"layer {layer}: each expert's most over-represented tokens (lift over overall frequency)")
for e, tt in A.top_tokens(toks, ch, tok, layer).items():
    print(f"  expert {e}: " + ", ".join(f"{t!r} x{l}" for t, l in tt))
NUM["specialisation_layer"] = layer
""")

md(r"""
## 8. Summary
""")

code(r"""
print(f"{'run':24s} {'tokens':>8s} {'total':>7s} {'active':>7s} {'final val':>10s} {'tok/s':>7s} {'peak GiB':>9s}")
rows = [("dense", D)] + [(f"MoE by {m}", RUNS[m]) for m in METHODS] + [("MoE from scratch", SC)] + [(n, G[n]) for n in GROW_RUNS]
for name, log in rows:
    print(f"{name:24s} {log['tokens_end']/1e6:7.1f}M {log['n_params']/1e6:6.1f}M {log['n_active']/1e6:6.1f}M "
          f"{log['final_val_loss']:10.4f} {log['tok_per_s']:7.0f} {log['peak_bytes']/GiB:9.2f}")
NUM["summary"] = {name: dict(tokens=log["tokens_end"], total=log["n_params"], active=log["n_active"], final_val=log["final_val_loss"],
                             final_train=log["final_train_loss"], tok_s=log["tok_per_s"], peak_gib=log["peak_bytes"]/GiB,
                             minutes=log["train_seconds"]/60) for name, log in rows}
(ART / "s14_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
print("saved", ART / "s14_numbers.json")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "s14-moe", "display_name": "s14-moe", "language": "python"}
out = Path(__file__).resolve().parent.parent / "s14_dense_to_moe.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
