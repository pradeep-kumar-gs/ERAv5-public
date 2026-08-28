# Loss Harness & Multi-Token Output Heads

This README is the primary artifact. Every number below is **measured** —
produced by executing `s9_loss_harness.ipynb` top to bottom, written to
`submission_artifacts/`, and never hand-edited into this file.

## What this proves, in one command

```bash
cd s9
python3 -m venv .venv
.venv/bin/pip install torch tokenizers matplotlib jupyter nbformat nbclient ipykernel numpy
.venv/bin/python -m ipykernel install --user --name s9-loss-harness --display-name "s9-loss-harness"
.venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=s9-loss-harness \
    --ExecutePreprocessor.timeout=600 \
    --output s9_loss_harness.ipynb s9_loss_harness.ipynb
```

No network calls beyond `pip install` — `data/tinyshakespeare.txt` is
committed to the repo, and the BPE tokenizer is trained from it locally each
run. This regenerates `submission_artifacts/`:

- `part1_numbers.json` — the seven Part 1 numbers below.
- `part2_losses.json` — the Part 2 loss history (`L1`, `L2` every 20 steps),
  final/first values, and the sum.
- `mtp_losses.png` — `L1` vs `L2` over 400 training steps.
- `artifacts/tokenizer-vocab.json`, `artifacts/tokenizer-merges.txt` — the
  trained byte-level BPE tokenizer (vocab size 6000).

Notebook: [`s9_loss_harness.ipynb`](s9_loss_harness.ipynb) — runs top to
bottom on CUDA, MPS (Apple Silicon), or CPU. Executed here on Apple Silicon
(`mps` backend).

## Model

A ~15M-parameter decoder-only transformer (`TinyGPT` in the notebook):
RMSNorm (pre-norm), SwiGLU feed-forward, causal self-attention, 6 layers,
6 heads, `d_model=384`, `block_size=1024`. Tokenizer: byte-level BPE trained
from scratch on `data/tinyshakespeare.txt`, vocab size 6000 — large enough
that the tied-vs-untied head comparison (#4/#5 below) is a real, visible
parameter delta rather than noise at char-level vocab sizes.

## Part 1 — the seven numbers

| # | Requirement | Where in notebook | Result |
|---|---|---|---|
| 1 | Shapes, one line each | 1.1 | `tokens (4,32)`, `hidden (4,32,384)`, `logits (4,32,6000)` — logits/hidden element ratio `15.6x == V/D` |
| 2 | Verify shift with strings, not ids | 1.2 | decoded `input[i]` vs `target[i]` printed side by side; correct-shift loss **8.7695** vs a deliberate zero-shift foil **8.7499** — both plausible scalars alone, only the decoded strings catch the bug |
| 3 | Padding mask changes contributing-token count | 1.3 | **14** contributing tokens with mask vs **24** without (10 pad positions excluded) |
| 4 | Doc-boundary masking, loss before/after | 1.4 | loss before masking boundary: **8.7375** nats; after: **8.7532** nats *(see note below)* |
| 5 | Untrained-model perplexity vs. vocab size | 1.5 | measured perplexity **6310.4** vs. vocab size **6000** (ratio to theoretical `ln(V)` loss: **1.0058**) |
| 6 | Tied vs. untied head parameter counts | 1.6 | tied: **13,319,040**; untied: **15,623,040**; difference: **2,304,000** `== vocab_size × d_model`; **14.7%** of the untied total |
| 7 | Peak memory: ordinary vs. chunked cross-entropy | 1.7 | ordinary: **3451.70 MB**; chunked (32×1024-token chunks): **480.39 MB**; ratio **7.19x**; losses identical to 1e-10 (`8.6995220184` both) |

*(Full JSON: [`submission_artifacts/part1_numbers.json`](submission_artifacts/part1_numbers.json).)*

**Note on #4** — the sign of "before vs. after" depends on whether the
model happens to find the cross-document token more or less surprising than
its in-document neighbors on that specific random init; it is not
guaranteed to always move in one direction on an untrained model. What's
guaranteed and what matters is that the position count driving the mean
changes by exactly one (29 → 28 positions in-notebook, `<eos>` sits at
position 14 of the packed sequence), i.e. that one position's own NLL term
is removed rather than silently kept in — the same mechanism as #3.

## Part 2 — the extra head (predict `t+2`)

One trunk, two linear heads reading the same hidden state `h_t`: `head`
predicts `t+1` (as in Part 1), `head2` predicts `t+2`. Both losses are
computed against real tokens (never against each other's predictions) and
summed for backprop: `L = L1 + L2`. Trained 400 steps, AdamW, `lr=3e-4`,
block size 256, on the same tiny-Shakespeare corpus.

| | L1 (predict t+1) | L2 (predict t+2) | sum |
|---|---|---|---|
| step 1 (start) | 8.3386 | 8.3881 | 16.7267 |
| step 400 (final) | **4.7478** | **5.4469** | **10.1948** |
| improvement | 3.5908 nats | 2.9412 nats | — |

Gap `L2 − L1`: **+0.0495** nats at step 1 → **+0.6991** nats at step 400.

*(Full history: [`submission_artifacts/part2_losses.json`](submission_artifacts/part2_losses.json); plot: [`submission_artifacts/mtp_losses.png`](submission_artifacts/mtp_losses.png).)*

**What happens to L2 relative to L1, and why.** At initialization the gap is
noise — neither head has learned anything. As training proceeds, `L1` drops
faster than `L2`, and the gap *widens* rather than closing, stabilizing
around +0.6–0.7 nats by step 400. Both heads read off the identical hidden
state `h_t`; during training neither head ever sees the other's prediction,
only real ground-truth tokens as context. `head` only has to answer "what
comes right after this context," and the hardest part of that answer is
already implicit in `h_t`. `head2` has to answer a strictly harder question
from the *same* `h_t`: "what comes after that, without being told what comes
right after this" — it has to implicitly marginalize over the unknown `t+1`
using only information available at `t`, which is strictly less than what
`head` effectively gets to condition on. The gap is the measured price of
that missing intermediate token, and it matches this course's own finding
(loss climbing with prediction distance across MTP heads).

## Reproducing

```bash
cd s9
python3 -m venv .venv && .venv/bin/pip install torch tokenizers matplotlib jupyter nbformat nbclient ipykernel numpy
.venv/bin/python -m ipykernel install --user --name s9-loss-harness --display-name "s9-loss-harness"
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=s9-loss-harness --ExecutePreprocessor.timeout=600 --output s9_loss_harness.ipynb s9_loss_harness.ipynb
```

Total runtime: ~4 minutes on Apple Silicon MPS (tokenizer training + all
Part 1 checks + 400-step Part 2 training loop).

## Files

- `s9_loss_harness.ipynb` — the notebook, executed with real outputs.
- `assignment.md` — the assignment text.
- `data/tinyshakespeare.txt` — training corpus (committed, no network needed at run time).
- `artifacts/` — trained BPE tokenizer files (regenerated each run).
- `submission_artifacts/` — measured JSON/PNG outputs (regenerated each run).
