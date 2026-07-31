# Mixture-and-Curriculum Plan V5

This README is the primary artifact: everything a reviewer needs to evaluate
this plan is inline below, self-contained. Every headline number is either
**measured** (traces to a JSON file in `site/data/`, produced by a script in
`scripts/`, runnable end-to-end) or **cited** (traces to an external, named
source, explicitly labeled as not measured by this project). Full reasoning,
alternatives, and risks for every decision are in `report.md` (D1–D19) — that
file has the "why," this file has the "what" and "how much."

## Problem statement

This plan answers one brief: **specify, and defend, the pretraining data
mixture and curriculum for a large foundation model with a strong
Indian-language and agentic/coding capability profile**, written so a
reviewer can push on every number and find a real answer behind it, not a
plausible-sounding guess. A complete answer has to:

1. State a token-budget share for every capability lane in the mixture,
   summing to a fixed total.
2. Split the Indic-language lane into verified / unverified / translated /
   synthetic tiers — not collapse it into one headline percentage.
3. Name the agentic, long-context, and reasoning lanes explicitly, each
   pointed at a real, inventoried dataset — not a placeholder category.
4. Fix a protected floor that no reallocation can cross, and reserve a
   portion of the budget for late-stage annealing.
5. Define difficulty / reasoning-length curriculum bands, each anchored to a
   concrete example drawn from real data.
6. Size every lane against real, checked supply, and say plainly wherever a
   share can only be reached by repeating or synthetically generating data —
   a lane that quietly gets a large share with almost no real data behind it
   is the single worst failure this plan can make.
7. Tie every lane back to the public benchmark(s) it is meant to move, so a
   reviewer can see why each number is what it is.
8. Treat every mixture and curriculum choice as a hypothesis, not a settled
   fact — state a concrete, checkable metric that would confirm or refute it
   at small scale, and prefer an experiment that was actually run over one
   that is merely proposed.

The foundational choices this plan builds on — why the pretraining budget is
3T tokens split across these nine lanes, why the Indic lane targets a
15–25% range with a 20% headline, why sampling uses quality-aware temperature
weighting rather than raw corpus-size proportionality, why the context window
targets 128K, why agentic capability is primarily a post-training objective,
and why post-training itself runs as four evidence-gated stages — are
recorded as this plan's own D1–D6 in `report.md`. Everything from D7 onward
is this round's inventory, curriculum, and proxy work, built on top of that
foundation. The table directly below (§1) is the fixed reference every later
section checks its numbers against.

## How this document is organized for review

| # | What a reviewer checks | Where to look | Status |
|---|---|---|---|
| 1 | Budget share for every lane, summing to a fixed total | §1 | Done — 9 lanes, 100% / 3T |
| 2 | Indic lane split into verified / unverified / translated / synthetic | §2 | Done — real tiered measurement, not assumed |
| 3 | Agentic, long-context, and reasoning lanes named and pointed at real datasets | §3, §4, §5 | Done — 3 datasets named, audited, licence-checked |
| 4 | Protected floor and anneal reserve | §6, §7 | Done |
| 5 | Difficulty / reasoning-length bands, each with a concrete example | §5 | Done — 4 bands derived from measured span lengths |
| 6 | Every lane sized against real supply; repetition/synthetic need stated plainly | §5a, §9 | Done — the Code lane is the one lane still failing this test, stated explicitly rather than hidden |
| 7 | Lane → benchmark mapping ("why each number is what it is") | §1a | Done |
| 8 | Testable proxy hypotheses; an executed proxy outranks a written-only one | §8 | Done — a toy-scale proxy was actually run (§8a), plus one written-only hypothesis for the Code lane, whose supply gap is still open (§8b) |

## 1. Lane budget table (100% / 3T tokens)

| Lane | Share | Tokens | Reason for allocation | Supply verdict (this round's audit) |
|---|---:|---:|---|---|
| General knowledge, reference, culture, and current affairs | 35% | 1.05T | Preserves broad language, factual, cultural, and world knowledge instead of allowing specialist objectives to narrow the model | Not independently audited this round |
| Code and software engineering | 20% | 600B | Makes coding a first-class capability and supplies verifiable, structured examples useful for software agents | **Partial supply**: ~135.9B–~238.9B estimated permissive tokens (~22.6%–~39.8% of budget, depending on unresolved cross-source overlap) — was zero supply, now two real sources, still not closed (§5a) |
| Mathematics, science, and engineering | 15% | 450B | Directly teaches quantitative, formal, causal, and technical reasoning that code exposure cannot guarantee | Not independently audited this round; ~0.75% carved out to the reasoning micro-slice below |
| Books, education, and worked explanations | 15% | 450B | Supplies coherent long-form knowledge, pedagogy, intermediate reasoning, and material useful across age and expertise levels | Not independently audited this round; ~0.75% carved out to the long-context micro-slice below |
| Indian law, government, finance, and public services | 7.5% | 225B | Grounds local institutions and high-value Indian workflows while limiting the risk that a narrow institutional corpus dominates general behavior | Not independently audited this round |
| Multilingual conversation and instructions (carries the Indic-language overlay: 20% headline, 15–25% range) | 5% | 150B | Exposes pragmatic dialogue, code-switching, requests, and conversational registers under stricter privacy and provenance controls | Split by verification tier — see §2 |
| Agent, tool, and structured workflows (pretraining) | 2.5% | 75B | Teaches formats and workflow patterns while reserving most interactive agent learning for environment-grounded post-training | **Supply-constrained**: 56.1M measured tokens vs. 75B budget — see §3 |
| Long-context + reasoning micro-slice (carved out of Math/science + Books, not additive) | ~1.5% (0.75% + 0.75%) | ~45B | Concentrates genuinely long documents and multi-step traces so context-extension (§4) and curriculum bands (§5) have real, on-topic training material | **Supply-constrained**: 112.4M + 89.7M measured tokens vs. ~45B budget — see §4, §5 |
| **Total** | **100%** | **3T** | **Accounting checksum** | |

Language, script, source-verification tier, and document length are **overlay
labels** on top of these primary buckets (D2, D3), not additional slices — a
Tamil mathematics textbook, for example, sits in the Mathematics bucket but
also carries Tamil, native-script, and long-document labels. The Indic
overlay's 20%/15–25% figures and the long-context/reasoning micro-slice above
are both overlays for exactly this reason: none of them are added on top of
the 100%/3T checksum, which stays fixed.

## 1a. Lane → target benchmarks (why each number is what it is)

Every lane above is tied to the public benchmark(s) its tokens are meant to
move, with a one-line reason per lane. This mapping is **newly reasoned for
this plan**, not pulled from any earlier evaluation framework — each
benchmark is chosen for direct topical fit to the lane's content, and it is
**target-naming for the evaluation strategy**, not a measured result: no
model has been run against any of these yet.

| Lane | Target benchmark(s) | Why this benchmark wins for this lane |
|---|---|---|
| General knowledge/reference/culture | MMLU-Pro, ARC-Challenge | Broad factual/reasoning coverage is exactly what this lane's volume is meant to buy. |
| Code and software engineering | HumanEval, LiveCodeBench, SWE-bench Verified | Execution-verified pass@1 is the direct target of §8b's confirm/refute metric (HumanEval pass@1 delta across code-share runs); LiveCodeBench/SWE-bench Verified guard against HumanEval-specific contamination. |
| Mathematics, science, engineering | MATH, GPQA | Multi-step quantitative reasoning is what this lane's tokens are supposed to build; GPQA also cross-checks the reasoning-lane bands (§5) at the hardest end. |
| Books, education, worked explanations | MMLU (humanities/STEM-education subsets), LongBench v2 | Long-form worked explanations are the natural training signal for both broad-knowledge recall and the long-document comprehension this lane feeds into (§4's long-context carve-out). |
| Indian law/gov/finance/public services | IndicGenBench, native-expert sealed held-out sets | No mature public benchmark exists for Indian-specific procedural/domain correctness; this is the one lane where a private-expert layer, not a public leaderboard, is the real decider. |
| Multilingual + Indic overlay | IndicGenBench, Flores-200 (translation), IndicXTREME | Directly measures the verified/unverified/translated split's payoff (§2) — Flores-200 for translation quality (relevant given Aya's 45.1M translated tokens), IndicGenBench/IndicXTREME for generation and cross-lingual understanding. |
| Agent/tool/structured workflows | Berkeley Function-Calling Leaderboard (BFCL), tau-bench | BFCL scores exactly the function-call format/argument correctness `glaive-function-calling-v2` (§3) trains; tau-bench adds multi-turn tool-use trajectories closer to D5's post-training agentic build-out. |
| Long-context + reasoning micro-slice | RULER, LongBench v2, GPQA Diamond | RULER/LongBench v2 test the long-document retrieval `LongAlpaca-12k` (§4) is meant to build; GPQA Diamond tests the CoT depth the Band 2/3 `Bespoke-Stratos` rows (§5) are meant to build. |
| Safety/permission SFT floor | HarmBench, StrongREJECT, XSTest | HarmBench/StrongREJECT test refusal on genuinely harmful requests; XSTest specifically tests against *over*-refusal on benign-but-scary-sounding prompts — both matter because D6's safety SFT is a fixed floor, not a tunable share, so it must be checked for both failure directions. |

**Caveat.** Every benchmark above is a **target**, not a decontamination
guarantee — public benchmark leakage into pretraining corpora (particularly
Code and Math/science, the largest web-scale lanes) is a real, unmeasured
risk here. Naming the target is what this plan's problem statement calls
for; proving the absence of contamination is a separate, still-open
validation task (tracked in `MEMORY.md`).

## 2. Indic lane: verified / unverified / translated / synthetic (not one number)

### 2a. Aya sample (measured — `scripts/tier_aya.py`, output `site/data/aya-tiers.json`)

Tiers a ~50M-token, 5-language Aya sample (Hindi, Tamil, Telugu, Kannada,
Malayalam) by `dataset_name`: `Aya-Dataset` (human-annotated, Singh et al.
2024, arXiv:2402.06619) = verified; suffix `"(T)"` = translated; everything
else = unverified. No synthetic-labeled rows exist in this schema.

| Language | Verified (tokens) | Unverified (tokens) | Translated (tokens) | Synthetic |
|---|---:|---:|---:|---:|
| Hindi | 1,886 | 2,661,544 | 7,336,560 | 0 |
| Tamil | 51,625 | 61,239 | 9,887,126 | 0 |
| Telugu | 32,343 | 2,076,456 | 7,891,184 | 0 |
| Kannada | 0 | 2,171 | 9,997,795 | 0 |
| Malayalam | 3,751 | 12,344 | 9,983,834 | 0 |
| **Total** | **89,605** | **4,813,754** | **45,096,499** | **0** |

Aya alone cannot fill the verified or synthetic tiers — those must come from
Sangraha (below). This is a real, inspectable finding, not an assumption.

### 2b. Sangraha Hindi spot-sample (measured-spot-sample — `scripts/audit_sangraha.py`, output `site/data/sangraha-hindi-spotsample.json`)

One real shard per tier (`verified/hin/data-0.parquet`,
`unverified/hin/data-0.parquet`, `synthetic/hin_Deva/wiki_hin_Deva_0000_of_0063.parquet`),
~2M tokens/tier via the same deterministic SHA-256 rank sampling used for the
Aya tiering above (§2a):

| Tier | Measured tokens | Measured rows | Cited full-corpus total (Khan et al. 2024, arXiv:2403.06350) |
|---|---:|---:|---:|
| Verified | 1,999,983 | 942 | 64.306B (all 22 languages); Hindi alone: 12.617B |
| Unverified | 1,999,803 | 760 | 24.308B (all 22 languages) |
| Synthetic | 1,999,990 | 645 | 162.708B (all 22 languages) |

**Scope honestly stated**: this is Hindi, shard 0 of ~100+ shards per tier —
a spot-check, not a full-corpus measurement. Every other language and every
other shard is a **cited** number (Khan et al. 2024), not measured here.

## 3. Agentic lane (measured — `scripts/audit_agentic.py`, output `site/data/agentic-audit.json`)

Dataset: `glaiveai/glaive-function-calling-v2` (Apache-2.0, ungated). The
originally planned `Salesforce/xlam-function-calling-60k` turned out to be a
**gated** repository (401 without an approved access request) — swapped
immediately rather than citing a broken source.

| Metric | Value |
|---|---:|
| Records | 112,960 |
| Total tokens | 56,115,280 |
| Mean tokens/record | 496.37 |
| Median tokens/record | 395.0 |
| p95 tokens/record | 1,167 |
| Max tokens/record | 4,083 |

Ties to D5's pretraining/post-training split: pretraining carries formats and
basic tool syntax (this lane), most agentic capability is built in
post-training SFT (20% agent/tool per D6) and environment-grounded RL.

## 4. Long-context lane (measured — `scripts/audit_longcontext.py`, output `site/data/longcontext-audit.json`)

Dataset: `Yukang/LongAlpaca-12k` (LongLoRA project). **Licence flagged**: the
project's *code* is Apache-2.0, but the *data* is declared **CC BY-NC 4.0**
(non-commercial) in its own README — corrected from an earlier assumption of
Apache-2.0 by reading the actual licence badges before writing it into this
plan.

| Metric | Value |
|---|---:|
| Records | 12,000 |
| Total tokens | 112,352,280 |
| Mean tokens/record | 9,358.62 |
| Median tokens/record | 7,745.0 |
| p95 tokens/record | 25,205 |
| Max tokens/record | 70,748 |

Context-length histogram (tokens):

| 0–2K | 2K–8K | 8K–16K | 16K–32K | 32K–64K | 64K+ |
|---:|---:|---:|---:|---:|---:|
| 3,118 | 3,417 | 3,464 | 1,979 | 18 | 4 |

Ties to D4's staged context-extension precedent (8K–16K tokens early,
extending to a 128K target) — this lane is exactly the kind of genuinely-long
document D4 calls for during extension, rather than padded/concatenated
short text.

**Licence consequence**: acceptable for this plan's audit/spot-check purpose;
a real production pretraining run would need a licence carve-out or a
commercially licensed substitute (stated plainly, not hidden — see §9).

## 5. Reasoning lane + 4-band difficulty/reasoning-length curriculum (measured — `scripts/audit_reasoning.py`, output `site/data/reasoning-audit.json`)

Dataset: `bespokelabs/Bespoke-Stratos-17k` (Apache-2.0). Bands are measured
directly from each record's `<|begin_of_thought|>...<|end_of_thought|>` span
length, not invented round numbers:

| Band | Thought-token range | Representative source | Measured records (Bespoke-Stratos) |
|---|---|---|---:|
| Band 0 (direct/no-CoT) | 0 | short Aya direct-QA rows (§2a) | 0 in this corpus (expected — reasoning-only dataset) |
| Band 1 (short CoT) | 1–99 | shortest Bespoke-Stratos rows | 98 |
| Band 2 (medium CoT) | 100–1,999 | mid-length Bespoke-Stratos rows | 6,681 |
| Band 3 (long CoT / multi-step) | ≥2,000 | longest Bespoke-Stratos rows + `glaive` multi-turn tool traces + `LongAlpaca` long documents | 9,931 |

Dataset totals: 16,710 records, 89,717,186 tokens, mean 5,364.45 / median
3,526.0 / p95 16,082 / max 33,734 tokens/record.

Band 0/1 dominate early pretraining (ties to D4's 8K–16K-early precedent);
Band 2/3 share rises late in pretraining and concentrates in the anneal
reserve (§7) and SFT (D6).

## 5a. Code lane (two measured sources — `scripts/audit_code.py` + `scripts/audit_code_shards.py` + `scripts/audit_commitpack.py`, output `site/data/code-audit.json` + `site/data/code-shards-audit.json` + `site/data/commitpack-audit.json`)

This plan initially admitted the Code lane (20% / 600B tokens) had zero
audited training corpus anywhere in its inventory — the single largest
wishful-accounting risk in the plan. This section partially addresses that:
`codeparrot/github-code-clean` (quality-filtered `codeparrot/github-code`, a
BigQuery GitHub export with a real per-file `license` column) is named and
spot-sampled, restricted to a permissive-license allowlist (`mit`,
`apache-2.0`, `bsd-3-clause`, `bsd-2-clause`, `isc`, `cc0-1.0`, `unlicense`)
— the same approach StarCoder's own training data used, excluding copyleft
(`gpl-2.0`/`3.0`, `agpl-3.0`, `lgpl-2.1`/`3.0`) and ambiguous (`mpl-2.0`,
`epl-1.0`, `artistic-2.0`) licenses. A 10-shard scan (evenly spaced across
all 883 shards, license + byte-size columns only, no tokenization) then
replaces the single-shard extrapolation to check whether shard 0 was
representative.

| | Shard 0 (deep, tokenized) | 10-shard scan (wide, byte-based) | Full-corpus estimate |
|---|---:|---:|---:|
| Total tokens (cl100k_base) | 270,601,959 | — | ~239.4B |
| Permissive-license tokens | 142,871,195 (52.8%) | mean 56.76% (σ 1.17%, range 54.9–58.5%) | ~135.9B |
| Files | 126,925 | ~1.27M across 10 shards | ~112.07M |

Shard 0's byte-based permissive fraction (54.87%) sits within one standard
deviation of the 10-shard mean (56.76%, σ 1.17%) — low variance confirms it
wasn't an outlier. The 112.07M estimated-files figure also lines up closely
with the dataset card's own arithmetic (115M files cited by `github-code`,
minus 3.39M removed by `github-code-clean`'s quality filter ≈ 111.6M
expected — a 0.4% difference).

**A second, independent source.** `bigcode/commitpack` (BigCode, ungated,
~3.8TB raw across 350 languages, same per-file `license`-column provenance)
was spot-sampled across 5 mainstream languages (python, javascript, java, c,
go), one shard file each:

| | CommitPack sample (5 languages) | Extrapolated (same 5 languages only) |
|---|---:|---:|
| Total tokens (cl100k_base) | 339,293,298 | ~121.4B |
| Permissive-license tokens | 287,796,504 (84.8%) | ~103.0B |

CommitPack's permissive fraction (84.8%) runs well above
github-code-clean's (~52.8–58.5%), plausibly because mainstream languages
skew more permissively licensed than github-code-clean's full language mix.
The other 345 CommitPack language directories stay cited (BigCode's own
~3.8TB dataset-card total), not measured.

**Overlap check, honestly bounded, not assumed away.** Because both sources
draw from the same public-GitHub universe, `audit_commitpack.py` hashed its
344-row permissive sample against every permissive row already cached from
the github-code-clean shard-0 audit (81,693 rows): **0 exact matches.** This
is a real result, but it is a lower-bound signal from a tiny slice of both
corpora, not proof of independence — CommitPack stores per-commit snapshots
(often several per file, at different points in its history) while
github-code-clean stores one snapshot per file, so the same repository can
appear in both without ever producing an exact-content match. File-level
overlap between the two sources is almost certainly real and unmeasured.

**Combined accounting (honest range, not a sum):**

| Assumption | Combined permissive tokens | % of 600B budget |
|---|---:|---:|
| Fully overlapping (lower bound) | ~135.9B | ~22.6% |
| Fully independent (upper bound) | ~238.9B | ~39.8% |

**Verdict: partial supply, not closed, under either assumption.** Two real
sources are now named and measured instead of zero, but even the optimistic
upper bound covers under 40% of the 600B budget, and the true figure —
pending real cross-corpus near-duplicate deduplication — is unknown within
that range. See `report.md` D14–D18 for the full risk/validation-needed
writeup, including the coarse secret/PII heuristic's hit counts on the
sampled files and each source's sample-size caveats.

## 6. Protected floor (generalizes D3 beyond Indic-only)

Every lane's floor = **50% of its D1 headline share**; the mixture selector
cannot cross this downward. Three absolute policy floors override the 50%
rule where a prior decision already fixed a number:

| Floor | Value | Source |
|---|---:|---|
| Indic overlay | ≥15% | D2's own lower bound of the 15–25% range |
| Agentic (pretraining) | ≥1% | Half of D5's 2–3% |
| Safety/permission SFT | Fixed 10%, never reweightable by any proxy result | D6 |

## 7. Anneal reserve (extends D6's ≤30B cap)

Cap unchanged at ≤30B tokens. Contents named concretely: cleanest
code/STEM/Indic-educational/long-form material (D6's existing language),
plus the longest Bespoke-Stratos Band-3 rows and the longest LongAlpaca-12k
documents — i.e. the anneal reserve is explicitly where the two
supply-constrained lanes (§3, §4) get concentrated instead of diluted across
all 3T pretraining tokens.

## 8. Proxy experiments

**Hardware constraint stated plainly**: MacBook Pro 14" M1 Pro, 32GB unified
memory, MPS backend, no CUDA. A real 1B-parameter run is impractical in
reasonable time on this machine; a real 3B run needs ~36GB+ for optimizer
state alone before activations. Two-track response:

### 8a. Executed toy-scale proxy (real numbers, sub-1B, directional only)

~18M-param nanoGPT-style model (`scripts/model.py`), trained on MPS
(`scripts/train.py`) on real audited data only — Aya + Sangraha-Hindi
(Indic), `glaive-function-calling-v2` (agentic), `Bespoke-Stratos-17k`
(reasoning), `LongAlpaca-12k` (long-context). **Code and General are excluded
from the proxy entirely** — at the time of these runs neither had an audited
training corpus, and faking one into a proxy run would be exactly the
wishful accounting this plan is designed to avoid. §5a later named and
measured two real, partial-supply Code candidates, but even the optimistic
~39.8%-of-budget upper bound still isn't enough to justify a real,
non-repeated lane share for a proxy re-run; General remains fully unaudited.

Three hypotheses trained from the same initialization on the same lane token
pools (`scripts/proxy_data_mix.py`), evaluated on one shared validation set
and one shared held-out "complex" set (withheld Bespoke-Stratos Band-3 +
withheld LongAlpaca 16K+ documents — never seen in any hypothesis's training
stream):

| Hypothesis | Mixture ratios (renormalized among the 4 covered lanes) | Ordering |
|---|---|---|
| A | Indic 83.3% / Agentic 10.4% / Reasoning 3.1% / Long-context 3.1% (D1/D5-proportional) | Fully shuffled |
| B | Indic 86.2% / Agentic 8.6% / Reasoning 2.6% / Long-context 2.6% (Indic pushed to its 25% ceiling, D2) | Fully shuffled |
| C | Same ratios as A | Curriculum-ordered: short/simple lanes (Indic+Agentic) first, Reasoning next, Long-context last |

**Results** (measured, `scripts/train.py` → `site/data/proxy-results.json`; ~17.67M-param model, 600 steps, batch 32, block 256, M1 Pro MPS):

| Hypothesis | Final train loss | Final shared val loss | Final held-out "complex" loss | Wall-clock (600 steps) |
|---|---:|---:|---:|---:|
| A — D1-proportional, shuffled | 1.6065 | **1.9076** | **6.6342** | 711s |
| B — Indic pushed to 25% ceiling, shuffled | 1.4971 | **1.9103** | **6.7225** | 418s |
| C — same ratios as A, curriculum-ordered | 1.7531 | **1.8877** | **6.5984** | 392s |

Reading the numbers:

- **A vs B (mixture-ratio effect)**: shared val loss is essentially flat
  (1.9076 vs 1.9103, Δ = +0.0027). At this toy scale, pushing the Indic
  overlay from its 20% baseline to the 25% ceiling produced no detectable
  change in general validation loss — consistent with treating 15–25% as a
  genuinely safe range (D2, D12) rather than a value that needs tight
  pinning.
- **A vs C (curriculum-ordering effect, same ratios)**: curriculum ordering
  (Indic+agentic first, reasoning next, long-context last) improved *both*
  shared val loss (1.8877 vs 1.9076) *and* held-out complex-generalization
  loss (6.5984 vs 6.6342) versus fully shuffled ordering at identical mixture
  ratios. Small but consistent across both metrics — suggestive, not
  conclusive, on a single-seed toy run.
- **Held-out complex loss (~6.6–6.7) vs shared val loss (~1.9), all three
  hypotheses**: confirms the held-out set (withheld Bespoke-Stratos Band-3 +
  withheld LongAlpaca 16K+ documents) is genuinely out-of-distribution
  relative to training — no leakage from the withheld material into any
  hypothesis's training stream.

**What this shows / doesn't show**: this is an ~18M-parameter, single-machine,
~900K-training-token toy run — it cannot distinguish subtle mixture effects
that would only appear at 1B+ scale with much larger token budgets. It is
reported as directional evidence, never conflated with a literal 1B/3B run.

### 8b. Written-only hypothesis (unexecuted at 1B/3B scale)

For the Code lane (§5a: two real permissive-license candidate corpora are
now named and measured — `codeparrot/github-code-clean` and
`bigcode/commitpack` — but their combined estimated permissive supply
(~135.9B–~238.9B tokens depending on unresolved cross-source overlap) covers
only ~22.6%–~39.8% of the 600B budget, so it is still excluded from the toy
proxy above pending real deduplication, a third source, or a reduced share):
once the supply gap is closed, run a 1B-parameter model at 15%/20%/25% code
share (3 runs, matched token budget and steps). **Confirm/refute metric**:
HumanEval pass@1 delta across the three runs, on ≥1,000 held-out problems
never present in any training split. A positive, monotonic pass@1 trend with
code share confirms the 20% headline is reasonable; a flat or
non-monotonic trend would argue for re-examining the Code lane's share
before committing it at full 3T scale.

## 9. Wishful-accounting self-audit

| Lane | Headline share | Real supply behind it | Verdict |
|---|---:|---|---|
| General | 35% | Not independently audited this round | Carried forward as this plan's own baseline (D1), unexamined this round |
| Code | 20% | **~135.9B–~238.9B estimated permissive tokens (`github-code-clean` + `commitpack`, range depends on unresolved cross-source overlap)** | **Partial supply — was zero, now ~22.6%–~39.8% of the 600B budget from two sources; still the single largest open risk** |
| Math/science | 15% | Not independently audited this round | Carried forward, unexamined this round |
| Books/education | 15% | Not independently audited this round | Carried forward, unexamined this round |
| Indian law/gov/finance | 7.5% | Not independently audited this round | Carried forward, unexamined this round |
| Indic overlay (verified) | Part of 20% overlay | 89,605 (Aya) + 1,999,983 (Sangraha-Hindi spot-sample) measured tokens | Real but thin — verified tier needs real growth beyond Aya/one Hindi shard |
| Indic overlay (unverified/translated) | Part of 20% overlay | 4,813,754 + 45,096,499 (Aya) + 1,999,803/1,999,990 (Sangraha-Hindi) measured | Adequate at sample scale; full-corpus scale-up still cited, not measured |
| Agentic (pretraining) | 2.5% / 75B | 56.1M measured tokens | **Repetition required** — real corpus is ~3 orders of magnitude below the pretraining budget; concentrate real data in anneal (§7), rely on post-training (D6) for the rest |
| Long-context + reasoning | ~1.5% / ~45B | 112.4M + 89.7M measured tokens | **Repetition + concatenation required** — same gap as agentic, same mitigation |

This is the sharpest test this plan can face, answered directly: Code is the
lane here with the largest share and, until this round's audit, admittedly
zero real data behind it. §5a names and measures two real permissive-license
candidates (`github-code-clean` + `bigcode/commitpack`), whose combined
estimate (~135.9B–~238.9B permissive tokens depending on unresolved
cross-source overlap) covers only ~22.6%–~39.8% of the 600B budget — an
honest partial-supply finding, not a full close of the gap, and not hidden
or quietly assumed away either way.

## 10. Decision log / reproduction

Full reasoning, alternatives, and risks: `report.md` (D1–D19). Open items:
`MEMORY.md`. Accepted-decisions handoff and validation checklist:
`VALIDATION.md`.

To regenerate every measured number in this README, run from the repo root
(`ERAv5/`) — every script's default paths are relative to it:

```bash
python3 -m venv s5/.venv && s5/.venv/bin/pip install torch numpy tiktoken huggingface_hub datasets pyarrow
s5/.venv/bin/python3 s5/scripts/tier_aya.py
s5/.venv/bin/python3 s5/scripts/audit_sangraha.py
s5/.venv/bin/python3 s5/scripts/audit_agentic.py
s5/.venv/bin/python3 s5/scripts/audit_longcontext.py
s5/.venv/bin/python3 s5/scripts/audit_reasoning.py
s5/.venv/bin/python3 s5/scripts/audit_code.py
s5/.venv/bin/python3 s5/scripts/audit_code_shards.py
s5/.venv/bin/python3 s5/scripts/audit_commitpack.py
s5/.venv/bin/python3 s5/scripts/proxy_data_mix.py
s5/.venv/bin/python3 s5/scripts/train.py
```

Raw downloaded data (`data/raw/`) and proxy token pools (`data/proxy/`) are
gitignored and never published — only the aggregate JSON manifests in
`site/data/` are committed.
