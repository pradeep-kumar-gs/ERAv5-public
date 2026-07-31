# Decision Log

Full decision log for the mixture-and-curriculum plan in `README.md`. Every
number in `README.md` traces to one of the decisions below, and every
decision below traces to either a JSON file in `site/data/` (measured) or an
explicit citation (cited, not measured — labeled as such).

## D1 — Adopt a balanced primary 3T sampling mixture

**Status.** Accepted 2026-07-24 as the baseline recipe for proxy testing.
Proxy results may change shares before a full-scale run, but any change must
be recorded with its measured capability gain and displaced-data cost.

**Decision.** Use the following mutually exclusive planning buckets across a
3T-token pretraining budget:

| Primary sampling bucket | Share | Exposures | Reason for allocation |
|---|---:|---:|---|
| General knowledge, reference, culture, and current affairs | 35% | 1.05T | Preserves broad language, factual, cultural, and world knowledge instead of allowing specialist objectives to narrow the model |
| Code and software engineering | 20% | 600B | Makes coding a first-class capability and supplies verifiable, structured examples useful for software agents |
| Mathematics, science, and engineering | 15% | 450B | Directly teaches quantitative, formal, causal, and technical reasoning that code exposure cannot guarantee |
| Books, education, and worked explanations | 15% | 450B | Supplies coherent long-form knowledge, pedagogy, intermediate reasoning, and material useful across age and expertise levels |
| Indian law, government, finance, and public services | 7.5% | 225B | Grounds local institutions and high-value Indian workflows while limiting the risk that a narrow institutional corpus dominates general behavior |
| Multilingual conversation and instructions | 5% | 150B | Exposes pragmatic dialogue, code-switching, requests, and conversational registers under stricter privacy and provenance controls |
| Agent, tool, and structured workflows | 2.5% | 75B | Teaches formats and workflow patterns while reserving most interactive agent learning for environment-grounded post-training |
| **Total** | **100%** | **3T** | **Accounting checksum** |

Language, script, source risk, quality, document length, and secondary
domains remain overlay labels layered on top of these buckets, not
additional slices — see D2 for the Indic overlay specifically. A Tamil
mathematics textbook, for example, occupies one primary bucket (Mathematics,
science, and engineering) but also carries Tamil, education, native-script,
and long-document labels.

**Why.** The mixture gives 35% of compute to the two explicitly targeted
specialties of code and STEM, retains 50% for broad knowledge and
educational depth, assigns a meaningful India-specific institutional
allocation, and keeps static agent traces deliberately small. It is balanced
enough to support a general foundation model while making its specialist
goals visible in the training budget.

**Alternatives considered.** A web-heavy mixture would maximize breadth but
weaken specialist capability and quality. A code/STEM-majority mixture could
score well on narrow tasks but reduce cultural, linguistic, and
conversational competence. A larger agent bucket would rely excessively on
synthetic static traces instead of verified interaction.

**Risks.** Bucket boundaries are editorial: books can contain science and
code, and public-service material can be educational. Forcing one primary
label may hide these intersections. The 7.5% institutional allocation can
overexpose duplicated legal templates or outdated policy. At fixed compute,
every increase displaces another capability.

**Validation needed.** Define deterministic primary-bucket precedence,
publish overlay distributions, audit duplication and date coverage, and run
mixture ablations at smaller scales. Reallocate only when measured marginal
capability gain per training token justifies the trade-off.

## D2 — Reserve 15–25% of natural-language exposure for Indic languages; use 20% as the headline target

**Status.** Accepted 2026-07-24.

**Decision.** Reserve a conditional range of 15–25% of natural-language
training exposures for Indic languages, with the final value selected only
after a usable-data audit. Treat language and domain as separate,
overlapping labels rather than forcing a document into one bucket. Present
**20%** as the single actionable headline recommendation — centered in the
range — and retain 15–25% as an audit boundary and sensitivity check, not
three equally prominent scenarios.

**Why.** Below 15%, Indic signal risks being diluted by English and global
web data; a fixed or higher quota could force excessive repetition of
scarce or weak material. The range lets data quality determine the final
allocation. Twenty percent communicates one clear default without assuming
enough clean data exists to sustain the maximum allocation — a concise plan
benefits from a stated default rather than making the reviewer choose the
core plan. Within the Indic allocation, sampling uses quality-aware
temperature weighting rather than raw corpus-size proportionality (D3), so
the largest corpora don't consume nearly all exposures while the smallest
aren't over-repeated.

**Benefits.** Improves native-language fluency, code-switching,
Indian-context knowledge, cross-lingual transfer, and the incentive to
improve Indic tokenization, while preserving room for English, code,
mathematics, science, and globally useful knowledge.

**Alternatives considered.** A 15% headline would reduce repetition and
protect general capability but may dilute Indic objectives. A 25% headline
would make a stronger multilingual commitment but depends on an unusually
large usable inventory.

**Risks.** Upsampling repeats documents and can increase memorization and
overfitting. Weak language identification may mislabel translated,
Romanized, or mixed-language text. Twenty percent remains an assumption
until corpus and ablation results exist, and because language and domain
labels overlap, it must never be added directly on top of the code/STEM
percentages as though all were disjoint buckets. Aggregate allocation can
also hide Hindi dominance and poor coverage of smaller languages.

**Validation needed.** Compare smaller training runs at multiple Indic
shares (15/20/25%); measure quality-versus-repetition curves per language;
report effective unique tokens and exposure counts separately; require
per-language capability and memorization checks before locking the final
percentage.

## D3 — Use quality-aware temperature sampling with floors and ceilings for Indic exposure

**Status.** Accepted 2026-07-24.

**Decision.** Do not sample Indic languages either equally or directly in
proportion to corpus size. Use a quality-adjusted temperature sampler,
p_l ∝ (n_l · q_l)^α, where n_l is a language's clean unique-token inventory,
q_l is a document-quality factor derived from reproducible signals, and
0 < α < 1 reduces domination by the largest corpora. Apply a minimum
exposure floor only to languages that pass the relevant quality threshold,
plus a per-document and per-source repetition ceiling.

**Why.** Proportional sampling would let the largest languages dominate;
equal sampling would severely over-repeat small corpora. Temperature
sampling is a controlled middle ground; floors preserve meaningful exposure
and ceilings prevent a capability claim from being manufactured by
recycling limited text.

**Alternatives considered.** Corpus-proportional sampling is simple but
reinforces data imbalance. Equal language allocation improves nominal
balance but ignores evidence volume and increases memorization. Fixed
hand-authored weights are difficult to reproduce and maintain.

**Risks.** Quality scores can encode bias against informal language,
dialects, and minority sources. A floor can still waste compute when clean
data is too small. One global temperature may not work across all tiers,
scripts, and training stages.

**Validation needed.** Tune α, floors, and ceilings through small-model
ablations. For every language, publish unique tokens, sampled exposures,
effective epochs, source concentration, benchmark gains, and memorization
results before finalizing numerical settings.

## D4 — Target a 128K context window through staged extension

**Status.** Accepted 2026-07-24.

**Decision.** Train most pretraining sequences at approximately 8K–16K
tokens, then progressively extend and post-train the model to a final
128K-token context window. Increase the share of genuinely long documents
and trajectories during extension rather than padding or concatenating
unrelated short texts.

**Why.** A long window supports code repositories, legal and policy
documents, long multilingual material, retrieval-augmented workflows, and
agent histories. Training every token at the maximum length would impose
substantial attention and data-packing costs; staged extension preserves
most of the benefit at lower total compute. The 128K target is set with a
named external comparator in view — a frozen Gemma 4 27B checkpoint ships a
256K-token context window (Gemma model card, checked 2026-07-17) — so
closing the gap fully to 256K was rejected as too large a compute and
long-context training-risk increase to take on unvalidated; 128K halves the
residual gap while keeping the staged-extension approach tractable.

**Alternatives considered.** Keeping the window at 64K was cheaper and
lower-risk but leaves long-context capability as a disclosed, unaddressed
gap versus the comparator. Matching 256K directly maximizes parity but
multiplies attention and packing cost and was rejected without a dedicated
compute/ablation justification. An 8K–16K-only model would be cheaper still
but constrains document and agentic tasks outright.

**Risks.** Supporting a 128K input does not prove the model uses distant
information reliably. A residual 2x context gap versus the comparator
remains and must be disclosed, not implied away. Positional extension can
degrade short-context quality; poor sequence packing can create artificial
cross-document relationships; long agent traces can amplify errors or
unsafe actions.

**Validation needed.** Select and ablate the positional method and
long-context schedule; measure accuracy by evidence position and context
length; test multilingual retrieval, long-code editing, legal-document
reasoning, distractor resistance, multi-turn tool use, and short-context
regressions.

## D5 — Make agentic capability primarily a post-training objective

**Status.** Accepted 2026-07-24.

**Decision.** Limit pretraining allocation for agent/tool structure to
roughly 2–3% (D1's 2.5%/75B row), covering tool documentation, APIs,
command formats, structured workflows, and high-confidence trajectories.
Teach most agentic behavior through supervised post-training,
executable-environment training, reinforcement learning, and safety
alignment (D6).

**Why.** Static next-token imitation can teach tool syntax but does not
prove a model can observe changing state, recover from failure, verify
results, or stop when a goal is complete. Interactive environments provide
outcome signals and reveal errors that polished synthetic traces conceal.
Supervised examples should cover task decomposition, tool choice,
execution, observation interpretation, recovery, verification, and
calibrated stopping; RL tasks should run in reproducible sandboxes with
machine-checkable outcomes; safety training should cover permission
boundaries, prompt injection, secret handling, fabricated tool results, and
confirmation before irreversible or high-impact actions.

**Alternatives considered.** Large-scale pretraining on synthetic traces is
easy to scale but encourages imitation of plausible-looking actions rather
than grounded behavior. Pure RL would be expensive, unstable, and
inefficient for teaching basic formats and workflows.

**Risks.** Reward hacking, benchmark overfitting, unsafe exploration,
excessive tool calls, premature stopping, and environment-specific
shortcuts can create misleading success. Synthetic trajectories can
propagate a teacher model's errors and style.

**Validation needed.** Use held-out interactive environments and measure
task success, cost, latency, invalid actions, recovery, verification,
stopping, permission compliance, and prompt-injection resistance. Split
environments by template and underlying task structure, not only by
instance, to reduce leakage.

## D6 — Use four evidence-gated post-training stages

**Status.** Accepted 2026-07-24.

**Decision.** Reserve up to 30B high-quality tokens for learning-rate
annealing; use up to 8B tokens for supervised fine-tuning; collect
approximately 2M calibrated preference comparisons; run up to 1M
outcome-verified RL episodes. Allocate SFT primarily as 25% general
instruction/conversation, 20% code, 15% mathematics/science, 20% agent/tool
workflows, 10% Indian public-service workflows, and 10% safety and
permission boundaries — this last figure is a fixed floor, never
reweightable by any proxy result (see D12). Require at least 20%
Indic-language exposure as an overlay across applicable SFT categories.

**Why.** Annealing concentrates scarce high-value material without
spending it all in the general pool. SFT teaches formats and desired
behavior, preference data distinguishes better responses, and RL practices
actions against outcomes. Agentic capability receives more emphasis after
pretraining, where executable and environmental verification is possible.

**Alternatives considered.** A single large SFT stage is simpler but
cannot reliably teach outcome-grounded action. Pure RL is inefficient for
basic formats and workflows. Large-scale imitation of synthetic traces is
cheap but can reward plausible narration instead of correct state changes.

**Risks.** The ceilings can encourage low-quality volume chasing. Model
judges can reproduce teacher bias; reward functions can be hacked;
translated preferences can erase native cultural and linguistic judgments.

**Validation needed.** Stop each stage when held-out marginal gains
flatten; deduplicate by task family; require native reviewers for Indic
preference and safety data; calibrate model judges against human or
executable outcomes; report success, cost, latency, invalid actions,
recovery, and permission compliance for agent tasks.

## D7 — Tier the Indic overlay by source-verification status, not headline tokens

**Status.** Accepted 2026-07-31.

**Decision.** Tier an existing ~50M-token Aya sample (Hindi, Tamil, Telugu,
Kannada, Malayalam) by `dataset_name` into three real categories, instead
of a cosmetic translated/non-translated distinction:
- **translated** — `dataset_name` ends `"(T)"` (machine/pipeline-translated
  from English source corpora, e.g. `Flan-CoT-submix (T)`).
- **verified** — `dataset_name == "Aya-Dataset"`, the one source in this
  split that is human-annotated by native-speaker volunteers (Singh et al.
  2024, *Aya Dataset: An Open-Access Collection for Multilingual Instruction
  Tuning*, arXiv:2402.06619), not a machine translation or unreviewed
  scrape.
- **unverified** — everything else: native-language web/templated content
  (`Telugu-news-articles`, `TamilStories`, `Thirukkural-instruct`, etc.)
  that is not translated but carries no human quality-verification label.

Implemented in `scripts/tier_aya.py`, which reuses the same deterministic
two-pass SHA-256 rank-sampling method (same pinned revision, same
per-language target) used across this project's audits, so this re-tiers
an already-measured sample rather than drawing a new one.

**Evidence.** `site/data/aya-tiers.json` (measured, 2026-07-31): tier
totals across all 5 languages — verified 89,605 tokens, unverified
4,813,754 tokens, translated 45,096,499 tokens, synthetic 0 tokens (sums to
the ~50M total). Per language: Hindi (verified 1,886 / unverified
2,661,544 / translated 7,336,560), Tamil (51,625 / 61,239 / 9,887,126),
Telugu (32,343 / 2,076,456 / 7,891,184), Kannada (0 / 2,171 / 9,997,795),
Malayalam (3,751 / 12,344 / 9,983,834).

**Why.** This plan's Indic-tiering commitment (D2; `README.md` §2) requires
splitting the Indic lane into verified/unverified/translated/synthetic
tiers rather than one headline number — a real, inspectable classification
closes that requirement instead of leaving it assumed.

**Risks / what this is not.** "Verified" here means human-annotated per
the Aya dataset's own methodology, not independently re-reviewed by this
project. Aya's schema has no synthetic-content label at all — the
synthetic tier is genuinely 0 for every language in this sample, not an
artifact of the classifier. This means Aya can only ever fill the
translated/unverified tiers of the Indic mixture; verified and synthetic
supply must come from elsewhere (D8).

## D8 — Spot-measure Sangraha's Hindi verified/unverified/synthetic tiers; cite the rest

**Status.** Accepted 2026-07-31.

**Decision.** D2 cited `ai4bharat/sangraha` (Khan et al. 2024,
arXiv:2403.06350) for its headline verified/unverified/synthetic split but
never downloaded or measured any of it. Rather than continuing to cite
second-hand, spot-sample one real shard per tier for Hindi only
(`verified/hin/data-0.parquet`, `unverified/hin/data-0.parquet`,
`synthetic/hin_Deva/wiki_hin_Deva_0000_of_0063.parquet` — confirmed via
`huggingface_hub.list_repo_files` before writing the sampler; the repo is
ungated), apply the same deterministic SHA-256 rank sampling as D7, ~2M
tokens/tier. Every other language, and every shard beyond shard 0, remains
a cited (not measured) number from the arXiv citation above.

**Evidence.** `site/data/sangraha-hindi-spotsample.json`
(measured-spot-sample, 2026-07-31): verified 1,999,983 tokens / 942 rows,
unverified 1,999,803 tokens / 760 rows, synthetic 1,999,990 tokens / 645
rows, each from a single ~250–380MB shard. Cited (arXiv:2403.06350):
251.321B tokens total across 22 languages — 64.306B verified / 24.308B
unverified / 162.708B synthetic; Hindi verified alone cited at 12.617B.

**Why.** This is the sharpest test of the whole plan: a lane can't claim a
large share just because a paper's abstract says so. One real measured
shard per tier turns "cited" into "spot-checked" for at least the
highest-resource language, and is honestly scoped as a spot-check, not a
full-corpus claim.

**Risks.** Single-shard sampling (942/760/645 rows out of ~100–113 shards
per tier) cannot be extrapolated to full-corpus ratios without a wider
sample — `README.md` and the JSON caveats both say this explicitly.
`verified` reflects Sangraha's own source-curation methodology, not
independent re-review.

## D9 — Name real training-corpus candidates for the agentic and long-context lanes

**Status.** Accepted 2026-07-31.

**Decision.** No earlier decision in this plan ever named a training
corpus for the agentic or long-context lanes — only evaluation benchmarks
(tau-bench, GAIA, OSWorld; RULER, LongBench v2) were referenced. Close this
gap with two real, licensed, audited datasets:
- **Agentic:** `glaiveai/glaive-function-calling-v2` (Apache-2.0, ungated,
  112,960 system-prompt + multi-turn tool-call conversations). The
  originally planned candidate, `Salesforce/xlam-function-calling-60k`,
  turned out to be a **gated repository** (401 Unauthorized without an
  approved HF access request) — discovered during verification, swapped
  immediately rather than citing a dataset this project cannot actually
  pull.
- **Long-context:** `Yukang/LongAlpaca-12k` (LongLoRA project, 12,000
  instruction/output pairs built specifically for context-extension
  training — see D10 for its licence status).

Both audited via `scripts/audit_agentic.py` / `scripts/audit_longcontext.py`,
using the same 9-stage audit template and shared `scripts/common_audit.py`
helper used across this project's dataset audits.

**Evidence.** `site/data/agentic-audit.json` (measured, 2026-07-31):
112,960 records, 56,115,280 tokens, mean 496 tokens/record, p95 1,167, max
4,083. `site/data/longcontext-audit.json` (measured, 2026-07-31): 12,000
records, 112,352,280 tokens, mean 9,359 tokens/record, p95 25,205, max
70,748; context-length histogram: 3,118 records under 2K tokens, 3,417 in
2K–8K, 3,464 in 8K–16K, 1,979 in 16K–32K, 22 above 32K.

**Why.** The agentic and long-context slots each need to point at a
specific inventory dataset, not a vague category. Both candidates are
real, checked for HF gating/access before being written into the plan
(consistent with this project's practice of verifying a dataset is
actually reachable before citing it — see D8's Sangraha gating check), and
both are small enough to audit locally end-to-end.

**Risks.** 56M tokens (agentic) and 112M tokens (long-context) are both far
below the 75B/1.5% pretraining allocations these lanes would need at full
3T scale — this is stated plainly as a supply-constrained lane in the
mixture table (D14), not hidden.

## D10 — Flag LongAlpaca-12k's non-commercial data licence

**Status.** Accepted 2026-07-31.

**Decision.** Record `Yukang/LongAlpaca-12k`'s licence precisely: the
LongLoRA project's **code** is Apache-2.0, but its own README badges the
**data** licence as **CC BY-NC 4.0** (non-commercial). Carry this forward
as "CC BY-NC 4.0 (data), Apache-2.0 (code) — non-commercial, flagged"
everywhere this dataset is cited, rather than the Apache-2.0 label an
earlier pass had assumed before checking the actual README.

**Evidence.** `Yukang/LongAlpaca-12k`'s dataset-card README, fetched
directly: `Code License: Apache 2.0`, `Data License: CC BY-NC 4.0`, `Weight
License: CC BY-NC 4.0` (three separate badges, three separate licences).

**Why.** Licence review is treated as an acquisition gate this plan must
not skip, not an afterthought bolted on after a dataset is already in use.
Mislabeling a non-commercial dataset as permissively licensed is exactly
the kind of quiet error an adversarial reviewer would catch first.

**Risks / consequence.** For this plan's audit/spot-check purpose the
non-commercial licence is acceptable. A real production pretraining run
using this lane would need either a negotiated licence carve-out or a
commercially licensed substitute — stated explicitly in `README.md`'s
wishful-accounting self-audit, not silently dropped.

## D11 — Define a 4-band reasoning-length curriculum from measured spans, not invented examples

**Status.** Accepted 2026-07-31.

**Decision.** Define the difficulty/reasoning-length curriculum (net-new;
no earlier decision in this plan defines a curriculum or difficulty bands)
using `bespokelabs/Bespoke-Stratos-17k`'s own
`<|begin_of_thought|>...<|end_of_thought|>` span length, measured directly
rather than assumed:

| Band | Thought-token range | Representative source |
|---|---|---|
| Band 0 (direct/no-CoT) | 0 (no thought span) | short Aya QA rows (D7) |
| Band 1 (short CoT) | 1–99 | shortest Bespoke-Stratos rows |
| Band 2 (medium CoT) | 100–1,999 | mid-length Bespoke-Stratos rows |
| Band 3 (long CoT / multi-step) | ≥2,000 | longest Bespoke-Stratos rows, `glaive-function-calling-v2` multi-turn tool traces, `LongAlpaca-12k` long documents |

Implemented in `scripts/audit_reasoning.py`, which buckets every one of the
16,710 Bespoke-Stratos records by its own measured thought-span length.

**Evidence.** `site/data/reasoning-audit.json` (measured, 2026-07-31): Band
1 = 98 records, Band 2 = 6,681 records, Band 3 = 9,931 records, Band 0 = 0
records (expected — this is a reasoning-focused corpus; no record lacks a
thought span, so Band 0 examples must come from a different, non-reasoning
source, i.e. Aya's short direct-QA rows per D7).

**Why.** This plan's curriculum requirement (`README.md` §5) calls for a
concrete example at each band — measuring the actual span-length
distribution of a real dataset gives concrete, defensible band boundaries
instead of round numbers picked without evidence.

**Risks.** Band boundaries (100 / 2,000 tokens) are a reasonable first cut
on this one corpus's distribution, not a validated optimum — the
toy-scale proxy run (D15, Hypothesis C) is the first test of whether
band-ordering (curriculum) actually changes anything measurable at this
scale.

## D12 — Generalize the protected floor beyond Indic-only

**Status.** Accepted 2026-07-31.

**Decision.** D3 defined a minimum-exposure floor and per-source repetition
ceiling, but scoped only to Indic language sampling. Generalize: every
lane's floor = 50% of its D1 headline share, which the mixture selector
(any future proxy-driven reweighting) cannot cross downward. Three
absolute policy floors override the 50% rule where D2/D6 already fixed a
number: Indic overlay ≥15% (D2's own lower bound), agentic pretraining ≥1%
(half of D5's 2–3%), safety/permission SFT fixed at 10% (D6) and never
reweightable by any proxy result.

**Why.** A dynamic mixture selector that can zero out a lane in response
to one proxy result is exactly the kind of instability an adversarial
reviewer would push on. A floor generalizes D3's existing Indic-specific
precedent instead of inventing an unrelated new mechanism.

**Risks.** The 50%-of-headline rule is a judgment call, not derived from
any run — it is deliberately conservative (allows only a 2x range around
each D1 share) precisely because no proxy evidence yet justifies a wider
range.

## D13 — Extend the anneal reserve's declared contents

**Status.** Accepted 2026-07-31.

**Decision.** Keep D6's ≤30B-token anneal cap unchanged, but name its
contents concretely instead of generically: cleanest
code/STEM/Indic-educational/long-form material (existing D6 language),
plus the longest Bespoke-Stratos Band-3 rows and the longest
LongAlpaca-12k documents — i.e. the anneal reserve is explicitly where the
two supply-constrained lanes from D9 get concentrated instead of diluted
across all 3T pretraining tokens.

**Why.** D9 already shows agentic (56M measured tokens) and long-context
(112M measured tokens) are far too small to matter at 75B/1.5%
pretraining scale if spread evenly. Concentrating what real data exists
into the anneal phase — where token volume matters less and
recency/quality matters more — is a defensible way to make a small real
corpus count, instead of pretending it's larger than it is.

**Risks.** This has not been proxy-tested (see D15's honest scope limits);
it is a design choice justified by D6's existing precedent and D9's
supply constraint, not by a run.

## D14 — Admit the Code lane has zero audited training-corpus supply

**Status.** Accepted 2026-07-31.

**Decision.** Carry D1's Code lane (20% / 600B tokens) forward unchanged
in the headline mixture table, but explicitly flag it in the
wishful-accounting self-audit as having **no training corpus audited
anywhere in this plan's inventory** — only evaluation benchmarks
(LiveCodeBench, SWE-bench Verified) are named. No dataset was substituted
in to paper over this, because no real, clearly licensed candidate had yet
been checked and confirmed.

**Why.** Quietly handing a large share to a lane with almost no real data
behind it is exactly the wishful accounting this plan is designed to
avoid. Naming the gap explicitly, rather than inventing a
plausible-sounding but unverified dataset, is the only defensible move.

**Risks / next step.** This is the single largest open risk in the whole
plan — 20% of a 3T-token budget with zero measured supply. Partially
addressed in D16 below (a real, partial-supply candidate corpus), which
downgrades this from "zero supply" to "partial supply" — it does not close
the gap.

## D15 — Run toy-scale proxy experiments; keep 1B/3B as written hypotheses where infeasible

**Status.** Accepted 2026-07-31.

**Decision.** This project's hardware (MacBook Pro 14" M1 Pro, 32GB unified
memory, MPS backend, no CUDA) cannot train a real 1B or 3B parameter model
in reasonable time (3B alone needs ~36GB+ for optimizer state before
activations; even a feasible 1B run would take tens to hundreds of hours on
MPS at realistic throughput). Two-track response:
1. **Executed toy-scale proxy** (~20M param nanoGPT-style model, MPS,
   `tiktoken` gpt2 encoding) trained on real audited slices only
   (Aya-tiered, Sangraha-Hindi, Bespoke-Stratos, glaive, LongAlpaca —
   excluding Code/General since D14 admits no real corpus exists there)
   across three competing mixture hypotheses (D1-proportional; Indic
   pushed to the 25% ceiling; curriculum-ordered vs. shuffled). Results
   reported as directional, sub-1B, never conflated with a literal 1B/3B
   bar.
2. **Written-only hypothesis** for what stays unexecuted at 1B/3B —
   chiefly the Code lane (D14): the named confirm/refute metric is
   HumanEval pass@1 delta across 15%/20%/25% code share at 1B parameters,
   ≥1,000 held-out problems, run only if compute becomes available.

**Evidence.** `scripts/proxy_data_mix.py`, `scripts/train.py`, results in
`site/data/proxy-results.json` (measured, 2026-07-31) and reproduced in
`README.md` §8a. Summary (~17.67M-param model, 600 steps, batch 32, block
256, M1 Pro MPS): Hypothesis A (D1-proportional, shuffled) reached shared
val loss 1.9076 / held-out-complex loss 6.6342; Hypothesis B (Indic pushed
to its 25% ceiling, shuffled) reached 1.9103 / 6.7225; Hypothesis C (same
ratios as A, curriculum-ordered) reached 1.8877 / 6.5984. A vs B shows no
detectable mixture-ratio effect at this scale (Δ = +0.0027 val loss); A vs
C shows a small but consistent improvement from curriculum ordering on both
the shared validation set and the held-out Band-3/long-document
generalization set. The held-out set's much higher absolute loss
(~6.6–6.7) versus the shared val set's (~1.9) across all three runs
confirms no leakage of withheld Band-3/long-document material into any
hypothesis's training stream.

**Why.** An executed proxy with real numbers is stronger evidence than a
written hypothesis, which in turn is stronger than an untested assertion.
Splitting the work this way earns real credit on the lanes with real data
(D7–D11) while being honest about the one lane (Code) where no data was
verified within scope.

**Risks.** A ~20M-parameter, single-machine, few-hour toy run cannot
distinguish subtle mixture effects that would only appear at 1B+ scale with
much larger token budgets — this is stated as a limitation, not glossed
over.

## D16 — Name and spot-sample a real, licensable Code-lane corpus

**Status.** Accepted 2026-07-31.

**Decision.** `codeparrot/github-code-clean` (a quality-filtered version of
the `codeparrot/github-code` BigQuery GitHub export) is named as the Code
lane's first real, audited candidate corpus. Audited via
`scripts/audit_code.py`: a single-shard (of 883) deterministic SHA-256
rank-sample restricted to a permissive-license allowlist (`mit`,
`apache-2.0`, `bsd-3-clause`, `bsd-2-clause`, `isc`, `cc0-1.0`,
`unlicense`), excluding copyleft (`gpl-2.0`/`3.0`, `agpl-3.0`,
`lgpl-2.1`/`3.0`) and ambiguous (`mpl-2.0`, `epl-1.0`, `artistic-2.0`)
licenses — the same per-file-license allowlist approach StarCoder's own
training data used, not an invented signal. Measured result
(`site/data/code-audit.json`): shard 0 carries 270,601,959 cl100k_base
tokens across 126,925 files, of which 52.8% (142,871,195 tokens, 81,693
files) fall under the permissive allowlist. A single-shard
order-of-magnitude extrapolation (×883 shards) estimates ~238.9B tokens
corpus-wide, ~126.2B permissive-license tokens — cross-checked against the
dataset card's own file-count arithmetic (115M files cited by
`github-code`, minus 3.39M removed by `github-code-clean`'s quality filter
≈ 111.6M expected files, versus this extrapolation's 112,074,775
estimated rows — a 0.4% difference), which gives real confidence in the
uniform-shard-size assumption underlying the extrapolation.

**Why.** D14 admitted the Code lane had zero audited supply anywhere in
this plan's inventory — the single largest wishful-accounting risk in the
plan. Rather than leaving that admission unaddressed once a spare cycle
was available, this closes part of the gap with a real, reachable,
per-file-licensed dataset, audited with the same spot-sample methodology
already used for Sangraha (D8).

**Risks.** ~126.2B estimated permissive tokens covers only ~21% of the
Code lane's current 600B-token budget (20% of the 3T total) at one epoch
— this does **not** close D14's gap, it downgrades it from "zero supply"
to "partial, single-source supply, requiring ~4–5x repetition or a reduced
Code-lane share." The extrapolation is a single-shard estimate, not a
full-corpus measurement — shard-to-shard variance in language/license mix
is unverified. License is inherited from each file's origin repository at
the dataset's 2022 BigQuery export time and is not re-verified per file; a
repository's license can change or be misdeclared after export.
Secret/PII scrubbing in the sample is a coarse regex heuristic (193
email-like and 3 secret-assignment-like hits in the 2M-token sample) — a
real training pipeline would need a production-grade secret scanner
before use, not this audit's heuristic.

**Validation needed.** Multi-shard sampling across more of the 883 shards
to replace the single-shard extrapolation with a real aggregate
measurement — **done in D17 below**, which revises the estimate to
~135.9B permissive tokens; a second, complementary code source to close
the remaining ~464B-token gap (or a formal reduction of the Code lane's
20% share); per-file license re-verification against each repository's
current state for at least a sub-sample.

## D17 — Multi-shard validation of the Code-lane permissive-token estimate

**Status.** Accepted 2026-07-31.

**Decision.** Ran `scripts/audit_code_shards.py`, a wide column-only scan
(`license` + `size`-in-bytes columns only, no text, no tokenization) across
10 shards of `codeparrot/github-code-clean`, evenly spaced across the full
883-shard range (indices 0, 88, 176, 264, 353, 441, 529, 618, 706, 794).
Measured permissive-byte fraction per shard: mean 56.76%, stdev 1.17%, min
54.87% (shard 0), max 58.47% (shard 176) — low variance confirms D16's
single-shard estimate was not an outlier. Calibrating a tokens-per-byte
ratio from `audit_code.py`'s exact tiktoken measurement of shard 0
(0.265378 tokens/byte), the 10-shard mean projects to ~239.4B total tokens
and ~135.9B permissive tokens corpus-wide
(`site/data/code-shards-audit.json`) — revising D16's single-shard
extrapolation of ~126.2B permissive tokens upward by ~7.7%, now backed by
real cross-shard variance data rather than a single data point.

**Why.** D16's own "Validation needed" section explicitly called for
multi-shard sampling to replace its single-shard extrapolation. This
closes that specific validation item cheaply — column-only reads avoid
re-tokenizing full file text across 10 shards — while confirming the
original estimate was directionally sound.

**Risks.** Still only 10 of 883 shards (~1.1%) sampled — evenly spaced,
not random, so cyclical patterns in the BigQuery export order (e.g. by
repository popularity or creation date) could bias the sample; unlikely
but not ruled out. Byte-to-token calibration relies on a single shard's
ratio — a language-mix shift across shards could change token density
without changing byte counts proportionally. ~135.9B permissive tokens
still covers only ~22.6% of the Code lane's 600B budget: this strengthens
confidence in the number, it does **not** change the conclusion that the
Code-lane supply gap remains open.

**Validation needed.** A second, independent code source to close the
remaining ~464B-token gap (or a reduced Code-lane share); random (not
evenly-spaced) shard sampling to rule out export-order bias; per-file
license re-verification against each repository's current state.

## D18 — Second, independent Code-lane source: `bigcode/commitpack`

**Status.** Accepted 2026-07-31.

**Decision.** Named and spot-sampled `bigcode/commitpack` (BigCode,
ungated, ~3.8TB raw across 350 languages, each row carrying a per-file
`license` column — the same provenance approach D16 used for
`github-code-clean`) as a second Code-lane source, via
`scripts/audit_commitpack.py`. Sampled one shard file each for 5 mainstream
programming languages (python, javascript, java, c, go), deliberately
excluding CommitPack's markup/data/doc-heavy directories (json, xml, text,
markdown, csv, yaml, html — which dominate its raw byte count but are not
"software engineering code"). Measured across the 5-language sample: 339.3M
total tokens, 84.8% permissively licensed (287.8M tokens) — a notably
higher permissive fraction than `github-code-clean`'s ~52.8–58.5%
per-shard range, plausibly because mainstream languages skew more
permissively licensed than the full github-code-clean mix. Extrapolating
only within these same 5 languages' full `data/<lang>/` directory sizes
(933.4B bytes, via a measured tokens-per-byte ratio of 0.130072) projects
~103.0B permissive tokens (`site/data/commitpack-audit.json`) — the other
345 language directories stay cited (BigCode's dataset-card total), not
measured.

Because CommitPack draws from the same public-GitHub universe as
github-code-clean, this script also ran a real overlap check: it hashed
every permissive row already cached locally from D16/D17's
github-code-clean shard 0 (81,693 rows) and compared this script's 344-row
permissive sample against that set. Result: **0 exact matches.**

**Why.** D16/D17 left the Code lane at ~135.9B measured permissive tokens
against a 600B budget (~22.6%) — the single largest open gap in the plan.
Rather than assume a second source closes that gap, or assume it's fully
redundant with the first, this measures a second real, licensable corpus
and tests the overlap question directly instead of leaving it as an
unstated assumption either way.

**Risks.** The 0-exact-match overlap result is a **lower bound signal, not
proof of independence** — it checked only 344 sampled rows against 1 of
github-code-clean's 883 shards, a tiny fraction of both corpora's true
intersection. More importantly, CommitPack stores **per-commit snapshots**
(often several per file, at different points in its history), while
github-code-clean stores **one snapshot per file** — the same public
repository can appear in both datasets without ever producing an
exact-content match, because the sampled snapshots simply captured
different commits of the same file. Byte-level exact-match overlap being
near-zero therefore does **not** imply the two corpora's permissive-token
estimates are safe to sum: file-level (not exact-content) overlap between
them is almost certainly substantial and remains unmeasured. Only 5 of
CommitPack's 350 language directories were sampled, one shard file each —
same single-shard-per-language caveat D16 carried before D17's multi-shard
validation.

**Combined Code-lane accounting (honest, non-additive).** Given the
unresolved overlap risk above, this plan does **not** report
135.9B + 103.0B ≈ 238.9B as the Code lane's combined supply. Instead:
- **Lower bound (~22.6% of budget):** treat the two sources as fully
  overlapping — D17's ~135.9B figure alone, unchanged from before D18.
- **Upper bound (~39.8% of budget):** treat the two sources as fully
  independent — the naive sum, ~238.9B tokens.
- **Stated range: ~135.9B–~238.9B permissive tokens (~22.6%–~39.8% of the
  600B budget)** — still short of 100% coverage under either assumption,
  so the Code lane remains open regardless of where the true overlap
  falls.

**Validation needed.** A real cross-corpus deduplication pass
(near-duplicate detection, e.g. MinHash/SimHash over normalized file
content, not just exact SHA-256) across a much larger joint sample from
both corpora, to replace the lower/upper-bound bracket above with a single
measured overlap-adjusted estimate; wider CommitPack language coverage
beyond the 5 sampled; per-file license re-verification against each
repository's current state for both sources.

## D19 — Lane → target-benchmark mapping

**Status.** Accepted 2026-07-31.

**Decision.** Added a new §1a to `README.md` ("Lane → target benchmarks")
mapping every one of the 9 lane rows in §1's budget table to the specific
public benchmark(s) that lane's tokens are meant to move the needle on,
with a one-line rationale per lane (Code → HumanEval/LiveCodeBench/
SWE-bench Verified, tied directly to §8b's HumanEval pass@1 confirm/refute
metric; Agentic → BFCL/tau-bench, tied to `glaive-function-calling-v2`'s
format; Long context + reasoning → RULER/LongBench v2/GPQA Diamond, tied to
`LongAlpaca-12k`/`Bespoke-Stratos`; Indic overlay →
IndicGenBench/Flores-200/IndicXTREME; General → MMLU-Pro/ARC-Challenge;
Math/science → MATH/GPQA; Books/education → MMLU humanities
subsets/LongBench v2; Indian law/gov/finance → IndicGenBench +
native-expert sealed sets, the one lane with no mature public benchmark;
Safety/permission SFT floor → HarmBench/StrongREJECT/XSTest, chosen
specifically to cover both under-refusal and over-refusal failure modes
since D6 fixes that share as a non-tunable floor).

**Why.** This plan's problem statement (`README.md`, opening section)
requires tying each lane back to the benchmark(s) it is meant to win, so a
reviewer can see why each number is what it is — this was the one
requirement not yet satisfied anywhere in `README.md` prior to this
decision. This mapping is newly reasoned here: no earlier decision in this
log names a specific public benchmark for any lane; earlier framing (D4's
comparator-relative context target) only ever set a pass/fail bar against
one external model, never a benchmark-by-benchmark target list.

**Risks.** This is target-naming for the evaluation strategy, not a run
result — no model has actually been evaluated against any benchmark named
here, so the mapping cannot yet be confirmed to discriminate the way it
claims to. Several benchmarks (HumanEval, MMLU in particular) are
widely-known contamination risks for exactly the largest lanes here (Code,
General) — named explicitly as a caveat in `README.md` §1a rather than
presented as clean. The Indian law/gov/finance lane has no mature public
benchmark at all, so its "target" is partly a private-set placeholder
rather than a citable, reproducible instrument — stated plainly rather
than forcing a poor-fit public benchmark onto it.

**Validation needed.** Once any of the audited lanes has a real trained
checkpoint (even the toy-scale §8a proxy, if extended with a task head),
confirm the named benchmarks actually separate the mixture hypotheses
tested in §8a/§8b — right now the mapping is unverified by any
measurement. A contamination check (n-gram overlap between the named
benchmarks' public test sets and the audited training samples in
`site/data/*.json`) is not yet done and would directly strengthen or
weaken the concern about public-benchmark contamination raised above.

## Open Decisions

- Real cross-corpus deduplication between `github-code-clean` and
  `commitpack` (D18) to collapse the ~135.9B–~238.9B bracket into a single
  overlap-adjusted estimate.
- A third Code-lane source, or a formal reduction of the Code lane's 20%
  share, if the deduplicated estimate still falls well short of the 600B
  budget.
- Real 1B-scale proxy run for the Code lane (D14/D16/D17/D18), pending
  either cloud compute or a wider-verified code training corpus, whichever
  comes first.
- Wider Sangraha sampling (multiple shards, multiple languages) beyond the
  single-shard Hindi spot-sample in D8.
- Native-review of the Aya tier classification in D7 (an Indic-language
  speaker confirming "unverified" rows are not silently low-quality).
- Cross-lane global deduplication across all six lanes' audited samples
  (each lane was deduplicated independently, not against each other) —
  D18's github-code-clean/commitpack overlap check is a first, partial
  step in this direction, not a substitute for it.
- A contamination check (D19): n-gram overlap between the named
  benchmarks' public test sets (HumanEval, MMLU, GPQA, etc.) and the
  audited training samples in `site/data/*.json`, to test rather than
  merely assert the concern about public-benchmark contamination.
