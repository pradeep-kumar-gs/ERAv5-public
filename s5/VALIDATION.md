# Accepted-Decisions Handoff and Validation Checklist

A compact record of what is measured vs. cited vs. still open in this plan,
for anyone (reviewer or future session) picking it up cold.

## What is measured (real numbers, reproducible via `README.md` §10)

| Claim | Script | Output JSON |
|---|---|---|
| Aya tiered into verified/unverified/translated, 5 languages | `scripts/tier_aya.py` | `site/data/aya-tiers.json` |
| Sangraha Hindi spot-sample, verified/unverified/synthetic, 1 shard each | `scripts/audit_sangraha.py` | `site/data/sangraha-hindi-spotsample.json` |
| Agentic lane audit (`glaive-function-calling-v2`) | `scripts/audit_agentic.py` | `site/data/agentic-audit.json` |
| Long-context lane audit (`LongAlpaca-12k`) | `scripts/audit_longcontext.py` | `site/data/longcontext-audit.json` |
| Reasoning lane audit + 4-band curriculum (`Bespoke-Stratos-17k`) | `scripts/audit_reasoning.py` | `site/data/reasoning-audit.json` |
| Toy-scale proxy data pools + 3 mixture hypotheses | `scripts/proxy_data_mix.py` | `site/data/proxy-data-manifest.json` |
| Toy-scale proxy training results (A/B/C) | `scripts/train.py` | `site/data/proxy-results.json` |
| Code lane deep spot-sample, permissive-license subset (`github-code-clean`) | `scripts/audit_code.py` | `site/data/code-audit.json` |
| Code lane wide 10-shard scan (license+size only) | `scripts/audit_code_shards.py` | `site/data/code-shards-audit.json` |
| Code lane second source, 5-language spot-sample (`commitpack`) + overlap check | `scripts/audit_commitpack.py` | `site/data/commitpack-audit.json` |

## What is cited, not measured (explicitly labeled as such everywhere it appears)

- Sangraha's full-corpus totals for all languages except the Hindi
  spot-sample above, and all shards beyond shard 0 — Khan et al. 2024,
  arXiv:2403.06350.
- General, Math/science, Books/education, Indian law/gov/finance lane
  shares — this plan's own baseline mixture (D1), not independently
  re-audited this round.
- All evaluation benchmarks (LiveCodeBench, SWE-bench Verified, MMLU-Pro,
  GPQA, MATH, tau-bench, GAIA, OSWorld, RULER, LongBench v2, HarmBench,
  StrongREJECT, XSTest) — named as targets in `README.md` §1a, not
  independently re-verified here.
- `codeparrot/github-code-clean`'s full-corpus totals beyond the 10 sampled
  shards (D16/D17) — the ~239.4B token / ~135.9B permissive-token figures
  are a 10-of-883-shard (~1.1%) measured aggregate, not a full-corpus
  measurement, and the shards sampled are evenly spaced rather than random
  (possible export-order bias, unlikely but unruled-out). Per-file
  licences are the dataset's 2022 BigQuery-export-time snapshot, not
  re-verified against each repository's current state.
- `bigcode/commitpack`'s full-corpus totals beyond the 5 sampled languages
  (D18) — the ~103.0B permissive-token estimate covers only python,
  javascript, java, c, and go's full directory sizes, extrapolated from
  one shard file per language; the other 345 language directories
  (including php, typescript, ruby, c#, rust) are cited from BigCode's
  dataset card, not measured. Whether github-code-clean and commitpack's
  permissive-token estimates are additive is itself unresolved (see
  partial-supply entry below) — cited as an open question, not settled
  either way.

## What is newly reasoned (target-naming, not measured or previously cited)

- **Lane → target-benchmark mapping (§1a, D19)**: each of the 9 lanes in
  §1's budget table is mapped to specific public benchmark(s) (e.g. Code →
  HumanEval/LiveCodeBench/SWE-bench Verified, Agentic → BFCL/tau-bench,
  Indic overlay → IndicGenBench/Flores-200/IndicXTREME). This is not a
  measurement (no model has been evaluated against these benchmarks) and
  not pulled from any earlier evaluation framework in this plan — it is a
  newly reasoned target list, explicitly caveated in `README.md` §1a and
  `report.md` D19 as unverified and subject to public-benchmark
  contamination risk.

## What is a partial-supply finding (real data, measured, but not enough of it)

- **Code lane (20% / 600B tokens, D14/D16–D18)**: `github-code-clean`'s
  permissive-license subset is named, spot-sampled, and multi-shard
  validated, estimated at ~135.9B tokens (D16/D17). `bigcode/commitpack`,
  a second independent source, is now also named and spot-sampled (D18),
  adding ~103.0B estimated permissive tokens from 5 sampled languages.
  Because both sources draw from the same public-GitHub universe, D18 ran
  a real (not assumed) overlap check — 0 exact-content matches between a
  344-row CommitPack sample and D16/D17's cached github-code-clean shard —
  but this is a lower-bound signal from a tiny slice of both corpora, not
  proof the two estimates are safe to sum (CommitPack stores per-commit
  snapshots, github-code-clean stores one snapshot per file, so the same
  repository can appear in both without an exact-content match). The plan
  therefore reports a range, not a sum: ~135.9B (fully overlapping) to
  ~238.9B (fully independent) permissive tokens, ~22.6%–~39.8% of the 600B
  budget. Downgraded from "zero supply" (D14) to "partial, two-source
  supply with an unresolved overlap bracket" (D16–D18); still the single
  largest open risk in the plan, named as such in `README.md` §5a/§9, not
  hidden either way.

## Checklist — every `README.md` headline number traces to something real

- [x] Lane budget table (§1): 9 lanes, each with a stated reason for its
      share (D1) and a supply verdict from this round's audit; Code
      partial-supply finding (D14/D16/D17) + agentic/long-context/reasoning
      slots named (D9) + Indic split into real tiers (D7/D8).
- [x] Indic tier table (§2): every cell labeled measured or cited; no
      unmarked numbers.
- [x] Agentic lane (§3): dataset named, gating verified before commit
      (xlam swap), audited end-to-end.
- [x] Long-context lane (§4): dataset named, licence read from source (not
      assumed), audited end-to-end, licence caveat stated.
- [x] Reasoning lane + curriculum bands (§5): bands derived from measured
      span lengths, not invented; each band has a real representative
      source.
- [x] Protected floor (§6): generalizes an existing precedent (D3), not
      invented from nothing.
- [x] Anneal reserve (§7): extends an existing cap (D6), contents named.
- [x] Code lane (§5a): two partial-supply corpora named, spot-sampled, and
      cross-checked for overlap (D16/D17/D18), real numbers traced to
      `site/data/code-audit.json`, `site/data/code-shards-audit.json`, and
      `site/data/commitpack-audit.json`; the unresolved overlap bracket and
      the gap to the 600B budget are both stated explicitly rather than
      implied closed.
- [x] Proxy experiments (§8): toy run actually executed with real numbers
      (§8a); unexecuted 1B/3B hypothesis stated with a concrete
      confirm/refute metric (§8b) for the lane still without enough real
      proxy-ready data (Code).
- [x] Wishful-accounting self-audit (§9): every lane's real supply vs.
      headline share stated; Code flagged as the one lane still failing this
      test, though no longer at zero supply.
- [x] Decision log pointer (§10): reproduction commands present and match
      the actual `scripts/` filenames.
- [x] Lane → target-benchmark mapping (§1a, D19): every lane in the budget
      table tied to specific public benchmark(s) with a one-line rationale;
      this is newly reasoned, not pulled from any earlier framework in this
      plan; explicitly caveated as target-naming, not a run result, with
      contamination risk stated.

## Known limitations (stated once here, not repeated at every mention)

- The toy-scale proxy (§8a) is ~18M parameters, single-machine, ~900K
  training tokens — directional only, never conflated with a real 1B/3B
  run.
- Sangraha's measured tier is Hindi, one shard per tier — a spot-check, not
  a full-corpus measurement.
- Aya's "verified" label reflects the dataset's own annotation methodology,
  not independent re-review by this project.
- No cross-lane deduplication was performed between the independently
  audited lanes (Aya, Sangraha, glaive, LongAlpaca, Bespoke-Stratos,
  github-code-clean, commitpack).
