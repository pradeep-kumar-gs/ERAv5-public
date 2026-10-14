# Session 8: Attention, chronologically

A web app explaining how attention works now. It covers 52 attention mechanisms from 2014 to 2026, **in the order they first appeared**. Each one is presented as an answer to the bill the previous mechanisms left behind, with what it buys, what it gives up, and when you would actually pick it (for a 2K chatbot, 128K document QA, or a 1M-token agent).

- **Live:** https://erav5s8.netlify.app
- **Repo:** https://github.com/pradeep-kumar-gs/ERAv5-public/tree/main/s8

## What's on the page

| Section | What it does |
|---|---|
| 1 · Start here | Interactive scaled dot-product attention: 6 tokens, every Q/K/V, score, scale, mask, softmax and output number visible. Toggle the causal mask and watch weight leak to the future. |
| 2 · The two bills | KV-cache calculator (context, users, dtype) across MHA/GQA/MQA/2-bit KV/MLA/sliding window/hybrid/CSA-style/linear, plus query-key pair counts for the compute bill. Reproduces the course's 6.44 GB / 51.54 GB numbers. |
| 3 · Timeline | An overview chart (family lanes × time, era bands), then 52 dated cards in strict chronological order. Each card has the problem at that moment, the mechanism, what it buys, what it gives up, when to pick it, and a fit verdict per workload. Four inline interactive pieces appear where they first become relevant: positions (learned/sinusoidal/RoPE/ALiBi), masks (dense to NSA/MoBA, plus H2O eviction), linear attention and the delta rule (the course's 140 / 95 vs 55 numbers), and RoPE extension (PI vs NTK-aware vs YaRN). |
| 4 · Reading it | The field changing its mind (exactness → length + memory → exactness → length via position → memory → mix/sparsify/compress), the parallel systems track (Flash/Paged/eviction/quantisation/Ring), tokenizer fertility as a context cost, the pattern behind it, a labelled prediction of what comes next, and a "ship today" table by workload. |
| 5 · Case study | LightningLM V4 choices from the course notes (DDDGDDDG, sparse budget, Memory Stream, DroPE 8K→256K), mapped to their public lineage and kept off the dated timeline because they have no public source. |
| 6 · Corrections | Where the course or common write-ups get it wrong, checked against primary sources (see below). |
| 7 · Sources | Every date with its primary source and how it was checked. |

All 17 mechanisms on the assignment's list are covered (18 cards, because the delta rule and Gated DeltaNet get one each), plus 34 additions: Bahdanau attention, Shaw relative positions, Transformer-XL, Sparse Transformer, T5 bias, Compressive Transformer, Reformer, Linformer, BigBird, Performer, NoPE, FlashAttention, Position Interpolation, H2O/SnapKV eviction, PagedAttention, Ring Attention, RetNet, Mamba, GLA, Jamba, KIVI 2-bit KV cache, LongRoPE, Mamba-2/SSD, parallel DeltaNet, Differential attention, Titans, MiniMax-01 Lightning, NSA, MoBA, gated attention, Qwen3-Next, DSA, MiniMax-M2's return to full attention, and Kimi Linear.

## What the timeline shows

**The question:** *What does the timeline actually show? Write what you saw once the mechanisms were in date order
that you could not see as a list. If you found a mechanism I did not cover, name it with its date and the paper or
release you got the date from.*

### What date order shows that a list hides

A list grouped by family (positions, cache sharing, linear, sparse) makes each family look like a steady line of
improvement. Put in date order, the same mechanisms show the field **changing its mind** about which bill to pay
down, and swinging back:

1. **Exactness (2017).** The Transformer drops recurrence for an exact all-pairs lookup. It creates the two bills
   everything after it is trying to pay: T² compute and a KV cache that grows with every token.
2. **Length, then memory (2019–2021).**
   - Sparse patterns (Sparse Transformer, Longformer, BigBird), Reformer, Linformer and Performer attack T².
   - Linear attention (2020) turns attention back into an RNN with a fixed-size state.
   - MQA (2019) is the first to target the *decoding* bill instead of compute.
   - RoPE (2021) and ALiBi (2021) redesign position so a model can survive inputs longer than it was trained on.
3. **Exactness wins back (2022 to mid-2023).** FlashAttention (2022) makes *exact* attention cheap on real
   hardware. Most of the 2020 approximations quietly lose. GQA (2023) fixes enough of the cache to become everyone's
   default.
4. **Length through position (mid-2023).** Within ten weeks, Position Interpolation (06-27), NTK-aware scaling
   (06-29, a Reddit post, not a paper) and YaRN (08-31) stretch models trained short. Attention sinks (09-29)
   explain why windowed attention collapses without its first tokens.
5. **Memory again (late 2023–2024).** In serving, the KV cache is now the dominant cost.
   - Fixed-state models come back, this time with learned forgetting and correction: Mamba, GLA, DeltaNet, Gated
     DeltaNet.
   - MLA (2024) compresses the cache while keeping exact access.
   - The systems track runs in parallel: PagedAttention, KV eviction (H2O/SnapKV), 2-bit KV caches (KIVI) and Ring
     Attention.
6. **Mix, sparsify, compress (2025–2026).**
   - Hybrids ship at scale: MiniMax-01, Qwen3-Next, Kimi Linear.
   - Sparsity becomes something the model learns: NSA, MoBA, DeepSeek Sparse Attention.
   - DeepSeek-V4 compresses the sequence itself.
   - On **2025-10-30** the field disagrees in public: MiniMax-M2 goes back to full attention the same day Kimi Linear
     argues for a linear hybrid.

**What I saw that a list can't show.**
- Nothing was "wrong". Every mechanism is somebody paying less on whichever bill is currently the largest:
  compute, then cache, then length, then cache again.
- The order also explains odd facts in a list: why linear attention appeared in 2020 and then went quiet (in 2022
  FlashAttention made exact attention cheap enough), and why it came back from late 2023 with RetNet, Mamba and GLA
  (the cache became the bill).
- **It lets you guess what comes next.** The newest sparse methods still score every (compressed) key with a cheap
  indexer, so selection itself is still quadratic. The next step should be hierarchical selection, coarse to fine.
  Position is becoming implicit (NoPE, partial RoPE, DroPE).

The site's [Section 4](https://erav5s8.netlify.app/#reading) has the same reading next to the cards, plus a
"what to ship today" table by workload.

### Mechanisms not covered in class

The class list had 17 mechanisms, and all of them are on the site. These **34 are additions**. Each date is the
first public appearance: arXiv v1 for papers, read from the arXiv API by `scripts/verify_dates.py`, and the official
release for models. The "How the date was checked" column of the sources table below gives the evidence for each.

| Date | Mechanism | Source the date comes from |
|---|---|---|
| 2014-09-01 | Additive attention (Bahdanau) | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) |
| 2018-03-06 | Relative position representations (Shaw) | [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) |
| 2019-01-09 | Transformer-XL (segment recurrence) | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) |
| 2019-04-23 | Sparse Transformer (strided / fixed patterns) | [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) |
| 2019-10-23 | T5 relative position bias | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) |
| 2019-11-13 | Compressive Transformer (compressed memory) | [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507) |
| 2020-01-13 | Reformer (LSH attention) | [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) |
| 2020-06-08 | Linformer (low-rank sequence projection) | [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) |
| 2020-07-28 | BigBird (window + global + random) | [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) |
| 2020-09-30 | Performer (FAVOR+ random features) | [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) |
| 2022-03-30 | NoPE (no positional encoding, causal) | [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634) |
| 2022-05-27 | FlashAttention (IO-aware exact attention) | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) |
| 2023-06-24 | KV-cache eviction (H2O, SnapKV) | [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) |
| 2023-06-27 | Position Interpolation (linear RoPE scaling) | [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) |
| 2023-07-17 | RetNet (retention) | [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) |
| 2023-09-12 | PagedAttention (vLLM) | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) |
| 2023-10-03 | Ring Attention (sequence-parallel exact attention) | [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) |
| 2023-12-01 | Mamba (selective state space) | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) |
| 2023-12-11 | Gated Linear Attention (GLA) | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) |
| 2024-02-05 | KV-cache quantisation (KIVI) | [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) |
| 2024-02-21 | LongRoPE (searched non-uniform scaling) | [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) |
| 2024-03-28 | Hybrid depth schedules (Jamba) | [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) |
| 2024-05-31 | Mamba-2 / State Space Duality | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| 2024-06-10 | DeltaNet, parallelised | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) |
| 2024-10-07 | Differential attention | [Differential Transformer](https://arxiv.org/abs/2410.05258) |
| 2024-12-31 | Titans (test-time learned memory) | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) |
| 2025-01-14 | Lightning Attention hybrid at scale (MiniMax-01) | [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) |
| 2025-02-16 | Native Sparse Attention (NSA) | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) |
| 2025-02-18 | MoBA (mixture of block attention) | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) |
| 2025-05-10 | Gated attention (output gate, sink-free) | [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) |
| 2025-09-11 | Qwen3-Next (3:1 Gated DeltaNet : gated attention) | [Qwen3-Next-80B-A3B model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| 2025-09-29 | DeepSeek Sparse Attention (DSA, lightning indexer) | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| 2025-10-30 | Back to full attention (MiniMax-M2) | [Why Did MiniMax M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) |
| 2025-10-30 | Kimi Linear (KDA + MLA, 3:1) | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) |

## Run it

```bash
python3 -m http.server 8000 --directory s8/site   # open http://127.0.0.1:8000
node s8/tests/test_lib.mjs                         # widget maths + data invariants (12 checks)
python3 s8/scripts/verify_dates.py                 # re-check every arXiv date against the arXiv API
```

The site is static (plain HTML/CSS/ES modules, no build step, no dependencies). Deployed with `netlify deploy --prod --dir s8/site`.

## Corrections to the course (Session 8 notes and class transcript)

1. **DroPE's mechanism is public.** The notes say the exact algorithm and "which rotary dimensions it changes" are not established. DroPE is [arXiv 2512.12167](https://arxiv.org/abs/2512.12167) (v1 2025-12-13, Sakana AI). It removes the positional embeddings entirely after pretraining, followed by a short recalibration, and claims zero-shot context extension.
2. **"Attention itself does not know token order"** holds for bidirectional attention. A causal decoder can infer position from the mask ([Haviv et al. 2022](https://arxiv.org/abs/2203.16634)), and DroPE relies on exactly this.
3. **"DeepSeek built for a million natively"** is true, but via a curriculum: the V4 report trains 4K → 16K → 64K → 1M, with dense attention first and sparsity switched on at 64K.
4. **Class claim: "RoPE was invented where 32K was the target … nobody tested it."** RoFormer pre-trains at max length 512 and evaluates long documents at 1024. There is no 32K target in the paper.
5. **Class claim: DeepSeek trained for "$50 million" vs "$500 million".** The DeepSeek-V3 report states $5.576M (2.788M H800 GPU hours at $2/h), and says explicitly that this excludes prior research and ablations.
6. **Class slip: Transformer "2018 and 17".** arXiv v1 is 2017-06-12.
7. **DroPE:** in class it was described correctly ("train with RoPE, then remove it"). It is the written notes that call the mechanism unknown. "8K gives extension of 32k" in class versus 8K → 256K in the notes is presumably 32×.
8. **Checked and correct:** V4 applies RoPE to the last 64 dims (inherited from MLA's 64-dim decoupled RoPE in DeepSeek-V2), and CSA/HCA/indexer details match the V4 report (m=4, m'=128, top-512/1024, 128-token window).

## Sources for dates

**Policy:** the date is the first public appearance. For papers that is the **arXiv v1 submission date**, read from the arXiv API's `published` field by `scripts/verify_dates.py`; 47/47 match. For models it is the official release or announcement, and for forum and blog origins it is the post timestamp. Earlier informal appearances are listed alongside, but the arXiv date is still the one used.

Date traps found while checking:
- **YaRN is 2023-08-31**, not September. The arXiv ID 2309.00071 reflects the announcement month, not the submission date.
- **NTK-aware scaling is a Reddit post**, not a paper. u/bloc97 posted it 2023-06-29 08:21 UTC (timestamp from the PullPush Reddit archive API), two days *after* Meta's PI paper (2023-06-27). PI's idea was itself on kaiokendev's SuperHOT blog, archived by the Wayback Machine on 2023-06-21.
- **RoPE** appeared on Jianlin Su's blog (kexue.fm/archives/8265) on 2021-03-23, four weeks before the RoFormer arXiv paper (date confirmed from the Wayback Machine capture).
- **Sliding window** dates to Longformer (2020-04-10), with banded patterns in the Sparse Transformer (2019-04-23). Mistral (2023-10-10) popularised it.
- **DeepSeek-V4** has three candidate dates: HF repos created 2026-04-22 (private), the official release post 2026-04-24 (used here), and arXiv v1 2026-04-26.
- **Qwen3-Next** weights were uploaded 2025-09-09 (HF commit history). The README went public and the announcement followed on 2025-09-11 (used here).

<!-- SOURCES_TABLE_START -->
| Date | Mechanism | Primary source | How the date was checked |
|---|---|---|---|
| 2014-09-01 | Additive attention (Bahdanau) *(added)* | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | arXiv 1409.0473 v1 (arXiv API) |
| 2017-05-08 | Absolute learned position embeddings | [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) | arXiv 1705.03122 v1 (arXiv API) |
| 2017-06-12 | Scaled dot-product attention (multi-head, causal) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | arXiv 1706.03762 v1 (arXiv API) |
| 2017-06-12 | Sinusoidal position encoding | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | arXiv 1706.03762 v1 (arXiv API) |
| 2018-03-06 | Relative position representations (Shaw) *(added)* | [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) | arXiv 1803.02155 v1 (arXiv API) |
| 2019-01-09 | Transformer-XL (segment recurrence) *(added)* | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) | arXiv 1901.02860 v1 (arXiv API) |
| 2019-04-23 | Sparse Transformer (strided / fixed patterns) *(added)* | [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) | arXiv 1904.10509 v1 (arXiv API) |
| 2019-10-23 | T5 relative position bias *(added)* | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | arXiv 1910.10683 v1 (arXiv API) |
| 2019-11-06 | Multi-query attention (MQA) | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | arXiv 1911.02150 v1 (arXiv API) |
| 2019-11-13 | Compressive Transformer (compressed memory) *(added)* | [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507) | arXiv 1911.05507 v1 (arXiv API) |
| 2019-12-25 | Top-k sparse attention (Explicit Sparse Transformer) | [Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection](https://arxiv.org/abs/1912.11637) | arXiv 1912.11637 v1 (arXiv API) |
| 2020-01-13 | Reformer (LSH attention) *(added)* | [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) | arXiv 2001.04451 v1 (arXiv API) |
| 2020-04-10 | Sliding-window (local) attention (Longformer) | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) | arXiv 2004.05150 v1 (arXiv API) |
| 2020-06-08 | Linformer (low-rank sequence projection) *(added)* | [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) | arXiv 2006.04768 v1 (arXiv API) |
| 2020-06-29 | Linear attention (Transformers are RNNs) | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) | arXiv 2006.16236 v1 (arXiv API) |
| 2020-07-28 | BigBird (window + global + random) *(added)* | [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) | arXiv 2007.14062 v1 (arXiv API) |
| 2020-09-30 | Performer (FAVOR+ random features) *(added)* | [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) | arXiv 2009.14794 v1 (arXiv API) |
| 2021-02-22 | Delta rule for linear attention (fast weights) | [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) | arXiv 2102.11174 v1 (arXiv API) |
| 2021-04-20 | Rotary position embedding (RoPE) | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | arXiv 2104.09864 v1 (arXiv API) |
| 2021-08-27 | ALiBi (attention with linear biases) | [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) | arXiv 2108.12409 v1 (arXiv API) |
| 2022-03-30 | NoPE (no positional encoding, causal) *(added)* | [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634) | arXiv 2203.16634 v1 (arXiv API) |
| 2022-05-27 | FlashAttention (IO-aware exact attention) *(added)* | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | arXiv 2205.14135 v1 (arXiv API) |
| 2023-05-22 | Grouped-query attention (GQA) | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | arXiv 2305.13245 v1 (arXiv API) |
| 2023-06-24 | KV-cache eviction (H2O, SnapKV) *(added)* | [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) | arXiv 2306.14048 v1 (arXiv API) |
| 2023-06-27 | Position Interpolation (linear RoPE scaling) *(added)* | [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) | arXiv 2306.15595 v1 (arXiv API) |
| 2023-06-29 | NTK-aware RoPE scaling | [NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | PullPush Reddit archive API: created_utc 2023-06-29 08:21:29 UTC |
| 2023-07-17 | RetNet (retention) *(added)* | [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) | arXiv 2307.08621 v1 (arXiv API) |
| 2023-08-31 | YaRN (NTK-by-parts + temperature) | [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) | arXiv 2309.00071 v1 (arXiv API) |
| 2023-09-12 | PagedAttention (vLLM) *(added)* | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | arXiv 2309.06180 v1 (arXiv API) |
| 2023-09-29 | Attention sinks (StreamingLLM) | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | arXiv 2309.17453 v1 (arXiv API) |
| 2023-10-03 | Ring Attention (sequence-parallel exact attention) *(added)* | [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) | arXiv 2310.01889 v1 (arXiv API) |
| 2023-12-01 | Mamba (selective state space) *(added)* | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | arXiv 2312.00752 v1 (arXiv API) |
| 2023-12-11 | Gated Linear Attention (GLA) *(added)* | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | arXiv 2312.06635 v1 (arXiv API) |
| 2024-02-05 | KV-cache quantisation (KIVI) *(added)* | [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) | arXiv 2402.02750 v1 (arXiv API) |
| 2024-02-21 | LongRoPE (searched non-uniform scaling) *(added)* | [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) | arXiv 2402.13753 v1 (arXiv API) |
| 2024-03-28 | Hybrid depth schedules (Jamba) *(added)* | [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) | arXiv 2403.19887 v1 (arXiv API) |
| 2024-05-07 | Multi-head latent attention (MLA) | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | arXiv 2405.04434 v1 (arXiv API) |
| 2024-05-31 | Mamba-2 / State Space Duality *(added)* | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | arXiv 2405.21060 v1 (arXiv API) |
| 2024-06-10 | DeltaNet, parallelised *(added)* | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | arXiv 2406.06484 v1 (arXiv API) |
| 2024-10-07 | Differential attention *(added)* | [Differential Transformer](https://arxiv.org/abs/2410.05258) | arXiv 2410.05258 v1 (arXiv API) |
| 2024-12-09 | Gated DeltaNet | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | arXiv 2412.06464 v1 (arXiv API) |
| 2024-12-31 | Titans (test-time learned memory) *(added)* | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) | arXiv 2501.00663 v1 (arXiv API) |
| 2025-01-14 | Lightning Attention hybrid at scale (MiniMax-01) *(added)* | [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) | arXiv 2501.08313 v1 (arXiv API) |
| 2025-02-16 | Native Sparse Attention (NSA) *(added)* | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) | arXiv 2502.11089 v1 (arXiv API) |
| 2025-02-18 | MoBA (mixture of block attention) *(added)* | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | arXiv 2502.13189 v1 (arXiv API) |
| 2025-05-10 | Gated attention (output gate, sink-free) *(added)* | [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) | arXiv 2505.06708 v1 (arXiv API) |
| 2025-09-11 | Qwen3-Next (3:1 Gated DeltaNet : gated attention) *(added)* | [Qwen3-Next-80B-A3B model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | HF repo commit history (weights uploaded 2025-09-09, README published 2025-09-11 03:40 UTC); public announcement 2025-09-11 UTC |
| 2025-09-29 | DeepSeek Sparse Attention (DSA, lightning indexer) *(added)* | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | GitHub repo created 2025-09-29 03:25 UTC; HF repo 2025-09-29 06:07 UTC. Full paper later: arXiv 2512.02556 (2025-12-02) |
| 2025-10-30 | Back to full attention (MiniMax-M2) *(added)* | [Why Did MiniMax M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) | Publication timestamp on the page |
| 2025-10-30 | Kimi Linear (KDA + MLA, 3:1) *(added)* | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | arXiv 2510.26692 v1 (arXiv API) |
| 2025-12-13 | DroPE (drop positional embeddings after pretraining) | [Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings](https://arxiv.org/abs/2512.12167) | arXiv 2512.12167 v1 (arXiv API) |
| 2026-04-24 | Compressed Sparse Attention + Heavily Compressed Attention (DeepSeek-V4) | [DeepSeek-V4 Preview Release](https://api-docs.deepseek.com/news/news260424) | Official DeepSeek news post dated 2026/04/24; HF repos created 2026-04-22 (private upload); technical report arXiv 2606.19348 v1 submitted 2026-04-26 |
<!-- SOURCES_TABLE_END -->
