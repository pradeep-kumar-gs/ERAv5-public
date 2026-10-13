import { mountStandard, mountKV, mountPosition, mountMasks, mountLinear, mountExtend } from "./widgets.js";

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
};
const SVG = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}, text) => {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
};

// ---------- theme (per-viewer convenience only) ----------
const themeBtn = document.getElementById("theme");
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch {}
themeBtn.addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch {}
});

const fmtDate = (d) => new Date(d + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const FIT_LABEL = { good: "good fit", ok: "workable", poor: "poor fit" };

const widgets = {};
const KV_FOCUS = {
  mqa: "mha,gqa,mqa", gqa: "mha,gqa,mqa", mla: "mha,gqa,mla", kivi: "gqa,kivi,mla", jamba: "gqa,hybrid,linear",
  "qwen3-next": "gqa,hybrid,linear", "kimi-linear": "gqa,mla,hybrid", "deepseek-v4-csa": "gqa,mla,csa",
};

function tryIt(m) {
  if (!m.widget) return null;
  const [kind, arg] = m.widget.split(":");
  const target = { standard: "w-standard", kv: "w-kv", position: "w-position", mask: "w-masks", linear: "w-linear", delta: "w-linear", extend: "w-extend" }[kind];
  const label = { standard: "See it computed", kv: "See its cache", position: "See it rotate / bias", mask: "See its mask", linear: "See the regrouping", delta: "See the delta write", extend: "Compare the scalings" }[kind];
  return h("div", { class: "try" }, h("button", {
    type: "button",
    onclick: () => {
      const w = widgets[target];
      if (w) w.preset(kind === "kv" ? KV_FOCUS[m.id] : arg);
      document.getElementById(target).scrollIntoView({ behavior: "smooth", block: "center" });
    },
  }, `${label} →`));
}

function card(m, fams) {
  const famColor = `var(--c-${m.family})`;
  return h("article", { class: "card", id: `m-${m.id}`, "data-family": m.family, style: `--famc:${famColor}` },
    h("div", { class: "meta" },
      h("span", { class: "date" }, fmtDate(m.date)),
      h("span", { class: "tag" }, h("span", { class: "fam", style: `background:${famColor}` }), fams[m.family]),
      h("span", { class: "tag" + (m.required ? " req" : "") }, m.required ? "course list" : "added")),
    h("h4", {}, m.name),
    h("div", { class: "src" }, "Source: ", h("a", { href: m.source.url, target: "_blank", rel: "noopener" }, m.source.title), ` · ${m.source.authors}`,
      m.source.kind === "arxiv" ? ` · arXiv ${m.source.id} v1` : ""),
    m.also ? h("ul", { class: "also" }, ...m.also.map((a) => h("li", {}, `${a.date}: `, h("a", { href: a.url, target: "_blank", rel: "noopener" }, a.label)))) : null,
    h("div", { class: "lbl" }, "The problem at that moment"), h("p", {}, m.problem),
    h("div", { class: "lbl" }, "Mechanism"), h("p", {}, m.how),
    h("div", { class: "tradeoff" },
      h("div", { class: "buys" }, h("div", { class: "lbl" }, "What it buys"), h("ul", {}, ...m.buys.map((x) => h("li", {}, x)))),
      h("div", { class: "costs" }, h("div", { class: "lbl" }, "What it gives up"), h("ul", {}, ...m.costs.map((x) => h("li", {}, x))))),
    h("div", { class: "lbl" }, "When you would actually pick it"), h("p", {}, m.pick),
    m.fit ? h("div", { class: "fit" }, ...Object.entries(m.fit).map(([w, [v, why]]) =>
      h("span", { class: v, "data-w": w, title: why }, `${{ chat: "2K chat", rag: "128K RAG", agent: "1M agent" }[w]}: `, h("b", {}, FIT_LABEL[v]), ` · ${why}`))) : null,
    tryIt(m));
}

function overview(data) {
  const lanes = Object.keys(data.families);
  const W = 1000, laneH = 30, top = 62, H = top + lanes.length * laneH + 30, left = 130;
  const t0 = Date.UTC(2014, 0, 1), t1 = Date.UTC(2026, 11, 31);
  const x = (d) => left + ((Date.parse(d) - t0) / (t1 - t0)) * (W - left - 14);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Timeline of attention mechanisms by family, 2014 to 2026" });
  data.eras.forEach((e, i) => {
    svg.append(s("rect", { x: x(e.from), y: 8, width: x(e.to) - x(e.from), height: H - 30, fill: i % 2 ? "var(--surface-2)" : "transparent" }));
  });
  data.eras.forEach((e, i) => {
    const lab = s("text", { x: x(e.from) + 4, y: 20 + (i % 3) * 14, "font-size": 11, "font-weight": 600 }, e.short);
    lab.append(s("title", {}, `${e.name}: ${e.wants}`));
    svg.append(lab);
  });
  lanes.forEach((f, i) => {
    const y = top + i * laneH + laneH / 2;
    svg.append(s("line", { x1: left, x2: W - 10, y1: y, y2: y, class: "axis" }), s("text", { x: 6, y: y + 4, "font-size": 12 }, data.families[f]));
  });
  for (let yr = 2014; yr <= 2026; yr++) svg.append(s("text", { x: x(`${yr}-01-01`), y: H - 8, "font-size": 11 }, yr));
  const hover = document.getElementById("ov-hover");
  const placed = {};
  const dots = data.mechanisms.map((m) => {
    const li = lanes.indexOf(m.family);
    const cx = x(m.date);
    const key = `${li}`;
    placed[key] = placed[key] || [];
    const near = placed[key].filter((p) => Math.abs(p - cx) < 11).length;
    placed[key].push(cx);
    const cy = top + li * laneH + laneH / 2 + (near % 2 ? 8 : near ? -8 : 0);
    const c = s("circle", { cx, cy, r: m.required ? 6.5 : 5, fill: `var(--c-${m.family})`, class: "dot", tabindex: 0, "data-family": m.family, "data-id": m.id });
    c.append(s("title", {}, `${m.date} · ${m.name}`));
    const show = () => (hover.textContent = `${fmtDate(m.date)} · ${m.name}. ${m.problem}`);
    c.addEventListener("mouseenter", show);
    c.addEventListener("focus", show);
    const go = () => {
      const el = document.getElementById(`m-${m.id}`);
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 1600);
    };
    c.addEventListener("click", go);
    c.addEventListener("keydown", (ev) => ev.key === "Enter" && go());
    svg.append(c);
    return c;
  });
  document.getElementById("overview").append(svg);
  return dots;
}

function filters(data, dots) {
  const el = document.getElementById("filters");
  let fam = null, work = null;
  const apply = () => {
    document.querySelectorAll(".card").forEach((c) => {
      const m = data.mechanisms.find((x) => `m-${x.id}` === c.id);
      const famOk = !fam || m.family === fam;
      const workOk = !work || (m.fit && m.fit[work][0] === "good");
      c.classList.toggle("dim", !(famOk && workOk));
      c.querySelectorAll(".fit span").forEach((sp) => sp.classList.toggle("hl", sp.dataset.w === work));
    });
    dots.forEach((d) => {
      const m = data.mechanisms.find((x) => x.id === d.dataset.id);
      const ok = (!fam || m.family === fam) && (!work || (m.fit && m.fit[work][0] === "good"));
      d.classList.toggle("dim", !ok);
    });
  };
  const famBtns = Object.entries(data.families).map(([k, v]) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": "false" }, h("span", { class: "fam", style: `background:var(--c-${k})` }), v);
    b.addEventListener("click", () => { fam = fam === k ? null : k; famBtns.forEach((x) => x.setAttribute("aria-pressed", x === b && fam ? "true" : "false")); apply(); });
    return b;
  });
  const workBtns = Object.entries(data.workloads).map(([k, v]) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": "false", title: v }, { chat: "2K chatbot", rag: "128K doc QA", agent: "1M agent" }[k]);
    b.addEventListener("click", () => { work = work === k ? null : k; workBtns.forEach((x) => x.setAttribute("aria-pressed", x === b && work ? "true" : "false")); apply(); });
    return b;
  });
  el.append(h("span", { class: "lab" }, "Highlight family:"), ...famBtns);
  el.append(h("div", { style: "flex-basis:100%;height:0" }), h("span", { class: "lab" }, "Highlight what's a good fit for:"), ...workBtns,
    h("span", { class: "small muted" }, "(the order never changes; it is always chronological)"));
}

const INTERLUDES = { alibi: ["w-position", mountPosition], "sliding-window": ["w-masks", mountMasks], "delta-rule": ["w-linear", mountLinear], yarn: ["w-extend", mountExtend] };

function cards(data) {
  const root = document.getElementById("cards");
  let era = null;
  for (const m of data.mechanisms) {
    if (m.era !== era) {
      era = m.era;
      const e = data.eras.find((x) => x.id === era);
      root.append(h("div", { class: "era" },
        h("div", { class: "years" }, `${e.from.slice(0, 4)}${e.to.slice(0, 4) !== e.from.slice(0, 4) ? "–" + e.to.slice(0, 4) : ""}`),
        h("h3", {}, e.name), h("div", { class: "wants" }, "The field wants: ", h("b", {}, e.wants)), h("p", {}, e.story)));
    }
    root.append(card(m, data.families));
    if (INTERLUDES[m.id]) {
      const [id, mount] = INTERLUDES[m.id];
      const w = h("div", { class: "widget interlude", id });
      root.append(w);
      widgets[id] = mount(w);
    }
  }
}

function reading(data) {
  document.getElementById("reading-body").innerHTML = `
  <div class="kicker">Section 4 · reading the timeline</div>
  <h2 class="sec-title">What the order shows that a list hides</h2>
  <p>Read the cards top to bottom and the field changes its mind four times:</p>
  <ol>
    <li><b>Exactness (2017).</b> Recurrence is dropped for exact all-pairs lookup. The immediate bills are T² compute and lost order.</li>
    <li><b>Length, and memory comes back (2019–2021).</b> Sparse patterns, windows and top-k attack T². Linear attention and fast weights turn attention back into an RNN with a fixed state. MQA notices the decoding bill. RoPE and ALiBi redesign position so it can survive longer inputs.</li>
    <li><b>Exactness wins again (2022 to mid-2023).</b> FlashAttention makes exact attention cheap on real hardware and most approximations lose. GQA fixes enough of the cache to become everyone's default.</li>
    <li><b>Length via position (mid-2023).</b> Models are trained short and stretched: PI, NTK-aware and YaRN within ten weeks. Attention sinks explain why windows collapse.</li>
    <li><b>Memory back again (late 2023–2024).</b> The cache is now the dominant serving bill. Fixed state returns with learned forgetting and correction (Mamba, GLA, DeltaNet, Gated DeltaNet), and MLA compresses the cache while keeping exact access.</li>
    <li><b>Mix, sparsify, compress (2025–2026).</b> Hybrids ship (MiniMax-01, Qwen3-Next, Kimi Linear), sparsity becomes trainable (NSA, MoBA, DSA), and DeepSeek-V4 compresses the sequence itself. The field also reverses in public: MiniMax-M2 goes back to full attention the same day Kimi Linear argues the opposite.</li>
  </ol>
  <p><b>A second, quieter track runs alongside.</b> Not every answer to the bill changes the model. FlashAttention (2022), PagedAttention and H2O eviction (2023), KIVI 2-bit caches (2024) and Ring Attention (2023) attack the same two bills in kernels, memory managers and clusters. They often buy more in practice than an architecture change, and they stack with it.</p>
  <div class="callout"><b>The pattern.</b> Every swing toward approximation (sparse, linear) ends in one of two ways. Either systems work makes exactness cheap again (FlashAttention, GQA), or evaluations expose lost recall (Mamba's copying gap, MiniMax-M2). Each new swing then keeps a little exact attention and approximates the rest. Hybrids are what that oscillation converges to.</div>
  <h3>Where it goes next (this author's prediction, not a fact)</h3>
  <ul>
    <li><b>The indexer is the next bill.</b> DSA and CSA still score every (compressed) key with a cheap indexer, and that piece is still quadratic. Expect hierarchical or multi-level indexing (coarse block → finer block → token) so selection itself becomes sub-quadratic.</li>
    <li><b>The exact layers of hybrids will themselves become compressed-sparse.</b> The "G" in a D-D-D-G schedule will read compressed, selected memory (CSA-like) instead of a full cache. That combines both families: fixed state for the bulk, compressed-exact for recall.</li>
    <li><b>Position becomes implicit.</b> NoPE layers, partial RoPE (V4 rotates only 64 dims) and DroPE all point the same way. Extension methods (PI/NTK/YaRN) will matter less as models train long natively or drop explicit position.</li>
    <li><b>Evaluation decides.</b> MiniMax-M2's lesson is that standard benchmarks hid multi-hop reasoning failures in hybrids. Whoever has the better long-context reasoning evals will decide which architecture wins.</li>
  </ul>
  <h3>Context length is counted in tokens, not documents</h3>
  <p>The same nominal window holds very different amounts of language. If a Telugu text needs about 3× the tokens of its English equivalent (Session 3's fertility numbers), a 256K-token window holds roughly a third as much Telugu. Every bill on this page is paid per token. For an India-first model, a better tokenizer is a long-context mechanism too.</p>
  <h3>If you had to ship today</h3>
  <table class="sources">
    <tr><th>Workload</th><th>Pick</th><th>Why not the fancier option</th></tr>
    <tr><td>2K-token chatbot, high QPS</td><td>Dense causal attention + RoPE + GQA (or MQA), FlashAttention, optional output gate</td><td>Sparse, compressed and linear layers are pure overhead at 2K. The cache is small, and exact recall is free.</td></tr>
    <tr><td>128K document QA</td><td>GQA or MLA + RoPE with YaRN (or native long training) + FlashAttention; NSA/DSA-style sparsity if prefill cost hurts</td><td>Pure linear and SSM layers cannot copy reliably from the document, and ALiBi buries far evidence.</td></tr>
    <tr><td>1M-token agent</td><td>A hybrid (Gated DeltaNet / KDA 3:1 with exact layers) or compressed-sparse attention (CSA/HCA), trained long natively</td><td>A GQA cache at 1M is tens of GB per user. Pure windows forget. Extension alone is near its ceiling.</td></tr>
  </table>`;
}

function caseStudy() {
  document.getElementById("case-body").innerHTML = `
  <div class="kicker">Section 5 · case study (course-sourced)</div>
  <h2 class="sec-title">How one real model combined these: LightningLM V4</h2>
  <p class="muted">These facts come from the ERA V5 Session 8 course notes (15 Aug 2026). They describe the course's own model, have no public paper, and cannot be dated or checked independently, so they are kept off the timeline. Each one is mapped to its public lineage.</p>
  <table class="sources">
    <tr><th>V4 choice (course)</th><th>Public lineage on the timeline</th><th>Honest status</th></tr>
    <tr><td><b>DDDGDDDG</b> depth schedule: 6 DeltaNet (D) + 2 sparse-attention (G) layers per 8. Kept from the 1.78B seed to 120B. Going from 1 G per 8 to 8 per 8 makes KV state 8.0× larger for about 1.41× mixing compute.</td><td>Jamba (2024-03-28) → Qwen3-Next 3:1 (2025-09-11) → Kimi Linear 3:1 (2025-10-30)</td><td>Worked across scale. Neighbouring ratios were not ablated, so it is a successful choice, not a proven optimum.</td></tr>
    <tr><td><b>Sparse G-layer budget</b> cut from 1024 to 256 because of backward-kernel contention, and varied per token.</td><td>Explicit top-k (2019) → NSA, MoBA (2025-02) → DSA top-2048 (2025-09-29)</td><td>An implementation constraint of that hardware/software stack, not a law. Re-measure on current kernels.</td></tr>
    <tr><td><b>Memory Stream</b>: one model-width vector carried across chunks, written with stop-gradient, injected through a learned per-token gate. Scale about 0.078 → 0.391 by the 2B stage, about 6% of the embedding magnitude.</td><td>Transformer-XL (2019-01-09) used the same stop-gradient trick for segment memory, at full width per token. Compressive Transformer (2019-11-13) compressed old memory instead of dropping it.</td><td>O(1) but a severe compression. Whether it still earns its place at very long native context is open.</td></tr>
    <tr><td><b>DroPE before annealing</b>: trained at 8K, reported 256K (32×).</td><td>NoPE (2022) → DroPE (2025-12-13)</td><td>A reported result for this model and procedure. A position function existing at 256K is not proof of capability there.</td></tr>
  </table>`;
}

function corrections() {
  const items = [
    ["fix", "DroPE's mechanism is public: it removes positional embeddings, not 'some rotary dimensions'",
      `The course (Section 9) says the record "does not establish the exact DroPE algorithm or which rotary dimensions it changes". The method is published: <a href="https://arxiv.org/abs/2512.12167">arXiv 2512.12167</a> (v1 13 Dec 2025, Sakana AI). It drops the positional embeddings entirely after pretraining and runs a short recalibration phase. The headline claim is zero-shot context extension without long-context fine-tuning. So the question is not which rotary dimensions change: all of them are removed. In the live class (transcript, ~01:35) the instructor described it correctly ("train with RoPE … then you can remove RoPE"), so it is the written notes that are out of step. The class also said "train to 8K and it gives us extension of 32k", while the notes say 8K → 256K (32×). 8K × 32 = 256K, so "32k" was presumably meant as "32×".`],
    ["fix", "Class claim: 'RoPE was invented where 32K was the target, 32K was achieved and nobody tested it'",
      `The RoFormer paper (<a href="https://arxiv.org/abs/2104.09864">arXiv 2104.09864</a>) pre-trains with a maximum sequence length of 512 and evaluates long documents at 1024. There is no 32K target or experiment in it. 32K-class RoPE contexts came two years later through extension work (PI, NTK-aware, YaRN, 2023) and base-frequency changes in long-context training.`],
    ["fix", "Class claim: DeepSeek trained for '$50 million' where OpenAI needed '$500 million'",
      `The figure DeepSeek actually published for V3 (<a href="https://arxiv.org/abs/2412.19437">arXiv 2412.19437</a>) is <b>$5.576M</b>: 2.788M H800 GPU hours at an assumed $2/GPU-hour. The report says this covers "only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments". So the class number is 10× the reported one, and the reported one is itself not the full cost. No OpenAI training cost has been published to compare against.`],
    ["date", "Class slip: 'Vaswani … invented in 2018 and 17'",
      `"Attention Is All You Need" is arXiv v1 <b>2017-06-12</b> (NeurIPS 2017). Small, but exactly the kind of date an agent will repeat confidently.`],
    ["nuance", "'Attention itself does not know token order' needs a qualifier for decoders",
      `True for bidirectional attention. With a causal mask, a decoder <em>can</em> infer position: token t averages over exactly t+1 tokens. Haviv et al. (<a href="https://arxiv.org/abs/2203.16634">2022-03-30</a>) trained competitive NoPE language models on this signal alone, and DroPE depends on exactly this effect. In Section 1's widget, "The" and "the" have identical q/k/v, yet their outputs differ once the mask is on.`],
    ["confirmed", "'DeepSeek-V4 applies RoPE to the last 64 dimensions': correct, and older than V4",
      `The V4 technical report (<a href="https://arxiv.org/abs/2606.19348">arXiv 2606.19348</a>) applies RoPE to the last 64 dims of queries and KV entries, keeping those dims in BF16 and the rest in FP8. The 64-dim partial ("decoupled") RoPE is inherited from MLA in DeepSeek-V2 (<a href="https://arxiv.org/abs/2405.04434">2024-05-07</a>).`],
    ["nuance", "'DeepSeek built for a million natively': native, but via a length curriculum",
      `V4's report trains at 4K → 16K → 64K → 1M. Dense attention runs first (the first 1T tokens for Flash), and sparse attention is switched on at 64K. So Road 2 ("build long, train long") still stages its length. The two roads in Section 17 are ends of a spectrum.`],
    ["confirmed", "CSA + 'heavily compressed dense form' + low-rank indexer: confirmed, with numbers",
      `Per the V4 report: CSA compresses m=4 tokens per entry (overlapping windows), the indexer (queries from a low-rank latent shared with main attention, run in FP4) picks top-512 (Flash) / 1024 (Pro) entries, and a 128-token uncompressed sliding window keeps recent detail. HCA compresses m'=128 and attends densely. At 1M, V4-Pro needs about 27% of V3.2's per-token FLOPs and 10% of its KV cache.`],
    ["date", "Date trap: YaRN is August 2023, not September",
      `arXiv ID 2309.00071 looks like September, but the ID is assigned at announcement. v1 was submitted on <b>31 Aug 2023</b>. Many write-ups say "Sep 2023". The same trap applies to any paper submitted at a month boundary.`],
    ["date", "Date trap: the NTK-aware 'paper' is a Reddit post, and PI was not first either",
      `NTK-aware scaling was posted by u/bloc97 on r/LocalLLaMA at <b>2023-06-29 08:21 UTC</b> (via the PullPush archive), two days after Meta's PI paper (<a href="https://arxiv.org/abs/2306.15595">2023-06-27</a>). The linear-interpolation idea itself was already on kaiokendev's SuperHOT blog, captured by the Wayback Machine on <b>2023-06-21</b>. An agent that dates NTK-aware by YaRN's citation, or says "Meta invented RoPE interpolation", gets the order wrong.`],
    ["date", "Date trap: RoPE predates its arXiv paper",
      `Jianlin Su published RoPE on his blog (kexue.fm/archives/8265) dated <b>2021-03-23</b>. RoFormer hit arXiv on 2021-04-20. The timeline uses the arXiv date for consistency and notes the blog.`],
    ["date", "Date trap: sliding-window attention is not a 2023 Mistral invention",
      `Mistral 7B (2023-10-10) popularised causal SWA with a rolling-buffer cache in open decoders. Local windows go back to Longformer (<a href="https://arxiv.org/abs/2004.05150">2020-04-10</a>), with banded patterns already in the Sparse Transformer (2019-04-23).`],
    ["date", "Date trap: DeepSeek-V4 has three candidate dates",
      `Hugging Face repos were created 2026-04-22 (private upload), the official release post is <b>2026-04-24</b>, and the arXiv report v1 is 2026-04-26. This page uses the public release. Similarly, Qwen3-Next weights were uploaded 2025-09-09 and announced on 2025-09-11.`],
    ["nuance", "Course-only numbers that this page cannot verify",
      `LightningLM V4's DDDGDDDG schedule, the 1024→256 sparse budget, the Memory Stream scales and the 8K→256K DroPE result have no public source. They are shown in the case study, labelled as course-sourced, and kept off the dated timeline.`],
  ];
  const label = { fix: "Correction", nuance: "Nuance", confirmed: "Checked: correct", date: "Date hazard" };
  const cls = { fix: "fix", nuance: "nuance", confirmed: "confirmed", date: "nuance" };
  document.getElementById("corr-body").append(
    h("div", { class: "kicker" }, "Section 6 · being right about it"),
    h("h2", { class: "sec-title" }, "Corrections and date hazards"),
    h("p", { class: "muted" }, "The assignment asked us to catch confident mistakes, including the instructor's. Every claim below was checked against a primary source; 'Checked: correct' means the course was right."),
    ...items.map(([k, title, body]) => h("div", { class: "corr" }, h("div", { class: `verdict ${cls[k]}` }, label[k]), h("h4", {}, title), h("p", { html: body }))));
}

function sources(data) {
  const kind = { arxiv: "arXiv v1", forum: "Forum post", release: "Official release", blog: "Official blog" };
  const tbl = h("table", { class: "sources" },
    h("tr", {}, h("th", {}, "Date"), h("th", {}, "Mechanism"), h("th", {}, "Primary source"), h("th", {}, "How the date was checked")),
    ...data.mechanisms.map((m) => h("tr", {},
      h("td", { class: "d" }, m.date),
      h("td", {}, h("a", { href: `#m-${m.id}` }, m.name)),
      h("td", {}, h("a", { href: m.source.url, target: "_blank", rel: "noopener" }, m.source.title)),
      h("td", { class: "small" }, m.source.kind === "arxiv" ? `${kind.arxiv} ${m.source.id}, arXiv API 'published' field` : `${kind[m.source.kind]}. ${m.source.verified_via || "Publication timestamp on the page"}`))));
  document.getElementById("src-body").append(
    h("div", { class: "kicker" }, "Section 7 · sources"),
    h("h2", { class: "sec-title" }, "Where every date comes from"),
    h("p", { class: "muted narrow" }, data.date_policy),
    h("div", { class: "scroll-x" }, tbl));
}

async function main() {
  const data = await (await fetch("data/mechanisms.json")).json();
  document.getElementById("n-mech").textContent = data.mechanisms.length;
  widgets["w-standard"] = mountStandard(document.getElementById("w-standard"));
  widgets["w-kv"] = mountKV(document.getElementById("w-kv"));
  cards(data);
  const dots = overview(data);
  filters(data, dots);
  reading(data);
  caseStudy();
  corrections();
  sources(data);
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}
main().catch((e) => {
  document.getElementById("cards").append(h("p", { class: "callout" }, `Could not load data/mechanisms.json (${e.message}). Serve this folder over HTTP rather than opening the file directly.`));
});
