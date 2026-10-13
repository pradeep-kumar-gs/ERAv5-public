import * as L from "./lib.js";

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
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
const f2 = (x) => (x === -Infinity ? "-∞" : Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(2));

function tabs(names, onPick, initial) {
  const wrap = h("div", { class: "tabs", role: "group" });
  const btns = names.map(([id, label]) =>
    h("button", { type: "button", "aria-pressed": id === initial ? "true" : "false", onclick: () => pick(id) }, label));
  function pick(id) {
    btns.forEach((b, i) => b.setAttribute("aria-pressed", names[i][0] === id ? "true" : "false"));
    onPick(id);
  }
  wrap.append(...btns);
  return { el: wrap, pick };
}

function slider(label, { min, max, step = 1, value }, onInput, fmt = (v) => v) {
  const out = h("span", { class: "mono" }, fmt(value));
  const inp = h("input", { type: "range", min, max, step, value, "aria-label": label });
  inp.addEventListener("input", () => { out.textContent = fmt(+inp.value); onInput(+inp.value); });
  return { el: h("label", {}, label, inp, out), set: (v) => { inp.value = v; out.textContent = fmt(v); } };
}

function heatTable(M, { rowLabels, colLabels, sel = -1, onRow, maxAbs } = {}) {
  const m = maxAbs ?? Math.max(1e-9, ...M.flat().filter(Number.isFinite).map(Math.abs));
  const t = h("table", { class: "mat" });
  if (colLabels) t.append(h("tr", {}, h("th", {}), ...colLabels.map((c) => h("th", {}, c))));
  M.forEach((row, i) => {
    const tr = h("tr", { class: i === sel ? "sel" : "" },
      h("th", { style: onRow ? "cursor:pointer" : "", onclick: onRow ? () => onRow(i) : null }, rowLabels ? rowLabels[i] : i));
    row.forEach((x) => {
      const td = h("td", {}, f2(x));
      if (x === -Infinity) td.className = "neg-inf";
      else td.style.background = `rgba(var(--heat), ${Math.min(1, Math.abs(x) / m) * 0.55})`;
      tr.append(td);
    });
    t.append(tr);
  });
  return h("div", { class: "scroll-x" }, t);
}

// ------------------------------------------------------------ standard attention
export function mountStandard(root) {
  const toks = ["The", "cat", "sat", "on", "the", "mat"];
  const emb = L.randMatrix(6, 4, 3, 1);
  emb[4] = [...emb[0]]; // "the" twice: identical vectors
  const Wq = L.randMatrix(4, 4, 21, 1.1), Wk = L.randMatrix(4, 4, 22, 1.1), Wv = L.randMatrix(4, 4, 23, 1);
  let step = "score", causal = true, q = 2;
  const body = h("div");
  const read = h("div", { class: "readout" });
  const steps = [["qkv", "1 · Q, K, V"], ["score", "2 · QKᵀ"], ["scale", "3 · ÷√dₖ"], ["mask", "4 · mask"], ["softmax", "5 · softmax"], ["out", "6 · ·V"]];
  const tb = tabs(steps, (id) => { step = id; draw(); }, step);
  const mask = h("input", { type: "checkbox", checked: true });
  mask.addEventListener("change", () => { causal = mask.checked; draw(); });

  function draw() {
    const A = L.attentionSteps(emb, Wq, Wk, Wv, causal);
    const lab = (name, el) => h("div", {}, h("div", { class: "lbl" }, name), el);
    const pick = (i) => { q = i; draw(); };
    const opts = { rowLabels: toks, colLabels: toks, sel: q, onRow: pick };
    const dims = ["d0", "d1", "d2", "d3"];
    body.replaceChildren();
    const mats = h("div", { class: "mats" });
    if (step === "qkv") {
      mats.append(lab("X (token vectors)", heatTable(emb, { rowLabels: toks, colLabels: dims, sel: q, onRow: pick })),
        lab("Q = X·Wq", heatTable(A.Q, { rowLabels: toks, colLabels: dims, sel: q, onRow: pick })),
        lab("K = X·Wk", heatTable(A.K, { rowLabels: toks, colLabels: dims, sel: q, onRow: pick })),
        lab("V = X·Wv", heatTable(A.V, { rowLabels: toks, colLabels: dims, sel: q, onRow: pick })));
      read.textContent = `"The" (row 0) and "the" (row 4) enter with identical vectors, so their q, k and v are identical too.\nNothing here knows that one came first. That is why position has to be added (see 2017 onward).`;
    } else if (step === "score") {
      mats.append(lab("S = Q·Kᵀ  (6 × 6 = 36 scores)", heatTable(A.S, opts)));
      read.textContent = `Row "${toks[q]}": one dot product per key → ${A.S[q].map(f2).join(", ")}\nEvery query is compared with every key. T tokens → T² scores: the compute bill.`;
    } else if (step === "scale") {
      mats.append(lab(`S / √${A.dk}`, heatTable(A.Ss, opts)));
      read.textContent = `Divide by √dₖ = ${Math.sqrt(A.dk)} so softmax doesn't saturate as head width grows.`;
    } else if (step === "mask") {
      mats.append(lab(causal ? "S/√dₖ + M  (future = −∞)" : "no mask (decoder would cheat)", heatTable(A.Sm, opts)));
      read.textContent = causal
        ? `Row "${toks[q]}" may read tokens 0..${q}. Future cells are −∞, so softmax gives them exactly 0.`
        : `Mask OFF: row "${toks[q]}" can see tokens after it. A trained decoder would put weight on the answer it is supposed to predict.`;
    } else if (step === "softmax") {
      mats.append(lab("P = softmax(row-wise)", heatTable(A.P, { ...opts, maxAbs: 1 })));
      const fut = A.P[q].slice(q + 1).reduce((a, b) => a + b, 0);
      read.textContent = `Row "${toks[q]}" weights: ${A.P[q].map((x) => x.toFixed(2)).join(", ")}  (sum = 1.00)\nWeight on future tokens: ${fut.toFixed(2)}${causal ? "  ← exactly zero with the mask" : "  ← leak!"}\nThe shared denominator ties every weight to every other score. That is what stops linear-time regrouping.`;
    } else {
      mats.append(lab("P", heatTable(A.P, { ...opts, maxAbs: 1 })), lab("V", heatTable(A.V, { rowLabels: toks, colLabels: dims })),
        lab("O = P·V", heatTable(A.O, { rowLabels: toks, colLabels: dims, sel: q, onRow: pick })));
      read.textContent = `Output for "${toks[q]}" = Σ pⱼ·vⱼ = [${A.O[q].map(f2).join(", ")}]\nDuring generation, every earlier k and v must be kept to do this for the next token: the KV cache.`;
    }
    body.append(mats);
  }
  root.append(
    h("h3", {}, "Standard attention, every number visible"),
    h("p", { class: "cap" }, "Six tokens, 4-dimensional head. Step through Q·Kᵀ → scale → mask → softmax → ·V. Click a row label to follow one query. Turn the mask off and watch weight leak to the future."),
    tb.el, h("div", { class: "controls" }, h("label", {}, mask, "causal mask")), body, read);
  draw();
  return { preset: () => tb.pick("softmax") };
}

// ------------------------------------------------------------ the two bills
export function mountKV(root) {
  const Ts = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
  let ti = 5, users = 1, dtype = "bf16", logScale = true, focus = null;
  const bars = h("div", { class: "bars" });
  const read = h("div", { class: "readout" });
  const fmtT = (i) => (Ts[i] >= 1048576 ? "1M" : `${Ts[i] / 1024}K`);
  const tS = slider("context", { min: 0, max: Ts.length - 1, value: ti }, (v) => { ti = v; draw(); }, fmtT);
  const uS = slider("active users", { min: 1, max: 64, value: users }, (v) => { users = v; draw(); });
  const dt = h("select", { "aria-label": "cache dtype" }, ...Object.keys(L.DTYPE_BYTES).map((k) => h("option", { value: k, selected: k === dtype }, k)));
  dt.addEventListener("change", () => { dtype = dt.value; draw(); });
  const lg = h("input", { type: "checkbox", checked: true });
  lg.addEventListener("change", () => { logScale = lg.checked; draw(); });

  function draw() {
    const T = Ts[ti];
    const rows = L.cacheBySchemes({ T, users, bytes: L.DTYPE_BYTES[dtype] });
    const max = Math.max(...rows.map((r) => r.b));
    const min = Math.min(...rows.map((r) => r.b));
    bars.replaceChildren(...rows.map((r) => {
      const frac = logScale ? (Math.log10(r.b) - Math.log10(min) + 0.3) / (Math.log10(max) - Math.log10(min) + 0.3) : r.b / max;
      return h("div", { class: "row" + (focus && !focus.includes(r.id) ? " dim" : "") },
        h("span", {}, r.label), h("div", { class: "bar" }, h("span", { style: `width:${(frac * 100).toFixed(1)}%` })), h("span", { class: "val" }, L.fmtBytes(r.b)));
    }));
    const pairs = (T * (T + 1)) / 2;
    read.textContent =
      `Bill 2 (memory) above: KV cache for ${users} user(s) at ${fmtT(ti)} tokens, ${dtype}. Weights not included.\n` +
      `Bill 1 (compute), query-key pairs per head per layer for one prefill:\n` +
      `  dense causal          ${pairs.toExponential(2)}\n` +
      `  sliding window 4096   ${(T * Math.min(T, 4096)).toExponential(2)}\n` +
      `  top-k 2048 (main)     ${(T * Math.min(T, 2048)).toExponential(2)}  + an indexer that still scores ~${pairs.toExponential(1)} pairs, cheaply\n` +
      `  linear / fixed state  ${(T * 128).toExponential(2)}  (≈ T·d state updates)`;
  }
  root.append(
    h("h3", {}, "The two bills: KV cache and score pairs"),
    h("p", { class: "cap" }, "One reference geometry for every scheme: 48 layers, 64 query heads, head_dim 128, so only the attention choice changes. At 32K with GQA-8 in bf16 this reproduces the course's 6.44 GB per user. MLA uses DeepSeek-V2's 512+64 latent. The CSA/HCA row is an illustration (half the layers at 1/4 positions, half at 1/128, each with a 128-token window), not DeepSeek-V4's exact cache."),
    h("div", { class: "controls" }, tS.el, uS.el, h("label", {}, "dtype", dt), h("label", {}, lg, "log scale")), bars, read);
  draw();
  return { preset: (ids) => { focus = ids ? ids.split(",") : null; draw(); } };
}

// ------------------------------------------------------------ positions
export function mountPosition(root) {
  let mode = "rope";
  const body = h("div");
  const read = h("div", { class: "readout" });
  const tb = tabs([["learned", "Learned table"], ["sinusoidal", "Sinusoidal"], ["rope", "RoPE"], ["alibi", "ALiBi"]], (id) => { mode = id; draw(); }, mode);
  const st = { m: 8, n: 2, theta: 0.3, on: true, maxLen: 32, head: 0 };

  function draw() {
    body.replaceChildren();
    if (mode === "learned") {
      const S = slider("query position", { min: 0, max: 63, value: st.qpos ?? 20 }, (v) => { st.qpos = v; draw(); });
      const qp = st.qpos ?? 20;
      const strip = h("div", { style: "display:grid;grid-template-columns:repeat(32,1fr);gap:2px;max-width:640px" },
        ...Array.from({ length: 64 }, (_, i) => h("div", {
          title: `position ${i}`,
          style: `height:18px;border-radius:2px;background:${i >= st.maxLen ? "var(--cell-off)" : `rgba(var(--heat),${0.25 + 0.5 * ((i * 7) % 11) / 11})`};${i === qp ? "outline:2px solid var(--ink)" : ""}`,
        })));
      body.append(h("div", { class: "controls" }, S.el), strip);
      read.textContent = qp < st.maxLen
        ? `Position ${qp} → row P[${qp}] of a ${st.maxLen}-row trained table, added to the token vector.`
        : `Position ${qp} → there is no row ${qp}. The table has ${st.maxLen} rows. This is the hard wall: no extrapolation, ever.`;
    } else if (mode === "sinusoidal") {
      const W = 64, D = 32, cell = 8;
      const svg = s("svg", { viewBox: `0 0 ${W * cell + 40} ${D * cell + 30}`, style: "max-width:600px;width:100%" });
      for (let p = 0; p < W; p++) {
        const v = L.sinusoidal(p * 4, D);
        v.forEach((x, d) => svg.append(s("rect", { x: 30 + p * cell, y: d * cell, width: cell, height: cell, fill: x >= 0 ? "var(--c1)" : "var(--c2)", "fill-opacity": Math.abs(x).toFixed(2) })));
      }
      svg.append(s("text", { x: 30, y: D * cell + 20, "font-size": 11 }, "position →  (0 … 252, step 4)"), s("text", { x: 0, y: 10, "font-size": 11 }, "dim"));
      body.append(svg);
      read.textContent = "Each row pair is a clock: top rows tick fast (neighbours differ), bottom rows tick slowly (far positions differ).\nDefined at every position, but the ALiBi paper shows models using it still fail beyond their training length.";
    } else if (mode === "rope") {
      const ctr = h("div", { class: "controls" });
      const mS = slider("query pos m", { min: 0, max: 40, value: st.m }, (v) => { st.m = v; paint(); });
      const nS = slider("key pos n", { min: 0, max: 40, value: st.n }, (v) => { st.n = v; paint(); });
      const thS = slider("θ (pair freq)", { min: 0.02, max: 1, step: 0.02, value: st.theta }, (v) => { st.theta = v; paint(); }, (v) => v.toFixed(2));
      const on = h("input", { type: "checkbox", checked: st.on });
      on.addEventListener("change", () => { st.on = on.checked; paint(); });
      const shift = h("button", { type: "button", class: "icon-btn", onclick: () => { st.m = Math.min(40, st.m + 10); st.n = Math.min(40, st.n + 10); mS.set(st.m); nS.set(st.n); paint(); } }, "shift both +10");
      const reset = h("button", { type: "button", class: "icon-btn", onclick: () => { st.m = 8; st.n = 2; mS.set(8); nS.set(2); paint(); } }, "reset");
      ctr.append(mS.el, nS.el, thS.el, h("label", {}, on, "RoPE on"), shift, reset);
      const svg = s("svg", { viewBox: "-110 -110 220 220", style: "max-width:300px;width:100%" });
      body.append(ctr, h("div", { class: "grid2" }, svg, h("div", { class: "explain" },
        "q and k are the same content vectors at every position. RoPE rotates q by m·θ and k by n·θ. The dot product depends only on the angle between them, so only n − m matters. Press ",
        h("b", {}, "shift both"), ": the absolute angles change, the score does not.")));
      const q = [0.9, 0.25], k = [0.75, -0.35];
      function paint() {
        svg.replaceChildren(s("circle", { r: 90, fill: "none", class: "axis" }), s("line", { x1: -100, x2: 100, y1: 0, y2: 0, class: "axis" }), s("line", { y1: -100, y2: 100, x1: 0, x2: 0, class: "axis" }));
        const a = st.on ? st.m * st.theta : 0, b = st.on ? st.n * st.theta : 0;
        const rq = L.rotate(q, a), rk = L.rotate(k, b);
        const arrow = ([x, y], col, lab) => [s("line", { x1: 0, y1: 0, x2: x * 90, y2: -y * 90, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }), s("text", { x: x * 98, y: -y * 98, "font-size": 12, fill: col }, lab)];
        svg.append(...arrow(rq, "var(--c1)", `q@${st.m}`), ...arrow(rk, "var(--c2)", `k@${st.n}`));
        const score = L.dot(rq, rk);
        read.textContent = st.on
          ? `rotation q = ${st.m}·${st.theta.toFixed(2)} = ${a.toFixed(2)} rad, k = ${b.toFixed(2)} rad\nrelative angle = (n − m)·θ = ${((st.n - st.m) * st.theta).toFixed(2)} rad   score q·k = ${score.toFixed(4)}\ndistance n − m = ${st.n - st.m}`
          : `RoPE off: score = ${score.toFixed(4)} whatever the positions. Content alone cannot tell near from far.`;
      }
      paint();
      return;
    } else {
      const H = 8, T = 16, slopes = L.alibiSlopes(H);
      const hS = slider("head", { min: 0, max: H - 1, value: st.head }, (v) => { st.head = v; draw(); }, (v) => `${v} (m = 1/${2 ** (v + 1)})`);
      const m = slopes[st.head];
      const M = Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => (j > i ? -Infinity : -m * (i - j))));
      body.append(h("div", { class: "controls" }, hS.el), heatTable(M, { maxAbs: Math.max(1e-9, m * (T - 1)) }));
      read.textContent = `Bias added to the logit = −m·(i − j). Head ${st.head} slope m = ${m.toFixed(4)}.\nSteep heads are effectively local windows; shallow heads see further. Far tokens are always penalised: the recency bias that hurts long-range retrieval.`;
    }
  }
  root.append(h("h3", {}, "Four ways to tell attention where tokens are"), h("p", { class: "cap" }, "Attention is order-blind (Section 1, step 1). Each scheme below is a different answer, and a different failure past the training length."), tb.el, body, read);
  draw();
  return { preset: (id) => tb.pick(id || "rope") };
}

// ------------------------------------------------------------ masks / sparsity
export function mountMasks(root) {
  const info = {
    full: ["Dense causal (2017)", "Every query reads every earlier key. Exact, T²/2 pairs."],
    strided: ["Sparse Transformer (2019)", "Local block + every stride-th token. Two hops reach anything; content-blind."],
    window: ["Sliding window (2020; Mistral 2023)", "Only the last w tokens. Cache capped at w per layer; anything older is invisible to this layer."],
    bigbird: ["BigBird (2020)", "Window + global tokens + random links. O(T), with a theory guarantee; fixed pattern."],
    sinks: ["Window + attention sinks (2023)", "Keep the first tokens forever. Fixes windowed streaming collapse; does not add memory."],
    h2o: ["H2O eviction (2023)", "Recent window + the k 'heavy hitters' with the most accumulated attention. The earliest tokens win because they have been accumulating attention the longest (the same tokens StreamingLLM calls sinks). The middle of the context is evicted for good, even if a later query needs it."],
    topk: ["Top-k (2019 → DSA 2025)", "Each query keeps its k best keys. Catch: you scored all of them to find the best. DSA makes that scoring cheap (indexer); it does not remove it."],
    block: ["MoBA (2025)", "Score blocks by q·mean(K), keep the top blocks plus the current one. Routing is cheap; a lone needle can hide in an 'average' block."],
    nsa: ["NSA (2025)", "Selected blocks (chosen via compressed block summaries) + sliding window. A third branch (not drawn) reads all compressed summaries coarsely."],
  };
  let type = "window";
  const st = { T: 32, window: 4, k: 4, block: 4 };
  const grid = h("div", { class: "maskgrid" });
  const title = h("b"), cap = h("div", { class: "explain" }), read = h("div", { class: "readout" });
  const tb = tabs(Object.entries(info).map(([k, v]) => [k, v[0].split(" (")[0]]), (id) => { type = id; draw(); }, type);
  const ctr = h("div", { class: "controls" },
    slider("T", { min: 12, max: 48, value: st.T }, (v) => { st.T = v; draw(); }).el,
    slider("window w", { min: 1, max: 12, value: st.window }, (v) => { st.window = v; draw(); }).el,
    slider("k (keys / blocks)", { min: 1, max: 8, value: st.k }, (v) => { st.k = v; draw(); }).el);
  function draw() {
    const M = L.maskPattern(type, st.T, { window: st.window, k: st.k, block: st.block, stride: Math.max(2, Math.round(Math.sqrt(st.T))) });
    grid.style.gridTemplateColumns = `repeat(${st.T}, 1fr)`;
    grid.replaceChildren(...M.flat().map((x) => h("div", { class: x ? `m${x}` : "" })));
    const { read: r, dense } = L.maskStats(M);
    title.textContent = info[type][0];
    cap.replaceChildren(title, " · ", info[type][1]);
    read.textContent = `pairs read: ${r} of ${dense} dense causal pairs (${((100 * r) / dense).toFixed(0)}%)` +
      (type === "topk" ? `\npairs SCORED to choose them: ${dense} (100%)` : "");
  }
  root.append(h("h3", {}, "Who reads whom: masks from dense to sparse"), h("p", { class: "cap" }, "Rows are queries, columns are keys, time runs down and right. Coloured cells are the keys a query actually attends to."),
    tb.el, ctr, h("div", { class: "grid2" }, h("div", {}, grid, h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--c1)" }), "local / dense"), h("span", {}, h("i", { style: "background:var(--c2)" }), "sink / global"), h("span", {}, h("i", { style: "background:var(--c3)" }), "selected / random"))), h("div", {}, cap, read)));
  draw();
  return { preset: (id) => tb.pick(id || "window") };
}

// ------------------------------------------------------------ linear regrouping + delta rule
export function mountLinear(root) {
  let q = 2, soft = false, overlap = 0;
  const left = h("div"), right = h("div");
  const qS = slider("query q", { min: -3, max: 3, step: 0.5, value: q }, (v) => { q = v; drawL(); });
  const sm = h("input", { type: "checkbox" });
  sm.addEventListener("change", () => { soft = sm.checked; drawL(); });
  const oS = slider("key overlap cos(A,B)", { min: 0, max: 0.9, step: 0.1, value: 0 }, (v) => { overlap = v; drawR(); }, (v) => v.toFixed(1));
  const readL = h("div", { class: "readout" }), readR = h("div", { class: "readout" }), table = h("div");

  function drawL() {
    const r = L.regroupDemo({ q });
    const direct = soft ? r.soft : r.direct;
    readL.textContent =
      `keys [0.5, 1.0, 1.5]  values [10, 20, 30]\n` +
      `DIRECT  scores q·k = [${r.scores.join(", ")}]\n` +
      (soft ? `        softmax → [${r.weights.map((x) => x.toFixed(2)).join(", ")}]\n` : "") +
      `        output = ${direct.toFixed(2)}\n` +
      `STATE   S = Σ k·v = ${r.state}  (built before q arrives, never grows)\n` +
      `        output = q·S = ${r.regrouped.toFixed(2)}\n` +
      (Math.abs(direct - r.regrouped) < 1e-9 ? "✓ identical: the past can be folded into one fixed state" : "✗ different: softmax's shared denominator needs every individual score after q arrives");
  }
  function drawR() {
    const { A, B } = L.keyPair(overlap);
    const seq = [{ key: A, v: 40, n: "A ← 40" }, { key: B, v: 70, n: "B ← 70" }, { key: A, v: 55, n: "A ← 55" }];
    const add = L.runWrites(seq, "add"), del = L.runWrites(seq, "delta");
    const rows = seq.map((w, i) => {
      const ra = L.runWrites(seq.slice(0, i + 1), "add"), rd = L.runWrites(seq.slice(0, i + 1), "delta");
      return [ra.read(A), ra.read(B), rd.read(A), rd.read(B)];
    });
    table.replaceChildren(heatTable(rows, { rowLabels: seq.map((w) => w.n), colLabels: ["add: A", "add: B", "delta: A", "delta: B"], maxAbs: 100 }));
    readR.textContent = `wanted at the end: A=55, B=70\nadditive → A=${add.read(A).toFixed(1)}, B=${add.read(B).toFixed(1)}\ndelta    → A=${del.read(A).toFixed(1)}, B=${del.read(B).toFixed(1)}\n` +
      (overlap > 0 ? `With overlapping keys even the delta rule disturbs the other key. A d-dim state holds about d clean associations. That is why hybrids keep some exact attention.` : `Orthogonal keys: the delta rule writes only the +15 correction and lands exactly on 55.`);
  }
  left.append(h("b", {}, "Remove softmax → attention becomes a fixed state"), h("div", { class: "controls" }, qS.el, h("label", {}, sm, "softmax on")), readL);
  right.append(h("b", {}, "Additive writes vs the delta rule"), h("div", { class: "controls" }, oS.el), table, readR);
  root.append(h("h3", {}, "Linear attention and the delta rule, with the course's numbers"), h("p", { class: "cap" }, "Left: the regrouping that makes linear attention O(T), and exactly why softmax blocks it. Right: why an add-only state returns 95 instead of 55, and what the delta rule fixes (and doesn't)."), h("div", { class: "grid2" }, left, right));
  drawL(); drawR();
  return { preset: () => {} };
}

// ------------------------------------------------------------ RoPE context extension
export function mountExtend(root) {
  const Ss = [2, 4, 8, 16, 32, 64], Ls = [2048, 4096, 8192];
  let si = 2, li = 1;
  const svgWrap = h("div", { class: "scroll-x" });
  const read = h("div", { class: "readout" });
  const ctr = h("div", { class: "controls" },
    slider("scale s", { min: 0, max: Ss.length - 1, value: si }, (v) => { si = v; draw(); }, (v) => `${Ss[v]}×`).el,
    slider("train length", { min: 0, max: Ls.length - 1, value: li }, (v) => { li = v; draw(); }, (v) => `${Ls[v] / 1024}K`).el);
  function draw() {
    const sc = Ss[si], Lt = Ls[li];
    const e = L.extensionScales({ d: 128, s: sc, Ltrain: Lt });
    const W = 640, H = 260, pad = 44, n = e.pi.length;
    const x = (i) => pad + (i / (n - 1)) * (W - pad - 10);
    const y = (v) => 14 + (Math.log2(1 / v) / Math.log2(64)) * (H - 50);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: `width:100%;min-width:520px` });
    [1, 1 / 4, 1 / 16, 1 / 64].forEach((v) => svg.append(s("line", { x1: pad, x2: W - 10, y1: y(v), y2: y(v), class: "axis" }), s("text", { x: 4, y: y(v) + 4, "font-size": 11 }, v === 1 ? "1 (kept)" : `1/${1 / v}`)));
    const cross = e.wavelength.findIndex((w) => w > Lt);
    if (cross > 0) svg.append(s("line", { x1: x(cross), x2: x(cross), y1: 10, y2: H - 30, stroke: "var(--ink-3)", "stroke-dasharray": "4 4" }), s("text", { x: x(cross) + 4, y: H - 34, "font-size": 11 }, "wavelength > train length →"));
    const line = (arr, col, lab, dy) => {
      svg.append(s("polyline", { points: arr.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: col, "stroke-width": 2.5 }));
      svg.append(s("text", { x: W - 120, y: 24 + dy, "font-size": 12, fill: col }, lab));
    };
    line(e.pi, "var(--c2)", "PI (2023-06-27)", 0);
    line(e.ntk, "var(--c3)", "NTK-aware (06-29)", 16);
    line(e.yarn, "var(--c1)", "YaRN (08-31)", 32);
    svg.append(s("text", { x: pad, y: H - 8, "font-size": 11 }, "RoPE pair index →   fast (local detail)  …  slow (long distance)"));
    svgWrap.replaceChildren(svg);
    const kept = e.yarn.filter((v) => v > 0.999).length, full = e.yarn.filter((v) => Math.abs(v - 1 / sc) < 1e-9).length;
    read.textContent =
      `Each line shows how much each rotary pair's frequency is slowed (θ_new / θ_old) to stretch ${Lt / 1024}K → ${(Lt * sc) / 1024}K.\n` +
      `PI: every pair slowed by ${sc}×, including the fast pairs that resolve neighbours (local blur).\n` +
      `NTK-aware: base 10000 → ${e.ntkBase.toFixed(0)}. Fast pairs almost untouched, slow pairs ≈ ${sc}×. Middle pairs partly extrapolate.\n` +
      `YaRN: ${kept} pairs kept exactly, ${full} fully interpolated, ramp between. Plus logit temperature ×${(e.yarnTemp ** 2).toFixed(3)}.\n` +
      `DroPE (2025-12-13) takes the other exit: delete the rotation after pretraining, recalibrate briefly, and rely on the causal mask's implicit position.`;
  }
  root.append(h("h3", {}, "Stretching RoPE: PI vs NTK-aware vs YaRN"), h("p", { class: "cap" }, "Head dim 128 (64 rotary pairs), base 10000. Same goal for all three: no rotation angle at the new length should be one the model never saw. They differ in which pairs pay for it."), ctr, svgWrap, read);
  draw();
  return { preset: () => {} };
}
