// Pure maths behind every widget. No DOM access, so scripts/test_lib.mjs can check it in Node.

export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randMatrix(rows, cols, seed, scale = 1) {
  const r = rng(seed);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (r() * 2 - 1) * scale));
}

export const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
export const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

export function softmax(row) {
  const m = Math.max(...row);
  if (m === -Infinity) return row.map(() => 0);
  const e = row.map((x) => Math.exp(x - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
}

// ---------- standard attention, every intermediate kept ----------
export function attentionSteps(X, Wq, Wk, Wv, causal = true) {
  const Q = matmul(X, Wq), K = matmul(X, Wk), V = matmul(X, Wv);
  const dk = Q[0].length;
  const S = matmul(Q, transpose(K));
  const Ss = S.map((r) => r.map((x) => x / Math.sqrt(dk)));
  const Sm = Ss.map((r, i) => r.map((x, j) => (causal && j > i ? -Infinity : x)));
  const P = Sm.map(softmax);
  const O = matmul(P, V);
  return { Q, K, V, S, Ss, Sm, P, O, dk };
}

// ---------- KV cache ----------
export const DTYPE_BYTES = { fp32: 4, bf16: 2, fp8: 1 };

export function kvCacheBytes({ layers, kvHeads, headDim, T, users = 1, bytes = 2 }) {
  return 2 * layers * kvHeads * headDim * T * users * bytes;
}

// Illustrative per-user cache for one reference geometry. Every scheme shares the
// same depth so only the attention choice changes. Values are bytes.
export function cacheBySchemes({ T, users = 1, bytes = 2, layers = 48, qHeads = 64, headDim = 128 }) {
  const gqa = (kv, t = T, L = layers) => 2 * L * kv * headDim * t * bytes;
  const linearState = (L) => L * 16 * headDim * headDim * 4; // 16 heads of fp32 d x d state
  const window = 4096;
  const mlaWidth = 512 + 64;
  const half = layers / 2;
  const csa = half * mlaWidth * (Math.ceil(T / 4) + Math.min(T, 128)) * bytes;
  const hca = half * mlaWidth * (Math.ceil(T / 128) + Math.min(T, 128)) * bytes;
  const rows = [
    { id: "mha", label: `MHA (${qHeads} KV heads)`, b: gqa(qHeads) },
    { id: "gqa", label: "GQA (8 KV heads)", b: gqa(8) },
    { id: "mqa", label: "MQA (1 KV head)", b: gqa(1) },
    { id: "kivi", label: "GQA-8, 2-bit KV (KIVI-style)", b: gqa(8) * (0.25 / bytes) },
    { id: "mla", label: "MLA (512 latent + 64 RoPE)", b: layers * mlaWidth * T * bytes },
    { id: "swa", label: `Sliding window w=${window} (GQA-8)`, b: gqa(8, Math.min(T, window)) },
    { id: "hybrid", label: "Hybrid 3:1 linear : GQA-8", b: gqa(8, T, layers / 4) + linearState((3 * layers) / 4) },
    { id: "csa", label: "CSA + HCA style (illustrative)", b: csa + hca },
    { id: "linear", label: "Pure linear / fixed state", b: linearState(layers) },
  ];
  return rows.map((r) => ({ ...r, b: r.b * users }));
}

export function fmtBytes(b) {
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b < 10 ? b.toFixed(2) : b < 100 ? b.toFixed(1) : b.toFixed(0)} ${u[i]}`;
}

// ---------- positions ----------
export function sinusoidal(pos, d) {
  return Array.from({ length: d }, (_, i) => {
    const k = Math.floor(i / 2);
    const a = pos / Math.pow(10000, (2 * k) / d);
    return i % 2 === 0 ? Math.sin(a) : Math.cos(a);
  });
}

export const rotate = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];

export function ropeScore(q, k, m, n, theta) {
  return dot(rotate(q, m * theta), rotate(k, n * theta));
}

export function alibiSlopes(h) {
  return Array.from({ length: h }, (_, i) => Math.pow(2, (-8 * (i + 1)) / h));
}

// ---------- RoPE context extension ----------
export function invFreq(d, base = 10000) {
  return Array.from({ length: d / 2 }, (_, i) => Math.pow(base, (-2 * i) / d));
}

// Returns, per rotary pair, theta_new / theta_old (1 = untouched, 1/s = fully interpolated).
export function extensionScales({ d = 128, base = 10000, s = 8, Ltrain = 4096, alpha = 1, beta = 32 }) {
  const f = invFreq(d, base);
  const pi = f.map(() => 1 / s);
  const ntkBase = base * Math.pow(s, d / (d - 2));
  const ntk = invFreq(d, ntkBase).map((x, i) => x / f[i]);
  const yarn = f.map((th) => {
    const r = Ltrain / ((2 * Math.PI) / th);
    const g = r < alpha ? 0 : r > beta ? 1 : (r - alpha) / (beta - alpha);
    return (1 - g) / s + g;
  });
  const wavelength = f.map((th) => (2 * Math.PI) / th);
  return { pi, ntk, yarn, wavelength, ntkBase, yarnTemp: 0.1 * Math.log(s) + 1 };
}

// ---------- linear attention regrouping (course Section 4 numbers) ----------
export function regroupDemo({ q = 2, keys = [0.5, 1, 1.5], values = [10, 20, 30] } = {}) {
  const scores = keys.map((k) => q * k);
  const direct = scores.reduce((s, x, i) => s + x * values[i], 0);
  const state = keys.reduce((s, k, i) => s + k * values[i], 0);
  const w = softmax(scores);
  const soft = w.reduce((s, x, i) => s + x * values[i], 0);
  return { scores, direct, state, regrouped: q * state, weights: w, soft };
}

// ---------- additive vs delta-rule writes into a fixed state ----------
// State maps a 2-D key to a scalar value: read(k) = S . k. Keys are unit vectors.
export function keyPair(overlap) {
  return { A: [1, 0], B: [overlap, Math.sqrt(1 - overlap * overlap)] };
}

export function runWrites(writes, rule, beta = 1) {
  let S = [0, 0];
  const trace = [];
  for (const { key, v } of writes) {
    const before = dot(S, key);
    const delta = rule === "delta" ? beta * (v - before) : v;
    S = [S[0] + delta * key[0], S[1] + delta * key[1]];
    trace.push({ before, written: delta, S: [...S] });
  }
  return { S, trace, read: (k) => dot(S, k) };
}

// ---------- attention masks / sparsity patterns ----------
// Cell codes: 0 = not read, 1 = local/window or dense, 2 = sink/global, 3 = selected/random.
export function maskPattern(type, T, p = {}) {
  const M = Array.from({ length: T }, () => new Array(T).fill(0));
  const w = p.window ?? 4;
  const r = rng(p.seed ?? 7);
  const scores = randMatrix(T, T, p.seed ?? 11, 1);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j <= i; j++) {
      const local = i - j < w;
      switch (type) {
        case "full": M[i][j] = 1; break;
        case "window": M[i][j] = local ? 1 : 0; break;
        case "sinks": M[i][j] = local ? 1 : j < (p.sinks ?? 2) ? 2 : 0; break;
        case "strided": {
          const st = p.stride ?? 4;
          M[i][j] = Math.floor(i / st) === Math.floor(j / st) ? 1 : (i - j) % st === 0 ? 3 : 0;
          break;
        }
        case "bigbird": {
          const g = p.global ?? 1;
          M[i][j] = local ? 1 : j < g ? 2 : r() < (p.random ?? 0.12) ? 3 : 0;
          break;
        }
        default: break;
      }
    }
  }
  if (type === "topk") {
    const k = p.k ?? 4;
    for (let i = 0; i < T; i++) {
      const idx = [...Array(i + 1).keys()].sort((a, b) => scores[i][b] - scores[i][a]).slice(0, k);
      idx.forEach((j) => (M[i][j] = 3));
    }
  }
  if (type === "h2o") {
    // Greedy heavy-hitter eviction: keep the recent window plus the k tokens with the largest
    // accumulated attention; once evicted, a token never comes back.
    const k = p.k ?? 4, acc = new Array(T).fill(0), kept = new Set();
    for (let i = 0; i < T; i++) {
      kept.add(i);
      const z = [...kept].reduce((s, j) => s + Math.exp(scores[i][j]), 0);
      kept.forEach((j) => (acc[j] += Math.exp(scores[i][j]) / z));
      const old = [...kept].filter((j) => i - j >= w);
      if (old.length > k) old.sort((a, b) => acc[a] - acc[b]).slice(0, old.length - k).forEach((j) => kept.delete(j));
      kept.forEach((j) => (M[i][j] = i - j < w ? 1 : 3));
    }
  }
  if (type === "block" || type === "nsa") {
    const b = p.block ?? 4, k = p.k ?? 2;
    for (let i = 0; i < T; i++) {
      const cur = Math.floor(i / b);
      const past = [...Array(cur).keys()];
      const blockScore = (blk) => {
        const js = [...Array(b).keys()].map((x) => blk * b + x);
        return js.reduce((s, j) => s + scores[i][j], 0) / b;
      };
      const chosen = past.sort((x, y) => blockScore(y) - blockScore(x)).slice(0, type === "block" ? k - 1 : k);
      for (let j = 0; j <= i; j++) {
        const blk = Math.floor(j / b);
        if (type === "block") M[i][j] = blk === cur ? 1 : chosen.includes(blk) ? 3 : 0;
        else M[i][j] = i - j < w ? 1 : chosen.includes(blk) ? 3 : 0;
      }
    }
  }
  return M;
}

export function maskStats(M) {
  const T = M.length;
  const read = M.reduce((s, row) => s + row.filter((x) => x > 0).length, 0);
  return { read, dense: (T * (T + 1)) / 2 };
}
