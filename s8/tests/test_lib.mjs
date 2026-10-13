// node s8/tests/test_lib.mjs — checks widget maths against the course's worked numbers and paper formulas.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as L from "../site/lib.js";

let n = 0;
const t = (name, fn) => { fn(); n++; console.log("ok -", name); };
const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

t("softmax rows sum to 1 and masked entries get 0", () => {
  const p = L.softmax([1, 2, -Infinity]);
  close(p.reduce((a, b) => a + b, 0), 1);
  assert.equal(p[2], 0);
});

t("causal attention: no weight on the future, rows sum to 1", () => {
  const X = L.randMatrix(6, 4, 1), W = () => L.randMatrix(4, 4, 2);
  const s = L.attentionSteps(X, W(), L.randMatrix(4, 4, 3), L.randMatrix(4, 4, 4), true);
  s.P.forEach((row, i) => {
    close(row.reduce((a, b) => a + b, 0), 1);
    row.forEach((x, j) => j > i && assert.equal(x, 0));
  });
  const open = L.attentionSteps(X, W(), L.randMatrix(4, 4, 3), L.randMatrix(4, 4, 4), false);
  assert.ok(open.P[0][5] > 0, "unmasked first row reads the future");
});

t("course Section 10: 48L x 8 KV x 128 x 32768 bf16 = 6.44 GB, x8 users = 51.54 GB", () => {
  const one = L.kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, T: 32768, bytes: 2 });
  assert.equal(L.fmtBytes(one), "6.44 GB");
  assert.equal(L.fmtBytes(one * 8), "51.5 GB");
  close(one * 8 / 1e9, 51.54, 0.01);
});

t("scheme ordering: MHA > GQA > MQA; GQA-8 matches kvCacheBytes; pure linear is T-independent", () => {
  const a = Object.fromEntries(L.cacheBySchemes({ T: 32768 }).map((r) => [r.id, r.b]));
  const b = Object.fromEntries(L.cacheBySchemes({ T: 1 << 20 }).map((r) => [r.id, r.b]));
  assert.ok(a.mha > a.gqa && a.gqa > a.mqa);
  assert.equal(a.gqa, L.kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, T: 32768 }));
  assert.equal(a.linear, b.linear);
  assert.equal(L.cacheBySchemes({ T: 8192 }).find((r) => r.id === "swa").b, L.cacheBySchemes({ T: 1 << 20 }).find((r) => r.id === "swa").b);
  // MLA: 576 numbers/token/layer vs MHA 2*64*128 = 16384 -> 3.5% (DeepSeek-V2 reports 93.3% saving vs its 67B dense)
  close(a.mla / a.mha, 576 / 16384);
});

t("course Section 4: direct 140, regrouped 140, softmax ~25.8", () => {
  const r = L.regroupDemo();
  assert.equal(r.direct, 140);
  assert.equal(r.state, 70);
  assert.equal(r.regrouped, 140);
  close(r.soft, 25.75, 0.01);
});

t("course Sections 5-6: additive write returns 95, delta rule returns 55 (orthogonal keys)", () => {
  const { A, B } = L.keyPair(0);
  const seq = [{ key: A, v: 40 }, { key: B, v: 70 }, { key: A, v: 55 }];
  const add = L.runWrites(seq, "add"), del = L.runWrites(seq, "delta");
  close(add.read(A), 95); close(add.read(B), 70);
  close(del.read(A), 55); close(del.read(B), 70);
});

t("delta rule: latest write is always read back exactly, even with overlapping keys", () => {
  const { A, B } = L.keyPair(0.6);
  const del = L.runWrites([{ key: A, v: 40 }, { key: B, v: 70 }], "delta");
  close(del.read(B), 70, 1e-9);
  assert.ok(Math.abs(del.read(A) - 40) > 1, "overlap disturbs the earlier key (capacity limit)");
});

t("RoPE score depends only on relative distance", () => {
  const q = [0.8, 0.3], k = [0.5, -0.4];
  close(L.ropeScore(q, k, 8, 2, 0.3), L.ropeScore(q, k, 18, 12, 0.3), 1e-12);
  assert.notEqual(L.ropeScore(q, k, 8, 2, 0.3).toFixed(6), L.ropeScore(q, k, 8, 4, 0.3).toFixed(6));
});

t("ALiBi slopes for 8 heads are 1/2 ... 1/256", () => {
  const s = L.alibiSlopes(8);
  close(s[0], 0.5); close(s[7], 1 / 256);
});

t("extension: PI uniform 1/s; NTK keeps pair 0, interpolates last pair by ~1/s; YaRN ramps 1 -> 1/s", () => {
  const e = L.extensionScales({ d: 128, s: 8, Ltrain: 4096 });
  e.pi.forEach((x) => close(x, 1 / 8));
  close(e.ntk[0], 1);
  close(e.ntk.at(-1), 1 / 8, 0.01);
  close(e.yarn[0], 1);
  close(e.yarn.at(-1), 1 / 8);
  assert.ok(e.yarn.every((x, i) => i === 0 || x <= e.yarn[i - 1] + 1e-12), "monotone ramp");
  close(e.yarnTemp, 0.1 * Math.log(8) + 1);
});

t("mask patterns respect causality and save work", () => {
  for (const type of ["full", "window", "sinks", "strided", "bigbird", "topk", "block", "nsa", "h2o"]) {
    const M = L.maskPattern(type, 24, { window: 4, k: 4, block: 4 });
    M.forEach((row, i) => row.forEach((x, j) => j > i && assert.equal(x, 0, `${type} reads future`)));
    const s = L.maskStats(M);
    if (type === "full") assert.equal(s.read, s.dense);
    else assert.ok(s.read < s.dense, `${type} should read fewer pairs`);
  }
  const ev = L.maskPattern("h2o", 24, { window: 4, k: 3 });
  ev.forEach((row, i) => {
    assert.ok(row.filter((x) => x > 0).length <= 4 + 3, "budget = window + k");
    row.forEach((x, j) => { if (j <= i && !x) for (let r = i + 1; r < 24; r++) assert.equal(ev[r][j], 0, "evicted token came back"); });
  });
  const tk = L.maskPattern("topk", 24, { k: 4 });
  tk.forEach((row, i) => assert.equal(row.filter((x) => x > 0).length, Math.min(4, i + 1)));
});

t("data: chronological, required list complete, every card has buys/costs/pick", () => {
  const d = JSON.parse(readFileSync(new URL("../site/data/mechanisms.json", import.meta.url)));
  const m = d.mechanisms;
  const dates = m.map((x) => x.date);
  assert.deepEqual(dates, [...dates].sort());
  const required = ["standard", "learned-abs", "sinusoidal", "rope", "alibi", "mqa", "gqa", "sliding-window", "attention-sinks",
    "ntk-aware", "yarn", "linear-attention", "delta-rule", "gated-deltanet", "mla", "explicit-topk", "deepseek-v4-csa", "drope"];
  required.forEach((id) => assert.ok(m.find((x) => x.id === id && x.required), `missing required ${id}`));
  for (const x of m) {
    assert.ok(x.buys.length && x.costs.length && x.pick, x.id);
    assert.ok(x.source.url.startsWith("https://"), x.id);
    const era = d.eras.find((e) => e.id === x.era);
    assert.ok(era.from <= x.date && x.date <= era.to, `${x.id} outside era ${x.era}`);
    if (x.source.kind !== "arxiv") assert.ok(x.source.verified_via || x.source.kind === "blog", `${x.id} needs verified_via`);
  }
});

console.log(`\n${n} passed`);
