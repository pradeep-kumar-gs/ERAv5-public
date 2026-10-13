"""cd s12 && .venv/bin/python tests/test_zsim.py

Fast checks of the simulator against the course's numbers. The multi-process
checks use 4 ranks so the whole file runs in about a minute.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zsim import STAGES, check_collectives, run              # noqa: E402
from zsim.engine import adam_                                # noqa: E402
from zsim.theory import (bucket_timeline, bytes_per_weight,  # noqa: E402
                         comm_in_P, gib_per_gpu, offload_split)

passed = 0


def t(name):
    def deco(fn):
        global passed
        fn()
        passed += 1
        print("ok -", name)
    return deco


def close(a, b, eps=1e-9):
    assert abs(a - b) <= eps, f"{a} != {b}"


def main():
    @t("course Section 7 table: 447.0 / 153.7 / 104.8 / 55.9 GiB at 8 GPUs, ZeRO-3 7.0 at 64")
    def _():
        assert [round(gib_per_gpu(s, 8), 1) for s in STAGES] == [447.0, 153.7, 104.8, 55.9]
        assert round(gib_per_gpu("zero3", 64), 1) == 7.0
        assert round(gib_per_gpu("zero2", 32), 1) == 68.1

    @t("course Section 10: no hiding gives 9.50 s (H100) and 5.52 s (B200); 83% hidden at 2 layers")
    def _():
        close(bucket_timeline(7.10, 2.40, bucket=12)["step"], 9.50, 1e-9)
        close(bucket_timeline(3.12, 2.40, bucket=12)["step"], 5.52, 1e-9)
        close(bucket_timeline(7.10, 2.40, bucket=2)["overlap_share"], 5 / 6)
        h100 = {k: bucket_timeline(7.10, 2.40, bucket=k, alpha=0.05)["step"] for k in (1, 2, 3, 4)}
        b200 = {k: bucket_timeline(3.12, 2.40, bucket=k, alpha=0.05)["step"] for k in (1, 2, 3, 4)}
        assert min(h100, key=h100.get) == 1 and min(b200, key=b200.get) == 2

    @t("Adam on a slice equals the same slice of Adam on the whole vector, bit for bit")
    def _():
        g = torch.Generator().manual_seed(0)
        w, gr = torch.randn(1003, generator=g), torch.randn(1003, generator=g)
        full = [w.clone(), torch.zeros(1003), torch.zeros(1003)]
        part = [w[200:451].clone(), torch.zeros(251), torch.zeros(251)]
        for step in (1, 2, 3):
            adam_(*full, gr * step, step, 1e-3)
            adam_(*part, gr[200:451] * step, step, 1e-3)
        assert torch.equal(full[0][200:451], part[0])

    @t("ring collectives on 4 processes: rank r owns chunk r, RS+AG == all-reduce, matches gloo")
    def _():
        res = check_collectives(world=4, numel=4 * 1024)
        for r in res:
            assert r["owns_chunk_rank"] and r["rs_then_ag_equals_ar"]
            assert r["max_abs_err_vs_gloo"] < 1e-5
            close(r["bytes"]["reduce_scatter"], r["input_bytes"] * 3 / 4)
            close(r["bytes"]["all_reduce"], r["input_bytes"] * 2 * 3 / 4)

    results = {s: run(stage=s, world=4, steps=3) for s in STAGES}

    @t("measured bytes per weight on 4 ranks equal the course formula for every stage")
    def _():
        for s, r in results.items():
            for rk in r["ranks"]:
                close(rk["persistent_total"] / rk["padded"], bytes_per_weight(s, 4))

    @t("measured bytes sent per rank equal 2(N-1)/N P, or 3(N-1)/N P for ZeRO-3")
    def _():
        for s, r in results.items():
            for rk in r["ranks"]:
                P = 2 * rk["padded"]
                close(sum(rk["bytes_sent"].values()) / P, comm_in_P(s, 4))

    @t("DP, ZeRO-1, ZeRO-2 and ZeRO-3 finish with bit-identical weights and losses")
    def _():
        shas = {r["master_sha"] for r in results.values()}
        assert len(shas) == 1, shas
        losses = {tuple(r["ranks"][0]["losses"]) for r in results.values()}
        assert len(losses) == 1

    @t("ZeRO-2 + offload: device 2+2/N, host 12/N, PCIe 4/N bytes per weight per step")
    def _():
        r = run(stage="zero2", world=4, steps=3, offload=True)
        exp = offload_split(4)
        for rk in r["ranks"]:
            close(rk["persistent_total"] / rk["padded"], exp["device"])
            close(rk["host_total"] / rk["padded"], exp["host"])
            close(rk["pcie_bytes"] / rk["padded"], exp["pcie_per_step"])
        assert r["master_sha"] == results["zero2"]["master_sha"], "offload must not change the maths"

    print(f"\n{passed} passed")


if __name__ == "__main__":
    main()
