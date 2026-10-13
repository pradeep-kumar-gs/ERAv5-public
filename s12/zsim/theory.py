"""The course's closed-form numbers, kept apart from anything measured.

The notebook prints these next to the measurements from the 32 processes; the
two are never mixed.
"""
GIB = 2 ** 30
CARD_GIB = 80e9 / GIB       # an 80 GB card is 74.5 GiB (course Section 2)

# bytes per weight: (replicated on every rank, sharded 1/N ways)
SPLIT = {
    "dp": (16, 0),          # 2 bf16 weight + 2 bf16 grad + 4 fp32 master + 8 Adam m,v
    "zero1": (4, 12),       # weight and grad stay; master + Adam are sharded
    "zero2": (2, 14),       # weight stays; grad joins the shard
    "zero3": (0, 16),       # everything is sharded
}


def bytes_per_weight(stage, n):
    rep, shard = SPLIT[stage]
    return rep + shard / n


def comm_in_P(stage, n):
    """Bytes each rank sends per step, in units of P (one bf16 copy of the weights).

    A ring collective sends (n-1)/n of its input per phase. DP and ZeRO-1/2 run
    two phases (reduce-scatter + all-gather); ZeRO-3 runs three (all-gather in
    forward, all-gather in backward, reduce-scatter of gradients).
    """
    phases = 3 if stage == "zero3" else 2
    return phases * (n - 1) / n


def gib_per_gpu(stage, n, n_params=30e9):
    return n_params * bytes_per_weight(stage, n) / GIB


def offload_split(n):
    """ZeRO-2 with the master copy and Adam state in host memory, per weight."""
    return dict(device=2 + 2 / n, host=12 / n, pcie_per_step=4 / n)


def bucket_timeline(compute, comm_total, layers=12, bucket=1, alpha=0.0, bwd_share=2 / 3):
    """Course Section 10 as an alpha-beta model, in seconds.

    Backward runs layer by layer from the end. A bucket of `bucket` layers is
    sent as soon as its last gradient exists; transfers queue on one link. Each
    transfer pays a fixed start cost `alpha` plus its share of `comm_total`.
    """
    fwd = compute * (1 - bwd_share)
    per_layer = compute * bwd_share / layers
    link_free, spans = 0.0, []
    for first in range(0, layers, bucket):
        n = min(bucket, layers - first)
        ready = fwd + per_layer * (first + n)
        start = max(ready, link_free)
        link_free = start + alpha + comm_total * n / layers
        spans.append((start, link_free))
    bwd_end = compute
    busy = sum(e - s for s, e in spans)
    hidden = sum(max(0.0, min(e, bwd_end) - s) for s, e in spans)
    return dict(step=max(bwd_end, link_free), overlap_share=hidden / busy,
                spans=spans, bwd_end=bwd_end, fwd_end=fwd)
