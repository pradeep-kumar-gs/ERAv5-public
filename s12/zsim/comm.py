"""Hand-rolled ring collectives over point-to-point sends, with a byte ledger.

Each rank only ever talks to its two neighbours: it sends to rank+1 and receives
from rank-1. Nothing here calls dist.all_reduce, so every byte that crosses a
process boundary is counted by this file.
"""
import time
from collections import defaultdict

import torch
import torch.distributed as dist


class Ring:
    """Ring reduce-scatter, all-gather and all-reduce among `world` processes.

    Chunk c of a reduce-scatter always finishes on rank c, and its partial sum
    always travels the same path (c+1, c+2, ..., c). The order of additions for
    every element is therefore fixed, whichever stage calls it. That is what lets
    data parallelism and the three ZeRO stages agree bit for bit.
    """

    def __init__(self, rank, world):
        self.rank, self.world = rank, world
        self.right, self.left = (rank + 1) % world, (rank - 1) % world
        self.bytes_sent = defaultdict(int)
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def _swap(self, send, recv):
        a = dist.isend(send, self.right)
        b = dist.irecv(recv, self.left)
        a.wait()
        b.wait()

    def reduce_scatter(self, flat, tag):
        """Sum `flat` over all ranks and return only chunk `rank` of the sum."""
        n, r = self.world, self.rank
        self.calls[tag] += 1
        if n == 1:
            return flat.clone()
        t0 = time.perf_counter()
        ch = list(flat.chunk(n))
        buf = torch.empty_like(ch[0])
        for s in range(n - 1):
            si, ri = (r - s - 1) % n, (r - s - 2) % n
            self._swap(ch[si], buf)
            self.bytes_sent[tag] += ch[si].numel() * ch[si].element_size()
            # add in fp32, store in the wire dtype: each hop rounds once
            ch[ri] = (ch[ri].float() + buf.float()).to(flat.dtype)
        self.seconds[tag] += time.perf_counter() - t0
        return ch[r]

    def all_gather(self, shard, out, tag):
        """Every rank contributes chunk `rank`; every rank ends with all of `out`."""
        n, r = self.world, self.rank
        self.calls[tag] += 1
        ch = list(out.chunk(n))
        ch[r].copy_(shard)
        if n == 1:
            return out
        t0 = time.perf_counter()
        for s in range(n - 1):
            si, ri = (r - s) % n, (r - s - 1) % n
            self._swap(ch[si], ch[ri])
            self.bytes_sent[tag] += ch[si].numel() * ch[si].element_size()
        self.seconds[tag] += time.perf_counter() - t0
        return out

    def all_reduce(self, flat, tag):
        """All-reduce is a reduce-scatter followed by an all-gather, nothing more."""
        mine = self.reduce_scatter(flat, tag)
        return self.all_gather(mine, torch.empty_like(flat), tag)

    def reset(self):
        self.bytes_sent.clear()
        self.seconds.clear()
        self.calls.clear()
