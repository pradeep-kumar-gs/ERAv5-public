"""One training process ("virtual GPU") running DP, ZeRO-1, ZeRO-2 or ZeRO-3.

What each stage keeps per weight (course Section 6):

    stage   bf16 weight   bf16 grad   fp32 master + Adam m + Adam v
    dp      full          full        full
    zero1   full          full        1/N shard
    zero2   full          1/N shard   1/N shard
    zero3   1/N shard     1/N shard   1/N shard

Every tensor that holds training state is registered in a Ledger by its real
storage size, so the memory numbers are read from tensors, not from a formula.
"""
import hashlib
import math
import queue
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch.autograd.graph import saved_tensors_hooks

from . import model as M
from .comm import Ring

BF16 = torch.bfloat16
STAGES = ("dp", "zero1", "zero2", "zero3")
HOST = ("host_master", "host_adam_m", "host_adam_v")


@dataclass
class RunConfig:
    stage: str = "dp"
    world: int = 32
    steps: int = 6
    global_batch: int = 32
    lr: float = 3e-3
    seed: int = 0
    offload: bool = False      # ZeRO-Offload: master + Adam state live in host memory
    precision: str = "bf16"    # "fp32" turns off bf16 to test DP against one big-batch process
    reduce_dtype: str = "same" # "fp32" sums gradients in fp32 on the wire (2x the gradient bytes)
    bucket_units: int = 1      # dp only: units per gradient bucket (1 keeps sums order-identical to ZeRO)
    overlap: bool = False      # dp only: send buckets during backward, on a comm thread
    pcie_gbps: float = 60.0    # course Section 8, used to price offload traffic
    vocab: int = 65
    d: int = 64
    n_layer: int = 4
    n_head: int = 4
    seq: int = 64
    data: str = "data/tinyshakespeare.txt"

    def dtype(self):
        return torch.float32 if self.precision == "fp32" else BF16

    def rdtype(self):
        return torch.float32 if self.reduce_dtype == "fp32" else self.dtype()

    def model(self):
        return M.ModelConfig(self.vocab, self.d, self.n_layer, self.n_head, self.seq, self.dtype())


class Ledger:
    """Live training-state bytes by category, with the peak it ever reached."""

    def __init__(self):
        self.live, self.peak, self.peak_by = {}, 0, {}

    def put(self, key, cat, t):
        self.live[key] = (cat, t.untyped_storage().nbytes())
        self._tick()

    def drop(self, key):
        self.live.pop(key, None)

    def by_cat(self, host=False):
        out = defaultdict(int)
        for cat, b in self.live.values():
            if (cat in HOST) == host:
                out[cat] += b
        return dict(out)

    def _tick(self):
        dev = self.by_cat()
        tot = sum(dev.values())
        if tot > self.peak:
            self.peak, self.peak_by = tot, dev


class Unit:
    def __init__(self, name, params, world, rank):
        self.name, self.params = name, params
        self.numel = sum(math.prod(s) for _, s in params)
        self.padded = -(-self.numel // world) * world   # ring chunks must be equal
        self.shard = self.padded // world
        self.lo, self.hi = rank * self.shard, (rank + 1) * self.shard


def adam_(w, m, v, g, t, lr, b1=0.9, b2=0.95, eps=1e-8):
    """Adam with one IEEE operation per line (no fused multiply-add), so a slice
    of the vector gets exactly the same result as the same slice updated in full."""
    m.mul_(b1).add_(g * (1 - b1))
    v.mul_(b2).add_(g * g * (1 - b2))
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    w.sub_(lr * mhat / (vhat.sqrt() + eps))


class Engine:
    def __init__(self, cfg: RunConfig, rank: int, ring: Ring):
        assert cfg.stage in STAGES
        assert cfg.global_batch % cfg.world == 0
        self.cfg, self.rank, self.N, self.ring = cfg, rank, cfg.world, ring
        self.mcfg = cfg.model()
        self.ledger = Ledger()
        self.units = [Unit(n, p, self.N, rank) for n, p in M.unit_specs(self.mcfg)]
        self.n_params = sum(u.numel for u in self.units)
        self.dt, self.rdt = cfg.dtype(), cfg.rdtype()
        self.z3 = cfg.stage == "zero3"
        self.sharded_grads = cfg.stage in ("zero2", "zero3")
        self.timeline = []
        self._build_state()
        self._load_data()

    # ---- state ------------------------------------------------------------
    def _build_state(self):
        gen = torch.Generator().manual_seed(self.cfg.seed)
        sharded_opt = self.cfg.stage != "dp"
        pre = "host_" if self.cfg.offload else ""
        for u in self.units:
            master = torch.zeros(u.padded)
            master[:u.numel] = M.init_unit(u.params, gen, self.mcfg.n_layer)
            if self.z3:
                u.shard_param = master[u.lo:u.hi].to(self.dt, copy=True)
                u.full = torch.empty(u.padded, dtype=self.dt).requires_grad_()
                u.full.untyped_storage().resize_(0)          # not materialised yet
                self.ledger.put(("param", u.name), "param_bf16", u.shard_param)
            else:
                u.full = master.to(self.dt, copy=True).requires_grad_()
                self.ledger.put(("param", u.name), "param_bf16", u.full)
            u.master = master[u.lo:u.hi].clone() if sharded_opt else master
            u.m, u.v = torch.zeros_like(u.master), torch.zeros_like(u.master)
            self.ledger.put(("master", u.name), pre + "master", u.master)
            self.ledger.put(("m", u.name), pre + "adam_m", u.m)
            self.ledger.put(("v", u.name), pre + "adam_v", u.v)
            if self.sharded_grads:
                u.grad_shard = torch.zeros(u.shard, dtype=self.rdt)
                self.ledger.put(("grad", u.name), "grad_bf16", u.grad_shard)
            else:
                u.full.grad = torch.zeros(u.padded, dtype=self.dt)
                self.ledger.put(("grad", u.name), "grad_bf16", u.full.grad)
            u.full.register_post_accumulate_grad_hook(lambda p, u=u: self._on_grad(u))

    def _load_data(self):
        text = open(self.cfg.data, encoding="utf-8").read()
        chars = sorted(set(text))
        assert len(chars) == self.mcfg.vocab
        lut = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([lut[c] for c in text], dtype=torch.long)

    def batch(self, step):
        """The global batch is the same for every world size; rank r takes its rows."""
        G, T = self.cfg.global_batch, self.mcfg.seq
        g = torch.Generator().manual_seed(10_000 * self.cfg.seed + step)
        ix = torch.randint(0, len(self.data) - T - 1, (G,), generator=g)
        x = torch.stack([self.data[i:i + T] for i in ix])
        y = torch.stack([self.data[i + 1:i + T + 1] for i in ix])
        b = G // self.N
        return x[self.rank * b:(self.rank + 1) * b], y[self.rank * b:(self.rank + 1) * b]

    # ---- ZeRO-3: materialise a unit's weights only while they are in use ---
    def gather(self, u, tag):
        u.full.untyped_storage().resize_(u.padded * u.full.element_size())
        self.ring.all_gather(u.shard_param, u.full.data, tag)
        self.ledger.put(("gathered", u.name), "gathered_bf16", u.full)
        self.param_ptrs.add(u.full.data_ptr())

    def release(self, u):
        self.param_ptrs.discard(u.full.data_ptr())   # the allocator may hand this address to an activation
        u.full.untyped_storage().resize_(0)
        self.ledger.drop(("gathered", u.name))

    # ---- gradient arrival (fires once per unit, during backward) ----------
    def _on_grad(self, u):
        self.timeline.append(("grad_ready", u.name, time.perf_counter()))
        if self.sharded_grads:
            self.ledger.put(("grad_full", u.name), "grad_full_transient", u.full.grad)
            u.grad_shard.copy_(self.ring.reduce_scatter(u.full.grad.to(self.rdt), "grad_reduce_scatter"))
            u.full.grad = None                                # the other N-1 slices are thrown away
            self.ledger.drop(("grad_full", u.name))
            if self.z3:
                self.release(u)
        elif self.cfg.overlap:
            self._ready.append(u)
            if len(self._ready) == self.cfg.bucket_units or u is self.units[0]:
                self._jobs.put(list(self._ready))
                self._ready.clear()

    # ---- dp gradient buckets ----------------------------------------------
    def _send_bucket(self, bucket):
        t0 = time.perf_counter()
        flat = torch.cat([u.full.grad for u in bucket]).to(self.rdt)
        red = self.ring.all_reduce(flat, "grad_all_reduce")
        for u, part in zip(bucket, red.split([u.padded for u in bucket])):
            u.reduced = part
        self.timeline.append(("comm", "+".join(u.name for u in bucket), t0, time.perf_counter()))

    def _comm_loop(self):
        while (bucket := self._jobs.get()) is not None:
            self._send_bucket(bucket)
            self._jobs.task_done()

    def buckets(self):
        """Backward order: head first. Same grouping the overlap thread uses."""
        k = self.cfg.bucket_units
        rev = self.units[::-1]
        return [rev[i:i + k] for i in range(0, len(rev), k)]

    # ---- one optimizer step -----------------------------------------------
    def step(self, step):
        t = {}
        self.ring.reset()
        self.timeline = []
        self.param_ptrs = set() if self.z3 else {u.full.data_ptr() for u in self.units}
        x, y = self.batch(step)
        if not self.sharded_grads:
            for u in self.units:
                u.full.grad.zero_()
        if self.cfg.overlap:
            self._jobs, self._ready = queue.Queue(), []
            comm = threading.Thread(target=self._comm_loop, daemon=True)
            comm.start()

        # forward ------------------------------------------------------------
        seen, act = set(), [0]

        def pack(tensor):
            st = tensor.untyped_storage()
            p = st.data_ptr()
            if p and p not in self.param_ptrs and p not in seen:
                seen.add(p)
                act[0] += st.nbytes()
            return tensor

        t0 = time.perf_counter()
        h = x
        with saved_tensors_hooks(pack, lambda tensor: tensor):
            for i, u in enumerate(self.units):
                if self.z3:
                    self.gather(u, "param_all_gather_fwd")
                h = M.unit_forward(u.name, M.views(u.full, u.params), h, self.mcfg,
                                   targets=y if u.name == "head" else None)
                if self.z3:
                    self.release(u)
                    if u.name != "head":
                        h.register_hook(lambda g, u=u: self.gather(u, "param_all_gather_bwd"))
        loss = h
        t["fwd"] = time.perf_counter() - t0
        self.activation_bytes = act[0]

        # backward -----------------------------------------------------------
        t0 = time.perf_counter()
        if self.z3:
            self.gather(self.units[-1], "param_all_gather_bwd")
        loss.backward()
        t["bwd"] = time.perf_counter() - t0
        t_bwd_end = time.perf_counter()

        # gradient sync for dp / zero1 ----------------------------------------
        t0 = time.perf_counter()
        if self.cfg.stage == "dp":
            if self.cfg.overlap:
                self._jobs.join()
                self._jobs.put(None)
                comm.join()
            else:
                for b in self.buckets():
                    self._send_bucket(b)
        t["sync_after_bwd"] = time.perf_counter() - t0

        # optimizer ------------------------------------------------------------
        t0 = time.perf_counter()
        self._update(step + 1)
        t["opt"] = time.perf_counter() - t0
        t["step"] = t["fwd"] + t["bwd"] + t["sync_after_bwd"] + t["opt"]
        t["adam"] = self.adam_seconds
        t["comm"] = sum(self.ring.seconds.values())
        t["compute"] = t["step"] - t["comm"]

        lv = loss.detach().float().reshape(1)
        dist.all_reduce(lv)                                   # metric only, not counted
        self.last = dict(loss=lv.item() / self.N, times=t, bwd_end=t_bwd_end,
                         bytes=dict(self.ring.bytes_sent), calls=dict(self.ring.calls),
                         comm_seconds=dict(self.ring.seconds))
        return self.last

    def _update(self, t):
        st, N, lr = self.cfg.stage, self.N, self.cfg.lr
        self.pcie_bytes, self.adam_seconds = 0, 0.0
        for u in self.units:
            if st == "dp":
                t0 = time.perf_counter()
                adam_(u.master, u.m, u.v, u.reduced.float() / N, t, lr)
                self.adam_seconds += time.perf_counter() - t0
                u.full.data.copy_(u.master.to(self.dt))
                del u.reduced
                continue
            if st == "zero1":
                g = self.ring.reduce_scatter(u.full.grad.to(self.rdt), "grad_reduce_scatter")
            else:
                g = u.grad_shard
            if self.cfg.offload:
                self.pcie_bytes += g.numel() * g.element_size()           # grad shard to host
            t0 = time.perf_counter()
            adam_(u.master, u.m, u.v, g.float() / N, t, lr)
            self.adam_seconds += time.perf_counter() - t0
            new = u.master.to(self.dt)
            if self.cfg.offload:
                self.pcie_bytes += new.numel() * new.element_size()       # bf16 shard back
            if self.z3:
                u.shard_param.copy_(new)
            else:
                self.ring.all_gather(new, u.full.data, "param_all_gather_update")

    # ---- results ------------------------------------------------------------
    def full_master(self):
        """Every rank's fp32 master copy, joined back into one vector (rank 0 only)."""
        mine = [u.master for u in self.units]
        if self.cfg.stage == "dp":
            return torch.cat([m[:u.numel] for m, u in zip(mine, self.units)]) if self.rank == 0 else None
        got = [None] * self.N if self.rank == 0 else None
        dist.gather_object(mine, got, dst=0)
        if self.rank != 0:
            return None
        return torch.cat([torch.cat([got[r][i] for r in range(self.N)])[:u.numel]
                          for i, u in enumerate(self.units)])


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def summary(eng: Engine, losses, step_records):
    dev = eng.ledger.by_cat()
    host = eng.ledger.by_cat(host=True)
    times = defaultdict(list)
    for r in step_records[1:]:           # drop the first step: allocator warm-up
        for k, v in r["times"].items():
            times[k].append(v)
    return dict(
        rank=eng.rank, n_params=eng.n_params,
        padded=sum(u.padded for u in eng.units),
        persistent=dev, persistent_total=sum(dev.values()),
        host=host, host_total=sum(host.values()),
        peak_total=eng.ledger.peak, peak_by=eng.ledger.peak_by,
        activation_bytes=eng.activation_bytes,
        bytes_sent=step_records[-1]["bytes"], calls=step_records[-1]["calls"],
        comm_seconds=step_records[-1]["comm_seconds"],
        pcie_bytes=getattr(eng, "pcie_bytes", 0),
        times={k: sum(v) / len(v) for k, v in times.items()},
        timeline=eng.timeline, bwd_end=step_records[-1]["bwd_end"],
        losses=losses,
    )


def config_dict(cfg):
    return asdict(cfg)
