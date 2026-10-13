"""Start `world` CPU processes, one per virtual GPU, and collect what they measured."""
import json
import os
import socket
import sys
import tempfile
from dataclasses import replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .comm import Ring
from .engine import Engine, RunConfig, config_dict, digest, summary


# Everything is on one machine: bind gloo to loopback, which also stops each of
# the 32 processes warning that it could not resolve the hostname.
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, cfg, port, out):
    torch.set_num_threads(1)                   # 32 processes share the CPU's cores
    torch.manual_seed(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=cfg.world)
    try:
        eng = Engine(cfg, rank, Ring(rank, cfg.world))
        records = [eng.step(s) for s in range(cfg.steps)]
        losses = [r["loss"] for r in records]
        mine = summary(eng, losses, records)
        master = eng.full_master()
        everyone = [None] * cfg.world if rank == 0 else None
        dist.gather_object(mine, everyone, dst=0)
        if rank == 0:
            torch.save(master, out + ".pt")
            with open(out, "w") as f:
                json.dump(dict(config=config_dict(cfg), ranks=everyone,
                               master_sha=digest(master), master_path=out + ".pt"), f)
    finally:
        dist.destroy_process_group()


def _collectives_worker(rank, world, port, out, numel):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    try:
        ring = Ring(rank, world)
        x = torch.randn(numel, generator=torch.Generator().manual_seed(rank))
        ref = x.clone()
        dist.all_reduce(ref)                                # gloo's own implementation
        mine = ring.reduce_scatter(x, "reduce_scatter")
        full = ring.all_gather(mine, torch.empty_like(x), "all_gather")
        ar = ring.all_reduce(x, "all_reduce")
        me = dict(rank=rank,
                  owns_chunk_rank=torch.equal(mine, full.chunk(world)[rank]),
                  rs_then_ag_equals_ar=torch.equal(full, ar),
                  max_abs_err_vs_gloo=(full - ref).abs().max().item(),
                  bytes=dict(ring.bytes_sent), seconds=dict(ring.seconds),
                  input_bytes=x.numel() * x.element_size())
        everyone = [None] * world if rank == 0 else None
        dist.gather_object(me, everyone, dst=0)
        if rank == 0:
            with open(out, "w") as f:
                json.dump(everyone, f)
    finally:
        dist.destroy_process_group()


def check_collectives(world=32, numel=32 * 4096):
    """Hand-rolled ring vs gloo's all_reduce, on `world` processes."""
    fd, out = tempfile.mkstemp(suffix=".json", prefix=f"collectives_N{world}_")
    os.close(fd)
    mp.spawn(_collectives_worker, args=(world, _free_port(), out, numel), nprocs=world, join=True)
    with open(out) as f:
        return json.load(f)


def run(cfg: RunConfig = None, out=None, **kw):
    """Run one configuration and return rank 0's view of every rank's results."""
    cfg = replace(cfg or RunConfig(), **kw)
    if out is None:
        fd, out = tempfile.mkstemp(suffix=".json", prefix=f"{cfg.stage}_N{cfg.world}_")
        os.close(fd)
    mp.spawn(_worker, args=(cfg, _free_port(), out), nprocs=cfg.world, join=True)
    with open(out) as f:
        res = json.load(f)
    res["master"] = torch.load(res["master_path"])
    return res
