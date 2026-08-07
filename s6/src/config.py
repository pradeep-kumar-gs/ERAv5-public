"""Single place for every run-wide constant. mixture.py, packing.py, train.py,
replay.py and run_demo.py all read from here instead of hardcoding numbers,
so the "final run" is fully described by one file.
"""
from __future__ import annotations

SEED = 1337

BLOCK_SIZE = 64
BATCH_SIZE = 8

LANES = ["prose", "code", "agent"]
PROTECTED_LANE = "code"
PROTECTED_FLOOR_FRACTION = 0.5  # of the lane's stage weight (mirrors s5's floor policy)

OPUS_ACCEPT_THRESHOLD = 0.05  # cosine similarity vs. golden-proxy gradient direction

# Curriculum: (name, first_step, last_step inclusive, lane weights, eligible lanes)
STAGES = [
    dict(name="warmup", start=1, end=16,
         weights={"prose": 0.7, "code": 0.2, "agent": 0.1},
         eligible={"prose", "code"}),  # agent tool-call data deferred until mid stage
    dict(name="mid", start=17, end=32,
         weights={"prose": 0.4, "code": 0.35, "agent": 0.25},
         eligible={"prose", "code", "agent"}),
    dict(name="anneal", start=33, end=48,
         weights={"prose": 0.2, "code": 0.5, "agent": 0.3},
         eligible={"prose", "code", "agent"}),
]
TOTAL_STEPS = STAGES[-1]["end"]

CHECKPOINT_EVERY = 10
CRASH_CHECKPOINT_STEP = 30      # main lineage checkpoints here, then a crash is simulated
RESUME_STEPS = [31, 32, 33, 34]  # steps replayed after resume; must match the reference pass exactly

REPLAY_WINDOW = (10, 13)   # inclusive step range replayed from the consumption ledger
FORK_FROM_STEP = 20        # earlier checkpoint a fork lineage branches from
FORK_STEPS = [21, 22, 23]  # steps the fork lineage trains on its own

# Tiny model dims -- mechanism correctness is the point, not scale.
N_LAYER = 2
N_HEAD = 2
N_EMBD = 64
LR = 3e-3
