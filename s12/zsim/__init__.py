"""zsim: data parallelism and ZeRO stages 1-3 on CPU processes standing in for GPUs."""
from .engine import STAGES, RunConfig
from .run import check_collectives, run

__all__ = ["STAGES", "RunConfig", "check_collectives", "run"]
