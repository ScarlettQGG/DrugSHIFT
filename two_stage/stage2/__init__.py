"""Stage 2 — neighborhood-aware delta adapter.

Reads a frozen Stage-1 model + manifest and learns a small adapter
that maps the raw EPIC delta (treat - ctrl in input space) into z-space
using cluster-aware kNN aggregation, leave-one-out neighbour prediction,
and Bayesian combination with Stage-1's per-protein EPIC σ². All Stage-1
artefacts are recomputed in-memory at each Stage-2 startup — no cache
file is persisted to disk.

Public surface:
    from two_stage.stage2 import Stage1Cache, NeighborhoodAdapter
    from two_stage.stage2 import train_adapter, run_inference
"""
from .stage1_cache import Stage1Cache
from .architecture import NeighborhoodAdapter
from .training import train_adapter
from .inference import run_inference

__all__ = [
    "Stage1Cache",
    "NeighborhoodAdapter",
    "train_adapter",
    "run_inference",
]
