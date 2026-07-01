"""two_stage — self-contained perturbation-aware multimodal cell-mapping model.

Two stages in one flat package:

    Stage 1  ``Stage1``        the static multimodal reference map
    Stage 2  ``NeighborhoodAdapter``  per-perturbation remodelling of that map

The EPIC modality that feeds Stage 1 is produced by a separate SEC-MS
preprocessing step (``joint_embed.py`` at the repository root), which is not
part of this package. Train either stage or both via ``python -m
two_stage.train --stage {1,2,both}`` (default both).

Public surface:
    from two_stage import (
        Stage1, make_model, NeighborhoodAdapter, Stage1Cache,
        train_stage1, train_adapter, run_inference,
    )
"""
from .model import Stage1, make_model, NeighborhoodAdapter
from .stage1_bridge import Stage1Cache
from .train import train_stage1, train_adapter
from .inference import run_inference

__all__ = [
    "Stage1", "make_model", "NeighborhoodAdapter", "Stage1Cache",
    "train_stage1", "train_adapter", "run_inference",
]
