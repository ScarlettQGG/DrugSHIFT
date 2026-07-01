"""two_stage — self-contained perturbation-aware multimodal cell-mapping model.

Two stages in one flat package:

    Stage 1  ``MUSEStage1``        the static multimodal reference map
    Stage 2  ``NeighborhoodAdapter``  per-perturbation remodelling of that map

A co-elution PPI encoder (``two_stage.joint_embed``) produces the EPIC modality
that feeds Stage 1. Train either stage or both via ``python -m two_stage.train
--stage {1,2,both}`` (default both).

Public surface:
    from two_stage import (
        MUSEStage1, make_model, NeighborhoodAdapter, Stage1Cache,
        train_stage1, train_adapter, run_inference,
    )
"""
from .model import MUSEStage1, make_model, NeighborhoodAdapter
from .cache import Stage1Cache
from .train import train_stage1, train_adapter
from .inference import run_inference

__all__ = [
    "MUSEStage1", "make_model", "NeighborhoodAdapter", "Stage1Cache",
    "train_stage1", "train_adapter", "run_inference",
]
