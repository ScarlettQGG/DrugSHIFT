"""two_stage — self-contained perturbation-aware multimodal cell-mapping model.

Two stages, one package:

    stage1   MUSE-style mask-aware multimodal autoencoder → the static
             reference cell map (``static_latent.tsv``).
    stage2   Neighbourhood-aware, coherence-weighted delta adapter that
             learns, per perturbation, how the Stage-1 map remodels.

A co-elution PPI encoder (``two_stage.joint_embed``) produces the EPIC
modality that feeds Stage 1.

Stage 2 imports the frozen Stage 1 model directly from this package
(``two_stage.stage1``) — no external dependency on a separate Stage-1
package is required.

Public surface:
    from two_stage.stage2 import (
        Stage1Cache, NeighborhoodAdapter, train_adapter, run_inference,
    )
"""

__all__ = ["stage1", "stage2", "joint_embed"]
