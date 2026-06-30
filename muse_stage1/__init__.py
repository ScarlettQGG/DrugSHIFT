"""muse_stage1 — MUSE-style Stage-1 multimodal anchor for the two-stage co-embedding pipeline.

Key differences vs. the existing Stage 1:
  * Joint latent z is a LEARNED fusion of per-modality latents (concat + MLP),
    not an unweighted mean. Missing modalities are zero-filled with mask bits.
  * Each modality decoder reconstructs its input FROM the joint z (not from
    the per-modality latent). This is the "preserve every modality"
    information-bottleneck constraint.
  * The triplet term is the MUSE within-modality structure-preserving form:
    per-modality pseudo-labels (Leiden / KMeans) define same-cluster
    positives; semi-hard negative mining within the batch. The joint z must
    respect each modality's own cluster geometry.
  * Per-modality recon + per-modality structure terms are balanced by Kendall
    homoscedastic uncertainty (learned log-sigma per term).

Exports:
    MUSEStage1                          (model.py)
    compute_all_pseudo_labels           (pseudo_labels.py)
    train_muse_stage1                   (train.py)
"""
from .dropout import apply_dropout, random_modality_dropout
from .losses import masked_recon_loss, structure_triplet_loss
from .model import MUSEStage1, make_model
from .pseudo_labels import compute_all_pseudo_labels
from .train import load_modality_matrices, train_muse_stage1
