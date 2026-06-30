"""Losses for MUSE-style Stage 1: masked recon-from-z + semi-hard structure triplets.

* `masked_recon_loss`: per-modality cosine reconstruction from the joint z,
  masked on missing modalities, Kendall-weighted.
* `semi_hard_triplet`: pairwise semi-hard negative mining inside the batch,
  conditioned on pseudo-labels (positives = same cluster, negatives = different
  cluster). Returns a scalar loss; uses cosine distance.
* `structure_triplet_loss`: applies `semi_hard_triplet` for each modality's
  pseudo-labels and Kendall-weights the per-modality terms.

Kendall homoscedastic uncertainty weighting (Kendall & Gal 2017):
    L_balanced = sum_m [ exp(-log_sigma_m) * L_m + 0.5 * log_sigma_m ]
The model learns log_sigma_m to balance terms by difficulty (large σ -> down-
weight that term -> equalises gradient contribution across modalities).
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction from joint z (per modality, masked, Kendall-weighted)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_recon(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-sample 1 - cos(x_hat, x). Returns (B,) tensor."""
    return 1.0 - F.cosine_similarity(x_hat, x, dim=-1)


def masked_recon_loss(
    z: torch.Tensor,
    decoders: nn.ModuleDict,
    inputs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    log_sigma_recon: torch.Tensor,
    modality_order: list,
) -> torch.Tensor:
    """Sum over modalities of Kendall-weighted masked cosine reconstruction.

    z              : (B, joint_dim)
    decoders[m]    : Module mapping z -> input_m space
    inputs[m]      : (B, d_m) — zero-filled where missing
    masks[m]       : (B,) — 1 if present, 0 if missing
    log_sigma_recon: (M,) trainable parameter aligned with modality_order
    """
    losses = []
    for i, m in enumerate(modality_order):
        mk = masks[m].bool()
        if mk.sum() < 1:
            continue
        x_hat = decoders[m](z[mk])
        x = inputs[m][mk]
        r = _cosine_recon(x_hat, x).mean()
        # Kendall: exp(-log_sigma) * L + 0.5 * log_sigma  (regression form)
        w = torch.exp(-log_sigma_recon[i])
        losses.append(w * r + 0.5 * log_sigma_recon[i])
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Semi-hard triplet (FaceNet semi-hard mining), in-batch, cosine distance
# ─────────────────────────────────────────────────────────────────────────────

def semi_hard_triplet(
    z: torch.Tensor,                # (N, d) — only valid anchors (already filtered)
    labels: torch.Tensor,           # (N,) int cluster ids
    margin: float = 0.3,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """In-batch semi-hard triplet on cosine distance.

    For each anchor i with at least one positive (same label, different sample)
    and at least one negative (different label):
      d_pos[i] = max_j {d(i,j) : label_j == label_i, j != i}     (hardest positive)
      d_neg[i] = min_j {d(i,j) : d(i,j) > d_pos[i], label_j != label_i}
                 — semi-hard negative; fallback to argmin negative if none exists.
    Returns mean of max(0, d_pos - d_neg + margin) over valid anchors, or None
    if no anchor has the required positives+negatives in this batch.
    """
    N = z.shape[0]
    if N < 3:
        return None
    Z = F.normalize(z, dim=-1)
    D = 1.0 - Z @ Z.T                           # cosine distance (N, N), in [0, 2]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    pos_mask = same & ~diag
    neg_mask = ~same & ~diag

    has_pos = pos_mask.any(dim=-1)
    has_neg = neg_mask.any(dim=-1)
    valid = has_pos & has_neg
    if valid.sum() == 0:
        return None

    # Hardest positive per anchor (max over positives)
    D_pos = D.masked_fill(~pos_mask, float("-inf"))
    d_pos, _ = D_pos.max(dim=-1)
    # Semi-hard negative: smallest neg distance strictly greater than d_pos
    D_neg = D.masked_fill(~neg_mask, float("inf"))
    sh_mask = D_neg > d_pos.unsqueeze(-1)
    D_sh = D_neg.masked_fill(~sh_mask, float("inf"))
    d_neg_sh, _ = D_sh.min(dim=-1)
    # Fallback: when no semi-hard exists, use hardest (smallest) negative
    d_neg_h, _ = D_neg.min(dim=-1)
    d_neg = torch.where(torch.isfinite(d_neg_sh), d_neg_sh, d_neg_h)

    # Apply only on valid anchors
    loss_vec = F.relu(d_pos[valid] - d_neg[valid] + margin)
    return loss_vec.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality structure triplet loss (within-modality, joint z)
# ─────────────────────────────────────────────────────────────────────────────

def structure_triplet_loss(
    z: torch.Tensor,                           # (B, joint_dim)
    pseudo_labels: Dict[str, torch.Tensor],    # {m: (B,) int, -1 if missing}
    masks: Dict[str, torch.Tensor],            # {m: (B,) bool/float}
    log_sigma_struct: torch.Tensor,            # (M,) Kendall params
    modality_order: list,
    margin: float = 0.3,
) -> torch.Tensor:
    """Kendall-weighted sum over modalities of within-modality semi-hard triplets.

    For each modality m, take the batch subset where mask_m=1 AND label_m != -1,
    apply semi-hard triplet in joint z-space using m's pseudo-labels.
    """
    losses = []
    for i, m in enumerate(modality_order):
        mk = masks[m].bool() & (pseudo_labels[m] >= 0)
        if mk.sum() < 3:
            continue
        zi = z[mk]
        lab = pseudo_labels[m][mk]
        # Need ≥2 distinct labels and ≥2 samples per at least one label
        unique, counts = torch.unique(lab, return_counts=True)
        if unique.numel() < 2 or counts.max() < 2:
            continue
        l = semi_hard_triplet(zi, lab, margin=margin)
        if l is None:
            continue
        w = torch.exp(-log_sigma_struct[i])
        losses.append(w * l + 0.5 * log_sigma_struct[i])
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Total loss helper
# ─────────────────────────────────────────────────────────────────────────────

def total_loss(
    model,
    z: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    pseudo_labels: Dict[str, torch.Tensor],
    lambda_recon: float = 1.0,
    lambda_struct: float = 1.0,
    margin: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Compose total loss = λ_recon * recon + λ_struct * structure.
    Returns a dict of scalar tensors for logging."""
    recon = masked_recon_loss(z, model.decoders, inputs, masks,
                              model.log_sigma_recon, model.modality_names)
    struct = structure_triplet_loss(z, pseudo_labels, masks,
                                    model.log_sigma_struct,
                                    model.modality_names, margin=margin)
    total = lambda_recon * recon + lambda_struct * struct
    return {"total": total, "recon": recon, "struct": struct}
