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
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
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

# ===================== Stage 2 losses =====================

def kendall_mse(pred: torch.Tensor, target: torch.Tensor, log_sigma2: torch.Tensor
                ) -> torch.Tensor:
    """Heteroscedastic regression: (1/σ²)·MSE + log σ²   (scalar)."""
    mse = (pred - target).pow(2).mean(dim=-1)
    return ((-log_sigma2).exp() * mse + log_sigma2).mean()


def loss_loo(delta_hat: torch.Tensor,
             delta_raw_proj: torch.Tensor,
             log_sigma2_pred: torch.Tensor) -> torch.Tensor:
    """L_LOO: neighbour-only prediction matches the centre's observed delta."""
    # We supervise δ̂ (the neighbour-only prediction) on δ_raw_proj (the
    # projected observation). The neighbour-only network never sees δ_raw_i in
    # its receptive field, so identity is structurally impossible.
    return kendall_mse(delta_hat, delta_raw_proj.detach(), log_sigma2_pred)


def loss_decoder_stable(decoder, z_treat: torch.Tensor, z_ref: torch.Tensor
                        ) -> torch.Tensor:
    """L_*_stable: decoder output should not drift between reference and treated z.
    Decoder is frozen (no_grad)."""
    with torch.no_grad():
        target = decoder(z_ref)
    pred = decoder(z_treat)
    return F.mse_loss(pred, target)


def loss_epic_recon(D_epic, z_treat: torch.Tensor, epic_treat: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    cosine: bool = False) -> torch.Tensor:
    """L_epic_recon: D_epic(z_treat) should reconstruct the observed EPIC_treat.

    `cosine=True` (default) compares DIRECTION: err = 1 − cos(D_epic(z_treat),
    EPIC_treat). This is the right anchor on a normalised co-embedding — the
    frozen decoder preserves EPIC direction at cos≈0.99 but attenuates magnitude,
    so a raw-MSE anchor is weak (a large z-drift barely changes the decoded
    magnitude) while a cosine anchor strongly pins the angular position. For the
    null, EPIC_treat≈EPIC_ctrl→D_epic(z_ref), so it drives the movement to ~0°.
    `mask` is per-protein presence (1 if EPIC_treat available for this protein)."""
    pred = D_epic(z_treat)
    if cosine:
        err = 1.0 - F.cosine_similarity(pred, epic_treat, dim=-1)
    else:
        err = (pred - epic_treat).pow(2).mean(dim=-1)
    if mask is not None:
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        return (err * m).sum() / denom
    return err.mean()


def loss_mahalanobis_prior(z_treat: torch.Tensor,
                            mu_emp: torch.Tensor,
                            sigma_inv: torch.Tensor) -> torch.Tensor:
    """L_prior — empirical-Gaussian prior on z.

    Penalises z_treat for moving outside the Stage-1 training distribution:
        d²_M(z) = (z − μ_emp)ᵀ Σ_emp⁻¹ (z − μ_emp)

    The expected value of d² under a true N(μ, Σ) is d_z (the dimensionality),
    so we subtract d_z and take a one-sided hinge: only penalise excess.
    This is biology-aware: real translocation can push z away from the mean,
    but ungoverned drift into low-density regions gets penalised.

    z_treat   : (B, d_z)
    mu_emp    : (d_z,)
    sigma_inv : (d_z, d_z)
    """
    d = z_treat.shape[-1]
    diff = z_treat - mu_emp.to(z_treat.device)
    # (B, d) @ (d, d) → (B, d)  then row-wise sum
    quad = (diff @ sigma_inv.to(z_treat.device)) * diff
    d_m2 = quad.sum(dim=-1)                          # (B,)
    excess = (d_m2 - float(d)).clamp_min(0.0)        # one-sided hinge
    return excess.mean()


def loss_image_confidence(decoder, z_treat: torch.Tensor, z_ref: torch.Tensor,
                          tol: float = 0.1) -> torch.Tensor:
    """Translocation-aware image constraint.

    Allows any change in WHICH compartment a protein localizes to (real biology),
    but penalises any DROP in confidence — i.e. predictions that go from
    "definitely nuclear" → "I have no idea" (uniform softmax). Implemented as a
    one-sided hinge on entropy increase beyond the reference entropy + a small
    tolerance.

    This is the right constraint for translocation: a protein that moves from
    cytoplasm → nucleus produces a confident *new* HPA prediction, not a
    high-entropy "could be anywhere" prediction. Wild moves to nonsense z-regions
    typically produce the latter.

    decoder : frozen Stage-1 D_image — outputs the per-class HPA logits/scores.
    """
    def _entropy(logits):
        # If decoder outputs are already probabilities (post-sigmoid/softmax),
        # treating them as logits via softmax just rescales — still valid as
        # a smooth, monotone "diffuse vs concentrated" measure.
        p = F.softmax(logits, dim=-1)
        # Numerical safety
        return -(p * (p.clamp_min(1e-9)).log()).sum(dim=-1)

    with torch.no_grad():
        H_ref = _entropy(decoder(z_ref))
    H_treat = _entropy(decoder(z_treat))
    # one-sided: only penalize if entropy *increased* beyond tolerance
    excess = (H_treat - H_ref - tol).clamp_min(0.0)
    return excess.mean()


# ───────────────────────────────────────────────────────────────────────────
# Weight container + total
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class LossWeights:
    LOO:   float = 1.0
    seq:   float = 1.0          # sequence truly doesn't change — strict
    image: float = 0.05         # SOFT only — translocation is real biology
    epic:  float = 1.0
    # Optional translocation-aware image loss (penalises only entropy increases,
    # not localization shifts). Off by default.
    image_confidence: float = 0.0
    # Empirical-Gaussian prior on z (Mahalanobis distance from training
    # distribution mean+covariance, hinge on excess beyond d_z). Only fires
    # when Stage 1 is VAE (cache.z_emp_mu and cache.z_emp_sigma_inv exist).
    # Default 0.0 — opt-in via --w_prior.
    prior: float = 0.0
    # L2 penalty on the LEARNED residual projection magnitude ||delta_residual||²
    # (sum over dims, so it is NOT diluted by d_z). The frozen Stage-1 decoders
    # have a large null space — z_treat can drift far in directions the decoders
    # cannot see, so L_epic/L_seq cannot constrain that drift. The residual MLP is
    # the free knob that produces that unconstrained drift; penalising it keeps the
    # remapped delta close to the faithful frozen baseline. Default 0.0 (off).
    residual: float = 0.0
    # Factorized-magnitude losses (only when adapter.factorized): mag = match the
    # learned scalar magnitude to the observed tangential magnitude; mag_l1 = L1 sparsity.
    mag: float = 0.0
    mag_l1: float = 0.0


def compose_loss(adapter, cache, forward_out: Dict[str, torch.Tensor],
                 epic_treat_i: torch.Tensor,
                 epic_treat_mask_i: Optional[torch.Tensor],
                 idx: torch.Tensor,
                 weights: LossWeights,
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compose the multi-loss for a batch.

    forward_out         : output of NeighborhoodAdapter.forward_with_neighbour_delta
    epic_treat_i        : observed EPIC_treat for centre proteins (B, d_epic)
    epic_treat_mask_i   : presence mask for EPIC_treat (B,)
    idx                 : centre indices (B,)
    weights             : LossWeights
    """
    device = idx.device
    z_treat   = forward_out["z_treat"]
    z_ref     = cache.z[idx].to(device)
    delta_hat = forward_out["delta_hat"]
    log_s2    = forward_out["log_sigma2_pred"]

    # 1) L_LOO
    L_LOO = loss_loo(delta_hat, forward_out["delta_raw_proj"], log_s2)

    # 2) L_seq_stable
    if cache.D_seq is not None:
        L_seq = loss_decoder_stable(cache.D_seq, z_treat, z_ref)
    else:
        L_seq = torch.zeros((), device=device)

    # 3) L_image_stable (SOFT — wild-move regularizer, NOT a translocation block)
    if cache.D_image is not None:
        L_img = loss_decoder_stable(cache.D_image, z_treat, z_ref)
    else:
        L_img = torch.zeros((), device=device)

    # 3b) L_image_confidence (optional, translocation-aware)
    if cache.D_image is not None and weights.image_confidence > 0:
        L_img_conf = loss_image_confidence(cache.D_image, z_treat, z_ref)
    else:
        L_img_conf = torch.zeros((), device=device)

    # 4) L_epic_recon
    L_epic = loss_epic_recon(cache.D_epic, z_treat, epic_treat_i,
                             mask=epic_treat_mask_i,
                             cosine=getattr(adapter, "spherical", False))

    # 5) L_prior (Mahalanobis on empirical Gaussian — only if VAE + opt-in)
    if (weights.prior > 0
            and getattr(cache, "z_emp_mu", None) is not None
            and getattr(cache, "z_emp_sigma_inv", None) is not None):
        L_prior = loss_mahalanobis_prior(z_treat,
                                          cache.z_emp_mu,
                                          cache.z_emp_sigma_inv)
    else:
        L_prior = torch.zeros((), device=device)

    # 6) L_residual — penalise free drift of the learned residual projection.
    if weights.residual > 0 and "delta_residual" in forward_out:
        L_residual = forward_out["delta_residual"].pow(2).sum(dim=-1).mean()
    else:
        L_residual = torch.zeros((), device=device)

    # 7) L_mag — factorized magnitude: match the observed tangential magnitude
    #    (so m reflects real change) + L1 sparsity (so noise/stable proteins → 0).
    if (forward_out.get("learned_magnitude") is not None
            and forward_out.get("m_raw_target") is not None):
        m = forward_out["learned_magnitude"]
        mt = forward_out["m_raw_target"].detach()
        L_mag = (m - mt).pow(2).mean()
        L_mag_l1 = m.abs().mean()
    else:
        L_mag = torch.zeros((), device=device)
        L_mag_l1 = torch.zeros((), device=device)

    L_total = (weights.LOO    * L_LOO
             + weights.seq    * L_seq
             + weights.image  * L_img
             + weights.image_confidence * L_img_conf
             + weights.epic   * L_epic
             + weights.prior  * L_prior
             + weights.residual * L_residual
             + weights.mag      * L_mag
             + weights.mag_l1   * L_mag_l1)

    return L_total, {
        "total":  float(L_total.item()),
        "LOO":    float(L_LOO.item()),
        "seq":    float(L_seq.item()),
        "image":  float(L_img.item()),
        "image_conf": float(L_img_conf.item()),
        "epic":   float(L_epic.item()),
        "prior":  float(L_prior.item()),
        "residual": float(L_residual.item()),
        "mag":      float(L_mag.item()),
        "sigma2_pred_mean": float(forward_out["sigma2_pred"].mean().item()),
        "sigma2_raw_mean":  float(forward_out["sigma2_raw"].mean().item()),
    }
