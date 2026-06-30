"""Losses for MUSE Stage-1 v2 VAE.

Components:
  L_recon_m  Gaussian NLL with learned per-protein per-dim variance from
             the probabilistic decoder. Masked on missing modalities.
  L_KL_m     KL( q(h_m|x_m) || N(0, I) ) per modality, masked.
             Aggregated as Σ_m β_m · L_KL_m with β_m provided externally
             (so the runner can warm-up / sweep β per modality).
  L_struct   Same Kendall-weighted within-modality semi-hard triplet as v1.
             Imported from muse_stage1.losses to avoid duplication.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the v1 structure-triplet implementation (same algorithm).
from muse_stage1.losses import structure_triplet_loss


# ───────────────────────────────────────────────────────────────────────────
# Gaussian NLL reconstruction (probabilistic decoder)
# ───────────────────────────────────────────────────────────────────────────

def gaussian_nll_masked(x: torch.Tensor,
                        mu: torch.Tensor,
                        log_sigma2: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Per-protein Gaussian NLL averaged over feature dim, then over the
    masked subset of the batch.

      NLL_i,d = ½ · [ (x_i,d − μ_i,d)² / σ²_i,d  +  log σ²_i,d  +  log 2π ]
    The log 2π is a constant — included so the magnitude matches a proper
    Gaussian NLL (mostly useful for absolute-value diagnostics).

    Returns scalar.
    """
    if mask.dim() != 1:
        mask = mask.view(-1)
    inv_var = torch.exp(-log_sigma2)
    nll = 0.5 * ((x - mu).pow(2) * inv_var + log_sigma2 + math.log(2.0 * math.pi))  # (B, d_m)
    per_protein = nll.mean(dim=-1)                          # (B,) avg over d_m
    w = mask.float()
    return (per_protein * w).sum() / w.sum().clamp_min(1.0)


def vae_reconstruction_loss(model,
                            z: torch.Tensor,
                            inputs: Dict[str, torch.Tensor],
                            masks: Dict[str, torch.Tensor]
                            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sum over modalities of Gaussian NLL recon from the joint z."""
    per_mod: Dict[str, torch.Tensor] = {}
    losses = []
    for m in model.modality_names:
        mu_x, log_s2_x = model.decode(z, m)
        l = gaussian_nll_masked(inputs[m], mu_x, log_s2_x, masks[m])
        per_mod[m] = l
        losses.append(l)
    if not losses:
        return z.new_tensor(0.0), per_mod
    return torch.stack(losses).sum(), per_mod


# ───────────────────────────────────────────────────────────────────────────
# Per-modality KL to standard normal
# ───────────────────────────────────────────────────────────────────────────

def kl_to_standard_normal(mu: torch.Tensor,
                           log_sigma2: torch.Tensor,
                           mask: torch.Tensor,
                           free_bits: float = 0.0) -> torch.Tensor:
    """KL( q(h|x) || N(0, I) ) averaged over the masked subset.

    Closed form per protein per dimension:
        kl_i,d = ½ · [ μ_i,d²  +  σ²_i,d  −  log σ²_i,d  −  1 ]

    With `free_bits > 0`, each latent dimension gets up to `free_bits` nats
    "for free" — only KL above that threshold is penalised. This is the
    standard trick from Kingma et al. (2016) for preventing posterior
    collapse: the model can use a fixed budget of KL without paying for it.
    `free_bits=0.5` is a common default; `0.0` disables (plain VAE).
    """
    if mask.dim() != 1:
        mask = mask.view(-1)
    kl_per_dim = 0.5 * (mu.pow(2) + torch.exp(log_sigma2) - log_sigma2 - 1.0)  # (B, d)
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp_min(float(free_bits))
    kl = kl_per_dim.sum(dim=-1)                                                 # (B,)
    w = mask.float()
    return (kl * w).sum() / w.sum().clamp_min(1.0)


def vae_kl_loss(mu_dict: Dict[str, torch.Tensor],
                log_s2_dict: Dict[str, torch.Tensor],
                masks: Dict[str, torch.Tensor],
                beta_dict: Dict[str, float],
                modality_order,
                free_bits: float = 0.0,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Σ_m β_m · KL(q(h_m|x_m) || N(0,I)) — with optional free-bits per dim."""
    per_mod: Dict[str, torch.Tensor] = {}
    losses = []
    for m in modality_order:
        kl = kl_to_standard_normal(mu_dict[m], log_s2_dict[m], masks[m],
                                    free_bits=free_bits)
        per_mod[m] = kl
        b = float(beta_dict.get(m, 1.0))
        losses.append(b * kl)
    if not losses:
        return mu_dict[next(iter(mu_dict))].new_tensor(0.0), per_mod
    return torch.stack(losses).sum(), per_mod


# ───────────────────────────────────────────────────────────────────────────
# Total loss helper
# ───────────────────────────────────────────────────────────────────────────

def total_loss_vae(model,
                   z: torch.Tensor,
                   mu_dict: Dict[str, torch.Tensor],
                   log_s2_dict: Dict[str, torch.Tensor],
                   inputs: Dict[str, torch.Tensor],
                   masks: Dict[str, torch.Tensor],
                   pseudo_labels: Dict[str, torch.Tensor],
                   beta_dict: Dict[str, float],
                   lambda_recon: float = 1.0,
                   lambda_struct: float = 1.0,
                   margin: float = 0.3,
                   free_bits: float = 0.0) -> Dict[str, torch.Tensor]:
    """Compose L_total = λ_recon · L_recon + λ_struct · L_struct + Σ_m β_m · L_KL_m.

    Returns a dict of scalar tensors for logging plus per-modality breakouts.
    `free_bits` is forwarded to vae_kl_loss for posterior-collapse protection.
    """
    L_recon, per_recon = vae_reconstruction_loss(model, z, inputs, masks)
    L_kl, per_kl = vae_kl_loss(mu_dict, log_s2_dict, masks, beta_dict,
                                model.modality_names, free_bits=free_bits)
    L_struct = structure_triplet_loss(z, pseudo_labels, masks,
                                       model.log_sigma_struct,
                                       model.modality_names, margin=margin)
    L_total = lambda_recon * L_recon + lambda_struct * L_struct + L_kl
    return {
        "total":  L_total,
        "recon":  L_recon,
        "kl":     L_kl,
        "struct": L_struct,
        "per_modality_recon": per_recon,
        "per_modality_kl":    per_kl,
    }
