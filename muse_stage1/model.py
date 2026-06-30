"""MUSE-style Stage-1 model.

Architecture:
    inputs_m  -> E_m  -> h_m  ──concat[h | mask]─→ Fuse ─→ z (joint)
                          │
                          └─────────── (h_m kept for diagnostics) ─→
    z         -> D_m  -> x_hat_m       (per-modality reconstruction from JOINT z)

Notes:
  * Missing modalities: input is zero-filled (placeholder "000") and the
    corresponding mask bit is 0; both the per-modality latent and its block in
    the concatenation are zeroed via the mask.
  * Per-modality decoders take the JOINT z (not h_m). This is the key MUSE-
    style information-bottleneck mechanism — z is forced to retain every
    modality's information because z must reconstruct each one.
  * Kendall homoscedastic uncertainty parameters log_sigma_recon[m] and
    log_sigma_struct[m] balance the per-modality loss terms automatically.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    """Compact 2-layer MLP with optional dropout, used for encoders/decoders/fusion."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0,
                 last_activation: Optional[str] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )
        self.last_activation = last_activation

    def forward(self, x):
        h = self.net(x)
        if self.last_activation == "tanh":
            h = torch.tanh(h)
        return h


class MUSEStage1(nn.Module):
    """MUSE-style joint embedding for >=2 modalities with missing-data masking.

    Parameters
    ----------
    modality_dims : dict {modality_name: input_dim}
        Input feature dimensionality per modality.
    latent_dim_per_modality : int
        Dim of per-modality latents h_m. Default 64.
    joint_dim : int
        Dim of the joint embedding z. Default 256.
    hidden_dim : int
        Hidden width inside encoders/decoders/fusion. Default 256.
    dropout : float
        Dropout on encoder MLPs. Default 0.0.
    """

    def __init__(self,
                 modality_dims: Dict[str, int],
                 latent_dim_per_modality: int = 64,
                 joint_dim: int = 256,
                 hidden_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.modality_names: List[str] = list(modality_dims.keys())
        self.modality_dims = dict(modality_dims)
        self.latent_dim = int(latent_dim_per_modality)
        self.joint_dim = int(joint_dim)

        # Per-modality encoders: input -> h_m
        self.encoders = nn.ModuleDict({
            m: _MLP(d, hidden_dim, self.latent_dim, dropout=dropout)
            for m, d in modality_dims.items()
        })
        # Fusion: [h_1 | ... | h_M | mask_1 | ... | mask_M] -> z
        fusion_in = self.latent_dim * len(self.modality_names) + len(self.modality_names)
        self.fusion = _MLP(fusion_in, hidden_dim, self.joint_dim,
                           dropout=dropout, last_activation=None)
        # Per-modality decoders: z -> x_hat_m
        self.decoders = nn.ModuleDict({
            m: _MLP(self.joint_dim, hidden_dim, d, dropout=0.0)
            for m, d in modality_dims.items()
        })
        # Kendall homoscedastic uncertainty (one per modality, per loss term).
        # Initialised at 0 -> exp(-0) = 1 weight; will be learned during training.
        n = len(self.modality_names)
        self.log_sigma_recon = nn.Parameter(torch.zeros(n))
        self.log_sigma_struct = nn.Parameter(torch.zeros(n))

    def encode(self, inputs: Dict[str, torch.Tensor],
               masks: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode and fuse.

        inputs : dict {m: (B, d_m)}. For missing entries the tensor must be
                 present (zero-filled is fine).
        masks  : dict {m: (B,) float/bool}, 1 if present, 0 if missing.

        Returns
        -------
        z      : (B, joint_dim) — the fused joint embedding.
        h_dict : dict {m: (B, latent_dim)} — per-modality latents (mask-gated).
        """
        h_blocks: List[torch.Tensor] = []
        mask_blocks: List[torch.Tensor] = []
        h_dict: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            x = inputs[m]
            mk = masks[m].float().unsqueeze(-1)        # (B,1)
            h = self.encoders[m](x) * mk                # mask-gated
            h_dict[m] = h
            h_blocks.append(h)
            mask_blocks.append(mk)
        h_cat = torch.cat(h_blocks + mask_blocks, dim=-1)  # (B, M*d + M)
        z = self.fusion(h_cat)
        return z, h_dict

    def decode(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """Decode joint z back to one modality's input space."""
        return self.decoders[modality](z)

    def forward(self, inputs, masks):
        return self.encode(inputs, masks)


def make_model(modality_dims: Dict[str, int],
               latent_dim_per_modality: int = 64,
               joint_dim: int = 256,
               hidden_dim: int = 256,
               dropout: float = 0.0) -> MUSEStage1:
    return MUSEStage1(modality_dims=modality_dims,
                      latent_dim_per_modality=latent_dim_per_modality,
                      joint_dim=joint_dim,
                      hidden_dim=hidden_dim,
                      dropout=dropout)
