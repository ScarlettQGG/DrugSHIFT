"""MUSE Stage-1 v2 model — VAE-style probabilistic encoders/decoders + deterministic fusion."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Building blocks
# ───────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """Compact 2-layer MLP — always includes Dropout layer so state_dict
    indices are stable across dropout values (see two_stage_v3 architecture
    discussion)."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.0,
                 last_activation: Optional[str] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),               # always present (p=0 = no-op)
            nn.Linear(hidden, out_dim),
        )
        self.last_activation = last_activation

    def forward(self, x):
        h = self.net(x)
        if self.last_activation == "tanh":
            h = torch.tanh(h)
        return h


class ProbabilisticEncoder(nn.Module):
    """x_m → (μ_m, log σ²_m). Shared trunk MLP + two output heads."""
    def __init__(self, in_dim: int, hidden: int, latent_dim: int,
                 dropout: float = 0.0,
                 log_sigma_bias_init: float = -2.0):
        super().__init__()
        self.trunk = _MLP(in_dim, hidden, hidden, dropout=dropout)
        self.mu_head        = nn.Linear(hidden, latent_dim)
        self.log_sigma2_head = nn.Linear(hidden, latent_dim)
        # Init log σ² head to a small negative value so encoder starts mostly
        # deterministic; the KL warm-up lets it open up as training progresses.
        nn.init.zeros_(self.log_sigma2_head.weight)
        with torch.no_grad():
            self.log_sigma2_head.bias.fill_(log_sigma_bias_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.trunk(x))
        mu = self.mu_head(h)
        log_s2 = self.log_sigma2_head(h)
        # Clamp for numerical stability: σ² ∈ [exp(-6), exp(2)] ≈ [0.0025, 7.4]
        log_s2 = log_s2.clamp(-6.0, 2.0)
        return mu, log_s2


class ProbabilisticDecoder(nn.Module):
    """z → (μ_x_m, log σ²_x_m). Shared trunk + two output heads."""
    def __init__(self, joint_dim: int, hidden: int, out_dim: int,
                 dropout: float = 0.0,
                 log_sigma_bias_init: float = 0.0):
        super().__init__()
        self.trunk = _MLP(joint_dim, hidden, hidden, dropout=dropout)
        self.mu_head        = nn.Linear(hidden, out_dim)
        self.log_sigma2_head = nn.Linear(hidden, out_dim)
        # Init log σ² head to 0 (σ²=1) so reconstruction loss starts as plain MSE.
        nn.init.zeros_(self.log_sigma2_head.weight)
        with torch.no_grad():
            self.log_sigma2_head.bias.fill_(log_sigma_bias_init)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.trunk(z))
        mu = self.mu_head(h)
        log_s2 = self.log_sigma2_head(h)
        log_s2 = log_s2.clamp(-6.0, 6.0)
        return mu, log_s2


# ───────────────────────────────────────────────────────────────────────────
# Main model
# ───────────────────────────────────────────────────────────────────────────

class MUSEStage1VAE(nn.Module):
    """VAE-style multimodal embedding (Option II+).

    Encoders are probabilistic (μ_m, σ²_m); fusion is deterministic on
    (sampled or mean) h_m + masks; decoders are probabilistic (μ_x_m, σ²_x_m).
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

        # Probabilistic encoders
        self.encoders = nn.ModuleDict({
            m: ProbabilisticEncoder(d, hidden_dim, self.latent_dim, dropout=dropout)
            for m, d in modality_dims.items()
        })
        # Deterministic fusion: [h_1 | … | h_M | mask_1 | … | mask_M] -> z
        fusion_in = self.latent_dim * len(self.modality_names) + len(self.modality_names)
        self.fusion = _MLP(fusion_in, hidden_dim, self.joint_dim,
                           dropout=dropout, last_activation=None)
        # Probabilistic decoders
        self.decoders = nn.ModuleDict({
            m: ProbabilisticDecoder(self.joint_dim, hidden_dim, d, dropout=0.0)
            for m, d in modality_dims.items()
        })
        # Kendall homoscedastic σ for the struct triplet loss (one per modality).
        # Reconstruction noise is captured by the decoder's σ² head per-protein
        # (so no homoscedastic recon term needed here).
        n = len(self.modality_names)
        self.log_sigma_struct = nn.Parameter(torch.zeros(n))

    # ---------- encode ----------
    def encode(self, inputs: Dict[str, torch.Tensor],
               masks: Dict[str, torch.Tensor],
               sample: Optional[bool] = None
               ) -> Tuple[torch.Tensor,
                          Dict[str, torch.Tensor],
                          Dict[str, torch.Tensor],
                          Dict[str, torch.Tensor]]:
        """Encode + fuse.

        sample : None → reparameterise iff self.training, else use μ_m.
                 True  → always sample.
                 False → always use μ_m (deterministic inference).

        Returns
        -------
        z          : (B, joint_dim)
        h_dict     : {m: (B, latent_dim)} — mask-gated h_m used for fusion
        mu_dict    : {m: (B, latent_dim)} — encoder μ_m (pre-mask)
        log_s2_dict: {m: (B, latent_dim)} — encoder log σ²_m
        """
        do_sample = self.training if sample is None else bool(sample)
        h_blocks: List[torch.Tensor] = []
        mask_blocks: List[torch.Tensor] = []
        h_dict: Dict[str, torch.Tensor] = {}
        mu_dict: Dict[str, torch.Tensor] = {}
        log_s2_dict: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            x = inputs[m]
            mk = masks[m].float().unsqueeze(-1)
            mu, log_s2 = self.encoders[m](x)
            mu_dict[m] = mu
            log_s2_dict[m] = log_s2
            if do_sample:
                std = torch.exp(0.5 * log_s2)
                eps = torch.randn_like(std)
                h = mu + eps * std
            else:
                h = mu
            h = h * mk                                 # mask-gate
            h_dict[m] = h
            h_blocks.append(h)
            mask_blocks.append(mk)
        h_cat = torch.cat(h_blocks + mask_blocks, dim=-1)
        z = self.fusion(h_cat)
        return z, h_dict, mu_dict, log_s2_dict

    # ---------- decode ----------
    def decode(self, z: torch.Tensor, modality: str
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ_x_m, log σ²_x_m) from the probabilistic decoder."""
        return self.decoders[modality](z)

    def decode_mean(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """Convenience: returns just μ_x_m. Backward-compat with v1 callers
        expecting a single tensor (e.g. Stage 2's σ²_EPIC computation)."""
        mu, _ = self.decoders[modality](z)
        return mu

    # ---------- forward (= encode) ----------
    def forward(self, inputs, masks):
        return self.encode(inputs, masks)


def make_vae_model(modality_dims: Dict[str, int],
                   latent_dim_per_modality: int = 64,
                   joint_dim: int = 256,
                   hidden_dim: int = 256,
                   dropout: float = 0.0) -> MUSEStage1VAE:
    return MUSEStage1VAE(modality_dims=modality_dims,
                         latent_dim_per_modality=latent_dim_per_modality,
                         joint_dim=joint_dim,
                         hidden_dim=hidden_dim,
                         dropout=dropout)
