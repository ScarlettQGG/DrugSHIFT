"""MUSE Stage-1 v2 — VAE-style joint multimodal embedding.

Option II+ architecture (see Stage 2 v3 README for design discussion):

    Probabilistic encoders per modality:
        E_m : x_m → μ_m(x_m), log σ²_m(x_m)
        h_m ~ N(μ_m, σ²_m · I)              during training
        h_m  = μ_m                          at inference (deterministic readout)

    Deterministic fusion:
        z = Fuse([h_1, …, h_M, mask_1, …, mask_M])

    Probabilistic decoders per modality:
        D_m : z → μ_x_m(z), log σ²_x_m(z)   per-protein per-dim Gaussian NLL

Losses:
    L_recon_m = Σ_i mask_m,i · ½·[(x_m,i − μ_x_m,i)² / σ²_x_m,i + log σ²_x_m,i]
    L_KL_m    = Σ_i mask_m,i · ½·[‖μ_m,i‖² + σ²_m,i − log σ²_m,i − 1].sum(-1)
    L_struct  = same Kendall-weighted within-modality semi-hard triplet as v1
    L_total   = Σ_m λ_recon_m·L_recon_m + Σ_m β_m·L_KL_m + λ_struct·L_struct

Post-training the runner additionally saves:
    z_emp_mu.tsv          empirical mean of z across all trained proteins
    z_emp_sigma.tsv       empirical covariance (diag + off-diag) of z
    per_modality_sigma/<m>.tsv per-protein σ²_m from the encoder (for Stage 2)
    decoder_sigma/<m>.tsv  per-protein σ²_x_m(μ_z) from the decoder (for Stage 2)
    vae_config.json        records that this is a VAE model with β-values used
"""
from .model import MUSEStage1VAE, make_vae_model

__all__ = ["MUSEStage1VAE", "make_vae_model"]
