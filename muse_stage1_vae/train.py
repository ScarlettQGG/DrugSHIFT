"""MUSE Stage-1 v2 training loop (VAE).

Reuses v1's data-loading + pseudo-label + modality-dropout machinery; only
the model + loss are new. Adds β-warmup for the KL term.

Output files (saved to outdir):
    static_latent.tsv          joint z per protein (L2-normed), deterministic
    static_latent_raw.tsv      joint z per protein (pre-norm)
    static_model.pth           torch state_dict
    static_loss.tsv            per-epoch loss curves (total/recon/kl/struct/dropfrac)
    per_modality/<m>.tsv       per-protein μ_m (the deterministic readout of h_m)
    per_modality_sigma/<m>.tsv per-protein σ²_m from the probabilistic encoder
    decoder_sigma/<m>.tsv      per-protein σ²_x_m(μ_z) from the probabilistic decoder
    z_emp_mu.tsv               empirical mean μ_emp of z, for the Stage-2 prior
    z_emp_sigma.tsv            empirical covariance Σ_emp of z (full matrix)
    pseudo_labels.json         per-modality cluster pseudo-labels (same as v1)
    vae_config.json            β values, KL-warmup spec, architectural choices
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse v1 utilities
from muse_stage1.dropout import apply_dropout, random_modality_dropout
from muse_stage1.train import (
    load_modality_matrices, build_universe, assemble_tensors,
    build_pseudo_label_tensor, MUSEDataset, collate, _write_tsv,
)
from muse_stage1.pseudo_labels import compute_all_pseudo_labels

from .model import make_vae_model
from .losses import total_loss_vae


# ───────────────────────────────────────────────────────────────────────────
# β-warmup helper
# ───────────────────────────────────────────────────────────────────────────

def _beta_schedule(beta_target: Dict[str, float],
                   epoch: int,
                   warmup_epochs: int,
                   schedule: str = "linear",
                   cyclical_period: int = 50,
                   ) -> Dict[str, float]:
    """β scheduling.

    schedule="linear":     0 → β_target over the first `warmup_epochs`; constant after.
    schedule="cyclical":   sawtooth — within each period of `cyclical_period`,
                           β ramps 0 → β_target over the first half then holds
                           β_target the second half. Resets at the start of each
                           period. This is Fu et al. (2019) cyclical annealing,
                           used to escape posterior collapse by giving the
                           model recurring "pure recon" warm-starts.
    """
    sched = (schedule or "linear").lower()
    if sched == "cyclical" and cyclical_period > 0:
        ep_mod = epoch % cyclical_period
        half = max(1, cyclical_period // 2)
        f = min(1.0, ep_mod / float(half))
    else:
        if warmup_epochs <= 0:
            f = 1.0
        else:
            f = min(1.0, (epoch + 1) / float(warmup_epochs))
    return {m: float(b * f) for m, b in beta_target.items()}


# ───────────────────────────────────────────────────────────────────────────
# Main training entry point
# ───────────────────────────────────────────────────────────────────────────

def train_muse_stage1_vae(
    manifest_path: str,
    outdir: str,
    *,
    latent_dim_per_modality: int = 64,
    joint_dim: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    n_epochs: int = 300,
    batch_size: int = 256,
    learn_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_recon: float = 1.0,
    lambda_struct: float = 1.0,
    margin: float = 0.3,
    # VAE-specific
    beta: float = 0.1,                                    # uniform default β
    beta_per_modality: Optional[Dict[str, float]] = None, # overrides `beta`
    beta_warmup_epochs: int = 30,
    beta_schedule: str = "linear",                        # "linear" | "cyclical"
    cyclical_period: int = 50,                            # only for "cyclical"
    free_bits: float = 0.0,                               # per-dim KL allowance
    # Modality dropout (same as v1)
    p_drop: float = 0.3,
    dropout_min_keep: int = 1,
    # Pseudo-labels (same as v1)
    pseudo_method: str = "leiden",
    pseudo_kmeans_k: int = 50,
    pseudo_knn_k: int = 15,
    pseudo_resolution: float = 1.0,
    seed: int = 0,
    device: Optional[str] = None,
):
    """End-to-end VAE Stage 1 training (Option II+ architecture)."""
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)

    # 1) Load + assemble (same as v1)
    print("[vae] loading modality matrices...")
    mods = load_modality_matrices(manifest_path)
    if not mods:
        raise RuntimeError(f"No untreated modality matrices loaded from {manifest_path}")
    universe = build_universe(mods)
    print(f"[vae] universe size: {len(universe)}  modalities: {list(mods)}")
    Xs, Ms = assemble_tensors(universe, mods)
    modality_dims = {m: X.shape[1] for m, X in Xs.items()}

    # 2) Pseudo-labels (same as v1)
    print("[vae] computing pseudo-labels...")
    pl_cache = os.path.join(outdir, "pseudo_labels.json")
    pseudo_labels_dict = compute_all_pseudo_labels(
        mods, method=pseudo_method,
        knn_k=pseudo_knn_k, leiden_resolution=pseudo_resolution,
        kmeans_k=pseudo_kmeans_k, seed=seed, cache_path=pl_cache,
    )
    label_arrays = build_pseudo_label_tensor(universe, pseudo_labels_dict)

    # 3) Dataset/loader
    ds = MUSEDataset(Xs, Ms, label_arrays)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True, num_workers=0)

    # 4) Model + optimizer
    model = make_vae_model(modality_dims,
                           latent_dim_per_modality=latent_dim_per_modality,
                           joint_dim=joint_dim, hidden_dim=hidden_dim,
                           dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate,
                              weight_decay=weight_decay)
    print(f"[vae] model: joint_dim={joint_dim}, per-modality latent="
          f"{latent_dim_per_modality}, params={sum(p.numel() for p in model.parameters()):,}")

    # 5) Resolve β per modality
    beta_target: Dict[str, float] = {m: float(beta) for m in modality_dims}
    if beta_per_modality:
        for m, b in beta_per_modality.items():
            if m in beta_target:
                beta_target[m] = float(b)
    print(f"[vae] β target per modality: {beta_target}")
    print(f"[vae] schedule={beta_schedule}  warmup={beta_warmup_epochs}  "
          f"cyclical_period={cyclical_period}  free_bits={free_bits}")

    # 6) Training loop
    log_path = os.path.join(outdir, "static_loss.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\ttotal\trecon\tkl\tstruct\tbeta_scale\tdrop_frac\n")
    t0 = time.time()

    for ep in range(n_epochs):
        model.train()
        beta_now = _beta_schedule(beta_target, ep, beta_warmup_epochs,
                                   schedule=beta_schedule,
                                   cyclical_period=cyclical_period)
        beta_scale = max(beta_now.values()) / max(max(beta_target.values()), 1e-9)
        losses_t, losses_r, losses_k, losses_s, drop_fracs = [], [], [], [], []
        for inputs, masks, labels in loader:
            inputs_orig   = {m: v.to(device) for m, v in inputs.items()}
            masks_present = {m: v.to(device) for m, v in masks.items()}
            labels_dev    = {m: v.to(device) for m, v in labels.items()}

            # Modality dropout (visible-only at encoder input; recon target = original)
            if p_drop > 0.0:
                keep = random_modality_dropout(masks_present,
                                                p_drop=p_drop,
                                                min_keep=dropout_min_keep)
                inputs_vis, masks_vis = apply_dropout(inputs_orig, masks_present, keep)
                dropped = sum((1.0 - keep[m].mean()).item() for m in keep) / max(1, len(keep))
                drop_fracs.append(dropped)
            else:
                inputs_vis, masks_vis = inputs_orig, masks_present
                drop_fracs.append(0.0)

            # Encode from visible inputs (reparameterised because model.train() is on)
            z, h_dict, mu_dict, log_s2_dict = model.encode(inputs_vis, masks_vis)

            # Compose loss against original targets
            lossd = total_loss_vae(model, z, mu_dict, log_s2_dict,
                                    inputs_orig, masks_present, labels_dev,
                                    beta_dict=beta_now,
                                    lambda_recon=lambda_recon,
                                    lambda_struct=lambda_struct,
                                    margin=margin,
                                    free_bits=free_bits)

            optim.zero_grad()
            lossd["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            losses_t.append(lossd["total"].item())
            losses_r.append(lossd["recon"].item())
            losses_k.append(lossd["kl"].item())
            losses_s.append(lossd["struct"].item())

        with open(log_path, "a") as f:
            f.write(f"{ep}\t{np.mean(losses_t):.6f}\t{np.mean(losses_r):.6f}"
                    f"\t{np.mean(losses_k):.6f}\t{np.mean(losses_s):.6f}"
                    f"\t{beta_scale:.3f}\t{np.mean(drop_fracs):.4f}\n")
        if ep == 0 or (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            print(f"[vae] ep {ep:>4}  total={np.mean(losses_t):.4f}  "
                  f"recon={np.mean(losses_r):.4f}  kl={np.mean(losses_k):.4f}  "
                  f"struct={np.mean(losses_s):.4f}  β·scale={beta_scale:.2f}  "
                  f"dropped={np.mean(drop_fracs):.3f}")

    print(f"[vae] training done in {(time.time()-t0)/60:.1f} min")

    # 7) Deterministic inference: compute z, per-modality μ_m, σ²_m, decoder σ²_x_m(μ_z)
    print("[vae] computing per-protein z, σ²_m, decoder σ²_x_m ...")
    model.eval()
    N = len(universe)
    Z   = np.zeros((N, joint_dim), dtype=np.float32)
    Mu_per   = {m: np.zeros((N, latent_dim_per_modality), dtype=np.float32) for m in modality_dims}
    S2_enc   = {m: np.zeros((N, latent_dim_per_modality), dtype=np.float32) for m in modality_dims}
    S2_dec   = {m: np.zeros((N, modality_dims[m]),       dtype=np.float32) for m in modality_dims}

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            inputs = {m: torch.from_numpy(Xs[m][start:end]).to(device) for m in modality_dims}
            masks  = {m: torch.from_numpy(Ms[m][start:end]).to(device) for m in modality_dims}
            z, h, mu, log_s2 = model.encode(inputs, masks, sample=False)
            Z[start:end] = z.cpu().numpy()
            for m in modality_dims:
                Mu_per[m][start:end] = mu[m].cpu().numpy()
                S2_enc[m][start:end] = torch.exp(log_s2[m]).cpu().numpy()
                mu_x, log_s2_x = model.decode(z, m)
                S2_dec[m][start:end] = torch.exp(log_s2_x).cpu().numpy()

    # 8) Save
    norms = np.linalg.norm(Z, axis=1, keepdims=True); norms = np.maximum(norms, 1e-9)
    Zn = Z / norms

    _write_tsv(os.path.join(outdir, "static_latent.tsv"),     universe, Zn)
    _write_tsv(os.path.join(outdir, "static_latent_raw.tsv"), universe, Z)
    for m, A in Mu_per.items():
        _write_tsv(os.path.join(outdir, "per_modality", f"{m}.tsv"), universe, A)
    for m, A in S2_enc.items():
        _write_tsv(os.path.join(outdir, "per_modality_sigma", f"{m}.tsv"), universe, A)
    for m, A in S2_dec.items():
        _write_tsv(os.path.join(outdir, "decoder_sigma", f"{m}.tsv"), universe, A)

    # Empirical Gaussian over z (full covariance)
    mu_emp    = Z.mean(axis=0)
    Sigma_emp = np.cov(Z.T)
    _write_tsv(os.path.join(outdir, "z_emp_mu.tsv"),    ["EMP_MU"],    mu_emp[None, :])
    with open(os.path.join(outdir, "z_emp_sigma.tsv"), "w") as f:
        f.write("row\t" + "\t".join(f"d{i}" for i in range(joint_dim)) + "\n")
        for i in range(joint_dim):
            f.write(f"r{i}\t" + "\t".join(f"{v:.6f}" for v in Sigma_emp[i]) + "\n")

    torch.save(model.state_dict(), os.path.join(outdir, "static_model.pth"))

    # VAE config — Stage 2 reads this to detect VAE Stage 1 and pull β values
    cfg = {
        "model_kind":        "MUSEStage1VAE",
        "modality_dims":     modality_dims,
        "latent_dim_per_modality": latent_dim_per_modality,
        "joint_dim":         joint_dim,
        "hidden_dim":        hidden_dim,
        "dropout":           dropout,
        "n_epochs":          n_epochs,
        "beta_target":       beta_target,
        "beta_warmup_epochs": beta_warmup_epochs,
        "beta_schedule":     beta_schedule,
        "cyclical_period":   cyclical_period,
        "free_bits":         free_bits,
        "lambda_recon":      lambda_recon,
        "lambda_struct":     lambda_struct,
        "margin":            margin,
        "p_drop":            p_drop,
        "seed":              seed,
    }
    with open(os.path.join(outdir, "vae_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[vae] outputs in {outdir}")
    print("  static_latent.tsv            (joint z, L2-normed)")
    print("  per_modality/<m>.tsv         (μ_m — deterministic h_m readout)")
    print("  per_modality_sigma/<m>.tsv   (σ²_m — encoder uncertainty per protein)")
    print("  decoder_sigma/<m>.tsv        (σ²_x_m — decoder uncertainty per protein)")
    print("  z_emp_mu.tsv / z_emp_sigma.tsv (empirical Gaussian prior on z)")
    print("  vae_config.json              (β values + arch params)")
    print("  static_model.pth             (full state_dict)")
    return model, universe, Zn
