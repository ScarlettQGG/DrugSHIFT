"""Stage 2 v3 training loop.

Full-batch training (everything fits — ~6k proteins, k=20 neighbours, d_z=256).
One adapter per condition, trained independently.

Inputs the trainer expects:
    cache_path                   Pickled Stage1Cache (from stage1_cache.build_cache)
    manifest_path                Same manifest as Stage 1; this trainer reads
                                 EPIC entries under the target condition label
                                 (e.g. condition='cisplatin').
    condition                    'cisplatin', 'vorinostat', ...
    cond_names                   list of all condition labels for the adapter's
                                 embedding table (you can train multiple
                                 conditions sequentially against the same
                                 embedding table for inference compatibility).

Outputs (saved under `outdir`):
    adapter.pt                   trained NeighborhoodAdapter state_dict
    config.json                  hyperparameters + cache path
    loss.tsv                     per-epoch loss curves
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Optional, Tuple
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .stage1_cache import Stage1Cache, load_epic_per_condition, _align
from .architecture_v3 import NeighborhoodAdapter
from .losses_v3 import LossWeights, compose_loss


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def load_aligned_epic(manifest_path: str, epic_name: str, universe,
                      condition: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (EPIC_ctrl, EPIC_treat, mask_treat) aligned to `universe`."""
    streams = load_epic_per_condition(manifest_path, epic_name)
    if "control" not in streams:
        raise RuntimeError(f"No 'control' EPIC stream in {manifest_path}")
    if condition not in streams:
        raise RuntimeError(f"No EPIC stream for condition={condition!r} in {manifest_path}. "
                           f"Available: {sorted(streams.keys())}")
    p_ctrl, X_ctrl = streams["control"]
    p_treat, X_treat = streams[condition]
    if X_ctrl.shape[1] != X_treat.shape[1]:
        raise RuntimeError(f"EPIC ctrl and treat have different d "
                           f"({X_ctrl.shape[1]} vs {X_treat.shape[1]})")
    Xc, mc = _align(universe, p_ctrl, X_ctrl)
    Xt, mt = _align(universe, p_treat, X_treat)
    # Centre: ctrl is the reference; if either is missing, mask the protein
    # for L_epic_recon but keep it in the batch (it can still get a δ̂ from
    # neighbours).
    mask_treat = (mt > 0.5).astype(np.float32)
    return (torch.from_numpy(Xc), torch.from_numpy(Xt), torch.from_numpy(mask_treat))


def gather_neighbour_delta(delta_raw_all: torch.Tensor,
                           knn_idx: torch.Tensor,
                           idx: torch.Tensor) -> torch.Tensor:
    """Return (B, k, d_epic) of neighbours' δ_raw."""
    n = knn_idx[idx]                     # (B, k)
    return delta_raw_all[n]              # advanced index


# ───────────────────────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────────────────────

def train_adapter(
    stage1_outdir: str,
    manifest_path: str,
    outdir: str,
    *,
    condition: str = "cisplatin",
    cond_names = ("cisplatin", "vorinostat"),
    n_epochs: int = 300,
    learn_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    weights: Optional[LossWeights] = None,
    d_e: int = 128, d_ctx: int = 128, d_attn: int = 128,
    d_clust: int = 16, d_cond: int = 8,
    hidden: int = 256,
    dropout: float = 0.1,
    sigma2_raw_floor: float = 0.05,
    sigma2_raw_scale: float = 1.0,
    sigma2_pred_init: Optional[float] = None,
    sigma2_epic_path: Optional[str] = None,
    coherence_gate: bool = False,
    coherence_gate_gamma: float = 1.0,
    spherical: bool = False,
    factorized: bool = False,
    drift_remove: bool = False,
    unified: bool = False,
    epic_name: str = "epic",
    k: int = 20,
    leiden_resolution: float = 1.0,
    cluster_compat_temp: float = 1.0,
    seed: int = 0,
    device: Optional[str] = None,
    seq_modality_hint: Optional[str] = None,
    image_modality_hint: Optional[str] = None,
):
    """End-to-end training for one condition's adapter.

    Builds Stage1Cache in-memory from `stage1_outdir` + `manifest_path` —
    no persisted cache file. Same call rebuilds on every invocation
    (cluster_id/kNN are deterministic given the same seed).
    """
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    if weights is None:
        weights = LossWeights()

    # ---- 1) Build cache in-memory from Stage 1 outputs ----
    print(f"[train] building cache in-memory from {stage1_outdir}")
    cache = Stage1Cache.from_stage1_dir(
        stage1_outdir, manifest_path,
        epic_name=epic_name, k=k,
        leiden_resolution=leiden_resolution,
        cluster_compat_temp=cluster_compat_temp,
        device=device, seed=seed,
        seq_name=seq_modality_hint,
        image_name=image_modality_hint,
        sigma2_epic_path=sigma2_epic_path,
    )
    print(f"[train] cache: N={cache.N}  d_z={cache.d_z}  K={cache.K}  k={cache.k_neighbours}")
    print(f"[train] decoder availability: seq={cache.D_seq is not None}  "
          f"image={cache.D_image is not None}  epic={cache.D_epic is not None}")

    # Move cache tensors to device
    cache.z = cache.z.to(device)
    cache.cluster_id = cache.cluster_id.to(device)
    cache.knn_idx = cache.knn_idx.to(device)
    cache.knn_w   = cache.knn_w.to(device)
    cache.sigma2_epic = cache.sigma2_epic.to(device)
    cache.h_per_modality = {m: H.to(device) for m, H in cache.h_per_modality.items()}

    # ---- 2) Load EPIC ctrl + treat aligned to cache.proteins ----
    print(f"[train] loading EPIC for condition={condition!r}")
    EPIC_ctrl, EPIC_treat, mask_treat = load_aligned_epic(
        manifest_path, cache.epic_name, cache.proteins, condition)
    EPIC_ctrl  = EPIC_ctrl.to(device)
    EPIC_treat = EPIC_treat.to(device)
    mask_treat = mask_treat.to(device)
    delta_raw_all = (EPIC_treat - EPIC_ctrl)                              # (N, d_epic)
    n_with_treat = int(mask_treat.sum().item())
    print(f"[train] EPIC: {n_with_treat}/{cache.N} proteins have treated profile")

    # ---- 3) Build adapter ----
    adapter = NeighborhoodAdapter(
        cache=cache, cond_names=list(cond_names),
        d_e=d_e, d_ctx=d_ctx, d_attn=d_attn,
        d_clust=d_clust, d_cond=d_cond, hidden=hidden, dropout=dropout,
        sigma2_raw_floor=sigma2_raw_floor, sigma2_raw_scale=sigma2_raw_scale,
        sigma2_pred_init=sigma2_pred_init,
        coherence_gate=coherence_gate, coherence_gate_gamma=coherence_gate_gamma,
        spherical=spherical, factorized=factorized, drift_remove=drift_remove, unified=unified,
    ).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[train] adapter params: {n_params:,}")

    cond_id_scalar = adapter.cond_to_id[condition]
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, adapter.parameters()),
                            lr=learn_rate, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    log_path = os.path.join(outdir, "loss.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\ttotal\tLOO\tseq\timage\timage_conf\tepic\tprior\ts2_pred\ts2_raw\tlr\n")

    # ---- 4) Full-batch training loop ----
    all_idx = torch.arange(cache.N, device=device)
    cond_ids = torch.full((cache.N,), cond_id_scalar, dtype=torch.long, device=device)

    print(f"[train] full-batch over N={cache.N} for {n_epochs} epochs (lr={learn_rate})")
    t0 = time.time()
    for ep in range(n_epochs):
        adapter.train()

        # Centre features
        delta_raw_i = delta_raw_all                                      # (N, d_epic)
        delta_raw_neigh = delta_raw_all[cache.knn_idx]                   # (N, k, d_epic)

        fo = adapter.forward_with_neighbour_delta(
            idx=all_idx,
            epic_ctrl_i=EPIC_ctrl,
            epic_treat_i=EPIC_treat,
            delta_raw_i=delta_raw_i,
            delta_raw_neigh=delta_raw_neigh,
            cond_id=cond_ids,
        )

        L, parts = compose_loss(
            adapter=adapter, cache=cache, forward_out=fo,
            epic_treat_i=EPIC_treat,
            epic_treat_mask_i=mask_treat,
            idx=all_idx, weights=weights,
        )

        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

        with open(log_path, "a") as f:
            f.write(f"{ep}\t{parts['total']:.6f}\t{parts['LOO']:.6f}\t"
                    f"{parts['seq']:.6f}\t{parts['image']:.6f}\t{parts['image_conf']:.6f}\t"
                    f"{parts['epic']:.6f}\t{parts['prior']:.6f}\t"
                    f"{parts['sigma2_pred_mean']:.4f}\t"
                    f"{parts['sigma2_raw_mean']:.4f}\t{sched.get_last_lr()[0]:.2e}\n")

        if ep == 0 or (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            print(f"[train] ep {ep:>4}  total={parts['total']:.4f}  "
                  f"LOO={parts['LOO']:.4f}  seq={parts['seq']:.4f}  "
                  f"img={parts['image']:.4f}  epic={parts['epic']:.4f}  "
                  f"σ²_pred={parts['sigma2_pred_mean']:.3f}")

    print(f"[train] done in {(time.time() - t0)/60:.1f} min")

    # ---- 5) Save ----
    adapter_path = os.path.join(outdir, "adapter.pt")
    torch.save(adapter.state_dict(), adapter_path)
    cfg = {
        "stage1_outdir":    stage1_outdir,
        "manifest_path":    manifest_path,
        "condition":        condition,
        "cond_names":       list(cond_names),
        "n_epochs":         n_epochs,
        "learn_rate":       learn_rate,
        "weight_decay":     weight_decay,
        "weights":          asdict(weights),
        "d_e":              d_e, "d_ctx": d_ctx, "d_attn": d_attn,
        "d_clust":          d_clust, "d_cond": d_cond,
        "hidden":           hidden, "dropout": dropout,
        "sigma2_raw_floor": sigma2_raw_floor,
        "sigma2_raw_scale": sigma2_raw_scale,
        "sigma2_pred_init": sigma2_pred_init,
        "sigma2_epic_path": sigma2_epic_path,
        "coherence_gate":   coherence_gate,
        "coherence_gate_gamma": coherence_gate_gamma,
        "spherical":        spherical,
        "factorized":       factorized,
        "drift_remove":     drift_remove,
        "unified":          unified,
        "epic_name":        cache.epic_name,
        "k":                k,
        "leiden_resolution": leiden_resolution,
        "cluster_compat_temp": cluster_compat_temp,
        "seq_modality_hint":   seq_modality_hint,
        "image_modality_hint": image_modality_hint,
        "seed":             seed,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[train] saved adapter -> {adapter_path}")
    return adapter, cfg


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1_outdir", required=True,
                   help="Stage 1 model directory (has static_model.pth + "
                        "static_latent.tsv + per_modality/<m>.tsv)")
    p.add_argument("--manifest",  required=True, help="Same manifest used by Stage 1")
    p.add_argument("--outdir",    required=True)
    p.add_argument("--condition", required=True,
                   help="One of the treated condition labels in the manifest, "
                        "e.g. 'cisplatin' or 'vorinostat'.")
    p.add_argument("--cond_names", nargs="+", default=["cisplatin", "vorinostat"],
                   help="All condition labels in the embedding table (so the "
                        "saved adapter can be loaded for any of these conditions later).")
    p.add_argument("--epic_name", default="epic",
                   help="EPIC modality name (case-insensitive match against the "
                        "Stage 1 model's encoder keys).")
    p.add_argument("--k", type=int, default=20,
                   help="kNN neighbourhood size for the cache graph.")
    p.add_argument("--leiden_resolution", type=float, default=1.0,
                   help="Leiden resolution for the cache's z-space clustering.")
    p.add_argument("--cluster_compat_temp", type=float, default=1.0,
                   help="<1 sharpens the cluster_compat edge gating; >1 softens.")

    p.add_argument("--n_epochs",  type=int, default=300)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--wd",        type=float, default=1e-4)
    p.add_argument("--w_LOO",     type=float, default=1.0)
    p.add_argument("--w_seq",     type=float, default=1.0)
    p.add_argument("--w_image",   type=float, default=0.05,
                   help="SOFT regulariser on D_image(z_treat) vs D_image(z_ref). "
                        "Kept small because real treatment-induced translocations "
                        "occur and a heavy weight would suppress that biology. "
                        "Set to 0 to disable entirely.")
    p.add_argument("--w_image_confidence", type=float, default=0.0,
                   help="Translocation-aware image loss: penalises ENTROPY "
                        "increases in D_image(z_treat) vs D_image(z_ref) without "
                        "constraining the direction of localization change. "
                        "Set positive (e.g. 0.1) to enable.")
    p.add_argument("--w_epic",    type=float, default=1.0)
    p.add_argument("--unified", action="store_true",
                   help="Single coherence-weighted drift-removed movement (magnitude+direction co-derived; supersedes combiner/factorized/gate).")
    p.add_argument("--drift_remove", action="store_true",
                   help="Remove the global treatment drift before magnitude/direction (isolates complex-specific signal; enables a stable population).")
    p.add_argument("--factorized_delta", action="store_true",
                   help="Factorize movement δ = m(p)·u(p): direction u from the "
                        "combiner, learned scalar magnitude m (sparsified). Requires "
                        "--spherical. Rank differential proteins by the learned m.")
    p.add_argument("--w_mag", type=float, default=1.0,
                   help="Weight matching learned magnitude to observed magnitude.")
    p.add_argument("--w_mag_l1", type=float, default=0.05,
                   help="L1 sparsity on the learned magnitude (stable majority).")
    p.add_argument("--spherical", action="store_true",
                   help="Renormalize z_treat onto the unit hypersphere (movement "
                        "becomes purely angular) AND use the cosine EPIC anchor. "
                        "Geometry-correct for the L2-normalized co-embedding. Off "
                        "by default → original Euclidean z_treat = z + δ + MSE anchor.")
    p.add_argument("--coherence_gate", action="store_true",
                   help="Scale each protein's movement by max(0, cos(δ_raw_proj, δ̂)). "
                        "Suppresses isolated/noisy differentials to ~0° (stable) and "
                        "keeps neighbour-coherent complex-wide remodelling → a stable "
                        "majority + remodelled tail.")
    p.add_argument("--coherence_gate_gamma", type=float, default=1.0,
                   help="Sharpness exponent on the coherence gate (>1 = stricter).")
    p.add_argument("--w_residual", type=float, default=0.0,
                   help="L2 penalty on ||delta_residual||² (the learned residual "
                        "projection). Constrains free drift in the decoder null "
                        "space — the main source of magnitude hallucination.")
    p.add_argument("--w_prior",   type=float, default=0.0,
                   help="Mahalanobis prior on z (VAE-aware). Penalises "
                        "z_treat for excess Mahalanobis distance to the "
                        "Stage 1 training distribution. Only fires when "
                        "Stage 1 is VAE-trained (z_emp_mu/Sigma exist). "
                        "Try 0.05-0.1 for VAE Stage 1; ignored for v1.")

    p.add_argument("--d_e",       type=int, default=128)
    p.add_argument("--d_ctx",     type=int, default=128)
    p.add_argument("--d_attn",    type=int, default=128)
    p.add_argument("--d_clust",   type=int, default=16)
    p.add_argument("--d_cond",    type=int, default=8)
    p.add_argument("--hidden",    type=int, default=256)
    p.add_argument("--dropout",   type=float, default=0.1)
    p.add_argument("--sigma2_raw_floor", type=float, default=0.05)
    p.add_argument("--sigma2_raw_scale", type=float, default=1.0)
    p.add_argument("--sigma2_pred_init", type=float, default=None,
                   help="Initial σ²_pred value for the heteroscedastic head. "
                        "If unset (default), auto-computed as mean(σ²_EPIC) × "
                        "sigma2_raw_scale so the Bayesian combination starts "
                        "balanced 50/50 — prevents σ²_pred collapse to the floor.")

    p.add_argument("--sigma2_epic_path", default=None,
                   help="TSV (protein<tab>sigma2) of EMPIRICAL per-protein "
                        "replicate σ² to use as σ²_EPIC instead of the Stage-1 "
                        "reconstruction heuristic. Gives the Bayesian combiner "
                        "real per-protein reliability so the observed differential "
                        "is kept for clean proteins. Pair with a low "
                        "--sigma2_raw_floor so the clean end isn't clipped.")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--device",    default=None)
    p.add_argument("--seq_modality_hint",   default=None)
    p.add_argument("--image_modality_hint", default=None)

    args = p.parse_args()
    weights = LossWeights(LOO=args.w_LOO, seq=args.w_seq, image=args.w_image,
                          image_confidence=args.w_image_confidence,
                          epic=args.w_epic, prior=args.w_prior,
                          residual=args.w_residual,
                          mag=args.w_mag, mag_l1=args.w_mag_l1)
    train_adapter(
        stage1_outdir=args.stage1_outdir,
        manifest_path=args.manifest, outdir=args.outdir,
        condition=args.condition, cond_names=tuple(args.cond_names),
        n_epochs=args.n_epochs, learn_rate=args.lr, weight_decay=args.wd,
        weights=weights,
        d_e=args.d_e, d_ctx=args.d_ctx, d_attn=args.d_attn,
        d_clust=args.d_clust, d_cond=args.d_cond, hidden=args.hidden,
        dropout=args.dropout,
        sigma2_raw_floor=args.sigma2_raw_floor,
        sigma2_raw_scale=args.sigma2_raw_scale,
        sigma2_pred_init=args.sigma2_pred_init,
        sigma2_epic_path=args.sigma2_epic_path,
        coherence_gate=args.coherence_gate,
        coherence_gate_gamma=args.coherence_gate_gamma,
        spherical=args.spherical,
        factorized=args.factorized_delta,
        drift_remove=args.drift_remove,
        unified=args.unified,
        epic_name=args.epic_name,
        k=args.k,
        leiden_resolution=args.leiden_resolution,
        cluster_compat_temp=args.cluster_compat_temp,
        seed=args.seed, device=args.device,
        seq_modality_hint=args.seq_modality_hint,
        image_modality_hint=args.image_modality_hint,
    )


if __name__ == "__main__":
    main()
