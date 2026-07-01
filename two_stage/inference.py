"""Stage 2 v3 inference.

Loads a trained adapter + the Stage1Cache it was trained against, runs the
forward pass on the full protein universe, and dumps:

    z_treat.tsv                — treated joint embedding (drop-in replacement
                                  for static_latent.tsv at this condition)
    delta_final.tsv            — δ_final (raw + neighbour-blended), in z-space
    delta_hat.tsv              — δ̂ (neighbour-only prediction)
    delta_raw_proj.tsv         — projected raw delta (baseline + residual)
    sigma2_pred.tsv            — per-protein uncertainty (high = isolated)
    coherence.tsv              — cos(δ_raw_proj, δ̂) per protein — useful for
                                  interpretability ("did neighbours agree with
                                  this protein's observation?")
"""
from __future__ import annotations
from typing import Optional
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from .cache import Stage1Cache
from .model import NeighborhoodAdapter
from .train import load_aligned_epic


def _write_tsv(path: str, names, X: np.ndarray, prefix="d"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if X.ndim == 1: X = X[:, None]
    d = X.shape[1]
    with open(path, "w") as f:
        f.write("protein\t" + "\t".join(f"{prefix}{i}" for i in range(d)) + "\n")
        for n, row in zip(names, X):
            f.write(n + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")


def run_inference(adapter_dir: str, manifest_path: str, outdir: str,
                  *, device: Optional[str] = None) -> dict:
    """Apply a trained adapter, write outputs to `outdir`."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)

    # ---- 1) load config ----
    cfg_path = os.path.join(adapter_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    condition = cfg["condition"]
    print(f"[inf] adapter dir: {adapter_dir}")
    print(f"[inf] condition:   {condition}")

    # ---- 2) build cache in-memory from Stage 1 outputs (no persisted cache) ----
    stage1_outdir = cfg.get("stage1_outdir")
    if not stage1_outdir:
        raise RuntimeError("config.json missing 'stage1_outdir'. This adapter "
                           "was likely trained with an older version of the "
                           "training script that used a cache file. Retrain.")
    cache = Stage1Cache.from_stage1_dir(
        stage1_outdir, cfg["manifest_path"],
        epic_name=cfg.get("epic_name", "epic"),
        k=cfg.get("k", 20),
        leiden_resolution=cfg.get("leiden_resolution", 1.0),
        cluster_compat_temp=cfg.get("cluster_compat_temp", 1.0),
        device=device, seed=cfg.get("seed", 0),
        seq_name=cfg.get("seq_modality_hint"),
        image_name=cfg.get("image_modality_hint"),
        sigma2_epic_path=cfg.get("sigma2_epic_path"),
    )
    cache.z = cache.z.to(device)
    cache.cluster_id = cache.cluster_id.to(device)
    cache.knn_idx = cache.knn_idx.to(device)
    cache.knn_w   = cache.knn_w.to(device)
    cache.sigma2_epic = cache.sigma2_epic.to(device)
    cache.h_per_modality = {m: H.to(device) for m, H in cache.h_per_modality.items()}

    # ---- 3) construct adapter, load weights ----
    # IMPORTANT: dropout must match training-time value because the dropout
    # layer occupies an index in nn.Sequential. Setting dropout=0 here would
    # produce a state_dict shape mismatch on load. We rely on adapter.eval()
    # below to actually disable dropout's effect at inference.
    adapter = NeighborhoodAdapter(
        cache=cache,
        cond_names=cfg["cond_names"],
        d_e=cfg["d_e"], d_ctx=cfg["d_ctx"], d_attn=cfg["d_attn"],
        d_clust=cfg["d_clust"], d_cond=cfg["d_cond"],
        hidden=cfg["hidden"], dropout=cfg["dropout"],
        sigma2_raw_floor=cfg["sigma2_raw_floor"],
        sigma2_raw_scale=cfg["sigma2_raw_scale"],
        sigma2_pred_init=cfg.get("sigma2_pred_init"),
        coherence_gate=cfg.get("coherence_gate", False),
        coherence_gate_gamma=cfg.get("coherence_gate_gamma", 1.0),
        spherical=cfg.get("spherical", False),
        factorized=cfg.get("factorized", False),
        drift_remove=cfg.get("drift_remove", False),
        unified=cfg.get("unified", False),
    ).to(device)
    state = torch.load(os.path.join(adapter_dir, "adapter.pt"),
                       map_location=device, weights_only=False)
    adapter.load_state_dict(state)
    adapter.eval()

    # ---- 4) EPIC + delta inputs ----
    EPIC_ctrl, EPIC_treat, mask_treat = load_aligned_epic(
        manifest_path, cache.epic_name, cache.proteins, condition)
    EPIC_ctrl  = EPIC_ctrl.to(device)
    EPIC_treat = EPIC_treat.to(device)
    delta_raw_all = (EPIC_treat - EPIC_ctrl)

    # ---- 5) forward ----
    all_idx = torch.arange(cache.N, device=device)
    cond_ids = torch.full((cache.N,), adapter.cond_to_id[condition],
                          dtype=torch.long, device=device)
    with torch.no_grad():
        delta_raw_neigh = delta_raw_all[cache.knn_idx]
        fo = adapter.forward_with_neighbour_delta(
            idx=all_idx, epic_ctrl_i=EPIC_ctrl, epic_treat_i=EPIC_treat,
            delta_raw_i=delta_raw_all, delta_raw_neigh=delta_raw_neigh,
            cond_id=cond_ids,
        )
        # neighbourhood coherence: cos(δ_raw_proj, δ̂) per protein
        a = F.normalize(fo["delta_raw_proj"], dim=-1)
        b = F.normalize(fo["delta_hat"],       dim=-1)
        coherence = (a * b).sum(dim=-1)

    # ---- 6) dump ----
    proteins = cache.proteins
    _write_tsv(os.path.join(outdir, "z_treat.tsv"),
               proteins, fo["z_treat"].cpu().numpy())
    _write_tsv(os.path.join(outdir, "delta_final.tsv"),
               proteins, fo["delta_final"].cpu().numpy())
    _write_tsv(os.path.join(outdir, "delta_hat.tsv"),
               proteins, fo["delta_hat"].cpu().numpy())
    _write_tsv(os.path.join(outdir, "delta_raw_proj.tsv"),
               proteins, fo["delta_raw_proj"].cpu().numpy())
    _write_tsv(os.path.join(outdir, "sigma2_pred.tsv"),
               proteins, fo["sigma2_pred"].cpu().numpy(), prefix="s2")
    _write_tsv(os.path.join(outdir, "coherence.tsv"),
               proteins, coherence.cpu().numpy(), prefix="coh")
    if fo.get("learned_magnitude") is not None:
        import pandas as _pd
        _pd.Series(fo["learned_magnitude"].cpu().numpy(), index=proteins,
                   name="learned_magnitude").to_csv(
            os.path.join(outdir, "learned_magnitude.tsv"), sep="\t", header=True)
        print("      learned_magnitude.tsv → factorized per-protein magnitude (rank differential proteins by this)")
    print(f"[inf] outputs in {outdir}")
    print(f"      z_treat.tsv      → drop-in replacement for static_latent.tsv at {condition}")
    print(f"      delta_final.tsv  → the denoised remodelling vector per protein")
    print(f"      coherence.tsv    → 1 cell per protein, in [-1, 1]: agreement of "
          "neighbour prediction with raw observation")
    return {
        "outdir": outdir,
        "z_treat":      fo["z_treat"].cpu(),
        "delta_final":  fo["delta_final"].cpu(),
        "delta_hat":    fo["delta_hat"].cpu(),
        "sigma2_pred":  fo["sigma2_pred"].cpu(),
        "coherence":    coherence.cpu(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter_dir", required=True, help="Directory containing adapter.pt + config.json")
    p.add_argument("--manifest",    required=True)
    p.add_argument("--outdir",      required=True)
    p.add_argument("--device",      default=None)
    args = p.parse_args()
    run_inference(args.adapter_dir, args.manifest, args.outdir, device=args.device)


if __name__ == "__main__":
    main()
