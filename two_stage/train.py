#!/usr/bin/env python3
"""Training entrypoints for the two-stage model.

Train Stage 1, Stage 2, or both with ``--stage {1,2,both}`` (default: both).

    # both stages, end to end
    python -m two_stage.train --stage both --manifest m.json --outdir out [opts]
    # only the Stage 1 reference map
    python -m two_stage.train --stage 1 --manifest m.json --outdir out/stage1
    # only a Stage 2 adapter on an existing Stage 1
    python -m two_stage.train --stage 2 --stage1_outdir out/stage1 \
        --manifest m.json --conditions cisplatin --outdir out/stage2
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dropout import apply_dropout, random_modality_dropout
from .losses import total_loss, LossWeights, compose_loss
from .model import Stage1, make_model, NeighborhoodAdapter
from .pseudo_labels import compute_all_pseudo_labels
from .cache import Stage1Cache, load_epic_per_condition, _align


# ===================== Stage 1 training =====================



# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _read_tsv(path: str) -> Tuple[List[str], np.ndarray]:
    """First column = protein name, remaining = float features."""
    proteins, rows = [], []
    with open(path) as f:
        header = f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            try:
                vec = [float(x) for x in p[1:]]
            except ValueError:
                continue
            proteins.append(p[0])
            rows.append(vec)
    return proteins, np.asarray(rows, dtype=np.float32)


_KNOWN_TREATED = {"cisplatin", "vorinostat", "negativectrl", "negative_ctrl",
                  "treated", "drug"}


def _is_untreated_condition(cond: str) -> bool:
    """Return True if `cond` is NOT one of the known treated condition labels.
    We accept any label not in the treated set — including empty/None,
    'untreated', 'untreated_1', 'untreated_full', etc. — as 'untreated'."""
    if cond is None:
        return True
    c = str(cond).strip().lower()
    if not c:
        return True
    return c not in _KNOWN_TREATED


def load_modality_matrices(manifest_path: str,
                           untreated_only: bool = True
                           ) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Read a manifest.json and load each modality's untreated input matrix.

    Returns {modality_name: (proteins, X[n, d])}. Modalities with multiple
    replicate entries are averaged across replicates (aligned on the union
    of their proteins).

    Robust to manifest variations:
      - Accepts any condition label that ISN'T in the known-treated set
        (cisplatin, vorinostat, negativeCTRL, ...) as 'untreated'. So
        'untreated', 'untreated_1', 'untreated_full', or no condition at all
        are all fine.
      - Does NOT fall back to `treated_path` — that's the treated data.
        Only `path` (the untreated input) is used.
      - Prints a diagnostic line per skipped entry so empty results are
        easy to debug.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    print(f"[muse] manifest: {manifest_path}  ({len(entries)} entries)")

    per_mod: Dict[str, List[Tuple[List[str], np.ndarray]]] = {}
    n_used = n_treated = n_no_path = n_no_mod = 0
    for e in entries:
        m = e.get("modality")
        if not m:
            n_no_mod += 1
            print(f"  [skip] entry without modality: {e}")
            continue
        cond = e.get("condition")
        if untreated_only and not _is_untreated_condition(cond):
            n_treated += 1
            continue
        path = e.get("path")
        if not path or not os.path.isfile(path):
            n_no_path += 1
            print(f"  [skip] {m} (condition={cond!r}): path missing/inaccessible -> {path!r}")
            continue
        per_mod.setdefault(m, []).append(_read_tsv(path))
        n_used += 1
    print(f"[muse] loaded {n_used} entries  |  skipped: treated={n_treated}  "
          f"no-path={n_no_path}  no-modality={n_no_mod}")
    print(f"[muse] modalities found: {sorted(per_mod)}")

    # Reduce: average replicate features per modality after aligning on the
    # union of proteins across the modality's replicate streams.
    out: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for m, streams in per_mod.items():
        if len(streams) == 1:
            out[m] = streams[0]
            continue
        prots = sorted(set().union(*[set(s[0]) for s in streams]))
        d = streams[0][1].shape[1]
        accum = np.zeros((len(prots), d), dtype=np.float32)
        count = np.zeros(len(prots), dtype=np.float32)
        for plist, X in streams:
            idx = {p: i for i, p in enumerate(plist)}
            for pi, p in enumerate(prots):
                if p in idx:
                    accum[pi] += X[idx[p]]
                    count[pi] += 1
        m_avg = accum / np.maximum(count, 1)[:, None]
        # Keep only proteins seen at least once
        keep = count > 0
        out[m] = ([p for p, k in zip(prots, keep) if k], m_avg[keep])
    return out


def build_universe(modality_matrices: Dict[str, Tuple[List[str], np.ndarray]]
                   ) -> List[str]:
    """Ordered union of proteins across all modalities."""
    u = sorted(set().union(*[set(p) for p, _ in modality_matrices.values()]))
    return u


def assemble_tensors(universe: List[str],
                     modality_matrices: Dict[str, Tuple[List[str], np.ndarray]]
                     ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build (X_m: [N, d_m], mask_m: [N]) per modality aligned to `universe`.
    Missing rows are zero-filled and have mask=0."""
    N = len(universe)
    p2i = {p: i for i, p in enumerate(universe)}
    Xs, Ms = {}, {}
    for m, (prots, X) in modality_matrices.items():
        d = X.shape[1]
        Xm = np.zeros((N, d), dtype=np.float32)
        Mm = np.zeros(N, dtype=np.float32)
        for pj, p in enumerate(prots):
            i = p2i.get(p)
            if i is None:
                continue
            Xm[i] = X[pj]
            Mm[i] = 1.0
        Xs[m] = Xm
        Ms[m] = Mm
        print(f"  modality {m:<10} d={d:<5}  coverage={int(Mm.sum())}/{N}")
    return Xs, Ms


def build_pseudo_label_tensor(
    universe: List[str],
    pseudo_labels: Dict[str, Dict[str, int]],
) -> Dict[str, np.ndarray]:
    """For each modality, return a (N,) int array with cluster id or -1 if missing."""
    out: Dict[str, np.ndarray] = {}
    for m, prot2lab in pseudo_labels.items():
        labs = np.full(len(universe), -1, dtype=np.int64)
        for i, p in enumerate(universe):
            if p in prot2lab:
                labs[i] = prot2lab[p]
        out[m] = labs
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MUSEDataset(Dataset):
    def __init__(self, Xs: Dict[str, np.ndarray], Ms: Dict[str, np.ndarray],
                 labels: Dict[str, np.ndarray]):
        self.Xs = {m: torch.from_numpy(X) for m, X in Xs.items()}
        self.Ms = {m: torch.from_numpy(M) for m, M in Ms.items()}
        self.labels = {m: torch.from_numpy(L) for m, L in labels.items()}
        first = next(iter(Xs.values()))
        self.N = first.shape[0]

    def __len__(self): return self.N

    def __getitem__(self, idx):
        inputs = {m: X[idx] for m, X in self.Xs.items()}
        masks = {m: M[idx] for m, M in self.Ms.items()}
        labs = {m: L[idx] for m, L in self.labels.items()}
        return inputs, masks, labs


def collate(batch):
    inputs = {m: torch.stack([b[0][m] for b in batch]) for m in batch[0][0]}
    masks  = {m: torch.stack([b[1][m] for b in batch]) for m in batch[0][1]}
    labels = {m: torch.stack([b[2][m] for b in batch]) for m in batch[0][2]}
    return inputs, masks, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _write_tsv(path: str, names: List[str], X: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = X.shape[1]
    with open(path, "w") as f:
        f.write("protein\t" + "\t".join(f"d{i}" for i in range(d)) + "\n")
        for n, row in zip(names, X):
            f.write(n + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")


def train_stage1(
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
    p_drop: float = 0.3,
    dropout_min_keep: int = 1,
    pseudo_method: str = "leiden",
    pseudo_kmeans_k: int = 50,
    pseudo_knn_k: int = 15,
    pseudo_resolution: float = 1.0,
    seed: int = 0,
    device: Optional[str] = None,
):
    """End-to-end MUSE Stage 1 training."""
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1) Load + assemble
    print("[muse] loading modality matrices...")
    mods = load_modality_matrices(manifest_path)
    if not mods:
        raise RuntimeError(
            f"No untreated modality matrices were loaded from {manifest_path}. "
            "Diagnostic skip-reasons are printed above. Common causes: "
            "(a) every entry has a 'condition' the loader interpreted as treated "
            "— this loader accepts any condition not in "
            f"{sorted(_KNOWN_TREATED)} as untreated, so check whether your "
            "manifest uses an unexpected treated label; "
            "(b) the 'path' field is missing or points to a file that doesn't "
            "exist; (c) the manifest stores file paths only in 'treated_path' "
            "(this loader does NOT read treated_path — by design — since that's "
            "the treated condition's data, not the untreated input). "
            "Print one of your manifest entries to inspect its shape."
        )
    universe = build_universe(mods)
    print(f"[muse] universe size: {len(universe)} proteins; modalities: {list(mods)}")
    Xs, Ms = assemble_tensors(universe, mods)
    modality_dims = {m: X.shape[1] for m, X in Xs.items()}

    # 2) Pseudo-labels (per modality)
    print("[muse] computing pseudo-labels...")
    pl_cache = os.path.join(outdir, "pseudo_labels.json")
    pseudo_labels = compute_all_pseudo_labels(
        mods, method=pseudo_method,
        knn_k=pseudo_knn_k, leiden_resolution=pseudo_resolution,
        kmeans_k=pseudo_kmeans_k, seed=seed, cache_path=pl_cache,
    )
    label_arrays = build_pseudo_label_tensor(universe, pseudo_labels)

    # 3) Dataset/loader
    ds = MUSEDataset(Xs, Ms, label_arrays)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True, num_workers=0)

    # 4) Model + optimizer
    model = make_model(modality_dims,
                       latent_dim_per_modality=latent_dim_per_modality,
                       joint_dim=joint_dim, hidden_dim=hidden_dim,
                       dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate,
                             weight_decay=weight_decay)
    print(f"[muse] model: joint_dim={joint_dim}, per-modality latent={latent_dim_per_modality}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # 5) Train
    log_path = os.path.join(outdir, "static_loss.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\ttotal\trecon\tstruct\tdrop_frac\n")
    t0 = time.time()
    print(f"[muse] modality dropout: p_drop={p_drop}  min_keep={dropout_min_keep}")
    for ep in range(n_epochs):
        model.train()
        losses_t, losses_r, losses_s, drop_fracs = [], [], [], []
        for inputs, masks, labels in loader:
            # Originals (recon TARGETS use these)
            inputs_orig   = {m: v.to(device) for m, v in inputs.items()}
            masks_present = {m: v.to(device) for m, v in masks.items()}
            labels        = {m: v.to(device) for m, v in labels.items()}

            # Modality dropout (training only): hide modalities at the encoder
            # input but keep recon targets unchanged. The decoder is held
            # accountable for reconstructing the hidden modalities, which
            # forces cross-modal prediction + missing-data robustness.
            if p_drop > 0.0:
                keep = random_modality_dropout(masks_present,
                                               p_drop=p_drop,
                                               min_keep=dropout_min_keep)
                inputs_vis, masks_vis = apply_dropout(inputs_orig, masks_present, keep)
                # diagnostic: fraction of (sample, modality) actually dropped
                dropped = sum((1.0 - keep[m].mean()).item() for m in keep) / max(1, len(keep))
                drop_fracs.append(dropped)
            else:
                inputs_vis, masks_vis = inputs_orig, masks_present
                drop_fracs.append(0.0)

            # Encode from the (possibly dropout-modified) visible inputs.
            z, _ = model.encode(inputs_vis, masks_vis)

            # Loss targets use the ORIGINAL inputs + presence masks — this
            # is the part that turns dropout into a cross-modal prediction
            # objective rather than just noise.
            lossd = total_loss(model, z, inputs_orig, masks_present, labels,
                               lambda_recon=lambda_recon,
                               lambda_struct=lambda_struct,
                               margin=margin)
            optim.zero_grad(); lossd["total"].backward(); optim.step()
            losses_t.append(lossd["total"].item())
            losses_r.append(lossd["recon"].item())
            losses_s.append(lossd["struct"].item())
        with open(log_path, "a") as f:
            f.write(f"{ep}\t{np.mean(losses_t):.6f}\t{np.mean(losses_r):.6f}"
                    f"\t{np.mean(losses_s):.6f}\t{np.mean(drop_fracs):.4f}\n")
        if ep == 0 or (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            print(f"[muse] ep {ep:>4}  total={np.mean(losses_t):.4f}  "
                  f"recon={np.mean(losses_r):.4f}  struct={np.mean(losses_s):.4f}  "
                  f"dropped={np.mean(drop_fracs):.3f}")
    print(f"[muse] training done in {(time.time()-t0)/60:.1f} min")

    # 6) Inference: z per protein
    print("[muse] computing per-protein joint z...")
    model.eval()
    Z = np.zeros((len(universe), joint_dim), dtype=np.float32)
    H_per = {m: np.zeros((len(universe), latent_dim_per_modality), dtype=np.float32)
             for m in modality_dims}
    with torch.no_grad():
        # Stream batches by index
        for start in range(0, len(universe), batch_size):
            end = min(start + batch_size, len(universe))
            inputs = {m: torch.from_numpy(Xs[m][start:end]).to(device) for m in modality_dims}
            masks  = {m: torch.from_numpy(Ms[m][start:end]).to(device) for m in modality_dims}
            z, h = model.encode(inputs, masks)
            Z[start:end] = z.cpu().numpy()
            for m in modality_dims:
                H_per[m][start:end] = h[m].cpu().numpy()

    # Save: anchor (L2-normed) + raw + per-modality + model
    norms = np.linalg.norm(Z, axis=1, keepdims=True); norms = np.maximum(norms, 1e-9)
    Zn = Z / norms
    _write_tsv(os.path.join(outdir, "static_latent.tsv"), universe, Zn)
    _write_tsv(os.path.join(outdir, "static_latent_raw.tsv"), universe, Z)
    permdir = os.path.join(outdir, "per_modality")
    for m, H in H_per.items():
        _write_tsv(os.path.join(permdir, f"{m}.tsv"), universe, H)
    torch.save(model.state_dict(), os.path.join(outdir, "static_model.pth"))
    print(f"[muse] outputs in {outdir}")
    print("  static_latent.tsv         (joint z, L2-normed) — anchor for downstream Stage 2 / hierarchy")
    print("  static_latent_raw.tsv     (joint z, pre-norm)")
    print("  per_modality/<m>.tsv      (per-modality h_m)")
    print("  static_model.pth          (encoder/decoder/fusion weights)")
    print("  pseudo_labels.json        (cluster labels used in training)")
    print("  static_loss.tsv           (per-epoch loss curves)")
    return model, universe, Zn

# ===================== Stage 2 training =====================



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


# ===================== Unified CLI =====================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["1", "2", "both"], default="both",
                   help="Which stage(s) to train. Default: both.")
    p.add_argument("--manifest", required=True, help="Path to manifest.json")
    p.add_argument("--outdir", required=True,
                   help="Output root. For --stage both: Stage 1 -> <outdir>/stage1, "
                        "adapters -> <outdir>/stage2/<cond>. For --stage 1 it IS the "
                        "Stage-1 dir; for --stage 2, adapters -> <outdir>/<cond>.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if omitted).")

    # ---- Stage 1 ----
    g1 = p.add_argument_group("Stage 1 (reference map)")
    g1.add_argument("--latent_dim_per_modality", type=int, default=64)
    g1.add_argument("--joint_dim", type=int, default=256)
    g1.add_argument("--hidden_dim", type=int, default=256)
    g1.add_argument("--s1_dropout", type=float, default=0.0)
    g1.add_argument("--s1_epochs", type=int, default=300)
    g1.add_argument("--s1_batch_size", type=int, default=256)
    g1.add_argument("--s1_lr", type=float, default=1e-4)
    g1.add_argument("--s1_wd", type=float, default=1e-5)
    g1.add_argument("--lambda_recon", type=float, default=1.0)
    g1.add_argument("--lambda_struct", type=float, default=1.0)
    g1.add_argument("--margin", type=float, default=0.3)
    g1.add_argument("--p_drop", type=float, default=0.3)
    g1.add_argument("--dropout_min_keep", type=int, default=1)
    g1.add_argument("--pseudo_method", default="leiden", choices=["leiden", "kmeans"])
    g1.add_argument("--pseudo_knn_k", type=int, default=15)
    g1.add_argument("--pseudo_resolution", type=float, default=1.0)
    g1.add_argument("--pseudo_kmeans_k", type=int, default=50)

    # ---- Stage 2 ----
    g2 = p.add_argument_group("Stage 2 (perturbation adapter)")
    g2.add_argument("--stage1_outdir", default=None,
                    help="Existing Stage-1 dir. Required for --stage 2; for "
                         "--stage both it defaults to <outdir>/stage1.")
    g2.add_argument("--conditions", nargs="+", default=None,
                    help="Conditions to train an adapter for (one per drug + the "
                         "negative control). Default: all of --cond_names.")
    g2.add_argument("--cond_names", nargs="+",
                    default=["cisplatin", "vorinostat", "negative_ctrl"],
                    help="All condition labels in the adapter's embedding table.")
    g2.add_argument("--epic_name", default="epic")
    g2.add_argument("--k", type=int, default=20)
    g2.add_argument("--leiden_resolution", type=float, default=1.0)
    g2.add_argument("--cluster_compat_temp", type=float, default=1.0)
    g2.add_argument("--n_epochs", type=int, default=300, help="Stage-2 epochs.")
    g2.add_argument("--lr", type=float, default=1e-3)
    g2.add_argument("--wd", type=float, default=1e-4)
    g2.add_argument("--w_LOO", type=float, default=1.0)
    g2.add_argument("--w_seq", type=float, default=1.0)
    g2.add_argument("--w_image", type=float, default=0.05)
    g2.add_argument("--w_image_confidence", type=float, default=0.0)
    g2.add_argument("--w_epic", type=float, default=1.0)
    g2.add_argument("--w_residual", type=float, default=0.0)
    g2.add_argument("--w_prior", type=float, default=0.0)
    g2.add_argument("--w_mag", type=float, default=1.0)
    g2.add_argument("--w_mag_l1", type=float, default=0.05)
    g2.add_argument("--unified", action="store_true",
                    help="Single coherence-weighted drift-removed movement (recommended).")
    g2.add_argument("--drift_remove", action="store_true",
                    help="Remove the global treatment drift (enables a stable population).")
    g2.add_argument("--factorized_delta", action="store_true")
    g2.add_argument("--spherical", action="store_true",
                    help="Keep movement on the unit-sphere tangent plane + cosine EPIC anchor.")
    g2.add_argument("--coherence_gate", action="store_true")
    g2.add_argument("--coherence_gate_gamma", type=float, default=1.0)
    g2.add_argument("--d_e", type=int, default=128)
    g2.add_argument("--d_ctx", type=int, default=128)
    g2.add_argument("--d_attn", type=int, default=128)
    g2.add_argument("--d_clust", type=int, default=16)
    g2.add_argument("--d_cond", type=int, default=8)
    g2.add_argument("--hidden", type=int, default=256)
    g2.add_argument("--dropout", type=float, default=0.1)
    g2.add_argument("--sigma2_raw_floor", type=float, default=0.05)
    g2.add_argument("--sigma2_raw_scale", type=float, default=1.0)
    g2.add_argument("--sigma2_pred_init", type=float, default=None)
    g2.add_argument("--sigma2_epic_path", default=None)
    g2.add_argument("--seq_modality_hint", default=None)
    g2.add_argument("--image_modality_hint", default=None)
    return p


def main():
    args = _build_parser().parse_args()

    if args.stage in ("1", "both"):
        s1_out = os.path.join(args.outdir, "stage1") if args.stage == "both" else args.outdir
        print(f"[two_stage] === Stage 1 -> {s1_out} ===")
        train_stage1(
            manifest_path=args.manifest, outdir=s1_out,
            latent_dim_per_modality=args.latent_dim_per_modality,
            joint_dim=args.joint_dim, hidden_dim=args.hidden_dim, dropout=args.s1_dropout,
            n_epochs=args.s1_epochs, batch_size=args.s1_batch_size,
            learn_rate=args.s1_lr, weight_decay=args.s1_wd,
            lambda_recon=args.lambda_recon, lambda_struct=args.lambda_struct,
            margin=args.margin, p_drop=args.p_drop, dropout_min_keep=args.dropout_min_keep,
            pseudo_method=args.pseudo_method, pseudo_kmeans_k=args.pseudo_kmeans_k,
            pseudo_knn_k=args.pseudo_knn_k, pseudo_resolution=args.pseudo_resolution,
            seed=args.seed, device=args.device,
        )

    if args.stage in ("2", "both"):
        if args.stage == "both":
            stage1_dir = os.path.join(args.outdir, "stage1")
        else:
            stage1_dir = args.stage1_outdir
            if not stage1_dir:
                raise SystemExit("--stage 2 requires --stage1_outdir (an existing Stage-1 dir).")
        conditions = args.conditions if args.conditions else list(args.cond_names)
        weights = LossWeights(LOO=args.w_LOO, seq=args.w_seq, image=args.w_image,
                              image_confidence=args.w_image_confidence,
                              epic=args.w_epic, prior=args.w_prior,
                              residual=args.w_residual, mag=args.w_mag, mag_l1=args.w_mag_l1)
        for cond in conditions:
            adapter_out = (os.path.join(args.outdir, "stage2", cond)
                           if args.stage == "both" else os.path.join(args.outdir, cond))
            print(f"[two_stage] === Stage 2 adapter [{cond}] -> {adapter_out} ===")
            train_adapter(
                stage1_outdir=stage1_dir, manifest_path=args.manifest, outdir=adapter_out,
                condition=cond, cond_names=tuple(args.cond_names),
                n_epochs=args.n_epochs, learn_rate=args.lr, weight_decay=args.wd,
                weights=weights,
                d_e=args.d_e, d_ctx=args.d_ctx, d_attn=args.d_attn,
                d_clust=args.d_clust, d_cond=args.d_cond, hidden=args.hidden,
                dropout=args.dropout,
                sigma2_raw_floor=args.sigma2_raw_floor, sigma2_raw_scale=args.sigma2_raw_scale,
                sigma2_pred_init=args.sigma2_pred_init, sigma2_epic_path=args.sigma2_epic_path,
                coherence_gate=args.coherence_gate, coherence_gate_gamma=args.coherence_gate_gamma,
                spherical=args.spherical, factorized=args.factorized_delta,
                drift_remove=args.drift_remove, unified=args.unified,
                epic_name=args.epic_name, k=args.k,
                leiden_resolution=args.leiden_resolution,
                cluster_compat_temp=args.cluster_compat_temp,
                seed=args.seed, device=args.device,
                seq_modality_hint=args.seq_modality_hint,
                image_modality_hint=args.image_modality_hint,
            )


if __name__ == "__main__":
    main()
