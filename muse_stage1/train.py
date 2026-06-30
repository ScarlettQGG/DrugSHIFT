"""Data loading + training loop for MUSE-style Stage 1.

Loads per-modality matrices from a manifest, builds a unified protein universe
(union across modalities), and trains the MUSE model with masked recon-from-z
+ within-modality semi-hard triplet losses.

Output files (saved to outdir):
    static_latent.tsv          joint z per protein (L2-normalised) — the anchor
    static_latent_raw.tsv      joint z per protein (pre-norm) — for Euclidean disp.
    static_model.pth           torch state_dict
    static_loss.tsv            per-epoch loss curves
    per_modality/<m>.tsv       per-modality latent h_m per protein (for diagnostics)
    pseudo_labels.json         cached cluster pseudo-labels (per modality)
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
from torch.utils.data import Dataset, DataLoader

from .dropout import apply_dropout, random_modality_dropout
from .losses import total_loss
from .model import MUSEStage1, make_model
from .pseudo_labels import compute_all_pseudo_labels


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


def train_muse_stage1(
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
