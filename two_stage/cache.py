"""Stage1Cache — frozen Stage-1 artefacts the v3 adapter needs.

What gets cached, and why:

    z              joint embedding (anchor) per protein
    h_<mod>        per-modality h_m BEFORE fusion — conditioning input for the
                   delta encoder. One per modality declared at Stage 1.
    cluster_id     Leiden pseudo-label on z-space — input feature AND
                   edge-weight multiplier for the kNN graph (Group C, soft form).
    cluster_proba  soft membership (one-hot smoothed), used for the soft
                   cluster_compat edge weight when cluster_proba is exposed.
    sigma2_epic    per-protein EPIC reliability, derived from the Stage-1
                   EPIC decoder's reconstruction residual. Reused directly
                   as σ²_raw in the Bayesian combination (feature 8).
    conf           per-protein anchor strength: harmonic mean of cos(h_m, z)
                   across modalities — high if every modality agrees with the
                   joint, low if z is supported by only one or two modalities.
                   Used as edge confidence (feature 5).
    knn_idx        k cosine-nearest neighbours in z-space (directed top-k;
                   the multiplicative edge weights below filter asymmetric
                   edges automatically).
    knn_w          edge weights (post-audit simplification — see README):
                       w_ij  =  conf_j  ·  cluster_compat(i, j)^(1/τ)
                   then row-normalised.  Two factors only — `conf_i` was
                   dropped (cancelled by row-norm) and `cos(z_i, z_j)` was
                   dropped (mostly redundant inside top-k cosine kNN).

    E_EPIC         frozen EPIC encoder — used at Stage-2 forward to project
                   EPIC_treat and EPIC_ctrl into z-space for δ_raw_proj (b).
    D_seq, D_image, D_epic
                   frozen Stage-1 decoders — used by the decoder-stability
                   losses (Group B).

The cache is computed once per Stage-1 run and pickled to disk so Stage 2
training never re-touches the Stage-1 model.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Local imports — kept lazy so the package is importable without the two_stage.stage1
# packages present (e.g. when loading a saved cache for inference).
def _import_muse_v1():
    from .model import Stage1, make_model
    return Stage1, make_model


def _import_muse_vae():
    raise RuntimeError("The VAE Stage-1 variant has been removed from this package.")


def _is_vae_stage1(stage1_outdir: str) -> bool:
    """Detect whether stage1_outdir is a VAE-trained model (Option II+).

    Three positive signals:
      1. vae_config.json exists with model_kind="MUSEStage1VAE"
      2. per_modality_sigma/ subdir exists
      3. state_dict has encoder mu_head / log_sigma2_head keys
    Any one of these is sufficient.
    """
    return False  # VAE Stage-1 variant removed — Stage 1 is always deterministic
    cfg_path = os.path.join(stage1_outdir, "vae_config.json")
    if os.path.isfile(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
            if cfg.get("model_kind") == "MUSEStage1VAE":
                return True
        except Exception:
            pass
    if os.path.isdir(os.path.join(stage1_outdir, "per_modality_sigma")):
        return True
    state_path = os.path.join(stage1_outdir, "static_model.pth")
    if os.path.isfile(state_path):
        try:
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            for k in st:
                if k.endswith(".mu_head.weight") or k.endswith(".log_sigma2_head.weight"):
                    return True
        except Exception:
            pass
    return False


def _resolve_modality_name(requested: Optional[str],
                            available: Sequence[str],
                            fuzzy: Sequence[str] = ()) -> Optional[str]:
    """Resolve a (possibly mis-cased / fuzzy) modality name to whatever the
    Stage-1 model actually used as its canonical key.

    Resolution order:
      1. exact match
      2. case-insensitive exact match
      3. any name in `available` containing `requested` (case-insensitive)
      4. any name in `available` matching one of `fuzzy` substrings
      5. None
    """
    if not available:
        return None
    avail = list(available)
    if requested:
        if requested in avail:
            return requested
        lower = {a.lower(): a for a in avail}
        if requested.lower() in lower:
            return lower[requested.lower()]
        for a in avail:
            if requested.lower() in a.lower() or a.lower() in requested.lower():
                return a
    for needle in fuzzy:
        for a in avail:
            if needle.lower() in a.lower():
                return a
    return None


# ───────────────────────────────────────────────────────────────────────────
# Data IO helpers
# ───────────────────────────────────────────────────────────────────────────

def _read_tsv(path: str) -> Tuple[List[str], np.ndarray]:
    """Stage-1's standard TSV format: first column = name, rest = float features."""
    proteins, rows = [], []
    with open(path) as f:
        _hdr = f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            try:
                vec = [float(x) for x in p[1:]]
            except ValueError:
                continue
            proteins.append(p[0])
            rows.append(vec)
    return proteins, np.asarray(rows, dtype=np.float32)


def _align(universe: List[str], proteins: List[str], X: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Align (proteins, X) onto `universe` — return (X_aligned, mask)."""
    idx = {p: i for i, p in enumerate(proteins)}
    d = X.shape[1]
    Xn = np.zeros((len(universe), d), dtype=np.float32)
    M = np.zeros(len(universe), dtype=np.float32)
    for i, p in enumerate(universe):
        j = idx.get(p)
        if j is not None:
            Xn[i] = X[j]; M[i] = 1.0
    return Xn, M


# ───────────────────────────────────────────────────────────────────────────
# Stage-1 model loading
# ───────────────────────────────────────────────────────────────────────────

def _infer_modality_dims_from_state(state: dict, is_vae: bool) -> Dict[str, int]:
    """Recover modality_dims from a saved MUSE state_dict.

    v1 deterministic encoder for modality m has first Linear at
        'encoders.<m>.net.0.weight'  shape (hidden, d_m)
    v2 VAE encoder has the trunk's first Linear at
        'encoders.<m>.trunk.net.0.weight'  shape (hidden, d_m)
    """
    out: Dict[str, int] = {}
    needle = ".trunk.net.0.weight" if is_vae else ".net.0.weight"
    for k, v in state.items():
        if k.startswith("encoders.") and k.endswith(needle):
            m = k[len("encoders."):-len(needle)]
            out[m] = int(v.shape[1])
    return out


def _infer_dims(state: dict, is_vae: bool) -> Tuple[Dict[str, int], int, int, int]:
    """Infer (modality_dims, joint_dim, latent_dim, hidden_dim) from state."""
    modality_dims = _infer_modality_dims_from_state(state, is_vae=is_vae)
    any_m = next(iter(modality_dims))
    if is_vae:
        hidden = int(state[f"encoders.{any_m}.trunk.net.0.weight"].shape[0])
        latent = int(state[f"encoders.{any_m}.mu_head.weight"].shape[0])
        # fusion is _MLP with always-on Dropout → final Linear is at net.3
        joint  = int(state["fusion.net.3.weight"].shape[0])
    else:
        hidden = int(state[f"encoders.{any_m}.net.0.weight"].shape[0])
        latent = int(state[f"encoders.{any_m}.net.3.weight"].shape[0])
        joint  = int(state["fusion.net.3.weight"].shape[0])
    return modality_dims, joint, latent, hidden


def load_muse_model(stage1_outdir: str, device: str = "cpu"
                    ) -> Tuple["nn.Module", Dict[str, int]]:
    """Reconstruct the MUSE Stage-1 model from a state_dict on disk.
    Auto-detects v1 deterministic vs v2 VAE based on stage1_outdir contents."""
    state_path = os.path.join(stage1_outdir, "static_model.pth")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"missing Stage-1 state at {state_path}")
    state = torch.load(state_path, map_location=device, weights_only=False)
    is_vae = _is_vae_stage1(stage1_outdir)
    modality_dims, joint, latent, hidden = _infer_dims(state, is_vae=is_vae)
    if is_vae:
        _, make_model = _import_muse_vae()
        print(f"[cache] detected VAE Stage 1 (Option II+)")
    else:
        _, make_model = _import_muse_v1()
    model = make_model(modality_dims, latent_dim_per_modality=latent,
                       joint_dim=joint, hidden_dim=hidden, dropout=0.0)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), modality_dims


# ───────────────────────────────────────────────────────────────────────────
# Cluster + kNN
# ───────────────────────────────────────────────────────────────────────────

def cluster_z_leiden(Z: np.ndarray, k: int = 15, resolution: float = 1.0,
                     seed: int = 0) -> np.ndarray:
    """Cluster z-space with Leiden on a mutual cosine kNN graph.
    Returns int array of cluster ids in [0, K)."""
    try:
        from .pseudo_labels import cluster_leiden
    except Exception:
        cluster_leiden = None
    if cluster_leiden is not None:
        lab = cluster_leiden(Z, k=k, resolution=resolution, seed=seed)
        if lab is not None:
            return np.asarray(lab, dtype=np.int64)
    # Fallback: KMeans
    from sklearn.cluster import KMeans
    K = max(20, int(np.sqrt(Z.shape[0])))
    km = KMeans(n_clusters=K, random_state=seed, n_init=4).fit(Z)
    return km.labels_.astype(np.int64)


def cluster_proba_from_id(cluster_id: np.ndarray, smooth: float = 0.05) -> np.ndarray:
    """One-hot cluster ids with a tiny uniform smoothing so cos() is well-defined
    even when two proteins share no nonzero coordinate (rare but possible)."""
    K = int(cluster_id.max()) + 1
    N = len(cluster_id)
    P = np.full((N, K), smooth / K, dtype=np.float32)
    P[np.arange(N), cluster_id] += (1.0 - smooth)
    # row-normalise so it's still a distribution
    P /= P.sum(axis=1, keepdims=True)
    return P


def build_knn_with_cluster_weights(
    Z: np.ndarray,
    conf: np.ndarray,
    cluster_proba: np.ndarray,
    k: int = 20,
    cluster_compat_temp: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (knn_idx[N, k], knn_w[N, k]).

    Simplified edge-weight design (post-audit, see README):

        w_ij  =  conf_j  ·  cluster_compat(i, j) ** (1/τ)

    Two factors, both genuinely doing work after row-normalisation:
      * conf_j               source neighbour's anchor reliability (the
                             centre's own conf_i is *cancelled* by the
                             row-normalisation step, so writing it as a
                             factor was misleading and is dropped).
      * cluster_compat(i,j)  soft Leiden cluster gating — same-cluster pairs
                             ≈ 1, different-cluster pairs ≈ ε.  Sharper as
                             τ → 0, softer as τ → ∞.

    Two factors that were dropped from earlier versions and why:
      * conf_i               redundant (cancels under row-norm; see audit).
      * cos(z_i, z_j)        redundant inside top-k cosine kNN — all
                             retained edges sit in a narrow high-cosine band,
                             so this barely discriminates. The attention layer
                             at training time can recover any closeness-within-
                             top-k ordering with much more flexibility.
    """
    from sklearn.neighbors import NearestNeighbors
    N = Z.shape[0]
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Zn)
    _, idx = nn.kneighbors(Zn)                  # (N, k+1); drop self at col 0
    knn_idx = idx[:, 1:].astype(np.int64)

    # cluster compatibility via cosine on (smoothed) one-hot cluster_proba
    Pn = cluster_proba / np.maximum(np.linalg.norm(cluster_proba, axis=1, keepdims=True), 1e-9)
    knn_w = np.zeros((N, k), dtype=np.float32)
    inv_tau = 1.0 / max(cluster_compat_temp, 1e-3)
    for i in range(N):
        nbrs   = knn_idx[i]
        compat = Pn[i] @ Pn[nbrs].T              # (k,)  ∈ [0, 1]
        knn_w[i] = conf[nbrs] * np.clip(compat, 0.0, None) ** inv_tau

    # Row-normalise so each row's neighbour weighting sums to 1.
    s = knn_w.sum(axis=1, keepdims=True)
    knn_w = np.where(s > 0, knn_w / np.maximum(s, 1e-9), knn_w)
    return knn_idx, knn_w.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
# Anchor confidence & EPIC reliability
# ───────────────────────────────────────────────────────────────────────────

def per_protein_anchor_confidence(model: "nn.Module",
                                  z: torch.Tensor,
                                  x_per_modality: Dict[str, torch.Tensor],
                                  mask_per_modality: Dict[str, torch.Tensor],
                                  ) -> np.ndarray:
    """conf_i = harmonic mean over PRESENT modalities of  1 / (1 + r̂_m_i),
    where r̂_m_i  =  ‖x_m_i  − D_m(z_i)‖² / d_m  normalised to mean 1
    on present proteins.

    Intuition: r_m_i is how well z explains modality m for protein i. Low
    residual across all modalities → joint z captures every signal → high
    confidence. High residual on most modalities → satellite (z dominated by
    one modality, misrepresents the others) → low confidence. Proteins
    missing a modality contribute 0 to that modality's term (n_present−1).

    NOTE: replaces the earlier `harmonic_mean_per_protein_cos`, which was
    based on cos(h_m, z) — h_m and z live in different dimensional spaces
    (d_h ≠ d_z), so that cosine never made sense and crashed on broadcast.
    """
    N = z.shape[0]
    rec_recip  = np.zeros(N, dtype=np.float64)         # Σ (1+r̂_m) over present m
    n_present  = np.zeros(N, dtype=np.float64)
    with torch.no_grad():
        for m, X in x_per_modality.items():
            if m not in model.decoders:
                continue                                # safety: skip modalities the model doesn't have
            mask_np = (mask_per_modality[m].cpu().numpy() > 0.5)
            x_hat = model.decode(z, m)                  # (N, d_m)
            r = (X - x_hat).pow(2).mean(dim=-1).cpu().numpy()
            # Normalise residual to mean 1 on present proteins so different
            # modalities sit on comparable scales.
            denom = max(r[mask_np].mean(), 1e-9) if mask_np.any() else 1.0
            r_norm = r / denom
            c = 1.0 / (1.0 + r_norm)                    # in (0, 1]
            c = np.clip(c, 1e-3, 1.0)                   # numerical safety
            present = mask_np.astype(np.float64)
            rec_recip += present * (1.0 / c)
            n_present += present
    conf = np.where(n_present > 0, n_present / np.maximum(rec_recip, 1e-9), 0.0)
    return conf.astype(np.float32)


def per_protein_epic_sigma2(model: "nn.Module",
                            X_epic: torch.Tensor,
                            mask_epic: torch.Tensor,
                            Z: torch.Tensor,
                            epic_name: str,
                            ) -> np.ndarray:
    """σ²_EPIC_i ∝ ‖x_epic_i − D_epic(z_i)‖² / d_epic   (clipped & normalised to mean 1).
    Proteins missing EPIC get σ² = large (→ predictions only rely on neighbours)."""
    with torch.no_grad():
        x_hat = model.decode(Z, epic_name)
        sq = (X_epic - x_hat).pow(2).mean(dim=-1).cpu().numpy()
    mask = mask_epic.cpu().numpy() > 0.5
    s2 = sq.copy()
    if mask.sum() > 0:
        s2[mask] = s2[mask] / max(s2[mask].mean(), 1e-9)           # mean 1 on present
    s2[~mask] = 10.0                                                # heavy downweight if EPIC missing
    s2 = np.clip(s2, 0.05, 20.0)                                    # safety
    return s2.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
# Manifest helpers for control / treated EPIC matrices
# ───────────────────────────────────────────────────────────────────────────

def _canonical_condition(cond_raw: str, path_hint: str = "") -> str:
    """Map a raw condition string (and optional path hint) to a canonical key.

    - 'untreated', 'control', 'ctrl', '' → 'control'
    - anything containing 'cisplatin' → 'cisplatin'
    - anything containing 'vorinostat' → 'vorinostat'
    - 'negativectrl', 'negative_ctrl' → 'negative_ctrl'   (kept separate from
      the untreated baseline because it's the treated-state vehicle ctrl)
    - else the raw string lowercased.
    """
    c = (cond_raw or "").strip().lower()
    h = (path_hint or "").lower()
    blob = c + " " + h
    if "cisplatin" in blob:    return "cisplatin"
    if "vorinostat" in blob:   return "vorinostat"
    if c in ("", "untreated", "control", "ctrl"):  return "control"
    if "negativectrl" in c or "negative_ctrl" in c: return "negative_ctrl"
    return c


def _average_replicates(streams_list: List[Tuple[List[str], np.ndarray]]
                        ) -> Tuple[List[str], np.ndarray]:
    """Union proteins across replicates; mean features over those that have it."""
    if len(streams_list) == 1:
        return streams_list[0]
    prots = sorted(set().union(*[set(s[0]) for s in streams_list]))
    d = streams_list[0][1].shape[1]
    accum = np.zeros((len(prots), d), dtype=np.float32)
    cnt = np.zeros(len(prots), dtype=np.float32)
    for p, X in streams_list:
        idx = {q: i for i, q in enumerate(p)}
        for k, q in enumerate(prots):
            if q in idx:
                accum[k] += X[idx[q]]; cnt[k] += 1
    keep = cnt > 0
    return ([p for p, k in zip(prots, keep) if k],
            (accum / np.maximum(cnt, 1)[:, None])[keep])


def load_epic_per_condition(manifest_path: str, epic_name: str,
                            ) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Read manifest.json and return {condition_label: (proteins, X)} for the
    EPIC modality. Robust to two common layouts:

      Layout A (one entry per condition):
        [{"modality": "EPIC", "condition": "untreated",  "path": "U.tsv"},
         {"modality": "EPIC", "condition": "cisplatin",  "path": "C.tsv"},
         {"modality": "EPIC", "condition": "vorinostat", "path": "V.tsv"}]

      Layout B (one entry with separate path fields per condition):
        [{"modality": "EPIC", "condition": "untreated", "path": "U.tsv",
          "treated_path": "T.tsv", "treatment": "cisplatin"}]
        OR
        [{"modality": "EPIC",
          "paths": {"untreated": "U.tsv", "cisplatin": "C.tsv", "vorinostat": "V.tsv"}}]

    Modality matching is case-insensitive.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    epic_lower = (epic_name or "").lower()

    epic_entries = [e for e in entries if (e.get("modality") or "").lower() == epic_lower]
    if not epic_entries:
        all_mods = sorted(set((e.get("modality") or "?") for e in entries))
        raise RuntimeError(
            f"No EPIC entries (modality matching {epic_name!r}) in {manifest_path}. "
            f"Manifest has modalities: {all_mods}")

    # Diagnostic: report what layout the manifest uses
    cond_counts: Dict[str, int] = {}
    have_treated_path = 0
    have_paths_dict   = 0
    for e in epic_entries:
        c = (e.get("condition") or "").strip().lower()
        cond_counts[c] = cond_counts.get(c, 0) + 1
        if e.get("treated_path"): have_treated_path += 1
        if isinstance(e.get("paths"), dict): have_paths_dict += 1
    print(f"[io] EPIC: {len(epic_entries)} entries; condition distribution: {cond_counts}")
    if have_treated_path:
        print(f"[io] EPIC: {have_treated_path} entries have treated_path field")
    if have_paths_dict:
        print(f"[io] EPIC: {have_paths_dict} entries have paths-dict layout")

    streams: Dict[str, List[Tuple[List[str], np.ndarray]]] = {}

    def _ingest(cond_key: str, path: str):
        if not path or not os.path.isfile(path):
            print(f"[io]   skip: cond={cond_key!r} path={path!r} (missing)")
            return
        prots, X = _read_tsv(path)
        streams.setdefault(cond_key, []).append((prots, X))
        print(f"[io]   ingested cond={cond_key!r}  n={len(prots)}  d={X.shape[1]}  from {path}")

    for e in epic_entries:
        cond_raw = (e.get("condition") or "").strip().lower()

        # Layout C: paths-dict
        if isinstance(e.get("paths"), dict):
            for k, p in e["paths"].items():
                _ingest(_canonical_condition(k, p), p)
            continue

        # Standard entry path
        path = e.get("path")
        if path:
            ck = _canonical_condition(cond_raw, path)
            _ingest(ck, path)

        # Treated path (layout B) — figure out which treatment it represents.
        # Priority for the treatment label:
        #   1. explicit `treatment` / `drug` / `condition_treated` field
        #   2. the entry's own `condition` field (in this manifest, treated
        #      entries have condition='cisplatin' or 'vorinostat')
        #   3. path substring match (last-resort)
        treated_path = e.get("treated_path")
        if treated_path:
            treatment = (e.get("treatment") or e.get("drug")
                         or e.get("condition_treated") or cond_raw or "")
            tk = _canonical_condition(treatment, treated_path)
            # Treated_path should NEVER end up tagged as 'control' (would
            # collide with the untreated path). If we somehow got 'control',
            # label it 'treated_unknown' so it's still loaded but the user
            # sees the mismatch.
            if tk == "control":
                tk = "treated_unknown"
            _ingest(tk, treated_path)

    if not streams:
        raise RuntimeError(
            f"Found {len(epic_entries)} EPIC entries but none yielded a "
            f"readable path. Check the 'path' / 'treated_path' / 'paths' "
            f"fields. Condition distribution seen: {cond_counts}")

    out: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for cond, sl in streams.items():
        out[cond] = _average_replicates(sl)
    print(f"[io] EPIC streams resolved: { {k: out[k][1].shape for k in out} }")
    return out


# ───────────────────────────────────────────────────────────────────────────
# Stage1Cache
# ───────────────────────────────────────────────────────────────────────────

class Stage1Cache:
    """All Stage-1 artefacts the v3 adapter needs.

    Built in-memory from a Stage 1 outdir + manifest — no persistence. Use
    `Stage1Cache.from_stage1_dir(stage1_outdir, manifest_path, ...)` as the
    primary entry point. The constructor is for advanced/testing use.
    """

    def __init__(self,
                 proteins: List[str],
                 modality_dims: Dict[str, int],
                 epic_name: str,
                 stage1_outdir: str,
                 z: torch.Tensor,
                 h_per_modality: Dict[str, torch.Tensor],
                 mask_per_modality: Dict[str, torch.Tensor],
                 cluster_id: torch.Tensor,
                 cluster_proba: torch.Tensor,
                 sigma2_epic: torch.Tensor,
                 conf: torch.Tensor,
                 knn_idx: torch.Tensor,
                 knn_w: torch.Tensor,
                 is_vae: bool = False,
                 sigma2_enc_per_modality: Optional[Dict[str, torch.Tensor]] = None,
                 sigma2_dec_per_modality: Optional[Dict[str, torch.Tensor]] = None,
                 z_emp_mu: Optional[torch.Tensor] = None,
                 z_emp_sigma_inv: Optional[torch.Tensor] = None):
        self.proteins = list(proteins)
        self.modality_dims = dict(modality_dims)
        self.epic_name = epic_name
        self.stage1_outdir = stage1_outdir
        self.z = z
        self.h_per_modality = h_per_modality
        self.mask_per_modality = mask_per_modality
        self.cluster_id = cluster_id
        self.cluster_proba = cluster_proba
        self.sigma2_epic = sigma2_epic
        self.conf = conf
        self.knn_idx = knn_idx
        self.knn_w = knn_w
        # VAE-specific (None when v1 deterministic Stage 1)
        self.is_vae = bool(is_vae)
        self.sigma2_enc_per_modality = sigma2_enc_per_modality or {}
        self.sigma2_dec_per_modality = sigma2_dec_per_modality or {}
        self.z_emp_mu = z_emp_mu                              # (d_z,)
        self.z_emp_sigma_inv = z_emp_sigma_inv                # (d_z, d_z), precomputed Σ_emp^-1
        # Filled by attach_modules() (or from_stage1_dir, which calls it inline)
        self._muse_model = None
        self.E_EPIC = None
        self.D_epic = None
        self.D_seq = None
        self.D_image = None

    # ---------- properties ----------
    @property
    def N(self) -> int: return len(self.proteins)
    @property
    def d_z(self) -> int: return int(self.z.shape[1])
    @property
    def d_h(self) -> int: return int(next(iter(self.h_per_modality.values())).shape[1])
    @property
    def K(self) -> int: return int(self.cluster_proba.shape[1])
    @property
    def k_neighbours(self) -> int: return int(self.knn_idx.shape[1])

    # ---------- module attachment ----------
    def _attach_modules_from_loaded(self, model,
                                    seq_name: Optional[str] = None,
                                    image_name: Optional[str] = None):
        """Attach encoders/decoders from an already-loaded MUSE model."""
        self._muse_model = model
        encoder_keys = list(model.encoders.keys())
        canonical_epic = _resolve_modality_name(self.epic_name, encoder_keys,
                                                fuzzy=["epic", "secms", "ms"])
        if canonical_epic is None:
            raise KeyError(
                f"EPIC modality {self.epic_name!r} not in model encoders "
                f"({encoder_keys}). Override with seq/image hints if needed.")
        if canonical_epic != self.epic_name:
            print(f"[cache] epic_name {self.epic_name!r} → resolved to canonical "
                  f"{canonical_epic!r}")
            self.epic_name = canonical_epic
        self.E_EPIC = model.encoders[canonical_epic]
        self.D_epic = model.decoders[canonical_epic]
        decoder_keys = list(model.decoders.keys())
        sn = _resolve_modality_name(seq_name, decoder_keys,
                                    fuzzy=["seq", "sequence", "esm", "prot_t5", "protein"])
        if sn: self.D_seq = model.decoders[sn]
        en = _resolve_modality_name(image_name, decoder_keys,
                                    fuzzy=["image", "hpa", "img"])
        if en: self.D_image = model.decoders[en]
        print(f"[cache] modules attached: epic={canonical_epic!r}  "
              f"seq={sn!r}  image={en!r}")
        return self

    def attach_modules(self, device: str = "cpu",
                       seq_name: Optional[str] = None,
                       image_name: Optional[str] = None):
        """Load the MUSE model from `self.stage1_outdir` and attach modules."""
        model, _ = load_muse_model(self.stage1_outdir, device=device)
        return self._attach_modules_from_loaded(model, seq_name=seq_name,
                                                image_name=image_name)

    # ---------- primary entry point ----------
    @classmethod
    def from_stage1_dir(cls,
                        stage1_outdir: str,
                        manifest_path: str,
                        *,
                        epic_name: str = "epic",
                        k: int = 20,
                        leiden_resolution: float = 1.0,
                        leiden_knn_k: int = 15,
                        cluster_compat_temp: float = 1.0,
                        device: str = "cpu",
                        seed: int = 0,
                        seq_name: Optional[str] = None,
                        image_name: Optional[str] = None,
                        sigma2_epic_path: Optional[str] = None) -> "Stage1Cache":
        """Build a Stage1Cache in memory from a Stage 1 outdir + manifest.

        Pipeline (see comments below):
          1. Load MUSE state_dict and instantiate the model.
          2. Resolve `epic_name` against the model's encoder keys.
          3. Read `static_latent.tsv` for the protein universe + z.
          4. Read `per_modality/<m>.tsv` for per-modality h_m.
          5. Read raw modality input matrices (via two_stage.stage1.train helpers).
          6. Compute σ²_EPIC, anchor confidence.
          7. Run Leiden on z-space; build cluster_compat kNN graph.
          8. Attach encoders/decoders to the cache and return.

        Returns a ready-to-use Stage1Cache with modules attached — no .save()
        step, no disk persistence.
        """
        print(f"[cache] Stage-1 outdir: {stage1_outdir}")
        print(f"[cache] manifest:       {manifest_path}")
        print(f"[cache] EPIC modality:  {epic_name}")
        np.random.seed(seed); torch.manual_seed(seed)

        # ---- 1) MUSE model ----
        model, modality_dims = load_muse_model(stage1_outdir, device=device)
        print(f"[cache] model: modalities={list(modality_dims)}  d_z={model.joint_dim}")

        # ---- 2) canonical EPIC modality name ----
        canonical_epic = _resolve_modality_name(epic_name, list(modality_dims),
                                                 fuzzy=["epic", "secms", "ms"])
        if canonical_epic is None:
            raise KeyError(
                f"epic_name={epic_name!r} doesn't match any Stage-1 modality "
                f"({list(modality_dims)}). Pass --epic_name explicitly.")
        if canonical_epic != epic_name:
            print(f"[cache] epic_name {epic_name!r} → canonical {canonical_epic!r}")
        epic_name = canonical_epic

        # ---- 3) universe + z ----
        universe, Z_np = _read_tsv(os.path.join(stage1_outdir, "static_latent.tsv"))
        print(f"[cache] universe: {len(universe)} proteins  d_z={Z_np.shape[1]}")

        # ---- 4) per-modality h_m ----
        h_per_modality: Dict[str, np.ndarray] = {}
        mask_per_modality: Dict[str, np.ndarray] = {}
        for m in modality_dims:
            path = os.path.join(stage1_outdir, "per_modality", f"{m}.tsv")
            if not os.path.isfile(path):
                print(f"[cache]   WARN: missing per_modality/{m}.tsv — treated as absent")
                d_h = (next(iter(h_per_modality.values())).shape[1]
                       if h_per_modality else model.latent_dim)
                h_per_modality[m] = np.zeros((len(universe), d_h), dtype=np.float32)
                mask_per_modality[m] = np.zeros(len(universe), dtype=np.float32)
                continue
            prots_m, Hm = _read_tsv(path)
            Hm_a, _ = _align(universe, prots_m, Hm)
            h_per_modality[m] = Hm_a
            mask_per_modality[m] = (np.linalg.norm(Hm_a, axis=1) > 1e-9).astype(np.float32)

        # ---- 5) raw modality inputs (for σ²_EPIC and conf) ----
        try:
            from .train import load_modality_matrices, assemble_tensors
        except Exception as exc:
            raise RuntimeError("two_stage.stage1 package required to build the cache "
                               "(needed for per-modality input loading)") from exc
        print("[cache] loading raw modality matrices...")
        mods_raw = load_modality_matrices(manifest_path, untreated_only=True)
        Xs_raw, Ms_raw = assemble_tensors(universe, mods_raw)
        x_t_per_modality:   Dict[str, torch.Tensor] = {}
        mask_t_per_modality: Dict[str, torch.Tensor] = {}
        for m in modality_dims:
            if m not in Xs_raw:
                print(f"[cache]   WARN: modality {m!r} in model but not in manifest — "
                      "treated as absent for conf computation.")
                x_t_per_modality[m]    = torch.zeros(len(universe), modality_dims[m], device=device)
                mask_t_per_modality[m] = torch.zeros(len(universe), device=device)
            else:
                x_t_per_modality[m]    = torch.from_numpy(Xs_raw[m]).to(device)
                mask_t_per_modality[m] = torch.from_numpy(Ms_raw[m]).to(device)

        # ---- 6) σ²_EPIC, anchor confidence, VAE-specific artefacts ----
        z_t = torch.from_numpy(Z_np).to(device)
        is_vae = _is_vae_stage1(stage1_outdir)

        # VAE-specific extras — encoder σ²_m and decoder σ²_x_m per protein,
        # plus the empirical Gaussian on z. All loaded from Stage 1 outputs.
        sigma2_enc_per_modality: Dict[str, torch.Tensor] = {}
        sigma2_dec_per_modality: Dict[str, torch.Tensor] = {}
        z_emp_mu_t = None
        z_emp_sigma_inv_t = None
        if is_vae:
            print("[cache] loading VAE per-modality σ² and empirical z prior...")
            for m in modality_dims:
                # σ²_m (encoder uncertainty)
                p_enc = os.path.join(stage1_outdir, "per_modality_sigma", f"{m}.tsv")
                if os.path.isfile(p_enc):
                    prots_m, arr = _read_tsv(p_enc)
                    arr_a, _ = _align(universe, prots_m, arr)
                    sigma2_enc_per_modality[m] = torch.from_numpy(arr_a)
                # σ²_x_m (decoder uncertainty) — per-protein per-input-dim
                p_dec = os.path.join(stage1_outdir, "decoder_sigma", f"{m}.tsv")
                if os.path.isfile(p_dec):
                    prots_d, arr = _read_tsv(p_dec)
                    arr_a, _ = _align(universe, prots_d, arr)
                    sigma2_dec_per_modality[m] = torch.from_numpy(arr_a)
            # Empirical mean + covariance for the L_prior Mahalanobis term
            p_mu = os.path.join(stage1_outdir, "z_emp_mu.tsv")
            p_sg = os.path.join(stage1_outdir, "z_emp_sigma.tsv")
            if os.path.isfile(p_mu):
                _, mu_arr = _read_tsv(p_mu)
                z_emp_mu_t = torch.from_numpy(mu_arr[0])  # single row
            if os.path.isfile(p_sg):
                _, sg_arr = _read_tsv(p_sg)
                # Σ_emp^-1; add small ridge for numerical stability
                ridge = 1e-3 * np.eye(sg_arr.shape[0], dtype=np.float32)
                try:
                    inv = np.linalg.inv(sg_arr + ridge).astype(np.float32)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(sg_arr + ridge).astype(np.float32)
                z_emp_sigma_inv_t = torch.from_numpy(inv)
            print(f"[cache] VAE artefacts: encoder σ² for "
                  f"{list(sigma2_enc_per_modality)};  decoder σ² for "
                  f"{list(sigma2_dec_per_modality)};  empirical prior "
                  f"{'yes' if z_emp_mu_t is not None else 'NO'}")

        # σ²_EPIC: VAE → mean of decoder σ²_x_EPIC over the EPIC feature dim
        # per protein. v1 → reconstruction-residual heuristic (existing code).
        if is_vae and epic_name in sigma2_dec_per_modality:
            s2_dec_epic = sigma2_dec_per_modality[epic_name].cpu().numpy()
            sigma2_epic = s2_dec_epic.mean(axis=1)
            # Normalise to mean 1 over proteins with EPIC present (so the
            # Bayesian sigma2_raw_scale knob still has a stable reference).
            if epic_name in Ms_raw:
                m_present = Ms_raw[epic_name] > 0.5
                denom = max(sigma2_epic[m_present].mean(), 1e-9) if m_present.any() else 1.0
                sigma2_epic = (sigma2_epic / denom).astype(np.float32)
            sigma2_epic = np.clip(sigma2_epic, 0.05, 20.0)
            print(f"[cache] σ²_EPIC from VAE decoder: mean={sigma2_epic.mean():.3f}  "
                  f"q10={np.quantile(sigma2_epic, 0.1):.3f}  "
                  f"q90={np.quantile(sigma2_epic, 0.9):.3f}")
        else:
            if epic_name not in Xs_raw:
                raise RuntimeError(
                    f"EPIC modality {epic_name!r} present in model but not "
                    f"loaded by two_stage.stage1.train.load_modality_matrices.")
            xe_t = torch.from_numpy(Xs_raw[epic_name]).to(device)
            me_t = torch.from_numpy(Ms_raw[epic_name]).to(device)
            sigma2_epic = per_protein_epic_sigma2(model, xe_t, me_t, z_t,
                                                  epic_name=epic_name)
            print(f"[cache] σ²_EPIC heuristic (v1): mean={sigma2_epic.mean():.3f}  "
                  f"q10={np.quantile(sigma2_epic, 0.1):.3f}  "
                  f"q90={np.quantile(sigma2_epic, 0.9):.3f}")

        # --- Empirical σ²_EPIC override (per-protein replicate noise) ---------
        # The heuristic σ²_EPIC (Stage-1 reconstruction residual) is normalised to
        # ~1 and gives little per-protein structure, so the Bayesian combiner ends
        # up trusting the neighbour head almost everywhere (w_pred≈0.99) and the
        # observed differential is discarded. Injecting the *empirical replicate*
        # σ² (variance of a protein's embedding across replicates) gives the
        # combiner real per-protein reliability: clean proteins (low σ²) keep their
        # observed delta; noisy proteins lean on neighbours. Values are used at
        # their natural scale (caller should set --sigma2_raw_floor low enough not
        # to clip the clean end). Missing proteins → high σ² (lean on neighbours).
        if sigma2_epic_path is not None:
            raw_map: Dict[str, float] = {}
            with open(sigma2_epic_path) as fh:
                next(fh, None)  # header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    try:
                        raw_map[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
            vals = np.array([raw_map.get(p, np.nan) for p in universe], dtype=np.float64)
            present = np.isfinite(vals) & (vals > 0)
            if present.sum() < 10:
                raise RuntimeError(
                    f"--sigma2_epic_path {sigma2_epic_path} matched only "
                    f"{int(present.sum())} of {len(universe)} cache proteins "
                    f"(check the ID namespace / gene symbols).")
            fill = float(np.quantile(vals[present], 0.9))  # missing → noisy
            vals[~present] = fill
            sigma2_epic = np.clip(vals, 1e-3, 20.0).astype(np.float32)
            print(f"[cache] σ²_EPIC OVERRIDE (empirical replicate) from "
                  f"{sigma2_epic_path}: matched {int(present.sum())}/{len(universe)} "
                  f"proteins; mean={sigma2_epic.mean():.3f} "
                  f"q10={np.quantile(sigma2_epic,0.1):.3f} "
                  f"median={np.median(sigma2_epic):.3f} "
                  f"q90={np.quantile(sigma2_epic,0.9):.3f}")

        # Anchor confidence: VAE → harmonic mean over modalities of
        # 1 / (1 + mean(σ²_m_i / σ²_m_mean_present)), where σ²_m_i is the
        # encoder's per-protein variance. v1 → decoder-residual heuristic.
        if is_vae and sigma2_enc_per_modality:
            N = len(universe)
            rec_recip = np.zeros(N, dtype=np.float64)
            n_present = np.zeros(N, dtype=np.float64)
            for m, s2_t in sigma2_enc_per_modality.items():
                mask = (Ms_raw.get(m, np.zeros(N, dtype=np.float32)) > 0.5)
                s2 = s2_t.cpu().numpy().mean(axis=1)         # avg over latent dim
                if mask.any():
                    denom = max(s2[mask].mean(), 1e-9)
                    r_norm = s2 / denom
                else:
                    r_norm = np.zeros_like(s2)
                c = 1.0 / (1.0 + r_norm)
                c = np.clip(c, 1e-3, 1.0)
                rec_recip += mask.astype(np.float64) * (1.0 / c)
                n_present += mask.astype(np.float64)
            conf = np.where(n_present > 0,
                            n_present / np.maximum(rec_recip, 1e-9),
                            0.0).astype(np.float32)
            print(f"[cache] conf from VAE encoder σ²: mean={conf.mean():.3f}  "
                  f"q10={np.quantile(conf, 0.1):.3f}  q90={np.quantile(conf, 0.9):.3f}")
        else:
            conf = per_protein_anchor_confidence(model, z_t,
                                                  x_t_per_modality,
                                                  mask_t_per_modality)
            print(f"[cache] conf heuristic (v1): mean={conf.mean():.3f}  "
                  f"q10={np.quantile(conf, 0.1):.3f}  q90={np.quantile(conf, 0.9):.3f}")

        # ---- 7) Leiden + cluster-aware kNN ----
        cluster_id = cluster_z_leiden(Z_np, k=leiden_knn_k,
                                      resolution=leiden_resolution, seed=seed)
        K = int(cluster_id.max() + 1)
        print(f"[cache] Leiden clusters: K={K}  (resolution={leiden_resolution})")
        cluster_proba = cluster_proba_from_id(cluster_id, smooth=0.05)

        knn_idx, knn_w = build_knn_with_cluster_weights(
            Z_np, conf, cluster_proba, k=k, cluster_compat_temp=cluster_compat_temp)
        same_cl = (cluster_id[:, None] == cluster_id[knn_idx]).astype(np.float32)
        frac_same = float(same_cl.mean())
        mass_same = float((knn_w * same_cl).sum(axis=1).mean())
        print(f"[cache] kNN: k={k}  rows sum to 1; same-cluster edge fraction={frac_same:.3f}  "
              f"avg row-mass on same-cluster edges={mass_same:.3f}")

        # ---- 8) construct + attach modules from the already-loaded model ----
        cache = cls(
            proteins=universe,
            modality_dims=modality_dims,
            epic_name=epic_name,
            stage1_outdir=stage1_outdir,
            z=torch.from_numpy(Z_np),
            h_per_modality={m: torch.from_numpy(H) for m, H in h_per_modality.items()},
            mask_per_modality={m: torch.from_numpy(M) for m, M in mask_per_modality.items()},
            cluster_id=torch.from_numpy(cluster_id),
            cluster_proba=torch.from_numpy(cluster_proba),
            sigma2_epic=torch.from_numpy(sigma2_epic),
            conf=torch.from_numpy(conf),
            knn_idx=torch.from_numpy(knn_idx),
            knn_w=torch.from_numpy(knn_w),
            is_vae=is_vae,
            sigma2_enc_per_modality=sigma2_enc_per_modality,
            sigma2_dec_per_modality=sigma2_dec_per_modality,
            z_emp_mu=z_emp_mu_t,
            z_emp_sigma_inv=z_emp_sigma_inv_t,
        )
        cache._attach_modules_from_loaded(model, seq_name=seq_name, image_name=image_name)
        return cache
