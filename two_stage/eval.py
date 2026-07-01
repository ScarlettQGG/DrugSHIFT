"""Stage 2 v3 biological evaluation.

Reads a trained adapter's inference outputs (z_treat, delta_final, coherence,
sigma2_pred) and its Stage 1 cache, runs four biology-facing evaluations:

  (1) CORUM complex remodeling
        For each known complex, compute mean pairwise cosine distance among
        members in z_ref vs z_treat. Δdistance > 0 → dissociation;
        Δdistance < 0 → tightening / formation.  Writes a ranked TSV.

  (2) Leiden cluster transition map
        Re-cluster z_treat at the same resolution as Stage 1; build a
        confusion matrix cluster_ref × cluster_treat. Off-diagonal mass
        quantifies subunit-swap-like reassignments.

  (3) HPA localization shift
        For each protein with HPA support, run frozen D_image on z_ref and
        z_treat, take top-class before/after + KL divergence between the
        predicted distributions. Writes per-protein predicted compartment
        changes — a direct readout of translocation.

  (4) Coherence × magnitude flags
        Per-protein category, from the cross of ‖δ_final‖ and the
        coherence cos(δ_raw_proj, δ̂):
            ‖δ‖ high, coh high  →  high-confidence remodelling
            ‖δ‖ high, coh low   →  inspect: solo move or noise
            ‖δ‖ low,  coh high  →  small but supported move
            ‖δ‖ low,  coh low   →  no remodelling

Run:
    python -m two_stage.stage2.eval \\
        --adapter_dir   output/stage2/adapter_cisplatin \\
        --inference_dir output/stage2/inference_cisplatin \\
        --corum         corum_humanComplexes0125.txt \\
        --outdir        output/stage2/eval_cisplatin
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 9
    _MPL_OK = True
except Exception:
    _MPL_OK = False

from .cache import Stage1Cache, cluster_z_leiden


# ───────────────────────────────────────────────────────────────────────────
# IO helpers
# ───────────────────────────────────────────────────────────────────────────

def _read_tsv(path: str) -> Tuple[List[str], np.ndarray]:
    """First column = protein name, rest = floats. Same format as Stage 1."""
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


def _write_tsv(path: str, header: List[str], rows: List[list]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _read_corum(path: str) -> Dict[str, List[str]]:
    """Minimal CORUM loader matching the user's format:
    tab-separated, complex_name column, subunits_gene_name column (';' joined)."""
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    low = [h.lower() for h in header]

    def _find(cands):
        for c in cands:
            for i, h in enumerate(low):
                if c in h:
                    return i
        return None

    name_col = _find(["complex_name", "complexname", "comprel_name",
                       "comprel", "complex"]) or 0
    mem_col = _find(["subunits_gene_name", "subunits(gene name)",
                       "subunits(gene", "gene_name", "subunits"])
    if mem_col is None:
        # last-resort: any column with ';' in it
        for r in rows[:50]:
            for i, v in enumerate(r):
                if v and ";" in v:
                    mem_col = i; break
            if mem_col is not None: break
    if mem_col is None:
        raise RuntimeError(f"could not find subunits column in {path}")

    out: Dict[str, List[str]] = {}
    for i, r in enumerate(rows):
        if mem_col >= len(r):
            continue
        members_raw = r[mem_col]
        if not members_raw:
            continue
        members = [m.strip() for m in members_raw.replace(",", ";").split(";") if m.strip()]
        members = [m for m in members if m.lower() != "none" and not m.startswith("(")]
        if len(members) < 2:
            continue
        name = r[name_col] if name_col < len(r) and r[name_col] else f"complex_{i}"
        out[name] = members
    print(f"[eval] CORUM: {path}  -> {len(out)} complexes (>=2 members)")
    return out


# ───────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ───────────────────────────────────────────────────────────────────────────

def _row_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def _mean_pairwise_cosine_dist(X: np.ndarray) -> float:
    """Mean cosine distance over all unordered pairs in X (rows = proteins)."""
    if X.shape[0] < 2:
        return float("nan")
    Xn = _row_normalize(X)
    sims = Xn @ Xn.T                      # (n, n)
    n = Xn.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sims[iu].mean())


# ───────────────────────────────────────────────────────────────────────────
# (1) CORUM complex remodeling
# ───────────────────────────────────────────────────────────────────────────

def evaluate_corum_remodeling(
    z_ref: np.ndarray, z_treat: np.ndarray,
    proteins: List[str], corum: Dict[str, List[str]],
    min_members_overlap: int = 3,
) -> List[dict]:
    p_idx = {p: i for i, p in enumerate(proteins)}
    out = []
    for name, members in corum.items():
        idx = [p_idx[m] for m in members if m in p_idx]
        if len(idx) < min_members_overlap:
            continue
        idx = np.asarray(idx)
        d_ref   = _mean_pairwise_cosine_dist(z_ref[idx])
        d_treat = _mean_pairwise_cosine_dist(z_treat[idx])
        if not (np.isfinite(d_ref) and np.isfinite(d_treat)):
            continue
        out.append({
            "complex":           name,
            "n_members_in_embed": int(len(idx)),
            "mean_dist_ref":      d_ref,
            "mean_dist_treat":    d_treat,
            "delta_dist":         d_treat - d_ref,        # > 0 = dissociation
            "members":            ";".join(sorted(p for p in members if p in p_idx)),
        })
    out.sort(key=lambda r: -r["delta_dist"])      # most-dissociating first
    return out


def plot_corum_top_bottom(rows: List[dict], out_stem: str, top_n: int = 25):
    if not _MPL_OK or not rows:
        return
    top    = rows[:top_n]
    bottom = rows[-top_n:]
    fig, ax = plt.subplots(1, 2, figsize=(13, max(4, 0.22 * top_n + 2)))
    for a, sub, title, color in [
        (ax[0], top,    f"Top {top_n} dissociating (Δd > 0)", "#d95f02"),
        (ax[1], bottom, f"Top {top_n} tightening (Δd < 0)",   "#1b9e77"),
    ]:
        names = [r["complex"][:40] for r in sub][::-1]
        d     = [r["delta_dist"] for r in sub][::-1]
        a.barh(range(len(names)), d, color=color, edgecolor="black", linewidth=0.3)
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=7)
        a.set_xlabel("Δ mean pairwise cosine dist  (z_treat − z_ref)")
        a.set_title(title, loc="left", fontsize=10)
        a.axvline(0, color="k", lw=0.5)
        a.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=160 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] CORUM remodeling plot -> {out_stem}.{{png,svg}}")


# ───────────────────────────────────────────────────────────────────────────
# (2) Leiden cluster transitions
# ───────────────────────────────────────────────────────────────────────────

def evaluate_cluster_transitions(
    z_treat: np.ndarray, cluster_ref: np.ndarray,
    leiden_knn_k: int = 15, leiden_resolution: float = 1.0, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cluster_treat, confusion[K_ref, K_treat], row_normalised_confusion)."""
    cluster_treat = cluster_z_leiden(z_treat, k=leiden_knn_k,
                                     resolution=leiden_resolution, seed=seed)
    K_ref   = int(cluster_ref.max() + 1)
    K_treat = int(cluster_treat.max() + 1)
    conf = np.zeros((K_ref, K_treat), dtype=np.int64)
    for r, t in zip(cluster_ref, cluster_treat):
        conf[int(r), int(t)] += 1
    row_norm = conf.astype(np.float32) / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    return cluster_treat, conf, row_norm


def summarise_transitions(conf: np.ndarray) -> dict:
    """Summary statistics for the cluster_ref × cluster_treat confusion."""
    total = int(conf.sum())
    same  = int(conf.diagonal()[:min(conf.shape)].sum())   # only meaningful if shapes match
    # The "stayed in same cluster id" is not literally diagonal if K_ref ≠ K_treat.
    # A robust metric: for each ref cluster, fraction sent to the modal treat cluster.
    row_max = conf.max(axis=1).sum()
    return {
        "total_proteins":            total,
        "ref_clusters":              int(conf.shape[0]),
        "treat_clusters":            int(conf.shape[1]),
        "mass_on_modal_target":      float(row_max / max(total, 1)),  # ~1 if no remodeling
        "mass_off_modal_target":     float(1.0 - row_max / max(total, 1)),
    }


def plot_transition_heatmap(row_norm: np.ndarray, out_stem: str):
    if not _MPL_OK:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(row_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Stage-2 (treated) Leiden cluster")
    ax.set_ylabel("Stage-1 (reference) Leiden cluster")
    ax.set_title("Cluster transition (row-normalised; row=ref cluster, col=treat cluster)",
                 fontsize=10, loc="left")
    plt.colorbar(im, ax=ax, label="fraction of ref cluster's proteins")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=160 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] transition heatmap -> {out_stem}.{{png,svg}}")


# ───────────────────────────────────────────────────────────────────────────
# (3) HPA localization shift
# ───────────────────────────────────────────────────────────────────────────

def evaluate_localization_shift(
    D_image, z_ref: torch.Tensor, z_treat: torch.Tensor,
    class_labels: Optional[List[str]] = None,
    kl_top_n: int = 50,
) -> List[dict]:
    """Per-protein top-class before/after and KL divergence of HPA predictions."""
    with torch.no_grad():
        p_ref   = F.softmax(D_image(z_ref),   dim=-1)
        p_treat = F.softmax(D_image(z_treat), dim=-1)
    top_ref   = p_ref.argmax(dim=-1).cpu().numpy()
    top_treat = p_treat.argmax(dim=-1).cpu().numpy()
    conf_ref   = p_ref.max(dim=-1).values.cpu().numpy()
    conf_treat = p_treat.max(dim=-1).values.cpu().numpy()
    # KL(p_ref || p_treat) — large = shift
    kl = (p_ref * (p_ref.clamp_min(1e-9).log()
                   - p_treat.clamp_min(1e-9).log())).sum(dim=-1).cpu().numpy()
    rows = []
    K = p_ref.shape[-1]
    if class_labels is None or len(class_labels) != K:
        class_labels = [f"hpa_{i}" for i in range(K)]
    for i in range(len(top_ref)):
        rows.append({
            "top_ref":         class_labels[int(top_ref[i])],
            "top_treat":       class_labels[int(top_treat[i])],
            "changed_top":     bool(top_ref[i] != top_treat[i]),
            "conf_ref":        float(conf_ref[i]),
            "conf_treat":      float(conf_treat[i]),
            "kl_ref_to_treat": float(kl[i]),
        })
    return rows


def plot_localization_summary(rows: List[dict], out_stem: str, top_n: int = 30):
    if not _MPL_OK or not rows:
        return
    kl = np.array([r["kl_ref_to_treat"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # KL distribution
    ax[0].hist(kl, bins=80, color="#7570b3", edgecolor="black", linewidth=0.3)
    ax[0].set_xlabel("KL( D_image(z_ref) || D_image(z_treat) )")
    ax[0].set_ylabel("# proteins")
    ax[0].set_title("HPA prediction shift distribution", fontsize=10, loc="left")
    ax[0].grid(axis="y", alpha=0.25)
    # Top compartment-change transitions
    changes = {}
    for r in rows:
        if r["changed_top"]:
            k = (r["top_ref"], r["top_treat"])
            changes[k] = changes.get(k, 0) + 1
    top = sorted(changes.items(), key=lambda kv: -kv[1])[:top_n]
    if top:
        labels = [f"{a} → {b}" for (a, b), _ in top]
        counts = [c for _, c in top]
        ax[1].barh(range(len(top)), counts[::-1], color="#1b9e77",
                   edgecolor="black", linewidth=0.3)
        ax[1].set_yticks(range(len(top)))
        ax[1].set_yticklabels(labels[::-1], fontsize=7)
        ax[1].set_xlabel("# proteins switching top-1 compartment")
        ax[1].set_title(f"Top {len(top)} compartment-change transitions",
                        fontsize=10, loc="left")
        ax[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=160 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] HPA shift plot -> {out_stem}.{{png,svg}}")


# ───────────────────────────────────────────────────────────────────────────
# (4) Coherence × magnitude flagging
# ───────────────────────────────────────────────────────────────────────────

def evaluate_coherence_magnitude(
    delta_final: np.ndarray, coherence: np.ndarray,
    mag_quantile: float = 0.5, coh_thresh: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Categorise each protein into one of four flags."""
    mag = np.linalg.norm(delta_final, axis=1)
    mag_thresh = float(np.quantile(mag, mag_quantile))
    cats = []
    for m, c in zip(mag, coherence):
        if m >  mag_thresh and c >  coh_thresh:  cats.append("hi-confidence remodelling")
        elif m >  mag_thresh and c <= coh_thresh: cats.append("inspect: solo move or noise")
        elif m <= mag_thresh and c >  coh_thresh: cats.append("small but supported")
        else:                                     cats.append("no remodelling")
    return mag, np.full_like(mag, mag_thresh), cats


def plot_coherence_magnitude(mag: np.ndarray, coherence: np.ndarray,
                              mag_thresh: float, coh_thresh: float,
                              out_stem: str):
    if not _MPL_OK:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coherence, mag, s=4, alpha=0.35, color="#3182bd")
    ax.axvline(coh_thresh, color="k", lw=0.6, alpha=0.5)
    ax.axhline(mag_thresh, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("coherence  cos(δ_raw_proj, δ̂)")
    ax.set_ylabel("‖δ_final‖")
    ax.set_title("Coherence × magnitude  (quadrants tell the story)",
                 fontsize=10, loc="left")
    # Annotate quadrants
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    txt_kw = dict(fontsize=8, color="#444", ha="center")
    ax.text((xlim[0] + coh_thresh) / 2, ylim[1] * 0.95,
            "inspect:\nsolo move / noise", **txt_kw)
    ax.text((coh_thresh + xlim[1]) / 2, ylim[1] * 0.95,
            "hi-confidence\nremodelling", **txt_kw)
    ax.text((xlim[0] + coh_thresh) / 2, ylim[0] + (ylim[1] - ylim[0]) * 0.05,
            "no remodelling", **txt_kw)
    ax.text((coh_thresh + xlim[1]) / 2, ylim[0] + (ylim[1] - ylim[0]) * 0.05,
            "small but supported", **txt_kw)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=160 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] coherence×magnitude plot -> {out_stem}.{{png,svg}}")


# ───────────────────────────────────────────────────────────────────────────
# Top-level driver
# ───────────────────────────────────────────────────────────────────────────

def _load_inference_dir(inference_dir: str
                        ) -> Tuple[List[str], np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """Load (proteins, z_treat, delta_final, sigma2_pred, coherence)."""
    proteins, z_treat = _read_tsv(os.path.join(inference_dir, "z_treat.tsv"))
    _, delta_final    = _read_tsv(os.path.join(inference_dir, "delta_final.tsv"))
    _, s2p            = _read_tsv(os.path.join(inference_dir, "sigma2_pred.tsv"))
    _, coh            = _read_tsv(os.path.join(inference_dir, "coherence.tsv"))
    return proteins, z_treat, delta_final, s2p[:, 0], coh[:, 0]


def run_eval(
    adapter_dir: str,
    inference_dir: str,
    corum_path: Optional[str],
    outdir: str,
    *,
    leiden_resolution: float = 1.0,
    leiden_knn_k: int = 15,
    device: Optional[str] = None,
    seed: int = 0,
):
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1) load adapter config + Stage1Cache (for z_ref, cluster_ref, D_image) ----
    with open(os.path.join(adapter_dir, "config.json")) as f:
        cfg = json.load(f)
    print(f"[eval] adapter:   {adapter_dir}")
    print(f"[eval] inference: {inference_dir}")
    print(f"[eval] condition: {cfg.get('condition')!r}")

    stage1_outdir = cfg.get("stage1_outdir")
    if not stage1_outdir:
        raise RuntimeError("config.json missing 'stage1_outdir'. Retrain the adapter.")
    cache = Stage1Cache.from_stage1_dir(
        stage1_outdir, cfg.get("manifest_path"),
        epic_name=cfg.get("epic_name", "epic"),
        k=cfg.get("k", 20),
        leiden_resolution=cfg.get("leiden_resolution", 1.0),
        cluster_compat_temp=cfg.get("cluster_compat_temp", 1.0),
        device=device, seed=cfg.get("seed", 0),
        seq_name=cfg.get("seq_modality_hint"),
        image_name=cfg.get("image_modality_hint"),
    )

    # ---- 2) load inference outputs ----
    proteins, z_treat, delta_final, sigma2_pred, coherence = _load_inference_dir(inference_dir)
    if proteins != cache.proteins:
        # Best-effort align: warn but still attempt index lookup
        n_match = sum(1 for a, b in zip(proteins, cache.proteins) if a == b)
        print(f"[eval] WARN: protein orders differ ({n_match}/{len(proteins)} match in order). "
              f"Assuming inference outputs are aligned to cache.proteins.")
    z_ref = cache.z.cpu().numpy()
    cluster_ref = cache.cluster_id.cpu().numpy()
    print(f"[eval] loaded: N={len(proteins)}  d_z={z_ref.shape[1]}  K_ref={int(cluster_ref.max()+1)}")

    # ---- 3) CORUM remodeling ----
    if corum_path and os.path.isfile(corum_path):
        corum = _read_corum(corum_path)
        rows = evaluate_corum_remodeling(z_ref, z_treat, proteins, corum,
                                         min_members_overlap=3)
        if rows:
            _write_tsv(os.path.join(outdir, "corum_remodeling.tsv"),
                       ["complex", "n_members_in_embed", "mean_dist_ref",
                        "mean_dist_treat", "delta_dist", "members"],
                       [[r[k] for k in ("complex", "n_members_in_embed",
                                         "mean_dist_ref", "mean_dist_treat",
                                         "delta_dist", "members")] for r in rows])
            plot_corum_top_bottom(rows, os.path.join(outdir, "corum_remodeling"),
                                  top_n=25)
            print(f"[eval] CORUM: {len(rows)} complexes ranked by Δdist; "
                  f"top dissociating: {rows[0]['complex']} "
                  f"(Δd={rows[0]['delta_dist']:+.3f}); "
                  f"top tightening: {rows[-1]['complex']} "
                  f"(Δd={rows[-1]['delta_dist']:+.3f})")
        else:
            print("[eval] CORUM: no complexes met overlap threshold")
    else:
        print(f"[eval] CORUM: skipped (no path or file missing): {corum_path!r}")

    # ---- 4) Cluster transitions ----
    cluster_treat, conf, row_norm = evaluate_cluster_transitions(
        z_treat, cluster_ref,
        leiden_knn_k=leiden_knn_k, leiden_resolution=leiden_resolution, seed=seed)
    _write_tsv(os.path.join(outdir, "cluster_assignments.tsv"),
               ["protein", "cluster_ref", "cluster_treat", "changed_cluster"],
               [[p, int(cluster_ref[i]), int(cluster_treat[i]),
                 int(cluster_ref[i] != cluster_treat[i])]
                for i, p in enumerate(proteins)])
    # Save the (small) confusion matrix as a TSV too
    with open(os.path.join(outdir, "cluster_confusion_counts.tsv"), "w") as f:
        f.write("ref\\treat\t" + "\t".join(str(c) for c in range(conf.shape[1])) + "\n")
        for r in range(conf.shape[0]):
            f.write(f"{r}\t" + "\t".join(str(int(x)) for x in conf[r]) + "\n")
    summ = summarise_transitions(conf)
    with open(os.path.join(outdir, "cluster_transitions_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    plot_transition_heatmap(row_norm, os.path.join(outdir, "cluster_transitions"))
    print(f"[eval] cluster transitions: K_ref={summ['ref_clusters']} → K_treat={summ['treat_clusters']}; "
          f"modal-target mass={summ['mass_on_modal_target']:.3f}  "
          f"(off-target={summ['mass_off_modal_target']:.3f})")

    # ---- 5) HPA localization shift ----
    if cache.D_image is not None:
        with torch.no_grad():
            z_ref_t   = cache.z.to(device)
            z_treat_t = torch.from_numpy(z_treat).to(device)
        rows = evaluate_localization_shift(cache.D_image, z_ref_t, z_treat_t)
        _write_tsv(os.path.join(outdir, "localization_shift.tsv"),
                   ["protein", "top_ref", "top_treat", "changed_top",
                    "conf_ref", "conf_treat", "kl_ref_to_treat"],
                   [[proteins[i], r["top_ref"], r["top_treat"],
                     int(r["changed_top"]), f"{r['conf_ref']:.4f}",
                     f"{r['conf_treat']:.4f}", f"{r['kl_ref_to_treat']:.4f}"]
                    for i, r in enumerate(rows)])
        plot_localization_summary(rows, os.path.join(outdir, "localization_shift"),
                                   top_n=30)
        n_changed = sum(1 for r in rows if r["changed_top"])
        print(f"[eval] HPA shift: {n_changed}/{len(rows)} proteins changed top-1 compartment")
    else:
        print("[eval] HPA shift: skipped (D_image not available in cache)")

    # ---- 6) Coherence × magnitude flags ----
    mag, mag_thresh_arr, cats = evaluate_coherence_magnitude(
        delta_final, coherence, mag_quantile=0.5, coh_thresh=0.3)
    _write_tsv(os.path.join(outdir, "coherence_magnitude_flags.tsv"),
               ["protein", "delta_final_norm", "coherence", "sigma2_pred", "flag"],
               [[proteins[i], f"{mag[i]:.4f}", f"{coherence[i]:.4f}",
                 f"{sigma2_pred[i]:.4f}", cats[i]]
                for i in range(len(proteins))])
    plot_coherence_magnitude(mag, coherence, float(mag_thresh_arr[0]), 0.3,
                              os.path.join(outdir, "coherence_magnitude"))
    counts = {c: cats.count(c) for c in set(cats)}
    print(f"[eval] coherence×magnitude flags: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # ---- 7) Write a small JSON manifest of what got produced ----
    with open(os.path.join(outdir, "eval_manifest.json"), "w") as f:
        json.dump({
            "adapter_dir":    adapter_dir,
            "inference_dir":  inference_dir,
            "outdir":         outdir,
            "condition":      cfg.get("condition"),
            "stage1_cache":   cfg.get("cache_path"),
            "outputs": [
                "corum_remodeling.tsv (+ .png/.svg)",
                "cluster_assignments.tsv",
                "cluster_confusion_counts.tsv",
                "cluster_transitions_summary.json",
                "cluster_transitions.{png,svg}",
                "localization_shift.tsv (+ summary plot)",
                "coherence_magnitude_flags.tsv (+ scatter)",
            ],
        }, f, indent=2)
    print(f"[eval] done -> {outdir}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter_dir",   required=True)
    p.add_argument("--inference_dir", required=True)
    p.add_argument("--corum",         default=None,
                   help="CORUM TSV (complex_name + subunits_gene_name). Skip if omitted.")
    p.add_argument("--outdir",        required=True)
    p.add_argument("--leiden_resolution", type=float, default=1.0)
    p.add_argument("--leiden_knn_k",      type=int,   default=15)
    p.add_argument("--device",        default=None)
    p.add_argument("--seed",          type=int, default=0)
    args = p.parse_args()
    run_eval(args.adapter_dir, args.inference_dir, args.corum, args.outdir,
             leiden_resolution=args.leiden_resolution,
             leiden_knn_k=args.leiden_knn_k,
             device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
