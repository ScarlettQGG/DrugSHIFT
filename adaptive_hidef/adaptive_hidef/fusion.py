"""Adaptive, size-graded persistence fusion of a single HiDeF run into one
hierarchy.

A single low-``k`` HiDeF run already carries the whole persistence spectrum:
each community's persistence is the highest chi at which it survives. Rather than
apply one flat persistence cut (which either keeps an unstable small-cluster tail
or deletes genuine small complexes along with the noise), a community is kept iff
its persistence meets a requirement that **ramps with community size**:

    chi_required(size) decreases log-linearly from ``chi_max`` for the smallest
    communities to ``chi_min`` for the largest.

Large top-level systems are therefore admitted even when only moderately stable
(preserving a deep, narrow top), while small communities must be highly stable to
survive (keeping curated complexes, not noise, at the leaves).

The discrete threshold levels between ``chi_min`` and ``chi_max`` are chosen from
the data, not hand-set:

    1. 1-D k-means clusters the candidate persistences for k = 2..K.
    2. A kneedle elbow on the k-means inertia curve selects the natural number of
       levels.
    3. The k-means centres (rounded and clamped to ``[chi_min, chi_max]``, with
       ``chi_max`` retained as the ceiling) become the allowed levels; each
       community's continuous ``chi_required`` is snapped to the nearest level.

Kept communities are re-woven through HiDeF's own weaver (``merge=True`` folds
near-duplicates), then pruned with the standard HiDeF hierarchy refiner. Member
ids throughout are co-embedding row indices.
"""

import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from hidef import weaver  # noqa: E402
from hidef import hidef_finder  # noqa: E402
from cellmaps_generate_hierarchy.maturehierarchy import (  # noqa: E402
    HiDeFHierarchyRefiner,
)


def load_raw_nodes(path):
    """Parse a raw HiDeF ``.nodes`` file.

    Each line is ``cluster_id <TAB> size <TAB> space-separated member ids <TAB>
    persistence``. Member ids are co-embedding row indices.

    :param path: path to the ``.nodes`` file.
    :returns: list of dicts with keys ``cluster_id``, ``size``,
        ``members`` (list[int]) and ``persistence`` (float).
    """
    clusters = []
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or not fields[0]:
                continue
            members = [int(x) for x in fields[2].split()]
            try:
                persistence = float(fields[3]) if len(fields) > 3 else 0.0
            except ValueError:
                persistence = 0.0
            clusters.append(
                {
                    "cluster_id": fields[0],
                    "size": int(fields[1]),
                    "members": members,
                    "persistence": persistence,
                }
            )
    return clusters


def chi_required(size, s_min, s_max, chi_min, chi_max):
    """Continuous persistence requirement for a community of the given size.

    Ramps log-linearly from ``chi_max`` at ``s_min`` to ``chi_min`` at ``s_max``.

    :param size: community size.
    :param s_min: smallest candidate community size.
    :param s_max: largest candidate community size.
    :param chi_min: persistence required for the largest communities.
    :param chi_max: persistence required for the smallest communities.
    :returns: continuous persistence threshold.
    """
    if s_max <= s_min:
        return chi_max
    frac = (math.log(size) - math.log(s_min)) / (math.log(s_max) - math.log(s_min))
    frac = min(max(frac, 0.0), 1.0)
    return chi_max - (chi_max - chi_min) * frac


def choose_levels(persistences, chi_min, chi_max, k_max=6, forced_k=0):
    """Select the discrete persistence levels from the data.

    Runs 1-D k-means on the candidate persistences for k = 2..``k_max`` and, by
    default, picks the number of levels with a kneedle elbow on the inertia
    curve; ``forced_k`` overrides the elbow while keeping data-driven centres.
    The chosen centres are rounded and clamped to ``[chi_min, chi_max]`` and
    ``chi_max`` is retained as the ceiling.

    :param persistences: candidate community persistences.
    :param chi_min: lower clamp for levels.
    :param chi_max: upper clamp / retained ceiling.
    :param k_max: maximum number of k-means levels to scan.
    :param forced_k: if > 0, use exactly this many levels instead of the elbow.
    :returns: tuple ``(levels, best_k, diagnostics)`` where ``levels`` is the
        sorted list of integer levels, ``best_k`` the selected number, and
        ``diagnostics`` a list of ``(k, inertia, centers)`` tuples.
    """
    x = np.clip(np.asarray(persistences, dtype=float), chi_min, chi_max + 2)
    x = x.reshape(-1, 1)
    ks = list(range(2, max(k_max, forced_k) + 1))
    inertias, centers_by_k = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x)
        inertias.append(km.inertia_)
        centers_by_k[k] = sorted(km.cluster_centers_.ravel().tolist())

    inertia = np.array(inertias)
    k_norm = (np.array(ks, float) - ks[0]) / (ks[-1] - ks[0])
    inertia_norm = (inertia[0] - inertia) / (inertia[0] - inertia[-1] + 1e-9)
    best_k = forced_k if forced_k > 0 else ks[int(np.argmax(inertia_norm - k_norm))]

    centers = centers_by_k[best_k]
    levels = sorted({int(round(min(max(c, chi_min), chi_max))) for c in centers})
    if chi_max not in levels:
        levels.append(chi_max)
    levels = sorted(set(levels))
    diagnostics = list(zip(ks, inertias, [centers_by_k[k] for k in ks]))
    return levels, best_k, diagnostics


def _snap(value, levels):
    """Return the level nearest to a continuous requirement."""
    return min(levels, key=lambda level: abs(level - value))


def select_communities(clusters, n_proteins, chi_min, chi_max, levels):
    """Keep communities whose persistence meets their snapped size requirement.

    :param clusters: parsed communities (from :func:`load_raw_nodes`).
    :param n_proteins: total number of proteins (root size); used to drop the
        root and any all-protein community from the candidate set.
    :param chi_min: persistence required for the largest communities.
    :param chi_max: persistence required for the smallest communities.
    :param levels: allowed discrete levels (from :func:`choose_levels`).
    :returns: tuple ``(kept, bands, size_range)`` where ``kept`` is a list of
        ``(sorted_members, persistence)`` (deduplicated by member set, largest
        first), ``bands`` maps each level to candidate/kept counts and its size
        span, and ``size_range`` is ``(s_min, s_max)``.
    """
    candidates = [c for c in clusters if 0 < c["size"] < n_proteins]
    sizes = [c["size"] for c in candidates]
    s_min, s_max = float(min(sizes)), float(max(sizes))

    kept = {}
    bands = {
        level: {"candidates": 0, "kept": 0, "size_min": math.inf, "size_max": 0}
        for level in levels
    }
    for cluster in candidates:
        threshold = _snap(
            chi_required(cluster["size"], s_min, s_max, chi_min, chi_max), levels
        )
        band = bands[threshold]
        band["candidates"] += 1
        band["size_min"] = min(band["size_min"], cluster["size"])
        band["size_max"] = max(band["size_max"], cluster["size"])
        if cluster["persistence"] >= threshold:
            band["kept"] += 1
            members = frozenset(cluster["members"])
            kept[members] = max(kept.get(members, 0.0), cluster["persistence"])

    kept_list = [(sorted(members), pers) for members, pers in kept.items()]
    kept_list.sort(key=lambda item: len(item[0]), reverse=True)
    return kept_list, bands, (s_min, s_max)


def weave_communities(kept, n_proteins, outprefix, weave_cutoff=0.75):
    """Re-weave kept communities into a containment hierarchy.

    A full-protein root community is prepended, then HiDeF's weaver rebuilds the
    parent/child DAG (``merge=True`` folds mutually contained near-duplicates)
    and writes ``<outprefix>.nodes`` / ``<outprefix>.edges``.

    :param kept: list of ``(members, persistence)`` from
        :func:`select_communities`.
    :param n_proteins: total number of proteins.
    :param outprefix: output path prefix for the woven hierarchy.
    :param weave_cutoff: containment threshold for the weaver.
    """
    rows, persistence = [np.ones(n_proteins, dtype=int)], [0.0]
    for members, pers in kept:
        vector = np.zeros(n_proteins, dtype=int)
        vector[members] = 1
        rows.append(vector)
        persistence.append(pers)

    woven = weaver.Weaver()
    woven.weave(rows, boolean=True, levels=False, merge=True, cutoff=weave_cutoff)
    hidef_finder.output_all(
        woven,
        [str(i) for i in range(n_proteins)],
        outprefix,
        persistence=persistence,
        iter=False,
        skipgml=True,
    )


def write_parent_cx2(proteins, path):
    """Write a minimal CX2 mapping node id (co-embedding row) -> gene symbol.

    This resolves the integer member ids in the hierarchy back to gene symbols.

    :param proteins: gene symbols in co-embedding row order.
    :param path: output ``.cx2`` path.
    """
    nodes = [{"id": i, "v": {"n": str(proteins[i])}} for i in range(len(proteins))]
    with open(path, "w") as handle:
        json.dump([{"nodes": nodes}], handle)


def _hierarchy_stats(nodes_path, edges_path):
    """Summarise structure of a HiDeF hierarchy (node/leaf/depth counts)."""
    size, level = {}, {}
    for line in open(nodes_path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 2 or not fields[0]:
            continue
        size[fields[0]] = int(fields[1])
        level[fields[0]] = int(fields[0].replace("Cluster", "").split("-")[0])

    parents = set()
    for line in open(edges_path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2:
            parents.add(fields[0])

    root = max(size, key=lambda c: size[c])
    top = sum(1 for line in open(edges_path) if line.split("\t")[0] == root)
    leaves = [c for c in size if c not in parents]
    leaf_sizes = np.array(sorted(size[c] for c in leaves))
    return {
        "nodes": len(size),
        "top_clusters": top,
        "max_depth": max(level.values()),
        "n_leaves": len(leaves),
        "leaf_median": float(np.median(leaf_sizes)),
        "leaves_4to8": int(((leaf_sizes >= 4) & (leaf_sizes <= 8)).sum()),
        "_leaf_sizes": leaf_sizes,
    }


def write_diagnostics(pruned_prefix, raw_nodes, raw_edges, outdir):
    """Write structure statistics and a leaf-size histogram.

    Compares the final pruned hierarchy with the raw HiDeF source it was fused
    from, writing ``fusion_stats.tsv`` and ``leaf_size_distribution.png``.

    :param pruned_prefix: prefix of the final pruned hierarchy.
    :param raw_nodes: raw HiDeF ``.nodes`` path (fusion source).
    :param raw_edges: raw HiDeF ``.edges`` path (fusion source).
    :param outdir: output directory for the diagnostic files.
    """
    configs = [
        ("adaptive_fused", pruned_prefix + ".pruned.nodes", pruned_prefix + ".pruned.edges"),
        ("raw_source", raw_nodes, raw_edges),
    ]
    rows, leaf_dists = [], {}
    for name, nodes_path, edges_path in configs:
        if os.path.exists(nodes_path) and os.path.exists(edges_path):
            stats = _hierarchy_stats(nodes_path, edges_path)
            leaf_dists[name] = stats.pop("_leaf_sizes")
            rows.append({"config": name, **stats})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(outdir, "fusion_stats.tsv"), sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for name, sizes in leaf_dists.items():
        ax.hist(
            np.clip(sizes, 0, 40),
            bins=np.arange(2, 42, 2),
            histtype="step",
            lw=2,
            label=f"{name} (n={len(sizes)})",
        )
    ax.set_xlabel("leaf size (capped at 40)")
    ax.set_ylabel("leaf clusters")
    ax.set_title("Adaptive hierarchy leaf-size distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "leaf_size_distribution.png"), dpi=300)
    plt.close(fig)
    return table
