"""Per-modality pseudo-label computation for MUSE-style structure triplets.

For each modality, cluster the proteins present in that modality using either:
  * Leiden on a mutual-kNN cosine graph  (preferred — natural cluster shapes)
  * KMeans  (fallback if leidenalg/igraph aren't installed)

Produces a per-modality dict {protein: cluster_id}. Proteins missing from a
modality get NO pseudo-label for it (the structure triplet for that modality
simply skips them — masked).

Cached to disk so the (expensive) clustering doesn't repeat per epoch.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np


def _row_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _knn_edges(X: np.ndarray, k: int = 15) -> np.ndarray:
    """Cosine kNN edges (mutual). Returns (n_edges, 2) int array."""
    from sklearn.neighbors import NearestNeighbors
    Xn = _row_normalize(X.astype(np.float32))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn)
    _, idx = nn.kneighbors(Xn)
    n = idx.shape[0]
    # Build undirected edge list, mutual filter
    sets = [set(idx[i, 1:].tolist()) for i in range(n)]
    edges = []
    for i in range(n):
        for j in sets[i]:
            if j > i and i in sets[j]:
                edges.append((i, j))
    if not edges:
        # Fall back to non-mutual kNN if mutual is too sparse
        edges = [(i, j) for i in range(n) for j in idx[i, 1:].tolist() if j > i]
    return np.asarray(edges, dtype=np.int64)


def cluster_leiden(X: np.ndarray, k: int = 15, resolution: float = 1.0,
                   seed: int = 0) -> Optional[np.ndarray]:
    """Leiden on a mutual cosine kNN graph. Returns membership array or None
    if leidenalg/igraph aren't available."""
    try:
        import igraph as ig
        import leidenalg
    except Exception:
        return None
    edges = _knn_edges(X, k=k)
    G = ig.Graph(n=X.shape[0], edges=list(map(tuple, edges.tolist())),
                 directed=False)
    G = G.simplify()
    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=np.int64)


def cluster_kmeans(X: np.ndarray, n_clusters: int = 50,
                   seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed) \
        .fit_predict(_row_normalize(X))


def cluster_one_modality(X: np.ndarray, method: str = "leiden",
                         knn_k: int = 15, leiden_resolution: float = 1.0,
                         kmeans_k: int = 50, seed: int = 0) -> Tuple[np.ndarray, str]:
    """Return (labels, method_used). Falls back to KMeans if Leiden unavailable."""
    if method.lower() == "leiden":
        labs = cluster_leiden(X, k=knn_k, resolution=leiden_resolution, seed=seed)
        if labs is not None:
            return labs, "leiden"
    return cluster_kmeans(X, n_clusters=kmeans_k, seed=seed), "kmeans"


def compute_all_pseudo_labels(
    modality_matrices: Dict[str, Tuple[List[str], np.ndarray]],
    method: str = "leiden",
    knn_k: int = 15,
    leiden_resolution: float = 1.0,
    kmeans_k: int = 50,
    seed: int = 0,
    cache_path: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """For each modality, cluster its proteins and return {protein: cluster_id}.

    modality_matrices : {m: (protein_names: List[str], X: ndarray[n_m, d_m])}
    cache_path        : if given, JSON path to cache labels for reuse.

    Returns
    -------
    dict {m: {protein_name: cluster_id (int)}}
    """
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        # sanity-check: same modalities & protein sets
        same = (set(cached.keys()) == set(modality_matrices.keys())
                and all(set(cached[m].keys()) == set(modality_matrices[m][0])
                        for m in cached))
        if same:
            print(f"[pseudo_labels] loaded cache from {cache_path}")
            return {m: {p: int(v) for p, v in cached[m].items()} for m in cached}
        print(f"[pseudo_labels] cache mismatch at {cache_path}; recomputing")

    out: Dict[str, Dict[str, int]] = {}
    for m, (prots, X) in modality_matrices.items():
        labs, used = cluster_one_modality(
            X, method=method, knn_k=knn_k,
            leiden_resolution=leiden_resolution, kmeans_k=kmeans_k, seed=seed,
        )
        n_clusters = int(labs.max()) + 1 if len(labs) else 0
        sizes = np.bincount(labs) if len(labs) else np.array([])
        print(f"[pseudo_labels] {m}: {used}  n={len(prots)}  "
              f"clusters={n_clusters}  median_size={int(np.median(sizes)) if len(sizes) else 0}  "
              f"max_size={int(sizes.max()) if len(sizes) else 0}")
        out[m] = {p: int(l) for p, l in zip(prots, labs.tolist())}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({m: {p: int(v) for p, v in d.items()} for m, d in out.items()},
                      f)
        print(f"[pseudo_labels] cached to {cache_path}")
    return out
