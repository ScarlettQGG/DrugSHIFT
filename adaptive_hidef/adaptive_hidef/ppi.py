"""Build cosine-similarity protein-protein interaction (PPI) networks from a
co-embedding.

The co-embedding assigns every protein a real-valued feature vector. Pairwise
cosine similarity of these vectors defines a weighted, fully connected graph.
A family of increasingly sparse networks is produced by keeping, for each
`cutoff` in a list, only the top `cutoff` fraction of edges ranked by weight
(e.g. `cutoff = 0.01` keeps the 1% strongest edges). Community detection is run
jointly over this family (a multiplex), which lets structure that is only
resolvable at a particular density contribute at that density.

Graph nodes are labelled by the protein's **row index in the co-embedding
table**. Because that labelling is carried through unchanged, the member ids in
the resulting hierarchy are row indices into the co-embedding and need no
remapping.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Fractions of top-weighted edges retained, one network per value. Matches the
# default cutoff ladder used by the reference hierarchy pipeline.
DEFAULT_CUTOFFS = (
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.10,
)


def read_coembedding(path):
    """Load a co-embedding TSV.

    The file has a header row; the first column holds the protein/gene symbol
    and the remaining columns are float features.

    :param path: path to the co-embedding TSV.
    :returns: tuple ``(proteins, features)`` where ``proteins`` is the list of
        symbols in row order and ``features`` is an ``(n_proteins, n_features)``
        float array.
    """
    frame = pd.read_csv(path, sep="\t")
    proteins = frame.iloc[:, 0].astype(str).tolist()
    features = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    return proteins, features


def ranked_edges(features):
    """Rank every distinct protein pair by scaled cosine similarity.

    Cosine similarity is computed for all pairs of rows and min-max scaled into
    ``[0, 1]`` across the whole matrix (only the relative ranking of edges is
    used downstream, so the scaling is immaterial to the result but kept for
    parity with the reference pipeline). Only the strict upper triangle is
    returned, so each unordered pair appears once and self-pairs are excluded.

    :param features: ``(n_proteins, n_features)`` float array.
    :returns: tuple ``(row_a, row_b)`` of equal-length int arrays giving the
        endpoint row indices of every edge, ordered by descending weight.
    """
    sim = cosine_similarity(features)
    sim -= sim.min()
    maximum = sim.max()
    if maximum > 0:
        sim /= maximum

    upper_a, upper_b = np.triu_indices(sim.shape[0], k=1)
    weights = sim[upper_a, upper_b]
    order = np.argsort(-weights, kind="stable")
    return upper_a[order], upper_b[order]


def write_ppi_networks(features, outdir, cutoffs=DEFAULT_CUTOFFS):
    """Write one edgelist file per cutoff into ``outdir``.

    Each file is a two-column, header-less TSV of integer node ids (co-embedding
    row indices); node ids are consistent across all files.

    :param features: ``(n_proteins, n_features)`` float array.
    :param outdir: directory to write the edgelist files into (created if
        absent).
    :param cutoffs: iterable of edge-retention fractions.
    :returns: list of written edgelist file paths, ordered as ``cutoffs``.
    """
    os.makedirs(outdir, exist_ok=True)
    row_a, row_b = ranked_edges(features)
    n_edges = len(row_a)

    paths = []
    for cutoff in cutoffs:
        keep = math.ceil(cutoff * n_edges)
        path = os.path.join(outdir, f"ppi_cutoff_{cutoff:g}.edgelist.tsv")
        pd.DataFrame(
            {0: row_a[:keep], 1: row_b[:keep]}
        ).to_csv(path, sep="\t", header=False, index=False)
        paths.append(path)
    return paths
