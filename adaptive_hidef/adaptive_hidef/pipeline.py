"""End-to-end pipeline: co-embedding -> adaptive protein-cluster hierarchy.

Three stages are run in order:

    1. ``ppi``    -- cosine-similarity PPI networks at a ladder of edge cutoffs.
    2. ``hidef``  -- one low-``k`` HiDeF run over the multiplex of those networks.
    3. ``fuse``   -- adaptive, size-graded persistence fusion into the final
                     pruned hierarchy.

Stage 2 is the expensive step; :func:`run_pipeline` accepts a pre-computed raw
HiDeF ``.nodes`` / ``.edges`` pair to skip it. All intermediate artifacts are
written under the output directory.
"""

import os

import pandas as pd

from cellmaps_generate_hierarchy.maturehierarchy import HiDeFHierarchyRefiner

from . import fusion
from . import ppi as ppi_module
from .hidef_runner import run_hidef


def fuse_hierarchy(
    raw_nodes_path,
    proteins,
    outdir,
    chi_min=5,
    chi_max=12,
    k_max=6,
    forced_levels=0,
    manual_levels=None,
    weave_cutoff=0.75,
    min_system_size=4,
    jaccard_threshold=0.9,
    containment_threshold=0.75,
    min_diff=1,
    raw_edges_path=None,
    verbose=True,
):
    """Run the adaptive fusion stage on a raw HiDeF ``.nodes`` file.

    :param raw_nodes_path: raw HiDeF ``.nodes`` file (member ids = co-embedding
        rows).
    :param proteins: gene symbols in co-embedding row order.
    :param outdir: output directory for the final hierarchy and diagnostics.
    :param chi_min: persistence required for the largest communities.
    :param chi_max: persistence required for the smallest communities; the
        precision/recall dial (lower keeps more, higher keeps fewer).
    :param k_max: maximum number of k-means levels to scan.
    :param forced_levels: force this many data-driven levels (0 = kneedle elbow).
    :param manual_levels: explicit list of integer levels, overriding selection.
    :param weave_cutoff: containment threshold for the weaver.
    :param min_system_size: minimum cluster size kept by the refiner.
    :param jaccard_threshold: Jaccard merge threshold for the refiner.
    :param containment_threshold: containment threshold for the refiner.
    :param min_diff: minimum level difference for the refiner.
    :param raw_edges_path: raw HiDeF ``.edges`` file for diagnostics (optional).
    :param verbose: print progress and level-selection detail.
    :returns: prefix of the final pruned hierarchy (``<outdir>/hidef_output``).
    """
    os.makedirs(outdir, exist_ok=True)
    n_proteins = len(proteins)

    clusters = fusion.load_raw_nodes(raw_nodes_path)
    candidates = [c for c in clusters if 0 < c["size"] < n_proteins]
    persistences = [c["persistence"] for c in candidates]

    if manual_levels:
        levels = sorted(int(x) for x in manual_levels)
        if verbose:
            print(f"[fuse] persistence levels (manual): {levels}")
    else:
        levels, best_k, diagnostics = fusion.choose_levels(
            persistences, chi_min, chi_max, k_max, forced_levels
        )
        if verbose:
            how = f"forced k={forced_levels}" if forced_levels else "kneedle elbow"
            print(f"[fuse] data-driven levels (1-D k-means on persistence, {how}):")
            for k, inertia, centers in diagnostics:
                mark = "  <- chosen" if k == best_k else ""
                rounded = [round(c, 1) for c in centers]
                print(f"       k={k}: inertia={inertia:8.0f}  centers={rounded}{mark}")
            skipped = [c for c in range(chi_min, chi_max + 1) if c not in levels]
            print(f"       ==> levels {levels}   (skipped {skipped})")
        pd.DataFrame(
            [
                {"k": k, "inertia": inertia, "centers": ";".join(f"{c:.2f}" for c in centers)}
                for k, inertia, centers in diagnostics
            ]
        ).to_csv(os.path.join(outdir, "level_selection.tsv"), sep="\t", index=False)

    kept, bands, (s_min, s_max) = fusion.select_communities(
        clusters, n_proteins, chi_min, chi_max, levels
    )
    if verbose:
        print(f"[fuse] size->persistence bands (sizes {int(s_min)}..{int(s_max)}):")
        for level in sorted(bands):
            band = bands[level]
            if band["candidates"]:
                print(
                    f"       chi={level}: size {band['size_min']}..{band['size_max']}  "
                    f"{band['candidates']} candidates, {band['kept']} kept"
                )
        print(f"[fuse] kept {len(kept)} communities")

    outprefix = os.path.join(outdir, "hidef_output")
    fusion.weave_communities(kept, n_proteins, outprefix, weave_cutoff)

    refiner = HiDeFHierarchyRefiner(
        ci_thre=containment_threshold,
        ji_thre=jaccard_threshold,
        min_term_size=min_system_size,
        min_diff=min_diff,
        provenance_utils=None,
    )
    refiner.refine_hierarchy(outprefix=outprefix)

    fusion.write_parent_cx2(proteins, os.path.join(outdir, "hierarchy_parent.cx2"))

    if raw_edges_path is None:
        raw_edges_path = raw_nodes_path.replace(".nodes", ".edges")
    table = fusion.write_diagnostics(outprefix, raw_nodes_path, raw_edges_path, outdir)
    if verbose:
        n_final = sum(1 for _ in open(outprefix + ".pruned.nodes"))
        print(f"[fuse] final pruned hierarchy: {n_final} clusters")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(table.to_string(index=False))
    return outprefix


def run_pipeline(
    coembedding_path,
    outdir,
    chi_min=5,
    chi_max=12,
    k=5,
    maxres=80,
    numthreads=1,
    cutoffs=ppi_module.DEFAULT_CUTOFFS,
    raw_nodes_path=None,
    raw_edges_path=None,
    verbose=True,
    **fuse_kwargs,
):
    """Run the full co-embedding -> hierarchy pipeline.

    :param coembedding_path: path to the co-embedding TSV.
    :param outdir: output directory (created if absent).
    :param chi_min: persistence required for the largest communities.
    :param chi_max: persistence required for the smallest communities (the
        precision/recall dial).
    :param k: HiDeF persistence threshold for the community-detection run.
    :param maxres: maximum Leiden resolution for the HiDeF run.
    :param numthreads: worker threads for the HiDeF run.
    :param cutoffs: edge-retention fractions for the PPI networks.
    :param raw_nodes_path: pre-computed raw HiDeF ``.nodes`` to reuse; when
        given, stages 1-2 are skipped.
    :param raw_edges_path: pre-computed raw HiDeF ``.edges`` (with
        ``raw_nodes_path``).
    :param verbose: print progress.
    :param fuse_kwargs: forwarded to :func:`fuse_hierarchy` (level and refiner
        knobs).
    :returns: prefix of the final pruned hierarchy.
    """
    os.makedirs(outdir, exist_ok=True)
    proteins, features = ppi_module.read_coembedding(coembedding_path)
    if verbose:
        print(f"[pipeline] {len(proteins)} proteins, {features.shape[1]} features")

    if raw_nodes_path is None:
        ppi_dir = os.path.join(outdir, "ppi")
        if verbose:
            print(f"[pipeline] 1/3 building {len(cutoffs)} PPI networks -> {ppi_dir}")
        edgelists = ppi_module.write_ppi_networks(features, ppi_dir, cutoffs)

        raw_prefix = os.path.join(outdir, "hidef_raw", "hidef_output")
        if verbose:
            print(f"[pipeline] 2/3 HiDeF (k={k}, maxres={maxres}) -> {raw_prefix}")
        raw_nodes_path, raw_edges_path = run_hidef(
            edgelists, raw_prefix, k=k, maxres=maxres, numthreads=numthreads
        )
    elif verbose:
        print(f"[pipeline] 1-2/3 skipped, reusing raw HiDeF: {raw_nodes_path}")

    if verbose:
        print(f"[pipeline] 3/3 adaptive fusion (chi {chi_min}..{chi_max})")
    return fuse_hierarchy(
        raw_nodes_path,
        proteins,
        outdir,
        chi_min=chi_min,
        chi_max=chi_max,
        raw_edges_path=raw_edges_path,
        verbose=verbose,
        **fuse_kwargs,
    )
