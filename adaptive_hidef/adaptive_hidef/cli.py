"""Command-line interface for the adaptive HiDeF hierarchy pipeline.

Exposes a single command that turns a co-embedding TSV into a pruned
protein-cluster hierarchy, writing the final node/edge files, the id->gene map
and all intermediates under an output directory.
"""

import argparse

from . import ppi as ppi_module
from .pipeline import run_pipeline


def build_parser():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="adaptive-hidef",
        description=(
            "Build an adaptive protein-cluster hierarchy from a co-embedding: "
            "cosine PPI networks -> low-k HiDeF -> size-graded persistence fusion."
        ),
    )
    parser.add_argument(
        "-i", "--coembedding", required=True,
        help="Input co-embedding TSV (header row; first column = gene symbol, "
             "remaining columns = float features).",
    )
    parser.add_argument(
        "-o", "--outdir", required=True,
        help="Output directory (final hierarchy + all intermediate files).",
    )

    method = parser.add_argument_group("method parameters")
    method.add_argument(
        "--chi_min", type=int, default=5,
        help="Persistence required for the LARGEST communities (top/backbone). "
             "Default 5.",
    )
    method.add_argument(
        "--chi_max", type=int, default=12,
        help="Persistence required for the SMALLEST communities (leaf complexes); "
             "the precision/recall dial (lower = more coverage, higher = more "
             "precision). Default 12.",
    )

    hidef = parser.add_argument_group("HiDeF community detection")
    hidef.add_argument("--k", type=int, default=5,
                       help="HiDeF persistence threshold for the run. Default 5.")
    hidef.add_argument("--maxres", type=float, default=80,
                       help="Maximum Leiden resolution. Default 80.")
    hidef.add_argument("--numthreads", type=int, default=1,
                       help="Worker threads for the resolution scan. Default 1.")
    hidef.add_argument(
        "--ppi_cutoffs", type=float, nargs="+", default=list(ppi_module.DEFAULT_CUTOFFS),
        help="Edge-retention fractions, one PPI network per value.",
    )
    hidef.add_argument(
        "--raw_nodes", default=None,
        help="Reuse a pre-computed raw HiDeF .nodes file and skip PPI + HiDeF "
             "(member ids must be co-embedding row indices).",
    )
    hidef.add_argument(
        "--raw_edges", default=None,
        help="Raw HiDeF .edges file matching --raw_nodes.",
    )

    fuse = parser.add_argument_group("fusion / level selection")
    fuse.add_argument("--k_max", type=int, default=6,
                      help="Maximum number of k-means levels to scan. Default 6.")
    fuse.add_argument("--n_levels", type=int, default=0,
                      help="Force this many data-driven levels instead of the "
                           "kneedle elbow (0 = auto).")
    fuse.add_argument("--levels", default="",
                      help="Comma-separated explicit levels, overriding selection.")

    refine = parser.add_argument_group("weaver / refiner")
    refine.add_argument("--weave_cutoff", type=float, default=0.75,
                        help="Containment threshold for the weaver. Default 0.75.")
    refine.add_argument("--min_system_size", type=int, default=4,
                        help="Minimum cluster size kept by the refiner. Default 4.")
    refine.add_argument("--jaccard_threshold", type=float, default=0.9,
                        help="Jaccard merge threshold for the refiner. Default 0.9.")
    refine.add_argument("--containment_threshold", type=float, default=0.75,
                        help="Containment threshold for the refiner. Default 0.75.")
    refine.add_argument("--min_diff", type=int, default=1,
                        help="Minimum level difference for the refiner. Default 1.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output.")
    return parser


def main(argv=None):
    """Parse arguments and run the pipeline."""
    args = build_parser().parse_args(argv)
    manual_levels = (
        [int(x) for x in args.levels.split(",")] if args.levels else None
    )
    run_pipeline(
        coembedding_path=args.coembedding,
        outdir=args.outdir,
        chi_min=args.chi_min,
        chi_max=args.chi_max,
        k=args.k,
        maxres=args.maxres,
        numthreads=args.numthreads,
        cutoffs=args.ppi_cutoffs,
        raw_nodes_path=args.raw_nodes,
        raw_edges_path=args.raw_edges,
        verbose=not args.quiet,
        k_max=args.k_max,
        forced_levels=args.n_levels,
        manual_levels=manual_levels,
        weave_cutoff=args.weave_cutoff,
        min_system_size=args.min_system_size,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
        min_diff=args.min_diff,
    )


if __name__ == "__main__":
    main()
