"""adaptive_hidef: build an adaptive protein-cluster hierarchy from a
co-embedding.

Pipeline: cosine-similarity PPI networks at a ladder of edge cutoffs -> a single
low-``k`` HiDeF multi-resolution community-detection run -> adaptive,
size-graded persistence fusion into one pruned containment hierarchy.
"""

from .pipeline import fuse_hierarchy, run_pipeline

__all__ = ["run_pipeline", "fuse_hierarchy"]
__version__ = "1.0.0"
