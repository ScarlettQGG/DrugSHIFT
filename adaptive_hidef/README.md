# Adaptive_hidef

Build an **adaptive protein-cluster hierarchy** from a protein embedding.

The pipeline turns a single embedding table into a pruned containment hierarchy in three stages:

1. **PPI networks** — pairwise cosine similarity of the embedding vectors
   defines a weighted graph; a ladder of increasingly sparse networks is formed
   by keeping the top *cutoff* fraction of edges at several cutoffs.
2. **HiDeF** — one low-`k` HiDeF run scans a range of Leiden resolutions over the
   multiplex of those networks and weaves the recurring communities into a
   multi-resolution hierarchy, recording each community's *persistence* (the
   highest `k`/chi at which it survives).
3. **Adaptive fusion** — communities are kept by a **size-graded persistence
   threshold**: large systems are admitted even when only moderately stable
   (preserving a deep, narrow top), while small communities must be highly stable
   to survive (keeping curated complexes, not noise, at the leaves). The kept
   communities are re-woven and pruned into the final hierarchy.

The threshold levels are chosen from the data (1-D k-means on the persistences +
a kneedle elbow), so only two numbers shape the result: `chi_min` (top) and
`chi_max` (bottom, a precision/recall dial).

## Requirements

- Python ≥ 3.9
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`
- [`hidef`](https://pypi.org/project/hidef/) == 1.1.5 (provides `hidef_finder.py`
  and the weaver)
- [`cellmaps_generate_hierarchy`](https://pypi.org/project/cellmaps-generate-hierarchy/)
  == 0.3.0 (provides the hierarchy refiner)
- `igraph` ≥ 0.10, `leidenalg` ≥ 0.9 (community detection back end)

See `requirements.txt` for exact pins.

## Installation

```bash
pip install -e .
```

This installs the `adaptive_hidef` package and the `adaptive-hidef`
command-line tool. (`hidef` and `cellmaps_generate_hierarchy` must be importable
in the same environment.)

## Input

A single **tab-separated embedding table** with a header row:

| protein | d0     | d1     | ... |
|---------|--------|--------|-----|
| A2M     | 0.013  | -0.021 | ... |
| AACS    | -0.004 | 0.009  | ... |

The first column is the protein/gene symbol; the remaining columns are float features. Row order defines the integer member ids used throughout the output.

## Usage

```bash
adaptive-hidef --coembedding coembedding_emd.tsv --outdir results/
```

Adjust the precision/recall trade-off with the fusion dial:

```bash
# more clusters / higher coverage
adaptive-hidef -i coembedding_emd.tsv -o results/ --chi_min 5 --chi_max 10

# fewer clusters / higher precision
adaptive-hidef -i coembedding_emd.tsv -o results/ --chi_min 5 --chi_max 15
```

Reuse a pre-computed HiDeF run (skip the expensive stages 1–2) and only re-fuse:

```bash
adaptive-hidef -i coembedding_emd.tsv -o results/ \
    --raw_nodes results/hidef_raw/hidef_output.nodes \
    --raw_edges results/hidef_raw/hidef_output.edges
```

The HiDeF community-detection run (stage 2) is single-threaded and takes on the order of an hour for a proteome-scale co-embedding; stages 1 and 3 finish in seconds.

### Python API

```python
from adaptive_hidef import run_pipeline

run_pipeline(
    coembedding_path="coembedding_emd.tsv",
    outdir="results/",
    chi_min=5,
    chi_max=12,
)
```

## Parameters

| flag | default | meaning |
|------|---------|---------|
| `--chi_min` | 5 | persistence required for the **largest** communities (top/backbone) |
| `--chi_max` | 12 | persistence required for the **smallest** communities (leaf complexes); the **precision/recall dial** — lower keeps more clusters, higher keeps fewer |
| `--k` | 5 | HiDeF persistence threshold for the community-detection run |
| `--maxres` | 80 | maximum Leiden resolution scanned by HiDeF |
| `--numthreads` | 1 | worker threads for the resolution scan |
| `--ppi_cutoffs` | 0.001 … 0.10 | edge-retention fractions, one PPI network per value |
| `--k_max` | 6 | maximum number of k-means levels scanned during selection |
| `--n_levels` | 0 | force this many data-driven levels instead of the kneedle elbow |
| `--levels` | — | comma-separated explicit levels, overriding data-driven selection |
| `--min_system_size` | 4 | minimum cluster size kept by the refiner |
| `--weave_cutoff`, `--jaccard_threshold`, `--containment_threshold`, `--min_diff` | 0.75 / 0.9 / 0.75 / 1 | standard weaver / refiner knobs |

## Output

Written under `--outdir`:

| file | description |
|------|-------------|
| `hidef_output.pruned.nodes` | **final hierarchy** — one cluster per line: `ClusterID`, size, space-separated member ids, persistence |
| `hidef_output.pruned.edges` | **final hierarchy** — `parent`, `child`, `type` (a DAG; a child may have multiple parents) |
| `hierarchy_parent.cx2` | member id (co-embedding row) → gene symbol, resolving the integer members |
| `fusion_stats.tsv` | structure summary (node/leaf/depth counts) of the final hierarchy vs. the raw HiDeF source |
| `leaf_size_distribution.png` | leaf-size histogram (dpi 300) |
| `level_selection.tsv` | k-means inertia / centres per k, the data behind the level choice |

Intermediate files are retained:

| path | description |
|------|-------------|
| `ppi/ppi_cutoff_*.edgelist.tsv` | the cosine PPI networks, one per cutoff |
| `hidef_raw/hidef_output.nodes` / `.edges` | raw HiDeF community-detection output (before fusion) |
| `hidef_output.nodes` / `.edges` / `.weaver` | the fused communities after weaving, before pruning |

**Member ids** are integer row indices into the input co-embedding;
`Cluster0-0` is the root (the whole protein universe). Resolve ids to gene
symbols with `hierarchy_parent.cx2`.

## Package layout

```
adaptive_hidef/
    ppi.py           # co-embedding -> cosine PPI edgelists at a ladder of cutoffs
    hidef_runner.py  # run hidef_finder.py over the multiplex of PPI networks
    fusion.py        # size-graded persistence selection, weaving, diagnostics
    pipeline.py      # orchestrates the three stages
    cli.py           # command-line interface
```
