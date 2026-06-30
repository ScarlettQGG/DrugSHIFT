"""
joint_embed.py
=================
GNN-based joint embedding for EPIC PPI graphs (replaces the Word2Vec
pipeline used in v1-v4).

Why GNN
-------
v1-v4 trained per-condition Word2Vec embeddings and tried to align them
either via co-training tricks (anchor sentences, init_from_base, ref_anchor
tokens) or post-hoc Procrustes. Both struggle:

  * Heavy co-training collapses real biological signal — same-protein tokens
    are pulled together by an auxiliary loss whose entire job is to minimise
    cross-condition distance.
  * Light co-training + Procrustes leaves each W2V run in a random
    rotation/initialisation; orthogonal Procrustes can't reconcile per-token
    initialisation noise, so the neg-control delta floor doesn't drop to
    zero even when graphs are nearly identical.

A shared-parameter GNN encoder applied to all four graphs solves the goal
*structurally*:

  * The encoder reads the same learnable "identity" vector for protein P
    regardless of which graph is being processed.
  * The encoder weights are shared across all four graphs, so the
    *transformation* is identical.
  * Therefore: same protein + same neighborhood -> same output, by
    construction. No anchor sentences, no Procrustes, no init_from_base.
  * Different neighborhood -> different output, in proportion to the
    topology change.

No node features required: every protein gets a learnable embedding indexed
by a global protein-id table that is shared across all four graphs.

Outputs match v3/v4 layout so the existing eval_output2 pipeline works:
    EPIC_{cond}_1.tsv               per-condition embedding (N x D)
    delta_EPIC_{cond}_1.tsv         delta = treated - untreated (N x D + 2)
    delta_summary.tsv               per-condition delta_norm summary
    neg_control_snr_summary.tsv     SNR vs neg_ctrl
    aligned/{cond}_embedding.tsv    same as EPIC_{cond}_1.tsv
    config.json                     hyperparameters + sources
    joint_gnn.pt                    PyTorch state_dict + protein-id map

Usage
-----
    python -m two_stage.joint_embed \
        --base /path/to/analysis_base \
        --outdir output/joint_embed/cutoff_0.7 \
        --cutoff 0.7 \
        --epochs 300 \
        --d-out 128

Dependencies
------------
    torch >= 1.13   (CPU is fine; GPU optional via --device cuda)
    numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path / data loading helpers (mirrors v4 conventions)
# ---------------------------------------------------------------------------

Edge = Tuple[str, str]
EdgeScores = Dict[Edge, float]

DEFAULT_BASE_CANDIDATES = [
    Path("./data/analysis_base"),
    Path("/path/to/analysis_base"),
]


def resolve_default_base() -> Path:
    for base in DEFAULT_BASE_CANDIDATES:
        if base.exists():
            return base
    return DEFAULT_BASE_CANDIDATES[0]


def default_sources(base: Path) -> Dict[str, Path]:
    return {
        "untreated": base / "2.Interaction_prediction/EPIC_out_0/untreated/ctrl_rf.pred.txt",
        "cisplatin": base / "2.Interaction_prediction/EPIC_out_0/cisplatin/cis_rf.pred.txt",
        "vorinostat": base / "2.Interaction_prediction/EPIC_out_0/vorinostat/vor_rf.pred.txt",
        "neg_ctrl": base / "new_coemb/EPIC_negCTRL_23/EPIC_output/ctrl_rf.pred.txt",
    }


def parse_assignment(items: Optional[Sequence[str]]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def edge_key(a: str, b: str) -> Edge:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def load_epic_edges(
    path: Path,
    cutoff: float = 0.7,
    inclusive: bool = False,
    progress_every: int = 5_000_000,
) -> EdgeScores:
    """Stream an EPIC prediction TSV and keep only high-confidence undirected edges."""
    edges: EdgeScores = {}
    n_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                print(f"    read {n_lines:,} lines from {path.name}; kept {len(edges):,} edges")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            if (score < cutoff) if inclusive else (score <= cutoff):
                continue
            k = edge_key(parts[0], parts[1])
            prev = edges.get(k)
            if prev is None or score > prev:
                edges[k] = score
    return edges


# ---------------------------------------------------------------------------
# Replicate detection + per-graph noise diagnostics
# (Tier 1 modifications: replicate-aware training)
# ---------------------------------------------------------------------------

import re as _re

# Names ending in _rep1, _rep2, _r1, _r2, _replicate3, etc. → tagged as replicates.
_REP_SUFFIX_PATTERN = _re.compile(r"_(?:rep|r|replicate)(\d+)$", _re.IGNORECASE)


def split_replicate_tag(name: str) -> Tuple[str, Optional[int]]:
    """Return (base_condition, replicate_id) — replicate_id is None if no tag.

    'cisplatin_rep1' → ('cisplatin', 1)
    'untreated_r3'   → ('untreated', 3)
    'vorinostat'     → ('vorinostat', None)
    """
    m = _REP_SUFFIX_PATTERN.search(name)
    if m:
        base = name[: m.start()]
        rep_id = int(m.group(1))
        return base, rep_id
    return name, None


def group_replicates(graph_names: Sequence[str]) -> Dict[str, List[str]]:
    """Group input graph names by their base condition.

    Returns {base: [graph_name, ...]} sorted by replicate id (None first).
    Single-graph conditions get a list of length 1.
    """
    by_base: Dict[str, List[Tuple[Optional[int], str]]] = {}
    for n in graph_names:
        base, rid = split_replicate_tag(n)
        by_base.setdefault(base, []).append((rid, n))
    return {b: [n for _, n in sorted(items, key=lambda x: (x[0] is None, x[0]))]
            for b, items in by_base.items()}


def graph_noise_stats(edges_by_cond: Dict[str, EdgeScores]
                      ) -> Dict[str, Dict[str, float]]:
    """Per-graph noise diagnostic stats: n_edges, density, score median/var."""
    out: Dict[str, Dict[str, float]] = {}
    for name, ed in edges_by_cond.items():
        if not ed:
            out[name] = dict(n_edges=0, n_nodes=0, density=0.0,
                              score_med=float("nan"), score_var=float("nan"))
            continue
        scores = np.fromiter(ed.values(), dtype=np.float64)
        nodes = set()
        for u, v in ed:
            nodes.add(u); nodes.add(v)
        n_nodes = len(nodes)
        n_edges = len(ed)
        # Density = 2 * |E| / (n * (n-1))
        density = (2.0 * n_edges) / max(n_nodes * (n_nodes - 1), 1)
        out[name] = dict(
            n_edges  = int(n_edges),
            n_nodes  = int(n_nodes),
            density  = float(density),
            score_med = float(np.median(scores)),
            score_var = float(np.var(scores)),
        )
    return out


def print_graph_noise_diagnostics(stats: Dict[str, Dict[str, float]],
                                   replicate_groups: Dict[str, List[str]]
                                   ) -> None:
    """Print per-graph stats + intra-replicate comparisons. Flag outliers."""
    print()
    print("[v6 diag] per-graph noise statistics:")
    print(f"{'graph':<28} {'n_edges':>10} {'n_nodes':>8} {'density':>10} "
          f"{'score_med':>10} {'score_var':>10}")
    for name, st in sorted(stats.items()):
        print(f"{name:<28} {st['n_edges']:>10} {st['n_nodes']:>8} "
              f"{st['density']:>10.4f} {st['score_med']:>10.4f} {st['score_var']:>10.4f}")

    # Replicate-pair sanity check: flag graphs where same-condition replicates
    # differ in edge count by >2× (likely one replicate is broken).
    warned = False
    for base, names in replicate_groups.items():
        if len(names) < 2:
            continue
        ns = [stats[n]["n_edges"] for n in names if n in stats]
        if not ns:
            continue
        lo, hi = min(ns), max(ns)
        if lo > 0 and hi / lo > 2.0:
            if not warned:
                print()
                print("[v6 diag] WARNING — replicate-pair edge counts differ by >2×:")
                warned = True
            print(f"   base={base!r}  reps={names}  edges={ns} (ratio {hi/lo:.1f}×)")
            print(f"     → consider downweighting the noisy replicate or excluding "
                  f"it from training (--exclude-graph {names[ns.index(lo)] if hi/lo > 4 else '...'}).")
    print()


# ---------------------------------------------------------------------------
# Per-graph loss weighting (Tier 1.2)
# ---------------------------------------------------------------------------

def compute_graph_weights(graphs: Dict[str, "GraphData"],
                           stats: Dict[str, Dict[str, float]],
                           strategy: str = "uniform",
                           floor: float = 0.1,
                           ceiling: float = 3.0,
                           ) -> Dict[str, float]:
    """Compute per-graph loss weights for noise-robust training.

    strategy:
      'uniform'         — every graph weighted 1.0 (default; backward-compat)
      'sqrt_edges'      — w_g = sqrt(n_edges_g / median(n_edges))
                          → sparser graphs (likely noisier reps) get less weight
      'density'         — w_g = density_g / median(density)
      'inv_score_var'   — w_g = median(score_var) / score_var_g
                          → graphs with broader score distributions (more
                            low-confidence edges) get less weight

    Weights are clamped to [floor, ceiling] for stability.
    """
    names = list(graphs.keys())
    if strategy == "uniform":
        return {n: 1.0 for n in names}
    raw: Dict[str, float] = {}
    if strategy == "sqrt_edges":
        ns = np.asarray([max(stats[n]["n_edges"], 1) for n in names], dtype=np.float64)
        ref = float(np.median(ns))
        for n, x in zip(names, ns):
            raw[n] = float(np.sqrt(x / max(ref, 1.0)))
    elif strategy == "density":
        ds = np.asarray([max(stats[n]["density"], 1e-9) for n in names], dtype=np.float64)
        ref = float(np.median(ds))
        for n, x in zip(names, ds):
            raw[n] = float(x / max(ref, 1e-9))
    elif strategy == "inv_score_var":
        vs = np.asarray([max(stats[n]["score_var"], 1e-9) for n in names], dtype=np.float64)
        ref = float(np.median(vs))
        for n, x in zip(names, vs):
            raw[n] = float(ref / max(x, 1e-9))
    else:
        raise ValueError(f"unknown graph-weight strategy: {strategy!r}")
    return {n: float(min(ceiling, max(floor, w))) for n, w in raw.items()}


# ---------------------------------------------------------------------------
# Per-graph tensor container
# ---------------------------------------------------------------------------

class GraphData:
    """
    Tensors for one condition's PPI graph.

    Attributes
    ----------
    name              : condition label
    num_nodes         : total proteins in the GLOBAL index
                        (every graph runs over the same N rows so embeddings
                        for the same protein index correspond across graphs)
    present_idx       : [P_g] global indices of proteins present in this graph
    adj_edge_index    : [2, 2E] symmetric edges (both directions) for adjacency
    adj_edge_weight   : [2E] EPIC scores, repeated for both directions
    pos_edge_index    : [2, E]  unique positive edges for link-prediction loss
    pos_edge_weight   : [E]     EPIC scores for the positive edges
    """

    def __init__(
        self,
        name: str,
        num_nodes: int,
        present_idx: torch.Tensor,
        adj_edge_index: torch.Tensor,
        adj_edge_weight: torch.Tensor,
        pos_edge_index: torch.Tensor,
        pos_edge_weight: torch.Tensor,
    ):
        self.name = name
        self.num_nodes = num_nodes
        self.present_idx = present_idx
        self.adj_edge_index = adj_edge_index
        self.adj_edge_weight = adj_edge_weight
        self.pos_edge_index = pos_edge_index
        self.pos_edge_weight = pos_edge_weight
        # Tier 3: optional per-positive-edge F-statistic weight (None if disabled)
        self.pos_edge_f_weight: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "GraphData":
        self.present_idx = self.present_idx.to(device)
        self.adj_edge_index = self.adj_edge_index.to(device)
        self.adj_edge_weight = self.adj_edge_weight.to(device)
        self.pos_edge_index = self.pos_edge_index.to(device)
        self.pos_edge_weight = self.pos_edge_weight.to(device)
        if self.pos_edge_f_weight is not None:
            self.pos_edge_f_weight = self.pos_edge_f_weight.to(device)
        return self

    def normalized_adj(self) -> torch.Tensor:
        """
        Build the row-normalised weighted sparse adjacency [N, N] used for
        SAGE-style mean aggregation:

            A_norm[i, j] = weight(i, j) / sum_k weight(i, k)

        After (A_norm @ h), row i is the weighted mean of its neighbours' h.
        """
        n = self.num_nodes
        device = self.adj_edge_index.device
        row = self.adj_edge_index[0]
        col = self.adj_edge_index[1]
        deg = torch.zeros(n, device=device)
        deg.scatter_add_(0, row, self.adj_edge_weight)
        deg_inv = 1.0 / deg.clamp(min=1e-12)
        norm_w = self.adj_edge_weight * deg_inv[row]
        idx = torch.stack([row, col], dim=0)
        A = torch.sparse_coo_tensor(idx, norm_w, size=(n, n), device=device).coalesce()
        return A


def build_graphs(
    edges_by_cond: Dict[str, EdgeScores],
    protein_to_idx: Dict[str, int],
) -> Dict[str, GraphData]:
    n_global = len(protein_to_idx)
    out: Dict[str, GraphData] = {}
    for cond, edges in edges_by_cond.items():
        if not edges:
            continue
        # forward (unique) edges
        fwd_rows: List[int] = []
        fwd_cols: List[int] = []
        fwd_w: List[float] = []
        present: Set[int] = set()
        for (u, v), w in edges.items():
            iu = protein_to_idx[u]
            iv = protein_to_idx[v]
            fwd_rows.append(iu)
            fwd_cols.append(iv)
            fwd_w.append(float(w))
            present.add(iu)
            present.add(iv)

        # symmetric adjacency (both directions)
        adj_rows = fwd_rows + fwd_cols
        adj_cols = fwd_cols + fwd_rows
        adj_w_list = fwd_w + fwd_w

        adj_edge_index = torch.tensor([adj_rows, adj_cols], dtype=torch.long)
        adj_edge_weight = torch.tensor(adj_w_list, dtype=torch.float32)
        pos_edge_index = torch.tensor([fwd_rows, fwd_cols], dtype=torch.long)
        pos_edge_weight = torch.tensor(fwd_w, dtype=torch.float32)
        present_idx = torch.tensor(sorted(present), dtype=torch.long)

        out[cond] = GraphData(
            name=cond,
            num_nodes=n_global,
            present_idx=present_idx,
            adj_edge_index=adj_edge_index,
            adj_edge_weight=adj_edge_weight,
            pos_edge_index=pos_edge_index,
            pos_edge_weight=pos_edge_weight,
        )
    return out


# ---------------------------------------------------------------------------
# Edge dropout regularization (Tier 2)
# ---------------------------------------------------------------------------

def edge_dropout_graph(g: "GraphData", p: float) -> "GraphData":
    """Return a new GraphData with random fraction `p` of edges removed.

    Operates on `pos_edge_index` (unique undirected edges); rebuilds the
    symmetric `adj_edge_index` from the survivors so the GNN's mean
    aggregation reflects only the dropped graph.

    Applied per training step — so different epochs see different edge
    subsets, forcing the encoder to learn structure that is robust to
    edge perturbation (which is what replicate-to-replicate variance
    looks like at the graph level).

    Importantly:
      * Applied uniformly across all graphs each epoch → does NOT bias
        treated-vs-untreated separation.
      * `present_idx` is left untouched: a protein is still considered
        "present" in this graph even if its edges happen to be dropped
        in this particular epoch.  This avoids spurious changes to the
        per-graph protein set across epochs.
    """
    if p <= 0.0:
        return g
    n_pos = g.pos_edge_index.size(1)
    if n_pos == 0:
        return g
    device = g.pos_edge_index.device
    keep_mask = torch.rand(n_pos, device=device) > p
    if keep_mask.sum() == 0:
        # avoid pathological all-dropped graphs
        return g

    new_pos = g.pos_edge_index[:, keep_mask]
    new_pos_w = g.pos_edge_weight[keep_mask]

    fwd_rows = new_pos[0]
    fwd_cols = new_pos[1]
    adj_idx = torch.stack([
        torch.cat([fwd_rows, fwd_cols]),
        torch.cat([fwd_cols, fwd_rows]),
    ], dim=0)
    adj_w = torch.cat([new_pos_w, new_pos_w])

    # Optionally carry over the per-edge F-statistic weight (Tier 3)
    new_f_w = None
    if getattr(g, "pos_edge_f_weight", None) is not None:
        new_f_w = g.pos_edge_f_weight[keep_mask]

    new_g = GraphData(
        name=g.name,
        num_nodes=g.num_nodes,
        present_idx=g.present_idx,
        adj_edge_index=adj_idx,
        adj_edge_weight=adj_w,
        pos_edge_index=new_pos,
        pos_edge_weight=new_pos_w,
    )
    new_g.pos_edge_f_weight = new_f_w  # type: ignore[attr-defined]
    return new_g


# ---------------------------------------------------------------------------
# F-statistic edge upweighting (Tier 3) — cross-condition disagreeing edges
# ---------------------------------------------------------------------------
#
# For each candidate edge we observe a 9-vector of presence (1 if the edge
# appears in that replicate's graph, 0 otherwise) when we have 3×3 reps.
# We compute the ANOVA F-statistic for the null "presence is the same
# across conditions":
#
#     F = between-condition variance / within-condition variance
#
# Edges with high F are "real change markers" — consistently present in one
# condition's reps and consistently absent in another's.  Edges with F ≈ 1
# or lower vary as much WITHIN a condition as ACROSS conditions, i.e. they
# are dominated by replicate noise, not signal.
#
# We use F to multiplicatively upweight the recon-loss contribution of
# discriminating edges so the encoder spends its capacity where the Stage 2
# delta computation cares about.

def compute_edge_f_statistics(
    edges_by_cond: Dict[str, EdgeScores],
    replicate_groups: Dict[str, List[str]],
) -> Dict[Edge, float]:
    """Return {edge: F_statistic} over the union of edges across all graphs.

    Only base conditions with ≥2 replicates contribute (within-condition
    variance is undefined otherwise).  If fewer than 2 conditions have
    replicates, returns {edge: 1.0} (no boost) for all edges.
    """
    # Only conditions with ≥2 replicates participate in the F-test
    parts: Dict[str, List[str]] = {
        base: names for base, names in replicate_groups.items()
        if len(names) >= 2
    }
    all_edges: Set[Edge] = set()
    for ed in edges_by_cond.values():
        all_edges |= set(ed.keys())

    if len(parts) < 2:
        # Not enough condition diversity for ANOVA-style F
        return {e: 1.0 for e in all_edges}

    eps_within = 1e-3  # avoid /0 for edges with within-var = 0

    f_out: Dict[Edge, float] = {}
    # Pre-compute condition reps for fast inner loop
    cond_rep_names = [names for names in parts.values()]
    for e in all_edges:
        per_cond_presence: List[List[float]] = []
        for names in cond_rep_names:
            vals = [1.0 if e in edges_by_cond.get(n, {}) else 0.0 for n in names]
            per_cond_presence.append(vals)
        # Within-condition variance (pooled, equal-weight conditions)
        within_var = float(np.mean(
            [np.var(v, ddof=0) for v in per_cond_presence]
        ))
        cond_means = [float(np.mean(v)) for v in per_cond_presence]
        between_var = float(np.var(cond_means, ddof=0))
        f_out[e] = between_var / (within_var + eps_within)
    return f_out


def attach_f_weights_to_graphs(
    graphs: Dict[str, "GraphData"],
    f_statistics: Dict[Edge, float],
    protein_to_idx: Dict[str, int],
    strength: float,
    floor: float = 0.5,
    ceiling: float = 5.0,
) -> Dict[str, float]:
    """For each graph, attach a per-positive-edge F-statistic-derived weight.

    Multiplicative formula:

        f_norm  = (F + 1) / (median(F) + 1)      # ratio-of-shift
        weight  = clamp(f_norm ** strength, floor, ceiling)

    With `strength=0` everything stays 1.0 (no change).  With `strength=1`
    the boost is linear in f_norm (high-F edges up to ceiling× weight).

    Returns a summary dict of per-graph statistics (median F seen, fraction
    of edges with weight > 1, etc.).
    """
    if strength <= 0 or not f_statistics:
        for g in graphs.values():
            n = g.pos_edge_index.size(1)
            g.pos_edge_f_weight = torch.ones(n, dtype=torch.float32,  # type: ignore[attr-defined]
                                              device=g.pos_edge_index.device)
        return {g.name: 1.0 for g in graphs.values()}

    f_vals = np.fromiter(f_statistics.values(), dtype=np.float64)
    f_median = float(np.median(f_vals)) if f_vals.size else 0.0

    idx_to_protein: List[str] = [""] * len(protein_to_idx)
    for p, i in protein_to_idx.items():
        idx_to_protein[i] = p

    summary: Dict[str, float] = {}
    for cond, g in graphs.items():
        ei = g.pos_edge_index.cpu().numpy()
        ws: List[float] = []
        n_edges = ei.shape[1]
        for k in range(n_edges):
            a = idx_to_protein[int(ei[0, k])]
            b = idx_to_protein[int(ei[1, k])]
            ek = edge_key(a, b)
            f_v = f_statistics.get(ek, f_median)
            f_norm = (f_v + 1.0) / (f_median + 1.0)
            w = max(floor, min(ceiling, f_norm ** strength))
            ws.append(w)
        wt = torch.tensor(ws, dtype=torch.float32, device=g.pos_edge_index.device)
        g.pos_edge_f_weight = wt  # type: ignore[attr-defined]
        # diagnostic: median weight + fraction > 1
        summary[cond] = float(np.median(ws)) if ws else 1.0
    return summary


# ---------------------------------------------------------------------------
# Model: weighted GraphSAGE-mean (manual, no PyG dependency)
# ---------------------------------------------------------------------------

class WeightedSAGEConv(nn.Module):
    """
    GraphSAGE-style mean aggregation with a separate self-loop branch:

        y_i = W_self x_i + W_nbr (weighted mean of x_j over j in N(i))

    The "weighted mean" is computed via a sparse matmul against the
    pre-normalised adjacency A_norm.  No PyTorch Geometric dependency.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_nbr = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # A_norm @ x  -> [N, in_dim]; rows for nodes with no neighbours come
        # out as zero vectors, so the self branch still gives them an output.
        nbr = torch.sparse.mm(A_norm, x)
        return self.lin_self(x) + self.lin_nbr(nbr)


class SharedPPIEncoder(nn.Module):
    """
    Shared-parameter encoder applied to each condition's graph in turn.

    Input: a learnable identity embedding table indexed by global protein id.
           This is the only "feature" the model has — there are no per-node
           input features required.

    Forward(graph) returns a [N_global, d_out] embedding tensor.  Rows for
    proteins not present in this graph carry only the self-loop signal
    (no neighbour information) and are ignored at output extraction time.

    Tier-2 σ²-head
    --------------
    If `predict_sigma2=True`, the encoder also exposes a small MLP head that
    maps each protein's embedding to a scalar log-σ² (per-protein, per-graph).
    The head is supervised in train_one_epoch() via a Kendall heteroscedastic
    loss against the *observed* replicate pairwise distances.  Critically the
    z used to supervise the σ²-head is `.detach()`ed, so this loss can ONLY
    update the σ²-head parameters — it cannot bias z toward smaller replicate
    distances (which would re-introduce the σ²_raw bias we already removed).

    Stage 2 ingests sigma2_per_protein.tsv as the empirical noise floor in
    its own heteroscedastic Kendall loss + Bayesian raw/neighbour combiner.
    """

    def __init__(
        self,
        num_proteins: int,
        d_in: int = 64,
        d_hid: int = 128,
        d_out: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        normalize_output: bool = True,
        predict_sigma2: bool = False,
    ):
        super().__init__()
        self.identity = nn.Embedding(num_proteins, d_in)
        nn.init.normal_(self.identity.weight, std=0.1)

        self.dropout = dropout
        self.normalize_output = normalize_output
        self.predict_sigma2 = predict_sigma2

        dims = [d_in] + [d_hid] * (num_layers - 1) + [d_out]
        self.convs = nn.ModuleList([
            WeightedSAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

        if predict_sigma2:
            # Small MLP head: z -> log-σ² scalar per protein.
            # Initialise the last layer's bias around log(0.1) (≈ -2.3) so
            # the model starts with a moderate σ² guess instead of σ² ≈ 1.
            head_hidden = max(d_out // 2, 16)
            self.sigma_head = nn.Sequential(
                nn.Linear(d_out, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
            with torch.no_grad():
                self.sigma_head[-1].bias.fill_(-2.3)
        else:
            self.sigma_head = None

    def forward(self, A_norm: torch.Tensor,
                return_sigma2: bool = False):
        h = self.identity.weight  # [N, d_in]
        for i, conv in enumerate(self.convs):
            h = conv(h, A_norm)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        if self.normalize_output:
            h = F.normalize(h, p=2, dim=1)

        if return_sigma2 and self.sigma_head is not None:
            # IMPORTANT: feed the DETACHED z into the head.  Together with the
            # detach in the Kendall supervision loss in train_one_epoch(), this
            # guarantees the σ²-head can never influence where z lives — only
            # learn to predict the noise around it.
            log_s2 = self.sigma_head(h.detach()).squeeze(-1)
            return h, log_s2
        return h


# ---------------------------------------------------------------------------
# Training step and diagnostics
# ---------------------------------------------------------------------------

def train_one_epoch(
    encoder: SharedPPIEncoder,
    graphs: Dict[str, GraphData],
    optimizer: torch.optim.Optimizer,
    neg_per_pos: int = 5,
    score_weight_pow: float = 1.0,
    graph_weights: Optional[Dict[str, float]] = None,
    replicate_groups: Optional[Dict[str, List[str]]] = None,
    replicate_consistency_weight: float = 0.0,
    edge_dropout_p: float = 0.0,
    sigma2_head_weight: float = 0.0,
    apply_f_edge_weights: bool = False,
) -> Dict[str, float]:
    """
    One optimiser step over the SUM of per-condition link-prediction losses,
    optionally with:
      * `graph_weights[g]` — per-graph multiplicative weight on the recon loss
        (Tier 1.2 noise-robust training). Defaults to 1.0 per graph.
      * `replicate_groups[base] = [graph_name, ...]` + `replicate_consistency_weight`
        — Tier 1.3 auxiliary loss penalising cross-replicate embedding distance
        within the same biological condition. Defaults to 0 (off).

    For each graph G:
        pos_logit  = h_u . h_v  for each unique edge (u,v) in G
        pos_loss   = - mean_{edges} EPIC_score^pow * log sigmoid(pos_logit)
        neg_logit  = h_a . h_b  for random (a,b) drawn uniformly from
                     proteins PRESENT in G  (neg_per_pos negatives per positive)
        neg_loss   = - mean log sigmoid(- neg_logit)
        cond_loss = w_g * (pos_loss + neg_loss)

    Replicate-consistency loss (when enabled):
        For each base condition with ≥2 replicate graphs:
          L_repl_base = mean over rep-pairs (i,j) of  mean over shared proteins
                          of ‖h_repi[p] − h_repj[p]‖²
        Total auxiliary = replicate_consistency_weight · mean over bases of L_repl_base
    """
    encoder.train()
    optimizer.zero_grad()
    device = next(encoder.parameters()).device
    graph_weights = graph_weights or {}

    losses: Dict[str, float] = {}
    total = torch.zeros((), device=device)
    # Cache encoder outputs per graph so auxiliary losses (replicate
    # consistency, σ²-head supervision) can reuse them without a second
    # forward pass.
    h_cache: Dict[str, torch.Tensor] = {}
    log_sigma2_cache: Dict[str, torch.Tensor] = {}
    use_sigma2 = (sigma2_head_weight > 0
                    and encoder.predict_sigma2
                    and encoder.sigma_head is not None)

    for cond, g in graphs.items():
        # Tier 2: edge dropout — per-epoch random ablation of positive edges
        # (and their symmetric counterparts in the adjacency).  Uniformly
        # across all graphs each step, so treated/untreated separation is
        # unaffected.
        g_step = edge_dropout_graph(g, edge_dropout_p) if edge_dropout_p > 0 else g
        A = g_step.normalized_adj()
        if use_sigma2:
            h, log_s2 = encoder(A, return_sigma2=True)
            log_sigma2_cache[cond] = log_s2
        else:
            h = encoder(A)
        h_cache[cond] = h

        pos_src = g_step.pos_edge_index[0]
        pos_dst = g_step.pos_edge_index[1]
        pos_w = g_step.pos_edge_weight

        pos_logit = (h[pos_src] * h[pos_dst]).sum(dim=-1)
        # weight positives by EPIC score so 0.95-edges drive learning more
        # than barely-passing 0.71-edges
        w = pos_w.clamp(min=1e-3) ** score_weight_pow
        # Tier 3: F-statistic edge upweighting — multiply by precomputed
        # per-edge weight (already clamped to [floor, ceiling] at attach time)
        if apply_f_edge_weights and g_step.pos_edge_f_weight is not None:
            w = w * g_step.pos_edge_f_weight
        pos_loss = -(w * F.logsigmoid(pos_logit)).mean()

        present = g_step.present_idx
        n_neg = max(1, pos_src.size(0) * neg_per_pos)
        ni = torch.randint(0, present.size(0), (n_neg,), device=present.device)
        nj = torch.randint(0, present.size(0), (n_neg,), device=present.device)
        neg_src = present[ni]
        neg_dst = present[nj]
        neg_logit = (h[neg_src] * h[neg_dst]).sum(dim=-1)
        neg_loss = -F.logsigmoid(-neg_logit).mean()

        # Per-graph weight (default 1.0 if not specified)
        w_g = float(graph_weights.get(cond, 1.0))
        cond_loss = w_g * (pos_loss + neg_loss)
        total = total + cond_loss
        losses[cond] = float(cond_loss.detach().cpu())

    # --- Replicate-consistency auxiliary loss (Tier 1.3) ---
    repl_loss = torch.zeros((), device=device)
    n_pairs = 0
    if replicate_consistency_weight > 0 and replicate_groups:
        for base, names in replicate_groups.items():
            if len(names) < 2:
                continue
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ni_, nj_ = names[i], names[j]
                    if ni_ not in h_cache or nj_ not in h_cache:
                        continue
                    # Intersect present proteins between the two replicate graphs
                    s_i = set(graphs[ni_].present_idx.cpu().tolist())
                    s_j = set(graphs[nj_].present_idx.cpu().tolist())
                    common = sorted(s_i & s_j)
                    if len(common) < 2:
                        continue
                    idx = torch.tensor(common, device=device, dtype=torch.long)
                    d = h_cache[ni_][idx] - h_cache[nj_][idx]
                    repl_loss = repl_loss + (d.pow(2).sum(dim=-1).mean())
                    n_pairs += 1
        if n_pairs > 0:
            repl_loss = repl_loss / float(n_pairs)
            total = total + replicate_consistency_weight * repl_loss

    # --- σ²-head supervision (Tier 2) ---
    # Kendall heteroscedastic loss: 0.5 * d² * exp(-log_σ²) + 0.5 * log_σ²
    # supervised against observed per-protein pairwise replicate distance d².
    # Both h and log_σ² are DETACHED at d² computation so the gradient only
    # flows into the σ²-head parameters — never into the embedding.
    sigma2_loss = torch.zeros((), device=device)
    n_sigma2_pairs = 0
    if use_sigma2 and replicate_groups:
        for base, names in replicate_groups.items():
            if len(names) < 2:
                continue
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ni_, nj_ = names[i], names[j]
                    if (ni_ not in h_cache or nj_ not in h_cache
                            or ni_ not in log_sigma2_cache
                            or nj_ not in log_sigma2_cache):
                        continue
                    s_i = set(graphs[ni_].present_idx.cpu().tolist())
                    s_j = set(graphs[nj_].present_idx.cpu().tolist())
                    common = sorted(s_i & s_j)
                    if len(common) < 2:
                        continue
                    idx = torch.tensor(common, device=device, dtype=torch.long)
                    # Observed pairwise squared distance per protein.
                    # DETACH h so this loss never reaches the encoder weights.
                    h_i_d = h_cache[ni_][idx].detach()
                    h_j_d = h_cache[nj_][idx].detach()
                    d2 = (h_i_d - h_j_d).pow(2).sum(dim=-1)  # [P_common]
                    # Average log-σ² from the two reps (gradient flows here)
                    log_s2 = 0.5 * (log_sigma2_cache[ni_][idx]
                                     + log_sigma2_cache[nj_][idx])
                    # Bound log-σ² to prevent numerical blowup
                    log_s2 = log_s2.clamp(-10.0, 5.0)
                    kendall = 0.5 * d2 * torch.exp(-log_s2) + 0.5 * log_s2
                    sigma2_loss = sigma2_loss + kendall.mean()
                    n_sigma2_pairs += 1
        if n_sigma2_pairs > 0:
            sigma2_loss = sigma2_loss / float(n_sigma2_pairs)
            total = total + sigma2_head_weight * sigma2_loss

    total.backward()
    optimizer.step()
    losses["total"]            = float(total.detach().cpu())
    losses["repl_consistency"] = float(repl_loss.detach().cpu())
    losses["n_replicate_pairs"] = float(n_pairs)
    losses["sigma2_head"]      = float(sigma2_loss.detach().cpu())
    losses["n_sigma2_pairs"]   = float(n_sigma2_pairs)
    return losses


@torch.no_grad()
def cross_condition_diagnostics(
    encoder: SharedPPIEncoder,
    graphs: Dict[str, GraphData],
    ref: str,
) -> Dict[str, float]:
    """
    Mean cosine(P_in_ref, P_in_cond) across proteins shared between ref and
    each other condition.

    Expectation if the model is doing the right thing:
        cos(untreated, neg_ctrl) -> close to 1     (technical replicates)
        cos(untreated, cisplatin) -> meaningfully < 1, with biology-driven tails
        cos(untreated, vorinostat) -> meaningfully < 1
    A flat 1.0 across everything = condition collapse.
    Floor near 0 = no shared geometry yet (early training).
    """
    encoder.eval()
    embs: Dict[str, torch.Tensor] = {
        cond: encoder(g.normalized_adj()) for cond, g in graphs.items()
    }
    if ref not in embs:
        return {}
    ref_h = embs[ref]
    ref_set = set(graphs[ref].present_idx.cpu().tolist())

    out: Dict[str, float] = {}
    for cond, h in embs.items():
        if cond == ref:
            continue
        cond_set = set(graphs[cond].present_idx.cpu().tolist())
        common = sorted(ref_set & cond_set)
        if not common:
            continue
        idx = torch.tensor(common, device=ref_h.device)
        a = F.normalize(ref_h[idx], dim=1)
        b = F.normalize(h[idx], dim=1)
        cos = (a * b).sum(dim=1).cpu().numpy()
        out[f"cos_mean_{cond}"] = float(cos.mean())
        out[f"cos_med_{cond}"] = float(np.median(cos))
    return out


# ---------------------------------------------------------------------------
# Embedding extraction, deltas, summaries
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_per_condition_embeddings(
    encoder: SharedPPIEncoder,
    graphs: Dict[str, GraphData],
    idx_to_protein: List[str],
) -> Dict[str, pd.DataFrame]:
    encoder.eval()
    out: Dict[str, pd.DataFrame] = {}
    for cond, g in graphs.items():
        A = g.normalized_adj()
        h = encoder(A).cpu().numpy()
        present = g.present_idx.cpu().numpy()
        arr = h[present]
        proteins = [idx_to_protein[i] for i in present]
        cols = [f"dim_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, index=proteins, columns=cols)
        df.index.name = "protein"
        out[cond] = df
    return out


@torch.no_grad()
def extract_per_condition_sigma2(
    encoder: SharedPPIEncoder,
    graphs: Dict[str, GraphData],
    idx_to_protein: List[str],
) -> Dict[str, pd.DataFrame]:
    """Per-condition per-protein σ²-head outputs.

    Returns {cond: DataFrame[protein, log_sigma2, sigma2]} for each input
    graph.  Only meaningful if the encoder was trained with the σ²-head.
    Stage 2 ingests the ref-condition's prediction (or the median across
    conditions) as the noise-floor prior in its Kendall loss.
    """
    if not encoder.predict_sigma2 or encoder.sigma_head is None:
        return {}
    encoder.eval()
    out: Dict[str, pd.DataFrame] = {}
    for cond, g in graphs.items():
        A = g.normalized_adj()
        _, log_s2 = encoder(A, return_sigma2=True)
        log_s2 = log_s2.detach().cpu().numpy()
        sigma2 = np.exp(log_s2)
        present = g.present_idx.cpu().numpy()
        proteins = [idx_to_protein[i] for i in present]
        df = pd.DataFrame({
            "log_sigma2": log_s2[present],
            "sigma2":     sigma2[present],
        }, index=proteins)
        df.index.name = "protein"
        out[cond] = df
    return out


def compute_delta_df(
    embs: Dict[str, pd.DataFrame],
    ref: str,
    target: str,
) -> pd.DataFrame:
    if ref not in embs or target not in embs:
        return pd.DataFrame()
    R = embs[ref]
    T = embs[target]
    if R.empty or T.empty:
        return pd.DataFrame()
    common = sorted(set(R.index) & set(T.index))
    if not common:
        return pd.DataFrame()
    X = R.loc[common].values.astype(float)
    Y = T.loc[common].values.astype(float)
    D = Y - X
    cols = [f"dim_{i}" for i in range(D.shape[1])]
    df = pd.DataFrame(D, index=common, columns=cols)
    df["delta_norm"] = np.linalg.norm(D, axis=1)
    df["ref_norm"] = np.linalg.norm(X, axis=1)
    df.index.name = "protein"
    return df


def summarize_deltas(deltas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cond, df in deltas.items():
        if df.empty:
            rows.append({"condition": cond, "n": 0})
            continue
        v = df["delta_norm"].values.astype(float)
        rows.append({
            "condition": cond,
            "n": int(len(v)),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95)),
            "std": float(v.std()),
        })
    return pd.DataFrame(rows).set_index("condition")


def summarize_snr(
    deltas: Dict[str, pd.DataFrame],
    treats: Sequence[str],
    neg: str,
) -> pd.DataFrame:
    if neg not in deltas or deltas[neg].empty:
        return pd.DataFrame()
    n = deltas[neg]
    n_med = float(n["delta_norm"].median())
    n_mean = float(n["delta_norm"].mean())
    rows = []
    for t in treats:
        if t not in deltas or deltas[t].empty:
            continue
        d = deltas[t]
        common = sorted(set(d.index) & set(n.index))
        if common:
            excess = d.loc[common, "delta_norm"] - n.loc[common, "delta_norm"]
        else:
            excess = pd.Series(dtype=float)
        rows.append({
            "condition": t,
            "n": int(len(d)),
            "n_paired_with_neg": int(len(common)),
            "raw_median": float(d["delta_norm"].median()),
            "neg_median": n_med,
            "median_ratio_vs_neg": float(d["delta_norm"].median() / max(n_med, 1e-12)),
            "mean_ratio_vs_neg": float(d["delta_norm"].mean() / max(n_mean, 1e-12)),
            "paired_excess_median": float(excess.median()) if len(excess) else float("nan"),
            "paired_excess_mean": float(excess.mean()) if len(excess) else float("nan"),
            "pct_paired_excess_gt0": float((excess > 0).mean()) if len(excess) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("condition")


def calibrate_excess_deltas(
    deltas: Dict[str, pd.DataFrame],
    treats: Sequence[str],
    neg: str,
    n_degree_bins: int = 8,
    weighted_degree: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Per-protein neg-control calibrated ranking.  Mirrors the v4 calibration:

        excess_delta_norm   = treatment_delta_norm - matched_neg_delta_norm
        delta_z_global_neg  = (treatment_delta_norm - median(neg)) / mad(neg)
        delta_z_degree_bin_neg : same z but neg statistics computed within
                                 degree-bins of the untreated weighted degree
                                 (so high-degree proteins are calibrated
                                 against high-degree neg-control proteins).
    """
    if neg not in deltas or deltas[neg].empty:
        return {t: pd.DataFrame() for t in treats}

    neg_df = deltas[neg][["delta_norm"]].rename(columns={"delta_norm": "matched_neg_delta_norm"})
    neg_vals = deltas[neg]["delta_norm"].astype(float).values
    neg_med = float(np.nanmedian(neg_vals))
    neg_mad = max(1.4826 * float(np.nanmedian(np.abs(neg_vals - neg_med))), 1e-9)

    if weighted_degree is not None:
        deg_df = pd.DataFrame.from_dict(weighted_degree, orient="index", columns=["ref_weighted_degree"])
        deg_df.index.name = "protein"
    else:
        deg_df = None

    out: Dict[str, pd.DataFrame] = {}
    for t in treats:
        if t not in deltas or deltas[t].empty:
            out[t] = pd.DataFrame()
            continue
        df = deltas[t][["delta_norm", "ref_norm"]].copy()
        df = df.rename(columns={"delta_norm": "raw_delta_norm"})
        df = df.join(neg_df, how="left")
        df["excess_delta_norm"] = df["raw_delta_norm"] - df["matched_neg_delta_norm"]
        df["global_neg_median"] = neg_med
        df["global_neg_mad"] = neg_mad
        df["delta_z_global_neg"] = (df["raw_delta_norm"] - neg_med) / neg_mad

        if deg_df is not None and not deg_df.empty:
            df = df.join(deg_df, how="left")
            try:
                neg_with_deg = (
                    deltas[neg][["delta_norm"]]
                    .join(deg_df, how="left")
                    .dropna(subset=["ref_weighted_degree"])
                )
                if len(neg_with_deg) >= n_degree_bins * 10:
                    neg_with_deg["degree_bin"] = pd.qcut(
                        neg_with_deg["ref_weighted_degree"],
                        q=n_degree_bins,
                        duplicates="drop",
                    )
                    bin_stats = (
                        neg_with_deg.groupby("degree_bin", observed=True)["delta_norm"]
                        .agg(bin_neg_median="median", bin_neg_count="count")
                    )
                    bin_stats["bin_neg_mad"] = neg_with_deg.groupby(
                        "degree_bin", observed=True
                    )["delta_norm"].apply(
                        lambda x: max(1.4826 * float(np.nanmedian(np.abs(x.values - np.nanmedian(x.values)))), 1e-9)
                    )
                    df["degree_bin"] = pd.cut(
                        df["ref_weighted_degree"],
                        bins=[iv.left for iv in bin_stats.index] + [bin_stats.index[-1].right],
                        include_lowest=True,
                    )
                    df = df.join(bin_stats, on="degree_bin")
                    df["delta_z_degree_bin_neg"] = (
                        df["raw_delta_norm"] - df["bin_neg_median"]
                    ) / df["bin_neg_mad"]
            except (ValueError, AttributeError):
                df["delta_z_degree_bin_neg"] = df["delta_z_global_neg"]

        sort_cols = [c for c in ["delta_z_degree_bin_neg", "excess_delta_norm", "raw_delta_norm"] if c in df.columns]
        df = df.sort_values(sort_cols, ascending=False)
        df.index.name = "protein"
        out[t] = df
    return out


def weighted_degree_table(edges: EdgeScores) -> Dict[str, float]:
    deg: Dict[str, float] = {}
    for (u, v), w in edges.items():
        deg[u] = deg.get(u, 0.0) + float(w)
        deg[v] = deg.get(v, 0.0) + float(w)
    return deg


# ---------------------------------------------------------------------------
# Seed alignment (only needed if --n-seeds > 1)
# ---------------------------------------------------------------------------

def align_seed_to_reference(
    seed_embs: Dict[str, pd.DataFrame],
    ref_seed_embs: Dict[str, pd.DataFrame],
    ref_condition: str,
) -> Dict[str, pd.DataFrame]:
    """
    Rotate every condition's embedding from this seed into seed-0's frame
    using orthogonal Procrustes estimated on the ref-condition embedding.

    Within ONE training run, the four conditions already share a frame
    (shared encoder weights), so this rotation is only needed across runs.
    """
    from scipy.linalg import orthogonal_procrustes

    if ref_condition not in seed_embs or ref_condition not in ref_seed_embs:
        return seed_embs
    A_df = seed_embs[ref_condition]
    B_df = ref_seed_embs[ref_condition]
    common = sorted(set(A_df.index) & set(B_df.index))
    if len(common) < 10:
        return seed_embs
    A = A_df.loc[common].values.astype(float)
    B = B_df.loc[common].values.astype(float)
    R, _ = orthogonal_procrustes(A, B)

    rotated: Dict[str, pd.DataFrame] = {}
    for cond, df in seed_embs.items():
        if df.empty:
            rotated[cond] = df
            continue
        rotated[cond] = pd.DataFrame(
            df.values.astype(float) @ R,
            index=df.index, columns=df.columns,
        )
        rotated[cond].index.name = "protein"
    return rotated


def average_seed_frames(
    seed_aligned_list: List[Dict[str, pd.DataFrame]],
    conditions: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for cond in conditions:
        frames = [s[cond] for s in seed_aligned_list
                  if cond in s and not s[cond].empty]
        if not frames:
            out[cond] = pd.DataFrame()
            continue
        common = sorted(set.intersection(*(set(f.index) for f in frames)))
        if not common:
            out[cond] = pd.DataFrame()
            continue
        arr = np.stack([f.loc[common].values.astype(float) for f in frames], axis=0).mean(axis=0)
        df = pd.DataFrame(arr, index=common, columns=frames[0].columns)
        df.index.name = "protein"
        out[cond] = df
    return out


# ---------------------------------------------------------------------------
# Single training run (one seed)
# ---------------------------------------------------------------------------

def train_single_seed(
    graphs: Dict[str, GraphData],
    n_proteins: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    idx_to_protein: List[str],
    graph_weights: Optional[Dict[str, float]] = None,
    replicate_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, pd.DataFrame], SharedPPIEncoder]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    encoder = SharedPPIEncoder(
        num_proteins=n_proteins,
        d_in=args.d_in,
        d_hid=args.d_hid,
        d_out=args.d_out,
        num_layers=args.num_layers,
        dropout=args.dropout,
        normalize_output=not args.no_normalize_output,
        predict_sigma2=(getattr(args, "sigma2_head_weight", 0.0) > 0),
    ).to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    apply_f_edge_weights = (getattr(args, "edge_change_weight_strength", 0.0) > 0)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        losses = train_one_epoch(
            encoder, graphs, optimizer,
            neg_per_pos=args.neg_per_pos,
            score_weight_pow=args.score_weight_pow,
            graph_weights=graph_weights,
            replicate_groups=replicate_groups,
            replicate_consistency_weight=getattr(args, "replicate_consistency_weight", 0.0),
            edge_dropout_p=getattr(args, "edge_dropout_p", 0.0),
            sigma2_head_weight=getattr(args, "sigma2_head_weight", 0.0),
            apply_f_edge_weights=apply_f_edge_weights,
        )
        if ep == 1 or ep % args.diag_every == 0 or ep == args.epochs:
            diag = cross_condition_diagnostics(encoder, graphs, ref=args.ref_cond)
            diag_str = "  ".join(f"{k}={v:+.3f}" for k, v in sorted(diag.items()))
            extra = ""
            if losses.get("n_replicate_pairs", 0) > 0:
                extra += f"  repl={losses['repl_consistency']:.4f} (pairs={int(losses['n_replicate_pairs'])})"
            if losses.get("n_sigma2_pairs", 0) > 0:
                extra += f"  σ²-head={losses['sigma2_head']:.4f}"
            print(f"    epoch {ep:>4d}/{args.epochs}  loss={losses['total']:.4f}{extra}  {diag_str}")
    print(f"    seed done in {time.time() - t0:.1f}s")

    embs = extract_per_condition_embeddings(encoder, graphs, idx_to_protein)
    return embs, encoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    base = resolve_default_base()
    parser = argparse.ArgumentParser(
        description="GNN-based shared-parameter joint embedding for EPIC PPI graphs (v6)."
    )
    parser.add_argument("--base", type=Path, default=base,
                        help="Base directory containing EPIC outputs.")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Optional NAME=PATH overrides for individual conditions.")
    parser.add_argument("--ref-cond", default="untreated")
    parser.add_argument("--neg-cond", default="neg_ctrl")
    parser.add_argument("--treat-conds", nargs="+", default=["cisplatin", "vorinostat"])
    parser.add_argument("--cutoff", type=float, default=0.7,
                        help="Edge score cutoff: only edges with score > cutoff are kept (use --inclusive-cutoff for >=).")
    parser.add_argument("--inclusive-cutoff", action="store_true")
    parser.add_argument("--outdir", type=Path,
                        default=Path("output/joint_embed/cutoff_0.7"))

    # model architecture
    parser.add_argument("--d-in", type=int, default=64,
                        help="Identity embedding dim (input to GNN).")
    parser.add_argument("--d-hid", type=int, default=128,
                        help="Hidden dim between GNN layers.")
    parser.add_argument("--d-out", type=int, default=128,
                        help="Output embedding dim.")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of GNN layers.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-normalize-output", action="store_true",
                        help="Disable L2 normalisation of output embeddings.")

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--neg-per-pos", type=int, default=5,
                        help="Negative samples per positive edge.")
    parser.add_argument("--score-weight-pow", type=float, default=1.0,
                        help="Exponent applied to EPIC score when weighting positives in the loss.")
    parser.add_argument("--diag-every", type=int, default=25,
                        help="Print cross-condition cosine diagnostics every K epochs.")
    parser.add_argument("--device", default="cpu",
                        help="torch device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seed runs to train and average. >1 enables Procrustes seed alignment.")

    # calibration
    parser.add_argument("--degree-bins", type=int, default=8,
                        help="Degree bins for neg-control z-score calibration.")

    # ---- Tier 1 replicate-aware training (opt-in) ----
    parser.add_argument("--graph-weight-strategy",
                        choices=["uniform", "sqrt_edges", "density", "inv_score_var"],
                        default="uniform",
                        help="Per-graph loss-weighting strategy. 'uniform' (default) "
                             "leaves v6 behaviour unchanged. 'sqrt_edges' / 'density' / "
                             "'inv_score_var' downweight sparser / noisier graphs to "
                             "prevent shared-encoder contamination from bad replicates.")
    parser.add_argument("--graph-weight-floor",   type=float, default=0.1,
                        help="Lower clamp on per-graph weight; default 0.1.")
    parser.add_argument("--graph-weight-ceiling", type=float, default=3.0,
                        help="Upper clamp on per-graph weight; default 3.0.")
    parser.add_argument("--replicate-consistency-weight", type=float, default=0.0,
                        help="Auxiliary loss coefficient pulling same-condition "
                             "replicate embeddings together (Tier 1.3). 0 = off. "
                             "Try 0.1 when training with replicate-tagged sources.")
    parser.add_argument("--exclude-graph", nargs="*", default=None,
                        help="Names of graphs to drop from training before any "
                             "weighting; useful for catastrophic single-replicate "
                             "failures detected by the startup noise diagnostic.")

    # ---- Tier 2 / Tier 3 improvements for Stage 2 coupling ----
    parser.add_argument("--edge-dropout-p", type=float, default=0.0,
                        help="Tier 2 regularization: random fraction of positive "
                             "edges dropped per training step. Applied uniformly "
                             "across all graphs (does not bias treated vs untreated "
                             "separation). 0.0 = off; 0.1 is the recommended start.")
    parser.add_argument("--sigma2-head-weight", type=float, default=0.0,
                        help="Tier 2: weight of the σ²-head Kendall loss. 0 = "
                             "off; set 0.1 to learn a per-protein log-σ² for "
                             "Stage 2 to ingest as the empirical noise floor. "
                             "Loss is detached from the embedding so this can "
                             "only update the σ² head, never bias z.")
    parser.add_argument("--edge-change-weight-strength", type=float, default=0.0,
                        help="Tier 3: strength of F-statistic-based edge upweight "
                             "in the recon loss. 0 = off; 0.5 = mild, 1.0 = full. "
                             "Up-weights edges with high (between-cond / within-cond) "
                             "variance — i.e. real treatment-discriminating edges. "
                             "Requires ≥2 replicates in ≥2 conditions to be useful.")
    parser.add_argument("--edge-change-weight-floor", type=float, default=0.5,
                        help="Lower clamp on per-edge F-statistic weight.")
    parser.add_argument("--edge-change-weight-ceiling", type=float, default=5.0,
                        help="Upper clamp on per-edge F-statistic weight.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("joint_embed: shared-parameter GNN encoder for EPIC PPI graphs")
    print("=" * 78)
    print(f"  base:         {args.base}")
    print(f"  outdir:       {args.outdir}")
    print(f"  cutoff:       {'>=' if args.inclusive_cutoff else '>'} {args.cutoff}")
    print(f"  ref:          {args.ref_cond}")
    print(f"  neg_ctrl:     {args.neg_cond}")
    print(f"  treatments:   {args.treat_conds}")
    print(f"  arch:         d_in={args.d_in} d_hid={args.d_hid} d_out={args.d_out} "
          f"layers={args.num_layers} dropout={args.dropout}")
    print(f"  train:        epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
          f"neg/pos={args.neg_per_pos} score_pow={args.score_weight_pow}")
    print(f"  device:       {device}  seeds={args.n_seeds}")

    # -------------------------------------------------------------------
    # Load EPIC edges — replicate-aware
    # -------------------------------------------------------------------
    sources_all = default_sources(args.base)
    sources_all.update(parse_assignment(args.sources))

    # For each required base condition, prefer replicate-tagged variants
    # (e.g. untreated_rep1, untreated_rep3) when any are supplied, and fall
    # back to the exact-name match only when none exist. This prevents the
    # exact-name default_sources() path (the cluster-only location) from being
    # pulled in alongside locally-supplied per-replicate --sources.
    required_bases = [args.ref_cond, args.neg_cond, *args.treat_conds]
    sources: Dict[str, "Path"] = {}
    for base in required_bases:
        # 1) replicate-tagged matches (collect ALL matching variants)
        rep_variants = {name: p for name, p in sources_all.items()
                        if split_replicate_tag(name)[1] is not None
                        and split_replicate_tag(name)[0] == base}
        if rep_variants:
            for name, p in rep_variants.items():
                sources.setdefault(name, p)
        # 2) exact-name match — only when no replicate variants were supplied
        elif base in sources_all:
            sources.setdefault(base, sources_all[base])
    # Optional user-requested graph exclusions (e.g. catastrophic single-rep failures)
    if args.exclude_graph:
        for ex in args.exclude_graph:
            if ex in sources:
                print(f"[v6] excluding graph by user request: {ex}")
                del sources[ex]
    # Sanity: at least one graph per required base
    found_bases = {split_replicate_tag(n)[0] for n in sources}
    missing = [b for b in required_bases if b not in found_bases]
    if missing:
        raise ValueError(f"missing source paths for conditions (no exact or "
                         f"replicate-tagged match): {missing}")
    for cond, p in sources.items():
        if not p.exists():
            raise FileNotFoundError(f"{cond}: {p}")

    print(f"\n[1/5] Loading EPIC edges (cutoff {'>=' if args.inclusive_cutoff else '>'} {args.cutoff})")
    edges_by_cond: Dict[str, EdgeScores] = {}
    for cond, path in sources.items():
        t0 = time.time()
        edges = load_epic_edges(path, cutoff=args.cutoff, inclusive=args.inclusive_cutoff)
        nodes = {n for e in edges for n in e}
        print(f"  {cond:>10}: {len(edges):,} edges, {len(nodes):,} nodes  "
              f"({time.time() - t0:.1f}s)  -> {path}")
        edges_by_cond[cond] = edges

    # -------------------------------------------------------------------
    # Global protein index
    # -------------------------------------------------------------------
    all_proteins = sorted({n for edges in edges_by_cond.values() for e in edges for n in e})
    if not all_proteins:
        print("ERROR: no proteins found above cutoff in any condition.", file=sys.stderr)
        return 2
    protein_to_idx = {p: i for i, p in enumerate(all_proteins)}
    idx_to_protein = list(all_proteins)
    n_proteins = len(all_proteins)
    print(f"  global protein index: {n_proteins:,} proteins (union of all conditions)")

    # save the protein index alongside the model so downstream can map back
    pd.DataFrame({"protein": idx_to_protein}).to_csv(
        args.outdir / "protein_index.tsv", sep="\t",
        index=True, index_label="idx",
    )

    # -------------------------------------------------------------------
    # Per-graph noise diagnostic + replicate grouping (Tier 1 modifications)
    # -------------------------------------------------------------------
    replicate_groups = group_replicates(list(edges_by_cond.keys()))
    noise_stats      = graph_noise_stats(edges_by_cond)
    print_graph_noise_diagnostics(noise_stats, replicate_groups)
    if any(len(v) > 1 for v in replicate_groups.values()):
        print("[v6] replicate groups detected:")
        for base, names in replicate_groups.items():
            if len(names) > 1:
                print(f"   {base}: {names}")

    # -------------------------------------------------------------------
    # Build per-graph tensors
    # -------------------------------------------------------------------
    print("\n[2/5] Building per-graph tensors")
    graphs = build_graphs(edges_by_cond, protein_to_idx)
    for cond, g in graphs.items():
        graphs[cond] = g.to(device)
        print(f"  {cond:>10}: {g.present_idx.size(0):,} present, "
              f"{g.pos_edge_index.size(1):,} undirected edges")

    # -------------------------------------------------------------------
    # Per-graph loss weights (Tier 1.2)
    # -------------------------------------------------------------------
    graph_weights = compute_graph_weights(
        graphs, noise_stats,
        strategy=args.graph_weight_strategy,
        floor=args.graph_weight_floor,
        ceiling=args.graph_weight_ceiling,
    )
    if args.graph_weight_strategy != "uniform":
        print(f"\n[v6] per-graph loss weights ({args.graph_weight_strategy}):")
        for n, w in sorted(graph_weights.items()):
            print(f"   {n:<28} w={w:.3f}")
    if args.replicate_consistency_weight > 0:
        n_rep_bases = sum(1 for v in replicate_groups.values() if len(v) > 1)
        print(f"[v6] replicate-consistency loss enabled: "
              f"weight={args.replicate_consistency_weight}, "
              f"{n_rep_bases} base condition(s) with replicates")

    if args.edge_dropout_p > 0:
        print(f"[v6] edge dropout enabled: p={args.edge_dropout_p}")
    if args.sigma2_head_weight > 0:
        print(f"[v6] σ²-head enabled: weight={args.sigma2_head_weight} "
              f"(detached from z; only supervises noise prediction)")

    # -------------------------------------------------------------------
    # F-statistic edge upweighting (Tier 3)
    # -------------------------------------------------------------------
    if args.edge_change_weight_strength > 0:
        n_multi_rep = sum(1 for v in replicate_groups.values() if len(v) >= 2)
        if n_multi_rep >= 2:
            print(f"\n[v6] F-statistic edge upweighting enabled "
                  f"(strength={args.edge_change_weight_strength}, "
                  f"floor={args.edge_change_weight_floor}, "
                  f"ceiling={args.edge_change_weight_ceiling})")
            t0 = time.time()
            f_stats = compute_edge_f_statistics(edges_by_cond, replicate_groups)
            f_summary = attach_f_weights_to_graphs(
                graphs, f_stats, protein_to_idx,
                strength=args.edge_change_weight_strength,
                floor=args.edge_change_weight_floor,
                ceiling=args.edge_change_weight_ceiling,
            )
            print(f"   computed F for {len(f_stats):,} edges  ({time.time()-t0:.1f}s)")
            print(f"   per-graph median weight: " + ", ".join(
                f"{c}={w:.3f}" for c, w in sorted(f_summary.items())
            ))
            # Save F-statistic diagnostic
            f_vals = np.fromiter(f_stats.values(), dtype=np.float64)
            quants = np.quantile(f_vals, [0.1, 0.5, 0.9, 0.99]) if f_vals.size else [0,0,0,0]
            print(f"   F quantiles (q10/q50/q90/q99): "
                  f"{quants[0]:.3f} / {quants[1]:.3f} / "
                  f"{quants[2]:.3f} / {quants[3]:.3f}")
            f_rows = []
            for e, fv in f_stats.items():
                f_rows.append({"a": e[0], "b": e[1], "f_stat": float(fv)})
            pd.DataFrame(f_rows).to_csv(
                args.outdir / "edge_f_statistics.tsv", sep="\t", index=False,
            )
        else:
            print("[v6] F-statistic upweighting requested but only "
                  f"{n_multi_rep} condition(s) have ≥2 replicates → skipping "
                  "(need ≥2 conditions with replicates for the F-test).")

    # -------------------------------------------------------------------
    # Train (optionally ensemble across seeds)
    # -------------------------------------------------------------------
    print(f"\n[3/5] Training shared GNN encoder ({args.n_seeds} seed(s))")
    seed_embs_list: List[Dict[str, pd.DataFrame]] = []
    last_encoder: Optional[SharedPPIEncoder] = None
    for s in range(args.n_seeds):
        seed = args.seed + s * 1000
        print(f"\n  seed {s + 1}/{args.n_seeds} (torch seed={seed})")
        embs, encoder = train_single_seed(
            graphs=graphs,
            n_proteins=n_proteins,
            args=args,
            seed=seed,
            device=device,
            idx_to_protein=idx_to_protein,
            graph_weights=graph_weights,
            replicate_groups=replicate_groups,
        )
        seed_embs_list.append(embs)
        last_encoder = encoder

    print("\n[4/5] Aggregating across seeds")
    if len(seed_embs_list) == 1:
        embs = seed_embs_list[0]
    else:
        ref_seed = seed_embs_list[0]
        aligned_seeds = [ref_seed]
        for k, s in enumerate(seed_embs_list[1:], start=1):
            print(f"  rotating seed {k+1} -> seed 1 frame (Procrustes on {args.ref_cond})")
            aligned_seeds.append(align_seed_to_reference(s, ref_seed, args.ref_cond))
        embs = average_seed_frames(aligned_seeds, conditions=list(graphs.keys()))
        # also save per-seed aligned outputs for inspection
        per_seed_dir = args.outdir / "seed_aligned"
        per_seed_dir.mkdir(exist_ok=True)
        for i, sd in enumerate(aligned_seeds):
            d = per_seed_dir / f"seed_{i}"
            d.mkdir(exist_ok=True)
            for cond, df in sd.items():
                if not df.empty:
                    df.to_csv(d / f"{cond}_embedding.tsv", sep="\t",
                              index=True, index_label="protein")

    # -------------------------------------------------------------------
    # Per-base-condition aggregates (replicate averaging)
    # -------------------------------------------------------------------
    # With replicate-tagged inputs (e.g. untreated_rep1/rep3) there is no single
    # base-condition graph, but the delta / calibration / σ² code below keys off
    # base conditions (args.ref_cond, args.treat_conds). Synthesize per-base
    # aggregates so those steps work: embeddings = mean of the replicate
    # unit-vector embeddings (NOT renormalized — matches the documented
    # convention), edges = mean score per edge across replicates.
    for _base, _members in group_replicates(list(embs.keys())).items():
        _reps = [m for m in _members if split_replicate_tag(m)[1] is not None]
        if _base in embs or not _reps:
            continue
        _frames = [embs[m] for m in _reps if m in embs and not embs[m].empty]
        if not _frames:
            continue
        _common = sorted(set.intersection(*(set(f.index) for f in _frames)))
        if not _common:
            continue
        _stack = np.stack([f.loc[_common].values.astype(float) for f in _frames], axis=0)
        _df = pd.DataFrame(_stack.mean(axis=0), index=_common, columns=_frames[0].columns)
        _df.index.name = "protein"
        embs[_base] = _df

    for _base, _members in group_replicates(list(edges_by_cond.keys())).items():
        _reps = [m for m in _members if split_replicate_tag(m)[1] is not None]
        if _base in edges_by_cond or not _reps:
            continue
        _sum: Dict[Edge, float] = {}
        _cnt: Dict[Edge, int] = {}
        for m in _reps:
            for e, s in edges_by_cond[m].items():
                _sum[e] = _sum.get(e, 0.0) + float(s)
                _cnt[e] = _cnt.get(e, 0) + 1
        edges_by_cond[_base] = {e: _sum[e] / _cnt[e] for e in _sum}

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------
    print("\n[5/5] Writing embeddings, deltas, summaries")

    aligned_dir = args.outdir / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    for cond, df in embs.items():
        if df.empty:
            continue
        df.to_csv(aligned_dir / f"{cond}_embedding.tsv", sep="\t",
                  index=True, index_label="protein")
        # also at the v3/v4 top-level filename pattern so eval_output2 reads it
        df.to_csv(args.outdir / f"EPIC_{cond}_1.tsv", sep="\t",
                  index=True, index_label="protein")

    deltas: Dict[str, pd.DataFrame] = {}
    for cond in embs:
        if cond == args.ref_cond:
            continue
        d = compute_delta_df(embs, args.ref_cond, cond)
        deltas[cond] = d
        if not d.empty:
            d.to_csv(args.outdir / f"delta_EPIC_{cond}_1.tsv", sep="\t",
                     index=True, index_label="protein")

    summary = summarize_deltas(deltas)
    summary.to_csv(args.outdir / "delta_summary.tsv", sep="\t")
    print("\nDelta summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    snr = summarize_snr(deltas, args.treat_conds, args.neg_cond)
    if not snr.empty:
        snr.to_csv(args.outdir / "neg_control_snr_summary.tsv", sep="\t")
        print("\nNeg-control SNR summary:")
        print(snr.to_string(float_format=lambda x: f"{x:.4f}"))

    # neg-control calibrated rankings (mirrors v4)
    ref_wdeg = weighted_degree_table(edges_by_cond[args.ref_cond])
    # treatments: calibrated against neg_ctrl
    calibrated = calibrate_excess_deltas(
        deltas, treats=args.treat_conds, neg=args.neg_cond,
        n_degree_bins=args.degree_bins, weighted_degree=ref_wdeg,
    )
    # neg_ctrl: self-calibrated (z-scores against its own degree-bin null).
    # Useful as a diagnostic — z distribution should peak near 0 with std ~1
    # if calibration is well-behaved — and required by downstream evaluators
    # that expect a calibrated_delta_<neg_cond>.tsv to exist for every condition.
    if args.neg_cond in deltas and not deltas[args.neg_cond].empty:
        neg_self = calibrate_excess_deltas(
            deltas, treats=[args.neg_cond], neg=args.neg_cond,
            n_degree_bins=args.degree_bins, weighted_degree=ref_wdeg,
        )
        calibrated.update(neg_self)
    for cond, df in calibrated.items():
        if not df.empty:
            df.to_csv(args.outdir / f"calibrated_delta_{cond}.tsv", sep="\t",
                      index=True, index_label="protein")

    # --- σ²-head outputs (Tier 2): per-protein learned noise estimate ---
    # Stage 2 ingests sigma2_per_protein.tsv as the empirical floor in its
    # Kendall heteroscedastic loss + Bayesian raw/neighbour combiner.
    if (last_encoder is not None
            and last_encoder.predict_sigma2
            and last_encoder.sigma_head is not None):
        sigma2_by_cond = extract_per_condition_sigma2(
            last_encoder, graphs, idx_to_protein,
        )
        # Synthesize per-base-condition σ² (mean log σ² across replicates) so the
        # canonical sigma2_per_protein.tsv (keyed off args.ref_cond) is written
        # even when only replicate-tagged graphs were supplied.
        for _base, _members in group_replicates(list(sigma2_by_cond.keys())).items():
            _reps = [m for m in _members if split_replicate_tag(m)[1] is not None]
            if _base in sigma2_by_cond or not _reps:
                continue
            _frames = [sigma2_by_cond[m] for m in _reps if m in sigma2_by_cond]
            if not _frames:
                continue
            _common = sorted(set.intersection(*(set(f.index) for f in _frames)))
            if not _common:
                continue
            _ls = np.stack([f.loc[_common, "log_sigma2"].values.astype(float)
                            for f in _frames], axis=0).mean(axis=0)
            _bdf = pd.DataFrame({"log_sigma2": _ls, "sigma2": np.exp(_ls)}, index=_common)
            _bdf.index.name = "protein"
            sigma2_by_cond[_base] = _bdf
        sigma2_dir = args.outdir / "sigma2"
        sigma2_dir.mkdir(exist_ok=True)
        # Per-condition outputs (diagnostic; varies by graph context)
        for cond, df in sigma2_by_cond.items():
            df.to_csv(sigma2_dir / f"sigma2_{cond}.tsv", sep="\t",
                       index=True, index_label="protein")
        # Canonical per-protein noise estimate = ref-condition's prediction
        # (the noise floor a protein has in its baseline state)
        if args.ref_cond in sigma2_by_cond:
            ref_sigma2 = sigma2_by_cond[args.ref_cond].copy()
            # If we have ref replicates, additionally average their σ² so the
            # canonical estimate isn't tied to one specific rep's graph
            ref_reps = [n for n in graphs
                          if split_replicate_tag(n)[0] == args.ref_cond
                          and split_replicate_tag(n)[1] is not None]
            if ref_reps:
                frames = [sigma2_by_cond[n] for n in ref_reps if n in sigma2_by_cond]
                if frames:
                    common = sorted(set.intersection(*(set(f.index) for f in frames)))
                    if common:
                        log_s2_stack = np.stack(
                            [f.loc[common, "log_sigma2"].values.astype(float)
                             for f in frames], axis=0)
                        ref_sigma2 = pd.DataFrame({
                            "log_sigma2": log_s2_stack.mean(axis=0),
                            "sigma2":     np.exp(log_s2_stack.mean(axis=0)),
                        }, index=common)
                        ref_sigma2.index.name = "protein"
            ref_sigma2.to_csv(args.outdir / "sigma2_per_protein.tsv", sep="\t",
                                index=True, index_label="protein")
            # Summary
            s = ref_sigma2["sigma2"].values
            print(f"\n[v6] σ²-head per-protein summary (ref-cond avg over "
                  f"{len(frames) if 'frames' in locals() and frames else 1} reps):")
            print(f"   n={len(s)}  median σ²={float(np.median(s)):.4f}  "
                  f"q10={float(np.quantile(s, 0.1)):.4f}  "
                  f"q90={float(np.quantile(s, 0.9)):.4f}")

    # save model + protein index map
    if last_encoder is not None:
        torch.save({
            "state_dict": last_encoder.state_dict(),
            "protein_to_idx": protein_to_idx,
            "config": {
                "d_in": args.d_in, "d_hid": args.d_hid, "d_out": args.d_out,
                "num_layers": args.num_layers, "dropout": args.dropout,
                "normalize_output": not args.no_normalize_output,
                "n_proteins": n_proteins,
            },
        }, args.outdir / "joint_gnn.pt")

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["sources"] = {k: str(v) for k, v in sources.items()}
    config["n_proteins"] = n_proteins
    with (args.outdir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(f"\n[done] outputs in {args.outdir}")
    print(f"  Key files (drop into your existing eval_output2 pipeline):")
    print(f"    EPIC_{args.ref_cond}_1.tsv               aligned ref embedding")
    print(f"    EPIC_<treatment>_1.tsv                aligned treatment embeddings")
    print(f"    delta_EPIC_<treatment>_1.tsv          per-protein delta vs {args.ref_cond}")
    print(f"    delta_summary.tsv                     per-condition delta_norm summary")
    print(f"    neg_control_snr_summary.tsv           SNR vs {args.neg_cond}")
    print(f"    calibrated_delta_<treatment>.tsv      neg-control calibrated rankings")
    print(f"    joint_gnn.pt                          model + protein-id map")
    print(f"    config.json                           hyperparameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
