# two-stage — perturbation-aware multimodal cell mapping

A two-stage model for learning how a **perturbation remodels the protein
interactome** of a cell, at the protein-complex level, from multimodal data.

Stage 1 builds a high-confidence **static multimodal cell map** (a joint
co-embedding of every protein from co-elution PPIs, AP-MS, imaging and
sequence). Stage 2 learns, **per perturbation**, how that map *remodels* —
which proteins move, in which coordinated directions, and with what
confidence — using the static map as a denoising prior over a noisy
treatment signal.

The worked example throughout is U2OS osteosarcoma cells treated with two
chemotherapeutics — **cisplatin** (DNA-damage agent) and **vorinostat**
(HDAC inhibitor) — with SEC-MS co-elution as the perturbed modality. The
method itself is agnostic to cell type, perturbation, and which modality
carries the perturbation signal.

Method foundation: Schaffer, Hu, Qian et al., *Nature* 2025 — "Multimodal
cell maps as a foundation for structural and functional genomics" (which
builds the *static* map). This repository adds the **perturbation-aware**
layer on top.

---

## Pipeline

```
Per-replicate, per-condition co-elution PPI graphs (e.g. EPIC on SEC-MS)
        │
        ▼  joint_embed.py
  Layer 1 · GNN  — shared-parameter GraphSAGE + shared identity table
                      → cross-condition-aligned per-protein EPIC embeddings
        │
        ▼  two_stage/stage1/
  Layer 2 · MUSE Stage 1 — mask-aware multimodal autoencoder
                           (EPIC + AP-MS + image + sequence)
                      → static_latent.tsv  (the reference cell map)
        │
        ▼  two_stage/stage2/
  Layer 3 · Stage 2 adapter — neighbourhood-aware, coherence-weighted
                              delta over the static map, per perturbation
                      → z_treat.tsv, learned_magnitude.tsv, direction modules
```

Each layer is a standalone, importable module; the three compose into the
full pipeline.

---

## The three layers

### Layer 1 — GNN EPIC encoder (`joint_embed.py`)
A **shared-parameter** GraphSAGE encoder applied to *every* per-replicate
co-elution PPI graph, with a **shared learnable identity table** (one vector
per protein, common to all graphs). The identity table is what aligns
embeddings across conditions/replicates *without* post-hoc Procrustes
rotation: same protein index → same identity vector → same output position
modulo the neighbourhood-driven shift, **and that shift is the treatment
delta**. Output is L2-normalised; an optional per-protein σ²-head estimates
replicate noise (detached from the embedding so it can never bias it).

### Layer 2 — MUSE Stage 1 multimodal map (`two_stage/stage1/`)
A MUSE-style multimodal autoencoder with three additions over the published
static map:
- **Union (not intersection) protein coverage** via mask-aware fusion — a
  protein with only one modality still gets a valid embedding.
- **Modality dropout** at the encoder input — forces cross-modal prediction,
  robust to inference-time missingness.
- **Kendall homoscedastic uncertainty weighting** — auto-balances modalities
  by difficulty.

Produces `static_latent.tsv`: the L2-normalised joint co-embedding used as
the reference map (the "anchor") for Stage 2.

### Layer 3 — Stage 2 neighbourhood adapter (`two_stage/stage2/`)
Per perturbation, learns how the static map remodels. Core ideas:

- **Leave-one-out neighbourhood prediction** — each protein's treatment delta
  is predicted from its kNN neighbours' deltas *alone*, never from its own
  observation. Identity mapping is structurally impossible; the adapter can
  only learn signal that propagates coherently through related proteins.
- **Cluster-aware kNN** over the static map, with edge weights gated by
  per-protein anchor reliability and a soft Leiden cluster-compatibility term
  (suppresses cross-complex bleed).
- **Unified coherence-weighted movement** (recommended) — a *single*
  operation co-derives magnitude and direction:
  `δ = max(0, cos(δ_obs, δ̂)) · δ_obs`, on the tangent plane of the unit
  sphere and with global drift removed. Neighbours agree → keep the move;
  disagree (noise) → it vanishes. This is what yields a genuine **stable
  protein population** plus a noise-decoupled remodelled tail, rather than
  hallucinated movement everywhere.

Outputs per condition: `z_treat.tsv` (remodelled map, drop-in for the static
latent), `learned_magnitude.tsv` (per-protein remodelling score — rank by
this), `direction modules` (coordinated remodelling, via
`direction_modules.py`), and per-protein uncertainty.

---

## Install

```bash
git clone git@github.com:ScarlettQGG/two-stage.git
cd two-stage
pip install -r requirements.txt        # numpy, pandas, scikit-learn, scipy, torch, requests
                                       # python-igraph + leidenalg are optional (KMeans fallback)
```

Tested with Python 3.10, PyTorch 2.2. No GPU required for the example sizes
(~6k proteins).

---

## Quick start

The three layers run in sequence. Paths and modality inputs are described in
a `manifest.json` (see [`docs/PIPELINE.md`](docs/PIPELINE.md) for the schema
and a full worked invocation).

```bash
# Layer 1 — encode each co-elution PPI graph into aligned per-protein embeddings
python -m two_stage.joint_embed --help

# Layer 2 — fuse modalities into the static reference map
python -m two_stage.stage1.runner --help          # writes static_latent.tsv

# Layer 3 — learn the per-perturbation remodelling (recommended config)
python -m two_stage.stage2.training \
    --stage1_outdir  <stage1_outdir> \
    --manifest       <manifest.json> \
    --epic_name      epic \
    --condition      cisplatin \
    --cond_names     cisplatin vorinostat negative_ctrl \
    --sigma2_epic_path <empirical_sigma2.tsv> \
    --sigma2_raw_scale 0.1 --sigma2_raw_floor 0.001 \
    --w_residual 1.0 --w_epic 10.0 \
    --spherical --unified --drift_remove \
    --n_epochs 300 --lr 1e-3

python -m two_stage.stage2.inference --help    # writes z_treat.tsv, learned_magnitude.tsv
python -m two_stage.stage2.eval        --help     # CORUM remodelling, cluster transitions, HPA shift
python -m two_stage.stage2.direction_modules --help   # coordinated remodelling modules + GO:BP
```

Always run a **negative control** condition (e.g. a held-out untreated
replicate pair) through Stage 2: it should produce ≈ no remodelling. This is
the primary guard against hallucinated signal.

---

## What "good" looks like (the U2OS example)

- **Stage 1 map** recovers known biology: CORUM-complex co-membership
  AUROC ≈ 0.83, STRING physical-interaction AUROC ≈ 0.82.
- **Negative control** shows no hallucination: per-protein remodelling
  magnitude median ≈ 0, ~1 CORUM complex remodelled (vs tens for drugs).
- **On-target MOA recovery** in the top remodelled proteins: cisplatin →
  oxidative-stress + DNA-repair (PRDX2/3, ATM, BRIP1, DSB-repair-via-HR);
  vorinostat → HDAC8 (direct target), NCOR1, transcription/ubiquitin/autophagy.
- **Coordinated remodelling**: proteins moving together share pathways
  (direction modules are FDR-significant), and the remodelled proteins are
  exactly those that recluster while the large-scale map structure is retained.
- The two-stage differential enriches for drug MOA **better** than either the
  raw co-elution feature differential or the GNN-embedding-subtraction
  differential.

---

## Honest limitations

This pipeline is **replicate-limited**. With only two usable replicates per
condition, the co-elution PPI signal is noisy (RF edge-score reproducibility
r ≈ 0.36; strong-edge persistence ≈ 3%), even though the underlying
co-elution *features* reproduce well (r ≈ 0.84). The unified, drift-removed
Stage 2 model is the most honest readout obtainable at this replicate depth —
treat per-complex calls as **hypotheses** to validate, not conclusions. The
single thing that moves the ceiling is **more replicates**.

The design history — the hallucination failure mode, every fix tried, and the
breakthrough — is documented in
[`two_stage/stage2/docs/`](two_stage/stage2/docs/).

---

## Repository layout

Everything lives in a single self-contained `two_stage/` package — Stage 2
imports the frozen Stage 1 model directly from `two_stage.stage1`, with no
dependency on any external package.

```
two_stage/
  joint_embed.py           Layer 1 · GNN co-elution encoder (the EPIC modality)
  stage1/                  Layer 2 · MUSE multimodal static map
    model.py  train.py  runner.py  losses.py  dropout.py  pseudo_labels.py
    vae/                   experimental VAE variant (not recommended)
  stage2/                  Layer 3 · perturbation adapter (the core)
    architecture.py          NeighborhoodAdapter
    training.py              per-condition training
    inference.py             apply a trained adapter
    losses.py                Kendall LOO + decoder-stability losses
    stage1_cache.py          loads the frozen Stage-1 model + builds kNN
    direction_modules.py     coordinated remodelling modules + GO:BP enrichment
    eval.py                  CORUM remodelling / cluster transitions / HPA shift
    docs/                    design rationale, failure analysis, readouts
docs/PIPELINE.md           manifest schema + full worked invocation
```

## Citation

If you use this code, please cite the foundation paper (Schaffer, Hu, Qian
et al., *Nature* 2025) and link this repository.

## License

MIT — see [LICENSE](LICENSE).
