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

---

## Pipeline

```
Per-replicate, per-condition co-elution PPI graphs (e.g. EPIC on SEC-MS)
        │
        ▼  two_stage.joint_embed
  Layer 1 · GNN  — shared-parameter GraphSAGE + shared identity table
                      → cross-condition-aligned per-protein EPIC embeddings
        │
        ▼  Stage 1  (two_stage.model.MUSEStage1)
  Layer 2 · MUSE Stage 1 — mask-aware multimodal autoencoder
                           (EPIC + AP-MS + image + sequence)
                      → static_latent.tsv  (the reference cell map)
        │
        ▼  Stage 2  (two_stage.model.NeighborhoodAdapter)
  Layer 3 · Stage 2 adapter — neighbourhood-aware, coherence-weighted
                              delta over the static map, per perturbation
                      → z_treat.tsv, learned_magnitude.tsv, direction modules
```

The two stages live in one flat `two_stage/` package and share code
(`model.py`, `losses.py`, `train.py`). Train either stage or both with
`python -m two_stage.train --stage {1,2,both}` (default both).

---

## The three layers

### Layer 1 — GNN EPIC encoder (`two_stage.joint_embed`)
A **shared-parameter** GraphSAGE encoder applied to *every* per-replicate
co-elution PPI graph, with a **shared learnable identity table** (one vector
per protein, common to all graphs). The identity table is what aligns
embeddings across conditions/replicates *without* post-hoc Procrustes
rotation: same protein index → same identity vector → same output position
modulo the neighbourhood-driven shift, **and that shift is the treatment
delta**. Output is L2-normalised; an optional per-protein σ²-head estimates
replicate noise (detached from the embedding so it can never bias it).

### Layer 2 — Stage 1 multimodal map (`two_stage.model.MUSEStage1`)
Stage 1 builds the static reference cell map. It is a mask-aware multimodal
autoencoder: per-modality encoders embed EPIC, AP-MS, image and sequence, a
fusion network combines them into one joint co-embedding, and per-modality
decoders reconstruct each input from that joint z. Key properties:
- **Union (not intersection) protein coverage** via mask-aware fusion — a
  protein present in only one modality still gets a valid embedding.
- **Modality dropout** at the encoder input — forces cross-modal prediction,
  so the map is robust to inference-time missingness.
- **Kendall homoscedastic uncertainty weighting** — automatically balances
  modalities by difficulty.

Produces `static_latent.tsv`: the L2-normalised joint co-embedding used as
the reference map (the "anchor") for Stage 2.

### Layer 3 — Stage 2 neighbourhood adapter (`two_stage.model.NeighborhoodAdapter`)
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

# Layers 2 + 3 — train BOTH stages end to end (recommended config).
# Stage 1 -> <outdir>/stage1 ; per-condition adapters -> <outdir>/stage2/<cond>
python -m two_stage.train --stage both \
    --manifest       <manifest.json> \
    --outdir         <outdir> \
    --epic_name      epic \
    --cond_names     cisplatin vorinostat negative_ctrl \
    --sigma2_epic_path <empirical_sigma2.tsv> \
    --sigma2_raw_scale 0.1 --sigma2_raw_floor 0.001 \
    --w_residual 1.0 --w_epic 10.0 \
    --spherical --unified --drift_remove \
    --n_epochs 300 --lr 1e-3

# ...or train a single stage:
#   python -m two_stage.train --stage 1 --manifest <m> --outdir <stage1_dir>
#   python -m two_stage.train --stage 2 --stage1_outdir <stage1_dir> \
#       --manifest <m> --conditions cisplatin --outdir <stage2_dir> [recommended flags]

python -m two_stage.inference --help          # writes z_treat.tsv, learned_magnitude.tsv
python -m two_stage.eval --help               # CORUM remodelling, cluster transitions, HPA shift
python -m two_stage.direction_modules --help  # coordinated remodelling modules + GO:BP
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

## Repository layout

Everything is one flat, self-contained `two_stage/` package — both stages
share `model.py` / `losses.py` / `train.py`, and Stage 2 loads the frozen
Stage 1 model from the same package, with no external dependency.

```
two_stage/
  joint_embed.py        Layer 1 · GNN co-elution encoder (produces the EPIC modality)
  model.py              MUSEStage1 (Stage 1) + NeighborhoodAdapter (Stage 2)
  losses.py             Stage 1 recon/triplet losses + Stage 2 LOO/decoder-stability losses
  train.py              unified CLI — `--stage {1,2,both}` (default both)
  cache.py              loads the frozen Stage-1 model + builds the cluster-aware kNN
  inference.py          apply a trained adapter → z_treat / learned_magnitude
  eval.py               CORUM remodelling / cluster transitions / HPA shift
  direction_modules.py  coordinated remodelling modules + GO:BP enrichment
  dropout.py            modality dropout (Stage 1)
  pseudo_labels.py      Leiden/KMeans pseudo-labels for the structure loss
docs/                   PIPELINE.md (manifest + worked run) + design/diagnosis notes
```

## Citation

## License

MIT — see [LICENSE](LICENSE).
