# `muse_stage1` — MUSE-style Stage 1 with semi-hard negatives

A drop-in alternative Stage-1 multimodal anchor for the two-stage co-embedding pipeline. Produces `static_latent.tsv` in the same format as the existing Stage 1, so downstream Stage 2 / hierarchy code consumes it unchanged.

## What's different vs. the existing Stage 1

| | Existing Stage 1 | `muse_stage1` |
|---|---|---|
| Joint anchor | **Mean** of L2-normed per-modality latents (lossy) | **Learned fusion** (concat with mask bits → MLP → joint z) |
| Reconstruction | Cross-modal recon from per-modality latents | Recon from joint z, per modality (forces z to retain every modality) |
| Triplet | "Same protein in two modalities" (identity-based) | **Within-modality pseudo-label triplet in joint z** (MUSE) — cluster each modality, pull same-cluster proteins together, push different-cluster apart |
| Negative mining | Random roll-permutation | **Semi-hard** (FaceNet) — hardest positive + smallest negative still farther than positive |
| Modality balance | Dissimilarity-weighted sampling on cross-modal triplet | **Kendall homoscedastic uncertainty** — learns log-σ per modality per loss term |
| Missing modalities | Masked in averaging | **Zero-fill + presence mask bits**, masked recon/triplet — no protein is wasted |
| Cross-modal alignment | Cross-modal identity triplet (`same protein in A and B should be close`) with dissimilarity-balanced sampling | **Modality dropout during training** — hide modalities at the encoder input; the decoder must still reconstruct them. Implicitly forces cross-modal prediction + missing-data robustness, replacing the explicit alignment triplet |

## Files

| | |
|---|---|
| `model.py`         | `MUSEStage1`: per-modality encoders + concat-fusion + per-modality decoders, Kendall log-σ params |
| `pseudo_labels.py` | Per-modality clustering (Leiden cosine kNN, KMeans fallback); cached to JSON |
| `losses.py`        | `masked_recon_loss` (recon-from-z), `semi_hard_triplet`, `structure_triplet_loss`, `total_loss` |
| `train.py`         | Data loaders, training loop with modality dropout, output writers |
| `dropout.py`       | `random_modality_dropout` + `apply_dropout` — per-sample hide one present modality, configurable `p_drop` and `min_keep` |
| `runner.py`        | CLI |

## Outputs (drop-in compatible)

```
<outdir>/
├── static_latent.tsv          # joint z per protein (L2-normed) — the anchor
├── static_latent_raw.tsv      # joint z per protein, pre-L2-norm
├── per_modality/<m>.tsv       # per-modality latents h_m (diagnostics)
├── static_model.pth           # full state_dict
├── pseudo_labels.json         # cached per-modality cluster labels
└── static_loss.tsv            # per-epoch curves: total / recon / struct
```

`static_latent.tsv` matches the format of the existing pipeline's anchor (`protein` + d columns), so `generate_hierarchy.py`, Stage 2's `--stage1_dir`, and `plot_two_stage_final.py --model_dir` all consume it without changes.

## Run

```bash
python -m muse_stage1.runner \
    --manifest ./input_secms_6074/manifest.json \
    --outdir   ./out_final2/muse_stage1_v1 \
    --joint_dim 256 \
    --n_epochs 300 \
    --batch_size 256 \
    --lambda_recon 1.0 --lambda_struct 1.0 \
    --margin 0.3
```

Reasonable defaults — start here, then iterate on `lambda_struct`, `margin`, and `joint_dim` based on the loss curves and the downstream hierarchy / cisplatin MoA tests.

## Key knobs

- **`batch_size`** (default 256). Larger = more diverse negatives → stronger semi-hard mining. Worth trying 384/512 if memory allows.
- **`joint_dim`** (default 256). Has to be big enough to hold all four modalities' info; the existing 128-dim mean anchor is undersized for this objective.
- **`lambda_struct`** (default 1.0). Within-modality triplet weight. Often needs to be increased relative to recon early in training because the recon term saturates fast and otherwise the structure signal is overwhelmed.
- **`margin`** (default 0.3 on cosine distance). The current Stage 1's 0.5 cosine margin tightens too aggressively; 0.3 is a softer starting point. Sweep 0.1 / 0.3 / 0.5 to map the trade-off — we expect smaller margin → less collapsed, more multi-scale.
- **`pseudo_method`** (default `leiden`). Falls back to `kmeans` if `leidenalg`/`igraph` aren't installed. Leiden gives more natural cluster shapes; KMeans is more reproducible.
- **`pseudo_resolution`** / **`pseudo_kmeans_k`**. Controls cluster granularity. Aim for clusters of ~30–150 proteins (so a 256-batch usually has multiple same-cluster pairs).
- **`p_drop`** (default 0.3) — per-sample modality-dropout probability during training. With 4 modalities and `min_keep=1`, this is the probability a given sample has one of its present modalities hidden at the encoder input that step. The decoder is still held accountable for reconstructing the hidden modality (target uses the original input), so this trains *cross-modal prediction* — predicting EPIC from APMS + Image + Sequence, etc. — and is what replaces the explicit cross-modal alignment triplet. Set to 0 to disable.
- **`dropout_min_keep`** (default 1) — never reduce a sample below this many visible modalities. Sequence-only proteins are protected automatically.

## What to test after training

1. **Linear-probe localization**: predict subcellular compartment from the joint z. If localization becomes linearly decodable (it isn't in the current mean anchor), per-modality information preservation is working.
2. **Effective dimension + NN-cos + mid-cosine mass**: should look less collapsed than the current Stage 1.
3. **Hierarchy sweep**: run the existing percentile-mode HiDeF sweep on `static_latent.tsv`. Predict deeper, tighter L1, and improved CORUM recovery if the structural triplets succeed.
4. **Stage 2 retrain**: refit Stage 2 on the new anchor (its `latent_dim` will need updating to match `joint_dim`). Check cisplatin MoA contrast — it should hold; if it drops sharply, the structural signal has displaced too much alignment.

## Dependencies

`torch`, `numpy`, `scikit-learn`. Optional but recommended: `leidenalg` + `python-igraph` for Leiden clustering (falls back to KMeans otherwise).
