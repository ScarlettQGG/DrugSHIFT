# Stage 2 v3 — neighborhood-aware delta adapter

## What this model does

Stage 2 v3 is a small, per-condition adapter that sits on top of a frozen MUSE Stage-1 multimodal reference map and turns a noisy, condition-specific EPIC observation into a denoised, biologically constrained movement in the joint co-embedding space. For every protein, it takes the raw EPIC delta — `EPIC_treat − EPIC_ctrl` in input space — and learns to remap it into z-space using a leave-one-out neighbourhood-prediction objective: at training time, each protein's delta is predicted from its kNN neighbours' deltas alone, never from its own observation, which makes identity-mapping structurally impossible and forces the adapter to learn the part of the signal that propagates coherently through biologically related proteins. The adapter then Bayesianly combines the neighbour-only prediction with the projected raw observation, weighted by the per-protein EPIC noise level (`σ²_EPIC`) inherited from Stage 1's own decoder residual and a learned heteroscedastic uncertainty head — so a protein with a clean EPIC profile keeps its raw signal, while a protein with a noisy profile (or none at all) inherits its delta from coherent neighbours. The graph is a top-k cosine kNN over the Stage 1 joint embedding, with edges weighted by `w_ij = conf_j · cluster_compat(i, j)^(1/τ)` — two factors, both genuinely doing work after row-normalisation: `conf_j` is the source neighbour's per-protein anchor reliability (harmonic mean of per-modality alignment with the joint, low for satellite proteins that landed in a cluster by accident), and `cluster_compat` is a soft Leiden gate that keeps cross-cluster edges from dominating biologically unrelated proteins. The earlier draft also multiplied by `conf_i` and `cos(z_i, z_j)`; both were dropped after audit — `conf_i` is mathematically cancelled by the row-normalisation step (it's a constant across row i), and `cos(z_i, z_j)` adds little beyond the kNN restriction itself because all retained edges sit in a narrow high-cosine band already, with the attention layer doing the fine-grained ranking among them. Biological priors are imposed asymmetrically through Stage 1's frozen decoders: **sequence reconstruction is constrained tightly because a protein's primary sequence is genuinely invariant to treatment, while image (HPA-localization) constraints are deliberately soft because translocation is a real treatment response — DNA-damage agents like cisplatin recruit proteins to chromatin, and HDAC inhibitors like vorinostat reshape nuclear localization broadly — and a heavy image-stability loss would suppress exactly the biology we want to detect.** The default image weight is small enough to penalise wild moves (e.g. predictions collapsing to a uniform HPA distribution) without blocking real localization changes; an optional translocation-aware variant penalises only *entropy increases* in `D_image(z_treat)` so that direction-of-change is free but confident-to-uncertain drift is discouraged. The result, per condition, is a treated joint embedding `z_treat` that reflects neighbourhood-consensus remodelling and translocation rather than noisy raw measurement, a delta-magnitude readout that reflects signal strength, and an uncertainty channel that flags proteins whose treated state is poorly supported by either their own data or their biological neighbourhood.

## Files

| file | role |
|---|---|
| `cache.py` | `Stage1Cache.from_stage1_dir(stage1_outdir, manifest)` — builds z, h_m, σ²_EPIC, conf, Leiden clusters, cluster-aware kNN in memory; attaches frozen Stage 1 decoders. **No persisted cache file** — every Stage 2 task recomputes (~30 sec) so Stage 1 is the single source of truth. |
| `model.py` (`NeighborhoodAdapter`) | `NeighborhoodAdapter` — self-context + delta encoder + attention + δ̂/σ²_pred head + (b)+residual δ_raw projection + Bayesian combination |
| `losses.py` | `L_LOO` (Kendall) + decoder-stability losses + `L_epic_recon` |
| `train.py` | Full-batch train loop + CLI; one adapter per condition |
| `inference.py` | Apply trained adapter; dump `z_treat`, `delta_final`, `delta_hat`, `sigma2_pred`, `coherence` |
| `eval.py` | Biological eval: CORUM remodeling, cluster transitions, HPA shift, coherence × ‖δ‖ flags |
| `direction_modules.py` | Decompose the confident set into coordinated movement modules + per-module GO:BP enrichment |

## How to run

The **recommended** configuration is unified + drift-removed (see the
top-level `docs/PIPELINE.md` for the full flag reference). Paths below are
placeholders.

```bash
# 1. Train one adapter per condition (cisplatin, vorinostat, and a negative control)
#    No "build cache" step — training rebuilds Stage 1 features in-memory.
python -m two_stage.train --stage 2 \
    --stage1_outdir    <stage1_outdir> \
    --manifest         <manifest.json> \
    --epic_name        epic \
    --condition        cisplatin --cond_names cisplatin vorinostat negative_ctrl \
    --outdir           <out>/adapter_cisplatin \
    --sigma2_epic_path <sigma2_epic_empirical.tsv> \
    --sigma2_raw_scale 0.1 --sigma2_raw_floor 0.001 \
    --w_residual 1.0 --w_epic 10.0 --spherical --unified --drift_remove \
    --n_epochs 300 --lr 1e-3

# 2. Inference (the adapter's config.json already remembers stage1_outdir + manifest_path)
python -m two_stage.inference \
    --adapter_dir <out>/adapter_cisplatin \
    --manifest    <manifest.json> \
    --outdir      <out>/inference_cisplatin

# 3. Eval
python -m two_stage.eval \
    --adapter_dir   <out>/adapter_cisplatin \
    --inference_dir <out>/inference_cisplatin \
    --corum         <corum_humanComplexes.txt> \
    --outdir        <out>/eval_cisplatin

# 4. Coordinated remodelling modules + GO:BP enrichment
python -m two_stage.direction_modules \
    --inference_dir <out>/inference_cisplatin --outdir <out>/modules_cisplatin
```

## Outputs (per condition)

- `z_treat.tsv` — treated joint embedding (drop-in replacement for `static_latent.tsv`)
- `delta_final.tsv` — denoised remodelling vector per protein (z-space)
- `delta_hat.tsv` — neighbour-only prediction (no self-info)
- `delta_raw_proj.tsv` — projected raw delta (frozen baseline + residual)
- `sigma2_pred.tsv` — per-protein neighbour-prediction uncertainty
- `coherence.tsv` — `cos(δ_raw_proj, δ̂)`; per-protein measure of agreement between raw observation and neighbour consensus

## Design choices (locked)

1. δ projection: **(b) frozen `E_EPIC` baseline + zero-init learned residual**
2. Graph: **cluster-aware kNN** — k=20 cosine, edges weighted by `conf_j · cluster_compat(i, j)^(1/τ)` (two factors after the audit dropped the redundant `conf_i` and `cos(z_i, z_j)` terms)
3. Conditioning: per-modality `h_m` + cluster id + condition id, all into `MLP_self`
4. Losses: `L_LOO` + `L_seq_stable` (heavy, 1.0) + `L_image_stable` (SOFT, 0.05 — translocation is real biology) + `L_epic_recon` (main, 1.0). Optional `L_image_confidence` (entropy-only image constraint that allows direction-of-translocation change) off by default — turn on with `--w_image_confidence 0.1` if you want a per-protein "stay confident about where you ARE, even if it changes" constraint. `L_smooth` was removed after audit — its signal was already implicit in `L_LOO` (because each centre protein's δ̂ is computed from its neighbours, two centres that share neighbours naturally produce similar δ̂).
5. No replicates, no APMS-treat — `L_replicate` and `L_apms_consist` are NOT used; APMS is conditioning only
6. One adapter per condition, trained independently

## Why this beats a plain graph mean filter

A bare `δ_smoothed = Σ w_ij · δ_raw_j` is uniformly smoothed, biology-blind, and has no per-protein noise model. This adapter adds: (i) per-protein learned uncertainty `σ²_pred` so well-supported proteins keep magnitude while isolated proteins shrink toward zero; (ii) Bayesian combination with Stage 1's per-protein `σ²_EPIC` so noisy measurements automatically lean on neighbours; (iii) leave-one-out supervision so identity is impossible and the model has to extract the *predictable* part of the signal; (iv) frozen Stage-1 decoder constraints so the delta cannot drift to biologically impossible z-regions (e.g. a region whose sequence reconstruction disagrees with the protein's actual sequence); (v) cluster-aware edge weights so cross-complex bleed is suppressed.
