# Stage 2 — consolidated conclusions (the full investigation arc)

Goal: use the Stage-1 high-confidence multimodal map to denoise the EPIC-only
treated differential and remap it as per-protein movement in the co-embedding,
so movement reflects rewired PPIs (complex (dis)association / translocation),
identifies differential proteins, and a stable majority stays put.

This documents the whole arc, the dead-ends, the breakthrough, and the hard limit.
All new mechanisms are flag-gated OFF by default (the original pipeline is unchanged).

---

## 1. The binding constraint (established first and confirmed throughout): 2 replicates
- EPIC RF edge scores reproduce at r≈0.36; strong-edge persistence ≈3% (95% partner
  turnover between two untreated replicates).
- Co-elution FEATURES are reproducible (Euclidean r=0.84, Jaccard 0.82) but the
  per-pair treatment effect is only ~1.5× noise — too weak for a sparse per-protein
  differential. Elution profiles r≈0.94 but proxy for PPI and still dense.
- v6 embedding SMOOTHS the noisy edges → usable, but its delta is **dense & holistic**
  (every protein shifts). Subtracting embeddings is therefore the wrong input — it
  can never yield a stable population.
- **No layer gives a clean, sparse, PPI-specific per-protein differential at n=2.**
  The only real fix is more EPIC replicates. Everything below is working within n=2.

## 2. The hallucination (original model)
A negative control (synthetic neg) produced as much "remodelling" as the drugs.
Decomposition: the Bayesian combiner weighted the neighbour/identity head ~99%
(`w_pred≈0.987`, heuristic σ²_raw≈3.76 ≫ σ²_pred floored at 0.05). The output was the
identity-conditioned head (≈0.12 magnitude regardless of input); the real observed
signal (δ_raw_proj) was discarded. The EPIC decoder is magnitude-attenuating (direction
cos 0.99, gain 0.08–0.31), so L_epic couldn't constrain magnitude either.

## 3. Geometry: the co-embedding is normalised (unit sphere)
Only ANGULAR movement is meaningful; radial is a meaningless DOF. ~99% of movement is
tangential, so the hallucination was REAL angular drift (neg 6.6°). Fix: renormalise
z_treat (`--spherical`) + cosine EPIC anchor (decoder preserves direction).

## 4. Fixes tried, in order (each flag-gated)
| fix | flag | effect |
|---|---|---|
| empirical σ²_EPIC | --sigma2_epic_path | combiner uses the observation (vs heuristic) |
| residual penalty | --w_residual | caps free drift of residual_proj → null 0.3°, AUROC 0.99 |
| spherical + cosine anchor | --spherical | correct geometry |
| coherence gate | --coherence_gate | bimodal-ish, but reshapes only |
| factorized δ=m·u | --factorized_delta | separate magnitude head; perfect null detection BUT magnitude noise-confounded / flat / no stable pop |
| drift removal | --drift_remove | remove global drift (49–65% of each protein's movement) |
| **unified** | --unified | **single coherence-weighted movement (magnitude+direction co-derived)** |

## 5. Why factorizing magnitude & direction was wrong
Separate heads/losses; the product m·u was never jointly supervised (decoder
magnitude-attenuating). Biologically magnitude and direction are the SAME signal
(signal strength). Decoupling → inconsistency → per-protein ranking unstable
(drugs traded enrichment between configs).

## 6. The breakthrough: UNIFIED coherence-weighted, drift-removed
`δ_final = max(0, cos(δ_obs_c, δ̂_c)) · δ_obs` with both tangential & drift-removed.
Magnitude and direction co-derive from ONE coherence operation: neighbours agree →
keep; disagree (noise) → both vanish. Results (UNIFIED2 = output/stage2_v3_UNIFIED2):
- **STABLE population 75% / 70%** (first model ever to produce one)
- **magnitude noise-decoupled** (corr with abundance −0.04/+0.00, vs angular −0.20,
  factorized −0.3..−0.5)
- **MOA-enriched top differential proteins** (nominal p, FDR-borderline at n=2):
  cisplatin → ROS detox (PRDX/ERO1A/TXNRD1) + DNA repair (ATM, BRIP1, recombinational
  repair) + apoptosis (caspase); vorinostat → ubiquitin (BRCA1/UBE2T) + transcription.
- treated-map still recovers CORUM (z_treat AUROC 0.82 vs untreated 0.83).

## 7. The "strong modules were the drift" insight
Earlier models' huge modules (Translation adjP 1e-57, Histone-Acetylation 6e-7) were
proteins moving WITH the global drift — non-specific. The coherence gate correctly
EXCLUDES them. You cannot have both the strong Translation modules AND a stable
population: the strong modules WERE the drift; the stable population REQUIRES removing
it. The remaining (weaker) modules — vesicle/endosome (cisplatin), DNA-templated
transcription/RB1-SET-ZMYND8 (vorinostat) — are the treatment-SPECIFIC signal.

## 8. RECOMMENDED final config
```
python -m two_stage.stage2.training \
  --stage1_outdir output/stage1 --manifest input_secms_6074/manifest.json \
  --epic_name epic --condition <cond> --cond_names cisplatin vorinostat negative_ctrl \
  --sigma2_epic_path input_secms_6074/sigma2_epic_empirical.tsv \
  --sigma2_raw_scale 0.1 --sigma2_raw_floor 0.001 --w_residual 1.0 --w_epic 10.0 \
  --spherical --unified --drift_remove --n_epochs 300 --lr 1e-3
```
Readouts: `learned_magnitude.tsv` (per-protein differential, rank by this — stable vs
remodeled), `direction_modules.py` (coordinated remodelling modules). Negative control
(neg_synth) validates no hallucination. Snapshot: output/stage2_v3_RECOMMENDED_unified/.

## 9. The one thing that moves the ceiling
**More EPIC replicates.** Every layer's treatment-vs-noise ratio is replicate-limited
(edge 3% persistence; feature 1.5× noise). With more reps, the feature-level differential
(option 1) sparsifies and a confident stable/remodeled split with strong, SPECIFIC
modules becomes achievable. Within n=2, the unified drift-removed model is the most
honest, treatment-specific readout obtainable.
