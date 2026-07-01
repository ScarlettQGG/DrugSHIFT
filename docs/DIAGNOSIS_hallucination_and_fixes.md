# Stage 2 v3 — hallucination diagnosis & fixes (June 2026)

Record of the investigation into why the Stage-2 neighbourhood adapter produced
"signal" on a negative control, what the co-embedding geometry implies, the fixes
implemented, and the honest conclusion. All fixes are **flag-gated and OFF by
default** — the original pipeline is unchanged unless a flag is passed.

---

## 0. Goal of Stage 2 (for context)
Stage 1 fuses 4 modalities (APMS, EPIC, Image, Sequence) into a co-embedding `z`
(256-d, **L2-normalized → unit hypersphere**), with per-modality decoders.
Stage 2 uses the **EPIC-only treated differential** to learn a per-protein
**movement** in `z`-space, denoised through the Stage-1 PPI neighbourhood, so the
treated map reflects rewired interactions (complex dissociation/formation,
translocation). The movement must be biologically meaningful and reflect the input.

## 1. The symptom
A dedicated negative-control adapter (synthetic `neg_synth`, near-zero input)
produced as much "high-confidence remodelling" as the treated conditions
(neg 32–47% flagged vs treated 4–8%), and treated-vs-neg `delta_final` AUROC was
~0.49 (indistinguishable). The model invented signal from noise.

## 2. Diagnosis (decomposition of `delta_final`)
`delta_final = w_raw·delta_raw_proj + w_pred·delta_hat`, with
`w_pred = σ²_raw/(σ²_raw+σ²_pred)`.

Measured:
- `σ²_raw` (heuristic σ²_EPIC) ≈ 3.76, `σ²_pred` collapsed to the 0.05 floor (77% of
  proteins) → **w_pred ≈ 0.987**. The observed differential got **1.3% weight**.
- `delta_final ≈ delta_hat` (cos 1.00 on neg, 0.92–0.94 treated). The output **is**
  the neighbour/identity head, which conditions on the protein's static identity
  + condition id → it emits a ~0.12 magnitude **regardless of input** = hallucination.
- The strong, well-separated observed signal (treated `delta_raw_proj` norm 1.25 vs
  neg 0.12) was discarded.

Two compounding root causes:
1. **σ²_pred collapse** — Kendall loss optimum `σ²* = residual MSE` (median 0.006) is
   far below the 0.05 floor → clamps → ratio σ²_raw/σ²_pred extreme → trust neighbours.
2. **`residual_proj` is a free MLP** (zero-init output, only weak global wd, no
   magnitude penalty). It produces unconstrained drift the decoders can't see
   (the EPIC decoder is magnitude-attenuating: a z-move of 1.09 changes decoded EPIC
   by only 0.085; gain 0.078–0.31). So `L_epic_recon`/`L_seq` sit at ~0 and cannot
   constrain — **not a weight bug** (w_epic=1.0), but decoder near-flatness + tiny
   per-element MSE scale.

Decoder check: EPIC reconstructs at **cos 0.99** (best modality; Kendall is NOT
down-weighting it). So the decoder represents EPIC's *direction* well but
**attenuates magnitude** and has a low-sensitivity subspace the residual exploited.

## 3. Geometry insight
`||z|| = 1.000` (unit sphere) → only **angular** movement is meaningful (it changes
the cosine neighbourhood). The radial component is a meaningless DOF. Decomposed
movement was ~99% tangential, so the hallucination was **real angular movement**
(neg drifted 6.6°), not a harmless radial artifact.

## 4. Fixes implemented (all flag-gated, OFF by default)
| flag | effect | file |
|---|---|---|
| `--sigma2_epic_path FILE` | inject empirical per-protein replicate σ² as σ²_EPIC (vs heuristic) so the combiner uses the observation for clean proteins | stage1_bridge, train, inference |
| `--w_residual W` | L2 penalty `‖delta_residual‖²` (sum-over-dims) — caps the free null-space drift | losses, training |
| `--spherical` | renormalize z_treat onto the sphere (pure angular movement) + cosine EPIC anchor (`1−cos(D_epic(z_treat),EPIC_treat)`) — geometry-correct | architecture, losses, training, inference |
| `--coherence_gate [--coherence_gate_gamma G]` | scale movement by `max(0, cos(δ_raw_proj, δ̂))` — suppress incoherent (noise) proteins, keep coherent (complex-wide) remodelling | architecture, training, inference |
| `--sigma2_raw_floor`, `--sigma2_raw_scale` | (pre-existing) tune σ² scale; use floor ~0.001 with empirical σ² so the clean end isn't clipped | architecture |

Empirical σ² file: `input_secms_6074/sigma2_epic_empirical.tsv` (v6 noise-floor,
symbol-mapped). 10-rep neg manifest: `input_secms_6074/manifest_10neg.json`.

## 5. Results (cisplatin vs negCTRL separation, on angular geometry)
| config | cis median | neg median | neg <2° (stable) | AUROC |
|---|---|---|---|---|
| original (heuristic σ²) | 8.6° | 6.6° | 0% | 0.49 |
| empirical σ² only | — | — | — | 0.05 (neg amplified to 1.09 — exposed the residual) |
| + residual penalty | 6.0° | **0.3°** | **97%** | **0.99** |
| + spherical (cosine anchor, no resid) | 7.6° | 3.4° | 0% | 0.95 |
| + coherence gate (full combo) | 5.1° | 1.8° | 59% | 0.80 |

- σ²-injection alone made it *worse* (routed output through the observation, but the
  observation contains the free residual → neg amplified). This **exposed** the
  residual as the real culprit.
- residual penalty → null collapses to ~0° (no hallucination), treated retained.
- coherence gate → emerging **stable majority + remodelled tail** (15–21% stable,
  18–21% >10°), but traded off null suppression.

## 6. Honest conclusion (the important part)
The fixes **repaired the mechanics**: no hallucination, healthy σ²_pred, the
observation drives the output, a stable null, principled spherical geometry.

But the **biology is not robust**:
- Per-protein top-movers enrich for ECM/secreted/developmental terms (the noisiest
  SEC-MS proteins; movement ~Spearman 0.20 with σ²_EPIC), **not** DDR/proteasome.
- The complex-level CORUM readout survives but **shifts with configuration**
  (NF-κB complexes with the gate vs RNF8-p97/condensin in the original).
- Treatment biology lives in **relative within-complex** changes, not per-protein
  displacement magnitude.

Root constraint: **2-replicate EPIC is the noisy bottleneck** (established at the
start). Stage 2 cannot manufacture robust biology from weak input; *which* complexes
top the list depends on the knobs.

## 7. Recommendations
1. Keep the mechanical fixes (off by default; enable for principled runs).
2. Treat Stage-2 complex calls as **hypotheses**, validated against the robust
   readouts: elution-layer GO (textbook DDR vs HDACi) and Stage-1 map quality
   (CORUM/STRING AUROC 0.83), plus orthogonal data.
3. **Highest-leverage improvement is the input, not the model**: more EPIC
   replicates / deeper ensembling to raise treatment SNR.
4. If pursuing the model: a Stage-1 EPIC decoder with an **isometry regularizer**
   (remove the low-sensitivity subspace) would let the cosine EPIC anchor replace
   the residual-penalty crutch; and report **relative/complex-level** remodelling,
   not per-protein |δ|.

## 8. Reproduce the principled run
```bash
python -m two_stage.train --stage 2 \
  --stage1_outdir output/stage1 \
  --manifest input_secms_6074/manifest.json --epic_name epic \
  --condition cisplatin --cond_names cisplatin vorinostat negative_ctrl \
  --sigma2_epic_path input_secms_6074/sigma2_epic_empirical.tsv \
  --sigma2_raw_scale 0.1 --sigma2_raw_floor 0.001 \
  --w_residual 1.0 --spherical --coherence_gate \
  --outdir <out> --n_epochs 300 --lr 1e-3
# negative-control validation: same flags, --condition negative_ctrl
#   --manifest input_secms_6074/manifest_10neg.json
# Omit all the new flags → original (pre-investigation) behaviour.
```
