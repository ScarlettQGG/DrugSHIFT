# Pipeline — manifest schema and worked invocation

This document describes the inputs the three layers expect and a full,
runnable invocation. Replace the placeholder paths with your own.

---

## 1. Input format

Every modality is a **per-protein feature matrix** stored as a TSV: rows are
proteins (first column = protein ID, used as the index), columns are feature
dimensions. All matrices share the same protein ID namespace.

> **ID convention.** Keep one namespace across all modalities. In the U2OS
> example the v6/EPIC embeddings are produced in UniProt and the other
> modalities (AP-MS, image, sequence) are gene-symbol; they are mapped to a
> common namespace (gene symbol) **before** Stage 1. If your modalities use
> different namespaces, harmonise them first — Stage 1 fuses by exact index
> match.

The modalities used in the worked example:

| modality   | source                                   | typical dim |
|------------|------------------------------------------|-------------|
| `EPIC`     | GNN embedding of co-elution PPI graph    | 128         |
| `APMS`     | node2vec over an AP-MS network           | —           |
| `Image`    | DenseNet-121 immunofluorescence features | —           |
| `Sequence` | protein language-model embedding         | —           |

`EPIC` is the perturbed modality (one matrix per condition); the others are
treatment-invariant context that anchors the static map.

---

## 2. `manifest.json`

A JSON **array** of modality entries. One entry per (modality × condition)
matrix. Both Stage 1 and Stage 2 read the same manifest.

```json
[
  {
    "name": "APMS_untreated_1",
    "modality": "APMS",
    "condition": "untreated",
    "path": "/abs/path/APMS_untreated_1.tsv",
    "include_in_static": true
  },
  {
    "name": "EPIC_cisplatin_1",
    "modality": "EPIC",
    "condition": "cisplatin",
    "path": "/abs/path/EPIC_cisplatin_1.tsv",
    "include_in_static": false
  }
]
```

| field                | meaning                                                            |
|----------------------|--------------------------------------------------------------------|
| `name`               | unique label for the entry                                          |
| `modality`           | one of `EPIC`, `APMS`, `Image`, `Sequence`                          |
| `condition`          | e.g. `untreated`, `cisplatin`, `vorinostat`, `negativeCTRL`        |
| `path`               | absolute path to the per-protein TSV                               |
| `include_in_static`  | `true` → used to build the Stage-1 static map; `false` → Stage-2 only |

The **static map** is built from `include_in_static: true` entries (untreated
EPIC + the invariant modalities). Per-condition treated EPIC matrices
(`include_in_static: false`) are consumed by Stage 2 to compute the treatment
delta.

---

## 3. Layer 1 — GNN encoder (`joint_embed.py`, preprocessing)

Encodes each per-replicate co-elution PPI graph into an aligned per-protein
embedding (the `EPIC` modality). Run once; produces
`EPIC_<cond>_avg.tsv` per condition (drop these into the manifest). See
`python joint_embed.py --help` for the graph-input flags.

Conventions enforced at output:
- embeddings are **L2-normalised** (cosine similarity is directly meaningful);
- per-condition averaging uses the **non-renormalised** centroid (mean of unit
  vectors); per-replicate files are unit-norm.

---

## 4. Layer 2 — Stage 1 map (`two_stage.model.Stage1`)

```bash
python -m two_stage.train --stage 1 \
    --manifest   input/manifest.json \
    --outdir     output/stage1 \
    --s1_epochs  <N>
```

Key outputs in `output/stage1/`:
- `static_latent.tsv` — L2-normalised joint co-embedding (the reference map);
- `static_latent_raw.tsv` — pre-norm joint z;
- `static_model.pth` — encoder/decoder/fusion weights (frozen by Stage 2);
- `per_modality/<m>.tsv` — per-modality latents (diagnostics).

Validate the map before going further: CORUM co-membership and STRING
physical-interaction AUROC on `static_latent.tsv` should be well above chance
(≈ 0.8 in the U2OS example).

---

## 5. Layer 3 — Stage 2 adapter (`two_stage.model.NeighborhoodAdapter`)

Train one adapter per condition (including a negative control). Recommended
**unified, drift-removed** configuration. Use `--conditions` to list which
conditions to train (each writes to `<outdir>/<condition>`):

```bash
python -m two_stage.train --stage 2 \
    --stage1_outdir    output/stage1 \
    --manifest         input/manifest.json \
    --outdir           output/stage2 \
    --epic_name        epic \
    --conditions       cisplatin vorinostat negative_ctrl \
    --cond_names       cisplatin vorinostat negative_ctrl \
    --sigma2_epic_path input/sigma2_epic_empirical.tsv \
    --sigma2_raw_scale 0.1 --sigma2_raw_floor 0.001 \
    --w_residual 1.0 --w_epic 10.0 \
    --spherical --unified --drift_remove \
    --n_epochs 300 --lr 1e-3
```

Or run Layers 2 + 3 together with `--stage both` (Stage 1 → `<outdir>/stage1`,
adapters → `<outdir>/stage2/<condition>`).

What the key flags do:

| flag                    | role                                                                 |
|-------------------------|----------------------------------------------------------------------|
| `--unified`             | single coherence-weighted movement (magnitude+direction co-derived)  |
| `--drift_remove`        | subtract global batch drift (otherwise everything looks coherent)    |
| `--spherical`           | keep movement on the unit-sphere tangent plane + cosine EPIC anchor  |
| `--w_residual`          | penalise free drift of the residual projection                       |
| `--sigma2_epic_path`    | inject **empirical** replicate σ² (vs the heuristic default)         |
| `--sigma2_raw_scale/floor` | scale/floor for the EPIC observation variance in the combiner     |
| `--w_epic`              | weight on EPIC reconstruction (treated graph from treated z)         |
| `--k`                   | kNN neighbours over the static map (default 20)                      |

Run `--help` for the full flag set (architecture dims, sequence/image
decoder-stability weights, etc.). The empirical σ² file is a per-protein TSV
of replicate noise (the v6 noise floor); align it to the same protein
namespace as the manifest.

### Inference, evaluation, modules

```bash
python -m two_stage.inference  --adapter_dir output/stage2/cisplatin ...
python -m two_stage.eval          --corum reference/corum_humanComplexes.txt ...
python -m two_stage.direction_modules --confident output/stage2/cisplatin/... ...
```

Stage-2 outputs per condition:
- `z_treat.tsv` — remodelled map (drop-in replacement for `static_latent.tsv`);
- `learned_magnitude.tsv` — per-protein remodelling score (**rank by this**);
- `delta_final.tsv` / `delta_hat.tsv` — full / neighbour-only deltas;
- `sigma2_pred.tsv`, `coherence.tsv` — per-protein uncertainty & neighbour agreement.

---

## 6. Reference data for evaluation

`two_stage/eval.py` and the enrichment readouts validate against external
gold standards that are **not** shipped with this repo (size/licensing):

- **CORUM** human complexes (e.g. `corum_humanComplexes.txt`) — complex
  remodelling;
- **STRING** physical links (`9606.protein.physical.links.detailed.v12.0.txt`)
  — interaction recovery;
- **Enrichr / speedrichr** GO:BP & Reactome — pathway enrichment of remodelled
  proteins (`direction_modules.py` calls the public REST API; needs network).

Download these from their providers and pass the paths via the eval CLI.

---

## 7. Sanity checklist

1. Stage-1 `static_latent.tsv` CORUM/STRING AUROC ≫ 0.5.
2. **Negative-control** Stage-2 run → near-zero `learned_magnitude` for almost
   all proteins, ≈ no CORUM complexes remodelled. (Primary anti-hallucination
   guard.)
3. Treated runs → a clear remodelled tail whose top proteins enrich for the
   expected mechanism of action.
4. Proteins moving together (`direction_modules.py`) share pathways.
