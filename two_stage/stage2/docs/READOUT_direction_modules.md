# Stage-2 recommended readout: coordinated direction-modules

**Why this is the standard readout (not per-protein magnitude).**
The Stage-2 treated map's biology is *relational*, not per-protein:
- Same-CORUM-complex proteins move in aligned directions (dir-cosine +0.58 cisplatin,
  +0.72 vorinostat, vs ~0.3–0.5 random; p≈0).
- Per-protein movement *magnitude* (angular displacement) is noise-confounded
  (Spearman −0.20 with abundance; top movers = low-abundance ECM/secreted).
- "All confident differential proteins" enriched as one list → diluted to nothing.
- But proteins grouped by movement *direction* → each cluster is a coherent pathway.

So the readout decomposes the confident set into **direction-modules** (groups moving
the same way = a coordinated remodelling event: complex translocation / co-dissociation
/ co-formation) and enriches each module.

**Validation (output/stage2_v3_FINAL/direction_modules/):** recovers on-target,
drug-specific biology that no per-protein metric could:
- **vorinostat → Histone H3 Acetylation (adjP 6.2e-07)** — the *direct* HDAC-inhibitor
  target — plus SWI/SNF+HDAC1 chromatin and ERAD proteostasis.
- **cisplatin → Mitochondrial Complex I, G2/M cell-cycle transition, tRNA modification,
  RNA splicing, membrane fusion** — mitotoxicity + cell-cycle arrest.
Intra-module direction cosine 0.86–0.96 (tightly coordinated).

## How to run (standard post-inference step)
It needs the negative-control inference (for the confident threshold), so it is a
post-`eval` step rather than inside `eval.py`:
```bash
python -m two_stage.stage2.direction_modules \
  --stage1_latent <stage1>/static_latent.tsv \
  --neg_latent    <out>/inference_negCTRL/z_treat.tsv \
  --drug cisplatin=<out>/inference_cisplatin \
  --drug vorinostat=<out>/inference_vorinostat \
  --outdir <out>/direction_modules
```
Output per drug: `<drug>_direction_modules.tsv` (module, n, median_angle, intra-module
direction coherence, top GO:BP, adjP, members) + `<drug>_module<k>_GOBP.tsv` (full
enrichment per module).

## Confident set + module definition
- confident = angular movement > max(5°, negCTRL q99) — treatment-specific, the null
  rarely reaches it.
- module = KMeans cluster of the unit movement (tangent) vectors.
- enrichment = Enrichr speedrichr, GO_Biological_Process_2023, measured proteome as bg.

## Caveat
The 2-replicate EPIC limit means per-protein scores stay noisy; the **module** level is
the robust unit. Report modules, not per-protein top-movers. The deeper fix is more
EPIC replicates. See DIAGNOSIS_hallucination_and_fixes.md.
