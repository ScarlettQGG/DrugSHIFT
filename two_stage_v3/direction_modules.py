#!/usr/bin/env python3
"""direction_modules.py — Stage-2 coordinated-remodelling readout.

The biological signal in the Stage-2 treated map is RELATIONAL, not per-protein:
proteins that move in the same DIRECTION in the co-embedding belong to the same
pathway/complex (validated: same-CORUM-complex pairs move with cosine +0.58/+0.72
vs random; per-protein magnitude is noise-confounded). So the right readout is:

  1. confident set   = proteins whose angular movement exceeds the negative
                       control (treatment-specific), floor 5°.
  2. direction-module = KMeans cluster of the unit movement vectors (proteins
                       moving the same way = a coordinated remodelling event).
  3. per-module GO:BP enrichment (Enrichr speedrichr, measured proteome as bg).

Each module = one coherent pathway/complex moving together (translocation /
co-dissociation / co-formation). Run per drug.

Usage:
  python -m two_stage_v3.direction_modules \
      --stage1_latent output/muse_stage1_v6sig0.5_v3compat/static_latent.tsv \
      --neg_latent    output/stage2_v3_FINAL/inference_negCTRL/z_treat.tsv \
      --drug cisplatin=output/stage2_v3_FINAL/inference_cisplatin \
      --drug vorinostat=output/stage2_v3_FINAL/inference_vorinostat \
      --outdir output/stage2_v3_FINAL/direction_modules
"""
import argparse, time, requests, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

SPEED = "https://maayanlab.cloud/speedrichr/api"


def load_z(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df = df[[c for c in df.columns if c != "protein"]]
    return df


def angular(zr, zt):
    com = zr.index.intersection(zt.index)
    Z = zr.loc[com].values; T = zt.loc[com].values
    cos = (Z * T).sum(1) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(T, axis=1) + 1e-9)
    return pd.Series(np.degrees(np.arccos(np.clip(cos, -1, 1))), index=com)


def enrich(genes, bg, lib, retries=4):
    for _ in range(retries):
        try:
            s = requests.Session()
            uid = s.post(f"{SPEED}/addList",
                         files={"list": (None, "\n".join(genes)),
                                "description": (None, "m")}).json()["userListId"]; time.sleep(0.4)
            bgid = s.post(f"{SPEED}/addbackground",
                          data={"background": "\n".join(bg)}).json()["backgroundid"]; time.sleep(0.4)
            r = s.post(f"{SPEED}/backgroundenrich",
                       data={"userListId": uid, "backgroundid": bgid,
                             "backgroundType": lib}).json().get(lib, [])
            return pd.DataFrame([{"term": x[1], "pval": x[2], "adj_pval": x[6],
                                  "n_overlap": len(x[5]), "overlap_genes": ";".join(x[5])}
                                 for x in r]).sort_values("adj_pval")
        except Exception:
            time.sleep(2)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_latent", required=True)
    ap.add_argument("--drug", action="append", required=True,
                    help="name=inference_dir (repeatable)")
    ap.add_argument("--neg_latent", default=None,
                    help="negative-control z_treat.tsv for the confident threshold")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_modules", type=int, default=15)
    ap.add_argument("--min_angle", type=float, default=None,
                    help="confident threshold (deg); default = max(5, null q99)")
    ap.add_argument("--min_module_size", type=int, default=15)
    ap.add_argument("--go_library", default="GO_Biological_Process_2023")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    zr = load_z(a.stage1_latent)

    thr = a.min_angle
    if thr is None and a.neg_latent:
        thr = max(5.0, float(angular(zr, load_z(a.neg_latent)).quantile(0.99)))
    thr = thr if thr is not None else 5.0
    print(f"[modules] confident angular threshold = {thr:.2f}°")

    all_rows = []
    for spec in a.drug:
        name, inf = spec.split("=", 1)
        zt = load_z(Path(inf) / "z_treat.tsv")
        ang = angular(zr, zt)
        conf = ang[ang > thr].index
        bg = ang.index.astype(str).tolist()                       # measured proteome
        D = zt.loc[conf].values - zr.loc[conf].values
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        K = max(2, min(a.n_modules, len(conf) // a.min_module_size))
        lab = KMeans(n_clusters=K, n_init=5, random_state=0).fit_predict(Dn)
        print(f"\n=== {name}: {len(conf)} confident proteins (> {thr:.1f}°) → {K} direction modules ===")
        rows = []
        for k in range(K):
            mask = lab == k
            mem = [str(p) for p in conf[mask]]
            if len(mem) < a.min_module_size:
                continue
            e = enrich(mem, bg, a.go_library)
            top = e.iloc[0] if len(e) else None
            rows.append({
                "drug": name, "module": k, "n_members": len(mem),
                "median_angle_deg": round(float(ang.loc[conf[mask]].median()), 2),
                "intra_module_dir_cosine": round(float(np.median(
                    (Dn[mask] @ Dn[mask].mean(0)) / (np.linalg.norm(Dn[mask].mean(0)) + 1e-9))), 3),
                "top_GOBP": top.term if top is not None else "",
                "adj_pval": float(top.adj_pval) if top is not None else np.nan,
                "n_overlap": int(top.n_overlap) if top is not None else 0,
                "top_genes": ";".join(top.overlap_genes.split(";")[:8]) if top is not None else "",
                "members": ";".join(mem),
            })
            if len(e):
                e.to_csv(out / f"{name}_module{k}_GOBP.tsv", sep="\t", index=False)
            time.sleep(0.5)
        tab = pd.DataFrame(rows).sort_values("adj_pval")
        tab.to_csv(out / f"{name}_direction_modules.tsv", sep="\t", index=False)
        all_rows.append(tab)
        for _, r in tab.iterrows():
            flag = "*" if r.adj_pval < 0.05 else " "
            print(f"  {flag} module {int(r.module):2d}  n={r.n_members:3d}  {r.median_angle_deg:4.1f}°  "
                  f"dir-coh={r.intra_module_dir_cosine:.2f}  {str(r.top_GOBP)[:44]:44s} "
                  f"adjP={r.adj_pval:.1e}  [{str(r.top_genes)[:30]}]")
    pd.concat(all_rows).to_csv(out / "all_direction_modules.tsv", sep="\t", index=False)
    print(f"\n[done] per-module pathway tables in {out}")


if __name__ == "__main__":
    main()
