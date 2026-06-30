#!/usr/bin/env python3
"""CLI for MUSE-style Stage 1.

Example:
    python -m muse_stage1.runner \
        --manifest ./input_secms_6074/manifest.json \
        --outdir   ./out_final2/muse_stage1_v1 \
        --joint_dim 256 \
        --n_epochs 300 \
        --batch_size 256 \
        --lambda_recon 1.0 --lambda_struct 1.0 \
        --margin 0.3

The output `static_latent.tsv` is drop-in compatible with downstream Stage 2 /
hierarchy code (same format as the existing pipeline's anchor).
"""
import argparse
from .train import train_muse_stage1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="Path to manifest.json")
    p.add_argument("--outdir", required=True, help="Where to write Stage-1 outputs")

    # Architecture
    p.add_argument("--latent_dim_per_modality", type=int, default=64)
    p.add_argument("--joint_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    # Training
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Larger is better for semi-hard mining (256-512 recommended).")
    p.add_argument("--learn_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_struct", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.3,
                   help="Triplet margin on cosine distance.")
    p.add_argument("--p_drop", type=float, default=0.3,
                   help="Per-sample modality-dropout probability during training. "
                        "0 disables. 0.3 = ~30%% of samples have one modality "
                        "hidden each batch.")
    p.add_argument("--dropout_min_keep", type=int, default=1,
                   help="Minimum number of modalities that must remain visible "
                        "per sample. Default 1 (never blank everything).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="'cuda' or 'cpu'; auto-detected if omitted.")

    # Pseudo-labels
    p.add_argument("--pseudo_method", default="leiden", choices=["leiden", "kmeans"],
                   help="Per-modality clustering. Leiden (cosine kNN) preferred; "
                        "kmeans is the fallback if leidenalg/igraph aren't installed.")
    p.add_argument("--pseudo_knn_k", type=int, default=15)
    p.add_argument("--pseudo_resolution", type=float, default=1.0,
                   help="Leiden resolution.")
    p.add_argument("--pseudo_kmeans_k", type=int, default=50,
                   help="K for KMeans fallback.")

    args = p.parse_args()
    train_muse_stage1(
        manifest_path=args.manifest, outdir=args.outdir,
        latent_dim_per_modality=args.latent_dim_per_modality,
        joint_dim=args.joint_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        learn_rate=args.learn_rate, weight_decay=args.weight_decay,
        lambda_recon=args.lambda_recon, lambda_struct=args.lambda_struct,
        margin=args.margin,
        p_drop=args.p_drop, dropout_min_keep=args.dropout_min_keep,
        pseudo_method=args.pseudo_method,
        pseudo_kmeans_k=args.pseudo_kmeans_k,
        pseudo_knn_k=args.pseudo_knn_k,
        pseudo_resolution=args.pseudo_resolution,
        seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
