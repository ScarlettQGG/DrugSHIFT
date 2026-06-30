#!/usr/bin/env python3
"""CLI for MUSE Stage-1 v2 (VAE).

Example:
    python -m muse_stage1_vae.runner \\
        --manifest ./input_secms_6074/manifest.json \\
        --outdir   ./out_final2/muse_stage1_vae_v1 \\
        --beta 0.1  --beta_warmup_epochs 30 \\
        --joint_dim 256 --n_epochs 300
"""
import argparse
import json
from .train import train_muse_stage1_vae


def _parse_beta_per_modality(s):
    """`--beta_per_modality "APMS=0.05,Image=0.1,Sequence=0.5,EPIC=0.2"`."""
    if not s:
        return None
    out = {}
    for kv in s.split(","):
        if "=" not in kv: continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir",   required=True)

    # Architecture
    p.add_argument("--latent_dim_per_modality", type=int, default=64)
    p.add_argument("--joint_dim",  type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout",    type=float, default=0.0)

    # Training
    p.add_argument("--n_epochs",     type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--learn_rate",   type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_struct", type=float, default=1.0)
    p.add_argument("--margin",       type=float, default=0.3)

    # VAE knobs
    p.add_argument("--beta",                 type=float, default=0.1,
                   help="KL weight applied to every modality unless overridden.")
    p.add_argument("--beta_per_modality",    default=None,
                   help="Comma-separated MODALITY=β list, e.g. "
                        "'APMS=0.05,Image=0.1,Sequence=0.5,EPIC=0.2'.")
    p.add_argument("--beta_warmup_epochs",   type=int,   default=30)
    p.add_argument("--beta_schedule",        default="linear",
                   choices=["linear", "cyclical"],
                   help="linear: 0→β over warmup_epochs.  cyclical: sawtooth "
                        "with `cyclical_period` epochs per cycle (Fu et al. 2019).")
    p.add_argument("--cyclical_period",      type=int,   default=50,
                   help="Only used when --beta_schedule cyclical.")
    p.add_argument("--free_bits",            type=float, default=0.0,
                   help="Per-latent-dim KL allowance in nats. Each dim's KL is "
                        "clamped to be at least this much before penalisation. "
                        "0.0 disables (plain VAE); 0.5 is a common posterior-"
                        "collapse safeguard.")

    # Modality dropout
    p.add_argument("--p_drop",          type=float, default=0.3)
    p.add_argument("--dropout_min_keep", type=int,   default=1)

    # Pseudo-labels
    p.add_argument("--pseudo_method", default="leiden", choices=["leiden", "kmeans"])
    p.add_argument("--pseudo_knn_k",       type=int,   default=15)
    p.add_argument("--pseudo_resolution",  type=float, default=1.0)
    p.add_argument("--pseudo_kmeans_k",    type=int,   default=50)

    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", default=None)

    args = p.parse_args()
    train_muse_stage1_vae(
        manifest_path=args.manifest, outdir=args.outdir,
        latent_dim_per_modality=args.latent_dim_per_modality,
        joint_dim=args.joint_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        learn_rate=args.learn_rate, weight_decay=args.weight_decay,
        lambda_recon=args.lambda_recon, lambda_struct=args.lambda_struct,
        margin=args.margin,
        beta=args.beta,
        beta_per_modality=_parse_beta_per_modality(args.beta_per_modality),
        beta_warmup_epochs=args.beta_warmup_epochs,
        beta_schedule=args.beta_schedule,
        cyclical_period=args.cyclical_period,
        free_bits=args.free_bits,
        p_drop=args.p_drop, dropout_min_keep=args.dropout_min_keep,
        pseudo_method=args.pseudo_method,
        pseudo_kmeans_k=args.pseudo_kmeans_k, pseudo_knn_k=args.pseudo_knn_k,
        pseudo_resolution=args.pseudo_resolution,
        seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
