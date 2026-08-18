"""Modality dropout for MUSE-style Stage 1.

Hides modalities at the *encoder input* during training while keeping the
reconstruction *target* unchanged — so the model has to reconstruct hidden
modalities from the remaining ones. This trains:

  1. Robustness to missing modalities at inference time (the data already has
     heterogeneous missingness; this matches the deployment distribution).
  2. Cross-modal information sharing: to reconstruct e.g. EPIC from a `z`
     built without EPIC's input, the fusion must learn that the other
     modalities predict EPIC.
  3. Consistency of `z` across modality subsets — the model is trained on
     every random subset, so `z(P)` becomes a function of "which protein"
     more than "which modalities were observed."

Implementation choice: each *sample* (protein) is independently considered
for dropout with probability `p_drop`. When triggered, the number of
modalities left visible is drawn uniformly from `[min_keep, n_present - 1]`
and that many are kept, chosen uniformly at random among the modalities that
*are* present for that sample. So a protein with all four modalities is
sometimes shown with three, sometimes two, sometimes one — the upper end of
the range is what ties the sparse corners of the map to the well-observed
core. Drawing up to `n_present - 1` means a triggered sample always loses at
least one modality. The `min_keep` floor (default 1) prevents hiding
everything for a sample that has few modalities to begin with — notably the
Sequence-only proteins are left alone.

Hiding exactly one modality per step is not enough: a four-modality protein
would then only ever be seen as three, its `z` never held accountable for
matching the one- or two-modality view, and proteins that are natively
one- or two-modality end up on their own manifold rather than the shared one.

Usage in the training loop:

    keep = random_modality_dropout(masks_present, p_drop=0.3, min_keep=1)
    inputs_visible, masks_visible = apply_dropout(inputs, masks_present, keep)
    z, _ = model.encode(inputs_visible, masks_visible)
    # Recon TARGET still uses the ORIGINAL inputs + masks_present —
    # this is what forces the model to reconstruct the hidden modalities.
    loss = total_loss(model, z, inputs, masks_present, labels, ...)
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch


def random_modality_dropout(
    masks_present: Dict[str, torch.Tensor],
    p_drop: float = 0.3,
    min_keep: int = 1,
) -> Dict[str, torch.Tensor]:
    """Generate a per-sample, per-modality keep mask for modality dropout.

    Parameters
    ----------
    masks_present : {modality_name: (B,) float tensor}
        Original data presence mask — 1 if the modality was measured for the
        sample, 0 if it's structurally missing in the data.
    p_drop : float, default 0.3
        Per-sample probability of triggering dropout — i.e. of hiding at
        least one present modality this step. How many are hidden is then
        uniform over every subset size from `min_keep` up to `n_present - 1`.
    min_keep : int, default 1
        Floor on the number of modalities that must remain visible per sample.
        With min_keep=1, a sample with only one present modality is never
        further reduced.

    Returns
    -------
    keep : {modality_name: (B,) float tensor}
        1.0 = remain visible to encoder this step; 0.0 = hide (zero input,
        zero mask bit). Multiply against `masks_present` to get the effective
        encoder mask: `effective_mask[m] = masks_present[m] * keep[m]`.
    """
    modalities = list(masks_present.keys())
    n_mod = len(modalities)
    first = next(iter(masks_present.values()))
    B = first.shape[0]
    device = first.device

    keep = {m: torch.ones(B, device=device) for m in modalities}
    if p_drop <= 0.0 or n_mod <= 1:
        return keep

    # Stack presence masks as (n_mod, B).
    M = torch.stack([masks_present[m] for m in modalities], dim=0).float()
    n_present = M.sum(dim=0)  # (B,)

    # Per-sample dropout decision (vectorised).
    # Only consider dropping a modality if there are strictly more than
    # min_keep modalities present — otherwise leave the sample alone.
    do_drop = (torch.rand(B, device=device) < p_drop) & (n_present > min_keep)
    if not do_drop.any():
        return keep

    # How many to leave visible: uniform over [min_keep, n_present - 1] for the
    # triggered samples, all of them for the rest. do_drop already guarantees
    # n_present > min_keep, so the range is non-empty and one modality is lost.
    span = n_present - min_keep
    n_keep = torch.where(
        do_drop,
        min_keep + torch.floor(torch.rand(B, device=device) * span),
        n_present,
    )

    # Which ones to keep: random scores over present (m, sample) pairs, -1 for
    # absent ones so they always rank last, then keep the top n_keep by score.
    rand_scores = torch.rand(n_mod, B, device=device)
    rand_scores = rand_scores.masked_fill(M < 0.5, -1.0)
    order = rand_scores.argsort(dim=0, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(0, order, torch.arange(n_mod, device=device).unsqueeze(1).expand(n_mod, B))

    # absent modalities keep a 1, as they did before: their input and mask are
    # already zero, so the flag only ever encodes the dropout decision
    keep_stack = ((rank < n_keep.unsqueeze(0)) | (M < 0.5)).float()

    return {m: keep_stack[i] for i, m in enumerate(modalities)}


def apply_dropout(
    inputs: Dict[str, torch.Tensor],
    masks_present: Dict[str, torch.Tensor],
    keep: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Apply the keep mask to inputs and combine with presence mask.

    Returns (inputs_visible, masks_visible) — what the encoder sees this step.
    The reconstruction *target* should still use the original `inputs` and
    `masks_present` (the un-dropped data), so the decoder is held accountable
    for predicting the hidden modalities.
    """
    inputs_visible = {m: inputs[m] * keep[m].unsqueeze(-1) for m in inputs}
    masks_visible = {m: masks_present[m] * keep[m] for m in masks_present}
    return inputs_visible, masks_visible
