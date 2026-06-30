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
for dropout with probability `p_drop`. When triggered, ONE currently-present
modality is hidden, chosen uniformly at random among the modalities that
*are* present for that sample. The `min_keep` floor (default 1) prevents
hiding everything for a sample that has few modalities to begin with —
notably the Sequence-only proteins are left alone.

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
        Per-sample probability of triggering dropout. With 4 modalities and
        a sample where all four are present, this is also the probability
        that exactly one is hidden this step.
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

    # Among present modalities, pick ONE uniformly at random per sample.
    # Trick: assign random scores in [0, 1) to present (m, sample) pairs
    # and -1 to absent ones; argmax over modalities is then uniformly
    # distributed across present modalities.
    rand_scores = torch.rand(n_mod, B, device=device)
    rand_scores = rand_scores.masked_fill(M < 0.5, -1.0)
    chosen_mod = rand_scores.argmax(dim=0)  # (B,) modality index to drop

    keep_stack = torch.ones(n_mod, B, device=device)
    sample_idx = torch.arange(B, device=device)
    keep_stack[chosen_mod[do_drop], sample_idx[do_drop]] = 0.0

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
