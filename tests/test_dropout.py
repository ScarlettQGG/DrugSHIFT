"""Modality dropout: keep-mask invariants + application to inputs/masks."""
import torch

from two_stage.dropout import random_modality_dropout, apply_dropout

MODS = ["EPIC", "APMS", "Image", "Sequence"]
B = 32


def _present():
    return {m: torch.ones(B) for m in MODS}


def test_no_dropout_when_p_zero():
    keep = random_modality_dropout(_present(), p_drop=0.0)
    assert all(float(keep[m].sum()) == B for m in MODS)


def test_min_keep_respected():
    """Every sample must retain at least min_keep visible modalities."""
    torch.manual_seed(1)
    keep = random_modality_dropout(_present(), p_drop=1.0, min_keep=1)
    visible = torch.stack([keep[m] for m in MODS], dim=0).sum(0)  # (B,)
    assert (visible >= 1).all()
    # with p_drop=1 and all 4 present, exactly one is dropped -> 3 visible
    assert (visible == 3).all()


def test_dropout_never_revives_absent_modality():
    masks = _present()
    masks["Image"] = torch.zeros(B)                 # structurally absent
    keep = random_modality_dropout(masks, p_drop=1.0, min_keep=1)
    _, masks_vis = apply_dropout({m: torch.ones(B, 4) for m in MODS}, masks, keep)
    assert float(masks_vis["Image"].sum()) == 0.0   # stays absent


def test_apply_dropout_zeroes_hidden_inputs():
    masks = _present()
    inputs = {m: torch.ones(B, 4) for m in MODS}
    keep = {m: torch.ones(B) for m in MODS}
    keep["APMS"] = torch.zeros(B)                    # force-hide APMS
    inp_vis, masks_vis = apply_dropout(inputs, masks, keep)
    assert float(inp_vis["APMS"].abs().sum()) == 0.0
    assert float(masks_vis["APMS"].sum()) == 0.0
    assert float(inp_vis["EPIC"].abs().sum()) > 0.0
