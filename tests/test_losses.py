"""Stage 1 losses: masked reconstruction + semi-hard triplet."""
import torch

from two_stage import make_model
from two_stage.losses import masked_recon_loss, semi_hard_triplet


def _model(mod_dims):
    return make_model(mod_dims, latent_dim_per_modality=8, joint_dim=16, hidden_dim=16)


def test_masked_recon_scalar_and_differentiable(mod_dims, inputs, masks):
    model = _model(mod_dims)
    z, _ = model.encode(inputs, masks)
    loss = masked_recon_loss(z, model.decoders, inputs, masks,
                             model.log_sigma_recon, model.modality_names)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.log_sigma_recon.grad is not None


def test_masked_recon_all_missing_is_zero(mod_dims, inputs, masks):
    model = _model(mod_dims)
    z, _ = model.encode(inputs, masks)
    empty = {m: torch.zeros_like(v) for m, v in masks.items()}
    loss = masked_recon_loss(z, model.decoders, inputs, empty,
                             model.log_sigma_recon, model.modality_names)
    assert float(loss) == 0.0


def test_masked_recon_ignores_missing_rows(mod_dims, inputs, masks):
    """Dropping one modality's mask should change the loss (it's excluded)."""
    model = _model(mod_dims)
    z, _ = model.encode(inputs, masks)
    full = masked_recon_loss(z, model.decoders, inputs, masks,
                             model.log_sigma_recon, model.modality_names)
    part = dict(masks)
    part["Sequence"] = torch.zeros_like(part["Sequence"])
    dropped = masked_recon_loss(z, model.decoders, inputs, part,
                                model.log_sigma_recon, model.modality_names)
    assert not torch.isclose(full, dropped)


def test_semi_hard_triplet_needs_three():
    z = torch.randn(2, 8)
    assert semi_hard_triplet(z, torch.tensor([0, 1])) is None


def test_semi_hard_triplet_scalar_on_structured_batch():
    # two clusters, clearly separated -> valid triplets exist
    a = torch.randn(6, 8) + 5.0
    b = torch.randn(6, 8) - 5.0
    z = torch.cat([a, b])
    labels = torch.tensor([0] * 6 + [1] * 6)
    out = semi_hard_triplet(z, labels, margin=0.3)
    assert out is not None and out.dim() == 0 and out >= 0.0
