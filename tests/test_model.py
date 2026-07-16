"""Stage 1 mask-aware autoencoder: shapes, mask-gating, decode round-trip."""
import torch

from two_stage import make_model


def _model(mod_dims):
    return make_model(mod_dims, latent_dim_per_modality=8, joint_dim=16, hidden_dim=16)


def test_encode_output_shapes(mod_dims, inputs, masks):
    model = _model(mod_dims)
    z, h = model.encode(inputs, masks)
    B = next(iter(inputs.values())).shape[0]
    assert z.shape == (B, 16)
    assert torch.isfinite(z).all()
    for m in mod_dims:
        assert h[m].shape == (B, 8)


def test_mask_gating_zeroes_missing_modality(mod_dims, inputs, masks):
    """A modality with mask=0 must contribute a zero per-modality latent."""
    model = _model(mod_dims)
    masks = dict(masks)
    masks["APMS"] = torch.zeros_like(masks["APMS"])
    _, h = model.encode(inputs, masks)
    assert torch.count_nonzero(h["APMS"]) == 0
    # a present modality is (almost surely) not all-zero
    assert torch.count_nonzero(h["EPIC"]) > 0


def test_decode_roundtrip_shapes(mod_dims, inputs, masks):
    model = _model(mod_dims)
    z, _ = model.encode(inputs, masks)
    for m, d in mod_dims.items():
        x_hat = model.decode(z, m)
        assert x_hat.shape == (z.shape[0], d)


def test_forward_is_differentiable(mod_dims, inputs, masks):
    model = _model(mod_dims)
    z, _ = model.encode(inputs, masks)
    z.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed back through the encoder/fusion"


def test_kendall_params_exist(mod_dims):
    model = _model(mod_dims)
    assert model.log_sigma_recon.shape[0] == len(mod_dims)
    assert model.log_sigma_struct.shape[0] == len(mod_dims)
