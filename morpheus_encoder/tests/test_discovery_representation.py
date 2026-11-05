import torch
import pytest

from morpheus_encoder.discovery_representation import (
    IdentityPreservingEncoder,
    RepresentationConfig,
    fit_response_free,
    represent,
)


def test_source_arm_is_exact_bypass():
    x = torch.randn(12, 6)
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=6))
    assert represent(model, x, "source") is x


def test_response_free_arms_train_without_response_or_relation_label():
    torch.manual_seed(3)
    x = torch.randn(20, 5)
    neighbours = torch.stack([torch.roll(torch.arange(20), 1),
                              torch.roll(torch.arange(20), -1)], dim=1)
    for arm in ("reconstruction", "ssl", "predictive", "morpheus_legacy", "morpheus_factorized", "morpheus"):
        model = IdentityPreservingEncoder(RepresentationConfig(d_input=5, mask_probability=0.15))
        losses = fit_response_free(model, x, arm, neighbours=neighbours, steps=3)
        z = represent(model, x, arm)
        assert len(losses) == 3
        assert torch.isfinite(z).all()
        assert z.shape == x.shape


def test_invalid_neighbour_contract_fails_closed():
    x = torch.randn(8, 4)
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=4))
    try:
        model.objective(x, "morpheus", neighbours=torch.ones(8, 2, 1))
    except ValueError as error:
        assert "neighbours" in str(error)
    else:
        raise AssertionError("invalid neighbourhood must be rejected")


def test_morpheus_uses_separate_self_and_context_predictor_heads():
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=4))
    assert model.predictor is not model.context_predictor
    assert set(model.predictor.parameters()).isdisjoint(set(model.context_predictor.parameters()))
    assert sum(p.numel() for p in model.predictor.parameters()) == \
           sum(p.numel() for p in model.context_predictor.parameters())


def test_all_missing_neighbours_falls_back_without_context_head_gradient():
    torch.manual_seed(18)
    x = torch.randn(10, 4)
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=4, mask_probability=0.0))
    loss = model.objective(x, "morpheus", neighbours=torch.full((10, 3), -1))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.context_predictor.parameters())


def test_factorized_contextual_objective_trains_finitely_with_real_neighbours():
    torch.manual_seed(19)
    x = torch.randn(18, 5)
    neighbours = torch.stack([torch.roll(torch.arange(18), 1),
                              torch.roll(torch.arange(18), -1)], dim=1)
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=5, mask_probability=0.1))
    losses = fit_response_free(model, x, "morpheus", neighbours=neighbours, steps=4)
    assert len(losses) == 4 and all(torch.isfinite(torch.tensor(losses)))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_predictor.parameters())


@pytest.mark.parametrize("bad", [torch.ones(3, 4, dtype=torch.long),
                                   torch.tensor([[1.0, float("nan")]])])
def test_observed_input_validation_rejects_nonfloat_or_nonfinite_values(bad):
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=4))
    with pytest.raises((TypeError, ValueError)):
        model.objective(bad, "morpheus")
