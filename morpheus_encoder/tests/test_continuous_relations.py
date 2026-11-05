"""Tests for the non-symbolic, universal continuous relation path."""
import pytest
import torch

from morpheus_encoder.continuous_relations import ContinuousRelationBank


def test_continuous_relation_is_finite_and_differentiable():
    torch.manual_seed(4)
    slots = torch.randn(3, 7, 11, requires_grad=True)
    bank = ContinuousRelationBank(11, 7, n_relations=4, d_relation=13, max_order=5)
    out = bank(slots)
    assert out.relation_state.shape == (3, 4, 13)
    assert out.value.shape == (3, 4)
    assert torch.isfinite(out.relation_state).all()
    loss = out.value.square().mean() + out.approx_description_bits.mean()
    loss.backward()
    assert slots.grad is not None and torch.isfinite(slots.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters())


def test_symmetric_relation_is_invariant_to_slot_permutation():
    torch.manual_seed(6)
    bank = ContinuousRelationBank(9, 6, n_relations=3, d_relation=10, max_order=4).eval()
    slots = torch.randn(2, 6, 9)
    perm = torch.tensor([4, 1, 5, 0, 3, 2])
    first, second = bank(slots), bank(slots[:, perm])
    assert torch.allclose(first.relation_state, second.relation_state, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first.value, second.value, atol=1e-6, rtol=1e-6)


def test_continuous_proposal_metadata_is_compact_and_refuses_symbolic_claim():
    out = ContinuousRelationBank(8, 5, n_relations=2, max_order=3)(torch.randn(4, 5, 8))
    assert out.is_symbolic_program is False
    assert out.relation_family == "continuous_set_relation"
    assert out.expected_arity.min() >= 2 and out.expected_arity.max() <= 3
    assert torch.isfinite(out.approx_description_bits).all()
    records = out.proposal_records()
    assert records[0]["id"] == "continuous_relation:000"
    assert records[0]["is_symbolic_program"] is False
    assert "certification" in records[0]["requires"]
    with pytest.raises(TypeError, match="not a symbolic program"):
        out.as_symbolic_program()
