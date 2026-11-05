import torch

from morpheus_encoder.joint_objects import (DifferentiableObjectFormation,
                                             RELATIONS)


def test_joint_object_bank_is_differentiable_and_bounded():
    torch.manual_seed(4)
    model = DifferentiableObjectFormation(d_slot=10, n_slots=7, n_objects=4,
                                          max_order=5)
    u = torch.randn(16, 7, 10, requires_grad=True)
    out = model(u)
    assert out.values.shape == (16, 4)
    assert out.membership.shape == (4, 7)
    assert out.type_probabilities.shape == (4, len(RELATIONS))
    assert out.arity_probabilities.shape == (4, 4)  # arities 2..5
    assert torch.allclose(out.type_probabilities.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out.arity_probabilities.sum(-1), torch.ones(4), atol=1e-6)
    out.values.square().mean().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    assert model.queries.grad is not None and torch.isfinite(model.queries.grad).all()
    assert model.type_head[-1].weight.grad is not None


def test_proposal_dictionary_freezes_on_one_environment_and_evaluates_another():
    torch.manual_seed(8)
    model = DifferentiableObjectFormation(d_slot=8, n_slots=5, n_objects=3,
                                          max_order=5)
    proposal = model(torch.randn(20, 5, 8))
    frozen = model.freeze(proposal, support_threshold=0.0)
    held_out = model.evaluate_frozen(torch.randn(17, 5, 8), frozen)
    assert held_out.shape == (17, 3)
    assert not frozen.membership.requires_grad
    rows = model.proposals(frozen, min_confidence=0.0)
    assert all(2 <= len(r["members"]) <= 5 for r in rows)
    assert all(len(r["members"]) == r["arity"] for r in rows)
    assert all(r["type"] in {"AND", "XOR", "HIGHER_ORDER"} for r in rows)


def test_initial_relation_and_arity_declarations_do_not_flip_between_episodes():
    """A new unlabeled episode may change support, not object semantics by noise."""
    torch.manual_seed(29)
    model = DifferentiableObjectFormation(d_slot=8, n_slots=6, n_objects=5,
                                          max_order=5)
    a = model.freeze(model(torch.randn(13, 6, 8)))
    b = model.freeze(model(torch.randn(17, 6, 8) + 2.5))
    assert torch.equal(a.type_index, b.type_index)
    assert torch.equal(a.arity, b.arity)
    assert torch.all(a.type_index == 0)  # stable initial AND declarations
    assert torch.all(a.arity == 2)


def test_freeze_never_discards_a_dense_object_and_normalizes_its_bounded_support():
    torch.manual_seed(19)
    model = DifferentiableObjectFormation(d_slot=6, n_slots=11, n_objects=3,
                                          max_order=4)
    out = model(torch.randn(9, 11, 6))
    frozen = model.freeze(out)
    active = (frozen.membership > 0).sum(-1)
    assert torch.equal(active.cpu(), frozen.arity.cpu())
    assert torch.all((frozen.arity >= 2) & (frozen.arity <= 4))
    assert torch.allclose(frozen.membership.sum(-1), torch.ones(3), atol=1e-6)
    assert all(len(x) >= 2 for x in frozen.members)


def test_zero_centred_scalar_slots_do_not_collapse_object_selection_to_columns_zero_one():
    """Standardized scalar source coordinates still need distinct addresses."""
    torch.manual_seed(71)
    model = DifferentiableObjectFormation(d_slot=1, n_slots=12, n_objects=8,
                                          max_order=4)
    u = torch.randn(256, 12, 1)
    u = u - u.mean(dim=0, keepdim=True)
    frozen = model.freeze(model(u))
    assert len(set(frozen.members)) > 1
    assert any(m != (0, 1) for m in frozen.members)


def test_coverage_admission_preserves_arity_and_relation_alternatives():
    """A soft argmax must not erase a plausible complete grammar form."""
    torch.manual_seed(72)
    model = DifferentiableObjectFormation(d_slot=1, n_slots=10, n_objects=5,
                                          max_order=5)
    out = model(torch.randn(32, 10, 1))
    rows = model.coverage_proposals(out, n_query_objects=2)
    assert len(rows) == 2 * 4 * 2  # two queries x arities 2..5 x AND/XOR
    assert {row["relation"] for row in rows} == {"AND", "XOR"}
    assert {row["arity"] for row in rows} == {2, 3, 4, 5}
    assert all(len(row["members"]) == row["arity"] for row in rows)
    assert all(row["admission"] == "typed_coverage" for row in rows)


def test_invalid_shape_and_wrong_dictionary_are_refused():
    model = DifferentiableObjectFormation(d_slot=6, n_slots=4, n_objects=2)
    try:
        model(torch.randn(4, 4, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong slot width was accepted")
    frozen = model.freeze(model(torch.randn(4, 4, 6)))
    other = DifferentiableObjectFormation(d_slot=6, n_slots=5, n_objects=2)
    try:
        other.evaluate_frozen(torch.randn(4, 5, 6), frozen)
    except ValueError:
        pass
    else:
        raise AssertionError("foreign frozen dictionary was accepted")


def test_frozen_forward_dictionary_keeps_joint_object_gradients_alive():
    torch.manual_seed(21)
    model = DifferentiableObjectFormation(d_slot=7, n_slots=5, n_objects=3)
    p = model(torch.randn(12, 5, 7))
    frozen = model.freeze(p)
    r_u = torch.randn(15, 5, 7)
    r = model(r_u)
    locked = model.evaluate_frozen(r_u, frozen)
    straight = model.evaluate_frozen(r_u, frozen, gradient_source=r)
    # The object that the compiler sees is identical, not merely close.
    assert torch.equal(locked, straight.detach())
    straight.square().mean().backward()
    assert model.queries.grad is not None
    assert model.type_head[-1].weight.grad is not None
