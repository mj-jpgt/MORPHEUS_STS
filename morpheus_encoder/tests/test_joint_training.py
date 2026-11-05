"""Contracts for the observed-data joint formation/replication training seam."""
import inspect

import torch

from morpheus_encoder.joint_objects import DifferentiableObjectFormation
from morpheus_encoder.joint_training import (
    JointReplicationTrainingHarness,
    observed_signed_evidence,
)
from morpheus_encoder.object_critic import ReplicationObjectCritic


def _harness():
    torch.manual_seed(71)
    return JointReplicationTrainingHarness(
        DifferentiableObjectFormation(d_slot=5, n_slots=4, n_objects=3, max_order=4),
        ReplicationObjectCritic(temperature=0.7, null_weight=0.8, complexity_weight=0.02),
    )


def _inputs(batch=13):
    torch.manual_seed(72)
    p = torch.randn(batch, 4, 5)
    r = torch.randn(batch + 2, 4, 5)
    n = torch.randn(2, batch + 1, 4, 5)
    return p, torch.randn(batch), r, torch.randn(batch + 2), n, torch.randn(2, batch + 1)


def test_contract_has_only_observed_slots_and_responses_not_hidden_labels():
    parameters = set(inspect.signature(JointReplicationTrainingHarness.forward).parameters)
    assert "truth" not in parameters
    assert "labels" not in parameters
    h = _harness()
    assert not hasattr(h, "truth")
    assert "truth" not in vars(h)


def test_replication_forward_values_are_exactly_proposal_frozen_values():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    output = h(p, py, r, ry, n, ny)
    expected = h.formation.evaluate_frozen(r, output.selection)
    # Gradient routing must not alter the candidate a later compiler sees.
    assert torch.equal(output.replication_values.detach(), expected)
    assert not output.selection.membership.requires_grad


def test_cadence_locked_selection_cannot_be_rewritten_by_responses():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    with torch.no_grad():
        proposal_output, selection = h.prepare_dictionary(p)
    # Alter every observed response.  Candidate values must retain the exact
    # proposal-side syntax selected before either response was opened.
    output = h.forward_with_selection(
        p, py * -3.0, r, ry + 7.0, n, ny,
        selection=selection, proposal_output=h.formation(p),
    )
    expected = h.formation.evaluate_frozen(r, selection)
    assert torch.equal(output.replication_values.detach(), expected)
    assert torch.equal(output.selection.membership, selection.membership)


def test_observed_objective_reaches_formation_parameters_with_finite_gradients():
    h = _harness()
    output = h(*_inputs())
    output.loss.backward()
    assert h.formation.queries.grad is not None
    assert torch.isfinite(h.formation.queries.grad).all()
    assert h.formation.arity_head[-1].weight.grad is not None
    assert torch.isfinite(h.formation.arity_head[-1].weight.grad).all()


def test_objective_uses_a_fixed_topk_not_a_diluting_full_bank_mean():
    h = _harness()
    value = torch.tensor([-4.0, -2.0, 0.2, 0.8, 1.0])
    # The default k=4 deliberately retains a small diversified discovery
    # bank, while avoiding a hundred opened but irrelevant slots erasing the
    # gradient of the best complete candidate.
    assert torch.isclose(h.aggregate_critic_value(value), torch.tensor(0.0))
    one = JointReplicationTrainingHarness(h.formation, h.critic, objective_topk=1)
    assert torch.isclose(one.aggregate_critic_value(value), torch.tensor(1.0))


def test_structured_null_has_a_penalty_and_shared_response_is_supported():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    output = h(p, py, r, ry, n, ny)
    assert torch.isfinite(output.critic.null_penalty).all()
    assert torch.all(output.critic.null_penalty >= 0)
    # A response can be held fixed across spatially transformed null inputs.
    shared = h(p, py, r, ry, n, torch.randn(n.shape[1]))
    assert shared.null_values.shape[:2] == n.shape[:2]


def test_signed_evidence_is_differentiable_and_directional():
    values = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
    positive = observed_signed_evidence(values, torch.tensor([0.0, 1.0, 2.0]))
    negative = observed_signed_evidence(values, torch.tensor([2.0, 1.0, 0.0]))
    assert positive.mean() > 0.9
    assert negative.mean() < -0.9
    positive.mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_signed_evidence_constant_coordinate_has_finite_zero_variance_gradient():
    # Regression for the N104 A100 smoke: sqrt(var).clamp_min has a finite
    # forward value at zero variance but a 0*infinity backward path.
    values = torch.zeros(5, 2, requires_grad=True)
    evidence = observed_signed_evidence(values, torch.arange(5, dtype=torch.float32))
    evidence.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_explicit_compiler_atomic_channel_scores_and_gradients_are_finite():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    # These are explicit named/readout coordinates; they are not derived from
    # DifferentiableObjectFormation.activity.
    pa = torch.randn(p.shape[0], 4, requires_grad=True)
    ra = torch.randn(r.shape[0], 4, requires_grad=True)
    na = torch.randn(n.shape[0], n.shape[1], 4, requires_grad=True)
    output = h(p, py, r, ry, n, ny,
               proposal_atomic_values=pa,
               replication_atomic_values=ra,
               null_atomic_values=na,
               atomic_names=("slot0", "slot1", "slot2", "slot3"))
    assert output.atomic_candidate_count == 4
    assert output.relational_candidate_count == h.formation.n_objects
    assert output.proposal_values.shape[1] == 4 + h.formation.n_objects
    assert [p["id"] for p in output.candidate_provenance[:4]] == [
        "slot0", "slot1", "slot2", "slot3"]
    assert all(p["kind"] == "frozen_relation" for p in output.candidate_provenance[4:])
    output.loss.backward()
    for tensor in (pa, ra, na):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_atomic_augmentation_preserves_exact_frozen_relation_forward_values():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    output = h(p, py, r, ry, n, ny,
               proposal_atomic_values=torch.randn(p.shape[0], 2),
               replication_atomic_values=torch.randn(r.shape[0], 2),
               null_atomic_values=torch.randn(n.shape[0], n.shape[1], 2),
               atomic_names=("slot0", "slot1"))
    expected = h.formation.evaluate_frozen(r, output.selection)
    assert torch.equal(output.relational_replication_values.detach(), expected)
    assert torch.equal(output.replication_values[:, 2:].detach(), expected)


def test_declared_atomic_parents_are_optional_and_only_penalize_relations():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    # The named atomic readout is deliberately strongly associated in both
    # environments; relation parents are still selected only from the frozen
    # input-only dictionary, never from this response.
    pa = torch.stack((py, py * 0.5, -py, torch.zeros_like(py)), dim=1)
    ra = torch.stack((ry, ry * 0.5, -ry, torch.zeros_like(ry)), dim=1)
    na = torch.randn(n.shape[0], n.shape[1], 4)
    base = h(p, py, r, ry, n, ny,
             proposal_atomic_values=pa, replication_atomic_values=ra,
             null_atomic_values=na, atomic_names=("slot0", "slot1", "slot2", "slot3"))
    nested = h(p, py, r, ry, n, ny,
               proposal_atomic_values=pa, replication_atomic_values=ra,
               null_atomic_values=na, atomic_names=("slot0", "slot1", "slot2", "slot3"),
               derive_atomic_parent_evidence=True)
    assert torch.equal(base.relational_replication_values.detach(),
                       nested.relational_replication_values.detach())
    assert torch.allclose(nested.critic.parent_novelty_penalty[:4],
                          torch.zeros(4), atol=1e-7)
    assert torch.isfinite(nested.critic.parent_novelty_penalty).all()
    assert nested.critic.parent_novelty_penalty[4:].max() > 0


def test_harness_passes_predeclared_parent_evidence_without_rewriting_dictionary():
    h = _harness()
    p, py, r, ry, n, ny = _inputs()
    # Relation-only candidate width is h.formation.n_objects.  Parents are
    # evidence tensors, deliberately supplied by the caller rather than
    # inferred from relation names or response values inside the harness.
    parent_p = torch.full((2, p.shape[0], h.formation.n_objects), 2.0)
    parent_r = torch.full((2, r.shape[0], h.formation.n_objects), 2.0)
    output = h(
        p, py, r, ry, n, ny,
        parent_discovery_evidence=parent_p,
        parent_replication_evidence=parent_r,
        parent_complexity=torch.ones(2, h.formation.n_objects),
    )
    assert torch.all(output.critic.parent_novelty_penalty > 0.9)
    expected = h.formation.evaluate_frozen(r, output.selection)
    assert torch.equal(output.replication_values.detach(), expected)
