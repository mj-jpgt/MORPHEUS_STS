"""Contracts for the general differentiable replication object critic."""
import torch
import pytest

from morpheus_encoder.object_critic import ReplicationObjectCritic


def _evidence(value, n=12):
    return torch.full((n, 1), float(value))


def test_replication_agreement_is_rewarded_and_disagreement_is_not():
    critic = ReplicationObjectCritic(temperature=1.0, null_weight=0.0,
                                     complexity_weight=0.0)
    agreeing = critic(_evidence(2.0), _evidence(2.0), _evidence(0.0), torch.zeros(1))
    opposing = critic(_evidence(2.0), _evidence(-2.0), _evidence(0.0), torch.zeros(1))
    assert float(agreeing.agreement[0]) > 0.9
    assert float(opposing.agreement[0]) < -0.9
    assert float(agreeing.value[0]) > float(opposing.value[0])


def test_structured_null_evidence_lowers_explanatory_value():
    critic = ReplicationObjectCritic(temperature=1.0, null_weight=1.0,
                                     complexity_weight=0.0)
    clean = critic(_evidence(2.0), _evidence(2.0), _evidence(0.0), torch.zeros(1))
    null_supported = critic(_evidence(2.0), _evidence(2.0), _evidence(2.0), torch.zeros(1))
    assert float(null_supported.null_penalty[0]) > float(clean.null_penalty[0])
    assert float(null_supported.value[0]) < float(clean.value[0])


def test_description_complexity_lowers_value():
    critic = ReplicationObjectCritic(temperature=1.0, null_weight=0.0,
                                     complexity_weight=0.25)
    simple = critic(_evidence(2.0), _evidence(2.0), _evidence(0.0), torch.zeros(1))
    complex_ = critic(_evidence(2.0), _evidence(2.0), _evidence(0.0), torch.tensor([8.0]))
    assert float(complex_.complexity_penalty[0]) > float(simple.complexity_penalty[0])
    assert float(complex_.value[0]) < float(simple.value[0])


def test_gradients_are_finite_for_soft_candidates_and_multiple_nulls():
    torch.manual_seed(17)
    discovery = torch.randn(9, 3, requires_grad=True)
    replication = torch.randn(7, 3, requires_grad=True)
    nulls = torch.randn(4, 6, 3, requires_grad=True)
    complexity = torch.rand(3, requires_grad=True)
    critic = ReplicationObjectCritic(temperature=0.7, null_weight=0.6,
                                     complexity_weight=0.1)
    output = critic(discovery, replication, nulls, complexity)
    (-output.value.mean()).backward()
    for tensor in (discovery, replication, nulls, complexity):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_replicated_lower_complexity_parent_reduces_child_novelty_value():
    """A child cannot receive full credit for evidence a parent already has."""
    critic = ReplicationObjectCritic(temperature=1.0, null_weight=0.0,
                                     parent_weight=0.75, complexity_weight=0.0)
    child_discovery = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    child_replication = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    no_parent = critic(child_discovery, child_replication, torch.zeros_like(child_discovery),
                       torch.tensor([2.0, 2.0]))
    # The first child has a replicating low-complexity parent.  The second has
    # only direction-conflicting parent evidence, which is not explanatory.
    with_parent = critic(
        child_discovery, child_replication, torch.zeros_like(child_discovery),
        torch.tensor([2.0, 2.0]),
        parent_discovery_evidence=torch.tensor([[[2.0, 2.0], [2.0, 2.0]]]),
        parent_replication_evidence=torch.tensor([[[2.0, -2.0], [2.0, -2.0]]]),
        parent_complexity=torch.tensor([[1.0, 1.0]]),
    )
    assert float(with_parent.parent_novelty_penalty[0]) > 0.9
    assert float(with_parent.parent_novelty_penalty[1]) == 0.0
    assert float(with_parent.value[0]) < float(no_parent.value[0])
    assert float(with_parent.value[1]) == pytest.approx(float(no_parent.value[1]))


def test_absent_parents_are_exactly_backward_compatible():
    critic = ReplicationObjectCritic(temperature=0.7, null_weight=0.4,
                                     parent_weight=0.9, complexity_weight=0.1)
    d, r, n = torch.randn(5, 3), torch.randn(4, 3), torch.randn(2, 6, 3)
    complexity = torch.tensor([1.0, 2.0, 3.0])
    first = critic(d, r, n, complexity)
    second = critic(d, r, n, complexity)
    assert torch.equal(first.value, second.value)
    assert torch.equal(first.parent_novelty_penalty, torch.zeros_like(first.value))
    assert torch.equal(first.parent_explanatory_strength, torch.zeros_like(first.value))


def test_parent_stack_gradients_are_finite_and_candidate_mask_is_respected():
    torch.manual_seed(33)
    d = torch.randn(6, 2, requires_grad=True)
    r = torch.randn(5, 2, requires_grad=True)
    n = torch.randn(3, 4, 2, requires_grad=True)
    pd = torch.randn(3, 6, 2, requires_grad=True)
    pr = torch.randn(3, 5, 2, requires_grad=True)
    critic = ReplicationObjectCritic(parent_weight=0.6)
    result = critic(
        d, r, n, torch.tensor([2.0, 2.0]),
        parent_discovery_evidence=pd,
        parent_replication_evidence=pr,
        parent_complexity=torch.ones(3, 2),
        parent_mask=torch.tensor([[True, False], [False, False], [True, True]]),
    )
    (-result.value.mean()).backward()
    for tensor in (d, r, n, pd, pr):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    # Candidate one does have an active parent only through the third row;
    # candidate zero has two.  Neither receives a hidden unmasked parent.
    assert torch.isfinite(result.parent_novelty_penalty).all()


@pytest.mark.parametrize(
    "parent_d,parent_r,parent_complexity,parent_mask,match",
    [
        (torch.ones(2, 4, 1), torch.ones(3, 4, 1), torch.ones(2, 1), None,
         "same parent count"),
        (torch.ones(2, 4, 2), torch.ones(2, 4, 2), torch.ones(2, 2), None,
         "candidate width"),
        (torch.ones(2, 4, 1), torch.ones(2, 4, 1), torch.ones(2, 1), torch.ones(2, 1),
         "parent_mask"),
    ],
)
def test_parent_stack_contract_is_validated(parent_d, parent_r, parent_complexity, parent_mask, match):
    critic = ReplicationObjectCritic()
    with pytest.raises(ValueError, match=match):
        critic(
            _evidence(1.0), _evidence(1.0), _evidence(0.0), torch.tensor([2.0]),
            parent_discovery_evidence=parent_d,
            parent_replication_evidence=parent_r,
            parent_complexity=parent_complexity,
            parent_mask=parent_mask,
        )


def test_active_parent_must_have_strictly_lower_declared_complexity():
    critic = ReplicationObjectCritic()
    with pytest.raises(ValueError, match="strictly lower"):
        critic(
            _evidence(1.0), _evidence(1.0), _evidence(0.0), torch.tensor([1.0]),
            parent_discovery_evidence=torch.ones(1, 3, 1),
            parent_replication_evidence=torch.ones(1, 3, 1),
            parent_complexity=torch.tensor([[1.0]]),
        )
