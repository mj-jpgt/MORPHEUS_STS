from types import SimpleNamespace

from dbench.metrics import (Claim, b80, hard_solve_rate, object_error_class,
                             object_fidelity, object_fidelity_report)
from dbench.world import Hypothesis


def _truth():
    return Hypothesis(id="h", mode="relational", operator="AND",
                      operands=("A", "B"), target="Y",
                      context={"C": 1}, lag=2,
                      params={"tau": .5}, true=True)


def _claim(operator="AND", operands=("A", "B"), context=None, lag=2):
    return Claim(id="c", operator=operator, operands=operands, target="Y",
                 context={"C": 1} if context is None else context, lag=lag,
                 params={"tau": .5})


def test_full_object_scores_one_but_ancestor_is_not_near_full():
    truth = _truth()
    assert object_fidelity(_claim(), truth)["object_fidelity"] == 1.0
    ancestor = _claim(operator="IDENTITY", operands=("A",), context={}, lag=None)
    f = object_fidelity(ancestor, truth)
    assert f["object_fidelity"] == 0.0
    assert object_error_class(ancestor, truth) == "strict_ancestor"


def test_wrong_operator_and_wrong_context_have_explicit_labels():
    truth = _truth()
    assert object_error_class(_claim(operator="XOR"), truth) == "correct_operands_wrong_operator"
    assert object_error_class(_claim(context={"C": 0}), truth) == "correct_relation_wrong_context"


def test_report_preserves_partial_claim_as_partial_and_hard_rate_requires_labels():
    truth = SimpleNamespace(objects=[_truth()])
    report = object_fidelity_report([_claim(operator="IDENTITY", operands=("A",), context={}, lag=None)], truth)
    assert report["counts"] == {"strict_ancestor": 1}
    hard = hard_solve_rate([{"hard": True, "object_fidelity_report": report}])
    assert hard["HardSolveRate"] == 0.0


def test_b80_never_extrapolates():
    assert b80({10: .5, 100: .81})["B80"] == 100
    assert b80({10: .5})["B80"] is None
