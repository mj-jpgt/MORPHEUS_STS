import pytest

from morpheus_encoder.capability_metrics import (
    adjust_candidate_pvalues_arm_family,
    adjust_family_pvalues,
    budget_to_80,
    discovery_vs_budget,
    paired_bootstrap_ci,
    paired_sign_permutation_test,
    sealed_stagewise_summary,
    structured_object_fidelity,
    validate_packet_receipts,
    validated_recovery_at_budget,
)


def _receipt(arm, packet_id, candidates, *, family="and"):
    return {"arm": arm, "family": family, "packet_id": packet_id, "candidates": candidates}


def test_structured_fidelity_keeps_a_partial_recovery_visible():
    truth = {"variables": {"A", "B", "C"}, "context": {"tissue": "kidney"},
             "relation": "nonlinear", "environment": {"cohort_a", "cohort_b"}}
    partial = {"variables": {"A", "B"}, "context": {"tissue": "kidney"},
               "relation": "additive", "environment": {"cohort_a", "cohort_b"}}
    result = structured_object_fidelity(truth, partial)
    assert result["components"]["variables"] == pytest.approx(2 / 3)
    assert result["components"]["relation"] == 0.0
    assert 0.0 < result["object_fidelity"] < 1.0


def test_fixed_budget_curve_and_b80_are_packet_level_and_safe():
    rows = [
        _receipt("morpheus", "p1", [{"rank": 1, "validated": True, "object_fidelity": 1.0}]),
        _receipt("morpheus", "p2", [{"rank": 2, "validated": True, "object_fidelity": 1.0}]),
        _receipt("morpheus", "p3", [{"rank": 3, "validated": False, "object_fidelity": 1.0}]),
    ]
    at_one = validated_recovery_at_budget(rows, budget=1)
    assert at_one["n_packets"] == 3
    assert at_one["recovery_rate"] == pytest.approx(1 / 3)
    curve = discovery_vs_budget(rows, max_budget=3)
    assert curve["recovery_curve"] == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    assert curve["discovery_auc"] == pytest.approx(5 / 9)
    assert budget_to_80(curve)["available"] is False
    assert budget_to_80(curve)["reason"] == "target_not_reached"


def test_paired_comparisons_preserve_packet_pairing_and_reject_duplicates():
    rows = []
    for packet, ssl, morph in [("p1", 0.1, 0.4), ("p2", 0.2, 0.6), ("p3", 0.3, 0.7)]:
        rows.extend([
            {"arm": "ssl", "family": "context", "packet_id": packet, "score": ssl},
            {"arm": "morpheus", "family": "context", "packet_id": packet, "score": morph},
        ])
    ci = paired_bootstrap_ci(rows, candidate_arm="morpheus", reference_arm="ssl",
                             metric="score", n_bootstrap=200, seed=3)
    assert ci["n_paired_packets"] == 3
    assert ci["mean_difference"] == pytest.approx((0.3 + 0.4 + 0.4) / 3)
    test = paired_sign_permutation_test(rows, candidate_arm="morpheus", reference_arm="ssl", metric="score")
    assert test["exact"] is True
    assert test["n_paired_packets"] == 3
    with pytest.raises(ValueError, match="duplicate"):
        validate_packet_receipts(rows + [dict(rows[0])])


def test_bh_adjustment_is_monotonic_and_family_wide():
    q = adjust_family_pvalues({"and": 0.01, "spatial": 0.04, "lag": 0.20})
    assert q["and"] == pytest.approx(0.03)
    assert q["spatial"] == pytest.approx(0.06)
    assert q["lag"] == pytest.approx(0.20)


def test_sealed_stagewise_summary_uses_frozen_summary_not_truth_or_rows():
    stages = [
        {"accessibility": 0.8, "candidate_exposed": True, "candidate_rank": 2,
         "relation_validated": True, "null_survived": True, "decoy_promoted": False},
        {"accessibility": 0.4, "candidate_exposed": False, "candidate_rank": None,
         "relation_validated": False, "null_survived": False, "decoy_promoted": True},
    ]
    rows = [
        {"arm": "morpheus", "family": "spatial", "packet_id": "p1", "sealed_stages": stages[0],
         "truth": {"this_must_not_be_read": "one"}, "n_spots": 10000},
        {"arm": "morpheus", "family": "spatial", "packet_id": "p2", "sealed_stages": stages[1],
         "truth": {"this_must_not_be_read": "different"}, "n_spots": 2},
    ]
    summary = sealed_stagewise_summary(rows)
    group = summary["by_arm_family"]["morpheus::spatial"]
    assert group["n_packets"] == 2
    assert group["candidate_exposure_rate"] == pytest.approx(0.5)
    assert group["mean_accessibility"] == pytest.approx(0.6)
    assert group["clean_stagewise_pass_rate"] == pytest.approx(0.5)


def test_candidate_bh_uses_all_packets_in_arm_family_not_per_packet():
    rows = [
        {"arm": "ssl", "family": "and", "packet_id": "p1",
         "candidate_pvalues": {"a": 0.01, "b": 0.04}},
        {"arm": "ssl", "family": "and", "packet_id": "p2",
         "candidate_pvalues": [{"candidate_id": "c", "p_value": 0.20}]},
    ]
    report = adjust_candidate_pvalues_arm_family(rows)
    candidates = report["by_arm_family"]["ssl::and"]["candidates"]
    assert report["by_arm_family"]["ssl::and"]["n_candidates"] == 3
    assert [x["q_value"] for x in candidates] == pytest.approx([0.03, 0.06, 0.20])
