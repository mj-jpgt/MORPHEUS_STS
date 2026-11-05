import json

from dbench.metrics import Claim
from dbench.world import Hypothesis
from dbench.object_cards import SCHEMA_VERSION, build_object_card


def _claim():
    return Claim(
        id="candidate-17", operator="AND", operands=("A", "B"),
        target="Y", context={"regime": "do_treat"}, lag=2,
        effect=0.41, p_value=0.01, params={"radius": 25.0},
        replication={"cohort-2": {"effect": 0.38, "p": 0.02}},
    )


def test_object_card_is_complete_json_and_preserves_missing_evidence():
    card = build_object_card(_claim())
    assert card["schema_version"] == SCHEMA_VERSION
    assert card["object"]["operands"] == ["A", "B"]
    assert card["physical_scale"] == {"spatial_radius": 25.0,
                                       "temporal_lag": 2,
                                       "context": {"regime": "do_treat"}}
    assert card["evaluation_only"]["object_fidelity"] is None
    assert card["nearest_known_analogues"] == []
    assert card["suggested_discriminating_experiment"] == {}
    json.dumps(card, sort_keys=True)


def test_truth_is_evaluation_only_and_does_not_mutate_the_candidate_statement():
    h = Hypothesis(id="truth", mode="relational", operator="AND",
                   operands=("A", "B"), target="Y",
                   context={"regime": "do_treat"}, lag=2,
                   params={"radius": 25.0})
    card = build_object_card(
        _claim(), truth=h,
        lower_order_effects={"A": {"effect": 0.02}, "B": {"effect": 0.01}},
        unique_higher_order_effect={"conditional_effect": 0.41},
        suggested_discriminating_experiment={"action": "clamp_A_high"},
    )
    assert card["object"]["operator"] == "AND"
    assert card["evaluation_only"]["object_fidelity"]["object_fidelity"] == 1.0
    assert card["unique_higher_order_effect"]["conditional_effect"] == 0.41
    assert card["suggested_discriminating_experiment"]["action"] == "clamp_A_high"


# ==========================================================================
# D6 / C2: schema v2 additions on top of the (unchanged) v1 keys above --
# `replication_level`, `structured_nulls`, `novelty`, `causal_tier` (with
# auto-downgrade), `failure_conditions`, and `card_from_receipt`.
# ==========================================================================
import pytest

from dbench.object_cards import (
    CAUSAL_TIERS, card_from_receipt, IMPORTANCE_DIMENSIONS, SCALE_FIELDS,
    score_importance, rank_by_importance, evidence_statement,
)


_V1_KEYS = {
    "schema_version", "object", "evaluation_only", "physical_scale",
    "lower_order_effects", "unique_higher_order_effect",
    "functional_response_surface", "constituent_ablations",
    "nuisance_sensitivity", "stability", "nearest_simpler_explanation",
    "nearest_known_analogues", "uncertainty",
    "suggested_discriminating_experiment", "external_validation",
    "translational_evidence", "provenance",
}


def test_v2_card_keeps_every_v1_key():
    card = build_object_card(_claim())
    assert _V1_KEYS <= set(card.keys())
    assert card["schema_version"] == SCHEMA_VERSION


def test_v2_adds_replication_level_structured_nulls_novelty_and_failure_conditions():
    card = build_object_card(
        _claim(),
        replication_level=2,
        structured_nulls=[{"kind": "spatial_block", "p": 0.01, "n": 199}],
        novelty={"label": "compositional_novelty", "claim_hash": "abc123"},
        failure_conditions=["sign reverses on external transfer"],
    )
    assert card["replication_level"] == 2
    assert card["structured_nulls"] == [{"kind": "spatial_block", "p": 0.01, "n": 199}]
    assert card["novelty"]["label"] == "compositional_novelty"
    assert card["failure_conditions"] == ["sign reverses on external transfer"]
    assert card["causal_tier"] == "association"
    assert card["causal_tier_downgraded"] is False


def test_invalid_replication_level_rejected():
    with pytest.raises(ValueError):
        build_object_card(_claim(), replication_level=7)


def test_invalid_causal_tier_rejected():
    with pytest.raises(ValueError):
        build_object_card(_claim(), causal_tier="definitely_causal")


def test_causal_tier_downgrades_on_causal_word_in_statement():
    card = build_object_card(
        _claim(), causal_tier="mechanistic_triangulation",
        statement="This program is necessary for the inflamed phenotype.",
    )
    assert card["causal_tier"] == "association"
    assert card["causal_tier_requested"] == "mechanistic_triangulation"
    assert card["causal_tier_downgraded"] is True
    assert card["causal_tier_downgrade_reason"]


def test_causal_tier_not_downgraded_when_statement_is_clean():
    card = build_object_card(
        _claim(), causal_tier="perturbational",
        statement="Removing this program reduces the score by 0.4.",
    )
    assert card["causal_tier"] == "perturbational"
    assert card["causal_tier_downgraded"] is False


def test_all_causal_tiers_round_trip_with_no_statement():
    for tier in CAUSAL_TIERS:
        card = build_object_card(_claim(), causal_tier=tier)
        assert card["causal_tier"] == tier
        assert card["causal_tier_downgraded"] is False


def _held_out_fold_receipt_row():
    return {
        "candidate_hash": "cand-abc123",
        "gene_symbols": ["HIF1A", "VEGFA"],
        "feature_form": "product",
        "fold": 2,
        "proposal_gain": 0.09,
        "replication_gain": 0.05,
        "null_p_value": 0.01,
        "null_gains": [0.0] * 199,
        "nulls": {"spatial_block": {"p": 0.02, "n": 199},
                 "within_patient_perm": {"p": 0.03, "n": 199}},
        "baseline_width": 45,
        "context": {"target_family": "spatial_organisation"},
    }


def test_card_from_receipt_builds_flexible_product_operator_from_gene_symbols():
    card = card_from_receipt(_held_out_fold_receipt_row())
    assert card["object"]["operator"] == "flexible_product"
    assert card["object"]["operands"] == ["HIF1A", "VEGFA"]
    assert card["object"]["id"] == "cand-abc123"


def test_card_from_receipt_infers_replication_level_1_for_held_out_fold():
    card = card_from_receipt(_held_out_fold_receipt_row())
    assert card["replication_level"] == 1


def test_card_from_receipt_with_no_fold_signal_defaults_to_level_0():
    row = {"candidate_hash": "c2", "gene_symbols": ["A", "B"],
          "feature_form": "linear_subspace", "proposal_gain": 0.1}
    card = card_from_receipt(row)
    assert card["object"]["operator"] == "flexible_subspace"
    assert card["replication_level"] == 0


def test_card_from_receipt_carries_structured_nulls_and_ledger_novelty():
    row = _held_out_fold_receipt_row()
    ledger_rec = {"candidate_id": "cand-abc123", "stage": "internal",
                 "evidence": {"novelty": {"label": "unresolved", "claim_hash": "h1"}}}
    card = card_from_receipt(row, ledger_rec)
    kinds = {n["kind"] for n in card["structured_nulls"]}
    assert kinds == {"spatial_block", "within_patient_perm"}
    assert card["novelty"]["label"] == "unresolved"
    assert card["provenance"]["ledger_stage"] == "internal"


def test_card_from_receipt_downgrades_causal_tier_with_statement():
    row = _held_out_fold_receipt_row()
    card = card_from_receipt(row, causal_tier="causal",
                             statement="HIF1A causes the hypoxic response.")
    assert card["causal_tier"] == "association"
    assert card["causal_tier_downgraded"] is True


# ==========================================================================
# E2: `compression`/`scale`/`human_involvement` blocks, `score_importance`,
# `rank_by_importance`, and `evidence_statement`.
# ==========================================================================

def test_build_object_card_carries_compression_scale_and_human_involvement():
    compression = {"compressed": True, "accepted": [{"form": "threshold_step",
                                                      "complexity": 2, "gain": 0.4}]}
    scale = {"entity_type": "spot", "spatial_support": "55um_visium_spot",
            "temporal_support": "single_timepoint", "modality": "spatial_transcriptomics",
            "biological_scale": "tissue_neighborhood", "containment": {"spot": "tissue_section"}}
    human_involvement = [{"who": "scientist-1", "what": "selected this candidate for follow-up",
                          "why": "feasibility", "when": "2026-09-25"}]
    card = build_object_card(_claim(), compression=compression, scale=scale,
                             human_involvement=human_involvement)
    assert card["compression"]["compressed"] is True
    assert card["compression"]["accepted"][0]["form"] == "threshold_step"
    assert set(card["scale"]) == set(SCALE_FIELDS)
    assert card["human_involvement"] == human_involvement


def test_build_object_card_defaults_compression_scale_human_involvement_to_empty():
    card = build_object_card(_claim())
    assert card["compression"] == {}
    assert card["scale"] == {}
    assert card["human_involvement"] == []


def test_card_from_receipt_derives_scale_from_row_context():
    row = _held_out_fold_receipt_row()
    row["context"] = dict(row["context"], scale={
        "entity_type": "spot", "spatial_support": "55um_visium_spot",
        "temporal_support": "single_timepoint", "modality": "spatial_transcriptomics",
        "biological_scale": "tissue_neighborhood", "containment": {"spot": "tissue_section"},
    })
    card = card_from_receipt(row)
    assert card["scale"]["entity_type"] == "spot"
    assert card["scale"]["modality"] == "spatial_transcriptomics"


def test_score_importance_table_has_every_dimension_and_never_collapses_to_a_scalar():
    table = score_importance({
        "explanatory_gain": 0.4,
        "robustness": {"fold_0": 0.3, "fold_1": 0.35},
        "effect_magnitude": 0.41,
        "important_quantity_tags": ["overall_survival"],
        "competing_explanations": ["nuisance_depth_confound"],
        "regime_description": "hypoxic renal tumor regions only",
        "replication_level": 2,
        "potential_consequences": ["biomarker for treatment stratification"],
    })
    assert set(table) == set(IMPORTANCE_DIMENSIONS)
    assert "importance_score" not in table
    for dim, entry in table.items():
        assert entry["assessed"] is True
        assert "value" in entry and "evidence" in entry
    assert table["robustness"]["value"]["n_environments"] == 2
    assert table["reproducibility"]["value"] == 2


def test_score_importance_reports_unassessed_dimensions_explicitly():
    table = score_importance({})
    assert all(entry["assessed"] is False for entry in table.values())
    assert all(entry["value"] in ([], None) for entry in table.values())


def test_rank_by_importance_refuses_card_without_novelty():
    card = build_object_card(_claim(), replication_level=1)  # novelty defaults to {}
    with pytest.raises(ValueError):
        rank_by_importance([card])


def test_rank_by_importance_refuses_card_without_replication_level():
    card = build_object_card(_claim(), novelty={"label": "known"})  # replication_level None
    with pytest.raises(ValueError):
        rank_by_importance([card])


def test_rank_by_importance_orders_by_explicit_rank_key():
    weak = build_object_card(_claim(), replication_level=1, novelty={"label": "known"})
    weak["importance"] = score_importance({"explanatory_gain": 0.1})
    strong = build_object_card(_claim(), replication_level=3, novelty={"label": "compositional_novelty"})
    strong["importance"] = score_importance({
        "explanatory_gain": 0.6, "robustness": {"a": 0.5}, "effect_magnitude": 0.6,
        "important_quantity_tags": ["survival"], "competing_explanations": ["x"],
        "regime_description": "broad", "replication_level": 3,
        "potential_consequences": ["y"],
    })
    ranked = rank_by_importance([weak, strong])
    assert ranked[0]["card"] is strong
    assert ranked[0]["rank_key"]["n_dimensions_assessed"] == 8
    assert ranked[1]["card"] is weak


def test_evidence_statement_uses_association_wording_for_association_tier():
    card = build_object_card(_claim(), causal_tier="association")
    text = evidence_statement(card)
    assert "associated with" in text
    assert "causes" not in text


def test_evidence_statement_refuses_causal_wording_on_a_non_causal_tier():
    card = build_object_card(_claim(), causal_tier="directional")
    with pytest.raises(ValueError):
        evidence_statement(card, statement="A causes Y directly.")


def test_evidence_statement_allows_causal_wording_on_the_causal_tier():
    card = build_object_card(_claim(), causal_tier="causal")
    text = evidence_statement(card, statement="A causes Y.")
    assert text == "A causes Y."


def test_card_carries_explanation_block_from_row_or_keyword_and_defaults_empty():
    assert card_from_receipt(_held_out_fold_receipt_row())["explanation"] == {}
    block = {"level": "A", "statistical_levels": ["A"], "delta_add": 0.0, "delta_nonadd": 0.001}
    row = dict(_held_out_fold_receipt_row(), explanation=block)
    assert card_from_receipt(row)["explanation"]["level"] == "A"
    card = card_from_receipt(_held_out_fold_receipt_row(), explanation=dict(block, level="C"))
    assert card["explanation"]["level"] == "C"
    json.dumps(card)
