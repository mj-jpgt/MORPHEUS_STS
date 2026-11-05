"""Tests for the prospective dataset registry and its two campaign protocol
drafts (Seat D: dataset registry + discovery portfolio).

These tests only check structure/availability metadata; per the anti-leakage
rule for this seat, they never assert anything about a biological/physical
finding or outcome for any candidate dataset.
"""
import json
from pathlib import Path

import pytest

from dbench.prospective_campaign import load_protocol

PROTOCOLS_DIR = Path(__file__).resolve().parent.parent / "protocols"
REGISTRY_PATH = PROTOCOLS_DIR / "DATASET_REGISTRY.json"

REQUIRED_RECORD_FIELDS = {
    "id",
    "domain",
    "modalities",
    "entity_type",
    "spatial_support",
    "temporal_support",
    "n_entities",
    "n_patients_or_environments",
    "context_variables_available",
    "replication_routes",
    "licence_access",
    "on_store",
    "known_leakage_risks",
    "preprocessing_needs",
    "suitability_rank",
}

REQUIRED_REPLICATION_LEVELS = {
    "level_1_new_donors_same_acquisition",
    "level_2_independent_cohort",
    "level_3_orthogonal_modality",
    "level_4_perturbation",
}

# Keys that would smuggle a discovery result/outcome/answer into a dataset
# record. The registry ranks datasets by suitability BEFORE seeing any
# result, so none of these may appear anywhere in it.
FORBIDDEN_KEY_SUBSTRINGS = (
    "outcome", "answer", "finding", "discovery_result", "p_value_observed",
    "effect_size", "correlation_observed", "replication_gain_observed",
)


def _iter_all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_all_keys(item)


@pytest.fixture(scope="module")
def registry():
    assert REGISTRY_PATH.exists(), f"missing {REGISTRY_PATH}"
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_loads_and_has_top_level_fields(registry):
    assert registry["schema"] == "morpheus_dataset_registry/v1"
    assert "ranked_at_utc" in registry and registry["ranked_at_utc"]
    assert "ranking_rule" in registry and len(registry["ranking_rule"]) > 50
    assert isinstance(registry["records"], list) and len(registry["records"]) >= 8


def test_every_record_has_required_fields(registry):
    for rec in registry["records"]:
        missing = REQUIRED_RECORD_FIELDS - set(rec)
        assert not missing, f"record {rec.get('id')} missing fields {missing}"
        missing_levels = REQUIRED_REPLICATION_LEVELS - set(rec["replication_routes"])
        assert not missing_levels, f"record {rec['id']} missing replication levels {missing_levels}"
        assert isinstance(rec["known_leakage_risks"], list)
        assert isinstance(rec["on_store"], list)


def test_ranks_are_unique_and_dense(registry):
    ranks = [rec["suitability_rank"] for rec in registry["records"]]
    assert len(ranks) == len(set(ranks)), "suitability_rank values must be unique"
    assert sorted(ranks) == list(range(1, len(ranks) + 1)), (
        "suitability_rank must be a dense 1..N ordering with no gaps or ties"
    )


def test_record_ids_are_unique(registry):
    ids = [rec["id"] for rec in registry["records"]]
    assert len(ids) == len(set(ids))


def test_registry_never_records_an_outcome_or_answer(registry):
    keys = {str(k).lower() for k in _iter_all_keys(registry)}
    for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
        hits = {k for k in keys if forbidden in k}
        assert not hits, f"registry contains a result-shaped key: {hits}"


def test_at_least_one_non_biological_domain_present(registry):
    domains = [rec["domain"] for rec in registry["records"]]
    assert any("non-biological" in d for d in domains), (
        "registry must include at least one non-biological discovery portfolio"
    )


def test_at_least_one_dataset_flagged_for_public_acquisition(registry):
    """hest_coad is catalogued (59 rows / 6 patients in HEST_v1_3_0.csv) but
    has zero matching st/wsi/cellvit files physically present on the store;
    it must be honestly recorded as needing acquisition, not silently
    dropped or misreported as available.
    """
    by_id = {rec["id"]: rec for rec in registry["records"]}
    assert "hest_coad" in by_id
    coad = by_id["hest_coad"]
    assert coad["n_entities"] == 0 and coad["on_store"] == []
    assert "NEEDS PUBLIC ACQUISITION" in coad["preprocessing_needs"]


def test_below_threshold_cohorts_are_ranked_below_cohorts_that_clear_it(registry):
    """The predeclared ranking rule puts every dataset with fewer than 10
    patients/environments below every dataset that clears that bar."""
    by_id = {rec["id"]: rec for rec in registry["records"]}
    at_or_above = [rec["suitability_rank"] for rec in registry["records"]
                  if isinstance(rec["n_patients_or_environments"], int)
                  and rec["n_patients_or_environments"] >= 10]
    below = [rec["suitability_rank"] for rec in registry["records"]
            if isinstance(rec["n_patients_or_environments"], int)
            and 0 < rec["n_patients_or_environments"] < 10]
    if at_or_above and below:
        assert max(at_or_above) < min(below)
    # the two headline campaign substrates must clear the threshold
    assert by_id["hest_ccrcc"]["n_patients_or_environments"] >= 10
    assert by_id["hest_breast_idc"]["n_patients_or_environments"] >= 10


@pytest.mark.parametrize("filename", [
    "hest_breast_idc_v01.json",
    "bmmc_cite_seq_v01.json",
])
def test_protocol_draft_passes_load_protocol(filename):
    protocol = load_protocol(PROTOCOLS_DIR / filename)
    assert protocol["discovery"]["candidate_budget"] == 50
    assert protocol["anti_leakage"]["literature_before_internal_freeze"] is False


@pytest.mark.parametrize("filename", [
    "hest_breast_idc_v01.json",
    "bmmc_cite_seq_v01.json",
])
def test_protocol_draft_matches_registry_source(filename, registry):
    protocol = json.loads((PROTOCOLS_DIR / filename).read_text(encoding="utf-8"))
    record_id = protocol["registry_source"]["record_id"]
    by_id = {rec["id"]: rec for rec in registry["records"]}
    assert record_id in by_id
    assert protocol["registry_source"]["suitability_rank"] == by_id[record_id]["suitability_rank"]


@pytest.mark.parametrize("filename", [
    "hest_breast_idc_v01.json",
    "bmmc_cite_seq_v01.json",
])
def test_protocol_draft_gates_have_honest_exists_flags(filename):
    """Every gate must declare `exists`; a gate marked exists=false must also
    be required=true so the campaign preflight will honestly refuse to
    claim it is ready for discovery."""
    protocol = json.loads((PROTOCOLS_DIR / filename).read_text(encoding="utf-8"))
    saw_a_false_gate = False
    for cohort in protocol["cohorts"]:
        for gate in cohort["gates"]:
            assert "exists" in gate, f"gate {gate.get('id')} in {filename} missing exists flag"
            if gate["exists"] is False:
                saw_a_false_gate = True
                assert gate.get("required", True) is True
    assert saw_a_false_gate, (
        f"{filename} should honestly record at least one not-yet-acquired gate"
    )


@pytest.mark.parametrize("filename", [
    "hest_breast_idc_v01.json",
    "bmmc_cite_seq_v01.json",
])
def test_protocol_draft_has_no_answer_or_outcome_fields(filename):
    protocol = json.loads((PROTOCOLS_DIR / filename).read_text(encoding="utf-8"))
    # `anti_leakage` is the schema's fixed set of leakage-guard flag NAMES
    # (e.g. `truth_or_outcome_in_discovery`, always False) -- it names the
    # absence of an outcome and is exempt from this scan; every other
    # section must still be free of result-shaped keys.
    scoped = {k: v for k, v in protocol.items() if k != "anti_leakage"}
    keys = {str(k).lower() for k in _iter_all_keys(scoped)}
    for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
        hits = {k for k in keys if forbidden in k}
        assert not hits, f"{filename} contains a result-shaped key: {hits}"
