"""Test A for the candidate ledger: data exists (a real ledger file is
created and populated), loader works (events + funnel round-trip through a
fresh CandidateLedger instance reading the same file), ledger round-trips
(re-reading after process restart gives identical funnel counts) -- plus
the immutability enforcement the ledger exists for.
"""
import json
import os

import pytest

from dbench.ledger import (CandidateLedger, LedgerError, object_hash,
                           LEDGER_EVIDENCE_FIELDS, complete_evidence)


def test_data_exists_and_loader_works(tmp_path):
    p = str(tmp_path / "campaign_a.jsonl")
    led = CandidateLedger(p)
    led.record("C001", "proposed", "CAUSE", ("BAP1_status", "immune_score"),
              parameters={"cohort": "ccrcc"}, data_ref="cptac_ccrcc_v1",
              data_sha256="abc123", evidence={"p": 0.01})
    led.record("C002", "proposed", "ADD", ("geneA",), data_ref="cptac_ccrcc_v1")
    assert os.path.exists(p)
    assert os.path.getsize(p) > 0
    # loader: a FRESH instance reading the same file sees the same events
    led2 = CandidateLedger(p)
    events = led2.all_events()
    assert len(events) == 2
    assert {e["candidate_id"] for e in events} == {"C001", "C002"}


def test_ledger_round_trips(tmp_path):
    p = str(tmp_path / "campaign_b.jsonl")
    led = CandidateLedger(p)
    led.record("C001", "proposed", "ADD", ("g1",))
    led.record("C001", "internal", "ADD", ("g1",), evidence={"RDC": 1})
    f1 = led.funnel()
    # simulate a fresh process: re-open the same path, no in-memory state
    led_fresh = CandidateLedger(p)
    f2 = led_fresh.funnel()
    assert f1 == f2
    assert f2["N_proposed"] == 1 and f2["N_internal"] == 1
    assert f2["N_external"] == 0


def test_funnel_counts_monotonic_and_correct(tmp_path):
    p = str(tmp_path / "campaign_c.jsonl")
    led = CandidateLedger(p)
    # 5 proposed, 3 reach internal, 2 reach external, 1 reaches actionable
    for i in range(5):
        led.record("C%d" % i, "proposed", "ADD", ("g%d" % i,))
    for i in range(3):
        led.record("C%d" % i, "internal", "ADD", ("g%d" % i,))
    for i in range(2):
        led.record("C%d" % i, "external", "ADD", ("g%d" % i,),
                  data_ref="external_cohort_1")
    led.record("C0", "orthogonal", "ADD", ("g0",))
    led.record("C0", "actionable", "ADD", ("g0",))
    led.record("C4", "proposed", "ADD", ("g4",), stop_reason="no replication")
    f = led.funnel()
    assert f["N_proposed"] == 5
    assert f["N_internal"] == 3
    assert f["N_external"] == 2
    assert f["N_orthogonal"] == 1
    assert f["N_actionable"] == 1


def test_frozen_object_cannot_be_redefined_after_external():
    obj = dict(operator="CAUSE", operands=("BAP1_status", "immune_score"),
              parameters={"cohort": "ccrcc"})

    def make(path):
        led = CandidateLedger(path)
        led.record("C_BAP1", "proposed", **obj)
        led.record("C_BAP1", "internal", **obj, evidence={"RDC": 1})
        led.record("C_BAP1", "external", **obj, data_ref="cptac_ccrcc",
                  evidence={"VDC": 1})
        return led

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "campaign_d.jsonl")
        led = make(p)
        # same object, different stage: fine
        led.record("C_BAP1", "orthogonal", **obj)
        # REDEFINED object (different operands) after external: must refuse
        bad = dict(obj)
        bad["operands"] = ("BAP1_status", "a_different_score")
        try:
            led.record("C_BAP1", "orthogonal", **bad)
            assert False, "expected LedgerError, none raised"
        except LedgerError:
            pass
        # the refusal must not have appended a corrupt/partial line
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        for line in lines:
            json.loads(line)   # every line still parses


def test_object_hash_is_order_and_type_insensitive_where_it_should_be():
    h1 = object_hash("ADD", ("g1", "g2"), parameters={"a": 1, "b": 2})
    h2 = object_hash("ADD", ("g2", "g1"), parameters={"b": 2, "a": 1})
    assert h1 == h2, "operand order and parameter key order must not matter"
    h3 = object_hash("ADD", ("g1", "g3"), parameters={"a": 1, "b": 2})
    assert h1 != h3, "a different operand must change the hash"


# ==========================================================================
# C3: `record_from_receipt` -- a convenience constructor over `record`, not
# a parallel append path, so it must obey the exact same immutability rule.
# ==========================================================================
from dbench.ledger import record_from_receipt


def _receipt_row(gene_symbols=("HIF1A", "VEGFA")):
    return {
        "candidate_hash": "cand-xyz",
        "gene_symbols": list(gene_symbols),
        "feature_form": "product",
        "fold": 2,
        "fold_gains": [0.09, 0.07, 0.11],
        "nulls": {"spatial_block": {"p": 0.02, "n": 199},
                 "within_patient_perm": {"p": 0.04, "n": 199}},
        "q_value": 0.08,
        "baseline_width": 45,
        "representation": {"checkpoint": "ssl_ep120.pt", "arm": "morpheus_factorized"},
        "proposal_family": "residual_forward_search",
        "search_budget": 50,
        "episode_identity": {"dataset_version": "renal-spatial-v02-20260925"},
        "context": {"target_family": "spatial_organisation"},
    }


def test_record_from_receipt_derives_operator_and_operands(tmp_path):
    led = CandidateLedger(str(tmp_path / "c1.jsonl"))
    rec = record_from_receipt(led, "cand-xyz", "proposed", _receipt_row())
    assert rec["object"]["operator"] == "flexible_product"
    assert rec["object"]["operands"] == ["HIF1A", "VEGFA"]


def test_record_from_receipt_evidence_carries_named_fields(tmp_path):
    led = CandidateLedger(str(tmp_path / "c2.jsonl"))
    rec = record_from_receipt(led, "cand-xyz", "internal", _receipt_row())
    ev = rec["evidence"]
    assert ev["fold_gains"] == [0.09, 0.07, 0.11]
    assert ev["null_p"] == {"spatial_block": 0.02, "within_patient_perm": 0.04}
    assert ev["q"] == 0.08
    assert ev["baseline_width"] == 45
    assert ev["representation_checkpoint"] == "ssl_ep120.pt"
    assert ev["representation_arm"] == "morpheus_factorized"
    assert ev["proposal_mechanism"] == "residual_forward_search"
    assert ev["search_budget"] == 50
    assert ev["dataset_version"] == "renal-spatial-v02-20260925"


def test_record_from_receipt_explicit_evidence_overrides_derived_fields(tmp_path):
    led = CandidateLedger(str(tmp_path / "c3.jsonl"))
    rec = record_from_receipt(led, "cand-xyz", "internal", _receipt_row(),
                              evidence={"q": 0.001})
    assert rec["evidence"]["q"] == 0.001
    assert rec["evidence"]["fold_gains"] == [0.09, 0.07, 0.11]


def test_record_from_receipt_hash_lock_still_trips_after_external(tmp_path):
    p = str(tmp_path / "c4.jsonl")
    led = CandidateLedger(p)
    record_from_receipt(led, "cand-xyz", "proposed", _receipt_row())
    record_from_receipt(led, "cand-xyz", "internal", _receipt_row())
    record_from_receipt(led, "cand-xyz", "external", _receipt_row(),
                        data_ref="cptac_ccrcc_v1")
    # same object, later stage: fine
    record_from_receipt(led, "cand-xyz", "orthogonal", _receipt_row())
    # redefined object (different gene symbols) after external: must refuse
    redefined = _receipt_row(gene_symbols=("HIF1A", "A_DIFFERENT_GENE"))
    with pytest.raises(LedgerError):
        record_from_receipt(led, "cand-xyz", "orthogonal", redefined)
    # the refusal must not have appended a corrupt/partial line
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    for line in lines:
        json.loads(line)


# ==========================================================================
# E3: `complete_evidence` -- fills every discoadd ledger field explicitly,
# rejects unknown keys unless namespaced `x_`, and never touches the
# append-only `record`/`record_from_receipt` path itself.
# ==========================================================================

def test_complete_evidence_fills_every_field_with_none_when_absent():
    out = complete_evidence({"evidence_tier": "association"})
    assert set(out) == set(LEDGER_EVIDENCE_FIELDS)
    assert out["evidence_tier"] == "association"
    for field in LEDGER_EVIDENCE_FIELDS:
        if field != "evidence_tier":
            assert out[field] is None


def test_complete_evidence_with_no_argument_is_all_none():
    out = complete_evidence()
    assert set(out) == set(LEDGER_EVIDENCE_FIELDS)
    assert all(v is None for v in out.values())


def test_complete_evidence_rejects_unknown_key():
    with pytest.raises(ValueError):
        complete_evidence({"foo": 1})


def test_complete_evidence_allows_x_prefixed_extension_key():
    out = complete_evidence({"x_internal_note": "seat-e"})
    assert out["x_internal_note"] == "seat-e"
    assert set(out) == set(LEDGER_EVIDENCE_FIELDS) | {"x_internal_note"}


def test_complete_evidence_output_is_a_valid_record_from_receipt_evidence_override(tmp_path):
    led = CandidateLedger(str(tmp_path / "c5.jsonl"))
    row = {"candidate_hash": "cand-e3", "gene_symbols": ["A", "B"], "feature_form": "product"}
    evidence = complete_evidence({"evidence_tier": "invariant_association",
                                  "characterization_ref": "card://cand-e3"})
    rec = record_from_receipt(led, "cand-e3", "internal", row, evidence=evidence)
    assert rec["evidence"]["evidence_tier"] == "invariant_association"
    assert rec["evidence"]["characterization_ref"] == "card://cand-e3"
    assert rec["evidence"]["human_involvement"] is None
