"""Test A for the novelty checker: a known object (BAP1-loss, from
dbench/TIER3_TARGETS.md) returns its reference; a synthetic random gene set
returns none.
"""
from dbench import novelty_check as NC


def test_known_object_returns_its_reference():
    res = NC.novelty_check("biology", "CAUSE",
                           ("BAP1_mutation_status", "immune_program_score"),
                           context={"cohort": "ccrcc"})
    assert res["novel"] is False
    assert res["n_hits"] >= 1
    refs = [h.get("reference", "") for h in res["hits"]]
    assert any("Clark" in r and "2019" in r for r in refs), res


def test_synthetic_random_gene_set_returns_none():
    res = NC.novelty_check("biology", "ADD",
                           ("ZZZFAKEGENE1", "ZZZFAKEGENE2", "ZZZFAKEGENE3"))
    assert res["novel"] is True
    assert res["n_hits"] == 0
    assert res["hits"] == []


def test_climate_known_teleconnection_hits():
    res = NC.novelty_check("climate", "SPACE", ("oni", "precip_grid"),
                           context={"season": "DJF"})
    assert res["novel"] is False
    sources = {h["source"] for h in res["hits"]}
    assert "teleconnection_catalogue" in sources or \
        "known_objects_registry" in sources


def test_materials_known_scaling_relation_hits():
    res = NC.novelty_check("materials", "LINCOMB",
                           ("E_ads_OH", "E_ads_OOH"),
                           context={"relation": "adsorption scaling"})
    assert res["novel"] is False
    assert any("Abild-Pedersen" in h.get("reference", "")
              for h in res["hits"])


def test_physics_lhco_membership_hits():
    res = NC.novelty_check("physics", "ANOMALY",
                           ("dijet_invariant_mass", "event_is_signal"),
                           context={"dataset": "lhco2020 blackbox1"})
    assert res["novel"] is False
    assert any("Kasieczka" in h.get("reference", "") for h in res["hits"])


# ==========================================================================
# D5: the literature-novelty client (`canonical_claim_record`,
# `search_novelty_literature`, `compositional_labels`, the ledger guard).
# Every test below either never touches the network (`offline=True`,
# the default) or mocks `urllib.request.urlopen` directly -- none of these
# tests may perform a live HTTP request.
# ==========================================================================
import json as _json
import urllib.parse as _urlparse

import pytest

from dbench.ledger import CandidateLedger


def _fake_response(payload):
    class _Resp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return _json.dumps(self._data).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Resp(payload)


def _urlopen_matching(required_terms, *, hit_count=3):
    """A fake `urllib.request.urlopen` returning `hit_count` results for any
    query containing every term in `required_terms` (case-insensitive) and
    zero results otherwise, shaped like a real OpenAlex/Europe PMC response."""
    required = [t.lower() for t in required_terms]

    def fake(url, timeout=15):
        parsed = _urlparse.urlparse(url)
        qs = _urlparse.parse_qs(parsed.query)
        text = (qs.get("search") or qs.get("query") or [""])[0].lower()
        hit = bool(required) and all(term in text for term in required)
        n = hit_count if hit else 0
        if "api.openalex.org" in url:
            payload = {"meta": {"count": n},
                      "results": [{"id": "https://openalex.org/W1",
                                  "display_name": "fixture work"}] * min(n, 5)}
        else:
            payload = {"hitCount": n,
                      "resultList": {"result": [{"id": "PMC1", "title": "fixture"}]
                                    * min(n, 5)}}
        return _fake_response(payload)
    return fake


def test_canonical_claim_record_hash_is_deterministic_and_order_insensitive():
    r1 = NC.canonical_claim_record(("B", "A"), "ADD", "positive",
                                   context={"y": 2, "x": 1})
    r2 = NC.canonical_claim_record(("A", "B"), "ADD", "positive",
                                   context={"x": 1, "y": 2})
    assert r1["claim_hash"] == r2["claim_hash"]
    r3 = NC.canonical_claim_record(("A", "C"), "ADD", "positive",
                                   context={"x": 1, "y": 2})
    assert r3["claim_hash"] != r1["claim_hash"]


def test_hash_computed_before_first_network_call(monkeypatch):
    calls = []
    orig = NC.canonical_claim_record

    def spy(*a, **kw):
        rec = orig(*a, **kw)
        calls.append("hash")
        return rec
    monkeypatch.setattr(NC, "canonical_claim_record", spy)

    def fake_urlopen(url, timeout=15):
        calls.append("network")
        return _fake_response({"meta": {"count": 0}, "results": []})
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    NC.search_novelty_literature(("ZZZFAKE1", "ZZZFAKE2"), "flexible_product",
                                 "positive", offline=False)
    assert "hash" in calls and "network" in calls
    assert calls.index("hash") < calls.index("network")


def test_offline_default_never_calls_network(monkeypatch):
    def boom(url, timeout=15):
        raise AssertionError("offline=True must never call urlopen")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    res = NC.search_novelty_literature(("ZZZFAKE1",), "flexible_subspace",
                                       "positive")
    assert res["label"] == "unresolved"


def test_query_set_has_the_required_kinds():
    rec = NC.canonical_claim_record(("A", "B"), "flexible_product", "positive",
                                    context={"cohort": "ccrcc"})
    qs = NC.build_query_set(rec, other_context="tcga_kirc", other_scale="patient")
    kinds = {q["kind"] for q in qs}
    assert kinds >= {"exact_full_relation", "pairwise_constituents",
                    "lower_order_subset", "other_context", "other_scale"}


def test_known_fixture_g18_ifn_gamma_mhc_inflamed_ccrcc(monkeypatch):
    """G18 (FINDINGS.md): IFN-gamma/MHC-high inflamed compartment against
    retained tubular metabolism in ccRCC, prognostic in TCGA-KIRC -- an
    established literature object, must resolve to `known`."""
    monkeypatch.setattr("urllib.request.urlopen",
                        _urlopen_matching(["ifng_program", "mhc_program"]))
    res = NC.search_novelty_literature(
        ("IFNG_program", "MHC_program"), "inflamed_vs_tubular_metabolism",
        "positive", context={"cohort": "ccrcc"}, spatial_scale="spot",
        environment="TCGA-KIRC", outcome="prognosis", offline=False)
    assert res["label"] == "known"
    assert res["claim_hash"]


def test_known_fixture_hif1a_hypoxia_ccrcc(monkeypatch):
    """HIF1A-driven hypoxia response in ccRCC -- established literature
    (VHL-HIF axis), must resolve to `known`."""
    monkeypatch.setattr("urllib.request.urlopen",
                        _urlopen_matching(["hif1a", "hypoxia_program"]))
    res = NC.search_novelty_literature(
        ("HIF1A", "hypoxia_program"), "hypoxia_driven_metabolic_state",
        "positive", context={"cohort": "ccrcc"}, offline=False)
    assert res["label"] == "known"


def test_fabricated_triple_zero_hits_is_unresolved(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        _urlopen_matching(["this_never_matches_anything"]))
    res = NC.search_novelty_literature(
        ("ZZZFAKEOPERAND1", "ZZZFAKEOPERAND2", "ZZZFAKEOPERAND3"),
        "ZZZFAKERELATION", "positive", offline=False)
    assert res["label"] == "unresolved"
    assert res["label"] != "strong_novelty_candidate"


def test_absence_of_hits_alone_never_yields_strong_novelty_candidate():
    assert NC.compositional_labels([]) == "unresolved"
    hits = [{"kind": "exact_full_relation", "n_results": 0},
           {"kind": "lower_order_subset", "n_results": 0}]
    assert NC.compositional_labels(hits) == "unresolved"


def test_strong_novelty_candidate_requires_positive_lower_order_evidence():
    hits = [{"kind": "exact_full_relation", "n_results": 0},
           {"kind": "pairwise_constituents", "n_results": 0},
           {"kind": "lower_order_subset", "n_results": 4}]
    assert NC.compositional_labels(hits) == "strong_novelty_candidate"


def test_compositional_novelty_when_only_parts_are_known():
    hits = [{"kind": "exact_full_relation", "n_results": 0},
           {"kind": "pairwise_constituents", "n_results": 2},
           {"kind": "lower_order_subset", "n_results": 5}]
    assert NC.compositional_labels(hits) == "compositional_novelty"


def test_require_stage_guard_raises_before_internal(tmp_path, monkeypatch):
    def boom(url, timeout=15):
        raise AssertionError("guard must refuse before any network call")
    monkeypatch.setattr("urllib.request.urlopen", boom)

    led = CandidateLedger(str(tmp_path / "campaign.jsonl"))
    led.record("C1", "proposed", "ADD", ("g1",))
    with pytest.raises(NC.NoveltyGuardError):
        NC.search_novelty_literature(
            ("g1",), "ADD", "positive", offline=False, ledger=led,
            candidate_id="C1", require_stage="internal")


def test_require_stage_guard_passes_at_internal_or_above(tmp_path):
    led = CandidateLedger(str(tmp_path / "campaign2.jsonl"))
    led.record("C1", "proposed", "ADD", ("g1",))
    led.record("C1", "internal", "ADD", ("g1",))
    res = NC.search_novelty_literature(
        ("g1",), "ADD", "positive", ledger=led, candidate_id="C1",
        require_stage="internal")
    assert res["label"] in NC.NOVELTY_LABELS


# ------------------------------------------------- literature cutoff (v02)
def test_before_date_reaches_both_clients_and_the_cache_key(monkeypatch, tmp_path):
    urls = []

    def fake(url, timeout=15):
        urls.append(url)
        if "api.openalex.org" in url:
            return _fake_response({"meta": {"count": 1}, "results": [{"id": "W1", "display_name": "old"}]})
        return _fake_response({"hitCount": 1, "resultList": {"result": [{"id": "P1", "title": "old"}]}})

    monkeypatch.setattr("urllib.request.urlopen", fake)
    oa = NC._openalex("TSK fibroblast", cache_dir=str(tmp_path), offline=False, before_date="2020-06-22")
    ep = NC._europepmc("TSK fibroblast", cache_dir=str(tmp_path), offline=False, before_date="2020-06-22")
    assert "filter=to_publication_date:2020-06-22" in urls[0]
    q = _urlparse.parse_qs(_urlparse.urlparse(urls[1]).query)["query"][0]
    assert q == "TSK fibroblast AND FIRST_PDATE:[1800-01-01 TO 2020-06-22]"
    assert oa["before_date"] == ep["before_date"] == "2020-06-22"
    k_cut = NC._cache_key("openalex", "TSK fibroblast", "2020-06-22")
    k_all = NC._cache_key("openalex", "TSK fibroblast")
    assert k_cut != k_all and k_all == NC._cache_key("openalex", "TSK fibroblast", None)
    assert (tmp_path / k_cut).exists() and not (tmp_path / k_all).exists()
    # the cached cut-off answer is served without a call; the unlimited query calls again
    NC._openalex("TSK fibroblast", cache_dir=str(tmp_path), offline=False, before_date="2020-06-22")
    assert len(urls) == 2
    NC._openalex("TSK fibroblast", cache_dir=str(tmp_path), offline=False)
    assert len(urls) == 3 and "filter=" not in urls[2]


def test_before_date_threads_through_search(monkeypatch):
    urls = []

    def fake(url, timeout=15):
        urls.append(url)
        if "api.openalex.org" in url:
            return _fake_response({"meta": {"count": 0}, "results": []})
        return _fake_response({"hitCount": 0, "resultList": {"result": []}})

    monkeypatch.setattr("urllib.request.urlopen", fake)
    r = NC.search_novelty_literature(("TSK", "FAP"), "flexible", "positive", offline=False,
                                     before_date="2020-06-22")
    assert r["before_date"] == "2020-06-22" and urls
    assert all("2020-06-22" in _urlparse.unquote(u) for u in urls)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        NC.search_novelty_literature(("TSK",), "x", "positive", before_date="2020-06-22&x=1")
