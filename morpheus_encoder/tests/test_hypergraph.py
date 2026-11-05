"""The typed Scientific Object Hypergraph: round-trip, hashing, validation.

The registry is append-only and is the thing a reviewer will use to reconstruct
what MORPHEUS proposed *before reality answered*. So the serialisation has to be
canonical (same graph, same bytes, same hash, on any machine and after any
round-trip) and the validator has to refuse structurally impossible graphs
rather than logging them.
"""
import json

import pytest

from morpheus_encoder.hypergraph import (Condition, Hyperedge, ObjectNode,
                                         Relation, SCHEMA_VERSION,
                                         SchemaError, SciHypergraph,
                                         check_vocabulary_agreement)


def _graph():
    nodes = [
        ObjectNode(id="obj000", kind="program", slot=0, expert="factor",
                   modality_gates={"rna": 1.0},
                   loading={"mean": 0.31234567890123456, "fire_rate": 0.25},
                   score=0.8),
        ObjectNode(id="obj001", kind="chart", slot=1, expert="manifold",
                   modality_gates={"rna": 0.4, "he": 0.6},
                   params={"regime_occupancy": [0.7, 0.3]}, score=0.6),
        ObjectNode(id="obj002", kind="state", slot=2, expert="relational",
                   modality_gates={"he": 1.0}, score=0.5),
    ]
    conds = [Condition(id="g0", kind="environment", key="donor", value="D3"),
             Condition(id="g1", kind="treatment", key="drug", value=1)]
    rels = [Relation(id="rel000", type="SPACE", source="obj000",
                     target="obj001", directed=True, lag=2, weight=0.42,
                     conditions=("g0",))]
    hes = [Hyperedge(id="he000", type="AND", members=("obj000", "obj002"),
                     score=0.9, effect=0.17, conditions=("g0", "g1"),
                     parents=(("obj000",), ("obj002",))),
           Hyperedge(id="he001", type="HIGHER_ORDER",
                     members=("obj000", "obj001", "obj002"), score=0.55)]
    return SciHypergraph(nodes=nodes, relations=rels, hyperedges=hes,
                         conditions=conds,
                         operating_point={"alpha": 0.05, "delta": 0.02,
                                          "B": 8.0, "kappa": 0.5},
                         meta={"world": "tier0_and"}).validate()


def test_round_trip_is_exact():
    g = _graph()
    back = SciHypergraph.from_json(g.to_json())
    assert back.content_hash() == g.content_hash()
    assert json.loads(back.canonical_json()) == json.loads(g.canonical_json())
    assert len(back.V) == 3 and len(back.E) == 1 and len(back.H) == 2
    assert len(back.Gamma) == 2
    assert back.node("obj001").modality_gates == {"rna": 0.4, "he": 0.6}
    assert back.H[0].members == ("obj000", "obj002")
    assert back.E[0].lag == 2 and back.E[0].directed


def test_round_trip_survives_a_second_pass():
    g = _graph()
    once = SciHypergraph.from_json(g.to_json())
    twice = SciHypergraph.from_json(once.to_json(indent=None))
    assert twice.content_hash() == g.content_hash()


def test_canonical_json_is_key_order_and_float_stable():
    g = _graph()
    h1 = g.content_hash()
    # reorder the dicts a producer might build in a different order
    g.nodes[0].loading = {"fire_rate": 0.25, "mean": 0.31234567890123456}
    assert g.content_hash() == h1
    # a change below the rounding floor must not move the hash
    g.nodes[0].loading["mean"] = 0.3123456789012345678
    assert g.content_hash() == h1
    # a real change must
    g.nodes[0].loading["mean"] = 0.31234567891
    assert g.content_hash() != h1


def test_schema_version_is_stamped_and_checked():
    g = _graph()
    assert g.to_dict()["schema_version"] == SCHEMA_VERSION
    d = g.to_dict()
    d["schema_version"] = "morpheus.sci_hypergraph/9.0.0"
    with pytest.raises(SchemaError):
        SciHypergraph.from_dict(d)
    d["schema_version"] = "something.else/1.0.0"
    with pytest.raises(SchemaError):
        SciHypergraph.from_dict(d)


def test_validation_catches_dangling_references():
    g = _graph()
    g.hyperedges.append(Hyperedge(id="heX", type="AND",
                                  members=("obj000", "ghost")))
    with pytest.raises(SchemaError) as e:
        g.validate()
    assert "ghost" in str(e.value)


def test_validation_catches_dangling_conditions():
    g = _graph()
    g.nodes[0].conditions = ("g0", "g9")
    with pytest.raises(SchemaError) as e:
        g.validate()
    assert "g9" in str(e.value)


def test_validation_catches_duplicate_ids():
    g = _graph()
    g.nodes.append(ObjectNode(id="obj000", kind="state"))
    with pytest.raises(SchemaError) as e:
        g.validate()
    assert "duplicate" in str(e.value)


def test_validation_catches_order_one_hyperedges():
    g = _graph()
    g.hyperedges.append(Hyperedge(id="heY", type="AND", members=("obj000",)))
    with pytest.raises(SchemaError) as e:
        g.validate()
    assert "order 1" in str(e.value)


def test_validation_catches_a_lag_without_a_direction():
    g = _graph()
    g.relations.append(Relation(id="relB", type="LAG", source="obj000",
                                target="obj001", directed=False, lag=1))
    with pytest.raises(SchemaError) as e:
        g.validate()
    assert "lag" in str(e.value)


def test_validation_catches_unknown_types():
    for mutate, needle in (
            (lambda g: g.nodes.append(ObjectNode(id="z", kind="vibe")), "kind"),
            (lambda g: g.hyperedges.append(
                Hyperedge(id="z", type="VIBE", members=("obj000", "obj001"))), "type"),
            (lambda g: g.relations.append(
                Relation(id="z", type="VIBE", source="obj000", target="obj001")), "type"),
            (lambda g: g.conditions.append(
                Condition(id="z", kind="vibe", key="k")), "kind")):
        g = _graph()
        mutate(g)
        with pytest.raises(SchemaError) as e:
            g.validate()
        assert needle in str(e.value)


def test_hyperedge_key_is_member_order_insensitive():
    a = Hyperedge(id="a", type="AND", members=("x", "y", "z"))
    b = Hyperedge(id="b", type="AND", members=("z", "x", "y"))
    assert a.key() == b.key()


def test_to_hypotheses_names_operands_and_uses_the_shared_vocabulary():
    from morpheus_encoder.hypergraph import OPERATORS
    hs = _graph().to_hypotheses(target="Y")
    assert len(hs) == 3 + 1 + 2
    for h in hs:
        assert h["operator"] in OPERATORS
        assert h["operands"], "a hypothesis with no operands has recovered nothing"
        assert h["target"] == "Y"
    conj = [h for h in hs if h["id"] == "he000"][0]
    assert conj["operator"] == "AND"
    assert set(conj["operands"]) == {"obj000", "obj002"}
    assert conj["context"] == {"donor": "D3", "drug": 1}
    assert conj["parents"] == [["obj000"], ["obj002"]]
    lagged = [h for h in hs if h["id"] == "rel000"][0]
    assert lagged["operator"] == "SPACE" and lagged["lag"] == 2


def test_registry_record_separates_object_from_provenance():
    g = _graph()
    r1 = g.to_registry_record(checkpoint="abc", rank=1, external="HEST-held-out")
    r2 = g.to_registry_record(checkpoint="def", rank=7, external="other")
    assert r1["content_hash"] == r2["content_hash"], (
        "the same object under different provenance must hash identically, or "
        "the registry cannot tell a re-proposal from a new object")
    assert r1["provenance"]["rank"] == 1
    assert r1["schema_version"] == SCHEMA_VERSION
    json.dumps(r1)          # must be plain-JSON serialisable


def test_vocabulary_agrees_with_discoverybench_when_it_exists():
    res = check_vocabulary_agreement()
    if not res["checked"]:
        pytest.skip("dbench not importable yet: %s" % res["reason"])
    assert res["agree"], (
        "operator vocabularies have drifted -- only here: %s, only in dbench: %s"
        % (res["only_here"], res["only_in_dbench"]))


def test_summary_reports_orders_and_hash():
    s = _graph().summary()
    assert s["n_nodes"] == 3 and s["n_hyperedges"] == 2
    assert s["hyperedge_orders"] == {"2": 1, "3": 1}
    assert len(s["content_hash"]) == 64


def test_registry_candidates_carry_the_object_schema_version():
    """The handoff to `morpheus_registry`, which stores `h` as loose JSON.

    The registry deliberately does not type-check `h` -- the Hypothesis
    Compiler's grammar is being defined in parallel -- so `h_schema_version` is
    the only thing telling a later reader which object grammar a stored
    candidate was written under. If it does not travel, the loose JSON is
    undecodable after the grammar moves.
    """
    from morpheus_encoder.hypergraph import SCHEMA_VERSION

    g = _graph()
    cands = g.to_registry_candidates(
        discovery_dataset="d", split="train", predicted_direction="+",
        predicted_effect=0.1, statistical_family="t",
        multiplicity_correction="BH", freeze_id="F1")
    assert cands, "a non-empty graph produced no registry candidates"
    assert len(cands) == len(g.to_hypotheses())
    ids = [c["candidate_id"] for c in cands]
    assert len(set(ids)) == len(ids), "candidate ids collide"
    for c in cands:
        assert c["h_schema_version"] == SCHEMA_VERSION
        assert isinstance(c["h"], dict) and c["h"]["operator"]
        assert c["freeze_id"] == "F1"
    ranks = sorted(c["rank_before_validation"] for c in cands)
    assert ranks == list(range(1, len(cands) + 1))


def test_registry_candidates_are_accepted_by_the_real_registry(tmp_path):
    """Interface agreement checked against the package, not against a mock."""
    reg = pytest.importorskip("morpheus_registry.discovery_registry")
    r = reg.DiscoveryRegistry(tmp_path / "reg")
    fid = r.append_freeze(
        checkpoint_hash="abc", adapter="morpheus_encoder", compiler_version="0",
        grammar_version="0", null_generators=["perm"], statistical_tests=["t"],
        multiplicity_family="BH", external_datasets=["ext"], ranking_rule="score",
        translational_endpoints=["os"])
    g = _graph()
    cands = g.to_registry_candidates(
        freeze_id=fid, discovery_dataset="d", split="train",
        predicted_direction="+", predicted_effect=0.1,
        statistical_family="t", multiplicity_correction="BH")
    written = [r.append_candidate(**c) for c in cands]
    assert len(written) == len(cands)
    assert bool(r.verify()), "registry tamper-verification failed after writing"
