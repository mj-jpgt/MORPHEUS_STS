"""Historical rediscovery scorer (`dbench/historical.py`) on a planted known answer.

The fixture is a `StudyEpisode` whose target gene set is the 6-neighbour mean
of a smooth latent field and whose answer proxy genes measure that field in
the spot itself: the planted hypothesis is `nbr6` of an answer-proxy gene."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from dbench import historical as H

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTOCOL = os.path.join(REPO, "dbench", "protocols", "historical_rediscovery_v01.json")
HASH_FILE = os.path.join(REPO, "runs", "development", "sprint_v06", "historical_protocol_sha256.txt")


def planted_task(direction: int = 1, transforms=("nbr6", "nbr18")) -> dict:
    return {
        "task_id": "SYN", "study_key": "synthetic", "untouched": True, "protocol_id": "synthetic",
        "protocol_sha256": "0" * 64,
        "publication": {"doi": "10.0000/synthetic", "first_published": "2020-06-23", "before_date": "2020-01-01"},
        "finding_text": ["planted"],
        "relation": {"target": "tgt", "answer_variables": ["niche"], "transform": list(transforms),
                     "direction": direction, "spatial_relation": "target = nbr6(niche)"},
        "given_to_morpheus": {"target_definition": "tgt", "rows": {"rule": "all"},
                              "variable_pools": ["genes", "programmes"],
                              "transforms": ["own", "nbr6", "nbr18", "contrast"], "excluded": []},
        "proxy_sets": {"tgt": {"role": "target", "genes": ["T1", "T2", "T3"]},
                       "niche": {"role": "answer", "genes": ["P1", "P2", "P3", "P4", "ABSENT1"]}},
        "scoring": {"top_k": 20, "budget": 64, "r_threshold": 0.9, "answer_proxies": ["niche"],
                    "answer_transforms": list(transforms)},
        "modality": "synthetic",
    }


def planted_episode(seed: int = 0, n_patients: int = 8, side: int = 15):
    from morpheus.tier2.substrates.hest_multi import StudyEpisode, knn_neighbors_kdtree

    rng = np.random.default_rng(seed)
    rows = side * side
    n = n_patients * rows
    grid = np.array([((i % side) + 0.5 * ((i // side) % 2), (i // side) * np.sqrt(3) / 2)
                     for i in range(rows)]) * 100.0
    coords = np.tile(grid, (n_patients, 1))
    groups = np.repeat(np.arange(n_patients), rows)
    nb6 = knn_neighbors_kdtree(coords, 6, groups)
    nb18 = knn_neighbors_kdtree(coords, 18, groups)
    z = rng.standard_normal(n)
    for _ in range(4):  # smooth latent field within sample
        z = 0.5 * z + 0.5 * H.neighbour_mean(z, nb6)
    z = (z - z.mean()) / z.std()
    genes, cols = [], []
    for j in range(1, 5):
        genes.append(f"P{j}"); cols.append(z + 0.15 * rng.standard_normal(n))
    lag = H.neighbour_mean(z, nb6)
    for j in range(1, 4):
        genes.append(f"T{j}"); cols.append(lag + 0.4 * rng.standard_normal(n))
    for j in range(30):
        genes.append(f"N{j}"); cols.append(rng.standard_normal(n))
    X = np.column_stack(cols)
    X = X + rng.standard_normal((n_patients, X.shape[1]))[groups]  # patient offsets
    Xc = X - np.vstack([X[groups == g].mean(0) for g in range(n_patients)])[groups]
    prog = np.column_stack([Xc[:, 7 + 3 * k: 10 + 3 * k].mean(1) for k in range(5)])
    half = n_patients // 2
    return StudyEpisode(
        study="synthetic_planted", target="gene:T1", target_family="gene", y=None, target_concept=None,
        patient=np.repeat([f"P{p}" for p in range(n_patients)], rows),
        sample_id=np.repeat([f"S{p}" for p in range(n_patients)], rows),
        barcode=np.array([f"bc{i}" for i in range(rows)] * n_patients), coords=coords.astype(np.float32),
        platform="visium", neighbours={6: nb6, 18: nb18},
        views={"genes": Xc.astype(np.float32), "programmes": prog.astype(np.float32), "hoptimus": None,
               "depth": (8 + 0.1 * rng.standard_normal((n, 1))).astype(np.float32)},
        columns={"genes": genes, "programmes": [f"mp:noise{k}" for k in range(5)]},
        covered=np.ones(n, dtype=bool), support_rows=np.flatnonzero(groups < half),
        query_rows=np.flatnonzero(groups >= half), exclusions={}, meta={"platform_of_row": ["visium"] * n})


@pytest.fixture(scope="module")
def ep():
    return planted_episode(0)


def _with_r(p: np.ndarray, groups: np.ndarray, r: float, seed: int = 1) -> np.ndarray:
    """A vector with within-patient correlation exactly r to p."""
    pc = H._centre(p.reshape(-1, 1), groups).ravel()
    e = H._centre(np.random.default_rng(seed).standard_normal(len(p)).reshape(-1, 1), groups).ravel()
    e -= pc * (e @ pc) / (pc @ pc)
    pc, e = pc / np.linalg.norm(pc), e / np.linalg.norm(e)
    return r * pc + math.sqrt(1 - r * r) * e


def test_proxy_scores_mean_of_present_genes(ep):
    task = planted_task()
    s = H.proxy_scores(ep, task)
    assert s["_present"]["niche"] == ["P1", "P2", "P3", "P4"]  # ABSENT1 ignored
    X = np.asarray(ep.views["genes"], dtype=float)
    assert np.allclose(s["niche"], X[:, :4].mean(1))
    g = H._groups(ep)
    for key in ("niche@nbr6", "niche@nbr18", "niche@contrast"):
        for p in np.unique(g):
            assert abs(s[key][g == p].mean()) < 1e-9  # within-patient centred
    assert np.allclose(s["niche@contrast"], H._centre((s["niche"] - H.neighbour_mean(s["niche"], ep.neighbours[6]))
                                                      .reshape(-1, 1), g).ravel())


def test_planted_set_hits_at_rank_1_and_noise_does_not(ep):
    task = planted_task()
    ranked = [["nbr6:genes:P1", "genes:N3"], ["genes:N1"], ["nbr6:genes:N5"], ["contrast:genes:N7", "genes:N2"]]
    out = H.score_ranked_list(ranked, ep, task)
    assert out["rank_of_first_hit"] == 1
    assert out["hit_variable"] == "nbr6:genes:P1" and out["hit_proxy"] == "niche"
    assert out["r"] >= 0.9 and out["r_target"] > 0
    noise = H.score_ranked_list(ranked[1:], ep, task)
    assert math.isinf(noise["rank_of_first_hit"]) and noise["hit_variable"] is None
    assert noise["n_evaluated"] == 3


def test_planted_set_found_later_in_the_list(ep):
    ranked = [["genes:N1"]] * 4 + [["nbr18:genes:P2"]] + [["genes:N2"]]
    out = H.score_ranked_list(ranked, ep, planted_task())
    assert out["rank_of_first_hit"] == 5 and out["hit_transform"] == "nbr18"


def test_r_threshold_095_counts_080_does_not(ep):
    task = planted_task()
    s = H.proxy_scores(ep, task)
    g = H._groups(ep)
    rows = np.arange(ep.n)
    for r, expect in ((0.95, 1), (0.8, math.inf)):
        v = _with_r(np.asarray(s["niche@nbr6"]), g, r)
        assert abs(H.within_patient_r(v, s["niche@nbr6"], g) - r) < 1e-9
        out = H.score_ranked_list([{"names": ["nbr6:custom"], "values": v.reshape(-1, 1)}], ep, task,
                                  scores=s, rows=rows)
        assert out["rank_of_first_hit"] == expect, (r, out["r"])


def test_transform_and_sign_rules(ep):
    # own-spot proxy gene is not under a stated transform -> no hit
    assert math.isinf(H.score_ranked_list([["genes:P1"]], ep, planted_task())["rank_of_first_hit"])
    # same variable, stated direction negative -> no hit
    assert math.isinf(H.score_ranked_list([["nbr6:genes:P1"]], ep, planted_task(direction=-1))["rank_of_first_hit"])
    # own transform allowed -> own proxy gene hits
    assert H.score_ranked_list([["genes:P1"]], ep, planted_task(transforms=("own",)))["rank_of_first_hit"] == 1


def test_parse_variable_grammar():
    assert H.parse_variable("genes:P1") == ("own", "genes:P1")
    assert H.parse_variable("own:nbr6:genes:P1") == ("nbr6", "genes:P1")
    assert H.parse_variable("nbr6:nbr18:programmes:mp:x") == ("nbr6.nbr18", "programmes:mp:x")


def test_design_excludes_target_genes_and_is_support_only_standardised(ep):
    task = planted_task()
    s = H.proxy_scores(ep, task)
    rows = H.task_rows(ep, task, s)
    y = H.target_values(ep, task, s, rows)
    d = H.build_design(ep, task, rows, y)
    assert not any(n.split(":")[-1] in ("T1", "T2", "T3") for n in d["names"])
    assert sorted(d["excluded"]["target_genes"]) == ["genes:T1", "genes:T2", "genes:T3"]
    assert "nbr6:genes:P1" in d["names"] and "nbr18:programmes:mp:noise0" in d["names"]
    sup = np.isin(rows, ep.support_rows)
    assert np.allclose(np.asarray(d["X"][sup], dtype=float).mean(0), 0, atol=1e-4)
    assert set(d["roles"].values()) == {"variable", "context"}


def test_row_rule_quantile(ep):
    task = planted_task()
    task["given_to_morpheus"]["rows"] = {"rule": "own_score_at_or_below_within_sample_quantile",
                                         "proxy": "niche", "quantile": 0.9}
    s = H.proxy_scores(ep, task)
    rows = H.task_rows(ep, task, s)
    frac = len(rows) / ep.n
    assert 0.88 <= frac <= 0.92
    out = H.score_ranked_list([["nbr6:genes:P1"]], ep, task)
    assert out["rank_of_first_hit"] == 1


def test_run_task_learned_stub_carries_before_date_and_hits(ep):
    pytest.importorskip("morpheus_encoder.learned_search.propose")
    res = H.run_task(planted_task(), ep, "learned_search", budget=32, model=None, n_boot=20)
    assert res["before_date"] == "2020-01-01"
    assert res["protocol_sha256"] == "0" * 64
    assert res["available"] and res["report"]["stub"] is True
    assert res["n_relationships_tested"] >= 1
    assert res["rank_of_first_hit"] != "inf", res["top"][:3]
    h = res["heldout_hit"]
    assert h is not None and "error" not in h, h
    assert h["relation"] in ("v07_ho", "v06")
    json.dumps(res)  # serialisable


class _FixedMethod:
    """A stand-in baseline: returns a fixed ranked list (spatial_lag of P2 at rank 2)."""

    def __init__(self, **kw):
        self.kw = kw

    def run(self, inp, budget):
        from dbench.benchmark.methods import Hypothesis, MethodOutput
        assert inp.X.shape[0] == len(inp.y) == len(inp.groups) == len(inp.neighbours)
        hyps = [Hypothesis(("genes:N4",), "single", 2.0), Hypothesis(("genes:P2",), "spatial_lag", 1.0)]
        return MethodOutput(hyps, {"n_relationships_tested": 7, **self.kw}, 0.0)


def test_run_task_baseline_path(ep):
    res = H.run_task(planted_task(), ep, "fixed", budget=16, methods={"fixed": _FixedMethod},
                     method_kwargs={"n_sets": 99}, n_boot=20)
    assert res["before_date"] == "2020-01-01"
    assert res["rank_of_first_hit"] == 2 and res["hit_variable"] == "nbr6:genes:P2"
    assert res["n_relationships_tested"] == 7 and res["info"]["n_sets"] == 99
    assert res["heldout_hit"]["n_replication"] == len(ep.query_rows)
    missing = H.run_task(planted_task(), ep, "nope", methods={})
    assert missing["available"] is False and missing["rank_of_first_hit"] == "inf"
    assert missing["before_date"] == "2020-01-01"


def test_frozen_protocol_loads_and_hash_is_pinned():
    from dbench.prospective_campaign import canonical_hash
    tasks = H.load_tasks(PROTOCOL)
    ids = [t["task_id"] for t in tasks]
    assert ids == ["H1", "H2", "H3", "H3b", "S0"]
    doc = json.load(open(PROTOCOL, encoding="utf-8"))
    assert all(t["protocol_sha256"] == canonical_hash(doc) for t in tasks)
    cut = {t["task_id"]: H.before_date(t) for t in tasks}
    assert cut == {"H1": "2020-06-23", "H2": "2023-01-23", "H3": "2024-03-25", "H3b": "2021-11-12",
                   "S0": "2022-02-28"}
    s0 = [t for t in tasks if t["task_id"] == "S0"][0]
    assert s0["untouched"] is False and "not untouched" in s0["label"]
    for t in tasks:
        assert t["scoring"]["top_k"] == 20 and t["scoring"]["r_threshold"] == 0.9
        assert t["given_to_morpheus"]["transforms"] == ["own", "nbr6", "nbr18", "contrast"]
    h3 = [t for t in tasks if t["task_id"] == "H3"][0]
    assert len(h3["proxy_sets"]["pmn_mdsc"]["genes"]) == 81 and len(h3["proxy_sets"]["club"]["genes"]) == 69
    assert set(h3["shared_genes_removed_from_both_scores"]) == (set(h3["proxy_sets"]["pmn_mdsc"]["genes"])
                                                                 & set(h3["proxy_sets"]["club"]["genes"]))
    if os.path.exists(HASH_FILE):
        assert open(HASH_FILE).read().split()[0] == canonical_hash(doc)


EXT = os.path.join(REPO, "dbench", "protocols", "ccrcc_external_replication_v01.json")
EXT_HASH = os.path.join(REPO, "runs", "development", "sprint_v06", "ccrcc_external_replication_sha256.txt")


def test_external_replication_protocol_frozen_rule():
    from dbench.prospective_campaign import canonical_hash
    doc = json.load(open(EXT, encoding="utf-8"))
    assert doc["schema"] == "morpheus_external_replication/v1"
    keys = [s["source_key"] for s in doc["sources"]]
    assert keys == ["li2022_ccrcc", "gse250163", "zenodo14620362"]
    assert all(s["pending_conversion"] for s in doc["sources"])
    pr = doc["pass_rule"]
    assert pr["all_required"] and "199 draws" in pr["permutation"] and "<= 0.05" in pr["permutation"]
    for k in ("identity", "arithmetic", "fresh_frozen_to_ffpe", "missing_genes", "morphology", "competitor"):
        assert k in doc["transfer_rule"]
    assert "never substituted" in doc["transfer_rule"]["missing_genes"]
    assert "0.5 um" in doc["transfer_rule"]["morphology"]
    if os.path.exists(EXT_HASH):
        assert open(EXT_HASH).read().split()[0] == canonical_hash(doc)


@pytest.mark.parametrize("name", ["renal_spatial_v07.json", "ependymoma_spatial_v01.json"])
def test_blind_drafts_pass_load_protocol(name):
    from dbench.prospective_campaign import _family_targets, load_protocol
    d = load_protocol(os.path.join(REPO, "dbench", "protocols", name))
    disc = d["discovery"]
    from morpheus_encoder.scripts import run_general_discovery_packet as runner
    other = "ependymoma_spatial_v01.json" if name == "renal_spatial_v07.json" else "renal_spatial_v07.json"
    other_rel = load_protocol(os.path.join(REPO, "dbench", "protocols", other))["discovery"]["relation"]
    accepted = set(runner.RELATIONS) | {getattr(runner, "V03_RELATION", "v03_rff")}
    assert d["status"].startswith(("DRAFT", "REGISTERED")) and disc["relation"] in accepted and disc["relation"] == other_rel
    assert disc["fold_seed"] == 20260926 and disc["n_folds"] == 5 and disc["literature_blind"]
    assert disc["proposers_in_all"] == ["learned_search", "classical_pair"]
    assert "no_cv_reranking" in disc["proposers"] and "interleaved" in disc["proposers"]["merge"]
    assert "higher_order_flag" in disc and "all folds" in disc["pooling"]["rule"] or "every fold" in disc["pooling"]["rule"]
    progs = [t for t in disc["target_loader_columns"] if t.startswith("programme:mp:")]
    morph_targets = [t for t in disc["target_loader_columns"] if t.startswith(("morph:k", "morph_concept_"))]
    assert len(progs) == 8 and len(morph_targets) == 16
    src = disc["programme_targets_source"]
    if os.path.exists(os.path.join(REPO, src["path"])):
        import hashlib
        assert hashlib.sha256(open(os.path.join(REPO, src["path"]), "rb").read()).hexdigest() == src["sha256"]
    assert _family_targets(disc)["context_conditioned_programme"] == sorted(progs)
    fams = _family_targets(disc)
    assert set(fams) == {"programme_from_morphology", "morphology_from_molecular", "context_conditioned_programme",
                         "cross_modal_sets"}
    assert fams["morphology_from_molecular"] == sorted(morph_targets)
    assert "ladder" not in disc  # the decomposition is reported, never gates the v0.5 pooling
    assert set(disc["outputs"]) >= {"general_discoveries", "higher_order_discoveries"}
    assert "BH q <= 0.10" in disc["advance_rule"]["internal"]


# ------------------------------------------------------------------ v02 hit rule
PROTOCOL_V02 = os.path.join(REPO, "dbench", "protocols", "historical_rediscovery_v02.json")
HASH_FILE_V02 = os.path.join(REPO, "runs", "development", "sprint_v06", "historical_protocol_v02_sha256.txt")


def _v02(task: dict) -> dict:
    task = json.loads(json.dumps(task))
    task["scoring"].pop("r_threshold", None)
    task["scoring"].update({"hit_rule_version": "v02_top10_representative", "representative_top_n": 10,
                            "representative_min_abs_r": 0.5})
    return task


def test_v02_planted_hits_noise_misses_and_reports_ceiling(ep):
    task = _v02(planted_task())
    out = H.score_ranked_list([["genes:N1"], ["nbr6:genes:P3", "genes:N2"]], ep, task)
    assert out["rank_of_first_hit"] == 2 and out["hit_variable"] == "nbr6:genes:P3"
    assert 0.9 < out["task_ceiling"] <= 1.0 and out["hit_abs_r"] >= 0.5
    top = out["ceiling"]["niche@nbr6"]["top"]
    assert len(top) == 10 and abs(top[0][1]) == pytest.approx(out["ceiling"]["niche@nbr6"]["ceiling"])
    assert 0.9 < out["best_signed_over_ceiling"] <= 1.0 + 1e-9
    noise = H.score_ranked_list([["genes:N1"], ["nbr6:genes:N5"]], ep, task)
    assert math.isinf(noise["rank_of_first_hit"])
    assert noise["best_signed_over_ceiling"] is None or noise["best_signed_over_ceiling"] < 0.5


def test_v02_min_abs_r_and_sign(ep):
    task = _v02(planted_task())
    s = H.proxy_scores(ep, task)
    g = H._groups(ep)
    rows = np.arange(ep.n)
    p = np.asarray(s["niche@nbr6"])
    # outside-design variables: qualify when |r| >= max(0.5, 10th largest design |r|)
    for r, expect in ((0.6, 1), (0.4, math.inf)):
        v = _with_r(p, g, r)
        out = H.score_ranked_list([{"names": ["nbr6:custom"], "values": v.reshape(-1, 1)}], ep, task,
                                  scores=s, rows=rows)
        thr = out["ceiling"]["niche@nbr6"]["threshold"]
        assert out["rank_of_first_hit"] == (expect if r >= thr else math.inf), (r, thr)
    assert math.isinf(H.score_ranked_list([["nbr6:genes:P1"]], ep, _v02(planted_task(direction=-1)))
                      ["rank_of_first_hit"])


def test_v02_rescore_saved_list(ep):
    task = _v02(planted_task())
    saved = {"task_id": "SYN", "study_key": "synthetic", "method": "fixed", "before_date": "2020-01-01",
             "n_relationships_tested": 7, "seconds": 1.0, "protocol_sha256": "0" * 64,
             "top": [{"rank": 1, "names": ["genes:N4"]}, {"rank": 2, "names": ["nbr6:genes:P2"]}]}
    res = H.rescore_saved(saved, task, ep, n_boot=20)
    assert res["rank_of_first_hit"] == 2 and res["heldout_hit"]["relation"] in ("v07_ho", "v06")
    assert res["rescored_from_protocol"] == "0" * 64 and res["before_date"] == "2020-01-01"


def test_v02_protocol_supersedes_v01_and_is_pinned():
    from dbench.prospective_campaign import canonical_hash
    v01 = json.load(open(PROTOCOL, encoding="utf-8"))
    v02 = json.load(open(PROTOCOL_V02, encoding="utf-8"))
    assert v02["supersedes"]["canonical_sha256"] == canonical_hash(v01)
    assert v02["scoring"]["hit_rule_version"] == "v02_top10_representative"
    assert "r_threshold" not in v02["scoring"] and "secondary_metric" in v02["scoring"]
    strip = lambda d: {k: v for k, v in d.items() if k not in ("protocol_id", "created_utc", "status", "scoring",
                                                               "supersedes", "rationale", "tasks")}
    assert strip(v01) == strip(v02)
    for a, b in zip(v01["tasks"], v02["tasks"]):
        assert {k: v for k, v in a.items() if k != "scoring"} == {k: v for k, v in b.items() if k != "scoring"}
        assert {k: v for k, v in a["scoring"].items() if k != "hit_rule"} == \
               {k: v for k, v in b["scoring"].items() if k != "hit_rule"}
    tasks = H.load_tasks(PROTOCOL_V02)
    assert [t["task_id"] for t in tasks] == ["H1", "H2", "H3", "H3b", "S0"]
    if os.path.exists(HASH_FILE_V02):
        assert open(HASH_FILE_V02).read().split()[0] == canonical_hash(v02)


EXT_V02 = os.path.join(REPO, "dbench", "protocols", "ccrcc_external_replication_v02.json")
EXT_V02_HASH = os.path.join(REPO, "runs", "development", "sprint_v06", "ccrcc_external_replication_v02_sha256.txt")


def test_external_replication_v02_changes_only_the_null_and_fitted_columns():
    from dbench.prospective_campaign import canonical_hash
    v01 = json.load(open(EXT, encoding="utf-8"))
    v02 = json.load(open(EXT_V02, encoding="utf-8"))
    assert v02["supersedes"]["canonical_sha256"] == canonical_hash(v01)
    changed = {"protocol_id", "created_utc", "status", "supersedes", "rationale", "transfer_rule", "pass_rule"}
    assert {k: v for k, v in v01.items() if k not in changed} == {k: v for k, v in v02.items() if k not in changed}
    t2 = dict(v02["transfer_rule"])
    assert "1e-6" in t2.pop("discovery_fitted_linear_columns") and t2 == v01["transfer_rule"]
    p1, p2 = dict(v01["pass_rule"]), dict(v02["pass_rule"])
    p1.pop("permutation")
    m = p2.pop("matched_null")
    assert "199 draws" in m and "refit" in m and "section_moran_i" in m and "<= 0.05" in m
    for k in ("matched_null_scope", "matched_null_implementation", "all_required_list"):
        p2.pop(k)
    assert p1 == p2
    if os.path.exists(EXT_V02_HASH):
        assert open(EXT_V02_HASH).read().split()[0] == canonical_hash(v02)
