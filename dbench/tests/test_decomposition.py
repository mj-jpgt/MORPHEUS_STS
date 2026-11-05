"""Known-answer tests for the nested explanation ladder (dbench/decomposition.py)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dbench import decomposition as dc

NAMES = (["depth_log1p", "cv_frac_A", "cv_frac_B", "cv_frac_C", "mp:P1", "mp:P2", "hallmark:H1", "hallmark:H2",
          "nbr:mp:P1", "nbr18:mp:P1", "pca:PC1", "nbr:PC1"])


def _world(seed, *, n_pat=10, rows=150, y_fn, h_fn):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_pat), rows)
    n = len(groups)
    B = rng.standard_normal((n, len(NAMES)))
    cols = {nm: B[:, i] for i, nm in enumerate(NAMES)}
    noise = rng.standard_normal(n)
    h = h_fn(cols, rng, n)
    y = y_fn(cols, h, noise)
    return y, B, groups, h


def _folds(y, B, groups, V, *, n_folds=5, hsh="ab" * 32, extra=None):
    pats = np.unique(groups)
    out = {}
    for f in range(n_folds):
        held = pats[f::n_folds]
        r = np.flatnonzero(np.isin(groups, held)); p = np.flatnonzero(~np.isin(groups, held))
        fd = dc.FoldData(y=y, B=B, baseline_names=list(NAMES), groups=groups, proposal=p, replication=r,
                         extra_competitors=extra or {})
        out[f] = dc.FoldLadder(fd).evaluate(V.reshape(len(y), -1), feature_form="linear_subspace",
                                            candidate_hash=hsh)
    return out


def test_levels_are_nested_and_grouped():
    info = dc.ladder_levels(NAMES)
    L = info["levels"]
    assert L["N"] == [0]
    assert set(L["N"]) < set(L["B1"]) < set(L["B2"]) < set(L["B3"]) and L["B3"] == list(range(len(NAMES)))
    assert info["group_counts"] == {"nuisance": 1, "composition": 3, "programmes": 4, "context": 4}


def test_stats_engine_matches_runner_cv_alpha_and_ridge():
    from morpheus_encoder.general_discovery import VALIDATION_ALPHA_GRID, _cv_alpha, _RidgeFit
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(9), 40)
    X = rng.standard_normal((len(groups), 6)) * [1, 2, 0.5, 3, 1, 1] + [0, 1, -2, 0, 5, 0]
    y = X @ [0.3, -0.2, 0.0, 0.1, 0.05, 0.0] + rng.standard_normal(len(groups))
    st = dc.patient_stats(X, y, groups)
    for seed in (0, 7, 12345):
        a_ref = _cv_alpha(X, y, groups, VALIDATION_ALPHA_GRID, seed)
        a, _ = dc.cv_alpha_from_stats(st, np.arange(6), seed, VALIDATION_ALPHA_GRID)
        assert a == a_ref
    for alpha in (1.0, 100.0, float("inf")):
        c, w = dc.fit_from_stats(st, np.arange(6), alpha)
        np.testing.assert_allclose(c + X @ w, _RidgeFit(X, y, alpha).predict(X), atol=1e-9)


@pytest.mark.parametrize("form,k", [("linear_subspace", 1), ("product", 2), ("product", 3)])
def test_ladder_equals_fitted_relation(form, k):
    """B3 Delta = FittedRelation(relation=v05) gain for k = 1 (its competitor is the reference);
    v04_relation_gain = FittedRelation(relation=v04) gain for every k."""
    from morpheus_encoder.general_discovery import VALIDATION_ALPHA_GRID, Candidate, FittedRelation, _r2
    rng = np.random.default_rng(11 + k)
    groups = np.repeat(np.arange(10), 80)
    n = len(groups)
    B = rng.standard_normal((n, len(NAMES)))
    V = rng.standard_normal((n, k))
    y = 0.4 * B[:, 1] + 0.5 * np.sin(2 * V[:, 0]) + (0.4 * V[:, 0] * V[:, -1] if k > 1 else 0) \
        + rng.standard_normal(n)
    held = [2, 7]
    r = np.flatnonzero(np.isin(groups, held)); p = np.flatnonzero(~np.isin(groups, held))
    hsh = "c0ffee" + "0" * 58
    fd = dc.FoldData(y=y, B=B, baseline_names=list(NAMES), groups=groups, proposal=p, replication=r,
                     extra_competitors={"v04": B})
    res = dc.FoldLadder(fd).evaluate(V, feature_form=form, candidate_hash=hsh, extra_levels=("v04",),
                                     v04_check="v04")
    cand = Candidate(tuple(range(k)), feature_form=form, weights=(1.0,) * k if form == "linear_subspace" else ())
    object.__setattr__(cand, "candidate_hash", hsh)
    for relation in ("v04", "v05"):
        f = FittedRelation(cand, V[p], y[p], context=None, baseline=B[p], model="rbf", ridge_alpha=1.0,
                           rbf_features=24, groups=groups[p], alpha_grid=VALIDATION_ALPHA_GRID, relation=relation)
        gain, simple, full = f.gain(V[r], y[r], context=None, baseline=B[r])
        assert abs(res["levels"]["B3"]["r2_full"] - full) < 1e-8
        if relation == "v04":
            assert abs(res["levels"]["v04"]["v04_relation_gain"] - gain) < 1e-8
        elif k == 1:
            assert abs(res["levels"]["B3"]["delta"] - gain) < 1e-8
    # the reference at B3 is the competitor-alone ridge
    from morpheus_encoder.general_discovery import _cv_alpha, _RidgeFit
    a = _cv_alpha(B[p], y[p], groups[p], VALIDATION_ALPHA_GRID, int(hsh[:8], 16))
    assert abs(res["levels"]["B3"]["r2_ref"] - _r2(y[r], _RidgeFit(B[p], y[p], a).predict(B[r]))) < 1e-8


def test_pooled_bootstrap_matches_direct_r2():
    y, B, g, h = _world(0, y_fn=lambda c, h, e: 0.5 * h + e, h_fn=lambda c, rng, n: rng.standard_normal(n))
    res = _folds(y, B, g, h)
    pooled = dc.pool_folds(res, n_boot=200)
    # direct pooled R^2 of the reference from the per-patient SSEs
    sse = sum(sum(res[f]["levels"]["N"]["sse_by_patient"]["ref"].values()) for f in res)
    ys = np.concatenate([y[np.isin(g, np.unique(g)[f::5])] for f in range(5)])
    assert abs(pooled["levels"]["N"]["r2_competitor"] - (1 - sse / np.sum((ys - ys.mean()) ** 2))) < 1e-10
    assert pooled["n_patients"] == 10


def test_planted_effect_on_nuisance_residual_is_irreducible():
    """H independent of every comparison column: Delta_0 ~ Delta_3 > 0."""
    y, B, g, h = _world(1, y_fn=lambda c, h, e: 0.6 * h + 0.5 * c["depth_log1p"] + 0.4 * c["cv_frac_A"]
                        + 0.4 * c["mp:P1"] + 0.4 * c["nbr:mp:P1"] + e,
                        h_fn=lambda c, rng, n: rng.standard_normal(n))
    pooled = dc.pool_folds(_folds(y, B, g, h), n_boot=300)
    d0, d3 = pooled["levels"]["N"]["delta"], pooled["levels"]["B3"]["delta"]
    assert d0 > 0.1 and d3 > 0.1
    assert abs(d0 - d3) < 0.02
    t = dc.assign_tier(pooled)
    assert t["tier"] == "irreducible"


def test_planted_composition_effect_is_absorbed_at_b1():
    """H = a composition column (+ small noise): Delta_0 > 0, Delta_1 ~ 0, absorbed by composition."""
    y, B, g, h = _world(2, y_fn=lambda c, h, e: 0.8 * c["cv_frac_B"] + 0.3 * c["mp:P2"] + e,
                        h_fn=lambda c, rng, n: c["cv_frac_B"] + 0.05 * rng.standard_normal(n))
    pooled = dc.pool_folds(_folds(y, B, g, h), n_boot=300)
    L = pooled["levels"]
    assert L["N"]["delta"] > 0.2
    assert abs(L["B1"]["delta"]) < 0.01 and abs(L["B3"]["delta"]) < 0.01
    t = dc.assign_tier(pooled)
    assert t["tier"] == "simple association (novelty unchecked)"
    assert t["absorbing_level"] == "B1"


def test_competitor_r2_nested_monotone_in_expectation():
    """Each level adds true signal: mean competitor R^2 rises N < B1 < B2 < B3; a level that adds
    only noise columns costs (in expectation) almost nothing."""
    def y_signal(c, h, e):
        return 0.4 * c["depth_log1p"] + 0.4 * c["cv_frac_A"] + 0.4 * c["hallmark:H1"] + 0.4 * c["pca:PC1"] + e

    def y_noise_levels(c, h, e):
        return 0.6 * c["depth_log1p"] + e

    for y_fn, strict in ((y_signal, True), (y_noise_levels, False)):
        r2 = {lv: [] for lv in dc.LEVELS}
        for s in range(4):
            y, B, g, h = _world(100 + s, y_fn=y_fn, h_fn=lambda c, rng, n: rng.standard_normal(n))
            pooled = dc.pool_folds(_folds(y, B, g, h, hsh="%064x" % (s + 1)), n_boot=50)
            for lv in dc.LEVELS:
                r2[lv].append(pooled["levels"][lv]["r2_competitor"])
        m = [float(np.mean(r2[lv])) for lv in dc.LEVELS]
        if strict:
            assert m[0] < m[1] < m[2] < m[3]
        else:
            assert min(np.diff(m)) > -0.01


def test_tiers():
    def pooled(d0, ci0, d3, ci3, d1=None, d2=None):
        L = {"N": {"delta": d0, "delta_ci95": ci0}, "B3": {"delta": d3, "delta_ci95": ci3},
             "B1": {"delta": d1 if d1 is not None else d0, "delta_ci95": ci0},
             "B2": {"delta": d2 if d2 is not None else d3, "delta_ci95": ci3}}
        return {"levels": L}
    assert dc.assign_tier(pooled(0.05, [0.02, 0.08], 0.0, [-0.01, 0.01]))["tier"] == "simple association (novelty unchecked)"
    assert dc.assign_tier(pooled(0.05, [0.02, 0.08], 0.04, [0.01, 0.06]))["tier"] == "irreducible"
    assert dc.assign_tier(pooled(0.05, [0.02, 0.08], 0.01, [0.002, 0.02]))["tier"] == "partly absorbed"
    t = dc.assign_tier(pooled(0.05, [-0.01, 0.08], 0.04, [0.01, 0.06]))
    assert t["tier"] == "none" and t["resolved_only_after_adjustment"]
    t = dc.assign_tier(pooled(0.05, [0.02, 0.08], 0.0, [-0.01, 0.01], d1=0.045, d2=0.0))
    assert t["absorbing_level"] == "B2"
    assert dc.shape_label([0.01, 0.02], [-0.01, 0.01]) == "linear_carries_it"


def test_target_programme_overlap_flag():
    sets = {"gavish:IG": ["IGHG1", "IGKC", "JCHAIN", "MZB1", "A", "B", "C", "D", "E", "F"],
            "hallmark:HALLMARK_X": ["IGHG1", "IGKC", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8"],
            "hallmark:HALLMARK_Y": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "IGKC"]}
    f = lambda v, t="nbr:IG": dc.target_programme_overlap(v, t, sets)
    assert f(["programmes:gavish:IG"])["target_programme_overlap"] is True
    assert f(["raw_genes:IGHG1"])["target_programme_overlap"] is True
    assert f(["raw_genes:CD3E"])["target_programme_overlap"] is False
    r = f(["context_neighbour_mean:nbr:hallmark:HALLMARK_X"], "own:IG")
    assert r["target_programme_overlap"] is True and r["variables"][0]["overlap_fraction_of_target_set"] == 0.2
    assert f(["programmes:hallmark:HALLMARK_Y"])["target_programme_overlap"] is False  # 10 %
    assert f(["scgpt:6"])["target_programme_overlap"] is False
    assert f(["scgpt:6"], "morph_pc4") is None


# ---------------------------------------------------------------- real fold --

_RUN_REL = Path("runs/prospective/renal-spatial-v04-20261011")
# the registered receipts are untracked: in a clone without them, read the main checkout's copy
RUN_V04 = next((p for p in (_RUN_REL, Path.home() / "morpheus_work" / _RUN_REL) if (p / "receipts").is_dir()), _RUN_REL)
STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"


@pytest.mark.skipif(not (RUN_V04 / "receipts").is_dir() or not os.path.isdir(STORE),
                    reason="registered v0.4 receipts / store not available")
def test_v04_competitor_reproduces_registered_receipt_gain():
    """morph_pc4 fold 0, scimilarity:15 (the v0.4 internal object): the ladder engine's v0.4-relation
    gain at the v0.4 comparison model equals the receipt's replication_gain to 1e-6."""
    from dbench.scripts import decomposition_ladder as dl
    receipt = json.loads((RUN_V04 / "receipts/molecular_to_morphology/f0_morph_pc4.json").read_text())
    row = next(c for c in receipt["candidates"] if c["variable_names"] == ["scimilarity:15"])
    ep = dl.load_episode("v04", 0, "morph_pc4")
    d = dl.family_design(ep, dl.read_protocol(dl.PROTOCOL_V04), "molecular_to_morphology", "morph_pc4")
    B = np.asarray(ep.context["baseline"], float)
    fd = dc.FoldData(y=np.asarray(ep.response, float), B=B, baseline_names=list(ep.metadata["baseline_columns"]),
                     groups=np.asarray(ep.context["patient_codes"]), proposal=np.asarray(ep.partitions.proposal),
                     replication=np.asarray(ep.partitions.replication), extra_competitors={"v04": B})
    V = dl.design_columns(d, ["scimilarity:15"])
    res = dc.FoldLadder(fd).evaluate(V, feature_form="linear_subspace", candidate_hash=row["candidate_hash"],
                                     levels=("B3",), extra_levels=("v04",), v04_check="v04")
    assert abs(res["levels"]["v04"]["v04_relation_gain"] - row["replication_gain"]) < 1e-6
    assert abs(res["levels"]["v04"]["r2_full"] - row["replication_r2_full"]) < 1e-6
