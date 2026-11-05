"""Structured nulls and the RDC metric.

A null that randomises everything is a different experiment, not a null, so
every generator has to be checked on both halves of its claim: the statistic it
says it preserves really is preserved, and the relationship it says it destroys
really is destroyed. `passes` in the generator's own report asserts exactly
that, and the tests here check that `passes` can be False -- a report that is
always True is a decoration.

The RDC tests are about ordering and attribution rather than about a number: a
count of surviving discoveries that cannot say which of them matched a planted
artifact is not a scientific measurement.
"""
import numpy as np
import pytest

from dbench import nulls as N
from dbench.environments import ALL_WORLDS, build_world
from dbench.metrics import (Claim, artifact_susceptibility, effect_calibration,
                            match_claims, n80, object_stability,
                            per_family_recall, rdc, replicated)
from dbench.util import acf, corr, knn_neighbors, morans_i


# --------------------------------------------------------------------------
# every family declares a null policy its structure can actually support
# --------------------------------------------------------------------------
def test_every_family_declares_a_null_policy():
    for fam in ALL_WORLDS:
        assert fam in N.DEFAULT_NULLS, fam
        for k in N.DEFAULT_NULLS[fam]:
            assert k in N.NULL_KINDS, (fam, k)


def test_declared_nulls_are_buildable_and_self_report_success():
    """The policy must not name a null the world cannot support."""
    fails = {}
    for fam, kinds in N.DEFAULT_NULLS.items():
        for kind in kinds:
            w = build_world(fam, 0)
            try:
                nw, rep = N.make_null(w, kind, seed=5)
            except Exception as exc:                       # pragma: no cover
                fails["%s/%s" % (fam, kind)] = repr(exc)
                continue
            if not rep.get("passes"):
                fails["%s/%s" % (fam, kind)] = rep
    assert not fails, fails


def test_a_null_world_marks_every_object_false():
    w = build_world("pure_and", 0)
    nw, _ = N.make_null(w, "iid_permute", seed=1)
    assert nw.truth.objects
    assert not nw.truth.true_objects
    assert all(h.artifact_class == "structured_null" for h in nw.truth.objects)


def test_unknown_null_kind_is_refused():
    w = build_world("pure_and", 0)
    with pytest.raises(KeyError):
        N.make_null(w, "shuffle_everything")


# --------------------------------------------------------------------------
# preserved / broken, per generator
# --------------------------------------------------------------------------
def test_iid_permute_preserves_the_marginal_and_destroys_the_association():
    w = build_world("additive_factors", 0)
    nw, rep = N.make_null(w, "iid_permute", seed=2)
    assert np.allclose(np.sort(w.target), np.sort(nw.target))
    for h in w.truth.true_objects:
        before = abs(float(corr(w.target, w.latents[h.operands[0]])))
        after = abs(float(corr(nw.target, nw.latents[h.operands[0]])))
        assert before > 0.2 and after < 0.06


def test_block_permutation_keeps_short_range_autocorrelation():
    w = build_world("temporal_lag", 0)
    nw, rep = N.make_null(w, "block_permute", seed=2)
    a0 = acf(np.asarray(w.target, float), 3)
    a1 = acf(np.asarray(nw.target, float), 3)
    assert rep["passes"], rep
    assert abs(a0[1] - a1[1]) < 0.35 * max(abs(a0[1]), 1e-9) + 0.1


def test_torus_shift_preserves_morans_i_and_breaks_the_alignment():
    w = build_world("spatial_interaction", 0)
    nw, rep = N.make_null(w, "torus_shift", seed=2)
    nb = w.structure.neighbors
    m0 = morans_i(np.asarray(w.target, float), nb)
    m1 = morans_i(np.asarray(nw.target, float), nb)
    assert m0 > 0.1
    assert abs(m0 - m1) <= max(0.02, 0.15 * abs(m0))
    assert np.allclose(np.sort(w.target), np.sort(nw.target))
    f0 = np.asarray(w.latents["A"], float) * np.asarray(w.latents["aggB"], float)
    assert abs(float(corr(w.target, f0))) > 0.3
    assert abs(float(corr(nw.target, f0))) < 0.15


def test_graph_rewire_preserves_degree():
    w = build_world("graph_motif", 0)
    nw, rep = N.make_null(w, "graph_rewire", seed=2)
    assert rep["passes"], rep


def test_graph_rewire_tests_the_complete_graph_by_nonlinear_object_when_declared():
    """A graph main-effect diagnostic is insufficient for a graph interaction.

    The boss-6 null must preserve degree/labels yet sever the predeclared
    hill(A) × graph-motif alignment.  This prevents the generic graph null
    from certifying itself by examining a deliberately weak motif main effect.
    """
    w = build_world("boss6_hill_graph", 0)
    nw, rep = N.make_null(w, "graph_rewire", seed=5)
    broken = rep["broken"]
    assert rep["passes"], rep
    assert broken["interaction_probe"] == "hillA"
    assert abs(broken["interaction_corr_with_Y_before"]) > 0.25
    assert abs(broken["interaction_corr_with_Y_after"]) < 0.25


def test_intervention_strip_keeps_the_observational_regime():
    w = build_world("intervention_causal", 0)
    nw, rep = N.make_null(w, "intervention_strip", seed=2)
    assert rep["passes"], rep
    obs = w.interventions.mask("obs")
    if obs.sum() > 100:
        assert np.allclose(np.sort(np.asarray(w.target)[obs]),
                           np.sort(np.asarray(nw.target)[obs]), atol=1e-6) or \
            abs(float(corr(np.asarray(w.target)[obs],
                           np.asarray(nw.target)[obs]))) > 0.5


def test_a_null_report_can_fail_its_own_check():
    """Handed a world whose spatial structure it cannot preserve, the torus
    shift must say so rather than return `passes=True`."""
    w = build_world("additive_factors", 0)
    with pytest.raises(ValueError):
        N.make_null(w, "torus_shift", seed=1)


def test_torus_shift_never_maps_a_point_into_its_own_neighbourhood():
    """A one-cell shift leaves the neighbourhood relationship intact, which is
    the relationship the null exists to destroy."""
    w = build_world("spatial_interaction", 0)
    for seed in range(8):
        _, rep = N.make_null(w, "torus_shift", seed=seed)
        dx, dy = rep["shift"]
        side = rep["side"]
        radius = float(w.structure.meta.get("radius", 1.5))
        assert min(dx, side - dx) > radius or min(dy, side - dy) > radius, rep


def test_knn_neighbors_groups_never_cross_a_group_boundary():
    """Two patients placed at identical coordinates: without `groups` every
    row's nearest neighbours would include the other patient's spots;
    with `groups`, a spot's neighbours must all share its own group."""
    rng = np.random.default_rng(0)
    coords_a = rng.normal(size=(20, 2))
    coords = np.vstack([coords_a, coords_a])  # patient B sits exactly on A
    groups = np.array([0] * 20 + [1] * 20)
    neighbours = knn_neighbors(coords, k=6, groups=groups)
    assert len(neighbours) == 40
    for row, nb in enumerate(neighbours):
        assert len(nb) == 6
        assert np.all(groups[nb] == groups[row])
    # groups=None keeps the original ungrouped behaviour (may cross rows)
    ungrouped = knn_neighbors(coords, k=6)
    assert any(groups[nb][0] != groups[row] for row, nb in enumerate(ungrouped))


def test_knn_neighbors_groups_handles_a_singleton_group():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    groups = np.array([0, 0, 1])
    neighbours = knn_neighbors(coords, k=6, groups=groups)
    assert neighbours[2].size == 0
    assert set(neighbours[0].tolist()) == {1}


# --------------------------------------------------------------------------
# RDC
# --------------------------------------------------------------------------
def _claim(cid, ops, p, eff, reps, operator="AND"):
    return Claim(id=cid, operator=operator, operands=tuple(ops), target="Y",
                 p_value=p, effect=eff,
                 replication={"R%d" % i: r for i, r in enumerate(reps)})


def test_replication_requires_sign_agreement():
    c = _claim("c", ("A0", "B0"), 1e-8, 0.5, [(-0.5, 1e-6)])
    assert replicated(c)["n_replicated"] == 0
    assert replicated(c, require_sign=False)["n_replicated"] == 1


def test_rdc_counts_only_claims_that_pass_both_gates():
    w = build_world("pure_and", 0)
    good = _claim("good", ("A0", "B0"), 1e-10, 0.5, [(0.5, 1e-6), (0.4, 1e-5)])
    weak_p = _claim("weakp", ("A1", "B1"), 0.4, 0.5, [(0.5, 1e-6)])
    no_rep = _claim("norep", ("A1", "B1"), 1e-10, 0.5, [(0.5, 0.9)])
    rep = rdc([good, weak_p, no_rep], w.truth)
    ids = [r["claim_id"] for r in rep["per_claim"] if r["counts_toward_rdc"]]
    assert ids == ["good"]
    assert rep["RDC"] == 1 and rep["n_surviving_true"] == 1


def test_the_utility_constraint_is_a_precondition_and_not_a_tiebreak():
    w = build_world("pure_and", 0)
    good = _claim("good", ("A0", "B0"), 1e-10, 0.5, [(0.5, 1e-6)])
    ok = rdc([good], w.truth, delta=0.02, utility=0.99, utility_baseline=1.0)
    bad = rdc([good], w.truth, delta=0.02, utility=0.90, utility_baseline=1.0)
    assert ok["RDC"] == 1
    # RDC is now the spec's quantity (true objects); the utility constraint
    # still zeroes it, and the uncapped counts are kept under names that
    # cannot be mistaken for it
    assert bad["RDC"] == 0 and bad["RDC_true_uncapped"] == 1
    assert bad["RDC_all_uncapped"] == 1
    assert bad["utility_constraint_met"] is False


def test_rdc_attributes_survivors_to_true_objects_artifacts_and_neither():
    from dbench.artifacts import ALL_CLASSES
    from dbench.families import make_pure_and
    w = make_pure_and(seed=0, n=4000, artifacts=("sign_reversing",))
    art = [h for h in w.truth.artifacts][0]
    claims = [
        _claim("true", ("A0", "B0"), 1e-10, 0.5, [(0.5, 1e-6)]),
        _claim("art", art.operands, 1e-10, 0.5, [(0.5, 1e-6)],
               operator=art.operator),
        _claim("none", ("D0", "D1"), 1e-10, 0.5, [(0.5, 1e-6)]),
    ]
    rep = rdc(claims, w.truth)
    assert rep["n_surviving_true"] == 1
    assert rep["n_surviving_planted_artifact"] == 1
    assert rep["n_surviving_unmatched"] == 1
    assert rep["false_discovery_proportion"] == pytest.approx(2 / 3)
    sus = rep["artifact_susceptibility"]
    assert sus["sign_reversing"]["n_survived_rdc"] == 1


def test_artifact_susceptibility_is_reported_per_class_not_pooled():
    rows = [{"artifact_class": "replication_mimicry", "counts_toward_rdc": True},
            {"artifact_class": "sign_reversing", "counts_toward_rdc": False},
            {"artifact_class": None, "counts_toward_rdc": True}]
    sus = artifact_susceptibility(rows)
    assert set(sus) == {"replication_mimicry", "sign_reversing"}
    assert sus["replication_mimicry"]["n_survived_rdc"] == 1
    assert sus["sign_reversing"]["n_survived_rdc"] == 0


def test_rdc_on_an_empty_claim_set_is_zero_and_not_an_error():
    w = build_world("pure_and", 0)
    rep = rdc([], w.truth)
    assert rep["RDC"] == 0 and rep["false_discovery_proportion"] == 0.0


# --------------------------------------------------------------------------
# secondary metrics
# --------------------------------------------------------------------------
def test_n80_is_the_smallest_n_reaching_the_target():
    hit = n80({1000: 0.2, 2000: 0.5, 4000: 0.82, 8000: 0.95})
    assert hit["n80"] == 4000 and hit["reached"] is True
    miss = n80({1000: 0.2, 2000: 0.5})
    assert miss["n80"] is None and miss["reached"] is False
    assert miss["max_observed_rate"] == 0.5 and miss["max_n_tried"] == 2000


def test_object_stability_is_over_object_identity_not_atom_index():
    a = [Claim(id="x", operator="AND", operands=("A", "B"))]
    b = [Claim(id="y", operator="AND", operands=("B", "A"))]
    c = [Claim(id="z", operator="AND", operands=("A", "C"))]
    same = object_stability([a, b])
    assert same["mean_jaccard"] == 1.0 and same["core_fraction"] == 1.0
    diff = object_stability([a, c])
    assert diff["mean_jaccard"] < 1.0 and diff["n_core"] == 0


def test_effect_calibration_compares_reported_against_planted_effects():
    w = build_world("additive_factors", 0)
    truth = w.truth
    claims = [Claim(id="c%d" % i, operator="ADD", operands=(h.operands[0],),
                    target="Y", effect=h.effect)
              for i, h in enumerate(truth.true_objects)]
    rep = effect_calibration(claims, truth)
    assert rep["n"] == len(claims)
    assert abs(rep["slope"] - 1.0) < 1e-6


def test_per_family_recall_keeps_families_separate():
    w = build_world("pure_and", 0)
    good = _claim("good", ("A0", "B0"), 1e-10, 0.5, [(0.5, 1e-6)])
    out = per_family_recall({"pure_and": rdc([good], w.truth),
                             "xor": rdc([], w.truth)})
    assert set(out) == {"pure_and", "xor"}
    assert out["pure_and"]["recall"] > 0
