"""The shared feature evaluator, including the composite program grammar.

This module is the contract between the benchmark and the Hypothesis Compiler:
a claim is only re-testable in a replication environment if both sides build the
same design column from the same typed object. Everything here therefore asserts
returned structure -- which columns come back, what they equal, which are absent
-- and never a summary statistic about how well something fits.

Two of these tests exist because the corresponding bug was live in this package
and passed every accuracy check:

  * `parent_columns` included an ADD object's own operand in its control set, so
    every true main effect was regressed on itself and returned exactly zero
    evidence. `additive_factors` recovered 0 of 3 objects and `pure_and` 2 of 4,
    with no error anywhere.
  * the gate indicator was a median split, which is constant whenever a binary
    gate is unbalanced. Every context-gated candidate returned effect 0.000 and
    p = 1.000 -- a perfectly monotonous result that looked like a hard family.
"""
import numpy as np
import pytest

from dbench.environments import ALL_WORLDS, build_world
from dbench.features import (EVALUABLE, GATE_OPS, IDENTITY_OPS, STRUCTURAL,
                             FeatureError, build_composite, build_feature,
                             composite_operands, composite_parent_columns,
                             gate_indicator, lag_column, normalize_term,
                             parent_columns, program_signature, term_column,
                             term_label)
from dbench.util import zscore
from dbench.world import OPERATORS


# --------------------------------------------------------------------------
# an object is never its own parent
# --------------------------------------------------------------------------
@pytest.mark.parametrize("operator", IDENTITY_OPS)
def test_identity_operators_exclude_their_operand_from_the_parent_family(operator):
    """The regression that produced a silent zero for every main effect."""
    w = build_world("additive_factors", 0)
    _, names = parent_columns(w, operator, ("A0",))
    assert "A0" not in names, (
        "%s(A0) lists A0 as its own parent; the partial test then regresses "
        "the feature on itself and returns zero evidence for a true object"
        % operator)


def test_a_main_effect_has_nonzero_evidence_against_its_parent_family():
    """The behavioural consequence of the test above, stated so that a
    regression cannot pass by renaming a column."""
    from dbench.oracle import partial_evidence
    w = build_world("additive_factors", 0)
    ctrl, _ = parent_columns(w, "ADD", ("A0",))
    ev = partial_evidence(w.target, w.latents["A0"], ctrl)
    assert ev["p"] < 1e-10 and abs(ev["effect"]) > 0.1


def test_combining_operators_do_keep_their_operands_as_parents():
    """The fix must not overshoot: a conjunction still has to beat its own
    constituents, which is the whole hierarchical-novelty rule."""
    w = build_world("pure_and", 0)
    _, names = parent_columns(w, "AND", ("A0", "B0"))
    assert "A0" in names and "B0" in names


def test_threshold_keeps_the_untransformed_variable_as_a_parent():
    w = build_world("threshold", 0)
    _, names = parent_columns(w, "THRESHOLD", ("A",), {"tau": 1.0})
    assert "A" in names


def test_lag_keeps_the_contemporaneous_term_and_does_not_duplicate_it():
    w = build_world("temporal_lag", 0)
    cols, names = parent_columns(w, "LAG", ("B",), lag=3)
    assert "B@t" in names
    assert names.count("B@t") == 1
    assert "B" not in names, "the contemporaneous parent is named once, as B@t"


def test_spatial_parents_include_the_aggregate_and_the_local_density():
    w = build_world("spatial_interaction", 0)
    _, names = parent_columns(w, "SPACE", ("A", "B"), {"agg": "mean"})
    assert "A" in names and "B" in names
    assert "agg(B)" in names and "density" in names


def test_declared_nuisance_covariates_are_always_parents():
    w = build_world("spatial_interaction", 0)
    _, names = parent_columns(w, "AND", ("A", "B"))
    for k in w.nuisance:
        assert "nuisance:" + k in names


def test_parent_columns_never_returns_duplicate_names():
    w = build_world("spatial_interaction", 0)
    for ops in (("A", "A"), ("A", "B")):
        _, names = parent_columns(w, "SPACE", ops, {"agg": "mean"})
        assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# the evaluator searches, it does not look answers up
# --------------------------------------------------------------------------
def test_threshold_refuses_to_supply_a_tau():
    w = build_world("threshold", 0)
    with pytest.raises(FeatureError):
        build_feature(w, "THRESHOLD", ("A",))
    assert build_feature(w, "THRESHOLD", ("A",), {"tau": 0.4}) is not None


def test_saturate_refuses_to_supply_a_K():
    w = build_world("hill", 0)
    with pytest.raises(FeatureError):
        build_feature(w, "SATURATE", ("A",))


def test_space_refuses_a_world_with_no_neighbourhood():
    w = build_world("additive_factors", 0)
    with pytest.raises(FeatureError):
        build_feature(w, "SPACE", ("A0", "A1"))


def test_structural_operators_are_refused_a_design_column():
    """SYMMETRY, ODE, DESIGN, MANIFOLD and CONFOUND are scored by specialists;
    silently returning a column for them would let a partial F-test pretend to
    have tested a law."""
    w = build_world("hidden_symmetry", 0)
    for op in STRUCTURAL:
        with pytest.raises(FeatureError):
            build_feature(w, op, ("r",))
        with pytest.raises(FeatureError):
            term_column(w, {"op": op, "operands": ["r"]})


def test_every_operator_is_either_evaluable_or_structural():
    assert set(EVALUABLE) | set(STRUCTURAL) == set(OPERATORS) - {"CONFOUND"} | \
        set(STRUCTURAL)
    for op in OPERATORS:
        assert op in EVALUABLE or op in STRUCTURAL, (
            "%s is in the shared vocabulary but the evaluator neither builds "
            "it nor declares it structural" % op)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def test_gate_indicator_is_not_a_median_split():
    """An unbalanced binary gate must still produce a two-valued indicator.

    The median split failed exactly here: with prevalence above one half the
    median equals the gate's maximum, `g > median` is false everywhere, and the
    indicator collapses to a constant.
    """
    g = np.array([1.0] * 70 + [0.0] * 30)
    ind = gate_indicator(g, {})
    assert set(np.unique(ind)) == {0.0, 1.0}
    assert 0.0 < ind.mean() < 1.0
    assert np.allclose(gate_indicator(g, {"gate_level": 0.0}), 1.0 - ind)


def test_gate_indicator_handles_a_three_level_gate_by_level():
    g = np.array([0, 1, 2, 0, 1, 2, 2], dtype=float)
    for lv in (0.0, 1.0, 2.0):
        ind = gate_indicator(g, {"gate_level": lv})
        assert np.array_equal(ind, (g == lv).astype(float))


def test_gate_indicator_refuses_a_continuous_gate_without_a_level():
    rng = np.random.default_rng(0)
    with pytest.raises(FeatureError):
        gate_indicator(rng.standard_normal(500), {})


def test_context_gated_world_gives_a_nondegenerate_gated_column():
    w = build_world("context_gated", 0)
    f = term_column(w, {"op": "CONDITION", "operands": ["G"], "params": {}})
    assert 0.05 < f.mean() < 0.95, (
        "the gate indicator is degenerate on the world it was written for")


# --------------------------------------------------------------------------
# term_column agrees with build_feature where they overlap
# --------------------------------------------------------------------------
def test_term_column_agrees_with_build_feature_on_shared_operators():
    """Two evaluators that disagree would let a claim mean one thing in the
    proposal environment and another in replication."""
    w = build_world("pure_and", 0)
    cases = [
        ("AND", ("A0", "B0"), {}, None),
        ("XOR", ("A0", "B0"), {}, None),
        ("THRESHOLD", ("A0",), {"tau": 0.3, "temp": 0.1}, None),
        ("DIFF", ("A0", "B0"), {}, None),
        ("RATIO", ("A0", "B0"), {}, None),
    ]
    for op, ops, params, lag in cases:
        a = build_feature(w, op, ops, params, lag)
        b = term_column(w, {"op": op, "operands": list(ops), "params": params,
                            "lag": lag})
        assert np.allclose(a, b), op

    wt = build_world("temporal_lag", 0)
    assert np.allclose(build_feature(wt, "LAG", ("B",), lag=3),
                       term_column(wt, {"op": "LAG", "operands": ["B"],
                                        "lag": 3}))
    # ADD is the one deliberate difference and it is a scaling only
    a = build_feature(w, "ADD", ("A0",))
    b = term_column(w, {"op": "ADD", "operands": ["A0"]})
    assert np.allclose(zscore(a), b)


# --------------------------------------------------------------------------
# composite programs
# --------------------------------------------------------------------------
def _boss1_program(tau=0.6, lag=2):
    return [{"op": "CONDITION", "operands": ["C"], "params": {"gate_level": 1}},
            {"op": "THRESHOLD", "operands": ["A"],
             "params": {"tau": tau, "temp": 0.12}},
            {"op": "SPACE", "operands": ["B"], "params": {"agg": "mean"},
             "lag": lag}]


def test_composite_reproduces_the_generator_of_the_boss_world():
    """The strongest available check on the evaluator: rebuild BOSS-1's design
    column from its typed program and compare against the column the generator
    actually used, up to scale."""
    w = build_world("boss1_cond_thresh_space_lag", 0)
    f = build_composite(w, _boss1_program(tau=w.meta["tau"], lag=w.meta["lag"]))
    truth = (np.asarray(w.latents["C"], float)
             * np.asarray(w.latents["thrA"], float)
             * np.asarray(w.latents["aggB_lag"], float))
    r = float(np.corrcoef(f, truth)[0, 1])
    assert r > 0.999, "composite rebuild correlates %.4f with the generator" % r


def test_composite_parents_are_exactly_the_strict_sub_products():
    w = build_world("boss1_cond_thresh_space_lag", 0)
    terms = _boss1_program()
    cols, names = composite_parent_columns(w, terms)
    labels = [term_label(t) for t in terms]
    # 2^3 - 2 = 6 proper non-empty subsets
    subprods = [n for n in names if all(
        part in labels for part in n.split("*"))]
    assert len(subprods) == 6, sorted(subprods)
    assert " * ".join(labels).replace(" ", "") not in [
        n.replace(" ", "") for n in names], (
        "the full product must not appear in its own parent family")
    # each leaf variable, and the declared nuisance covariates
    for nm in composite_operands(terms):
        assert nm in names
    assert "density" in names
    for k in w.nuisance:
        assert "nuisance:" + k in names
    assert len(names) == len(set(names))
    assert all(len(c) == w.n for c in cols)


def test_composite_parent_columns_match_a_hand_built_sub_product():
    w = build_world("boss1_cond_thresh_space_lag", 0)
    terms = _boss1_program()
    cols, names = composite_parent_columns(w, terms)
    lab = "%s*%s" % (term_label(terms[0]), term_label(terms[1]))
    idx = names.index(lab)
    hand = term_column(w, terms[0]) * term_column(w, terms[1])
    assert np.allclose(cols[idx], hand)


def test_nested_operands_build_a_program_tree():
    """BOSS-2's object is XOR(A, majority-aggregate of B): no flat product of
    terms can express it, so the grammar has to nest."""
    w = build_world("boss2_xor_space_regime", 0)
    # the neighbourhood channel is the mean aggregate, binarised at its own
    # median by XOR itself -- see the balance argument in `make_boss2`
    inner = {"op": "SPACE", "operands": ["B"], "params": {"agg": "mean"}}
    terms = [{"op": "CONDITION", "operands": ["regime"],
              "params": {"gate_level": 1}},
             {"op": "XOR", "operands": ["A", inner]}]
    f = build_composite(w, terms)
    maj = np.asarray(w.latents["majB"], float)
    A = np.asarray(w.latents["A"], float)
    S = np.asarray(w.latents["regime"], float)
    hand = S * ((A != maj).astype(float) - 0.5)
    assert np.allclose(f, hand)
    assert composite_operands(terms) == ("A", "B", "regime")
    assert program_signature(terms) == ("CONDITION", "SPACE", "XOR")


def test_program_signature_is_the_compositional_split_unit():
    """Order-insensitive and parameter-free: a new seed or a new tau of a
    held-out composition must still be held out."""
    a = program_signature(_boss1_program(tau=0.6, lag=2))
    b = program_signature(_boss1_program(tau=-1.1, lag=4))
    assert a == b == ("CONDITION", "SPACE", "THRESHOLD")
    c = program_signature(list(reversed(_boss1_program())))
    assert c == a


def test_term_label_distinguishes_discrete_choices_and_not_only_continuous():
    t1 = {"op": "SPACE", "operands": ["B"], "params": {"agg": "mean"}, "lag": 2}
    t2 = {"op": "SPACE", "operands": ["B"], "params": {"agg": "majority"},
          "lag": 2}
    t3 = {"op": "SPACE", "operands": ["B"], "params": {"agg": "mean"}, "lag": 3}
    assert len({term_label(t1), term_label(t2), term_label(t3)}) == 3


def test_normalize_term_accepts_both_dict_and_tuple_forms():
    a = normalize_term({"op": "AND", "operands": ["A", "B"], "params": {},
                        "lag": None})
    b = normalize_term(("AND", ("A", "B"), {}, None))
    assert a == b


# --------------------------------------------------------------------------
# lags are panel-aware
# --------------------------------------------------------------------------
def test_lag_column_does_not_leak_across_units_in_a_panel():
    w = build_world("boss1_cond_thresh_space_lag", 0)
    x = np.asarray(w.latents["A"], float)
    lagged = lag_column(w, x, 2)
    unit = np.asarray(w.structure.unit)
    time = np.asarray(w.structure.time)
    for u in (0, 5, 17):
        m = np.where(unit == u)[0]
        order = m[np.argsort(time[m])]
        assert np.allclose(lagged[order[2:]], x[order[:-2]])
        assert np.allclose(lagged[order[:2]], 0.0), (
            "the first observations of a unit have no history and must be "
            "zero, not the tail of the previous unit")


# --------------------------------------------------------------------------
# every Tier-I family's own declared truth reproduces its generator
# --------------------------------------------------------------------------
def test_every_tier1_familys_true_y_objects_reproduce_the_generator():
    """Generalises the boss-world-specific check in `test_boss_worlds.py`
    (`test_every_boss_worlds_own_declared_terms_reproduce_its_generator`) to
    every Tier-I family, and to the flat `build_feature` path as well as
    `build_composite`.

    For a TRUE, Y-targeted Hypothesis with NO declared parents, the
    generator's own `_finish()` sets `Hypothesis.effect = corr(Y, feature)`
    directly (no partial-out) -- so reconstructing the feature via
    `build_feature`/`build_composite` and recomputing `corr(Y, rebuilt)`
    must reproduce that SAME stored number, to floating-point agreement,
    with no hand-built reference needed for each family. This is exactly
    the class of check that caught two live bugs (2026-09-15, extending the
    boss6/graph_motif audit): `build_feature`'s CONDITION/CAUSE/ANOMALY
    branch defaulted an UNDECLARED gate to the hypothesis's own sole
    operand whenever it had only one, median-split-SIGNING a plain
    mechanism effect (`latent_confound`'s `CAUSE:U->Y`/`CAUSE:W->Y`,
    `intervention_causal`'s `CAUSE:X1->Y`) instead of returning it
    directly -- discarding magnitude and silently scoring every such claim
    against the wrong reference (measured: `latent_confound`'s `CAUSE:U->Y`
    read `corr=0.576` against the generator's own `effect=0.716` before the
    fix).

    Hypotheses WITH declared parents use partial correlation internally
    (a control set this test cannot cheaply reconstruct in general, since
    `parent_features` -- the actual arrays used -- are not preserved on the
    finished `World`), so those get a weaker, still-real check: the
    reconstruction must not raise and must be non-degenerate (nonzero
    variance) -- confirming the evaluator can at least BUILD the declared
    object, which is what a `FeatureError` or an all-zero column would
    otherwise hide silently.
    """
    NONFEATURE_OPS = {"SYMMETRY", "ODE", "DESIGN"}
    # `intervention_causal`'s `CAUSE:X3->Y|do_X2` (the one genuine categorical-
    # gate gap this suite used to pin as a KNOWN_GAP) is fixed as of
    # 2026-09-15 (Bench-Owner): see `test_intervention_causal_categorical_gate_
    # reproduces_the_generator_exactly` below for the dedicated abs_corr check
    # against the raw generative signal. No KNOWN_GAP remains; this loop now
    # runs unconditionally over every TRUE, Y-targeted Hypothesis.
    checked_no_parents = 0
    checked_has_parents = 0
    for fam in sorted(ALL_WORLDS):
        w = build_world(fam, 0)
        for h in w.truth.objects:
            if not h.true or h.operator in NONFEATURE_OPS:
                continue
            if getattr(h, "target", "Y") != "Y":
                continue
            terms = h.params.get("terms")
            f = build_composite(w, terms) if terms else build_feature(
                w, h.operator, h.operands, h.params, h.lag)
            f = np.asarray(f, dtype=np.float64).ravel()
            if not h.parents:
                r = float(np.corrcoef(w.target, f)[0, 1])
                assert h.effect is None or not np.isfinite(h.effect) or \
                    abs(abs(r) - abs(h.effect)) < 0.02, (
                        "%s/%s: rebuilt corr=%.4f vs stored effect=%.4f"
                        % (fam, h.id, r, h.effect))
                checked_no_parents += 1
            else:
                assert f.std() > 1e-9, "%s/%s: reconstruction is degenerate" % (fam, h.id)
                checked_has_parents += 1
    assert checked_no_parents >= 10, "the no-parents branch checked suspiciously few objects"
    assert checked_has_parents >= 10, "the has-parents branch checked suspiciously few objects"


def test_intervention_causal_categorical_gate_reproduces_the_generator_exactly():
    """The categorical-regime gate closed 2026-09-15 (Bench-Owner).

    `CAUSE:X3->Y|do_X2`'s true generative signal is `gate * X3` where
    `gate = (regime == "do_X2")` -- a NAMED categorical env-label column
    `build_feature`'s flat CAUSE path cannot consult at all. The fix
    declares one derived latent (`X3_gated_do_X2`, registered via `b.derive`
    exactly like boss6's `A_clip`) equal to that exact product, and a
    single-term `IDENTITY` composite over it -- which reproduces the signal
    exactly because IDENTITY's z-scoring is applied to the SAME quantity,
    not to X3 alone (multiplying a separately z-scored X3 by the gate
    AFTER the fact does not work: gate * zscore(X3) is not a scalar
    multiple of gate * X3 whenever the do_X3 override shifts X3's
    population mean off zero, which it does -- that path only reaches
    ~0.93, measured directly, not a near-miss).
    """
    for seed in range(5):
        w = build_world("intervention_causal", seed)
        h = next(x for x in w.truth.objects if x.id == "CAUSE:X3->Y|do_X2")
        assert h.params.get("terms"), "the categorical-gate terms composite is missing"
        X3 = np.asarray(w.latents["X3"], dtype=np.float64)
        regime = np.asarray(w.env.labels["regime"])
        gate = (regime == "do_X2").astype(np.float64)
        f_true = gate * X3
        f_rebuilt = np.asarray(build_composite(w, h.params["terms"]), dtype=np.float64).ravel()
        r = float(np.corrcoef(f_true, f_rebuilt)[0, 1])
        assert abs(r) >= 0.999, "seed %d: abs_corr=%.6f < 0.999" % (seed, abs(r))
        # The derived latent must be declared, not silently added -- the
        # same lint/runtime pair `test_derived_latent_declaration.py`
        # enforces for every other world.
        assert "X3_gated_do_X2" in w.meta.get("derived_latents", ())
