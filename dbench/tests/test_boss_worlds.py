"""Composite boss worlds and their specialists.

These are the benchmark's headline instances: "a system that wins only ten
independent toy tasks is not a universal discovery learner." Each world
composes three or four primitives plus, in BOSS-1, a confounder, and each has a
specialist that is handed the *composition template* and nothing else -- which
variable fills each slot, the threshold location, the Hill constants, the
aggregation kind, the lag and the gate level are all searched.

BOSS-6 is the one declared exception to the three-primitive floor: it is
graph-native (`GRAPH`, `SATURATE`), and its topological difficulty -- a
degree/composition-matched planted-motif excess under an explicit rewiring
null -- is not additional named-operator composition, so it is not padded
with an invented third primitive just to clear the general bar.

BOSS-1 is deliberately expensive (it evaluates a few thousand programs, then a
grid-wide permutation null) and takes on the order of a minute. It is left in
the suite rather than marked away: it is the world the spec writes out
explicitly, and a test for it that never runs would be worse than none.
"""
import numpy as np
import pytest

from dbench import oracle as O
from dbench.boss import BOSS_COMPOSITION, BOSS_WORLDS
from dbench.environments import build_world
from dbench.features import build_composite, program_signature
from dbench.util import benjamini_hochberg, corr


def _flagged(claims, alpha=0.05):
    if not claims:
        return []
    p = np.array([c.p_value for c in claims])
    rej, _ = benjamini_hochberg(p, alpha)
    return [c for c, r in zip(claims, rej) if r]


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def test_each_boss_world_declares_its_composition_and_uses_it():
    for name, comp in BOSS_COMPOSITION.items():
        w = build_world(name, 0)
        assert tuple(w.meta["composition"]) == tuple(comp)
        if name == "boss6_hill_graph":
            # declared exception: graph-native, 2 named primitives (see the
            # module docstring)
            assert len(comp) == 2
        else:
            assert len(comp) >= 3, "a boss world composes at least three primitives"
        declared = set()
        for h in w.truth.true_objects:
            declared |= set(h.params.get("composition", ()))
            declared.add(h.operator)
        assert set(comp) <= declared, (name, sorted(set(comp) - declared))


def test_boss1_is_the_spec_example_with_all_six_stressors():
    """conditionality, thresholding, space, time, higher-order interaction and
    confounding, in one world."""
    w = build_world("boss1_cond_thresh_space_lag", 0)
    assert set(w.meta["composition"]) == {"CONDITION", "THRESHOLD", "SPACE",
                                          "LAG"}
    assert w.structure.kind == "spatiotemporal"
    assert w.structure.coords is not None and w.structure.time is not None
    assert "Q" in w.nuisance
    conf = [h for h in w.truth.artifacts if h.operator == "CONFOUND"]
    assert conf and conf[0].params["confounder"] == "Q"


def test_boss1_confounder_creates_a_real_but_wrong_association():
    """Q raises both A and Y, so A~Y is real, reproducible, and not a
    mechanism. If it were weak the confounding arm would test nothing."""
    w = build_world("boss1_cond_thresh_space_lag", 0)
    A = np.asarray(w.latents["A"], float)
    Q = np.asarray(w.nuisance["Q"], float)
    y = np.asarray(w.target, float)
    assert abs(float(corr(A, Q))) > 0.4
    assert abs(float(corr(y, A))) > 0.15


def test_boss2_constituents_are_marginally_uninformative():
    """A spatial parity inside a minority regime: pooling over time dilutes an
    already-zero marginal."""
    w = build_world("boss2_xor_space_regime", 0)
    y = np.asarray(w.target, float)
    band = 4.0 / np.sqrt(w.n)
    for nm in ("A", "B", "majB", "aggB"):
        r = abs(float(corr(y, np.asarray(w.latents[nm], float))))
        assert r < band, (
            "%s carries marginal |r| = %.4f against a sampling band of %.4f"
            % (nm, r, band))
    # falsifiability: the composed object itself is strongly associated
    f = (np.asarray(w.latents["regime"], float)
         * ((np.asarray(w.latents["A"], float)
             != np.asarray(w.latents["majB"], float)).astype(float) - 0.5))
    assert abs(float(corr(y, f))) > 10 * band
    frac = float(np.mean(np.asarray(w.latents["regime"], float)))
    assert 0.05 < frac < 0.95


def test_boss3_effect_is_present_only_under_intervention():
    w = build_world("boss3_hill_space_cause", 0)
    y = np.asarray(w.target, float)
    h = np.asarray(w.latents["hillD"], float)
    m = w.interventions.mask("do_treat")
    assert abs(float(corr(y[m], h[m]))) > 0.3
    assert abs(float(corr(y[~m], h[~m]))) < 0.25


def test_boss4_regime_flips_the_sign_of_one_coefficient():
    w = build_world("boss4_ode_manifold_anomaly", 0)
    assert w.target is None, "BOSS-4 has no response; it is a law world"
    assert 0.0 < w.meta["regime_prevalence"] < 0.2
    assert w.meta["coefs"]["a2"] != 0.0


def test_boss5_stores_an_exact_design_answer():
    w = build_world("boss5_design_and_threshold", 0)
    eig = w.meta["eig"]
    assert w.meta["a_star"] == max(eig, key=eig.get)
    ordered = sorted(eig.values(), reverse=True)
    assert ordered[0] - ordered[1] > 0.02, (
        "a near-tie design problem cannot distinguish a right answer from a "
        "lucky one")


# --------------------------------------------------------------------------
# the composite evaluator reproduces each generator
# --------------------------------------------------------------------------
def test_the_shared_evaluator_rebuilds_every_boss_relational_object():
    """If the evaluator and the generator disagreed, a claim would mean one
    thing when it was made and another when it was replicated."""
    cases = {
        "boss1_cond_thresh_space_lag": (
            lambda w: [{"op": "CONDITION", "operands": ["C"],
                        "params": {"gate_level": 1}},
                       {"op": "THRESHOLD", "operands": ["A"],
                        "params": {"tau": w.meta["tau"], "temp": 0.12}},
                       {"op": "SPACE", "operands": ["B"],
                        "params": {"agg": "mean"}, "lag": w.meta["lag"]}],
            lambda w: (np.asarray(w.latents["C"], float)
                       * np.asarray(w.latents["thrA"], float)
                       * np.asarray(w.latents["aggB_lag"], float))),
        "boss2_xor_space_regime": (
            lambda w: [{"op": "CONDITION", "operands": ["regime"],
                        "params": {"gate_level": 1}},
                       {"op": "XOR", "operands": [
                           "A", {"op": "SPACE", "operands": ["B"],
                                 "params": {"agg": "mean"}}]}],
            lambda w: (np.asarray(w.latents["regime"], float)
                       * ((np.asarray(w.latents["A"], float)
                           != np.asarray(w.latents["majB"], float))
                          .astype(float) - 0.5))),
        "boss3_hill_space_cause": (
            lambda w: [{"op": "CAUSE", "operands": ["treat"],
                        "params": {"gate_level": 1}},
                       {"op": "SATURATE", "operands": [
                           {"op": "SPACE", "operands": ["D"],
                            "params": {"agg": "mean"}}],
                        "params": {"K": w.meta["K"], "n": w.meta["nexp"]}}],
            lambda w: (np.asarray(w.latents["treat"], float)
                       * np.asarray(w.latents["hillD"], float))),
    }
    for fam, (mk, ref) in cases.items():
        w = build_world(fam, 0)
        f = build_composite(w, mk(w))
        t = ref(w)
        r = float(np.corrcoef(f, t)[0, 1])
        assert r > 0.999, "%s: evaluator/generator correlation %.5f" % (fam, r)


def test_every_boss_worlds_own_declared_terms_reproduce_its_generator():
    """The regression the above test could not catch: it hand-rolls its own
    `terms` list inside the test, so it never notices if a world's OWN
    `Hypothesis.params["terms"]` (what `_truth_values`/`recovery_metrics`
    actually reads at evaluation time) is missing or wrong. BOSS-1's
    `params["terms"]` was missing entirely until 2026-09-15: every caller of
    the generic (non-test) reconstruction path fell through to
    `build_feature`'s flat CONDITION branch (`gate * z(A) * z(B)`), which
    cannot represent BOSS-1's declared THRESHOLD/SPACE/LAG chain and, with
    the gate fields read from the wrong dict key, degenerated to a near-
    constant reference (`abs_corr` versus the true generative signal as low
    as 0.03-0.05 -- see 2026-09-15 Arch-Owner/Bench-Owner posts). This test
    reads each world's OWN declared `terms` and checks it against the
    generator's own stored latents directly, so a future regression here is
    caught by CI, not by a coordinator noticing an oddly low correlation."""
    hand_truth = {
        "boss1_cond_thresh_space_lag": (
            "BOSS1:C&thr(A)&aggB_lag",
            lambda lat: (np.asarray(lat["C"], float) * np.asarray(lat["thrA"], float)
                        * np.asarray(lat["aggB_lag"], float))),
        "boss2_xor_space_regime": (
            "BOSS2:xor(A,majB)|regime",
            lambda lat: (np.asarray(lat["regime"], float)
                        * ((np.asarray(lat["A"], float) != np.asarray(lat["majB"], float))
                           .astype(float) - 0.5))),
        "boss3_hill_space_cause": (
            "BOSS3:hill(aggD)|do_treat",
            lambda lat: np.asarray(lat["treat"], float) * np.asarray(lat["hillD"], float)),
        "boss7_causal_hill_lag": (
            "BOSS7:do_treat*hill(A_lag)",
            lambda lat: (np.asarray(lat["treat"], float)
                        * np.asarray(lat["hillA_lag3"], float))),
        "boss6_hill_graph": (
            "BOSS6:hill(A)*labelled_triangle_excess",
            lambda lat: (_zscore(np.asarray(lat["hillA"], float))
                        * _zscore(np.asarray(lat["planted_motifs"], float)))),
    }
    for fam, (hyp_id, ref) in hand_truth.items():
        w = build_world(fam, 0)
        hyp = next(h for h in w.truth.objects if h.id == hyp_id)
        terms = hyp.params.get("terms")
        assert terms, "%s: Hypothesis.params has no 'terms' composite declared" % fam
        f = build_composite(w, terms)
        t = ref(w.latents)
        r = float(np.corrcoef(f, t)[0, 1])
        assert r > 0.999, ("%s: declared params['terms'] does not reproduce "
                           "the stored generative signal (r=%.5f)" % (fam, r))


def test_boss6_truth_operand_is_the_generative_variable_not_the_raw_count():
    """Correctness fix, 2026-09-15 (owner directive): BOSS-6's declared
    truth operand was `motif_count` (the raw, un-rewired triangle count),
    but the generative feature always multiplied by `planted_motifs` (the
    excess AFTER degree-preserving rewiring) -- a different latent, per the
    world's own docstring ("not reducible to ... an unconstrained triangle
    count"). Fixed by changing the declared operand, not by reinterpreting
    the feature. `motif_count` stays in `parents` as a real, simpler
    alternative explanation to control against -- it is a legitimate
    covariate, just never the object's own operand. Pinning the corrected
    identity here so a future edit to either side is deliberate."""
    w = build_world("boss6_hill_graph", 0)
    hyp = next(h for h in w.truth.objects if h.id == "BOSS6:hill(A)*labelled_triangle_excess")
    assert hyp.operands == ("A", "planted_motifs")
    assert "motif_count" not in hyp.operands
    assert ("motif_count",) in hyp.parents


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def test_program_signatures_match_the_declared_compositions():
    w = build_world("boss2_xor_space_regime", 0)
    sig = program_signature([
        {"op": "CONDITION", "operands": ["regime"], "params": {"gate_level": 1}},
        {"op": "XOR", "operands": ["A", {"op": "SPACE", "operands": ["B"],
                                         "params": {"agg": "mean"}}]}])
    assert set(sig) <= set(BOSS_COMPOSITION["boss2_xor_space_regime"]) | \
        {"CONDITION"}


# --------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ("boss2_xor_space_regime",
                                    "boss3_hill_space_cause",
                                    "boss4_ode_manifold_anomaly",
                                    "boss5_design_and_threshold"))
def test_boss_recovery_at_high_snr(family):
    w = build_world(family, 11, snr="high")
    rep = O.run_family(w, family)
    assert rep["recovered"] is True, {"family": family, "report": rep}


def test_boss1_recovery_at_high_snr():
    w = build_world("boss1_cond_thresh_space_lag", 11, snr="high")
    rep = O.run_family(w, "boss1_cond_thresh_space_lag")
    assert rep["recovered"] is True, rep
    assert rep["recall"] == 1.0


def test_the_boss_specialists_report_how_wide_their_search_was():
    """A composite found without a search has paid no multiplicity price and
    the p-value means nothing."""
    w = build_world("boss3_hill_space_cause", 11, snr="high")
    claims = O.scan_boss3(w)
    assert claims
    for c in claims:
        assert c.params["n_programs_searched"] > 50
        assert c.params["n_structural_programs"] > 1
        assert 0.0 <= c.p_value <= 1.0
        assert c.params["p_before_search_correction"] <= c.p_value + 1e-12 or \
            c.params["p_bonferroni"] is not None


def test_boss4_names_which_stage_failed_rather_than_only_pass_or_fail():
    w = build_world("boss4_ode_manifold_anomaly", 11, snr="high")
    rep = O.specialist_boss4(w)
    assert set(rep["stages"]) == {"chart", "regime", "law", "sign_flip"}
    assert rep["recovered"] == all(rep["stages"].values())
    assert rep["pooled_equations_ignoring_regime"]["dx"] != \
        rep["equations_outside_regime"]["dx"], (
        "a pooled fit that ignores the regime should not match the correct "
        "law, or the composition is not doing any work")


def test_boss5_design_uses_the_estimated_threshold_not_the_true_one():
    """Doing the design calculation, rather than reading the answer: if the
    relational stage estimated the wrong tau the design ranking would be
    allowed to be wrong with it."""
    w = build_world("boss5_design_and_threshold", 11, snr="high")
    rep = O.specialist_boss5_design(w)
    assert "tau_estimated" in rep and "tau_truth" in rep
    assert rep["tau_estimated"] != rep["tau_truth"], (
        "the estimate landing exactly on the true value would mean the "
        "specialist read it rather than searched for it")
    assert abs(rep["tau_estimated"] - rep["tau_truth"]) < 0.6
    assert rep["a_star_estimated"] == rep["a_star_truth"]
