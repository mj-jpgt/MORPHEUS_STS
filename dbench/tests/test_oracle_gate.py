"""The oracle gate: structure of the specialists, and recovery on cheap families.

The full gate over all twenty-three worlds takes tens of minutes and is run as a
freeze artefact (`dbench/gate_report.json`), not inside the test suite. What is
asserted here is everything that can go silently wrong without changing a
runtime: that a specialist searches instead of reading the answer, that its
p-values do not depend on the interpreter's hash seed, that the max-statistic
correction does what it claims, and that recovery holds on the families cheap
enough to re-derive every run.
"""
import numpy as np
import pytest

from dbench import oracle as O
from dbench.environments import ALL_WORLDS, build_world
from dbench.metrics import Claim, match_claims
from dbench.util import benjamini_hochberg

FAST = ("additive_factors", "pure_and", "xor", "threshold", "hill",
        "context_gated", "regime_switch", "graph_motif",
        "intervention_causal")
NONREL_FAST = ("manifold", "anomaly", "mixture", "hidden_symmetry",
               "sparse_ode", "experimental_design", "latent_confound")


# --------------------------------------------------------------------------
# every family has a specialist, and the gate covers every world
# --------------------------------------------------------------------------
def test_every_registered_world_has_a_specialist():
    """A world with no specialist reports `recovered=None`, which is not a
    result: the gate would silently exclude it."""
    missing = []
    for fam in sorted(ALL_WORLDS):
        has = (fam in O.NONRELATIONAL or fam in O.BOSS_RELATIONAL
               or fam == "boss5_design_and_threshold"
               or bool(O.relational_claims(build_world(fam, 0), fam)))
        if not has:
            missing.append(fam)
    assert not missing, missing


# --------------------------------------------------------------------------
# the specialists search
# --------------------------------------------------------------------------
def test_scans_never_see_operator_derived_latents():
    """`aggB`, `B_lag3`, `thrA` are an operator already applied at the true
    parameter. If a scan could pick them up, recovery would be a lookup."""
    for fam in ("spatial_interaction", "temporal_lag",
                "boss1_cond_thresh_space_lag", "boss3_hill_space_cause"):
        w = build_world(fam, 0)
        derived = set(w.meta.get("derived_latents", ()) or ())
        assert derived
        assert not (set(O._scan_names(w)) & derived), fam
        assert set(O._scan_names(w, include_derived=True)) & derived


def test_the_threshold_specialist_finds_tau_rather_than_being_given_it():
    w = build_world("threshold", 0)
    claims = O.scan_threshold(w)
    hit = [c for c in claims if c.operands == ("A",)]
    assert hit
    tau_hat = hit[0].params["tau"]
    assert abs(tau_hat - w.meta["tau"]) < 0.5
    assert hit[0].params["tau_grid_size"] > 1, (
        "a threshold found without a grid has paid no multiplicity price")


def test_the_lag_specialist_recovers_the_lag_and_reports_it():
    """The whole lag grid is searched, so adjacent lags of an autocorrelated
    series are flagged too; what identifies the object is that the true lag
    carries the largest partial effect of any lag on any variable."""
    w = build_world("temporal_lag", 0)
    claims = O.scan_lags(w)
    p = np.array([c.p_value for c in claims])
    rej, _ = benjamini_hochberg(p, 0.05)
    flagged = [c for c, r in zip(claims, rej) if r]
    assert any(c.operands == ("B",) and c.lag == w.meta["lag"] for c in flagged)
    best = max(claims, key=lambda c: abs(c.effect))
    assert best.operands == ("B",) and best.lag == w.meta["lag"], (
        best.id, best.effect)


def test_the_gated_specialist_enumerates_levels_and_not_a_median_split():
    w = build_world("context_gated", 0)
    claims = O.scan_gated(w)
    assert claims
    assert all("gate_level" in c.params for c in claims)
    assert all(abs(c.effect) > 0 for c in claims), (
        "a design column that is constant returns effect exactly zero for "
        "every candidate; that was the median-split failure")


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def test_specialist_seeds_do_not_depend_on_the_interpreter_hash_seed():
    """Python randomises `hash()` of strings per process. A benchmark whose
    frozen p-values move between runs cannot be replicated by anyone."""
    import subprocess
    import sys
    code = (
        "import warnings;warnings.filterwarnings('ignore');"
        "import dbench.oracle as O;"
        "from dbench.environments import build_world;"
        "w=build_world('temporal_lag',0);"
        "print(round(sum(c.p_value for c in O.scan_lags(w,n_perm=49)),9))")
    outs = set()
    for hs in ("0", "1", "12345"):
        import os
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env)
        assert r.returncode == 0, r.stderr[-800:]
        outs.add(r.stdout.strip())
    assert len(outs) == 1, (
        "specialist p-values differ across PYTHONHASHSEED values: %s" % outs)


# --------------------------------------------------------------------------
# the max-statistic correction
# --------------------------------------------------------------------------
def test_maxT_is_calibrated_on_a_grid_of_pure_noise():
    """Every program in the grid is noise, so the family-wise p-values must be
    roughly uniform and must reject at about alpha, not far above it."""
    from dbench.environments import build_world as bw
    w = bw("additive_factors", 0)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(w.n)
    programs = []
    for i in range(20):
        f = rng.standard_normal(w.n)
        programs.append({"terms": [{"op": "ADD", "operands": ["D0"]}],
                         "feature": f})
    # patch the parent family to empty so the test is about the null, not
    # about the parents
    import dbench.features as F
    old = F.composite_parent_columns
    F.composite_parent_columns = lambda world, terms, **kw: ([], [])
    O.composite_parent_columns = F.composite_parent_columns
    try:
        p, info = O.maxT_permutation(w, y, programs, n_perm=499, seed=1)
    finally:
        F.composite_parent_columns = old
        O.composite_parent_columns = old
    p = np.asarray(p)
    assert info["n_programs"] == 20 and info["n_perm"] == 499
    assert (p <= 0.05).mean() <= 0.15, (
        "family-wise rejection rate %.3f on a pure-noise grid" % (p <= 0.05).mean())
    assert p.min() >= 1.0 / 500


def test_maxT_resolution_does_not_degrade_with_grid_width():
    """The property Bonferroni lacks: adding programs must not raise the floor
    of the achievable p-value."""
    w = build_world("pure_and", 0)
    y = np.asarray(w.target, float)
    a = np.asarray(w.latents["A0"], float) * np.asarray(w.latents["B0"], float)
    def grid(k):
        rng = np.random.default_rng(0)
        progs = [{"terms": [{"op": "AND", "operands": ["A0", "B0"]}],
                  "feature": a}]
        for _ in range(k):
            progs.append({"terms": [{"op": "ADD", "operands": ["D0"]}],
                          "feature": rng.standard_normal(w.n)})
        return progs
    p_small, _ = O.maxT_permutation(w, y, grid(5), n_perm=199, seed=0)
    p_big, _ = O.maxT_permutation(w, y, grid(200), n_perm=199, seed=0)
    assert p_small[0] == p_big[0] == 1.0 / 200, (p_small[0], p_big[0])


# --------------------------------------------------------------------------
# recovery on the cheap families
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAST)
def test_high_snr_recovery_on_the_cheap_relational_families(family):
    w = build_world(family, 11, snr="high")
    rep = O.run_family(w, family)
    assert rep["recovered"] is True, {
        "family": family, "recall": rep.get("recall"),
        "n_flagged": rep.get("n_flagged"),
        "n_true_in_scope": rep.get("n_true_objects_in_scope")}
    assert rep["recall"] == 1.0


@pytest.mark.parametrize("family", NONREL_FAST)
def test_high_snr_recovery_for_the_non_relational_specialists(family):
    w = build_world(family, 11, snr="high")
    rep = O.run_family(w, family)
    assert rep["recovered"] is True, {"family": family, "report": rep}
    assert rep["criterion"]


def test_the_manifold_family_is_genuinely_curved():
    """The spectral specialist must be doing work a centroid method cannot: if
    k-means matched it, the family would not be testing manifold discovery."""
    w = build_world("manifold", 11, snr="high")
    rep = O.specialist_manifold(w)
    assert rep["branch_ari"] > 0.5
    assert rep["branch_ari"] - rep["branch_ari_kmeans"] > 0.3, rep


def test_a_specialist_reports_failure_on_a_world_where_nothing_is_planted():
    """A gate that could only ever return `recovered=True` is not evidence."""
    import dbench.nulls as N
    w = build_world("threshold", 11, snr="high")
    nw, _ = N.make_null(w, "iid_permute", seed=3)
    claims = O.scan_threshold(nw)
    p = np.array([c.p_value for c in claims])
    rej, _ = benjamini_hochberg(p, 0.05)
    assert rej.sum() == 0


def test_null_control_reports_a_rate_with_its_own_power():
    w = build_world("threshold", 11, snr="high")
    rep = O.run_null_control(w, "threshold", n_reps=4)
    assert rep["n_reps"] == 4
    assert 0.0 <= rep["familywise_error_rate"] <= 1.0
    assert rep["criterion"].startswith("P(Bin")
    assert len(rep["per_replicate"]) == 4
    seeds = [r["seed"] for r in rep["per_replicate"]]
    assert len(set(seeds)) == 4, "the replicates must be independent draws"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def test_matching_is_on_operands_not_on_a_discovery_type_label():
    """A system that says 'this is a relational discovery' without naming the
    operands has recovered nothing."""
    w = build_world("pure_and", 0)
    truth = w.truth
    right = Claim(id="c", operator="AND", operands=("A0", "B0"), target="Y")
    wrong = Claim(id="c", operator="AND", operands=("D0", "D1"), target="Y")
    assert any(d["true"] for d in match_claims([right], truth)["matched"])
    assert not match_claims([wrong], truth)["matched"]


def test_matching_ignores_operand_order():
    w = build_world("pure_and", 0)
    a = Claim(id="a", operator="AND", operands=("A0", "B0"), target="Y")
    b = Claim(id="b", operator="AND", operands=("B0", "A0"), target="Y")
    ma = match_claims([a], w.truth)["matched"]
    mb = match_claims([b], w.truth)["matched"]
    assert [d["object_id"] for d in ma] == [d["object_id"] for d in mb]


# --------------------------------------------------------------------------
# grid-search tie-break (the underflow defect)
# --------------------------------------------------------------------------
# `scan_lincomb`, `scan_threshold` and `scan_saturation` used to keep a
# running "best so far" and replace it only on a strict `p <` improvement.
# Once a grid is wide or the signal strong enough that many points underflow
# `stats.f.sf` to exactly 0.0, that rule stops firing after the first point of
# a tied run and hands the claim to whichever grid point the loop reached
# first -- an artefact of iteration order. Measured on `linear_combination`
# seed 11: 71 of 127 weight points underflowed and the old rule's winner
# (weights 0.4375/0.5625) was 30 grid steps from the true optimum
# (0.75/0.25, the planted weight); `_grid_argbest` (ranking by F, which does
# not underflow) recovers the planted weight exactly.
def _ev(F):
    return {"F": F, "p": 0.0, "effect": 0.0, "s": 0.0, "beta": 0.0}


def test_grid_argbest_does_not_return_the_first_underflowing_point():
    """All ten points here would report `p == 0.0` in float64 (F this large
    always saturates `stats.f.sf`); the true maximum is planted in the
    middle of the run, at index 6, not at the first tied point (index 0)."""
    Fs = [1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e30, 1e18, 1e19, 1e14]
    evs = [_ev(F) for F in Fs]
    anchors = list(range(len(Fs)))
    idx = O._grid_argbest(evs, anchors)
    assert idx == 6, (
        "the strict-p< rule would stop at the first point whose (underflowed) "
        "p ties the running minimum, which is not the true argmax")


def test_grid_argbest_finds_the_true_maximum_under_any_grid_ordering():
    """A grid with one true, unambiguous maximum must return it regardless of
    how the grid is ordered -- reversing or shuffling the scan must not move
    the winner, because the winner is a property of the data, not the loop."""
    rng = np.random.default_rng(0)
    base_F = rng.uniform(1.0, 50.0, size=40)
    true_idx = 17
    base_F[true_idx] = 999.0          # unambiguous single maximum
    anchors = list(range(len(base_F)))
    order_forward = list(range(len(base_F)))
    order_reverse = order_forward[::-1]
    order_shuffled = list(order_forward)
    rng.shuffle(order_shuffled)
    for order in (order_forward, order_reverse, order_shuffled):
        evs = [_ev(base_F[i]) for i in order]
        anc = [anchors[i] for i in order]
        idx = O._grid_argbest(evs, anc)
        assert anc[idx] == true_idx, (order, anc[idx])


def test_grid_argbest_residual_tie_is_broken_by_value_not_position():
    """When F is exactly tied, the declared rule (closest to the anchor mean,
    then smallest anchor) must pick the same point whichever order the tied
    points are visited in -- a position-based tie-break (first one wins)
    would fail this under reversal."""
    Fs = [5.0, 5.0, 5.0]
    anchors = [10.0, 1.0, 4.0]           # mean = 5.0 -> closest is 4.0 (idx 2)
    evs = [_ev(F) for F in Fs]
    idx_fwd = O._grid_argbest(evs, anchors)
    idx_rev = O._grid_argbest(evs[::-1], anchors[::-1])
    assert anchors[idx_fwd] == 4.0
    assert anchors[::-1][idx_rev] == 4.0


def test_scan_lincomb_recovers_the_planted_weight_exactly_after_the_fix():
    """Regression pin for the defect itself: on `linear_combination` seed 11,
    the LINCOMB specialist's winning weight must equal the planted weight
    (0.75, 0.25) exactly, and must match the truth object (similarity ~= 1),
    not merely flag *a* combination."""
    w = build_world("linear_combination", 11, role="proposal", snr="high")
    claims = O.scan_lincomb(w)
    p = np.array([c.p_value for c in claims])
    rej, _ = benjamini_hochberg(p, alpha=O.ALPHA)
    flagged = [c for c, r in zip(claims, rej) if r]
    win = [c for c in flagged if c.id == "LINCOMB:A,B"]
    assert win, "the true pair must survive BH"
    wa, wb = win[0].params["weights"]
    assert abs(wa - 0.75) < 1e-9 and abs(wb - 0.25) < 1e-9
    m = match_claims(flagged, w.truth)
    matched = m.get("matched") or []
    assert any(d["true"] and d["similarity"] > 0.999 for d in matched), (
        "the pre-fix tie-break reported a different, unmatched weight here")
