"""Test A (LOCKIN minimal test stack) for the Tier II molecular substrate.

Four checks, per new-family policy ("truth exists, null valid, positive
control works, score can vary"):

  1. truth exists       -- the world declares exactly one true CAUSE and one
                           false CONFOUND artifact, with real effect sizes.
  2. null is valid       -- `null_stratum_shuffle` keyed on the real Site
                           label breaks the true within-site pairing while
                           leaving the real background covariance matrix
                           (`spike_in.covariance_preservation_check`) and
                           every real per-Site mean untouched.
  3. positive control    -- `specialist_molecular_confound` recovers the true
                           object and refuses the confound at snr="high".
  4. score can vary      -- weakening the planted effect (beta) moves the
                           recovered evidence away from saturation; this is
                           not a monotone 0/0 or 1/1 result (see MEMORY:
                           "Monotonous results = bug hypothesis first").
"""
import numpy as np
import pytest

from dbench import nulls as N
from dbench import spike_in as S


def _world(seed=0, role="proposal", **kw):
    return S.make_molecular_confound(seed=seed, role=role, **kw)


def test_truth_exists():
    w = _world()
    true_objs = w.truth.true_objects
    art_objs = w.truth.artifacts
    assert len(true_objs) == 1 and true_objs[0].operator == "CAUSE"
    assert len(art_objs) == 1 and art_objs[0].operator == "CONFOUND"
    assert art_objs[0].artifact_class == "nuisance_confound"
    assert art_objs[0].parents == (("Site",),)
    # real effect sizes, not placeholders
    assert np.isfinite(true_objs[0].effect) and abs(true_objs[0].effect) > 0
    assert np.isfinite(art_objs[0].effect) and abs(art_objs[0].effect) > 0
    # the declared substrate really is real data, not a synthetic stand-in
    assert w.domain == "molecular"
    assert w.meta["real_substrate"] is True
    assert w.env.labels["Site"] is not None and len(w.env.labels["Site"]) == w.n


def test_null_is_valid():
    w = _world(n=2000)
    nw, rep = N.make_null(w, "stratum_shuffle", keys=("Site",), seed=17)
    assert rep["passes"], rep
    # every real per-Site mean of Y is exactly preserved
    assert rep["preserved"]["max_abs_stratum_mean_change"] < 1e-9
    # every object in the null world is false by construction
    assert all(not h.true for h in nw.truth.objects)
    # the real background covariance matrix is untouched: only Y moved
    cov_rep = S.covariance_preservation_check(w, nw)
    assert cov_rep["passes"], cov_rep
    assert cov_rep["identical_arrays"], (
        "stratum_shuffle must not touch any real feature column, only Y")
    # the true within-site pairing is actually broken (not a no-op null)
    from dbench import oracle as O
    a_name = next(k for k in w.latents if k.startswith("A_"))
    ev_before = O.partial_evidence(w.target, w.latents[a_name], [])
    ev_after = O.partial_evidence(nw.target, nw.latents[a_name], [])
    assert ev_before["p"] < 1e-6
    assert ev_after["p"] > 0.01, (
        "the null must destroy the true CAUSE association; got p=%.4g"
        % ev_after["p"])


def test_positive_control_recovers_the_true_object_and_refuses_the_confound():
    w = _world(n=3000, snr="high")
    rec = S.specialist_molecular_confound(w)
    assert rec["recovered"], rec


def test_score_can_vary_with_effect_strength():
    """A monotone 0/0 or 1/1 across the sweep would be the bug-hypothesis
    trigger (MEMORY: monotonous-results-check-for-bugs). This family's score
    must move."""
    from dbench import oracle as O
    ps = []
    for beta_scale in (0.0, 0.15, 0.4, 1.0, 2.5):
        w = S.make_molecular_confound(seed=3, role="proposal", n=2500,
                                      snr="high")
        # rebuild Y at a swept beta directly from the same real substrate
        # draw, holding the confound term fixed, so only the true effect
        # strength moves.
        a_name = next(k for k in w.latents if k.startswith("A_"))
        A = w.latents[a_name]
        rq = np.random.default_rng(3)
        base_shift = w.target - (1.3 * 1.6 * A)  # peel off the default beta*A
        Y = beta_scale * A + base_shift
        site_dummies = S._one_hot_drop_first(w.env.labels["Site"])
        ev = O.partial_evidence(Y, A, site_dummies)
        ps.append(ev["p"])
    assert ps[0] > 0.05, ("beta=0 must NOT be significant; got p=%.4g"
                          % ps[0])
    assert ps[-1] < 1e-6, ("beta=2.5x must be clearly significant; got "
                           "p=%.4g" % ps[-1])
    assert len(set(round(p, 12) for p in ps)) > 1, (
        "the p-value must move across the effect sweep, not sit at one "
        "saturated value: %r" % ps)
