"""Test A for the Tier II HEST spatial substrate (`make_hest_spatial`)."""
import numpy as np

from dbench import spike_in as S
from dbench.util import morans_i


def _world(seed=0, role="proposal", **kw):
    return S.make_hest_spatial(seed=seed, role=role, **kw)


def test_truth_exists():
    w = _world()
    true_objs = w.truth.true_objects
    art_objs = w.truth.artifacts
    assert len(true_objs) == 1 and true_objs[0].operator == "SPACE"
    assert len(art_objs) == 1 and art_objs[0].artifact_class == "spatially_smooth_decoy"
    assert np.isfinite(true_objs[0].effect) and abs(true_objs[0].effect) > 0
    assert np.isfinite(art_objs[0].effect)
    assert w.domain == "spatial"
    assert w.meta["real_substrate"] is True
    assert w.structure.coords is not None and w.structure.neighbors is not None
    # the decoy really is genuinely, strongly spatially autocorrelated on the
    # real slide -- not a strawman
    c_name = next(k for k in w.latents if k.startswith("C_"))
    mi = morans_i(w.latents[c_name], w.structure.neighbors)
    assert mi > 0.2, ("the artifact gene must be genuinely spatially "
                      "autocorrelated on the real tissue; measured I=%.3f" % mi)


def test_null_is_valid():
    """`hest_iid_permute_null` is a full permutation of Y, adopted after three
    geometry-preserving nulls (translation, rotation, spatial-cluster shuffle)
    were swept over 15 seeds each and found miscalibrated on this specific
    real, small (1084-spot) sample -- documented in the function's own
    docstring. Full permutation is exactly calibrated under exchangeability
    by construction; what it does NOT claim is that Y's own autocorrelation
    survives, only that every real gene field does."""
    w = _world(n=None)
    nw, rep = S.hest_iid_permute_null(w, seed=17)
    assert rep["passes"], rep
    assert all(not h.true for h in nw.truth.objects)
    # real gene fields are completely untouched by the null -- only Y moves
    for k in w.latents:
        assert np.array_equal(w.latents[k], nw.latents[k])
    cov_rep = S.covariance_preservation_check(w, nw)
    assert cov_rep["identical_arrays"], cov_rep
    # the true within-neighbourhood pairing is actually broken
    from dbench import oracle as O
    aggB = w.latents["aggB"]
    ev_before = O.partial_evidence(w.target, aggB, [])
    ev_after = O.partial_evidence(nw.target, aggB, [])
    assert ev_before["p"] < 1e-6
    assert ev_after["p"] > 0.01, (
        "the null must destroy the true SPACE association; got p=%.4g"
        % ev_after["p"])


def test_positive_control_recovers_space_object():
    """A plain partial-F test of Y against the true neighbourhood aggregate,
    handed the oracle latent (as the family's own positive control, mirroring
    every other Tier I/II family's `specialist_*` convention)."""
    from dbench import oracle as O
    w = _world(snr="high")
    aggB = w.latents["aggB"]
    ev = O.partial_evidence(w.target, aggB, [])
    assert ev["p"] < 1e-6 and abs(ev["effect"]) > 0.3, ev
    # and the decoy, while genuinely spatially smooth, is not what drives Y
    c_name = next(k for k in w.latents if k.startswith("C_"))
    ev_c = O.partial_evidence(w.target, w.latents[c_name], [])
    assert abs(ev_c["effect"]) < 0.3 * abs(ev["effect"]), (
        "the decoy must not rival the true object's effect size", ev, ev_c)


def test_score_can_vary_with_effect_strength():
    from dbench import oracle as O
    w = _world(snr="high")
    aggB = w.latents["aggB"]
    base_noise_component = w.target - 1.4 * 1.6 * (aggB - aggB.mean()) / (
        aggB.std() or 1.0)
    ps = []
    for beta_scale in (0.0, 0.2, 0.5, 1.0, 2.0):
        Y = beta_scale * (aggB - aggB.mean()) / (aggB.std() or 1.0) \
            + base_noise_component
        ev = O.partial_evidence(Y, aggB, [])
        ps.append(ev["p"])
    assert ps[0] > 0.05, "beta=0 must NOT be significant; got p=%.4g" % ps[0]
    assert ps[-1] < 1e-6, "beta=2x must be clearly significant; got p=%.4g" % ps[-1]
    assert len(set(round(p, 12) for p in ps)) > 1, (
        "p-value must move across the sweep, not saturate: %r" % ps)
