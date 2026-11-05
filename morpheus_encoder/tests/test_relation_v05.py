"""v0.5 single-variable relation fix (flag ``relation="v05"``; default ``"v04"`` unchanged)."""
from __future__ import annotations

import numpy as np
import pytest

from morpheus_encoder.general_discovery import (
    VALIDATION_ALPHA_GRID, Candidate, FittedRelation, evaluate_frozen_candidate, evaluate_structured_nulls,
)


def _world(effect: str, *, n_patients: int = 16, per: int = 120, strength: float = 1.0, seed: int = 0):
    """Baseline (2 cols) + one planted variable x + 3 noise columns; patient groups."""
    rng = np.random.default_rng(seed)
    n = n_patients * per
    groups = np.repeat(np.arange(n_patients), per)
    B = rng.normal(size=(n, 2))
    X = rng.normal(size=(n, 4))
    x = X[:, 0]
    signal = {"linear": x, "sine": np.sin(2.5 * x), "none": np.zeros(n)}[effect]
    y = 0.8 * B[:, 0] + strength * signal + rng.normal(size=n)
    proposal = groups < n_patients * 3 // 4
    return X, y, B, groups, proposal, signal


def _eval(X, y, B, groups, proposal, *, variables=(0,), relation="v04", form="linear_subspace"):
    cand = Candidate(tuple(variables), feature_form=form,
                     weights=(1.0,) * len(variables) if form == "linear_subspace" else ())
    p, r = proposal, ~proposal
    return evaluate_frozen_candidate(cand, X[p], y[p], X[r], y[r], proposal_baseline=B[p],
                                     replication_baseline=B[r], model="rbf", proposal_groups=groups[p],
                                     alpha_grid=VALIDATION_ALPHA_GRID, relation=relation,
                                     replication_groups=groups[r])


# Pinned with the pre-change module (HEAD 11ff360) on _world("sine", seed=3):
# the default path must reproduce the v0.3/v0.4 numbers exactly.
PINNED_V04 = {"replication_gain": 0.2413902636481322, "replication_r2_simple": 0.28491258924806995,
              "replication_r2_full": 0.5263028528962022, "alpha_simple": 10.0, "alpha_full": 10.0}


def test_flag_off_reproduces_pinned_v04_numbers():
    X, y, B, groups, proposal, _ = _world("sine", seed=3)
    rec = _eval(X, y, B, groups, proposal)
    assert rec.relation == "v04" and rec.decomposition is None
    for key, value in PINNED_V04.items():
        assert getattr(rec, key) == pytest.approx(value, abs=1e-9, rel=0), key


def test_planted_linear_variable_total_gain_with_flag_and_near_zero_without():
    X, y, B, groups, proposal, signal = _world("linear", seed=1)
    r = ~proposal
    resid = y[r] - 0.8 * B[r, 0]
    share = np.var(signal[r]) / np.var(y[r])  # variance share of the planted term
    off = _eval(X, y, B, groups, proposal, relation="v04")
    on = _eval(X, y, B, groups, proposal, relation="v05")
    assert abs(off.replication_gain) < 0.01
    assert on.replication_gain == pytest.approx(share, abs=0.04)
    d = on.decomposition
    assert d["gain_linear"] == pytest.approx(share, abs=0.04)
    assert abs(d["gain_nonlinear_beyond_linear"]) < 0.01
    assert d["gain_total"] == pytest.approx(d["gain_linear"] + d["gain_nonlinear_beyond_linear"], abs=1e-12)
    assert d["gain_total"] == pytest.approx(on.replication_gain, abs=1e-12)
    assert d["gain_linear_ci"][0] > 0 and d["shape"] == "linear_carries_it"
    assert np.var(resid) > 0


def test_planted_sinusoid_has_gain_in_both_versions_and_beyond_linear_equals_v04():
    X, y, B, groups, proposal, _ = _world("sine", seed=2)
    off = _eval(X, y, B, groups, proposal, relation="v04")
    on = _eval(X, y, B, groups, proposal, relation="v05")
    assert off.replication_gain > 0.05 and on.replication_gain > 0.05
    d = on.decomposition
    # The v0.5 "linear" model is the v0.4 competitor (same design, seed, grid), so
    # the beyond-linear part is the registered v0.4 gain exactly.
    assert d["gain_nonlinear_beyond_linear"] == pytest.approx(off.replication_gain, abs=1e-12)
    assert d["alpha_linear"] == off.alpha_simple
    assert d["gain_nonlinear_beyond_linear_ci"][0] > 0
    assert d["shape"] in ("beyond_linear_only", "linear_and_beyond_linear")


def test_flag_on_full_model_unchanged_and_pairs_keep_their_comparison():
    X, y, B, groups, proposal, _ = _world("sine", seed=4)
    off = _eval(X, y, B, groups, proposal, variables=(0, 1), form="product")
    on = _eval(X, y, B, groups, proposal, variables=(0, 1), form="product", relation="v05")
    assert on.replication_gain == off.replication_gain
    assert (on.alpha_simple, on.alpha_full) == (off.alpha_simple, off.alpha_full)
    d = on.decomposition
    assert set(d) >= {"gain_linear_additive", "gain_linear_additive_ci"} and "gain_total" not in d


def test_flag_on_nulls_work_and_reject_a_pure_noise_variable():
    for effect, expect_small_p in (("linear", True), ("none", False)):
        X, y, B, groups, proposal, _ = _world(effect, seed=5, strength=0.5)
        rec = _eval(X, y, B, groups, proposal, relation="v05")
        p, r = proposal, ~proposal
        rng_cache: dict = {}

        def perm(index, _X, _y, _c, X_r=X[r], y_r=y[r], g=groups[r]):
            if index not in rng_cache:
                rng = np.random.default_rng(index)
                out = y_r.copy()
                for k in np.unique(g):
                    idx = np.flatnonzero(g == k)
                    out[idx] = y_r[idx[rng.permutation(len(idx))]]
                rng_cache[index] = out
            return X_r, rng_cache[index], None

        nulled = evaluate_structured_nulls(rec, X[p], y[p], proposal_baseline=B[p], replication_baseline=B[r],
                                           null_generator=perm, n_null=99)
        assert nulled.relation == "v05" and len(nulled.null_gains) == 99
        if expect_small_p:
            assert nulled.null_p_value <= 0.02
        else:
            assert nulled.null_p_value > 0.05 or rec.replication_gain < 0.005


def test_null_refit_without_cache_uses_the_receipt_relation():
    from morpheus_encoder import general_discovery as gd

    X, y, B, groups, proposal, _ = _world("linear", seed=6)
    rec = _eval(X, y, B, groups, proposal, relation="v05")
    gd._FITTED_CACHE.clear()
    p, r = proposal, ~proposal
    same = evaluate_structured_nulls(rec, X[p], y[p], proposal_baseline=B[p], replication_baseline=B[r],
                                     null_generator=lambda i, a, b, c: (X[r], y[r], None), n_null=3)
    assert all(g == pytest.approx(rec.replication_gain, abs=1e-12) for g in same.null_gains)


def test_empty_reference_and_bad_relation():
    X, y, _, groups, proposal, _ = _world("linear", seed=7)
    cand = Candidate((0,), feature_form="linear_subspace", weights=(1.0,))
    p, r = proposal, ~proposal
    rec = evaluate_frozen_candidate(cand, X[p], y[p], X[r], y[r], model="rbf", proposal_groups=groups[p],
                                    alpha_grid=VALIDATION_ALPHA_GRID, relation="v05")
    assert rec.replication_r2_simple == pytest.approx(0.0, abs=0.02) and rec.replication_gain > 0.2
    with pytest.raises(ValueError):
        FittedRelation(cand, X[p], y[p], context=None, baseline=None, model="rbf", ridge_alpha=1.0,
                       rbf_features=24, relation="v99")
