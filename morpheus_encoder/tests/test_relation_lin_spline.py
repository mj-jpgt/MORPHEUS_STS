"""Relation lin_spline_v01 (patient-level, n ~ 100): v06 two-penalty engine with a deterministic block --
linear term + 3-df natural cubic spline per variable, pairwise products of the linear terms for k >= 2."""
from __future__ import annotations

import numpy as np
import pytest

from morpheus_encoder.general_discovery import (
    VALIDATION_ALPHA_GRID, Candidate, evaluate_frozen_candidate, natural_spline_basis, natural_spline_knots,
)


def _world(effect, *, seed=0, n=110, width=6, k=1):
    """One row per patient (CPTAC-like: 88 proposal / 22 held-out); a comparison block with one
    informative column; hypothesis variables X[:, :k]."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, width))
    X = rng.normal(size=(n, 4))
    y = 0.5 * B[:, 0] + effect(X) + rng.normal(size=n)
    groups = np.arange(n)
    proposal = np.arange(n) < 88
    return X, y, B, groups, proposal


def _eval(X, y, B, groups, proposal, variables=(0,), form="linear_subspace", relation="lin_spline_v01"):
    cand = Candidate(tuple(variables), feature_form=form,
                     weights=(1.0,) * len(variables) if form == "linear_subspace" else ())
    p, r = proposal, ~proposal
    return evaluate_frozen_candidate(cand, X[p], y[p], X[r], y[r], proposal_baseline=B[p], replication_baseline=B[r],
                                     model="rbf", proposal_groups=groups[p], alpha_grid=VALIDATION_ALPHA_GRID,
                                     relation=relation, replication_groups=groups[r])


def test_natural_spline_basis_known_answer():
    x = np.linspace(-3, 3, 601)
    knots = natural_spline_knots(x)
    assert np.allclose(knots, [-3, -1, 1, 3])
    N = natural_spline_basis(x, knots)
    assert N.shape == (601, 2)
    # linear beyond the boundary knots: zero second difference outside [-3, 3] (extrapolated)
    out = np.linspace(3, 6, 31)
    second = np.diff(natural_spline_basis(out, knots), n=2, axis=0)
    assert np.abs(second).max() < 1e-9
    assert np.allclose(natural_spline_basis(np.array([-5.0, -3.0]), knots), 0.0)   # zero left of the first knot
    # a cubic in the span is not linear: the basis carries curvature inside
    assert np.abs(np.diff(N, n=2, axis=0)).max() > 1e-4


def test_planted_linear_effect_recovered_linear_carries_it():
    rec = _eval(*_world(lambda X: 0.6 * X[:, 0]))
    d = rec.decomposition
    assert rec.replication_gain > 0.1 and d["gain_total_ci"][0] > 0
    assert d["gain_linear_ci"][0] > 0 and d["shape"] in ("linear_carries_it", "linear_and_beyond_linear")


def test_planted_smooth_monotone_effect_recovered():
    rec = _eval(*_world(lambda X: np.tanh(2.0 * X[:, 0])))
    assert rec.replication_gain > 0.1 and rec.decomposition["gain_total_ci"][0] > 0


def test_planted_step_is_beyond_linear():
    """A symmetric step (no linear part) at 2 SD of noise: the spline part is resolved above zero in
    each of six worlds. With 22 held-out patients a 1.5 SD step resolves in only 2-3 of 6 (power, not a
    fault: point estimates 0.02-0.23, all positive)."""
    gains = []
    for seed in range(6):
        d = _eval(*_world(lambda X: 2.0 * (np.abs(X[:, 0]) > 1.0), seed=seed)).decomposition
        gains.append(d["gain_nonlinear_beyond_linear"])
        assert d["gain_nonlinear_beyond_linear_ci"][0] > 0
        assert d["shape"] in ("beyond_linear_only", "linear_and_beyond_linear")
    assert np.median(gains) > 0.1


def test_planted_product_k2_beyond_additive():
    rec = _eval(*_world(lambda X: 1.0 * X[:, 0] * X[:, 1], k=2), variables=(0, 1), form="product")
    assert rec.replication_gain > 0.1
    assert rec.decomposition["gain_linear_additive"] < 0.05     # a product has little additive linear part


def test_permuted_block_median_gain_is_null():
    gains = []
    for seed in range(20):
        X, y, B, g, p = _world(lambda X: 0.0 * X[:, 0], seed=seed)
        gains.append(_eval(X, y, B, g, p).replication_gain)
    assert np.median(gains) <= 0.001


def test_deterministic_and_receipt_records_penalties():
    w = _world(lambda X: 0.6 * X[:, 0])
    a, b = _eval(*w), _eval(*w)
    assert a.replication_gain == b.replication_gain
    assert {"alpha_comparison", "alpha_pair_full", "alpha_pair_linear"} <= set(a.decomposition)
    assert a.decomposition["relation"] == "lin_spline_v01"
