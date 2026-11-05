"""Known-answer tests for the controlled relation generators (dbench.benchmark.generators)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from dbench.benchmark.generators import (
    FAMILIES, HELDOUT_FAMILIES, TRAIN_FAMILIES, arity, centre_within, plant, sample_family,
)

N_GROUPS, PER_GROUP = 8, 250


def _data(k, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(N_GROUPS), PER_GROUP)
    V = rng.standard_normal((len(groups), k)) + rng.standard_normal((N_GROUPS, k))[groups]
    return V, groups


def _rff_ridge_r2(Z, y, *, n_features=24, alpha=1.0, seed=123):
    """In-sample R^2 of a ridge on an independent random-Fourier-feature map (+ linear terms)."""
    Z = (Z - Z.mean(0)) / Z.std(0)
    rng = np.random.default_rng(seed)
    om = rng.standard_normal((Z.shape[1], n_features))
    ph = rng.uniform(0, 2 * np.pi, n_features)
    D = np.column_stack([Z, np.cos(Z @ om + ph)])
    D = D - D.mean(0)
    yc = y - y.mean()
    coef = np.linalg.solve(D.T @ D + alpha * np.eye(D.shape[1]), D.T @ yc)
    return 1 - np.sum((yc - D @ coef) ** 2) / np.sum(yc ** 2)


@pytest.mark.parametrize("family", FAMILIES)
def test_centred_within_group_deterministic_and_seed_dependent(family):
    V, groups = _data(arity(family))
    g1, i1 = plant(V, family, np.random.default_rng(5), groups=groups)
    g2, i2 = plant(V, family, np.random.default_rng(5), groups=groups)
    g3, _ = plant(V, family, np.random.default_rng(6), groups=groups)
    assert g1.shape == (len(V),) and np.all(np.isfinite(g1)) and np.var(g1) > 0
    assert np.array_equal(g1, g2) and i1 == i2
    assert not np.allclose(g1, g3)
    for g in np.unique(groups):
        assert abs(g1[groups == g].mean()) < 1e-9
    assert i1["family"] == family and i1["arity"] == arity(family)


@pytest.mark.parametrize("seed", range(4))
def test_residualised_2_has_no_marginal_effects(seed):
    V, groups = _data(2, seed)
    g, info = plant(V, "residualised_2", np.random.default_rng(seed), groups=groups)
    assert info["kept_variance_share"] > 0.01
    for j in range(2):
        assert abs(np.corrcoef(g, V[:, j])[0, 1]) < 1e-6
        assert _rff_ridge_r2(V[:, [j]], g) < 0.01
    assert _rff_ridge_r2(V, g, n_features=200, alpha=1e-3) > 0.2  # the joint structure is still there


@pytest.mark.parametrize("seed", range(3))
def test_residualised_3_has_no_pairwise_effects(seed):
    V, groups = _data(3, seed)
    g, _ = plant(V, "residualised_3", np.random.default_rng(seed), groups=groups)
    for j in range(3):
        assert abs(np.corrcoef(g, V[:, j])[0, 1]) < 1e-6
    for a, b in combinations(range(3), 2):
        assert _rff_ridge_r2(V[:, [a, b]], g) < 0.02


def test_heldout_families_never_sampled_from_training_families():
    rng = np.random.default_rng(0)
    drawn = [sample_family(rng) for _ in range(1000)]
    assert set(drawn) == set(TRAIN_FAMILIES)
    assert not set(drawn) & set(HELDOUT_FAMILIES)
    V, groups = _data(2)
    for _ in range(40):  # the spatial training family never falls back to a held-out generator
        _, info = plant(V, "spatial_mix", rng, groups=groups)
        assert info["base"] in ("smooth_2d", "neural_2")


@pytest.mark.parametrize("family", ("gated_2", "gated_3"))
def test_gated_training_families_are_conditional_and_remove_marginal_shortcuts(family):
    V, groups = _data(arity(family), seed=31)
    g, info = plant(V, family, np.random.default_rng(19), groups=groups)
    assert info["base"] == "random_conditional_continuous"
    assert 0.01 < info["active_share"] < 0.99
    assert info["kept_variance_share"] > 1e-4
    for j in range(V.shape[1]):
        assert abs(np.corrcoef(g, V[:, j])[0, 1]) < 1e-6


def test_arity_and_errors():
    assert [arity(f) for f in ("smooth_1d", "neural_4", "gp_3", "spatial_mix_gp", "spatial_lag")] == [1, 4, 3, 2, 1]
    with pytest.raises(ValueError):
        arity("nonsense")
    V, groups = _data(3)
    with pytest.raises(ValueError):
        plant(V, "gp_2", np.random.default_rng(0), groups=groups)
    with pytest.raises(ValueError):
        plant(V[:, :2], "product_2", np.random.default_rng(0), groups=groups)


def test_tied_values_and_no_groups():
    rng = np.random.default_rng(1)
    x = np.where(rng.random(2000) < 0.6, 0.0, rng.exponential(size=2000))  # many zeros (sparse gene)
    g, _ = plant(x[:, None], "smooth_1d", np.random.default_rng(2))
    assert np.isfinite(g).all() and abs(g.mean()) < 1e-9
    c = centre_within(np.ones((4, 2)), np.array([0, 0, 1, 1]))
    assert np.allclose(c, 0)
