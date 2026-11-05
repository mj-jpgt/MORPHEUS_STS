"""Known-answer tests for the v02 support statistics, realised gains and morphology dictionary."""
import numpy as np
import pytest

from morpheus_encoder.learned_search import support_statistics as ss
from morpheus_encoder.learned_search import morphology_dictionary as md


def _world(seed=0, n_pat=6, rows=150, p=6):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_pat), rows)
    X = rng.standard_normal((len(g), p))
    return rng, g, X


def _brute_lopo(V, y, g, alpha, n_rff=ss.N_RFF):
    """Reference: explicit leave-one-patient-out refits of the same design."""
    import torch
    mu, sd = V.mean(0), V.std(0)
    Phi = ss._features(torch.as_tensor(((V - mu) / sd)[:, None, :]), n_rff, False)[:, 0].numpy()
    r, _ = ss.nuisance_residual(y, None)
    pen = np.ones(Phi.shape[1])
    pen[-1] = 1e-6
    num = den = 0.0
    for p in np.unique(g):
        tr, te = g != p, g == p
        A = Phi[tr]
        beta = np.linalg.solve(A.T @ A + alpha * np.diag(pen), A.T @ r[tr])
        num += np.sum(r[te] ** 2) - np.sum((r[te] - Phi[te] @ beta) ** 2)
        den += np.sum((y[te] - y[te].mean()) ** 2)
    return num / den


def test_fit_sets_matches_explicit_lopo_and_finds_the_signal():
    rng, g, X = _world()
    y = np.sin(2 * X[:, 0]) + 0.5 * rng.standard_normal(len(g))
    y -= np.array([y[g == p].mean() for p in g])
    res = ss.fit_sets(X, y, g, alphas=(10.0,))
    assert res["cv_gain"].shape == (6,) and res["lopo_gain"].shape == (6, 6)
    for j in (0, 3):
        assert res["cv_gain"][j] == pytest.approx(_brute_lopo(X[:, [j]], y, g, 10.0), abs=1e-8)
    assert res["cv_gain"][0] > 0.5 and np.all(res["cv_gain"][1:] < 0.02)


def test_pair_carries_joint_information_singles_do_not():
    rng, g, X = _world(1, n_pat=8, rows=200)
    y = X[:, 0] * X[:, 1] + 0.3 * rng.standard_normal(len(g))
    n_s = 6 * 200
    out = ss.realised_gains(X, y, n_s, g, pairs=[(0, 1), (2, 3)])
    assert np.all(out["single"] < 0.05)
    assert out["pair"][0] > 0.4 and out["pair"][1] < 0.05
    sets = ss.support_cv_gains(X[:n_s], y[:n_s], g[:n_s], [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)])
    assert sets[1] > 0.4 and sets[0] < 0.05 and sets[2] > 0.3 and sets[3] > 0.2
    mixed = ss.realised_gains(X, y, n_s, g, singles=False, sets=[(0, 1), (2,), (0, 1, 3)])
    assert mixed["sets"][0] == pytest.approx(out["pair"][0]) and mixed["sets"][2] > 0.3


def test_nuisance_is_removed_before_the_gain():
    rng, g, X = _world(2)
    depth = rng.standard_normal(len(g))
    y = 2 * depth + 0.3 * rng.standard_normal(len(g))
    X[:, 0] = depth + 0.01 * rng.standard_normal(len(g))   # a candidate that is the nuisance
    res = ss.fit_sets(X, y, g, nuis_s=depth)
    assert abs(res["cv_gain"][0]) < 0.01


def test_moran_columns_equals_section_moran_i():
    from morpheus.tier2.substrates.hest_multi import section_moran_i
    from morpheus_encoder.learned_search.episodes import knn_neighbours
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(3), 80)
    coords = rng.random((len(g), 2))
    nb = knn_neighbours(coords, g, 6)
    nb[rng.random(nb.shape) < 0.2] = -1
    Z = rng.standard_normal((len(g), 4))
    Z[:, 1] = coords[:, 0] * 3 + 0.1 * Z[:, 1]
    got = ss.moran_columns(Z, nb, g)
    for j in range(4):
        ref = section_moran_i(Z[:, j], nb, g.astype(str))
        assert got[j] == pytest.approx(np.mean(list(ref.values())), abs=1e-10)
    assert got[1] > 0.5


def test_candidate_statistics_shape_and_values():
    from morpheus_encoder.learned_search.episodes import knn_neighbours
    rng, g, X = _world(4)
    depth = rng.standard_normal(len(g))
    X[:, 2] = depth
    y = X[:, 0] + 0.5 * rng.standard_normal(len(g))
    nb = knn_neighbours(rng.random((len(g), 2)), g, 6)
    S, names = ss.candidate_statistics(X, y, g, nb6=nb, nuisance=depth, composition=X[:, [3, 4]],
                                       transform=np.array([0, 1, 2, 3, 0, 0]), view=np.zeros(6, int))
    assert S.shape == (6, len(ss.STAT_NAMES)) and S.dtype == np.float32 and names == list(ss.STAT_NAMES)
    c = {n: i for i, n in enumerate(names)}
    assert S[0, c["cv_gain"]] > 0.5 and S[0, c["abs_r"]] > 0.7 and S[0, c["within_between_share"]] > 0.9
    assert S[2, c["abs_r_depth"]] == pytest.approx(1.0, abs=1e-6)
    assert S[3, c["r2_composition"]] == pytest.approx(1.0, abs=1e-6) and S[0, c["r2_composition"]] < 0.05
    assert S[1, c["t_nbr6"]] == 1 and S[1, c["t_own"]] == 0
    assert np.all(np.isfinite(S))


def test_morphology_dictionary_kmeans_and_similarities(tmp_path):
    rng = np.random.default_rng(5)
    true = md._unit(rng.standard_normal((4, 32)))
    E = np.concatenate([true[i] + 0.05 * rng.standard_normal((100, 32)) for i in range(4)])
    C, info = md.spherical_kmeans(E, 4, seed=0)
    assert C.shape == (4, 32) and sorted(info["cluster_sizes"]) == [100] * 4
    assert np.all((md._unit(C) @ true.T).max(0) > 0.99)
    sim = md.similarities(E, C)
    assert sim.shape == (400, 4) and np.all(sim <= 1 + 1e-5)
    C2, _ = md.spherical_kmeans(E, 4, seed=0)
    np.testing.assert_array_equal(C, C2)
    sample, counts = md.balanced_sample({"a": E[:300], "b": E[300:]}, per_patient=150)
    assert counts == {"a": 150, "b": 100} and len(sample) == 250


def test_fused_training_labels_equal_the_separate_calls():
    from morpheus_encoder.learned_search.episodes import EpisodeSampler, add_morph_view, synthetic_study
    rng = np.random.default_rng(0)
    st = add_morph_view(synthetic_study(1), rng.standard_normal((8, 64)).astype(np.float32))
    ep = EpisodeSampler([st], concepts=False, morph_target_prob=0.5, seed=1).sample(rng, kind="real")
    pairs = [(0, 5), (3, 9), (10, 40)]
    S, sn, lab = ss.episode_statistics_and_labels(ep, lambda S, sn: pairs)
    S2, _ = ss.episode_statistics(ep)
    lab2 = ss.episode_realised_gains(ep, pairs)
    np.testing.assert_allclose(S, S2, atol=1e-6)
    np.testing.assert_allclose(lab["single"], lab2["single"], atol=1e-10)
    np.testing.assert_allclose(lab["pair"], lab2["pair"], atol=1e-10)
    assert S.shape == (ep.X.shape[1], len(ss.STAT_NAMES))


def test_gpu_float32_fit_matches_cpu_float64():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    rng, g, X = _world(7, n_pat=8, rows=500, p=40)
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.standard_normal(len(g))
    n_s = 6 * 500
    a = ss.realised_gains(X, y, n_s, g, pairs=[(1, 2), (3, 4)])
    b = ss.realised_gains(X, y, n_s, g, pairs=[(1, 2), (3, 4)], device="cuda")
    for k in ("single", "single_cv", "pair", "pair_cv"):
        np.testing.assert_allclose(a[k], b[k], atol=2e-4)
