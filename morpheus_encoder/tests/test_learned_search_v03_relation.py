"""Known-answer tests for the v03 relation family, baselines, gains and diagnostics."""
import numpy as np
import pytest
import torch

from morpheus_encoder.learned_search import relation_v03 as rv


def _task(seed=0, n_pat=8, rows=150, p=6, n_sup=6, y_fn=None, known=None):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_pat), rows)
    X = rng.standard_normal((len(g), p))
    y = y_fn(X, rng) if y_fn else X[:, 0] + rng.standard_normal(len(g))
    y = y - np.array([y[g == q].mean() for q in range(n_pat)])[g]
    depth = rng.standard_normal(len(g))
    return rv.Task(X=X, y=y, n_support=n_sup * rows, groups=g, depth=depth, known=known, composition=None)


def _brute_lopo(task, j, alpha):
    s = task.s
    base = rv.baseline_for(task, "nuis")
    V = task.X[:s, [j]]
    Z = torch.as_tensor(((V - V.mean(0)) / V.std(0))[:, None, :])
    bw = rv.median_bandwidth(Z)
    Phi = rv.features(Z, bw)[:, 0].numpy()
    pen = np.ones(Phi.shape[1])
    pen[-1] = 1e-6
    g = task.groups[:s]
    num = den = 0.0
    for p in np.unique(g):
        tr, te = g != p, g == p
        beta = np.linalg.solve(Phi[tr].T @ Phi[tr] + alpha * np.diag(pen), Phi[tr].T @ base.r_s[tr])
        num += np.sum(base.r_cv[te] ** 2) - np.sum((base.r_cv[te] - Phi[te] @ beta) ** 2)
        y = task.y[:s][te]
        den += np.sum((y - y.mean()) ** 2)
    return num / den


def test_lopo_gain_matches_explicit_refits_and_finds_signal():
    task = _task(y_fn=lambda X, r: np.sin(2 * X[:, 0]) + 0.5 * r.standard_normal(len(X)))
    res = rv.gains(task, np.arange(6), "nuis")
    j = int(np.argmax(res["g_sup"]))
    assert j == 0 and res["g_sup"][0] > 0.5 and res["g"][0] > 0.5
    a = float(res["alpha"][0])
    assert res["g_sup"][0] == pytest.approx(_brute_lopo(task, 0, a), abs=1e-6)
    assert np.all(res["g_sup"][1:] < 0.03) and np.all(res["g"][1:] < 0.03)


def test_decomposition_gain_separates_interaction_from_additive():
    inter = _task(1, y_fn=lambda X, r: X[:, 0] * X[:, 1] + 0.3 * r.standard_normal(len(X)))
    add = _task(2, y_fn=lambda X, r: np.sin(2 * X[:, 0]) + np.tanh(2 * X[:, 1]) + 0.3 * r.standard_normal(len(X)))
    di = rv.decomp_gains(inter, [(0, 1)])
    da = rv.decomp_gains(add, [(0, 1)])
    assert di["g"][0] > 0.5 and di["g_sup"][0] > 0.5
    assert abs(da["g"][0]) < 0.03 and abs(da["g_sup"][0]) < 0.03


def test_known_baseline_removes_known_covariates():
    rng = np.random.default_rng(3)
    task = _task(3, y_fn=lambda X, r: X[:, 0] + X[:, 1] + 0.5 * r.standard_normal(len(X)))
    task.known = task.X[:, [0]] + 0.01 * rng.standard_normal((len(task.y), 1))
    nuis = rv.gains(task, np.array([0, 1]), "nuis")
    known = rv.gains(task, np.array([0, 1]), "known")
    assert nuis["g"][0] > 0.2 and abs(known["g"][0]) < 0.02 and known["g"][1] > 0.2


def test_diagnostics_shape_and_signs():
    from morpheus_encoder.learned_search.episodes import knn_neighbours
    task = _task(4)
    task.nb6 = knn_neighbours(np.random.default_rng(0).random((len(task.y), 2)), task.groups, 6)
    task.composition = task.X[:, [3, 4, 5]]
    D, single = rv.diagnostics(task, "nuis")
    c = {n: i for i, n in enumerate(rv.DIAG_NAMES)}
    assert D.shape == (6, len(rv.DIAG_NAMES)) and np.all(np.isfinite(D))
    assert D[0, c["g_sup"]] > 0.3 and D[0, c["sign_agreement"]] == 1.0
    assert D[3, c["r_comp1"]] == pytest.approx(1.0, abs=1e-6) and D[0, c["n_support_patients"]] == 6
    assert D[0, c["within_share"]] == pytest.approx(1.0)


def test_gpu_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    task = _task(5, p=10, y_fn=lambda X, r: X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + r.standard_normal(len(X)))
    a = rv.gains(task, np.arange(10), "nuis")
    b = rv.gains(task, np.arange(10), "nuis", device="cuda")
    np.testing.assert_allclose(a["g_sup"], b["g_sup"], atol=3e-4)
    np.testing.assert_allclose(a["g"], b["g"], atol=3e-4)


def test_label_key_is_order_free_in_patients():
    k1 = rv.label_key("s", ["b", "a"], ["c"], "gene:X", "nuis", "L0")
    assert k1 == rv.label_key("s", ["a", "b"], ["c"], "gene:X", "nuis", "L0")
    assert k1 != rv.label_key("s", ["a", "b"], ["c"], "gene:X", "known", "L0")


def test_known_baseline_never_contains_the_target():
    hm = pytest.importorskip("morpheus.tier2.substrates.hest_multi")
    sets = hm._programme_sets()
    if not sets.get("mp"):
        pytest.skip("programme gene sets not available")
    names = ["mp:Hypoxia", "hallmark:HALLMARK_HYPOXIA", "hallmark:HALLMARK_ADIPOGENESIS", "composition:0"]
    drop = rv.known_exclusions(names, "programme:mp:Hypoxia")
    assert 0 in drop and 3 not in drop
    gene = hm._programme_members("mp:Hypoxia")[0]
    gdrop = rv.known_exclusions(names, f"gene:{gene}")
    assert 0 in gdrop and 3 not in gdrop
    assert rv.known_exclusions(names, "resid:G1") == [] and rv.known_exclusions(names, "morph:k03") == []
    # through baseline_for: a target that is itself a known column keeps a non-trivial B_known residual
    rng = np.random.default_rng(0)
    task = _task(6, y_fn=lambda X, r: X[:, 0] + 0.5 * r.standard_normal(len(X)))
    task.known = np.column_stack([task.y, rng.standard_normal(len(task.y)), rng.standard_normal(len(task.y)),
                                  rng.standard_normal(len(task.y))])
    task.known_names, task.target_name = names, "programme:mp:Hypoxia"
    base = rv.baseline_for(task, "known")
    assert np.var(base.r_s) > 0.5 * np.var(task.y[:task.s])
    K, dropped = rv.known_covariates(task)
    assert "mp:Hypoxia" in dropped and K.shape[1] == 4 - len(dropped)
