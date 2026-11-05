"""Residual-bank generator: head independence, cross-fitted residuals, planted recovery, config round trip."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from morpheus_encoder.hypothesis_generator import GeneratorConfig
from morpheus_encoder.residual_bank_generator import (
    LEVELS, ResidualBankConfig, ResidualBankNetwork, comparison_levels, generate_residual_bank_hypotheses,
    multitask_loss, residual_bank,
)

COMPARISON_NAMES = ["depth_log1p", "cv_frac_A", "cv_frac_B", "mp:p1", "hallmark:h1", "gavish:g1",
                    "nbr:mp:p1", "nbr18:hallmark:h1"]


def synthetic_bank(seed: int, n_patients: int = 10, rows: int = 150, n_genes: int = 40, n_fm: int = 20):
    rng = np.random.default_rng(seed)
    n = n_patients * rows
    groups = np.repeat(np.arange(n_patients), rows)
    X = rng.standard_normal((n, n_genes + n_fm))
    B = rng.standard_normal((n, len(COMPARISON_NAMES)))
    y = B @ rng.normal(0, 0.5, B.shape[1]) + rng.standard_normal(n)
    for g in range(n_patients):
        m = groups == g
        X[m] -= X[m].mean(0)
        B[m] -= B[m].mean(0)
        y[m] = (y[m] - y[m].mean()) / y[m].std()
    views = {"raw_genes": np.arange(n_genes), "scgpt": np.arange(n_genes, n_genes + n_fm)}
    return X, y, B, groups, views


def test_levels_follow_agent_d_groups():
    lv = comparison_levels(COMPARISON_NAMES)
    assert [list(lv[k]) for k in LEVELS] == [[0], [0, 1, 2], [0, 1, 2, 3, 4, 5], list(range(8))]


def test_heads_independent_given_shared_towers():
    torch.manual_seed(0)
    net = ResidualBankNetwork([range(0, 6), range(6, 10)], 4, hidden=16, rank=4).eval()
    with torch.no_grad():  # move every output layer off its zero initialisation
        for layer in [*net.main, *net.linear, net.higher[-1]]:
            layer.weight.normal_(0, 0.3)
            layer.bias.normal_(0, 0.3)
        for q in net.Q:
            q.normal_(0, 0.3)
    x, r = torch.randn(64, 10), torch.randn(64, 4)

    def grads(weights):
        net.zero_grad()
        multitask_loss(net(x), r, torch.tensor(weights)).backward()
        return [[p.grad[idx].clone() for p, idx in _slices(net, k)] for k in range(4)], \
            [t.grad.clone() for t in net.towers[0].parameters()]

    full, tower_full = grads([0.25, 0.25, 0.25, 0.25])
    zeroed, tower_zeroed = grads([0.25, 0.25, 0.0, 0.25])
    for k in (0, 1, 3):
        for a, b in zip(full[k], zeroed[k]):
            assert torch.equal(a, b)
    assert all(torch.count_nonzero(g) == 0 for g in zeroed[2])
    assert any(torch.count_nonzero(g) for g in full[2])
    # the shared towers do feel head 2's loss
    assert any(not torch.equal(a, b) for a, b in zip(tower_full, tower_zeroed))
    # default weights are the ``_train`` objective (mean squared error over all heads)
    pred = net(x)
    assert torch.allclose(multitask_loss(pred, r), ((pred - r) ** 2).mean())


def _slices(net, k):
    out = [(net.mix, (slice(None), slice(None), k))]
    for layer in [*net.main, *net.linear, net.higher[-1]]:
        out += [(layer.weight, k), (layer.bias, k)]
    return out


def test_head_parameters_are_the_slices():
    net = ResidualBankNetwork([range(3)], 2, hidden=8, rank=2)
    assert len(net.head_parameters(1)) == 1 + 2 * (len(net.main) + len(net.linear) + 1)


def test_residuals_are_cross_fitted():
    """A leak planted in one inner fold's response changes that fold's residual by
    exactly the leak: the model predicting those rows never saw them."""
    from morpheus_encoder.hypothesis_generator import grouped_folds

    X, y, B, groups, _ = synthetic_bank(0)
    y = y + 2.0 * B[:, 1]  # strong comparison signal: ridge beats the fold mean either way
    levels = comparison_levels(COMPARISON_NAMES)
    base = residual_bank(B, y, groups, levels, seed=3)
    folds = grouped_folds(groups, len(y), 5, 3)
    rows = folds == 2
    leak = np.zeros_like(y)
    leak[rows] = 5.0 * B[rows, 3]
    for g in np.unique(groups[rows]):
        m = groups == g
        leak[m] -= leak[m].mean()
    leaked = residual_bank(B, y + leak, groups, levels, seed=3)
    assert all(leaked.info[lv]["model"] == base.info[lv]["model"] == "ridge" for lv in ("B1", "B2", "B3"))
    for k, lv in enumerate(levels):
        if lv == "N" and base.info[lv]["model"] != leaked.info[lv]["model"]:
            continue
        np.testing.assert_allclose(leaked.residuals[rows, k] - base.residuals[rows, k], leak[rows], atol=1e-8)


def test_planted_linear_effect_recovered_by_head_zero():
    cfg = ResidualBankConfig(generator=GeneratorConfig(n_fits=2, hidden=32, rank=4, max_epochs=40, min_epochs=5,
                                                       patience=5, attribution_rows=600, reference_rows=128),
                             attribution_pool=24, pair_pool=6)
    hits = 0
    for seed in range(10):
        X, y, B, groups, views = synthetic_bank(100 + seed)
        r0 = residual_bank(B, y, groups, comparison_levels(COMPARISON_NAMES, ("N",))).residuals[:, 0]
        a = int(np.random.default_rng(seed).integers(X.shape[1]))
        g = X[:, a] * np.sqrt(0.10 * r0.var() / X[:, a].var())
        res = generate_residual_bank_hypotheses(X, y + g, groups, comparison=B, comparison_names=COMPARISON_NAMES,
                                                views=views, max_candidates=8, seed=seed, device="cpu", config=cfg)
        head0 = [c.variables for c in res.per_level["N"]]
        hits += (a,) in head0[:3]
        assert set(res.per_level) == set(LEVELS)
        for c in res.candidates:
            assert c.metadata["provenance"]["levels"]
    assert hits >= 9, hits


def test_zero_marginal_product_nominated_by_every_head():
    """Known answer (Agent Q's product_2 world): x_a x_b with its linear projection on (x_a, x_b)
    removed, at 10 % of r(0) variance, is a product hypothesis in every head's list."""
    cfg = ResidualBankConfig(generator=GeneratorConfig(n_fits=2, hidden=32, rank=4, max_epochs=40, min_epochs=5,
                                                       patience=5, attribution_rows=600, reference_rows=128),
                             attribution_pool=24, pair_pool=6)
    hits = 0
    for seed in range(5):
        X, y, B, groups, views = synthetic_bank(200 + seed)
        a, b = (int(c) for c in np.random.default_rng(seed).choice(X.shape[1], 2, replace=False))
        prod = X[:, a] * X[:, b]
        D = np.column_stack([np.ones(len(y)), X[:, a], X[:, b]])
        g = prod - D @ np.linalg.lstsq(D, prod, rcond=None)[0]
        assert abs(np.corrcoef(g, X[:, a])[0, 1]) < 1e-8 and abs(np.corrcoef(g, X[:, b])[0, 1]) < 1e-8
        r0 = residual_bank(B, y, groups, comparison_levels(COMPARISON_NAMES, ("N",))).residuals[:, 0]
        g *= np.sqrt(0.10 * r0.var() / g.var())
        res = generate_residual_bank_hypotheses(X, y + g, groups, comparison=B, comparison_names=COMPARISON_NAMES,
                                                views=views, max_candidates=8, seed=seed, device="cpu", config=cfg)
        pair = tuple(sorted((a, b)))
        found = [any(c.variables == pair and c.feature_form == "product" for c in res.per_level[lv])
                 for lv in LEVELS]
        hits += all(found)
    assert hits >= 4, hits
    off = ResidualBankConfig(generator=cfg.generator, attribution_pool=24, pair_pool=6, pair_screen=False)
    assert ResidualBankConfig.from_dict(off.to_dict()) == off


def test_heads_separate_levels():
    """A protected column that proxies a composition column (B1) carries signal for the N head
    only; a column outside every comparison level carries it for every head."""
    cfg = ResidualBankConfig(generator=GeneratorConfig(n_fits=2, hidden=32, rank=4, max_epochs=40, min_epochs=5,
                                                       patience=5, attribution_rows=600, reference_rows=128),
                             attribution_pool=24, pair_pool=6)
    X, y, B, groups, views = synthetic_bank(7)
    rng = np.random.default_rng(7)
    a, b = 3, 45
    X[:, a] = B[:, 1] + 0.1 * rng.standard_normal(len(y))
    y = 0.6 * B[:, 1] + 0.4 * X[:, b] + rng.standard_normal(len(y))
    res = generate_residual_bank_hypotheses(X, y, groups, comparison=B, comparison_names=COMPARISON_NAMES,
                                            views=views, max_candidates=8, seed=0, device="cpu", config=cfg)
    rank = {lv: [c.variables for c in res.per_level[lv]] for lv in LEVELS}
    assert rank["N"].index((a,)) <= 1
    assert all((b,) in rank[lv][:2] for lv in LEVELS)
    assert all((a,) not in rank[lv][:4] for lv in ("B1", "B2", "B3"))
    assert res.report["n_multi_level"] < res.report["n_distinct"]


def test_config_round_trips():
    cfg = ResidualBankConfig(generator=GeneratorConfig(n_fits=3, library=("ridge",)), attribution_pool=7,
                             levels=("N", "B3"))
    assert ResidualBankConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert ResidualBankConfig.from_dict(ResidualBankConfig().to_dict()) == ResidualBankConfig()


def test_only_protected_views_are_read():
    X, y, B, groups, views = synthetic_bank(1, rows=60)
    views = {**views, "programmes": np.arange(5)}
    cfg = ResidualBankConfig(generator=GeneratorConfig(n_fits=1, hidden=8, rank=2, max_epochs=3, min_epochs=1),
                             attribution_pool=4, pair_pool=2)
    res = generate_residual_bank_hypotheses(X, y, groups, comparison=B, comparison_names=COMPARISON_NAMES,
                                            views={"programmes": views["programmes"]} | {"scgpt": views["scgpt"]},
                                            max_candidates=4, device="cpu", config=cfg)
    assert res.report["views_not_read"] == ["programmes"]
    assert all(set(c.variables) <= set(views["scgpt"].tolist()) for c in res.candidates)
    with pytest.raises(ValueError):
        generate_residual_bank_hypotheses(X, y, groups, comparison=B, views={"programmes": views["programmes"]},
                                          device="cpu", config=cfg)


def test_partial_pair_screen_matches_brute_force_and_caps():
    from morpheus_encoder.residual_bank_generator import residual_pair_screen

    rng = np.random.default_rng(3)
    n = 4000
    X = rng.standard_normal((n, 6))
    X[:, 1] += 0.5 * X[:, 0]
    e = 0.8 * X[:, 0] + 0.3 * X[:, 2] * X[:, 3] + rng.standard_normal(n)
    Xs = torch.as_tensor(X, dtype=torch.float64)
    got = residual_pair_screen(Xs, e, top=15, proxy=0.99, partial=True)
    for (i, j), v in got.items():
        D = np.column_stack([np.ones(n), X[:, i], X[:, j]])
        p = X[:, i] * X[:, j]
        rp = p - D @ np.linalg.lstsq(D, p, rcond=None)[0]
        re = e - D @ np.linalg.lstsq(D, e, rcond=None)[0]
        assert abs(np.corrcoef(rp, re)[0, 1] ** 2 - v) < 1e-6
    assert max(got, key=got.get) == (2, 3)
    capped = residual_pair_screen(Xs, e, top=15, proxy=0.99, partial=True, cap_per_column=1)
    cols = [c for pair in capped for c in pair]
    assert len(cols) == len(set(cols)) and (2, 3) in capped


def test_head_gates_receive_only_their_head_loss():
    from morpheus_encoder.residual_bank_generator import HeadGatedBankNetwork

    torch.manual_seed(0)
    net = HeadGatedBankNetwork([range(0, 6), range(6, 10)], 3, hidden=16, rank=4).eval()
    with torch.no_grad():
        for layer in [*net.main, *net.linear, net.higher[-1]]:
            layer.weight.normal_(0, 0.3)
        for q in net.Q:
            q.normal_(0, 0.3)
    x, r = torch.randn(64, 10), torch.randn(64, 3)
    net.zero_grad()
    multitask_loss(net(x), r, torch.tensor([0.5, 0.0, 0.5])).backward()
    assert all(g.log_alpha.grad is None or torch.count_nonzero(g.log_alpha.grad) == 0 for g in net.head_gates[1])
    assert any(torch.count_nonzero(g.log_alpha.grad) for g in net.head_gates[0])
    assert net(x).shape == (64, 3)
    assert net.openness(10, 1).shape == (10,)


def test_patient_r2_stopping_and_head_gates_recover_planted_linear():
    X, y, B, groups, views = synthetic_bank(101)
    r0 = residual_bank(B, y, groups, comparison_levels(COMPARISON_NAMES, ("N",))).residuals[:, 0]
    a = 17
    g = X[:, a] * np.sqrt(0.10 * r0.var() / X[:, a].var())
    gen = GeneratorConfig(n_fits=2, hidden=32, rank=4, max_epochs=40, min_epochs=5, patience=5,
                          attribution_rows=600, reference_rows=128)
    for extra in ({"stop_metric": "patient_r2"}, {"head_gates": True, "l0_width_normalised": True}):
        cfg = ResidualBankConfig(generator=gen, attribution_pool=24, pair_pool=6, **extra)
        res = generate_residual_bank_hypotheses(X, y + g, groups, comparison=B, comparison_names=COMPARISON_NAMES,
                                                views=views, max_candidates=8, seed=0, device="cpu", config=cfg)
        assert (a,) in [c.variables for c in res.per_level["N"]][:3], extra
        assert all("left_initial_state" in f and "heldout_mean_patient_r2" in f for f in res.report["fits"])
