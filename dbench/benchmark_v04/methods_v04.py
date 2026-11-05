"""Baselines of METHOD_BENCHMARK_SPEC section 3, scoring the shared pool of one (study, target, split).

Every function takes a context ``ctx`` (support rows only; no query row is ever passed) and returns
``(single_scores (m,), pair_scores (k,), info dict)``:

    ctx.X       (n_s, m) pool singles, support rows, within-patient centred and scaled by the episode builder
    ctx.r       (n_s,)  target residualised on B_known (relation_v03.baseline_for(task, "known").r_s)
    ctx.groups  (n_s,)  patient codes
    ctx.pairs   (k, 2)  positions into the m singles
    ctx.seed    int

Deviations from the specification are listed in ``DEVIATIONS`` and echoed into the result tables.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

DEVIATIONS = {
    "ebm": "interactions = the 200 pool pairs with the highest FAST strength (spec 3.3.2 fallback restricted to the "
           "pool); the other pool pairs are scored by their FAST strength scaled below every fitted pair term",
    "gbm_shap": "SHAP interaction values (spec 3.4.3) computed on the 96 singles with the largest mean |SHAP| and 1,000 "
                "support spots: a full 1,400-feature interaction tensor is infeasible; pool pairs with a member outside "
                "the shortlist score 0 (stated in the table)",
    "gbm_h": "Friedman H^2 on a 20-point grid per variable, 100 background spots",
}


@dataclass
class Ctx:
    X: np.ndarray
    r: np.ndarray
    groups: np.ndarray
    pairs: np.ndarray
    seed: int = 0
    n_jobs: int = 4
    nb6: np.ndarray | None = None   # (n_s, 6) support-row neighbour indices (-1 = none), for spatial binning


def _products(Z, pairs):
    P = Z[:, pairs[:, 0]] * Z[:, pairs[:, 1]]
    P = P - P.mean(0)
    sd = P.std(0)
    sd[sd < 1e-8] = 1.0
    return P / sd


def _std(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def _group_folds(groups, k, seed):
    from sklearn.model_selection import GroupKFold
    pats = np.unique(groups)
    k = max(2, min(k, len(pats)))
    return list(GroupKFold(k).split(np.zeros(len(groups)), groups=groups))


# ------------------------------------------------------------------ 3.13 random
def random_scores(ctx: Ctx, seeds: int = 20):
    m, k = ctx.X.shape[1], len(ctx.pairs)
    return ([np.random.default_rng(ctx.seed * 1000 + s).random(m) for s in range(seeds)],
            [np.random.default_rng(ctx.seed * 1000 + 500 + s).random(k) for s in range(seeds)], {})


# ------------------------------------------------------------------ 3.2 linear interaction screen
def linear_screen(ctx: Ctx):
    """RidgeCV(alphas=logspace(-3, 3, 13), GroupKFold(5), intercept) on singles + product columns.  Closed form with one
    eigendecomposition per fold (GPU when available): the solution and the CV rule (mean fold R^2, as sklearn's
    GridSearch over Ridge.score) are those of ``sklearn.linear_model.RidgeCV(cv=GroupKFold(5))``."""
    Z = _std(ctx.X.astype(np.float64))
    A = np.column_stack([Z, _products(Z, ctx.pairs)])
    y = ctx.r.astype(np.float64)
    folds = _group_folds(ctx.groups, 5, ctx.seed)
    alphas = np.logspace(-3, 3, 13)
    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:  # pragma: no cover
        torch = None
    if torch is None:
        from sklearn.linear_model import RidgeCV
        mdl = RidgeCV(alphas=alphas, cv=folds, fit_intercept=True).fit(A, y)
        c = np.abs(mdl.coef_)
        return c[: Z.shape[1]], c[Z.shape[1]:], {"alpha": float(mdl.alpha_), "path": "sklearn"}
    At = torch.as_tensor(A, dtype=torch.float64, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float64, device=dev)
    r2 = torch.zeros(len(alphas), dtype=torch.float64, device=dev)
    for tr, va in folds:
        tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
        Xtr, ytr = At[tr_t], yt[tr_t]
        mu, ym = Xtr.mean(0), ytr.mean()
        Xc, yc = Xtr - mu, ytr - ym
        evals, V = torch.linalg.eigh(Xc.T @ Xc)
        rhs = V.T @ (Xc.T @ yc)
        Xv, yv = At[va_t] - mu, yt[va_t]
        XvV = Xv @ V
        sst = ((yv - yv.mean()) ** 2).sum()
        for i, a in enumerate(alphas):
            beta_v = rhs / (evals + a)
            pred = XvV @ beta_v + ym
            r2[i] += 1.0 - ((yv - pred) ** 2).sum() / sst
    best = float(alphas[int(torch.argmax(r2))])
    mu, ym = At.mean(0), yt.mean()
    Xc, yc = At - mu, yt - ym
    evals, V = torch.linalg.eigh(Xc.T @ Xc)
    beta = V @ ((V.T @ (Xc.T @ yc)) / (evals + best))
    c = beta.abs().cpu().numpy()
    return c[: Z.shape[1]], c[Z.shape[1]:], {"alpha": best, "path": "gpu-eigh"}


# ------------------------------------------------------------------ 3.3 EBM
def ebm(ctx: Ctx, n_pair_terms: int = 200, outer_bags: int = 8, max_rounds: int = 5000):
    from interpret.glassbox import ExplainableBoostingRegressor
    from interpret.utils import measure_interactions
    X = ctx.X.astype(np.float64)
    m = X.shape[1]
    pairs = [(int(a), int(b)) for a, b in ctx.pairs]
    fast = measure_interactions(X, ctx.r, interactions=pairs)
    fd = {tuple(sorted((int(k[0]), int(k[1])))): float(v) for k, v in fast}
    fs = np.array([fd.get(tuple(sorted(p)), 0.0) for p in pairs])
    top = np.argsort(-fs)[:n_pair_terms]
    sel = [pairs[i] for i in top]
    mdl = ExplainableBoostingRegressor(interactions=sel, outer_bags=outer_bags, inner_bags=0, max_rounds=max_rounds,
                                       random_state=ctx.seed, n_jobs=ctx.n_jobs)
    mdl.fit(X, ctx.r)
    imp = np.asarray(mdl.term_importances(), dtype=float)
    names = list(mdl.term_names_)
    s1 = np.zeros(m)
    pair_imp = {}
    for t, v in zip(names, imp):
        parts = [p.strip() for p in t.split(" & ")]
        idx = [int(p[len("feature_"):]) if p.startswith("feature_") else None for p in parts]
        if len(idx) == 1 and idx[0] is not None:
            s1[idx[0]] = v
        elif len(idx) == 2 and None not in idx:
            pair_imp[tuple(sorted(idx))] = v
    floor = (min(pair_imp.values()) if pair_imp else 0.0)
    s2 = np.empty(len(pairs))
    fmax = max(fs.max(), 1e-12)
    for i, p in enumerate(pairs):
        key = tuple(sorted(p))
        s2[i] = pair_imp[key] if key in pair_imp else floor * (fs[i] / fmax) * 0.999
    return s1, s2, {"n_pair_terms": len(sel)}


# ------------------------------------------------------------------ 3.4 GBM with SHAP interactions and Friedman H
def _fit_lgbm(ctx: Ctx):
    import lightgbm as lgb
    folds = _group_folds(ctx.groups, 5, ctx.seed)
    tr, va = folds[0]
    mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, random_state=ctx.seed, n_jobs=ctx.n_jobs, verbose=-1)
    mdl.fit(ctx.X[tr], ctx.r[tr], eval_set=[(ctx.X[va], ctx.r[va])],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    return mdl


def gbm_shap(ctx: Ctx, n_short: int = 96, n_spots: int = 1000):
    import shap
    mdl = _fit_lgbm(ctx)
    ex = shap.TreeExplainer(mdl)
    rng = np.random.default_rng(ctx.seed)
    rows = rng.choice(len(ctx.r), min(2000, len(ctx.r)), replace=False)
    sv = np.asarray(ex.shap_values(ctx.X[rows]))
    s1 = np.abs(sv).mean(0)
    short = np.argsort(-s1)[:n_short]
    sub = rows[: n_spots]
    iv = _interactions_on(ex, ctx.X[sub], short)
    pos = {int(c): j for j, c in enumerate(short)}
    s2 = np.zeros(len(ctx.pairs))
    for i, (a, b) in enumerate(ctx.pairs):
        if int(a) in pos and int(b) in pos:
            s2[i] = np.abs(iv[:, pos[int(a)], pos[int(b)]]).mean() * 2.0
    return s1, s2, {"n_iter": int(mdl.best_iteration_ or 500), "short": int(n_short)}


def _interactions_on(ex, Xsub, short):
    """SHAP interaction tensor restricted to ``short`` features (full tensor over every column is infeasible):
    the model is explained on all columns, entries for the shortlist are kept (chunked to bound memory)."""
    out = np.zeros((len(Xsub), len(short), len(short)), dtype=np.float32)
    for lo in range(0, len(Xsub), 25):
        full = np.asarray(ex.shap_interaction_values(Xsub[lo: lo + 25]))
        out[lo: lo + 25] = full[:, short][:, :, short]
    return out


def friedman_h(ctx: Ctx, grid: int = 20, n_bg: int = 100):
    mdl = _fit_lgbm(ctx)
    X = ctx.X
    rng = np.random.default_rng(ctx.seed)
    bg = X[rng.choice(len(X), min(n_bg, len(X)), replace=False)]
    qs = np.linspace(0.02, 0.98, grid)
    pairs = ctx.pairs
    need = sorted({int(v) for p in pairs for v in p})
    gv = {j: np.quantile(X[:, j], qs) for j in need}

    def pd1(j):
        out = np.empty(grid)
        Z = np.repeat(bg, grid, axis=0)
        Z[:, j] = np.tile(gv[j], len(bg))
        return mdl.predict(Z).reshape(len(bg), grid).mean(0)

    P1 = {j: pd1(j) for j in need}
    H = np.zeros(len(pairs))
    for i, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        Z = np.repeat(bg, grid * grid, axis=0)
        Z[:, a] = np.tile(np.repeat(gv[a], grid), len(bg))
        Z[:, b] = np.tile(np.tile(gv[b], grid), len(bg))
        f = mdl.predict(Z).reshape(len(bg), grid, grid).mean(0)
        f = f - f.mean()
        fa, fb = P1[a] - P1[a].mean(), P1[b] - P1[b].mean()
        resid = f - fa[:, None] - fb[None, :]
        H[i] = float((resid ** 2).sum() / max((f ** 2).sum(), 1e-12))
    return np.zeros(X.shape[1]), H, {}


# ------------------------------------------------------------------ 3.7 RuleFit
def rulefit(ctx: Ctx):
    from imodels import RuleFitRegressor
    m = ctx.X.shape[1]
    names = [f"f{j}" for j in range(m)]
    mdl = RuleFitRegressor(max_rules=500, tree_size=4, random_state=ctx.seed)
    mdl.fit(ctx.X, ctx.r, feature_names=names)
    s1 = np.zeros(m)
    pair = {}
    import re
    for rule in getattr(mdl, "rules_", []):
        coef = abs(float(getattr(rule, "coef", 0.0)))
        if coef == 0:
            continue
        feats = sorted({int(x) for x in re.findall(r"f(\d+)", str(rule.rule))}) if hasattr(rule, "rule") else []
        for j in feats:
            s1[j] += coef
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                pair[(feats[a], feats[b])] = pair.get((feats[a], feats[b]), 0.0) + coef
    s2 = np.array([pair.get(tuple(sorted((int(a), int(b)))), 0.0) for a, b in ctx.pairs])
    return s1, s2, {"n_rules": int(len(getattr(mdl, "rules_", [])))}


# ------------------------------------------------------------------ 3.11 stability selection
def stability_selection(ctx: Ctx, n_pairs_of_halves: int = 50, n_alpha: int = 30):
    from sklearn.linear_model import lasso_path
    Z = _std(ctx.X.astype(np.float64))
    A = np.column_stack([Z, _products(Z, ctx.pairs)])
    y = ctx.r - ctx.r.mean()
    pats = np.unique(ctx.groups)
    rng = np.random.default_rng(ctx.seed)
    amax = float(np.max(np.abs(A.T @ y)) / len(y))
    alphas = np.logspace(np.log10(amax), np.log10(amax * 1e-2), n_alpha)
    count = np.zeros((A.shape[1], n_alpha))
    n = 0
    for _ in range(n_pairs_of_halves):
        perm = rng.permutation(pats)
        h = len(perm) // 2
        for half in (perm[:h], perm[h: 2 * h]):
            rows = np.isin(ctx.groups, half)
            if rows.sum() < 10:
                continue
            _, coefs, _ = lasso_path(A[rows], y[rows], alphas=alphas)
            count += (coefs != 0)
            n += 1
    prob = (count / max(n, 1)).max(1)
    return prob[: Z.shape[1]], prob[Z.shape[1]:], {"n_subsamples": n}


def check_linear_equivalence():
    """Known-answer test: the closed-form GPU path equals sklearn RidgeCV(GroupKFold(5)) coefficients."""
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(3)
    n, m = 600, 40
    X = rng.normal(size=(n, m))
    g = np.repeat(np.arange(8), n // 8)
    pairs = np.array(sorted({tuple(sorted(rng.choice(m, 2, replace=False))) for _ in range(120)}))
    r = X[:, 2] * 0.6 + X[:, 4] * X[:, 7] * 0.9 + rng.normal(size=n)
    ctx = Ctx(X=X, r=r, groups=g, pairs=pairs, seed=0)
    s1, s2, info = linear_screen(ctx)
    Z = _std(X)
    A = np.column_stack([Z, _products(Z, pairs)])
    ref = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=_group_folds(g, 5, 0), fit_intercept=True).fit(A, r)
    c = np.abs(ref.coef_)
    err = float(np.max(np.abs(np.concatenate([s1, s2]) - c)))
    assert abs(np.log10(info["alpha"]) - np.log10(ref.alpha_)) < 1e-9 and err < 1e-6, (info, ref.alpha_, err)
    return err


METHODS = {"linear_screen": linear_screen, "ebm": ebm, "gbm_shap": gbm_shap, "gbm_h": friedman_h, "rulefit": rulefit,
           "stability_selection": stability_selection}


def run(name: str, ctx: Ctx):
    t = time.time()
    if name not in METHODS:
        from dbench.benchmark_v04 import methods_slow as S
        return S.run(name, ctx)
    s1, s2, info = METHODS[name](ctx)
    info = dict(info, seconds=round(time.time() - t, 1))
    return s1, s2, info
