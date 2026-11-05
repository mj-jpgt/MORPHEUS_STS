"""Benchmark v05 baselines: the v04 definitions with declared compute bounds, one entry point per method.

Every method receives the same support rows (``Ctx``: X, r, groups, pairs, names, seed) and returns
``(single scores, pair scores, info)``.  CPU baselines see a patient-balanced subsample of at most ``ROW_CAP`` support
spots (declared deviation, ``DEVIATIONS``); ours, the linear screen and NID use every support row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from dbench.benchmark_v04 import methods_v04 as M4
from dbench.benchmark_v04 import methods_slow as MS
from dbench.benchmark_v04.methods_v04 import Ctx, _group_folds, _std

ROW_CAP = 4000


@dataclass
class Ctx5(Ctx):
    names: list = None   # names of the pool singles (spatial version is parsed from the name)

DEVIATIONS = {
    "row_cap": f"CPU-bound baselines (EBM, GBM+SHAP, GBM+H, RuleFit (2,000 spots), iRF (2,000 spots, 256-column shortlist), MISTy-style forest, stability selection, "
               f"knockoffs, fair modes, PySR) see a patient-balanced random subsample of at most {ROW_CAP} support spots "
               f"(seeded); ours, the linear screen, NID and TabDistill (its own <= 10,000-spot rule) use all support rows",
    "ebm": "outer_bags=2, max_rounds=1000 (spec 8 bags, 5000 rounds); interactions = the 200 pool pairs with the highest "
           "FAST strength (spec fallback restricted to the pool); the other pool pairs are ranked below every fitted pair "
           "term by their FAST strength",
    "gbm_shap": "SHAP values on all columns for singles; SHAP interaction values on 1,000 support spots from a second LightGBM "
                "(same settings) refitted on the 96 columns with the largest mean |SHAP|; pool pairs with a member outside "
                "the shortlist score 0",
    "gbm_h": "Friedman H^2 on a 20-point grid per variable and 20 background spots (timing: 50 exceeded the 240 s budget), for all pool pairs; singles unscored (0)",
    "misty": "scikit-learn MISTy-style views (intra = own-spot values, juxta = 6-neighbour mean, para = 18-neighbour mean; "
             "contrast columns have no MISTy view and score -inf), one random forest per view weighted by its OOB R^2; "
             "pairs are DERIVED as the product of the two single importances (MISTy has no pair output)",
    "stability_selection": "complementary-pairs subsampling with 15 pairs of half-samples (30 subsamples; spec 50 pairs) and "
                           "20 alphas, Lasso paths on singles plus all pool product columns",
}


def cap_ctx(ctx: Ctx, cap: int = ROW_CAP) -> Ctx:
    """Patient-balanced, seeded subsample of the support rows (proportional per patient, at least one row)."""
    n = len(ctx.r)
    if n <= cap:
        return ctx
    rng = np.random.default_rng(ctx.seed)
    keep = []
    for g in np.unique(ctx.groups):
        i = np.flatnonzero(ctx.groups == g)
        keep.append(rng.choice(i, min(len(i), max(1, int(round(cap * len(i) / n)))), replace=False))
    keep = np.sort(np.concatenate(keep))
    return replace(ctx, X=ctx.X[keep], r=ctx.r[keep], groups=ctx.groups[keep], nb6=None)


# ------------------------------------------------------------------ EBM
def ebm(ctx: Ctx):
    return M4.ebm(ctx, n_pair_terms=200, outer_bags=2, max_rounds=1000)


# ------------------------------------------------------------------ GBM + SHAP interactions (two stage)
def gbm_shap(ctx: Ctx, n_short: int = 96, n_spots: int = 1000):
    import shap
    mdl = M4._fit_lgbm(ctx)
    rng = np.random.default_rng(ctx.seed)
    rows = rng.choice(len(ctx.r), min(2000, len(ctx.r)), replace=False)
    sv = np.asarray(shap.TreeExplainer(mdl).shap_values(ctx.X[rows]))
    s1 = np.abs(sv).mean(0)
    short = np.argsort(-s1, kind="stable")[:n_short]
    sub = replace(ctx, X=ctx.X[:, short])
    mdl2 = M4._fit_lgbm(sub)
    ex2 = shap.TreeExplainer(mdl2)
    sel = rows[:n_spots]
    iv = np.zeros((len(sel), len(short), len(short)), dtype=np.float32)
    for lo in range(0, len(sel), 100):
        iv[lo: lo + 100] = np.asarray(ex2.shap_interaction_values(ctx.X[sel[lo: lo + 100]][:, short]))
    pos = {int(c): j for j, c in enumerate(short)}
    s2 = np.zeros(len(ctx.pairs))
    for i, (a, b) in enumerate(ctx.pairs):
        if int(a) in pos and int(b) in pos:
            s2[i] = np.abs(iv[:, pos[int(a)], pos[int(b)]]).mean() * 2.0
    return s1, s2, {"short": int(n_short)}


def friedman_h_batched(ctx: Ctx, grid: int = 20, n_bg: int = 20, chunk_pairs: int = 25):
    """Friedman's H^2 for every pool pair: the arithmetic of methods_v04.friedman_h (same model, grid, background, partial
    dependence, H^2 formula) with one preallocated design buffer and one predict call per chunk of pairs (identical per-row
    predictions, so H is unchanged; checked by check_h_equal)."""
    mdl = M4._fit_lgbm(ctx)
    X = ctx.X
    rng = np.random.default_rng(ctx.seed)
    bg = X[rng.choice(len(X), min(n_bg, len(X)), replace=False)]
    qs = np.linspace(0.02, 0.98, grid)
    pairs = ctx.pairs
    need = sorted({int(v) for p in pairs for v in p})
    gv = {j: np.quantile(X[:, j], qs) for j in need}
    nb = len(bg)

    def pd1(j):
        Z = np.repeat(bg, grid, axis=0)
        Z[:, j] = np.tile(gv[j], nb)
        return mdl.predict(Z).reshape(nb, grid).mean(0)
    P1 = {j: pd1(j) for j in need}
    rows = nb * grid * grid
    base = np.repeat(bg, grid * grid, axis=0)
    buf = np.empty((chunk_pairs * rows, X.shape[1]), dtype=X.dtype)
    H = np.zeros(len(pairs))
    for lo in range(0, len(pairs), chunk_pairs):
        sub = pairs[lo: lo + chunk_pairs]
        for k, (a, b) in enumerate(sub):
            a, b = int(a), int(b)
            blk = buf[k * rows:(k + 1) * rows]
            blk[:] = base
            blk[:, a] = np.tile(np.repeat(gv[a], grid), nb)
            blk[:, b] = np.tile(np.tile(gv[b], grid), nb)
        pred = mdl.predict(buf[: len(sub) * rows]).reshape(len(sub), nb, grid, grid)
        for k, (a, b) in enumerate(sub):
            a, b = int(a), int(b)
            f = pred[k].mean(0)
            f = f - f.mean()
            fa, fb = P1[a] - P1[a].mean(), P1[b] - P1[b].mean()
            resid = f - fa[:, None] - fb[None, :]
            H[lo + k] = float((resid ** 2).sum() / max((f ** 2).sum(), 1e-12))
    return np.zeros(X.shape[1]), H, {}


def gbm_h(ctx: Ctx):
    return friedman_h_batched(ctx, grid=20, n_bg=20)


# ------------------------------------------------------------------ MISTy-style views
def misty(ctx: Ctx, n_trees: int = 100):
    from sklearn.ensemble import RandomForestRegressor
    from dbench.harness.core8 import _version
    X, y = ctx.X.astype(np.float64), ctx.r.astype(np.float64)
    vers = np.array([_version(n) for n in ctx.names])
    out = np.full(X.shape[1], -np.inf)
    info = {}
    for v in ("own", "nbr6", "nbr18"):
        cols = np.flatnonzero(vers == v)
        if not len(cols):
            continue
        rf = RandomForestRegressor(n_estimators=n_trees, max_features="sqrt", min_samples_leaf=20, oob_score=True,
                                   random_state=ctx.seed, n_jobs=ctx.n_jobs).fit(X[:, cols], y)
        out[cols] = rf.feature_importances_ * max(float(rf.oob_score_), 0.0)
        info[f"oob_r2_{v}"] = float(rf.oob_score_)
    fin = np.where(np.isfinite(out), out, 0.0)
    s2 = fin[ctx.pairs[:, 0]] * fin[ctx.pairs[:, 1]]
    return out, s2, info


def stability_selection(ctx: Ctx, n_pairs_of_halves: int = 15, n_alpha: int = 20):
    """Complementary-pairs stability selection (Shah and Samworth); same arithmetic as methods_v04.stability_selection with
    the Lasso paths of the half-samples run in parallel."""
    from joblib import Parallel, delayed
    from sklearn.linear_model import lasso_path
    Z = _std(ctx.X.astype(np.float64))
    A = np.column_stack([Z, M4._products(Z, ctx.pairs)])
    y = ctx.r - ctx.r.mean()
    pats = np.unique(ctx.groups)
    rng = np.random.default_rng(ctx.seed)
    amax = float(np.max(np.abs(A.T @ y)) / len(y))
    alphas = np.logspace(np.log10(amax), np.log10(amax * 1e-2), n_alpha)
    halves = []
    for _ in range(n_pairs_of_halves):
        perm = rng.permutation(pats)
        h = len(perm) // 2
        halves += [perm[:h], perm[h: 2 * h]]
    rows_l = [np.flatnonzero(np.isin(ctx.groups, hv)) for hv in halves]
    rows_l = [r for r in rows_l if len(r) >= 10]
    def one(rows):
        _, coefs, _ = lasso_path(A[rows], y[rows], alphas=alphas)
        return coefs != 0
    cnt = sum(Parallel(n_jobs=ctx.n_jobs, backend="threading")(delayed(one)(r) for r in rows_l))
    prob = (cnt / max(len(rows_l), 1)).max(1)
    return prob[: Z.shape[1]], prob[Z.shape[1]:], {"n_subsamples": len(rows_l)}


def rulefit(ctx: Ctx):
    """imodels.RuleFitRegressor(max_rules=500, tree_size=4) on 2,000 spots.  Coefficients are read from _get_rules() (the v04
    extractor read Rule.coef, which does not exist, and scored every candidate 0: found by a planted-interaction
    known-answer test, corrected before any final table).  Singles: sum |coef| of rules and linear terms containing the
    candidate; pairs: sum |coef| of rules containing both."""
    import re
    from imodels import RuleFitRegressor
    c = cap_ctx(ctx, 2000)
    m = c.X.shape[1]
    mdl = RuleFitRegressor(max_rules=500, tree_size=4, random_state=c.seed).fit(c.X, c.r, feature_names=[f"f{j}" for j in range(m)])
    df = mdl._get_rules()
    df = df[df["coef"].abs() > 0]
    s1 = np.zeros(m)
    pair = {}
    for rule, coef in zip(df["rule"], df["coef"].abs()):
        feats = sorted({int(x) for x in re.findall(r"f(\d+)", str(rule))})
        for j in feats:
            s1[j] += coef
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                pair[(feats[a], feats[b])] = pair.get((feats[a], feats[b]), 0.0) + coef
    s2 = np.array([pair.get((min(int(a), int(b)), max(int(a), int(b))), 0.0) for a, b in ctx.pairs])
    return s1, s2, {"n_rules": int(len(df))}


# ------------------------------------------------------------------ iRF (shortlist)
def irf(ctx: Ctx, n_short: int = 256, n_rows: int = 2000, n_estimators: int = 100, n_iterations: int = 3, n_bootstraps: int = 3):
    """imodels.IRFRegressor on the 256 singles with the largest |corr| with the residual; pairs scored by the largest
    interaction stability of any recovered interaction containing both members (pairs outside the shortlist score 0)."""
    from imodels import IRFRegressor
    c = cap_ctx(ctx, n_rows)
    sl = MS.shortlist_idx(c, n_short)
    mdl = IRFRegressor(n_estimators=n_estimators, n_iterations=n_iterations, n_bootstraps=n_bootstraps, n_jobs=ctx.n_jobs,
                       random_state=ctx.seed).fit(c.X[:, sl], c.r)
    s1 = np.zeros(ctx.X.shape[1])
    s1[sl] = np.asarray(mdl.feature_importances_, dtype=float)
    pair = {}
    for key, v in mdl.interaction_stability_.items():
        key = [int(sl[j]) for j in key]
        for a in range(len(key)):
            for b in range(a + 1, len(key)):
                k = (min(key[a], key[b]), max(key[a], key[b]))
                pair[k] = max(pair.get(k, 0.0), float(v))
    s2 = np.array([pair.get((min(int(a), int(b)), max(int(a), int(b))), 0.0) for a, b in ctx.pairs])
    return s1, s2, {"n_stability_terms": len(mdl.interaction_stability_)}


# ------------------------------------------------------------------ fair mode on a shortlist
def _fair_short(ctx: Ctx, fit_fn, n_short: int = 256, n_rows: int = 2000, n_hold: int = 600):
    """Grouped permutation importance on held-out support folds (spec 3.14) for the 256 shortlisted singles and 2,000
    spots; pool pairs with a member outside the shortlist score 0."""
    c = cap_ctx(ctx, n_rows)
    sl = MS.shortlist_idx(c, n_short)
    pos = {int(j): k for k, j in enumerate(sl)}
    ok = [i for i, (a, b) in enumerate(ctx.pairs) if int(a) in pos and int(b) in pos]
    pr = np.array([(pos[int(ctx.pairs[i][0])], pos[int(ctx.pairs[i][1])]) for i in ok]).reshape(-1, 2)
    sub = replace(c, X=c.X[:, sl], pairs=pr)
    s1_s, s2_s, info = MS.fair_scores(sub, fit_fn, lambda m, X: m.predict(X), n_hold=n_hold)
    s1 = np.full(ctx.X.shape[1], np.nan)
    s1[sl] = s1_s
    s2 = np.full(len(ctx.pairs), 0.0)
    s2[ok] = s2_s
    return s1, s2, dict(info, n_short=n_short, n_pairs_in_shortlist=len(ok))


def fair_rulefit(ctx: Ctx):
    from imodels import RuleFitRegressor
    def fit(X, r):
        return RuleFitRegressor(max_rules=500, tree_size=4, random_state=ctx.seed).fit(X, r, feature_names=[f"f{j}" for j in range(X.shape[1])])
    return _fair_short(ctx, fit, n_short=96, n_rows=1500, n_hold=200)   # repair attempt: RuleFit.predict dominates


def fair_gbm(ctx: Ctx):
    import lightgbm as lgb
    def fit(X, r):
        return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, n_jobs=ctx.n_jobs, verbose=-1, random_state=ctx.seed).fit(X, r)
    return _fair_short(ctx, fit)


def fair_irf(ctx: Ctx):
    from imodels import IRFRegressor
    def fit(X, r):
        return IRFRegressor(n_estimators=100, n_iterations=3, n_bootstraps=1, n_rit=10, n_jobs=ctx.n_jobs, random_state=ctx.seed).fit(X, r)
    return _fair_short(ctx, fit)


def fair_ebm(ctx: Ctx):
    from interpret.glassbox import ExplainableBoostingRegressor
    def fit(X, r):
        return ExplainableBoostingRegressor(interactions=0, outer_bags=2, inner_bags=0, max_rounds=500, random_state=ctx.seed, n_jobs=ctx.n_jobs).fit(X, r)
    return _fair_short(ctx, fit)


# ------------------------------------------------------------------ registry
CAPPED = {"ebm": ebm, "gbm_shap": gbm_shap, "gbm_h": gbm_h, "misty": misty, "stability_selection": stability_selection,
          "rulefit": rulefit, "irf": None, "knockoffs": MS.knockoffs, "pysr": MS.pysr_shortlist,
          "fair_gbm": fair_gbm, "fair_rulefit": fair_rulefit, "fair_ebm": fair_ebm, "fair_irf": fair_irf,
          "fair_misty": None}
FULL = {"linear_screen": M4.linear_screen, "nid": MS.nid, "tabdistill": MS.tabdistill}


def fair_misty(ctx: Ctx):
    """Fair mode for the MISTy-style forest: the per-view forests are fitted on the other folds and scored by grouped
    permutation (spec 3.14); implemented as one forest over all non-contrast columns weighted as in ``misty``."""
    from sklearn.ensemble import RandomForestRegressor
    from dbench.harness.core8 import _version
    vers = np.array([_version(n) for n in ctx.names])
    ok = np.flatnonzero(np.isin(vers, ["own", "nbr6", "nbr18"]))
    rf_fit = lambda X, r: RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=20,  # noqa: E731
                                                random_state=ctx.seed, n_jobs=ctx.n_jobs).fit(X, r)
    sub = replace(ctx, X=ctx.X[:, ok], pairs=np.zeros((0, 2), dtype=int))
    s1_ok, _, info = MS.fair_scores(sub, rf_fit, lambda m, X: m.predict(X))
    s1 = np.full(ctx.X.shape[1], -np.inf)
    s1[ok] = s1_ok
    # pair scores: joint permutation minus singles, on the same fitted forests (non-contrast members only)
    pos = {int(c): j for j, c in enumerate(ok)}
    ok_pairs = [i for i, (a, b) in enumerate(ctx.pairs) if int(a) in pos and int(b) in pos]
    s2 = np.zeros(len(ctx.pairs))
    if ok_pairs:
        sub2 = replace(ctx, X=ctx.X[:, ok], pairs=np.array([(pos[int(ctx.pairs[i][0])], pos[int(ctx.pairs[i][1])]) for i in ok_pairs]))
        _, p2, _ = MS.fair_scores(sub2, rf_fit, lambda m, X: m.predict(X))
        s2[ok_pairs] = p2
    return s1, s2, info


CAPPED["fair_misty"] = fair_misty
CAPPED["irf"] = irf
NOCAP = {"knockoffs"}   # spatial 4-spot binning (own rule) needs the neighbour graph of the full support rows
ALL = sorted(set(CAPPED) | set(FULL))


def run_method(name: str, ctx: Ctx):
    """(single, pair, info) for ``name``; CPU baselines on the row-capped context."""
    t = time.time()
    if name in FULL:
        s1, s2, info = FULL[name](ctx)
    elif name in NOCAP:
        s1, s2, info = CAPPED[name](ctx)
    else:
        s1, s2, info = CAPPED[name](cap_ctx(ctx))
    return np.asarray(s1, float), np.asarray(s2, float), dict(info, seconds=round(time.time() - t, 1))
