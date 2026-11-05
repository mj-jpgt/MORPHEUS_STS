"""Slow baselines of METHOD_BENCHMARK_SPEC section 3 (NID, knockoffs, PySR, TabDistill, iRF) and fair mode (3.14).

Same context and return convention as methods_v04 (support rows only; ``(single scores, pair scores, info)``).
Shortlists and subsamples are declared in ``DEVIATIONS`` and echoed in every table that uses these rows.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from dbench.benchmark_v04.methods_v04 import Ctx, _group_folds, _std

DEVIATIONS = {
    "nid": "reference NID interaction-strength formula (mtsang/neural-interaction-detection, pairwise) evaluated on the pool "
           "pairs; MLP [140,100,60,20], L1 5e-5 on the first layer, early stopping on a grouped validation fold; trained "
           "with torch (the repository's PyTorch trainer is not used verbatim: its fixed training loop is replaced by the "
           "specified penalty and stopping rule)",
    "knockoffs": "Gaussian model-X knockoffs with lasso statistics (knockpy) on the 256 singles with the largest |corr| "
                 "with the residual and the pool pairs inside that shortlist; other pool pairs score 0; rows aggregated "
                 "to bins of up to 4 adjacent spots per patient (nb6 greedy bins); Sigma by Ledoit-Wolf",
    "pysr": "shortlist of the 64 singles with the largest |Spearman|; pairs outside the shortlist score 0; 10 targets per "
            "study, on the declared split subset",
    "tabdistill": "TabPFN v2 (local checkpoint) on the 256-single shortlist (<= 10,000 spots), ProxySPEX (order 2) value "
                  "function on 128 support spots; pairs outside the shortlist score 0",
    "irf": "imodels.IRFRegressor, 4 iterations, 500 trees",
    "fair_mode": "5 patient-grouped folds of the support; held-out rows subsampled to 600 per fold; one within-patient "
                 "permutation per variable; pair = joint permutation (same row permutation of both columns) minus the sum "
                 "of the two single increases",
}
SPEX_PATH = os.path.expanduser("~/spex_only")
TABPFN_WEIGHTS = os.path.expanduser("~/tabpfn_weights/tabpfn-v2-regressor-09gpqh39.ckpt")


def shortlist_idx(ctx: Ctx, n: int, rank: str = "pearson"):
    r = ctx.r - ctx.r.mean()
    Z = _std(ctx.X.astype(np.float64))
    if rank == "spearman":
        from scipy.stats import rankdata
        Z = _std(np.apply_along_axis(rankdata, 0, ctx.X))
        r = rankdata(r) - rankdata(r).mean()
    c = np.abs(Z.T @ r) / (np.linalg.norm(r) * np.sqrt(len(r)) + 1e-12)
    return np.argsort(-c, kind="stable")[:n]


# ------------------------------------------------------------------ 3.5 NID
def nid_scores_from_weights(weights, pairs):
    """Reference formulas: w_input = |W1| (H, p), w_later = product of the |W| above (1, H); pair strength =
    sum_h min(w_input[h, i], w_input[h, j]) * w_later[h]; main effect = sum_h w_input[h, i] * w_later[h]."""
    w_input = np.abs(weights[0])
    w_later = np.abs(weights[-1])
    for i in range(len(weights) - 2, 0, -1):
        w_later = w_later @ np.abs(weights[i])
    wl = w_later.reshape(-1)
    main = (w_input * wl[:, None]).sum(0)
    pr = np.asarray(pairs)
    strength = (np.minimum(w_input[:, pr[:, 0]], w_input[:, pr[:, 1]]) * wl[:, None]).sum(0)
    return main, strength


def nid(ctx: Ctx, epochs: int = 150, l1: float = 5e-5, batch: int = 128):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(ctx.seed)
    m = ctx.X.shape[1]
    X = torch.as_tensor(_std(ctx.X.astype(np.float64)), dtype=torch.float32, device=dev)
    y = torch.as_tensor(ctx.r - ctx.r.mean(), dtype=torch.float32, device=dev)
    tr, va = _group_folds(ctx.groups, 5, ctx.seed)[0]
    tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(m, 140), torch.nn.ReLU(), torch.nn.Linear(140, 100), torch.nn.ReLU(),
                              torch.nn.Linear(100, 60), torch.nn.ReLU(), torch.nn.Linear(60, 20), torch.nn.ReLU(),
                              torch.nn.Linear(20, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for lo in range(0, len(perm), batch):
            b = perm[lo: lo + batch]
            loss = ((net(X[b]).squeeze(-1) - y[b]) ** 2).mean() + l1 * net[0].weight.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(((net(X[va_t]).squeeze(-1) - y[va_t]) ** 2).mean())
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    W = [net[i].weight.detach().cpu().numpy() for i in (0, 2, 4, 6, 8)]
    main, pair = nid_scores_from_weights(W, ctx.pairs)
    return main, pair, {"epochs": ep + 1, "val_mse": best}


def check_nid_against_reference(repo_dir: str):
    """Known-answer test: the vectorised pair strength equals the repository's interpret_pairwise_interactions."""
    sys.path.insert(0, repo_dir)
    import importlib
    ref = importlib.import_module("neural_interaction_detection")
    rng = np.random.default_rng(0)
    W = [rng.normal(size=(14, 9)), rng.normal(size=(10, 14)), rng.normal(size=(6, 10)), rng.normal(size=(2, 6)), rng.normal(size=(1, 2))]
    pairs = np.array([(i, j) for i in range(9) for j in range(i + 1, 9)])
    _, ours = nid_scores_from_weights(W, pairs)
    w_in, w_later = ref.preprocess_weights(W)
    theirs = dict(ref.interpret_pairwise_interactions(w_in, w_later))
    err = max(abs(ours[k] - theirs[(int(a), int(b))]) for k, (a, b) in enumerate(pairs))
    assert err < 1e-9, err
    return err


# ------------------------------------------------------------------ 3.12 knockoffs
def spatial_bins(groups, nb6_rows, seed):
    """Greedy bins of up to 4 adjacent spots within patient; nb6_rows (n, 6) neighbour row indices (-1 = none)."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    bin_of = -np.ones(n, dtype=int)
    k = 0
    for i in rng.permutation(n):
        if bin_of[i] >= 0:
            continue
        bin_of[i] = k
        got = 1
        for j in nb6_rows[i]:
            if j >= 0 and bin_of[j] < 0 and groups[j] == groups[i] and got < 4:
                bin_of[j] = k
                got += 1
        k += 1
    return bin_of


def knockoffs(ctx: Ctx, n_short: int = 256):
    from knockpy.knockoff_filter import KnockoffFilter
    sl = shortlist_idx(ctx, n_short)
    pos = {int(c): j for j, c in enumerate(sl)}
    pidx = [i for i, (a, b) in enumerate(ctx.pairs) if int(a) in pos and int(b) in pos]
    Z = _std(ctx.X[:, sl].astype(np.float64))
    cols = [Z] + ([Z[:, [pos[int(ctx.pairs[i][0])] for i in pidx]] * Z[:, [pos[int(ctx.pairs[i][1])] for i in pidx]]] if pidx else [])
    A = np.column_stack(cols)
    bins = spatial_bins(ctx.groups, ctx.nb6, ctx.seed)
    nb = bins.max() + 1
    cnt = np.bincount(bins, minlength=nb).astype(float)
    Aa = np.stack([np.bincount(bins, weights=A[:, j], minlength=nb) for j in range(A.shape[1])], 1) / cnt[:, None]
    ya = np.bincount(bins, weights=ctx.r, minlength=nb) / cnt
    kf = KnockoffFilter(fstat="lasso", ksampler="gaussian")
    kf.forward(X=Aa, y=ya, fdr=0.2, shrinkage="ledoitwolf")
    W = np.asarray(kf.W, dtype=float)
    s1 = np.zeros(ctx.X.shape[1])
    s1[sl] = W[: len(sl)]
    s2 = np.zeros(len(ctx.pairs))
    for k, i in enumerate(pidx):
        s2[i] = W[len(sl) + k]
    return s1, s2, {"n_cols": int(A.shape[1]), "n_bins": int(nb), "n_pairs_in_shortlist": len(pidx)}


# ------------------------------------------------------------------ 3.9 PySR
def pysr_shortlist(ctx: Ctx, n_short: int = 64, procs: int = 4):
    from pysr import PySRRegressor
    sl = shortlist_idx(ctx, n_short, rank="spearman")
    X = ctx.X[:, sl].astype(np.float64)
    model = PySRRegressor(niterations=40, binary_operators=["+", "*", "-", "/"], unary_operators=["exp", "log", "square"],
                          maxsize=20, populations=30, procs=procs, random_state=ctx.seed, deterministic=True,
                          parallelism="serial", progress=False, verbosity=0, temp_equation_file=True,
                          variable_names=[f"x{j}" for j in range(len(sl))])
    model.fit(X, ctx.r)
    import re
    eq = model.equations_
    s1 = np.zeros(ctx.X.shape[1])
    pos = {int(c): j for j, c in enumerate(sl)}
    pair_score = {}
    for _, row in eq.iterrows():
        sc = float(row.get("score", 0.0) or 0.0)
        used = sorted({int(x) for x in re.findall(r"x(\d+)", str(row["equation"]))})
        for j in used:
            s1[sl[j]] += sc
        for a in range(len(used)):
            for b in range(a + 1, len(used)):
                key = (int(sl[used[a]]), int(sl[used[b]]))
                pair_score[key] = pair_score.get(key, 0.0) + sc
    s2 = np.array([pair_score.get(tuple(sorted((int(a), int(b)))), 0.0) for a, b in ctx.pairs])
    return s1, s2, {"n_eq": int(len(eq))}


# ------------------------------------------------------------------ 3.10 TabDistill
def tabdistill(ctx: Ctx, n_short: int = 256, max_rows: int = 10000, n_eval: int = 128, sample_budget: int = 2048):
    import torch
    if SPEX_PATH not in sys.path:
        sys.path.append(SPEX_PATH)
    import spectralexplain as spex
    from tabpfn import TabPFNRegressor
    sl = shortlist_idx(ctx, n_short, rank="spearman")
    pos = {int(c): j for j, c in enumerate(sl)}
    rng = np.random.default_rng(ctx.seed)
    Z = _std(ctx.X[:, sl].astype(np.float64))
    fit_rows = rng.choice(len(ctx.r), min(max_rows, len(ctx.r)), replace=False)
    ev = rng.choice(len(ctx.r), min(n_eval, len(ctx.r)), replace=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=TABPFN_WEIGHTS, device=dev, n_estimators=4, random_state=ctx.seed,
                            ignore_pretraining_limits=True)
    model.fit(Z[fit_rows], ctx.r[fit_rows])
    Ze, ye = Z[ev], ctx.r[ev]
    sst = max(float(((ye - ye.mean()) ** 2).sum()), 1e-12)

    def value(masks, chunk: int = 16):
        masks = np.asarray(masks, dtype=bool)
        out = []
        for c0 in range(0, len(masks), chunk):
            mm = masks[c0: c0 + chunk]
            batch = np.concatenate([np.where(m[None, :], Ze, 0.0) for m in mm])
            pred = model.predict(batch).reshape(len(mm), len(ev))
            out += list(1.0 - ((pred - ye[None]) ** 2).sum(1) / sst)
        return out
    expl = spex.Explainer(value_function=value, features=len(sl), sample_budget=sample_budget, max_order=2, algorithm="proxyspex")
    inter = expl.interactions(index="mobius")
    coefs = getattr(inter, "interactions", None) or dict(inter)
    s1 = np.zeros(ctx.X.shape[1])
    pair = {}
    for key, v in coefs.items():
        k = tuple(sorted(int(sl[j]) for j in key))
        if len(k) == 1:
            s1[k[0]] = abs(v)
        elif len(k) == 2:
            pair[k] = abs(v)
    s2 = np.array([pair.get(tuple(sorted((int(a), int(b)))), 0.0) for a, b in ctx.pairs])
    return s1, s2, {"n_terms": len(coefs)}


# ------------------------------------------------------------------ 3.8 iRF
def irf(ctx: Ctx):
    from imodels import IRFRegressor
    mdl = IRFRegressor(n_estimators=500, n_iterations=4, random_state=ctx.seed, n_jobs=ctx.n_jobs)
    mdl.fit(ctx.X, ctx.r)
    m = ctx.X.shape[1]
    s1 = np.asarray(getattr(mdl, "feature_importances_", np.zeros(m)), dtype=float)
    stab = {}
    for key in ("interactions_", "rit_scores_", "stability_"):
        if hasattr(mdl, key):
            obj = getattr(mdl, key)
            if isinstance(obj, dict):
                stab = {tuple(sorted(int(i) for i in (k if hasattr(k, "__iter__") else [k]))): float(v) for k, v in obj.items()}
            break
    s2 = np.array([stab.get(tuple(sorted((int(a), int(b)))), 0.0) for a, b in ctx.pairs])
    return s1, s2, {"n_stability_terms": len(stab), "attrs": [a for a in dir(mdl) if a.endswith("_") and not a.startswith("_")][:20]}


# ------------------------------------------------------------------ 3.14 fair mode
def fair_scores(ctx: Ctx, fit_fn, predict_fn, n_hold: int = 600, batch_rows: int = 200000):
    """Grouped permutation importance on held-out support folds.  ``fit_fn(X, r) -> model``;
    ``predict_fn(model, X) -> (n,)``."""
    m = ctx.X.shape[1]
    folds = _group_folds(ctx.groups, 5, ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    s_acc = np.zeros(m)
    p_acc = np.zeros(len(ctx.pairs))
    nf = 0
    for tr, va in folds:
        model = fit_fn(ctx.X[tr], ctx.r[tr])
        va = np.sort(rng.choice(va, min(n_hold, len(va)), replace=False))
        Xv, yv, gv = ctx.X[va], ctx.r[va], ctx.groups[va]
        # one within-patient permutation of the rows, shared by every variable
        perm = np.arange(len(va))
        for p in np.unique(gv):
            idx = np.flatnonzero(gv == p)
            perm[idx] = idx[rng.permutation(len(idx))]
        base = float(((predict_fn(model, Xv) - yv) ** 2).mean())
        n = len(va)
        per = max(1, batch_rows // n)

        def inc(cols_list):
            out = np.zeros(len(cols_list))
            for lo in range(0, len(cols_list), per):
                chunk = cols_list[lo: lo + per]
                Xb = np.tile(Xv, (len(chunk), 1))
                for k, cols in enumerate(chunk):
                    Xb[k * n:(k + 1) * n][:, cols] = Xv[perm][:, cols]
                pred = predict_fn(model, Xb).reshape(len(chunk), n)
                out[lo: lo + len(chunk)] = ((pred - yv[None]) ** 2).mean(1) - base
            return out
        single_inc = inc([[j] for j in range(m)])
        pair_inc = inc([[int(a), int(b)] for a, b in ctx.pairs])
        s_acc += single_inc
        p_acc += pair_inc - single_inc[ctx.pairs[:, 0]] - single_inc[ctx.pairs[:, 1]]
        nf += 1
    return s_acc / nf, p_acc / nf, {"folds": nf}


def _lgbm_fit(X, r):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, n_jobs=8, verbose=-1, random_state=0).fit(X, r)


def fair_gbm(ctx: Ctx):
    return fair_scores(ctx, _lgbm_fit, lambda mdl, X: mdl.predict(X))


def fair_rulefit(ctx: Ctx):
    from imodels import RuleFitRegressor

    def fit(X, r):
        mdl = RuleFitRegressor(max_rules=500, tree_size=4, random_state=ctx.seed)
        return mdl.fit(X, r, feature_names=[f"f{j}" for j in range(X.shape[1])])
    return fair_scores(ctx, fit, lambda mdl, X: mdl.predict(X))


def fair_ebm(ctx: Ctx):
    from interpret.glassbox import ExplainableBoostingRegressor

    def fit(X, r):
        return ExplainableBoostingRegressor(interactions=0, outer_bags=4, inner_bags=0, max_rounds=1000,
                                            random_state=ctx.seed, n_jobs=8).fit(X, r)
    return fair_scores(ctx, fit, lambda mdl, X: mdl.predict(X))


def fair_irf(ctx: Ctx):
    from imodels import IRFRegressor

    def fit(X, r):
        return IRFRegressor(n_estimators=200, n_iterations=4, random_state=ctx.seed, n_jobs=8).fit(X, r)
    return fair_scores(ctx, fit, lambda mdl, X: mdl.predict(X))


SLOW = {"nid": nid, "knockoffs": knockoffs, "pysr": pysr_shortlist, "tabdistill": tabdistill, "irf": irf,
        "fair_gbm": fair_gbm, "fair_rulefit": fair_rulefit, "fair_ebm": fair_ebm, "fair_irf": fair_irf}


def run(name: str, ctx: Ctx):
    t = time.time()
    s1, s2, info = SLOW[name](ctx)
    return s1, s2, dict(info, seconds=round(time.time() - t, 1))
