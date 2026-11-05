"""The core eight baselines (governing spec 3.5, items 1-8) as harness search functions.

Each receives the support-only task, the baseline B and the budget K, works on the target after
f_B (the baseline residual on support), and returns its ranked variable sets (index tuples into the
candidate pool, at most K, no prediction).  The harness fits every registered set with the fixed
family and scores it, so the comparison is about selection.  The candidate pool already holds the
four spatial versions of every variable, so the classical methods get no neighbour graph (their
own neighbour means reduce to the column itself and map back to it).

1 ``linear_screen``  marginal + explicit pairwise screen, lasso path (``linear_interaction_screen``)
2 ``ebm``            EBM / GA2M pairwise term importance (``ebm_ga2m``)
3 ``gbm_h``          gradient boosting + Friedman H interaction ranking (``gbm_interaction``)
4 ``rulefit``        RuleFit rules -> variable sets (``rulefit``)
5 ``tabdistill``     TabPFN (v2 regressor weights) on the 256-candidate shortlist + ProxySPEX
                     interaction attribution (Kang et al. 2025), top interactions of order <= 4
6 ``pysr``           PySR on the same 256-candidate shortlist (``pysr_shortlist``)
7 ``residual_gen``   the previous MORPHEUS residual neural generator (``residual_bank_generator``)
8 ``random``         K random sets (size 1-4) fitted with the fixed family, ranked by support gain
Shortlist (5, 6): interaction-aware, ~256 candidates: top 192 singles by within-patient |r| with the baseline
residual + the constituents of the top 32 pairs of the whole pool by |r| of the centred product with the residual
+ the next singles by |r| (``shortlist``).
"""
from __future__ import annotations

import os

import numpy as np

SHORTLIST = 256
TABPFN_WEIGHTS = os.path.expanduser("~/tabpfn_weights/tabpfn-v2-regressor-09gpqh39.ckpt")
SPEX_PATH = os.path.expanduser("~/spex_only")


def _residual(task, B):
    if getattr(task, "y_resid", None) is not None:   # callers outside the harness pass the response directly
        return None, np.asarray(task.y_resid, dtype=float)
    from morpheus_encoder.learned_search import relation_v03 as rv
    base = rv.baseline_for(task, "nuis" if B == "decomp" else B)
    return base, np.asarray(base.r_s, dtype=float)


def shortlist(task, B, n: int = SHORTLIST, device=None, n_single: int = 192, n_pairs: int = 32) -> np.ndarray:
    """Interaction-aware shared shortlist (TabDistill, PySR): the top ``n_single`` candidates by
    within-patient |r| with the baseline residual, plus the constituents of the top ``n_pairs`` pairs of
    the WHOLE pool ranked by |r| of the centred product x_i x_j with the residual (two matrix products:
    X^T diag(y) X and (X^2)^T (X^2)), then the next singles by |r| up to ``n``.  A purely marginal screen
    would exclude the constituents of residualised interactions, whose marginals are about zero."""
    _, y = _residual(task, B)
    X = np.asarray(task.X, dtype=np.float64)
    g = np.asarray(task.groups)
    Xc, yc = X.copy(), y.copy()
    for p in np.unique(g):
        r = g == p
        Xc[r] -= Xc[r].mean(0)
        yc[r] -= yc[r].mean()
    Xc /= np.maximum(Xc.std(0), 1e-12)
    ny = np.sqrt((yc ** 2).sum()) + 1e-12
    r1 = np.abs(Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0)) * ny + 1e-12)
    num = np.abs((Xc * yc[:, None]).T @ Xc)                      # sum_n x_i x_j y
    X2 = Xc ** 2
    den = np.sqrt(np.maximum(X2.T @ X2, 1e-24)) * ny             # ||x_i x_j||
    r2 = num / den
    np.fill_diagonal(r2, -np.inf)
    iu = np.triu_indices(len(r1), 1)
    top_pairs = np.argsort(-r2[iu], kind="stable")[:n_pairs]
    order = list(np.argsort(-r1, kind="stable"))
    chosen = list(dict.fromkeys(order[:n_single] + [int(v) for t in top_pairs for v in (iu[0][t], iu[1][t])]))
    for c in order[n_single:]:
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(int(c))
    return np.sort(np.asarray(chosen[:max(n, len(chosen))], dtype=int))


def _method_input(task, y, cols, seed: int):
    from dbench.benchmark.methods import MethodInput
    X = np.ascontiguousarray(task.X[:, cols], dtype=np.float64)
    n = len(y)
    names = [f"c{j}" for j in range(len(cols))]
    return MethodInput(X=X, y=y, comparison=np.zeros((n, 0)), groups=np.asarray(task.groups),
                       neighbours=-np.ones((n, 6), dtype=np.int64), coords=None, names=names,
                       views={"raw_genes": np.arange(len(cols))}, roles={"raw_genes": "variable"},   # one protected view (residual generator)
                       family="molecular", seed=seed, comparison_full=np.zeros((n, 0)), comparison_names=[])


def _to_index(variables, cols) -> tuple | None:
    out = set()
    for v in variables:
        base = str(v).split(":")[-1]
        if not base.startswith("c"):
            return None
        out.add(int(cols[int(base[1:])]))
    return tuple(sorted(out)) if out else None


def _isolated(fn):
    """Run ``fn`` in a forked child (os.fork + pipe, so it also works inside daemonic pool workers) so
    that a native crash (a segfault inside a BLAS / coordinate-descent call) becomes a recorded error
    for this job instead of killing the worker and hanging the pool."""
    import os
    import pickle
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:   # child
        os.close(r)
        try:
            try:
                payload = pickle.dumps(("ok", fn()))
            except Exception as exc:  # noqa: BLE001
                payload = pickle.dumps(("err", f"{type(exc).__name__}: {exc}"))
            with os.fdopen(w, "wb") as f:
                f.write(payload)
        finally:
            os._exit(0)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        data = f.read()
    _, status = os.waitpid(pid, 0)
    if not data:
        raise RuntimeError(f"child died without a result (wait status {status})")
    st, val = pickle.loads(data)
    if st != "ok":
        raise RuntimeError(val)
    return val


def classical(method_name: str, use_shortlist: bool = False, device=None, seed: int = 0):
    from dbench.benchmark.methods import METHODS

    def search(task, B, K):
        base, y = _residual(task, B)
        cols = shortlist(task, B, device=device) if use_shortlist else np.arange(task.X.shape[1])
        inp = _method_input(task, y, cols, seed)
        out = _isolated(lambda: METHODS[method_name]().run(inp, max(K or 128, 128)))
        if not out.available:
            raise RuntimeError(f"{method_name} unavailable: {out.info.get('unavailable')}")
        ranked, seen = [], set()
        for h in out.hypotheses:
            idx = _to_index(h.variables, cols)
            if idx is None or idx in seen or len(idx) > 4:
                continue
            seen.add(idx)
            ranked.append((idx, None if K else float(h.score), None))   # K=None: every scored set with its score
            if K is not None and len(ranked) >= K:
                break
        return ranked
    return search


def random_search(device=None, seed: int = 0):
    from morpheus_encoder.learned_search import relation_v03 as rv

    def search(task, B, K):
        base, _ = _residual(task, B)
        p = task.X.shape[1]
        rng = np.random.default_rng(seed + p)
        sets = set()
        while len(sets) < K:
            k = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.4, 0.25, 0.1]))
            sets.add(tuple(sorted(int(i) for i in rng.choice(p, k, replace=False))))
        sets = sorted(sets)
        g = np.zeros(len(sets))
        for k in {len(s) for s in sets}:
            idx = [i for i, s in enumerate(sets) if len(s) == k]
            g[idx] = rv.gains(task, np.asarray([sets[i] for i in idx]), "nuis" if B == "decomp" else B, base=base,
                              device=device, query=False)["g_sup"]
        order = np.argsort(-g, kind="stable")
        return [(sets[i], None, None) for i in order]
    return search


def tabdistill(device=None, seed: int = 0, max_rows: int = 2048, n_eval: int = 128, sample_budget: int = 1024,
               n_estimators: int = 4):
    """TabPFN fitted on support (shortlist columns, the baseline residual as target); ProxySPEX over
    feature masks of the value function v(mask) = R^2 of TabPFN's predictions on ``n_eval`` support
    rows with the masked-out candidates set to their support mean; interactions (order <= 4) ranked
    by |Mobius coefficient|."""
    import sys

    def search(task, B, K):
        if SPEX_PATH not in sys.path:
            sys.path.append(SPEX_PATH)
        import spectralexplain as spex
        from tabpfn import TabPFNRegressor

        base, y = _residual(task, B)
        cols = shortlist(task, B, device=device)
        rng = np.random.default_rng(seed)
        Z = np.asarray(task.X[:, cols], dtype=np.float64)
        Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
        fit_rows = rng.choice(len(y), min(max_rows, len(y)), replace=False)
        ev = rng.choice(len(y), min(n_eval, len(y)), replace=False)
        model = TabPFNRegressor(model_path=TABPFN_WEIGHTS, device=device or "cpu", n_estimators=n_estimators,
                                random_state=seed, ignore_pretraining_limits=True)
        model.fit(Z[fit_rows], y[fit_rows])
        Ze, ye = Z[ev], y[ev]
        sst = max(float(((ye - ye.mean()) ** 2).sum()), 1e-12)

        def value(masks, chunk: int = 16):   # predictions in chunks of masks (GPU memory)
            masks = np.asarray(masks, dtype=bool)
            out = []
            for c0 in range(0, len(masks), chunk):
                mm = masks[c0:c0 + chunk]
                batch = np.concatenate([np.where(m[None, :], Ze, 0.0) for m in mm])
                pred = model.predict(batch).reshape(len(mm), len(ev))
                out += list(1.0 - ((pred - ye[None]) ** 2).sum(1) / sst)
            return out
        expl = spex.Explainer(value_function=value, features=len(cols), sample_budget=sample_budget, max_order=4,
                              algorithm="proxyspex")
        inter = expl.interactions(index="mobius")
        coefs = getattr(inter, "interactions", None) or dict(inter)
        ranked, seen = [], set()
        for key, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
            idx = tuple(sorted(int(cols[j]) for j in key))
            if not idx or len(idx) > 4 or idx in seen:
                continue
            seen.add(idx)
            ranked.append((idx, None if K else float(abs(v)), None))
            if K is not None and len(ranked) >= K:
                break
        return ranked
    return search


def rulefit_sklearn(device=None, seed: int = 0, n_own: int = 40, n_trees: int = 100, max_depth: int = 3,
                    max_rows: int = 6000):
    """RuleFit (Friedman and Popescu 2008) in plain scikit-learn: rules = every internal-node path of
    100 depth-3 gradient-boosted trees on the shared interaction-aware shortlist (``shortlist``); rule indicators + linear terms; LassoLarsCV (5 patient-grouped folds); a rule's
    variable set ranked by the summed |coefficient| x rule support SD.  Used because imodels'
    RuleFitRegressor segfaults inside its own Lasso alpha search on these inputs."""
    def search(task, B, K):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LassoLarsCV
        from sklearn.model_selection import GroupKFold

        _, y = _residual(task, B)
        X = np.asarray(task.X, dtype=np.float64)
        Xc = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
        cols = shortlist(task, B, device=device)   # the shared interaction-aware shortlist
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(y), min(max_rows, len(y)), replace=False))
        Z, yy, g = Xc[rows][:, cols], y[rows], np.asarray(task.groups)[rows]
        gb = GradientBoostingRegressor(n_estimators=n_trees, max_depth=max_depth, learning_rate=0.1, subsample=0.5,
                                       random_state=seed).fit(Z, yy)
        rules, feats = [], []
        for est in gb.estimators_[:, 0]:
            t = est.tree_

            def walk(node, conds):
                for child, op in ((t.children_left[node], "<="), (t.children_right[node], ">")):
                    if child < 0:
                        continue
                    c = conds + [(t.feature[node], op, t.threshold[node])]
                    rules.append(c)
                    feats.append(tuple(sorted({int(f) for f, _, _ in c})))
                    if t.children_left[child] >= 0:
                        walk(child, c)
            if t.children_left[0] >= 0:
                walk(0, [])
        R = np.column_stack([np.all([(Z[:, f] <= th) if op == "<=" else (Z[:, f] > th) for f, op, th in c], 0)
                             for c in rules]).astype(float) if rules else np.zeros((len(yy), 0))
        keep = R.std(0) > 0
        R, feats = R[:, keep], [f for f, k in zip(feats, keep) if k]
        sd = R.std(0)
        D = np.column_stack([(R - R.mean(0)) / sd, Z])
        n_split = min(5, len(np.unique(g)))
        # LARS path (no sklearn coordinate-descent kernel, which segfaults in this environment on some inputs)
        las = LassoLarsCV(cv=list(GroupKFold(n_split).split(D, yy, g)), max_iter=500, max_n_alphas=200).fit(D, yy)
        score: dict = {}
        for j, f in enumerate(feats):
            score[f] = score.get(f, 0.0) + abs(las.coef_[j])
        for j in range(len(cols)):
            score[(j,)] = score.get((j,), 0.0) + abs(las.coef_[len(feats) + j])
        ranked = sorted((v, f) for f, v in score.items() if v > 0 and len(f) <= 4)
        ranked = [(tuple(sorted(int(cols[i]) for i in f)), v) for v, f in sorted(ranked, key=lambda t: -t[0])]
        out, seen = [], set()
        for idx, v in ranked:
            if idx not in seen:
                seen.add(idx)
                out.append((idx, None if K else float(v), None))
        return out if K is None else out[:K]
    search.parallel_ok = True
    return search


def pool_scorer(search_factory, seed: int = 0):
    """Pool scores of a set-proposing method: its own importance score of each pool single and pair
    (``search(task, B, None)`` returns every set the method scored, with the score); members the method
    did not score get the lowest score (ties broken at random by the evaluation)."""
    def score(task, B, singles, pairs):
        d, order = {}, []
        for idx, v, _ in search_factory()(task, B, None):
            order.append(tuple(int(i) for i in idx))
            if v is not None and np.isfinite(v):
                d[tuple(idx)] = max(d.get(tuple(idx), -np.inf), float(v))
        s1 = np.array([d.get((int(i),), -np.inf) for i in singles])
        s2 = np.array([d.get(tuple(sorted((int(a), int(b)))), -np.inf) for a, b in pairs])
        return s1, s2, list(dict.fromkeys(order))[:64]   # the method's own top-64 list, for precision@k
    return score


def random_pool(seed: int = 0):
    def score(task, B, singles, pairs):
        rng = np.random.default_rng(seed + len(singles) + int(task.X.shape[0]))
        return rng.random(len(singles)), rng.random(len(pairs)), []
    return score


def pool_scorers(device=None) -> dict:
    """Name -> pool scorer for the published methods (our method's comes from search_v03.pool_scores)."""
    return {"linear_screen": pool_scorer(lambda: classical("linear_interaction_screen", device=device)),
            "ebm": pool_scorer(lambda: classical("ebm_ga2m", device=device)),
            "gbm_h": pool_scorer(lambda: classical("gbm_interaction", device=device)),
            "rulefit": pool_scorer(lambda: rulefit_sklearn(device=device)),
            "pysr": pool_scorer(lambda: classical("pysr_shortlist", use_shortlist=True, device=device)),
            "tabdistill": pool_scorer(lambda: tabdistill(device=device)),
            "tabdistill_reduced": pool_scorer(lambda: tabdistill(device=device, n_eval=64, sample_budget=256, n_estimators=1)),
            "random": random_pool(),
            "ebm_full": ebm_full_pool(), "gbm_h_full": gbm_h_full_pool(),
            "misty": misty_pool(), "nid": nid_pool()}


def core8(device=None) -> dict:
    return {"linear_screen": lambda: classical("linear_interaction_screen", device=device),
            "ebm": lambda: classical("ebm_ga2m", device=device),
            "gbm_h": lambda: classical("gbm_interaction", device=device),
            "rulefit": lambda: rulefit_sklearn(device=device),
            "rulefit_imodels": lambda: classical("rulefit", device=device),   # crash reproducer only (run notes)
            "tabdistill": lambda: tabdistill(device=device),
            "pysr": lambda: classical("pysr_shortlist", use_shortlist=True, device=device),
            "residual_gen": lambda: classical("residual_bank_generator", device=device),
            "random": lambda: random_search(device=device),
            "nid": lambda: classical("nid_interaction", device=device),
            "misty": lambda: classical("misty_rf", device=device)}


# --- full-pool scorers for the two named methods that define a score for every single and every pair ---
def _pool_xy(task, B, singles, max_rows: int = 6000, seed: int = 0):
    _, y = _residual(task, B)
    X = np.asarray(task.X, dtype=np.float64)[:, np.asarray(singles)]
    rows = np.sort(np.random.default_rng(seed).choice(len(y), min(max_rows, len(y)), replace=False))
    return X[rows], y[rows], np.asarray(task.groups)[rows]


def ebm_full_pool(seed: int = 0):
    """EBM on the pool singles (mains only), main-term importance for every single; FAST interaction strength
    (interpret.utils.measure_interactions, the statistic EBM uses to rank pairs) for every pool pair, measured on
    the residual of the fitted mains."""
    def score(task, B, singles, pairs):
        import sys
        sys.path.insert(0, os.path.expanduser("~/q_pylib"))
        from interpret.glassbox import ExplainableBoostingRegressor
        from interpret.utils import measure_interactions
        X, y, g = _pool_xy(task, B, singles, seed=seed)
        ebm = ExplainableBoostingRegressor(interactions=0, outer_bags=4, max_rounds=1000, random_state=seed, n_jobs=1)
        ebm.fit(X, y)
        s1 = np.asarray(ebm.term_importances(), dtype=float)[:len(singles)]
        pos = {int(c): j for j, c in enumerate(np.asarray(singles))}
        want = [(pos[int(a)], pos[int(b)]) for a, b in pairs]
        res = measure_interactions(X, y - ebm.predict(X), interactions=want)
        d = {tuple(sorted(k)): float(v) for k, v in res}
        s2 = np.array([d.get(tuple(sorted(w)), 0.0) for w in want])
        return s1, s2, []
    return score


def gbm_h_full_pool(seed: int = 0, grid: int = 60):
    """Gradient boosting on the pool singles; permutation importance (held-out patients, as gbm_interaction) for
    every single; Friedman's H (unnormalised, partial dependence on ``grid`` sampled rows) for every pool pair."""
    def score(task, B, singles, pairs):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, y, g = _pool_xy(task, B, singles, seed=seed)
        rng = np.random.default_rng(seed)
        pats = np.unique(g)
        te = np.isin(g, rng.choice(pats, max(1, len(pats) // 3), replace=False))
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(X[~te], y[~te])
        Xt, yt = X[te], y[te]
        sub = rng.choice(len(yt), min(1500, len(yt)), replace=False)
        Xt, yt = Xt[sub], yt[sub]
        r2 = lambda p: 1 - np.sum((yt - p) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-12)  # noqa: E731
        base = r2(m.predict(Xt))
        s1 = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            Xp = Xt.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            s1[j] = base - r2(m.predict(Xp))
        final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(X, y)
        rows = rng.choice(len(y), min(grid, len(y)), replace=False)
        Xr = X[rows]
        pos = {int(c): j for j, c in enumerate(np.asarray(singles))}
        need = sorted({pos[int(v)] for p in pairs for v in p})
        pd1 = {}
        for j in need:
            Z = np.repeat(Xr, len(rows), axis=0)
            Z[:, j] = np.tile(Xr[:, j], len(rows))
            pd = final.predict(Z).reshape(len(rows), len(rows)).mean(0)
            pd1[j] = pd - pd.mean()
        s2 = np.zeros(len(pairs))
        for k, (a, b) in enumerate(pairs):
            ia, ib = pos[int(a)], pos[int(b)]
            Z = np.repeat(Xr, len(rows), axis=0)
            Z[:, ia] = np.tile(Xr[:, ia], len(rows))
            Z[:, ib] = np.tile(Xr[:, ib], len(rows))
            pd = final.predict(Z).reshape(len(rows), len(rows)).mean(0)
            s2[k] = float(np.std(pd - pd.mean() - pd1[ia] - pd1[ib]))
        return s1, s2, []
    return score


# --- MISTy-style multi-view random forest and neural interaction detection (full-pool where defined) ---
def _version(name: str) -> str:
    head = str(name).split(":", 1)[0]
    return head if head in ("own", "nbr6", "nbr18", "contrast") else "own"


def misty_importance(X, y, names, seed: int = 0, n_trees: int = 100, max_rows: int = 6000) -> np.ndarray:
    """MISTy (Tanevski et al. 2022) in scikit-learn: one random forest per view (intra = own-spot values,
    juxta = 6-neighbour means, para = 18-neighbour means), each variable's impurity importance weighted by its
    view's out-of-bag R^2 (clipped at 0).  Contrast versions have no MISTy view and get -inf."""
    from sklearn.ensemble import RandomForestRegressor
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=float)
    rows = np.sort(np.random.default_rng(seed).choice(len(y), min(max_rows, len(y)), replace=False))
    out = np.full(X.shape[1], -np.inf)
    vers = np.array([_version(n) for n in names])
    for v in ("own", "nbr6", "nbr18"):
        cols = np.flatnonzero(vers == v)
        if not len(cols):
            continue
        rf = RandomForestRegressor(n_estimators=n_trees, max_features="sqrt", min_samples_leaf=20, oob_score=True,
                                   random_state=seed, n_jobs=1).fit(X[rows][:, cols], y[rows])
        out[cols] = rf.feature_importances_ * max(float(rf.oob_score_), 0.0)
    return out


def misty_pool(seed: int = 0):
    def score(task, B, singles, pairs):
        _, y = _residual(task, B)
        names = [task.cand_names[int(i)] for i in singles]
        s1 = misty_importance(np.asarray(task.X)[:, np.asarray(singles)], y, names, seed)
        order = [(int(singles[j]),) for j in np.argsort(-np.where(np.isfinite(s1), s1, -1e300), kind="stable")[:64]]
        return s1, np.full(len(pairs), -np.inf), order        # pairs: not defined (MISTy scores variables)
    return score


def nid_pool(seed: int = 0, hidden=(64, 32, 16), l1: float = 3e-3, steps: int = 2000, batch: int = 256):
    """Neural interaction detection (Tsang, Cheng and Liu 2018): MLP with L1 on the first layer, trained on the
    pool singles; main strength sum_h z_h |W1_hi|; pair strength sum_h z_h min(|W1_hi|, |W1_hj|) for EVERY pool
    pair (z_h = aggregated |outgoing weights| of first-layer unit h)."""
    def score(task, B, singles, pairs):
        import torch
        _, y = _residual(task, B)
        X = np.asarray(task.X, dtype=np.float64)[:, np.asarray(singles)]
        torch.manual_seed(seed)
        Xt = torch.tensor((X - X.mean(0)) / np.maximum(X.std(0), 1e-12), dtype=torch.float32)
        yt = torch.tensor((y - y.mean()) / max(y.std(), 1e-12), dtype=torch.float32)
        g = np.asarray(task.groups)
        pats = np.unique(g)
        va = torch.tensor(np.isin(g, np.random.default_rng(seed).choice(pats, max(1, len(pats) // 5), replace=False)))
        tr = torch.nonzero(~va).squeeze(1)
        dims = (X.shape[1],) + tuple(hidden) + (1,)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        net = torch.nn.Sequential(*layers[:-1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, state = np.inf, None
        gen = torch.Generator().manual_seed(seed)
        for step in range(steps):
            b = tr[torch.randint(len(tr), (min(batch, len(tr)),), generator=gen)]
            loss = ((net(Xt[b]).squeeze(1) - yt[b]) ** 2).mean() + l1 * net[0].weight.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 100 == 0 and va.any():
                with torch.no_grad():
                    v = float(((net(Xt[va]).squeeze(1) - yt[va]) ** 2).mean())
                if v < best:
                    best, state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
        if state is not None:
            net.load_state_dict(state)
        Ws = [m.weight.detach().abs().numpy() for m in net if isinstance(m, torch.nn.Linear)]
        z = Ws[-1]
        for W in reversed(Ws[1:-1]):
            z = z @ W
        z = z.ravel()
        W1 = Ws[0]
        s1 = (z[:, None] * W1).sum(0)
        pos = {int(c): j for j, c in enumerate(np.asarray(singles))}
        s2 = np.array([float((z * np.minimum(W1[:, pos[int(a)]], W1[:, pos[int(b)]])).sum()) for a, b in pairs])
        return s1, s2, []
    return score
