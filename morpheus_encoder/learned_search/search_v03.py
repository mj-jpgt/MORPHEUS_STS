"""Search at inference (spec 1.6) and the harness entry point ``search(task, B, K)``.

1. g_theta once over all candidate tokens (support rows only).
2. Singles: exact support gain G_sup (every single is fitted; cheap) - mu_delta - kappa sigma_delta.
3. Pairs: every pair scored mu_sup - mu_delta - kappa sigma_delta from the pair head.
4. Beam: the top n1 singles and top n2 pairs (n1 = n2 = (K - 12) / 2, so that every fit, including
   the 8 triples and 4 quadruples below, counts against K); fit the pairs, exact G_sup replaces mu_sup.
5. Extend: triples from the best quarter of fitted pairs + each of the top 32 singles, scored by the
   mean over constituent pairs of (pair support gain, exact where fitted, else mu_sup) - mu_delta -
   kappa sigma_delta; fit the top 8.  Quadruples the same way from the top 2 triples; fit the top 4.
   (The spec's "exact support gain of the triple" would need every proposed triple fitted; the
   constituent-pair score is the budget-respecting equivalent.)
6. Final rank of the fitted sets: by exact support gain G_sup(S) (default, ``final_rank="gsup"``;
   owner decision 2026-10-21: on development the LCB final rank cost yield while the learned
   beam chose better sets than the support-fit screen).  ``final_rank="lcb"`` is the opt-in
   variant: G_sup(S) - mu_delta(S) - kappa sigma_delta(S), constituent-pair means for |S| > 2.
   The beam (which sets get fitted) always uses the learned LCB scores.
``model=None`` is the support-fit screen (baseline 11): mu_delta = sigma = 0 and a pair's support
gain predicted as the mean of its singles'.  Nothing reads query rows.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from . import relation_v03 as rv

KAPPA = 1.0
N_TRIPLES, N_QUADS, TOP_SINGLES_EXT = 8, 4, 32


def _support_only(task: rv.Task) -> rv.Task:
    s = task.s
    if s == len(task.y):
        return task
    cut = lambda a: None if a is None else np.asarray(a)[:s]  # noqa: E731
    nb = None if task.nb6 is None else np.where(np.asarray(task.nb6)[:s] < s, np.asarray(task.nb6)[:s], -1)
    return rv.Task(X=task.X[:s], y=task.y[:s], n_support=s, groups=task.groups[:s], depth=cut(task.depth),
                   known=cut(task.known), composition=cut(task.composition), nb6=nb, sections=cut(task.sections),
                   within_share=task.within_share)


def score_model(model, task: rv.Task, B: str, L: str, diag: np.ndarray, base: rv.BaselineFit):
    """(single mu_delta, single sigma, H) in R^2 units; H for the pair head."""
    with torch.no_grad():
        H = model.encode(task.X, base.r_s, task.groups, task.sections, task.nb6, diag, B, L)
        mu, ls = model.single_head(H)
    from .model_v03 import SCALE
    return mu.cpu().numpy() / SCALE, ls.exp().cpu().numpy() / SCALE, H


def search(task: rv.Task, B: str = "nuis", K: int = 64, *, model=None, L: str = "L0", kappa: float = KAPPA,
           device=None, return_details: bool = False, scorer=None, final_rank: str = "gsup", admissible=None):
    """Ranked sets for one target on SUPPORT rows: ``[(index tuple into task.X columns, predicted
    gain = G_sup - mu_delta, predicted sd or None), ...]`` best first; ``len`` <= K fitted sets.

    ``admissible(S) -> bool`` (a protocol family's requirement) constrains candidate generation:
    only admissible singles, pairs, triples and quadruples are fitted, and the budget of a size
    with no admissible member (e.g. singles under an own + neighbour requirement) goes to pairs."""
    from .model_v03 import SCALE
    task = _support_only(task)
    dev = torch.device(device) if device is not None else (model.device if model is not None else None)
    Bfit = "nuis" if B == "decomp" else B
    base = rv.baseline_for(task, Bfit)
    diag, single = rv.diagnostics(task, Bfit, base=base, device=dev)
    p = task.X.shape[1]
    gs = single["g_sup"]
    if scorer is not None:   # an external (baseline) scorer through the identical beam
        mu1, sg1 = (np.asarray(a, dtype=np.float64) for a in scorer.singles(task, B, L, diag, base))
        H = None
    elif model is not None:
        model.eval()
        mu1, sg1, H = score_model(model, task, B, L, diag, base)
    else:
        mu1, sg1, H = np.zeros(p), np.zeros(p), None
    s1 = gs - mu1 - kappa * sg1
    n_ext = min(N_TRIPLES + N_QUADS, K // 4)
    n_tri, n_quad = n_ext - n_ext // 3, n_ext // 3          # 8 + 4 at K >= 48
    n1 = max(1, (K - n_ext) // 2)
    n2 = max(0, K - n_ext - n1)
    top1 = np.argsort(-s1, kind="stable")
    adm1 = top1 if admissible is None else np.asarray([i for i in top1 if admissible((int(i),))], dtype=int)
    n2 += n1 - min(n1, len(adm1))
    n1 = min(n1, len(adm1))
    # pair matrix over every candidate (chunked); baseline 11: mean of the singles' support gains
    if scorer is not None:
        msup, mdel, msig = (np.asarray(a, dtype=np.float64) for a in scorer.pairs(task, B, L, diag, base, gs))
    elif H is not None:
        with torch.no_grad():
            PM = model.pair_matrix(H).cpu().numpy()
        msup, mdel, msig = PM[..., 0] / SCALE, PM[..., 1] / SCALE, np.exp(PM[..., 2]) / SCALE
    else:
        msup = 0.5 * (gs[:, None] + gs[None, :])
        mdel = msig = np.zeros((p, p))
    ps = msup - mdel - kappa * msig
    iu = np.triu_indices(p, 1)
    order = np.argsort(-ps[iu], kind="stable")
    if admissible is None:
        order = order[:n2]
    else:
        keep = []
        for j in order:
            if admissible((int(iu[0][j]), int(iu[1][j]))):
                keep.append(j)
                if len(keep) == n2:
                    break
        order = np.asarray(keep, dtype=int)
    pairs = np.stack([iu[0][order], iu[1][order]], 1) if len(order) else np.zeros((0, 2), dtype=int)
    sup_exact = {}
    if len(pairs):
        fp = rv.fit_sets(task.X[:, pairs], base, task.y, task.groups, device=dev)["g_sup"]
        sup_exact.update({(int(a), int(b)): float(g) for (a, b), g in zip(pairs, fp)})

    def pair_sup(a, b):
        a, b = min(a, b), max(a, b)
        return sup_exact.get((a, b), float(msup[a, b]))

    def pair_score(S):
        return float(np.mean([pair_sup(a, b) - mdel[a, b] - kappa * msig[a, b] for a, b in combinations(S, 2)]))

    def final(S, g):
        if len(S) == 1:
            return g - mu1[S[0]] - kappa * sg1[S[0]]
        cp = list(combinations(S, 2))
        return g - np.mean([mdel[a, b] for a, b in cp]) - kappa * np.mean([msig[a, b] for a, b in cp])

    def expected(S, g):
        if len(S) == 1:
            return g - mu1[S[0]], sg1[S[0]]
        cp = list(combinations(S, 2))
        return g - np.mean([mdel[a, b] for a, b in cp]), np.mean([msig[a, b] for a, b in cp])

    fitted = [((int(i),), float(gs[i])) for i in adm1[:n1]] + [((int(a), int(b)), sup_exact[(int(a), int(b))])
                                                              for a, b in pairs]
    ext_singles = [int(i) for i in top1[:TOP_SINGLES_EXT]]
    pair_rank = sorted([f for f in fitted if len(f[0]) == 2], key=lambda f: -final(*f))
    for size, n_fit, seeds in ((3, n_tri, pair_rank[:max(1, len(pair_rank) // 4)]), (4, n_quad, None)):
        if size == 4:
            tri = sorted([f for f in fitted if len(f[0]) == 3], key=lambda f: -final(*f))
            seeds = tri[:2]
        props = {tuple(sorted(S + (c,))) for S, _ in seeds for c in ext_singles if c not in S}
        if admissible is not None:
            props = {S for S in props if admissible(S)}
        props = sorted(props, key=lambda S: -pair_score(S))[:n_fit]
        if props:
            g = rv.fit_sets(task.X[:, np.asarray(props)], base, task.y, task.groups, device=dev)["g_sup"]
            for S, gg in zip(props, g):
                fitted.append((tuple(int(x) for x in S), float(gg)))
    if final_rank not in ("gsup", "lcb"):
        raise ValueError(f"final_rank must be 'gsup' or 'lcb', not {final_rank!r}")
    ranked = sorted(fitted, key=(lambda f: -f[1]) if final_rank == "gsup" else (lambda f: -final(*f)))
    out = []
    for S, g in ranked[:K]:
        e, sd = expected(S, g)
        has_sd = (scorer is not None and getattr(scorer, "has_sd", True)) or (scorer is None and model is not None)
        out.append((S, float(e), float(sd) if has_sd else None))
    if return_details:
        return out, {"n_fitted": len(fitted), "diag": diag, "single_g_sup": gs, "mu_delta": mu1, "sigma": sg1,
                     "fitted": [(S, g, float(final(S, g))) for S, g in fitted]}
    return out


MOLECULAR_VIEW_IDS = ("gene", "programme")


def family_admissible(names: list, view_ids: np.ndarray, family: dict | None):
    """A protocol family's requirement on pool candidates (index tuples), as the packet runner
    applies it: ``context`` = at least one own-spot and at least one nbr6 / nbr18 / contrast
    variable; ``cross_modal`` = at least one molecular and at least one morphology variable."""
    if not family:
        return None
    from .episodes import VIEW_VOCAB
    own = np.array([str(n).startswith("own:") for n in names])
    mol = np.isin(view_ids, [VIEW_VOCAB.index(v) for v in MOLECULAR_VIEW_IDS])
    morph = np.asarray(view_ids) == VIEW_VOCAB.index("morphology")
    need_ctx, need_x = bool(family.get("context")), bool(family.get("cross_modal"))
    if not (need_ctx or need_x):
        return None

    def ok(S):
        S = list(S)
        if need_ctx and not (own[S].any() and (~own[S]).any()):
            return False
        if need_x and not (mol[S].any() and morph[S].any()):
            return False
        return True
    return ok


MAX_ROWS_PER_PATIENT = 600   # the training episodes' cap (episodes.EpisodeSampler.max_rows_per_patient)


def bridge_task(X_raw, y_raw, groups, neighbours6, *, names, views, nuisance_raw=None, composition=None,
                sections=None, device=None, max_rows_per_patient: int | None = MAX_ROWS_PER_PATIENT,
                seed: int = 0):
    """(relation_v03.Task, pool) from UN-CENTRED proposal-row inputs, constructed exactly as a
    training episode: base columns and target centred within patient over every row, the four spatial
    versions over every row, then (as training) at most ``max_rows_per_patient`` rows per patient kept,
    support-sd scaling (``centre_scale``) on the kept rows, the nuisance as centred and scaled log depth,
    the composition variables, sections for Moran's I, neighbours among the kept rows, and the
    within-patient covariance share of every candidate from raw per-patient means (own / nbr6 / nbr18
    share the base column's between-patient covariance, contrast has none), as
    ``episodes_v03.SamplerV03._extras``.  ``pool.names`` / ``pool.ep.meta`` name the candidates."""
    from .episodes import centre_scale, neighbour_operators, transform_block
    from .search_v02 import build_pool
    Xr = np.asarray(X_raw, dtype=np.float64)
    yr = np.asarray(y_raw, dtype=np.float64)
    pool = build_pool(Xr, yr, groups, neighbours6, names=names, views=views, embed=None, nuisance=nuisance_raw,
                      device=device)
    ep = pool.ep
    g = np.asarray(ep.groups)
    n = len(g)
    pats, inv = np.unique(g, return_inverse=True)
    index = {str(c): i for i, c in enumerate(names)}
    base_cols = np.array([index[nm.split(":", 1)[1]] for nm in pool.names[:len(pool.names) // 4]])
    m = len(base_cols)
    B = Xr[:, base_cols]
    pm = np.stack([B[g == p].mean(0) for p in pats])
    ty = np.array([yr[g == p].mean() for p in pats])
    Bc, yc = B - pm[inv], yr - ty[inv]
    nb = np.asarray(ep.nb6) if ep.nb6 is not None else -np.ones((n, 6), dtype=np.int64)
    A6, A18 = neighbour_operators(nb)
    rng = np.random.default_rng(seed)
    sel = []
    for p in range(len(pats)):
        r = np.flatnonzero(inv == p)
        if max_rows_per_patient is not None and len(r) > max_rows_per_patient:
            r = np.sort(rng.choice(r, max_rows_per_patient, replace=False))
        sel.append(r)
    sel = np.sort(np.concatenate(sel))   # rows keep their given order
    gs = g[sel]
    sup = np.ones(len(sel), dtype=bool)
    T = transform_block(Bc, A6, A18, sel).astype(np.float64)
    w = np.abs((T * yc[sel, None]).mean(0))
    _, cnt = np.unique(gs, return_counts=True)
    wts = cnt / cnt.sum()
    dm, dy = pm - (wts[:, None] * pm).sum(0), ty - (wts * ty).sum()
    bc = np.abs((wts[:, None] * dm * dy[:, None]).sum(0))
    share = (w / (w + np.concatenate([bc, bc, bc, np.zeros(m)]) + 1e-12)).astype(np.float32)
    pos = -np.ones(n, dtype=np.int64)
    pos[sel] = np.arange(len(sel))
    nbk = np.where(nb[sel] >= 0, pos[np.maximum(nb[sel], 0)], -1)
    depth = None
    if nuisance_raw is not None:
        d = np.asarray(nuisance_raw, dtype=np.float64).reshape(n, -1)[:, 0]
        d = d - np.array([d[g == p].mean() for p in pats])[inv]
        depth = centre_scale(d[sel], gs, sup).astype(np.float64)
    task = rv.Task(X=centre_scale(T, gs, sup), y=centre_scale(yc[sel], gs, sup).astype(np.float64),
                   n_support=len(sel), groups=gs, depth=depth, known=None,
                   composition=None if composition is None else np.asarray(composition, dtype=np.float64)[sel],
                   nb6=nbk, sections=None if sections is None else np.asarray(sections)[sel], within_share=share)
    return task, pool


def propose_sets_v03(model, X, y, groups, neighbours6, *, names, views, embed=None, B: str = "nuis", K: int = 64,
                     nuisance=None, device=None, raw=None, family=None, **_):
    """``propose_sets`` contract on proposal (support) rows: [(prefixed-name tuple, support gain, info)]
    in the default final order.

    ``raw`` (discovery-packet runner): ``{"X", "y", "depth", "composition", "sections"}`` UN-CENTRED
    proposal-row values aligned with ``X`` / ``y``; the task is then built exactly as a training
    episode (``bridge_task``).  Without it the given (already standardised) rows are used, with
    ``nuisance`` as the depth.  ``family``: ``{"context": bool, "cross_modal": bool}``, the
    protocol family requirement, enforced at candidate generation (``search(admissible=...)``).
    With ``embed`` the v03 morphology similarities are appended (standardised on these rows; the
    statistics stay on ``model.morph_stats`` for the all-row rebuild in ``propose.transform_columns``).
    ``info``: support_gain, predicted_shrinkage (mu_delta), predicted_sd, predicted_gain."""
    X = np.asarray(X, dtype=np.float64)
    names, views = [str(c) for c in names], {k: list(v) for k, v in dict(views or {}).items()} or None
    Xr = None if raw is None else np.asarray(raw["X"], dtype=np.float64)
    if embed is not None:
        from .morphology_dictionary import load_dictionary_v03, similarities_v03, standardise_stats
        d = load_dictionary_v03()
        mu, sd = standardise_stats(np.asarray(embed, dtype=np.float32))
        model.morph_stats = (mu, sd)
        M = similarities_v03(embed, d["centres"], mu, sd)
        views = views or {"other": list(range(X.shape[1]))}
        views["morph"] = list(range(X.shape[1], X.shape[1] + M.shape[1]))
        names = names + [f"morph:k{j:02d}" for j in d["kept"]]
        X = np.concatenate([X, M], 1)
        if Xr is not None:
            Xr = np.concatenate([Xr, M], 1)
    if raw is not None:
        task, pool = bridge_task(Xr, raw["y"], groups, neighbours6, names=names, views=views,
                                 nuisance_raw=raw.get("depth"), composition=raw.get("composition"),
                                 sections=raw.get("sections"), device=device)
    else:
        from .search_v02 import build_pool
        pool = build_pool(X, y, groups, neighbours6, names=names, views=views, embed=None, nuisance=nuisance,
                          device=device)
        ep = pool.ep
        task = rv.Task(X=ep.X, y=np.asarray(ep.y, dtype=np.float64), n_support=ep.n_support, groups=ep.groups,
                       depth=pool.nuisance, known=None, composition=None, nb6=ep.nb6)
    adm = family_admissible(pool.names, np.asarray(pool.ep.meta.view), family)
    ranked, det = search(task, B, K, model=model, device=device, return_details=True, admissible=adm)
    gsup = {S: g for S, g, _ in det["fitted"]}
    out = []
    for i, (S, g, sd) in enumerate(ranked):
        out.append((tuple(pool.names[j] for j in S), float(gsup[S]),
                    {"n_sets_evaluated": det["n_fitted"], "path_rank": i, "support_gain": float(gsup[S]),
                     "predicted_shrinkage": float(gsup[S] - g), "predicted_sd": sd, "predicted_gain": float(g),
                     "g_ho_support": None, "response_surface": None}))
    return out


def pool_scores(task: rv.Task, B: str, model, pairs, *, L: str = "L0", device=None):
    """Scores of every single and of given pairs without fitting pairs (full-pool retrieval tables):
    singles G_sup - mu_delta (exact support gain), pairs mu_sup - mu_delta (pair head); returns
    (single_scores (p,), pair_scores (m,), single_sd (p,), pair_sd (m,)) in R^2 units."""
    from .model_v03 import SCALE
    task = _support_only(task)
    dev = torch.device(device) if device is not None else model.device
    Bfit = "nuis" if B == "decomp" else B
    base = rv.baseline_for(task, Bfit)
    diag, single = rv.diagnostics(task, Bfit, base=base, device=dev)
    model.eval()
    mu1, sg1, H = score_model(model, task, B, L, diag, base)
    pr = torch.as_tensor(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), device=H.device)
    with torch.no_grad():
        ms, md, ls = model.pair_head(H, pr)
    ms, md, sd = ms.cpu().numpy() / SCALE, md.cpu().numpy() / SCALE, ls.exp().cpu().numpy() / SCALE
    return single["g_sup"] - mu1, ms - md, sg1, sd
