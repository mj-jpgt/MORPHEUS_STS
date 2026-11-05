"""Support-side statistics and realised held-out gains (learned search v02).

The relation is the closed-form one of the v02 search: a ridge on
``[random Fourier features of the set | the set linearly | 1]`` (16 features per
variable, one fixed map per set size, so no variable identity enters), fitted on
the residual of the target after the nuisance model (intercept + log depth,
ordinary least squares on support rows; targets and candidates are already
centred within patient, which absorbs patient, platform and study offsets).
The ridge penalty is chosen per set from ``ALPHAS`` by patient-grouped
leave-one-patient-out (LOPO) gain on support rows.

* general gain on rows R = (SSE_R(nuisance) - SSE_R(nuisance + relation)) / SST_R(y),
  i.e. R^2(full) - R^2(nuisance), the v07 general-gain definition.
* :func:`fit_sets` fits any batch of equally sized sets from per-patient Gram
  matrices, so LOPO and query gains of thousands of sets cost one pass.
* :func:`candidate_statistics` returns the dense ``[n_candidates, n_stats]``
  statistics array (``STAT_NAMES``), support rows only.
* :func:`realised_gains` fits on support rows and scores on query rows: the
  training label of the v02 network, and the label of the non-neural baselines.

Episode wrappers: :func:`episode_statistics`, :func:`episode_realised_gains`
(``learned_search.episodes.Episode``; uses ``ep.nuisance`` and ``ep.nb6`` when
present).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

N_RFF = 16
ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
RFF_SEED = 20261021
N_COMPOSITION = 8

STAT_NAMES: tuple[str, ...] = (
    "cv_gain",              # pooled LOPO general gain of the single (RFF ridge)
    "cv_gain_linear",       # the same with the linear term only
    "lopo_mean_gain",       # mean over held-out patients of the per-patient gain
    "lopo_var_gain",        # variance over held-out patients of the per-patient gain
    "lopo_frac_positive",   # fraction of held-out patients with positive gain
    "abs_r",                # |pooled within-patient correlation| with the target
    "within_between_share",  # (mean per-patient r)^2 / mean per-patient r^2: shared vs patient-specific
    "moran_candidate",      # Moran's I of the candidate (mean over sections)
    "moran_target",         # Moran's I of the target (same for every candidate)
    "abs_r_depth",          # |correlation| with log depth
    "r2_composition",       # R^2 of the candidate on the composition axes
    "t_own", "t_nbr6", "t_nbr18", "t_contrast",   # spatial transform of the candidate
    "is_morph",             # morphology (1) or molecular (0) variable
)


# ------------------------------------------------------------------ helpers
def _dev(device) -> torch.device:
    return torch.device(device) if device is not None else torch.device("cpu")


def _codes(groups: np.ndarray, n_pseudo: int = 5, seed: int = 0) -> np.ndarray:
    """Patient codes 0..P-1; with fewer than 3 patients, random pseudo-patient folds."""
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    if inv.max() + 1 >= 3:
        return inv.astype(np.int64)
    return (np.random.default_rng(seed).permutation(len(inv)) % n_pseudo).astype(np.int64)


def _rff_map(k: int, n_rff: int, seed: int = RFF_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000 * k)
    return rng.normal(size=(k, n_rff * k)), rng.uniform(0, 2 * np.pi, size=n_rff * k)


def _features(V: torch.Tensor, n_rff: int, linear_only: bool) -> torch.Tensor:
    """(n, m, k) standardised values -> (n, m, F) design [rff | linear | 1]."""
    one = torch.ones(V.shape[:2] + (1,), dtype=V.dtype, device=V.device)
    if linear_only:
        return torch.cat([V, one], 2)
    om, ph = _rff_map(V.shape[2], n_rff)
    om = torch.as_tensor(om, dtype=V.dtype, device=V.device)
    ph = torch.as_tensor(ph, dtype=V.dtype, device=V.device)
    R = np.sqrt(2.0 / om.shape[1]) * torch.cos(V @ om + ph)
    return torch.cat([R, V, one], 2)


def nuisance_residual(ys: np.ndarray, nuis_s: np.ndarray | None, yq: np.ndarray | None = None,
                      nuis_q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Residual of y after OLS on [1, nuisance] fitted on support rows (support, query)."""
    ys = np.asarray(ys, dtype=np.float64)
    Ds = np.ones((len(ys), 1)) if nuis_s is None else np.column_stack([np.ones(len(ys)), np.asarray(nuis_s, float)])
    beta = np.linalg.lstsq(Ds, ys, rcond=None)[0]
    rs = ys - Ds @ beta
    if yq is None:
        return rs, None
    yq = np.asarray(yq, dtype=np.float64)
    Dq = np.ones((len(yq), 1)) if nuis_q is None else np.column_stack([np.ones(len(yq)), np.asarray(nuis_q, float)])
    return rs, yq - Dq @ beta


# ------------------------------------------------------------------ the fit
def _fit_dtype(dev: torch.device) -> torch.dtype:
    """float32 on the GPU (TF32 off inside the fit), float64 on the CPU."""
    return torch.float32 if dev.type == "cuda" else torch.float64


def _as3(V, dev) -> torch.Tensor:
    V = torch.as_tensor(V, device=dev).to(_fit_dtype(dev))
    return V[:, :, None] if V.ndim == 2 else V


def fit_sets(Vs, ys: np.ndarray, groups: np.ndarray, *, nuis_s: np.ndarray | None = None,
             Vq=None, yq: np.ndarray | None = None, nuis_q: np.ndarray | None = None,
             n_rff: int = N_RFF, alphas: Sequence[float] = ALPHAS, linear_only: bool = False,
             device=None, chunk: int | None = None) -> dict:
    """Fit ``m`` sets of ``k`` variables each (batched torch; one Gram pass, then solves).

    ``Vs`` (n_s, m, k) support values (``Vq`` (n_q, m, k) query values; numpy or
    torch), ``ys``/``yq`` targets centred within patient, ``groups`` support patient
    labels.  Returns ``cv_gain`` (m,) pooled LOPO general gain, ``lopo_gain`` (P, m)
    per held-out patient, ``alpha`` (m,) chosen penalty, and ``query_gain`` (m,) when
    query rows are given (fit on all support rows with the chosen penalty)."""
    dev = _dev(device)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _fit_sets(Vs, ys, groups, nuis_s, Vq, yq, nuis_q, n_rff, alphas, linear_only, dev, chunk)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _fit_sets(Vs, ys, groups, nuis_s, Vq, yq, nuis_q, n_rff, alphas, linear_only, dev, chunk) -> dict:
    dt = _fit_dtype(dev)
    codes = _codes(groups)
    order = np.argsort(codes, kind="stable")              # patients as contiguous row blocks
    bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    P = len(bounds) - 1
    rs, rq = nuisance_residual(ys, nuis_s, yq, nuis_q)
    ys = np.asarray(ys, dtype=np.float64)
    sst_p = np.array([np.sum((ys[codes == p] - ys[codes == p].mean()) ** 2) for p in range(P)])
    sst_t = torch.as_tensor(np.maximum(sst_p, 1e-12), device=dev, dtype=dt)
    Vs = _as3(Vs, dev)[torch.as_tensor(order, device=dev)]
    n_s, m, k = Vs.shape
    r_t = torch.as_tensor(rs[order], device=dev, dtype=dt)
    rr_t = torch.stack([(r_t[bounds[p]:bounds[p + 1]] ** 2).sum() for p in range(P)])
    mu, sd = Vs.mean(0), Vs.std(0, unbiased=False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    F = (k if linear_only else (n_rff + 1) * k) + 1
    chunk = chunk or max(4, int(6e7 // max(1, n_s * F)))
    pen = torch.ones(F, dtype=dt, device=dev)
    pen[-1] = 1e-6
    eye = torch.diag(pen)
    cv, lopo_all, alpha_all, qg = [], [], [], []
    if Vq is not None:
        Vq = _as3(Vq, dev)
        sst_q = max(float(np.sum((np.asarray(yq) - np.mean(yq)) ** 2)), 1e-12)
        rq_t = torch.as_tensor(rq, device=dev, dtype=dt)
    al = torch.as_tensor(list(alphas), dtype=dt, device=dev)
    for a0 in range(0, m, chunk):
        sl = slice(a0, min(m, a0 + chunk))
        Phi = _features((Vs[:, sl] - mu[sl]) / sd[sl], n_rff, linear_only)
        G = torch.stack([torch.einsum("nmf,nmg->mfg", Phi[bounds[p]:bounds[p + 1]], Phi[bounds[p]:bounds[p + 1]])
                         for p in range(P)])                                           # (P, mc, F, F)
        b = torch.stack([torch.einsum("nmf,n->mf", Phi[bounds[p]:bounds[p + 1]], r_t[bounds[p]:bounds[p + 1]])
                         for p in range(P)])                                           # (P, mc, F)
        del Phi
        Gt, bt = G.sum(0), b.sum(0)
        A = (Gt[None, None] - G[None]) + al[:, None, None, None, None] * eye           # (A, P, mc, F, F)
        beta = torch.linalg.solve(A, (bt[None] - b)[None, ..., None].expand(len(al), -1, -1, -1, -1))[..., 0]
        sse = rr_t[None, :, None] - 2 * (beta * b[None]).sum(-1) + torch.einsum("apmf,pmfg,apmg->apm", beta, G, beta)
        lopo = rr_t[None, :, None] - sse                                               # (A, P, mc)
        best = lopo.sum(1).argmax(0)                                                   # (mc,)
        lb = lopo.gather(0, best[None, None, :].expand(1, P, -1))[0]                   # (P, mc)
        cv.append(lb.sum(0) / sst_t.sum())
        lopo_all.append(lb / sst_t[:, None])
        alpha_all.append(al[best])
        if Vq is not None:
            bfull = torch.linalg.solve(Gt + al[best][:, None, None] * eye, bt[..., None])[..., 0]
            pred = torch.einsum("nmf,mf->nm", _features((Vq[:, sl] - mu[sl]) / sd[sl], n_rff, linear_only), bfull)
            qg.append(((rq_t ** 2).sum() - ((rq_t[:, None] - pred) ** 2).sum(0)) / sst_q)
    f64 = lambda t: t.double().cpu().numpy()  # noqa: E731
    return {"cv_gain": f64(torch.cat(cv)), "lopo_gain": f64(torch.cat(lopo_all, 1)), "alpha": f64(torch.cat(alpha_all)),
            "query_gain": None if Vq is None else f64(torch.cat(qg))}


# ------------------------------------------------------------- statistics
def _onehot(codes: np.ndarray, dev, dtype=torch.float64) -> torch.Tensor:
    c = torch.as_tensor(codes, device=dev)
    return torch.nn.functional.one_hot(c, int(codes.max()) + 1).to(dtype).T          # (P, n)


def moran_columns(Z, nb6: np.ndarray, sections: np.ndarray, *, device=None) -> np.ndarray:
    """Moran's I of every column of ``Z`` (n, p), mean over sections with >= 7 rows:
    column-batched ``hest_multi.section_moran_i`` (identical per section), torch."""
    dev = _dev(device)
    Z = torch.as_tensor(Z, dtype=torch.float64, device=dev)
    n, p = Z.shape
    _, sec = np.unique(np.asarray(sections).astype(str), return_inverse=True)
    S = _onehot(sec, dev)
    size = S.sum(1)
    Zc = Z - (S.T @ ((S @ Z) / size.clamp(min=1)[:, None]))
    nb = np.asarray(nb6, dtype=np.int64)
    valid = (nb >= 0) & (nb < n)
    valid &= np.where(valid, sec[np.clip(nb, 0, n - 1)] == sec[:, None], False)
    cnt = valid.sum(1)
    nbt = torch.as_tensor(np.where(valid, nb, 0), device=dev)
    vt = torch.as_tensor(valid, device=dev, dtype=torch.float64)
    lag = torch.zeros_like(Zc)
    for j in range(nb.shape[1]):
        lag += Zc[nbt[:, j]] * vt[:, j:j + 1]
    lag /= torch.as_tensor(np.maximum(cnt, 1), device=dev, dtype=torch.float64)[:, None]
    num, den = S @ (Zc * lag), S @ (Zc ** 2)
    s0 = S @ torch.as_tensor((cnt > 0).astype(float), device=dev)
    ok = ((size >= 7) & (s0 > 0))[:, None] & (den > 0)
    I = torch.where(ok, (size / s0.clamp(min=1))[:, None] * num / torch.where(ok, den, torch.ones_like(den)),
                    torch.zeros_like(num))
    n_ok = ok.sum(0)
    return torch.where(n_ok > 0, I.sum(0) / n_ok.clamp(min=1), torch.zeros_like(I[0])).cpu().numpy()


def _within_t(Z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return Z - S.T @ ((S @ Z) / S.sum(1).clamp(min=1)[:, None])


def _within(Z: np.ndarray, codes: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    out = Z.copy()
    for g in np.unique(codes):
        r = codes == g
        out[r] -= out[r].mean(0)
    return out


def composition_axes(X: np.ndarray, view: np.ndarray | None, transform: np.ndarray | None, n_s: int,
                     n_axes: int = N_COMPOSITION) -> np.ndarray | None:
    """Composition proxy: the top support principal axes of the own-spot programme scores."""
    if view is None or transform is None:
        return None
    from .episodes import VIEW_VOCAB
    cols = np.flatnonzero((np.asarray(view) == VIEW_VOCAB.index("programme")) & (np.asarray(transform) == 0))
    if len(cols) < 2:
        return None
    A = np.asarray(X[:, cols], dtype=np.float64)
    mu = A[:n_s].mean(0)
    w, v = np.linalg.eigh(np.cov((A[:n_s] - mu).T))
    return (A - mu) @ v[:, ::-1][:, :min(n_axes, len(w))]


def _statistics(Xs, ys, groups, *, nb6=None, nuisance=None, composition=None, transform=None, view=None,
                device=None, Xq=None, yq=None, nuis_q=None) -> tuple[np.ndarray, list, np.ndarray | None]:
    dev = _dev(device)
    ys = np.asarray(ys, dtype=np.float64)
    Xt = torch.as_tensor(np.asarray(Xs), dtype=torch.float64, device=dev)
    n, p = Xt.shape
    codes = _codes(groups)
    out = torch.zeros((p, len(STAT_NAMES)), dtype=torch.float64, device=dev)
    col = {s: i for i, s in enumerate(STAT_NAMES)}
    rff = fit_sets(Xt, ys, groups, nuis_s=nuisance, Vq=Xq, yq=yq, nuis_q=nuis_q, device=dev)
    lin = fit_sets(Xt, ys, groups, nuis_s=nuisance, linear_only=True, alphas=(1.0,), device=dev)
    lg = torch.as_tensor(rff["lopo_gain"], device=dev)
    out[:, col["cv_gain"]] = torch.as_tensor(rff["cv_gain"], device=dev)
    out[:, col["cv_gain_linear"]] = torch.as_tensor(lin["cv_gain"], device=dev)
    out[:, col["lopo_mean_gain"]] = lg.mean(0)
    out[:, col["lopo_var_gain"]] = lg.var(0, unbiased=False)
    out[:, col["lopo_frac_positive"]] = (lg > 0).double().mean(0)
    S = _onehot(codes, dev)
    Zc = _within_t(Xt, S)
    yc = _within_t(torch.as_tensor(ys, device=dev)[:, None], S)[:, 0]
    zz = (Zc ** 2).sum(0)
    zn = zz.sqrt() + 1e-12
    out[:, col["abs_r"]] = (Zc.T @ yc).abs() / (zn * (yc.norm() + 1e-12))
    rp = (S @ (Zc * yc[:, None])) / ((S @ Zc ** 2) * (S @ yc ** 2)[:, None]).sqrt().clamp(min=1e-12)   # (P, p)
    out[:, col["within_between_share"]] = rp.mean(0) ** 2 / ((rp ** 2).mean(0) + 1e-12)
    if nb6 is not None:
        out[:, col["moran_candidate"]] = torch.as_tensor(moran_columns(Xt, nb6, codes, device=dev), device=dev)
        out[:, col["moran_target"]] = float(moran_columns(ys[:, None], nb6, codes, device=dev)[0])
    if nuisance is not None:
        d = _within_t(torch.as_tensor(np.asarray(nuisance, dtype=np.float64), device=dev)[:, None], S)[:, 0]
        out[:, col["abs_r_depth"]] = (Zc.T @ d).abs() / (zn * (d.norm() + 1e-12))
    if composition is not None and np.asarray(composition).shape[1]:
        C = _within_t(torch.as_tensor(np.asarray(composition, dtype=np.float64), device=dev), S)
        Q, R = torch.linalg.qr(C)
        dg = R.diagonal().abs()
        Q = Q[:, dg > 1e-8 * max(1.0, float(dg.max()))]
        out[:, col["r2_composition"]] = ((Q.T @ Zc) ** 2).sum(0) / zn ** 2
    if transform is not None:
        t = torch.as_tensor(np.asarray(transform, dtype=np.int64), device=dev)
        out[torch.arange(p, device=dev), col["t_own"] + t] = 1.0
    if view is not None:
        from .episodes import VIEW_VOCAB
        out[:, col["is_morph"]] = torch.as_tensor(np.isin(np.asarray(view), [VIEW_VOCAB.index("morphology"),
                                                                              VIEW_VOCAB.index("concept")]),
                                                  dtype=torch.float64, device=dev)
    res = np.nan_to_num(out.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return res, list(STAT_NAMES), rff["query_gain"]


def candidate_statistics(Xs: np.ndarray, ys: np.ndarray, groups: np.ndarray, *, nb6: np.ndarray | None = None,
                         nuisance: np.ndarray | None = None, composition: np.ndarray | None = None,
                         transform: np.ndarray | None = None, view: np.ndarray | None = None,
                         device=None) -> tuple[np.ndarray, list]:
    """Dense ``[n_candidates, len(STAT_NAMES)]`` float32 statistics on SUPPORT rows only.

    ``Xs`` (n_s, p) candidate values (every spatial version is its own column),
    ``ys`` target, ``groups`` patients, ``nb6`` (n_s, 6) support-local neighbours
    (-1 padded), ``nuisance`` (n_s,) log depth, ``composition`` (n_s, c),
    ``transform`` (p,) index into own/nbr6/nbr18/contrast, ``view`` (p,) index
    into ``episodes.VIEW_VOCAB``.  Batched torch on ``device`` (CPU by default)."""
    S, names, _ = _statistics(Xs, ys, groups, nb6=nb6, nuisance=nuisance, composition=composition,
                              transform=transform, view=view, device=device)
    return S, names


# ---------------------------------------------------------- realised gains
def realised_gains(X: np.ndarray, y: np.ndarray, n_support: int, groups: np.ndarray, *,
                   nuisance: np.ndarray | None = None, singles: bool | Sequence[int] = True,
                   pairs: Sequence[Sequence[int]] | None = None, sets: Sequence[Sequence[int]] | None = None,
                   device=None) -> dict:
    """Fit on support rows ``[:n_support]``, score the general gain on query rows.

    Returns ``{"single": (p,) or (len(singles),), "pair": (len(pairs),), "sets": [...],
    "single_cv": support LOPO gain}``; ``sets`` may mix sizes (grouped by size)."""
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    s = int(n_support)
    g = np.asarray(groups)[:s]
    nz = None if nuisance is None else np.asarray(nuisance, dtype=np.float64)
    kw = dict(nuis_s=None if nz is None else nz[:s], nuis_q=None if nz is None else nz[s:], device=device)
    out: dict = {}
    if singles is not False:
        idx = np.arange(X.shape[1]) if singles is True else np.asarray(singles, dtype=int)
        r = fit_sets(X[:s][:, idx], y[:s], g, Vq=X[s:][:, idx], yq=y[s:], **kw)
        out["single"], out["single_cv"] = r["query_gain"], r["cv_gain"]
    if pairs is not None and len(pairs):
        pi = np.asarray(pairs, dtype=int)
        r = fit_sets(X[:s][:, pi], y[:s], g, Vq=X[s:][:, pi], yq=y[s:], **kw)
        out["pair"], out["pair_cv"] = r["query_gain"], r["cv_gain"]
    if sets is not None and len(sets):
        q, cv = np.zeros(len(sets)), np.zeros(len(sets))
        by_k: dict = {}
        for i, st in enumerate(sets):
            by_k.setdefault(len(st), []).append(i)
        for k, ids in by_k.items():
            si = np.asarray([list(sets[i]) for i in ids], dtype=int)
            r = fit_sets(X[:s][:, si], y[:s], g, Vq=X[s:][:, si], yq=y[s:], **kw)
            q[ids], cv[ids] = r["query_gain"], r["cv_gain"]
        out["sets"], out["sets_cv"] = q, cv
    return out


def support_cv_gains(Xs: np.ndarray, ys: np.ndarray, groups: np.ndarray, sets: Sequence[Sequence[int]], *,
                     nuisance: np.ndarray | None = None, device=None) -> np.ndarray:
    """Support LOPO general gain of arbitrary sets (mixed sizes): the search's refit score."""
    cv = np.zeros(len(sets))
    by_k: dict = {}
    for i, st in enumerate(sets):
        by_k.setdefault(len(st), []).append(i)
    Xs = np.asarray(Xs)
    for k, ids in by_k.items():
        si = np.asarray([list(sets[i]) for i in ids], dtype=int)
        cv[ids] = fit_sets(Xs[:, si], ys, groups, nuis_s=nuisance, device=device)["cv_gain"]
    return cv


# ------------------------------------------------------------ episode API
def _episode_parts(ep):
    if ep.y is None:
        raise ValueError("v02 episodes need a measured target")
    s = int(ep.n_support)
    nb = getattr(ep, "nb6", None)
    nbs = None
    if nb is not None:
        nb = np.asarray(nb, dtype=np.int64)[:s]
        nbs = np.where(nb < s, nb, -1)
    return s, nbs, getattr(ep, "nuisance", None)


def episode_statistics(ep, *, device=None) -> tuple[np.ndarray, list]:
    """``candidate_statistics`` of an ``Episode`` (support rows only)."""
    s, nbs, nz = _episode_parts(ep)
    comp = composition_axes(ep.X, ep.meta.view, ep.meta.transform, s)
    return candidate_statistics(ep.X[:s], ep.y[:s], ep.groups[:s], nb6=nbs,
                                nuisance=None if nz is None else np.asarray(nz)[:s],
                                composition=None if comp is None else comp[:s],
                                transform=ep.meta.transform, view=ep.meta.view, device=device)


def episode_realised_gains(ep, pairs: Sequence[Sequence[int]] | None = None, *, singles: bool = True,
                           sets=None, device=None) -> dict:
    """``realised_gains`` of an ``Episode``: fit on its support rows, score on its query rows."""
    _, _, nz = _episode_parts(ep)
    return realised_gains(ep.X, ep.y, ep.n_support, ep.groups, nuisance=nz, singles=singles, pairs=pairs,
                          sets=sets, device=device)


def episode_statistics_and_labels(ep, pairs_fn, *, device=None) -> tuple[np.ndarray, list, dict]:
    """One pass for training: support statistics, realised query gains of every single
    (from the same Gram matrices), then of ``pairs_fn(S, stat_names)`` pairs."""
    s, nbs, nz = _episode_parts(ep)
    comp = composition_axes(ep.X, ep.meta.view, ep.meta.transform, s)
    nzq = None if nz is None else np.asarray(nz)[s:]
    S, sn, q = _statistics(ep.X[:s], ep.y[:s], ep.groups[:s], nb6=nbs,
                           nuisance=None if nz is None else np.asarray(nz)[:s],
                           composition=None if comp is None else comp[:s], transform=ep.meta.transform,
                           view=ep.meta.view, device=device, Xq=ep.X[s:], yq=ep.y[s:], nuis_q=nzq)
    pairs = np.asarray(pairs_fn(S, sn), dtype=int)
    lab = {"single": q, "pairs": pairs, "pair": None}
    if len(pairs):
        lab["pair"] = realised_gains(ep.X, ep.y, s, ep.groups, nuisance=nz, singles=False, pairs=pairs,
                                     device=device)["pair"]
    return S, sn, lab
