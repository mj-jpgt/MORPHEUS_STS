"""Residual molecular programmes (spec 0.4 item 4; the original paper's procedure, RNA only).

1. Ridge of log expression (HVGs) on the declared covariates (log depth, composition, Gavish
   meta-programme scores), penalty by donor-grouped inner CV (one penalty for all genes).
2. Truncated SVD of the residual; rank by Horn's parallel analysis (50 column permutations,
   95th percentile, stop at the first non-retained component, ceiling 20).
3. Loadings aligned across folds by cosine assignment (``align``).
Each programme is a signed gene loading vector; its per-spot score is the residual projected
on the loading.
"""
from __future__ import annotations

import numpy as np

K_MAX = 20
N_PERM = 50
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _grouped_ridge_alpha(Y: np.ndarray, C: np.ndarray, groups: np.ndarray, alphas=ALPHAS, n_folds: int = 5) -> float:
    pats = np.unique(groups)
    fold = {p: i % min(n_folds, len(pats)) for i, p in enumerate(pats)}
    f = np.array([fold[g] for g in groups])
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        sse = 0.0
        for k in np.unique(f):
            tr, te = f != k, f == k
            beta = np.linalg.solve(C[tr].T @ C[tr] + a * np.eye(C.shape[1]), C[tr].T @ Y[tr])
            sse += float(((Y[te] - C[te] @ beta) ** 2).sum())
        if sse < best:
            best, best_a = sse, a
    return best_a


def _top_eigs(M: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.utils.extmath import randomized_svd
    k = min(k, M.shape[0] - 1, M.shape[1])
    return randomized_svd(M, n_components=k, random_state=seed)


def parallel_analysis_rank(R: np.ndarray, *, k_max: int = K_MAX, n_perm: int = N_PERM, seed: int = 42) -> int:
    _, s, _ = _top_eigs(R, k_max, 0)
    real = s ** 2
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(real)))
    for d in range(n_perm):
        Pm = np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])])
        _, sn, _ = _top_eigs(Pm, k_max, 0)
        null[d, :len(sn)] = sn[:len(real)] ** 2
    p95 = np.percentile(null, 95, axis=0)
    k = 0
    while k < len(real) and real[k] > p95[k]:
        k += 1
    return k


def fit(Y: np.ndarray, C: np.ndarray, groups: np.ndarray, *, fit_rows: np.ndarray | None = None,
        k_max: int = K_MAX, n_perm: int = N_PERM, seed: int = 42) -> dict:
    """``Y`` (n, g) log expression of the HVGs, ``C`` (n, c) covariates, ``groups`` donors.

    Fitted on ``fit_rows`` (default all); returns ``loadings`` (k, g), ``scores`` (n, k) for every
    row, the covariate ridge (``beta``, ``alpha``) and the standardisation used."""
    Y = np.asarray(Y, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    rows = np.arange(len(Y)) if fit_rows is None else np.asarray(fit_rows)
    ym, cm = Y[rows].mean(0), C[rows].mean(0)
    cs = C[rows].std(0)
    cs[cs < 1e-8] = 1.0
    Cz = np.column_stack([(C - cm) / cs])
    Yc = Y - ym
    a = _grouped_ridge_alpha(Yc[rows], Cz[rows], np.asarray(groups)[rows])
    beta = np.linalg.solve(Cz[rows].T @ Cz[rows] + a * np.eye(Cz.shape[1]), Cz[rows].T @ Yc[rows])
    R = Yc - Cz @ beta
    k = parallel_analysis_rank(R[rows], k_max=k_max, n_perm=n_perm, seed=seed)
    if k == 0:
        return {"loadings": np.zeros((0, Y.shape[1])), "scores": np.zeros((len(Y), 0)), "beta": beta,
                "alpha": a, "k": 0, "y_mean": ym, "c_mean": cm, "c_sd": cs}
    _, _, vt = _top_eigs(R[rows], k, seed)
    vt = vt * np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])[:, None]   # sign: largest loading positive
    return {"loadings": vt, "scores": R @ vt.T, "beta": beta, "alpha": a, "k": int(k), "y_mean": ym,
            "c_mean": cm, "c_sd": cs}


def project(model: dict, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Scores of new rows with a frozen programme model (transport by projection)."""
    Cz = (np.asarray(C, dtype=np.float64) - model["c_mean"]) / model["c_sd"]
    R = np.asarray(Y, dtype=np.float64) - model["y_mean"] - Cz @ model["beta"]
    return R @ model["loadings"].T


def align(ref: np.ndarray, loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine assignment of ``loadings`` rows to ``ref`` rows (Hungarian on |cos|), with sign
    flips.  Returns (permutation into loadings, signs, |cos| of the matched pairs)."""
    from scipy.optimize import linear_sum_assignment
    A = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    B = loadings / np.linalg.norm(loadings, axis=1, keepdims=True)
    cos = A @ B.T
    r, c = linear_sum_assignment(-np.abs(cos))
    return c, np.sign(cos[r, c]), np.abs(cos[r, c])
