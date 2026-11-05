"""Shared numerics for DiscoveryBench generators, nulls and metrics.

Nothing here is scientific content. It is the small set of primitives every
family needs: reproducible RNG derivation, standardisation, ridge fits used by
recoverability checks, autocorrelated noise, and neighbourhood construction.
"""
from __future__ import annotations

import hashlib

import numpy as np


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
def package_source_sha256():
    """{module: sha256} for every generator module in the package.

    Used by `oracle.run_gate` to stamp which generators a gate report was
    produced against. A manifest can bind a gate report by hash and still be
    wrong about it: the report may have been produced before the generators it
    is filed against changed. That happened -- a manifest was written whose
    generators all matched the tree and whose bound gate predated two of the
    changes in it -- and nothing in the document said so, because every
    ordering check compared file times rather than content lineage.

    Deliberately stdlib-only and importing nothing from the package, so it can
    be called from anywhere without a cycle.
    """
    import hashlib as _h
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    out = {}
    for fn in sorted(_os.listdir(here)):
        if fn.endswith(".py"):
            with open(_os.path.join(here, fn), "rb") as fh:
                out[fn] = _h.sha256(fh.read()).hexdigest()
    return out


def derive_seed(*parts) -> int:
    """A 63-bit seed derived from arbitrary strings/ints.

    Generators must never reuse a parent seed for a child stream: a
    replication environment that reuses the proposal noise stream is not an
    independent environment, it is the same data with a different label.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


# --------------------------------------------------------------------------
# elementary transforms
# --------------------------------------------------------------------------
def zscore(x, axis=0, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def soft_threshold(a, tau, temp=0.15):
    """sigma((a - tau)/T): the differentiable THRESHOLD operator."""
    return sigmoid((np.asarray(a, dtype=np.float64) - tau) / temp)


def hill(a, K=1.0, nexp=3.0):
    """a^n / (K^n + a^n): the SATURATE operator. Requires a >= 0."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    an = a ** nexp
    return an / (K ** nexp + an)


def orthonormal(rng, d, k):
    """(d, k) with orthonormal columns, without calling LAPACK.

    `np.linalg.qr` was the second reproducibility hazard in the generator and
    the more serious one, because it is not merely thread-dependent: measured
    on two machines at one thread each, the same seed gave different Householder
    factors under different LAPACK builds, and since every family's observation
    model goes through this call, *all 115* frozen worlds disagreed between the
    two. A manifest nobody else's machine can reproduce cannot show that a
    benchmark was frozen before training.

    Modified Gram-Schmidt over the first `k` columns replaces it. Every step is
    an elementwise product and a `np.sum` -- numpy's own pairwise reduction, not
    a BLAS dot -- so the result depends on nothing but the input. The same
    `(d, m)` Gaussian block is drawn as before, so the generator's stream
    position is unchanged.

    The basis this returns is not the one QR returned; it is a different
    orthonormal basis of the same random subspace, which is all any caller
    needs. World fingerprints therefore moved when this landed, and the frozen
    manifest records that.
    """
    m = max(d, k)
    if k > d:
        raise ValueError(
            "orthonormal needs k <= d; got d=%d, k=%d. No world in the "
            "package asks for more columns than the observed dimension." % (d, k))
    A = rng.standard_normal((d, m))
    cols = []
    for j in range(m):
        if len(cols) >= k:
            break
        v = np.array(A[:, j], dtype=np.float64)
        for u in cols:                       # fixed order: deterministic
            v = v - float(np.sum(v * u)) * u
        for u in cols:                       # one reorthogonalisation pass
            v = v - float(np.sum(v * u)) * u
        nrm = float(np.sqrt(np.sum(v * v)))
        if nrm > 1e-10:
            cols.append(v / nrm)
    if len(cols) < k:                        # pragma: no cover
        raise ValueError("could not build %d independent columns" % k)
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# fits used by recoverability / oracle checks
# --------------------------------------------------------------------------
def ridge_r2(X, y, alpha=1.0, folds=2, seed=0):
    """Cross-fitted ridge R^2 of y on X.

    Cross-fitted rather than in-sample because an in-sample R^2 with d
    comparable to n is close to 1 for any y and would certify anything as
    recoverable.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % folds
    pred = np.empty(n)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if tr.sum() < 2 or te.sum() < 1:
            return float("nan")
        mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-9)
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ (y[tr] - ym))
        pred[te] = Xte @ w + ym
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def partial_out(y, controls):
    """Residual of y after OLS removal of `controls` (list of 1-D arrays)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    cols = [np.ones(len(y))]
    for c in controls:
        c = np.asarray(c, dtype=np.float64)
        if c.ndim == 1:
            cols.append(c)
        else:
            cols.extend(c.T)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr(y, x, controls):
    """Correlation of y and x after removing `controls` from both."""
    ry = partial_out(y, controls)
    rx = partial_out(x, controls)
    sy, sx = ry.std(), rx.std()
    if sy < 1e-12 or sx < 1e-12:
        return 0.0
    return float(np.corrcoef(ry, rx)[0, 1])


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------
# structured noise
# --------------------------------------------------------------------------
def ar1(n, rho, rng, scale=1.0):
    """AR(1) series with stationary unit variance, scaled by `scale`."""
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1.0 - rho ** 2) * e[t]
    return scale * x


def seasonal(n, period, rng, n_harm=2, scale=1.0):
    """A deterministic seasonal cycle with random phases."""
    t = np.arange(n)
    out = np.zeros(n)
    for h in range(1, n_harm + 1):
        ph = rng.uniform(0, 2 * np.pi)
        out += np.cos(2 * np.pi * h * t / period + ph) / h
    return scale * out / max(out.std(), 1e-9)


def _lattice_shape(coords):
    """(nx, ny) if `coords` is exactly a row-major rectangular lattice, else None.

    Row-major means `np.meshgrid(xs, ys, indexing="ij")` raveled -- what
    `grid_coords` produces and what every spatial world in the package passes.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return None
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    if len(xs) * len(ys) != len(coords):
        return None
    want = np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])
    if not np.array_equal(coords, want):
        return None
    # the sampler below assumes unit spacing on each axis
    if len(xs) > 1 and not np.allclose(np.diff(xs), 1.0):
        return None
    if len(ys) > 1 and not np.allclose(np.diff(ys), 1.0):
        return None
    return int(len(xs)), int(len(ys))


def _periodic_gaussian_taps(n, sd):
    """A length-`n` circulant kernel whose autocorrelation is the SE kernel.

    Convolving white noise with a Gaussian of standard deviation s gives a
    field whose covariance is a Gaussian of standard deviation s*sqrt(2), so
    the taps use `sd / sqrt(2)`. Periodic images are summed in, so the kernel
    is exactly circulant even when the grid is only a few standard deviations
    wide, and it is normalised so the field has unit variance.
    """
    s = float(sd) / np.sqrt(2.0)
    j = np.arange(n, dtype=np.float64)
    j = np.minimum(j, n - j)                     # distance on the ring
    reach = int(np.ceil(4.0 * s / max(n, 1))) + 1
    g = np.zeros(n, dtype=np.float64)
    for m in range(-reach, reach + 1):           # fixed order: deterministic
        g = g + np.exp(-0.5 * ((j + m * n) / s) ** 2)
    nrm = float(np.sqrt((g * g).sum()))
    return g / nrm if nrm > 0 else g


def _circular_smooth(field, taps, axis):
    """Circulant convolution by explicit shifts, accumulated in a fixed order."""
    out = np.zeros_like(field)
    for k in range(len(taps)):
        w = taps[k]
        if w != 0.0:
            out = out + w * np.roll(field, k, axis=axis)
    return out


def gp_field(coords, rng, length_scale=3.0, jitter=1e-6):
    """A draw from a squared-exponential GP at `coords`.

    This is what makes a spatial world's nuisance genuinely autocorrelated,
    which is the statistic the spatial structured null has to preserve.

    Bit-reproducibility
    -------------------
    The original implementation formed the dense kernel, took
    `np.linalg.cholesky` and applied `L @ w`. Both of those are LAPACK/BLAS
    calls, and both were measured on this machine to return different bits at
    different thread counts: the Cholesky at every size tried (256, 1156 and
    1600), and the mat-vec at 1156, where OpenBLAS splits the reduction
    dimension by thread count and returns three distinct results at one, four
    and eight threads. That is why `boss3_hill_space_cause` produced three
    different world fingerprints and `spatial_interaction` and
    `boss1_cond_thresh_space_lag` two: a benchmark world's identity depended on
    an environment variable, which is exactly what a frozen manifest cannot
    tolerate.

    On a regular lattice -- which is what every spatial world in the package
    passes -- the same field is drawn instead by circulant convolution of white
    noise with a Gaussian kernel. Only elementwise multiplies, `np.roll` and
    additions in a fixed order are used, so the result is bitwise identical at
    any thread count and under any BLAS build, and the covariance is exactly
    the squared-exponential kernel rather than an approximation of it.

    The field is now also exactly periodic on the lattice, which removes one
    contributor to the wrap seam that `nulls.null_torus_shift` measures -- both
    that null and the permutation reference in `oracle._torus_index` treat the
    lattice as a torus. It does not remove the seam: the response is not the
    field alone, and `radius_neighbors` still builds a Euclidean, non-periodic
    neighbourhood graph in which edge cells have fewer neighbours. Measured
    Moran's I relative change under a torus shift is 0.04 to 0.11 across the
    three spatial worlds, against the 0.15 tolerance -- lower than before but
    not zero, and the null generator still measures it rather than assuming
    it.

    `jitter` is retained in the signature and is unused on the lattice path;
    there is no factorisation to regularise. Off-lattice coordinates fall back
    to the dense Cholesky, which is *not* thread-stable -- no world in the
    package takes that path, and `tests/test_determinism.py` asserts it.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    shape = _lattice_shape(coords)
    if shape is None:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        K = np.exp(-0.5 * d2 / length_scale ** 2) + jitter * np.eye(n)
        return np.linalg.cholesky(K) @ rng.standard_normal(n)
    nx, ny = shape
    # exactly as many draws as the dense path consumed, so every downstream
    # stream in the generator stays at the same position
    w = rng.standard_normal(n).reshape(nx, ny)
    f = _circular_smooth(w, _periodic_gaussian_taps(nx, length_scale), axis=0)
    f = _circular_smooth(f, _periodic_gaussian_taps(ny, length_scale), axis=1)
    return f.reshape(n)


def grid_coords(side):
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])


def knn_neighbors(coords, k=6, groups=None):
    """Row indices of the k nearest neighbours (self excluded).

    `groups`, if given, is a (n,) array of group ids (e.g. patient/sample
    identity): neighbours are then computed independently WITHIN each group,
    so a spatial neighbourhood can never cross a patient boundary -- the
    same guarantee `Substrate.coord_groups` already gives the injection
    path, extended here to the discovery-episode neighbour graph. Returned
    indices are always global row positions, and a group of size <= 1 gets
    an empty neighbour list rather than an error.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if groups is None:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        idx = np.argsort(d2, axis=1)[:, :k]
        return [idx[i] for i in range(n)]
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("groups must align with coords")
    out: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        m = len(rows)
        kk = min(k, m - 1)
        if kk <= 0:
            for r in rows:
                out[r] = np.array([], dtype=np.int64)
            continue
        sub = coords[rows]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        order = np.argsort(d2, axis=1)[:, :kk]
        for local_i, r in enumerate(rows):
            out[r] = rows[order[local_i]]
    return out


def radius_neighbors(coords, radius=1.5):
    coords = np.asarray(coords, dtype=np.float64)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return [np.where(d2[i] <= radius ** 2)[0] for i in range(len(coords))]


def morans_i(x, neighbors):
    """Moran's I: the spatial-autocorrelation statistic the spatial null
    must preserve. Reported, not assumed."""
    x = np.asarray(x, dtype=np.float64).ravel()
    z = x - x.mean()
    n = len(x)
    num, w_sum = 0.0, 0.0
    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        num += float(z[i] * z[nb].sum())
        w_sum += len(nb)
    denom = float((z ** 2).sum())
    if w_sum == 0 or denom < 1e-12:
        return 0.0
    return (n / w_sum) * (num / denom)


def acf(x, max_lag=10):
    """Sample autocorrelation, lags 1..max_lag."""
    x = np.asarray(x, dtype=np.float64).ravel()
    z = x - x.mean()
    v = float((z ** 2).sum())
    if v < 1e-12:
        return np.zeros(max_lag)
    return np.array([float((z[:-l] * z[l:]).sum()) / v
                     for l in range(1, max_lag + 1)])


def power_spectrum(x):
    return np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) - np.mean(x))) ** 2


# --------------------------------------------------------------------------
# multiplicity
# --------------------------------------------------------------------------
def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=np.float64)
    ok = np.isfinite(p)
    m = int(ok.sum())
    rej = np.zeros(len(p), dtype=bool)
    q = np.full(len(p), np.nan)
    if m == 0:
        return rej, q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ps = p[order]
    qs = ps * m / np.arange(1, m + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    q[order] = np.minimum(qs, 1.0)
    below = np.where(ps <= alpha * np.arange(1, m + 1) / m)[0]
    if len(below):
        rej[order[:below[-1] + 1]] = True
    return rej, q
