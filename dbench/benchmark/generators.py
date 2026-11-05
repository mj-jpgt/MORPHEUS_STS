"""Controlled relation generators for planted worlds and training episodes.

``plant(V, family, rng, groups=...)`` returns an UNSCALED signal g(V) for the
already-chosen variable values ``V`` (n, k) and an info dict with the
generator parameters.  g is centred within group (patient).  Scaling to a
variance fraction of a residual is the caller's job
(``dbench.benchmark.tasks.calibrate``).  For spatial families the caller
passes ``V = [own_a, nbr6_b]`` or ``[own_a, contrast_a]``.

Every family evaluates its function on per-column standardised values
(mean 0, sd 1 over all rows), so length-scales and breakpoints are in units of
the column's spread.

Training families (``TRAIN_FAMILIES``):

* ``smooth_1d``       random cubic B-spline, quantile knots, 6-8 coefficients ~ N(0, 1)
* ``smooth_2d``       tensor-product cubic B-spline (6-8 coefficients per axis) or a
                      sum of 6 random Gaussian bumps (centres at random rows)
* ``neural_k``        random MLP k -> 16 -> 16 -> 1, tanh, weights N(0, 1) * 1.5 / sqrt(fan_in)
* ``residualised_k``  neural_k (or, with probability 1/4, the product of the k columns)
                      minus its least-squares projection on every proper lower-order
                      flexible fit: for each proper non-empty subset of the variables, its
                      linear terms plus a fixed random-Fourier-feature map (48 features for
                      one variable, 96 for two), all centred within group.  k = 2: both
                      marginals are removed; k = 3: all singles and all pairs.
* ``spatial_mix``     smooth_2d or neural_2 over the two spatial columns the caller passes
* ``gated_k``          a randomly thresholded, continuous k-variable response;
                       all proper-subset flexible components are removed.  This
                       exposes conditional structure during meta-training
                       without creating an AND/XOR-specific model head.

Held-out families (evaluation only; ``HELDOUT_FAMILIES``), plus the existing
``product_2``, ``context_threshold`` and ``spatial_lag`` of ``tasks.py``:

* ``gp_k``            RBF-kernel Gaussian-process sample through 256 random Fourier
                      features with N(0, 1) output weights; length-scale ~ U[0.5, 2]
* ``piecewise_k``     2-3 random axis-aligned breakpoints per variable (random interior
                      quantiles); each cell a random constant or a random linear piece
* ``spatial_mix_gp``  gp_2 over [own_a, nbr6_b]
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

TRAIN_FAMILIES = ("smooth_1d", "smooth_2d", "neural_2", "neural_3", "neural_4", "residualised_2",
                  "residualised_3", "spatial_mix", "gated_2", "gated_3")
HELDOUT_FAMILIES = ("gp_2", "gp_3", "piecewise_2", "piecewise_3", "spatial_mix_gp")
#: families whose two columns are a spatial pair chosen by the caller
SPATIAL_FAMILIES = ("spatial_mix", "spatial_mix_gp")
FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES

_ARITY = {"smooth_1d": 1, "smooth_2d": 2, "neural_2": 2, "neural_3": 3, "neural_4": 4, "residualised_2": 2,
          "residualised_3": 3, "spatial_mix": 2, "gated_2": 2, "gated_3": 3,
          "gp_2": 2, "gp_3": 3, "piecewise_2": 2, "piecewise_3": 3,
          "spatial_mix_gp": 2,
          # existing tasks.py classes (planted there, listed here so callers can ask one place)
          "single_linear": 1, "additive_2": 2, "product_2": 2, "context_threshold": 2, "spatial_lag": 1}

#: random-Fourier-feature widths of the lower-order projection basis (per subset size)
RESIDUALISE_FEATURES = {1: 48, 2: 96, 3: 144}
GP_FEATURES = 256


def arity(family: str) -> int:
    """Number of variables the family takes."""
    try:
        return _ARITY[family]
    except KeyError:
        raise ValueError(f"unknown relation family {family!r}") from None


def sample_family(rng: np.random.Generator, families: tuple = TRAIN_FAMILIES) -> str:
    """Uniform draw of one family name (training samplers pass ``TRAIN_FAMILIES``)."""
    return str(families[int(rng.integers(len(families)))])


# ------------------------------------------------------------------ helpers
def _groups(groups, n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64) if groups is None else np.asarray(groups)


def centre_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (columns of a matrix independently)."""
    out = np.asarray(values, dtype=float).copy()
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    if out.ndim == 1:
        out -= (np.bincount(inv, weights=out) / counts)[inv]
    else:
        for j in range(out.shape[1]):
            out[:, j] -= (np.bincount(inv, weights=out[:, j]) / counts)[inv]
    return out


def _standardise(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    sd = V.std(0)
    sd[sd < 1e-12] = 1.0
    return (V - V.mean(0)) / sd


def _bspline_basis(x: np.ndarray, n_coef: int) -> np.ndarray:
    """Cubic B-spline basis (n, n_coef) with interior knots at quantiles of ``x``."""
    from scipy.interpolate import BSpline

    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        raise ValueError("constant column cannot carry a spline")
    n_interior = n_coef - 4
    inner = np.quantile(x, np.linspace(0, 1, n_interior + 2)[1:-1]) if n_interior > 0 else np.array([])
    inner = np.unique(inner[(inner > lo) & (inner < hi)])
    if len(inner) < n_interior:  # tied values (e.g. many zeros): evenly spaced knots instead
        inner = np.linspace(lo, hi, n_interior + 2)[1:-1]
    t = np.concatenate([[lo] * 4, inner, [hi] * 4])
    B = BSpline.design_matrix(np.clip(x, lo, hi), t, 3, extrapolate=False)
    return np.asarray(B.todense())


def _mlp(Z: np.ndarray, rng: np.random.Generator, width: int = 16, scale: float = 1.5) -> tuple[np.ndarray, dict]:
    k = Z.shape[1]
    W1 = rng.standard_normal((k, width)) * scale / np.sqrt(k)
    b1 = rng.standard_normal(width) * 0.5
    W2 = rng.standard_normal((width, width)) * scale / np.sqrt(width)
    b2 = rng.standard_normal(width) * 0.5
    w3 = rng.standard_normal(width) * scale / np.sqrt(width)
    g = np.tanh(np.tanh(Z @ W1 + b1) @ W2 + b2) @ w3
    return g, {"width": width, "weight_scale": scale, "W1": W1.round(4).tolist()}


def _smooth_2d(Z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    if rng.random() < 0.5:
        ca, cb = (int(c) for c in rng.integers(6, 9, size=2))
        Ba, Bb = _bspline_basis(Z[:, 0], ca), _bspline_basis(Z[:, 1], cb)
        C = rng.standard_normal((Ba.shape[1], Bb.shape[1]))
        return np.einsum("ni,ij,nj->n", Ba, C, Bb), {"form": "tensor_bspline", "n_coef": [ca, cb]}
    centres = Z[rng.choice(len(Z), size=6, replace=False)]
    widths = rng.uniform(0.5, 1.5, size=6)
    amps = rng.standard_normal(6)
    d2 = ((Z[:, None, :] - centres[None]) ** 2).sum(-1)
    g = (amps * np.exp(-0.5 * d2 / widths ** 2)).sum(1)
    return g, {"form": "gaussian_bumps", "widths": widths.round(4).tolist(), "amplitudes": amps.round(4).tolist()}


def _gp(Z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    ell = float(rng.uniform(0.5, 2.0))
    omega = rng.standard_normal((Z.shape[1], GP_FEATURES)) / ell
    phase = rng.uniform(0, 2 * np.pi, GP_FEATURES)
    w = rng.standard_normal(GP_FEATURES)
    g = np.sqrt(2.0 / GP_FEATURES) * np.cos(Z @ omega + phase) @ w
    return g, {"length_scale": ell, "n_features": GP_FEATURES}


def _piecewise(Z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n, k = Z.shape
    cell = np.zeros(n, dtype=np.int64)
    breaks, mult = [], 1
    for j in range(k):
        nb = int(rng.integers(2, 4))
        q = np.sort(rng.uniform(0.1, 0.9, size=nb))
        b = np.quantile(Z[:, j], q)
        breaks.append(b.round(4).tolist())
        cell += mult * np.searchsorted(b, Z[:, j])
        mult *= nb + 1
    n_cells = mult
    const = rng.standard_normal(n_cells)
    linear = rng.random(n_cells) < 0.5
    slopes = rng.standard_normal((n_cells, k)) * linear[:, None]
    g = const[cell] + (slopes[cell] * Z).sum(1)
    return g, {"breakpoints": breaks, "n_cells": int(n_cells), "share_linear_cells": float(linear.mean())}


def _gated(Z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Random conditional function rather than a named Boolean operator.

    Each dimension supplies a gate at a random interior quantile.  The active
    region receives an independent continuous response (linear or smooth);
    inactive regions receive a different smooth baseline.  Residualisation in
    ``plant`` subsequently removes lower-order shortcuts.
    """
    k = Z.shape[1]
    quantiles = rng.uniform(.25, .75, size=k)
    thresholds = np.asarray([np.quantile(Z[:, j], quantiles[j]) for j in range(k)])
    mask = np.all(Z > thresholds[None, :], axis=1)
    # A signed continuous function prevents the episode stream collapsing to
    # one conjunction template while still teaching conditional selection.
    w_on, w_off = rng.standard_normal(k), rng.standard_normal(k)
    curvature = rng.uniform(.15, .8, size=k)
    on = Z @ w_on + (curvature * Z ** 2).sum(1)
    off = Z @ w_off - .5 * (curvature * Z ** 2).sum(1)
    g = np.where(mask, on, off)
    return g, {"gate_quantiles": quantiles.round(4).tolist(), "active_share": float(mask.mean()),
               "form": "random_conditional_continuous"}


def lower_order_basis(Z: np.ndarray, groups: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Within-group-centred basis of every proper lower-order flexible fit of the columns of ``Z``.

    For each proper non-empty subset of the k columns: its linear terms and a
    random-Fourier-feature map (unit length-scale on standardised values;
    ``RESIDUALISE_FEATURES`` wide for the subset size)."""
    k = Z.shape[1]
    rng = np.random.default_rng(seed)
    blocks = []
    for s in range(1, k):
        for S in combinations(range(k), s):
            Zs = Z[:, S]
            m = RESIDUALISE_FEATURES[s]
            omega = rng.standard_normal((s, m))
            phase = rng.uniform(0, 2 * np.pi, m)
            blocks += [Zs, np.cos(Zs @ omega + phase)]
    return centre_within(np.column_stack(blocks), groups)


def residualise(g: np.ndarray, Z: np.ndarray, groups: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, float]:
    """``g`` minus its least-squares projection on ``lower_order_basis``; returns (residual, kept variance share)."""
    gc = centre_within(g, groups)
    D = lower_order_basis(Z, groups, seed=seed)
    U, sv, _ = np.linalg.svd(D, full_matrices=False)
    Q = U[:, sv > 1e-10 * sv.max()]  # orthonormal basis of the lower-order span
    r = gc - Q @ (Q.T @ gc)
    r = r - Q @ (Q.T @ r)  # second pass: orthogonality to round-off
    return centre_within(r, groups), float(np.var(r) / max(np.var(gc), 1e-300))


# -------------------------------------------------------------------- plant
def plant(V: np.ndarray, family: str, rng: np.random.Generator, *,
          groups: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Unscaled, within-group-centred planted signal g(V) and the generator parameters.

    ``V`` (n, k) holds the already-chosen (and, for spatial families,
    already-transformed) variable values, k == ``arity(family)``."""
    k = arity(family)
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    if V.shape[1] != k:
        raise ValueError(f"family {family!r} takes {k} variables, got {V.shape[1]}")
    if family not in FAMILIES:
        raise ValueError(f"{family!r} is planted by tasks.relation_signal, not by generators.plant")
    gr = _groups(groups, len(V))
    Z = _standardise(V)
    info: dict = {"family": family, "arity": k}
    if family == "smooth_1d":
        c = int(rng.integers(6, 9))
        B = _bspline_basis(Z[:, 0], c)
        coef = rng.standard_normal(B.shape[1])
        g = B @ coef
        info.update({"n_coef": c, "coefficients": coef.round(4).tolist()})
    elif family == "smooth_2d":
        g, extra = _smooth_2d(Z, rng)
        info.update(extra)
    elif family.startswith("neural_"):
        g, extra = _mlp(Z, rng)
        info.update(extra)
    elif family.startswith("residualised_"):
        base = "product" if rng.random() < 0.25 else "neural"
        if base == "product":
            g0 = np.prod(Z, axis=1)
        else:
            g0, extra = _mlp(Z, rng)
            info.update(extra)
        g, kept = residualise(g0, Z, gr, seed=int(rng.integers(2 ** 31)))
        info.update({"base": base, "kept_variance_share": kept})
    elif family == "spatial_mix":
        base = "smooth_2d" if rng.random() < 0.5 else "neural_2"
        if base == "smooth_2d":
            g, extra = _smooth_2d(Z, rng)
        else:
            g, extra = _mlp(Z, rng)
        info.update({"base": base, **extra})
    elif family.startswith("gated_"):
        g0, extra = _gated(Z, rng)
        g, kept = residualise(g0, Z, gr, seed=int(rng.integers(2 ** 31)))
        info.update({"base": "random_conditional_continuous", "kept_variance_share": kept, **extra})
    elif family.startswith("gp_") or family == "spatial_mix_gp":
        g, extra = _gp(Z, rng)
        info.update(extra)
    elif family.startswith("piecewise_"):
        g, extra = _piecewise(Z, rng)
        info.update(extra)
    else:  # pragma: no cover - FAMILIES is exhaustive above
        raise ValueError(family)
    g = centre_within(g, gr)
    if float(np.var(g)) <= 1e-15:
        raise ValueError(f"family {family!r} produced a constant signal")
    return g, info
