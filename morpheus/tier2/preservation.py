"""Generic, substrate-agnostic preservation statistics.

Each function here measures one property that a substrate-specific
preservation check (in `substrates/*.py`) then asserts is unchanged
before/after injection and before/after the substrate's own structured
null. Kept substrate-agnostic on purpose -- "Moran's I" and "exact
composition" mean the same computation whether the categorical grouping is
donor, patient, or tissue region; only which array each substrate hands in
differs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HOME = os.path.expanduser("~")
if _HOME not in sys.path:
    sys.path.insert(0, _HOME)

from cscd.spike_in import covariance_preservation_check  # noqa: E402  (reused, not rewritten)

__all__ = ["covariance_preservation_check", "exact_composition",
           "moran_i", "autocorrelation_at_lag", "month_of_year_profile",
           "seasonality_preserved"]


def exact_composition(labels):
    """Exact per-category counts -- a dict, so "preserved" means dict-equal,
    not merely close."""
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def moran_i(values, coords, groups=None, k=6):
    """Global Moran's I of `values` over a row-standardised k-NN spatial
    weight matrix, computed separately within each `groups` block (so
    neighbours never cross e.g. a patient boundary) and pooled by summing
    the numerator/denominator terms across blocks before dividing -- the
    standard way to combine Moran's I over disjoint regions without letting
    a large, homogeneous block dominate a naive average of per-block I's.
    """
    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if groups is None:
        groups = np.zeros(len(values), dtype=int)
    groups = np.asarray(groups)

    num_total, den_total, w_sum_total, n_total = 0.0, 0.0, 0.0, 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < max(3, k + 1):
            continue
        v = values[idx]
        vbar = v.mean()
        z = v - vbar
        pts = coords[idx]
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, len(idx) - 1)
        nn = np.argsort(d2, axis=1)[:, :kk]
        W = np.zeros((len(idx), len(idx)))
        for i in range(len(idx)):
            W[i, nn[i]] = 1.0 / kk
        num_total += float((W * np.outer(z, z)).sum())
        den_total += float((z ** 2).sum())
        w_sum_total += float(W.sum())
        n_total += len(idx)

    if den_total <= 0 or w_sum_total <= 0:
        return float("nan")
    return float((n_total / w_sum_total) * (num_total / den_total))


def autocorrelation_at_lag(series, lag=1):
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    num = float((x[: n - lag] * x[lag:]).sum())
    den = float((x ** 2).sum())
    return num / den if den > 0 else float("nan")


def month_of_year_profile(values, month_of_year):
    """Mean value by calendar month (1..12) -- the seasonal cycle, as a
    12-vector. Two series have "the same seasonality" iff this vector
    matches, independent of which years contributed which months."""
    values = np.asarray(values, dtype=np.float64)
    month_of_year = np.asarray(month_of_year)
    return np.array([values[month_of_year == m].mean() for m in range(1, 13)])


def seasonality_preserved(values_before, values_after, month_of_year, tol=1e-8):
    prof0 = month_of_year_profile(values_before, month_of_year)
    prof1 = month_of_year_profile(values_after, month_of_year)
    max_dev = float(np.nanmax(np.abs(prof0 - prof1)))
    return {"max_abs_deviation": max_dev, "passes": bool(max_dev < tol),
            "profile_before": prof0.tolist(), "profile_after": prof1.tolist()}
