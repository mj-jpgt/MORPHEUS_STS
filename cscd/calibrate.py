"""Calibration of correspondence statistics against a pooled permutation null.

Why this exists. Two obvious choices both fail here:

  per-hypothesis permutation   resolution floors at 1/(B+1). BH over a family
                               of thousands needs p <= alpha/m at the smallest
                               rank, which sits far below that floor, so the
                               procedure can never reject. Measured: 0 of
                               16,896 flagged at B = 100.

  Fisher-z / analytic          unbounded resolution, but the statistic is a
                               correlation with a product of sparse codes --
                               mostly exact zeros, heavily non-normal. Measured:
                               409 spurious of 420 flags, i.e. anti-conservative
                               by more than an order of magnitude.

So the null is pooled across hypotheses and stratified by activation count.
Pooling buys resolution (B * m draws instead of B); stratifying keeps the
comparison fair, because a conjunction that fires 30 times and one that fires
600 times have different null spreads. This mirrors the abundance-stratified
null the project already uses on the molecular side.
"""
from __future__ import annotations

import numpy as np

from .matching import conjunction_feature


def _standardise(M):
    M = M - M.mean(0, keepdims=True)
    s = np.linalg.norm(M, axis=0, keepdims=True)
    return M / np.maximum(s, 1e-12)


def _partial_out(y, controls):
    """Residual of y after least-squares removal of `controls` (plus intercept)."""
    X = np.column_stack([np.ones(len(y))] + list(controls))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def build_feature_matrix(ZA, hyps):
    """A-side features, one column per DISTINCT A-tuple.

    Hypotheses reuse the same A-tuple across every B atom, so a column per
    hypothesis duplicates each feature m_B times. At 96 atoms and order 2 that
    is a (n, 446976) array -- 4.94 GiB -- against (n, 4656) here.

    Returns (F, col_of_hyp) where `col_of_hyp[i]` indexes hypothesis i's column.
    """
    order = {}
    cols = []
    col_of_hyp = np.empty(len(hyps), dtype=np.int64)
    for i, (a_tuple, _) in enumerate(hyps):
        key = tuple(a_tuple)
        j = order.get(key)
        if j is None:
            j = len(cols)
            order[key] = j
            f = conjunction_feature(ZA, key)
            if len(key) > 1:
                # TopK codes are non-negative, so a raw product carries the
                # main effects of its constituents: z_i*z_j correlates with
                # whatever z_i and z_j each correlate with. Testing the raw
                # product therefore rediscovers marginal structure under a
                # conjunctive label. Partial the constituents out so the
                # feature is the interaction and nothing else.
                f = _partial_out(f, [ZA[:, t].astype(np.float64) for t in key])
            cols.append(f)
        col_of_hyp[i] = j
    return np.column_stack(cols), col_of_hyp


def block_permutation_index(n, block_len, rng):
    """A permutation of range(n) built by reordering contiguous blocks.

    Rows within a block keep their original relative order and each row is
    used exactly once (a genuine permutation, not a resampling-with-repeats),
    but only whole blocks are shuffled. This is the moving-block permutation:
    for time-ordered, autocorrelated units (e.g. calendar months), a plain
    `rng.permutation(n)` is exchangeable only under an i.i.d. assumption that
    autocorrelated series violate, which makes the resulting null anti-
    conservative. Keeping blocks intact preserves within-block dependence
    while still destroying any real correspondence with the other side.
    """
    edges = list(range(0, n, block_len))
    blocks = [np.arange(s, min(s + block_len, n)) for s in edges]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def pooled_stratified_null(ZA, ZB, hyps, n_perm=200, n_strata=8,
                           min_active=20, seed=0, alpha=0.05, block_len=None):
    """Effects and p-values for every hypothesis under one pooled null.

    `block_len`, if given, replaces the exchangeable row permutation with a
    moving-block permutation (see `block_permutation_index`) of that block
    length -- the calibration this project's autocorrelated arms (e.g.
    monthly climate data) require. Left at the default `None` for
    exchangeable units, where a plain permutation is already valid.

    Returns the same record shape as `matching.flag_discoveries` so the two are
    interchangeable in the pipeline and in the E3(c) comparison.
    """
    n = ZA.shape[0]
    F, col = build_feature_matrix(ZA, hyps)
    b_idx = np.array([b for _, b in hyps])
    n_active = np.array([int((conjunction_feature(ZA, a) > 0).sum())
                         for a, _ in hyps])
    usable = (n_active >= min_active) & (F.std(0)[col] > 1e-12)

    Fs = _standardise(F)
    Bs = _standardise(ZB.astype(np.float64))
    # one matmul gives every (feature, b-atom) correlation at once
    Rall = Fs.T @ Bs
    obs = np.full(len(hyps), np.nan)
    obs[usable] = Rall[col[usable], b_idx[usable]]

    # strata on activation count, so like is compared with like
    strata = np.full(len(hyps), -1)
    if usable.any():
        q = np.quantile(n_active[usable], np.linspace(0, 1, n_strata + 1)[1:-1])
        strata[usable] = np.searchsorted(q, n_active[usable])

    # pooled null: one permutation of B's rows regenerates every hypothesis
    rng = np.random.default_rng(seed)
    null_by_stratum = {s: [] for s in range(n_strata)}
    for _ in range(n_perm):
        idx = (block_permutation_index(n, block_len, rng) if block_len
               else rng.permutation(n))
        Bp = Bs[idx]
        Rp = Fs.T @ Bp
        a = np.abs(Rp[col[usable], b_idx[usable]])
        for s in range(n_strata):
            m = strata[usable] == s
            if m.any():
                null_by_stratum[s].append(a[m])
    pools = {s: (np.concatenate(v) if v else np.array([]))
             for s, v in null_by_stratum.items()}
    for s in pools:
        pools[s].sort()

    out = []
    for i, (a_tuple, b) in enumerate(hyps):
        if not usable[i]:
            out.append({"a": a_tuple, "b": int(b), "effect": np.nan,
                        "n_active": int(n_active[i]), "evaluable": False,
                        "p": np.nan, "stratum": None, "null_n": 0})
            continue
        s = int(strata[i])
        pool = pools[s]
        # count of null draws at least as extreme, +1 for the observed value
        k = pool.size - np.searchsorted(pool, abs(obs[i]), side="left")
        p = (k + 1.0) / (pool.size + 1.0)
        out.append({"a": a_tuple, "b": int(b), "effect": float(obs[i]),
                    "n_active": int(n_active[i]), "evaluable": True,
                    "p": float(p), "stratum": s, "null_n": int(pool.size),
                    "p_floor": 1.0 / (pool.size + 1.0)})
    return out


def calibration_report(ZA, ZB, hyps, n_perm=200, n_strata=8, min_active=20,
                       seed=0, alpha=0.05, block_len=None):
    """Realised false-positive rate of the null on data with no planted signal.

    Permutes B once to destroy any real correspondence, then runs the whole
    calibration on that permuted data. A well-calibrated null rejects about
    `alpha` of the family; anything far above means the test is
    anti-conservative and no downstream FCR can be trusted.

    `block_len`, if given, uses a moving-block permutation (see
    `pooled_stratified_null`) both to build the "no true signal" control (so
    it is scrambled the same way the calibration null scrambles B) and for
    the calibration itself -- an autocorrelated series scrambled by a plain
    permutation would understate the FPR a naive permutation null actually
    realises on unscrambled data.
    """
    from .matching import benjamini_hochberg
    rng = np.random.default_rng(seed + 991)
    idx = (block_permutation_index(ZB.shape[0], block_len, rng) if block_len
           else rng.permutation(ZB.shape[0]))
    ZB_null = ZB[idx]
    scored = pooled_stratified_null(ZA, ZB_null, hyps, n_perm=n_perm,
                                    n_strata=n_strata, min_active=min_active,
                                    seed=seed, block_len=block_len)
    ev = [s for s in scored if s["evaluable"]]
    rej, q = benjamini_hochberg([s["p"] for s in scored], alpha=alpha, nan_policy="declared")
    raw = np.mean([s["p"] <= alpha for s in ev]) if ev else float("nan")
    return {
        "n_evaluable": len(ev),
        "uncorrected_fpr": float(raw),
        "nominal_alpha": alpha,
        "n_rejected_after_bh": int(rej.sum()),
        "bh_fdr_on_pure_null": float(rej.sum() / max(len(ev), 1)),
        "n_perm": n_perm,
    }
