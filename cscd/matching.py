"""Stage 3: propose correspondences, then calibrate them.

Hypotheses are built over *subsets* of A-side atoms, not single atoms. That is
the whole point: a correspondence carried by a conjunction is marginally
invisible, so a matcher restricted to singletons cannot recover it no matter
how the dictionaries are trained. Order-2 subsets are the minimum needed to
test Prop. 2; `max_order` raises it at combinatorial cost.

Everything is scored on held-out units. Discovery units fix the dictionaries;
they never contribute to an effect estimate.
"""
from __future__ import annotations

import itertools

import numpy as np


# ------------------------------------------------------------ hypotheses ----
def build_hypotheses(m_a, m_b, max_order=2, a_subset=None, b_subset=None):
    """All (a_tuple, b_index) candidates up to `max_order` A-side atoms.

    Returned as an explicit list so the test family size is knowable before
    anything is scored, which is what multiplicity control requires.
    """
    A = list(range(m_a)) if a_subset is None else list(a_subset)
    B = list(range(m_b)) if b_subset is None else list(b_subset)
    hyps = []
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(A, order):
            for b in B:
                hyps.append((combo, b))
    return hyps


def conjunction_feature(ZA, a_tuple):
    """The A-side quantity a hypothesis is about.

    Order 1 is the atom itself. Higher orders are the product of the atoms,
    which is zero unless every atom in the tuple is active -- an AND over the
    sparse code, and the reason this reaches structure a marginal method
    cannot.
    """
    f = ZA[:, a_tuple[0]].astype(np.float64).copy()
    for j in a_tuple[1:]:
        f = f * ZA[:, j]
    return f


def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-12 else 0.0


def score_hypotheses(ZA, ZB, hyps, min_active=20):
    """Effect size per hypothesis on the units supplied.

    A conjunction that almost never fires cannot be estimated; those are
    returned with `evaluable=False` and a NaN effect rather than a number that
    looks like evidence. This mirrors the NOT_EVALUABLE discipline already used
    upstream.
    """
    out = []
    for a_tuple, b in hyps:
        f = conjunction_feature(ZA, a_tuple)
        n_active = int((f > 0).sum())
        if n_active < min_active or f.std() < 1e-12:
            out.append({"a": a_tuple, "b": int(b), "effect": np.nan,
                        "n_active": n_active, "evaluable": False})
            continue
        out.append({"a": a_tuple, "b": int(b),
                    "effect": _pearson(f, ZB[:, b]),
                    "n_active": n_active, "evaluable": True})
    return out


# ------------------------------------------------------------ calibration ---
def analytic_pvalues(scored, n_eval):
    """Two-sided p-values for a Pearson correlation via the Fisher-z transform.

    Permutation p-values cannot be the primary calibration here. Their
    resolution floors at 1/(B+1), and BH over a family of m tests needs
    p <= alpha/m at the smallest rank. With m in the thousands, the floor sits
    orders of magnitude above the threshold and the procedure can never reject
    anything -- measured directly: 0 of 16,896 hypotheses flagged at B = 100.
    Fisher-z has unbounded resolution, so the family size stops being a wall.

    Permutation remains available via `permutation_pvalues` and is the
    calibration comparator in the E3(c) ablation, which is where the
    normality of the conjunction feature gets checked rather than assumed.
    """
    for s in scored:
        if not s["evaluable"] or n_eval < 4:
            s["p"] = np.nan
            continue
        r = float(np.clip(s["effect"], -0.999999, 0.999999))
        z = np.arctanh(r) * np.sqrt(max(n_eval - 3, 1))
        # two-sided normal tail, computed via erfc for tail accuracy
        from math import erfc
        s["p"] = float(erfc(abs(z) / np.sqrt(2.0)))
        s["fisher_z"] = float(z)
    return scored


def permutation_pvalues(ZA, ZB, scored, n_perm=1000, seed=0):
    """Two-sided empirical p-values against a unit-permutation null.

    Permuting B's rows breaks the correspondence while preserving both
    modalities' marginal structure, so the null retains everything except the
    thing being tested.
    """
    rng = np.random.default_rng(seed)
    n = ZA.shape[0]
    feats, idx = {}, []
    for i, s in enumerate(scored):
        if not s["evaluable"]:
            continue
        feats[i] = conjunction_feature(ZA, s["a"])
        idx.append(i)

    exceed = {i: 0 for i in idx}
    for _ in range(n_perm):
        perm = rng.permutation(n)
        ZBp = ZB[perm]
        for i in idx:
            r = abs(_pearson(feats[i], ZBp[:, scored[i]["b"]]))
            if r >= abs(scored[i]["effect"]):
                exceed[i] += 1

    floor = 1.0 / (n_perm + 1)
    for i, s in enumerate(scored):
        if i in exceed:
            s["p"] = (exceed[i] + 1) / (n_perm + 1)
        else:
            s["p"] = np.nan
        s["p_floor"] = floor
        s["n_perm"] = n_perm
    return scored


def benjamini_hochberg(pvals, alpha=0.05, *, nan_policy, family_size=None):
    """BH step-up. Returns a boolean mask of rejections and per-test q-values.

    `nan_policy` is REQUIRED and has no default. It names the multiplicity
    denominator, which is the only thing about this function a caller can get
    wrong in a direction that flatters them:

      "declared"   m = the whole list handed in (or `family_size` if the
                   declared family is larger than the list). A NaN entry is
                   COUNTED in m and can never be rejected. Correct when a NaN
                   means "this hypothesis was in the family, was tested, and
                   returned no usable statistic" -- e.g. a conjunction whose
                   support fell below `min_active`, which is a property of the
                   REPRESENTATION, not of the family.
      "evaluable"  m = the number of non-NaN entries. Correct only when a NaN
                   means "this hypothesis was never part of the family and no
                   test was performed", and when that exclusion does not depend
                   on the same data that produced the p-values.

    The default used to be "evaluable", silently. That is anti-conservative in
    a specific direction and it biases every cross-arm comparison: a degenerate
    representation makes more of its hypotheses non-evaluable, so it is
    corrected over a SMALLER m and held to a MORE LENIENT bar than a healthy
    one. Measured on the step-6 reconstruction arm: 5 of 8 hypotheses
    non-evaluable gave an effective m = 3 against a declared family of 8, and
    one extra flag. There is no default now, so the choice is on the record at
    every call site.

    Note that "declared" is never anti-conservative relative to the old
    behaviour: m >= #evaluable, so every q is >= the q the old code produced.
    Where evaluability is data-dependent, "evaluable" has no FDR guarantee at
    all -- it is selection on the same data that produced the p-values.
    """
    if nan_policy not in ("declared", "evaluable"):
        raise ValueError(
            "nan_policy must be 'declared' (non-evaluable members stay in the "
            "multiplicity family) or 'evaluable' (m = number of tests actually "
            "performed); got %r" % (nan_policy,))
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    n_ok = int(ok.sum())

    if nan_policy == "evaluable":
        if family_size is not None:
            raise ValueError("family_size is meaningful only with "
                             "nan_policy='declared'; with 'evaluable' the "
                             "denominator is the count of non-NaN p-values")
        m = n_ok
    else:
        m = len(p) if family_size is None else int(family_size)
        if m < len(p):
            raise ValueError("family_size %d is smaller than the %d p-values "
                             "supplied; the declared family cannot be smaller "
                             "than the tests in it" % (m, len(p)))

    q = np.full_like(p, np.nan)
    if n_ok == 0 or m == 0:
        return np.zeros_like(p, dtype=bool), q

    # The finite entries hold ranks 1..n_ok among m, because every member that
    # is in the family but has no statistic sits at the top of the ordering
    # (equivalently p = 1.0) and can only ever tighten what is below it. That
    # is why `prev` starts at 1.0 -- padding with explicit ones is inert.
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    prev = 1.0
    for rank in range(n_ok - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * m / (rank + 1))
        q[i] = prev = val
    return (q <= alpha) & ok, q


def flag_discoveries(ZA_eval, ZB_eval, hyps, alpha=0.05, n_perm=1000,
                     min_active=20, seed=0, calibration="analytic"):
    """Score, calibrate, and apply BH. Returns every hypothesis, flagged or not.

    `calibration` is "analytic" (Fisher-z, the default and the only one whose
    resolution survives a large family) or "permutation" (the E3(c)
    comparator). Unflagged rows are retained deliberately: the family size and
    the unsupported results are part of the record, not noise to be dropped.
    """
    scored = score_hypotheses(ZA_eval, ZB_eval, hyps, min_active=min_active)
    if calibration == "permutation":
        scored = permutation_pvalues(ZA_eval, ZB_eval, scored,
                                     n_perm=n_perm, seed=seed)
    else:
        scored = analytic_pvalues(scored, ZA_eval.shape[0])
    rej, q = benjamini_hochberg([s["p"] for s in scored], alpha=alpha, nan_policy="declared")
    for s, r, qq in zip(scored, rej, q):
        s["q"] = float(qq) if not np.isnan(qq) else np.nan
        s["flagged"] = bool(r)
    return scored


# ----------------------------------------------------- atom -> factor map ---
def align_atoms_to_factors(Z, F, min_abs_r=0.30):
    """Map each learned atom to the true generative factor it best tracks.

    Needed only for benchmark scoring, where ground truth is in factor space
    and discoveries are in atom space. An atom that matches nothing above
    `min_abs_r` maps to None, so a flagged hypothesis resting on it cannot be
    silently credited as a true discovery.
    """
    mapping = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        if z.std() < 1e-12:
            mapping.append({"atom": j, "factor": None, "r": 0.0})
            continue
        rs = np.array([abs(_pearson(z, F[:, f])) for f in range(F.shape[1])])
        best = int(np.argmax(rs))
        mapping.append({
            "atom": j,
            "factor": best if rs[best] >= min_abs_r else None,
            "r": float(rs[best]),
        })
    return mapping


# ------------------------------------------- factor-SUBSPACE credit rule ----
def _corr_matrix(A, B):
    """Column-wise Pearson correlations between two centred matrices."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    sa = np.linalg.norm(Ac, axis=0)
    sb = np.linalg.norm(Bc, axis=0)
    sa[sa < 1e-12] = np.inf
    sb[sb < 1e-12] = np.inf
    return (Ac.T @ Bc) / np.outer(sa, sb)


def align_tuples_to_factors_subspace(Z, F, max_order=2, min_abs_r=0.30,
                                     atom_map=None):
    """Credit an A-side atom TUPLE with the factor set its SPAN recovers.

    `align_atoms_to_factors` asks each atom separately which single factor it
    tracks. A representation that spreads one factor across several atoms --
    PCA is the extreme case here -- can carry a planted conjunction perfectly
    while no individual atom clears |r| >= 0.30 against any single factor, and
    the pair is then uncreditable before any engine runs. This rule asks the
    weaker, and for a *pair* hypothesis the more appropriate, question: does
    the 2-dimensional span of the two atoms recover two distinct factors?

    For a tuple T and factor f the criterion is the multiple correlation
    R^2(f | span(Z[:, T])), thresholded at `min_abs_r ** 2` so that an
    order-1 tuple reduces EXACTLY to the single-atom rule (R^2 = r^2). The
    |T| factors with the largest R^2 are credited, and only if all of them
    clear the threshold and are distinct.

    The returned map is the UNION of this rule with the single-atom rule
    (passed in as `atom_map`, or recomputed): a tuple every one of whose atoms
    maps individually keeps that credit even if the span happens to rank some
    third factor higher. That makes the subspace rule a strict superset of the
    published rule, so its structural cap can never be lower -- the two are
    comparable by construction rather than by inspection.

    Returns {tuple: frozenset(factors) or None}.
    """
    Z = np.asarray(Z, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    m, kF = Z.shape[1], F.shape[1]
    if atom_map is None:
        atom_map = align_atoms_to_factors(Z, F, min_abs_r=min_abs_r)
    thr = float(min_abs_r) ** 2

    R = _corr_matrix(Z, F)                       # (m, kF) atom x factor
    G = _corr_matrix(Z, Z)                       # (m, m)  atom x atom
    R = np.nan_to_num(R)
    G = np.nan_to_num(G)
    np.fill_diagonal(G, 1.0)

    out = {}
    for j in range(m):
        f = atom_map[j]["factor"]
        out[(j,)] = frozenset([f]) if f is not None else None
    if max_order < 2:
        return out

    for i in range(m):
        for j in range(i + 1, m):
            # single-atom credit first: it is the published rule and wins
            fi, fj = atom_map[i]["factor"], atom_map[j]["factor"]
            if fi is not None and fj is not None and fi != fj:
                out[(i, j)] = frozenset([fi, fj])
                continue
            rij = float(G[i, j])
            denom = 1.0 - rij * rij
            if denom < 1e-9:                     # atoms collinear: no 2-d span
                out[(i, j)] = None
                continue
            ri, rj = R[i], R[j]
            # R^2 of each factor on span(z_i, z_j), closed form for 2 predictors
            r2 = (ri * ri + rj * rj - 2.0 * rij * ri * rj) / denom
            r2 = np.clip(r2, 0.0, 1.0)
            top = np.argsort(-r2, kind="stable")[:2]
            if len(set(top.tolist())) == 2 and float(r2[top[1]]) >= thr:
                out[(i, j)] = frozenset(int(t) for t in top)
            else:
                out[(i, j)] = None
    return out
