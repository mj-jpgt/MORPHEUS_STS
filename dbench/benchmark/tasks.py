"""Planted worlds on the real substrate.

A world is: real target y -> level-k residual r (cross-fitted ridge on the
level's comparison columns; proposal rows cross-fitted over proposal patients,
held-out rows predicted by the model fit on all proposal patients, so held-out
response values never enter any fit) -> y* = r + g(X_S), with g scaled so that
var(g) = f * var(r) exactly over all rows of the episode.

Relation classes (functional forms follow ``morpheus.tier2.objects`` /
``inject.py``: marginal, conjunctive product, conditional/threshold,
spatial_neighborhood; the scaling to a fraction of a residual's variance is new
because neither ``inject.py`` nor ``dbench.spike_in`` plants into a residual):

* ``single_linear``      g = x_a
* ``additive_2``         g = x_a + x_b
* ``product_2``          g = x_a * x_b with its linear projection on (x_a, x_b)
                         removed, so both marginal linear effects are exactly 0
* ``context_threshold``  g = x_a * 1[c > median(c)], c a k = 6 neighbour-mean
                         programme score (the effect exists only where the
                         neighbourhood programme is high)
* ``spatial_lag``        g = within-patient k = 6 neighbour mean of x_a
* ``and_gate_2``         a residualised conjunction of two independently
                         thresholded measured variables
* ``xor_gate_2``         a residualised exclusive-or of two independently
                         thresholded measured variables

Every g is centred within patient (as the response is) before scaling.

Generator families (``dbench.benchmark.generators``; training families
``TRAIN_FAMILIES`` and held-out families ``HELDOUT_FAMILIES``) are planted by
``generators.plant`` over k = ``arity(family)`` weakly correlated columns
(pairwise |r| < 0.3).  The two spatial families take [own_a, nbr6_b] or
[own_a, contrast_a]; their truth variables carry the transform prefix
(``nbr6:``, ``contrast:``; own columns keep their bare names, so world ids and
truths of the original five classes are unchanged).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import generators
from .generators import HELDOUT_FAMILIES, SPATIAL_FAMILIES, TRAIN_FAMILIES, arity
from .substrate import CONTEXT_FAMILY, FM_VIEWS, GENE_VIEWS, BenchSubstrate, variable_name

ORIGINAL_CLASSES = ("single_linear", "additive_2", "product_2", "context_threshold", "spatial_lag",
                    "and_gate_2", "xor_gate_2")
GENERATOR_CLASSES = TRAIN_FAMILIES + HELDOUT_FAMILIES
RELATION_CLASSES = ORIGINAL_CLASSES + GENERATOR_CLASSES
#: hypothesis-side name of each planted class
HYPOTHESIS_CLASS = {"single_linear": "single", "additive_2": "additive", "product_2": "product",
                    "context_threshold": "context", "spatial_lag": "spatial_lag",
                    "and_gate_2": "higher_order", "xor_gate_2": "higher_order"}
for _fam in GENERATOR_CLASSES:  # by arity; a spatial family mixes own and lagged values, so not spatial_lag
    HYPOTHESIS_CLASS[_fam] = {1: "single", 2: "product"}.get(arity(_fam), "higher_order")
#: share of spatial_mix worlds whose pair is [own_a, contrast_a] rather than [own_a, nbr6_b]
CONTRAST_SHARE = 0.5
EFFECT_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)
SOURCES = ("gene", "fm")
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


@dataclass(frozen=True)
class Truth:
    relation: str
    variables: tuple  # sorted design column names
    effect_fraction: float
    level: str
    source: str

    @property
    def hypothesis_class(self) -> str:
        return HYPOTHESIS_CLASS[self.relation]


@dataclass
class World:
    world_id: str
    truth: Truth
    response: np.ndarray
    residual: np.ndarray
    signal: np.ndarray
    family: str
    calibration: dict = field(default_factory=dict)


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def grouped_fold_ids(groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    uniq = np.unique(groups)
    k = max(2, min(n_splits, len(uniq)))
    assign = dict(zip(uniq.tolist(), (np.random.default_rng(seed).permutation(len(uniq)) % k).tolist()))
    return np.array([assign[g] for g in groups.tolist()])


def _ridge(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    coef = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ coef + ym


def cross_fitted_ridge_residual(B: np.ndarray, y: np.ndarray, groups: np.ndarray, proposal_mask,
                                *, n_splits: int = 5, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Residual of ``y`` on ``B`` without touching held-out responses.

    ``proposal_mask``: boolean mask of proposal rows (or their index array); every other row is
    held out.  Ridge strength by patient-grouped CV on proposal rows; proposal residuals are
    cross-fitted over proposal patients; held-out residuals use the model fit on all proposal rows."""
    mask = np.asarray(proposal_mask)
    if mask.dtype == bool:
        P, H = np.flatnonzero(mask), np.flatnonzero(~mask)
    else:
        P = mask.astype(np.int64)
        keep = np.ones(len(y), dtype=bool)
        keep[P] = False
        H = np.flatnonzero(keep)
    return _cross_fit(np.asarray(B, dtype=float), np.asarray(y, dtype=float), np.asarray(groups), P, H,
                      n_splits=n_splits, seed=seed)


def _cross_fit(B, y, groups, P, H, *, n_splits: int, seed: int) -> tuple[np.ndarray, dict]:
    folds = grouped_fold_ids(groups[P], n_splits, seed)
    best = (None, -np.inf)
    for alpha in RIDGE_GRID:
        pred = np.empty(len(P))
        for k in np.unique(folds):
            te = folds == k
            pred[te] = _ridge(B[P][~te], y[P][~te], B[P][te], alpha)
        r2 = 1 - np.sum((y[P] - pred) ** 2) / np.sum((y[P] - y[P].mean()) ** 2)
        if r2 > best[1]:
            best, cross = (alpha, r2), pred
    r = np.empty_like(y)
    r[P] = y[P] - cross
    hpred = _ridge(B[P], y[P], B[H], best[0])
    r[H] = y[H] - hpred
    r2_h = 1 - np.sum((y[H] - hpred) ** 2) / np.sum((y[H] - y[H].mean()) ** 2)
    return r, {"width": int(B.shape[1]), "alpha": best[0], "proposal_cv_r2": float(best[1]),
               "heldout_r2": float(r2_h), "residual_variance": float(r.var())}


def level_residual(sub: BenchSubstrate, level: str, *, n_splits: int = 5, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Remove the level's predictable part of the real target without touching held-out responses
    (``cross_fitted_ridge_residual`` on the level's comparison columns)."""
    B = np.asarray(sub.level_matrix(level), dtype=float)
    y = np.asarray(sub.y_real, dtype=float)
    r, info = _cross_fit(B, y, np.asarray(sub.groups), np.asarray(sub.proposal), np.asarray(sub.heldout),
                         n_splits=n_splits, seed=seed)
    return r, {"level": level, **info}


def centre_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for g in np.unique(groups):
        rows = groups == g
        out[rows] -= out[rows].mean()
    return out


def relation_signal(sub: BenchSubstrate, relation: str, cols: tuple, kinds: tuple | None = None,
                    rng: np.random.Generator | None = None, *, return_info: bool = False):
    """Unscaled g(X_S) for the design columns ``cols`` (context column last for context_threshold).

    Generator families dispatch to ``generators.plant`` over ``sub.transform_matrix(cols, kinds)``
    (``kinds`` defaults to own values; ``rng`` is required there)."""
    X = sub.X
    info: dict = {}
    if relation in GENERATOR_CLASSES:
        if rng is None:
            raise ValueError(f"{relation!r} needs a random generator")
        kinds = tuple(kinds) if kinds is not None else ("own",) * len(cols)
        g, info = generators.plant(sub.transform_matrix(cols, kinds), relation, rng, groups=sub.groups)
    elif relation == "single_linear":
        g = X[:, cols[0]].astype(float)
    elif relation == "additive_2":
        g = X[:, cols[0]].astype(float) + X[:, cols[1]].astype(float)
    elif relation == "product_2":
        a = centre_within(X[:, cols[0]], sub.groups)
        b = centre_within(X[:, cols[1]], sub.groups)
        p = centre_within(a * b, sub.groups)
        D = np.column_stack([a, b])
        g = p - D @ np.linalg.lstsq(D, p, rcond=None)[0]  # zero marginal linear effects, exactly
    elif relation == "context_threshold":
        c = X[:, cols[1]].astype(float)
        g = X[:, cols[0]].astype(float) * (c > np.median(c))
    elif relation == "spatial_lag":
        g = sub.lag(int(cols[0]))
    elif relation in ("and_gate_2", "xor_gate_2"):
        # These are deliberately not model heads.  They are real-substrate
        # controlled *evaluation* functions whose complete object cannot be
        # recovered by an additive lower-order decomposition.  Thresholds are
        # calculated per patient so a batch/patient mean cannot carry the
        # answer.  We then remove a flexible lower-order basis, exactly as for
        # residualised_k generator worlds, to make the scientific question
        # about the joint object rather than a marginal proxy.
        a = centre_within(X[:, cols[0]], sub.groups)
        b = centre_within(X[:, cols[1]], sub.groups)
        hi_a = np.zeros(len(a), dtype=float)
        hi_b = np.zeros(len(b), dtype=float)
        thresholds = {}
        for patient in np.unique(sub.groups):
            rows = sub.groups == patient
            qa, qb = float(np.median(a[rows])), float(np.median(b[rows]))
            hi_a[rows], hi_b[rows] = a[rows] > qa, b[rows] > qb
            thresholds[str(patient)] = [qa, qb]
        if relation == "and_gate_2":
            raw = (hi_a * hi_b).astype(float)
        else:
            raw = np.logical_xor(hi_a > 0, hi_b > 0).astype(float)
        g, kept = generators.residualise(raw, np.column_stack([a, b]), sub.groups,
                                          seed=int(rng.integers(2 ** 31)) if rng is not None else 0)
        info = {"threshold_scope": "within_patient_median", "kept_variance_share": kept,
                "thresholds": thresholds}
    else:
        raise ValueError(f"unknown relation {relation!r}")
    g = centre_within(g, sub.groups)
    return (g, info) if return_info else g


def calibrate(signal: np.ndarray, residual: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    """Scale ``signal`` so var(scaled) / var(residual) == fraction (population variance, all rows)."""
    v = float(np.var(signal))
    if v <= 1e-15:
        raise ValueError("planted signal is constant")
    scale = float(np.sqrt(fraction * np.var(residual) / v))
    return signal * scale, scale



def calibrate_by_patient(signal: np.ndarray, residual: np.ndarray, groups: np.ndarray,
                         fraction: float) -> tuple[np.ndarray, dict]:
    """Scale a controlled intervention separately within each patient."""
    signal, residual, groups = (np.asarray(signal, dtype=float), np.asarray(residual, dtype=float),
                                 np.asarray(groups))
    out = np.zeros_like(signal)
    scales, per_group = {}, {}
    for g in np.unique(groups):
        rows = groups == g
        sv, rv = float(np.var(signal[rows])), float(np.var(residual[rows]))
        if sv <= 1e-15 or rv <= 1e-15:
            raise ValueError(f"patient {g!r} cannot support a calibrated packet")
        scale = float(np.sqrt(float(fraction) * rv / sv))
        out[rows] = signal[rows] * scale
        scales[str(g)] = scale
        per_group[str(g)] = float(np.var(out[rows]) / rv)
    return out, {"per_patient_scale": scales, "per_patient_fraction": per_group}


def packet_validity(signal: np.ndarray, residual: np.ndarray, groups: np.ndarray,
                    proposal: np.ndarray, heldout: np.ndarray, *, minimum_fraction: float = 0.04) -> dict:
    """Method-independent fixture QA, recorded before any proposer runs."""
    signal, residual = np.asarray(signal, dtype=float), np.asarray(residual, dtype=float)
    def frac(rows):
        rv = float(np.var(residual[rows]))
        return float(np.var(signal[rows]) / rv) if rv > 1e-15 else 0.0
    p, q = frac(proposal), frac(heldout)
    return {"valid": bool(p >= minimum_fraction and q >= minimum_fraction),
            "minimum_fraction": float(minimum_fraction),
            "proposal_fraction": p, "heldout_fraction": q,
            "reasons": ([] if p >= minimum_fraction else ["proposal_signal_too_small"]) +
                       ([] if q >= minimum_fraction else ["heldout_signal_too_small"])}

def _usable(sub: BenchSubstrate, cols: np.ndarray) -> np.ndarray:
    """Columns that can carry a planted effect: not dominated by one repeated value on proposal rows."""
    Xp = sub.X[sub.proposal][:, cols]
    ok = []
    for j, c in enumerate(cols):
        vals, counts = np.unique(np.round(Xp[:, j], 6), return_counts=True)
        if counts.max() / len(Xp) < 0.5 and Xp[:, j].std() > 0.5:
            ok.append(c)
    return np.asarray(ok, dtype=int)


def variable_pool(sub: BenchSubstrate, relation: str, source: str) -> np.ndarray:
    views = GENE_VIEWS if source == "gene" else FM_VIEWS
    if relation == "context_threshold":  # the context family's views (no scGPT there)
        views = tuple(v for v in views if v in CONTEXT_FAMILY)
    cols = np.concatenate([np.asarray(sub.views[v], dtype=int) for v in views if v in sub.views])
    return _usable(sub, np.sort(cols))


def context_pool(sub: BenchSubstrate) -> np.ndarray:
    cols = np.asarray(sub.views.get("context_neighbour_mean", []), dtype=int)
    keep = [c for c in cols if str(sub.names[c]).split(":", 1)[-1].startswith("nbr:")
            and not str(sub.names[c]).split(":", 1)[-1].startswith("nbr:PC")]
    return _usable(sub, np.asarray(keep, dtype=int))


def _weakly_correlated(Xp: np.ndarray, pool: np.ndarray, k: int, rng: np.random.Generator) -> tuple:
    for _ in range(500):
        cols = [int(c) for c in rng.choice(pool, k, replace=False)]
        R = np.corrcoef(Xp[:, cols].T)
        if np.all(np.abs(R[np.triu_indices(k, 1)]) < 0.3):
            return tuple(sorted(cols))
    raise RuntimeError(f"no weakly correlated set of {k} columns found")


def draw_variables(sub: BenchSubstrate, relation: str, source: str, rng: np.random.Generator) -> tuple:
    """(columns, transforms): k design columns and one transform per column.

    Original classes and non-spatial generator families: own values (``spatial_lag`` applies its lag
    inside the relation, so its truth keeps the bare name).  Spatial families: [own_a, nbr6_b] with
    |r(own_a, nbr6_b)| < 0.3 on proposal rows, or, for ``spatial_mix`` with probability
    ``CONTRAST_SHARE``, [own_a, contrast_a]."""
    pool = variable_pool(sub, relation, source)
    if relation in ("single_linear", "spatial_lag"):
        return (int(rng.choice(pool)),), ("own",)
    if relation == "context_threshold":
        return (int(rng.choice(pool)), int(rng.choice(context_pool(sub)))), ("own", "own")
    Xp = sub.X[sub.proposal]
    if relation in SPATIAL_FAMILIES:
        if relation == "spatial_mix" and rng.random() < CONTRAST_SHARE:
            a = int(rng.choice(pool))
            return (a, a), ("own", "contrast")
        P = sub.proposal
        for _ in range(500):
            a, b = (int(c) for c in rng.choice(pool, 2, replace=False))
            if abs(np.corrcoef(sub.X[P, a], sub.transform(b, "nbr6")[P])[0, 1]) < 0.3:
                return (a, b), ("own", "nbr6")
        raise RuntimeError("no weakly correlated own / neighbour pair found")
    if relation in GENERATOR_CLASSES:
        k = arity(relation)
        if k == 1:
            return (int(rng.choice(pool)),), ("own",)
        return _weakly_correlated(Xp, pool, k, rng), ("own",) * k
    for _ in range(200):
        a, b = (int(c) for c in rng.choice(pool, 2, replace=False))
        if abs(np.corrcoef(Xp[:, a], Xp[:, b])[0, 1]) < 0.3:
            return tuple(sorted((a, b))), ("own", "own")
    raise RuntimeError("no weakly correlated pair found")


def draw_columns(sub: BenchSubstrate, relation: str, source: str, rng: np.random.Generator) -> tuple:
    """Design columns of a planted relation (``draw_variables`` without the transforms)."""
    return draw_variables(sub, relation, source, rng)[0]


def make_world(sub: BenchSubstrate, *, relation: str, fraction: float, level: str, seed: int, source: str,
               residual_cache: dict | None = None, calibration_mode: str = "global") -> World:
    if relation not in RELATION_CLASSES:
        raise ValueError(f"unknown relation {relation!r}")
    world_id = f"{relation}|{source}|f{fraction:g}|{level}|s{seed}"
    rng = np.random.default_rng(stable_seed(sub.meta.get("target", ""), world_id))
    cache = residual_cache if residual_cache is not None else {}
    if level not in cache:
        cache[level] = level_residual(sub, level)
    residual, level_info = cache[level]
    cols, kinds = draw_variables(sub, relation, source, rng)
    raw, gen_info = relation_signal(sub, relation, cols, kinds, rng, return_info=True)
    if calibration_mode == "global":
        signal, scale = calibrate(raw, residual, fraction)
        scale_info = {"scale": scale}
    elif calibration_mode == "patient_balanced":
        signal, scale_info = calibrate_by_patient(raw, residual, sub.groups, fraction)
        scale = None
    else:
        raise ValueError(f"unknown calibration_mode {calibration_mode!r}")
    names = tuple(sorted(variable_name(sub.names[c], k) for c, k in zip(cols, kinds)))
    fr = lambda rows: float(np.var(signal[rows]) / np.var(residual[rows]))  # noqa: E731
    calibration = {"mode": calibration_mode, "fraction_all_rows": float(np.var(signal) / np.var(residual)),
                   "fraction_proposal_rows": fr(sub.proposal), "fraction_heldout_rows": fr(sub.heldout),
                   "scale": scale, **scale_info, "columns": list(map(int, cols)),
                   "context_column": str(sub.names[cols[1]]) if relation == "context_threshold" else None,
                   "level": level_info,
                   "validity": packet_validity(signal, residual, sub.groups, sub.proposal, sub.heldout)}
    if relation in GENERATOR_CLASSES or relation in ("and_gate_2", "xor_gate_2"):
        calibration.update({"transforms": list(kinds), "generator": _small(gen_info)})
    return World(world_id=world_id, truth=Truth(relation, names, float(fraction), level, source),
                 response=residual + signal, residual=residual, signal=signal,
                 family="context" if relation == "context_threshold" else "molecular", calibration=calibration)


def _small(info: dict) -> dict:
    """Generator parameters without the bulky weight arrays (kept JSON-sized in every row)."""
    return {k: v for k, v in info.items() if k not in ("W1", "coefficients")}
