"""Nested explanation ladder for frozen hypotheses (protocol v0.6 design; post hoc for v0.4 / v0.5).

For a hypothesis H (one or a few bank columns and the registered relation's
feature) and a nested sequence of comparison models

    N  = nuisance only (log sequencing depth; RNA-quality columns when the bank exposes them)
    B1 = N  + CellViT class fractions (composition)
    B2 = B1 + own-spot programme scores (41 Gavish + 50 Hallmark)
    B3 = B2 + neighbour means (k = 6, 18) of the programmes and PCA, + any other registered
         competitor column = the full registered v0.5 comparison model,

the held-out gain at level k is

    Delta_k(H) = Q(B_k + H) - Q(B_k),

where Q is the held-out-patient R^2 of an inner-CV ridge fitted on the proposal
patients of a registered fold. ``Q(B_k + H)`` is the registered relation's FULL
model (the joint random-Fourier map of H's variables and feature, seeded from
the candidate hash, + B_k linearly; order >= 3 also carries the pairwise
products) and ``Q(B_k)`` is the candidate-free reference (B_k alone, its own
ridge strength chosen with the same grouped inner CV and seed). For a
single-variable hypothesis this is exactly ``FittedRelation(relation="v05")``
with ``baseline = B_k`` (its competitor *is* the reference). ``Q(B_k + linear H)``
is reported beside it, so each Delta_k splits into a linear part and the part
beyond linear.

Pooling: the folds' held-out patients are disjoint, so each patient's rows are
predicted once; the pooled Delta_k is the R^2 difference of the pooled
out-of-fold predictions and its CI resamples whole patients (one shared set of
draws for every contrast, as the runner's ``_gain_interval``).

Engine: the ridge fits are the runner's (``_RidgeFit``: standardise on the
training rows, unpenalised intercept, ridge on standardised columns; strength
by ``_cv_alpha``'s patient-grouped 5-fold CV over ``VALIDATION_ALPHA_GRID``),
computed from per-patient sufficient statistics (sums, cross-products) so the
inner CV of a 400-column design costs a few small eigendecompositions instead
of 40 passes over 46k rows. ``tests/test_decomposition.py`` asserts equality
with ``FittedRelation`` / ``_cv_alpha`` to 1e-8 on synthetic data and with a
registered v0.4 receipt gain to 1e-6 on a real fold.

Literature used (pass of 2026-10-14):
* Borcard, Legendre & Drapeau (1992) Ecology 73:1045 -- variation partitioning
  of a response between environmental and spatial predictor sets; the same
  question as "molecular vs composition vs spatial context" here.
* Chevan & Sutherland (1991) Am. Stat. 45:90 (hierarchical partitioning) and
  Groemping (2006) J. Stat. Softw. 17(1) (relaimpo): sequential R^2 increments
  depend on the entry order. This ladder is ONE pre-declared order (nuisance ->
  composition -> programmes -> context), not an order average (LMG / Shapley);
  the "absorbing level" is therefore "the first level, in this order, that
  removes most of the gain", not a unique share.
* Lescroart, Stansbury & Gallant (2015) Front. Comput. Neurosci. 9:135 and
  de Heer et al. (2017) J. Neurosci. 37:6539 -- cross-validated variance
  partitioning with ridge encoding models; caveat that separately fitted models
  are regularised differently. Here every model picks its own strength by the
  same grouped inner CV (as the registered relation does), and the strengths are
  recorded.
* Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018) JASA 113:1094 (LOCO) and
  Williamson, Gilbert, Simon & Carone (2021) Biometrics 77:9; (2023) JASA
  118:1645 -- variable importance as a difference in held-out predictiveness
  with sample splitting / cross-fitting. Their caveat applies: at zero true
  importance the plug-in difference is a boundary case and its interval is not
  calibrated, so "CI includes 0" is read as "not resolved above 0", never as
  "shown to be 0".
* Chernozhukov et al. (2018) Econometrics J. 21:C1 -- cross-fitting: nuisance
  fits on one sample, evaluation on another; here proposal patients vs
  held-out patients of each registered fold.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

LEVELS: tuple[str, ...] = ("N", "B1", "B2", "B3")
LEVEL_GROUPS: dict[str, tuple[str, ...]] = {
    "N": ("nuisance",), "B1": ("nuisance", "composition"), "B2": ("nuisance", "composition", "programmes"),
    "B3": ("nuisance", "composition", "programmes", "context")}
#: what the step INTO each level adds (used to name the absorbing level)
LEVEL_STEP_NAME = {"B1": "cell composition (CellViT class fractions)",
                   "B2": "known programmes (Gavish + Hallmark own-spot scores)",
                   "B3": "spatial context (neighbour means k=6/18 of programmes and PCA, other registered columns)"}
NUISANCE_COLUMNS = ("depth_log1p",)
NUISANCE_PREFIXES = ("qc:",)
COMPOSITION_PREFIXES = ("cv_frac_",)
PROGRAMME_PREFIXES = ("mp:", "hallmark:", "gavish:")
TIERS = ("simple association (novelty unchecked)", "partly absorbed", "irreducible", "none")


def _validation_grid() -> tuple[float, ...]:
    from morpheus_encoder.general_discovery import VALIDATION_ALPHA_GRID
    return tuple(VALIDATION_ALPHA_GRID)


# --------------------------------------------------------------------------
# Comparison-model columns -> ladder levels
# --------------------------------------------------------------------------

def column_group(name: str) -> str:
    """nuisance / composition / programmes / context for one competitor column name."""
    n = str(name)
    if n in NUISANCE_COLUMNS or n.startswith(NUISANCE_PREFIXES):
        return "nuisance"
    if n.startswith(COMPOSITION_PREFIXES):
        return "composition"
    if n.startswith(PROGRAMME_PREFIXES):
        return "programmes"
    return "context"


def ladder_levels(baseline_names: Sequence[str]) -> dict:
    """Column indices of N, B1, B2, B3 within the registered competitor (nested by construction;
    B3 = every registered column) plus the group of every column."""
    groups = [column_group(n) for n in baseline_names]
    levels = {lv: [i for i, g in enumerate(groups) if g in LEVEL_GROUPS[lv]] for lv in LEVELS}
    counts = {g: groups.count(g) for g in ("nuisance", "composition", "programmes", "context")}
    return {"levels": levels, "groups": groups, "group_counts": counts,
            "level_widths": {lv: len(ix) for lv, ix in levels.items()}}


# --------------------------------------------------------------------------
# Ridge from per-patient sufficient statistics (exactly the runner's _RidgeFit / _cv_alpha)
# --------------------------------------------------------------------------

@dataclass
class PatientStats:
    """Per-patient sums over a row set of the design M (n x d) and response y.

    ``n[g]``, ``s[g] = sum M``, ``G[g] = M'M``, ``h[g] = M'y``, ``sy[g]``, ``yy[g]``."""
    patients: np.ndarray
    n: np.ndarray
    s: np.ndarray
    G: np.ndarray
    h: np.ndarray
    sy: np.ndarray
    yy: np.ndarray

    def total(self, mask: np.ndarray | None = None) -> dict:
        m = slice(None) if mask is None else np.asarray(mask, bool)
        return {"n": float(self.n[m].sum()), "s": self.s[m].sum(0), "G": self.G[m].sum(0),
                "h": self.h[m].sum(0), "sy": float(self.sy[m].sum()), "yy": float(self.yy[m].sum())}


def patient_stats(M: np.ndarray, y: np.ndarray, groups: np.ndarray, *, B_stats: "PatientStats | None" = None,
                  n_extra: int | None = None) -> PatientStats:
    """Sufficient statistics per patient. With ``B_stats`` (statistics of the trailing columns,
    already computed) only the leading ``n_extra`` columns' blocks are computed."""
    M = np.asarray(M, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    pats = np.unique(groups)
    d = M.shape[1]
    n = np.zeros(len(pats)); sy = np.zeros(len(pats)); yy = np.zeros(len(pats))
    s = np.zeros((len(pats), d)); G = np.zeros((len(pats), d, d)); h = np.zeros((len(pats), d))
    for i, g in enumerate(pats):
        rows = np.flatnonzero(groups == g)
        D, t = M[rows], y[rows]
        n[i], sy[i], yy[i] = len(rows), t.sum(), t @ t
        s[i], h[i] = D.sum(0), D.T @ t
        if B_stats is None:
            G[i] = D.T @ D
        else:
            e = int(n_extra)
            E = D[:, :e]
            G[i, :e, :] = E.T @ D
            G[i, e:, :e] = G[i, :e, e:].T
            G[i, e:, e:] = B_stats.G[i]
    if B_stats is not None and not np.array_equal(B_stats.patients, pats):
        raise ValueError("B_stats patients differ")
    return PatientStats(pats, n, s, G, h, sy, yy)


def _standardised_system(tot: dict, cols: np.ndarray):
    n = tot["n"]
    mean = tot["s"][cols] / n
    ybar = tot["sy"] / n
    Gc = tot["G"][np.ix_(cols, cols)] - n * np.outer(mean, mean)
    var = np.diag(Gc) / n
    scale = np.sqrt(np.clip(var, 0.0, None))
    scale[scale < 1e-12] = 1.0  # _Standardizer's floor
    A = Gc / np.outer(scale, scale)
    b = (tot["h"][cols] - n * mean * ybar) / scale
    return mean, scale, ybar, A, b


def _coefficients(mean, scale, ybar, V, lam, Vb, alpha):
    """Raw-scale (intercept, weights) of the ridge at strength alpha (inf = training mean)."""
    if not np.isfinite(alpha):
        return ybar, np.zeros(len(mean))
    beta = V @ (Vb / (lam + float(alpha)))
    w = beta / scale
    return ybar - float(mean @ w), w


def _sse(tot: dict, cols: np.ndarray, c: float, w: np.ndarray) -> float:
    """Sum of squared residuals of prediction c + M[:, cols] w over the rows summarised by tot."""
    s, G, h = tot["s"][cols], tot["G"][np.ix_(cols, cols)], tot["h"][cols]
    return float(tot["yy"] - 2.0 * (c * tot["sy"] + w @ h) + c * c * tot["n"] + 2.0 * c * (w @ s) + w @ G @ w)


def _eig(A):
    lam, V = np.linalg.eigh(A)
    return lam, V


def cv_alpha_from_stats(st: PatientStats, cols: Sequence[int], seed: int, grid: Sequence[float] | None = None,
                        n_splits: int = 5) -> tuple[float, dict]:
    """The runner's ``_cv_alpha(design, y, groups, grid, seed)`` (grouped 5-fold, pooled held-out R^2,
    first best wins) from per-patient statistics."""
    grid = tuple(grid) if grid is not None else _validation_grid()
    cols = np.asarray(cols, int)
    assignment = np.random.default_rng(seed).permutation(len(st.patients)) % n_splits
    tot = st.total()
    sst = tot["yy"] - tot["sy"] ** 2 / tot["n"]
    sse = np.zeros(len(grid))
    for k in range(n_splits):
        test = assignment == k
        if not test.any() or st.n[~test].sum() < 2:
            continue
        tr, te = st.total(~test), st.total(test)
        if len(cols):
            mean, scale, ybar, A, b = _standardised_system(tr, cols)
            lam, V = _eig(A)
            Vb = V.T @ b
        else:
            mean = scale = np.zeros(0); ybar = tr["sy"] / tr["n"]; lam = V = Vb = None
        for j, a in enumerate(grid):
            if len(cols):
                c, w = _coefficients(mean, scale, ybar, V, lam, Vb, a)
            else:
                c, w = ybar, np.zeros(0)
            sse[j] += _sse(te, cols, c, w)
    scores = [0.0 if sst <= 1e-15 else 1.0 - v / sst for v in sse]
    best_j, best = 0, -np.inf
    for j, sc in enumerate(scores):
        if sc > best:
            best_j, best = j, sc
    return float(grid[best_j]), {"cv_r2": dict(zip([str(g) for g in grid], scores))}


def fit_from_stats(st: PatientStats, cols: Sequence[int], alpha: float) -> tuple[float, np.ndarray]:
    """Intercept and raw-scale weights of ``_RidgeFit`` on all rows summarised by ``st``."""
    cols = np.asarray(cols, int)
    tot = st.total()
    if not len(cols):
        return tot["sy"] / tot["n"], np.zeros(0)
    mean, scale, ybar, A, b = _standardised_system(tot, cols)
    if not np.isfinite(alpha):
        return ybar, np.zeros(len(cols))
    beta = np.linalg.solve(A + float(alpha) * np.eye(len(cols)), b)
    w = beta / scale
    return ybar - float(mean @ w), w


# --------------------------------------------------------------------------
# The registered relation's blocks for one hypothesis
# --------------------------------------------------------------------------

def hypothesis_seed(candidate_hash: str) -> int:
    return int(str(candidate_hash)[:8], 16)


def content_hash(*parts: Any) -> str:
    """Deterministic stand-in candidate hash (folds where the hypothesis was not frozen,
    development columns): sha256 of the hypothesis identity."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def relation_blocks(V: np.ndarray, proposal: np.ndarray, *, feature_form: str, candidate_hash: str,
                    n_rbf: int = 24) -> dict:
    """Columns the registered relation builds from H's variable values V (n x k), fitted on the
    proposal rows exactly as ``FittedRelation(model="rbf")``: ``linear`` (V), ``joint`` (one random
    Fourier map of [V, feature], k * n_rbf features, seed = candidate hash), ``additive`` (one map per
    variable, seeds + 1 + j; k >= 2 only: the v0.4/v0.5 order>=2 competitor), ``extra`` (pairwise
    products, k >= 3)."""
    from morpheus_encoder.general_discovery import Candidate, _RBFMap
    V = np.asarray(V, float).reshape(len(V), -1)
    k = V.shape[1]
    cand = Candidate(tuple(range(k)), feature_form=feature_form,
                     weights=(1.0,) * k if feature_form == "linear_subspace" else ())
    seed = hypothesis_seed(candidate_hash)
    feat = cand.feature(V).reshape(-1, 1)
    joint_in = np.column_stack((V, feat))
    jm = _RBFMap(joint_in[proposal], n_features=n_rbf * max(k, 1), seed=seed)
    out = {"linear": V, "joint": jm.apply(joint_in), "additive": None, "extra": None}
    if k >= 2:
        out["additive"] = np.column_stack([_RBFMap(V[proposal][:, [j]], n_features=n_rbf, seed=seed + 1 + j)
                                           .apply(V[:, [j]]) for j in range(k)])
    if k >= 3:
        out["extra"] = np.column_stack([(V[:, i] * V[:, j]) for i in range(k) for j in range(i + 1, k)])
    return out


# --------------------------------------------------------------------------
# One (target, fold): every hypothesis at every level
# --------------------------------------------------------------------------

@dataclass
class FoldData:
    """Arrays of one registered fold: response, full registered competitor (B3 order), patient codes,
    proposal / held-out row indices, competitor names and optional extra competitors (e.g. the v0.4
    comparison model) keyed by name."""
    y: np.ndarray
    B: np.ndarray
    baseline_names: list
    groups: np.ndarray
    proposal: np.ndarray
    replication: np.ndarray
    extra_competitors: dict = field(default_factory=dict)


class FoldLadder:
    """Shared statistics of one fold's competitor(s); ``evaluate`` scores one hypothesis."""

    def __init__(self, fd: FoldData, levels: Mapping[str, Sequence[int]] | None = None,
                 grid: Sequence[float] | None = None):
        self.fd = fd
        self.grid = tuple(grid) if grid is not None else _validation_grid()
        self.levels = dict(levels) if levels is not None else ladder_levels(fd.baseline_names)["levels"]
        p, r = np.asarray(fd.proposal), np.asarray(fd.replication)
        self.p, self.r = p, r
        self.y = np.asarray(fd.y, float)
        self.g = np.asarray(fd.groups)
        # competitor blocks: the registered B3 and any extra comparison model, side by side
        self.comp_names = ["B3"] + list(fd.extra_competitors)
        blocks = [np.asarray(fd.B, float)] + [np.asarray(v, float) for v in fd.extra_competitors.values()]
        self.comp_offsets, off = {}, 0
        for name, b in zip(self.comp_names, blocks):
            self.comp_offsets[name] = np.arange(off, off + b.shape[1])
            off += b.shape[1]
        self.Bcat = np.column_stack(blocks)
        self.B_stats = patient_stats(self.Bcat[p], self.y[p], self.g[p])
        self.held_patients = np.unique(self.g[r])

    def competitor_cols(self, level: str) -> np.ndarray:
        if level in self.levels:
            return self.comp_offsets["B3"][np.asarray(self.levels[level], int)]
        return self.comp_offsets[level]

    def evaluate(self, V: np.ndarray, *, feature_form: str, candidate_hash: str,
                 levels: Sequence[str] = LEVELS, extra_levels: Sequence[str] = (),
                 v04_check: str | None = None) -> dict:
        """Delta_k for one hypothesis with variable values V (all rows of the fold).

        Models per level L: ``ref`` = B_L, ``lin`` = linear H + B_L, ``full`` = registered
        relation's full design + B_L. ``v04_check`` (an extra competitor's name) also fits the
        v0.4 relation's competitor (k = 1: linear H + B; k >= 2: additive maps + products + B) so
        ``full - v04_competitor`` reproduces a registered v0.4 replication gain."""
        p, r, y, g = self.p, self.r, self.y, self.g
        blk = relation_blocks(V, p, feature_form=feature_form, candidate_hash=candidate_hash)
        k = blk["linear"].shape[1]
        parts = {"linear": blk["linear"], "joint": blk["joint"]}
        if blk["extra"] is not None:
            parts["extra"] = blk["extra"]
        if blk["additive"] is not None:
            parts["additive"] = blk["additive"]
        names, off, E = {}, 0, []
        for nm, X in parts.items():
            names[nm] = np.arange(off, off + X.shape[1]); off += X.shape[1]; E.append(X)
        E = np.column_stack(E)
        e = E.shape[1]
        M_p = np.column_stack((E[p], self.Bcat[p]))
        st = patient_stats(M_p, y[p], g[p], B_stats=self.B_stats, n_extra=e)
        M_r = np.column_stack((E[r], self.Bcat[r]))
        seed = hypothesis_seed(candidate_hash)
        full_core = names["joint"] if "extra" not in names else np.concatenate([names["joint"], names["extra"]])
        out: dict = {"k": int(k), "seed": seed, "levels": {}}
        held = {str(q): np.flatnonzero(g[r] == q) for q in self.held_patients}
        yr = y[r]

        def fit_pred(cols):
            a, _ = cv_alpha_from_stats(st, cols, seed, self.grid)
            c, w = fit_from_stats(st, cols, a)
            return a, c + M_r[:, cols] @ w

        def r2(pred):
            ss = float(np.sum((yr - yr.mean()) ** 2))
            return 0.0 if ss <= 1e-15 else float(1.0 - np.sum((yr - pred) ** 2) / ss)

        for lv in list(levels) + list(extra_levels):
            bc = e + self.competitor_cols(lv)
            models = {"ref": bc, "lin": np.concatenate([names["linear"], bc]),
                      "full": np.concatenate([full_core, bc])}
            if v04_check == lv:
                comp = names["linear"] if k == 1 else np.concatenate(
                    [names["additive"]] + ([names["extra"]] if "extra" in names else []))
                models["v04_competitor"] = np.concatenate([comp, bc])
            res = {"width": int(len(bc) - 0)}
            per_patient = {}
            for m, cols in models.items():
                a, pred = fit_pred(cols)
                res[f"alpha_{m}"] = a
                res[f"r2_{m}"] = r2(pred)
                per_patient[m] = {q: float(np.sum((yr[ix] - pred[ix]) ** 2)) for q, ix in held.items()}
            res["delta"] = res["r2_full"] - res["r2_ref"]
            res["delta_linear"] = res["r2_lin"] - res["r2_ref"]
            res["delta_beyond_linear"] = res["r2_full"] - res["r2_lin"]
            if "r2_v04_competitor" in res:
                res["v04_relation_gain"] = res["r2_full"] - res["r2_v04_competitor"]
            res["sse_by_patient"] = per_patient
            out["levels"][lv] = res
        out["y_by_patient"] = {q: {"n": int(len(ix)), "sy": float(yr[ix].sum()), "yy": float(yr[ix] @ yr[ix])}
                               for q, ix in held.items()}
        return out


# --------------------------------------------------------------------------
# Pooling over folds, patient bootstrap, tiers
# --------------------------------------------------------------------------

def _pooled_r2(n, sy, yy, sse):
    sst = yy - sy * sy / n
    return np.where(sst > 1e-15, 1.0 - sse / np.where(sst > 1e-15, sst, 1.0), 0.0)


def pool_folds(fold_results: Mapping[int, dict], *, levels: Sequence[str] = LEVELS, n_boot: int = 1000,
               seed: int = 0) -> dict:
    """Pooled Delta_k (R^2 of the pooled out-of-fold predictions) and patient-bootstrap CIs.

    ``fold_results[f]`` is ``FoldLadder.evaluate`` output. Patients are keyed (fold, code); each is
    held out in exactly one fold."""
    keys, ystat = [], []
    for f, res in sorted(fold_results.items()):
        for q, s in res["y_by_patient"].items():
            keys.append((f, q)); ystat.append((s["n"], s["sy"], s["yy"]))
    ystat = np.asarray(ystat, float)
    models = ("ref", "lin", "full")
    sse = {}
    for lv in levels:
        for m in models:
            sse[(lv, m)] = np.asarray([fold_results[f]["levels"][lv]["sse_by_patient"][m][q] for f, q in keys])
    n_pat = len(keys)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat) for _ in range(int(n_boot))]
                 ).astype(float)
    ones = np.ones((1, n_pat))
    out: dict = {"n_patients": n_pat, "folds": sorted(int(f) for f in fold_results), "n_boot": int(n_boot),
                 "levels": {}}
    for lv in levels:
        r2 = {}
        r2b = {}
        for m in models:
            tot = ones @ np.column_stack((ystat, sse[(lv, m)]))
            r2[m] = float(_pooled_r2(*tot[0]))
            bt = W @ np.column_stack((ystat, sse[(lv, m)]))
            r2b[m] = _pooled_r2(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])
        entry = {"r2_competitor": r2["ref"], "r2_with_linear_H": r2["lin"], "r2_with_H": r2["full"]}
        for name, (a, b) in (("delta", ("full", "ref")), ("delta_linear", ("lin", "ref")),
                             ("delta_beyond_linear", ("full", "lin"))):
            d = r2b[a] - r2b[b]
            entry[name] = r2[a] - r2[b]
            entry[name + "_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out["levels"][lv] = entry
    return out


def shape_label(lin_ci: Sequence[float], beyond_ci: Sequence[float]) -> str:
    """The v0.5 receipt rule (``_linear_decomposition``) on pooled CIs."""
    lin_lo, non_lo = lin_ci[0], beyond_ci[0]
    return ("linear_and_beyond_linear" if lin_lo > 0 and non_lo > 0 else
            "linear_carries_it" if lin_lo > 0 else
            "beyond_linear_only" if non_lo > 0 else "not_resolved")


def assign_tier(pooled: dict) -> dict:
    """Tier from the pooled Delta_0 and Delta_3 (point estimates and patient-bootstrap CIs).

    simple association (novelty unchecked): Delta_0 CI > 0 and Delta_3 CI includes 0;
    irreducible: Delta_0 CI > 0, Delta_3 CI > 0 and Delta_3 / Delta_0 >= 0.5;
    partly absorbed: Delta_0 CI > 0, Delta_3 CI > 0 and Delta_3 / Delta_0 < 0.5;
    none: everything else (incl. Delta_3 CI entirely below 0, flagged)."""
    L = pooled["levels"]
    d0, d3 = L["N"]["delta"], L["B3"]["delta"]
    lo0 = L["N"]["delta_ci95"][0]
    lo3, hi3 = L["B3"]["delta_ci95"]
    ratio = d3 / d0 if d0 > 0 else None
    if lo0 > 0 and lo3 <= 0 <= hi3:
        tier = "simple association (novelty unchecked)"
    elif lo0 > 0 and lo3 > 0:
        tier = "irreducible" if ratio is not None and ratio >= 0.5 else "partly absorbed"
    else:
        tier = "none"
    drops = {lv: L[prev]["delta"] - L[lv]["delta"] for prev, lv in zip(LEVELS[:-1], LEVELS[1:])}
    absorb = max(drops, key=lambda lv: drops[lv]) if d0 > 0 and max(drops.values()) > 0 else None
    # Not a tier (the tiers are defined on Delta_N): a gain that is resolved only once the comparison
    # model has removed variance the hypothesis does not explain (Delta_N CI includes 0, Delta_B3 CI > 0).
    revealed = bool(lo0 <= 0 and lo3 > 0)
    return {"tier": tier, "delta3_over_delta0": ratio, "delta3_ci_below_zero": bool(hi3 < 0),
            "resolved_only_after_adjustment": revealed,
            "drop_by_step": drops, "absorbing_level": absorb,
            "absorbing_explanation": LEVEL_STEP_NAME.get(absorb) if absorb else None,
            "absorbed_fraction": (drops[absorb] / d0) if absorb else None}


def within_patient_r(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> float:
    """Pearson r after centring both within patient (the leak guard's statistic)."""
    a = np.asarray(a, float).copy(); b = np.asarray(b, float).copy()
    for q in np.unique(groups):
        m = groups == q
        a[m] -= a[m].mean(); b[m] -= b[m].mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else 0.0


# --------------------------------------------------------------------------
# Expected-by-construction flag for programme targets
# --------------------------------------------------------------------------

_SET_PREFIXES = ("programmes:", "context_neighbour_mean:", "nbr18:", "nbr:", "mp:")
OVERLAP_THRESHOLD = 0.2


def _gene_set_of(variable: str, gene_sets: Mapping[str, Sequence[str]]):
    """(set key, is neighbour mean) when a bank column is a gene-set score or its neighbour mean."""
    v, neighbour = str(variable), False
    changed = True
    while changed:
        changed = False
        for p in _SET_PREFIXES:
            if v.startswith(p):
                neighbour |= p in ("nbr:", "nbr18:", "context_neighbour_mean:")
                v, changed = v[len(p):], True
    return (v, neighbour) if v in gene_sets else (None, neighbour)


def target_programme_overlap(variable_names: Sequence[str], target: str,
                             gene_sets: Mapping[str, Sequence[str]], threshold: float = OVERLAP_THRESHOLD) -> dict | None:
    """For a programme target (``nbr:X`` / ``own:X``, X a Gavish programme): is any hypothesis variable
    the target programme itself, a Gavish/Hallmark set sharing >= ``threshold`` of its genes with X
    (as a fraction of either set), the neighbour mean of such a set, or a raw gene of X?
    ``gene_sets`` maps ``gavish:<name>`` / ``hallmark:<name>`` to gene lists (the bank's sets).
    Returns None for other targets."""
    t = str(target)
    if not t.startswith(("nbr:", "own:")):
        return None
    key = "gavish:" + t.split(":", 1)[1]
    if key not in gene_sets:
        return {"target_programme_overlap": None, "reason": "target programme %r not among the bank's sets" % key}
    X = set(gene_sets[key])
    rows, flag = [], False
    for v in variable_names:
        v = str(v)
        if v.startswith("raw_genes:"):
            g = v.split(":", 1)[1]
            hit = g in X
            rows.append({"variable": v, "kind": "raw gene", "in_target_programme": hit})
            flag |= hit
            continue
        s, neighbour = _gene_set_of(v, gene_sets)
        if s is None:
            rows.append({"variable": v, "kind": "not a gene-set score"})
            continue
        A = set(gene_sets[s])
        inter = len(A & X)
        fa, fx = inter / max(len(A), 1), inter / max(len(X), 1)
        hit = s == key or max(fa, fx) >= threshold
        rows.append({"variable": v, "kind": "neighbour mean of a gene set" if neighbour else "own-spot gene-set score",
                     "gene_set": s, "is_target_programme": s == key, "shared_genes": inter,
                     "overlap_fraction_of_variable_set": round(fa, 4), "overlap_fraction_of_target_set": round(fx, 4),
                     "flag": bool(hit)})
        flag |= hit
    return {"target_programme_overlap": bool(flag), "target_set": key, "threshold": threshold, "variables": rows}


# --------------------------------------------------------------------------
# In-run ladder (protocol v0.6): per-level pass rule and tier from the passes
# --------------------------------------------------------------------------

#: A level passes when the held-out gain CI lies above 0, the full model beats the held-out mean
#: (R^2 > 0) and the within-patient permutation null rejects (p <= 0.05): Agent L's v0.6 rule.
DEFAULT_PASS_RULE: dict = {"ci_low_gt": 0.0, "r2_heldout_gt": 0.0,
                           "permutation_null": "within_patient_perm", "p_max": 0.05}
#: First matching rule wins; a hypothesis matching none is "none" (reported, never dropped).
DEFAULT_TIER_RULES: tuple = (
    {"tier": "irreducible", "must_pass": ["N", "B3"], "min_ratio_B3_over_N": 0.5},
    {"tier": "partly absorbed", "must_pass": ["N", "B3"]},
    {"tier": "simple association (novelty unchecked)", "must_pass": ["N"]},
)


def level_passes(entry: Mapping[str, Any], rule: Mapping[str, Any] | None = None) -> dict:
    """Pass decision of one ladder level (``entry``: gain, ci, r2_heldout, nulls{kind: {p}})."""
    rule = dict(DEFAULT_PASS_RULE if rule is None else rule)
    null = (entry.get("nulls") or {}).get(rule["permutation_null"]) or {}
    p = null.get("p")
    checks = {"ci_low": bool(entry["ci"][0] > rule["ci_low_gt"]),
              "r2_heldout": bool(entry["r2_heldout"] > rule["r2_heldout_gt"]),
              "permutation_null": bool(p is not None and p <= rule["p_max"])}
    return {"pass": all(checks.values()), "checks": checks}


def tier_from_levels(levels: Mapping[str, Mapping[str, Any]],
                     rules: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """Tier of one hypothesis in one fold from its per-level passes (first matching rule wins).

    ``levels[L]`` carries ``gain`` and ``pass``; ``min_ratio_B3_over_N`` compares the gains at B3
    and N. Also reports the step that removes most of the gain (``absorbing_level``, as
    :func:`assign_tier`) and the case where only the adjusted level passes."""
    rules = list(DEFAULT_TIER_RULES if rules is None else rules)
    present = [lv for lv in LEVELS if lv in levels]
    passed = {lv: bool(levels[lv]["pass"]) for lv in present}
    gain = {lv: float(levels[lv]["gain"]) for lv in present}
    ratio = gain["B3"] / gain["N"] if {"N", "B3"} <= set(gain) and gain["N"] > 0 else None
    tier, matched = "none", None
    for i, rule in enumerate(rules):
        if not all(passed.get(lv, False) for lv in rule.get("must_pass", [])):
            continue
        if "min_ratio_B3_over_N" in rule and (ratio is None or ratio < float(rule["min_ratio_B3_over_N"])):
            continue
        tier, matched = str(rule["tier"]), i
        break
    drops = {lv: gain[prev] - gain[lv] for prev, lv in zip(present[:-1], present[1:])}
    absorb = (max(drops, key=lambda lv: drops[lv])
              if drops and gain[present[0]] > 0 and max(drops.values()) > 0 else None)
    return {"tier": tier, "rule_index": matched, "passed_levels": [lv for lv in present if passed[lv]],
            "gain_B3_over_N": ratio, "drop_by_step": drops, "absorbing_level": absorb,
            "absorbing_explanation": LEVEL_STEP_NAME.get(absorb) if absorb else None,
            "resolved_only_after_adjustment": bool(passed.get("B3") and not passed.get("N"))}
