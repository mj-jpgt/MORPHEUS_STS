"""Frozen relation f_S of the fixed family (training spec 0.3, 0.7, 0.8, 0.10) as a portable object.

The downstream pipeline applies a relation WITHOUT refitting (replication rule 2.3), so the relation
must exist as numbers, not as a fitted Python object. :class:`FrozenRelation` stores them: the random
Fourier directions and phases, the bandwidth, the ridge coefficients over ``[RFF(S) | S]`` and the
penalty. Conventions follow the fixed relation family (and Agent M's ``relation_v03``):

* per-patient intercepts are within-patient centring of the target, every candidate and every
  baseline covariate; candidates and covariates are then divided by their pooled within-patient SD
  on the cohort's own support rows (``standardise_within``), and so is the target, so a relation
  predicts the target in within-patient SD units on any cohort;
* N (B_nuis) is a ridge on within-centred log depth; B_known adds the Hallmark and Gavish programme
  scores and the composition variables; B_decomp(S) is the best proper-subset partition of S, every
  block with its own RFF map, fitted jointly;
* f_S is fitted to the residual of the target after the baseline, ridge penalty from
  ``ALPHAS`` by leave-one-patient-out CV (patient-grouped K-fold when there are more than 12
  patients), 256 RFF features with the median-heuristic bandwidth on support.

When Agent M's registry carries its own frozen parameters, :meth:`FrozenRelation.from_json` reads
any dict with the same keys, so the downstream code does not depend on how the relation was fitted.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .locks import canonical_hash

D_RFF = 256
ALPHAS = (0.01, 0.1, 1.0, 10.0)
BASE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
RFF_SEED = 20261021


# ----------------------------------------------------------------------------- arithmetic
def within_centre(X, groups) -> np.ndarray:
    X = np.asarray(X, float)
    two = X.reshape(len(X), -1).copy()
    g = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
    sums = np.zeros((g.max() + 1, two.shape[1]))
    np.add.at(sums, g, two)
    cnt = np.bincount(g).astype(float)[:, None]
    two -= (sums / cnt)[g]
    return two.reshape(X.shape)


def standardise_within(X, groups, sd=None):
    """Within-patient centring then division by the pooled within-patient SD (computed here unless
    given). Returns (Z, sd)."""
    Xc = within_centre(X, groups)
    two = Xc.reshape(len(Xc), -1)
    if sd is None:
        sd = two.std(0)
    sd = np.where(np.asarray(sd) > 1e-12, sd, 1.0)
    return (two / sd).reshape(Xc.shape), np.asarray(sd, float)


def rff_dirs(k: int, seed: int = RFF_SEED, d: int = D_RFF):
    rng = np.random.default_rng(seed + 1000 * k)
    return rng.standard_normal((k, d)), rng.uniform(0, 2 * np.pi, d)


def median_bandwidth(Z, n_sub: int = 512, seed: int = 0) -> float:
    Z = np.asarray(Z, float).reshape(len(Z), -1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    S = Z[idx]
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    med = float(np.sqrt(np.median(d2[np.triu_indices(len(S), 1)]))) if len(S) > 1 else 1.0
    return med if med > 1e-9 else 1.0


def rff_features(Z, W, b, bw) -> np.ndarray:
    Z = np.asarray(Z, float).reshape(len(Z), -1)
    return np.sqrt(2.0 / W.shape[1]) * np.cos(Z @ W / bw + b)


def _ridge(F, r, alpha):
    A = F.T @ F + alpha * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ r)


def patient_folds(groups, max_folds: int = 12, seed: int = 0) -> list[np.ndarray]:
    pats = np.array(sorted(np.unique(np.asarray(groups).astype(str))))
    if len(pats) > max_folds:
        rng = np.random.default_rng(seed)
        rng.shuffle(pats)
        return [np.asarray(f) for f in np.array_split(pats, max_folds)]
    return [np.array([p]) for p in pats]


def cv_alpha(F, r, groups, alphas=ALPHAS) -> tuple[float, dict]:
    groups = np.asarray(groups).astype(str)
    folds = patient_folds(groups)
    if len(folds) < 2:
        return float(alphas[len(alphas) // 2]), {}
    # per-fold Gram matrices once; each held-out fold's normal equations are total minus fold
    stats = []
    for held in folds:
        h = np.isin(groups, held)
        if h.all() or not h.any():
            continue
        Fh, rh = F[h], r[h]
        stats.append((Fh.T @ Fh, Fh.T @ rh, float(rh @ rh)))
    G_tot = sum(s[0] for s in stats)
    b_tot = sum(s[1] for s in stats)
    eye = np.eye(F.shape[1])
    sse = {}
    for a in alphas:
        s = 0.0
        for G_h, b_h, rr_h in stats:
            c = np.linalg.solve(G_tot - G_h + a * eye, b_tot - b_h)
            s += rr_h - 2 * float(c @ b_h) + float(c @ G_h @ c)
        sse[float(a)] = s
    return min(sse, key=sse.get), sse


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / sst if sst > 1e-15 else 0.0


# ----------------------------------------------------------------------------- baseline
@dataclass
class Baseline:
    """Ridge on within-standardised covariates (no intercept: everything is within-centred)."""
    names: list
    coef: np.ndarray
    alpha: float

    def predict(self, C) -> np.ndarray:
        if not len(self.names):
            return np.zeros(len(C))
        return np.asarray(C, float).reshape(len(C), -1) @ self.coef


def fit_baseline(C, y, groups, names, alphas=BASE_ALPHAS) -> Baseline:
    C = np.asarray(C, float).reshape(len(y), -1)
    if C.shape[1] == 0:
        return Baseline([], np.zeros(0), 0.0)
    a, _ = cv_alpha(C, y, groups, alphas)
    return Baseline(list(names), _ridge(C, y, a), a)


def crossfit_baseline(C, y, groups, names, k: int = 5, seed: int = 20261016) -> np.ndarray:
    """Out-of-fold baseline predictions by patient folds (the baseline is refit on the cohort it is
    applied to, as 2.3 allows; it never sees the held-out patient)."""
    C = np.asarray(C, float).reshape(len(y), -1)
    if C.shape[1] == 0:
        return np.zeros(len(y))
    groups = np.asarray(groups).astype(str)
    pats = np.array(sorted(np.unique(groups)))
    np.random.default_rng(seed).shuffle(pats)
    out = np.zeros(len(y))
    for held in np.array_split(pats, min(k, len(pats))):
        h = np.isin(groups, held)
        if (~h).sum() == 0:
            continue
        out[h] = fit_baseline(C[~h], y[~h], groups[~h], names).predict(C[h])
    return out


# ----------------------------------------------------------------------------- frozen relation
@dataclass
class FrozenRelation:
    candidates: list                 # candidate names (variable, version)
    target: str
    baseline: str                    # "nuis" | "known"
    blocks: list                     # partition of candidate indices; one RFF map per block
    W: list                          # per block (k_b, D)
    b: list                          # per block (D,)
    bw: list                         # per block bandwidth
    coef: np.ndarray                 # over [RFF blocks..., linear terms of all candidates]
    alpha: float
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.candidates)

    def design(self, Z) -> np.ndarray:
        Z = np.asarray(Z, float).reshape(len(Z), -1)
        parts = [rff_features(Z[:, list(bl)], np.asarray(W), np.asarray(b), bw)
                 for bl, W, b, bw in zip(self.blocks, self.W, self.b, self.bw)]
        return np.column_stack(parts + [Z])

    def predict(self, Z) -> np.ndarray:
        return self.design(Z) @ np.asarray(self.coef)

    def gradient(self, Z) -> np.ndarray:
        """d f / d z_j per row (n, k), in closed form."""
        Z = np.asarray(Z, float).reshape(len(Z), -1)
        coef = np.asarray(self.coef)
        G = np.tile(coef[-self.k:], (len(Z), 1))
        off = 0
        for bl, W, b, bw in zip(self.blocks, self.W, self.b, self.bw):
            W, b = np.asarray(W), np.asarray(b)
            D = W.shape[1]
            c = coef[off:off + D]
            s = -np.sqrt(2.0 / D) * np.sin(Z[:, list(bl)] @ W / bw + b)        # (n, D)
            G[:, list(bl)] += (s * c) @ W.T / bw
            off += D
        return G

    def to_json(self) -> dict:
        return {"candidates": list(self.candidates), "target": self.target, "baseline": self.baseline,
                "blocks": [list(map(int, bl)) for bl in self.blocks],
                "W": [np.asarray(w).tolist() for w in self.W], "b": [np.asarray(x).tolist() for x in self.b],
                "bw": [float(x) for x in self.bw], "coef": np.asarray(self.coef).tolist(),
                "alpha": float(self.alpha), "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "FrozenRelation":
        return cls(list(d["candidates"]), d["target"], d["baseline"], [tuple(bl) for bl in d["blocks"]],
                   [np.asarray(w) for w in d["W"]], [np.asarray(x) for x in d["b"]], list(d["bw"]),
                   np.asarray(d["coef"]), float(d["alpha"]), dict(d.get("meta", {})))

    def params_hash(self) -> str:
        return canonical_hash(self.to_json())


def fit_relation(Z, r, groups, candidates, target, baseline, blocks=None, alphas=ALPHAS,
                 seed: int = RFF_SEED) -> FrozenRelation:
    """Fit f_S (or a decomposition, when ``blocks`` partitions S) to the baseline residual ``r``."""
    Z = np.asarray(Z, float).reshape(len(r), -1)
    k = Z.shape[1]
    blocks = [tuple(range(k))] if blocks is None else [tuple(b) for b in blocks]
    Ws, bs, bws = [], [], []
    for bl in blocks:
        W, b = rff_dirs(len(bl), seed)
        Ws.append(W)
        bs.append(b)
        bws.append(median_bandwidth(Z[:, list(bl)]))
    rel = FrozenRelation(list(candidates), target, baseline, blocks, Ws, bs, bws, np.zeros(D_RFF * len(blocks) + k),
                         0.0)
    F = rel.design(Z)
    a, sse = cv_alpha(F, r, groups, alphas)
    rel.coef, rel.alpha = _ridge(F, r, a), a
    rel.meta = {"cv_sse": sse, "n_support_rows": int(len(r)),
                "n_support_patients": int(len(np.unique(np.asarray(groups).astype(str))))}
    return rel


def proper_partitions(k: int) -> list[list[tuple]]:
    """Every partition of range(k) into >= 2 non-empty blocks (training spec 0.7)."""
    items = list(range(k))

    def parts(seq):
        if not seq:
            yield []
            return
        first, rest = seq[0], seq[1:]
        for p in parts(rest):
            for i in range(len(p)):
                yield p[:i] + [(first,) + p[i]] + p[i + 1:]
            yield [(first,)] + p
    return [sorted(p) for p in parts(items) if len(p) >= 2]


def fit_decomposition(Z, r, groups, candidates, target, baseline) -> FrozenRelation | None:
    """B_decomp(S): best proper partition by grouped-CV error, frozen with that partition."""
    Z = np.asarray(Z, float).reshape(len(r), -1)
    if Z.shape[1] < 2:
        return None
    best = None
    for p in proper_partitions(Z.shape[1]):
        rel = fit_relation(Z, r, groups, candidates, target, baseline, blocks=p)
        err = min(rel.meta["cv_sse"].values()) if rel.meta["cv_sse"] else 0.0
        if best is None or err < best[0]:
            best = (err, rel)
    best[1].meta["decomposition"] = [list(b) for b in best[1].blocks]
    return best[1]


def standardised_contributions(rel: FrozenRelation, Z, r) -> list[float]:
    """Drop in R^2 (against ``r``) when candidate j is replaced by its support mean (0 after
    within-standardisation) with the others held (characterisation 1.1)."""
    Z = np.asarray(Z, float).reshape(len(r), -1)
    full = r2(r, rel.predict(Z))
    out = []
    for j in range(rel.k):
        Zj = Z.copy()
        Zj[:, j] = 0.0
        out.append(full - r2(r, rel.predict(Zj)))
    return out


def dominant_term(rel: FrozenRelation, Z, r) -> dict:
    """The candidate with the largest standardised contribution and the sign of its mean partial
    derivative on support rows (the 'dominant term' of replication rule 2.2)."""
    contrib = standardised_contributions(rel, Z, r)
    j = int(np.argmax(contrib))
    g = rel.gradient(Z)[:, j]
    return {"index": j, "candidate": rel.candidates[j], "contribution": float(contrib[j]),
            "sign": int(np.sign(g.mean())), "mean_partial_derivative": float(g.mean()),
            "contributions": [float(c) for c in contrib]}


def per_patient_partial_signs(Z, r, groups, j: int, min_rows: int = 10) -> dict:
    """Within each patient, the sign of candidate j's coefficient in an OLS of the residual target on
    all candidates of S (partial association)."""
    Z = np.asarray(Z, float).reshape(len(r), -1)
    groups = np.asarray(groups).astype(str)
    out = {}
    for p in np.unique(groups):
        m = groups == p
        if m.sum() < max(min_rows, Z.shape[1] + 2):
            continue
        X = np.column_stack([np.ones(m.sum()), Z[m]])
        beta = np.linalg.lstsq(X, r[m], rcond=None)[0]
        out[p] = int(np.sign(beta[1 + j]))
    return out


def all_subsets(k: int) -> list[tuple]:
    return [c for n in range(1, k) for c in itertools.combinations(range(k), n)]
