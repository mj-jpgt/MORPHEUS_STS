"""Hypothesis-search methods behind one interface.

Every method receives a ``MethodInput`` built from PROPOSAL rows only (the
held-out patients' rows are never sliced in) and returns a ranked list of
``Hypothesis`` objects (a set of design-column names + a relation class).  The
harness truncates every list to the same budget; a method never sees the
budget-B lists of another method or anything about the planted truth.

Methods (``METHODS``):
  enumeration        the registered enumeration proposer (``enumerate_design``)
                     over the family's views, exactly as the runner calls it
  fm_enumeration     the identical enumeration restricted to the frozen
                     molecular FM views (scGPT, SCimilarity)
  gbm_interaction    histogram gradient boosting, inner patient-grouped CV,
                     out-of-patient permutation importance -> shortlist ->
                     Friedman H-statistic (unnormalised) for pairs
  ebm_ga2m           explainable boosting machine (interpret), mains + FAST
                     pairwise interactions; skipped when interpret is absent
  pysr               symbolic regression on a shortlist; skipped when PySR or
                     Julia is absent
  spatial_lag_screen bivariate Moran-type screen: correlation of the response
                     with each variable's within-patient neighbour mean, given
                     the variable's own-spot value
  residual_generator ``hypothesis_generator.generate_hypotheses`` as it is
  residual_bank_generator  hook for Agent G's v0.6 generator
                     (``morpheus_encoder.residual_bank_generator``)
  linear_interaction_screen  lasso over mains, neighbour means and pairwise products
  rulefit            imodels RuleFit on a marginal shortlist
  nid_interaction    neural interaction detection (L1 first-layer MLP, Tsang et al.)
  pysr_shortlist     PySR on a 30-variable shortlist (needs the Julia runtime juliacall installs)
  greedy_flex_forward  forward selection scored by a flexible relation (the simpler equivalent of PySR)
  random_search_matched / heuristic_search  set searches matched on the number of sets evaluated
  oracle             handed the truth (tests and reference rows only)

Every method reports ``info["n_relationships_tested"]``: the number of variable sets (or terms,
rules, expressions) the method scored against the response.  For the registered proposers it is
counted from their own budget formulas (see each class).
"""
from __future__ import annotations

import importlib
import importlib.util
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .substrate import CONTEXT_ROLE_VIEWS, FM_VIEWS, BenchSubstrate, neighbour_mean
from .tasks import Truth, World, grouped_fold_ids

HYPOTHESIS_CLASSES = ("single", "additive", "product", "context", "spatial_lag", "higher_order")


@dataclass(frozen=True)
class Hypothesis:
    variables: tuple
    relation: str
    score: float = float("nan")
    detail: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(sorted(map(str, self.variables))))
        if self.relation not in HYPOTHESIS_CLASSES:
            raise ValueError(f"unknown hypothesis class {self.relation!r}")

    @property
    def key(self) -> tuple:
        return (self.variables, self.relation)


@dataclass
class MethodInput:
    """Proposal rows only.  ``neighbours`` is re-indexed into proposal-row space."""

    X: np.ndarray
    y: np.ndarray
    comparison: np.ndarray
    groups: np.ndarray
    neighbours: np.ndarray
    coords: np.ndarray | None
    names: list
    views: dict          # searchable views of this world's family -> design column indices
    roles: dict
    family: str
    seed: int
    device: str = "cpu"
    n_threads: int = 4
    comparison_full: np.ndarray | None = None  # B3 columns (proposal rows), for multi-level generators
    comparison_names: list | None = None


@dataclass
class MethodOutput:
    hypotheses: list
    info: dict
    seconds: float
    available: bool = True


def method_input(sub: BenchSubstrate, world: World, *, seed: int, device: str = "cpu", n_threads: int = 4,
                 views: tuple | None = None) -> MethodInput:
    P = sub.proposal
    remap = -np.ones(sub.n, dtype=np.int64)
    remap[P] = np.arange(len(P))
    nb = sub.neighbours[P]
    nb = np.where(nb >= 0, remap[np.where(nb >= 0, nb, 0)], -1)
    if (nb[sub.neighbours[P] >= 0] < 0).any():
        raise AssertionError("a proposal row has a held-out neighbour")
    fviews = sub.family_views(world.family, restrict=views)
    return MethodInput(X=sub.X[P], y=np.asarray(world.response, dtype=float)[P],
                       comparison=np.asarray(sub.level_matrix(world.truth.level), dtype=float)[P],
                       groups=sub.groups[P], neighbours=nb, coords=None if sub.coords is None else sub.coords[P],
                       names=sub.names, views=fviews, roles={v: sub.roles[v] for v in fviews},
                       family=world.family, seed=seed, device=device, n_threads=n_threads,
                       comparison_full=np.asarray(sub.comparison, dtype=float)[P],
                       comparison_names=list(sub.comparison_names))


def _searchable(inp: MethodInput) -> np.ndarray:
    return np.sort(np.concatenate([np.asarray(c, dtype=int) for c in inp.views.values()]))


def _role_of(inp: MethodInput) -> dict:
    return {int(c): inp.roles[v] for v, cols in inp.views.items() for c in cols}


def classify_columns(cols, inp: MethodInput, kind: str) -> str:
    """Map a proposer's structure to the shared hypothesis vocabulary."""
    role = _role_of(inp)
    has_context = any(role.get(int(c)) == "context" for c in cols)
    if kind == "single":
        return "single"
    if kind == "additive":
        return "additive"
    if kind in ("context_conditioned", "context"):
        return "context"
    # product / interaction
    if has_context:
        return "context"
    return "product" if len(cols) == 2 else ("single" if len(cols) == 1 else "higher_order")


def candidate_to_hypothesis(c, inp: MethodInput) -> Hypothesis:
    cols = tuple(int(i) for i in c.variables)
    prov = (c.metadata or {}).get("provenance") or {}
    kind = prov.get("hypothesis_kind")
    if kind is None:
        if c.feature_form == "linear_subspace":
            kind = "single" if len(cols) == 1 else "additive"
        else:
            kind = "product"
    kind = {"additive_set": "additive", "interaction": "product"}.get(kind, kind)
    names = [inp.names[i] for i in cols]
    detail = {"proposal_family": c.proposal_family}
    if prov:
        detail.update({k: prov.get(k) for k in ("branch", "hypothesis_kind", "recurrent_in_generator")})
    return Hypothesis(tuple(names), classify_columns(cols, inp, kind), float(c.proposal_score), detail)


def dedupe(hyps: list) -> list:
    seen, out = set(), []
    for h in hyps:
        if h.key not in seen:
            seen.add(h.key)
            out.append(h)
    return out


class Method:
    name = "method"
    budget_dependent = False  # True: the harness re-runs the method at every budget

    def available(self) -> tuple[bool, str]:
        return True, ""

    def propose(self, inp: MethodInput, budget: int) -> tuple[list, dict]:
        raise NotImplementedError

    def run(self, inp: MethodInput, budget: int) -> MethodOutput:
        ok, why = self.available()
        if not ok:
            return MethodOutput([], {"unavailable": why}, 0.0, available=False)
        from threadpoolctl import threadpool_limits

        t = time.time()
        with threadpool_limits(limits=inp.n_threads):
            hyps, info = self.propose(inp, budget)
        hyps = dedupe(hyps)
        return MethodOutput(hyps, info, time.time() - t)


# ----------------------------------------------------------------- enumeration
class Enumeration(Method):
    """``hypothesis_generator.enumerate_design`` as the runner calls it for a family."""

    name = "enumeration"
    budget_dependent = True
    restrict: tuple | None = None

    def propose(self, inp, budget):
        from morpheus_encoder.hypothesis_generator import CONTEXT_VIEWS, MOLECULAR_VIEWS, enumerate_design

        views = {v: c for v, c in inp.views.items()
                 if self.restrict is None or v in self.restrict or v in CONTEXT_ROLE_VIEWS}
        require_context = inp.family == "context"
        found = enumerate_design(inp.X, inp.y, views=views, enumerate_views=list(views),
                                 max_candidates=budget, require_context=require_context,
                                 molecular_views=MOLECULAR_VIEWS, context_views=CONTEXT_VIEWS)
        p = int(sum(len(c) for c in views.values()))
        if require_context:  # n_molecular x n_context product screen (enumerate_design defaults)
            tested = 24 * 8
        else:  # propose_candidates: every single screened, all pairs of the screened pool, bounded forward orders
            pool = min(p, max(8, int(np.ceil(np.sqrt(budget * 5 * 4)))))
            tested = p + pool * (pool - 1) // 2 + 3 * max(8, budget // 2) * pool
        return [candidate_to_hypothesis(c, inp) for c in found], {"views": sorted(views),
                                                                  "require_context": require_context,
                                                                  "n_relationships_tested": int(tested)}


class FMEnumeration(Enumeration):
    name = "fm_enumeration"
    restrict = FM_VIEWS


# ------------------------------------------------------------- gradient boosting
def _grouped_r2(y, pred):
    return 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)


def _used_features(model, width: int) -> np.ndarray:
    """Columns split on by any tree of a fitted HistGradientBoostingRegressor."""
    try:
        used = [p.nodes["feature_idx"][~p.nodes["is_leaf"].astype(bool)]
                for trees in model._predictors for p in trees]
        return np.unique(np.concatenate(used)) if used else np.arange(0)
    except Exception:  # private layout changed: fall back to every column
        return np.arange(width)


class GBMInteraction(Method):
    """Gradient-boosting interaction search (sklearn HistGradientBoosting).

    1. two configurations scored by 3-fold patient-grouped CV on proposal rows;
    2. out-of-patient permutation importance (chosen configuration, each inner
       held-out fold) over every searchable column -> shortlist of ``k``;
    3. on the model refit to all proposal rows, partial dependence on ``m``
       sampled rows: main-effect strength sd(PD_j), and Friedman's unnormalised
       H for each shortlisted pair, sd(PD_jk - PD_j - PD_k).
    Hypotheses: singles (sd PD_j), pairs as product/context (H), pairs as
    additive (min of the two main-effect strengths); one list sorted by score."""

    name = "gbm_interaction"
    configs = ({"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 100, "min_samples_leaf": 40},
               {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 150, "min_samples_leaf": 40})

    def __init__(self, shortlist: int = 20, pd_rows: int = 80):
        self.k, self.m = shortlist, pd_rows

    def _model(self, cfg, seed):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(early_stopping=False, l2_regularization=1.0, random_state=seed, **cfg)

    def propose(self, inp, budget):
        cols = _searchable(inp)
        X, y = np.asarray(inp.X[:, cols], dtype=np.float32), inp.y
        folds = grouped_fold_ids(inp.groups, 3, inp.seed)
        rng = np.random.default_rng(inp.seed)
        scores, fitted = [], []
        stage = {"start": time.time()}
        for cfg in self.configs:
            pred, models = np.empty(len(y)), []
            for k in np.unique(folds):
                te = folds == k
                m = self._model(cfg, inp.seed).fit(X[~te], y[~te])
                pred[te] = m.predict(X[te])
                models.append((te, m))
            scores.append(_grouped_r2(y, pred))
            fitted.append(models)
        best = int(np.argmax(scores))
        stage["inner_cv"] = time.time()
        importance = np.zeros(X.shape[1])
        for te, m in fitted[best]:
            Xt, yt = X[te], y[te]
            sub = rng.choice(len(yt), size=min(1500, len(yt)), replace=False)
            Xt, yt = Xt[sub], yt[sub]
            base = _grouped_r2(yt, m.predict(Xt))
            for j in _used_features(m, X.shape[1]):  # an unused column has importance exactly 0
                Xp = Xt.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                importance[j] += base - _grouped_r2(yt, m.predict(Xp))
        importance /= len(fitted[best])
        order = np.argsort(-importance, kind="stable")
        short = [int(j) for j in order[:self.k] if importance[j] > 0]
        stage["importance"] = time.time()
        final = self._model(self.configs[best], inp.seed).fit(X, y)
        stage["final_fit"] = time.time()
        rows = rng.choice(len(y), size=min(self.m, len(y)), replace=False)
        Xr = X[rows]
        pd1 = {}
        for j in short:
            Z = np.repeat(Xr, len(rows), axis=0)
            Z[:, j] = np.tile(Xr[:, j], len(rows))
            pd = final.predict(Z).reshape(len(rows), len(rows)).mean(0)  # PD at each sampled value of x_j
            pd1[j] = pd - pd.mean()
        hyps = []
        role = _role_of(inp)
        for j in short:
            hyps.append(Hypothesis((inp.names[cols[j]],), "single", float(pd1[j].std()),
                                   {"importance": float(importance[j])}))
        for a_i, a in enumerate(short):
            for b in short[a_i + 1:]:
                Z = np.repeat(Xr, len(rows), axis=0)
                Z[:, a] = np.tile(Xr[:, a], len(rows))
                Z[:, b] = np.tile(Xr[:, b], len(rows))
                pd = final.predict(Z).reshape(len(rows), len(rows)).mean(0)
                h = float(np.std(pd - pd.mean() - pd1[a] - pd1[b]))
                names = (inp.names[cols[a]], inp.names[cols[b]])
                kind = classify_columns((cols[a], cols[b]), inp, "product")
                hyps.append(Hypothesis(names, kind, h, {"H_unnormalised": h}))
                if not any(role.get(int(cols[c])) == "context" for c in (a, b)):
                    hyps.append(Hypothesis(names, "additive", float(min(pd1[a].std(), pd1[b].std())),
                                           {"main_effects": [float(pd1[a].std()), float(pd1[b].std())]}))
        hyps.sort(key=lambda h: (-h.score, h.key))
        stage["h_statistic"] = time.time()
        marks = list(stage.items())
        seconds = {k: round(t - marks[i][1], 1) for i, (k, t) in enumerate(marks[1:])}
        return hyps, {"stage_seconds": seconds, "config": self.configs[best], "inner_cv_r2": [float(s) for s in scores],
                      "shortlist": [inp.names[cols[j]] for j in short], "n_features": int(len(cols)),
                      "n_relationships_tested": int(len(cols) + len(short) * (len(short) - 1) // 2)}


# ------------------------------------------------------------------------- EBM
class EBMSearch(Method):
    """Explainable boosting (GA2M): one EBM over every searchable column with
    FAST pairwise interaction detection; hypotheses are its terms ranked by
    term importance, plus additive pairs of its two strongest mains."""

    name = "ebm_ga2m"

    def __init__(self, interactions: int = 32, max_rounds: int = 1000):
        self.interactions, self.max_rounds = interactions, max_rounds

    def available(self):
        try:
            importlib.import_module("interpret.glassbox")
            return True, ""
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, f"interpret not importable ({type(exc).__name__})"

    def propose(self, inp, budget):
        from interpret.glassbox import ExplainableBoostingRegressor

        cols = _searchable(inp)
        X = np.asarray(inp.X[:, cols], dtype=np.float64)
        ebm = ExplainableBoostingRegressor(outer_bags=1, inner_bags=0, interactions=self.interactions,
                                           max_rounds=self.max_rounds, n_jobs=inp.n_threads,
                                           random_state=inp.seed)
        ebm.fit(X, inp.y)
        imp = np.asarray(ebm.term_importances(), dtype=float)
        hyps, mains = [], []
        for t, feats in enumerate(ebm.term_features_):
            names = tuple(inp.names[cols[f]] for f in feats)
            if len(feats) == 1:
                hyps.append(Hypothesis(names, "single", float(imp[t])))
                mains.append((float(imp[t]), int(feats[0])))
            else:
                kind = classify_columns(tuple(cols[f] for f in feats), inp, "product")
                hyps.append(Hypothesis(names, kind, float(imp[t])))
        mains.sort(reverse=True)
        top = mains[:16]
        role = _role_of(inp)
        for i, (sa, a) in enumerate(top):
            for sb, b in top[i + 1:]:
                if role.get(int(cols[a])) != "context" and role.get(int(cols[b])) != "context":
                    hyps.append(Hypothesis((inp.names[cols[a]], inp.names[cols[b]]), "additive", min(sa, sb)))
        hyps.sort(key=lambda h: (-h.score, h.key))
        p = int(len(cols))  # FAST screens every pair of features
        return hyps, {"n_terms": len(ebm.term_features_), "n_features": p,
                      "n_relationships_tested": p + p * (p - 1) // 2}


# ------------------------------------------------------------------------ PySR
class PySRSearch(Method):
    name = "pysr"

    def available(self):
        if importlib.util.find_spec("pysr") is None:
            return False, "pysr not installed in morpheus_venv"
        if shutil.which("julia") is None:
            return False, "julia not on PATH (PySR needs a Julia runtime)"
        return True, ""

    def propose(self, inp, budget):  # pragma: no cover - not available on the box
        raise NotImplementedError("PySR route not wired: unavailable on the box (see DESIGN.md)")


# -------------------------------------------------------------- spatial screen
class SpatialLagScreen(Method):
    """Bivariate Moran-type screen.  For each searchable variable column x_j
    the within-patient k = 6 neighbour mean L_j (row-standardised spatial lag)
    is formed; score = |partial correlation of y with L_j given x_j| (the lag's
    association beyond the variable's own-spot value, so an own-spot effect is
    not reported as spatial).  Hypotheses: spatial_lag(x_j), ranked."""

    name = "spatial_lag_screen"

    def propose(self, inp, budget):
        role = _role_of(inp)
        cols = np.asarray([c for c in _searchable(inp) if role[int(c)] == "variable"], dtype=int)
        X = np.asarray(inp.X[:, cols], dtype=float)
        L = neighbour_mean(X, inp.neighbours)
        y = inp.y - inp.y.mean()

        def z(a):
            a = a - a.mean(0)
            return a / np.maximum(np.sqrt((a ** 2).mean(0)), 1e-12)
        Xz, Lz, yz = z(X), z(L), y / max(y.std(), 1e-12)
        r_xy = (Xz * yz[:, None]).mean(0)
        r_ly = (Lz * yz[:, None]).mean(0)
        r_xl = (Xz * Lz).mean(0)
        partial = (r_ly - r_xy * r_xl) / np.sqrt(np.maximum((1 - r_xy ** 2) * (1 - r_xl ** 2), 1e-12))
        order = np.argsort(-np.abs(partial), kind="stable")[:budget]
        hyps = [Hypothesis((inp.names[cols[j]],), "spatial_lag", float(abs(partial[j])),
                           {"bivariate_moran_r": float(r_ly[j]), "own_r": float(r_xy[j])}) for j in order]
        return hyps, {"n_features": int(len(cols)), "n_relationships_tested": int(len(cols))}


# ---------------------------------------------------------- residual generator
def reconstruct_budget(cands: list, budget: int) -> list:
    """Apply ``distil``'s own budget rule to a larger emission: recurrent joint
    (interaction / context-conditioned) entries keep a reserve of budget // 3,
    the rest by score, then non-recurrent fill-ins by score."""
    def kind(c):
        return c.metadata["provenance"]["hypothesis_kind"]

    rec = [c for c in cands if c.metadata["provenance"]["recurrent_in_generator"]]
    non = [c for c in cands if not c.metadata["provenance"]["recurrent_in_generator"]]
    key = lambda c: (-c.proposal_score, kind(c), c.variables)  # noqa: E731
    rec.sort(key=key)
    joint = [c for c in rec if kind(c) in ("interaction", "context_conditioned")]
    other = [c for c in rec if kind(c) not in ("interaction", "context_conditioned")]
    reserve = min(len(joint), max(1, budget // 3)) if joint else 0
    chosen = joint[:reserve] + other[:budget - reserve]
    if len(chosen) < budget:
        chosen += joint[reserve:reserve + budget - len(chosen)]
    chosen.sort(key=key)
    non.sort(key=lambda c: (-c.proposal_score, c.variables))
    return (chosen + non[:max(0, budget - len(chosen))])[:budget]


class ResidualGenerator(Method):
    """``generate_hypotheses`` with the runner's arguments for the family
    (graph branch and the molecular-x-context admissibility rule only for the
    context family; conventional explanation = the world's level columns)."""

    name = "residual_generator"

    def __init__(self, config_overrides: dict | None = None):
        self.overrides = dict(config_overrides or {})

    def propose(self, inp, budget):
        from morpheus_encoder.hypothesis_generator import GeneratorConfig, generate_hypotheses

        context = inp.family == "context"
        role = _role_of(inp)

        def admissible(cols):
            roles = {role.get(int(c)) for c in cols}
            return "variable" in roles and "context" in roles

        neighbours = [row[row >= 0] for row in inp.neighbours]
        cfg = GeneratorConfig(graph=context, **self.overrides)
        result = generate_hypotheses(
            inp.X, inp.y, inp.groups, conventional=inp.comparison, views=inp.views, roles=inp.roles,
            column_names=inp.names, neighbours=neighbours, coords=inp.coords, neighbour_of={},
            max_candidates=budget, seed=inp.seed, device=inp.device, config=cfg,
            admissible=admissible if context else None)
        cands = list(result.candidates)
        report = {k: result.report.get(k) for k in ("residual_variance_share", "n_recurring_structures",
                                                     "n_emitted", "emitted_kinds", "cost", "graph_branch")}
        tested = report.get("n_emitted") or len(cands)  # structures the generator scored and emitted
        return [candidate_to_hypothesis(c, inp) for c in cands], {"report": report, "_candidates": cands,
                                                                   "n_relationships_tested": int(tested)}


class ResidualBankGenerator(ResidualGenerator):
    """Hook for Agent G's v0.6 residual-bank generator (one head per ladder level):
    ``morpheus_encoder.residual_bank_generator.generate_residual_bank_hypotheses``,
    given the full B3 comparison of the proposal rows and its column names (the
    generator derives the N/B1/B2/B3 heads itself through Agent D's ladder)."""

    name = "residual_bank_generator"
    module = "morpheus_encoder.residual_bank_generator"

    def available(self):
        try:
            mod = importlib.import_module(self.module)
        except Exception as exc:
            return False, f"{self.module} not importable ({type(exc).__name__}; Agent G wave 4)"
        if not hasattr(mod, "generate_residual_bank_hypotheses"):
            return False, "generate_residual_bank_hypotheses not defined yet"
        return True, ""

    def propose(self, inp, budget):
        mod = importlib.import_module(self.module)
        result = mod.generate_residual_bank_hypotheses(
            inp.X, inp.y, inp.groups, comparison=inp.comparison_full, comparison_names=inp.comparison_names,
            views=inp.views, roles=inp.roles, column_names=inp.names, max_candidates=budget, seed=inp.seed,
            device=inp.device)
        rep = result.report or {}
        tested = rep.get("n_distinct") or len(result.candidates)
        return [candidate_to_hypothesis(c, inp) for c in result.candidates], {
            "report": rep, "n_relationships_tested": int(tested)}


# ===================================================== sprint baselines (Agent A)
# Shared pieces: a candidate pool of own-spot variables plus their within-patient
# 6-neighbour means (names ``nbr6:<column>``), a mapping from a variable set to a
# hypothesis, and one flexible scorer (patient-grouped CV R^2 of a random-Fourier-
# feature ridge with linear terms) used by every search that evaluates sets.

def _variable_columns(inp: MethodInput) -> np.ndarray:
    role = _role_of(inp)
    return np.asarray([c for c in _searchable(inp) if role[int(c)] == "variable"], dtype=int)


def _abs_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    return np.abs(Xc.T @ yc) / np.maximum(den, 1e-12)


def _nbr_means(X: np.ndarray, cols, neighbours: np.ndarray, chunk: int = 256) -> np.ndarray:
    """6-neighbour means of the given columns, computed in column chunks (bounded memory)."""
    cols = np.asarray(cols, dtype=int)
    out = np.empty((len(X), len(cols)))
    for s in range(0, len(cols), chunk):
        out[:, s:s + chunk] = neighbour_mean(np.asarray(X[:, cols[s:s + chunk]], dtype=float), neighbours)
    return out


def _nbr_name(name: str) -> str:
    return f"nbr6:{name}"


class Pool:
    """Searchable own columns plus 6-neighbour means of variable columns, with hypothesis-side names."""

    def __init__(self, inp: MethodInput, *, own_cols=None, nbr_cols=None):
        own = _searchable(inp) if own_cols is None else np.asarray(own_cols, dtype=int)
        nbr = _variable_columns(inp) if nbr_cols is None else np.asarray(nbr_cols, dtype=int)
        Xo = np.asarray(inp.X[:, own], dtype=float)
        Xn = _nbr_means(inp.X, nbr, inp.neighbours)
        self.Z = np.column_stack([Xo, Xn])
        self.names = [inp.names[c] for c in own] + [_nbr_name(inp.names[c]) for c in nbr]
        self.base = list(map(int, own)) + list(map(int, nbr))
        self.is_nbr = np.r_[np.zeros(len(own), bool), np.ones(len(nbr), bool)]


def _shortlist_pool(inp: MethodInput, n_own: int, n_nbr: int, extra: list | None = None) -> Pool:
    """Top ``n_own`` searchable columns and top ``n_nbr`` variable-column neighbour means by marginal |r|."""
    own = _searchable(inp)
    r_own = _abs_corr(np.asarray(inp.X[:, own], dtype=float), inp.y)
    keep_own = list(own[np.argsort(-r_own, kind="stable")[:n_own]])
    for c in extra or []:
        if c not in keep_own:
            keep_own.append(int(c))
    var = _variable_columns(inp)
    L = _nbr_means(inp.X, var, inp.neighbours)
    keep_nbr = var[np.argsort(-_abs_corr(L, inp.y), kind="stable")[:n_nbr]]
    return Pool(inp, own_cols=keep_own, nbr_cols=keep_nbr)


def set_hypothesis(names, inp: MethodInput, score: float, detail: dict | None = None) -> Hypothesis:
    """Variable set -> hypothesis.  A lone neighbour mean is the ``spatial_lag`` of its bare column;
    otherwise single / product (context if a context column is involved) / higher_order by size."""
    names = tuple(names)
    index = {n: i for i, n in enumerate(inp.names)}
    if len(names) == 1 and names[0].startswith("nbr6:") and names[0][5:] in index:
        return Hypothesis((names[0][5:],), "spatial_lag", float(score), detail or {})
    role = _role_of(inp)
    bases = [index.get(n[5:] if n.startswith("nbr6:") else n) for n in names]
    if len(names) == 1:
        kind = "single"
    elif any(b is not None and role.get(int(b)) == "context" for b in bases):
        kind = "context"
    else:
        kind = "product" if len(names) == 2 else "higher_order"
    return Hypothesis(names, kind, float(score), detail or {})


class FlexScorer:
    """Patient-grouped CV R^2 of a ridge on [linear terms | random Fourier features] of a column set.

    Rows: a fixed random subsample (``max_rows``) of the proposal rows; folds: patients.  Every call
    is counted in ``n_evaluated``."""

    def __init__(self, inp: MethodInput, *, max_rows: int = 6000, n_features: int = 48, n_folds: int = 3,
                 alpha: float = 1.0):
        rng = np.random.default_rng(inp.seed + 17)
        n = len(inp.y)
        self.rows = np.sort(rng.choice(n, size=min(max_rows, n), replace=False))
        self.y = np.asarray(inp.y, dtype=float)[self.rows]
        self.folds = grouped_fold_ids(np.asarray(inp.groups)[self.rows], n_folds, inp.seed)
        self.n_features, self.alpha, self.seed = n_features, alpha, inp.seed
        self.n_evaluated = 0
        self._omega = {}

    def _omega_for(self, k):
        if k not in self._omega:
            rng = np.random.default_rng(self.seed * 31 + k)
            self._omega[k] = (rng.standard_normal((k, self.n_features)), rng.uniform(0, 2 * np.pi, self.n_features))
        return self._omega[k]

    def score(self, Zs: np.ndarray) -> float:
        """``Zs``: (n_proposal, k) values of the set (all proposal rows; subsampled here)."""
        self.n_evaluated += 1
        Z = np.asarray(Zs, dtype=float)[self.rows]
        om, ph = self._omega_for(Z.shape[1])
        pred = np.empty(len(self.y))
        for f in np.unique(self.folds):
            te = self.folds == f
            mu, sd = Z[~te].mean(0), Z[~te].std(0)
            sd[sd < 1e-12] = 1.0
            A, Bt = (Z[~te] - mu) / sd, (Z[te] - mu) / sd
            Da = np.column_stack([A, np.sqrt(2.0 / self.n_features) * np.cos(A @ om + ph)])
            Db = np.column_stack([Bt, np.sqrt(2.0 / self.n_features) * np.cos(Bt @ om + ph)])
            dm = Da.mean(0)
            ym = self.y[~te].mean()
            Da, Db = Da - dm, Db - dm
            coef = np.linalg.solve(Da.T @ Da + self.alpha * np.eye(Da.shape[1]), Da.T @ (self.y[~te] - ym))
            pred[te] = Db @ coef + ym
        return float(1 - np.sum((self.y - pred) ** 2) / max(np.sum((self.y - self.y.mean()) ** 2), 1e-12))


def _rank_sets(scored: dict, pool: Pool, inp: MethodInput, budget: int, tag: str) -> list:
    """Rank evaluated sets (tuples of pool indices -> CV R^2) by incremental CV R^2: a single by its
    own score, a larger set by its score minus the best score among its evaluated proper subsets."""
    from itertools import combinations

    hyps = []
    for S, r2 in scored.items():
        best_sub = 0.0
        for s in range(1, len(S)):
            for T in combinations(S, s):
                if T in scored:
                    best_sub = max(best_sub, scored[T])
        inc = r2 - best_sub if len(S) > 1 else r2
        hyps.append(set_hypothesis([pool.names[i] for i in S], inp, inc, {"cv_r2": r2, "search": tag}))
    hyps.sort(key=lambda h: (-h.score, h.key))
    return hyps


# ------------------------------------------------------- 1 linear screen
class LinearInteractionScreen(Method):
    """Regularised linear screen: marginal |r| shortlist of 40 own columns -> main terms, their
    6-neighbour means, all pairwise products of the 40, and own x neighbour-mean products among
    the top 16 -> lasso path with patient-grouped CV (3 folds) -> terms ranked by |coefficient| at
    the CV-chosen penalty, ties (zero coefficients) broken by entry order along the path."""

    name = "linear_interaction_screen"

    def __init__(self, shortlist: int = 40, cross: int = 16, n_alphas: int = 30, max_rows: int = 9000):
        self.k, self.cross, self.n_alphas, self.max_rows = shortlist, cross, n_alphas, max_rows

    def propose(self, inp, budget):
        from sklearn.linear_model import lasso_path

        own = _searchable(inp)
        r = _abs_corr(np.asarray(inp.X[:, own], dtype=float), inp.y)
        top = own[np.argsort(-r, kind="stable")[:self.k]]
        role = _role_of(inp)
        Xo = np.asarray(inp.X[:, top], dtype=float)
        var_top = np.asarray([c for c in top if role[int(c)] == "variable"], dtype=int)
        Xn = _nbr_means(inp.X, var_top, inp.neighbours)
        names_o = [inp.names[c] for c in top]
        names_n = [_nbr_name(inp.names[c]) for c in var_top]
        terms, sets = [Xo, Xn], [(n,) for n in names_o] + [(n,) for n in names_n]
        pairs = []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                pairs.append(Xo[:, i] * Xo[:, j])
                sets.append((names_o[i], names_o[j]))
        for i in range(min(self.cross, len(top))):
            for j in range(min(self.cross, len(var_top))):
                pairs.append(Xo[:, i] * Xn[:, j])
                sets.append((names_o[i], names_n[j]))
        T = np.column_stack(terms + [np.column_stack(pairs)] if pairs else terms)
        T = T - T.mean(0)
        T /= np.maximum(T.std(0), 1e-12)
        y = inp.y - inp.y.mean()
        rng = np.random.default_rng(inp.seed)
        rows = np.sort(rng.choice(len(y), size=min(self.max_rows, len(y)), replace=False))
        T, y, groups = T[rows], y[rows], np.asarray(inp.groups)[rows]
        alpha_max = np.abs(T.T @ y).max() / len(y)
        alphas = alpha_max * np.logspace(0, -3, self.n_alphas)
        folds = grouped_fold_ids(groups, 3, inp.seed)
        mse = np.zeros(len(alphas))
        for f in np.unique(folds):
            te = folds == f
            _, coefs, _ = lasso_path(T[~te], y[~te] - y[~te].mean(), alphas=alphas, max_iter=2000, tol=1e-3)
            pred = T[te] @ coefs + y[~te].mean()
            mse += ((y[te][:, None] - pred) ** 2).sum(0)
        best = int(np.argmin(mse))
        _, coefs, _ = lasso_path(T, y, alphas=alphas, max_iter=2000, tol=1e-3)
        coef = coefs[:, best]
        nz = coefs != 0
        entry = np.where(nz.any(1), nz.argmax(1), len(alphas))
        order = sorted(range(T.shape[1]), key=lambda t: (-abs(coef[t]), entry[t], t))
        hyps = [set_hypothesis(sets[t], inp, float(abs(coef[t])), {"entry": int(entry[t])}) for t in order]
        return hyps, {"n_relationships_tested": int(T.shape[1]), "alpha_index": best,
                      "n_nonzero": int((coef != 0).sum()), "n_rows": int(len(rows))}


# ----------------------------------------------------------------- 4 RuleFit
class RuleFitSearch(Method):
    """RuleFit (imodels ``RuleFitRegressor``: depth-limited tree ensemble -> rules + linear terms ->
    lasso) on a pool of 40 own columns and 16 neighbour means shortlisted by marginal |r|; each
    surviving rule's variable set is scored by the sum of its rules' importances (|coef| x
    sqrt(support (1 - support)) as imodels reports it); linear terms give singles."""

    name = "rulefit"

    def __init__(self, n_own: int = 40, n_nbr: int = 16, max_rules: int = 200, max_rows: int = 6000):
        self.n_own, self.n_nbr, self.max_rules, self.max_rows = n_own, n_nbr, max_rules, max_rows

    def available(self):
        if importlib.util.find_spec("imodels") is None:
            return False, "imodels not installed"
        return True, ""

    def propose(self, inp, budget):
        import re

        from imodels import RuleFitRegressor

        pool = _shortlist_pool(inp, self.n_own, self.n_nbr)
        rng = np.random.default_rng(inp.seed)
        rows = np.sort(rng.choice(len(inp.y), size=min(self.max_rows, len(inp.y)), replace=False))
        fn = [f"f{i}" for i in range(pool.Z.shape[1])]
        model = RuleFitRegressor(n_estimators=100, tree_size=4, max_rules=self.max_rules, random_state=inp.seed)
        model.fit(pool.Z[rows], inp.y[rows], feature_names=fn)
        rules = model._get_rules(exclude_zero_coef=False) if hasattr(model, "_get_rules") else model.get_rules()
        scores: dict = {}
        for _, row in rules.iterrows():
            imp = float(row.get("importance", abs(row["coef"])) or 0.0)
            vs = tuple(sorted({int(m) for m in re.findall(r"\bf(\d+)\b", str(row["rule"]))}))
            if vs and imp > 0:
                scores[vs] = scores.get(vs, 0.0) + imp
        hyps = [set_hypothesis([pool.names[i] for i in S], inp, s, {"rulefit_importance": s})
                for S, s in scores.items()]
        hyps.sort(key=lambda h: (-h.score, h.key))
        return hyps, {"n_relationships_tested": int(len(rules)), "n_rules_nonzero": int((rules["coef"] != 0).sum()),
                      "pool": len(pool.names)}


# ------------------------------------------------- 5 neural interaction detection
class NIDInteraction(Method):
    """Neural interaction detection (Tsang, Cheng & Liu 2018).  An MLP (p -> 64 -> 32 -> 16 -> 1, ReLU)
    with an L1 penalty on the first layer is trained on every searchable own column plus the
    6-neighbour mean of every variable column (at least 2,000 Adam steps; early stopping every 100
    steps on one held-out patient fold).  Each first-layer
    unit h gets an aggregated influence z_h = (|w_out|^T |W_3| |W_2|)_h; for unit h the inputs
    sorted by |W_1[h, i]| give candidate sets {top-2, top-3, top-4} with strength z_h * min |W_1|
    over the set (summed over units); singles get sum_h z_h |W_1[h, i]|.  The ranked list
    interleaves singles, pairs and triples (then 4-sets), since raw strengths are not comparable
    across set sizes."""

    name = "nid_interaction"

    def __init__(self, epochs: int = 40, l1: float = 3e-3, hidden=(64, 32, 16), batch: int = 256,
                 min_steps: int = 2000, eval_every: int = 100):
        self.epochs, self.l1, self.hidden, self.batch = epochs, l1, hidden, batch
        self.min_steps, self.eval_every = min_steps, eval_every

    def available(self):
        if importlib.util.find_spec("torch") is None:
            return False, "torch not installed"
        return True, ""

    def fit_strengths(self, inp, pool: Pool):
        import torch

        torch.manual_seed(inp.seed)
        torch.set_num_threads(max(1, int(inp.n_threads)))
        dev = "cuda" if inp.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        Z = pool.Z
        mu, sd = Z.mean(0), Z.std(0)
        sd[sd < 1e-12] = 1.0
        X = torch.tensor((Z - mu) / sd, dtype=torch.float32, device=dev)
        y = torch.tensor((inp.y - inp.y.mean()) / max(inp.y.std(), 1e-12), dtype=torch.float32, device=dev)
        folds = grouped_fold_ids(np.asarray(inp.groups), 5, inp.seed)
        va = torch.tensor(folds == 0, device=dev)
        tr_idx = torch.nonzero(~va).squeeze(1)
        dims = (X.shape[1],) + tuple(self.hidden) + (1,)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        net = torch.nn.Sequential(*layers[:-1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, best_state, patience, step = np.inf, None, 0, 0
        g = torch.Generator(device="cpu").manual_seed(inp.seed)
        # train by steps, not epochs: small proposal sets would otherwise get a few hundred updates
        n_steps = max(self.min_steps, self.epochs * int(np.ceil(len(tr_idx) / self.batch)))
        while step < n_steps and patience < 6:
            net.train()
            perm = tr_idx[torch.randperm(len(tr_idx), generator=g).to(dev)]
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                loss = ((net(X[b]).squeeze(1) - y[b]) ** 2).mean() + self.l1 * net[0].weight.abs().sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                if step % self.eval_every == 0 or step >= n_steps:
                    net.eval()
                    with torch.no_grad():
                        v = float(((net(X[va]).squeeze(1) - y[va]) ** 2).mean())
                    net.train()
                    if v < best - 1e-5:
                        best, patience = v, 0
                        best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                    else:
                        patience += 1
                if step >= n_steps or patience >= 6:
                    break
        net.load_state_dict(best_state)
        Ws = [m.weight.detach().abs().cpu().numpy() for m in net if isinstance(m, torch.nn.Linear)]
        z = Ws[-1]
        for W in reversed(Ws[1:-1]):
            z = z @ W
        z = z.ravel()  # influence of each first-layer unit
        W1 = Ws[0]
        mains = (z[:, None] * W1).sum(0)
        inter: dict = {}
        top = np.argsort(-W1, axis=1)[:, :4]
        for h in range(W1.shape[0]):
            for s in (2, 3, 4):
                S = tuple(sorted(int(i) for i in top[h, :s]))
                inter[S] = inter.get(S, 0.0) + float(z[h] * W1[h, list(S)].min())
        return mains, inter, {"val_mse": float(best), "device": dev}

    def cached(self, inp):
        """(pool, mains, interactions, info) fitted once per method input (shared with the shortlist
        of greedy_flex_forward / pysr_shortlist when they run on the same world)."""
        cache = inp.__dict__.setdefault("_shared", {})
        key = ("nid", self.epochs, self.l1, self.hidden, self.batch, self.min_steps)
        if key not in cache:
            pool = cache.get("pool") or Pool(inp)
            cache["pool"] = pool
            cache[key] = (pool,) + tuple(self.fit_strengths(inp, pool))
        return cache[key]

    def propose(self, inp, budget):
        pool, mains, inter, info = self.cached(inp)
        by_size = {1: sorted(((float(mains[i]), (i,)) for i in range(len(mains))), reverse=True)[:max(budget, 64)]}
        for S, v in inter.items():
            by_size.setdefault(len(S), []).append((v, S))
        for k in by_size:
            by_size[k].sort(key=lambda t: (-t[0], t[1]))
        hyps, i = [], 0
        while len(hyps) < max(budget, 128) and any(i < len(by_size.get(k, [])) for k in (1, 2, 3, 4)):
            for k in (1, 2, 3):
                if i < len(by_size.get(k, [])):
                    v, S = by_size[k][i]
                    hyps.append(set_hypothesis([pool.names[j] for j in S], inp, v, {"nid_strength": v}))
            if i < len(by_size.get(4, [])) and i % 2 == 1:
                v, S = by_size[4][i // 2]
                hyps.append(set_hypothesis([pool.names[j] for j in S], inp, v, {"nid_strength": v}))
            i += 1
        n_tested = len(mains) + len(inter)
        return hyps, {"n_relationships_tested": int(n_tested), "pool": len(pool.names), **info}


# ------------------------------------------------- 6 greedy flexible forward search
class GreedyFlexForward(Method):
    """Forward selection over a 30-variable shortlist (20 by marginal |r| over own columns and
    neighbour means, 10 more from the NID single-variable strengths), sets of up to 4 variables,
    each scored by patient-grouped CV R^2 of a random-Fourier-feature ridge (``FlexScorer``).
    Level 1 evaluates every shortlisted variable; each later level extends the best ``beam`` sets
    of the previous level by every shortlisted variable.  Every evaluated set counts; the list is
    ranked by incremental CV R^2 (``_rank_sets``)."""

    name = "greedy_flex_forward"

    def __init__(self, n_marginal: int = 20, n_nid: int = 10, beam: int = 6, max_order: int = 4):
        self.n_marginal, self.n_nid, self.beam, self.max_order = n_marginal, n_nid, beam, max_order

    def shortlist(self, inp) -> tuple[Pool, list]:
        full, mains, _, _ = NIDInteraction().cached(inp)
        r = _abs_corr(full.Z, inp.y)
        chosen = list(np.argsort(-r, kind="stable")[:self.n_marginal])
        for i in np.argsort(-mains, kind="stable"):
            if len(chosen) >= self.n_marginal + self.n_nid:
                break
            if i not in chosen:
                chosen.append(int(i))
        return full, chosen

    def propose(self, inp, budget):
        full, chosen = self.shortlist(inp)
        scorer = FlexScorer(inp)
        scored: dict = {}
        level = []
        for i in chosen:
            scored[(i,)] = scorer.score(full.Z[:, [i]])
            level.append((i,))
        for _ in range(2, self.max_order + 1):
            frontier = sorted(level, key=lambda S: -scored[S])[:self.beam]
            level = []
            for S in frontier:
                for i in chosen:
                    T = tuple(sorted(set(S) | {i}))
                    if len(T) == len(S) + 1 and T not in scored:
                        scored[T] = scorer.score(full.Z[:, list(T)])
                        level.append(T)
        hyps = _rank_sets(scored, full, inp, budget, "greedy_forward")
        return hyps, {"n_relationships_tested": scorer.n_evaluated, "shortlist": [full.names[i] for i in chosen]}


# --------------------------------------------------------- 6' PySR (if Julia works)
class PySRShortlist(Method):
    """Symbolic regression (PySR) on a 30-variable shortlist (20 by marginal |r| over own columns and
    neighbour means, 10 from NID single-variable strengths); the variable sets of the hall-of-fame
    equations, ranked by PySR's score (loss improvement per unit complexity), then the remaining
    shortlisted singles by marginal |r|.  ``n_relationships_tested`` = expressions evaluated
    (niterations x populations x population_size x ncycles_per_iteration, PySR's search volume)."""

    name = "pysr_shortlist"
    budget_dependent = False

    def __init__(self, niterations: int = 30, populations: int = 12, population_size: int = 33,
                 timeout: int = 240, max_rows: int = 3000):
        self.kw = dict(niterations=niterations, populations=populations, population_size=population_size)
        self.timeout, self.max_rows = timeout, max_rows

    def available(self):
        if importlib.util.find_spec("pysr") is None:
            return False, "pysr not installed in morpheus_venv"
        return True, ""

    def propose(self, inp, budget):
        import re

        from pysr import PySRRegressor

        full, chosen = GreedyFlexForward().shortlist(inp)
        rng = np.random.default_rng(inp.seed)
        rows = np.sort(rng.choice(len(inp.y), size=min(self.max_rows, len(inp.y)), replace=False))
        Z = full.Z[:, chosen][rows]
        Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
        model = PySRRegressor(binary_operators=["+", "-", "*"], unary_operators=["tanh", "square"],
                              maxsize=20, timeout_in_seconds=self.timeout, random_state=inp.seed,
                              deterministic=True, parallelism="serial", verbosity=0, progress=False,
                              temp_equation_file=True, **self.kw)
        model.fit(Z, inp.y[rows])
        eq = model.equations_
        scores: dict = {}
        for _, row in eq.iterrows():
            vs = tuple(sorted({int(m) for m in re.findall(r"\bx(\d+)\b", str(row["equation"]))}))
            if vs:
                scores[vs] = max(scores.get(vs, -np.inf), float(row["score"]))
        hyps = [set_hypothesis([full.names[chosen[i]] for i in S], inp, s, {"pysr_score": s})
                for S, s in scores.items()]
        hyps.sort(key=lambda h: (-h.score, h.key))
        r = _abs_corr(full.Z[:, chosen], inp.y)
        hyps += [set_hypothesis([full.names[chosen[i]]], inp, -1.0 + float(r[i]))
                 for i in np.argsort(-r, kind="stable")]
        volume = int(np.prod(list(self.kw.values())) * 550)
        return hyps, {"n_relationships_tested": volume, "n_equations": int(len(eq)),
                      "shortlist": [full.names[i] for i in chosen]}


# ------------------------------------------------- 8 random and heuristic search
class RandomSearchMatched(Method):
    """Random variable sets (size 1-4; probabilities 0.25 / 0.4 / 0.25 / 0.1) drawn from every
    searchable own column and every variable column's neighbour mean, ``n_sets`` of them, each
    scored by ``FlexScorer``; ranked by CV R^2.  The search floor at a matched number of sets."""

    name = "random_search_matched"

    def __init__(self, n_sets: int = 256):
        self.n_sets = n_sets

    def propose(self, inp, budget):
        pool = Pool(inp)
        rng = np.random.default_rng(inp.seed + 101)
        scorer = FlexScorer(inp)
        scored: dict = {}
        while len(scored) < self.n_sets:
            k = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.4, 0.25, 0.1]))
            S = tuple(sorted(int(i) for i in rng.choice(len(pool.names), k, replace=False)))
            if S not in scored:
                scored[S] = scorer.score(pool.Z[:, list(S)])
        hyps = [set_hypothesis([pool.names[i] for i in S], inp, r2, {"cv_r2": r2}) for S, r2 in scored.items()]
        hyps.sort(key=lambda h: (-h.score, h.key))
        return hyps, {"n_relationships_tested": scorer.n_evaluated}


class HeuristicSearch(Method):
    """Marginal-ranked search: own columns and neighbour means ranked by marginal |r|; the top 16
    singles, then every pair of the top 16, then triples of the top 8 (stopping at ``max_sets``),
    each scored by ``FlexScorer``; ranked by incremental CV R^2."""

    name = "heuristic_search"

    def __init__(self, max_sets: int = 256, n_single: int = 16, n_triple: int = 8):
        self.max_sets, self.n_single, self.n_triple = max_sets, n_single, n_triple

    def propose(self, inp, budget):
        from itertools import combinations

        pool = Pool(inp)
        order = list(np.argsort(-_abs_corr(pool.Z, inp.y), kind="stable"))
        scorer = FlexScorer(inp)
        scored: dict = {}
        plan = [(i,) for i in order[:self.n_single]]
        plan += [tuple(sorted(p)) for p in combinations(order[:self.n_single], 2)]
        plan += [tuple(sorted(t)) for t in combinations(order[:self.n_triple], 3)]
        for S in plan[:self.max_sets]:
            scored[tuple(int(i) for i in S)] = scorer.score(pool.Z[:, list(S)])
        hyps = _rank_sets(scored, pool, inp, budget, "heuristic")
        return hyps, {"n_relationships_tested": scorer.n_evaluated}


# ---------------------------------------------------------------------- oracle
class Oracle(Method):
    """Handed the truth: rank 1 is the planted hypothesis (tests / reference)."""

    name = "oracle"

    def __init__(self, truth: Truth):
        self.truth = truth

    def propose(self, inp, budget):
        filler = [Hypothesis((inp.names[int(c)],), "single", 0.0) for c in _searchable(inp)[:budget]]
        return [Hypothesis(self.truth.variables, self.truth.hypothesis_class, 1.0)] + filler, {
            "n_relationships_tested": 1}


METHODS: dict[str, Callable[[], Method]] = {
    "enumeration": Enumeration, "fm_enumeration": FMEnumeration, "gbm_interaction": GBMInteraction,
    "ebm_ga2m": EBMSearch, "pysr": PySRSearch, "spatial_lag_screen": SpatialLagScreen,
    "residual_generator": ResidualGenerator, "residual_bank_generator": ResidualBankGenerator,
    "linear_interaction_screen": LinearInteractionScreen, "rulefit": RuleFitSearch,
    "nid_interaction": NIDInteraction, "pysr_shortlist": PySRShortlist, "greedy_flex_forward": GreedyFlexForward,
    "random_search_matched": RandomSearchMatched, "heuristic_search": HeuristicSearch,
}


def register(name: str, factory: Callable[[], Method]) -> None:
    """Add a method under ``name`` (used by the learned-search wrappers below)."""
    METHODS[name] = factory


# --- learned search wrappers (Agent C) ---
#
# ``learned_search`` / ``learned_search_rowsplit`` / ``learned_search_pertask``: the
# learned scientific-search model (``morpheus_encoder.learned_search``) through
# ``learned_search.propose.propose`` on the world's proposal rows and family views.
# Checkpoints: ``MORPHEUS_LEARNED_CHECKPOINT`` (meta-trained) and
# ``MORPHEUS_LEARNED_CHECKPOINT_ROWSPLIT`` (row-split ablation), or the constructor
# argument.  Without a checkpoint the method reports itself unavailable, unless
# ``MORPHEUS_LEARNED_STUB=1`` (wiring tests: the marginal-screen stub, flagged in info).
# Prefixed set names map to the shared hypothesis names: ``own:<col>`` -> ``<col>``,
# ``nbr6|nbr18|contrast:<col>`` -> the same string (``substrate.split_variable`` /
# ``sub.transform`` resolve it); a ``concept:<j>`` variable has no benchmark column, so
# such a set is skipped and counted in ``info["skipped_concept_sets"]``.

import os as _os


def learned_name_to_hypothesis_name(name: str) -> str | None:
    """``own:raw_genes:g1`` -> ``raw_genes:g1``; ``nbr6:raw_genes:g1`` unchanged; concept -> None."""
    from morpheus_encoder.learned_search.propose import parse_name
    transform, base = parse_name(name)
    if base.startswith("concept:"):
        return None
    return base if transform == "own" else f"{transform}:{base}"


def learned_candidate_to_hypothesis(c, inp: MethodInput) -> Hypothesis | None:
    """A ``learned_set`` candidate -> the shared ``Hypothesis`` (None for a concept set)."""
    prov = (c.metadata or {}).get("provenance") or {}
    names = [learned_name_to_hypothesis_name(t["name"]) for t in prov.get("transforms") or ()]
    if not names or any(n is None for n in names):
        return None
    detail = {"proposal_family": c.proposal_family, "hypothesis_kind": prov.get("hypothesis_kind"),
              "g_ho_support": prov.get("g_ho_support"), "path_rank": prov.get("path_rank")}
    h = set_hypothesis(names, inp, float(c.proposal_score), detail)
    if h.relation == "product" and any(n.split(":", 1)[0] in ("nbr18", "contrast") for n in names):
        # set_hypothesis looks for context roles through nbr6 names only; a nbr18 / contrast
        # transform of a context column is still a context hypothesis
        role = _role_of(inp)
        index = {n: i for i, n in enumerate(inp.names)}
        bases = [index.get(n.split(":", 1)[1]) if n.split(":", 1)[0] in ("nbr6", "nbr18", "contrast")
                 and n.split(":", 1)[1] in index else index.get(n) for n in names]
        if any(b is not None and role.get(int(b)) == "context" for b in bases):
            h = Hypothesis(h.variables, "context", h.score, h.detail)
    return h


def _learned_search_hypotheses(model, inp: MethodInput, budget: int, *, stub: bool,
                                strategy: str = "pointer") -> tuple[list, dict]:
    from morpheus_encoder.learned_search.propose import propose
    cols = _searchable(inp)
    local = {int(c): j for j, c in enumerate(cols)}
    views = {v: [local[int(c)] for c in vc] for v, vc in inp.views.items()}
    out = propose(model, np.asarray(inp.X[:, cols], dtype=float), inp.y, inp.groups, inp.neighbours, inp.coords,
                  [inp.names[int(c)] for c in cols], views, budget=max(int(budget), 1), seed=int(inp.seed),
                  device=inp.device, strategy=strategy)
    hyps, skipped = [], 0
    for c in out.candidates:
        h = learned_candidate_to_hypothesis(c, inp)
        if h is None:
            skipped += 1
            continue
        hyps.append(h)
    return hyps, {"n_relationships_tested": int(out.n_relationships_tested), "stub": bool(stub),
                  "strategy": strategy, "skipped_concept_sets": skipped, "report": out.report}


class LearnedSearch(Method):
    """Meta-trained learned search (frozen checkpoint).  One ranked list, cut by the harness."""
    name = "learned_search"
    budget_dependent = False
    env_key = "MORPHEUS_LEARNED_CHECKPOINT"
    strategy = "pointer"

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint or _os.environ.get(self.env_key) or None
        self._model = None

    def _stub(self) -> bool:
        return self.checkpoint is None and _os.environ.get("MORPHEUS_LEARNED_STUB") == "1"

    def available(self):
        if self.checkpoint is None and not self._stub():
            return False, f"no checkpoint (constructor argument or {self.env_key})"
        if self.checkpoint is not None and not _os.path.exists(self.checkpoint):
            return False, f"checkpoint {self.checkpoint} not found"
        return True, ""

    def model(self, inp: MethodInput):
        if self.checkpoint is None:
            return None
        if self._model is None:
            from morpheus_encoder.learned_search.propose import load_model
            self._model = load_model(self.checkpoint, device=inp.device)
        return self._model

    def propose(self, inp, budget):
        hyps, info = _learned_search_hypotheses(self.model(inp), inp, budget, stub=self.checkpoint is None,
                                                strategy=self.strategy)
        return hyps, {**info, "checkpoint": self.checkpoint}


class LearnedSearchPointerExpansion(LearnedSearch):
    """Locked pointer-preserving conditional partner expansion."""
    name = "learned_search_pointer_expansion"
    strategy = "pointer_expand"



class LearnedSearchMultiRouteExpansion(LearnedSearch):
    """Pointer-preserving expansion with a support-only pair-sketch anchor route."""
    name = "learned_search_multiroute_expansion"
    strategy = "multiroute_expand"


class LearnedSearchMultiRouteV2(LearnedSearch):
    """Pointer-preserving product, context-gate, and nonlinear proposal routes."""
    name = "learned_search_multiroute_v2"
    strategy = "multiroute_v2"


class LearnedSearchMultiRouteHybridV11(LearnedSearch):
    """v1 multiroute backbone with a bounded direct context-gate reserve."""
    name = "learned_search_multiroute_hybrid_v11"
    strategy = "multiroute_hybrid_v11"

class LearnedSearchRowSplit(LearnedSearch):
    """The same architecture trained on row-split episodes (support and query from the same patients)."""
    name = "learned_search_rowsplit"
    env_key = "MORPHEUS_LEARNED_CHECKPOINT_ROWSPLIT"


class LearnedSearchPerTask(LearnedSearch):
    """The same architecture trained from scratch on this world's proposal rows only
    (``learned_search.ablation.train_per_task``, 600 steps), then asked for sets."""
    name = "learned_search_pertask"
    steps = 600

    def __init__(self, checkpoint: str | None = None, steps: int = 600):
        super().__init__(checkpoint)
        self.steps = int(steps)

    def available(self):
        try:
            from morpheus_encoder.learned_search.ablation import train_per_task  # noqa: F401
        except ImportError as exc:
            if _os.environ.get("MORPHEUS_LEARNED_STUB") == "1":
                return True, ""
            return False, f"learned_search.ablation.train_per_task not importable: {exc}"
        return True, ""

    def propose(self, inp, budget):
        try:
            from morpheus_encoder.learned_search.ablation import train_per_task
        except ImportError:
            hyps, info = _learned_search_hypotheses(None, inp, budget, stub=True)
            return hyps, {**info, "steps": 0}
        cols = _searchable(inp)
        local = {int(c): j for j, c in enumerate(cols)}
        views = {v: [local[int(c)] for c in vc] for v, vc in inp.views.items()}
        model, _, train_info = train_per_task(np.asarray(inp.X[:, cols], dtype=float), inp.y, inp.groups,
                                              inp.neighbours, names=[inp.names[int(c)] for c in cols], views=views,
                                              steps=self.steps, seed=int(inp.seed), device=inp.device)
        hyps, info = _learned_search_hypotheses(model, inp, budget, stub=False)
        return hyps, {**info, "steps": self.steps, "train": train_info}


register("learned_search", LearnedSearch)
register("learned_search_pointer_expansion", LearnedSearchPointerExpansion)
register("learned_search_multiroute_expansion", LearnedSearchMultiRouteExpansion)
register("learned_search_multiroute_v2", LearnedSearchMultiRouteV2)
register("learned_search_multiroute_hybrid_v11", LearnedSearchMultiRouteHybridV11)
register("learned_search_rowsplit", LearnedSearchRowSplit)
register("learned_search_pertask", LearnedSearchPerTask)


# --- summary-statistics baselines (learned search v02 spec, baselines 1-4) ---
#
# The same candidate pool, support statistics, beam and closed-form refit as the v02 network
# (``learned_search.search_v02.search_by_scores``) at the same budget B (refits); only the
# per-candidate score differs (``summary_baselines`` scorers).  Fitted models:
# ``MORPHEUS_STATS_BASELINES`` (default runs/development/learned_search_v02/stats_baselines).
REFIT_BUDGET = 64


class _ScoreSearchModel:
    """``propose_sets`` adapter so ``propose.propose`` (and its name -> hypothesis mapping) runs the
    shared v02 beam + refit with an external scorer."""

    def __init__(self, scorer, budget: int):
        self.scorer, self.budget = scorer, int(budget)

    def propose_sets(self, X, y, groups, neighbours6, *, names, views, embed=None, n_paths=None, seed=0,
                     device=None):
        from morpheus_encoder.learned_search.search_v02 import search_by_scores
        return search_by_scores(X, y, groups, neighbours6, names=names, views=views, embed=embed,
                                rank_by=self.scorer, budget=self.budget, seed=seed, device=device)


class SummaryBaseline(Method):
    name = "summary_baseline"
    budget_dependent = False
    scorer_name = "support_screen"

    def __init__(self, model_dir: str | None = None, refits: int = REFIT_BUDGET):
        self.model_dir = model_dir or _os.environ.get("MORPHEUS_STATS_BASELINES",
                                                      "runs/development/learned_search_v02/stats_baselines")
        self.refits = int(refits)

    def available(self):
        if importlib.util.find_spec("morpheus_encoder.learned_search.search_v02") is None:
            return False, "learned_search.search_v02 not importable"
        if self.scorer_name != "support_screen" and not _os.path.exists(_os.path.join(self.model_dir,
                                                                                      "PROVENANCE.json")):
            return False, f"no fitted statistics baselines under {self.model_dir}"
        return True, ""

    def propose(self, inp, budget):
        from .summary_baselines import load_scorer
        model = _ScoreSearchModel(load_scorer(self.scorer_name, self.model_dir), self.refits)
        hyps, info = _learned_search_hypotheses(model, inp, self.refits, stub=False)
        return hyps, {**info, "n_relationships_tested": self.refits, "scorer": self.scorer_name,
                      "model_dir": self.model_dir}


class StatsGBM(SummaryBaseline):
    name = scorer_name = "stats_gbm"


class StatsRidge(SummaryBaseline):
    name = scorer_name = "stats_ridge"


class SupportLopoHeuristic(SummaryBaseline):
    name = scorer_name = "support_lopo_heuristic"


class SupportScreen(SummaryBaseline):
    name = scorer_name = "support_screen"


for _cls in (StatsGBM, StatsRidge, SupportLopoHeuristic, SupportScreen):
    register(_cls.name, _cls)


# --- v02 learned search (frozen GainSearchModel checkpoints; ``propose.load_model`` dispatches) ---
class LearnedSearchV02(LearnedSearch):
    """v02 network: predicted held-out gain -> the shared beam + refit at B = 64 (its default)."""
    name = "learned_search_v02"
    env_key = "MORPHEUS_LEARNED_CHECKPOINT"

    def _stub(self) -> bool:
        return False


class LearnedSearchV02RowSplit(LearnedSearchV02):
    """The same network trained on row-split episodes (support and query share patients)."""
    name = "learned_search_v02_rowsplit"
    env_key = "MORPHEUS_LEARNED_CHECKPOINT_ROWSPLIT"


register("learned_search_v02", LearnedSearchV02)
register("learned_search_v02_rowsplit", LearnedSearchV02RowSplit)


# --- RuleFit (scikit-learn) and TabPFN-v2 + SPEX on a MethodInput (historical rediscovery, transfer) ---
class _HarnessScored(Method):
    """A ``dbench.harness.core8`` scorer applied to proposal rows: the response is used as given (it is
    already the task's residual), candidates are the searchable columns, and every set the method scores
    is returned in its order with its score."""
    budget_dependent = False
    factory = None
    device = None

    def propose(self, inp, budget):
        from types import SimpleNamespace
        cols = _searchable(inp)
        task = SimpleNamespace(X=np.asarray(inp.X[:, cols], dtype=np.float64), groups=np.asarray(inp.groups),
                               y_resid=np.asarray(inp.y, dtype=float))
        hyps = []
        for idx, score, _ in type(self).factory(self.device)(task, "nuis", None):
            names = [inp.names[int(cols[i])] for i in idx]
            hyps.append(set_hypothesis(names, inp, float(score if score is not None else 0.0)))
        return hyps, {"n_relationships_tested": len(hyps)}


def _rulefit_factory(device):
    from dbench.harness.core8 import rulefit_sklearn
    return rulefit_sklearn(device=device)


def _tabpfn_factory(device):
    from dbench.harness.core8 import tabdistill
    return tabdistill(device=device)


class RuleFitSklearn(_HarnessScored):
    """RuleFit (Friedman and Popescu 2008), scikit-learn reimplementation (imodels 3.0.4 segfaults in its
    Lasso alpha search), on the shared interaction-aware shortlist."""
    name = "rulefit_sklearn"
    factory = staticmethod(_rulefit_factory)


class TabPFNSPEX(_HarnessScored):
    """TabPFN-v2 regressor fitted on the shortlist + ProxySPEX interaction attribution (order <= 4)."""
    name = "tabpfn_spex"
    factory = staticmethod(_tabpfn_factory)
    device = _os.environ.get("MORPHEUS_TABPFN_DEVICE") or None


register("rulefit_sklearn", RuleFitSklearn)
register("tabpfn_spex", TabPFNSPEX)


class MISTyRF(Method):
    """MISTy (Tanevski et al. 2022, Genome Biology), scikit-learn reimplementation: random forests on the intra
    (own-spot), juxta (6-neighbour mean) and para (18-neighbour mean) views; variables ranked by impurity
    importance weighted by the view's out-of-bag R^2.  Proposes single variables (MISTy has no set output)."""
    name = "misty_rf"
    budget_dependent = False

    def propose(self, inp, budget):
        from dbench.harness.core8 import misty_importance
        cols = _searchable(inp)
        names = [inp.names[int(c)] for c in cols]
        imp = misty_importance(np.asarray(inp.X[:, cols], dtype=np.float64), inp.y, names, seed=int(inp.seed))
        order = np.argsort(-np.where(np.isfinite(imp), imp, -1e300), kind="stable")
        hyps = [set_hypothesis([names[j]], inp, float(imp[j])) for j in order if np.isfinite(imp[j])]
        return hyps, {"n_relationships_tested": int(np.isfinite(imp).sum())}


register("misty_rf", MISTyRF)
