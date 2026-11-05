"""Bridge from a learned-search model to frozen discovery candidates.

``propose`` runs on PROPOSAL rows only.  The model
(``LearnedSearchModel.propose_sets``, Agent B) returns ranked unique variable
sets named with a transform prefix:

* ``own:<column name>``      the design column itself;
* ``nbr6:<column name>``     within-patient mean over the (up to) 6 spatial neighbours;
* ``nbr18:<column name>``    within-patient mean over the 18 nearest spots by coordinates
                             (``dbench.benchmark.substrate.knn_within_groups``, the
                             benchmark's and the v0.5 competitor's rule); without
                             coordinates, the two-hop ring of the 6-neighbour graph;
* ``contrast:<column name>`` own - nbr6;

where ``<column name>`` is an entry of ``names`` (one per column of ``X``) or
``concept:<j>`` (column ``j`` of the model's concept layer applied to the
frozen H-Optimus-0 embedding ``embed``).  Every non-``own`` variable becomes an
extra design column (``extra_columns``, deduplicated), and a candidate's
``variables`` index the concatenation ``[X | extra_columns]``.  ``extra_specs``
records how each extra column was built, so a caller can rebuild the same
columns on other rows (``transform_columns``: the held-out rows of the same
episode; neighbours never cross a patient, so proposal rows are unaffected).

With ``model=None`` a marginal screen stands in for the model (singles and
pairs ranked by patient-grouped cross-validated R^2 of a linear fit, candidates
from the top variables by within-patient |r|) and ``report["stub"]`` is True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from morpheus_encoder.general_discovery import Candidate

TRANSFORMS: tuple[str, ...] = ("own", "nbr6", "nbr18", "contrast")


@dataclass
class ProposeResult:
    candidates: list            # ranked Candidate objects; variables index [X | extra_columns]
    extra_columns: np.ndarray   # (n, m) transformed variables not already in X (proposal rows)
    extra_names: list           # m prefixed names, e.g. "nbr6:programmes:hallmark_EMT"
    n_relationships_tested: int  # variable sets the proposer scored
    report: dict
    extra_specs: list = field(default_factory=list)  # [{"name", "transform", "base", "column"|"concept"}]


# ----------------------------------------------------------------- transforms
def pad_neighbours(neighbours: Any, n: int) -> np.ndarray | None:
    """(n, k) int array, -1 padded, from a padded array or a list of index arrays."""
    if neighbours is None:
        return None
    if isinstance(neighbours, np.ndarray) and neighbours.ndim == 2:
        out = neighbours.astype(np.int64)
    else:
        rows = list(neighbours)
        k = max((len(v) for v in rows), default=0)
        out = -np.ones((len(rows), max(k, 1)), dtype=np.int64)
        for i, idx in enumerate(rows):
            out[i, :len(idx)] = np.asarray(idx, dtype=np.int64)
    if len(out) != n:
        raise ValueError(f"neighbours has {len(out)} rows, X has {n}")
    return out


def ring2(neighbours6: np.ndarray) -> np.ndarray:
    """Two-hop neighbourhood (neighbours and their neighbours, never the row itself), -1 padded."""
    n = len(neighbours6)
    sets = []
    for i in range(n):
        first = neighbours6[i][neighbours6[i] >= 0]
        second = neighbours6[first].ravel() if len(first) else np.zeros(0, dtype=np.int64)
        both = np.unique(np.concatenate((first, second[second >= 0])))
        sets.append(both[both != i])
    return pad_neighbours(sets, n)


def parse_name(name: str) -> tuple[str, str]:
    """``"nbr6:raw_genes:COL1A1"`` -> ``("nbr6", "raw_genes:COL1A1")``; an unprefixed name is ``own``."""
    head, _, rest = str(name).partition(":")
    if head in TRANSFORMS and rest:
        return head, rest
    return "own", str(name)


def concept_values(model: Any, embed: Any) -> np.ndarray:
    """The model's concept layer on the frozen embedding, (n, n_concepts) numpy."""
    if model is None or embed is None:
        raise ValueError("concept variables need both a model (its concept layer) and embed")
    for attr in ("concept_values", "concepts"):
        fn = getattr(model, attr, None)
        if callable(fn):
            return np.asarray(fn(np.asarray(embed, dtype=np.float32)), dtype=float)
    raise AttributeError("the model exposes neither concept_values(embed) nor concepts(embed)")


def morph_values(embed: Any, model: Any = None) -> np.ndarray:
    """(n, 64) cosine similarities of the frozen embedding to the frozen morphology dictionary,
    column = centre index.  v03 models (``kind == "gain_v03"``): the v03 dictionary with the
    standardisation the model computed on its proposal (support) rows (``model.morph_stats``), so an
    all-row rebuild equals the proposal-row build; centres dropped by the three-study rule stay 0.
    Otherwise the v02 dictionary."""
    if embed is None:
        raise ValueError("morph variables need embed")
    E = np.asarray(embed, dtype=np.float32)
    if getattr(model, "kind", "") == "gain_v03":
        from .morphology_dictionary import load_dictionary_v03, similarities_v03
        d = load_dictionary_v03()
        mu, sd = model.morph_stats
        out = np.zeros((len(E), 64))
        out[:, d["kept"]] = similarities_v03(E, d["centres"], mu, sd)
        return out
    from .morphology_dictionary import morph_view
    return np.asarray(morph_view(E)[0], dtype=float)


def _base_values(spec: Mapping[str, Any], X: np.ndarray, concepts: np.ndarray | None,
                 morph: np.ndarray | None = None) -> np.ndarray:
    if "morph" in spec:
        return morph[:, int(spec["morph"])]
    if "concept" in spec:
        if concepts is None:
            raise ValueError(f"{spec['name']}: concept values unavailable")
        return concepts[:, int(spec["concept"])]
    return np.asarray(X[:, int(spec["column"])], dtype=float)


def neighbours18(neighbours6: np.ndarray | None, coords: Any | None, groups: Any | None) -> np.ndarray:
    """18 nearest same-patient spots by coordinates when both are given, else the two-hop ring."""
    if coords is not None and groups is not None:
        from dbench.benchmark.substrate import knn_within_groups
        return knn_within_groups(np.asarray(coords, dtype=float), np.asarray(groups), 18)
    if neighbours6 is None:
        raise ValueError("nbr18 needs coordinates and groups, or neighbours6")
    return ring2(neighbours6)


def transform_columns(specs: Sequence[Mapping[str, Any]], X: Any, neighbours6: Any, *,
                      embed: Any | None = None, model: Any | None = None, coords: Any | None = None,
                      groups: Any | None = None) -> np.ndarray:
    """Build the columns named by ``specs`` on any rows (neighbour means via
    ``dbench.benchmark.substrate.neighbour_mean``; within-patient neighbours)."""
    from dbench.benchmark.substrate import neighbour_mean
    X = np.asarray(X, dtype=float)
    n = len(X)
    if not specs:
        return np.zeros((n, 0))
    nb6 = pad_neighbours(neighbours6, n)
    nb18 = None
    concepts = concept_values(model, embed) if any("concept" in s for s in specs) else None
    morph = morph_values(embed, model) if any("morph" in s for s in specs) else None
    out = np.empty((n, len(specs)))
    for j, spec in enumerate(specs):
        base = _base_values(spec, X, concepts, morph)
        t = spec["transform"]
        if t == "own":
            out[:, j] = base
            continue
        if t == "nbr18":
            if nb18 is None:
                nb18 = neighbours18(nb6, coords, groups)
            out[:, j] = neighbour_mean(base, nb18)
            continue
        if nb6 is None:
            raise ValueError(f"{spec['name']}: a neighbour transform needs neighbours6")
        if t == "nbr6":
            out[:, j] = neighbour_mean(base, nb6)
        elif t == "contrast":
            out[:, j] = base - neighbour_mean(base, nb6)
        else:
            raise ValueError(f"unknown transform {t!r}")
    return out


class _Resolver:
    """Prefixed names -> indices into [X | extra columns], accumulating extra columns."""

    def __init__(self, X: np.ndarray, names: Sequence[str]):
        self.X, self.names = X, [str(c) for c in names]
        self.index = {c: i for i, c in enumerate(self.names)}
        self.specs: list[dict] = []
        self.extra_index: dict[str, int] = {}

    def spec(self, name: str) -> dict:
        transform, base = parse_name(name)
        head, _, rest = base.partition(":")
        if base in self.index:
            return {"name": f"{transform}:{base}", "transform": transform, "base": base,
                    "column": self.index[base]}
        if head == "concept" and rest.isdigit():
            return {"name": f"{transform}:{base}", "transform": transform, "base": base, "concept": int(rest)}
        if head == "morph" and rest.startswith("k") and rest[1:].isdigit():
            return {"name": f"{transform}:{base}", "transform": transform, "base": base, "morph": int(rest[1:])}
        raise KeyError(f"{name!r}: base {base!r} is neither a column name nor concept:<j>")

    def resolve(self, name: str) -> tuple[int, dict]:
        spec = self.spec(name)
        if spec["transform"] == "own" and "column" in spec:
            return int(spec["column"]), spec
        key = spec["name"]
        if key not in self.extra_index:
            self.extra_index[key] = self.X.shape[1] + len(self.specs)
            self.specs.append(spec)
        return self.extra_index[key], spec


def load_model(checkpoint: str | None, *, device: Any = None) -> Any:
    """``LearnedSearchModel.load(checkpoint, map_location=device)`` (Agent B's model module); ``None``
    for no checkpoint (the caller then runs the marginal-screen stub)."""
    if not checkpoint:
        return None
    from morpheus_encoder.learned_search import model as model_module
    cls = getattr(model_module, "LearnedSearchModel", None)
    if cls is None or not callable(getattr(cls, "load", None)):
        raise ImportError("morpheus_encoder.learned_search.model.LearnedSearchModel.load(path, map_location=) "
                          "is not available in this checkout")
    return cls.load(checkpoint, map_location=str(device) if device else "cpu")


#: Public name: maps prefixed variable names to indices into [X | extra columns] (shared by
#: the packet runner across proposers).
NameResolver = _Resolver


# ---------------------------------------------------------------- stub model
def _within_centre(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    for g in np.unique(groups):
        rows = groups == g
        a[rows] -= a[rows].mean(axis=0)
    return a


def neighbour_operator(nb: np.ndarray):
    """Row-normalised sparse neighbour-mean operator (own value where a row has none);
    ``neighbour_operator(nb) @ X`` equals ``neighbour_mean`` column by column."""
    from scipy import sparse
    n = len(nb)
    valid = nb >= 0
    count = valid.sum(1)
    rows = np.repeat(np.arange(n), count)
    cols = nb[valid]
    vals = np.repeat(1.0 / np.maximum(count, 1), count)
    lonely = np.flatnonzero(count == 0)
    rows, cols, vals = (np.concatenate((rows, lonely)), np.concatenate((cols, lonely)),
                        np.concatenate((vals, np.ones(len(lonely)))))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def grouped_cv_r2(V: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int = 0, n_splits: int = 5,
                  alpha: float = 1.0) -> float:
    """Patient-grouped K-fold held-out R^2 of a ridge on within-patient-centred columns."""
    Vc, yc = _within_centre(V, groups), _within_centre(y, groups)
    uniq = np.unique(groups)
    folds = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_splits))
    fold = np.array([folds[g] for g in groups])
    pred = np.zeros(len(y))
    for k in range(n_splits):
        te = fold == k
        if te.sum() == 0 or (~te).sum() < 2:
            continue
        mu, sd = Vc[~te].mean(0), Vc[~te].std(0)
        sd[sd < 1e-12] = 1.0
        A, B = (Vc[~te] - mu) / sd, (Vc[te] - mu) / sd
        beta = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (yc[~te] - yc[~te].mean()))
        pred[te] = B @ beta + yc[~te].mean()
    total = float(np.sum((yc - yc.mean()) ** 2))
    return 0.0 if total <= 1e-15 else float(1.0 - np.sum((yc - pred) ** 2) / total)


def _stub_sets(X: np.ndarray, y: np.ndarray, groups: np.ndarray, nb6: np.ndarray | None, names: Sequence[str],
               *, budget: int, seed: int, n_top: int = 16, n_pair: int = 8) -> tuple[list, int]:
    blocks, labels = [X], [f"own:{c}" for c in names]
    if nb6 is not None:
        lag = np.asarray(neighbour_operator(nb6) @ X)
        blocks += [lag, X - lag]
        labels += [f"nbr6:{c}" for c in names] + [f"contrast:{c}" for c in names]
    V = _within_centre(np.column_stack(blocks), groups)
    yc = _within_centre(y, groups)
    sd = V.std(0)
    ysd = max(float(yc.std()), 1e-12)
    r = np.where(sd > 1e-12, (V * (yc - yc.mean())[:, None]).mean(0) / (np.maximum(sd, 1e-12) * ysd), 0.0)
    order = [int(i) for i in np.argsort(-np.abs(r), kind="stable") if sd[i] > 1e-12][:n_top]
    sets = [(i,) for i in order] + list(combinations(order[:n_pair], 2))
    scored = [(tuple(labels[i] for i in s), grouped_cv_r2(V[:, list(s)], y, groups, seed=seed),
               {"g_ho_support": None, "path_rank": None, "response_surface": None, "n_sets_evaluated": len(sets),
                "marginal_abs_r": [float(abs(r[i])) for i in s]}) for s in sets]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:budget], len(sets)


# -------------------------------------------------------------------- propose
def propose(model, X, y, groups, neighbours6, coords, names: list, views: dict, *, embed=None, budget: int = 128,
            n_paths=None, seed: int = 0, device=None, strategy: str = "pointer", nuisance=None, raw=None,
            family=None) -> ProposeResult:
    """Ranked candidates from the learned-search model (proposal rows only); see the module docstring."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.zeros(len(y), dtype=int) if groups is None else np.asarray(groups)
    if len(names) != X.shape[1]:
        raise ValueError(f"names has {len(names)} entries, X has {X.shape[1]} columns")
    nb6 = pad_neighbours(neighbours6, len(X))
    if model is None:
        sets, n_tested = _stub_sets(X, y, groups, nb6, names, budget=int(budget), seed=int(seed))
        report: dict = {"stub": True, "proposer": "learned_search",
                        "ranking": "marginal screen (model=None): top-16 variables by within-patient |r| over "
                                   "own / nbr6 / contrast, singles + pairs of the top 8, ranked by patient-grouped "
                                   "5-fold CV R^2 of a linear fit"}
    else:
        kwargs = dict(names=list(names), views=dict(views or {}), embed=embed,
                      n_paths=n_paths, seed=int(seed), device=device)
        # Preserve compatibility with frozen pointer-only model adapters.
        if strategy != "pointer":
            kwargs["strategy"] = strategy
        if nuisance is not None:   # v03 models: log depth of the proposal rows (the nuisance baseline)
            kwargs["nuisance"] = nuisance
        if raw is not None:        # v03 models: un-centred proposal-row inputs (training construction)
            kwargs["raw"] = raw
        if family is not None:     # v03 models: the protocol family requirement, at candidate generation
            kwargs["family"] = family
        sets = list(model.propose_sets(X, y, groups, nb6, **kwargs))[:int(budget)]
        n_tested = max([int((info or {}).get("n_sets_evaluated", 0) or 0) for _, _, info in sets] + [len(sets)])
        report = {"stub": False, "proposer": "learned_search", "model": type(model).__name__,
                  "strategy": strategy}
    resolver = _Resolver(X, names)
    candidates: list[Candidate] = []
    seen: set = set()
    for rank, (set_names, cv_r2, info) in enumerate(sets):
        info = dict(info or {})
        resolved = [resolver.resolve(nm) for nm in set_names]
        pairs = sorted({idx: spec for idx, spec in resolved}.items())
        variables = tuple(idx for idx, _ in pairs)
        if variables in seen:
            continue
        seen.add(variables)
        k = len(variables)
        path_rank = info.get("path_rank")
        candidates.append(Candidate(
            variables, feature_form="product" if k >= 2 else "linear_subspace",
            weights=() if k >= 2 else (1.0,), proposal_family=str(info.get("proposal_family", "learned_search")),
            proposal_score=float(cv_r2),
            metadata={"proposer": "learned_search", "provenance": {
                "hypothesis_kind": "learned_set",
                "transforms": [{"name": spec["name"], "transform": spec["transform"], "base": spec["base"]}
                               for _, spec in pairs],
                "g_ho_support": info.get("g_ho_support"),
                "learned": {k: info.get(k) for k in ("support_gain", "predicted_shrinkage", "predicted_sd",
                                                     "predicted_gain") if k in info} or None,
                "path_rank": rank if path_rank is None else path_rank,
                "recurrent_in_generator": None}}))
    if getattr(model, "kind", "") == "gain_v03":   # v03: nbr18 = two-hop ring of nbr6 (training and the pool)
        coords = None
    extra = transform_columns(resolver.specs, X, nb6, embed=embed, model=model, coords=coords,
                              groups=None if coords is None else groups)
    report.update({"n_sets_returned": len(sets), "n_candidates": len(candidates),
                   "n_extra_columns": len(resolver.specs), "budget": int(budget), "seed": int(seed),
                   "coords_given": coords is not None,
                   "nbr18_rule": "18 nearest same-patient spots (coordinates)" if coords is not None
                   else "two-hop ring of the 6-neighbour graph"})
    return ProposeResult(candidates, extra, [s["name"] for s in resolver.specs], int(n_tested), report,
                         [dict(s) for s in resolver.specs])
