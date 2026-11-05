"""General, split-safe discovery primitives for MORPHEUS.

This module deliberately has no vocabulary of scientific relation labels.  It
proposes compact variable sets, subspaces and context-conditioned features,
then asks whether a capacity-controlled relationship explains independent data
better than a simpler account of the same observed variables.  It is designed
to accept lightweight episode objects by duck typing: callers may supply
``episode.X``, ``episode.y`` and optional ``episode.context`` arrays, or pass
arrays directly to the public functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
import weakref

import numpy as np


Array = np.ndarray


def _array(value: Any, *, ndim: int | None = None, name: str = "array") -> Array:
    out = np.asarray(value, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {out.ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def episode_arrays(episode: Any) -> tuple[Array, Array, Array | None]:
    """Extract ``X``, ``y`` and optional ``context`` from an episode by duck type.

    Either attributes or mapping keys are accepted, so panel/data adapters do
    not need to inherit from a discovery-core base class.
    """
    get = episode.get if isinstance(episode, Mapping) else lambda key, default=None: getattr(episode, key, default)
    X, y, context = get("X"), get("y"), get("context", None)
    if X is None or y is None:
        raise ValueError("episode must expose X and y")
    X, y = _array(X, ndim=2, name="episode.X"), _array(y, ndim=1, name="episode.y")
    if len(X) != len(y):
        raise ValueError("episode X/y length mismatch")
    context = None if context is None else _array(context, ndim=2, name="episode.context")
    if context is not None and len(context) != len(X):
        raise ValueError("episode context length mismatch")
    return X, y, context


def contextual_feature_matrix(
    X: Any, *, neighbours: Any | None = None, time: Any | None = None,
    unit: Any | None = None, lags: Sequence[int] = (1, 2, 3),
) -> tuple[Array, tuple[Mapping[str, Any], ...]]:
    """Append generic observed context views while retaining feature provenance.

    A neighbourhood mean and within-unit lagged copies are general contextual
    transformations, not declarations of a scientific relationship.  They are
    available to every representation arm under the same budget and carry a
    machine-readable provenance record so later characterization can say which
    scale a selected coordinate came from. Missing context simply omits that
    view; no zeros are fabricated as if they were measurements.
    """
    base = _array(X, ndim=2, name="X")
    n, d = base.shape
    matrices = [base]
    provenance: list[Mapping[str, Any]] = [
        {"source": "observed", "feature": int(j)} for j in range(d)
    ]
    if neighbours is not None:
        index = np.asarray(neighbours, dtype=object)
        if index.shape[0] != n:
            raise ValueError("neighbours must contain one iterable per observation")
        aggregate = np.zeros_like(base)
        for row, values in enumerate(index):
            ids = np.asarray(values, dtype=int).reshape(-1)
            if ids.size and (np.any(ids < 0) or np.any(ids >= n)):
                raise ValueError("neighbour index exceeds observation range")
            if ids.size:
                aggregate[row] = base[ids].mean(axis=0)
        matrices.append(aggregate)
        provenance.extend({"source": "neighbour_mean", "feature": int(j)} for j in range(d))
    if time is not None:
        t = np.asarray(time).reshape(-1)
        if t.shape[0] != n:
            raise ValueError("time must align with X")
        groups = np.zeros(n, dtype=int) if unit is None else np.asarray(unit).reshape(-1)
        if groups.shape[0] != n:
            raise ValueError("unit must align with X")
        for lag in sorted({int(value) for value in lags if int(value) > 0}):
            delayed = np.zeros_like(base)
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                order = rows[np.argsort(t[rows], kind="stable")]
                if len(order) > lag:
                    delayed[order[lag:]] = base[order[:-lag]]
            matrices.append(delayed)
            provenance.extend({"source": "lag", "lag": lag, "feature": int(j)} for j in range(d))
            # A spatiotemporal mechanism may depend on a neighbour's *past*
            # measurement rather than a separate spatial or temporal view.
            # This is a generic, response-free composition of two available
            # context operators; it is not a boss-world-specific feature.
            if neighbours is not None:
                delayed_neighbour = np.zeros_like(base)
                for row, values in enumerate(index):
                    ids = np.asarray(values, dtype=int).reshape(-1)
                    if ids.size:
                        delayed_neighbour[row] = delayed[ids].mean(axis=0)
                matrices.append(delayed_neighbour)
                provenance.extend({"source": "neighbour_mean_lag", "lag": lag,
                                   "feature": int(j)} for j in range(d))
    return np.column_stack(matrices), tuple(provenance)


def _canonical(value: Any) -> str:
    """Stable serialization for hashes, including numpy scalar values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


@dataclass(frozen=True)
class Candidate:
    """A response-selected, immutable scientific relationship candidate.

    ``variables`` index representation coordinates. ``weights`` parameterize a
    sparse linear subspace when present; otherwise the candidate's feature is
    a product over its variables. Context coordinates are optional and are
    multiplied with the base feature. The compact ``feature_form`` describes
    only how to evaluate a generic numerical feature, not its scientific name.
    """

    variables: tuple[int, ...]
    context_variables: tuple[int, ...] = ()
    weights: tuple[float, ...] = ()
    feature_form: str = "product"
    proposal_family: str = "screen"
    proposal_score: float = float("nan")
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.variables or any(i < 0 for i in self.variables):
            raise ValueError("variables must be nonempty nonnegative indices")
        if tuple(sorted(set(self.variables))) != self.variables:
            raise ValueError("variables must be sorted and unique")
        if any(i < 0 for i in self.context_variables):
            raise ValueError("context_variables must be nonnegative indices")
        if self.weights and len(self.weights) != len(self.variables):
            raise ValueError("weights must match variables")
        if self.feature_form not in {"linear_subspace", "product"}:
            raise ValueError("feature_form must be linear_subspace or product")
        payload = {
            "variables": self.variables, "context_variables": self.context_variables,
            "weights": tuple(float(x) for x in self.weights),
            "feature_form": self.feature_form, "proposal_family": self.proposal_family,
            "metadata": dict(self.metadata),
        }
        # A frozen dataclass alone does not prevent mutation through a supplied
        # dict. Copy and expose a read-only view so the hash actually names a
        # frozen proposal specification.
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "candidate_hash", sha256(_canonical(payload).encode()).hexdigest())

    def feature(self, X: Any, context: Any | None = None) -> Array:
        X = _array(X, ndim=2, name="X")
        if max(self.variables) >= X.shape[1]:
            raise ValueError("candidate variable exceeds X width")
        values = X[:, self.variables]
        if self.feature_form == "linear_subspace":
            weights = np.asarray(self.weights if self.weights else np.ones(len(self.variables)), dtype=float)
            out = values @ weights
        else:
            out = np.prod(values, axis=1)
        if self.context_variables:
            if context is None:
                raise ValueError("candidate requires context")
            context = _array(context, ndim=2, name="context")
            if context.shape[0] != X.shape[0] or max(self.context_variables) >= context.shape[1]:
                raise ValueError("context shape is incompatible with candidate")
            out = out * np.prod(context[:, self.context_variables], axis=1)
        return out


@dataclass(frozen=True)
class DiscoveryReceipt:
    """Held-out evidence for one already frozen candidate."""

    candidate_hash: str
    candidate: Candidate
    proposal_gain: float
    replication_gain: float
    proposal_r2_simple: float
    proposal_r2_full: float
    replication_r2_simple: float
    replication_r2_full: float
    n_proposal: int
    n_replication: int
    model: str
    baseline_width: int = 0
    null_gains: tuple[float, ...] = ()
    null_p_value: float | None = None
    alpha_simple: float | None = None
    alpha_full: float | None = None
    #: Relation version (see :data:`RELATIONS`). ``"v04"`` is the registered
    #: v0.3/v0.4 comparison; ``"v05"`` makes a single-variable gain the total
    #: information gain over the baseline and records the linear / beyond-linear
    #: decomposition in ``decomposition``.
    relation: str = "v04"
    decomposition: Mapping[str, Any] | None = None


def _correlation(x: Array, y: Array) -> float:
    x, y = np.asarray(x), np.asarray(y)
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return 0.0 if denom <= 1e-15 else float(np.dot(x, y) / denom)


def _linear_residual(y: Array, X: Array) -> Array:
    """Return a low-capacity additive residual for generic forward search."""
    if X.size == 0:
        return y.copy()
    return y - _ridge_fit_predict(X, y, X, alpha=1.0)


def _product_screen_score(X: Array, y: Array, variables: tuple[int, ...]) -> float:
    """Rank a coordinate set with direct and lower-order-residual evidence."""
    product = np.prod(X[:, variables], axis=1)
    residual = _linear_residual(y, X[:, variables])
    return max(abs(_correlation(product, y)), abs(_correlation(product, residual)))


def propose_candidates(
    X: Any, y: Any, *, context: Any | None = None, max_candidates: int = 32,
    max_pairs: int = 16, max_subspaces: int = 8, max_order: int = 5,
    forward_beam_width: int | None = None, max_forward_per_order: int | None = None,
    min_context_candidates: int | None = None,
) -> tuple[Candidate, ...]:
    """Create a bounded heterogeneous proposal family from proposal data only.

    The family combines marginal variables, sparse response-selected coordinate
    sets through bounded residual/forward expansion, principal subspaces, and
    context-conditioned variables. It intentionally does not inspect candidate
    truth or relation labels. ``max_pairs`` retains its old meaning; later
    orders use the separately bounded forward-search budget.
    """
    X, y = _array(X, ndim=2, name="X"), _array(y, ndim=1, name="y")
    if len(y) != len(X):
        raise ValueError("X and y length mismatch")
    if max_candidates < 1 or max_pairs < 0 or max_subspaces < 0:
        raise ValueError("proposal budgets must be nonnegative and max_candidates positive")
    if not 1 <= max_order <= 5:
        raise ValueError("max_order must be between 1 and 5")
    forward_beam_width = int(forward_beam_width if forward_beam_width is not None else max(8, max_pairs))
    max_forward_per_order = int(max_forward_per_order if max_forward_per_order is not None else max_pairs)
    if forward_beam_width < 1 or max_forward_per_order < 0:
        raise ValueError("forward-search budgets must be nonnegative (beam must be positive)")
    scores = np.array([abs(_correlation(X[:, j], y)) for j in range(X.shape[1])])
    order = np.argsort(-scores, kind="stable")
    result: list[Candidate] = []
    # Marginal (one-coordinate) candidates retain easy effects and provide a
    # meaningful simpler-model competitor for more complex candidates.
    for j in order[:max_candidates]:
        result.append(Candidate((int(j),), feature_form="linear_subspace", weights=(1.0,),
                                proposal_family="marginal", proposal_score=float(scores[j])))
    # A deterministic screened pool makes combinatorial search bounded. Small
    # representations retain all coordinates, so a pure higher-order signal is
    # not excluded merely because its individual marginals vanish.
    pool_size = min(X.shape[1], max(8, int(np.ceil(np.sqrt(max_candidates * max_order * 4)))))
    pool = tuple(int(j) for j in order[:pool_size])
    previous: list[tuple[float, tuple[int, ...]]] = []
    if max_order >= 2 and max_pairs:
        pair_scores: list[tuple[float, tuple[int, ...]]] = []
        for left_pos, left in enumerate(pool):
            for right in pool[left_pos + 1:]:
                variables = (left, right) if left < right else (right, left)
                pair_scores.append((_product_screen_score(X, y, variables), variables))
        previous = sorted(pair_scores, key=lambda item: (-item[0], item[1]))[:max_pairs]
        for score, variables in previous:
            result.append(Candidate(variables, feature_form="product",
                                    proposal_family="forward_screen", proposal_score=float(score)))
    # Expand only the best already-scored sparse sets. At every order the
    # candidate is re-scored against response and additive residual, retaining
    # incremental structure without a hand-written relation family.
    for current_order in range(3, max_order + 1):
        if not previous or not max_forward_per_order:
            break
        expansions: dict[tuple[int, ...], float] = {}
        for _, parent in previous[:forward_beam_width]:
            for coordinate in pool:
                if coordinate in parent:
                    continue
                variables = tuple(sorted((*parent, coordinate)))
                if len(variables) != current_order:
                    continue
                score = _product_screen_score(X, y, variables)
                if score > expansions.get(variables, -np.inf):
                    expansions[variables] = score
        previous = sorted(((score, variables) for variables, score in expansions.items()),
                          key=lambda item: (-item[0], item[1]))[:max_forward_per_order]
        for score, variables in previous:
            result.append(Candidate(variables, feature_form="product",
                                    proposal_family="forward_residual", proposal_score=float(score)))
    # Response-independent PCA directions are sparse proxies for broad latent
    # axes; their response score only ranks the already defined directions.
    if max_subspaces and X.shape[1] >= 2:
        _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
        for direction in vt[:min(max_subspaces, len(vt))]:
            support = np.argsort(-np.abs(direction))[:min(4, max_order, X.shape[1])]
            support = tuple(sorted(int(i) for i in support))
            weights = tuple(float(direction[i]) for i in support)
            value = X[:, support] @ np.asarray(weights)
            result.append(Candidate(support, weights=weights, feature_form="linear_subspace",
                                    proposal_family="subspace", proposal_score=abs(_correlation(value, y))))
    if context is not None:
        context = _array(context, ndim=2, name="context")
        if context.shape[0] != X.shape[0]:
            raise ValueError("context length mismatch")
        context_scores = np.array([abs(_correlation(context[:, k], y)) for k in range(context.shape[1])])
        for j in order[:min(4, len(order))]:
            for k in np.argsort(-context_scores, kind="stable")[:min(2, len(context_scores))]:
                score = abs(_correlation(X[:, j] * context[:, k], y))
                result.append(Candidate((int(j),), (int(k),), feature_form="linear_subspace", weights=(1.0,),
                                        proposal_family="context_screen", proposal_score=float(score)))
    # Deduplicate exact frozen specifications and apply a shared budget.  When
    # context is supplied, context-conditioned objects must have a protected
    # place in the proposal family.  Otherwise high-scoring marginal features
    # can consume the entire generic budget before a single intrinsic-context
    # relation is retained; that is proposal-family starvation, not evidence
    # against the contextual relation.  The reserved count is fixed from the
    # public budget, never from a replication score.
    unique: dict[str, Candidate] = {}
    for candidate in result:
        if candidate.candidate_hash not in unique:
            unique[candidate.candidate_hash] = candidate
    ranked = sorted(unique.values(), key=lambda c: (-np.nan_to_num(c.proposal_score, nan=-np.inf), c.candidate_hash))
    if context is None:
        return tuple(ranked[:max_candidates])
    context_ranked = [c for c in ranked if c.context_variables]
    ordinary_ranked = [c for c in ranked if not c.context_variables]
    reserve = min(len(context_ranked), max(1, int(min_context_candidates if min_context_candidates is not None
                                                   else max_candidates // 3)))
    # Preserve the overall response-ranked order within each fixed family;
    # merge then makes the family allocation transparent and deterministic.
    selected = context_ranked[:reserve] + ordinary_ranked[:max_candidates - reserve]
    return tuple(sorted(selected, key=lambda c: (-np.nan_to_num(c.proposal_score, nan=-np.inf), c.candidate_hash)))


#: Protocol v0.4 validation grid: both the competitor and the full model pick
#: their ridge strength independently by patient-grouped inner CV on proposal
#: rows over this grid, which ends in the zero (fold-mean) model.  The default
#: grid (``FittedRelation.ALPHA_GRID``) is kept for v0.3 reproduction.
VALIDATION_ALPHA_GRID: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, float("inf"))


#: Relation versions.  ``"v04"`` (default; v0.3/v0.4 bit-for-bit): a
#: single-variable candidate's competitor already holds the variable LINEARLY,
#: so its gain is non-linearity beyond the linear term, not information.
#: ``"v05"``: a single-variable candidate's competitor is the baseline (plus
#: any context columns) only, so ``replication_gain`` is the variable's total
#: held-out information gain; the receipt also records ``gain_linear`` and
#: ``gain_nonlinear_beyond_linear`` (which sum to it exactly).  For k >= 2 the
#: comparison is unchanged and ``gain_linear_additive`` is recorded beside it.
RELATIONS: tuple[str, ...] = ("v04", "v05", "v06", "v06_offset", "lin_spline_v01", "v07_ho")

#: v0.6 relations: the comparison block (baseline + context) no longer shares one
#: ridge penalty with the hypothesis block (Agent Q's benchmark: under one shared
#: penalty a wide comparison block shrinks a true 2 % hypothesis to zero gain).
#: ``"v06"``: one ridge over [comparison | hypothesis block] with TWO penalties
#: chosen jointly by patient-grouped inner CV on a small grid
#: (``TWO_PENALTY_GRID_COMPARISON`` x ``TWO_PENALTY_GRID_BLOCK``); the
#: competitor is the same two-penalty fit of [comparison | simpler block] (one
#: variable: the comparison model alone with its own penalty, as in v0.5).
#: ``"v06_offset"``: the comparison model is fit first with its own inner-CV
#: penalty and carried as a cross-fitted offset; the hypothesis block
#: (random-feature map + linear constituents), residualised on the comparison
#: columns the same way, is fit to the offset residual with its own penalty.
#: Both were compared on the benchmark worlds (runs/development/relation_v06);
#: ``"v06"`` matched the offset form's power, had the lower false-pass rate under
#: permuted hypothesis blocks, and reproduces v0.5 where the comparison block is
#: informative.
#: ``"lin_spline_v01"`` (patient-level relation class, n ~ 100: the random-feature
#: map is overparameterised there -- Agent P's seed-spread finding): the v06
#: two-penalty engine with a deterministic block -- per variable the linear term
#: plus a natural cubic spline with 3 degrees of freedom (boundary knots at the
#: proposal-row minimum / maximum, interior knots at the proposal-row terciles;
#: ``natural_spline_basis``), and for k >= 2 the pairwise products of the linear
#: terms.  Competitor: k = 1 the comparison model alone; k >= 2 the additive
#: per-variable linear + spline block (so the gain is joint structure beyond the
#: smooth additive effects).  Linear model: the linear constituents.  No seed
#: enters the fit.
V06_RELATIONS: tuple[str, ...] = ("v06", "v06_offset", "lin_spline_v01")

#: ``"v07_ho"`` (general and higher-order information, ajas.md "Define
#: higher-order information correctly"): the v06 two-penalty engine with
#: ``replication_gain`` = the GENERAL gain, full model minus the comparison model
#: alone (reference), for every k.  Full model: comparison + [joint random-feature
#: map of the selected variables (and their candidate feature) + the variables
#: linearly (+ pairwise products for k >= 3)], exactly the v06 full block.  The
#: decomposition also fits, in the same function family and with matched
#: capacity, every proper-subset decomposition of the set (k = 2: A+B; k = 3:
#: AB+C, AC+B, BC+A, A+B+C; k = 4: all 14 set partitions): one joint map per
#: subset (n_rbf features per variable; a singleton's map is the v06
#: per-constituent map) + the variables linearly.  ``G_HO`` = held-out R^2 of the
#: full model minus the best decomposition's, with a held-out-patient bootstrap
#: interval; ``gain_joint_vs_additive`` is the v06 gain of the same set.
#: A row is a higher-order discovery when the ``G_HO`` interval's lower bound > 0.
HO_RELATIONS: tuple[str, ...] = ("v07_ho",)

#: Relations whose penalties come from inner CV (a null evaluator / row CI reuses
#: the receipt's own fit instead of ``fixed_alphas``).
CV_RELATIONS: tuple[str, ...] = V06_RELATIONS + HO_RELATIONS


def set_partitions(items: Sequence[int]) -> list[list[tuple[int, ...]]]:
    """All set partitions of ``items`` (Bell(k) of them), each a list of sorted tuples."""
    items = list(items)
    if not items:
        return [[]]
    head, rest = items[0], items[1:]
    out = []
    for part in set_partitions(rest):
        out.append([(head,)] + part)
        for i in range(len(part)):
            out.append(part[:i] + [tuple(sorted((head,) + part[i]))] + part[i + 1:])
    return out


def decomposition_names(k: int) -> list[tuple[str, list[tuple[int, ...]]]]:
    """Proper-subset decompositions of a k-set (every set partition but the whole set),
    named by letters in candidate-variable order, larger blocks first: k = 3 ->
    ``AB+C, AC+B, BC+A, A+B+C``; 1 / 4 / 14 decompositions for k = 2 / 3 / 4."""
    letters = "ABCDEFGH"
    parts = [sorted(p, key=lambda s: (-len(s), s)) for p in set_partitions(range(k)) if len(p) > 1]
    parts.sort(key=lambda p: ([-len(s) for s in p], p))
    return [("+".join("".join(letters[i] for i in s) for s in p), p) for p in parts]


def natural_spline_knots(x: Array) -> Array:
    """Boundary knots at min / max, interior knots at the terciles (proposal rows)."""
    x = np.asarray(x, dtype=float)
    return np.quantile(x, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])


def natural_spline_basis(x: Array, knots: Array) -> Array:
    """Non-linear columns of the natural cubic spline basis (ESL eq. 5.4-5.5): with K knots,
    N_{k+2}(x) = d_k(x) - d_{K-1}(x), d_k = ((x - xi_k)_+^3 - (x - xi_K)_+^3) / (xi_K - xi_k),
    k = 1..K-2. With 4 knots: 2 columns; with the linear term, 3 degrees of freedom.
    Linear beyond the boundary knots. Degenerate (tied) knots give zero columns."""
    x = np.asarray(x, dtype=float).reshape(-1)
    xi = np.asarray(knots, dtype=float)
    K = len(xi)

    def d(k):
        span = xi[-1] - xi[k]
        if span <= 1e-12:
            return np.zeros_like(x)
        return (np.clip(x - xi[k], 0, None) ** 3 - np.clip(x - xi[-1], 0, None) ** 3) / span
    last = d(K - 2)
    return np.column_stack([d(k) - last for k in range(K - 2)])
TWO_PENALTY_GRID_COMPARISON: tuple[float, ...] = (1e2, 1e3, 1e4, 1e5, 1e6, float("inf"))
TWO_PENALTY_GRID_BLOCK: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, float("inf"))


#: Registered hypothesis sources. Every one emits ``Candidate`` objects, so the
#: downstream freeze -> held-out fit -> null path is identical for all of them.
#:
#: ``residual_bank`` (protocol v0.6) resolves to
#: ``morpheus_encoder.residual_bank_generator.generate_residual_bank_hypotheses``
#: (Agent G's module; owned there, only dispatched here).  Contract it must keep:
#: called with the PROPOSAL rows only -- ``X`` (bank design, protected views
#: listed in ``views``), ``y``, ``groups`` (patient codes), ``comparison`` (the
#: registered comparison model's columns) and ``levels`` (level name -> column
#: indices of N, B1, B2, B3 from ``dbench.decomposition.ladder_levels``, the
#: same levels the in-run ladder validates at), plus ``views``, ``roles``,
#: ``column_names``, ``max_candidates``, ``seed``, ``device``.  It builds the
#: cross-fitted residual bank r(k) = y - f_k(B_k) on those rows itself and
#: returns an object with ``candidates`` (ranked ``Candidate`` objects: variable
#: column indices into ``X``, relation class in ``feature_form`` --
#: ``linear_subspace`` for a single variable, ``product`` for an interaction --
#: and ``metadata["proposer"] == "residual_bank"`` with the nominating levels in
#: ``metadata["provenance"]``) and ``report`` (JSON-safe search report and
#: cost).  A checkout without the module fails with a clear ImportError.
#:
#: ``learned_search`` (sprint, development mode) resolves to
#: ``morpheus_encoder.learned_search.propose.propose`` (the learned scientific-search
#: model, or its marginal-screen stub without a checkpoint); the packet runner
#: dispatches it and appends its transformed variables as extra design columns.
#: ``classical_pair`` (blind-discovery protocols): the benchmark's ``ebm_ga2m`` and
#: ``gbm_interaction`` searches on the proposal rows of the same design, dispatched by the
#: packet runner (``_classical_pair_pool``).
PROPOSERS = ("enumeration", "residual_generator", "all", "residual_bank", "learned_search", "classical_pair")
RESIDUAL_BANK_MODULE = "morpheus_encoder.residual_bank_generator"


def residual_bank_proposer():
    """``generate_residual_bank_hypotheses`` from :data:`RESIDUAL_BANK_MODULE`, or a clear error."""
    import importlib
    try:
        module = importlib.import_module(RESIDUAL_BANK_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"proposer 'residual_bank' needs {RESIDUAL_BANK_MODULE} (the residual-bank generator, "
            f"Agent G), which is not importable in this checkout: {exc}") from exc
    fn = getattr(module, "generate_residual_bank_hypotheses", None)
    if fn is None:
        raise ImportError(f"{RESIDUAL_BANK_MODULE} has no generate_residual_bank_hypotheses "
                          "(see the proposer contract above PROPOSERS)")
    return fn


def propose_candidates_by(
    proposer: str, X: Any, y: Any, *, groups: Any | None = None, baseline: Any | None = None,
    context: Any | None = None, max_candidates: int = 32, enumeration_width: int | None = None,
    generator_kwargs: Mapping[str, Any] | None = None, **enumeration_kwargs: Any,
) -> tuple[tuple[Candidate, ...], dict | None]:
    """Dispatch to a registered proposer (proposal data only).

    ``enumeration`` is :func:`propose_candidates` over the first
    ``enumeration_width`` columns (default: all), so its candidates are the
    ones it would produce without the generator's extra design columns.
    ``residual_generator`` is
    :func:`morpheus_encoder.hypothesis_generator.generate_hypotheses` with the
    nuisance ``baseline`` as its conventional explanation. ``all`` returns
    both families (each with its own ``max_candidates`` budget). Returns the
    candidates and the generator's search report (``None`` for enumeration).
    ``residual_bank`` calls :func:`residual_bank_proposer` with ``baseline`` as
    the comparison model and ``generator_kwargs`` (which must carry ``levels``).
    """
    if proposer not in PROPOSERS:
        raise ValueError(f"proposer must be one of {PROPOSERS}")
    if proposer in ("learned_search", "classical_pair"):
        raise ValueError(f"proposer {proposer!r} needs neighbours, column names and views: the packet runner "
                         "dispatches it (run_general_discovery_packet._propose)")
    if proposer == "residual_bank":
        result = residual_bank_proposer()(X, y, groups, comparison=baseline, max_candidates=max_candidates,
                                          **dict(generator_kwargs or {}))
        return tuple(result.candidates), result.report
    found: list[Candidate] = []
    report = None
    if proposer in ("enumeration", "all"):
        X_enum = _array(X, ndim=2, name="X")
        if enumeration_width is not None:
            X_enum = X_enum[:, :int(enumeration_width)]
        found.extend(propose_candidates(X_enum, y, context=context, max_candidates=max_candidates,
                                        **enumeration_kwargs))
    if proposer in ("residual_generator", "all"):
        from morpheus_encoder.hypothesis_generator import generate_hypotheses
        result = generate_hypotheses(X, y, groups, conventional=baseline, context=context,
                                     max_candidates=max_candidates, **dict(generator_kwargs or {}))
        found.extend(result.candidates)
        report = result.report
    return tuple(found), report


def _standardize_train_apply(train: Array, test: Array) -> tuple[Array, Array]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge_fit_predict(train: Array, y: Array, test: Array, alpha: float) -> Array:
    train, test = _standardize_train_apply(train, test)
    design = np.column_stack((np.ones(len(train)), train))
    query = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return query @ beta


def _rbf_features(train: Array, test: Array, *, n_features: int, seed: int) -> tuple[Array, Array]:
    train, test = _standardize_train_apply(train, test)
    rng = np.random.default_rng(seed)
    width = max(1, train.shape[1])
    omega = rng.normal(size=(width, n_features))
    phase = rng.uniform(0, 2 * np.pi, size=n_features)
    scale = np.sqrt(2.0 / n_features)
    return scale * np.cos(train @ omega + phase), scale * np.cos(test @ omega + phase)


def _simple_design(X: Array, candidate: Candidate, context: Array | None,
                    baseline: Array | None = None) -> Array:
    """The best-simpler-competitor design B (D3): the candidate's own
    constituents/context, PLUS, for an order->=3 candidate, every pairwise
    product among its own constituents (so B already contains the 2-way
    interactions the candidate could trivially reduce to -- otherwise
    "beats B" could just mean "found any pairwise interaction", not the
    higher-order structure actually being claimed), PLUS a caller-supplied
    nuisance baseline (declared covariates such as depth/composition that
    must never itself count as the discovery).
    """
    values = X[:, candidate.variables]
    pieces = [values]
    if candidate.context_variables:
        assert context is not None
        pieces.append(context[:, candidate.context_variables])
    if values.shape[1] >= 3:
        for i in range(values.shape[1]):
            for j in range(i + 1, values.shape[1]):
                pieces.append((values[:, i] * values[:, j]).reshape(-1, 1))
    if baseline is not None:
        pieces.append(baseline)
    return np.column_stack(pieces)


def _r2(y: Array, prediction: Array) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if total <= 1e-15 else float(1.0 - np.sum((y - prediction) ** 2) / total)


class _Standardizer:
    def __init__(self, train: Array):
        self.mean = train.mean(axis=0, keepdims=True)
        self.scale = train.std(axis=0, keepdims=True)
        self.scale[self.scale < 1e-12] = 1.0

    def __call__(self, x: Array) -> Array:
        return (x - self.mean) / self.scale


class _RidgeFit:
    """Ridge fit on standardised columns; `predict` reuses the training statistics."""

    def __init__(self, train: Array, y: Array, alpha: float):
        self.std = _Standardizer(train)
        design = np.column_stack((np.ones(len(train)), self.std(train)))
        if not np.isfinite(alpha):
            # Infinite shrinkage is the zero model: predict the training mean.
            self.beta = np.zeros(design.shape[1])
            self.beta[0] = float(np.mean(y))
            return
        penalty = np.eye(design.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        try:
            inverse = np.linalg.pinv(gram)
        except np.linalg.LinAlgError:
            # LAPACK's divide-and-conquer SVD can fail to converge on a badly
            # conditioned Gram matrix (e.g. a near-constant product column on a
            # handful of rows); the QR-iteration driver is the robust fallback.
            # Only reached where the default path raised, so results elsewhere
            # are unchanged.
            from scipy.linalg import svd as _svd
            u, sv, vt = _svd(gram, full_matrices=False, lapack_driver="gesvd")
            keep = sv > sv.max() * max(gram.shape) * np.finfo(float).eps
            inverse = (vt[keep].T / sv[keep]) @ u[:, keep].T
        self.beta = inverse @ design.T @ y

    def predict(self, test: Array) -> Array:
        return np.column_stack((np.ones(len(test)), self.std(test))) @ self.beta


class _RBFMap:
    """Random Fourier feature map fixed on the training columns (`apply` never refits)."""

    def __init__(self, train: Array, *, n_features: int, seed: int):
        self.std = _Standardizer(train)
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(size=(max(1, train.shape[1]), n_features))
        self.phase = rng.uniform(0, 2 * np.pi, size=n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def apply(self, x: Array) -> Array:
        return self.scale * np.cos(self.std(x) @ self.omega + self.phase)


def _extra_linear(X: Array, candidate: Candidate, context: Array | None) -> Array | None:
    """Context columns and, for order >= 3, the pairwise constituent products: the
    lower-order terms both competitor and full model receive linearly."""
    values = X[:, candidate.variables]
    pieces = []
    if candidate.context_variables:
        assert context is not None
        pieces.append(context[:, candidate.context_variables])
    if values.shape[1] >= 3:
        for i in range(values.shape[1]):
            for j in range(i + 1, values.shape[1]):
                pieces.append((values[:, i] * values[:, j]).reshape(-1, 1))
    return np.column_stack(pieces) if pieces else None


def _reference_design(X: Array, candidate: Candidate, context: Array | None, baseline: Array | None) -> Array:
    """The candidate-free reference: the candidate's context columns and the
    nuisance baseline, linearly (an empty design is the intercept-only model)."""
    pieces = [context[:, candidate.context_variables] if candidate.context_variables and context is not None
              else None, baseline]
    pieces = [p for p in pieces if p is not None]
    return np.column_stack(pieces) if pieces else np.zeros((len(X), 0))


def _linear_design(X: Array, candidate: Candidate, context: Array | None, baseline: Array | None) -> Array:
    """The reference plus every constituent LINEARLY (no products, no maps)."""
    return np.column_stack((X[:, candidate.variables], _reference_design(X, candidate, context, baseline)))


def _cv_alpha(design: Array, y: Array, groups: Array | None, grid: tuple[float, ...], seed: int,
              n_splits: int = 5) -> float:
    """Ridge strength by grouped K-fold CV (held-out R^2) on the training split only."""
    n = len(y)
    if groups is None:
        folds = np.random.default_rng(seed).permutation(n) % n_splits
    else:
        uniq = np.unique(groups)
        assignment = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_splits))
        folds = np.array([assignment[g] for g in groups])
    best_alpha, best_score = grid[0], -np.inf
    for alpha in grid:
        preds = np.empty(n)
        for k in range(n_splits):
            test = folds == k
            if test.sum() == 0 or (~test).sum() < 2:
                continue
            preds[test] = _RidgeFit(design[~test], y[~test], alpha).predict(design[test])
        score = _r2(y, preds)
        if score > best_score:
            best_alpha, best_score = alpha, score
    return float(best_alpha)


def _group_fold_ids(groups: Array | None, n: int, seed: int, n_splits: int = 5) -> Array:
    """The grouped fold assignment ``_cv_alpha`` uses (same seed -> same folds)."""
    if groups is None:
        return np.random.default_rng(seed).permutation(n) % n_splits
    uniq = np.unique(groups)
    assignment = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_splits))
    return np.array([assignment[g] for g in groups])


class _ZeroFit:
    """An empty hypothesis block: contributes exactly nothing to the prediction."""

    def predict(self, design: Array) -> Array:
        return np.zeros(len(design))


class _MultiRidge:
    """Ridge of several targets on the same standardised columns (one penalty);
    an infinite penalty predicts the training means.  Used to residualise the
    hypothesis-block columns on the comparison columns."""

    def __init__(self, train: Array, targets: Array, alpha: float):
        self.std = _Standardizer(train)
        Z = np.column_stack((np.ones(len(train)), self.std(train)))
        if not np.isfinite(alpha):
            self.beta = np.zeros((Z.shape[1], targets.shape[1]))
            self.beta[0] = targets.mean(axis=0)
            return
        pen = np.eye(Z.shape[1]) * float(alpha)
        pen[0, 0] = 0.0
        self.beta = np.linalg.pinv(Z.T @ Z + pen) @ Z.T @ targets

    def predict(self, test: Array) -> Array:
        return np.column_stack((np.ones(len(test)), self.std(test))) @ self.beta


class _OffsetFit:
    """v0.6 offset model: comparison prediction + hypothesis-block prediction on the
    block columns residualised on the comparison columns (``projector``).
    ``predict`` takes [comparison columns | block columns]."""

    def __init__(self, comparison: "_RidgeFit", block: Any, width: int, projector: "_MultiRidge | None" = None):
        self.comparison, self.block, self.width, self.projector = comparison, block, int(width), projector

    def predict(self, design: Array) -> Array:
        R, Xb = design[:, :self.width], design[:, self.width:]
        if self.projector is not None:
            Xb = Xb - self.projector.predict(R)
        return self.comparison.predict(R) + self.block.predict(Xb)


def _two_penalty_beta(Z: Array, y: Array, penalties: Array) -> Array:
    """Ridge with per-column penalties on [1 | standardised columns]; an
    infinite penalty removes the column (coefficient 0)."""
    keep = np.isfinite(penalties)
    beta = np.zeros(Z.shape[1])
    Zk = Z[:, keep]
    gram = Zk.T @ Zk + np.diag(penalties[keep])
    try:
        beta[keep] = np.linalg.solve(gram, Zk.T @ y)
    except np.linalg.LinAlgError:
        beta[keep] = np.linalg.lstsq(gram, Zk.T @ y, rcond=None)[0]
    return beta


class _TwoPenaltyFit:
    """v0.6 alternative: one ridge over [comparison | block] with penalty
    ``alpha_comparison`` on the comparison columns and ``alpha_block`` on the
    block columns (intercept unpenalised; columns standardised on training rows)."""

    def __init__(self, train: Array, y: Array, width: int, alpha_comparison: float, alpha_block: float):
        self.std = _Standardizer(train)
        Z = np.column_stack((np.ones(len(train)), self.std(train)))
        pen = np.concatenate(([0.0], np.full(width, float(alpha_comparison)),
                              np.full(train.shape[1] - width, float(alpha_block))))
        self.beta = _two_penalty_beta(Z, y, pen)

    def predict(self, test: Array) -> Array:
        return np.column_stack((np.ones(len(test)), self.std(test))) @ self.beta


def _cv_two_penalty(design: Array, width: int, y: Array, groups: Array | None, seed: int,
                    grid_comparison: tuple[float, ...] = TWO_PENALTY_GRID_COMPARISON,
                    grid_block: tuple[float, ...] = TWO_PENALTY_GRID_BLOCK, n_splits: int = 5) -> tuple[float, float]:
    """(comparison, block) penalties by grouped K-fold CV (held-out R^2) on training rows."""
    n = len(y)
    folds = _group_fold_ids(groups, n, seed, n_splits)
    pairs = [(a, b) for a in grid_comparison for b in grid_block]
    preds = np.zeros((len(pairs), n))
    for k in range(n_splits):
        te = folds == k
        if te.sum() == 0 or (~te).sum() < 2:
            continue
        std = _Standardizer(design[~te])
        Ztr = np.column_stack((np.ones((~te).sum()), std(design[~te])))
        Zte = np.column_stack((np.ones(te.sum()), std(design[te])))
        for i, (a, b) in enumerate(pairs):
            pen = np.concatenate(([0.0], np.full(width, a), np.full(design.shape[1] - width, b)))
            preds[i, te] = Zte @ _two_penalty_beta(Ztr, y[~te], pen)
    scores = [_r2(y, p) for p in preds]
    best = int(np.argmax(scores))
    return float(pairs[best][0]), float(pairs[best][1])


class FittedRelation:
    """The candidate's relation and its best simpler competitor, fit ONCE on the
    proposal split; `gain(X, y, context, baseline)` scores any replication-side
    arrays (the independent split or a structured-null transformation of it)
    without refitting, so 199 null draws cost 199 predictions, not 199 fits.

    Competitor B (D3, discoadd "best simpler explanation"): the nuisance
    baseline linearly, the context columns and (order >= 3) pairwise products
    linearly, and the constituents either linearly (`model="ridge"`, or a
    single-variable candidate) or as an ADDITIVE flexible model (one random
    Fourier map per constituent, `model="rbf"`, order >= 2). Full model: the
    same linear parts plus the constituents and the candidate feature through
    ONE joint flexible map (`rbf`) or linearly (`ridge`). The gain therefore
    measures joint structure beyond every constituent's own (possibly
    nonlinear) effect and beyond the declared nuisance -- never nonlinearity
    of the baseline, which both models see identically.

    ``relation="v05"``: a single-variable candidate's competitor is the
    context + baseline only (no linear copy of the variable), so its gain is
    total information gain; order >= 2 is unchanged (see :data:`RELATIONS`).
    """

    ALPHA_GRID: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)

    def __init__(self, candidate: Candidate, X: Array, y: Array, *, context: Array | None,
                 baseline: Array | None, model: str, ridge_alpha: float, rbf_features: int,
                 groups: Array | None = None, cv_alpha: bool = True,
                 alpha_grid: tuple[float, ...] | None = None,
                 fixed_alphas: tuple[float, float] | None = None, relation: str = "v04"):
        if model not in ("ridge", "rbf"):
            raise ValueError("model must be ridge or rbf")
        if relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}")
        self.candidate, self.model, self.n_rbf = candidate, model, int(rbf_features)
        self.relation = relation
        # v0.5: a single variable's competitor is the baseline (+ context) only.
        self.reference_competitor = relation == "v05" and len(candidate.variables) == 1
        seed = int(candidate.candidate_hash[:8], 16)
        values = X[:, candidate.variables]
        self.additive_maps: list[_RBFMap] | None = None
        self.joint_map: _RBFMap | None = None
        if model == "rbf":
            k = values.shape[1]
            # Capacity is matched: the additive competitor spends n_rbf features
            # per constituent (k * n_rbf in total) and the joint map receives the
            # same k * n_rbf features, so a gain cannot come from one side
            # simply having more parameters (a wider model's predictions also
            # have larger variance under a permuted response, which would bias
            # the null gains).
            if k >= 2:
                self.additive_maps = [_RBFMap(values[:, [j]], n_features=self.n_rbf, seed=seed + 1 + j)
                                      for j in range(k)]
            joint_train = np.column_stack((values, candidate.feature(X, context).reshape(-1, 1)))
            self.joint_map = _RBFMap(joint_train, n_features=self.n_rbf * max(k, 1), seed=seed)
        if relation in HO_RELATIONS:
            self._init_v07(X, y, context, baseline, groups,
                           tuple(alpha_grid) if alpha_grid is not None else self.ALPHA_GRID, seed,
                           cv_alpha, ridge_alpha)
            return
        if relation in V06_RELATIONS:
            # Inner CV is deterministic, so a null evaluator's refit reproduces the
            # same fit without ``fixed_alphas`` (ignored here).
            self._init_v06(X, y, context, baseline, groups,
                           tuple(alpha_grid) if alpha_grid is not None else self.ALPHA_GRID, seed,
                           cv_alpha, ridge_alpha)
            return
        simple_design = self._simple_design(X, context, baseline)
        full_design = self._full_design(X, context, baseline)
        # Capacity control: the ridge strength of EACH model is chosen by grouped
        # (patient-level when `groups` is given) 5-fold cross-validation inside
        # the proposal split over a fixed grid; the replication split is never
        # consulted. Without this a fixed alpha over ~100 random features
        # overfits the proposal patients and transfers with negative R^2.
        # ``alpha_grid`` (protocol v0.4: VALIDATION_ALPHA_GRID, which ends in the
        # zero model) replaces the default grid; ``fixed_alphas`` reproduces an
        # already chosen pair exactly (a null evaluator's refit).
        grid = tuple(alpha_grid) if alpha_grid is not None else self.ALPHA_GRID
        if fixed_alphas is not None:
            self.alpha_simple, self.alpha_full = (float(a) for a in fixed_alphas)
        else:
            self.alpha_simple = _cv_alpha(simple_design, y, groups, grid, seed) if cv_alpha else ridge_alpha
            self.alpha_full = _cv_alpha(full_design, y, groups, grid, seed) if cv_alpha else ridge_alpha
        self.simple = _RidgeFit(simple_design, y, self.alpha_simple)
        self.full = _RidgeFit(full_design, y, self.alpha_full)

    # ---- v0.6: separate penalties for the comparison and hypothesis blocks ----
    def _block(self, X: Array, context: Array | None, which: str) -> Array:
        """Hypothesis-block columns (no comparison columns).  ``linear``: the
        constituents linearly; ``simple``: empty for one variable, otherwise the
        per-constituent maps + linear constituents (+ pairwise products for
        k >= 3); ``full``: the joint map + linear constituents (+ pairwise)."""
        values = X[:, self.candidate.variables]
        k = values.shape[1]
        if self.relation == "lin_spline_v01":
            if which == "linear":
                return values
            if which == "simple" and k == 1:
                return np.zeros((len(X), 0))
            smooth = [np.column_stack((values[:, [j]], natural_spline_basis(values[:, j], self.spline_knots[j])))
                      for j in range(k)]
            products = [(values[:, i] * values[:, j]).reshape(-1, 1) for i in range(k) for j in range(i + 1, k)]
            return np.column_stack(smooth + (products if which == "full" else []))
        pairwise = [(values[:, i] * values[:, j]).reshape(-1, 1)
                    for i in range(k) for j in range(i + 1, k)] if k >= 3 else []
        if which == "linear":
            pieces = [values]
        elif which == "simple":
            if k == 1:
                return np.zeros((len(X), 0))
            core = (np.column_stack([m.apply(values[:, [j]]) for j, m in enumerate(self.additive_maps)])
                    if self.additive_maps is not None else None)
            pieces = [core, values, *pairwise]
        else:
            feature = self.candidate.feature(X, context).reshape(-1, 1)
            joint = np.column_stack((values, feature))
            core = self.joint_map.apply(joint) if self.joint_map is not None else feature
            pieces = [core, values, *pairwise]
        return np.column_stack([p for p in pieces if p is not None])

    def _init_v06(self, X, y, context, baseline, groups, grid, seed, cv_alpha, ridge_alpha) -> None:
        if self.relation == "lin_spline_v01":  # knots fitted on the proposal rows only
            self.spline_knots = [natural_spline_knots(X[:, j]) for j in self.candidate.variables]
        R = _reference_design(X, self.candidate, context, baseline)
        self._r_width = R.shape[1]
        a_c = _cv_alpha(R, y, groups, grid, seed) if cv_alpha else ridge_alpha
        self.comparison = _RidgeFit(R, y, a_c)
        self.v06_info: dict[str, Any] = {"alpha_comparison": float(a_c)}
        blocks = {w: self._block(X, context, w) for w in ("simple", "linear", "full")}
        if self.relation == "v06_offset":
            folds = _group_fold_ids(groups, len(y), seed)
            offset = np.empty(len(y))
            for f in np.unique(folds):
                te = folds == f
                offset[te] = _RidgeFit(R[~te], y[~te], a_c).predict(R[te])
            resid = y - offset
            self.v06_info["offset"] = ("cross-fitted comparison prediction (5 grouped folds); block columns "
                                       "residualised on the comparison columns, cross-fitted")
            fits = {}
            for w, Bk in blocks.items():
                if Bk.shape[1] == 0:  # one variable: the competitor is the comparison model alone
                    fits[w] = _OffsetFit(self.comparison, _ZeroFit(), self._r_width)
                    self.v06_info[f"alpha_block_{w}"] = float("inf")
                    continue
                # Frisch-Waugh-Lovell with ridge: residualise the block columns on the
                # comparison columns (same penalty, cross-fitted like the offset), so the
                # block is credited only with what the comparison model cannot express.
                Bres = np.empty_like(Bk)
                for f in np.unique(folds):
                    te = folds == f
                    Bres[te] = Bk[te] - _MultiRidge(R[~te], Bk[~te], a_c).predict(R[te])
                projector = _MultiRidge(R, Bk, a_c)
                a = _cv_alpha(Bres, resid, groups, grid, seed) if cv_alpha else ridge_alpha
                fits[w] = _OffsetFit(self.comparison, _RidgeFit(Bres, resid, a), self._r_width, projector)
                self.v06_info[f"alpha_block_{w}"] = float(a)
        else:
            fits = {}
            for w, Bk in blocks.items():
                if Bk.shape[1] == 0:
                    fits[w] = _OffsetFit(self.comparison, _ZeroFit(), self._r_width)
                    self.v06_info[f"alpha_pair_{w}"] = [float(a_c), "inf"]
                    continue
                D = np.column_stack((R, Bk))
                pair = _cv_two_penalty(D, self._r_width, y, groups, seed) if cv_alpha else (ridge_alpha, ridge_alpha)
                fits[w] = _TwoPenaltyFit(D, y, self._r_width, *pair)
                self.v06_info[f"alpha_pair_{w}"] = [float(a) if np.isfinite(a) else "inf" for a in pair]
            # A block chosen at the zero model contributes nothing, so its model IS its
            # comparison: a block-zeroed competitor / linear model predicts as the comparison
            # model alone, a block-zeroed full model predicts as the competitor (gain exactly 0;
            # never a gain from the comparison penalty the pair search shifted).
            zeroed = [w for w in ("simple", "linear", "full")
                      if blocks[w].shape[1] > 0 and self.v06_info[f"alpha_pair_{w}"][1] == "inf"]
            for w in zeroed:
                if w != "full":
                    fits[w] = _OffsetFit(self.comparison, _ZeroFit(), self._r_width)
            self._full_from_simple = "full" in zeroed
            if zeroed:
                self.v06_info["zero_block_models"] = zeroed
        self.simple, self.full, self.linear_fit = fits["simple"], fits["full"], fits["linear"]
        if self.relation == "v06_offset":
            self.alpha_simple = self.v06_info["alpha_block_simple"]
            self.alpha_full = self.v06_info["alpha_block_full"]
        else:
            ps, pf = self.v06_info["alpha_pair_simple"], self.v06_info["alpha_pair_full"]
            self.alpha_simple = float(ps[1]) if ps[1] != "inf" else float("inf")
            self.alpha_full = float(pf[1]) if pf[1] != "inf" else float("inf")

    # ---- v0.7: general gain + proper-subset decompositions (G_HO) ----
    def _subset_feature(self, X: Array, subset: tuple[int, ...]) -> Array:
        """The candidate feature restricted to a subset of its variables (no context factor)."""
        values = X[:, [self.candidate.variables[i] for i in subset]]
        if self.candidate.feature_form == "linear_subspace":
            w = (np.asarray([self.candidate.weights[i] for i in subset], dtype=float)
                 if self.candidate.weights else np.ones(len(subset)))
            return values @ w
        return np.prod(values, axis=1)

    def _subset_seed(self, subset: tuple[int, ...]) -> int:
        if len(subset) == 1:  # the v06 per-constituent map
            return int(self.candidate.candidate_hash[:8], 16) + 1 + subset[0]
        return int(sha256(f"{self.candidate.candidate_hash}:{subset}".encode()).hexdigest()[:8], 16)

    def _make_decomposition_maps(self, X: Array) -> None:
        values = X[:, self.candidate.variables]
        k = values.shape[1]
        self.decompositions = decomposition_names(k)
        self._subset_maps: dict[tuple[int, ...], _RBFMap | None] = {}
        for _, part in self.decompositions:
            for s in part:
                if s in self._subset_maps:
                    continue
                if self.model != "rbf":
                    self._subset_maps[s] = None
                elif len(s) == 1 and self.additive_maps is not None:
                    self._subset_maps[s] = self.additive_maps[s[0]]
                else:
                    train = values[:, list(s)] if len(s) == 1 else np.column_stack(
                        (values[:, list(s)], self._subset_feature(X, s).reshape(-1, 1)))
                    self._subset_maps[s] = _RBFMap(train, n_features=self.n_rbf * len(s), seed=self._subset_seed(s))

    def _decomposition_block(self, X: Array, part: list[tuple[int, ...]]) -> Array:
        """Concatenated joint maps of the partition's subsets + every variable linearly
        (``model="ridge"``: the variables + each multi-variable subset's feature, linearly)."""
        values = X[:, self.candidate.variables]
        pieces = []
        for s in part:
            m = self._subset_maps[s]
            if m is None:
                if len(s) >= 2:
                    pieces.append(self._subset_feature(X, s).reshape(-1, 1))
                continue
            inp = values[:, list(s)] if len(s) == 1 else np.column_stack(
                (values[:, list(s)], self._subset_feature(X, s).reshape(-1, 1)))
            pieces.append(m.apply(inp))
        pieces.append(values)
        return np.column_stack(pieces)

    def _decomposition_blocks(self, X: Array, context: Array | None) -> dict[str, Array]:
        """``{"decomp:<name>": block}`` for every proper-subset decomposition."""
        del context  # context columns sit in the reference design, never in a block
        return {f"decomp:{name}": self._decomposition_block(X, part) for name, part in self.decompositions}

    def _v07_blocks(self, X: Array, context: Array | None) -> dict[str, Array]:
        blocks = {w: self._block(X, context, w) for w in ("simple", "linear", "full")}
        blocks.update(self._decomposition_blocks(X, context))
        return blocks

    def _init_v07(self, X, y, context, baseline, groups, grid, seed, cv_alpha, ridge_alpha) -> None:
        """v06 two-penalty fits of reference, linear, the v06 additive competitor
        (``simple``), the full model and every proper-subset decomposition."""
        self._make_decomposition_maps(X)
        R = _reference_design(X, self.candidate, context, baseline)
        self._r_width = R.shape[1]
        a_c = _cv_alpha(R, y, groups, grid, seed) if cv_alpha else ridge_alpha
        self.comparison = _RidgeFit(R, y, a_c)
        reference = _OffsetFit(self.comparison, _ZeroFit(), self._r_width)
        self.v06_info = {"alpha_comparison": float(a_c)}
        fits: dict[str, Any] = {}
        zeroed = []
        for w, Bk in self._v07_blocks(X, context).items():
            if Bk.shape[1] == 0:  # one variable: no additive competitor beyond the comparison model
                fits[w] = reference
                self.v06_info[f"alpha_pair_{w}"] = [float(a_c), "inf"]
                continue
            D = np.column_stack((R, Bk))
            pair = _cv_two_penalty(D, self._r_width, y, groups, seed) if cv_alpha else (ridge_alpha, ridge_alpha)
            self.v06_info[f"alpha_pair_{w}"] = [float(a) if np.isfinite(a) else "inf" for a in pair]
            if not np.isfinite(pair[1]):
                # A block at the zero model contributes nothing: its model IS the comparison
                # model (never a gain from the comparison penalty the pair search shifted).
                zeroed.append(w)
                fits[w] = reference
            else:
                fits[w] = _TwoPenaltyFit(D, y, self._r_width, *pair)
        if zeroed:
            self.v06_info["zero_block_models"] = zeroed
        self._full_from_simple = "full" in zeroed  # general gain exactly 0
        self.simple, self.full, self.linear_fit = reference, fits["full"], fits["linear"]
        self.additive_fit = fits["simple"]
        self.decomposition_fits = {name: fits[f"decomp:{name}"] for name, _ in self.decompositions}
        self.alpha_simple = float("inf")
        pf = self.v06_info["alpha_pair_full"]
        self.alpha_full = float(pf[1]) if pf[1] != "inf" else float("inf")

    def ho_predictions(self, X: Array, *, context: Array | None, baseline: Array | None) -> dict[str, Array]:
        """v07_ho: every model's predictions on any rows (no refit).  ``full_v06`` is the
        v06 full prediction (the additive competitor's where the full block is zeroed)."""
        R = _reference_design(X, self.candidate, context, baseline)
        blocks = self._v07_blocks(X, context)
        design = {w: np.column_stack((R, Bk)) for w, Bk in blocks.items()}
        ref = self.comparison.predict(R)
        out = {"reference": ref, "linear": self.linear_fit.predict(design["linear"]),
               "additive": self.additive_fit.predict(design["simple"])}
        full = None if self._full_from_simple else self.full.predict(design["full"])
        out["full"] = ref if full is None else full
        out["full_v06"] = out["additive"] if full is None else full
        for name, _ in self.decompositions:
            out[f"decomp:{name}"] = self.decomposition_fits[name].predict(design[f"decomp:{name}"])
        return out

    def _v06_design(self, X: Array, context: Array | None, baseline: Array | None, which: str) -> Array:
        return np.column_stack((_reference_design(X, self.candidate, context, baseline),
                                self._block(X, context, which)))

    def _simple_design(self, X: Array, context: Array | None, baseline: Array | None) -> Array:
        if self.relation in HO_RELATIONS:  # the general gain's competitor: the comparison model alone
            return _reference_design(X, self.candidate, context, baseline)
        if self.relation in V06_RELATIONS:
            return self._v06_design(X, context, baseline, "simple")
        if self.reference_competitor:
            return _reference_design(X, self.candidate, context, baseline)
        values = X[:, self.candidate.variables]
        core = (np.column_stack([m.apply(values[:, [j]]) for j, m in enumerate(self.additive_maps)])
                if self.additive_maps is not None else values)
        pieces = [core, _extra_linear(X, self.candidate, context), baseline]
        return np.column_stack([p for p in pieces if p is not None])

    def _full_design(self, X: Array, context: Array | None, baseline: Array | None) -> Array:
        if self.relation in V06_RELATIONS or self.relation in HO_RELATIONS:
            return self._v06_design(X, context, baseline, "full")
        values = X[:, self.candidate.variables]
        joint = np.column_stack((values, self.candidate.feature(X, context).reshape(-1, 1)))
        core = self.joint_map.apply(joint) if self.joint_map is not None else joint
        pieces = [core, _extra_linear(X, self.candidate, context), baseline]
        return np.column_stack([p for p in pieces if p is not None])

    def predictions(self, X: Array, *, context: Array | None, baseline: Array | None) -> tuple[Array, Array]:
        """Competitor and full predictions for a replication-side design (no refit)."""
        simple = self.simple.predict(self._simple_design(X, context, baseline))
        if getattr(self, "_full_from_simple", False):  # v06 full block at the zero model
            return simple, simple
        return simple, self.full.predict(self._full_design(X, context, baseline))

    def gain(self, X: Array, y: Array, *, context: Array | None, baseline: Array | None) -> tuple[float, float, float]:
        simple_pred, full_pred = self.predictions(X, context=context, baseline=baseline)
        simple_r2, full_r2 = _r2(y, simple_pred), _r2(y, full_pred)
        return full_r2 - simple_r2, simple_r2, full_r2


def _fit_gain(
    candidate: Candidate, X_train: Array, y_train: Array, X_test: Array, y_test: Array,
    *, context_train: Array | None, context_test: Array | None, model: str, ridge_alpha: float,
    rbf_features: int, baseline_train: Array | None = None, baseline_test: Array | None = None,
) -> tuple[float, float, float]:
    fitted = FittedRelation(candidate, X_train, y_train, context=context_train, baseline=baseline_train,
                            model=model, ridge_alpha=ridge_alpha, rbf_features=rbf_features)
    return fitted.gain(X_test, y_test, context=context_test, baseline=baseline_test)


def evaluate_frozen_candidate(
    candidate: Candidate, proposal_X: Any, proposal_y: Any, replication_X: Any, replication_y: Any,
    *, proposal_context: Any | None = None, replication_context: Any | None = None,
    proposal_baseline: Any | None = None, replication_baseline: Any | None = None,
    model: str = "ridge", ridge_alpha: float = 1.0, rbf_features: int = 24,
    proposal_groups: Any | None = None, cv_alpha: bool = True,
    alpha_grid: tuple[float, ...] | None = None, relation: str = "v04",
    replication_groups: Any | None = None, n_boot: int = 200,
) -> DiscoveryReceipt:
    """Fit only on the proposal split and report independent incremental gain.

    `proposal_groups` (e.g. patient codes of the proposal rows) makes the ridge
    strength selection grouped, so no patient leaks between CV folds.
    `alpha_grid` replaces the default grid for both models (each chosen
    independently); the chosen strengths are recorded on the receipt.
    `relation="v05"` (see :data:`RELATIONS`) records the linear decomposition
    in `receipt.decomposition`; with `replication_groups` it carries
    held-out-patient bootstrap intervals (`n_boot` draws)."""
    pX, py = _array(proposal_X, ndim=2, name="proposal_X"), _array(proposal_y, ndim=1, name="proposal_y")
    rX, ry = _array(replication_X, ndim=2, name="replication_X"), _array(replication_y, ndim=1, name="replication_y")
    if len(pX) != len(py) or len(rX) != len(ry) or pX.shape[1] != rX.shape[1]:
        raise ValueError("proposal/replication dimensions are incompatible")
    pc = None if proposal_context is None else _array(proposal_context, ndim=2, name="proposal_context")
    rc = None if replication_context is None else _array(replication_context, ndim=2, name="replication_context")
    if (pc is None) != (rc is None) or (pc is not None and (len(pc) != len(pX) or len(rc) != len(rX))):
        raise ValueError("proposal and replication context must be supplied together and align")
    pb = None if proposal_baseline is None else _array(proposal_baseline, ndim=2, name="proposal_baseline")
    rb = None if replication_baseline is None else _array(replication_baseline, ndim=2, name="replication_baseline")
    if (pb is None) != (rb is None) or (pb is not None and (len(pb) != len(pX) or len(rb) != len(rX))):
        raise ValueError("proposal and replication baseline must be supplied together and align")
    pg = None if proposal_groups is None else np.asarray(proposal_groups)
    fitted = FittedRelation(candidate, pX, py, context=pc, baseline=pb, model=model,
                            ridge_alpha=ridge_alpha, rbf_features=rbf_features, groups=pg, cv_alpha=cv_alpha,
                            alpha_grid=alpha_grid, relation=relation)
    proposal_gain, proposal_simple, proposal_full = fitted.gain(pX, py, context=pc, baseline=pb)
    replication_gain, replication_simple, replication_full = fitted.gain(rX, ry, context=rc, baseline=rb)
    decomposition = None
    if relation == "v05":
        rg = None if replication_groups is None else np.asarray(replication_groups)
        if rg is not None and len(rg) != len(rX):
            raise ValueError("replication_groups must align with replication rows")
        decomposition = _linear_decomposition(fitted, pX, py, rX, ry, pc, rc, pb, rb, pg, alpha_grid,
                                              cv_alpha, ridge_alpha, rg, int(n_boot))
    elif relation in V06_RELATIONS:
        rg = None if replication_groups is None else np.asarray(replication_groups)
        if rg is not None and len(rg) != len(rX):
            raise ValueError("replication_groups must align with replication rows")
        decomposition = _linear_decomposition_v06(fitted, pX, py, rX, ry, pc, rc, pb, rb, rg, int(n_boot))
    elif relation in HO_RELATIONS:
        rg = None if replication_groups is None else np.asarray(replication_groups)
        if rg is not None and len(rg) != len(rX):
            raise ValueError("replication_groups must align with replication rows")
        decomposition = _ho_decomposition(fitted, pX, py, rX, ry, pc, rc, pb, rb, rg, int(n_boot))
    receipt = DiscoveryReceipt(candidate.candidate_hash, candidate, proposal_gain, replication_gain,
                               proposal_simple, proposal_full, replication_simple, replication_full,
                               len(pX), len(rX), model,
                               baseline_width=0 if pb is None else int(pb.shape[1]),
                               alpha_simple=float(fitted.alpha_simple), alpha_full=float(fitted.alpha_full),
                               relation=relation, decomposition=decomposition)
    _FITTED_CACHE[_fit_key(candidate.candidate_hash, pX, py, pc, pb)] = fitted
    return receipt


def _patient_bootstrap(y: Array, predictions: Mapping[str, Array], contrasts: Mapping[str, tuple[str, str]],
                       groups: Array, n_boot: int, seed: int) -> dict[str, list[float]]:
    """2.5/97.5 percentile intervals of held-out R^2 differences, resampling
    held-out groups (patients) with replacement; fixed predictions and one
    shared set of draws for every contrast (the runner's ``_gain_interval``
    scheme)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    draws: dict[str, list[float]] = {key: [] for key in contrasts}
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        r2 = {name: _r2(y[rows], pred[rows]) for name, pred in predictions.items()}
        for key, (full, simple) in contrasts.items():
            draws[key].append(r2[full] - r2[simple])
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in draws.items()}


def _linear_decomposition(fitted: "FittedRelation", pX: Array, py: Array, rX: Array, ry: Array,
                          pc: Array | None, rc: Array | None, pb: Array | None, rb: Array | None,
                          pg: Array | None, alpha_grid: tuple[float, ...] | None, cv_alpha: bool,
                          ridge_alpha: float, rg: Array | None, n_boot: int) -> dict:
    """v0.5 decomposition of a fitted relation's held-out gain (proposal fits only).

    Models (each picks its own ridge strength by the same grouped inner CV
    and seed as the relation): ``reference`` = context + baseline; ``linear``
    = reference + every constituent linearly; ``full`` = the relation's full
    model.  k = 1: ``gain_total = gain_linear + gain_nonlinear_beyond_linear``
    exactly (``linear`` is the v0.4 competitor, so the beyond-linear part
    equals the v0.4 gain).  k >= 2: ``gain_linear_additive`` (linear vs
    reference) is recorded beside the unchanged joint-structure gain."""
    candidate = fitted.candidate
    seed = int(candidate.candidate_hash[:8], 16)
    grid = tuple(alpha_grid) if alpha_grid is not None else FittedRelation.ALPHA_GRID
    k = len(candidate.variables)

    def fit(design_fn):
        train = design_fn(pX, candidate, pc, pb)
        alpha = _cv_alpha(train, py, pg, grid, seed) if cv_alpha else ridge_alpha
        model = _RidgeFit(train, py, alpha)
        return float(alpha), model.predict(design_fn(rX, candidate, rc, rb)), model.predict(train)

    if fitted.reference_competitor:
        a_ref = float(fitted.alpha_simple)
        ref_r = fitted.simple.predict(_reference_design(rX, candidate, rc, rb))
        ref_p = fitted.simple.predict(_reference_design(pX, candidate, pc, pb))
    else:
        a_ref, ref_r, ref_p = fit(_reference_design)
    a_lin, lin_r, lin_p = fit(_linear_design)
    _, full_r = fitted.predictions(rX, context=rc, baseline=rb)
    _, full_p = fitted.predictions(pX, context=pc, baseline=pb)
    r2 = {"reference": _r2(ry, ref_r), "linear": _r2(ry, lin_r), "full": _r2(ry, full_r)}
    r2_p = {"reference": _r2(py, ref_p), "linear": _r2(py, lin_p), "full": _r2(py, full_p)}
    out: dict[str, Any] = {"k": k, "alpha_reference": a_ref, "alpha_linear": a_lin,
                           "alpha_full": float(fitted.alpha_full),
                           "replication_r2": r2, "proposal_r2": r2_p}
    if k == 1:
        contrasts = {"gain_total": ("full", "reference"), "gain_linear": ("linear", "reference"),
                     "gain_nonlinear_beyond_linear": ("full", "linear")}
    else:
        contrasts = {"gain_linear_additive": ("linear", "reference")}
    for key, (a, b) in contrasts.items():
        out[key] = r2[a] - r2[b]
    if rg is not None:
        cis = _patient_bootstrap(ry, {"reference": ref_r, "linear": lin_r, "full": full_r}, contrasts,
                                 rg, n_boot, seed)
        for key, ci in cis.items():
            out[f"{key}_ci"] = ci
        if k == 1:
            lin_lo, non_lo = cis["gain_linear"][0], cis["gain_nonlinear_beyond_linear"][0]
            out["shape"] = ("linear_and_beyond_linear" if lin_lo > 0 and non_lo > 0 else
                            "linear_carries_it" if lin_lo > 0 else
                            "beyond_linear_only" if non_lo > 0 else "not_resolved")
    return out


def _linear_decomposition_v06(fitted: "FittedRelation", pX: Array, py: Array, rX: Array, ry: Array,
                              pc: Array | None, rc: Array | None, pb: Array | None, rb: Array | None,
                              rg: Array | None, n_boot: int) -> dict:
    """v0.6 decomposition: ``reference`` = the comparison model alone (its own
    penalty), ``linear`` = comparison + linear constituents, ``full`` = the
    relation's full model, each fit as the relation fits it (offset or two
    penalties).  Same fields as the v0.5 decomposition."""
    candidate = fitted.candidate
    seed = int(candidate.candidate_hash[:8], 16)
    k = len(candidate.variables)
    R_r = _reference_design(rX, candidate, rc, rb)
    R_p = _reference_design(pX, candidate, pc, pb)
    ref_r, ref_p = fitted.comparison.predict(R_r), fitted.comparison.predict(R_p)
    lin_r = fitted.linear_fit.predict(fitted._v06_design(rX, rc, rb, "linear"))
    lin_p = fitted.linear_fit.predict(fitted._v06_design(pX, pc, pb, "linear"))
    _, full_r = fitted.predictions(rX, context=rc, baseline=rb)
    _, full_p = fitted.predictions(pX, context=pc, baseline=pb)
    r2 = {"reference": _r2(ry, ref_r), "linear": _r2(ry, lin_r), "full": _r2(ry, full_r)}
    r2_p = {"reference": _r2(py, ref_p), "linear": _r2(py, lin_p), "full": _r2(py, full_p)}
    info = {key: value for key, value in fitted.v06_info.items()}
    out: dict[str, Any] = {"k": k, "relation": fitted.relation, **info,
                           "replication_r2": r2, "proposal_r2": r2_p}
    if k == 1:
        contrasts = {"gain_total": ("full", "reference"), "gain_linear": ("linear", "reference"),
                     "gain_nonlinear_beyond_linear": ("full", "linear")}
    else:
        contrasts = {"gain_linear_additive": ("linear", "reference")}
    for key, (a, b) in contrasts.items():
        out[key] = r2[a] - r2[b]
    if rg is not None:
        cis = _patient_bootstrap(ry, {"reference": ref_r, "linear": lin_r, "full": full_r}, contrasts,
                                 rg, n_boot, seed)
        for key, ci in cis.items():
            out[f"{key}_ci"] = ci
        if k == 1:
            lin_lo, non_lo = cis["gain_linear"][0], cis["gain_nonlinear_beyond_linear"][0]
            out["shape"] = ("linear_and_beyond_linear" if lin_lo > 0 and non_lo > 0 else
                            "linear_carries_it" if lin_lo > 0 else
                            "beyond_linear_only" if non_lo > 0 else "not_resolved")
    return out


def _patient_bootstrap_r2(y: Array, predictions: Mapping[str, Array], groups: Array, n_boot: int,
                          seed: int) -> dict[str, Array]:
    """Per-draw held-out R^2 of every prediction, resampling held-out groups with
    replacement (the same draws as :func:`_patient_bootstrap` for the same seed)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    draws: dict[str, list[float]] = {name: [] for name in predictions}
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        for name, pred in predictions.items():
            draws[name].append(_r2(y[rows], pred[rows]))
    return {name: np.asarray(v) for name, v in draws.items()}


def _ho_decomposition(fitted: "FittedRelation", pX: Array, py: Array, rX: Array, ry: Array,
                      pc: Array | None, rc: Array | None, pb: Array | None, rb: Array | None,
                      rg: Array | None, n_boot: int) -> dict:
    """v07_ho decomposition (proposal fits only; held-out R^2 on the replication rows).

    ``gain_general`` = full - reference (the receipt's ``replication_gain``);
    ``gain_linear_additive`` = linear - reference; ``gain_joint_vs_additive`` =
    the v06 gain (v06 full vs the v06 per-constituent-map competitor);
    ``G_HO`` = full - max over the proper-subset decompositions (k >= 2; None
    for one variable), its interval from the same held-out-patient draws with
    the maximum taken inside every draw; ``best_decomposition`` = the
    decomposition with the highest held-out R^2."""
    candidate = fitted.candidate
    seed = int(candidate.candidate_hash[:8], 16)
    k = len(candidate.variables)
    pr = fitted.ho_predictions(rX, context=rc, baseline=rb)
    pp = fitted.ho_predictions(pX, context=pc, baseline=pb)
    r2 = {name: _r2(ry, pred) for name, pred in pr.items()}
    r2_p = {name: _r2(py, pred) for name, pred in pp.items()}
    dec = [f"decomp:{name}" for name, _ in fitted.decompositions]
    out: dict[str, Any] = {"k": k, "relation": fitted.relation, **dict(fitted.v06_info),
                           "replication_r2": r2, "proposal_r2": r2_p,
                           "decompositions": [name for name, _ in fitted.decompositions]}
    out["gain_general"] = r2["full"] - r2["reference"]
    out["gain_linear_additive"] = r2["linear"] - r2["reference"]
    out["gain_joint_vs_additive"] = r2["full_v06"] - r2["additive"]
    if k == 1:
        out["gain_linear"] = out["gain_linear_additive"]
        out["gain_nonlinear_beyond_linear"] = r2["full"] - r2["linear"]
    if dec:
        best = max(dec, key=lambda d: (r2[d], d))
        out["best_decomposition"] = best.split(":", 1)[1]
        out["G_HO"] = r2["full"] - r2[best]
        out["G_HO_proposal"] = r2_p["full"] - max(r2_p[d] for d in dec)
    else:
        out["best_decomposition"], out["G_HO"], out["G_HO_proposal"] = None, None, None
    if rg is not None:
        draws = _patient_bootstrap_r2(ry, pr, rg, n_boot, seed)

        def interval(values: Array) -> list[float]:
            return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

        out["gain_general_ci"] = interval(draws["full"] - draws["reference"])
        out["gain_linear_additive_ci"] = interval(draws["linear"] - draws["reference"])
        out["gain_joint_vs_additive_ci"] = interval(draws["full_v06"] - draws["additive"])
        if k == 1:
            out["gain_linear_ci"] = out["gain_linear_additive_ci"]
            out["gain_nonlinear_beyond_linear_ci"] = interval(draws["full"] - draws["linear"])
        out["G_HO_ci"] = (interval(draws["full"] - np.max(np.vstack([draws[d] for d in dec]), axis=0))
                          if dec else None)
        out["n_boot"] = int(n_boot)
    return out


# The fitted relation behind a receipt, keyed by candidate and by the CONTENT
# of the arrays it was fitted on (proposal design, response, context, baseline), so a
# null evaluator can reuse it instead of refitting (bounded; cleared when it
# grows past a few hundred entries).  An `id()` key is not an identity once the
# array is freed (CPython recycles addresses), and a float32 caller gets a
# fresh float64 copy on every call, so only a content key is both correct and
# stable.  Digests are memoised per live array object (weak reference checked
# on every hit) so a caller reusing one matrix hashes it once.
_FITTED_CACHE: dict[tuple, FittedRelation] = {}
_DIGESTS: dict[int, tuple[Any, str]] = {}


def _digest(a: Array | None) -> str | None:
    if a is None:
        return None
    hit = _DIGESTS.get(id(a))
    if hit is not None and hit[0]() is a:
        return hit[1]
    h = sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).data)
    digest = h.hexdigest()
    if len(_DIGESTS) > 256:
        _DIGESTS.clear()
    _DIGESTS[id(a)] = (weakref.ref(a), digest)
    return digest


def _fit_key(candidate_hash: str, pX: Array, py: Array, pc: Array | None, pb: Array | None) -> tuple:
    return (candidate_hash, _digest(pX), _digest(py), _digest(pc), _digest(pb))


def evaluate_episode_candidate(
    candidate: Candidate, proposal_episode: Any, replication_episode: Any, **kwargs: Any,
) -> DiscoveryReceipt:
    """Episode-adapter convenience wrapper around :func:`evaluate_frozen_candidate`."""
    pX, py, pc = episode_arrays(proposal_episode)
    rX, ry, rc = episode_arrays(replication_episode)
    return evaluate_frozen_candidate(candidate, pX, py, rX, ry,
                                     proposal_context=pc, replication_context=rc, **kwargs)


NullGenerator = Callable[[int, Array, Array, Array | None], tuple[Array, Array, Array | None]]


def evaluate_structured_nulls(
    receipt: DiscoveryReceipt, proposal_X: Any, proposal_y: Any, *, proposal_context: Any | None = None,
    proposal_baseline: Any | None = None, replication_baseline: Any | None = None,
    null_generator: NullGenerator, n_null: int = 100, ridge_alpha: float = 1.0,
) -> DiscoveryReceipt:
    """Apply caller-defined structure-preserving alternatives to a frozen candidate.

    The hook is deliberately domain-owned: tissue callers can supply spatial
    shifts, graph callers degree-preserving rewires, and temporal callers
    lag-preserving alternatives without this core assuming their structure.
    A nuisance `baseline` (declared covariates the discovery must beat, not
    itself claim) is, by construction, held FIXED across every null draw --
    it is not part of the structure a null is meant to destroy -- so both
    `proposal_baseline` and `replication_baseline` are the same fixed
    matrices used to build `receipt` in the first place.
    """
    if n_null < 1:
        raise ValueError("n_null must be positive")
    pX, py = _array(proposal_X, ndim=2, name="proposal_X"), _array(proposal_y, ndim=1, name="proposal_y")
    pc = None if proposal_context is None else _array(proposal_context, ndim=2, name="proposal_context")
    pb = None if proposal_baseline is None else _array(proposal_baseline, ndim=2, name="proposal_baseline")
    rb = None if replication_baseline is None else _array(replication_baseline, ndim=2, name="replication_baseline")
    # The competitor and full models depend only on the proposal split, which no
    # null touches: fit once, then score every nulled replication set.
    fitted = _FITTED_CACHE.get(_fit_key(receipt.candidate.candidate_hash, pX, py, pc, pb))
    relation = getattr(receipt, "relation", "v04")
    if fitted is None or fitted.model != receipt.model or fitted.relation != relation:
        fixed = (None if receipt.alpha_simple is None or receipt.alpha_full is None
                 else (receipt.alpha_simple, receipt.alpha_full))
        fitted = FittedRelation(receipt.candidate, pX, py, context=pc, baseline=pb, model=receipt.model,
                                ridge_alpha=ridge_alpha, rbf_features=24, fixed_alphas=fixed,
                                relation=relation)
    if len(_FITTED_CACHE) > 512:
        _FITTED_CACHE.clear()
    gains: list[float] = []
    # Generators receive the proposal arrays read-only (they must not mutate
    # them); copying a 17k x 300 matrix per draw dominated the null cost.
    # A response-only null returns the very same design arrays every draw, so
    # the two prediction vectors are reused while the generator keeps handing
    # back the identical (X, context) objects and each draw costs two R^2
    # evaluations.  The reused entry holds strong references to those objects,
    # so an identity match cannot be a freed array whose address was recycled
    # by a later draw (a design-changing null always gets fresh predictions).
    last: tuple[Any, Any, tuple[Array, Array]] | None = None
    for index in range(n_null):
        nX, ny, nc = null_generator(index, pX, py, pc)
        if last is None or last[0] is not nX or last[1] is not nc:
            last = (nX, nc, fitted.predictions(nX, context=nc, baseline=rb))
        simple_pred, full_pred = last[2]
        gains.append(float(_r2(ny, full_pred) - _r2(ny, simple_pred)))
    observed = receipt.replication_gain
    p_value = (1.0 + sum(g >= observed for g in gains)) / (len(gains) + 1.0)
    return DiscoveryReceipt(**{**receipt.__dict__, "null_gains": tuple(gains), "null_p_value": float(p_value)})


def evaluate_response_permutation_nulls(
    receipt: DiscoveryReceipt, proposal_X: Any, proposal_y: Any,
    null_X: Any, null_y: Any, *, proposal_context: Any | None = None,
    null_context: Any | None = None, proposal_baseline: Any | None = None,
    null_baseline: Any | None = None, n_null: int = 99, seed: int = 0,
    ridge_alpha: float = 1.0, rbf_features: int = 24,
) -> DiscoveryReceipt:
    """Fast empirical nulls for a frozen candidate and a fixed null environment.

    Both simple and full models are fit exactly once on the proposal split.
    Their predictions on the predeclared null environment are then compared to
    independently permuted null responses. This is mathematically the same
    fitted-model comparison as repeatedly calling ``evaluate_frozen_candidate``
    when null features are fixed, but avoids thousands of redundant fits in a
    paper-scale packet panel. It is appropriate only for a declared response-
    permutation null; spatial/graph rewiring callers should retain the generic
    ``evaluate_structured_nulls`` hook.
    """
    if n_null < 1:
        raise ValueError("n_null must be positive")
    pX, py = _array(proposal_X, ndim=2, name="proposal_X"), _array(proposal_y, ndim=1, name="proposal_y")
    nX, ny = _array(null_X, ndim=2, name="null_X"), _array(null_y, ndim=1, name="null_y")
    pc = None if proposal_context is None else _array(proposal_context, ndim=2, name="proposal_context")
    nc = None if null_context is None else _array(null_context, ndim=2, name="null_context")
    if len(pX) != len(py) or len(nX) != len(ny) or pX.shape[1] != nX.shape[1]:
        raise ValueError("proposal/null dimensions are incompatible")
    if (pc is None) != (nc is None) or (pc is not None and (len(pc) != len(pX) or len(nc) != len(nX))):
        raise ValueError("proposal and null context must be supplied together and align")
    pb = None if proposal_baseline is None else _array(proposal_baseline, ndim=2, name="proposal_baseline")
    nb = None if null_baseline is None else _array(null_baseline, ndim=2, name="null_baseline")
    if (pb is None) != (nb is None) or (pb is not None and (len(pb) != len(pX) or len(nb) != len(nX))):
        raise ValueError("proposal and null baseline must be supplied together and align")
    simple_train = _simple_design(pX, receipt.candidate, pc, pb)
    simple_null = _simple_design(nX, receipt.candidate, nc, nb)
    feature_train = receipt.candidate.feature(pX, pc).reshape(-1, 1)
    feature_null = receipt.candidate.feature(nX, nc).reshape(-1, 1)
    if receipt.model == "ridge":
        full_train, full_null = (np.column_stack((simple_train, feature_train)),
                                 np.column_stack((simple_null, feature_null)))
    elif receipt.model == "rbf":
        full_train, full_null = _rbf_features(np.column_stack((simple_train, feature_train)),
                                              np.column_stack((simple_null, feature_null)),
                                              n_features=rbf_features,
                                              seed=int(receipt.candidate_hash[:8], 16))
    else:
        raise ValueError("receipt model must be ridge or rbf")
    simple_prediction = _ridge_fit_predict(simple_train, py, simple_null, ridge_alpha)
    full_prediction = _ridge_fit_predict(full_train, py, full_null, ridge_alpha)
    observed_gain = _r2(ny, full_prediction) - _r2(ny, simple_prediction)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_null):
        permuted = ny[rng.permutation(len(ny))]
        gains.append(float(_r2(permuted, full_prediction) - _r2(permuted, simple_prediction)))
    p_value = (1.0 + sum(gain >= observed_gain for gain in gains)) / (n_null + 1.0)
    return DiscoveryReceipt(**{**receipt.__dict__, "null_gains": tuple(gains), "null_p_value": float(p_value)})


# --------------------------------------------------------------------------
# E1: interpretability AFTER evidence (discoadd: "Interpretability comes
# after evidence, not before it"). `compress_relation` never runs before a
# candidate has already been frozen and evaluated -- it takes the same
# (X_p, y_p, X_r, y_r) split every other discovery-core function takes, and
# every compressed form is fit ONLY on the proposal split, exactly like the
# flexible relation itself.
# --------------------------------------------------------------------------

#: Names of every compression family `compress_relation` attempts, in the
#: order attempted (not the order returned -- `accepted` is complexity-sorted).
COMPRESSION_FORMS = ("isotonic_monotone", "threshold_step", "sparse_polynomial",
                    "logical_and", "piecewise_regime")


def _degree2_monomials(V: Array) -> tuple[Array, tuple[str, ...]]:
    """Degree<=2 monomials over already-selected columns: linear terms,
    squares, then pairwise products. Deterministic order and names."""
    n, k = V.shape
    columns = [V[:, i] for i in range(k)]
    names = ["x%d" % i for i in range(k)]
    for i in range(k):
        columns.append(V[:, i] ** 2)
        names.append("x%d^2" % i)
    for i in range(k):
        for j in range(i + 1, k):
            columns.append(V[:, i] * V[:, j])
            names.append("x%d*x%d" % (i, j))
    return np.column_stack(columns), tuple(names)


def compress_relation(
    candidate: Candidate, X_p: Any, y_p: Any, X_r: Any, y_r: Any, *,
    baseline: tuple[Any | None, Any | None] | None = None,
    tolerance: float = 0.1, complexity_budget: int = 12,
    context_p: Any | None = None, context_r: Any | None = None,
    ridge_alpha: float = 1.0, rbf_features: int = 24,
) -> dict:
    """Attempt bounded, capacity-controlled compressions of an already
    frozen, already validated candidate -- never before validation.

    Each compressed form is FIT ONLY on ``(X_p, y_p)`` (``baseline``, if
    given, is a ``(baseline_p, baseline_r)`` pair of fixed nuisance-covariate
    matrices, held fixed exactly like every other discovery-core function's
    ``proposal_baseline``/``replication_baseline``). Its held-out incremental
    gain is then measured on ``(X_r, y_r)`` against the SAME simpler-
    competitor design ``B`` every other function in this module uses
    (:func:`_simple_design`), so a compressed form's gain is directly
    comparable to the flexible (RBF) relation's own held-out gain,
    ``flexible_gain``, computed the identical way.

    A form is **accepted** only when both hold:

    * ``gain >= flexible_gain - tolerance`` (it does not give up more than
      ``tolerance`` of the flexible relation's own held-out incremental
      explanatory power), and
    * ``complexity <= complexity_budget`` (its parameter count is cheap
      enough to call "compressed").

    Families attempted (discoadd: "Interpretability comes after evidence"):

    ``isotonic_monotone``   a monotone step function of the candidate's own
                            feature value (``sklearn.isotonic.IsotonicRegression``).
    ``threshold_step``      a single split of the candidate's own feature
                            value into two levels, threshold chosen on
                            ``D_P`` only.
    ``sparse_polynomial``   an L1-selected degree<=2 polynomial over the
                            candidate's own variables.
    ``logical_and``         (product candidates only) a sign-pattern/AND
                            rule: each constituent binarized at its ``D_P``
                            median, combined by AND.
    ``piecewise_regime``    two linear regimes of the candidate's own
                            feature value, split on its leading variable.

    Returns ``{"compressed": bool, "reason": str | None, "flexible_gain":
    float, "tolerance": float, "complexity_budget": int, "forms": [...],
    "accepted": [...]}``. ``forms`` records every attempted family (even
    rejected ones) for auditability; ``accepted`` is the subset that passed,
    ranked simplest-first (``(complexity, -gain)``). When nothing passes,
    ``compressed`` is ``False`` and ``reason`` explains why -- a formula is
    never fabricated to fill this field.
    """
    Xp, yp = _array(X_p, ndim=2, name="X_p"), _array(y_p, ndim=1, name="y_p")
    Xr, yr = _array(X_r, ndim=2, name="X_r"), _array(y_r, ndim=1, name="y_r")
    if len(Xp) != len(yp) or len(Xr) != len(yr) or Xp.shape[1] != Xr.shape[1]:
        raise ValueError("proposal/replication dimensions are incompatible")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if complexity_budget < 1:
        raise ValueError("complexity_budget must be positive")
    bp, br = (None, None) if baseline is None else baseline
    bp = None if bp is None else _array(bp, ndim=2, name="baseline_p")
    br = None if br is None else _array(br, ndim=2, name="baseline_r")
    cp = None if context_p is None else _array(context_p, ndim=2, name="context_p")
    cr = None if context_r is None else _array(context_r, ndim=2, name="context_r")

    flexible_gain, _, _ = _fit_gain(
        candidate, Xp, yp, Xr, yr, context_train=cp, context_test=cr,
        model="rbf", ridge_alpha=ridge_alpha, rbf_features=rbf_features,
        baseline_train=bp, baseline_test=br,
    )

    def plugin_gain(extra_train: Array, extra_test: Array) -> float:
        """Replace the flexible relation's nonlinearity with a compressed
        feature block, keeping B and the fitting procedure identical."""
        extra_train = np.atleast_2d(np.asarray(extra_train, dtype=float).reshape(len(Xp), -1))
        extra_test = np.atleast_2d(np.asarray(extra_test, dtype=float).reshape(len(Xr), -1))
        simple_train = _simple_design(Xp, candidate, cp, bp)
        simple_test = _simple_design(Xr, candidate, cr, br)
        full_train = np.column_stack((simple_train, extra_train))
        full_test = np.column_stack((simple_test, extra_test))
        simple_pred = _ridge_fit_predict(simple_train, yp, simple_test, ridge_alpha)
        full_pred = _ridge_fit_predict(full_train, yp, full_test, ridge_alpha)
        return _r2(yr, full_pred) - _r2(yr, simple_pred)

    base_feature_p = candidate.feature(Xp, cp)
    base_feature_r = candidate.feature(Xr, cr)
    values_p = Xp[:, candidate.variables]
    values_r = Xr[:, candidate.variables]

    forms: list[dict] = []

    # 1. Monotone compression: isotonic regression of y on the candidate's
    # own already-validated feature value.
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(base_feature_p, yp)
        pred_p = iso.predict(base_feature_p)
        pred_r = iso.predict(base_feature_r)
        gain = plugin_gain(pred_p, pred_r)
        complexity = int(len(np.unique(np.round(pred_p, 8))))
        forms.append({"form": "isotonic_monotone", "gain": float(gain),
                     "complexity": complexity, "detail": {}})
    except Exception as exc:  # pragma: no cover - defensive only
        forms.append({"form": "isotonic_monotone", "gain": float("-inf"),
                     "complexity": 10 ** 9, "detail": {"error": str(exc)}})

    # 2. Single-threshold step on the candidate's own feature value,
    # threshold chosen on D_P only (least training SSE).
    quantiles = np.unique(np.quantile(base_feature_p, np.linspace(0.05, 0.95, 19)))
    best_threshold, best_sse = None, np.inf
    for t in quantiles:
        indicator = (base_feature_p > t).astype(float)
        if indicator.std() <= 1e-12:
            continue
        design = np.column_stack((np.ones_like(indicator), indicator))
        beta, *_ = np.linalg.lstsq(design, yp, rcond=None)
        resid = yp - design @ beta
        sse = float(np.dot(resid, resid))
        if sse < best_sse:
            best_sse, best_threshold = sse, float(t)
    if best_threshold is not None:
        ind_p = (base_feature_p > best_threshold).astype(float)
        ind_r = (base_feature_r > best_threshold).astype(float)
        gain = plugin_gain(ind_p, ind_r)
        forms.append({"form": "threshold_step", "gain": float(gain),
                     "complexity": 2, "detail": {"threshold": best_threshold}})

    # 3. Sparse degree<=2 polynomial interaction over the candidate's own
    # variables, L1-selected on D_P.
    try:
        from sklearn.linear_model import Lasso
        monomials_p, names = _degree2_monomials(values_p)
        monomials_r, _ = _degree2_monomials(values_r)
        mean = monomials_p.mean(axis=0)
        scale = monomials_p.std(axis=0)
        scale[scale < 1e-12] = 1.0
        Mp, Mr = (monomials_p - mean) / scale, (monomials_r - mean) / scale
        best_poly = None
        for alpha in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(Mp, yp)
            nz = np.flatnonzero(lasso.coef_)
            if nz.size == 0:
                continue
            complexity = int(nz.size) + 1
            gain = plugin_gain(lasso.predict(Mp), lasso.predict(Mr))
            key = (float(gain), -complexity)
            if best_poly is None or key > best_poly["_key"]:
                best_poly = {"form": "sparse_polynomial", "gain": float(gain),
                             "complexity": complexity,
                             "detail": {"alpha": alpha,
                                       "monomials": [names[i] for i in nz]},
                             "_key": key}
        if best_poly is not None:
            best_poly.pop("_key")
            forms.append(best_poly)
    except Exception as exc:  # pragma: no cover - defensive only
        forms.append({"form": "sparse_polynomial", "gain": float("-inf"),
                     "complexity": 10 ** 9, "detail": {"error": str(exc)}})

    # 4. Logical description (sign-pattern/AND rule), product candidates only.
    if candidate.feature_form == "product":
        medians = np.median(values_p, axis=0)
        above_p = (values_p > medians).astype(float)
        above_r = (values_r > medians).astype(float)
        and_p = np.prod(above_p, axis=1)
        and_r = np.prod(above_r, axis=1)
        if and_p.std() > 1e-12:
            gain = plugin_gain(and_p, and_r)
            forms.append({"form": "logical_and", "gain": float(gain),
                         "complexity": int(len(candidate.variables)) + 1,
                         "detail": {"rule": "AND(variable_i > median_i)",
                                   "medians": [float(m) for m in medians]}})

    # 5. Piecewise regime (2 regimes) on the candidate's leading variable.
    regime_variable = candidate.variables[0]
    regime_p = Xp[:, regime_variable]
    regime_r = Xr[:, regime_variable]
    best_regime_threshold, best_regime_sse = None, np.inf
    for t in np.unique(np.quantile(regime_p, np.linspace(0.2, 0.8, 13))):
        mask_low = regime_p <= t
        if mask_low.sum() < 5 or (~mask_low).sum() < 5:
            continue
        f_low = np.where(mask_low, base_feature_p, 0.0)
        f_high = np.where(mask_low, 0.0, base_feature_p)
        design = np.column_stack((np.ones_like(base_feature_p), f_low, f_high))
        beta, *_ = np.linalg.lstsq(design, yp, rcond=None)
        resid = yp - design @ beta
        sse = float(np.dot(resid, resid))
        if sse < best_regime_sse:
            best_regime_sse, best_regime_threshold = sse, float(t)
    if best_regime_threshold is not None:
        mask_low_p = regime_p <= best_regime_threshold
        mask_low_r = regime_r <= best_regime_threshold
        extra_p = np.column_stack((np.where(mask_low_p, base_feature_p, 0.0),
                                   np.where(mask_low_p, 0.0, base_feature_p)))
        extra_r = np.column_stack((np.where(mask_low_r, base_feature_r, 0.0),
                                   np.where(mask_low_r, 0.0, base_feature_r)))
        gain = plugin_gain(extra_p, extra_r)
        forms.append({"form": "piecewise_regime", "gain": float(gain),
                     "complexity": 3,
                     "detail": {"regime_variable": int(regime_variable),
                               "threshold": best_regime_threshold}})

    for f in forms:
        f["accepted"] = bool(f["gain"] >= flexible_gain - tolerance
                             and f["complexity"] <= complexity_budget)

    accepted = sorted((f for f in forms if f["accepted"]),
                      key=lambda f: (f["complexity"], -f["gain"]))
    return {
        "compressed": bool(accepted), "flexible_gain": float(flexible_gain),
        "tolerance": float(tolerance), "complexity_budget": int(complexity_budget),
        "forms": forms, "accepted": accepted,
        "reason": None if accepted else
                 "no compression form retained held-out gain within tolerance "
                 "of the flexible relation's gain within the complexity budget",
    }


# --------------------------------------------------------------------------
# Explanation decomposition (ideas.md: "A simple relationship does not fail
# because a simple comparator explains it"). Classification only: nothing
# here changes an acceptance rule; it reports HOW a reproducible relation
# explains its target.
# --------------------------------------------------------------------------

#: Levels A-D are statistical (set by `decompose_explanation`); E and F are
#: evidence levels set only by external evidence flags, never by statistics.
EXPLANATION_LEVELS = {
    "A": "simple: a robust single-variable or simple relation",
    "B": "additive synthesis: the additive combination of constituents beats every constituent alone",
    "C": "non-additive interaction: the joint relation beats the best additive synthesis",
    "D": "context-dependent: the relation changes across the context variable's range",
    "E": "mechanistic/perturbational evidence (set by evidence flags only)",
    "F": "translational/clinical consequence (set by evidence flags only)",
}
_STRUCTURAL_ORDER = ("A", "B", "C", "D")


def _as_index(rows: Any, n: int, name: str) -> Array:
    rows = np.asarray(rows)
    if rows.dtype == bool:
        if rows.shape != (n,):
            raise ValueError("%s mask must have length %d" % (name, n))
        return np.flatnonzero(rows)
    rows = rows.astype(np.int64).reshape(-1)
    if rows.size and (rows.min() < 0 or rows.max() >= n):
        raise ValueError("%s index out of range" % name)
    return rows


def _interval(values: Array) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def classify_explanation_levels(point: Mapping[str, Any], intervals: Mapping[str, Any], Q: Mapping[str, float],
                                *, k: int, context_tested: bool, min_delta: float = 0.005
                                ) -> tuple[bool, list[str], str | None]:
    """The statistical level rule of `decompose_explanation`, exposed so a
    stored decomposition can be re-read under exactly the same rule.

    ``supported``: the lower interval bound of the relation's held-out gain
    over the baseline (or, with a context, over a context-only model) is
    positive AND the best relation model itself beats the held-out mean
    (``Q > 0``; the same transfer condition the campaign's validation rule
    uses -- a gain between two negative R^2 values is a less-bad transfer,
    not a held-out explanation). Then A; B/C/D when the lower bounds of
    ``delta_add`` / ``delta_nonadd`` / ``delta_context`` exceed ``min_delta``."""
    def lower(key: str) -> float:
        if key not in point or point.get(key) is None:
            return float("nan")
        return (intervals.get(key) or [point[key]])[0]

    best_q = max(Q.get("joint", -np.inf), Q.get("additive", -np.inf))
    within = lower("gain_relation_within_context") > 0 and max(
        Q.get("relation_x_context", -np.inf), Q.get("relation_plus_context", -np.inf)) > 0
    supported = bool((lower("gain_relation") > 0 and best_q > 0) or (context_tested and within))
    statistical: list[str] = []
    if supported:
        statistical.append("A")
        if k >= 2 and lower("delta_add") > min_delta:
            statistical.append("B")
        if k >= 2 and lower("delta_nonadd") > min_delta:
            statistical.append("C")
        if context_tested and lower("delta_context") > min_delta:
            statistical.append("D")
    level = max(statistical, key=_STRUCTURAL_ORDER.index) if statistical else None
    return supported, statistical, level


def decompose_explanation(
    candidate: Candidate, X: Any, y: Any, *, groups: Any, proposal: Any, replication: Any,
    context: Any | None = None, baseline: Any | None = None, candidate_context: Any | None = None,
    rbf_features: int = 24, n_boot: int = 200, min_delta: float = 0.005,
    n_context_regimes: int = 3, evidence_flags: Mapping[str, Any] | None = None, seed: int = 0,
) -> dict:
    """Classify how an already validated relation explains its target.

    Every model is fit ONLY on the ``proposal`` rows (ridge strength by
    patient-grouped CV over ``groups``) and scored by held-out R^2 (``Q``)
    on the ``replication`` rows; the declared nuisance ``baseline`` enters
    every model linearly, so deltas are free of nuisance structure.

    * ``Q(X_j)`` -- each constituent alone through its own flexible map
      (``rbf_features`` random Fourier features), the same per-constituent
      maps `FittedRelation` uses for its additive competitor;
    * ``Q(additive)`` -- the sum of those per-constituent maps;
    * ``Q(joint)`` -- one joint map over the constituents and the candidate
      feature, capacity matched (``k * rbf_features`` features).

    ``delta_add = Q(additive) - max_j Q(X_j)`` and
    ``delta_nonadd = Q(joint) - Q(additive)``, with percentile intervals from
    a bootstrap that resamples held-out GROUPS (patients) with the fits held
    fixed. When ``context`` (one variable, length n) is given, the context
    test compares a joint map over (constituents, feature, context) with the
    relation map plus a separate context map (capacity matched):
    ``delta_context = Q(relation x context) - Q(relation + context)``, and
    reports the held-out effect of the relation inside each proposal-defined
    context regime (quantile bins).

    Statistical levels (`classify_explanation_levels`): the relation is
    ``supported`` when the lower bound of its held-out gain over the baseline
    (or, for a relation that exists only within a context, over a
    context-only model) is positive and the relation model beats the
    held-out mean (Q > 0). Then A always;
    B if ``k >= 2`` and the lower bound of ``delta_add`` exceeds
    ``min_delta``; C likewise for ``delta_nonadd``; D likewise for
    ``delta_context``. Levels E and F are added ONLY from
    ``evidence_flags={"E": {"supported": bool, "basis": str}, "F": {...}}``
    supplied by the translational evidence branch -- never from these
    statistics.
    """
    Xa, ya = _array(X, ndim=2, name="X"), _array(y, ndim=1, name="y")
    n = len(ya)
    if len(Xa) != n:
        raise ValueError("X and y length mismatch")
    g = np.asarray(groups)
    if len(g) != n:
        raise ValueError("groups must align with y")
    p_idx, r_idx = _as_index(proposal, n, "proposal"), _as_index(replication, n, "replication")
    if not len(p_idx) or not len(r_idx):
        raise ValueError("proposal and replication must be nonempty")
    if set(g[p_idx].tolist()) & set(g[r_idx].tolist()):
        raise ValueError("proposal and replication share a group; the decomposition must be group-disjoint")
    B = None if baseline is None else _array(baseline, ndim=2, name="baseline")
    if B is not None and len(B) != n:
        raise ValueError("baseline must align with y")
    cc = None if candidate_context is None else _array(candidate_context, ndim=2, name="candidate_context")
    c = None if context is None else np.asarray(context, dtype=float).reshape(n, -1)
    if c is not None and c.shape[1] != 1:
        raise ValueError("context must be a single variable")

    k = len(candidate.variables)
    values = Xa[:, candidate.variables]
    feature = candidate.feature(Xa, cc).reshape(-1, 1)
    yp, yr = ya[p_idx], ya[r_idx]
    gp, gr = g[p_idx], g[r_idx]
    base_seed = int(candidate.candidate_hash[:8], 16)

    def _fit_predict(blocks: list[Array]) -> Array:
        pieces = list(blocks) + ([B] if B is not None else [])
        if not pieces:
            return np.full(len(r_idx), float(yp.mean()))
        design = np.column_stack(pieces)
        alpha = _cv_alpha(design[p_idx], yp, gp, FittedRelation.ALPHA_GRID, base_seed)
        return _RidgeFit(design[p_idx], yp, alpha).predict(design[r_idx])

    def _map(columns: Array, width: int, offset: int) -> Array:
        return _RBFMap(columns[p_idx], n_features=width, seed=base_seed + offset).apply(columns)

    single_blocks = [_map(values[:, [j]], rbf_features, 1 + j) for j in range(k)]
    joint_block = _map(np.column_stack((values, feature)), rbf_features * max(k, 1), 0)
    predictions: dict[str, Array] = {"baseline": _fit_predict([])}
    for j in range(k):
        predictions["single_%d" % j] = _fit_predict([single_blocks[j]])
    predictions["additive"] = _fit_predict(single_blocks) if k >= 2 else predictions["single_0"]
    predictions["joint"] = _fit_predict([joint_block])
    regimes: list[dict] = []
    if c is not None:
        c_block = _map(c, rbf_features, 101)
        cross_block = _map(np.column_stack((values, feature, c)), rbf_features * (max(k, 1) + 1), 202)
        predictions["context_only"] = _fit_predict([c_block])
        predictions["relation_plus_context"] = _fit_predict([joint_block, c_block])
        predictions["relation_x_context"] = _fit_predict([cross_block])
        cuts = np.unique(np.quantile(c[p_idx, 0], np.linspace(0, 1, n_context_regimes + 1)[1:-1]))
        regime_r = np.searchsorted(cuts, c[r_idx, 0], side="right")
        regime_p = np.searchsorted(cuts, c[p_idx, 0], side="right")
        base_r = predictions["baseline"]
        for level in range(len(cuts) + 1):
            mask = regime_r == level
            entry = {"regime": int(level), "n_heldout": int(mask.sum()), "n_proposal": int((regime_p == level).sum()),
                     "context_range_heldout": ([float(c[r_idx, 0][mask].min()), float(c[r_idx, 0][mask].max())]
                                               if mask.any() else None)}
            if mask.sum() >= 10:
                # Held-out gain of the (global) joint relation over the baseline
                # inside this regime, and the regime-local slope of the
                # baseline-adjusted response on the relation's contribution.
                contribution = predictions["joint"][mask] - base_r[mask]
                resid = yr[mask] - base_r[mask]
                entry["gain_over_baseline"] = _r2(yr[mask], predictions["joint"][mask]) - _r2(yr[mask], base_r[mask])
                var = float(np.var(contribution))
                entry["slope_on_relation"] = (float(np.cov(contribution, resid, bias=True)[0, 1] / var)
                                             if var > 1e-15 else None)
            regimes.append(entry)

    names = list(predictions)
    single_names = ["single_%d" % j for j in range(k)]

    def _summaries(rows: Array) -> dict[str, float]:
        q = {m: _r2(yr[rows], predictions[m][rows]) for m in names}
        out = {"Q_" + m: q[m] for m in names}
        best_single = max(q[s] for s in single_names)
        out["delta_add"] = (q["additive"] - best_single) if k >= 2 else 0.0
        out["delta_nonadd"] = q["joint"] - q["additive"]
        out["gain_relation"] = max(q["joint"], q["additive"]) - q["baseline"]
        if c is not None:
            out["delta_context"] = q["relation_x_context"] - q["relation_plus_context"]
            out["gain_relation_within_context"] = (max(q["relation_x_context"], q["relation_plus_context"])
                                                   - q["context_only"])
        return out

    point = _summaries(np.arange(len(r_idx)))
    uniq = np.unique(gr)
    rows_by_group = {u: np.flatnonzero(gr == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_group[u] for u in pick])
        for key, value in _summaries(rows).items():
            draws[key].append(value)
    ci = {key: _interval(np.asarray(vals)) for key, vals in draws.items()} if n_boot else {}

    q_point = {m: point["Q_" + m] for m in names}
    supported, statistical, statistical_level = classify_explanation_levels(
        point, ci, q_point, k=k, context_tested=c is not None, min_delta=min_delta)
    flags = {key: dict(value) for key, value in (evidence_flags or {}).items() if key in ("E", "F")}
    levels = list(statistical)
    for key in ("E", "F"):
        if flags.get(key, {}).get("supported"):
            levels.append(key)
    level = levels[-1] if levels else None
    return {
        "schema": "morpheus-explanation-decomposition/v1",
        "candidate_hash": candidate.candidate_hash, "k": int(k),
        "n_proposal": int(len(p_idx)), "n_replication": int(len(r_idx)),
        "n_replication_groups": int(len(uniq)), "n_boot": int(n_boot), "min_delta": float(min_delta),
        "Q": {m: point["Q_" + m] for m in names},
        "delta_add": point["delta_add"], "delta_nonadd": point["delta_nonadd"],
        "gain_relation": point["gain_relation"],
        "delta_context": point.get("delta_context"),
        "gain_relation_within_context": point.get("gain_relation_within_context"),
        "intervals": ci, "context_regimes": regimes, "context_tested": c is not None,
        "supported": supported, "statistical_levels": statistical, "statistical_level": statistical_level,
        "evidence_flags": flags, "levels": levels, "level": level,
        "level_meaning": EXPLANATION_LEVELS.get(level) if level else
                         "relation not supported on the held-out groups (no level assigned)",
        "notes": ("single constituent: delta_add is 0 by definition and delta_nonadd compares the joint map "
                  "with the constituent's own map" if k == 1 else ""),
    }
