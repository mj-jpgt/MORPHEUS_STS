"""Target-conditioned residual hypothesis generator.

The generator is a *proposer*: it nominates small explicit hypotheses that the
existing freeze -> held-out fit -> structured-null path in
``general_discovery`` then evaluates exactly as it evaluates enumerated
candidates.  Nothing it computes is evidence.  Three stages, all on the
proposal (discovery) patients only:

1. ``fit_conventional`` removes what the predeclared conventional explanation
   (depth, composition, known programme scores, ...) already predicts, with
   patient-grouped cross-fitting and a small model library chosen on inner
   folds.  Candidate variables never enter this step.
2. ``ResidualInteractionNetwork`` (per-view sparse gates, shallow towers, a
   low-rank cross-view bilinear block and a small higher-order MLP) and, for
   spatial data, ``GraphContextBranch`` (residual GraphSAGE-style message
   passing over the within-sample spot graph with distance-gated edges) are
   trained on the cross-fitted residual with patient-grouped early stopping.
3. ``distil`` aggregates 8-16 patient-subsample/seed fits: inclusion frequency,
   permutation (ablation) importance, pairwise second-difference interaction
   attribution and response-surface stability, and emits 1-5 variable
   ``Candidate`` objects (single, additive set, product interaction,
   context-conditioned) with a ``provenance`` block in their metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from morpheus_encoder.general_discovery import Candidate

Array = np.ndarray

#: Column roles in a hypothesis design.  ``variable`` columns may form any
#: hypothesis; ``context`` columns (own-unit conventional quantities) only
#: condition a variable; ``neighbour_context`` columns (neighbourhood
#: aggregates) may condition a variable or stand alone.
ROLES = ("variable", "context", "neighbour_context")


# --------------------------------------------------------------------------
# Patient-grouped folds and the conventional-explanation step
# --------------------------------------------------------------------------

def grouped_folds(groups: Any | None, n_rows: int, n_splits: int, seed: int) -> Array:
    """Fold id per row; whole groups (patients) share a fold.  Deterministic."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.permutation(n_rows) % n_splits
    groups = np.asarray(groups)
    unique = np.unique(groups)
    k = max(2, min(n_splits, len(unique)))
    assignment = dict(zip(unique.tolist(), (rng.permutation(len(unique)) % k).tolist()))
    return np.array([assignment[g] for g in groups.tolist()])


def _has_repeated_groups(groups: Any | None) -> bool:
    """True when some group (patient/donor) has more than one row."""
    return groups is not None and len(np.unique(np.asarray(groups))) < len(np.asarray(groups))


def _library_model(name: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    alphas = np.logspace(-2, 4, 13)
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    if name == "gam":
        # Additive cubic splines per column + ridge: a regularised GAM.
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3), RidgeCV(alphas=alphas))
    if name == "gbm":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             early_stopping=False, random_state=seed)
    raise ValueError(f"unknown conventional library model {name!r}")


def _r2(y: Array, prediction: Array) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if total <= 1e-15 else float(1.0 - np.sum((y - prediction) ** 2) / total)


@dataclass(frozen=True)
class ConventionalFit:
    """Cross-fitted conventional prediction and residual on proposal rows."""

    prediction: Array
    residual: Array
    model: str
    inner_r2: Mapping[str, float]
    n_splits: int


def fit_conventional(X_views: Any, y: Any, groups: Any | None = None, *,
                     library: Sequence[str] = ("ridge", "gam", "gbm"),
                     n_splits: int = 5, seed: int = 0, centre_within_groups: bool = True) -> ConventionalFit:
    """Remove what the predeclared conventional explanation predicts.

    ``X_views`` is the conventional design (an array or a mapping of arrays,
    concatenated); it must never contain candidate variables.  Every library
    model is cross-fitted over patient-grouped folds of the supplied (proposal)
    rows, the model with the best out-of-fold R^2 is chosen, and its out-of-fold
    predictions define the residual: no row's residual uses a model that saw
    that row's patient.

    With ``groups`` and ``centre_within_groups`` every conventional column is
    centred within its group (patient/donor), exactly as the response is.  Real episodes centre the response
    within patient while conventional columns (depth, composition, programme
    scores) keep between-patient offsets; on the kidney smoke the raw design
    transferred to held-out patients with R^2 -1.1 (ridge) and the
    group-centred one with 0.86.  Centring uses only a row's own group, so it
    is equally available for a held-out patient.  For spatial data the
    groups are the samples (one sample per patient in the kidney cohort).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    # One row per group (patient-level episode): within-group centring would zero
    # every column and the residual, so it is skipped (the response is not
    # within-group centred either).
    centre_within_groups = bool(centre_within_groups and _has_repeated_groups(groups))
    if isinstance(X_views, Mapping):
        pieces = [np.asarray(v, dtype=float).reshape(len(y), -1) for _, v in sorted(X_views.items())]
        K = np.column_stack(pieces) if pieces else np.zeros((len(y), 0))
    else:
        K = np.zeros((len(y), 0)) if X_views is None else np.asarray(X_views, dtype=float).reshape(len(y), -1)
    if groups is not None and centre_within_groups and K.shape[1] and np.isfinite(K).all():
        g = np.asarray(groups)
        K = K.copy()
        for value in np.unique(g):
            rows = g == value
            K[rows] -= K[rows].mean(0)
    folds = grouped_folds(groups, len(y), n_splits, seed)
    fold_ids = np.unique(folds)
    if K.shape[1] == 0 or not np.isfinite(K).all():
        if K.shape[1] and not np.isfinite(K).all():
            raise ValueError("conventional design must be finite")
        pred = np.empty_like(y)
        for k in fold_ids:
            pred[folds == k] = y[folds != k].mean()
        return ConventionalFit(pred, y - pred, "mean", {"mean": _r2(y, pred)}, len(fold_ids))
    # The fold-mean model always competes: when no library model transfers to
    # held-out patients, the honest conventional explanation is "nothing
    # transferable", not the least-bad overfit (whose residual would carry
    # its errors into the search).
    mean_pred = np.empty_like(y)
    for k in fold_ids:
        mean_pred[folds == k] = y[folds != k].mean()
    scores: dict[str, float] = {"mean": _r2(y, mean_pred)}
    predictions: dict[str, Array] = {"mean": mean_pred}
    for name in library:
        pred = np.empty_like(y)
        for k in fold_ids:
            test = folds == k
            model = _library_model(name, seed).fit(K[~test], y[~test])
            pred[test] = model.predict(K[test])
        predictions[name], scores[name] = pred, _r2(y, pred)
    order = ["mean", *library]
    best = max(scores, key=lambda name: (scores[name], -order.index(name)))
    residual = y - predictions[best]
    if groups is not None and centre_within_groups:
        # A cross-fitted model's per-patient offset is its own transfer bias,
        # not structure a candidate variable could explain within patients.
        g = np.asarray(groups)
        for value in np.unique(g):
            rows = g == value
            residual[rows] -= residual[rows].mean()
    return ConventionalFit(y - residual, residual, best, scores, len(fold_ids))


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

class HardConcreteGate(nn.Module):
    """Per-feature L0 gate (Louizos et al. hard-concrete relaxation)."""

    BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1

    def __init__(self, width: int, init_open: float = 0.9):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((width,), float(np.log(init_open / (1 - init_open)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        return x * (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    def penalty(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self.BETA * np.log(-self.GAMMA / self.ZETA)).sum()

    def open_mask(self) -> torch.Tensor:
        return (torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA) > 0.0


def _tower(width: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))


class ResidualInteractionNetwork(nn.Module):
    """r_hat = sum_m f_m(h_m) + sum_{m<=n} (z_m P_mn) . (z_n Q_mn) + f_higher([h_1..h_M]).

    Each view m has its own sparse gate (``hard_concrete`` L0, or
    ``group_lasso`` over each input column's weights) giving gated inputs
    z_m = g_m * X_m, and its own shallow tower h_m = MLP_m(z_m), so no view's
    coordinates are compressed into another's.  The factorised block is a
    rank-``rank`` bilinear form between gated *inputs* of every view pair,
    including a view with itself (a factorisation machine): a pure product of
    two measured variables has a direct, identifiable parameterisation and a
    nonzero gradient from the first step, which a tower-only network on
    hundreds of noise columns does not provide.  The higher-order MLP reads
    every tower output.
    """

    def __init__(self, view_columns: Sequence[Sequence[int]], *, hidden: int = 128, rank: int = 16,
                 dropout: float = 0.1, gate: str = "hard_concrete", higher_hidden: int = 64,
                 init_open: float = 0.9):
        super().__init__()
        if gate not in ("hard_concrete", "group_lasso"):
            raise ValueError("gate must be hard_concrete or group_lasso")
        self.gate_kind = gate
        self.view_columns = [torch.as_tensor(list(c), dtype=torch.long) for c in view_columns]
        for m, cols in enumerate(self.view_columns):
            self.register_buffer(f"_cols{m}", cols, persistent=False)
        widths = [len(c) for c in self.view_columns]
        self.gates = nn.ModuleList(HardConcreteGate(w, init_open) if gate == "hard_concrete" else nn.Identity()
                                   for w in widths)
        self.towers = nn.ModuleList(_tower(w, hidden, dropout) for w in widths)
        self.main = nn.ModuleList(nn.Linear(hidden, 1) for _ in widths)
        self.linear = nn.ModuleList(nn.Linear(w, 1) for w in widths)
        self.pairs = [(m, n) for m in range(len(widths)) for n in range(m, len(widths))]
        self.P = nn.ParameterList(nn.Parameter(torch.randn(widths[m], rank) * 0.05) for m, _ in self.pairs)
        self.Q = nn.ParameterList(nn.Parameter(torch.zeros(widths[n], rank)) for _, n in self.pairs)
        self.higher = nn.Sequential(nn.Linear(hidden * len(widths), higher_hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(higher_hidden, 1))
        # Output heads start at zero: an untrained network predicts the
        # residual mean, so early stopping never prefers a worse-than-zero fit.
        for head in [*self.main, *self.linear, self.higher[-1]]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    #: Two-phase training: (1) gates, linear heads and the factorised block
    #: select a sparse set of columns; (2) towers and the higher-order MLP are
    #: added on the gated inputs.  Training towers from the start memorises
    #: the proposal rows through hundreds of still-open noise columns before
    #: any gate can close.
    n_phases = 2

    def param_groups(self, cfg: "GeneratorConfig", phase: int) -> list[dict]:
        gates = [p_ for n_, p_ in self.named_parameters() if n_.endswith("log_alpha")]
        fast = [*self.P, *self.Q, *self.linear.parameters()]
        slow = [*self.towers.parameters(), *self.main.parameters(), *self.higher.parameters()]
        for p_ in slow:
            p_.requires_grad_(phase > 0)
        groups = [{"params": fast, "lr": cfg.factor_learning_rate, "weight_decay": cfg.weight_decay}]
        if gates:
            groups.append({"params": gates, "lr": cfg.gate_learning_rate, "weight_decay": 0.0})
        if phase > 0:
            groups.append({"params": slow, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zs = [gate(x[:, getattr(self, f"_cols{m}")]) for m, gate in enumerate(self.gates)]
        hs = [tower(z) for tower, z in zip(self.towers, zs)]
        out = sum(head(h).squeeze(-1) + lin(z).squeeze(-1)
                  for head, lin, h, z in zip(self.main, self.linear, hs, zs))
        for p, (m, n) in enumerate(self.pairs):
            out = out + ((zs[m] @ self.P[p]) * (zs[n] @ self.Q[p])).sum(-1)
        return out + self.higher(torch.cat(hs, dim=-1)).squeeze(-1)

    def _column_norms(self) -> list[torch.Tensor]:
        norms = [t[0].weight.pow(2).sum(0) + lin.weight.pow(2).sum(0) for t, lin in zip(self.towers, self.linear)]
        for p, (m, n) in enumerate(self.pairs):
            norms[m] = norms[m] + self.P[p].pow(2).sum(1)
            norms[n] = norms[n] + self.Q[p].pow(2).sum(1)
        return [v.clamp_min(1e-12).sqrt() for v in norms]

    def sparsity_penalty(self) -> torch.Tensor:
        if self.gate_kind == "hard_concrete":
            return sum(g.penalty() for g in self.gates)
        return sum(v.sum() for v in self._column_norms())

    def open_columns(self) -> Array:
        """Global column indices whose gate can pass signal at evaluation time."""
        out = []
        norms = None if self.gate_kind == "hard_concrete" else self._column_norms()
        for m, (gate, cols) in enumerate(zip(self.gates, self.view_columns)):
            if norms is None:
                mask = gate.open_mask().detach().cpu().numpy()
            else:
                v = norms[m].detach().cpu().numpy()
                mask = v > 1e-2 * max(float(v.max()), 1e-12)
            out.extend(cols.numpy()[mask].tolist())
        return np.asarray(sorted(out), dtype=int)


class GraphContextBranch(nn.Module):
    """Residual GraphSAGE-style branch: own variables modulated by a learned
    neighbourhood state built from gated context features.

    r_hat_i = w0 . g(x_i) + sum_j g(x_i)_j * c_j(h_i), where h_i comes from
    ``layers`` residual message-passing steps over the context features of
    unit i and its within-sample neighbours, with a learned distance gate on
    every edge.  ``c_j(h_i)`` is a varying coefficient: "variable j matters
    when the neighbourhood is in state h".  Input matrix layout is
    ``[variables | context]``.
    """

    def __init__(self, n_variables: int, n_context: int, *, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, init_open: float = 0.9):
        super().__init__()
        self.n_variables = n_variables
        self.var_gate = HardConcreteGate(n_variables, init_open)
        self.ctx_gate = HardConcreteGate(n_context, init_open)
        self.inp = nn.Linear(n_context, hidden)
        self.self_lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.nbr_lin = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.edge = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))
        self.drop = nn.Dropout(dropout)
        self.coef = nn.Linear(hidden, n_variables)
        self.main = nn.Linear(n_variables, 1)
        for head in (self.coef, self.main):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.neighbours: torch.Tensor | None = None
        self.distance: torch.Tensor | None = None

    n_phases = 1

    def param_groups(self, cfg: "GeneratorConfig", phase: int) -> list[dict]:
        gates = [p_ for n_, p_ in self.named_parameters() if n_.endswith("log_alpha")]
        rest = [p_ for n_, p_ in self.named_parameters() if not n_.endswith("log_alpha")]
        return [{"params": rest, "lr": cfg.graph_learning_rate, "weight_decay": cfg.weight_decay},
                {"params": gates, "lr": cfg.gate_learning_rate, "weight_decay": 0.0}]

    def set_graph(self, neighbours: torch.Tensor, distance: torch.Tensor) -> None:
        self.neighbours, self.distance = neighbours, distance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.neighbours is not None and self.distance is not None
        xv, xc = x[:, :self.n_variables], x[:, self.n_variables:]
        h = torch.nn.functional.gelu(self.inp(self.ctx_gate(xc)))
        valid = (self.neighbours >= 0).float()
        index = self.neighbours.clamp_min(0)
        weight = torch.sigmoid(self.edge(self.distance.unsqueeze(-1))).squeeze(-1) * valid
        denom = weight.sum(1, keepdim=True).clamp_min(1e-6)
        for self_lin, nbr_lin, norm in zip(self.self_lin, self.nbr_lin, self.norms):
            message = (h[index] * weight.unsqueeze(-1)).sum(1) / denom
            h = h + self.drop(torch.nn.functional.gelu(norm(self_lin(h) + nbr_lin(message))))
        g = self.var_gate(xv)
        return self.main(g).squeeze(-1) + (g * self.coef(h)).sum(-1)

    def sparsity_penalty(self) -> torch.Tensor:
        return self.var_gate.penalty() + self.ctx_gate.penalty()

    def open_columns(self) -> Array:
        var = np.flatnonzero(self.var_gate.open_mask().detach().cpu().numpy())
        ctx = np.flatnonzero(self.ctx_gate.open_mask().detach().cpu().numpy()) + self.n_variables
        return np.concatenate([var, ctx]).astype(int)


# --------------------------------------------------------------------------
# Configuration, design and training
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratorConfig:
    n_fits: int = 8
    hidden: int = 128
    rank: int = 16
    dropout: float = 0.1
    gate: str = "hard_concrete"
    gate_init_open: float = 0.9
    sparsity: float = 1e-2
    max_epochs: int = 120
    patience: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    gate_learning_rate: float = 5e-2
    factor_learning_rate: float = 3e-3
    max_grad_norm: float = 1.0
    graph_learning_rate: float = 3e-3
    min_epochs: int = 15
    weight_decay: float = 1e-4
    stop_fraction: float = 0.25
    top_k: int = 12
    min_importance: float = 0.005
    min_interaction: float = 0.004
    min_interaction_ratio: float = 0.15
    min_frequency: float = 0.5
    min_stability: float = 0.5
    proxy_correlation: float = 0.8
    non_recurrent_per_fit: int = 2
    attribution_rows: int = 2000
    reference_rows: int = 512
    graph: bool = True
    graph_hidden: int = 64
    graph_layers: int = 2
    graph_steps: int = 300
    library: tuple[str, ...] = ("ridge", "gam", "gbm")
    inner_folds: int = 5
    max_order: int = 5


@dataclass
class HypothesisDesign:
    """Candidate design: columns of ``X`` partitioned into named views.

    The first ``n_base`` columns are the episode's own candidate matrix, so
    a candidate over base columns has exactly the indices an enumerated
    candidate would have.  ``conventional`` is the generator's conventional
    explanation (the predeclared baseline plus any bank views that are
    ordinary predictors of this target family); it is never a candidate
    source.
    """

    X: Array
    views: dict[str, Array]
    roles: dict[str, str]
    modalities: dict[str, str]
    column_names: list[str]
    n_base: int
    neighbour_of: dict[int, int] = field(default_factory=dict)
    excluded: list[dict] = field(default_factory=list)
    conventional: Array | None = None
    conventional_names: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"n_base": int(self.n_base), "width": int(self.X.shape[1]),
                "views": {name: {"width": int(len(cols)), "role": self.roles[name],
                                 "modality": self.modalities.get(name, "unknown")}
                          for name, cols in self.views.items()},
                "conventional_width": 0 if self.conventional is None else int(self.conventional.shape[1]),
                "conventional_views": sorted({c.split(":", 1)[0] for c in self.conventional_names}),
                "excluded": list(self.excluded)}


def _neighbour_mean(values: Array, neighbours: Sequence[Any]) -> Array:
    out = np.array(values, dtype=float, copy=True)
    for i, idx in enumerate(neighbours):
        idx = np.asarray(idx, dtype=int)
        if idx.size:
            out[i] = values[idx].mean(axis=0)
    return out


#: Bank views that identify a unit's position or membership rather than
#: measure it; the conventional step already centres within patient/donor.
_IDENTIFIER_VIEWS = ("coords", "patient", "donor", "site")
#: Bank views that are ordinary predictors of a neighbourhood-programme
#: target (protocol v0.4 competitor: neighbour means of programmes and PCA,
#: own-spot PCA): conventional explanation, not candidates.
_NEIGHBOURHOOD_CONVENTIONAL = ("context_neighbour_mean", "pca")
#: Bank views measuring molecular state (for a family's context requirement).
MOLECULAR_VIEWS = ("raw_genes", "pca", "nmf", "programmes", "scimilarity", "scgpt", "candidate_matrix")
#: Bank views measuring tissue context.
CONTEXT_VIEWS = ("context_neighbour_mean", "cellvit_structure")


def _excluded_view(name: str, modality: str, target: str) -> str | None:
    """Target-family exclusion: a view that restates the response is never a
    candidate source (the response is a deterministic function of it).  The
    bank applies its own column-level exclusions first; this is a second,
    view-level guard."""
    lname, lmod = name.lower(), modality.lower()
    if target.startswith("morph_pc") and ("hoptimus" in lname or lmod in ("image", "pathology", "histology")):
        return "response is a principal component of this image embedding"
    if target.startswith("cv_") and "cellvit" in lname:
        return "response is a CellViT measurement"
    if lmod in ("protein", "adt") or "adt" in lname:
        return "protein view would contain the target protein"
    if lname in _IDENTIFIER_VIEWS:
        return "identifier view (position/membership), not a measurement"
    return None


def _proposal_standardise(values: Array, proposal: Array, groups: Array | None = None) -> Array:
    """Centre each column within patient/donor (as the response is), then
    z-score on proposal rows (response-free).  Frozen FM views can share one
    dominant direction (scGPT spot embeddings: mean pairwise cosine 0.97);
    centring and scaling per column removes it.  When every group is a single
    row (a patient-level episode: one row per patient) within-group centring
    is undefined (it would zero every column), so it is skipped and the column
    is only standardised on the proposal (fitting) rows."""
    if _has_repeated_groups(groups):
        values = np.array(values, dtype=float, copy=True)
        for value in np.unique(groups):
            rows = groups == value
            values[rows] -= values[rows].mean(0)
    mean = values[proposal].mean(0, keepdims=True)
    sd = values[proposal].std(0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return ((values - mean) / sd).astype(np.float32)


def _abs_correlation(X: Array, y: Array) -> Array:
    sd = X.std(0)
    out = np.zeros(X.shape[1])
    safe = sd > 1e-12
    if safe.any():
        out[safe] = np.abs(((X[:, safe] - X[:, safe].mean(0)) * (y - y.mean())[:, None]).mean(0)
                           / (sd[safe] * max(float(y.std()), 1e-12)))
    return out


def episode_design(episode: Any, candidate_matrix: Array, *, target: str, response: Array | None = None,
                   candidate_views: Sequence[str] | None = None, context_views: Sequence[str] = (),
                   max_abs_correlation: float = 0.95) -> HypothesisDesign:
    """Adapter from an episode to a multi-view hypothesis design.

    With ``episode.metadata['bank']`` present (``view -> {X, columns,
    modality, scale, fitted_on, response_free}``), every allowed bank view
    becomes its own variable view (z-scored on proposal rows); for a
    neighbourhood-programme target the bank's neighbour means, PCA and
    programme views join the conventional explanation instead.  Without a
    bank, the design is the candidate matrix plus the predeclared
    conventional columns as a context view.  Spatial episodes also receive
    within-sample neighbourhood means of the conventional columns as a
    ``neighbour_context`` view.

    Leak guards: target-family views are excluded by name/modality; any
    extra column (candidate or conventional) whose proposal-row |correlation|
    with the response reaches ``max_abs_correlation`` is excluded; for
    ``nbr_mp`` targets own-spot programme columns are never averaged over
    neighbours (that would rebuild the response).

    ``candidate_views`` (a relationship family's view list) restricts the
    searchable views to those bank views; the candidate matrix is then carried
    as the first ``n_base`` columns for index compatibility but is not
    searched, and no fallback context blocks are added.  Views named in
    ``context_views`` take the ``context`` role (own-unit context that only
    conditions a molecular variable).  For ``own_mp``/``nbr_mp`` targets the
    target programme's own and neighbour-mean columns are never candidates.
    Bank views are centred within patient/donor before z-scoring.
    """
    metadata = getattr(episode, "metadata", {}) or {}
    context = getattr(episode, "context", {}) or {}
    proposal = np.asarray(episode.partitions.proposal)
    base = np.asarray(candidate_matrix, dtype=float)
    n = base.shape[0]
    is_nbr = target.startswith("nbr_mp")
    group_key = "patient_codes" if "patient_codes" in context else ("donor_codes" if "donor_codes" in context
                                                                    else None)
    groups = None if group_key is None else np.asarray(context[group_key])
    programme = (metadata.get("own_mp_program") or metadata.get("nbr_mp_program") or {}) \
        if (target.startswith("own_mp") or is_nbr) else {}
    programme_name = programme.get("program_name") if isinstance(programme, Mapping) else None
    symbols = list(metadata.get("gene_symbols", []) or [])
    base_names = symbols if len(symbols) == base.shape[1] else [f"x{j}" for j in range(base.shape[1])]
    blocks: list[tuple[str, Array, list[str], str, str]] = [("candidate_matrix", base, base_names, "variable",
                                                             "molecular")]
    conventional_blocks: list[tuple[Array, list[str]]] = []
    excluded: list[dict] = []
    baseline = context.get("baseline")
    names = list(metadata.get("baseline_columns", []) or [])
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float).reshape(n, -1)
        names = names if len(names) == baseline.shape[1] else [f"baseline{j}" for j in range(baseline.shape[1])]
        conventional_blocks.append((baseline, [f"baseline:{c}" for c in names]))
    bank = metadata.get("bank") or context.get("bank")
    if bank:
        for name in sorted(bank):
            view = bank[name]
            values = np.asarray(view["X"], dtype=float)
            modality = str(view.get("modality", "unknown"))
            reason = _excluded_view(name, modality, target)
            if reason is None and not view.get("response_free", False):
                reason = "view is not declared response-free"
            if reason is None and values.shape[0] != n:
                reason = "rows do not align with the episode"
            if reason is None and name == "raw_genes" and values.shape == base.shape and np.allclose(values, base):
                reason = "identical to the candidate matrix"
            routed_conventional = is_nbr and name in _NEIGHBOURHOOD_CONVENTIONAL
            if reason is None and candidate_views is not None and name not in candidate_views \
                    and not routed_conventional:
                reason = "not a candidate view of this relationship family"
            if reason is not None:
                excluded.append({"view": name, "reason": reason})
                continue
            values = values.reshape(n, -1)
            # One ``view:name`` prefix: bank columns may already carry their view name.
            columns = [str(c) if str(c).startswith(f"{name}:") else f"{name}:{c}"
                       for c in view.get("columns", range(values.shape[1]))]
            finite = np.isfinite(values).all(0)
            if not finite.all():
                # Unmeasured is not zero: a column with any missing row (e.g. an
                # image embedding that does not cover every spot) is left out
                # rather than imputed, and the omission is recorded.
                excluded.append({"view": name, "columns_dropped": int((~finite).sum()),
                                 "reason": "non-finite values (partial coverage); not imputed",
                                 "rows_missing_fraction": float((~np.isfinite(values)).any(1).mean())})
                values = values[:, finite]
                columns = [c for c, k in zip(columns, finite) if k]
                if not columns:
                    continue
            if programme_name:
                # own_mp: the programme itself and its neighbour mean; nbr_mp:
                # only its neighbour mean (the own-spot score is a competitor).
                own = [not (c.endswith(f"gavish:{programme_name}") and (not is_nbr or "nbr:" in c))
                       for c in columns]
                if not all(own):
                    excluded.append({"view": name, "reason": "target programme (own or neighbour mean)",
                                     "columns": [c for c, k in zip(columns, own) if not k]})
                    values = values[:, np.asarray(own)]
                    columns = [c for c, k in zip(columns, own) if k]
            values = _proposal_standardise(values, proposal, groups)
            if routed_conventional:
                conventional_blocks.append((values, columns))
                continue
            role = ("context" if name in context_views else
                    "neighbour_context" if name == "context_neighbour_mean" else "variable")
            blocks.append((name, values, columns, role, modality))
    family_mode = candidate_views is not None
    if baseline is not None and not family_mode:
        blocks.append(("context_conventional", baseline, [f"context_conventional:{c}" for c in names], "context", "conventional"))
    neighbours = context.get("neighbors")
    if neighbours is not None and baseline is not None and not family_mode:
        keep = [j for j, name in enumerate(names)
                if not name.startswith("nbr_") and not (is_nbr and name.startswith("mp:"))]
        if keep:
            values = _neighbour_mean(baseline[:, keep], list(neighbours))
            blocks.append(("neighbour_conventional", values, [f"neighbour_conventional:{names[j]}" for j in keep],
                           "neighbour_context", "spatial"))
    yp = None if response is None else np.asarray(response, dtype=float)[proposal]
    # Candidate design (float32: a full bank is thousands of columns).
    widths = [b[1].shape[1] for b in blocks]
    X = np.empty((n, sum(widths)), dtype=np.float32)
    offset = 0
    for (_, values, _, _, _), w in zip(blocks, widths):
        X[:, offset:offset + w] = values
        offset += w
    column_names = [c for b in blocks for c in b[2]]
    keep_mask = np.ones(X.shape[1], dtype=bool)
    if yp is not None:
        corr = _abs_correlation(X[proposal], yp)
        for j in np.flatnonzero(corr >= max_abs_correlation):
            if j >= base.shape[1]:
                keep_mask[j] = False
                excluded.append({"column": column_names[j], "reason": f"|corr| with response {corr[j]:.3f}"})
    views: dict[str, Array] = {}
    roles: dict[str, str] = {}
    modalities: dict[str, str] = {}
    start = 0
    new_index = np.cumsum(keep_mask) - 1
    for (name, _, _, role, modality), w in zip(blocks, widths):
        cols = np.arange(start, start + w)
        start += w
        cols = cols[keep_mask[cols]]
        if family_mode and name == "candidate_matrix":
            continue  # carried for index compatibility, not searched
        if len(cols):
            views[name], roles[name], modalities[name] = new_index[cols], role, modality
    if not keep_mask.all():
        X = X[:, keep_mask]
    column_names = [c for c, k in zip(column_names, keep_mask) if k]
    neighbour_of: dict[int, int] = {}
    index_of = {c: i for i, c in enumerate(column_names)}
    for col in views.get("neighbour_conventional", []):
        source = index_of.get("context_conventional:" + column_names[col][len("neighbour_conventional:"):])
        if source is not None:
            neighbour_of[int(source)] = int(col)
    conventional = None
    conventional_names: list[str] = []
    if conventional_blocks:
        conventional = np.column_stack([b[0] for b in conventional_blocks]).astype(float)
        conventional_names = [c for b in conventional_blocks for c in b[1]]
        if yp is not None:
            n_baseline = 0 if baseline is None else baseline.shape[1]
            corr = _abs_correlation(conventional[proposal], yp)
            drop = [j for j in np.flatnonzero(corr >= max_abs_correlation) if j >= n_baseline]
            for j in drop:
                excluded.append({"conventional_column": conventional_names[j],
                                 "reason": f"|corr| with response {corr[j]:.3f}"})
            if drop:
                keep = np.setdiff1d(np.arange(conventional.shape[1]), drop)
                conventional = conventional[:, keep]
                conventional_names = [conventional_names[j] for j in keep]
    return HypothesisDesign(X=X, views=views, roles=roles, modalities=modalities, column_names=column_names,
                            n_base=int(base.shape[1]), neighbour_of=neighbour_of, excluded=excluded,
                            conventional=conventional, conventional_names=conventional_names)


def enumerate_design(X: Array, y: Array, *, views: Mapping[str, Array], enumerate_views: Sequence[str],
                     max_candidates: int, require_context: bool = False,
                     n_molecular: int = 24, n_context: int = 8,
                     molecular_views: Sequence[str] = MOLECULAR_VIEWS,
                     context_views: Sequence[str] = CONTEXT_VIEWS) -> tuple[Candidate, ...]:
    """The enumeration proposer over a relationship family's bank views.

    ``X`` holds proposal rows of the design; candidates index its columns.
    Without a context requirement this is ``propose_candidates`` (singles,
    screened low-order products, PCA subspaces) over the family's views.
    With one, every candidate is a molecular x context product: the
    ``n_molecular`` molecular and ``n_context`` context columns with the
    largest |correlation| with ``y`` are paired and the pairs ranked by the
    same product screen ``propose_candidates`` uses.
    """
    from morpheus_encoder.general_discovery import _product_screen_score, propose_candidates

    cols = np.asarray(sorted(int(c) for v in enumerate_views if v in views for c in views[v]), dtype=int)
    if not len(cols):
        return ()
    meta = {"proposer": "enumeration"}
    y = np.asarray(y, dtype=float)
    if not require_context:
        sub = np.asarray(X[:, cols], dtype=float)
        found = propose_candidates(sub, y, max_candidates=max_candidates, max_pairs=max_candidates // 2,
                                   max_subspaces=max_candidates // 4)
        # cols is increasing, so the remap keeps variables sorted and weights aligned.
        return tuple(Candidate(tuple(int(cols[i]) for i in c.variables), weights=c.weights,
                               feature_form=c.feature_form, proposal_family=c.proposal_family,
                               proposal_score=c.proposal_score, metadata=meta) for c in found)
    view_of = {int(c): v for v, vc in views.items() for c in vc}
    molecular = np.asarray([c for c in cols if view_of[c] in molecular_views], dtype=int)
    contextual = np.asarray([c for c in cols if view_of[c] in context_views], dtype=int)
    if not len(molecular) or not len(contextual):
        return ()
    top_m = molecular[np.argsort(-_abs_correlation(np.asarray(X[:, molecular], dtype=float), y),
                                 kind="stable")[:n_molecular]]
    top_c = contextual[np.argsort(-_abs_correlation(np.asarray(X[:, contextual], dtype=float), y),
                                  kind="stable")[:n_context]]
    Xd = np.asarray(X[:, np.concatenate([top_m, top_c])], dtype=float)
    local = {int(c): k for k, c in enumerate(np.concatenate([top_m, top_c]))}
    scored = []
    for m in top_m:
        for c in top_c:
            pair = tuple(sorted((int(m), int(c))))
            scored.append((_product_screen_score(Xd, y, (local[pair[0]], local[pair[1]])), pair))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(Candidate(pair, feature_form="product", proposal_family="context_product_screen",
                           proposal_score=float(score), metadata=meta) for score, pair in scored[:max_candidates])


def _standardise(X: Array, clip: float = 5.0) -> Array:
    """z-score on the supplied (proposal) rows, winsorised at +-``clip``: a
    product term over heavy-tailed expression otherwise lets a few extreme
    spots dominate a fit.  Network input only; candidates use raw values."""
    mean, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return np.clip((X - mean) / sd, -clip, clip)


def _device(device: str | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _train(model: nn.Module, X: torch.Tensor, r: torch.Tensor, train: Array, stop: Array,
           cfg: GeneratorConfig, seed: int, *, full_batch: bool = False) -> tuple[nn.Module, float]:
    """MSE on the residual + sparsity penalty; patient-grouped early stopping.

    Runs ``model.n_phases`` phases, each with its own optimiser (from
    ``model.param_groups``) and early stopping on the held-out patients'
    rows; a phase starts from the previous phase's best state.  Gate logits
    and the factorised block take larger steps than the towers: with the
    towers' step size an L0 gate needs thousands of updates to close.
    """
    torch.manual_seed(seed)
    train_t = torch.as_tensor(train, device=X.device)
    stop_t = torch.as_tensor(stop, device=X.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    with torch.no_grad():
        pred = model(X) if full_batch else model(X[stop_t])
        # The untrained network predicts ~0 (zero-initialised heads): a fit
        # must beat "no residual structure" on held-out patients to be kept.
        best = float(((pred[stop_t] if full_batch else pred) - r[stop_t]).pow(2).mean())
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for phase in range(model.n_phases):
        optimiser = torch.optim.AdamW(model.param_groups(cfg, phase))
        stale = 0
        if full_batch:
            epochs, patience = cfg.graph_steps, 2 * cfg.patience
        else:
            epochs, patience = (cfg.max_epochs if phase == 0 else max(cfg.max_epochs // 2, 1)), cfg.patience
        for epoch in range(epochs):
            model.train()
            if full_batch:
                optimiser.zero_grad()
                pred = model(X)
                loss = ((pred[train_t] - r[train_t]) ** 2).mean() + cfg.sparsity * model.sparsity_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimiser.step()
            else:
                order = train_t[torch.randperm(len(train_t), generator=generator).to(X.device)]
                for start in range(0, len(order), cfg.batch_size):
                    batch = order[start:start + cfg.batch_size]
                    optimiser.zero_grad()
                    loss = ((model(X[batch]) - r[batch]) ** 2).mean() + cfg.sparsity * model.sparsity_penalty()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimiser.step()
            model.eval()
            with torch.no_grad():
                pred = model(X) if full_batch else model(X[stop_t])
                stop_loss = float(((pred[stop_t] if full_batch else pred) - r[stop_t]).pow(2).mean())
            if stop_loss < best - 1e-5:
                best, stale = stop_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch >= cfg.min_epochs:
                stale += 1
                if stale >= patience:
                    break
        # Every phase ends at the best state seen so far (possibly the
        # untrained, zero-output network when no epoch transferred).
        model.load_state_dict(best_state)
    model.eval()
    return model, best


# --------------------------------------------------------------------------
# Attribution (per fit) and distillation (across fits)
# --------------------------------------------------------------------------

@dataclass
class FitAttribution:
    importance: dict[int, float]
    selected: list[int]
    pair_share: dict[tuple[int, int], float]
    pair_ratio: dict[tuple[int, int], float]
    single_profile: dict[int, Array]
    pair_profile: dict[tuple[int, int], Array]
    residual_r2: float
    branch: str


def _attribute(predict: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, r: Array,
               eval_rows: Array, reference_rows: Array, candidate_columns: Array, cfg: GeneratorConfig,
               seed: int, branch: str, allowed_pair: Callable[[int, int], bool]) -> FitAttribution:
    """Permutation (ablation) importance and second-difference interaction
    attribution on held-out-patient rows.

    For one column j, importance = E[(f(x) - f(x with x_j replaced))^2] / 2,
    which equals the variance f attributes to x_j (main effect plus its
    interactions) when columns are independent.  For a pair (i, j) the
    second difference D = f(x) - f(x_i') - f(x_j') + f(x_i', x_j') isolates
    their joint term; E[D^2] / 4 is the variance of that term.  Both are
    expressed as a share of the residual variance on the same rows.
    ``predict`` maps a full input matrix to predictions at ``eval_rows`` /
    ``reference_rows`` (the graph branch needs every node for messages).
    ``seed`` fixes the replacement permutation; callers pass the same seed in
    every fit so reference-row response profiles are comparable across fits.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = torch.as_tensor(rng.permutation(n), device=X.device)
    var_r = max(float(np.var(r[eval_rows])), 1e-12)
    with torch.no_grad():
        base_full = predict(X)
    base_eval = base_full[eval_rows]
    # Share of held-out-patient residual energy the fit removes relative to
    # predicting zero (the early-stopping criterion); <= 0 means no transfer.
    r_eval = r[eval_rows]
    residual_r2 = float(1.0 - np.mean((r_eval - base_eval.cpu().numpy()) ** 2) / max(np.mean(r_eval ** 2), 1e-12))

    def replaced(cols: Sequence[int]) -> torch.Tensor:
        Xm = X.clone()
        for c in cols:
            Xm[:, c] = X[perm, c]
        return Xm

    importance: dict[int, float] = {}
    single_full: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for c in candidate_columns.tolist():
            out = predict(replaced([c]))
            single_full[c] = out
            importance[c] = float(((base_eval - out[eval_rows]) ** 2).mean() / 2.0 / var_r)
    ranked = sorted(importance, key=lambda c: (-importance[c], c))
    selected = [c for c in ranked[:cfg.top_k] if importance[c] >= cfg.min_importance]
    pair_share: dict[tuple[int, int], float] = {}
    pair_ratio: dict[tuple[int, int], float] = {}
    pair_profile: dict[tuple[int, int], Array] = {}
    with torch.no_grad():
        for a_pos, a in enumerate(selected):
            for b in selected[a_pos + 1:]:
                i, j = (a, b) if a < b else (b, a)
                if not allowed_pair(i, j):
                    continue
                both = predict(replaced([i, j]))
                diff = base_full - single_full[i] - single_full[j] + both
                share = float((diff[eval_rows] ** 2).mean() / 4.0 / var_r)
                pair_share[(i, j)] = share
                pair_ratio[(i, j)] = share / max(min(importance[i], importance[j]), 1e-12)
                pair_profile[(i, j)] = diff[reference_rows].cpu().numpy()
    single_profile = {c: (base_full - single_full[c])[reference_rows].cpu().numpy() for c in selected}
    return FitAttribution(importance, selected, pair_share, pair_ratio, single_profile, pair_profile,
                          residual_r2, branch)


def _merge_proxies(attributions: Sequence[FitAttribution], X: Array, view_of: Array,
                   threshold: float) -> tuple[list[FitAttribution], dict[int, list[int]]]:
    """Count recurrence at the level of near-duplicate measurements.

    Thousands of correlated columns (genes of one programme, FM coordinates)
    let independent fits pick different members of the same signal, so no
    single column recurs although the signal does.  Columns any fit selected
    are grouped greedily within their view (|r| >= ``threshold`` on proposal
    rows, most often selected member first; a unit's own context and its
    neighbourhood mean stay distinct claims) and each fit's attributions are re-keyed to the
    group representative; the other members are reported as proxies.
    """
    chosen = sorted({c for a in attributions for c in a.selected}
                    | {c for a in attributions for pair in a.pair_share for c in pair})
    if len(chosen) < 2 or threshold >= 1.0:
        return list(attributions), {}
    count = {c: sum(c in a.selected for a in attributions) for c in chosen}
    rows = np.arange(0, len(X), max(1, len(X) // 20000))
    V = np.asarray(X[rows][:, chosen], dtype=float)
    V = V - V.mean(0)
    norm = np.sqrt((V ** 2).sum(0))
    norm[norm < 1e-12] = 1.0
    corr = np.abs((V / norm).T @ (V / norm))
    position = {c: k for k, c in enumerate(chosen)}
    representative: dict[int, int] = {}
    for c in sorted(chosen, key=lambda c: (-count[c], c)):
        if c in representative:
            continue
        representative[c] = c
        for d in chosen:
            if (d not in representative and view_of[d] == view_of[c]
                    and corr[position[c], position[d]] >= threshold):
                representative[d] = c
    proxies: dict[int, list[int]] = {}
    for d, c in representative.items():
        if d != c:
            proxies.setdefault(c, []).append(d)
    merged = []
    for a in attributions:
        importance: dict[int, float] = {}
        single_profile: dict[int, Array] = {}
        selected: list[int] = []
        for c in a.selected:
            rc = representative[c]
            if rc not in importance or a.importance[c] > importance[rc]:
                importance[rc], single_profile[rc] = a.importance[c], a.single_profile[c]
            if rc not in selected:
                selected.append(rc)
        pair_share: dict[tuple[int, int], float] = {}
        pair_ratio: dict[tuple[int, int], float] = {}
        pair_profile: dict[tuple[int, int], Array] = {}
        for (i, j), share in a.pair_share.items():
            key = tuple(sorted((representative[i], representative[j])))
            if key[0] == key[1] or (key in pair_share and pair_share[key] >= share):
                continue
            pair_share[key], pair_ratio[key], pair_profile[key] = share, a.pair_ratio[(i, j)], a.pair_profile[(i, j)]
        merged.append(FitAttribution(importance, selected, pair_share, pair_ratio, single_profile, pair_profile,
                                     a.residual_r2, a.branch))
    return merged, proxies


def _profile_stability(profiles: list[Array]) -> float:
    """Mean pairwise correlation of a response profile across fits."""
    if len(profiles) < 2:
        return 0.0
    values = []
    for a in range(len(profiles)):
        for b in range(a + 1, len(profiles)):
            x, y = profiles[a] - profiles[a].mean(), profiles[b] - profiles[b].mean()
            denom = float(np.sqrt((x @ x) * (y @ y)))
            values.append(0.0 if denom <= 1e-15 else float(x @ y / denom))
    return float(np.mean(values))


@dataclass
class GeneratorResult:
    candidates: tuple[Candidate, ...]
    report: dict


def _round(value: float) -> float:
    return float(round(float(value), 4))


def generate_hypotheses(X: Any, y: Any, groups: Any | None = None, *, conventional: Any | None = None,
                        views: Mapping[str, Sequence[int]] | None = None,
                        roles: Mapping[str, str] | None = None,
                        column_names: Sequence[str] | None = None,
                        context: Any | None = None,
                        neighbours: Sequence[Any] | None = None, coords: Any | None = None,
                        neighbour_of: Mapping[int, int] | None = None,
                        max_candidates: int = 24, seed: int = 0, device: str | None = None,
                        config: GeneratorConfig | None = None,
                        admissible: Callable[[tuple[int, ...]], bool] | None = None) -> GeneratorResult:
    """Propose 1-5 variable hypotheses from the residual of ``y`` on ``conventional``.

    ``X`` (proposal rows only) is the candidate design; ``views`` maps a view
    name to its column indices (default: one view) and ``roles`` gives each
    view's role (see ``ROLES``).  An optional separate ``context`` matrix is
    appended internally as a context view and context-conditioned hypotheses
    over it are emitted with ``Candidate.context_variables``.  ``neighbours``
    (row -> neighbour row indices within the proposal rows) enables the graph
    branch.  Emitted variables always index ``X`` (or ``context``).  Columns
    in no view are carried but never searched.  ``admissible`` (a predicate on
    a sorted column tuple) restricts which hypotheses may be emitted, e.g. a
    family's "at least one molecular and one context variable" rule.
    """
    started = time.time()
    cfg = config or GeneratorConfig()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and align with y")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    n, p = X.shape
    views = {name: np.asarray(cols, dtype=int) for name, cols in (views or {"X": range(p)}).items()}
    roles = {name: (roles or {}).get(name, "variable") for name in views}
    names = list(column_names) if column_names is not None else [f"x{j}" for j in range(p)]
    n_context = 0
    if context is not None:
        context = np.asarray(context, dtype=float).reshape(n, -1)
        n_context = context.shape[1]
        views["context"] = np.arange(p, p + n_context)
        roles["context"] = "context"
        names = names + [f"context{k}" for k in range(n_context)]
        X = np.column_stack([X, context])
    if any(role not in ROLES for role in roles.values()):
        raise ValueError(f"roles must be among {ROLES}")
    width = X.shape[1]
    role_of = np.empty(width, dtype=object)
    view_of = np.empty(width, dtype=object)
    for name, cols in views.items():
        role_of[cols], view_of[cols] = roles[name], name

    conv = fit_conventional(conventional, y, groups, library=cfg.library, n_splits=cfg.inner_folds, seed=seed)
    r_raw = conv.residual
    r_scale = max(float(r_raw.std()), 1e-12)
    r = np.clip((r_raw - r_raw.mean()) / r_scale, -8.0, 8.0)
    Xs = _standardise(X)
    constant = X.std(0) < 1e-12
    dev = _device(device)
    X_t = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=dev)
    group_arr = np.asarray(groups) if groups is not None else np.arange(n) % 10
    unique_groups = np.unique(group_arr)
    rng = np.random.default_rng(seed)
    reference_rows = np.sort(rng.choice(n, size=min(cfg.reference_rows, n), replace=False))

    def allowed_pair(i: int, j: int) -> bool:
        return not (role_of[i] != "variable" and role_of[j] != "variable")

    attributions: list[FitAttribution] = []
    n_parameters = 0
    view_list = list(views)
    n_stop = max(1, int(round(cfg.stop_fraction * len(unique_groups))))
    group_order = np.random.default_rng(seed + 104729).permutation(unique_groups)
    # Graph inputs: own variables plus own context; neighbour-context columns
    # are explicit aggregates and stay out (the branch learns its own).
    graph_ok = bool(cfg.graph and neighbours is not None)
    if graph_ok:
        var_cols = np.flatnonzero((role_of == "variable") & ~constant)
        ctx_cols = np.flatnonzero((role_of == "context") & ~constant)
        graph_ok = len(var_cols) > 0 and len(ctx_cols) > 0
    if graph_ok:
        k = max((len(v) for v in neighbours), default=0)
        nbr = np.full((n, max(k, 1)), -1, dtype=np.int64)
        dist = np.ones((n, max(k, 1)), dtype=np.float32)
        xy = None if coords is None else np.asarray(coords, dtype=float).reshape(n, -1)
        for i, idx in enumerate(neighbours):
            idx = np.asarray(idx, dtype=int)
            nbr[i, :len(idx)] = idx
            if xy is not None and len(idx):
                dist[i, :len(idx)] = np.sqrt(((xy[idx] - xy[i]) ** 2).sum(1))
        valid = nbr >= 0
        if xy is not None and valid.any():
            dist = dist / max(float(np.median(dist[valid])), 1e-12)
        graph_cols = np.concatenate([var_cols, ctx_cols])
        Xg_t = X_t[:, torch.as_tensor(graph_cols, device=dev)]
        nbr_t = torch.as_tensor(nbr, device=dev)
        dist_t = torch.as_tensor(dist, device=dev)

    for fit in range(cfg.n_fits):
        fit_seed = seed * 1009 + fit
        fit_rng = np.random.default_rng(fit_seed)
        # Held-out patients rotate through one fixed shuffle, so every patient
        # is held out about equally often; independent draws repeatedly held
        # out the same few patients and let one atypical patient decide most
        # fits' early stopping.
        width = min(n_stop, len(unique_groups) - 1) or 1
        stop_groups = group_order[(fit * width + np.arange(width)) % len(unique_groups)]
        stop = np.flatnonzero(np.isin(group_arr, stop_groups))
        train = np.flatnonzero(~np.isin(group_arr, stop_groups))
        eval_rows = stop if len(stop) <= cfg.attribution_rows else np.sort(
            fit_rng.choice(stop, size=cfg.attribution_rows, replace=False))
        torch.manual_seed(fit_seed)
        net = ResidualInteractionNetwork([views[v] for v in view_list], hidden=cfg.hidden, rank=cfg.rank,
                                         dropout=cfg.dropout, gate=cfg.gate, init_open=cfg.gate_init_open).to(dev)
        n_parameters += sum(p_.numel() for p_ in net.parameters())
        net, _ = _train(net, X_t, r_t, train, stop, cfg, fit_seed)
        open_cols = np.asarray([c for c in net.open_columns() if not constant[c]], dtype=int)

        # The residual network is row-wise: evaluate only the rows attribution
        # reads (held-out patients + the fixed reference rows).
        needed = torch.as_tensor(np.union1d(eval_rows, reference_rows), device=dev)

        def predict_net(Xm: torch.Tensor, _net=net, _rows=needed) -> torch.Tensor:
            out = torch.zeros(len(Xm), device=Xm.device)
            sub = Xm[_rows]
            out[_rows] = torch.cat([_net(sub[s:s + 8192]) for s in range(0, len(sub), 8192)])
            return out

        attributions.append(_attribute(predict_net, X_t, r, eval_rows, reference_rows, open_cols, cfg,
                                       seed, "residual_network", allowed_pair))
        if graph_ok:
            torch.manual_seed(fit_seed + 7)
            branch = GraphContextBranch(len(var_cols), len(ctx_cols), hidden=cfg.graph_hidden,
                                        layers=cfg.graph_layers, dropout=cfg.dropout,
                                        init_open=cfg.gate_init_open).to(dev)
            branch.set_graph(nbr_t, dist_t)
            n_parameters += sum(p_.numel() for p_ in branch.parameters())
            branch, _ = _train(branch, Xg_t, r_t, train, stop, cfg, fit_seed + 7, full_batch=True)
            local_open = branch.open_columns()

            def predict_graph(Xm: torch.Tensor, _b=branch) -> torch.Tensor:
                return _b(Xm)

            local = _attribute(predict_graph, Xg_t, r, eval_rows, reference_rows, local_open, cfg,
                               seed, "graph_branch",
                               lambda i, j: (i < len(var_cols)) != (j < len(var_cols)))
            # Map local graph columns back to design columns; a context column
            # B stands for "the neighbourhood level of B" -> its explicit
            # neighbour-mean column when the design has one.
            to_global = {int(lc): int(graph_cols[lc]) for lc in range(len(graph_cols))}

            def gmap(lc: int) -> int:
                g = to_global[lc]
                if lc >= len(var_cols) and neighbour_of and g in neighbour_of:
                    return int(neighbour_of[g])
                return g

            attributions.append(FitAttribution(
                importance={gmap(c): v for c, v in local.importance.items() if c < len(var_cols)},
                selected=[gmap(c) for c in local.selected if c < len(var_cols)],
                pair_share={tuple(sorted((gmap(i), gmap(j)))): v for (i, j), v in local.pair_share.items()},
                pair_ratio={tuple(sorted((gmap(i), gmap(j)))): v for (i, j), v in local.pair_ratio.items()},
                single_profile={gmap(c): v for c, v in local.single_profile.items() if c < len(var_cols)},
                pair_profile={tuple(sorted((gmap(i), gmap(j)))): v for (i, j), v in local.pair_profile.items()},
                residual_r2=local.residual_r2, branch="graph_branch"))

    candidates, distil_report = distil(attributions, X=X, residual=r_raw, role_of=role_of, view_of=view_of,
                                       names=names, n_original=p, cfg=cfg, max_candidates=max_candidates,
                                       n_parameters=n_parameters, admissible=admissible)
    by_branch: dict[str, list[float]] = {}
    for a in attributions:
        by_branch.setdefault(a.branch, []).append(a.residual_r2)
    report = {
        "proposer": "residual_generator",
        "conventional": {"model": conv.model, "inner_r2": {k: _round(v) for k, v in conv.inner_r2.items()},
                         "n_splits": conv.n_splits,
                         "width": 0 if conventional is None else int(np.asarray(conventional).reshape(n, -1).shape[1])},
        "residual_variance_share": _round(float(np.var(r_raw) / max(np.var(y), 1e-12))),
        "network_heldout_patient_improvement_over_zero": {k: [_round(v) for v in vals] for k, vals in by_branch.items()},
        "graph_branch": bool(graph_ok),
        "views": {name: {"width": int(len(cols)), "role": roles[name]} for name, cols in views.items()},
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "cost": {"fits": len(attributions), "parameters": int(n_parameters),
                 "wall_clock_seconds": round(time.time() - started, 2), "device": str(dev)},
        **distil_report,
    }
    return GeneratorResult(candidates, report)


def distil(attributions: Sequence[FitAttribution], *, X: Array, residual: Array, role_of: Array, view_of: Array,
           names: Sequence[str], n_original: int, cfg: GeneratorConfig, max_candidates: int,
           n_parameters: int = 0,
           admissible: Callable[[tuple[int, ...]], bool] | None = None) -> tuple[tuple[Candidate, ...], dict]:
    """Turn per-fit attributions into ranked 1-5 variable candidates.

    Only structures recurring in at least ``min_frequency`` of the fits of a
    branch, with a response profile whose mean cross-fit correlation is at
    least ``min_stability``, are emitted.  Score = frequency x max(stability, 0) x attributed
    residual variance, where a single variable is credited only with its
    main-effect share (importance minus the interaction shares it takes part
    in), so a variable that matters only jointly ranks through its pair.

    Recurrence is a ranking signal, not a filter: each fit's best
    ``non_recurrent_per_fit`` structures that did not recur are emitted too
    (deduplicated, ranked after every recurring structure, filling the
    remaining budget) with ``recurrent_in_generator=False``; held-out
    validation is the test.
    """
    attributions, proxies = _merge_proxies(attributions, X, view_of, cfg.proxy_correlation)
    n_fits_branch: dict[str, int] = {}
    for a in attributions:
        n_fits_branch[a.branch] = n_fits_branch.get(a.branch, 0) + 1
    single_hits: dict[int, list[tuple[str, float, float]]] = {}
    pair_hits: dict[tuple[int, int], list[tuple[str, float, float]]] = {}
    single_prof: dict[int, list[Array]] = {}
    pair_prof: dict[tuple[int, int], list[Array]] = {}
    for a in attributions:
        interaction_load = {c: 0.0 for c in a.selected}
        for (i, j), share in a.pair_share.items():
            if share >= cfg.min_interaction and a.pair_ratio[(i, j)] >= cfg.min_interaction_ratio:
                interaction_load[i] = interaction_load.get(i, 0.0) + share
                interaction_load[j] = interaction_load.get(j, 0.0) + share
                pair_hits.setdefault((i, j), []).append((a.branch, share, a.pair_ratio[(i, j)]))
                pair_prof.setdefault((i, j), []).append(a.pair_profile[(i, j)])
        for c in a.selected:
            main = max(a.importance[c] - interaction_load.get(c, 0.0), 0.0)
            single_hits.setdefault(c, []).append((a.branch, a.importance[c], main))
            single_prof.setdefault(c, []).append(a.single_profile[c])

    def freq(hits: list[tuple[str, float, float]]) -> tuple[float, str]:
        counts: dict[str, int] = {}
        for branch, *_ in hits:
            counts[branch] = counts.get(branch, 0) + 1
        best = max(counts, key=lambda b: (counts[b] / n_fits_branch[b], b))
        return counts[best] / n_fits_branch[best], best

    n_fits_total = len(attributions)
    cost = {"fits": n_fits_total, "parameters": int(n_parameters)}
    entries: list[dict] = []
    stable_singles: list[tuple[float, int, float]] = []
    for c, hits in single_hits.items():
        f, branch = freq(hits)
        if f < cfg.min_frequency or role_of[c] == "context":
            continue
        stability = _profile_stability(single_prof[c])
        if stability < cfg.min_stability:
            continue
        main = float(np.mean([h[2] for h in hits]))
        imp = float(np.mean([h[1] for h in hits]))
        stable_singles.append((main, c, f))
        entries.append({"kind": "single", "columns": (c,), "frequency": f, "branch": branch,
                        "stability": stability, "attributed": main, "importance": imp, "interaction": None,
                        "score": f * max(stability, 0.0) * main})
    stable_pairs: dict[tuple[int, int], float] = {}
    for (i, j), hits in pair_hits.items():
        f, branch = freq(hits)
        if f < cfg.min_frequency:
            continue
        stability = _profile_stability(pair_prof[(i, j)])
        if stability < cfg.min_stability:
            continue
        share = float(np.mean([h[1] for h in hits]))
        kind = "interaction" if role_of[i] == "variable" and role_of[j] == "variable" else "context_conditioned"
        score = f * max(stability, 0.0) * share
        stable_pairs[(i, j)] = score
        entries.append({"kind": kind, "columns": (i, j), "frequency": f, "branch": branch,
                        "stability": stability, "attributed": share, "importance": None,
                        "interaction": share, "score": score})
    # Higher order: a triple whose three pairs all recur is nominated as one
    # joint object (the validator's competitor already holds its pairs).
    pair_keys = sorted(stable_pairs)
    seen_triples = set()
    for a_i, (i, j) in enumerate(pair_keys):
        if cfg.max_order < 3:
            break
        for k in sorted({c for pair in pair_keys for c in pair} - {i, j}):
            triple = tuple(sorted((i, j, k)))
            if triple in seen_triples:
                continue
            sub = [tuple(sorted(p)) for p in ((triple[0], triple[1]), (triple[0], triple[2]), (triple[1], triple[2]))]
            if all(s in stable_pairs for s in sub) and all(role_of[c] == "variable" for c in triple):
                seen_triples.add(triple)
                entries.append({"kind": "interaction", "columns": triple,
                                "frequency": min(freq(pair_hits[s])[0] for s in sub), "branch": "residual_network",
                                "stability": float(np.mean([_profile_stability(pair_prof[s]) for s in sub])),
                                "attributed": float(np.mean([np.mean([h[1] for h in pair_hits[s]]) for s in sub])),
                                "importance": None, "interaction": None,
                                "score": min(stable_pairs[s] for s in sub)})
    # Additive sets: the strongest recurring main effects, sizes 2..max_order.
    stable_singles.sort(key=lambda t: (-t[0], t[1]))
    members_all = [c for main, c, _ in stable_singles if main > 0 and role_of[c] != "context"]
    residual = np.asarray(residual, dtype=float)
    for size in sorted({min(len(members_all), s) for s in (3, cfg.max_order)}):
        if size < 2:
            continue
        members = tuple(sorted(members_all[:size]))
        V = X[:, members] - X[:, members].mean(0)
        gram = V.T @ V + 1e-6 * len(V) * np.eye(len(members))
        weights = np.linalg.solve(gram, V.T @ (residual - residual.mean()))
        mains = {c: m for m, c, _ in stable_singles}
        freqs = {c: f_ for _, c, f_ in stable_singles}
        stability = float(np.mean([_profile_stability(single_prof[c]) for c in members]))
        attributed = float(sum(mains[c] for c in members))
        f = float(np.mean([freqs[c] for c in members]))
        entries.append({"kind": "additive_set", "columns": members, "frequency": f, "branch": "residual_network",
                        "stability": stability, "attributed": attributed, "importance": None, "interaction": None,
                        "weights": tuple(float(np.round(w, 6)) for w in weights),
                        "score": f * max(stability, 0.0) * attributed})

    for e in entries:
        e["recurrent"] = True
    if admissible is not None:
        entries = [e for e in entries if admissible(tuple(sorted(int(c) for c in e["columns"])))]
    recurrent_keys = {tuple(sorted(e["columns"])) for e in entries}
    extra: dict[tuple[int, ...], dict] = {}
    for a in attributions:
        picks: list[tuple[float, tuple[int, ...]]] = [
            (a.importance[c], (c,)) for c in a.selected if role_of[c] not in ("context", None)]
        picks += [(share, (i, j)) for (i, j), share in a.pair_share.items()
                  if share >= cfg.min_interaction and a.pair_ratio[(i, j)] >= cfg.min_interaction_ratio]
        picks.sort(key=lambda item: (-item[0], item[1]))
        taken = 0
        for value, key in picks:
            if taken >= cfg.non_recurrent_per_fit:
                break
            if key in recurrent_keys or (admissible is not None and not admissible(key)):
                continue
            taken += 1
            if key in extra:
                continue
            if len(key) == 1:
                hits, profiles = single_hits[key[0]], single_prof[key[0]]
                kind, attributed = "single", float(np.mean([h[2] for h in hits]))
            else:
                hits, profiles = pair_hits[key], pair_prof[key]
                attributed = float(np.mean([h[1] for h in hits]))
                kind = ("interaction" if role_of[key[0]] == "variable" and role_of[key[1]] == "variable"
                        else "context_conditioned")
            f, branch = freq(hits)
            extra[key] = {"kind": kind, "columns": key, "frequency": f, "branch": branch,
                          "stability": _profile_stability(profiles), "attributed": attributed,
                          "importance": None, "interaction": attributed if len(key) == 2 else None,
                          "score": f * max(attributed, 1e-9), "recurrent": False}

    def to_candidate(e: dict) -> Candidate:
        cols = tuple(int(c) for c in e["columns"])
        provenance = {
            "proposer": "residual_generator", "branch": e["branch"], "hypothesis_kind": e["kind"],
            "seed_fits": n_fits_branch.get(e["branch"], n_fits_total),
            "views": sorted({str(view_of[c]) for c in cols}),
            "column_names": [str(names[c]) for c in cols],
            "proxies": {str(names[c]): [str(names[q]) for q in proxies.get(c, [])] for c in cols if proxies.get(c)},
            "inclusion_frequency": _round(e["frequency"]),
            "interaction_attribution": None if e["interaction"] is None else _round(e["interaction"]),
            "response_surface_stability": _round(e["stability"]),
            "attributed_residual_variance": _round(e["attributed"]),
            "recurrent_in_generator": bool(e["recurrent"]),
            "cost": cost,
        }
        context_cols = tuple(c - n_original for c in cols if c >= n_original)
        own_cols = tuple(c for c in cols if c < n_original)
        common = dict(proposal_family="residual_generator", proposal_score=float(_round(e["score"])),
                      metadata={"proposer": "residual_generator", "provenance": provenance})
        if context_cols and own_cols:
            # A separately supplied context matrix: use the schema's own
            # context mechanism (feature = variable x context).
            return Candidate(own_cols, context_cols, weights=(1.0,) * len(own_cols),
                             feature_form="product" if len(own_cols) > 1 else "linear_subspace", **common)
        if e["kind"] == "single":
            return Candidate(cols, weights=(1.0,), feature_form="linear_subspace", **common)
        if e["kind"] == "additive_set":
            order = np.argsort(cols)
            return Candidate(tuple(cols[o] for o in order), weights=tuple(e["weights"][o] for o in order),
                             feature_form="linear_subspace", **common)
        return Candidate(tuple(sorted(cols)), feature_form="product", **common)

    entries = [e for e in entries if e["score"] > 0 and not (any(c >= n_original for c in e["columns"])
                                                              and all(c >= n_original for c in e["columns"]))]
    entries.sort(key=lambda e: (-e["score"], e["kind"], e["columns"]))
    joint = [e for e in entries if e["kind"] in ("interaction", "context_conditioned")]
    other = [e for e in entries if e["kind"] not in ("interaction", "context_conditioned")]
    # Protected share for joint objects (as in ``propose_candidates``) so a
    # long list of strong main effects cannot starve every interaction.
    reserve = min(len(joint), max(1, max_candidates // 3)) if joint else 0
    chosen = joint[:reserve] + other[:max_candidates - reserve]
    if len(chosen) < max_candidates:
        chosen += joint[reserve:reserve + max_candidates - len(chosen)]
    chosen.sort(key=lambda e: (-e["score"], e["kind"], e["columns"]))
    non_recurrent = sorted((e for e in extra.values() if not all(c >= n_original for c in e["columns"])),
                           key=lambda e: (-e["score"], e["columns"]))
    chosen += non_recurrent[:max(0, max_candidates - len(chosen))]
    candidates: list[Candidate] = []
    hashes = set()
    for e in chosen:
        cand = to_candidate(e)
        if cand.candidate_hash not in hashes:
            hashes.add(cand.candidate_hash)
            candidates.append(cand)
    report = {"n_recurring_structures": len(entries), "n_emitted": len(candidates),
              "non_recurrent": {"cap_per_fit": int(cfg.non_recurrent_per_fit), "found": len(extra),
                                "emitted": sum(1 for c in candidates
                                               if not c.metadata["provenance"]["recurrent_in_generator"])},
              "per_fit_selected": [{"branch": a.branch, "columns": [str(names[c]) for c in a.selected[:6]]}
                                   for a in attributions],
              "emitted_kinds": {k: sum(1 for c in candidates if c.metadata["provenance"]["hypothesis_kind"] == k)
                                for k in ("single", "additive_set", "interaction", "context_conditioned")}}
    return tuple(candidates), report


def proposal_neighbours(neighbours: Sequence[Any], proposal_rows: Array, n_rows: int) -> list[Array]:
    """Remap episode-level neighbour lists to proposal-local indices; edges to
    non-proposal rows are dropped (they cannot exist for a within-patient
    graph, but the remap never lets the generator read them)."""
    lookup = np.full(int(n_rows), -1, dtype=int)
    lookup[np.asarray(proposal_rows, dtype=int)] = np.arange(len(proposal_rows))
    out = []
    for row in np.asarray(proposal_rows, dtype=int):
        mapped = lookup[np.asarray(neighbours[int(row)], dtype=int)]
        out.append(mapped[mapped >= 0])
    return out


__all__ = ["CONTEXT_VIEWS", "MOLECULAR_VIEWS", "enumerate_design", "ConventionalFit", "GeneratorConfig", "GeneratorResult", "GraphContextBranch", "HardConcreteGate",
           "HypothesisDesign", "ResidualInteractionNetwork", "distil", "episode_design", "fit_conventional",
           "generate_hypotheses", "grouped_folds", "proposal_neighbours"]
