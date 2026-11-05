"""Residual-bank hypothesis generator: one network, one output head per explanation level.

Design (``.claude/ideas.md`` v0.6).  The comparison model is split into nested
levels N ⊂ B1 ⊂ B2 ⊂ B3 (column groups from ``dbench.decomposition.ladder_levels``):
N = nuisance (log depth and any RNA-quality column; the patient is removed by the
within-patient standardisation of response and columns), B1 = N + CellViT class
fractions, B2 = B1 + the 41 Gavish and 50 Hallmark own-spot programmes, B3 = B2 +
neighbour means (k = 6, 18) of the programmes and PCA and every other registered
competitor column (= the registered v0.5 comparison model).

1. ``residual_bank``: r(k) = y - f_k(B_k) on the proposal rows, cross-fitted by
   ``hypothesis_generator.fit_conventional`` (patient-grouped inner folds, ridge with
   its own inner alpha choice, the fold-mean model always competes, columns centred
   within patient, residual re-centred within patient).  No row's residual comes
   from a model that saw that row's patient.
2. ``ResidualBankNetwork``: ``hypothesis_generator.ResidualInteractionNetwork``
   (per-view hard-concrete sparse gates, per-view MLP towers, rank-r bilinear layer
   between the gated inputs of every view pair, higher-order MLP) with its output
   layers widened to one head per level.  Gates, towers, bilinear factors and the
   higher-order hidden layer are shared; each head has its own output weights and
   its own mixing of the bilinear rank components.  Trained with
   ``hypothesis_generator._train`` on the residual matrix with every column scaled to
   unit variance, so the mean squared error over heads is the multi-task loss with
   per-head weights 1 / var r(k) (``multitask_loss``).  Early stopping on held-out
   patients' rows of the proposal set, two phases, rotating held-out patients over
   ``n_fits`` fits, exactly as the single-head generator.
3. ``nominate``: per head and fit, every open column is pre-ranked by gate openness x
   |gradient x input| on the fit's held-out-patient rows; the top
   ``attribution_pool`` columns get permutation importance and the top ``pair_pool`` of
   those pairwise second-difference interaction shares (``hypothesis_generator._attribute``).
   The pool is several times the list length so that a head's top list is a selection,
   not the whole attributed pool (with a pool of 24 every head listed the same ~31
   structures on the first real targets).
   Across fits a single scores its main-effect share (importance minus the
   interaction shares it takes part in, as ``distil``) and a pair its interaction
   share, averaged over fits (a fit that did not attribute it counts 0).  Each head
   emits its top ``max_candidates`` hypotheses (a third reserved for pairs, as
   ``distil``).  A hypothesis nominated by several heads is one ``Candidate`` whose
   provenance carries the set of levels.
4. ``residual_pair_screen`` (products with zero marginal effects).  The sparse gates are
   trained on marginal gradients and close both members of a pure product before the
   bilinear layer learns it (planted zero-marginal products at 2-10 % of the residual:
   0 recovered by the network alone; every fit ended with 2-15 open columns), and a
   network Hessian is exactly zero for gated-off columns.  So pairs are also scored,
   per head, by |corr(e_k, x_i x_j)|^2 over all column pairs, where e_k = r(k) minus
   the head's prediction on each fit's held-out patients (fits rotate patients, so e_k
   is cross-fitted over the proposal rows; rows never held out keep r(k)).  The squared
   correlation is the share of e_k's variance a product feature carries, the same
   units as the network's attributed shares; screened pairs merge with the
   network's pairs under the pair reserve (provenance ``pair_source``).  Pairs of
   near-duplicate columns (|r| >= ``proxy_correlation``) are skipped.
   With ``pair_screen_partial`` a pair is scored by the partial correlation of x_i x_j
   with e_k given x_i and x_j (what the product adds beyond both main effects), and
   ``pair_cap_per_column`` keeps at most that many screened pairs per column per
   head, so one strong column does not multiply into many pairs (wave 4, f2 morph_pc3:
   15 of 75 pairs were pca:PC2 x another column).

Options measured in wave 5 (``ResidualBankConfig``): ``head_gates`` (per-head
hard-concrete gates on the input layer, towers and bilinear factors still shared),
``l0_width_normalised`` (each view's L0 penalty weighted by mean view width / view
width: same total pressure, every view weighted equally) and ``stop_metric =
"patient_r2"`` (early stopping on the mean over held-out patients and heads of
R^2 against the zero prediction instead of the pooled squared error).

Only protected views enter the network (raw HVG genes, scGPT, SCimilarity, PCA,
NMF); comparison columns never do.  Nothing here sees held-out patients: callers
pass proposal rows only.  The graph branch is not used in this version (the
molecular -> morphology family runs without it in the registered runner too).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from morpheus_encoder.general_discovery import Candidate
from morpheus_encoder.hypothesis_generator import (
    GeneratorConfig, ResidualInteractionNetwork, _attribute, _device, _standardise, _train, fit_conventional,
)

Array = np.ndarray

LEVELS: tuple[str, ...] = ("N", "B1", "B2", "B3")
#: Views the network may read: molecular measurements and their frozen-FM views.
PROTECTED_VIEWS: tuple[str, ...] = ("raw_genes", "scgpt", "scimilarity", "pca", "nmf")


@dataclass(frozen=True)
class ResidualBankConfig:
    """Bank-specific options around the single-head generator's ``GeneratorConfig``."""

    generator: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(n_fits=4))
    levels: tuple[str, ...] = LEVELS
    protected_views: tuple[str, ...] = PROTECTED_VIEWS
    attribution_pool: int = 96       # columns per head and fit that get permutation importance
    pair_pool: int = 16              # of those, the top columns by importance whose pairs are attributed
    pair_reserve: float = 1.0 / 3.0  # share of each head's list reserved for pairs (``distil``'s rule)
    pair_screen: bool = True         # exhaustive pair screen on each head's held-out network residual
    pair_screen_top: int = 64        # pairs per head kept from the screen
    pair_screen_partial: bool = True  # score x_i x_j by partial correlation with e_k given x_i and x_j
    pair_cap_per_column: int = 3     # screened pairs per column per head (0 = no cap)
    head_gates: bool = False         # per-head input gates in front of the shared towers
    l0_width_normalised: bool = False  # weight each view's L0 penalty by mean view width / view width
    stop_metric: str = "loss"        # early stopping: "loss" (``_train``) or "patient_r2" (mean held-out-patient R^2)
    residual_library: tuple[str, ...] = ("ridge",)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["generator"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in out["generator"].items()}
        for k in ("levels", "protected_views", "residual_library"):
            out[k] = list(out[k])
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResidualBankConfig":
        data = dict(data)
        gen = {k: (tuple(v) if isinstance(v, list) else v) for k, v in dict(data.pop("generator", {})).items()}
        for k in ("levels", "protected_views", "residual_library"):
            if k in data:
                data[k] = tuple(data[k])
        return cls(generator=GeneratorConfig(**gen), **data)


# --------------------------------------------------------------------------
# Levels and the residual bank
# --------------------------------------------------------------------------

def comparison_levels(comparison_names: Sequence[str], levels: Sequence[str] = LEVELS) -> dict[str, Array]:
    """Column indices of each nested level within the comparison model (Agent D's groups)."""
    from dbench.decomposition import ladder_levels

    ladder = ladder_levels(list(map(str, comparison_names)))["levels"]
    return {lv: np.asarray(ladder[lv], dtype=int) for lv in levels}


@dataclass(frozen=True)
class ResidualBank:
    residuals: Array           # (n, K) raw cross-fitted residuals, one column per level
    levels: tuple[str, ...]
    info: dict


def residual_bank(comparison: Any, y: Any, groups: Any | None, level_columns: Mapping[str, Sequence[int]], *,
                  n_splits: int = 5, seed: int = 0, library: Sequence[str] = ("ridge",)) -> ResidualBank:
    """r(k) = y - f_k(B_k), cross-fitted over patient-grouped folds of the supplied (proposal) rows."""
    y = np.asarray(y, dtype=float).reshape(-1)
    B = np.zeros((len(y), 0)) if comparison is None else np.asarray(comparison, dtype=float).reshape(len(y), -1)
    out, info = [], {}
    for level, cols in level_columns.items():
        cols = np.asarray(cols, dtype=int)
        fit = fit_conventional(B[:, cols] if len(cols) else None, y, groups, library=tuple(library),
                               n_splits=n_splits, seed=seed)
        out.append(fit.residual)
        info[level] = {"width": int(len(cols)), "model": fit.model,
                       "inner_r2": {k: float(round(v, 4)) for k, v in fit.inner_r2.items()},
                       "residual_variance_share": float(round(np.var(fit.residual) / max(np.var(y), 1e-12), 4))}
    return ResidualBank(np.column_stack(out), tuple(level_columns), info)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class ResidualBankNetwork(ResidualInteractionNetwork):
    """``ResidualInteractionNetwork`` with one output head per explanation level.

    r_hat_k = sum_m (main_mk(h_m) + lin_mk(z_m)) + sum_{m<=n} ((z_m P_mn) * (z_n Q_mn)) . c_mnk + out_k(H),
    with gates g_m (z_m = g_m * X_m), towers h_m, bilinear factors P, Q and the
    higher-order hidden layer H shared by all heads.
    """

    def __init__(self, view_columns: Sequence[Sequence[int]], n_heads: int, *, hidden: int = 128, rank: int = 16,
                 dropout: float = 0.1, gate: str = "hard_concrete", higher_hidden: int = 64,
                 init_open: float = 0.9):
        super().__init__(view_columns, hidden=hidden, rank=rank, dropout=dropout, gate=gate,
                         higher_hidden=higher_hidden, init_open=init_open)
        self.n_heads = int(n_heads)
        widths = [len(c) for c in self.view_columns]
        self.main = nn.ModuleList(nn.Linear(hidden, self.n_heads) for _ in widths)
        self.linear = nn.ModuleList(nn.Linear(w, self.n_heads) for w in widths)
        self.higher[-1] = nn.Linear(higher_hidden, self.n_heads)
        # Ones: at initialisation every head reads the bilinear block as the
        # single-head network does (sum over rank components).
        self.mix = nn.Parameter(torch.ones(len(self.pairs), rank, self.n_heads))
        for layer in [*self.main, *self.linear, self.higher[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def param_groups(self, cfg: GeneratorConfig, phase: int) -> list[dict]:
        groups = super().param_groups(cfg, phase)
        groups[0]["params"] = [*groups[0]["params"], self.mix]
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zs = [gate(x[:, getattr(self, f"_cols{m}")]) for m, gate in enumerate(self.gates)]
        hs = [tower(z) for tower, z in zip(self.towers, zs)]
        out = sum(head(h) + lin(z) for head, lin, h, z in zip(self.main, self.linear, hs, zs))
        for p, (m, n) in enumerate(self.pairs):
            out = out + ((zs[m] @ self.P[p]) * (zs[n] @ self.Q[p])) @ self.mix[p]
        return out + self.higher(torch.cat(hs, dim=-1))

    def head_parameters(self, k: int) -> list[torch.Tensor]:
        """Head k's own parameter slices (output rows and its bilinear mixing weights)."""
        out = [self.mix[:, :, k]]
        for layer in [*self.main, *self.linear, self.higher[-1]]:
            out += [layer.weight[k], layer.bias[k]]
        return out

    def openness(self, width: int, k: int | None = None) -> Array:
        """Evaluation-time gate value per design column (0 for columns in no view); ``k`` is
        ignored (the gates are shared by every head)."""
        out = np.zeros(width)
        for gate, cols in zip(self.gates, self.view_columns):
            if isinstance(gate, nn.Identity):
                out[cols.numpy()] = 1.0
                continue
            s = torch.sigmoid(gate.log_alpha) * (gate.ZETA - gate.GAMMA) + gate.GAMMA
            out[cols.numpy()] = s.clamp(0.0, 1.0).detach().cpu().numpy()
        return out


def _view_weights(view_columns, normalise: bool) -> list[float]:
    widths = [len(c) for c in view_columns]
    mean = float(np.mean(widths))
    return [mean / w if normalise else 1.0 for w in widths]


class WidthNormalisedBankNetwork(ResidualBankNetwork):
    """Shared gates; each view's L0 penalty weighted by mean view width / view width."""

    def sparsity_penalty(self) -> torch.Tensor:
        weights = _view_weights(self.view_columns, True)
        return sum(w * g.penalty() for w, g in zip(weights, self.gates))


class HeadGatedBankNetwork(ResidualBankNetwork):
    """Per-head hard-concrete input gates in front of the shared towers, bilinear
    factors and higher-order layer.  Head k reads ``towers(g_k * X)``, so the
    towers run once per head (training cost scales with the number of heads).
    The L0 penalty is averaged over heads (each head's loss carries weight 1/K)."""

    def __init__(self, view_columns, n_heads, *, normalise: bool = True, **kwargs):
        super().__init__(view_columns, n_heads, **kwargs)
        from morpheus_encoder.hypothesis_generator import HardConcreteGate

        init_open = kwargs.get("init_open", 0.9)
        self.head_gates = nn.ModuleList(
            nn.ModuleList(HardConcreteGate(len(c), init_open) for c in self.view_columns) for _ in range(n_heads))
        self.gates = self.head_gates[0]
        self._weights = _view_weights(self.view_columns, normalise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [x[:, getattr(self, f"_cols{m}")] for m in range(len(self.view_columns))]
        outs = []
        for k, gates in enumerate(self.head_gates):
            zs = [gate(v) for gate, v in zip(gates, xs)]
            hs = [tower(z) for tower, z in zip(self.towers, zs)]
            o = sum(head(h)[:, k] + lin(z)[:, k] for head, lin, h, z in zip(self.main, self.linear, hs, zs))
            for p, (m, n) in enumerate(self.pairs):
                o = o + ((zs[m] @ self.P[p]) * (zs[n] @ self.Q[p])) @ self.mix[p][:, k]
            outs.append(o + self.higher(torch.cat(hs, dim=-1))[:, k])
        return torch.stack(outs, dim=1)

    def sparsity_penalty(self) -> torch.Tensor:
        total = sum(w * g.penalty() for gates in self.head_gates for w, g in zip(self._weights, gates))
        return total / len(self.head_gates)

    def openness(self, width: int, k: int | None = None) -> Array:
        if k is None:
            return np.max([self.openness(width, j) for j in range(len(self.head_gates))], axis=0)
        out = np.zeros(width)
        for gate, cols in zip(self.head_gates[k], self.view_columns):
            s = torch.sigmoid(gate.log_alpha) * (gate.ZETA - gate.GAMMA) + gate.GAMMA
            out[cols.numpy()] = s.clamp(0.0, 1.0).detach().cpu().numpy()
        return out


def _train_patient_r2(model: nn.Module, X: torch.Tensor, r: torch.Tensor, train: Array, stop: Array,
                      stop_groups: Array, cfg: GeneratorConfig, seed: int) -> tuple[nn.Module, float]:
    """``hypothesis_generator._train`` (mini-batch path) with early stopping on the
    mean over held-out patients and heads of R^2 against the zero prediction,
    1 - SSE_p / sum r_p^2, instead of the pooled squared error.  The untrained
    network scores 0, so a fit is kept only if it beats zero on that average.
    Returns the best NEGATIVE mean R^2 (minimised)."""
    torch.manual_seed(seed)
    train_t = torch.as_tensor(train, device=X.device)
    stop_t = torch.as_tensor(stop, device=X.device)
    masks = [torch.as_tensor(stop_groups == g, device=X.device) for g in np.unique(stop_groups)]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    r_stop = r[stop_t]
    denom = [(r_stop[m] ** 2).sum(0).clamp_min(1e-12) for m in masks]

    def score() -> float:
        model.eval()
        with torch.no_grad():
            pred = model(X[stop_t])
        r2 = torch.stack([1 - ((pred[m] - r_stop[m]) ** 2).sum(0) / d for m, d in zip(masks, denom)])
        return -float(r2.mean())

    best = score()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for phase in range(model.n_phases):
        optimiser = torch.optim.AdamW(model.param_groups(cfg, phase))
        stale = 0
        epochs = cfg.max_epochs if phase == 0 else max(cfg.max_epochs // 2, 1)
        for epoch in range(epochs):
            model.train()
            order = train_t[torch.randperm(len(train_t), generator=generator).to(X.device)]
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start:start + cfg.batch_size]
                optimiser.zero_grad()
                loss = ((model(X[batch]) - r[batch]) ** 2).mean() + cfg.sparsity * model.sparsity_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimiser.step()
            current = score()
            if current < best - 1e-5:
                best, stale = current, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch >= cfg.min_epochs:
                stale += 1
                if stale >= cfg.patience:
                    break
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def multitask_loss(prediction: torch.Tensor, residuals: torch.Tensor,
                   head_weights: torch.Tensor | None = None) -> torch.Tensor:
    """sum_k w_k MSE_k on unit-variance residuals; default w_k = 1/K, which is the
    ``_train`` objective (mean squared error over all heads)."""
    mse = ((prediction - residuals) ** 2).mean(0)
    if head_weights is None:
        return mse.mean()
    return (head_weights * mse).sum()


# --------------------------------------------------------------------------
# Nomination
# --------------------------------------------------------------------------

def _gradient_input(net: ResidualBankNetwork, X_eval: torch.Tensor) -> Array:
    """mean |x_j d r_hat_k / d x_j| over rows, per head: (K, p)."""
    Xg = X_eval.clone().requires_grad_(True)
    out = net(Xg)
    rows = []
    for k in range(out.shape[1]):
        g, = torch.autograd.grad(out[:, k].sum(), Xg, retain_graph=k + 1 < out.shape[1])
        rows.append((g * Xg).abs().mean(0).detach().cpu().numpy())
    return np.vstack(rows)


def _aggregate_head(attributions: Sequence, cfg: GeneratorConfig, n_fits: int) -> tuple[dict, dict]:
    """Mean main-effect share per single and interaction share per pair over fits."""
    singles: dict[int, list[float]] = {}
    pairs: dict[tuple[int, int], list[float]] = {}
    for a in attributions:
        load: dict[int, float] = {}
        for (i, j), share in a.pair_share.items():
            pairs.setdefault((i, j), []).append(share)
            if share >= cfg.min_interaction and a.pair_ratio[(i, j)] >= cfg.min_interaction_ratio:
                load[i] = load.get(i, 0.0) + share
                load[j] = load.get(j, 0.0) + share
        for c, imp in a.importance.items():
            singles.setdefault(c, []).append(max(imp - load.get(c, 0.0), 0.0))
    s = {c: {"score": float(np.sum(v) / n_fits), "frequency": len(v) / n_fits} for c, v in singles.items()}
    p = {k: {"score": float(np.sum(v) / n_fits), "frequency": len(v) / n_fits} for k, v in pairs.items()}
    return s, p


def _head_list(singles: dict, pairs: dict, budget: int, reserve_share: float) -> list[tuple[tuple[int, ...], dict]]:
    s = sorted(((c,), v) for c, v in singles.items() if v["score"] > 0)
    p = sorted((k, v) for k, v in pairs.items() if v["score"] > 0)
    s.sort(key=lambda kv: (-kv[1]["score"], kv[0]))
    p.sort(key=lambda kv: (-kv[1]["score"], kv[0]))
    reserve = min(len(p), max(1, int(budget * reserve_share))) if p else 0
    chosen = p[:reserve] + s[:budget - reserve]
    if len(chosen) < budget:
        chosen += p[reserve:reserve + budget - len(chosen)]
    chosen.sort(key=lambda kv: (-kv[1]["score"], kv[0]))
    return chosen[:budget]


def residual_pair_screen(Xs: torch.Tensor, e: Array, *, top: int, proxy: float, partial: bool = False,
                         cap_per_column: int = 0) -> dict[tuple[int, int], float]:
    """Top ``top`` column pairs (``Xs``: rows x p) by |corr(e, x_i x_j)|^2, or with ``partial``
    by the squared partial correlation of x_i x_j with e given x_i and x_j (closed form
    from moment matrices: one pass, every pair).  ``cap_per_column`` > 0 keeps at most
    that many pairs per column (greedy in score order)."""
    n, p = Xs.shape
    if p < 2:
        return {}
    Xc = Xs - Xs.mean(0)
    et = torch.as_tensor((e - e.mean()) / max(float(e.std()), 1e-12), dtype=Xs.dtype, device=Xs.device)
    cov_pe = (Xc * et[:, None]).T @ Xc / n                       # cov(x_i x_j, e)
    gram = Xc.T @ Xc / n                                          # E[x_i x_j] = mean of the product
    var_p = ((Xc ** 2).T @ (Xc ** 2) / n - gram ** 2).clamp_min(1e-9)
    if not partial:
        c2 = cov_pe ** 2 / var_p
    else:
        A = (Xc ** 2).T @ Xc / n                                  # A[i, j] = cov(x_i x_j, x_i)
        a1, a2 = A, A.T                                           # cov(p_ij, x_i), cov(p_ij, x_j)
        b = Xc.T @ et / n                                         # cov(x_i, e)
        sv = torch.diagonal(gram)
        si, sj = sv[:, None], sv[None, :]
        bi, bj = b[:, None], b[None, :]
        det = (si * sj - gram ** 2).clamp_min(1e-9)
        aGb = (a1 * (sj * bi - gram * bj) + a2 * (si * bj - gram * bi)) / det
        aGa = (sj * a1 ** 2 - 2 * gram * a1 * a2 + si * a2 ** 2) / det
        bGb = (sj * bi ** 2 - 2 * gram * bi * bj + si * bj ** 2) / det
        c2 = (cov_pe - aGb) ** 2 / ((var_p - aGa).clamp_min(1e-9) * (1.0 - bGb).clamp_min(1e-9))
    sd = torch.diagonal(gram).clamp_min(1e-12).sqrt()
    corr = gram / torch.outer(sd, sd)
    c2 = torch.where(corr.abs() >= proxy, torch.zeros_like(c2), c2)
    c2 = torch.triu(torch.nan_to_num(c2, nan=0.0), diagonal=1)
    k = min(top if cap_per_column <= 0 else max(top * 64, 4096), p * (p - 1) // 2)
    vals, idx = torch.topk(c2.flatten(), k)
    out: dict[tuple[int, int], float] = {}
    count: dict[int, int] = {}
    for v, i in zip(vals.tolist(), idx.tolist()):
        if v <= 0 or len(out) >= top:
            break
        a, b_ = int(i // p), int(i % p)
        if cap_per_column > 0 and (count.get(a, 0) >= cap_per_column or count.get(b_, 0) >= cap_per_column):
            continue
        out[(a, b_)] = float(v)
        count[a], count[b_] = count.get(a, 0) + 1, count.get(b_, 0) + 1
    return out


@dataclass
class BankResult:
    candidates: tuple[Candidate, ...]               # merged across heads (round robin by rank), <= max_candidates
    per_level: dict[str, tuple[Candidate, ...]]     # each head's ranked list
    report: dict
    bank: ResidualBank | None = None                # the cross-fitted residuals the heads were trained on


def generate_residual_bank_hypotheses(X: Any, y: Any, groups: Any | None = None, *, comparison: Any | None,
                                      comparison_names: Sequence[str] | None = None,
                                      levels: Mapping[str, Sequence[int]] | None = None,
                                      views: Mapping[str, Sequence[int]] | None = None,
                                      roles: Mapping[str, str] | None = None,
                                      column_names: Sequence[str] | None = None, max_candidates: int = 32,
                                      seed: int = 0, device: str | None = None,
                                      config: ResidualBankConfig | None = None) -> BankResult:
    """Nominate up to ``max_candidates`` singles / products per explanation level.

    ``X``, ``y``, ``groups`` and ``comparison`` hold proposal rows only.  Levels
    come from ``levels`` (level -> comparison column indices), else from
    ``comparison_names`` through Agent D's ladder, else two heads: no
    comparison ("N") and the whole supplied comparison ("full").  Views outside
    ``config.protected_views`` are never read (``views=None``: all columns, one view).
    ``roles`` is accepted for interface compatibility; every protected view is a
    variable view.
    """
    started = time.time()
    bank_cfg = config or ResidualBankConfig()
    cfg = bank_cfg.generator
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and align with y")
    n, p = X.shape
    names = list(column_names) if column_names is not None else [f"x{j}" for j in range(p)]
    all_views = {k: np.asarray(v, dtype=int) for k, v in (views or {"raw_genes": range(p)}).items()}
    used_views = {k: v for k, v in all_views.items() if k in bank_cfg.protected_views and len(v)}
    if not used_views:
        raise ValueError(f"no protected view among {sorted(all_views)}")
    if levels is None:
        if comparison_names is not None:
            levels = comparison_levels(comparison_names, bank_cfg.levels)
        else:
            width = 0 if comparison is None else np.asarray(comparison).reshape(n, -1).shape[1]
            levels = {"N": np.arange(0), "full": np.arange(width)}
    level_names = tuple(levels)
    bank = residual_bank(comparison, y, groups, levels, n_splits=cfg.inner_folds, seed=seed,
                         library=bank_cfg.residual_library)
    R_raw = bank.residuals
    R = R_raw - R_raw.mean(0)
    R = np.clip(R / np.maximum(R.std(0), 1e-12), -8.0, 8.0)
    K = R.shape[1]

    view_list = list(used_views)
    searchable = np.sort(np.concatenate([used_views[v] for v in view_list]))
    view_of = np.empty(p, dtype=object)
    for v in view_list:
        view_of[used_views[v]] = v
    constant = X.std(0) < 1e-12
    dev = _device(device)
    X_t = torch.as_tensor(_standardise(X), dtype=torch.float32, device=dev)
    R_t = torch.as_tensor(R, dtype=torch.float32, device=dev)
    group_arr = np.asarray(groups) if groups is not None else np.arange(n) % 10
    unique_groups = np.unique(group_arr)
    rng = np.random.default_rng(seed)
    reference_rows = np.sort(rng.choice(n, size=min(cfg.reference_rows, n), replace=False))
    n_stop = max(1, int(round(cfg.stop_fraction * len(unique_groups))))
    group_order = np.random.default_rng(seed + 104729).permutation(unique_groups)
    attr_cfg = replace(cfg, min_importance=0.0, top_k=bank_cfg.pair_pool)
    per_head_attr: list[list] = [[] for _ in range(K)]
    heldout_pred_all = np.zeros_like(R)  # each row's head predictions from a fit that held its patient out
    fit_reports = []
    n_parameters = 0
    for fit in range(cfg.n_fits):
        fit_seed = seed * 1009 + fit
        fit_rng = np.random.default_rng(fit_seed)
        width = min(n_stop, len(unique_groups) - 1) or 1
        stop_groups = group_order[(fit * width + np.arange(width)) % len(unique_groups)]
        stop = np.flatnonzero(np.isin(group_arr, stop_groups))
        train = np.flatnonzero(~np.isin(group_arr, stop_groups))
        eval_rows = stop if len(stop) <= cfg.attribution_rows else np.sort(
            fit_rng.choice(stop, size=cfg.attribution_rows, replace=False))
        torch.manual_seed(fit_seed)
        kwargs = dict(hidden=cfg.hidden, rank=cfg.rank, dropout=cfg.dropout, gate=cfg.gate,
                      init_open=cfg.gate_init_open)
        view_cols = [used_views[v] for v in view_list]
        if bank_cfg.head_gates:
            net = HeadGatedBankNetwork(view_cols, K, normalise=bank_cfg.l0_width_normalised, **kwargs).to(dev)
        elif bank_cfg.l0_width_normalised:
            net = WidthNormalisedBankNetwork(view_cols, K, **kwargs).to(dev)
        else:
            net = ResidualBankNetwork(view_cols, K, **kwargs).to(dev)
        n_parameters += sum(p_.numel() for p_ in net.parameters())
        fit_started = time.time()
        zero_loss = float((R[stop] ** 2).mean())
        if bank_cfg.stop_metric == "loss":
            net, stop_loss = _train(net, X_t, R_t, train, stop, cfg, fit_seed)
            left_initial = stop_loss < zero_loss - 1e-5
        elif bank_cfg.stop_metric == "patient_r2":
            net, stop_score = _train_patient_r2(net, X_t, R_t, train, stop, group_arr[stop], cfg, fit_seed)
            left_initial = stop_score < -1e-5
        else:
            raise ValueError("stop_metric must be loss or patient_r2")
        train_seconds = time.time() - fit_started
        net.eval()
        openness_by_head = [net.openness(p, k) for k in range(K)]
        openness = np.max(openness_by_head, axis=0)
        gi = _gradient_input(net, X_t[torch.as_tensor(eval_rows, device=dev)])
        needed = torch.as_tensor(np.union1d(eval_rows, reference_rows), device=dev)
        with torch.no_grad():
            heldout_pred = net(X_t[torch.as_tensor(stop, device=dev)]).cpu().numpy()
        heldout_pred_all[stop] = heldout_pred
        r_stop = R[stop]
        improvement = 1.0 - ((r_stop - heldout_pred) ** 2).mean(0) / np.maximum((r_stop ** 2).mean(0), 1e-12)
        patient_r2 = {}
        for k, lv in enumerate(level_names):
            vals = []
            for g in np.unique(group_arr[stop]):
                m = group_arr[stop] == g
                vals.append(1.0 - float(((r_stop[m, k] - heldout_pred[m, k]) ** 2).sum())
                            / max(float((r_stop[m, k] ** 2).sum()), 1e-12))
            patient_r2[lv] = float(round(np.mean(vals), 4))
        fit_reports.append({"held_out_patients": [str(g) for g in stop_groups],
                            "open_columns": int((openness[searchable] > 0).sum()),
                            "open_columns_by_head": {lv: int((openness_by_head[k][searchable] > 0).sum())
                                                     for k, lv in enumerate(level_names)},
                            "left_initial_state": bool(left_initial),
                            "train_seconds": round(train_seconds, 1),
                            "heldout_mean_patient_r2": patient_r2,
                            "heldout_improvement_over_zero": {lv: float(round(v, 4))
                                                              for lv, v in zip(level_names, improvement)}})
        for k in range(K):
            open_k = openness_by_head[k]
            score = open_k * gi[k]
            cand = searchable[(open_k[searchable] > 0) & ~constant[searchable]]
            pool = cand[np.argsort(-score[cand], kind="stable")[:bank_cfg.attribution_pool]]

            def predict_k(Xm: torch.Tensor, _k=k, _net=net, _rows=needed) -> torch.Tensor:
                out = torch.zeros(len(Xm), device=Xm.device)
                sub = Xm[_rows]
                out[_rows] = torch.cat([_net(sub[s:s + 8192])[:, _k] for s in range(0, len(sub), 8192)])
                return out

            per_head_attr[k].append(_attribute(predict_k, X_t, R[:, k], eval_rows, reference_rows,
                                               np.sort(pool), attr_cfg, seed, f"head_{level_names[k]}",
                                               lambda i, j: True))

    # Per-head lists, then one Candidate per distinct structure with its set of levels.
    lists: dict[str, list] = {}
    pair_source: dict[tuple[str, tuple[int, int]], str] = {}
    if bank_cfg.pair_screen:
        Xs_search = X_t[:, torch.as_tensor(searchable, device=dev)]
    for k, level in enumerate(level_names):
        singles, pairs = _aggregate_head(per_head_attr[k], cfg, cfg.n_fits)
        for key in pairs:
            pair_source[(level, key)] = "network"
        if bank_cfg.pair_screen:
            screened = residual_pair_screen(Xs_search, R[:, k] - heldout_pred_all[:, k],
                                            top=bank_cfg.pair_screen_top, proxy=cfg.proxy_correlation,
                                            partial=bank_cfg.pair_screen_partial,
                                            cap_per_column=bank_cfg.pair_cap_per_column)
            for (i, j), share in screened.items():
                key = (int(searchable[i]), int(searchable[j]))
                if key not in pairs or pairs[key]["score"] < share:
                    pairs[key] = {"score": share, "frequency": 1.0}
                    pair_source[(level, key)] = "residual_pair_screen"
        lists[level] = _head_list(singles, pairs, max_candidates, bank_cfg.pair_reserve)
    nominated: dict[tuple[int, ...], dict] = {}
    for level, entries in lists.items():
        for rank, (cols, stats) in enumerate(entries, start=1):
            e = nominated.setdefault(cols, {"levels": [], "rank": {}, "score": {}, "frequency": {}, "source": {}})
            e["levels"].append(level)
            if len(cols) == 2:
                e["source"][level] = pair_source.get((level, cols), "network")
            e["rank"][level], e["score"][level], e["frequency"][level] = rank, stats["score"], stats["frequency"]
    cost = {"fits": int(cfg.n_fits), "heads": K, "parameters": int(n_parameters), "device": str(dev)}

    def to_candidate(cols: tuple[int, ...]) -> Candidate:
        e = nominated[cols]
        provenance = {"proposer": "residual_bank", "levels": list(e["levels"]),
                      "hypothesis_kind": "single" if len(cols) == 1 else "interaction",
                      "rank_by_level": dict(e["rank"]),
                      "attributed_residual_variance_by_level": {k: float(round(v, 5)) for k, v in e["score"].items()},
                      "inclusion_frequency_by_level": {k: float(round(v, 4)) for k, v in e["frequency"].items()},
                      "views": sorted({str(view_of[c]) for c in cols}),
                      "column_names": [str(names[c]) for c in cols], "seed_fits": int(cfg.n_fits), "cost": cost,
                      **({"pair_source_by_level": dict(e["source"])} if len(cols) == 2 else {})}
        common = dict(proposal_family="residual_bank", proposal_score=float(round(max(e["score"].values()), 6)),
                      metadata={"proposer": "residual_bank", "provenance": provenance})
        if len(cols) == 1:
            return Candidate(cols, weights=(1.0,), feature_form="linear_subspace", **common)
        return Candidate(tuple(sorted(cols)), feature_form="product", **common)

    built = {cols: to_candidate(cols) for cols in nominated}
    per_level = {level: tuple(built[cols] for cols, _ in entries) for level, entries in lists.items()}
    merged: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()
    for rank in range(max((len(v) for v in lists.values()), default=0)):
        for level in level_names:
            if rank < len(lists[level]):
                cols = lists[level][rank][0]
                if cols not in seen:
                    seen.add(cols)
                    merged.append(built[cols])
    report = {
        "proposer": "residual_bank",
        "levels": {lv: {"columns": [int(c) for c in levels[lv]], **bank.info[lv]} for lv in level_names},
        "views": {v: int(len(c)) for v, c in used_views.items()},
        "views_not_read": sorted(set(all_views) - set(used_views)),
        "fits": fit_reports,
        "n_nominated_by_level": {lv: len(v) for lv, v in per_level.items()},
        "n_distinct": len(nominated),
        "n_multi_level": sum(1 for e in nominated.values() if len(e["levels"]) > 1),
        "config": bank_cfg.to_dict(),
        "cost": {**cost, "wall_clock_seconds": round(time.time() - started, 2)},
    }
    return BankResult(tuple(merged[:max_candidates]), per_level, report, bank)


__all__ = ["LEVELS", "PROTECTED_VIEWS", "BankResult", "HeadGatedBankNetwork", "ResidualBank", "ResidualBankConfig",
           "ResidualBankNetwork", "WidthNormalisedBankNetwork",
           "comparison_levels", "generate_residual_bank_hypotheses", "multitask_loss", "residual_bank",
           "residual_pair_screen"]
