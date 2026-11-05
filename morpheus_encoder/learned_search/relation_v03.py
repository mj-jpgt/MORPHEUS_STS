"""Fixed relation family, baselines, gains and transfer diagnostics (spec 0.3, 0.7, 0.8, 0.10, 1.2, 1.4).

* Relation f_S: ridge on [256 random Fourier features of S (Gaussian kernel, median-heuristic
  bandwidth on support) | S linearly | 1], fitted in closed form on support rows to the target
  after the baseline f_B; penalty from ``ALPHAS`` by leave-one-patient-out (patient-grouped)
  CV on support.  Inputs standardised by support statistics.  The RFF directions are one fixed
  draw per set size, so no candidate identity enters.
* Baselines: ``nuis`` = N (log depth; per-patient intercepts are the within-patient centring of
  every target and covariate, which also absorbs study and platform); ``known`` = N + the
  Hallmark/Gavish programme scores + the composition variables; ``decomp`` = the best proper
  subset decomposition of a pair (f(A) + g(B), same family, fitted jointly).  Baseline ridges
  choose their penalty by LOPO CV.
* Gains: G = Q(f_B + f_S) - Q(f_B) on query rows (R^2 against the within-patient centred
  target); G_sup the same by LOPO inside support (baseline residuals cross-fitted); shrinkage
  Delta = G_sup - G.  Query gains are averaged over ``n_draws`` random query-patient subsets.

Everything is batched torch: per-patient Gram matrices once per chunk of sets, then batched
Cholesky solves for every penalty and held-out patient.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

D_RFF = 256
ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
BASE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
RFF_SEED = 20261021
N_DRAWS = 3
BASELINES = ("nuis", "known", "decomp")
LEVELS = ("L0", "L1", "L2")

DIAG_NAMES: tuple[str, ...] = (
    "g_sup",               # support LOPO gain of the single under B
    "g_sup_lopo_sd",       # sd over held-out patients of the per-patient gain
    "within_share",        # |within-patient cov| / (|within| + |between|) of candidate and target
    "moran_candidate",     # Moran's I on the section graph (mean over sections)
    "moran_target",
    "r_depth",             # Pearson r with log depth (within patient)
    "r_comp1", "r_comp2", "r_comp3",   # Pearson r with each composition variable (within patient)
    "n_support_patients",
    "sign_agreement",      # fraction of support patients whose association sign matches the pooled sign
)


def _dev(device) -> torch.device:
    return torch.device(device) if device is not None else torch.device("cpu")


def _dt(dev: torch.device) -> torch.dtype:
    return torch.float32 if dev.type == "cuda" else torch.float64


def _codes(groups) -> np.ndarray:
    return np.unique(np.asarray(groups).astype(str), return_inverse=True)[1].astype(np.int64)


def _onehot(codes: np.ndarray, dev, dtype) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.as_tensor(codes, device=dev), int(codes.max()) + 1).to(dtype).T


def _within(Z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return Z - S.T @ ((S @ Z) / S.sum(1).clamp(min=1)[:, None])


class _NoTF32:
    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


# ---------------------------------------------------------------- baseline fit
@dataclass
class BaselineFit:
    r_s: np.ndarray        # support residual of y after f_B (full-support fit)
    r_cv: np.ndarray       # support residual cross-fitted by leave-one-patient-out
    r_q: np.ndarray | None  # query residual (f_B from all support rows)
    alpha: float


def fit_baseline(ys, groups_s, C_s=None, *, yq=None, C_q=None, alphas=BASE_ALPHAS) -> BaselineFit:
    """Ridge of the (within-patient centred) target on within-patient centred covariates C
    (n, c); intercept free.  ``C=None`` is the per-patient-intercept model alone."""
    ys = np.asarray(ys, dtype=np.float64)
    if C_s is None or np.asarray(C_s).shape[1] == 0:
        return BaselineFit(ys - ys.mean(), ys - ys.mean(), None if yq is None else np.asarray(yq) - ys.mean(), 0.0)
    C = np.asarray(C_s, dtype=np.float64)
    mu, sd = C.mean(0), C.std(0)
    sd[sd < 1e-8] = 1.0
    Z = np.column_stack([(C - mu) / sd, np.ones(len(C))])
    codes = _codes(groups_s)
    P = codes.max() + 1
    pen = np.ones(Z.shape[1])
    pen[-1] = 1e-8
    G = [Z[codes == p].T @ Z[codes == p] for p in range(P)]
    b = [Z[codes == p].T @ ys[codes == p] for p in range(P)]
    Gt, bt = sum(G), sum(b)
    best, best_a, r_cv = np.inf, alphas[0], None
    for a in alphas:
        r = np.empty_like(ys)
        for p in range(P):
            beta = np.linalg.solve(Gt - G[p] + a * np.diag(pen), bt - b[p])
            r[codes == p] = ys[codes == p] - Z[codes == p] @ beta
        sse = float((r ** 2).sum())
        if sse < best:
            best, best_a, r_cv = sse, a, r
    beta = np.linalg.solve(Gt + best_a * np.diag(pen), bt)
    r_q = None
    if yq is not None:
        Zq = np.column_stack([(np.asarray(C_q, dtype=np.float64) - mu) / sd, np.ones(len(yq))])
        r_q = np.asarray(yq, dtype=np.float64) - Zq @ beta
    return BaselineFit(ys - Z @ beta, r_cv, r_q, float(best_a))


# ------------------------------------------------------------------- features
def _rff_dirs(k: int, seed: int = RFF_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000 * k)
    return rng.standard_normal((k, D_RFF)), rng.uniform(0, 2 * np.pi, D_RFF)


def median_bandwidth(Z: torch.Tensor, n_sub: int = 256, seed: int = 0) -> torch.Tensor:
    """(m,) median pairwise Euclidean distance among ``n_sub`` support rows, per set; Z (n, m, k)."""
    n = Z.shape[0]
    idx = torch.as_tensor(np.random.default_rng(seed).permutation(n)[:min(n, n_sub)], device=Z.device)
    S = Z[idx].transpose(0, 1)                                                # (m, s, k)
    d = torch.cdist(S, S)                                                     # (m, s, s)
    iu = torch.triu_indices(S.shape[1], S.shape[1], 1, device=Z.device)
    return d[:, iu[0], iu[1]].median(1).values.clamp(min=1e-3)


def features(Z: torch.Tensor, bw: torch.Tensor, blocks: Sequence[Sequence[int]] | None = None) -> torch.Tensor:
    """(n, m, k) standardised values -> (n, m, F): per block [256 RFF], then all k linearly, then 1.

    ``blocks=None``: one joint block (the relation f_S); a partition of range(k) gives the
    decomposition design (each block its own RFF map, same family)."""
    k = Z.shape[2]
    blocks = [list(range(k))] if blocks is None else [list(b) for b in blocks]
    parts = []
    for bi, b in enumerate(blocks):
        om, ph = _rff_dirs(len(b), RFF_SEED + 7 * bi if blocks is not None and len(blocks) > 1 else RFF_SEED)
        om = torch.as_tensor(om, dtype=Z.dtype, device=Z.device)
        ph = torch.as_tensor(ph, dtype=Z.dtype, device=Z.device)
        x = Z[:, :, b] / bw[None, :, None] if bw.ndim == 1 else Z[:, :, b] / bw[bi][None, :, None]
        parts.append(np.sqrt(2.0 / D_RFF) * torch.cos(x @ om + ph))
    parts += [Z, torch.ones(Z.shape[:2] + (1,), dtype=Z.dtype, device=Z.device)]
    return torch.cat(parts, 2)


# ------------------------------------------------------------------------ fit
def _spd_solve(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Batched Cholesky solve of SPD systems; a system that is numerically not positive definite
    in float32 (e.g. a near-constant candidate) is retried with a relative diagonal jitter, then
    solved in float64."""
    L, info = torch.linalg.cholesky_ex(A)
    x = torch.cholesky_solve(rhs, L)[..., 0]
    bad = (info > 0) | ~torch.isfinite(x).all(-1)
    if bool(bad.any()):
        Ab = A[bad].double()
        jit = 1e-5 * Ab.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-6)[..., None, None]
        Ab = Ab + jit * torch.eye(A.shape[-1], dtype=Ab.dtype, device=A.device)
        x[bad] = torch.linalg.solve(Ab, rhs[bad].double())[..., 0].to(x.dtype)
    return torch.nan_to_num(x)


def fit_sets(Vs, base: BaselineFit, ys, groups_s, *, Vq=None, yq=None, groups_q=None,
             blocks: Sequence[Sequence[int]] | None = None, alphas=ALPHAS, n_draws: int = N_DRAWS,
             draw_seed: int = 0, device=None, chunk: int | None = None, return_beta: bool = False) -> dict:
    """Fit m sets of k candidates (``Vs`` (n_s, m, k)) to the baseline residual.

    Returns ``g_sup`` (m,) pooled LOPO gain, ``lopo`` (P, m) per-patient gain, ``alpha`` (m,),
    and with query rows ``g`` (m,) query gain averaged over ``n_draws`` query-patient subsets
    (``g_full`` on all query rows, ``sse_q`` (m,) query SSE of f_B + f_S).

    ``return_beta``: also ``model`` (support standardisation, bandwidths, coefficients of every set,
    for ``predict`` on any rows) and, with query rows, ``pred_q`` (n_q, m) = f_S on the query rows
    (the full prediction is f_B + f_S = (yq - base.r_q) + pred_q)."""
    dev = _dev(device)
    with _NoTF32():
        return _fit(Vs, base, ys, groups_s, Vq, yq, groups_q, blocks, alphas, n_draws, draw_seed, dev, chunk,
                    return_beta)


def predict(model: dict, Vq, *, device=None, chunk: int = 64) -> np.ndarray:
    """(n, m) f_S predictions of fitted sets (``fit_sets(..., return_beta=True)["model"]``) on new rows
    ``Vq`` (n, m, k): the same support standardisation, bandwidth and coefficients."""
    dev = _dev(device)
    dt = _dt(dev)
    Vq = torch.as_tensor(np.asarray(Vq), device=dev).to(dt)
    Vq = Vq[:, :, None] if Vq.ndim == 2 else Vq
    mu, sd = (torch.as_tensor(model[k], device=dev, dtype=dt) for k in ("mu", "sd"))
    bw, beta = torch.as_tensor(model["bw"], device=dev, dtype=dt), torch.as_tensor(model["beta"], device=dev, dtype=dt)
    blocks = model["blocks"]
    out = []
    with _NoTF32():
        for a0 in range(0, Vq.shape[1], chunk):
            sl = slice(a0, min(Vq.shape[1], a0 + chunk))
            Z = (Vq[:, sl] - mu[sl]) / sd[sl]
            b = bw[sl] if blocks is None else bw[:, sl]
            out.append(torch.einsum("nmf,mf->nm", features(Z, b, blocks), beta[sl]))
    return torch.cat(out, 1).double().cpu().numpy()


def _fit(Vs, base, ys, groups_s, Vq, yq, groups_q, blocks, alphas, n_draws, draw_seed, dev, chunk,
         return_beta: bool = False) -> dict:
    dt = _dt(dev)
    codes = _codes(groups_s)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    P = len(bounds) - 1
    ys = np.asarray(ys, dtype=np.float64)
    sst_p = np.array([np.sum((ys[codes == p] - ys[codes == p].mean()) ** 2) for p in range(P)])
    Vs = torch.as_tensor(np.asarray(Vs) if not torch.is_tensor(Vs) else Vs, device=dev).to(dt)
    Vs = Vs[:, :, None] if Vs.ndim == 2 else Vs
    Vs = Vs[torch.as_tensor(order, device=dev)]
    n_s, m, k = Vs.shape
    r = torch.as_tensor(base.r_s[order], device=dev, dtype=dt)
    rcv = torch.as_tensor(base.r_cv[order], device=dev, dtype=dt)
    rr = torch.stack([(rcv[bounds[p]:bounds[p + 1]] ** 2).sum() for p in range(P)])
    mu, sd = Vs.mean(0), Vs.std(0, unbiased=False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    nb = 1 if blocks is None else len(blocks)
    F = D_RFF * nb + k + 1
    chunk = chunk or max(2, int(1.5e8 // max(1, n_s * F * len(alphas))))
    pen = torch.ones(F, dtype=dt, device=dev)
    pen[-1] = 1e-6
    eye = torch.diag(pen)
    al = torch.as_tensor(list(alphas), dtype=dt, device=dev)
    sst = torch.as_tensor(np.maximum(sst_p, 1e-12), dtype=dt, device=dev)
    out = {"g_sup": [], "lopo": [], "alpha": [], "g": [], "g_full": [], "sse_q": [], "gain_qp": []}
    keep = {"mu": [], "sd": [], "bw": [], "beta": [], "pred_q": []}
    if Vq is not None:
        Vq = torch.as_tensor(np.asarray(Vq) if not torch.is_tensor(Vq) else Vq, device=dev).to(dt)
        Vq = Vq[:, :, None] if Vq.ndim == 2 else Vq
        qcodes = _codes(groups_q)
        Sq = _onehot(qcodes, dev, dt)                                        # (Pq, n_q)
        yq = np.asarray(yq, dtype=np.float64)
        sstq_p = torch.as_tensor([np.sum((yq[qcodes == p] - yq[qcodes == p].mean()) ** 2)
                                  for p in range(qcodes.max() + 1)], dtype=dt, device=dev)
        rq = torch.as_tensor(base.r_q, dtype=dt, device=dev)
        rrq_p = Sq @ rq ** 2                                                 # (Pq,)
        Pq = len(sstq_p)
        rng = np.random.default_rng(draw_seed)
        size = max(1, int(round(2 * Pq / 3))) if Pq >= 2 else 1
        draws = torch.as_tensor(np.stack([np.isin(np.arange(Pq), rng.choice(Pq, size, replace=False))
                                          for _ in range(n_draws)]), dtype=dt, device=dev)   # (D, Pq)
    for a0 in range(0, m, chunk):
        sl = slice(a0, min(m, a0 + chunk))
        Z = (Vs[:, sl] - mu[sl]) / sd[sl]
        if blocks is None:
            bw = median_bandwidth(Z)
        else:
            bw = torch.stack([median_bandwidth(Z[:, :, list(b)]) for b in blocks])
        Phi = features(Z, bw, blocks)
        G = torch.stack([torch.einsum("nmf,nmg->mfg", Phi[bounds[p]:bounds[p + 1]], Phi[bounds[p]:bounds[p + 1]])
                         for p in range(P)])                                  # (P, mc, F, F)
        b = torch.stack([torch.einsum("nmf,n->mf", Phi[bounds[p]:bounds[p + 1]], r[bounds[p]:bounds[p + 1]])
                         for p in range(P)])                                  # train-side moments
        c = torch.stack([torch.einsum("nmf,n->mf", Phi[bounds[p]:bounds[p + 1]], rcv[bounds[p]:bounds[p + 1]])
                         for p in range(P)])                                  # held-out-side moments
        del Phi
        Gt, bt = G.sum(0), b.sum(0)
        A = (Gt[None, None] - G[None]) + al[:, None, None, None, None] * eye   # (A, P, mc, F, F)
        rhs = (bt[None] - b)[None, ..., None].expand(len(al), -1, -1, -1, -1)
        beta = _spd_solve(A, rhs)
        sse = rr[None, :, None] - 2 * (beta * c[None]).sum(-1) + torch.einsum("apmf,pmfg,apmg->apm", beta, G, beta)
        lopo = rr[None, :, None] - sse                                       # (A, P, mc)
        best = lopo.sum(1).argmax(0)
        lb = lopo.gather(0, best[None, None, :].expand(1, P, -1))[0]
        out["g_sup"].append(lb.sum(0) / sst.sum())
        out["lopo"].append(lb / sst[:, None])
        out["alpha"].append(al[best])
        if Vq is not None or return_beta:
            bfull = _spd_solve(Gt + al[best][:, None, None] * eye, bt[..., None])
        if return_beta:
            keep["mu"].append(mu[sl]), keep["sd"].append(sd[sl]), keep["bw"].append(bw), keep["beta"].append(bfull)
        if Vq is not None:
            Zq = (Vq[:, sl] - mu[sl]) / sd[sl]
            pred = torch.einsum("nmf,mf->nm", features(Zq, bw, blocks), bfull)
            if return_beta:
                keep["pred_q"].append(pred)
            sse_p = Sq @ ((rq[:, None] - pred) ** 2)                         # (Pq, mc)
            gain_p = rrq_p[:, None] - sse_p
            out["g"].append(((draws @ gain_p) / (draws @ sstq_p)[:, None]).mean(0))
            out["g_full"].append(gain_p.sum(0) / sstq_p.sum())
            out["sse_q"].append(sse_p.sum(0))
            out["gain_qp"].append(gain_p)
    f64 = lambda xs, d=0: torch.cat(xs, d).double().cpu().numpy() if xs else None  # noqa: E731
    res = {"g_sup": f64(out["g_sup"]), "lopo": f64(out["lopo"], 1), "alpha": f64(out["alpha"]),
           "g": f64(out["g"]), "g_full": f64(out["g_full"]), "sse_q": f64(out["sse_q"]),
           "gain_qp": f64(out["gain_qp"], 1)}
    if Vq is not None:   # per query patient pieces (harness bootstrap): G = sum gain_qp / sum sst_qp
        res["sst_qp"] = sstq_p.double().cpu().numpy()
        res["qpatients"] = np.unique(np.asarray(groups_q).astype(str))
    if return_beta:
        res["model"] = {"mu": f64(keep["mu"]), "sd": f64(keep["sd"]), "beta": f64(keep["beta"]),
                        "bw": f64(keep["bw"], 0 if blocks is None else 1), "blocks": blocks}
        if Vq is not None:
            res["pred_q"] = f64(keep["pred_q"], 1)
    return res


# ------------------------------------------------------------------ the task
@dataclass
class Task:
    """Support/query rows of one target: candidate values X (n, p), support rows first."""
    X: np.ndarray
    y: np.ndarray                 # (n,) within-patient centred target
    n_support: int
    groups: np.ndarray            # (n,) patient codes
    depth: np.ndarray | None      # (n,) log depth (within-patient centred)
    known: np.ndarray | None      # (n, c) programme scores + composition (within-patient centred)
    composition: np.ndarray | None  # (n, 3)
    nb6: np.ndarray | None = None  # (n, 6) local neighbours
    sections: np.ndarray | None = None
    within_share: np.ndarray | None = None  # (p,) |within| / (|within| + |between|) patient covariance share
    known_names: list | None = None   # names of the ``known`` columns ("mp:..." / "hallmark:..." programmes, "composition:j")
    target_name: str | None = None    # "programme:<name>" | "gene:<symbol>" | "resid:..." | "morph:..."
    panel: Sequence[str] | None = None  # genes the programme scores are computed from (overlap rule)

    @property
    def s(self) -> int:
        return int(self.n_support)


OVERLAP_THRESHOLD = 0.2   # hest_multi.gene_set_overlap_columns: |A & B| / min(|A|, |B|)


def known_exclusions(known_names: Sequence[str], target: str | None, panel: Sequence[str] | None = None) -> list:
    """Indices of ``known`` columns removed from B_known for this target (spec 0.8 B_known = nuisance +
    Hallmark / Gavish programme scores + composition, never containing the target): a programme target
    drops its own score and every programme whose gene set overlaps it (overlap coefficient >= 0.2 on
    the scoring panel, the hest_multi rule); a gene target drops every programme containing the gene;
    residual programmes and morphology targets drop nothing."""
    if not target or not known_names:
        return []
    from morpheus.tier2.substrates import hest_multi as hm
    kind, _, name = str(target).partition(":")
    prog_idx = [i for i, n in enumerate(known_names) if not str(n).startswith("composition:")]
    pnames = [str(known_names[i]) for i in prog_idx]
    if kind == "programme":
        hits = hm.gene_set_overlap_columns(name, pnames, panel if panel is not None else
                                           set().union(*[set(hm._programme_members(n)) for n in pnames + [name]]),
                                           threshold=OVERLAP_THRESHOLD)
        drop = set(hits) | {k for k, n in enumerate(pnames) if n == name}
    elif kind == "gene":
        drop = {k for k, n in enumerate(pnames) if name in set(hm._programme_members(n))}
    else:
        drop = set()
    return sorted(prog_idx[k] for k in drop)


def known_covariates(task: Task) -> tuple:
    """(known matrix without the excluded columns, dropped column names) for ``task``."""
    if task.known is None:
        return None, []
    K = np.asarray(task.known)
    drop = known_exclusions(task.known_names, task.target_name, task.panel) if task.known_names else []
    keep = np.setdiff1d(np.arange(K.shape[1]), drop)
    return K[:, keep], [str(task.known_names[i]) for i in drop]


def baseline_for(task: Task, B: str) -> BaselineFit:
    s = task.s
    if B == "known":
        known, _ = known_covariates(task)
        cols = [c for c in (task.depth[:, None] if task.depth is not None else None, known) if c is not None]
    else:   # nuis (decomp is relative to a pair's own decomposition, fitted on the nuis residual)
        cols = [task.depth[:, None]] if task.depth is not None else []
    C = np.column_stack(cols) if cols else None
    return fit_baseline(task.y[:s], task.groups[:s], None if C is None else C[:s], yq=task.y[s:],
                        C_q=None if C is None else C[s:])


def set_values(task: Task, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sets = np.asarray(sets, dtype=np.int64)
    sets = sets[:, None] if sets.ndim == 1 else sets
    return task.X[:task.s][:, sets], task.X[task.s:][:, sets]


def gains(task: Task, sets, B: str, *, base: BaselineFit | None = None, device=None, query: bool = True,
          draw_seed: int = 0) -> dict:
    """Labels of equally sized sets under baseline B (``nuis`` | ``known``)."""
    base = base or baseline_for(task, B)
    Vs, Vq = set_values(task, sets)
    s = task.s
    return fit_sets(Vs, base, task.y[:s], task.groups[:s], Vq=Vq if query else None, yq=task.y[s:],
                    groups_q=task.groups[s:], device=device, draw_seed=draw_seed)


def decomp_gains(task: Task, pairs, *, base: BaselineFit | None = None, full: dict | None = None, device=None,
                 draw_seed: int = 0) -> dict:
    """Higher-order labels of pairs: G(S, B_decomp(S)) = Q(f_S) - Q(f_A + g_B) (support LOPO and query)."""
    base = base or baseline_for(task, "nuis")
    Vs, Vq = set_values(task, pairs)
    s = task.s
    kw = dict(Vq=Vq, yq=task.y[s:], groups_q=task.groups[s:], device=device, draw_seed=draw_seed)
    full = full or fit_sets(Vs, base, task.y[:s], task.groups[:s], **kw)
    dec = fit_sets(Vs, base, task.y[:s], task.groups[:s], blocks=[[0], [1]], **kw)
    return {"g_sup": full["g_sup"] - dec["g_sup"], "g": full["g"] - dec["g"], "lopo": full["lopo"] - dec["lopo"],
            "gain_qp": full["gain_qp"] - dec["gain_qp"], "sst_qp": full["sst_qp"], "qpatients": full["qpatients"]}


# ---------------------------------------------------------------- diagnostics
def moran_columns(Z, nb6, sections, *, device=None) -> np.ndarray:
    """Moran's I per column, mean over sections with >= 7 rows (column-batched
    ``hest_multi.section_moran_i``; sections without neighbour pairs are skipped)."""
    from .support_statistics import moran_columns as mc
    return mc(Z, nb6, sections, device=device)


def diagnostics(task: Task, B: str, *, base: BaselineFit | None = None, single: dict | None = None,
                device=None) -> tuple[np.ndarray, dict]:
    """Fixed transfer diagnostics (``DIAG_NAMES``) of every candidate under B, support rows only.
    Returns (float32 [p, 11], the singles' fit dict)."""
    dev = _dev(device)
    s = task.s
    base = base or baseline_for(task, B)
    p = task.X.shape[1]
    if single is None:
        single = fit_sets(task.X[:s], base, task.y[:s], task.groups[:s], device=dev)
    codes = _codes(task.groups[:s])
    dt = torch.float64
    Sg = _onehot(codes, dev, dt)
    X = torch.as_tensor(task.X[:s], dtype=dt, device=dev)
    Zc = _within(X, Sg)
    yr = torch.as_tensor(base.r_s, dtype=dt, device=dev)
    zn = (Zc ** 2).sum(0).sqrt() + 1e-12
    out = torch.zeros((p, len(DIAG_NAMES)), dtype=dt, device=dev)
    col = {nm: i for i, nm in enumerate(DIAG_NAMES)}
    out[:, col["g_sup"]] = torch.as_tensor(single["g_sup"], device=dev)
    out[:, col["g_sup_lopo_sd"]] = torch.as_tensor(single["lopo"].std(0), device=dev)
    out[:, col["within_share"]] = (1.0 if task.within_share is None
                                   else torch.as_tensor(task.within_share, dtype=dt, device=dev))
    if task.nb6 is not None:
        nb = np.asarray(task.nb6)[:s]
        nb = np.where(nb < s, nb, -1)
        sec = codes if task.sections is None else _codes(np.asarray(task.sections)[:s])
        out[:, col["moran_candidate"]] = torch.as_tensor(moran_columns(X, nb, sec, device=dev), device=dev)
        out[:, col["moran_target"]] = float(moran_columns(yr[:, None], nb, sec, device=dev)[0])

    def corr(v):
        v = _within(torch.as_tensor(np.asarray(v, dtype=np.float64), device=dev)[:, None], Sg)[:, 0]
        return (Zc.T @ v) / (zn * (v.norm() + 1e-12))

    if task.depth is not None:
        out[:, col["r_depth"]] = corr(task.depth[:s])
    if task.composition is not None:
        for j in range(min(3, task.composition.shape[1])):
            out[:, col[f"r_comp{j + 1}"]] = corr(task.composition[:s, j])
    out[:, col["n_support_patients"]] = float(codes.max() + 1)
    rp = (Sg @ (Zc * yr[:, None]))                                            # per-patient covariance sign
    pooled = (Zc * yr[:, None]).sum(0)
    out[:, col["sign_agreement"]] = (torch.sign(rp) == torch.sign(pooled)[None]).double().mean(0)
    res = np.nan_to_num(out.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return res, single


def label_key(study: str, support_patients, query_patients, target: str, B: str, L: str, sets=None) -> str:
    """sha256 key of (study, support patients, query patients, target spec, B, L[, sets])."""
    h = hashlib.sha256()
    for part in (study, ",".join(sorted(map(str, support_patients))), ",".join(sorted(map(str, query_patients))),
                 target, B, L):
        h.update(str(part).encode())
        h.update(b"|")
    if sets is not None:
        h.update(np.ascontiguousarray(np.asarray(sets, dtype=np.int64)).tobytes())
    return h.hexdigest()
