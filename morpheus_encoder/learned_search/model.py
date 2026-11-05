"""The learned search model: tokens -> set encoder -> selector -> relation fit.

Given the support patients of a task, the model chooses which small set (1-4)
of candidate variables (molecular, morphology concept, spatial-context
transforms) is worth testing.  Nothing names a relation shape: the selected
values are passed to :class:`RelationFit`, a closed-form random-Fourier-feature
ridge solved on support rows, so the training loss is the query error of a
genuinely refitted relation and its gradient reaches the selector through the
one-hot selections (``V = X @ onehot``).

* :class:`SetEncoder`  4 pre-norm Transformer blocks over the candidate tokens
  (``nn.TransformerEncoderLayer(norm_first=True)``, scaled-dot-product
  attention), or an induced-set variant with 64 inducing points (``"isab"``).
* :class:`PointerSelector`  a GRU state and a dot-product pointer over tokens
  plus STOP; at each step the pointer keys are refreshed with *conditional
  probes*: a small network maps each support row's (selected values so far, y,
  current partial-fit residual) to 8 numbers, and each candidate's mean product
  with them (and with its square) is added to its key.  This is how the search
  conditions on what is already chosen, e.g. the second member of a pair with
  no marginal effect.  Straight-through Gumbel-softmax, up to ``k_max`` steps,
  ``n_paths`` sampled paths.
* :class:`TopKSelector`  the fallback: one score per token and
  ``morpheus_encoder.sparse.topk_gate`` with k = 1..k_max across paths.
* :class:`RelationFit`  joint fit (linear terms + RFF over all selected values)
  and every proper-subset decomposition (``decompositions(k)``: 1/4/14 for
  k = 2/3/4), each block with its own RFF map; ``g_ho`` = Q(joint) - max over
  decompositions of Q.
"""
from __future__ import annotations

import math
from itertools import combinations
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodes import TRANSFORMS, VIEW_VOCAB, Episode, TokenMeta, centre_scale, neighbour_operators, view_id
from .tokens import CandidateTokenEncoder, ConceptLayer, concept_transforms


@dataclass
class ModelConfig:
    d_model: int = 192
    n_heads: int = 6
    n_blocks: int = 4
    encoder_kind: str = "sab"      # "sab" | "isab"
    selector: str = "pointer"      # "pointer" | "topk"
    k_max: int = 4
    n_paths: int = 12
    n_concepts: int = 64
    n_samples: int = 256
    n_rff: int = 16
    ridge_alpha: float = 100.0   # measured on the kidney dev set: best held-out oracle R^2 (10 overfits)
    embed_dim: int = 1536
    concept_hidden: int = 256
    probe_m: int = 8
    n_inducing: int = 64
    obs_hidden: int = 64
    gene_cap: int = 2048   # propose-time cap on gene-view columns (response-free variance); training uses 768 HVG


# ------------------------------------------------------------------ partitions
def set_partitions(items: Sequence[int]) -> list:
    """Every partition of ``items`` into non-empty blocks."""
    items = list(items)
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    out = []
    for part in set_partitions(rest):
        out.append([[first]] + part)
        for i in range(len(part)):
            out.append(part[:i] + [[first] + part[i]] + part[i + 1:])
    return out


def decompositions(k: int) -> list:
    """Proper-subset decompositions of a k-set (partitions into >= 2 blocks): 1/4/14 for k = 2/3/4."""
    return [sorted(tuple(sorted(b)) for b in p) for p in set_partitions(range(k)) if len(p) >= 2]


# -------------------------------------------------------------------- relation
class RelationFit(nn.Module):
    """Closed-form RFF kernel ridge over the selected values (plus linear terms)."""

    def __init__(self, n_rff: int = 16, alpha: float = 10.0, k_max: int = 4, seed: int = 20261015):
        super().__init__()
        self.n_rff, self.alpha, self.k_max = n_rff, float(alpha), k_max
        g = torch.Generator().manual_seed(seed)
        for s in range(1, k_max + 1):
            for pos in range(k_max):
                D = n_rff * s
                ell = torch.where(torch.arange(D) % 2 == 0, torch.tensor(0.7), torch.tensor(1.5))
                self.register_buffer(f"om_{s}_{pos}", torch.randn(s, D, generator=g) / ell, persistent=False)
                self.register_buffer(f"ph_{s}_{pos}", torch.rand(D, generator=g) * 2 * math.pi, persistent=False)

    def features(self, Z: torch.Tensor, blocks: Sequence[Sequence[int]]) -> torch.Tensor:
        parts = [Z]
        for pos, b in enumerate(blocks):
            s = len(b)
            om, ph = getattr(self, f"om_{s}_{pos}"), getattr(self, f"ph_{s}_{pos}")
            parts.append(math.sqrt(2.0 / om.shape[1]) * torch.cos(Z[..., list(b)] @ om + ph))
        return torch.cat(parts, -1)

    def fit_predict(self, Vs: torch.Tensor, ys: torch.Tensor, Vt: torch.Tensor,
                    blocks: Sequence[Sequence[int]] | None = None) -> torch.Tensor:
        """Fit on (Vs, ys), predict at Vt.  Accepts a leading batch dimension
        (Vs (B, n, k), Vt (B, m, k), ys (n,) or (B, n)) -> (B, m)."""
        with torch.autocast(device_type=Vs.device.type, enabled=False):
            Vs, ys, Vt = Vs.float(), ys.float(), Vt.float()
            k = Vs.shape[-1]
            blocks = [tuple(range(k))] if blocks is None else blocks
            mu, sd = Vs.mean(-2, keepdim=True), Vs.std(-2, keepdim=True).clamp(min=1e-6)
            Ps, Pt = self.features((Vs - mu) / sd, blocks), self.features((Vt - mu) / sd, blocks)
            pm = Ps.mean(-2, keepdim=True)
            A = Ps - pm
            if Vs.ndim == 3 and ys.ndim == 1:
                ys = ys.expand(Vs.shape[0], -1)
            ym = ys.mean(-1, keepdim=True)
            At = A.transpose(-1, -2)
            G = At @ A + self.alpha * torch.eye(A.shape[-1], device=A.device)
            beta = torch.linalg.solve_ex(G, (At @ (ys - ym)[..., None]))[0][..., 0]  # SPD; no host sync
            return ((Pt - pm) @ beta[..., None])[..., 0] + ym

    @staticmethod
    def r2(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """R^2 over the last dimension (batched when ``pred`` has a leading batch dimension)."""
        y, pred = y.float(), pred.float()
        sst = ((y - y.mean(-1, keepdim=True)) ** 2).sum(-1).clamp(min=1e-8)
        return 1.0 - ((y - pred) ** 2).sum(-1) / sst

    def query_r2(self, Vs, ys, Vq, yq, blocks=None) -> torch.Tensor:
        return self.r2(yq, self.fit_predict(Vs, ys, Vq, blocks))

    def cv_r2(self, V: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, blocks=None, n_folds: int = 3,
              seed: int = 0) -> torch.Tensor:
        """Patient-grouped ``n_folds`` cross-validated R^2 (row folds if < n_folds patients)."""
        fold = _fold_ids(groups, n_folds, seed)
        pred = torch.zeros(V.shape[:-1], device=V.device, dtype=torch.float32)
        for f in torch.unique(fold):
            te = fold == f
            pred[..., te] = self.fit_predict(V[..., ~te, :], y[~te], V[..., te, :], blocks)
        return self.r2(y, pred)

    def g_ho(self, V_fit, y_fit, V_eval=None, y_eval=None, *, groups=None, n_folds: int = 3) -> torch.Tensor:
        """Q(joint) - max over proper-subset decompositions of Q (same RFF family).

        Q is the query R^2 when evaluation rows are given, else the grouped CV R^2
        on the fitting rows."""
        k = V_fit.shape[1]
        if k < 2:
            return torch.zeros((), device=V_fit.device)

        def Q(blocks):
            if V_eval is not None:
                return self.query_r2(V_fit, y_fit, V_eval, y_eval, blocks)
            return self.cv_r2(V_fit, y_fit, groups, blocks, n_folds)

        joint = Q([tuple(range(k))])
        best = torch.stack([Q(p) for p in decompositions(k)]).max()
        return joint - best


def _fold_ids(groups: torch.Tensor, n_folds: int, seed: int) -> torch.Tensor:
    uniq, inv = torch.unique(groups, return_inverse=True)
    gen = torch.Generator().manual_seed(seed)
    if len(uniq) >= n_folds:
        perm = torch.randperm(len(uniq), generator=gen).to(groups.device)
        return (perm % n_folds)[inv]
    return (torch.randperm(len(groups), generator=gen) % n_folds).to(groups.device)


# ---------------------------------------------------------------- set encoder
class _MAB(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.lq, self.lk, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, k):
        kk = self.lk(k)
        h = q + self.attn(self.lq(q), kk, kk, need_weights=False)[0]
        return h + self.ff(self.l2(h))


class _ISAB(nn.Module):
    def __init__(self, d: int, heads: int, m: int):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)
        self.m1, self.m2 = _MAB(d, heads), _MAB(d, heads)

    def forward(self, x):
        h = self.m1(self.I.expand(x.shape[0], -1, -1), x)
        return self.m2(x, h)


class SetEncoder(nn.Module):
    def __init__(self, d: int, heads: int, blocks: int, kind: str = "sab", n_inducing: int = 64):
        super().__init__()
        self.kind = kind
        if kind == "sab":
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.net = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        elif kind == "isab":
            self.net = nn.Sequential(*[_ISAB(d, heads, n_inducing) for _ in range(blocks)])
        else:
            raise ValueError(kind)
        self.norm = nn.LayerNorm(d)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(T[None]))[0]


# ------------------------------------------------------------------- selectors
def _probe_scale(n: int) -> float:
    return math.sqrt(n) / 10.0


class _Probe(nn.Module):
    """Per-row conditional probe vectors and each candidate's mean products with them."""

    def __init__(self, k_max: int, m: int):
        super().__init__()
        self.k_max = k_max
        self.net = nn.Sequential(nn.Linear(2 * k_max + 2, 32), nn.GELU(), nn.Linear(32, m))

    def forward(self, V: torch.Tensor, filled: torch.Tensor, ys: torch.Tensor, e: torch.Tensor,
                Xs: torch.Tensor, Xs2: torch.Tensor) -> torch.Tensor:
        # V (paths, n_s, k_max) support values; filled (paths, k_max); e (paths, n_s)
        with torch.autocast(device_type=Xs.device.type, enabled=False):
            paths, n_s, _ = V.shape
            inp = torch.cat([V.float(), filled.float()[:, None, :].expand(paths, n_s, self.k_max),
                             ys.float()[None, :, None].expand(paths, n_s, 1), e.float()[..., None]], -1)
            q = self.net(inp)                                    # (paths, n_s, m)
            s = _probe_scale(n_s) / n_s
            c1 = torch.einsum("pnm,nj->pjm", q, Xs) * s
            c2 = torch.einsum("pnm,nj->pjm", q, Xs2) * s
            return torch.cat([c1, c2], -1)                       # (paths, P, 2m)


def _gumbel_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    if generator is not None and generator.device != x.device:
        u = torch.rand(x.shape, generator=generator, device=generator.device).to(x.device)
    else:
        u = torch.rand(x.shape, generator=generator, device=x.device)
    u = u.clamp(1e-9, 1 - 1e-9)
    return -torch.log(-torch.log(u))


class FlexScreen(nn.Module):
    """For every candidate, the in-sample R^2 of a 1-D random-Fourier-feature ridge of
    the current residual ``e`` on that candidate alone (support rows), as a z-like
    score ``(n R^2 - d) / sqrt(2 d)``.  The per-candidate Gram inverses depend only on
    X, so they are computed once per episode; each pointer step then costs one
    batched product.  Detached: an input to the pointer keys, not a trained path."""

    def __init__(self, n_rff: int = 16, chunk: int = 512, seed: int = 7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("om", torch.randn(n_rff, generator=g), persistent=False)
        self.register_buffer("ph", torch.rand(n_rff, generator=g) * 2 * math.pi, persistent=False)
        self.chunk, self.alpha = chunk, 1.0

    def _phi(self, Xc: torch.Tensor) -> torch.Tensor:
        # Xc (n, c) -> (c, n, d) centred features [x, cos(x w + b)]
        F_ = torch.cat([Xc.T[..., None], torch.cos(Xc.T[..., None] * self.om + self.ph)], -1)
        return F_ - F_.mean(1, keepdim=True)

    @torch.no_grad()
    def prepare(self, Xs: torch.Tensor) -> list:
        out = []
        for a in range(0, Xs.shape[1], self.chunk):
            Fc = self._phi(Xs[:, a:a + self.chunk].float())
            G = Fc.transpose(1, 2) @ Fc + self.alpha * torch.eye(Fc.shape[-1], device=Fc.device)
            out.append(torch.linalg.inv_ex(G)[0])
        return out

    @torch.no_grad()
    def forward(self, Xs: torch.Tensor, e: torch.Tensor, ginv: list) -> torch.Tensor:
        # e (paths, n) -> (paths, P) z-like scores
        n = Xs.shape[0]
        e = e.float() - e.float().mean(1, keepdim=True)
        ee = (e * e).sum(1).clamp(min=1e-8)
        outs = []
        for i, a in enumerate(range(0, Xs.shape[1], self.chunk)):
            Fc = self._phi(Xs[:, a:a + self.chunk].float())          # (c, n, d)
            b = torch.einsum("cnd,pn->pcd", Fc, e)                     # (paths, c, d)
            q = torch.einsum("pcd,cde,pce->pc", b, ginv[i], b)
            outs.append(q / ee[:, None])
        r2 = torch.cat(outs, 1)
        d = 1 + self.om.numel()
        return (n * r2 - d) / math.sqrt(2 * d)


class InteractionScreen(nn.Module):
    """For every candidate x_j, evidence that the current residual ``e`` depends on x_j
    jointly with the last selected variable a: the scores of e on the products
    {x_j, x_j^2 - 1} x {a, a^2 - 1, cos(w_m a + b_m)} (heteroscedasticity-robust),
    summed as a chi-square with 2M degrees of freedom and returned as
    ``(s - 2M) / sqrt(4M)``.  Finds the partner of a pair with no marginal effect,
    which the 1-D screen cannot.  Four batched products per pointer step; detached."""

    def __init__(self, n_rff: int = 6, seed: int = 11):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("om", torch.randn(n_rff, generator=g), persistent=False)
        self.register_buffer("ph", torch.rand(n_rff, generator=g) * 2 * math.pi, persistent=False)

    @torch.no_grad()
    def forward(self, Xs: torch.Tensor, Xs2: torch.Tensor, e: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a = a.float()
        a = (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True).clamp(min=1e-6)
        psi = torch.cat([a[..., None], a[..., None] ** 2 - 1, torch.cos(a[..., None] * self.om + self.ph)], -1)
        psi = psi - psi.mean(1, keepdim=True)                       # (paths, n, M)
        e = e.float() - e.float().mean(1, keepdim=True)
        W = psi * e[..., None]
        W2 = psi ** 2 * (e ** 2)[..., None]
        s = 0.0
        for B in (Xs, Xs2):
            num = torch.einsum("nj,pnm->pjm", B, W) ** 2
            den = torch.einsum("nj,pnm->pjm", B * B, W2).clamp(min=1e-8)
            s = s + (num / den).sum(-1)
        dof = 2 * psi.shape[-1]
        return (s - dof) / math.sqrt(2 * dof)


class PointerSelector(nn.Module):
    def __init__(self, d: int, k_max: int = 4, probe_m: int = 8):
        super().__init__()
        self.d, self.k_max = d, k_max
        self.h0 = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.probe = _Probe(k_max, probe_m)
        self.wc = nn.Linear(2 * probe_m, d, bias=False)
        self.stop = nn.Linear(d, 1)
        self.inp = nn.Linear(d + 2 * probe_m, d)
        self.flex = FlexScreen()
        self.inter = InteractionScreen()
        self.wf = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, d, bias=False))
        # direct paths for the screen: a pointer prior towards candidates that explain the
        # current residual, and a STOP prior when no remaining candidate does (learned
        # slope / threshold; the attention and probe terms learn everything else)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(0.5))
        self.att_scale = nn.Parameter(torch.tensor(1.0))   # weight of the learned attention term
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.thr = nn.Parameter(torch.tensor(3.0))

    def forward(self, H, Xall, n_s, ys, allowed, relation: RelationFit, *, n_paths: int, tau: float,
                teacher: Sequence[int] | None = None, greedy_first: bool = True,
                generator: torch.Generator | None = None) -> dict:
        """``n_paths`` sampled paths; with ``teacher`` one extra path is forced along the
        planted set (set cross-entropy: at each step the remaining true tokens' total
        probability, then STOP) and only its loss ``ce`` is returned for it."""
        P = H.shape[0]
        dev = H.device
        T = n_paths if teacher is not None else None
        n_tot = n_paths + (1 if teacher is not None else 0)
        Xs = Xall[:n_s].float()
        Xs2 = Xs * Xs - 1.0
        K = self.wk(H)                                           # (P, d)
        h = torch.tanh(self.h0(H.mean(0)))[None].expand(n_tot, -1).contiguous()
        V = torch.zeros(n_tot, Xall.shape[0], self.k_max, device=dev)
        filled = torch.zeros(n_tot, self.k_max, dtype=torch.bool, device=dev)
        selected = torch.zeros(n_tot, P, dtype=torch.bool, device=dev)
        active = torch.ones(n_tot, dtype=torch.bool, device=dev)
        e = ys.float()[None].expand(n_tot, -1).clone()
        onehots: list[list] = [[] for _ in range(n_tot)]
        chosen: list[list] = [[] for _ in range(n_tot)]
        soft_acc = torch.zeros(n_tot, P, device=dev)
        logp_path = torch.zeros(n_tot, device=dev)
        ce = torch.zeros((), device=dev)
        logits0 = None
        truth = [int(j) for j in teacher] if teacher is not None else []
        ginv = self.flex.prepare(Xs)
        for t in range(self.k_max):  # STOP is implied after k_max variables
            c = self.probe(V[:, :n_s], filled, ys, e, Xs, Xs2)   # (paths, P, 2m)
            zf = self.flex(Xs, e, ginv)                           # (paths, P) residual flexible screen
            if t > 0:                                             # interaction with the last selected variable
                zi = self.inter(Xs, Xs2, e, V[:, :n_s, t - 1])
            else:
                zi = torch.zeros_like(zf)
            ec = e - e.mean(1, keepdim=True)
            zv = self.flex(Xs, ec * ec, ginv)                     # variance screen: partners of interactions
            fz = torch.stack([zf / 10.0, torch.log1p(F.relu(zf)), torch.log1p(F.relu(zi)),
                              torch.log1p(F.relu(zv))], -1)
            keys = K[None] + self.wc(c.to(K.dtype)) + self.wf(fz.to(K.dtype))
            zc = zf.clamp(-3.0, 30.0) / 3.0
            zic = zi.clamp(-3.0, 30.0) / 3.0
            lg = (self.att_scale * torch.einsum("pjd,pd->pj", keys, self.wq(h)) / math.sqrt(self.d) + self.beta * zc
                  + self.beta2 * zic + self.beta3 * zv.clamp(-3.0, 30.0) / 3.0)
            zmax = torch.maximum(zc, zic).masked_fill(~(allowed[None] & ~selected), -1.0).max(1, keepdim=True).values
            st = self.stop(h) + self.gamma * (self.thr / 3.0 - zmax)
            full = torch.cat([lg, st], 1).float()
            mask = torch.cat([allowed[None].expand(n_tot, -1) & ~selected,
                              torch.full((n_tot, 1), t > 0, device=dev)], 1)
            full = full.masked_fill(~mask, float("-inf"))
            if t == 0:
                logits0 = full[:n_paths, :P]
            g = _gumbel_like(full, generator)
            if greedy_first:
                g[0] = 0.0
            z = full + g
            y_soft = torch.softmax(z / tau, 1)
            idx = torch.argmax(z, 1)
            logsm = torch.log_softmax(full, 1)
            if T is not None:
                remaining = [j for j in truth if j not in chosen[T]]
                if remaining:
                    rem = torch.tensor(remaining, device=dev)
                    ce = ce - torch.logsumexp(logsm[T, rem], 0)
                    forced = rem[torch.argmax(logsm[T, rem])]
                elif t > 0 and bool(active[T]):
                    ce = ce - logsm[T, P]
                    forced = torch.tensor(P, device=dev)
                else:
                    forced = torch.tensor(P, device=dev)
                idx = idx.clone()
                idx[T] = forced
                y_soft = y_soft.clone()
                y_soft[T] = y_soft[T].detach()
            oh = F.one_hot(idx, P + 1).float() + (y_soft - y_soft.detach())
            samp = torch.ones(n_tot, device=dev)
            if T is not None:
                samp[T] = 0.0
            soft_acc = soft_acc + y_soft[:, :P] * (active.float() * samp)[:, None]
            lp = logsm.gather(1, idx[:, None])[:, 0].detach()
            logp_path = logp_path + torch.where(active, lp, torch.zeros_like(lp))
            stop_now = idx == P
            active_new = active & ~stop_now
            tok = oh[:, :P] * active_new[:, None].float()
            v = tok @ Xall.float().T                              # (paths, n)
            V = V.clone()
            V[:, :, t] = v
            filled = filled.clone()
            filled[:, t] = active_new
            idx_l, act_l = idx.tolist(), active_new.tolist()   # one host sync per step
            selected = selected | (F.one_hot(idx, P + 1)[:, :P].bool() & active_new[:, None])
            for p in range(n_tot):
                if act_l[p]:
                    onehots[p].append(tok[p])
                    chosen[p].append(idx_l[p])
            sel_c = torch.einsum("pj,pjm->pm", tok.to(c.dtype), c)
            h = self.gru(self.inp(torch.cat([(tok.to(H.dtype) @ H), sel_c.to(H.dtype)], 1)), h)
            active = active_new
            if not any(act_l):
                break
            with torch.no_grad():  # partial-fit residual on support rows; every active path has t + 1 variables
                Vp = V[:, :n_s, :t + 1].detach()
                e = torch.where(active[:, None], ys.float()[None] - relation.fit_predict(Vp, ys, Vp), e)
        # Support-only first-step diagnostics for bounded conditional proposal beams.
        initial_flex = zf[:n_paths].detach()
        initial_variance = zv[:n_paths].detach()
        return {"onehots": onehots[:n_paths], "chosen": chosen[:n_paths], "V": V[:n_paths],
                "soft": soft_acc[:n_paths], "ce": ce, "logits0": logits0, "logp": logp_path[:n_paths],
                "initial_flex": initial_flex, "initial_variance": initial_variance,
                "teacher_chosen": chosen[T] if T is not None else None}


class TopKSelector(nn.Module):
    """Fallback: one score per token, hard top-k with ``sparse.topk_gate``'s straight-through gradient."""

    def __init__(self, d: int, k_max: int = 4, probe_m: int = 8):
        super().__init__()
        self.k_max = k_max
        self.probe = _Probe(k_max, probe_m)
        self.wk = nn.Linear(d, d)
        self.wc = nn.Linear(2 * probe_m, d, bias=False)
        self.score = nn.Linear(d, 1)
        self.flex = FlexScreen()
        self.wf = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, d, bias=False))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, Xall, n_s, ys, allowed, relation, *, n_paths: int, tau: float,
                teacher: Sequence[int] | None = None, greedy_first: bool = True,
                generator: torch.Generator | None = None) -> dict:
        from morpheus_encoder.sparse import topk_gate

        P, dev = H.shape[0], H.device
        Xs = Xall[:n_s].float()
        V0 = torch.zeros(1, n_s, self.k_max, device=dev)
        filled = torch.zeros(1, self.k_max, dtype=torch.bool, device=dev)
        c = self.probe(V0, filled, ys, ys.float()[None], Xs, Xs * Xs - 1.0)[0]
        z = self.flex(Xs, ys.float()[None], self.flex.prepare(Xs))[0]
        y0 = ys.float() - ys.float().mean()
        zv = self.flex(Xs, (y0 * y0)[None], self.flex.prepare(Xs))[0]
        fz = torch.stack([z / 10.0, torch.log1p(F.relu(z)), torch.zeros_like(z), torch.log1p(F.relu(zv))], -1)
        s = self.score(F.gelu(self.wk(H) + self.wc(c.to(H.dtype)) + self.wf(fz.to(H.dtype)))).squeeze(1).float()
        s = s + self.beta * z.clamp(-3.0, 30.0) / 3.0
        s = s.masked_fill(~allowed, float("-inf"))
        logits0 = s[None]
        if teacher is not None:
            logp = torch.log_softmax(s, 0)
            ce = -logp[torch.tensor(list(teacher), device=dev)].mean()
            return {"onehots": [], "chosen": [], "V": None, "soft": torch.zeros(1, P, device=dev), "ce": ce,
                    "logits0": logits0}
        onehots, chosen, softs, logps = [], [], [], []
        ls = torch.log_softmax(s, 0).detach()
        n_allowed = int(allowed.sum())
        for p in range(n_paths):
            k = min(1 + p % self.k_max, n_allowed)
            z = s if (p == 0 and greedy_first) else s + _gumbel_like(s, generator)
            gate = topk_gate(z / tau, k)
            idx = torch.topk(z, k).indices
            rows = []
            for j in idx.tolist():
                e = F.one_hot(torch.tensor(j, device=dev), P).float()
                rows.append(e * (1.0 + gate[j] - gate[j].detach()))
            onehots.append(rows)
            chosen.append(idx.tolist())
            softs.append(torch.softmax(z / tau, 0))
            logps.append(ls[idx].sum())
        return {"onehots": onehots, "chosen": chosen, "V": None, "soft": torch.stack(softs),
                "ce": torch.zeros((), device=dev), "logits0": logits0, "logp": torch.stack(logps)}


def _diversity(soft: torch.Tensor) -> torch.Tensor:
    if soft.shape[0] < 2:
        return torch.zeros((), device=soft.device)
    S = F.normalize(soft, dim=1)
    G = S @ S.T
    n = G.shape[0]
    return (G.sum() - G.diagonal().sum()) / (n * (n - 1))


# ----------------------------------------------------------------------- model
def _centre_scale_t(M: torch.Tensor, groups: torch.Tensor, n_s: int) -> torch.Tensor:
    """Torch version of ``episodes.centre_scale`` (support-row patient means, support sd)."""
    one = M.ndim == 1
    if one:
        M = M[:, None]
    _, inv = torch.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    sup = torch.zeros(len(groups), dtype=torch.bool, device=M.device)
    sup[:n_s] = True
    w_sup = sup.float()
    cnt_s = torch.zeros(G, device=M.device).index_add_(0, inv, w_sup)
    cnt_a = torch.zeros(G, device=M.device).index_add_(0, inv, torch.ones_like(w_sup))
    sum_s = torch.zeros(G, M.shape[1], device=M.device, dtype=M.dtype).index_add_(0, inv, M * w_sup[:, None])
    sum_a = torch.zeros(G, M.shape[1], device=M.device, dtype=M.dtype).index_add_(0, inv, M)
    mean = torch.where((cnt_s > 0)[:, None], sum_s / cnt_s.clamp(min=1)[:, None], sum_a / cnt_a.clamp(min=1)[:, None])
    out = M - mean[inv]
    sd = out[:n_s].std(0).clamp(min=1e-6)
    out = out / sd
    return out[:, 0] if one else out


class LearnedSearchModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.concept = ConceptLayer(cfg.embed_dim, cfg.concept_hidden, cfg.n_concepts)
        self.tokens = CandidateTokenEncoder(cfg.d_model, cfg.obs_hidden, cfg.n_samples)
        self.encoder = SetEncoder(cfg.d_model, cfg.n_heads, cfg.n_blocks, cfg.encoder_kind, cfg.n_inducing)
        sel = PointerSelector if cfg.selector == "pointer" else TopKSelector
        self.selector = sel(cfg.d_model, cfg.k_max, cfg.probe_m)
        self.relation = RelationFit(cfg.n_rff, cfg.ridge_alpha, cfg.k_max)
        self.meta: dict = {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- io
    def save(self, path, **meta) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), "meta": {**self.meta, **meta}},
                   str(path))

    @classmethod
    def load(cls, path, map_location="cpu") -> "LearnedSearchModel":
        ck = torch.load(str(path), map_location=map_location, weights_only=False)
        if ck.get("kind") == GainSearchModel.kind:   # v02 checkpoints load through the same call
            return GainSearchModel.load(path, map_location=map_location)
        if ck.get("kind") == "gain_v03":
            from .model_v03 import GainModelV03
            return GainModelV03.load(path, map_location=map_location)
        m = cls(ModelConfig(**ck["config"]))
        missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
        if unexpected or [k for k in missing if not k.endswith("att_scale")]:
            raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
        m.meta = dict(ck.get("meta", {}))
        m.eval()
        return m

    # -- concepts
    @torch.no_grad()
    def concept_values(self, embed) -> np.ndarray:
        E = torch.as_tensor(np.asarray(embed, dtype=np.float32), device=self.device)
        return self.concept(E).float().cpu().numpy()

    concepts = concept_values

    # -- tensors for one episode
    def prepare(self, ep: Episode, device=None) -> dict:
        dev = torch.device(device) if device is not None else self.device
        X = torch.as_tensor(ep.X, device=dev, dtype=torch.float32)
        groups = torch.as_tensor(ep.groups, device=dev)
        names = list(ep.meta.names)
        transform = torch.as_tensor(ep.meta.transform, device=dev)
        view = torch.as_tensor(ep.meta.view, device=dev)
        allowed = torch.ones(X.shape[1], dtype=torch.bool, device=dev)
        decor = torch.zeros((), device=dev)
        y = None if ep.y is None else torch.as_tensor(ep.y, device=dev, dtype=torch.float32)
        if ep.concept is not None and ep.concept["E"] is not None and len(ep.concept["E"]):
            c = ep.concept
            E = torch.as_tensor(np.asarray(c["E"], dtype=np.float32), device=dev)
            own = torch.as_tensor(c["own"], device=dev)
            nb6 = torch.as_tensor(c["nb6"], device=dev)
            nb18 = torch.as_tensor(c["nb18"], device=dev)
            C = self.concept(E).float()
            decor = ConceptLayer.decorrelation(C)
            Ct = _centre_scale_t(concept_transforms(C, own, nb6, nb18), groups, ep.n_support)
            K = self.cfg.n_concepts
            p = X.shape[1]
            X = torch.cat([X, Ct], 1)
            names += [f"{t}:concept:{j}" for t in TRANSFORMS for j in range(K)]
            transform = torch.cat([transform, torch.arange(4, device=dev).repeat_interleave(K)])
            view = torch.cat([view, torch.full((4 * K,), VIEW_VOCAB.index("concept"), device=dev)])
            allowed = torch.cat([allowed, torch.ones(4 * K, dtype=torch.bool, device=dev)])
            if ep.concept_target is not None:
                k = int(ep.concept_target) % K
                Cd = self.concept.detached(E)[own]
                y = _centre_scale_t(Cd[:, k], groups, ep.n_support).detach()
                allowed[p + k] = False            # own
                allowed[p + 3 * K + k] = False    # contrast (contains own)
        if y is None:
            raise ValueError("episode has neither y nor a concept target with an embedding")
        coords = torch.as_tensor(ep.coords, device=dev, dtype=torch.float32)
        return {"X": X, "y": y, "n_s": int(ep.n_support), "groups": groups, "coords": coords, "names": names,
                "transform": transform, "view": view, "allowed": allowed, "decor": decor}

    def encode(self, prep: dict, generator=None) -> torch.Tensor:
        n_s = prep["n_s"]
        T = self.tokens(prep["X"][:n_s], prep["y"][:n_s], prep["coords"][:n_s], prep["groups"][:n_s],
                        prep["transform"], prep["view"], generator=generator)
        return self.encoder(T)

    def select(self, prep: dict, H: torch.Tensor, *, n_paths: int, tau: float = 1.0, teacher=None,
               greedy_first: bool = True, generator=None) -> dict:
        return self.selector(H, prep["X"], prep["n_s"], prep["y"][:prep["n_s"]], prep["allowed"], self.relation,
                             n_paths=n_paths, tau=tau, teacher=teacher, greedy_first=greedy_first,
                             generator=generator)

    @staticmethod
    def path_values(prep: dict, rows: list) -> torch.Tensor:
        """(n, k) values of a path's selected tokens (``V = X @ onehot``, gradients kept)."""
        with torch.autocast(device_type=prep["X"].device.type, enabled=False):
            return torch.stack([prep["X"] @ r.float() for r in rows], 1)

    # -- training loss on one episode
    def episode_loss(self, ep: Episode, *, tau: float, n_paths: int | None = None, generator=None,
                     weights: dict | None = None) -> dict:
        w = {"ce": 0.5, "div": 0.01, "decor": 0.01, "hinge": 0.1, **(weights or {})}
        n_paths = n_paths or self.cfg.n_paths
        prep = self.prepare(ep)
        n_s = prep["n_s"]
        y = prep["y"]
        H = self.encode(prep, generator)
        truth = list(ep.true_idx) if (ep.kind == "controlled" and ep.true_idx) else None
        merged = truth is not None and isinstance(self.selector, PointerSelector)
        sel = self.select(prep, H, n_paths=n_paths, tau=tau, greedy_first=False, generator=generator,
                          teacher=truth if merged else None)
        ks_all = [len(c) for c in sel["chosen"]]
        if sel["V"] is not None:   # the pointer already holds V = X @ onehot (with the straight-through gradient)
            Vs_list = [sel["V"][p, :, :k] for p, k in enumerate(ks_all)]
        else:
            Vs_list = [self.path_values(prep, rows) for rows in sel["onehots"]]
        L = torch.zeros(len(Vs_list), device=y.device)
        for k in sorted(set(ks_all)):
            idx = [i for i, kk in enumerate(ks_all) if kk == k]
            Vb = sel["V"][idx][:, :, :k] if sel["V"] is not None else torch.stack([Vs_list[i] for i in idx])
            r2 = self.relation.query_r2(Vb[:, :n_s], y[:n_s], Vb[:, n_s:], y[n_s:])
            L = L.index_put((torch.tensor(idx, device=y.device),), 1.0 - r2)
        rel = L.min() + 0.1 * L.mean()
        out = {"rel": rel, "best_r2": 1.0 - L.min().detach(), "mean_r2": 1.0 - L.mean().detach(),
               "k_mean": float(np.mean([len(c) for c in sel["chosen"]]))}
        total = rel + w["div"] * _diversity(sel["soft"]) + w["decor"] * prep["decor"]
        out["div"] = _diversity(sel["soft"]).detach()
        out["decor"] = prep["decor"].detach()
        if truth:
            ce = sel["ce"] if merged else self.select(prep, H, n_paths=1, tau=tau, teacher=truth)["ce"]
            total = total + w["ce"] * ce
            out["ce"] = ce.detach()
            fam = ep.family or ""
            if len(truth) >= 2 and (fam.startswith("residualised") or fam.startswith("neural")):
                ks = [len(c) for c in sel["chosen"]]
                cand = [i for i, k in enumerate(ks) if k >= 2]
                if cand:
                    Ld = L.detach().cpu().numpy()
                    b = min(cand, key=lambda i: float(Ld[i]))
                    V = Vs_list[b]
                    g = self.relation.g_ho(V[:n_s], y[:n_s], V[n_s:], y[n_s:])
                    total = total + w["hinge"] * F.relu(0.002 - g)
                    out["g_ho"] = g.detach()
        out["total"] = total
        return out

    # -- ranking (support rows only)
    @torch.no_grad()
    def rank_sets(self, prep: dict, *, n_paths: int = 32, seed: int = 0, n_folds: int = 3,
                  g_ho_top: int | None = 16, rank_by: str = "gain") -> list:
        """Unique variable sets from ``n_paths`` paths (path 0 greedy) and every prefix of
        each path, scored on SUPPORT rows only by patient-grouped CV R^2 of the relation.

        ``rank_by="gain"`` (default) orders sets by their CV R^2 minus the best CV R^2
        of their (k-1)-subsets (singletons: their CV R^2), so a set ranks by what its
        full composition adds; ``"cv"`` by CV R^2; ``"logp"`` by path log-probability."""
        n_s = prep["n_s"]
        sub = {k: v for k, v in prep.items()}
        for key in ("X", "y", "groups", "coords"):
            sub[key] = prep[key][:n_s]
        gen = torch.Generator(device=sub["X"].device).manual_seed(int(seed))
        H = self.encode(sub, gen)
        sel = self.select(sub, H, n_paths=n_paths, tau=1.0, greedy_first=True, generator=gen)
        lp_all = sel["logp"].tolist()
        seen: dict = {}
        for p, ch in enumerate(sel["chosen"]):
            for j in range(1, len(ch) + 1):
                key = tuple(sorted(ch[:j]))
                if key not in seen or lp_all[p] > lp_all[seen[key][0]]:
                    seen[key] = (p, j)
        y, g = sub["y"], sub["groups"]
        need = set(seen)
        for key in list(seen):
            if len(key) >= 2:
                need |= {tuple(sorted(set(key) - {j})) for j in key}
        cvs: dict = {}
        for k in sorted({len(key) for key in need}):
            ks = sorted(key for key in need if len(key) == k)
            Vb = torch.stack([sub["X"][:, list(key)] for key in ks])
            for key, cv in zip(ks, self.relation.cv_r2(Vb, y, g, n_folds=n_folds, seed=seed).tolist()):
                cvs[key] = float(cv)
        ranked = []
        for key, (p, j) in seen.items():
            gain = cvs[key] if len(key) == 1 else cvs[key] - max(cvs[tuple(sorted(set(key) - {i}))] for i in key)
            ranked.append((key, cvs[key], {"path_rank": p, "order": tuple(sel["chosen"][p][:j]),
                                          "path_logp": float(lp_all[p]), "gain_over_subsets": float(gain),
                                          "support_cv_r2": cvs[key]}))
        if rank_by == "logp":
            ranked.sort(key=lambda t: (-t[2]["path_logp"], t[2]["path_rank"]))
        elif rank_by == "cv":
            ranked.sort(key=lambda t: (-t[1], t[2]["path_rank"]))
        else:
            ranked.sort(key=lambda t: (-t[2]["gain_over_subsets"], t[2]["path_rank"]))
        for i, (key, cv, info) in enumerate(ranked):
            info["n_subset_fits"] = len(need)
            if len(key) >= 2 and (g_ho_top is None or i < g_ho_top):
                V = sub["X"][:, list(key)]
                info["g_ho_support"] = float(self.relation.g_ho(V, y, groups=g, n_folds=n_folds))
            else:
                info["g_ho_support"] = 0.0 if len(key) < 2 else None
        return ranked

    @torch.no_grad()
    def rank_sets_conditional_beam(self, prep: dict, *, n_paths: int = 32, seed: int = 0,
                                   seed_pool: int = 16, triple_pairs: int = 8,
                                   triple_partners: int = 7, n_folds: int = 3) -> list:
        """Bounded support-only set proposer for weak-marginal compositions."""
        n_s = prep["n_s"]
        sub = {k: v for k, v in prep.items()}
        for key in ("X", "y", "groups", "coords"):
            sub[key] = prep[key][:n_s]
        gen = torch.Generator(device=sub["X"].device).manual_seed(int(seed))
        H = self.encode(sub, gen)
        sel = self.select(sub, H, n_paths=n_paths, tau=1.0, greedy_first=True, generator=gen)
        P = sub["X"].shape[1]
        scores = {
            "pointer": sel["logits0"].float().mean(0).detach().cpu().numpy(),
            "flex": sel["initial_flex"].float().mean(0).detach().cpu().numpy(),
            "variance": sel["initial_variance"].float().mean(0).detach().cpu().numpy(),
        }
        rank = {key: np.empty(P, dtype=int) for key in scores}
        for key, val in scores.items():
            order = np.argsort(-val, kind="stable")
            rank[key][order] = np.arange(P)
        candidate = set()
        per_signal = min(12, P)
        for val in scores.values():
            candidate.update(np.argsort(-val, kind="stable")[:per_signal].tolist())
        allowed = sub["allowed"].detach().cpu().numpy().astype(bool)
        candidate = [j for j in candidate if allowed[j]]
        candidate.sort(key=lambda j: (min(rank[k][j] for k in rank),
                                      sum(rank[k][j] for k in rank), j))
        pool = candidate[:min(int(seed_pool), len(candidate))]
        singleton = [(j,) for j in pool]
        pairs = [tuple(sorted(x)) for x in combinations(pool, 2)]
        y, g = sub["y"], sub["groups"]

        def score(keys):
            out = {}
            for size in sorted({len(k) for k in keys}):
                same = sorted(k for k in keys if len(k) == size)
                if same:
                    Vb = torch.stack([sub["X"][:, list(k)] for k in same])
                    vals = self.relation.cv_r2(Vb, y, g, n_folds=n_folds, seed=seed).tolist()
                    out.update({k: float(v) for k, v in zip(same, vals)})
            return out

        initial = singleton + pairs
        cvs = score(initial)
        pair_gain = {k: cvs[k] - max(cvs[(j,)] for j in k) for k in pairs}
        best_pairs = sorted(pairs, key=lambda k: (-pair_gain[k], k))[:int(triple_pairs)]
        triples = []
        for pair in best_pairs:
            partners = [j for j in pool if j not in pair][:int(triple_partners)]
            triples.extend(tuple(sorted(pair + (j,))) for j in partners)
        triples = sorted(set(triples))
        sets = singleton + pairs + triples
        if len(sets) > 192:
            raise RuntimeError(f"conditional beam cap violated: {len(sets)} sets")
        cvs.update(score(triples))
        ranked = []
        sources = {j: [k for k in scores if rank[k][j] < per_signal] for j in pool}
        for key in sets:
            gain = cvs[key] if len(key) == 1 else cvs[key] - max(
                cvs[tuple(sorted(set(key) - {j}))] for j in key)
            info = {"proposal_family": "conditional_beam", "support_cv_r2": float(cvs[key]),
                    "gain_over_subsets": float(gain), "beam_seed_pool": int(len(pool)),
                    "beam_seed_sources": {str(j): sources.get(j, []) for j in key},
                    "initial_score_ranks": {str(j): {k: int(rank[k][j]) for k in rank} for j in key}}
            ranked.append((key, float(cvs[key]), info))
        ranked.sort(key=lambda t: (-t[2]["gain_over_subsets"], -t[1], t[0]))
        for key, _, info in ranked:
            info["n_subset_fits"] = int(len(sets))
            # Candidate ranking uses grouped support CV gain alone. Higher-order
            # decomposition belongs to later held-out characterization, not to
            # this bounded proposal stage.
            info["g_ho_support"] = 0.0 if len(key) < 2 else None
        return ranked

    @torch.no_grad()
    def rank_sets_pointer_expansion(self, prep: dict, *, n_paths: int = 32, seed: int = 0,
                                    max_sets: int = 192, n_anchors: int = 16,
                                    n_partners: int = 5, n_folds: int = 3,
                                    max_screen_rows: int = 4096) -> list:
        """Preserve pointer proposals, then add support-only conditional partners.

        The original pointer sets are retained verbatim. For a bounded number
        of their strongest constituent variables, a one-variable residual is
        passed through the selector's learned interaction screen to nominate
        partners. This tests missing pair construction without discarding
        spatial tokens already selected by the frozen pointer.
        """
        pointer = self.rank_sets(prep, n_paths=n_paths, seed=seed, n_folds=n_folds,
                                 g_ho_top=0, rank_by="gain")
        base = [tuple(key) for key, _, _ in pointer]
        kept = set(base)
        anchors = []
        for key, _, _ in pointer:
            for j in key:
                if j not in anchors:
                    anchors.append(j)
                if len(anchors) >= int(n_anchors):
                    break
            if len(anchors) >= int(n_anchors):
                break
        n_s = prep["n_s"]
        X, y, groups = prep["X"][:n_s], prep["y"][:n_s], prep["groups"][:n_s]
        # The interaction statistic is a proposal screen, not the final
        # relation test. Bound its working matrix deterministically; every
        # emitted candidate is still scored by grouped CV on all support rows.
        if len(X) > int(max_screen_rows):
            take = torch.linspace(0, len(X) - 1, int(max_screen_rows), device=X.device).round().long()
            Xscreen, yscreen = X[take], y[take]
        else:
            Xscreen, yscreen = X, y
        Xscreen2 = Xscreen * Xscreen - 1.0
        allowed = prep["allowed"].detach().cpu().numpy().astype(bool)
        added = {}
        for anchor in anchors:
            if len(kept) >= int(max_sets):
                break
            value = X[:, [anchor]]
            residual = y - self.relation.fit_predict(value, y, value)
            if hasattr(self.selector, "inter"):
                screen = self.selector.inter(Xscreen, Xscreen2, residual[take][None] if len(X) > int(max_screen_rows)
                                             else residual[None],
                                             value[take, 0][None] if len(X) > int(max_screen_rows)
                                             else value[:, 0][None])[0]
            else:
                screen = self.selector.flex(X, residual[None], self.selector.flex.prepare(X))[0]
            order = np.argsort(-screen.detach().float().cpu().numpy(), kind="stable")
            taken = 0
            for partner in order:
                partner = int(partner)
                if partner == anchor or not allowed[partner]:
                    continue
                key = tuple(sorted((int(anchor), partner)))
                if key in kept:
                    continue
                kept.add(key)
                added[key] = {"anchor": int(anchor), "partner": partner, "conditional_partner_rank": int(taken)}
                taken += 1
                if taken >= int(n_partners) or len(kept) >= int(max_sets):
                    break
        sets = sorted(kept)
        need = set(sets)
        for key in sets:
            if len(key) >= 2:
                need |= {tuple(sorted(set(key) - {j})) for j in key}
        cvs = {}
        for k in sorted({len(key) for key in need}):
            same = sorted(key for key in need if len(key) == k)
            Vb = torch.stack([X[:, list(key)] for key in same])
            vals = self.relation.cv_r2(Vb, y, groups, n_folds=n_folds, seed=seed).tolist()
            cvs.update({key: float(v) for key, v in zip(same, vals)})
        base_info = {tuple(key): dict(info) for key, _, info in pointer}
        ranked = []
        for key in sets:
            gain = cvs[key] if len(key) == 1 else cvs[key] - max(
                cvs[tuple(sorted(set(key) - {j}))] for j in key)
            info = {"proposal_family": "pointer_conditional_expansion",
                    "pointer_preserved": key in base_info,
                    "support_cv_r2": float(cvs[key]), "gain_over_subsets": float(gain),
                    "n_pointer_sets": int(len(base)), "n_conditional_added": int(len(added)),
                    "conditional_detail": added.get(key)}
            if key in base_info:
                info["pointer_path_rank"] = base_info[key].get("path_rank")
            ranked.append((key, float(cvs[key]), info))
        ranked.sort(key=lambda t: (-t[2]["gain_over_subsets"], -t[1], t[0]))
        for key, _, info in ranked:
            info["n_subset_fits"] = int(len(need))
            info["g_ho_support"] = 0.0 if len(key) < 2 else None
        return ranked

    # -- the contract used by propose.py (Agent C)
    @torch.no_grad()
    def rank_sets_multiroute_expansion(self, prep: dict, *, n_paths: int = 32, seed: int = 0,
                                       max_sets: int = 192, n_pair_sketch: int = 16,
                                       n_partners: int = 5, n_folds: int = 3,
                                       max_screen_rows: int = 4096) -> list:
        """Pointer expansion plus a support-only second-order anchor route.

        The pair sketch is intentionally small and transparent: it measures
        group-centred third cross-moments E[x_i x_j y] across every searchable
        representation.  It can surface a weak-marginal product or gated pair
        without replacing the learned pointer.  All emitted candidates are
        subsequently ranked by the normal grouped support-CV relation scorer.
        """
        inherited = self.rank_sets_pointer_expansion(
            prep, n_paths=n_paths, seed=seed, max_sets=max_sets,
            n_partners=n_partners, n_folds=n_folds, max_screen_rows=max_screen_rows)
        kept = {tuple(key) for key, _, _ in inherited}
        inherited_info = {tuple(key): dict(info) for key, _, info in inherited}
        n_s = prep["n_s"]
        X, y, groups = prep["X"][:n_s], prep["y"][:n_s], prep["groups"][:n_s]
        if len(X) > int(max_screen_rows):
            take = torch.linspace(0, len(X) - 1, int(max_screen_rows), device=X.device).round().long()
            Xscreen, yscreen, gscreen = X[take].float(), y[take].float(), groups[take]
        else:
            take = None
            Xscreen, yscreen, gscreen = X.float(), y.float(), groups
        Xc, yc = Xscreen.clone(), yscreen.clone()
        for group in torch.unique(gscreen):
            rows = gscreen == group
            Xc[rows] -= Xc[rows].mean(0, keepdim=True)
            yc[rows] -= yc[rows].mean()
        moment = torch.abs((Xc.T @ (Xc * yc[:, None])) / max(len(Xc), 1))
        allowed = prep["allowed"].detach().cpu().numpy().astype(bool)
        legal = torch.as_tensor(allowed, device=X.device, dtype=torch.bool)
        moment[~legal, :] = -torch.inf
        moment[:, ~legal] = -torch.inf
        moment.fill_diagonal_(-torch.inf)
        top = torch.topk(moment.reshape(-1), k=min(int(n_pair_sketch) * 8, moment.numel())).indices.tolist()
        width, sketch_pairs = moment.shape[0], []
        for flat in top:
            a, b = divmod(int(flat), width)
            key = tuple(sorted((a, b)))
            if a == b or key in sketch_pairs:
                continue
            sketch_pairs.append(key)
            if len(sketch_pairs) >= int(n_pair_sketch):
                break
        sketch_added, anchors = {}, []
        for key in sketch_pairs:
            if len(kept) >= int(max_sets):
                break
            kept.add(key)
            sketch_added[key] = {"pair_sketch_score": float(moment[key[0], key[1]].item())}
            anchors.extend(key)
        anchors = list(dict.fromkeys(anchors))
        Xscreen2 = Xscreen * Xscreen - 1.0
        added = {}
        for anchor in anchors:
            if len(kept) >= int(max_sets):
                break
            value = X[:, [anchor]]
            residual = y - self.relation.fit_predict(value, y, value)
            if hasattr(self.selector, "inter"):
                screen = self.selector.inter(
                    Xscreen, Xscreen2,
                    residual[take][None] if take is not None else residual[None],
                    value[take, 0][None] if take is not None else value[:, 0][None])[0]
            else:
                screen = self.selector.flex(X, residual[None], self.selector.flex.prepare(X))[0]
            order = np.argsort(-screen.detach().float().cpu().numpy(), kind="stable")
            taken = 0
            for partner in order:
                partner = int(partner)
                key = tuple(sorted((int(anchor), partner)))
                if partner == anchor or not allowed[partner] or key in kept:
                    continue
                kept.add(key)
                added[key] = {"anchor": int(anchor), "partner": partner,
                              "conditional_partner_rank": int(taken)}
                taken += 1
                if taken >= int(n_partners) or len(kept) >= int(max_sets):
                    break
        sets = sorted(kept)
        need = set(sets)
        for key in sets:
            if len(key) >= 2:
                need |= {tuple(sorted(set(key) - {j})) for j in key}
        cvs = {}
        for size in sorted({len(key) for key in need}):
            same = sorted(key for key in need if len(key) == size)
            Vb = torch.stack([X[:, list(key)] for key in same])
            vals = self.relation.cv_r2(Vb, y, groups, n_folds=n_folds, seed=seed).tolist()
            cvs.update({key: float(v) for key, v in zip(same, vals)})
        ranked = []
        for key in sets:
            gain = cvs[key] if len(key) == 1 else cvs[key] - max(
                cvs[tuple(sorted(set(key) - {j}))] for j in key)
            info = {"proposal_family": "multiroute_pair_sketch_expansion",
                    "pointer_preserved": key in inherited_info,
                    "support_cv_r2": float(cvs[key]), "gain_over_subsets": float(gain),
                    "n_pair_sketch_pairs": int(len(sketch_pairs)),
                    "n_pair_sketch_added": int(len(sketch_added)),
                    "n_conditional_added": int(len(added)),
                    "pair_sketch_detail": sketch_added.get(key),
                    "conditional_detail": added.get(key)}
            if key in inherited_info:
                info["inherited_pointer_expansion"] = True
            ranked.append((key, float(cvs[key]), info))
        ranked.sort(key=lambda t: (-t[2]["gain_over_subsets"], -t[1], t[0]))
        for key, _, info in ranked:
            info["n_subset_fits"] = int(len(need))
            info["g_ho_support"] = 0.0 if len(key) < 2 else None
        if len(ranked) > int(max_sets):
            raise RuntimeError("multiroute expansion cap violated")
        return ranked

    @torch.no_grad()
    def rank_sets_multiroute_v2(self, prep: dict, *, n_paths: int = 32, seed: int = 0,
                                max_sets: int = 192, n_pair_sketch: int = 16,
                                n_context_pairs: int = 16, n_binned_pairs: int = 16,
                                n_folds: int = 3, max_screen_rows: int = 4096) -> list:
        """Pointer-preserving proposal with product, gate, and nonlinear routes.

        The routes only use support rows.  Product screening uses the existing
        group-centred third moment.  Context screening explicitly searches a
        variable multiplied by an upper/lower quantile gate of a context token.
        Nonlinear screening uses a small two-dimensional quantile table and
        retains only interaction variance left after additive row/column effects.
        They are proposal screens; every retained set is ranked with the shared
        grouped-CV relation scorer below.
        """
        pointer = self.rank_sets(prep, n_paths=n_paths, seed=seed, g_ho_top=None)
        kept = {tuple(key) for key, _, _ in pointer}
        pointer_info = {tuple(key): dict(info) for key, _, info in pointer}
        n_s = prep["n_s"]
        X, y, groups = prep["X"][:n_s], prep["y"][:n_s], prep["groups"][:n_s]
        if len(X) > int(max_screen_rows):
            take = torch.linspace(0, len(X) - 1, int(max_screen_rows), device=X.device).round().long()
            Xs, ys, gs = X[take].float(), y[take].float(), groups[take]
        else:
            Xs, ys, gs = X.float(), y.float(), groups
        # Centre every screen within patient, so large patient-level shifts do
        # not nominate a pair which will fail patient-disjoint evaluation.
        Xc, yc = Xs.clone(), ys.clone()
        for group in torch.unique(gs):
            rows = gs == group
            Xc[rows] -= Xc[rows].mean(0, keepdim=True)
            yc[rows] -= yc[rows].mean()
        legal = prep["allowed"].bool()
        names = prep["names"]
        route_detail: dict[tuple, dict] = {}

        def add_top(score: torch.Tensor, n: int, route: str, detail=None, *, restrict=None):
            if restrict is not None:
                valid = restrict
                score = score.clone()
                score[~valid] = -torch.inf
            score = score.clone()
            score[~legal, :] = -torch.inf
            score[:, ~legal] = -torch.inf
            score.fill_diagonal_(-torch.inf)
            flat = torch.topk(score.reshape(-1), k=min(int(n) * 12, score.numel())).indices.tolist()
            width, picked = score.shape[0], []
            for idx in flat:
                a, b = divmod(int(idx), width)
                key = tuple(sorted((a, b)))
                if a == b or key in picked:
                    continue
                picked.append(key)
                if len(picked) >= int(n):
                    break
            for key in picked:
                if len(kept) >= int(max_sets):
                    break
                kept.add(key)
                route_detail.setdefault(key, {}).update({"route": route, "screen_score": float(score[key].item())})
                if detail is not None:
                    route_detail[key].update(detail(key))
            return picked

        # 1) Weak-marginal products and other cubic interactions.
        moment = torch.abs((Xc.T @ (Xc * yc[:, None])) / max(len(Xc), 1))
        add_top(moment, n_pair_sketch, "third_moment")

        # 2) A gate is deliberately scored against every declared context token,
        # rather than preselecting contexts by their (often zero) marginal score.
        context = torch.as_tensor(["context_" in str(name) for name in names], device=X.device, dtype=torch.bool)
        ordinary = legal & ~context
        if bool(context.any()) and bool(ordinary.any()):
            best = torch.full_like(moment, -torch.inf)
            best_meta: dict[tuple, dict] = {}
            for quantile in (1.0 / 3.0, 0.5, 2.0 / 3.0):
                cuts = torch.quantile(Xs[:, context], quantile, dim=0)
                for direction, gate in (("upper", Xs[:, context] >= cuts),
                                        ("lower", Xs[:, context] < cuts)):
                    gate = gate.float()
                    for group in torch.unique(gs):
                        rows = gs == group
                        gate[rows] -= gate[rows].mean(0, keepdim=True)
                    score_part = torch.abs(Xc.T @ (yc[:, None] * gate)) / max(len(Xc), 1)
                    columns = torch.where(context)[0]
                    full = torch.full_like(moment, -torch.inf)
                    full[:, columns] = score_part
                    replace = full > best
                    best = torch.maximum(best, full)
                    for a, b in zip(*torch.where(replace)):
                        best_meta[tuple(sorted((int(a), int(b))))] = {"gate_quantile": quantile,
                                                                       "gate_direction": direction}
            allowed_pair = ordinary[:, None] & context[None, :]
            add_top(best, n_context_pairs, "context_gate", lambda key: best_meta.get(key, {}),
                    restrict=allowed_pair | allowed_pair.T)

        # 3) A compact non-parametric 2-D screen.  The feature pool protects
        # transformed views from being crowded out by one high-variance view.
        marginal = torch.abs(Xc.T @ yc)
        pool = set(torch.topk(marginal.masked_fill(~legal, -torch.inf), k=min(192, int(legal.sum()))).indices.tolist())
        transforms = prep["transform"]
        for transform in torch.unique(transforms):
            mask = legal & (transforms == transform)
            if bool(mask.any()):
                pool.update(torch.topk(marginal.masked_fill(~mask, -torch.inf),
                                       k=min(64, int(mask.sum()))).indices.tolist())
        pool = sorted(pool)
        if len(pool) >= 2:
            V = Xs[:, pool]
            cuts = torch.quantile(V, torch.tensor([.25, .5, .75], device=X.device), dim=0)
            bins = (V[None, :, :] > cuts[:, None, :]).sum(0).long()
            onehot = [(bins == b).float() for b in range(4)]
            sums = torch.stack([a.T @ (yc[:, None] * b) for a in onehot for b in onehot]).reshape(4, 4, len(pool), len(pool))
            counts = torch.stack([a.T @ b for a in onehot for b in onehot]).reshape(4, 4, len(pool), len(pool)).clamp_min(1.0)
            means = sums / counts
            weights = counts / counts.sum((0, 1), keepdim=True)
            row = (weights * means).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1e-6)
            col = (weights * means).sum(0, keepdim=True) / weights.sum(0, keepdim=True).clamp_min(1e-6)
            grand = (weights * means).sum((0, 1), keepdim=True)
            interaction = means - row - col + grand
            local_score = torch.sqrt((weights * interaction.square()).sum((0, 1)))
            binned = torch.full_like(moment, -torch.inf)
            pi = torch.as_tensor(pool, device=X.device)
            binned[pi[:, None], pi[None, :]] = local_score
            add_top(binned, n_binned_pairs, "binned_2d")

        sets = sorted(kept)
        need = set(sets)
        for key in sets:
            if len(key) >= 2:
                need |= {tuple(sorted(set(key) - {j})) for j in key}
        cvs = {}
        for size in sorted({len(key) for key in need}):
            same = sorted(key for key in need if len(key) == size)
            values = self.relation.cv_r2(torch.stack([X[:, list(key)] for key in same]), y, groups,
                                          n_folds=n_folds, seed=seed).tolist()
            cvs.update({key: float(value) for key, value in zip(same, values)})
        ranked = []
        for key in sets:
            gain = cvs[key] if len(key) == 1 else cvs[key] - max(cvs[tuple(sorted(set(key) - {j}))] for j in key)
            info = {"proposal_family": "multiroute_v2", "pointer_preserved": key in pointer_info,
                    "support_cv_r2": float(cvs[key]), "gain_over_subsets": float(gain),
                    "route_detail": route_detail.get(key), "n_pointer_sets": len(pointer)}
            ranked.append((key, float(cvs[key]), info))
        ranked.sort(key=lambda t: (-t[2]["gain_over_subsets"], -t[1], t[0]))
        for _, _, info in ranked:
            info["n_subset_fits"] = int(len(need))
            info["g_ho_support"] = None
        if len(ranked) > int(max_sets):
            raise RuntimeError("multiroute v2 cap violated")
        return ranked

    @torch.no_grad()
    def rank_sets_multiroute_hybrid_v11(self, prep: dict, *, n_paths: int = 32, seed: int = 0,
                                         max_sets: int = 192, n_context_pairs: int = 16,
                                         n_folds: int = 3, max_screen_rows: int = 4096) -> list:
        """v1 multiroute backbone plus a small, direct context-gate reserve.

        This deliberately does *not* inherit v2's binned two-dimensional
        screen.  v2 accidentally restarted from the raw pointer and therefore
        displaced the conditional partners which made v1 useful on spatial and
        piecewise packets.  Here, v1 is the candidate backbone.  A bounded
        number of support-only context gates is added after making room only by
        removing the lowest-ranked conditional additions; raw-pointer and
        third-moment candidates are never displaced.

        Context-screen metadata is kept in tensors until after the final small
        top-k is selected.  The old v2 code synchronised Python bookkeeping for
        every cell of a feature-by-context matrix, which dominated runtime.
        """
        if max_sets <= n_context_pairs:
            raise ValueError("max_sets must exceed the context reserve")
        inherited = self.rank_sets_multiroute_expansion(
            prep, n_paths=n_paths, seed=seed, max_sets=max_sets,
            n_folds=n_folds, max_screen_rows=max_screen_rows)
        inherited_info = {tuple(key): dict(info) for key, _, info in inherited}

        n_s = prep["n_s"]
        X, y, groups = prep["X"][:n_s], prep["y"][:n_s], prep["groups"][:n_s]
        if len(X) > int(max_screen_rows):
            take = torch.linspace(0, len(X) - 1, int(max_screen_rows), device=X.device).round().long()
            Xs, ys, gs = X[take].float(), y[take].float(), groups[take]
        else:
            Xs, ys, gs = X.float(), y.float(), groups
        Xc, yc = Xs.clone(), ys.clone()
        for group in torch.unique(gs):
            rows = gs == group
            Xc[rows] -= Xc[rows].mean(0, keepdim=True)
            yc[rows] -= yc[rows].mean()

        legal = prep["allowed"].bool()
        names = prep["names"]
        context = torch.as_tensor(["context_" in str(name) for name in names],
                                  device=X.device, dtype=torch.bool)
        ordinary = legal & ~context
        context_idx = torch.where(context)[0]
        ordinary_idx = torch.where(ordinary)[0]
        route_detail: dict[tuple, dict] = {}

        # Preserve all v1 candidates unless a genuine context reserve is
        # available.  This makes non-context packets exactly v1 rather than a
        # silently weakened version of it.
        kept = {tuple(key) for key, _, _ in inherited}
        if len(context_idx) and len(ordinary_idx):
            protected = {
                key for key, _, info in inherited
                if info.get("pointer_preserved") or info.get("pair_sketch_detail") is not None
            }
            # The reserve is an upper bound.  Some episodes legitimately have
            # more than max_sets - reserve protected v1 candidates; preserve
            # them and use only the capacity that remains rather than declaring
            # a valid packet unavailable.
            base_cap = max(int(max_sets) - int(n_context_pairs), len(protected))
            # inherited is sorted strongest first; remove only weak, non-pointer
            # conditional additions from its tail.
            for key, _, _ in reversed(inherited):
                if len(kept) <= base_cap:
                    break
                key = tuple(key)
                if key not in protected:
                    kept.remove(key)
            if len(kept) > base_cap:
                raise RuntimeError("could not trim non-protected v1 candidates for context allocation")

            # Direct context gates.  All large-array work remains on-device;
            # provenance is read only for the at-most n_context_pairs selected
            # pairs below.
            best = torch.full((len(ordinary_idx), len(context_idx)), -torch.inf,
                              device=X.device, dtype=Xc.dtype)
            best_q = torch.full(best.shape, -1, device=X.device, dtype=torch.int8)
            best_dir = torch.full(best.shape, -1, device=X.device, dtype=torch.int8)
            for q_index, quantile in enumerate((1.0 / 3.0, 0.5, 2.0 / 3.0)):
                cuts = torch.quantile(Xs[:, context_idx], quantile, dim=0)
                for d_index, gate in enumerate((Xs[:, context_idx] >= cuts, Xs[:, context_idx] < cuts)):
                    gate = gate.float()
                    for group in torch.unique(gs):
                        rows = gs == group
                        gate[rows] -= gate[rows].mean(0, keepdim=True)
                    score = torch.abs(Xc[:, ordinary_idx].T @ (yc[:, None] * gate)) / max(len(Xc), 1)
                    replace = score > best
                    best = torch.where(replace, score, best)
                    best_q = torch.where(replace, torch.full_like(best_q, q_index), best_q)
                    best_dir = torch.where(replace, torch.full_like(best_dir, d_index), best_dir)

            # Inspect a modest oversample to deduplicate and skip pairs already
            # in v1, while retaining a strict small context allocation.
            flat = torch.topk(best.reshape(-1), k=min(int(n_context_pairs) * 12, best.numel())).indices
            for raw in flat.tolist():
                a_local, c_local = divmod(int(raw), len(context_idx))
                a, c = int(ordinary_idx[a_local]), int(context_idx[c_local])
                key = tuple(sorted((a, c)))
                if key in kept:
                    continue
                kept.add(key)
                q = int(best_q[a_local, c_local].item())
                direction = int(best_dir[a_local, c_local].item())
                route_detail[key] = {
                    "route": "context_gate",
                    "screen_score": float(best[a_local, c_local].item()),
                    "gate_quantile": (1.0 / 3.0, 0.5, 2.0 / 3.0)[q],
                    "gate_direction": ("upper", "lower")[direction],
                }
                if len(route_detail) >= int(n_context_pairs) or len(kept) >= int(max_sets):
                    break

        sets = sorted(kept)
        need = set(sets)
        for key in sets:
            if len(key) >= 2:
                need |= {tuple(sorted(set(key) - {j})) for j in key}
        cvs = {}
        for size in sorted({len(key) for key in need}):
            same = sorted(key for key in need if len(key) == size)
            values = self.relation.cv_r2(torch.stack([X[:, list(key)] for key in same]), y, groups,
                                          n_folds=n_folds, seed=seed).tolist()
            cvs.update({key: float(value) for key, value in zip(same, values)})
        ranked = []
        for key in sets:
            gain = cvs[key] if len(key) == 1 else cvs[key] - max(
                cvs[tuple(sorted(set(key) - {j}))] for j in key)
            inherited_detail = inherited_info.get(key, {})
            info = {
                "proposal_family": "multiroute_hybrid_v11",
                "inherited_v1": key in inherited_info,
                "pointer_preserved": bool(inherited_detail.get("pointer_preserved")),
                "support_cv_r2": float(cvs[key]), "gain_over_subsets": float(gain),
                "route_detail": route_detail.get(key),
                "context_slots_reserved": int(max_sets - base_cap),
                "n_v1_candidates_before_reserve": int(len(inherited)),
            }
            if key in inherited_info:
                info["v1_detail"] = {k: inherited_detail.get(k) for k in
                                     ("pair_sketch_detail", "conditional_detail", "inherited_pointer_expansion")}
            ranked.append((key, float(cvs[key]), info))
        ranked.sort(key=lambda t: (-t[2]["gain_over_subsets"], -t[1], t[0]))
        for _, _, info in ranked:
            info["n_subset_fits"] = int(len(need))
            info["g_ho_support"] = None
        if len(ranked) > int(max_sets):
            raise RuntimeError("multiroute hybrid v1.1 cap violated")
        return ranked


    @torch.no_grad()
    def propose_sets(self, X, y, groups, neighbours6, *, names: list, views: dict, embed=None, n_paths=32,
                     seed: int = 0, device=None, strategy: str = "pointer") -> list:
        """Ranked unique variable sets on the given (support / proposal) rows only.

        Returns ``[(names_tuple, support_cv_r2, info), ...]``; names are
        ``<transform>:<column name>`` or ``<transform>:concept:<j>``; ``info`` has
        ``g_ho_support``, ``path_rank``, ``response_surface`` and, on the first
        element, ``n_sets_evaluated``."""
        n_paths = int(n_paths or 32)
        if device is not None:
            self.to(device)
        self.eval()
        ep = episode_from_arrays(X, y, groups, neighbours6, names=names, views=views, embed=embed,
                                 gene_cap=self.cfg.gene_cap)
        prep = self.prepare(ep)
        if strategy == "pointer":
            ranked = self.rank_sets(prep, n_paths=n_paths, seed=seed, g_ho_top=None)
        elif strategy == "conditional_beam":
            ranked = self.rank_sets_conditional_beam(prep, n_paths=n_paths, seed=seed)
        elif strategy == "pointer_expand":
            ranked = self.rank_sets_pointer_expansion(prep, n_paths=n_paths, seed=seed)
        elif strategy == "multiroute_expand":
            ranked = self.rank_sets_multiroute_expansion(prep, n_paths=n_paths, seed=seed)
        elif strategy == "multiroute_v2":
            ranked = self.rank_sets_multiroute_v2(prep, n_paths=n_paths, seed=seed)
        elif strategy == "multiroute_hybrid_v11":
            ranked = self.rank_sets_multiroute_hybrid_v11(prep, n_paths=n_paths, seed=seed)
        else:
            raise ValueError(f"unknown proposal strategy {strategy!r}")
        out = []
        for pos, (key, cv, info) in enumerate(ranked):
            # The pointer records sampling order; the conditional beam is a set
            # proposer, so its deterministic canonical order is the sorted set.
            order = [j for j in info.get("order", key)]
            info = dict(info)
            # A surface is final-shortlist characterization, not a ranking input.
            # Avoid fitting it for every member of the capped conditional beam.
            info["response_surface"] = self.response_surface(prep, order) if pos < 16 else None
            info["order"] = [prep["names"][j] for j in order]
            out.append((tuple(prep["names"][j] for j in key), float(cv), info))
        if out:  # every set whose relation was fitted (returned sets and the subsets used for their gain)
            out[0][2]["n_sets_evaluated"] = int(ranked[0][2]["n_subset_fits"])
        return out

    @torch.no_grad()
    def response_surface(self, prep: dict, order: Sequence[int], n_grid: int = 11) -> dict:
        n_s = prep["n_s"]
        V = prep["X"][:n_s, list(order)].float()
        y = prep["y"][:n_s]
        k = V.shape[1]
        axes = list(range(min(k, 2)))
        q = torch.linspace(0.05, 0.95, 21 if k == 1 else n_grid, device=V.device)
        med = V.median(0).values
        vals = [torch.quantile(V[:, a], q) for a in axes]
        if k == 1:
            G = vals[0][:, None]
        else:
            A, B = torch.meshgrid(vals[0], vals[1], indexing="ij")
            G = med[None].repeat(A.numel(), 1)
            G[:, 0], G[:, 1] = A.reshape(-1), B.reshape(-1)
        pred = self.relation.fit_predict(V, y, G)
        if k > 1:
            pred = pred.reshape(len(q), len(q))
        return {"axes": [prep["names"][order[a]] for a in axes], "quantiles": q.cpu().tolist(),
                "values": [v.cpu().tolist() for v in vals], "prediction": pred.cpu().tolist(),
                "others_at_median": [prep["names"][j] for j in order[2:]]}


# --------------------------------------------------------- arrays -> episode
def episode_from_arrays(X, y, groups, neighbours6, *, names, views, embed=None, gene_cap: int = 1024) -> Episode:
    """All given rows as support rows of one episode (no query rows)."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    _, gi = np.unique(groups.astype(str), return_inverse=True)
    nb6 = -np.ones((n, 6), dtype=np.int64) if neighbours6 is None else np.asarray(neighbours6, dtype=np.int64)
    views = dict(views or {})
    if not views:
        views = {"other": list(range(X.shape[1]))}
    cols, vid = [], []
    for v, vc in views.items():
        vc = np.asarray(vc, dtype=int)
        if view_id(v) == VIEW_VOCAB.index("gene") and len(vc) > gene_cap:
            Xc = X[:, vc] - np.stack([X[gi == g][:, vc].mean(0) for g in range(gi.max() + 1)])[gi]
            vc = np.sort(vc[np.argsort(-Xc.var(0), kind="stable")[:gene_cap]])  # response-free variance cap
        cols.append(vc)
        vid += [view_id(v)] * len(vc)
    cols = np.concatenate(cols)
    _, first = np.unique(cols, return_index=True)
    first = np.sort(first)
    cols, vid = cols[first], np.asarray(vid)[first]
    A6, A18 = neighbour_operators(nb6)
    B = X[:, cols]
    from .episodes import transform_block
    T = transform_block(B, A6, A18, np.arange(n))
    support = np.ones(n, dtype=bool)
    T = centre_scale(T, gi, support)
    tnames = [f"{t}:{names[c]}" for t in TRANSFORMS for c in cols]
    meta = TokenMeta(tnames, np.repeat(np.arange(4), len(cols)).astype(np.int64), np.tile(vid, 4).astype(np.int64))
    concept = None
    if embed is not None:
        from .episodes import ring2_indices
        concept = {"E": np.asarray(embed, dtype=np.float32), "own": np.arange(n), "nb6": nb6,
                   "nb18": ring2_indices(nb6)}
    return Episode(X=T, y=centre_scale(np.asarray(y, dtype=np.float64), gi, support), n_support=n,
                   groups=gi.astype(np.int64), coords=np.zeros((n, 2), dtype=np.float32), meta=meta,
                   concept=concept, kind="real")


# ===================================================================== v02
#: Statistic columns rescaled by 10 before the symmetric log (gains are a few % of variance).
_GAIN_STATS = ("cv_gain", "cv_gain_linear", "lopo_mean_gain")
GAIN_SCALE = 10.0   # network outputs predict 10 x realised query gain


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


@dataclass
class GainConfig:
    d_model: int = 192
    n_heads: int = 6
    n_blocks: int = 4
    n_samples: int = 256
    point_hidden: int = 64
    patient_dim: int = 16
    max_patients: int = 64
    n_stats: int = 16
    m_top: int = 64


class GainSearchModel(nn.Module):
    """Learned search v02: predicts the held-out general gain of every candidate and of
    pairs, identity-free.

    * per candidate: DeepSets over ``n_samples`` support spots of (value, target,
      product, nbr6 mean of the value) + a patient-code embedding (codes permuted per
      episode in training), mean-pooled, concatenated with the symlog support
      statistics (``support_statistics.STAT_NAMES``);
    * Set Transformer body: ``n_blocks`` pre-norm blocks (d, heads, scaled-dot-product
      attention) over all candidates;
    * heads: single gain; pair gain from a symmetric MLP over (h_a + h_b, h_a * h_b)."""

    kind = "gain_v02"

    def __init__(self, cfg: GainConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or GainConfig()
        h = cfg.point_hidden
        self.patient = nn.Embedding(cfg.max_patients, cfg.patient_dim)
        self.point = nn.Sequential(nn.Linear(4 + cfg.patient_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                   nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.GELU())
        self.inp = nn.Sequential(nn.Linear(h + cfg.n_stats, cfg.d_model), nn.GELU(),
                                 nn.Linear(cfg.d_model, cfg.d_model))
        self.encoder = SetEncoder(cfg.d_model, cfg.n_heads, cfg.n_blocks, "sab")
        d = cfg.d_model
        self.single = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.meta: dict = {}
        self._pool_cache = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- io
    def save(self, path, **meta) -> None:
        torch.save({"kind": self.kind, "config": asdict(self.cfg), "state_dict": self.state_dict(),
                    "meta": {**self.meta, **meta}}, str(path))

    @classmethod
    def load(cls, path, map_location="cpu") -> "GainSearchModel":
        ck = torch.load(str(path), map_location=map_location, weights_only=False)
        m = cls(GainConfig(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        m.meta = dict(ck.get("meta", {}))
        return m.eval()

    # -- encoding
    def _stats_in(self, S: torch.Tensor, stat_names: Sequence[str]) -> torch.Tensor:
        scale = torch.ones(S.shape[1], device=S.device)
        for nm in _GAIN_STATS:
            if nm in stat_names:
                scale[list(stat_names).index(nm)] = GAIN_SCALE
        return _symlog(S * scale)

    def encode(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, nb6: np.ndarray | None, S: np.ndarray,
               stat_names: Sequence[str], *, generator: torch.Generator | None = None,
               permute_patients: bool = False) -> torch.Tensor:
        """(p, d) candidate encodings from SUPPORT rows (X (n_s, p), y (n_s,))."""
        dev = self.device
        n, p = X.shape
        g = torch.Generator().manual_seed(0) if generator is None else generator
        idx = torch.randperm(n, generator=g)[:self.cfg.n_samples]
        ii = idx.numpy()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
        xs = Xt[idx.to(dev)]                                                   # (s, p)
        if nb6 is not None:
            nb = torch.as_tensor(np.asarray(nb6)[ii], device=dev)
            valid = (nb >= 0) & (nb < n)
            gathered = Xt[nb.clamp(0, n - 1)] * valid[..., None]               # (s, 6, p)
            cnt = valid.sum(1, keepdim=True).clamp(min=1)
            lag = torch.where(valid.any(1, keepdim=True), gathered.sum(1) / cnt, xs)
        else:
            lag = xs
        ys = torch.as_tensor(np.asarray(y)[ii], dtype=torch.float32, device=dev)[:, None].expand_as(xs)
        _, codes = np.unique(np.asarray(groups)[ii], return_inverse=True)
        codes = torch.as_tensor(codes, device=dev) % self.cfg.max_patients
        if permute_patients:
            codes = torch.randperm(self.cfg.max_patients, generator=g).to(dev)[codes]
        pe = self.patient(codes)                                               # (s, e)
        feat = torch.stack([xs, ys, xs * ys, lag], -1)                         # (s, p, 4)
        feat = torch.cat([feat, pe[:, None, :].expand(-1, p, -1)], -1)
        pooled = self.rho(self.point(feat).mean(0))                            # (p, h)
        St = torch.as_tensor(np.asarray(S), dtype=torch.float32, device=dev)
        T = self.inp(torch.cat([pooled, self._stats_in(St, stat_names)], 1))
        return self.encoder(T)

    def single_scores(self, H: torch.Tensor) -> torch.Tensor:
        return self.single(H)[:, 0]

    def pair_scores(self, H: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        a, b = H[pairs[:, 0]], H[pairs[:, 1]]
        return self.pair(torch.cat([a + b, a * b], 1))[:, 0]

    # -- rank_by interface of search_v02.search_pool
    def prepare(self, pool) -> None:
        ep, n_s = pool.ep, pool.ep.n_support
        nb = None if ep.nb6 is None else np.asarray(ep.nb6)[:n_s]
        with torch.no_grad():
            H = self.encode(ep.X[:n_s], ep.y[:n_s], ep.groups[:n_s], nb, pool.S, pool.stat_names)
            self._pool_cache = (H, self.single_scores(H).float().cpu().numpy())

    def singles(self, S, stat_names) -> np.ndarray:
        return self._pool_cache[1]

    def pairs(self, S, stat_names, pairs_idx, pair_stats) -> np.ndarray:
        H = self._pool_cache[0]
        with torch.no_grad():
            return self.pair_scores(H, torch.as_tensor(np.asarray(pairs_idx), device=H.device)).float().cpu().numpy()

    def propose_sets(self, X, y, groups, neighbours6, *, names: list, views: dict, embed=None, n_paths=None,
                     seed: int = 0, device=None, strategy: str = "pointer", budget: int = 64,
                     nuisance=None, rank_by=None) -> list:
        """``[(prefixed-name tuple, support_cv_gain, info)]`` best first (support rows only):
        network scores (or an external ``rank_by``) -> beam -> closed-form refit of B sets."""
        from .search_v02 import build_pool, search_pool
        if device is not None:
            self.to(device)
        self.eval()
        pool = build_pool(X, y, groups, neighbours6, names=names, views=views, embed=embed, nuisance=nuisance,
                          device=self.device)
        ranked = search_pool(pool, self if rank_by is None else rank_by, budget=int(budget), m_top=self.cfg.m_top,
                             device=self.device)
        self._pool_cache = None
        return [(tuple(pool.names[j] for j in key), cv, info) for key, cv, info in ranked]

    # -- training
    def episode_loss(self, ep, S: np.ndarray, stat_names: Sequence[str], labels: dict, *,
                     generator: torch.Generator | None = None, margin: float = 0.1, n_rank: int = 512) -> dict:
        """Huber on 10 x realised query gain + pairwise margin ranking, singles and pairs."""
        n_s = ep.n_support
        nb = None if ep.nb6 is None else np.asarray(ep.nb6)[:n_s]
        g = torch.Generator().manual_seed(0) if generator is None else generator
        H = self.encode(ep.X[:n_s], ep.y[:n_s], ep.groups[:n_s], nb, S, stat_names, generator=g,
                        permute_patients=True)
        dev = H.device
        ps = self.single_scores(H)
        ts = torch.as_tensor(labels["single"], dtype=torch.float32, device=dev) * GAIN_SCALE
        out = {"huber_single": F.huber_loss(ps, ts), "rank_single": _rank_loss(ps, ts, g, margin, n_rank)}
        if labels.get("pairs") is not None and len(labels["pairs"]):
            pidx = torch.as_tensor(np.asarray(labels["pairs"]), device=dev)
            pp = self.pair_scores(H, pidx)
            tp = torch.as_tensor(labels["pair"], dtype=torch.float32, device=dev) * GAIN_SCALE
            out["huber_pair"] = F.huber_loss(pp, tp)
            out["rank_pair"] = _rank_loss(pp, tp, g, margin, n_rank)
        out["total"] = sum(out.values())
        return out


def _rank_loss(pred: torch.Tensor, target: torch.Tensor, g: torch.Generator, margin: float, n: int) -> torch.Tensor:
    """Margin ranking over sampled candidate pairs of one episode; half of the pairs
    contrast a top-32 candidate (by realised gain) with a random one."""
    m = len(pred)
    if m < 2:
        return pred.sum() * 0
    top = torch.argsort(target, descending=True)[:32].cpu()
    i = torch.cat([top[torch.randint(len(top), (n // 2,), generator=g)], torch.randint(m, (n - n // 2,), generator=g)])
    j = torch.randint(m, (n,), generator=g)
    i, j = i.to(pred.device), j.to(pred.device)
    diff = target[i] - target[j]
    keep = diff.abs() > 1e-3 * GAIN_SCALE
    if not keep.any():
        return pred.sum() * 0
    return F.relu(margin - torch.sign(diff[keep]) * (pred[i][keep] - pred[j][keep])).mean()
