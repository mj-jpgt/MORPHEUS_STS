"""Candidate tokens: morphology concepts and the per-variable dataset encoder.

* :class:`ConceptLayer` maps the frozen 1536-d H-Optimus-0 spot embedding to 64
  morphology concept activations (Linear 1536->256, LayerNorm, GELU, Linear
  256->64, LayerNorm), trained jointly with the search model.  ``decorrelation``
  is the mean squared off-diagonal correlation of the concepts;
  ``detached`` evaluates the layer with detached weights, used for concept
  *targets* so a target can never pull the layer towards being predictable.
* :class:`CandidateTokenEncoder` (E_sample) turns one candidate variable's
  behaviour on the support set into a d-dimensional token: a DeepSets network
  over 256 sampled support observations of (x_ij, y_i, x_ij*y_i, x_ij^2 - 1,
  coordinates) mean-pooled, plus four sufficient statistics computed on every
  support row (r(x, y), r(x^2, y), r(x, y^2), the share of support patients whose
  within-patient covariance has the pooled sign) and transform/view embeddings.
  Tokens are built in chunks of 1,024 with gradient checkpointing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .episodes import TRANSFORMS, VIEW_VOCAB


class ConceptLayer(nn.Module):
    def __init__(self, in_dim: int = 1536, hidden: int = 256, n_concepts: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, n_concepts), nn.LayerNorm(n_concepts))
        self.n_concepts = n_concepts

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        return self.net(E)

    def detached(self, E: torch.Tensor) -> torch.Tensor:
        params = {k: v.detach() for k, v in self.named_parameters()}
        with torch.no_grad():
            return torch.func.functional_call(self, params, (E,))

    @staticmethod
    def decorrelation(C: torch.Tensor) -> torch.Tensor:
        C = C.float()
        Z = (C - C.mean(0)) / (C.std(0) + 1e-5)
        R = Z.T @ Z / max(len(Z) - 1, 1)
        off = R - torch.diag(torch.diag(R))
        k = R.shape[0]
        return (off ** 2).sum() / max(k * (k - 1), 1)


def neighbour_gather_mean(C: torch.Tensor, nb: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
    """Mean of ``C`` over -1-padded local neighbour indices (own row where none)."""
    valid = (nb >= 0).to(C.dtype)
    G = C[nb.clamp(min=0)] * valid[..., None]
    cnt = valid.sum(1, keepdim=True)
    return torch.where(cnt > 0, G.sum(1) / cnt.clamp(min=1), C[own])


def concept_transforms(C: torch.Tensor, own: torch.Tensor, nb6: torch.Tensor, nb18: torch.Tensor) -> torch.Tensor:
    """(n, 4K) [own | nbr6 | nbr18 | contrast] concept values for the episode rows."""
    o = C[own]
    n6 = neighbour_gather_mean(C, nb6, own)
    n18 = neighbour_gather_mean(C, nb18, own)
    return torch.cat([o, n6, n18, o - n6], dim=1)


def _corr_cols(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson r of each column of A (n, p) with b (n,)."""
    A = A - A.mean(0)
    b = b - b.mean()
    num = (A * b[:, None]).sum(0)
    den = A.norm(dim=0) * b.norm() + 1e-8
    return num / den


def sufficient_stats(Xs: torch.Tensor, ys: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """(p, 4) cheap statistics on all support rows."""
    r1 = _corr_cols(Xs, ys)
    r2 = _corr_cols(Xs * Xs, ys)
    r3 = _corr_cols(Xs, ys * ys)
    _, inv = torch.unique(groups, return_inverse=True)
    n_g = int(inv.max()) + 1
    cov = torch.zeros(n_g, Xs.shape[1], device=Xs.device, dtype=Xs.dtype)
    cov.index_add_(0, inv, Xs * ys[:, None])
    agree = (torch.sign(cov) == torch.sign(r1)[None]).to(Xs.dtype).mean(0) * 2 - 1
    return torch.stack([r1, r2, r3, agree], dim=1)


class CandidateTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 192, obs_hidden: int = 64, n_samples: int = 256, chunk: int = 1024):
        super().__init__()
        self.n_samples, self.chunk = n_samples, chunk
        self.phi = nn.Sequential(nn.Linear(6, obs_hidden), nn.GELU(), nn.Linear(obs_hidden, obs_hidden), nn.GELU(),
                                 nn.Linear(obs_hidden, obs_hidden))
        self.t_emb = nn.Embedding(len(TRANSFORMS), 16)
        self.v_emb = nn.Embedding(len(VIEW_VOCAB), 16)
        self.out = nn.Sequential(nn.Linear(obs_hidden + 4 + 32, d_model), nn.GELU(), nn.Linear(d_model, d_model),
                                 nn.LayerNorm(d_model))

    def _pool(self, Xc: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # Xc (m, c) -> (c, obs_hidden)
        m, c = Xc.shape
        x = Xc.T[..., None]                               # (c, m, 1)
        yy = y[None, :, None].expand(c, m, 1)
        cc = ctx[None].expand(c, m, ctx.shape[1])
        feats = torch.cat([x, yy, x * yy, x * x - 1.0, cc], dim=-1)
        return self.phi(feats).mean(1)

    def forward(self, Xs: torch.Tensor, ys: torch.Tensor, coords_s: torch.Tensor, groups_s: torch.Tensor,
                transform: torch.Tensor, view: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        n_s, p = Xs.shape
        m = min(self.n_samples, n_s)
        gdev = generator.device if generator is not None else Xs.device
        idx = torch.randperm(n_s, generator=generator, device=gdev)[:m].to(Xs.device)
        Xm, ym, cm = Xs[idx], ys[idx], coords_s[idx]
        pooled = []
        for start in range(0, p, self.chunk):
            Xc = Xm[:, start:start + self.chunk]
            if torch.is_grad_enabled() and (Xc.requires_grad or any(q.requires_grad for q in self.phi.parameters())):
                pooled.append(checkpoint(self._pool, Xc, ym, cm, use_reentrant=False))
            else:
                pooled.append(self._pool(Xc, ym, cm))
        pooled = torch.cat(pooled, 0)
        stats = sufficient_stats(Xs.float(), ys.float(), groups_s).to(pooled.dtype)
        z = torch.cat([pooled, stats, self.t_emb(transform).to(pooled.dtype), self.v_emb(view).to(pooled.dtype)], 1)
        return self.out(z)
