"""The gain predictor g_theta of learned search v03 (spec 1.2, 1.3, 1.5).

Candidate token = [DeepSets(mean + max, 128) over 256 support spots of (x, y_resid, x * y_resid,
nbr6 mean of x, nbr6 mean of y_resid, patient one-hot -> 8, section one-hot -> 4) | symlog
transfer diagnostics] -> d; + learned embeddings of the baseline B and the level L.  Set
Transformer (4 pre-norm blocks, d = 192, 6 heads).  Heads, all in units of 10 x R^2:
single -> (mu_delta, log sigma_delta); pair -> (mu_sup, mu_delta, log sigma_delta) from a rank-32
symmetric bilinear form (8 channels) + a symmetric linear term, then a two-layer MLP, so every
pair of an episode is scored with matrix products.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .relation_v03 import BASELINES, DIAG_NAMES, LEVELS

SCALE = 10.0           # outputs are 10 x R^2
LOGSIG_MIN, LOGSIG_MAX = -4.0, 3.0


@dataclass
class ConfigV03:
    d_model: int = 192
    n_heads: int = 6
    n_blocks: int = 4
    n_samples: int = 256
    pool_dim: int = 128
    rank: int = 32
    channels: int = 8
    max_codes: int = 64


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class GainModelV03(nn.Module):
    kind = "gain_v03"

    def __init__(self, cfg: ConfigV03 | None = None):
        super().__init__()
        from .model import SetEncoder
        self.cfg = cfg = cfg or ConfigV03()
        h, d = cfg.pool_dim, cfg.d_model
        self.patient = nn.Linear(cfg.max_codes, 8, bias=False)   # one-hot -> 8
        self.section = nn.Linear(cfg.max_codes, 4, bias=False)   # one-hot -> 4
        self.phi = nn.Sequential(nn.Linear(5 + 12, h), nn.GELU(), nn.Linear(h, h))
        self.inp = nn.Sequential(nn.Linear(2 * h + len(DIAG_NAMES), d), nn.GELU(), nn.Linear(d, d))
        self.b_emb = nn.Embedding(len(BASELINES), d)
        self.l_emb = nn.Embedding(len(LEVELS), d)
        self.encoder = SetEncoder(d, cfg.n_heads, cfg.n_blocks, "sab")
        self.single = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.bil = nn.Linear(d, cfg.channels * cfg.rank, bias=False)
        self.lin = nn.Linear(d, cfg.channels)
        self.pair_mlp = nn.Sequential(nn.Linear(cfg.channels, 64), nn.GELU(), nn.Linear(64, 3))
        self.meta: dict = {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path, **meta) -> None:
        torch.save({"kind": self.kind, "config": asdict(self.cfg), "state_dict": self.state_dict(),
                    "meta": {**self.meta, **meta}}, str(path))

    @classmethod
    def load(cls, path, map_location="cpu") -> "GainModelV03":
        ck = torch.load(str(path), map_location=map_location, weights_only=False)
        m = cls(ConfigV03(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        m.meta = dict(ck.get("meta", {}))
        return m.eval()

    def propose_sets(self, X, y, groups, neighbours6, *, names, views, embed=None, n_paths=None, seed: int = 0,
                     device=None, strategy=None, K: int = 64, B: str = "nuis", nuisance=None, raw=None, family=None):
        """``propose.propose`` contract (support / proposal rows only): ``search_v03.propose_sets_v03``."""
        from .search_v03 import propose_sets_v03
        if device is not None:
            self.to(device)
        return propose_sets_v03(self, X, y, groups, neighbours6, names=names, views=views, embed=embed, B=B, K=K,
                                nuisance=nuisance, device=self.device, raw=raw, family=family)

    def concept_values(self, embed) -> np.ndarray:
        """The frozen morphology representation for callers that ask a checkpoint for its
        "concepts" (discovery-packet ``morph_concepts`` view): cosine similarities to the kept v03
        dictionary centres, standardised with ``self.morph_stats`` when a proposal set them, else on
        the given rows.  No learned layer is involved."""
        from .morphology_dictionary import load_dictionary_v03, similarities_v03, standardise_stats
        E = np.asarray(embed, dtype=np.float32)
        mu, sd = getattr(self, "morph_stats", None) or standardise_stats(E)
        return similarities_v03(E, load_dictionary_v03()["centres"], mu, sd).astype(np.float64)

    concepts = concept_values

    # ---------------------------------------------------------------- encode
    def encode(self, Xs: np.ndarray, y_resid: np.ndarray, groups: np.ndarray, sections: np.ndarray | None,
               nb6: np.ndarray | None, diag: np.ndarray, B: str, L: str, *, generator=None,
               permute_codes: bool = False) -> torch.Tensor:
        """(p, d) candidate representations from SUPPORT rows only."""
        dev, cfg = self.device, self.cfg
        n, p = Xs.shape
        g = torch.Generator().manual_seed(0) if generator is None else generator
        idx = torch.randperm(n, generator=g)[:cfg.n_samples].numpy()
        X = torch.as_tensor(np.asarray(Xs), dtype=torch.float32, device=dev)
        yr = torch.as_tensor(np.asarray(y_resid), dtype=torch.float32, device=dev)
        yr = yr / yr.std().clamp(min=1e-6)
        it = torch.as_tensor(idx, device=dev)
        xs, ys = X[it], yr[it]
        if nb6 is not None:
            nb = torch.as_tensor(np.asarray(nb6)[idx], device=dev)
            valid = (nb >= 0) & (nb < n)
            nbc = nb.clamp(0, n - 1)
            cnt = valid.sum(1, keepdim=True).clamp(min=1)
            lag_x = torch.where(valid.any(1, keepdim=True), (X[nbc] * valid[..., None]).sum(1) / cnt, xs)
            lag_y = torch.where(valid.any(1), (yr[nbc] * valid).sum(1) / cnt[:, 0], ys)
        else:
            lag_x, lag_y = xs, ys

        def onehot(codes, layer):
            c = torch.as_tensor(np.unique(np.asarray(codes)[idx].astype(str), return_inverse=True)[1], device=dev)
            if permute_codes:
                c = torch.randperm(cfg.max_codes, generator=g).to(dev)[c % cfg.max_codes]
            return layer(F.one_hot(c % cfg.max_codes, cfg.max_codes).float())

        pe = onehot(groups, self.patient)
        se = onehot(groups if sections is None else sections, self.section)
        ctx = torch.cat([pe, se], 1)[:, None, :].expand(-1, p, -1)
        feat = torch.cat([torch.stack([xs, ys[:, None].expand_as(xs), xs * ys[:, None], lag_x,
                                       lag_y[:, None].expand_as(xs)], -1), ctx], -1)          # (s, p, 17)
        e = self.phi(feat)
        pooled = torch.cat([e.mean(0), e.max(0).values], 1)
        D = torch.as_tensor(np.asarray(diag), dtype=torch.float32, device=dev).clone()
        D[:, 0:2] = D[:, 0:2] * SCALE                                     # gains in 10 x R^2
        T = self.inp(torch.cat([pooled, _symlog(D)], 1))
        T = T + self.b_emb.weight[BASELINES.index(B)] + self.l_emb.weight[LEVELS.index(L)]
        return self.encoder(T)

    # ----------------------------------------------------------------- heads
    def single_head(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.single(H)
        return o[:, 0], o[:, 1].clamp(LOGSIG_MIN, LOGSIG_MAX)

    def pair_head(self, H: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(mu_sup, mu_delta, log sigma) for index pairs (m, 2)."""
        c, r = self.cfg.channels, self.cfg.rank
        Z = self.bil(H).view(-1, c, r)
        v = self.lin(H)
        a, b = pairs[:, 0], pairs[:, 1]
        o = self.pair_mlp((Z[a] * Z[b]).sum(-1) / r ** 0.5 + v[a] + v[b])
        return o[:, 0], o[:, 1], o[:, 2].clamp(LOGSIG_MIN, LOGSIG_MAX)

    def pair_matrix(self, H: torch.Tensor, idx: torch.Tensor | None = None, chunk: int = 256) -> torch.Tensor:
        """(q, q, 3) pair outputs among candidates ``idx`` (all by default), chunked matrix products."""
        c, r = self.cfg.channels, self.cfg.rank
        Hs = H if idx is None else H[idx]
        Z = self.bil(Hs).view(-1, c, r)
        v = self.lin(Hs)
        q = len(Hs)
        out = torch.empty((q, q, 3), device=H.device)
        for a0 in range(0, q, chunk):
            bil = torch.einsum("icr,jcr->ijc", Z[a0:a0 + chunk], Z) / r ** 0.5
            o = self.pair_mlp(bil + v[a0:a0 + chunk, None] + v[None])
            o[..., 2] = o[..., 2].clamp(LOGSIG_MIN, LOGSIG_MAX)
            out[a0:a0 + chunk] = o
        return out


# ------------------------------------------------------------------- loss
def gaussian_nll(t: torch.Tensor, mu: torch.Tensor, logsig: torch.Tensor) -> torch.Tensor:
    return (0.5 * ((t - mu) / logsig.exp()) ** 2 + logsig).mean()


def listmle_top_weighted(scores: torch.Tensor, realised: torch.Tensor) -> torch.Tensor:
    """ListMLE with position weights 1 / log(1 + rank) (rank 1 = best realised)."""
    order = torch.argsort(realised, descending=True)
    s = scores[order]
    lse = torch.logcumsumexp(s.flip(0), 0).flip(0)                      # log sum_{j >= i} exp(s_j)
    rank = torch.arange(1, len(s) + 1, device=s.device, dtype=s.dtype)
    w = 1.0 / torch.log1p(rank)
    return (w * (lse - s)).sum() / w.sum()


def episode_loss(model: GainModelV03, ep: dict, *, lam: float = 1.0, generator=None) -> dict:
    """``ep``: one labelled episode (see ``train_v03.label_episode``)."""
    clip = lambda a: torch.as_tensor(np.clip(np.nan_to_num(np.asarray(a, dtype=np.float64)), -1, 1) * SCALE,  # noqa: E731
                                     dtype=torch.float32, device=model.device)
    fin = lambda *a: torch.as_tensor(np.all([np.isfinite(np.asarray(x, dtype=np.float64)) for x in a], 0),  # noqa: E731
                                     device=model.device)   # non-finite labels are dropped, never trained on
    s = ep["n_support"]
    H = model.encode(ep["X"][:s], ep["y_resid"], ep["groups"][:s], ep["sections"][:s], ep["nb6_s"], ep["diag"],
                     ep["B"], ep["L"], generator=generator, permute_codes=True)
    mu_d, ls_d = model.single_head(H)
    si = torch.as_tensor(ep["single_idx"], device=model.device)          # labelled singles
    fs = fin(ep["single_gsup"], ep["single_g"])
    su = torch.as_tensor(ep["single_uniform"], device=model.device) & fs  # mask over single_idx
    gsup_s, g_s = clip(ep["single_gsup"]), clip(ep["single_g"])
    out = {"nll_single": gaussian_nll((gsup_s - g_s)[su], mu_d[si][su], ls_d[si][su])}
    pr = torch.as_tensor(ep["pairs"], device=model.device)
    fp = fin(ep["pair_gsup"], ep["pair_g"])
    pu = torch.as_tensor(ep["pair_uniform"], device=model.device) & fp
    gsup_p, g_p = clip(ep["pair_gsup"]), clip(ep["pair_g"])
    ms, md, ls = model.pair_head(H, pr)
    out["nll_pair"] = gaussian_nll((gsup_p - g_p)[pu], md[pu], ls[pu])
    out["mse_pair_sup"] = ((gsup_p - ms)[pu] ** 2).mean()
    keep = torch.cat([fs, fp])
    scores = torch.cat([gsup_s - mu_d[si], gsup_p - md])[keep]
    out["rank"] = lam * listmle_top_weighted(scores, torch.cat([g_s, g_p])[keep])
    if ep.get("decomp_g") is not None:   # the same pairs under B_decomp
        Hd = model.encode(ep["X"][:s], ep["y_resid_nuis"], ep["groups"][:s], ep["sections"][:s], ep["nb6_s"],
                          ep["diag_nuis"], "decomp", ep["L"], generator=generator, permute_codes=True)
        msd, mdd, lsd = model.pair_head(Hd, pr)
        dsup, dg = clip(ep["decomp_gsup"]), clip(ep["decomp_g"])
        fd = fin(ep["decomp_gsup"], ep["decomp_g"])
        pud = torch.as_tensor(ep["pair_uniform"], device=model.device) & fd
        out["nll_pair_decomp"] = gaussian_nll((dsup - dg)[pud], mdd[pud], lsd[pud])
        out["mse_pair_sup_decomp"] = ((dsup - msd)[pud] ** 2).mean()
        out["rank_decomp"] = lam * listmle_top_weighted((dsup - mdd)[fd], dg[fd])
    out["total"] = sum(out.values())
    return out
