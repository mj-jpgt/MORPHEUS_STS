"""Learned search v02: candidate pool, beam over sets, closed-form refit.

One code path for the network and for every score-based baseline
(:func:`search_by_scores`):

1. pool: the four spatial versions (own / nbr6 / nbr18 / contrast) of every
   given column plus the 64 ``morph:kNN`` similarities to the frozen morphology
   dictionary (when ``embed`` is given), centred within patient; support
   statistics (``support_statistics.candidate_statistics``) for every candidate.
2. beam: singles ordered by score; a pair score matrix over the top-M singles;
   triples and quads grown from the best ``width`` pairs / triples, a set scored by
   the mean pair score over its members; per-size quotas of the budget B.
3. refit: the B sets in closed form on support rows (batched per set size),
   final ranking by support LOPO general gain minus ``PARSIMONY`` per variable
   beyond the first (a superset must beat its size class, else a smaller set
   ranks first).  ``n_sets_evaluated`` = B.

``rank_by`` is a statistic name (``"cv_gain"``: the support-fit screen), a (p,)
array in pool order, or an object with ``singles(S, stat_names) -> (p,)`` and
optionally ``pairs(S, stat_names, pairs_idx, pair_stats) -> (m,)`` and
``prepare(pool)`` (the network uses ``prepare`` to see the pool's values).
Without ``pairs`` a pair scores the mean of its members' single scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from . import support_statistics as ss
from .episodes import Episode, VIEW_VOCAB, centre_scale

M_TOP = 64
WIDTH = 16
QUOTAS = (0.25, 0.25, 0.25, 0.25)
PARSIMONY = 0.002   # support LOPO gain each extra variable must add in the final ranking


@dataclass
class Pool:
    ep: Episode            # all given rows as support rows; X (n, p) candidate values
    S: np.ndarray          # (p, n_stats) support statistics
    stat_names: list
    names: list            # p prefixed candidate names
    nuisance: np.ndarray | None


def build_pool(X, y, groups, neighbours6, *, names, views, embed=None, nuisance=None, gene_cap: int = 1024,
               morph_dictionary: str | None = None, device=None) -> Pool:
    from .model import episode_from_arrays
    from .propose import pad_neighbours

    X = np.asarray(X, dtype=np.float64)
    names, views = [str(c) for c in names], {k: list(v) for k, v in dict(views or {}).items()}
    if not views:
        views = {"other": list(range(X.shape[1]))}
    if embed is not None:
        from .morphology_dictionary import DEFAULT_DIR, morph_view
        M, mnames = morph_view(np.asarray(embed, dtype=np.float32), morph_dictionary or DEFAULT_DIR)
        views["morph"] = list(range(X.shape[1], X.shape[1] + M.shape[1]))
        X, names = np.concatenate([X, M], 1), names + mnames
    nb = pad_neighbours(neighbours6, len(X))
    ep = episode_from_arrays(X, y, groups, nb, names=names, views=views, embed=None, gene_cap=gene_cap)
    ep.nb6 = None if nb is None else nb[:, :6]
    nz = None
    if nuisance is not None:
        nz = centre_scale(np.asarray(nuisance, dtype=np.float64).reshape(len(X), -1)[:, 0], ep.groups,
                          np.ones(len(X), dtype=bool)).astype(np.float64)
    ep.nuisance = nz
    S, sn = ss.episode_statistics(ep, device=device)
    return Pool(ep, S, sn, list(ep.meta.names), nz)


def pool_from_episode(ep: Episode, *, device=None) -> Pool:
    """Pool on an episode's support rows (training and development)."""
    S, sn = ss.episode_statistics(ep, device=device)
    return Pool(ep, S, sn, list(ep.meta.names), getattr(ep, "nuisance", None))


def _set_score(P: np.ndarray, members: Sequence[int]) -> float:
    return float(np.mean([P[a, b] for a, b in combinations(members, 2)]))


def beam_sets(single: np.ndarray, pair_matrix: np.ndarray | None, top: np.ndarray, *, budget: int,
              width: int = WIDTH, k_max: int = 4, quotas: Sequence[float] = QUOTAS) -> list:
    """Up to ``budget`` sets (tuples of pool indices) with their beam scores.

    ``pair_matrix`` (M, M) scores pairs of ``top`` (pool indices of the top-M singles)."""
    order = np.argsort(-single, kind="stable")
    M = len(top)
    if pair_matrix is None:
        pair_matrix = 0.5 * (single[top][:, None] + single[top][None, :])
    P = np.asarray(pair_matrix, dtype=np.float64)
    by_k = {1: [((int(i),), float(single[i])) for i in order]}
    if k_max >= 2 and M >= 2:
        iu = np.triu_indices(M, 1)
        o = np.argsort(-P[iu], kind="stable")
        by_k[2] = [((int(iu[0][j]), int(iu[1][j])), float(P[iu][j])) for j in o]
    for k in range(3, k_max + 1):
        if k - 1 not in by_k:
            break
        seen, grown = set(), []
        for base, _ in by_k[k - 1][:width]:
            for c in range(M):
                if c in base:
                    continue
                key = tuple(sorted(base + (c,)))
                if key not in seen:
                    seen.add(key)
                    grown.append((key, _set_score(P, key)))
        grown.sort(key=lambda t: -t[1])
        by_k[k] = grown
    for k in range(2, k_max + 1):   # local top-M indices -> pool indices
        if k in by_k:
            by_k[k] = [(tuple(int(top[a]) for a in key), s) for key, s in by_k[k]]
    want = {k: int(round(budget * quotas[k - 1])) for k in range(1, k_max + 1)}
    want[1] += budget - sum(want.values())
    out, spare = [], 0
    for k in range(k_max, 0, -1):          # unused quota of larger sizes passes to smaller ones
        take = by_k.get(k, [])[:want[k] + spare]
        spare = want[k] + spare - len(take)
        out += [(key, s, k) for key, s in take]
    return out[:budget]


def _single_scores(rank_by: Any, pool: Pool) -> np.ndarray:
    if isinstance(rank_by, str):
        return pool.S[:, pool.stat_names.index(rank_by)].astype(np.float64)
    if hasattr(rank_by, "singles"):
        return np.asarray(rank_by.singles(pool.S, pool.stat_names), dtype=np.float64)
    s = np.asarray(rank_by, dtype=np.float64)
    if s.shape != (len(pool.names),):
        raise ValueError(f"rank_by has shape {s.shape}, pool has {len(pool.names)} candidates")
    return s


def search_pool(pool: Pool, rank_by: Any = "cv_gain", *, budget: int = 64, m_top: int = M_TOP, width: int = WIDTH,
                k_max: int = 4, pair_stats: bool = False, device=None) -> list:
    """Beam + closed-form refit on a pool: ``[(index tuple, support_cv_gain, info)]`` best first."""
    if hasattr(rank_by, "prepare"):
        rank_by.prepare(pool)
    single = _single_scores(rank_by, pool)
    single = np.where(np.isfinite(single), single, -np.inf)
    top = np.argsort(-single, kind="stable")[:min(m_top, len(single))]
    P = None
    n_extra = 0
    if hasattr(rank_by, "pairs") and len(top) >= 2:
        iu = np.triu_indices(len(top), 1)
        pidx = np.stack([top[iu[0]], top[iu[1]]], 1)
        pst = None
        if pair_stats:   # support fits of every top-M pair: counted as tested relationships
            n_s = pool.ep.n_support
            r = ss.fit_sets(pool.ep.X[:n_s][:, pidx], pool.ep.y[:n_s], pool.ep.groups[:n_s],
                            nuis_s=None if pool.nuisance is None else pool.nuisance[:n_s], device=device)
            lg = r["lopo_gain"]
            pst = {"cv_gain": r["cv_gain"], "lopo_mean_gain": lg.mean(0), "lopo_var_gain": lg.var(0)}
            n_extra = len(pidx)
        sc = np.asarray(rank_by.pairs(pool.S, pool.stat_names, pidx, pst), dtype=np.float64)
        P = np.full((len(top), len(top)), -np.inf)
        P[iu] = sc
        P[(iu[1], iu[0])] = sc
    sets = beam_sets(single, P, top, budget=budget, width=width, k_max=k_max)
    n_s = pool.ep.n_support
    cv = ss.support_cv_gains(pool.ep.X[:n_s], pool.ep.y[:n_s], pool.ep.groups[:n_s], [s for s, _, _ in sets],
                             nuisance=None if pool.nuisance is None else pool.nuisance[:n_s], device=device)
    # parsimony: each variable beyond the first must add PARSIMONY of held-out variance to outrank its size
    score = cv - PARSIMONY * (np.array([len(s) for s, _, _ in sets]) - 1)
    ranked = sorted(range(len(sets)), key=lambda i: (-score[i], i))
    n_tested = len(sets) + n_extra
    return [(sets[i][0], float(cv[i]), {"beam_score": float(sets[i][1]), "k": sets[i][2], "path_rank": int(i),
                                        "g_ho_support": None, "response_surface": None,
                                        "n_sets_evaluated": n_tested}) for i in ranked]


def search_by_scores(X, y, groups, neighbours6, *, names, views, embed=None, coords=None, rank_by: Any = "cv_gain",
                     budget: int = 64, seed: int = 0, device=None, nuisance=None, pair_stats: bool = False,
                     n_paths=None) -> list:
    """``propose_sets``-format output for any scorer (no checkpoint needed):
    ``[(prefixed-name tuple, support_cv_gain, info)]``; every ``info["n_sets_evaluated"]``
    is the number of relationships fitted (B, plus the top-M pairs when ``pair_stats``)."""
    del coords, seed, n_paths   # deterministic; nbr18 is the two-hop ring of neighbours6
    pool = build_pool(X, y, groups, neighbours6, names=names, views=views, embed=embed, nuisance=nuisance,
                      device=device)
    return [(tuple(pool.names[j] for j in key), cv, info)
            for key, cv, info in search_pool(pool, rank_by, budget=budget, pair_stats=pair_stats, device=device)]


def network_rank_by(checkpoint, device=None):
    """A frozen v02 network as a ``rank_by`` object (``prepare(pool)`` / ``singles`` / ``pairs``)."""
    from .model import GainSearchModel
    m = GainSearchModel.load(checkpoint, map_location="cpu")
    return (m.to(device) if device is not None else m).eval()
