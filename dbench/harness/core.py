"""Evaluation harness (governing spec 3.1-3.3).

One harness for every method and task:

1. tasks: a cohort (``episodes_v03.StudyV03``), a target, patient-grouped folds; each fold's
   ``relation_v03.Task`` has support rows first.  The method receives a SUPPORT-ONLY copy.
2. method: ``search(task_support, B, K) -> [(index tuple into task.X columns, predicted gain or
   None, predicted sd or None), ...]`` best first (2-tuples accepted); at most K sets fitted.
3. registry: every (target, fold, B, null kind, draw) list, as candidate NAMES, is written to
   ``registry.json`` and its sha256 recorded BEFORE any query row is scored.
4. scoring: each registered set is fitted on that fold's support rows with the fixed family
   (0.7) and scored on that fold's query rows at B_nuis, B_known and B_decomp(S) (0.8, 0.10),
   kept per query patient (gain numerator, total sum of squares) so that intervals pool across
   folds: a set's pooled gain is sum(numerators) / sum(SST) over the query patients of every fold
   that registered it (no fold's query patients were seen by that fold's search), with a 95 %
   patient-bootstrap interval.
5. null targets, searched by the SAME method: ``patient`` = each patient's target replaced by
   another patient's (derangement within support and within query, values carried in spatial
   order), ``scramble`` = candidate rows permuted across spots within patient (all candidates by
   the same permutation, so candidate structure is kept and its alignment with the target lost).
6. Yield_k = mean over folds of #{S in the fold's top k : pooled interval lower bound > 0};
   net yield = real Yield_k - mean null Yield_k (3.2), at k in {1, 5, 10, 20} per baseline.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

KS = (1, 5, 10, 20)
BASELINES = ("nuis", "known", "decomp")
NULL_KINDS = ("patient", "scramble")
N_BOOT = 2000


# ------------------------------------------------------------------ tasks
def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def patient_folds(patients: Sequence[str], n_folds: int, seed: int) -> list[list[str]]:
    p = np.random.default_rng(seed).permutation(sorted(map(str, patients)))
    return [sorted(p[i::n_folds].tolist()) for i in range(n_folds)]


def _fixed_sampler():
    from morpheus_encoder.learned_search.episodes_v03 import SamplerV03

    class FixedSampler(SamplerV03):
        """SamplerV03 with the study, patients and target fixed (level L0)."""

        def set_task(self, support, query, target, kind="real"):
            self._fix = (list(support), list(query), target)
            self.force_level = "L0"

        def _patients(self, study, rng, n_sup, n_q, query_study):
            return self._fix[0], self._fix[1]

        def _target(self, study, rng, *, cross=False):
            c = study.index[self._fix[2]]
            return study.names[c], study.base[:, c], [c] + list(study.exclusions.get(c, [])), []
    return FixedSampler


def build_task(study, target: str, support: Sequence[str], query: Sequence[str], seed: int, *,
               kind: str = "real", family: str | None = None, fraction: float | None = None):
    """(relation_v03.Task, candidate names, episode) for fixed patients and target."""
    from morpheus_encoder.learned_search.episodes_v03 import task_from_episode

    from morpheus_encoder.learned_search.episodes_v03 import SamplerV03

    rng = np.random.default_rng(seed)
    if hasattr(SamplerV03, "episode_for"):
        sampler = SamplerV03([study], controlled_prob=float(kind == "controlled"), seed=seed)
        kw = {} if kind == "real" else {"family": family, "fraction": fraction}
        ep = sampler.episode_for(study, target, list(support), list(query), rng=rng, kind=kind, **kw)
    else:   # before episodes_v03 exposed episode_for
        sampler = _fixed_sampler()([study], controlled_prob=float(kind == "controlled"), seed=seed)
        sampler.set_task(support, query, target)
        ep = sampler._sample(rng, kind=kind, family=family, fraction=fraction, study=study)
    return task_from_episode(ep), list(ep.meta.names), ep


def support_only(task):
    s = task.s
    cut = lambda a: None if a is None else a[:s]  # noqa: E731
    nb = None if task.nb6 is None else np.where(task.nb6[:s] < s, task.nb6[:s], -1)
    return replace(task, X=task.X[:s], y=task.y[:s], groups=task.groups[:s], depth=cut(task.depth),
                   known=cut(task.known), composition=cut(task.composition), nb6=nb, sections=cut(task.sections))


def null_task(task, ep, kind: str, seed: int):
    """Null versions of a task (module docstring, item 5)."""
    rng = np.random.default_rng(seed)
    s, g = task.s, np.asarray(task.groups)
    if kind == "scramble":
        X = task.X.copy()
        for p in np.unique(g):
            r = np.flatnonzero(g == p)
            X[r] = X[rng.permutation(r)]
        return replace(task, X=X)
    if kind != "patient":
        raise ValueError(kind)
    c = np.asarray(ep.coords, dtype=float)
    axis = c @ np.linalg.svd(c - c.mean(0), full_matrices=False)[2][0]   # first spatial axis
    y = task.y.copy()
    for side in (np.arange(s), np.arange(s, len(g))):
        pats = np.unique(g[side])
        if len(pats) < 2:
            continue
        perm = pats[np.roll(np.arange(len(pats)), 1 + int(rng.integers(len(pats) - 1)))]   # a derangement
        for p, donor in zip(pats, perm):
            rp, rd = side[g[side] == p], side[g[side] == donor]
            src = task.y[rd][np.argsort(axis[rd], kind="stable")]
            pos = np.interp(np.linspace(0, 1, len(rp)), np.linspace(0, 1, len(src)), np.arange(len(src)))
            vals = src[np.rint(pos).astype(int)]
            y[rp[np.argsort(axis[rp], kind="stable")]] = vals - vals.mean()
    return replace(task, y=y)


# ------------------------------------------------------------------ registry
def registry_hash(registry: list) -> str:
    return hashlib.sha256(json.dumps(registry, sort_keys=True, default=str).encode()).hexdigest()


def freeze_registry(registry: list, path: Path) -> str:
    """Write the registry and return its sha256; called before any query scoring."""
    h = registry_hash(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sha256": h, "entries": registry}, default=str))
    return h


# ------------------------------------------------------------------ scoring
def _set_partitions(items: tuple):
    if len(items) == 1:
        yield [list(items)]
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        yield [[first]] + part
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]


def _partitions(items: tuple) -> list:
    """Every partition of ``items`` into at least two blocks (the decomposition family, 0.7)."""
    return [p for p in _set_partitions(tuple(items)) if len(p) >= 2]


def _per_patient(fit_fn, base, task, V_s, V_q, **kw):
    """Per-query-patient (gain numerator (Pq, m), SST (Pq,)); one fit per query patient until
    ``relation_v03`` returns them directly."""
    s = task.s
    gq = np.asarray(task.groups[s:])
    full = fit_fn(V_s, base, task.y[:s], task.groups[:s], Vq=V_q, yq=task.y[s:], groups_q=gq, n_draws=1, **kw)
    if "gain_qp" in full:   # relation_v03 returns the per-patient pieces directly
        return np.asarray(full["gain_qp"]), np.asarray(full["sst_qp"]), full
    pats = np.unique(gq)
    num, sst = [], []
    yq = task.y[s:]
    out0 = None
    for p in pats:
        r = gq == p
        b = replace(base, r_q=base.r_q[r])
        out = fit_fn(V_s, b, task.y[:s], task.groups[:s], Vq=V_q[r], yq=yq[r], groups_q=gq[r], n_draws=1, **kw)
        if out0 is None:
            out0 = out
        sp = float(np.sum((yq[r] - yq[r].mean()) ** 2))
        num.append(out["g_full"] * sp)
        sst.append(sp)
    return np.array(num), np.array(sst), out0


def score_sets(task, sets: list, Bs=BASELINES, device=None) -> dict:
    """{B: (numerators (Pq, n_sets), SST (Pq,), support gain (n_sets,))} for registered sets
    (index tuples, any size 1-4) on this task's query rows; B_decomp only for |S| >= 2 (NaN else)."""
    from morpheus_encoder.learned_search import relation_v03 as rv

    s = task.s
    Pq = len(np.unique(task.groups[s:]))
    out = {}
    bases = {"nuis": rv.baseline_for(task, "nuis")}
    if "known" in Bs:
        bases["known"] = rv.baseline_for(task, "known")
    for B in Bs:
        num = np.full((Pq, len(sets)), np.nan)
        sst = None
        gsup = np.full(len(sets), np.nan)
        for k in sorted({len(x) for x in sets}):
            idx = [i for i, x in enumerate(sets) if len(x) == k]
            arr = np.asarray([sets[i] for i in idx], dtype=np.int64)
            Vs, Vq = rv.set_values(task, arr)
            if B in ("nuis", "known"):
                n, sst, o = _per_patient(rv.fit_sets, bases[B], task, Vs, Vq, device=device)
                num[:, idx], gsup[idx] = n, o["g_sup"]
            elif k >= 2:
                n_full, sst, o_full = _per_patient(rv.fit_sets, bases["nuis"], task, Vs, Vq, device=device)
                best_num, best_sup = None, None
                for part in _partitions(tuple(range(k))):    # best decomposition chosen on SUPPORT gain
                    n_dec, _, o_dec = _per_patient(rv.fit_sets, bases["nuis"], task, Vs, Vq, blocks=part,
                                                   device=device)
                    gs = np.where(np.isfinite(o_dec["g_sup"]), o_dec["g_sup"], -np.inf)   # overflowed fits never win
                    if best_sup is None:
                        best_num, best_sup = n_dec, gs.copy()
                    else:
                        better = gs > best_sup
                        best_num = np.where(better[None], n_dec, best_num)
                        best_sup = np.where(better, gs, best_sup)
                num[:, idx] = n_full - best_num
                gsup[idx] = o_full["g_sup"] - best_sup
        if sst is None:
            gq = np.asarray(task.groups[s:])
            yq = task.y[s:]
            sst = np.array([np.sum((yq[gq == p] - yq[gq == p].mean()) ** 2) for p in np.unique(gq)])
        out[B] = (num, sst, gsup)
    return out


def pooled_interval(num: np.ndarray, sst: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Pooled gain sum(num)/sum(sst) over patients and its 95 % patient-bootstrap interval."""
    num, sst = np.asarray(num, float), np.asarray(sst, float)
    ok = np.isfinite(num)
    num, sst = num[ok], sst[ok]
    if not len(num):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(num), np.full(len(num), 1 / len(num)), size=n_boot)
    boot = (w @ num) / np.maximum(w @ sst, 1e-12)
    return float(num.sum() / max(sst.sum(), 1e-12)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


# ------------------------------------------------------------------ yields
@dataclass
class FoldResult:
    """One (target, fold, condition) search: registered set names in rank order, predicted gains,
    and per-B per-query-patient scores of the registered sets."""
    target: str
    fold: int
    condition: str                 # "real" | "null:patient:<draw>" | "null:scramble:<draw>"
    names: list                    # [[candidate names], ...] rank order
    pred: list                     # predicted gain per set or None
    scores: dict                   # B -> (num (Pq, m), sst (Pq,), gsup (m,))
    patients: list                 # query patient labels of this fold (globally unique)
    pred_sd: list | None = None    # predicted sd per set (methods with a variance head) or None


def yields(results: list, ks=KS, Bs=BASELINES, n_boot: int = N_BOOT) -> dict:
    """Yield_k per (condition group, target, B): pooled-interval discoveries in the top k, mean over folds."""
    groups: dict = {}
    for r in results:
        cond = r.condition.split(":")[0] + (":" + r.condition.split(":")[1] if r.condition != "real" else "")
        groups.setdefault((cond, r.target, r.condition), []).append(r)
    out: dict = {}
    for (cond, target, condition), rs in groups.items():
        for B in Bs:
            rsB = [r for r in rs if B in r.scores]
            if not rsB:
                continue
            pool: dict = {}
            for r in rsB:
                num, sst, _ = r.scores[B]
                for j, nm in enumerate(r.names):
                    key = tuple(sorted(nm))
                    pool.setdefault(key, ([], []))
                    pool[key][0].append(num[:, j])
                    pool[key][1].append(sst)
            lower = {k: pooled_interval(np.concatenate(a), np.concatenate(b), n_boot)[1] for k, (a, b) in pool.items()}
            for k in ks:
                y = np.mean([sum(1 for nm in r.names[:k] if lower[tuple(sorted(nm))] > 0) for r in rsB])
                out.setdefault((cond, target, B, k), []).append(float(y))
    # mean over null draws within a condition group
    return {key: float(np.mean(v)) for key, v in out.items()}


def net_yields(results: list, ks=KS, Bs=BASELINES, null_kinds=NULL_KINDS, n_boot: int = N_BOOT) -> dict:
    """{B: {k: {"real", "null:<kind>", "net", "net:<kind>"}}} averaged over targets."""
    y = yields(results, ks, Bs, n_boot)
    targets = sorted({r.target for r in results})
    out: dict = {}
    for B in Bs:
        for k in ks:
            real = [y.get(("real", t, B, k), np.nan) for t in targets]
            d = {"real": float(np.nanmean(real))}
            nulls = []
            for nk in null_kinds:
                v = [y.get((f"null:{nk}", t, B, k), np.nan) for t in targets]
                d[f"null:{nk}"] = float(np.nanmean(v))
                d[f"net:{nk}"] = float(np.nanmean(np.array(real) - np.array(v)))
                nulls.append(np.array(v))
            d["net"] = float(np.nanmean(np.array(real) - np.nanmean(np.stack(nulls), 0))) if nulls else d["real"]
            out.setdefault(B, {})[k] = d
    return out


def calibration(results: list, B: str = "nuis") -> dict:
    """Predicted vs realised (fold query) gain over every scored set that carries a prediction,
    real targets only: Spearman; ECE = mean |predicted - realised| over 10 equal-count bins of the
    prediction; for methods with a variance head also the coverage ECE = mean |observed - nominal|
    coverage of the Gaussian central intervals at 10 %, 20 %, ..., 90 %."""
    from scipy.stats import norm, spearmanr

    pred, sd, real = [], [], []
    for r in results:
        if r.condition != "real" or B not in r.scores:
            continue
        num, sst, _ = r.scores[B]
        g = np.nansum(num, 0) / max(float(np.sum(sst)), 1e-12)
        sds = r.pred_sd or [None] * len(r.pred)
        for j, p in enumerate(r.pred):
            if p is not None and np.isfinite(p) and np.isfinite(g[j]):
                pred.append(float(p))
                real.append(float(g[j]))
                sd.append(np.nan if sds[j] is None else float(sds[j]))
    if len(pred) < 3:
        return {"spearman": None, "ece": None, "coverage_ece": None, "n": len(pred)}
    pred, real, sd = np.array(pred), np.array(real), np.array(sd)
    bins = np.array_split(np.argsort(pred), min(10, len(pred)))
    ece = float(sum(len(b) * abs(pred[b].mean() - real[b].mean()) for b in bins) / len(pred))
    rho = spearmanr(pred, real).correlation
    cov = None
    ok = np.isfinite(sd) & (sd > 0)
    if ok.sum() >= 3:
        z = np.abs(real[ok] - pred[ok]) / sd[ok]
        levels = np.arange(0.1, 1.0, 0.1)
        cov = float(np.mean([abs(np.mean(z <= norm.ppf(0.5 + lv / 2)) - lv) for lv in levels]))
    return {"spearman": float(rho) if np.isfinite(rho) else 0.0, "ece": ece, "coverage_ece": cov, "n": int(len(pred))}


# ------------------------------------------------------------------ run
_TASK_CACHE: dict = {}
_G: dict = {}   # state inherited by forked workers (jobs, the search function, scoring jobs)


def _tasks(study, targets, folds, null_kinds, n_null, seed):
    """[(target, fold, [(condition, task)], names, study name, query)], cached across calls: the tasks
    and their null versions do not depend on the method."""
    out = []
    for ti, target in enumerate(targets):
        for f, query in enumerate(folds):
            key = (study.name, target, tuple(query), seed, tuple(null_kinds), n_null)
            if key not in _TASK_CACHE:
                support = [p for p in map(str, study.patients) if p not in set(query)]
                task, names, ep = build_task(study, target, support, query, seed=seed + 1000 * ti + f)
                conds = [("real", task)] + [(f"null:{nk}:{d}", null_task(task, ep, nk, seed + 7919 * (d + 1) + 31 * ti + f))
                                            for nk in null_kinds for d in range(n_null)]
                _TASK_CACHE[key] = (conds, names, ep.study, list(query))
            out.append((target, f, *_TASK_CACHE[key]))
    return out


GPU_SEARCH_WORKERS = int(__import__("os").environ.get("MORPHEUS_HARNESS_GPU_WORKERS", 2))   # cap beside training (coordinator: <= 4; 2 after the 20 % slowdown rule)
GPU_MEM_GB = 1.5           # per search worker (plus ~0.5 GB CUDA context): 4 workers stay within 8 GB


def _search_job(i):
    import torch
    torch.set_num_threads(1)
    if getattr(_G["search"], "gpu", False) and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, GPU_MEM_GB * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory), 0)
    target, f, cond, tk, B, K = _G["jobs"][i]
    t = time.time()
    out = list(_G["search"](support_only(tk), B, K))[:K]
    return out, time.time() - t


def _score_job(i):
    import torch
    torch.set_num_threads(1)
    tk, top, Bs = _G["score"][i]
    return score_sets(tk, top, Bs=Bs, device=None)


def _pool_map(fn, n, workers):
    import multiprocessing as mp
    if workers <= 1 or n <= 1:
        return [fn(i) for i in range(n)]
    with mp.get_context("fork").Pool(min(workers, n)) as pool:
        return pool.map(fn, range(n), chunksize=1)


def run(study, targets: Sequence[str], search: Callable, *, n_folds: int = 3, K: int = 64, Bs=BASELINES,
        search_Bs=("nuis",), null_kinds=NULL_KINDS, n_null: int = 1, seed: int = 0, score_top: int = max(KS),
        out_dir: Path | None = None, device=None, workers: int = 1, log=print) -> dict:
    """Search every (target, fold, condition) on support only, freeze the registry, then score.

    ``search_Bs``: the baselines the method is asked to search under (the list registered under
    B is scored at B; B_decomp is scored on the B_nuis list).  Only the top ``score_top`` sets are
    scored (Yield_k needs k <= 20); predictions of those sets feed the calibration.
    ``workers``: forked CPU processes for the scoring fits, and for the searches too when
    ``search.parallel_ok`` is true (a CPU search; a GPU model searches in this process)."""
    t0 = time.time()
    folds = patient_folds(study.patients, n_folds, seed)
    built = _tasks(study, targets, folds, null_kinds, n_null, seed)
    t_tasks = time.time() - t0
    jobs, meta = [], []
    for target, f, conds, names, st, query in built:
        for cond, tk in conds:
            for B in search_Bs:
                jobs.append((target, f, cond, tk, B, K))
                meta.append((names, st, query))
    _G.update(jobs=jobs, search=search)
    if getattr(search, "gpu", False):    # forked GPU workers; this process never initialises CUDA
        out = _pool_map(_search_job, len(jobs), min(workers, GPU_SEARCH_WORKERS))
    elif getattr(search, "parallel_ok", False):
        out = _pool_map(_search_job, len(jobs), workers)
    else:
        out = [_search_job(i) for i in range(len(jobs))]
    rankings, per_search = [o[0] for o in out], [o[1] for o in out]
    t_search = time.time() - t0 - t_tasks
    num = lambda v: None if v is None else float(v)  # noqa: E731
    registry = []
    for (target, f, cond, tk, B, _), (names, st, query), ranked in zip(jobs, meta, rankings):
        registry.append({"target": target, "fold": f, "condition": cond, "B": B,
                         "sets": [[names[i] for i in r[0]] for r in ranked],
                         "pred": [num(r[1]) if len(r) > 1 else None for r in ranked],
                         "pred_sd": [num(r[2]) if len(r) > 2 else None for r in ranked]})
    h = freeze_registry(registry, out_dir / "registry.json") if out_dir is not None else registry_hash(registry)
    score_jobs, keep = [], []
    for j, ((target, f, cond, tk, B, _), ranked) in enumerate(zip(jobs, rankings)):
        top = [tuple(int(i) for i in r[0]) for r in ranked[:score_top]]
        if top:
            score_jobs.append((tk, top, [b for b in Bs if b == B or (b == "decomp" and B == "nuis")]))
            keep.append(j)
    _G["score"] = score_jobs
    scores = _pool_map(_score_job, len(score_jobs), workers)
    _G.clear()
    results = []
    for j, sc in zip(keep, scores):
        reg, (names, st, query) = registry[j], meta[j]
        results.append(FoldResult(reg["target"], reg["fold"], reg["condition"], reg["sets"][:score_top],
                                  reg["pred"][:score_top], sc, [f"{st}:{q}" for q in query], reg["pred_sd"][:score_top]))
    t_score = time.time() - t0 - t_tasks - t_search
    log(f"tasks {t_tasks:.0f} s, search {t_search:.0f} s, scoring {t_score:.0f} s")
    scored_Bs = tuple(b for b in Bs if any(b in r.scores for r in results))
    return {"registry_sha256": h, "results": results,
            "net_yield": net_yields(results, Bs=scored_Bs, null_kinds=null_kinds),
            "calibration": calibration(results) if "nuis" in scored_Bs else None, "seconds": time.time() - t0,
            "timing": {"tasks": t_tasks, "search": t_search, "scoring": t_score,
                       "seconds_per_search": float(np.mean(per_search)) if per_search else None,
                       "search_device": "cuda" if getattr(search, "gpu", False) else "cpu"}}
