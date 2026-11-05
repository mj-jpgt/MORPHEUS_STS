"""Streaming variant of study_lean.study_v03_lean for the 638,238-spot GSE307534 cohort (peak memory).

Arithmetic identical to ``episodes_v03._dispersion_pool`` and ``study_v03_lean``: the per-gene mean and mean square of the
log-CPM matrix are accumulated over row blocks instead of forming ``L.multiply(L)`` and ``L.tocsc()`` copies; the
log-CPM matrix is used without a full-size row-index copy; raw counts are released once the log-CPM matrix exists.
The pool is the same top-2000 dispersion set (float summation order differs at the 1e-16 level).
Also provides a memory watchdog: the process exits when its RSS exceeds a ceiling, so that it never takes the box down.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) / 1048576
    return 0.0


def start_watchdog(limit_gb: float, log=print):
    def run():
        while True:
            r = rss_gb()
            if r > limit_gb:
                log(f"WATCHDOG: RSS {r:.0f} GB above {limit_gb} GB, exiting", flush=True)
                os._exit(97)
            time.sleep(1.0)
    threading.Thread(target=run, daemon=True).start()


def dispersion_pool_stream(L, groups, n, chunk=40000):
    import scipy.sparse as sp
    ncol = L.shape[1]
    s1, s2 = np.zeros(ncol), np.zeros(ncol)
    for lo in range(0, L.shape[0], chunk):
        blk = L[lo: lo + chunk]
        s1 += np.asarray(blk.sum(0)).ravel()
        s2 += np.asarray(blk.multiply(blk).sum(0)).ravel()
    mean, sq = s1 / L.shape[0], s2 / L.shape[0]
    var = sq - mean ** 2
    disp = np.where(mean > 1e-3, var / np.maximum(mean, 1e-12), 0.0)
    pool = np.sort(np.argsort(-disp, kind="stable")[:n])
    return pool, disp[pool], var[pool]


def study_v03_lean2(raw, *, name, platform, organ, spec=None, role="test", resid_dir=None, log=print):
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search import episodes_v03 as E
    from morpheus_encoder.learned_search import residual_programmes as rp

    resid_dir = resid_dir or E.RESID_DIR
    keep = np.arange(raw.n)
    groups = np.asarray(raw.patient).astype(str)[keep]
    log(f"  lean2: start RSS {rss_gb():.0f} GB", flush=True)
    L = raw.logcpm()
    raw.counts = None
    log(f"  lean2: logcpm built, counts released, RSS {rss_gb():.0f} GB", flush=True)
    pool, _, _ = dispersion_pool_stream(L, groups, E.N_POOL)
    log(f"  lean2: pool, RSS {rss_gb():.0f} GB", flush=True)
    G = np.asarray(L[:, pool].todense(), dtype=np.float32)
    raw.meta.pop("_logcpm_full", None)
    del L
    log(f"  lean2: dense gene block {G.shape}, RSS {rss_gb():.0f} GB", flush=True)
    genes = [str(g) for g in np.asarray(raw.genes)[pool]]
    P = np.asarray(raw.programmes, dtype=np.float32)[keep]
    pnames = [str(c) for c in raw.programme_names]
    depth = np.log1p(np.asarray(raw.depth, dtype=np.float64))[keep]
    comp, comp_source = E.composition(raw, spec, keep, P.astype(np.float64), name=name)
    comp = np.asarray(comp, dtype=np.float32)
    gav = [j for j, c in enumerate(pnames) if not c.startswith("hallmark")]
    rpath = Path(resid_dir) / f"{name}_{raw.meta.get('cache_key', 'nokey')}.npz"
    if rpath.exists():
        with np.load(rpath) as z:
            R, rk = z["scores"], int(z["k"])
    else:
        hv = np.argsort(-G.var(0), kind="stable")[:E.N_RESID_HVG]
        log(f"  lean2: residual programmes fit, RSS {rss_gb():.0f} GB", flush=True)
        model = rp.fit(np.asarray(G[:, hv], dtype=np.float64), np.column_stack([depth, comp, P[:, gav]]).astype(np.float64), groups)
        R, rk = model["scores"], model["k"]
        rpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(rpath, scores=R, loadings=model["loadings"], hvg=np.asarray(genes)[hv], k=rk, alpha=model["alpha"])
    log(f"  lean2: residual programmes ready, RSS {rss_gb():.0f} GB", flush=True)
    R = np.asarray(R, dtype=np.float32)
    names = [f"resid:G{j + 1}" for j in range(rk)] + [f"programme:{c}" for c in pnames]
    views = {"resid": list(range(rk)), "programme": list(range(rk, rk + len(pnames)))}
    views["gene"] = list(range(len(names), len(names) + len(genes)))
    names += [f"gene:{g}" for g in genes]
    base = np.concatenate([R, P, G], 1)
    del G
    pats = np.unique(groups)
    pm = np.zeros((len(pats), base.shape[1]), dtype=np.float32)
    for i, p in enumerate(pats):
        r = np.flatnonzero(groups == p)
        pm[i] = base[r].mean(0)
        base[r] -= pm[i]
    known = np.column_stack([P, comp]).astype(np.float32)
    compc = comp.copy()
    for i, p in enumerate(pats):
        r = np.flatnonzero(groups == p)
        known[r] -= known[r].mean(0)
        compc[r] -= compc[r].mean(0)
    depth_c = depth.copy()
    for p in pats:
        r = groups == p
        depth_c[r] -= depth[r].mean()
    gcols = np.asarray(views["gene"])
    between = pm[:, gcols].var(0)
    ok = gcols[between >= np.quantile(between, 0.25)]
    gt = ok[np.argsort(-base[:, ok].var(0), kind="stable")[:300]]
    index = {c: i for i, c in enumerate(names)}
    excl = {}
    for j, c in zip(views["programme"], pnames):
        excl[j] = [index[f"gene:{g}"] for g in hm._programme_members(c) if f"gene:{g}" in index]
    nb6 = np.asarray(raw.neighbours[6], dtype=np.int64)[:, :6]
    log(f"  lean2: done, RSS {rss_gb():.0f} GB", flush=True)
    return E.StudyV03(name=name, base=base, names=names, views=views, groups=groups, neighbours6=nb6,
                      coords=np.asarray(raw.coords, dtype=np.float32)[keep], embed=None, platform=platform,
                      target_cols=list(views["resid"]) + list(views["programme"]) + [int(c) for c in gt],
                      exclusions=excl, residual_cols=list(views["programme"]),
                      subsample_views={"gene": E.N_EPISODE_GENES}, depth=depth_c.astype(np.float32),
                      known=known, composition=compc.astype(np.float32),
                      known_names=pnames + [f"composition:{j}" for j in range(comp.shape[1])],
                      panel=[str(g) for g in np.asarray(raw.genes)],
                      sections=np.asarray(raw.sample_id).astype(str)[keep], patient_means=pm.astype(np.float32),
                      organ=organ, morph_targets=[], role=role, comp_source=comp_source, embed_raw=None, morph_centres=None)
