"""Memory-lean copy of episodes_v03.study_v03_from_raw for the 638,238-spot GSE307534 cohort.

The arithmetic is the one of the frozen function; only dtypes and copies change: float32 gene and base blocks, per-patient
in-place centring (no full-size fancy-index copy), raw arrays released by the caller.  Declared in the test-study notes:
GSE307534 gene and base blocks are float32 (the frozen function builds float64 and peaks above 100 GB).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def study_v03_lean(raw, *, name, platform, organ, spec=None, role="test", resid_dir=None):
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search import episodes_v03 as E
    from morpheus_encoder.learned_search import residual_programmes as rp

    resid_dir = resid_dir or E.RESID_DIR
    keep = np.arange(raw.n)
    groups = np.asarray(raw.patient).astype(str)[keep]
    L = raw.logcpm()[keep]
    pool, _, _ = E._dispersion_pool(L, groups, E.N_POOL)
    G = np.asarray(L[:, pool].todense(), dtype=np.float32)
    del L
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
        model = rp.fit(np.asarray(G[:, hv], dtype=np.float64), np.column_stack([depth, comp, P[:, gav]]).astype(np.float64), groups)
        R, rk = model["scores"], model["k"]
        rpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(rpath, scores=R, loadings=model["loadings"], hvg=np.asarray(genes)[hv], k=rk, alpha=model["alpha"])
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
    inv = np.searchsorted(pats, groups)
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
