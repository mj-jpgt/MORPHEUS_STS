"""Step 2 of METHOD_BENCHMARK_SPEC: cached labels for the test studies (RFF family; GAM family in gam_labels.py).

    python -m dbench.benchmark_v04.labels STUDY [--max-targets N] [--out DIR] [--pilot]

Per (study, target): 10 patient-grouped splits (3 query patients, the rest support); one fixed pool of singles (the
candidates present in every split) and 2,000 uniformly sampled pairs; for every set the pooled query gain G over
B_known / B_nuis (singles, pairs) and over the best proper-subset decomposition B_decomp (pairs), with patient-bootstrap
lower bounds at levels 95/90/99, and the p95 of the patient-permuted-target null G (5 draws, same sets).
Binary label = lower bound > 0 AND G > null p95 (derived in tables.py).  Everything but the label is stored raw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HOME = Path.home()
SW, BENCH = str(HOME / "morpheus_work_sw"), str(HOME / "morpheus_work_bench")
# morpheus / morpheus_encoder come from the second_wave tree (frozen method files), dbench from the bench tree
sys.path.insert(0, SW)
import morpheus  # noqa: E402,F401
import morpheus_encoder  # noqa: E402,F401
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402

N_PAIRS, N_SPLITS, N_QUERY, N_NULL, N_BOOT = 2000, 10, 3, 5, 2000
SEED = 20261027
LEVELS = {"95": 0.025, "90": 0.05, "99": 0.005}
GLIAL = ["GFAP", "AQP4", "S100B", "NDRG2", "GJA1", "CST3", "NTRK2"]
BRAIN_STUDIES = {"gse237183_glioma"}
LEAN_STUDIES = {"gse307534_lung_precursor_b3"}


def diag():
    from dbench.benchmark_v04 import specs
    from morpheus.tier2.substrates import hest_multi as hm
    print(json.dumps({"sys_path0": sys.path[0], "sys_path1": sys.path[1], "specs": specs.__file__,
                      "hest_multi": hm.__file__, "cwd": os.getcwd(),
                      "keys": sorted(k for k in specs.all_specs() if "gse" in k or "luad" in k)}), flush=True)


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ study loading
def load_test_study(key: str):
    from dbench.benchmark_v04.specs import all_specs
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.episodes_v03 import study_v03_from_raw
    spec = all_specs()[key]
    raw = hm.load_study(spec)
    if key in LEAN_STUDIES:   # 638,238 spots: float32 builder with identical arithmetic (study_lean.py)
        from dbench.benchmark_v04.study_lean import study_v03_lean
        st = study_v03_lean(raw, name=key, platform=spec.platform, organ=str(spec.organ).lower(), spec=spec,
                            role="test")
        hm._RAW_MEMO.clear()
        del raw
        return st
    st = study_v03_from_raw(raw, name=key, platform=spec.platform, organ=str(spec.organ).lower(), spec=spec,
                            role="test")
    if key in BRAIN_STUDIES:
        add_glial(st, raw)
    hm._RAW_MEMO.clear()
    return st


def add_glial(st, raw):
    """bknown_glial_extension_v01: own-spot log1p-CPM of the proxy genes, appended to the known block as one column
    per gene (centred within patient); the proxy mean is formed per set (leave-out rule) in ``known_variant``."""
    genes = [str(g) for g in np.asarray(raw.genes)]
    pos = {g: i for i, g in enumerate(genes)}
    present = [g for g in GLIAL if g in pos]
    keep = np.arange(raw.n)
    if raw.hoptimus is not None:
        keep = np.flatnonzero(np.asarray(raw.covered, dtype=bool))
    L = raw.logcpm()[keep]
    cols = np.asarray(L[:, [pos[g] for g in present]].todense(), dtype=np.float64)
    pats = np.unique(st.groups)
    inv = np.searchsorted(pats, st.groups)
    cols = cols - np.stack([cols[st.groups == p].mean(0) for p in pats])[inv]
    st.known = np.column_stack([st.known, cols.astype(np.float32)])
    st.known_names = list(st.known_names) + [f"glialgene:{g}" for g in present]
    st.glial_present = present


def known_variant(task, drop_genes=()):
    """Task copy whose B_known has the glial proxy computed without ``drop_genes`` (no-op without glial columns)."""
    names = list(task.known_names or [])
    gi = [i for i, n in enumerate(names) if str(n).startswith("glialgene:")]
    if not gi:
        return task
    from dataclasses import replace
    gdrop = {str(g) for g in drop_genes}
    use = [i for i in gi if names[i].split(":", 1)[1] not in gdrop]
    base = [i for i in range(len(names)) if i not in set(gi)]
    K = np.asarray(task.known)
    cols = [K[:, base]]
    nm = [names[i] for i in base]
    if use:
        cols.append(K[:, use].mean(1, keepdims=True))
        nm.append("composition:glial_proxy")   # prefix keeps it out of the programme-overlap exclusion
    return replace(task, known=np.concatenate(cols, 1), known_names=nm)


def proxy_genes_of(names_of_set, target_name):
    """Proxy genes appearing in a set's variables (any spatial version) or in the target."""
    out = set()
    for n in list(names_of_set) + [target_name]:
        g = str(n).split("gene:", 1)[1] if "gene:" in str(n) else None
        if g in GLIAL:
            out.add(g)
    return tuple(sorted(out))


# ------------------------------------------------------------------ targets and splits
def eligible_state_targets(st):
    """Frozen state scores only (spec 2.1): RNA-only residual programmes, Gavish/Hallmark programme scores and, where
    H&E exists, morphology centre similarities.  Gene targets are excluded."""
    names = [str(st.names[c]) for c in st.target_cols]
    names += [str(st.names[c]) for c in getattr(st, "morph_targets", [])]
    return sorted({n for n in names if n.startswith(("resid:", "programme:", "morph:"))})


def choose_targets(st, n=40, seed=SEED):
    el = eligible_state_targets(st)
    if len(el) <= n:
        return el
    rng = np.random.default_rng(seed)
    return sorted(str(x) for x in rng.choice(el, n, replace=False))


def splits_for(st, target, n=N_SPLITS, nq=N_QUERY):
    pats = sorted(map(str, st.patients))
    out = []
    for r in range(n):
        rng = np.random.default_rng(stable_seed("bench-split", st.name, target, r))
        q = sorted(rng.choice(pats, nq, replace=False).tolist())
        out.append((r, [p for p in pats if p not in q], q))
    return out


# ------------------------------------------------------------------ scoring
def pooled(num, sst, levels=LEVELS, n_boot=N_BOOT, seed=0):
    """(G (m,), {level: lower bound (m,)}) pooled over the query patients with a patient bootstrap."""
    num = np.where(np.isfinite(num), num, 0.0)
    G = num.sum(0) / max(float(sst.sum()), 1e-12)
    if not levels:
        return G, {}
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(sst), np.full(len(sst), 1 / len(sst)), size=n_boot)
    boot = (w @ num) / np.maximum(w @ sst, 1e-12)[:, None]
    return G, {k: np.quantile(boot, q, axis=0) for k, q in levels.items()}


def score_groups(task, sets, Bs, target_name, names, device):
    """core.score_sets, grouped by the glial leave-out key when the study has glial columns."""
    from dbench.harness import core
    has_glial = any(str(n).startswith("glialgene:") for n in (task.known_names or []))
    if not has_glial:
        return core.score_sets(task, sets, Bs=Bs, device=device)
    keys = [proxy_genes_of([names[i] for i in s], target_name) for s in sets]
    out = {}
    for B in Bs:
        num = sst = gs = None
        for key in sorted(set(keys)):
            idx = [i for i, k in enumerate(keys) if k == key]
            r = core.score_sets(known_variant(task, key), [sets[i] for i in idx], Bs=(B,), device=device)[B]
            if num is None:
                num = np.full((r[0].shape[0], len(sets)), np.nan)
                gs = np.full(len(sets), np.nan)
                sst = r[1]
            num[:, idx], gs[idx] = r[0], r[2]
        out[B] = (num, sst, gs)
    return out


def label_target(st, target, device, log=print, max_splits=None):
    """All labels of one (study, target): dict of arrays + pool, or None when the pool is degenerate."""
    from dbench.harness import core
    seed = stable_seed("bench-task", st.name, target)
    built = []
    for r, support, query in splits_for(st, target):
        try:
            task, names, ep = core.build_task(st, target, support, query, seed=seed)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            log(f"  {target} split {r}: {exc}")
            continue
        built.append((r, support, query, task, list(names), ep))
    if len(built) < 5:
        return None
    singles = sorted(set.intersection(*[set(b[4]) for b in built]))
    rng = np.random.default_rng(stable_seed("bench-pairs", st.name, target))
    n_pairs = min(N_PAIRS, len(singles) * (len(singles) - 1) // 2)
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(len(singles), 2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    pairs = sorted(pairs)
    out = {"pool_singles": singles, "pool_pairs": [[singles[a], singles[b]] for a, b in pairs], "splits": []}
    for r, support, query, task, names, ep in built[:max_splits]:
        idx = {n: i for i, n in enumerate(names)}
        s_sets = [(idx[n],) for n in singles]
        p_sets = [(idx[singles[a]], idx[singles[b]]) for a, b in pairs]
        rec = {"split": r, "support": support, "query": query, "seconds": {}}
        for kind, sets, Bs in (("single", s_sets, ("known", "nuis")), ("pair", p_sets, ("known", "nuis", "decomp"))):
            t0 = time.time()
            sc = score_groups(task, sets, Bs, task.target_name, names, device)
            for B in Bs:
                num, sst, gsup = sc[B]
                G, lbs = pooled(num, sst, seed=stable_seed(st.name, target, r, kind, B))
                rec[f"{kind}|{B}|G"] = G.astype(np.float32)
                rec[f"{kind}|{B}|gsup"] = np.asarray(gsup, dtype=np.float32)
                for lv, lb in lbs.items():
                    rec[f"{kind}|{B}|lb{lv}"] = lb.astype(np.float32)
                null = []
                for d in range(N_NULL):
                    nt = core.null_task(task, ep, "patient", stable_seed("bench-null", st.name, target, r, d))
                    nsc = score_groups(nt, sets, (B,), task.target_name, names, device)[B]
                    null.append(pooled(nsc[0], nsc[1], levels={})[0])
                null = np.concatenate(null)
                rec[f"{kind}|{B}|null_p95"] = float(np.quantile(null, 0.95))
            rec["seconds"][kind] = round(time.time() - t0, 1)
        out["splits"].append(rec)
        log(f"  {target} split {r}: singles {rec['seconds']['single']} s, pairs {rec['seconds']['pair']} s")
    return out


def save_target(od: Path, target: str, res: dict):
    od.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(target.encode()).hexdigest()[:10]
    meta = {k: res[k] for k in ("pool_singles", "pool_pairs")}
    arrays = {}
    for rec in res["splits"]:
        for k, v in rec.items():
            if isinstance(v, np.ndarray):
                arrays[f"s{rec['split']}|{k}"] = v
            elif isinstance(v, float):
                arrays[f"s{rec['split']}|{k}"] = np.float64(v)
    np.savez_compressed(od / f"{h}.npz", **arrays)
    meta["target"] = target
    meta["splits"] = [{k: rec[k] for k in ("split", "support", "query", "seconds")} for rec in res["splits"]]
    (od / f"{h}.json").write_text(json.dumps(meta))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study")
    ap.add_argument("--max-targets", type=int, default=40)
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--out", default=str(Path(BENCH) / "runs/development/benchmark_v04/labels"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pilot", action="store_true", help="first target, first 2 splits")
    ap.add_argument("--workers", type=int, default=1, help="forked workers, one target each (targets are independent)")
    a = ap.parse_args(argv)
    diag()
    t0 = time.time()
    st = load_test_study(a.study)
    print(f"loaded {a.study}: {st.n} spots, {len(st.patients)} patients, {len(st.names)} columns, "
          f"comp={st.comp_source} in {time.time() - t0:.0f} s", flush=True)
    od = Path(a.out) / a.study
    tp = od / "targets.json"
    if tp.exists():
        targets = json.loads(tp.read_text())["targets"]
    else:
        targets = choose_targets(st, a.n_targets)
        od.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps({"study": a.study, "seed": SEED, "targets": targets,
                                  "eligible": eligible_state_targets(st), "rule": "state scores only (resid, programme, morph); gene targets excluded",
                                  "sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest()}))
    sub = "pilot" if a.pilot else "cache"
    todo = []
    for t in targets[: a.max_targets]:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if not (od / sub / f"{h}.npz").exists():
            todo.append(t)
    if a.pilot:
        todo = todo[:1]
    global _WORK
    _WORK = (st, od / sub, a.device, 2 if a.pilot else None)
    if a.workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        # fork before any CUDA use: the parent holds the study (numpy) only; each worker initialises CUDA itself
        with mp.get_context("fork").Pool(min(a.workers, len(todo))) as pool:
            for line in pool.imap_unordered(_work, todo, chunksize=1):
                print(line, flush=True)
    else:
        for t in todo:
            print(_work(t), flush=True)


_WORK = None


def _work(t):
    st, od, device, max_splits = _WORK
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass
    t1 = time.time()
    res = label_target(st, t, device, max_splits=max_splits)
    if res is None:
        return f"{t}: degenerate, skipped"
    save_target(od, t, res)
    return (f"{t}: {len(res['pool_singles'])} singles, {len(res['pool_pairs'])} pairs, {len(res['splits'])} splits, "
            f"{time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
