"""Label engine v05: the v04 labels (labels.py, protocol hashed in freeze addenda 2/4/7) with identical arithmetic but the
Gram matrices and Cholesky factors of each candidate-set design computed once and reused for every right-hand side.

For one (study, target, split) the v04 code refits the same RFF design 24 times for pairs (3 baselines x (real + 5
patient-permuted nulls), plus the decomposition design) and 12 times for singles.  The design (and therefore Gram, LOPO
systems, penalty grid) does not depend on y; only the right-hand sides (baseline residuals) do.  ``fit_multi`` solves all
right-hand sides against one factorisation.  ``--verify`` recomputes one cached target and compares with the v04 cache.

    python -m dbench.benchmark_v05.labels_fast STUDY [--out DIR]
    python -m dbench.benchmark_v05.labels_fast STUDY --verify TARGET_INDEX [--splits 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
import numpy as np
import torch

BENCH = Path(L.BENCH)
OUT = BENCH / "runs/benchmark_v05"


def load_study_v05(key):
    if key not in L.LEAN_STUDIES:
        return L.load_test_study(key)
    from dbench.benchmark_v04.specs import all_specs
    from morpheus.tier2.substrates import hest_multi as hm
    from dbench.benchmark_v05 import study_lean2 as S2
    S2.start_watchdog(float(os.environ.get("W3_RSS_LIMIT_GB", 125)))
    import pickle
    cache = Path.home() / "w3_cache" / f"{key}_study_lean2.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            st = pickle.load(f)
        print(f"  study loaded from cache {cache}, RSS {S2.rss_gb():.0f} GB", flush=True)
        return st
    spec = all_specs()[key]
    raw = hm.load_study(spec)
    print(f"  raw loaded, RSS {S2.rss_gb():.0f} GB", flush=True)
    st = S2.study_v03_lean2(raw, name=key, platform=spec.platform, organ=str(spec.organ).lower(), spec=spec, role="test")
    hm._RAW_MEMO.clear()
    del raw
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(str(cache) + ".tmp", "wb") as f:
        pickle.dump(st, f, protocol=5)
    os.replace(str(cache) + ".tmp", cache)
    return st


def _spd_solve_multi(A, rhs):
    """cholesky_solve for (..., F, F) systems with (..., F, J) right-hand sides; fallback as relation_v03._spd_solve."""
    Lc, info = torch.linalg.cholesky_ex(A)
    x = torch.cholesky_solve(rhs, Lc)
    bad = (info > 0) | ~torch.isfinite(x).flatten(-2).all(-1)
    if bool(bad.any()):
        Ab = A[bad].double()
        jit = 1e-5 * Ab.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-6)[..., None, None]
        Ab = Ab + jit * torch.eye(A.shape[-1], dtype=Ab.dtype, device=A.device)
        x[bad] = torch.linalg.solve(Ab, rhs[bad].double()).to(x.dtype)
    return torch.nan_to_num(x)


def fit_multi(Vs, Vq, groups_s, groups_q, jobs, *, blocks=None, device="cuda", chunk=32, alphas=None):
    """jobs: list of (BaselineFit, ys, yq).  Returns per job {"g_sup": (m,), "gain_qp": (Pq, m), "sst_qp": (Pq,)}.
    Same arithmetic as ``relation_v03._fit`` (only gain_qp and g_sup are used by the labels)."""
    from morpheus_encoder.learned_search import relation_v03 as rv
    alphas = rv.ALPHAS if alphas is None else alphas
    dev = rv._dev(device)
    dt = rv._dt(dev)
    codes = rv._codes(groups_s)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    P = len(bounds) - 1
    J = len(jobs)
    ys_l = [np.asarray(j[1], dtype=np.float64) for j in jobs]
    sst_p = np.stack([[np.sum((ys[codes == p] - ys[codes == p].mean()) ** 2) for p in range(P)] for ys in ys_l], 1)  # (P,J)
    Vs = torch.as_tensor(np.asarray(Vs), device=dev).to(dt)
    Vs = Vs[:, :, None] if Vs.ndim == 2 else Vs
    Vs = Vs[torch.as_tensor(order, device=dev)]
    n_s, m, k = Vs.shape
    r = torch.as_tensor(np.stack([j[0].r_s[order] for j in jobs], 1), device=dev, dtype=dt)       # (n,J)
    rcv = torch.as_tensor(np.stack([j[0].r_cv[order] for j in jobs], 1), device=dev, dtype=dt)
    rr = torch.stack([(rcv[bounds[p]:bounds[p + 1]] ** 2).sum(0) for p in range(P)])             # (P,J)
    mu, sd = Vs.mean(0), Vs.std(0, unbiased=False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    nb = 1 if blocks is None else len(blocks)
    F = rv.D_RFF * nb + k + 1
    pen = torch.ones(F, dtype=dt, device=dev)
    pen[-1] = 1e-6
    eye = torch.diag(pen)
    al = torch.as_tensor(list(alphas), dtype=dt, device=dev)
    sst = torch.as_tensor(np.maximum(sst_p, 1e-12), dtype=dt, device=dev)                         # (P,J)
    Vq = torch.as_tensor(np.asarray(Vq), device=dev).to(dt)
    Vq = Vq[:, :, None] if Vq.ndim == 2 else Vq
    qcodes = rv._codes(groups_q)
    Sq = rv._onehot(qcodes, dev, dt)
    yqs = [np.asarray(j[2], dtype=np.float64) for j in jobs]
    sstq = torch.as_tensor(np.stack([[np.sum((yq[qcodes == p] - yq[qcodes == p].mean()) ** 2)
                                      for p in range(qcodes.max() + 1)] for yq in yqs], 1), dtype=dt, device=dev)  # (Pq,J)
    rq = torch.as_tensor(np.stack([j[0].r_q for j in jobs], 1), dtype=dt, device=dev)            # (nq,J)
    rrq = Sq @ rq ** 2                                                                           # (Pq,J)
    g_sup, gain = [], []
    with rv._NoTF32():
        for a0 in range(0, m, chunk):
            sl = slice(a0, min(m, a0 + chunk))
            Z = (Vs[:, sl] - mu[sl]) / sd[sl]
            bw = rv.median_bandwidth(Z) if blocks is None else torch.stack([rv.median_bandwidth(Z[:, :, list(b)]) for b in blocks])
            Phi = rv.features(Z, bw, blocks)
            G = torch.stack([torch.einsum("nmf,nmg->mfg", Phi[bounds[p]:bounds[p + 1]], Phi[bounds[p]:bounds[p + 1]])
                             for p in range(P)])                                                  # (P,mc,F,F)
            b = torch.stack([torch.einsum("nmf,nj->mfj", Phi[bounds[p]:bounds[p + 1]], r[bounds[p]:bounds[p + 1]]) for p in range(P)])
            c = torch.stack([torch.einsum("nmf,nj->mfj", Phi[bounds[p]:bounds[p + 1]], rcv[bounds[p]:bounds[p + 1]]) for p in range(P)])
            del Phi
            Gt, bt = G.sum(0), b.sum(0)                                                           # (mc,F,F), (mc,F,J)
            A = (Gt[None, None] - G[None]) + al[:, None, None, None, None] * eye                  # (A,P,mc,F,F)
            rhs = (bt[None] - b)[None].expand(len(al), -1, -1, -1, -1).contiguous()               # (A,P,mc,F,J)
            beta = _spd_solve_multi(A, rhs)
            del A, rhs
            Gb = torch.einsum("pmfg,apmgj->apmfj", G, beta)
            sse = rr[None, :, None, :] - 2 * (beta * c[None]).sum(3) + (beta * Gb).sum(3)         # (A,P,mc,J)
            del Gb, beta, G
            lopo = rr[None, :, None, :] - sse
            best = lopo.sum(1).argmax(0)                                                          # (mc,J)
            lb = lopo.gather(0, best[None, None].expand(1, P, -1, -1))[0]                         # (P,mc,J)
            g_sup.append(lb.sum(0) / sst.sum(0)[None, :])                                          # (mc,J)
            Af = Gt[None] + al[:, None, None, None] * eye                                          # (A,mc,F,F)
            bf_all = _spd_solve_multi(Af, bt[None].expand(len(al), -1, -1, -1).contiguous())      # (A,mc,F,J)
            bfull = bf_all.gather(0, best[None, :, None, :].expand(1, -1, F, -1))[0]               # (mc,F,J)
            del Af, bf_all
            Zq = (Vq[:, sl] - mu[sl]) / sd[sl]
            pred = torch.einsum("nmf,mfj->nmj", rv.features(Zq, bw, blocks), bfull)                # (nq,mc,J)
            sse_p = torch.einsum("qn,nmj->qmj", Sq, (rq[:, None, :] - pred) ** 2)                  # (Pq,mc,J)
            gain.append(rrq[:, None, :] - sse_p)
    g_sup = torch.cat(g_sup, 0).double().cpu().numpy()           # (m,J)
    gain = torch.cat(gain, 1).double().cpu().numpy()             # (Pq,m,J)
    sstq = sstq.double().cpu().numpy()
    return [{"g_sup": g_sup[:, j], "gain_qp": gain[:, :, j], "sst_qp": sstq[:, j]} for j in range(J)]


def _fit_kind(nulls, kind, sets, device, chunk):
    """({(B, i): fit}, {i: decomposition fit}); nulls[0] is the real task, the rest its patient-permuted nulls."""
    from morpheus_encoder.learned_search import relation_v03 as rv
    task = nulls[0]
    s = task.s
    base = {B: [rv.baseline_for(t, B) for t in nulls] for B in ("known", "nuis")}
    arr = np.asarray(sets, dtype=np.int64)
    Vs, Vq = rv.set_values(task, arr)
    gs, gq = task.groups[:s], task.groups[s:]
    jobs_full = [(base[B][i], t.y[:s], t.y[s:]) for B in ("known", "nuis") for i, t in enumerate(nulls)]
    full = fit_multi(Vs, Vq, gs, gq, jobs_full, device=device, chunk=chunk)
    res = {(B, i): full[bi * len(nulls) + i] for bi, B in enumerate(("known", "nuis")) for i in range(len(nulls))}
    dec = {}
    if kind == "pair":
        jobs_dec = [(base["nuis"][i], t.y[:s], t.y[s:]) for i, t in enumerate(nulls)]
        dres = fit_multi(Vs, Vq, gs, gq, jobs_dec, blocks=[[0], [1]], device=device, chunk=chunk)
        dec = {i: dres[i] for i in range(len(nulls))}
    return res, dec


def _scatter(store, part, idx, n):
    for k_, v in part.items():
        o = store.setdefault(k_, {"g_sup": np.full(n, np.nan), "gain_qp": np.full((len(v["sst_qp"]), n), np.nan),
                                  "sst_qp": v["sst_qp"]})
        o["g_sup"][idx], o["gain_qp"][:, idx] = v["g_sup"], v["gain_qp"]


def _fit_kind_grouped(nulls, kind, sets, names, target, device, chunk):
    """As _fit_kind; studies with glial columns refit per glial-proxy key as labels.score_groups does."""
    if not any(str(n).startswith("glialgene:") for n in (nulls[0].known_names or [])):
        return _fit_kind(nulls, kind, sets, device, chunk)
    keys = [L.proxy_genes_of([names[i] for i in st_], target) for st_ in sets]
    res, dec = {}, {}
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        r, d = _fit_kind([L.known_variant(t, key) for t in nulls], kind, [sets[i] for i in idx], device, chunk)
        _scatter(res, r, idx, len(sets))
        _scatter(dec, d, idx, len(sets))
    return res, dec


def score_split(task, ep, sets_s, sets_p, names, st_name, target, r, device, chunk):
    """Same record as labels.label_target for one split."""
    from dbench.harness import core
    nulls = [task] + [core.null_task(task, ep, "patient", L.stable_seed("bench-null", st_name, target, r, d))
                      for d in range(L.N_NULL)]
    rec = {"split": r}
    secs = {}
    for kind, sets, Bs in (("single", sets_s, ("known", "nuis")), ("pair", sets_p, ("known", "nuis", "decomp"))):
        t0 = time.time()
        res, dec = _fit_kind_grouped(nulls, kind, sets, names, target, device, chunk)
        for B in Bs:
            def piece(i):
                if B == "decomp":
                    f, d = res[("nuis", i)], dec[i]
                    return f["gain_qp"] - d["gain_qp"], f["sst_qp"], f["g_sup"] - d["g_sup"]
                f = res[(B, i)]
                return f["gain_qp"], f["sst_qp"], f["g_sup"]
            num, sst_, gsup = piece(0)
            G, lbs = L.pooled(num, sst_, seed=L.stable_seed(st_name, target, r, kind, B))
            rec[f"{kind}|{B}|G"] = G.astype(np.float32)
            rec[f"{kind}|{B}|gsup"] = np.asarray(gsup, dtype=np.float32)
            for lv, lb in lbs.items():
                rec[f"{kind}|{B}|lb{lv}"] = lb.astype(np.float32)
            null = [L.pooled(*piece(i)[:2], levels={})[0] for i in range(1, len(nulls))]
            rec[f"{kind}|{B}|null_p95"] = float(np.quantile(np.concatenate(null), 0.95))
        secs[kind] = round(time.time() - t0, 1)
    rec["seconds"] = secs
    return rec


def label_target_fast(st, target, device, log=print, max_splits=None, chunk=32):
    from dbench.harness import core
    seed = L.stable_seed("bench-task", st.name, target)
    built = []
    for r, support, query in L.splits_for(st, target):
        try:
            task, names, ep = core.build_task(st, target, support, query, seed=seed)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            log(f"  {target} split {r}: {exc}")
            continue
        built.append((r, support, query, task, list(names), ep))
    if len(built) < 5:
        return None
    singles = sorted(set.intersection(*[set(b[4]) for b in built]))
    rng = np.random.default_rng(L.stable_seed("bench-pairs", st.name, target))
    n_pairs = min(L.N_PAIRS, len(singles) * (len(singles) - 1) // 2)
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
        rec = score_split(task, ep, s_sets, p_sets, names, st.name, target, r, device, chunk)
        rec.update(support=support, query=query)
        out["splits"].append(rec)
        log(f"  {target} split {r}: singles {rec['seconds']['single']} s, pairs {rec['seconds']['pair']} s")
    return out


def verify(st, tindex, splits, chunk):
    """Known-answer test: recompute cached splits and compare with the v04 label cache."""
    from dbench.benchmark_v04 import tables as T
    tp = BENCH / "runs/development/benchmark_v04/labels" / st.name / "targets.json"
    target = json.loads(tp.read_text())["targets"][tindex]
    h = hashlib.sha256(target.encode()).hexdigest()[:10]
    z = np.load(BENCH / "runs/development/benchmark_v04/labels" / st.name / "cache" / f"{h}.npz")
    res = label_target_fast(st, target, "cuda", max_splits=splits, chunk=chunk)
    report = {"target": target, "seconds": [rec["seconds"] for rec in res["splits"]]}
    for rec in res["splits"]:
        r = rec["split"]
        for k, v in rec.items():
            if not isinstance(v, (np.ndarray, float)):
                continue
            ref = z[f"s{r}|{k}"]
            report.setdefault("maxabs|" + k, []).append(float(np.max(np.abs(np.asarray(v, float) - np.asarray(ref, float)))))
        for kind, B in (("single", "known"), ("single", "nuis"), ("pair", "known"), ("pair", "nuis"), ("pair", "decomp")):
            new = (rec[f"{kind}|{B}|lb95"] > 0) & (rec[f"{kind}|{B}|G"] > rec[f"{kind}|{B}|null_p95"])
            if B == "decomp":
                new &= rec["pair|known|G"] > 0
            old = T.label(z, r, kind, B, "95")
            report.setdefault(f"label_agree|{kind}|{B}", []).append([float((new == old).mean()), float(old.mean()), float(new.mean())])
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study")
    ap.add_argument("--out", default=str(OUT / "labels"))
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--verify", type=int, default=None)
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    L.diag()
    t0 = time.time()
    st = load_study_v05(a.study)
    print(f"loaded {a.study}: {st.n} spots, {len(st.patients)} patients, {len(st.names)} columns in {time.time() - t0:.0f} s", flush=True)
    if a.verify is not None:
        print(json.dumps(verify(st, a.verify, a.splits, a.chunk)), flush=True)
        return
    od = Path(a.out) / a.study
    tp = od / "targets.json"
    od.mkdir(parents=True, exist_ok=True)
    if tp.exists():
        targets = json.loads(tp.read_text())["targets"]
    else:
        targets = L.choose_targets(st, a.n_targets)
        tp.write_text(json.dumps({"study": a.study, "seed": L.SEED, "targets": targets, "eligible": L.eligible_state_targets(st),
                                  "rule": "state scores only (resid, programme, morph); gene targets excluded",
                                  "sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest()}))
    cache = od / "cache"
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if (cache / f"{h}.npz").exists():
            continue
        t1 = time.time()
        res = label_target_fast(st, t, a.device, chunk=a.chunk)
        if res is None:
            print(f"{t}: degenerate, skipped", flush=True)
            continue
        L.save_target(cache, t, res)
        print(f"{t}: {len(res['pool_singles'])} singles, {len(res['pool_pairs'])} pairs, {len(res['splits'])} splits, {time.time() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main()
