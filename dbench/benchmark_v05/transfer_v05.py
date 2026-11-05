"""Cross-study replication tasks for B2 / MC1 (spec transfer levels): support = every patient of a discovery cohort, query =
every patient of an independent cohort, no refit on the query; same pool, labels and methods as the B1 tasks.

    python -m dbench.benchmark_v05.transfer_v05 SUPPORT QUERY labels  [--n-targets 40]
    python -m dbench.benchmark_v05.transfer_v05 SUPPORT QUERY scores --stage gpu|cpu --methods a,b [--workers N]

Study name in the tables: ``SUPPORT__to__QUERY``.  Label = patient-bootstrap lower bound of the held-out gain on the query
cohort above 0 and gain above the 95th percentile of the patient-permuted-target null (the B1 rule), over B_known; the
relation is fitted on the discovery cohort only.  One split (index 0) per target.  Targets: programme scores defined in both
cohorts (resid and morphology targets are not defined in the external cohorts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
import numpy as np

from dbench.benchmark_v05 import labels_fast as LF
from dbench.benchmark_v05 import run_scores as RS

BENCH = Path(L.BENCH)
V5 = BENCH / "runs/benchmark_v05"
SEED = 20261102


def load(key):
    return L.load_test_study(key)


def pair_name(s, q):
    return f"{s}__to__{q}"


def pick_targets(st, qs, n):
    ok = [t for t in L.eligible_state_targets(st) if t.startswith("programme:") and t in qs.index]
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(ok, min(n, len(ok)), replace=False).tolist()) if len(ok) > n else sorted(ok)


def build(st, qs, target, tag):
    from morpheus_encoder.learned_search.episodes_v03 import SamplerV03, task_from_episode
    sampler = SamplerV03([st], controlled_prob=0.0, seed=SEED)
    sup = [str(p) for p in st.patients]
    qp = [str(p) for p in qs.patients]
    ep = sampler.episode_for(st, target, sup, qp, rng=np.random.default_rng(L.stable_seed("xfer", tag, target)),
                             query_study=qs, level="L2")
    return task_from_episode(ep), list(ep.meta.names), ep, sup, qp


def pool_of(names, tag, target):
    singles = sorted(names)
    rng = np.random.default_rng(L.stable_seed("bench-pairs", tag, target))
    n_pairs = min(L.N_PAIRS, len(singles) * (len(singles) - 1) // 2)
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(len(singles), 2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    return singles, sorted(pairs)


def labels(support, query, n_targets, chunk=32):
    st, qs = load(support), load(query)
    tag = pair_name(support, query)
    od = V5 / "labels" / tag
    od.mkdir(parents=True, exist_ok=True)
    targets = pick_targets(st, qs, n_targets)
    (od / "targets.json").write_text(json.dumps({"study": tag, "seed": SEED, "targets": targets, "rule": "programme targets defined in both cohorts",
                                                 "sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest()}))
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if (od / "cache" / f"{h}.npz").exists():
            continue
        t0 = time.time()
        try:
            task, names, ep, sup, qp = build(st, qs, t, tag)
        except ValueError as exc:
            print(f"{t}: {exc}", flush=True)
            continue
        singles, pairs = pool_of(names, tag, t)
        idx = {n: i for i, n in enumerate(names)}
        s_sets = [(idx[n],) for n in singles]
        p_sets = [(idx[singles[a]], idx[singles[b]]) for a, b in pairs]
        rec = LF.score_split(task, ep, s_sets, p_sets, names, tag, t, 0, "cuda", chunk)
        rec.update(support=sup, query=qp)
        res = {"pool_singles": singles, "pool_pairs": [[singles[a], singles[b]] for a, b in pairs], "splits": [rec]}
        L.save_target(od / "cache", t, res)
        print(f"{t}: {len(singles)} singles, {len(pairs)} pairs, {time.time() - t0:.0f} s", flush=True)


def score_job(args):
    support, query, target, methods, stage, n_jobs = args
    st, qs = RS._STUDY
    tag = pair_name(support, query)
    h = RS.thash(target)
    todo = [m for m in methods if RS.pending(RS.OUT / "scores" / tag / m, h)]
    if not todo:
        return f"{target}: nothing to do"
    from dbench.harness import core
    task, names, ep, sup, qp = build(st, qs, target, tag)
    singles, pairs = pool_of(names, tag, target)
    lab = V5 / "labels" / tag / "cache" / f"{h}.json"
    if lab.exists():
        meta = json.loads(lab.read_text())
        assert meta["pool_singles"] == singles, "pool differs from the label cache"
    lines = []
    ctx = None
    for m in todo:
        arrays, secs, status, infos = {}, {}, {}, {}
        t1 = time.time()
        try:
            if m in ("sel_screen", "sel_screen_joint"):
                from dbench.benchmark_v04 import score_methods as SM
                tk = L.known_variant(core.support_only(task), ())
                s1, s2, info = SM.score_ours(m, tk, names, singles, pairs, "cuda")
                stt = "done"
            else:
                from dbench.benchmark_v04 import score_methods as SM
                c0, _, _ = SM._ctx(task, names, singles, pairs, L.stable_seed("bench-method", tag, target, 0), n_jobs)
                ctx = RS.M5.Ctx5(X=c0.X, r=c0.r, groups=c0.groups, pairs=c0.pairs, seed=c0.seed, n_jobs=c0.n_jobs, nb6=c0.nb6, names=list(singles))
                if m == "random":
                    S1, S2, _ = RS.M5.M4.random_scores(ctx)
                    for k in range(len(S1)):
                        arrays[f"s0|single|{k}"], arrays[f"s0|pair|{k}"] = S1[k].astype(np.float32), S2[k].astype(np.float32)
                    secs[0], status[0] = 0.0, "done"
                    RS.write_receipt(RS.OUT / "scores" / tag / m, h, target, m, arrays, secs, status, infos)
                    continue
                if stage == "gpu":
                    import signal
                    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("budget")))
                    signal.alarm(int(RS.BUDGET.get(m, 240)))
                    try:
                        s1, s2, info = RS.M5.run_method(m, ctx)
                        stt = "done"
                    finally:
                        signal.alarm(0)
                else:
                    stt, res, _ = RS.run_budgeted(m, ctx, RS.BUDGET.get(m, 240))
                    if stt == "done":
                        s1, s2, info = res
                    else:
                        info = {"message": str(res)[:300]}
        except TimeoutError:
            stt, info = "timeout", {"message": "budget"}
        except Exception as exc:   # noqa: BLE001
            stt, info = "error", {"message": repr(exc)[:300]}
        secs[0], status[0] = round(time.time() - t1, 1), stt
        if stt == "done":
            arrays["s0|single"], arrays["s0|pair"] = np.asarray(s1, np.float32), np.asarray(s2, np.float32)
        infos[0] = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
        RS.write_receipt(RS.OUT / "scores" / tag / m, h, target, m, arrays, secs, status, infos)
        lines.append(f"{m} {stt} {secs[0]}s")
    return f"{tag} {target}: " + "; ".join(lines)


def scores(support, query, methods, stage, workers, jobs):
    st, qs = load(support), load(query)
    RS._STUDY = (st, qs)
    tag = pair_name(support, query)
    targets = json.loads((V5 / "labels" / tag / "targets.json").read_text())["targets"]
    args = [(support, query, t, methods, stage, jobs) for t in targets]
    if stage == "gpu" or workers <= 1:
        for a in args:
            print(score_job(a), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        for line in pool.imap_unordered(score_job, args, chunksize=1):
            print(line, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("support")
    ap.add_argument("query")
    ap.add_argument("mode", choices=["labels", "scores"])
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--stage", default="cpu")
    ap.add_argument("--methods", default="sel_screen,sel_screen_joint,random,linear_screen")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    L.diag()
    if a.mode == "labels":
        labels(a.support, a.query, a.n_targets)
    else:
        scores(a.support, a.query, a.methods.split(","), a.stage, a.workers, a.jobs)


if __name__ == "__main__":
    main()
