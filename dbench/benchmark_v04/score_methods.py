"""Score the shared pool of every (study, target, split) with the benchmark methods; one npz per target and method.

    python -m dbench.benchmark_v04.score_methods STUDY --methods sel_screen,sel_screen_joint,random,linear_screen \
        [--splits 0-9] [--max-targets N] [--jobs J] [--out DIR]

The pool, splits and tasks are rebuilt with the same seeds as labels.py (checked against the label cache when present).
Methods see support rows only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from dbench.benchmark_v04 import labels as L  # sets sys.path (second_wave first, then bench)
import numpy as np

BENCH = Path(L.BENCH)
FAST = ("sel_screen", "sel_screen_joint", "random", "linear_screen", "ebm", "gbm_shap", "gbm_h", "rulefit",
        "stability_selection")


def build_pool_and_tasks(st, target):
    from dbench.harness import core
    seed = L.stable_seed("bench-task", st.name, target)
    built = []
    for r, support, query in L.splits_for(st, target):
        try:
            task, names, ep = core.build_task(st, target, support, query, seed=seed)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue
        built.append((r, support, query, task, list(names), ep))
    singles = sorted(set.intersection(*[set(b[4]) for b in built]))
    rng = np.random.default_rng(L.stable_seed("bench-pairs", st.name, target))
    n_pairs = min(L.N_PAIRS, len(singles) * (len(singles) - 1) // 2)
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(len(singles), 2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    return built, singles, sorted(pairs)


def _ctx(task, names, singles, pairs, seed, n_jobs):
    from dbench.harness import core
    from morpheus_encoder.learned_search import relation_v03 as rv
    from dbench.benchmark_v04.methods_v04 import Ctx
    tk = L.known_variant(core.support_only(task), ())
    base = rv.baseline_for(tk, "known")
    idx = {n: i for i, n in enumerate(names)}
    cols = np.array([idx[n] for n in singles])
    s = tk.s
    return Ctx(X=np.asarray(tk.X[:s][:, cols], dtype=np.float64), r=np.asarray(base.r_s, dtype=np.float64),
               groups=np.asarray(tk.groups[:s]), pairs=np.asarray(pairs), seed=seed, n_jobs=n_jobs,
               nb6=None if tk.nb6 is None else np.asarray(tk.nb6[:s])), tk, cols


def score_ours(name, tk, names, singles, pairs, device):
    from dbench.harness.ablation_selection import pool_scorer
    idx = {n: i for i, n in enumerate(names)}
    s_idx = np.array([idx[n] for n in singles])
    p_idx = np.array([(idx[singles[a]], idx[singles[b]]) for a, b in pairs])
    s1, s2, _ = pool_scorer(name, device)(tk, "known", s_idx, p_idx)
    return np.asarray(s1, float), np.asarray(s2, float), {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study")
    ap.add_argument("--methods", default="sel_screen,sel_screen_joint,random,linear_screen")
    ap.add_argument("--splits", default="0-9")
    ap.add_argument("--slow-subset", action="store_true", help="only the fixed seeded 3-of-10 splits of each target")
    ap.add_argument("--max-targets", type=int, default=40)
    ap.add_argument("--target-index", type=int, default=None,
                    help="Score exactly this zero-based target index.  Used by the bounded worker scheduler.")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(BENCH / "runs/development/benchmark_v04/scores"))
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.splits.split("-")) if "-" in a.splits else (int(a.splits), int(a.splits))
    L.diag()
    st = L.load_test_study(a.study)
    tp = BENCH / "runs/development/benchmark_v04/labels" / a.study / "targets.json"
    targets = json.loads(tp.read_text())["targets"][: a.max_targets]
    if a.target_index is not None:
        if not 0 <= a.target_index < len(targets):
            raise ValueError(f"target index {a.target_index} outside 0..{len(targets) - 1}")
        targets = [targets[a.target_index]]
    from dbench.benchmark_v04 import methods_v04 as M
    methods = a.methods.split(",")
    if "linear_screen" in methods:
        print("linear_screen GPU-vs-sklearn max coefficient error", M.check_linear_equivalence(), flush=True)
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        todo = [m for m in methods if not (Path(a.out) / a.study / m / f"{h}.npz").exists()]
        if not todo:
            continue
        t0 = time.time()
        built, singles, pairs = build_pool_and_tasks(st, t)
        lab = BENCH / "runs/development/benchmark_v04/labels" / a.study / "cache" / f"{h}.json"
        if lab.exists():
            meta = json.loads(lab.read_text())
            assert meta["pool_singles"] == singles and meta["pool_pairs"] == [[singles[x], singles[y]] for x, y in pairs], "pool differs from the label cache"
        from dbench.benchmark_v04 import tables as T
        subset = T.slow_subset(a.study, t)
        for m in todo:
            arrays, secs = {}, {}
            for r, support, query, task, names, ep in built:
                if not lo <= r <= hi:
                    continue
                if a.slow_subset and r not in subset:
                    continue
                t1 = time.time()
                if m in ("sel_screen", "sel_screen_joint"):
                    from dbench.harness import core
                    tk = L.known_variant(core.support_only(task), ())
                    s1, s2, info = score_ours(m, tk, names, singles, pairs, a.device)
                else:
                    ctx, _, _ = _ctx(task, names, singles, pairs, L.stable_seed("bench-method", a.study, t, r), a.jobs)
                    if m == "random":
                        S1, S2, info = M.random_scores(ctx)
                        for k in range(len(S1)):
                            arrays[f"s{r}|single|{k}"], arrays[f"s{r}|pair|{k}"] = S1[k].astype(np.float32), S2[k].astype(np.float32)
                        secs[r] = round(time.time() - t1, 2)
                        continue
                    try:
                        s1, s2, info = M.run(m, ctx)
                    except Exception as exc:   # one serious attempt per split; the error is recorded, not hidden
                        import traceback
                        (Path(a.out) / a.study).mkdir(parents=True, exist_ok=True)
                        with open(Path(a.out) / a.study / "errors.log", "a") as ef:
                            ef.write(json.dumps({"method": m, "target": t, "split": r, "error": repr(exc)[:400],
                                                 "trace": traceback.format_exc()[-1200:]}) + chr(10))
                        print(f"  {m} {t} split {r} FAILED: {repr(exc)[:160]}", flush=True)
                        continue
                arrays[f"s{r}|single"], arrays[f"s{r}|pair"] = np.asarray(s1, np.float32), np.asarray(s2, np.float32)
                secs[r] = round(time.time() - t1, 1)
            od = Path(a.out) / a.study / m
            if not arrays:
                continue
            od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f"{h}.npz", **arrays)
            (od / f"{h}.json").write_text(json.dumps({"target": t, "seconds": secs, "method": m}))
            print(f"{a.study} {t} {m}: {len(secs)} splits, {sum(secs.values()):.0f} s", flush=True)
        print(f"{t}: total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
