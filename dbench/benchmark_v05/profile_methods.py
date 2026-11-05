"""Profile every baseline on one (study, target, split): seconds and peak memory at several support-row caps."""
import sys, time, json, resource
from dbench.benchmark_v04 import labels as L
from dbench.benchmark_v04 import score_methods as SM
from dbench.benchmark_v04 import methods_v04 as M
import numpy as np

study, tidx = sys.argv[1], int(sys.argv[2])
methods = sys.argv[3].split(",")
caps = [int(x) for x in sys.argv[4].split(",")]
t0 = time.time()
st = L.load_test_study(study)
print("load", round(time.time() - t0), "s", flush=True)
targets = json.load(open(L.BENCH + "/runs/development/benchmark_v04/labels/" + study + "/targets.json"))["targets"]
t = targets[tidx]
from dbench.harness import core
seed = L.stable_seed("bench-task", st.name, t)
r, support, query = L.splits_for(st, t)[0]
t1 = time.time()
task, names, ep = core.build_task(st, t, support, query, seed=seed)
print("build_task", round(time.time() - t1, 1), "s", t, "X", task.X.shape, "names", len(names), "maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, flush=True)
singles = sorted(names)
rng = np.random.default_rng(1)
pairs = sorted({tuple(sorted(rng.choice(len(singles), 2, replace=False).tolist())) for _ in range(2000)})
t1 = time.time()
ctx, tk, cols = SM._ctx(task, names, singles, pairs, 1, 8)
print("ctx", round(time.time() - t1, 1), "s support rows", ctx.X.shape, flush=True)
def cap(ctx, n):
    if n >= len(ctx.r): return ctx
    rr = np.random.default_rng(0); keep = []
    for g in np.unique(ctx.groups):
        i = np.flatnonzero(ctx.groups == g); k = max(1, int(round(n * len(i) / len(ctx.r))))
        keep.append(rr.choice(i, min(k, len(i)), replace=False))
    keep = np.sort(np.concatenate(keep))
    from dataclasses import replace
    return replace(ctx, X=ctx.X[keep], r=ctx.r[keep], groups=ctx.groups[keep], nb6=None)
for n in caps:
    c = cap(ctx, n)
    for m in methods:
        t1 = time.time()
        try:
            s1, s2, info = M.run(m, c)
            print(f"RESULT cap={n} rows={len(c.r)} {m}: {time.time()-t1:.1f} s  nonzero pairs {int((np.asarray(s2)!=0).sum())}", flush=True)
        except Exception as e:
            print(f"RESULT cap={n} {m}: FAILED {time.time()-t1:.1f} s {repr(e)[:200]}", flush=True)
