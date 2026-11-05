import sys, time, numpy as np
sys.path.insert(0, "/home/ubuntu/morpheus_work_bench")
from dbench.benchmark_v04 import methods_v04 as M
rng = np.random.default_rng(0)
n, m, k = 800, 60, 150
X = rng.normal(size=(n, m)); g = np.repeat(np.arange(8), 100)
pairs = np.array(sorted({tuple(sorted(rng.choice(m, 2, replace=False))) for _ in range(k)}))
r = X[:, 3] * 0.5 + X[:, 5] * X[:, 9] * 0.8 + rng.normal(size=n)
ctx = M.Ctx(X=X, r=r, groups=g, pairs=pairs, seed=1, n_jobs=2)
planted = [i for i, p in enumerate(pairs) if tuple(p) == (5, 9)]
for name in M.METHODS:
    t = time.time()
    try:
        s1, s2, info = M.run(name, ctx)
        rk = (np.argsort(-s2) == (planted[0] if planted else 0)).nonzero()[0] if planted else None
        print(name, "ok", round(time.time() - t, 1), "s", info, "single top", int(np.argmax(s1)), "planted pair rank", rk, flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc(); print(name, "FAILED", exc, flush=True)
