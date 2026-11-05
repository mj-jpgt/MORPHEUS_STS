import sys, time, os, numpy as np
sys.path.insert(0, "/home/ubuntu/morpheus_work_bench")
from dbench.benchmark_v04 import methods_v04 as M, methods_slow as S
print("NID reference check err", S.check_nid_against_reference(os.path.expanduser("~/downloads_untrusted/nid_20261028/repo")), flush=True)
rng = np.random.default_rng(0)
n, m, k = 800, 60, 150
X = rng.normal(size=(n, m)); g = np.repeat(np.arange(8), 100)
pairs = np.array(sorted({tuple(sorted(rng.choice(m, 2, replace=False))) for _ in range(k)}))
pairs = np.vstack([pairs, [[5, 9]]]) if not any((pairs == [5, 9]).all(1)) else pairs
r = X[:, 3] * 0.5 + X[:, 5] * X[:, 9] * 0.8 + rng.normal(size=n)
nb = -np.ones((n, 6), dtype=int)
for i in range(n):
    for j, d in enumerate((-1, 1)):
        if 0 <= i + d < n and g[i + d] == g[i]: nb[i, j] = i + d
ctx = M.Ctx(X=X, r=r, groups=g, pairs=pairs, seed=1, n_jobs=4, nb6=nb)
pl = [i for i, p in enumerate(pairs) if tuple(p) == (5, 9)][0]
for name in sys.argv[1:] or list(S.SLOW):
    t = time.time()
    try:
        s1, s2, info = M.run(name, ctx)
        print(name, "ok", round(time.time() - t, 1), "s", info, "top single", int(np.argmax(s1)), "planted pair rank", int((np.argsort(-s2) == pl).nonzero()[0][0]), flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc(); print(name, "FAILED", repr(exc)[:300], flush=True)
