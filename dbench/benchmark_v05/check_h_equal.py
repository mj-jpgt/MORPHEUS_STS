"""Known-answer check: batched Friedman H equals the v04 per-pair loop (same model, grid, background) on one stored split."""
import json, time, numpy as np
from dbench.benchmark_v04 import labels as L, score_methods as SM
from dbench.benchmark_v05 import methods_v05 as M5, run_scores as RS
study = "gse237183_glioma"
st = L.load_test_study(study)
t = json.loads((RS.label_dir(study) / "targets.json").read_text())["targets"][0]
built, singles, pairs = SM.build_pool_and_tasks(st, t)
r, support, query, task, names, ep = [b for b in built if b[0] == 3][0]
ctx = RS.make_ctx(task, names, singles, pairs, study, t, r, 3)
c = M5.cap_ctx(ctx)
t0 = time.time(); _, H0, _ = M5.M4.friedman_h(c, grid=20, n_bg=20); t_loop = time.time() - t0
t0 = time.time(); _, H1, _ = M5.friedman_h_batched(c, grid=20, n_bg=20); t_batch = time.time() - t0
print(json.dumps({"target": t, "split": r, "pairs": len(H0), "loop_seconds": round(t_loop, 1), "batched_seconds": round(t_batch, 1),
                  "max_abs_diff": float(np.max(np.abs(H0 - H1))), "bit_identical": bool(np.array_equal(H0, H1))}), flush=True)
