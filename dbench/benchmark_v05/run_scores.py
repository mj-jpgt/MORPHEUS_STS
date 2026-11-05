"""Budgeted scorer for the benchmark methods on the shared pools (spec section 8, W3 speed plan).

    python -m dbench.benchmark_v05.run_scores STUDY --stage cpu --methods ebm,gbm_shap,... --workers 6 [--subset]
    python -m dbench.benchmark_v05.run_scores STUDY --stage gpu --methods sel_screen,sel_screen_joint,random,linear_screen,nid

One study load per process; targets are processed by forked workers (cpu stage) with every (method, split) fit run in its
own process group under a wall-clock budget.  A budget overrun or an error is recorded in the receipt
(``status`` per split) and never replaced by another method.  Receipts: OUT/scores/STUDY/METHOD/HASH.{npz,json}.
Pools, splits, tasks and seeds are those of v04 (checked against the label cache).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
from dbench.benchmark_v04 import score_methods as SM
from dbench.benchmark_v04 import tables as T
import numpy as np

from dbench.benchmark_v05 import methods_v05 as M5

BENCH = Path(L.BENCH)
OUT = BENCH / "runs/benchmark_v05"
LABEL_ROOTS = [OUT / "labels", BENCH / "runs/development/benchmark_v04/labels"]

#: wall-clock budget per (method, target, split), seconds
BUDGET = {"ebm": 360, "gbm_shap": 180, "gbm_h": 300, "misty": 120, "stability_selection": 360, "rulefit": 360, "irf": 300,
          "knockoffs": 240, "pysr": 600, "fair_gbm": 300, "fair_rulefit": 360, "fair_ebm": 300, "fair_irf": 360,
          "fair_misty": 300, "tabdistill": 300, "nid": 180, "linear_screen": 180}
#: methods declared on the seeded 3-of-10 split subset (cost), and PySR on 10 targets (spec 3.9)
SLOW_SUBSET = {"ebm", "gbm_h", "rulefit", "stability_selection", "gbm_shap", "knockoffs", "pysr", "tabdistill", "irf", "fair_gbm", "fair_rulefit", "fair_ebm", "fair_irf", "fair_misty"}
PYSR_TARGETS = 10


def label_dir(study: str) -> Path:
    for r in LABEL_ROOTS:
        if (r / study / "targets.json").exists():
            return r / study
    raise FileNotFoundError(study)


def thash(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:10]


def _child(fn, ctx, path):
    os.setsid()
    import multiprocessing as _mp
    _mp.current_process()._config["daemon"] = False   # let joblib/loky use n_jobs in the grandchild
    try:
        res = fn(ctx)
        with open(path, "wb") as f:
            pickle.dump(("ok", res), f)
    except BaseException as exc:   # noqa: BLE001
        with open(path, "wb") as f:
            pickle.dump(("error", repr(exc)[:300] + " | " + traceback.format_exc()[-600:]), f)
    os._exit(0)


def run_budgeted(name, ctx, budget):
    """(status, (s1, s2, info) | message, seconds).  Forks; kills the child's process group on overrun."""
    fd, path = tempfile.mkstemp(prefix="w3_", suffix=".pkl", dir="/dev/shm")
    os.close(fd)
    t0 = time.time()
    pid = os.fork()
    if pid == 0:
        _child(lambda c: M5.run_method(name, c), ctx, path)
    status = "timeout"
    while time.time() - t0 < budget:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done:
            status = "done"
            break
        time.sleep(0.5)
    if status == "timeout":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)   # the child's own process group only
            except ProcessLookupError:
                pass
            time.sleep(2)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    secs = round(time.time() - t0, 1)
    out = None
    if status == "done":
        try:
            with open(path, "rb") as f:
                out = pickle.load(f)
        except Exception as exc:   # noqa: BLE001
            out = ("error", f"no result: {exc!r}")
    try:
        os.unlink(path)
    except OSError:
        pass
    if status == "timeout":
        return "timeout", f"budget {budget} s", secs
    return ("done" if out[0] == "ok" else out[0]), out[1], secs


def make_ctx(task, names, singles, pairs, study, target, r, n_jobs):
    ctx, _, _ = SM._ctx(task, names, singles, pairs, L.stable_seed("bench-method", study, target, r), n_jobs)
    return M5.Ctx5(X=ctx.X, r=ctx.r, groups=ctx.groups, pairs=ctx.pairs, seed=ctx.seed, n_jobs=ctx.n_jobs, nb6=ctx.nb6,
                   names=list(singles))


def write_receipt(od: Path, h: str, target: str, method: str, arrays: dict, secs: dict, status: dict, infos: dict):
    od.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(od / f"{h}.npz", **arrays)
    (od / f"{h}.json").write_text(json.dumps({"target": target, "method": method, "seconds": secs, "status": status,
                                              "info": infos}))


def pending(od, h):
    return not (od / f"{h}.json").exists()


def load_receipt(od, h):
    """(arrays, seconds, status, infos) of an existing receipt, split keys as ints; None when absent."""
    jp = od / f"{h}.json"
    if not jp.exists():
        return None
    d = json.loads(jp.read_text())
    arrays = {}
    if (od / f"{h}.npz").exists():
        z = np.load(od / f"{h}.npz")
        arrays = {k: z[k] for k in z.files}
    status = {int(k): v for k, v in d.get("status", {}).items()}
    secs = {int(k): v for k, v in d.get("seconds", {}).items()}
    infos = {int(k): v for k, v in d.get("info", {}).items()}
    if not status:   # receipts copied from v04 carry no status: every stored split is done
        for k in arrays:
            status[int(k.split("|")[0][1:])] = "done"
    return arrays, secs, status, infos


def target_job(args):
    """Resume rule: cells with status done are kept; timed-out, errored and 'skipped after 2 consecutive failures' cells are re-run
    under the current budget."""
    study, ti, target, methods, subset_only, n_jobs, gpu = args
    st = _STUDY
    h = thash(target)
    subset0 = T.slow_subset(study, target)

    def missing(m):
        old = load_receipt(OUT / "scores" / study / m, h)
        want = [r for r in range(10) if not ((m in SLOW_SUBSET or subset_only) and r not in subset0)]
        return [r for r in want if not old or old[2].get(r) != "done"]
    todo = [m for m in methods if missing(m)]
    if m_skip := [m for m in todo if m == "pysr" and ti >= PYSR_TARGETS]:
        todo = [m for m in todo if m not in m_skip]
    if not todo:
        return f"{target}: nothing to do"
    t0 = time.time()
    built, singles, pairs = SM.build_pool_and_tasks(st, target)
    lab = label_dir(study) / "cache" / f"{h}.json"
    if lab.exists():
        meta = json.loads(lab.read_text())
        assert meta["pool_singles"] == singles and meta["pool_pairs"] == [[singles[x], singles[y]] for x, y in pairs], "pool differs from the label cache"
    subset = T.slow_subset(study, target)
    ctxs = {r: make_ctx(task, names, singles, pairs, study, target, r, n_jobs) for r, support, query, task, names, ep in built}
    lines = []
    for m in todo:
        old = load_receipt(OUT / "scores" / study / m, h)
        arrays, secs, status, infos = ({}, {}, {}, {}) if old is None else tuple(dict(x) for x in old)
        consecutive = 0
        for r in sorted(ctxs):
            if (m in SLOW_SUBSET or subset_only) and r not in subset:
                continue
            if status.get(r) == "done":
                continue
            if consecutive >= 2:
                status[r] = "skipped after 2 consecutive failures"
                continue
            if m == "random":
                S1, S2, _ = M5.M4.random_scores(ctxs[r])
                for k in range(len(S1)):
                    arrays[f"s{r}|single|{k}"], arrays[f"s{r}|pair|{k}"] = S1[k].astype(np.float32), S2[k].astype(np.float32)
                secs[r], status[r] = 0.0, "done"
                continue
            stt, res, sec = run_budgeted(m, ctxs[r], BUDGET.get(m, 240))
            secs[r], status[r] = sec, stt
            if stt == "done":
                s1, s2, info = res
                arrays[f"s{r}|single"], arrays[f"s{r}|pair"] = np.asarray(s1, np.float32), np.asarray(s2, np.float32)
                infos[r] = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
                consecutive = 0
            else:
                infos[r] = {"message": str(res)[:300]}
                consecutive += 1
        write_receipt(OUT / "scores" / study / m, h, target, m, arrays, secs, status, infos)
        n_ok = sum(1 for v in status.values() if v == "done")
        lines.append(f"{m} {n_ok}/{len(status)} ok {sum(secs.values()):.0f}s")
    return f"{study} t{ti} {target}: " + "; ".join(lines) + f" [total {time.time() - t0:.0f} s]"


_STUDY = None


def main(argv=None):
    global _STUDY
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study")
    ap.add_argument("--stage", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4, help="threads per fit")
    ap.add_argument("--subset", action="store_true", help="run every listed method on the 3-of-10 split subset only")
    ap.add_argument("--targets", default="0-39")
    ap.add_argument("--budget-scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    for k in BUDGET:
        BUDGET[k] = BUDGET[k] * a.budget_scale
    L.diag()
    methods = a.methods.split(",")
    t0 = time.time()
    from dbench.benchmark_v05.labels_fast import load_study_v05
    _STUDY = load_study_v05(a.study)
    print(f"loaded {a.study} in {time.time() - t0:.0f} s", flush=True)
    targets = json.loads((label_dir(a.study) / "targets.json").read_text())["targets"]
    lo, hi = (int(x) for x in a.targets.split("-")) if "-" in a.targets else (int(a.targets), int(a.targets))
    jobs = [(a.study, i, targets[i], methods, a.subset, a.jobs, a.stage == "gpu") for i in range(lo, min(hi, len(targets) - 1) + 1)]
    if a.stage == "gpu":
        for j in jobs:
            print(target_job_gpu(j), flush=True)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        for line in pool.imap_unordered(target_job, jobs, chunksize=1):
            print(line, flush=True)


def target_job_gpu(args):
    """GPU stage: methods run in-process (no fork: CUDA); the timeout is a SIGALRM raised in Python."""
    study, ti, target, methods, subset_only, n_jobs, _ = args
    st = _STUDY
    h = thash(target)
    todo = [m for m in methods if pending(OUT / "scores" / study / m, h)]
    if not todo:
        return f"{target}: nothing to do"
    from dbench.harness import core
    t0 = time.time()
    built, singles, pairs = SM.build_pool_and_tasks(st, target)
    lab = label_dir(study) / "cache" / f"{h}.json"
    if lab.exists():
        meta = json.loads(lab.read_text())
        assert meta["pool_singles"] == singles and meta["pool_pairs"] == [[singles[x], singles[y]] for x, y in pairs], "pool differs from the label cache"
    subset = T.slow_subset(study, target)
    lines = []
    for m in todo:
        arrays, secs, status, infos = {}, {}, {}, {}
        for r, support, query, task, names, ep in built:
            if (m in SLOW_SUBSET or subset_only) and r not in subset:
                continue
            t1 = time.time()
            try:
                if m in ("sel_screen", "sel_screen_joint"):
                    tk = L.known_variant(core.support_only(task), ())
                    s1, s2, info = SM.score_ours(m, tk, names, singles, pairs, "cuda")
                elif m == "random":
                    ctx = make_ctx(task, names, singles, pairs, study, target, r, n_jobs)
                    S1, S2, _ = M5.M4.random_scores(ctx)
                    for k in range(len(S1)):
                        arrays[f"s{r}|single|{k}"], arrays[f"s{r}|pair|{k}"] = S1[k].astype(np.float32), S2[k].astype(np.float32)
                    secs[r], status[r] = round(time.time() - t1, 2), "done"
                    continue
                else:
                    ctx = make_ctx(task, names, singles, pairs, study, target, r, n_jobs)
                    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("budget")))
                    signal.alarm(int(BUDGET.get(m, 240)))
                    try:
                        s1, s2, info = M5.run_method(m, ctx)
                    finally:
                        signal.alarm(0)
                arrays[f"s{r}|single"], arrays[f"s{r}|pair"] = np.asarray(s1, np.float32), np.asarray(s2, np.float32)
                status[r] = "done"
                infos[r] = {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str))}
            except TimeoutError:
                status[r] = "timeout"
            except Exception as exc:   # noqa: BLE001
                status[r] = "error"
                infos[r] = {"message": (repr(exc)[:200] + " | " + traceback.format_exc()[-400:])}
            secs[r] = round(time.time() - t1, 1)
        write_receipt(OUT / "scores" / study / m, h, target, m, arrays, secs, status, infos)
        lines.append(f"{m} {sum(1 for v in status.values() if v == 'done')}/{len(status)} ok {sum(secs.values()):.0f}s")
    return f"{study} t{ti} {target}: " + "; ".join(lines) + f" [total {time.time() - t0:.0f} s]"


if __name__ == "__main__":
    main()
