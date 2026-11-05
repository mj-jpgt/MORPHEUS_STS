"""Few-shot curve: full-pool AUROC and precision@10 against the number of support patients.

For each development target and fold, the fold's support patients are subsampled to n in {2, 4, 8, 16,
all} (5 seeds; n >= the fold's support count means all), the query patients and the pool labels
(``pool.labels.json``, B_known, pooled over every fold's query patients) stay fixed, and each method scores
the full pool from the subsampled support.  Metrics: AUROC and precision@10 of the method's pool ranking
(singles), mean over folds, 95 % bootstrap over targets.

    python -m dbench.harness.fewshot --methods g_theta,ebm,gbm_h,linear_screen,random --targets 40 --device cuda
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import core

SIZES = (2, 4, 8, 16, 0)       # 0 = all support patients
SEEDS = (0, 1, 2, 3, 4)
OUT = Path("runs/development/benchmark_v03/fewshot")


def subsample(task, patients_keep: set):
    """Task restricted to the support rows of ``patients_keep`` (query rows unchanged)."""
    s = task.s
    g = np.asarray(task.groups)
    keep = np.concatenate([np.flatnonzero(np.isin(g[:s], list(patients_keep))), np.arange(s, len(g))])
    if not np.isin(g[:s], list(patients_keep)).any():
        raise ValueError(f"no support rows for patients {sorted(patients_keep)[:4]} (id dtype {g.dtype})")
    remap = -np.ones(len(g), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    cut = lambda a: None if a is None else np.asarray(a)[keep]  # noqa: E731
    nb = None
    if task.nb6 is not None:
        nb0 = np.asarray(task.nb6)[keep]
        nb = np.where(nb0 >= 0, remap[np.where(nb0 >= 0, nb0, 0)], -1)
    return replace(task, X=task.X[keep], y=task.y[keep], n_support=int(np.sum(keep < s)), groups=g[keep],
                   depth=cut(task.depth), known=cut(task.known), composition=cut(task.composition), nb6=nb,
                   sections=cut(task.sections))


def _job(i):
    import torch
    torch.set_num_threads(1)
    fn, tk, keep, s_idx, p_idx = core._G["few"][i]
    try:   # subsample here: one task copy per running job, not every copy up front (~107 GB)
        r = fn(core.support_only(subsample(tk, keep)), "known", s_idx, p_idx)
    except Exception as exc:   # e.g. a method that cannot fit 2 support patients: recorded, not fatal
        print(f"few-shot job {i} failed: {type(exc).__name__}: {exc}"[:300], flush=True)
        return None
    return np.asarray(r[0], float), np.asarray(r[1], float)


def scorers(device=None) -> dict:
    from .core8 import pool_scorers
    table = pool_scorers(device=device)
    try:
        from morpheus_encoder.learned_search.search_v03 import pool_scores
        from .baselines import model_search  # noqa: F401  (frozen path convention)
        ck = "runs/development/learned_search_v03/{}/frozen.pt"

        def ours(run, kappa: float):
            """Predicted held-out gain G_sup - mu_delta (singles) and mu_sup - mu_delta (pairs, pair head);
            kappa > 0: the shrinkage lower confidence bound (minus kappa sd), the opt-in ordering."""
            cache = {}

            def score(task, B, singles, pairs):
                if "m" not in cache:
                    import os
                    from morpheus_encoder.learned_search.model_v03 import GainModelV03
                    path = os.environ.get("MORPHEUS_GTHETA", ck.format(run))
                    cache["m"] = GainModelV03.load(path, map_location=device or "cpu").to(device or "cpu")
                # core.support_only keeps known_names / target_name / panel (search_v03._support_only drops them on a
                # full task, which removes the B_known exclusion rule)
                s1, s2, sd1, sd2 = pool_scores(core.support_only(task), B, cache["m"], np.asarray(pairs), device=device)[:4]
                s1, s2 = np.asarray(s1, float)[np.asarray(singles)], np.asarray(s2, float)
                if kappa:
                    s1 = s1 - kappa * np.asarray(sd1, float)[np.asarray(singles)]
                    s2 = s2 - kappa * np.asarray(sd2, float)
                return s1, s2, []
            return score
        table["g_theta"] = ours("release", 0.0)
        table["g_theta_lcb"] = ours("release", 1.0)
        table["g_theta_rowsplit"] = ours("release_rowsplit", 0.0)   # Q2: row-split repaired checkpoint
        from .ablation_selection import ablation_scorers   # selection ablation (methods table)
        table.update(ablation_scorers(device))
    except ImportError:
        pass
    return table


def run(methods: list, n_targets: int, workers: int, device=None, out: Path = OUT) -> dict:
    from .pool import dev_folds, target_pools
    lab = json.loads((Path("runs/development/benchmark_v03/pool") / "labels.json").read_text())
    strict_path = Path("runs/development/benchmark_v03/pool") / "labels_flagged.json"
    lab_strict = json.loads(strict_path.read_text()) if strict_path.exists() else None
    spec, built = dev_folds()
    pools = target_pools(built)
    targets = spec["targets"][:n_targets]
    table = scorers(device)
    res = {}
    for m in methods:
        jobs, keys = [], []
        for t, f, conds, names, st, q in built:
            if t not in targets:
                continue
            tk = dict(conds)["real"]
            idx = {n: i for i, n in enumerate(names)}
            singles, pairs = pools[t]
            s_idx = np.array([idx[n] for n in singles])
            p_idx = np.array([(idx[a], idx[b]) for a, b in pairs])
            sp = np.unique(np.asarray(tk.groups)[:tk.s]).tolist()  # native dtype: np.isin in subsample must match
            for n in SIZES:
                for sd in SEEDS if (n and n < len(sp)) else (0,):
                    keep = set(np.random.default_rng(core.stable_seed(t, f, n, sd)).choice(sp, n, replace=False)) \
                        if n and n < len(sp) else set(sp)
                    jobs.append((table[m], tk, keep, s_idx, p_idx))
                    keys.append((t, f, n, sd))
        core._G["few"] = jobs
        gpu = str(device).startswith("cuda")   # every method sequential on the GPU (no fork after CUDA init)
        scores = [_job(i) for i in range(len(jobs))] if gpu else core._pool_map(_job, len(jobs), workers)
        core._G.clear()
        from .pool import _ap_auc
        from .public_metrics import boot_mean
        n_failed = sum(sc is None for sc in scores)
        failed_by_size, per, jobs_out = {}, {"single": {}, "pair": {}, "single_strict": {}, "pair_strict": {}}, []
        for (t, f, n, sd), sc in zip(keys, scores):
            if sc is None:
                failed_by_size[str(n)] = failed_by_size.get(str(n), 0) + 1
                continue
            rec = {"target": t, "fold": str(f), "size": int(n), "draw": int(sd)}
            for kind, sv in zip(("single", "pair"), sc):
                for tag, L in (("", lab), ("_strict", lab_strict)):
                    if L is None:
                        continue
                    y = np.asarray(L["labels"][f"{t}|{kind}|known"], bool)
                    if len(sv) != len(y):   # a scorer without pair scores (MISTy) or a failed pair pass
                        continue
                    au, ap = _ap_auc(y, sv, seeds=5)
                    fin = np.where(np.isfinite(sv), sv, -np.inf)
                    p10 = float(y[np.argsort(-fin, kind="stable")[:10]].mean())
                    per[kind + tag].setdefault(n, {}).setdefault(t, []).append((au, ap, p10))
                    rec[kind + tag] = {"auroc": au, "ap": ap, "precision@10": p10}
            jobs_out.append(rec)

        def summary(d):
            return {str(n): {"auroc": boot_mean([np.mean([a for a, _, _ in v if a is not None]) for v in dd.values()]),
                             "ap": boot_mean([np.mean([b for _, b, _ in v if b is not None]) for v in dd.values()]),
                             "precision@10": boot_mean([np.mean([p for _, _, p in v]) for v in dd.values()]),
                             "n_jobs": sum(len(v) for v in dd.values())} for n, dd in d.items()}
        res[m] = {"n_failed_jobs": n_failed, "failed_by_size": failed_by_size, "n_targets": len(targets),
                  **summary(per["single"]), "pairs": summary(per["pair"]),
                  "strict": {"singles": summary(per["single_strict"]), "pairs": summary(per["pair_strict"])}}
        out.mkdir(parents=True, exist_ok=True)
        (out / f"jobs_{m}_t{len(targets)}.json").write_text(json.dumps(jobs_out))
        out.mkdir(parents=True, exist_ok=True)
        (out / f"fewshot_{m}{'' if len(targets) == 40 else f'_t{len(targets)}'}.json").write_text(json.dumps(res[m], indent=1))
        print(m, json.dumps({n: round(d["auroc"][0] or 0, 3) for n, d in res[m].items() if isinstance(d, dict) and "auroc" in d}),
              "failed jobs", n_failed, flush=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", default="g_theta,ebm,gbm_h,linear_screen,random")
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    run(args.methods.split(","), args.targets, args.workers, args.device)


if __name__ == "__main__":
    main()
