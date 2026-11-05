"""Standard ranking / retrieval metrics over the harness's frozen registries (owner directive 2026-10-21).

Ground truth, method-independent: a candidate relation (a set of 1-4 candidates) REPLICATES at baseline B
if its realised held-out gain, fitted on each fold's support patients and scored on that fold's query
patients and pooled over every fold whose candidate pool contains its members (each fold keeps its own
random 256 of the 2,000 pool genes), has a 95 % patient-bootstrap interval with lower bound > 0 (the
harness's own realised outcome).  The candidate universe of a target is the union of every set any
compared method registered for it.

Per query unit (development: target x fold; controlled worlds: world) and method:
* AUROC and average precision over the universe, the method's registered sets ranked in its order and
  every set it did not register ranked below them, ties broken at random (mean over 20 seeds);
* precision@k and recall@k (k = 5, 10, 20): replicating sets in the method's top k over k, and over all
  replicating sets of the universe;
* controlled worlds also: hit@k (k = 1, 5, 10, 20) and reciprocal rank of the planted relation, and
  constituent recall@k (from the harness rows).
Metrics are averaged over folds within a target and reported as the mean over targets (worlds) with a 95 %
bootstrap interval over targets (worlds), the query units of the benchmark (2,000 draws).

    python -m dbench.harness.public_metrics dev --B nuis --methods screen=runs/.../dev_metric_screen/step000000 ...
    python -m dbench.harness.public_metrics worlds --out runs/development/benchmark_v03/Q1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

KS = (5, 10, 20)
SEEDS = 20
N_BOOT = 2000
#: display names (owner directive): published methods first, ours named, statistical controls apart
NAMES = {"ebm": "explainable boosting machine (EBM)", "gbm_h": "gradient boosting + Friedman's H",
         "rulefit": "RuleFit (scikit-learn reimplementation)", "pysr": "PySR symbolic regression",
         "tabdistill": "TabPFN-v2 + SPEX", "tabdistill_reduced": "TabPFN-v2 + SPEX (reduced attribution budget: 256 masks, 64 rows, 1 estimator)", "linear_screen": "linear interaction screen (lasso, mains + pairwise)",
         "random": "random selection", "residual_gen": "MORPHEUS residual generator (previous version)",
         "g_theta": "learned relation search (g_theta)", "g_theta_rowsplit": "learned relation search, row-split episodes",
         "g_theta_lcb": "learned relation search, shrinkage-corrected ordering",
         "screen": "support-fit ranking", "lopo_heuristic": "variance-penalised support ranking",
         "stat_gbm": "gradient-boosting shrinkage predictor",
         "misty": "MISTy multi-view random forest (scikit-learn reimplementation)",
         "nid": "neural interaction detection (NID)",
         "ebm_full": "explainable boosting machine (EBM; mains + FAST strength for every pair)",
         "gbm_h_full": "gradient boosting + Friedman's H (permutation importance + H for every pair)"}
CONTROLS = ("screen", "lopo_heuristic", "stat_gbm")


def display(m: str) -> str:
    if m.startswith("full@") or m.startswith("rowsplit@"):
        run, step = m.split("@")
        return f"learned relation search{'' if run == 'full' else ', row-split episodes'} (step {step})"
    return NAMES.get(m, m)


# ------------------------------------------------------------------ ranking metrics
def ranking_metrics(ranked: list, universe: list, label: dict, ks=KS, seeds: int = SEEDS) -> dict:
    """``ranked``: the method's sets in order; ``universe``: every set; ``label``: set -> bool."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    ranked = list(dict.fromkeys(ranked))
    pos = sum(label[s] for s in universe)
    out = {f"precision@{k}": float(np.mean([label[s] for s in ranked[:k]])) if ranked[:k] else 0.0 for k in ks}
    out.update({f"recall@{k}": (sum(label[s] for s in ranked[:k]) / pos) if pos else None for k in ks})
    rest = [s for s in universe if s not in set(ranked)]
    y_r = np.array([label[s] for s in ranked], dtype=float)
    y_u = np.array([label[s] for s in rest], dtype=float)
    if 0 < pos < len(universe):
        au, ap = [], []
        for sd in range(seeds):
            y = np.concatenate([y_r, np.random.default_rng(sd).permutation(y_u)])
            score = -np.arange(len(y), dtype=float)
            au.append(roc_auc_score(y, score))
            ap.append(average_precision_score(y, score))
        out["auroc"], out["ap"] = float(np.mean(au)), float(np.mean(ap))
    else:
        out["auroc"] = out["ap"] = None
    out["n_universe"], out["n_replicating"] = len(universe), int(pos)
    return out


def boot_mean(v, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return (None, None, None)
    m = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ------------------------------------------------------------------ ground truth
def _label_job(i):
    import torch
    torch.set_num_threads(1)
    tk, sets, B = core._G["label"][i]
    sc = core.score_sets(tk, sets, Bs=(B,), device=None)[B]
    return sc[0], sc[1]          # per query patient numerators (Pq, m), SST (Pq,)


def labels_pooled(per_fold: list, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Replicates (bool per set): pooled over the folds' query patients, 95 % patient-bootstrap lower > 0."""
    num = np.concatenate([f[0] for f in per_fold], 0)          # (P, m)
    sst = np.concatenate([f[1] for f in per_fold], 0)          # (P,)
    num = np.where(np.isfinite(num), num, 0.0)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(sst), np.full(len(sst), 1 / len(sst)), size=n_boot)      # (n_boot, P)
    boot = (w @ num) / np.maximum(w @ sst, 1e-12)[:, None]
    return np.quantile(boot, 0.025, axis=0) > 0


# ------------------------------------------------------------------ development targets
def load_registry(path: Path, B: str = "nuis") -> dict:
    """(target, fold) -> [set (sorted name tuple), ...] for the REAL condition searched under ``B``."""
    reg = json.loads((Path(path) / "registry.json").read_text())
    if core.registry_hash(reg["entries"]) != reg["sha256"]:
        raise ValueError(f"{path}: registry does not match its sha256")
    return {(e["target"], e["fold"]): [tuple(sorted(s)) for s in e["sets"]]
            for e in reg["entries"] if e["condition"] == "real" and e.get("B", "nuis") == B}


def dev_table(methods: dict, B: str = "nuis", workers: int = 12, out: Path | None = None, list_B: str | None = None) -> dict:
    """``methods``: name -> registry directory.  Lists searched under B are scored against labels at B."""
    from .dev_metric import TARGETS, load_dev_study, load_targets

    spec = load_targets(TARGETS)
    study = load_dev_study(spec["study"], spec["partitions"])
    folds = core.patient_folds(study.patients, spec["n_folds"], spec["seed"])
    built = core._tasks(study, spec["targets"], folds, (), 0, spec["seed"])      # real tasks only
    regs = {m: load_registry(p, list_B or B) for m, p in methods.items()}
    jobs, keys = [], []
    universe = {}
    for t, f, conds, names, st, q in built:
        u = sorted({s for r in regs.values() for (tt, ff), sets in r.items() if tt == t for s in sets})
        universe[t] = u
    present = {}
    for t, f, conds, names, st, q in built:
        index = {n: i for i, n in enumerate(names)}
        tk = dict(conds)["real"]
        ok = [s for s in universe[t] if all(n in index for n in s)]   # the fold's 256-gene subset varies
        present[(t, f)] = ok
        jobs.append((tk, [tuple(index[n] for n in s) for s in ok], B))
        keys.append((t, f))
    core._G["label"] = jobs
    res = core._pool_map(_label_job, len(jobs), workers)
    core._G.clear()
    lab = {}
    for t in universe:
        num, sst = {s: [] for s in universe[t]}, {s: [] for s in universe[t]}
        for (tt, f), (n_, s_) in zip(keys, res):
            if tt != t:
                continue
            for j, sset in enumerate(present[(tt, f)]):
                num[sset].append(n_[:, j])
                sst[sset].append(s_)
        lab[t] = {sset: bool(labels_pooled([(np.concatenate(num[sset])[:, None], np.concatenate(sst[sset]))])[0])
                  for sset in universe[t]}
    rows = {}
    for m, r in regs.items():
        per_target = {}
        for t in universe:
            ms = [ranking_metrics(r.get((t, f), []), universe[t], lab[t]) for f in range(spec["n_folds"])]
            per_target[t] = {k: (None if all(x[k] is None for x in ms) else
                                 float(np.mean([x[k] for x in ms if x[k] is not None]))) for k in ms[0]}
        rows[m] = {k: boot_mean([d[k] for d in per_target.values()]) for k in next(iter(per_target.values()))}
    base_rate = boot_mean([np.mean(list(lab[t].values())) for t in universe])
    result = {"B": B, "list_B": list_B or B, "targets_sha256": spec["sha256"], "n_targets": len(universe), "methods": rows,
              "universe_size_mean": float(np.mean([len(u) for u in universe.values()])), "base_rate": base_rate,
              "registries": {m: str(p) for m, p in methods.items()}}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        tag = B if not list_B or list_B == B else f"{B}_lists{list_B}"
        (out / f"dev_public_{tag}.json").write_text(json.dumps(result, indent=1))
        (out / f"dev_public_{tag}.md").write_text(table_md(
            result, f"Development targets (kidney_injury, 40 targets x 3 folds), labels at B_{B}, lists searched "
                    f"under B_{list_B or B}"))
    return result


def table_md(result: dict, title: str, extra: tuple = ()) -> str:
    def f(t, nd=3):
        return "-" if t is None or t[0] is None else f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"
    cols = ("auroc", "ap", "precision@5", "precision@10", "precision@20", "recall@10", "recall@20") + tuple(extra)
    head = "| method | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|"
    main = [m for m in result["methods"] if m.split("@")[0] not in CONTROLS]
    ctrl = [m for m in result["methods"] if m.split("@")[0] in CONTROLS]
    lines = [f"## {title}", "",
             f"Ground truth: a relation replicates if its realised held-out gain (pooled over folds, 95 % patient "
             f"bootstrap) excludes zero. Universe = union of every compared method's registered sets "
             f"(mean {result.get('universe_size_mean', 0):.0f} per target; replicating share "
             f"{f(result.get('base_rate'))}). Unregistered sets rank below a method's list, ties broken at random "
             f"(20 seeds). Intervals: 95 % bootstrap over targets.", "", head]
    for m in main:
        lines.append(f"| {display(m)} | " + " | ".join(f(result["methods"][m].get(c)) for c in cols) + " |")
    if ctrl:
        lines += ["", "Statistical controls (supplementary):", "", head]
        for m in ctrl:
            lines.append(f"| {display(m)} | " + " | ".join(f(result["methods"][m].get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ controlled worlds
def worlds_table(q1: Path, B: str = "nuis", workers: int = 12, methods=None) -> dict:
    from .dev_metric import load_dev_study
    from .worlds import world_task

    q1 = Path(q1)
    by_world: dict = {}
    for p in sorted((q1 / "registry").glob("*.json")):
        reg = json.loads(p.read_text())
        for e in reg["entries"]:
            if e["condition"] == "real" and e["B"] == B and (methods is None or e["method"] in methods):
                by_world.setdefault(e["world"], {})[e["method"]] = [tuple(sorted(s)) for s in e["sets"]]
    rows = {}
    for p in sorted(q1.glob("rows_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    rows.setdefault((r["world"], r["method"]), r)
    by_world = {w: d for w, d in by_world.items() if all((w, m) in rows for m in d)}
    study = load_dev_study()
    jobs, keys, universe, tasks = [], [], {}, {}
    for w, d in by_world.items():
        fam, f, s = w.split("|")
        task, names, ep, *_ = world_task(study, fam, float(f[1:]), int(s[1:]))
        index = {n: i for i, n in enumerate(names)}
        universe[w] = sorted({x for sets in d.values() for x in sets})
        jobs.append((task, [tuple(index[n] for n in x) for x in universe[w]], B))
        keys.append(w)
    core._G["label"] = jobs
    res = core._pool_map(_label_job, len(jobs), workers)
    core._G.clear()
    per = {}
    for w, r in zip(keys, res):
        lab = dict(zip(universe[w], map(bool, labels_pooled([r]))))
        for m, sets in by_world[w].items():
            mm = ranking_metrics(sets, universe[w], lab)
            row = rows[(w, m)]
            mm.update({f"hit@{k}": float(bool(row.get(f"recovered@{k}"))) for k in (1, 5, 10, 20)})
            mm["mrr"] = row.get("reciprocal_rank")
            mm["constituent_recall@10"] = row.get("constituent_recall@10")
            mm["heldout_family"] = row.get("heldout_family")
            per.setdefault(m, []).append(mm)
    out = {"B": B, "n_worlds": len(by_world), "methods": {}}
    for m, L in per.items():
        out["methods"][m] = {k: boot_mean([x[k] for x in L]) for k in L[0] if k != "heldout_family"}
        out["methods"][m]["n_worlds"] = (len(L), None, None)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("dev", "worlds"))
    ap.add_argument("--B", default="nuis")
    ap.add_argument("--methods", nargs="*", default=[], help="dev: name=registry_dir ...")
    ap.add_argument("--out", default="runs/development/benchmark_v03/public")
    ap.add_argument("--q1", default="runs/development/benchmark_v03/Q1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--list-B", default=None, help="dev: use the lists searched under this baseline (default --B)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.mode == "dev":
        res = dev_table(dict(m.split("=", 1) for m in args.methods), B=args.B, workers=args.workers, out=out,
                        list_B=args.list_B)
        print(table_md(res, f"Development targets, labels at B_{args.B}"))
    else:
        res = worlds_table(Path(args.q1), B=args.B, workers=args.workers)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"worlds_public_{args.B}.json").write_text(json.dumps(res, indent=1))
        text = table_md(res, f"Controlled worlds ({res['n_worlds']} worlds with every listed method), B_{args.B}",
                        extra=("hit@1", "hit@10", "mrr", "constituent_recall@10"))
        (out / f"worlds_public_{args.B}.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
