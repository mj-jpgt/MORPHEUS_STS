"""Final-ranking variants of stored checkpoint registries (no retraining).

For a development registry (real targets and both nulls, lists searched under B_nuis), every registered
set (all K, not only the top 20) is scored on its fold's query rows at B_nuis and B_known; the score also
returns its support LOPO gain G_sup.  The same sets are then re-ordered by
  support   : G_sup
  mu        : predicted gain G_sup - mu_delta (the registry's ``pred``; kappa = 0)
  lcb0.5    : pred - 0.5 sd
  lcb1      : pred - 1.0 sd (the search's own final rank)
and each ordering gets precision@10 and net yield@10 (real minus mean null, the harness's pooled
patient-bootstrap rule) and average precision of the list against the same replication labels.

    python -m dbench.harness.rerank_variants runs/development/learned_search_v03/full/harness/step010000 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

VARIANTS = ("support", "mu", "lcb0.5", "lcb1")
BS = ("nuis", "known")


def _score_all(i):
    import torch
    torch.set_num_threads(1)
    tk, sets = core._G["score"][i]
    return core.score_sets(tk, sets, Bs=BS, device=None)


def order(variant: str, gsup, pred, sd) -> np.ndarray:
    gsup, pred = np.asarray(gsup, float), np.asarray([np.nan if p is None else p for p in pred], float)
    sd = np.asarray([0.0 if s is None else s for s in sd], float)
    key = {"support": gsup, "mu": pred, "lcb0.5": pred - 0.5 * sd, "lcb1": pred - sd}[variant]
    key = np.where(np.isfinite(key), key, -np.inf)
    return np.argsort(-key, kind="stable")


def average_precision(labels: list) -> float | None:
    y = np.asarray(labels, float)
    if y.sum() == 0:
        return None
    hits = np.cumsum(y)
    return float(np.sum(y * hits / np.arange(1, len(y) + 1)) / y.sum())


def evaluate(reg_dir: Path, workers: int = 12) -> dict:
    from .dev_metric import TARGETS, load_dev_study, load_targets

    spec = load_targets(TARGETS)
    study = load_dev_study(spec["study"], spec["partitions"])
    folds = core.patient_folds(study.patients, spec["n_folds"], spec["seed"])
    reg = json.loads((Path(reg_dir) / "registry.json").read_text())
    built = core._tasks(study, spec["targets"], folds, tuple(spec["null_kinds"]), spec["n_null"], spec["seed"])
    tasks = {(t, f, c): (tk, names, st, q) for t, f, conds, names, st, q in built for c, tk in conds}
    entries = [e for e in reg["entries"] if e["sets"]]
    jobs = []
    for e in entries:
        tk, names, st, q = tasks[(e["target"], e["fold"], e["condition"])]
        index = {n: i for i, n in enumerate(names)}
        jobs.append((tk, [tuple(index[n] for n in s) for s in e["sets"]]))
    core._G["score"] = jobs
    scores = core._pool_map(_score_all, len(jobs), workers)
    core._G.clear()
    out = {}
    for v in VARIANTS:
        results = []
        for e, sc in zip(entries, scores):
            o = order(v, sc["nuis"][2], e["pred"], e.get("pred_sd") or [None] * len(e["sets"]))
            top = o[:max(core.KS)]
            re = {B: (sc[B][0][:, top], sc[B][1], sc[B][2][top]) for B in BS}
            results.append(core.FoldResult(e["target"], e["fold"], e["condition"], [e["sets"][j] for j in top],
                                           [e["pred"][j] for j in top], re, []))
        ny = core.net_yields(results, ks=(10,), Bs=BS)
        # AP of the full re-ordered list against the pooled replication labels (real targets)
        aps = {}
        for B in BS:
            pool: dict = {}
            for e, sc in zip(entries, scores):
                if e["condition"] != "real":
                    continue
                for j, s in enumerate(e["sets"]):
                    pool.setdefault((e["target"], tuple(sorted(s))), ([], []))
                    pool[(e["target"], tuple(sorted(s)))][0].append(sc[B][0][:, j])
                    pool[(e["target"], tuple(sorted(s)))][1].append(sc[B][1])
            lower = {k: core.pooled_interval(np.concatenate(a), np.concatenate(b), 500)[1] for k, (a, b) in pool.items()}
            vals = []
            for e, sc in zip(entries, scores):
                if e["condition"] != "real":
                    continue
                o = order(v, sc["nuis"][2], e["pred"], e.get("pred_sd") or [None] * len(e["sets"]))
                vals.append(average_precision([lower[(e["target"], tuple(sorted(e["sets"][j])))] > 0 for j in o]))
            aps[B] = float(np.mean([x for x in vals if x is not None]))
        out[v] = {B: {"precision@10": ny[B][10]["real"] / 10, "net_yield@10": ny[B][10]["net"], "ap": aps[B]}
                  for B in BS}
    res = {"registry": str(reg_dir), "registry_sha256": reg["sha256"], "variants": out}
    (Path(reg_dir) / "rerank_variants.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    for d in args.dirs:
        r = evaluate(Path(d), args.workers)
        for v, dd in r["variants"].items():
            print(d.rstrip("/").split("/")[-3], d.rstrip("/").split("/")[-1], v,
                  " ".join(f"{B}: P@10={x['precision@10']:.3f} AP={x['ap']:.3f} net@10={x['net_yield@10']:.2f}"
                           for B, x in dd.items()), flush=True)


if __name__ == "__main__":
    main()
