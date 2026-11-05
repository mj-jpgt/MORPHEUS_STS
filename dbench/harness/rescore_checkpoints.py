"""Re-score stored development-metric registries at other baselines (comparison tables only).

Reads ``registry.json`` of a dev_metric evaluation (a training checkpoint's
``<run>/harness/stepNNNNNN/`` or a baseline's ``dev_metric_<method>/step000000/``), rebuilds the
identical tasks (same targets, folds, seeds, null versions), and scores the SAME registered sets at
B_nuis (reproduces the stored selection metric, as a check), B_known and B_decomp with the same pooled
patient bootstrap and nulls.  Writes ``rescore.json`` beside the registry: net yield at k = 1, 5, 10,
20 per baseline and the k = 10 ceiling (share of the real top 10 that replicate).  The selection
metric stays B_nuis k = 10 (spec 3.4).

    python -m dbench.harness.rescore_checkpoints runs/development/learned_search_v03/full/harness/step002000 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

BS = ("nuis", "known", "decomp")


def rescore(reg_dir: Path, *, workers: int = 10, targets_path=None, bs=BS, device=None) -> dict:
    from .dev_metric import TARGETS, load_dev_study, load_targets

    spec = load_targets(targets_path or TARGETS)
    study = load_dev_study(spec["study"], spec["partitions"])
    folds = core.patient_folds(study.patients, spec["n_folds"], spec["seed"])
    reg = json.loads((Path(reg_dir) / "registry.json").read_text())
    if core.registry_hash(reg["entries"]) != reg["sha256"]:
        raise ValueError(f"{reg_dir}: registry does not match its sha256")
    built = core._tasks(study, spec["targets"], folds, tuple(spec["null_kinds"]), spec["n_null"], spec["seed"])
    tasks = {(t, f, cond): (tk, names, st, q) for t, f, conds, names, st, q in built for cond, tk in conds}
    jobs, meta = [], []
    for e in reg["entries"]:
        tk, names, st, q = tasks[(e["target"], e["fold"], e["condition"])]
        index = {n: i for i, n in enumerate(names)}
        top = [tuple(index[n] for n in s) for s in e["sets"][:max(core.KS)]]
        if top:
            jobs.append((tk, top, list(bs)))
            meta.append((e, st, q))
    if device is not None and str(device).startswith("cuda"):   # one process on the GPU
        scores = [core.score_sets(tk, top, Bs=b, device=device) for tk, top, b in jobs]
    else:
        core._G["score"] = jobs
        scores = core._pool_map(core._score_job, len(jobs), workers)
        core._G.clear()
    results = [core.FoldResult(e["target"], e["fold"], e["condition"], e["sets"][:max(core.KS)],
                               e["pred"][:max(core.KS)], sc, [f"{st}:{x}" for x in q], (e.get("pred_sd") or [None] * len(e["sets"]))[:max(core.KS)])
               for (e, st, q), sc in zip(meta, scores)]
    ny = core.net_yields(results, Bs=tuple(bs))
    out = {"registry": str(Path(reg_dir) / "registry.json"), "registry_sha256": reg["sha256"],
           "targets_sha256": spec["sha256"], "net_yield": {B: {str(k): d for k, d in ny[B].items()} for B in bs},
           "ceiling_k10": {B: ny[B][10]["real"] / 10 for B in bs},
           "calibration_nuis": core.calibration(results, "nuis")}
    stored = Path(reg_dir) / "dev_metric.json"
    if stored.exists():
        out["stored_net_yield_k10_Bnuis"] = json.loads(stored.read_text()).get("net_yield_k10_Bnuis")
    (Path(reg_dir) / f"rescore_{'_'.join(bs)}.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--bs", default=",".join(BS), help="baselines to score (B_decomp is the slow one on CPU)")
    ap.add_argument("--device", default=None, help="cuda: score in this process on the GPU")
    args = ap.parse_args(argv)
    bs = tuple(args.bs.split(","))
    for d in args.dirs:
        r = rescore(Path(d), workers=args.workers, bs=bs, device=args.device)
        line = {B: {k: round(r["net_yield"][B][k]["net"], 2) for k in ("1", "5", "10", "20")} for B in bs}
        print(d, json.dumps(line), "ceiling@10", {B: round(v, 3) for B, v in r["ceiling_k10"].items()},
              "stored", r.get("stored_net_yield_k10_Bnuis"), flush=True)


if __name__ == "__main__":
    main()
