"""Paired g_theta - screen B_decomp net yield on the 40 development targets from the proposer_repair_v01 registries
(end-to-end K = 64, same beam). B_decomp is scored on each method's B_nuis-searched top 20, exactly as core.run does;
the scores were not persisted by the comparison run, so they are recomputed from the frozen registries (no new search).

    python -m dbench.harness.decomp_paired --workers 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

D = Path("runs/development/benchmark_v03/proposer_repair_v01")


def main(argv=None) -> None:
    from .dev_metric import load_dev_study, load_targets
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    spec, study = load_targets(), load_dev_study()
    targets = list(spec["targets"][:40])
    folds = core.patient_folds(study.patients, 3, spec["seed"])
    built = core._tasks(study, targets, folds, core.NULL_KINDS, 1, spec["seed"])
    tasks = {(t, int(f), c): (tk, names, st, q) for t, f, conds, names, st, q in built for c, tk in conds}
    per_target, summary = {}, {}
    for m in ("g_theta", "screen"):
        reg = json.loads((D / m / "registry.json").read_text())
        assert core.registry_hash(reg["entries"]) == reg["sha256"], f"{m}: registry sha mismatch"
        entries = [e for e in reg["entries"] if e["B"] == "nuis"]
        jobs, keep = [], []
        for e in entries:
            tk, names, st, q = tasks[(e["target"], int(e["fold"]), e["condition"])]
            idx = {n: i for i, n in enumerate(names)}
            top = [tuple(idx[n] for n in s) for s in e["sets"][:max(core.KS)]]
            if top:
                jobs.append((tk, top, ["decomp"]))
                keep.append((e, names, st, q))
        core._G["score"] = jobs
        scores = core._pool_map(core._score_job, len(jobs), a.workers)
        core._G.clear()
        results = [core.FoldResult(e["target"], e["fold"], e["condition"], e["sets"][:max(core.KS)],
                                   e["pred"][:max(core.KS)], sc, [f"{st}:{qq}" for qq in q], e["pred_sd"][:max(core.KS)])
                   for (e, names, st, q), sc in zip(keep, scores)]
        cells = core.yields(results, Bs=("decomp",), n_boot=core.N_BOOT)
        summary[m] = core.net_yields(results, Bs=("decomp",), null_kinds=core.NULL_KINDS)
        per_target[m] = {}
        for t in targets:
            for k in core.KS:
                real = cells.get(("real", t, "decomp", k), float("nan"))
                nulls = [cells.get((f"null:{kind}", t, "decomp", k), float("nan")) for kind in core.NULL_KINDS]
                per_target[m].setdefault(t, {})[f"decomp@{k}"] = {"real": float(real), "null": float(np.nanmean(nulls)),
                                                                  "net": float(real - np.nanmean(nulls))}
    rng, paired = np.random.default_rng(20261023), {}
    for k in core.KS:
        d = np.array([per_target["g_theta"][t][f"decomp@{k}"]["net"] - per_target["screen"][t][f"decomp@{k}"]["net"] for t in targets])
        boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        paired[f"decomp@{k}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                                 "n_targets": len(d)}
    out = {"source": "proposer_repair_v01 registries (sha-checked), B_decomp rescored on the B_nuis top 20",
           "paired_gtheta_minus_screen": paired, "net_yield": {m: {b: {str(k): v for k, v in c.items()} for b, c in s.items()}
                                                               for m, s in summary.items()},
           "per_target": per_target}
    (D / "DECOMP_PAIRED.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
