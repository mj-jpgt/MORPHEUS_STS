"""Transfer evaluation (governing spec 3.6, Q3a) and any real-target cohort run through the harness.

For each named study (the withheld organ, ``transfer_holdouts.json``: bladder_batf3 and
bladder_epithelial; platform transfer cohorts when frozen), every method searches a fixed set of
real targets (half morphology similarities, half molecular targets, drawn once with the seed) on
patient-grouped folds; registry frozen before scoring; nulls by the same method.  Reported: net
yield at k in {1, 5, 10, 20} under B_nuis / B_known / B_decomp, calibration Spearman / ECE /
coverage ECE, and the mean gap between predicted and realised gain of the scored sets.

    python -m dbench.harness.transfer --studies bladder_batf3,bladder_epithelial --role meta_training \\
        --methods screen,lopo_heuristic,stat_gbm --checkpoint <frozen g_theta> --out runs/development/benchmark_v03/transfer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

OUT = Path("runs/development/benchmark_v03/transfer")


def load_study(name: str, role: str, partitions: str = "dbench/protocols/sprint_partitions_v01.json"):
    from morpheus_encoder.learned_search.episodes_v03 import studies_v03
    out, info = studies_v03(partitions, roles=(role,), only={name})
    if not out[role]:
        raise RuntimeError(f"{name} did not load: {info}")
    return out[role][0]


def gap(results: list, B: str = "nuis") -> float | None:
    d = []
    for r in results:
        if r.condition != "real" or B not in r.scores:
            continue
        num, sst, _ = r.scores[B]
        g = np.nansum(num, 0) / max(float(np.sum(sst)), 1e-12)
        d += [p - g[j] for j, p in enumerate(r.pred) if p is not None and np.isfinite(g[j])]
    return float(np.mean(d)) if d else None


def run_study(study, methods: dict, *, n_targets: int, n_folds: int, K: int, seed: int, out: Path, workers: int) -> dict:
    from .dev_metric import choose_targets
    n_folds = max(2, min(n_folds, len(study.patients) // 2))
    targets = choose_targets(study, n=n_targets, seed=seed)
    rows = {}
    for name, search in methods.items():
        res = core.run(study, targets, search, n_folds=n_folds, K=K, Bs=core.BASELINES, search_Bs=("nuis", "known"),
                       seed=seed, out_dir=out / study.name / name, workers=workers)
        rows[name] = {"net_yield": res["net_yield"], "calibration": res["calibration"], "gap": gap(res["results"]),
                      "registry_sha256": res["registry_sha256"], "n_targets": len(targets), "n_folds": n_folds,
                      "seconds": round(res["seconds"], 1)}
    return {"study": study.name, "organ": study.organ, "platform": study.platform, "targets": targets, "methods": rows}


def write_table(out: Path, studies: list) -> str:
    f = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
    lines = ["# Transfer evaluation (spec 3.6)", ""]
    for s in studies:
        lines += [f"## {s['study']} ({s['organ']}, {s['platform']})", "",
                  "| method | net@10 nuis | net@10 known | net@10 decomp | calibration Spearman | ECE | coverage ECE | "
                  "predicted - realised |", "|---|---|---|---|---|---|---|---|"]
        for m, d in s["methods"].items():
            ny = lambda B: (d["net_yield"].get(B) or {}).get(10, (d["net_yield"].get(B) or {}).get("10", {})).get("net")  # noqa: E731
            c = d["calibration"] or {}
            lines.append(f"| {m} | {f(ny('nuis'))} | {f(ny('known'))} | {f(ny('decomp'))} | {f(c.get('spearman'), 3)} | "
                         f"{f(c.get('ece'), 4)} | {f(c.get('coverage_ece'), 3)} | {f(d['gap'], 4)} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "TRANSFER.md").write_text(text)
    (out / "transfer.json").write_text(json.dumps(studies, indent=1, default=str))
    return text


def main(argv=None) -> None:
    from .baselines import EXTRA_METHODS, model_search

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--studies", required=True)
    ap.add_argument("--role", default="meta_training")
    ap.add_argument("--methods", default="screen,lopo_heuristic,stat_gbm")
    ap.add_argument("--checkpoint", default=None, help="frozen g_theta, run as 'g_theta'")
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    table = EXTRA_METHODS(device=args.device)
    methods = {m: table[m]() for m in args.methods.split(",") if m}
    if args.checkpoint:
        methods["g_theta"] = model_search(args.checkpoint, device=args.device)
    out = Path(args.out)
    studies = [run_study(load_study(s, args.role), methods, n_targets=args.n_targets, n_folds=args.n_folds, K=args.K,
                         seed=args.seed, out=out, workers=args.workers) for s in args.studies.split(",")]
    print(write_table(out, studies))


if __name__ == "__main__":
    main()
