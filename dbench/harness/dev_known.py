"""Development targets searched under B_known (spec 3.2: the list registered under B is scored at B).

Same 40 targets, folds, seeds and nulls as ``dev_metric``; the method searches with B = "known" and its
list is scored at B_known.  Writes ``<out>/<method>/dev_known.json`` and the frozen registry.

    python -m dbench.harness.dev_known --methods screen,lopo_heuristic,stat_gbm,g_theta --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import core

OUT = Path("runs/development/benchmark_v03/dev_known")


def main(argv=None) -> None:
    from .baselines import EXTRA_METHODS
    from .dev_metric import TARGETS, load_dev_study, load_targets

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", default="screen,lopo_heuristic,stat_gbm")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    spec = load_targets(TARGETS)
    study = load_dev_study(spec["study"], spec["partitions"])
    table = EXTRA_METHODS(device=args.device)
    for m in args.methods.split(","):
        od = Path(args.out) / m
        res = core.run(study, spec["targets"], table[m](), n_folds=spec["n_folds"], K=spec["K"], Bs=("known",),
                       search_Bs=("known",), null_kinds=tuple(spec["null_kinds"]), n_null=spec["n_null"],
                       seed=spec["seed"], out_dir=od, workers=args.workers, log=lambda *a: print(*a, flush=True))
        ny = res["net_yield"]["known"]
        out = {"method": m, "net_yield_Bknown": {str(k): d for k, d in ny.items()},
               "ceiling_k10": ny[10]["real"] / 10, "calibration_known": core.calibration(res["results"], "known"),
               "targets_sha256": spec["sha256"], "registry_sha256": res["registry_sha256"],
               "timing": res["timing"]}
        (od / "dev_known.json").write_text(json.dumps(out, indent=1, default=float))
        print(m, json.dumps({k: round(d["net"], 2) for k, d in out["net_yield_Bknown"].items()}),
              "ceiling@10", round(out["ceiling_k10"], 3), flush=True)


if __name__ == "__main__":
    main()
