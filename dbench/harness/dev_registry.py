"""Registries of any method on the 40 development targets (real targets only, searched under B_nuis and
B_known), for the ranking metrics of ``public_metrics`` (labels are method-independent, so no nulls).

    python -m dbench.harness.dev_registry --methods linear_screen,random,rulefit,ebm,gbm_h --workers 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import core

OUT = Path("runs/development/benchmark_v03/dev_registry")


def main(argv=None) -> None:
    from .baselines import EXTRA_METHODS
    from .dev_metric import TARGETS, load_dev_study, load_targets

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    spec = load_targets(TARGETS)
    study = load_dev_study(spec["study"], spec["partitions"])
    table = EXTRA_METHODS(device=args.device)
    for m in args.methods.split(","):
        od = Path(args.out) / m
        if (od / "registry.json").exists():
            print(m, "exists", flush=True)
            continue
        fn = table[m]()
        fn.parallel_ok = True    # CPU search functions; each core-eight call isolates itself in a forked child
        res = core.run(study, spec["targets"], fn, n_folds=spec["n_folds"], K=spec["K"], Bs=("nuis", "known"),
                       search_Bs=("nuis", "known"), null_kinds=(), n_null=0, seed=spec["seed"], score_top=0,
                       out_dir=od, workers=args.workers, log=lambda *a: print(m, *a, flush=True))
        (od / "meta.json").write_text(json.dumps({"method": m, "registry_sha256": res["registry_sha256"],
                                                  "targets_sha256": spec["sha256"], "timing": res["timing"]},
                                                 indent=1, default=float))
        print(m, "done", res["timing"], flush=True)


if __name__ == "__main__":
    main()
