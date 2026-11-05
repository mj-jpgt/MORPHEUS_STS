"""Transfer evaluation with the full-pool retrieval metrics (spec 3.6; owner format 2026-10-21).

For each held-out study (organ holdout: bladder_batf3, bladder_epithelial; platform transfer cohorts when
loaded), real targets drawn once with the seed, patient-grouped folds, the shared full pool and its
method-independent labels (B_known singles and pairs, B_decomp pairs, B_nuis lenient), and every method's
pool scores from the fold's support rows; AUROC and average precision per method.

    python -m dbench.harness.transfer_pool --studies bladder_batf3,bladder_epithelial --role meta_training \
        --methods g_theta,linear_screen,random,rulefit,ebm,gbm_h,pysr,tabdistill --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import pool

OUT = Path("runs/development/benchmark_v03/transfer_pool")


def main(argv=None) -> None:
    from .dev_metric import choose_targets
    from .transfer import load_study

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--studies", required=True)
    ap.add_argument("--role", default="meta_training")
    ap.add_argument("--methods", default="g_theta,linear_screen,random,rulefit,ebm,gbm_h,pysr,tabdistill")
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    for name in args.studies.split(","):
        study = load_study(name, args.role)
        ctx = pool.study_folds(study, choose_targets(study, n=args.n_targets, seed=args.seed), args.n_folds, args.seed)
        out = OUT / name
        if not (out / "labels.json").exists():      # resumable: labels and each method's scores are kept
            pool.compute_labels(args.workers, None, out, ctx=ctx)
        for m in args.methods.split(","):
            if (out / f"scores_{m}_known.json").exists():
                continue
            dev = args.device if m.startswith(("g_theta", "sel_")) or m == "tabdistill" else None
            # classical lasso-based methods run in this process so each call is isolated in a forked child
            # (scikit-learn's coordinate descent segfaults in this environment on some inputs)
            w = 1 if m in ("linear_screen", "rulefit", "pysr") else args.workers
            pool.compute_scores([m], w, dev, out, ctx=ctx)
        print(pool.table(out, title=f"{name} ({study.organ}, {study.platform}; transfer)"), flush=True)


if __name__ == "__main__":
    main()
