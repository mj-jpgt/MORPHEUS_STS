"""End-to-end K = 64 learned relation search vs support-fit screen on the held-out organ (bladder_epithelial): B_nuis
searches on the same beam, null targets (patient shuffle, scramble), B_decomp scored on each method's B_nuis top 20
(core.run); per-target net yield, precision@k and null false-discovery fraction with paired bootstrap.

    python -m dbench.harness.bladder_decomp --device cuda --workers 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core

OUT = Path("runs/development/benchmark_v03/bladder_decomp")
CHECKPOINT = "runs/development/learned_search_v03/release/frozen.pt"


def main(argv=None) -> None:
    from .baselines import model_search, screen_search
    from .dev_metric import choose_targets
    from .transfer import load_study
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", default="bladder_epithelial")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args(argv)
    study = load_study(a.study, "meta_training")
    targets = choose_targets(study, n=a.n_targets, seed=a.seed)
    out = OUT / a.study
    out.mkdir(parents=True, exist_ok=True)
    per_target, summary = {}, {}
    for name, search in (("g_theta", model_search(CHECKPOINT, device=a.device)), ("screen", screen_search(device=a.device))):
        res = core.run(study, targets, search, n_folds=3, K=64, Bs=("nuis", "decomp"), search_Bs=("nuis",),
                       null_kinds=core.NULL_KINDS, n_null=1, seed=a.seed, score_top=max(core.KS), out_dir=out / name,
                       device=a.device, workers=a.workers)
        cells = core.yields(res["results"], Bs=("decomp",), n_boot=core.N_BOOT)
        summary[name] = {b: {str(k): v for k, v in c.items()} for b, c in res["net_yield"].items()}
        per_target[name] = {}
        for t in targets:
            for k in core.KS:
                real = cells.get(("real", t, "decomp", k), float("nan"))
                null = float(np.nanmean([cells.get((f"null:{kind}", t, "decomp", k), float("nan")) for kind in core.NULL_KINDS]))
                per_target[name].setdefault(t, {})[f"decomp@{k}"] = {"real": float(real), "null": null, "net": float(real - null)}
    rng = np.random.default_rng(20261026)

    def boot(v):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        return [float(v.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    ts = [t for t in targets if t in per_target["g_theta"] and t in per_target["screen"]]
    paired = {}
    for k in core.KS:
        g = {q: [per_target["g_theta"][t][f"decomp@{k}"][q] for t in ts] for q in ("real", "null", "net")}
        s = {q: [per_target["screen"][t][f"decomp@{k}"][q] for t in ts] for q in ("real", "null", "net")}
        paired[str(k)] = {"net_diff": boot(np.subtract(g["net"], s["net"])),
                          "precision": {"g_theta": boot(np.divide(g["real"], k)), "screen": boot(np.divide(s["real"], k)),
                                        "diff": boot(np.divide(np.subtract(g["real"], s["real"]), k))},
                          "null_fdr": {"g_theta": boot(np.divide(g["null"], k)), "screen": boot(np.divide(s["null"], k)),
                                       "diff": boot(np.divide(np.subtract(g["null"], s["null"]), k))}}
    (out / "BLADDER_DECOMP.json").write_text(json.dumps({"study": a.study, "targets": targets, "net_yield": summary,
                                                         "per_target": per_target, "paired": paired}, indent=1, default=float))
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
