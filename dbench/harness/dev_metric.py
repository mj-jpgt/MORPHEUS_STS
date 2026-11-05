"""Development-study metric for checkpoint selection (governing spec 3.4).

``dev_metric(search_fn, step)`` -> net yield at k = 10 under B_nuis over a FIXED set of 40 real
targets of the development study (kidney_injury), through the harness (patient-grouped folds,
support-only search, frozen registry, both null kinds), plus the calibration Spearman of the
predicted vs realised gain of the scored sets.

``search_fn(task, B, K)``: ``task`` is a support-only ``relation_v03.Task``; returns
``[(index tuple into task.X columns, predicted gain or None, predicted sd or None), ...]`` best
first, at most K fitted.

The 40 targets (20 molecular: residual programmes, programme scores, pool genes; 20 morphology
similarities), the folds and every seed are fixed once in ``TARGETS`` with a sha256:

    python -m dbench.harness.dev_metric --write-targets
    python -m dbench.harness.dev_metric --smoke     # the support-fit screen through the metric
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import core

PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"
DEV_STUDY = "kidney_injury"
TARGETS = Path("runs/development/benchmark_v03/dev_targets_v01.json")
N_TARGETS, N_FOLDS, K, SEED = 40, 3, 64, 20261021
WORKERS = 12   # forked CPU processes for the scoring fits (and CPU searches); the GPU stays with training
_STUDY: dict = {}


def load_dev_study(name: str = DEV_STUDY, partitions: str = PARTITIONS):
    if name not in _STUDY:
        from morpheus_encoder.learned_search.episodes_v03 import studies_v03
        out, info = studies_v03(partitions, roles=("development",), only={name})
        if not out["development"]:
            raise RuntimeError(f"development study {name} did not load: {info}")
        _STUDY[name] = out["development"][0]
    return _STUDY[name]


def choose_targets(study, n: int = N_TARGETS, seed: int = SEED) -> list[str]:
    """Half molecular (uniform over the study's molecular target columns), half morphology."""
    rng = np.random.default_rng(seed)
    mol = sorted(str(study.names[c]) for c in study.target_cols)
    morph = sorted(str(study.names[c]) for c in getattr(study, "morph_targets", []))
    n_morph = min(n // 2, len(morph))
    pick = list(rng.choice(morph, n_morph, replace=False)) if n_morph else []
    pick += list(rng.choice(mol, n - n_morph, replace=False))
    return [str(x) for x in pick]


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in spec.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()


def write_targets(path: Path = TARGETS) -> dict:
    study = load_dev_study()
    spec = {"study": DEV_STUDY, "partitions": PARTITIONS, "targets": choose_targets(study), "n_folds": N_FOLDS,
            "folds": core.patient_folds(study.patients, N_FOLDS, SEED), "K": K, "seed": SEED, "k": 10,
            "baseline": "nuis", "null_kinds": list(core.NULL_KINDS), "n_null": 1,
            "rule": "20 morphology similarities + 20 molecular target columns drawn once with the seed; folds "
                    "patient-grouped with the same seed"}
    spec["sha256"] = spec_hash(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))
    return spec


def load_targets(path: Path = TARGETS) -> dict:
    spec = json.loads(Path(path).read_text())
    if spec_hash(spec) != spec["sha256"]:
        raise ValueError(f"{path} does not match its sha256")
    return spec


def dev_metric(search_fn, step: int, *, device=None, out_dir: str | Path | None = None,
               targets_path: Path = TARGETS, max_targets: int | None = None, workers: int = WORKERS,
               log=lambda *_: None) -> dict:
    """{"net_yield_k10_Bnuis", "calibration_spearman", ...} for one checkpoint (spec 3.4)."""
    spec = load_targets(targets_path)
    study = load_dev_study(spec["study"], spec["partitions"])
    if core.patient_folds(study.patients, spec["n_folds"], spec["seed"]) != spec["folds"]:
        raise ValueError("development study patients differ from the frozen folds")
    targets = spec["targets"][:max_targets]
    od = None if out_dir is None else Path(out_dir) / f"step{int(step):06d}"
    res = core.run(study, targets, search_fn, n_folds=spec["n_folds"], K=spec["K"], Bs=("nuis",),
                   search_Bs=("nuis",), null_kinds=tuple(spec["null_kinds"]), n_null=spec["n_null"],
                   seed=spec["seed"], out_dir=od, device=None, workers=workers, log=log)
    ny = res["net_yield"]["nuis"]
    out = {"step": int(step), "net_yield_k10_Bnuis": ny[10]["net"],
           "calibration_spearman": (res["calibration"] or {}).get("spearman"),
           "calibration_ece": (res["calibration"] or {}).get("ece"),
           "calibration_coverage_ece": (res["calibration"] or {}).get("coverage_ece"),
           "yield_k10_real": ny[10]["real"], **{f"yield_k10_{k}": v for k, v in ny[10].items() if k.startswith("null")},
           "net_yield_Bnuis": {str(k): d["net"] for k, d in ny.items()}, "n_targets": len(targets),
           "targets_sha256": spec["sha256"], "registry_sha256": res["registry_sha256"],
           "seconds": round(res["seconds"], 1),
           "timing": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["timing"].items()}}
    if od is not None:
        (od / "dev_metric.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-targets", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="run --method through dev_metric")
    ap.add_argument("--max-targets", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--calls", type=int, default=1)
    ap.add_argument("--method", default="screen", help="a baselines.EXTRA_METHODS name (screen = baseline 11)")
    ap.add_argument("--out", default="runs/development/benchmark_v03/dev_metric_smoke")
    args = ap.parse_args(argv)
    if args.write_targets:
        spec = write_targets()
        print(json.dumps({k: spec[k] for k in ("study", "n_folds", "sha256")}), len(spec["targets"]), "targets")
    if args.smoke:
        from .baselines import EXTRA_METHODS
        t = time.time()
        fn = EXTRA_METHODS(device=args.device)[args.method]()
        for call in range(args.calls):   # a second call shows the cost with cached tasks
            print(json.dumps(dev_metric(fn, call, device=args.device, out_dir=args.out, max_targets=args.max_targets,
                                        workers=args.workers, log=lambda *a: print(*a, flush=True)), indent=1),
                  f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
