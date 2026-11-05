"""Exact-budget end-to-end test of released neural proposer versus support screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import core
from .baselines import model_search, screen_search
from .dev_metric import K, N_FOLDS, load_dev_study, load_targets

OUT = Path("runs/development/benchmark_v03/proposer_repair_v01")
CHECKPOINT = Path("runs/development/learned_search_v03/release/frozen.pt")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sets(rows):
    return [list(s) for s, *_ in rows]


def contracts(device: str | None, out: Path) -> dict:
    """Verify deterministic support-only admission and that query values cannot affect it."""
    spec = load_targets()
    study = load_dev_study(spec["study"], spec["partitions"])
    target, query = spec["targets"][0], spec["folds"][0]
    support = [str(p) for p in study.patients if str(p) not in set(map(str, query))]
    task, _, _ = core.build_task(study, target, support, query, seed=spec["seed"])
    st = core.support_only(task)
    y = np.asarray(task.y).copy()
    y[task.s:] = np.random.default_rng(20261023).permutation(y[task.s:])
    query_changed = core.support_only(replace(task, y=y))
    methods = {"g_theta": model_search(str(CHECKPOINT), device=device), "screen": screen_search(device=device)}
    checks = {}
    for name, search in methods.items():
        a, b = list(search(st, "nuis", K)), list(search(st, "nuis", K))
        q = list(search(query_changed, "nuis", K))
        A, B, Q = _sets(a), _sets(b), _sets(q)
        checks[name] = {"n_sets": len(A), "deterministic": A == B, "query_isolated": A == Q,
                        "budget_respected": len(A) <= K,
                        "indices_valid": all(all(0 <= int(i) < st.X.shape[1] for i in s) for s in A)}
    passed = all(all(v.values()) for v in checks.values())
    result = {"contract_version": "proposer_repair_v01", "checkpoint": str(CHECKPOINT),
              "checkpoint_sha256": _hash(CHECKPOINT), "targets_sha256": spec["sha256"], "target": target,
              "fold": 0, "K": K, "checks": checks, "passed": passed}
    out.mkdir(parents=True, exist_ok=True)
    (out / "CONTRACTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("proposer contract failed; refusing comparison")
    return result


def _summary(result: dict) -> dict:
    return {"registry_sha256": result["registry_sha256"], "seconds": round(float(result["seconds"]), 2),
            "timing": result["timing"], "calibration": result.get("calibration"),
            "net_yield": {b: {str(k): v for k, v in cells.items()} for b, cells in result["net_yield"].items()}}


def _per_target_net(result: dict) -> dict:
    """Patient-fold query yield per target; preserves the unit for paired inference."""
    cells = core.yields(result["results"], Bs=("nuis", "known"), n_boot=core.N_BOOT)
    targets = sorted({r.target for r in result["results"]})
    out = {}
    for target in targets:
        out[target] = {}
        for baseline in ("nuis", "known"):
            for k in core.KS:
                real = cells.get(("real", target, baseline, k), float("nan"))
                nulls = [cells.get((f"null:{kind}", target, baseline, k), float("nan")) for kind in core.NULL_KINDS]
                out[target][f"{baseline}@{k}"] = float(real - np.nanmean(nulls))
    return out


def _paired_delta(left: dict, right: dict, n_boot: int = 10000) -> dict:
    common = sorted(set(left).intersection(right))
    rng, out = np.random.default_rng(20261023), {}
    for metric in (f"{b}@{k}" for b in ("nuis", "known") for k in core.KS):
        d = np.asarray([left[t][metric] - right[t][metric] for t in common], float)
        if not len(d):
            continue
        boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        out[metric] = {"mean": float(d.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                       "n_targets": int(len(d))}
    return out


def compare(device: str | None, out: Path, n_targets: int, workers: int) -> dict:
    c = out / "CONTRACTS.json"
    if not c.exists() or not json.loads(c.read_text()).get("passed"):
        raise SystemExit("run --contracts successfully before --compare")
    spec, study = load_targets(), load_dev_study()
    targets = list(spec["targets"][:int(n_targets)])
    methods = {"g_theta": model_search(str(CHECKPOINT), device=device), "screen": screen_search(device=device)}
    output = {"comparison_version": "proposer_repair_v01", "checkpoint_sha256": _hash(CHECKPOINT),
              "targets_sha256": spec["sha256"], "n_targets": len(targets), "n_folds": N_FOLDS, "K": K,
              "relation_fits_per_search_max": K,
              "candidate_grammar": "shared search_v03 beam; support-only input; fixed relation backend",
              "methods": {}}
    per_target = {}
    for name, search in methods.items():
        result = core.run(study, targets, search, n_folds=N_FOLDS, K=K, Bs=("nuis", "known", "decomp"),
                          search_Bs=("nuis", "known"), null_kinds=core.NULL_KINDS, n_null=1,
                          seed=spec["seed"], score_top=max(core.KS), out_dir=out / name, device=device, workers=workers)
        output["methods"][name] = _summary(result)
        per_target[name] = _per_target_net(result)
        output["methods"][name]["per_target_net_yield"] = per_target[name]
        (out / f"{name}.json").write_text(json.dumps(output["methods"][name], indent=2, sort_keys=True) + "\n")
    output["paired_gtheta_minus_screen"] = _paired_delta(per_target["g_theta"], per_target["screen"])
    (out / "COMPARISON.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contracts", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if not (a.contracts or a.compare):
        ap.error("choose --contracts and/or --compare")
    out, device = Path(a.out), (None if a.device in ("", "none", "cpu") else a.device)
    if a.contracts:
        print(json.dumps(contracts(device, out), indent=2))
    if a.compare:
        print(json.dumps(compare(device, out, a.targets, a.workers), indent=2))


if __name__ == "__main__":
    main()
