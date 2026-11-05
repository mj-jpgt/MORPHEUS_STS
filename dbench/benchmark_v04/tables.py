"""Derive binary labels from the raw label cache (labels_v01 + amendments 1, 2) and report prevalences.

    python -m dbench.benchmark_v04.tables prevalence STUDY
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BENCH = Path.home() / "morpheus_work_bench"
ROOT = BENCH / "runs/development/benchmark_v04"
LEVELS = ("95", "90", "99")


def cache_dir(study):
    return ROOT / "labels" / study / "cache"


def load_target(study, target):
    h = hashlib.sha256(target.encode()).hexdigest()[:10]
    d = cache_dir(study)
    if not (d / f"{h}.npz").exists():
        return None
    return json.loads((d / f"{h}.json").read_text()), np.load(d / f"{h}.npz")


def label(z, split, kind, B="known", level="95", guard=True):
    """Binary label (bool array): lower bound(level) > 0 and G > null p95; B_decomp pairs also need G_known > 0 when
    ``guard`` (amendment 2)."""
    G = z[f"s{split}|{kind}|{B}|G"]
    lb = z[f"s{split}|{kind}|{B}|lb{level}"]
    thr = float(z[f"s{split}|{kind}|{B}|null_p95"])
    lab = (lb > 0) & (G > thr)
    if B == "decomp" and guard:
        lab &= z[f"s{split}|pair|known|G"] > 0
    return lab


def comparator_failure(z, split):
    """Share of pool pairs whose best-decomposition comparator has query R^2 < -1 : (G_nuis - G_decomp) < -1."""
    return float(np.mean((z[f"s{split}|pair|nuis|G"] - z[f"s{split}|pair|decomp|G"]) < -1.0))


def slow_subset(study, target, n=3, total=10):
    """Fixed seeded subset of splits on which the slow baselines are run (declared before the probe was read)."""
    rng = np.random.default_rng(int(hashlib.sha256(f"bench-slow-subset|{study}|{target}".encode()).hexdigest()[:8], 16))
    return sorted(int(x) for x in rng.choice(total, n, replace=False))


def splits_of(meta):
    return [s["split"] for s in meta["splits"]]


def prevalence(study):
    targets = json.loads((ROOT / "labels" / study / "targets.json").read_text())["targets"]
    rows = {}
    per_target = {}
    cf = []
    n_t = 0
    for t in targets:
        r = load_target(study, t)
        if r is None:
            continue
        meta, z = r
        n_t += 1
        ss = splits_of(meta)
        for kind, Bs in (("single", ("known", "nuis")), ("pair", ("known", "nuis", "decomp"))):
            for B in Bs:
                for lv in LEVELS:
                    guards = (True, False) if B == "decomp" else (True,)
                    for g in guards:
                        key = f"{kind}|{B}|{lv}" + ("" if B != "decomp" else ("|guarded" if g else "|unguarded"))
                        v = float(np.mean([label(z, s, kind, B, lv, g).mean() for s in ss]))
                        rows.setdefault(key, []).append(v)
        cf.append(float(np.mean([comparator_failure(z, s) for s in ss])))
        per_target[t] = {"single_known_95": rows["single|known|95"][-1], "pair_known_95": rows["pair|known|95"][-1]}
    out = {"study": study, "n_targets": n_t, "mean_prevalence": {k: float(np.mean(v)) for k, v in rows.items()},
           "min_max_known95": {k: [float(np.min(rows[k])), float(np.max(rows[k]))] for k in ("single|known|95", "pair|known|95")},
           "targets_with_no_positive_single_known95": int(sum(v == 0 for v in rows["single|known|95"])),
           "targets_with_no_positive_pair_known95": int(sum(v == 0 for v in rows["pair|known|95"])),
           "comparator_failure_rate": {"mean": float(np.mean(cf)), "max": float(np.max(cf)), "share_targets_gt_5pct": float(np.mean(np.array(cf) > 0.05))},
           "per_target": per_target}
    (ROOT / f"prevalence_{study}.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prevalence",))
    ap.add_argument("study")
    a = ap.parse_args(argv)
    o = prevalence(a.study)
    print(json.dumps({k: v for k, v in o.items() if k != "per_target"}, indent=1))


if __name__ == "__main__":
    main()
