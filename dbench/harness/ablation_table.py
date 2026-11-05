"""Selection ablation table: AUROC / AP of B_known replication over the shared pool (singles, pairs) per selector,
95 % bootstrap over targets, and paired differences vs the reference (per-target difference, bootstrapped).

    python -m dbench.harness.ablation_table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POOL = Path("runs/development/benchmark_v03/pool")
A = Path("runs/development/benchmark_v03/ablation_selection")
ROWS = [("g_theta", POOL, "learned relation search (release checkpoint; reference)"),
        ("sel_gbm", A / "pool", "GBM on the 11 transfer diagnostics + B / L (baseline 9, relabelled cache)"),
        ("sel_ridge", A / "pool", "ridge on the same diagnostics"),
        ("sel_screen", A / "pool", "support-fit screen (support CV gain, no shrinkage model)"),
        ("sel_lopo", A / "pool", "support gain - LOPO SD heuristic"),
        ("sel_learned1", A / "pool", "learned singles + absolute product-residual correlation for pair selection"),
        ("random", POOL, "random (count-matched)")]


def per_target(scores: dict, labels: dict) -> dict:
    from .pool import _ap_auc
    out = {}
    for key, d in scores.items():
        t, f = key.split("|")
        for kind, sk in (("single", "single"), ("pair", "pair")):
            y = labels.get(f"{t}|{kind}|known")
            s = np.asarray(d.get(sk, []), float)
            if y is None or len(s) != len(y):
                continue
            au, ap = _ap_auc(y, s, seeds=5)
            out.setdefault((t, kind), []).append((au, ap))
    return {k: (np.mean([a for a, _ in v if a is not None]) if any(a is not None for a, _ in v) else None,
                np.mean([b for _, b in v if b is not None]) if any(b is not None for _, b in v) else None)
            for k, v in out.items()}


def boot(v, n=2000, seed=0):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(strict: bool = False) -> None:
    lab = json.loads((POOL / ("labels_flagged.json" if strict else "labels.json")).read_text())["labels"]
    res = {}
    for m, d, _ in ROWS:
        p = d / f"scores_{m}_known.json"
        if p.exists():
            res[m] = per_target(json.loads(p.read_text())["scores"], lab)
    ref = res["g_theta"]
    f = lambda v: "—" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
    lines = ["## Selection ablation (methods table): which sets to fit, at matched budget", "",
             "Development targets (40 kidney-injury x 3 patient folds), labels = shared-pool B_known replication" + (" under the strict "
             "beyond-known rule (positives containing an excluded programme score, a member gene of an excluded programme or a "
             "programme containing the gene target are negatives)" if strict else "") + ". Every selector "
             "uses the same diagnostics, LCB beam (26 / 26 / 8 / 4 at K = 64), v03_rff refit and support-gain ordering; it "
             "scores every pool single and pair with its own predicted held-out gain. AUROC / AP per (target, fold), mean over "
             "folds per target, 95 % bootstrap over targets; difference = selector minus reference, per target, bootstrapped.", "",
             "| selector | AUROC singles | AP singles | AUROC pairs | AP pairs | ΔAUROC singles | ΔAP singles | ΔAUROC pairs | ΔAP pairs |",
             "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for m, _, name in ROWS:
        if m not in res:
            lines.append(f"| {name} | pending | | | | | | | |")
            continue
        r, cells, dcells, rec = res[m], [], [], {}
        for kind in ("single", "pair"):
            for j, met in enumerate(("auroc", "ap")):
                ts = [t for (t, k) in r if k == kind]
                v = boot([r[(t, kind)][j] for t in ts])
                dv = boot([r[(t, kind)][j] - ref[(t, kind)][j] for t in ts
                           if (t, kind) in ref and r[(t, kind)][j] is not None and ref[(t, kind)][j] is not None]) if m != "g_theta" else None
                cells.append(f(v)), dcells.append(f(dv) if m != "g_theta" else "—")
                rec[f"{met}_{kind}"], rec[f"delta_{met}_{kind}"] = v, dv
        out[m] = rec
        lines.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")
    (A / "ablation_table.md").write_text("\n".join(lines) + "\n")
    (A / "ablation_table.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    main(strict="--strict" in sys.argv)
