"""Corrected development retrieval table: every learned / selector score recomputed from identical core.support_only
tasks (pool_v2); full-pool baseline scorers (EBM, NID, MISTy, random) from pool/ after the reproduction check.
AUROC / AP of B_known replication over the shared pool, as-registered and strict, Δ vs the support-fit screen.

    python -m dbench.harness.dev_retrieval_v2 [--strict]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .ablation_table import boot, per_target

B = Path("runs/development/benchmark_v03")
ROWS = [("g_theta", B / "pool_v2", "learned relation search (release checkpoint)"),
        ("sel_hybrid", B / "pool_v2", "hybrid: exact support gain - network shrinkage"),
        ("sel_screen_joint", B / "pool_v2", "support-fit screen, exact joint pair gain"),
        ("sel_screen", B / "pool_v2", "support-fit screen"),
        ("sel_learned1", B / "pool_v2", "network singles + product-residual correlation pairs"),
        ("sel_gbm", B / "pool_v2", "GBM on the transfer diagnostics (baseline 9)"),
        ("sel_ridge", B / "pool_v2", "ridge on the transfer diagnostics"),
        ("sel_lopo", B / "pool_v2", "support gain - LOPO SD heuristic"),
        ("ebm_full", B / "pool", "explainable boosting machine (EBM; full pool)"),
        ("nid", B / "pool", "neural interaction detection (NID)"),
        ("gbm_h_full", B / "pool", "gradient boosting + Friedman's H (full pool)"),
        ("misty", B / "pool", "MISTy multi-view random forest (singles only)"),
        ("random", B / "pool", "random")]
REF = "sel_screen"


def main(strict: bool = False) -> str:
    lab = json.loads((B / "pool" / ("labels_flagged.json" if strict else "labels.json")).read_text())["labels"]
    res = {m: per_target(json.loads((d / f"scores_{m}_known.json").read_text())["scores"], lab)
           for m, d, _ in ROWS if (d / f"scores_{m}_known.json").exists()}
    f = lambda v: "—" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
    lines = [f"## Development retrieval, corrected ({'strict beyond-known' if strict else 'as registered'})", "",
             "40 kidney-injury targets x 3 patient folds; B_known replication over the shared pool; every learned and selector "
             "score from identical core.support_only tasks with today's code; EBM / NID / GBM+H / MISTy / random from their "
             "stored full-pool scores (support rows only; see the reproduction check). Δ = method minus the support-fit "
             "screen, per target, 95 % bootstrap over targets.", "",
             "| method | AUROC singles | AP singles | AUROC pairs | AP pairs | ΔAUROC singles | ΔAP singles | ΔAUROC pairs | ΔAP pairs |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, _, name in ROWS:
        if m not in res:
            continue
        r, cells, dcells = res[m], [], []
        ref = res.get(REF, {})
        for kind in ("single", "pair"):
            for j in (0, 1):
                ts = [t for (t, k) in r if k == kind and r[(t, k)][j] is not None]
                cells.append(f(boot([r[(t, kind)][j] for t in ts])))
                diff = [r[(t, kind)][j] - ref[(t, kind)][j] for t in ts if (t, kind) in ref and ref[(t, kind)][j] is not None]
                dcells.append("—" if m == REF or not diff else f(boot(diff)))
        lines.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")
    out = B / "public" / f"dev_retrieval_v2{'_strict' if strict else ''}.md"
    out.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main(strict="--strict" in sys.argv))
