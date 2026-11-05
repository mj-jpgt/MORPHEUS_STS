"""Few-shot table: AUROC / AP / precision@10 of B_known singles over the full pool by number of support patients.

    python -m dbench.harness.fewshot_table
"""
from __future__ import annotations

import json
from pathlib import Path

from .public_metrics import NAMES

IN = Path("runs/development/benchmark_v03/fewshot")
OUT = Path("runs/development/benchmark_v03/public/fewshot.md")
ORDER = ("g_theta", "g_theta_lcb", "ebm_full", "gbm_h_full", "linear_screen", "random")
LABEL = {"g_theta": "learned relation search", "g_theta_lcb": "learned relation search (shrinkage-corrected ordering)",
         "ebm_full": "explainable boosting machine (EBM; full pool)",
         "gbm_h_full": "gradient boosting + Friedman's H (full pool)"}
SIZES = ("2", "4", "8", "0")


def main() -> None:
    def fmt(v):
        return "—" if not v or v[0] is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    lines = ["## Few-shot: retrieval of replicating singles from k support patients (labels at B_known)", "",
             "Each method sees only the support rows of k randomly drawn support patients (5 draws per size; 'all' = every "
             "support patient of the fold, 14 in this study) and scores every single of the full pool; labels are the "
             "development table's pooled B_known replication labels. Mean over draws and folds per target, 95 % bootstrap "
             "interval over targets (40 targets; EBM and GBM + H on the first 10). Failed jobs are excluded and counted.", "",
             "| method | support patients | jobs scored | failed | AUROC | AP | precision@10 |", "|---|---|---|---|---|---|---|"]
    notes = []
    for m in ORDER:
        p = IN / f"fewshot_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        fails = d.get("failed_by_size", {})
        for n in SIZES:
            r = d.get(n)
            if r is None and n not in fails:
                continue
            r = r or {}
            lines.append(f"| {LABEL.get(m, NAMES.get(m, m))} | {'all' if n == '0' else n} | {r.get('n_jobs', 0)} | "
                         f"{fails.get(n, 0)} | {fmt(r.get('auroc'))} | {fmt(r.get('ap'))} | {fmt(r.get('precision@10'))} |")
    if (IN / "fewshot_linear_screen.json").exists():
        f = json.loads((IN / "fewshot_linear_screen.json").read_text()).get("failed_by_size", {})
        notes.append("Linear interaction screen: with very few support patients its internal 3-fold patient "
                     "cross-validation leaves degenerate folds and scikit-learn's coordinate descent crashes (segfault, "
                     "recorded per job as a failure); failed jobs by support size: "
                     + ", ".join(f"{'all' if k == '0' else k}: {v}" for k, v in sorted(f.items(), key=lambda x: int(x[0])))
                     + ". Rows with failures are a selected subset of draws.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines + [""] + [f"Note. {n}" for n in notes]) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
