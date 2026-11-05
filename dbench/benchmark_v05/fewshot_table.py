"""B3 few-shot learning curves from the stored few-shot labels and scores: AP against the number of support patients n.

    python -m dbench.benchmark_v05.fewshot_table STUDY[,STUDY...]

One metrics_v05 table per (study, n) (draws are the splits, 8 targets, hierarchical bootstrap over targets); the learning-curve table
collects AP [95% CI] per method and n; the full-data AP (B1, same 8 targets) is the last column; 'n at 80% of full-data AP' is the smallest
n in the grid whose mean AP reaches 80% of the method's full-data AP on the same targets (not reached = '>max n').
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from dbench.benchmark_v05 import metrics_v05 as M

V5 = M.V5
METHODS = ["sel_screen", "sel_screen_joint", "linear_screen", "nid", "misty", "random"]
NS = [2, 3, 5, 8]


def main():
    studies = sys.argv[1].split(",")
    out = []
    for st in studies:
        rows = {m: {} for m in METHODS}
        full = {}
        base_targets = json.loads((V5 / "labels" / f"{st}__fs{NS[0]}" / "targets.json").read_text())["targets"]
        for n in NS:
            name = f"{st}__fs{n}"
            if not (V5 / "labels" / name / "cache").exists() or not any((V5 / "labels" / name / "cache").glob("*.npz")):
                continue
            M._CACHE.clear()
            M.table([name], METHODS, "95", "known", name=f"fewshot_{st}_n{n}")
            js = json.loads((V5 / f"fewshot_{st}_n{n}.json").read_text())["rows"]
            for m in METHODS:
                for kind in ("single", "pair"):
                    d = js.get(m, {}).get(f"ap_{kind}")
                    if d:
                        rows[m][(n, kind)] = (d["mean"], d["lo"], d["hi"])
        # full-data AP on the same targets (B1 receipts and labels)
        for m in METHODS:
            for kind in ("single", "pair"):
                v = []
                for t in base_targets:
                    d = M.per_split(st, m, t, "95", "known")
                    if d and d[kind]:
                        v.append(M.mean_over(d[kind], None, "ap"))
                if v:
                    full[(m, kind)] = float(np.nanmean(v))
        for kind in ("single", "pair"):
            out.append(f"### B3 few-shot, {st}, {kind}s: AP against support patients (first {len(base_targets)} targets, 5 draws per n; 95% bootstrap over targets)\n")
            out.append("| method | " + " | ".join(f"n={n}" for n in NS) + " | full-data AP (same targets) | smallest n at 80% of full-data AP |")
            out.append("|---|" + "---|" * (len(NS) + 2))
            for m in METHODS:
                cells = [f"{rows[m][(n, kind)][0]:.3f} [{rows[m][(n, kind)][1]:.3f}, {rows[m][(n, kind)][2]:.3f}]" if (n, kind) in rows[m] else "-" for n in NS]
                f = full.get((m, kind))
                reach = next((n for n in NS if (n, kind) in rows[m] and f and rows[m][(n, kind)][0] >= 0.8 * f), None)
                out.append(f"| {M.DISPLAY.get(m, m)} | " + " | ".join(cells) + f" | {f:.3f} | {reach if reach else '>' + str(NS[-1])} |" if f is not None else f"| {M.DISPLAY.get(m, m)} | " + " | ".join(cells) + " | - | - |")
            out.append("")
    text = "\n".join(out) + "\nRandom is the chance reference. Support patients are nested across n within a draw; the three query patients are the same for every n of a draw. Labels: v05 label engine on that (support, query).\n"
    (V5 / ("b3_fewshot_" + "_".join(studies) + ".md")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
