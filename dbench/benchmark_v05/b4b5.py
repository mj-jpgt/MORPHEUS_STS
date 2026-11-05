"""B4 (higher-order specificity, Meylan blind) and B5 (historical rediscovery) consolidated from existing receipts (read only).

B5: Hit@10, Hit@20, MRR per method per task from historical_rediscovery_v02_final/RESULTS.md (hit rule v02, protocol sha f79f4170...);
the adopted support-fit ordering row is from benchmark_v03/historical_adopted.log (rank = inf: no hit in its 64 tested sets on H1-H3).
B4: the F.1 table (fold-level B_decomp pass; learned beam vs support-fit vs joint at equal budget) and the paired differences, verbatim.
"""
import re
from pathlib import Path

V06 = Path.home() / "morpheus_work_v06/runs/development"
rows = [ln for ln in (V06 / "historical_rediscovery_v02_final/RESULTS.md").read_text().splitlines() if ln.startswith("| H")]
data = {}
for ln in rows:
    c = [x.strip() for x in ln.strip("|").split("|")]
    task, method, rank = c[0], c[3], c[4]
    data.setdefault(method, {})[task] = None if rank == "none" else int(rank)
names = {"ebm_ga2m": "EBM (GA2M)", "gbm_interaction": "GBM + Friedman H", "learned_search": "learned beam (release checkpoint; hit-rule v02 list)",
         "linear_interaction_screen": "linear interaction screen", "random_search_matched": "random (count-matched)"}
tasks = ["H1", "H2", "H3"]
out = ["### B5 historical rediscovery (hit rule v02: a top-20 hypothesis containing a variable among the 10 design columns with the largest within-patient |r| with the answer proxy, |r| >= 0.5, stated sign)", "",
       "| method | rank of first hit H1 / H2 / H3 | Hit@10 (of 3 tasks) | Hit@20 (of 3) | MRR |", "|---|---|---|---|---|"]
for m, d in sorted(data.items()):
    rk = [d.get(t) for t in tasks]
    h10 = sum(1 for r in rk if r is not None and r <= 10)
    h20 = sum(1 for r in rk if r is not None and r <= 20)
    mrr = sum((1.0 / r) if r else 0.0 for r in rk) / len(tasks)
    out.append(f"| {names.get(m, m)} | {' / '.join('none' if r is None else str(r) for r in rk)} | {h10} | {h20} | {mrr:.3f} |")
out.append("| ours: support-fit ordering (adopted ordering, 64 tested sets) | none / none / none | 0 | 0 | 0.000 |")
out += ["", "Random is the count-matched chance reference. Tasks are 3 (cSCC, CRC, prostate); with n = 3 no interval is informative. H1/H2 proxies are protocol-derived (not source-exact); the repaired query-validated scorer (benchmark_v03/public/historical_repaired.md) keeps the H2 hits of the learned beam (rank 3), the linear screen (5), PySR (4) and RuleFit (2) and removes the random hit.",
        "Sources: runs/development/historical_rediscovery_v02_final/RESULTS.md; runs/development/benchmark_v03/historical_adopted.log; benchmark_v03/public/historical_repaired.md."]
b4 = (V06 / "benchmark_v03/public/meylan_decomp_selection_f1.md").read_text()
dp = (V06 / "benchmark_v03/public/decomp_paired.md").read_text()
rep = (V06 / "benchmark_v03/public/historical_repaired.md").read_text()
text = ("\n".join(out) + "\n\nChance-separation check FAILED under hit rule v02: count-matched random hits all 3 tasks (MRR 0.194) and every method is at or below it. "
        "The repaired query-validated scorer below is the informative variant.\n\n" + rep + "\n\n### B4 higher-order specificity, Meylan blind cohort (F.1)\n\n" + b4) + "\n\n### B4 paired decomposition differences (development, public baselines)\n\n" + dp
dest = Path.home() / "morpheus_work_bench/runs/benchmark_v05/b4_b5_consolidated.md"
dest.write_text(text)
print("\n".join(out))
