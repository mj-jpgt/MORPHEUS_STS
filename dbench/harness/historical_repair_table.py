"""Historical rediscovery under the repaired query-validated scorer (v03_query_validated; the other session's
dbench/historical_repair.py used unmodified) beside the v02 rank, for every method's locked H1-H3 list.

    python -m dbench.harness.historical_repair_table
"""
from __future__ import annotations

import json
from pathlib import Path

from .historical_yield import NAMES

B = Path("runs/development/benchmark_v03")
R = B / "historical_repair_v03"


def main() -> str:
    v02 = {(r["task"], r["method"]): r["rank_of_first_hit"] for r in json.loads((B / "public" / "historical.json").read_text())}
    lines = ["## Historical rediscovery: v02 hit rule vs repaired query-validated scorer (v03_query_validated)", "",
             "Provenance: locked v02 lists (LOCK_H1_H2.md, LOCK_H3.md) rescored query-side only by "
             "dbench/harness/historical_repair_v03.py (unmodified); no proposer is run. A task is primary-eligible when its proxies "
             "are transcribed from the source (source_exact) and it has at least 3 query patients.", ""]
    for task in ("H1", "H2", "H3"):
        fp = R / "ours" / task / "FEASIBILITY.json"
        if not fp.exists():
            lines += [f"### {task}: pending", ""]
            continue
        fe = json.loads(fp.read_text())
        why = []
        if not fe.get("source_exact"):
            why.append("proxies not source-exact (" + ", ".join(fe.get("source_flags", [])) + ")")
        if fe.get("n_query_patients", 0) < 3:
            why.append(f"{fe.get('n_query_patients')} query patients")
        lines += [f"### {task}: primary_eligible = {fe.get('primary_eligible')}" + (f" ({'; '.join(why)})" if why else ""), "",
                  "| method | rank of first hit (v02) | rank of first query-validated hit (repaired) |", "|---|---|---|"]
        for src in ("ours", "baselines"):
            for p in sorted((R / src / task).glob("*/result.json")):
                r = json.loads(p.read_text())
                m = r.get("method")
                rk = r.get("rank_of_first_query_validated_hit")
                rk = "none" if rk in (None, float("inf"), "inf", "Infinity") or (isinstance(rk, float) and rk == float("inf")) else rk
                v = v02.get((task, m))
                lines.append(f"| {NAMES.get(m, m)} | {'none' if v in (None, 'inf') else v} | {rk} |")
        lines.append("")
    (B / "public" / "historical_repaired.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
