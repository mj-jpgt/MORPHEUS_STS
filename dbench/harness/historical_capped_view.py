"""Distinct-objects view of our locked historical top-20 lists: the anchor-multiplicity cap m applied as a filter of
the SAME top 20 (no new search; sets removed, none added), with per-set flags from the protocol's held-out scoring
(the yield@20 / known-yield / higher-order@20 rules of historical_yield).

    python -m dbench.harness.historical_capped_view --m 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cap_rerank import capped
from .historical_redundancy import redundancy
from .historical_yield import PROTOCOL, task_context

H = Path("runs/development/benchmark_v03/historical")


def flags(result: dict, task: dict, cache: dict, n_boot: int = 200) -> list:
    from dbench import historical as h
    episode, rows, y, design, comparison, groups, support, query, comparison_known, _ = task_context(task, cache)
    out = []
    for entry in result["top"][:20]:
        f = {"names": entry["names"], "yield": False, "yield_known": False, "higher_order": False}
        try:
            V = np.column_stack([h._resolve_values(nm, design, episode, rows) for nm in entry["names"]])
            ho = h._heldout(V, y, groups, support, query, comparison, None, n_boot)
            ci = ho.get("gain_general_ci") or [None]
            f["yield"] = bool(ci and ci[0] is not None and float(ci[0]) > 0)
            gci = ho.get("G_HO_ci") or [None]
            f["higher_order"] = f["yield"] and V.shape[1] >= 2 and bool(gci and gci[0] is not None and float(gci[0]) > 0)
            kci = h._heldout(V, y, groups, support, query, comparison_known, None, n_boot).get("gain_general_ci") or [None]
            f["yield_known"] = bool(kci and kci[0] is not None and float(kci[0]) > 0)
        except Exception:
            pass
        out.append(f)
    return out


def main(argv=None) -> None:
    from dbench.historical import load_tasks
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, default=8)
    a = ap.parse_args(argv)
    tasks = {t["task_id"]: t for t in load_tasks(PROTOCOL)}
    path = H / f"capped_view_m{a.m}.json"
    res, cache = (json.loads(path.read_text()) if path.exists() else {}), {}
    for tid in ("H1", "H2", "H3"):
        if tid in res:   # resumable: one task at a time
            continue
        r = json.loads((H / "ours" / tid / "learned_search" / "result.json").read_text())
        fl = flags(r, tasks[tid], cache)
        keep = capped([tuple(f["names"]) for f in fl], a.m)
        kept = [f for f in fl if tuple(f["names"]) in set(keep)]
        rd = redundancy([f["names"] for f in kept])
        res[tid] = {"m": a.m, "n_kept": len(kept), "yield@20": sum(f["yield"] for f in kept),
                    "yield@20_known": sum(f["yield_known"] for f in kept), "higher_order@20": sum(f["higher_order"] for f in kept),
                    **rd, "uncapped_check": {"yield@20": sum(f["yield"] for f in fl), "yield@20_known": sum(f["yield_known"] for f in fl),
                                             "higher_order@20": sum(f["higher_order"] for f in fl)},
                    "flags": fl}
        print(tid, json.dumps({k: v for k, v in res[tid].items() if k != "flags"}), flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
