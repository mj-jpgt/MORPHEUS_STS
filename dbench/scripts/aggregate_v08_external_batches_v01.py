"""Apply one declared BH family over externally evaluated v08 batch receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bh(values: list[float]) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1.0)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prior-confirmation-root", type=Path, action="append", default=[],
                    help="Completed confirmation root(s) in the same declared BH family.")
    args = ap.parse_args()
    plan = json.loads((args.root / "batch_plan.json").read_text())
    expected = set(plan["candidate_ids"])
    records = {}
    for path in sorted(args.root.glob("batch_*/cards/*/records/02_external_replication.record.json")):
        doc = json.loads(path.read_text())
        fid, result = doc["finding_id"], doc["result"]
        records[fid] = {"receipt": str(path), "result": result}
    for root in args.prior_confirmation_root:
        for path in sorted(root.glob("cards/*/records/02_external_replication.record.json")):
            doc = json.loads(path.read_text())
            fid, result = doc["finding_id"], doc["result"]
            if fid in records:
                raise RuntimeError(f"duplicate external receipt for {fid}: {path}")
            records[fid] = {"receipt": str(path), "result": result}
    missing = sorted(expected - set(records))
    evaluable = [fid for fid in sorted(records) if records[fid]["result"].get("pooled", {}).get("nuis")]
    q = bh([records[fid]["result"]["pooled"]["nuis"]["p_boot"] for fid in evaluable]) if evaluable else []
    global_results = {}
    for fid, qq in zip(evaluable, q):
        result = records[fid]["result"]
        nuisance = result["pooled"]["nuis"]
        direction = result.get("direction", {}).get("fraction_agree")
        criteria = {"ci_excludes_zero": bool(nuisance["ci"][0] > 0),
                    "direction_agreement_ge_0.60": bool(direction is not None and direction >= 0.60),
                    "bh_q_le_0.05": bool(qq <= 0.05)}
        global_results[fid] = {"status": "tested", "global_bh_q": float(qq),
                               "criteria": criteria, "externally_replicated": bool(all(criteria.values())),
                               "receipt": records[fid]["receipt"]}
    for fid, value in records.items():
        if fid not in global_results:
            global_results[fid] = {"status": "not_testable", "externally_replicated": False,
                                   "receipt": value["receipt"],
                                   "testability": value["result"].get("testability")}
    output = {"schema": "morpheus_v08_external_global_bh/v01", "plan": plan,
              "prior_confirmation_roots": [str(x) for x in args.prior_confirmation_root],
              "n_expected": len(expected), "n_receipts": len(records), "missing": missing,
              "n_evaluable": len(evaluable),
              "n_replicated": sum(x.get("externally_replicated", False) for x in global_results.values()),
              "results": global_results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("n_expected", "n_receipts", "n_evaluable", "n_replicated", "missing")}, indent=2))


if __name__ == "__main__":
    main()
