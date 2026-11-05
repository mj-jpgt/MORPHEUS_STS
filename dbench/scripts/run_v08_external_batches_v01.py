"""Run v08 external confirmation in small resumable batches, then aggregate globally.

The relation freezer is memory-bound on the Lambda host.  Batching changes
only process lifetime: every candidate retains its frozen relation, source
cohort and external cohorts.  The companion aggregator recomputes BH over the
complete declared family, so no biological conclusion is based on the
per-batch BH values emitted by the primitive runner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dbench.downstream.blind import internal_set, read_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--include-proposed", action="store_true")
    ap.add_argument("--proposed-only", action="store_true",
                    help="Run only proposed hypotheses (requires --include-proposed).")
    ap.add_argument("--external-cellvit-root", default=None)
    ap.add_argument("--exclude-confirmation-root", type=Path, default=None,
                    help="A completed confirmation root whose receipt IDs should not be rerun.")
    ap.add_argument("--candidate-ids-file", type=Path, default=None,
                    help="Restrict to the candidate ids listed (one per line) within the selected families.")
    args = ap.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run = read_run(args.run_dir)
    families = {x for x in args.families.split(",") if x}
    rows = internal_set(run)
    if args.include_proposed:
        rows += [r for r in run["funnel"]["table"] if r["ledger_stage"] == "proposed"]
    if args.proposed_only:
        if not args.include_proposed:
            raise ValueError("--proposed-only requires --include-proposed")
        rows = [r for r in rows if r["ledger_stage"] == "proposed"]
    rows = [r for r in rows if str(r.get("family")) in families]
    rows.sort(key=lambda r: r["candidate_id"])
    excluded: set[str] = set()
    if args.exclude_confirmation_root:
        for path in args.exclude_confirmation_root.glob("cards/*/records/02_external_replication.record.json"):
            try:
                excluded.add(str(json.loads(path.read_text())["finding_id"]))
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        rows = [r for r in rows if r["candidate_id"] not in excluded]
    if args.candidate_ids_file:
        wanted = {x.strip() for x in args.candidate_ids_file.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["candidate_id"] in wanted]
        if len(rows) != len(wanted):
            raise ValueError(f"{len(wanted) - len(rows)} requested candidate ids are not in the selected scope")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "morpheus_v08_external_batches/v01", "run_dir": args.run_dir,
            "families": sorted(families), "batch_size": args.batch_size,
            "include_proposed": bool(args.include_proposed),
            "proposed_only": bool(args.proposed_only),
            "excluded_previously_confirmed": sorted(excluded),
            "candidate_ids": [r["candidate_id"] for r in rows],
            "candidate_stages": {r["candidate_id"]: r["ledger_stage"] for r in rows},
            "external_cellvit_root": args.external_cellvit_root}
    (out / "batch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for batch_index, start in enumerate(range(0, len(rows), args.batch_size), 1):
        batch = rows[start:start + args.batch_size]
        batch_out = out / f"batch_{batch_index:03d}"
        summary = batch_out / "replication_summary.json"
        if summary.exists():
            print(json.dumps({"batch": batch_index, "status": "already_complete", "ids": [r["candidate_id"] for r in batch]}))
            continue
        cmd = [sys.executable, "-m", "dbench.downstream.blind", "--run-dir", args.run_dir,
               "--out", str(batch_out), "--stage", "replication", "--families", ",".join(sorted(families)),
               "--candidate-ids", ",".join(r["candidate_id"] for r in batch)]
        if args.include_proposed:
            cmd.append("--include-proposed")
        if args.external_cellvit_root:
            cmd += ["--external-cellvit-root", args.external_cellvit_root]
        print(json.dumps({"batch": batch_index, "status": "starting", "ids": [r["candidate_id"] for r in batch]}), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
