"""Audit and query-only rescore of saved historical rediscovery receipts.

This runner never invokes a proposer. It writes a separate hash-addressed
directory, leaving v02 receipts untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dbench import historical as H
from dbench.historical import load_tasks
from dbench.historical_repair import audit_task_feasibility, learned_direct_grammar_ranked, rescore_saved_v03, score_ranked_list_v03
from dbench.scripts.historical_rediscovery import task_episode

PROTOCOL = "dbench/protocols/historical_rediscovery_v02.json"
PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"
DEFAULT_INPUT = "runs/development/benchmark_v03/historical/ours"
DEFAULT_OUT = "runs/development/benchmark_v03/historical_repair_v03"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--partitions", default=PARTITIONS)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tasks", default="H1,H2,H3,H3b,S0")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rerun-learned-direct", action="store_true",
                    help="support-only learned replay with own-level input columns and one spatial transform")
    ap.add_argument("--checkpoint", default="runs/development/learned_search_v03/release/frozen.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    protocol, partitions, src, out = map(Path, (args.protocol, args.partitions, args.input, args.out))
    tasks = {t["task_id"]: t for t in load_tasks(protocol)}
    model = None
    if args.rerun_learned_direct:
        from morpheus_encoder.learned_search.model_v03 import GainModelV03
        model = GainModelV03.load(args.checkpoint, map_location=args.device).to(args.device)
    manifest = {"repair": "historical_query_validated_v03", "protocol": str(protocol),
                "protocol_sha256": _sha(protocol), "partitions": str(partitions),
                "partitions_sha256": _sha(partitions), "input": str(src), "tasks": []}
    for tid in [x for x in args.tasks.split(",") if x]:
        task = tasks[tid]
        episode, genes, notes = task_episode(task, str(partitions))
        scores = H.proxy_scores(episode, task, gene_matrix=genes)
        rows = H.task_rows(episode, task, scores)
        y = H.target_values(episode, task, scores, rows)
        design = H.build_design(episode, task, rows, y)
        audit = audit_task_feasibility(task, episode, gene_matrix=genes, n_boot=args.n_boot)
        audit["episode_notes"] = notes
        td = out / tid
        td.mkdir(parents=True, exist_ok=True)
        (td / "FEASIBILITY.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        rescored = []
        for receipt in sorted((src / tid).glob("*/result.json")):
            saved = json.loads(receipt.read_text())
            row = rescore_saved_v03(saved, task, episode, gene_matrix=genes, scores=scores, rows=rows, y=y,
                                    design=design, n_boot=args.n_boot)
            row["source_receipt"] = str(receipt)
            row["source_receipt_sha256"] = _sha(receipt)
            row["episode_notes"] = notes
            rd = td / str(saved.get("method", receipt.parent.name))
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "result.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rescored.append(row)
        if model is not None:
            ranked, report = learned_direct_grammar_ranked(task, episode, model, gene_matrix=genes, budget=64)
            row = {"task_id": tid, "study_key": task["study_key"], "method": "learned_search_direct_grammar",
                   "direct_grammar_report": report,
                   **score_ranked_list_v03(ranked, episode, task, gene_matrix=genes, n_boot=args.n_boot)}
            rd = td / "learned_search_direct_grammar"
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "result.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rescored.append(row)
        manifest["tasks"].append({"task_id": tid, "feasibility": audit,
                                  "n_rescored_receipts": len(rescored), "source_exact": audit["source_exact"],
                                  "primary_eligible": audit["primary_eligible"]})
        print(f"[historical-repair] {tid}: primary_eligible={audit['primary_eligible']} receipts={len(rescored)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
