"""Historical rediscovery with the adopted ordering (exact support-fit gain) as 'ours'.

Same tasks, episodes (task_episode), candidate design, budget and hit rule as historical_gpu.py; the only change is the
proposer: the registered 'support_screen' method (search_by_scores, rank_by = support-fit cross-validated gain), no network.
Rule: dbench/protocols/historical_adopted_ordering_v01.json (hashed before this driver ran).

    python -m dbench.harness.historical_adopted --tasks H1,H2,H3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "dbench/protocols/historical_rediscovery_v02.json"
PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"
RULE = "dbench/protocols/historical_adopted_ordering_v01.json"


def main(argv=None) -> None:
    from dbench.historical import load_tasks, run_task
    from dbench.scripts.historical_rediscovery import task_episode
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default="H1,H2,H3")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--out", default="runs/development/benchmark_v03/historical/adopted")
    args = ap.parse_args(argv)
    rule_sha = hashlib.sha256(Path(RULE).read_bytes()).hexdigest()
    tasks = {t["task_id"]: t for t in load_tasks(PROTOCOL)}
    for tid in args.tasks.split(","):
        task = tasks[tid]
        episode, gene_matrix, notes = task_episode(task, PARTITIONS)
        res = run_task(task, episode, "support_screen", budget=args.budget, gene_matrix=gene_matrix)
        res["episode_notes"] = notes
        res["adopted_ordering_rule_sha256"] = rule_sha
        d = Path(args.out) / tid / "support_screen"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(f"[historical_adopted] {tid}: available={res.get('available')} rank={res.get('rank_of_first_hit')} "
              f"tested={res.get('n_relationships_tested')} ({res.get('seconds', 0):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
