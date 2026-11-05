"""Historical rediscovery for our method with the frozen model on the GPU (the protocol runner keeps the model
on the CPU, where the search's per-candidate fits take hours on the larger studies).  Same protocol, split,
design, budget and hit scoring (``dbench.historical.run_task``); writes ``<out>/<task>/<method>/result.json``
for ``learned_search`` and the count-matched ``random_search_matched``.

    python -m dbench.harness.historical_gpu --checkpoint runs/development/learned_search_v03/release/frozen.pt --tasks H3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL = "dbench/protocols/historical_rediscovery_v02.json"
PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"


def main(argv=None) -> None:
    from dbench.historical import load_tasks, run_task
    from dbench.scripts.historical_rediscovery import load_model, task_episode

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", default="H1,H2,H3")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--out", default="runs/development/benchmark_v03/historical/ours")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    model = load_model(args.checkpoint).to(args.device)
    tasks = {t["task_id"]: t for t in load_tasks(PROTOCOL)}
    for tid in args.tasks.split(","):
        task = tasks[tid]
        episode, gene_matrix, notes = task_episode(task, PARTITIONS)
        res = run_task(task, episode, "learned_search", budget=args.budget, model=model, gene_matrix=gene_matrix)
        res["episode_notes"] = notes
        tested = int(res.get("n_relationships_tested") or 0)
        rnd = run_task(task, episode, "random_search_matched", budget=args.budget, gene_matrix=gene_matrix,
                       method_kwargs={"n_sets": max(args.budget, tested)})
        for r in (res, rnd):
            d = Path(args.out) / tid / r["method"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "result.json").write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
            print(f"[historical] {tid} {r['method']}: rank={r.get('rank_of_first_hit')} tested={r.get('n_relationships_tested')} "
                  f"({r.get('seconds', 0):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
