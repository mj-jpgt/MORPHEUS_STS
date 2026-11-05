"""Historical rediscovery: yield@20 and higher-order@20 beside rank of first hit (owner decision 2026-10-22).

For every stored run ``<dir>/<task>/<method>/result.json`` of the hashed protocol, each of the method's top-20
hypotheses is fitted on the study's support patients and scored on its query patients with the protocol's own
held-out relation and comparison model (``dbench.historical._heldout``, the hit-scoring path):
  yield@20         = number of top-20 hypotheses whose held-out general gain has a 95 % bootstrap CI lower bound > 0
  higher-order@20  = how many of those (|S| >= 2) also have a G_HO CI lower bound > 0.
  yield@20 (known) = the same count under the known-biology comparison model: nuisance + the study's Hallmark /
                     Gavish programme scores (centred within patient), dropping every programme whose gene set
                     overlaps the task's target gene set (overlap coefficient >= 0.2 on the panel), the B_known
                     exclusion rule of relation_v03.known_covariates. These studies carry no composition view.
These are the same replication labels as the development retrieval table, applied to untouched studies.

    python -m dbench.harness.historical_yield runs/development/benchmark_v03/historical/baselines runs/.../ours
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = "dbench/protocols/historical_rediscovery_v02.json"
PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"
NAMES = {"learned_search": "learned relation search", "random_search_matched": "random selection (count-matched)",
         "linear_interaction_screen": "linear interaction screen", "rulefit_sklearn": "RuleFit (scikit-learn reimplementation)",
         "pysr_shortlist": "PySR symbolic regression", "gbm_interaction": "gradient boosting + Friedman's H",
         "ebm_ga2m": "explainable boosting machine (EBM)", "tabpfn_spex": "TabPFN-v2 + SPEX",
         "nid_interaction": "neural interaction detection (NID)", "misty_rf": "MISTy multi-view random forest (scikit-learn reimplementation)"}


def task_context(task, cache: dict):
    from dbench import historical as h
    from dbench.scripts.historical_rediscovery import task_episode
    if task["task_id"] not in cache:
        episode, gene_matrix, _ = task_episode(task, PARTITIONS)
        scores = h.proxy_scores(episode, task, gene_matrix=gene_matrix)
        rows = h.task_rows(episode, task, scores)
        y = h.target_values(episode, task, scores, rows)
        design = h.build_design(episode, task, rows, y)
        comparison = h._nuisance(episode, rows)[0]
        comparison_known, dropped = known_comparison(episode, task, rows, comparison)
        groups = h._groups(episode)[rows]
        support = np.flatnonzero(np.isin(rows, np.asarray(episode.support_rows)))
        query = np.flatnonzero(np.isin(rows, np.asarray(episode.query_rows)))
        cache.clear()
        cache[task["task_id"]] = (episode, rows, y, design, comparison, groups, support, query, comparison_known,
                                  dropped)
    return cache[task["task_id"]]


def known_comparison(episode, task: dict, rows: np.ndarray, nuisance: np.ndarray) -> tuple:
    """(nuisance + programme scores without the target-overlapping ones, dropped programme names)."""
    from dbench import historical as h
    from morpheus.tier2.substrates import hest_multi as hm
    names = [str(n) for n in episode.columns["programmes"]]
    panel = {str(g) for g in episode.columns.get("genes", [])}
    target = set(task["proxy_sets"][task["relation"]["target"]]["genes"])
    a = target & panel if panel else target
    keep, dropped = [], []
    for j, n in enumerate(names):
        b = set(hm._programme_members(n))
        b = b & panel if panel else b
        if a and b and len(a & b) / min(len(a), len(b)) >= hm.OVERLAP_THRESHOLD:
            dropped.append(n)
        else:
            keep.append(j)
    P = np.asarray(episode.views["programmes"], dtype=np.float64)[rows][:, keep]
    return np.column_stack([nuisance, h._centre(P, h._groups(episode)[rows])]), dropped


def score_result(result: dict, task: dict, cache: dict, n_boot: int = 200, top_k: int = 20) -> dict:
    from dbench import historical as h
    episode, rows, y, design, comparison, groups, support, query, comparison_known, dropped = task_context(task, cache)
    n_yield = n_ho = n_scored = n_yield_known = 0
    per_set = []
    for entry in (result.get("top") or [])[:top_k]:
        rec = {"names": list(entry["names"]), "yield": False, "yield_known": False, "higher_order": False, "scored": False}
        per_set.append(rec)
        try:
            V = np.column_stack([h._resolve_values(nm, design, episode, rows) for nm in entry["names"]])
            ho = h._heldout(V, y, groups, support, query, comparison, None, n_boot)
        except Exception:   # an unresolvable name counts as not replicating
            continue
        n_scored += 1
        rec["scored"] = True
        try:
            kci = h._heldout(V, y, groups, support, query, comparison_known, None, n_boot).get("gain_general_ci") or [None]
            rec["yield_known"] = bool(kci) and kci[0] is not None and float(kci[0]) > 0
            n_yield_known += int(rec["yield_known"])
        except Exception:
            pass
        ci = ho.get("gain_general_ci") or [None]
        if ci and ci[0] is not None and float(ci[0]) > 0:
            n_yield += 1
            rec["yield"] = True
            gci = ho.get("G_HO_ci") or [None]
            if V.shape[1] >= 2 and gci and gci[0] is not None and float(gci[0]) > 0:
                n_ho += 1
                rec["higher_order"] = True
    rank = result.get("rank_of_first_hit")
    return {"task": task["task_id"], "method": result.get("method"), "rank_of_first_hit": rank,
            "hit@10": rank not in (None, "inf") and int(rank) <= 10, "hit@20": rank not in (None, "inf") and int(rank) <= 20,
            "reciprocal_rank": 0.0 if rank in (None, "inf") else 1.0 / int(rank),
            "yield@20": n_yield, "higher_order@20": n_ho, "yield@20_known": n_yield_known, "n_top_scored": n_scored,
            "known_programmes_dropped": dropped,
            "n_relationships_tested": result.get("n_relationships_tested"), "sets": per_set}


def main(argv=None) -> None:
    from dbench.historical import load_tasks
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="runs/development/benchmark_v03/public/historical.md")
    ap.add_argument("--tasks", default=None, help="comma list; with --rows-out only the rows JSON is written")
    ap.add_argument("--rows-out", default=None)
    args = ap.parse_args(argv)
    tasks = {t["task_id"]: t for t in load_tasks(PROTOCOL)}
    rows, cache = [], {}
    for tid in sorted(tasks):
        if args.tasks and tid not in args.tasks.split(","):
            continue
        for d in args.dirs:
            for p in sorted(Path(d).glob(f"{tid}/*/result.json")):
                res = json.loads(p.read_text())
                if not res.get("available", True):
                    continue
                rows.append(score_result(res, tasks[tid], cache))
                print(json.dumps({k: v for k, v in rows[-1].items() if k != "sets"}), flush=True)
    if args.rows_out:
        Path(args.rows_out).write_text(json.dumps(rows, indent=1))
        return
    lines = ["# Historical rediscovery (hashed protocol v02; untouched studies)", "",
             "Rank of first hit = the protocol's hit rule on the top 20. yield@20 = top-20 hypotheses whose held-out "
             "general gain on the study's query patients has a 95 % bootstrap CI lower bound > 0; higher-order@20 = "
             "those that also have G_HO CI lower bound > 0 (the protocol's relation and comparison model). These are the "
             "same replication labels as the development retrieval table, applied to untouched studies. Every method "
             "receives the same data and broad target and is cut to the same 64 tested sets.", "",
             "yield@20 (known) = the same count under the known-biology comparison model (nuisance + the study's "
             "Hallmark / Gavish programme scores, minus every programme overlapping the target gene set at overlap "
             "coefficient >= 0.2; the B_known exclusion rule of the development table; no composition view exists for "
             "these studies). Under the nuisance-only model yield@20 sits near the ceiling of 20 for every method; the "
             "known-biology column is the one that separates methods.", "",
             "| task | method | rank of first hit | hit@10 | hit@20 | reciprocal rank | yield@20 (nuisance) | yield@20 (known) | higher-order@20 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {NAMES.get(r['method'], r['method'])} | {r['rank_of_first_hit']} | "
                     f"{'yes' if r['hit@10'] else 'no'} | {'yes' if r['hit@20'] else 'no'} | {r['reciprocal_rank']:.3f} | "
                     f"{r['yield@20']} | {r['yield@20_known']} | {r['higher_order@20']} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
