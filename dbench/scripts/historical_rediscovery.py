"""Run the frozen historical-rediscovery protocol: every task x every method.

    python -m dbench.scripts.historical_rediscovery \
        --protocol dbench/protocols/historical_rediscovery_v01.json \
        --checkpoint runs/development/learned_search_v01/frozen.pt   # or: none (marginal-screen stub)
        --methods learned_search,ebm_ga2m,gbm_interaction,linear_interaction_screen,random_search_matched \
        --out runs/development/sprint_v06/historical/

Writes ``<task>__<method>.json`` per run and ``TABLE.md``.  The protocol hash
is checked against ``--expected-sha256`` (default: the journalled
``runs/development/sprint_v06/historical_protocol_sha256.txt``) before anything
is read.  ``learned_search`` runs first so ``random_search_matched`` can be
matched on the number of sets the learned search evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from dbench.historical import before_date, load_tasks, run_task

DEFAULT_METHODS = "learned_search,ebm_ga2m,gbm_interaction,linear_interaction_screen,random_search_matched"
SPLIT_SEED = 20261015
HASH_FILE = "runs/development/sprint_v06/historical_protocol_sha256.txt"
HASH_FILES = {"historical-rediscovery-v01": HASH_FILE,
              "historical-rediscovery-v02": "runs/development/sprint_v06/historical_protocol_v02_sha256.txt"}
HEADER_LINES: list = []
CKPT_SHA = None


def hash_file_for(protocol: str) -> str:
    pid = json.loads(Path(protocol).read_text(encoding="utf-8")).get("protocol_id")
    return HASH_FILES.get(pid, HASH_FILE)


def load_model(checkpoint: str | None):
    if checkpoint in (None, "", "none", "None"):
        return None
    import importlib
    mod = importlib.import_module("morpheus_encoder.learned_search.model")
    for name in ("load_frozen", "load_checkpoint"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn(checkpoint)
    cls = getattr(mod, "LearnedSearchModel", None)
    if cls is not None and callable(getattr(cls, "load", None)):
        return cls.load(checkpoint)
    raise RuntimeError("morpheus_encoder.learned_search.model exposes no checkpoint loader "
                       "(load_frozen / load_checkpoint / LearnedSearchModel.load)")


def split_patients(patients, seed: int = SPLIT_SEED) -> tuple[list, list]:
    """Protocol split: sorted keys, shuffled with default_rng(seed), first ceil(n/2) support."""
    keys = sorted(set(map(str, patients)))
    order = np.random.default_rng(seed).permutation(len(keys))
    shuffled = [keys[i] for i in order]
    h = int(math.ceil(len(keys) / 2))
    return shuffled[:h], shuffled[h:]


def task_episode(task: dict, partitions: str):
    """(episode, gene_matrix or None, notes) through Agent D's loader."""
    from morpheus.tier2.substrates import hest_multi as hm

    specs = hm.study_specs(partitions)
    spec = specs[task["study_key"]]
    if getattr(spec, "pending_conversion", False):
        raise RuntimeError(f"{task['study_key']} is pending conversion")
    support, query = split_patients(spec.patient_of.values())
    target_gene = task["proxy_sets"][task["relation"]["target"]]["genes"][0]
    episode = hm.study_episode(spec, target=f"gene:{target_gene}", support_patients=support,
                               query_patients=query, seed=0)
    notes = {"support_patients": support, "query_patients": query, "episode_target": f"gene:{target_gene}",
             "episode_target_note": "the loader target only drives the episode's own exclusions; the task "
                                    "target is the protocol's gene-set score"}
    gene_matrix = None
    try:
        raw = hm.load_study(spec)
        key = {(s, b): i for i, (s, b) in enumerate(zip(map(str, raw.sample_id), map(str, raw.barcode)))}
        idx = np.array([key[(str(s), str(b))] for s, b in zip(episode.sample_id, episode.barcode)])
        gene_matrix = (raw.logcpm(idx), list(raw.genes))
        notes["gene_matrix"] = "full common panel (StudyRaw.logcpm)"
    except Exception as exc:
        notes["gene_matrix"] = f"episode genes view only ({type(exc).__name__}: {str(exc)[:160]})"
    return episode, gene_matrix, notes


def synthetic_task_and_episode(seed: int = 0):
    """Development smoke only: a planted answer on Agent D's synthetic episode."""
    from dbench.tests.test_historical import planted_episode, planted_task
    return planted_task(), planted_episode(seed), None, {"synthetic": True}


def _fmt_rank(r) -> str:
    return "none" if r in ("inf", None) else str(r)


def write_table(rows: list[dict], path: Path) -> None:
    if any(r.get("hit_rule_version") == "v02_top10_representative" for r in rows):
        return write_table_v02(rows, path)
    lines = ["# Historical rediscovery", "",
             "Hit = a top-20 hypothesis variable under a stated transform with within-patient |r| >= 0.9 "
             "with the answer proxy and the stated sign. Held-out gain / G_HO: first hit on query patients.", "",
             "| task | study | cutoff | method | rank of first hit | hit variable | r | sets evaluated | "
             "held-out gain [CI] | G_HO [CI] | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines[2:2] = HEADER_LINES
    for r in rows:
        h = r.get("heldout_hit") or {}
        gain = h.get("gain_general", h.get("replication_gain"))
        gci = h.get("gain_general_ci")
        gho, gho_ci = h.get("G_HO"), h.get("G_HO_ci")

        def f(x):
            return "" if x is None else f"{x:.3f}"

        lines.append("| {t} | {s} | {d} | {m} | {rk} | {hv} | {rr} | {n} | {g}{gc} | {o}{oc} | {sec:.0f} |".format(
            t=r["task_id"], s=r["study_key"], d=r.get("before_date", ""), m=r["method"],
            rk=_fmt_rank(r.get("rank_of_first_hit")) if r.get("available", True) else (r.get("unavailable") or "unavailable"),
            hv=r.get("hit_variable") or "", rr=f(r.get("r")), n=r.get("n_relationships_tested", ""),
            g=f(gain), gc=f" [{gci[0]:.3f}, {gci[1]:.3f}]" if gci else "", o=f(gho),
            oc=f" [{gho_ci[0]:.3f}, {gho_ci[1]:.3f}]" if gho_ci else "", sec=float(r.get("seconds", 0.0))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_v02(rows: list[dict], path: Path) -> None:
    def f(x):
        return "" if x is None else f"{x:.3f}"

    lines = ["# Historical rediscovery (hit rule v02)", "", *HEADER_LINES,
             "Hit = a top-20 hypothesis containing a variable that, under a stated transform, is among the 10 design "
             "columns with the largest within-patient |r| with the answer proxy, has |r| >= 0.5 and the stated sign. "
             "Ceiling = the largest |r| any design column reaches with an answer proxy under a stated transform. "
             "Secondary = the largest stated-sign proxy |r| among the method's top-20 variables / ceiling. "
             "Held-out gain / G_HO: the hit's set on query patients (relation v07_ho).", "",
             "| task | study | cutoff | method | rank of first hit | hit variable | hit abs(r) | ceiling | secondary "
             "(best abs(r) / ceiling) | sets evaluated | held-out gain [CI] | G_HO [CI] | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        h = r.get("heldout_hit") or {}
        gain, gci = h.get("gain_general", h.get("replication_gain")), h.get("gain_general_ci")
        gho, gho_ci = h.get("G_HO"), h.get("G_HO_ci")
        sec = r.get("seconds_original_run", r.get("seconds", 0.0)) or 0.0
        rk = (_fmt_rank(r.get("rank_of_first_hit")) if r.get("available", True)
              else (r.get("unavailable") or "unavailable"))
        best = r.get("best_signed_over_ceiling")
        lines.append(" | ".join([
            "", r["task_id"], r["study_key"], r.get("before_date", ""), r["method"], rk, r.get("hit_variable") or "",
            f(r.get("hit_abs_r")), f(r.get("task_ceiling")),
            "" if best is None else f"{best:.3f} ({r.get('best_signed_variable')})",
            str(r.get("n_relationships_tested", "") or ""),
            f(gain) + (f" [{gci[0]:.3f}, {gci[1]:.3f}]" if gci else ""),
            f(gho) + (f" [{gho_ci[0]:.3f}, {gho_ci[1]:.3f}]" if gho_ci else ""), f"{float(sec):.0f}", ""]).strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="dbench/protocols/historical_rediscovery_v01.json")
    ap.add_argument("--partitions", default="dbench/protocols/sprint_partitions_v01.json")
    ap.add_argument("--checkpoint", default="none")
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: H1,H2,H3 + S0; H3b if H3 pending)")
    ap.add_argument("--out", default="runs/development/sprint_v06/historical/")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--expected-sha256", default=None)
    ap.add_argument("--synthetic", action="store_true", help="development smoke on a planted synthetic answer")
    ap.add_argument("--nested", action="store_true", help="write <out>/<task>/<method>/result.json and RESULTS.md")
    ap.add_argument("--table-only", action="store_true", help="rebuild RESULTS.md from <out>/<task>/<method>/result.json")
    ap.add_argument("--rescore-from", default=None,
                    help="re-score saved <dir>/<task>/<method>/result.json top-20 lists under --protocol (no method re-run)")
    ap.add_argument("--table-name", default=None, help="results table file name (default RESULTS.md / TABLE.md)")
    ap.add_argument("--awaiting", default="", help="comma list of methods listed as awaiting (e.g. learned_search)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if "runs/prospective" in str(out.resolve()).replace("\\", "/"):
        raise SystemExit("refusing to write under runs/prospective")
    out.mkdir(parents=True, exist_ok=True)
    if not args.synthetic:
        import hashlib
        from dbench.prospective_campaign import canonical_hash
        raw = hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
        canon = canonical_hash(json.loads(Path(args.protocol).read_text(encoding="utf-8")))
        HEADER_LINES[:] = [f"Protocol `{args.protocol}`: canonical-JSON sha256 `{canon}` (verified against "
                           f"`{hash_file_for(args.protocol)}` before any data were read); raw-byte sha256 `{raw}`.", ""]
    if args.table_only:
        rows = [json.loads(f.read_text()) for f in sorted(out.glob("*/*/result.json"))]
        order = {"S0": 0, "H1": 1, "H2": 2, "H3": 3, "H3b": 4}
        rows.sort(key=lambda r: (order.get(r["task_id"], 9), r["method"]))
        tasks = {t["task_id"]: t for t in load_tasks(args.protocol)}
        done = sorted({r["task_id"] for r in rows}, key=lambda t: order.get(t, 9))
        write_table(rows + awaiting_rows([(tasks[t],) for t in done], args.awaiting),
                    out / (args.table_name or "RESULTS.md"))
        return 0
    model = load_model(args.checkpoint)
    global CKPT_SHA
    if args.checkpoint not in (None, "", "none", "None") and Path(args.checkpoint).exists():
        import hashlib
        CKPT_SHA = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "learned_search" in methods:
        methods.remove("learned_search")
        methods.insert(0, "learned_search")

    if args.synthetic:
        task, episode, gm, notes = synthetic_task_and_episode(args.seed)
        plan = [(task, episode, gm, notes)]
    else:
        tasks = load_tasks(args.protocol)
        expected = args.expected_sha256
        hf = hash_file_for(args.protocol)
        if expected is None and Path(hf).exists():
            expected = Path(hf).read_text().split()[0]
        if expected and tasks[0]["protocol_sha256"] != expected:
            raise SystemExit(f"protocol hash {tasks[0]['protocol_sha256']} != journalled {expected}")
        by_id = {t["task_id"]: t for t in tasks}
        wanted = [s for s in args.tasks.split(",") if s] or ["H1", "H2", "H3", "S0"]
        plan = []
        for tid in wanted:
            task = by_id[tid]
            try:
                episode, gm, notes = task_episode(task, args.partitions)
            except Exception as exc:
                if tid == "H3" and "H3b" in by_id and "H3b" not in wanted:
                    print(f"[historical] H3 unavailable ({exc}); using fallback H3b", flush=True)
                    task = by_id["H3b"]
                    try:
                        episode, gm, notes = task_episode(task, args.partitions)
                    except Exception as exc2:
                        print(f"[historical] H3b unavailable: {exc2}", flush=True)
                        continue
                else:
                    print(f"[historical] {tid} unavailable: {exc}", flush=True)
                    (out / f"{tid}__UNAVAILABLE.json").write_text(json.dumps(
                        {"task_id": tid, "before_date": before_date(task), "reason": str(exc)[:500]}, indent=2))
                    continue
            plan.append((task, episode, gm, notes))

    rows = []
    if args.rescore_from:
        from dbench.historical import rescore_saved
        src = Path(args.rescore_from)
        for task, episode, gm, notes in plan:
            for f in sorted((src / task["task_id"]).glob("*/result.json")):
                saved = json.loads(f.read_text())
                res = rescore_saved(saved, task, episode, gene_matrix=gm, n_boot=args.n_boot)
                res["episode_notes"] = notes
                res["rescored_from"] = str(f)
                d = out / task["task_id"] / saved["method"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "result.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
                rows.append(res)
                print(f"[historical] rescored {task['task_id']} {saved['method']}: rank={res['rank_of_first_hit']} "
                      f"secondary={res.get('best_signed_over_ceiling')}", flush=True)
        write_table(rows + awaiting_rows(plan, args.awaiting), out / (args.table_name or "RESULTS.md"))
        return 0
    for task, episode, gm, notes in plan:
        learned_tested = None
        for m in methods:
            kwargs = {}
            if m == "random_search_matched":
                budget = args.budget or int(task["scoring"].get("budget", 128))
                kwargs = {"n_sets": max(budget, learned_tested or 0)}
            t0 = time.time()
            res = run_task(task, episode, m, budget=args.budget, model=model if m == "learned_search" else None,
                           gene_matrix=gm, seed=args.seed, n_threads=args.n_threads, n_boot=args.n_boot,
                           method_kwargs=kwargs)
            res["episode_notes"] = notes
            res["checkpoint"] = args.checkpoint
            res["checkpoint_sha256"] = CKPT_SHA
            if m == "learned_search":
                learned_tested = int(res.get("n_relationships_tested") or 0)
            if args.nested:
                d = out / task["task_id"] / m
                d.mkdir(parents=True, exist_ok=True)
                (d / "result.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
            else:
                (out / f"{task['task_id']}__{m}.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
            rows.append(res)
            write_table(rows + awaiting_rows(plan, args.awaiting), out / (args.table_name or ("RESULTS.md" if args.nested else "TABLE.md")))
            print(f"[historical] {task['task_id']} {m}: rank={res.get('rank_of_first_hit')} "
                  f"tested={res.get('n_relationships_tested')} ({time.time() - t0:.0f}s)", flush=True)
    write_table(rows + awaiting_rows(plan, args.awaiting), out / (args.table_name or ("RESULTS.md" if args.nested else "TABLE.md")))
    return 0


def awaiting_rows(plan, awaiting: str) -> list:
    names = [m.strip() for m in awaiting.split(",") if m.strip()]
    return [{"task_id": t["task_id"], "study_key": t["study_key"], "before_date": before_date(t), "method": m,
             "available": False, "unavailable": "awaiting frozen checkpoint", "seconds": 0.0}
            for t, *_ in plan for m in names]


if __name__ == "__main__":
    sys.exit(main())
