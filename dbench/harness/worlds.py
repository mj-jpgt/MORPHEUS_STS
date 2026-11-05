"""Controlled worlds through the harness (governing spec 3.3, Q1).

420 worlds on the real substrate of the development study (never trained on): one real target's
residual plus a planted relation among one to four real candidates, drawn by the v03 episode
machinery (``SamplerV03.episode_for(kind="controlled")``), on one fixed patient split per world.

Grid: 7 families x 4 signal fractions (0.5 / 1 / 2 / 5 % of residual variance) x 15 seeds; the
five held-out evaluation families (Gaussian-process and piecewise functions, never generated in
training) are reported separately from the two training-family representatives.

Per world and method (the method sees support rows only; registry frozen before scoring):
net yield at k in {1, 5, 10, 20} at B_nuis / B_known / B_decomp against the same method on the
two null versions of the world; complete-object recovery @k (the planted set with its spatial
versions); constituent recall @k; reciprocal rank of the planted set; realised higher-order gain
of the top-ranked set vs the planted set's own; sets registered; calibration inputs.

    python -m dbench.harness.worlds --methods screen --shard 0/4 --out runs/development/benchmark_v03/Q1
    python -m dbench.harness.worlds --summarise --out runs/development/benchmark_v03/Q1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import core

HELDOUT = ("gp_2", "gp_3", "piecewise_2", "piecewise_3", "spatial_mix_gp")
TRAINING = ("neural_2", "residualised_2")
FAMILIES = HELDOUT + TRAINING
FRACTIONS = (0.005, 0.01, 0.02, 0.05)
SEEDS = tuple(range(15))
OUT = Path("runs/development/benchmark_v03/Q1")
QUERY_SHARE = 1 / 3


def grid() -> list:
    return [(fam, f, s) for fam in FAMILIES for f in FRACTIONS for s in SEEDS]


def world_id(fam: str, f: float, s: int) -> str:
    return f"{fam}|f{f:g}|s{s}"


def world_task(study, fam: str, f: float, s: int):
    """(task, names, episode, target, support, query) of one world: target and split from the seed."""
    from .dev_metric import choose_targets

    rng = np.random.default_rng(core.stable_seed(world_id(fam, f, s)))
    target = choose_targets(study, n=2, seed=int(rng.integers(1 << 30)))[int(rng.integers(2))]
    pats = rng.permutation(sorted(map(str, study.patients)))
    nq = max(2, int(round(len(pats) * QUERY_SHARE)))
    query, support = sorted(pats[:nq].tolist()), sorted(pats[nq:].tolist())
    task, names, ep = core.build_task(study, target, support, query, seed=int(rng.integers(1 << 30)),
                                      kind="controlled", family=fam, fraction=f)
    return task, names, ep, target, support, query


def object_metrics(names_ranked: list, truth: tuple, ks=core.KS) -> dict:
    """Complete-object recovery @k, constituent recall @k, reciprocal rank of the planted set."""
    t = tuple(sorted(truth))
    sets = [tuple(sorted(s)) for s in names_ranked]
    rank = next((i + 1 for i, s in enumerate(sets) if s == t), None)
    out = {"rank": rank, "reciprocal_rank": 0.0 if rank is None else 1.0 / rank}
    for k in ks:
        seen = set().union(*map(set, sets[:k])) if sets[:k] else set()
        out[f"recovered@{k}"] = rank is not None and rank <= k
        out[f"constituent_recall@{k}"] = len(seen & set(t)) / max(len(t), 1)
    return out


def run_world(study, wid: tuple, methods: dict, *, K: int = 64, Bs=core.BASELINES, null_kinds=core.NULL_KINDS,
              out_dir: Path, device=None) -> list:
    """One row per method for one world (registry frozen per world and method before scoring)."""
    fam, f, s = wid
    task, names, ep, target, support, query = world_task(study, fam, f, s)
    truth_idx = tuple(int(i) for i in ep.true_idx)
    truth_names = [names[i] for i in truth_idx]
    seed = core.stable_seed(world_id(*wid))
    conds = [("real", task)] + [(f"null:{nk}:0", core.null_task(task, ep, nk, seed + 17 * (j + 1)))
                                for j, nk in enumerate(null_kinds)]
    rows = []
    truth_scores = core.score_sets(task, [truth_idx], Bs=Bs, device=device)
    for mname, search in methods.items():
        t0 = time.time()
        registry, ranked_by = [], {}
        try:
            for cond, tk in conds:
                for B in ("nuis", "known"):
                    ranked_by[(cond, B)] = list(search(core.support_only(tk), B, K))[:K]
        except Exception as exc:   # one failing method must not end the shard; the row records why
            import traceback
            rows.append({"world": world_id(*wid), "family": fam, "heldout_family": fam in HELDOUT, "fraction": f,
                         "seed": s, "method": mname, "error": f"{type(exc).__name__}: {exc}"[:300],
                         "traceback": traceback.format_exc()[-1500:]})
            continue
        for (cond, B), ranked in ranked_by.items():
            registry.append({"world": world_id(*wid), "method": mname, "condition": cond, "B": B,
                             "sets": [[names[i] for i in r[0]] for r in ranked]})
        h = core.freeze_registry(registry, out_dir / "registry" / f"{world_id(*wid).replace('|', '_')}__{mname}.json")
        search_s = time.time() - t0
        results = []
        for (cond, B), ranked in ranked_by.items():
            tk = dict(conds)[cond]
            top = [tuple(int(i) for i in r[0]) for r in ranked[:max(core.KS)]]
            if not top:
                continue
            sc = core.score_sets(tk, top, Bs=[B] + (["decomp"] if B == "nuis" else []), device=device)
            results.append(core.FoldResult(world_id(*wid), 0, cond, [[names[i] for i in st] for st in top],
                                           [r[1] if len(r) > 1 else None for r in ranked[:len(top)]], sc,
                                           list(query), [r[2] if len(r) > 2 else None for r in ranked[:len(top)]]))
        ny = core.net_yields(results, Bs=Bs, null_kinds=null_kinds)
        real_nuis = next((r for r in results if r.condition == "real" and "nuis" in r.scores), None)
        top_ho = None
        if real_nuis is not None and len(real_nuis.names[0]) >= 2:
            num, sst, _ = real_nuis.scores["decomp"]
            top_ho = float(np.nansum(num[:, 0]) / max(sst.sum(), 1e-12))
        tn, ts, _ = truth_scores.get("decomp", (None, None, None))
        truth_ho = None if tn is None or len(truth_idx) < 2 else float(np.nansum(tn[:, 0]) / max(ts.sum(), 1e-12))
        tg = truth_scores["nuis"]
        row = {"world": world_id(*wid), "family": fam, "heldout_family": fam in HELDOUT, "fraction": f, "seed": s,
               "method": mname, "target": target, "truth": truth_names, "registry_sha256": h,
               "net_yield": ny, "truth_gain_nuis": float(np.nansum(tg[0][:, 0]) / max(tg[1].sum(), 1e-12)),
               "top_higher_order_gain": top_ho, "truth_higher_order_gain": truth_ho,
               "sets_registered": len(ranked_by[("real", "nuis")]), "search_seconds": round(search_s, 1),
               "seconds": round(time.time() - t0, 1),
               **object_metrics(real_nuis.names if real_nuis else [], tuple(truth_names)),
               "calibration_pairs": [[p, float(np.nansum(real_nuis.scores["nuis"][0][:, j]) /
                                             max(real_nuis.scores["nuis"][1].sum(), 1e-12)), sd]
                                     for j, (p, sd) in enumerate(zip(real_nuis.pred, real_nuis.pred_sd or []))
                                     if p is not None] if real_nuis else []}
        rows.append(row)
    return rows


def method_table(names: list, device=None) -> dict:
    from . import baselines
    table = {"screen": lambda: baselines.screen_search(device=device)}
    table.update(getattr(baselines, "EXTRA_METHODS", lambda device=None: {})(device=device))
    return {n: table[n]() for n in names}


def main(argv=None) -> None:
    from .dev_metric import load_dev_study

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", default="screen")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-worlds", type=int, default=None)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--tag", default="q1", help="rows file tag (one per concurrent launch)")
    ap.add_argument("--seeds", default=None, help="comma list: restrict the grid (expensive baselines on seeds 0-4)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarise:
        from .summary import write_q1
        print(write_q1(out))
        return
    study = load_dev_study()
    i, n = (int(x) for x in args.shard.split("/"))
    wanted = [m for m in args.methods.split(",") if m]
    rows_path = out / f"rows_{args.tag}_shard{i}of{n}.jsonl"
    done = set()
    for p in out.glob("rows_*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:   # errored rows are retried
                    done.add((r["world"], r["method"]))
    methods = method_table(wanted, device=args.device)
    seeds = None if not args.seeds else {int(x) for x in args.seeds.split(",")}
    worlds_todo = [w for w in grid() if seeds is None or w[2] in seeds]
    for wid in worlds_todo[i::n][:args.max_worlds]:
        todo = {m: fn for m, fn in methods.items() if (world_id(*wid), m) not in done}
        if not todo:
            continue
        for m, fn in todo.items():   # one method at a time: each row is written as soon as it exists
            t = time.time()
            rows = run_world(study, wid, {m: fn}, K=args.K, out_dir=out, device=args.device)
            with rows_path.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=float) + chr(10))
            print(f"[{time.strftime('%H:%M:%S')}] {world_id(*wid)} {time.time() - t:.0f} s "
                  + " ".join(f"{r['method']}: ERROR {r['error']}" if "error" in r else
                             f"{r['method']}: net@10={r['net_yield']['nuis'][10]['net']:.2f} rr={r['reciprocal_rank']:.2f}"
                             for r in rows), flush=True)


if __name__ == "__main__":
    main()
