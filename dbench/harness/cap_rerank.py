"""Anchor-multiplicity cap on the development targets: B_known yield@20 and higher-order@20 of each 64-set list
re-ranked greedily (method order; skip a set that would make any variable appear in more than m chosen sets).

Lists: g_theta searched under B_nuis and B_known, random (count-matched) under both. Labels are the development
table's: a set replicates if its realised B_known held-out gain pooled over the three folds' query patients has a
95 % patient-bootstrap lower bound > 0; higher-order = replicating, |S| >= 2 and pooled B_decomp lower bound > 0.

    python -m dbench.harness.cap_rerank labels --workers 8     # resumable, one cache file per (target, fold)
    python -m dbench.harness.cap_rerank table
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import core

OUT = Path("runs/development/benchmark_v03/cap_rerank")
REG = Path("runs/development/benchmark_v03/dev_registry")
LISTS = {"g_theta_nuis": ("g_theta", "nuis"), "g_theta_known": ("g_theta", "known"),
         "random_nuis": ("random", "nuis"), "random_known": ("random", "known")}
CAPS = (2, 4, 8, None)
NAME = {"g_theta_nuis": "learned relation search (lists searched under B_nuis)",
        "g_theta_known": "learned relation search (lists searched under B_known)",
        "random_nuis": "random selection (count-matched; B_nuis lists)",
        "random_known": "random selection (count-matched; B_known lists)"}


def _context(build: bool = True):
    from .dev_metric import TARGETS, load_dev_study, load_targets
    from .public_metrics import load_registry
    spec = load_targets(TARGETS)
    built = None
    if build:   # the label pass needs the tasks; the table reads the cached labels only
        study = load_dev_study(spec["study"], spec["partitions"])
        folds = core.patient_folds(study.patients, spec["n_folds"], spec["seed"])
        built = core._tasks(study, spec["targets"], folds, (), 0, spec["seed"])
    regs = {k: load_registry(REG / m, B) for k, (m, B) in LISTS.items()}
    universe = {}
    for t in spec["targets"]:
        universe[t] = sorted({s for r in regs.values() for (tt, _f), ss in r.items() if tt == t for s in ss})
    return spec, built, regs, universe


def _label_job(i):
    import torch
    torch.set_num_threads(1)
    tk, sets, path = core._G["cap"][i]
    if Path(path).exists():
        return
    sc = core.score_sets(tk, sets, Bs=("known", "decomp"))
    np.savez(path + ".tmp.npz", known_num=sc["known"][0], known_sst=sc["known"][1],
             decomp_num=sc["decomp"][0], decomp_sst=sc["decomp"][1])
    Path(path + ".tmp.npz").rename(path)
    print("labelled", Path(path).name, len(sets), flush=True)


def labels(workers: int) -> None:
    spec, built, regs, universe = _context()
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    jobs, present = [], {}
    for t, f, conds, names, st, q in built:
        index = {n: i for i, n in enumerate(names)}
        ok = [s for s in universe[t] if all(n in index for n in s)]
        present[f"{t}|{f}"] = [list(s) for s in ok]
        jobs.append((dict(conds)["real"], [tuple(index[n] for n in s) for s in ok],
                     str(OUT / "cache" / f"{t}__{f}.npz")))
    (OUT / "present.json").write_text(json.dumps(present))
    core._G["cap"] = jobs
    core._pool_map(_label_job, len(jobs), workers)


def capped(sets: list, m, k: int = 20) -> list:
    """Greedy top-k in the given order, skipping a set that would put any variable in more than m chosen sets."""
    if m is None:
        return list(sets[:k])
    c, out = Counter(), []
    for s in sets:
        if all(c[v] < m for v in s):
            out.append(s)
            c.update(s)
            if len(out) == k:
                break
    return out


def table(strict: bool = False) -> str:
    from .historical_redundancy import redundancy
    from .public_metrics import boot_mean, labels_pooled
    spec, built, regs, universe = _context(build=False)
    present = json.loads((OUT / "present.json").read_text())
    rep, ho = {}, {}
    for t in universe:
        per = {s: ([], [], [], []) for s in universe[t]}
        for f in range(spec["n_folds"]):
            z = np.load(OUT / "cache" / f"{t}__{f}.npz")
            for j, s in enumerate(map(tuple, present[f"{t}|{f}"])):
                per[s][0].append(z["known_num"][:, j])
                per[s][1].append(z["known_sst"])
                per[s][2].append(z["decomp_num"][:, j])
                per[s][3].append(z["decomp_sst"])
        for s, (kn, ks, dn, ds) in per.items():
            r = bool(labels_pooled([(np.concatenate(kn)[:, None], np.concatenate(ks))])[0]) if kn else False
            h = len(s) >= 2 and r and bool(labels_pooled([(np.concatenate(dn)[:, None], np.concatenate(ds))])[0])
            if strict:   # strict beyond-known: flagged sets are negatives (dbench.harness.dev_flags)
                from .dev_flags import set_flags
                if set_flags(t, s)["any"]:
                    r = h = False
            rep[(t, s)], ho[(t, s)] = r, h
    rows = [(k, m) for k in ("g_theta_nuis", "g_theta_known") for m in CAPS] + [("random_nuis", None), ("random_known", None)]
    keys = ("yield@20", "higher_order@20", "n_sets", "anchors", "max_multiplicity", "mean_jaccard")
    res = {}
    for k, m in rows:
        per_t = []
        for t in universe:
            vals = []
            for f in range(spec["n_folds"]):
                top = capped(regs[k].get((t, str(f)), regs[k].get((t, f), [])), m)
                rd = redundancy([list(s) for s in top])
                vals.append((sum(rep.get((t, s), False) for s in top), sum(ho.get((t, s), False) for s in top), len(top),
                             rd["anchors"] or 0, rd["max_multiplicity"] or 0, rd["mean_jaccard"] or 0.0))
            per_t.append(np.mean(np.array(vals, dtype=float), 0))
        cols = np.array(per_t).T
        res[f"{k}|{m}"] = {n: boot_mean(c) for n, c in zip(keys, cols)}

    def fmt(v):
        return "—" if v[0] is None else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    lines = ["## Development targets: anchor-multiplicity cap on the top 20 (labels at B_known"
             + (", strict beyond-known)" if strict else ")"), "",
             "40 kidney-injury targets x 3 patient folds; each list is the method's 64 registered sets in its own ranking "
             "order; the cap takes sets greedily and skips one that would put any variable in more than m chosen sets. "
             "yield@20 = sets replicating at B_known (realised held-out gain pooled over the folds' query patients, 95 % "
             "patient bootstrap lower bound > 0); higher-order@20 = those with |S| >= 2 and pooled B_decomp lower bound > 0. "
             "Mean over folds per target, 95 % bootstrap interval over the 40 targets.", "",
             "| list | cap m | sets kept | yield@20 | higher-order@20 | anchors | max mult. | Jaccard |",
             "|---|---|---|---|---|---|---|---|"]
    for k, m in rows:
        r = res[f"{k}|{m}"]
        lines.append(f"| {NAME[k]} | {m or 'none'} | {r['n_sets'][0]:.1f} | {fmt(r['yield@20'])} | "
                     f"{fmt(r['higher_order@20'])} | {fmt(r['anchors'])} | {fmt(r['max_multiplicity'])} | "
                     f"{fmt(r['mean_jaccard'])} |")
    tag = "_strict" if strict else ""
    (OUT / f"cap_table{tag}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"cap_table{tag}.json").write_text(json.dumps(res, indent=1))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("labels", "table"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "labels":
        labels(a.workers)
    else:
        print(table(strict=a.strict))


if __name__ == "__main__":
    main()
