"""Historical rediscovery: redundancy of every method's stored top-20 (no refits), beside yield@20.

  anchors          = number of distinct variables that appear in >= half of the top-20 sets
  max multiplicity = the largest number of top-20 sets any single variable appears in
  mean Jaccard     = mean pairwise Jaccard similarity of the top-20 sets (variable names as tested)

    python -m dbench.harness.historical_redundancy runs/development/benchmark_v03/historical/baselines runs/.../ours
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from .historical_yield import NAMES


def redundancy(sets: list) -> dict:
    sets = [frozenset(s) for s in sets]
    if not sets:
        return {"anchors": None, "max_multiplicity": None, "mean_jaccard": None}
    c = Counter(v for s in sets for v in s)
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
    return {"anchors": sum(n >= len(sets) / 2 for n in c.values()), "max_multiplicity": max(c.values()),
            "mean_jaccard": sum(jac) / len(jac) if jac else 0.0}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--yield-json", default="runs/development/benchmark_v03/public/historical.json")
    ap.add_argument("--out", default="runs/development/benchmark_v03/public/historical.md")
    args = ap.parse_args(argv)
    red = {}
    for d in args.dirs:
        for p in sorted(Path(d).glob("H*/*/result.json")):
            r = json.loads(p.read_text())
            if r.get("available", True) and r.get("top"):
                red[(r["task_id"], r["method"])] = redundancy([e["names"] for e in r["top"][:20]])
    rows = json.loads(Path(args.yield_json).read_text())
    sc_path = Path(args.dirs[0]).parent / "second_catalogue.json"
    strict = {(r["task"], r["method"]): r["yield_known_strict_removed"] for r in json.loads(sc_path.read_text())} if sc_path.exists() else {}
    cp = Path(args.dirs[0]).parent / "capped_view_m8.json"
    capped = json.loads(cp.read_text()) if cp.exists() else {}
    for r in rows:
        r.update(red.get((r["task"], r["method"]), {}))
    lines = ["# Historical rediscovery (hashed protocol v02; untouched studies)", "",
             "Rank of first hit = the protocol's hit rule on the top 20. yield@20 = top-20 hypotheses whose held-out general "
             "gain on the study's query patients has a 95 % bootstrap CI lower bound > 0, under the nuisance-only comparison "
             "model (depth, platform) and under the known-biology comparison model (nuisance + the study's Hallmark / Gavish "
             "programme scores minus every programme overlapping the target gene set at overlap coefficient >= 0.2, the B_known "
             "exclusion rule; these studies carry no composition view). higher-order@20 = nuisance-replicating sets of size >= 2 "
             "with G_HO CI lower bound > 0. Redundancy of the top 20: anchors = variables in >= 10 of the 20 sets; max mult. = the "
             "most sets any one variable appears in; Jaccard = mean pairwise Jaccard of the sets. Every method receives the same "
             "data and broad target and is cut to the same 64 tested sets. Under the nuisance-only model yield@20 is 15-20 of 20 for "
             "every method except count-matched random on H2 (8), so it does not separate methods; the known-biology column "
             "(0-19 of 20) is the one that discriminates as registered. Strict beyond-known (owner ruling 2026-10-23): a set counts only "
             "if it contains no excluded programme score (any version), no member gene of an excluded programme and no programme "
             "containing a gene target; under strict on H1 every method's known-passing sets contained a near-copy of the target, "
             "so the as-registered known column measured how often a method lists a target-overlapping programme; strict "
             "known-yield is 0-5 for every method on H1-H3. Learned relation search leads higher-order@20 on H3 (10) and trails "
             "EBM, GBM + H and NID on H2 (4 vs 5-10). The distinct-objects view re-presents our SAME locked top 20 with an "
             "anchor-multiplicity cap m = 8 (sets taken in list order, a set skipped if it would put any variable in more than 8 "
             "kept sets; no new search, nothing added, so fewer than 20 sets remain: its yield and higher-order entries are counts over the kept sets, not @20, and rank columns do not apply). On the "
             "kidney-injury development set the same cap applied to the 64-set lists costs -1.05 yield@20 and -1.80 "
             "higher-order@20; 90 % of the capped-away sets replicate (27 % higher-order), against 77 % of their replacements.", "",
             "| task | method | rank of first hit | hit@20 | reciprocal rank | yield@20 (nuisance) | yield@20 (known) | "
             "yield@20 (known, strict) | higher-order@20 | anchors | max mult. | Jaccard |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f="{}": "â€”" if v is None else f.format(v)  # noqa: E731
    for r in sorted(rows, key=lambda r: (r["task"], r["method"] != "learned_search", NAMES.get(r["method"], r["method"]))):
        lines.append(f"| {r['task']} | {NAMES.get(r['method'], r['method'])} | {r['rank_of_first_hit']} | "
                     f"{'yes' if r['hit@20'] else 'no'} | {r['reciprocal_rank']:.3f} | {r['yield@20']} | "
                     f"{fmt(r.get('yield@20_known'))} | {fmt(strict.get((r['task'], r['method'])))} | {r['higher_order@20']} | {fmt(r.get('anchors'))} | "
                     f"{fmt(r.get('max_multiplicity'))} | {fmt(r.get('mean_jaccard'), '{:.2f}')} |")
        if r["method"] == "learned_search":
            c = capped.get(r["task"])
            if c is None:
                lines.append(f"| {r['task']} | learned relation search, distinct-objects view (m = 8) | pending | | | | | | | | | |")
            else:
                lines.append(f"| {r['task']} | learned relation search, distinct-objects view (m = 8; {c['n_kept']} of the 20 sets kept) | "
                             f"— | — | — | {c['yield@20']} | {c['yield@20_known']} | — | {c['higher_order@20']} | {c['anchors']} | "
                             f"{c['max_multiplicity']} | {c['mean_jaccard']:.2f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
