"""second_catalogue_known: a set is flagged when any of its variables is a member gene of a programme that the B_known
rule excluded for overlap with the target (the gene is then known biology through the excluded programme even though
the comparison model cannot contain it).  Strict variant: also flags a variable that IS an excluded programme score.

    python -m dbench.harness.second_catalogue historical      # per-set rows from historical/per_set/H*.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

H = Path("runs/development/benchmark_v03/historical")


def members(programmes) -> set:
    from morpheus.tier2.substrates import hest_multi as hm
    out = set()
    for p in programmes:
        out |= {str(g) for g in hm._programme_members(str(p))}
    return out


def base(v: str) -> tuple:
    """('genes' | 'programmes' | other, name) of a variable name after stripping neighbourhood transforms."""
    v = re.sub(r"^(nbr\d+:)+", "", str(v))
    kind, _, name = v.partition(":")
    return kind, name


def flags(names, dropped, genes) -> tuple:
    gene = any(base(v)[0] == "genes" and base(v)[1] in genes for v in names)
    prog = any(base(v)[0] == "programmes" and base(v)[1] in set(map(str, dropped)) for v in names)
    return gene, gene or prog


def historical() -> str:
    from .historical_yield import NAMES
    rows = []
    for t in ("H1", "H2", "H3"):
        p = H / "per_set" / f"{t}.json"
        if p.exists():
            rows += json.loads(p.read_text())
    lines = ["| task | method | yield@20 (known) | flagged (member gene of an excluded programme) | known, flagged removed | "
             "flagged incl. excluded programme scores | known, strict removal |", "|---|---|---|---|---|---|---|"]
    out = []
    for r in sorted(rows, key=lambda r: (r["task"], r["method"] != "learned_search", r["method"])):
        g = members(r["known_programmes_dropped"])
        kn = [s for s in r["sets"] if s["yield_known"]]
        f1 = [s for s in kn if flags(s["names"], r["known_programmes_dropped"], g)[0]]
        f2 = [s for s in kn if flags(s["names"], r["known_programmes_dropped"], g)[1]]
        rec = {"task": r["task"], "method": r["method"], "yield_known": len(kn), "flagged_gene": len(f1),
               "yield_known_gene_removed": len(kn) - len(f1), "flagged_strict": len(f2), "yield_known_strict_removed": len(kn) - len(f2),
               "check_matches_count": len(kn) == r["yield@20_known"]}
        out.append(rec)
        lines.append(f"| {r['task']} | {NAMES.get(r['method'], r['method'])} | {len(kn)} | {len(f1)} | {len(kn) - len(f1)} | "
                     f"{len(f2)} | {len(kn) - len(f2)} |")
    (H / "second_catalogue.json").write_text(json.dumps(out, indent=1))
    (H / "second_catalogue.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "historical":
        print(historical())
