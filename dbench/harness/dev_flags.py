"""Development-set flags for B_known positives (no relabelling; a lookup per target):
  second_catalogue_known  a variable is a member gene of a programme that relation_v03.known_exclusions dropped for
                          this target (the gene is known biology through the excluded programme)
  target_leak             gene target G: a variable is a programme score (any spatial version) whose gene set contains G
Flagged positives are treated as not-beyond-known in the "flags removed" tables.

    python -m dbench.harness.dev_flags labels     # pool/labels_flagged.json + counts
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

B = Path("runs/development/benchmark_v03")


@lru_cache(maxsize=1)
def _study_names():
    from .dev_metric import TARGETS, load_dev_study, load_targets
    spec = load_targets(TARGETS)
    st = load_dev_study(spec["study"], spec["partitions"])
    return [str(x) for x in st.known_names], list(map(str, st.panel)), spec["targets"]


@lru_cache(maxsize=4096)
def _programme_genes(p: str) -> frozenset:
    from morpheus.tier2.substrates import hest_multi as hm
    return frozenset(map(str, hm._programme_members(p)))


@lru_cache(maxsize=256)
def target_context(target: str) -> tuple:
    from morpheus_encoder.learned_search.relation_v03 import known_exclusions
    kn, panel, _ = _study_names()
    excluded = [kn[i] for i in known_exclusions(kn, target, panel)]
    genes = frozenset().union(*[_programme_genes(p) for p in excluded]) if excluded else frozenset()
    return tuple(excluded), genes


def _var(v: str) -> tuple:
    """('gene' | 'programme' | ..., name) of a development variable (own: / nbrN: / contrast: prefixes stripped)."""
    v = re.sub(r"^(own|contrast|nbr\d+):", "", str(v))
    kind, _, name = v.partition(":")
    return kind, name


def set_flags(target: str, names) -> dict:
    excluded, genes = target_context(target)
    vs = [_var(v) for v in names]
    sc = any(k == "gene" and n in genes for k, n in vs)
    leak = False
    if str(target).startswith("gene:"):
        g = str(target).split(":", 1)[1]
        leak = any(k == "programme" and g in _programme_genes(n) for k, n in vs)
    ex = any(k == "programme" and n in set(excluded) for k, n in vs)
    return {"excluded_programme": ex, "second_catalogue_known": sc, "target_leak": leak, "any": ex or sc or leak}


def labels() -> dict:
    lab = json.loads((B / "pool" / "labels.json").read_text())
    out = json.loads(json.dumps(lab))
    counts = {k: {"pool": 0, "positives": 0, "excluded_programme": 0, "second_catalogue_known": 0, "target_leak": 0, "any": 0,
                  "pool_flagged_any": 0} for k in ("single", "pair")}
    for t, pool in lab["pools"].items():
        for kind, members in (("single", [[s] for s in pool["singles"]]), ("pair", pool["pairs"])):
            key = f"{t}|{kind}|known"
            y = list(lab["labels"][key])
            for j, s in enumerate(members):
                f = set_flags(t, s)
                c = counts[kind]
                c["pool"] += 1
                c["pool_flagged_any"] += int(f["any"])
                if y[j]:
                    c["positives"] += 1
                    for k in ("excluded_programme", "second_catalogue_known", "target_leak", "any"):
                        c[k] += int(f[k])
                    if f["any"]:
                        y[j] = False
            out["labels"][key] = y
    out["flags"] = "strict beyond-known: positives with an excluded programme score, a member gene of an excluded programme or a programme containing the gene target set to False (dbench.harness.dev_flags)"
    out["flag_counts"] = counts
    (B / "pool" / "labels_flagged.json").write_text(json.dumps(out))
    return counts


if __name__ == "__main__":
    if sys.argv[1] == "labels":
        print(json.dumps(labels()))
