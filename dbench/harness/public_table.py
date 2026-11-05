"""The paper-facing development retrieval table: full-pool AUROC / AP (``pool``) beside precision@k /
recall@k of each method's own top-k list (``public_metrics.dev_table``), by published method name.

    python -m dbench.harness.public_table --pk runs/development/benchmark_v03/public/pk_all/dev_public_known.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .public_metrics import display

POOL = Path("runs/development/benchmark_v03/pool")
OUT = Path("runs/development/benchmark_v03/public")
ORDER = ("g_theta", "g_theta_lcb", "ebm", "gbm_h", "nid", "misty", "rulefit", "pysr", "tabdistill", "tabdistill_reduced", "linear_screen", "random")
#: methods whose own scores define a value for every single and every pair of the pool (full-pool scorers)
FULL_POOL = {"ebm": "ebm_full", "gbm_h": "gbm_h_full"}
#: methods that score only the members they surface (their pool AUROC / AP would be pinned near 0.5 by ties)
PARTIAL = ("rulefit", "pysr", "tabdistill", "tabdistill_reduced", "linear_screen")


def f(t, nd=3):
    return "-" if not t or t[0] is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pk", required=True, help="dev_public_known.json (precision@k / recall@k on the B_known lists)")
    args = ap.parse_args(argv)
    known = json.loads((POOL / "pool_table_known.json").read_text())
    nuis = json.loads((POOL / "pool_table_nuis.json").read_text()) if (POOL / "pool_table_nuis.json").exists() else {"rows": {}}
    pk = json.loads(Path(args.pk).read_text())["methods"]
    sh = known["positive_share"]
    methods = [m for m in ORDER if m in known["rows"] or m in pk]
    head = ("| method | AUROC singles | AP singles | AUROC pairs | AP pairs | AUROC pairs B_decomp | AP pairs B_decomp | "
            "P@10 | P@20 | R@10 | R@20 |")
    lines = ["# Replication retrieval on the development targets (kidney_injury, 40 targets x 3 patient folds)", "",
             f"Primary labels: a relation replicates if its realised held-out gain beyond B_known (nuisance + programme "
             f"scores excluding the target and overlapping programmes + CellViT PanNuke composition) is above zero "
             f"(pooled over the folds' query patients, 95 % patient bootstrap); pairs also beyond B_decomp (best additive "
             f"decomposition). Pool: every candidate shared by the three folds plus 256 seeded pairs per target; positive "
             f"share: singles {sh['single|known']:.3f}, pairs {sh['pair|known']:.3f} (B_known), pairs {sh['pair|decomp']:.3f} "
             f"(B_decomp). AUROC / AP: every method scores every pool member (members a method does not score rank last, "
             f"ties random). P@k / R@k: each method's own top-k list searched under B_known (R@k against every replicating "
             f"set any compared method listed). Intervals: 95 % bootstrap over targets. EBM and gradient boosting are scored on "
             f"the full pool with their own statistics (EBM main-term importance and FAST interaction strength; permutation "
             f"importance and Friedman's H). RuleFit, PySR, TabPFN-v2 + SPEX and the linear screen produce a score only for "
             f"the few members they surface (2-5 % of singles, about 0.1 % of pairs), so a full-pool AUROC / AP is not "
             f"defined for them (ties would pin it near 0.5); they are compared on their own lists (P@k, R@k). B_decomp "
             f"labels agree poorly across the three patient folds (Cohen kappa 0.03 vs 0.25 for B_known): at these effect "
             f"sizes the higher-order label is close to noise, so the B_decomp columns are not evidence about any method.",
             "", head, "|---" * 11 + "|"]
    for m in methods:
        r = known["rows"].get(FULL_POOL.get(m, m), {})
        if m in PARTIAL:
            r = {}
        if m in ("misty",):   # scores every single but no pair (a variable-importance method)
            r = {k: v for k, v in r.items() if k.startswith("single")}
        p = pk.get(m, {})
        lines.append(f"| {display(m)} | {f(r.get('single@B_known', {}).get('auroc'))} | {f(r.get('single@B_known', {}).get('ap'))} | "
                     f"{f(r.get('pair@B_known', {}).get('auroc'))} | {f(r.get('pair@B_known', {}).get('ap'))} | "
                     f"{f(r.get('pair@B_decomp', {}).get('auroc'))} | {f(r.get('pair@B_decomp', {}).get('ap'))} | "
                     f"{f(p.get('precision@10'))} | {f(p.get('precision@20'))} | {f(p.get('recall@10'))} | {f(p.get('recall@20'))} |")
    if nuis["rows"]:
        shn = nuis.get("positive_share", sh)
        lines += ["", f"Supplementary, lenient labels at B_nuis (positive share singles {shn['single|nuis']:.3f}, pairs "
                      f"{shn['pair|nuis']:.3f}):", "", "| method | AUROC singles | AP singles | AUROC pairs | AP pairs |",
                  "|---|---|---|---|---|"]
        for m in [m for m in ORDER if m in nuis["rows"]]:
            r = nuis["rows"][m]
            lines.append(f"| {display(m)} | {f(r.get('single@B_nuis', {}).get('auroc'))} | {f(r.get('single@B_nuis', {}).get('ap'))} | "
                         f"{f(r.get('pair@B_nuis', {}).get('auroc'))} | {f(r.get('pair@B_nuis', {}).get('ap'))} |")
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (OUT / "dev_retrieval.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
