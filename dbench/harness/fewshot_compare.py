"""Matched few-shot comparison (same 10 development targets): per support size, AUROC / AP of B_known replication
over the shared pool, singles and pairs, as-registered and strict, with paired differences vs the support-fit screen
(per-target mean over folds and draws, bootstrapped over targets).

    python -m dbench.harness.fewshot_compare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

F = Path("runs/development/benchmark_v03/fewshot")
OUT = Path("runs/development/benchmark_v03/public/fewshot_matched.md")
METHODS = [("g_theta", "learned relation search"), ("sel_hybrid", "hybrid: exact support gain - network shrinkage"),
           ("sel_screen_joint", "support-fit screen, joint pair gain"), ("sel_screen", "support-fit screen"),
           ("ebm_full", "EBM (full pool)")]
REF = "sel_screen"
SIZES = (2, 4, 8, 0)


def per_target(m: str, nt: int = 10) -> dict:
    p = F / f"jobs_{m}_t{nt}.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()):
        for kind in ("single", "pair", "single_strict", "pair_strict"):
            if kind in r:
                for met in ("auroc", "ap"):
                    v = r[kind][met]
                    if v is not None:
                        out.setdefault((kind, met, r["size"], r["target"]), []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if not len(v):
        return None
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(nt: int = 10) -> None:
    data = {m: per_target(m, nt) for m, _ in METHODS}
    f = lambda v: "—" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
    lines = [f"## Few-shot, matched {nt} development targets: which sets replicate, from k support patients", "",
             "AUROC / AP of B_known replication over the shared pool (5 draws per size; all = every support patient). "
             "Δ = method minus the support-fit screen, per target, 95 % bootstrap over targets. 'strict' = beyond-known "
             "positives only (sets with an excluded programme score, a member gene of one or a programme containing the gene "
             "target are negatives).", ""]
    for kind, title in (("single", "singles"), ("pair", "pairs"), ("single_strict", "singles, strict"), ("pair_strict", "pairs, strict")):
        lines += [f"### {title}", "", "| method | support patients | AUROC | AP | ΔAUROC vs screen | ΔAP vs screen |", "|---|---|---|---|---|---|"]
        for m, name in METHODS:
            d = data[m]
            for n in SIZES:
                ts = sorted({t for (k, met, sz, t) in d if k == kind and sz == n and met == "auroc"})
                if not ts:
                    continue
                cells = []
                for met in ("auroc", "ap"):
                    cells.append(f(boot([d[(kind, met, n, t)] for t in ts if (kind, met, n, t) in d])))
                for met in ("auroc", "ap"):
                    r = data[REF]
                    diff = [d[(kind, met, n, t)] - r[(kind, met, n, t)] for t in ts if (kind, met, n, t) in d and (kind, met, n, t) in r]
                    cells.append("—" if m == REF or not diff else f(boot(diff)))
                lines.append(f"| {name} | {'all' if n == 0 else n} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT if nt == 10 else OUT.with_name(f"fewshot_matched_t{nt}.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
