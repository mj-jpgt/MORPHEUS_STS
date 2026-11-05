"""Q2: patient-disjoint release checkpoint (13c36faa) vs row-split repaired checkpoint (release_rowsplit, 139b317c, step
2000) under the public metrics: dev full pool (strict and as registered), 40-target few-shot, bladder L3; Δ = row-split
minus release, per target, 95 % bootstrap over targets.

    python -m dbench.harness.q2_compare
"""
from __future__ import annotations

import json
from pathlib import Path

from .ablation_table import boot, per_target
from .fewshot_compare import per_target as fs_per_target

B = Path("runs/development/benchmark_v03")
A, R = "g_theta", "g_theta_rowsplit"


def f(v):
    return "—" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def pool_rows(d: Path, labels: Path, title: str) -> list:
    lab = json.loads(labels.read_text())["labels"]
    res = {m: per_target(json.loads((d / f"scores_{m}_known.json").read_text())["scores"], lab) for m in (A, R)
           if (d / f"scores_{m}_known.json").exists()}
    if len(res) < 2:
        return [f"### {title}", "", "pending", ""]
    lines = [f"### {title}", "", "| | AUROC singles | AP singles | AUROC pairs | AP pairs |", "|---|---|---|---|---|"]
    for m, name in ((A, "release (patient-disjoint)"), (R, "row-split repaired")):
        cells = []
        for kind in ("single", "pair"):
            for j in (0, 1):
                cells.append(f(boot([v[j] for (t, k), v in res[m].items() if k == kind and v[j] is not None])))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = []
    for kind in ("single", "pair"):
        for j in (0, 1):
            cells.append(f(boot([res[R][(t, k)][j] - res[A][(t, k)][j] for (t, k) in res[A]
                                 if k == kind and (t, k) in res[R] and res[A][(t, k)][j] is not None and res[R][(t, k)][j] is not None])))
    lines += [f"| Δ row-split − release | " + " | ".join(cells) + " |", ""]
    return lines


def fewshot_rows() -> list:
    a, r = fs_per_target(A, 40), fs_per_target(R, 40)
    if not r:
        return ["### Few-shot, 40 targets", "", "pending", ""]
    lines = ["### Few-shot, 40 targets (AUROC; Δ row-split − release)", "",
             "| labels | support patients | release | row-split | Δ AUROC | Δ AP |", "|---|---|---|---|---|---|"]
    for kind in ("single", "pair", "single_strict", "pair_strict"):
        for n in (2, 4, 8, 0):
            ts = sorted({t for (k, met, sz, t) in a if k == kind and sz == n and met == "auroc"})
            if not ts:
                continue
            va = boot([a[(kind, "auroc", n, t)] for t in ts])
            vr = boot([r[(kind, "auroc", n, t)] for t in ts if (kind, "auroc", n, t) in r])
            da = boot([r[(kind, "auroc", n, t)] - a[(kind, "auroc", n, t)] for t in ts if (kind, "auroc", n, t) in r])
            dp = boot([r[(kind, "ap", n, t)] - a[(kind, "ap", n, t)] for t in ts if (kind, "ap", n, t) in r and (kind, "ap", n, t) in a])
            lines.append(f"| {kind} | {'all' if n == 0 else n} | {f(va)} | {f(vr)} | {f(da)} | {f(dp)} |")
    return lines + [""]


def main() -> str:
    lines = ["## Q2: patient-disjoint release vs row-split repaired checkpoint (public metrics)", "",
             "Release = runs/development/learned_search_v03/release/frozen.pt (13c36faa); row-split = release_rowsplit/frozen.pt "
             "(139b317c, step 2000). Caveat (Agent M): the row-split repair's B_known calibration error grew (0.141 -> 0.198) "
             "while the release repair's fell (0.097 -> 0.023). Both scored on identical core.support_only tasks.", ""]
    lines += pool_rows(B / "pool_v2", B / "pool" / "labels_flagged.json", "Development full pool, strict labels")
    lines += pool_rows(B / "pool_v2", B / "pool" / "labels.json", "Development full pool, as registered")
    lines += fewshot_rows()
    for s in ("bladder_epithelial", "bladder_batf3"):
        d = B / "transfer_pool" / s
        lines += pool_rows(d, d / "labels.json", f"{s} (L3{', supplementary' if s.endswith('batf3') else ''})")
    (B / "public" / "q2_rowsplit_vs_release.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main())
