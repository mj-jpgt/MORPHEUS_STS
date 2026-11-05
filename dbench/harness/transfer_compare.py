"""Transfer comparison (organ / platform held out of training): AUROC / AP of B_known replication over the study's
shared pool for ours, the support-fit screen, the joint-pair screen and the hybrid, with paired differences vs the
screen (per target, bootstrapped over targets).

    python -m dbench.harness.transfer_compare runs/development/benchmark_v03/transfer_pool/bladder_epithelial
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .ablation_table import boot, per_target

METHODS = [("g_theta", "learned relation search"), ("sel_hybrid", "hybrid: exact support gain - network shrinkage"),
           ("sel_screen_joint", "support-fit screen, joint pair gain"), ("sel_screen", "support-fit screen")]
REF = "sel_screen"


def main(d: str) -> str:
    d = Path(d)
    lab = json.loads((d / "labels.json").read_text())["labels"]
    res = {m: per_target(json.loads((d / f"scores_{m}_known.json").read_text())["scores"], lab)
           for m, _ in METHODS if (d / f"scores_{m}_known.json").exists()}
    f = lambda v: "—" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
    lines = [f"## Transfer: {d.name} (B_known replication over the study's shared pool)", "",
             "Δ = method minus the support-fit screen, per target, 95 % bootstrap over targets.", "",
             "| method | AUROC singles | AP singles | AUROC pairs | AP pairs | ΔAUROC singles | ΔAP singles | ΔAUROC pairs | ΔAP pairs |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, name in METHODS:
        if m not in res:
            lines.append(f"| {name} | pending | | | | | | | |")
            continue
        r, cells, dcells = res[m], [], []
        for kind in ("single", "pair"):
            for j in (0, 1):
                ts = [t for (t, k) in r if k == kind and r[(t, k)][j] is not None]
                cells.append(f(boot([r[(t, kind)][j] for t in ts])))
                ref = res.get(REF, {})
                diff = [r[(t, kind)][j] - ref[(t, kind)][j] for t in ts if (t, kind) in ref and ref[(t, kind)][j] is not None]
                dcells.append("—" if m == REF or not diff else f(boot(diff)))
        lines.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")
    (d / "transfer_compare.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print(main(sys.argv[1]))
