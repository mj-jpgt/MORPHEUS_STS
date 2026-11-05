#!/usr/bin/env python3
"""Panel k_transfer for the four evidence cards: no-refit transfer to colorectal and ependymoma cohorts (style of panel d)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_evidence_cards_v02 as v2
from plot_evidence_cards_v02 import CARDS, QA, SCRATCH, apply_morpheus_style, copy_plot_code, finish, write_manifest  # noqa: F401
from plot_evidence_cards_v01 import COLOR  # noqa: E402,F811

HERE = Path(__file__).resolve().parent
ROWS = [("valdeolivas2024_crc", "Colorectal (n = 6)"), ("ependymoma_pfa", "Ependymoma (n = 11)")]
SRC = "translation_v10/layer_p5.json + p5_known_addendum.json; frozen f_S, no refit; patient bootstrap 95% CI; 99-draw Moran-matched null"


def main():
    apply_morpheus_style(scienceplots=False)
    J = json.load(open(SCRATCH / "cards" / "p5_transfer_values.json"))
    vals = [x for g in J["cards"].values() for r in g.values() for k in ("nuis", "known") if r.get(k) for x in r[k][1:]]
    vals += [r["nuis_null_q95"] for g in J["cards"].values() for r in g.values() if r.get("nuis_null_q95") is not None]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = 0.06 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    for gid, (cid, label) in CARDS.items():
        out = HERE / f"card_{gid}_{cid}"
        rows = []
        fig, ax = plt.subplots(figsize=(4.6, 2.0), layout="constrained")
        y = np.arange(len(ROWS))[::-1]
        off = 0.14
        for yi, (ck, lab) in zip(y, ROWS):
            r = J["cards"][gid][ck]
            n, k, q = r["nuis"], r.get("known"), r.get("nuis_null_q95")
            ax.plot([n[1], n[2]], [yi + off] * 2, color=COLOR, linewidth=1.2, solid_capstyle="butt")
            ax.plot(n[0], yi + off, "o", color=COLOR, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
            if k:
                ax.plot([k[1], k[2]], [yi - off] * 2, color=COLOR, linewidth=1.2, solid_capstyle="butt")
                ax.plot(k[0], yi - off, "s", markerfacecolor="white", markeredgecolor=COLOR, markeredgewidth=1.1, markersize=5)
            if q is not None:
                ax.plot([q, q], [yi - 0.38, yi + 0.38], color="0.45", linewidth=1.0, zorder=1)
            rows.append({"relation": gid, "row": lab, "cohort": ck, "n_patients": J["cohorts"][ck]["n_patients"],
                         "gain_nuis": n[0], "nuis_lo": n[1], "nuis_hi": n[2],
                         "gain_known": k[0] if k else np.nan, "known_lo": k[1] if k else np.nan, "known_hi": k[2] if k else np.nan,
                         "nuis_null_q95": np.nan if q is None else q,
                         "known_null_q95": np.nan if r.get("known_null_q95") is None else r["known_null_q95"],
                         "positive_control_gain": J["positive_control"][ck][0], "positive_control_lo": J["positive_control"][ck][1],
                         "positive_control_hi": J["positive_control"][ck][2]})
        ax.axvline(0, color="0.45", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_yticks(y, [lab for _, lab in ROWS])
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Held-out gain (R$^2$)")
        ax.grid(axis="x", color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)
        h = [Line2D([], [], marker="o", color=COLOR, linewidth=0, markersize=5, label="Over nuisance"),
             Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=COLOR, linewidth=0, markersize=5,
                    label="Over known biology"),
             Line2D([], [], color="0.45", linewidth=1.0, label="Matched null 95th pct.")]
        ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        finish(fig, out, "k_transfer", pd.DataFrame(rows))
        # manifest, README, QA
        mp = out / "figure_manifest.json"
        man = json.load(open(mp))
        man["panels"]["k_transfer"] = [f"k_transfer.{e}" for e in ("pdf", "svg", "png")] + ["k_transfer_plot_data.parquet"]
        man["source_files"] = sorted(set(man["source_files"]) | {str(SCRATCH / "cards" / "p5_transfer_values.json")})
        man["k_transfer_source"] = SRC
        man["scale_rule_v02"]["k_transfer"] = f"x-limits shared across the four cards: {xlim[0]:.3f} to {xlim[1]:.3f}; zero dashed; positive-control values in plot data, not drawn"
        man["generated_by"] = sorted(set(man["generated_by"]) | {str(Path(__file__).resolve())})
        notes = [n for n in man.get("notes", []) if not n.startswith("k_transfer")]
        notes.append("k_transfer: not-testable cohorts - GSE294385: " + J["not_testable"]["gse294385"] +
                     "; Ji 2020: " + J["not_testable"]["ji2020_cscc"])
        notes.append("k_transfer: null tick omitted where the matched-null 95th percentile is null; "
                     + ("known-biology gain not computed for this relation, only nuisance markers drawn" if gid in ("G5", "G7")
                        else "both gains drawn"))
        man["notes"] = notes
        write_manifest(out, man)
        rd = out / "README.md"
        t = rd.read_text(encoding="utf-8")
        t = "\n".join(l for l in t.split("\n") if not l.startswith("Panel k shows")).rstrip() + "\n"
        t += ("\nPanel k shows no-refit transfer to two other cancers; the not-testable cohorts (GSE294385, Ji 2020) are listed "
              "in the manifest notes with the reason strings from the source JSON.\n")
        rd.write_text(t, encoding="utf-8")
        qa = out / "QA_REPORT.md"
        q = QA[f"{out.name}/k_transfer"]
        t = "\n".join(l for l in qa.read_text(encoding="utf-8").split("\n") if "k_transfer" not in l).rstrip() + "\n"
        t += (f"- k_transfer: shared x-limits {xlim[0]:.3f} to {xlim[1]:.3f}; legend outside axes; overlaps {len(q['overlap_issues'])}, "
              f"non-white {q['nonwhite_fraction']}; positive control in plot data only\n")
        qa.write_text(t, encoding="utf-8")
        copy_plot_code(out, v2.__file__)
        shutil_k = out / "plot_code_k.py"
        shutil_k.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        print("done", gid, flush=True)
    qp = HERE / "evidence_cards_v02_qa.json"
    allq = json.load(open(qp))
    allq.update({k: v for k, v in QA.items() if k.endswith("/k_transfer")})
    qp.write_text(json.dumps(allq, indent=1, default=str))
    print(xlim)


if __name__ == "__main__":
    main()
