"""Benchmark v05 figure: AP of relation ranking, ours vs 13 baselines, three untouched test studies.

Reads table1_<study>.json written by dbench/benchmark_v05 (W3). Interim tables until the 12:00 UTC
hard stop; rerun unchanged when the final tables land.

Panels (separate files, no statistics text on the panels, per the figure rules):
  a  AP singles per method, three studies as columns; dot = mean, bar = 95% hierarchical bootstrap CI;
     hollow marker = method flagged incomplete (>10% of splits timed out); chance (random AP) as a dotted line.
  b  AP pairs, same layout.
  c  paired difference ours minus method, AP pairs on shared targets, 95% CI; the headline panel.
  d  paired difference ours minus method, AP singles on shared targets.
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE

STUDIES = [("table1_glioma.json", "Glioma (GSE237183)"),
           ("table1_nakajima.json", "LUAD (Takano 2024)"),
           ("table1_gse307534.json", "Lung precursor (GSE307534)")]

# display order and names; ours first, then baselines grouped by family
METHODS = [
    ("sel_screen", "ours: support-fit ordering"),
    ("sel_screen_joint", "ours: joint pair gain (suppl.)"),
    ("ebm", "EBM"),
    ("nid", "NID"),
    ("linear_screen", "linear interaction screen"),
    ("gbm_shap", "GBM + SHAP interactions"),
    ("rulefit", "RuleFit"),
    ("irf", "iRF"),
    ("knockoffs", "Model-X knockoffs"),
    ("stability_selection", "stability selection"),
    ("misty", "MISTy-style forest"),
    ("gbm_h", "GBM + Friedman H"),
    ("pysr", "PySR (shortlist)"),
    ("random", "random"),
]
OURS = {"sel_screen", "sel_screen_joint"}
C_OURS = "#1f5fa8"      # blue
C_BASE = "#6b6b6b"      # grey
C_CHANCE = "#b05a2a"    # muted orange for the chance line

def load():
    tabs = []
    for fn, title in STUDIES:
        d = json.load(open(DATA / fn))
        tabs.append((title, d))
    return tabs

def incomplete(d, m):
    ex = d["extra"].get(m, {}).get("single", {})
    att, to = ex.get("splits_attempted", 0), ex.get("timeout_or_error_splits", 0)
    return att > 0 and to / att > 0.10

def panel_ap(tabs, key, fname):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    for ax, (title, d) in zip(axes, tabs):
        ys = np.arange(len(METHODS))[::-1]
        for y, (m, name) in zip(ys, METHODS):
            r = d["rows"].get(m, {}).get(key)
            if not r:
                continue
            col = C_OURS if m in OURS else C_BASE
            hollow = incomplete(d, m)
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.6, solid_capstyle="butt")
            ax.plot(r["mean"], y, marker="o", ms=6.5, color=col,
                    mfc=("white" if hollow else col), mew=1.6)
        chance = d["rows"]["random"][key]["mean"]
        ax.axvline(chance, color=C_CHANCE, ls=":", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 0.8)
        ax.set_xlabel("average precision (95% CI)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(np.arange(len(METHODS))[::-1])
    axes[0].set_yticklabels([n for _, n in METHODS], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=250)
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)

def panel_diff(tabs, key, fname):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    ms = [(m, n) for m, n in METHODS if m != "sel_screen"]
    for ax, (title, d) in zip(axes, tabs):
        ys = np.arange(len(ms))[::-1]
        for y, (m, name) in zip(ys, ms):
            r = d["rows"].get(m, {}).get(key)
            if not r or r.get("diff_ours_minus") is None:
                continue
            col = C_OURS if m in OURS else C_BASE
            hollow = incomplete(d, m)
            ax.plot([r["diff_lo"], r["diff_hi"]], [y, y], color=col, lw=1.6)
            ax.plot(r["diff_ours_minus"], y, marker="o", ms=6.5, color=col,
                    mfc=("white" if hollow else col), mew=1.6)
        ax.axvline(0, color="#333333", lw=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.3, 0.9)
        ax.set_xlabel("ours minus method, AP (95% CI)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(np.arange(len(ms))[::-1])
    axes[0].set_yticklabels([n for _, n in ms], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=250)
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)

def qa(tabs):
    rows = []
    for title, d in tabs:
        for m, name in METHODS:
            for key in ("ap_single", "ap_pair"):
                r = d["rows"].get(m, {}).get(key)
                if r:
                    rows.append(dict(study=title, method=m, metric=key, mean=r["mean"], lo=r["lo"], hi=r["hi"],
                                     diff=r.get("diff_ours_minus"), diff_lo=r.get("diff_lo"), diff_hi=r.get("diff_hi"),
                                     incomplete=incomplete(d, m)))
    json.dump(rows, open(OUT / "benchmark_v05_qa.json", "w"), indent=1)

if __name__ == "__main__":
    tabs = load()
    panel_ap(tabs, "ap_single", "a_ap_singles")
    panel_ap(tabs, "ap_pair", "b_ap_pairs")
    panel_diff(tabs, "ap_pair", "c_diff_pairs")
    panel_diff(tabs, "ap_single", "d_diff_singles")
    qa(tabs)
    print("done")


# ---- pooled panels and operator ablation (added after the final tables landed, 2026-11-03) ----
def panel_pooled(fname="e_pooled"):
    d = json.load(open(DATA / "table1_pooled_3studies.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=True)
    for ax, key, title in zip(axes, ("ap_single", "ap_pair"), ("single variables", "variable pairs")):
        ys = np.arange(len(METHODS))[::-1]
        for y, (m, name) in zip(ys, METHODS):
            r = d["rows"].get(m, {}).get(key)
            if not r:
                continue
            col = C_OURS if m in OURS else C_BASE
            hollow = incomplete(d, m)
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.6)
            ax.plot(r["mean"], y, marker="o", ms=6.5, color=col, mfc=("white" if hollow else col), mew=1.6)
        ax.axvline(d["rows"]["random"][key]["mean"], color=C_CHANCE, ls=":", lw=1.2)
        ax.set_title(f"pooled, three test studies: {title}", fontsize=10)
        ax.set_xlim(0, 0.6)
        ax.set_xlabel("average precision (95% CI)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(np.arange(len(METHODS))[::-1])
    axes[0].set_yticklabels([n for _, n in METHODS], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=250); fig.savefig(OUT / f"{fname}.pdf"); plt.close(fig)

ABL = [("none", 161, 28, 10), ("compartment context", 111, 12, 10), ("heterogeneity / gradient", 150, 17, 1),
       ("residual-programme targets", 111, 28, 10), ("ring decay", 111, 27, 9)]
# the "all morphology-involving candidates" row of ablation_operators.md is omitted: 0 of the 200 frozen v08 candidates
# carry an H-Optimus concept or morphology-dictionary variable, so that row removes nothing (W3, 2026-11-03).

def panel_ablation(fname="f_operator_ablation"):
    # counts transcribed from data/final/ablation_operators.md (registered v08 renal catalogue, 200 frozen candidates)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    cols = [("internally supported", 1), ("beyond known biology", 2), ("externally replicated", 3)]
    ys = np.arange(len(ABL))[::-1]
    for ax, (title, j) in zip(axes, cols):
        vals = [row[j] for row in ABL]
        ax.barh(ys, vals, color=[C_OURS if row[0] == "none" else C_BASE for row in ABL], height=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("relations remaining")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(["removed: " + r[0] if r[0] != "none" else "full catalogue" for r in ABL], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=250); fig.savefig(OUT / f"{fname}.pdf"); plt.close(fig)

if __name__ == "__main__":
    panel_pooled()
    panel_ablation()
