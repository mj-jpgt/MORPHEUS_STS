#!/usr/bin/env python3
"""Evidence-card panels (a-f) for four highlighted MORPHEUS relations (G3, G4, G5, G7).

Reads the per-relation card inputs (characterisation_v10/cards, W1d crops) and writes one bundle per relation.
No statistics are printed on any panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

SKILL = Path(r"C:\Users\mobar\OneDrive\biorag\.agents\skills\morpheus-publication-figures\scripts")
sys.path.insert(0, str(SKILL))
from export_figure import copy_plot_code, export_figure, save_plot_data, write_manifest  # noqa: E402
from morpheus_style import apply_morpheus_style  # noqa: E402

COLOR = "#E69F00"
CARDS = {
    "G3": ("71ed4499fb99", "EMT-I gradient <- myogenesis het. + angiogenesis + EMT IV"),
    "G4": ("67256d9e53d5", "EMT-I heterogeneity <- apical-junction het. + angiogenesis + EMT IV"),
    "G5": ("93b02a9038ea", "Glutathione gradient <- hypoxia grad. + SLC3A1"),
    "G7": ("caf32d321d27", "Glutathione gradient <- hypoxia het. + SLC3A1"),
}
SHORT = {"het:programmes:hallmark:HALLMARK_MYOGENESIS": "Myogenesis het.",
         "het:programmes:hallmark:HALLMARK_APICAL_JUNCTION": "Apical-junction het.",
         "programmes:hallmark:HALLMARK_ANGIOGENESIS": "Angiogenesis",
         "programmes:mp:EMT IV": "EMT IV",
         "grad:programmes:mp:Hypoxia": "Hypoxia grad.",
         "het:programmes:mp:Hypoxia": "Hypoxia het.",
         "raw_genes:SLC3A1": "SLC3A1"}
LADDER_ROWS = [("ladder_d0", "Nuisance only"), ("ladder_d1", "+ known biology"),
               ("ladder_d2", "+ composition"), ("ladder_d3", "+ both")]
COHORTS = [("gse250163", "GSE250163"), ("li2022_ccrcc", "Li 2022"), ("zenodo14620362", "Zenodo")]
ZLAB = "Within-patient z"


def sym_lim(*arrs, q=98):
    return float(np.percentile(np.abs(np.concatenate([np.ravel(a) for a in arrs])), q))


def panel_a(spot, sample, out):
    from scipy.spatial import cKDTree
    m = spot["sample_id"] == sample
    xy = spot["coords"][m]
    t, s = spot["target_std"][m], spot["state"][m]
    lim = sym_lim(t, s)
    df = pd.DataFrame({"sample_id": sample, "barcode": spot["barcode"][m], "x_px": xy[:, 0], "y_px": xy[:, 1],
                       "target_std": t, "state": s})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="constrained")
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    sc = None
    for ax, v in zip(axes, (t, s)):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=4, marker="o",
                        linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(sc, ax=axes, location="right", shrink=0.7, pad=0.02)
    cb.set_label(ZLAB)
    fig.canvas.draw()
    bb = axes[0].get_window_extent()
    xr = np.ptp(xy[:, 0]) + spacing
    yr = np.ptp(xy[:, 1]) + spacing
    ppp = min(bb.width / xr, bb.height / yr) * 72 / fig.dpi  # points per pixel
    for ax in axes:
        for coll in ax.collections:
            coll.set_sizes([(spacing * ppp) ** 2])
    export_figure(fig, out, stem="a_spatial")
    save_plot_data(df, out, "a_spatial_plot_data")
    plt.close(fig)
    return {"colour_limit": lim, "n_spots": int(m.sum()), "spot_spacing_px": spacing}


def panel_b(rs, out):
    names = rs["names"]
    n = len(names)
    fig = plt.figure(figsize=(2.0 * n + 3.2, 2.5), layout="constrained")
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.25])
    axs = []
    rows = []
    ymin = min(min(rs["one_d"][k]["partial_dependence"]) for k in names)
    ymax = max(max(rs["one_d"][k]["partial_dependence"]) for k in names)
    pad = 0.05 * (ymax - ymin)
    for i, k in enumerate(names):
        ax = fig.add_subplot(gs[0, i], sharey=axs[0] if axs else None)
        g = np.array(rs["one_d"][k]["grid"])
        p = np.array(rs["one_d"][k]["partial_dependence"])
        ax.plot(g, p, color=COLOR, linewidth=1.6)
        ax.axhline(0, color="0.45", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_xlabel(SHORT[k] + " (z)")
        if i == 0:
            ax.set_ylabel("f(S)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(color="0.93", linewidth=0.5)
        ax.set_axisbelow(True)
        axs.append(ax)
        rows += [{"panel": "1d", "variable": SHORT[k], "grid": a, "partial_dependence": b} for a, b in zip(g, p)]
    t = rs["two_d"]
    xg, yg, S = np.array(t["x_grid"]), np.array(t["y_grid"]), np.array(t["surface"]).T  # stored [x, y]
    lim = float(np.abs(S).max())
    ax = fig.add_subplot(gs[0, n])
    cs = ax.contourf(xg, yg, S, levels=np.linspace(-lim, lim, 15), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xlabel(SHORT[t["x"]] + " (z)")
    ax.set_ylabel(SHORT[t["y"]] + " (z)")
    cb = fig.colorbar(cs, ax=ax, location="right", pad=0.03)
    cb.set_label("f(S)")
    cb.set_ticks(np.round(np.linspace(-lim, lim, 5), 2))
    for i, xv in enumerate(xg):
        for j, yv in enumerate(yg):
            rows.append({"panel": "2d", "variable": f"{SHORT[t['x']]} x {SHORT[t['y']]}", "x": xv, "y": yv,
                         "surface": S[j, i]})
    export_figure(fig, out, stem="b_response")
    save_plot_data(pd.DataFrame(rows), out, "b_response_plot_data")
    plt.close(fig)
    return {"surface_limit": lim, "pair": [SHORT[t["x"]], SHORT[t["y"]]]}


def panel_c(ladder, out):
    df = pd.DataFrame([{"row": lab, "estimate": ladder[k]["estimate"], "lo": ladder[k]["ci"][0],
                        "hi": ladder[k]["ci"][1]} for k, lab in LADDER_ROWS])
    fig, ax = plt.subplots(figsize=(4.2, 2.1), layout="constrained")
    y = np.arange(len(df))[::-1]
    for yi, r in zip(y, df.itertuples()):
        ax.plot([r.lo, r.hi], [yi, yi], color=COLOR, linewidth=1.2, solid_capstyle="butt")
        ax.plot(r.estimate, yi, "o", color=COLOR, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
    ax.axvline(0, color="0.45", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_yticks(y, df.row)
    ax.set_ylim(-0.6, len(df) - 0.4)
    lo = min(0, df.lo.min())
    hi = df.hi.max()
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlabel("Held-out gain over baseline (R$^2$)")
    ax.grid(axis="x", color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)
    export_figure(fig, out, stem="c_ladder")
    save_plot_data(df, out, "c_ladder_plot_data")
    plt.close(fig)


def panel_d(forest, out):
    rows = [{"row": "Pooled", "gain_nuis": forest["pooled"]["nuis"]["gain"],
             "nuis_lo": forest["pooled"]["nuis"]["ci"][0], "nuis_hi": forest["pooled"]["nuis"]["ci"][1],
             "gain_known": forest["pooled"]["known"]["gain"],
             "known_lo": forest["pooled"]["known"]["ci"][0], "known_hi": forest["pooled"]["known"]["ci"][1],
             "n_patients": sum(forest["per_cohort"][c]["n_patients"] for c, _ in COHORTS)}]
    for c, lab in COHORTS:
        pc = forest["per_cohort"][c]
        rows.append({"row": lab, "gain_nuis": pc["gain_nuis"], "gain_known": pc["gain_known"],
                     "n_patients": pc["n_patients"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.6, 2.2), layout="constrained")
    y = np.arange(len(df))[::-1]
    off = 0.14
    for yi, r in zip(y, df.itertuples()):
        if r.row == "Pooled":
            ax.plot([r.nuis_lo, r.nuis_hi], [yi + off] * 2, color=COLOR, linewidth=1.2, solid_capstyle="butt")
            ax.plot([r.known_lo, r.known_hi], [yi - off] * 2, color=COLOR, linewidth=1.2, solid_capstyle="butt")
        ax.plot(r.gain_nuis, yi + off, "o", color=COLOR, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(r.gain_known, yi - off, "s", markerfacecolor="white", markeredgecolor=COLOR, markeredgewidth=1.1,
                markersize=5)
    ax.axvline(0, color="0.45", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_yticks(y, df.row)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel("Held-out gain (R$^2$)")
    ax.grid(axis="x", color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)
    h = [Line2D([], [], marker="o", color=COLOR, linewidth=0, markersize=5, label="Over nuisance"),
         Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=COLOR, linewidth=0, markersize=5,
                label="Over known biology")]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    export_figure(fig, out, stem="d_external")
    save_plot_data(df, out, "d_external_plot_data")
    plt.close(fig)


def patient_colors(patients):
    uniq = sorted(set(patients))
    pal = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors[:max(0, len(uniq) - 20)])
    return {p: pal[i % len(pal)] for i, p in enumerate(uniq)}, len(uniq)


def panel_e(um, out):
    xy = um["umap"]
    st, neo, pat = um["state"], um["neoplastic"], um["patient"]
    lim = sym_lim(st)
    cmap, npat = patient_colors(pat)
    order = np.random.default_rng(0).permutation(len(st))
    fig = plt.figure(figsize=(8.4, 3.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.06])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    kw = dict(s=1.5, linewidths=0, rasterized=True)
    sc0 = axs[0].scatter(xy[order, 0], xy[order, 1], c=st[order], cmap="RdBu_r", vmin=-lim, vmax=lim, **kw)
    sc1 = axs[1].scatter(xy[order, 0], xy[order, 1], c=neo[order], cmap="viridis", vmin=0, vmax=1, **kw)
    cols = np.array([cmap[p] for p in pat])
    axs[2].scatter(xy[order, 0], xy[order, 1], c=cols[order], **kw)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.set_xlabel("UMAP 1")
    axs[0].set_ylabel("UMAP 2")
    for sc, i, lab in [(sc0, 0, "State (within-patient z)"), (sc1, 1, "Neoplastic fraction")]:
        cax = fig.add_subplot(gs[1, i])
        fig.colorbar(sc, cax=cax, orientation="horizontal").set_label(lab)
    fig.add_subplot(gs[1, 2]).axis("off")
    df = pd.DataFrame({"index": um["index"], "umap1": xy[:, 0], "umap2": xy[:, 1], "state": st,
                       "neoplastic": neo, "patient": pat, "draw_rank": np.argsort(order)})
    export_figure(fig, out, stem="e_umap")
    save_plot_data(df, out, "e_umap_plot_data")
    plt.close(fig)
    return {"n_patients": npat, "state_limit": lim}


def trim_left(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    a = np.asarray(bg.convert("RGB"))
    left = a[:, : a.shape[1] // 2 - 4]
    m = (left < 245).any(axis=2)
    m[:24] = False  # title row
    rows = np.where(m.sum(1) > 20)[0]
    cols = np.where(m.sum(0) > 20)[0]
    return left[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def panel_f(cdir, out):
    idx = json.load(open(cdir / "index.json"))
    pairs = sorted({r["pair"] for r in idx})
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.35 * len(pairs), 2.9), layout="constrained")
    rows = []
    for r in idx:
        j = pairs.index(r["pair"])
        i = 0 if r["side"] == "high" else 1
        img = trim_left(cdir / r["image"])
        axes[i, j].imshow(img)
        rows.append({"pair": r["pair"], "side": r["side"], "sample": r["sample"], "barcode": r["barcode"],
                     "state": r["state"], "n_nuclei": r["n_nuclei_in_patch"], "crop_h_px": img.shape[0],
                     "crop_w_px": img.shape[1]})
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, 0].set_ylabel("High state")
    axes[1, 0].set_ylabel("Low state")
    export_figure(fig, out, stem="f_crops")
    save_plot_data(pd.DataFrame(rows), out, "f_crops_plot_data")
    plt.close(fig)


def readme(gid, cid, label, spot, forest, sample, info, path):
    pc = forest["per_cohort"]
    npat = len(set(spot["patient"]))
    extra = ""
    if gid in ("G5", "G7"):
        extra = (f"\nExternal replication is carried by GSE250163 (gain over nuisance {pc['gse250163']['gain_nuis']:.3f}) "
                 f"and Zenodo ({pc['zenodo14620362']['gain_nuis']:.3f}); Li 2022 per-cohort gain is near zero "
                 f"({pc['li2022_ccrcc']['gain_nuis']:.3f} over nuisance, {pc['li2022_ccrcc']['gain_known']:.3f} "
                 f"over known biology; forest.json).\n")
    txt = f"""# Evidence card {gid} ({cid}): {label}

Question: does the relation add held-out predictive signal for the target beyond nuisance covariates and known
biology, is it carried by composition, and does it hold in independent cohorts without refitting?

Sources: characterisation_v10/cards (spot_values, umap, ladder, forest, response_surface) and W1d crops
(composition-matched H&E pairs; clean left half extracted from the card render at about 256 px; CellViT boxes omitted).

Independent unit: patient ({npat} discovery patients). Spots appear only for visualisation.

Selection rule: spatial section = the section with the most scored spots ({sample}, {info['n_spots']} spots; same
for all four cards). UMAP = frozen upstream coordinates (20,000 spots), not recomputed. Crops = the six
composition-matched high/low pairs stored in the card.

Metrics: held-out gain over baseline (R2); partial dependence f(S) on within-patient z; target and state are
within-patient z.

Uncertainty: patient-bootstrap 95% intervals (ladder; pooled rows of the external panel). Per-cohort gains carry no interval.

Supports: the state tracks the target on a real section; the response surface shows the shape of the relation;
the ladder shows how much of the nuisance-only gain is retained after known biology and composition;
the external panel shows no-refit transfer to three ccRCC cohorts; the crops show composition-matched high and low tissue.
{extra}
Does not establish: causation, mechanism, novelty beyond known biology, a physical length scale (pixel size not
certified; no scale bar), or that UMAP separation validates the state. Coordinates are read as (x, y) = (column, row).
"""
    path.write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    apply_morpheus_style(scienceplots=False)
    spots = {g: dict(np.load(a.cards / c / "spot_values.npz", allow_pickle=True)) for g, (c, _) in CARDS.items()}
    best = {}
    for g, s in spots.items():
        u, n = np.unique(s["sample_id"], return_counts=True)
        best[g] = (str(u[n.argmax()]), int(n.max()))
    samples = {b[0] for b in best.values()}
    assert len(samples) == 1, best
    sample = samples.pop()
    for g, (cid, label) in CARDS.items():
        d = a.cards / cid
        out = a.out / f"card_{g}_{cid}"
        out.mkdir(parents=True, exist_ok=True)
        rs = json.load(open(d / "response_surface.json"))
        ladder = json.load(open(d / "ladder.json"))
        forest = json.load(open(d / "forest.json"))
        um = dict(np.load(d / "umap.npz", allow_pickle=True))
        ia = panel_a(spots[g], sample, out)
        ib = panel_b(rs, out)
        panel_c(ladder, out)
        panel_d(forest, out)
        ie = panel_e(um, out)
        panel_f(d / "crops", out)
        readme(g, cid, label, spots[g], forest, sample, ia, out / "README.md")
        panels = {k: [f"{k}.pdf", f"{k}.svg", f"{k}.png", f"{k}_plot_data.*"] for k in
                  ("a_spatial", "b_response", "c_ladder", "d_external", "e_umap", "f_crops")}
        write_manifest(out, {
            "figure_id": f"fig6_evidence_card_{g}_{cid}", "relation": label,
            "source_files": [str(d / n) for n in ("spot_values.npz", "umap.npz", "ladder.json", "forest.json",
                                                  "response_surface.json", "crops/index.json")],
            "source_version": "characterisation_v10/cards; W1d crops",
            "independent_unit": "patient; spots for visualisation only",
            "plot_family": "evidence card (spatial, response, ladder, forest, embedding, contact sheet)",
            "effect_metric": "held-out gain over baseline (R2)",
            "uncertainty": "patient-bootstrap 95% interval (ladder, pooled external rows); per-cohort gains have no interval",
            "selection_rule": f"spatial section = most scored spots ({sample}, {ia['n_spots']}), identical for all four "
                              "cards; UMAP frozen upstream (20,000 spots); crops = six stored composition-matched pairs",
            "scale_rule": f"panel a: shared symmetric limit +-{ia['colour_limit']:.3f} at the 98th percentile of |values| "
                          f"over both maps; panel b: 1-D panels share y, 2-D surface symmetric about zero "
                          f"(+-{ib['surface_limit']:.3g}); panel e: state symmetric +-{ie['state_limit']:.3f} "
                          "(98th percentile), neoplastic 0-1",
            "legend_placement": "outside_axes (colorbars outside or below; d_external legend right)",
            "panels": panels, "generated_by": __file__,
            "notes": [
                "a_spatial: no scale bar because pixel size is not certified; equal aspect, y inverted; coords columns read as (x, y)",
                f"e_umap: patient panel has no legend ({ie['n_patients']} patients); tab20 extended with the first tab20b colours so every patient is distinct; random draw order seed 0",
                "f_crops: crops extracted from the W1d card render at about 256 px (left clean H&E half, margins and title trimmed); CellViT boxes omitted",
                "b_response: stored surface indexed [x, y] (verified against marginal partial dependence) and transposed for contourf",
                "d_external: per-cohort gains have no interval in forest.json"],
        })
        copy_plot_code(out, __file__)
        print("done", g, flush=True)
    (a.out / "evidence_cards_log.json").write_text(json.dumps({"section_rule": best, "chosen": sample}, indent=1))


if __name__ == "__main__":
    main()
