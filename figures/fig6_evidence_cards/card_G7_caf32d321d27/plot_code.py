#!/usr/bin/env python3
"""Evidence cards v02: held-out score on panel a, plus H&E overlays (g), residual maps (h), constituent maps (i),
composition maps (j) and two pair figures. Section INT21. No statistics are printed on any panel.

Held-out score: relation fitted on the 23 other patients, predicted on INT21 (patient meylan2022_ccrcc:Patient 4, 4,974 rows).
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from PIL import Image
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_evidence_cards_v01 import CARDS, SHORT, ZLAB, sym_lim  # noqa: E402
from plot_evidence_cards_v01 import apply_morpheus_style, copy_plot_code, export_figure, save_plot_data, write_manifest  # noqa: E402

SECTION = "INT21"
DS = 8
THR = 1.0
CELLVIT = ["Neoplastic", "Inflammatory", "Connective"]
SCRATCH = Path(r"C:\Users\mobar\AppData\Local\Temp\claude\C--Users-mobar-OneDrive-biorag\183cb9cc-8cb0-4a9d-ad6f-8c70cbac5491\scratchpad")
QA = {}


def load_card(gid):
    cid = CARDS[gid][0]
    d = SCRATCH / "cards" / cid
    old = dict(np.load(d / "spot_values.npz", allow_pickle=True))
    ho = dict(np.load(d / f"{cid}_heldout_int21.npz", allow_pickle=True))
    idx = {(str(a), str(b)): j for j, (a, b) in enumerate(zip(old["sample_id"], old["barcode"]))}
    sel = np.array([idx[(SECTION, str(b))] for b in ho["barcode"]])
    assert np.allclose(old["coords"][sel], ho["coords"]) and np.allclose(old["target_std"][sel], ho["target_std"], atol=1e-5)
    Zr = old["constituents_raw"][sel].astype(float)
    Zz = (Zr - Zr.mean(0)) / Zr.std(0)  # within-section z, INT21 rows only
    return dict(xy=ho["coords"], barcode=ho["barcode"].astype(str), target=ho["target_std"], held=ho["state_heldout"],
                insample=ho["state_insample"], nres=ho["nuis_res_heldout"], rres=ho["rel_res_heldout"],
                Zz=Zz, names=[str(n) for n in old["constituent_names"]], cv=old["cellvit_fractions"][sel].astype(float))


def spacing_of(xy):
    return float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))


def check_overlaps(fig, key):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_window_extent(r).width > 0]
    issues = []
    for (a, ba), (b, bb) in itertools.combinations(texts, 2):
        ix = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
        iy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
        if ix > 0.5 and iy > 0.5:
            issues.append(f"text overlap: {a!r} / {b!r}")
    for a, b in itertools.combinations(fig.axes, 2):
        ba, bb = a.get_window_extent(r), b.get_window_extent(r)
        ix = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
        iy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
        if ix > 0.5 and iy > 0.5:
            issues.append("axes overlap")
    QA[key] = {"overlap_issues": issues, "n_axes": len(fig.axes)}


def finish(fig, out, stem, df):
    key = f"{out.name}/{stem}"
    check_overlaps(fig, key)
    export_figure(fig, out, stem=stem)
    save_plot_data(df, out, f"{stem}_plot_data")
    plt.close(fig)
    a = np.asarray(Image.open(out / f"{stem}.png").convert("RGB")).astype(int)
    QA[key]["nonwhite_fraction"] = round(float((a < 245).any(2).mean()), 3)


def style_ax(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def resize_spots(fig, axes, xy):
    fig.canvas.draw()
    sp = spacing_of(xy)
    bb = axes[0].get_window_extent()
    ppp = min(bb.width / (np.ptp(xy[:, 0]) + sp), bb.height / (np.ptp(xy[:, 1]) + sp)) * 72 / fig.dpi
    for ax in axes:
        for c in ax.collections:
            c.set_sizes([(sp * ppp) ** 2])


def maps_shared(xy, vals, titles, out, stem, df, lim=None, cmap="RdBu_r", vmin=None, vmax=None, label=ZLAB, figw=6.4):
    n = len(vals)
    fig, axes = plt.subplots(1, n, figsize=(figw, 3.1), layout="constrained")
    axes = np.atleast_1d(axes)
    if vmin is None:
        vmin, vmax = -lim, lim
    sc = None
    for ax, v, t in zip(axes, vals, titles):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=4, linewidths=0, rasterized=True)
        style_ax(ax)
        ax.set_title(t)
    fig.colorbar(sc, ax=list(axes), location="right", shrink=0.7, pad=0.02).set_label(label)
    resize_spots(fig, list(axes), xy)
    finish(fig, out, stem, df)


def maps_own(xy, vals, titles, lims, out, stem, df):
    n = len(vals)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 0.2, 3.3), layout="constrained")
    for ax, v, t, lim in zip(axes, vals, titles, lims):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=4, linewidths=0, rasterized=True)
        style_ax(ax)
        ax.set_title(t)
        cb = fig.colorbar(sc, ax=ax, location="bottom", shrink=0.85, pad=0.03, aspect=18)
        cb.set_ticks([-lim, 0, lim])
        cb.set_ticklabels([f"{-lim:.1f}", "0", f"{lim:.1f}"])
    resize_spots(fig, list(axes), xy)
    finish(fig, out, stem, df)


def base_df(c):
    return pd.DataFrame({"sample_id": SECTION, "barcode": c["barcode"], "x_px": c["xy"][:, 0], "y_px": c["xy"][:, 1]})


def he_overlay(img, xy, v, lim, out, stem, df, label=None):
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained")
    ax.imshow(img, extent=(0, W, H, 0), interpolation="lanczos")
    sc = None
    if v is not None:
        m = v > THR
        sc = ax.scatter(xy[m, 0] / DS, xy[m, 1] / DS, c=v[m], cmap="RdBu_r", vmin=-lim, vmax=lim, s=4, alpha=0.85,
                        linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, location="right", shrink=0.7, pad=0.02).set_label(label)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if v is not None:
        fig.canvas.draw()
        sp = spacing_of(xy) / DS
        bb = ax.get_window_extent()
        ppp = min(bb.width / W, bb.height / H) * 72 / fig.dpi
        sc.set_sizes([(sp * ppp) ** 2])
    finish(fig, out, stem, df)


def update_text(out, n, qa_text):
    rd = out / "README.md"
    txt = rd.read_text(encoding="utf-8").split("\n## v02 update")[0].rstrip() + "\n"
    txt += f"""
## v02 update
Panel a is now the held-out score: the relation was fitted on the other 23 patients and predicted on {SECTION} (one patient, {n} spots);
the earlier in-sample version is kept as a_spatial_insample_v01.*. New panels: g_he_tissue (plain H&E), g_he_state and g_he_target
(held-out score and target drawn as spots only where the value exceeds 1, on the H&E), h_residual (held-out nuisance residual and
residual after the relation), i_constituents (within-section z of each constituent plus the held-out score), j_composition (CellViT
neoplastic, inflammatory, connective fractions). H&E mapping: x_img = x_px / 8, y_img = y_px / 8 on the 8x-downsampled whole-slide
image (17128 x 16878 px; image_meta.json), rows increasing downward. No scale bar because pixel size is not certified.
"""
    rd.write_text(txt, encoding="utf-8")
    qa = out / "QA_REPORT.md"
    t = qa.read_text(encoding="utf-8") if qa.exists() else ""
    t = t.split("\n## v02 checks")[0].rstrip() + "\n\n## v02 checks\n" + qa_text
    qa.write_text(t, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    apply_morpheus_style(scienceplots=False)
    meta = json.load(open(SCRATCH / "int21" / "image_meta.json"))
    img = np.asarray(Image.open(SCRATCH / "int21" / "INT21_ds8.png").convert("RGB"))
    H, W = img.shape[:2]
    fx, fy = meta["full_dims_wh"][0] / W, meta["full_dims_wh"][1] / H
    assert abs(fx - DS) < 0.05 and abs(fy - DS) < 0.05, (fx, fy)
    cards = {g: load_card(g) for g in CARDS}
    for g1, g2 in (("G3", "G4"), ("G5", "G7")):
        c1, c2 = cards[g1], cards[g2]
        assert (c1["barcode"] == c2["barcode"]).all()
        lim = sym_lim(c1["held"], c2["held"])
        df = pd.DataFrame({"barcode": c1["barcode"], "x_px": c1["xy"][:, 0], "y_px": c1["xy"][:, 1],
                           f"state_heldout_{g1}": c1["held"], f"state_heldout_{g2}": c2["held"]})
        stem = f"pair_{g1}_{g2}_state"
        maps_shared(c1["xy"], [c1["held"], c2["held"]], [g1, g2], a.out, stem, df, lim=lim)
        QA[f"{a.out.name}/{stem}"]["colour_limit"] = lim
    for gid, (cid, label) in CARDS.items():
        c = cards[gid]
        out = a.out / f"card_{gid}_{cid}"
        xy = c["xy"]
        if not (out / "a_spatial_insample_v01.png").exists():
            for ext in ("pdf", "svg", "png"):
                shutil.move(out / f"a_spatial.{ext}", out / f"a_spatial_insample_v01.{ext}")
            shutil.move(out / "a_spatial_plot_data.parquet", out / "a_spatial_insample_v01_plot_data.parquet")
        if not (out / "plot_code_v01.py").exists():
            shutil.copy2(out / "plot_code.py", out / "plot_code_v01.py")
        lim = sym_lim(c["target"], c["held"])
        df = base_df(c).assign(target_std=c["target"], state_heldout=c["held"], state_insample=c["insample"])
        maps_shared(xy, [c["target"], c["held"]], ["Target", "Held-out score"], out, "a_spatial", df, lim=lim)
        he_overlay(img, xy, None, lim, out, "g_he_tissue",
                   pd.DataFrame([{"section": SECTION, "image": "INT21_ds8.png", "img_w": W, "img_h": H,
                                  "downsample_x": fx, "downsample_y": fy}]))
        dfg = base_df(c).assign(x_img=xy[:, 0] / DS, y_img=xy[:, 1] / DS)
        he_overlay(img, xy, c["held"], lim, out, "g_he_state", dfg.assign(state_heldout=c["held"], drawn=c["held"] > THR),
                   "Held-out score")
        he_overlay(img, xy, c["target"], lim, out, "g_he_target", dfg.assign(target_std=c["target"], drawn=c["target"] > THR), ZLAB)
        rl = sym_lim(c["nres"], c["rres"])
        maps_shared(xy, [c["nres"], c["rres"]], ["Nuisance residual", "Residual after relation"], out, "h_residual",
                    base_df(c).assign(nuis_res_heldout=c["nres"], rel_res_heldout=c["rres"]), lim=rl, label="Residual (z)")
        vals = [c["Zz"][:, j] for j in range(c["Zz"].shape[1])] + [c["held"]]
        titles = [SHORT[k] for k in c["names"]] + ["Held-out score"]
        lims = [sym_lim(v) for v in vals]
        dfi = base_df(c)
        for t, v in zip(titles, vals):
            dfi[t] = v
        maps_own(xy, vals, titles, lims, out, "i_constituents", dfi)
        dfj = base_df(c)
        for k, nm in enumerate(CELLVIT):
            dfj[nm.lower()] = c["cv"][:, k]
        maps_shared(xy, [c["cv"][:, k] for k in range(3)], CELLVIT, out, "j_composition", dfj, cmap="viridis", vmin=0,
                    vmax=1, label="Fraction", figw=8.6)
        pan = {k: [f"{k}.pdf", f"{k}.svg", f"{k}.png", f"{k}_plot_data.parquet"] for k in
               ("a_spatial", "a_spatial_insample_v01", "g_he_tissue", "g_he_state", "g_he_target", "h_residual",
                "i_constituents", "j_composition")}
        mp = out / "figure_manifest.json"
        man = json.load(open(mp))
        man["panels"].update(pan)
        gb = man.get("generated_by")
        gb = gb if isinstance(gb, list) else [gb]
        man["generated_by"] = sorted(set(gb) | {str(Path(__file__).resolve())})
        man["source_files"] = sorted(set(man["source_files"]) | {
            str(SCRATCH / "cards" / cid / f"{cid}_heldout_int21.npz"), str(SCRATCH / "int21" / "INT21_ds8.png"),
            str(SCRATCH / "int21" / "image_meta.json")})
        man["held_out_score"] = ("a_spatial is held-out: fit on the other 23 patients, predicted on INT21 "
                                 "(box ~/scratch_w7/heldout_int21.py); in-sample v01 kept as a_spatial_insample_v01")
        man["selection_rule_v02"] = (f"section {SECTION}; g_he_state/g_he_target draw only spots with value > {THR} "
                                     "(within-patient z units); i_constituents standardised over INT21 rows only")
        man["scale_rule_v02"] = {
            "a_spatial": f"shared symmetric limit +-{lim:.3f} (98th percentile of |target, held-out score|)",
            "g_he_state/g_he_target": f"same limit +-{lim:.3f}, RdBu_r, alpha 0.85, spots > {THR} only",
            "h_residual": f"shared symmetric limit +-{rl:.3f} (98th percentile of |both residuals|)",
            "i_constituents": "per-map symmetric limit at the 98th percentile: " + ", ".join(f"{t} {l:.2f}" for t, l in zip(titles, lims)),
            "j_composition": "shared 0-1 viridis"}
        man["he_mapping"] = {"x_img": "x_px / 8", "y_img": "y_px / 8", "image": "INT21_ds8.png", "image_wh": [W, H],
                             "downsample_per_axis": [fx, fy], "origin": "top, rows increase downward",
                             "scale_bar": "none (pixel size not certified)"}
        write_manifest(out, man)
        qa = [f"- files: {', '.join(sorted(pan))} (pdf/svg/png + plot data each); v01 a_spatial kept as a_spatial_insample_v01",
              f"- colour limits: a_spatial +-{lim:.3f}; g_he_state and g_he_target the same; h_residual +-{rl:.3f}; i_constituents per map (98th pct); j_composition 0-1",
              f"- alignment: g_he_target overlay compared against a_spatial (target, left) and fig5 emt1_heterogeneity_overlay_INT21_v01/02_molecular_score_target.png; band positions agree (visual)",
              "- script check of text and axes bounding boxes (overlap count, non-white fraction):"]
        for k, vq in QA.items():
            if k.startswith(out.name + "/"):
                qa.append(f"  - {k.split('/')[1]}: overlaps {len(vq['overlap_issues'])}, non-white {vq['nonwhite_fraction']}")
        update_text(out, len(xy), "\n".join(qa) + "\n")
        copy_plot_code(out, __file__)
        print("done", gid, flush=True)
    (a.out / "evidence_cards_v02_qa.json").write_text(json.dumps(QA, indent=1, default=str))


if __name__ == "__main__":
    main()
