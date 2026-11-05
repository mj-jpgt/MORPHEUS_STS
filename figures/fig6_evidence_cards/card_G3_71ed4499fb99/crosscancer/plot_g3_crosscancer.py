"""G3 cross-cancer overlay: the same relation drawn on kidney (held out, fitted on the other 23 patients) and on
colorectal tissue (frozen kidney relation, no refit). One row per organ, three columns:
  o_<organ>_he      plain H&E of the section
  o_<organ>_target  target (EMT-I gradient) drawn where it exceeds +1 within-patient z
  o_<organ>_score   G3 score drawn where it exceeds +1 within-patient z
Shared colour bar o_colourbar; preview G3_crosscancer_preview.png. No statistics on the panels.
Usage: python plot_g3_crosscancer.py <scratchpad> ; expects
  <scratchpad>/cards/71ed4499fb99/71ed4499fb99_heldout_int21.npz, <scratchpad>/int21/INT21_level2.png + image_meta.json
  <scratchpad>/crc/g3_crc_<SEC>.npz, <scratchpad>/crc/<SEC>_level2.png + <SEC>_image_meta.json (SEC from crc/SECTION.txt)
"""
import json, pathlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SCR = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(__file__).resolve().parent
CMAP = plt.get_cmap("RdBu_r"); VMAX = 3.0; THR = 1.0

def organ_panels(tag, npz, score_key, img_png, meta_json):
    z = np.load(npz, allow_pickle=True); meta = json.load(open(meta_json))
    dsx = float(meta.get("downsample_x", meta.get("level_downsample", meta.get("downsample", 4.0))))
    dsy = float(meta.get("downsample_y", dsx))
    img = np.asarray(Image.open(img_png))
    c = z["coords"].astype(float) / np.array([dsx, dsy])
    # crop the image to the spot bounding box plus a margin so sections that cover part of the slide fill the panel
    pad = 0.04 * max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    x0, y0 = max(0, int(c[:, 0].min() - pad)), max(0, int(c[:, 1].min() - pad))
    x1, y1 = min(img.shape[1], int(c[:, 0].max() + pad)), min(img.shape[0], int(c[:, 1].max() + pad))
    img = img[y0:y1, x0:x1]; c = c - np.array([x0, y0]); H, W = img.shape[:2]
    # keep only spots on this image (the CRC npz may hold one section already)
    sid = z["sample_id"].astype(str) if "sample_id" in z else None
    s_t, s_s = z["target_std"].astype(float), z[score_key].astype(float)
    # spot marker size from the measured spacing
    from scipy.spatial import cKDTree
    sp = float(np.median(cKDTree(c).query(c, k=2)[0][:, 1]))
    def panel(name, vals=None):
        fig = plt.figure(figsize=(6.4, 6.4 * H / W)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        if vals is not None:
            m = vals > THR
            ax.scatter(c[m, 0], c[m, 1], c=vals[m], cmap=CMAP, vmin=-VMAX, vmax=VMAX,
                       s=(0.55 * sp * 72 / fig.dpi * (6.4 / (W / fig.dpi))) ** 2 * 0.9, lw=0.25, edgecolors="k", alpha=0.95)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
        fig.savefig(OUT / f"{name}.png", dpi=300); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    panel(f"o_{tag}_he"); panel(f"o_{tag}_target", s_t); panel(f"o_{tag}_score", s_s)
    return dict(n=int(len(c)), n_target_hi=int((s_t > THR).sum()), n_score_hi=int((s_s > THR).sum()),
                spearman=float(np.corrcoef(np.argsort(np.argsort(s_t)), np.argsort(np.argsort(s_s)))[0, 1]))

qa = {}
qa["kidney"] = organ_panels("kidney", SCR / "cards/71ed4499fb99/71ed4499fb99_heldout_int21.npz", "state_heldout",
                            SCR / "int21/INT21_level2.png", SCR / "int21/image_meta.json")
crc_dir = SCR / "crc"
SECTIONS = [a for a in sys.argv[2:]] or (["ZEN42"] if (crc_dir / "g3_crc_ZEN42.npz").exists() else [])
for sec in SECTIONS:
    tag = "colorectal" if len(SECTIONS) == 1 else f"colorectal_{sec}"
    qa[tag] = organ_panels(tag, crc_dir / f"g3_crc_{sec}.npz", "state_frozen",
                           crc_dir / f"{sec}_level2.png", crc_dir / f"{sec}_image_meta.json")
    qa[tag]["section"] = sec
# colour bar
fig, ax = plt.subplots(figsize=(1.0, 3.2)); fig.subplots_adjust(0.1, 0.05, 0.4, 0.95)
cb = matplotlib.colorbar.ColorbarBase(ax, cmap=CMAP, norm=matplotlib.colors.Normalize(-VMAX, VMAX))
cb.set_label("within-patient z (drawn where > +1)")
fig.savefig(OUT / "o_colourbar.png", dpi=300); fig.savefig(OUT / "o_colourbar.pdf"); plt.close(fig)
# preview
rows = [k for k in qa]
fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5.2 * len(rows)), squeeze=False)
titles = ["H&E", "target: EMT-I gradient > +1 z", "G3 score > +1 z"]
for r, organ in enumerate(rows):
    for j, kind in enumerate(("he", "target", "score")):
        axes[r, j].imshow(np.asarray(Image.open(OUT / f"o_{organ}_{kind}.png"))); axes[r, j].set_axis_off()
        axes[r, j].set_title(("kidney ccRCC, held out: " if organ == "kidney" else "colorectal, frozen kidney relation: ") + titles[j], fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "G3_crosscancer_preview.png", dpi=150); plt.close(fig)
json.dump(qa, open(OUT / "crosscancer_qa.json", "w"), indent=1); print(json.dumps(qa))
