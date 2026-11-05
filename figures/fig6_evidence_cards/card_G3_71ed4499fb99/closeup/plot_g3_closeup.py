"""G3 vascular close-up (card 71ed4499fb99, section INT21, patient 4, held-out score fitted on the other 23 patients).
Panel l_context: H&E at half resolution (6000 x 6000 full-res px window at x 0-6000, y 1482-7482) with the held-out
G3 score drawn only where it exceeds +1 (within-patient z), and the close-up window outlined.
Panel m_closeup: plain H&E at full resolution, 3000 x 3000 full-res px at x 1143-4143, y 2982-5982, no overlay.
Panel n_closeup_spots: the same close-up with the spot capture areas (score > +1; circle diameter = 0.55 x measured centre spacing, the Visium 55/100 um ratio) outlined so the reader can place the spots on the vessels.
Pixel size is not certified for this slide (card README), so no scale bar is drawn. The dark diagonal line at the upper right
of the close-up is a tissue fold, not a structure.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import pathlib, sys
Image.MAX_IMAGE_PIXELS = None
SCR = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(__file__).resolve().parent
z = np.load(SCR / "cards/71ed4499fb99/71ed4499fb99_heldout_int21.npz")
c, s = z["coords"], z["state_heldout"]
ctx = np.asarray(Image.open(SCR / "g3crop/g3_l1_x0_y1482_w6000_h6000_ds2.000.png"))
clo = np.asarray(Image.open(SCR / "g3crop/g3_l0_x1143_y2982_w3000_h3000.png"))
X0, Y0, DS = 0, 1482, 2.0
x0, y0 = 1143, 2982
cmap = plt.get_cmap("RdBu_r"); vmax = 3.5
# l: context with overlay
fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(ctx); m = (c[:, 0] >= X0) & (c[:, 0] < X0 + 6000) & (c[:, 1] >= Y0) & (c[:, 1] < Y0 + 6000) & (s > 1)
sc = ax.scatter((c[m, 0] - X0) / DS, (c[m, 1] - Y0) / DS, c=s[m], cmap=cmap, vmin=-vmax, vmax=vmax, s=34, lw=0.3, edgecolors="k", alpha=0.9)
ax.add_patch(Rectangle(((x0 - X0) / DS, (y0 - Y0) / DS), 3000 / DS, 3000 / DS, fill=False, ec="k", lw=1.6))
ax.set_axis_off(); fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT / "l_context.png", dpi=300); fig.savefig(OUT / "l_context.pdf"); plt.close(fig)
# colour bar as its own file
fig, ax = plt.subplots(figsize=(1.0, 3.2)); fig.subplots_adjust(0.1, 0.05, 0.4, 0.95)
cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=matplotlib.colors.Normalize(-vmax, vmax)); cb.set_label("Held-out score (within-patient z)")
fig.savefig(OUT / "l_colourbar.png", dpi=300); fig.savefig(OUT / "l_colourbar.pdf"); plt.close(fig)
# m: plain close-up
Image.fromarray(clo).save(OUT / "m_closeup.png")
# n: close-up with spot outlines
fig, ax = plt.subplots(figsize=(6.4, 6.4)); ax.imshow(clo)
m2 = (c[:, 0] >= x0) & (c[:, 0] < x0 + 3000) & (c[:, 1] >= y0) & (c[:, 1] < y0 + 3000) & (s > 1)
# spot capture area: Visium spot diameter / centre spacing = 55/100, spacing measured from the data
from scipy.spatial import cKDTree
spacing = float(np.median(cKDTree(c).query(c, k=2)[0][:, 1])); R = 0.275 * spacing
for xx, yy, v in zip(c[m2, 0] - x0, c[m2, 1] - y0, s[m2]):
    ax.add_patch(Circle((xx, yy), R, fill=False, ec=cmap((v + vmax) / (2 * vmax)), lw=1.8))
ax.set_axis_off(); fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT / "n_closeup_spots.png", dpi=300); fig.savefig(OUT / "n_closeup_spots.pdf"); plt.close(fig)
# chat preview: three panels side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
axes[0].imshow(np.asarray(Image.open(OUT / "l_context.png"))); axes[0].set_title("context, held-out G3 score > +1")
axes[1].imshow(clo); axes[1].set_title("close-up, plain H&E (boxed region)")
axes[2].imshow(np.asarray(Image.open(OUT / "n_closeup_spots.png"))); axes[2].set_title("close-up, spots with score > +1 outlined")
for a in axes: a.set_axis_off()
fig.tight_layout(); fig.savefig(OUT / "G3_vascular_closeup_preview.png", dpi=160); plt.close(fig)
print("spacing px", spacing, "radius px", R); print("n spots in context", m.sum(), "in closeup", m2.sum())
