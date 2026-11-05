"""W1d: evidence-card data per externally replicated distinct relation (arrays and image crops; no prose).

Per relation a directory runs/development/characterisation_v10/cards/<relation_id>/ with
  spot_values.npz     per spot of the relation rows: sample_id, barcode, patient, coords, target (raw), constituents (raw),
                      state f_S (in-sample frozen relation), nuisance residual, relation residual, CellViT fractions
  umap.npz            2-D UMAP of a 20 000-spot subsample on [constituents, target, CellViT fractions, state] (within-patient z)
  response_surface.json  1-D partial dependence of f_S for every constituent and a 2-D surface for the pair with the largest range
  composition_matched.json  250 um windows, state-high vs state-low pairs matched on CellViT composition (Mahalanobis)
  crops/pair_XX_{high,low}.png + crops/index.json   spot H&E patches with CellViT nuclei boxes at the extreme spot of each region
  ladder.json, forest.json  ladder bars (Delta0-Delta3, % retained) and per-cohort external forest data
"""
import json
import sys
from pathlib import Path

import numpy as np

from dbench.downstream import characterise_v10 as ch
from dbench.downstream import decode_v10 as dv

OUT = ch.OUT / "cards"
HEST = Path("/lambda/nfs/geeg/biorag3_persistent_20260711/morpheus_substrate_mirror/hest/ccrcc")
CLASS_COLOURS = {"Neoplastic": "#d62728", "Inflammatory": "#2ca02c", "Connective": "#1f77b4", "Epithelial": "#9467bd",
                 "Dead": "#7f7f7f"}


def partial_dependence(rel, n_rows=1500, n_grid=20, seed=0):
    from morpheus_encoder.learned_search import relation_v03 as rv
    base = rv.fit_baseline(rel.ys, rel.pat, rel.D, yq=rel.ys, C_q=rel.D)
    o = rv.fit_sets(rel.Z[:, None, :], base, rel.ys, rel.pat, device=ch.DEVICE, return_beta=True)
    model = o["model"]
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(rel.Z), min(n_rows, len(rel.Z)), replace=False)
    Zs = rel.Z[rows]
    k = Zs.shape[1]
    out = {"names": rel.names, "one_d": {}, "grid_unit": "within-patient z"}
    ranges = []
    for j in range(k):
        grid = np.quantile(rel.Z[:, j], np.linspace(0.05, 0.95, n_grid))
        pd_ = np.zeros(n_grid)
        for g_i, gv in enumerate(grid):
            V = Zs.copy()[:, None, :].astype(np.float32)
            V[:, 0, j] = gv
            pd_[g_i] = float(rv.predict(model, V, device=ch.DEVICE)[:, 0].mean())
        out["one_d"][rel.names[j]] = {"grid": grid.tolist(), "partial_dependence": pd_.tolist()}
        ranges.append(float(pd_.max() - pd_.min()))
    if k >= 2:
        a, b = np.argsort(ranges)[::-1][:2]
        ga = np.quantile(rel.Z[:, a], np.linspace(0.05, 0.95, 12))
        gb = np.quantile(rel.Z[:, b], np.linspace(0.05, 0.95, 12))
        sub = Zs[:300]
        surf = np.zeros((12, 12))
        for i, x in enumerate(ga):
            for jx, y in enumerate(gb):
                V = sub.copy()[:, None, :].astype(np.float32)
                V[:, 0, a] = x
                V[:, 0, b] = y
                surf[i, jx] = float(rv.predict(model, V, device=ch.DEVICE)[:, 0].mean())
        pred = surf
        out["two_d"] = {"x": rel.names[a], "y": rel.names[b], "x_grid": ga.tolist(), "y_grid": gb.tolist(),
                        "surface": pred.tolist()}
    out["pd_ranges"] = dict(zip(rel.names, ranges))
    return out


def umap_coords(rel, ctx, state, resid, n=20000, seed=0):
    import umap
    from dbench.downstream.relation import standardise_within
    rng = np.random.default_rng(seed)
    ix = np.sort(rng.choice(len(state), min(n, len(state)), replace=False))
    cv = ctx.cellvit[rel.ok]
    F = np.column_stack([rel.Z, rel.ys, cv, state])[ix]
    F = standardise_within(F, rel.pat[ix])[0]
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(F)
    return ix, emb


def crops(ctx, rel, state, pairs_idx, reg, out_dir, n_pairs=6):
    import h5py
    import pandas as pd
    from shapely import wkb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    out_dir.mkdir(parents=True, exist_ok=True)
    sample = ctx.sec[rel.ok]
    barcode = np.asarray(ctx.coh.raw.barcode)[rel.ok].astype(str)
    index = []
    cache = {}
    for pi, pr in enumerate(pairs_idx[:n_pairs]):
        for side in ("high", "low"):
            r = pr[side]
            rows = np.asarray(reg["spot_rows"][r])
            j = rows[np.argmax(state[rows])] if side == "high" else rows[np.argmin(state[rows])]
            sid, bc = sample[j], barcode[j]
            try:
                if sid not in cache:
                    cache[sid] = (h5py.File(HEST / f"{sid}_patches.h5"),
                                  pd.read_parquet(HEST / f"{sid}_cellvit_seg.parquet"))
                h5, seg = cache[sid]
                bcs = np.array([b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in h5["barcode"][:]])
                k = np.flatnonzero(bcs == bc)
                if len(k) == 0:
                    index.append({"pair": pi, "side": side, "sample": sid, "barcode": bc, "error": "barcode not in patches"})
                    continue
                k = int(k[0])
                img = h5["img"][k]
                cx, cy = h5["coords"][k]
                if "_geom" not in cache.setdefault(sid + "_g", {}):
                    cache[sid + "_g"]["_geom"] = [wkb.loads(g) for g in seg["geometry"].values]
                    cache[sid + "_g"]["cent"] = np.array([[g.centroid.x, g.centroid.y] for g in cache[sid + "_g"]["_geom"]])
                cent = cache[sid + "_g"]["cent"]
                geoms = cache[sid + "_g"]["_geom"]
                # HEST patch coords are the top-left pixel (x, y) of the 224 px patch (verified: nuclei are darker at the
                # CellViT centroids only under this convention)
                ax_, ay_ = cx, cy
                half = 0
                w_ = img.shape[0]
                sel = np.flatnonzero((cent[:, 0] > ax_) & (cent[:, 0] < ax_ + w_) & (cent[:, 1] > ay_) & (cent[:, 1] < ay_ + w_))
                fig, axs = plt.subplots(1, 2, figsize=(7, 3.6))
                axs[0].imshow(img)
                axs[1].imshow(img)
                boxes = []
                for s in sel:
                    minx, miny, maxx, maxy = geoms[s].bounds
                    x0, y0, w, h = minx - ax_, miny - ay_, maxx - minx, maxy - miny
                    cls = str(seg["class"].values[s])
                    axs[1].add_patch(mp.Rectangle((x0, y0), w, h, fill=False, ec=CLASS_COLOURS.get(cls, "k"), lw=0.8))
                    boxes.append({"class": cls, "box": [float(x0), float(y0), float(w), float(h)]})
                for a_ in axs:
                    a_.axis("off")
                axs[0].set_title(f"{side} state {state[j]:.2f}", fontsize=8)
                axs[1].set_title(f"CellViT nuclei n={len(sel)}", fontsize=8)
                p = out_dir / f"pair_{pi:02d}_{side}.png"
                fig.savefig(p, dpi=110, bbox_inches="tight")
                plt.close(fig)
                index.append({"pair": pi, "side": side, "sample": sid, "barcode": bc, "state": float(state[j]),
                              "n_nuclei_in_patch": int(len(sel)), "image": p.name, "boxes": boxes[:400]})
            except Exception as e:
                index.append({"pair": pi, "side": side, "sample": sid, "barcode": bc, "error": f"{type(e).__name__}: {e}"})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def build(ctx, r, pcs_ok=True):
    from dbench.downstream.characterise import matched_high_low, regions
    from morpheus_encoder.learned_search import relation_v03 as rv
    rid = r["relation_id"]
    d = OUT / rid
    d.mkdir(parents=True, exist_ok=True)
    rel = ch.Relation(ctx, rid, r["names"], r["target"], r["family"], r["row_views"])
    state = dv.state_of(rel)
    base = rv.fit_baseline(rel.ys, rel.pat, rel.D)
    nuis_res = np.asarray(base.r_s)
    rel_res = nuis_res - state
    ok = rel.ok
    np.savez_compressed(
        d / "spot_values.npz", sample_id=ctx.sec[ok], barcode=np.asarray(ctx.coh.raw.barcode)[ok].astype(str),
        patient=rel.pat, coords=np.asarray(ctx.coh.coords)[ok], target_raw=ctx.col(r["target"])[ok],
        constituent_names=np.array(r["names"]), constituents_raw=rel.Zraw_ok, target_std=rel.ys, state=state,
        nuisance_residual=nuis_res, relation_residual=rel_res, cellvit_fractions=ctx.cellvit[ok],
        cellvit_columns=np.array(["neoplastic", "inflammatory", "connective"]))
    (d / "response_surface.json").write_text(json.dumps(partial_dependence(rel)))
    try:
        ix, emb = umap_coords(rel, ctx, state, rel_res)
        np.savez_compressed(d / "umap.npz", index=ix, umap=emb, state=state[ix], target=rel.ys[ix], patient=rel.pat[ix],
                            neoplastic=ctx.cellvit[ok][ix, 0])
    except Exception as e:
        (d / "umap_error.txt").write_text(f"{type(e).__name__}: {e}")
    # composition-matched high/low windows (250 um; Mahalanobis on CellViT fractions + deconvolved groups)
    comp_all, comp_cv = ctx.composition_block()
    C = np.asarray(ctx.coh.coords)[ok]
    reg = regions(C, ctx.sec[ok], state, comp_cv[ok][:, :8])
    match = matched_high_low(reg, n=12, max_dist=1.5)
    (d / "composition_matched.json").write_text(json.dumps({
        "pairs": match["pairs"], "status": match["status"],
        "regions": {"section": reg["section"], "state": reg["state"], "fractions": reg["fractions"],
                    "centre_um": reg["centre"]},
        "fraction_columns": ["neoplastic", "inflammatory", "connective", "tumour_dec", "immune_dec", "endothelial_dec",
                             "mural_dec", "tubule_dec"],
        "target_mean": [float(np.mean(rel.ys[np.asarray(rows)])) for rows in reg["spot_rows"]]}))
    pairs = [{"high": p["high"], "low": p["low"]} for p in match["pairs"]]
    idx = crops(ctx, rel, state, pairs, reg, d / "crops")
    # ladder + forest
    ev = [json.loads(l) for l in (ch.OUT / "evidence_rows_final.jsonl").read_text().splitlines() if l.strip()] \
        if (ch.OUT / "evidence_rows_final.jsonl").exists() else []
    lad = {x["test"]: {"estimate": x["estimate"], "ci": x["ci"], "pct_retained": x.get("pct_retained"), "status": x["status"]}
           for x in ev if x["relation_id"] == rid and x["test"].startswith("ladder")}
    (d / "ladder.json").write_text(json.dumps(lad, indent=1))
    fam = json.loads(Path("runs/confirmation/v08_external_family_bh_v01/family_bh_masked.json").read_text())
    rec = next(x for x in fam["confirmatory"] + fam["exploratory_proposed_only"]["table"] if x["candidate_id"] == rid)
    full = json.loads(Path(rec["receipt"]).read_text())["result"]
    forest = {"pooled": full.get("pooled"), "per_cohort": full.get("per_cohort"), "direction": full.get("direction"),
              "family_bh_q": rec.get("family_bh_q"), "n_members_distinct_relation": r["n_members"]}
    (d / "forest.json").write_text(json.dumps(forest, indent=1))
    return {"relation_id": rid, "n_rows": int(ok.sum()), "n_pairs": len(pairs),
            "crops_ok": sum(1 for x in idx if "image" in x), "crops_err": [x for x in idx if "error" in x][:2]}


def main():
    ctx = ch.Ctx()
    rels = ch.relation_list(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rels[: int(sys.argv[2]) if len(sys.argv) > 2 else None]:
        if (OUT / r["relation_id"] / "forest.json").exists():
            continue
        print(json.dumps(build(ctx, r)), flush=True)


if __name__ == "__main__":
    main()
