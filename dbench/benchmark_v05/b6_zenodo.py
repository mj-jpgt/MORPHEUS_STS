"""B6 annotation recovery on the Zenodo ccRCC TLS Visium cohort (3 sections, 3 patients; pathology labels manual_anno).

Spot features (all z-scored within section, then PCA-free equal width d = number of relation features):
  relations    per-spot f_S scores of the frozen kidney relations (translation_v10 scores_zenodo14620362.npz; read only)
  expression   top-d principal components of log1p-CPM over the 2,000 most variable genes
  hoptimus     top-d principal components of the frozen H-Optimus-0 spot embeddings
  composition  4 lineage marker scores (epithelial, stromal, immune, endothelial)
  random       d i.i.d. Gaussian features (chance reference)
k-means (k = number of annotation classes present, 50 restarts) on pooled labelled spots; ARI, NMI, purity, homogeneity,
completeness, V-measure, silhouette (of the clustering in the feature space).  Patient-level bootstrap (3 patients, 2,000
resamples, fixed cluster assignment) for the intervals; UNASSIGNED spots are excluded.

    python -m dbench.benchmark_v05.b6_zenodo
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score,
                             silhouette_score, v_measure_score)

TX = Path.home() / "morpheus_work_tx/runs/development/translation_v10"
ZT = Path("/lambda/nfs/geeg/biorag3_persistent_20260711/spatial/external_ccrcc_20261016/zenodo/raw/extracted/TLS_VISIUM_USZ/h5ad_preprocessed")
OUT = Path.home() / "morpheus_work_bench/runs/benchmark_v05/b6"
MARK = {"epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CA9", "NDUFA4L2"], "stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"],
        "immune": ["PTPRC", "CD3E", "CD68", "MS4A1", "JCHAIN", "CD74"], "endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "EMCN"]}
KEEP = ["TUM", "NOR", "TLS", "INFL"]


def zs(x):
    x = np.asarray(x, float)
    return (x - x.mean(0)) / np.where(x.std(0) < 1e-9, 1.0, x.std(0))


def purity(y, c):
    return float(sum(np.bincount(y[c == k]).max() for k in np.unique(c)) / len(y))


def metrics(y, c, X):
    out = {"ARI": adjusted_rand_score(y, c), "NMI": normalized_mutual_info_score(y, c), "purity": purity(y, c),
           "homogeneity": homogeneity_score(y, c), "completeness": completeness_score(y, c), "V": v_measure_score(y, c)}
    if len(np.unique(c)) > 1:
        idx = np.random.default_rng(0).choice(len(y), min(4000, len(y)), replace=False)
        out["silhouette"] = float(silhouette_score(X[idx], c[idx])) if len(np.unique(c[idx])) > 1 else np.nan
    else:
        out["silhouette"] = np.nan
    return {k: float(v) for k, v in out.items()}


def main():
    S = np.load(TX / "scores_zenodo14620362.npz", allow_pickle=True)
    sec, bc, pat = S["section"].astype(str), S["barcode"].astype(str), S["patient"].astype(str)
    rel_cols = [k for k in S.files if k.startswith("fS|")]
    R = np.column_stack([S[k] for k in rel_cols]).astype(float)
    good = np.isfinite(R).mean(0) > 0.9          # relations scored on at least 90 % of spots
    R = R[:, good]
    R = np.where(np.isfinite(R), R, np.nanmedian(R, 0, keepdims=True))
    d = R.shape[1]
    feats = {"relations": [], "expression": [], "hoptimus": [], "composition": [], "random": []}
    labels, patients, rows = [], [], []
    rng = np.random.default_rng(20261102)
    for s in np.unique(sec):
        key = s.split("_")[-1]
        a = anndata.read_h5ad(ZT / f"{key}.h5ad")
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        lib = X.sum(1, keepdims=True)
        L = np.log1p(X / np.maximum(lib, 1) * 1e4)
        genes = np.asarray(a.var_names).astype(str)
        gpos = {g: i for i, g in enumerate(genes)}
        ann = dict(zip(np.asarray(a.obs_names).astype(str), a.obs["manual_anno"].astype(str)))
        r = np.flatnonzero(sec == s)
        lab = np.array([ann.get(b, "") for b in bc[r]])
        emb = np.load(next((TX / "zenodo_hoptimus/new/embeddings").glob(f"*_{key}.npz")), allow_pickle=True)
        epos = {b: i for i, b in enumerate(emb["barcodes"].astype(str))}
        obsset = set(np.asarray(a.obs_names).astype(str))
        ok = np.array([(l in KEEP) and (b in epos) and (b in obsset) for l, b in zip(lab, bc[r])])
        ri = r[ok]
        names = np.asarray(a.obs_names).astype(str)
        npos = {b: i for i, b in enumerate(names)}
        li = np.array([npos[b] for b in bc[ri]])
        ei = np.array([epos[b] for b in bc[ri]])
        hv = np.argsort(-L.var(0))[:2000]
        pe = PCA(d, random_state=0).fit_transform(zs(L[li][:, hv]))
        ph = PCA(d, random_state=0).fit_transform(zs(emb["embeddings"][ei]))
        comp = np.column_stack([L[li][:, [gpos[g] for g in gs if g in gpos]].mean(1) for gs in MARK.values()])
        feats["relations"].append(zs(R[ri]))
        feats["expression"].append(zs(pe))
        feats["hoptimus"].append(zs(ph))
        feats["composition"].append(zs(comp))
        feats["random"].append(zs(rng.normal(size=(len(ri), d))))
        labels.append(np.array([KEEP.index(l) for l in lab[ok]]))
        patients.append(np.full(len(ri), pat[ri[0]]))
        rows.append(len(ri))
    y = np.concatenate(labels)
    p = np.concatenate(patients)
    res, assign = {}, {}
    k = len(np.unique(y))
    for name, blocks in feats.items():
        Xf = np.vstack(blocks)
        c = KMeans(k, n_init=20, random_state=0).fit_predict(Xf)
        assign[name] = (c, Xf)
        res[name] = {"point": metrics(y, c, Xf), "per_patient": {pp: metrics(y[p == pp], c[p == pp], Xf[p == pp]) for pp in np.unique(p)}}
    pats = np.unique(p)
    brng = np.random.default_rng(1)
    for name, (c, Xf) in assign.items():
        bs = {m: [] for m in res[name]["point"]}
        for _ in range(500):
            sel = brng.choice(pats, len(pats), replace=True)
            idx = np.concatenate([np.flatnonzero(p == q) for q in sel])
            mm = metrics(y[idx], c[idx], Xf[idx])
            for m, v in mm.items():
                bs[m].append(v)
        res[name]["ci95"] = {m: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))] for m, v in bs.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b6_zenodo_tls.json").write_text(json.dumps({"n_spots": int(len(y)), "classes": KEEP, "class_counts": np.bincount(y).tolist(),
                                                       "n_relation_features": d, "patients": pats.tolist(), "rows": res}, indent=1))
    ms = ["ARI", "NMI", "purity", "homogeneity", "completeness", "V", "silhouette"]
    lines = [f"### B6 annotation recovery, Zenodo ccRCC TLS Visium: {len(y)} labelled spots, classes {dict(zip(KEEP, np.bincount(y).tolist()))}; k-means k={k}, d={d} features per set; patient bootstrap (3 patients, 500 resamples, fixed assignment)", "",
             "| feature set | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for name in feats:
        r = res[name]
        lines.append(f"| {name} | " + " | ".join(f"{r['point'][m]:.3f} [{r['ci95'][m][0]:.3f}, {r['ci95'][m][1]:.3f}]" for m in ms) + " |")
    lines += ["", "Random features are the chance reference. Relations = frozen kidney relations discovered by support-fit ordering + learned beam (W2 score file)."]
    (OUT / "b6_zenodo_tls.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
