"""Pool ST-Net (n=23) with HER2+ Andersson (n=8) into a single n=31 breast
molecular/tissue substrate for the discovery arm.

Mirrors cscd/build_breast.py's construction verbatim in every stage that
does not need to change for a second cohort (log1p(CPM) -> 30-comp PCA ->
ridge-residualize -> TopK dictionary, cross-fit 5-fold by patient,
Hungarian-aligned; 98 standardized CellViT descriptors -> TopKSAE). The two
things that DO change, and why:

  Molecular side: the gene universe is re-intersected against HER2+'s own
  11,871-gene universe (ST-Net alone used 11,787; the two legacy-ST cohorts
  were not profiled against the same fixed panel). A cohort indicator is
  added to the declared ridge-residualization covariates alongside
  log-depth and the 5 CellViT micro composition fractions, so any pooled
  linear cohort-mean shift in the 30 PCA components is removed before the
  dictionary sees the data.

  Tissue side: raw CellViT descriptors are standardized WITHIN each cohort
  separately (each cohort's own spot mean/std) before pooling, rather than
  globally as build_breast.py does for a single cohort. This is not
  cosmetic: several of the 98 descriptors (area, nn_distance, packing
  density) are in raw pixel units, and the two cohorts' pixel_size_um
  differ enough to matter -- HER2+'s Patient A sits at ~0.92-0.95 um/px
  against a ~0.69 um/px cohort/ST-Net median, a ~35% scale difference on
  the same imaging platform. Global standardization would leave that scale
  difference as a real (if small) cohort-level and patient-level offset in
  every pixel-scaled feature; within-cohort standardization removes the
  cohort-level part of it by construction. It does NOT fix the Patient-A-
  specific residual, which is exactly why the task's mandatory
  leave-one-patient-out check on any flagged cell exists downstream.

Both choices are declared before any hypothesis is scored -- this script
only ever loads raw counts/features and never touches breast_discovery.py's
correspondence-scoring logic.

Cohort-predictability check: after fitting, this script also reports a
leave-one-patient-out nearest-centroid classification of cohort identity
from the pooled 31x20 molecular and 31x8 tissue state matrices, against a
label-permutation null, so "pooling worked" is a reported number and not an
assumption.

Usage (on a100box, from ~):
    python -m cscd.build_breast_pooled --out-dir cscd/data/breast_pooled
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .build_breast import (MICRO_CLASS_COLS, WSJ_ROOT, cross_fit_dictionary,
                           randomized_pca)
from .residualize import ridge_residualize

HER2_ROOT = "cscd/data/her2"
GENCODE_GTF = "/lambda/nfs/geeg/biorag3_persistent_20260711/reference/gencode.v47.annotation.gtf.gz"
GENE_MAP_CACHE = "cscd/data/gencode_v47_ensg_to_symbol.tsv"


def load_ensg_to_symbol(cache_path=GENE_MAP_CACHE, gtf_path=GENCODE_GTF):
    """Ensembl gene id (version stripped) -> HGNC symbol, 1:1 only.

    ST-Net's gene universe is Ensembl ids; HER2+'s is symbols (both are
    HEST's own var_names, unchanged). No id-mapping column exists in either
    cohort's h5ad var table, so this maps through GENCODE v47's own GTF,
    already resident on the box as a reference file (no download). Symbols
    that resolve from more than one Ensembl id are dropped rather than
    resolved by fiat, so the mapping used for pooling is unambiguous in
    both directions.
    """
    import gzip
    import os
    import re

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, sep="\t", header=None, names=["ensg", "symbol"])
    else:
        pat = re.compile(r'gene_id "([^".]+)[^"]*"; gene_type[^;]*; gene_name "([^"]+)"')
        rows = []
        with gzip.open(gtf_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2] != "gene":
                    continue
                m = pat.search(f[8])
                if m:
                    rows.append((m.group(1), m.group(2)))
        df = pd.DataFrame(rows, columns=["ensg", "symbol"])
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, sep="\t", header=False, index=False)
    dup_symbol = df["symbol"].duplicated(keep=False)
    df = df[~dup_symbol]
    assert not df["ensg"].duplicated().any()
    return dict(zip(df["ensg"], df["symbol"]))


def log(**kw):
    print(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}),
         flush=True)


def loo_nearest_centroid_balanced_accuracy(X, cohort, reg=1e-3):
    """Leave-one-out nearest-(regularized-Mahalanobis)-centroid classification.

    No sklearn dependency (consistent with the rest of this arm's
    from-scratch numerics). Pooled within-class covariance is re-estimated
    on the n-1 training patients each fold so the held-out patient never
    contributes to its own classifier.
    """
    n, d = X.shape
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, ctr = X[tr], cohort[tr]
        mu0 = Xtr[ctr == 0].mean(0)
        mu1 = Xtr[ctr == 1].mean(0)
        centered = np.where(ctr[:, None] == 0, Xtr - mu0, Xtr - mu1)
        cov = centered.T @ centered / max(len(Xtr) - 2, 1) + reg * np.eye(d)
        covinv = np.linalg.pinv(cov)
        d0 = (X[i] - mu0) @ covinv @ (X[i] - mu0)
        d1 = (X[i] - mu1) @ covinv @ (X[i] - mu1)
        preds[i] = 0 if d0 < d1 else 1
    rec0 = float(np.mean(preds[cohort == 0] == 0)) if (cohort == 0).any() else float("nan")
    rec1 = float(np.mean(preds[cohort == 1] == 1)) if (cohort == 1).any() else float("nan")
    bal_acc = (rec0 + rec1) / 2
    return bal_acc, preds


def cohort_predictability_report(X, cohort, name, n_perm=2000, seed=0, reg=1e-3):
    obs, preds = loo_nearest_centroid_balanced_accuracy(X, cohort, reg=reg)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(cohort)
        null[i], _ = loo_nearest_centroid_balanced_accuracy(X, perm, reg=reg)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {
        "name": name, "n_patients": int(X.shape[0]), "n_dims": int(X.shape[1]),
        "loo_balanced_accuracy": float(obs),
        "null_median_balanced_accuracy": float(np.median(null)),
        "null_p95_balanced_accuracy": float(np.quantile(null, 0.95)),
        "p_cohort_predictable": p,
        "n_perm": n_perm,
        "interpretation": ("cohort identity NOT reliably predictable from pooled "
                          "states (good: batch handling worked)" if p > 0.05 else
                          "cohort identity IS predictable from pooled states "
                          "(pooling did not remove the batch signal)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2-dir", default=HER2_ROOT)
    ap.add_argument("--out-dir", default="cscd/data/breast_pooled")
    ap.add_argument("--n-mol-atoms", type=int, default=20)
    ap.add_argument("--mol-k", type=int, default=4)
    ap.add_argument("--n-tis-atoms", type=int, default=8)
    ap.add_argument("--tis-k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-pca", type=int, default=30)
    ap.add_argument("--min-depth", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- load ST-Net (WS-J's build, read-only) and HER2+ (this arm's build) --
    log(event="load_start", stnet_root=WSJ_ROOT, her2_root=args.her2_dir)
    counts_s = np.load(os.path.join(WSJ_ROOT, "counts.f32.npy"), mmap_mode="r")
    genes_s = np.load(os.path.join(WSJ_ROOT, "genes.npy"), allow_pickle=True).astype(str)
    meta_s = pd.read_parquet(os.path.join(WSJ_ROOT, "meta.parquet"))
    tis_s = np.load(os.path.join(WSJ_ROOT, "morpheus_features.npy"))
    tis_names_s = [str(x) for x in np.load(
        os.path.join(WSJ_ROOT, "morpheus_feature_names.npy"), allow_pickle=True)]

    counts_h = np.load(os.path.join(args.her2_dir, "counts.f32.npy"), mmap_mode="r")
    genes_h = np.load(os.path.join(args.her2_dir, "genes.npy"), allow_pickle=True).astype(str)
    meta_h = pd.read_parquet(os.path.join(args.her2_dir, "meta.parquet"))
    tis_h = np.load(os.path.join(args.her2_dir, "morpheus_features.npy"))
    tis_names_h = [str(x) for x in np.load(
        os.path.join(args.her2_dir, "morpheus_feature_names.npy"), allow_pickle=True)]
    assert tis_names_s == tis_names_h, "tissue feature definitions diverged between cohorts"
    tis_names = tis_names_s

    log(event="loaded", n_spots_stnet=counts_s.shape[0], n_genes_stnet=len(genes_s),
        n_spots_her2=counts_h.shape[0], n_genes_her2=len(genes_h))

    # ---- common gene universe ----------------------------------------------
    # ST-Net's var_names are Ensembl gene ids; HER2+'s are HGNC symbols (both
    # unchanged HEST var_names -- neither cohort's h5ad carries the other id
    # type). Map ST-Net's ids to symbols through GENCODE v47 (1:1 only, see
    # load_ensg_to_symbol) so both cohorts are indexed the same way.
    ensg2sym = load_ensg_to_symbol()
    pos_s_by_ensg = {g: i for i, g in enumerate(genes_s.tolist())}
    stnet_symbol_of_ensg = {g: ensg2sym[g] for g in genes_s.tolist() if g in ensg2sym}
    n_stnet_unmapped = len(genes_s) - len(stnet_symbol_of_ensg)
    stnet_symbols = set(stnet_symbol_of_ensg.values())
    her2_symbols = set(genes_h.tolist())
    common = np.asarray(sorted(stnet_symbols & her2_symbols))
    pos_h = {g: i for i, g in enumerate(genes_h.tolist())}
    sym_to_ensg = {v: k for k, v in stnet_symbol_of_ensg.items()}
    col_s = np.asarray([pos_s_by_ensg[sym_to_ensg[s]] for s in common.tolist()], dtype=np.int64)
    col_h = np.asarray([pos_h[s] for s in common.tolist()], dtype=np.int64)
    log(event="gene_intersection", n_common_genes=len(common),
        n_stnet_total=len(genes_s), n_stnet_unmapped_to_symbol=n_stnet_unmapped,
        n_her2_only=len(her2_symbols) - len(common),
        mapping="ST-Net ENSG -> HGNC symbol via GENCODE v47 GTF (1:1 only), "
               "then intersected with HER2+'s native symbol var_names")

    counts_s = np.asarray(counts_s, dtype=np.float64)[:, col_s]
    counts_h = np.asarray(counts_h, dtype=np.float64)[:, col_h]
    assert not np.isnan(tis_s).any() and not np.isinf(tis_s).any()
    assert not np.isnan(tis_h).any() and not np.isinf(tis_h).any()

    pid_s = meta_s["patient_id"].to_numpy().astype(str)
    pid_h = meta_h["patient_id"].to_numpy().astype(str)
    assert not (set(pid_s.tolist()) & set(pid_h.tolist())), "patient id collision across cohorts"

    # ---- QC floor, per-cohort spot depth, before any pooling ------------------
    def qc(counts):
        depth0 = counts.sum(1)
        return depth0 >= args.min_depth, int((depth0 < args.min_depth).sum())

    keep_s, ndrop_s = qc(counts_s)
    keep_h, ndrop_h = qc(counts_h)
    log(event="qc_floor", min_depth=args.min_depth,
        n_dropped_stnet=ndrop_s, n_kept_stnet=int(keep_s.sum()),
        n_dropped_her2=ndrop_h, n_kept_her2=int(keep_h.sum()))

    counts_s, tis_s, pid_s = counts_s[keep_s], tis_s[keep_s], pid_s[keep_s]
    counts_h, tis_h, pid_h = counts_h[keep_h], tis_h[keep_h], pid_h[keep_h]

    cohort = np.concatenate([np.zeros(counts_s.shape[0]), np.ones(counts_h.shape[0])])
    counts = np.vstack([counts_s, counts_h])
    pid = np.concatenate([pid_s, pid_h])
    patients = sorted(set(pid.tolist()))
    n_stnet_patients = len(set(pid_s.tolist()))
    n_her2_patients = len(set(pid_h.tolist()))
    log(event="pooled_sizes", n_spots=int(counts.shape[0]), n_patients=len(patients),
        n_stnet_patients=n_stnet_patients, n_her2_patients=n_her2_patients)

    # ---- molecular side ---------------------------------------------------
    depth = counts.sum(1)
    assert (depth > 0).all(), "zero-depth spot present after QC floor"
    cpm = counts / depth[:, None] * 1e4
    logcpm = np.log1p(cpm)
    del cpm, counts

    log(event="pca_start", n_components=args.n_pca)
    Xc = logcpm - logcpm.mean(0)
    H, ev_ratio = randomized_pca(Xc, args.n_pca, seed=args.seed)
    ev_ratio_sum = float(ev_ratio.sum())
    log(event="pca_done", explained_var_ratio_sum=ev_ratio_sum)
    del logcpm, Xc

    log_depth = np.log1p(depth)
    tis_pooled_raw = np.vstack([tis_s, tis_h])
    C = np.column_stack([log_depth] + [tis_pooled_raw[:, j] for j in MICRO_CLASS_COLS]
                        + [cohort])
    C = (C - C.mean(0)) / np.maximum(C.std(0), 1e-9)
    covariate_names = (["log_depth"] + [tis_names[j] for j in MICRO_CLASS_COLS]
                       + ["cohort_indicator_her2"])
    log(event="covariates", names=covariate_names)

    R, resid_info = ridge_residualize(H, C, seed=args.seed)
    log(event="residualised", alphas=[f["alpha"] for f in resid_info["per_fold"]])

    log(event="mol_dictionary_start", n_atoms=args.n_mol_atoms, k=args.mol_k)
    ZP_mol, mol_hist = cross_fit_dictionary(R, pid, patients, args.n_mol_atoms,
                                            args.mol_k, args.epochs, args.seed,
                                            device=args.device)
    log(event="mol_dictionary_done",
        final_mse=[h["mse"] for h in mol_hist], dead=[h["dead_atoms"] for h in mol_hist])

    # ---- tissue side: within-cohort standardization, then pool ------------
    def zscore(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)

    tis_s_z = zscore(tis_s)
    tis_h_z = zscore(tis_h)
    tis_z = np.vstack([tis_s_z, tis_h_z])
    log(event="tis_dictionary_start", n_atoms=args.n_tis_atoms, k=args.tis_k,
        standardization="within-cohort (separate mean/std per cohort, then pooled)")
    ZP_tis, tis_hist = cross_fit_dictionary(tis_z, pid, patients, args.n_tis_atoms,
                                            args.tis_k, args.epochs, args.seed,
                                            device=args.device)
    log(event="tis_dictionary_done",
        final_mse=[h["mse"] for h in tis_hist], dead=[h["dead_atoms"] for h in tis_hist])

    # ---- cohort-predictability check ---------------------------------------
    patient_cohort = np.array([0 if p in set(pid_s.tolist()) else 1 for p in patients])
    predict_mol = cohort_predictability_report(ZP_mol, patient_cohort, "molecular_states",
                                                seed=args.seed)
    predict_tis = cohort_predictability_report(ZP_tis, patient_cohort, "tissue_states",
                                                seed=args.seed + 1)
    predict_both = cohort_predictability_report(np.hstack([ZP_mol, ZP_tis]), patient_cohort,
                                                 "molecular_plus_tissue_states",
                                                 seed=args.seed + 2)
    log(event="cohort_predictability", molecular=predict_mol, tissue=predict_tis,
        combined=predict_both)

    # ---- write outputs ------------------------------------------------------
    np.savez(os.path.join(out_dir, "molecular_states.npz"),
            scores=ZP_mol, patient_ids=np.array(patients),
            n_atoms=args.n_mol_atoms, k=args.mol_k)
    np.savez(os.path.join(out_dir, "tissue_states.npz"),
            scores=ZP_tis, patient_ids=np.array(patients),
            n_atoms=args.n_tis_atoms, k=args.tis_k,
            source_feature_names=np.array(tis_names))

    manifest = {
        "cohort": "hest_breast_pooled_stnet_her2",
        "organ": "Breast",
        "pooled_from": ["hest_breast_stnet (He et al., ST-Net)",
                        "her2_andersson (Andersson et al., HER2+)"],
        "source": "MahmoodLab/hest on HuggingFace (HEST-1k)",
        "n_patients": len(patients), "n_stnet_patients": n_stnet_patients,
        "n_her2_patients": n_her2_patients,
        "n_spots": int(counts_s.shape[0] + counts_h.shape[0]),
        "n_spots_stnet": int(counts_s.shape[0]), "n_spots_her2": int(counts_h.shape[0]),
        "n_spots_dropped_qc_stnet": ndrop_s, "n_spots_dropped_qc_her2": ndrop_h,
        "n_genes_common": len(common), "n_genes_stnet_only": len(genes_s) - len(common),
        "n_genes_her2_only": len(genes_h) - len(common),
        "qc_min_depth": args.min_depth,
        "patients": patients,
        "batch_handling": {
            "molecular": "cohort indicator appended to the declared ridge-"
                        "residualization covariates (log_depth + 5 CellViT "
                        "micro composition fractions + cohort_indicator_her2); "
                        "removes a pooled linear cohort-mean shift in the 30 "
                        "PCA components before dictionary fitting.",
            "tissue": "within-cohort standardization (each cohort's own spot "
                     "mean/std) before pooling and dictionary fitting, "
                     "because several of the 98 raw descriptors are in pixel "
                     "units and the two cohorts' pixel_size_um differ "
                     "(HER2+ Patient A ~0.92-0.95 um/px vs a ~0.69 um/px "
                     "cohort/ST-Net median); chosen over global "
                     "standardization for exactly that reason.",
            "cohort_predictability_check": {
                "molecular_states": predict_mol,
                "tissue_states": predict_tis,
                "molecular_plus_tissue_states": predict_both,
            },
        },
        "molecular_side": {
            "normalisation": "CPM (1e4) then log1p over the common-gene "
                             "intersection of both cohorts",
            "raw_components": f"randomized PCA (Halko), {args.n_pca} components, "
                              f"explained_variance_ratio_sum={ev_ratio_sum:.4f}",
            "declared_covariates": covariate_names,
            "residualisation": "cross-fitted ridge (cscd.residualize.ridge_residualize), "
                               "5 folds, alpha in {0.1,1.0,10.0} chosen per fold",
            "dictionary": {"method": "TopK sparse autoencoder (cscd.dictionary.TopKSAE)",
                           "n_atoms": args.n_mol_atoms, "k": args.mol_k,
                           "epochs": args.epochs,
                           "cross_fit": "5-fold by patient, Hungarian-aligned "
                                        "across folds, aggregated to patient "
                                        "mean, mirroring cscd/build_breast.py"},
        },
        "tissue_side": {
            "route": "HEST's own official per-nucleus CellViT segmentation for "
                     "both cohorts; feature code identical byte-for-byte "
                     "(sha256-verified) across ST-Net and HER2+",
            "n_raw_features": int(tis_pooled_raw.shape[1]),
            "dictionary": {"method": "TopK sparse autoencoder (cscd.dictionary.TopKSAE)",
                           "n_atoms": args.n_tis_atoms, "k": args.tis_k,
                           "epochs": args.epochs,
                           "cross_fit": "5-fold by patient, Hungarian-aligned, "
                                        "aggregated to patient mean"},
        },
        "seed": args.seed,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "cscd/build_breast_pooled.py",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    log(event="done", out_dir=out_dir)


if __name__ == "__main__":
    main()
