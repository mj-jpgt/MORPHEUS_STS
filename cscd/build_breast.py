"""Build the paired breast substrate: molecular states + tissue states.

Mirrors real_discovery.py's construction, on a new cohort and organ (HEST-1k's
He et al. ST-Net breast cancer dataset, 23 patients / 68 sections / 30,655
spots) that the discovery arm has never touched. Source substrate (raw counts,
gene universe, CellViT-derived tissue morphology features) was already built
by WS-J on a100box's persistent store; this script only adds the
residualisation + dictionary stages the discovery arm needs, and never
re-derives or overwrites WS-J's own artifacts.

Molecular side
    H = top 30 PCA components of log1p(CPM) over the 11,787-gene intersection
    C = declared covariates, fixed before any tissue-molecular comparison:
        log-depth + the 5 CellViT micro-scale class fractions (composition)
    R = ridge_residualize(H, C)                      (cscd.residualize)
    dictionary = TopKSAE(n_atoms=20, k=4), 5-fold cross-fit by patient
    -> 23 x 20 patient-level molecular state matrix

Tissue side
    98 CellViT-derived micro/meso morphology features per spot, already
    computed by WS-J from HEST's own official nuclear segmentation (no RNA
    involved anywhere in this construction)
    standardised, dictionary = TopKSAE(n_atoms=8, k=2), 5-fold cross-fit by
    patient
    -> 23 x 8 patient-level tissue state matrix

Usage (on a100box, from ~):
    python -m cscd.build_breast --out-dir cscd/data/breast
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .dictionary import SAEConfig, TopKSAE
from .residualize import ridge_residualize


def randomized_pca(X, n_components, seed=0, n_iter=7, oversample=10):
    """Truncated PCA via randomized range finding (Halko et al.).

    Avoids a scikit-learn dependency, which conflicts with this box's system
    numpy on this venv. X is assumed already mean-centered by the caller.
    Returns (scores, explained_variance_ratio).
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    l = min(n_components + oversample, p)
    Omega = rng.standard_normal((p, l))
    Y = X @ Omega
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(X.T @ Y)
        Y, _ = np.linalg.qr(X @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    scores = (U * S)[:, :n_components]
    total_var = float((X ** 2).sum())
    ev_ratio = (S[:n_components] ** 2) / total_var
    return scores, ev_ratio

WSJ_ROOT = ("/lambda/nfs/geeg/biorag3_persistent_20260711/code/"
           "ws_wsJ_pergene_20260824/data/breast_stnet")

MICRO_CLASS_COLS = [9, 10, 11, 12, 13]   # neoplastic, connective, epithelial,
                                          # inflammatory, dead (micro scale)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def log(**kw):
    print(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}),
         flush=True)


def cross_fit_dictionary(R, pid, patients, n_atoms, k, epochs, seed, n_folds=5,
                         device="cpu"):
    """Same cross-fitting + Hungarian fold-alignment as real_discovery.py.

    Every patient gets a dictionary score from a fold that never saw them.
    Returns (ZP, per_fold_hist).
    """
    rng = np.random.default_rng(seed)
    folds = np.array([i % n_folds for i in rng.permutation(len(patients))])
    ZP = np.zeros((len(patients), n_atoms))
    ref = None
    hists = []
    for f in range(n_folds):
        tr_p = {patients[i] for i in range(len(patients)) if folds[i] != f}
        spot_tr = np.isin(pid, list(tr_p))
        sae = TopKSAE(R.shape[1], SAEConfig(n_atoms=n_atoms, k=k, epochs=epochs,
                                            seed=seed * 100 + f))
        sae.fit(R[spot_tr], device=device)
        Zf = sae.transform(R, device=device)
        Pf = np.zeros((len(patients), n_atoms))
        for i, p in enumerate(patients):
            Pf[i] = Zf[pid == p].mean(0)
        if ref is None:
            ref = Pf
        else:
            Ccost = np.abs(np.corrcoef(ref.T, Pf.T)[:n_atoms, n_atoms:])
            Ccost = np.nan_to_num(Ccost)
            r_i, c_i = linear_sum_assignment(-Ccost)
            order = np.empty(n_atoms, dtype=int)
            order[r_i] = c_i
            Pf = Pf[:, order]
        hold = folds == f
        ZP[hold] = Pf[hold]
        hists.append(sae.history_[-1])
    return ZP, hists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="cscd/data/breast")
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

    # ---- load WS-J's already-built substrate (read-only source) -----------
    log(event="load_start", root=WSJ_ROOT)
    counts = np.load(os.path.join(WSJ_ROOT, "counts.f32.npy"), mmap_mode="r")
    genes = np.load(os.path.join(WSJ_ROOT, "genes.npy"), allow_pickle=True)
    meta = pd.read_parquet(os.path.join(WSJ_ROOT, "meta.parquet"))
    tis = np.load(os.path.join(WSJ_ROOT, "morpheus_features.npy"))
    tis_names = np.load(os.path.join(WSJ_ROOT, "morpheus_feature_names.npy"),
                        allow_pickle=True).tolist()
    tis_names = [str(s) for s in tis_names]
    substrate_receipt = json.load(open(os.path.join(WSJ_ROOT, "substrate_receipt.json")))
    feature_receipt = json.load(open(os.path.join(WSJ_ROOT, "morpheus_feature_receipt.json")))

    counts = np.asarray(counts, dtype=np.float64)
    n_spots, n_genes = counts.shape
    pid = meta["patient_id"].to_numpy()
    patients = sorted(meta["patient_id"].unique().tolist())
    log(event="loaded", n_spots=n_spots, n_genes=n_genes, n_patients=len(patients),
        tissue_shape=list(tis.shape))
    assert tis.shape[0] == n_spots
    assert not np.isnan(tis).any() and not np.isinf(tis).any()

    # ---- QC floor, declared before any residualisation or matching --------
    # A handful of spots carry near-zero UMI depth (measured: 1 of 30,655 at
    # exactly zero); log1p(CPM) is undefined/degenerate there. Drop anything
    # under a fixed floor of 100 total counts, well below the cohort median
    # of 3,289, so this removes only spots too sparse to inform a state score.
    depth0 = counts.sum(1)
    qc_keep = depth0 >= args.min_depth
    n_dropped = int((~qc_keep).sum())
    log(event="qc_floor", min_depth=args.min_depth, n_dropped=n_dropped,
        n_kept=int(qc_keep.sum()))
    counts = counts[qc_keep]
    tis = tis[qc_keep]
    pid = pid[qc_keep]
    n_spots = counts.shape[0]

    # ---- molecular side -----------------------------------------------------
    depth = counts.sum(1)
    assert (depth > 0).all(), "zero-depth spot present after QC floor"
    cpm = counts / depth[:, None] * 1e4
    logcpm = np.log1p(cpm)
    del cpm

    log(event="pca_start", n_components=args.n_pca)
    Xc = logcpm - logcpm.mean(0)
    H, ev_ratio = randomized_pca(Xc, args.n_pca, seed=args.seed)
    ev_ratio_sum = float(ev_ratio.sum())
    log(event="pca_done", explained_var_ratio_sum=ev_ratio_sum)
    del logcpm, Xc

    log_depth = np.log1p(depth)
    C = np.column_stack([log_depth] + [tis[:, j] for j in MICRO_CLASS_COLS])
    C = (C - C.mean(0)) / np.maximum(C.std(0), 1e-9)
    covariate_names = (["log_depth"] +
                       [tis_names[j] for j in MICRO_CLASS_COLS])
    log(event="covariates", names=covariate_names)

    R, resid_info = ridge_residualize(H, C, seed=args.seed)
    log(event="residualised", alphas=[f["alpha"] for f in resid_info["per_fold"]])

    log(event="mol_dictionary_start", n_atoms=args.n_mol_atoms, k=args.mol_k)
    ZP_mol, mol_hist = cross_fit_dictionary(R, pid, patients, args.n_mol_atoms,
                                            args.mol_k, args.epochs, args.seed,
                                            device=args.device)
    log(event="mol_dictionary_done",
        final_mse=[h["mse"] for h in mol_hist], dead=[h["dead_atoms"] for h in mol_hist])

    # ---- tissue side ----------------------------------------------------------
    tis_z = (tis - tis.mean(0)) / np.maximum(tis.std(0), 1e-9)
    log(event="tis_dictionary_start", n_atoms=args.n_tis_atoms, k=args.tis_k)
    ZP_tis, tis_hist = cross_fit_dictionary(tis_z, pid, patients, args.n_tis_atoms,
                                            args.tis_k, args.epochs, args.seed,
                                            device=args.device)
    log(event="tis_dictionary_done",
        final_mse=[h["mse"] for h in tis_hist], dead=[h["dead_atoms"] for h in tis_hist])

    # ---- write outputs ----------------------------------------------------
    np.savez(os.path.join(out_dir, "molecular_states.npz"),
            scores=ZP_mol, patient_ids=np.array(patients),
            n_atoms=args.n_mol_atoms, k=args.mol_k)
    np.savez(os.path.join(out_dir, "tissue_states.npz"),
            scores=ZP_tis, patient_ids=np.array(patients),
            n_atoms=args.n_tis_atoms, k=args.tis_k,
            source_feature_names=np.array(tis_names))

    manifest = {
        "cohort": "hest_breast_stnet",
        "organ": "Breast",
        "dataset_title": "Integrating spatial gene expression and breast tumour "
                         "morphology via deep learning (He et al., ST-Net)",
        "source": "MahmoodLab/hest on HuggingFace (HEST-1k)",
        "license": "CC BY-NC-SA 4.0",
        "n_patients": len(patients), "n_samples": int(meta["sample_id"].nunique()),
        "n_spots": n_spots, "n_spots_dropped_qc": n_dropped, "n_genes": n_genes,
        "qc_min_depth": args.min_depth,
        "patients": patients,
        "raw_substrate_provenance": {
            "root": WSJ_ROOT, "note": "counts/genes/meta/tissue-feature matrices "
                                       "were already built by a prior workstream "
                                       "(WS-J) from HEST's own official assets; "
                                       "read-only here, not modified.",
            "substrate_receipt": substrate_receipt,
            "tissue_feature_receipt_summary": {
                k: v for k, v in feature_receipt.items()
                if k not in ("per_sample",)
            },
        },
        "molecular_side": {
            "normalisation": "CPM (1e4) then log1p over the 11,787-gene "
                             "cross-sample intersection",
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
                                        "mean, mirroring cscd/real_discovery.py"},
        },
        "tissue_side": {
            "route": "full CellViT-style nuclear segmentation (not the "
                     "pathology-encoder fallback) -- HEST's own official "
                     "per-nucleus segmentation, aggregated to 98 micro/meso "
                     "morphology + composition + spatial-organisation "
                     "features per spot by WS-J, using no RNA at any step",
            "n_raw_features": int(tis.shape[1]),
            "feature_code_sha256": feature_receipt.get("feature_code_sha256"),
            "dictionary": {"method": "TopK sparse autoencoder (cscd.dictionary.TopKSAE)",
                           "n_atoms": args.n_tis_atoms, "k": args.tis_k,
                           "epochs": args.epochs,
                           "cross_fit": "5-fold by patient, Hungarian-aligned, "
                                        "aggregated to patient mean"},
        },
        "seed": args.seed,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "cscd/build_breast.py",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    log(event="done", out_dir=out_dir)


if __name__ == "__main__":
    main()
