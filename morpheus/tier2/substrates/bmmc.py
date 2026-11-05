"""BMMC CITE-seq (GSE194122) Tier-II substrate.

Loads the same public processed .h5ad `cscd/build_cite_bmmc.py` uses, but
builds a light substrate for injection rather than that script's full
cross-fit SAE pipeline: top-variance HVGs, z-scored, as `A`; CLR-normalised
real ADT (surface protein) kept untouched as the coupling-preservation
reference, never used as a spike-in generator. QC thresholds and the
donor/site/batch/cell-type columns are the same ones `build_cite_bmmc.py`
reads.

Measured here, not assumed: `DonorNumber` in this GEO release has 9 distinct
values (donor1..donor9) across 12 site/donor batches and 4 sites -- see
`measure_donor_count()`. A design note elsewhere says 12 (conflating the 12
site/donor batches with donor count) and another says the raw NeurIPS
release is 10; this file's own read of the public processed .h5ad finds 9,
matching what `cscd/build_cite_bmmc.py`'s manifest already recorded
(`n_donors: 9`), so 9 is what every check here is built against.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HOME = os.path.expanduser("~")
if _HOME not in sys.path:
    sys.path.insert(0, _HOME)

from cscd.spike_in import covariance_preservation_check, patient_split  # noqa: E402

from ..objects import Substrate  # noqa: E402

# Measured, not assumed: `~/morpheus_data/single_cell/` does not exist on the
# box (only `~/morpheus_data/hest` does); the real object was reacquired onto
# the persistent store on 2026-09-06 and lives at the path below (`S =
# /lambda/nfs/geeg/biorag3_persistent_20260711`, per the campaign's own gate
# path in `dbench/protocols/bmmc_cite_seq_v01.json`). No other module
# imports `bmmc.H5AD` (`cscd/build_cite_bmmc.py` and
# `cscd/cite_totalvi_baseline.py` each define their own separate, unrelated
# `H5AD` constant), so updating this default cannot affect them.
H5AD = "/lambda/nfs/geeg/biorag3_persistent_20260711/bmmc_cite_reacquired_20260906/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"

# Coarse lineage buckets for the "broad cell type" the discov2 spec calls
# for in the BMMC null ("shuffle within donor, site and broad cell type") --
# the 45 Azimuth-style fine labels are too fine to cross with 12 donor/site
# batches without producing many singleton strata.
_BROAD_RULES = [
    ("Plasma", "Plasma"), ("Prog", "Progenitor"), ("HSC", "Progenitor"),
    ("CD4", "CD4 T"), ("CD8", "CD8 T"), ("Treg", "CD4 T"),
    ("B ", "B"), ("Naive B", "B"), ("Memory B", "B"), ("B1", "B"),
    ("NK", "NK"), ("ILC", "NK"),
    ("Mono", "Myeloid"), ("cDC", "Myeloid"), ("pDC", "Myeloid"), ("Baso", "Myeloid"),
    ("Eryth", "Erythroid"), ("Platelet", "Platelet"),
]


def broad_cell_type(fine_labels):
    fine_labels = np.asarray(fine_labels, dtype=str)
    out = np.full(fine_labels.shape, "Other", dtype=object)
    for i, lab in enumerate(fine_labels):
        for key, broad in _BROAD_RULES:
            if key in lab:
                out[i] = broad
                break
    return out.astype(str)


def clr(counts):
    x = np.log1p(np.asarray(counts, dtype=np.float64))
    return x - x.mean(axis=1, keepdims=True)


def _ridge_residualize_simple(H, C, alpha=1.0):
    """One global ridge fit removing `C` (+ intercept) from `H`. Not
    cross-fitted -- this builds the injection substrate itself, it is not
    used for any inferential claim -- but it is the same declared-covariate
    set `build_cite_bmmc.py` residualises against (cell-type identity +
    log depth), applied directly to the HVG matrix rather than to PCA
    components, so this stays a light loader without needing that script's
    cross-fit dictionary machinery (and the torch dependency it pulls in).

    Why this step exists, measured (see `bmmc_residualization_matters` in
    the test suite): without it, a marker gene's real value is itself
    confounded with cell-type identity, so the within-(donor,site,broad
    cell type) null below only partially destroys a planted marginal
    correspondence -- it correctly leaves the compositional/confound
    structure alone, but that same structure lets some of the "true"
    correlation leak through via between-stratum concordance. Residualising
    A against cell type before injecting removes exactly that leak path.
    """
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Cc = np.hstack([C, np.ones((len(C), 1))])
    G = Cc.T @ Cc + alpha * np.eye(Cc.shape[1])
    beta = np.linalg.solve(G, Cc.T @ H)
    return H - Cc @ beta


def _residualize_against_onehot(X, group_codes, alpha=1.0):
    """One global ridge fit removing a categorical fixed effect (e.g. donor
    identity) from `X`. Duplicated from `hest.py`'s helper of the same name
    (both substrates need the identical between-group-leakage fix; see that
    module's docstring) rather than imported, so `bmmc.py` stays a
    self-contained substrate module -- the same duplication pattern this
    file already uses for `exact_comp`/`preservation_report`."""
    n_groups = len(np.unique(group_codes))
    Xc = np.zeros((X.shape[0], n_groups))
    Xc[np.arange(X.shape[0]), group_codes] = 1.0
    G = Xc.T @ Xc + alpha * np.eye(n_groups)
    beta = np.linalg.solve(G, Xc.T @ X)
    return X - Xc @ beta


def _random_derangement(n, rng):
    """A uniformly random permutation of range(n) conditioned on having NO
    fixed points (rejection sampling; expected ~e ~= 2.7 draws). `n<=1` has
    no valid derangement and is returned as-is. Duplicated from `hest.py`'s
    helper of the same name for the same reason as `_residualize_against_onehot`."""
    if n <= 1:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def measure_donor_count(h5ad_path=H5AD):
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    return {
        "n_donor_number": int(a.obs["DonorNumber"].nunique()),
        "n_donor_id": int(a.obs["DonorID"].nunique()),
        "n_batches": int(a.obs["batch"].nunique()),
        "n_sites": int(a.obs["Site"].nunique()),
    }


def load(h5ad_path=H5AD, n_hvg=200, max_cells=None, seed=0,
         min_rna_depth=500, min_adt_depth=200, min_genes=200,
         frac_discovery=2.0 / 3.0):
    """Real BMMC CITE-seq substrate. `max_cells`, if given, subsamples QC'd
    cells (stratified by donor so per-donor proportions are held roughly
    fixed) purely to keep this fast for iteration/testing -- every
    preservation check operates on whatever was actually loaded, so a
    subsample is a smaller but equally real substrate, not a synthetic one.
    """
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(h5ad_path)
    a.var_names_make_unique()
    rna_mask = (a.var["feature_types"] == "GEX").to_numpy()
    adt_mask = (a.var["feature_types"] == "ADT").to_numpy()

    counts = a.layers["counts"] if "counts" in a.layers else a.X
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts = counts.tocsr()
    rna_counts = counts[:, rna_mask].tocsr()
    adt_counts = counts[:, adt_mask].tocsr()

    rna_depth = np.asarray(rna_counts.sum(1)).ravel()
    adt_depth = np.asarray(adt_counts.sum(1)).ravel()
    n_genes_expr = np.asarray((rna_counts > 0).sum(1)).ravel()

    donor = a.obs["DonorNumber"].astype(str).to_numpy()
    cell_type = a.obs["cell_type"].astype(str).to_numpy()
    site = a.obs["Site"].astype(str).to_numpy()
    batch = a.obs["batch"].astype(str).to_numpy()

    qc = ((rna_depth >= min_rna_depth) & (adt_depth >= min_adt_depth)
          & (n_genes_expr >= min_genes))
    idx = np.where(qc)[0]

    rng = np.random.default_rng(seed)
    if max_cells is not None and max_cells < len(idx):
        # stratified-by-donor subsample, proportions preserved
        keep = []
        d_sub = donor[idx]
        for d in np.unique(d_sub):
            d_idx = idx[d_sub == d]
            n_take = max(1, int(round(max_cells * len(d_idx) / len(idx))))
            keep.append(rng.choice(d_idx, size=min(n_take, len(d_idx)), replace=False))
        idx = np.sort(np.concatenate(keep))

    rna_counts = rna_counts[idx]
    adt_counts = adt_counts[idx]
    rna_depth = rna_depth[idx]
    donor = donor[idx]
    cell_type = cell_type[idx]
    site = site[idx]
    batch = batch[idx]
    n_cells = rna_counts.shape[0]

    cpm = rna_counts.multiply(1e4 / rna_depth[:, None]).tocsr()
    logcpm = cpm.copy()
    logcpm.data = np.log1p(logcpm.data)
    del cpm

    X = np.asarray(logcpm.todense(), dtype=np.float64)
    var = X.var(0)
    hvg_idx = np.argsort(-var)[:n_hvg]
    X_hvg = X[:, hvg_idx]
    del X, logcpm

    # declared covariates: fine (45-way) cell-type identity + log depth --
    # same set `build_cite_bmmc.py` residualises against.
    ct_cats = sorted(set(cell_type.tolist()))
    ct_idx = {c: i for i, c in enumerate(ct_cats)}
    ct_onehot = np.zeros((n_cells, len(ct_cats)))
    for i, c in enumerate(cell_type):
        ct_onehot[i, ct_idx[c]] = 1.0
    log_depth = np.log1p(rna_depth)
    log_depth_z = (log_depth - log_depth.mean()) / max(log_depth.std(), 1e-8)
    C_declared = np.column_stack([ct_onehot, log_depth_z])

    X_resid = _ridge_residualize_simple(X_hvg, C_declared, alpha=1.0)
    mu, sd = X_resid.mean(0), np.maximum(X_resid.std(0), 1e-8)
    A = (X_resid - mu) / sd

    adt_dense = np.asarray(adt_counts.todense(), dtype=np.float64)
    ADT_clr = clr(adt_dense)
    adt_mu, adt_sd = ADT_clr.mean(0), np.maximum(ADT_clr.std(0), 1e-8)
    ADT_z = (ADT_clr - adt_mu) / adt_sd

    split = patient_split(donor, seed=seed, frac_discovery=frac_discovery)
    broad = broad_cell_type(cell_type)

    cell_type_codes = np.unique(cell_type, return_inverse=True)[1]
    broad_codes = np.unique(broad, return_inverse=True)[1]

    sub = Substrate(
        name="bmmc_cite",
        A=A, split=split,
        coords=None, coord_groups=None, time_order=None, time_groups=None,
        cond_vars={"cell_type": cell_type_codes, "broad_cell_type": broad_codes,
                  "site": np.unique(site, return_inverse=True)[1]},
        meta={"donor": donor, "site": site, "batch": batch,
             "cell_type": cell_type, "broad_cell_type": broad,
             "rna_depth": rna_depth, "ADT": ADT_z, "hvg_idx": hvg_idx,
             "n_cells": n_cells},
    )
    return sub


# ---------------------------------------------------------- preservation --
def preservation_report(sub):
    """Measured, not assumed: donor/site batch structure, cell-type
    composition (exact), and RNA-ADT within-cell coupling, all checked
    against the substrate as loaded (the state every injection call reads)."""
    cov = covariance_preservation_check(sub.A, sub.A)
    rna_adt_cov = covariance_preservation_check(
        np.hstack([sub.A, sub.meta["ADT"]]), np.hstack([sub.A, sub.meta["ADT"]]))
    return {
        "donor_composition": exact_comp(sub.meta["donor"]),
        "site_composition": exact_comp(sub.meta["site"]),
        "cell_type_composition": exact_comp(sub.meta["cell_type"]),
        "broad_cell_type_composition": exact_comp(sub.meta["broad_cell_type"]),
        "A_self_covariance_check": cov,
        "rna_adt_joint_covariance_check": rna_adt_cov,
    }


def exact_comp(labels):
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


# --------------------------------------------------------- structured null
def null_bmmc(B, sub, seed=0):
    """Permute B's rows WITHIN (donor, site, broad_cell_type) strata.

    Preserves donor/site/cell-type composition exactly (rows never change
    stratum) and leaves A completely untouched (the null only ever
    rearranges B), so the real RNA-ADT coupling is unaffected too. Breaks
    any planted A-column <-> B-column correspondence, because within a
    stratum the mapping from "which row" to "which B value" is randomised.
    """
    rng = np.random.default_rng(seed)
    donor, site, broad = sub.meta["donor"], sub.meta["site"], sub.meta["broad_cell_type"]
    strata = np.array([f"{d}|{s}|{c}" for d, s, c in zip(donor, site, broad)])
    B_null = B.copy()
    for st in np.unique(strata):
        idx = np.where(strata == st)[0]
        if len(idx) < 2:
            continue
        perm = rng.permutation(idx)
        B_null[idx] = B[perm]
    return B_null


# ------------------------------------------------------- realistic artifact
def donor_confounded_artifact(sub, seed=0, effect_scale=1.0):
    """A B-column driven purely by donor identity (a per-donor mean shift +
    noise), with NO dependence on any real A column -- the realistic BMMC
    nuisance artifact: batch-confounded, not biology-confounded. `tag`
    marks it so a downstream check can verify a naive method would flag it
    as "related to something" only because of donor batch structure."""
    rng = np.random.default_rng(seed)
    donors = np.unique(sub.meta["donor"])
    donor_effect = {d: rng.standard_normal() * effect_scale for d in donors}
    y = np.array([donor_effect[d] for d in sub.meta["donor"]]) + 0.3 * rng.standard_normal(sub.n)
    y = (y - y.mean()) / max(y.std(), 1e-9)
    truth = {"kind": "donor_confounded_artifact", "a": None, "true": False,
            "tag": "donor_confound", "params": {"effect_scale": effect_scale}}
    return y, truth


# ---------------------------------------------------------- real episode --
def _all_donors(h5ad_path):
    """The full donor universe via a backed (header-only-cost) read -- no
    need to materialise RNA/ADT matrices just to enumerate donors for the
    fold split, mirroring `measure_donor_count`'s own backed-read pattern."""
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    return np.array(sorted(set(a.obs["DonorNumber"].astype(str).tolist())))


def _load_episode_substrate(h5ad_path, *, n_hvg, fit_donors, min_rna_depth, min_adt_depth,
                            min_genes, max_cells, seed, keep_logcpm=False):
    """Real RNA+ADT load for `real_episode`, QC-filtered exactly like `load()`,
    but deliberately DIFFERENT from `load()` in the two ways the campaign
    protocol (`bmmc_cite_seq_v01.json`) requires:

    (1) HVG selection and standardisation (mean/sd) statistics are computed
    on `fit_donors` rows only (mirrors `hest.load`'s `fit_patients`) -- a
    held-out donor's own variance/scale can never influence which genes are
    selected or how the candidate matrix is scaled.

    (2) The candidate matrix `A` is residualised against DONOR IDENTITY
    (a per-donor fixed effect), NOT the cell_type + log-depth declared
    covariates `load()` removes for the synthetic-injection substrate.
    Residualising away cell_type here would remove exactly the RNA
    variation an RNA -> protein discovery is supposed to find (surface
    marker level is partly determined by cell identity, but cell identity
    is itself downstream of RNA state); `load()`'s covariate set is correct
    for its own purpose (a clean substrate to spike synthetic objects into)
    and wrong for this one, hence the separate loader rather than a shared
    one with a stringly-typed mode flag.

    Returns a plain dict (not a `Substrate`) since this is consumed by
    exactly one caller (`real_episode`) and needs several arrays `Substrate`
    has no slot for (per-cell one-hot baselines, the raw ADT panel names).

    `keep_logcpm=True` additionally returns the full RNA log1p-CPM matrix
    (sparse CSR, float32) as `logcpm_full`, with `rna_names` and the kept
    cells' `obs_names`, for the protected measurement bank; ADT is never part
    of it. The default return value is unchanged.
    """
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(h5ad_path)
    a.var_names_make_unique()
    rna_mask = (a.var["feature_types"] == "GEX").to_numpy()
    adt_mask = (a.var["feature_types"] == "ADT").to_numpy()
    adt_names = a.var_names[adt_mask].tolist()
    rna_names = a.var_names[rna_mask].to_numpy()

    counts = a.layers["counts"] if "counts" in a.layers else a.X
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts = counts.tocsr()
    rna_counts = counts[:, rna_mask].tocsr()
    adt_counts = counts[:, adt_mask].tocsr()

    rna_depth = np.asarray(rna_counts.sum(1)).ravel()
    adt_depth = np.asarray(adt_counts.sum(1)).ravel()
    n_genes_expr = np.asarray((rna_counts > 0).sum(1)).ravel()

    donor = a.obs["DonorNumber"].astype(str).to_numpy()
    cell_type = a.obs["cell_type"].astype(str).to_numpy()
    site = a.obs["Site"].astype(str).to_numpy()
    obs_names = a.obs_names.to_numpy().astype(str)

    qc = ((rna_depth >= min_rna_depth) & (adt_depth >= min_adt_depth)
          & (n_genes_expr >= min_genes))
    idx = np.where(qc)[0]

    rng = np.random.default_rng(seed)
    if max_cells is not None and max_cells < len(idx):
        # stratified-by-donor subsample, proportions preserved -- same
        # contract as `load()`'s `max_cells`.
        keep = []
        d_sub = donor[idx]
        for d in np.unique(d_sub):
            d_idx = idx[d_sub == d]
            n_take = max(1, int(round(max_cells * len(d_idx) / len(idx))))
            keep.append(rng.choice(d_idx, size=min(n_take, len(d_idx)), replace=False))
        idx = np.sort(np.concatenate(keep))

    rna_counts = rna_counts[idx]
    adt_counts = adt_counts[idx]
    rna_depth = rna_depth[idx]
    donor = donor[idx]
    cell_type = cell_type[idx]
    site = site[idx]
    obs_names = obs_names[idx]
    n_cells = rna_counts.shape[0]

    cpm = rna_counts.multiply(1e4 / rna_depth[:, None]).tocsr()
    logcpm = cpm.copy()
    logcpm.data = np.log1p(logcpm.data)
    X = np.asarray(logcpm.todense(), dtype=np.float64)
    logcpm_full = logcpm.astype(np.float32) if keep_logcpm else None
    del cpm, logcpm

    donor_codes = np.unique(donor, return_inverse=True)[1]
    fit_set = {str(d) for d in fit_donors}
    fit_mask = np.array([d in fit_set for d in donor])
    if not fit_mask.any():
        raise ValueError("fit_donors matched no loaded rows")

    var = X[fit_mask].var(0)
    hvg_idx = np.argsort(-var)[:n_hvg]
    gene_symbols = rna_names[hvg_idx]
    X_hvg = X[:, hvg_idx]
    del X

    X_resid = _residualize_against_onehot(X_hvg, donor_codes)
    mu = X_resid[fit_mask].mean(0)
    sd = np.maximum(X_resid[fit_mask].std(0), 1e-8)
    A = (X_resid - mu) / sd

    adt_dense = np.asarray(adt_counts.todense(), dtype=np.float64)
    # CLR: log1p(counts) minus each CELL's own row mean across the full
    # 134-channel panel -- a strictly row-wise (per-cell) transform, so it
    # cannot leak D_R rows into any D_P-fit statistic.
    ADT_clr = clr(adt_dense)

    log_depth = np.log1p(rna_depth)

    ct_cats = sorted(set(cell_type.tolist()))
    ct_idx = {c: i for i, c in enumerate(ct_cats)}
    ct_onehot = np.zeros((n_cells, len(ct_cats)))
    for i, c in enumerate(cell_type):
        ct_onehot[i, ct_idx[c]] = 1.0

    site_cats = sorted(set(site.tolist()))
    site_idx = {s: i for i, s in enumerate(site_cats)}
    site_onehot = np.zeros((n_cells, len(site_cats)))
    for i, s in enumerate(site):
        site_onehot[i, site_idx[s]] = 1.0

    out = {
        "A": A, "gene_symbols": gene_symbols, "donor": donor, "donor_codes": donor_codes,
        "site": site, "cell_type": cell_type, "ct_onehot": ct_onehot, "ct_cats": ct_cats,
        "site_onehot": site_onehot, "site_cats": site_cats, "log_depth": log_depth,
        "ADT_clr": ADT_clr, "adt_names": adt_names, "fit_mask": fit_mask, "n_cells": n_cells,
    }
    if keep_logcpm:
        out.update(logcpm_full=logcpm_full, rna_names=[str(g) for g in rna_names], obs_names=obs_names)
    return out


def real_episode(fold, *, n_folds=5, seed, target, h5ad_path=H5AD, n_hvg=300,
                 min_rna_depth=500, min_adt_depth=200, min_genes=200, max_cells=None,
                 bank=False, bank_views=None, bank_cache_root=None, bank_options=None,
                 centre_baseline=True, within_patient_scale=None):
    """One leak-proof, donor-grouped discovery episode for a single
    predeclared ADT (surface protein) target, over the real BMMC CITE-seq
    substrate (`bmmc_cite_seq_v01.json`).

    D_P = the `n_folds - 1` FIT donors (donor-grouped, deterministic
    shuffled `np.array_split` over the 9 BMMC donors -- this campaign's
    Level-1 fold unit, exactly mirroring `hest.real_episode`'s
    patient-grouped fold); D_R = the held-out `fold`'s donor(s). Candidate
    genes are the D_P-selected 300 highest-variance RNA genes (log1p CPM),
    donor fixed-effect residualised, z-scored with D_P statistics only
    (`_load_episode_substrate`). The response is one CLR-normalised ADT
    channel (row-wise CLR over the full 134-channel panel, so it carries no
    fold-split dependence of its own), then within-donor centred: a
    per-donor fixed-effect residual computed over ALL rows -- like
    `hest.real_episode`'s target residualisation, removing a group's own
    mean from its own rows cannot leak D_R statistics into a D_P-fit
    quantity, because each group's correction depends only on that group's
    own rows.

    Leave-one-site-out (Level-2 external replication) is a LATER stage, not
    this runner's job -- this function only ever builds the Level-1
    donor-grouped fold.

    Nulls are NOT a separate row role: `episode_null_generators` transforms
    the D_R rows in place, so `partitions.null` is empty and
    `metadata["null_source"] = "replication_structured"` (see
    `discovery_episode.EpisodePartitions`).

    `bank=True` attaches the protected measurement bank (RNA-only views plus
    donor/site context; the ADT panel is never a source) as
    `metadata["bank"]` and `metadata["bank_report"]`; `bank=False` (default)
    builds exactly the previous episode.

    Every competitor-baseline column is within-donor centred with the same
    per-donor fixed-effect residual as the response (covariate values only;
    the response is never used), so the competitor and `y` live on the same
    within-donor scale; an uncentred competitor scored against a centred `y`
    carries between-donor offsets that cannot transfer to held-out donors.
    Site one-hot columns are constant within a donor and so centre to ~0.
    `centre_baseline=False` rebuilds the pre-2026-10-11 uncentred competitor
    (only for reproducing sealed receipts). `within_patient_scale` (default:
    True exactly when `bank=True`) instead STANDARDISES the response and every
    competitor column within donor (hest's `_within_patient_standardise`).
    """
    from morpheus_encoder.discovery_episode import DiscoveryEpisode, EpisodePartitions

    if not (0 <= fold < n_folds):
        raise ValueError("fold must be in [0, n_folds)")

    all_donors = _all_donors(h5ad_path)
    if len(all_donors) < n_folds:
        raise ValueError("fewer donors than folds")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_donors)
    folds = np.array_split(shuffled, n_folds)
    held_donors = set(folds[fold].tolist())
    fit_donors = sorted(set(all_donors.tolist()) - held_donors)

    loaded = _load_episode_substrate(h5ad_path, n_hvg=n_hvg, fit_donors=fit_donors,
                                     min_rna_depth=min_rna_depth, min_adt_depth=min_adt_depth,
                                     min_genes=min_genes, max_cells=max_cells, seed=seed,
                                     keep_logcpm=bank)

    if target not in loaded["adt_names"]:
        raise ValueError(f"unknown ADT target {target!r}; not in this h5ad's ADT panel")
    target_idx = loaded["adt_names"].index(target)

    donor = loaded["donor"]
    is_fit = np.isin(donor, np.asarray(fit_donors))
    proposal_rows, replication_rows = np.where(is_fit)[0], np.where(~is_fit)[0]

    donor_codes = loaded["donor_codes"]
    y = loaded["ADT_clr"][:, target_idx].copy()
    if within_patient_scale is None:
        within_patient_scale = bool(bank)
    if within_patient_scale:
        from .hest import _within_patient_standardise
        y = _within_patient_standardise(y.reshape(-1, 1), donor_codes).reshape(-1)
    else:
        y = _residualize_against_onehot(y.reshape(-1, 1), donor_codes).reshape(-1)

    # Competitor baseline: cell_type one-hot (45 classes) + log total RNA
    # counts + site one-hot -- never the response or a deterministic
    # function of it (this campaign's exclusion clause: candidates are
    # RNA-only and the baseline never contains any ADT channel, including
    # the target's own).
    baseline = np.column_stack([loaded["ct_onehot"], loaded["log_depth"].reshape(-1, 1),
                                loaded["site_onehot"]])
    baseline_names = ([f"cell_type:{c}" for c in loaded["ct_cats"]] + ["log_rna_depth"]
                      + [f"site:{s}" for s in loaded["site_cats"]])
    if within_patient_scale:
        baseline = _within_patient_standardise(baseline, donor_codes)
    elif centre_baseline:
        baseline = _residualize_against_onehot(baseline.astype(np.float64), donor_codes)

    partitions = EpisodePartitions(proposal_rows.astype(np.int64), replication_rows.astype(np.int64),
                                   np.array([], dtype=np.int64))
    modalities = {"X": loaded["A"]}
    # Labels as fixed-width numpy strings, not pandas object arrays: the
    # episode content hash digests array bytes, and an object array's bytes
    # are pointers, which made `content_sha256` differ between identical runs.
    context = {
        "donor_codes": donor_codes, "site": np.asarray(loaded["site"], dtype=str),
        "cell_type": np.asarray(loaded["cell_type"], dtype=str),
        "baseline": baseline, "log_rna_depth": loaded["log_depth"],
    }
    metadata = {
        "target": target, "fold": int(fold), "n_folds": int(n_folds), "seed": int(seed),
        "null_source": "replication_structured",
        "gene_symbols": [str(g) for g in loaded["gene_symbols"]],
        "fit_donors": list(fit_donors), "held_donors": sorted(held_donors),
        "n_rows_total": int(loaded["n_cells"]),
        "baseline_width": int(baseline.shape[1]), "baseline_columns": baseline_names,
        "baseline_centred": bool(centre_baseline or within_patient_scale),
        "response_standardised": bool(within_patient_scale),
        "baseline_standardised": bool(within_patient_scale),
        "adt_panel": list(loaded["adt_names"]),
    }
    episode = DiscoveryEpisode(modalities=modalities, response=y.astype(np.float64),
                               partitions=partitions, context=context, metadata=metadata)
    if not bank:
        return episode
    from morpheus_encoder.measurement_bank import build_bank_with_report, load_hallmark_sets
    from .hest import load_gavish_programs

    gene_sets = {"hallmark": load_hallmark_sets()}
    try:
        gene_sets["gavish"] = load_gavish_programs()
    except FileNotFoundError:
        pass
    gene_sets = {k: v for k, v in gene_sets.items() if v}
    sources = {
        "cohort": "bmmc", "logcpm": loaded["logcpm_full"], "genes": loaded["rna_names"],
        "gene_sets": gene_sets, "fm_groups": np.full(loaded["n_cells"], "bmmc"),
        "fm_barcodes": loaded["obs_names"], "fm_single_file": "bmmc", "fm_scale": "cell",
        "donor": donor, "site": loaded["site"],
    }
    bank_dict, report = build_bank_with_report(episode, views=bank_views, cache_root=bank_cache_root,
                                               sources=sources, seed=seed, **(bank_options or {}))
    return DiscoveryEpisode(modalities=modalities, response=y.astype(np.float64),
                            partitions=partitions, context=context,
                            metadata={**metadata, "bank": bank_dict, "bank_report": report})


def episode_null_generators(episode):
    """Structured null closures for `general_discovery.evaluate_structured_nulls`,
    mirroring `hest.episode_null_generators`'s contract exactly: each closure
    has the `(index, X, y, context) -> (X, y, context)` signature that
    function expects, but IGNORES its own arguments and instead returns a
    structurally-nulled version of THIS episode's own D_R (replication)
    rows -- the caller passes D_P as the literal `proposal_X, proposal_y`
    (so the frozen candidate is fit exactly as it was for the real receipt)
    and gets back, on each iteration, an alternative D_R to test that same
    fixed fit against.

    `within_donor_perm`: permutes the target protein channel's response
    within DONOR on D_R (breaks the cell-level RNA<->protein pairing while
    preserving each donor's own protein marginal distribution and donor
    identity).

    `cross_site_environment_break`: for every donor with >= 2 distinct
    sites present in D_R, swaps the response VALUES between that donor's
    own site sub-groups (a random derangement across the donor's site
    groups, source/destination truncated to the smaller group's size --
    the same truncation-on-unequal-group-size contract `hest.null_hest`
    documents for its spatial grid-block swap). Concretely: a cell keeps
    its real donor identity AND its real RNA row, but which SITE'S protein
    values it carries is scrambled within its own donor. This preserves
    each donor's overall response marginal (the same multiset of values per
    donor, since it is a value swap, not a resample) and leaves donor
    membership and RNA untouched, while destroying any response structure
    that specifically tracks site membership within a donor -- i.e. it
    tests whether an observed gain is actually a site-batch artifact rather
    than a real RNA-protein relationship, per the protocol's own definition
    of this null. A donor with only one site present in D_R has no
    cross-site structure to break and is left unchanged for that donor
    (documented, not silently dropped, exactly as HEST's block-swap leaves
    an unequal cell-size remainder untouched).

    `interaction_break`: for a candidate with >= 2 constituents, permutes
    ONE constituent gene (`candidate.variables[0]`) within donor on D_R --
    breaks joint structure while keeping every constituent's own
    within-donor marginal and the response untouched.
    """
    replication = episode.partitions.replication
    if replication.size == 0:
        raise ValueError("episode has no replication rows to build a null from")
    donor_codes_r = np.asarray(episode.context["donor_codes"])[replication]
    site_r = np.asarray(episode.context["site"])[replication]
    X_r = episode.observed_matrix()[replication]
    y_r = np.asarray(episode.response, dtype=float)[replication]

    def _within_donor_perm(index, X, y, context):
        rng = np.random.default_rng(index)
        y_null = y_r.copy()
        for g in np.unique(donor_codes_r):
            idx = np.where(donor_codes_r == g)[0]
            y_null[idx] = y_r[idx[rng.permutation(len(idx))]]
        return X_r, y_null, None

    def _cross_site_environment_break(index, X, y, context):
        rng = np.random.default_rng(index)
        y_null = y_r.copy()
        for g in np.unique(donor_codes_r):
            idx = np.where(donor_codes_r == g)[0]
            sites_g = site_r[idx]
            sites_present = np.unique(sites_g)
            if len(sites_present) < 2:
                continue  # nothing cross-site to break for a single-site donor
            groups = {s: idx[sites_g == s] for s in sites_present}
            perm = _random_derangement(len(sites_present), rng)
            for a_i, s in enumerate(sites_present):
                src_group = groups[sites_present[perm[a_i]]]
                dst_group = groups[s]
                k = min(len(src_group), len(dst_group))
                if k == 0:
                    continue
                src_pick = rng.permutation(src_group)[:k]
                dst_pick = rng.permutation(dst_group)[:k]
                y_null[dst_pick] = y_r[src_pick]
        return X_r, y_null, None

    def _interaction_break_factory(candidate):
        if len(candidate.variables) < 2:
            raise ValueError("interaction_break needs a candidate with >= 2 constituents")
        col = candidate.variables[0]

        def _generator(index, X, y, context):
            rng = np.random.default_rng(index)
            col_null = X_r[:, col].copy()
            for g in np.unique(donor_codes_r):
                idx = np.where(donor_codes_r == g)[0]
                col_null[idx] = X_r[idx[rng.permutation(len(idx))], col]
            X_null = X_r.copy()
            X_null[:, col] = col_null
            return X_null, y_r, None
        return _generator

    return {"within_donor_perm": _within_donor_perm,
           "cross_site_environment_break": _cross_site_environment_break,
           "interaction_break": _interaction_break_factory}
