"""HEST-1k spatial transcriptomics Tier-II substrate: two real cohorts,
ccRCC (24 Visium patients) and kidney_injury (23 samples / 22 patients).

Reads the `{prefix}{id}_st.h5ad` + `{prefix}{id}_metadata.json` files
acquired into `~/morpheus_data/hest/{cohort}/` (patient identity comes from
`metadata["patient"]`, NOT the accession index, which is just download
order). ccRCC's 24 accessions are ALL tissue="Lymph node" -- but that tag
describes the HISTOLOGICAL CHARACTER of the profiled regions, not the
anatomical specimen site: this cohort is Meylan et al. 2022, Immunity,
PMID 35231421 ("Tertiary lymphoid structures generate and propagate
anti-tumor antibody-producing plasma cells in renal cell cancer"), which
used spatial transcriptomics on intratumoral tertiary lymphoid structures
(TLS) WITHIN PRIMARY ccRCC TUMOUR TISSUE, not metastatic/tumour-draining
lymph nodes -- see `subseries`, e.g. "ccRCC tumor frozen_c_57", confirming
the specimen is tumour. Corrected here from an earlier, wrong reading of
this same "Lymph node" tag as the anatomical site (flagged by the Tier III
curation agent against the source paper). Either way, this cohort's
per-sample "tissue"/"organ" field carries zero diversity (every accession
reads the same TLS-histology tag), so sample-preparation method
(frozen/FFPE) stands in as its real per-sample categorical axis.
kidney_injury DOES have real tissue/disease diversity
(17 "Diseased" + 6 "Healthy" samples, one patient contributing both a
Medulla and a Cortex sample) -- both cohorts' real per-sample fields are
captured below rather than assumed to match ccRCC's shape.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_HOME = os.path.expanduser("~")
if _HOME not in sys.path:
    sys.path.insert(0, _HOME)

from cscd.calibrate import block_permutation_index  # noqa: E402
from cscd.spike_in import covariance_preservation_check, patient_split  # noqa: E402
from dbench.util import knn_neighbors  # noqa: E402

from ..objects import Substrate  # noqa: E402
from ..preservation import moran_i  # noqa: E402

HEST_ROOT = os.path.join(_HOME, "morpheus_data", "hest")
COHORTS = {
    "ccrcc": {"dir": os.path.join(HEST_ROOT, "ccrcc"), "prefix": "INT"},
    "kidney_injury": {"dir": os.path.join(HEST_ROOT, "kidney_injury"), "prefix": "NCBI"},
}
DATA_DIR = COHORTS["ccrcc"]["dir"]  # kept for backwards compatibility


def measure_available_samples(cohort="ccrcc", data_dir=None):
    cfg = COHORTS[cohort]
    d = data_dir or cfg["dir"]
    prefix = cfg["prefix"]
    return sorted(
        int(os.path.basename(p)[len(prefix):].split("_")[0])
        for p in glob.glob(os.path.join(d, f"{prefix}*_st.h5ad"))
    )


def _residualize_against_onehot(X, group_codes, alpha=1.0):
    """One global ridge fit removing a categorical fixed effect (e.g.
    patient identity) from `X`. Shared by `load()` (for the molecular
    substrate) and `load_image_embeddings()` (for morphology embeddings) so
    both get the same between-group-leakage fix, not two copies of it."""
    n_groups = len(np.unique(group_codes))
    Xc = np.zeros((X.shape[0], n_groups))
    Xc[np.arange(X.shape[0]), group_codes] = 1.0
    G = Xc.T @ Xc + alpha * np.eye(n_groups)
    beta = np.linalg.solve(G, Xc.T @ X)
    return X - Xc @ beta


def _within_patient_standardise(X, group_codes, eps=1e-8):
    """Exact per-group standardisation: each column minus its group mean,
    divided by its own within-group sd (ddof=0, floored at `eps`). Every
    column is scaled by its OWN values only -- a covariate never sees the
    response. Used for v0.4 episodes, where both the response and the
    competitor must be on a comparable within-patient scale."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    for g in np.unique(group_codes):
        m = group_codes == g
        block = X[m] - X[m].mean(axis=0)
        sd = block.std(axis=0)
        out[m] = block / np.maximum(sd, eps)
    return out


HEST_LOAD_CACHE_VERSION = "hest_load_v1_20261011"
HEST_LOAD_LOCAL_CACHE = os.path.join(_HOME, "cache", "hest_load")


def _is_real_cohort_data(cohort, data_dir):
    real = os.path.realpath(data_dir)
    return real == os.path.realpath(COHORTS[cohort]["dir"]) or real.startswith(
        os.path.realpath("/lambda/nfs/geeg/biorag3_persistent_20260711") + os.sep)


def _load_cache_key(cohort, d, sample_ids, params):
    """Params digest for one `load` call: cohort, every input file's identity
    (name, size, mtime_ns of each sample's h5ad and metadata json), the
    resolved sample list, all load parameters (incl. fit patients and the
    programme gene sets) and the code version."""
    import hashlib
    prefix = COHORTS[cohort]["prefix"]
    files = []
    for sid in sample_ids:
        for suffix in ("_st.h5ad", "_metadata.json"):
            path = os.path.join(d, f"{prefix}{sid}{suffix}")
            if os.path.exists(path):
                st = os.stat(path)
                files.append([os.path.basename(path), int(st.st_size), int(st.st_mtime_ns)])
    record = {"cohort": cohort, "data_dir": os.path.realpath(d), "files": files,
              "sample_ids": [str(x) for x in sample_ids], "code_version": HEST_LOAD_CACHE_VERSION,
              **params}
    blob = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20], record


def _substrate_to_arrays(sub):
    """Flatten a loaded Substrate into named arrays plus a JSON header, exactly
    (no dtype changes; sparse matrices kept as CSR parts)."""
    import scipy.sparse as sp
    arrays, header = {}, {"name": sub.name, "fields": {}, "cond_vars": [], "meta": {}}
    for f in ("A", "split", "coords", "coord_groups", "time_order", "time_groups"):
        v = getattr(sub, f)
        header["fields"][f] = v is not None
        if v is not None:
            arrays[f"f__{f}"] = np.asarray(v)
    for k, v in sub.cond_vars.items():
        header["cond_vars"].append(k)
        arrays[f"c__{k}"] = np.asarray(v)
    for k, v in sub.meta.items():
        if sp.issparse(v):
            v = v.tocsr()
            header["meta"][k] = {"kind": "csr", "shape": list(v.shape)}
            arrays[f"m__{k}__data"], arrays[f"m__{k}__indices"], arrays[f"m__{k}__indptr"] = (
                v.data, v.indices, v.indptr)
        elif isinstance(v, np.ndarray):
            if v.dtype == object:
                raise TypeError(f"meta[{k!r}] is an object array; not cacheable exactly")
            header["meta"][k] = {"kind": "array"}
            arrays[f"m__{k}"] = v
        else:
            header["meta"][k] = {"kind": "json", "value": v,
                                 "type": type(v).__name__}
    return arrays, header


def _substrate_from_arrays(z, header):
    import scipy.sparse as sp
    fields = {f: (z[f"f__{f}"] if present else None) for f, present in header["fields"].items()}
    meta = {}
    for k, spec in header["meta"].items():
        if spec["kind"] == "csr":
            meta[k] = sp.csr_matrix((z[f"m__{k}__data"], z[f"m__{k}__indices"], z[f"m__{k}__indptr"]),
                                    shape=tuple(spec["shape"]))
        elif spec["kind"] == "array":
            meta[k] = z[f"m__{k}"]
        else:
            v = spec["value"]
            if spec["type"] == "int" and v is not None:
                v = int(v)
            meta[k] = v
    return Substrate(name=header["name"], cond_vars={k: z[f"c__{k}"] for k in header["cond_vars"]},
                     meta=meta, **fields)


def load(cohort="ccrcc", data_dir=None, sample_ids=None, n_hvg=300, seed=0,
         frac_discovery=2.0 / 3.0, min_depth=200, fit_patients=None, program_sets=None,
         keep_logcpm=False, cache_root="default"):
    """`_load_uncached` behind an exact on-disk cache.

    One file per distinct call at `<cache_root>/<cohort>/<key>.npz` (key over
    the input files' identity, every parameter and the code version) with a
    `MANIFEST.json` holding the params and the file's sha256, verified on every
    read; a mismatch recomputes. `cache_root="default"` uses `~/cache/hest_load`
    (local disk, not the store) when that directory exists and the data are the
    real cohort (default mirror or the store); `None` disables it, and test
    trees never use it by default. A cached load is bit-identical to a fresh
    one (`tests: test_hest_load_cache_is_bit_exact`).
    """
    cfg = COHORTS[cohort]
    d = data_dir or cfg["dir"]
    kwargs = dict(cohort=cohort, data_dir=d, sample_ids=sample_ids, n_hvg=n_hvg, seed=seed,
                  frac_discovery=frac_discovery, min_depth=min_depth, fit_patients=fit_patients,
                  program_sets=program_sets, keep_logcpm=keep_logcpm)
    if cache_root == "default":
        cache_root = (HEST_LOAD_LOCAL_CACHE if os.path.isdir(HEST_LOAD_LOCAL_CACHE)
                      and _is_real_cohort_data(cohort, d) else None)
    if cache_root is None:
        return _load_uncached(**kwargs)
    resolved = sample_ids if sample_ids is not None else measure_available_samples(cohort=cohort, data_dir=d)
    params = {"n_hvg": int(n_hvg), "seed": int(seed), "frac_discovery": float(frac_discovery),
              "min_depth": float(min_depth), "keep_logcpm": bool(keep_logcpm),
              "fit_patients": None if fit_patients is None else sorted(str(p) for p in fit_patients),
              "program_sets": program_sets}
    key, record = _load_cache_key(cohort, d, resolved, params)
    cache_dir = os.path.join(cache_root, cohort)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.npz")
    name = os.path.basename(path)
    entry = _read_multiscale_manifest(cache_dir).get(name)
    if entry and os.path.exists(path) and _sha256_file(path) == entry.get("sha256"):
        with np.load(path, allow_pickle=False) as z:
            return _substrate_from_arrays(z, entry["header"])
    sub = _load_uncached(**{**kwargs, "sample_ids": resolved})
    arrays, header = _substrate_to_arrays(sub)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    _update_multiscale_manifest(cache_dir, name, {"params": record, "header": header,
                                                  "sha256": _sha256_file(path)},
                                version=HEST_LOAD_CACHE_VERSION)
    return sub


def _load_uncached(cohort="ccrcc", data_dir=None, sample_ids=None, n_hvg=300, seed=0,
                   frac_discovery=2.0 / 3.0, min_depth=200, fit_patients=None, program_sets=None,
                   keep_logcpm=False):
    """Real spatial substrate for one HEST cohort, concatenated over patients.

    `sample_ids`, if given, restricts to a subset of accessions (purely to
    bound runtime while iterating; the default is every accession the
    acquired data supports for this cohort).

    `fit_patients`, if given, restricts HVG SELECTION and the STANDARDISATION
    (mean/sd) statistics to those patients' rows only. Every requested sample
    is still loaded and returned -- a caller such as `real_episode` needs
    every fold's rows in one substrate to build a leak-proof proposal/
    replication split -- but a held-out patient's own variance/scale can no
    longer influence which genes are selected or how `A` is scaled. Patient
    identity residualisation (`_residualize_against_onehot`, below) is
    unaffected by this: it is a per-patient fixed effect whose normal
    equations decouple by group, i.e. within-patient centring, and is
    leak-free regardless of which patients are "fit" (see
    `tests/test_hest.py::test_patient_residualization_removes_between_patient_variance`).

    `program_sets`, if given, is a list of Gavish-style metaprogram dicts
    (`{"mp_id"/"name", "genes": [...]}`); each is scored as the per-spot mean
    log1p-CPM of its genes -- matched by symbol against the FULL gene panel,
    BEFORE the HVG cut, so a program's score does not depend on which 300
    genes happened to be the highest-variance ones -- and written to
    `meta["program_scores"]` ((n_spots, n_programs)) and
    `meta["program_names"]`. A program with no genes present in this
    cohort's common gene set gets an all-zero column rather than raising,
    since coverage varies by cohort/HVG-independent gene panel overlap.

    `keep_logcpm=True` additionally keeps the full common-gene log1p-CPM
    matrix (sparse CSR, float32) as `meta["logcpm_full"]` with its symbols in
    `meta["all_genes"]`, for the protected measurement bank. Nothing else
    changes; the default leaves `meta` exactly as before.
    """
    import anndata as ad
    import scipy.sparse as sp

    cfg = COHORTS[cohort]
    d = data_dir or cfg["dir"]
    prefix = cfg["prefix"]
    if sample_ids is None:
        sample_ids = measure_available_samples(cohort=cohort, data_dir=d)

    ads, patients, tissues, sids, preps, disease, barcodes, spot_diameters = ([], [], [], [], [], [], [], [])
    for sid in sample_ids:
        sid_str = f"{prefix}{sid}"
        h5ad_path = os.path.join(d, f"{sid_str}_st.h5ad")
        meta_path = os.path.join(d, f"{sid_str}_metadata.json")
        if not (os.path.exists(h5ad_path) and os.path.exists(meta_path)):
            continue
        a = ad.read_h5ad(h5ad_path)
        with open(meta_path) as fh:
            m = json.load(fh)
        depth = np.asarray(a.obs["total_counts"].to_numpy(), dtype=np.float64)
        keep = depth >= min_depth
        if keep.sum() < 10:
            continue
        a = a[keep].copy()
        try:
            diameter = float(a.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"])
        except (KeyError, TypeError, ValueError):
            # The ordinary substrate remains usable when a test fixture or an
            # unusual assay lacks Visium geometry.  Only the new radius-bank
            # pathway requires this value and will refuse an uncalibrated row.
            diameter = float("nan")
        ads.append(a)
        patients.extend([str(m["patient"])] * a.n_obs)
        tissues.extend([str(m.get("tissue", m.get("organ", "unknown")))] * a.n_obs)
        disease.extend([str(m.get("disease_state", "unknown"))] * a.n_obs)
        subseries = str(m.get("subseries", ""))
        prep = "frozen" if "frozen" in subseries else (
            "ffpe" if "ffpe" in subseries else "unknown")
        preps.extend([prep] * a.n_obs)
        sids.extend([sid_str] * a.n_obs)
        barcodes.extend(a.obs_names.tolist())
        spot_diameters.extend([diameter] * a.n_obs)

    common_genes = set(ads[0].var_names)
    for a in ads[1:]:
        common_genes &= set(a.var_names)
    common_genes = sorted(common_genes)

    patients = np.array(patients)
    tissues = np.array(tissues)
    preps = np.array(preps)
    disease = np.array(disease)
    sample_id_arr = np.array(sids)
    barcode_arr = np.array(barcodes)
    spot_diameter_arr = np.asarray(spot_diameters, dtype=np.float64)

    depths, coords, logcpm_blocks = [], [], []
    for a in ads:
        a = a[:, common_genes]
        counts = a.X
        if not sp.issparse(counts):
            counts = sp.csr_matrix(counts)
        counts = counts.tocsr()
        d = np.asarray(counts.sum(1)).ravel()
        d = np.maximum(d, 1.0)
        cpm = counts.multiply(1e4 / d[:, None]).tocsr()
        cpm.data = np.log1p(cpm.data)
        # Stay sparse (float64 data) until the HVG cut: densifying all ~36k
        # genes for 24 samples cost 35-48 GB per worker and got cells OOM-killed.
        logcpm_blocks.append(cpm.astype(np.float64))
        depths.append(d)
        coords.append(np.asarray(a.obsm["spatial"], dtype=np.float64))

    X = sp.vstack(logcpm_blocks).tocsr()
    depth = np.concatenate(depths)
    coords_all = np.vstack(coords)
    del logcpm_blocks, ads

    if fit_patients is not None:
        fit_set = {str(p) for p in fit_patients}
        fit_mask = np.array([p in fit_set for p in patients])
        if not fit_mask.any():
            raise ValueError("fit_patients matched no loaded rows")
    else:
        fit_mask = np.ones(len(patients), dtype=bool)

    program_scores = program_names = None
    if program_sets:
        gene_pos = {g: i for i, g in enumerate(common_genes)}
        program_scores = np.zeros((X.shape[0], len(program_sets)), dtype=np.float64)
        program_names = []
        for pi, prog in enumerate(program_sets):
            genes = prog.get("genes", [])
            cols = [gene_pos[g] for g in genes if g in gene_pos]
            program_names.append(str(prog.get("name", prog.get("mp_id", pi))))
            if cols:
                program_scores[:, pi] = np.asarray(X[:, cols].mean(axis=1)).ravel()

    X_fit = X[fit_mask]
    mean_fit = np.asarray(X_fit.mean(axis=0)).ravel()
    var = np.asarray(X_fit.multiply(X_fit).mean(axis=0)).ravel() - mean_fit ** 2
    del X_fit
    hvg_idx = np.argsort(-var)[:n_hvg]
    gene_symbols = np.asarray(common_genes)[hvg_idx]
    X_hvg = np.asarray(X[:, hvg_idx].todense(), dtype=np.float64)
    logcpm_full = X.astype(np.float32) if keep_logcpm else None
    del X

    patient_codes = np.unique(patients, return_inverse=True)[1]

    # Residualise against patient identity (fixed effect) before this
    # becomes the injection substrate. Measured, not assumed (see
    # `tests/test_hest.py::test_between_patient_variance_would_have_leaked`):
    # 68% of this cohort's top-variance HVG's variance is BETWEEN patients,
    # not within -- an unresidualised A would make an injected "within-
    # patient spatial" relationship partly a between-patient batch effect,
    # which a within-patient-only spatial null preserves BY DESIGN (patient
    # identity must stay intact), so the object would never look absent
    # after the null through no fault of the null itself. Depth is
    # deliberately NOT residualised out here (unlike BMMC, which does
    # remove depth): HEST's realistic artifact class is depth-confounded,
    # so depth must remain a live nuisance axis in A.
    X_resid = _residualize_against_onehot(X_hvg, patient_codes)

    # Standardisation statistics are measured on FIT patients only (see the
    # `fit_patients` docstring above); a caller not using folds gets the old
    # behaviour exactly, since `fit_mask` is then all-True.
    mu = X_resid[fit_mask].mean(0)
    sd = np.maximum(X_resid[fit_mask].std(0), 1e-8)
    A = (X_resid - mu) / sd
    split = patient_split(patients, seed=seed, frac_discovery=frac_discovery)
    tissue_codes = np.unique(tissues, return_inverse=True)[1]
    prep_codes = np.unique(preps, return_inverse=True)[1]
    disease_codes = np.unique(disease, return_inverse=True)[1]

    sub = Substrate(
        name=f"hest_{cohort}",
        A=A, split=split,
        coords=coords_all, coord_groups=patient_codes,
        time_order=None, time_groups=None,
        cond_vars={"tissue": tissue_codes, "prep_method": prep_codes,
                  "disease_state": disease_codes},
        meta={"cohort": cohort, "patient": patients, "tissue": tissues,
             "prep_method": preps, "disease_state": disease,
             "sample_id": sample_id_arr, "barcode": barcode_arr,
             "depth": depth, "hvg_idx": hvg_idx, "n_spots": A.shape[0],
             "spot_diameter_px": spot_diameter_arr,
             "n_patients": len(np.unique(patients)),
             "gene_symbols": gene_symbols, "fit_mask": fit_mask,
             "fit_patients": None if fit_patients is None else sorted(str(p) for p in fit_patients),
             "program_scores": program_scores, "program_names": program_names},
    )
    if keep_logcpm:
        sub.meta["logcpm_full"] = logcpm_full
        sub.meta["all_genes"] = np.asarray(common_genes)
    return sub


# ---------------------------------------------------------- preservation --
def preservation_report(sub, k=6):
    cov = covariance_preservation_check(sub.A, sub.A)
    moran_depth = moran_i(sub.meta["depth"], sub.coords, sub.coord_groups, k=k)
    return {
        "patient_composition": exact_comp(sub.meta["patient"]),
        "tissue_composition": exact_comp(sub.meta["tissue"]),
        "prep_method_composition": exact_comp(sub.meta["prep_method"]),
        "disease_state_composition": exact_comp(sub.meta["disease_state"]),
        "A_self_covariance_check": cov,
        "moran_i_depth": moran_depth,
    }


def exact_comp(labels):
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def _spatial_rank_within_patient(coords, groups):
    """Row -> its rank in a canonical (row-major grid) order WITHIN its
    patient block, used by the toroidal-shift null below."""
    n = len(groups)
    rank = np.empty(n, dtype=np.int64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        pts = coords[idx]
        order = np.lexsort((pts[:, 0], pts[:, 1]))  # sort by y then x
        rank[idx[order]] = np.arange(len(idx))
    return rank


def _random_derangement(n, rng):
    """A uniformly random permutation of range(n) conditioned on having NO
    fixed points (rejection sampling; expected ~e ~= 2.7 draws, trivial at
    the block counts this null uses). `n<=1` has no valid derangement and
    is returned as-is (a single block can only map to itself)."""
    if n <= 1:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def _grid_block_id(coords, groups, nx=4, ny=4):
    """A roughly-equal-count 2D grid partition PER PATIENT: bin by x-rank
    into `nx` strips, then within each strip bin by y-rank into `ny` rows,
    giving nx*ny spatially-compact cells (a balanced 2D quadtree-ish split,
    not raw coordinate-value binning, so cell counts stay close even on an
    irregularly-shaped tissue section)."""
    n = len(groups)
    block_id = np.full(n, -1, dtype=np.int64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        pts = coords[idx]
        x_rank = np.argsort(np.argsort(pts[:, 0]))
        x_bin = np.clip(x_rank * nx // len(idx), 0, nx - 1)
        bid = np.zeros(len(idx), dtype=np.int64)
        for xb in range(nx):
            m = np.where(x_bin == xb)[0]
            if len(m) == 0:
                continue
            y_rank = np.argsort(np.argsort(pts[m, 1]))
            y_bin = np.clip(y_rank * ny // len(m), 0, ny - 1)
            bid[m] = xb * ny + y_bin
        block_id[idx] = bid
    return block_id


# --------------------------------------------------------- structured null
def null_hest(B, sub, seed=0, nx=6, ny=6):
    """2D spatial block swap of B within each patient's spot grid.

    Two things had to be fixed to get here, both measured, not assumed:

    (1) A must be residualised against PATIENT IDENTITY before it is used as
    the injection substrate (see `load()`; `test_hest.py::
    test_patient_residualization_removes_between_patient_variance`).
    Un-residualised, 68% of this cohort's top-variance HVG's variance sits
    BETWEEN patients, not within -- and a within-patient-only spatial null
    correctly leaves between-patient structure alone (patient identity has
    to stay intact), so a "true" object built on unresidualised A leaked
    straight through ANY within-patient null (measured r=0.98 -> r=0.66
    after either a toroidal shift or a naive 1D block permutation) for a
    reason that had nothing to do with the null's own geometry.

    (2) Once that confound is removed, a toroidal shift and a 1D
    raster-order block permutation (reusing the time-series moving-block
    trick along row-major order) BOTH now correctly destroy the planted
    correspondence (r -> ~0) -- but they still fail the OTHER half of the
    requirement, preserving spatial autocorrelation, because a Visium
    spot's 6 nearest neighbours are ~2 same-row and ~4 in the rows above/
    below, and a "contiguous in 1D order" block is compact along one row
    but not row-to-row. Measured fraction of the pre-null Moran's I
    retained after each null (diagnostic near-r=1 planted signal): toroidal
    shift 14-33% (6 vs. 12 patients), 1D raster block 6-7%, this 2D
    grid-block swap 55%, vs. <1% for a naive i.i.d. permutation that
    ignores space entirely. Both simpler variants are kept below (renamed)
    purely as
    regression guards in `tests/test_hest.py`.

    The fix: partition each patient into an actual 2D grid of `nx * ny`
    balanced-count cells (`_grid_block_id`), then swap whole cells' worth of
    B-values between a randomly paired source/destination cell (truncated to
    the smaller of the two cells' spot counts; the handful of leftover spots
    at an unequal cell-size boundary keep their pre-null value -- documented
    here, and bounded by construction to well under `max(nx,ny)` spots per
    patient, since counts are balanced rank-wise). Within a cell, ALL
    directions of neighbour are compact (unlike the 1D attempts), so a
    spot's real 2D neighbours mostly land in the same transplanted cell,
    preserving local Moran's I, while the transplanted cell's real content
    came from an unrelated part of the tissue, destroying alignment with any
    large-scale gradient in the real field.
    """
    rng = np.random.default_rng(seed)
    block_id = _grid_block_id(sub.coords, sub.coord_groups, nx=nx, ny=ny)
    B_null = B.copy()
    G = nx * ny
    for g in np.unique(sub.coord_groups):
        idx_g = np.where(sub.coord_groups == g)[0]
        bids = block_id[idx_g]
        pos_by_block = {b: idx_g[bids == b] for b in range(G)}
        # A DERANGEMENT, not a plain permutation: an unconstrained
        # `rng.permutation(G)` has an expected 1 fixed point regardless of
        # G, i.e. one block in ~every patient would keep its OWN original
        # content -- the same class of bug flagged independently on the
        # DiscoveryBench agent's torus permutation (shifts as small as one
        # cell, mapping points onto their own neighbours). Guaranteeing no
        # block maps to itself closes this off for the null this file
        # actually ships (see `tests/test_hest.py::
        # test_null_hest_block_permutation_has_no_fixed_points`).
        perm_blocks = _random_derangement(G, rng)
        for b in range(G):
            src, dst = pos_by_block[perm_blocks[b]], pos_by_block[b]
            k = min(len(src), len(dst))
            if k == 0:
                continue
            src_pick = rng.permutation(src)[:k]
            dst_pick = rng.permutation(dst)[:k]
            B_null[dst_pick] = B[src_pick]
    return B_null


def _null_hest_toroidal_shift_DO_NOT_USE(B, sub, seed=0, shift_frac=0.5):
    """Kept only so the regression test can demonstrate the failure mode
    documented on `null_hest` above; not exported, not used anywhere else."""
    rng = np.random.default_rng(seed)
    rank = _spatial_rank_within_patient(sub.coords, sub.coord_groups)
    B_null = np.empty_like(B)
    for g in np.unique(sub.coord_groups):
        idx = np.where(sub.coord_groups == g)[0]
        L = len(idx)
        order = idx[np.argsort(rank[idx])]
        shift = int(rng.integers(1, max(2, int(round(L * shift_frac)))))
        B_null[order] = B[np.roll(order, shift)]
    return B_null


def _null_hest_1d_raster_block_DO_NOT_USE(B, sub, seed=0, block_len=25):
    """Kept only so the regression test can demonstrate the failure mode
    documented on `null_hest` above; not exported, not used anywhere else."""
    rng = np.random.default_rng(seed)
    rank = _spatial_rank_within_patient(sub.coords, sub.coord_groups)
    B_null = np.empty_like(B)
    for g in np.unique(sub.coord_groups):
        idx = np.where(sub.coord_groups == g)[0]
        order = idx[np.argsort(rank[idx])]
        perm = block_permutation_index(len(order), block_len, rng)
        B_null[order] = B[order[perm]]
    return B_null


# ------------------------------------------------------- realistic artifact
def depth_confounded_artifact(sub, seed=0, effect_scale=1.0):
    """A B-column driven by per-spot sequencing depth (the classic spatial
    transcriptomics nuisance: apparent "spatial pattern" that is actually a
    depth gradient), with no dependence on any real A column."""
    rng = np.random.default_rng(seed)
    z_depth = np.log1p(sub.meta["depth"])
    z_depth = (z_depth - z_depth.mean()) / max(z_depth.std(), 1e-9)
    y = effect_scale * z_depth + 0.3 * rng.standard_normal(sub.n)
    y = (y - y.mean()) / max(y.std(), 1e-9)
    truth = {"kind": "depth_confounded_artifact", "a": None, "true": False,
            "tag": "depth_confound", "params": {"effect_scale": effect_scale}}
    return y, truth


# ---------------------------------------------------- real morphology arm
EMBEDDING_BACKBONES = ("hoptimus", "dinov2", "gigapath")


def _normalize_barcode(b):
    """Measured, not assumed: the hoptimus embedding job serialised its
    barcodes as the string repr of a 1-element list (`"['AAAC...-1']"`,
    dtype <U22) while dinov2 and gigapath stored the plain barcode string
    (`'AAAC...-1'`, dtype <U18) -- confirmed by comparing all three
    backbones' `INT1.npz` against the source .h5ad's own `obs_names`.
    Un-wraps the former back to the latter; a no-op on already-plain
    barcodes."""
    b = str(b)
    if b.startswith("['") and b.endswith("']"):
        return b[2:-2]
    return b


def load_image_embeddings(sub, backbone, cohort=None, embed_root=None,
                          min_coverage=0.95):
    """Real per-spot morphology embeddings, aligned to `sub`'s own row
    order by (sample_id, barcode) -- NOT assumed to be in the same order
    the .npz files store them in. Returns `(E, covered)`: `covered` is a
    boolean mask, since coverage is measured rather than assumed to be
    100% -- e.g. kidney_injury's QC-passing spots (this loader's own
    `min_depth` filter) are not always a perfect subset of the embedding
    job's patch coverage (measured: 13/370 spots missing for one sample).
    Raises only if a SAMPLE's coverage drops below `min_coverage`, which
    would indicate a real alignment problem rather than a few edge spots.

    Written to consume, never to duplicate: reads the acquired
    `~/morpheus_data/hest/{cohort}/embeddings/{backbone}/{sample_id}.npz`
    files directly and returns a plain (n, dim) float32 array in memory; it
    does not write a derived copy anywhere (disk budget).
    """
    cohort = cohort or sub.meta["cohort"]
    root = embed_root or os.path.join(HEST_ROOT, cohort, "embeddings", backbone)
    E = None
    covered = np.zeros(sub.n, dtype=bool)
    for sid in np.unique(sub.meta["sample_id"]):
        npz_path = os.path.join(root, f"{sid}.npz")
        d = np.load(npz_path, allow_pickle=True)
        emb, bc = d["embeddings"], d["barcodes"]
        bc_to_row = {_normalize_barcode(b): i for i, b in enumerate(bc)}
        if E is None:
            E = np.zeros((sub.n, emb.shape[1]), dtype=np.float32)
        rows = np.where(sub.meta["sample_id"] == sid)[0]
        missing = 0
        for r in rows:
            j = bc_to_row.get(sub.meta["barcode"][r])
            if j is None:
                missing += 1
                continue
            E[r] = emb[j]
            covered[r] = True
        cov_frac = 1.0 - missing / max(len(rows), 1)
        if cov_frac < min_coverage:
            raise ValueError(f"{sid}: only {cov_frac:.1%} barcode coverage in "
                             f"{npz_path} (< {min_coverage:.0%} floor) -- this "
                             f"looks like a real alignment problem, not a "
                             f"few edge spots")
    return E, covered


def measure_morphology_coupling_ceiling(sub, backbone, cohort=None, n_pca_a=20,
                                        n_pca_img=20, k_canon=10, reg=5.0, seed=0):
    """Out-of-sample (patient-block held-out) multi-dimensional canonical
    correlation between the real molecular substrate `sub.A` and real
    per-spot morphology embeddings from `backbone`, summed as nats across
    positive canonical components -- the SAME method used for the climate
    substrate's coupling-ceiling measurement (year-block held out there;
    patient-block held out here), so the two numbers are comparable.

    Both sides are residualised against patient identity first (the same
    fix `load()` already applies to the molecular side, applied here to the
    morphology side too) -- otherwise a between-patient batch effect in the
    image embeddings would inflate this the same way it inflated the raw
    (un-residualised) molecular correlation before that fix was found.
    """
    from sklearn.decomposition import PCA

    E, covered = load_image_embeddings(sub, backbone, cohort=cohort)
    coverage_frac = float(covered.mean())
    A_full = sub.A
    if not covered.all():
        # Restrict to spots with a real embedding rather than silently
        # treating an uncovered spot's all-zero row as data.
        A_full, E = A_full[covered], E[covered]
        split_c, patient_c = sub.split[covered], sub.meta["patient"][covered]
    else:
        split_c, patient_c = sub.split, sub.meta["patient"]

    patient_codes = np.unique(patient_c, return_inverse=True)[1]
    E_resid = _residualize_against_onehot(E.astype(np.float64), patient_codes)

    disc, held = split_c == 0, split_c == 1
    pca_a = PCA(n_components=n_pca_a, random_state=seed).fit(A_full[disc])
    pca_e = PCA(n_components=n_pca_img, random_state=seed).fit(E_resid[disc])
    A_r, E_r = pca_a.transform(A_full), pca_e.transform(E_resid)

    def fit_cca_multi(X, Y, reg, k):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        n = X.shape[0]
        Cxx = Xc.T @ Xc / n + reg * np.eye(X.shape[1])
        Cyy = Yc.T @ Yc / n + reg * np.eye(Y.shape[1])
        Cxy = Xc.T @ Yc / n
        ex, Ux = np.linalg.eigh(Cxx)
        ey, Uy = np.linalg.eigh(Cyy)
        Wx = Ux @ np.diag(1 / np.sqrt(np.maximum(ex, 1e-8))) @ Ux.T
        Wy = Uy @ np.diag(1 / np.sqrt(np.maximum(ey, 1e-8))) @ Uy.T
        U, s, Vt = np.linalg.svd(Wx @ Cxy @ Wy)
        kk = min(k, U.shape[1], Vt.shape[0])
        return Wx @ U[:, :kk], Wy @ Vt[:kk, :].T, X.mean(0), Y.mean(0)

    Wx, Wy, mx, my = fit_cca_multi(A_r[disc], E_r[disc], reg=reg, k=k_canon)
    Zx, Zy = (A_r[held] - mx) @ Wx, (E_r[held] - my) @ Wy
    rs = [float(np.corrcoef(Zx[:, i], Zy[:, i])[0, 1]) for i in range(Wx.shape[1])]

    from ..calibration import r_to_nats
    nats_each = [r_to_nats(r) for r in rs if r > 0]
    return {
        "cohort": cohort or sub.meta["cohort"], "backbone": backbone,
        "coverage_frac": coverage_frac,
        "n_patients_discovery": int(len(np.unique(patient_c[disc]))),
        "n_patients_held_out": int(len(np.unique(patient_c[held]))),
        "oos_canonical_correlations": rs,
        "nats_sum_positive_dims": float(sum(nats_each)),
    }


# ------------------------------------------------------- Gavish metaprograms
_GAVISH_PATH_CANDIDATES = (
    os.path.join(_HOME, "morpheus_work", "morpheus-rebase", "v2", "data",
                "metaprograms", "gavish_2023_malignant_mps.json"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
        "morpheus-rebase", "v2", "data", "metaprograms",
        "gavish_2023_malignant_mps.json"),
)


def load_gavish_programs(path=None):
    """The 41 Gavish et al. 2023 malignant metaprograms as
    `[{"mp_id", "name", "n_genes", "genes": [...]}, ...]`.

    Tries the box's conventional path first, then the equivalent path
    relative to this repository's root, so the same call works locally and
    on the box without a hard-coded absolute path in either place.
    """
    if path is not None:
        with open(path) as fh:
            return json.load(fh)["programs"]
    for candidate in _GAVISH_PATH_CANDIDATES:
        if os.path.exists(candidate):
            with open(candidate) as fh:
                return json.load(fh)["programs"]
    raise FileNotFoundError(
        "gavish_2023_malignant_mps.json not found at any of: " + ", ".join(_GAVISH_PATH_CANDIDATES))


# --------------------------------------------------------- real morphology --
# CellViT/PanNuke's 5-class taxonomy, as measured across ALL 24 ccRCC
# accessions' `*_cellvit_seg.parquet` files (A0 recon) -- not assumed from
# the model's published label set, since a cohort-specific run can drop a
# class entirely.
CELLVIT_CLASSES = ("Neoplastic", "Connective", "Epithelial", "Inflammatory", "Dead")


def load_cellvit_summary(sub, cohort=None, data_dir=None, classes=CELLVIT_CLASSES,
                         cache_root="default"):
    """Real per-spot nucleus summaries from the CellViT segmentation parquet.

    A nucleus is assigned to a spot if its polygon centroid falls within
    that spot's OWN radius (`scalefactors/spot_diameter_fullres` / 2, read
    per sample rather than assumed constant across the cohort); most nuclei
    fall under no spot at all, since Visium spots are a sparse array of
    discs, not a tiling of the tissue -- this is measured per spot via a
    k-d tree over that sample's nucleus centroids, not assumed.

    Returns `(features, covered)`: `features` is `(n, 2 + len(classes))`
    with columns `[cv_count, cv_area, cv_frac_<class> ...]` (area is the
    mean nuclear polygon area, in the WSI's own pixel^2 units, of nuclei
    assigned to that spot); `covered` is False for a spot with zero
    assigned nuclei, following the same measured-not-assumed coverage
    contract as `load_image_embeddings`.

    Computed per sample and, like `load_cellvit_multiscale_features` (same
    `cache_root` rule, same MANIFEST/sha256 contract), cached per sample under
    `<cache_root>/<cohort>/summary/`; the params hash covers the exact spot
    rows, the spot radius, classes, the parquet identity and the code version.
    """
    import h5py

    cohort = cohort or sub.meta["cohort"]
    cfg = COHORTS[cohort]
    d = data_dir or cfg["dir"]
    if cache_root == "default":
        cache_root = _default_multiscale_cache_root(cohort, d)
    cache_dir = None
    if cache_root is not None:
        cache_dir = os.path.join(cache_root, cohort, "summary")
        os.makedirs(cache_dir, exist_ok=True)
    manifest = _read_multiscale_manifest(cache_dir) if cache_dir else {}
    n_classes = len(classes)
    features = np.zeros((sub.n, 2 + n_classes), dtype=np.float64)
    covered = np.zeros(sub.n, dtype=bool)
    for sid in np.unique(sub.meta["sample_id"]):
        rows = np.where(sub.meta["sample_id"] == sid)[0]
        parquet_path = os.path.join(d, f"{sid}_cellvit_seg.parquet")
        with h5py.File(os.path.join(d, f"{sid}_st.h5ad"), "r") as f:
            radius = float(f["uns/spatial/ST/scalefactors/spot_diameter_fullres"][()]) / 2.0
        spot_xy = np.asarray(sub.coords[rows], dtype=np.float64)
        got = None
        if cache_dir:
            path, params, _ = _multiscale_cache_entry(cache_dir, sid, parquet_path, spot_xy,
                                                      np.full(len(rows), radius), classes, 0,
                                                      version=CELLVIT_SUMMARY_CACHE_VERSION)
            name = os.path.basename(path)
            record = manifest.get(name)
            if record and os.path.exists(path) and _sha256_file(path) == record.get("sha256"):
                with np.load(path) as z:
                    got = (z["features"], z["covered"])
        if got is None:
            got = _cellvit_summary_sample(parquet_path, spot_xy, radius, classes)
            if cache_dir:
                tmp = f"{path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, features=got[0], covered=got[1])
                os.replace(tmp, path)
                record = {"params": params, "sha256": _sha256_file(path)}
                _update_multiscale_manifest(cache_dir, name, record, version=CELLVIT_SUMMARY_CACHE_VERSION)
                manifest[name] = record
        features[rows], covered[rows] = got
    return features, covered


CELLVIT_SUMMARY_CACHE_VERSION = "cellvit_summary_v1_20261011"


def _cellvit_summary_sample(parquet_path, spot_xy, radius, classes):
    """One sample's `[cv_count, cv_area, cv_frac_<class> ...]` and coverage."""
    import pandas as pd
    from scipy.spatial import cKDTree
    from shapely import wkb as _wkb

    n = len(spot_xy)
    features = np.zeros((n, 2 + len(classes)), dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    df = pd.read_parquet(parquet_path, columns=["geometry", "class"])
    if len(df) == 0:
        return features, covered
    geoms = [_wkb.loads(bytes(g)) for g in df["geometry"].to_numpy()]
    cx = np.array([g.centroid.x for g in geoms])
    cy = np.array([g.centroid.y for g in geoms])
    area = np.array([g.area for g in geoms])
    cls = df["class"].to_numpy()
    # `sub.coords` is `[pxl_col, pxl_row]` (measured, A0 recon), the same
    # full-res pixel space the CellViT geometries were segmented in.
    tree = cKDTree(np.column_stack([cx, cy]))
    for i in range(n):
        nb = tree.query_ball_point(spot_xy[i], r=radius)
        if not nb:
            continue
        covered[i] = True
        features[i, 0] = len(nb)
        features[i, 1] = float(area[nb].mean())
        nb_cls = cls[nb]
        for ci, c in enumerate(classes):
            features[i, 2 + ci] = float((nb_cls == c).mean())
    return features, covered


CELLVIT_NUCLEAR_CACHE_VERSION = "cellvit_nuclear_morphometry_v1_20261013"
# Footprint-exact morphology targets: nuclei whose centroid lies inside the
# spot circle (the transcript footprint), per class and over all classes.
NUC_GROUPS = {"neoplastic": ("Neoplastic",), "connective": ("Connective",),
              "inflammatory": ("Inflammatory",), "all": None}
NUC_STATS = ("area", "ecc", "density")
NUC_MIN_NUCLEI = 5


def _polygon_moments(geoms):
    """Exact area, centroid-centred second moments and eccentricity of every
    polygon's exterior ring (Green's theorem; vectorised over shapely 2
    arrays). Eccentricity of the equivalent-moment ellipse:
    sqrt(1 - lambda_min / lambda_max) -- 0 for a disc or square, -> 1 for a line."""
    import shapely

    rings = shapely.get_exterior_ring(geoms)
    xy, idx = shapely.get_coordinates(rings, return_index=True)
    n = len(geoms)
    first = np.zeros(n, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    first[idx[starts]] = starts
    xy = xy - xy[first[idx]]  # shift each polygon to its first vertex (precision)
    same = idx[:-1] == idx[1:]
    x0, y0, x1, y1 = xy[:-1, 0][same], xy[:-1, 1][same], xy[1:, 0][same], xy[1:, 1][same]
    g = idx[:-1][same]
    cross = x0 * y1 - x1 * y0
    A = np.bincount(g, cross, n) / 2.0
    cx = np.bincount(g, (x0 + x1) * cross, n) / 6.0
    cy = np.bincount(g, (y0 + y1) * cross, n) / 6.0
    ixx = np.bincount(g, (x0 * x0 + x0 * x1 + x1 * x1) * cross, n) / 12.0
    iyy = np.bincount(g, (y0 * y0 + y0 * y1 + y1 * y1) * cross, n) / 12.0
    ixy = np.bincount(g, (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * cross, n) / 24.0
    safe = np.where(np.abs(A) > 0, A, 1.0)
    mx, my = cx / safe, cy / safe
    m20 = ixx / safe - mx ** 2
    m02 = iyy / safe - my ** 2
    m11 = ixy / safe - mx * my
    half = np.sqrt(((m20 - m02) / 2.0) ** 2 + m11 ** 2)
    lmax, lmin = (m20 + m02) / 2.0 + half, (m20 + m02) / 2.0 - half
    ecc = np.sqrt(np.clip(1.0 - np.clip(lmin, 0, None) / np.where(lmax > 0, lmax, 1.0), 0.0, 1.0))
    return np.abs(A), ecc


def _group_median(keys, values, n_keys):
    out = np.full(n_keys, np.nan)
    if not len(keys):
        return out
    order = np.lexsort((values, keys))
    k, v = keys[order], values[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(k)) + 1, len(k)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = b - a
        out[k[a]] = v[a + m // 2] if m % 2 else 0.5 * (v[a + m // 2 - 1] + v[a + m // 2])
    return out


def _nuclear_morphometry_sample(parquet_path, spot_xy, radius, min_nuclei=NUC_MIN_NUCLEI):
    """One sample: per spot and nucleus group, the count, median nuclear area,
    median eccentricity and density (nuclei per 10^4 px^2 of spot circle) of
    nuclei whose polygon centroid lies within the spot radius. Columns are
    `[<group>_count, <group>_area, <group>_ecc, <group>_density]` for each
    group in `NUC_GROUPS` order; area/ecc are NaN below `min_nuclei`."""
    import pandas as pd
    import shapely
    from scipy.spatial import cKDTree

    n = len(spot_xy)
    out = np.full((n, 4 * len(NUC_GROUPS)), np.nan)
    frame = pd.read_parquet(parquet_path, columns=["geometry", "class"])
    geoms = shapely.from_wkb(frame["geometry"].to_numpy()) if len(frame) else np.array([])
    cls = frame["class"].astype(str).to_numpy() if len(frame) else np.array([], dtype=str)
    if len(geoms):
        area, ecc = _polygon_moments(geoms)
        cent = shapely.get_coordinates(shapely.centroid(geoms))
        dist, spot = cKDTree(np.asarray(spot_xy, dtype=float)).query(cent, k=1)
        inside = dist <= radius
    else:
        area = ecc = np.array([]); spot = np.array([], dtype=np.int64); inside = np.array([], dtype=bool)
    spot_area = np.pi * radius ** 2
    for gi, (group, members) in enumerate(NUC_GROUPS.items()):
        m = inside & (np.ones(len(cls), dtype=bool) if members is None else np.isin(cls, members))
        keys = np.asarray(spot[m], dtype=np.int64)
        count = np.bincount(keys, minlength=n).astype(float)
        med_area = _group_median(keys, area[m], n)
        med_ecc = _group_median(keys, ecc[m], n)
        low = count < min_nuclei
        med_area[low] = np.nan
        med_ecc[low] = np.nan
        out[:, 4 * gi:4 * gi + 4] = np.column_stack([count, med_area, med_ecc, count / spot_area * 1e4])
    return out


def load_cellvit_nuclear_morphometry(sub, cohort=None, data_dir=None, cache_root="default",
                                     min_nuclei=NUC_MIN_NUCLEI):
    """Per-spot nuclear morphometry of the nuclei inside each spot circle
    (`_nuclear_morphometry_sample`), cached per sample under
    `<cache_root>/<cohort>/nuclear/` with the same rule and MANIFEST/sha256
    contract as `load_cellvit_summary`. Returns `{group: {"count", "area",
    "ecc", "density"}}` over `NUC_GROUPS`."""
    import h5py

    cohort = cohort or sub.meta["cohort"]
    d = data_dir or COHORTS[cohort]["dir"]
    if cache_root == "default":
        cache_root = _default_multiscale_cache_root(cohort, d)
    cache_dir = None
    if cache_root is not None:
        cache_dir = os.path.join(cache_root, cohort, "nuclear")
        os.makedirs(cache_dir, exist_ok=True)
    manifest = _read_multiscale_manifest(cache_dir) if cache_dir else {}
    full = np.full((sub.n, 4 * len(NUC_GROUPS)), np.nan)
    for sid in np.unique(sub.meta["sample_id"]):
        rows = np.where(sub.meta["sample_id"] == sid)[0]
        parquet_path = os.path.join(d, f"{sid}_cellvit_seg.parquet")
        with h5py.File(os.path.join(d, f"{sid}_st.h5ad"), "r") as f:
            radius = float(f["uns/spatial/ST/scalefactors/spot_diameter_fullres"][()]) / 2.0
        spot_xy = np.asarray(sub.coords[rows], dtype=np.float64)
        got = None
        if cache_dir:
            path, params, _ = _multiscale_cache_entry(cache_dir, sid, parquet_path, spot_xy,
                                                      np.full(len(rows), radius), list(NUC_GROUPS),
                                                      int(min_nuclei), version=CELLVIT_NUCLEAR_CACHE_VERSION)
            name = os.path.basename(path)
            record = manifest.get(name)
            if record and os.path.exists(path) and _sha256_file(path) == record.get("sha256"):
                with np.load(path) as z:
                    got = z["morphometry"]
        if got is None:
            got = _nuclear_morphometry_sample(parquet_path, spot_xy, radius, min_nuclei=min_nuclei)
            if cache_dir:
                tmp = f"{path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, morphometry=got)
                os.replace(tmp, path)
                record = {"params": params, "sha256": _sha256_file(path)}
                _update_multiscale_manifest(cache_dir, name, record, version=CELLVIT_NUCLEAR_CACHE_VERSION)
                manifest[name] = record
        full[rows] = got
    return {group: {stat: full[:, 4 * gi + si] for si, stat in enumerate(("count", "area", "ecc", "density"))}
            for gi, group in enumerate(NUC_GROUPS)}


CELLVIT_MULTISCALE_CACHE_VERSION = "cellvit_multiscale_v1_20261011"
_STORE_ROOT = "/lambda/nfs/geeg/biorag3_persistent_20260711"
CELLVIT_MULTISCALE_STORE_CACHE = os.path.join(_STORE_ROOT, "external", "cellvit_multiscale_cache_20261011")
_MULTISCALE_KEYS = ("nuclear", "composition", "graph", "boundary", "covered")


def _default_multiscale_cache_root(cohort, data_dir):
    """The store cache, only for real cohort data (the cohort's default mirror
    or a tree under the store) and only where the store cache directory
    already exists -- never on a laptop, never for a test tree."""
    if not os.path.isdir(CELLVIT_MULTISCALE_STORE_CACHE):
        return None
    real = os.path.realpath(data_dir)
    if real != os.path.realpath(COHORTS[cohort]["dir"]) and not real.startswith(
            os.path.realpath(_STORE_ROOT) + os.sep):
        return None
    return CELLVIT_MULTISCALE_STORE_CACHE


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _multiscale_sample_features(parquet_path, anchor_xy, anchor_radius, sid, classes, k_neighbours):
    import pandas as pd
    from shapely import wkb as _wkb
    from morpheus_encoder.tissue_feature_bank import derive_cell_graph_features

    n = len(anchor_xy)
    frame = pd.read_parquet(parquet_path, columns=["geometry", "class"])
    if not len(frame):
        return {"nuclear": np.zeros((n, 3)), "composition": np.zeros((n, len(classes) + 1)),
                "graph": np.zeros((n, 4)), "boundary": np.zeros((n, 3)),
                "covered": np.zeros(n, dtype=bool)}
    geoms = [_wkb.loads(bytes(g)) for g in frame["geometry"].to_numpy()]
    cell_xy = np.asarray([(g.centroid.x, g.centroid.y) for g in geoms])
    return derive_cell_graph_features(
        cell_xy=cell_xy, cell_class=frame["class"].astype(str).tolist(),
        cell_area=np.asarray([g.area for g in geoms]), cell_groups=np.full(len(frame), str(sid)),
        anchor_xy=anchor_xy, anchor_groups=np.full(n, str(sid)), anchor_radius=anchor_radius,
        classes=classes, k_neighbours=k_neighbours)


def _multiscale_cache_entry(cache_dir, sid, parquet_path, anchor_xy, anchor_radius, classes,
                            k_neighbours, version=None):
    """`(path, params, digest)` for one sample. The params hash covers every
    input the features depend on: the exact anchor rows (coordinates and
    radii), classes, k, the parquet identity (name and size) and the code
    version, so a different row filter or parameter can never hit a stale file."""
    import hashlib
    anchors = hashlib.sha256(np.ascontiguousarray(anchor_xy, dtype=np.float64).tobytes()
                             + np.ascontiguousarray(anchor_radius, dtype=np.float64).tobytes()).hexdigest()
    params = {"sample_id": str(sid), "classes": list(map(str, classes)), "k_neighbours": int(k_neighbours),
              "n_anchors": int(len(anchor_xy)), "anchors_sha256": anchors,
              "parquet": os.path.basename(parquet_path), "parquet_bytes": int(os.path.getsize(parquet_path)),
              "code_version": version or CELLVIT_MULTISCALE_CACHE_VERSION}
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{sid}__{digest}.npz"), params, digest


def _update_multiscale_manifest(cache_dir, name, record, version=None):
    """Merge one file's record into MANIFEST.json under an exclusive lock
    (parallel folds share the cache) and replace the manifest atomically."""
    path = os.path.join(cache_dir, "MANIFEST.json")
    lock = open(os.path.join(cache_dir, ".manifest.lock"), "a")
    try:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX
            pass
        manifest = {"code_version": version or CELLVIT_MULTISCALE_CACHE_VERSION, "files": {}}
        if os.path.exists(path):
            with open(path) as fh:
                manifest = json.load(fh)
        manifest.setdefault("files", {})[name] = record
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:
        lock.close()


def _read_multiscale_manifest(cache_dir):
    path = os.path.join(cache_dir, "MANIFEST.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh).get("files", {})


def load_cellvit_multiscale_features(sub, cohort=None, data_dir=None, classes=CELLVIT_CLASSES,
                                     k_neighbours=6, cache_root="default"):
    """Measured cell-level tissue descriptors projected to spatial anchors.

    Unlike :func:`load_cellvit_summary`, this preserves cell-type adjacency and
    tumour-boundary information before aggregating to a Visium spot.  The
    result contains only measurements and explicit coverage; it does not fit a
    state dictionary or inspect a molecular response.

    Every descriptor is sample-local (cells and anchors are grouped by sample),
    so it is computed one sample at a time and, when `cache_root` is set,
    cached per sample at `<cache_root>/<cohort>/<sample_id>__<params-hash>.npz`
    with a `MANIFEST.json` (params, code version, sha256 per file). A cached
    file is used only when its sha256 matches the manifest; otherwise it is
    recomputed and rewritten. `cache_root="default"` uses the store cache when
    that directory exists and `data_dir` is the cohort's own mirror; `None`
    disables caching.
    """
    cohort = cohort or sub.meta["cohort"]
    cfg = COHORTS[cohort]
    d = data_dir or cfg["dir"]
    if cache_root == "default":
        cache_root = _default_multiscale_cache_root(cohort, d)
    cache_dir = None
    if cache_root is not None:
        cache_dir = os.path.join(cache_root, cohort)
        os.makedirs(cache_dir, exist_ok=True)
    manifest = _read_multiscale_manifest(cache_dir) if cache_dir else {}
    n = sub.n
    out = {"nuclear": np.zeros((n, 3)), "composition": np.zeros((n, len(classes) + 1)),
           "graph": np.zeros((n, 4)), "boundary": np.zeros((n, 3)), "covered": np.zeros(n, dtype=bool)}
    sample_ids = np.asarray(sub.meta["sample_id"])
    coords = np.asarray(sub.coords, dtype=float)
    radius_all = np.asarray(sub.meta["spot_diameter_px"], dtype=float) / 2.0
    for sid in np.unique(sample_ids):
        rows = np.where(sample_ids == sid)[0]
        parquet_path = os.path.join(d, f"{sid}_cellvit_seg.parquet")
        anchor_xy, anchor_radius = coords[rows], radius_all[rows]
        feats = None
        if cache_dir:
            path, params, _ = _multiscale_cache_entry(cache_dir, sid, parquet_path, anchor_xy,
                                                      anchor_radius, classes, k_neighbours)
            name = os.path.basename(path)
            record = manifest.get(name)
            if record and os.path.exists(path) and _sha256_file(path) == record.get("sha256"):
                with np.load(path) as z:
                    feats = {k: z[k] for k in _MULTISCALE_KEYS}
        if feats is None:
            feats = _multiscale_sample_features(parquet_path, anchor_xy, anchor_radius, sid,
                                                classes, k_neighbours)
            if cache_dir:
                tmp = f"{path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, **{k: np.asarray(feats[k]) for k in _MULTISCALE_KEYS})
                os.replace(tmp, path)
                record = {"params": params, "sha256": _sha256_file(path)}
                _update_multiscale_manifest(cache_dir, name, record)
                manifest[name] = record
        for k in _MULTISCALE_KEYS:
            out[k][rows] = feats[k]
    return out


# ---------------------------------------------------------- real episode --
def _patient_of_sample(prefix, sid, data_dir):
    with open(os.path.join(data_dir, f"{prefix}{sid}_metadata.json")) as fh:
        return str(json.load(fh)["patient"])


def real_episode(fold, *, n_folds=5, seed=0, target, embed_root=None,
                 sample_ids=None, cohort="ccrcc", data_dir=None, n_hvg=300,
                 min_depth=200, program_sets=None, k_neighbors=6,
                 include_tissue_features=False, bank=False, bank_views=None,
                 bank_cache_root=None, bank_options=None, centre_baseline=True,
                 within_patient_scale=None, substrate_version="v04"):
    """One leak-proof, patient-grouped discovery episode for a single
    predeclared target, over the real ccRCC HEST substrate.

    D_P = the `n_folds - 1` FIT folds (patient-grouped, deterministic
    shuffled `np.array_split`); D_R = the held-out `fold`. Every response
    `y` is within-patient centred (a per-patient fixed-effect residual, the
    same leak-free operation `load()` already applies to the molecular
    substrate). Nulls are NOT a separate row role: `episode_null_generators`
    transforms the D_R rows in place, so `partitions.null` is empty and
    `metadata["null_source"] = "replication_structured"` (see
    `discovery_episode.EpisodePartitions`).

    `bank=True` attaches the protected measurement bank
    (`morpheus_encoder.measurement_bank`) as `metadata["bank"]` plus
    `metadata["bank_report"]` (unavailable/excluded views and why). With
    `bank=True` the rows are restricted (before partitions are remapped) to
    spots where the H-Optimus-0 embedding, the CellViT nuclei and the
    multiscale CellViT descriptors all exist, so no bank view has missing
    rows; `metadata["coverage_fraction"]`/`["coverage_rule"]` record this and
    nothing is imputed.

    Targets: `morph_pc{k}`, `cv_count`, `cv_area`, `cv_frac_<class>`,
    `nbr_mp{k}` (neighbour mean of the k-th highest within-patient Moran's I
    programme, ranked on D_P) and `own_mp{k}` (that same programme's own-spot
    score; its competitor drops the target programme and adds no neighbour
    means).

    Every competitor-baseline column is within-patient centred with the same
    per-patient fixed-effect residual as the response, over the kept rows
    (covariate values only; the response is never used): an uncentred
    competitor carries between-patient offsets that cannot transfer to
    held-out patients and made the 2026-09-26 competitor score R^2 ~ 0 there.
    `centre_baseline=False` rebuilds the pre-2026-10-11 episode (response
    centred over all loaded rows, competitor uncentred) solely to reproduce
    sealed v0.3 receipts.

    `within_patient_scale` (default: True exactly when `bank=True`, i.e. every
    v0.4 episode) goes further and STANDARDISES the response and every
    competitor column within patient (minus the patient mean, divided by that
    column's own within-patient sd, floored). A centred-only competitor still
    failed to transfer for `morph_pc1` because the response's within-patient
    spread differs ~3.4x across held-out patients. For `nbr_mp*` targets with
    the bank, the competitor also carries the bank's own-spot PCA view and its
    neighbour-mean PCA columns (protocol v0.4 `when_target_is_neighbourhood_
    programme`), standardised the same way.

    `substrate_version="v05"` (requires `bank=True`; default "v04" rebuilds the
    registered v0.4 episode exactly) adds, per protocol v0.5:
    * competitor = Gavish + Hallmark own-spot scores for every target; for
      morphology targets (`morph_pc*`, `cv_*`, `nuc_*`) also the neighbour
      means at k = 6 (`nbr:`) and k = 18 (`nbr18:`) of the 91 programme scores
      and the 50 aligned PCA columns; every competitor column whose pooled
      within-patient sd on proposal rows is < 0.01 is dropped first
      (`metadata["baseline_dropped_near_constant"]`);
    * bank: fold-0-aligned PCA/NMF identities, near-constant columns dropped,
      and a leak guard (own_mp/nbr_mp: |within-patient r| >= 0.9 with the target
      programme's own k = 6 neighbour mean, every view; nuc_*: with the
      response, cellvit_structure view), recorded in `bank_report`;
    * `metadata["target_degenerate_warning"]` (> 90 % zeros or a pooled
      within-patient sd < 1e-3 before standardising).
    Targets `nuc_<group>_<stat>` (group in neoplastic/connective/inflammatory/
    all, stat in area/ecc/density) are the footprint-exact nuclear morphometry
    of `load_cellvit_nuclear_morphometry`; a spot with fewer than 5 nuclei of
    the group is not covered for that target.
    """
    from morpheus_encoder.discovery_episode import DiscoveryEpisode, EpisodePartitions

    if substrate_version not in ("v04", "v05"):
        raise ValueError(f"unknown substrate_version {substrate_version!r}")
    v05 = substrate_version == "v05"
    if v05 and not bank:
        raise ValueError("substrate_version='v05' builds its competitor from the bank; pass bank=True")

    if not (0 <= fold < n_folds):
        raise ValueError("fold must be in [0, n_folds)")
    cfg = COHORTS[cohort]
    d = data_dir or cfg["dir"]
    prefix = cfg["prefix"]
    if sample_ids is None:
        sample_ids = measure_available_samples(cohort=cohort, data_dir=d)
    sample_patient = {sid: _patient_of_sample(prefix, sid, d) for sid in sample_ids}
    all_patients = np.array(sorted(set(sample_patient.values())))
    if len(all_patients) < n_folds:
        raise ValueError("fewer patients than folds")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_patients)
    folds = np.array_split(shuffled, n_folds)
    held_patients = set(folds[fold].tolist())
    fit_patients = sorted(set(all_patients.tolist()) - held_patients)

    if program_sets is None:
        program_sets = load_gavish_programs()

    sub = load(cohort=cohort, data_dir=d, sample_ids=sample_ids, n_hvg=n_hvg, seed=seed,
              min_depth=min_depth, fit_patients=fit_patients, program_sets=program_sets,
              keep_logcpm=bank)

    patients = sub.meta["patient"]
    is_fit = np.isin(patients, np.asarray(fit_patients))
    proposal_rows_g, replication_rows_g = np.where(is_fit)[0], np.where(~is_fit)[0]

    program_scores = sub.meta["program_scores"]
    program_names = sub.meta["program_names"]
    depth_log = np.log1p(sub.meta["depth"])
    patient_codes = sub.coord_groups
    neighbours_g = knn_neighbors(sub.coords, k=k_neighbors, groups=patient_codes)

    cv_features, cv_covered = load_cellvit_summary(sub, cohort=cohort, data_dir=d)
    cv_count, cv_area, cv_frac = cv_features[:, 0], cv_features[:, 1], cv_features[:, 2:]

    def _neighbour_mean(values, neighbours):
        out = np.zeros_like(values, dtype=np.float64)
        for i, idx in enumerate(neighbours):
            out[i] = values[idx].mean() if len(idx) else values[i]
        return out

    is_nbr_target = target.startswith("nbr_mp")
    is_own_target = target.startswith("own_mp")
    nuc_group = nuc_stat = None
    if target.startswith("nuc_"):
        parts = target.split("_")
        if len(parts) != 3 or parts[1] not in NUC_GROUPS or parts[2] not in NUC_STATS:
            raise ValueError(f"unknown nuclear target {target!r}")
        nuc_group, nuc_stat = parts[1], parts[2]
    is_morph_family = target.startswith(("morph_pc", "cv_", "nuc_"))
    # The competitor baseline must never contain the response or a
    # deterministic function of it (protocol v0.3 exclusion clause): for a
    # CellViT-fraction target every class fraction is dropped (they sum to
    # one, so the other four determine the target); for a neighbour-program
    # target the target program's own neighbour mean is dropped below; for an
    # own-spot program target the target program itself is dropped (and no
    # neighbour-mean block is added).
    baseline_cols = [depth_log.reshape(-1, 1)]
    baseline_names = ["depth_log1p"]
    if not target.startswith("cv_frac_"):
        # a class density is count x that class's fraction / spot area: its
        # own class fraction leaves the competitor (other classes stay)
        frac_keep = [ci for ci, c in enumerate(CELLVIT_CLASSES)
                     if not (nuc_stat == "density" and NUC_GROUPS.get(nuc_group) == (c,))]
        baseline_cols.append(cv_frac[:, frac_keep])
        baseline_names += [f"cv_frac_{CELLVIT_CLASSES[ci]}" for ci in frac_keep]
    nbr_program_scores = None
    if is_nbr_target:
        nbr_program_scores = np.column_stack(
            [_neighbour_mean(program_scores[:, j], neighbours_g) for j in range(program_scores.shape[1])])

    covered = np.ones(sub.n, dtype=bool)
    tissue = None
    image = None
    image_covered = None
    nbr_mp_info = None
    own_mp_info = None
    own_program_drop = None
    extra_cols, extra_names = [], []
    if target.startswith("morph_pc"):
        pc_index = int(target[len("morph_pc"):]) - 1
        if pc_index < 0:
            raise ValueError("morph_pc targets are 1-indexed")
        from sklearn.decomposition import PCA
        E, covered = load_image_embeddings(sub, "hoptimus", cohort=cohort, embed_root=embed_root)
        image, image_covered = E, covered.copy()
        fit_rows = np.where(is_fit & covered)[0]
        # The target is a small, explicitly declared set of morphology PCs.
        # A full decomposition of every H-Optimus coordinate is unnecessary
        # and made each patient-held-out fold spend most of its time solving
        # directions that cannot enter the target.  This is a deterministic
        # truncated PCA fit on D_P only; it changes neither rows nor the
        # target definition, only makes the repeated held-out study practical.
        pca = PCA(n_components=max(pc_index + 1, 5), random_state=seed,
                  svd_solver="randomized", iterated_power=5).fit(E[fit_rows])
        pcs = pca.transform(E)
        y_g = _residualize_against_onehot(pcs, patient_codes)[:, pc_index]
    elif target == "cv_count":
        y_g, covered = np.log1p(cv_count), cv_covered.copy()
    elif target == "cv_area":
        y_g, covered = cv_area.copy(), cv_covered.copy()
    elif target.startswith("cv_frac_"):
        cls = target[len("cv_frac_"):]
        if cls not in CELLVIT_CLASSES:
            raise ValueError(f"unknown CellViT class {cls!r}")
        y_g, covered = cv_frac[:, CELLVIT_CLASSES.index(cls)].copy(), cv_covered.copy()
    elif nuc_group is not None:
        morpho = load_cellvit_nuclear_morphometry(sub, cohort=cohort, data_dir=d)[nuc_group]
        y_g = np.asarray(morpho[nuc_stat], dtype=np.float64).copy()
        covered = (morpho["count"] >= NUC_MIN_NUCLEI) & np.asarray(cv_covered, dtype=bool)
        y_g[~covered] = 0.0  # never used: uncovered rows are dropped below
    elif is_nbr_target or is_own_target:
        kind = "nbr_mp" if is_nbr_target else "own_mp"
        program_index, program_info = _moran_ranked_program(
            int(target[len(kind):]), kind, program_scores, program_names, sub.coords,
            patient_codes, proposal_rows_g, k_neighbors)
        keep = [j for j in range(program_scores.shape[1]) if j != program_index]
        if is_nbr_target:
            y_g = _neighbour_mean(program_scores[:, program_index], neighbours_g)
            extra_cols.append(nbr_program_scores[:, keep])
            extra_names += [f"nbr_mp:{program_names[j] if program_names else j}" for j in keep]
            nbr_mp_info = program_info
        else:
            y_g = program_scores[:, program_index].astype(np.float64).copy()
            own_program_drop = program_index
            own_mp_info = program_info
    else:
        raise ValueError(f"unknown target {target!r}")

    prog_keep = [j for j in range(program_scores.shape[1]) if j != own_program_drop]
    baseline_cols.append(program_scores[:, prog_keep])
    baseline_names += [f"mp:{program_names[j] if program_names else j}" for j in prog_keep]
    baseline_g = np.column_stack(baseline_cols + extra_cols).astype(np.float64)
    baseline_names += extra_names
    assert baseline_g.shape[1] == len(baseline_names)

    coverage_rule = "response defined"
    if include_tissue_features or bank:
        # Coverage is an observed-data condition, not a value to impute.  The
        # tissue dictionary (and every bank view) may be built only where the
        # molecular response, the aligned H-Optimus-0 image embedding, the
        # CellViT nuclei within the spot radius and the multiscale CellViT
        # descriptors all exist for a spot. Intersecting this mask *before* the
        # patient partitions are remapped prevents a tissue-side missingness
        # pattern from leaking into a later candidate or from creating
        # zero-valued pseudo-tissue.
        tissue = load_cellvit_multiscale_features(sub, cohort=cohort, data_dir=d)
        if image is None:
            image, image_covered = load_image_embeddings(
                sub, "hoptimus", cohort=cohort, embed_root=embed_root)
        tissue_coverage = np.asarray(tissue["covered"], dtype=bool) & np.asarray(image_covered, dtype=bool)
        if bank:
            tissue_coverage &= np.asarray(cv_covered, dtype=bool)
        covered &= tissue_coverage
        coverage_rule = ("response defined & H-Optimus-0 embedding & CellViT multiscale anchor covered"
                         + (" & CellViT nuclei within spot radius" if bank else "")
                         + (f" & >= {NUC_MIN_NUCLEI} {nuc_group} nuclei in the spot circle"
                            if nuc_group is not None else "")
                         + "; intersected before partition remap; nothing imputed")

    kept_idx = np.where(covered)[0]
    if len(kept_idx) < 2 * n_folds:
        raise ValueError(f"target {target!r} covers too few spots ({len(kept_idx)}) to episode")
    if within_patient_scale is None:
        within_patient_scale = bool(bank)
    codes_kept = np.unique(patient_codes[kept_idx], return_inverse=True)[1]
    if within_patient_scale:
        y_kept = _within_patient_standardise(y_g[kept_idx].reshape(-1, 1), codes_kept).reshape(-1)
        baseline_kept = _within_patient_standardise(baseline_g[kept_idx], codes_kept)
    elif centre_baseline:
        # Response AND every competitor column are within-patient centred over
        # the rows actually used (covariate values only for the competitor; the
        # response never enters it). An uncentred competitor scored against a
        # centred response carries between-patient offsets that cannot
        # transfer to held-out patients.
        y_kept = _residualize_against_onehot(y_g[kept_idx].reshape(-1, 1), codes_kept).reshape(-1)
        baseline_kept = _residualize_against_onehot(baseline_g[kept_idx], codes_kept)
    else:
        # Pre-2026-10-11 behaviour (sealed v0.3 receipts): response centred over
        # all loaded rows, competitor uncentred.
        y_kept = _residualize_against_onehot(y_g.reshape(-1, 1), patient_codes).reshape(-1)[kept_idx]
        baseline_kept = baseline_g[kept_idx]
    remap = -np.ones(sub.n, dtype=np.int64)
    remap[kept_idx] = np.arange(len(kept_idx))
    proposal_local = np.sort(remap[np.intersect1d(kept_idx, proposal_rows_g)])
    replication_local = np.sort(remap[np.intersect1d(kept_idx, replication_rows_g)])
    # Recomputed WITHIN kept-row space (not remapped), so a dropped
    # uncovered spot cannot leave a dangling neighbour reference and
    # neighbours still never cross a sample/patient boundary.
    coords_k, patient_codes_k = sub.coords[kept_idx], patient_codes[kept_idx]
    neighbours_k = knn_neighbors(coords_k, k=k_neighbors, groups=patient_codes_k)

    partitions = EpisodePartitions(proposal_local.astype(np.int64), replication_local.astype(np.int64),
                                   np.array([], dtype=np.int64))
    modalities = {"X": sub.A[kept_idx]}
    context = {
        "coords": coords_k, "patient_codes": patient_codes_k, "neighbors": neighbours_k,
        "spot_diameter_px": np.asarray(sub.meta["spot_diameter_px"])[kept_idx],
        "baseline": baseline_kept, "program_scores": program_scores[kept_idx],
        "depth_log1p": depth_log[kept_idx], "cellvit_frac": cv_frac[kept_idx],
    }
    if include_tissue_features:
        # These are measured tissue-side quantities, kept wholly separate from
        # the molecular response.  The caller fits its tissue dictionary on
        # D_P only; this loader performs no PCA, clustering, or response fit.
        context["tissue_raw"] = {
            "nuclear": tissue["nuclear"][kept_idx],
            "composition": tissue["composition"][kept_idx],
            "graph": tissue["graph"][kept_idx],
            "boundary": tissue["boundary"][kept_idx],
            "architecture": image[kept_idx],
        }
    metadata = {
        "cohort": cohort, "fold": int(fold), "n_folds": int(n_folds), "target": target, "seed": int(seed),
        "null_source": "replication_structured",
        "gene_symbols": [str(g) for g in sub.meta["gene_symbols"]],
        "fit_patients": list(fit_patients), "held_patients": sorted(held_patients),
        "n_rows_total": int(sub.n), "n_rows_kept": int(len(kept_idx)),
        "tissue_features_required": bool(include_tissue_features),
        "tissue_coverage_fraction": (float(len(kept_idx) / sub.n) if include_tissue_features else None),
        "coverage_fraction": float(len(kept_idx) / sub.n), "coverage_rule": coverage_rule,
        "cellvit_classes": list(CELLVIT_CLASSES), "program_names": list(program_names),
        "baseline_width": int(baseline_g.shape[1]), "baseline_columns": baseline_names,
        "baseline_centred": bool(centre_baseline or within_patient_scale),
        "response_standardised": bool(within_patient_scale),
        "baseline_standardised": bool(within_patient_scale),
        "nbr_mp_program": nbr_mp_info, "own_mp_program": own_mp_info,
    }
    if nuc_group is not None:
        metadata["nuc_target"] = {"group": nuc_group, "stat": nuc_stat, "min_nuclei": NUC_MIN_NUCLEI,
                                  "classes": (list(NUC_GROUPS[nuc_group]) if NUC_GROUPS[nuc_group]
                                              else list(CELLVIT_CLASSES)),
                                  "n_covered": int(len(kept_idx))}
    if v05:
        from morpheus_encoder.measurement_bank import within_group_sd
        raw_y = y_g[kept_idx]
        zero_frac = float(np.mean(raw_y == 0))
        wsd = float(within_group_sd(raw_y.reshape(-1, 1), codes_kept, np.arange(len(kept_idx)))[0])
        warn = []
        if zero_frac > 0.9:
            warn.append(f"{zero_frac:.1%} of response values are zero")
        if wsd < 1e-3:
            warn.append(f"pooled within-patient sd {wsd:.2e} < 1e-3")
        metadata["substrate_version"] = "v05"
        metadata["target_degenerate_warning"] = "; ".join(warn) or None
        metadata["response_zero_fraction"] = zero_frac
        metadata["response_within_patient_sd"] = wsd
    episode = DiscoveryEpisode(modalities=modalities, response=y_kept.astype(np.float64),
                               partitions=partitions, context=context, metadata=metadata)
    if not bank:
        return episode
    sources = _bank_sources(sub, kept_idx, episode, target=target, cohort=cohort, data_dir=d,
                            embed_root=embed_root, program_sets=program_sets, tissue=tissue,
                            image=image, image_covered=image_covered,
                            program_info=nbr_mp_info or own_mp_info)
    from morpheus_encoder.measurement_bank import build_bank_with_report
    options = dict(bank_options or {})
    if v05:
        from morpheus_encoder.measurement_bank import score_gene_sets
        import hashlib as _hashlib
        logcpm_k = sources["logcpm"]
        blocks, prog_names, matched = [], [], []
        for prefix_, sets_ in sources["gene_sets"].items():
            Xs, nm, mt = score_gene_sets(logcpm_k, sources["genes"], sets_, prefix_)
            blocks.append(Xs); prog_names += nm; matched += mt
        prog91 = np.column_stack(blocks).astype(np.float32)
        sources["programmes_precomputed"] = (prog91, prog_names, matched)
        sources["groups"] = codes_kept
        fold0_fit = sorted(set(all_patients.tolist()) - set(folds[0].tolist()))
        ref_rows = np.flatnonzero(np.isin(np.asarray(patients)[kept_idx], np.asarray(fold0_fit)))
        ident = "\n".join(f"{a}|{b}" for a, b in zip(np.asarray(sub.meta["sample_id"])[kept_idx][ref_rows],
                                                     np.asarray(sub.meta["barcode"])[kept_idx][ref_rows]))
        sources["reference_proposal_rows"] = ref_rows
        sources["reference_key"] = _hashlib.sha256(ident.encode()).hexdigest()
        if is_nbr_target or is_own_target:
            from morpheus_encoder.measurement_bank import _neighbour_mean as _bank_nbr_mean
            j = (nbr_mp_info or own_mp_info)["program_index"]
            sources["leak_guard"] = {
                "reference": _bank_nbr_mean(program_scores[kept_idx][:, [j]].astype(np.float64),
                                            neighbours_k)[:, 0],
                "threshold": 0.9, "views": None,
                "reason": "within-patient |r| >= 0.9 with the target programme's own k=6 neighbour mean "
                          "(a spatial-smoothing near-copy of the excluded target neighbour mean)"}
        elif nuc_group is not None:
            sources["leak_guard"] = {
                "reference": y_g[kept_idx], "threshold": 0.9, "views": ["cellvit_structure"],
                "reason": "within-patient |r| >= 0.9 with the nuclear-morphometry response"}
        options = {"align_decompositions": True, "near_constant_sd": 0.01, **options}
    bank_dict, report = build_bank_with_report(episode, views=bank_views, cache_root=bank_cache_root,
                                               sources=sources, seed=seed, **options)
    if is_nbr_target:
        # Protocol v0.4 neighbourhood-programme competitor: own-spot PCA and
        # neighbour-mean PCA, taken from the bank (fitted on D_P genes only,
        # response-free) and put on the same within-patient scale.
        extra, names = [], []
        if "pca" in bank_dict:
            extra.append(np.asarray(bank_dict["pca"]["X"], dtype=np.float64))
            names += [f"pca:{c}" for c in bank_dict["pca"]["columns"]]
        if "context_neighbour_mean" in bank_dict:
            view = bank_dict["context_neighbour_mean"]
            keep = [i for i, c in enumerate(view["columns"]) if c.startswith("nbr:PC")]
            extra.append(np.asarray(view["X"], dtype=np.float64)[:, keep])
            names += [view["columns"][i] for i in keep]
        if names:
            block = np.column_stack(extra)
            if within_patient_scale:
                block = _within_patient_standardise(block, codes_kept)
            elif centre_baseline:
                block = _residualize_against_onehot(block, codes_kept)
            context = {**context, "baseline": np.column_stack([context["baseline"], block])}
            metadata = {**metadata, "baseline_columns": list(metadata["baseline_columns"]) + names,
                        "baseline_width": int(context["baseline"].shape[1])}
    if v05:
        context, metadata = _v05_competitor(
            context, metadata, raw_baseline=baseline_g[kept_idx], sources=sources, bank_dict=bank_dict,
            codes_kept=codes_kept, coords_k=coords_k, neighbours_k=neighbours_k,
            proposal=proposal_local, morph_family=is_morph_family,
            scale=within_patient_scale, centre=centre_baseline)
    return DiscoveryEpisode(modalities=modalities, response=y_kept.astype(np.float64),
                            partitions=partitions, context=context,
                            metadata={**metadata, "bank": bank_dict, "bank_report": report})


def _v05_competitor(context, metadata, *, raw_baseline, sources, bank_dict, codes_kept, coords_k,
                    neighbours_k, proposal, morph_family, scale, centre):
    """Protocol v0.5 competitor: Hallmark own-spot scores for every target and,
    for morphology targets, neighbour means (k = 6 and 18, within patient,
    self excluded) of the 91 programme scores and the aligned PCA columns.
    Response-free throughout. Columns with pooled within-patient sd < 0.01 on
    proposal rows (before standardising) are dropped from the whole
    competitor and listed."""
    from morpheus_encoder.measurement_bank import _neighbour_mean as _bank_nbr_mean, within_group_sd

    prog, prog_names, _ = sources["programmes_precomputed"]
    prog = np.asarray(prog, dtype=np.float64)
    hall = [i for i, c in enumerate(prog_names) if c.startswith("hallmark:")]
    blocks = [prog[:, hall]]
    names = [prog_names[i] for i in hall]
    if morph_family:
        base, base_names = [prog], list(prog_names)
        if "pca" in bank_dict:
            base.append(np.asarray(bank_dict["pca"]["X"], dtype=np.float64))
            base_names += list(bank_dict["pca"]["columns"])
        base = np.column_stack(base)
        nb18 = knn_neighbors(coords_k, k=18, groups=codes_kept)
        for prefix_, nb in (("nbr:", neighbours_k), ("nbr18:", nb18)):
            blocks.append(_bank_nbr_mean(base, nb).astype(np.float64))
            names += [prefix_ + c for c in base_names]
    added = np.column_stack(blocks)
    prev_scaled = np.asarray(context["baseline"], dtype=np.float64)
    prev_names = list(metadata["baseline_columns"])
    n_prev = raw_baseline.shape[1]
    # columns already appended after the base competitor (nbr_mp PCA block)
    # are kept as they are; the near-constant rule reads raw values
    raw_all = np.column_stack([raw_baseline, prev_scaled[:, n_prev:], added])
    sd = within_group_sd(raw_all, codes_kept, proposal)
    if scale:
        added = _within_patient_standardise(added, codes_kept)
    elif centre:
        added = _residualize_against_onehot(added, codes_kept)
    all_X = np.column_stack([prev_scaled, added])
    all_names = prev_names + names
    keep = np.flatnonzero(sd >= 0.01)
    dropped = [all_names[i] for i in np.flatnonzero(sd < 0.01)]
    context = {**context, "baseline": all_X[:, keep]}
    metadata = {**metadata, "baseline_columns": [all_names[i] for i in keep],
                "baseline_width": int(len(keep)), "baseline_dropped_near_constant": dropped,
                "baseline_added_v05": names}
    return context, metadata


def _moran_ranked_program(rank, kind, program_scores, program_names, coords, patient_codes,
                          proposal_rows, k_neighbors):
    """The k-th highest within-patient Moran's I programme, ranked on the
    proposal (D_P) rows only -- the single ranking behind both `nbr_mp{k}`
    (its neighbour mean) and `own_mp{k}` (its own-spot score), so the two
    target kinds name the same programme on the same fold. Returns the
    programme index and the receipt record (the ranking can vary by fold)."""
    moran_fit = np.array([
        moran_i(program_scores[proposal_rows, j], coords[proposal_rows],
               patient_codes[proposal_rows], k=k_neighbors)
        for j in range(program_scores.shape[1])
    ])
    top8 = np.argsort(-moran_fit)[:8]
    if not (1 <= rank <= len(top8)):
        raise ValueError(f"{kind} rank must be in [1, {len(top8)}]")
    program_index = int(top8[rank - 1])
    return program_index, {
        "rank": rank, "program_index": program_index,
        "program_name": program_names[program_index] if program_names else None,
        "moran_i_on_fit": float(moran_fit[program_index]),
        "top8_program_names": ([program_names[i] for i in top8] if program_names
                               else [int(i) for i in top8]),
    }


def _target_exclusions(target, program_info):
    """Bank columns that are the response or a deterministic function of it
    (the same exclusion clause the competitor baseline follows)."""
    if target.startswith("morph_pc"):
        return {"hoptimus": {"columns": None,
                             "reason": "response is a principal component of this embedding"}}
    if target == "cv_count":
        return {"cellvit_structure": {"columns": ["nuclear:log1p_count"],
                                      "reason": "response is the per-spot CellViT nucleus count"}}
    if target == "cv_area":
        return {"cellvit_structure": {"columns": ["nuclear:log1p_mean_area"],
                                      "reason": "response is the per-spot mean nuclear area"}}
    if target.startswith("cv_frac_"):
        cols = [f"composition:frac_{c}" for c in CELLVIT_CLASSES] + ["composition:class_entropy"]
        return {"cellvit_structure": {"columns": cols,
                                      "reason": "class fractions sum to one, so the others determine the response"}}
    if target.startswith("nuc_") and target.endswith("_density"):
        group = target.split("_")[1]
        cols = ["nuclear:log1p_count"] + ([f"composition:frac_{c}" for c in NUC_GROUPS[group]]
                                          if NUC_GROUPS.get(group) else [])
        return {"cellvit_structure": {"columns": cols,
                                      "reason": "density = count x class fraction / spot area"}}
    name = (program_info or {}).get("program_name")
    if target.startswith("nbr_mp") and name:
        return {"context_neighbour_mean": {"columns": [f"nbr:gavish:{name}"],
                                           "reason": "response is this programme's neighbour mean"}}
    if target.startswith("own_mp") and name:
        return {"programmes": {"columns": [f"gavish:{name}"],
                               "reason": "response is this programme's own-spot score"},
                "context_neighbour_mean": {"columns": [f"nbr:gavish:{name}"],
                                           "reason": "this programme's neighbour mean is excluded with the target"}}
    return {}


def _bank_sources(sub, kept_idx, episode, *, target, cohort, data_dir, embed_root, program_sets,
                  tissue, image, image_covered, program_info):
    """Row-aligned raw inputs for `measurement_bank.build_bank`, restricted to
    the episode's kept rows. Optional sources that cannot be read are left out
    so the bank reports that view as unavailable rather than failing."""
    from morpheus_encoder.measurement_bank import SPOT_CAVEAT, load_hallmark_sets

    gene_sets = {"gavish": list(program_sets or [])}
    hallmark = load_hallmark_sets()
    if hallmark:
        gene_sets["hallmark"] = hallmark
    sources = {
        "cohort": f"hest_{cohort}", "logcpm": sub.meta["logcpm_full"][kept_idx],
        "genes": [str(g) for g in sub.meta["all_genes"]], "gene_sets": gene_sets,
        "fm_groups": np.asarray(sub.meta["sample_id"])[kept_idx],
        "fm_barcodes": np.asarray(sub.meta["barcode"])[kept_idx],
        "fm_caveat": SPOT_CAVEAT, "fm_scale": "spot",
        "coords": np.asarray(episode.context["coords"]),
        "patient": np.asarray(sub.meta["patient"])[kept_idx],
        "neighbours": episode.context["neighbors"],
        "cellvit_classes": list(CELLVIT_CLASSES),
        "exclude": _target_exclusions(target, program_info),
    }
    try:
        if image is None:
            image, image_covered = load_image_embeddings(sub, "hoptimus", cohort=cohort, embed_root=embed_root)
        sources["hoptimus"] = (image[kept_idx], np.asarray(image_covered, dtype=bool)[kept_idx])
    except (OSError, ValueError, KeyError):
        pass
    try:
        if tissue is None:
            tissue = load_cellvit_multiscale_features(sub, cohort=cohort, data_dir=data_dir)
        sources["cellvit"] = {k: np.asarray(v)[kept_idx] for k, v in tissue.items()}
    except (OSError, ValueError, KeyError):
        pass
    return sources


def episode_null_generators(episode, *, nx=6, ny=6):
    """Structured null closures for `general_discovery.evaluate_structured_nulls`.

    Built once per (fold, target) episode and reused for every frozen
    candidate. Each closure has the `(index, X, y, context) -> (X, y,
    context)` signature that function expects, but -- as its docstring
    explicitly allows -- IGNORES its own arguments and instead returns a
    structurally-nulled version of THIS episode's own D_R (replication)
    rows. The caller therefore passes D_P as the literal `proposal_X,
    proposal_y` (so the frozen B/H model is fit exactly as it was for the
    real receipt) and gets back, on each iteration, an alternative D_R to
    test that same fixed fit against.

    `spatial_block` and `within_patient_perm` destroy the response's
    relationship to the observed genes while preserving (respectively)
    local spatial autocorrelation and patient membership. `interaction_break`
    additionally needs to know which single constituent of a >=2-way
    candidate to shuffle, so it is a candidate -> generator FACTORY, not a
    ready closure.
    """
    replication = episode.partitions.replication
    if replication.size == 0:
        raise ValueError("episode has no replication rows to build a null from")
    coords_r = np.asarray(episode.context["coords"])[replication]
    patient_r = np.asarray(episode.context["patient_codes"])[replication]
    X_r = episode.observed_matrix()[replication]
    y_r = np.asarray(episode.response, dtype=float)[replication]
    # `context` here is always None: every real-episode candidate is
    # proposed over `episode.observed_matrix()` (gene expression only, no
    # `context_variables`), and a fixed nuisance `baseline` is threaded
    # through separately (see `evaluate_structured_nulls`'s own
    # `proposal_baseline`/`replication_baseline`). Returning `None` keeps
    # this consistent with how the runner builds the un-nulled receipt.
    fake_sub = _NullSubstrate(coords_r, patient_r)
    # A nulled response for draw `index` is identical for every candidate in the
    # packet, so each draw is built once and reused (the same holds per
    # constituent column for interaction_break); the block swap is pure Python
    # over patients x grid cells and was the packet's dominant cost.
    _y_block: dict = {}
    _y_perm: dict = {}
    _x_break: dict = {}

    def _spatial_block(index, X, y, context):
        if index not in _y_block:
            _y_block[index] = null_hest(y_r, fake_sub, seed=index, nx=nx, ny=ny)
        return X_r, _y_block[index], None

    def _within_patient_perm(index, X, y, context):
        if index not in _y_perm:
            rng = np.random.default_rng(index)
            y_null = y_r.copy()
            for g in np.unique(patient_r):
                idx = np.where(patient_r == g)[0]
                y_null[idx] = y_r[idx[rng.permutation(len(idx))]]
            _y_perm[index] = y_null
        return X_r, _y_perm[index], None

    def _interaction_break_factory(candidate):
        if len(candidate.variables) < 2:
            raise ValueError("interaction_break needs a candidate with >= 2 constituents")
        col = candidate.variables[0]

        def _generator(index, X, y, context):
            key = (col, index)
            if key not in _x_break:
                # memoise only the nulled COLUMN (a vector); a full matrix per
                # (column, draw) grew to tens of GB per worker.
                _x_break[key] = null_hest(X_r[:, col], fake_sub, seed=index, nx=nx, ny=ny)
            X_null = X_r.copy()
            X_null[:, col] = _x_break[key]
            return X_null, y_r, None
        return _generator

    return {"spatial_block": _spatial_block, "within_patient_perm": _within_patient_perm,
           "interaction_break": _interaction_break_factory}


class _NullSubstrate:
    """The minimal `null_hest(B, sub, ...)` duck-typed contract: `sub.coords`
    and `sub.coord_groups` -- not a real `Substrate` because an episode's
    replication rows have no `.A`/`.split` to preserve."""

    def __init__(self, coords, coord_groups):
        self.coords, self.coord_groups = coords, coord_groups
