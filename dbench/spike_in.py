"""Tier II: planted objects on a real scientific substrate (LOCKIN Sec. 7).

Tier I (`dbench.families`) is exact synthetic truth: every latent is a drawn
Gaussian and the "real substrate" is a random orthonormal projection. Tier II
keeps the exact-truth discipline -- a declared `TruthSet`, a positive control,
a matched structured null -- but the background a method has to search through
is REAL data: real cell-to-cell covariance, real batch/site structure, real
biology, not a synthetic loading matrix.

This module is deliberately kept OUT of `dbench.environments.ALL_WORLDS`. The
Tier I manifest's generator set is the import closure from `ALL_WORLDS`
(`freeze.py`), and Tier II worlds are not byte-reproducible across machines in
the way Tier I's are -- they read a real, large, external data file rather
than drawing from `np.random.default_rng`. Importing this module from
`families.py` or `environments.py` would silently pull a non-portable,
multi-hundred-MB-backing generator into the Tier I freeze. Tier II gets its
own manifest (`tier2_manifest`, below) instead.

## Substrate: BMMC CITE-seq (GSE194122, the Multimodal Single-Cell Data
Integration challenge cohort), already on the persistent store.

90261 cells x 14087 features (RNA + 134 real ADT antibody-capture proteins),
4 real sequencing sites, 9 real donors, real cell-type labels. `Site` is a
genuine, substantial technical covariate in this cohort (ADT counts are
processing-batch-sensitive) -- exactly the "batch-structured molecular"
domain `dbench.world.DOMAINS` names and `dbench.splits.DEFERRED_DOMAINS`
reserves for Tier II.

`extract_bmmc_substrate.py` (run once, not part of this module) reads the
real h5ad, subsamples 6001 cells stratified on real (Site, DonorNumber) so
the cache keeps the real joint composition, and measures each of 20 real ADT
markers' real between-Site variance fraction (eta^2). Two markers are then
selected BY THAT MEASUREMENT, not by name:

  A_MARKER: the least Site-driven real marker (CD86, real eta^2 = 0.0018) --
    a clean real-data substrate for the planted true CAUSE, so the injected
    causal signal is not itself already confounded by Site in the real data.
  D_MARKER: the most Site-driven real marker (CD44, real eta^2 = 0.5611) --
    the genuine, measured (not invented) confound substrate: a method that
    scans for `D -> Y` will find a real, reproducible, high-SNR association
    that is entirely mediated by real Site membership.

## The planted object

  Y = beta * zscore(real A_MARKER) + kappa * site_shift(real Site) + noise

`site_shift` assigns each of the 4 real Site labels a fixed per-site level
(fixed by a hash of the site label, not fit to the data) and is added AS A
BLOCK CONSTANT -- every cell at that real site gets the same shift, so
composition and covariance are otherwise the real, untouched data. This is
the standard semi-synthetic recipe (real structure + one injected signal),
mirrored from `dbench.families.make_latent_confound`, but keyed on a REAL
grouping variable instead of a synthetic continuous U.

True object:  CAUSE:  A_MARKER -> Y                        (mode=mechanism)
False object: CONFOUND: D_MARKER -> Y, confounder=Site      (mode=mechanism,
              true=False, artifact_class="nuisance_confound")

Recovery means reporting A_MARKER and refusing D_MARKER, exactly as in
`make_latent_confound` -- the same scoring contract, now on real covariance.

## Structured null

`dbench.nulls.null_stratum_shuffle` was written for exactly this substrate
("This is the BMMC-style null", its own docstring) and needs no change:
shuffling Y within real Site strata preserves every real Site-level mean
(so the D_MARKER confound, which lives entirely at the Site level, survives
untouched -- correct, since it is supposed to remain "real, reproducible in
every environment, and not a mechanism") while destroying the within-site
A_MARKER pairing (so CAUSE:A_MARKER->Y is a false discovery in the null, as
required for FDR calibration).

`covariance_preservation_check` below is the promised implementation of the
forward reference in `nulls.py`'s module docstring
("the same discipline `spike_in.covariance_preservation_check` applies to
the semi-synthetic path"): it asserts that a null constructed this way left
the REAL background feature covariance matrix exactly unchanged, which is
what "matched structured null that preserves the substrate's ... structure"
(LOCKIN task 2) cashes out to when the background is real, unmodified data
and only the synthetic target is permuted.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .util import corr, derive_seed, knn_neighbors, morans_i, partial_corr, zscore
from .world import EnvMeta, Hypothesis, Structure, TruthSet, World

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_tier2_cache", "bmmc_adt_substrate.npz")

_SUBSTRATE_CACHE = {}


def _load_substrate():
    """Real (X, panel, site, donor, cell_type, a_marker, d_marker) -- cached
    in-process so repeated `make_molecular_confound` calls (one per role per
    seed in an episode) do not re-read the .npz."""
    if "d" not in _SUBSTRATE_CACHE:
        if not os.path.exists(_CACHE_PATH):
            raise FileNotFoundError(
                "Tier II substrate cache missing at %r. Run "
                "extract_bmmc_substrate.py once (reads the real BMMC h5ad "
                "from the persistent store) to build it." % _CACHE_PATH)
        d = np.load(_CACHE_PATH, allow_pickle=True)
        _SUBSTRATE_CACHE["d"] = {
            "X": d["X"], "panel": [str(x) for x in d["panel"]],
            "site": d["site"], "donor": d["donor"],
            "cell_type": d["cell_type"],
            "a_marker": str(d["a_marker"]), "d_marker": str(d["d_marker"]),
            "eta": d["eta"],
        }
    return _SUBSTRATE_CACHE["d"]


def _site_shift(site_labels, kappa, d_pool, site_pool):
    """A per-site level tied to D_MARKER's OWN real per-site mean (measured
    on the full substrate pool, not the current subsample, so it is a fixed
    quantity independent of which cells a given world draws) -- not an
    arbitrary alphabetical assignment.

    Earlier design used a fixed per-site integer level unrelated to any real
    per-site pattern; because the real between-site pattern of the confound
    marker is not monotone in alphabetical site order, that shift's linear
    correlation with the marker's real per-site means partly cancelled
    (measured: marginal p=0.44 despite each channel individually carrying a
    strong real Site effect). Tying the shift to the marker's own measured
    per-site mean removes that accidental cancellation and makes the
    intended channel -- "Y and D_MARKER share exactly D_MARKER's real
    Site-level pattern" -- the textbook confound construction, not a
    coincidence of label ordering.
    """
    d_pool = np.asarray(d_pool, dtype=np.float64)
    grand = float(d_pool.mean())
    uniq = sorted(set(site_pool.tolist()))
    level = {s: float(d_pool[site_pool == s].mean()) - grand for s in uniq}
    sd = float(np.std(list(level.values()))) or 1.0
    return kappa * np.array([level[s] / sd for s in site_labels],
                            dtype=np.float64)


def make_molecular_confound(seed=0, role="proposal", n=1500, snr="high",
                            artifacts=(), noise=0.8, kappa=1.4):
    """Tier II, domain=molecular: real BMMC CITE-seq substrate.

    Same `World` contract as every Tier I family in `dbench.families`, and
    the same CAUSE-vs-CONFOUND object pair as `make_latent_confound`, but
    every feature column is real single-cell protein data and the confound
    is keyed on the real Site covariate rather than a synthetic U.
    """
    scale = {"high": 1.6, "med": 1.0, "low": 0.5}[snr]
    sub = _load_substrate()
    n_pool = sub["X"].shape[0]
    n = min(int(n), n_pool)
    rs = np.random.default_rng(derive_seed("molecular_confound", seed, role,
                                           "sample"))
    rq = np.random.default_rng(derive_seed("molecular_confound", seed, role,
                                           "measurement_noise"))
    idx = rs.choice(n_pool, size=n, replace=False)

    panel = sub["panel"]
    X_real = sub["X"][idx]                          # (n, len(panel)), real
    site = sub["site"][idx]
    donor = sub["donor"][idx]
    cell_type = sub["cell_type"][idx]
    a_marker, d_marker = sub["a_marker"], sub["d_marker"]
    ja, jd = panel.index(a_marker), panel.index(d_marker)

    A = zscore(X_real[:, ja])
    D = zscore(X_real[:, jd])

    beta = 1.3 * scale
    shift = _site_shift(site, kappa * scale, sub["X"][:, jd], sub["site"])
    Y = beta * A + shift + noise * rq.standard_normal(n)

    lat = {"A_" + a_marker: A, "D_" + d_marker: D}
    for j, m in enumerate(panel):
        if m not in (a_marker, d_marker):
            lat["R_" + m] = zscore(X_real[:, j])     # real distractor markers

    objs = [
        Hypothesis(id="CAUSE:%s->Y" % a_marker, mode="mechanism",
                  operator="CAUSE", operands=(("A_" + a_marker),),
                  target="Y", params={"beta": float(beta)}, true=True,
                  notes="real ADT marker %s, measured real Site eta^2=%.4f "
                        "(near Site-invariant); genuine planted cause"
                        % (a_marker, float(sub["eta"][ja]))),
        Hypothesis(id="ART:%s" % d_marker, mode="mechanism",
                  operator="CONFOUND", operands=(("D_" + d_marker),),
                  target="Y", true=False, artifact_class="nuisance_confound",
                  parents=(("Site",),), params={"confounder": "Site"},
                  notes="real ADT marker %s, measured real Site eta^2=%.4f; "
                        "Y's Site-level shift and %s's real Site dependence "
                        "jointly produce a reproducible, non-causal "
                        "association" % (d_marker, float(sub["eta"][jd]), d_marker)),
    ]

    feature_of = {objs[0].id: A, objs[1].id: D}

    finished = []
    for h in objs:
        f = feature_of[h.id]
        eff = corr(Y, f)
        finished.append(_replace_effect(h, eff))

    modalities = {"X": zscore(X_real)}   # the real observed protein panel

    return World(
        name="molecular_confound/%s/s%d" % (role, seed),
        family="molecular_confound", domain="molecular",
        latents=lat, modalities=modalities, target=Y, target_name="Y",
        nuisance={"cell_type": cell_type},
        env=EnvMeta(role=role, labels={"Site": site, "Donor": donor},
                   world_seed=int(seed), template="molecular_confound",
                   tags={"snr": snr, "substrate": "BMMC_CITE_GSE194122",
                         "n_real_cells_in_pool": int(n_pool)}),
        structure=Structure(kind="iid"),
        interventions=None,
        truth=TruthSet(objects=tuple(finished)),
        meta={"snr": snr, "n": int(n), "a_marker": a_marker,
             "d_marker": d_marker, "panel": panel, "real_substrate": True,
             "tier": 2},
    )


def _replace_effect(h, eff):
    from dataclasses import replace as _r
    return _r(h, effect=float(eff))


TIER2_FAMILIES = {
    "molecular_confound": make_molecular_confound,
}


def _one_hot_drop_first(labels):
    """Categorical Site labels -> (n, K-1) dummy columns, one dropped as the
    reference level, for use as `partial_evidence` controls."""
    uniq = sorted(set(np.asarray(labels).tolist()))
    cols = [(np.asarray(labels) == lv).astype(np.float64) for lv in uniq[1:]]
    return cols


def specialist_molecular_confound(world):
    """Oracle-latent recovery check for `make_molecular_confound`, mirroring
    `dbench.oracle.specialist_confound`'s "recovery is a refusal as well as a
    discovery" contract: the real cause must be significant even after
    conditioning on real Site, and the confounded marker must NOT be, once
    Site is conditioned on.

    This is also this family's Test-A positive control: at snr="high" it must
    return `recovered=True` when handed the oracle latents.
    """
    from . import oracle as O
    y = world.target
    a_name = next(k for k in world.latents if k.startswith("A_"))
    d_name = next(k for k in world.latents if k.startswith("D_"))
    A = np.asarray(world.latents[a_name], dtype=float)
    D = np.asarray(world.latents[d_name], dtype=float)
    site_dummies = _one_hot_drop_first(world.env.labels["Site"])
    ev_a_marg = O.partial_evidence(y, A, [])
    ev_a_given_site = O.partial_evidence(y, A, site_dummies)
    ev_d_marg = O.partial_evidence(y, D, [])
    ev_d_given_site = O.partial_evidence(y, D, site_dummies)
    # Real single-cell sample sizes make a pure p-value test the wrong
    # instrument for "not a cause once Site is conditioned on": at n in the
    # thousands even a tiny real residual correlation (e.g. two real ADT
    # markers sharing a hair of biological covariance unrelated to Site)
    # reaches p < 0.01, which would make EVERY marker "significant" and the
    # refusal side of this test vacuous. The refusal is therefore graded on
    # effect SIZE (partial correlation), with a bound fixed relative to the
    # true object's own conditional effect rather than to zero -- D's
    # residual association with Y, once Site is removed, must be a small
    # fraction of A's.
    d_given_site_small = (abs(ev_d_given_site["effect"])
                         < 0.25 * abs(ev_a_given_site["effect"]))
    return {
        "a_name": a_name, "d_name": d_name,
        "p_A_marginal": ev_a_marg["p"], "p_A_given_site": ev_a_given_site["p"],
        "p_D_marginal": ev_d_marg["p"], "p_D_given_site": ev_d_given_site["p"],
        "effect_A_given_site": ev_a_given_site["effect"],
        "effect_D_given_site": ev_d_given_site["effect"],
        "recovered": bool(ev_a_marg["p"] < 0.01 and ev_a_given_site["p"] < 0.01
                          and ev_d_marg["p"] < 0.01
                          and d_given_site_small),
        "criterion": ("A significant marginally and given Site; D significant "
                     "marginally, but its residual effect given Site is under "
                     "1/4 of A's -- the refusal is graded on effect size, not "
                     "a p-value, because real-data sample sizes make near-"
                     "zero residual correlations routinely 'significant'"),
    }


# --------------------------------------------------------------------------
# the promised covariance-preservation check
# --------------------------------------------------------------------------
def covariance_preservation_check(world, null_world, tol=1e-9):
    """Assert a Tier II structured null left the REAL background covariance
    matrix exactly alone.

    Tier I nulls (`dbench.nulls`) preserve statistics of SYNTHETIC data
    approximately (autocorrelation within a tolerance, moments exactly).
    Tier II can do better and should be checked that it does: the background
    columns here are real, unmodified measurements, so a null that only
    permutes/shifts the synthetic target `Y` must leave the real covariance
    matrix of every other channel bit-for-bit identical, not merely close.
    Reports the measured Frobenius-relative deviation whether or not it
    passes, so a failure is a measurement rather than an assumption.
    """
    names = sorted(set(world.latents) & set(null_world.latents))
    if not names:
        raise ValueError("no shared real channels between world and its null")
    A = np.column_stack([np.asarray(world.latents[k], dtype=np.float64)
                         for k in names])
    B = np.column_stack([np.asarray(null_world.latents[k], dtype=np.float64)
                         for k in names])
    Ca = np.cov(A, rowvar=False)
    Cb = np.cov(B, rowvar=False)
    num = float(np.linalg.norm(Ca - Cb, ord="fro"))
    den = max(float(np.linalg.norm(Ca, ord="fro")), 1e-12)
    rel = num / den
    return {
        "kind": "covariance_preservation_check", "channels": names,
        "frobenius_abs": num, "frobenius_rel": rel, "tol": tol,
        "identical_arrays": bool(np.array_equal(A, B)),
        "passes": bool(rel <= tol),
    }


def tier2_manifest():
    """A lightweight provenance record for the Tier II delivery -- NOT a
    byte-reproducibility manifest like Tier I's (real external data is not
    expected to hash identically across copies of the source file), but
    enough to pin which real substrate, which real markers, and which
    extraction seed produced the cache a Tier II run reads."""
    import hashlib
    import json
    meta_path = _CACHE_PATH.replace(".npz", ".meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    cache_sha256 = None
    if os.path.exists(_CACHE_PATH):
        h = hashlib.sha256()
        with open(_CACHE_PATH, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        cache_sha256 = h.hexdigest()
    return {"families": sorted(TIER2_FAMILIES), "substrate_meta": meta,
           "cache_sha256": cache_sha256, "cache_path": _CACHE_PATH}


# ==========================================================================
# Tier II, substrate 2: real HEST Visium spatial transcriptomics
# ==========================================================================
#
# C5 (plan `calm-mixing-candy.md`, 2026-09-26): `make_hest_spatial` and
# `make_hest_context_threshold_space` used to read a SINGLE cached HEST
# slide (`_HEST_CACHE_PATH` below, sample INT1 only) and drew both the
# "proposal" and "replication" role's rows from that one slide/patient --
# not patient-disjoint, because there was only ever one patient. They now
# read the real, patient-disjoint 24-patient ccRCC cohort through
# `morpheus.tier2.substrates.hest.load` instead (`_load_hest_role_substrate`
# below): a FIXED, seed-independent patient-level split (`_hest_role_patient_sets`)
# assigns every real patient to D_P ("proposal") or D_R ("replication"),
# disjoint by construction, and HVG selection / standardisation statistics
# are fit on D_P only (`fit_patients=d_p`) regardless of which role is
# being read -- a replication packet never influences its own gene
# selection or scaling. `_load_hest_substrate`/`_HEST_CACHE_PATH` below are
# UNCHANGED and kept: `dbench.diagnostic_path.make_hest_spatial_packet` and
# `morpheus_encoder.scripts.score_hest_general_receipts` still read the old
# single-slide cache directly and are out of this change's file scope.

_HEST_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_tier2_cache", "hest_int1_substrate.npz")

_HEST_CACHE = {}


def _load_hest_substrate():
    if "d" not in _HEST_CACHE:
        if not os.path.exists(_HEST_CACHE_PATH):
            raise FileNotFoundError(
                "Tier II HEST substrate cache missing at %r. Run "
                "extract_hest_substrate.py once (reads the real HEST h5ad "
                "pulled from MahmoodLab/hest) to build it." % _HEST_CACHE_PATH)
        d = np.load(_HEST_CACHE_PATH, allow_pickle=True)
        _HEST_CACHE["d"] = {
            "X": d["X"], "genes": [str(x) for x in d["genes"]],
            "coords": d["coords"], "b_gene": str(d["b_gene"]),
            "c_gene": str(d["c_gene"]), "moran_b": float(d["moran_b"]),
            "moran_c": float(d["moran_c"]),
        }
    return _HEST_CACHE["d"]


# --------------------------------------------------------------------------
# C5: the patient-disjoint, multi-slide real HEST loader.
# --------------------------------------------------------------------------

#: FIXED and independent of any packet `seed`: every call, whatever `seed`
#: or `role` it is made with, must agree on which real patients are D_P
#: (proposal) vs D_R (replication). Only injected noise and within-role row
#: sub-sampling vary with a packet's own `seed`.
_HEST_ROLE_SPLIT_SEED = 20260926

_HEST_ROLE_SUB_CACHE = {}


def _hest_all_patients(cohort="ccrcc", data_dir=None):
    """Real patient ids from `_metadata.json` only -- no expression matrix
    read -- so the D_P/D_R patient partition can be fixed before the
    (larger) `hest.load()` call that reads every real h5ad."""
    from morpheus.tier2.substrates import hest as _hest_substrate
    cfg = _hest_substrate.COHORTS[cohort]
    d = data_dir or cfg["dir"]
    prefix = cfg["prefix"]
    patients = []
    for sid in _hest_substrate.measure_available_samples(cohort=cohort, data_dir=d):
        meta_path = os.path.join(d, f"{prefix}{sid}_metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as fh:
            patients.append(str(json.load(fh)["patient"]))
    return sorted(set(patients))


def _hest_role_patient_sets(cohort="ccrcc", data_dir=None, frac_discovery=0.5):
    """A deterministic, packet-seed-independent patient-level split: D_P
    (proposal/fit patients) and D_R (replication patients), disjoint by
    construction (`cscd.spike_in.patient_split` assigns every real patient
    to exactly one side), fixed by `_HEST_ROLE_SPLIT_SEED`."""
    from cscd.spike_in import patient_split as _patient_split
    all_patients = np.array(_hest_all_patients(cohort=cohort, data_dir=data_dir))
    if len(all_patients) < 4:
        raise ValueError(
            "need at least 4 real HEST patients for a patient-disjoint "
            "D_P/D_R split, found %d for cohort %r" % (len(all_patients), cohort))
    split = _patient_split(all_patients, seed=_HEST_ROLE_SPLIT_SEED,
                           frac_discovery=frac_discovery)
    d_p = sorted(all_patients[split == 0].tolist())
    d_r = sorted(all_patients[split == 1].tolist())
    assert not (set(d_p) & set(d_r)), "D_P/D_R patient sets must be disjoint"
    return d_p, d_r


def _select_hest_b_c_genes(X_dp, genes, coords_dp, groups_dp, k_neighbors=6):
    """Predeclared, response-blind B_GENE/C_GENE picks, computed on D_P
    only, before any Y is injected.

    B_GENE is the single highest-fit-variance HVG gene (`genes[0]`, since
    `hest.load` already sorts its returned gene panel by descending D_P
    variance). C_GENE reuses `_hest_decoy_gene`'s own rule (max Moran's I
    among genes only weakly correlated with B_GENE, on real D_P rows with
    real patient-grouped neighbours) -- the same predeclared-decoy
    discipline `make_hest_context_threshold_space` already applies to its
    own third (D_GENE) decoy, extended one step earlier in the pipeline.
    """
    b_gene = str(genes[0])
    B_dp = X_dp[:, 0]
    neighbors_dp = knn_neighbors(coords_dp, k=k_neighbors, groups=groups_dp)
    c_gene, _ = _hest_decoy_gene(X_dp, genes, [B_dp], neighbors_dp, {b_gene})
    return b_gene, c_gene


def _load_hest_role_substrate(role, *, cohort="ccrcc", data_dir=None,
                              n_hvg=300, min_depth=200, k_neighbors=6):
    """Real, patient-disjoint HEST substrate rows for one controlled-packet
    role ("proposal" -> D_P patients, anything else -> D_R patients).

    `A` is already patient-identity-residualised and standardised by
    `hest.load` (fit on D_P only, `fit_patients=d_p`), so callers use its
    columns directly as per-gene latents without re-`zscore`-ing them.
    B_GENE/C_GENE are picked once per (cohort, data_dir, n_hvg) and cached,
    so every role and every packet `seed` agrees on the same two genes.
    """
    key = (cohort, data_dir, n_hvg, min_depth, k_neighbors)
    if key not in _HEST_ROLE_SUB_CACHE:
        from morpheus.tier2.substrates import hest as _hest_substrate
        d_p, d_r = _hest_role_patient_sets(cohort=cohort, data_dir=data_dir)
        sub = _hest_substrate.load(cohort=cohort, data_dir=data_dir, n_hvg=n_hvg,
                                   min_depth=min_depth, fit_patients=d_p)
        patients = sub.meta["patient"]
        genes = [str(g) for g in sub.meta["gene_symbols"]]
        dp_mask = np.isin(patients, np.asarray(d_p))
        if not dp_mask.any():
            raise ValueError("D_P patient set matched no rows after load()'s "
                             "own depth filtering for cohort %r" % cohort)
        b_gene, c_gene = _select_hest_b_c_genes(
            sub.A[dp_mask], genes, sub.coords[dp_mask], sub.coord_groups[dp_mask],
            k_neighbors=k_neighbors)
        jc = genes.index(c_gene)
        neighbors_dp = knn_neighbors(sub.coords[dp_mask], k=k_neighbors,
                                     groups=sub.coord_groups[dp_mask])
        moran_c = morans_i(sub.A[dp_mask][:, jc], neighbors_dp)
        _HEST_ROLE_SUB_CACHE[key] = {
            "sub": sub, "d_p": d_p, "d_r": d_r, "genes": genes,
            "b_gene": b_gene, "c_gene": c_gene, "moran_c": float(moran_c),
        }
    cached = _HEST_ROLE_SUB_CACHE[key]
    role_patients = cached["d_p"] if role == "proposal" else cached["d_r"]
    patients = cached["sub"].meta["patient"]
    row_mask = np.isin(patients, np.asarray(role_patients))
    if not row_mask.any():
        raise ValueError("role %r matched no rows for cohort %r (after "
                         "load()'s own depth filtering)" % (role, cohort))
    rows = np.where(row_mask)[0]
    return {
        "X": cached["sub"].A[rows], "genes": cached["genes"],
        "coords": cached["sub"].coords[rows],
        "groups": cached["sub"].coord_groups[rows],
        "patients": patients[rows],
        "b_gene": cached["b_gene"], "c_gene": cached["c_gene"],
        "moran_c": cached["moran_c"],
        "d_p": cached["d_p"], "d_r": cached["d_r"],
    }


def make_hest_spatial(seed=0, role="proposal", n=None, snr="high",
                      artifacts=(), noise=0.8, beta_scale=1.4, k_neighbors=6,
                      cohort="ccrcc", data_dir=None, n_hvg=300, min_depth=200):
    """Tier II, domain=spatial: real HEST Visium substrate, real ccRCC
    cohort (24 patients), patient-disjoint by role (C5).

    Same `World` contract as Tier I's `spatial_interaction` family: one true
    SPACE object, one spatially-autocorrelated-but-uninvolved real gene as
    the artifact to reject, real k-NN neighbourhood structure computed
    within each real patient. `role="proposal"` reads only D_P patients;
    every other role reads only D_R patients -- disjoint by construction
    (`_hest_role_patient_sets`), so a replication packet can never share a
    single patient's tissue with its proposal packet.
    """
    scale = {"high": 1.6, "med": 1.0, "low": 0.5}[snr]
    sub_role = _load_hest_role_substrate(role, cohort=cohort, data_dir=data_dir,
                                         n_hvg=n_hvg, min_depth=min_depth,
                                         k_neighbors=k_neighbors)
    n_pool = sub_role["X"].shape[0]
    n = n_pool if n is None else min(int(n), n_pool)
    rs = np.random.default_rng(derive_seed("hest_spatial", seed, role, "sample"))
    rq = np.random.default_rng(derive_seed("hest_spatial", seed, role,
                                           "measurement_noise"))
    idx = rs.choice(n_pool, size=n, replace=False) if n < n_pool else np.arange(n_pool)

    genes = sub_role["genes"]
    # `A` is already patient-residualised and standardised, fit on D_P only
    # (`hest.load(fit_patients=d_p)`), so no further `zscore` is applied.
    X_real = sub_role["X"][idx]
    coords = sub_role["coords"][idx]
    groups = sub_role["groups"][idx]
    patients_role = sub_role["patients"][idx]
    b_gene, c_gene = sub_role["b_gene"], sub_role["c_gene"]
    jb, jc = genes.index(b_gene), genes.index(c_gene)

    neighbors = knn_neighbors(coords, k=k_neighbors, groups=groups)
    B = X_real[:, jb]
    C = X_real[:, jc]
    aggB = np.array([float(B[nb].mean()) if len(nb) else 0.0
                     for nb in neighbors])

    beta = beta_scale * scale
    Y = beta * zscore(aggB) + noise * rq.standard_normal(n)

    lat = {"B_" + b_gene: B, "C_" + c_gene: C, "aggB": aggB}
    for j, g in enumerate(genes):
        if g not in (b_gene, c_gene):
            lat["R_" + g] = X_real[:, j]

    moran_c_local = morans_i(C, neighbors)
    n_patients_role = int(len(set(patients_role.tolist())))
    true_h = Hypothesis(
        id="SPACE:%s->Y" % b_gene, mode="relational", operator="SPACE",
        operands=(("B_" + b_gene),), target="Y",
        params={"beta": float(beta), "agg": "mean", "k_neighbors": k_neighbors},
        true=True,
        notes="real gene %s, aggregated over a real k=%d spatial "
              "neighbour graph computed WITHIN each real patient, over the "
              "real HEST ccRCC cohort (role=%s, %d real patients)"
              % (b_gene, k_neighbors, role, n_patients_role))
    art_h = Hypothesis(
        id="ART:%s_spatially_smooth" % c_gene, mode="relational",
        operator="SPACE", operands=(("C_" + c_gene),), target="Y",
        true=False, artifact_class="spatially_smooth_decoy",
        params={"measured_morans_i": float(sub_role["moran_c"]),
               "measured_morans_i_on_this_draw": float(moran_c_local)},
        notes="real gene %s is independently, genuinely spatially "
              "autocorrelated on real D_P patients (Moran's I measured, not "
              "invented) but has no role in Y; a size-only spatial-smoothness "
              "detector flags it for the wrong reason" % c_gene)

    from dataclasses import replace as _r
    from .util import corr as _corr
    true_h = _r(true_h, effect=float(_corr(Y, aggB)))
    art_h = _r(art_h, effect=float(_corr(Y, C)))

    modalities = {"X": X_real}

    return World(
        name="hest_spatial/%s/s%d" % (role, seed),
        family="hest_spatial", domain="spatial",
        latents=lat, modalities=modalities, target=Y, target_name="Y",
        nuisance={},
        env=EnvMeta(role=role, labels={"patient": patients_role},
                   world_seed=int(seed), template="hest_spatial",
                   tags={"snr": snr, "substrate": "HEST_ccRCC_real_multi_patient",
                         "n_real_spots_in_pool": int(n_pool),
                         "n_real_patients_in_pool": n_patients_role}),
        structure=Structure(kind="spatial", coords=coords, neighbors=neighbors),
        interventions=None,
        truth=TruthSet(objects=(true_h, art_h)),
        meta={"snr": snr, "n": int(n), "b_gene": b_gene, "c_gene": c_gene,
             "real_substrate": True, "tier": 2,
             "d_p_patients": sub_role["d_p"], "d_r_patients": sub_role["d_r"]},
    )


def _hest_decoy_gene(X_real, genes, correlates, neighbors, excluded,
                     corr_thresh=0.20):
    """Choose one real, spatially smooth competing explanation before Y.

    The rule examines only the fixed HEST substrate: among genes weakly
    correlated with every array in `correlates` (the true inputs already
    chosen), use the spatially smoothest one.  It is therefore a realistic
    predeclared decoy, not a response-selected foil.  `correlates` is a
    list so this one rule serves both C5's B_GENE-only decoy pick
    (`_select_hest_b_c_genes`, correlates=[B]) and this module's original
    B-and-C-excluding D_GENE pick (correlates=[B, C]) without duplicating it.
    Callers pass already-scaled (or already-standardised) arrays; nothing
    here re-`zscore`s a column, since `hest.load`'s `A` is standardised once,
    fit on D_P, and must not be re-scaled per candidate gene.
    """
    candidates = []
    for j, gene in enumerate(genes):
        if gene in excluded:
            continue
        value = X_real[:, j]
        if any(abs(corr(value, c)) > corr_thresh for c in correlates):
            continue
        candidates.append((morans_i(value, neighbors), str(gene), value))
    if not candidates:
        for j, gene in enumerate(genes):
            if gene not in excluded:
                value = X_real[:, j]
                candidates.append((morans_i(value, neighbors), str(gene), value))
    _score, gene, value = max(candidates, key=lambda x: (x[0], x[1]))
    return gene, value


def make_hest_context_threshold_space(seed=0, role="proposal", n=None,
                                      snr="high", artifacts=(), noise=0.75,
                                      beta_scale=1.8, k_neighbors=6,
                                      cohort="ccrcc", data_dir=None,
                                      n_hvg=300, min_depth=200):
    """A compositional object placed on measured HEST expression and geometry.

    ``Y = beta * 1[C > median(C)] * 1[mean_N(B) > median(mean_N(B))] + eps``.
    B and C are real measured genes, and ``N`` is the real per-patient
    spatial neighbour graph.  The controlled response creates known ground
    truth; everything that makes its discovery difficult -- gene
    covariance, tissue geometry and a spatially smooth real decoy -- comes
    from the real, patient-disjoint HEST ccRCC cohort (C5) rather than a
    generated tissue simulator or a single cached slide.
    """
    scale = {"high": 1.6, "med": 1.0, "low": 0.5}[snr]
    sub_role = _load_hest_role_substrate(role, cohort=cohort, data_dir=data_dir,
                                         n_hvg=n_hvg, min_depth=min_depth,
                                         k_neighbors=k_neighbors)
    n_pool = sub_role["X"].shape[0]
    n = n_pool if n is None else min(int(n), n_pool)
    rs = np.random.default_rng(derive_seed("hest_context_threshold_space", seed, role, "sample"))
    rq = np.random.default_rng(derive_seed("hest_context_threshold_space", seed, role, "measurement_noise"))
    idx = rs.choice(n_pool, size=n, replace=False) if n < n_pool else np.arange(n_pool)
    genes = sub_role["genes"]
    X_real = sub_role["X"][idx]
    coords = sub_role["coords"][idx]
    groups = sub_role["groups"][idx]
    b_gene, c_gene = sub_role["b_gene"], sub_role["c_gene"]
    B = X_real[:, genes.index(b_gene)]
    C = X_real[:, genes.index(c_gene)]
    neighbors = knn_neighbors(coords, k=k_neighbors, groups=groups)
    aggB = np.array([float(B[nb].mean()) if len(nb) else 0.0 for nb in neighbors])
    tau_b, tau_c = float(np.median(aggB)), float(np.median(C))
    gate, threshold = (C > tau_c).astype(float), (aggB > tau_b).astype(float)
    signal = gate * threshold
    beta = beta_scale * scale
    Y = beta * signal + noise * rq.standard_normal(n)
    decoy_gene, D = _hest_decoy_gene(X_real, genes, [B, C], neighbors, {b_gene, c_gene})
    aggD = np.array([float(D[nb].mean()) if len(nb) else 0.0 for nb in neighbors])
    lat = {"B_" + b_gene: B, "C_" + c_gene: C, "D_" + decoy_gene: D,
           "aggB": aggB, "aggD": aggD, "gateC": gate, "thrAggB": threshold}
    for j, gene in enumerate(genes):
        if gene not in {b_gene, c_gene, decoy_gene}:
            lat["R_" + gene] = X_real[:, j]
    true_h = Hypothesis(
        id="HEST:COND(C)&THRESH(SPACE(B))->Y", mode="relational",
        operator="CONDITION", operands=("B_" + b_gene, "C_" + c_gene), target="Y",
        params={"beta": float(beta), "tau_aggB": tau_b, "tau_C": tau_c,
                "agg": "mean", "k_neighbors": int(k_neighbors),
                "composition": ["CONDITION", "THRESHOLD", "SPACE"],
                "terms": [{"op": "CONDITION", "operands": ["C_" + c_gene],
                           "params": {"tau": tau_c}},
                          {"op": "THRESHOLD", "operands": ["B_" + b_gene],
                           "params": {"tau": tau_b}},
                          {"op": "SPACE", "operands": ["B_" + b_gene],
                           "params": {"agg": "mean"}}]},
        context={"gate_var": "C_" + c_gene, "gate_threshold": tau_c},
        notes="controlled context-by-thresholded-neighbourhood signal on real, "
              "patient-disjoint HEST ccRCC tissue")
    art_h = Hypothesis(
        id="ART:%s_spatially_smooth" % decoy_gene, mode="relational", operator="SPACE",
        operands=("D_" + decoy_gene,), target="Y", true=False,
        artifact_class="spatially_smooth_decoy",
        params={"measured_morans_i": float(morans_i(D, neighbors)), "selection": "fixed_substrate_rule"},
        notes="real spatially smooth gene selected without Y and not part of the planted object")
    from dataclasses import replace as _r
    true_h = _r(true_h, effect=float(corr(Y, signal)))
    art_h = _r(art_h, effect=float(corr(Y, aggD)))
    return World(
        name="hest_context_threshold_space/%s/s%d" % (role, seed),
        family="hest_context_threshold_space", domain="spatial", latents=lat,
        modalities={"X": X_real}, target=Y, target_name="Y", nuisance={},
        env=EnvMeta(role=role, labels={"patient": sub_role["patients"][idx]},
                    world_seed=int(seed), template="hest_context_threshold_space",
                    tags={"snr": snr, "substrate": "HEST_ccRCC_real_multi_patient"}),
        structure=Structure(kind="spatial", coords=coords, neighbors=neighbors),
        interventions=None, truth=TruthSet(objects=(true_h, art_h)),
        meta={"snr": snr, "n": int(n), "b_gene": b_gene, "c_gene": c_gene,
              "decoy_gene": decoy_gene, "real_substrate": True, "tier": 2,
              "composition": ("CONDITION", "THRESHOLD", "SPACE"),
              "d_p_patients": sub_role["d_p"], "d_r_patients": sub_role["d_r"]})


TIER2_FAMILIES_HEST = {"hest_spatial": make_hest_spatial,
                       "hest_context_threshold_space": make_hest_context_threshold_space}


def hest_iid_permute_null(world, seed=0):
    """Structured null for the HEST substrate: full permutation of Y.

    Three geometry-based nulls were tried and measured before this one was
    adopted -- rigid toroidal translation with KD-tree nearest-spot
    matching, rotation about the tissue centroid, and a row-major-
    linearised block permutation (
    on a lexicographic spot ordering) -- and all three were measured to be
    MISCALIBRATED on this specific real sample (INT1, 1084 spots): the
    real B_GENE neighbourhood field's spatial autocorrelation length
    scale is comparable to the tissue's own extent, so a rigid geometric
    transform frequently maps the field back onto a correlated copy of
    itself. Swept over 15 seeds each: translation gave a systematically
    NEGATIVE, non-null effect (KD-tree degeneracy -- shifted queries
    collapsed onto under 30% of the real spots, since this tissue section
    does not fill its bounding box); rotation was better calibrated in
    Moran's-I preservation (median relative change ~0.03) but still
    flagged at p<0.05 on 10/15 seeds against a nominal 0.05; a spatial
    k-means within-cluster shuffle (the direct analogue of
    , which IS well-calibrated for the
    BMMC substrate) flagged on 15/15 -- this field is not organised into
    coarse regions the way a batch/site covariate is; it is smooth at
    every scale the k-NN graph reaches.

    A full permutation is the one operation on this substrate whose
    calibration is not in question (it is exact under exchangeability of
    Y, by construction), at the honestly-stated cost of over-breaking:
    it also destroys Y's own autocorrelation, not only its alignment
    with the candidate aggregate. Every real gene column is untouched,
    so  on the real background still
    applies and passes -- what is NOT claimed here is that Y's own
    (entirely synthetic) autocorrelation survives, which the geometric
    nulls above were meant to provide and were measured not to on this
    sample.
    """
    from . import nulls as _N
    return _N.null_iid_permute(world, seed=seed)
