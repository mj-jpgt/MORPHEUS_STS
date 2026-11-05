"""Effect-size calibration in substrate-native units.

The problem this exists to fix (measured, not hypothetical, in
`FINDINGS_P1.md` T1.3 and re-measured for this module below): a fixed
"correlation r=0.4" spike-in means something completely different on a
substrate with n=912 exchangeable-block time points than on a substrate with
tens of thousands of spots. Reusing the same raw r would either be invisible
noise on one substrate or an impossibly clean signal on another.

So "strength" is defined relative to two substrate-native numbers instead
of a fixed correlation:

  detection floor    the substrate's own MDE-80 at its evaluation n (Fisher-z,
                     same formula `cscd/real_discovery.mde_80` already uses).
                     strength=1.0 plants a signal `floor_multiple` times this
                     floor -- i.e. comfortably detectable at this n, not an
                     arbitrary constant.

  coupling ceiling    the largest mutual information (nats) this substrate's
                     A/B pairing has been observed to sustain, MEASURED out-
                     of-sample (a held-out year-block for climate, a held-out
                     patient-block for HEST/BMMC) rather than assumed.
                     Converted to an equivalent Gaussian-channel correlation
                     via r = sqrt(1 - exp(-2*I)), and used as a hard cap
                     (with a safety margin). Measured ceilings turned out
                     modest everywhere they were checked -- climate ~2.7
                     nats, HEST ccRCC 0.29-0.58 nats depending on morphology
                     backbone (REPLACING an earlier 5.5-nats prior that
                     could not be measured before real image embeddings
                     existed for this cohort), HEST kidney_injury
                     0.24-0.55 nats, BMMC's donor-held-out RNA<->ADT ~0.13
                     nats. None of these binds in practice: a realistic
                     strength=1.0 spike-in's floor-based target r sits far
                     below even the WEAKEST measured ceiling on every
                     substrate checked, so the cap is carried for
                     substrates where it might someday matter rather than
                     because it currently changes behaviour.

`strength` is therefore comparable across BMMC, HEST and NCEP: it always
means "this many multiples of the substrate's own floor, capped at what the
substrate's own ceiling allows" -- never a fixed absolute correlation.
"""
from __future__ import annotations

import numpy as np


def mde_80(n, alpha=0.05):
    """Minimum detectable correlation at 80% power, n independent units.
    Identical formula to `cscd/real_discovery.mde_80`; duplicated here so
    this package has no import-path dependence on cscd's internals."""
    if n <= 4:
        return float("nan")
    return float(np.tanh(2.802 / np.sqrt(n - 3)))


def nats_to_r(nats):
    """Equivalent Gaussian-channel correlation for a given mutual information
    (nats): I = -0.5*ln(1-r^2)  =>  r = sqrt(1 - exp(-2I))."""
    nats = max(float(nats), 0.0)
    return float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * nats))))


def r_to_nats(r):
    r = float(np.clip(abs(r), 0.0, 0.999999))
    return float(-0.5 * np.log(1.0 - r * r))


# Empirically measured coupling ceilings (nats), per the brief and re-checked
# against this build's own loaders in `substrates/*.measure_ceiling_nats`.
# ccRCC/HEST ~5.5 nats and NCEP/climate ~2.6-2.7 nats are the two numbers the
# brief hands us directly; BMMC has no prior number, so its ceiling is left
# to be measured rather than guessed (see substrates/bmmc.py).
KNOWN_CEILINGS_NATS = {
    # climate: measured via out-of-sample (year-block held out) multi-dim
    # canonical correlation between real A and real B_real, summed as nats
    # across positive components. Matches the brief's 2.6-2.7 prior closely.
    "climate_ncep": 2.72,
    # hest_ccrcc: ORIGINALLY carried as the brief's 5.5-nats prior, because
    # no real morphology (image) embeddings existed for this cohort to
    # measure against. They now do (hoptimus/dinov2/gigapath). Measured the
    # SAME way as climate (out-of-sample, patient-block held out instead of
    # year-block), on the full 24-patient cohort:
    #   hoptimus 0.58 nats   gigapath 0.54 nats   dinov2 0.29 nats
    # All three are far below the 5.5 prior, and dinov2 measures roughly
    # half of the other two -- a difference between BACKBONES (dinov2 is a
    # smaller, general-purpose model; hoptimus/gigapath are pathology
    # foundation models), not evidence the tissue itself carries no
    # coupling. Kept here as the single most capable backbone measured
    # (hoptimus); see `hest_ccrcc_by_backbone` below for all three and
    # `substrates/hest.measure_morphology_coupling_ceiling` to reproduce.
    # OLD_PRIOR_5.5_REPLACED_BY_MEASUREMENT.
    "hest_ccrcc": 0.58,
    "hest_ccrcc_by_backbone": {"hoptimus": 0.58, "gigapath": 0.54, "dinov2": 0.29},
    "hest_ccrcc_old_prior_nats": 5.5,
    # hest_kidney_injury: second real substrate, not available at the time
    # of the original build. Measured the same way, full 22-patient cohort:
    #   hoptimus 0.55 nats   gigapath 0.41 nats   dinov2 0.24 nats
    "hest_kidney_injury": 0.55,
    "hest_kidney_injury_by_backbone": {"hoptimus": 0.55, "gigapath": 0.41, "dinov2": 0.24},
    # lhco2020: particle physics, added with the Tier-II physics/materials
    # extension. NO PRIOR EXISTED for this substrate -- the brief supplies a
    # ceiling for climate (2.6-2.7) and HEST (5.5, since replaced by
    # measurement) but none here, so this number is a first measurement and
    # is recorded as such. Measured the same way as climate and HEST via
    # `cca_ceiling_nats`, with a whole BLACK BOX held out (the four
    # collections were generated independently), between the two real views
    # of the same events: 14-d jet substructure and the frozen utility
    # baseline's `high_level_12d` event kinematics. 25,000 events per box.
    # Headline is the MEDIAN over the four choices of held-out box.
    #
    # READ THIS NUMBER WITH ITS CAVEAT. It is NOT the same kind of quantity as
    # HEST's 0.58: HEST pairs two independent MEASUREMENT modalities of the
    # same tissue (transcript counts vs a stained image), and climate pairs
    # two physically distinct layers of the atmosphere. LHCO's two views are
    # both deterministic functions of the SAME 700 reconstructed particles,
    # so a high canonical correlation (top oos r ~ 0.90) partly reflects that
    # shared origin rather than a cross-instrument coupling. It is still the
    # right cap for this substrate -- it bounds what an injected object may
    # ask A and B to share here -- but it should not be read as "LHCO is
    # three times as coupled as climate".
    "lhco2020": 1.79,
    "lhco2020_by_held_out_box": {"background": 1.85, "blackbox1": 1.74,
                                 "blackbox2": 1.86, "blackbox3": 1.71},
    "lhco2020_sensitivity_nats_min_max": [1.17, 3.04],
    "lhco2020_prior_nats": None,   # none existed
    # oc20_is2re: materials, same extension, also with NO PRIOR. Held-out
    # block is a whole official split cell (val_id / ood_ads / ood_cat /
    # ood_both) so the OOD axis is never crossed by the fit. A = the frozen
    # utility baseline's `composition_structure_19d` fingerprint,
    # B_real = the energy view (y_init, y_relaxed) -- genuinely 2-dimensional,
    # so the sum runs over at most 2 canonical components and is reported
    # that way rather than padded. 8,000 systems per split.
    #
    # Unlike LHCO's, this pairing IS cross-measurement: `y_relaxed` is a
    # separate DFT relaxation, not a re-summary of the same initial-structure
    # fields the fingerprint reads. The two numbers are therefore not directly
    # rankable against each other, and both carry that note.
    #
    # 0.22 nats is LOW -- the lowest here except BMMC's 0.13 -- and that is a
    # measurement, not a defect: it is the same substrate whose
    # structure->energy Ridge sits at 1.19-1.42 eV MAE against a 1.55-1.78 eV
    # trivial floor. The response is BMMC's: calibration stays floor-based
    # (`floor_multiple` x MDE80), and the cap is checked rather than assumed
    # not to bind -- at strength=3 and n_discovery ~ 16,000 a spike-in sits
    # at ~0.02 nats, an order of magnitude under this ceiling.
    "oc20_is2re": 0.22,
    "oc20_is2re_by_held_out_split": {"val_id": 0.14, "val_ood_ads": 0.31,
                                     "val_ood_cat": 0.12, "val_ood_both": 0.33},
    "oc20_is2re_sensitivity_nats_min_max": [0.11, 0.27],
    "oc20_is2re_prior_nats": None,   # none existed
}


def calibrate_target_r(n, ceiling_nats=None, strength=1.0, floor_multiple=3.0,
                        ceiling_safety=0.9):
    """Target population correlation for a "true" object at this substrate's
    evaluation n, given `strength`.

    strength=1.0 -> floor_multiple x MDE80(n), capped at
    ceiling_safety x (ceiling correlation), if a ceiling is given.
    """
    floor = mde_80(n)
    if not np.isfinite(floor):
        raise ValueError(f"n={n} too small for a detection floor")
    raw = floor_multiple * float(strength) * floor
    if ceiling_nats is not None:
        cap = ceiling_safety * nats_to_r(ceiling_nats)
        return float(np.clip(raw, 0.0, cap))
    return float(np.clip(raw, 0.0, 0.98))


def calibration_summary(substrate_name, n, ceiling_nats=None, strengths=(0.5, 1.0, 2.0)):
    """One dict reporting exactly what `strength` means on this substrate at
    this n -- for the report, not for any test assertion."""
    floor = mde_80(n)
    out = {
        "substrate": substrate_name, "n": n, "detection_floor_mde80": floor,
        "ceiling_nats": ceiling_nats,
        "ceiling_r": nats_to_r(ceiling_nats) if ceiling_nats is not None else None,
    }
    out["target_r_by_strength"] = {
        str(s): calibrate_target_r(n, ceiling_nats, strength=s) for s in strengths
    }
    return out


# --------------------------------------------------------- generic ceiling
def cca_ceiling_nats(A, B, split, n_pca_a=20, n_pca_b=20, k_canon=10, reg=5.0,
                     seed=0, group_labels=None):
    """Out-of-sample multi-dimensional canonical correlation between two REAL
    views of the same rows, with an entity BLOCK held out, summed as nats
    across the positive canonical components.

    This is the estimator `substrates/hest.measure_morphology_coupling_ceiling`
    already uses (year-block held out for climate, patient-block for HEST),
    lifted out verbatim in its arithmetic so the LHCO (black-box held out) and
    OC20 (split-cell held out) numbers are computed the SAME way and are
    therefore comparable to the climate/HEST ones rather than merely
    similar-looking. `hest.py` keeps its own copy untouched -- this function
    is additive, nothing existing was rewritten to call it.

    Fit (PCA on each side, then regularised CCA) uses `split == 0` rows only;
    every reported correlation is measured on `split == 1` rows, which come
    from blocks the fit never saw. `group_labels`, if given, is only carried
    into the report (how many blocks each side had), never into the fit.
    """
    from sklearn.decomposition import PCA

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    split = np.asarray(split)
    disc, held = split == 0, split == 1
    if disc.sum() < 10 or held.sum() < 10:
        raise ValueError(f"need >=10 rows each side, got {disc.sum()}/{held.sum()}")

    na = min(n_pca_a, A.shape[1], int(disc.sum()))
    nb = min(n_pca_b, B.shape[1], int(disc.sum()))
    pa = PCA(n_components=na, random_state=seed).fit(A[disc])
    pb = PCA(n_components=nb, random_state=seed).fit(B[disc])
    A_r, B_r = pa.transform(A), pb.transform(B)

    X, Y = A_r[disc], B_r[disc]
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
    kk = min(k_canon, U.shape[1], Vt.shape[0])
    Wx, Wy = Wx @ U[:, :kk], Wy @ Vt[:kk, :].T

    Zx = (A_r[held] - X.mean(0)) @ Wx
    Zy = (B_r[held] - Y.mean(0)) @ Wy
    rs = []
    for i in range(kk):
        if Zx[:, i].std() < 1e-12 or Zy[:, i].std() < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(Zx[:, i], Zy[:, i])[0, 1]))
    nats_each = [r_to_nats(r) for r in rs if r > 0]
    out = {
        "n_discovery": int(disc.sum()), "n_held_out": int(held.sum()),
        "n_pca_a": int(na), "n_pca_b": int(nb), "k_canon": int(kk), "reg": float(reg),
        "oos_canonical_correlations": rs,
        "nats_sum_positive_dims": float(sum(nats_each)),
    }
    if group_labels is not None:
        g = np.asarray(group_labels)
        out["blocks_discovery"] = sorted(str(x) for x in np.unique(g[disc]))
        out["blocks_held_out"] = sorted(str(x) for x in np.unique(g[held]))
    return out


def ceiling_sensitivity(A, B, split, n_pca_grid=(5, 10, 20), reg_grid=(0.1, 1.0, 5.0),
                        k_canon=10, seed=0, group_labels=None):
    """`cca_ceiling_nats` swept over `n_pca` and regularisation, the same
    sensitivity check the HEST agent ran -- reported so a ceiling is never a
    single number from a single hyper-parameter choice."""
    rows = []
    for npc in n_pca_grid:
        for reg in reg_grid:
            try:
                r = cca_ceiling_nats(A, B, split, n_pca_a=npc, n_pca_b=npc,
                                     k_canon=k_canon, reg=reg, seed=seed,
                                     group_labels=group_labels)
            except ValueError as exc:  # pragma: no cover - guarded by caller
                rows.append({"n_pca": npc, "reg": reg, "error": str(exc)})
                continue
            rows.append({"n_pca": r["n_pca_a"], "reg": reg,
                         "nats": r["nats_sum_positive_dims"],
                         "top_r": max(r["oos_canonical_correlations"])})
    finite = [r["nats"] for r in rows if "nats" in r]
    return {"grid": rows,
            "nats_min": float(min(finite)) if finite else float("nan"),
            "nats_max": float(max(finite)) if finite else float("nan"),
            "nats_median": float(np.median(finite)) if finite else float("nan")}
