#!/usr/bin/env python3
"""External spatial replication of frozen kidney relations (Agent E).

Applies `dbench/protocols/ccrcc_external_replication_v01.json` (Agent F, hashed
before any external value was read) to every frozen hypothesis of a registered
pooled funnel (`funnel_aggregate.json` + ledger) on the converted external
ccRCC cohorts, loaded through `hest_multi`:

* identity: the same gene symbols / programme gene lists (the set's genes that
  entered the discovery computation, i.e. present in the discovery panel), the
  same neighbourhood (k = 6 / 18 nearest spots within section, KD-tree), the
  same target; nothing re-selected.
* arithmetic: log1p CPM at 1e4 over the section panel, programme = mean over
  its genes, features within-patient centred, response within-patient
  standardised; FFPE sources: every variable, competitor column and the target
  replaced by within-section rank normal scores Phi^-1((r - 0.5) / n) (and the
  discovery sign recomputed under the same transform).
* not transferable: a missing gene; a programme missing any discovery gene;
  morphology / H-Optimus constituents or targets unless the image is
  <= 0.5 um/px; columns fitted on the discovery cohort without a frozen
  definition in the ledger (fold-fitted PCA, NMF, scGPT / SCimilarity
  coordinates).
* competitor: log1p depth + the 41 Gavish + 50 Hallmark own-spot scores of the
  external cohort, minus the target programme, its neighbour mean and the
  candidate's own variables; near-constant columns dropped; leak guard
  (within-patient |r| >= 0.9 to the target on the proposal patients) per fold.
* engine: `general_discovery.FittedRelation` (model "rbf", 24 features,
  VALIDATION_ALPHA_GRID, patient-grouped inner CV) with the relation of record;
  patient-grouped cross-fitting, K = min(5, n patients), default_rng(20261016);
  pooled out-of-fold gain, 1000-draw patient bootstrap CI, 199-draw
  within-patient permutation p, sign agreement. Sources with < 4 patients:
  leave-one-section-out folds and a within-section 6 x 6 block bootstrap,
  labelled section-level and never pooled with patient-level sources.

Outputs `runs/development/external_replication_v01/<run>/<cohort>/` (one JSON
receipt per relation + table.json), `<run>/summary.json` and
`<run>/REPLICATION.md`. Nothing is written into the registered run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
PROTOCOL = "dbench/protocols/ccrcc_external_replication_v01.json"
PROTOCOL_HASH_FILE = "runs/development/sprint_v06/ccrcc_external_replication_sha256.txt"
OUT_ROOT = "runs/development/external_replication_v01"
# protocol version -> (rule file, hash file, receipt root)
PROTOCOLS = {
    "v01": (PROTOCOL, PROTOCOL_HASH_FILE, OUT_ROOT),
    "v02": ("dbench/protocols/ccrcc_external_replication_v02.json",
            "runs/development/sprint_v06/ccrcc_external_replication_v02_sha256.txt",
            "runs/development/external_replication_v02"),
}
DEFINITIONS_ROOT = "runs/development/external_replication_v01/definitions"
MATCH_TOL, MATCH_TOL_WIDE, MATCH_MAX_PROPOSALS, N_MATCHED = 0.05, 0.10, 20000, 199
LINEAR_TOL = 1e-6
SPLIT_SEED = 20261016
N_BOOT, N_PERM = 1000, 199
MAX_IMAGE_UM_PER_PX = 0.5
# source key (protocol) -> converted directory (convert_external_visium.py) and chemistry class
SOURCES = {
    "li2022_ccrcc": (f"{STORE}/spatial/external_ccrcc_20261016/mendeley_li2022", "fresh_frozen_whole_transcriptome"),
    "gse250163": (f"{STORE}/spatial/external_ccrcc_20261016/geo", "ffpe_probe"),
    "zenodo14620362": (f"{STORE}/spatial/external_ccrcc_20261016/zenodo", "ffpe_probe"),
}
MORPH_TARGET = re.compile(r"^(morph_pc\d+|nuc_.*|cv_.*|concept.*)$")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_hash(payload) -> str:  # dbench.prospective_campaign.canonical_hash
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


def verify_protocol(path: str = PROTOCOL, hash_file: str = PROTOCOL_HASH_FILE) -> dict:
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    recorded = Path(hash_file).read_text().split()[0]
    canon = canonical_hash(doc)
    if canon != recorded:
        raise SystemExit(f"protocol canonical hash {canon} != recorded {recorded}: refusing to read external values")
    return {"path": path, "raw_sha256": hashlib.sha256(raw).hexdigest(), "canonical_sha256": canon,
            "recorded_canonical_sha256": recorded, "verified": True, "doc": doc}


# ----------------------------------------------------------------------------- parsing
def parse_variable(name: str, allow_linear: bool = False) -> dict:
    """Frozen variable name -> definition, or a not-transferable reason."""
    nbr = 0
    body = name
    m = re.match(r"^context_neighbour_mean:(nbr18|nbr):(.*)$", name)
    if m:
        nbr, body = (18 if m.group(1) == "nbr18" else 6), m.group(2)
    if body.startswith("raw_genes:"):
        return {"name": name, "kind": "gene", "gene": body.split(":", 1)[1], "nbr": nbr}
    for pre, setkind in (("programmes:gavish:", "mp"), ("programmes:hallmark:", "hallmark"),
                         ("gavish:", "mp"), ("hallmark:", "hallmark")):
        if body.startswith(pre):
            return {"name": name, "kind": "programme", "set": f"{setkind}:{body[len(pre):]}", "nbr": nbr}
    if re.match(r"^(pca:)?PC\d+", body) or body.startswith("pca:"):
        pc = body.split(":")[-1]
        if allow_linear:
            return {"name": name, "kind": "linear", "column": ("nbr:" if nbr == 6 else "pca:") + pc, "nbr": 0}
        why = "discovery-fitted PCA column (fold-fitted loadings; no frozen loading vector in the ledger)"
    elif body.split(":")[0] in ("scgpt", "scimilarity", "nmf"):
        why = f"{body.split(':')[0]} coordinate fitted on the discovery cohort (no frozen transferable definition)"
    elif body.split(":")[0] in ("hoptimus", "cellvit_structure", "morph"):
        why = "morphology / H-Optimus-0 constituent (image resolution)"
    else:
        why = f"unrecognised variable kind {body.split(':')[0]!r}"
    return {"name": name, "kind": "untransferable", "reason": why, "nbr": nbr}


def parse_target(target: str) -> dict:
    m = re.match(r"^(own|nbr|nbr18):(.+)$", target)
    if m:
        return {"name": target, "kind": "programme", "set": f"mp:{m.group(2)}",
                "nbr": {"own": 0, "nbr": 6, "nbr18": 18}[m.group(1)]}
    m = re.match(r"^programme:(mp|hallmark):(.+)$", target)
    if m:
        return {"name": target, "kind": "programme", "set": f"{m.group(1)}:{m.group(2)}", "nbr": 0}
    if MORPH_TARGET.match(target):
        return {"name": target, "kind": "morphology"}
    return {"name": target, "kind": "untransferable", "reason": f"unrecognised target {target!r}"}


def load_linear_definition(defs_dir: str | None, column: str, passing_folds) -> tuple[dict | None, str]:
    """Frozen definition of a discovery-fitted linear column (freeze_linear_definitions.py). Aligned fits
    (v0.5: every fold aligned to the fold-0 reference fit): the fold-0 fit when the column exists there, else
    the lowest passing fold where it exists. Unaligned fits (v0.4: a column name means a different component
    in each fold): the lowest passing fold."""
    if not defs_dir or not Path(defs_dir).is_dir():
        return None, "no frozen definitions for this campaign"
    fname = column.replace(":", "__") + ".json"
    passing = sorted(int(f) for f in (passing_folds or []))
    aligned = False
    p0 = Path(defs_dir) / "0" / fname
    if p0.is_file():
        r0 = json.loads(p0.read_text()).get("match_abs_r")
        aligned = r0 is not None and r0 == r0  # NaN for unaligned fits
    order = ([0] if aligned else []) + passing + [0, 1, 2, 3, 4]
    for f in dict.fromkeys(order):
        pth = Path(defs_dir) / str(f) / fname
        if pth.is_file():
            d = json.loads(pth.read_text())
            if not d["reproduction"]["max_abs_diff_float32_cast"] <= LINEAR_TOL:
                return None, f"frozen definition (fold {f}) does not reproduce the discovery column to {LINEAR_TOL}"
            d["_path"] = str(pth)
            return d, f"fold {f}"
    return None, f"no frozen definition of {column} in any fold"


def set_genes(sets: dict, key: str) -> list | None:
    kind, name = key.split(":", 1)
    for s in sets[kind]:
        if s["name"] == name:
            return list(s["genes"])
    return None


# ----------------------------------------------------------------------------- arithmetic
def rank_normal_within(X: np.ndarray, sections: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata

    X = np.asarray(X, dtype=np.float64)
    X2 = X.reshape(len(X), -1)
    out = np.empty_like(X2)
    for s in np.unique(sections):
        m = sections == s
        n = int(m.sum())
        r = rankdata(X2[m], method="average", axis=0)
        out[m] = norm.ppf((r - 0.5) / n)
    return out.reshape(X.shape)


class Cohort:
    """One loaded study (hest_multi StudyRaw) with column builders."""

    def __init__(self, key: str, raw, chemistry: str, image_um_per_px: float | None, few: bool):
        self.key, self.raw, self.chemistry, self.image_um = key, raw, chemistry, image_um_per_px
        self.few = few
        self.panel = {g: i for i, g in enumerate(raw.genes)}
        self.patient = np.asarray(raw.patient).astype(str)
        self.section = np.asarray(raw.sample_id).astype(str)
        self.coords = np.asarray(raw.coords, dtype=np.float64)
        self._X = None

    @property
    def logcpm(self):
        if self._X is None:
            self._X = self.raw.logcpm().tocsc()
        return self._X

    def column(self, kind: str, genes: list, nbr: int) -> np.ndarray:
        idx = [self.panel[g] for g in genes]
        v = np.asarray(self.logcpm[:, idx].mean(axis=1)).ravel()
        if nbr:
            from morpheus.tier2.substrates.hest_multi import _neighbour_mean
            v = _neighbour_mean(v, self.raw.neighbours[nbr])
        return v

    def value(self, v: dict, rank: bool = False) -> np.ndarray:
        """Untransformed value of a constituent (gene, programme or frozen linear column)."""
        if v["kind"] != "linear":
            return self.column(v["kind"], v["genes"], v["nbr"])
        d = v["definition"]
        idx = [self.panel[g] for g in d["genes"]]
        x = self.logcpm[:, idx].toarray().astype(np.float32).astype(np.float64)
        if rank:  # chemistry transform first (fresh_frozen_to_ffpe), then the frozen standardisation
            x = rank_normal_within(x, self.section)
        z = (x - np.asarray(d["centre"])) / np.asarray(d["scale"])
        out = z @ np.asarray(d["loadings"])
        if d.get("neighbour_k"):
            from morpheus.tier2.substrates.hest_multi import _neighbour_mean
            out = _neighbour_mean(out, self.raw.neighbours[int(d["neighbour_k"])])
        return out

    def gene_means(self) -> np.ndarray:
        if getattr(self, "_gene_means", None) is None:
            self._gene_means = np.asarray(self.logcpm.mean(axis=0)).ravel()
        return self._gene_means

    def transform(self, X: np.ndarray, *, response: bool, rank: bool) -> np.ndarray:
        from morpheus.tier2.substrates.hest import _within_patient_standardise
        from morpheus.tier2.substrates.hest_multi import _centre_within

        X = np.asarray(X, dtype=np.float64)
        two = X.reshape(len(X), -1)
        if rank:
            out = rank_normal_within(two, self.section)
        elif response:
            out = _within_patient_standardise(two, self.patient)
        else:
            out = _centre_within(two, self.patient)
        return out.reshape(X.shape)


def load_cohort(key: str, data_dir: str, chemistry: str):
    from morpheus.tier2.substrates import hest_multi as hm

    ids = sorted(p[:-5] for p in os.listdir(f"{data_dir}/st") if p.endswith(".h5ad"))
    meta = {i: json.loads(Path(f"{data_dir}/metadata/{i}.json").read_text()) for i in ids}
    spec = hm.StudySpec(key=key, dataset_title=key, role="external_replication", layout="external",
                        sample_ids=tuple(ids), patient_of={i: meta[i]["patient"] for i in ids}, organ="Kidney",
                        st_technology="Visium", platform="visium", embed_roots=(), data_dir=data_dir)
    raw = hm.load_study(spec, cache_root=None)
    ums = [meta[i].get("pixel_size_um") for i in ids if meta[i].get("with_morphology")]
    image_um = max(ums) if ums else None
    n_pat = len(set(spec.patient_of.values()))
    return Cohort(key, raw, chemistry, image_um, few=n_pat < 4), meta


def load_discovery():
    from morpheus.tier2.substrates import hest_multi as hm

    spec = hm.study_specs()["meylan2022_ccrcc"]
    return Cohort("meylan2022_ccrcc", hm.load_study(spec), "fresh_frozen_whole_transcriptome", None, False)


# ----------------------------------------------------------------------------- statistics
def _r2_from_stats(n, sy, syy, sse):
    sst = syy - sy * sy / n
    return np.where(sst > 1e-15, 1.0 - sse / np.where(sst > 1e-15, sst, 1.0), 0.0)


def unit_stats(y, simple, full, units):
    """Per resampling unit: n, sum y, sum y^2, SSE(simple), SSE(full)."""
    keys = np.unique(units)
    S = np.zeros((len(keys), 5))
    for i, u in enumerate(keys):
        m = units == u
        S[i] = (m.sum(), y[m].sum(), (y[m] ** 2).sum(), ((y[m] - simple[m]) ** 2).sum(),
                ((y[m] - full[m]) ** 2).sum())
    return S


def pooled_gain(S):
    t = S.sum(0)
    return float(_r2_from_stats(t[0], t[1], t[2], t[4]) - _r2_from_stats(t[0], t[1], t[2], t[3]))


def bootstrap_ci(S, n_boot, seed, strata=None):
    """Percentile 95% CI of the pooled gain over resampled units (within strata when given)."""
    rng = np.random.default_rng(seed)
    n = len(S)
    gains = np.empty(n_boot)
    for b in range(n_boot):
        if strata is None:
            w = np.bincount(rng.integers(0, n, n), minlength=n)
        else:
            w = np.zeros(n)
            for s in np.unique(strata):
                ix = np.flatnonzero(strata == s)
                w += np.bincount(ix[rng.integers(0, len(ix), len(ix))], minlength=n)
        t = w @ S
        gains[b] = _r2_from_stats(t[0], t[1], t[2], t[4]) - _r2_from_stats(t[0], t[1], t[2], t[3])
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def block_units(coords, sections, grid=6):
    """Within-section 6 x 6 spatial blocks (labels '<section>|<i>,<j>') for the section-level bootstrap."""
    lab = np.empty(len(coords), dtype=object)
    for s in np.unique(sections):
        m = sections == s
        c = coords[m]
        lo, hi = c.min(0), c.max(0) + 1e-9
        ij = np.floor((c - lo) / (hi - lo) * grid).astype(int)
        lab[m] = [f"{s}|{a},{b}" for a, b in ij]
    return lab.astype(str)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def within_sign(x, y, groups):
    from morpheus.tier2.substrates.hest_multi import _centre_within

    xc = _centre_within(np.asarray(x, float).reshape(-1, 1), groups).ravel()
    yc = _centre_within(np.asarray(y, float).reshape(-1, 1), groups).ravel()
    r = float(xc @ yc / (np.sqrt((xc ** 2).sum() * (yc ** 2).sum()) + 1e-12))
    return int(np.sign(r)), r


def evaluate_relation(F, y, B, patient, section, coords, *, feature_form="linear_subspace", relation="v05",
                      few=False, seed=SPLIT_SEED, n_boot=N_BOOT, n_perm=N_PERM, baseline_names=None,
                      supp_draws=0):
    """Cross-fitted held-out gain of the frozen relation on one external cohort.

    F (n,k) transformed features, y (n,) transformed response, B (n,p) transformed competitor."""
    from morpheus.tier2.substrates.hest_multi import _proxy_columns
    from morpheus_encoder.general_discovery import VALIDATION_ALPHA_GRID, Candidate, FittedRelation

    F = np.asarray(F, float).reshape(len(y), -1)
    k = F.shape[1]
    cand = Candidate(variables=tuple(range(k)), feature_form=feature_form, proposal_family="external_replication")
    if few:
        units_fold = section
        folds = [[s] for s in sorted(np.unique(section))]
    else:
        pats = np.array(sorted(np.unique(patient)))
        np.random.default_rng(seed).shuffle(pats)
        K = min(5, len(pats))
        folds = [list(f) for f in np.array_split(pats, K)]
        units_fold = patient
    simple, full = np.zeros(len(y)), np.zeros(len(y))
    fold_rec, fits = [], []
    for f, held in enumerate(folds):
        h = np.isin(units_fold, held)
        p = ~h
        prow = np.flatnonzero(p)
        leak = _proxy_columns(y, B, prow, patient, 0.9) if B.shape[1] else []
        keep = np.setdiff1d(np.arange(B.shape[1]), leak)
        fitted = FittedRelation(cand, F[p], y[p], context=None, baseline=B[p][:, keep], model="rbf",
                                ridge_alpha=1.0, rbf_features=24, groups=patient[p],
                                alpha_grid=VALIDATION_ALPHA_GRID, relation=relation)
        s_, f_ = fitted.predictions(F[h], context=None, baseline=B[h][:, keep])
        simple[h], full[h] = s_, f_
        fits.append((fitted, h, keep))
        fold_rec.append({"fold": f, "held_out": [str(x) for x in held], "n_rows": int(h.sum()),
                         "leak_guard_dropped": [baseline_names[i] if baseline_names else int(i) for i in leak],
                         "alpha_simple": float(getattr(fitted, "alpha_simple", np.nan)),
                         "alpha_full": float(getattr(fitted, "alpha_full", np.nan))})
    units = block_units(coords, section) if few else patient
    S = unit_stats(y, simple, full, units)
    gain = pooled_gain(S)
    strata = np.array([u.split("|")[0] for u in np.unique(units)]) if few else None
    ci = bootstrap_ci(S, n_boot, seed, strata=strata)
    rng = np.random.default_rng(seed + 1)
    null = np.full(max(n_perm, 1), np.nan)
    groups_ix = [np.flatnonzero(patient == q) for q in np.unique(patient)]
    sst = None
    for d in range(n_perm):
        yp = y.copy()
        for ix in groups_ix:
            yp[ix] = y[rng.permutation(ix)]
        Sp = unit_stats(yp, simple, full, units)
        null[d] = pooled_gain(Sp)
    pval = float((1 + np.sum(null >= gain)) / (n_perm + 1)) if n_perm else None
    # Supplementary (not part of the frozen pass rule), computed only when asked: within-patient permutation
    # of the frozen FEATURE on all rows with the models REFIT on each fold's proposal rows (ridge strengths fixed
    # at the observed fit's). With fixed predictions, permuting the response leaves y . prediction ~ 0, so the
    # protocol null gain collapses to the deterministic (var(simple) - var(full)) / SST offset and its p is
    # 0.005 or 1.000 almost always (the arithmetic of the discovery packet's within_patient_perm).
    pval_f, null_f = None, np.array([np.nan])
    if supp_draws:
        rng_f = np.random.default_rng(seed + 2)
        null_f = np.empty(supp_draws)
        for d in range(supp_draws):
            Fp = F.copy()
            for ix in groups_ix:
                Fp[ix] = F[rng_f.permutation(ix)]
            sp_, fp_ = np.zeros(len(y)), np.zeros(len(y))
            for fitted, h, keep in fits:
                p_ = ~h
                ref = FittedRelation(cand, Fp[p_], y[p_], context=None, baseline=B[p_][:, keep], model="rbf",
                                     ridge_alpha=1.0, rbf_features=24, groups=patient[p_], relation=relation,
                                     fixed_alphas=(fitted.alpha_simple, fitted.alpha_full))
                sp_[h], fp_[h] = ref.predictions(Fp[h], context=None, baseline=B[h][:, keep])
            null_f[d] = pooled_gain(unit_stats(y, sp_, fp_, units))
        pval_f = float((1 + np.sum(null_f >= gain)) / (supp_draws + 1))
    return {"gain": gain, "gain_ci": ci, "perm_p": pval, "n_perm": n_perm, "n_boot": n_boot,
            "perm_null_sd": float(np.nanstd(null)) if n_perm else None,
            "perm_null_mean": float(np.nanmean(null)) if n_perm else None,
            "_fits": fits, "_units": units, "_cand": cand,
            "supplementary_refit_feature_perm_p": pval_f, "supplementary_draws": int(supp_draws),
            "supplementary_null_mean": float(np.nanmean(null_f)) if supp_draws else None,
            "level": "section" if few else "patient", "folds": fold_rec, "n_rows": int(len(y)),
            "r2_simple": float(_r2_from_stats(*S.sum(0)[:3], S.sum(0)[3])),
            "r2_full": float(_r2_from_stats(*S.sum(0)[:3], S.sum(0)[4]))}


# ----------------------------------------------------------------------------- matched null (v02)
def _median_moran(values, coh) -> float:
    from morpheus.tier2.substrates.hest_multi import section_moran_i

    m = section_moran_i(values, coh.raw.neighbours[6], coh.section)
    return float(np.median(list(m.values()))) if m else float("nan")


def matched_programmes(coh, v: dict, *, seed: int, n_accept=N_MATCHED, tol=MATCH_TOL, tol_wide=MATCH_TOL_WIDE,
                       max_proposals=MATCH_MAX_PROPOSALS, batch=200) -> dict:
    """Random gene programmes for one gene-defined constituent: as many genes as the constituent, each drawn
    from the constituent gene's mean-expression decile of the external panel (own genes excluded), scored
    as mean log1p-CPM (measurement_bank.score_gene_sets arithmetic), given the constituent's neighbour
    transform, accepted when the median within-section Moran's I (hest_multi.section_moran_i, k = 6) is
    within `tol` of the constituent's (`tol_wide` if fewer than `n_accept` are accepted in
    `max_proposals`). Returns untransformed values (n, n_accept) and the record."""
    import scipy.sparse as sp

    from morpheus.tier2.substrates.hest_multi import _neighbour_mean

    own = [coh.panel[g] for g in (v["definition"]["genes"] if v["kind"] == "linear" else v["genes"])]
    nbr_k = int(v["definition"].get("neighbour_k", 0)) if v["kind"] == "linear" else int(v["nbr"])
    target_I = _median_moran(coh.value(v), coh)
    means = coh.gene_means()
    edges = np.quantile(means, np.linspace(0, 1, 11)[1:-1])
    dec = np.searchsorted(edges, means, side="right")
    own_set = set(own)
    pools = {d: np.array([g for g in np.flatnonzero(dec == d) if g not in own_set]) for d in np.unique(dec[own])}
    need = {d: int(np.sum(dec[own] == d)) for d in pools}
    m = len(own)
    rng = np.random.default_rng(seed)
    acc, acc_w, n_prop = [], [], 0
    X = coh.logcpm.tocsr() if not sp.isspmatrix_csr(coh.logcpm) else coh.logcpm
    while n_prop < max_proposals and len(acc) < n_accept:
        b = min(batch, max_proposals - n_prop)
        rows, cols = [], []
        for j in range(b):
            gs = np.concatenate([rng.choice(pools[d], size=need[d], replace=need[d] > len(pools[d]))
                                 for d in pools])
            rows.append(gs)
            cols.append(np.full(len(gs), j))
        M = sp.csr_matrix((np.full(m * b, 1.0 / m), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(X.shape[1], b))
        S = np.asarray((X @ M).todense(), dtype=np.float64)
        if nbr_k:
            S = _neighbour_mean(S, coh.raw.neighbours[nbr_k])
        for j in range(b):
            I = _median_moran(S[:, j], coh)
            if abs(I - target_I) <= tol and len(acc) < n_accept:
                acc.append(S[:, j].astype(np.float32))
            elif abs(I - target_I) <= tol_wide and len(acc_w) < n_accept:
                acc_w.append(S[:, j].astype(np.float32))
        n_prop += b
    rec = {"constituent": v["name"], "n_genes": m, "target_median_moran": target_I, "n_proposals": n_prop,
           "n_accepted_tol": len(acc), "tol": tol, "neighbour_k": nbr_k}
    if len(acc) >= n_accept:
        return {"values": np.column_stack(acc[:n_accept]), **rec, "tol_used": tol}
    pool = acc + acc_w
    if len(pool) >= n_accept:
        return {"values": np.column_stack(pool[:n_accept]), **rec, "tol_used": tol_wide,
                "n_accepted_wide": len(pool)}
    return {"values": None, **rec, "tol_used": None, "n_accepted_wide": len(pool),
            "status": "not computable (too few Moran-matched programmes)"}


def matched_null(coh, tr, F, y, B, res, *, rank: bool, relation: str, seed: int, n_draws=N_MATCHED) -> dict:
    """p of the observed gain against Moran-matched random programmes replacing every gene-defined
    constituent; both models refit on each fold's proposal rows per draw (ridge strengths held at the
    observed fit's inner-CV choice)."""
    from morpheus_encoder.general_discovery import FittedRelation

    draws, recs = [], []
    for j, v in enumerate(tr["variables"]):
        mp = matched_programmes(coh, v, seed=seed + 101 * j, n_accept=n_draws)
        recs.append({k: val for k, val in mp.items() if k != "values"})
        if mp["values"] is None:
            return {"matched_p": None, "status": mp["status"], "constituents": recs}
        draws.append(coh.transform(np.asarray(mp["values"], np.float64), response=False, rank=rank))
    gains = np.empty(n_draws)
    for d in range(n_draws):
        Fd = F.copy()
        for j in range(len(draws)):
            Fd[:, j] = draws[j][:, d]
        s_, f_ = np.zeros(len(y)), np.zeros(len(y))
        for fitted, h, keep in res["_fits"]:
            p_ = ~h
            ref = FittedRelation(res["_cand"], Fd[p_], y[p_], context=None, baseline=B[p_][:, keep],
                                 model="rbf", ridge_alpha=1.0, rbf_features=24, groups=coh.patient[p_],
                                 relation=relation, fixed_alphas=(fitted.alpha_simple, fitted.alpha_full))
            s_[h], f_[h] = ref.predictions(Fd[h], context=None, baseline=B[h][:, keep])
        gains[d] = pooled_gain(unit_stats(y, s_, f_, res["_units"]))
    return {"matched_p": float((1 + np.sum(gains >= res["gain"])) / (n_draws + 1)), "status": "computed",
            "n_draws": n_draws, "null_mean": float(gains.mean()), "null_q95": float(np.quantile(gains, 0.95)),
            "constituents": recs}


# ----------------------------------------------------------------------------- hypotheses
def load_hypotheses(run_dir: str) -> list[dict]:
    run = Path(run_dir)
    agg = json.loads((run / "funnel_aggregate.json").read_text())
    ledger = {}
    for p in run.glob("*_ledger.jsonl"):
        for ln in p.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                ledger[e["candidate_id"]] = e
    default_rel = "v05" if "-v05-" in agg["campaign_id"] else ("v04" if "-v04-" in agg["campaign_id"] else None)
    out = []
    for r in agg["table"]:
        e = ledger.get(r["candidate_id"], {})
        rel = _relation_of_record(run, e, r) or default_rel
        out.append({**r, "relation": rel, "object_hash": e.get("object_hash"),
                    "ledger_stage": e.get("stage")})
    return out


def _relation_of_record(run: Path, entry: dict, row: dict) -> str | None:
    for ref in str(entry.get("data_ref") or "").split(";"):
        p = run.parent.parent.parent / ref if ref.startswith("runs/") else Path(ref)
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for c in d.get("candidates", []):
            if c.get("variable_names") == row["variable_names"] and c.get("relation"):
                return str(c["relation"])
    return None


def transferability(h: dict, sets: dict, disc: Cohort, coh: Cohort, defs_dir: str | None = None,
                     allow_linear: bool = False) -> dict:
    vars_ = [parse_variable(v, allow_linear=allow_linear) for v in h["variable_names"]]
    tgt = parse_target(h["target"])
    missing, reasons = [], []
    if tgt["kind"] == "morphology":
        if coh.image_um is None or coh.image_um > MAX_IMAGE_UM_PER_PX:
            reasons.append("morphology target, image resolution (no full-resolution image at <= 0.5 um/px; "
                           f"best {coh.image_um} um/px)")
        else:
            reasons.append("morphology target: external CellViT / morphology PCs not computed")
    elif tgt["kind"] != "programme":
        reasons.append(tgt["reason"])
    defs = []
    for v in vars_ + ([tgt] if tgt["kind"] == "programme" else []):
        if v["kind"] == "untransferable":
            reasons.append(f"{v['name']}: {v['reason']}")
            continue
        if v["kind"] == "linear":
            d, why = load_linear_definition(defs_dir, v["column"], h.get("passing_folds"))
            if d is None:
                reasons.append(f"{v['name']}: {why}")
                continue
            v = {**v, "definition": d, "definition_source": why, "definition_path": d["_path"]}
            genes = list(d["genes"])
        elif v["kind"] == "gene":
            genes = [v["gene"]]
        elif v["kind"] == "programme":
            g = set_genes(sets, v["set"])
            if g is None:
                reasons.append(f"{v['name']}: programme {v['set']} not found")
                continue
            genes = [x for x in dict.fromkeys(g) if x in disc.panel]  # genes that entered the discovery score
        else:
            continue
        miss = [x for x in genes if x not in coh.panel]
        if miss:
            missing += [f"{v['name']}:{x}" for x in miss]
        defs.append({**v, "genes": genes})
    if missing:
        reasons.append(f"{len(missing)} discovery gene(s) absent from the external panel")
    return {"transferable": not reasons, "reasons": reasons, "missing_genes": missing,
            "variables": [d for d in defs if d is not tgt and d["name"] != tgt["name"]] if not reasons else [],
            "target": next((d for d in defs if d["name"] == tgt["name"]), None) if not reasons else None}


def build_arrays(coh: Cohort, tr: dict, rank: bool):
    F = np.column_stack([coh.value(v, rank=rank) for v in tr["variables"]])
    y = coh.column("programme", tr["target"]["genes"], tr["target"]["nbr"])
    return coh.transform(F, response=False, rank=rank), coh.transform(y, response=True, rank=rank).ravel()


def competitor(coh: Cohort, tr: dict, rank: bool):
    prog = np.asarray(coh.raw.programmes, float)
    names = [str(n) for n in coh.raw.programme_names]
    tset = tr["target"]["set"]
    own_vars = {v["set"] for v in tr["variables"] if v["kind"] == "programme" and v["nbr"] == 0}
    drop = {tset} | own_vars
    keep = [i for i, n in enumerate(names) if n not in drop and coh.raw.programme_matched[i] > 0]
    B = np.column_stack([np.log1p(np.asarray(coh.raw.depth, float))] + [prog[:, keep]])
    bnames = ["log1p_depth"] + [names[i] for i in keep]
    sd = np.sqrt(np.maximum(0, coh.transform(B, response=False, rank=False).var(0)))
    ok = np.flatnonzero(sd >= 0.01)
    B, bnames = B[:, ok], [bnames[i] for i in ok]
    return coh.transform(B, response=False, rank=rank), bnames, {"excluded": sorted(drop),
                                                                 "near_constant_dropped": int(len(sd) - len(ok))}


def run(run_dir: str, out_root: str | None, sources: list[str], run_name: str | None = None, n_boot=N_BOOT,
        n_perm=N_PERM, supp_draws=99, protocol="v01", n_matched=N_MATCHED) -> dict:
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    ppath, phash, proot = PROTOCOLS[protocol]
    prot = verify_protocol(ppath, phash)
    out_root = out_root or proot
    v02 = protocol == "v02"
    defs_dir = str(Path(DEFINITIONS_ROOT) / Path(run_dir).name)
    hyps = load_hypotheses(run_dir)
    run_name = run_name or Path(run_dir).name
    out = Path(out_root) / run_name
    out.mkdir(parents=True, exist_ok=True)
    sets = _programme_sets()
    disc = load_discovery()
    summary = {"schema": "morpheus_external_replication_result/v1", "created_utc": _utc(),
               "registered_run": str(run_dir), "n_hypotheses": len(hyps),
               "protocol": {k: v for k, v in prot.items() if k != "doc"}, "sources": {}}
    for src in sources:
        data_dir, chem = SOURCES[src]
        coh, meta = load_cohort(src, data_dir, chem)
        rank = chem == "ffpe_probe"
        rows = []
        sdir = out / src
        sdir.mkdir(parents=True, exist_ok=True)
        for h in hyps:
            tr = transferability(h, sets, disc, coh, defs_dir=defs_dir, allow_linear=v02)
            rec = {"candidate_id": h["candidate_id"], "object_hash": h.get("object_hash"), "family": h["family"],
                   "target": h["target"], "variable_names": h["variable_names"], "discovery_stage": h["stage"],
                   "relation": h["relation"], "source": src, "chemistry_class": chem,
                   "n_sections": int(len(set(coh.section))), "n_patients": int(len(set(coh.patient))),
                   "image_um_per_px": coh.image_um, "transferable": tr["transferable"],
                   "not_transferable_reasons": tr["reasons"], "missing_genes": tr["missing_genes"],
                   "linear_definitions": {v["name"]: {"path": v["definition_path"], "source": v["definition_source"],
                                                      "reproduction": v["definition"]["reproduction"]}
                                          for v in tr["variables"] if v["kind"] == "linear"}}
            if tr["transferable"]:
                F, y = build_arrays(coh, tr, rank)
                B, bnames, cinfo = competitor(coh, tr, rank)
                Fd, yd = build_arrays(disc, tr, rank)
                feat_d = Fd.sum(1) if Fd.ndim > 1 else Fd
                feat_e = F.sum(1) if F.ndim > 1 else F
                sd_, rd = within_sign(feat_d, yd, disc.patient)
                se_, re_ = within_sign(feat_e, y, coh.patient)
                res = evaluate_relation(F, y, B, coh.patient, coh.section, coh.coords,
                                        feature_form=h.get("feature_form", "linear_subspace"),
                                        relation=h["relation"], few=coh.few, n_boot=n_boot,
                                        baseline_names=bnames,
                                        supp_draws=(supp_draws if sd_ == se_ else 0) if not v02 else 0,
                                        n_perm=0 if v02 else n_perm)
                res.update(discovery_sign=sd_, discovery_within_r=rd, external_sign=se_, external_within_r=re_,
                           sign_agree=bool(sd_ == se_ and sd_ != 0), competitor_width=int(B.shape[1]),
                           competitor=cinfo,
                           pass_ci_and_sign=bool(res["gain"] > 0 and res["gain_ci"][0] > 0
                                                 and sd_ == se_ and sd_ != 0))
                if v02:
                    if res["pass_ci_and_sign"]:
                        mn = matched_null(coh, tr, F, y, B, res, rank=rank, relation=h["relation"],
                                          seed=int(h["candidate_id"].split("-")[-1][:8], 16) % (2 ** 31),
                                          n_draws=n_matched)
                    else:
                        mn = {"matched_p": None, "status": "not run (fails the gain CI or sign requirement)"}
                    res["matched_null"] = mn
                    res["passed"] = bool(res["pass_ci_and_sign"] and mn.get("matched_p") is not None
                                         and mn["matched_p"] <= 0.05)
                else:
                    res["passed"] = bool(res["pass_ci_and_sign"] and res["perm_p"] <= 0.05)
                rec.update({k: val for k, val in res.items() if not k.startswith("_")})
            rows.append(rec)
            tag = re.sub(r"[^A-Za-z0-9]+", "_", h["candidate_id"].split("-")[-1])
            (sdir / f"{tag}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        tested = [r for r in rows if r["transferable"]]
        pkey = "matched_p" if v02 else "perm_p"
        withp = [r for r in tested if (r.get("matched_null", {}).get("matched_p") if v02 else r.get("perm_p"))
                 is not None]
        for r, q in zip(withp, bh([(r["matched_null"]["matched_p"] if v02 else r["perm_p"]) for r in withp])):
            r["bh_q"] = float(q)
        table = {"source": src, "level": "section" if coh.few else "patient", "chemistry_class": chem,
                 "n_sections": int(len(set(coh.section))), "n_patients": int(len(set(coh.patient))),
                 "n_spots": int(coh.raw.n), "image_um_per_px": coh.image_um, "n_transferable": len(tested),
                 "n_passed": sum(bool(r.get("passed")) for r in tested),
                 "n_pass_ci_and_sign": sum(bool(r.get("pass_ci_and_sign")) for r in tested), "rows": rows}
        (sdir / "table.json").write_text(json.dumps(table, indent=1, default=str) + "\n")
        summary["sources"][src] = {k: v for k, v in table.items() if k != "rows"}
        summary["sources"][src]["rows"] = rows
    summary["protocol_version"] = protocol
    summary["across_sources"] = across_sources(hyps, summary["sources"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out / "REPLICATION.md").write_text(render_md(summary))
    return summary


def across_sources(hyps, srcs) -> list[dict]:
    out = []
    for h in hyps:
        per = {}
        for s, t in srcs.items():
            r = next(x for x in t["rows"] if x["candidate_id"] == h["candidate_id"])
            per[s] = ("not transferable" if not r["transferable"] else
                      ("pass" if r.get("passed") else "fail")) + (" (section-level)" if t["level"] == "section" else "")
        patient_level = [s for s, t in srcs.items() if t["level"] == "patient"]
        rev = [s for s in patient_level
               if next(x for x in srcs[s]["rows"] if x["candidate_id"] == h["candidate_id"]).get("sign_agree") is False]
        out.append({"candidate_id": h["candidate_id"], "target": h["target"], "variable_names": h["variable_names"],
                    "discovery_stage": h["stage"], "per_source": per,
                    "externally_replicated": any(per[s] == "pass" for s in patient_level),
                    "sign_reversal_in": rev})
    return out


def _fmt(r):
    if not r["transferable"]:
        return "not transferable"
    if "matched_null" in r:
        mn = r["matched_null"]
        mp = (f"matched p={mn['matched_p']:.3f}" if mn.get("matched_p") is not None
              else ("matched null not computable" if "computable" in mn.get("status", "") else "matched null not run"))
        return (f"{r['gain']:+.4f} [{r['gain_ci'][0]:+.4f}, {r['gain_ci'][1]:+.4f}] {mp} "
                f"sign {'=' if r['sign_agree'] else 'x'} {'PASS' if r['passed'] else 'fail'}")
    return (f"{r['gain']:+.4f} [{r['gain_ci'][0]:+.4f}, {r['gain_ci'][1]:+.4f}] p={r['perm_p']:.3f} "
            + (f"(refit feat-perm p={r['supplementary_refit_feature_perm_p']:.3f}) "
               if r.get("supplementary_refit_feature_perm_p") is not None else "")
            + f"sign {'=' if r['sign_agree'] else 'x'} "
            f"{'PASS' if r['passed'] else 'fail'}")


def render_md(s) -> str:
    L = [f"# External spatial replication: {Path(s['registered_run']).name}", "",
         f"Protocol `{s['protocol']['path']}`: canonical sha256 `{s['protocol']['canonical_sha256']}` "
         f"(matches the recorded hash; raw file sha256 `{s['protocol']['raw_sha256']}`). Created {s['created_utc']}.",
         f"{s['n_hypotheses']} frozen hypotheses from the registered pool.", ""]
    L += ["| source | level | chemistry | sections | patients | spots | image um/px | transferable | "
          "pass CI + sign | passed (full rule) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in s["sources"].items():
        L.append(f"| {k} | {t['level']} | {t['chemistry_class']} | {t['n_sections']} | {t['n_patients']} | "
                 f"{t['n_spots']} | {t['image_um_per_px']} | {t['n_transferable']} | "
                 f"{t.get('n_pass_ci_and_sign', '')} | {t['n_passed']} |")
    L += ["", "## Transferable relations", "",
          "| hypothesis | target | variables | " + " | ".join(s["sources"]) + " |",
          "|---|---|---|" + "---|" * len(s["sources"])]
    for a in s["across_sources"]:
        cells = []
        any_t = False
        for k, t in s["sources"].items():
            r = next(x for x in t["rows"] if x["candidate_id"] == a["candidate_id"])
            any_t |= r["transferable"]
            cells.append(_fmt(r))
        if any_t:
            L.append(f"| {a['candidate_id'].split('-')[-1]} ({a['discovery_stage']}) | {a['target']} | "
                     f"{', '.join(a['variable_names'])} | " + " | ".join(cells) + " |")
    L += ["", "## Not transferable anywhere (reason in the first source)", ""]
    from collections import Counter
    c = Counter()
    for a in s["across_sources"]:
        rs = [next(x for x in t["rows"] if x["candidate_id"] == a["candidate_id"]) for t in s["sources"].values()]
        if not any(r["transferable"] for r in rs):
            why = rs[0]["not_transferable_reasons"][0] if rs[0]["not_transferable_reasons"] else "?"
            c[why[:140]] += 1
    for why, n in c.most_common():
        L.append(f"- {n} x {why}")
    if s.get("protocol_version") == "v02":
        L += ["", "Matched null (v02): every gene-defined constituent replaced by a random gene programme of the same "
              "size (decile-matched mean expression, own genes excluded) whose median within-section Moran's I is "
              "within 0.05 (0.10 fallback) of the constituent's; both models refit on every fold's proposal patients "
              "per draw with the ridge strengths of the observed fit; 199 draws. Computed only where the gain CI "
              "and the sign already pass (all three are required)."]
    elif True:
        L += [""]
    L += ["", "## Note on the permutation p (v01)", "",
          "The frozen rule permutes the response within patient on the held-out rows with the fitted models fixed "
          "(the discovery packet's `within_patient_perm`). With fixed predictions y . prediction ~ 0 after permutation, "
          "so every null gain is close to the deterministic offset (var(competitor pred) - var(full pred)) / SST and "
          "the protocol p is 0.005 or 1.000 for almost every relation: it tests the observed gain against that offset, "
          "not against a chance association. Where the sign agrees, the table also shows a supplementary p "
          "(`refit feat-perm`: the frozen feature permuted within patient on all rows, both models refit on each "
          "fold's proposal patients at the observed ridge strengths, 99 draws). It is "
          "not part of the frozen pass rule; changing the rule needs a new protocol id (Agent F).", ""]
    supp = []
    for k, t in s["sources"].items():
        for r in t["rows"]:
            if r["transferable"] and r["gain"] > 0 and r["gain_ci"][0] > 0 and r["sign_agree"] \
                    and (r.get("supplementary_refit_feature_perm_p") or 1.0) <= 0.05 and "matched_null" not in r:
                supp.append(f"{r['candidate_id'].split('-')[-1]} ({k}{', section-level' if t['level'] == 'section' else ''})")
    L.append(f"Pass with the supplementary feature-permutation p in place of the protocol p: {len(supp)}"
             + (": " + ", ".join(supp) if supp else ""))
    rep = [a for a in s["across_sources"] if a["externally_replicated"]]
    rev = [a for a in s["across_sources"] if a["sign_reversal_in"]]
    L += ["", f"Externally replicated (pass in >= 1 patient-level source): {len(rep)}"
          + (": " + ", ".join(a["candidate_id"].split("-")[-1] for a in rep) if rep else ""),
          f"Sign reversal in a patient-level source: {len(rev)}"
          + (": " + ", ".join(f"{a['candidate_id'].split('-')[-1]} ({', '.join(a['sign_reversal_in'])})" for a in rev)
             if rev else ""), ""]
    return "\n".join(L) + "\n"


def plant_check(run_dir: str, out_root: str, source: str, beta: float = 0.5, run_name: str | None = None,
                protocol: str = "v01", n_matched: int = N_MATCHED) -> dict:
    """Known answer on a real external cohort. The first transferable relation's feature (its
    competitor-orthogonal part) is copied into the response after the response's own association with it is
    removed: with the discovery sign it must pass, sign-flipped it must fail. Under v02 a third case: a
    Moran-matched DECOY (a random gene programme drawn exactly as the matched null draws them, smooth like the
    feature but unrelated to the response) evaluated as the relation's feature must fail the matched null.
    Engine: relation v05 (single variable: competitor = baseline only)."""
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    ppath, phash, _ = PROTOCOLS[protocol]
    verify_protocol(ppath, phash)
    v02 = protocol == "v02"
    hyps = load_hypotheses(run_dir)
    sets, disc = _programme_sets(), load_discovery()
    data_dir, chem = SOURCES[source]
    coh, _ = load_cohort(source, data_dir, chem)
    rank = chem == "ffpe_probe"
    defs_dir = str(Path(DEFINITIONS_ROOT) / Path(run_dir).name)
    for h in hyps:
        tr = transferability(h, sets, disc, coh, defs_dir=defs_dir, allow_linear=v02)
        if tr["transferable"] and all(v["kind"] != "linear" for v in tr["variables"]):
            break
    else:
        raise SystemExit(f"no transferable relation in {source}")
    F, y = build_arrays(coh, tr, rank)
    B, bnames, _ = competitor(coh, tr, rank)
    Fd, yd = build_arrays(disc, tr, rank)
    sd_, _ = within_sign(Fd.sum(1), yd, disc.patient)
    f = F.sum(1)
    Bc = np.column_stack([np.ones(len(f)), B])
    fr = f - Bc @ np.linalg.lstsq(Bc, f, rcond=None)[0]
    fz = (fr - fr.mean()) / (fr.std() + 1e-12)
    A = np.column_stack([np.ones(len(f)), f, fz])  # the response keeps no share of f or of fz
    yres = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    seed = int(h["candidate_id"].split("-")[-1][:8], 16) % (2 ** 31)
    out = {"source": source, "protocol": protocol, "candidate_id": h["candidate_id"], "discovery_sign": sd_,
           "beta": beta, "relation_used": "v05", "cases": {}}

    def _score(Fx, yx, expect_sign):
        res = evaluate_relation(Fx, yx, B, coh.patient, coh.section, coh.coords, relation="v05", few=coh.few,
                                baseline_names=bnames, n_perm=0 if v02 else N_PERM,
                                supp_draws=0 if v02 else 19)
        se_, _ = within_sign(Fx.sum(1), yx, coh.patient)
        ci_sign = bool(res["gain"] > 0 and res["gain_ci"][0] > 0 and se_ == expect_sign)
        row = {"gain": res["gain"], "gain_ci": res["gain_ci"], "external_sign": se_,
               "sign_agree": bool(se_ == expect_sign), "pass_ci_and_sign": ci_sign, "level": res["level"]}
        if v02:
            trx = {**tr, "variables": [dict(tr["variables"][0])]}
            mn = matched_null(coh, trx, Fx, yx, B, res, rank=rank, relation="v05", seed=seed + 7,
                              n_draws=n_matched) if ci_sign else {"matched_p": None, "status": "not run"}
            row["matched_null"] = {k: v for k, v in mn.items() if k != "constituents"}
            row["passed"] = bool(ci_sign and mn.get("matched_p") is not None and mn["matched_p"] <= 0.05)
        else:
            row["perm_p"] = res["perm_p"]
            row["passed"] = bool(ci_sign and res["perm_p"] <= 0.05)
        return row

    for case, sgn in (("copied_with_discovery_sign", sd_), ("copied_sign_flipped", -sd_)):
        out["cases"][case] = _score(F, yres + sgn * beta * fz, sd_)
    ok = out["cases"]["copied_with_discovery_sign"]["passed"] and not out["cases"]["copied_sign_flipped"]["passed"]
    if v02:
        # decoy: one Moran-matched random programme for the constituent, unrelated to the response
        mp = matched_programmes(coh, tr["variables"][0], seed=seed + 999, n_accept=1)
        if mp["values"] is None:
            out["cases"]["moran_matched_decoy"] = {"status": mp.get("status")}
            ok = False
        else:
            dec = coh.transform(np.asarray(mp["values"][:, :1], np.float64), response=False, rank=rank)
            ydec = y  # the real response: the decoy has no planted share in it
            sdec = within_sign(dec.ravel(), ydec, coh.patient)[0]
            row = _score(dec, ydec, sdec)  # sign set to the decoy's own, so only gain + matched null can stop it
            row["decoy_median_moran"] = _median_moran(np.asarray(mp["values"][:, 0], float), coh)
            row["feature_median_moran"] = mp["target_median_moran"]
            out["cases"]["moran_matched_decoy"] = row
            ok = ok and not row["passed"]
    out["known_answer_ok"] = bool(ok)
    d = Path(out_root) / (run_name or Path(run_dir).name) / source
    d.mkdir(parents=True, exist_ok=True)
    (d / "planted_known_answer.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="registered campaign directory (read only)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--protocol", default="v01", choices=sorted(PROTOCOLS))
    ap.add_argument("--n-matched", type=int, default=N_MATCHED)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--supp-draws", type=int, default=99, help="refit feature-permutation draws (supplementary)")
    ap.add_argument("--plant-check", default="", help="run the planted known answer on this source only")
    a = ap.parse_args()
    out_root = a.out_root or PROTOCOLS[a.protocol][2]
    if Path(out_root).resolve().is_relative_to(Path(a.run_dir).resolve()) or "runs/prospective" in str(
            Path(out_root).resolve()):
        raise SystemExit("refusing to write under a registered run directory")
    if a.plant_check:
        print(json.dumps(plant_check(a.run_dir, out_root, a.plant_check, protocol=a.protocol,
                                     n_matched=a.n_matched), indent=1, default=str))
        return 0
    s = run(a.run_dir, out_root, a.sources.split(","), n_boot=a.n_boot, n_perm=a.n_perm,
              supp_draws=a.supp_draws, protocol=a.protocol, n_matched=a.n_matched)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in s["sources"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
