"""PREVIEW downstream pass on a candidate list built before the registered aggregation (preview_candidates.py),
or on the registered internal set later (same code, ``--candidates`` from funnel_aggregate.json).

Writes ONLY under ``--out`` (refuses runs/prospective). No state cards, no literature step, no novelty queries.
Stages (``--stage``; each reads the pickled freeze):

* ``freeze``       relation of record (blind.freeze_of_record, v03_rff, protocol relation_commit) on ALL discovery
                   patients; morphology-dictionary constituents are computed from the frozen H-Optimus-0 spot
                   embeddings as the runner computes the morph_dictionary_v03 view (standardisation on all covered
                   discovery spots); uncovered spots dropped.
* ``replication``  replication v03 (replication_v03.run_family: no refit, FFPE rank-normal, pooled B_nuis CI,
                   dominant-term sign in >= 60% of external patients, BH q <= 0.05) + the same evaluation on one
                   cohort at a time + Moran-matched random-gene null with the frozen relation applied unchanged
                   (reported, never gating; 49 draws in the preview).
* ``characterise`` 1.1-1.6 (constituents, gene decoding + ORA + adjacent programmes + cell-type association,
                   localisation, own-version loss under v03_rff, Squidpy neighbourhood, LIANA+ only when implied).
* ``therapeutic``  DepMap 26Q1 / PRISM 24Q2 / Open Targets for implicated genes + Section 4 prioritisation.
* ``distill``      assay_distillation_v03 state (b); H&E-only ridge / MLP students, held-out spot and patient
                   Spearman; CPTAC slide transfer against the bulk-rebuilt state where it can be rebuilt.
* ``clinical_lock`` prespecification locks only (CPTAC OS, TCGA-KIRC OS and PFS, JAVELIN PFS where a gene signature
                   exists); nothing is fitted.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from dbench.downstream import bknown, locks
from dbench.downstream.blind import freeze_of_record, known_columns_v03

STUDY = "meylan2022_ccrcc"
COHORTS = ["li2022_ccrcc", "gse250163", "zenodo14620362"]
N_MATCHED_PREVIEW = 49
PANEL_OVERRIDE = None
ROW_RESTRICTION = None  # e.g. "normal_appearing": the family's row restriction (architecture-table label)   # {gene: index}; programme constituents scored on these genes only (panel-matched replication)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _json(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def fid_of(c: dict) -> str:
    return c.get("candidate_id") or ("preview-" + locks.canonical_hash(
        [c["family"], c["target"], c["feature_form"], sorted(c["variable_names"])])[:12])


# ----------------------------------------------------------------------------- discovery values
_MORPH: dict = {}


def morph_columns(coh) -> tuple[dict, np.ndarray]:
    """{'morph:kNN': values aligned to coh rows}, covered mask; runner arithmetic, all discovery patients."""
    if _MORPH.get("study") == STUDY:
        return _MORPH["cols"], _MORPH["cov"]
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.morphology_dictionary import (load_dictionary_v03, similarities_v03,
                                                                        standardise_stats)
    spec = hm.study_specs()[STUDY]
    pats = sorted(set(map(str, spec.patient_of.values())))
    ep = hm.study_episode(spec, target="concept:0", support_patients=pats[:-1], query_patients=pats[-1:],
                          seed=0, with_morphology=True)
    dct = load_dictionary_v03()
    E = np.asarray(ep.views["hoptimus"], dtype=np.float32)
    covered = np.asarray(ep.covered, bool)
    mu, sd = standardise_stats(E[covered])
    M = similarities_v03(E, dct["centres"], mu, sd).astype(np.float64)
    idx = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(ep.sample_id, ep.barcode))}
    rows = np.array([idx.get((str(a), str(b)), -1) for a, b in zip(coh.raw.sample_id, coh.raw.barcode)])
    cov = (rows >= 0) & covered[np.maximum(rows, 0)]
    have = rows >= 0   # the runner computes the similarity on every spot (covered spots only set the standardisation)
    cols = {}
    for j, k in enumerate(dct["kept"]):
        v = np.full(len(rows), np.nan)
        v[have] = M[rows[have], j]
        cols[f"morph:k{int(k):02d}"] = v
    _MORPH.update(cols=cols, cov=cov, study=STUDY)
    return cols, cov


def discovery_arrays_any(candidates, target) -> dict:
    """discovery.discovery_arrays + morphology-dictionary constituents + v03 B_known; finite rows only."""
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.downstream.values import cohort_values
    from dbench.downstream.discovery_any import cohort

    coh = cohort(STUDY)
    sets = _programme_sets()
    panel = PANEL_OVERRIDE if PANEL_OVERRIDE is not None else coh.panel
    cols = []
    comp = None
    for c in candidates:
        head, _, rest = c.partition(":")
        body_c = rest if head in ("own", "nbr6", "nbr18", "contrast") else c
        if body_c.startswith("composition:"):   # CellViT PanNuke fraction (the runner's composition view)
            from dbench.downstream import characterise as C_
            from dbench.downstream.values import V07_TRANSFORMS, apply_version
            if comp is None:
                comp = _discovery().cellvit()
            names_ = list(getattr(C_, "PANNUKE", ["Neoplastic", "Connective", "Epithelial", "Inflammatory", "Dead"]))
            v = np.asarray(comp["fractions"][:, names_.index(body_c.split(":", 1)[1])], float)
            v = np.where(comp["covered"], v, np.nan)
            ver = V07_TRANSFORMS[head] if head in ("own", "nbr6", "nbr18", "contrast") else "v0"
            cols.append(apply_version(np.nan_to_num(v), ver, coh.raw.neighbours) if ver != "v0" else v)
            continue
        if c.startswith("morph_dictionary_v03:") or (head in ("own", "nbr6", "nbr18", "contrast")
                                                     and rest.startswith("morph_dictionary_v03:")):
            from dbench.downstream.values import V07_TRANSFORMS, apply_version
            body = c if c.startswith("morph_dictionary_v03:") else rest
            ver = "v0" if body is c else V07_TRANSFORMS[head]
            mc, _ = morph_columns(coh)
            cols.append(apply_version(mc[body.split(":", 1)[1]], ver, coh.raw.neighbours))
        else:
            cols.append(np.asarray(cohort_values(coh, c, sets, panel), float).ravel())
    Z = np.column_stack(cols)
    if target.startswith("morph:k"):
        y = np.asarray(morph_columns(coh)[0][target], float)
    elif target.startswith(("inferred_cnv:", "budding_like:", "gland_architecture:")):   # architecture-table targets
        from morpheus.tier2.substrates import hest_multi as hm
        y = np.asarray(hm.architecture_table(hm.study_specs()[STUDY], coh.raw)[target], float)
    else:
        y = np.asarray(cohort_values(coh, target, sets, panel), float).ravel()
    K, kn, kinfo = known_columns_v03(STUDY, target, coh)
    ok = np.isfinite(Z).all(1) & np.isfinite(y) & np.isfinite(K).all(1)
    if ROW_RESTRICTION:   # the family's row restriction: transforms built on the full tissue, rows restricted
        from morpheus.tier2.substrates import hest_multi as hm
        ok &= np.asarray(hm.architecture_table(hm.study_specs()[STUDY], coh.raw)[ROW_RESTRICTION], bool)
    return {"cohort": coh, "rows": np.flatnonzero(ok), "Z": Z[ok], "y": y[ok],
            "depth": np.asarray(coh.raw.depth, float)[ok], "patient": np.asarray(coh.patient).astype(str)[ok],
            "section": np.asarray(coh.section).astype(str)[ok], "known": K[ok], "known_names": kn,
            "known_info": kinfo}


# ----------------------------------------------------------------------------- freeze
def stage_freeze(cands, out, commit):
    frozen = {}
    for c in cands:
        fid = fid_of(c)
        log("freeze", fid, c["variable_names"], c["target"])
        try:
            d = discovery_arrays_any(c["variable_names"], c["target"])
            ff, aux = freeze_of_record(fid, c["variable_names"], c["target"], d, "v03_rff", c["feature_form"],
                                       {"finding_hash": fid, "preview": True}, commit=commit)
        except Exception as e:   # e.g. a composition variable (not served by this accessor); recorded, skipped
            log("freeze skipped", fid, type(e).__name__, str(e)[:200])
            Path(f"{out}/freeze_skipped.json").write_text(json.dumps(
                {**(json.loads(Path(f"{out}/freeze_skipped.json").read_text()) if Path(f"{out}/freeze_skipped.json").exists() else {}),
                 fid: f"{type(e).__name__}: {e}"}, indent=1))
            continue
        aux.update(rows=d["rows"], known_info=d["known_info"], Z_raw=d["Z"])
        frozen[fid] = {"candidate": c, "ff": ff, "aux": aux}
        Path(f"{out}/frozen_relations").mkdir(parents=True, exist_ok=True)
        Path(f"{out}/frozen_relations/{fid}.json").write_text(json.dumps(_json(ff.rel_nuis.to_json()), indent=1))
    with open(f"{out}/frozen.pkl", "wb") as f:
        pickle.dump(frozen, f)
    return {fid: {"n_rows": int(len(v["aux"]["ys"])), "g_sup_nuis": v["ff"].rel_nuis.g_sup,
                  "g_sup_known": None if v["ff"].rel_known is None else v["ff"].rel_known.g_sup,
                  "repro_max_abs": v["ff"].rel_nuis.repro_max_abs, "dominant": v["ff"].dominant}
            for fid, v in frozen.items()}


def _discovery():
    if STUDY == "meylan2022_ccrcc":
        from dbench.downstream.dryrun import Discovery
        return Discovery()
    from dbench.downstream.discovery_any import DiscoveryAny
    return DiscoveryAny(STUDY)


def _load(out):
    with open(f"{out}/frozen.pkl", "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- replication
def matched_null_pooled(ff, arrays, *, seed=20261023, n=N_MATCHED_PREVIEW) -> dict:
    """Moran-matched random programmes (external_replication.matched_programmes) replace every gene-defined
    constituent in every external cohort; frozen relation applied unchanged; pooled B_nuis gain per draw."""
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.downstream.replication_v03 import pooled_gain, transform_cohort, unit_stats
    from dbench.downstream.values import parse_candidate, set_genes
    from dbench.scripts.external_replication import SOURCES, load_cohort, matched_programmes

    sets = _programme_sets()
    nbr = {"v0": 0, "v6": 6, "v18": 18}
    gene_cols = []
    for j, c in enumerate(ff.rel_nuis.candidates):
        try:
            pc = parse_candidate(c)
        except Exception:
            continue
        if pc["version"] not in nbr or pc["kind"] not in ("gene", "programme"):
            continue
        genes = [pc["genes_key"]] if pc["kind"] == "gene" else set_genes(sets, pc["set"])
        gene_cols.append((j, {"name": c, "kind": pc["kind"], "genes": genes, "nbr": nbr[pc["version"]]}))
    if not gene_cols:
        return {"status": "no gene-defined constituent"}
    draws, recs = {}, {}
    for a in arrays:
        coh, _ = load_cohort(a.key, *SOURCES[a.key])
        for j, v in gene_cols:
            v = dict(v, genes=[g for g in v["genes"] if g in coh.panel])
            if not v["genes"]:
                return {"status": f"constituent {v['name']} absent from {a.key}"}
            m = matched_programmes(coh, v, seed=seed + j, n_accept=n)
            recs[f"{a.key}|{v['name']}"] = {k: m[k] for k in m if k != "values"}
            if m["values"] is None:
                return {"status": "not computable", "records": recs}
            draws[(a.key, j)] = m["values"]

    def pooled(replace):
        S_all = []
        for a in arrays:
            Z = np.array(a.Z, float, copy=True)
            if replace is not None:
                for j, _ in gene_cols:
                    Z[:, j] = draws[(a.key, j)][:, replace]
            a2 = dataclasses.replace(a, Z=Z)
            t = transform_cohort(a2)
            units = np.array([f"{a.key}|{p}" for p in t["patient"]])
            _, S = unit_stats(t["y"], t["base_nuis"], t["base_nuis"] + ff.rel_nuis.predict(t["Z"]), units)
            S_all.append(S)
        return pooled_gain(np.vstack(S_all))

    obs = pooled(None)
    null = [pooled(i) for i in range(n)]
    return {"status": "run", "observed_pooled_gain": obs, "n_draws": n, "null_mean": float(np.mean(null)),
            "null_q95": float(np.quantile(null, 0.95)), "p": float((1 + sum(x >= obs for x in null)) / (n + 1)),
            "replaced_constituents": [v["name"] for _, v in gene_cols], "records": recs,
            "reading": "reported, never gating (ccrcc_external_replication_v03 moran_matched_null)"}


def stage_replication(out):
    from dbench.downstream.replication_v03 import evaluate_finding, load_external, run_family

    frozen = _load(out)
    root = f"{out}/locks"
    cat = bknown.covariate_sets()
    for fid, v in frozen.items():
        c = v["candidate"]
        if locks.read_record(root, fid, "internal_validation") is not None:
            continue
        locks.write_lock(root, fid, "registry_freeze", inputs={"relation_params_hash": v["ff"].rel_nuis.params_hash()},
                         declared_analysis={"source": "PREVIEW candidate list (pre-aggregation)"})
        locks.write_record(root, fid, "registry_freeze", _json({"finding_id": fid, **c}))
        locks.write_lock(root, fid, "internal_validation", inputs={}, declared_analysis={"rule": "preview fold rule"})
        locks.write_record(root, fid, "internal_validation", _json({k: c.get(k) for k in (
            "passing_folds", "known_folds", "higher_order_folds", "mean_gain_B_nuis_passing",
            "mean_gain_B_known_known_folds", "worst_null_p_passing_folds")}))
    log("replication v03 family of", len(frozen))
    res = run_family([v["ff"] for v in frozen.values()], root, COHORTS, catalogue=cat)
    out_rows = {}
    for fid, v in frozen.items():
        r = res[fid]
        row = {"externally_replicated": r["externally_replicated"], "bh_q": r.get("bh_q"),
               "criteria": r.get("criteria"), "pooled": r["pooled"], "direction": r.get("direction"),
               "testability": r["testability"], "per_cohort_pooled_run": {}}
        arrays, _ = load_external(v["ff"], COHORTS, cat)
        for a in arrays:   # the same evaluation on one cohort at a time (no BH; reported)
            e = evaluate_finding(v["ff"], [a])
            g = e["pooled"].get("nuis") or {}
            fa = (e.get("direction") or {}).get("fraction_agree")
            row["per_cohort_pooled_run"][a.key] = {
                "gain_nuis": g.get("gain"), "ci": g.get("ci"), "p_boot": g.get("p_boot"),
                "gain_known": (e["pooled"].get("known") or {}).get("gain"),
                "direction_fraction_agree": fa, "n_patients": e["per_cohort"][a.key]["n_patients"],
                "pass_ci_and_sign": bool(g.get("ci") and g["ci"][0] > 0 and fa is not None and fa >= 0.6)}
        out_rows[fid] = _json(row)
        Path(f"{out}/replication_v03.json").write_text(json.dumps(out_rows, indent=1))
    return {fid: {"replicated": r["externally_replicated"], "q": r["bh_q"],
                  "per_cohort_pass": {k: x["pass_ci_and_sign"] for k, x in r["per_cohort_pooled_run"].items()}}
            for fid, r in out_rows.items()}


def stage_matched_null(out):
    """(Re)compute the Moran-matched null for every testable finding and update replication_v03.json."""
    from dbench.downstream.replication_v03 import load_external

    frozen = _load(out)
    cat = bknown.covariate_sets()
    path = Path(f"{out}/replication_v03.json")
    rows = json.loads(path.read_text())
    for fid, v in frozen.items():
        arrays, _ = load_external(v["ff"], COHORTS, cat)
        if not arrays:
            rows[fid]["matched_smooth_programme_null"] = {"status": "not testable externally"}
            continue
        log("matched null", fid)
        try:
            rows[fid]["matched_smooth_programme_null"] = _json(matched_null_pooled(v["ff"], arrays))
        except Exception as ex:
            rows[fid]["matched_smooth_programme_null"] = {"status": f"error {type(ex).__name__}: {ex}"}
        path.write_text(json.dumps(rows, indent=1))
    return {f: {k: (r.get("matched_smooth_programme_null") or {}).get(k) for k in ("status", "observed_pooled_gain",
                                                                                  "null_q95", "p")}
            for f, r in rows.items()}


NOT_TRANSFERABLE_MORPH = ("not transferable: no external image <= 0.5 um/px (external rule; Zenodo is 0.66 um/px, Li 2022 "
                          "and GSE250163 carry no full-resolution image); external evidence routed through the "
                          "tissue-level CPTAC -> TCGA H&E student (stage tissue_external)")


def _calibration(t, f, n_boot=2000, seed=20261023):
    """Calibration of the frozen relation's f_S against the external B_nuis residual r = y - f_B (both on the
    transformed, within-patient standardised scale): slope and intercept of OLS r ~ f, calibration-in-the-large
    mean(r - f), patient-bootstrap 95% CI of the slope. Slope 1 = amplitude transfers; 0 < slope < 1 = right
    direction, frozen amplitude too large."""
    r = np.asarray(t["y"] - t["base_nuis"], float)
    f = np.asarray(f, float)
    pat = np.asarray(t["patient"]).astype(str)

    def slope(ix):
        fc, rc = f[ix] - f[ix].mean(), r[ix] - r[ix].mean()
        return float(fc @ rc / max(fc @ fc, 1e-12))
    allix = np.arange(len(r))
    b1 = slope(allix)
    pats = np.unique(pat)
    rows = {p_: np.flatnonzero(pat == p_) for p_ in pats}
    rng = np.random.default_rng(seed)
    bs = [slope(np.concatenate([rows[p_] for p_ in rng.choice(pats, len(pats))])) for _ in range(n_boot)]
    per_patient = {str(p_): slope(ix) for p_, ix in rows.items()}
    return {"calibration_slope": b1, "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "calibration_intercept": float(r.mean() - b1 * f.mean()),
            "calibration_in_the_large": float(np.mean(r - f)), "sd_f": float(f.std()), "sd_residual": float(r.std()),
            "per_patient_slope": per_patient, "n_rows": int(len(r)), "n_patients": int(len(pats))}


def stage_calibration(out):
    """Declared reporting tier + calibration of the frozen relation in each external cohort (no refit)."""
    from dbench.downstream.replication_v03 import load_external, transform_cohort

    frozen = _load(out)
    cat = bknown.covariate_sets()
    path = Path(f"{out}/replication_v03.json")
    rows = json.loads(path.read_text())
    summ = {}
    for fid, v in frozen.items():
        ff, row = v["ff"], rows[fid]
        if any("morph_dictionary_v03:" in cc for cc in ff.rel_nuis.candidates) or                 str(ff.rel_nuis.target).startswith("morph:k"):
            row["tier"] = "not transferable"
            row["not_transferable_reason"] = NOT_TRANSFERABLE_MORPH
            summ[fid] = {"tier": row["tier"]}
            continue
        arrays, _ = load_external(ff, COHORTS, cat)
        cal = {}
        for a in arrays:
            t = transform_cohort(a)
            cal[a.key] = _calibration(t, ff.rel_nuis.predict(t["Z"]))
        row["calibration"] = cal
        fa = (row.get("direction") or {}).get("fraction_agree")
        if row["externally_replicated"]:
            tier = "replicated (v03 rule)"
        elif fa is not None and fa >= 0.6:
            tier = "sign replicates, gain does not"
        else:
            tier = "does not replicate"
        row["tier"] = tier
        summ[fid] = {"tier": tier, "pooled_gain": ((row.get("pooled") or {}).get("nuis") or {}).get("gain"),
                     "sign_fraction": fa,
                     "calibration": {k: {"slope": c["calibration_slope"], "slope_ci95": c["slope_ci95"],
                                         "citl": c["calibration_in_the_large"]} for k, c in cal.items()}}
    row_note = {"tiers": "declared reporting tiers beside the registered rule (coordinator 2026-10-23): 'replicated "
                         "(v03 rule)', 'sign replicates, gain does not' (dominant-term sign >= 60% of external "
                         "patients without the pooled-gain criterion), 'does not replicate', 'not transferable'",
                "calibration": "OLS of the external B_nuis residual on the frozen f_S, transformed scale; no refit"}
    rows["_notes"] = row_note
    path.write_text(json.dumps(_json(rows), indent=1))
    return summ


def stage_tissue_external(out):
    """Tissue-level H&E student route (labelled; not the v03 replication rule): the state (b) ridge student fitted
    on all covered discovery spots predicts each CPTAC and TCGA-KIRC slide (mean over tiles); Spearman with the
    bulk score of the finding's target programme, and with the bulk-rebuilt state where it exists."""
    import pandas as pd
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.downstream import bulk, distill as D
    from dbench.downstream.dryrun import CPTAC_TILES, TCGA_TILES, Discovery, slide_predictions
    from dbench.downstream.values import parse_candidate, set_genes

    frozen = _load(out)
    disc = _discovery()
    sets = _programme_sets()
    E_all = np.asarray(disc.coh.raw.hoptimus, np.float32)
    cov_all = np.asarray(getattr(disc.coh.raw, "covered", np.ones(len(E_all), bool)), bool) & np.isfinite(E_all).all(1)
    students = {}
    for fid, v in frozen.items():
        rows = v["aux"]["rows"]
        keep = cov_all[rows]
        students[fid] = D.RidgeStudent().fit(E_all[rows][keep], state_b(v["ff"], v["aux"]["Z_raw"])[keep],
                                             v["aux"]["patient"][keep])
    log("TCGA slides")
    tp = slide_predictions(students, sorted(glob.glob(TCGA_TILES)),
                           lambda p_, z: "-".join(os.path.basename(p_).split("-")[:3]))
    log("CPTAC slides")
    cp = slide_predictions(students, sorted(glob.glob(CPTAC_TILES)),
                           lambda p_, z: "-".join(os.path.basename(p_).split("-")[:2]))
    tp.pop("_mean_embedding", None)
    cp.pop("_mean_embedding", None)
    bulks = {"cptac": bulk.cptac_tables()["rna"], "tcga": bulk.tcga_tables()["rna"]}
    preds = {"cptac": cp, "tcga": tp}
    res = {}
    for fid, v in frozen.items():
        tgt = parse_candidate(v["candidate"]["target"])
        r = {"label": "tissue-level CPTAC -> TCGA H&E student (state (b) ridge); not the v03 replication rule",
             "target": v["candidate"]["target"]}
        for coh in ("cptac", "tcga"):
            expr, pr = bulks[coh], preds[coh][fid]
            ids = sorted(set(pr) & set(expr.index))
            rec = {"n_matched": len(ids)}
            if len(ids) < 10:
                rec.update(status="too few matched slides", example_slide_ids=sorted(pr)[:3],
                           example_bulk_ids=[str(i) for i in expr.index[:3]])
                r[coh] = rec
                continue
            a = np.array([pr[i] for i in ids])
            if tgt["kind"] == "programme":
                ts, ng = bulk.set_score(expr.loc[ids], set_genes(sets, tgt["set"]))
                rec["vs_bulk_target_programme"] = {"spearman": D.spearman(a, ts.values), "n_genes": ng,
                                                   "ci95": _boot_spearman(a, ts.values)}
            bs, how = bulk_state_b(v["ff"], expr.loc[ids], sets)
            if bs is not None:
                rec["vs_bulk_rebuilt_state"] = {"spearman": D.spearman(a, bs.values), "rebuilt": how,
                                                "ci95": _boot_spearman(a, bs.values)}
            r[coh] = rec
        res[fid] = r
    Path(f"{out}/tissue_external.json").write_text(json.dumps(_json(res), indent=1))
    pd.DataFrame({f"{fid}|{c}": pd.Series(preds[c][fid]) for fid in frozen for c in preds}).to_csv(
        f"{out}/tissue_slide_predictions.csv")
    return {f: {c: {k: (x.get("spearman"), x.get("ci95")) for k, x in r[c].items() if isinstance(x, dict)}
                for c in ("cptac", "tcga")} for f, r in res.items()}


def _boot_spearman(a, b, n_boot=2000, seed=20261021):
    from dbench.downstream import distill as D

    rng = np.random.default_rng(seed)
    bs = [D.spearman(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


# ----------------------------------------------------------------------------- characterisation
def heldout_gain_v03(Z, ys, pat, dep, n_folds=6):
    """Pooled held-out gain of the v03_rff relation over the depth baseline across patient folds."""
    from morpheus_encoder.learned_search import relation_v03 as rv

    from dbench.downstream.relation import patient_folds

    sse, sse0 = 0.0, 0.0
    for held in patient_folds(pat, max_folds=n_folds, seed=0):
        h = np.isin(pat, held)
        base = rv.fit_baseline(ys[~h], pat[~h], dep[~h], yq=ys[h], C_q=dep[h])
        o = rv.fit_sets(Z[~h][:, None, :], base, ys[~h], pat[~h], Vq=Z[h][:, None, :], yq=ys[h], groups_q=pat[h],
                        device="cpu", return_beta=True)
        sse += float(np.sum((np.asarray(base.r_q) - np.asarray(o["pred_q"])[:, 0]) ** 2))
        sse0 += float(np.sum(np.asarray(base.r_q) ** 2))
    return 1.0 - sse / sse0


def stage_characterise(out):
    from dbench.downstream import characterise as C
    from dbench.downstream.dryrun import Discovery, _reindex_nb
    from dbench.downstream.relation import standardise_within, standardised_contributions
    from dbench.downstream.values import candidate_genes, cohort_values, parse_candidate

    frozen = _load(out)
    disc = _discovery()
    cv = disc.cellvit()
    refs = C.reference_sets(disc.sets)
    prog = np.asarray(disc.coh.raw.programmes, float)
    pnames = [str(n) for n in disc.coh.raw.programme_names]
    X = disc.coh.logcpm
    summary = {}
    for fid, v in frozen.items():
        log("characterise", fid)
        ff, aux = v["ff"], v["aux"]
        rows = aux["rows"]
        Zs, ys, pat = aux["Zs"], aux["ys"], aux["patient"]
        state = np.full(len(disc.section), np.nan)
        state[rows] = ff.rel_nuis.predict(Zs)
        target = np.full(len(disc.section), np.nan)
        target[rows] = ys
        rec = {"state_definition": "a_within_patient (f_S of the frozen v03_rff relation)"}
        grad = ff.rel_nuis.gradient(Zs).mean(0)
        contrib = standardised_contributions(ff.rel_nuis, Zs, aux["r_nuis"])
        rec["constituents"] = [{"candidate": cc, "class": C.candidate_class(cc),
                                "spatial_version": C.candidate_version(cc), "mean_partial_derivative": float(grad[j]),
                                "partial_association_sign": int(np.sign(grad[j])),
                                "standardised_contribution_support": float(contrib[j])}
                               for j, cc in enumerate(ff.rel_nuis.candidates)]
        genes = []
        for cc in ff.rel_nuis.candidates:
            try:
                genes += candidate_genes(parse_candidate(cc), disc.sets, disc.coh.panel)
            except Exception:
                pass
        ok = np.isfinite(state)
        ri = np.flatnonzero(ok)
        Xs = X[ri]
        Xs = np.asarray(Xs.todense() if hasattr(Xs, "todense") else Xs, float)
        Xc = standardise_within(Xs, disc.patient[ok])[0].reshape(len(ri), -1)
        sc = standardise_within(state[ok], disc.patient[ok])[0].ravel()
        corr = np.nan_to_num(Xc.T @ sc / len(sc))
        o = np.argsort(-corr)
        rec["decoding_top_genes"] = {"positive": [(disc.genes[i], float(corr[i])) for i in o[:25]],
                                     "negative": [(disc.genes[i], float(corr[i])) for i in o[::-1][:25]],
                                     "statistic": "within-patient Pearson of log-CPM with the state"}
        gi = {g: i for i, g in enumerate(disc.genes)}
        impl = []
        for cc in ff.rel_nuis.candidates:   # implicated genes: gene constituents + the 25 members of each programme
            try:                             # constituent most correlated with the state (within patient)
                pc = parse_candidate(cc)
                if pc["kind"] not in ("gene", "programme"):
                    continue
                g_c = candidate_genes(pc, disc.sets, disc.coh.panel)
            except Exception:
                continue
            impl += g_c if pc["kind"] == "gene" else sorted((g for g in g_c if g in gi),
                                                             key=lambda g: -corr[gi[g]])[:25]
        rec["implicated_genes"] = list(dict.fromkeys(impl))
        rec["enrichment"] = C.enrichment({"constituents": genes, "state_top100_positive": [disc.genes[i] for i in o[:100]],
                                          "state_top100_negative": [disc.genes[i] for i in o[::-1][:100]]},
                                         disc.genes, refs, score=state[ok], programme_scores=prog[ok],
                                         programme_names=pnames)
        rec["enrichment"]["adjacent_known_biology"] = rec["enrichment"]["adjacent_known_biology"][:15]
        okc = ok & cv["covered"]
        names = list(getattr(C, "PANNUKE", range(cv["fractions"].shape[1])))
        rec["cell_type_association"] = {str(nm): float(np.corrcoef(state[okc], cv["fractions"][okc, i])[0, 1])
                                        for i, nm in enumerate(names)}
        loc = C.localisation(state[ok], target[ok], disc.section[ok], _reindex_nb(disc.nb6, ok), cv["strata"][ok],
                             coords=np.asarray(disc.coh.raw.coords, float)[ok], out_dir=f"{out}/findings/{fid}/maps")
        rec["localization"] = loc
        dep = standardise_within(np.log1p(np.asarray(disc.coh.raw.depth, float)[rows]), pat)[0].reshape(-1, 1)
        full = heldout_gain_v03(Zs, ys, pat, dep)
        ovl = {"heldout_gain_full": full, "rows": {}, "relation": "v03_rff, 6 patient folds"}
        for j, cc in enumerate(ff.rel_nuis.candidates):
            try:
                pc = parse_candidate(cc)
            except Exception:
                continue
            if pc.get("version") == "v0":
                continue
            own_name = cc.split(":", 1)[1]
            ov = np.asarray(cohort_values(disc.coh, own_name, disc.sets, disc.coh.panel), float).ravel()[rows]
            Zj = Zs.copy()
            Zj[:, j] = standardise_within(ov, pat)[0].ravel()
            g = heldout_gain_v03(Zj, ys, pat, dep)
            ovl["rows"][cc] = {"own_version": own_name, "heldout_gain_own_version": g, "gain_lost": full - g}
        rec["scale_decomposition"] = ovl
        involves_space = any(C.candidate_version(cc) != "v0" for cc in ff.rel_nuis.candidates)
        rec["squidpy"] = C.neighbourhood(state[ok], cv["strata"][ok], disc.coords_um[ok], disc.section[ok],
                                         moran_median=loc["moran_median"], involves_space=involves_space)
        cond = C.liana_condition(genes, rec["squidpy"])
        if cond[0]:
            rec["liana"] = C.liana_bivariate(X[ri], disc.genes, state[ok], disc.coords_um[ok], disc.section[ok],
                                             condition=cond)
        else:
            rec["liana"] = {"status": "skipped", "reason": cond[1]}
        rec["pathologist_read"] = "off (scratched by the owner 2026-10-21)"
        if str(ff.rel_nuis.target).startswith("morph:k"):   # reverse reading of a morphology-centre target
            mc, _ = morph_columns(disc.coh)
            tv = mc[str(ff.rel_nuis.target)]
            top = np.argsort(-np.nan_to_num(tv, nan=-np.inf))[:32]
            rec["centre_reverse_reading"] = {
                "nearest_spots": [{"section": str(disc.section[i]), "barcode": str(np.asarray(disc.coh.raw.barcode)[i]),
                                   "similarity": float(tv[i])} for i in top[:16]],
                "mean_cellvit_fractions_top32": dict(zip(map(str, getattr(C, "PANNUKE", range(cv["fractions"].shape[1]))),
                                                         np.asarray(cv["fractions"][top]).mean(0).round(3).tolist())),
                "molecular_state_predicting_it": "decoding_top_genes (state = frozen f_S of the molecular neighbourhood constituents)",
                "note": "patch images are the HEST spot patches of the listed (section, barcode)"}
        rec = _json(rec)
        Path(f"{out}/findings/{fid}").mkdir(parents=True, exist_ok=True)
        Path(f"{out}/findings/{fid}/characterisation.json").write_text(json.dumps(rec, indent=1))
        summary[fid] = {"top_positive_genes": [g for g, _ in rec["decoding_top_genes"]["positive"][:10]],
                        "top_adjacent": rec["enrichment"]["adjacent_known_biology"][:3],
                        "ora_top_positive": [(t.get("set") or t.get("name"), t.get("q")) for t in
                                             rec["enrichment"]["tests"]["state_top100_positive"]["top"][:3]],
                        "cell_type_association": rec["cell_type_association"],
                        "moran_median": rec["localization"].get("moran_median"),
                        "own_version_loss": rec["scale_decomposition"], "liana": rec["liana"].get("status", "run")}
        Path(f"{out}/characterisation_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {f: "ok" for f in summary}


# ----------------------------------------------------------------------------- therapeutic
def implicated_genes(ff, sets, panel) -> list:
    from dbench.downstream.values import candidate_genes, parse_candidate

    genes = []
    for cc in ff.rel_nuis.candidates:
        try:
            pc = parse_candidate(cc)
        except Exception:
            continue
        if pc["kind"] not in ("gene", "programme"):
            continue
        g = candidate_genes(pc, sets, panel)
        genes += g if pc["kind"] == "gene" else g[:25]
    return list(dict.fromkeys(genes))


def stage_therapeutic(out, cache):
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.downstream import therapeutic as T
    from dbench.scripts.external_replication import load_discovery

    frozen = _load(out)
    rep = json.loads(Path(f"{out}/replication_v03.json").read_text())
    from dbench.downstream.discovery_any import cohort
    sets, panel = _programme_sets(), cohort(STUDY).panel
    dm = T.DepMap(cache_root=cache)
    otv = T.open_targets_version(cache)
    mondo = T.resolve_mondo(otv["cache_dir"])
    res, pri_in = {}, []
    for fid, v in frozen.items():
        cpath = Path(f"{out}/findings/{fid}/characterisation.json")
        genes = (json.loads(cpath.read_text()).get("implicated_genes") if cpath.exists() else None) or \
            implicated_genes(v["ff"], sets, panel)
        dep = [dm.gene(g) for g in genes]
        ot = [T.ot_gene(g, otv["cache_dir"], mondo["disease_id_used"]) for g in genes]
        res[fid] = _json({"genes": genes, "depmap": dep, "open_targets": ot, "depmap_release": dm.version_record(),
                          "platform_release": otv, "disease_ids": mondo})
        handle = any((d or {}).get("context_dependency") for d in dep) or any(
            (o or {}).get("actionable_in_principle") or (o or {}).get("clinically_precedented") for o in ot)
        pooled = (rep[fid]["pooled"] or {}).get("nuis") or {}
        pri_in.append({"id": fid, "external_ci_lower": (pooled.get("ci") or [None])[0],
                       "higher_order": len(v["candidate"].get("higher_order_folds") or []) >= 3,
                       "grade": None, "has_handle": bool(handle), "external_gain": pooled.get("gain")})
    pr = T.prioritise(pri_in)
    Path(f"{out}/therapeutic.json").write_text(json.dumps({"per_finding": res, "prioritisation": _json(pr),
                                                          "note": "literature grade not assessed (no novelty step "
                                                                  "in the preview)"}, indent=1))
    return {p["id"]: {"score": p["score"], "indicators": p["indicators"]} for p in pr}


# ----------------------------------------------------------------------------- distillation
def state_b(ff, Z_raw):
    """assay_distillation_v03 state (b): the frozen f_S on support-standardised constituents (one mean / sd over
    all discovery spots, no within-patient centring)."""
    Zg = (Z_raw - Z_raw.mean(0)) / (Z_raw.std(0) + 1e-12)
    return ff.rel_nuis.predict(Zg)


def bulk_state_b(ff, expr, sets):
    """State (b) rebuilt from bulk: gene constituent = log expression, programme constituent = set score;
    neighbour versions take their own version (proxy); morphology constituents cannot be rebuilt."""
    import pandas as pd

    from dbench.downstream import bulk
    from dbench.downstream.values import parse_candidate, set_genes

    cols, notes = [], []
    for cc in ff.rel_nuis.candidates:
        if "morph_dictionary_v03:" in cc:
            return None, "morphology-dictionary constituent: not rebuildable from bulk"
        pc = parse_candidate(cc)
        if pc["version"] != "v0":
            notes.append(f"{cc} -> own version (bulk has no neighbourhood)")
        if pc["kind"] == "gene":
            if pc["genes_key"] not in expr.columns:
                return None, f"{pc['genes_key']} absent from bulk"
            cols.append(expr[pc["genes_key"]].values.astype(float))
        else:
            s, _ = bulk.set_score(expr, set_genes(sets, pc["set"]))
            if s is None:
                return None, f"{pc['set']} has no genes in bulk"
            cols.append(s.values.astype(float))
    Z = np.column_stack(cols)
    Zg = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return pd.Series(ff.rel_nuis.predict(Zg), index=expr.index), "; ".join(notes) or "exact"


def stage_distill(out):
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.downstream import bulk, distill as D
    from dbench.downstream.dryrun import CPTAC_TILES, Discovery, slide_predictions

    frozen = _load(out)
    disc = _discovery()
    sets = _programme_sets()
    E_all = np.asarray(disc.coh.raw.hoptimus, np.float32)
    cov_all = np.asarray(getattr(disc.coh.raw, "covered", np.ones(len(E_all), bool)), bool) & np.isfinite(E_all).all(1)
    renal = STUDY == "meylan2022_ccrcc"
    cpt = bulk.cptac_tables() if renal else None
    res, students, bulk_states = {}, {}, {}
    for fid, v in frozen.items():
        ff, aux = v["ff"], v["aux"]
        rows = aux["rows"]
        sb = state_b(ff, aux["Z_raw"])
        keep = cov_all[rows]
        E, y, g = E_all[rows][keep], sb[keep], aux["patient"][keep]
        share = D.between_patient_share(y, g)
        log("distill", fid, "n", len(y), "between-patient share", share)
        r = {"state_definition": "b_between_patient_inclusive", "n_spots": int(len(y)), "between_patient_share": share}
        rr = D.heldout_spot_eval(D.RidgeStudent, E, y, g, n_perm=3)
        rr.pop("_pred")
        r["ridge"] = rr
        rm = D.heldout_spot_eval(D.MLPStudent, E, y, g)
        rm.pop("_pred")
        r["mlp"] = rm
        students[f"{fid}|ridge"] = D.RidgeStudent().fit(E, y, g)
        students[f"{fid}|mlp"] = D.MLPStudent().fit(E, y, g)
        if renal:
            bs, how = bulk_state_b(ff, cpt["rna"], sets)
        else:
            bs, how = None, f"{STUDY}: no matched bulk slide cohort declared (CPTAC transfer is renal only)"
        r["cptac_bulk_state"] = how if bs is None else {"rebuilt": how, "n": int(len(bs))}
        if bs is not None:
            bulk_states[fid] = bs
        res[fid] = r
        Path(f"{out}/distillation.json").write_text(json.dumps(_json(res), indent=1))
    log("CPTAC slides")
    preds = slide_predictions(students, sorted(glob.glob(CPTAC_TILES)) if renal else [],
                              lambda p, z: "-".join(os.path.basename(p).split("-")[:2]))
    preds.pop("_mean_embedding", None)
    for fid, r in res.items():
        bs = bulk_states.get(fid)
        if bs is None:
            r["cptac_slide_transfer"] = {"status": "not assessed", "reason": r["cptac_bulk_state"]}
            continue
        r["cptac_slide_transfer"] = {}
        for kind in ("ridge", "mlp"):
            p = preds[f"{fid}|{kind}"]
            ids = sorted(set(p) & set(bs.index))
            if len(ids) < 10:
                r["cptac_slide_transfer"][kind] = {"n": len(ids), "status": "too few matched slides",
                                                   "example_slide_ids": sorted(p)[:3],
                                                   "example_bulk_ids": [str(i) for i in bs.index[:3]]}
                continue
            a, b = np.array([p[i] for i in ids]), bs[ids].values
            r["cptac_slide_transfer"][kind] = {"n": len(ids), "spearman": D.spearman(a, b)}
    Path(f"{out}/distillation.json").write_text(json.dumps(_json(res), indent=1))
    return {f: {"ridge_spot": r["ridge"]["tile_spearman"], "ridge_patient": r["ridge"]["patient_spearman"],
                "ridge_patient_ci": r["ridge"]["patient_spearman_ci95"],
                "mlp_spot": r["mlp"]["tile_spearman"], "mlp_patient": r["mlp"]["patient_spearman"],
                "between_patient_share": r["between_patient_share"], "cptac": r["cptac_slide_transfer"]}
            for f, r in res.items()}


# ----------------------------------------------------------------------------- clinical prespecification
def stage_clinical_lock(out):
    from dbench.downstream import clinical as K

    frozen = _load(out)
    root = f"{out}/clinical_locks"
    fids = sorted(frozen)
    written = {}
    for fid, v in frozen.items():
        ff = v["ff"]
        gene_sig = not any("morph_dictionary_v03:" in cc for cc in ff.rel_nuis.candidates)
        score = {"finding": fid, "relation_params_hash": ff.rel_nuis.params_hash(),
                 "state_definition": "b_between_patient_inclusive",
                 "bulk_score": ("frozen f_S on bulk-rebuilt constituents (gene log expression / set score; neighbour "
                                "versions -> own version), single-intercept standardisation") if gene_sig else None,
                 "slide_score": "mean tile prediction of the state (b) H&E ridge student fitted on all discovery spots",
                 "score_used": "bulk_score" if gene_sig else "slide_score"}
        plans = [("cptac_os", "CPTAC ccRCC", "OS", ["age", "sex", "stage"], "Cox OS ~ score/SD + age + sex + stage"),
                 ("tcga_os", "TCGA-KIRC", "OS", ["age", "sex", "stage"], "Cox OS ~ score/SD + stage + age + sex"),
                 ("tcga_pfs", "TCGA-KIRC", "PFS", ["age", "sex", "stage"], "Cox PFS ~ score/SD + stage + age + sex")]
        if gene_sig:
            plans.append(("javelin_pfs", "JAVELIN Renal 101", "PFS", ["age", "sex"],
                          "Cox PFS ~ arm + score/SD + arm x score/SD + age + sex"))
        for key, cohort, ep, covs, model in plans:
            lk = K.write_clinical_lock(root, f"{fid}__{key}", cohort=cohort, endpoint=ep, score_definition=score,
                                       covariates=covs, model=model, family=[f"{f}__{key}" for f in fids],
                                       allow_violation=True)
            written[f"{fid}__{key}"] = lk.get("lock_hash")
    Path(f"{out}/clinical_prespecification.json").write_text(json.dumps(
        {"status": "locks only; no fitting until the registered aggregate", "locks": written,
         "note": "allow_violation: the preview has no literature / prioritisation records; the registered pass "
                 "writes its own locks in order"}, indent=1))
    return {"n_locks": len(written)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["freeze", "replication", "matched_null", "calibration", "tissue_external", "characterise", "therapeutic", "distill", "clinical_lock"])
    ap.add_argument("--protocol", default="dbench/protocols/renal_spatial_v07.json")
    ap.add_argument("--cache", default="runs/development/downstream_v01/cache")
    ap.add_argument("--study", default="meylan2022_ccrcc")
    ap.add_argument("--row-restriction", default=None)
    a = ap.parse_args()
    global STUDY, ROW_RESTRICTION
    STUDY = a.study
    ROW_RESTRICTION = a.row_restriction
    if "runs/prospective" in os.path.abspath(a.out):
        raise SystemExit("refusing to write under runs/prospective")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cands = json.loads(Path(a.candidates).read_text())["candidates"]
    commit = json.loads(Path(a.protocol).read_text())["release"].get("relation_commit")
    fn = {"freeze": lambda: stage_freeze(cands, a.out, commit), "replication": lambda: stage_replication(a.out),
          "matched_null": lambda: stage_matched_null(a.out),
          "calibration": lambda: stage_calibration(a.out),
          "tissue_external": lambda: stage_tissue_external(a.out),
          "characterise": lambda: stage_characterise(a.out), "therapeutic": lambda: stage_therapeutic(a.out, a.cache),
          "distill": lambda: stage_distill(a.out), "clinical_lock": lambda: stage_clinical_lock(a.out)}[a.stage]
    t0 = time.time()
    res = {"stage": a.stage, "result": fn()}
    res["seconds"] = round(time.time() - t0, 1)
    Path(f"{a.out}/stage_{a.stage}.json").write_text(json.dumps(_json(res), indent=1))
    print(json.dumps(_json(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
