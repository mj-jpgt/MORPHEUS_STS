"""Integration dry run (downstream spec 10.2 item 7): the original paper's G18 state and the kidney v0.5
relation own:Glutathione <- nbr Fatty-acid metabolism pushed through lock -> (replication) ->
characterisation -> novelty -> prioritisation -> orthogonal -> therapeutic -> distillation -> clinical ->
state card, with every step locked before it opens its data.

    PYTHONPATH=$PWD:$HOME/q_pylib ~/morpheus_venv/bin/python -m dbench.downstream.dryrun --out runs/development/downstream_v01/dryrun_<date>

Legacy findings carry no learned-search registry entry; the registry and internal-validation records hold
the original receipts (marked legacy). The Glutathione relation fails the v03 rule, so every step after
replication is run for integration only and its card is marked ``dry_run_beyond_stop`` (Section 0: nothing
below replication is run on a failed finding for the headline).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from . import bknown, bulk, characterise as C, clinical as K, distill as D, locks, novelty as N, therapeutic as T

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
G_SCORES = f"{STORE}/code/ws_phase1_tissue_20260822/phase2_ccrcc_outer_0{{}}/output/heldout_G_scores_with_ST_fullres_coordinates_v2.npz"
TCGA_TILES = f"{STORE}/external/tcga_kirc_he_descriptors_20261013/tiles/*.npz"
CPTAC_TILES = f"{STORE}/code/ws_cptac_embed/production/ccrcc/chunk_*/slides/*.npz"
HEST_DIR = f"{STORE}/morpheus_substrate_mirror/hest/ccrcc"
ORIGINAL = {"tcga_pooled": 0.470, "tcga_loso": 0.395, "tcga_within_site_cv": 0.423, "tcga_within_site_loso": 0.417,
            "cptac_rna_protein": 0.742947952122264, "javelin_interaction_hr": 0.765,
            "javelin_interaction_ci": [0.617, 0.949], "source": "figures/PANEL_CAPTIONS.md Fig 6; wsAG TABLE_TASK3"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _json(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def stage(root, fid, st, inputs, declared, fn, *, allow_violation=False):
    locks.write_lock(root, fid, st, inputs=inputs, declared_analysis=declared, allow_violation=allow_violation)
    res = fn()
    locks.write_record(root, fid, st, _json({k: v for k, v in res.items() if not k.startswith("_")}))
    return res


# ----------------------------------------------------------------------------- discovery cohort
class Discovery:
    def __init__(self):
        from dbench.scripts.external_replication import load_discovery
        from morpheus.tier2.substrates.hest_multi import _programme_sets

        self.coh = load_discovery()
        r = self.coh.raw
        self.sets = _programme_sets()
        self.section = np.asarray(r.sample_id).astype(str)
        self.patient = np.asarray(r.patient).astype(str)
        self.nb6 = r.neighbours[6]
        um = {s: json.load(open(f"{HEST_DIR}/{s}_metadata.json"))["pixel_size_um_estimated"] for s in np.unique(self.section)}
        self.px_um = np.array([um[s] for s in self.section])
        self.coords_um = np.asarray(r.coords, float) * self.px_um[:, None]
        self.genes = list(r.genes)

    def cellvit(self):
        if hasattr(self, "_cv"):
            return self._cv
        import h5py

        from morpheus.tier2.substrates.hest import CELLVIT_CLASSES, _cellvit_summary_sample

        F = np.zeros((len(self.section), 2 + len(CELLVIT_CLASSES)))
        cov = np.zeros(len(self.section), bool)
        for s in np.unique(self.section):
            rows = np.flatnonzero(self.section == s)
            with h5py.File(f"{HEST_DIR}/{s}_st.h5ad", "r") as f:
                rad = float(f["uns/spatial/ST/scalefactors/spot_diameter_fullres"][()]) / 2
            F[rows], cov[rows] = _cellvit_summary_sample(f"{HEST_DIR}/{s}_cellvit_seg.parquet",
                                                         np.asarray(self.coh.raw.coords, float)[rows], rad, CELLVIT_CLASSES)
        self._cv = {"fractions": F[:, 2:], "count": F[:, 0], "area": F[:, 1], "covered": cov,
                    "strata": C.composition_strata(F[:, 2:], cov)}
        return self._cv

    def patch_loader(self, sec, rows):
        import h5py

        bc = set(np.asarray(self.coh.raw.barcode).astype(str)[rows])
        with h5py.File(f"{HEST_DIR}/{sec}_patches.h5", "r") as f:
            codes = [b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in f["barcode"][:]]
            idx = sorted(i for i, b in enumerate(codes) if b in bc)[:9]
            imgs = [f["img"][i] for i in idx]
        if not imgs:
            return None
        while len(imgs) < 9:
            imgs.append(np.full_like(imgs[0], 255))
        return np.vstack([np.hstack(imgs[i:i + 3]) for i in (0, 3, 6)])


def g18_spot_scores(disc: Discovery) -> np.ndarray:
    """G18 state score per discovery spot: the frozen six-fold consensus loading applied to the
    covariate-adjusted spot residual (log depth, the three CellViT composition fractions, the 41 Gavish
    scores; per-patient intercepts). The stored per-fold held-out scores are NOT pooled: their sign is
    fold-specific (Spearman with the consensus projection ranges -0.26 to +0.37 across the six folds)."""
    genes, w = bulk.frozen_loading("G18")
    cv = disc.cellvit()
    names = [str(n) for n in disc.coh.raw.programme_names]
    mp = np.asarray(disc.coh.raw.programmes, float)[:, [i for i, n in enumerate(names) if n.startswith("mp:")]]
    F = cv["fractions"]
    covs = np.column_stack([np.log1p(np.asarray(disc.coh.raw.depth, float)), F[:, 0], F[:, 1], F[:, 3], mp])
    from .values import residual_projection

    s, cov = residual_projection(disc.coh.logcpm, disc.genes, covs, genes, w, groups=disc.patient)
    log("G18 consensus projection coverage", cov)
    return s


def g18_fold_sign_check(disc: Discovery, s) -> dict:
    from scipy.stats import spearmanr

    key = {f"HEST-{a}-{b}": i for i, (a, b) in enumerate(zip(disc.section, np.asarray(disc.coh.raw.barcode).astype(str)))}
    out = {}
    for f in range(6):
        z = np.load(G_SCORES.format(f), allow_pickle=True)
        j = list(z["program_ids"]).index("G18")
        rows = [(key[k], v) for k, v in zip(z["spot_keys"], z["scores"][:, j]) if k in key]
        ix = np.array([r for r, _ in rows])
        out[f"outer_{f}"] = float(spearmanr(s[ix], [v for _, v in rows])[0])
    return out


# ----------------------------------------------------------------------------- common steps
def characterisation_common(disc, state, target, *, constituents, gene_lists, involves_space, lr_genes,
                            own_loss=None, out_dir, pathologist_packet: bool = False):
    cv = disc.cellvit()
    ok = np.isfinite(state)
    refs = C.reference_sets(disc.sets)
    prog = np.asarray(disc.coh.raw.programmes, float)
    rec = {"constituents": constituents}
    rec["enrichment"] = C.enrichment(gene_lists, disc.genes, refs, score=state[ok], programme_scores=prog[ok],
                                     programme_names=[str(n) for n in disc.coh.raw.programme_names])
    rec["enrichment"]["adjacent_known_biology"] = rec["enrichment"]["adjacent_known_biology"][:15]
    loc = C.localisation(state[ok], target[ok], disc.section[ok], _reindex_nb(disc.nb6, ok), cv["strata"][ok],
                         coords=np.asarray(disc.coh.raw.coords, float)[ok], out_dir=f"{out_dir}/maps")
    rec["localization"] = loc
    rec["scale_decomposition"] = own_loss or {"status": "skipped", "reason": "no neighbour/contrast version or morphology candidate"}
    rec["squidpy"] = C.neighbourhood(state[ok], cv["strata"][ok], disc.coords_um[ok], disc.section[ok],
                                     moran_median=loc["moran_median"], involves_space=involves_space)
    cond = C.liana_condition(lr_genes, rec["squidpy"])
    X = disc.coh.logcpm
    rec["liana"] = C.liana_bivariate(X[np.flatnonzero(ok)], disc.genes, state[ok], disc.coords_um[ok], disc.section[ok],
                                     condition=cond)
    rec["liana"]["condition"] = cond[1]
    if pathologist_packet:  # 1.7 scratched by the owner (2026-10-21): optional, off by default, never generated
        reg = C.regions(disc.coords_um[ok], disc.section[ok], state[ok], cv["fractions"][ok])
        reg["spot_rows"] = [np.flatnonzero(ok)[r].tolist() for r in reg["spot_rows"]]
        m = C.matched_high_low(reg)
        rec["pathologist_read"] = {**C.pathologist_packet(reg, m, f"{out_dir}/pathologist_packet", disc.patch_loader),
                                   "match": m, "reading": "optional; not part of the readiness list"}
    rec["centre_labels"] = locks.not_assessed("no morphology-centre constituent; the pathologist session that would "
                                              "label centres was scratched")
    return rec


def _reindex_nb(nb, ok):
    new = -np.ones(len(ok), int)
    new[ok] = np.arange(ok.sum())
    sub = nb[ok]
    return np.where(sub >= 0, new[np.maximum(sub, 0)], -1)


def novelty_step(root, fid, char_rec, char_hash, target_label, *, higher_order, cache_dir, out_dir):
    labels = N.constituent_labels({"constituents": char_rec["constituents"]}, target_label)
    qs = N.generate_queries(labels)
    lk = N.write_literature_lock(root, fid, char_hash, labels, qs, date=time.strftime("%Y-%m-%d"))
    res = N.run_queries(root, fid, qs, labels, cache_dir=cache_dir, offline=False)
    tl = N.humanise(target_label) if target_label else None
    draft = N.draft_grade(res["records"], labels, tl, higher_order=higher_order)
    form = N.grading_form(fid, res["records"], labels, tl, draft)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/grading_form.json").write_text(json.dumps(_json(form), indent=1))
    out = {"query_log": res["query_log"], "record_ids": [r.get("doi") or r.get("id") for r in res["records"]],
           "grade": {"draft": draft["draft_grade"], "final": "pending two reviewers"},
           "secondary_tags": draft["secondary_tags_draft"], "reviewer_agreement": "pending",
           "literature_lock_hash": lk["lock_hash"], "n_screened_records": len(res["records"]),
           "supporting_record_ids_draft": draft["supporting_record_ids"][:50], "form": f"{out_dir}/grading_form.json"}
    locks.write_record(root, fid, "literature", _json(out))
    return out


def therapeutic_step(root, fid, genes, occupancy, cache_root):
    dm = T.DepMap(cache_root=cache_root)
    otv = T.open_targets_version(cache_root)
    mondo = T.resolve_mondo(otv["cache_dir"])
    locks.write_lock(root, fid, "therapeutic", inputs={"genes": list(genes)},
                     declared_analysis={"depmap": dm.version_record(), "open_targets": otv, "disease": mondo,
                                        "rules": {"context_dependency": "kidney-lineage (RCC) median Chronos <= -0.5 and not "
                                                                        "common essential", "actionable_in_principle":
                                                  "tractability label at or above discovery precedence",
                                                  "clinically_precedented": "drug at phase >= 2 in any indication"}})
    dep = [dm.gene(g) for g in genes]
    ot = [T.ot_gene(g, otv["cache_dir"], mondo["disease_id_used"]) for g in genes]
    out = {"depmap": dep, "open_targets": ot, "depmap_release": dm.version_record(), "platform_release": otv,
           "relevance_statement": T.relevance_statement(fid, dep, ot, occupancy), "disease_ids": mondo}
    locks.write_record(root, fid, "therapeutic", _json(out))
    return out


# ----------------------------------------------------------------------------- slide cohorts
def slide_predictions(students: dict, paths, ids_fn, abmil=None, max_tiles=4096):
    """Per slide: mean of tile predictions for tile students, attention-pooled for the bag student."""
    out = {k: {} for k in students}
    out["_mean_embedding"] = {}
    if abmil is not None:
        out["gated_attention"] = {}
    rng = np.random.default_rng(0)
    for p in paths:
        z = np.load(p, allow_pickle=True)
        E = np.asarray(z["embeddings"], np.float32)
        pid = ids_fn(p, z)
        out["_mean_embedding"][pid] = E.astype(np.float64).mean(0)
        for k, s in students.items():
            out[k][pid] = float(np.mean(s.predict(E)))
        if abmil is not None:
            sub = E[rng.choice(len(E), min(max_tiles, len(E)), replace=False)]
            out["gated_attention"][pid] = float(abmil.predict_bags([sub])[0])
    return out


def distillation_step(root, fid, disc, state, refs: dict, out_dir):
    """refs: {'tcga': pd.Series (patient -> bulk score), 'cptac_rna': ..., 'cptac_protein': ...}."""
    import pandas as pd

    ok = np.isfinite(state)
    E = np.asarray(disc.coh.raw.hoptimus, np.float32)
    cov = getattr(disc.coh.raw, "covered", np.ones(len(state), bool))
    ok &= np.asarray(cov, bool) & np.isfinite(E).all(1)
    E, y, g = E[ok], state[ok], disc.patient[ok]
    locks.write_lock(root, fid, "distillation", inputs={"n_spots": int(ok.sum()), "state_hash": locks.canonical_hash(np.round(y, 6).tolist())},
                     declared_analysis={"students": ["ridge", "mlp_512_gelu_dropout0.2", "gated_attention_regions_1mm"],
                                        "baselines": D.BASELINE_STATUS, "metrics": "7.5", "ppi": "7.6",
                                        "success": "best student pooled Spearman > 0.3, CI excludes permuted null, "
                                                   "within-site-centred Spearman above STPath and DeepSpot"})
    res = {"baselines": D.BASELINE_STATUS}
    log("distill: held-out ridge")
    res["discovery_ridge"] = {k: v for k, v in D.heldout_spot_eval(D.RidgeStudent, E, y, g, n_perm=3).items() if k != "_pred"}
    log("distill: held-out mlp")
    res["discovery_mlp"] = {k: v for k, v in D.heldout_spot_eval(D.MLPStudent, E, y, g, n_perm=0).items() if k != "_pred"}
    cv = disc.cellvit()
    Xcv = np.column_stack([cv["fractions"], cv["count"], cv["area"]])[ok]
    res["discovery_cellvit_ridge"] = {k: v for k, v in D.heldout_spot_eval(D.RidgeStudent, Xcv, y, g).items() if k != "_pred"}
    res["discovery_cellvit_ridge"]["descriptors"] = "7 spot-level CellViT descriptors (5 PanNuke fractions, count, mean area); the 106-descriptor set is not computed per spot on the store"
    log("distill: fit final students")
    ridge = D.RidgeStudent().fit(E, y, g)
    mlp = D.MLPStudent().fit(E, y, g)
    bags, bt, bg = D.region_bags(E, y, disc.coords_um[ok], disc.section[ok], g)
    abm = D.GatedAttentionStudent().fit(bags, bt, bg)
    res["gated_attention_training"] = {"n_region_bags": len(bags), "epochs": abm.epochs}
    log("distill: TCGA slides")
    tp = sorted(glob.glob(TCGA_TILES))
    tcga = slide_predictions({"ridge": ridge, "mlp": mlp}, tp, lambda p, z: "-".join(os.path.basename(p).split("-")[:3]), abm)
    log("distill: CPTAC slides")
    cp = sorted(glob.glob(CPTAC_TILES))
    cptac = slide_predictions({"ridge": ridge, "mlp": mlp}, cp, lambda p, z: "-".join(os.path.basename(p).split("-")[:2]), abm)
    # permuted-target null at slide level: ridge refit on permuted spot targets, 20 draws
    rng = np.random.default_rng(7)
    E_tcga_mean = tcga.pop("_mean_embedding")
    cptac.pop("_mean_embedding")
    pids = sorted(set(E_tcga_mean) & set(refs["tcga"].index))
    Em = np.vstack([E_tcga_mean[p] for p in pids])
    sites = [D.site_of(p) for p in pids]
    null = [D.spearman(D.RidgeStudent().fit(E, rng.permutation(y), g).predict(Em), refs["tcga"][pids].values)
            for _ in range(20)]
    res["tcga"] = {}
    for k, preds in tcga.items():
        pr = np.array([preds.get(p, np.nan) for p in pids])
        res["tcga"][k] = D.slide_cohort_eval(pr, refs["tcga"][pids].values, sites, null=null)
    best = max(res["tcga"].values(), key=lambda v: v["pooled_spearman"])
    res["tcga_null"] = {"n": 20, "mean": float(np.mean(null)), "q95": float(np.quantile(null, 0.95))}
    res["success"] = D.success_rule(best, {"STPath": {}, "DeepSpot": {}})
    res["cptac"] = {}
    for refname in ("cptac_rna", "cptac_protein"):
        for k, preds in cptac.items():
            ids = sorted(set(preds) & set(refs[refname].index))
            pr = np.array([preds[i] for i in ids])
            res["cptac"][f"{k}_vs_{refname}"] = {"n": len(ids), "spearman": D.spearman(pr, refs[refname][ids].values)}
    # original-paper protocol on the stored embeddings (TCGA-internal ridge on slide means)
    res["original_protocol_tcga_internal"] = D.tcga_internal_detector(Em, refs["tcga_legacy"][pids].values, sites)
    res["original_protocol_tcga_internal"]["original_numbers"] = ORIGINAL
    # 7.6: PPI for the mean TCGA score (labelled = patients with RNA; unlabelled = all slides)
    allp = sorted(tcga["ridge"])
    lab = [p for p in allp if p in refs["tcga"].index]
    fr = np.array([tcga["ridge"][p] for p in lab])
    yr = refs["tcga"][lab].values
    half = lab[: len(lab) // 2]
    cal = np.polyfit([tcga["ridge"][p] for p in half], refs["tcga"][half].values, 1)
    rest = lab[len(lab) // 2:]
    res["ppi_tcga_mean_score"] = D.ppi_mean(refs["tcga"][rest].values, np.polyval(cal, [tcga["ridge"][p] for p in rest]),
                                             np.polyval(cal, [tcga["ridge"][p] for p in allp]))
    res["ppi_note"] = ("every TCGA slide patient has RNA, so the labelled subset is the second half of the RNA patients "
                       "and the student is put on the bulk-score scale with the first half; this demonstrates 7.6, it "
                       "is not a new estimate")
    res["site_decomposition"] = {k: {kk: v[kk] for kk in ("pooled_spearman", "loso_calibrated_spearman",
                                                           "within_site_centred_spearman")} for k, v in res["tcga"].items()}
    pd.DataFrame({k: pd.Series(v) for k, v in tcga.items()}).to_csv(f"{out_dir}/tcga_slide_predictions.csv")
    locks.write_record(root, fid, "distillation", _json(res))
    return res


# ----------------------------------------------------------------------------- G18
def run_g18(root, out, cache):
    fid = "legacy-ccRCC-G18"
    fdir = f"{out}/{fid}"
    genes, w = bulk.frozen_loading("G18")
    lhash = locks.sha256_file(bulk.LOADINGS)
    card = locks.StateCard(root, fid)
    ident = {"finding_id": fid, "cohort": "ccRCC (HEST Meylan 2022, INT1-INT24)", "target": "state: residual programme G18",
             "candidates": ["residual:G18@v0"], "relation_params_hash": lhash,
             "registry_hash": "legacy (original paper; no learned-search registry)"}
    stage(root, fid, "registry_freeze", {"loadings_npz_sha256": lhash}, {"legacy": True}, lambda: ident)
    card.update("identity", ident)
    val = {"gain_nuis": locks.not_assessed("legacy state (residual programme), not a relation S -> y"),
           "gain_known": locks.not_assessed("legacy state"), "gain_decomp": locks.not_assessed("legacy state"),
           "discovery_labels": ["legacy: frozen six-fold consensus residual programme of the original paper"],
           "null_yield_at_rank": locks.not_assessed("no registry rank")}
    stage(root, fid, "internal_validation", {}, {"legacy": True}, lambda: val)
    card.update("validation", val)
    rep = {"per_cohort": locks.not_assessed("v03 tests relations S -> y; G18 is a state"), "pooled": None,
           "direction_agreement": None, "q": None, "platform_transformation": "n/a",
           "rule_hash": None, "status": "not applicable"}
    stage(root, fid, "external_replication", {}, {"rule": "v03 not applicable to a state"}, lambda: rep)
    card.update("replication", rep)

    disc = Discovery()
    state = g18_spot_scores(disc)
    fold_check = g18_fold_sign_check(disc, state)
    log("G18 spot scores", np.isfinite(state).sum(), fold_check)
    order = np.argsort(w)
    top = {"positive": [genes[i] for i in order[::-1][:50]], "negative": [genes[i] for i in order[:50]]}
    constituents = {"status": "run", "rows": [{"candidate": "residual:G18@v0", "class": "residual programme",
                                               "spatial_version": "v0", "partial_association_sign": 1,
                                               "standardised_contribution_heldout": "n/a (state, not a relation)",
                                               "loaded_genes": C.top_loaded(genes, w)}]}

    def _char():
        return characterisation_common(disc, state, state, constituents=constituents, gene_lists=top,
                                       involves_space=False, lr_genes=top["positive"][:25] + top["negative"][:25],
                                       out_dir=fdir)
    log("G18 characterisation")
    ch = stage(root, fid, "characterisation", {"state": "frozen consensus loading on the covariate-adjusted spot residual",
                                                "fold_score_sign_check": fold_check},
               {"procedures": "1.1-1.7", "enrichment_lists": "top 50 positive / negative loaded genes"}, _char)
    card.update("characterization", _json(ch))
    log("G18 novelty")
    nov = novelty_step(root, fid, ch, locks.read_record(root, fid, "characterisation")["record_hash"], None,
                       higher_order=False, cache_dir=f"{cache}/literature", out_dir=fdir)
    card.update("novelty", nov)
    th_genes = top["positive"][:25] + top["negative"][:25]
    otv = T.open_targets_version(cache)
    did = T.resolve_mondo(otv["cache_dir"])["disease_id_used"]
    handle = any(T.ot_gene(g, otv["cache_dir"], did).get("handle_sm_or_ab") for g in th_genes)
    pr = T.prioritise([{"id": fid, "external_ci_lower": None, "higher_order": False, "grade": nov["grade"]["draft"],
                        "has_handle": handle, "external_gain": None}])
    stage(root, fid, "prioritisation", {}, {"rule": "Section 4"}, lambda: {"ranking": pr})
    log("G18 orthogonal")

    def _orth():
        mp = {s["name"]: s["genes"] for s in disc.sets["mp"]}
        r = bulk.finding_bulk_support("residual", loading=(genes, w), sets_mp=mp)
        cp = bulk.cptac_tables()
        lr, _ = bulk.legacy_rankz_score(cp["rna"], genes, w)
        lp, _ = bulk.legacy_rankz_score(cp["protein"], genes, w)
        r["legacy_rankz_rna_protein"] = {**bulk.spearman(lr, lp), "original": ORIGINAL["cptac_rna_protein"]}
        return r
    orth = stage(root, fid, "orthogonal", {"loadings_npz_sha256": lhash},
                 {"score": "frozen loading on covariate-adjusted bulk residual; legacy rank-z for reproduction"}, _orth)
    card.update("orthogonal", {"cptac_rna_protein": orth.get("cptac_rna_protein"), "tcga_bulk_score": orth["tcga_rna"],
                               "legacy_reproduction": orth["legacy_rankz_rna_protein"]})
    log("G18 therapeutic")
    th = therapeutic_step(root, fid, th_genes, ch["localization"]["top_decile_occupancy"], cache)
    card.update("therapeutic", {k: th[k] for k in locks.CARD_SECTIONS["therapeutic"]})
    log("G18 distillation")
    tcga = bulk.tcga_tables()
    refs = {"tcga": orth["_scores"]["tcga_rna"], "cptac_rna": orth["_scores"]["cptac_rna"],
            "cptac_protein": orth["_scores"]["cptac_protein"],
            "tcga_legacy": bulk.legacy_rankz_score(tcga["rna"], genes, w)[0]}
    ds = distillation_step(root, fid, disc, state, refs, fdir)
    card.update("distillation", {"per_model": {"tcga": ds["tcga"], "cptac": ds["cptac"],
                                               "discovery": {k: ds[k] for k in ("discovery_ridge", "discovery_mlp",
                                                                                "discovery_cellvit_ridge")},
                                               "baselines": D.BASELINE_STATUS},
                                 "success": ds["success"], "site_decomposition": ds["site_decomposition"],
                                 "original_protocol": ds["original_protocol_tcga_internal"]})
    log("G18 clinical")
    cl = clinical_step(root, fid, genes, w, orth["_scores"]["tcga_rna"])
    card.update("clinical", cl)
    card.set_partition_manifest("legacy (original-paper six outer folds)")
    return card


def clinical_step(root, fid, genes, w, tcga_score):
    j = K.load_javelin()
    score_def = {"programme": "G18", "loading_sha256": locks.sha256_file(bulk.LOADINGS),
                 "javelin_arithmetic": "legacy rank-z over JAVELIN patients dot frozen loading (as the original paper)",
                 "tcga_arithmetic": "frozen loading on covariate-adjusted bulk residual (Section 5)"}
    resumed = locks.lock_path(root, fid, "clinical").exists()
    lk = locks.require_lock(root, fid, "clinical") if resumed else K.write_clinical_lock(root, fid, cohort="JAVELIN Renal 101 + TCGA-KIRC", endpoint="PFS (JAVELIN); OS (TCGA)",
                               score_definition=score_def, covariates=["age", "sex", "TCGA: stage"],
                               model="JAVELIN: Cox PFS ~ arm + score/SD + arm x score/SD + age + sex; TCGA: Cox OS ~ "
                                     "score/SD + stage + age + sex (grade unavailable)",
                               family=[fid])
    js, _ = bulk.legacy_rankz_score(j["expr"], genes, w)
    inter = K.interaction_test(j["clinical"], js.values, binary_median=False, covariates=("age", "male"))
    inter_unadj = K.interaction_test(j["clinical"], js.values, binary_median=False)
    c = K.tcga_clinical()
    ids = sorted(set(c.index) & set(tcga_score.index))
    cc = c.loc[ids]
    prog = K.prognostic_test(cc["OS_time"].values, cc["OS_event"].values, tcga_score[ids].values,
                             {k: cc[k].values for k in K.TCGA_COVARIATES})
    known = K.known_programme_comparison(js.values, j["expr"], j["signatures"])
    from .replication_v03 import bh

    q = bh([inter["interaction"]["p"], prog["hr_per_sd"]["p"]])
    out = {"lock": lk["declared_analysis"], "hazard_ratios": {"javelin_interaction_adjusted": inter,
                                                               "javelin_interaction_unadjusted": inter_unadj,
                                                               "tcga_os": prog},
           "q": {"javelin_interaction": float(q[0]), "tcga_os": float(q[1])},
           "incremental_discrimination": {"tcga_delta_c": prog.get("delta_c"), "tcga_lr": prog["lr_statistic"]},
           "known_programme_correlations": known,
           "original": {"javelin_interaction_hr": ORIGINAL["javelin_interaction_hr"], "ci": ORIGINAL["javelin_interaction_ci"]},
           "javelin_pipeline_test": K.reproduce_javelin_published(j), "availability": K.AVAILABILITY,
           "resumed_under_existing_lock": ("the first attempt stopped on a scoring bug (zero-variance JAVELIN genes "
                                           "gave NaN scores, no model was fitted); the lock and its declared analysis "
                                           "are unchanged") if resumed else None}
    locks.write_record(root, fid, "clinical", _json(out))
    return _json(out)


# ----------------------------------------------------------------------------- Glutathione relation
def run_glutathione(root, out, cache):
    from . import discovery, replication_v03 as V

    fid = "kidney-v05-687cd597c626"
    cands = ["context_neighbour_mean:nbr:hallmark:HALLMARK_FATTY_ACID_METABOLISM"]
    tgt = "own:Glutathione"
    cat = bknown.covariate_sets()
    d = discovery.discovery_arrays(cands, tgt, cat)
    ff, aux = discovery.freeze_finding(fid, cands, tgt, d, {"finding_hash": "renal-spatial-v05 funnel 687cd597c626"})
    card = locks.StateCard(root, fid)
    ident = {"finding_id": fid, "cohort": "ccRCC (HEST Meylan 2022)", "target": tgt, "candidates": cands,
             "relation_params_hash": ff.rel_nuis.params_hash(), "registry_hash": "renal-spatial-v05-20261014 funnel (legacy)"}
    stage(root, fid, "registry_freeze", {"relation_params_hash": ff.rel_nuis.params_hash()},
          {"frozen_relation": ff.rel_nuis.to_json()}, lambda: ident)
    card.update("identity", ident)
    val = {"gain_nuis": locks.not_assessed("legacy v0.5 receipts (internal stage), not re-scored by the new harness"),
           "gain_known": locks.not_assessed("legacy"), "gain_decomp": locks.not_assessed("single candidate"),
           "discovery_labels": ["legacy v0.5 ledger stage: internal; v02 external: pass li2022 + gse250163"],
           "null_yield_at_rank": locks.not_assessed("legacy")}
    stage(root, fid, "internal_validation", {}, {"legacy": True}, lambda: val)
    card.update("validation", val)
    log("Glutathione replication v03")
    res = V.run_family([ff], root, ["li2022_ccrcc", "gse250163", "zenodo14620362"], catalogue=cat)[fid]
    card.update("replication", _json({"per_cohort": res["per_cohort"], "pooled": res["pooled"],
                                      "direction_agreement": res["direction"], "q": res["bh_q"],
                                      "platform_transformation": "v03 rule 2.3", "rule_hash": res["rule_hash"],
                                      "externally_replicated": res["externally_replicated"]}))
    card.doc["dry_run_beyond_stop"] = not res["externally_replicated"]
    card.doc["headline_eligible"] = bool(res["externally_replicated"])
    card.save()
    disc = Discovery()
    state = np.full(len(disc.section), np.nan)
    state[:] = ff.rel_nuis.predict(aux["Zs"])
    from .values import cohort_values

    own = cohort_values(disc.coh, "hallmark:HALLMARK_FATTY_ACID_METABOLISM@v0", disc.sets, disc.coh.panel)
    from .relation import standardise_within

    own_s, _ = standardise_within(own, disc.patient)
    log("Glutathione characterisation")

    def _char():
        cons = C.constituent_table(ff.rel_nuis, aux["Zs"], aux["r_nuis"], aux["patient"])
        loss = C.own_version_loss(ff.rel_nuis, aux["Zs"], aux["r_nuis"], aux["patient"], {cands[0]: own_s.ravel()})
        cv = disc.cellvit()
        ok = cv["covered"]
        oba = C.organisation_beyond_abundance(state[ok], cv["fractions"][ok], cv["count"][ok], cv["area"][ok],
                                              _reindex_nb(disc.nb6, ok), disc.section[ok], disc.patient[ok], n_null=99)
        fa = [s for s in disc.sets["hallmark"] if s["name"] == "HALLMARK_FATTY_ACID_METABOLISM"][0]["genes"]
        gl = [s for s in disc.sets["mp"] if s["name"] == "Glutathione"][0]["genes"]
        rec = characterisation_common(disc, state, aux["ys"], constituents=cons, gene_lists={"constituents": fa},
                                      involves_space=True, lr_genes=list(fa) + list(gl),
                                      own_loss={"own_version": loss, "organisation_beyond_abundance": oba,
                                                "status": "run"}, out_dir=f"{out}/{fid}")
        return rec
    ch = stage(root, fid, "characterisation", {"relation_params_hash": ff.rel_nuis.params_hash()},
               {"procedures": "1.1-1.7", "dry_run_beyond_stop": not res["externally_replicated"]}, _char)
    card.update("characterization", _json(ch))
    log("Glutathione novelty")
    nov = novelty_step(root, fid, ch, locks.read_record(root, fid, "characterisation")["record_hash"], tgt,
                       higher_order=False, cache_dir=f"{cache}/literature", out_dir=f"{out}/{fid}")
    card.update("novelty", nov)
    fa_genes = [s for s in disc.sets["hallmark"] if s["name"] == "HALLMARK_FATTY_ACID_METABOLISM"][0]["genes"][:25]
    pr = T.prioritise([{"id": fid, "external_ci_lower": res["pooled"]["nuis"]["ci"][0], "higher_order": False,
                        "grade": nov["grade"]["draft"], "has_handle": True, "external_gain": res["pooled"]["nuis"]["gain"]}])
    stage(root, fid, "prioritisation", {}, {"rule": "Section 4"}, lambda: {"ranking": pr})
    log("Glutathione therapeutic")
    th = therapeutic_step(root, fid, fa_genes, ch["localization"]["top_decile_occupancy"], cache)
    card.update("therapeutic", {k: th[k] for k in locks.CARD_SECTIONS["therapeutic"]})
    for sec in ("orthogonal", "distillation", "clinical"):
        locks.write_lock(root, fid, sec, inputs={}, declared_analysis={"dry_run": "integration of lock order only"})
    card.update("orthogonal", {"cptac_rna_protein": locks.not_assessed("neighbour-version relation: bulk cannot carry "
                                                                         "the spatial constituent"),
                               "tcga_bulk_score": locks.not_assessed("as above")})
    card.update("distillation", {"per_model": locks.not_assessed("dry run: G18 carries the distillation test"),
                                 "success": None, "site_decomposition": None})
    card.update("clinical", {"lock": None, "hazard_ratios": locks.not_assessed("dry run"), "q": None,
                             "incremental_discrimination": None, "known_programme_correlations": None})
    return card


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="glutathione,g18")
    a = ap.parse_args()
    if "runs/prospective" in os.path.abspath(a.out):
        raise SystemExit("refusing to write under runs/prospective")
    root = f"{a.out}/cards"
    cache = "runs/development/downstream_v01/cache"
    summary = {}
    if "glutathione" in a.only:
        c = run_glutathione(root, a.out, cache)
        summary["glutathione"] = c.completeness()
    if "g18_clinical" in a.only:
        fid = "legacy-ccRCC-G18"
        rec = locks.read_record(root, fid, "orthogonal")
        genes, w = bulk.frozen_loading("G18")
        mp = {s["name"]: s["genes"] for s in __import__("morpheus.tier2.substrates.hest_multi", fromlist=["x"])._programme_sets()["mp"]}
        r = bulk.finding_bulk_support("residual", loading=(genes, w), sets_mp=mp)
        card = locks.StateCard(root, fid)
        card.update("clinical", clinical_step(root, fid, genes, w, r["_scores"]["tcga_rna"]))
        card.set_partition_manifest("legacy (original-paper six outer folds)")
        summary["g18"] = card.completeness()
        summary["g18_orthogonal_record_hash"] = rec["record_hash"]
    elif "g18" in a.only:
        c = run_g18(root, a.out, cache)
        summary["g18"] = c.completeness()
    Path(f"{a.out}/summary.json").write_text(json.dumps(summary, indent=1))
    log("done", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
