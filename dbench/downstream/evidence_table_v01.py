#!/usr/bin/env python
"""Evidence table + per-relation reporting records, strictly from receipts (re-runnable).

Usage: python evidence_table_v01.py [--root ~/morpheus_work_v06] [--tx ~/morpheus_work_tx] [--out DIR]
Reads inputs read-only, hashes every input at build time, writes evidence_table.{csv,parquet,md},
relation_records.json, summary_numbers.json, input_sha256.json, validate_errors.json.
"""
from __future__ import annotations
import argparse, glob, hashlib, importlib.util, json, math, os, statistics
from collections import Counter, defaultdict
import pandas as pd

NA = "not_assessed"
COLS = ["relation_id", "dedup_group", "stage", "family", "target", "constituents", "test_layer", "test",
        "cohort", "status", "estimate", "ci_lo", "ci_hi", "p", "q", "mde", "pct_retained", "role",
        "positive_control_passed", "source_file", "source_sha256",
        "variant", "value", "grade_stage", "n_patients", "reason", "receipt"]
NUM = ["estimate", "ci_lo", "ci_hi", "p", "q", "mde", "pct_retained"]
ID_PREFIX = "renal-spatial-v08-multiscale-20261026-"
HIGHLIGHT = ("71ed4499fb99", "67256d9e53d5", "93b02a9038ea", "caf32d321d27")
HNOTE = {"b99a9e88476d": "externally confirmed; internal confounder and power assessments underpowered (MDE stated)",
         "f837f7cc8f8b": "externally confirmed; internal confounder and power assessments underpowered (MDE stated)",
         "1d4493b06b8f": "externally confirmed; recovered-known category, not highlighted",
         "042b776f122b": "externally confirmed; recovered-known category, not highlighted"}
FORCE_CANDIDATE = ("9ee19a9c22f0", "416936937c6c")
P_FAM = "runs/confirmation/v08_external_family_bh_v01/family_bh_masked.json"
CH = "runs/development/characterisation_v10/"
NV = "runs/development/novelty_v10/"
LEDGER = "v08:runs/registered_v08/renal-spatial-v08-multiscale-20261026/renal_spatial_v08_multiscale_ledger.jsonl"
TXR = "runs/development/translation_v10/"


class Ctx:
    def __init__(self, a):
        self.root = os.path.expanduser(a.root)
        self.tx = os.path.expanduser(a.tx)
        self.out = os.path.expanduser(a.out)
        self.v08 = os.path.expanduser(a.v08)
        self.sha = {}

    def path(self, p):
        if p.startswith("tx:"):
            return os.path.join(self.tx, p[3:])
        if p.startswith("v08:"):
            return os.path.join(self.v08, p[4:])
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def hashed(self, p):
        fp = self.path(p)
        if fp not in self.sha:
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            self.sha[fp] = h.hexdigest()
        return fp

    def js(self, p):
        with open(self.hashed(p)) as f:
            return json.load(f)

    def jsonl(self, p):
        with open(self.hashed(p)) as f:
            return [json.loads(l) for l in f if l.strip()]

    def short(self, p):
        fp = self.path(p)
        for base, tag in ((self.root, "v06:"), (self.tx, "tx:"), (self.v08, "v08reg:")):
            if fp.startswith(base + os.sep):
                return tag + fp[len(base) + 1:]
        return fp

    def sha_of(self, p):
        return self.sha[self.path(p)]


def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else float("nan")


def row(ctx, src, **kw):
    r = {c: NA for c in COLS}
    r.update(kw)
    for c in NUM:
        r[c] = num(r[c])
    r["source_file"] = ctx.short(src)
    r["source_sha256"] = ctx.sha_of(src)
    return r


def nz(v, default=NA):
    return default if v is None or v == "" else v


def ps_(ok):
    return "pass" if ok else "fail"


def agg(statuses):
    if not statuses:
        return NA
    if any(x == "fail" for x in statuses):
        return "fail"
    if all(x == "pass" for x in statuses):
        return "pass"
    return NA


def med(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return {"median": statistics.median(v) if v else NA, "n": len(v)}


def build(ctx):
    fam = ctx.js(P_FAM)
    ev = ctx.jsonl(CH + "evidence_rows.jsonl")
    ctx.js(CH + "positive_controls.json")
    ctx.js(CH + "deviations.json")
    d1 = {os.path.basename(f): ctx.js(f) for f in sorted(glob.glob(ctx.path(CH + "d1/*.json")))}
    grades = {s: ctx.js(NV + f"grades_stage{s}.json") for s in "123"}
    collapsed = ctx.js(NV + "grades_stage3_collapsed_scale.json")
    rk = ctx.js(NV + "recovered_known_v01.json")
    ctx.js(NV + "headline_candidates_stage3_N3_N4.json")
    T = "tx:" + TXR
    trel = ctx.js(T + "relations.json")
    util = ctx.js(T + "utility_labels.json")
    tsum = ctx.js(T + "summary_counts.json")
    ctx.js(T + "positive_controls.json")
    ctx.hashed(T + "deviations.log")
    layers = {}
    for f in sorted(glob.glob(os.path.join(ctx.tx, TXR, "layer_*.json"))):
        name = os.path.basename(f)[6:-5]
        layers[name] = ("tx:" + TXR + os.path.basename(f), ctx.js("tx:" + TXR + os.path.basename(f)))
    ledger = {}
    for l in ctx.jsonl(LEDGER):
        prev = ledger.get(l["candidate_id"])
        if prev is None or l["stage"] == "internal" or prev["stage"] != "internal":
            ledger[l["candidate_id"]] = l   # last internal entry wins, else last proposed
    ctx.hashed("dbench/protocols/UNIFIED_BIOLOGICAL_DISCOVERY_NOVELTY_V1.md")
    ud_path = ctx.hashed("dbench/unified_discovery.py")
    spec = importlib.util.spec_from_file_location("unified_discovery", ud_path)
    ud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ud)

    group_of = {}
    for gi, g in enumerate(fam["dedup_groups"], 1):
        for m in g["members"]:
            group_of[m] = f"G{gi}"
    cands = {}
    for c in fam["confirmatory"]:
        cands[c["candidate_id"]] = (c, "replicated" if c["replicated"] else "internally_supported")
    for c in fam["exploratory_proposed_only"]["table"]:
        cands[c["candidate_id"]] = (c, "proposed_only")
    meta = {rid: dict(stage=st, family=c["family"], target=c["target"], constituents="|".join(c["candidates"]))
            for rid, (c, st) in cands.items()}
    for r in trel:
        if r["relation_id"] not in meta:
            st = "internally_supported" if r.get("tier") in ("Confirmed", "Conserved") else "proposed_only"
            meta[r["relation_id"]] = dict(stage=st, family="legacy:%s:%s" % (r.get("source"), r.get("set")),
                                          target=r.get("target", NA), constituents="|".join(r.get("variables") or []))
    for d in d1.values():
        meta["D1:" + d["target"]] = dict(stage=NA, family="D1_residual_programme", target=d["target"], constituents=NA)
    for rid in {r["relation_id"] for r in ev}:
        meta.setdefault(rid, dict(stage=NA, family=NA, target=NA, constituents=NA))
    trel_by = {r["relation_id"]: r for r in trel}

    def base(rid):
        m = meta.get(rid) or dict(stage=NA, family=NA, target=NA, constituents=NA)
        return dict(relation_id=rid, dedup_group=group_of.get(rid, "ungrouped"), stage=m["stage"], family=m["family"],
                    target=m["target"], constituents=m["constituents"])

    rows = []
    ps = lambda ok: "pass" if ok else "fail"
    # ---- replication layer
    for rid, (c, st) in cands.items():
        b = base(rid)
        npc = c.get("n_patients_per_cohort") or {}
        b.update(test_layer="replication", cohort="|".join(sorted(npc)) or NA, receipt=nz(c.get("receipt")),
                 n_patients=sum(npc.values()) or NA)
        if c["status"] != "tested":
            for t in ("ext_gain_beyond_nuisance", "ext_family_bh_q", "ext_direction_agreement",
                      "ext_gain_beyond_known_biology", "ext_gain_beyond_decomposition", "ext_replicated"):
                rows.append(row(ctx, P_FAM, **b, test=t, status="not_testable", reason=c["status"]))
            continue
        cr, g, q = c["criteria"], c["gain"], c.get("family_bh_q")
        rows.append(row(ctx, P_FAM, **b, test="ext_gain_beyond_nuisance", status=ps(cr["ci_excludes_zero"]),
                        estimate=g["nuis"]["gain"], ci_lo=g["nuis"]["ci95"][0], ci_hi=g["nuis"]["ci95"][1],
                        p=g["nuis"]["p_boot"], q=q, role="replication_criterion"))
        rows.append(row(ctx, P_FAM, **b, test="ext_family_bh_q", status=ps(cr["bh_q_le_0.05"]), estimate=q, q=q,
                        p=c.get("batch_bh_q"), role="replication_criterion"))
        rows.append(row(ctx, P_FAM, **b, test="ext_direction_agreement", status=ps(cr["direction_agreement_ge_0.60"]),
                        estimate=c["direction_pooled"], role="replication_criterion"))
        for t, k in (("ext_gain_beyond_known_biology", "known"), ("ext_gain_beyond_decomposition", "decomp")):
            lo = g[k]["ci95"][0]
            rows.append(row(ctx, P_FAM, **b, test=t, status=ps(lo > 0), estimate=g[k]["gain"], ci_lo=lo,
                            ci_hi=g[k]["ci95"][1], p=g[k]["p_boot"], role="informational",
                            reason="status = CI lower bound > 0; not a replication criterion"))
        rows.append(row(ctx, P_FAM, **b, test="ext_replicated", status=ps(c["replicated"]), role="decision",
                        estimate=float(c["replicated"])))
    # ---- characterisation layer
    CHF = CH + "evidence_rows.jsonl"
    for r in ev:
        ci = r.get("ci") or [None, None]
        t = r["test"]
        coh = ("external_no_refit" if t in ("known_biology_external_norefit", "composition_external") else
               "external" if t == "d1_external_recovery" else "discovery_internal_LOPO")
        pr = r.get("pct_retained", r.get("pct_of_original"))
        var = r.get("variant") or r.get("variable") or r.get("tubule_marker_only") or NA
        pc = r.get("positive_control_passed")
        rows.append(row(ctx, CHF, **base(r["relation_id"]), test_layer="characterisation", test=t, cohort=coh,
                        status=nz(r.get("status")), estimate=r.get("estimate"), ci_lo=ci[0], ci_hi=ci[1], p=r.get("p"),
                        q=r.get("q"), mde=r.get("mde"), pct_retained=pr, role=nz(r.get("role")),
                        positive_control_passed=NA if pc is None else str(pc), variant=str(var),
                        grade_stage=str(r.get("stage")), reason=nz(r.get("reason")), value=nz(r.get("grader_reads"))))
    # ---- novelty layer
    rk_by = {x["relation_id"]: x for x in rk["rows"]}
    RKF = NV + "recovered_known_v01.json"
    for s, g in grades.items():
        gf = NV + f"grades_stage{s}.json"
        for x in g["relations"]:
            rid = x["relation_id"]
            common = dict(test_layer="novelty", cohort="literature", status="not_applicable", grade_stage=s)
            for gl in "ABC":
                gd = x.get("grader_" + gl)
                if gd:
                    rows.append(row(ctx, gf, **base(rid), **common, test="novelty_grader_" + gl, value=gd["category"],
                                    variant="provisional" if gd.get("provisional") else "final", role="grader_category"))
            rows.append(row(ctx, gf, **base(rid), **common, test="novelty_resolved_category", value=x["resolved_category"],
                            role="resolved", reason=x["resolution_rule"]))
            rows.append(row(ctx, gf, **base(rid), **common, test="novelty_needs_adjudication",
                            value=str(x["needs_adjudication"]), role="adjudication_flag"))
            rows.append(row(ctx, gf, **base(rid), **common, test="novelty_checker_tier",
                            value=str(x.get("checker_tier")), role="checker"))
            if rid in rk_by:
                rows.append(row(ctx, RKF, **base(rid), **common, test="recovered_known_category",
                                value=rk_by[rid]["category"], role="recovered_known"))
    # ---- translation layers
    for L, (fp, d) in layers.items():
        for r in d:
            pc = r.get("positive_control_passed")
            rows.append(row(ctx, fp, **base(r["relation_id"]), test_layer="translation_" + L, test=r["test"],
                            cohort=nz(r.get("cohort")), status=nz(r.get("status")), estimate=r.get("estimate"),
                            ci_lo=r.get("ci_low"), ci_hi=r.get("ci_high"), p=r.get("p"), q=r.get("q"), mde=r.get("mde"),
                            role="not_applicable", positive_control_passed=NA if pc is None else str(pc),
                            variant=nz(r.get("assay")), n_patients=nz(r.get("n_patients")), reason=nz(r.get("reason"))))
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS:
        if c not in NUM:
            df[c] = df[c].astype(str)
    recs, omitted = build_records(ctx, ud, cands, group_of, ev, grades, collapsed, trel_by, util, df, d1, ledger)
    summ = build_summary(df, recs, ev, grades, collapsed, cands, rk, tsum, omitted)
    return df, recs, summ, ud


def build_records(ctx, ud, cands, group_of, ev, grades, collapsed, trel_by, util, df, d1, ledger):
    evby = defaultdict(lambda: defaultdict(list))
    for r in ev:
        evby[r["relation_id"]][r["test"]].append(r)
    layer_stat = defaultdict(lambda: defaultdict(list))
    for r in df[df.test_layer.str.startswith("translation_")].itertuples():
        layer_stat[r.relation_id][r.test_layer].append(r.status)
    kap = {}
    for s, g in grades.items():
        k = {"kappa_AB_full_scale": g["kappa_all_items"], "kappa_relations": g["kappa_relations"],
             "exact_agreement_all": g["exact_agreement_all"], "n_items": g["n_items"], "graders": g["graders"]}
        if "pairwise" in g:
            k["pairwise_full_scale"] = g["pairwise"]
            k["collapsed_scale"] = {kk: collapsed[kk] for kk in ("AB", "AC", "BC")}
            k["collapsed_scale_share_N2_or_higher"] = collapsed["share_N2_or_higher"]
            k["collapsed_scale_n_items"] = collapsed["n_items"]
        else:
            k["pairwise_full_scale"] = NA
            k["collapsed_scale"] = NA
        kap[s] = k
    DT = {"compartment_context": ["D5_context_dependent_modulation"]}
    recs, omitted = {}, []

    def ext_ctx(rid):
        c = cands.get(rid, (None, None))[0]
        if c is None or c["status"] != "tested":
            return NA, NA
        pc = c.get("direction_per_cohort") or {}
        npc = c.get("n_patients_per_cohort") or {}
        return ([{"cohort": k, "direction": v, "n_patients": npc.get(k)} for k, v in pc.items()],
                sum(1 for v in pc.values() if v >= 0.6))

    UT = (("Biological", "independent_phenotype_information", "translation_phenotype"),
          ("Assay-translatable", "external_assay_fidelity", "translation_assay"),
          ("Clinically associated", "independent_outcome_association", "translation_clinical"))
    for s, g in grades.items():
        for x in g["relations"]:
            rid = x["relation_id"]
            c, st = cands[rid]
            cat = x["resolved_category"]
            gl = [k for k in "ABC" if x.get("grader_" + k)]
            pick = next((k for k in gl if x["grader_" + k]["category"] == cat), gl[0])
            sent = x["grader_" + pick]["novel_property_sentence"]
            kb = NA
            for k in [pick] + [k for k in gl if k != pick]:
                cp = x["grader_" + k].get("closest_prior_record")
                if cp:
                    kb = dict(cp, from_grader=k, from_resolved_grader=(k == pick))
                    break
            rows = evby[rid]
            gms = rows.get("gms_claim_matched", [None])[0]
            kbr = rows.get("known_biology", [None])[0]
            reads = kbr.get("grader_reads") if kbr else None
            if gms and gms.get("role") == "decisive":
                dtest, drow = "gms_claim_matched", gms
            else:
                dtest, drow = "known_biology", kbr
            if reads == "external":
                dtest, drow = "known_biology_external_norefit", rows["known_biology_external_norefit"][0]
            decisive = ({"test": dtest, "estimate": drow.get("estimate"), "patient_level_ci95": drow.get("ci"),
                         "q": drow.get("q"), "status": drow.get("status"), "role": drow.get("role"),
                         "grader_reads": reads or NA} if drow else NA)
            tier = {"replicated": "Externally confirmed", "internally_supported": "Internally supported"}.get(st, "Candidate")
            if rid.endswith(FORCE_CANDIDATE):
                tier = "Candidate"
            provisional = (s == "3") or bool(x.get("provisional_N4"))
            contexts, cons = ext_ctx(rid)
            a = {"frozen_definition": "inherited", "patient_disjoint_evaluation": "inherited"}
            for t, rs in rows.items():
                sts = [r["status"] for r in rs]
                a[t] = "fail" if "fail" in sts else ("pass" if all(v == "pass" for v in sts) else NA)
            led = (ledger.get(rid) or {}).get("evidence")
            pf = [str(f) for f in (led or {}).get("passing_folds", [])]
            npass = len(pf)
            internal_rule = NA
            sb_worst = NA
            if led:
                ci_ok = bool(pf) and all(led["fold_gain_ci_lo"].get(f, -1) > 0 for f in pf)
                perm = max((led["null_p_by_fold"][f]["within_patient_perm"] for f in pf), default=1.0)
                sb_worst = max((led["null_p_by_fold"][f]["spatial_block"] for f in pf), default=1.0)
                internal_rule = ps_(npass >= 3 and ci_ok and perm <= 0.05 and led["q"] <= 0.10)
            a["statistical_validity"] = internal_rule
            a["claim_matched_spatial_ablation"] = a.get("gms_claim_matched", NA)
            if c["family"] == "heterogeneity_gradient":
                pr = [r for r in rows.get("het_matched_null", []) if r.get("n_null") == 199 or "n_null" not in r]
                pr = pr[0] if pr else None
                a["spatial_calibrated_null"] = (NA if not pr or pr.get("p") is None or pr.get("status") == "not_assessed"
                                                else ps_(pr["p"] <= 0.05))
                het_detail = ({"primary_row": "het_matched_null", "n_null": pr.get("n_null"), "p": pr.get("p"),
                               "q_across_relations": pr.get("q"), "row_status": pr.get("status"),
                               "sensitivity_49_p": (rows.get("het_matched_null_49_sensitivity") or [{}])[0].get("p"),
                               "rule": "raw p <= 0.05 on the primary 199-draw row"} if pr else NA)
            else:
                a["spatial_calibrated_null"] = NA if sb_worst == NA else ps_(sb_worst <= 0.05)
            if c["family"] != "heterogeneity_gradient":
                het_detail = NA
            dirn = c.get("direction_pooled") if c["status"] == "tested" else None
            a["patient_level_reproducibility"] = (NA if not led or dirn is None else ps_(npass >= 3 and dirn >= 0.60))
            tc, cp = a.get("technical", NA), a.get("composition", NA)
            a["confounder_robustness"] = ("pass" if tc == "pass" and cp == "pass" else
                                          NA if NA in (tc, cp) else "fail")
            own = (rows.get("ladder_d0") or [{}])[0].get("estimate")
            mde = drow.get("mde") if drow else None
            if mde is not None and own is not None and mde <= own:
                a["detection_power"] = "pass"
            else:
                a["detection_power"] = NA
            a["external_replication"] = "pass" if c["replicated"] else ("fail" if c["status"] == "tested" else NA)
            ut = (util.get(rid) or {}).get("utility")
            for lab, req, lay in UT:
                ls = layer_stat[rid].get(lay, [])
                if ut is not None and lab in ut:
                    a[req] = "inherited" if lab == "Biological" else "pass"
                elif ls and ut is not None and not any(v == "pass" for v in ls):
                    a[req] = "fail"
                else:
                    a[req] = NA
            a["validated_incremental_outcome_prediction"] = NA
            a["clinical_decision_benefit"] = NA
            a["causal_functional_evidence"] = NA
            recs[rid] = {
                "object_id": rid, "source_relation_id": rid, "record_kind": "graded_v08",
                "discovery_types": DT.get(c["family"], ["D4_spatial_multiscale_dependency"]),
                "specific_novelty_claim": sent, "claim_source_grader": pick,
                "primary_novelty": cat,
                "novelty_adjudication": "provisional" if provisional else "adjudicated",
                "novelty_provisional": provisional, "needs_adjudication": x["needs_adjudication"],
                "grader_categories": {k: x["grader_" + k]["category"] for k in gl},
                "resolution_rule": x["resolution_rule"], "grade_stage": s, "grader_kappa": kap[s],
                "evidence_tier": tier, "stage": st, "decisive_test": dtest, "decisive_statistic": decisive,
                "known_biology_alternative": kb, "assessments": a,
                "control_status_detail": {t: [r["status"] for r in rs] for t, rs in rows.items()},
                "external_contexts": contexts, "conservation_count": cons,
                "external_contexts_text": ("externally replicated (exploratory family)" if (st == "proposed_only" and c["replicated"]) else
                                           "externally replicated (confirmatory family)" if st == "replicated" else
                                           "tested externally, not replicated" if c["status"] == "tested" else "not testable externally"),
                "internal_rule_detail": ({"passing_folds_of_5": npass, "q": led["q"], "worst_within_patient_perm_p": perm,
                                          "worst_spatial_block_p": sb_worst, "ci_lower_bounds_gt_0": ci_ok,
                                          "ledger_source": "v08 registered ledger"} if led else NA),
                "spatial_null_detail": het_detail,
                "reproducibility_detail": {"passing_folds_of_5": npass, "external_direction_agreement": dirn},
                "detection_power_note": {"decisive_mde": mde, "own_nuisance_gain_ladder_d0": own},
                "utility_labels": ut if ut is not None else [],
                "utility_status": "utility_labels.json" if ut is not None else NA,
                "legacy_mapping": False, "highlighted": rid.endswith(HIGHLIGHT), "highlight_note": HNOTE.get(rid[-12:], NA), "family": c["family"], "target": c["target"],
                "dedup_group": group_of.get(rid, "ungrouped"),
                "exploratory_replicated": bool(st == "proposed_only" and c["replicated"]),
            }
    for rid, r in trel_by.items():
        if rid in recs:
            continue
        src = dict(r)
        if rid in cands:
            c, st = cands[rid]
            src["tier"] = "Candidate" if st == "proposed_only" else src.get("tier")
            src["external_replication"] = {"pooled": True} if st == "replicated" else None
        rec = ud.legacy_record(src)
        ut = (util.get(rid) or {}).get("utility", [])
        contexts, cons = ext_ctx(rid)
        rec.update(record_kind="legacy_or_ungraded", novelty_provisional=NA, grader_kappa=NA, external_contexts=contexts,
                   conservation_count=cons, utility_labels=ut, target=r.get("target"),
                   note="ungraded or legacy relation: validator legacy mapping; novelty not imputed beyond it")
        if rec["evidence_tier"] == "Externally confirmed":
            rec["assessments"]["external_replication"] = "pass"
        for lab, req, _ in UT:
            if lab in ut:
                rec["assessments"][req] = "inherited" if lab == "Biological" else "pass"
        recs[rid] = rec
    for d in d1.values():
        omitted.append({"object_id": "D1:" + d["target"],
                        "reason": "no novelty grade in novelty_v10; no controlled-vocabulary primary_novelty without imputing"})
    return recs, omitted


def build_summary(df, recs, ev, grades, collapsed, cands, rk, tsum, omitted):
    S = {"n_rows": int(len(df)), "row_counts_per_layer": df.test_layer.value_counts().to_dict(),
         "n_relations_in_table": int(df.relation_id.nunique()), "n_records": len(recs),
         "records_by_kind": dict(Counter(r["record_kind"] for r in recs.values()))}
    g = [r for r in recs.values() if r["record_kind"] == "graded_v08"]
    S["counts_stage_x_novelty_x_tier"] = dict(Counter(f"{r['stage']}|{r['primary_novelty']}|{r['evidence_tier']}" for r in g))
    ac = {}
    for f in ("statistical_validity", "claim_matched_spatial_ablation", "spatial_calibrated_null",
              "patient_level_reproducibility", "confounder_robustness", "detection_power", "external_replication"):
        cnt = Counter()
        for r in g:
            v = r["assessments"][f]
            if f == "statistical_validity" and v == "fail" and r["stage"] == "proposed_only":
                v = "not internally supported (exploratory family)"
            cnt[v] += 1
        ac[f] = dict(cnt)
    S["assessment_counts_graded_v08"] = ac
    S["highlighted_records"] = [r["object_id"] for r in g if r["highlighted"]]
    S["graded_by_stage"] = dict(Counter(r["stage"] for r in g))
    S["all_records_by_tier"] = dict(Counter(r["evidence_tier"] for r in recs.values()))
    lad = defaultdict(dict)
    for r in ev:
        if r["test"].startswith("ladder_"):
            lad[r["relation_id"]][r["test"]] = r.get("pct_retained")
    S["median_pct_retained_by_stage"] = {}
    for st in ("replicated", "internally_supported", "proposed_only"):
        ids = [r["object_id"] for r in g if r["stage"] == st]
        S["median_pct_retained_by_stage"][st] = {
            "n_relations": len(ids),
            "beyond_composition(ladder_d2)": med([lad[i].get("ladder_d2") for i in ids]),
            "beyond_known_biology(ladder_d1)": med([lad[i].get("ladder_d1") for i in ids]),
            "beyond_both(ladder_d3)": med([lad[i].get("ladder_d3") for i in ids])}
    S["grader_kappas"] = {s: {"kappa_AB_full_scale": gg["kappa_all_items"], "kappa_relations": gg["kappa_relations"],
                              "exact_agreement_all": gg["exact_agreement_all"], "n_items": gg["n_items"],
                              "pairwise_full_scale": gg.get("pairwise", NA),
                              "collapsed_scale": ({k: collapsed[k] for k in ("AB", "AC", "BC", "n_items", "share_N2_or_higher")}
                                                  if s == "3" else NA)} for s, gg in grades.items()}
    S["checker_false_known_rates"] = {s: gg["false_known_rate_beside_every_grade"] for s, gg in grades.items()}
    S["replication_family"] = {
        "n_confirmatory": sum(1 for c, st in cands.values() if st != "proposed_only"),
        "n_replicated_confirmatory": sum(1 for c, st in cands.values() if st == "replicated"),
        "n_proposed_only": sum(1 for c, st in cands.values() if st == "proposed_only"),
        "n_proposed_only_replicated": sum(1 for c, st in cands.values() if st == "proposed_only" and c["replicated"])}
    S["recovered_known_by_category"] = rk["summary"]["by_category"]
    S["status_counts_per_layer_test"] = {f"{lay}|{t}": gdf.status.value_counts().to_dict()
                                         for (lay, t), gdf in df.groupby(["test_layer", "test"])}
    S["status_counts_per_layer"] = {lay: gdf.status.value_counts().to_dict() for lay, gdf in df.groupby("test_layer")}
    S["translation_summary_counts_input"] = tsum.get("counts")
    S["records_omitted"] = omitted
    return S


def to_md(df):
    d = df.copy()
    for c in NUM:
        d[c] = d[c].map(lambda v: NA if v != v else f"{v:.4g}")
    cols = [c for c in COLS if c not in ("source_sha256", "constituents", "receipt")]
    d = d[cols].copy()
    d["source_file"] = d["source_file"].str.replace(r".*/", "", regex=True)
    d["relation_id"] = d["relation_id"].str.replace(ID_PREFIX, "v08:", regex=False)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rr in d.itertuples(index=False):
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in rr) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/morpheus_work_v06")
    ap.add_argument("--tx", default="~/morpheus_work_tx")
    ap.add_argument("--v08", default="~/morpheus_work_v08")
    ap.add_argument("--out", default="~/morpheus_work_v06/runs/confirmation/evidence_table_v01")
    ctx = Ctx(ap.parse_args())
    os.makedirs(ctx.out, exist_ok=True)
    df, recs, summ, ud = build(ctx)
    errs = {rid: ud.validate_record(r) for rid, r in recs.items()}
    errs = {k: v for k, v in errs.items() if v}
    summ["validate_record_errors"] = {"n_records_with_errors": len(errs), "n_records": len(recs), "errors": errs}
    out = lambda n: os.path.join(ctx.out, n)
    csvdf = df.copy()
    for c in NUM:
        csvdf[c] = csvdf[c].map(lambda v: NA if v != v else repr(v))
    csvdf.to_csv(out("evidence_table.csv"), index=False)
    df.to_parquet(out("evidence_table.parquet"), index=False)
    with open(out("evidence_table.md"), "w") as f:
        f.write(to_md(df))
    json.dump(recs, open(out("relation_records.json"), "w"), indent=1, default=str)
    json.dump(summ, open(out("summary_numbers.json"), "w"), indent=1, default=str)
    json.dump(errs, open(out("validate_errors.json"), "w"), indent=1)
    json.dump({ctx.short(k): v for k, v in sorted(ctx.sha.items())}, open(out("input_sha256.json"), "w"), indent=1)
    print("rows", len(df), df.test_layer.value_counts().to_dict())
    print("records", len(recs), "validate errors", len(errs))
    for k, v in errs.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
