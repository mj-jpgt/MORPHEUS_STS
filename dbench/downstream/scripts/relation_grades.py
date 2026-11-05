"""Relation-level grades (amendment v08b), grader R. Applies the frozen decision procedure of
dbench/protocols/relation_novelty_v08b.json to the grader's per-record judgements (JUDGED below: every prior-art
record cited with its level and reason; every other read record is logged with the default reason).

Interpretation notes (logged in the output, applied identically to every relation):
 * levels 1-3 are judged for the relation's dependency as a whole (target with its constituent set, or with the single
   constituent label when the relation has one); a record about one constituent of a multi-constituent relation is an
   'individual_constituent_association' (level 4 evidence), following the owner's wording of categories 3 and 4;
 * a constituent whose label equals or shares a term with the target label (same process, e.g. G1/S and HMG-rich
   proliferation programmes, or a stroma fraction predicting a stroma-like morphology centre) cannot be resolved by
   text retrieval; it is counted as individually associated ('same-process') and the relation is tagged same_process;
 * 'HO None' (no higher-order test recorded for the relation) is treated as not passing;
 * a residual-with-HO sensitivity reading (residual relations that pass the higher-order test read as compositional
   novelty) is reported beside the primary grade; the primary grade follows the frozen residual rule.

    python -m dbench.downstream.scripts.relation_grades OUT_DIR
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = "runs/development/relation_regrade"
DEFAULT_REASON = ("not prior art: read (title+abstract); no sentence states a dependency between the target and a "
                  "constituent (co-mention, co-enrichment in a list of pathways, shared response to an intervention, "
                  "shared cell type or pathway membership)")
# grader R judgements: {relation id prefix: {"c": {label substring: (doi, reason)}, "l1": (doi, reason), "l2": ...,
#                                           "joint3": (doi, reason)}}
JUDGED = {
    "crc:contrast:hallmark:HALLMARK_GLYCOLYSIS": {"c": {
        "mTOR": ("10.1038/s41420-025-02888-w", "mTORC1 activation accelerates glycolysis in CRC cells (LBX2)"),
        "MYC": ("10.62347/okyw9037", "MYC-driven glycolytic reprogramming in CRC (GTPBP4)")}},
    "crc:contrast:hallmark:HALLMARK_HYPOXIA": {"c": {
        "epithelial-mesenchymal": ("10.1016/j.cellsig.2026.112847", "hypoxia-induced EMT via HIF-1a/VEGF in CRC"),
        "p53": ("10.1186/s12885-026-16184-y", "p53-HIF interaction under hypoxia in CRC cells")}},
    "crc:contrast:hallmark:HALLMARK_OXIDATIVE_PHOSPHORYLATION": {"c": {
        "glycolysis": ("10.1038/s41598-026-43173-9", "reciprocal glycolysis vs OXPHOS reprogramming in CRC cells; "
                       "pairwise spatial form also reported (CAF glycolysis fuels OXPHOS in adjacent CRC cells, "
                       "10.1016/j.yexcr.2025.114867) but not the joint relation")}},
    "crc:mp:Cell Cylce HMG-rich:d7a2": {"c": {
        "hypoxia": ("10.1016/j.tranon.2026.102817", "hypoxia-induced proliferation in CRC (TDP-43/HIF1)"),
        "MYC": ("10.3389/fcell.2025.1687485", "HGF-MET-MYC signalling drives CRC proliferation")}},
    "crc:mp:EMT-III": {"c": {
        "PDAC-related": ("10.3390/diagnostics14141512", "mucin family regulates EMT in CRC")}},
    "crc:mp:MYC:34a6": {"c": {
        "proliferation": ("10.3389/fonc.2025.1680561", "c-MYC drives CRC proliferation (ZFP161)")}},
    "crc:mp:MYC:da27": {"c": {
        "G2/M": ("10.3389/fonc.2025.1680561", "c-MYC drives CRC proliferation (ZFP161)"),
        "E2F": ("10.1016/j.ccell.2026.09.014", "MYC-E2F transcriptional networks in MSS CRC")},
        "note": "joint MYC/E2F/G2M co-variation reported only in another disease without the spatial form "
                "(10.3389/fgene.2026.1878093, neuroblastoma proliferative-replication programme)"},
    "crc:mp:PDAC-classical": {"joint3": ("10.1186/s40880-019-0361-0", "mucinous CRC associated with MUC2 and MUC5AC "
                                         "mucin overexpression (joint, non-spatial)")},
    "oscc:1": {"c": {
        "AEBP1": ("10.3390/cancers15174303", "CAF-derived AEBP1/ACLP activates CAFs in OSCC"),
        "senescence": ("10.1016/j.ijrobp.2026.02.209", "HNSCC CAFs undergo senescence while retaining pro-tumour state"),
        "angiogenesis": ("10.1016/j.cellsig.2026.112409", "CTHRC1+ CAFs co-localise with and promote angiogenesis in HNSCC")}},
    "oscc:4": {"c": {
        "senescence": ("10.1021/acsnano.5c18165", "proliferation arrest and tumour-cell senescence coupled (CDK inhibition)")},
        "joint3": ("10.3390/cimb47070558", "second reading pass: senescence induction halts proliferation in OSCC cells; "
                   "with G1/S and G2/M same-process, this is the relation's dependency reported jointly, non-spatial")},
    "meylan:Metal-response_v2": {"c": {
        "epithelial-mesenchymal transition\" OR \"EMT\" OR \"cancer": ("10.1016/j.cbi.2026.112254", "second reading pass: MT2A enhances EMT in ccRCC"),
        "myofibroblast": ("10.1007/s12672-025-02949-7", "second reading pass: MT1X knockdown activates the EMT pathway in RCC (inverse)")}},
    "crc:contrast:mp:Hypoxia": {"c": {
        "metallothionein": ("10.1155/2021/2427427", "second reading pass: MT1E/MT2A differ between hypoxia-score groups in CRC")}},
    "oscc:16": {"c": {
        "proliferation": ("10.21203/rs.3.rs-5443141/v1", "second reading pass: LDHA knockdown reduces HNSCC proliferation")}},
    "oscc:5": {"c": {
        "proliferation": ("10.22088/ijmcm.bums.10.1.45", "c-MYC knockdown controls OSCC proliferation")}},
    "oscc:7": {"c": {
        "oxidative": ("10.1002/advs.76385", "CAF contact induces EMT-like programmes and mitochondrial transfer raises "
                      "OXPHOS in budding OSCC cells at the tumour-stroma interface")}},
    "p2:valdeolivas2024_crc:programme:mp:Cell Cycle - G2/M": {"c": {
        "glycolysis": ("10.62347/ossn5233", "HKDC1 promotes CRC proliferation through glycolysis and cell-cycle pathways"),
        "MYC": ("10.3389/fonc.2025.1680561", "c-MYC drives CRC proliferation")},
        "note": "relation holds on normal-appearing mucosa; prior art is in tumour tissue"},
    "p3:budding_like": {"c": {
        "stroma": ("10.1007/s00330-026-12922-9", "fibroblast activation (FAPI PET) associated with tumour budding in CRC")}},
    "p4:meylan2022:44e2c067268e": {"c": {
        "IFI30": ("10.2174/0115748928367598250124123411", "IFI30 described as an interferon-gamma-induced protein in ccRCC")}},
    "p5:hallmark:HALLMARK_GLYCOLYSIS:b8257d93a451": {"c": {
        "stroma": ("10.1002/ags3.70004", "aerobic glycolysis in CAFs and metabolic coupling with colon cancer cells")}},
    "p5:hallmark:HALLMARK_HYPOXIA:21ccbbea17e1": {"c": {
        "classical": ("10.1007/s00535-026-02461-x", "hypoxia-mediated CA9 reprogramming in colorectal mucinous adenocarcinoma")}},
    "p5:morph:k03": {"c": {"stroma": ("10.1016/j.ebiom.2026.106397", "tumour cells and tumour stroma interaction in rectal cancer")}},
    "p5:morph:k20": {"c": {"stroma": ("10.1016/j.ebiom.2026.106397", "tumour cells and tumour stroma interaction in rectal cancer")}},
    "p5:morph:k04": {"c": {"LGALS1": ("10.18632/oncotarget.25845", "galectin-1 increased in stroma-high CRC")}},
    "p5:mp:Cell Cycle - G1/S:3685": {"c": {"MYC": ("10.1007/s00403-025-03964-y", "MYC methylation inhibits cSCC proliferation (TINCR)")}},
    "p5:mp:Cell Cycle - G2/M:0caec7cc86b8": {"c": {"MYC": ("10.3389/fonc.2025.1680561", "c-MYC drives CRC proliferation")}},
    "p5:mp:Cell Cylce HMG-rich:0ae0": {"c": {"MYC": ("10.3389/fonc.2025.1680561", "c-MYC drives CRC proliferation")}},
    "p5:mp:Interferon/MHC-II (I):2cdc4ba2f98d": {"c": {
        "IL-6": ("10.1016/j.canlet.2025.218209", "lactate/IL-6 secretion inversely related to HLA-DR+ myeloid cells in RCC")}},
    "p5:mp:MYC:874ea88fdb77": {"l1": ("10.21203/rs.3.rs-8852518/v1", "primary CRC spatial transcriptomics: hyper-"
                                     "proliferative regions enriched for E2F and MYC segregated from CAF-enriched "
                                     "stromal interfaces (joint dependency, spatial form, same disease)")},
    "meylan:IFN-MHCII-I_C1QB_v2": {"l2": ("10.1038/s41421-025-00811-2", "follow-up full-text/reading check: in "
        "colorectal cancer, C1QC+ resident-like macrophages co-localise with CD4+ T cells and MHC-II expression mediates "
        "their interaction (Cell Discov 2025, abstract and full text); with the pan-cancer review PMC13672365 (C1QC+ "
        "antigen-presenting TAMs localise to T cell-rich interfaces). The joint spatial dependency (C1Q macrophage "
        "neighbourhood + T cells -> MHC-II antigen presentation) is reported in another disease; no ccRCC record reports "
        "it (owner-supplied ccRCC records checked: not prior art). First pass had required the exact TRBC2 symbol for "
        "the T-cell constituent; the judgement is semantic (TRBC2 = T-cell receptor beta constant)")},
    "p5:mp:Proteasomal degradation:aebb": {"c": {"MYC": ("10.1016/j.yexcr.2026.114944", "c-MYC ubiquitin-proteasome degradation in CRC")}},
    "p5:mp:Proteasomal degradation:dde0": {"c": {"MYC": ("10.1016/j.yexcr.2026.114944", "c-MYC ubiquitin-proteasome degradation in CRC")}},
}
OWNER = {
    "meylan:IFN-MHCII-I_C1QB_v2": [
        ("10.1016/j.cell.2021.04.038", "not prior art (abstract; full text not open access): TREM2/APOE/C1Q+ TAM subpopulation, spatially validated, recurrence-associated; no MHC-II or T-cell neighbourhood dependency stated (shared cell type)"),
        ("10.3389/fgene.2023.1109991", "not prior art (full text): C1QC prognostic and correlated with immune infiltration in KIRC, highest in macrophages; no MHC-II/interferon-programme dependency"),
        ("10.1136/jitc-2022-006195", "not prior art (full text): geospatial immune distributions in ccRCC; no C1Q sentence"),
        ("Song et al. JITC 2025 (C1Q macrophage populations across ccRCC regions)", "not located in Europe PMC by author/title/topic search; logged, not judged; DOI requested"),
        ("10.3389/fgene.2023.1207233", "full-text check: interferon-regulatory genes and C1QA/B/C highly expressed in ccRCC TAMs (shared cell type); association of cell-type markers, not the dependency")],
    "meylan:IFN-MHCII-II_v2": [
        ("10.1016/j.cell.2021.04.038", "not prior art: C1Q+ TAM state; Astrocytes/Metal-response/Secreted II constituents absent"),
        ("10.3389/fgene.2023.1109991", "not prior art: as above")],
    "meylan:Metal-response_v2": [
        ("10.7150/ijbs.117535", "association between constituents reported (FKBP10 promotes EMT in ccRCC, with spatial transcriptomics); metallothionein (the target) not reported -> not prior art for the relation"),
        ("10.1007/s12672-026-05303-7", "not prior art (full text): ccRCC metabolic heterogeneity / lactylation review; no metallothionein-FKBP10/EMT dependency")],
    "meylan:Glutathione_sign-only_v2": [
        ("10.1038/s41588-024-01662-5", "not prior art (full text): ccRCC metabolic reprogramming with progression; no glutathione-ANPEP/OXPHOS dependency sentence"),
        ("10.1186/s12967-024-04848-x", "not prior art (full text): glutathione-high (AA+) and OXPHOS-high macrophage subclasses are distinct states; co-mention"),
        ("10.1007/s12672-026-05303-7", "not prior art (full text): review; no glutathione dependency")],
}
CONSERVED_MEANING = {"59b472a9f62f", "6e645d9788a9", "2cdc4ba2f98d", "0caec7cc86b8", "dde0f3870460", "aebb42950c58",
                     "52c4bf5c044d", "21ccbbea17e1"}   # decoded meaning preserved (R's P5 decoding)


def terms(q):
    return [t.strip().strip('"').lower() for t in re.split(r"\s+OR\s+", q.strip("()"))]


def judged_for(rid):
    best = None
    for k, v in JUDGED.items():
        ok = rid == k or (rid.startswith(k) and not (k.startswith("oscc:") or k.startswith("tcga:")))
        if ok and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else {}


def main():
    out = Path(sys.argv[1])
    rels = {r["id"]: r for r in json.load(open(f"{R}/relations_frozen.json"))}
    rows = []
    for f in sorted(Path(f"{R}/retrieval").glob("retrieval_*.json")):
        if f.name == "retrieval_index.json":
            continue
        ret = json.loads(f.read_text())
        rid = ret["id"]
        rel = rels[rid]
        j = judged_for(rid)
        tt = set(terms(ret["target_query"]))
        spatial = any(v.split(":")[0] in ("nbr6", "nbr18", "contrast") for v in rel["variables"])
        # constituent labels, including those equal to / overlapping the target label (same process)
        labs = {}
        for v in rel["variables"]:
            q = (rel.get("labels") or {}).get(v)
            if q is None:
                from dbench.downstream.scripts.relation_regrade import label_of, split_var
                P = json.load(open("dbench/protocols/relation_novelty_v08b.json"))
                q = label_of(split_var(v)[1], P)[0]
            labs.setdefault(q, []).append(v)
        indiv, evidence, same_proc = {}, [], []
        for q, vs in labs.items():
            if q == ret["target_query"] or (set(terms(q)) & tt):
                indiv[q] = "same-process"
                same_proc.append(q)
                continue
            hit = [(k, d) for k, d in (j.get("c") or {}).items() if k.lower() in q.lower()]
            if hit:
                indiv[q] = "associated"
                evidence.append({"record": hit[0][1][0], "level": "individual_constituent_association",
                                 "constituent": q, "reason": hit[0][1][1]})
            else:
                indiv[q] = "none"
        HO = rel.get("higher_order") is True
        l1, l2, j3 = j.get("l1"), j.get("l2"), j.get("joint3")
        for lv, e in (("same_dependency_same_spatial_same_disease", l1), ("same_dependency_other_disease", l2),
                      ("constituent_association_same_disease_not_spatial", j3)):
            if e:
                evidence.append({"record": e[0], "level": lv, "reason": e[1]})
        allind = all(v in ("associated", "same-process") for v in indiv.values()) and len(indiv) > 0
        anyind = any(v in ("associated", "same-process") for v in indiv.values())
        if l1:
            cat = "1 no novelty identified"
        elif l2:
            cat = "2 context extension"
        elif j3:
            cat = "3 spatial extension" if spatial else "1 no novelty identified"
        elif allind and HO:
            cat = "4 compositional novelty"
        elif not any(v == "associated" for v in indiv.values()) and not same_proc:
            cat = "5 new association"
        elif not anyind:
            cat = "5 new association"
        else:
            cat = "3 spatial extension" if spatial else "1 no novelty identified"
        residual = not (l1 or l2 or j3) and anyind and not (allind and HO)
        sens = ("4 compositional novelty" if residual and HO else cat)
        k12 = (rel.get("candidate_id") or "")
        k12 = k12[-12:] if isinstance(k12, str) else ""
        novel = not cat.startswith("1")
        ti = rel.get("tier_input", "")
        if not novel:
            tier = "not reported (no novelty identified)"
        elif str(ti).startswith("Confirmed"):
            organs = set(rel.get("organs_stat_conserved") or [])
            org_n = len({{"meylan2022_ccrcc": "kidney", "kidney_injury": "kidney", "ependymoma_pfa": "brain",
                          "gse237183_glioma": "brain", "valdeolivas2024_crc": "colon", "ji2020_cscc": "skin",
                          "gse208253_oscc": "oral"}.get(o, o) for o in organs})
            tier = "Conserved" if (org_n >= 3 and k12 in CONSERVED_MEANING) else "Confirmed"
        else:
            tier = "Candidate"
        if rid.startswith("tcga:failure"):
            tier = "Confirmed (CPTAC replication reported by grader T; gain-above-null not re-verified by R)" if novel else tier
        organs = rel.get("organs_stat_conserved") or []
        org_set = {{"meylan2022_ccrcc": "kidney", "kidney_injury": "kidney", "ependymoma_pfa": "brain",
                    "gse237183_glioma": "brain", "valdeolivas2024_crc": "colon", "ji2020_cscc": "skin",
                    "gse208253_oscc": "oral"}.get(o, o) for o in organs}
        tags = []
        if len(org_set) >= 3 and k12 in CONSERVED_MEANING:
            tags.append("Conserved")
        cited = {e["record"] for e in evidence}
        log = [{"record": x["doi"] or x["id"], "title": x["title"], "year": x["year"],
                "verdict": ("prior art (see evidence)" if (x["doi"] or x["id"]) in cited else DEFAULT_REASON)}
               for x in ret["records_read"]]
        rows.append({
            "id": rid, "phenomenon": rel.get("phenomenon") or [], "study": rel["study"],
            "frozen_relation": {"target": rel["target"], "variables": rel["variables"],
                                "spatial_versions": [v.split(":")[0] if v.split(":")[0] in ("nbr6", "nbr18", "contrast")
                                                     else "own" for v in rel["variables"]],
                                "row_restriction": rel.get("row_restriction"), "candidate_id": rel.get("candidate_id"),
                                "p4_context": rel.get("p4"), "higher_order": rel.get("higher_order")},
            "labels": {"target": ret["target_query"], "constituents": list(labs)},
            "decoded_axes": rel.get("decoded_meaning") or rel.get("label_decoding") or None,
            "tier_input": ti, "tier": tier, "primary_grade": cat,
            "sensitivity_grade_residual_HO_as_compositional": sens,
            "secondary_tags": tags, "statistically_conserved_organs": sorted(org_set) if organs else None,
            "same_process_constituents": same_proc, "individual_association": indiv,
            "prior_art_evidence": evidence, "grader_note": j.get("note"),
            "n_records_retrieved": ret["n_records_retrieved"], "n_not_read_no_comention": ret["n_not_read_no_comention"],
            "records_considered": log, "alias_sensitivity": "unchanged (HGNC alias retrieval added no prior-art record "
                                                            "that changes the grade)",
            "owner_supplied_records": [{"record": a, "verdict": b} for a, b in OWNER.get(rid, [])],
            "grader_agreement": "pending (grader T blind; compare when both are done)"})
    out.mkdir(parents=True, exist_ok=True)
    doc = {"protocol": "dbench/protocols/relation_novelty_v08b.json",
           "protocol_canonical_hash": "d0fc44d2a8f46f7eee143df7fbf244de9f726b126533c2e0852d54a730fb1cc3",
           "grader": "R", "interpretation_notes": __doc__, "relations": rows}
    (out / "relation_regrade.json").write_text(json.dumps(doc, indent=1, default=str))
    c = Counter((r["tier"].split(" (")[0], r["primary_grade"]) for r in rows)
    print(len(rows))
    for k, v in sorted(c.items()):
        print(k, v)
    print(Counter(r["primary_grade"] for r in rows))
    print(Counter(r["sensitivity_grade_residual_HO_as_compositional"] for r in rows))


if __name__ == "__main__":
    main()
