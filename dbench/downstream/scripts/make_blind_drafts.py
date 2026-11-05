"""Blind-discovery protocol drafts (Agent R, 2026-10-21, owner directive ~19:00 UTC).

    PYTHONPATH=$PWD ~/morpheus_venv/bin/python -m dbench.downstream.scripts.make_blind_drafts

1. renal_spatial_v07.json and ependymoma_spatial_v01.json -> DRAFT r5: the external rule points at the
   downstream replication v03 protocol, a ``downstream`` block names the protocols that consume the run
   directory, ``registration`` lists every field filled at registration and the exact commands. The checkpoint
   path / sha256 stay 'to be filled at freeze' (never filled here).
2. New DRAFT protocols on the same template (hest_multi runner, learned_search + classical_pair, relation v07_ho,
   nuisance-only comparison, decomposition reported, all-fold pooling, pass rule as v0.5):
   cscc_spatial_v01 (Ji 2020), crc_spatial_v01 (Valdeolivas 2024), prostate_spatial_v01 (Kiviaho 2024,
   molecular + spatial only), ccrcc_li2022_spatial_v01 and ccrcc_gse250163_spatial_v01 (external ccRCC cohorts
   as discovery substrates, cross-cohort replication among Meylan / Li 2022 / GSE250163 / Zenodo).
Each file gets a canonical sha256 beside it (<name>.sha256). Nothing is written under runs/.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dbench.downstream.locks import canonical_hash

P = Path("dbench/protocols")
STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
EMB = f"{STORE}/external/hest_fm_embeddings_20261016"
CKPT_TODO = "to be filled at freeze"
ORDER_OF_OPERATIONS = [
    "1 checkpoint freeze: release.learned_search_checkpoint path + sha256 and the release commits filled by the coordinator",
    "2 historical-rediscovery scoring on this cohort completed and its result locked (historical cohorts only); ccRCC "
    "external cohorts: renal-spatial-v07 replication v03 run and recorded on them first",
    "3 programme targets fixed by name by the blind-targets procedure (Gavish meta-programmes ranked by median "
    "within-section Moran's I, all sections, response-free) and written into discovery.targets / relationship_families",
    "4 protocol status REGISTERED, canonical sha256 recorded; python -m dbench.prospective_campaign --protocol <file> "
    "--run-dir runs/prospective/<campaign_id> (manifest + empty ledger; ready_for_discovery must be true)",
    "5 fold sweep (dbench/scripts/blind_sweep.sh), --aggregate, cross-fold scoring (--crossfold-requests, "
    "--score-frozen), final --aggregate: hypotheses frozen and hashed in the ledger",
    "6 decomposition ladder on the frozen set (reported, never gated)",
    "7 replication v03 on the declared replication cohorts (rule hashed before external values)",
    "8 characterisation (literature closed), then the literature lock and the novelty pass (OpenAlex, Europe PMC) -- "
    "no literature is consulted at any earlier step",
]


def downstream_block(external: str, replication_cohorts: list) -> dict:
    return {"replication": external, "replication_cohorts": replication_cohorts,
            "consumer": "dbench/downstream/blind.py --run-dir runs/prospective/<campaign_id> --out "
                        "runs/development/downstream_blind/<campaign_id> (reads campaign_manifest.json, "
                        "funnel_aggregate.json, the ledger and receipts read only)",
            "characterisation": "downstream Section 1.1-1.6 (dbench.downstream.characterise; 1.7 pathologist read "
                                "scratched by the owner)",
            "novelty": "dbench.downstream.novelty (literature lock before any query)",
            "prioritisation": "dbench.downstream.therapeutic.prioritise (Section 4)",
            "distillation": "dbench/protocols/assay_distillation_v03.json (state (a) within-patient, state (b) "
                            "between-patient-inclusive; 7a on (b))",
            "b_known_catalogue": "dbench/downstream/data/bknown_ccrcc_v01.json (renal); other organs: Hallmark + "
                                 "Gavish only until an organ catalogue is declared"}


def registration_block(cid_stem: str, file: str) -> dict:
    return {"fields_filled_at_registration": [
        "campaign_id: '<stem>-DRAFT' -> '<stem>-<YYYYMMDD>'",
        "status: DRAFT -> REGISTERED",
        "release.learned_search_checkpoint.path / .sha256 (freeze)",
        "discovery.proposers.learned_search.checkpoint_path / checkpoint_sha256 (same values)",
        "cohorts[*].gates[id=learned_search_frozen_checkpoint].path (same path)",
        "release.relation_commit / propose_commit / loader_commit / measurement_bank_commit (the commit the sweep runs)",
        "programme targets by name where the draft says 'to be filled at registration'"],
        "commands": [f"python -m dbench.prospective_campaign --protocol dbench/protocols/{file} --run-dir "
                     f"runs/prospective/{cid_stem}-<YYYYMMDD>",
                     f"CAMPAIGN_PROTOCOL=dbench/protocols/{file} bash dbench/scripts/blind_sweep.sh "
                     f"runs/prospective/{cid_stem}-<YYYYMMDD> <workers>"],
        "never": "these fields are not filled by any agent before the freeze; no write under runs/prospective "
                 "before registration"}


def update_existing(name: str, external: str, replication_cohorts: list, stem: str) -> dict:
    d = json.loads((P / name).read_text())
    d["revision"] = ("DRAFT r5 (2026-10-21, Agent R): advance_rule.external -> replication v03 (downstream "
                     "Section 2); downstream and registration blocks; order_of_operations. ") + d["revision"]
    d["discovery"]["advance_rule"]["external"] = external
    d["downstream"] = downstream_block(external, replication_cohorts)
    d["registration"] = registration_block(stem, name)
    d["order_of_operations"] = ORDER_OF_OPERATIONS
    for c in d["cohorts"]:
        for g in c.get("gates", []):
            if g["id"] == "learned_search_frozen_checkpoint":
                g["path"] = CKPT_TODO
                g["note"] = "filled at freeze with release.learned_search_checkpoint.path (v01 frozen.pt retired)"
    return d


def new_draft(template: dict, *, stem: str, study: str, title: str, n_sections: int, n_patients: int,
              embed: str | None, external: str, replication_cohorts: list, notes: list, historical: str | None) -> dict:
    d = copy.deepcopy(template)
    d["campaign_id"] = f"{stem}-DRAFT"
    d["revision"] = "DRAFT r1 (2026-10-21, Agent R): same template as renal-spatial-v07 / ependymoma-spatial-v01 r5"
    d["supersedes"] = f"none: first blind-discovery campaign on {title}"
    d["status"] = ("DRAFT (not registered; runs only after the checkpoint freeze"
                   + (f" and after the historical-rediscovery scoring of task {historical} on this cohort is locked"
                      if historical else "") + ")")
    d["ledger"] = {"filename": f"{stem.replace('-', '_')}_ledger.jsonl"}
    gates = [{"id": "hest_multi_study", "path": "dbench/protocols/sprint_partitions_v01.json", "required": True,
              "note": f"study key {study}"}]
    if embed:
        gates.append({"id": "hoptimus_embeddings", "path": embed, "required": True})
    gates.append({"id": "learned_search_frozen_checkpoint", "path": CKPT_TODO, "required": True})
    d["cohorts"] = [{"id": f"{study}_discovery", "role": "blind discovery (held-out patients = Level 1 replication)",
                     "study_key": study, "sections": n_sections, "patients": n_patients, "gates": gates}]
    disc = d["discovery"]
    disc["n_folds"] = min(5, n_patients)
    disc["outer_split"] = (f"{disc['n_folds']} grouped patient folds over the {n_patients} patients of hest_multi study "
                           f"'{study}' ({n_sections} sections; all sections of a patient in one fold), deterministic "
                           "shuffle with fold_seed; each fold in turn is D_R, the others D_P")
    progs = "to be filled at registration by the blind-targets procedure (8 Gavish meta-programmes ranked by median within-section Moran's I over all sections)"
    disc["targets"]["spatial_organisation"] = {"programme:mp:<rank 1..8>": progs}
    disc["programme_targets_source"] = {"rule": progs, "path": f"runs/development/sprint_v06/blind_targets_{study}.json (to be written)"}
    fams = disc["relationship_families"]
    for f in fams.values():
        if f["targets"] and str(f["targets"][0]).startswith("programme:"):
            f["targets"] = [f"programme:mp:<rank {i}>" for i in range(1, 9)]
    if not embed:
        for f in ("programme_from_morphology", "morphology_from_molecular"):
            fams.pop(f, None)
        fams["cross_modal_sets"]["candidate_views"] = ["raw_genes", "programmes"]
        disc["rows"] = "all spots (no H&E embedding for this study: molecular + spatial families only)"
    disc["advance_rule"]["external"] = external
    d["downstream"] = downstream_block(external, replication_cohorts)
    d["registration"] = registration_block(stem, f"{stem.replace('-', '_')}.json")
    d["order_of_operations"] = ORDER_OF_OPERATIONS
    d["translation"] = {"note": "declared at registration from the downstream protocols; no clinical cohort is fixed in "
                                "this draft", "forbidden": ["calling a held-out fold an external replication",
                                                            "outcome-guided state selection"]}
    d["positive_controls"] = {"note": "none declared in the draft"}
    d["notes"] = notes
    return d


def write(name: str, doc: dict) -> str:
    (P / name).write_text(json.dumps(doc, indent=1) + "\n")
    h = canonical_hash(doc)
    (P / (name[:-5] + ".sha256")).write_text(f"{h}  {name} (canonical; DRAFT)\n")
    return h


def main() -> int:
    v03 = "dbench/protocols/ccrcc_external_replication_v03.json"
    ext_renal = (f"{v03}: frozen relation applied without refitting on li2022_ccrcc, gse250163, zenodo14620362 "
                 "(pooled B_nuis interval excluding 0, dominant-term sign in >= 60% of external patients, BH q <= 0.05 "
                 "within the family); CPTAC ccRCC bulk transfer reported as scale extension (downstream Section 5)")
    out = {}
    out["renal_spatial_v07.json"] = write("renal_spatial_v07.json", update_existing(
        "renal_spatial_v07.json", ext_renal, ["li2022_ccrcc", "gse250163", "zenodo14620362"], "renal-spatial-v07"))
    out["ependymoma_spatial_v01.json"] = write("ependymoma_spatial_v01.json", update_existing(
        "ependymoma_spatial_v01.json",
        "external spatial replication unavailable (downstream Section 2.1: no independent public PFA Visium cohort); "
        "findings reported at internal stage with external replication marked unavailable", [], "ependymoma-spatial-v01"))
    tpl = json.loads((P / "ependymoma_spatial_v01.json").read_text())
    for k in ("downstream", "registration", "order_of_operations"):
        tpl.pop(k, None)
    rule_x = lambda cohorts: (f"{v03} rule (frozen relation, no refit, pooled interval + 60% sign + BH) applied to "
                              f"{', '.join(cohorts)}; a v03-equivalent rule file per organ is hashed before external values")
    specs = [
        ("cscc-spatial-v01", "ji2020_cscc", "Ji 2020 cutaneous SCC (HEST NCBI759-NCBI770+)", 16, 5, f"{EMB}/ji2020_cscc/embeddings/hoptimus",
         "external spatial replication unavailable at drafting (no second public cSCC Visium cohort on the store)", [],
         ["historical task H1 (TSK ~ fibroblast/endothelial neighbourhood) is scored first; a discovery matching it is "
          "graded 'no novelty identified' by the novelty step, never removed"], "H1"),
        ("crc-spatial-v01", "valdeolivas2024_crc", "Valdeolivas 2024 colorectal cancer (Zenodo 7551712)", 12, 6,
         f"{EMB}/valdeolivas2024_crc/embeddings/hoptimus",
         "external spatial replication: the 10x FFPE colorectal demo sections (hest_multi 'reserve_ffpe_human_colorectal_"
         "cancer...') if patient-level metadata allow, else unavailable; decided before registration", [],
         ["historical task H2 (SPP1+ macrophage ~ myofibroblast) is scored first"], "H2"),
        ("prostate-spatial-v01", "kiviaho2024_prostate", "Kiviaho 2024 prostate (GEO GSE278936, 48 patients)", 48, 48, None,
         "external spatial replication: hest_bench_prostate / tenx_demo_prostate sections (decided before registration)", [],
         ["no H&E embedding on the store for this study: molecular + spatial families only",
          "historical task H3 (PMN-MDSC ~ club cells) is scored first"], "H3"),
        ("ccrcc-li2022-spatial-v01", "li2022_ccrcc", "Li 2022 ccRCC (Mendeley 10.17632/g67bkbnhhg.1)", 16, 8, None,
         rule_x(["gse250163", "zenodo14620362", "meylan2022_ccrcc"]), ["gse250163", "zenodo14620362", "meylan2022_ccrcc"],
         ["external ccRCC cohort used as a discovery substrate only after renal-spatial-v07 replication v03 has been run "
          "and recorded on it", "if historical task H3b is active, the H3b scoring is locked first",
          "no H-Optimus-0 embedding on the store for this study at drafting: molecular + spatial families only"], "H3b (if active)"),
        ("ccrcc-gse250163-spatial-v01", "gse250163", "GSE250163 FFPE ccRCC (5 patients)", 5, 5, None,
         rule_x(["li2022_ccrcc", "zenodo14620362", "meylan2022_ccrcc"]), ["li2022_ccrcc", "zenodo14620362", "meylan2022_ccrcc"],
         ["5 patients: leave-one-patient-out folds; small-cohort discovery, its yield is reported beside the null yield",
          "used as a discovery substrate only after renal-spatial-v07 replication v03 has been run and recorded on it",
          "Zenodo 14620362 (3 patients) is a replication cohort only: too few patients for grouped-fold discovery"], None),
    ]
    for stem, study, title, ns, npat, embed, ext, reps, notes, hist in specs:
        name = f"{stem.replace('-', '_')}.json"
        out[name] = write(name, new_draft(tpl, stem=stem, study=study, title=title, n_sections=ns, n_patients=npat,
                                          embed=embed, external=ext, replication_cohorts=reps, notes=notes,
                                          historical=hist))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
