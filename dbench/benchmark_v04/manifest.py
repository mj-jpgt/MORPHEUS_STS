"""Step 1b/1c: cohort-role manifest and test-study declaration (read-only on registered files)."""
import json, hashlib, pathlib
H = pathlib.Path.home(); BENCH = H/"morpheus_work_bench"; OUT = BENCH/"runs/development/benchmark_v04"
P = json.load(open(H/"morpheus_work_sw/dbench/protocols/sprint_partitions_v01.json"))
freeze = (OUT/"method_freeze.sha256").read_text().strip()
STATE = "~/morpheus_work/.claude/MORPHEUS_STATE.md"
# decisions that touched a ranking, label, threshold or hyperparameter, with the study each was made on
decisions = [
 {"id":"D1","what":"release checkpoint step selection (repair step 1000 by harness net yield@10, B_nuis); B_known repair under fixed B_known (b7879b1)","studies":["kidney_injury"],"evidence":"release/FROZEN.md; state 2026-10-22"},
 {"id":"D2","what":"adopt support-fit ordering as the ranking (learned shrinkage re-rank demoted): decision rule stated 10-23 08:55 and triggered 10-23 11:34 on dev few-shot (kidney_injury, 40 targets), bladder L3 retrieval rows (bladder_epithelial; batf3 degenerate) and Ji 2020 L2 retrieval rows","studies":["kidney_injury","bladder_epithelial","bladder_batf3","ji2020_cscc"],"evidence":"state 2026-10-23 09:32 (decider, ordered bladder L3 + Ji L2), 11:34 (rule triggered on bladder), 11:44 (Ji L2 same direction)"},
 {"id":"D3","what":"historical hit rule v01 -> v02 and repaired query-validated scorer; commitment to rework the objective if ours did not beat count-matched random across H1-H3","studies":["ji2020_cscc","valdeolivas2024_crc","kiviaho2024_prostate"],"evidence":"state 2026-10-19..22 (H1-H3 results 10-22 16:23: ours none/3/none vs random 10/5/3); protocols historical_rediscovery_v02, historical_adopted_ordering_v01"},
 {"id":"D4","what":"transfer rule v03 -> v03.1 (gain must exceed p95 of matched nulls): motivated by Meylan -> GSE208253 OSCC null rates (36/70 permuted-programme nulls replicated); stated as consistent with the CRC external replication rule","studies":["gse208253_oscc","meylan2022_ccrcc","valdeolivas2024_crc","gse294385"],"evidence":"dbench/protocols/transfer_v03_1_addendum.json (reason field); state 2026-10-26 12:31 and 2026-10-26 (rate table also lists glioma 8/70, not cited in the reason)"},
 {"id":"D5","what":"null-yield sweeps added per arm after saturated fold validation (32/32) read on a GSE208253 cell; GSE208253 run under a DEVELOPMENT protocol (gse208253_oscc_dev_v01)","studies":["gse208253_oscc"],"evidence":"state 2026-10-26 06:00-06:05"},
 {"id":"D6","what":"B_known glial-proxy extension (ependymoma Astrocytes object); declared applicable to glioma","studies":["ependymoma_pfa"],"evidence":"bknown_glial_extension_v01.json f46957f; state 2026-10-26"},
 {"id":"D7","what":"tumour-row mask NNLS v2 Tumor >= 0.385, vocabulary v3.1, per-patient supplementary column, pooling v2, small-cohort rule","studies":["meylan2022_ccrcc","ji2020_cscc","valdeolivas2024_crc"],"evidence":"state 2026-10-23..26"},
 {"id":"D8","what":"method shaped by Meylan (blind run, grades, Gate 0/1); Meylan findings need external replication","studies":["meylan2022_ccrcc"],"evidence":"owner consolidated protocol summary 2026-10-27"},
]
touched = {}
for d in decisions:
    for s in d["studies"]: touched.setdefault(s, []).append(d["id"])
# primary role per cohort (spec vocabulary), plus influence flags
role_map = {"meta_training":"meta-training","development":"development","historical":"historical rediscovery","blind":"blind discovery","external_replication":"external replication","sanity":"blind discovery","reserve":"reserve (unused)"}
coh = {}
for k, v in P["studies"].items():
    r = role_map[v["role"]]
    e = {"role": r, "partition_manifest_role": v["role"], "n_sections": v.get("n_sections"), "n_patients": v.get("n_patients"), "organ": v.get("organ"),
         "platform": v.get("platform") or v.get("platforms"), "decisions_made_on_it": touched.get(k, [])}
    if k in ("bladder_epithelial","bladder_batf3"):
        e["note"] = "manifest lists meta_training, but transfer_holdouts.json (2026-10-21) withholds both from training: the release checkpoint never trained on them; they are development by D2"
        e["role"] = "development (organ holdout of the release checkpoint; decision data for D2)"
    if k in ("ji2020_cscc",):
        e["role"] = "historical rediscovery; development by D2/D3/D7"
    if k in ("valdeolivas2024_crc","kiviaho2024_prostate"): e["role"] = "historical rediscovery; development-influenced (D3)" + ("; D4,D7" if k=="valdeolivas2024_crc" else "")
    if k == "meylan2022_ccrcc": e["role"] = "blind discovery (registered v07); development-influenced (method shaped by Meylan, D7/D8)"
    if k == "ependymoma_pfa": e["role"] = "blind discovery; development-influenced (D6)"
    if k == "gse294385": e["role"] = "external replication (CRC); development-influenced (D4)"
    coh[k] = e
for k, role, n_sec, n_pat, org, note in [
  ("gse237183_glioma","TEST (proposed)",19,11,"Brain (glioma)","used only as a P5 transfer target and in P4/P5 phenomena tests; no decision cites it; appears in the same null-rate table as D4 (8/70 permuted nulls) but is not cited in the rule reason"),
  ("gse208253_oscc","development-influenced (D4, D5); proposed second test study only with disclosure",12,12,"Oral cavity (OSCC)","proposer comparison (DEVELOPMENT protocol), P5 target; cited in the v03.1 reason")]:
    coh[k] = {"role": role, "partition_manifest_role": "(not in sprint_partitions_v01; natural_tasks_20261026 via hest_multi.study_specs 64587af)", "n_sections": n_sec, "n_patients": n_pat, "organ": [org], "platform": "visium", "decisions_made_on_it": touched.get(k, []), "note": note}
for k in ("gse307534_lung_precursor","nakajima_luad"):
    coh[k]["note"] = "raw on store; conversion to hest_multi layout queued by M/B1; never read by MORPHEUS; embeddings exist under external/hest_fm_embeddings_20261016; candidate untouched replacement pending conversion"
man = {"schema":"morpheus_cohort_role_manifest/v1","manifest_version":"1.0","freeze_hash_sha256":freeze,
       "partitions_manifest_sha256": hashlib.sha256((H/"morpheus_work_sw/dbench/protocols/sprint_partitions_v01.json").read_bytes()).hexdigest(),
       "decisions": decisions, "cohorts": coh,
       "roles_summary": {r: sorted(k for k,v in coh.items() if v["role"].startswith(r)) for r in ("meta-training","development","historical","blind","external","TEST","reserve")}}
(OUT/"cohort_role_manifest.json").write_text(json.dumps(man, indent=1))
print({r: len(v) for r, v in man["roles_summary"].items()})
