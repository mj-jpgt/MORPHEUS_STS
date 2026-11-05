import json, pathlib
H = pathlib.Path.home(); OUT = H/"morpheus_work_bench/runs/development/benchmark_v04"
fr = (OUT/"method_freeze.sha256").read_text().strip()
doc = {"schema":"morpheus_test_studies/v1","version":"0.1-PROPOSED","status":"PROPOSED; no test-study run before the coordinator confirms","freeze_hash_sha256":fr,
 "audit_basis":"cohort_role_manifest.json decisions D1-D8 (state doc 2026-10-19..27)",
 "excluded":[
  {"study":"bladder_epithelial","reason":"decision data for D2: its L3 retrieval rows triggered the 10-23 11:34 rule that demoted the learned re-rank and fixed support-fit ordering; also 4 patients (below the 8-patient floor)"},
  {"study":"bladder_batf3","reason":"2 patients; scorer failure (all support-fit scores exactly 0.0; forked scorer segfault for GBM+H/NID/linear) not yet diagnosed or repaired by B1; its rows were part of the D2 table"},
  {"study":"ji2020_cscc","reason":"decision data for D2 (Ji L2 rows 10-23 11:44), D3 (historical H1, hit-rule v02 and scorer repair), D7; 5 patients"},
  {"study":"gse208253_oscc","reason":"strict reading: informed D4 (transfer rule v03.1 cites Meylan->OSCC permuted-null replication 36/70) and D5 (null-yield sweeps; DEVELOPMENT protocol). Not excluded here only if the coordinator accepts disclosure; see alternatives"}],
 "test_studies":[
  {"study":"gse237183_glioma","hest_layout":"/lambda/nfs/geeg/biorag3_persistent_20260711/spatial/natural_tasks_20261026/GSE237183_zenodo8105466/GSE237183_hest_layout",
   "patients":11,"sections":19,"spots":38610,"platform":"Visium","organ":"Brain (glioma)","power":"3 query + up to 8 support; n sweep capped at 8 (12 not reachable)",
   "clean_because":"used only as a P5 transfer target and a P4/P5 phenomenon cohort; no ranking, label, threshold or hyperparameter decision cites it",
   "caveats":["appears beside OSCC in the null-rate table behind D4 (8/70 permuted nulls) though the rule reason cites OSCC only","no Visium-registered H&E (CODEX scans only): candidate pool is molecular (genes, programme scores, residual programmes; four spatial versions); morphology-centre candidates and targets unavailable","no segmentation: B_known composition uses programme principal axes plus the hashed glial-proxy extension (D6) - needs coordinator ruling","11 patients: IDH-wt GBM patients carry several sections, 6 IDH-mutant patients one section each"],
   "transfer_levels":{"L0":"support and query = disjoint patients of glioma","L1":"support = ependymoma_pfa (brain, Visium) molecular candidates, query = glioma patients","L2":"support = kidney_injury or meylan2022_ccrcc (different organ), query = glioma patients; molecular candidates only"}},
  {"study":"SECOND STUDY: coordinator to choose","options":[
   {"study":"gse208253_oscc","patients":12,"sections":12,"note":"order given by coordinator; disclose D4/D5; molecular-only (no registered morphology); L1 none on box, L2 = skin cSCC/CRC/kidney","taint":"touches label-threshold decision D4"},
   {"study":"gse307534_lung_precursor / nakajima_luad","note":"never read by MORPHEUS; H-Optimus-0 embeddings exist; raw conversion to hest_multi layout queued (M, convert_lung.py 4bbfd19), patient counts unverified; would add morphology candidates (Nakajima has H&E)","taint":"none found; only S08 replication will read them"},
   {"study":"kiviaho2024_prostate","patients":48,"note":"largest power; molecular-only; historical H3 negative result fed D3 (rework trigger), so influence on method decisions is arguable","taint":"D3"}]}],
 "minimum":"spec requires >= 2 test studies each >= 8 patients if replacing bladder/cSCC",
 "recommendation":"glioma (clean) plus either the untouched lung cohort after conversion or OSCC with D4/D5 disclosed and rerun of the labels under a patient-permuted-target null (the benchmark null is not the v03.1 matched null)"}
(OUT/"test_studies.json").write_text(json.dumps(doc, indent=1)); print("ok")
