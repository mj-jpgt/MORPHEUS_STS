"""Step 1a of METHOD_BENCHMARK_SPEC: hash the adopted method into method_freeze.json (read-only on the method code)."""
import hashlib, json, subprocess, sys, pathlib, datetime
H = pathlib.Path.home()
SW, V06, BENCH = H/"morpheus_work_sw", H/"morpheus_work_v06", H/"morpheus_work_bench"
OUT = BENCH/"runs/development/benchmark_v04"
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def git(repo, *a): return subprocess.check_output(["git","-C",str(repo),*a],text=True).strip()
def tree(repo, rel, glob="*.py"):
    return {str(p.relative_to(repo)): sha(p) for p in sorted((repo/rel).glob(glob))}
LS = V06/"runs/development/learned_search_v03"
files = {}
files.update(tree(SW,"morpheus_encoder/learned_search"))
files["morpheus_encoder/general_discovery.py"] = sha(SW/"morpheus_encoder/general_discovery.py")
files["morpheus/tier2/substrates/hest_multi.py"] = sha(SW/"morpheus/tier2/substrates/hest_multi.py")
files.update({"bench:"+k: v for k, v in tree(BENCH,"dbench/harness").items()})
prot = {}
for repo, names in ((SW, ["transfer_v03_1_addendum.json","sprint_partitions_v01.json","historical_rediscovery_v02.json"]),
                    (V06, ["bknown_glial_extension_v01.json","historical_adopted_ordering_v01.json"])):
    for n in names: prot[n] = sha(repo/"dbench/protocols"/n)
cfg = {"release/FROZEN.md": sha(LS/"release/FROZEN.md"), "release/run_info.json": sha(LS/"release/run_info.json"),
       "transfer_holdouts.json": sha(LS/"transfer_holdouts.json"),
       "benchmark_v03/dev_targets_v01.json": sha(V06/"runs/development/benchmark_v03/dev_targets_v01.json"),
       "full/frozen.pt(base, step 13000)": sha(LS/"full/frozen.pt")}
spec = H/"morpheus_work/.claude/METHOD_BENCHMARK_SPEC.md"
doc = {
 "schema": "morpheus_method_freeze/v1",
 "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
 "spec": {"file": "METHOD_BENCHMARK_SPEC.md", "sha256": sha(spec)},
 "adopted_method": {
   "ranking": "support-fit ordering: patient-grouped cross-validated gain on the support patients over B_known (screen scores every pool member)",
   "beam": "learned beam (release checkpoint) choosing which sets are fitted, budget K=64; supplementary row only in Table 1",
   "relation_family": "v03_rff: random-feature kernel ridge with linear terms, closed form, patient-grouped penalty (general_discovery.py v03_rff, relation_v03.py)",
   "baselines": {"B_nuis": "nuisance", "B_known": "known biology, strict exclusions (target, overlapping programmes, their member genes, programmes containing a gene target); primary",
                 "B_decomp": "best proper-subset decomposition (pairs)"},
   "label_rule": "v03.1 gain-above-null replication: interval rule (patient bootstrap 95%) AND gain above the 95th percentile of the matched null; benchmark binary label per spec 2.2 (interval lower bound > 0 and gain > p95 of patient-permuted-target null for that (s,t)); GAM second family; 90/99 sensitivity. Label code is benchmark infrastructure written after this freeze against the spec hash above.",
   "release_checkpoint": {"path": "runs/development/learned_search_v03/release/frozen.pt", "sha256": sha(LS/"release/frozen.pt")},
 },
 "git": {"second_wave": {"path": "~/morpheus_work_sw", "branch": git(SW,"branch","--show-current"), "commit": git(SW,"rev-parse","HEAD"),
                         "learned_search_tree": git(SW,"rev-parse","HEAD:morpheus_encoder/learned_search")},
         "v06_master": {"path": "~/morpheus_work_v06", "branch": "master", "commit_at_bench_branch_point": git(BENCH,"rev-parse","HEAD"),
                        "dbench_harness_last_commit": git(BENCH,"log","-1","--format=%H","--","dbench/harness"),
                        "dbench_harness_tree": git(BENCH,"rev-parse","HEAD:dbench/harness")},
         "bench_worktree": {"path": "~/morpheus_work_bench", "branch": git(BENCH,"branch","--show-current")}},
 "code_files_sha256": files, "protocol_files_sha256": prot, "config_files_sha256": cfg,
 "environment": {"python": sys.version.split()[0]},
}
canon = json.dumps(doc, sort_keys=True, separators=(",",":"))
doc["freeze_hash_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
doc["freeze_hash_note"] = "sha256 of the canonical JSON (sorted keys, compact separators) of this document without freeze_hash_sha256 and freeze_hash_note"
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"method_freeze.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
(OUT/"method_freeze.sha256").write_text(doc["freeze_hash_sha256"]+"\n")
print(doc["freeze_hash_sha256"], len(files), "code files")
