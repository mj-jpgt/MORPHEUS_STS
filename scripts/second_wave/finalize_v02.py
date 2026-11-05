"""Finalise the second-wave CRC / cSCC protocols from the DRAFTs (Agent W2, 2026-10-23).
Writes crc_spatial_v02.json / cscc_spatial_v02.json (REGISTERED content: only families the runner serves) and
crc_spatial_v03_addendum.json / cscc_spatial_v03_addendum.json (DRAFT: only the deferred families, each with
blocked_on). Usage: python scripts/second_wave/finalize_v02.py"""
from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "dbench" / "protocols"
CKPT = "runs/development/learned_search_v03/release/frozen.pt"
CKSHA = "13c36faa8cd9caa7121381b59dc06e5d35069492b565e1fa786e2e03a46f3446"
STAMP = "20261023"
HEAD = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
CONTRAST = "aaca0ad"
REPORT = ("Primary lists are the registered uncapped top-20 (B_nuis-conditioned search, K = 64). A companion distinct-objects view "
          "applies an anchor-multiplicity cap m = 8 as a re-presentation of the same lists (no new search or fits); m was fixed from "
          "the development table (cap_rerank/cap_table.md, commit 3b75b57) before any aggregate of this run was read. Registered "
          "pooling counts are unaffected.")
REPORTING = {"top20_lists": REPORT,
             "development_table": {"path": "runs/development/benchmark_v03/cap_rerank/cap_table.md",
                                   "commit_in_v06_clone": "3b75b57", "commit_in_sw_clone": "00b8523 (cherry-pick)"},
             "anchor_multiplicity_cap_m": 8}


def rank_match(name):
    return re.fullmatch(r"programme:mp:<rank (\d)>", name)


DEFER = {
    "metabolic_consumer_epithelial": "row restriction to epithelial-dominant spots (CellViT PanNuke Neoplastic fraction largest of the three composition fractions); candidate view 'composition' (today only B_known covariates, not a candidate view); force-inclusion of the declared genes SLC16A1, SLC16A3, LDHA, LDHB in the raw_genes panel",
    "metabolic_producer_stromal_immune": "row restriction to stromal- and immune-dominant spots (CellViT Connective or Inflammatory fraction largest); candidate view 'composition'; force-inclusion of the declared genes in the raw_genes panel",
    "invisible_field": "row restriction 'normal_appearing' (morpheus_encoder/tissue_architecture.normal_appearing, thresholds 0.5 / 0.5) wired into the runner; candidate view distance_to_neoplastic_um; spot-level target inferred_cnv:score in hest_multi.parse_target (tissue_architecture.infer_cnv); the four tumour-elevated programme targets fixed by name with the blind procedure on normal_appearing vs neoplastic-dominant spots BEFORE registration; candidate view 'composition'",
    "tumour_budding": "spot-level targets budding_like:index and gland_architecture:index in hest_multi.parse_target (tissue_architecture.budding_view / gland_view); invasive-front row restriction; nuclear-shape nuisance columns in the comparison model; force-inclusion of the declared genes KRT17, ITGB1, POSTN, CTHRC1; candidate view 'composition'",
    "invasive_front_morphology": "invasive-front row restriction (neoplastic fraction band between 0.1 and 0.9 with a nbr6 neighbour on each side of 0.5, from the segmentation map) wired into the runner",
    "metabolic_contrast_genes": "contrast:gene:<SYM> targets in hest_multi.parse_target and the episode builder (own gene minus its 6-neighbour mean); commit aaca0ad serves contrast:programme: only",
}


def fill(obj, names):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            m = rank_match(k)
            out[f"programme:{names[int(m.group(1)) - 1]}" if m else k] = fill(v, names)
        return out
    if isinstance(obj, list):
        out = []
        for x in obj:
            m = rank_match(x) if isinstance(x, str) else None
            out.append(f"programme:{names[int(m.group(1)) - 1]}" if m else fill(x, names))
        return out
    return obj


def build(stem, study, draft):
    d = json.load(open(P / draft))
    bt = json.load(open(ROOT / "runs/development/sprint_v06" / f"blind_targets_{study}.json"))
    names = [t["programme"] for t in bt["targets"]]
    loader = {t["loader_target"]: t for t in bt["targets"]}
    disc = d["discovery"]
    fams = disc["relationship_families"]
    deferred = {}
    for name in list(fams):
        f = fams[name]
        if not f.get("targets"):
            fams.pop(name)
        elif name == "metabolic_contrast":
            keep = [t for t in f["targets"] if t.startswith("contrast:programme:")]
            drop = [t for t in f["targets"] if t not in keep]
            f["targets"] = keep
            f.pop("blocked_on", None)
            f["pending_note"] = (f"loader commit {CONTRAST} serves contrast:programme: targets; candidate view 'composition' is not served by "
                                 "the runner and is recorded as unavailable in every receipt; rows are all covered spots; "
                                 "the four contrast:gene targets are in deferred_families.metabolic_contrast_genes")
            g = copy.deepcopy(f)
            g["targets"] = drop
            deferred["metabolic_contrast_genes"] = g
        elif f.get("pending_view"):
            deferred[name] = copy.deepcopy(fams.pop(name))
    for name, g in deferred.items():
        g["blocked_on"] = DEFER[name]
        g.pop("pending_view", None)
        g.pop("pending_note", None)
    if "invisible_field" in deferred:
        deferred["invisible_field"].setdefault("declared_outputs", []).append({
            "id": "per_patient_two_arm_disagreement_table",
            "definition": "per patient: the normal-appearing spot sets of the pathologist arm (Zenodo class 'non-neoplastic epithelium / mucosa') and the computational normal-appearing arm (tissue_architecture.normal_appearing), their overlap, the spots only one arm calls normal-appearing, and each frozen hypothesis' gain on each arm's set; reported whatever the agreement"})
    for fam in fams.values():
        fam.pop("pending_view", None)
    fams = fill(fams, names)
    disc["relationship_families"] = fams
    disc["targets"] = fill(disc["targets"], names)
    sp = disc["targets"]["spatial_organisation"]
    for k in list(sp):
        t = loader[k]
        sp[k] = f"rank {t['rank']} of the blind-targets procedure (median within-section Moran's I {t['median_moran_i']})"
    disc["programme_targets_source"] = {"rule": bt["rule"], "path": f"runs/development/sprint_v06/blind_targets_{study}.json",
                                        "partitions_sha256_at_computation": bt["partitions_sha256"],
                                        "targets": [t["loader_target"] for t in bt["targets"]]}
    old = disc["target_loader_columns"]
    tpl = ("hest_multi target '{t}' (fixed by name; rank {r} in runs/development/sprint_v06/blind_targets_" + study +
           ".json); the score is the mean within-patient-centred log1p-CPM of the programme's genes on the full panel, within-patient standardised")
    cols = {}
    for t in sorted({t for f in fams.values() for t in f["targets"]}):
        if t.startswith("morph:") and t in old:
            cols[t] = old[t]
        elif t in loader:
            cols[t] = {"loader_target": t, "maps_to": tpl.format(t=t, r=loader[t]["rank"])}
        elif t.startswith("contrast:"):
            cols[t] = {"loader_target": t, "maps_to": f"hest_multi contrast_programme target (own score minus its 6-neighbour mean; commit {CONTRAST}), within-patient standardised"}
    disc["target_loader_columns"] = cols
    disc["pending_views"] = {"rule": "pending families are listed under deferred_families and are not swept by this registration"}
    disc["runner_support"] = ("runner --episode hest_multi serves own_mp / concept / contrast_programme targets on any partition study; "
                              "views it does not serve (composition candidate view, scimilarity / scgpt where absent) are recorded as unavailable in every receipt")
    rel = d["release"]
    rel["learned_search_checkpoint"] = {"path": CKPT, "sha256": CKSHA}
    rel["name"] = "same frozen release as renal-spatial-v07 / ependymoma-spatial-v01 (checkpoint, relation v03_rff, hest_multi loader)"
    rel["commit_note"] = ("hashes are those of ~/morpheus_work_sw (branch second_wave), which contains the v06 clone history; "
                          "cherry-picks: b877b05 -> 8b1246e (chunked cross-fold), 3b75b57 -> 00b8523 (cap table)")
    rel["release_commits"] = {"clone_head_before_registration": HEAD, "contrast_loader": CONTRAST, "tissue_architecture": "f1c91dd",
                              "inversion_v03": "1455244", "transfer_v03": "fe423cf", "historical_lock": "0b95dee",
                              "chunked_crossfold_launcher": "8b1246e", "cap_table": "00b8523",
                              "note": "the registration commit itself is the next commit on second_wave and is recorded in the sweep driver log"}
    disc["proposers"]["learned_search"]["checkpoint_path"] = CKPT
    disc["proposers"]["learned_search"]["checkpoint_sha256"] = CKSHA
    for c in d["cohorts"]:
        for g in c["gates"]:
            if g["id"] == "learned_search_frozen_checkpoint":
                g["path"], g["note"] = CKPT, f"release checkpoint, sha256 {CKSHA}"
    d["reporting"] = REPORTING
    d["notes"] = list(d.get("notes", [])) + ["families the runner cannot serve are in deferred_families and are registered as the v03 addendum (blocked_on names the capability each needs)"]
    d["order_of_operations"] = [
        "1 checkpoint freeze: runs/development/learned_search_v03/release/frozen.pt, sha256 " + CKSHA,
        "2 historical-rediscovery H1/H2 version-1 lists locked at commit 0b95dee (runs/development/benchmark_v03/historical/LOCK_H1_H2.md), which declares the CRC and cSCC cohorts open for second-wave work",
        f"3 programme targets fixed by name by the blind-targets procedure (median within-section Moran's I over all sections, response-free): runs/development/sprint_v06/blind_targets_{study}.json",
        "4 protocol status REGISTERED, canonical sha256 recorded; python -m dbench.prospective_campaign (manifest + empty ledger; ready_for_discovery must be true)",
        "5 fold sweep (dbench/scripts/blind_sweep.sh), chunked cross-fold scoring, registered aggregation: hypotheses frozen and hashed in the ledger",
        "6 replication v03 on the declared replication cohorts (rule hashed before external values)",
        "7 characterisation (literature closed), then the literature lock and the novelty pass -- no literature is consulted at any earlier step"]
    if study == "valdeolivas2024_crc":
        d["downstream"]["replication_cohorts"] = ["gse294385", "gse280318_hd"]
        d["downstream"]["replication"] = ("external replication cohorts declared in the partitions manifest: gse294385 (molecular only; no full-resolution H&E in GEO) "
                                          "and gse280318_hd (Visium HD sections only; its Visium v2 section is HEST TENX152 in meta_training); rule hashed before external values")
    d.pop("draft", None)
    d.pop("registration_note", None)
    d["registration"] = {"never": "no write under runs/prospective by hand", "note": "v01 names are hashed under the historical lock; this is v02"}
    flat = stem.replace("-", "_")
    # addendum (deferred families only)
    a = copy.deepcopy(d)
    a["campaign_id"] = f"{stem}-v03-DRAFT"
    a["status"] = "DRAFT (v03 addendum: deferred families only; flips to REGISTERED when the runner serves the blocked_on capabilities)"
    a["revision"] = f"DRAFT r1 (Agent W2, 2026-10-23): deferred families of {stem}-v02; same checkpoint, folds, pass rules and reporting block"
    a["supersedes"] = f"none: addendum to {stem}-v02-{STAMP} (families not swept there)"
    a["ledger"] = {"filename": f"{flat}_v03_ledger.jsonl"}
    a["discovery"]["relationship_families"] = copy.deepcopy(deferred)
    a["discovery"]["targets"] = {"note": "targets of each deferred family are listed in relationship_families"}
    a["discovery"]["target_loader_columns"] = {}
    a["discovery"]["programme_targets_source"]["note"] = "tumour-elevated programme ranks of invisible_field are filled by name before registration"
    a.pop("deferred_families", None)
    a["order_of_operations"].insert(3, "3b every blocked_on capability wired into the runner and tested; tumour-elevated programme ranks fixed by name")
    d["status"] = f"REGISTERED 2026-10-23 by Agent W2 on the coordinator's instruction (checkpoint release/frozen.pt {CKSHA[:12]}, code base {HEAD})"
    d["campaign_id"] = f"{stem}-v02-{STAMP}"
    d["revision"] = "r1 (Agent W2, 2026-10-23): second wave; programme targets by name; deferred families recorded. Draft: " + str(d["revision"])[:200]
    d["ledger"] = {"filename": f"{flat}_v02_ledger.jsonl"}
    d["deferred_families"] = {k: {"blocked_on": v["blocked_on"], "registered_in": f"dbench/protocols/{flat}_v03_addendum.json",
                                  "targets": v["targets"]} for k, v in deferred.items()}
    d["registration"]["commands"] = [
        f"python -m dbench.prospective_campaign --protocol dbench/protocols/{flat}_v02.json --run-dir runs/prospective/{d['campaign_id']}",
        f"CAMPAIGN_PROTOCOL=dbench/protocols/{flat}_v02.json bash dbench/scripts/blind_sweep.sh runs/prospective/{d['campaign_id']} <workers>"]
    return d, a


if __name__ == "__main__":
    for stem, study, draft in (("crc-spatial", "valdeolivas2024_crc", "crc_spatial_v01_DRAFT.json"),
                               ("cscc-spatial", "ji2020_cscc", "cscc_spatial_v01_DRAFT.json")):
        d, a = build(stem, study, draft)
        flat = stem.replace("-", "_")
        (P / f"{flat}_v02.json").write_text(json.dumps(d, indent=1) + "\n")
        (P / f"{flat}_v03_addendum.json").write_text(json.dumps(a, indent=1) + "\n")
        print(d["campaign_id"], {k: len(v["targets"]) for k, v in d["discovery"]["relationship_families"].items()},
              "| deferred:", {k: len(v["targets"]) for k, v in a["discovery"]["relationship_families"].items()})
