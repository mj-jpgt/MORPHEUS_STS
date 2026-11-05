"""Turn the v03 addendum DRAFTs into REGISTERED protocols against the phenomena-wiring commit e559bb0 (Agent W2, 2026-10-23).
Reads dbench/protocols/<stem>_v03_addendum.json (DRAFT, commit b091d69) and rewrites it in place (history keeps the DRAFT).
Usage: python scripts/second_wave/finalize_v03.py"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "dbench" / "protocols"
CKSHA = "13c36faa8cd9caa7121381b59dc06e5d35069492b565e1fa786e2e03a46f3446"
WIRING = "e559bb0"
RESTRICTION = {"metabolic_consumer_epithelial": "epithelial_dominant", "metabolic_producer_stromal_immune": "stromal_immune_dominant",
               "metabolic_contrast_genes": "segmented", "invisible_field": "normal_appearing",
               "tumour_budding": "invasive_front", "invasive_front_morphology": "invasive_front"}
NORMAL_APPEARING_ROWS = (
    "normal_appearing spots, defined by morpheus_encoder.tissue_architecture.normal_appearing as the runner applies it: epithelial fraction + "
    "neoplastic fraction (CellViT PanNuke) >= 0.5, neoplastic fraction <= 0.5, and the inferred copy-number score <= the 90th percentile of "
    "the healthy reference sections ZEN37 / ZEN41 (CRC; the within-section reference as flagged fallback where no healthy section is declared); "
    "distance to the nearest neoplastic-dominant spot from the segmentation map is recorded for every spot (candidate view normal_appearing: distance_to_neoplastic_um). "
    "The pathologist spot class is the independent check and never defines the rows.")
PATH_CLASS_DECL = ("the pathologist spot class is not carried inside the normal_appearing view; the per-patient two-arm disagreement table "
                   "(pathologist arm vs computational arm) is computed downstream from the Valdeolivas Pathology_SpotAnnotations "
                   "(morpheus_encoder/scripts/crosscheck_normal_appearing.py), declared as a downstream output, reported whatever the agreement")
NUCLEAR_DECL = ("nuclear-shape nuisance columns (nuclear eccentricity, major-axis orientation dispersion) are not wired into the comparison "
                "model of this family; declared post-hoc characterisation: every frozen hypothesis' effect is re-estimated with nuclear "
                "elongation added to the comparison model; reported, not gated")
TARGETS_NOW = {
    "metabolic_contrast_genes": "contrast:gene:<SYM> parsed by hest_multi.parse_target (own gene minus its 6-neighbour mean)",
    "invisible_field": "inferred_cnv:score parsed by hest_multi.parse_target",
    "tumour_budding": "budding_like:index (also bud_count, bud_nuclei_fraction) and gland_architecture:index (and its five statistics) parsed by hest_multi.parse_target",
}


def build(stem, study):
    flat = stem.replace("-", "_")
    d = json.load(open(P / f"{flat}_v03_addendum.json"))
    te = json.load(open(ROOT / "runs/development/sprint_v06" / f"tumour_elevated_{study}.json"))
    names = [t["loader_target"] for t in te["targets"]]
    fams = d["discovery"]["relationship_families"]
    for name, f in fams.items():
        was = f.pop("blocked_on", None)
        f["wired_in"] = f"runner commit {WIRING}: " + (was or "")
        f["row_restriction"] = RESTRICTION[name]
        adm = f.setdefault("admissibility", {})
        if adm.get("context") and adm.get("cross_modal"):
            adm["combine"] = "both"
        if name in TARGETS_NOW:
            f["targets_wiring"] = TARGETS_NOW[name]
        if name == "invisible_field":
            f["targets"] = [n if not n.startswith("programme:mp:<tumour-elevated") else None for n in f["targets"]]
            f["targets"] = names + [t for t in f["targets"] if t]
            f["targets_rule"] = (f["targets_rule"].split(", all sections")[0] + ", all sections, response-free; fixed by name below from "
                                 f"runs/development/sprint_v06/tumour_elevated_{study}.json before registration), plus the inferred copy-number score")
            f["rows"] = NORMAL_APPEARING_ROWS
            f["candidate_views_note"] = ("normal_appearing serves distance_to_neoplastic_um; " + PATH_CLASS_DECL)
            f["declared_outputs"] = [o for o in f.get("declared_outputs", []) if o["id"] != "per_patient_two_arm_disagreement_table"] + [{
                "id": "per_patient_two_arm_disagreement_table",
                "definition": "per patient: normal-appearing spot sets of the pathologist arm (class 'non-neoplastic epithelium / mucosa') and of the computational arm, their overlap, the spots only one arm calls normal-appearing, and each frozen hypothesis' gain on each arm's set",
                "computed": PATH_CLASS_DECL}]
            ctl = f.get("controls")
            if isinstance(ctl, dict):
                ctl["healthy_sections"] = ("healthy sections ZEN37 / ZEN41 (patient 6) are the CNV reference of the normal_appearing rule and independent controls "
                                           "in which a frozen configuration must be absent (never discovery folds)") if study.startswith("valdeolivas") else ctl.get("healthy_sections")
        if name == "tumour_budding":
            f["declared_post_hoc_characterisation"] = NUCLEAR_DECL
            ctl = f.get("controls")
            if isinstance(ctl, dict):
                ctl.pop("cross_section_check", None)
        if name == "invasive_front_morphology":
            f["rows"] = ("invasive_front row restriction as the runner applies it (tissue_architecture invasive-front label from the CellViT neoplastic fraction band; "
                         "the pathologist classes are the independent check and never define it)")
    d["discovery"]["programme_targets_source"] = {
        "rule": te["rule"], "path": f"runs/development/sprint_v06/tumour_elevated_{study}.json", "targets": names,
        "partitions_sha256_at_computation": te["partitions_sha256"],
        "note": "tumour-elevated programme targets of invisible_field; the generic programme targets are those of the v02 protocol (blind_targets file)"}
    d["discovery"]["runner_support"] = (f"runner commit {WIRING}: row_restriction, admissibility.combine = both, declared_genes force-inclusion, candidate views "
                                        "composition / budding_like / gland_architecture / inferred_cnv / normal_appearing, targets contrast:gene, inferred_cnv:score, budding_like:*, gland_architecture:*")
    d["discovery"].pop("pending_views", None)
    rel = d["release"]
    rel["release_commits"].update({"phenomena_wiring": WIRING, "gpu_memory_and_score_frozen": "a27a8a6",
                                   "code_base_of_this_registration": WIRING + " (clean worktree) plus this protocol's registration commit"})
    d["declared_not_wired"] = {"nuclear_shape_nuisance_columns": NUCLEAR_DECL, "pathologist_class_in_view": PATH_CLASS_DECL}
    d["campaign_id"] = f"{stem}-v03-20261023"
    d["revision"] = f"r1 (Agent W2, 2026-10-23): deferred families of {stem}-v02-20261023, registered after the runner wiring ({WIRING})"
    d["status"] = f"REGISTERED 2026-10-23 by Agent W2 on the coordinator's instruction (checkpoint release/frozen.pt {CKSHA[:12]}, code base {WIRING})"
    d["notes"] = [n for n in d.get("notes", []) if "deferred_families" not in n] + [f"addendum to {stem}-v02-20261023: the families it could not serve"]
    d["registration"] = {"never": "no write under runs/prospective by hand",
                         "commands": [f"python -m dbench.prospective_campaign --protocol dbench/protocols/{flat}_v03_addendum.json --run-dir runs/prospective/{d['campaign_id']}",
                                      f"CAMPAIGN_PROTOCOL=dbench/protocols/{flat}_v03_addendum.json bash dbench/scripts/blind_sweep.sh runs/prospective/{d['campaign_id']} <workers>"]}
    (P / f"{flat}_v03_addendum.json").write_text(json.dumps(d, indent=1) + "\n")
    return d


if __name__ == "__main__":
    for stem, study in (("crc-spatial", "valdeolivas2024_crc"), ("cscc-spatial", "ji2020_cscc")):
        d = build(stem, study)
        print(d["campaign_id"], {k: (len(v["targets"]), v["row_restriction"], v["admissibility"].get("combine")) for k, v in d["discovery"]["relationship_families"].items()})
