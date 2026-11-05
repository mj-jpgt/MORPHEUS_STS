"""Draft (never apply) the registration-time edit to dbench/protocols/sprint_partitions_v01.json as a unified diff,
written to dbench/protocols/second_wave_partitions_DRAFT.diff:

* valdeolivas2024_crc: the two healthy sections ZEN37 / ZEN41 (patient 6) recorded as control sections of the CRC
  cohort (control_ids / control_patient_of / control_data_dir), never discovery folds;
* gse294385 and gse280318_hd added as external_replication studies for CRC (TENX152 overlap note, HD-only rule),
  with roles.external_replication and counts.external_replication updated accordingly; nothing else changed.

The partitions file is hashed (every StudySpec carries its sha256); the edit lands only at v02 registration.
Usage: python scripts/second_wave/draft_partitions_diff.py   (reads the converted-layout manifests for ids and pixel sizes)
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARTS = ROOT / "dbench" / "protocols" / "sprint_partitions_v01.json"
OUT = ROOT / "dbench" / "protocols" / "second_wave_partitions_DRAFT.diff"
STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
SW = f"{STORE}/spatial/second_wave_20261022"
GEO_TERMS = "GEO public (no explicit licence; NCBI places no restrictions on use or distribution)"
HD_PATIENT = {"P1CRC": "P1", "P2CRC": "P2", "P5CRC": "P5", "P3NAT": "P3", "P5NAT": "P5"}


def _dump(doc: dict, indent: int) -> str:
    return json.dumps(doc, indent=indent, ensure_ascii=False) + "\n"


def _sections(manifest: Path) -> list[dict]:
    return json.loads(manifest.read_text())["sections"]


def _pixel_sizes(layout: Path, ids) -> list[float]:
    return sorted({round(float(json.loads((layout / "metadata" / f"{s}.json").read_text())["pixel_size_um"]), 3) for s in ids})


def main() -> int:
    raw = PARTS.read_text(encoding="utf-8")
    doc = json.loads(raw)
    indent = len(raw.split("\n", 2)[1]) - len(raw.split("\n", 2)[1].lstrip(" "))
    if _dump(doc, indent) != raw:
        print("WARNING: json round-trip does not reproduce the file byte for byte; the diff will carry formatting noise", file=sys.stderr)
    new = json.loads(raw)

    crc = new["studies"]["valdeolivas2024_crc"]
    crc["control_ids"] = ["ZEN37", "ZEN41"]
    crc["control_patient_of"] = {"ZEN37": "valdeolivas2024_crc:Patient 6", "ZEN41": "valdeolivas2024_crc:Patient 6"}
    crc["control_data_dir"] = f"{SW}/valdeolivas_healthy_hest_layout"
    crc["control_role"] = ("control sections: the two sections annotated healthy in the same study (HEST ZEN37 / ZEN41 = Zenodo "
                           "7760264 A798015_Rep2 / A798015_Rep1, patient 6), absent from the local HEST copy and converted from "
                           "the Zenodo per-sample zips (CC BY 4.0) with their pathologist annotation CSVs beside them; they are "
                           "the Problem 2 healthy-section control and never enter the discovery folds (hest_ids unchanged)")
    crc["control_annotations"] = f"{SW}/valdeolivas_healthy_hest_layout/annotations/<id>_pathologist_annotations.csv"

    l1 = Path(f"{SW}/geo_GSE294385_hest_layout")
    ids1 = [r["id"] for r in _sections(l1 / "manifest.json")]
    new["studies"]["gse294385"] = {
        "role": "external_replication", "replication_for": ["valdeolivas2024_crc"],
        "dataset_title": "Spatial Multi-Omics Landscape of Colorectal Cancer Macro- and Micrometastases (Liu et al. 2026 Cancer Cell)",
        "dataset_titles": [], "layout": "flat", "data_dir": str(l1), "organ": ["Bowel"], "disease": ["Cancer / COADREAD metastasis"],
        "st_technology": ["Visium"], "platform": "visium", "hest_ids": ids1, "n_sections": len(ids1), "n_patients": len(ids1),
        "patient_of": {s: f"gse294385:{s}" for s in ids1},
        "patient_source": "GEO carries no patient identity: patient key = sample id, so patient-held-out folds are sample-held-out "
                          "(the paper reports 11 patients); sections of one patient may fall in different folds",
        "embed_roots": [], "licence": [GEO_TERMS], "previously_influenced_morpheus": False,
        "evidence": "converted by Agent S (scripts/second_wave/convert_geo_crc.py, second wave 2026-10-22); never read by MORPHEUS "
                    "model development; not in HEST",
        "pending_conversion": False,
        "source": "GEO GSE294385, 37 Visium sections of CRC metastases, Space Ranger outputs; no WSI, no annotations, no site labels",
        "platform_of": {s: "visium" for s in ids1}, "source_kind": "external_converted", "excluded_ids": {},
        "pixel_size_um": _pixel_sizes(l1, ids1),
        "note": "molecular replication only (no full-resolution H&E in GEO: no morphology, H-Optimus-0 or CellViT view)"}

    l2 = Path(f"{SW}/geo_GSE280318_visiumhd_hest_layout")
    ids2 = [r["id"] for r in _sections(l2 / "pseudo_visium_56um" / "manifest.json")]
    pat2 = {s: f"gse280318_hd:GSE280318_{HD_PATIENT[s.split('_', 1)[1]]}" for s in ids2}
    new["studies"]["gse280318_hd"] = {
        "role": "external_replication", "replication_for": ["valdeolivas2024_crc"],
        "dataset_title": "Characterization of immune cell populations in the tumor microenvironment of colorectal cancer using high "
                         "definition spatial profiling [VisiumHD] (Oliveira et al. 2025 Nat Genet; GSE280315 / GSE280318)",
        "dataset_titles": [], "layout": "flat", "data_dir": str(l2 / "pseudo_visium_56um"),
        "alt_data_dir": {"native_8um": str(l2 / "native_8um")},
        "organ": ["Bowel"], "disease": ["Cancer / COADREAD", "Normal adjacent"], "st_technology": ["Visium HD"],
        "platform": "visium", "hest_ids": ids2, "n_sections": len(ids2), "n_patients": len(set(pat2.values())),
        "patient_of": pat2, "patient_source": "section name P<n> (GEO sample titles); P5CRC and P5NAT share patient P5",
        "embed_roots": [], "licence": [GEO_TERMS], "previously_influenced_morpheus": False,
        "evidence": "converted by Agent S (scripts/second_wave/convert_geo_crc.py, second wave 2026-10-22); the HD sections are not "
                    "in HEST and were never read by MORPHEUS model development",
        "pending_conversion": False,
        "source": "GEO GSE280315 (Visium HD sub-series of GSE280318), 5 FFPE sections, 8 um bins; served here as 56 um pseudo-spots "
                  "(7 x 7 bins summed, >= 25 bins under tissue) with the native 8 um layout beside it",
        "platform_of": {s: "visium" for s in ids2}, "source_kind": "external_converted", "excluded_ids": {},
        "pixel_size_um": _pixel_sizes(l2 / "pseudo_visium_56um", ids2),
        "meta_training_overlap": "the Visium v2 section P2CRC of this study is HEST TENX152, a member of the meta_training study "
                                 "tenx_demo_bowel; the search model has seen that section",
        "rule": "HD sections only: no Visium v2, Xenium or Flex member of GSE280318 may be used for replication, and every report "
                "on this study states the TENX152 overlap",
        "note": "full-resolution tissue_image.btf not downloaded: expression + spatial only, no patches"}

    er = sorted(set(new["roles"]["external_replication"]) | {"gse294385", "gse280318_hd"})
    new["roles"]["external_replication"] = er
    c = new["counts"]["external_replication"]
    c["n_studies"] = len(er)
    c["n_sections"] = sum(new["studies"][k]["n_sections"] for k in er)
    c["n_patients"] = sum(new["studies"][k]["n_patients"] for k in er)

    header = [
        "# DRAFT registration-time edit to dbench/protocols/sprint_partitions_v01.json (second wave, Agent S, 2026-10-22).",
        "# NOT APPLIED. The partitions file is hashed (StudySpec.partitions_sha256 and the registered protocols record its sha256),",
        "# so this edit lands only at v02 registration of the CRC and cSCC protocols, together with the new protocol hashes.",
        "# Content: (1) valdeolivas2024_crc gains control_ids ZEN37 / ZEN41 (patient 6, healthy) as control sections, hest_ids unchanged;",
        "# (2) gse294385 and gse280318_hd added under external_replication for CRC (TENX152 meta-training overlap, HD-only rule);",
        "# (3) roles.external_replication and counts.external_replication updated for (2). Nothing else changes.",
        "# Apply with: patch -p1 < dbench/protocols/second_wave_partitions_DRAFT.diff   (from the repo root, at registration only)",
    ]
    rel = "dbench/protocols/sprint_partitions_v01.json"
    body = difflib.unified_diff(raw.splitlines(keepends=True), _dump(new, indent).splitlines(keepends=True),
                                fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3)
    OUT.write_text("\n".join(header) + "\n" + "".join(body), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), OUT.stat().st_size, "bytes; roles.external_replication =", er,
          "; counts.external_replication =", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
