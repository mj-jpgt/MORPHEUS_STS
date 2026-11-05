"""Convert the two healthy Valdeolivas 2024 sections (Zenodo 7760264 per-sample Space Ranger zips
SN123_A798015_Rep1 and SN124_A798015_Rep2 = HEST ZEN41 and ZEN37, patient 6) into the flat HEST layout
with the existing converter dbench/scripts/convert_external_visium.py (imported, not modified):
<out>/st/<id>.h5ad, <out>/metadata/<id>.json, no patches (their capture areas sit inside the SN123 / SN124
.ndpi slides; image extraction is a later task), <out>/annotations/<id>_pathologist_annotations.csv copied
from Pathology_SpotAnnotations.zip, and <out>/manifest.json with sha256 of every written file.

Usage: python scripts/second_wave/convert_valdeolivas_healthy.py --zenodo-dir DIR --out OUT [--catalogue CSV]
Then the read-only load check: --check loads OUT through hest_multi (cache_root=None) and prints row counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
CATALOGUE = f"{STORE}/spatial/hest_meta/HEST_v1_3_0.csv"
SECTIONS = {"ZEN41": "SN123_A798015_Rep1", "ZEN37": "SN124_A798015_Rep2"}   # HEST id -> Zenodo sample
MEMBERS = {"matrix_h5": "filtered_feature_bc_matrix.h5", "positions": "spatial/tissue_positions_list.csv",
           "scalefactors": "spatial/scalefactors_json.json", "hires": "spatial/tissue_hires_image.png"}


def _converter():
    spec = importlib.util.spec_from_file_location("convert_external_visium",
                                                  ROOT / "dbench" / "scripts" / "convert_external_visium.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _catalogue_rows(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh) if r["id"] in SECTIONS}


def _extract(zpath: Path, sample: str, dest: Path) -> dict:
    """Extract only the Space Ranger members we need (untrusted data, never executed)."""
    files = {}
    with zipfile.ZipFile(zpath) as z:
        for key, member in MEMBERS.items():
            name = f"{sample}/{member}"
            target = dest / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            files[key] = target
    return files


def convert(zenodo_dir: Path, out: Path, catalogue: str) -> None:
    cv = _converter()
    cat = _catalogue_rows(catalogue)
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(exist_ok=True)
    rows, sources = [], {}
    ann_zip = zipfile.ZipFile(zenodo_dir / "Pathology_SpotAnnotations.zip")
    for sid, sample in SECTIONS.items():
        c = cat[sid]
        assert c["subseries"] == sample.split("_", 1)[1], (sid, c["subseries"], sample)
        zpath = zenodo_dir / f"{sample}.zip"
        files = _extract(zpath, sample, out / "raw" / sample)
        meta = {"patient": c["patient"], "organ": c["organ"], "oncotree_code": c["oncotree_code"],
                "disease_state": c["disease_state"], "tissue": c["tissue"], "st_technology": c["st_technology"],
                "preservation_method": c["preservation_method"], "chemistry": "Visium fresh-frozen (poly-A capture), Space Ranger",
                "dataset_title": c["dataset_title"], "license": c["license"] or "Creative Commons Attribution 4.0 International",
                "source_accession": f"zenodo:7760264/{sample}.zip", "hest_id": sid, "zenodo_sample": sample,
                "zenodo_slide": sample.split("_", 1)[0], "hest_catalogue_pixel_size_um_embedded": c["pixel_size_um_embedded"],
                "image_note": "Space Ranger hires PNG only in this layout; the full-resolution H&E is capture area "
                              f"{sample} of the Zenodo .ndpi slide {sample.split('_', 1)[0]} (0.23 um/px), not yet extracted"}
        rows.append(cv.convert_spaceranger_dir(files, out, sid, meta))
        ann_member = f"Pathology_SpotAnnotations/Pathologist_Annotations_{sample}.csv"
        ann_out = out / "annotations" / f"{sid}_pathologist_annotations.csv"
        ann_out.write_bytes(ann_zip.read(ann_member))
        sources[sid] = {"zip": zpath.name, "zip_sha256": cv.sha256(zpath), "annotation_member": ann_member,
                        "annotation_sha256": cv.sha256(ann_out)}
    cv.write_manifest(out, "valdeolivas_zenodo_7760264_healthy", rows,
                      {"license": "CC BY 4.0", "raw_dir": str(out / "raw"), "sources": sources,
                       "note": "healthy (patient 6) sections absent from the local HEST copy; ids follow the HEST catalogue; "
                               "no patches (see metadata.image_note)"})
    files = json.loads((out / "manifest.json").read_text())["files_sha256"]
    for p in sorted((out / "annotations").glob("*.csv")):
        files[f"annotations/{p.name}"] = cv.sha256(p)
    man = json.loads((out / "manifest.json").read_text())
    man["files_sha256"] = files
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str) + "\n")
    for r in rows:
        print(r)


def check(out: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from morpheus.tier2.substrates import hest_multi as hm
    spec = hm.StudySpec(key="valdeolivas2024_crc_healthy", dataset_title="Valdeolivas 2024 healthy sections (Zenodo 7760264)",
                        role="control", layout="flat", sample_ids=tuple(SECTIONS),
                        patient_of={sid: "valdeolivas2024_crc:Patient 6" for sid in SECTIONS}, organ="Bowel",
                        st_technology="Visium", platform="visium", embed_roots=(), data_dir=str(out),
                        platform_of={sid: "visium" for sid in SECTIONS})
    raw = hm.load_study(spec, cache_root=None)
    import numpy as np
    ids, counts = np.unique(raw.sample_id, return_counts=True)
    print("loaded rows:", raw.n, "| per sample:", dict(zip(ids.tolist(), counts.tolist())),
          "| patients:", sorted(set(raw.patient.tolist())), "| genes:", raw.meta.get("n_genes", None) or getattr(raw, "genes", np.array([])).shape)
    print("meta keys:", sorted(k for k in raw.meta if not k.startswith("_"))[:20])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zenodo-dir", default=f"{STORE}/spatial/second_wave_20261022/valdeolivas_zenodo_7760264")
    ap.add_argument("--out", default=f"{STORE}/spatial/second_wave_20261022/valdeolivas_healthy_hest_layout")
    ap.add_argument("--catalogue", default=CATALOGUE)
    ap.add_argument("--check", action="store_true", help="only load the converted sections through hest_multi (read-only)")
    a = ap.parse_args()
    if a.check:
        check(Path(a.out))
    else:
        convert(Path(a.zenodo_dir), Path(a.out), a.catalogue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
