"""Convert the two second-wave GEO colorectal cohorts into the flat HEST layout with the existing converter
dbench/scripts/convert_external_visium.py (imported, not modified).

  gse294385      37 Visium sections (Liu et al. 2026) -> <root>/geo_GSE294385_hest_layout/{st,metadata}/ ; no patches (no WSI).
                 GEO carries no patient identity: patient key = sample id (patient-held-out folds are sample-held-out).
  gse280318_hd   5 Visium HD sections (Oliveira et al. 2025) -> <root>/geo_GSE280318_visiumhd_hest_layout/
                   native_8um/          bins as served (8 um)
                   pseudo_visium_56um/  bins summed onto a 56 um square grid (7 x 7 bins) aligned to the array
                                        indices of tissue_positions: counts summed, full-res coordinates averaged,
                                        a grid cell kept only if >= 50 % of its 49 bins (>= 25) are under tissue.
  check          load every layout read-only through hest_multi (in-memory StudySpec, cache_root=None).

Usage: python scripts/second_wave/convert_geo_crc.py {gse294385|gse280318_hd|check} [--root DIR]
Downloaded files are untrusted data: they are parsed, never executed.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
SW = f"{STORE}/spatial/second_wave_20261022"
GEO_TERMS = "GEO public (no explicit licence; NCBI places no restrictions on use or distribution)"
HD_SECTIONS = {"P1CRC": ("GSM8594567", "P1", "colon cancer"), "P2CRC": ("GSM8594568", "P2", "colon cancer"),
               "P5CRC": ("GSM8594569", "P5", "colon cancer"), "P3NAT": ("GSM8594570", "P3", "normal adjacent tissue"),
               "P5NAT": ("GSM8594571", "P5", "normal adjacent tissue")}
PSEUDO_UM, BIN_UM = 56.0, 8.0
BINS_PER_SIDE = int(PSEUDO_UM // BIN_UM)                 # 7
MIN_BINS = int(np.ceil(0.5 * BINS_PER_SIDE ** 2))        # 25 of 49


def _converter():
    spec = importlib.util.spec_from_file_location("convert_external_visium",
                                                  ROOT / "dbench" / "scripts" / "convert_external_visium.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_readme(out: Path, text: str) -> None:
    (out / "README.md").write_text(text.strip() + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- GSE294385
def _gsm_records(family_text: Path) -> dict:
    recs, cur = {}, None
    for line in family_text.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("^SAMPLE = "):
            cur = line.split("= ", 1)[1].strip(); recs[cur] = {}
        elif cur and line.startswith("!Sample_title = "):
            recs[cur]["title"] = line.split("= ", 1)[1].strip()
        elif cur and line.startswith("!Sample_source_name_ch1 = "):
            recs[cur]["source_name"] = line.split("= ", 1)[1].strip()
        elif cur and line.startswith("!Sample_characteristics_ch1 = tissue: "):
            recs[cur]["tissue"] = line.split("tissue: ", 1)[1].strip()
    return recs


def convert_gse294385(root: Path) -> None:
    cv = _converter()
    raw = root / "geo_GSE294385"
    out = root / "geo_GSE294385_hest_layout"
    out.mkdir(parents=True, exist_ok=True)
    recs = _gsm_records(raw / "GSE294385_family_text.txt")
    rows = []
    for d in sorted((raw / "extracted").iterdir()):
        gsm, title = d.name.split("_", 1)
        inner = d / title
        fm = inner / "filtered_feature_bc_matrix"
        files = {"matrix": fm / "matrix.mtx.gz", "features": fm / "features.tsv.gz", "barcodes": fm / "barcodes.tsv.gz",
                 "positions": inner / "spatial" / "tissue_positions.csv",
                 "scalefactors": inner / "spatial" / "scalefactors_json.json",
                 "hires": inner / "spatial" / "tissue_hires_image.png"}
        sid = f"GSE294385_{title}"
        rec = recs.get(gsm, {})
        meta = {"patient": sid, "patient_note": "GEO carries no patient identity for this series; patient key = sample id, so "
                                                "patient-held-out folds are sample-held-out (the paper reports 11 patients)",
                "gsm": gsm, "sample_title": rec.get("title", title), "tissue": rec.get("tissue", "Colorectal cancer metastasis"),
                "source_name": rec.get("source_name"), "organ": "Bowel", "oncotree_code": "COADREAD",
                "disease_state": "Cancer (metastasis)", "st_technology": "Visium",
                "chemistry": "Visium (Space Ranger output; CytAssist image present for the M-ST samples, absent for IX samples)",
                "dataset_title": "Spatial Multi-Omics Landscape of Colorectal Cancer Macro- and Micrometastases (GSE294385, "
                                 "Liu et al. 2026 Cancer Cell)", "license": GEO_TERMS, "source_accession": f"GSE294385/{gsm}",
                "image_note": "Space Ranger hires PNG (<= 2000 px) and CytAssist image only; no full-resolution H&E in GEO: "
                              "expression + spatial only, no patches, no morphology view",
                "with_cytassist_image": (inner / "spatial" / "cytassist_image.tiff").exists()}
        rows.append(cv.convert_spaceranger_dir(files, out, sid, meta))
        print(rows[-1], flush=True)
    cv.write_manifest(out, "geo_GSE294385", rows, {
        "license": GEO_TERMS, "raw_dir": str(raw), "raw_sha256_manifest": str(raw / "sha256_manifest.txt"),
        "patient_identity": "absent in GEO: patient key = sample id; patient-held-out folds are sample-held-out folds; "
                            "the paper reports 11 patients, so sections of one patient may fall in different folds",
        "images": "no WSI (hires PNG + CytAssist image only); no patches; no annotations",
        "converter": "dbench/scripts/convert_external_visium.py (convert_spaceranger_dir) via scripts/second_wave/convert_geo_crc.py"})
    _write_readme(out, f"""
# GSE294385 in the flat HEST layout (st/, metadata/; no patches)

37 Visium sections of colorectal cancer metastases (Liu et al. 2026, Cancer Cell), molecular only.
Ids: GSE294385_<sample title>; metadata carries the GSM id, the sample title and the GEO tissue field
("Colorectal cancer metastasis" for every sample).

LIMITATION: GEO carries no patient identity (and no site label) for this series. The patient key equals the
sample id, so patient-held-out folds are sample-held-out folds: sections from the same patient (the paper
reports 11 patients) may sit on both sides of a split. Use this cohort for molecular replication with that
caveat stated; no morphology, H-Optimus-0 or CellViT view is possible (no WSI in GEO); no annotations.

GEO terms: {GEO_TERMS}. Raw files and their sha256: {raw}/sha256_manifest.txt. manifest.json lists every written
file with sha256.
""")


# ----------------------------------------------------------------------------- GSE280318 Visium HD
def _hd_inputs(raw: Path, sec: str, gsm: str):
    pre = raw / sec / f"{gsm}_{sec}_"
    pos = pd.read_parquet(io.BytesIO(gzip.open(f"{pre}tissue_positions.parquet.gz").read()))
    pos = pos.set_index("barcode")
    sf = json.loads(gzip.open(f"{pre}scalefactors_json.json.gz").read())
    return pos, sf, f"{pre}filtered_feature_bc_matrix.h5"


def _pseudo_visium(X, bc, pos: pd.DataFrame):
    """Sum 8 um bins onto a 56 um square grid: cell (r, c) = (array_row // 7, array_col // 7)."""
    p = pos.reindex(bc)
    cr = (p["array_row"].to_numpy() // BINS_PER_SIDE).astype(np.int64)
    cc = (p["array_col"].to_numpy() // BINS_PER_SIDE).astype(np.int64)
    key = cr * 100000 + cc
    uniq, inv, nbins = np.unique(key, return_inverse=True, return_counts=True)
    A = sp.csr_matrix((np.ones(len(bc), dtype=np.float32), (inv, np.arange(len(bc)))), shape=(len(uniq), len(bc)))
    Xp = (A @ X).tocsr()
    xy_r = A @ p["pxl_row_in_fullres"].to_numpy() / nbins
    xy_c = A @ p["pxl_col_in_fullres"].to_numpy() / nbins
    keep = nbins >= MIN_BINS
    rows = uniq // 100000
    cols = uniq % 100000
    barcodes = np.array([f"PV56_{r:03d}_{c:03d}" for r, c in zip(rows, cols)])
    ppos = pd.DataFrame({"in_tissue": 1, "array_row": rows, "array_col": cols,
                         "pxl_row_in_fullres": xy_r, "pxl_col_in_fullres": xy_c, "n_bins": nbins}, index=barcodes)
    return Xp[keep], barcodes[keep], ppos.iloc[keep], {"n_cells_touched": int(len(uniq)), "n_kept": int(keep.sum()),
                                                        "median_bins_per_kept_pseudo_spot": float(np.median(nbins[keep])),
                                                        "bins_in_kept": int(nbins[keep].sum()), "bins_total": int(len(bc))}


def convert_gse280318_hd(root: Path) -> None:
    cv = _converter()
    raw = root / "geo_GSE280318_visiumhd"
    out = root / "geo_GSE280318_visiumhd_hest_layout"
    native, pseudo = out / "native_8um", out / "pseudo_visium_56um"
    rows_n, rows_p, agg = [], [], {}
    rule = (f"pseudo_visium_56um: 8 um bins summed onto a {PSEUDO_UM:g} um square grid aligned to the array indices of "
            f"tissue_positions (cell = (array_row // {BINS_PER_SIDE}, array_col // {BINS_PER_SIDE}), {BINS_PER_SIDE}x{BINS_PER_SIDE} "
            f"= {BINS_PER_SIDE**2} bins per cell); counts summed, full-resolution pixel coordinates averaged over the bins under "
            f"tissue; a cell is kept only if >= {MIN_BINS} of its {BINS_PER_SIDE**2} bins (>= 50 %) are under tissue; the pseudo-spot "
            f"diameter is recorded as {PSEUDO_UM:g} um (spot_diameter_fullres = 56 / microns_per_pixel)")
    for sec, (gsm, patient, tissue) in HD_SECTIONS.items():
        pos, sf, h5 = _hd_inputs(raw, sec, gsm)
        X, feats, bc = cv.read_10x_h5(h5)
        mpp = float(sf["microns_per_pixel"])
        base = {"patient": f"GSE280318_{patient}", "gsm": gsm, "section": sec, "tissue": tissue, "organ": "Bowel",
                "oncotree_code": "COADREAD", "disease_state": "Cancer" if "CRC" in sec else "Normal adjacent",
                "st_technology": "Visium HD", "chemistry": "Visium HD FFPE (human probe set), Space Ranger v3.0",
                "preservation_method": "FFPE",
                "dataset_title": "Characterization of immune cell populations in the tumor microenvironment of colorectal cancer "
                                 "using high definition spatial profiling [VisiumHD] (GSE280315 / GSE280318, Oliveira et al. 2025)",
                "license": GEO_TERMS, "source_accession": f"GSE280315/{gsm}", "pixel_size_um": mpp,
                "pixel_size_um_source": "scalefactors microns_per_pixel", "bin_size_um": float(sf["bin_size_um"]),
                "image_note": "tissue_image.btf (full-resolution H&E) exists in GEO but was not downloaded; lowres PNG only "
                              "here: expression + spatial only, no patches",
                "meta_training_overlap": "the Visium v2 section P2CRC of this study is HEST TENX152 (meta_training study "
                                         "tenx_demo_bowel); HD sections are not in HEST; replication must use HD sections only"}
        sid = f"GSE280318HD_{sec}"
        md = dict(base, layout="native_8um", spot_unit="8 um bin as served")
        rows_n.append(cv.write_section(native, sid, X, feats, bc, pos, sf, md))
        print("native", rows_n[-1], flush=True)
        Xp, bcp, ppos, stats = _pseudo_visium(X, bc, pos)
        sfp = dict(sf, spot_diameter_fullres=PSEUDO_UM / mpp, pseudo_spot_um=PSEUDO_UM, bins_per_pseudo_spot_max=BINS_PER_SIDE ** 2,
                   min_bins_per_pseudo_spot=MIN_BINS)
        mdp = dict(base, layout="pseudo_visium_56um", spot_unit=f"{PSEUDO_UM:g} um pseudo-spot (sum of 8 um bins)",
                   aggregation_rule=rule, **{f"pseudo_{k}": v for k, v in stats.items()})
        rows_p.append(cv.write_section(pseudo, sid, Xp, feats, bcp, ppos.drop(columns=["n_bins"]), sfp, mdp))
        agg[sid] = stats
        print("pseudo", rows_p[-1], stats, flush=True)
        del X, Xp
    common = {"license": GEO_TERMS, "raw_dir": str(raw), "raw_sha256_manifest": str(raw / "sha256_manifest.txt"),
              "patients": {s: f"GSE280318_{p}" for s, (g, p, t) in HD_SECTIONS.items()},
              "meta_training_overlap": "P2CRC Visium v2 = HEST TENX152 (meta_training); HD sections only for replication",
              "converter": "dbench/scripts/convert_external_visium.py (read_10x_h5, write_section) via scripts/second_wave/convert_geo_crc.py"}
    cv.write_manifest(native, "geo_GSE280318_visiumhd_native_8um", rows_n, dict(common, layout="native_8um: bins as served"))
    cv.write_manifest(pseudo, "geo_GSE280318_visiumhd_pseudo_visium_56um", rows_p,
                      dict(common, layout="pseudo_visium_56um", aggregation_rule=rule, aggregation_stats=agg))
    top = {"schema": "second_wave_two_layout_cohort_v1", "cohort": "geo_GSE280318_visiumhd",
           "layouts": {"native_8um": "bins as served (8 um); manifest native_8um/manifest.json",
                       "pseudo_visium_56um": rule + "; manifest pseudo_visium_56um/manifest.json"},
           "aggregation_stats": agg, **common}
    (out / "manifest.json").write_text(json.dumps(top, indent=2, default=str) + "\n")
    _write_readme(out, f"""
# GSE280318 Visium HD in the flat HEST layout, two layouts

* native_8um/  (st/, metadata/): the 8 um bins as served by GEO (GSE280315), one h5ad per section.
* pseudo_visium_56um/ (st/, metadata/): {rule}.

Sections: P1CRC, P2CRC, P5CRC (tumour), P3NAT, P5NAT (normal adjacent); patients P1, P2, P3, P5 (P5CRC and P5NAT
share patient P5). No patches: the full-resolution tissue_image.btf files were not downloaded.

Meta-training overlap: the Visium v2 section P2CRC of this study is HEST TENX152, a member of the meta_training
study tenx_demo_bowel; the HD sections are not in HEST. Any replication must use the HD sections only and say so.

GEO terms: {GEO_TERMS}. Raw files and sha256: {raw}/sha256_manifest.txt.
""")


# ----------------------------------------------------------------------------- read-only load check
def check(root: Path, max_samples: int = 0) -> None:
    """Read-only hest_multi load of each layout; max_samples > 0 restricts every study to its first N sections
    (a full-study load of gse294385 passed 87 GB RSS on the shared box and was stopped)."""
    sys.path.insert(0, str(ROOT))
    from morpheus.tier2.substrates import hest_multi as hm

    def load(key, data_dir: Path, patient_of, cap: bool = True):
        if max_samples and cap:          # the 56 um pseudo-spot layout is small by construction and always loads whole
            patient_of = dict(list(patient_of.items())[:max_samples])
            key = f"{key} (first {len(patient_of)} section(s) only)"
        spec = hm.StudySpec(key=key, dataset_title=key, role="external_replication", layout="flat",
                            sample_ids=tuple(patient_of), patient_of=patient_of, organ="Bowel", st_technology="Visium",
                            platform="visium", embed_roots=(), data_dir=str(data_dir),
                            platform_of={s: "visium" for s in patient_of})
        raw = hm.load_study(spec, cache_root=None)
        ids, counts = np.unique(raw.sample_id, return_counts=True)
        print(f"## {key}: rows {raw.n} | genes {len(raw.genes)} | convention {raw.meta.get('genes_convention')} | "
              f"patients {len(set(raw.patient.tolist()))} | dropped {raw.meta.get('dropped_samples')}")
        print("   rows per sample:", dict(zip(ids.tolist(), counts.tolist())), flush=True)

    d = root / "geo_GSE294385_hest_layout"
    if (d / "manifest.json").exists():
        ids = [r["id"] for r in json.loads((d / "manifest.json").read_text())["sections"]]
        load("gse294385", d, {s: f"gse294385:{s}" for s in ids})
    for layout in ("native_8um", "pseudo_visium_56um"):
        d = root / "geo_GSE280318_visiumhd_hest_layout" / layout
        if (d / "manifest.json").exists():
            load(f"gse280318_hd_{layout}", d, {f"GSE280318HD_{s}": f"gse280318_hd:GSE280318_{p}" for s, (g, p, t) in HD_SECTIONS.items()},
                 cap=(layout == "native_8um"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["gse294385", "gse280318_hd", "check"])
    ap.add_argument("--root", default=SW)
    ap.add_argument("--max-samples", type=int, default=0, help="check: load only the first N sections of each study")
    a = ap.parse_args()
    root = Path(a.root)
    if a.task == "check":
        check(root, a.max_samples)
    else:
        {"gse294385": convert_gse294385, "gse280318_hd": convert_gse280318_hd}[a.task](root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
