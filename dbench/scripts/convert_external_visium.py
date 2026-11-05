#!/usr/bin/env python3
"""Convert external Visium cohorts into the flat HEST-1k layout (Agent E).

Output per section ``<out>/st/{id}.h5ad``, ``<out>/metadata/{id}.json`` and,
only where a full-resolution H&E image exists, ``<out>/patches/{id}.h5``:

* st: raw counts (CSR float32 of integer values, as HEST), ``obs`` with
  ``in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres,
  total_counts, n_genes_by_counts``; ``var`` indexed by gene symbol (made
  unique) with ``gene_ids``; ``obsm["spatial"] = [x, y] = [pxl_col, pxl_row]``
  in full-resolution pixels; ``uns["spatial"]["ST"]["scalefactors"]`` with
  ``spot_diameter_fullres`` (and the SpaceRanger factors when known).
* metadata: ``id, patient, organ, oncotree_code, st_technology,
  dataset_title, license, source_accession, pixel_size_um, with_morphology``
  plus chemistry and provenance fields.
* patches: HEST key layout -- ``img`` uint8 (N,224,224,3) with attrs
  ``pixel_size, downsample, patch_size_src, patch_size_target``; ``barcode``
  (N,1) bytes; ``coords`` (N,2) int64 top-left corner in full-res pixels.
  Tiles cover 112 um (224 px at 0.5 um/px) centred on the spot; spots whose
  tile leaves the image are dropped, as HEST does.

Sources: ``mendeley_li2022`` (Li et al. 2022 Cancer Cell, Mendeley
10.17632/g67bkbnhhg.1), ``geo`` (GSE250163), ``zenodo`` (record 14620362),
``kiviaho`` (GSE278936 prostate, expression + spatial only).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

VISIUM_SPOT_UM = 55.0
TARGET_UM_PER_PX = 0.5
PATCH_TARGET = 224
POS_COLS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _open(p):
    p = str(p)
    return gzip.open(p, "rb") if p.endswith(".gz") else open(p, "rb")


# ----------------------------------------------------------------------------- readers
def read_positions(src) -> pd.DataFrame:
    """SpaceRanger tissue_positions(_list).csv, with or without header."""
    raw = src.read() if hasattr(src, "read") else _open(src).read()
    first = raw.split(b"\n", 1)[0].decode()
    header = 0 if first.startswith("barcode") else None
    df = pd.read_csv(io.BytesIO(raw), header=header)
    if header is None:
        df.columns = POS_COLS
    return df.set_index("barcode")


def read_mtx_triplet(matrix, features, barcodes):
    """10x MTX (genes x barcodes) -> CSR (barcodes x genes), features df, barcodes."""
    from scipy.io import mmread

    m = mmread(_open(matrix) if not hasattr(matrix, "read") else matrix)
    X = sp.csr_matrix(m.T)
    f = pd.read_csv(_open(features) if not hasattr(features, "read") else features, sep="\t", header=None)
    f = f.iloc[:, :3] if f.shape[1] >= 3 else f.assign(ft="Gene Expression")
    f.columns = ["gene_ids", "symbol", "feature_types"]
    b = pd.read_csv(_open(barcodes) if not hasattr(barcodes, "read") else barcodes, sep="\t", header=None)[0]
    return X, f, b.astype(str).to_numpy()


def read_10x_h5(path):
    import h5py

    with h5py.File(path, "r") as h:
        g = h["matrix"]
        shape = tuple(g["shape"][:])
        X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape).T.tocsr()
        bc = g["barcodes"][:].astype(str)
        f = g["features"]
        feats = pd.DataFrame({"gene_ids": f["id"][:].astype(str), "symbol": f["name"][:].astype(str),
                              "feature_types": f["feature_type"][:].astype(str)})
    return X, feats, bc


# ----------------------------------------------------------------------------- writer
def make_patches(img: np.ndarray, centres_xy: np.ndarray, barcodes: np.ndarray, pixel_size_um: float):
    """224-px tiles covering 112 um centred on each spot (HEST convention)."""
    from PIL import Image

    downsample = TARGET_UM_PER_PX / pixel_size_um
    src = int(round(PATCH_TARGET * downsample))
    half = src // 2
    H, W = img.shape[:2]
    keep, tiles, tl = [], [], []
    for i, (x, y) in enumerate(np.rint(centres_xy).astype(np.int64)):
        x0, y0 = x - half, y - half
        if x0 < 0 or y0 < 0 or x0 + src > W or y0 + src > H:
            continue
        t = np.asarray(img[y0:y0 + src, x0:x0 + src, :3])
        if src != PATCH_TARGET:
            t = np.asarray(Image.fromarray(t).resize((PATCH_TARGET, PATCH_TARGET), Image.Resampling.BICUBIC))
        tiles.append(t.astype(np.uint8))
        tl.append((x0, y0))
        keep.append(i)
    attrs = {"pixel_size": float(pixel_size_um), "downsample": float(downsample),
             "patch_size_src": int(src), "patch_size_target": PATCH_TARGET}
    imgs = np.stack(tiles) if tiles else np.zeros((0, PATCH_TARGET, PATCH_TARGET, 3), np.uint8)
    return imgs, np.asarray(barcodes)[keep], np.asarray(tl, dtype=np.int64).reshape(-1, 2), attrs


def write_section(out: Path, sid: str, X, feats: pd.DataFrame, barcodes, positions: pd.DataFrame,
                  scalefactors: dict, meta: dict, fullres_image: np.ndarray | None = None,
                  fullres_pixel_size_um: float | None = None, hires_image: np.ndarray | None = None,
                  in_tissue_only: bool = True) -> dict:
    """Write st/metadata(/patches) for one section; returns a summary row."""
    import anndata as ad
    import h5py

    barcodes = np.asarray(barcodes).astype(str)
    pos = positions.reindex(barcodes)
    if pos.isna().any(axis=None):
        missing = int(pos.isna().any(axis=1).sum())
        raise ValueError(f"{sid}: {missing} barcodes without positions")
    X = sp.csr_matrix(X)
    if in_tissue_only:
        m = pos["in_tissue"].to_numpy().astype(bool)
        X, barcodes, pos = X[m], barcodes[m], pos.iloc[m]
    if X.nnz and not np.allclose(X.data, np.round(X.data)):
        raise ValueError(f"{sid}: counts are not integers (normalised input?)")
    sym = feats["symbol"].astype(str).to_numpy()
    var = pd.DataFrame({"gene_ids": feats["gene_ids"].astype(str).to_numpy(),
                        "feature_types": feats["feature_types"].astype(str).to_numpy()},
                       index=ad.utils.make_index_unique(pd.Index(sym, name=None)))
    obs = pd.DataFrame({c: pos[c].to_numpy() for c in POS_COLS[1:]}, index=pd.Index(barcodes, name=None))
    obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    obs["n_genes_by_counts"] = np.diff(X.indptr).astype(np.int64)
    a = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    xy = np.c_[pos["pxl_col_in_fullres"].to_numpy(), pos["pxl_row_in_fullres"].to_numpy()].astype(np.float64)
    a.obsm["spatial"] = xy
    sf = {k: float(v) for k, v in scalefactors.items()}
    st = {"scalefactors": sf, "images": {}}
    if hires_image is not None:
        st["images"]["hires"] = np.asarray(hires_image)
    a.uns["spatial"] = {"ST": st}
    for d in ("st", "metadata", "patches"):
        (out / d).mkdir(parents=True, exist_ok=True)
    st_path = out / "st" / f"{sid}.h5ad"
    a.write_h5ad(st_path)
    md = dict(meta)
    md.update(id=sid, spots_under_tissue=int(a.n_obs), nb_genes=int(a.n_vars),
              spot_diameter_fullres=sf.get("spot_diameter_fullres"),
              with_morphology=fullres_image is not None, converted_utc=_utc())
    n_patches = 0
    if fullres_image is not None:
        imgs, bc_keep, tl, attrs = make_patches(fullres_image, xy, barcodes, float(fullres_pixel_size_um))
        with h5py.File(out / "patches" / f"{sid}.h5", "w") as h:
            d = h.create_dataset("img", data=imgs, chunks=(1, PATCH_TARGET, PATCH_TARGET, 3) if len(imgs) else None)
            for k, v in attrs.items():
                d.attrs[k] = v
            h.create_dataset("barcode", data=np.asarray(bc_keep, dtype=object).astype("S").reshape(-1, 1))
            h.create_dataset("coords", data=tl)
        n_patches = int(len(imgs))
        md["n_patches"] = n_patches
    (out / "metadata" / f"{sid}.json").write_text(json.dumps(md, indent=2, default=str) + "\n")
    return {"id": sid, "patient": md.get("patient"), "n_spots": int(a.n_obs), "n_genes": int(a.n_vars),
            "n_patches": n_patches, "with_morphology": md["with_morphology"],
            "st_sha256": sha256(st_path)}


def write_manifest(out: Path, source: str, rows: list[dict], extra: dict) -> None:
    man = {"schema": "external_visium_conversion_v1", "source": source, "created_utc": _utc(),
           "n_sections": len(rows), "n_patients": len({r["patient"] for r in rows}),
           "n_spots": int(sum(r["n_spots"] for r in rows)), "sections": rows, **extra}
    files = {}
    for d in ("st", "metadata", "patches"):
        for p in sorted((out / d).glob("*")):
            files[f"{d}/{p.name}"] = sha256(p)
    man["files_sha256"] = files
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str) + "\n")


def pixel_size_from_spot(spot_diameter_fullres: float) -> float:
    return VISIUM_SPOT_UM / float(spot_diameter_fullres)


def _read_png(src) -> np.ndarray:
    from PIL import Image

    raw = src.read() if hasattr(src, "read") else _open(src).read()
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


# ----------------------------------------------------------------------------- sources
def convert_spaceranger_dir(files: dict, out: Path, sid: str, meta: dict, fullres_image_path=None) -> dict:
    """files: matrix_h5 | (matrix, features, barcodes); positions; scalefactors; optional hires."""
    if "matrix_h5" in files:
        X, feats, bc = read_10x_h5(files["matrix_h5"])
    else:
        X, feats, bc = read_mtx_triplet(files["matrix"], files["features"], files["barcodes"])
    pos = read_positions(files["positions"])
    sf = json.loads(_open(files["scalefactors"]).read())
    px = pixel_size_from_spot(sf["spot_diameter_fullres"])
    hires = _read_png(files["hires"]) if files.get("hires") else None
    md = dict(meta, pixel_size_um=px, pixel_size_um_source="55 um / spot_diameter_fullres")
    if hires is not None and "tissue_hires_scalef" in sf:
        md["hires_um_per_px"] = px / float(sf["tissue_hires_scalef"])
    img = None
    if fullres_image_path is not None:
        import tifffile

        img = tifffile.imread(str(fullres_image_path))
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        md["fullres_image"] = str(fullres_image_path)
        md["fullres_px_height"], md["fullres_px_width"] = int(img.shape[0]), int(img.shape[1])
    return write_section(out, sid, X, feats, bc, pos, sf, md, fullres_image=img, fullres_pixel_size_um=px,
                         hires_image=hires)


def convert_geo_gse250163(raw: Path, out: Path) -> list[dict]:
    tar = raw / "GSE250163_RAW.tar"
    stage = raw / "extracted"
    if not stage.is_dir():
        stage.mkdir()
        with tarfile.open(tar) as t:
            t.extractall(stage, filter="data")
    rows = []
    for h5 in sorted(stage.glob("GSM*_filtered_feature_bc_matrix.h5")):
        pre = h5.name.replace("_filtered_feature_bc_matrix.h5", "")
        gsm, sec = pre.split("_", 1)
        files = {"matrix_h5": h5, "positions": stage / f"{pre}_tissue_positions_list.csv.gz",
                 "scalefactors": stage / f"{pre}_scalefactors_json.json.gz",
                 "hires": stage / f"{pre}_tissue_hires_image.png.gz"}
        meta = {"patient": f"GSE250163_{sec}", "organ": "Kidney", "oncotree_code": "CCRCC",
                "st_technology": "Visium", "chemistry": "Visium FFPE (human probe set, v1)",
                "preservation_method": "FFPE",
                "dataset_title": "Role of copper in progression of clear cell renal cell carcinoma [Spatial transcriptomics]",
                "license": "GEO public (no explicit licence; NCBI GEO data are freely available)",
                "source_accession": f"GSE250163/{gsm}", "section": sec, "stage": "III",
                "image_note": "hires PNG only (no full-resolution H&E in GEO): expression + spatial only"}
        rows.append(convert_spaceranger_dir(files, out, f"GSE250163_{sec}", meta))
    return rows


def convert_zenodo_14620362(raw: Path, out: Path) -> list[dict]:
    """Counts from SpaceRanger filtered matrices; spot centres in the deposited TIF frame from the
    Loupe manual alignment (oligo row/col -> imageX/imageY; a similarity transform of the SpaceRanger
    full-res coordinates, max residual ~1.3 px); manual annotations from h5ad_preprocessed."""
    import zipfile

    import anndata as ad
    import tifffile

    stage = raw / "extracted"
    if not stage.is_dir():
        stage.mkdir()
        with zipfile.ZipFile(raw / "TLS_VISIUM_USZ.zip") as z:  # kidney (KC*) members only; LC* are lung
            z.extractall(stage, members=[n for n in z.namelist() if not re.search(r"(^|/)LC\d", n)])
    root = next(stage.rglob("10x_Visium")).parent
    rows = []
    for d in sorted(p for p in (root / "10x_Visium").glob("KC*") if p.is_dir()):
        kc = d.name
        fm = d / "filtered_feature_bc_matrix"
        X, feats, bc = read_mtx_triplet(next(fm.glob("matrix.mtx*")), next(fm.glob("features.tsv*")),
                                        next(fm.glob("barcodes.tsv*")))
        pos = read_positions(d / "spatial" / "tissue_positions.csv")
        sr_sf = json.loads((d / "spatial" / "scalefactors_json.json").read_text())
        al = json.loads((root / "tif_slides" / f"{kc}_manual_loupe_alignment.json").read_text())
        olig = pd.DataFrame(al["oligo"]).set_index(["row", "col"])
        key = list(zip(pos["array_row"], pos["array_col"]))
        pos["pxl_col_in_fullres"] = olig.loc[key, "imageX"].to_numpy()
        pos["pxl_row_in_fullres"] = olig.loc[key, "imageY"].to_numpy()
        dia = float(olig["dia"].iloc[0])
        sf = {"spot_diameter_fullres": dia, "fiducial_diameter_fullres": float(al["fiducial"][0]["dia"]),
              "spaceranger_spot_diameter_fullres": float(sr_sf["spot_diameter_fullres"])}
        tif = root / "tif_slides" / f"{kc}.tif"
        with tifffile.TiffFile(str(tif)) as t:
            img = t.pages[0].asarray()
            xr = t.pages[0].tags.get("XResolution")
            imagej_um = (xr.value[1] / xr.value[0]) if xr else None
        px = pixel_size_from_spot(dia)
        meta = {"patient": f"Z14620362_{kc}", "organ": "Kidney", "oncotree_code": "CCRCC",
                "st_technology": "Visium", "chemistry": "Visium FFPE (Human Probe Set v1), Space Ranger 2.1.0",
                "preservation_method": "FFPE",
                "dataset_title": "10x Visium Spatial Transcriptomics Dataset: Kidney (3) and Lung (5) Cancer with "
                                 "Tertiary Lymphoid Structures",
                "license": "CC BY 4.0", "source_accession": f"zenodo:14620362/{kc}",
                "pixel_size_um": px, "pixel_size_um_source": "55 um / Loupe alignment spot diameter (TIF frame)",
                "pixel_size_um_imagej_tag": imagej_um, "fullres_image": str(tif),
                "fullres_px_height": int(img.shape[0]), "fullres_px_width": int(img.shape[1]),
                "meets_0p5_um_per_px_rule": bool(px <= 0.5),
                "histology_note": "record states 'kidney cancer'; ccRCC subtype not stated in the record"}
        Xs = sp.csr_matrix(X)
        sec = write_section(out, f"Z14620362_{kc}", Xs, feats, bc, pos, sf, meta, fullres_image=img,
                            fullres_pixel_size_um=px)
        # manual annotations (TLS / Immune / Tumour / Normal) from the authors' preprocessed AnnData
        pre = ad.read_h5ad(root / "h5ad_preprocessed" / f"{kc}.h5ad", backed="r")
        st_path = out / "st" / f"Z14620362_{kc}.h5ad"
        a = ad.read_h5ad(st_path)
        for c in ("manual_anno", "manual_anno_tls", "ground_truth"):
            if c in pre.obs:
                a.obs[c] = pre.obs[c].astype(str).reindex(a.obs_names).fillna("NOT_IN_PREPROCESSED").to_numpy()
        a.write_h5ad(st_path)
        sec["st_sha256"] = sha256(st_path)
        rows.append(sec)
    return rows


def convert_mendeley_li2022(raw: Path, out: Path) -> list[dict]:
    """Merged AnnData per region; split by section using obs / uns['spatial'] keys."""
    import anndata as ad

    rows = []
    for fn, region in (("visium_merge_inter_upload.h5ad", "interface"), ("visium_merge_core_upload.h5ad", "core")):
        a = ad.read_h5ad(raw / fn)
        sec_col = _section_column(a)
        Xraw = a.raw.X if a.raw is not None and a.raw.shape[0] == a.n_obs else a.X
        var = a.raw.var if Xraw is not a.X else a.var
        if "SYMBOL" in var:  # Li 2022 upload: index = Ensembl id, SYMBOL column
            gid, sym = var.index.astype(str).to_numpy(), var["SYMBOL"].astype(str).to_numpy()
        else:
            sym = var.index.astype(str).to_numpy()
            gid = var["gene_ids"].astype(str).to_numpy() if "gene_ids" in var else sym
        feats = pd.DataFrame({"gene_ids": gid, "symbol": sym, "feature_types": "Gene Expression"})
        spatial = a.uns.get("spatial", {})
        for sec in sorted(a.obs[sec_col].astype(str).unique()):
            m = (a.obs[sec_col].astype(str) == sec).to_numpy()
            sub = a.obs.loc[m]
            xy = np.asarray(a.obsm["spatial"])[m]
            bc = sub.index.astype(str).to_numpy()
            pos = pd.DataFrame({"in_tissue": sub["in_tissue"].to_numpy() if "in_tissue" in sub else 1,
                                "array_row": sub["array_row"].to_numpy() if "array_row" in sub else -1,
                                "array_col": sub["array_col"].to_numpy() if "array_col" in sub else -1,
                                "pxl_row_in_fullres": xy[:, 1], "pxl_col_in_fullres": xy[:, 0]}, index=bc)
            lib = _library_key(spatial, sec)
            sf = dict(spatial[lib]["scalefactors"]) if lib else {}
            hires = spatial[lib]["images"].get("hires") if lib and "images" in spatial[lib] else None
            pat = _patient_of(sub)
            px = pixel_size_from_spot(sf["spot_diameter_fullres"]) if "spot_diameter_fullres" in sf else None
            meta = {"patient": f"Li2022_{pat}", "organ": "Kidney", "oncotree_code": "CCRCC",
                    "st_technology": "Visium", "chemistry": "Visium fresh-frozen (poly-A capture)",
                    "preservation_method": "Fresh frozen", "region": region,
                    "region_code": str(sub["region"].iloc[0]) if "region" in sub else None,
                    "spaceranger_library": lib,
                    "spaceranger_metadata": dict(spatial[lib].get("metadata", {})) if lib else None,
                    "dataset_title": "Mapping single cell transcriptomes in kidney cancer (Li et al. 2022 Cancer Cell)",
                    "license": "CC BY 4.0", "source_accession": f"Mendeley 10.17632/g67bkbnhhg.1/{fn}:{sec}",
                    "pixel_size_um": px, "pixel_size_um_source": "55 um / spot_diameter_fullres",
                    "image_note": "only the downscaled image embedded in the AnnData: expression + spatial only"}
            if px is not None and "tissue_hires_scalef" in sf:
                meta["hires_um_per_px"] = px / float(sf["tissue_hires_scalef"])
            sid = "Li2022_" + re.sub(r"[^A-Za-z0-9]+", "_", str(sec)).strip("_")
            rows.append(write_section(out, sid, Xraw[m], feats, bc, pos, sf, meta, hires_image=hires))
    return rows


def _section_column(a) -> str:
    for c in ("library_id", "sample", "section", "sample_id", "batch", "Sample", "orig.ident"):
        if c in a.obs and a.obs[c].nunique() > 1:
            return c
    raise ValueError(f"no section column among {list(a.obs.columns)}")


def _library_key(spatial: dict, sec: str):
    if sec in spatial:
        return sec
    hits = [k for k in spatial if sec in k or k in sec]
    return hits[0] if len(hits) == 1 else None


def _patient_of(obs: pd.DataFrame) -> str:
    for c in ("donor_id", "patient", "Patient", "donor", "patient_id", "PatientID"):
        if c in obs:
            v = obs[c].astype(str).unique()
            if len(v) == 1:
                return v[0]
    raise ValueError(f"no single patient column among {list(obs.columns)}")


def convert_kiviaho(raw_dir: Path, out: Path) -> list[dict]:
    tar = raw_dir / "GSE278936_RAW.tar"
    clin = pd.read_excel(raw_dir / "kiviaho2024_supp_table_S1_patient_clinical.xlsx", sheet_name="Discovery")
    clin["key"] = clin["Patient ID"].astype(str).str.replace(" ", "_")
    clin = clin.set_index("key")
    rows = []
    with tarfile.open(tar) as t:
        members = {m.name: m for m in t.getmembers()}
        pres = sorted({n.rsplit("_barcodes.tsv.gz", 1)[0] for n in members if n.endswith("_barcodes.tsv.gz")})
        for pre in pres:
            gsm, name = pre.split("_", 1)
            def get(suf):  # decompressed in-memory stream of one archive member
                return io.BytesIO(gzip.decompress(t.extractfile(members[f"{pre}_{suf}"]).read()))

            X, feats, bc = read_mtx_triplet(get("matrix.mtx.gz"), get("features.tsv.gz"), get("barcodes.tsv.gz"))
            pos = read_positions(get("tissue_positions_list.csv.gz"))
            sf = json.loads(get("scalefactors_json.json.gz").read())
            hires = _read_png(get("tissue_hires_image.png.gz"))
            px = pixel_size_from_spot(sf["spot_diameter_fullres"])
            group = name.rsplit("_", 1)[0]
            c = clin.loc[name] if name in clin.index else None
            meta = {"patient": f"Kiviaho2024_{name}" if c is not None else f"Kiviaho2024_{group}",
                    "patient_source": "Discovery sheet 'Patient ID' (one section per patient)" if c is not None
                    else "not in the Discovery sheet; MET sections share one patient key per GEO group (assumption)",
                    "organ": "Prostate", "oncotree_code": "PRAD" if group != "BPH" else "BPH",
                    "group": group, "role": "discovery" if c is not None else "metastasis_not_in_discovery_sheet",
                    "st_technology": "Visium", "chemistry": "Visium fresh-frozen (poly-A capture)",
                    "preservation_method": "Fresh frozen",
                    "dataset_title": "Single cell and spatial transcriptomics highlight the interaction of club-like cells "
                                     "with immunosuppressive myeloid cells in prostate cancer (Kiviaho et al. 2024)",
                    "license": "GEO public (no explicit licence; NCBI GEO data are freely available)",
                    "source_accession": f"GSE278936/{gsm}", "pixel_size_um": px,
                    "pixel_size_um_source": "55 um / spot_diameter_fullres",
                    "hires_um_per_px": px / float(sf["tissue_hires_scalef"]),
                    "image_note": "hires PNG only: expression + spatial only"}
            if c is not None:
                for k in ("Age at prostatectomy", "Gleason grade group from prostatectomy", "serum PSA (ng/ml)",
                          "T-stage", "Treatment before prostatectomy", "Sample ID", "Spots under tissue"):
                    v = c.get(k)
                    meta[k] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
            rows.append(write_section(out, f"Kiviaho2024_{name}", X, feats, bc, pos, sf, meta, hires_image=hires))
    return rows


def convert_geo_gse238264(raw_dir: Path, out: Path) -> list[dict]:
    """Convert the public processed HCC Visium archive, one patient per section."""
    tar_path = raw_dir / "GSE238264_RAW.tar"
    outcome = {
        "GSM7661255": ("HCC1", "responder"),
        "GSM7661256": ("HCC2", "responder"),
        "GSM7661257": ("HCC3", "responder"),
        "GSM7661258": ("HCC4", "responder"),
        "GSM7661259": ("HCC5", "nonresponder"),
        "GSM7661260": ("HCC6", "nonresponder"),
        "GSM7661261": ("HCC7", "nonresponder"),
    }
    rows = []
    with tarfile.open(tar_path) as outer:
        outer_members = {
            Path(member.name).name.split("_", 1)[0]: member
            for member in outer.getmembers() if member.isfile()
        }
        if set(outer_members) != set(outcome):
            raise ValueError("GSE238264 sample archive set does not match the public receipt")
        for gsm, (patient, response) in sorted(outcome.items()):
            source = outer.extractfile(outer_members[gsm])
            if source is None:
                raise ValueError(f"cannot read nested archive for {gsm}")
            with tarfile.open(fileobj=io.BytesIO(source.read()), mode="r:gz") as nested:
                members = {member.name: member for member in nested.getmembers() if member.isfile()}
                root = next(name.split("/", 1)[0] for name in members if name.endswith("filtered_feature_bc_matrix.h5"))

                def payload(suffix: str) -> io.BytesIO:
                    member = members[f"{root}/{suffix}"]
                    handle = nested.extractfile(member)
                    if handle is None:
                        raise ValueError(f"{gsm}: cannot read {suffix}")
                    return io.BytesIO(handle.read())

                X, feats, bc = read_10x_h5(payload("filtered_feature_bc_matrix.h5"))
                pos = read_positions(payload("spatial/tissue_positions_list.csv"))
                sf = json.loads(payload("spatial/scalefactors_json.json").read())
                hires = _read_png(payload("spatial/tissue_hires_image.png"))
            px = pixel_size_from_spot(sf["spot_diameter_fullres"])
            hires_px = px / float(sf["tissue_hires_scalef"])
            meta = {
                "patient": patient,
                "response": response,
                "patient_source": "GEO sample identity; one Visium section per patient",
                "organ": "Liver",
                "oncotree_code": "LIHC",
                "st_technology": "Visium",
                "chemistry": "Visium fresh-frozen (poly-A capture)",
                "preservation_method": "Fresh frozen",
                "dataset_title": "HCC cabozantinib+nivolumab public processed Visium release",
                "license": "GEO public (processed archive)",
                "source_accession": f"GSE238264/{gsm}",
                "image_note": "released SpaceRanger tissue_hires_image.png; patches use the recorded hires pixel scale",
            }
            rows.append(write_section(
                out, f"GSE238264_{patient}", X, feats, bc, pos, sf, meta,
                fullres_image=hires, fullres_pixel_size_um=hires_px, hires_image=hires,
            ))
    return rows


def convert_geo_gse294385(raw_dir: Path, out: Path) -> list[dict]:
    """Convert the public CRC micrometastasis Visium archive.

    Patient identities are supplied only by the separately hashed source map.
    Source pathology labels are copied as annotations, never treated as
    independent outcomes.  The adapter does not attach micro/macro/site labels
    to metadata: those belong to a later, preregistered phenotype contract.
    """
    import anndata as ad

    tar_path = raw_dir / "GSE294385_RAW.tar"
    patient_map_path = raw_dir / "patient_map.json"
    pathology_path = raw_dir / "pathology_availability.json"
    mapping = json.loads(patient_map_path.read_text())
    patient_by_section = {row["section_id"]: row["patient_id"] for row in mapping["rows"]}
    pathology = json.loads(pathology_path.read_text())
    if pathology["n_sections"] != len(patient_by_section):
        raise ValueError("GSE294385 pathology/patient map section count mismatch")
    labels_by_section: dict[str, pd.DataFrame] = {}
    # The availability receipt is intentionally summary-only.  Read the source
    # spot table here to attach its three stated annotation layers by barcode.
    spot_table = raw_dir / "visium_metadata_raw.csv"
    spot = pd.read_csv(spot_table, usecols=["Barcode", "Layer1", "Layer2", "Layer3"])
    for section, sub in spot.groupby(spot["Barcode"].str.split("~").str[0], sort=False):
        labels_by_section[str(section)] = sub.set_index(sub["Barcode"].str.split("~").str[1])[["Layer1", "Layer2", "Layer3"]]
    rows = []
    with tarfile.open(tar_path) as outer:
        nested_members = {Path(member.name).name.split("_", 1)[1].removesuffix(".tar.gz"): member
                          for member in outer.getmembers() if member.isfile() and member.name.endswith(".tar.gz")}
        if set(nested_members) != set(patient_by_section):
            raise ValueError("GSE294385 nested sections do not match patient map")
        for section in sorted(nested_members):
            handle = outer.extractfile(nested_members[section])
            if handle is None:
                raise ValueError(f"cannot read nested section {section}")
            with tarfile.open(fileobj=io.BytesIO(handle.read()), mode="r:gz") as nested:
                members = {item.name: item for item in nested.getmembers() if item.isfile()}
                root = section

                def payload(suffix: str) -> io.BytesIO:
                    item = members[f"{root}/{suffix}"]
                    source = nested.extractfile(item)
                    if source is None:
                        raise ValueError(f"{section}: cannot read {suffix}")
                    return io.BytesIO(source.read())

                def gz_payload(suffix: str) -> io.BytesIO:
                    return io.BytesIO(gzip.decompress(payload(suffix).read()))

                X, feats, bc = read_mtx_triplet(gz_payload("filtered_feature_bc_matrix/matrix.mtx.gz"),
                                                gz_payload("filtered_feature_bc_matrix/features.tsv.gz"),
                                                gz_payload("filtered_feature_bc_matrix/barcodes.tsv.gz"))
                pos = read_positions(payload("spatial/tissue_positions.csv"))
                sf = json.loads(payload("spatial/scalefactors_json.json").read())
                hires = _read_png(payload("spatial/tissue_hires_image.png"))
            px = pixel_size_from_spot(sf["spot_diameter_fullres"])
            meta = {
                "patient": patient_by_section[section],
                "patient_source": "source-authored Figure2A.R sample_groups; separately hashed patient_map.json",
                "organ": "Colorectal primary/metastasis mixed; derive site only from explicit source pathology contract",
                "oncotree_code": "COADREAD",
                "st_technology": "Visium",
                "chemistry": "Visium fresh-frozen (poly-A capture)",
                "preservation_method": "Fresh frozen",
                "dataset_title": "Spatial evolutionary landscape of colorectal cancer micrometastases",
                "license": "GEO public processed archive",
                "source_accession": f"GSE294385/{section}",
                "pixel_size_um": px, "pixel_size_um_source": "55 um / spot_diameter_fullres",
                "image_note": "released SpaceRanger tissue_hires_image.png; no full-resolution H&E provenance asserted",
            }
            sid = f"GSE294385_{section}"
            row = write_section(out, sid, X, feats, bc, pos, sf, meta, hires_image=hires)
            labels = labels_by_section.get(section)
            if labels is None:
                raise ValueError(f"no pathology labels for {section}")
            st_path = out / "st" / f"{sid}.h5ad"
            a = ad.read_h5ad(st_path)
            aligned = labels.reindex(a.obs_names)
            # A few retained count-matrix barcodes have no source pathology
            # label.  Keep those spots and make their status explicit instead
            # of silently dropping them or treating them as a pathology class.
            missing = aligned["Layer1"].isna()
            for column in ("Layer1", "Layer2", "Layer3"):
                a.obs[f"source_pathology_{column.lower()}"] = aligned[column].fillna("NOT_ANNOTATED").astype(str).to_numpy()
            a.uns["source_pathology_annotation"] = {
                "source": "authors' visium_meta_raw.rds",
                "n_count_matrix_barcodes": int(a.n_obs),
                "n_unannotated_count_matrix_barcodes": int(missing.sum()),
                "rule": "unannotated barcodes are retained with NOT_ANNOTATED and excluded by phenotype-specific contracts",
            }
            a.write_h5ad(st_path)
            row["st_sha256"] = sha256(st_path)
            rows.append(row)
    return rows


SOURCES = {
    "mendeley_li2022": (convert_mendeley_li2022, "CC BY 4.0"),
    "geo": (convert_geo_gse250163, "GEO public"),
    "zenodo": (convert_zenodo_14620362, "CC BY 4.0"),
    "kiviaho": (convert_kiviaho, "GEO public"),
    "gse238264": (convert_geo_gse238264, "GEO public processed archive"),
    "gse294385": (convert_geo_gse294385, "GEO public processed archive"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--raw", required=True, help="directory holding the downloaded source files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fn, lic = SOURCES[args.source]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = fn(Path(args.raw), out)
    raw_files = {p.name: sha256(p) for p in sorted(Path(args.raw).glob("*")) if p.is_file()}
    write_manifest(out, args.source, rows, {"license": lic, "raw_dir": args.raw, "raw_files_sha256": raw_files})
    print(json.dumps({"source": args.source, "n_sections": len(rows),
                      "n_patients": len({r["patient"] for r in rows}),
                      "n_spots": sum(r["n_spots"] for r in rows),
                      "with_morphology": sum(r["with_morphology"] for r in rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
