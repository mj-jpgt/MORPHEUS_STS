"""Biology external cohort loaders (LOCKIN Sec. 11). Every loader returns
`(features, sample_ids, endpoint, treatment_arm)`; `treatment_arm` is `None`
for a non-randomised (prognostic-only) cohort.

Two kinds of source live under here:
  - what was already on the persistent store (CPTAC ccRCC, JAVELIN derived
    contrasts) -- checked directly, not assumed.
  - what this delivery ACQUIRED itself from public, no-auth routes (TransNEO,
    Ovarian Bevacizumab clinical tables, PEACE processed data, IMPRESS
    clinical/IHC metadata) under `EXTERNAL_ACQUISITION_DIR`, each with a
    MANIFEST.json (route, fetch date, sha256 per file) alongside it. Every
    one of those five files was byte-identical (sha256 match) to an earlier
    2026-09-06 fetch recorded in `CLINICAL_MANIFESTS_DIR`, whose bytes could
    not be relocated on this box -- so this delivery re-fetched from the
    same verified-live routes rather than re-requesting anything gated.
    Controlled-access portions (raw sequencing under EGA DACs, CMB's dbGaP
    molecular arm) were NOT attempted -- no account created, no access
    requested -- and are recorded as owner-action in each MANIFEST.json.

See `AVAILABILITY` for what is present / controlled-access / absent, and
`dbench.cohorts.availability_table()` for the one-call summary.
"""
from __future__ import annotations

import glob
import json
import os

import h5py
import numpy as np
import pandas as pd

CPTAC_CCRCC_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                   "hf_tcga_cptac_cgga/cptac/tables/ccrcc")
CPTAC_MAPPING_JSON = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                     "hf_tcga_cptac_cgga/cptac/cptac_ccrcc_patient_mapping.json")
CPTAC_HE_DESCRIPTORS_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/"
                           "external/cptac_ccrcc_he_descriptors_20260925")
TCGA_KIRC_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                "tcga_kirc_20260925")
CPTAC_BRCA_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                  "hf_tcga_cptac_cgga/cptac/tables/brca")
CPTAC_BRCA_MAPPING_JSON = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                          "hf_tcga_cptac_cgga/cptac/cptac_brca_patient_mapping.json")
CPTAC_BRCA_HE_DESCRIPTORS_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/"
                                "external/cptac_brca_he_descriptors_20260926")
TCGA_BRCA_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                "tcga_brca_20260926")
PANNUKE_CLASSES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
JAVELIN_DIR = "/home/ubuntu/cscd/w5_data"
CLINICAL_MANIFESTS_DIR = (
    "/lambda/nfs/geeg/biorag3_persistent_20260711/morpheus_work_20260907/"
    "coordinator_snapshot_20260911/morpheus/clinical_manifests")
EXTERNAL_ACQUISITION_DIR = (
    "/lambda/nfs/geeg/biorag3_persistent_20260711/"
    "external_acquisition_20260911")
TRANSNEO_DIR = os.path.join(EXTERNAL_ACQUISITION_DIR, "transneo")
OVARIAN_BEV_DIR = os.path.join(EXTERNAL_ACQUISITION_DIR, "ovarian_bevacizumab")
PEACE_DIR = os.path.join(EXTERNAL_ACQUISITION_DIR, "peace")
IMPRESS_DIR = os.path.join(EXTERNAL_ACQUISITION_DIR, "impress")
CMB_DIR = os.path.join(EXTERNAL_ACQUISITION_DIR, "cancer_moonshot_biobank")

AVAILABILITY = {
    "cptac_ccrcc": {
        "present": True, "controlled_access": False,
        "supports": ["prognostic association", "H&E morphology (110 WSIs)"],
        "note": ("Real: proteomics__umich.parquet, transcriptomics__"
                "{bcm,washu,broad}.parquet, CNV__washu.parquet, "
                "cibersort__washu.parquet, xcell__washu.parquet, "
                "somatic_mutation__harmonized.parquet, clinical__mssm.csv. "
                "Not randomised -- prognostic/molecular-subtype association "
                "only, no treatment arm. `load_cptac_ccrcc(modality=...)` "
                "also serves \"rna\" (transcriptomics__washu.parquet) and "
                "\"he\" (H-optimus PCs + CellViT nuclei summary, computed "
                "2026-09-25 from the 110 selected WSIs -- see "
                "`S/external/cptac_ccrcc_he_descriptors_20260925/"
                "MANIFEST.json`; CellViT columns may be NaN for slides "
                "still in the resumable background batch)."),
    },
    "tcga_kirc": {
        "present": True, "controlled_access": False,
        "supports": ["prognostic association (second cohort)"],
        "note": ("ACQUIRED 2026-09-25 from the public cBioPortal REST API "
                "(no account, no auth) -- study "
                "kirc_tcga_pan_can_atlas_2018: 510 patients x 18,484 "
                "protein-coding genes RNA-seq v2 (RSEM), 512 patients "
                "clinical (OS_STATUS/OS_MONTHS/DFS_STATUS/DFS_MONTHS). Not "
                "randomised -- prognostic only. See "
                "`dbench/scripts/tcga_kirc_acquisition.py` and "
                "`S/external/tcga_kirc_20260925/MANIFEST.json`. "
                "`load_tcga_kirc()`. H&E: 513 open-access GDC DX1 slides "
                "(2026-10-13, `dbench/scripts/tcga_kirc_wsi.py`) -> H-Optimus-0 "
                "(TF32) + CellViT descriptors in "
                "`S/external/tcga_kirc_he_descriptors_20261013/`; "
                "`load_tcga_kirc_patient_table()` (no grade)."),
    },
    "cptac_brca": {
        "present": "partial", "controlled_access": False,
        "supports": ["prognostic association (RNA/protein transfer)"],
        "note": ("Real, already on-store: proteomics__umich.parquet (125 "
                "patients), transcriptomics__washu.parquet (121 patients), "
                "somatic_mutation__harmonized.parquet, tumor_purity__"
                "washu.parquet, clinical__mssm.csv (134 patients, same "
                "column names as CPTAC ccRCC's clinical table -- "
                "'Survival status (1, dead; 0, alive)' / 'Overall "
                "survival, days'). `load_cptac_brca(modality=...)` serves "
                "\"protein\" (default), \"rna\" and endpoints "
                "\"tumor_purity\", \"pik3ca_status\" (BRCA's own most "
                "common driver mutation, the BAP1-status analogue), "
                "\"os\"/\"os_days\". modality=\"he\" is NOT present: this "
                "cohort (the original 2016 CPTAC BRCA/OV/CO pilot, "
                "patient IDs like '01BR001') has NO public whole-slide-"
                "image route verified live 2026-09-26 -- 'CPTAC-BRCA' is "
                "not a collection in the TCIA/NBIA collection list (only "
                "the CPTAC3 cohorts CCRCC/LUAD/LSCC/UCEC/STAD/PDA/GBM/CM/"
                "HNSCC/SAR/AML have a PathDB pathology-portal crosswalk, "
                "confirmed by downloading and filtering the exact source "
                "CSV the ccRCC crosswalk used -- zero BRCA rows), and the "
                "GDC project CPTAC-2 (\"CPTAC-Breast, Colon, Ovary\") "
                "carries only WXS/RNA-seq/miRNA-seq data, zero Slide Image "
                "files (checked via the GDC files-facet API). No on-store "
                "table carries a CPTAC-BRCA<->TCGA-BRCA barcode crosswalk "
                "either (somatic_mutation__harmonized.parquet's "
                "Tumor_Sample_Barcode is '01BR001_T'-style, not a TCGA "
                "barcode). `cptac_brca_wsi`/`cptac_brca_he_descriptors` "
                "therefore stay ungated -- this is a verified public-data "
                "absence, not a licence/account gate, so it is reported "
                "'partial' rather than forced with a different route. "
                "UPDATE 2026-10-15 (Agent W): the CPTAC-BRCA H&E slides ARE "
                "public on the NCI Imaging Data Commons as DICOM slide "
                "microscopy (collection cptac_brca, CC BY 4.0, anonymous "
                "public S3; 119 of the proteomics+RNA patients); acquired by "
                "`dbench/scripts/wsi_slide_descriptors.py --cohort cptac_brca` "
                "and served by `load_cptac('brca')`."),
    },
    "tcga_brca": {
        "present": True, "controlled_access": False,
        "supports": ["prognostic association (second breast cohort)"],
        "note": ("ACQUIRED 2026-09-26 from the public cBioPortal REST API "
                "(no account, no auth), same route/script as TCGA-KIRC -- "
                "study brca_tcga_pan_can_atlas_2018. See "
                "`dbench/scripts/tcga_kirc_acquisition.py --study "
                "brca_tcga_pan_can_atlas_2018` and "
                "`S/external/tcga_brca_20260926/MANIFEST.json`. Not "
                "randomised -- prognostic only. `load_tcga_brca()`."),
    },
    "javelin_renal_101": {
        "present": "partial", "controlled_access": False,
        "supports": ["treatment interaction (derived contrasts only)"],
        "note": ("~/cscd/w5_data/javelin_{gene,enrichment}_delta_ab*.csv are "
                "REAL but are already-computed avelumab+axitinib-vs-"
                "sunitinib gene/pathway CONTRAST tables from a prior "
                "analysis, not the raw per-patient RNA-seq + response + arm "
                "table `dbench.cohorts` needs to build (features, "
                "sample_ids, endpoint, arm) from scratch. No raw supplement "
                "found on this store despite being named in the earlier "
                "acquisition notes -- reported as partial, not present, "
                "rather than building a loader around a derived table that "
                "already bakes in someone else's contrast choice."),
    },
    "transneo": {
        "present": True, "controlled_access": "partial",
        "supports": ["treatment-response (pCR, RCB score/category)"],
        "note": ("ACQUIRED 2026-09-11 from the public GitHub raw + Nature "
                "CDN routes (Sammut et al. 2022 Nature) -- 162 RNA-seq "
                "samples x genes + 168-patient clinical/response table "
                "(Donor.ID, RCB.score, RCB.category, pCR.RD). Bytes are "
                "sha256-identical to the 2026-09-06 prior fetch recorded in "
                "clinical_manifests/transneo/MANIFEST.json. Raw DNA/RNA-seq "
                "reads under EGAS00001004582 need an EGA DAC application -- "
                "NOT attempted, no account created. See "
                "`load_transneo()`."),
    },
    "impress": {
        "present": True, "controlled_access": False,
        "supports": ["treatment-response (pCR, RCB category)"],
        "note": ("ACQUIRED 2026-09-11 from the public Google Drive folder "
                "(gdown, no auth) -- cohort_meta.xlsx, 126 patients (62 "
                "HER2+, 64 TNBC), clinical + registered-IHC quantitative "
                "features (CD8/CD163/PD-L1 %, ER/PR %, age) and pCR/RCB "
                "labels. Bytes sha256-identical to the 2026-09-06 prior "
                "fetch. Full 252-SVS whole-slide image corpus (25-35GB) "
                "NOT fetched -- not needed for the tabular loader below and "
                "deferred to a future delivery. See `load_impress()`."),
    },
    "ovarian_bevacizumab": {
        "present": True, "controlled_access": False,
        "supports": ["treatment-response (CA-125, slide-level)"],
        "note": ("ACQUIRED 2026-09-11 from TCIA's direct HTTPS no-auth "
                "route (CC BY 4.0) -- CA125-data_20230207.xlsx (286 "
                "slide-level treatment-effect records, 78 unique patients) "
                "+ Final-patient_list.xlsx (clinical fields). Bytes "
                "sha256-identical to the 2026-09-06 prior fetch recorded in "
                "clinical_manifests/ovarian_bevacizumab/MANIFEST.json. "
                "253.8GB of whole-slide images are TCIA-native/Aspera-only "
                "(IBM Aspera Connect, a proprietary EULA-gated client) -- "
                "NOT fetched, owner-action if the images are needed; this "
                "is a client-software gate, not a data-licence gate. See "
                "`load_ovarian_bevacizumab()`."),
    },
    "peace": {
        "present": True, "controlled_access": True,
        "supports": ["molecular association (AR copy number, tumour "
                    "content) -- no treatment-response endpoint exists in "
                    "this cohort"],
        "note": ("ACQUIRED 2026-09-11 from Zenodo (fully open, DOI "
                "10.5281/zenodo.8125338) -- 45 RNA-seq samples (4 patients, "
                "PEA172 + 3 CASCADE) x 38,066 genes, plus AR_copy_number_ACE "
                "and tumour-content (TC) per sample. Zip sha256-identical to "
                "the 2026-09-06 prior fetch. No treatment-response label "
                "exists anywhere in this substrate (confirmed directly, 15 "
                "sheets inspected) -- it characterises post-treatment clonal "
                "architecture, not a response outcome. Raw WGS/RNA-seq under "
                "EGAS00001006598 needs an EGA DAC (EGAC00001002865) -- NOT "
                "attempted. See `load_peace()`."),
    },
    "cancer_moonshot_biobank": {
        "present": "partial", "controlled_access": "partial",
        "supports": ["imaging catalogue only -- no treatment-response "
                    "endpoint in the public route"],
        "note": ("Public pathology route (idc-index 0.12.5, no auth) "
                "RE-VERIFIED LIVE 2026-09-11 against the live IDC index: "
                "1,032,911 total rows, 1,336 cmb_* SM-modality (pathology) "
                "rows across 9 cancer types, all CC BY 4.0 -- identical "
                "counts to the 2026-09-06 prior fetch. NOT re-downloaded as "
                "bytes (no endpoint exists in the public imaging metadata to "
                "build a (features,ids,endpoint,arm) loader against -- see "
                "`load_cancer_moonshot_catalogue()` for the honest "
                "catalogue-only shape instead). CTDC's open clinical subset "
                "exists but its field-level content is unverified (JS SPA); "
                "the full molecular/genomic dataset is controlled-access via "
                "dbGaP -- NOT attempted, no account created."),
    },
}


def availability_table():
    rows = []
    for name, info in AVAILABILITY.items():
        rows.append({"cohort": name, **info})
    return rows


def print_availability_table():
    for r in availability_table():
        print("%-24s present=%-30s controlled_access=%-6s supports=%s"
             % (r["cohort"], str(r["present"]), str(r["controlled_access"]),
                r["supports"]))


def _cptac_endpoint(endpoint, sample_ids):
    """The real outcome/association column, reindexed to `sample_ids` (so
    it works regardless of which modality's own sample ordering/coverage
    called it). "tumor_purity" (tumor_purity__washu.parquet, default),
    "bap1_status" (somatic_mutation__harmonized.parquet), "os" (real binary
    vital status, clinical__mssm.csv) or "os_days" (real continuous overall
    survival in days, same table) -- "os"/"os_days" are the predeclared D4
    prognosis analysis's real endpoint, not a derived risk score.
    """
    if endpoint == "tumor_purity":
        pur = pd.read_parquet(os.path.join(CPTAC_CCRCC_DIR,
                                           "tumor_purity__washu.parquet"))
        return pur["TumorPurity"].reindex(sample_ids).to_numpy(dtype=float)
    if endpoint == "bap1_status":
        mut = pd.read_parquet(os.path.join(CPTAC_CCRCC_DIR,
                                           "somatic_mutation__harmonized.parquet"))
        # Patient_ID is the real index here too; "Gene" is the exact real
        # column name (not a substring match -- this table also carries
        # ~120 HGNC/GO/ClinVar annotation columns whose names happen to
        # contain "gene", which a substring search would wrongly hit).
        bap1_samples = set(mut.index[mut["Gene"].astype(str).str.upper()
                                     == "BAP1"].astype(str))
        return np.array([1.0 if s in bap1_samples else 0.0 for s in sample_ids])
    if endpoint in ("os", "os_days"):
        clin = pd.read_csv(os.path.join(CPTAC_CCRCC_DIR, "clinical__mssm.csv"))
        clin = clin.set_index("Patient_ID")
        if endpoint == "os":
            col = clin["Survival status (1, dead; 0, alive)"]
        else:
            col = clin["Overall survival, days"]
        return pd.to_numeric(col.reindex(sample_ids), errors="coerce").to_numpy(dtype=float)
    raise ValueError("unknown endpoint %r" % endpoint)


def _cptac_brca_endpoint(endpoint, sample_ids):
    """The BRCA-cohort analogue of `_cptac_endpoint`, kept as its own
    function (not a parameterised shared helper) so the ccRCC loader/tests
    above are untouched. "tumor_purity" (tumor_purity__washu.parquet,
    default), "pik3ca_status" (somatic_mutation__harmonized.parquet --
    PIK3CA is BRCA's own most frequently mutated driver gene, 48/134
    patients here, checked directly -- the BAP1-status analogue, not a
    borrowed kidney gene), "os" (real binary vital status, clinical__
    mssm.csv) or "os_days" (real continuous overall survival in days, same
    table).
    """
    if endpoint == "tumor_purity":
        pur = pd.read_parquet(os.path.join(CPTAC_BRCA_DIR,
                                           "tumor_purity__washu.parquet"))
        return pur["TumorPurity"].reindex(sample_ids).to_numpy(dtype=float)
    if endpoint == "pik3ca_status":
        mut = pd.read_parquet(os.path.join(CPTAC_BRCA_DIR,
                                           "somatic_mutation__harmonized.parquet"))
        pik3ca_samples = set(mut.index[mut["Gene"].astype(str).str.upper()
                                       == "PIK3CA"].astype(str))
        return np.array([1.0 if s in pik3ca_samples else 0.0 for s in sample_ids])
    if endpoint in ("os", "os_days"):
        clin = pd.read_csv(os.path.join(CPTAC_BRCA_DIR, "clinical__mssm.csv"))
        clin = clin.set_index("Patient_ID")
        if endpoint == "os":
            col = clin["Survival status (1, dead; 0, alive)"]
        else:
            col = clin["Overall survival, days"]
        return pd.to_numeric(col.reindex(sample_ids), errors="coerce").to_numpy(dtype=float)
    raise ValueError("unknown endpoint %r" % endpoint)


def load_cptac_brca(endpoint="tumor_purity", modality="protein"):
    """(features, sample_ids, y, treatment_arm=None) for the real CPTAC
    BRCA cohort (the original 2016 CPTAC BRCA/OV/CO pilot, patient IDs like
    '01BR001'). `modality` selects the real substrate: "protein"
    (proteomics__umich.parquet, default, 125 patients) or "rna"
    (transcriptomics__washu.parquet, 121 patients, chosen for the same
    washu-consistency reason as `load_cptac_ccrcc`). modality="he" is NOT
    supported -- see `AVAILABILITY["cptac_brca"]` for the verified-absent
    public WSI route (no CPTAC-BRCA collection on TCIA/NBIA, no Slide Image
    files under GDC's CPTAC-2 project, no on-store CPTAC<->TCGA barcode
    crosswalk); it raises ValueError like any other unsupported modality
    rather than silently falling back. `endpoint` selects "tumor_purity"
    (default), "pik3ca_status", "os" (binary vital status) or "os_days"
    (continuous overall survival).
    """
    if not os.path.isdir(CPTAC_BRCA_DIR):
        raise FileNotFoundError(CPTAC_BRCA_DIR)

    if modality == "protein":
        table = pd.read_parquet(os.path.join(CPTAC_BRCA_DIR,
                                             "proteomics__umich.parquet"))
        sample_ids = table.index.astype(str).tolist()
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
        features = table.to_numpy(dtype=float)
    elif modality == "rna":
        table = pd.read_parquet(os.path.join(CPTAC_BRCA_DIR,
                                             "transcriptomics__washu.parquet"))
        sample_ids = table.index.astype(str).tolist()
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
        features = table.to_numpy(dtype=float)
    else:
        raise ValueError("unknown modality %r" % modality)

    y = _cptac_brca_endpoint(endpoint, sample_ids)
    return features, sample_ids, y, None


def load_tcga_brca(endpoint="os"):
    """(features, sample_ids, y, treatment_arm=None) for the real TCGA-BRCA
    cohort (cBioPortal study `brca_tcga_pan_can_atlas_2018`), ACQUIRED
    2026-09-26 from the public cBioPortal REST API (no account, no auth --
    see `dbench/scripts/tcga_kirc_acquisition.py --study
    brca_tcga_pan_can_atlas_2018` and
    `S/external/tcga_brca_20260926/MANIFEST.json`). This is the second
    public breast-cancer survival cohort the hest-breast-idc-v01 protocol's
    `actionable` advance rule calls for. Features are the real RNA-seq v2
    (RSEM) gene-level expression matrix, gene-symbol columns. `endpoint`
    selects the real survival column from clinical.parquet: "os" (binary
    vital status, 1.0=DECEASED/0.0=LIVING, from the real "N:LABEL"-coded
    OS_STATUS field) or "os_months" (continuous OS_MONTHS). Not randomised
    -- no treatment arm.
    """
    rna_path = os.path.join(TCGA_BRCA_DIR, "rna_seq_v2_mrna.parquet")
    clin_path = os.path.join(TCGA_BRCA_DIR, "clinical.parquet")
    if not os.path.exists(rna_path) or not os.path.exists(clin_path):
        raise FileNotFoundError(TCGA_BRCA_DIR)

    rna = pd.read_parquet(rna_path).set_index("patient_id")
    sample_ids = rna.index.astype(str).tolist()
    features = rna.to_numpy(dtype=float)

    clin = pd.read_parquet(clin_path).set_index("patient_id")
    if endpoint == "os":
        def _status_to_float(v):
            if not isinstance(v, str):
                return np.nan
            if v.startswith("1"):
                return 1.0
            if v.startswith("0"):
                return 0.0
            return np.nan
        y = clin["OS_STATUS"].reindex(sample_ids).map(_status_to_float).to_numpy(dtype=float)
    elif endpoint == "os_months":
        y = pd.to_numeric(clin["OS_MONTHS"].reindex(sample_ids), errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("unknown endpoint %r" % endpoint)

    return features, sample_ids, y, None


def _load_cptac_he_features():
    """(features, sample_ids) for modality="he": H-optimus PCs 1-5 (D2) for
    every one of the 110 selected slides (`embeddings.h5`, computed
    2026-09-25 -- see `dbench/scripts/cptac_hoptimus_descriptors.py`), left-
    joined with whatever CellViT nuclei-summary rows already exist
    (`cellvit_summary.parquet`, filled in incrementally by a resumable
    background batch -- see `dbench/scripts/cptac_cellvit_descriptors.py`).
    A patient not yet processed by CellViT gets real H-optimus PCs and real
    NaN (never a fabricated 0) for the CellViT columns -- callers must
    handle that explicitly, exactly as the batch job's own progress ledger
    does.
    """
    h5_path = os.path.join(CPTAC_HE_DESCRIPTORS_DIR, "embeddings.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        patient_ids = [p.decode() if isinstance(p, bytes) else p
                      for p in f["patient_ids"][:]]
        pcs = np.asarray(f["hoptimus_pcs"][:], dtype=float)

    df = pd.DataFrame(pcs, columns=[f"hoptimus_pc{i + 1}" for i in range(pcs.shape[1])])
    df.insert(0, "patient_id", patient_ids)

    cellvit_cols = (["count_density_per_mm2", "mean_nuclear_area_um2"]
                    + [f"class_fraction_{c}" for c in PANNUKE_CLASSES])
    cellvit_path = os.path.join(CPTAC_HE_DESCRIPTORS_DIR, "cellvit_summary.parquet")
    if os.path.exists(cellvit_path):
        cv = pd.read_parquet(cellvit_path)
        keep = ["patient_id"] + [c for c in cellvit_cols if c in cv.columns]
        df = df.merge(cv[keep], on="patient_id", how="left")
    for c in cellvit_cols:
        if c not in df.columns:
            df[c] = np.nan

    sample_ids = df["patient_id"].tolist()
    features = df.drop(columns=["patient_id"]).to_numpy(dtype=float)
    return features, sample_ids


def load_cptac_ccrcc(endpoint="tumor_purity", modality="protein"):
    """(features, sample_ids, y, treatment_arm=None) for the real CPTAC
    ccRCC cohort. `modality` selects which real substrate the features come
    from: "protein" (proteomics__umich.parquet, default), "rna"
    (transcriptomics__washu.parquet -- chosen over broad's transcript-level
    table and bcm for consistency with the other washu-sourced covariates
    this loader already uses; see B0_availability.json for the full
    justification), or "he" (H-optimus PCs + CellViT nuclei summary over
    the 110 selected WSIs -- a slide-level SCALE EXTENSION of the spot-level
    HEST morphology target, never the same measurement). `endpoint`
    selects which real outcome/association column to return -- "tumor_purity"
    (default), "bap1_status", "os" (binary vital status) or "os_days"
    (continuous overall survival).
    """
    if not os.path.isdir(CPTAC_CCRCC_DIR):
        raise FileNotFoundError(CPTAC_CCRCC_DIR)

    if modality == "protein":
        table = pd.read_parquet(os.path.join(CPTAC_CCRCC_DIR,
                                             "proteomics__umich.parquet"))
        # Patient_ID is the real index (not a plain column); the column axis
        # is a real (gene_name, ENSP_id) MultiIndex -- gene name (level 0) is
        # kept, the Ensembl protein id (level 1) dropped, matching how every
        # other loader in this module names features by gene symbol.
        sample_ids = table.index.astype(str).tolist()
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
        features = table.to_numpy(dtype=float)
    elif modality == "rna":
        table = pd.read_parquet(os.path.join(CPTAC_CCRCC_DIR,
                                             "transcriptomics__washu.parquet"))
        sample_ids = table.index.astype(str).tolist()
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(0)
        features = table.to_numpy(dtype=float)
    elif modality == "he":
        features, sample_ids = _load_cptac_he_features()
    else:
        raise ValueError("unknown modality %r" % modality)

    y = _cptac_endpoint(endpoint, sample_ids)
    return features, sample_ids, y, None


def load_tcga_kirc(endpoint="os"):
    """(features, sample_ids, y, treatment_arm=None) for the real TCGA-KIRC
    cohort (cBioPortal study `kirc_tcga_pan_can_atlas_2018`), ACQUIRED
    2026-09-25 from the public cBioPortal REST API (no account, no auth --
    see `dbench/scripts/tcga_kirc_acquisition.py` and
    `S/external/tcga_kirc_20260925/MANIFEST.json`). This is the second
    prognostic cohort D4 calls for, independent of CPTAC. Features are the
    real RNA-seq v2 (RSEM) gene-level expression matrix (510 patients x
    18,484 protein-coding genes, gene-symbol columns). `endpoint` selects
    the real survival column from clinical.parquet: "os" (binary vital
    status, 1.0=DECEASED/0.0=LIVING, from the real "N:LABEL"-coded
    OS_STATUS field) or "os_months" (continuous OS_MONTHS). Not randomised
    -- no treatment arm.
    """
    rna_path = os.path.join(TCGA_KIRC_DIR, "rna_seq_v2_mrna.parquet")
    clin_path = os.path.join(TCGA_KIRC_DIR, "clinical.parquet")
    if not os.path.exists(rna_path) or not os.path.exists(clin_path):
        raise FileNotFoundError(TCGA_KIRC_DIR)

    rna = pd.read_parquet(rna_path).set_index("patient_id")
    sample_ids = rna.index.astype(str).tolist()
    features = rna.to_numpy(dtype=float)

    clin = pd.read_parquet(clin_path).set_index("patient_id")
    if endpoint == "os":
        def _status_to_float(v):
            # cBioPortal encodes vital status as "0:LIVING"/"1:DECEASED" --
            # the leading digit IS the real binary event, not a derived call.
            if not isinstance(v, str):
                return np.nan
            if v.startswith("1"):
                return 1.0
            if v.startswith("0"):
                return 0.0
            return np.nan
        y = clin["OS_STATUS"].reindex(sample_ids).map(_status_to_float).to_numpy(dtype=float)
    elif endpoint == "os_months":
        y = pd.to_numeric(clin["OS_MONTHS"].reindex(sample_ids), errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("unknown endpoint %r" % endpoint)

    return features, sample_ids, y, None


# --------------------------------------------------------------------------
# CPTAC ccRCC patient-level table and discovery episode (tissue_to_patient).
# One row per patient (tumours only).  Every fitted transform (slide PCA,
# programme z-statistics, standardisation) is fitted on `fit_patients` only.
# Outcomes are never part of the table: they are served only through
# `load_cptac_survival_cohort`, which `translational.run_predeclared_survival`
# calls after reading the frozen, hashed survival spec.
# --------------------------------------------------------------------------

CPTAC_CELLVIT_STRUCTURE_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/"
                              "external/cptac_ccrcc_cellvit_structure_20261012")
CPTAC_DRIVER_GENES = ("VHL", "PBRM1", "SETD2", "BAP1", "KDM5C")
CIBERSORT_META = ("P-value", "Pearson Correlation", "Absolute score")
XCELL_SUMMARY = ("ImmuneScore", "StromaScore", "MicroenvironmentScore")
_SHA_CACHE: dict = {}


def dedupe_symbols(features, symbols):
    """One column per gene symbol: where a table carries several columns for
    one symbol (e.g. protein isoforms keyed by ENSP id), keep the column with
    the most finite values (ties: first). Without this, a symbol->column dict
    silently keeps the LAST duplicate, which for HINT1 protein is a minor
    isoform measured in 4 tumours instead of the canonical one measured in
    110. Returns (features, symbols, {symbol: n_duplicates_dropped})."""
    features = np.asarray(features, dtype=float)
    best: dict = {}
    finite = np.isfinite(features).sum(axis=0)
    dropped: dict = {}
    for j, s in enumerate(symbols):
        if s not in best:
            best[s] = j
        else:
            dropped[s] = dropped.get(s, 0) + 1
            if finite[j] > finite[best[s]]:
                best[s] = j
    keep = sorted(best.values())
    return features[:, keep], [symbols[j] for j in keep], dropped


def _file_sha256(path):
    import hashlib
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def cptac_patient_ids(mapping_json=None):
    """The slide-mapped CPTAC ccRCC patients (110; one slide each)."""
    with open(mapping_json or CPTAC_MAPPING_JSON) as fh:
        return sorted({r["patient_id"] for r in json.load(fh)["records"]})


def cptac_patient_folds(patients, strata, n_folds=5, seed=20261012):
    """Deterministic stage-stratified patient folds: shuffle with `seed`, then
    deal each stratum round-robin (continuing the dealer across strata) so
    every fold gets a similar stage mix. Returns a list of held-patient lists."""
    patients = list(patients)
    strata = [str(s) for s in strata]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    folds = [[] for _ in range(n_folds)]
    k = 0
    for s in sorted(set(strata)):
        for i in order:
            if strata[i] == s:
                folds[k % n_folds].append(patients[i])
                k += 1
    return [sorted(f) for f in folds]


def _gene_sets():
    """41 Gavish malignant metaprograms + 50 MSigDB Hallmarks, keyed by the
    protocol's target id stem (`mp<k>`, `hm_<NAME>`), plus the source files."""
    from morpheus.tier2.substrates.hest import _GAVISH_PATH_CANDIDATES, load_gavish_programs
    from morpheus_encoder.measurement_bank import DEFAULT_CACHE_ROOT, load_hallmark_sets
    sets = {f"mp{p['mp_id']}": [str(g) for g in p["genes"]] for p in load_gavish_programs()}
    sets.update({"hm_" + h["name"].replace("HALLMARK_", ""): list(h["genes"]) for h in load_hallmark_sets()})
    paths = [p for p in (*_GAVISH_PATH_CANDIDATES[:1], os.environ.get("MORPHEUS_HALLMARK_GMT"),
                         os.path.join(DEFAULT_CACHE_ROOT, "genesets", "msigdb_hallmark_2024.1.Hs.gmt"))
             if p and os.path.exists(p)]
    return sets, paths


def programme_scores(values, symbols, gene_sets, fit_rows, *, min_genes=3, min_fit_finite=0.5):
    """Mean of per-gene z-scores, z-statistics fitted on `fit_rows` only.
    A gene counts if finite in >= `min_fit_finite` of fit rows; a row's score
    needs >= `min_genes` finite member genes, else NaN (never imputed).
    Returns ({set: score vector}, {set: [genes used]})."""
    values = np.asarray(values, dtype=float)
    index = {s: j for j, s in enumerate(symbols)}
    fit = values[fit_rows]
    mu, sd = np.nanmean(fit, axis=0), np.nanstd(fit, axis=0)
    ok = (np.isfinite(fit).mean(axis=0) >= min_fit_finite) & (sd > 1e-12)
    scores, used = {}, {}
    for name, genes in gene_sets.items():
        cols = [index[g] for g in genes if g in index and ok[index[g]]]
        used[name] = [symbols[j] for j in cols]
        if len(cols) < min_genes:
            scores[name] = np.full(values.shape[0], np.nan)
            continue
        z = (values[:, cols] - mu[cols]) / sd[cols]
        n_ok = np.isfinite(z).sum(axis=1)
        with np.errstate(invalid="ignore"):
            s = np.nanmean(np.where(np.isfinite(z), z, np.nan), axis=1)
        s[n_ok < min_genes] = np.nan
        scores[name] = s
    return scores, used


def _slide_pca(X, fit_rows, k):
    """Standardise on fit rows, PCA fitted on fit rows, project every row."""
    X = np.asarray(X, dtype=float)
    mu, sd = X[fit_rows].mean(0), X[fit_rows].std(0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    _, s, vt = np.linalg.svd(Z[fit_rows], full_matrices=False)
    k = min(k, vt.shape[0])
    evr = (s ** 2 / np.sum(s ** 2))[:k]
    return Z @ vt[:k].T, evr


def _molecular_table(name, tables_dir=None):
    t = pd.read_parquet(os.path.join(tables_dir or CPTAC_CCRCC_DIR, name))
    t.index = t.index.astype(str)
    syms = [str(c) for c in (t.columns.get_level_values(0) if isinstance(t.columns, pd.MultiIndex) else t.columns)]
    return t, syms


# ---- pan-cancer tissue->patient (Agent W, 2026-10-15): every CPTAC type with
# H&E slides + proteomics + RNA, and its TCGA replication project. Slide
# descriptors are written by dbench/scripts/wsi_slide_descriptors.py (one
# directory per cohort: embeddings.h5, cellvit_summary.parquet,
# cellvit_structure.parquet -- the TCGA-KIRC layout).
_STORE_ROOT = "/lambda/nfs/geeg/biorag3_persistent_20260711"
CPTAC_TABLES_ROOT = f"{_STORE_ROOT}/external/hf_tcga_cptac_cgga/cptac/tables"
_RNA_PREFERENCE = ("transcriptomics__washu.parquet", "transcriptomics__bcm.parquet")
# drivers: the five most commonly mutated genes of each type (context family only);
# flags: (clinical column, case-insensitive substring) -> 0/1 histology indicator;
# tcga_flags: the same indicator from TCGA ICD-O-3 histology codes, for replication.
CPTAC_TYPES = {
    "luad": {"drivers": ("EGFR", "KRAS", "STK11", "TP53", "KEAP1"), "flags": {}, "tcga": "luad", "tcga_flags": {}},
    "lscc": {"drivers": ("TP53", "CDKN2A", "PIK3CA", "NFE2L2", "KMT2D"),
             "flags": {"keratinizing": ("histologic_type", "keratinizing squamous")},
             "tcga": "lusc", "tcga_flags": {"keratinizing": ("8071/3",)}},
    "brca": {"drivers": ("PIK3CA", "TP53", "GATA3", "CDH1", "MAP3K1"),
             "flags": {"lobular": ("histologic_type", "lobular")}, "tcga": "brca",
             "tcga_flags": {"lobular": ("8520/3", "8522/3")}},
    "hnscc": {"drivers": ("TP53", "CDKN2A", "PIK3CA", "FAT1", "NOTCH1"),
              "flags": {"larynx": ("tumor_site", "larynx")}, "tcga": "hnsc", "tcga_flags": {}},
    "ucec": {"drivers": ("PTEN", "PIK3CA", "ARID1A", "TP53", "CTNNB1"),
             "flags": {"serous": ("histologic_type", "serous")}, "tcga": "ucec",
             "tcga_flags": {"serous": ("8441/3", "8460/3", "8461/3")}},
    "coad": {"drivers": ("APC", "TP53", "KRAS", "PIK3CA", "BRAF"),
             "flags": {"mucinous": ("histologic_type", "mucinous")}, "tcga": "coad",
             "tcga_flags": {"mucinous": ("8480/3",)}},
    "pdac": {"drivers": ("KRAS", "TP53", "CDKN2A", "SMAD4", "ARID1A"), "flags": {}, "tcga": "paad",
             "tcga_flags": {}},
    "gbm": {"drivers": ("IDH1", "TP53", "PTEN", "EGFR", "NF1"), "flags": {}, "tcga": "gbm", "tcga_flags": {}},
    "ov": {"drivers": ("TP53", "BRCA1", "BRCA2", "NF1", "CDK12"), "flags": {}, "tcga": "ov", "tcga_flags": {}},
}
PANCANCER_DATE = "20261015"


def cptac_type_paths(cancer_type):
    """Store paths of one CPTAC type's tables and slide descriptors."""
    tables = os.path.join(CPTAC_TABLES_ROOT, cancer_type)
    rna = next((f for f in _RNA_PREFERENCE if os.path.exists(os.path.join(tables, f))), _RNA_PREFERENCE[0])
    return {"tables": tables, "rna_file": rna,
            "he_dir": f"{_STORE_ROOT}/external/cptac_{cancer_type}_he_descriptors_{PANCANCER_DATE}",
            "wsi_dir": f"{_STORE_ROOT}/external/cptac_{cancer_type}_wsi_{PANCANCER_DATE}"}


def tcga_project_paths(project):
    """Store paths of one TCGA project's RNA/clinical tables and slide descriptors."""
    project = project.lower().replace("tcga-", "")
    if project == "kirc":
        return {"tables": TCGA_KIRC_DIR, "he_dir": TCGA_KIRC_HE_DIR}
    return {"tables": TCGA_BRCA_DIR if project == "brca" else f"{_STORE_ROOT}/external/tcga_{project}_{PANCANCER_DATE}",
            "he_dir": f"{_STORE_ROOT}/external/tcga_{project}_he_descriptors_{PANCANCER_DATE}"}


def _cptac_ids(path):
    if path.endswith(".parquet"):
        return set(pd.read_parquet(path, columns=[]).index.astype(str))
    return set(pd.read_csv(path, usecols=["Patient_ID"])["Patient_ID"].astype(str))


def cptac_type_patient_ids(cancer_type, *, tables_dir=None, he_dir=None):
    """Tumours of one CPTAC type with proteomics, RNA, clinical AND a slide descriptor."""
    p = cptac_type_paths(cancer_type)
    tables_dir, he_dir = tables_dir or p["tables"], he_dir or p["he_dir"]
    rna = next((f for f in _RNA_PREFERENCE if os.path.exists(os.path.join(tables_dir, f))), _RNA_PREFERENCE[0])
    mol = (_cptac_ids(os.path.join(tables_dir, "proteomics__umich.parquet"))
           & _cptac_ids(os.path.join(tables_dir, rna)) & _cptac_ids(os.path.join(tables_dir, "clinical__mssm.csv")))
    h5 = os.path.join(he_dir, "embeddings.h5")
    if not os.path.exists(h5):
        raise FileNotFoundError(f"no slide descriptors yet for cptac_{cancer_type}: {h5}")
    with h5py.File(h5, "r") as f:
        slide = {x.decode() if isinstance(x, bytes) else str(x) for x in f["patient_ids"][:]}
    return sorted(p_ for p_ in mol & slide if not p_.endswith(".N"))


def _roman_stage(values):
    """'Stage IIIA' / 'STAGE IIB' -> 3 / 2; anything without a stage numeral -> NaN."""
    num = pd.Series(values, dtype=object).astype(str).str.upper().str.extract(r"STAGE\s+(IV|I{1,3})")[0]
    return num.map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}).astype(float)


def _generic_clinical(clin, flags, *, min_known=0.5):
    """Patient-level clinical competitor for a non-kidney CPTAC type: age;
    sex_male when both sexes occur; stage and grade when >= `min_known` of
    patients have them (else the column is left out and listed); histology flags."""
    age = pd.to_numeric(clin["age"].astype(str).str.replace(">=", "", regex=False), errors="coerce")
    out, absent = {"age": age}, []
    sex = (clin["sex"] == "Male").astype(float)
    if sex.nunique() > 1:
        out["sex_male"] = sex
    else:
        absent.append("sex_male (one sex)")
    stage = _roman_stage(clin["tumor_stage_pathological"]).set_axis(clin.index)
    grade = clin["histologic_grade"].astype(str).str.extract(r"G(\d)")[0].astype(float)
    for name, col in (("stage", stage), ("grade", grade)):
        if col.notna().mean() >= min_known:
            out[name] = col
        else:
            absent.append(f"{name} (known for {col.notna().mean():.0%})")
    for name, (column, needle) in flags.items():
        out[name] = clin[column].astype(str).str.lower().str.contains(needle.lower(), regex=False).astype(float)
    return pd.DataFrame(out, index=clin.index), absent


def _rescale_cellvit_to_um(cols, cv_path, patients):
    """CellViT area/density from target_patch_mpp to the slide's level-0 mpp (true um), per slide's wsi_mpp."""
    mpp = (pd.read_parquet(cv_path).drop_duplicates("patient_id", keep="last")
           .set_index("patient_id")["wsi_mpp"].reindex(patients).to_numpy(dtype=float))
    factor = (mpp / CELLVIT_TARGET_PATCH_MPP) ** 2
    cols["cv_mean_nuclear_area_um2"] = cols["cv_mean_nuclear_area_um2"] * factor
    cols["cv_count_density_per_mm2"] = cols["cv_count_density_per_mm2"] / factor


def _slide_technical(wsi_dir, patients):
    """slide_frozen = 1 for an OCT (frozen-section) slide, from the slide selection; [] when all FFPE/unknown."""
    p = os.path.join(wsi_dir, "selection.csv")
    if not os.path.exists(p):
        return None
    sel = pd.read_csv(p, dtype=str)
    if "embedding_medium" not in sel.columns or not (sel["embedding_medium"] == "OCT").any():
        return None
    sel = sel.drop_duplicates("patient_id").set_index("patient_id")
    return pd.DataFrame({"slide_frozen": (sel["embedding_medium"] == "OCT").astype(float)}).reindex(patients)


CELLVIT_SUMMARY_COLS = (["count_density_per_mm2", "mean_nuclear_area_um2"]
                        + [f"class_fraction_{c}" for c in PANNUKE_CLASSES])
_STRUCTURE_META = ("slide_id", "status", "mpp_used", "cells_json_sha256")


def _slide_structure_blocks(add, track, patients, fit_rows, n_pcs, h5_path, cv_path, struct_path):
    """Shared by the CPTAC and TCGA-KIRC patient tables: `hoptimus_slide`
    (hos_pc*: PCA of the 3072-d H-Optimus-0 mean+sd, fitted on fit_rows),
    `cellvit_summary` (cv_*), `cellvit_structure` (cvs_*). Patients without a
    row get NaN. Returns (explained variance ratio, [missing block names],
    H-Optimus inference precision recorded in the h5)."""
    with h5py.File(track(h5_path), "r") as f:
        pids = [p.decode() if isinstance(p, bytes) else str(p) for p in f["patient_ids"][:]]
        ms = pd.DataFrame(np.asarray(f["hoptimus_meanstd"][:], dtype=float), index=pids).reindex(patients)
        precision = str(f.attrs.get("precision", "fp32"))  # CPTAC 09-25 set predates the attr: fp32
    pcs, evr = _slide_pca(ms.to_numpy(), fit_rows, n_pcs)
    add("hoptimus_slide", pd.DataFrame(pcs, index=patients,
                                       columns=[f"hos_pc{i + 1}" for i in range(pcs.shape[1])]))
    missing = []
    if os.path.exists(cv_path):
        cv = pd.read_parquet(track(cv_path)).drop_duplicates("patient_id", keep="last").set_index("patient_id")
        add("cellvit_summary", cv[CELLVIT_SUMMARY_COLS].add_prefix("cv_"))
    else:
        missing.append("cellvit_summary")
    if os.path.exists(struct_path):
        st = pd.read_parquet(track(struct_path)).drop_duplicates("patient_id", keep="last").set_index("patient_id")
        num = [c for c in st.columns if c not in _STRUCTURE_META and pd.api.types.is_numeric_dtype(st[c])]
        add("cellvit_structure", st[num].add_prefix("cvs_"))
    else:
        missing.append("cellvit_structure")
    return evr, missing, precision


def load_cptac_patient_table(fit_patients=None, *, n_hoptimus_pcs=10, rna=True, protein=True,
                             cohort="cptac_ccrcc", tables_dir=None, he_dir=None, wsi_dir=None):
    """One aligned patient-level CPTAC ccRCC table (110 slide-mapped tumours).

    Returns {"table": DataFrame indexed by patient_id, "blocks": {block:
    [columns]}, "provenance": {path: sha256}, "fit_patients": [...],
    "fit_scope": "fold" | "all", "programme_genes": {...}}. Blocks:
      structure  -- hoptimus_slide (`hos_pc*`, PCA of the 3072-d H-Optimus-0
                    slide mean+sd embedding fitted on fit_patients),
                    cellvit_summary (`cv_*`: density, mean nuclear area,
                    class fractions; existing 09-25 summary, area/density
                    carry the target_patch_mpp scale, a constant factor) and
                    cellvit_structure (`cvs_*`: morphometrics, kNN graph,
                    tile heterogeneity; present once the 10-12 extraction
                    exists, otherwise recorded as missing);
      responses  -- rna_programmes (`rna_mp*`, `rna_hm_*` on log2(washu+1))
                    and protein_programmes (`prot_mp*`, `prot_hm_*` on umich
                    log-ratios, canonical isoforms via dedupe_symbols);
      competitor -- clinical (age, sex, stage, grade, clear-cell flag),
                    deconvolution (ESTIMATE purity/immune/stromal, CIBERSORT
                    22 fractions, xCell summary scores), depth_proxy
                    (detected genes, log total washu expression);
      context    -- mutation (VHL, PBRM1, SETD2, BAP1, KDM5C any somatic call).
    No outcome column is ever loaded here.
    `cohort="tcga_kirc"` returns the TCGA-KIRC replication table instead
    (`load_tcga_kirc_patient_table`; `protein` is ignored there).
    `cohort="cptac_<type>"` (any key of CPTAC_TYPES; see `load_cptac`) gives
    the same blocks for that type: RNA from washu (bcm where washu is absent),
    the generic clinical competitor (`_generic_clinical`), the type's driver
    mutations, and a `slide_technical` block (slide_frozen) when any selected
    slide is an OCT section. `tcga_<project>` returns `load_tcga(project)`."""
    if cohort == "tcga_kirc":
        return load_tcga_kirc_patient_table(fit_patients, n_hoptimus_pcs=n_hoptimus_pcs, rna=rna)
    if cohort.startswith("tcga_"):
        return load_tcga(cohort[5:], fit_patients, n_hoptimus_pcs=n_hoptimus_pcs, rna=rna)
    ctype = cohort[6:] if cohort.startswith("cptac_") else None
    if cohort != "cptac_ccrcc" and ctype not in CPTAC_TYPES:
        raise ValueError("unknown cohort %r" % cohort)
    kidney = cohort == "cptac_ccrcc"
    if kidney:
        tables, rna_file = CPTAC_CCRCC_DIR, "transcriptomics__washu.parquet"
        patients = cptac_patient_ids()
        h5_path = os.path.join(CPTAC_HE_DESCRIPTORS_DIR, "embeddings.h5")
        cv_path = os.path.join(CPTAC_HE_DESCRIPTORS_DIR, "cellvit_summary.parquet")
        struct_path = os.path.join(CPTAC_CELLVIT_STRUCTURE_DIR, "cellvit_structure.parquet")
    else:
        paths = cptac_type_paths(ctype)
        tables, rna_file = tables_dir or paths["tables"], paths["rna_file"]
        he = he_dir or paths["he_dir"]
        patients = cptac_type_patient_ids(ctype, tables_dir=tables, he_dir=he)
        h5_path, cv_path, struct_path = (os.path.join(he, "embeddings.h5"), os.path.join(he, "cellvit_summary.parquet"),
                                         os.path.join(he, "cellvit_structure.parquet"))
    fit = list(patients if fit_patients is None else fit_patients)
    unknown = sorted(set(fit) - set(patients))
    if unknown:
        raise ValueError("fit_patients not in the slide-mapped cohort: %s" % unknown[:5])
    fit_rows = np.array([patients.index(p) for p in fit])
    prov, blocks, cols = {}, {}, {}

    def _track(path):
        prov[path] = _file_sha256(path)
        return path

    def _add(block, frame):
        frame = frame.reindex(patients)
        blocks[block] = list(frame.columns)
        cols.update({c: frame[c].to_numpy(dtype=float) for c in frame.columns})

    # structure: H-Optimus-0 slide embedding PCs (fold-fitted), CellViT summary + structure
    evr, missing, precision = _slide_structure_blocks(
        _add, _track, patients, fit_rows, n_hoptimus_pcs, h5_path, cv_path, struct_path)
    if not kidney and "cellvit_summary" not in missing:  # true um, as TCGA-KIRC (kidney keeps its 09-25 columns)
        _rescale_cellvit_to_um(cols, cv_path, patients)

    # responses: programme scores (fold-fitted z statistics)
    genes_used = {}
    sets, set_paths = _gene_sets() if (rna or protein) else ({}, [])
    for path in set_paths:
        _track(path)
    if rna:
        t, syms = _molecular_table(rna_file, tables)
        _track(os.path.join(tables, rna_file))
        raw = t.reindex(patients).to_numpy(dtype=float)
        x, s, _ = dedupe_symbols(np.log2(np.clip(raw, 0, None) + 1.0), syms)
        sc, used = programme_scores(x, s, sets, fit_rows)
        _add("rna_programmes", pd.DataFrame({f"rna_{k}": v for k, v in sc.items()}, index=patients))
        genes_used.update({f"rna_{k}": v for k, v in used.items()})
        _add("depth_proxy", pd.DataFrame({"rna_detected_genes": (raw > 0).sum(1).astype(float),
                                          "rna_log_total": np.log(np.nansum(raw, 1))}, index=patients))
    if protein:
        t, syms = _molecular_table("proteomics__umich.parquet", tables)
        _track(os.path.join(tables, "proteomics__umich.parquet"))
        x, s, _ = dedupe_symbols(t.reindex(patients).to_numpy(dtype=float), syms)
        sc, used = programme_scores(x, s, sets, fit_rows)
        _add("protein_programmes", pd.DataFrame({f"prot_{k}": v for k, v in sc.items()}, index=patients))
        genes_used.update({f"prot_{k}": v for k, v in used.items()})

    # competitor: clinical, deconvolution
    clin = pd.read_csv(_track(os.path.join(tables, "clinical__mssm.csv")), low_memory=False)
    clin = clin.set_index(clin["Patient_ID"].astype(str))
    absent_clinical = []
    if kidney:
        age = pd.to_numeric(clin["age"].astype(str).str.replace(">=", "", regex=False), errors="coerce")
        stage = clin["tumor_stage_pathological"].map({"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4})
        grade = clin["histologic_grade"].astype(str).str.extract(r"G(\d)")[0].astype(float)
        _add("clinical", pd.DataFrame({
            "age": age, "sex_male": (clin["sex"] == "Male").astype(float), "stage": stage.astype(float),
            "grade": grade, "clear_cell": (clin["histologic_type"] == "Clear cell renal cell carcinoma").astype(float)}))
    else:
        frame, absent_clinical = _generic_clinical(clin, CPTAC_TYPES[ctype]["flags"])
        _add("clinical", frame)
        tech = _slide_technical(wsi_dir or cptac_type_paths(ctype)["wsi_dir"], patients)
        if tech is not None:
            _add("slide_technical", tech)
    pur = pd.read_parquet(_track(os.path.join(tables, "tumor_purity__washu.parquet")))
    cib = pd.read_parquet(_track(os.path.join(tables, "cibersort__washu.parquet")))
    xc = pd.read_parquet(_track(os.path.join(tables, "xcell__washu.parquet")))
    for frame in (pur, cib, xc):
        frame.index = frame.index.astype(str)
    cib = cib.drop(columns=[c for c in CIBERSORT_META if c in cib.columns])
    deconv = pd.concat([pur.add_prefix("estimate_"), cib.add_prefix("cibersort_"),
                        xc[list(XCELL_SUMMARY)].add_prefix("xcell_")], axis=1)
    _add("deconvolution", deconv.reindex(patients))
    mut = pd.read_parquet(_track(os.path.join(tables, "somatic_mutation__harmonized.parquet")),
                          columns=["Gene"])
    mut.index = mut.index.astype(str)
    drivers = CPTAC_DRIVER_GENES if kidney else CPTAC_TYPES[ctype]["drivers"]
    _add("mutation", pd.DataFrame({f"mut_{g}": [float(p in set(mut.index[mut["Gene"] == g])) for p in patients]
                                   for g in drivers}, index=patients))

    table = pd.DataFrame(cols, index=pd.Index(patients, name="patient_id"))
    out = {"table": table, "blocks": blocks, "provenance": prov, "fit_patients": sorted(fit),
           "fit_scope": "all" if fit_patients is None else "fold", "missing_blocks": missing,
           "programme_genes": genes_used, "hoptimus_slide_explained_variance": [float(v) for v in evr],
           "hoptimus_precision": precision, "cohort": cohort}
    if not kidney:
        coverage = {"n_patients": len(patients)}
        coverage.update({b: int(table[c].notna().any(axis=1).sum()) for b, c in blocks.items()})
        out.update({"coverage": coverage, "absent_clinical": absent_clinical, "rna_file": rna_file})
    return out


def load_cptac(cancer_type, fit_patients=None, **kwargs):
    """Patient-level table of one CPTAC type (`ccrcc`, or a key of CPTAC_TYPES):
    `load_cptac_patient_table(fit_patients, cohort="cptac_<type>", ...)`."""
    return load_cptac_patient_table(fit_patients, cohort=f"cptac_{cancer_type}", **kwargs)


TCGA_KIRC_HE_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/"
                    "external/tcga_kirc_he_descriptors_20261013")
CELLVIT_TARGET_PATCH_MPP = 0.25  # CellViT HIPT x40 segments at 0.25 um/px


def tcga_kirc_patient_ids(he_dir=None, tables_dir=None):
    """TCGA-KIRC patients with an H-Optimus-0 slide descriptor AND RNA."""
    with h5py.File(os.path.join(he_dir or TCGA_KIRC_HE_DIR, "embeddings.h5"), "r") as f:
        slide = {p.decode() if isinstance(p, bytes) else str(p) for p in f["patient_ids"][:]}
    rna = pd.read_parquet(os.path.join(tables_dir or TCGA_KIRC_DIR, "rna_seq_v2_mrna.parquet"), columns=["patient_id"])
    return sorted(slide & set(rna["patient_id"].astype(str)))


def load_tcga_kirc_patient_table(fit_patients=None, *, n_hoptimus_pcs=10, rna=True, he_dir=None, tables_dir=None):
    """TCGA-KIRC: `load_tcga("kirc", ...)` (see there)."""
    return load_tcga("kirc", fit_patients, n_hoptimus_pcs=n_hoptimus_pcs, rna=rna, he_dir=he_dir,
                     tables_dir=tables_dir)


def load_tcga(project, fit_patients=None, *, n_hoptimus_pcs=10, rna=True, he_dir=None, tables_dir=None):
    """TCGA-KIRC replication counterpart of `load_cptac_patient_table`: same
    return keys and the same structure / RNA-programme column names, one row
    per patient with a slide descriptor and RNA. Slide descriptors come from
    `dbench/scripts/tcga_kirc_wsi.py` (the CPTAC tiling/aggregation and CellViT
    code, one GDC DX1 slide per patient). Blocks: hoptimus_slide, cellvit_summary
    (cv_*; area/density rescaled from CellViT's target_patch_mpp to the slide's
    level-0 mpp, i.e. true um -- the CPTAC cv_ columns are not: constant 4x),
    cellvit_structure, rna_programmes (log2(RSEM+1)), depth_proxy, clinical
    (age, sex, stage; grade = NaN because the TCGA PanCancer clinical table has
    no grade; clear_cell = ICD-O-3 8310/3). Not available: protein,
    deconvolution, mutation (listed in `missing_blocks`). Outcomes are never in
    the table: see `load_tcga_kirc_survival_cohort`.

    Generalised (Agent W): `project` is "kirc" or the TCGA project of a CPTAC
    type ("luad", "lusc", "brca", "hnsc", "ucec", "coad", "paad", "gbm", "ov";
    paths from `tcga_project_paths`). Stage = leading Roman numeral of
    AJCC_PATHOLOGIC_TUMOR_STAGE; sex_male only when both sexes occur; the
    histology flags of the matching CPTAC type from ICD-O-3 codes
    (`CPTAC_TYPES[...]["tcga_flags"]`); KIRC keeps clear_cell and grade = NaN."""
    project = project.lower().replace("tcga-", "")
    paths = tcga_project_paths(project)
    he_dir, tables_dir = he_dir or paths["he_dir"], tables_dir or paths["tables"]
    patients = tcga_kirc_patient_ids(he_dir, tables_dir)
    fit = list(patients if fit_patients is None else fit_patients)
    unknown = sorted(set(fit) - set(patients))
    if unknown:
        raise ValueError("fit_patients not in the slide-mapped cohort: %s" % unknown[:5])
    fit_rows = np.array([patients.index(p) for p in fit])
    prov, blocks, cols = {}, {}, {}

    def _track(path):
        prov[path] = _file_sha256(path)
        return path

    def _add(block, frame):
        frame = frame.reindex(patients)
        blocks[block] = list(frame.columns)
        cols.update({c: frame[c].to_numpy(dtype=float) for c in frame.columns})

    cv_path = os.path.join(he_dir, "cellvit_summary.parquet")
    evr, missing, precision = _slide_structure_blocks(_add, _track, patients, fit_rows, n_hoptimus_pcs,
                                           os.path.join(he_dir, "embeddings.h5"), cv_path,
                                           os.path.join(he_dir, "cellvit_structure.parquet"))
    if "cellvit_summary" not in missing:
        _rescale_cellvit_to_um(cols, cv_path, patients)
    missing += ["protein_programmes", "deconvolution", "mutation"]

    genes_used = {}
    if rna:
        sets, set_paths = _gene_sets()
        for path in set_paths:
            _track(path)
        t = pd.read_parquet(_track(os.path.join(tables_dir, "rna_seq_v2_mrna.parquet")))
        t = t.set_index(t["patient_id"].astype(str)).drop(columns=["patient_id"])
        raw = t.reindex(patients).to_numpy(dtype=float)
        x, s, _ = dedupe_symbols(np.log2(np.clip(raw, 0, None) + 1.0), [str(c) for c in t.columns])
        sc, used = programme_scores(x, s, sets, fit_rows)
        _add("rna_programmes", pd.DataFrame({f"rna_{k}": v for k, v in sc.items()}, index=patients))
        genes_used.update({f"rna_{k}": v for k, v in used.items()})
        _add("depth_proxy", pd.DataFrame({"rna_detected_genes": (raw > 0).sum(1).astype(float),
                                          "rna_log_total": np.log(np.nansum(raw, 1))}, index=patients))

    clin = pd.read_parquet(_track(os.path.join(tables_dir, "clinical.parquet")))
    clin = clin.set_index(clin["patient_id"].astype(str))
    if project == "kirc":
        stage = clin["AJCC_PATHOLOGIC_TUMOR_STAGE"].map({"STAGE I": 1, "STAGE II": 2, "STAGE III": 3, "STAGE IV": 4})
        _add("clinical", pd.DataFrame({
            "age": pd.to_numeric(clin["AGE"], errors="coerce"), "sex_male": (clin["SEX"] == "Male").astype(float),
            "stage": stage.astype(float), "grade": np.nan,
            "clear_cell": (clin["ICD_O_3_HISTOLOGY"] == "8310/3").astype(float)}, index=clin.index))
    else:
        ctype = next(t for t, s in CPTAC_TYPES.items() if s["tcga"] == project)
        sex = (clin["SEX"] == "Male").astype(float)
        frame = {"age": pd.to_numeric(clin["AGE"], errors="coerce")}
        if sex.reindex(patients).nunique() > 1:
            frame["sex_male"] = sex
        frame["stage"] = _roman_stage(clin["AJCC_PATHOLOGIC_TUMOR_STAGE"]).set_axis(clin.index)
        for name, codes in CPTAC_TYPES[ctype]["tcga_flags"].items():
            frame[name] = clin["ICD_O_3_HISTOLOGY"].isin(codes).astype(float)
        _add("clinical", pd.DataFrame(frame, index=clin.index))

    table = pd.DataFrame(cols, index=pd.Index(patients, name="patient_id"))
    # CellViT runs in the background and is paused/resumed: slides not yet
    # segmented are NaN (never 0); the coverage per block is stated here.
    coverage = {"n_patients": len(patients)}
    coverage.update({b: int(table[c].notna().any(axis=1).sum()) for b, c in blocks.items()})
    return {"table": table, "blocks": blocks, "provenance": prov, "fit_patients": sorted(fit),
            "fit_scope": "all" if fit_patients is None else "fold", "missing_blocks": missing,
            "programme_genes": genes_used, "hoptimus_slide_explained_variance": [float(v) for v in evr],
            "hoptimus_precision": precision, "coverage": coverage, "cohort": f"tcga_{project}"}


def load_tcga_kirc_survival_cohort(cohort_spec, candidate_features, *, he_dir=None, tables_dir=None):
    """TCGA-KIRC counterpart of `load_cptac_survival_cohort` (same contract:
    refuses without `frozen_candidates_sha256`). Endpoints (cBioPortal
    PanCancer Atlas, months): "os", "pfs", "dss". `candidate_features` maps
    candidate id -> vector over `tcga_kirc_patient_ids()` order."""
    if not cohort_spec.get("frozen_candidates_sha256"):
        raise ValueError("survival cohort spec must name the frozen candidate hash before outcomes are read")
    ep = cohort_spec["endpoint"]
    if ep not in ("os", "pfs", "dss"):
        raise ValueError("unknown endpoint %r" % ep)
    tables_dir = tables_dir or TCGA_KIRC_DIR
    patients = tcga_kirc_patient_ids(he_dir, tables_dir)
    clin = pd.read_parquet(os.path.join(tables_dir, "clinical.parquet"))
    clin = clin.set_index(clin["patient_id"].astype(str)).reindex(patients)
    time_ = pd.to_numeric(clin[f"{ep.upper()}_MONTHS"], errors="coerce").to_numpy(float)
    event = clin[f"{ep.upper()}_STATUS"].map(
        lambda v: float(v[0]) if isinstance(v, str) and v[:1] in ("0", "1") else np.nan).to_numpy(float)
    tab = load_tcga_kirc_patient_table(None, rna=False, he_dir=he_dir, tables_dir=tables_dir)["table"]
    names = list(cohort_spec["covariates"])
    return {"time": time_, "event": event, "covariates": tab.reindex(patients)[names].to_numpy(float),
            "covariate_names": names,
            "expression": {k: np.asarray(v, dtype=float) for k, v in candidate_features.items()}}


STRUCTURE_BLOCKS = ("hoptimus_slide", "cellvit_summary", "cellvit_structure")
COMPETITOR_BLOCKS = ("clinical", "deconvolution", "depth_proxy", "slide_technical")  # slide_technical: non-kidney types with OCT slides
CONTEXT_BLOCKS = ("clinical", "mutation")
CPTAC_FAMILIES = {
    "structure_to_programme": {"response_block": "rna_programmes", "context": False},
    "structure_to_protein": {"response_block": "protein_programmes", "context": False},
    "structure_x_clinical_context_to_programme": {"response_block": "rna_programmes", "context": True},
}


def cptac_patient_episode(target, fold, *, family="structure_to_programme", n_folds=5, seed=20261012,
                          n_hoptimus_pcs=10, competitor="primary", table_loader=None, cohort="cptac_ccrcc"):
    """A patient-level `DiscoveryEpisode` for the CPTAC tissue_to_patient
    families: rows = tumours, proposal = fit-fold patients, replication =
    held-fold patients (stage-stratified folds), no spatial graph.

    Candidate matrix (modality "candidates", names in metadata gene_symbols):
    every structure column (+ clinical/mutation context columns for the
    context family), z-scored on proposal rows. Response: `target` (a
    programme score from the family's response block), z-scored on proposal
    rows. Baseline (context["baseline"]): the competitor blocks, z-scored on
    proposal rows ("primary": clinical + deconvolution + depth proxy;
    "informational": clinical + depth proxy). Rows missing the response, any structure column or any
    competitor column are dropped and listed (never imputed); a column with
    any missing value among kept rows is dropped and listed. Nulls come from
    `cptac_episode_null_generators` (metadata null_source)."""
    from morpheus_encoder.discovery_episode import DiscoveryEpisode, EpisodePartitions

    if family not in CPTAC_FAMILIES:
        raise ValueError("unknown family %r" % family)
    spec = CPTAC_FAMILIES[family]
    if table_loader is None and cohort != "cptac_ccrcc":
        def table_loader(fit, **kw):
            return load_cptac_patient_table(fit, cohort=cohort, **kw)
    loader = table_loader or load_cptac_patient_table
    base = loader(None, n_hoptimus_pcs=n_hoptimus_pcs, rna=False, protein=False)
    patients = list(base["table"].index)
    stage = (base["table"]["stage"].fillna(0).to_numpy() if "stage" in base["table"]
             else np.zeros(len(patients)))  # a type without stage (GBM): one stratum
    folds = cptac_patient_folds(patients, stage, n_folds=n_folds, seed=seed)
    held = set(folds[fold])
    fit = [p for p in patients if p not in held]
    res = loader(fit, n_hoptimus_pcs=n_hoptimus_pcs)
    tab, blocks = res["table"], res["blocks"]
    if target not in blocks.get(spec["response_block"], []):
        raise ValueError("target %r is not in block %s" % (target, spec["response_block"]))
    cand_cols = [c for b in STRUCTURE_BLOCKS for c in blocks.get(b, [])]
    if spec["context"]:
        cand_cols += [c for b in CONTEXT_BLOCKS for c in blocks.get(b, [])]
    if competitor not in ("primary", "informational"):
        raise ValueError("competitor must be 'primary' or 'informational'")
    comp_blocks = COMPETITOR_BLOCKS if competitor == "primary" else ("clinical", "depth_proxy", "slide_technical")
    comp_cols = [c for b in comp_blocks for c in blocks.get(b, [])]
    # columns missing for most patients are dropped first, then incomplete rows
    dropped_cols = [c for c in cand_cols + comp_cols if tab[c].notna().mean() < 0.9]
    cand_cols = [c for c in cand_cols if c not in dropped_cols]
    comp_cols = [c for c in comp_cols if c not in dropped_cols]
    need = tab[[target] + cand_cols + comp_cols]
    keep = need.notna().all(axis=1)
    dropped_rows = sorted(tab.index[~keep])
    tab = tab.loc[keep]
    pids = list(tab.index)
    prop = np.array([i for i, p in enumerate(pids) if p not in held], dtype=np.int64)
    repl = np.array([i for i, p in enumerate(pids) if p in held], dtype=np.int64)

    def _z(frame):
        a = frame.to_numpy(dtype=float)
        mu, sd = a[prop].mean(0), a[prop].std(0)
        sd[sd < 1e-12] = 1.0
        return (a - mu) / sd

    X = _z(tab[cand_cols])
    y = _z(tab[[target]]).reshape(-1)
    B = _z(tab[comp_cols])
    bank = {}
    for name, members in (("hoptimus_slide", ["hoptimus_slide"]), ("cellvit_structure", ["cellvit_summary",
                                                                                        "cellvit_structure"]),
                          ("clinical_context", list(CONTEXT_BLOCKS) if spec["context"] else [])):
        names = [c for b in members for c in blocks.get(b, []) if c in cand_cols]
        if names:
            idx = [cand_cols.index(c) for c in names]
            bank[name] = {"X": X[:, idx], "columns": names, "response_free": True,
                          "modality": "pathology" if name == "hoptimus_slide" else
                          ("tissue_structure" if name == "cellvit_structure" else "clinical"),
                          "scale": "patient (whole slide)", "fitted_on": "proposal patients"}
    metadata = {
        "episode": "cptac_patient", "family": family, "target": target, "fold": int(fold),
        "n_folds": int(n_folds), "fold_seed": int(seed), "gene_symbols": cand_cols,
        "baseline_columns": comp_cols, "competitor": competitor, "fit_patients": [pids[i] for i in prop],
        "held_patients": [pids[i] for i in repl], "dropped_rows": dropped_rows, "dropped_columns": dropped_cols,
        "missing_blocks": res["missing_blocks"], "provenance": res["provenance"],
        "programme_genes": res["programme_genes"].get(target, []), "null_source": "replication_structured",
        "response_standardised": "z-score on proposal patients (patient is the unit; no within-patient centring)",
        "bank": bank,
    }
    return DiscoveryEpisode(
        modalities={"candidates": X}, response=y,
        partitions=EpisodePartitions(proposal=prop, replication=repl, null=np.array([], dtype=np.int64)),
        context={"baseline": B, "patient_codes": np.array(pids),
                 "stage": (tab["stage"].fillna(0).to_numpy(dtype=float) if "stage" in tab else np.zeros(len(pids)))},
        metadata=metadata)


def cptac_episode_null_generators(episode, *, constituent_strata="stage"):
    """Structured nulls on the held-out patients, `(index, X, y, context) ->
    (X, y, context)` closures that ignore their arguments (the hest/bmmc
    contract): `patient_perm` permutes the response across held patients;
    `descriptor_perm_within_stage` permutes whole candidate rows (the slide
    descriptors) among held patients of the same stage, keeping the response
    and the competitor fixed; `interaction_break` (factory, |S| >= 2)
    permutes the candidate's first constituent within stage."""
    repl = episode.partitions.replication
    X_r = episode.observed_matrix()[repl]
    y_r = np.asarray(episode.response, dtype=float)[repl]
    strata = np.asarray(episode.context[constituent_strata])[repl]
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]

    def _within(rng):
        perm = np.arange(len(y_r))
        for g in groups:
            perm[g] = g[rng.permutation(len(g))]
        return perm

    def patient_perm(index, X, y, context):
        return X_r, y_r[np.random.default_rng(index).permutation(len(y_r))], None

    def descriptor_perm_within_stage(index, X, y, context):
        return X_r[_within(np.random.default_rng(index))], y_r, None
    # the runner applies the same row permutation to any extra design columns
    descriptor_perm_within_stage.row_permutation = lambda index: _within(np.random.default_rng(index))

    def interaction_break(candidate):
        j = int(candidate.variables[0])

        def gen(index, X, y, context):
            Xn = X_r.copy()
            Xn[:, j] = X_r[_within(np.random.default_rng(index)), j]
            return Xn, y_r, None
        return gen

    return {"patient_perm": patient_perm, "descriptor_perm_within_stage": descriptor_perm_within_stage,
            "interaction_break": interaction_break}


def load_cptac_survival_cohort(cohort_spec, candidate_features):
    """`load_cohort` callable for `translational.run_predeclared_survival`
    (which reads the frozen survival spec BEFORE calling this). Requires the
    spec entry to carry `frozen_candidates_sha256` so outcomes are only ever
    read for a hashed candidate list. `candidate_features` maps candidate id
    -> feature vector over `cptac_patient_ids()` order. Endpoints: "os"
    (Overall survival, days; Survival status) or "rfs_or_death" (event =
    recurrence or death; time = recurrence-free days if recurred, else
    overall-survival days)."""
    if not cohort_spec.get("frozen_candidates_sha256"):
        raise ValueError("survival cohort spec must name the frozen candidate hash before outcomes are read")
    patients = cptac_patient_ids()
    clin = pd.read_csv(os.path.join(CPTAC_CCRCC_DIR, "clinical__mssm.csv"), low_memory=False)
    clin = clin.set_index(clin["Patient_ID"].astype(str)).reindex(patients)
    os_t = pd.to_numeric(clin["Overall survival, days"], errors="coerce").to_numpy(float)
    os_e = pd.to_numeric(clin["Survival status (1, dead; 0, alive)"], errors="coerce").to_numpy(float)
    if cohort_spec["endpoint"] == "os":
        time_, event = os_t, os_e
    elif cohort_spec["endpoint"] == "rfs_or_death":
        rec = pd.to_numeric(clin["Recurrence status (1, yes; 0, no)"], errors="coerce").to_numpy(float)
        rfs = pd.to_numeric(clin["Recurrence-free survival, days"], errors="coerce").to_numpy(float)
        time_ = np.where(rec == 1, rfs, os_t)
        event = ((rec == 1) | (os_e == 1)).astype(float)
    else:
        raise ValueError("unknown endpoint %r" % cohort_spec["endpoint"])
    tab = load_cptac_patient_table(None, rna=False, protein=False)["table"].reindex(patients)
    names = list(cohort_spec["covariates"])
    return {"time": time_, "event": event, "covariates": tab[names].to_numpy(float), "covariate_names": names,
            "expression": {k: np.asarray(v, dtype=float) for k, v in candidate_features.items()}}


def transfer_candidate(receipt_row, features, symbols):
    """Evaluate one frozen discovery candidate's feature on new
    patient-level data, keyed by GENE SYMBOL (never by raw column index --
    a candidate discovered against HEST's HVG-indexed spot matrix has no
    stable column position in an unrelated cohort's own gene ordering; only
    its gene symbols are portable). This is the L2/L3 transfer step (D4):
    it evaluates the candidate's feature, it does not itself compute a
    replication gain or p-value (that stays with
    `general_discovery.evaluate_frozen_candidate` on whichever episode the
    caller builds around the returned feature values).

    `receipt_row` is any mapping (dict, pandas.Series, ...) exposing:
      - "gene_symbols": the candidate's variables as gene symbol strings,
        in the same order its "weights" (if any) were defined over.
      - "feature_form": "product" (default, matches
        `general_discovery.Candidate`) or "linear_subspace".
      - "weights": only used for "linear_subspace"; defaults to an
        unweighted sum (all ones) when absent, exactly like `Candidate`.

    `features` is a real (n_patients, n_genes) matrix; `symbols` is its
    matching gene-symbol column index (`len(symbols) == features.shape[1]`).

    If ANY of the candidate's gene symbols is absent from `symbols`, the
    WHOLE candidate is `not_transferable` -- evaluating over only the
    genes that happen to be present would silently change what is being
    measured (a 3-gene product with 1 gene missing is not a diluted version
    of the same candidate, it is a different candidate).
    """
    def _get(key, default=None):
        if hasattr(receipt_row, "get"):
            return receipt_row.get(key, default)
        try:
            return receipt_row[key]
        except (KeyError, IndexError, TypeError):
            return default

    gene_symbols = list(_get("gene_symbols") or [])
    if not gene_symbols:
        raise ValueError("receipt_row must carry a non-empty gene_symbols list")
    feature_form = _get("feature_form", "product") or "product"
    if feature_form not in ("product", "linear_subspace"):
        raise ValueError("unknown feature_form %r" % feature_form)
    weights = _get("weights", None)

    features = np.asarray(features, dtype=float)
    symbols = list(symbols)
    if features.shape[1] != len(symbols):
        raise ValueError("features width must match len(symbols)")
    symbol_to_col = {s: i for i, s in enumerate(symbols)}

    missing = [s for s in gene_symbols if s not in symbol_to_col]
    if missing:
        return {"status": "not_transferable", "gene_symbols": gene_symbols,
               "missing_symbols": missing, "feature_values": None}

    cols = [symbol_to_col[s] for s in gene_symbols]
    values = features[:, cols]
    if feature_form == "linear_subspace":
        w = np.asarray(weights, dtype=float) if weights else np.ones(len(cols))
        if len(w) != len(cols):
            raise ValueError("weights must match gene_symbols length")
        feature_values = values @ w
    else:
        feature_values = np.prod(values, axis=1)

    return {"status": "transferred", "gene_symbols": gene_symbols,
           "feature_form": feature_form, "n_genes_matched": len(cols),
           "feature_values": feature_values}


# ---------------------------------------------------------------------------
# BMMC CITE-seq -> external CITE-seq transfer (Stuart et al. 2019 bone marrow)
#
# The registered BMMC run builds its candidate columns on the proposal donors:
# raw genes = log1p CPM (1e4) of the top-variance genes, PCA = loadings of the
# proposal-standardised raw genes, FM views = frozen SCimilarity / scGPT
# embeddings, programmes = mean log1p CPM over published gene sets; then every
# column is centred within donor and scaled by its proposal-row sd
# (hypothesis_generator._proposal_standardise). None of these fitted
# quantities is written to the receipts, so `bmmc_transfer_params` recomputes
# them deterministically from the registered episode and they are stored in a
# separate file next to the run (receipts and runner untouched).
# ---------------------------------------------------------------------------

STUART2019_FM_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                     "molecular_fm_embeddings_20261010/stuart2019")
CITE_EXTERNAL_RULE = (
    "post-registration rule (the BMMC v0.4 protocol declares no external cohort): the frozen feature is built on "
    "the external cells with the registered run's proposal-fitted transforms, every column centred within the "
    "external sample (hashtag) as the run centres within donor; the antibody target is CLR-normalised over the "
    "external panel and centred within sample; association = Spearman over cells; p = permutation of the target "
    "within sample (9,999); replicated = same sign as in BMMC and p <= 0.05")
TRANSFER_VIEWS = ("raw_genes", "pca", "programmes", "scgpt", "scimilarity")


def _centre_within(values, groups):
    values = np.array(values, dtype=float, copy=True)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        rows = groups == g
        values[rows] -= values[rows].mean(0)
    return values


def _view_columns(view: dict) -> list:
    return [str(c) for c in view.get("columns", [])]


def bmmc_transfer_params(episode, *, seed: int | None = None) -> dict:
    """Everything needed to rebuild the bank columns of one registered BMMC
    episode (built with ``bank=True``) on new cells: the raw-gene list, the
    PCA standardisation and loadings (recomputed exactly as the bank does and
    checked against the bank's own scores), and for every transferable view
    the proposal-row sd after within-donor centring. ``seed`` is the bank's
    seed, which the BMMC episode sets to its fold seed (read from the episode
    metadata when not given)."""
    from sklearn.utils.extmath import randomized_svd
    if seed is None:
        seed = int(episode.metadata.get("seed", 0))
    bank = episode.metadata["bank"]
    proposal = np.asarray(episode.partitions.proposal)
    groups = np.asarray(episode.context["donor_codes"])
    out = {"views": {}, "fold": episode.metadata.get("fold"), "target": episode.metadata.get("target"),
           "fit_donors": list(episode.metadata.get("fit_donors") or []),
           "episode_content_sha256": episode.receipt_identity()["content_sha256"]}
    raw = np.asarray(bank["raw_genes"]["X"], dtype=np.float64)
    genes = [c.split(":", 1)[-1] for c in _view_columns(bank["raw_genes"])]
    mu, sd = raw[proposal].mean(0), raw[proposal].std(0)
    sd[sd < 1e-8] = 1.0
    if "pca" in bank:
        k = int(np.asarray(bank["pca"]["X"]).shape[1])
        Z = (raw - mu) / sd
        _, _, vt = randomized_svd(Z[proposal], n_components=k, n_iter=5, random_state=seed, flip_sign=True)
        scores = Z @ vt.T
        err = float(np.max(np.abs(scores - np.asarray(bank["pca"]["X"], dtype=np.float64))))
        out["pca"] = {"loadings": vt, "max_abs_error_vs_bank": err}
    out["raw_genes"] = {"genes": genes, "proposal_mean": mu, "proposal_sd": sd}
    for name in TRANSFER_VIEWS:
        if name not in bank:
            continue
        X = _centre_within(np.asarray(bank[name]["X"], dtype=np.float64), groups)
        s = X[proposal].std(0)
        s[s < 1e-12] = 1.0
        out["views"][name] = {"columns": [c.split(":", 1)[-1] for c in _view_columns(bank[name])], "proposal_sd": s}
    return out


def save_transfer_params(path, params: dict) -> str:
    """One npz per (fold, target) episode; arrays stored flat, metadata as JSON."""
    arrays, meta = {}, {k: params[k] for k in ("fold", "target", "fit_donors", "episode_content_sha256")}
    arrays["raw_genes__genes"] = np.asarray(params["raw_genes"]["genes"])
    arrays["raw_genes__proposal_mean"] = params["raw_genes"]["proposal_mean"]
    arrays["raw_genes__proposal_sd"] = params["raw_genes"]["proposal_sd"]
    if "pca" in params:
        arrays["pca__loadings"] = params["pca"]["loadings"]
        meta["pca_max_abs_error_vs_bank"] = params["pca"]["max_abs_error_vs_bank"]
    for name, v in params["views"].items():
        arrays["view__%s__columns" % name] = np.asarray(v["columns"])
        arrays["view__%s__proposal_sd" % name] = v["proposal_sd"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, meta=np.asarray(json.dumps(meta, default=str)), **arrays)
    return path


def load_transfer_params(path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = dict(meta, views={}, raw_genes={"genes": [str(g) for g in z["raw_genes__genes"]],
                                          "proposal_mean": z["raw_genes__proposal_mean"],
                                          "proposal_sd": z["raw_genes__proposal_sd"]})
    if "pca__loadings" in z.files:
        out["pca"] = {"loadings": z["pca__loadings"], "max_abs_error_vs_bank": meta.get("pca_max_abs_error_vs_bank")}
    for f in z.files:
        if f.startswith("view__") and f.endswith("__columns"):
            name = f.split("__")[1]
            out["views"][name] = {"columns": [str(c) for c in z[f]], "proposal_sd": z["view__%s__proposal_sd" % name]}
    return out


def load_stuart2019_cells(fm_root: str = STUART2019_FM_DIR, root: str | None = None) -> dict:
    """Stuart et al. 2019 bone-marrow cells: raw counts (all genes) with library
    sizes, ADT counts, hashtag sample labels, and the frozen-model embeddings
    (SCimilarity, scGPT) aligned by barcode when they exist."""
    from dbench import translational as tr
    d = tr.load_stuart2019_bmcite(root or tr.STUART2019_DIR)
    d["embeddings"] = {}
    for model in ("scimilarity", "scgpt"):
        p = os.path.join(fm_root, model, "stuart2019.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            index = {str(b): i for i, b in enumerate(z["barcodes"])}
            rows = [index.get(c, -1) for c in d["cells"]]
            E = np.full((len(rows), z["embeddings"].shape[1]), np.nan, dtype=np.float32)
            ok = np.asarray(rows) >= 0
            E[ok] = z["embeddings"][np.asarray(rows)[ok]]
            d["embeddings"][model] = E
    return d


def _not_transferable(variable, reason):
    return {"status": "not_transferable", "variable": variable, "reason": reason}


def build_cite_variable(variable: str, cells: dict, params: dict, gene_sets: dict | None = None):
    """One frozen BMMC bank column rebuilt on external cells, scaled exactly
    like the registered run (centred within sample, divided by the BMMC
    proposal sd). Returns (values, info) or a `not_transferable` dict."""
    view, name = variable.split(":", 1)
    if view not in TRANSFER_VIEWS:
        return _not_transferable(variable, "view %r is not transferable to external cells" % view)
    vinfo = params["views"].get(view)
    if vinfo is None or name not in vinfo["columns"]:
        return _not_transferable(variable, "column absent from the registered episode's transform parameters")
    col_sd = float(vinfo["proposal_sd"][vinfo["columns"].index(name)])
    genes = list(cells["genes"])
    gindex = {g: i for i, g in enumerate(genes)}
    lib = np.maximum(np.asarray(cells["rna_total"], dtype=float), 1.0)

    def logcpm(cols):
        return np.log1p(np.asarray(cells["rna_counts"], dtype=float)[:, cols] / lib[:, None] * 1e4)

    info: dict = {"view": view}
    if view == "raw_genes":
        if name not in gindex:
            return _not_transferable(variable, "gene %s absent from the external RNA matrix" % name)
        values = logcpm([gindex[name]])[:, 0]
    elif view == "pca":
        if "pca" not in params:
            return _not_transferable(variable, "PCA loadings not in the transform parameters")
        k = int(name.replace("PC", "")) - 1
        rg = params["raw_genes"]["genes"]
        present = [g in gindex for g in rg]
        if np.mean(present) < 0.9:
            return _not_transferable(variable, "only %.0f%% of the PCA genes are in the external matrix"
                                     % (100 * np.mean(present)))
        Z = np.zeros((len(cells["cells"]), len(rg)))
        idx = [i for i, ok in enumerate(present) if ok]
        Z[:, idx] = (logcpm([gindex[rg[i]] for i in idx]) - params["raw_genes"]["proposal_mean"][idx]) \
            / params["raw_genes"]["proposal_sd"][idx]
        values = Z @ params["pca"]["loadings"][k]
        info["pca_genes_present"] = float(np.mean(present))
    elif view == "programmes":
        set_name = name.split(":", 1)[-1]
        members = (gene_sets or {}).get(set_name)
        if not members:
            return _not_transferable(variable, "gene set %s not supplied" % set_name)
        cols = [gindex[g] for g in members if g in gindex]
        if len(cols) < 3:
            return _not_transferable(variable, "fewer than 3 genes of %s in the external matrix" % set_name)
        values = logcpm(cols).mean(1)
        info["programme_genes_present"] = len(cols)
    else:
        E = cells.get("embeddings", {}).get(view)
        if E is None:
            return _not_transferable(variable, "external cells not embedded with %s" % view)
        values = np.asarray(E[:, int(name)], dtype=float)
        if not np.isfinite(values).all():
            return _not_transferable(variable, "some external cells lack a %s embedding" % view)
    values = _centre_within(values.reshape(-1, 1), cells["sample"])[:, 0] / col_sd
    return values, info


def build_cite_target(adt_target: str, cells: dict):
    """The antibody channel on external cells: name mapped across panels,
    CLR over the external panel, centred within sample."""
    from dbench import translational as tr
    hits = [a for a in cells["adt_names"] if tr.normalise_adt_name(a) == tr.normalise_adt_name(adt_target)]
    if not hits:
        return _not_transferable(adt_target, "antibody %s not in the external panel" % adt_target)
    y = tr.clr_rows(cells["adt_counts"])[:, cells["adt_names"].index(hits[0])]
    return _centre_within(y.reshape(-1, 1), cells["sample"])[:, 0], {"antibody": hits[0]}


def _within_group_permutation_spearman(x, y, groups, *, n_perm: int, seed: int):
    from scipy.stats import rankdata
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm))
    for i in range(int(n_perm)):
        perm = np.empty(len(ry), dtype=np.int64)
        for m in members:
            perm[m] = m[rng.permutation(len(m))]
        null[i] = np.mean(rx * ry[perm])
    return rho, float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n_perm + 1))


def cite_external_replication(hypothesis: dict, cells: dict, params: dict, *, our_sign: int,
                              gene_sets: dict | None = None, n_perm: int = 9999, seed: int = 0) -> dict:
    """The post-registration external rule for one frozen BMMC hypothesis
    ({"target", "variable_names", "feature_form"}) on external cells."""
    out = {"rule": CITE_EXTERNAL_RULE, "registered": False, "target": hypothesis["target"],
           "variable_names": list(hypothesis["variable_names"])}
    t = build_cite_target(hypothesis["target"], cells)
    if isinstance(t, dict):
        return dict(out, **t)
    y, tinfo = t
    cols = []
    for v in hypothesis["variable_names"]:
        b = build_cite_variable(v, cells, params, gene_sets=gene_sets)
        if isinstance(b, dict):
            return dict(out, **b)
        cols.append(b[0])
    V = np.column_stack(cols)
    feature = V.sum(1) if hypothesis.get("feature_form", "linear_subspace") == "linear_subspace" else V.prod(1)
    rho, p = _within_group_permutation_spearman(feature, y, cells["sample"], n_perm=n_perm, seed=seed)
    return dict(out, status="assessed", antibody=tinfo["antibody"], n_cells=int(len(y)),
                n_samples=int(len(np.unique(cells["sample"]))), rho=rho, p_within_sample_perm=p, our_sign=int(our_sign),
                replicated=bool(np.sign(rho) == np.sign(our_sign) and p <= 0.05))


def load_transneo(endpoint="pcr"):
    """(features, sample_ids, endpoint, treatment_arm=None) for the real
    TransNEO breast-cancer NAC cohort (Sammut et al. 2022 Nature). Features
    are the real RNA-seq raw counts (genes x 162 sequenced samples,
    transposed to samples x genes here); `endpoint` selects the real
    response column from the clinical table -- "pcr" (pCR.RD, binarised
    pCR=1/RD=0), "rcb_category" (ordinal 0=pCR,1=I,2=II,3=III), or
    "rcb_score" (continuous RCB.score). Not randomised -- no treatment arm
    (all patients received NAC by protocol)."""
    counts_path = os.path.join(TRANSNEO_DIR,
                               "transneo-diagnosis-RNAseq-rawcounts.tsv.gz")
    clin_path = os.path.join(TRANSNEO_DIR, "Supplementary_Tables.xlsx")
    if not os.path.exists(counts_path) or not os.path.exists(clin_path):
        raise FileNotFoundError(TRANSNEO_DIR)
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)  # genes x samples
    clin = pd.read_excel(clin_path, sheet_name="Supplementary Table 1")
    clin = clin.set_index("Donor.ID")

    sample_ids = [s for s in counts.columns if s in clin.index]
    features = counts[sample_ids].to_numpy(dtype=float).T  # samples x genes

    if endpoint == "pcr":
        y = (clin.loc[sample_ids, "pCR.RD"].astype(str) == "pCR").astype(float)
        y = y.where(clin.loc[sample_ids, "pCR.RD"].notna(), np.nan).to_numpy()
    elif endpoint == "rcb_category":
        order = {"pCR": 0, "RCB-I": 1, "RCB-II": 2, "RCB-III": 3}
        y = clin.loc[sample_ids, "RCB.category"].map(order).to_numpy(dtype=float)
    elif endpoint == "rcb_score":
        y = clin.loc[sample_ids, "RCB.score"].to_numpy(dtype=float)
    else:
        raise ValueError("unknown endpoint %r" % endpoint)

    return features, sample_ids, y, None


def load_ovarian_bevacizumab(endpoint="treatment_effect"):
    """(features, sample_ids, endpoint, treatment_arm=None) for the real
    TCIA Ovarian Bevacizumab Response cohort. NOTE granularity: the response
    label is recorded per SLIDE, not per patient (286 slide records, 78
    unique patients) -- this loader aggregates to one row per (Patient ID,
    slide record) pair, i.e. `sample_ids` has repeats at the patient level
    by construction, exactly reflecting the source file's own granularity
    rather than silently collapsing it. Features are the two real CA-125
    titres (before/after). A handful of real CA-125 readings are assay-
    censored above the instrument's reportable ceiling, written in the
    source file as e.g. "H>200"/"H> 500" rather than a number -- these are
    parsed to their stated numeric floor (200.0 / 500.0) rather than dropped
    or coerced to NaN, since the floor IS the real (lower-bound) reading;
    the "H>" prefix is not otherwise interpreted. `endpoint="treatment_effect"`
    returns the real categorical label (1.0="effective", 0.0="invalid") both
    sheets use as their own name for the two sheets they are split across."""
    p = os.path.join(OVARIAN_BEV_DIR, "CA125-data_20230207.xlsx")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    eff = pd.read_excel(p, sheet_name="Ovary.effective-162")
    inv = pd.read_excel(p, sheet_name="Ovary.invalid-126")
    eff = eff.assign(_effective=1.0)
    inv = inv.assign(_effective=0.0)
    df = pd.concat([eff, inv], ignore_index=True, sort=False)

    if endpoint != "treatment_effect":
        raise ValueError("unknown endpoint %r" % endpoint)
    sample_ids = df["Patient ID"].astype(str).tolist()

    def _parse_ca125(col):
        numeric = pd.to_numeric(col, errors="coerce")
        censored = col.astype(str).str.extract(r">\s*([\d.]+)")[0].astype(float)
        return numeric.fillna(censored).to_numpy(dtype=float)

    features = np.column_stack([_parse_ca125(df["CA-125 before"]),
                               _parse_ca125(df["CA-125 after"])])
    y = df["_effective"].to_numpy(dtype=float)
    return features, sample_ids, y, None


def load_peace(endpoint="ar_copy_number"):
    """(features, sample_ids, endpoint, treatment_arm=None) for the real
    PEACE/CASCADE mCRPC rapid-autopsy cohort (Hasan et al. 2023 Nat Commun).
    Features are the real RNA-seq count matrix (38,066 genes x 45 samples,
    4 patients: PEA172 + 3 CASCADE). There is NO treatment-response endpoint
    anywhere in this substrate (checked directly across all 15 Source_Data
    sheets) -- `endpoint` therefore selects a real molecular/histological
    covariate instead: "ar_copy_number" (AR_copy_number_ACE, continuous) or
    "tumor_content" (TC, continuous). Not randomised, no arm."""
    zdir = os.path.join(PEACE_DIR, "extracted")
    if not os.path.isdir(zdir):
        raise FileNotFoundError(zdir)
    subdirs = [d for d in glob.glob(os.path.join(zdir, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(zdir)
    xlsx = os.path.join(subdirs[0], "Source_Data.xlsx")
    counts = pd.read_excel(xlsx, sheet_name="count_data", index_col=0)  # genes x samples
    meta = pd.read_excel(xlsx, sheet_name="meta_data_RNAseq")
    meta = meta.set_index("sample_ID")

    sample_ids = [s for s in counts.columns if s in meta.index]
    features = counts[sample_ids].to_numpy(dtype=float).T  # samples x genes

    if endpoint == "ar_copy_number":
        y = meta.loc[sample_ids, "AR_copy_number_ACE"].to_numpy(dtype=float)
    elif endpoint == "tumor_content":
        y = meta.loc[sample_ids, "TC"].to_numpy(dtype=float)
    else:
        raise ValueError("unknown endpoint %r" % endpoint)

    return features, sample_ids, y, None


def load_impress(endpoint="pcr"):
    """(features, sample_ids, endpoint, treatment_arm=None) for the real
    IMPRESS breast-cancer NAC cohort (Huang et al. 2023 npj Precis Oncol),
    pooling both subtype sheets (62 HER2+ + 64 TNBC = 126 patients). Features
    are the real clinical/registered-IHC quantitative columns shared or
    comparable across both sheets (Age, RCB value, and, where present per
    subtype, ER%/PR%/CD8/CD163/PD-L1 percentages -- HER2+-only and
    TNBC-only IHC columns are zero-filled with an explicit `_subtype`
    indicator column carrying the real subtype flag so the zero-fill is
    never mistaken for a real reading). `endpoint="pcr"` returns the real
    binary pCR call (HER2+ sheet's "pCR" / TNBC sheet's "pCR (no-0, yes-1)",
    the same variable under two column names); `endpoint="rcb_category"`
    returns "RCB category" as a string (missing = pCR by the paper's own
    design, not data loss)."""
    p = os.path.join(IMPRESS_DIR, "cohort_meta.xlsx")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    her2 = pd.read_excel(p, sheet_name="HER2+")
    tnbc = pd.read_excel(p, sheet_name="TNBC")
    her2 = her2.rename(columns={"pCR": "pcr01"})
    tnbc = tnbc.rename(columns={"pCR (no-0, yes-1)": "pcr01"})
    her2["_subtype"] = 1.0  # HER2+
    tnbc["_subtype"] = 0.0  # TNBC
    common_num = ["Age", "RCB value", "_subtype"]
    her2_ihc = {"CD8-intratumoral  (cutoff 10%, 1- positive, 0-negative)": "cd8_intra",
               "CD163-intratumoral  (cutoff 10%, 1- positive, 0-negative)": "cd163_intra"}
    tnbc_ihc = {"CD8-intratumoral (%)": "cd8_intra",
               "CD163-intratumoral (%)": "cd163_intra"}
    her2 = her2.rename(columns=her2_ihc)
    tnbc = tnbc.rename(columns=tnbc_ihc)
    feat_cols = common_num + ["cd8_intra", "cd163_intra"]
    for c in feat_cols:
        if c not in her2.columns:
            her2[c] = 0.0
        if c not in tnbc.columns:
            tnbc[c] = 0.0
    her2["Unique ID"] = "HER2_" + her2["Unique ID"].astype(str)
    tnbc["Unique ID"] = "TNBC_" + tnbc["Unique ID"].astype(str)
    df = pd.concat([her2, tnbc], ignore_index=True, sort=False)

    sample_ids = df["Unique ID"].astype(str).tolist()
    features = df[feat_cols].to_numpy(dtype=float)
    if endpoint == "pcr":
        y = df["pcr01"].to_numpy(dtype=float)
    elif endpoint == "rcb_category":
        # The two sheets encode the same ordinal category two different
        # ways (HER2+: "I"/"II"/"III"; TNBC: "1.0"/"2.0"/"3.0") and both use
        # a real NaN to mean "pCR, RCB not assessed" (by the paper's own
        # design -- RCB is scored only in non-pCR patients) -- normalised
        # to one consistent string set rather than left as two encodings.
        norm = {"I": "I", "1.0": "I", "II": "II", "2.0": "II",
               "III": "III", "3.0": "III"}
        y = df["RCB category"].astype(str).map(lambda v: norm.get(v, "pCR")).to_numpy()
    else:
        raise ValueError("unknown endpoint %r" % endpoint)

    return features, sample_ids, y, None


def load_cancer_moonshot_catalogue():
    """NOT a (features, sample_ids, endpoint, arm) loader -- the real public
    Cancer Moonshot Biobank route (idc-index, no auth) is imaging-only and
    carries no treatment-response endpoint (checked directly against the
    live IDC index; confirmed again 2026-09-11), so this returns the real,
    live-verified per-cancer-type pathology catalogue (patients/series/
    size_GB/licence) instead of fabricating an endpoint. See
    `dbench.cohorts.AVAILABILITY["cancer_moonshot_biobank"]` for why."""
    p = os.path.join(CMB_DIR, "live_route_verification.json")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return json.load(open(p))


def load_javelin_derived_contrasts():
    """The REAL, but derived (not raw-patient-level), JAVELIN gene/pathway
    contrast tables -- returned as-is with a loud caveat rather than
    reshaped to look like a (features, sample_ids, endpoint, arm) cohort,
    since it is not one (no per-patient rows, no arm column: it is already
    a two-group contrast someone else computed)."""
    path = os.path.join(JAVELIN_DIR, "javelin_gene_delta_ab.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def load_clinical_manifest(cohort):
    p = os.path.join(CLINICAL_MANIFESTS_DIR, cohort, "MANIFEST.json")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return json.load(open(p))
