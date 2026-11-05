"""Write dbench/protocols/{crc,cscc}_spatial_v01_DRAFT.json: the CRC and cSCC
blind-discovery drafts plus the second-wave target families (programme document
Problems 1 to 3). Everything shared (schema, release, relation v03_rff, pass rule,
pooling, nuisance comparison model, null suite) is copied unchanged from the
existing drafts crc_spatial_v01.json / cscc_spatial_v01.json, which follow the
renal_spatial_v07 / ependymoma_spatial_v01 template. Nothing is hashed or registered.

Usage (from the repo root, read-only on everything but the two DRAFT files):
    python scripts/second_wave/build_second_wave_drafts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ROOT / "dbench" / "protocols"
S = "/lambda/nfs/geeg/biorag3_persistent_20260711"
SW = f"{S}/spatial/second_wave_20261022"
TRANSFORMS = ["own", "nbr6", "nbr18", "contrast"]
DRAFT_TAG = "DRAFT r8-SW (Agent S, 2026-10-22): second-wave target families from the programme document " \
            "(Problem 1 metabolic division of labour, Problem 2 invisible field, Problem 3 tumour budding); " \
            "data_sources block; new views budding_like / gland_architecture / inferred_cnv / normal_appearing " \
            "are pending (Agent M) and marked pending_view. Shared blocks unchanged from r7. "

PENDING = "pending runner support: this family needs a view or a row restriction the hest_multi runner does not serve " \
          "at drafting; it is recorded here so the names and definitions are fixed before any outcome is seen, and it " \
          "enters the sweep only when the view exists and the protocol is re-issued and registered"

METABOLIC_TARGETS = [
    "programme:hallmark:HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "programme:hallmark:HALLMARK_GLYCOLYSIS",
    "programme:hallmark:HALLMARK_HYPOXIA",
    "programme:mp:Hypoxia",
    "programme:mp:Respiration",
    "gene:SLC16A1",
    "gene:SLC16A3",
    "gene:LDHA",
    "gene:LDHB",
]
COMPLEMENTARY = {
    "programme:mp:Hypoxia": "programme:mp:Respiration",
    "programme:mp:Respiration": "programme:mp:Hypoxia",
    "programme:hallmark:HALLMARK_HYPOXIA": "programme:hallmark:HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "programme:hallmark:HALLMARK_GLYCOLYSIS": "programme:hallmark:HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "programme:hallmark:HALLMARK_OXIDATIVE_PHOSPHORYLATION": "programme:hallmark:HALLMARK_GLYCOLYSIS",
    "gene:SLC16A1": "gene:SLC16A3", "gene:SLC16A3": "gene:SLC16A1",
    "gene:LDHA": "gene:LDHB", "gene:LDHB": "gene:LDHA",
}


def metabolic_families(organ: str) -> dict:
    common = {
        "transforms": TRANSFORMS,
        "budget": 50,
        "candidate_views": ["programmes", "raw_genes", "composition", "morph_dictionary_v03"],
        "declared_genes": ["SLC16A1", "SLC16A3", "LDHA", "LDHB"],
        "declared_genes_note": "force-included in the raw_genes panel beside the D_P-selected HVGs (pending runner support; "
                               "recorded in episode exclusions when absent from the panel)",
        "complementary_programme": COMPLEMENTARY,
        "complementary_note": "the producer-side (or consumer-side) programme named for each target; a declaration for "
                              "reporting, not a restriction: the search proposes whatever set it ranks",
        "controls": "known-biology baseline B_known already contains the hypoxia and glycolysis programmes (minus the target's "
                    "exclusions), so a pass means the configuration adds to them; B_decomp rejects anything a producer-plus-"
                    "consumer pair explains; interaction_break removes co-localisation artefacts",
        "organ_comparison": "the same three families run in ccRCC (covered by the registered renal-spatial-v07 hypoxia and "
                            "glycolysis targets) and in colorectal cancer, so the renal-versus-colorectal direction difference "
                            "is tested as a compositional interaction, not assumed",
        "pending_view": True,
        "pending_note": PENDING + "; the row restriction by CellViT composition and the composition candidate view are the "
                                  "pending parts",
    }
    fams = {
        "metabolic_consumer_epithelial": {
            **common,
            "targets": METABOLIC_TARGETS,
            "rows": "epithelial-dominant spots (consumer side): the CellViT PanNuke Neoplastic fraction is the largest of the "
                    "three composition fractions under the spot; rows with no segmented nucleus are dropped, nothing imputed",
            "requirement": "every hypothesis carries >= 1 nbr6, nbr18 or contrast molecular variable (the producer-side "
                           "programme or gene in the neighbourhood) and >= 1 composition or morph_dictionary_v03 variable "
                           "(the tissue between the compartments)",
            "question": "which producer-side programme in the neighbourhood, together with the composition or morphology of "
                        "the tissue between, explains the consumer-side metabolic state of epithelial-dominant spots beyond "
                        "every pair",
            "admissibility": {"context": True, "cross_modal": True,
                              "rule": ">= 1 nbr6/nbr18/contrast raw_genes or programmes variable AND >= 1 composition or "
                                      "morph_dictionary_v03 variable; applied at generation and as a runner post-filter"},
        },
        "metabolic_producer_stromal_immune": {
            **common,
            "targets": METABOLIC_TARGETS,
            "rows": "stromal- and immune-dominant spots (producer side): the CellViT Connective or Inflammatory fraction is the "
                    "largest of the three composition fractions under the spot; rows with no segmented nucleus are dropped",
            "requirement": "as metabolic_consumer_epithelial, with the epithelial-dominant neighbourhood as the complementary side",
            "question": "which consumer-side programme in the neighbourhood, together with composition or morphology, explains "
                        "the producer-side metabolic state of stromal- and immune-dominant spots beyond every pair",
            "admissibility": {"context": True, "cross_modal": True,
                              "rule": ">= 1 nbr6/nbr18/contrast raw_genes or programmes variable AND >= 1 composition or "
                                      "morph_dictionary_v03 variable; applied at generation and as a runner post-filter"},
        },
        "metabolic_contrast": {
            **common,
            "targets": [f"contrast:{t}" for t in METABOLIC_TARGETS],
            "rows": "all spots with an H-Optimus-0 embedding and a segmentation",
            "targets_note": "contrast = own - nbr6 of the metabolic programme or gene (the division-of-labour quantity itself); "
                            "the target grammar of hest_multi.parse_target serves gene:, programme:, nbr6:/nbr18:programme: and "
                            "concept: only, so contrast targets are pending runner support",
            "blocked_on": "Agent M post-sweep loader change: extend morpheus/tier2/substrates/hest_multi.parse_target and the "
                          "episode target builder to serve 'contrast:gene:<SYM>' and 'contrast:programme:<prefix>:<name>' "
                          "(own minus nbr6 of the named column, within-patient standardised, same exclusions as the own target); "
                          "not before the registered Meylan and ependymoma sweeps finish, because hest_multi is in their import tree; "
                          "until then this family cannot be swept and is excluded from registration",
            "requirement": "every hypothesis carries >= 1 own variable and >= 1 nbr6, nbr18 or contrast variable; the target's "
                           "own programme and its member genes are excluded under every transform",
            "question": "what configuration of neighbourhood programmes, composition and morphology explains where a spot's "
                        "metabolic programme departs from its neighbourhood",
            "admissibility": {"context": True, "cross_modal": False,
                              "rule": ">= 1 own variable AND >= 1 nbr6/nbr18/contrast variable"},
        },
    }
    for f in fams.values():
        f["problem"] = "1 metabolic division of labour between tissue compartments"
        f["organ"] = organ
    return fams


def invisible_field_family(organ: str, normal_definition: str, healthy_controls: str) -> dict:
    return {
        "invisible_field": {
            "problem": "2 the invisible field: normal-appearing tissue that is molecularly abnormal",
            "organ": organ,
            "targets": [
                "programme:mp:<tumour-elevated rank 1>",
                "programme:mp:<tumour-elevated rank 2>",
                "programme:mp:<tumour-elevated rank 3>",
                "programme:mp:<tumour-elevated rank 4>",
                "inferred_cnv:score",
            ],
            "targets_rule": "the four Gavish meta-programmes most elevated in the same study's tumour spots (mean within-patient-"
                            "standardised score on neoplastic-dominant spots minus on normal_appearing spots, all sections, "
                            "response-free, fixed by name at registration and recorded in programme_targets_source), plus the "
                            "inferred copy-number score (inferred_cnv view: expression-derived, chromosome-arm smoothed, "
                            "within-section rank standardised; definition fixed by Agent M before registration)",
            "rows": "normal_appearing spots: " + normal_definition,
            "candidate_views": ["normal_appearing", "programmes", "raw_genes", "composition", "morph_dictionary_v03"],
            "candidate_views_note": "normal_appearing serves distance_to_neoplastic_um (nearest spot with a neoplastic-dominant "
                                    "segmentation, from the segmentation map, recorded for every spot) and the per-spot pathologist "
                                    "class where it exists; the neighbourhood stromal and immune programmes come from programmes/"
                                    "raw_genes under nbr6/nbr18",
            "transforms": TRANSFORMS,
            "budget": 50,
            "requirement": "every hypothesis carries >= 1 own-spot epithelial-state variable, >= 1 nbr6/nbr18/contrast variable "
                           "and the distance variable or a composition variable (the conjunction hypothesis: abnormality requires "
                           "an epithelial state together with a specific neighbourhood together with proximity, not any one alone); "
                           "sets failing it are counted in proposer_report and not frozen in this family",
            "question": "which configuration of molecular and neighbourhood states marks normal-appearing epithelium near a "
                        "tumour as molecularly abnormal, and in which contexts does it fail to appear",
            "admissibility": {"context": True, "cross_modal": True,
                              "rule": ">= 1 own variable AND >= 1 nbr6/nbr18/contrast variable AND (distance_to_neoplastic_um "
                                      "or a composition variable); applied at generation and as a runner post-filter"},
            "controls": {
                "healthy_sections": healthy_controls,
                "interaction_break": "as the null suite (removes co-localisation artefacts)",
                "contamination_check": "every frozen hypothesis is re-scored (same frozen variables, transforms and relation) after "
                                       "excluding spots with any neoplastic nucleus and spots within one spot (nearest-neighbour "
                                       "ring) of a neoplastic-dominant spot; reported beside the pooled gain; a hypothesis whose "
                                       "gain interval no longer excludes zero is marked contamination_sensitive (a flag, never a gate)",
                "distance_dependence": "the conditional slope of the target on the configuration at distance quartiles, with "
                                       "patient-bootstrap intervals (characterisation step)",
            },
            "exceptions_as_objects": "spots with the same neighbourhood exposure and proximity that lack the field signature are "
                                     "characterised separately (downstream characterisation); their contexts are the candidate "
                                     "protective or permissive conditions",
            "pending_view": True,
            "pending_note": PENDING + "; normal_appearing and inferred_cnv are the pending views",
        }
    }


def budding_families() -> dict:
    front = ("invasive-front spots: the band where the CellViT neoplastic fraction falls from tumour to stroma (spots with "
             "neoplastic fraction between 0.1 and 0.9 and at least one nbr6 neighbour on each side of 0.5), located from the "
             "segmentation map; the Zenodo pathologist spot classes (tumour, stroma, immune mixtures) are the independent "
             "check on the band and never define it")
    explanatory = {
        "candidate_views": ["programmes", "raw_genes", "composition", "morph_dictionary_v03"],
        "declared_genes": ["KRT17", "ITGB1", "POSTN", "CTHRC1"],
        "declared_genes_note": "own-spot malignant signature genes (KRT17, ITGB1) and neighbourhood stromal genes (POSTN, CTHRC1) "
                               "force-included in the raw_genes panel beside the D_P-selected HVGs (pending runner support)",
        "declared_programmes": ["programme:mp:EMT-I", "programme:mp:EMT-II", "programme:mp:EMT-III", "programme:mp:EMT IV",
                                "programme:mp:Hypoxia", "programme:hallmark:HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
        "declared_programmes_note": "partial-EMT and hypoxia (own spot) and extracellular-matrix organisation (neighbourhood) as "
                                    "served by the programmes view; a declaration for reporting, not a restriction",
    }
    return {
        "tumour_budding": {
            "problem": "3 why glands break into buds at some stretches of the invasive front",
            "organ": "colorectal",
            "targets": ["budding_like:index", "gland_architecture:index"],
            "targets_rule": "budding_like:index = count of small isolated clusters of one to four neoplastic nuclei surrounded by "
                            "connective-tissue nuclei within the spot radius (CellViT PanNuke classes); gland_architecture:index = "
                            "epithelial nuclear-ring score from the same segmentation; both within-patient standardised; definitions "
                            "fixed by Agent M before registration and checked once against an ITBCC-style count on the Zenodo "
                            "full-resolution H&E (small internal check, informational)",
            "rows": front,
            **explanatory,
            "transforms": TRANSFORMS,
            "budget": 50,
            "requirement": "every hypothesis carries >= 1 own-spot molecular variable and >= 1 nbr6/nbr18/contrast variable "
                           "(malignant state together with stromal context)",
            "question": "within one invasive front, which configuration of malignant state, stromal programmes, immune composition "
                        "and morphology separates budding-like stretches from intact-gland stretches",
            "admissibility": {"context": True, "cross_modal": True,
                              "rule": ">= 1 own raw_genes/programmes variable AND >= 1 nbr6/nbr18/contrast variable; applied at "
                                      "generation and as a runner post-filter"},
            "controls": {
                "decomposition": "B_decomp is the test of the central claim, that budding depends on a conjunction of malignant "
                                 "state and stromal context rather than on either",
                "cross_section_check": "nuclear-shape statistics from the same segmentation (mean nuclear eccentricity and major-axis "
                                       "orientation dispersion under the spot) enter the nuisance comparison model for this family, "
                                       "so a configuration that only tracks the local sectioning angle cannot pass",
            },
            "pending_view": True,
            "pending_note": PENDING + "; budding_like, gland_architecture, the front row restriction and the nuclear-shape nuisance "
                                      "columns are the pending parts",
        },
        "invasive_front_morphology": {
            "problem": "3 (generic version on the same spots)",
            "organ": "colorectal",
            "targets": ["morph:k00", "morph:k03", "morph:k04", "morph:k05", "morph:k06", "morph:k07", "morph:k08", "morph:k15",
                        "morph:k17", "morph:k18", "morph:k20", "morph:k22", "morph:k26", "morph:k27", "morph:k28", "morph:k29"],
            "rows": front,
            "candidate_views": ["raw_genes", "programmes", "composition"],
            "transforms": TRANSFORMS,
            "budget": 50,
            "question": "morphology dictionary coordinates on the invasive-front spots explained by molecular state and composition "
                        "(the generic morphology_from_molecular family restricted to the front)",
            "admissibility": {"context": False, "cross_modal": False,
                              "rule": "molecular and composition variables only; the target's own column and columns with "
                                      "within-patient |r| >= 0.9 to it are dropped"},
            "pending_view": True,
            "pending_note": PENDING + "; the front row restriction is the pending part",
        },
    }


def not_declared(problem: str, reason: str) -> dict:
    return {"problem": problem, "status": "not declared for this cohort in the programme document", "reason": reason,
            "targets_note": "no targets: a family block without targets is not part of the sweep (dbench.prospective_campaign._family_targets)"}


CRC_SECTIONS = {  # HEST id: (patient key, Zenodo sample id, Zenodo slide)
    "ZEN49": ("Patient 1", "A121573_Rep1", "SN048"), "ZEN48": ("Patient 1", "A121573_Rep2", "SN048"),
    "ZEN47": ("Patient 2", "A416371_Rep1", "SN048"), "ZEN46": ("Patient 2", "A416371_Rep2", "SN048"),
    "ZEN45": ("Patient 3", "A120838_Rep1", "SN84"), "ZEN44": ("Patient 3", "A120838_Rep2", "SN84"),
    "ZEN43": ("Patient 4", "A551763_Rep1", "SN123"), "ZEN39": ("Patient 4", "A551763_Rep2", "SN124"),
    "ZEN42": ("Patient 5", "A595688_Rep1", "SN123"), "ZEN38": ("Patient 5", "A595688_Rep2", "SN124"),
    "ZEN40": ("Patient 7", "A938797_Rep1", "SN123"), "ZEN36": ("Patient 7", "A938797_Rep2", "SN124"),
}
CRC_HEALTHY_ABSENT = {"ZEN41": ("Patient 6", "A798015_Rep1", "SN123", "SN123_A798015_Rep1.zip"),
                      "ZEN37": ("Patient 6", "A798015_Rep2", "SN124", "SN124_A798015_Rep2.zip")}
NDPI = {"SN048": "Visium_Frozen_SN_V10B01-048_new_CRC_2021_02_16.ndpi", "SN84": "Visium_Frozen_SN_V19S23-084.ndpi",
        "SN123": "Visium_Frozen_SN_V19S23-123.ndpi", "SN124": "Visium_Frozen_SN_V19S23-124.ndpi"}


def crc_data_sources(zenodo_state: dict) -> dict:
    return {
        "hest_sections": {sid: {"patient": f"valdeolivas2024_crc:{p}", "zenodo_sample": z, "zenodo_slide": s,
                                "st": f"{S}/spatial/hest_data/st/{sid}.h5ad",
                                "wsi_hest": f"{S}/spatial/hest_data/wsis/{sid}.tif (0.46 um/px, 40x, fresh frozen)",
                                "cellvit": f"{S}/spatial/hest_data/cellvit_seg/{sid}_cellvit_seg.parquet (copy of "
                                           f"external/hest_modalities_20260809/cellvit_seg/{sid}_cellvit_seg.parquet)",
                                "hoptimus": f"{S}/external/hest_fm_embeddings_20261016/valdeolivas2024_crc/embeddings/hoptimus/{sid}.npz",
                                "wsi_zenodo": f"{SW}/valdeolivas_zenodo_7760264/{NDPI[s]}"}
                          for sid, (p, z, s) in CRC_SECTIONS.items()},
        "hest_catalogue": f"{S}/spatial/hest_meta/HEST_v1_3_0.csv (column subseries = Zenodo sample id)",
        "healthy_sections": {sid: {"patient": f"valdeolivas2024_crc:{p}", "zenodo_sample": z, "zenodo_slide": s,
                                   "zenodo_file": f"https://zenodo.org/records/7760264/files/{f}",
                                   "local_zip": f"{SW}/valdeolivas_zenodo_7760264/{f}",
                                   "st": f"{SW}/valdeolivas_healthy_hest_layout/st/{sid}.h5ad",
                                   "metadata": f"{SW}/valdeolivas_healthy_hest_layout/metadata/{sid}.json",
                                   "annotations": f"{SW}/valdeolivas_healthy_hest_layout/annotations/{sid}_pathologist_annotations.csv",
                                   "status": "not in the local HEST copy; converted 2026-10-22 to the flat HEST layout "
                                             "(scripts/second_wave/convert_valdeolivas_healthy.py, no patches); loads through "
                                             "hest_multi as patient 6 (ZEN41 1683 and ZEN37 1653 rows at min_depth 200); the "
                                             "full-resolution H&E is a capture area of the Zenodo .ndpi slide, not yet extracted"}
                             for sid, (p, z, s, f) in CRC_HEALTHY_ABSENT.items()},
        "healthy_layout_manifest": f"{SW}/valdeolivas_healthy_hest_layout/manifest.json",
        "cellvit_note": "HEST-distributed CellViT outputs (<id>_cellvit_seg.parquet + .geojson.zip: geometry, class, cell_id) "
                        "exist for all 12 sections under external/hest_modalities_20260809/cellvit_seg and were copied 2026-10-22 "
                        "(sha256-verified) into spatial/hest_data/cellvit_seg, the first entry of morpheus_encoder.learned_search."
                        "episodes_v03.CELLVIT_DIRS, so cellvit_path(sid) resolves every section without a code change",
        "zenodo_7760264": {
            "record": "https://zenodo.org/records/7760264 (Valdeolivas et al. 2024 npj Precis Oncol; doi 10.5281/zenodo.7760264)",
            "licence": "CC BY 4.0 (record metadata license id cc-by-4.0, access_right open, no click-through)",
            "local_dir": f"{SW}/valdeolivas_zenodo_7760264",
            "files": ["Pathology_SpotAnnotations.zip"] + [NDPI[k] for k in ("SN048", "SN84", "SN123", "SN124")],
            "sha256_manifest": f"{SW}/valdeolivas_zenodo_7760264/sha256_manifest.txt",
            "slide_to_samples": {"SN048": ["A121573_Rep1", "A121573_Rep2", "A416371_Rep1", "A416371_Rep2"],
                                 "SN84": ["A120838_Rep1", "A120838_Rep2"],
                                 "SN123": ["A551763_Rep1", "A595688_Rep1", "A798015_Rep1", "A938797_Rep1"],
                                 "SN124": ["A551763_Rep2", "A595688_Rep2", "A798015_Rep2", "A938797_Rep2"]},
            "annotation_use": "pathologist spot classes (tumour, stroma, immune mixtures, non-neoplastic mucosa, ...) define the "
                              "pathologist arm of normal_appearing (Problem 2) and the independent check on the invasive-front "
                              "band (Problem 3); they never enter as candidate variables",
            **zenodo_state,
        },
        "replication_candidates": {
            "GSE294385": {"source": "GEO GSE294385 (Liu et al. 2026 Cancer Cell), 37 Visium sections, 11 patients, Space Ranger outputs",
                          "files": "GSE294385_RAW.tar 7.9 GB: per-sample tar.gz with filtered_feature_bc_matrix, spatial/ (tissue_hires_image.png "
                                   "<= 2000 px, tissue_lowres_image.png, cytassist_image.tiff ~25 MB, tissue_positions.csv, scalefactors)",
                          "h_and_e": "no full-resolution microscope H&E in GEO (hires PNG and CytAssist image only)",
                          "annotations": "none in GEO; sample records carry only 'tissue: Colorectal cancer metastasis' (no site, patient or class)",
                          "licence": "no licence stated on the GEO record; NCBI places no restrictions on use or distribution of GEO data "
                                     "(NCBI policies page, Molecular Data Usage)",
                          "local_dir": f"{SW}/geo_GSE294385 (GSE294385_RAW.tar, samples/<GSM>_<id>.tar.gz, extracted/<GSM>_<id>/, "
                                       "GSE294385_family_text.txt, README.md, sha256_manifest.txt)",
                          "status": "downloaded 2026-10-22 (scripts/second_wave/fetch_geo_crc.sh gse294385); usable for molecular "
                                    "replication only; not usable for morphology or CellViT; not converted"},
            "GSE280318": {"source": "GEO GSE280318 SuperSeries (Oliveira et al. 2025 Nat Genet, 10x Genomics); sub-series GSE280315 Visium HD "
                                    "(5 sections: P1CRC, P2CRC, P5CRC, P3NAT, P5NAT; 8 um bins; tissue_image.btf 5.8-8.1 GB each, 56 GB total "
                                    "with GSE280313 Visium v2 and GSE280314 Xenium)",
                          "h_and_e": "full-resolution microscope images present (tissue_image.btf per Visium HD section; he_image.ome.tif per Xenium section)",
                          "annotations": "no pathologist spot annotations; Metadata.parquet carries deconvolution and unsupervised clustering labels",
                          "licence": "no licence stated on the GEO record; NCBI places no restrictions (as above); the same data on the 10x "
                                     "Genomics site carry Creative Commons Attribution in the HEST catalogue",
                          "overlap_with_meta_training": "the Visium v2 section P2CRC of this study is HEST TENX152, a member of the "
                                                        "meta_training study tenx_demo_bowel (sprint_partitions_v01.json); the Visium HD "
                                                        "sections are not in HEST; replication must use the HD sections only and say so",
                          "local_dir": f"{SW}/geo_GSE280318_visiumhd (P1CRC, P2CRC, P5CRC, P3NAT, P5NAT: filtered_feature_bc_matrix.h5, "
                                       "tissue_positions.parquet.gz, tissue_lowres_image.png.gz, scalefactors_json.json.gz; Metadata.parquet.gz "
                                       "for the three CRC sections only (GEO serves none for the NAT sections, NOT_SERVED.txt); "
                                       "GSE280315_family_text.txt, README.md, sha256_manifest.txt)",
                          "status": "Visium HD expression and positions downloaded 2026-10-22 (scripts/second_wave/fetch_geo_crc.sh "
                                    "gse280318_hd); tissue_image.btf, image.tif and he_image.ome.tif not downloaded; not converted"},
        },
    }


def cscc_data_sources() -> dict:
    ids = {f"MEND{i}": ("Patient 1", "visium", "spatial/hest_data/cellvit_seg") for i in (37, 38, 39, 40)}
    for sid, p in (("NCBI759", 10), ("NCBI760", 10), ("NCBI761", 10), ("NCBI762", 9), ("NCBI763", 9), ("NCBI764", 9),
                   ("NCBI765", 5), ("NCBI766", 5), ("NCBI767", 5), ("NCBI768", 2), ("NCBI769", 2), ("NCBI770", 2)):
        ids[sid] = (f"Patient {p}", "st_100um", "spatial/hest_data/cellvit_seg")
    return {
        "hest_sections": {sid: {"patient": f"ji2020_cscc:{p}", "platform": plat,
                                "st": f"{S}/spatial/hest_data/st/{sid}.h5ad",
                                "wsi_hest": f"{S}/spatial/hest_data/wsis/{sid}.tif",
                                "pixel_size_um": "0.172 (MEND, 40x FFPE)" if sid.startswith("MEND") else "0.55 (NCBI, 20x FFPE)",
                                "cellvit": f"{S}/{cv}/{sid}_cellvit_seg.parquet",
                                "hoptimus": f"{S}/external/hest_fm_embeddings_20261016/ji2020_cscc/embeddings/hoptimus/{sid}.npz"}
                          for sid, (p, plat, cv) in ids.items()},
        "hest_catalogue": f"{S}/spatial/hest_meta/HEST_v1_3_0.csv",
        "normal_skin": "every one of the 16 local sections is disease_state Cancer in the HEST catalogue; no site-matched normal-skin "
                       "ST section is local, so the Problem 2 healthy-section control is unavailable in this cohort and the "
                       "normal_appearing definition rests on the computational arm only (no pathologist spot class exists for Ji 2020)",
        "cellvit_note": "HEST-distributed CellViT outputs exist for all 16 sections under spatial/hest_data/cellvit_seg (the four MEND "
                        "pairs copied there 2026-10-22, sha256-verified, from external/hest_modalities_20260809/cellvit_seg); "
                        "episodes_v03.cellvit_path resolves every section without a code change",
        "pathologist_annotations": "none public for Ji 2020 ST sections",
    }


def build(src: str, dst: str, organ: str, families: dict, data_sources: dict, cohort_gates: list, notes_add: list) -> Path:
    p = json.loads((PROTOCOLS / src).read_text(encoding="utf-8"))
    p["campaign_id"] = p["campaign_id"].replace("-v01-DRAFT", "-v01-DRAFT-SW")
    p["revision"] = DRAFT_TAG + p["revision"]
    p["status"] = "DRAFT (second-wave target families; not hashed, not registered; " + p["status"].split("DRAFT (", 1)[-1]
    p["draft"] = True
    p["registration_note"] = "register as v02; crc_spatial_v01.json and cscc_spatial_v01.json carry .sha256 from the historical " \
                             "lock and must not be modified"
    p["programme_document"] = "MORPHEUS Second-Wave Discovery Programme (Problems 1 to 3 declared here; Problem 4 is a characterisation " \
                              "test on the registered Meylan run; Problem 5 is the cross-organ replication step)"
    # the v01 drafts key the eight blind programme targets as one range placeholder; _expand_target needs a
    # prefix ending in digits, so spell the eight placeholders out (same form as the family target lists)
    so = p["discovery"]["targets"]["spatial_organisation"]
    if "programme:mp:<rank 1..8>" in so:
        note = so.pop("programme:mp:<rank 1..8>")
        for i in range(1, 9):
            so[f"programme:mp:<rank {i}>"] = note
    fams = p["discovery"]["relationship_families"]
    for name, block in families.items():
        if name in fams:
            raise SystemExit(f"family {name} already present in {src}")
        fams[name] = block
    p["discovery"]["pending_views"] = {
        "budding_like": "per-spot budding-like index from the CellViT segmentation (Agent M)",
        "gland_architecture": "per-spot gland-architecture index from epithelial nuclear rings (Agent M)",
        "inferred_cnv": "expression-derived copy-number score per spot (Agent M)",
        "normal_appearing": "row flag + distance_to_neoplastic_um + pathologist class where present (Agent M)",
        "composition": "CellViT PanNuke Neoplastic / Inflammatory / Connective fractions as a candidate view (served today only as "
                       "B_known covariates via episodes_v03.composition)",
        "rule": "a family marked pending_view: true enters the sweep only after its view exists in the runner and this protocol is "
                "re-issued; nothing else in the protocol changes",
    }
    p["discovery"]["comparison_model"]["family_additions"] = {
        "tumour_budding": "nuclear-shape statistics from the CellViT segmentation (mean eccentricity, orientation dispersion) enter the "
                          "nuisance ridge for this family only (cross-section check)",
    }
    p["cohorts"][0]["gates"].extend(cohort_gates)
    p["data_sources"] = data_sources
    p["notes"] = list(p.get("notes") or []) + notes_add
    out = PROTOCOLS / dst
    out.write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    zenodo_state = {}
    manifest = Path(f"{SW}/valdeolivas_zenodo_7760264/sha256_manifest.txt")
    if manifest.exists():
        zenodo_state["download_status"] = "downloaded 2026-10-22, md5 verified against the record, sha256 manifest written"
        zenodo_state["sha256"] = {line.split()[1]: line.split()[0] for line in manifest.read_text().splitlines() if line.strip()}
    else:
        zenodo_state["download_status"] = "download in progress at drafting (scripts/second_wave/fetch_zenodo_7760264.sh)"

    crc_fams = {**metabolic_families("colorectal"),
                **invisible_field_family(
                    "colorectal",
                    "two definitions that must agree: (a) the pathologist spot class 'non-neoplastic epithelium / mucosa' from "
                    "Pathology_SpotAnnotations.zip (Zenodo 7760264), (b) computational: no neoplastic nucleus under the spot "
                    "(CellViT) and morphology dictionary coordinates in the normal-mucosa region (the kept centres most populated "
                    "by pathologist-normal spots on D_P); the agreement rate is reported per section; distance to the nearest "
                    "neoplastic-dominant spot from the segmentation map is recorded for every spot",
                    "the two sections annotated healthy in the same study (Zenodo A798015_Rep1 / A798015_Rep2 = HEST ZEN41 / ZEN37, "
                    "patient 6) are independent controls in which the configuration must be absent; they are NOT in the local HEST "
                    "copy (data_sources.healthy_sections_not_local) and must be converted before registration"),
                **budding_families()}
    crc_gates = [
        {"id": "zenodo_7760264_annotations", "path": f"{SW}/valdeolivas_zenodo_7760264/Pathology_SpotAnnotations.zip", "required": True,
         "note": "pathologist spot classes, CC BY 4.0"},
        {"id": "zenodo_7760264_wsis", "path": f"{SW}/valdeolivas_zenodo_7760264/sha256_manifest.txt", "required": True,
         "note": "four .ndpi whole slides (SN048, SN84, SN123, SN124), md5-verified, sha256 manifest"},
        {"id": "cellvit_outputs", "path": f"{S}/spatial/hest_data/cellvit_seg/ZEN49_cellvit_seg.parquet", "required": True,
         "note": "HEST CellViT parquet for all 12 sections copied into the runner's first CELLVIT_DIRS entry (sha256-verified copies "
                 "of external/hest_modalities_20260809/cellvit_seg)"},
        {"id": "healthy_sections_converted", "path": f"{SW}/valdeolivas_healthy_hest_layout/manifest.json", "required": True,
         "note": "ZEN37 / ZEN41 (A798015 Rep2 / Rep1, patient 6) converted from the Zenodo per-sample zips to the flat HEST layout "
                 "with pathologist annotation CSVs beside them; the Problem 2 healthy-section control"},
    ]
    crc_notes = [
        "second wave (Agent S, 2026-10-22): Problems 1 to 3 families added as pending_view families; the four generic families are unchanged",
        "the 12 HEST sections are 6 patients (HEST patients 1-5 and 7); the programme document's '7 patients' counts the healthy patient 6 "
        "whose two sections (ZEN37, ZEN41) are absent from HEST and were converted from Zenodo into second_wave_20261022/"
        "valdeolivas_healthy_hest_layout (controls for Problem 2, never discovery folds)",
        "replication: GSE294385 molecular-only (no full-resolution H&E in GEO); GSE280318 Visium HD only (its Visium v2 P2CRC section is "
        "HEST TENX152 in meta_training)",
    ]
    out1 = build("crc_spatial_v01.json", "crc_spatial_v01_DRAFT.json", "colorectal", crc_fams,
                 crc_data_sources(zenodo_state), crc_gates, crc_notes)

    cscc_fams = {
        **{k: not_declared("1 metabolic division of labour",
                           "the programme document runs the metabolic family in ccRCC and colorectal cancer only (organ comparison); "
                           "cSCC is not named") for k in ("metabolic_consumer_epithelial", "metabolic_producer_stromal_immune", "metabolic_contrast")},
        **invisible_field_family(
            "cutaneous squamous cell carcinoma",
            "computational definition only (no pathologist spot class exists for Ji 2020): no neoplastic nucleus under the spot "
            "(CellViT) and morphology dictionary coordinates in the normal-epidermis region (the kept centres most populated by "
            "spots with zero neoplastic nuclei on D_P); distance to the nearest neoplastic-dominant spot from the segmentation map "
            "is recorded for every spot; the pathologist arm is recorded as unavailable",
            "no healthy or site-matched normal-skin ST section is local (all 16 HEST sections are disease_state Cancer); the "
            "healthy-section control is recorded as unavailable for this cohort"),
        "tumour_budding": not_declared("3 tumour budding", "the programme document formulates budding for colorectal cancer "
                                       "(Valdeolivas discovery, GSE294385 / GSE280318 replication); not declared for cSCC"),
        "invasive_front_morphology": not_declared("3 (generic version)", "as tumour_budding"),
    }
    cscc_gates = [
        {"id": "cellvit_outputs_ncbi", "path": f"{S}/spatial/hest_data/cellvit_seg/NCBI759_cellvit_seg.parquet", "required": True,
         "note": "HEST CellViT parquet for NCBI759-NCBI770 (on the runner's search list)"},
        {"id": "cellvit_outputs_mend", "path": f"{S}/spatial/hest_data/cellvit_seg/MEND37_cellvit_seg.parquet", "required": True,
         "note": "HEST CellViT parquet for MEND37-MEND40, sha256-verified copies placed on the runner's search list 2026-10-22"},
    ]
    cscc_notes = [
        "second wave (Agent S, 2026-10-22): Problem 2 family added as a pending_view family; Problems 1 and 3 recorded as not declared "
        "for this cohort by the programme document (blocks without targets, not swept); the four generic families are unchanged",
    ]
    out2 = build("cscc_spatial_v01.json", "cscc_spatial_v01_DRAFT.json", "cutaneous squamous cell carcinoma", cscc_fams,
                 cscc_data_sources(), cscc_gates, cscc_notes)
    for out in (out1, out2):
        print("wrote", out.relative_to(ROOT), out.stat().st_size, "bytes")


if __name__ == "__main__":
    sys.exit(main())
