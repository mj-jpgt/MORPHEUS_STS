# Reproducibility gaps in the August 2026 MORPHEUS paper: details and receipts (2026-11-02)

All facts below were read from the frozen workspaces that built the August figures
(`.agents/spatial_organisation_20260824`, `.agents/claims_closure_20260824`, `figures/fig4_prospective_independence`,
`figures/fig6_assay_distillation`). The manuscript text itself is not in this repository (the local `iclr paper.zip` is
the September benchmark paper), so figure numbers below refer to the August figure bundles.

## 1. H&E detector used for assay distillation (Fig. 6 C, D, F; bundle fig6_assay_distillation)

Detector A (primary; `.agents/spatial_organisation_20260824/RUN_LOG.md` lines 44-72 and 105-125;
`detector_A_frozen.json`):
- Input: TCGA-KIRC diagnostic slides (GDC, open access; 519 slides, 513 patients tiled, 510 with a bulk RNA score).
- Tiles: 112 um edge resampled to 224 x 224 px (0.5 um/px) on a 448 um lattice; tissue gate HSV saturation >= 15 and
  value <= 235 on the lowest pyramid level, tile kept if >= 35% tissue; slides with < 40 tiles dropped; about
  1,160 tiles per slide, about 600k tiles.
- Embedding: H-Optimus-0 (bioptimus/H-optimus-0, revision b145cc1e..., weight sha bf47b846...), 1,536-d class
  token, fp32 with TF32 off (bf16 rejected to match the spatial-cohort embeddings).
- Model: ridge regression from the patient-mean tile embedding (standardised features) to the frozen bulk-RNA
  consensus state score (G18, G11); RidgeCV over 24 penalties logspace(0, 5, 24); patient-disjoint 5-fold
  out-of-fold, 10 repeats, seed 20260824. No survival outcome enters the fit. Because ridge is linear, the same
  weights applied per tile give the tile maps in panel B; their patient mean equals the slide-level readout.
- Frozen score function for the target: rank each gene across patients, z-score, dot with the consensus loading
  restricted to shared genes, standardise.
- Result: held-out Spearman 0.470 (G18, n = 510), R2 0.216; permuted-loading null mean -0.080. Weights sha
  28d378cb....
Detector B (secondary; `detector_B_frozen.json`): ridge from the H-Optimus-0 spot embedding of the 24 HEST ccRCC
samples (73,813 spots) to the frozen spot-level score, patient-disjoint within HEST, alpha 149.2; pooled held-out
Spearman 0.245 (G18). Trained entirely outside TCGA.
Representation ablation (Fig. 6 F, n = 289): identical task with phikon, lunit_dino and the 33-centre morphology
dictionary in place of H-Optimus-0 (`wsAH_hestbench_and_stain/tables/AH_DETECTOR_*`).

## 2. The 127-donor breast atlas (Fig. 4 H; `wsAK_breast_replication/receipt.json`)

- Human Breast Cancer Single Cell Atlas, Global Atlas object, CZ CELLxGENE collection
  9432ae97-4803-4b9f-8f64-2b41e42ad3cb, dataset de5416ef-bbfa-41f3-99d4-01a3554173f5 (h5ad asset
  0fbf6fa3-8463-46ab-bda8-9703848b58db, 5.17 GB, schema 7.1.0). Open access, no account.
- Citation: Chen et al. A highly resolved integrated single-cell atlas of human breast cancers. NAR Genomics and
  Bioinformatics 2026. doi:10.1093/nargab/lqaf217 (bioRxiv 2025.03.13.643025; code github montilab/brca_atlas;
  figshare 10.6084/m9.figshare.28685012.v1). Verify the author list against the journal record.
- Content: 621,200 cells from 138 untreated primary breast tumour donors across 8 constituent studies (Pal 2021,
  Wu 2021 Nat Genet, Bassez 2021, Qian 2020, Tietscher 2023, Wang 2024, Liu 2023, Gao 2021); dissociated
  scRNA-seq, 10x 3' v2/v3 and 5' v1/v2.
- Used: 346,696 cells from 127 donors after the < 200-cell donor floor (11 donors dropped), restricted to the 280
  panel genes (273 direct symbol matches, 7 HGNC renames to reach 280/280); residual space factorised from scratch
  (parallel-analysis rank 19, ridge alpha 10, 17 known programmes conditioned on). Independence: none of the 8
  studies is the Xenium discovery cohort (CHUV Lausanne); leave-one-study-out holds S07 at p = 0.000999 in all 8.

## 3. "Nakajima" versus "Peng et al." (lung cohorts in Fig. 4 H and Fig. 5 G)

Two different lung cohorts were used and the paper must name both consistently:
- "Nakajima" is our internal directory label for the Visium data downloaded from the University of Tokyo
  Human Genome Center repository (kero.hgc.jp, Ad-SpatialAnalysis_2024; FF and FFPE Visium of AIS, MIA and
  invasive LUAD; 30 donors, 53,122 spots in the fig4 manifest; 44 sections / 30 donors in the benchmark). The
  publication is Takano Y, Suzuki J, Nomura K, et al. Spatially resolved gene expression profiling of tumor
  microenvironment reveals key steps of lung adenocarcinoma development. Nat Commun 2024;15:10637.
  doi:10.1038/s41467-024-54671-7. No author named Nakajima; label it "Takano 2024".
- "Peng et al." is GEO GSE307534 (56 Visium CytAssist FFPE sections, 25 patients; normal, AAH, AIS, MIA, LUAD):
  Peng, Sinjab, Dai, et al. Multimodal spatial-omics reveal co-evolution of alveolar progenitors and
  proinflammatory niches in progression of lung precursor lesions (2026; PMID 41202811 per the GEO series record).
- Fig. 4 H quotes lung S08 as supported in both: GSE307534 0.353 (p = 0.000999) and Takano 2024 0.311
  (p = 0.020), shown as separate rows because the WS-AL morphology rung is not independent of GSE307534
  (fig4 QC_REPORT line 34). Fig. 5 G uses Takano 2024 (n = 30 donors) as the independent organisation cohort.
- Fix: wherever a panel label says "Nakajima", replace with "Takano 2024 (kero.hgc.jp)" and cite Peng et al. only
  for GSE307534.

## 4. The 5K panel ablation (`claims_closure_20260824/MASTER_RUN_LOG.md`)

What was done: the lung Xenium discovery cohort GSE311609 carries two panels on different physical sections, the
289-gene panel (lung289) and the Xenium Prime 5K panel (prime5k, 6 sections, 6 donors). Panels were never pooled;
every step was rerun unmodified on the 5K sections as a separate population. Numbers to quote:
- Panel-native molecular discovery (WS-Q): parallel-analysis rank 75 on the 5K panel (76 before the RNG seeding
  fix; exact oracle 74), versus 29 on the 289-gene panel; 5 components pass the cross-half gate at every rank
  under the greedy matcher, 8 under Hungarian matching (lines 1367, 1376-1378, 1488, 1984).
- Tissue vocabulary (WS-R) contributions to the stable tier: breast-280 11 of 14, lung-289 2 of 14, Prime-5K
  5 of 14 (line 1227).
- Organisation screen (WS-W/WS-X): 7 Prime-5K components gated; the scrambled-catalogue control clears 0 of 14
  tests under either null at the gated family size (lines 2212, 2394-2396); per-test false-positive rates
  0.059 (spatial) and 0.088-0.177 (molecular) (lines 2378-2379).
- Caveat recorded in the log: the 123 SUPPORTED cross-view cells on Prime-5K are not a headline (6 donors, no
  gene-scrambled catalogue of its own, 42 of 123 with a negative matched-null median); report excess over the
  null 95th percentile (median +0.093), not raw effect (line 1953-1956).
- Cross-panel pairs: Prime-5K pairs added only +0.062 / +0.070 / +0.017 (line 1270).

## 5. MDE (minimum detectable effect)

Two definitions were used and both must be stated once:
- For rediscovery and transport effects (Fig. 4 H, fig4 manifest "mde" fields): the 95th percentile of the
  matched null (T_null_p95), i.e. the smallest effect that would clear alpha = 0.05 under that null
  (`wsAD_calibration_and_vocabulary/RUN_LOG.md` line 535). Example: GSE210616 S07 0.239 against MDE 0.253.
- For Cox interaction and hazard-ratio tests (Fig. 6 G; `primary_interaction_with_verdicts.csv` column
  MDE_interaction_HR_80pct): exp((z_0.975 + z_0.80) x SE) = exp(2.8 x SE), the interaction HR detectable with 80%
  power at two-sided alpha 0.05 given the fitted standard error. Example: JAVELIN G20 SE 0.114 gives MDE 1.38.
- A "bounded null" is a non-significant result reported together with its MDE (`BRIEFING_ADDENDUM.md` line 41).

## 6. The G20 survival model (Fig. 6 G; `wsD_claim4_treatment_interaction`)

- Data: JAVELIN Renal 101 (Motzer 2020 Nat Med supplementary tables; n = 726, 358 PFS events), primary endpoint
  PFS; CheckMate-025 (Braun 2020 supplementary; n = 248, 189 OS events), secondary, OS.
- Score: frozen consensus loading of each state (G18, G11, G05, G15, G20) applied to the trial's bulk RNA with the
  frozen score function (rank genes across patients, z-score, dot with the loading on shared genes,
  standardise); state entered continuous per SD; no refitting; 24/24 loading provenance checks.
- Model: Cox proportional hazards (lifelines CoxPHFitter), endpoint ~ state + arm + state:arm + covariates;
  covariates age + sex for JAVELIN (the trial's own Supplementary Table 6 adjustment set), age + sex + MSKCC
  intermediate + MSKCC poor for CheckMate-025. Interaction HR per SD of state.
- Multiplicity: BH over the predeclared family of 10 (5 states x 2 cohorts, primary endpoints); seed 20260824;
  2,000 bootstraps for secondary intervals.
- Controls: positive, the trial's own 23 co-expression modules rescored and remodelled (27 matched; Pearson 0.82
  between recomputed and published log HRs); negative, 1,000 permutations of the arm indicator with the full
  model refit (Wald SE matches permutation SD within 3%; G20 empirical p = 0.002). Proportional hazards:
  Schoenfeld test on every interaction model (p 0.60-0.69). Robust to winsorising the score.
- Result: G20 interaction HR 0.668 (0.534-0.835), p = 4.1e-4, BH q = 0.0041; G18 0.765 (0.617-0.949), q = 0.073
  (nominal only); all CheckMate-025 tests bounded nulls (MDE HR 1.54-1.59).
- Qualification already recorded: G20 correlates at Spearman -0.85 with the trial's own Organic-acid-metabolic
  module (published interaction HR 2.31); with that module and its arm interaction in the model, G20 falls to
  0.865 (0.588-1.273), p = 0.46. The paper should say G20 is an independent recovery of a predictive axis the
  trial reported, not a new predictive biomarker.
