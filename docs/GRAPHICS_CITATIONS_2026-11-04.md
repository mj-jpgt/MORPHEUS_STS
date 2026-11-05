# Graphic citations for the MORPHEUS research report (2026-11-04)

Every graphic listed below was created by the authors in 2026. Each table row gives the file, the citation line to place beneath the graphic, and the underlying data whose source must also be named in that line. APA-style source citations are collected in section 4 and may be pasted verbatim.

## 1. Standard attribution lines

- **Author-created plot or diagram (no external image inside):**
  Note. Created by the authors (Abejide, 2026) using Python (matplotlib, NumPy, SciPy). Unpublished.
- **Author-created overlay on a tissue image:**
  Note. Created by the authors (Abejide, 2026) using Python (matplotlib, NumPy, SciPy, Pillow, tifffile). H&E image adapted from [source], distributed in HEST-1k (Jaume et al., 2024). Unpublished.
- **Author-created table:**
  Note. Compiled by the authors (Abejide, 2026) from the MORPHEUS run receipts (Hugging Face dataset mj0jpgg/MORPHEUS). Unpublished.

Software versions for graphics rendered on the author's workstation: Python 3.13.5, matplotlib 3.10.8, NumPy 2.4.3, SciPy 1.16.0, Pillow 12.3.0. Graphics rendered on the GPU box used Python 3.10 with the same libraries (versions in the box environment file archived with the dataset). None of these graphics has appeared in a prior publication, preprint, poster or thesis, so no self-citation is needed. If the August 2026 classical-method manuscript is cited for any reused panel, add its full reference after the author line.

## 2. Figure 6 evidence cards (figures/fig6_evidence_cards/)

| file(s) | citation line beneath the graphic | underlying data to name |
|---|---|---|
| card_G*/a_spatial, a_spatial_insample_v01 | Created by the authors (Abejide, 2026) using Python (matplotlib). Spot coordinates and expression from Meylan et al. (2022) via HEST-1k (Jaume et al., 2024). | Meylan 2022; HEST-1k |
| card_G*/b_response | Created by the authors (Abejide, 2026) using Python (matplotlib). Expression data from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G*/c_ladder | Created by the authors (Abejide, 2026) using Python (matplotlib). Baselines use MSigDB Hallmark gene sets (Liberzon et al., 2015) and Gavish et al. (2023) meta-programmes. | Meylan 2022; Hallmark; Gavish 2023 |
| card_G*/d_external | Created by the authors (Abejide, 2026) using Python (matplotlib). External cohorts: Li et al. (2022), Bischoff et al. (2025, GEO GSE250163), Dawo et al. (2025, Zenodo 14620362). | Li 2022; GSE250163; Zenodo 14620362 |
| card_G*/e_umap | Created by the authors (Abejide, 2026) using Python (matplotlib, umap-learn; McInnes et al., 2018). Morphology embeddings from H-Optimus-0 (Saillard et al., 2024) on Meylan et al. (2022) H&E via HEST-1k. | H-Optimus-0; Meylan 2022; HEST-1k; UMAP |
| card_G*/f_crops | Created by the authors (Abejide, 2026) using Python (Pillow, matplotlib). H&E crops adapted from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G*/g_he_tissue, g_he_state, g_he_target | Created by the authors (Abejide, 2026) using Python (matplotlib, Pillow, tifffile). H&E whole-slide image of section INT21 adapted from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G*/h_residual, i_constituents | Created by the authors (Abejide, 2026) using Python (matplotlib). H&E underlay adapted from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G*/j_composition | Created by the authors (Abejide, 2026) using Python (matplotlib). Nuclear composition from CellViT segmentations (Horst et al., 2024) shipped with HEST-1k; H&E from Meylan et al. (2022). | CellViT; Meylan 2022; HEST-1k |
| card_G*/k_transfer | Created by the authors (Abejide, 2026) using Python (matplotlib). Transfer cohorts: Valdeolivas et al. (2024) and Fu et al. (2023), both via HEST-1k. | Valdeolivas 2024; Fu 2023; HEST-1k |
| pair_G3_G4_state, pair_G5_G7_state | Created by the authors (Abejide, 2026) using Python (matplotlib). Data from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G3/closeup/l_context, l_colourbar, m_closeup, n_closeup_spots | Created by the authors (Abejide, 2026) using Python (matplotlib, Pillow, tifffile). H&E whole-slide image of section INT21 adapted from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G3/crosscancer/o_kidney_he, o_kidney_target, o_kidney_score | Created by the authors (Abejide, 2026) using Python (matplotlib, Pillow, tifffile). H&E of section INT21 adapted from Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |
| card_G3/crosscancer/o_colorectal_he, o_colorectal_target, o_colorectal_score, p_colorectal_closeup, q_colorectal_closeup_spots | Created by the authors (Abejide, 2026) using Python (matplotlib, Pillow, tifffile). H&E of section ZEN42 adapted from Valdeolivas et al. (2024) via HEST-1k. | Valdeolivas 2024; HEST-1k |
| card_G3/crosscancer/o_colourbar, G3_crosscancer_preview | Created by the authors (Abejide, 2026) using Python (matplotlib). Preview sheet combines the panels above; cite the same sources. | as above |

## 3. Figure 7 benchmark (figures/fig7_benchmark/)

| file(s) | citation line beneath the graphic | underlying data to name |
|---|---|---|
| a_ap_singles, b_ap_pairs, c_diff_pairs, d_diff_singles, e_pooled | Created by the authors (Abejide, 2026) using Python (matplotlib). Test studies: Greenwald et al. (2024, GEO GSE237183), Takano et al. (2024), Peng et al. (2026, GEO GSE307534). Baselines implemented with interpret (EBM), LightGBM, scikit-learn, PySR (Cranmer, 2023). | GSE237183; Takano 2024; GSE307534 |
| f_operator_ablation | Created by the authors (Abejide, 2026) using Python (matplotlib). Counts from the registered renal catalogue on Meylan et al. (2022) via HEST-1k. | Meylan 2022; HEST-1k |

Results tables (BENCHMARK_V05_TABLES_FINAL, RESULTS_SINCE_BENCHMARK_V05, evidence table, data-source table): Compiled by the authors (Abejide, 2026) from the MORPHEUS run receipts (Hugging Face dataset mj0jpgg/MORPHEUS; code repository on GitHub, URL to be added). Unpublished.

## 4. Source citations (APA style) for the data and tools inside the graphics

- Bischoff, M. E., ... Czyzyk-Krzeska, M. F. (2025). Copper drives remodeling of metabolic state and progression of clear cell renal cell carcinoma. *Cancer Discovery, 15*, 401-426. Data: GEO GSE250163.
- Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. *arXiv:2305.01582*.
- Dawo, S., Nonchev, K., & Silina, K. (2025). *10x Visium spatial transcriptomics dataset: Kidney (3) and lung (5) cancer with tertiary lymphoid structures* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14620362
- Fu, R., Norris, G. A., Willard, N., et al. (2023). Spatial transcriptomic analysis delineates epithelial and mesenchymal subpopulations and transition stages in childhood ependymoma. *Neuro-Oncology, 25*, 786-798. https://doi.org/10.1093/neuonc/noac219
- Gavish, A., Tyler, M., Greenwald, A. C., et al. (2023). Hallmarks of transcriptional intratumour heterogeneity across a thousand tumours. *Nature, 618*, 598-606. https://doi.org/10.1038/s41586-023-06130-4
- Greenwald, A. C., Darnell, N. G., Hoefflin, R., et al. (2024). Integrative spatial analysis reveals a multi-layered organization of glioblastoma. *Cell, 187*, 2485-2501. https://doi.org/10.1016/j.cell.2024.03.029. Data: GEO GSE237183; Zenodo 12624860.
- Horst, F., Rempe, M., Heine, L., et al. (2024). CellViT: Vision transformers for precise cell segmentation and classification. *Medical Image Analysis, 94*, 103143. https://doi.org/10.1016/j.media.2024.103143
- Jaume, G., Doucet, P., Song, A. H., et al. (2024). HEST-1k: A dataset for spatial transcriptomics and histology image analysis. *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track*. arXiv:2406.16192. Data: https://huggingface.co/datasets/MahmoodLab/hest
- Li, R., Ferdinand, J. R., Loudon, K. W., et al. (2022). Mapping single-cell transcriptomes in the intra-tumoral and associated territories of kidney cancer. *Cancer Cell, 40*, 1583-1599. https://doi.org/10.1016/j.ccell.2022.11.001. Data: Mendeley Data, https://doi.org/10.17632/g67bkbnhhg.1
- Liberzon, A., Birger, C., Thorvaldsdottir, H., Ghandi, M., Mesirov, J. P., & Tamayo, P. (2015). The Molecular Signatures Database hallmark gene set collection. *Cell Systems, 1*, 417-425. https://doi.org/10.1016/j.cels.2015.12.004
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*.
- Meylan, M., Petitprez, F., Becht, E., et al. (2022). Tertiary lymphoid structures generate and propagate anti-tumor antibody-producing plasma cells in renal cell cancer. *Immunity, 55*, 527-541. https://doi.org/10.1016/j.immuni.2022.02.001. Sections INT1-INT24 distributed in HEST-1k (Jaume et al., 2024).
- Peng, ..., Sinjab, A., Dai, E., et al. (2026). Multimodal spatial-omics reveal co-evolution of alveolar progenitors and proinflammatory niches in progression of lung precursor lesions. PMID 41202811. Data: GEO GSE307534. (Verify journal and author list before submission.)
- Saillard, C., Jenatton, R., Llinares-Lopez, F., et al. (2024). *H-optimus-0* [Model]. Bioptimus. https://huggingface.co/bioptimus/H-optimus-0
- Takano, Y., Suzuki, J., Nomura, K., et al. (2024). Spatially resolved gene expression profiling of tumor microenvironment reveals key steps of lung adenocarcinoma development. *Nature Communications, 15*, 10637. https://doi.org/10.1038/s41467-024-54671-7
- Valdeolivas, A., Amberg, B., Giroud, N., et al. (2024). Profiling the heterogeneity of colorectal cancer consensus molecular subtypes using spatial transcriptomics. *npj Precision Oncology, 8*, 10. https://doi.org/10.1038/s41698-023-00488-4. Sections ZEN36-ZEN49 distributed in HEST-1k (Jaume et al., 2024).

Items to verify before submission: the Peng et al. (2026) journal and author list; the Meylan et al. (2022) page range; the CellViT checkpoint used for the HEST-shipped segmentations.
