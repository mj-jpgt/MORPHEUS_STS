# Results obtained 2026-11-03 (benchmark v05 close-out and after)

All numbers are from the final files in `~/morpheus_work_bench/runs/benchmark_v05/` (commit b646f85), `~/morpheus_work_tx/runs/development/translation_v10/` and `~/morpheus_work_v06/runs/confirmation/evidence_table_v01/`. Test studies: glioma GSE237183, LUAD Takano 2024 (directory label nakajima_luad_b3), lung precursor GSE307534; all untouched during development. Labels: a candidate is positive when its RFF ridge relation, fitted on support patients over the known-biology baseline, has a patient-bootstrap 95% lower bound of held-out gain above 0 and a gain above the 95th percentile of the patient-permuted null. AP = average precision; intervals are hierarchical bootstraps (studies, then targets) with 2,000 replicates; paired differences are ours minus method on shared targets and splits; Holm across methods per endpoint.

## T1. Benchmark, pooled over the three test studies (Table 1 pooled)

| method | targets (gli/LUAD/lung) | AP singles [95% CI] | ours minus method, singles | AP pairs [95% CI] | ours minus method, pairs | seconds per fit (median) |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 40/37/40 | 0.306 [0.240, 0.381] | - | 0.347 [0.249, 0.459] | - | 1.3-4.2 |
| ours: joint pair gain (supplementary) | 40/37/40 | 0.306 | 0.000 | 0.381 [0.290, 0.489] | -0.034 [-0.067, -0.011] | 2.5-8.1 |
| hybrid: linear singles + support-fit pairs (suppl.) | 40/37/40 | 0.304 [0.244, 0.371] | +0.003 [-0.060, +0.059] | 0.347 | 0.000 | - |
| random (chance, 20 seeds) | 40/37/40 | 0.028 [0.017, 0.043] | +0.279 [+0.217, +0.348] | 0.036 [0.015, 0.067] | +0.313 [+0.231, +0.405] | 0.2-0.7 |
| linear interaction screen | 40/37/40 | 0.304 [0.244, 0.371] | +0.003 [-0.060, +0.059] | 0.045 [0.019, 0.078] | +0.303 [+0.224, +0.392] | 1.8-2.9 |
| NID | 36/37/40 | 0.318 [0.246, 0.404] | -0.003 [-0.076, +0.058] | 0.166 [0.116, 0.221] | +0.205 [+0.124, +0.288] | 3.5-5.9 |
| EBM (native) | 36/29/37 | 0.359 [0.268, 0.454] | -0.018 [-0.073, +0.022] | 0.340 [0.258, 0.425] | +0.042 [-0.033, +0.100], Holm p 0.60 | 141-215 |
| GBM + SHAP interactions | 36/30/37 | 0.336 [0.259, 0.418] | -0.005 [-0.068, +0.046] | 0.065 [0.036, 0.105] | +0.312 [+0.241, +0.384] | 26-94 |
| RuleFit | 33/29/36 | 0.326 [0.241, 0.420] | +0.032 [-0.021, +0.080] | 0.050 [0.023, 0.092] | +0.342 [+0.265, +0.422] | 156-186 |
| iRF | 36/29/37 | 0.297 [0.225, 0.374] | +0.032 [-0.015, +0.065] | 0.048 [0.028, 0.072] | +0.328 [+0.250, +0.413] | 178-263 |
| Model-X knockoffs | 36/31/37 | 0.250 [0.175, 0.327] | +0.088 [+0.011, +0.158], Holm p 0.16 | 0.054 [0.028, 0.093] | +0.325 [+0.252, +0.399] | 6-9 |
| MISTy-style forest | 40/37/40 | 0.140 [0.095, 0.196] | +0.167 [+0.125, +0.213] | 0.101 [0.057, 0.162] | +0.247 [+0.181, +0.320] | 6-7 |
| GBM + Friedman H | 36/27/37 | 0.031 [0.017, 0.051] | +0.275 [+0.215, +0.335] | 0.052 [0.031, 0.085] | +0.305 [+0.242, +0.370] | 156-213 |
| PySR (30-variable shortlist) | 9/10/10 | 0.212 [0.114, 0.348] (ours on same targets 0.289) | - | 0.053 [0.007, 0.138] | +0.266 [+0.120, +0.421] | - |
| stability selection (incomplete: 143 of 307 fits timed out) | 21/18/17 | 0.137 [0.098, 0.178] | +0.178 [+0.121, +0.247] | 0.068 [0.026, 0.120] | +0.328 [+0.197, +0.457] | 247-493 |
| EBM (fair inputs, 10 targets) | 9/10/10 | 0.269 [0.110, 0.443] | +0.022 [-0.091, +0.100] | 0.083 [0.016, 0.162] | +0.235 [+0.081, +0.406] | 60-68 |
| GBM (fair inputs, 10 targets) | 9/10/10 | 0.230 [0.100, 0.363] | +0.062 [-0.030, +0.127] | 0.099 [0.035, 0.163] | +0.220 [+0.078, +0.368] | 33-51 |
| MISTy (fair inputs, 10 targets) | 9/10/10 | 0.144 [0.043, 0.251] | +0.145 [+0.075, +0.253] | 0.075 [0.009, 0.192] | +0.245 [+0.111, +0.410] | 102-164 |
| TabDistill | - | not completed (all 90 fits over the 300 s budget; repair attempt ran 34 min without finishing) | | | | |
| fair iRF, fair RuleFit | - | not completed (timed out almost everywhere) | | | | |

Prevalence (singles / pairs): glioma 1.0% / 1.0%, LUAD 3.4% / 4.9%, lung precursor 1.6% / 1.8%. Pooled chance AP 0.028 / 0.036. Primary pair score (mean of singles) pre-declared in freeze addendum 3 (2026-10-27).

## T2. Benchmark, per study: ours against the three nearest baselines

| study | ours singles / pairs | EBM singles / pairs (ours minus EBM) | NID singles / pairs (ours minus NID) | linear screen singles / pairs (ours minus linear) |
|---|---|---|---|---|
| glioma | 0.274 / 0.270 | 0.374 / 0.368 (-0.075 [-0.132, -0.026] / -0.027 [-0.120, +0.059]) | 0.380 / 0.208 (-0.081 / +0.134) | 0.339 / 0.018 (-0.064 / +0.252) |
| LUAD Takano | 0.365 / 0.457 | 0.424 / 0.393 (+0.015 [-0.020, +0.049] / +0.081 [+0.014, +0.159], Holm p 0.044) | 0.305 / 0.179 (+0.060 / +0.280) | 0.304 / 0.077 (+0.061 / +0.383) |
| lung precursor | 0.278 / 0.314 | 0.278 / 0.258 (+0.005 [-0.006, +0.017] / +0.069 [+0.021, +0.122], Holm p 0.004) | 0.267 / 0.112 (+0.011 / +0.202) | 0.268 / 0.040 (+0.010 / +0.274) |

EBM AUROC on pairs is higher than ours (0.84 to 0.91 vs 0.59 to 0.70): EBM orders the whole list better, ours orders the top better. Top-10 overlap of singles with ours: NID 0.15 / 0.22 / 0.20, linear 0.15 / 0.19 / 0.18, EBM 0.16 / 0.27 / 0.24.

## T3. Transfer without refit, Meylan ccRCC to three external ccRCC cohorts (B2 / MC1; Li 2022, GSE250163, Zenodo; 20 programme targets each, pooled)

| method | AP singles [95% CI] | ours minus method, singles | AP pairs [95% CI] | ours minus method, pairs | P@10 singles / pairs |
|---|---|---|---|---|---|
| ours: support-fit ordering | 0.228 [0.166, 0.301] | - | 0.238 [0.169, 0.316] | - | 0.302 / 0.418 |
| random | 0.074 [0.054, 0.097] | +0.153 [+0.090, +0.230] | 0.085 [0.057, 0.117] | +0.153 [+0.087, +0.229] | 0.070 / 0.082 |
| linear interaction screen | 0.286 [0.199, 0.390] | -0.057 [-0.131, -0.009] | 0.089 [0.061, 0.122] | +0.149 [+0.082, +0.225] | 0.343 / 0.088 |
| EBM | 0.267 [0.195, 0.357] | -0.038 [-0.100, +0.003] | 0.212 [0.155, 0.272] | +0.026 [-0.011, +0.071] | 0.291 / 0.362 |
| NID | 0.209 [0.152, 0.277] | +0.019 [-0.014, +0.049] | 0.187 [0.135, 0.246] | +0.050 [+0.017, +0.090] | 0.274 / 0.352 |
| MISTy-style forest | 0.242 [0.174, 0.324] | -0.014 [-0.087, +0.055] | 0.168 [0.123, 0.213] | +0.070 [+0.020, +0.127] | 0.274 / 0.317 |
| GBM + SHAP | 0.236 [0.154, 0.342] | -0.008 [-0.082, +0.047] | 0.109 [0.080, 0.143] | +0.129 [+0.073, +0.197] | 0.218 / 0.175 |
| RuleFit | 0.147 [0.099, 0.214] | +0.086 [+0.048, +0.129] | 0.088 [0.059, 0.121] | +0.154 [+0.091, +0.231] | 0.156 / 0.094 |
| Model-X knockoffs | 0.153 [0.116, 0.196] | +0.075 [+0.041, +0.116] | 0.090 [0.062, 0.122] | +0.147 [+0.082, +0.221] | 0.193 / 0.124 |
| GBM + Friedman H | 0.074 [0.054, 0.097] | +0.153 [+0.090, +0.230] | 0.096 [0.069, 0.128] | +0.142 [+0.080, +0.213] | 0.080 / 0.103 |
| iRF (incomplete: 19 of 60 fits timed out) | 0.169 [0.122, 0.213] | +0.052 [-0.015, +0.125] | 0.088 [0.057, 0.121] | +0.141 [+0.050, +0.228] | 0.204 / 0.091 |

Prevalence 5.4% singles, 6.6% pairs. Stability selection: no receipts (all fits timed out).

## T4. Calibration of the predicted held-out gain (release checkpoint 13c36faa)

| study | kind | slope of realised on predicted [95% CI] | Spearman | Kendall tau | ECE pass probability | ECE constant prevalence |
|---|---|---|---|---|---|---|
| glioma | single | 0.42 [0.21, 0.72] | 0.24 | 0.16 | 0.0033 | 0.0079 |
| glioma | pair | 0.46 [0.14, 0.69] | 0.22 | 0.16 | 0.0073 | 0.0100 |
| LUAD Takano | single | 0.89 [0.75, 0.96] | 0.20 | 0.14 | 0.0205 | 0.0400 |
| LUAD Takano | pair | 1.36 [1.16, 1.44] | 0.14 | 0.10 | 0.0336 | 0.0571 |
| lung precursor | single | 0.09 [0.02, 0.21] | 0.15 | 0.11 | 0.0075 | 0.0161 |
| lung precursor | pair | -0.04 [-0.32, 0.12] | 0.14 | 0.09 | 0.0083 | 0.0234 |

Reading: ranking is informative everywhere; the magnitude is not calibrated across cohorts. Gains winsorised at the 0.1 and 99.9 percentiles (a few ill-conditioned fits reach |G| about 1e5).

## T5. False-discovery rate of the top-10 picks under two nulls

| study | kind | real precision@10 | FDR@10, permuted target | FDR@10, scrambled candidates | net precision (permuted / scrambled) |
|---|---|---|---|---|---|
| glioma | single | 0.107 | 0.012 | 0.009 | 0.095 / 0.098 |
| glioma | pair | 0.200 | 0.012 | 0.005 | 0.188 / 0.195 |
| LUAD Takano | single | 0.293 | 0.024 | 0.004 | 0.269 / 0.289 |
| LUAD Takano | pair | 0.372 | 0.024 | 0.003 | 0.348 / 0.369 |
| lung precursor | single | 0.128 | 0.020 | 0.013 | 0.108 / 0.115 |
| lung precursor | pair | 0.266 | 0.022 | 0.005 | 0.243 / 0.261 |

## T6. Fold stability (10 patient-grouped splits per target): top-10 Jaccard between splits

| study | ours singles / pairs | linear screen singles / pairs | NID singles / pairs | chance |
|---|---|---|---|---|
| glioma | 0.127 / 0.196 | 0.425 / 0.117 | 0.243 / 0.121 | 0.0035 |
| LUAD Takano | 0.426 / 0.456 | 0.663 / 0.191 | 0.545 / 0.211 | 0.0031 |
| lung precursor | 0.183 / 0.288 | 0.542 / 0.146 | 0.392 / 0.116 | 0.0034 |

## T7. Operator leave-out on the registered renal catalogue (200 frozen candidates)

| removed | internally supported | beyond known biology | externally replicated |
|---|---|---|---|
| nothing (full catalogue) | 161 | 28 | 10 |
| compartment context | 111 | 12 | 10 |
| heterogeneity / gradient | 150 | 17 | 1 |
| residual-programme targets | 111 | 28 | 10 |
| ring decay | 111 | 27 | 9 |

0 of the 200 frozen candidates carry an H-Optimus or morphology-dictionary variable; compartment context uses deconvolved compartments, not H&E.

## T8. Learned beam ablations (E1, E2) and budget sweep

E1 matched budget, singles (positives among the beam's 26 labelled proposals vs the top 26 of exhaustive support-fit ordering):

| study | beam positives (precision) | exhaustive top-26 positives (precision) | beam minus exhaustive |
|---|---|---|---|
| glioma | 0.99 (0.038) | 1.48 (0.057) | -0.019 [-0.030, -0.008] |
| LUAD Takano | 5.19 (0.200) | 6.35 (0.244) | -0.044 [-0.073, -0.023] |
| lung precursor | 1.34 (0.052) | 2.08 (0.080) | -0.028 [-0.048, -0.013] |

Pool AP, beam (64 scored sets, rest tied) minus exhaustive: glioma -0.067 [-0.142, -0.004] singles, -0.244 [-0.345, -0.150] pairs; lung precursor -0.050 [-0.071, -0.030], -0.290 [-0.401, -0.186]. No matched pair comparison exists (the beam's pairs almost never fall in the 2,000-pair label pool).

E2, pair score on known-biology labels (joint gain vs mean of singles): glioma +0.063 [+0.027, +0.107]; LUAD +0.023 [+0.010, +0.038]; lung precursor +0.016 [-0.008, +0.035]. E2 on the higher-order (decomposition) label: every ranking at chance on all three studies (prevalence 1.4% / 2.0% / 1.5%; 28 to 40 positive pairs per 2,000).

Budget sweep (8 targets, splits 0/3/6; confirmed labelled sets per target, a lower bound since only 0.41 of K sets carry a label):

| study / selector | K=16 | K=64 | K=256 |
|---|---|---|---|
| LUAD, learned beam | 4.1 | 15.7 [11.8, 19.5] | 29.2 [23.8, 34.3] |
| LUAD, marginal-screen beam | 4.5 | 17.2 [13.3, 21.2] | 50.5 [40.1, 59.7] |
| LUAD, random beam | 0.7 | 2.5 | 11.4 |
| glioma, learned / screen / random | 0.54 / 0.83 / 0.04 | 0.58 / 1.29 / 0.04 | 0.79 / 2.33 / 0.50 |
| lung precursor, learned / screen / random | 0.21 / 0.50 / 0.00 | 0.25 / 0.83 / 0.25 | 0.33 / 1.75 / 0.58 |

## T9. Few-shot (B3), LUAD Takano, 8 targets, 5 draws, same 3 query patients per draw

| method | n=2 | n=3 | n=5 | n=8 | all patients |
|---|---|---|---|---|---|
| ours, singles | 0.199 [0.159, 0.239] | 0.230 | 0.269 | 0.293 [0.242, 0.346] | 0.346 |
| ours, pairs | 0.222 | 0.274 | 0.322 | 0.366 | 0.418 |
| linear screen, singles | 0.205 | 0.206 | 0.230 | 0.246 | 0.269 |
| MISTy-style forest, singles | 0.213 | 0.235 | 0.254 | 0.265 | 0.280 |
| random, singles | 0.058 | 0.061 | 0.067 | 0.078 | 0.096 |

All other methods are below ours on pairs at every n (NID 0.196, MISTy 0.248 at n=8). Glioma and lung precursor curves have CIs of +/-0.15 or wider and are not ordered. Slow baselines were not run on this grid.

## T10. Historical rediscovery (B5): fails its chance check under the locked hit rule

| method | rank of first hit, v02 rule (H1 cSCC / H2 CRC / H3 prostate) | repaired query-validated scorer (H2 only is scoreable) |
|---|---|---|
| random (count-matched) | 4 / 6 / 6 | none |
| ours: support-fit ordering | none / none / none | none |
| learned beam | none / 4 / none | 3 |
| RuleFit | none / none / none | 2 |
| PySR | - | 4 |
| linear interaction screen | none / 5 / 11 | 5 |
| EBM, GBM + Friedman H | none | none |

H1 is not primary-eligible (proxies not source-exact; 2 query patients); H3 pending. Report as a sanity exercise or drop.

## T11. Annotation recovery on external ccRCC slides (B6; Zenodo, 3 patients, 3,760 annotated spots, k-means k=4)

| feature set | ARI [95% CI] | NMI | purity |
|---|---|---|---|
| frozen kidney relations | 0.074 [0.071, 0.145] | 0.100 | 0.633 |
| H-Optimus-0 embedding | 0.088 [0.003, 0.281] | 0.096 | 0.633 |
| raw expression | 0.018 [0.018, 0.408] | 0.041 | 0.559 |
| CellViT composition | 0.027 [0.024, 0.085] | 0.042 | 0.589 |
| random | 0.000 | 0.001 | 0.534 |

## T12. H&E-only student (B7 / MC6; adopted method only)

| cohort | model | median Spearman, held-out patients | relations at Spearman >= 0.4 (of 45) |
|---|---|---|---|
| Meylan (held out) | ridge on H-Optimus-0 | 0.19 | 13% |
| Meylan (held out) | small MLP | 0.14 | 7% |
| Zenodo (external) | ridge | 0.05 | 0% |

## T13. Higher-order specificity on the Meylan blind cohort (B4, fold-level decomposition pass)

| arm | k | precision@k | null FDR | net precision |
|---|---|---|---|---|
| learned search | 10 | 0.048 [0.034, 0.061] | 0.036 [0.025, 0.049] | 0.011 [-0.009, 0.031] |
| learned search | 20 | 0.072 [0.060, 0.084] | 0.038 [0.026, 0.051] | 0.034 [0.017, 0.051] |
| support-screen search | 10 | 0.025 [0.017, 0.033] | 0.059 [0.048, 0.071] | -0.035 [-0.049, -0.022] |
| support-screen search | 20 | 0.052 [0.042, 0.065] | 0.064 [0.056, 0.074] | -0.012 [-0.027, 0.001] |
| joint-screen search | 10 | 0.032 [0.022, 0.042] | 0.057 [0.048, 0.066] | -0.025 [-0.037, -0.013] |

## T14. Cross-cancer transfer over the known-biology baseline (P5 addendum D20; frozen relation, no refit)

| relation | cohort | gain over nuisance (P5) | gain over known biology [95% CI] | matched-null p95 | transfers over known biology |
|---|---|---|---|---|---|
| G3 | colorectal (Valdeolivas 2024, 6 patients) | 0.300 | 0.261 [0.194, 0.341] | -0.116 | yes |
| G4 | colorectal | 0.044 | 0.009 [-0.209, 0.249] | -0.143 | no |
| het-934c62f7ec26 | colorectal | 0.366 | 0.318 [0.156, 0.537] | -0.185 | yes |
| G3 | ependymoma PFA (11 patients) | 0.084 | 0.050 [-0.018, 0.114] | -0.042 | no |
| G4 | ependymoma PFA | 0.126 | 0.084 [-0.044, 0.160] | -0.139 | no |
| het-ec7174547e19 | ependymoma PFA | 0.174 | 0.062 [-0.078, 0.119] | -0.092 | no |

G5 and G7 fail in both cohorts over nuisance alone. GSE294385 and Ji 2020 are not testable (discovery genes absent from the panels). The "conserved" flag in the older P5 matrix is superseded.

## T15. Evidence table after the rebuild (runs/confirmation/evidence_table_v01, 2026-11-03)

| quantity | value |
|---|---|
| rows / relations / records | 6,229 / 235 / 71 |
| validation errors | 0 |
| records by tier | 11 externally confirmed, 20 internally supported, 40 candidates |
| graded v08 records by stage | 8 replicated, 5 internally supported, 24 proposed only |
| replicated records, median % of nuisance-only gain retained | 98% beyond known biology, 93% beyond composition, 92% beyond both |
| new layer | translation_p5_known, 6 rows (T14) |

## T16. Engineering results (why the evaluations are now fast)

| item | before | after |
|---|---|---|
| label engine, seconds per split | 580 (v04) | 15 (v05; shared Gram / Cholesky; labels identical except 0.13% of nuisance-only singles, no known-biology label affected) |
| fits in flight / threads per fit / budget | 2 / 3 / 240 s | 10 / 2 / 480 s |
| timeouts per slow method | 24 to 45 | 0 to 13 (stability selection excepted) |
| Friedman H | per-pair loop, 102 s | batched, 83 s, bit-identical |
| RuleFit | at chance (extractor read a non-existent attribute, scored all 0) | fixed, known-answer tested; glioma singles AP 0.393 |
| GSE307534 loader | OOM-killed twice | streaming loader, peak 107 GB then 6 GB |
