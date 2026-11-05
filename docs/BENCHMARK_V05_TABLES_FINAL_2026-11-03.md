# Benchmark v05 FINAL tables, regenerated with PySR rows (box 2026-11-03 late; files in ~/morpheus_work_bench/runs/benchmark_v05/)

Labels: label engine v05, pass rule at 95%, baseline B_known. Primary pair score = mean of the two single support gains, declared in freeze addendum 3 (2026-10-27). Incomplete: stability selection, fair iRF, fair RuleFit. TabDistill: all 90 fits timed out at 300 s, recorded as a timeout row (one repair attempt running). B5 historical rediscovery fails its chance check under hit rule v02; the repaired query-validated scorer is the informative variant. E2 on the decomposition label is at chance on all three studies.

## table1_pooled_3studies

### Table 1: studies gse237183_glioma, nakajima_luad_b3, gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0201, pairs 0.0256
Per-study prevalence: {"single": {"gse237183_glioma": 0.01, "nakajima_luad_b3": 0.0343, "gse307534_lung_precursor_b3": 0.016}, "pair": {"gse237183_glioma": 0.0098, "nakajima_luad_b3": 0.0489, "gse307534_lung_precursor_b3": 0.0183}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 40/37/40 | 0.306 [0.240, 0.381] | 15.1 / 15.1 | 0.304 | - | 0.347 [0.249, 0.459] | 13.2 / 13.2 | 0.339 | - | 0.194 / 0.332 |
| ours: support-fit ordering, joint pair gain (suppl.) | 40/37/40 | 0.306 [0.240, 0.381] | 15.1 / 15.1 | 0.304 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.381 [0.290, 0.489] | 14.6 / 13.2 | 0.339 | -0.034 [-0.067, -0.011], p_Holm 1.000 | 0.194 / 0.353 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 40/37/40 | 0.304 [0.244, 0.371] | 15.1 / 15.1 | 0.304 | +0.003 [-0.060, +0.059], p_Holm 1.000 | 0.347 [0.249, 0.459] | 13.2 / 13.2 | 0.339 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.173 / 0.332 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 40/37/40 | 0.304 [0.244, 0.371] | 15.1 / 15.1 | 0.304 | +0.003 [-0.060, +0.059], p_Holm 1.000 | 0.381 [0.290, 0.489] | 14.6 / 13.2 | 0.339 | -0.034 [-0.067, -0.011], p_Holm 1.000 | 0.173 / 0.353 |
| random (chance reference, 20 seeds) | 40/37/40 | 0.028 [0.017, 0.043] | 1.4 / 15.1 | 0.304 | +0.279 [+0.217, +0.348], p_Holm 0.000 | 0.036 [0.015, 0.067] | 1.3 / 13.2 | 0.339 | +0.313 [+0.231, +0.405], p_Holm 0.000 | 0.024 / 0.033 |
| linear interaction screen | 40/37/40 | 0.304 [0.244, 0.371] | 15.1 / 15.1 | 0.304 | +0.003 [-0.060, +0.059], p_Holm 1.000 | 0.045 [0.019, 0.078] | 1.7 / 13.2 | 0.339 | +0.303 [+0.224, +0.392], p_Holm 0.000 | 0.173 / 0.060 |
| NID | 36/37/40 | 0.318 [0.246, 0.404] | 15.6 / 15.5 | 0.313 | -0.003 [-0.076, +0.058], p_Holm 1.000 | 0.166 [0.116, 0.221] | 6.3 / 14.1 | 0.365 | +0.205 [+0.124, +0.288], p_Holm 0.000 | 0.176 / 0.194 |
| MISTy-style forest (pairs derived) | 40/37/40 | 0.140 [0.095, 0.196] | 6.9 / 15.1 | 0.304 | +0.167 [+0.125, +0.213], p_Holm 0.000 | 0.101 [0.057, 0.162] | 3.7 / 13.2 | 0.339 | +0.247 [+0.181, +0.320], p_Holm 0.000 | 0.146 / 0.164 |
| Model-X knockoffs | 36/31/37 | 0.250 [0.175, 0.327] | 12.0 / 16.1 | 0.333 | +0.088 [+0.011, +0.158], p_Holm 0.175 | 0.054 [0.028, 0.093] | 2.0 / 14.2 | 0.372 | +0.325 [+0.252, +0.399], p_Holm 0.000 | 0.169 / 0.093 |
| GBM + SHAP interactions (native) | 36/30/37 | 0.336 [0.259, 0.418] | 16.1 / 15.8 | 0.326 | -0.005 [-0.068, +0.046], p_Holm 1.000 | 0.065 [0.036, 0.105] | 2.4 / 14.2 | 0.370 | +0.312 [+0.241, +0.384], p_Holm 0.000 | 0.211 / 0.088 |
| RuleFit (native) | 33/29/36 | 0.326 [0.241, 0.420] | 14.6 / 16.0 | 0.352 | +0.032 [-0.021, +0.080], p_Holm 1.000 | 0.050 [0.023, 0.092] | 1.7 / 13.8 | 0.384 | +0.342 [+0.265, +0.422], p_Holm 0.000 | 0.185 / 0.040 |
| stability selection **(incomplete: 143 of 307 target x split fits timed out)** | 21/18/17 | 0.137 [0.098, 0.178] | 4.7 / 10.7 | 0.314 | +0.178 [+0.121, +0.247], p_Holm 0.000 | 0.068 [0.026, 0.120] | 1.7 / 10.2 | 0.398 | +0.328 [+0.197, +0.457], p_Holm 0.000 | 0.174 / 0.090 |
| GBM + Friedman H | 36/27/37 | 0.031 [0.017, 0.051] | 1.5 / 15.1 | 0.302 | +0.275 [+0.215, +0.335], p_Holm 0.000 | 0.052 [0.031, 0.085] | 2.0 / 14.3 | 0.352 | +0.305 [+0.242, +0.370], p_Holm 0.000 | 0.030 / 0.060 |
| EBM (native) | 36/29/37 | 0.359 [0.268, 0.454] | 17.7 / 16.7 | 0.333 | -0.018 [-0.073, +0.022], p_Holm 1.000 | 0.340 [0.258, 0.425] | 13.5 / 15.0 | 0.372 | +0.042 [-0.033, +0.100], p_Holm 0.600 | 0.214 / 0.291 |
| iRF (native) | 36/29/37 | 0.297 [0.225, 0.374] | 14.4 / 15.8 | 0.323 | +0.032 [-0.015, +0.065], p_Holm 1.000 | 0.048 [0.028, 0.072] | 1.8 / 14.4 | 0.368 | +0.328 [+0.250, +0.413], p_Holm 0.000 | 0.191 / 0.036 |
| GBM (fair) | 9/10/10 | 0.230 [0.100, 0.363] | 6.2 / 7.9 | 0.289 | +0.062 [-0.030, +0.127], p_Holm 0.968 | 0.099 [0.035, 0.163] | 2.1 / 6.5 | 0.317 | +0.220 [+0.078, +0.368], p_Holm 0.000 | 0.225 / 0.171 |
| MISTy-style forest (fair) | 9/10/10 | 0.144 [0.043, 0.251] | 4.0 / 7.9 | 0.289 | +0.145 [+0.075, +0.253], p_Holm 0.000 | 0.075 [0.009, 0.192] | 1.6 / 6.5 | 0.317 | +0.245 [+0.111, +0.410], p_Holm 0.000 | 0.249 / 0.188 |
| EBM (fair) | 9/10/10 | 0.269 [0.110, 0.443] | 7.2 / 7.9 | 0.289 | +0.022 [-0.091, +0.100], p_Holm 1.000 | 0.083 [0.016, 0.162] | 1.8 / 6.5 | 0.317 | +0.235 [+0.081, +0.406], p_Holm 0.000 | 0.240 / 0.175 |
| iRF (fair) **(incomplete: 34 of 35 target x split fits timed out)** | 1/0/0 | 0.011 [0.011, 0.011] | 2.2 / 1.1 | 0.005 | -0.006 [-0.006, -0.006], p_Holm 1.000 | 0.002 [0.002, 0.002] | 1.5 / 0.9 | 0.001 | -0.001 [-0.001, -0.001], p_Holm 1.000 | 0.000 / 0.000 |
| RuleFit (fair) **(incomplete: 17 of 40 target x split fits timed out)** | 1/6/0 | 0.140 [0.002, 0.323] | 3.4 / 4.0 | 0.285 | +0.034 [+0.000, +0.065], p_Holm 0.000 | 0.147 [0.115, 0.181] | 1.4 / 4.0 | 0.409 | +0.262 [+0.176, +0.349], p_Holm 0.000 | 0.243 / 0.292 |
| PySR (shortlist) | 9/10/10 | 0.212 [0.114, 0.348] | 5.7 / 7.9 | 0.289 | +0.078 [-0.041, +0.222], p_Holm 1.000 | 0.053 [0.007, 0.138] | 1.2 / 6.5 | 0.317 | +0.266 [+0.120, +0.421], p_Holm 0.000 | 0.186 / 0.042 |
| TabPFN + SPEX | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies gse237183_glioma, nakajima_luad_b3, gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0201, pairs 0.0256

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.273 / 0.297 | 0.295 / 0.324 | 0.656 / 0.657 | 0.547 / 0.546 | 0.644 / 0.658 | 0.307 / 0.338 | 0.072 / 0.098 | 0.152 / 0.237 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.273 / 0.326 | 0.295 / 0.351 | 0.656 / 0.696 | 0.547 / 0.583 | 0.644 / 0.675 | 0.307 / 0.373 | 0.072 / 0.086 | 0.152 / 0.249 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.285 / 0.297 | 0.297 / 0.324 | 0.618 / 0.657 | 0.530 / 0.546 | 0.635 / 0.658 | 0.290 / 0.338 | 0.062 / 0.098 | 0.127 / 0.237 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.285 / 0.326 | 0.297 / 0.351 | 0.618 / 0.696 | 0.530 / 0.583 | 0.635 / 0.675 | 0.290 / 0.373 | 0.062 / 0.086 | 0.127 / 0.249 |
| random (chance reference, 20 seeds) | 0.007 / 0.005 | 0.014 / 0.009 | 0.169 / 0.198 | 0.074 / 0.086 | 0.500 / 0.500 | 0.024 / 0.032 | 0.000 / 0.000 | 0.024 / 0.032 |
| linear interaction screen | 0.285 / 0.016 | 0.297 / 0.024 | 0.618 / 0.283 | 0.530 / 0.167 | 0.635 / 0.507 | 0.290 / 0.045 | 0.062 / 0.013 | 0.127 / 0.054 |
| NID | 0.304 / 0.162 | 0.318 / 0.204 | 0.637 / 0.625 | 0.550 / 0.416 | 0.642 / 0.569 | 0.304 / 0.166 | 0.008 / 0.018 | 0.130 / 0.156 |
| MISTy-style forest (pairs derived) | 0.101 / 0.061 | 0.125 / 0.080 | 0.502 / 0.486 | 0.414 / 0.353 | 0.562 / 0.539 | 0.143 / 0.105 | 0.029 / 0.006 | 0.123 / 0.142 |
| Model-X knockoffs | 0.281 / 0.033 | 0.296 / 0.042 | 0.672 / 0.360 | 0.493 / 0.179 | 0.599 / 0.508 | 0.221 / 0.050 | 0.010 / 0.001 | 0.131 / 0.088 |
| GBM + SHAP interactions (native) | 0.316 / 0.033 | 0.333 / 0.039 | 0.779 / 0.387 | 0.673 / 0.275 | 0.651 / 0.514 | 0.326 / 0.063 | 0.093 / 0.000 | 0.149 / 0.061 |
| RuleFit (native) | 0.322 / 0.014 | 0.349 / 0.022 | 0.775 / 0.260 | 0.662 / 0.153 | 0.647 / 0.505 | 0.319 / 0.046 | 0.015 / -0.002 | 0.129 / 0.038 |
| stability selection | 0.201 / 0.012 | 0.241 / 0.022 | 0.729 / 0.427 | 0.381 / 0.234 | 0.544 / 0.508 | 0.123 / 0.068 | -0.019 / 0.046 | 0.146 / 0.084 |
| GBM + Friedman H | 0.006 / 0.031 | 0.011 / 0.059 | 0.211 / 0.332 | 0.090 / 0.120 | 0.500 / 0.509 | 0.027 / 0.052 | -0.004 / 0.005 | 0.029 / 0.063 |
| EBM (native) | 0.326 / 0.334 | 0.341 / 0.369 | 0.746 / 0.757 | 0.636 / 0.527 | 0.666 / 0.653 | 0.354 / 0.330 | 0.085 / -0.046 | 0.158 / 0.214 |
| iRF (native) | 0.279 / 0.016 | 0.311 / 0.023 | 0.764 / 0.227 | 0.638 / 0.116 | 0.635 / 0.506 | 0.292 / 0.046 | 0.028 / -0.002 | 0.145 / 0.036 |
| GBM (fair) | 0.157 / 0.063 | 0.187 / 0.068 | 0.583 / 0.486 | 0.518 / 0.330 | 0.601 / 0.528 | 0.232 / 0.105 | 0.042 / -0.001 | 0.185 / 0.146 |
| MISTy-style forest (fair) | 0.079 / 0.013 | 0.093 / 0.028 | 0.543 / 0.372 | 0.461 / 0.292 | 0.563 / 0.512 | 0.158 / 0.072 | 0.026 / 0.006 | 0.223 / 0.173 |
| EBM (fair) | 0.222 / 0.035 | 0.240 / 0.039 | 0.636 / 0.433 | 0.489 / 0.322 | 0.622 / 0.516 | 0.273 / 0.081 | 0.043 / -0.003 | 0.211 / 0.141 |
| iRF (fair) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.010 / 0.002 | 0.498 / 0.499 | 0.000 / 0.000 | 0.002 / -0.015 | 0.000 / 0.000 |
| RuleFit (fair) | 0.016 / 0.011 | 0.033 / 0.015 | 0.472 / 0.900 | 0.320 / 0.686 | 0.566 / 0.502 | 0.167 / 0.141 | 0.050 / -0.005 | 0.248 / 0.212 |
| PySR (shortlist) | 0.186 / 0.003 | 0.192 / 0.017 | 0.760 / 0.234 | 0.631 / 0.087 | 0.580 / 0.499 | 0.197 / 0.049 | -0.001 / -0.003 | 0.114 / 0.046 |

Label set per study: gse237183_glioma: label engine v05; nakajima_luad_b3: label engine v05; gse307534_lung_precursor_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## table1_glioma

### Table 1: studies gse237183_glioma; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0100, pairs 0.0098
Per-study prevalence: {"single": {"gse237183_glioma": 0.01}, "pair": {"gse237183_glioma": 0.0098}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 40 | 0.274 [0.187, 0.362] | 27.4 / 27.4 | 0.274 | - | 0.270 [0.176, 0.375] | 27.7 / 27.7 | 0.270 | - | 0.116 / 0.224 |
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | 0.274 [0.187, 0.362] | 27.4 / 27.4 | 0.274 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.334 [0.232, 0.440] | 34.2 / 27.7 | 0.270 | -0.063 [-0.107, -0.027], p_Holm 1.000 | 0.116 / 0.254 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 40 | 0.339 [0.239, 0.439] | 33.8 / 27.4 | 0.274 | -0.064 [-0.113, -0.021], p_Holm 1.000 | 0.270 [0.176, 0.375] | 27.7 / 27.7 | 0.270 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.143 / 0.224 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 40 | 0.339 [0.239, 0.439] | 33.8 / 27.4 | 0.274 | -0.064 [-0.113, -0.021], p_Holm 1.000 | 0.334 [0.232, 0.440] | 34.2 / 27.7 | 0.270 | -0.063 [-0.107, -0.027], p_Holm 1.000 | 0.143 / 0.254 |
| random (chance reference, 20 seeds) | 40 | 0.017 [0.013, 0.022] | 1.7 / 27.4 | 0.274 | +0.256 [+0.168, +0.347], p_Holm 0.000 | 0.016 [0.010, 0.022] | 1.6 / 27.7 | 0.270 | +0.255 [+0.159, +0.359], p_Holm 0.000 | 0.013 / 0.012 |
| linear interaction screen | 40 | 0.339 [0.239, 0.439] | 33.8 / 27.4 | 0.274 | -0.064 [-0.113, -0.021], p_Holm 1.000 | 0.018 [0.012, 0.025] | 1.9 / 27.7 | 0.270 | +0.252 [+0.156, +0.356], p_Holm 0.000 | 0.143 / 0.018 |
| NID | 36 | 0.380 [0.261, 0.501] | 37.3 / 29.3 | 0.299 | -0.081 [-0.139, -0.031], p_Holm 1.000 | 0.208 [0.135, 0.286] | 20.3 / 33.3 | 0.341 | +0.134 [+0.052, +0.216], p_Holm 0.012 | 0.143 / 0.181 |
| MISTy-style forest (pairs derived) | 40 | 0.119 [0.087, 0.153] | 11.9 / 27.4 | 0.274 | +0.155 [+0.092, +0.223], p_Holm 0.000 | 0.063 [0.039, 0.093] | 6.5 / 27.7 | 0.270 | +0.207 [+0.108, +0.316], p_Holm 0.000 | 0.070 / 0.059 |
| Model-X knockoffs | 36 | 0.294 [0.201, 0.393] | 28.8 / 29.3 | 0.299 | +0.006 [-0.037, +0.047], p_Holm 1.000 | 0.031 [0.016, 0.050] | 3.0 / 33.3 | 0.341 | +0.310 [+0.199, +0.431], p_Holm 0.000 | 0.121 / 0.029 |
| GBM + SHAP interactions (native) | 36 | 0.368 [0.256, 0.486] | 36.1 / 29.3 | 0.299 | -0.069 [-0.128, -0.018], p_Holm 1.000 | 0.048 [0.023, 0.082] | 4.7 / 33.3 | 0.341 | +0.294 [+0.179, +0.419], p_Holm 0.000 | 0.149 / 0.042 |
| RuleFit (native) **(incomplete: 12 of 115 target x split fits timed out)** | 33 | 0.345 [0.238, 0.457] | 31.1 / 29.3 | 0.325 | -0.020 [-0.081, +0.033], p_Holm 1.000 | 0.025 [0.016, 0.035] | 2.1 / 30.9 | 0.356 | +0.332 [+0.218, +0.459], p_Holm 0.000 | 0.136 / 0.020 |
| stability selection **(incomplete: 46 of 106 target x split fits timed out)** | 21 | 0.158 [0.100, 0.213] | 13.3 / 27.8 | 0.330 | +0.172 [+0.084, +0.272], p_Holm 0.000 | 0.026 [0.013, 0.040] | 2.0 / 32.9 | 0.425 | +0.401 [+0.226, +0.581], p_Holm 0.000 | 0.128 / 0.027 |
| GBM + Friedman H | 36 | 0.017 [0.012, 0.022] | 1.6 / 29.3 | 0.299 | +0.282 [+0.183, +0.389], p_Holm 0.000 | 0.034 [0.024, 0.044] | 3.3 / 33.3 | 0.341 | +0.308 [+0.197, +0.428], p_Holm 0.000 | 0.013 / 0.046 |
| EBM (native) | 36 | 0.374 [0.259, 0.492] | 36.7 / 29.3 | 0.299 | -0.075 [-0.132, -0.026], p_Holm 1.000 | 0.368 [0.261, 0.480] | 35.9 / 33.3 | 0.341 | -0.027 [-0.120, +0.059], p_Holm 1.000 | 0.151 / 0.260 |
| iRF (native) | 36 | 0.314 [0.216, 0.414] | 30.8 / 29.3 | 0.299 | -0.014 [-0.065, +0.033], p_Holm 1.000 | 0.032 [0.015, 0.052] | 3.1 / 33.3 | 0.341 | +0.310 [+0.201, +0.433], p_Holm 0.000 | 0.129 / 0.020 |
| GBM (fair) | 9 | 0.335 [0.151, 0.511] | 83.6 / 87.4 | 0.350 | +0.017 [-0.172, +0.186], p_Holm 1.000 | 0.064 [0.008, 0.142] | 29.0 / 184.1 | 0.408 | +0.344 [+0.148, +0.564], p_Holm 0.005 | 0.120 / 0.033 |
| MISTy-style forest (fair) | 9 | 0.138 [0.059, 0.229] | 34.5 / 87.4 | 0.350 | +0.214 [+0.061, +0.390], p_Holm 0.067 | 0.011 [0.004, 0.022] | 5.2 / 184.1 | 0.408 | +0.398 [+0.185, +0.621], p_Holm 0.005 | 0.072 / 0.008 |
| EBM (fair) | 9 | 0.436 [0.221, 0.632] | 108.9 / 87.4 | 0.350 | -0.085 [-0.230, +0.004], p_Holm 1.000 | 0.018 [0.005, 0.032] | 8.1 / 184.1 | 0.408 | +0.392 [+0.182, +0.610], p_Holm 0.005 | 0.146 / 0.021 |
| iRF (fair) **(incomplete: 20 of 21 target x split fits timed out)** | 1 | 0.011 [0.011, 0.011] | 2.2 / 1.1 | 0.005 | -0.006 [-0.006, -0.006], p_Holm 1.000 | 0.002 [0.002, 0.002] | 1.5 / 0.9 | 0.001 | -0.001 [-0.001, -0.001], p_Holm 1.000 | 0.000 / 0.000 |
| RuleFit (fair) **(incomplete: 16 of 22 target x split fits timed out)** | 1 | 0.002 [0.002, 0.002] | 7.0 / 8.0 | 0.003 | +0.000 [+0.000, +0.000], p_Holm 0.000 | - | nan / nan | nan | - | 0.000 / - |
| PySR (shortlist) | 9 | 0.332 [0.154, 0.506] | 83.0 / 87.4 | 0.350 | +0.018 [-0.110, +0.129], p_Holm 1.000 | 0.008 [0.005, 0.011] | 3.4 / 184.1 | 0.408 | +0.402 [+0.187, +0.630], p_Holm 0.005 | 0.099 / 0.002 |
| TabPFN + SPEX | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies gse237183_glioma; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0100, pairs 0.0098

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.287 / 0.285 | 0.312 / 0.317 | 0.634 / 0.585 | 0.503 / 0.452 | 0.630 / 0.633 | 0.271 / 0.276 | 0.025 / 0.054 | 0.074 / 0.139 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.287 / 0.344 | 0.312 / 0.381 | 0.634 / 0.649 | 0.503 / 0.509 | 0.630 / 0.658 | 0.271 / 0.326 | 0.025 / 0.022 | 0.074 / 0.159 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.347 / 0.285 | 0.357 / 0.317 | 0.634 / 0.585 | 0.559 / 0.452 | 0.657 / 0.633 | 0.325 / 0.276 | 0.015 / 0.054 | 0.090 / 0.139 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.347 / 0.344 | 0.357 / 0.381 | 0.634 / 0.649 | 0.559 / 0.509 | 0.657 / 0.658 | 0.325 / 0.326 | 0.015 / 0.022 | 0.090 / 0.159 |
| random (chance reference, 20 seeds) | 0.007 / 0.005 | 0.014 / 0.009 | 0.113 / 0.102 | 0.050 / 0.046 | 0.500 / 0.500 | 0.013 / 0.012 | 0.000 / 0.000 | 0.013 / 0.012 |
| linear interaction screen | 0.347 / 0.010 | 0.357 / 0.018 | 0.634 / 0.131 | 0.559 / 0.066 | 0.657 / 0.501 | 0.325 / 0.014 | 0.015 / 0.010 | 0.090 / 0.017 |
| NID | 0.397 / 0.241 | 0.402 / 0.282 | 0.671 / 0.598 | 0.599 / 0.427 | 0.674 / 0.590 | 0.359 / 0.191 | 0.025 / 0.033 | 0.087 / 0.119 |
| MISTy-style forest (pairs derived) | 0.110 / 0.065 | 0.122 / 0.075 | 0.428 / 0.402 | 0.362 / 0.256 | 0.555 / 0.528 | 0.123 / 0.068 | 0.026 / -0.014 | 0.051 / 0.043 |
| Model-X knockoffs | 0.342 / 0.024 | 0.358 / 0.028 | 0.677 / 0.223 | 0.554 / 0.105 | 0.616 / 0.508 | 0.243 / 0.030 | -0.002 / -0.001 | 0.077 / 0.024 |
| GBM + SHAP interactions (native) | 0.396 / 0.035 | 0.400 / 0.041 | 0.741 / 0.251 | 0.645 / 0.175 | 0.666 / 0.516 | 0.344 / 0.046 | 0.104 / -0.000 | 0.087 / 0.028 |
| RuleFit (native) | 0.394 / 0.010 | 0.412 / 0.018 | 0.742 / 0.152 | 0.632 / 0.112 | 0.656 / 0.504 | 0.323 / 0.021 | 0.020 / -0.003 | 0.082 / 0.018 |
| stability selection | 0.313 / 0.004 | 0.371 / 0.017 | 0.694 / 0.221 | 0.319 / 0.083 | 0.555 / 0.501 | 0.124 / 0.022 | -0.035 / 0.046 | 0.090 / 0.030 |
| GBM + Friedman H | 0.007 / 0.044 | 0.013 / 0.085 | 0.120 / 0.270 | 0.050 / 0.101 | 0.500 / 0.509 | 0.012 / 0.032 | -0.006 / 0.001 | 0.014 / 0.043 |
| EBM (native) | 0.383 / 0.451 | 0.392 / 0.478 | 0.694 / 0.740 | 0.603 / 0.544 | 0.676 / 0.670 | 0.363 / 0.351 | 0.084 / -0.150 | 0.095 / 0.158 |
| iRF (native) | 0.340 / 0.024 | 0.368 / 0.032 | 0.750 / 0.140 | 0.630 / 0.083 | 0.649 / 0.511 | 0.308 / 0.035 | 0.006 / -0.002 | 0.081 / 0.017 |
| GBM (fair) | 0.354 / 0.109 | 0.409 / 0.109 | 0.648 / 0.292 | 0.612 / 0.147 | 0.652 / 0.546 | 0.308 / 0.096 | -0.010 / -0.005 | 0.063 / 0.017 |
| MISTy-style forest (fair) | 0.156 / 0.009 | 0.156 / 0.037 | 0.463 / 0.083 | 0.436 / 0.058 | 0.576 / 0.502 | 0.156 / 0.007 | -0.016 / 0.003 | 0.036 / 0.010 |
| EBM (fair) | 0.490 / 0.026 | 0.504 / 0.026 | 0.741 / 0.167 | 0.682 / 0.083 | 0.702 / 0.512 | 0.409 / 0.026 | -0.011 / -0.005 | 0.081 / 0.010 |
| iRF (fair) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.010 / 0.002 | 0.498 / 0.499 | 0.000 / 0.000 | 0.002 / -0.015 | 0.000 / 0.000 |
| RuleFit (fair) | 0.000 / - | 0.000 / - | 0.000 / - | 0.002 / - | 0.499 / - | 0.000 / - | -0.007 / - | 0.000 / - |
| PySR (shortlist) | 0.358 / 0.001 | 0.364 / 0.021 | 0.778 / 0.008 | 0.723 / 0.022 | 0.649 / 0.499 | 0.303 / 0.001 | -0.008 / -0.006 | 0.052 / 0.005 |

Label set per study: gse237183_glioma: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## table1_nakajima

### Table 1: studies nakajima_luad_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0343, pairs 0.0489
Per-study prevalence: {"single": {"nakajima_luad_b3": 0.0343}, "pair": {"nakajima_luad_b3": 0.0489}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 37 | 0.365 [0.264, 0.476] | 10.6 / 10.6 | 0.365 | - | 0.457 [0.354, 0.565] | 9.3 / 9.3 | 0.457 | - | 0.330 / 0.495 |
| ours: support-fit ordering, joint pair gain (suppl.) | 37 | 0.365 [0.264, 0.476] | 10.6 / 10.6 | 0.365 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.480 [0.371, 0.595] | 9.8 / 9.3 | 0.457 | -0.023 [-0.038, -0.010], p_Holm 1.000 | 0.330 / 0.502 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 37 | 0.304 [0.216, 0.407] | 8.9 / 10.6 | 0.365 | +0.061 [+0.035, +0.089], p_Holm 0.000 | 0.457 [0.354, 0.565] | 9.3 / 9.3 | 0.457 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.251 / 0.495 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 37 | 0.304 [0.216, 0.407] | 8.9 / 10.6 | 0.365 | +0.061 [+0.035, +0.089], p_Holm 0.000 | 0.480 [0.371, 0.595] | 9.8 / 9.3 | 0.457 | -0.023 [-0.038, -0.010], p_Holm 1.000 | 0.251 / 0.502 |
| random (chance reference, 20 seeds) | 37 | 0.043 [0.029, 0.057] | 1.2 / 10.6 | 0.365 | +0.325 [+0.220, +0.438], p_Holm 0.000 | 0.067 [0.044, 0.092] | 1.4 / 9.3 | 0.457 | +0.393 [+0.282, +0.509], p_Holm 0.000 | 0.039 / 0.064 |
| linear interaction screen | 37 | 0.304 [0.216, 0.407] | 8.9 / 10.6 | 0.365 | +0.061 [+0.035, +0.089], p_Holm 0.000 | 0.077 [0.051, 0.105] | 1.6 / 9.3 | 0.457 | +0.383 [+0.272, +0.498], p_Holm 0.000 | 0.251 / 0.122 |
| NID | 37 | 0.305 [0.213, 0.413] | 8.9 / 10.6 | 0.365 | +0.060 [+0.032, +0.089], p_Holm 0.000 | 0.179 [0.144, 0.216] | 3.7 / 9.3 | 0.457 | +0.280 [+0.196, +0.367], p_Holm 0.000 | 0.259 / 0.280 |
| MISTy-style forest (pairs derived) | 37 | 0.199 [0.157, 0.242] | 5.8 / 10.6 | 0.365 | +0.168 [+0.098, +0.248], p_Holm 0.000 | 0.167 [0.120, 0.214] | 3.4 / 9.3 | 0.457 | +0.293 [+0.191, +0.403], p_Holm 0.000 | 0.289 / 0.334 |
| Model-X knockoffs | 31 | 0.280 [0.199, 0.375] | 8.0 / 12.3 | 0.431 | +0.151 [+0.091, +0.222], p_Holm 0.000 | 0.094 [0.063, 0.126] | 1.9 / 9.3 | 0.464 | +0.374 [+0.265, +0.483], p_Holm 0.000 | 0.276 / 0.203 |
| GBM + SHAP interactions (native) | 30 | 0.367 [0.257, 0.491] | 10.5 / 11.8 | 0.412 | +0.045 [+0.004, +0.085], p_Holm 0.046 | 0.104 [0.074, 0.135] | 2.1 / 9.2 | 0.460 | +0.359 [+0.259, +0.461], p_Holm 0.000 | 0.319 / 0.163 |
| RuleFit (native) | 29 | 0.382 [0.263, 0.525] | 10.1 / 12.2 | 0.459 | +0.076 [+0.029, +0.125], p_Holm 0.004 | 0.093 [0.057, 0.135] | 1.7 / 8.9 | 0.481 | +0.391 [+0.282, +0.509], p_Holm 0.000 | 0.280 / 0.068 |
| stability selection **(incomplete: 43 of 101 target x split fits timed out)** | 18 | 0.101 [0.071, 0.130] | 1.7 / 4.2 | 0.244 | +0.144 [+0.087, +0.204], p_Holm 0.000 | 0.123 [0.088, 0.160] | 1.5 / 3.8 | 0.322 | +0.199 [+0.124, +0.275], p_Holm 0.000 | 0.260 / 0.180 |
| GBM + Friedman H **(incomplete: 13 of 118 target x split fits timed out)** | 27 | 0.051 [0.034, 0.071] | 1.5 / 9.9 | 0.334 | +0.286 [+0.194, +0.390], p_Holm 0.000 | 0.086 [0.060, 0.113] | 1.8 / 8.6 | 0.404 | +0.321 [+0.226, +0.425], p_Holm 0.000 | 0.053 / 0.085 |
| EBM (native) | 29 | 0.424 [0.310, 0.559] | 12.7 / 13.2 | 0.440 | +0.015 [-0.020, +0.049], p_Holm 0.371 | 0.393 [0.300, 0.502] | 8.4 / 10.1 | 0.474 | +0.081 [+0.014, +0.159], p_Holm 0.044 | 0.333 / 0.373 |
| iRF (native) | 29 | 0.343 [0.243, 0.456] | 9.6 / 11.3 | 0.402 | +0.060 [+0.032, +0.089], p_Holm 0.000 | 0.070 [0.044, 0.097] | 1.4 / 9.0 | 0.457 | +0.390 [+0.271, +0.513], p_Holm 0.000 | 0.307 / 0.059 |
| GBM (fair) | 10 | 0.256 [0.220, 0.286] | 2.7 / 3.6 | 0.349 | +0.093 [+0.064, +0.122], p_Holm 0.000 | 0.162 [0.137, 0.188] | 1.2 / 3.0 | 0.419 | +0.257 [+0.197, +0.314], p_Holm 0.000 | 0.523 / 0.457 |
| MISTy-style forest (fair) | 10 | 0.253 [0.206, 0.293] | 2.6 / 3.6 | 0.349 | +0.096 [+0.073, +0.118], p_Holm 0.000 | 0.196 [0.160, 0.231] | 1.4 / 3.0 | 0.419 | +0.223 [+0.174, +0.269], p_Holm 0.000 | 0.647 / 0.550 |
| EBM (fair) | 10 | 0.267 [0.228, 0.300] | 2.8 / 3.6 | 0.349 | +0.082 [+0.057, +0.107], p_Holm 0.000 | 0.161 [0.136, 0.187] | 1.2 / 3.0 | 0.419 | +0.258 [+0.200, +0.312], p_Holm 0.000 | 0.527 / 0.480 |
| iRF (fair) | no receipts (timeout / not run) | | | | | | | | | |
| RuleFit (fair) | 6 | 0.277 [0.198, 0.347] | 2.9 / 3.5 | 0.332 | +0.055 [+0.040, +0.067], p_Holm 0.000 | 0.147 [0.111, 0.187] | 1.1 / 3.0 | 0.409 | +0.262 [+0.162, +0.359], p_Holm 0.000 | 0.486 / 0.292 |
| PySR (shortlist) | 10 | 0.124 [0.103, 0.143] | 1.3 / 3.6 | 0.349 | +0.225 [+0.181, +0.266], p_Holm 0.000 | 0.140 [0.114, 0.165] | 1.0 / 3.0 | 0.419 | +0.279 [+0.216, +0.338], p_Holm 0.000 | 0.394 / 0.118 |
| TabPFN + SPEX | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies nakajima_luad_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0343, pairs 0.0489

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.267 / 0.313 | 0.288 / 0.335 | 0.719 / 0.795 | 0.615 / 0.702 | 0.670 / 0.696 | 0.366 / 0.434 | 0.121 / 0.162 | 0.283 / 0.401 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.267 / 0.319 | 0.288 / 0.336 | 0.719 / 0.791 | 0.615 / 0.722 | 0.670 / 0.713 | 0.366 / 0.469 | 0.121 / 0.149 | 0.283 / 0.406 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.253 / 0.313 | 0.270 / 0.335 | 0.679 / 0.795 | 0.542 / 0.702 | 0.629 / 0.696 | 0.288 / 0.434 | 0.102 / 0.162 | 0.211 / 0.401 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.253 / 0.319 | 0.270 / 0.336 | 0.679 / 0.791 | 0.542 / 0.722 | 0.629 / 0.713 | 0.288 / 0.469 | 0.102 / 0.149 | 0.211 / 0.406 |
| random (chance reference, 20 seeds) | 0.008 / 0.005 | 0.014 / 0.009 | 0.253 / 0.353 | 0.108 / 0.149 | 0.500 / 0.500 | 0.039 / 0.064 | -0.000 / 0.000 | 0.039 / 0.063 |
| linear interaction screen | 0.253 / 0.009 | 0.270 / 0.016 | 0.679 / 0.463 | 0.542 / 0.297 | 0.629 / 0.509 | 0.288 / 0.080 | 0.102 / 0.019 | 0.211 / 0.112 |
| NID | 0.255 / 0.124 | 0.286 / 0.177 | 0.692 / 0.767 | 0.559 / 0.488 | 0.633 / 0.568 | 0.297 / 0.191 | 0.015 / 0.035 | 0.221 / 0.259 |
| MISTy-style forest (pairs derived) | 0.111 / 0.062 | 0.162 / 0.088 | 0.648 / 0.650 | 0.569 / 0.512 | 0.587 / 0.557 | 0.204 / 0.167 | 0.031 / 0.021 | 0.259 / 0.306 |
| Model-X knockoffs | 0.294 / 0.038 | 0.308 / 0.051 | 0.806 / 0.575 | 0.545 / 0.299 | 0.610 / 0.511 | 0.258 / 0.090 | 0.026 / 0.002 | 0.238 / 0.198 |
| GBM + SHAP interactions (native) | 0.297 / 0.037 | 0.335 / 0.042 | 0.839 / 0.611 | 0.735 / 0.442 | 0.662 / 0.516 | 0.364 / 0.101 | 0.107 / -0.001 | 0.256 / 0.114 |
| RuleFit (native) | 0.339 / 0.022 | 0.369 / 0.027 | 0.879 / 0.407 | 0.762 / 0.204 | 0.668 / 0.509 | 0.380 / 0.089 | 0.021 / -0.003 | 0.207 / 0.069 |
| stability selection | 0.020 / 0.007 | 0.035 / 0.012 | 0.728 / 0.721 | 0.470 / 0.445 | 0.525 / 0.507 | 0.114 / 0.123 | 0.004 / 0.032 | 0.245 / 0.163 |
| GBM + Friedman H | 0.005 / 0.010 | 0.008 / 0.034 | 0.335 / 0.446 | 0.151 / 0.172 | 0.499 / 0.507 | 0.047 / 0.081 | -0.002 / 0.010 | 0.050 / 0.103 |
| EBM (native) | 0.346 / 0.285 | 0.367 / 0.325 | 0.868 / 0.846 | 0.712 / 0.585 | 0.691 / 0.678 | 0.418 / 0.402 | 0.115 / 0.073 | 0.280 / 0.320 |
| iRF (native) | 0.281 / 0.003 | 0.313 / 0.008 | 0.793 / 0.340 | 0.723 / 0.135 | 0.651 / 0.499 | 0.336 / 0.066 | 0.067 / -0.004 | 0.257 / 0.064 |
| GBM (fair) | 0.035 / 0.017 | 0.067 / 0.030 | 0.933 / 0.967 | 0.793 / 0.695 | 0.616 / 0.509 | 0.304 / 0.154 | 0.134 / -0.000 | 0.475 / 0.409 |
| MISTy-style forest (fair) | 0.043 / 0.020 | 0.084 / 0.036 | 1.000 / 0.967 | 0.847 / 0.771 | 0.601 / 0.533 | 0.278 / 0.196 | 0.051 / 0.012 | 0.617 / 0.502 |
| EBM (fair) | 0.036 / 0.017 | 0.073 / 0.028 | 0.900 / 0.933 | 0.624 / 0.699 | 0.623 / 0.508 | 0.316 / 0.153 | 0.136 / -0.002 | 0.523 / 0.399 |
| RuleFit (fair) | 0.033 / 0.011 | 0.067 / 0.015 | 0.944 / 0.900 | 0.638 / 0.686 | 0.633 / 0.502 | 0.335 / 0.141 | 0.107 / -0.005 | 0.496 / 0.212 |
| PySR (shortlist) | 0.027 / 0.004 | 0.035 / 0.009 | 0.960 / 0.647 | 0.829 / 0.215 | 0.510 / 0.498 | 0.115 / 0.136 | 0.009 / -0.004 | 0.252 / 0.125 |

Label set per study: nakajima_luad_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## table1_gse307534

### Table 1: studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0160, pairs 0.0183
Per-study prevalence: {"single": {"gse307534_lung_precursor_b3": 0.016}, "pair": {"gse307534_lung_precursor_b3": 0.0183}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 40 | 0.278 [0.192, 0.371] | 17.4 / 17.4 | 0.278 | - | 0.314 [0.210, 0.425] | 17.1 / 17.1 | 0.314 | - | 0.137 / 0.277 |
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | 0.278 [0.192, 0.371] | 17.4 / 17.4 | 0.278 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.329 [0.223, 0.442] | 18.0 / 17.1 | 0.314 | -0.016 [-0.035, +0.008], p_Holm 1.000 | 0.137 / 0.302 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 40 | 0.268 [0.183, 0.360] | 16.8 / 17.4 | 0.278 | +0.010 [-0.001, +0.022], p_Holm 0.266 | 0.314 [0.210, 0.425] | 17.1 / 17.1 | 0.314 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.126 / 0.277 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 40 | 0.268 [0.183, 0.360] | 16.8 / 17.4 | 0.278 | +0.010 [-0.001, +0.022], p_Holm 0.266 | 0.329 [0.223, 0.442] | 18.0 / 17.1 | 0.314 | -0.016 [-0.035, +0.008], p_Holm 1.000 | 0.126 / 0.302 |
| random (chance reference, 20 seeds) | 40 | 0.024 [0.016, 0.034] | 1.5 / 17.4 | 0.278 | +0.255 [+0.167, +0.350], p_Holm 0.000 | 0.024 [0.013, 0.038] | 1.3 / 17.1 | 0.314 | +0.290 [+0.187, +0.401], p_Holm 0.000 | 0.019 / 0.022 |
| linear interaction screen | 40 | 0.268 [0.183, 0.360] | 16.8 / 17.4 | 0.278 | +0.010 [-0.001, +0.022], p_Holm 0.266 | 0.040 [0.025, 0.058] | 2.2 / 17.1 | 0.314 | +0.274 [+0.177, +0.380], p_Holm 0.000 | 0.126 / 0.040 |
| NID | 40 | 0.267 [0.182, 0.360] | 16.7 / 17.4 | 0.278 | +0.011 [+0.000, +0.022], p_Holm 0.192 | 0.112 [0.076, 0.153] | 6.1 / 17.1 | 0.314 | +0.202 [+0.124, +0.285], p_Holm 0.000 | 0.127 / 0.120 |
| MISTy-style forest (pairs derived) | 40 | 0.100 [0.072, 0.132] | 6.3 / 17.4 | 0.278 | +0.178 [+0.105, +0.260], p_Holm 0.000 | 0.072 [0.047, 0.097] | 4.0 / 17.1 | 0.314 | +0.242 [+0.148, +0.348], p_Holm 0.000 | 0.080 / 0.100 |
| Model-X knockoffs | 37 | 0.177 [0.106, 0.254] | 10.5 / 16.8 | 0.283 | +0.107 [+0.063, +0.157], p_Holm 0.000 | 0.037 [0.023, 0.051] | 2.0 / 18.1 | 0.327 | +0.291 [+0.190, +0.396], p_Holm 0.000 | 0.109 / 0.046 |
| GBM + SHAP interactions (native) | 37 | 0.273 [0.188, 0.368] | 16.2 / 16.8 | 0.283 | +0.010 [-0.001, +0.022], p_Holm 0.266 | 0.043 [0.027, 0.061] | 2.4 / 18.1 | 0.327 | +0.285 [+0.186, +0.388], p_Holm 0.000 | 0.165 / 0.058 |
| RuleFit (native) | 36 | 0.250 [0.164, 0.347] | 14.7 / 17.1 | 0.290 | +0.040 [+0.016, +0.072], p_Holm 0.000 | 0.032 [0.020, 0.045] | 1.8 / 18.4 | 0.334 | +0.303 [+0.198, +0.416], p_Holm 0.000 | 0.138 / 0.030 |
| stability selection **(incomplete: 54 of 100 target x split fits timed out)** | 17 | 0.151 [0.092, 0.211] | 7.0 / 17.0 | 0.368 | +0.219 [+0.115, +0.325], p_Holm 0.000 | 0.055 [0.027, 0.086] | 2.1 / 17.0 | 0.439 | +0.387 [+0.220, +0.544], p_Holm 0.000 | 0.134 / 0.062 |
| GBM + Friedman H | 37 | 0.026 [0.018, 0.035] | 1.5 / 16.9 | 0.282 | +0.257 [+0.165, +0.360], p_Holm 0.000 | 0.037 [0.024, 0.052] | 2.1 / 18.6 | 0.325 | +0.289 [+0.188, +0.394], p_Holm 0.000 | 0.024 / 0.049 |
| EBM (native) | 37 | 0.278 [0.189, 0.375] | 16.5 / 16.8 | 0.283 | +0.005 [-0.006, +0.017], p_Holm 0.570 | 0.258 [0.177, 0.340] | 14.3 / 18.1 | 0.327 | +0.069 [+0.021, +0.122], p_Holm 0.004 | 0.159 / 0.240 |
| iRF (native) | 37 | 0.234 [0.158, 0.320] | 13.8 / 16.8 | 0.283 | +0.049 [+0.024, +0.079], p_Holm 0.000 | 0.042 [0.024, 0.061] | 2.3 / 18.1 | 0.327 | +0.286 [+0.185, +0.393], p_Holm 0.000 | 0.136 / 0.029 |
| GBM (fair) | 10 | 0.098 [0.014, 0.260] | 9.8 / 17.3 | 0.174 | +0.075 [+0.005, +0.174], p_Holm 0.000 | 0.070 [0.008, 0.177] | 13.9 / 28.0 | 0.141 | +0.071 [+0.001, +0.216], p_Holm 0.081 | 0.030 / 0.023 |
| MISTy-style forest (fair) | 10 | 0.042 [0.013, 0.084] | 4.1 / 17.3 | 0.174 | +0.131 [+0.019, +0.267], p_Holm 0.005 | 0.017 [0.006, 0.036] | 3.4 / 28.0 | 0.141 | +0.124 [+0.004, +0.287], p_Holm 0.081 | 0.027 / 0.007 |
| EBM (fair) | 10 | 0.106 [0.015, 0.268] | 10.5 / 17.3 | 0.174 | +0.067 [+0.005, +0.160], p_Holm 0.000 | 0.071 [0.009, 0.178] | 14.1 / 28.0 | 0.141 | +0.070 [-0.000, +0.216], p_Holm 0.114 | 0.047 / 0.023 |
| iRF (fair) | no receipts (timeout / not run) | | | | | | | | | |
| RuleFit (fair) | no receipts (timeout / not run) | | | | | | | | | |
| PySR (shortlist) | 10 | 0.181 [0.047, 0.329] | 18.0 / 17.3 | 0.174 | -0.008 [-0.130, +0.079], p_Holm 1.000 | 0.011 [0.007, 0.016] | 2.2 / 28.0 | 0.141 | +0.130 [+0.001, +0.305], p_Holm 0.098 | 0.064 / 0.005 |
| TabPFN + SPEX | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0160, pairs 0.0183

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.265 / 0.293 | 0.286 / 0.319 | 0.614 / 0.593 | 0.523 / 0.484 | 0.634 / 0.644 | 0.284 / 0.306 | 0.071 / 0.080 | 0.098 / 0.170 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.265 / 0.317 | 0.286 / 0.338 | 0.614 / 0.648 | 0.523 / 0.517 | 0.634 / 0.654 | 0.284 / 0.325 | 0.071 / 0.087 | 0.098 / 0.184 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.256 / 0.293 | 0.263 / 0.319 | 0.542 / 0.593 | 0.489 / 0.484 | 0.620 / 0.644 | 0.256 / 0.306 | 0.068 / 0.080 | 0.079 / 0.170 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.256 / 0.317 | 0.263 / 0.338 | 0.542 / 0.648 | 0.489 / 0.517 | 0.620 / 0.654 | 0.256 / 0.325 | 0.068 / 0.087 | 0.079 / 0.184 |
| random (chance reference, 20 seeds) | 0.007 / 0.005 | 0.014 / 0.010 | 0.142 / 0.138 | 0.064 / 0.063 | 0.500 / 0.500 | 0.019 / 0.021 | -0.000 / 0.000 | 0.019 / 0.021 |
| linear interaction screen | 0.256 / 0.030 | 0.263 / 0.037 | 0.542 / 0.256 | 0.489 / 0.140 | 0.620 / 0.511 | 0.256 / 0.043 | 0.068 / 0.012 | 0.079 / 0.032 |
| NID | 0.260 / 0.122 | 0.267 / 0.153 | 0.547 / 0.510 | 0.492 / 0.332 | 0.620 / 0.548 | 0.257 / 0.116 | -0.015 / -0.015 | 0.081 / 0.090 |
| MISTy-style forest (pairs derived) | 0.082 / 0.055 | 0.090 / 0.076 | 0.430 / 0.407 | 0.312 / 0.291 | 0.543 / 0.531 | 0.102 / 0.081 | 0.029 / 0.011 | 0.060 / 0.079 |
| Model-X knockoffs | 0.208 / 0.038 | 0.221 / 0.048 | 0.532 / 0.284 | 0.379 / 0.134 | 0.571 / 0.504 | 0.163 / 0.030 | 0.007 / 0.002 | 0.079 / 0.042 |
| GBM + SHAP interactions (native) | 0.255 / 0.027 | 0.263 / 0.035 | 0.757 / 0.300 | 0.638 / 0.207 | 0.625 / 0.510 | 0.270 / 0.041 | 0.067 / 0.001 | 0.105 / 0.040 |
| RuleFit (native) | 0.233 / 0.012 | 0.266 / 0.021 | 0.704 / 0.222 | 0.591 / 0.142 | 0.616 / 0.503 | 0.254 / 0.028 | 0.004 / -0.000 | 0.096 / 0.027 |
| stability selection | 0.270 / 0.026 | 0.317 / 0.037 | 0.767 / 0.339 | 0.355 / 0.175 | 0.553 / 0.515 | 0.131 / 0.059 | -0.025 / 0.060 | 0.105 / 0.059 |
| GBM + Friedman H | 0.007 / 0.039 | 0.012 / 0.059 | 0.177 / 0.280 | 0.068 / 0.086 | 0.500 / 0.511 | 0.022 / 0.042 | -0.004 / 0.005 | 0.023 / 0.043 |
| EBM (native) | 0.250 / 0.267 | 0.263 / 0.305 | 0.676 / 0.685 | 0.593 / 0.452 | 0.630 / 0.610 | 0.280 / 0.237 | 0.057 / -0.061 | 0.100 / 0.164 |
| iRF (native) | 0.218 / 0.022 | 0.253 / 0.031 | 0.748 / 0.202 | 0.562 / 0.129 | 0.606 / 0.508 | 0.232 / 0.038 | 0.011 / -0.000 | 0.098 / 0.026 |
| GBM (fair) | 0.081 / 0.064 | 0.084 / 0.064 | 0.167 / 0.200 | 0.149 / 0.148 | 0.535 / 0.530 | 0.085 / 0.066 | 0.003 / 0.002 | 0.017 / 0.012 |
| MISTy-style forest (fair) | 0.038 / 0.009 | 0.038 / 0.012 | 0.167 / 0.067 | 0.101 / 0.047 | 0.513 / 0.503 | 0.041 / 0.012 | 0.043 / 0.003 | 0.015 / 0.008 |
| EBM (fair) | 0.138 / 0.063 | 0.143 / 0.063 | 0.267 / 0.200 | 0.163 / 0.184 | 0.540 / 0.529 | 0.094 / 0.065 | 0.003 / -0.001 | 0.029 / 0.013 |
| PySR (shortlist) | 0.174 / 0.005 | 0.178 / 0.021 | 0.543 / 0.047 | 0.341 / 0.024 | 0.580 / 0.501 | 0.174 / 0.008 | -0.005 / 0.000 | 0.039 / 0.008 |

Label set per study: gse307534_lung_precursor_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## E1E2_gse237183_glioma

# E1 (beam K=64 vs exhaustive support-fit scoring of the same pool; B_known, level 95; ours column = beam64)

### Table 1: studies gse237183_glioma; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0100, pairs 0.0098
Per-study prevalence: {"single": {"gse237183_glioma": 0.01}, "pair": {"gse237183_glioma": 0.0098}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied) | 40 | 0.207 [0.136, 0.285] | 20.6 / 20.6 | 0.207 | - | 0.026 [0.013, 0.044] | 2.6 / 2.6 | 0.026 | - | 0.082 / 0.051 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | 0.274 [0.187, 0.362] | 27.4 / 20.6 | 0.207 | -0.067 [-0.142, -0.004], p_Holm 1.000 | 0.270 [0.176, 0.375] | 27.7 / 2.6 | 0.026 | -0.244 [-0.345, -0.150], p_Holm 1.000 | 0.116 / 0.224 |
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | 0.274 [0.187, 0.362] | 27.4 / 20.6 | 0.207 | -0.067 [-0.142, -0.004], p_Holm 1.000 | 0.334 [0.232, 0.440] | 34.2 / 2.6 | 0.026 | -0.308 [-0.416, -0.206], p_Holm 1.000 | 0.116 / 0.254 |

### Table 2 (secondary, same score files): studies gse237183_glioma; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0100, pairs 0.0098

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied) | 0.195 / 0.016 | 0.197 / 0.025 | 0.473 / 0.155 | 0.433 / 0.112 | 0.593 / 0.505 | 0.197 / 0.022 | 0.162 / 0.144 | 0.048 / 0.034 |
| ours: support-fit ordering (pairs = mean of singles) | 0.287 / 0.285 | 0.312 / 0.317 | 0.634 / 0.585 | 0.503 / 0.452 | 0.630 / 0.633 | 0.271 / 0.276 | 0.025 / 0.054 | 0.074 / 0.139 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.287 / 0.344 | 0.312 / 0.381 | 0.634 / 0.649 | 0.503 / 0.509 | 0.630 / 0.658 | 0.271 / 0.326 | 0.025 / 0.022 | 0.074 / 0.159 |

Label set per study: gse237183_glioma: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.


# E2: pair rankings, labels B_known (decomp labels guarded by G_known > 0); ours column = joint pair gain

### Table 1: studies gse237183_glioma; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0100, pairs 0.0098
Per-study prevalence: {"single": {"gse237183_glioma": 0.01}, "pair": {"gse237183_glioma": 0.0098}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | 0.274 [0.187, 0.362] | 27.4 / 27.4 | 0.274 | - | 0.334 [0.232, 0.440] | 34.2 / 34.2 | 0.334 | - | 0.116 / 0.254 |
| ho_pair | 40 | 0.274 [0.187, 0.362] | 27.4 / 27.4 | 0.274 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.021 [0.011, 0.032] | 2.1 / 34.2 | 0.334 | +0.313 [+0.208, +0.425], p_Holm 0.000 | 0.116 / 0.015 |
| ho_pair_guarded | 40 | 0.274 [0.187, 0.362] | 27.4 / 27.4 | 0.274 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.048 [0.037, 0.060] | 4.9 / 34.2 | 0.334 | +0.285 [+0.188, +0.387], p_Holm 0.000 | 0.116 / 0.035 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | 0.274 [0.187, 0.362] | 27.4 / 27.4 | 0.274 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.270 [0.176, 0.375] | 27.7 / 34.2 | 0.334 | +0.063 [+0.027, +0.107], p_Holm 0.000 | 0.116 / 0.224 |
| random (chance reference, 20 seeds) | 40 | 0.017 [0.013, 0.022] | 1.7 / 27.4 | 0.274 | +0.256 [+0.168, +0.347], p_Holm 0.000 | 0.016 [0.010, 0.022] | 1.6 / 34.2 | 0.334 | +0.318 [+0.215, +0.426], p_Holm 0.000 | 0.013 / 0.012 |

### Table 2 (secondary, same score files): studies gse237183_glioma; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0100, pairs 0.0098

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 0.287 / 0.344 | 0.312 / 0.381 | 0.634 / 0.649 | 0.503 / 0.509 | 0.630 / 0.658 | 0.271 / 0.326 | 0.025 / 0.022 | 0.074 / 0.159 |
| ho_pair | 0.287 / 0.014 | 0.312 / 0.016 | 0.634 / 0.116 | 0.503 / 0.062 | 0.630 / 0.503 | 0.271 / 0.019 | 0.025 / -0.005 | 0.074 / 0.012 |
| ho_pair_guarded | 0.287 / 0.031 | 0.312 / 0.061 | 0.634 / 0.251 | 0.503 / 0.101 | 0.630 / 0.512 | 0.271 / 0.036 | 0.025 / -0.035 | 0.074 / 0.034 |
| ours: support-fit ordering (pairs = mean of singles) | 0.287 / 0.285 | 0.312 / 0.317 | 0.634 / 0.585 | 0.503 / 0.452 | 0.630 / 0.633 | 0.271 / 0.276 | 0.025 / 0.054 | 0.074 / 0.139 |
| random (chance reference, 20 seeds) | 0.007 / 0.005 | 0.014 / 0.009 | 0.113 / 0.102 | 0.050 / 0.046 | 0.500 / 0.500 | 0.013 / 0.012 | 0.000 / 0.000 | 0.013 / 0.012 |

Label set per study: gse237183_glioma: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.


# E2: pair rankings, labels B_decomp (decomp labels guarded by G_known > 0); ours column = joint pair gain

### Table 1: studies gse237183_glioma; RFF labels at 95%, baseline B_decomp; mean prevalence singles nan, pairs 0.0138
Per-study prevalence: {"single": {"gse237183_glioma": NaN}, "pair": {"gse237183_glioma": 0.0138}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | - | nan / nan | nan | - | 0.018 [0.014, 0.021] | 1.3 / 1.3 | 0.018 | - | - / 0.019 |
| ho_pair | 40 | - | nan / nan | nan | - | 0.018 [0.014, 0.023] | 1.3 / 1.3 | 0.018 | -0.001 [-0.002, +0.001], p_Holm 1.000 | - / 0.018 |
| ho_pair_guarded | 40 | - | nan / nan | nan | - | 0.018 [0.014, 0.022] | 1.3 / 1.3 | 0.018 | +0.000 [-0.001, +0.001], p_Holm 1.000 | - / 0.014 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | - | nan / nan | nan | - | 0.018 [0.014, 0.021] | 1.3 / 1.3 | 0.018 | -0.000 [-0.001, +0.001], p_Holm 1.000 | - / 0.018 |
| random (chance reference, 20 seeds) | 40 | - | nan / nan | nan | - | 0.018 [0.015, 0.021] | 1.3 / 1.3 | 0.018 | -0.000 [-0.001, +0.002], p_Holm 1.000 | - / 0.015 |

### Table 2 (secondary, same score files): studies gse237183_glioma; RFF labels at 95%, baseline B_decomp; mean prevalence singles nan, pairs 0.0138

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | - / 0.006 | - / 0.011 | - / 0.161 | - / 0.063 | - / 0.502 | - / 0.018 | - / -0.023 | - / 0.018 |
| ho_pair | - / 0.005 | - / 0.010 | - / 0.145 | - / 0.061 | - / 0.501 | - / 0.016 | - / 0.002 | - / 0.017 |
| ho_pair_guarded | - / 0.004 | - / 0.007 | - / 0.113 | - / 0.054 | - / 0.500 | - / 0.015 | - / -0.002 | - / 0.013 |
| ours: support-fit ordering (pairs = mean of singles) | - / 0.006 | - / 0.011 | - / 0.153 | - / 0.061 | - / 0.502 | - / 0.017 | - / -0.019 | - / 0.018 |
| random (chance reference, 20 seeds) | - / 0.005 | - / 0.011 | - / 0.132 | - / 0.058 | - / 0.500 | - / 0.014 | - / 0.000 | - / 0.014 |

Label set per study: gse237183_glioma: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

<!-- E1-MATCHED-BEGIN -->

## E1 matched budget (existing receipts and labels; B_known, 95%)

The pool-AP rows above score the beam's 64 fitted sets and leave every other pool member tied at the bottom; they are 'pool AP, 64 scored sets', not a ranking failure of the beam's proposals. Matched budget: precision over the sets the beam proposed that carry a label (listed singles; listed pairs inside the 2,000-pair pool) against the top-n of exhaustive support-fit ordering with n = the beam's count of the same kind, and the top 64.

| study | kind | targets | beam: mean labelled proposals | beam: mean positives | beam precision | exhaustive top-n (matched): mean positives | exhaustive precision (matched) | exhaustive top-64: mean positives | precision@64 | paired difference in precision (beam minus exhaustive, matched n) |
|---|---|---|---|---|---|---|---|---|---|---|
| gse237183_glioma | single | 40 | 26.0 | 0.99 | 0.038 | 1.48 | 0.057 | 2.27 | 0.035 | -0.019 [-0.030, -0.008] |
| gse237183_glioma | pair | 40 | 0.0 | 0.01 | 0.025 | 0.01 | 0.075 | 3.66 | 0.057 | -0.050 [-0.125, +0.000] |

## B_decomp label prevalence (guarded by G_known > 0), pairs only

| study | mean prevalence in the 2,000-pair pool | mean positive pairs per (target, split) | (target, split) cells | cells with no positive pair |
|---|---|---|---|---|
| gse237183_glioma | 0.0138 | 27.7 | 400 | 4 |

At the prevalences and positive-pair counts in this table the B_decomp label is too rare to rank on at these cohort sizes; an all-chance E2 on B_decomp is a power statement, not evidence about the rankings.

<!-- E1-MATCHED-END -->

## E1E2_nakajima_luad_b3

# E1 (beam K=64 vs exhaustive support-fit scoring of the same pool; B_known, level 95; ours column = beam64)

### Table 1: studies nakajima_luad_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0343, pairs 0.0489
Per-study prevalence: {"single": {"nakajima_luad_b3": 0.0343}, "pair": {"nakajima_luad_b3": 0.0489}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied) | 37 | 0.290 [0.214, 0.370] | 8.4 / 8.4 | 0.290 | - | 0.162 [0.095, 0.236] | 3.3 / 3.3 | 0.162 | - | 0.281 / 0.328 |
| ours: support-fit ordering (pairs = mean of singles) | 37 | 0.365 [0.264, 0.476] | 10.6 / 8.4 | 0.290 | -0.076 [-0.115, -0.041], p_Holm 1.000 | 0.457 [0.354, 0.565] | 9.3 / 3.3 | 0.162 | -0.298 [-0.428, -0.176], p_Holm 1.000 | 0.330 / 0.495 |
| ours: support-fit ordering, joint pair gain (suppl.) | 37 | 0.365 [0.264, 0.476] | 10.6 / 8.4 | 0.290 | -0.076 [-0.115, -0.041], p_Holm 1.000 | 0.480 [0.371, 0.595] | 9.8 / 3.3 | 0.162 | -0.321 [-0.458, -0.191], p_Holm 1.000 | 0.330 / 0.502 |

### Table 2 (secondary, same score files): studies nakajima_luad_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0343, pairs 0.0489

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied) | 0.178 / 0.016 | 0.192 / 0.026 | 0.648 / 0.456 | 0.591 / 0.380 | 0.633 / 0.555 | 0.293 / 0.158 | 0.138 / 0.101 | 0.237 / 0.303 |
| ours: support-fit ordering (pairs = mean of singles) | 0.267 / 0.313 | 0.288 / 0.335 | 0.719 / 0.795 | 0.615 / 0.702 | 0.670 / 0.696 | 0.366 / 0.434 | 0.121 / 0.162 | 0.283 / 0.401 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.267 / 0.319 | 0.288 / 0.336 | 0.719 / 0.791 | 0.615 / 0.722 | 0.670 / 0.713 | 0.366 / 0.469 | 0.121 / 0.149 | 0.283 / 0.406 |

Label set per study: nakajima_luad_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.


# E2: pair rankings, labels B_known (decomp labels guarded by G_known > 0); ours column = joint pair gain

### Table 1: studies nakajima_luad_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0343, pairs 0.0489
Per-study prevalence: {"single": {"nakajima_luad_b3": 0.0343}, "pair": {"nakajima_luad_b3": 0.0489}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 37 | 0.365 [0.264, 0.476] | 10.6 / 10.6 | 0.365 | - | 0.480 [0.371, 0.595] | 9.8 / 9.8 | 0.480 | - | 0.330 / 0.502 |
| ho_pair | 37 | 0.365 [0.264, 0.476] | 10.6 / 10.6 | 0.365 | -0.000 [-0.000, -0.000], p_Holm 1.000 | 0.064 [0.042, 0.088] | 1.3 / 9.8 | 0.480 | +0.419 [+0.302, +0.541], p_Holm 0.000 | 0.330 / 0.064 |
| ho_pair_guarded | 37 | 0.365 [0.264, 0.476] | 10.6 / 10.6 | 0.365 | -0.000 [-0.000, -0.000], p_Holm 1.000 | 0.147 [0.114, 0.180] | 3.0 / 9.8 | 0.480 | +0.335 [+0.231, +0.445], p_Holm 0.000 | 0.330 / 0.121 |
| ours: support-fit ordering (pairs = mean of singles) | 37 | 0.365 [0.264, 0.476] | 10.6 / 10.6 | 0.365 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.457 [0.354, 0.565] | 9.3 / 9.8 | 0.480 | +0.023 [+0.010, +0.038], p_Holm 0.000 | 0.330 / 0.495 |
| random (chance reference, 20 seeds) | 37 | 0.043 [0.029, 0.057] | 1.2 / 10.6 | 0.365 | +0.325 [+0.220, +0.438], p_Holm 0.000 | 0.067 [0.044, 0.092] | 1.4 / 9.8 | 0.480 | +0.416 [+0.299, +0.536], p_Holm 0.000 | 0.039 / 0.064 |

### Table 2 (secondary, same score files): studies nakajima_luad_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0343, pairs 0.0489

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 0.267 / 0.319 | 0.288 / 0.336 | 0.719 / 0.791 | 0.615 / 0.722 | 0.670 / 0.713 | 0.366 / 0.469 | 0.121 / 0.149 | 0.283 / 0.406 |
| ho_pair | 0.267 / 0.003 | 0.288 / 0.006 | 0.719 / 0.335 | 0.615 / 0.152 | 0.670 / 0.497 | 0.366 / 0.059 | 0.121 / -0.041 | 0.283 / 0.063 |
| ho_pair_guarded | 0.267 / 0.041 | 0.288 / 0.101 | 0.719 / 0.562 | 0.615 / 0.247 | 0.670 / 0.545 | 0.366 / 0.143 | 0.121 / 0.096 | 0.283 / 0.129 |
| ours: support-fit ordering (pairs = mean of singles) | 0.267 / 0.313 | 0.288 / 0.335 | 0.719 / 0.795 | 0.615 / 0.702 | 0.670 / 0.696 | 0.366 / 0.434 | 0.121 / 0.162 | 0.283 / 0.401 |
| random (chance reference, 20 seeds) | 0.008 / 0.005 | 0.014 / 0.009 | 0.253 / 0.353 | 0.108 / 0.149 | 0.500 / 0.500 | 0.039 / 0.064 | -0.000 / 0.000 | 0.039 / 0.063 |

Label set per study: nakajima_luad_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.


# E2: pair rankings, labels B_decomp (decomp labels guarded by G_known > 0); ours column = joint pair gain

### Table 1: studies nakajima_luad_b3; RFF labels at 95%, baseline B_decomp; mean prevalence singles nan, pairs 0.0201
Per-study prevalence: {"single": {"nakajima_luad_b3": NaN}, "pair": {"nakajima_luad_b3": 0.0201}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | - | nan / nan | nan | - | 0.030 [0.022, 0.038] | 1.5 / 1.5 | 0.030 | - | - / 0.040 |
| ho_pair | 40 | - | nan / nan | nan | - | 0.027 [0.020, 0.034] | 1.4 / 1.5 | 0.030 | +0.002 [+0.000, +0.005], p_Holm 0.039 | - / 0.027 |
| ho_pair_guarded | 40 | - | nan / nan | nan | - | 0.030 [0.022, 0.038] | 1.5 / 1.5 | 0.030 | -0.000 [-0.004, +0.003], p_Holm 0.502 | - / 0.034 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | - | nan / nan | nan | - | 0.029 [0.022, 0.037] | 1.5 / 1.5 | 0.030 | +0.000 [-0.000, +0.001], p_Holm 0.357 | - / 0.038 |
| random (chance reference, 20 seeds) | 40 | - | nan / nan | nan | - | 0.024 [0.019, 0.029] | 1.2 / 1.5 | 0.030 | +0.006 [+0.003, +0.009], p_Holm 0.000 | - / 0.021 |

### Table 2 (secondary, same score files): studies nakajima_luad_b3; RFF labels at 95%, baseline B_decomp; mean prevalence singles nan, pairs 0.0201

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | - / 0.008 | - / 0.012 | - / 0.253 | - / 0.113 | - / 0.507 | - / 0.034 | - / -0.014 | - / 0.034 |
| ho_pair | - / 0.004 | - / 0.010 | - / 0.200 | - / 0.089 | - / 0.503 | - / 0.027 | - / 0.022 | - / 0.029 |
| ho_pair_guarded | - / 0.007 | - / 0.013 | - / 0.227 | - / 0.111 | - / 0.506 | - / 0.032 | - / -0.004 | - / 0.032 |
| ours: support-fit ordering (pairs = mean of singles) | - / 0.008 | - / 0.018 | - / 0.246 | - / 0.113 | - / 0.508 | - / 0.035 | - / -0.031 | - / 0.037 |
| random (chance reference, 20 seeds) | - / 0.005 | - / 0.011 | - / 0.177 | - / 0.075 | - / 0.500 | - / 0.020 | - / 0.000 | - / 0.021 |

Label set per study: nakajima_luad_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

<!-- E1-MATCHED-BEGIN -->

## E1 matched budget (existing receipts and labels; B_known, 95%)

The pool-AP rows above score the beam's 64 fitted sets and leave every other pool member tied at the bottom; they are 'pool AP, 64 scored sets', not a ranking failure of the beam's proposals. Matched budget: precision over the sets the beam proposed that carry a label (listed singles; listed pairs inside the 2,000-pair pool) against the top-n of exhaustive support-fit ordering with n = the beam's count of the same kind, and the top 64.

| study | kind | targets | beam: mean labelled proposals | beam: mean positives | beam precision | exhaustive top-n (matched): mean positives | exhaustive precision (matched) | exhaustive top-64: mean positives | precision@64 | paired difference in precision (beam minus exhaustive, matched n) |
|---|---|---|---|---|---|---|---|---|---|---|
| nakajima_luad_b3 | single | 40 | 26.0 | 5.19 | 0.200 | 6.35 | 0.244 | 12.58 | 0.197 | -0.044 [-0.073, -0.023] |
| nakajima_luad_b3 | pair | 40 | 0.1 | 0.04 | 0.039 | 0.05 | 0.139 | 15.59 | 0.244 | -0.100 [-0.200, -0.025] |

## B_decomp label prevalence (guarded by G_known > 0), pairs only

| study | mean prevalence in the 2,000-pair pool | mean positive pairs per (target, split) | (target, split) cells | cells with no positive pair |
|---|---|---|---|---|
| nakajima_luad_b3 | 0.0201 | 40.1 | 400 | 8 |

At the prevalences and positive-pair counts in this table the B_decomp label is too rare to rank on at these cohort sizes; an all-chance E2 on B_decomp is a power statement, not evidence about the rankings.

<!-- E1-MATCHED-END -->

## E1E2_gse307534_lung_precursor_b3

# E1 (beam K=64 vs exhaustive support-fit scoring of the same pool; B_known, level 95; ours column = beam64)

### Table 1: studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0160, pairs 0.0183
Per-study prevalence: {"single": {"gse307534_lung_precursor_b3": 0.016}, "pair": {"gse307534_lung_precursor_b3": 0.0183}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied) | 40 | 0.229 [0.154, 0.311] | 14.3 / 14.3 | 0.229 | - | 0.025 [0.011, 0.041] | 1.4 / 1.4 | 0.025 | - | 0.095 / 0.056 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | 0.278 [0.192, 0.371] | 17.4 / 14.3 | 0.229 | -0.050 [-0.071, -0.030], p_Holm 1.000 | 0.314 [0.210, 0.425] | 17.1 / 1.4 | 0.025 | -0.290 [-0.401, -0.186], p_Holm 1.000 | 0.137 / 0.277 |
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | 0.278 [0.192, 0.371] | 17.4 / 14.3 | 0.229 | -0.050 [-0.071, -0.030], p_Holm 1.000 | 0.329 [0.223, 0.442] | 18.0 / 1.4 | 0.025 | -0.305 [-0.419, -0.197], p_Holm 1.000 | 0.137 / 0.302 |

### Table 2 (secondary, same score files): studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0160, pairs 0.0183

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied) | 0.203 / 0.004 | 0.206 / 0.006 | 0.504 / 0.111 | 0.471 / 0.087 | 0.604 / 0.499 | 0.226 / 0.020 | 0.088 / 0.102 | 0.063 / 0.041 |
| ours: support-fit ordering (pairs = mean of singles) | 0.265 / 0.293 | 0.286 / 0.319 | 0.614 / 0.593 | 0.523 / 0.484 | 0.634 / 0.644 | 0.284 / 0.306 | 0.071 / 0.080 | 0.098 / 0.170 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.265 / 0.317 | 0.286 / 0.338 | 0.614 / 0.648 | 0.523 / 0.517 | 0.634 / 0.654 | 0.284 / 0.325 | 0.071 / 0.087 | 0.098 / 0.184 |

Label set per study: gse307534_lung_precursor_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.


# E2: pair rankings, labels B_known (decomp labels guarded by G_known > 0); ours column = joint pair gain

### Table 1: studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0160, pairs 0.0183
Per-study prevalence: {"single": {"gse307534_lung_precursor_b3": 0.016}, "pair": {"gse307534_lung_precursor_b3": 0.0183}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | 0.278 [0.192, 0.371] | 17.4 / 17.4 | 0.278 | - | 0.329 [0.223, 0.442] | 18.0 / 18.0 | 0.329 | - | 0.137 / 0.302 |
| ho_pair | 40 | 0.278 [0.192, 0.371] | 17.4 / 17.4 | 0.278 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.028 [0.016, 0.044] | 1.6 / 18.0 | 0.329 | +0.301 [+0.192, +0.417], p_Holm 0.000 | 0.137 / 0.029 |
| ho_pair_guarded | 40 | 0.278 [0.192, 0.371] | 17.4 / 17.4 | 0.278 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.074 [0.052, 0.098] | 4.0 / 18.0 | 0.329 | +0.256 [+0.164, +0.353], p_Holm 0.000 | 0.137 / 0.067 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | 0.278 [0.192, 0.371] | 17.4 / 17.4 | 0.278 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.314 [0.210, 0.425] | 17.1 / 18.0 | 0.329 | +0.016 [-0.008, +0.035], p_Holm 0.083 | 0.137 / 0.277 |
| random (chance reference, 20 seeds) | 40 | 0.024 [0.016, 0.034] | 1.5 / 17.4 | 0.278 | +0.255 [+0.167, +0.350], p_Holm 0.000 | 0.024 [0.013, 0.038] | 1.3 / 18.0 | 0.329 | +0.306 [+0.198, +0.419], p_Holm 0.000 | 0.019 / 0.022 |

### Table 2 (secondary, same score files): studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0160, pairs 0.0183

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 0.265 / 0.317 | 0.286 / 0.338 | 0.614 / 0.648 | 0.523 / 0.517 | 0.634 / 0.654 | 0.284 / 0.325 | 0.071 / 0.087 | 0.098 / 0.184 |
| ho_pair | 0.265 / 0.013 | 0.286 / 0.016 | 0.614 / 0.191 | 0.523 / 0.078 | 0.634 / 0.503 | 0.284 / 0.027 | 0.071 / -0.015 | 0.098 / 0.027 |
| ho_pair_guarded | 0.265 / 0.046 | 0.286 / 0.115 | 0.614 / 0.381 | 0.523 / 0.138 | 0.634 / 0.521 | 0.284 / 0.061 | 0.071 / 0.017 | 0.098 / 0.072 |
| ours: support-fit ordering (pairs = mean of singles) | 0.265 / 0.293 | 0.286 / 0.319 | 0.614 / 0.593 | 0.523 / 0.484 | 0.634 / 0.644 | 0.284 / 0.306 | 0.071 / 0.080 | 0.098 / 0.170 |
| random (chance reference, 20 seeds) | 0.007 / 0.005 | 0.014 / 0.010 | 0.142 / 0.138 | 0.064 / 0.063 | 0.500 / 0.500 | 0.019 / 0.021 | -0.000 / 0.000 | 0.019 / 0.021 |

Label set per study: gse307534_lung_precursor_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.


# E2: pair rankings, labels B_decomp (decomp labels guarded by G_known > 0); ours column = joint pair gain

### Table 1: studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_decomp; mean prevalence singles nan, pairs 0.0147
Per-study prevalence: {"single": {"gse307534_lung_precursor_b3": NaN}, "pair": {"gse307534_lung_precursor_b3": 0.0147}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | 40 | - | nan / nan | nan | - | 0.023 [0.017, 0.029] | 1.5 / 1.5 | 0.023 | - | - / 0.026 |
| ho_pair | 40 | - | nan / nan | nan | - | 0.018 [0.013, 0.025] | 1.2 / 1.5 | 0.023 | +0.004 [+0.002, +0.007], p_Holm 0.002 | - / 0.019 |
| ho_pair_guarded | 40 | - | nan / nan | nan | - | 0.020 [0.015, 0.027] | 1.4 / 1.5 | 0.023 | +0.002 [+0.000, +0.004], p_Holm 0.013 | - / 0.022 |
| ours: support-fit ordering (pairs = mean of singles) | 40 | - | nan / nan | nan | - | 0.023 [0.018, 0.029] | 1.6 / 1.5 | 0.023 | -0.000 [-0.002, +0.001], p_Holm 0.670 | - / 0.028 |
| random (chance reference, 20 seeds) | 40 | - | nan / nan | nan | - | 0.018 [0.014, 0.023] | 1.2 / 1.5 | 0.023 | +0.004 [+0.002, +0.007], p_Holm 0.000 | - / 0.014 |

### Table 2 (secondary, same score files): studies gse307534_lung_precursor_b3; RFF labels at 95%, baseline B_decomp; mean prevalence singles nan, pairs 0.0147

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering, joint pair gain (suppl.) | - / 0.008 | - / 0.013 | - / 0.178 | - / 0.089 | - / 0.504 | - / 0.022 | - / -0.004 | - / 0.021 |
| ho_pair | - / 0.003 | - / 0.007 | - / 0.128 | - / 0.059 | - / 0.501 | - / 0.016 | - / -0.009 | - / 0.017 |
| ho_pair_guarded | - / 0.006 | - / 0.010 | - / 0.140 | - / 0.066 | - / 0.503 | - / 0.020 | - / 0.002 | - / 0.022 |
| ours: support-fit ordering (pairs = mean of singles) | - / 0.008 | - / 0.014 | - / 0.213 | - / 0.102 | - / 0.503 | - / 0.021 | - / 0.001 | - / 0.022 |
| random (chance reference, 20 seeds) | - / 0.005 | - / 0.010 | - / 0.124 | - / 0.055 | - / 0.500 | - / 0.015 | - / 0.000 | - / 0.014 |

Label set per study: gse307534_lung_precursor_b3: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

<!-- E1-MATCHED-BEGIN -->

## E1 matched budget (existing receipts and labels; B_known, 95%)

The pool-AP rows above score the beam's 64 fitted sets and leave every other pool member tied at the bottom; they are 'pool AP, 64 scored sets', not a ranking failure of the beam's proposals. Matched budget: precision over the sets the beam proposed that carry a label (listed singles; listed pairs inside the 2,000-pair pool) against the top-n of exhaustive support-fit ordering with n = the beam's count of the same kind, and the top 64.

| study | kind | targets | beam: mean labelled proposals | beam: mean positives | beam precision | exhaustive top-n (matched): mean positives | exhaustive precision (matched) | exhaustive top-64: mean positives | precision@64 | paired difference in precision (beam minus exhaustive, matched n) |
|---|---|---|---|---|---|---|---|---|---|---|
| gse307534_lung_precursor_b3 | single | 40 | 26.0 | 1.34 | 0.052 | 2.08 | 0.080 | 3.45 | 0.054 | -0.028 [-0.048, -0.013] |
| gse307534_lung_precursor_b3 | pair | 40 | 0.1 | 0.00 | 0.000 | 0.02 | 0.075 | 5.13 | 0.080 | -0.075 [-0.150, -0.013] |

## B_decomp label prevalence (guarded by G_known > 0), pairs only

| study | mean prevalence in the 2,000-pair pool | mean positive pairs per (target, split) | (target, split) cells | cells with no positive pair |
|---|---|---|---|---|
| gse307534_lung_precursor_b3 | 0.0147 | 29.4 | 400 | 14 |

At the prevalences and positive-pair counts in this table the B_decomp label is too rare to rank on at these cohort sizes; an all-chance E2 on B_decomp is a power statement, not evidence about the rankings.

<!-- E1-MATCHED-END -->

## budget_table

### E3 / budget: confirmed relations among the K fitted sets (labelled pool members passing the v05 label), mean over targets and splits [95% target bootstrap]

| study | selector | K=16 | K=32 | K=64 | K=128 | K=256 | labelled share of K at K=64 |
|---|---|---|---|---|---|---|---|
| gse237183_glioma (8 targets) | learned | 0.54 [0.08, 1.12] | 0.54 [0.08, 1.12] | 0.58 [0.08, 1.21] | 0.67 [0.17, 1.25] | 0.79 [0.29, 1.38] | 0.41 |
| gse237183_glioma (8 targets) | marginal_screen | 0.83 [0.33, 1.38] | 1.00 [0.54, 1.46] | 1.29 [0.75, 1.79] | 1.58 [1.00, 2.12] | 2.33 [1.58, 3.21] | 0.41 |
| gse237183_glioma (8 targets) | random | 0.04 [0.00, 0.12] | 0.12 [0.00, 0.29] | 0.04 [0.00, 0.12] | 0.17 [0.00, 0.42] | 0.50 [0.17, 0.83] | 0.41 |
| nakajima_luad_b3 (8 targets) | learned | 4.12 [3.25, 5.00] | 8.08 [6.37, 9.83] | 15.67 [11.79, 19.54] | 22.38 [17.29, 27.67] | 29.17 [23.83, 34.29] | 0.41 |
| nakajima_luad_b3 (8 targets) | marginal_screen | 4.46 [3.50, 5.33] | 8.17 [6.42, 9.92] | 17.21 [13.25, 21.17] | 33.17 [25.87, 40.50] | 50.54 [40.08, 59.67] | 0.41 |
| nakajima_luad_b3 (8 targets) | random | 0.67 [0.38, 1.00] | 0.92 [0.50, 1.33] | 2.54 [2.12, 3.00] | 5.08 [4.04, 6.29] | 11.38 [9.92, 12.92] | 0.41 |
| gse307534_lung_precursor_b3 (8 targets) | learned | 0.21 [0.00, 0.50] | 0.21 [0.00, 0.46] | 0.25 [0.00, 0.58] | 0.29 [0.00, 0.67] | 0.33 [0.00, 0.79] | 0.41 |
| gse307534_lung_precursor_b3 (8 targets) | marginal_screen | 0.50 [0.00, 1.21] | 0.71 [0.17, 1.29] | 0.83 [0.29, 1.46] | 1.17 [0.42, 1.92] | 1.75 [0.71, 2.79] | 0.40 |
| gse307534_lung_precursor_b3 (8 targets) | random | 0.00 [0.00, 0.00] | 0.12 [0.00, 0.29] | 0.25 [0.04, 0.50] | 0.21 [0.04, 0.46] | 0.58 [0.08, 1.17] | 0.41 |

## b3_fewshot_gse237183_glioma_nakajima_luad_b3_gse307534_lung_precursor_b3

### B3 few-shot, gse237183_glioma, singles: AP against support patients (first 8 targets, 5 draws per n; 95% bootstrap over targets)

| method | n=2 | n=3 | n=5 | n=8 | full-data AP (same targets) | smallest n at 80% of full-data AP |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.081 [0.022, 0.150] | 0.155 [0.063, 0.255] | 0.263 [0.060, 0.495] | 0.147 [0.054, 0.250] | 0.274 | 5 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.081 [0.022, 0.150] | 0.155 [0.063, 0.255] | 0.263 [0.060, 0.495] | 0.147 [0.054, 0.250] | 0.274 | 5 |
| linear interaction screen | 0.201 [0.069, 0.349] | 0.187 [0.060, 0.334] | 0.341 [0.106, 0.613] | 0.195 [0.048, 0.381] | 0.358 | 5 |
| NID | 0.053 [0.017, 0.092] | 0.156 [0.058, 0.265] | 0.338 [0.103, 0.605] | 0.190 [0.045, 0.371] | 0.423 | >8 |
| MISTy-style forest (pairs derived) | 0.085 [0.015, 0.173] | 0.053 [0.019, 0.090] | 0.123 [0.019, 0.247] | 0.097 [0.033, 0.169] | 0.107 | 5 |
| random (chance reference, 20 seeds) | 0.016 [0.007, 0.028] | 0.026 [0.014, 0.037] | 0.014 [0.009, 0.020] | 0.020 [0.011, 0.033] | 0.014 | 2 |

### B3 few-shot, gse237183_glioma, pairs: AP against support patients (first 8 targets, 5 draws per n; 95% bootstrap over targets)

| method | n=2 | n=3 | n=5 | n=8 | full-data AP (same targets) | smallest n at 80% of full-data AP |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.084 [0.025, 0.161] | 0.152 [0.045, 0.275] | 0.249 [0.026, 0.520] | 0.146 [0.026, 0.310] | 0.244 | 5 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.102 [0.028, 0.206] | 0.155 [0.045, 0.287] | 0.321 [0.083, 0.583] | 0.176 [0.034, 0.346] | 0.325 | 5 |
| linear interaction screen | 0.019 [0.009, 0.033] | 0.020 [0.011, 0.030] | 0.015 [0.011, 0.019] | 0.012 [0.006, 0.018] | 0.016 | 2 |
| NID | 0.039 [0.019, 0.064] | 0.072 [0.030, 0.127] | 0.166 [0.030, 0.354] | 0.130 [0.013, 0.282] | 0.208 | >8 |
| MISTy-style forest (pairs derived) | 0.022 [0.010, 0.036] | 0.033 [0.017, 0.051] | 0.041 [0.014, 0.077] | 0.050 [0.021, 0.081] | 0.085 | >8 |
| random (chance reference, 20 seeds) | 0.019 [0.007, 0.033] | 0.019 [0.011, 0.027] | 0.013 [0.006, 0.022] | 0.011 [0.006, 0.018] | 0.009 | 2 |

### B3 few-shot, nakajima_luad_b3, singles: AP against support patients (first 8 targets, 5 draws per n; 95% bootstrap over targets)

| method | n=2 | n=3 | n=5 | n=8 | full-data AP (same targets) | smallest n at 80% of full-data AP |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.199 [0.159, 0.239] | 0.230 [0.175, 0.283] | 0.269 [0.214, 0.325] | 0.293 [0.242, 0.346] | 0.346 | 8 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.199 [0.159, 0.239] | 0.230 [0.175, 0.283] | 0.269 [0.214, 0.325] | 0.293 [0.242, 0.346] | 0.346 | 8 |
| linear interaction screen | 0.205 [0.184, 0.225] | 0.206 [0.178, 0.231] | 0.230 [0.196, 0.259] | 0.246 [0.209, 0.280] | 0.269 | 5 |
| NID | 0.164 [0.138, 0.190] | 0.175 [0.152, 0.196] | 0.191 [0.161, 0.216] | 0.194 [0.165, 0.221] | 0.226 | 5 |
| MISTy-style forest (pairs derived) | 0.213 [0.173, 0.254] | 0.235 [0.184, 0.284] | 0.254 [0.204, 0.308] | 0.265 [0.214, 0.320] | 0.280 | 3 |
| random (chance reference, 20 seeds) | 0.058 [0.052, 0.065] | 0.061 [0.055, 0.068] | 0.067 [0.058, 0.076] | 0.078 [0.067, 0.090] | 0.096 | 8 |

### B3 few-shot, nakajima_luad_b3, pairs: AP against support patients (first 8 targets, 5 draws per n; 95% bootstrap over targets)

| method | n=2 | n=3 | n=5 | n=8 | full-data AP (same targets) | smallest n at 80% of full-data AP |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.222 [0.176, 0.263] | 0.274 [0.208, 0.338] | 0.322 [0.254, 0.390] | 0.366 [0.306, 0.427] | 0.418 | 8 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.232 [0.185, 0.275] | 0.278 [0.211, 0.342] | 0.322 [0.254, 0.390] | 0.363 [0.303, 0.425] | 0.418 | 8 |
| linear interaction screen | 0.082 [0.068, 0.098] | 0.090 [0.076, 0.105] | 0.105 [0.089, 0.124] | 0.119 [0.101, 0.141] | 0.157 | >8 |
| NID | 0.133 [0.115, 0.152] | 0.164 [0.136, 0.194] | 0.186 [0.154, 0.216] | 0.196 [0.165, 0.227] | 0.234 | 8 |
| MISTy-style forest (pairs derived) | 0.150 [0.117, 0.184] | 0.192 [0.150, 0.237] | 0.219 [0.171, 0.267] | 0.248 [0.199, 0.303] | 0.286 | 8 |
| random (chance reference, 20 seeds) | 0.072 [0.061, 0.086] | 0.081 [0.070, 0.092] | 0.094 [0.079, 0.110] | 0.107 [0.088, 0.128] | 0.136 | >8 |

### B3 few-shot, gse307534_lung_precursor_b3, singles: AP against support patients (first 8 targets, 5 draws per n; 95% bootstrap over targets)

| method | n=2 | n=3 | n=5 | n=8 | full-data AP (same targets) | smallest n at 80% of full-data AP |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.081 [0.028, 0.166] | 0.062 [0.028, 0.106] | 0.130 [0.030, 0.260] | 0.124 [0.020, 0.280] | 0.170 | >8 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.081 [0.028, 0.166] | 0.062 [0.028, 0.106] | 0.130 [0.030, 0.260] | 0.124 [0.020, 0.280] | 0.170 | >8 |
| linear interaction screen | 0.192 [0.063, 0.345] | 0.095 [0.034, 0.164] | 0.218 [0.073, 0.414] | 0.156 [0.030, 0.323] | 0.170 | 2 |
| NID | 0.069 [0.023, 0.129] | 0.041 [0.016, 0.071] | 0.237 [0.075, 0.446] | 0.149 [0.032, 0.305] | 0.175 | 5 |
| MISTy-style forest (pairs derived) | 0.040 [0.023, 0.060] | 0.053 [0.021, 0.092] | 0.108 [0.046, 0.181] | 0.064 [0.015, 0.123] | 0.069 | 5 |
| random (chance reference, 20 seeds) | 0.031 [0.016, 0.051] | 0.022 [0.015, 0.031] | 0.016 [0.009, 0.027] | 0.018 [0.010, 0.028] | 0.015 | 2 |

### B3 few-shot, gse307534_lung_precursor_b3, pairs: AP against support patients (first 8 targets, 5 draws per n; 95% bootstrap over targets)

| method | n=2 | n=3 | n=5 | n=8 | full-data AP (same targets) | smallest n at 80% of full-data AP |
|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.123 [0.031, 0.227] | 0.063 [0.024, 0.118] | 0.091 [0.016, 0.215] | 0.107 [0.008, 0.286] | 0.175 | >8 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.094 [0.029, 0.192] | 0.069 [0.026, 0.129] | 0.120 [0.025, 0.248] | 0.134 [0.017, 0.328] | 0.141 | 5 |
| linear interaction screen | 0.030 [0.015, 0.048] | 0.020 [0.012, 0.030] | 0.014 [0.009, 0.019] | 0.011 [0.006, 0.018] | 0.011 | 2 |
| NID | 0.044 [0.019, 0.073] | 0.039 [0.016, 0.065] | 0.101 [0.034, 0.187] | 0.129 [0.024, 0.286] | 0.123 | 5 |
| MISTy-style forest (pairs derived) | 0.033 [0.017, 0.050] | 0.023 [0.013, 0.034] | 0.029 [0.010, 0.058] | 0.015 [0.008, 0.025] | 0.029 | 2 |
| random (chance reference, 20 seeds) | 0.029 [0.016, 0.045] | 0.022 [0.013, 0.034] | 0.013 [0.008, 0.020] | 0.013 [0.008, 0.020] | 0.009 | 2 |

Random is the chance reference. Support patients are nested across n within a draw; the three query patients are the same for every n of a draw. Labels: v05 label engine on that (support, query).

## fdr_table

### FDR of support-fit ordering under null targets (picks made on the real support; label rule applied to the null query)

| study | kind | K | real precision@K | FDR, patient-permuted target | FDR, candidate-scrambled | net precision (permuted) | net precision (scrambled) | targets | pooled picks |
|---|---|---|---|---|---|---|---|---|---|
| gse237183_glioma | single | 10 | 0.107 [0.076, 0.144] | 0.012 [0.009, 0.014] | 0.009 [0.006, 0.012] | 0.095 [0.064, 0.130] | 0.098 [0.066, 0.135] | 40 | 2000 |
| gse237183_glioma | single | 20 | 0.066 [0.047, 0.087] | 0.011 [0.009, 0.013] | 0.010 [0.007, 0.012] | 0.054 [0.036, 0.074] | 0.056 [0.038, 0.077] | 40 | 4000 |
| gse237183_glioma | pair | 10 | 0.200 [0.129, 0.286] | 0.012 [0.010, 0.014] | 0.005 [0.003, 0.007] | 0.188 [0.114, 0.269] | 0.195 [0.121, 0.279] | 40 | 2000 |
| gse237183_glioma | pair | 20 | 0.124 [0.077, 0.181] | 0.013 [0.011, 0.015] | 0.004 [0.002, 0.006] | 0.111 [0.063, 0.163] | 0.120 [0.072, 0.174] | 40 | 4000 |
| nakajima_luad_b3 | single | 10 | 0.293 [0.194, 0.394] | 0.024 [0.018, 0.031] | 0.004 [0.003, 0.006] | 0.269 [0.178, 0.364] | 0.289 [0.195, 0.389] | 40 | 2000 |
| nakajima_luad_b3 | single | 20 | 0.256 [0.159, 0.353] | 0.024 [0.018, 0.030] | 0.004 [0.003, 0.006] | 0.232 [0.146, 0.326] | 0.252 [0.160, 0.352] | 40 | 4000 |
| nakajima_luad_b3 | pair | 10 | 0.372 [0.267, 0.478] | 0.024 [0.019, 0.029] | 0.003 [0.002, 0.005] | 0.348 [0.246, 0.459] | 0.369 [0.263, 0.472] | 40 | 2000 |
| nakajima_luad_b3 | pair | 20 | 0.305 [0.204, 0.407] | 0.025 [0.021, 0.030] | 0.003 [0.001, 0.004] | 0.280 [0.185, 0.379] | 0.302 [0.203, 0.401] | 40 | 4000 |
| gse307534_lung_precursor_b3 | single | 10 | 0.128 [0.086, 0.172] | 0.020 [0.018, 0.023] | 0.013 [0.010, 0.016] | 0.108 [0.069, 0.149] | 0.115 [0.076, 0.158] | 40 | 2000 |
| gse307534_lung_precursor_b3 | single | 20 | 0.090 [0.056, 0.128] | 0.019 [0.017, 0.021] | 0.013 [0.010, 0.016] | 0.071 [0.039, 0.107] | 0.077 [0.046, 0.116] | 40 | 4000 |
| gse307534_lung_precursor_b3 | pair | 10 | 0.266 [0.175, 0.359] | 0.022 [0.020, 0.025] | 0.005 [0.003, 0.008] | 0.243 [0.154, 0.333] | 0.261 [0.175, 0.354] | 40 | 2000 |
| gse307534_lung_precursor_b3 | pair | 20 | 0.163 [0.106, 0.221] | 0.022 [0.020, 0.024] | 0.006 [0.004, 0.008] | 0.141 [0.085, 0.198] | 0.158 [0.105, 0.217] | 40 | 4000 |

## b2_pooled_3external

### Table 1: studies meylan2022_ccrcc__to__li2022_ccrcc, meylan2022_ccrcc__to__gse250163, meylan2022_ccrcc__to__zenodo14620362; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0539, pairs 0.0661
Per-study prevalence: {"single": {"meylan2022_ccrcc__to__li2022_ccrcc": 0.0565, "meylan2022_ccrcc__to__gse250163": 0.0564, "meylan2022_ccrcc__to__zenodo14620362": 0.0487}, "pair": {"meylan2022_ccrcc__to__li2022_ccrcc": 0.0693, "meylan2022_ccrcc__to__gse250163": 0.0715, "meylan2022_ccrcc__to__zenodo14620362": 0.0576}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 17/15/17 | 0.228 [0.166, 0.301] | 4.2 / 4.2 | 0.228 | - | 0.238 [0.169, 0.316] | 3.6 / 3.6 | 0.236 | - | 0.302 / 0.418 |
| ours: support-fit ordering, joint pair gain (suppl.) | 17/15/17 | 0.228 [0.166, 0.301] | 4.2 / 4.2 | 0.228 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.260 [0.181, 0.349] | 3.9 / 3.6 | 0.236 | -0.022 [-0.040, -0.007], p_Holm 1.000 | 0.302 / 0.439 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 17/15/17 | 0.286 [0.199, 0.390] | 5.3 / 4.2 | 0.228 | -0.057 [-0.131, -0.009], p_Holm 1.000 | 0.238 [0.169, 0.316] | 3.6 / 3.6 | 0.236 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.343 / 0.418 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 17/15/17 | 0.286 [0.199, 0.390] | 5.3 / 4.2 | 0.228 | -0.057 [-0.131, -0.009], p_Holm 1.000 | 0.260 [0.181, 0.349] | 3.9 / 3.6 | 0.236 | -0.022 [-0.040, -0.007], p_Holm 1.000 | 0.343 / 0.439 |
| random (chance reference, 20 seeds) | 17/15/17 | 0.074 [0.054, 0.097] | 1.4 / 4.2 | 0.228 | +0.153 [+0.090, +0.230], p_Holm 0.000 | 0.085 [0.057, 0.117] | 1.3 / 3.6 | 0.236 | +0.153 [+0.087, +0.229], p_Holm 0.000 | 0.070 / 0.082 |
| linear interaction screen | 17/15/17 | 0.286 [0.199, 0.390] | 5.3 / 4.2 | 0.228 | -0.057 [-0.131, -0.009], p_Holm 1.000 | 0.089 [0.061, 0.122] | 1.3 / 3.6 | 0.236 | +0.149 [+0.082, +0.225], p_Holm 0.000 | 0.343 / 0.088 |
| NID | 17/15/17 | 0.209 [0.152, 0.277] | 3.9 / 4.2 | 0.228 | +0.019 [-0.014, +0.049], p_Holm 1.000 | 0.187 [0.135, 0.246] | 2.8 / 3.6 | 0.236 | +0.050 [+0.017, +0.090], p_Holm 0.000 | 0.274 / 0.352 |
| MISTy-style forest (pairs derived) | 17/15/17 | 0.242 [0.174, 0.324] | 4.5 / 4.2 | 0.228 | -0.014 [-0.087, +0.055], p_Holm 1.000 | 0.168 [0.123, 0.213] | 2.5 / 3.6 | 0.236 | +0.070 [+0.020, +0.127], p_Holm 0.007 | 0.274 / 0.317 |
| Model-X knockoffs | 17/15/17 | 0.153 [0.116, 0.196] | 2.8 / 4.2 | 0.228 | +0.075 [+0.041, +0.116], p_Holm 0.000 | 0.090 [0.062, 0.122] | 1.4 / 3.6 | 0.236 | +0.147 [+0.082, +0.221], p_Holm 0.000 | 0.193 / 0.124 |
| GBM + SHAP interactions (native) | 17/15/17 | 0.236 [0.154, 0.342] | 4.4 / 4.2 | 0.228 | -0.008 [-0.082, +0.047], p_Holm 1.000 | 0.109 [0.080, 0.143] | 1.6 / 3.6 | 0.236 | +0.129 [+0.073, +0.197], p_Holm 0.000 | 0.218 / 0.175 |
| RuleFit (native) | 16/15/17 | 0.147 [0.099, 0.214] | 2.7 / 4.3 | 0.232 | +0.086 [+0.048, +0.129], p_Holm 0.000 | 0.088 [0.059, 0.121] | 1.3 / 3.6 | 0.241 | +0.154 [+0.091, +0.231], p_Holm 0.000 | 0.156 / 0.094 |
| stability selection | no receipts (timeout / not run) | | | | | | | | | |
| GBM + Friedman H | 17/15/17 | 0.074 [0.054, 0.097] | 1.4 / 4.2 | 0.228 | +0.153 [+0.090, +0.230], p_Holm 0.000 | 0.096 [0.069, 0.128] | 1.4 / 3.6 | 0.236 | +0.142 [+0.080, +0.213], p_Holm 0.000 | 0.080 / 0.103 |
| EBM (native) | 17/15/17 | 0.267 [0.195, 0.357] | 5.0 / 4.2 | 0.228 | -0.038 [-0.100, +0.003], p_Holm 1.000 | 0.212 [0.155, 0.272] | 3.2 / 3.6 | 0.236 | +0.026 [-0.011, +0.071], p_Holm 0.394 | 0.291 / 0.362 |
| iRF (native) **(incomplete: 19 of 60 target x split fits timed out)** | 3/14/17 | 0.169 [0.122, 0.213] | 3.2 / 4.2 | 0.220 | +0.052 [-0.015, +0.125], p_Holm 0.684 | 0.088 [0.057, 0.121] | 1.3 / 3.7 | 0.236 | +0.141 [+0.050, +0.228], p_Holm 0.006 | 0.204 / 0.091 |
| GBM (fair) | 17/15/17 | 0.194 [0.136, 0.265] | 3.6 / 4.2 | 0.228 | +0.034 [-0.009, +0.083], p_Holm 0.737 | 0.102 [0.073, 0.136] | 1.5 / 3.6 | 0.236 | +0.136 [+0.075, +0.206], p_Holm 0.000 | 0.195 / 0.188 |
| MISTy-style forest (fair) | 17/15/17 | 0.207 [0.148, 0.287] | 3.8 / 4.2 | 0.228 | +0.021 [-0.049, +0.083], p_Holm 1.000 | 0.110 [0.077, 0.147] | 1.7 / 3.6 | 0.236 | +0.127 [+0.068, +0.197], p_Holm 0.000 | 0.255 / 0.224 |
| EBM (fair) | 17/15/17 | 0.230 [0.151, 0.336] | 4.3 / 4.2 | 0.228 | -0.002 [-0.080, +0.059], p_Holm 1.000 | 0.098 [0.066, 0.133] | 1.5 / 3.6 | 0.236 | +0.140 [+0.079, +0.211], p_Holm 0.000 | 0.194 / 0.172 |
| iRF (fair) | no receipts (timeout / not run) | | | | | | | | | |
| RuleFit (fair) | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies meylan2022_ccrcc__to__li2022_ccrcc, meylan2022_ccrcc__to__gse250163, meylan2022_ccrcc__to__zenodo14620362; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0539, pairs 0.0661

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.202 / 0.080 | 0.228 / 0.130 | 0.774 / 0.614 | 0.539 / 0.516 | 0.570 / 0.574 | 0.203 / 0.224 | -0.001 / 0.016 | 0.225 / 0.381 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.202 / 0.093 | 0.228 / 0.139 | 0.774 / 0.678 | 0.539 / 0.553 | 0.570 / 0.588 | 0.203 / 0.253 | -0.001 / -0.003 | 0.225 / 0.384 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.215 / 0.080 | 0.249 / 0.130 | 0.880 / 0.614 | 0.671 / 0.516 | 0.603 / 0.574 | 0.266 / 0.224 | 0.018 / 0.016 | 0.249 / 0.381 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.215 / 0.093 | 0.249 / 0.139 | 0.880 / 0.678 | 0.671 / 0.553 | 0.603 / 0.588 | 0.266 / 0.253 | 0.018 / -0.003 | 0.249 / 0.384 |
| random (chance reference, 20 seeds) | 0.014 / 0.005 | 0.026 / 0.010 | 0.406 / 0.421 | 0.164 / 0.180 | 0.499 / 0.500 | 0.065 / 0.081 | 0.001 / 0.001 | 0.068 / 0.083 |
| linear interaction screen | 0.215 / 0.005 | 0.249 / 0.017 | 0.880 / 0.410 | 0.671 / 0.208 | 0.603 / 0.504 | 0.266 / 0.089 | 0.018 / 0.021 | 0.249 / 0.101 |
| NID | 0.177 / 0.060 | 0.198 / 0.100 | 0.796 / 0.718 | 0.574 / 0.544 | 0.556 / 0.554 | 0.177 / 0.186 | 0.043 / 0.054 | 0.195 / 0.323 |
| MISTy-style forest (pairs derived) | 0.139 / 0.050 | 0.165 / 0.064 | 0.715 / 0.640 | 0.605 / 0.476 | 0.582 / 0.548 | 0.222 / 0.169 | 0.092 / 0.057 | 0.199 / 0.257 |
| Model-X knockoffs | 0.129 / 0.008 | 0.160 / 0.018 | 0.763 / 0.507 | 0.551 / 0.317 | 0.538 / 0.505 | 0.140 / 0.091 | 0.002 / -0.003 | 0.147 / 0.126 |
| GBM + SHAP interactions (native) | 0.164 / 0.032 | 0.188 / 0.039 | 0.776 / 0.695 | 0.639 / 0.461 | 0.589 / 0.513 | 0.238 / 0.107 | 0.076 / -0.001 | 0.169 / 0.144 |
| RuleFit (native) | 0.087 / 0.010 | 0.125 / 0.015 | 0.651 / 0.529 | 0.440 / 0.302 | 0.540 / 0.502 | 0.144 / 0.088 | 0.023 / -0.003 | 0.117 / 0.090 |
| GBM + Friedman H | 0.016 / 0.008 | 0.028 / 0.027 | 0.455 / 0.524 | 0.175 / 0.220 | 0.501 / 0.511 | 0.069 / 0.102 | -0.002 / 0.003 | 0.072 / 0.117 |
| EBM (native) | 0.203 / 0.055 | 0.235 / 0.100 | 0.796 / 0.757 | 0.616 / 0.427 | 0.588 / 0.572 | 0.235 / 0.213 | 0.125 / 0.058 | 0.228 / 0.324 |
| iRF (native) | 0.084 / 0.008 | 0.141 / 0.012 | 0.787 / 0.560 | 0.593 / 0.278 | 0.548 / 0.503 | 0.157 / 0.087 | 0.047 / -0.001 | 0.172 / 0.083 |
| GBM (fair) | 0.146 / 0.016 | 0.178 / 0.025 | 0.754 / 0.678 | 0.531 / 0.434 | 0.562 / 0.516 | 0.186 / 0.112 | 0.001 / 0.002 | 0.154 / 0.170 |
| MISTy-style forest (fair) | 0.120 / 0.019 | 0.152 / 0.028 | 0.693 / 0.620 | 0.587 / 0.448 | 0.576 / 0.520 | 0.210 / 0.118 | 0.020 / -0.009 | 0.188 / 0.192 |
| EBM (fair) | 0.164 / 0.018 | 0.194 / 0.026 | 0.693 / 0.617 | 0.584 / 0.349 | 0.568 / 0.513 | 0.198 / 0.106 | 0.001 / -0.008 | 0.140 / 0.158 |

Label set per study: meylan2022_ccrcc__to__li2022_ccrcc: label engine v05; meylan2022_ccrcc__to__gse250163: label engine v05; meylan2022_ccrcc__to__zenodo14620362: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## b2_li2022_ccrcc

### Table 1: studies meylan2022_ccrcc__to__li2022_ccrcc; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0565, pairs 0.0693
Per-study prevalence: {"single": {"meylan2022_ccrcc__to__li2022_ccrcc": 0.0565}, "pair": {"meylan2022_ccrcc__to__li2022_ccrcc": 0.0693}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 17 | 0.244 [0.141, 0.374] | 4.3 / 4.3 | 0.244 | - | 0.228 [0.124, 0.353] | 3.3 / 3.3 | 0.228 | - | 0.218 / 0.318 |
| ours: support-fit ordering, joint pair gain (suppl.) | 17 | 0.244 [0.141, 0.374] | 4.3 / 4.3 | 0.244 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.253 [0.136, 0.394] | 3.7 / 3.3 | 0.228 | -0.025 [-0.055, -0.003], p_Holm 1.000 | 0.218 / 0.329 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 17 | 0.361 [0.219, 0.523] | 6.4 / 4.3 | 0.244 | -0.117 [-0.240, -0.018], p_Holm 1.000 | 0.228 [0.124, 0.353] | 3.3 / 3.3 | 0.228 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.253 / 0.318 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 17 | 0.361 [0.219, 0.523] | 6.4 / 4.3 | 0.244 | -0.117 [-0.240, -0.018], p_Holm 1.000 | 0.253 [0.136, 0.394] | 3.7 / 3.3 | 0.228 | -0.025 [-0.055, -0.003], p_Holm 1.000 | 0.253 / 0.329 |
| random (chance reference, 20 seeds) | 17 | 0.076 [0.042, 0.120] | 1.4 / 4.3 | 0.244 | +0.168 [+0.053, +0.306], p_Holm 0.000 | 0.084 [0.040, 0.150] | 1.2 / 3.3 | 0.228 | +0.144 [+0.041, +0.270], p_Holm 0.008 | 0.073 / 0.086 |
| linear interaction screen | 17 | 0.361 [0.219, 0.523] | 6.4 / 4.3 | 0.244 | -0.117 [-0.240, -0.018], p_Holm 1.000 | 0.085 [0.039, 0.151] | 1.2 / 3.3 | 0.228 | +0.144 [+0.040, +0.271], p_Holm 0.008 | 0.253 / 0.106 |
| NID | 17 | 0.241 [0.142, 0.363] | 4.3 / 4.3 | 0.244 | +0.003 [-0.053, +0.057], p_Holm 1.000 | 0.189 [0.107, 0.283] | 2.7 / 3.3 | 0.228 | +0.039 [+0.001, +0.090], p_Holm 0.168 | 0.171 / 0.288 |
| MISTy-style forest (pairs derived) | 17 | 0.275 [0.141, 0.442] | 4.9 / 4.3 | 0.244 | -0.032 [-0.195, +0.146], p_Holm 1.000 | 0.166 [0.090, 0.260] | 2.4 / 3.3 | 0.228 | +0.063 [-0.021, +0.162], p_Holm 0.498 | 0.213 / 0.286 |
| Model-X knockoffs | 17 | 0.155 [0.094, 0.225] | 2.7 / 4.3 | 0.244 | +0.089 [+0.021, +0.170], p_Holm 0.045 | 0.089 [0.043, 0.152] | 1.3 / 3.3 | 0.228 | +0.140 [+0.036, +0.267], p_Holm 0.008 | 0.126 / 0.095 |
| GBM + SHAP interactions (native) | 17 | 0.307 [0.157, 0.485] | 5.4 / 4.3 | 0.244 | -0.063 [-0.193, +0.041], p_Holm 1.000 | 0.112 [0.066, 0.173] | 1.6 / 3.3 | 0.228 | +0.116 [+0.027, +0.227], p_Holm 0.010 | 0.171 / 0.155 |
| RuleFit (native) | 16 | 0.187 [0.092, 0.318] | 3.2 / 4.4 | 0.258 | +0.072 [+0.020, +0.138], p_Holm 0.014 | 0.092 [0.045, 0.158] | 1.3 / 3.3 | 0.241 | +0.151 [+0.043, +0.282], p_Holm 0.008 | 0.138 / 0.093 |
| stability selection | no receipts (timeout / not run) | | | | | | | | | |
| GBM + Friedman H | 17 | 0.075 [0.041, 0.119] | 1.3 / 4.3 | 0.244 | +0.169 [+0.054, +0.308], p_Holm 0.000 | 0.103 [0.058, 0.164] | 1.5 / 3.3 | 0.228 | +0.125 [+0.036, +0.235], p_Holm 0.008 | 0.084 / 0.089 |
| EBM (native) | 17 | 0.327 [0.203, 0.468] | 5.8 / 4.3 | 0.244 | -0.083 [-0.198, +0.004], p_Holm 1.000 | 0.186 [0.102, 0.288] | 2.7 / 3.3 | 0.228 | +0.042 [-0.029, +0.129], p_Holm 0.780 | 0.165 / 0.288 |
| iRF (native) **(incomplete: 17 of 20 target x split fits timed out)** | 3 | 0.166 [0.063, 0.254] | 2.1 / 2.8 | 0.215 | +0.050 [-0.059, +0.245], p_Holm 1.000 | 0.097 [0.018, 0.147] | 1.0 / 2.1 | 0.195 | +0.100 [-0.005, +0.296], p_Holm 0.780 | 0.167 / 0.100 |
| GBM (fair) | 17 | 0.237 [0.140, 0.352] | 4.2 / 4.3 | 0.244 | +0.007 [-0.044, +0.081], p_Holm 1.000 | 0.104 [0.056, 0.173] | 1.5 / 3.3 | 0.228 | +0.124 [+0.030, +0.241], p_Holm 0.018 | 0.165 / 0.171 |
| MISTy-style forest (fair) | 17 | 0.245 [0.118, 0.406] | 4.3 / 4.3 | 0.244 | -0.002 [-0.164, +0.168], p_Holm 1.000 | 0.099 [0.047, 0.174] | 1.4 / 3.3 | 0.228 | +0.130 [+0.027, +0.256], p_Holm 0.032 | 0.176 / 0.165 |
| EBM (fair) | 17 | 0.314 [0.182, 0.459] | 5.6 / 4.3 | 0.244 | -0.071 [-0.186, +0.010], p_Holm 1.000 | 0.100 [0.053, 0.165] | 1.4 / 3.3 | 0.228 | +0.129 [+0.034, +0.245], p_Holm 0.008 | 0.118 / 0.171 |
| iRF (fair) | no receipts (timeout / not run) | | | | | | | | | |
| RuleFit (fair) | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies meylan2022_ccrcc__to__li2022_ccrcc; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0565, pairs 0.0693

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.299 / 0.085 | 0.317 / 0.154 | 0.765 / 0.529 | 0.460 / 0.402 | 0.555 / 0.570 | 0.178 / 0.224 | -0.039 / -0.015 | 0.162 / 0.300 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.299 / 0.107 | 0.317 / 0.152 | 0.765 / 0.529 | 0.460 / 0.447 | 0.555 / 0.589 | 0.178 / 0.260 | -0.039 / -0.005 | 0.162 / 0.276 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.314 / 0.085 | 0.332 / 0.154 | 0.941 / 0.529 | 0.619 / 0.402 | 0.637 / 0.570 | 0.335 / 0.224 | 0.003 / -0.015 | 0.174 / 0.300 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.314 / 0.107 | 0.332 / 0.152 | 0.941 / 0.529 | 0.619 / 0.447 | 0.637 / 0.589 | 0.335 / 0.260 | 0.003 / -0.005 | 0.174 / 0.276 |
| random (chance reference, 20 seeds) | 0.015 / 0.005 | 0.030 / 0.010 | 0.382 / 0.415 | 0.175 / 0.186 | 0.500 / 0.500 | 0.066 / 0.081 | 0.002 / 0.000 | 0.072 / 0.087 |
| linear interaction screen | 0.314 / 0.003 | 0.332 / 0.010 | 0.941 / 0.412 | 0.619 / 0.159 | 0.637 / 0.500 | 0.335 / 0.081 | 0.003 / 0.001 | 0.174 / 0.097 |
| NID | 0.241 / 0.058 | 0.258 / 0.119 | 0.824 / 0.647 | 0.521 / 0.387 | 0.560 / 0.556 | 0.187 / 0.191 | 0.014 / -0.001 | 0.132 / 0.265 |
| MISTy-style forest (pairs derived) | 0.174 / 0.075 | 0.196 / 0.084 | 0.706 / 0.647 | 0.498 / 0.423 | 0.609 / 0.554 | 0.270 / 0.176 | 0.007 / -0.008 | 0.166 / 0.232 |
| Model-X knockoffs | 0.172 / 0.006 | 0.190 / 0.013 | 0.706 / 0.435 | 0.382 / 0.290 | 0.528 / 0.503 | 0.121 / 0.088 | 0.008 / -0.004 | 0.104 / 0.105 |
| GBM + SHAP interactions (native) | 0.237 / 0.038 | 0.256 / 0.042 | 0.706 / 0.647 | 0.624 / 0.384 | 0.620 / 0.513 | 0.304 / 0.108 | 0.016 / -0.003 | 0.141 / 0.121 |
| RuleFit (native) | 0.123 / 0.014 | 0.202 / 0.019 | 0.625 / 0.537 | 0.360 / 0.265 | 0.558 / 0.506 | 0.184 / 0.098 | 0.021 / -0.003 | 0.113 / 0.089 |
| GBM + Friedman H | 0.017 / 0.011 | 0.035 / 0.046 | 0.447 / 0.494 | 0.165 / 0.137 | 0.501 / 0.520 | 0.068 / 0.120 | 0.005 / 0.002 | 0.078 / 0.123 |
| EBM (native) | 0.295 / 0.045 | 0.329 / 0.080 | 0.765 / 0.765 | 0.591 / 0.424 | 0.605 / 0.569 | 0.272 / 0.201 | 0.036 / -0.005 | 0.156 / 0.265 |
| iRF (native) | 0.028 / 0.006 | 0.080 / 0.008 | 0.667 / 0.667 | 0.472 / 0.236 | 0.542 / 0.505 | 0.154 / 0.100 | 0.115 / 0.002 | 0.200 / 0.080 |
| GBM (fair) | 0.236 / 0.016 | 0.261 / 0.025 | 0.765 / 0.588 | 0.491 / 0.398 | 0.576 / 0.520 | 0.211 / 0.119 | 0.036 / 0.003 | 0.138 / 0.162 |
| MISTy-style forest (fair) | 0.163 / 0.011 | 0.186 / 0.018 | 0.706 / 0.529 | 0.539 / 0.382 | 0.588 / 0.510 | 0.237 / 0.102 | -0.028 / -0.011 | 0.144 / 0.150 |
| EBM (fair) | 0.287 / 0.025 | 0.301 / 0.030 | 0.706 / 0.706 | 0.524 / 0.324 | 0.590 / 0.517 | 0.241 / 0.115 | 0.031 / -0.008 | 0.103 / 0.132 |

Label set per study: meylan2022_ccrcc__to__li2022_ccrcc: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## b2_gse250163

### Table 1: studies meylan2022_ccrcc__to__gse250163; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0564, pairs 0.0715
Per-study prevalence: {"single": {"meylan2022_ccrcc__to__gse250163": 0.0564}, "pair": {"meylan2022_ccrcc__to__gse250163": 0.0715}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 15 | 0.240 [0.145, 0.336] | 4.2 / 4.2 | 0.240 | - | 0.274 [0.162, 0.392] | 3.8 / 3.8 | 0.274 | - | 0.340 / 0.513 |
| ours: support-fit ordering, joint pair gain (suppl.) | 15 | 0.240 [0.145, 0.336] | 4.2 / 4.2 | 0.240 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.306 [0.177, 0.446] | 4.3 / 3.8 | 0.274 | -0.032 [-0.061, -0.006], p_Holm 1.000 | 0.340 / 0.533 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 15 | 0.274 [0.169, 0.384] | 4.9 / 4.2 | 0.240 | -0.033 [-0.097, +0.019], p_Holm 1.000 | 0.274 [0.162, 0.392] | 3.8 / 3.8 | 0.274 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.407 / 0.513 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 15 | 0.274 [0.169, 0.384] | 4.9 / 4.2 | 0.240 | -0.033 [-0.097, +0.019], p_Holm 1.000 | 0.306 [0.177, 0.446] | 4.3 / 3.8 | 0.274 | -0.032 [-0.061, -0.006], p_Holm 1.000 | 0.407 / 0.533 |
| random (chance reference, 20 seeds) | 15 | 0.083 [0.049, 0.124] | 1.5 / 4.2 | 0.240 | +0.155 [+0.058, +0.263], p_Holm 0.000 | 0.098 [0.050, 0.153] | 1.4 / 3.8 | 0.274 | +0.175 [+0.066, +0.300], p_Holm 0.000 | 0.078 / 0.091 |
| linear interaction screen | 15 | 0.274 [0.169, 0.384] | 4.9 / 4.2 | 0.240 | -0.033 [-0.097, +0.019], p_Holm 1.000 | 0.102 [0.052, 0.161] | 1.4 / 3.8 | 0.274 | +0.170 [+0.061, +0.298], p_Holm 0.000 | 0.407 / 0.107 |
| NID | 15 | 0.200 [0.128, 0.269] | 3.5 / 4.2 | 0.240 | +0.040 [+0.006, +0.080], p_Holm 0.066 | 0.203 [0.121, 0.296] | 2.8 / 3.8 | 0.274 | +0.071 [+0.017, +0.137], p_Holm 0.020 | 0.327 / 0.427 |
| MISTy-style forest (pairs derived) | 15 | 0.239 [0.124, 0.374] | 4.2 / 4.2 | 0.240 | +0.001 [-0.105, +0.083], p_Holm 1.000 | 0.178 [0.105, 0.251] | 2.5 / 3.8 | 0.274 | +0.095 [+0.027, +0.176], p_Holm 0.003 | 0.287 / 0.329 |
| Model-X knockoffs | 15 | 0.159 [0.108, 0.210] | 2.8 / 4.2 | 0.240 | +0.080 [+0.022, +0.143], p_Holm 0.033 | 0.105 [0.058, 0.157] | 1.5 / 3.8 | 0.274 | +0.168 [+0.065, +0.287], p_Holm 0.000 | 0.247 / 0.128 |
| GBM + SHAP interactions (native) | 15 | 0.224 [0.137, 0.334] | 4.0 / 4.2 | 0.240 | +0.015 [-0.069, +0.095], p_Holm 1.000 | 0.126 [0.076, 0.181] | 1.8 / 3.8 | 0.274 | +0.147 [+0.048, +0.270], p_Holm 0.000 | 0.247 / 0.212 |
| RuleFit (native) | 15 | 0.133 [0.090, 0.179] | 2.4 / 4.2 | 0.240 | +0.106 [+0.030, +0.190], p_Holm 0.028 | 0.101 [0.054, 0.155] | 1.4 / 3.8 | 0.274 | +0.171 [+0.066, +0.293], p_Holm 0.000 | 0.200 / 0.121 |
| stability selection | no receipts (timeout / not run) | | | | | | | | | |
| GBM + Friedman H | 15 | 0.084 [0.050, 0.124] | 1.5 / 4.2 | 0.240 | +0.154 [+0.057, +0.263], p_Holm 0.000 | 0.105 [0.057, 0.161] | 1.5 / 3.8 | 0.274 | +0.167 [+0.062, +0.288], p_Holm 0.000 | 0.091 / 0.131 |
| EBM (native) | 15 | 0.248 [0.159, 0.334] | 4.4 / 4.2 | 0.240 | -0.009 [-0.050, +0.036], p_Holm 1.000 | 0.249 [0.159, 0.344] | 3.5 / 3.8 | 0.274 | +0.025 [-0.016, +0.071], p_Holm 0.558 | 0.373 / 0.427 |
| iRF (native) | 14 | 0.173 [0.116, 0.227] | 3.2 / 4.6 | 0.245 | +0.071 [+0.003, +0.147], p_Holm 0.235 | 0.091 [0.045, 0.147] | 1.4 / 4.1 | 0.276 | +0.184 [+0.071, +0.319], p_Holm 0.000 | 0.250 / 0.107 |
| GBM (fair) | 15 | 0.166 [0.105, 0.225] | 2.9 / 4.2 | 0.240 | +0.073 [+0.015, +0.140], p_Holm 0.066 | 0.115 [0.066, 0.171] | 1.6 / 3.8 | 0.274 | +0.157 [+0.056, +0.277], p_Holm 0.000 | 0.227 / 0.233 |
| MISTy-style forest (fair) | 15 | 0.212 [0.126, 0.300] | 3.8 / 4.2 | 0.240 | +0.027 [-0.024, +0.078], p_Holm 1.000 | 0.126 [0.075, 0.181] | 1.8 / 3.8 | 0.274 | +0.147 [+0.053, +0.255], p_Holm 0.000 | 0.300 / 0.300 |
| EBM (fair) | 15 | 0.191 [0.110, 0.272] | 3.4 / 4.2 | 0.240 | +0.049 [-0.010, +0.110], p_Holm 0.499 | 0.115 [0.067, 0.169] | 1.6 / 3.8 | 0.274 | +0.158 [+0.057, +0.276], p_Holm 0.000 | 0.240 / 0.240 |
| iRF (fair) | no receipts (timeout / not run) | | | | | | | | | |
| RuleFit (fair) | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies meylan2022_ccrcc__to__gse250163; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0564, pairs 0.0715

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.172 / 0.085 | 0.201 / 0.122 | 0.733 / 0.667 | 0.555 / 0.583 | 0.576 / 0.584 | 0.222 / 0.251 | 0.039 / 0.064 | 0.253 / 0.457 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.172 / 0.091 | 0.201 / 0.143 | 0.733 / 0.800 | 0.555 / 0.618 | 0.576 / 0.599 | 0.222 / 0.283 | 0.039 / 0.046 | 0.253 / 0.460 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.187 / 0.085 | 0.236 / 0.122 | 0.933 / 0.667 | 0.741 / 0.583 | 0.589 / 0.584 | 0.244 / 0.251 | 0.037 / 0.064 | 0.300 / 0.457 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.187 / 0.091 | 0.236 / 0.143 | 0.933 / 0.800 | 0.741 / 0.618 | 0.589 / 0.599 | 0.244 / 0.283 | 0.037 / 0.046 | 0.300 / 0.460 |
| random (chance reference, 20 seeds) | 0.017 / 0.005 | 0.029 / 0.010 | 0.440 / 0.467 | 0.173 / 0.192 | 0.499 / 0.500 | 0.074 / 0.095 | -0.000 / -0.000 | 0.075 / 0.094 |
| linear interaction screen | 0.187 / 0.005 | 0.236 / 0.010 | 0.933 / 0.467 | 0.741 / 0.280 | 0.589 / 0.505 | 0.244 / 0.105 | 0.037 / 0.005 | 0.300 / 0.117 |
| NID | 0.163 / 0.053 | 0.185 / 0.089 | 0.800 / 0.800 | 0.614 / 0.581 | 0.545 / 0.552 | 0.163 / 0.195 | -0.026 / 0.003 | 0.220 / 0.413 |
| MISTy-style forest (pairs derived) | 0.150 / 0.041 | 0.172 / 0.060 | 0.733 / 0.627 | 0.703 / 0.505 | 0.574 / 0.538 | 0.217 / 0.168 | 0.090 / 0.077 | 0.195 / 0.284 |
| Model-X knockoffs | 0.112 / 0.009 | 0.172 / 0.024 | 0.853 / 0.720 | 0.629 / 0.379 | 0.547 / 0.506 | 0.165 / 0.108 | 0.006 / -0.000 | 0.190 / 0.133 |
| GBM + SHAP interactions (native) | 0.142 / 0.031 | 0.178 / 0.043 | 0.800 / 0.813 | 0.674 / 0.531 | 0.577 / 0.515 | 0.219 / 0.124 | 0.072 / 0.002 | 0.193 / 0.189 |
| RuleFit (native) | 0.058 / 0.011 | 0.077 / 0.017 | 0.800 / 0.627 | 0.552 / 0.381 | 0.534 / 0.502 | 0.140 / 0.099 | 0.012 / -0.000 | 0.141 / 0.115 |
| GBM + Friedman H | 0.018 / 0.010 | 0.029 / 0.021 | 0.493 / 0.667 | 0.220 / 0.283 | 0.501 / 0.509 | 0.077 / 0.112 | -0.004 / 0.006 | 0.075 / 0.137 |
| EBM (native) | 0.172 / 0.068 | 0.204 / 0.143 | 0.800 / 0.800 | 0.628 / 0.417 | 0.575 / 0.570 | 0.214 / 0.225 | 0.102 / 0.058 | 0.270 / 0.393 |
| iRF (native) | 0.108 / 0.011 | 0.156 / 0.016 | 0.929 / 0.614 | 0.669 / 0.335 | 0.546 / 0.502 | 0.153 / 0.091 | 0.019 / -0.000 | 0.171 / 0.102 |
| GBM (fair) | 0.093 / 0.019 | 0.146 / 0.028 | 0.733 / 0.800 | 0.544 / 0.472 | 0.543 / 0.518 | 0.158 / 0.129 | -0.001 / 0.005 | 0.170 / 0.193 |
| MISTy-style forest (fair) | 0.119 / 0.026 | 0.172 / 0.037 | 0.667 / 0.800 | 0.649 / 0.650 | 0.574 / 0.519 | 0.214 / 0.132 | 0.059 / -0.016 | 0.217 / 0.233 |
| EBM (fair) | 0.101 / 0.024 | 0.154 / 0.036 | 0.667 / 0.733 | 0.638 / 0.455 | 0.550 / 0.517 | 0.172 / 0.127 | 0.001 / -0.006 | 0.170 / 0.217 |

Label set per study: meylan2022_ccrcc__to__gse250163: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## b2_zenodo14620362

### Table 1: studies meylan2022_ccrcc__to__zenodo14620362; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0487, pairs 0.0576
Per-study prevalence: {"single": {"meylan2022_ccrcc__to__zenodo14620362": 0.0487}, "pair": {"meylan2022_ccrcc__to__zenodo14620362": 0.0576}}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), repeated in the v05 addendum chain 13-16.
Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.

| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 17 | 0.201 [0.100, 0.329] | 4.1 / 4.1 | 0.201 | - | 0.211 [0.105, 0.337] | 3.7 / 3.7 | 0.211 | - | 0.347 / 0.424 |
| ours: support-fit ordering, joint pair gain (suppl.) | 17 | 0.201 [0.100, 0.329] | 4.1 / 4.1 | 0.201 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.220 [0.111, 0.349] | 3.8 / 3.7 | 0.211 | -0.009 [-0.021, +0.003], p_Holm 1.000 | 0.347 / 0.453 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 17 | 0.222 [0.126, 0.342] | 4.6 / 4.1 | 0.201 | -0.021 [-0.044, +0.005], p_Holm 1.000 | 0.211 [0.105, 0.337] | 3.7 / 3.7 | 0.211 | +0.000 [+0.000, +0.000], p_Holm 1.000 | 0.371 / 0.424 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 17 | 0.222 [0.126, 0.342] | 4.6 / 4.1 | 0.201 | -0.021 [-0.044, +0.005], p_Holm 1.000 | 0.220 [0.111, 0.349] | 3.8 / 3.7 | 0.211 | -0.009 [-0.021, +0.003], p_Holm 1.000 | 0.371 / 0.453 |
| random (chance reference, 20 seeds) | 17 | 0.064 [0.039, 0.090] | 1.3 / 4.1 | 0.201 | +0.136 [+0.037, +0.269], p_Holm 0.000 | 0.071 [0.040, 0.107] | 1.2 / 3.7 | 0.211 | +0.139 [+0.036, +0.270], p_Holm 0.000 | 0.059 / 0.067 |
| linear interaction screen | 17 | 0.222 [0.126, 0.342] | 4.6 / 4.1 | 0.201 | -0.021 [-0.044, +0.005], p_Holm 1.000 | 0.081 [0.048, 0.118] | 1.4 / 3.7 | 0.211 | +0.130 [+0.030, +0.261], p_Holm 0.005 | 0.371 / 0.053 |
| NID | 17 | 0.188 [0.096, 0.308] | 3.9 / 4.1 | 0.201 | +0.013 [-0.018, +0.051], p_Holm 1.000 | 0.171 [0.085, 0.284] | 3.0 / 3.7 | 0.211 | +0.041 [-0.009, +0.117], p_Holm 0.540 | 0.324 / 0.341 |
| MISTy-style forest (pairs derived) | 17 | 0.210 [0.131, 0.288] | 4.3 / 4.1 | 0.201 | -0.010 [-0.067, +0.067], p_Holm 1.000 | 0.160 [0.094, 0.237] | 2.8 / 3.7 | 0.211 | +0.051 [-0.020, +0.144], p_Holm 0.565 | 0.324 / 0.335 |
| Model-X knockoffs | 17 | 0.144 [0.077, 0.236] | 3.0 / 4.1 | 0.201 | +0.056 [+0.013, +0.109], p_Holm 0.026 | 0.078 [0.045, 0.117] | 1.4 / 3.7 | 0.211 | +0.132 [+0.033, +0.265], p_Holm 0.005 | 0.206 / 0.149 |
| GBM + SHAP interactions (native) | 17 | 0.178 [0.097, 0.295] | 3.6 / 4.1 | 0.201 | +0.023 [-0.015, +0.072], p_Holm 1.000 | 0.088 [0.054, 0.125] | 1.5 / 3.7 | 0.211 | +0.122 [+0.032, +0.234], p_Holm 0.005 | 0.235 / 0.156 |
| RuleFit (native) | 17 | 0.121 [0.060, 0.206] | 2.5 / 4.1 | 0.201 | +0.079 [+0.033, +0.132], p_Holm 0.000 | 0.071 [0.041, 0.107] | 1.2 / 3.7 | 0.211 | +0.139 [+0.036, +0.271], p_Holm 0.000 | 0.129 / 0.067 |
| stability selection | no receipts (timeout / not run) | | | | | | | | | |
| GBM + Friedman H | 17 | 0.064 [0.041, 0.089] | 1.3 / 4.1 | 0.201 | +0.136 [+0.039, +0.269], p_Holm 0.000 | 0.080 [0.048, 0.116] | 1.4 / 3.7 | 0.211 | +0.130 [+0.034, +0.254], p_Holm 0.005 | 0.066 / 0.088 |
| EBM (native) | 17 | 0.227 [0.132, 0.344] | 4.7 / 4.1 | 0.201 | -0.026 [-0.052, +0.003], p_Holm 1.000 | 0.200 [0.123, 0.286] | 3.5 / 3.7 | 0.211 | +0.011 [-0.044, +0.078], p_Holm 1.000 | 0.335 / 0.371 |
| iRF (native) | 17 | 0.168 [0.101, 0.252] | 3.5 / 4.1 | 0.201 | +0.033 [-0.016, +0.093], p_Holm 1.000 | 0.075 [0.046, 0.110] | 1.3 / 3.7 | 0.211 | +0.135 [+0.037, +0.256], p_Holm 0.000 | 0.194 / 0.067 |
| GBM (fair) | 17 | 0.180 [0.088, 0.304] | 3.7 / 4.1 | 0.201 | +0.021 [-0.016, +0.058], p_Holm 1.000 | 0.087 [0.052, 0.128] | 1.5 / 3.7 | 0.211 | +0.123 [+0.028, +0.245], p_Holm 0.000 | 0.194 / 0.159 |
| MISTy-style forest (fair) | 17 | 0.163 [0.098, 0.237] | 3.4 / 4.1 | 0.201 | +0.037 [-0.022, +0.109], p_Holm 1.000 | 0.107 [0.064, 0.157] | 1.9 / 3.7 | 0.211 | +0.104 [+0.017, +0.217], p_Holm 0.018 | 0.288 / 0.206 |
| EBM (fair) | 17 | 0.186 [0.094, 0.308] | 3.8 / 4.1 | 0.201 | +0.016 [-0.015, +0.049], p_Holm 1.000 | 0.078 [0.043, 0.120] | 1.4 / 3.7 | 0.211 | +0.132 [+0.032, +0.265], p_Holm 0.000 | 0.224 / 0.106 |
| iRF (fair) | no receipts (timeout / not run) | | | | | | | | | |
| RuleFit (fair) | no receipts (timeout / not run) | | | | | | | | | |

### Table 2 (secondary, same score files): studies meylan2022_ccrcc__to__zenodo14620362; RFF labels at 95%, baseline B_known; mean prevalence singles 0.0487, pairs 0.0576

| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |
|---|---|---|---|---|---|---|---|---|
| ours: support-fit ordering (pairs = mean of singles) | 0.134 / 0.071 | 0.166 / 0.114 | 0.824 / 0.647 | 0.601 / 0.564 | 0.579 / 0.567 | 0.209 / 0.196 | -0.004 / 0.001 | 0.259 / 0.385 |
| ours: support-fit ordering, joint pair gain (suppl.) | 0.134 / 0.080 | 0.166 / 0.121 | 0.824 / 0.706 | 0.601 / 0.594 | 0.579 / 0.577 | 0.209 / 0.216 | -0.004 / -0.050 | 0.259 / 0.415 |
| hybrid: linear-screen singles + support-fit pairs (suppl.) | 0.142 / 0.071 | 0.178 / 0.114 | 0.765 / 0.647 | 0.652 / 0.564 | 0.585 / 0.567 | 0.220 / 0.196 | 0.013 / 0.001 | 0.274 / 0.385 |
| hybrid: linear-screen singles + joint support-fit pairs (suppl.) | 0.142 / 0.080 | 0.178 / 0.121 | 0.765 / 0.706 | 0.652 / 0.594 | 0.585 / 0.577 | 0.220 / 0.216 | 0.013 / -0.050 | 0.274 / 0.415 |
| random (chance reference, 20 seeds) | 0.011 / 0.005 | 0.020 / 0.009 | 0.397 / 0.382 | 0.143 / 0.163 | 0.500 / 0.500 | 0.056 / 0.068 | 0.001 / 0.002 | 0.058 / 0.068 |
| linear interaction screen | 0.142 / 0.008 | 0.178 / 0.031 | 0.765 / 0.353 | 0.652 / 0.185 | 0.585 / 0.507 | 0.220 / 0.080 | 0.013 / 0.058 | 0.274 / 0.088 |
| NID | 0.128 / 0.069 | 0.152 / 0.091 | 0.765 / 0.706 | 0.587 / 0.663 | 0.564 / 0.555 | 0.180 / 0.171 | 0.143 / 0.161 | 0.232 / 0.291 |
| MISTy-style forest (pairs derived) | 0.092 / 0.035 | 0.126 / 0.049 | 0.706 / 0.647 | 0.615 / 0.500 | 0.564 / 0.553 | 0.178 / 0.163 | 0.178 / 0.102 | 0.235 / 0.256 |
| Model-X knockoffs | 0.104 / 0.007 | 0.117 / 0.018 | 0.729 / 0.365 | 0.643 / 0.282 | 0.540 / 0.506 | 0.134 / 0.078 | -0.008 / -0.004 | 0.147 / 0.141 |
| GBM + SHAP interactions (native) | 0.112 / 0.025 | 0.129 / 0.031 | 0.824 / 0.624 | 0.620 / 0.468 | 0.569 / 0.511 | 0.190 / 0.088 | 0.141 / -0.002 | 0.174 / 0.121 |
| RuleFit (native) | 0.080 / 0.004 | 0.095 / 0.009 | 0.529 / 0.424 | 0.409 / 0.261 | 0.527 / 0.500 | 0.109 / 0.067 | 0.034 / -0.006 | 0.097 / 0.066 |
| GBM + Friedman H | 0.012 / 0.004 | 0.021 / 0.014 | 0.424 / 0.412 | 0.142 / 0.241 | 0.502 / 0.505 | 0.061 / 0.076 | -0.005 / 0.001 | 0.062 / 0.091 |
| EBM (native) | 0.141 / 0.051 | 0.173 / 0.077 | 0.824 / 0.706 | 0.629 / 0.441 | 0.585 / 0.579 | 0.220 / 0.212 | 0.237 / 0.121 | 0.259 / 0.315 |
| iRF (native) | 0.115 / 0.008 | 0.186 / 0.013 | 0.765 / 0.400 | 0.639 / 0.263 | 0.555 / 0.502 | 0.162 / 0.071 | 0.006 / -0.006 | 0.144 / 0.067 |
| GBM (fair) | 0.108 / 0.015 | 0.128 / 0.022 | 0.765 / 0.647 | 0.558 / 0.431 | 0.568 / 0.512 | 0.188 / 0.089 | -0.032 / -0.003 | 0.153 / 0.156 |
| MISTy-style forest (fair) | 0.078 / 0.021 | 0.098 / 0.030 | 0.706 / 0.529 | 0.575 / 0.314 | 0.565 / 0.530 | 0.178 / 0.121 | 0.030 / 0.001 | 0.203 / 0.194 |
| EBM (fair) | 0.103 / 0.006 | 0.128 / 0.013 | 0.706 / 0.412 | 0.591 / 0.269 | 0.564 / 0.504 | 0.181 / 0.075 | -0.030 / -0.009 | 0.147 / 0.126 |

Label set per study: meylan2022_ccrcc__to__zenodo14620362: label engine v05. Random is the chance reference (AP = prevalence; balanced accuracy 0.5).
Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference.

## calibration_table

### Calibration of the release checkpoint (predicted vs realised held-out gain over B_known)

| study | kind | n pool members | slope [95% CI] | intercept | Spearman | Kendall tau | ECE of pass probability | ECE of constant prevalence |
|---|---|---|---|---|---|---|---|---|
| gse237183_glioma | single | 567080 | 0.416 [0.214, 0.722] | -0.00094 | 0.241 | 0.163 | 0.0033 | 0.0079 |
| gse237183_glioma | pair | 800000 | 0.457 [0.144, 0.692] | +0.00003 | 0.221 | 0.158 | 0.0073 | 0.0100 |
| nakajima_luad_b3 | single | 636800 | 0.888 [0.754, 0.957] | +0.00132 | 0.199 | 0.144 | 0.0205 | 0.0400 |
| nakajima_luad_b3 | pair | 800000 | 1.360 [1.161, 1.439] | +0.00457 | 0.135 | 0.097 | 0.0336 | 0.0571 |
| gse307534_lung_precursor_b3 | single | 584360 | 0.092 [0.021, 0.206] | -0.00031 | 0.146 | 0.106 | 0.0075 | 0.0161 |
| gse307534_lung_precursor_b3 | pair | 800000 | -0.039 [-0.315, 0.123] | -0.00019 | 0.135 | 0.089 | 0.0083 | 0.0234 |

## stability_folds

### Fold stability of support-fit ordering (10 patient-grouped splits per target)

| study | method | kind | mean within-target SD of AP across splits | between-target SD of AP | top-10 Jaccard between splits | top-20 Jaccard | chance Jaccard top-10 |
|---|---|---|---|---|---|---|---|
| gse237183_glioma | sel_screen | single | 0.182 | 0.288 | 0.127 | 0.106 | 0.0035 |
| gse237183_glioma | sel_screen | pair | 0.153 | 0.319 | 0.196 | 0.140 | 0.0025 |
| gse237183_glioma | sel_screen_joint | single | 0.182 | 0.288 | 0.127 | 0.106 | 0.0035 |
| gse237183_glioma | sel_screen_joint | pair | 0.169 | 0.329 | 0.248 | 0.179 | 0.0025 |
| gse237183_glioma | linear_screen | single | 0.189 | 0.324 | 0.425 | 0.334 | 0.0035 |
| gse237183_glioma | linear_screen | pair | 0.015 | 0.023 | 0.117 | 0.125 | 0.0025 |
| gse237183_glioma | nid | single | 0.232 | 0.320 | 0.243 | 0.172 | 0.0035 |
| gse237183_glioma | nid | pair | 0.118 | 0.217 | 0.121 | 0.107 | 0.0025 |
| nakajima_luad_b3 | sel_screen | single | 0.178 | 0.281 | 0.426 | 0.417 | 0.0031 |
| nakajima_luad_b3 | sel_screen | pair | 0.185 | 0.248 | 0.456 | 0.414 | 0.0025 |
| nakajima_luad_b3 | sel_screen_joint | single | 0.178 | 0.281 | 0.426 | 0.417 | 0.0031 |
| nakajima_luad_b3 | sel_screen_joint | pair | 0.181 | 0.269 | 0.489 | 0.456 | 0.0025 |
| nakajima_luad_b3 | linear_screen | single | 0.156 | 0.265 | 0.663 | 0.634 | 0.0031 |
| nakajima_luad_b3 | linear_screen | pair | 0.037 | 0.076 | 0.191 | 0.205 | 0.0025 |
| nakajima_luad_b3 | nid | single | 0.146 | 0.296 | 0.545 | 0.509 | 0.0031 |
| nakajima_luad_b3 | nid | pair | 0.097 | 0.075 | 0.211 | 0.207 | 0.0025 |
| gse307534_lung_precursor_b3 | sel_screen | single | 0.184 | 0.294 | 0.183 | 0.158 | 0.0034 |
| gse307534_lung_precursor_b3 | sel_screen | pair | 0.132 | 0.342 | 0.288 | 0.201 | 0.0025 |
| gse307534_lung_precursor_b3 | sel_screen_joint | single | 0.184 | 0.294 | 0.183 | 0.158 | 0.0034 |
| gse307534_lung_precursor_b3 | sel_screen_joint | pair | 0.138 | 0.360 | 0.369 | 0.257 | 0.0025 |
| gse307534_lung_precursor_b3 | linear_screen | single | 0.185 | 0.286 | 0.542 | 0.458 | 0.0034 |
| gse307534_lung_precursor_b3 | linear_screen | pair | 0.031 | 0.057 | 0.146 | 0.162 | 0.0025 |
| gse307534_lung_precursor_b3 | nid | single | 0.180 | 0.290 | 0.392 | 0.323 | 0.0034 |
| gse307534_lung_precursor_b3 | nid | pair | 0.071 | 0.104 | 0.116 | 0.109 | 0.0025 |

## ablation_operators

### Operator and morphology leave-out ablation on the registered v08 catalogue (renal, 200 frozen candidates; counts of relations that remain)

HEADER NOTE: 0 of the 200 frozen candidates carry an H-Optimus concept or morphology-dictionary variable (every variable and target is a gene, a programme score, a residual-programme target, or a ring / neighbour / contrast / heterogeneity / gradient transform of those). The morphology row therefore removes nothing and must not be read as morphology contributing nothing: the v08 registered catalogue contains no morphology candidates, so the morphology contribution is not assessed here. compartment_context restricts rows to operator-defined interface spots whose compartment labels come from the deconvolved compartments (registered_deconvolution v3.1); it is counted as a spatial-context operator, not as morphology, and has its own row.

| removed | internally supported (B_nuis rule) | beyond known biology (B_known) | externally replicated (W1 summary at read time) |
|---|---|---|---|
| none | 161 | 28 | 10 |
| family compartment_context | 111 | 12 | 10 |
| family heterogeneity_gradient | 150 | 17 | 1 |
| family residual_programme_targets | 111 | 28 | 10 |
| family ring_decay | 111 | 27 | 9 |
| all morphology-involving candidates (0 exist; row removes nothing) | 161 | 28 | 10 |

Families: ['compartment_context', 'heterogeneity_gradient', 'residual_programme_targets', 'ring_decay']. Replicated list: 10 ids (10 found in the frozen table). Source: /home/ubuntu/morpheus_work_v08/runs/registered_v08/renal-spatial-v08-multiscale-20261026/funnel_aggregate_receipt_matrix_repaired.json; ~/morpheus_work_v06/runs/confirmation/v08_downstream_strict_spatial_v08/replication_summary.json (W1 owns both; W1 external validation was still running, so the replicated column is a lower bound).

## b4_b5_consolidated

### B5 historical rediscovery (hit rule v02: a top-20 hypothesis containing a variable among the 10 design columns with the largest within-patient |r| with the answer proxy, |r| >= 0.5, stated sign)

| method | rank of first hit H1 / H2 / H3 | Hit@10 (of 3 tasks) | Hit@20 (of 3) | MRR |
|---|---|---|---|---|
| EBM (GA2M) | none / none / none | 0 | 0 | 0.000 |
| GBM + Friedman H | none / none / none | 0 | 0 | 0.000 |
| learned beam (release checkpoint; hit-rule v02 list) | none / 4 / none | 1 | 1 | 0.083 |
| linear interaction screen | none / 5 / 11 | 1 | 2 | 0.097 |
| random (count-matched) | 4 / 6 / 6 | 3 | 3 | 0.194 |
| ours: support-fit ordering (adopted ordering, 64 tested sets) | none / none / none | 0 | 0 | 0.000 |

Random is the count-matched chance reference. Tasks are 3 (cSCC, CRC, prostate); with n = 3 no interval is informative. H1/H2 proxies are protocol-derived (not source-exact); the repaired query-validated scorer (benchmark_v03/public/historical_repaired.md) keeps the H2 hits of the learned beam (rank 3), the linear screen (5), PySR (4) and RuleFit (2) and removes the random hit.
Sources: runs/development/historical_rediscovery_v02_final/RESULTS.md; runs/development/benchmark_v03/historical_adopted.log; benchmark_v03/public/historical_repaired.md.

Chance-separation check FAILED under hit rule v02: count-matched random hits all 3 tasks (MRR 0.194) and every method is at or below it. The repaired query-validated scorer below is the informative variant.

## Historical rediscovery: v02 hit rule vs repaired query-validated scorer (v03_query_validated)

Provenance: locked v02 lists (LOCK_H1_H2.md, LOCK_H3.md) rescored query-side only by dbench/harness/historical_repair_v03.py (unmodified); no proposer is run. A task is primary-eligible when its proxies are transcribed from the source (source_exact) and it has at least 3 query patients.

### H1: primary_eligible = False (proxies not source-exact (fibroblast:protocol_derived_proxy, endothelial:protocol_derived_proxy); 2 query patients)

| method | rank of first hit (v02) | rank of first query-validated hit (repaired) |
|---|---|---|
| learned relation search | none | none |
| random selection (count-matched) | 10 | none |
| explainable boosting machine (EBM) | none | none |
| gradient boosting + Friedman's H | none | none |
| linear interaction screen | none | none |
| MISTy multi-view random forest (scikit-learn reimplementation) | none | none |
| neural interaction detection (NID) | 19 | none |
| PySR symbolic regression | none | none |
| RuleFit (scikit-learn reimplementation) | 16 | none |

### H2: primary_eligible = False (proxies not source-exact (spp1_macrophage:protocol_derived_proxy, myofibroblast:protocol_derived_proxy))

| method | rank of first hit (v02) | rank of first query-validated hit (repaired) |
|---|---|---|
| learned relation search | 3 | 3 |
| random selection (count-matched) | 5 | none |
| explainable boosting machine (EBM) | none | none |
| gradient boosting + Friedman's H | none | none |
| linear interaction screen | 5 | 5 |
| MISTy multi-view random forest (scikit-learn reimplementation) | 13 | none |
| neural interaction detection (NID) | none | none |
| PySR symbolic regression | 4 | 4 |
| RuleFit (scikit-learn reimplementation) | 2 | 2 |

### H3: pending



### B4 higher-order specificity, Meylan blind cohort (F.1)

# Meylan blind-cohort replicate: higher-order selection (fold-level B_decomp pass, learned beam vs support screen vs joint)

F.1: Meylan blind-cohort replicate of the higher-order selection table (fold-level B_decomp), from real and null arms.

precision@k = share of the arm's top-k proposed sets per (family, target, fold) -- in the arm's own proposal order
(receipt 'rank') -- whose fold-level G_HO against the best proper-subset decomposition has a 95 % interval lower bound
above 0 (the fold-level analogue of the pooled B_decomp pass: the null arms have no cross-fold scoring). Null FDR = the same
statistic on the patient-permuted-target and scrambled-candidate arms (mean of the two, per target). Net precision =
precision@k - null FDR. Per-target means over folds; 95 % bootstrap over (family, target) cells; paired differences by cell.
Rule written here before the table was computed; no threshold other than LB > 0 is used.

Stated deviation (accepted by the coordinator 2026-10-27): the development table (public/decomp_public.md) uses the POOLED definition, a set passes when its G_HO against the best decomposition has a 95 % patient-bootstrap lower bound above 0 pooled over the folds query patients, and its null FDR comes from null-target end-to-end runs. The Meylan null arms have no cross-fold scoring, so this table uses the FOLD-LEVEL definition on both real and null arms; the two are not interchangeable and are kept side by side.

| arm | k | cells | precision@k | null FDR | net precision |
|---|---|---|---|---|---|
| learned_search | 1 | 40 | 0.060 [0.030, 0.095] | 0.031 [0.012, 0.054] | 0.029 [-0.008, 0.065] |
| learned_search | 5 | 40 | 0.043 [0.025, 0.063] | 0.040 [0.025, 0.056] | 0.003 [-0.020, 0.027] |
| learned_search | 10 | 40 | 0.048 [0.034, 0.061] | 0.036 [0.025, 0.049] | 0.011 [-0.009, 0.031] |
| learned_search | 20 | 40 | 0.072 [0.060, 0.084] | 0.038 [0.026, 0.051] | 0.034 [0.017, 0.051] |
| support_screen_search | 1 | 40 | 0.025 [0.005, 0.045] | 0.036 [0.017, 0.056] | -0.011 [-0.034, 0.010] |
| support_screen_search | 5 | 40 | 0.022 [0.013, 0.032] | 0.048 [0.035, 0.062] | -0.026 [-0.040, -0.012] |
| support_screen_search | 10 | 40 | 0.025 [0.017, 0.033] | 0.059 [0.048, 0.071] | -0.035 [-0.049, -0.022] |
| support_screen_search | 20 | 40 | 0.052 [0.042, 0.065] | 0.064 [0.056, 0.074] | -0.012 [-0.027, 0.001] |
| joint_screen_search | 1 | 40 | 0.025 [0.005, 0.050] | 0.049 [0.026, 0.074] | -0.024 [-0.056, 0.009] |
| joint_screen_search | 5 | 40 | 0.023 [0.014, 0.033] | 0.053 [0.042, 0.063] | -0.030 [-0.043, -0.015] |
| joint_screen_search | 10 | 40 | 0.032 [0.022, 0.042] | 0.057 [0.048, 0.066] | -0.025 [-0.037, -0.013] |
| joint_screen_search | 20 | 40 | 0.053 [0.043, 0.064] | 0.068 [0.058, 0.077] | -0.015 [-0.027, -0.002] |

Paired difference in net precision (by cell):
| k | learned - support | learned - joint | joint - support |
|---|---|---|---|
| 1 | 0.040 [-0.000, 0.083] | 0.053 [0.007, 0.100] | -0.012 [-0.055, 0.027] |
| 5 | 0.029 [0.004, 0.057] | 0.033 [0.011, 0.055] | -0.004 [-0.022, 0.016] |
| 10 | 0.046 [0.024, 0.068] | 0.036 [0.018, 0.053] | 0.010 [-0.006, 0.027] |
| 20 | 0.046 [0.031, 0.062] | 0.048 [0.032, 0.064] | -0.003 [-0.014, 0.010] |

Fold-level B_nuis validation rate of all proposed sets (share validated), real vs nulls:
| arm | real | patient null | scramble null |
|---|---|---|---|
| learned_search | 0.485 (3104/6400) | 0.019 (102/5408) | 0.076 (403/5280) |
| support_screen_search | 0.493 (3153/6400) | 0.006 (24/4288) | 0.126 (562/4448) |
| joint_screen_search | 0.498 (3190/6400) | 0.006 (32/5216) | 0.096 (498/5184) |


### B4 paired decomposition differences (development, public baselines)

## End-to-end K = 64 search, B_decomp net yield: learned relation search vs support-fit screen (development)

Provenance: proposer_repair_v01 registries (same beam, K = 64, 40 kidney-injury targets x 3 folds; sha-checked); B_decomp scored on each method's B_nuis-searched top 20 exactly as core.run does, recomputed because the comparison run did not persist scores (dbench/harness/decomp_paired.py). The recomputed aggregates reproduce COMPARISON.json.

| k | learned: real / null / net | screen: real / null / net | paired net difference (learned - screen) |
|---|---|---|---|
| 1 | 0.20 / 0.01 / 0.19 | 0.19 / 0.14 / 0.05 | +0.137 [+0.013, +0.258] |
| 5 | 1.02 / 0.20 / 0.83 | 1.33 / 0.85 / 0.48 | +0.350 [-0.021, +0.729] |
| 10 | 2.01 / 0.35 / 1.65 | 2.84 / 1.87 / 0.97 | +0.683 [+0.133, +1.246] |
| 20 | 3.92 / 0.70 / 3.22 | 5.77 / 3.40 / 2.36 | +0.858 [-0.046, +1.771] |

Saturation: no target reaches real B_decomp yield 10 of 10 at k = 10 for either method (learned 0/40, screen 0/40). The screen's higher-order picks replicate on null targets (null yield 1.87 at k = 10 vs 0.35 for the learned search), so most of its larger real count is what a null target also yields. Net of the nulls the learned search leads at k = 1 and k = 10 with intervals excluding 0; at k = 5 and 20 its lead has intervals that include 0.

## b7_he_student

### B7 spatial-state prediction from H&E (frozen H-Optimus-0 features; ridge and MLP students): per-spot Spearman between predicted and measured relation state, relations of the adopted method (support-fit ordering + learned beam)

| student | cohort | relations | median Spearman | fraction >= 0.4 | max | status pass |
|---|---|---|---|---|---|---|
| student_mlp | Meylan-heldout-patients | 45 | 0.141 | 0.07 | 0.555 | 9 |
| student_mlp | Zenodo-USZ (KC1-3; patch 0.67 um/px) | 45 | 0.047 | 0.00 | 0.204 | 0 |
| student_ridge | Meylan-heldout-patients | 45 | 0.193 | 0.13 | 0.542 | 12 |
| student_ridge | Zenodo-USZ (KC1-3; patch 0.67 um/px) | 45 | 0.045 | 0.00 | 0.253 | 0 |

Source: runs/development/translation_v10/layer_assay.json (W2). Positive control: TGF-beta relation passes in the same layer.

### MC6 panel efficiency (adopted method only; competing methods were not run through this layer): reduced-panel fidelity on external spots

| panel size (genes) | test | relations | median Spearman | fraction >= 0.8 |
|---|---|---|---|---|
| panel100 | panel100_external_spot_spearman | 143 | 0.225 | 0.08 |
| panel100 | panel100_heldout_spot_spearman | 57 | 0.261 | 0.11 |
| panel10 | panel10_external_spot_spearman | 143 | 0.206 | 0.05 |
| panel10 | panel10_heldout_spot_spearman | 57 | 0.208 | 0.00 |
| panel25 | panel25_external_spot_spearman | 143 | 0.228 | 0.06 |
| panel25 | panel25_heldout_spot_spearman | 57 | 0.247 | 0.02 |
| panel50 | panel50_external_spot_spearman | 143 | 0.228 | 0.06 |
| panel50 | panel50_heldout_spot_spearman | 57 | 0.256 | 0.11 |
| panel | panel_min_size_heldout_rho_ge_0.8 | 6 | 35.000 | 1.00 |

## seconds_per_method

| method | gse237183_glioma: median s per split (splits ok / timeouts / errors) | nakajima_luad_b3: median s per split (splits ok / timeouts / errors) | gse307534_lung_precursor_b3: median s per split (splits ok / timeouts / errors) |
|---|---|---|---|
| beam64 | 5.5 (400/0/0) | - (0/0/0) | - (0/0/0) |
| ebm | 144.1 (120/0/0) | 214.9 (111/7/0) | 141.6 (120/0/0) |
| fair_ebm | 59.6 (30/0/0) | 68.2 (30/0/0) | 67.3 (30/0/0) |
| fair_gbm | 48.7 (30/0/0) | 33.1 (30/0/0) | 51.1 (30/0/0) |
| fair_irf | 314.0 (1/20/0) | - (0/14/0) | - (0/0/0) |
| fair_misty | 122.9 (30/0/0) | 102.0 (30/0/0) | 163.8 (30/0/0) |
| fair_rulefit | 282.4 (6/16/0) | 267.4 (17/1/0) | - (0/0/0) |
| gbm_h | 174.4 (120/0/0) | 213.4 (105/13/0) | 155.7 (119/1/0) |
| gbm_shap | 39.7 (120/0/0) | 94.0 (116/4/0) | 26.1 (120/0/0) |
| ho_pair | 18.6 (400/0/0) | - (0/0/0) | - (0/0/0) |
| ho_pair_guarded | 18.6 (400/0/0) | - (0/0/0) | - (0/0/0) |
| hybrid_lin_joint | - (0/0/0) | - (0/0/0) | - (0/0/0) |
| hybrid_lin_sel | - (0/0/0) | - (0/0/0) | - (0/0/0) |
| irf | 262.5 (120/0/0) | 177.8 (107/9/0) | 222.6 (120/0/0) |
| knockoffs | 6.1 (120/0/0) | 7.1 (120/0/0) | 9.2 (120/0/0) |
| linear_screen | 1.9 (400/0/0) | 2.9 (400/0/0) | 1.8 (400/0/0) |
| misty | 7.0 (400/0/0) | 6.4 (400/0/0) | 6.7 (400/0/0) |
| nid | 3.5 (120/0/0) | 5.9 (400/0/0) | 3.5 (400/0/0) |
| random | 0.2 (400/0/0) | 0.7 (400/0/0) | 0.2 (400/0/0) |
| rulefit | 156.4 (103/12/0) | 186.0 (107/10/0) | 168.9 (118/2/0) |
| sel_screen | 1.3 (400/0/0) | 4.2 (400/0/0) | 2.1 (400/0/0) |
| sel_screen_joint | 2.5 (400/0/0) | 8.1 (400/0/0) | 4.8 (400/0/0) |
| stability_selection | 382.8 (60/46/0) | 246.9 (58/43/0) | 493.0 (46/54/0) |

Before (seconds_per_method_before_relaunch.md; 2 fits in flight, 240 s budget): timeouts / attempted target x split fits, glioma / Nakajima / GSE307534: stability selection 43 of 62 / 45 of 100 / 30 of 48; GBM+H 25 of 65 / 33 of 108 / 5 of 60; EBM 5 of 74 / 24 of 110 / 0 of 60; iRF 25 of 64 / 22 of 112 / 0 of 60; RuleFit 16 of 73 / 14 of 115 / 5 of 60.
After (10 fits in flight, 2 threads per fit, 480 s budget, resume rule): see the table above, cells are median seconds per split (splits ok / timeouts / errors); GBM+H timeouts fall to 0 / 13 / 1, EBM to 0 / 7 / 0, iRF to 0 / 9 / 0, RuleFit to 12 / 10 / 2; stability selection stays incomplete (46 / 43 / 54 timeouts at 480 s) and fair_irf and fair_rulefit are not completed.
Friedman H per split (glioma target 0 split 3, same model): loop 102.3 s, batched 83.2 s, values bit-identical (log h_equal_check.out). Box load 45-55 before, 20-35 after.

## top-10 overlap with ours

### overlap@10 of top-10 singles with ours: support-fit ordering (pairs = mean of singles)

| method | gse237183_glioma | nakajima_luad_b3 | gse307534_lung_precursor_b3 |
|---|---|---|---|
| NID | 0.146 (n=120) | 0.222 (n=400) | 0.195 (n=400) |
| linear interaction screen | 0.151 (n=400) | 0.192 (n=400) | 0.182 (n=400) |
| ours: support-fit ordering, joint pair gain (suppl.) | 1.000 (n=400) | 1.000 (n=400) | 1.000 (n=400) |
| EBM (native) | 0.161 (n=69) | 0.270 (n=86) | 0.238 (n=60) |
| GBM + SHAP interactions (native) | 0.139 (n=66) | 0.211 (n=80) | 0.222 (n=60) |

## b6_zenodo_tls (B6, Zenodo TLS annotations, 3 patients)

### B6 annotation recovery, Zenodo ccRCC TLS Visium: 3760 labelled spots, classes {'TUM': 2007, 'NOR': 1377, 'TLS': 204, 'INFL': 172}; k-means k=4, d=43 features per set; patient bootstrap (3 patients, 500 resamples, fixed assignment)

| feature set | ARI | NMI | purity | homogeneity | completeness | V | silhouette |
|---|---|---|---|---|---|---|---|
| relations | 0.074 [0.071, 0.145] | 0.100 [0.100, 0.245] | 0.633 [0.569, 0.795] | 0.115 [0.115, 0.266] | 0.089 [0.089, 0.228] | 0.100 [0.100, 0.245] | 0.096 [0.084, 0.152] |
| expression | 0.018 [0.018, 0.408] | 0.041 [0.041, 0.499] | 0.559 [0.559, 0.906] | 0.045 [0.045, 0.630] | 0.037 [0.037, 0.472] | 0.041 [0.041, 0.499] | -0.000 [-0.034, 0.014] |
| hoptimus | 0.088 [0.003, 0.281] | 0.096 [0.039, 0.366] | 0.633 [0.484, 0.783] | 0.113 [0.053, 0.475] | 0.083 [0.031, 0.296] | 0.096 [0.039, 0.366] | 0.007 [0.004, 0.016] |
| composition | 0.027 [0.024, 0.085] | 0.042 [0.033, 0.121] | 0.589 [0.542, 0.753] | 0.050 [0.044, 0.135] | 0.036 [0.026, 0.109] | 0.042 [0.033, 0.121] | 0.216 [0.197, 0.251] |
| random | -0.000 [-0.001, 0.000] | 0.001 [0.001, 0.002] | 0.534 [0.439, 0.753] | 0.001 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.002] | 0.015 [0.015, 0.018] |

Random features are the chance reference. Relations = frozen kidney relations discovered by support-fit ordering + learned beam (W2 score file).

## tabdistill_not_run

# TabPFN + SPEX (TabDistill): not completed

- Serious attempt: v04/v05 definition (TabPFN v2 local checkpoint, 256-column shortlist, <= 10,000 spots, ProxySPEX order 2, sample budget 2,048, 128 evaluation spots) on 10 targets x 3 splits x 3 studies = 90 fits, wall-clock budget 300 s per fit: 90 of 90 timed out (receipts in scores/*/tabdistill carry status timeout; no score files).
- Repair attempt: 4,000 fit spots, 64 evaluation spots, sample budget 512 (glioma, first target, split 3): still running after 34 minutes; stopped by me. A fit at this cost cannot be run on the 3-study grid in the available time.
- Per METHOD_BENCHMARK_SPEC section 9.1 the method is reported as not run with this record; no weaker version is substituted under the same name.

## seconds_per_method_before_relaunch (for the speed-up comparison)

| method | gse237183_glioma: median s per split (splits ok / timeouts / errors) | nakajima_luad_b3: median s per split (splits ok / timeouts / errors) | gse307534_lung_precursor_b3: median s per split (splits ok / timeouts / errors) |
|---|---|---|---|
| beam64 | 5.5 (400/0/0) | - (0/0/0) | - (0/0/0) |
| ebm | 164.5 (69/5/0) | 201.2 (86/24/0) | 131.9 (60/0/0) |
| fair_ebm | 59.6 (30/0/0) | 66.2 (24/0/0) | - (0/0/0) |
| fair_gbm | 48.7 (30/0/0) | 32.6 (24/0/0) | - (0/0/0) |
| fair_irf | 314.0 (1/20/0) | - (0/14/0) | - (0/0/0) |
| fair_misty | 122.9 (30/0/0) | 90.7 (24/0/0) | - (0/0/0) |
| fair_rulefit | 282.4 (6/16/0) | 267.4 (17/1/0) | - (0/0/0) |
| gbm_h | 202.0 (40/25/0) | 153.8 (75/33/0) | 134.1 (55/5/0) |
| gbm_shap | 41.1 (63/13/0) | 27.6 (80/28/0) | 25.2 (59/1/0) |
| ho_pair | 18.6 (400/0/0) | - (0/0/0) | - (0/0/0) |
| ho_pair_guarded | 18.6 (400/0/0) | - (0/0/0) | - (0/0/0) |
| hybrid_lin_joint | - (0/0/0) | - (0/0/0) | - (0/0/0) |
| hybrid_lin_sel | - (0/0/0) | - (0/0/0) | - (0/0/0) |
| irf | 243.1 (39/25/0) | 168.9 (90/22/0) | 194.9 (60/0/0) |
| knockoffs | 7.6 (84/0/0) | 7.1 (120/0/0) | 7.7 (60/0/0) |
| linear_screen | 1.9 (400/0/0) | 2.9 (400/0/0) | 1.8 (400/0/0) |
| misty | 7.5 (280/0/0) | 6.4 (400/0/0) | 6.0 (200/0/0) |
| nid | 3.5 (120/0/0) | 5.9 (400/0/0) | 3.5 (400/0/0) |
| random | 0.2 (400/0/0) | 0.7 (400/0/0) | 0.2 (400/0/0) |
| rulefit | 128.4 (57/16/0) | 185.6 (101/14/0) | 88.8 (55/5/0) |
| sel_screen | 1.3 (400/0/0) | 4.2 (400/0/0) | 2.1 (400/0/0) |
| sel_screen_joint | 2.5 (400/0/0) | 8.1 (400/0/0) | 4.8 (400/0/0) |
| stability_selection | 246.6 (19/43/0) | 245.5 (55/45/0) | 284.0 (18/30/0) |
