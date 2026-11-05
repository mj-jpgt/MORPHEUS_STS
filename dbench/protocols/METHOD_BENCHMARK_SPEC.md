# MORPHEUS Method Benchmark. Baselines, Metrics, Statistics, and Sweeps

(Owner specification, received 2026-10-27. Verbatim. Governing for the method benchmark; coordinator annotations at the end.)

**Audience.** Build agents.
**Purpose.** Produce the paper-ready comparison of the adopted MORPHEUS method against named public methods for predicting which candidate relations replicate in held-out patients. This replaces the 10-23 table as the headline, because the ranking was revised after that study had been seen.
**Freeze.** Section 1 runs before anything else. No method code changes after the freeze hash is written.
**Rule.** Agents produce code, results, tables, figures, and logs. Agents do not draft prose for the Science Talent Search report.

---

## 1. Freeze and data roles

1. Hash the current adopted method (support-fit ordering, learned beam, relation family, baselines B, label rule) and write `method_freeze.json` with the git commit, config files, and model weights hash.
2. Declare **development studies** as every study that influenced any ranking or label decision, including the study behind the 10-23 table. Results on these are reported as development performance only.
3. Declare **test studies** as at least two studies that never influenced any ranking, label, threshold, or hyperparameter decision. Prefer the bladder and cSCC holdouts if their history is clean. If either was used to choose anything, replace it with an untouched HEST study with at least 8 patients. Record the choice and the reason in `test_studies.json` before any test run.
4. Declare **transfer levels** for each test study. L0 is new patients in the same study as support. L1 is a different study, same organ and platform. L2 is a different organ.
5. All baselines and all sweeps run on exactly the same test studies, targets, candidate pools, folds, and labels.

## 2. The evaluation task

### 2.1 Units

- **Study s.** One test study.
- **Target t.** A frozen state score (residual programme, morphology centre similarity, tissue state). Use 40 targets per study, sampled once at random from the frozen target list with a fixed seed and saved to `targets.json`. If a study has fewer eligible targets, use all.
- **Support patients S and query patients Q.** Patient-grouped split. Default is all patients but 3 in S, 3 in Q, repeated over 10 random splits per target. The few-shot sweep in Section 6 changes the size of S.
- **Candidate pool.** For each (s, t), singles are every candidate in the registered pool (all variables in all four spatial versions, excluding the target's own variable). Pairs are a fixed random sample of 2,000 pairs drawn uniformly from all pairs of singles, seeded and saved. In controlled worlds, planted pairs are added to the pool. Every method scores exactly this pool.

### 2.2 Labels

For every candidate in the pool, compute on query patients only.

1. **Continuous label.** Held-out gain G over baseline B_known, fitted on S and scored on Q with the fixed relation family.
2. **Binary label (replicates).** 1 if the patient-bootstrap 95% interval of G on Q has a lower bound above 0 and G exceeds the 95th percentile of the patient-permuted-target null for that (s, t). Otherwise 0.
3. **Second-family label.** Repeat 1 and 2 with a GAM (pyGAM, tensor term for pairs, spline terms for singles) as the relation family. Every primary table is reported under both label families. This removes the advantage of ranking and labelling with the same function class.
4. **Label strictness sensitivity.** Recompute binary labels with interval levels 90% and 99%. Secondary only.
5. Report the prevalence (fraction of positives) for singles and pairs per study under every label definition. AP is uninterpretable without it.

Labels are computed once, cached by hash of (s, t, S, Q, candidate, family, level), and shared by every method.

## 3. Methods

Every method receives only the support patients' spots for the target and the candidate pool, and outputs one real-valued score per single and per pair. Higher means more likely to replicate. No method sees Q.

Each baseline is run in two modes where applicable.
- **Native mode.** The method's own importance or interaction score.
- **Fair mode.** Grouped permutation importance computed on held-out support folds (Section 3.14). This gives every model family the same out-of-patient scoring that MORPHEUS uses, and isolates whether the advantage comes from the method or from the criterion.

### 3.1 Ours (adopted method, frozen)
Exactly as hashed in Section 1. Singles and pairs scored by support-fit ordering (patient-grouped cross-validated gain on S over B_known). Report also the learned-beam variant at equal budget as a supplementary row.

### 3.2 Linear interaction screen
1. Standardize singles on S. Build product columns for every pair in the pool.
2. Fit `sklearn.linear_model.RidgeCV` with alphas `np.logspace(-3, 3, 13)` and `GroupKFold(5)` on patient.
3. Score singles by absolute standardized coefficient. Score pairs by absolute coefficient of the product column.
4. Include B_known covariates as unpenalized columns by residualizing the target on them first.

### 3.3 Explainable Boosting Machine
1. `interpret.glassbox.ExplainableBoostingRegressor(interactions=pairs_in_pool_as_index_tuples, outer_bags=8, inner_bags=0, max_rounds=5000, random_state=seed)`. Residualize the target on B_known before fitting.
2. If passing 2,000 pairs is infeasible, pass `interactions=200` so EBM selects pairs by FAST, and score unselected pairs by their FAST ranking position via `interpret.utils.measure_interactions`.
3. Singles scored by `term_importances()` for main terms. Pairs scored by term importance for pair terms.

### 3.4 Gradient boosting with SHAP interactions and Friedman's H
1. `lightgbm.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8)` with early stopping on a `GroupKFold` validation fold.
2. Singles scored by mean absolute SHAP value from `shap.TreeExplainer`.
3. Pairs scored by mean absolute SHAP interaction value from `TreeExplainer.shap_interaction_values` on a random subsample of 2,000 support spots.
4. Secondary pair score is Friedman's H statistic computed from partial dependence (implement directly from `sklearn.inspection.partial_dependence` on a 20-point grid per variable, H squared as in Friedman and Popescu 2008). Report H as its own row.

### 3.5 Neural Interaction Detection (NID)
1. Use the reference implementation from Tsang, Cheng, and Liu (ICLR 2018), repository `mtsang/neural-interaction-detection`.
2. MLP with layers [140, 100, 60, 20], L1 penalty 5e-5 on first-layer weights, trained on S with early stopping on a grouped validation fold.
3. Pairs scored by the NID interaction strength. Singles scored by the NID main-effect strength (aggregated first-layer weight times downstream influence).

### 3.6 MISTy-style multiview forest
1. Use `liana` `MistyData` (Python) with an intraview of the spot's own values and paraviews matching our neighbour versions.
2. Singles scored by the importance of the candidate in the view that contains it.
3. MISTy does not score pairs natively. Score pairs by the product of the two single importances and label this explicitly in the table as derived.

### 3.7 RuleFit
1. `imodels.RuleFitRegressor(max_rules=500, tree_size=4, random_state=seed)`.
2. Singles scored by the sum of absolute coefficients of rules and linear terms containing the candidate.
3. Pairs scored by the sum of absolute coefficients of rules containing both candidates.

### 3.8 Iterative random forest (iRF)
This is the established method for stable high-order interactions (Basu, Kumbier, Brown, Yu, PNAS 2018) and is the most directly relevant tree-based competitor for replicable interactions.
1. `imodels.IRFRegressor` with 4 iterations, 500 trees, and random intersection trees on the final forest.
2. Pairs scored by interaction stability score across bootstrap replicates.
3. Singles scored by final-iteration feature weights.

### 3.9 PySR on a shortlist
1. Shortlist the top 64 singles by absolute Spearman correlation with the residualized target on S.
2. `pysr.PySRRegressor(niterations=40, binary_operators=["+","*","-","/"], unary_operators=["exp","log","square"], maxsize=20, populations=30)`.
3. Singles scored by the summed Pareto-front score of expressions containing them. Pairs scored the same way for expressions containing both.
4. Run on 10 targets per study only, because of cost. Report on that subset and mark it.

### 3.10 TabDistill
1. Fit TabPFN (v2) on S, subsampling to at most 10,000 spots and using the 256-single shortlist from 3.9's procedure at size 256.
2. Run SPEX (repository `basics-lab/spectral-explain`) on the fitted TabPFN to obtain first- and second-order interaction coefficients.
3. Singles scored by absolute first-order coefficient. Pairs scored by absolute second-order coefficient. Pairs outside the shortlist score 0 and the table states the shortlist.

### 3.11 Stability selection
1. Columns are singles plus product columns for pool pairs, target residualized on B_known.
2. Complementary-pairs stability selection (Shah and Samworth 2013). Draw 50 complementary pairs of half-samples at the patient level (100 subsamples).
3. On each subsample fit `sklearn.linear_model.Lasso` along a grid of 30 alphas and record which columns enter.
4. Score each column by its maximum selection probability over the alpha grid.

### 3.12 Model-X knockoffs
1. `knockpy.knockoff_filter.KnockoffFilter(fstat="lasso", ksampler="gaussian")`, with Sigma estimated by Ledoit-Wolf shrinkage on S.
2. Columns are singles plus product columns for pool pairs. To reduce within-patient dependence, aggregate to one row per patient per spatial bin of 4 adjacent spots, and state this in the table note.
3. Score each column by its knockoff W statistic.

### 3.13 Random
Uniform random scores, 20 seeds, reported as mean and interval.

### 3.14 Fair mode, applied to 3.3, 3.4, 3.6, 3.7, and 3.8
1. Split S into 5 patient-grouped folds.
2. For each fold, fit the method's model on the other folds.
3. On the held-out fold, compute the loss increase when a single is permuted within patient. That is the single's fold score.
4. For a pair, compute the loss increase when both are jointly permuted within patient, minus the sum of the two single increases. That is the pair's interaction fold score.
5. Average fold scores. This is the fair-mode score.

## 4. Metrics

All metrics are computed per (study, target) and then averaged across targets within study (macro average), unless stated.

### 4.1 Primary endpoints (preregistered)
- **AP (average precision, area under the precision-recall curve)** for singles and for pairs, under the RFF label at 95%. AP is primary because positives are rare. Always report alongside prevalence and **AP lift** (AP divided by prevalence).

### 4.2 Secondary endpoints
- **AUROC** for singles and pairs.
- **Precision@k and Recall@k** for k in {5, 10, 20, 50}.
- **nDCG@k** for k in {10, 20}, with graded relevance equal to the continuous held-out gain clipped below at 0 (Järvelin and Kekäläinen 2002). This rewards ranking the largest replicating effects highest.
- **Net precision@k.** Precision@k on real targets minus precision@k on patient-permuted null targets.
- **Spearman correlation** between method score and continuous held-out gain.
- **Calibration** for methods whose scores are probabilities (ours if it outputs one, stability selection). Brier score, expected calibration error with 10 equal-mass bins, and a reliability diagram.
- **Compute.** Wall-clock seconds per target on fixed hardware.

### 4.3 End-to-end endpoints
- For each method, take its top 20 relations per discovery study, apply them without refit to the external cohorts, and report the number confirmed under the gain-above-null rule.
- Report the same count restricted to relations graded 2 to 5 under the relation-level novelty rubric.

### 4.4 Controlled-world endpoints
- Complete-object recovery@k for k in {1, 5, 10}, constituent recall, and mean reciprocal rank of the planted set.

## 5. Statistics

### 5.1 Structure
Candidates are nested in targets, targets in studies, and labels depend on patients. Naive tests that treat candidates as independent (including DeLong's AUROC test) understate uncertainty, so all inference uses resampling that respects this nesting.

### 5.2 Intervals and paired comparisons
1. **Hierarchical bootstrap, 2,000 replicates.** When there are 3 or more test studies, resample studies with replacement, then targets within each sampled study with replacement, keeping each target's candidate pool and labels intact. With fewer than 3 studies, resample targets within each study and report studies separately.
2. Recompute every metric on each replicate for every method.
3. **Paired difference.** On each replicate compute ours minus baseline. Report the mean difference, its 95% percentile interval, and a one-sided bootstrap p-value (fraction of replicates with difference at or below 0).
4. **Multiplicity.** Holm correction across all baselines within each primary endpoint (AP singles, AP pairs). Secondary endpoints are reported with intervals and no significance claims.
5. **Label uncertainty.** Repeat the primary comparison under the 90% and 99% label definitions and under the GAM label family. A result is called robust when the paired interval excludes zero under all four.

### 5.3 Power
1. From development studies, estimate the bootstrap standard deviation of the paired AP difference between ours and EBM.
2. Choose the number of targets per test study so that this standard deviation is at most 0.03, which gives roughly 80% power to detect an AP difference of 0.085 at one-sided alpha 0.05. If 40 targets is not enough, increase to the number required or to all eligible targets.
3. Report the achieved minimum detectable difference beside every primary comparison.

## 6. Sweeps

Run the main grid for every method. Run secondary sweeps for ours, EBM, GBM, iRF, stability selection, and random only.

### 6.1 Main grid
- **Support patients n (few-shot axis).** n in {2, 3, 5, 8, 12}, capped at what each study allows with 3 query patients held fixed. 20 random support draws per n. Report AP against n as a learning curve, the area under that curve, and the smallest n at which each method reaches 80% of its full-data AP.
- **Top-k.** k in {5, 10, 20, 50} for precision, recall, and nDCG.
- **Label baseline.** B_nuis, B_known, B_decomp for pairs. B_known is primary.
- **Transfer level.** L0, L1, L2 as available.

### 6.2 Secondary sweeps
- **Search budget K** (number of sets fitted, for ours and for any proposer that fits sets) in {16, 32, 64, 128, 256}. Report confirmed relations against K.
- **Spots per patient** in {64, 256, 1024}, holding n fixed at the study maximum.
- **Effect size** in controlled worlds at 0.5%, 1%, 2%, and 5% of residual variance, split into training-family and held-out-family relations.
- **Pair pool size** in {500, 2,000, 8,000} for ours, EBM, and iRF, to show ranking quality as the search space grows.
- **Label strictness** at 90%, 95%, 99%.

## 7. Tables and figures

1. **Table 1 (headline).** Test studies, primary endpoints. Rows are methods in native and fair mode. Columns are AP singles, AP pairs, AP lift, AUROC singles, AUROC pairs, each with intervals, paired difference against ours, Holm-adjusted p. Prevalence in the header.
2. **Table 2.** Top-k metrics (precision@10, recall@20, nDCG@10, net precision@10).
3. **Table 3.** End-to-end external confirmation and novel confirmations per method.
4. **Table 4.** Transfer levels L0, L1, L2 for primary endpoints.
5. **Figure A.** Few-shot learning curves (AP against support patients) with bootstrap bands for every method.
6. **Figure B.** Budget curve (confirmed relations against K).
7. **Figure C.** Controlled-world recovery against effect size, training and held-out families.
8. **Figure D.** Reliability diagram for methods with probability outputs.
9. **Supplementary table.** Robustness of Table 1 under the GAM label family and the 90% and 99% label levels.
10. **Supplementary table.** Ours with learned rerank versus support-fit ordering, and learned beam versus random and marginal beams at equal budget. Labelled as component ablations.

## 8. Run order

1. Freeze and declare studies (Section 1).
2. Build and cache labels for all test studies under every label definition.
3. Run ours, random, linear screen, EBM, GBM, RuleFit, stability selection. These are fast and fill Table 1.
4. Run fair mode for EBM, GBM, RuleFit, MISTy, iRF.
5. Run NID, MISTy, iRF, knockoffs, TabDistill.
6. Run PySR on its 10-target subset.
7. Run main-grid sweeps, then secondary sweeps.
8. Compute statistics and produce tables and figures.
9. Write `benchmark_results/` with one file per table and figure, each stamped with the freeze hash and test-study manifest version.

## 9. Stop rules and honesty rules

1. If a baseline fails after one serious attempt and one repair, report it as not run with the error, and do not substitute a weaker version under the same name.
2. Do not change any method, label, threshold, or pool after the first test-study result is seen.
3. Report every row, including rows where ours loses. If fair-mode baselines close the gap, Table 1 says so.
4. Development-study results never appear in Table 1.

---

## Consolidated protocol summary (owner, same message)

- Write one manifest listing every cohort and its role (meta-training, development, test, historical rediscovery, blind discovery, external replication).
- Mark as development every study that influenced any ranking, label, or threshold, including the study behind the 10-23 table.
- Declare at least two untouched test studies for the benchmark, preferring the bladder and cSCC holdouts if their history is clean.
- Hash the adopted method into method_freeze.json before any test-study run. No method change after that.
- Record that Meylan shaped the method, so every Meylan finding needs external replication. Oral SCC and the external ccRCC and CRC cohorts are the clean evidence.

**1.2 Layer 1, decoupled state discovery (unchanged from the classical paper).** Molecular branch: ridge adjustment of log expression on the 47 declared covariates with donor-grouped inner CV, truncated SVD of the residual, rank by 50-permutation parallel analysis, cross-fold alignment by cosine assignment, RNA only. Tissue branch, image only, two fixed unit sets: the 106 CellViT descriptors across five scales clustered per scale with patient-balanced K-means, and the 64 frozen H-Optimus-0 centres (per-study standardized, each populated in at least three studies), both frozen before any RNA is seen. Re-run the coupling ablation once under the current null stack so its numbers match the rest of the paper.

**1.3 Layer 2, relation search on frozen states.** Targets are frozen residual programmes, tissue states, and centre similarities. Candidates are genes, programmes, morphology units, each in four spatial versions (own, six-neighbour mean, eighteen-neighbour mean, contrast). One relation family for every method, RFF kernel ridge with linear terms, closed form, patient-grouped penalty. Gain over three declared baselines (nuisance, known biology, best proper-subset decomposition). Ranking by support-fit ordering. Learned beam chooses which sets are fitted at a fixed budget.

**1.4 Layer 3, support criteria (one definition).** A relation is internally supported when all hold: held-out gain interval (patient bootstrap) excludes zero; gain exceeds the 95th percentile of patient-permuted targets, candidate-scrambled targets, and Moran-matched random programmes; BH q at most 0.05 within its prespecified family; at least 60% of held-out patients agree in direction; passes the artifact screen (UMI, mitochondrial fraction, edge distance, fold and necrosis masks, haemoglobin and red-cell content, slide). A relation is externally confirmed when applied without refit to an independent cohort, its gain exceeds all null draws there, and its calibration slope is reported. A relation is conserved when confirmed in three or more organs under both transfer rules with decoded meaning preserved. Organization-beyond-abundance ΔR² (scrambled-configuration null) is a tag on any relation involving morphology.

**2. The novelty protocol (amendment v08b, frozen before grading).** Decoding is separate from novelty (calibrated grader: gene-disjoint scoring, Jaccard ≥ 0.2 against MSigDB C2, C5, C8 and Gavish, Moran-matched null; report decoded axes, never use them as the verdict). Grade the full relation. Primary grade in order: 1 no novelty identified; 2 context extension; 3 spatial extension; 4 compositional novelty (constituents reported individually, never jointly, and the relation passes the higher-order or conditional test); 5 new association. Secondary tags Conserved (three or more organs), Mechanistic (perturbation connection surviving the coherent-random null and signature perturbation). Prior-art rule: only a record reporting the same dependency counts. Procedure: templated queries through OpenAlex and Europe PMC; two independent graders; disagreements resolved by citing the record. Grades 2 to 5 count as novel; report counts per category. Templates, rule and graders hashed before grading; no regrading after translational results.

**3. Method benchmark** (this spec). Minimum if time runs short: untouched test studies, fair-mode baselines, and the end-to-end confirmation table.

**4. Discovery work this week, in priority order.** 1 Relation-level re-grade (done 10-27). 2 TCGA centre k45 decoding (done: known grade axis → stop). 3 Benchmark core (freeze, labels, fast baselines, fair mode, slower baselines, sweeps). 4 Gate 1 and Test A (report the detectability table either way; campaign only if Gate 1 passes). 5 Time-split harness (pre-cutoff catalogue, three preregistered post-cutoff ccRCC findings, hits and misses per method). 6 Meylan equal-budget arms (blind replication of the higher-order specificity result). 7 Decision: highest tier and grade becomes the showcase; if nothing new reaches tier 2, G18 is the showcase as a confirmed spatial extension and the catalogue counts plus the benchmark lead.

**5. Translation for confirmed relations.** Characterization panel for the top three to five (constituent table, pathway mapping, localization maps, composition-matched state-high/state-low regions at Mahalanobis below 1.0, Squidpy only if organization is implicated, LIANA+ only if a constituent is a ligand or receptor). Orthogonal support: CPTAC RNA–protein correlation for every ccRCC relation with quantified proteins. Distillation: H&E students (ridge, MLP, attention-pooled) against STPath and DeepSpot zero-shot, with TCGA pooled and within-site-centred Spearman; a minimal gene panel (at most 10 genes) where histology is the wrong readout. Clinical: one prespecified endpoint per relation per cohort, incremental value beyond stage, grade, age, sex, molecular subtype and a frozen pathology baseline; compare against published JAVELIN and IMmotion151 signatures and report recovery of a known axis honestly. Perturbation connectivity: report the existing null for the five Meylan objects as a result; new relations only under the coherent-random null with signature perturbation.

**6. What to extract from Paz's paper.** A gap stated in terms of unused data; one countable catalogue (tiered relation count); one clean metric on known ground truth (benchmark AP/AUROC vs best named baseline); a calibration figure (Test A detectability by effect size); independently corroborated showcase panels (three to five relations: matched-region H&E, spatial map, external replication, CPTAC protein, cross-organ result; S08's figure is the template); a demonstration at scale (each validated detector across every slide where it is validated — TCGA-KIRC and CPTAC-CCRCC for kidney relations — with slides processed, compute per slide, cost of spatial profiling from a cited source; never deploy where external validation is missing); limits stated as findings (grader 100 %, haemoglobin nuisance axis, glial abundance missing from CellViT, 24-patient detectability ceiling, development-versus-test gap); a public release (relation atlas, frozen manifests, calibrated grader, code on GitHub at submission).

**7. What to extract from Xu's paper.** A two-by-two gap table (columns: discovers unspecified states, predicts replication in unseen patients, calibrated novelty, routine-H&E readout; rows: SpaGCN, BANKSY, CellCharter, STPath and DeepSpot, Path2Space, PathPrism, EBM and iRF, MORPHEUS); one central idea with a reason for each design choice (decoupling, residualization on 47 covariates, patients as the unit, support-fit ordering as an out-of-patient criterion, refit-free replication, relation-level novelty, one relation family, three baselines); honest parity reporting; one concrete use-case figure (discovery → TCGA slide prediction → clinical table); tight scope; a one-sentence lay summary (owner writes); code availability statement.

---

## Coordinator annotations (2026-10-27)

**Already in place:** relation family v03_rff with B_nuis/B_known/B_decomp; shared-pool scoring of every method by its own importance (`dbench/benchmark`); named baselines EBM, GBM+H, NID, MISTy-style forest, RuleFit, PySR, linear screen, random (`benchmark_v03/public/`); few-shot axis (`fewshot*.md`); historical and end-to-end machinery; relation-level rubric and graders (done); artifact screen; patient-permuted / candidate-scrambled / Moran-matched nulls; v03.1 gain-above-null replication. The 10-23 dev table (`dev_retrieval.md`, REPRODUCED) becomes development performance only.

**New under this spec:** `method_freeze.json` and the cohort-role manifest; test-study audit (bladder_epithelial, bladder_batf3, Ji cSCC: usable only if the state doc shows no ranking/label/threshold decision was made on them — bladder_batf3 additionally has a scorer failure under diagnosis; GSE237183 glioma, 11 patients, is the leading untouched replacement; GSE208253 OSCC was used only for a reported proposer comparison, no decision); GAM label family (pyGAM) and 90/99 % strictness labels with cached labels; fair mode (grouped permutation importance) for EBM, GBM, MISTy, RuleFit, iRF; iRF (`imodels.IRFRegressor`), stability selection (Shah–Samworth), Model-X knockoffs (`knockpy`), SHAP interactions for GBM; AP primary with prevalence and lift; nDCG, Spearman, calibration; hierarchical bootstrap (2,000) with Holm and the power rule; budget / spots-per-patient / pool-size sweeps; end-to-end confirmation table per method; `benchmark_results/` stamped with the freeze hash.

**Cannot run as written:** TabDistill (TabPFN v2 weights require an account/licence we do not accept) → "not run" unless the owner places the weights on the box. NID reference repo and knockpy/pyGAM/imodels/shap/lightgbm are public pip/GitHub installs (allowed).

**Layer 1 coupling-ablation rerun and the Layer 3 "organization-beyond-abundance ΔR²" tag are discovery-side items** handed to Builder B1 after the benchmark core; §5 translation items extend `REMAINING_WORK.md` §E (CPTAC RNA–protein correlation; STPath/DeepSpot zero-shot comparators; attention-pooled student; JAVELIN/IMmotion151 signature comparison; scale demonstration across TCGA-KIRC/CPTAC slides).

---

## Addendum (owner, 2026-10-27): component experiments E1–E5

1. **E1. Beam versus exhaustive scoring on a fixed pool.** Claim it decides: which component produces the retrieval headline. Design: add one row to the benchmark, the adopted method with the beam at its default budget against exhaustive support-fit scoring of the same pool, same test studies and labels. Metric: AP for singles and pairs, with the paired bootstrap difference. Placement: one row in Table 1. If they tie, the paper attributes retrieval to the criterion and the beam's role to E3 and E5.
2. **E2. The decomposition baseline for complex relationships.** Claim it decides: that scoring pairs against the best lower-order decomposition is what makes complex retrieval specific. Design: rank the pair pool two ways, by joint gain over known biology alone and by gain over the best proper-subset decomposition. Evaluate on real test studies and on controlled worlds with residualized pure interactions that carry no marginal signal. Metric: pair AP, null pass rate among the top 20 pairs, and recovery of planted pure interactions. Placement: a small panel in the benchmark figure. This is the strongest evidence that the system ranks complex relationships well for the right reason, and it costs little because both scores already exist.
3. **E3. Search-space scaling.** Claim it decides: that the meta-trained beam helps where exhaustive search becomes impossible. Design: pair pools of 500, 2,000, and 8,000, plus triples, at a fixed fitting budget of 64. Compare the learned beam, a random beam, and a marginal-screen beam, plus exhaustive scoring wherever it is still feasible. Metric: confirmed relations found, null pass rate of the beam's picks, and compute time. Placement: the main figure for the deep learning contribution. A widening gap as the space grows is the result that makes the beam matter.
4. **E4. Patient-disjoint versus patient-shared episodes.** Claim it decides: that training on realized replication, the core idea of the meta-training, is what gives the beam its advantage. Design: retrain the beam with support and query drawn from the same patients, everything else identical, and rerun E3 at the 2,000-pair size. Metric: same as E3. Placement: one sentence in the main text, table in the supplement. This is the single meta-training ablation worth running, because it tests the idea rather than an implementation detail.
5. **E5. Transfer of the beam to a withheld organ.** Claim it decides: that the beam learned how to search rather than what one tissue looks like. Design: run E3 at the 2,000-pair size on the withheld organ, and plot predicted against realized shrinkage. Metric: E3's metrics, plus calibration slope and Spearman between predicted and realized shrinkage. Placement: a panel beside E3.

**Coordinator annotations on E1–E5.** E1: the dev-pool version exists (`learned_vs_screen.md`, development only); the Table-1 row is on the test studies. E2: both scores exist in every receipt (gain over B_known; G_HO against the best proper-subset decomposition); the controlled pure-interaction worlds exist in `dbench/benchmark/generators.py` (residualised interactions, marginals ≈ 0). E3: extends spec §6.2 (pool size, budget) with triples and the random and marginal-screen beams; the learned beam is the release checkpoint 13c36faa…. E4: no retrain needed — the patient-shared (row-split) checkpoint already exists, `release_rowsplit/frozen.pt` sha 139b317c… (step 2000; same architecture, episodes drawn from shared patients), so E4 = E3 at 2,000 pairs with that checkpoint; state in the table that it is the pre-existing row-split model. E5: the withheld organs are the test studies (glioma, lung) at transfer level L2; the predictor's Δ output is the predicted shrinkage, realised shrinkage = support gain − query gain. B3 owns E1–E5 after Table 1's fast rows; E3/E5 are the deep-learning figure.
