# Kidney protocol v0.5 — change notes (Agent L, 2026-10-12; development input, not a registration)

v0.4 (`renal_spatial_v04.json`, run `renal-spatial-v04-20261011`) stays as registered. These are the
changes the v0.4 characterisation implies for v0.5. The coordinator writes the v0.5 JSON.

## 1. Single-variable relation: measure information, not only non-linearity
- v0.3/v0.4: for a one-variable hypothesis the comparison model already holds the variable **linearly**
  (`_simple_design`, k = 1) and the full model swaps it for a 24-feature random-Fourier map, so the
  registered gain is R²(baseline + RBF(x)) − R²(baseline + linear x). A strong purely linear effect scores
  ≈ 0 and is neither ranked up by the CV proposal screen nor frozen (Agent C, exact refit to 1e-8).
- v0.5: `discovery.relation = "v05"` (implemented behind a flag in `general_discovery.py`; default
  `"v04"` reproduces v0.3/v0.4 exactly). One variable: competitor = baseline (+ context) only, full =
  baseline + RBF(x), so `replication_gain` is total held-out information gain. Every receipt row records
  `gain_linear` (baseline + linear x vs baseline) and `gain_nonlinear_beyond_linear` (baseline + RBF(x) vs
  baseline + linear x), which sum exactly to the total, each with a held-out-patient bootstrap CI, plus
  `shape` ∈ {linear_carries_it, beyond_linear_only, linear_and_beyond_linear, not_resolved} → Level A vs
  beyond-A per hypothesis. k ≥ 2: comparison unchanged (per-variable flexible maps), plus
  `gain_linear_additive` (baseline + linear x₁..x_k vs baseline) with CI.
- The CV proposal ranking (`_proposal_cv_gain`) uses the same relation, so linear single variables can
  enter the shared budget. Same inner-CV alpha selection (`VALIDATION_ALPHA_GRID`); competitor and full
  model still choose alpha independently.
- Wording for reading v0.4: any v0.4 single-variable pass is a claim about shape beyond the linear term.

## 2. Pass rule includes the within-patient permutation null
- Agent C found registered passes (full R² > 0, gain CI > 0) with both structured nulls at p = 1.0 where
  `full_alpha` ≈ 10 × `competitor_alpha`: the "gain" survives permuting the held-out response within
  patient, so it is calibration between two differently shrunk models, not information in the variable.
- v0.5 per-fold pass: held-out full-model R² > 0 AND gain CI low > 0 AND within-patient-permutation
  p ≤ 0.05 (and spatial-block p ≤ 0.05 where the claim is spatial). `proposed` recurrence counts only
  such folds. Optional second safeguard: report the gain also with the full model refit at the
  competitor's alpha (like-for-like shrinkage) and flag rows where the two disagree in sign.

## 3. Positive controls must test something the competitor does not already contain
- v0.4's IG control fails by design: the competitor holds the Gavish IG programme score (and the
  Inflammatory fraction), and under v0.4's relation any IG gene's linear effect sits inside its own
  comparison model.
- v0.5: (a) controls are scored with the v0.5 relation (linear part counts); (b) the control's own
  programme is dropped from the competitor and from the programmes view for that control only
  (Agent C's development control, fold 0/1, `cv_frac_Inflammatory`), or the control is a variable with no
  competitor analogue; (c) keep a planted control (a known linear and a known sinusoidal term added to a
  held-out copy of a real target) whose expected gain is stated in advance; (d) report each control's
  linear add gain beside the relation gain.

## 4. Generator yield in the context family
- `molecular_x_context_to_tissue_phenotype`: the generator emitted nothing in 45 of 48 fold-0..3
  receipts under the molecular-and-context requirement (1 of 384 top-8 CV slots). v0.5 should either
  (a) let the generator search molecular × context products directly (seed it with the screened
  molecular singles × context columns), or (b) give the context family its own reserved generator budget
  with the requirement applied after distillation rather than as a hard admissibility gate, and record
  emitted/admissible/frozen counts per receipt.
- Agent C's audit also shows generator picks that pass are mostly `recurrent_in_generator = false`;
  keep sending non-recurrent picks to validation (do not reintroduce the stability filter as a gate).

## 5. Development Level-A screen (input to v0.5 controls and targets)
- `runs/development/level_a_screen/` (dev clone `~/morpheus_work_dev`): linear add-one gain of every
  candidate-view column, folds 0–4, the 12 morphology + 8 own_mp targets (20; the brief's "24" counts
  differently), exact joint refit for every column the screen flags, patient-bootstrap CI,
  within-patient-permutation p. Files: `pooled.json`, `summary.json`, `decode_top_w*.json`.
- Strong linear relations exist that the v0.4 relation scored ≈ 0: e.g. neighbour IFN-α / IFN-γ hallmark
  → own IFN/MHC-II (I) programme, linear gain 0.07–0.08 (registered 0.007); cv_count ← `pca:PC4` passes 4/5
  folds linearly (0.036; registered 0.004).
- **v0.5 must add a leak guard for own_mp targets**: neighbour means of programmes/PCs that share genes with
  the target programme are close relatives of the excluded "target neighbour mean" (spatial smoothing of
  the response). Exclude, or put in the competitor, any neighbour-mean column whose proposal-row |r| with
  the response's own neighbour mean (k = 6) is ≥ 0.9 (`decode_top.json` records this r for the top hits:
  neighbour IFN-α / IFN-γ hallmark vs own IFN/MHC-II (I) 0.93, neighbour EMT hallmark vs own EMT-I 0.92,
  neighbour allograft rejection 0.86, neighbour fatty-acid / bile-acid hallmark vs own Glutathione 0.77–0.78).
  The registered run's own_mp proposed entries (`nbr:hallmark:*` → own:IFN, own:IG …) carry the same issue.
- PCA / NMF (and their neighbour means) are refit per fold, so `pca:PC17` in fold 2 is not `pca:PC17` in
  fold 4: v0.5 recurrence across folds should match such coordinates by loading similarity (or fit the
  decomposition once on all proposal-eligible patients per fold set) rather than by index. SCimilarity,
  scGPT and H-Optimus-0 coordinates are fixed embeddings and are safe to match by index.
- Two morphology targets (`cv_frac_Dead`, `cv_frac_Epithelial`) are near-degenerate (competitor chooses the
  zero model, R² = 0.000 in all folds; no column reaches 0.002): drop or redefine them in v0.5.
