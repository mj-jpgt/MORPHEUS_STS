# Step-4 acceptance gate, re-run — PREDECLARATION

Written **before** the measuring script's criteria hash is frozen and before
any hybrid-vs-specialist number has been computed from the repaired encoder.
Amendments are **append-only** in §9; no criterion above §9 may be edited after
a result is seen.

This file does **not** supersede `PREDECLARATION_STEP4.md`. That file, its
criteria hash `8c3905d4851c4a50baa27da0e5dd65f9c5bd412a20183e5d10aecad1fad2253a`,
and its §9 A1 result stand in the record exactly as run: **all three arms
NON-EVALUABLE under validity check V2**, on a hybrid whose factor expert
trained for exactly one step. Nothing here deletes or rewrites that. The
re-run is a separate measurement of a repaired encoder, reported alongside it.

Gate text is unchanged (build plan step 4, `.claude/discov2.md:532`):

> "Require that the untrained-for-discovery hybrid representation approximately
> matches cNMF on additive worlds, appropriate nonlinear methods on manifold
> worlds and standard representation models on ordinary utility tasks. If it
> cannot subsume its constituent specialists, fix the architecture before
> proceeding."

This is a **parity** gate. An untrained encoder that beats every specialist in
every regime is a bug hypothesis, not a pass; §6's SUSPICIOUS flag is unchanged
and applies with more force this time, because the previous run's dead pathway
means a large sweep in *either* direction is a suspicious change and not a
result.

---

## 1. What changed since `PREDECLARATION_STEP4.md`, and what did not

**Two** things changed. Both are named here, before any number is produced.

### 1.1 The encoder was repaired

`PREDECLARATION_STEP4.md` §9 A1 attributed the failure to *optimization, not
representation*: the factor expert received gradient at step 0 and exactly
0.000e+00 at step 1 and every step after, so all three arms scored a randomly
initialised encoder plus one Adam step. Five defects were measured and pinned to
values by tests. The repair, its rationale, and its acceptance targets were
declared in advance in `morpheus_encoder/REMEDIATION_STEP4.md`, criteria sha256
`bbef651e0c03a5f2a4640dcbc2bc76415fb8a88e945a8cc4aa4ca03d72ea56e5`, and the
targets were fixed there before the repair existed.

The repaired encoder is what this run measures. Per-file sha256 of
`morpheus_encoder/` is recorded in the run receipt.

### 1.2 The representation width is reported at two values, not one

`PREDECLARATION_STEP4.md` §9 A1 §"Reason 2" recorded that **V2 fired on the
specialist side, with no reference to the encoder at all**: at d_rep = 16 the
best fitted specialist was within 0.065–0.082 of a random 16-column Gaussian
projection on all three arms, against a required margin of 0.10. The gate as
predeclared could not have certified subsumption from *any* encoder.

The instrument characterisation that followed was computed on **specialists and
the random-projection floor only**, over 3 seeds, with **no encoder of any kind
in it** — that is what makes it usable here rather than a result-driven choice:

| width | additive: best − floor | manifold: best − floor | utility: best − floor |
|---|---|---|---|
| d = 2 | 0.084 | 0.595 | — |
| d = 4 | 0.132 | 0.569 | — |
| **d = 8** | **0.144** | **0.173** | **0.134** |
| d = 16 (the original) | 0.053–0.072 | 0.076–0.082 | 0.068–0.091 |

**d_rep = 8 is the single width clearing V2 (≥ 0.10) on all three arms.**
d_rep = 16 is the worst of {2, 4, 8, 16} for both gating arms: `S_add`'s
max-over-components selection bias saturates, and at 16 dimensions a random
projection already preserves the neighbourhoods `S_man` asks about
(Johnson–Lindenstrauss).

**Changing a width after seeing a failure is goalpost-moving unless the change
is disciplined, so:**

1. the width was chosen by **encoder-blind** instrument characterisation — the
   hybrid was never scored at any candidate width before this file was written;
2. this file is hashed before the measuring script runs, and the script refuses
   to run against a changed criteria section;
3. **both widths are reported.** d_rep = 8 is the primary; d_rep = 16 is run in
   full, at the same seeds, with the same statistics, and reported in the same
   table. A width is not a filter you get to apply twice;
4. the original d_rep = 16 result stays in the record as NON-EVALUABLE with its
   validity numbers, unchanged;
5. no tolerance, statistic, world, seed, specialist or validity threshold moves.
   §§2, 5, 6 below are the previous file's §§2, 5, 6 verbatim in substance.

**What is not claimed.** d_rep = 8 clearing V2 is a property of the *instrument*
and says nothing about the encoder. If the gate passes at 8 and fails at 16, the
honest reading is "the arms are evaluable at 8 and were not at 16", and the
verdict at 16 is NON-EVALUABLE, not FAIL — exactly as V2 already specifies.

---

## 2. World selection — unchanged

```
ADDITIVE_WORLDS = ("additive_factors",)
MANIFOLD_WORLDS = ("manifold",)
UTILITY_WORLDS  = ("additive_factors", "hill", "threshold",
                   "context_gated", "xor")
SEEDS           = (0, 1, 2, 3, 4)
ROLE            = "proposal"
SECONDARY       = ("mixture",)
```

Reasons are those of `PREDECLARATION_STEP4.md` §2 and are not restated.

**Validity check V1 (integrity, pre-run), unchanged in substance and
strengthened in what it is bound to.** For every gate family, regenerate seeds 0
and 1 and require `World.fingerprint()` to equal the frozen manifest's. Any
mismatch aborts the cell.

**Disclosed: `dbench/MANIFEST.frozen.json` was rebuilt between the two runs, and
this file is therefore bound to world fingerprints rather than to the manifest
document hash.** The previous run asserted `manifest_sha256 =
b5c54d3d...956da85` on a 47622-byte file whose own sha256 was `0cfe3146...4ae6ec`
(`runs/gate_step4/PROVENANCE.json`, mtime 2026-09-06T15:21:11Z). The file now
present is 50899 bytes, carries `frozen_at_utc = 2026-09-06T17:48:44Z` and
`manifest_sha256 = d4f511d8824aa143b99b8af68ca076c45dbc6f065f8f9326e07cdd63a0e51ab6`.
It was not rebuilt by anything in this remediation; the change was found by the
re-run's own start-up check, which refused to proceed, and it is recorded here
rather than absorbed.

The document hash covers `frozen_at_utc`, so *any* re-freeze moves it whether or
not a world moved. What matters for this gate is whether the worlds moved, and
that is checkable directly and was checked before this paragraph was written:
**all 14 world fingerprints recorded in the previous run's cells
(`V1_fingerprint.expected`, 7 families x seeds 0 and 1) are byte-identical in
the new manifest.** The gate families are unchanged; the manifest grew.

The run therefore asserts, at start-up and as an abort condition:

1. every one of those 14 previously recorded fingerprints still matches the
   manifest now on disk — this is the substantive continuity check, and it binds
   this run to the same worlds the previous run measured;
2. per-cell V1 as before, regenerating each world and comparing to the manifest;
3. the manifest document's own sha256, size and `frozen_at_utc` are recorded in
   the receipt as observed, not asserted against a constant.

A whole-document hash that changes for a reason unrelated to the worlds is a
weaker instrument than the fingerprints it contains, and pinning it would have
forced a choice between aborting on a non-event and editing a constant after the
fact. `PREDECLARATION_STEP4.md`'s own §9 A2 already recorded that this freeze is
one boss world short of re-derivable (`boss3_hill_space_cause` is thread-count
dependent), so the document hash was known to be a fragile handle before this
run existed. None of the seven families this gate uses is affected.

---

## 3. Fitting protocol for the untrained hybrid — unchanged except the slot count

```
objective      reconstruction MSE of the observed modality X only
               = out.bank.per_expert["factor"].aux["recon_mse"]
               NO discovery loss, NO replication reward, NO null adversary,
               NO target Y anywhere in the fit
split          proposal split only; fit on the fit half, see §5
optimizer      Adam, lr = 1e-3                       (FIT_LR, unchanged)
steps          2000                                  (FIT_STEPS, unchanged)
batch          512 rows, sampled with replacement, torch generator seeded per run
encoder cfg    EncoderConfig(n_slots=d_rep, d_recon=d_obs, d_env=inp.d_env,
                             everything else at the shipped default)
device         cuda if available else cpu; dtype float32
seeding        torch.manual_seed(seed)
```

**No initialisation helper is called.** `init_dictionary`, `init_thresholds`
and `init_decoder_bias` all exist and all help; none is used here. The repaired
head has to leave the trivial solution under the same protocol that killed the
shipped one, or it has not been repaired. This is the previous run's protocol
verbatim.

**`n_slots = d_rep`.** This is the same rule already applied to every specialist
and not a new one: cNMF is *fitted* at `n_atoms = d_rep`, NMF and PCA at
`n_components = d_rep`, and the hybrid's object slots are its components. The
previous run held `n_slots = 16` because `d_rep` was 16; the two were never
independent.

Because that choice is arguable, the alternative is measured too and reported
as a **non-gating diagnostic**: `hybrid_wide_reduced`, the d_rep = 16 encoder's
factor readout PCA-reduced to 8 by the same §4 rule applied to every method.
If the two disagree materially that is a finding about width sensitivity and it
is reported, not hidden.

---

## 4. Readouts — unchanged

Common width **d_rep** for every method on every arm; any readout wider than
d_rep is reduced by PCA to d_rep components fitted on the fit half — one rule,
applied identically to every method, hybrid and specialist alike.

| arm | hybrid readout | native width |
|---|---|---|
| additive | `factor` expert loadings `W` | d_rep |
| manifold | `manifold` expert `chart_coords` `z`, flattened over slots, PCA→d_rep | 4·d_rep |
| utility | **joint**: `concat[W, relational activations, PCA_d(z)]`, PCA→d_rep | ≥ 3·d_rep |

On every arm the joint readout is also scored as a second, non-gating cell, and
the un-reduced manifold readout is reported as a diagnostic.

---

## 5. Comparison statistics — unchanged

Split: `rng(seed).permutation(n)`, first half fit, second half test.
Representations are fitted on the fit half and applied to all rows; supervised
readouts train on the fit half and score on the test half.

```
S_add(R)  = mean_j max_c |corr(R[:,c], L[:,j])|        on the test half
            L = generative latent columns, meta["derived_latents"] excluded
S_man(R)  = KNeighborsRegressor(n_neighbors=10) R2 of the chart coordinate t
S_util(R) = Ridge(alpha=1.0) R2 of world.target from standardised R
```

Secondaries, reported and non-gating: `S_add_ridge` (mean R2 of each latent
given R), branch ARI from `SpectralClustering(n_clusters=3,
affinity="nearest_neighbors", n_neighbors=12)`, and `mixture` under branch ARI.

**Specialist score per arm = the maximum over the arm's specialist set**, which
inflates the specialist and makes the bar harder for the hybrid. Specialist sets
and entry points are those of `PREDECLARATION_STEP4.md` §1 and §5, unchanged:

| arm | specialist set (all fitted at d_rep) |
|---|---|
| additive | cNMF (`cscd.missing_baselines._cnmf_one_modality`, n_runs=8), NMF, PCA (plain), PCA (step-3 relu convention) |
| manifold | kernel PCA (rbf), Isomap, LLE, Laplacian eigenmaps |
| utility | PCA (plain), PCA (step-3 relu), kernel PCA (rbf), NMF |

---

## 6. Decision rule — unchanged

```
TOLERANCE = 0.05

per-seed cell:  passes  iff  specialist_best - hybrid  <=  TOLERANCE
arm verdict:    PASS    iff  mean over the 5 seeds of (specialist_best - hybrid)
                             <= TOLERANCE
                        and  at least 3 of 5 per-seed cells pass
gate at width d: PASS   iff  all three arms PASS
                        and  no unresolved SUSPICIOUS flag
                        and  no arm is NON-EVALUABLE
step-4 gate:    PASS    iff  the gate PASSES at the primary width d_rep = 8
```

**Random-projection floor.** Per arm and seed, `R_floor = X @ G`,
`G ~ N(0, 1/d_obs)`, d_rep columns, same split, same statistic.

**V2 (discriminative power).** An arm is NON-EVALUABLE, not PASS, unless
`specialist_best - floor >= 0.10` on the seed mean.

**V3 (could this statistic have returned otherwise).** Report the full spread
`max - min` over {hybrid, every specialist, floor, raw X} on the seed mean. If
`< 0.05` the arm is NON-EVALUABLE.

**SUSPICIOUS flag.** Raise if `hybrid - specialist_best > +0.10` on the seed
mean in two or more arms. The named checks before it may be cleared:
(i) the fit never touches `world.target` or `world.latents`; (ii) test and fit
halves are disjoint in every readout; (iii) d_rep really is d_rep for the
hybrid; (iv) the specialists receive the same `X` the hybrid received.

**Failure is information.** If an arm fails, the required N+1 response separates
*representation* failure from *optimization* failure by reporting the fit-loss
trace, the per-step gradient-norm trace into adapters and slots, the eval-mode
live-code fraction, and the un-reduced readout diagnostic alongside the verdict.
Loosening `TOLERANCE`, changing a statistic, dropping an arm, or adding a third
width after seeing these two is not an available response.

---

## 7. Encoder-liveness receipt — new, and required

The previous run's headline defect was invisible in its own report: the fit-loss
trace moved by 1% and nothing recorded that the gradient was identically zero.
This run therefore records, for **every** hybrid fit in **every** cell:

- gradient L2 norm into `adapters.*` and into `slots.*` at steps
  {0, 1, 5, 20, 100, 299, 999, 1999};
- the training-mode live-code fraction at those steps;
- the eval-mode live-code fraction `mean(W > 0)` on all rows after the fit;
- the final reconstruction MSE.

**A cell whose minimum adapter- or slot-gradient norm past step 1 is 0.0, or
whose eval-mode live-code fraction is 0.0, is reported as a dead-pathway cell
and its arm is NON-EVALUABLE** — the failure mode of the previous run is now a
declared abort condition rather than something to be discovered afterwards.

---

## 8. What this gate does not claim

It does not claim the hybrid *beats* any specialist; it claims parity within
0.05. It does not test discovery — no hypothesis is compiled, no replication is
measured, no null is run. It does not use any artifact from
`dbench/adversarial.py`; those are held out for post-training evaluation and are
not instantiated here or allowed to touch any hyperparameter.

---

## 9. Amendments (append-only)

### A1 - 2026-09-06 - why the frozen manifest changed: a documented re-freeze, not drift

Section 2 recorded that `dbench/MANIFEST.frozen.json` had been rebuilt between
the two runs (47622 -> 50899 bytes, `manifest_sha256` b5c54d3d... -> d4f511d8...),
that it was not rebuilt by this remediation, and that all 14 gate-family world
fingerprints were byte-identical. That was the *fact*. This is the *reason*,
supplied by the agent that made the change, and it is recorded here so the
change is a provenance record rather than a provenance gap.

`util.gp_field`, the squared-exponential GP sampler behind every spatial world,
was **thread-count dependent**: both `np.linalg.cholesky` and the `L @ w`
mat-vec move under OpenBLAS at different thread counts. The magnitude is about
2.4e-11 relative - statistically inert, byte-fatal, and amplified because
`zscore`'s standard-deviation reduction crosses an ulp boundary and shifts every
element of the result.

Scope, which is why the manifest grew: 115 worlds were re-fingerprinted at 1, 4
and 8 threads; **15 moved, across exactly the three `gp_field`-calling families**
(`spatial_interaction`, `boss1_cond_thresh_space_lag`, `boss3_hill_space_cause`),
and 9 of the 69 private-split worlds were among them. The fix is at the source -
lattice fields are now drawn by circulant convolution of white noise with a
periodic Gaussian kernel, so there is no BLAS call at all and the result is
bit-identical at any thread count and under any BLAS build. Pinning
`OMP_NUM_THREADS` was considered and deliberately rejected, because a freeze that
is only reproducible under one environment variable is not a freeze. Post-fix,
0 of 115 worlds are unstable. `b5c54d3d...` is retained in the new manifest's
`provenance.thread_reproducibility.superseded_manifest_sha256`.

This **supersedes `PREDECLARATION_STEP4.md` section 9 A2**, which recorded
`boss3_hill_space_cause` as thread-count dependent and its private-seed
commitment as unverifiable by regeneration. A2 was correct about the symptom and
identified one of the three affected families; the cause was `gp_field`, and it
is fixed rather than merely characterised. A2 is not edited - amendments are
append-only in both files - and this paragraph is the pointer.

**Two consequences for this gate, both checkable and both checked.**

1. `np.linalg.lstsq` and `np.linalg.qr` are thread-**stable**. That is what
   confined the damage to `gp_field` callers rather than to all 23 families, and
   it is why no specialist score or gate verdict in either run was ever affected.
2. **None of the seven families this gate uses calls `gp_field`**, which is the
   mechanism behind section 2's observation that all 14 fingerprints were
   byte-identical. That observation is therefore not a lucky coincidence; it is
   what the scope of the defect predicts.

**One hazard this gate does own, measured rather than assumed.** The same agent
confirmed that `np.linalg.solve` at n=250 also moves with thread count. It is on
no world-generation path, but it *is* on this gate's readout path: `S_util` and
the `S_add_ridge` secondary both use `sklearn.linear_model.Ridge`, whose default
dense solver is a Cholesky solve. The systems are small (d_rep x d_rep, i.e.
8x8 or 16x16), which is the opposite end from n=250, but "small so probably
fine" is an assumption and this file does not run on assumptions. The measured
answer is in `runs/gate_step4b/thread_stability.json` and is summarised in the
amendment that reports the gate result. Nothing in section 6 changes either way:
if a statistic moved at 1e-11 it is inert against a TOLERANCE of 0.05, and the
point of measuring is to be able to say so with a number instead of a shrug.


### A2 - 2026-09-06 - result of the re-run: the pathway is live and the gate does not pass

Run against this file at criteria sha256
`fee557514ff2fa9613b3d6ec12a0c524d002e6ba5733abe349320804cbb7beec`, asserted at
start-up, together with `PREDECLARATION_STEP4.md` at
`8c3905d4851c4a50baa27da0e5dd65f9c5bd412a20183e5d10aecad1fad2253a` (unchanged)
and the 14 world-fingerprint continuity check (14/14). 35 cells, 7 families x 5
seeds, 70 encoder fits. Full table in `runs/gate_step4b/gate_report.json`,
rendered by `scripts/gate_step4b_table.py`.

**Verdict: FAIL at d_rep = 8, NON-EVALUABLE at d_rep = 16. Step 6 is not
entered.**

#### The encoder is alive. Section 7's abort condition never fired.

| quantity | pre-repair (A1) | this run, 70 fits |
|---|---|---|
| gradient into adapters past step 1 | **0.000e+00** | min **4.70e-03**, never zero |
| eval-mode live-code fraction | **0.000** | mean **0.2595**, min 0.0678, never zero |
| dead-pathway cells (section 7) | (condition did not exist) | **0 of 35 at either width** |

That is the repair, confirmed at gate scale rather than on one probe. The
collapse still happens - the code is exactly 0.0000 at step 1 in every cell -
but it is **no longer absorbing**: the atoms come back over the following
hundreds of steps. The `legacy()` control reproduces A1's exactly-zero gradient
on demand, which is what licenses reading the difference as caused by the
repair rather than by anything else that changed.

#### d_rep = 8 (primary): all three arms evaluable, all three fail on parity

| arm | family | hybrid | specialist best | shortfall | random floor | hybrid - floor (SE) | per-seed pass | verdict |
|---|---|---|---|---|---|---|---|---|
| additive | additive_factors | 0.3916 | 0.6033 | 0.2117 | 0.4549 | -0.0633 (0.0398) | 0/5 | **FAIL** |
| manifold | manifold | 0.7273 | 0.9621 | 0.2348 | 0.7422 | -0.0149 (0.0390) | 0/5 | **FAIL** |
| utility | additive_factors | 0.2155 | 0.6453 | 0.4298 | 0.3698 | | 0/5 | |
| utility | hill | 0.1689 | 0.4355 | 0.2666 | 0.1677 | | 0/5 | |
| utility | threshold | 0.1381 | 0.2520 | 0.1140 | 0.1195 | | 1/5 | |
| utility | context_gated | -0.0024 | 0.0370 | 0.0394 | -0.0079 | | 4/5 | |
| utility | xor | 0.0032 | 0.0215 | 0.0183 | -0.0054 | | 4/5 | |
| **utility** | **(all)** | **0.1047** | **0.2783** | **0.1736** | **0.1287** | **-0.0241 (0.0167)** | **9/25** | **FAIL** |

Validity, all clearing: V2 (specialist minus floor) = **+0.1484 / +0.2199 /
+0.1495** against the 0.10 bar; V3 (spread over all methods) = 0.2155 / 0.2348 /
0.1582 against the 0.05 bar. The arms are measurable and the statistics do
separate the methods, which is exactly what was not true at d = 16.

#### d_rep = 16: preserved in the record as NON-EVALUABLE

| arm | hybrid | specialist best | shortfall | floor | V2 (need >= 0.10) | verdict |
|---|---|---|---|---|---|---|
| additive | 0.4921 | 0.6022 | 0.1101 | 0.5306 | **+0.0715 FIRED** | NON-EVALUABLE |
| manifold | 0.8280 | 0.9475 | 0.1196 | 0.8653 | **+0.0822 FIRED** | NON-EVALUABLE |
| utility | 0.1983 | 0.2752 | 0.0769 | 0.1842 | **+0.0910 FIRED** | NON-EVALUABLE |

These reproduce the encoder-blind instrument characterisation to three decimals
(0.0715 / 0.0822 / 0.0910), on the specialist side, with the encoder playing no
part in the number. Section 1.2's claim that the width was chosen blind is
therefore checkable after the fact and checks out. **d = 16 is not reported as a
failure of the encoder; it is a width at which this gate cannot certify anything
about any encoder**, which is what V2 exists to say. The original run's d = 16
verdict stands unamended.

#### Parity, not victory: the SUSPICIOUS flag did not fire, in either direction

`hybrid - specialist_best > +0.10` in two or more arms: **not raised at either
width**, no arm beaten. The pre-repair signature was `additive_factors` hybrid
0.0000 against PCA 0.6022 - a dead pathway. A repaired encoder that now swept
*past* fitted specialists everywhere would deserve the same suspicion in the
other direction, and it does not: on every gating arm the hybrid sits **at or
just below** the random-projection floor (-0.063 +/- 0.040, -0.015 +/- 0.039,
-0.024 +/- 0.017), i.e. within about 1.6 standard errors of it.

#### Could each cell have returned otherwise? Per cell, not just per arm.

Checked for all 70 cells: within-cell spread over {every specialist, hybrid,
random floor, raw X}. **63 of 70 cells have spread >= 0.05 and could have
separated the methods. 7 could not**, and they are not scattered:

| width | non-discriminating cells | spread |
|---|---|---|
| d = 8 | context_gated s3, xor s1, xor s4 | 0.027 - 0.033 |
| d = 16 | context_gated s3, xor s1, xor s2, xor s4 | 0.030 - 0.050 |

All seven are `context_gated` and `xor` on the utility arm, where **no method
beats R2 ~ 0.06, including raw X** - a property of those families' targets under
a linear readout, not of the encoder. This matters because it is where the
per-seed passes come from: those two families supply 8 of the 9 passing cells at
d = 8. Restricting to cells that can discriminate:

| width | arm | all cells | discriminating cells only |
|---|---|---|---|
| d = 8 | utility | 9/25 pass, shortfall 0.1736 | **6/22 pass, shortfall 0.1974** |
| d = 16 | utility | 10/25 pass, shortfall 0.0769 | 6/21 pass, shortfall 0.0921 |

The verdict is FAIL either way, so this changes nothing about the outcome - but
a pass earned on a cell where nothing works is the `gate_v1`/`v2` false-pass
pathology, and it is reported rather than left in the pass count unremarked.
The N+1 for the gate itself is to promote V3 from an arm-level check to a
per-cell one, so a non-discriminating cell is excluded rather than counted.

#### Two audit findings checked against this gate before the table was assembled

1. **The parity statistic does not inherit step 3's unbudgeted recall.** The
   step-3 scorer's `conjunctive_recall = conj_hit / 3` has no flag budget and is
   bounded below by the flag rate, so it compares flag rates rather than methods.
   This gate never calls it: grep over both run scripts returns no hit for
   `score_run`, `conjunctive_recall`, `align_atoms_to_factors`, or any p-value or
   flag machinery, and the only three things imported from the prior project
   (`pca_fit_transform`, `nmf_fit_transform`, `_cnmf_one_modality`) are
   representation fitters returning a matrix. `S_add`, `S_man` and `S_util` flag
   nothing - there is no hypothesis space and no flag set anywhere in the scoring
   path, so there is no flag rate that could differ between arms.
   The one budget that *does* exist is `S_add`'s `max_c` search over the columns
   of R, and section 4's width rule fixes it. Audited rather than asserted
   (`scripts/budget_audit.py`, `runs/gate_step4b/budget_audit.json`): **every
   method on every arm at both widths realises exactly d_rep columns** - cNMF,
   NMF, PCA, PCA-step3, kernel PCA, Isomap, LLE, Laplacian eigenmaps, the random
   floor and the hybrid factor readout alike. The arms are budget-matched by
   construction, and now by measurement.
2. **`sparse.sparsity_stats["max_offsupport_mass"]` is inert and is not cited
   anywhere in this result.** It computes `zero = (g.abs() <= 0.0)` and then
   takes `max |g|` over that set - the maximum of a quantity over the set where
   it is <= 0 - so it cannot return anything but 0.0. Confirmed here on softmax,
   uniform noise, all-ones, a constant, a genuinely sparse tensor and a uniformly
   1e-9 tensor: 0.0 for all six. It appears in every `cell_*.json` under
   `gate_stats` and is ignored. The live statistic is `frac_exact_zero`, and the
   live evidence for this run's sparsity is the eval-mode code fraction above
   (mean 0.2595 non-zero, i.e. ~74% exact zeros). At `tol=1e-3` the statistic
   does separate; that change belongs with a re-run of everything that reports
   it, not with this one, and `test_experts.py` now carries a live
   `frac_exact_zero` assertion beside the vacuous one.
   No step-3 aggregate number is cited in this gate - only step 3's entry points
   and hyperparameters - so the one-cell-stale aggregate does not reach it.

Thread-count reproducibility of the readout path was measured separately
(`runs/gate_step4b/thread_stability.json`): representations move by up to
4.37e-11 across 1/4/8 threads, every gate statistic by at most **2.68e-14**
against a tolerance of 0.05, and both gate-family world fingerprints are
identical across thread counts.

#### Failure-mode attribution

Section 6 requires the failure to be attributed rather than reported. It is a
**representation** failure, and the other four candidates are excluded by
measurement rather than by argument:

* **not optimization.** 70/70 fits carry non-zero gradient into adapters and
  slots past step 1 (min 4.70e-03), eval-mode live code averages 0.2595 and is
  never zero, reconstruction MSE falls throughout, and section 7's dead-pathway
  condition never fires. This is precisely the failure A1 had, and it is gone.
* **not benchmark/instrument.** At d = 8, V2 clears on all three arms
  (+0.148 / +0.220 / +0.150) and V3 clears on all three (0.216 / 0.235 / 0.158).
  The instrument can tell these methods apart; it says the hybrid is not among
  the good ones. The seven non-discriminating cells are named above and removing
  them makes the shortfall worse, not better.
* **not power.** 5 seeds per arm; the hybrid-minus-floor standard errors are
  0.017-0.040 while the shortfalls against specialists are 0.21-0.43, i.e. 5 to
  25 standard errors. No plausible increase in seeds closes that.
* **not hypothesis-language.** This gate compiles no hypothesis, matches no
  object and runs no null; there is no language in which the claim could have
  been mis-stated.
* **representation.** After 2000 steps of reconstruction-only fitting, the
  untrained-for-discovery hybrid's readout carries no more of the generative
  structure than a random projection of the same width, on all three statistics.
  The architecture can now *train*; what it trains to, under reconstruction
  alone, is not a representation that competes with cNMF on additive worlds or
  with kernel PCA on manifold worlds.

#### The N+1

The gate text says "if it cannot subsume its constituent specialists, fix the
architecture before proceeding", and that is now a well-posed engineering target
rather than a blocked measurement. Named next steps, in the order their evidence
supports:

1. **The factor readout is the weakest arm and the one with a mechanical
   suspect.** `S_add` maxes over columns; the hybrid's code is sparse (74% exact
   zeros) while every specialist's is dense, so a sparse code is being scored by
   a statistic that rewards dense coverage. Measure `S_add` on the *dense
   pre-activation* alongside the gated code - if the gap closes, the shortfall
   is a readout mismatch rather than a representation deficit, and that is a
   different fix.
2. **Reconstruction-only fitting may be the wrong objective for this
   comparison.** cNMF is fitted to reconstruct with a non-negativity prior that
   is exactly the generative form of `additive_factors`; the hybrid is fitted to
   reconstruct with far more capacity and no such prior, and 2000 steps at
   lr 1e-3 leaves live code still climbing at step 2000. Both a longer budget
   and an explicit sparsity/non-negativity pressure are cheap to test.
3. **Slot count is confounded with width.** n_slots = d_rep was chosen so the
   hybrid's component count matches every specialist's. The non-gating
   `hybrid_wide_reduced` diagnostic (the d = 16 encoder PCA-reduced to 8) is in
   every cell record and separates "the encoder needs more slots" from "the
   readout needs fewer columns".
4. **Promote V3 to a per-cell check**, so the seven non-discriminating cells are
   excluded rather than counted as passes.


### A3 - 2026-09-06 - the N+1 on A2's FAIL: three iterations, predeclared before any of them ran

A2 attributed the d_rep = 8 failure to **representation**, having excluded
optimization, benchmark, power and hypothesis-language by measurement. This
amendment declares the three iterations that follow from A2's own N+1 list,
**in its order of suspicion**, together with what each would show and what
result sends the work to the next one. It is written before any of the three
has been run and before any number below the A2 table has been computed.

**No criterion in sections 1-8 moves.** `TOLERANCE` stays 0.05, `S_add` /
`S_man` / `S_util` stay, the seeds stay, the specialist sets stay, V2 and V3
stay at 0.10 and 0.05, the fitting protocol of section 3 stays. Everything
here is either a **parallel column** on the existing statistic, a **training
budget**, or a **diagnostic the run was supposed to emit and did not**. No
slot, expert, router or loss is added; the build plan's step 6 is not entered.
If any iteration reaches parity, the response is to predeclare the new setting
and re-run the **full** gate under it, reporting the old and the new table
side by side - not to report the iteration's own subset as a gate result.

The criteria hash of sections 1-8 is unchanged at
`fee557514ff2fa9613b3d6ec12a0c524d002e6ba5733abe349320804cbb7beec`; the
whole-file sha256 immediately before this amendment was
`22a74214082bc744572862ef8999b6e46d97812dbd414ef6910c6ac3b8b55cdd`. The
whole-file sha256 with this amendment appended, frozen before the first
iteration ran, is recorded in `PREDECLARATION_STEP4B.sha256.json` under
`a3_freeze`.

---

#### Iteration 1 - is `S_add` judging a sparse code by a coverage statistic?

The suspicion, stated as a mechanism rather than a mood: `S_add(R) =
mean_j max_c |corr(R[:,c], L[:,j])|` credits a latent to whichever column
correlates with it best, so a representation is rewarded for **covering** the
latent set densely. The hybrid's factor readout is 74% exact zeros
(eval live-code 0.2595) while every specialist's is dense. This is the same
*shape* as the step-3 finding that `matching.align_atoms_to_factors` capped
PCA's whole column at 0.222 before any engine ran
(`morpheus_encoder/step3/AUDIT_FINDINGS.md` section 1). It is a measurement
question and it is asked before any training question.

**I1.1 - known-answer, encoder-free, run first.** On `additive_factors`,
seeds 0-4, proposal role, same split as section 5, at d_rep = 8, construct
codes that **provably contain** the planted additive factors and score them
with `S_add` and with the `S_add_ridge` secondary:

| code | construction |
|---|---|
| `dense_oracle` | `L` itself, the 8 generative latent columns |
| `col_mask(p)` | `L` with, per column, all but the largest-magnitude fraction `p` of entries set to exact 0, for `p` in {1.00, 0.50, 0.2595, 0.10} |
| `row_topk(k)` | `L` with, per row, all but the `k` largest-magnitude entries set to exact 0, for `k` in {8, 4, 2, 1} - the shape a JumpReLU/top-k code actually takes |
| `col_mask_shuffled(p)` | `col_mask` whose support is drawn at random instead of by magnitude, at the same `p` - separates "sparse" from "sparse in the informative places" |

`p = 0.2595` and `k = 2` are the gate's own measured live-code fraction and
its rounding to an integer per-row budget; they are read off A2 and are not
tuned.

Declared readings, fixed now:

* `S_add(dense_oracle)` must be **1.000**. If it is not, the harness is wrong
  and nothing else in iteration 1 is readable.
* If `S_add(row_topk(2)) >= 0.90 x S_add(dense_oracle)`, then at the gate's
  own sparsity level `S_add` does **not** materially penalise sparsity, and the
  scorer is **exonerated as an explanation of a 0.21 shortfall**. Iteration 1
  proceeds to I1.3 only as a confirmation and the work moves to iteration 2.
* If `S_add(row_topk(2)) <= 0.4549` - the additive arm's measured random
  floor - then a code that provably contains the factors scores at or below a
  random projection, `S_add` is the defect, and the correct response is to
  report the gate's additive arm as **NON-EVALUABLE for sparse readouts** and
  to name the statistic that would replace it, not to re-score the hybrid
  under a friendlier number.
* Between those two, the penalty is partial and is reported as a number:
  `1 - S_add(row_topk(2)) / S_add(dense_oracle)` is the fraction of the
  hybrid's shortfall that sparsity alone can account for.

Every value is pinned in `runs/gate_step4b_iter/known_answer_sadd.json`.

**I1.2 - matched sparsification control, encoder-free.** The oracle answers
"can `S_add` see a sparse copy of the truth". This answers "what does the same
sparsification do to a real dense representation": refit cNMF, NMF, PCA,
PCA-step3 and the random floor at d_rep = 8 on the same split, apply the
identical `row_topk(2)` mask to each, and report `S_add` before and after.
Declared reading: the mean drop across the five is the scorer's sparsity
penalty measured on real representations; if it is below 0.05 the scorer is
exonerated on this evidence too, and if it exceeds the 0.2117 additive
shortfall the scorer alone could account for the whole result.

**I1.3 - the dense pre-activation column, on every cell.** For all 35 cells,
at d_rep = 8, refit the hybrid under the section 3 protocol **unchanged** and
score, in the same process and on the same split:

* `hybrid` - the gated code `W`, a reproduction of A2's column;
* `hybrid_dense_preact` - `head.pre_activation(u)`, the same tensor **before**
  the JumpReLU gate, same width, same section 4 PCA rule;
* every specialist, the random-projection floor, and raw `X`, all re-fitted and
  re-scored in the same process.

So that the hybrid is not given a second scoring the others do not get, the
`_dense` column is reported for **every** method. For cNMF, NMF, PCA,
PCA-step3, the floor and raw `X` the dense representation **is** the
representation, so the two columns are the same number by construction; that
is stated in the table rather than presented as a second measurement. The
column that can move is the hybrid's, and the point of reporting the others is
that the comparison is against the same bar.

Declared readings:

* **reproduction first.** The refit `hybrid` column must land within 0.05 of
  A2's per-arm value. A larger drift is a nondeterminism finding about the run
  and is reported *before* anything is read off the new column.
* if `hybrid_dense_preact - floor >= 0` on the seed mean of an arm while
  `hybrid - floor < 0` on the same arm, the shortfall is **partly the readout
  and the scorer**, and the record must say: *the sparse readout is judged by a
  coverage statistic*. That is a finding about the gate, not about the encoder,
  and it does not by itself turn a FAIL into a PASS - parity against the
  specialist best is still the gate.
* if neither column clears the floor, **the scorer is exonerated** and the work
  moves to iteration 2.
* if the two columns are **identical on every cell**, that is a wiring bug
  hypothesis before it is a finding, and the iteration is not reported until
  the pre-activation tensor is shown to differ from the gated one.

**I1.4 - what a pass would have to look like.** If iteration 1 produced a
column at parity (shortfall <= 0.05 against the specialist best on all three
arms), the readout definition in section 4 would have changed, which is a
criterion. It would therefore be predeclared as a **new** width/readout rule
in a further amendment and the **full** gate re-run under it, with A2's table
reprinted beside it. Reporting I1.3's numbers as the gate result is not an
available response.

---

#### Iteration 2 - training budget, run only if iteration 1 does not resolve it

A2's own trace has live code at 0.2732 at step 1999 and **still climbing**, and
`REMEDIATION_STEP4.md` R2 missed its 400-step target by 0.001 and reached
0.3964 at 2000 on the additive world. The hypothesis is that 2000 steps at
lr 1e-3 measures a representation mid-recovery.

**Predeclared subset, fixed before running:** **seed 0 of each of the seven
gate families** (`additive_factors`, `manifold`, `hill`, `threshold`,
`context_gated`, `xor`, `mixture`), at **d_rep = 8 only**. One seed per family,
as declared; no family is dropped and no seed is chosen after seeing a result.

**Protocol:** section 3 verbatim except `FIT_STEPS`. One fit per cell run to
**20,000 steps** (10x the gate's budget), with the full readout extracted and
scored at checkpoints **{2000, 4000, 8000, 14000, 20000}**. Step 2000 is the
anchor and must reproduce the gate's cell to within 0.05. Specialists and the
floor are refit once per cell in the same process and are **budget-invariant**
by construction - they do not get a longer fit, because "the same everything
else" is what makes the trajectory readable.

Reported per checkpoint: eval live-code fraction, final recon MSE, the arm
statistic for the hybrid, the specialist best, the random floor, and
`min(W)` (the non-negativity check below).

Declared readings:

* **parity at any checkpoint** - `specialist_best - hybrid <= 0.05` on an arm -
  means the gate's budget was the ceiling. The response is to predeclare the
  new budget in a further amendment and re-run the **full** gate (35 cells,
  both widths) at it, reporting both tables. It is not to report this subset.
* **plateau below the floor** - the change in the arm statistic over the last
  two checkpoints is `< 0.01` and the hybrid is still below the random floor -
  **exonerates budget**, and the failure stands as attributed.
* **still rising at 20,000 without parity** is reported as *not converged at
  10x the gate budget*. It is not a pass and it is not a plateau; it is a
  statement that the ceiling was not located within a 10x budget.

**Non-negativity pressure is not added pre-emptively.** It is considered only
if (a) the additive arm alone fails at 20,000 while another arm reaches parity,
**and** (b) `min(W) < 0` in the fitted code, i.e. the factor expert's code has
actually gone signed. Both are checked and reported as numbers. If (b) is false
- and the JumpReLU forward `a . 1[a > theta]` with `theta > 0` says it should
be - then "add a non-negativity prior" is a change to the architecture and it
belongs to step 6's named capability, not here.

---

#### Iteration 3 - slot count vs readout width, and a diagnostic that was never computed

**Stated before running, because it changes what iteration 3 is:
`hybrid_wide_reduced` is `null` in all 70 entries of
`runs/gate_step4b/gate_report.json` and absent from all 35 `cell_*.json`.** The
section 3 diagnostic was never computed. The cause, named here so it is a
provenance record and not a later discovery:
`scripts/gate_step4b_run.py:302-304` computes

```
wide = hyb_by_slots.get(max(D_REPS))
R_factor_wide = (pca_to(wide["W"], fit, d_rep)
                 if wide is not None and wide is not hyb else None)
```

**inside** the `for d_rep in D_REPS` loop with `D_REPS = (8, 16)`. At `d_rep=8`
the d=16 fit does not exist yet, so `.get(16)` is `None`; at `d_rep=16`
`wide is hyb`. The branch is unreachable at both widths, so the field is
`None` in every cell and `aggregate_width` faithfully propagated 70 nulls. A2's
N+1 item 3 states the diagnostic "is in every cell record"; it is in every cell
record as a null. Iteration 3 therefore **computes** the quantity rather than
reading it.

**Measurement:** for all 35 cells, fit the hybrid at `n_slots = 16` under the
section 3 protocol and reduce its factor code to 8 columns by the section 4
PCA rule fitted on the fit half - the same rule every over-wide specialist
readout already gets. Score it on each arm the cell carries, against **the same
cell's d_rep = 8 specialists and d_rep = 8 random floor**, so the comparison is
at one width and only the slot count differs. Report `hybrid` (8 slots),
`hybrid_wide_reduced` (16 slots, reduced to 8) and `hybrid_dense_preact_wide`
in the same table.

Declared readings:

* wide-reduced clears the floor where the narrow one does not: the binding
  constraint is the **8-slot bottleneck**, not the objective. Because d = 16
  fails V2 for reasons that have nothing to do with any encoder (A2: +0.0715 /
  +0.0822 / +0.0910 against a 0.10 bar, reproducing the encoder-blind
  characterisation), that is a **genuine tension in the gate's design** - the
  only width at which the instrument discriminates may be a width at which the
  encoder is bottlenecked - and it is reported as such, not pared down to
  whichever width flatters the encoder.
* both clear or neither clears: slot count is **not** the binding constraint,
  and the attribution in A2 stands unchanged.
* wide-reduced clears **and** reaches parity with the specialist best: same
  rule as everywhere else in this amendment - predeclare the new slot rule and
  re-run the full gate.

---

#### F1 - `sparsity_stats["max_offsupport_mass"]` is replaced by a statistic that can fail

`sparse.py:238-248` computes `zero = (g.abs() <= tol)` with `tol = 0.0` and
then `max |g|` over that set: the maximum of a quantity over the set on which
it is at most zero. It can only return 0.0, which the step-3 audit confirmed on
six inputs including softmax, uniform noise, all-ones and a constant. It is
recorded in every `gate_step4*/cell_*.json` under `gate_stats`.

The fix, declared before it is written: give `sparsity_stats` a **default
`tol` that separates**, and report the off-support mass so that the number can
fail. `tol = 1e-3` is the value the audit measured as separating (entmax
6.43e-4 vs softmax 9.86e-4) and is used unchanged. `test_sparse.py:28`'s
`== 0.0` assertion is replaced by a test that asserts a **value** on a
known-sparse input **and** on a known-dense input and fails if the two are
equal - the property the current test cannot have, since it asserts 0.0 on a
single input.

Every field in every affected receipt is either regenerated or marked
`not_measured` with a pointer to this paragraph. **No gate verdict, arm score
or validity number changes**, because A2 already records that the statistic is
cited nowhere in the result; the fix is to the receipt, not to the finding.

#### F2 - V3 is promoted from an arm-level check to a per-cell one

A2's own analysis found 7 of 70 cells with within-cell spread over
{every specialist, hybrid, floor, raw X} below 0.05, all of them
`context_gated` or `xor` on the utility arm, where no method beats R2 ~ 0.06
including raw `X`; those two families supply 8 of the 9 passing cells at d = 8.
V3 as written in section 6 is an **arm** check, so those cells count as passes.
Promoted to a per-cell check, a cell whose spread is `< V3_SPREAD` is
**excluded** from both the pass count and the seed mean rather than counted.

This is applied to the existing d = 8 and d = 16 records as a **re-reading**,
not a re-run, and both readings are printed. A2 already reports the direction:
6/22 rather than 9/25 at d = 8, with the shortfall **worse** (0.1974 vs
0.1736). The verdict does not change either way, which is why this is a fix to
the instrument's bookkeeping and not a result.

---

#### What the three iterations cannot do

None of them can turn A2's FAIL into a PASS by itself. I1.3 can move the
attribution from "the encoder" to "the encoder and the way it is read"; I2 can
move it to "the budget"; I3 can move it to "the width". Each of those, if it
reaches parity, obliges a fresh predeclaration and a **full** re-run, and this
paragraph exists so that obligation is on the record before the numbers are.
If all three fail, the report says in one sentence **which capability is
missing**, because that sentence is what step 6 needs and it is the only thing
that licenses adding architecture.


### A4 - 2026-09-06 - iteration 2 and part of iteration 3 move to a second machine, and what that costs

Written **before** any number was computed on the second machine, and before
the first process was launched there. A3's criteria are unchanged; this
amendment declares only *where* two of its iterations run, what that changes,
and the check that replaces nothing.

**Why.** The machine the gate ran on is a 16 GB laptop shared with nine other
build agents. Iteration 1's known-answer work (A3 I1.1/I1.2) and the first
20 cells of I1.3/iteration 3 completed there. The 20,000-step budget extension
did not: the chained job died with `torch.OutOfMemoryError` while 6.68 GiB of
8 GiB of device memory was free, i.e. the allocation failed for host reasons,
and the coordinator reports the sibling compiler agent's blocks were
OOM-killed three times on the same host. A predeclared cell lost mid-run is
worse than a cell run elsewhere and said so.

#### A4.1 - the obstacle that had to be measured first, not assumed away

`PREDECLARATION_STEP4B.md` section 2 binds V1 to `World.fingerprint()`. **That
handle is not portable, and the world it names is.** Measured before anything
was moved (`runs/gate_step4b_iter/world_fingerprint_portability.json`), on
`additive_factors|s0|proposal`, between the gate's machine
(Python 3.13.5 / numpy 2.4.3 / Windows) and the box
(Python 3.10.12 / numpy 2.2.6 / Linux), from the same source at the same seed:

| component | same? |
|---|---|
| all eight generative latents `A0-A2, D0-D4` | **bit-identical** |
| the target `Y` | **bit-identical** |
| the observed modality `X` | **75,289 of 144,000 entries differ** |
| `TruthSet` `effect` scalars | differ in the last ulp (0.6175192229192265 vs ...266) |
| `fingerprint()` | **differs** |

The whole divergence in `X` is `max |Δ| = 6.66e-16` absolute. Against a gate
`TOLERANCE` of 0.05 that is inert by fourteen orders of magnitude; against a
sha256 it is fatal. This is the same *shape* as A1's `gp_field` finding and a
**different axis** -- interpreter and library build rather than thread count --
and `additive_factors` never calls `gp_field`, so it is not that defect
resurfacing. A1 fixed its axis at the source; this one is recorded, not fixed,
because fixing it means pinning an interpreter and A1 already rejected a freeze
that is only reproducible under one environment variable.

#### A4.2 - V1 is not loosened; the world stops being regenerated

The available responses were: (a) run the box's own worlds and drop V1;
(b) replace the byte check with a numerical tolerance; (c) stop regenerating.
(a) and (b) both weaken the check that exists to say "this is the world the
gate measured". So (c):

`scripts/world_bytes.py --export` writes every array the fingerprint hashes
(`modalities`, `latents`, `target`) for the 23 declared cells from **the
machine the manifest was frozen on**, having first asserted that each world's
`fingerprint()` equals the frozen manifest's. On the box, `load_world`
regenerates the world, **measures and records** the per-array deviation from
the shipped bytes, refuses if any array deviates by more than `1e-9` (which
would be a real difference rather than a build difference), substitutes the
shipped arrays, and then runs **section 2's check unchanged**: `fingerprint()`
must equal the frozen manifest's, byte for byte, or the cell aborts.

So the check that executes on the box is the same equality on the same hash
of the same bytes. What changed is that the bytes are transported instead of
recomputed. Both the regeneration deviation and the post-substitution
fingerprint are in every cell receipt, so the substitution is auditable rather
than implicit. For seeds 2-4, which the manifest does not carry (section 2
freezes seeds 0 and 1), the same equality is asserted against the exporting
machine's fingerprint, which is strictly more than the local run checked.

#### A4.3 - the environment difference that remains, and the control for it

`torch` is pinned to **2.6.0+cu124** on the box, the gate's exact version, in a
venv built for this work; `numpy`/`scipy`/`scikit-learn` are the closest the
box's Python 3.10 admits (2.2.6 / 1.15.3 / 1.7.0 against 2.4.3 / 1.16.0 /
1.7.0). The GPU differs and cannot be matched: A100-SXM4-40GB against the
laptop's RTX 4060. cuBLAS/cuDNN kernel selection differs by architecture, so
the fits will not be bitwise identical to the gate's and no claim here assumes
they are.

Two things make that readable rather than a confound:

1. **Iteration 2 is internally matched.** Every trajectory point is compared
   against the specialist best and the random floor computed *in the same
   process, on the same world, at the same width*. Nothing in its declared
   reading -- parity, plateau, or still-rising -- is a cross-machine
   comparison.
2. **The anchor is the measurement of the environment change, not an
   assumption about it.** A3 already requires the step-2000 checkpoint to
   reproduce the gate's own cell to within 0.05. That check now does double
   duty: it is the quantitative statement of how much the A100 and the
   laptop disagree under an otherwise identical protocol. If it clears, the
   move is demonstrated inert at the gate's own tolerance. **If it does not
   clear, iteration 2 is reported as a within-box trajectory and the
   cross-machine delta is reported beside it as its own finding** -- it is not
   quietly averaged into A2's table.

Additionally, `additive_factors|s0` and `manifold|s0` are run for I1.3 /
iteration 3 on **both** machines and reported as an explicit
cross-environment control (box cells carry a `_box` suffix so they sit beside
the laptop's rather than overwrite them). That turns "does the machine
matter?" into a number on the two cells that matter most, at no extra
argument.

#### A4.4 - what is not moving

Iteration 1's known-answer and matched-sparsification work (I1.1, I1.2) ran on
the gate's own machine and is complete; it is not re-run. The 20 I1.3 /
iteration 3 cells already in flight on the laptop finish there rather than
being duplicated. Nothing in A3's criteria, thresholds, statistics, seeds,
specialists or fitting protocol changes, and no result already recorded is
re-scored.


### A5 - 2026-09-07 - the three iterations, measured. The gate still does not pass, and the missing capability now has a name.

Run against sections 1-8 at criteria sha256
`fee557514ff2fa9613b3d6ec12a0c524d002e6ba5733abe349320804cbb7beec`, unchanged
and asserted at start-up by every script below. A3 was frozen at whole-file
sha256 `577023f769bcea8adc396e0279a5554667d4490568b0cc5b19ebc1c57903c0c0`
before iteration 1 ran; A4 at
`bed8fb90a2d3a9e3a41bfeaed5545ccc68ec39a24c87b1ea536b9006a6035ab3` before the
first process was launched on the second machine. Receipts:
`runs/gate_step4b_iter/` (50 files), rendered by `scripts/iter_tables.py`.

**Verdict: unchanged. FAIL at d_rep = 8, NON-EVALUABLE at d_rep = 16, step 6
not entered.** No iteration reached parity, so no full re-run was triggered
and none is reported. What the three iterations did produce is an
attribution that is now specific enough to name a capability.

---

#### Iteration 1 - the scorer is not what stopped it. Measured three ways.

**I1.1 - known answer.** `S_add` was given codes that provably contain the
planted factors, at the gate's own sparsity, on the same five seeds
(`runs/gate_step4b_iter/known_answer_sadd.json`):

| code | zeros | `S_add` | `S_add_ridge` |
|---|---:|---:|---:|
| `dense_oracle` (= `L`) | 0.000 | **1.0000** | 1.0000 |
| `row_topk(k=4)` | 0.500 | 0.9475 | 0.8970 |
| **`row_topk(k=2)`** - the gate's own live-code budget | **0.750** | **0.8152** | 0.6618 |
| `row_topk(k=1)` | 0.875 | 0.6564 | 0.4269 |
| `col_mask(p=0.2595)` - the gate's measured live fraction | 0.741 | 0.8581 | 0.7345 |
| `col_mask_shuffled(p=0.2595)` - same zeros, random support | 0.741 | 0.5124 | 0.2583 |

The harness check passes exactly: `S_add(dense_oracle) = 1.0000`. The
predeclared 0.90 exoneration threshold is **not** met - `0.8152 / 1.0000 =
0.815`, so `S_add` does carry a **18.5% relative sparsity penalty** at the
hybrid's own sparsity, and that number is now on the record rather than
suspected. The predeclared "scorer is the defect" threshold is not met either:
0.8152 is far above the additive arm's random floor of 0.4549.

The reading the two thresholds bracket is the decisive one, and it is
arithmetic rather than judgement: **a code that contains the factors, at
exactly 75% exact zeros, scores 0.8152 - above the best specialist (0.6033)
and 1.79x the random floor.** The hybrid scores 0.3916. Sparsity costs 18.5%;
the hybrid is short by 52% of the oracle. `S_add` would have rewarded a sparse
code that had found the factors, so it is not what stopped this one. The
`col_mask_shuffled` row is the control that makes that readable: at the same
zero fraction, support chosen by magnitude scores 0.8581 and support chosen at
random scores 0.5124, so the statistic is measuring *where* the mass is, which
is the thing a factor readout is supposed to get right.

**I1.2 - the same mask on real dense representations.** Every additive-arm
specialist and the floor, refitted at d_rep = 8 and put through the identical
`row_topk` mask:

| method | dense | k=8 | k=4 | **k=2** | k=1 |
|---|---:|---:|---:|---:|---:|
| cNMF | 0.5610 | 0.5610 | 0.4062 | **0.3555** | 0.3138 |
| NMF | 0.5737 | 0.5737 | 0.4000 | **0.3396** | 0.2972 |
| PCA | 0.5949 | 0.5949 | 0.5609 | **0.4824** | 0.3895 |
| PCA (step-3) | 0.5135 | 0.5135 | 0.4437 | **0.4100** | 0.3425 |
| random floor | 0.4549 | 0.4549 | 0.4269 | **0.3628** | 0.2894 |

Mean drop at the gate's sparsity: **0.1495**, which is neither below the
declared 0.05 (so the penalty is real) nor above the 0.2117 additive shortfall
(so it cannot account for the result on its own). This is reported as a
**diagnostic and not as a re-scoring**: masking a fitted cNMF code destroys
information cNMF actually had, which is not the same experiment as fitting
cNMF under a sparsity constraint, and section 6's floor is not moved. It is
also why the sparsity-matched floor appears as a non-gating column below.

**I1.3 - the dense pre-activation, on all 35 cells, every method rescored in
the same process.** 70 encoder fits under section 3's protocol unchanged; the
only addition is a tap on `FactorExpert.forward` guarded by
`if not self.training`, so the training path is untouched.

*Reproduction first, as A3 requires.* Of the 70 fits, the **50 run on the
machine the gate ran on reproduce it exactly: `|Δ recon_mse| = 0.000e+00` and
`|Δ live_code| = 0.000e+00`, bitwise, on every one.** The 20 run on the box
(A4) differ by at most 1.25e-02 and 3.81e-02, inside the 0.05 the amendment
declared. Nothing is read off the new columns that this does not license.

*The bug hypothesis first, as the posture requires.* `preact == gated code` in
**0 of 35** cells; the gated code is 72.3% exact zeros and the pre-activation
is **0.0%**; `min(W) = -0.0` against `min(pre-activation) = -17.45`. The two
columns are two different tensors.

| arm | hybrid (gated) | **hybrid dense pre-activation** | random floor | sparsity-matched floor | specialist best |
|---|---:|---:|---:|---:|---:|
| additive (n=5) | 0.3916 ± 0.0356 | **0.4310 ± 0.0142** | 0.4549 | 0.3476 | 0.6033 (PCA 0.5949, cNMF 0.5610, NMF 0.5737, PCA-s3 0.5135) |
| manifold (n=5) | 0.7273 ± 0.0231 | **0.7273 ± 0.0231** | 0.7422 | 0.5830 | 0.9621 (kPCA 0.9621, Lap 0.9577, LLE 0.9435, Iso 0.9041) |
| utility (n=25) | 0.1040 ± 0.0198 | **0.1129 ± 0.0216** | 0.1287 | 0.0814 | 0.2783 (kPCA 0.2628, PCA 0.2470, NMF 0.2358, PCA-s3 0.1905) |

For cNMF, NMF, PCA, PCA-step3, the floor and raw `X` the dense representation
**is** the representation, so their two columns are the same number by
construction; that is why they appear once. The column that could move is the
hybrid's, and on the manifold arm it does not move at all - by construction,
because that arm's readout is the manifold expert's `chart_coords`, which no
JumpReLU gate touches. On the two arms where the gated code is in the readout
the dense pre-activation helps a little (+0.039 additive, +0.009 utility) and
**neither column clears the random floor on any arm**.

**A3's declared reading therefore applies verbatim: the scorer is exonerated
and the work moves to iteration 2.** The sparsity penalty is real, it is
0.185 relative on an oracle code and 0.150 absolute on real dense codes, and
it is not large enough to be the story. The one thing it does license saying:
the gate's random-projection floor is **dense**, and the sparsity-matched
floor - the same projection at the hybrid's own live-code budget - is 0.3476 /
0.5830 / 0.0814, which the hybrid clears on all three arms. That is reported
as a diagnostic, not as a verdict; section 6's floor did not move.

---

#### Iteration 3 - slot count is the binding constraint at this width, and that is a tension in the gate, not a pass

The diagnostic A2 said was "in every cell record" was `null` in all 70; A3
named the unreachable branch. It is computed here: the 16-slot encoder's
readout, reduced to 8 columns by section 4's rule, scored against **the same
cell's d_rep = 8 specialists and floor**.

| arm | hybrid, 8 slots | **hybrid, 16 slots → 8 cols** | floor | narrow − floor | **wide − floor** | shortfall narrow → wide |
|---|---:|---:|---:|---:|---:|---:|
| additive | 0.3916 | **0.4828 ± 0.0219** | 0.4549 | −0.0633 | **+0.0278** | 0.2117 → **0.1205** |
| manifold | 0.7273 | **0.7996 ± 0.0622** | 0.7422 | −0.0149 | **+0.0574** | 0.2348 → **0.1625** |
| utility | 0.1040 | **0.1625 ± 0.0334** | 0.1287 | −0.0248 | **+0.0338** | 0.1743 → **0.1158** |

**The wide-then-reduced readout clears the random floor on all three arms
where the narrow one clears none**, and cuts the shortfall by 31-43%. A3's
first declared reading is the one that fires: the binding constraint at
d_rep = 8 is the **8-slot bottleneck**, not the objective alone.

A3 also declared what that obliges, and it is reported rather than pared away:
**this is a genuine tension in the gate's design.** d_rep = 16 is the width at
which the encoder is not bottlenecked and the width at which V2 fires for
reasons that have nothing to do with any encoder (A2: +0.0715 / +0.0822 /
+0.0910 against a 0.10 bar, reproducing the encoder-blind characterisation).
d_rep = 8 is the only width at which the instrument discriminates and it is a
width at which the encoder is short of slots. **Neither width can certify this
architecture**, and no re-run at either width would change that. Reporting the
wide-reduced column as though it were the gate's answer would be picking the
width that flatters the encoder, which section 1.2 forbids and which this
paragraph exists to refuse.

Parity is not reached either way: 0.1205 / 0.1625 / 0.1158 against a
`TOLERANCE` of 0.05, i.e. 2.3x to 3.3x the bar.

---

#### Iteration 2 - a 10x budget does not close it, and on two arms it makes it worse

Seven cells, seed 0 of each family, d_rep = 8, section 3's protocol verbatim
except `FIT_STEPS`, one fit to 20,000 steps scored at {2000, 4000, 8000,
14000, 20000}. Specialists and floor fit once per cell in the same process.

*The anchor, which A4.3 made do double duty.* Step 2000 reproduces the gate's
own cells to `max |Δ| = 0.0306` over 21 compared quantities, inside 0.05. The
two-machine move is therefore **demonstrated inert at the gate's own
tolerance** rather than assumed to be. The dedicated cross-environment control
agrees: `additive_factors|s0` and `manifold|s0` fitted on both machines under
an identical protocol differ by at most 0.0299 and 0.0306 across every scored
column.

| arm | 2,000 | 4,000 | 8,000 | 14,000 | 20,000 | specialist best | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| additive (n=1) | 0.4595 | **0.5397** | 0.5305 | 0.5170 | 0.5011 | 0.6173 | 0.4257 |
| manifold (n=1) | **0.6997** | 0.3784 | 0.3445 | 0.5081 | 0.5593 | 0.9650 | 0.8150 |
| utility (n=5) | 0.0919 | 0.0980 | 0.0985 | 0.1079 | **0.1148** | 0.2792 | 0.0892 |
| live code, utility cells (n=5) | 0.309 | 0.400 | 0.427 | 0.434 | 0.460 | | |

Shortfalls: additive 0.1577 → **0.0776** at 4,000 → 0.1162 at 20,000; manifold
0.2653 → 0.4058; utility 0.1874 → 0.1644. **No arm reaches parity at any
checkpoint**, so A3's re-run trigger does not fire and no new gate table is
reported. The best number anywhere is the additive arm's 0.0776 at 4,000
steps - half the gate's own shortfall on that cell, still 1.6x the tolerance,
and it gets worse with more training, so it is not a budget that was missed.

**The load-bearing observation is the one that comes from putting the
reconstruction loss beside the arm statistic.** In all seven cells the
objective improves monotonically at every checkpoint, and the readout's
agreement with the generative structure does not follow it:

| cell | recon MSE 2,000 → 20,000 | arm statistic 2,000 → 20,000 |
|---|---|---|
| `additive_factors` s0 | 0.2342 → **0.1787** (−24%) | `S_add` 0.4595 → 0.5397 at 4,000 → **0.5011** |
| `manifold` s0 | 0.1548 → **0.0943** (−39%) | `S_man` 0.6997 → **0.3445** at 8,000 → 0.5593 |
| `threshold` s0 | 0.2081 → **0.1835** (−12%) | `S_util` 0.1359 → **0.1207** |
| `hill` s0 | 0.2150 → **0.1825** (−15%) | `S_util` 0.1162 → 0.1766 |
| `xor` s0 | 0.2289 → **0.1876** (−18%) | `S_util` 0.0014 → 0.0054 |
| `context_gated` s0 | 0.2125 → **0.1806** (−15%) | `S_util` −0.0007 → 0.0113 |
| `mixture` s0 | 0.2844 → **0.2256** (−21%) | (no gating arm) |

On the manifold cell the loss falls by 39% while the statistic **halves**, and
it never returns to its 2,000-step value. On the additive cell the loss falls
by another 8% between 4,000 and 20,000 while `S_add` falls by 0.039. The
optimisation is not stalling, running out of budget, or collapsing - it is
succeeding at what it was asked to do, and what it was asked to do is not the
thing the gate measures.

*The non-negativity clause A3 attached to this iteration is not triggered, and
the check is reported rather than skipped:* `min(W) = -0.0` over every cell and
every checkpoint. The factor code has **not** gone signed - the JumpReLU
forward `a·1[a > θ]` with `θ = softplus(log θ) > 0` guarantees it - so "add a
non-negativity prior" is a change to the architecture and belongs to step 6,
not here.

---

#### F1 - `max_offsupport_mass` is repaired, and it immediately found something

The statistic could only return 0.0: `max |g|` over `{|g| ≤ 0.0}`. It is
redefined in `sparse.py` as `max {|g| : 0 < |g| ≤ tol}` at `tol = 1e-3` - the
largest magnitude that is nonzero but small enough that calling it zero would
be a lie - with `frac_subtol_nonzero` beside it. `frac_exact_zero`,
`mean_active`, `min_active`, `max_active` and `l0_over_n` still count **exact**
zeros at any `tol` and are numerically unchanged, which is why no prior claim
moves.

Pinned on both sides of the question in
`test_sparse.py::test_offsupport_mass_separates_a_sparse_gate_from_a_dense_one`
(float64, `manual_seed(0)`, 64x16, logits `randn*4`):

| input | `frac_exact_zero` | `frac_subtol_nonzero` | `max_offsupport_mass` |
|---|---:|---:|---:|
| **known-sparse** `entmax15` | 0.878906 | **0.001953** | **6.43096e-4** |
| **known-dense** `softmax` | 0.000000 | **0.556641** | **9.76603e-4** |
| uniformly 1e-9 (old value: 0.0) | 0.0 | **1.000000** | **1e-9** |
| all-ones | 0.0 | 0.000000 | 0.0 |

The two differ by 285x on `frac_subtol_nonzero`, and 0.0 remains reachable and
meaningful (all-ones) rather than being the only value. The uniformly-1e-9 row
is the case the dead statistic reported as an absence.

Six further `== 0.0` assertions across `test_experts.py`, `test_slots.py`,
`test_adapters.py` and `test_encoder.py` were vacuous for the same reason and
are replaced by live ones. Two of them were hiding facts:
`RelationalExpert`'s pair gate carries **36.5% of its coordinates as
nonzero-but-below-1e-3 mass** beside 38.5% exact zeros, and `ModalityRouter`
at temperature 4.0 reaches **no exact zeros at all** - both now asserted as
measured values rather than as a number that could not fail.

**Receipts:** 220 dead fields across 72 `runs/**/*.json` are marked
`"not_measured"` by `scripts/mark_dead_field.py`, textually so the rest of
each file is byte-identical, with the parsed tree asserted equal before and
after and every file's sha256 recorded on both sides in
`runs/gate_step4b_iter/dead_field_remediation.json`. They are marked rather
than regenerated because regenerating them means refitting a run that has
already been reported. No gate verdict, arm score or validity number changes.

#### F2 - V3 per cell

A re-reading of the existing records; nothing refitted, no score changed
(`runs/gate_step4b_iter/v3_per_cell.json`).

| width | arm | as run (arm-level V3) | per-cell V3 | excluded | of which counted as passes |
|---|---|---|---|---:|---:|
| d = 8 | additive | 0/5, shortfall 0.2117 | 0/5, 0.2117 | 0 | 0 |
| d = 8 | manifold | 0/5, shortfall 0.2348 | 0/5, 0.2348 | 0 | 0 |
| d = 8 | **utility** | **9/25, shortfall 0.1736** | **6/22, shortfall 0.1974** | 3 | **3** |
| d = 16 | utility | 10/25, shortfall 0.0769 | 6/21, shortfall 0.0921 | 4 | **4** |

Every excluded cell was a counted pass, and the shortfall is **worse** under
the stricter reading. The verdict does not change at either width, which is
why this is a fix to the instrument's bookkeeping rather than a result.

#### A footnote that is not this gate's, and is on the record because it was found here

`World.fingerprint()` - the handle section 2 binds V1 to - is **not portable**.
`additive_factors|s0|proposal` regenerated under a different interpreter/numpy
build has bit-identical latents and target, `X` differing in 75,289 of 144,000
entries by at most **6.66e-16**, one truth `effect` scalar differing in the
last ulp, and therefore a different fingerprint. A1 fixed the thread-count axis
of this at the source; this is the library-build axis and `additive_factors`
never calls `gp_field`, so it is not that defect resurfacing. It cost nothing
here because A4.2 stopped regenerating the world rather than loosening the
check, and the byte-exact V1 then passed on the box for every cell. It is
recorded (`runs/gate_step4b_iter/world_fingerprint_portability.json`,
`scripts/world_env_reproducibility.py`) because the next agent to move any
frozen-manifest work to another machine will hit it.

---

#### Attribution, updated: what the three iterations removed, and what is left

A2 excluded optimization, benchmark, power and hypothesis-language. These
three iterations remove three more candidates by measurement:

* **not the scorer.** A code containing the factors at the hybrid's exact
  sparsity scores 0.8152, above every specialist. The measured sparsity
  penalty is 0.185 relative and the hybrid is short by 0.52 of the oracle.
* **not the budget.** At 10x the budget no arm reaches parity; the additive
  arm peaks at 2x and declines, and the manifold arm loses half its score
  while the loss falls 39%.
* **not only the width, though width is real.** 16 slots reduced to 8 columns
  clears the floor on all three arms and still misses parity by 2.3-3.3x the
  tolerance. And the width that would remove the bottleneck is the width at
  which this gate cannot certify anything about any encoder.

#### The missing capability, in one sentence

**The encoder lacks an objective that makes its slots correspond to generative
factors: under reconstruction alone the loss falls monotonically at every
checkpoint while the readout's agreement with the planted latents peaks early
and then decays - `S_man` halves as the loss falls 39% - so what step 6 has to
supply is a training signal that selects for an *identifiable* code
(replication-grounded or discovery-grounded), not more slots, more steps, or a
different readout.**

That sentence is the only thing here that licenses adding architecture, and it
is deliberately a statement about the **objective** rather than about a module:
nothing measured in these three iterations is evidence that a slot, an expert,
a router or a loss term is missing from the *forward* model. **"The
reconstruction-fitted hybrid does not exceed a random projection of the same
width" stands, unsoftened, as a statement about this architecture under this
objective** - and it is the control against which replication-grounded
training has to be read. If step 6's training clears this floor where
reconstruction does not, this table is what makes that a result rather than a
claim.
