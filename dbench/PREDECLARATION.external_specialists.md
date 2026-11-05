# PREDECLARATION — external named specialists in DiscoveryBench

Written before `dbench/specialists_external.py` existed. The sha256 of this
file is recorded in the module docstring and in every report the module emits;
a report whose `predeclaration_sha256` does not match this file's hash was
produced against a different set of commitments and must not be read as
confirming them.

Scope: wire the owner's **named** specialist classes onto the `World`
interface so Headline 2's bank contains the methods the paper claims to
compare against. This predeclaration covers the two classes being built now
(interaction detectors; cNMF/NMF), the SINDy-AE replacement for
`specialist_ode`, and the two relabels. It does **not** cover LaLiGAN or
CRL, which are not built here.

## 0. Bound inputs (sha256 + mtime; `.git` is empty, so hashes are the record)

| file | sha256 | mtime (local) |
|---|---|---|
| `dbench/oracle.py` | `bda937c7de7c70c82748528004f61a2735f33dffbb7ad14178adb5b1ec498ec9` | 2026-09-07 04:14:31 |
| `dbench/world.py` | `f0974d77a4424dbb97459abde91d9f679a12891e32463649b7f95a0bd4371d28` | 2026-09-06 21:15:43 |
| `dbench/metrics.py` | `7be146f39403e3a12add3316737c5d4e69290528e4f328628f55ada710c31365` | 2026-09-07 03:47:53 |
| `dbench/families.py` | `777fa807f8b0b4904ccc077f1d0a91414bfb75af982b2c2163725a4b98bd9b45` | 2026-09-07 03:34:53 |
| `dbench/opvocab.py` | `858f37364cf10d78db90396d267df7c108eafeff78c1c38c38b249061a256bdc` | 2026-09-07 03:46:13 |
| `cscd/interaction_baselines.py` | `a860590861465f9671f34791f2b0d35f65ae5266f1c96cef360c61821c69848f` | 2026-09-06 22:02:20 |
| `cscd/matching.py` | `cbdce7e0eab8b62db4e6e67ac5ce4f553acbeb726c99af51577b8fe461c34203` | 2026-09-06 22:06:02 |
| `cscd/baselines.py` | `302d6d9552d06223540f154f7523a6132820677dbf67ef940ff6281cddff343c` | 2026-09-02 00:13:07 |
| `cscd/missing_baselines.py` | `bbe989e39ee5edeebf990ac3ae363deff6b17a164769107541133ffa853ea1bb` | 2026-09-02 21:07:22 |

`dbench/MANIFEST.frozen.json` (`3ff2d86f…`, frozen 2026-09-07T08:08:50Z)
records `oracle.py` at `47e087cfe23d3e41bc9849ace0b28234b8cd60106843c64f0fa0c4e6c5ae5eb1`,
which is **not** the on-disk hash above: the dbench owner has edited
`oracle.py` since that freeze. That manifest is therefore already superseded
before this work starts, and this predeclaration binds the on-disk state, not
the manifest. `specialists_external.py` is a bound artefact that must enter the
dbench owner's *next* freeze; nothing here re-freezes anything.

## 1. What each named class is handed, and why that is the honest mapping

### 1.1 Interaction detectors (Diamond, NID, hierNet, glinternet, HSIC, kernel CCA)

`cscd.interaction_baselines.DETECTORS` expects
`(ZA_fit, ZB_fit, ZA_eval, ZB_eval, hyps, alpha, seed, min_active)` — paired
codes plus a frozen hypothesis list over `(a_tuple, b)`.

**Mapping.**

* `ZA` = the world's **candidate-variable matrix**: the columns
  `oracle._scan_names(world)` selects — 1-D, non-degenerate, **operator-derived
  latents excluded** — each z-scored.
* `ZB` = the response as a one-column modality: `zscore(world.target)` reshaped
  `(n, 1)`. Every hypothesis therefore has `b = 0`.
* `hyps` = `cscd.matching.build_hypotheses(dA, 1, max_order=2)`.
* Each detector then answers "does the pair `(x_i, x_j)` carry non-additive
  signal for `Y`", which is Diamond's own framing, and order-1 rows answer
  "does `x_i` carry additive signal for `Y`".

**Why `ZA` is the candidate-latent matrix and not the raw modality `X`.**
`oracle.py`'s entire bank is defined as "a specialist **handed the true
latents**" (its module docstring). Handing the external detectors the 48-column
observed modality while the incumbent bank gets the latents would make `max_j`
a comparison between a latent-fed specialist and an `X`-fed one, which is not a
head-to-head. The two are also different in kind: `X` is a dense orthonormal
mixing of *every* latent plus a `0.8`-weighted nuisance mixture plus
measurement noise, so no column of `X` tracks any single latent, no claim built
from an `X` column can carry a base-named operand, and the recovery would be
capped near zero by the attribution rule before any engine ran — exactly the
failure mode the audit already recorded for PCA's 0.222 column. Predeclared
consequence: the `X`-fed variant is reported as a **secondary** row with its
structural cap printed, never as the class's headline number.

**This is not the oracle failure mode the relabels in §4 fix.** Candidate
latents are the *search space* — the specialist still has to find which pair,
and distractors are in the scan so precision can fall. `specialist_ode` reading
`latents["x"]` and `specialist_symmetry` reading `latents["r"]`/`["theta"]` are
the *answer* — the state coordinates and the invariant, with nothing left to
search. The line is: candidate set = fair; solved coordinate = oracle.

**Multiplicity.** All six go through `interaction_baselines._apply_bh` with
`nan_policy="declared"` — the shared `m` fix. `m` = `len(hyps)`, identical for
every detector on a given world. Diamond additionally reports its **native**
knockoff / knockoff+ selection, which is its distinguishing FDR mechanism and
is not routed through BH.

**Claim vocabulary — no third spelling.** A flagged row becomes a
`dbench.metrics.Claim` in **dbench's** vocabulary with base-named operands:

* order-1 `(i,)` → `operator="ADD"`, `operands=(names[i],)`
* order-2 `(i,j)` → `operator="AND"`, `operands=(names[i], names[j])`
* `target=world.target_name`, `lag=None`, `context={}`

The module asserts `opvocab.to_canonical(op) == op` and `op in world.OPERATORS`
for every operator it emits, so a drift into a compiler spelling (`MUL`) or a
new spelling fails loudly rather than silently forfeiting
`object_similarity`'s 0.20 operator weight.

**Fit/eval.** When a replication world is supplied, fit on the proposal world
and evaluate on the replication world. When only one world is supplied, a
deterministic structure-respecting 50/50 row split is used (contiguous blocks
for temporal worlds, a spatial half-plane for spatial worlds, a seeded
permutation otherwise) and `split_mode` is recorded in every report. A report
that does not state `split_mode` is not readable.

**Disclosed infelicity, not argued away.** `min_active` counts
`(conjunction_feature(...) > 0).sum()`, written for non-negative sparse codes.
On signed z-scored latents a product is positive for about half the rows, so
`n_active ≈ n/2` and the guard never binds at these `n`. It is left as-is
rather than reinterpreted, because changing it would change what every
`cscd`-side number means.

### 1.2 cNMF / NMF

`XA = X, XB = Y` is **not** right for a factorization: cNMF learns a dictionary
over one modality, and `Y` is not a modality.

**Mapping.** Fit `MISSING_BASELINES["cnmf"]` (resp. `BASELINES["nmf"]`) on the
world's observed modality `X` for **both** of its two modality arguments and
keep only `ZA`. This uses the registered public entry point rather than a
private one-modality helper; the discarded `ZB` costs one redundant fit and is
recorded as such. `fit_idx` is the fit half of the same splitter as §1.1, so
the dictionary is never fit on the evaluation rows.

**k.** Primary `k` = the world's **declared latent count**, i.e.
`len(oracle._scan_names(world))`. Secondary grid = the step-3 matrix's own
`_bl_worlds.CNMF_WIDTHS` restricted to what this substrate can support:
`[16, 32, 48]` (48 = `d_obs`; the grid's 64 and 128 both saturate at
`min(n_atoms, n_fit, d) = 48` and are reported as saturating rather than run
twice).

**Scoring.** The step-3 matrix's own rule: `cscd.matching.align_atoms_to_factors`
with `min_abs_r = 0.30`, atoms scored on the **eval** half against the planted
additive factors `F` — the operands of the world's true `ADD` objects, as
oracle latent columns. Recovery = number of **distinct** true factors some atom
maps to, over the number of planted factors.

**Structural caps printed alongside**, because the audit showed this rule caps
PCA's column at 0.222 before any engine runs:

* `cap_identity_X`: the same alignment rule applied to the raw columns of `X`.
  This is the informative cap and it can return anything in [0, 1].
* `cap_oracle_latents`: the rule applied to the oracle latent matrix itself.
  **Predeclared to be 1.000 by construction** (each planted factor is its own
  column, `|r| = 1`). It is a check that the rule is wired up, not a result;
  if it is not 1.000 the adapter is wrong. This 1.000 is declared here in
  advance precisely so it is not reported later as if it could have come out
  otherwise.

cNMF makes no significance statement, so its "flag set" is the set of atoms
that map to some factor. `flag_rate = n_atoms_mapped / n_atoms`.

### 1.3 SINDy-AE (Task 2)

`specialist_ode` reads `world.latents["x"]` and `["y"]` — the true state. The
replacement `specialist_sindy_ae` reads only `world.modalities["X"]`.

Construction: Champion, Lusch, Kutz & Brunton, *Data-driven discovery of
coordinates and governing equations*, PNAS 2019, **116**(45):22445–22451,
DOI `10.1073/pnas.1906995116` — **confirmed to resolve** via Crossref
(`api.crossref.org/works/10.1073/pnas.1906995116`) on 2026-09-07; the PNAS
landing page itself returns 403 to an unauthenticated fetch, which is a
paywall, not a bad DOI.

Loss, as in the paper:
`||x − ψ(φ(x))||² + λ₁||ẋ − (∂ψ/∂z)ΘΞ||² + λ₂||∇φ(x)·ẋ − ΘΞ||² + λ₃||Ξ||₁`,
with sequential thresholding of `Ξ`. `Θ` is the degree-≤3 polynomial library in
`z` — the same 10 terms `oracle._poly_library` uses. `ẋ` is a finite difference
of the **observed** `X` at `world.meta["dt"]`. The latent dimension is supplied
as a hyperparameter (`= 2`), exactly as Champion et al. supply it; that is a
declared hyperparameter, not a peek at the answer. A final `pysindy` `STLSQ`
refit on the frozen encoder's latent trajectory is what produces the reported
`Ξ`, so `pysindy` 2.1.0 is genuinely the sparse-regression engine.

**Scoring.** The AE's coordinates are identified only up to permutation, sign
and scale, so the scorer — and only the scorer — aligns `z` to
`(latents["x"], latents["y"])` by correlation before comparing patterns. That
is the same discipline `align_atoms_to_factors` uses: truth is allowed in the
scorer and forbidden in the method. Reported: `pattern_recall` (fraction of the
4 planted terms `dx/x³, dx/y³, dy/x³, dy/y³` present in the aligned `Ξ`),
`pattern_precision`, and coefficient error after rescaling `z` to the aligned
state's standard deviation. Pattern is primary; coefficients are secondary
because diagonal rescaling preserves the pattern and not the coefficients.

**Predeclared non-fallback.** If the autoencoder cannot identify the
coordinates at this world's SNR, that is reported as a finding about the
family's difficulty for a real SINDy-AE and enters the table as such. It is not
permitted to fall back to reading `latents`.

## 2. Known-answer tests (values pinned here, before any run)

A named specialist that fails its own home regime through this adapter is an
**adapter defect** until the adapter has been shown faithful on a
known-answer. These are those known-answers.

**KA-1 — Diamond on `pure_and`, seed 11, snr=high.** The world plants two pure
conjunctions `A0*B0` and `A1*B1` with zero marginal signal, plus one additive
decoy pair `(U0, V0)` and 4 distractors. `dA = 10`, `hyps = 10 + 45 = 55`.
Required: at Diamond's **native knockoff+** selection, both
`AND:(A0,B0)` and `AND:(A1,B1)` are selected — conjunctive recall `2/2 = 1.000`.
`flag_rate` and FCR reported beside it. Failure ⇒ adapter defect (wrong `ZB`
orientation, wrong standardisation, or the fit/eval split leaking), to be
investigated before any Headline-2 number is produced.

**KA-1b — the marginal check.** On the same world `linear_cca` (order-1 rows)
must **not** recover `A0` or `B0` as `ADD` objects: they are exactly marginally
uncorrelated with `Y` by construction. A `linear_cca` that flags them is
evidence of a leak in the adapter, not of a strong baseline.

**KA-2 — cNMF on `additive_factors`, seed 11, snr=high.** 3 planted factors
`A0, A1, A2`, 5 distractors, `d_obs = 48`. Required: factor recall
`≥ 3/3 = 1.000` at the primary `k`, i.e. **at least** what the specialist
bank's `scan_main_effects` achieves (3/3, per `gate_report.json`'s
`additive_factors` row). Home regime; a miss is an adapter defect.
`cap_oracle_latents` must read `1.000`; `cap_identity_X` is printed and is
whatever it is.

**KA-3 — null calibration, world as the replicate unit.** For each of the six
detectors, on `R = 20` independently seeded `iid_permute` nulls of a **tabular**
world (`pure_and`; `additive_factors` as a second substrate), the per-world
false-certification rate — the fraction of null **worlds** on which the
detector raises at least one flag — must sit inside the exact binomial band for
`Bin(20, 0.05)`. The acceptance rule, fixed here: `P(Bin(20, 0.05) ≥ k) > 0.05`,
i.e. **`k ≤ 3` null worlds with a flag passes; `k ≥ 4` fails**
(`P(≥3) = 0.0755`, `P(≥4) = 0.0159`). Never a per-hypothesis rate: a
hypothesis-unit rate would count 55 correlated tests within one world as 55
replicates and would make any detector look calibrated.

**KA-4 — could-it-return-otherwise.** Every rate of exactly 0.000 or 1.000 in
any report carries a `could_return_otherwise` field naming the specific
alternative outcome the statistic admitted. `cap_oracle_latents = 1.000` is
declared in §1.2 as the one number that could **not** have come out otherwise.

## 3. Expected home-regime recovery per class (predeclared, not tuned)

| class | home regime | predeclared expectation |
|---|---|---|
| Diamond (knockoff+) | `pure_and` | recovers both planted conjunctions |
| NID | `pure_and` | recovers ≥1 planted conjunction; NID's search is a top-15% pair screen, so a miss at `dA=10` (6 pairs kept of 45) is a *budget* result, reported with the screen size |
| hierNet | `pure_and` | weak: strong hierarchy forbids an interaction whose constituents have no main effect, which is exactly `pure_and`. A near-zero recall here is the **method's own published constraint**, not an adapter defect, and is the single most informative cell in the table |
| glinternet | `pure_and` | weak hierarchy relaxes the same constraint; expected better than hierNet, worse than Diamond |
| HSIC | `pure_and` | expected to recover: the product feature is exactly a dependence the kernel statistic sees |
| kernel CCA | `pure_and` | expected to recover; `linear_cca` expected to **miss** (KA-1b) |
| cNMF / NMF | `additive_factors` | 3/3 |
| SINDy-AE | `sparse_ode` | pattern recall reported from `X`; no predeclared floor, because no prior run of a real SINDy-AE on this generator exists. What *is* predeclared: the method never reads `latents` |

hierNet's and `linear_cca`'s predeclared weakness on `pure_and` is the point.
A named specialist that passes its home regime and fails elsewhere is what
Headline 2 wants to be able to show; a named specialist whose *published
constraint* excludes the regime is a different and equally reportable fact, and
it is written down here in advance so it cannot be presented afterwards as a
discovery about the baseline's quality.

## 4. Relabels (Task 3)

* `oracle.specialist_symmetry` → `oracle.specialist_symmetry_oracle`
* `oracle.specialist_ode` → `oracle.specialist_ode_oracle`

Both keep the old names as deprecated aliases so no existing caller breaks,
both gain `is_oracle: True` and an `oracle_reason` in their returned dict, and
both are excluded from `max_j` by default. A `max_j_including_oracles` column
is the only place they may appear. Rationale, fixed here: a specialist handed
the answer must not set the bar MORPHEUS is compared against.

`NONRELATIONAL` gains an `ORACLE_SPECIALISTS` set naming exactly these two, so
a consumer can compute both columns without hard-coding names.

`HEADLINE2_maxj_table_v3.json`'s **provenance is amended** to state the
exclusion. Its **numbers are not recomputed** — the H2 owner re-runs under a
fresh predeclaration.

## 5. What would mean an adapter is wrong

1. Diamond misses a planted conjunction on `pure_and` at its native knockoff+
   budget (KA-1).
2. cNMF recovers fewer than 3 of 3 planted factors on `additive_factors` at the
   primary `k` (KA-2).
3. `cap_oracle_latents ≠ 1.000` on any world (the alignment rule is miswired).
4. A detector's per-world null FCR lands outside the `Bin(20, 0.05)` band in
   the **anti-conservative** direction (KA-3) — that is a leak, most likely the
   fit/eval split.
5. `linear_cca` flags `A0` or `B0` as an `ADD` object on `pure_and` (KA-1b) —
   a marginally-null constituent cannot be found by a marginal method.
6. Any emitted operator fails `to_canonical(op) == op` or is absent from
   `world.OPERATORS` — a third spelling has been introduced.
7. `m` (`len(hyps)`) differs between two detectors on the same world — the
   shared-`m` fix has been lost.
8. SINDy-AE's reported recovery is unchanged when `world.modalities["X"]` is
   replaced by noise — it would mean the method is not reading `X`.

## 6. Not built here

* **LaLiGAN** (Rose-STL-Lab/LaLiGAN, ICML 2024) — not built. `specialist_symmetry`
  remains, relabelled as an oracle.
* **CRL (CITRIS / BISCUIT)** — not built. `scan_intervention_edges` remains and
  is *not* relabelled: it searches over already-named latents but is not handed
  the edge, so it is a fair-but-narrow specialist, not an oracle.

Cost estimates for the owner are in the final report, not here.

## 7. Compute discipline

Anything that fits a model runs on `morpheus2`: OMP/MKL/OPENBLAS/NUMEXPR = 4,
at most 4 concurrent jobs, `setsid nohup`, bounded polls, kills matched on the
full command line only. `~/morpheus_h1_work` and `~/morpheus_h1d_work` are
other agents' work and are read-only. `~/cscd` on the box is the synced copy
that carries the required `nan_policy=` argument. Local RAM is ~0.5 GB, so
local runs are limited to import checks and scoring.

---

# APPENDIX A — effect sweeps for the six `CeilingAuditRequired` records

Appended 2026-09-07, before any sweep was run, in response to the reporting
standard's adapter refusing to serialise six records that read `recall = 1.000`
with no effect sweep anywhere in the artefact.

**The file above this line is unchanged.** Its first 18109 bytes still hash to
`cb090ef91daf434d48bb93937641768150f99aad0abb8d021027ae9fd33de02d`, which is
the hash every record produced before this appendix carries. Records produced
by the sweep carry the hash of the whole file including this appendix. Both
hashes are in `dbench/EXTERNAL_SPECIALISTS_RESULTS.json`, so which commitment
a given number was produced under is readable off the number.

## A.0 Why

Five interaction detectors (`nid`, `hiernet_own`, `glinternet_own`, `hsic`,
`kernel_cca`) on `pure_and` and `NMF` on `additive_factors` each report
`recall = 1.000`, none is a declared control, and the artefact contains no
sweep for any of them — only Diamond's *alpha* sweep, which moves the decision
threshold and not the signal. The step-3 leakage audit
(`morpheus_encoder/step3/AUDIT_FINDINGS.md`) is exactly about this: on
CERTIFY-D the published grid sat at `r = 0.989` while every engine's first drop
sat at `r = 0.070–0.556`, i.e. **1.7x to 13.5x below the operating point**, so
the 1.00s carried no information about the methods. A 1.000 without the
strength at which it first drops is that number again.

## A.1 The strength axis, per arm, and why it is that one

### `pure_and` (the five detectors, plus `diamond_knockoff` and `linear_cca`)

The step-3 repair added a conjunction-amplitude axis to `cscd/certify_d.py`
(`conj_beta`, with `r_for_conj_beta` / `conj_beta_for_r`) because `noise` alone
moved `r(product, target)` only 0.989 -> 0.944 there. **That machinery cannot
be reused directly here**: it plants `FB[:,b] = beta * FA[a1] * FA[a2] +
noise * e` inside CERTIFY-D's own generator, and a dbench world is not built by
that generator. The dbench analogue of `conj_beta` would be
`make_pure_and`'s `beta = 1.4 * SNR_PRESETS[snr]`, and `snr` is **not usable as
the axis**: it takes three discrete values (1.0 / 0.5 / 0.25), which is four
grid points at best and cannot reach the `r = 0.07–0.20` region where the
audit found every first drop.

So the axis is **`make_pure_and`'s `noise` keyword**, swept upward, which sweeps
the planted conjunction's strength downward continuously. `noise` is a declared
family parameter (`make_pure_and(..., noise=1.0)`), reaches the whole region of
interest, and — unlike `snr` — is continuous.

The **strength coordinate reported is not `noise`**. It is
`r = corr(A0*B0, Y)`, the world's own declared `Hypothesis.effect` for the
planted `AND` object, which is the *same quantity* the step-3 ladder is indexed
by (`CONJ_R_LEVELS` names worlds by target correlation, not by raw amplitude,
"because beta's meaning moves with noise, r's does not"). That makes the two
substrates directly comparable. `noise` values are solved analytically from
`r = 1.4 / sqrt(4.92 + noise^2)` and the *measured* effect is what gets
recorded.

**Two disclosed differences from the CERTIFY-D axis**, stated before the run so
neither can be offered afterwards as an explanation of a mismatch:

1. `noise` weakens the **whole world**, not the conjunction alone: both planted
   conjunctions and both additive decoy main effects (`ADD:U0`, `ADD:V0`) lose
   strength together. CERTIFY-D's `conj_beta` moved the conjunction only. The
   record's headline recall is over all four objects, so the sweep also reports
   the **conjunctive-only** recall (2 `AND` objects) separately — that is the
   quantity comparable to the audit's table, and if the two disagree the
   difference is attributable.
2. `r` on this world is bounded above by **0.631** (at `noise = 0`:
   `1.4 / sqrt(4.92)`), because the other planted terms are in `Var(Y)`. The
   audit's top two rungs, `r = 0.989` and `0.894`, are unreachable here. The
   sweep therefore starts at the world's ceiling and the comparison is over the
   overlapping range.

**Grid** (nominal `r`; the audit's own rungs, marked `*`, plus the reachable
ceiling and two rungs below the audit's floor):

    0.631 (ceiling, noise=0), 0.569 (noise=1, the published operating point),
    0.556*, 0.400, 0.300, 0.199*, 0.133*, 0.100*, 0.070*, 0.050, 0.035

**Replication:** 5 independently seeded worlds per grid point (seeds 11–15).
The **world is the replicate unit**. Recall at a grid point is the mean over
those 5 worlds.

### `additive_factors` (NMF, and cNMF beside it)

A factorizer never sees the response, so no response-side axis applies. The
coordinator's analogue — **the additive factor's loading strength** — is the
right one, and on this generator it is set by the observation operator:
`_observe` builds `X` as an orthonormal mixing of the 8 latents plus a fixed
`obs_noise = 0.35`, and `make_additive_factors` does not expose `obs_noise`.
The one exposed lever that moves loading strength is **`d_obs`**: with
orthonormal loadings the per-column signal s.d. is `~sqrt(k / d_obs)` against a
fixed noise s.d., so widening the modality weakens every factor's footprint in
every column.

The **strength coordinate reported** is the measured loading strength itself:
`mean over the 3 planted factors of max_c |corr(X[:,c], A_i)|` — the best
single-column correlation, which is the exact quantity
`align_atoms_to_factors` thresholds at 0.30.

**Grid** (`d_obs`): 48 (published), 96, 192, 384, 576, 768, 1152, 1536.
Measured axis check, run before this appendix was written and covering the axis
only, no engine: loading strength 0.607, 0.628, 0.464, 0.386, —, 0.314, —,
0.252, and the number of `X` columns clearing 0.30 falls 33, 47, 46, 19, —, 8,
—, 0.

**Replication:** 3 seeds per grid point (11–13); cNMF at `d_obs = 1536` with
`n_runs = 8` is the expensive cell and 3 seeds is what fits the box budget. The
world is again the replicate unit. `n_runs` is **not** reduced to buy speed:
that would change the method, which is the step-3 rule.

## A.2 What is recorded

For every grid point, per engine:

* `strength` — the measured coordinate above (r, or loading strength)
* `estimate` — the engine's **headline recall**, i.e. the same bank-scope recall
  that read 1.000 and triggered the audit. `effect_sweep` is emitted as
  `list[[strength, estimate]]` so the adapter reads it without a schema change.
* beside every point, in `effect_sweep_detail`: `flag_rate`, `fcr`,
  `recall_in_grammar`, conjunctive-only recall and marginal-only recall
  (`pure_and`), `n_flagged`, `n_candidates`, and per-seed values.
  For the factorizers, `cap_identity_X` beside every point as well.
* `first_drop_strength` — **the operative number**: the largest grid strength at
  which mean recall is `< 1.000`. `last_strength_at_1.000` is recorded beside
  it. Monotonicity is not assumed; the full curve is reported and a
  non-monotone curve is reported as non-monotone rather than smoothed.

`flag_rate` and `fcr` sit beside every point because recall that stays at 1.000
while the flag rate climbs is **saturation, not robustness** — `hsic` already
reads 1.000 at a flag rate of 0.673 and FCR 0.892 at the operating point, and
`hsic`'s matched-budget recall there is 0.000.

## A.3 Prediction, made before the sweep ran

The step-3 audit's first-drop points on CERTIFY-D (oracle features, n = 12000,
3 seeds):

| engine | first drop |
|---|---|
| diamond_knockoff | r = 0.070 |
| glinternet_own | r = 0.100 |
| kernel_cca | r = 0.133 |
| hsic | r = 0.199 |
| nid | r = 0.556 |
| hiernet_own | r = 0.556 |
| linear_cca | never reaches 1.000 |

**Primary prediction.** The dbench `pure_and` sweep reproduces that **rank
order**: `diamond <= glinternet <= kernel_cca <= hsic <= {nid, hiernet}`, with
`linear_cca` never at 1.000. Absolute first-drop `r` values need not coincide —
the substrates differ — but if the ordering holds, that is cross-substrate
replication of the audit's ranking, which is the claim worth having.

**Named alternative A, declared now so it cannot be an after-the-fact story.**
`hiernet_own` tracks `glinternet_own` rather than `nid` on this substrate, i.e.
it drops far later than `r = 0.556`. Reason to expect it: at the dbench
operating point the two are numerically **identical** (6 flags, recall 1.000,
FCR 0.333), which they are not on CERTIFY-D, because `hiernet_own`'s
strong-hierarchy clamp is a soft projected-gradient constraint and `pure_and`'s
constituents have exactly zero main effect. If A holds, the ordering is
`diamond <= {glinternet, hiernet} <= kernel_cca <= hsic <= nid`.

**Named alternative B.** Every dbench first drop sits at a **higher** `r` than
its CERTIFY-D counterpart, because the `noise` axis weakens all four planted
objects together while `conj_beta` weakened one, so a 4-object denominator
degrades sooner than a conjunction-only one. If B holds, the conjunctive-only
column — which is reported — should sit closer to the CERTIFY-D numbers than
the headline column does. That is the check that separates B from a genuine
substrate difference.

**Factorizers.** NMF and cNMF hold `recall = 1.000` until the loading strength
approaches the credit rule's own 0.30 threshold, and the first drop falls
between loading strength ~0.31 and ~0.25 (`d_obs` 768–1536). The prediction is
specifically that **the drop is the CREDIT RULE binding, not the factorizer
failing**: `cap_identity_X` should fall through 1.000 at the same grid point.
If NMF drops while `cap_identity_X` is still 1.000, that is the factorizer
failing and the prediction is wrong.

## A.4 What would mean the sweep is wrong

1. Recall does not leave 1.000 anywhere on the `pure_and` grid, down to
   `r = 0.035`. The audit found every product-aware engine dropping by
   `r = 0.070`; an engine still at 1.000 at half that would mean the strength
   axis is not moving what it claims to move, and the measured
   `Hypothesis.effect` column is the check.
2. The measured `r` at a grid point does not match the nominal `r` the `noise`
   value was solved for, to within sampling error — the axis is mis-specified.
3. `linear_cca` reaches 1.000 anywhere on the grid. It cannot: `pure_and`'s
   constituents are exactly marginally uncorrelated with `Y` (Prop. 2), which
   is the same reason it reads 0.111 on CERTIFY-D and 0.500 here.
4. A factorizer's recall drops while `cap_identity_X` is still 1.000 — then the
   drop is the engine, not the rule, and A.3's factorizer prediction is wrong.
5. `flag_rate` falls to 0 while recall stays at 1.000 — arithmetically
   impossible, and would mean the flag set and the matched set are being read
   from different runs.
