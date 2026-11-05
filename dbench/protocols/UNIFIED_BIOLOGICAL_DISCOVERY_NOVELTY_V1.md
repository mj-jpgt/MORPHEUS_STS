# MORPHEUS Unified Biological Discovery and Novelty Criteria — v1.0

This is the normative project contract for reporting a MORPHEUS biological
object.  It implements the governing principle supplied by the project owner:
a discovery is a reproducible molecular state, tissue state, spatial
configuration, or dependency whose *specific properties* are not adequately
explained by established biology or relevant technical artifacts.

Novelty is evaluated for the complete biological object—not the novelty of an
individual gene or pathway.  Discovery type, validity, novelty and utility
are independent fields and must never be collapsed into a single score.

## Controlled vocabularies

Discovery types are `D1_molecular_state`, `D2_morphology_first`,
`D3_morpho_molecular_correspondence`, `D4_spatial_multiscale_dependency`,
`D5_context_dependent_modulation`, `D6_nonadditive_interaction`,
`D7_irreducible_higher_order_interaction`, and
`D8_conserved_biological_organization`.

Primary novelty categories are `N0_known_biology_recovered`,
`N1_context_extension`, `N2_spatial_extension`,
`N3_compositional_novelty`, `N4_new_molecular_state_or_association`, and
`N5_new_functional_mechanism`.

Evidence tiers are `Candidate`, `Internally supported`, `Externally
confirmed`, and `Functionally supported`.  Utility labels are `Biological`,
`Assay-translatable`, `Clinically associated`, `Clinically predictive`, and
`Clinically actionable`.

## Required claim-specific tests

For a frozen multiscale relation `R`, the decisive estimand is

`G_MS = Perf(Y | B, R) - Perf(Y | B, R_reduced)`

on held-out patients, with comparable model capacity. `B` contains nuisance,
composition and established-biology controls. `R_reduced` removes precisely
the claimed scale, spatial arrangement or context—not an arbitrary subset.
Cross-scale claims remove the claimed scale; organization claims use
abundance/composition-preserving controls; neighbourhood claims compare to
matched own-location/nonspatial explanations; gradients use compartment- and
autocorrelation-aware alternatives; and modulation compares against additive
context and predictor main effects. Spatial variation must not be described
as temporal dynamics without longitudinal or perturbational evidence.

`D4` requires a held-out full-versus-claim-matched-reduction comparison,
spatially calibrated null and replication. `D5` requires effect modification
beyond additive main effects. `D6` requires a matched additive decomposition.
`D7` requires all eligible lower-order decompositions, corrected inference,
and replication. `D8` requires frozen no-refit transfer, preserved direction
and biological interpretation, plus matched nulls. A multiscale candidate is
not a validated multiscale dependency merely because it contains multiple
radii or variables.

The universal fields are frozen definition, patient-disjoint evaluation,
statistical validity, confounder robustness, reproducibility and detection
power. New confirmation analyses require predeclared `q <= 0.05`, a
patient-level confidence interval excluding zero, and the claim-specific
tests above. Historical registrations preserve their original thresholds.

`known_biology_incremental_gain` is required only for a claim of incremental
predictive information. A failed decomposition disqualifies D7 only; it does
not erase a valid molecular, morphology-first, spatial, or context claim.

An unperformed assessment is represented as `not_assessed`; it is never
silently promoted to pass or interpreted as failure. `N4` remains
`provisional` until targeted literature adjudication, synonym matching and
known-axis alternatives have been reviewed. `N5` requires causal/functional
evidence, not an observational drug or outcome association.

## Reporting record

Every highlighted object must carry: its discovery type(s); a one-sentence
specific novelty claim; primary novelty category and adjudication state;
evidence tier; decisive statistic and its patient-level interval; known-biology
alternative; assessment status for all relevant controls; external contexts
and conservation count where applicable; and utility labels. A clinical
association is not clinical prediction, and neither is clinical actionability.

The classical paper is mapped retrospectively from frozen S08/G03/T03/G18 and
related records: no rerun is implied, historical thresholds are preserved,
and missing tests remain `not_assessed`. The deep-learning paper applies the
same record format to frozen registries and blind discoveries. Existing
predictive-gain/decomposition results are retained, but are interpreted only
for the claims that they actually test.
