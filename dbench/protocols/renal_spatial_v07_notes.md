# Kidney protocol v0.7 — comparison model and explanation decomposition (DRAFT, Agent F, 2026-10-15; not a registration)

Applies to `renal_spatial_v07.json` and, unchanged, to `ependymoma_spatial_v01.json`.

**Why the comparison model is nuisance-only.** Up to v0.6 every hypothesis had to beat a wide biological comparison
model (CellViT composition, 91 programme scores, their neighbour means; 378 columns on kidney fold 0). That model answers
a different question from the one v0.7 asks. It makes "discovery" mean "what is left after everything we already measure
has explained the target", so a real relation that composition or a known programme also carries is scored as nothing,
and whether a hypothesis passes depends on how wide the comparison model happened to be. In v0.7 the comparison model holds
only what no biologist would call an explanation: log1p spot depth, the patient (removed by within-patient
standardisation of the response and of every column), platform when sections differ and study one-hots when an episode pools studies. The hypothesis is then judged on
held-out patients by relation `v07_ho` against that model, with the v0.5 pass rule (held-out full-model R² > 0, gain
patient-bootstrap CI above 0, within-patient permutation p ≤ 0.05 in a fold; ≥ 3 of 5 passing folds, mean passing gain
≥ 0.01, structured nulls, BH q ≤ 0.10 within family). Whether a relation needs the joint object is a separate,
reported quantity, G_HO, and the discoveries with a G_HO interval above 0 are listed as higher-order discoveries beside the
general ones; no discovery is dropped for being linear.

**How the explanation decomposition is reported.** For every frozen hypothesis the v0.6 levels are computed on the same
folds after the sweep, with the same arithmetic as `dbench.decomposition.ladder_levels`: N (nuisance), B1 (+ CellViT
class fractions), B2 (+ the 91 own-spot programme scores, minus the target and its leak-guarded columns) and B3 (+ their
6- and 18-neighbour means). At each level the held-out gain Δ_k = R²(B_k + H) − R²(B_k) is pooled out of fold over the
held-out patients with a patient-bootstrap CI and its within-patient permutation null, exactly as `pool_folds` does. The
table beside each discovery gives Δ_N … Δ_B3, the first level at which the interval includes 0 (the absorbing step:
composition, known programmes or neighbourhood programmes) and the v0.6 tier name, so the reader sees which known
explanation already carries the relation and how much of it is left. None of this gates: the decomposition is written
under `discovery.explanation_decomposition` and not under `discovery.ladder`, because `aggregate_funnel` pools by tier
whenever receipts carry a ladder block, and v0.7's pooling must stay the v0.5 passing-fold rule.

Open items before registration (DRAFT r2): the frozen checkpoint path and sha256, a runner option for the interleaved merge of learned_search and classical_pair without cross-validated re-ranking, the cross-fold scoring pass in pooling, a hest_multi episode path for concept targets and ependymoma, and, for ependymoma, an external cohort and B1 (CellViT has
not been run on those slides).
