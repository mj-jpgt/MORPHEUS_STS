# Evidence cards G3, G4, G5, G7: overlay plan and panel captions

Written 2026-11-02 after inspecting every rendered panel (v01 a–f and v02 a, g, h, i, j, pairs), the plot data behind
them, the card builder (`dbench/downstream/cards_v10.py`, `decode_v10.state_of`), the held-out refit
(`scratchpad/heldout_int21.py`) and the INT21 whole-slide image.

## 0. Facts that constrain every caption

- Section: INT21 of the discovery cohort (Patient 4; FFPE probe-based Visium; 4,974 scored spots), chosen as the
  section with the most scored spots; identical for all four cards.
- Panel a (v02) shows the HELD-OUT score: the relation refit on the other 23 patients and predicted on INT21. It agrees
  with the in-sample score at rank correlation 0.99 or better on all four cards. The v01 in-sample panel is kept as
  `a_spatial_insample_v01`.
- Held-out R² gain on INT21 alone (depth baseline against depth + relation): G3 0.26, G4 0.39, G5 0.39, G7 0.38. These
  are single-section numbers with no interval. Inference rests on the ladder (c, 24 patients) and the external panel
  (d, 16 patients).
- Colour on a, g, h, i, e = within-patient z. Panel a shares one symmetric colour limit between target and score, so the
  score (about half the target's spread) looks paler; that is shrinkage, not a rendering fault. H&E underlays draw a
  spot only where its value exceeds +1 z.
- G5 and G7 share the target (Glutathione gradient); their target maps are identical. G3 and G4 are two views of EMT-I
  (target rank correlation 0.80). Score overlap on the section: G3 vs G4 0.86, G5 vs G7 0.79, G3 vs G5 0.42.
- Panel j (composition) is a property of the section and is identical across the four cards.
- Panel b is partial dependence. Curved contours do not by themselves prove an interaction (an additive model with
  non-linear parts also gives curved contours). No caption claims an interaction from the picture.
- Tissue reading of INT21 (confirmed on the H&E underlays): pale tumour nodules separated by dark cellular fibrous
  septa; a dense stromal band with many blood-filled lumens along the upper-left edge and at the upper right; a pale
  homogeneous region at the lower left. The faint circles at the left and bottom edges of the H&E are the slide's
  fiducial frame, not data.

## 1. Overlay plan (status)

| id | overlay | claim it serves | status |
|---|---|---|---|
| O1 | H&E underlay: plain H&E, held-out score above +1 z, target above +1 z | the reader sees which tissue structures the relation marks | done (g_he_tissue, g_he_state, g_he_target) |
| O2 | Held-out score on INT21 (refit on 23 patients) | turns panel a from orientation into held-out evidence | done (a_spatial re-rendered) |
| O3 | Residual before and after the relation, held out | shows on tissue what the gain removes and what it leaves | done (h_residual) |
| O4 | Constituent maps beside the score | the score is a configuration of its constituents, not any one of them | done (i_constituents) |
| O5 | CellViT composition maps | the composition rung of the ladder in picture form | done (j_composition) |
| O6 | Paired cards G3/G4 and G5/G7 on one colour scale | the dedup groups are two routes to one tissue state | done (pair_G3_G4_state, pair_G5_G7_state) |
| O7 | Target and no-refit score on one GSE250163 and one Zenodo section | panel d in picture form | open; needs external per-spot scores from the box |
| O8 | Cross-cancer transfer forest (colorectal, ependymoma; over nuisance and over known biology; matched-null tick) | the P5 claim: which relations leave kidney | done (k_transfer; values in scratchpad cards/p5_transfer_values.json) |

All panels: separate files, no statistics printed on panels, colour bars outside the axes, pdf/svg/png with plot
data, manifests and QA reports. Mirrored to the store under figures_v08_paper/fig6_evidence_cards.

## 2. Shared legend text (once, for the whole card figure)

Evidence cards for the four highlighted relations. All spatial maps show section INT21 of the discovery cohort, the
section with the most scored spots; colour is within-patient z. The relation score on the section is held out: the
relation was refit on the other 23 patients and predicted on this one. H&E underlays draw a spot only where its value
exceeds +1 z; the faint circles at the left and bottom edges are the slide's fiducial frame. Inference rests on the
ladder (c) and the external panel (d); the unit is the patient (24 discovery, 16 external) with patient-bootstrap 95%
intervals; per-cohort points carry no interval. No scale bar: the pixel size is not certified.

Reading order for every card: where the state sits in the tissue (a, g); what it is made of (i, b); what it explains
and what it leaves (h, c, j); whether it holds in other kidney cohorts (d) and in other cancers (k); who carries it
(e); what it looks like at cell scale (f).

Panel k, shared text: the frozen relation applied without refitting to two other cancers that carry all of its
genes, colorectal (Valdeolivas 2024, 6 patients) and PFA ependymoma (11 patients). Filled circle: gain over the
cohort's depth baseline; open square: gain over the cohort's own known-biology baseline (its Hallmark and Gavish
scores); grey tick: 95th percentile of 99 Moran-matched random programmes under the same transfer. A transfer counts
when the interval and the point both clear zero and the tick, with direction agreement in at least 60% of patients.
Two further cohorts (GSE294385 colorectal metastases; Ji 2020 skin) could not be tested because discovery genes are
absent from their panels. The positive control (a split-half hypoxia relation) transfers in both cohorts.

## 3. Captions per card

### G3 — EMT-I gradient explained by myogenesis heterogeneity, angiogenesis and EMT IV

a. Where the state sits. Target (left) and held-out relation score (right) on the same section. The score marks a
subset of the target's high bands: the stromal band at the upper-left edge and the band at the lower right. The
central and right-hand bands, where the target is also high, are not captured. The relation explains part of the
target's spatial structure, not all of it.

g. The same on tissue. Plain H&E (g_he_tissue), held-out score above +1 z (g_he_state) and target above +1 z
(g_he_target). Score spots lie on the dense cellular stroma at the upper-left edge, in the vessel-rich stroma at the
upper right, and on the fibrous band at the lower right; none lie inside the pale tumour nodules. The target follows
every fibrous septum between nodules as well; set beside the score it shows which bands the relation accounts for and
which it does not. The relation describes a stromal, vessel-rich state at the tumour margin.

i. What it is made of. Each constituent on the section as within-section z, beside the held-out score. Myogenesis
heterogeneity is high only in the upper-left stroma, the upper-right corner and the lower-right band, which is where
the score is high. Angiogenesis and EMT IV are high along every septum and low in the lower-left region; where only
they are high, the score stays low. The score is a configuration: myogenesis heterogeneity sets where it can be high,
the other two modulate it.

b. Shape of the relation. Partial dependence on each constituent (left) and on the two strongest together (right).
Myogenesis heterogeneity carries most of the response and is close to linear; angiogenesis adds a weak positive
slope; EMT IV rises and then saturates above about one standard deviation. The near-vertical contours say the same
as panel i: the joint surface is dominated by myogenesis heterogeneity, shifted slightly by angiogenesis.

h. What it explains and what it leaves. Held-out residual of the target after the depth-only baseline (left) and
after adding the relation (right). The relation removes the upper-left and lower-right bands, overshooting to
negative values in parts of them, and leaves the central and right-hand septa untouched. On this section the gain is
0.26 in R²; the remaining bands are what the relation does not know.

c. Explanation ladder. Held-out gain in R² over 24 patients when the relation is added to a nuisance-only baseline,
then to baselines that also contain known programme scores, cell composition, or both. The gain falls from 0.21 to
0.17 (80% retained) and every interval excludes zero: the relation is not a restatement of known programmes or of
cell-type composition.

j. Composition on the section. CellViT nuclear fractions: neoplastic fills the nodules; connective is high in the
upper-left stroma and at the upper right; inflammatory is highest at the upper right and along the right-hand
stroma. The score band is a connective-rich region, so composition is a plausible rival explanation; the ladder (c)
shows it accounts for only a small part of the gain.

d. External replication without refitting. Pooled gain over nuisance 0.30 (95% CI 0.26–0.32) and over known biology
0.24 (0.20–0.27) across three independent ccRCC cohorts; every cohort is positive on both baselines. The frozen
relation transfers within kidney cancer.

k. Transfer to other cancers. In colorectal cancer the frozen relation gains 0.30 (0.25–0.39) over the depth
baseline and keeps 0.26 (0.19–0.34) over the colorectal cohort's own known biology, far above the matched null; the
same configuration of myogenesis heterogeneity, angiogenesis and EMT IV predicts the EMT-I gradient in a tumour of a
different organ. In ependymoma the gain over nuisance is small but positive (0.08, 0.00–0.15) and does not survive
known biology, so the claim there is directional only. G3 is the one card with a cross-cancer claim.

e. Who carries it. Frozen UMAP of 20,000 spots coloured by score, neoplastic fraction and patient. High scores
concentrate in one lobe at the bottom of the embedding, in the low-neoplastic region (median neoplastic fraction
0.05 among the top 5% of spots; cohort mean 0.42); 23 of 24 patients contribute, none more than 16%. The state is
shared across patients, not a single-patient artefact.

f. Cell-scale appearance (supplementary). Six composition-matched pairs of H&E crops at high (top) and low (bottom)
score. Within each pair the tissue looks alike: three pairs from one section are pink, near-acellular fibrous tissue,
one pair is blood-rich, one is densely cellular. The state is not a visible change in cell content at this
magnification; it is molecular and spatial.

Verified observations: score red at upper-left diagonal band, lower-right streak, small top-right corner, faint ring
at the left; target red additionally along a central diagonal band, a right-hand vertical band and a bottom
horizontal band. Crops: INT20 x3 (1/0, 11/0, 1/0 nuclei), INT15/INT13 (28/22), INT19 (52/23), INT6 (69/74).

### G4 — EMT-I heterogeneity explained by apical-junction heterogeneity, angiogenesis and EMT IV

a. Where the state sits. The target is highest in the upper-left stroma, along the central and right-hand septa and
the bottom band; the held-out score reproduces the upper-left band, the top-right corner and the lower-right band and
leaves the central septum unmarked. The score map is nearly the same as G3's: two relations from the EMT-I family
point at the same tissue structures.

g. The same on tissue. Score spots above +1 z occupy the same structures as in G3, with a darker upper-left band, an
extra column of spots along the left edge, and the strongest cluster at the top-right corner. None lie inside the
nodules.

i. What it is made of. Apical-junction heterogeneity reproduces the score map on its own; angiogenesis and EMT IV
(the same maps as in G3) are high along the septa where the score is low. As in G3, one heterogeneity term sets where
the state can exist.

b. Shape of the relation. Apical-junction heterogeneity carries the response and is close to linear; angiogenesis adds
a weak positive slope; EMT IV rises and saturates above about half a standard deviation. On the joint surface the
score is highest where apical-junction heterogeneity and EMT IV are both high and lowest where both are low.

h. What it explains and what it leaves. The relation removes the upper-left stroma and the lower-right band, with
negative overshoot at the far upper left and at the lower right; the central and right-hand septa remain. Gain on this
section 0.39 in R².

c. Explanation ladder. Gain falls from 0.34 (0.25–0.41) to 0.30 (0.19–0.38) after known biology and composition (87%
retained); the largest internal gain of the four cards, clear of zero at every rung.

j. Composition on the section. Identical to G3's panel: the score band is connective-rich, and the ladder shows
composition explains only a small part of the gain.

d. External replication without refitting. Pooled gain 0.31 (0.25–0.36) over nuisance and 0.25 (0.19–0.31) over known
biology; all three cohorts positive on both baselines, GSE250163 the strongest.

k. Transfer to other cancers. Neither cohort passes: colorectal 0.04 (−0.16 to 0.30), ependymoma 0.13 (−0.01 to
0.20) over nuisance, and less over known biology. The points are positive and clear the matched null, but the
intervals include zero. Although G4 marks the same kidney tissue state as G3, the apical-junction route does not
carry it to other organs; the two routes are equivalent in kidney and not beyond it.

e. Who carries it. High scores form a sharp spur at the top of the embedding, in low-neoplastic spots (median 0.11
among the top 5%); 23 of 24 patients contribute, none more than 17%.

f. Cell-scale appearance (supplementary). Six matched pairs; within-pair appearance is alike. Pair 5 comes from a
pale-stained section and pair 6 pairs a pale section with a normally stained one, so the contrast there is staining,
not state.

pair_G3_G4_state. The two held-out scores on one colour scale mark the same two structures; G4 is stronger in the
upper-left stroma (rank correlation 0.86 on the section). Two routes, one tissue state.

Verified observations: score red at upper-left diagonal band, top-right corner blob, right-edge blob, lower-right
streak. EMT IV flattens above z ≈ 0.5. Crops: INT15/INT13 (35/21), INT24 (45/27), INT20 (5/0), INT15 (19/20), INT3
(20/41, pale), INT2/INT19 (6/13).

### G5 — Glutathione gradient explained by hypoxia gradient and SLC3A1

a. Where the state sits. The target is high along many narrow bands that follow the fibrous boundaries between
tissue nodules and the edge of the pale region at the lower left. The held-out score traces the main bands as a crisp
network: the arc bounding the central nodule, the loop around the upper nodule, the right-hand septum and the
lower-left edge; the finer interior bands stay unmarked.

g. The same on tissue. Score spots above +1 z sit along the boundary line between tumour nodules and the fibrous
septa: the rim of the upper nodule, the upper-left rim of the central nodule, the right-hand septum, the junction of
septa at the lower right and the edge of the pale lower-left region. They are on the boundary, not deep in the stroma
and not in the nodule centres. The relation describes a boundary state where hypoxia changes steeply.

i. What it is made of. The hypoxia-gradient map alone reproduces the band network; SLC3A1 is patchy, with low values
in the upper-left stroma and the lower-left region. The score is the hypoxia-gradient map modulated by SLC3A1.

b. Shape of the relation. The hypoxia-gradient response is convex: flat below the mean and accelerating above it;
SLC3A1 adds a weak, saturating positive slope. The score is highest where both are high.

h. What it explains and what it leaves. The relation removes the main boundary bands, over-correcting the upper-left
arc to negative values; what remains is the ring around the small upper nodule, the upper-right corner and scattered
interior patches. Gain on this section 0.39 in R².

c. Explanation ladder. Gain 0.18 (0.08–0.25) over nuisance and 0.17 (0.07–0.25) after known biology and composition
(99% retained); adding known programme scores does not reduce the gain at all.

j. Composition on the section. The score bands run where the neoplastic fraction switches between nodule and septum,
so they are composition boundaries as much as hypoxia boundaries; the ladder shows that adjusting for composition
leaves the gain intact.

d. External replication without refitting. Pooled gain 0.07 (0.01–0.13) over nuisance and 0.08 (0.02–0.13) over known
biology. Replication rests on GSE250163 (0.21) and Zenodo (0.22); Li 2022 is null (−0.01). The pooled interval
excludes zero, but the claim is two cohorts of three.

k. Transfer to other cancers. No transfer: gains over nuisance of −0.04 in colorectal and 0.02 in ependymoma, both
intervals spanning zero; the known-biology gain was not computed because the first rung failed. The hypoxia
boundary state is a kidney finding, consistent with its weaker replication inside kidney (d).

e. Who carries it. High scores occupy the lower centre of the embedding, in spots of intermediate neoplastic fraction
(median 0.25 among the top 5%); all 24 patients contribute, none more than 16%. The two single-patient lobes at the
top of the embedding carry low scores.

f. Cell-scale appearance (supplementary). Six matched pairs; pair 2 is less well matched (12 vs 32 nuclei).
Within-pair appearance is alike; one pair is near-acellular fibrous tissue.

Verified observations: 2-D colour range ±1.6 (largest of the four); hypoxia-gradient curve from −0.38 at z = −1.1 to
+0.98 at z = 2. Crops: INT15 (18/20), INT21 (12/32), INT15 (24/29), INT23 (0/0), INT5 (46/30), INT17 (50/25).

### G7 — Glutathione gradient explained by hypoxia heterogeneity and SLC3A1

a. Where the state sits. Same target as G5 (left map identical). The held-out score traces the same band network
with broader bands, extra patches in the interior and a strong cluster at the lower-right junction of the septa.
Hypoxia heterogeneity marks the same boundaries as the hypoxia gradient, less sharply.

g. The same on tissue. The same nodule boundaries as G5 but as broader patches, with the strongest cluster at the
junction of septa at the lower right and an extra patch on the stromal strand at the left centre.

i. What it is made of. Hypoxia heterogeneity alone reproduces the broader band pattern, including the lower-right
junction cluster; SLC3A1 as in G5.

b. Shape of the relation. The hypoxia-heterogeneity response is sigmoidal: steepest between the mean and one standard
deviation above it, flattening beyond; SLC3A1 adds a weak saturating slope. The score is highest where both are high.

h. What it explains and what it leaves. Similar to G5; the lower-right junction is over-corrected to negative values.
Gain on this section 0.38 in R².

c. Explanation ladder. Gain 0.11 (0.04–0.17) over nuisance and 0.11 (0.03–0.17) after known biology and composition
(97% retained); the smallest internal gain of the four, still clear of zero at every rung.

j. Composition on the section. As G5: boundary bands at the nodule–septum switch; composition adjustment leaves the
gain intact.

d. External replication without refitting. Pooled gain 0.05 (0.00–0.11) over nuisance and 0.05 (0.00–0.09) over known
biology; the lower bounds sit just above zero. GSE250163 (0.15) and Zenodo (0.21) are positive; Li 2022 is null
(−0.02).

k. Transfer to other cancers. No transfer: −0.05 in colorectal and 0.02 in ependymoma over nuisance, intervals
spanning zero; known-biology gain not computed. As for G5, the boundary state does not leave kidney.

e. Who carries it. High scores occupy a lobe at the right of the embedding, in low-neoplastic spots (median 0.12 among
the top 5%); all 24 patients contribute, none more than 15%. The single-patient lobes at the left carry low scores.

f. Cell-scale appearance (supplementary). Five of six pairs come from section INT21 and show the same tumour tissue
at high and low score; nucleus counts are matched (12–32). The state is invisible at this magnification.

pair_G5_G7_state. Same boundary network on one colour scale; G7 broader, with a stronger lower-right junction (rank
correlation 0.79 on the section). Gradient and heterogeneity of hypoxia are two readouts of one boundary state.

Verified observations: pooled lower bounds 0.002 (nuisance) and 0.004 (known). Crops: INT16 (23/24), INT21 x5
(12/32, 18/14, 21/16, 21/15, 12/18).

## 4. Fixes and flags before the cards go into the paper

1. G5 and G7 share the target map; if both cards appear together, show the target once. Panel j is identical across
   all four cards; show it once.
2. The shared colour limit makes the score map paler than the target; keep it and say so once in the legend.
3. The single-section held-out gains (0.26–0.39) exceed the cohort-level ladder gains (0.11–0.34). INT21 was chosen
   for spot count, not for fit, but a reviewer may ask; the caption calls them single-section numbers with no
   interval.
4. G3's external pooled gain (0.30) exceeds its internal nuisance-only gain (0.21). W1/W2 should confirm the internal
   and external nuisance baselines are comparable before a reviewer asks.
5. No caption claims an interaction from the response surface.
6. Crop sheets stay supplementary: deterministic composition matching, several sections and stains per sheet.
7. Polish: the g_he_* colour bars span the full symmetric range although only spots above +1 z are drawn; a bar
   clipped to [1, max] would be cleaner.
8. Open: O7 (external-section overlay) if time allows after the W3 tables.
9. Panel k on G5 and G7 carries a legend entry for the open square although no square is drawn (legend kept identical
   across cards); the caption says the known-biology gain was not computed. On G5/G7 the null tick crosses the
   colorectal interval because the null sits near zero there; that is the data, not a rendering fault.
10. The evidence table (evidence_table_v01) holds the over-nuisance P5 rows but not the known-biology addendum;
    re-run `make all` there if the addendum should be ingested before the table is cited.

## 5. Added 2026-11-03: G3 vascular close-up (card_G3_71ed4499fb99/closeup/)

| id | overlay | what it adds | status |
|---|---|---|---|
| O9 | G3 vascular close-up: l_context (half-resolution H&E, 6000 x 6000 full-res px window, held-out G3 score drawn where > +1, close-up boxed), m_closeup (plain full-resolution H&E, 3000 x 3000 full-res px), n_closeup_spots (same, spot capture areas with score > +1 outlined), l_colourbar | anatomical reading of the main finding: the high-score band is the cellular, capillary-rich stroma between the capsular vessels and the clear-cell tumour nests | done |

Caption (G3, panels l-n). Held-out G3 score on the upper-left stroma of section INT21 (patient 4; relation fitted on the other 23 patients). (l) Spots with score above +1 (within-patient z) on the H&E; the box marks the close-up. (m) The boxed region at full resolution: two large erythrocyte-filled vessels lie at the capsular edge (upper left), a band of cellular stroma threaded with small congested capillaries runs diagonally, and clear-cell tumour nests fill the lower right. (n) The same region with the capture area of every spot scoring above +1 outlined and coloured by score. The highest scores sit in the capillary-rich stromal band; the large capsular vessels carry moderate scores; the tumour nests carry none. Pixel size is not certified for this slide, so no scale bar is drawn; the dark diagonal line at the upper right of (m) and (n) is a tissue fold. Spots are shown for visualisation only; the independent unit is the patient.

Decision on the owner's 2026-11-03 list: G4 overlay = card_G4 g_he_state (exists); G5 tissue example = card_G5 g_he_state (exists); G3 close-up = O9 (added; a convincing vessel is visible); cross-cancer G3 overlays = not built, quantitative transfer (panel k) is the evidence.

## 6. Added 2026-11-03: G3 cross-cancer overlay (card_G3_71ed4499fb99/crosscancer/)

| id | overlay | what it adds | status |
|---|---|---|---|
| O10 | G3 on kidney and colorectal tissue, same drawing rule: o_kidney_he / o_kidney_target / o_kidney_score (INT21, held out) and o_colorectal_he / o_colorectal_target / o_colorectal_score (Valdeolivas 2024 section ZEN42, patient 5, frozen kidney relation, no refit); p_colorectal_closeup and q_colorectal_closeup_spots (level-2 close-up of the densest high-score region); o_colourbar | the P5 transfer in picture form: the same relation marks tumour-adjacent fibrous stroma in both organs and leaves epithelium empty | done (per-spot values from W8, overlay_g3_crc/; pooled gain reproduces P5 0.300 to 1e-16) |

Per-section gain of the frozen relation on colorectal (12 sections, 6 patients, all positive): 0.10 to 0.51; ZEN42 0.514 (Spearman 0.74, 1,192 spots); ZEN48 (most spots, 2,382) 0.41 but few spots above +1, so ZEN42 is shown.
