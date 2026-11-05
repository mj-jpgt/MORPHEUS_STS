# Tier III: blind historical rediscovery targets (LOCKIN Sec. 7)

Not a predeclaration document -- a compact target list per domain that has
real data already on the persistent store, checked directly rather than
assumed. Each target: the data file, the object as a declared
(operands, operator, parameters, context, lag, neighbourhood) truth in
dbench's own vocabulary, the literature reference establishing it, and how
success is scored through the RDC harness. This is the substrate Headline
5's prospective campaigns ramp on -- these five ARE the historical/blind
targets a discovery system is checked against before any campaign runs on
genuinely new data.

## Biology (data confirmed on the store)

Store paths checked directly:
`/lambda/nfs/geeg/biorag3_persistent_20260711/morpheus_substrate_mirror/hest/ccrcc/`
(real H&E whole-slide images + DINOv2/GigaPath/H-Optimus foundation-model
patch embeddings, `embeddings/{dinov2,gigapath,hoptimus}/`) and
`external/hf_tcga_cptac_cgga/cptac/tables/ccrcc/` (real CPTAC clear-cell
renal cell carcinoma cohort: `proteomics__umich.parquet`,
`transcriptomics__{bcm,washu,broad}.parquet`, `CNV__washu.parquet`,
`cibersort__washu.parquet`, `xcell__washu.parquet`, `somatic_mutation__
harmonized.parquet`, `clinical__mssm.csv`).

1. **BAP1-loss transcriptional/immune program.** BAP1 (a tumour suppressor,
   ~10-15% of ccRCC) loss defines a distinct, aggressive molecular subtype
   with a characteristic immune-infiltration signature.
   - Data: `somatic_mutation__harmonized.parquet` (BAP1 status) joined to
     `transcriptomics__washu.parquet` and `cibersort__washu.parquet`.
   - Truth: operator=CAUSE (do-effect framing: BAP1-loss as the intervention
     proxy), operands=(BAP1_mutation_status, immune_program_score),
     context={"cohort": "ccrcc"}, parameters={effect: "shifts CD8 T-cell and
     M2 macrophage CIBERSORT fractions"}, lag=None, neighbourhood=None.
   - Reference: Clark et al., *Cell* 2019, "Integrated Proteogenomic
     Characterization of Clear Cell Renal Cell Carcinoma" (CPTAC ccRCC
     flagship paper; BAP1 subtype + immune correlates).
   - Scored: RDC harness treats BAP1 status as the context/regime variable
     (CONDITION-style), the CIBERSORT immune fraction as the target; success
     = recovering the BAP1-conditioned shift with replication across the
     cohort's site/batch splits as D_R.

2. **CIBERSORT vs xCell immune-deconvolution agreement (a known, replicated
   cross-method concordance, not a single-method artifact).** Two
   independent deconvolution methods on the SAME bulk transcriptome should
   agree on the dominant immune cell fractions (T-cell, macrophage) in
   ccRCC's known "immune-hot" phenotype.
   - Data: `cibersort__washu.parquet` and `xcell__washu.parquet`.
   - Truth: operator=ADD (or LINCOMB), operands=(cibersort_CD8T,
     xcell_CD8T), target=each other (a cross-method agreement object, not a
     Y-relationship), context={"cohort": "ccrcc"}.
   - Reference: Newman et al. 2015 (CIBERSORT), Aran et al. 2017 (xCell);
     agreement in ccRCC specifically reported in the CPTAC ccRCC paper
     above.
   - Scored: recall = correlation-based object recovery (a LINCOMB/ADD
     alignment check) between the two independently-computed immune scores;
     replication = held-out patient subsets.

3. **Spatial immune/stromal compartmentalisation in ccRCC histology (a
   known tumour-vs-stroma spatial organisation, visible in H&E).** ccRCC's
   clear-cell morphology and immune infiltrate show a well-documented
   spatial organisation (immune cells enriched at the tumour-stroma
   interface, not uniformly distributed).
   - Data: `hest/ccrcc/embeddings/hoptimus/` (patch-level foundation-model
     embeddings with tissue coordinates).
   - Truth: operator=SPACE, operands=(tumour_region_indicator,
     immune_patch_score), parameters={"agg": "mean", "neighbourhood":
     "k-NN on patch coordinates"}, context=None, lag=None.
   - Reference: standard renal pathology (WHO/ISUP grading literature);
     immune-stroma spatial organisation specifically documented in
     ccRCC spatial transcriptomics studies (e.g. the STHELAR kidney atlas,
     also on this store at `external_acquisition_20260822/sthelar_kidney/`).
   - Scored: SPACE-operator recall via `dbench.oracle.scan_spatial`-style
     neighbourhood aggregation against the real patch coordinates; the
     "true" answer is qualitative (tumour-proximal immune enrichment is
     KNOWN, not a planted number), so success is a directional, replicated
     recovery (positive spatial coefficient, consistent sign across held-out
     slides), not an exact-value match.

## Climate (data confirmed on the store)

Store path checked directly:
`morpheus_substrate_mirror/climate/ncep_ncar/` (real NCEP/NCAR reanalysis
monthly means: `air.2m.mon.mean.nc`, `hgt.mon.mean.nc`,
`prate.sfc.mon.mean.nc`) and `morpheus_substrate_mirror/climate/
teleconnections/` (real monthly teleconnection indices: `oni.ascii.txt`
[ENSO], `nao_index.tim`, `pna_index.tim`, `ao.index.b50.current.ascii`,
`ea_index.tim`, `wp_index.tim`, `epnp_index.tim`, `scand_index.tim`,
`tnh_index.tim`, `pt_index.tim`, `eawr_index.tim`, `poleur_index.tim`).

1. **ENSO (ONI) teleconnection to North American winter precipitation, with
   a known seasonal lag.** El Nino winters are well-documented to bring
   above-average precipitation to the southern US and below-average to the
   Pacific Northwest, peaking DJF with ONI leading by 0-2 months.
   - Data: `oni.ascii.txt` (ONI index) + `prate.sfc.mon.mean.nc` (real
     precipitation field).
   - Truth: operator=LAG (composed with SPACE for the spatial footprint),
     operands=(ONI, precip_grid), parameters={"lag": "0-2 months"},
     context={"season": "DJF"}, neighbourhood="the known dipole spatial
     pattern (southern-US positive, Pacific-NW negative)".
   - Reference: Ropelewski & Halpert 1986, *Mon. Wea. Rev.*, "North American
     Precipitation and Temperature Patterns Associated with the El
     Nino/Southern Oscillation"; the canonical ENSO-teleconnection paper.
   - Scored: LAG+SPACE composite recall (mirrors `boss1`'s own
     CONDITION-THRESHOLD-SPACE-LAG shape, minus the threshold/condition
     terms) against the REAL reanalysis field; replication = held-out
     decades of the reanalysis record; a correct rediscovery must recover
     BOTH the lag window and the dipole sign pattern, not just a marginal
     correlation.

2. **NAO teleconnection to European winter temperature (a known, opposite-
   sign dipole).** Positive NAO winters bring mild, wet conditions to
   northern Europe and cold conditions to the Mediterranean/Greenland --
   the other canonical teleconnection, useful as a SECOND known object on
   the same substrate (checks the discovery system finds two independent,
   non-confounded objects rather than one and its aliases).
   - Data: `nao_index.tim` + `air.2m.mon.mean.nc`.
   - Truth: operator=SPACE (near-contemporaneous, lag~0-1 month),
     operands=(NAO, temp_grid), context={"season": "DJF"},
     neighbourhood="opposite-sign dipole, northern Europe vs
     Mediterranean/Greenland".
   - Reference: Hurrell 1995, *Science*, "Decadal Trends in the North
     Atlantic Oscillation: Regional Temperatures and Precipitation".
   - Scored: same SPACE-composite path as target 1; a discovery system run
     on this substrate should recover ENSO and NAO as two DISTINCT objects
     (a confound/distinctness check, mirroring `dbench.compose_scoring`'s
     own distinctness rule) -- crediting only one while missing the other
     is a partial, not a full, rediscovery.

3. **PNA pattern's known 500hPa geopotential-height footprint.** The
   Pacific-North American pattern has an established, textbook 500hPa
   height anomaly signature (distinct from both ENSO's and NAO's), giving a
   THIRD, independently-verifiable object on the same reanalysis file.
   - Data: `pna_index.tim` + `hgt.mon.mean.nc`.
   - Truth: operator=SPACE, operands=(PNA, hgt500_grid),
     context={"season": "DJF"}, neighbourhood="the PNA's four-centre
     height-anomaly pattern (canonical Wallace & Gutzler centres)".
   - Reference: Wallace & Gutzler 1981, *Mon. Wea. Rev.*, "Teleconnections
     in the Geopotential Height Field during the Northern Hemisphere
     Winter" (the paper that defines PNA/NAO as objects in the first
     place).
   - Scored: as targets 1-2; the three together give a positive-control set
     of KNOWN, textbook, mutually-distinct spatial teleconnection objects on
     one real substrate before any prospective climate campaign runs.

## Physics (data confirmed on the store; a genuine planted-answer benchmark, not merely real data)

Store path checked directly:
`morpheus_substrate_mirror/particle_physics/lhco2020/` (real LHC Olympics
2020 anomaly-detection challenge data: `events_LHCO2020_BlackBox{1,2,3}.h5`
+ `.masterkey` files, `events_LHCO2020_backgroundMC_Pythia.h5`).

1. **The BlackBox1 planted resonance.** LHCO2020 is itself a blind-
   rediscovery benchmark by design: BlackBox1 contains a real
   collider-physics background (QCD dijet) PLUS an injected new-physics
   signal (a heavy resonance decaying to two other particles, each further
   decaying to jet pairs) at an undisclosed mass and cross-section during
   the original 2020 challenge -- the answer (the resonance mass and decay
   chain) was published after the challenge closed and is now public via
   the `.masterkey` file already on this store.
   - Data: `events_LHCO2020_BlackBox1.h5` (events) +
     `events_LHCO2020_BlackBox1.masterkey` (the ground-truth signal
     parameters, already on the store).
   - Truth: operator=ANOMALY (a rare-population/regime object -- the
     signal is a small minority of events in a dominant QCD background,
     exactly the `dbench.families.make_anomaly`/composition ANOMALY shape),
     operands=(dijet_invariant_mass, event_is_signal),
     parameters={"resonance_mass": "from masterkey", "prevalence": "from
     masterkey, a small minority fraction"}, context=None, neighbourhood=
     None (an event-level, not spatial, object).
   - Reference: Kasieczka et al. 2021, *Rep. Prog. Phys.*, "The LHC Olympics
     2020: A Community Challenge for Anomaly Detection in High Energy
     Physics" (the challenge paper; the masterkey answer is the
     community-verified ground truth).
   - Scored: through the SAME native anomaly specialist added to the H2
     bank this delivery (`dbench.oracle.scan_anomaly`: robust z-score +
     isolation forest on a parent-adjusted residual) plus the existing
     Mahalanobis specialist (`specialist_anomaly`), run on the real dijet
     mass spectrum; recall = whether the flagged excess localises to the
     masterkey's true resonance mass window; FCR = flags outside that
     window; replication = the independent BlackBox2/BlackBox3 files (each
     with their own, different masterkey answers) as a held-out check that
     the SAME detector, not a re-tuned one, finds each box's own signal.

## Materials (data confirmed on the store)

Store path checked directly: `morpheus_substrate_mirror/materials/
oc20_is2re/is2res_train_val_test_lmdbs/` (real Open Catalyst 2020 IS2RE
task: initial adsorbate-catalyst structures paired with relaxed adsorption
energies, LMDB format, `README_is2res.md` on the same path).

1. **Adsorption-energy scaling relations (a well-established structure-
   property motif in heterogeneous catalysis).** Adsorption energies of
   chemically related species (e.g. *OH, *O, *OOH on the same metal/alloy
   surface family) are known to scale linearly with each other across
   catalyst compositions -- the basis of the entire computational catalyst
   -screening literature.
   - Data: `is2res_train_val_test_lmdbs/data/` (structure + relaxed energy
     pairs).
   - Truth: operator=LINCOMB (a single linear scaling relation, not a sum
     of independent effects -- matches `dbench.families.make_linear_
     combination`'s own truth object shape), operands=(E_ads_species_A,
     E_ads_species_B), parameters={"slope": "~0.5-1.0 depending on species
     pair, from the scaling-relations literature", "intercept": "species-
     pair-specific"}, context={"surface_family": "transition metal (111)
     facets"}, neighbourhood="graph neighbourhood of the adsorbate binding
     site (the local coordination environment)".
   - Reference: Abild-Pedersen et al. 2007, *Phys. Rev. Lett.*, "Scaling
     Properties of Adsorption Energies for Hydrogen-Containing Molecules on
     Transition-Metal Surfaces" (the foundational scaling-relations paper);
     Norskov et al. 2011, *Nature Chemistry* review for the broader
     structure-property framework OC20 itself was built to accelerate.
   - Scored: LINCOMB recall on the real relaxed-energy pairs (species A vs
     species B adsorption energy across the real catalyst-composition
     space in the dataset); the object is GRAPH-neighbourhood-aware (the
     local binding site's coordination, not the whole crystal), matching
     `dbench.families.graph_motif`'s domain rather than a bare tabular
     LINCOMB; replication = held-out catalyst compositions (OC20's own
     train/val/test split already encodes this).

(A second physics substrate, `oc20_is2re` doubles as materials -- no
distinct particle-physics-only structural motif beyond the LHCO2020 anomaly
above was identified as already on the store; stated plainly rather than
padded to reach a target count.)
