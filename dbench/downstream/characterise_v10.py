"""W1b: claim-matched characterisation tests per distinct relation on the discovery cohort (Meylan 2022 ccRCC).

Engine: relation_v03 (fit_baseline + fit_sets, leave-one-patient-out gains, 24 patients).  Every number is a held-out
gain on whole patients; intervals are patient-level bootstraps of the per-patient contributions.

Tests per relation (evidence_rows.jsonl, one row per test):
  gms_claim_matched      G_MS = Perf(Y|B,R) - Perf(Y|B,R_reduced); reduction matched to the claim
                         (ring -> own location; ringS/ringE/nbrS/nbrE -> pooled; het -> own spot; grad ->
                          own spot + smooth ring1 + compartment label), B = depth + known biology + composition
  gms_own_spot           the same with every operator reduced to its own-spot value
  known_biology          gain over depth + B_known (50 Hallmark + 41 Gavish programmes minus target overlaps, 3 composition vars)
  composition            gain over B_known + NNLS deconvolution fractions + CellViT fractions at own spot and nbr6,
                         p from a composition-matched null (rows of the relation features permuted within
                         patient x composition stratum)
  technical              gain over B_known + haemoglobin/artifact score + valid-neighbour count
  ladder_d0..d3          gain over depth / +known / +composition / +technical and own-spot constituents
  het_matched_null       (het/grad relations) spatially smooth random programmes matched for Moran I and for correlation
                         with the target replace the het/grad variable
  segmentation_sensitivity  compartment-operator relations: gain under alternative CellViT dominance thresholds
Every test family first runs a positive control (a known relation that must pass) and its MDE (2.8 x bootstrap SE);
a test whose MDE exceeds the relation's own gain over nuisance is reported not_assessed.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

STUDY = "meylan2022_ccrcc"
RUN = "/home/ubuntu/morpheus_work_v08/runs/registered_v08/renal-spatial-v08-multiscale-20261026"
CELLVIT_ROOT = "runs/confirmation/v08_external_cellvit_zenodo_v01"
DECONV = ("/home/ubuntu/morpheus_work_cx/runs/development/counterexample/inputs/"
          "meylan2022_ccrcc_cx_inputs_v1_20261026_nnlsv2_deconvrows.npz")
ARCH = ("/home/ubuntu/cache/hest_multi/architecture/"
        "meylan2022_ccrcc_c3bf14aa1d01c9f0379a_architecture_v1_20261023.npz")
OUT = Path("runs/development/characterisation_v10")
N_BOOT, N_NULL, DEVICE = 2000, 49, "cuda"
MDE_Z = 2.8
HB_GENES = ("HBA1", "HBA2", "HBB")
OPS_POOLED = {"nbrS6": "nbr6", "nbrE6": "nbr6", "nbrS18": "nbr18", "nbrE18": "nbr18",
              "ringS1": "ring1", "ringE1": "ring1", "ringS2": "ring2", "ringE2": "ring2",
              "ringS3": "ring3", "ringE3": "ring3"}


def _codes(g):
    return np.unique(np.asarray(g).astype(str), return_inverse=True)[1].astype(np.int64)


def _within_std(X, pat):
    from dbench.downstream.relation import standardise_within
    return standardise_within(np.asarray(X, float), pat)[0].reshape(len(pat), -1)


class Ctx:
    """Discovery-cohort data shared by all relations."""

    def __init__(self):
        from morpheus.tier2.substrates.hest_multi import _programme_sets

        from dbench.downstream.external_cellvit_v08 import v08_external_cellvit_context
        from dbench.downstream.values import _adjacency, gradient_operator
        from dbench.scripts.external_replication import load_discovery

        self.coh = load_discovery()
        self.sets = _programme_sets()
        self.cctx = v08_external_cellvit_context(CELLVIT_ROOT)
        self.pat = np.asarray(self.coh.patient).astype(str)
        self.sec = np.asarray(self.coh.section).astype(str)
        self.n = len(self.pat)
        self.nb6 = np.asarray(self.coh.raw.neighbours[6], dtype=np.int64)
        self.nb18 = np.asarray(self.coh.raw.neighbours[18], dtype=np.int64)
        self.depth = np.log1p(np.asarray(self.coh.raw.depth, float))
        self._cache = {}
        dz = np.load(DECONV, allow_pickle=True)
        idx = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(dz["sample_id"], dz["barcode"]))}
        rows = np.array([idx.get((str(a), str(b)), -1) for a, b in zip(self.coh.raw.sample_id, self.coh.raw.barcode)])
        assert (rows >= 0).all()
        self.deconv = np.asarray(dz["F"], float)[rows]
        self.deconv_types = [str(t) for t in dz["deconv_types"]]
        a = np.load(ARCH, allow_pickle=True)
        cv = np.column_stack([a["neoplastic_fraction"], a["inflammatory_fraction"], a["connective_fraction"]])
        cv = np.asarray(cv, float)
        for s in np.unique(self.sec):                       # section-mean fill for spots without nuclei (registered fallback)
            r = np.flatnonzero(self.sec == s)
            fin = np.isfinite(cv[r]).all(1)
            cv[r[~fin]] = cv[r[fin]].mean(0) if fin.any() else 0.0
        self.cellvit = cv
        self.epi_fraction = np.asarray(a["epithelial_fraction"], float)
        self.n_nuclei = np.asarray(a["n_nuclei"], float)
        self.segmented = np.asarray(a["segmented"], bool)
        X = self.coh.logcpm
        hb = [self.coh.panel[g] for g in HB_GENES if g in self.coh.panel]
        self.hb = np.asarray(X[:, hb].mean(axis=1)).ravel()
        self.n_valid_nb = (self.nb6 >= 0).sum(1).astype(float)
        self.adj18 = _adjacency(self.nb18, self.n)
        self.Wx, self.Wy, _ = gradient_operator(np.asarray(self.coh.raw.coords, float), self.nb18)
        lab = self.cctx["compartments"](self.coh)
        self.epi_label = np.asarray(lab["E"], bool)
        self.stro_label = np.asarray(lab["S"], bool)

    # ---- columns
    def col(self, name):
        if name not in self._cache:
            from dbench.downstream.values import cohort_values
            self._cache[name] = np.asarray(cohort_values(self.coh, name, self.sets, self.coh.panel, self.cctx), float)
        return self._cache[name]

    def nbmean(self, v, k=6):
        from dbench.downstream.values import neighbour_mean
        return neighbour_mean(np.nan_to_num(v), self.coh.raw.neighbours[k])

    def apply_op(self, op, v):
        from dbench.downstream.values import neighbour_moments
        if op == "het":
            return neighbour_moments(np.asarray(v, float)[:, None], self.adj18)[1][:, 0]
        if op == "grad":
            return np.sqrt((self.Wx @ v) ** 2 + (self.Wy @ v) ** 2)
        raise ValueError(op)

    def composition_block(self):
        if "comp" not in self._cache:
            grp = {"tumour": ["Tumor"], "immune": ["B_cell", "Macrophage", "Mast", "Monocyte", "NK_cell", "Plasma", "T_cell"],
                   "endothelial": [t for t in self.deconv_types if t.startswith("Endothelial")],
                   "mural": ["Pericyte", "vSMC"], "tubule": ["Collecting_duct", "Distal_tubule", "Loop_of_Henle",
                                                              "Proximal_tubule", "Urothelium"]}
            g = np.column_stack([self.deconv[:, [self.deconv_types.index(t) for t in ts]].sum(1) for ts in grp.values()])
            own = np.column_stack([self.deconv, self.cellvit, g])
            nb = np.column_stack([self.nbmean(own[:, j]) for j in range(own.shape[1])])
            self._cache["comp"] = (np.column_stack([own, nb]), np.column_stack([self.cellvit, g, nb[:, -(g.shape[1] + 3):]]))
        return self._cache["comp"]


# ---------------------------------------------------------------- relation parsing
def target_for_known(target):
    from dbench.downstream.values import parse_candidate
    pt = parse_candidate(target)
    if pt.get("kind") == "operator":
        base = pt["base"]
        return f"programme:{base['set']}" if base["kind"] == "programme" else f"gene:{base['genes_key']}"
    return target


def op_of(name):
    m = re.match(r"^(ring[123]|ring[SE][123]|nbr[SE](?:6|18)|het|grad):", name)
    return m.group(1) if m else None


def body_of(name):
    return name.split(":", 1)[1]


def reductions(names):
    """(claim-matched columns, own-spot columns) as lists of column specs (strings, or ('grad_alt', body))."""
    claim, own = [], []
    for n in names:
        op = op_of(n)
        if op is None:
            claim.append(n)
            own.append(n)
        elif op in OPS_POOLED:
            tr = OPS_POOLED[op]
            claim.append(f"{tr}:{body_of(n)}" if tr.startswith("nbr") else f"{tr}:{body_of(n)}")
            own.append(body_of(n))
        elif op.startswith("ring"):
            claim.append(body_of(n))
            own.append(body_of(n))
        elif op == "het":
            claim.append(body_of(n))
            own.append(body_of(n))
        elif op == "grad":
            claim.append(("grad_alt", body_of(n)))
            own.append(body_of(n))
    return claim, own


def build_columns(ctx, specs):
    cols = []
    for s in specs:
        if isinstance(s, tuple):
            v = ctx.col(s[1])
            cols += [v, ctx.col("ring1:" + s[1]), v * ctx.epi_label.astype(float)]
        else:
            cols.append(ctx.col(s))
    return np.column_stack(cols)


# ---------------------------------------------------------------- engine
def fit_gain(Z, ys, pat, C):
    """LOPO gain of feature sets Z (n,m,k) over the baseline ridge on C.  Returns per-patient contributions."""
    from morpheus_encoder.learned_search import relation_v03 as rv
    base = rv.fit_baseline(ys, pat, C)
    out = rv.fit_sets(Z, base, ys, pat, device=DEVICE)
    codes = _codes(pat)
    sst = np.array([np.sum((ys[codes == p] - ys[codes == p].mean()) ** 2) for p in range(codes.max() + 1)])
    return np.asarray(out["lopo"]) * sst[:, None], sst        # (P, m) contributions, (P,)


def summarise(contrib, sst, seed=0, n_boot=N_BOOT):
    """Pooled gain, patient-bootstrap CI / one-sided p / SE / MDE.  contrib (P,) or difference of contributions."""
    P = len(sst)
    est = float(contrib.sum() / sst.sum())
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, P, size=(n_boot, P))
    boot = contrib[ix].sum(1) / sst[ix].sum(1)
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {"estimate": est, "ci95": ci, "p": float((1 + np.sum(boot <= 0)) / (n_boot + 1)),
            "se": float(boot.std()), "mde": float(MDE_Z * boot.std()),
            "n_patients": int(P), "patients_positive": int((contrib > 0).sum())}


def status_of(s, ref_gain, pc_ok, alpha=0.05):
    if not pc_ok:
        return "not_assessed", "positive control failed"
    if s["n_patients"] < 5:
        return "not_assessed", "fewer than 5 patients"
    if s["mde"] > abs(ref_gain):
        return "not_assessed", f"underpowered: MDE {s['mde']:.3f} exceeds the relation's own gain {ref_gain:.3f}"
    if s["ci95"][0] > 0 and s["p"] <= alpha:
        return "pass", ""
    return "fail", ""


def row(rid, test, s, status, reason, pc, extra=None):
    return {"relation_id": rid, "test": test, "estimate": s.get("estimate"), "ci": s.get("ci95"), "p": s.get("p"),
            "q": None, "mde": s.get("mde"), "positive_control_passed": pc, "status": status, "reason": reason,
            **(extra or {})}


class Relation:
    def __init__(self, ctx, rid, names, target, family, row_views):
        from dbench.downstream.blind import known_columns_v03
        from dbench.downstream.values import discovery_row_mask
        self.ctx, self.rid, self.names, self.target, self.family = ctx, rid, list(names), target, family
        self.claim, self.own = reductions(self.names)
        mask = (discovery_row_mask(ctx.coh, row_views, ctx.sets, ctx.coh.panel, ctx.cctx)
                if row_views else np.ones(ctx.n, bool))
        Z = build_columns(ctx, self.names)
        Zc = build_columns(ctx, self.claim)
        Zo = build_columns(ctx, self.own)
        y = ctx.col(target)
        K, self.known_names, self.known_info = known_columns_v03(STUDY, target_for_known(target), ctx.coh)
        comp_all, comp_cv = ctx.composition_block()
        tech = np.column_stack([ctx.hb, ctx.n_valid_nb])
        ok = mask & np.isfinite(Z).all(1) & np.isfinite(Zc).all(1) & np.isfinite(Zo).all(1) & np.isfinite(y)
        ok &= np.isfinite(K).all(1)
        self.ok = ok
        self.pat = ctx.pat[ok]
        self.codes = _codes(self.pat)
        w = lambda A: _within_std(np.asarray(A, float)[ok], self.pat)  # noqa: E731
        self.Z, self.Zc, self.Zo = w(Z), w(Zc), w(Zo)
        self.ys = _within_std(y[ok], self.pat).ravel()
        self.D, self.K, self.COMP, self.TECH = w(ctx.depth), w(K), w(comp_all), w(tech)
        self.own_nuis = w(Zo)
        self.n_rows = int(ok.sum())
        self.n_patients = int(len(np.unique(self.pat)))
        self.Zraw_ok = Z[ok]

    def base(self, *parts):
        return np.column_stack(parts)


def gain_over(rel, C, Z=None):
    Z = rel.Z if Z is None else Z
    contrib, sst = fit_gain(Z[:, None, :], rel.ys, rel.pat, C)
    return contrib[:, 0], sst


# ---------------------------------------------------------------- positive controls
def positive_controls(ctx):
    """Known relations that must pass before any discovery is scored."""
    res = {}
    rng = np.random.default_rng(7)

    def mk(target, feats, op_views=None):
        class R:  # minimal relation shell
            pass
        r = R()
        y = ctx.col(target)
        Z = np.column_stack([ctx.col(f) for f in feats])
        ok = np.isfinite(Z).all(1) & np.isfinite(y)
        r.pat = ctx.pat[ok]
        r.ys = _within_std(y[ok], r.pat).ravel()
        r.Z = _within_std(Z[ok], r.pat)
        r.D = _within_std(ctx.depth[ok], r.pat)
        r.ok = ok
        return r

    # PC1 own-spot recovery: Hypoxia programme from the Hallmark hypoxia score; shuffled feature as negative control
    r = mk("programme:mp:Hypoxia", ["programmes:hallmark:HALLMARK_HYPOXIA"])
    c, sst = gain_over(r, r.D)
    s = summarise(c, sst)
    perm = r.Z.copy()
    for p in np.unique(r.pat):
        ix = np.flatnonzero(r.pat == p)
        perm[ix] = r.Z[rng.permutation(ix)]
    cn, sstn = gain_over(r, r.D, perm)
    sn = summarise(cn, sstn)
    res["gain_recovery"] = {"summary": s, "negative_control": sn, "passed": bool(s["ci95"][0] > 0 and sn["ci95"][0] <= 0)}
    # PC2 G_MS: ring1 mean of hypoxia predicted by ring1 of the Hallmark score beats its own-spot value
    r2 = mk("ring1:programme:mp:Hypoxia", ["ring1:programmes:hallmark:HALLMARK_HYPOXIA"])
    own = _within_std(ctx.col("programmes:hallmark:HALLMARK_HYPOXIA")[r2.ok], r2.pat)
    c1, sst = gain_over(r2, r2.D)
    c0, _ = gain_over(r2, r2.D, own)
    s2 = summarise(c1 - c0, sst)
    res["gms"] = {"summary": s2, "passed": bool(s2["ci95"][0] > 0)}
    # PC3/PC4 matched nulls are run in relation form (see matched_null_controls)
    return res


# ---------------------------------------------------------------- nulls
def composition_strata(ctx, rel):
    """patient x composition strata: tertiles (within patient) of the own-spot CellViT neoplastic and inflammatory
    fractions and a median split of the nbr6 neoplastic fraction (18 strata per patient)."""
    cv = ctx.cellvit[rel.ok]
    nbn = ctx.nbmean(ctx.cellvit[:, 0])[rel.ok]
    lab = np.zeros(len(cv), dtype=np.int64)

    def q(v, k):
        e = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
        return np.searchsorted(e, v, side="right")
    for p in np.unique(rel.pat):
        ix = np.flatnonzero(rel.pat == p)
        lab[ix] = q(cv[ix, 0], 3) * 6 + q(cv[ix, 1], 3) * 2 + q(nbn[ix], 2)
    return rel.codes * 100 + lab


def comp_target_null(ys, pat, Z, C, strata, n_null=199, seed=3):
    """Within-stratum permutation of the TARGET (patient x composition strata), baseline and relation refit on each
    draw: null distribution of the pooled LOPO gain over the composition baseline."""
    from morpheus_encoder.learned_search import relation_v03 as rv
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    out = np.empty(n_null)
    for d in range(n_null):
        perm = np.arange(len(ys))
        for ix in groups:
            perm[ix] = ix[rng.permutation(len(ix))]
        yp = ys[perm]
        base = rv.fit_baseline(yp, pat, C)
        o = rv.fit_sets(Z[:, None, :], base, yp, pat, device=DEVICE)
        out[d] = float(np.asarray(o["g_sup"])[0])
    return out


def gene_moran(ctx):
    """Moran I (nbr6 graph, within-patient standardised) of every panel gene; cached on the context."""
    if "gene_moran" in ctx._cache:
        return ctx._cache["gene_moran"]
    import scipy.sparse as sp
    nb = ctx.nb6
    r, c = np.nonzero(nb >= 0)
    A = sp.csr_matrix((np.ones(len(r)), (r, nb[r, c])), shape=(ctx.n, ctx.n))
    A = sp.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    X = ctx.coh.logcpm
    G = X.shape[1]
    out = np.zeros(G)
    ps = [np.flatnonzero(ctx.pat == p) for p in np.unique(ctx.pat)]
    for g0 in range(0, G, 1000):
        Z = np.asarray(X[:, g0:g0 + 1000].toarray(), dtype=np.float32)
        for ix in ps:
            Z[ix] -= Z[ix].mean(0)
            sd = Z[ix].std(0)
            Z[ix] /= np.where(sd > 1e-6, sd, 1.0)
        out[g0:g0 + 1000] = (Z * (A @ Z)).mean(0) / np.maximum((Z * Z).mean(0), 1e-9)
    ctx._cache["gene_moran"] = out
    return out


def smooth_pool(ctx, n_sets=6000, seed=5):
    """Pool of random gene-set scores (own spot, within-patient standardised, float32) with sizes 1-300 genes drawn from
    genes above a random Moran quantile; Moran I of every score on the full cohort.  Built once per process."""
    if "pool" in ctx._cache:
        return ctx._cache["pool"]
    import scipy.sparse as sp
    gm = gene_moran(ctx)
    rng = np.random.default_rng(seed)
    X = ctx.coh.logcpm
    qs = np.array([0.0, 0.5, 0.7, 0.85, 0.95])
    sets, sizes, cols = [], [], np.empty((ctx.n, n_sets), dtype=np.float32)
    ps = [np.flatnonzero(ctx.pat == p) for p in np.unique(ctx.pat)]
    for j in range(n_sets):
        q = rng.choice(qs)
        pool = np.flatnonzero(gm >= np.quantile(gm, q))
        size = int(round(np.exp(rng.uniform(0, np.log(300)))))
        gi = rng.choice(pool, size=min(size, len(pool)), replace=False)
        v = np.asarray(X[:, gi].mean(axis=1)).ravel().astype(np.float32)
        for ix in ps:
            v[ix] -= v[ix].mean()
            sd = v[ix].std()
            v[ix] /= sd if sd > 1e-6 else 1.0
        cols[:, j] = v
        sets.append(gi)
        sizes.append(len(gi))
    nb = ctx.nb6
    r, c = np.nonzero(nb >= 0)
    A = sp.csr_matrix((np.ones(len(r)), (r, nb[r, c])), shape=(ctx.n, ctx.n))
    A = sp.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    moran = (cols * (A @ cols)).mean(0) / np.maximum((cols * cols).mean(0), 1e-9)
    ctx._cache["pool"] = {"sets": sets, "sizes": np.array(sizes), "cols": cols, "moran": moran, "A": A}
    return ctx._cache["pool"]


def matched_programmes(ctx, rel, var_index, n_accept=199):
    """Spatially smooth random programmes matched to the variable: from the pool, size within 0.5x-1.5x of the
    variable's programme and with at most 20 % of its genes in the variable's programme (the overlap rule), keep the n_accept nearest in (Moran I, |r| with the target within
    patient).  Match quality is returned and gates the verdict.  Returns (own-spot raw-scale columns, n pool, info)."""
    from dbench.downstream.values import candidate_genes, parse_candidate
    base_name = body_of(rel.names[var_index])
    P = smooth_pool(ctx)
    v0 = ctx.col(base_name)
    z0 = _within_std(v0, ctx.pat).ravel().astype(np.float32)
    m0 = float((z0 * (P["A"] @ z0)).mean() / (z0 * z0).mean())
    ok = rel.ok
    ys = rel.ys.astype(np.float32)
    r0 = abs(float(np.corrcoef(z0[ok], ys)[0, 1]))
    C = P["cols"][ok]
    corr = np.abs((C - C.mean(0)).T @ (ys - ys.mean())) / (np.maximum(C.std(0), 1e-9) * ys.std() * len(ys))
    if base_name.startswith("raw_genes:"):
        own = {base_name.split(":", 1)[1]}
    else:
        own = set(candidate_genes(parse_candidate(base_name), ctx.sets, ctx.coh.panel))
    size0 = max(1, len(own))
    genes = np.asarray(ctx.coh.raw.genes)
    own_idx = set(np.flatnonzero(np.isin(genes, list(own))).tolist())
    elig = np.array([(0.4 * size0 <= P["sizes"][j] <= 2.0 * size0 or size0 <= 2 and P["sizes"][j] <= 3) and
                     len(own_idx & set(P["sets"][j].tolist())) <= 0.2 * len(P["sets"][j]) for j in range(len(P["sizes"]))])
    d = np.where(elig, np.abs(P["moran"] - m0) / 0.1 + np.abs(corr - r0) / 0.1, np.inf)
    order = np.argsort(d)[:n_accept]
    order = order[np.isfinite(d[order])]
    dm, dr = np.abs(P["moran"][order] - m0), np.abs(corr[order] - r0)
    sel = [P["cols"][:, j].astype(float) for j in order]
    info = {"moran0": m0, "abs_r0": r0, "size": size0, "n_eligible": int(elig.sum())}
    if len(order) >= 10:
        info.update(median_abs_dmoran=float(np.median(dm)), median_abs_dr=float(np.median(dr)),
                    max_abs_dmoran=float(dm.max()), max_abs_dr=float(dr.max()), n_selected=int(len(order)),
                    median_abs_dmoran_nearest49=float(np.median(dm[:49])), median_abs_dr_nearest49=float(np.median(dr[:49])))
    else:
        info.update(median_abs_dmoran=9.0, median_abs_dr=9.0, max_abs_dmoran=9.0, max_abs_dr=9.0)
    return sel, int(elig.sum()), info


def het_matched_null(ctx, rel, C):
    """199 nearest matched smooth random programmes (nearest first, so the first 49 are the earlier 49-draw null,
    kept as a sensitivity row)."""
    idx = [i for i, n in enumerate(rel.names) if op_of(n) in ("het", "grad")]
    if not idx:
        return None
    i = idx[0]
    op = op_of(rel.names[i])
    cands, tried, info = matched_programmes(ctx, rel, i)
    m49, r49 = info.get("median_abs_dmoran_nearest49", 9.0), info.get("median_abs_dr_nearest49", 9.0)
    if len(cands) < 49 or m49 > 0.15 or r49 > 0.15:
        return {"status": "not_assessed", "reason": f"matching quality poor or too few matched programmes ({len(cands)} eligible); "
                f"nearest-49 median |dMoran| {m49:.2f}, median |dr| {r49:.2f} (limit 0.15)", **info}
    Zn = np.repeat(rel.Z[:, None, :], len(cands), axis=1).astype(np.float32)
    v0 = ctx.col(body_of(rel.names[i]))
    pix = [np.flatnonzero(ctx.pat == p) for p in np.unique(ctx.pat)]
    for d, v in enumerate(cands):
        w = np.empty_like(v)                      # give the random programme the variable per-patient mean and SD
        for ix in pix:
            w[ix] = v[ix] * v0[ix].std() + v0[ix].mean()
        col = ctx.apply_op(op, w)[rel.ok]
        Zn[:, d, i] = _within_std(col, rel.pat).ravel()
    contrib, sst = fit_gain(Zn, rel.ys, rel.pat, C)
    return {"null_gains": (contrib.sum(0) / sst.sum()).tolist(), "n_draws": len(cands), "n_tried": tried, **info}


def het_null_control(ctx):
    """Positive control for the matched-null machinery: a het relation on hypoxia that must be rejected."""
    rel = Relation(ctx, "pc_het", ["het:programmes:hallmark:HALLMARK_HYPOXIA"], "het:programme:mp:Hypoxia",
                   "heterogeneity_gradient", [])
    c, sst = gain_over(rel, rel.D)
    obs = float(c.sum() / sst.sum())
    nul = het_matched_null(ctx, rel, rel.D)
    if nul is None or "null_gains" not in nul:
        return {"passed": False, "reason": str(nul)}
    g = np.array(nul["null_gains"])
    p = float((1 + np.sum(g >= obs)) / (len(g) + 1))
    return {"observed": obs, "null_mean": float(g.mean()), "p": p, "passed": bool(p <= 0.05 and obs > 0)}


def comp_null_control(ctx):
    """Positive control: own-spot hypoxia recovery must reject.  Negative control: an own-spot immune marker programme
    predicted from compartment (CellViT inflammatory) fractions, a pure-composition relation that must NOT reject.
    The null is informative only if both hold."""
    rel = Relation(ctx, "pc_comp", ["programmes:hallmark:HALLMARK_HYPOXIA"], "programme:mp:Hypoxia", "", [])
    C = np.column_stack([rel.D, rel.COMP])
    c, sst = gain_over(rel, C)
    obs = float(c.sum() / sst.sum())
    g = comp_target_null(rel.ys, rel.pat, rel.Z, C, composition_strata(ctx, rel))
    p = float((1 + np.sum(g >= obs)) / (len(g) + 1))
    y = ctx.col("programme:mp:IG")
    zf = np.column_stack([ctx.cellvit[:, 1], ctx.nbmean(ctx.cellvit[:, 1])])

    class N:
        pass
    n = N()
    n.ok = np.isfinite(y)
    n.pat = ctx.pat[n.ok]
    n.codes = _codes(n.pat)
    n.ys = _within_std(y[n.ok], n.pat).ravel()
    n.Z = _within_std(zf[n.ok], n.pat)
    n.D = _within_std(ctx.depth[n.ok], n.pat)
    n.COMP = _within_std(ctx.composition_block()[0][n.ok], n.pat)
    Cn = np.column_stack([n.D, n.COMP])
    cn, sstn = gain_over(n, Cn)
    obsn = float(cn.sum() / sstn.sum())
    gn = comp_target_null(n.ys, n.pat, n.Z, Cn, composition_strata(ctx, n))
    pn = float((1 + np.sum(gn >= obsn)) / (len(gn) + 1))
    informative = bool(p <= 0.05 and obs > 0 and pn > 0.05)
    return {"positive": {"observed": obs, "null_mean": float(g.mean()), "null_p95": float(np.quantile(g, 0.95)), "p": p},
            "negative": {"observed": obsn, "null_mean": float(gn.mean()), "null_p95": float(np.quantile(gn, 0.95)), "p": pn},
            "informative": informative, "n_null": len(g),
            "passed": bool(p <= 0.05 and obs > 0)}


# ---------------------------------------------------------------- per-relation battery
def alt_compartment_gain(ctx, rel, C):
    """Segmentation sensitivity: recompute compartment operators under alternative CellViT dominance thresholds
    (epithelial fraction cut 0.4 / 0.6 around the registered label) and re-score the relation."""
    out = {}
    ops = [(i, op_of(n)) for i, n in enumerate(rel.names) if op_of(n) in OPS_POOLED]
    if not ops:
        return None
    from dbench.downstream.values import compartment_mean, graph_rings
    rings = graph_rings(ctx.nb6, 3)
    for cut in (0.4, 0.6):
        epi = ctx.segmented & (ctx.epi_fraction >= cut)
        stro = ctx.segmented & (ctx.epi_fraction < 1 - cut)
        Z = rel.Z.copy()
        for i, op in ops:
            base = ctx.col(body_of(rel.names[i]))
            member = epi if (op.startswith("nbrE") or op.startswith("ringE")) else stro
            if op.startswith("nbr"):
                M, _ = compartment_mean(base[:, None], ctx.coh.raw.neighbours[int(op[4:])], member)
            else:
                M, _ = compartment_mean(base[:, None], rings[int(op[-1]) - 1], member)
            col = M[:, 0][rel.ok]
            fin = np.isfinite(col)
            col = np.where(fin, col, np.nanmean(col))
            Z[:, i] = _within_std(col, rel.pat).ravel()
        c, sst = gain_over(rel, C, Z)
        out[f"epithelial_fraction_cut_{cut}"] = summarise(c, sst)
    return out


def characterise_relation(ctx, rid, names, target, family, row_views, pcs, ext=None):
    t0 = time.time()
    rel = Relation(ctx, rid, names, target, family, row_views)
    rows, rec = [], {"relation_id": rid, "target": target, "candidates": names, "family": family,
                     "n_rows": rel.n_rows, "n_patients": rel.n_patients, "known_info": rel.known_info}
    C0 = rel.D
    C1 = np.column_stack([rel.D, rel.K])
    C2 = np.column_stack([rel.D, rel.K, rel.COMP])
    CT = np.column_stack([rel.D, rel.K, rel.TECH])
    C3 = np.column_stack([rel.D, rel.K, rel.COMP, rel.TECH, rel.own_nuis])
    gains = {}
    for name, C in (("d0", C0), ("d1", C1), ("d2", C2), ("tech", CT), ("d3", C3)):
        c, sst = gain_over(rel, C)
        gains[name] = (c, sst)
    sm = {k: summarise(*v) for k, v in gains.items()}
    ref = sm["d0"]["estimate"]
    d0 = sm["d0"]["estimate"]
    # ladder
    for k, lab in (("d0", "ladder_d0"), ("d1", "ladder_d1"), ("d2", "ladder_d2"), ("d3", "ladder_d3")):
        st, rs = status_of(sm[k], ref, True)
        rows.append(row(rid, lab, sm[k], st, rs, True, {"pct_retained": (sm[k]["estimate"] / d0 * 100) if d0 > 0 else None}))
    st, rs = status_of(sm["d1"], ref, True)
    rows.append(row(rid, "known_biology", sm["d1"], st, rs, True))
    st, rs = status_of(sm["tech"], ref, True)
    rows.append(row(rid, "technical", sm["tech"], st, rs, True))
    # composition: target-permutation null within patient x composition strata (199 draws); the CI carries the test
    # when the null is uninformative (positive control does not reject, or the negative control also rejects)
    strata = composition_strata(ctx, rel)
    gnull = comp_target_null(rel.ys, rel.pat, rel.Z, C2, strata)
    p_null = float((1 + np.sum(gnull >= sm["d2"]["estimate"])) / (len(gnull) + 1))
    s = dict(sm["d2"])
    informative = bool(pcs["comp_null"].get("informative"))
    if informative:
        s["p"] = p_null
    st, rs = status_of(s, ref, pcs["comp_null"]["passed"])
    rows.append(row(rid, "composition", s, st, rs, pcs["comp_null"]["passed"],
                    {"null_informative": informative, "p_target_permutation_null": p_null,
                     "null_mean": float(gnull.mean()), "null_p95": float(np.quantile(gnull, 0.95)), "n_null": len(gnull),
                     "null_reported": informative}))
    # G_MS
    for tag, Zr in (("gms_claim_matched", rel.Zc), ("gms_own_spot", rel.Zo)):
        cr, sstr = gain_over(rel, C1, Zr)
        diff = gains["d1"][0] - cr
        s = summarise(diff, gains["d1"][1])
        st, rs = status_of(s, ref, pcs["gms"]["passed"])
        rows.append(row(rid, tag, s, st, rs, pcs["gms"]["passed"],
                        {"reduced_columns": [c if isinstance(c, str) else list(c) for c in (rel.claim if tag.endswith("matched") else rel.own)],
                         "gain_full": sm["d1"]["estimate"], "gain_reduced": float(cr.sum() / sstr.sum())}))
    # heterogeneity matched null (199 draws; nearest 49 = sensitivity row)
    if any(op_of(n) in ("het", "grad") for n in rel.names):
        nul = het_matched_null(ctx, rel, C1)
        if nul and "null_gains" in nul:
            gall = np.array(nul["null_gains"])
            mo = {k: nul[k] for k in ("moran0", "abs_r0", "size", "median_abs_dmoran", "median_abs_dr", "max_abs_dmoran",
                                      "max_abs_dr", "median_abs_dmoran_nearest49", "median_abs_dr_nearest49")}
            for tag, g in (("het_matched_null", gall), ("het_matched_null_49_sensitivity", gall[:49])):
                s = dict(sm["d1"])
                s["p"] = float((1 + np.sum(g >= s["estimate"])) / (len(g) + 1))
                st, rs = status_of(s, ref, pcs["het_null"]["passed"])
                rows.append(row(rid, tag, s, st, rs, pcs["het_null"]["passed"],
                                {"null_mean": float(g.mean()), "null_p95": float(np.quantile(g, 0.95)), "n_null": len(g),
                                 "matched_on": mo}))
        elif nul:
            for tag in ("het_matched_null", "het_matched_null_49_sensitivity"):
                rows.append(row(rid, tag, {}, "not_assessed", nul["reason"], pcs["het_null"]["passed"]))
    # segmentation sensitivity
    if any(op_of(n) in OPS_POOLED for n in rel.names):
        seg = alt_compartment_gain(ctx, rel, C1)
        for k, s in seg.items():
            st, rs = status_of(s, ref, True)
            rows.append(row(rid, "segmentation_sensitivity", s, st, rs, True, {"variant": k,
                                                                                  "registered_gain": sm["d1"]["estimate"]}))
    else:
        rows.append(row(rid, "segmentation_sensitivity", {}, "not_testable",
                        "relation uses no CellViT-derived operator; composition control uses fixed CellViT output "
                        "(no alternative CellViT model on disk)", True))
    # external no-refit columns (family receipts) and the independence structure of the own-spot reduction
    internal_kb = next(r for r in rows if r["test"] == "known_biology")
    if ext and ext.get("known"):
        k = ext["known"]
        rows.append({"relation_id": rid, "test": "known_biology_external_norefit", "estimate": k["gain"], "ci": k["ci95"],
                     "p": k.get("p_boot"), "q": None, "mde": None, "positive_control_passed": True,
                     "status": "pass" if k["ci95"][0] > 0 else "fail",
                     "reason": "external pooled no-refit gain over B_known in Li 2022 + GSE250163 + Zenodo (family receipts); "
                               "not part of the family BH"})
        internal_kb["grader_reads"] = "external" if internal_kb["status"] == "not_assessed" else "internal"
    rows.append({"relation_id": rid, "test": "composition_external", "estimate": None, "ci": None, "p": None, "q": None,
                 "mde": None, "positive_control_passed": True, "status": "not_assessed",
                 "reason": "CellViT output exists only for the discovery cohort and the 3-patient Zenodo cohort; Li 2022 and "
                           "GSE250163 have no segmentation, so external composition control is not computable"})
    if any(op_of(n) in ("het", "grad") for n in rel.names):
        roles = {"gms_claim_matched": "decisive", "known_biology": "consistency", "composition": "consistency"}
        for r in rows:
            if r["test"] in roles or r["test"] == "gms_own_spot":
                r["role"] = roles.get(r["test"], "same_as_gms_claim_matched")
                r["evidence_group"] = "own_spot_programme_reduction"
                r["independent_of_group"] = False
                r["note"] = ("het/grad: known-biology, composition and G_MS share the own-spot programme reduction; count one "
                             "decisive test (G_MS claim-matched) plus two consistency checks")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["gains"] = {k: v for k, v in sm.items()}
    return rec, rows, rel


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r = np.empty_like(q)
    r[o] = np.minimum(q, 1)
    return r


def finalise(rows):
    """BH q per test across relations; a row becomes pass only with q <= 0.05 (where it was pass-on-p)."""
    by = {}
    for r in rows:
        if r["p"] is not None and r["status"] in ("pass", "fail"):
            by.setdefault((r["test"], r.get("variant")), []).append(r)
    for rs in by.values():
        q = bh([r["p"] for r in rs])
        for r, qq in zip(rs, q):
            r["q"] = float(qq)
            if r["status"] == "pass" and qq > 0.05:
                r["status"], r["reason"] = "fail", "raw p passes, BH q > 0.05 across relations"
    return rows


def relation_list(stage):
    """Distinct relations (same target, Jaccard >= 0.5 of candidate sets) per stage, best member = largest nuisance gain."""
    d = json.loads((Path("runs/confirmation/v08_external_family_bh_v01/family_bh_masked.json")).read_text())
    proto = json.loads((Path(RUN) / "campaign_manifest.json").read_text())["protocol"]["discovery"]["relationship_families"]
    tab = d["confirmatory"]
    if stage == "1":
        pool = [r for r in tab if r.get("replicated")]
    elif stage == "2":
        pool = [r for r in tab if r["status"] == "tested" and not r["replicated"]]
    elif stage == "3":
        pool = [r for r in d["exploratory_proposed_only"]["table"] if r["status"] == "tested"]
    else:
        raise ValueError(stage)
    parent = list(range(len(pool)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            if pool[a]["target"] != pool[b]["target"]:
                continue
            sa, sb = set(pool[a]["candidates"]), set(pool[b]["candidates"])
            if len(sa & sb) / len(sa | sb) >= 0.5:
                parent[find(a)] = find(b)
    groups = {}
    for i, r in enumerate(pool):
        groups.setdefault(find(i), []).append(r)
    out = []
    for g in groups.values():
        best = max(g, key=lambda r: r["gain"]["nuis"]["gain"])
        views = [v for v in proto[best["family"]]["candidate_views"] if v in ("compartment_neighbours", "rings", "heterogeneity_gradient")]
        out.append({"relation_id": best["candidate_id"], "names": best["candidates"], "target": best["target"],
                    "family": best["family"], "row_views": views, "n_members": len(g),
                    "members": [r["candidate_id"] for r in g], "external_gain_nuis": best["gain"]["nuis"], "ext": best["gain"],
                    "external_direction": best["direction_per_cohort"],
                    "external_q": best["family_bh_q"], "stage": stage})
    return sorted(out, key=lambda x: -x["n_members"])


def aggregate():
    """Per-test JSON + markdown across every relation scored so far (BH q across relations within a test)."""
    rows = [json.loads(ln) for ln in (OUT / "evidence_rows.jsonl").read_text().splitlines() if ln.strip()]
    rows = finalise(rows)
    (OUT / "evidence_rows_final.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tests = sorted({r["test"] for r in rows})
    for t in tests:
        rs = [r for r in rows if r["test"] == t]
        (OUT / f"test_{t}.json").write_text(json.dumps(rs, indent=1))
        L = [f"## {t}", "", "| relation | stage | members | estimate [95% CI] | p | q | MDE | pos. control | status | reason |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            ci = ("[" + ", ".join("" if x is None else f"{x:.3f}" for x in r["ci"]) + "]") if r.get("ci") else ""
            est = f"{r['estimate']:.3f}" if r.get("estimate") is not None else ""
            f = lambda x: "" if x is None else f"{x:.3g}"  # noqa: E731
            L.append(f"| {r['relation_id'].split('-multiscale-20261026-')[-1]} | {r.get('stage')} | {r.get('n_members')} | "
                     f"{est} {ci} | {f(r['p'])} | {f(r['q'])} | {f(r['mde'])} | {r['positive_control_passed']} | "
                     f"{r['status']} | {r['reason']} |")
        (OUT / f"test_{t}.md").write_text("\n".join(L) + "\n")
    import collections
    print(json.dumps({t: dict(collections.Counter(r["status"] for r in rows if r["test"] == t)) for t in tests}))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "3"])
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        return aggregate()
    (OUT / "relations").mkdir(exist_ok=True)
    ctx = Ctx()
    pc_path = OUT / "positive_controls.json"
    if pc_path.exists():
        pcs = json.loads(pc_path.read_text())
    else:
        pcs = positive_controls(ctx)
        pcs["het_null"] = het_null_control(ctx)
        pcs["comp_null"] = comp_null_control(ctx)
        pc_path.write_text(json.dumps(pcs, indent=1))
    print(json.dumps({k: v.get("passed") for k, v in pcs.items()}), flush=True)
    if not all(v.get("passed") for v in pcs.values()):
        raise SystemExit("positive control failed; fix the pipeline before scoring any relation")
    rels = relation_list(args.stage)[: args.limit]
    ev = OUT / "evidence_rows.jsonl"
    for r in rels:
        done = OUT / "relations" / f"{r['relation_id']}.json"
        if done.exists():
            continue
        try:
            rec, rows, _ = characterise_relation(ctx, r["relation_id"], r["names"], r["target"], r["family"],
                                                 r["row_views"], pcs, ext=r.get("ext"))
        except Exception as e:                                   # never let one relation truncate the list
            rec = {"relation_id": r["relation_id"], "error": f"{type(e).__name__}: {e}"}
            rows = [{"relation_id": r["relation_id"], "test": "all", "estimate": None, "ci": None, "p": None, "q": None,
                     "mde": None, "positive_control_passed": True, "status": "not_testable", "reason": rec["error"]}]
        for x in rows:
            x["stage"], x["n_members"] = args.stage, r["n_members"]
        rec.update({k: r[k] for k in ("n_members", "members", "stage", "external_gain_nuis", "external_q", "ext", "external_direction")})
        done.write_text(json.dumps(rec, indent=1))
        with open(ev, "a") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        print(r["relation_id"], rec.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
