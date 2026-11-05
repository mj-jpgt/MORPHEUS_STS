"""S10: P5 conserved rules. No-refit transfer of the 8 kidney relations (G1-G8 best members) + a hypoxia Hallmark positive-control relation into
Valdeolivas CRC, GSE294385 CRC metastases, Ji cSCC and ependymoma (same operator arithmetic, within-patient standardisation, v03 estimator).
v03.1 rule per cell: pooled gain over the nuisance baseline with patient-bootstrap CI excluding 0 AND observed gain above the Moran-matched null p95
(random gene programmes matched on expression decile and own-spot Moran I replace every gene-defined constituent; the spatial operator is then applied);
direction agreement >= 0.6 is reported as a third column. usage: python -m ... COHORT_KEY"""
import json, sys, time, dataclasses
import numpy as np
from dbench.translation_v10 import common as C, values_v06_eaca09b as VV
from dbench.downstream.replication_v03 import (ExternalArrays, FrozenFinding, evaluate_finding, unit_stats, pooled_gain, bootstrap, bh)
from dbench.downstream.relation import standardise_within, dominant_term, crossfit_baseline
from dbench.downstream.discovery_any import cohort as any_cohort
from dbench.scripts.external_replication import matched_programmes, load_discovery
from morpheus.tier2.substrates.hest_multi import _programme_sets

N_NULL = 99
GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
CTRL6 = ["raw_genes:VEGFA", "raw_genes:SLC2A1", "raw_genes:PGK1", "raw_genes:ENO1", "raw_genes:P4HA1", "raw_genes:NDRG1"]   # canonical HIF targets present in all five cohort panels
CONTROL = {"candidates": ["raw_genes:VEGFA", "raw_genes:SLC2A1"], "target": "programme:hallmark:HALLMARK_HYPOXIA"}   # target replaced by hyp_target (cohort-available Hallmark genes minus predictors)


class Rel:
    def __init__(self, base):
        self.base = base
        self.candidates, self.target = base.candidates, base.target

    @property
    def k(self):
        return len(self.candidates)

    def predict(self, Z):
        return self.base.predict(Z)

    def gradient(self, Z, eps=1e-3):
        Z = np.asarray(Z, float)
        G = np.zeros_like(Z)
        for j in range(Z.shape[1]):
            E = np.zeros_like(Z)
            E[:, j] = eps
            G[:, j] = (self.predict(Z + E) - self.predict(Z - E)) / (2 * eps)
        return G


def apply_op(coh, op, v):
    """The spatial operator of values.cohort_values applied to an arbitrary own-spot vector."""
    v = np.asarray(v, float)
    n = len(v)
    if op is None:
        return v
    if op.startswith("ring") and not op.startswith(("ringS", "ringE")):
        rings = VV.graph_rings(np.asarray(coh.raw.neighbours[6], dtype=np.int64), 3)
        M, _ = VV.compartment_mean(v[:, None], rings[int(op[-1]) - 1], np.ones(n, dtype=bool))
        return np.asarray(M[:, 0], float)
    if op == "het":
        _, var = VV.neighbour_moments(v[:, None], VV._adjacency(np.asarray(coh.raw.neighbours[18], dtype=np.int64), n))
        return np.asarray(var[:, 0], float)
    if op == "grad":
        wx, wy, ok = VV.gradient_operator(np.asarray(coh.raw.coords, dtype=float), np.asarray(coh.raw.neighbours[18], dtype=np.int64))
        out = np.sqrt(np.asarray(wx @ v) ** 2 + np.asarray(wy @ v) ** 2)
        out[~ok] = np.nan
        return out
    raise VV.NotTestable(f"operator {op}")


def constituent(name, sets, panel):
    c = VV.parse_candidate(name)
    op = None
    if c["kind"] == "operator":
        op, c = c["operator"], c["base"]
    genes = VV.candidate_genes(c, sets, panel)
    return op, c, genes


COMMON_HYP = None


def half_sets(sets):
    """Split-half Hallmark-hypoxia control: genes present in all five cohort panels, fixed shuffle (seed 20261102), A = first half (predictor programme), B = second half (target)."""
    common = json.load(open(f"{C.OUT}/hyp_common_genes.json"))
    rng = np.random.default_rng(20261102)
    g = list(rng.permutation(common))
    h = len(g) // 2
    return sorted(g[:h]), sorted(g[h:])


def custom_values(coh, name, A, B):
    gs = A if name == "custom:hypA" else B
    gs = [x for x in gs if x in coh.panel]
    return np.asarray(coh.logcpm[:, [coh.panel[x] for x in gs]].mean(1)).ravel(), gs


def hyp_target(coh, sets, exclude=("VEGFA", "SLC2A1")):
    """Positive-control target: Hallmark hypoxia genes available in this cohort, minus the two predictor genes (VEGFA, SLC2A1), mean log1p-CPM."""
    genes = [g for g in VV.set_genes(sets, "hallmark:HALLMARK_HYPOXIA") if g in coh.panel and g not in exclude]
    return np.asarray(coh.logcpm[:, [coh.panel[g] for g in genes]].mean(1)).ravel(), len(genes)


def _rctx():
    try:
        from dbench.translation_v10.cellvit_ctx import v08_external_cellvit_context
        return v08_external_cellvit_context(f"{C.V06}/runs/confirmation/v08_external_cellvit_zenodo_v01")
    except Exception as e:
        print("no compartment context", repr(e))
        return None


RCTX = None


def source_fixtures(sets, only_ctrl6=False):
    global RCTX
    RCTX = _rctx()
    disc = load_discovery()
    out = {}
    from dbench.downstream.blind import freeze_of_record
    todo = [(g, C.GROUPS["best"][g]) for g in GROUPS] if not C.ONLY else [(C.short_id(r), r) for r in C.v08_ids() if C.short_id(r) in C.ONLY]
    for g, rid in ([] if (only_ctrl6 or C.CTRL_MODE) else todo):
        rel = Rel(C.FrozenV08(rid))
        s = C.score_v08(rel.base, disc, sets, disc.panel, RCTX)
        ok = s["ok"]
        pat = s["patient"][ok]
        dep = standardise_within(np.log1p(np.asarray(disc.raw.depth, float))[ok], pat)[0].reshape(-1, 1)
        base = crossfit_baseline(dep, s["ys"][ok], pat, ["log1p_depth"])
        dom = dominant_term(rel, s["Zs"][ok], s["ys"][ok] - base)
        out[g] = FrozenFinding(rid, rel, None, None, dom, {"group": g})
    if C.ONLY:
        return out
    if C.CTRL_MODE == "half":
        A, B = half_sets(sets)
        Z = custom_values(disc, "custom:hypA", A, B)[0][:, None]
        y = custom_values(disc, "custom:hypB", A, B)[0]
        d = {"Z": Z, "y": y, "depth": np.asarray(disc.raw.depth, float), "patient": disc.patient, "known": None}
        ff, _ = freeze_of_record("ctrl_half", ["custom:hypA"], "custom:hypB", d, "v03_rff", None, {"commit": "w2"})
        ff.rel_nuis = Rel(ff.rel_nuis)
        return {"CTRL_HYPOXIA_HALF": ff}
    if only_ctrl6:
        y, _n = hyp_target(disc, sets, exclude=tuple(c.split(":")[1] for c in CTRL6))
        Z = np.column_stack([np.asarray(VV.cohort_values(disc, c, sets, disc.panel), float) for c in CTRL6])
        d = {"Z": Z, "y": y, "depth": np.asarray(disc.raw.depth, float), "patient": disc.patient, "known": None}
        ff, _ = freeze_of_record("ctrl_hypoxia6", CTRL6, CONTROL["target"], d, "v03_rff", None, {"commit": "w2"})
        ff.rel_nuis = Rel(ff.rel_nuis)
        return {"CTRL_HYPOXIA6": ff}
    # hypoxia Hallmark positive control: relation fitted on Meylan (same v03_rff estimator), then frozen
    Z = np.column_stack([np.asarray(VV.cohort_values(disc, c, sets, disc.panel), float) for c in CONTROL["candidates"]])
    y, _n = hyp_target(disc, sets)
    d = {"Z": Z, "y": y, "depth": np.asarray(disc.raw.depth, float), "patient": disc.patient, "known": None}
    ff, _ = freeze_of_record("ctrl_hypoxia", CONTROL["candidates"], CONTROL["target"], d, "v03_rff", None, {"commit": "w2"})
    ff.rel_nuis = Rel(ff.rel_nuis)
    out["CTRL_HYPOXIA"] = ff
    return out


def cell(ff, coh, key, sets, panel, null_cache):
    rel = ff.rel_nuis
    try:
        if rel.candidates == ["custom:hypA"]:
            A, B = half_sets(sets)
            Z = custom_values(coh, "custom:hypA", A, B)[0][:, None]
            y = custom_values(coh, "custom:hypB", A, B)[0]
            n_t = None
        else:
            Z = np.column_stack([np.asarray(VV.cohort_values(coh, c, sets, panel), float) for c in rel.candidates])
        if rel.candidates == ["custom:hypA"]:
            pass
        elif rel.target == CONTROL["target"]:
            y, n_t = hyp_target(coh, sets, exclude=tuple(c.split(":")[1] for c in rel.candidates))
        else:
            y = np.asarray(VV.cohort_values(coh, rel.target, sets, panel), float)
    except VV.NotTestable as e:
        return {"status": "not_assessed", "reason": f"untransferable: {str(e)[:260]}"}
    ok = np.isfinite(Z).all(1) & np.isfinite(y)
    if ok.sum() < 200:
        return {"status": "not_assessed", "reason": f"only {int(ok.sum())} finite rows"}
    dep, pat = np.asarray(coh.raw.depth, float)[ok], np.asarray(coh.patient)[ok]
    arr = ExternalArrays(key, Z[ok], y[ok], dep, pat)
    res = evaluate_finding(ff, [arr], n_boot=1000)
    pc = res["per_cohort"][key]
    pooled = res["pooled"]["nuis"]
    out = {"status": "ok", "n_rows": pc["n_rows"], "n_patients": pc["n_patients"], "gain": pooled["gain"], "ci_low": pooled["ci"][0], "ci_high": pooled["ci"][1],
           "p_boot": pooled["p_boot"], "direction_agreement": pc.get("direction_agreement"), "n_direction_patients": len(pc.get("patient_signs", {}))}
    # matched null
    from dbench.downstream.replication_v03 import transform_cohort
    t = transform_cohort(arr)
    from scipy.stats import spearmanr
    fS = rel.predict(t["Z"])
    resid = t["y"] - t["base_nuis"]
    out["diagnostics"] = {"spearman_fS_vs_target": float(spearmanr(fS, t["y"])[0]), "calibration_slope_resid_on_fS": float(np.polyfit(fS, resid, 1)[0]),
                          "sd_fS": float(fS.std()), "sd_resid": float(resid.std())}
    units = np.array([f"{key}|{p}" for p in t["patient"]])
    gene_cols = []
    for j, c in enumerate(rel.candidates):
        if c == "custom:hypA":
            A, B = half_sets(sets)
            gene_cols.append((j, None, {"kind": "programme", "name": c}, [x for x in A if x in coh.panel]))
            continue
        op, base, genes = constituent(c, sets, panel)
        gene_cols.append((j, op, base, genes))
    draws = {}
    for j, op, base, genes in gene_cols:
        ck = (key, base["name"] if "name" in base else str(genes[:3]), tuple(genes))
        if ck not in null_cache:
            v = {"name": base.get("name", "c"), "kind": "gene" if base["kind"] == "gene" else "programme", "genes": [g for g in genes if g in coh.panel], "nbr": 0}
            m = matched_programmes(coh, v, seed=C.SEED + j, n_accept=N_NULL, max_proposals=12000, tol_wide=0.15)
            null_cache[ck] = m
        m = null_cache[ck]
        if m["values"] is None:
            out["null"] = {"status": "not_computable", "reason": m.get("status", "too few Moran-matched programmes")}
            break
        draws[j] = (op, m["values"])
    if "null" not in out:
        gains = []
        okidx = np.flatnonzero(ok)
        for d in range(N_NULL):
            Zd = np.array(Z[ok], float, copy=True)
            for j, (op, vals) in draws.items():
                full = apply_op(coh, op, vals[:, d])
                Zd[:, j] = full[okidx]
            if not np.isfinite(Zd).all():
                continue
            Zs = standardise_within(Zd, t["patient"])[0].reshape(len(t["y"]), -1)
            _, S = unit_stats(t["y"], t["base_nuis"], t["base_nuis"] + rel.predict(Zs), units)
            gains.append(pooled_gain(S))
        gains = np.asarray(gains)
        out["null"] = {"status": "ok", "n_draws": int(len(gains)), "null_mean": float(gains.mean()), "null_q95": float(np.quantile(gains, 0.95)),
                       "p_matched": float((1 + (gains >= out["gain"]).sum()) / (len(gains) + 1)),
                       "replaced": [rel.candidates[j] for j in draws], "moran_tol_used": {str(j): null_cache[k_]["tol_used"] for j, k_ in zip(draws, [kk for kk in null_cache if kk[0] == key][-len(draws):])} if False else None, "note": "operator applied after random-programme scoring; Moran matched on own-spot value"}
        out["pass_interval"] = bool(out["ci_low"] > 0)
        out["pass_null_p95"] = bool(out["gain"] > out["null"]["null_q95"])
        out["pass_direction_ge_0.6"] = bool(out["direction_agreement"] is not None and out["direction_agreement"] >= 0.6)
        out["transfers_v03_1"] = bool(out["pass_interval"] and out["pass_null_p95"])
    return out


def load_target(key):
    if key != "gse294385":
        return any_cohort(key)
    from morpheus.tier2.substrates import hest_multi as hm
    from dbench.scripts.external_replication import Cohort
    spec = hm.study_specs()[key]
    ids = tuple(sorted(spec.sample_ids)[::3])   # memory: 348k-spot cohort; every third section (declared deviation D16)
    spec2 = dataclasses.replace(spec, sample_ids=ids, patient_of={i: spec.patient_of[i] for i in ids})
    return Cohort(key, hm.load_study(spec2), "fresh_frozen_whole_transcriptome", None, False)


import dbench.translation_v10.common as _c
_c.CTRL_MODE = None


def main():
    key = sys.argv[1]
    sets = _programme_sets()
    t0 = time.time()
    ctrl6 = len(sys.argv) > 2 and sys.argv[2] == "ctrl6"
    C.CTRL_MODE = "half" if (len(sys.argv) > 2 and sys.argv[2] == "ctrlhalf") else None
    src = source_fixtures(sets, only_ctrl6=ctrl6 or bool(C.CTRL_MODE))
    C.log("fixtures", list(src), round(time.time() - t0))
    disc_panel = load_discovery().panel
    coh = load_target(key)
    C.log("cohort", key, coh.raw.n)
    cache, res = {}, {}
    for name, ff in src.items():
        t1 = time.time()
        try:
            res[name] = cell(ff, coh, key, sets, disc_panel, cache)
        except Exception as e:
            res[name] = {"status": "not_assessed", "reason": f"{type(e).__name__}: {str(e)[:240]}"}
        r = res[name]
        C.log(name, key, r.get("status"), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("gain", "ci_low", "ci_high", "direction_agreement", "transfers_v03_1")}, round(time.time() - t1), "s")
        json.dump(res, open(f"{C.OUT}/p5{'ctrl6' if ctrl6 else ('ctrlhalf' if C.CTRL_MODE else '')}_{key}{'_D' if C.ONLY else ''}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
