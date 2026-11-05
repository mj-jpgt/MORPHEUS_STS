"""S11: P5 addendum. No-refit gain of the frozen f_S over the KNOWN-BIOLOGY baseline B_known in the target cohort, as in ccrcc_external_replication_v01/v03:
B_known = log1p depth + the cohort's own-spot Hallmark and Gavish programme scores (response-free columns, target programme and candidates' own sets removed),
within-patient standardised, cross-fitted ridge; frozen f_S added to the baseline prediction; patient bootstrap; Moran-matched null p95 (same engine as s10).
usage: python -m ... COHORT REL1,REL2,..."""
import sys, json, time
import numpy as np
from dbench.translation_v10 import common as C, values_v06_eaca09b as VV
from dbench.translation_v10 import s10_p5 as P
from dbench.downstream.replication_v03 import ExternalArrays, transform_cohort, unit_stats, pooled_gain, bootstrap, known_columns
from dbench.downstream.relation import standardise_within
from dbench.downstream.discovery_any import cohort as any_cohort
from dbench.scripts.external_replication import matched_programmes, load_discovery
from morpheus.tier2.substrates.hest_multi import _programme_sets

N_NULL = 99


def base_set(name):
    c = VV.parse_candidate(name)
    if c["kind"] == "operator":
        c = c["base"]
    return c.get("set")


def main():
    key = sys.argv[1]
    names = sys.argv[2].split(",")
    sets = _programme_sets()
    panel = load_discovery().panel
    coh = P.load_target(key)
    out = {}
    cache = {}
    sid2rid = {C.short_id(r): r for r in C.v08_ids()}
    for nm in names:
        rid = C.GROUPS["best"][nm] if nm in C.GROUPS["best"] else sid2rid[nm]
        rel = P.Rel(C.FrozenV08(rid))
        t0 = time.time()
        try:
            Z = np.column_stack([np.asarray(VV.cohort_values(coh, c, sets, panel), float) for c in rel.candidates])
            y = np.asarray(VV.cohort_values(coh, rel.target, sets, panel), float)
        except VV.NotTestable as e:
            out[nm] = {"status": "not_assessed", "reason": f"untransferable: {str(e)[:200]}"}
            continue
        ok = np.isfinite(Z).all(1) & np.isfinite(y)
        tset = base_set(rel.target)
        own = {base_set(c) for c in rel.candidates} - {None}
        try:
            K, kn = known_columns(coh, tset, own, None)
        except Exception as e:
            out[nm] = {"status": "not_assessed", "reason": f"B_known cannot be rebuilt in this cohort: {type(e).__name__}: {str(e)[:200]}"}
            continue
        ok &= np.isfinite(K).all(1)
        pat = np.asarray(coh.patient)[ok]
        arr = ExternalArrays(key, Z[ok], y[ok], np.asarray(coh.raw.depth, float)[ok], pat, K[ok], kn)
        t = transform_cohort(arr)
        if "base_known" not in t:
            out[nm] = {"status": "not_assessed", "reason": "B_known columns empty after exclusions"}
            continue
        units = np.array([f"{key}|{p}" for p in t["patient"]])
        fS = rel.predict(t["Z"])
        keys, S = unit_stats(t["y"], t["base_known"], t["base_known"] + fS, units)
        gain = pooled_gain(S)
        bs = bootstrap(S, [key] * len(keys))
        r = {"status": "ok", "gain_over_B_known": gain, "ci_low": bs["ci"][0], "ci_high": bs["ci"][1], "p_boot": bs["p_boot"], "n_patients": int(len(keys)),
             "n_rows": int(ok.sum()), "n_known_columns": len(kn) + 1}
        # gain over the nuisance baseline recomputed on the same rows, for reference
        keys2, S2 = unit_stats(t["y"], t["base_nuis"], t["base_nuis"] + fS, units)
        r["gain_over_B_nuis_same_rows"] = pooled_gain(S2)
        # matched null with B_known fixed
        draws = {}
        okidx = np.flatnonzero(ok)
        for j, c in enumerate(rel.candidates):
            op, base, genes = P.constituent(c, sets, panel)
            ck = (key, str(sorted(genes)[:5]), tuple(genes))
            if ck not in cache:
                v = {"name": base.get("name", "c"), "kind": "gene" if base["kind"] == "gene" else "programme", "genes": [g for g in genes if g in coh.panel], "nbr": 0}
                cache[ck] = matched_programmes(coh, v, seed=C.SEED + j, n_accept=N_NULL, max_proposals=12000, tol_wide=0.15)
            m = cache[ck]
            if m["values"] is None:
                draws = None
                r["null"] = {"status": "not_computable", "reason": m.get("status")}
                break
            draws[j] = (op, m["values"])
        if draws is not None:
            gains = []
            for d in range(N_NULL):
                Zd = np.array(Z[ok], float, copy=True)
                for j, (op, vals) in draws.items():
                    Zd[:, j] = P.apply_op(coh, op, vals[:, d])[okidx]
                if not np.isfinite(Zd).all():
                    continue
                Zs = standardise_within(Zd, t["patient"])[0].reshape(len(t["y"]), -1)
                _, Sd = unit_stats(t["y"], t["base_known"], t["base_known"] + rel.predict(Zs), units)
                gains.append(pooled_gain(Sd))
            g = np.asarray(gains)
            r["null"] = {"status": "ok", "n_draws": int(len(g)), "null_q95": float(np.quantile(g, 0.95)), "null_mean": float(g.mean()), "p_matched": float((1 + (g >= gain).sum()) / (len(g) + 1))}
            r["transfers_over_B_known"] = bool(r["ci_low"] > 0 and gain > r["null"]["null_q95"])
        out[nm] = r
        C.log(nm, key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("gain_over_B_known", "ci_low", "ci_high", "gain_over_B_nuis_same_rows", "transfers_over_B_known")}, round(time.time() - t0), "s")
        json.dump(out, open(f"{C.OUT}/p5_known_{key}.json", "w"), indent=1, default=float)
    json.dump(out, open(f"{C.OUT}/p5_known_{key}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
