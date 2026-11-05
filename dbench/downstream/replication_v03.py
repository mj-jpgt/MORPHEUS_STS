"""External spatial replication under ``dbench/protocols/ccrcc_external_replication_v03.json``
(downstream spec Section 2). Runs beside Agent E's v02 runner (``dbench/scripts/external_replication.py``),
whose semantics are untouched; this module only imports its cohort loaders.

Per finding a rule file is written and hashed before any external value is read (2.2). The frozen
relation is then applied WITHOUT refitting (2.3): candidates and target are within-patient standardised
on the external cohort, the nuisance model (and, for the reported B_known, the known-biology baseline)
is fitted on the external cohort by patient cross-fitting, and the frozen f_S is added to the baseline
prediction. Success (all required): pooled patient-bootstrap 95% interval of
G(S, y, B_nuis, L_ext) excluding zero, dominant-term sign agreement in >= 60% of external patients,
BH q <= 0.05 within the family. B_known and B_decomp gains and the Moran-matched null are reported and
do not gate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import locks
from .relation import (FrozenRelation, crossfit_baseline, per_patient_partial_signs, standardise_within)

PROTOCOL = "dbench/protocols/ccrcc_external_replication_v03.json"
PROTOCOL_HASH = "dbench/protocols/ccrcc_external_replication_v03.sha256"
N_BOOT = 2000
BOOT_SEED = 20261021
SIGN_FRACTION = 0.60
Q_LEVEL = 0.05
MIN_PATIENT_ROWS = 10


def verify_protocol(path: str = PROTOCOL, hash_file: str = PROTOCOL_HASH) -> dict:
    doc = json.loads(Path(path).read_text())
    rec = Path(hash_file).read_text().split()[0]
    got = locks.canonical_hash(doc)
    if got != rec:
        raise SystemExit(f"v03 protocol canonical hash {got} != recorded {rec}: refusing to read external values")
    return {"path": path, "canonical_sha256": got, "raw_sha256": locks.sha256_file(path), "doc": doc}


# ----------------------------------------------------------------------------- statistics
def unit_stats(y, base, full, units) -> tuple[np.ndarray, np.ndarray]:
    keys = np.unique(units)
    S = np.zeros((len(keys), 5))
    for i, u in enumerate(keys):
        m = units == u
        S[i] = (m.sum(), y[m].sum(), (y[m] ** 2).sum(), ((y[m] - base[m]) ** 2).sum(), ((y[m] - full[m]) ** 2).sum())
    return keys, S


def _r2(t, col):
    sst = t[..., 2] - t[..., 1] ** 2 / t[..., 0]
    return np.where(sst > 1e-15, 1 - t[..., col] / np.where(sst > 1e-15, sst, 1), 0.0)


def pooled_gain(S) -> float:
    t = S.sum(0)
    return float(_r2(t, 4) - _r2(t, 3))


def bootstrap(S, strata, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Patient bootstrap (resampling within cohort strata): percentile 95% CI and one-sided p
    = (1 + #{draws <= 0}) / (B + 1)."""
    rng = np.random.default_rng(seed)
    n = len(S)
    strata = np.asarray(strata)
    gains = np.empty(n_boot)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for b in range(n_boot):
        w = np.zeros(n)
        for ix in groups:
            w += np.bincount(ix[rng.integers(0, len(ix), len(ix))], minlength=n)
        t = w @ S
        gains[b] = _r2(t, 4) - _r2(t, 3)
    return {"ci": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
            "p_boot": float((1 + np.sum(gains <= 0)) / (n_boot + 1)), "n_boot": int(n_boot)}


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# ----------------------------------------------------------------------------- arrays
@dataclass
class ExternalArrays:
    """One external cohort, untransformed values (the transformation is applied here, identically)."""
    key: str
    Z: np.ndarray                     # (n, k) raw candidate values
    y: np.ndarray                     # (n,) raw target
    depth: np.ndarray                 # (n,) total counts
    patient: np.ndarray
    known: np.ndarray | None = None   # (n, p) raw known-biology covariates (without depth)
    known_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class FrozenFinding:
    finding_id: str
    rel_nuis: FrozenRelation
    rel_known: FrozenRelation | None
    rel_decomp: FrozenRelation | None
    dominant: dict                    # relation.dominant_term on discovery
    meta: dict = field(default_factory=dict)


def transform_cohort(a: ExternalArrays) -> dict:
    """Rule 2.3: within-patient standardisation on the external cohort, baselines cross-fitted on it."""
    pat = np.asarray(a.patient).astype(str)
    Zs, _ = standardise_within(a.Z, pat)
    Zs = Zs.reshape(len(a.y), -1)
    ys, _ = standardise_within(a.y, pat)
    ys = ys.ravel()
    dep, _ = standardise_within(np.log1p(np.asarray(a.depth, float)), pat)
    base_n = crossfit_baseline(dep.reshape(-1, 1), ys, pat, ["log1p_depth"])
    out = {"Z": Zs, "y": ys, "patient": pat, "base_nuis": base_n}
    if a.known is not None and a.known.shape[1]:
        K, _ = standardise_within(a.known, pat)
        K = np.column_stack([dep, K.reshape(len(ys), -1)])
        out["base_known"] = crossfit_baseline(K, ys, pat, ["log1p_depth"] + list(a.known_names))
    return out


def evaluate_finding(ff: FrozenFinding, cohorts: Sequence[ExternalArrays], n_boot=N_BOOT) -> dict:
    """Pooled external gains (B_nuis gating; B_known, B_decomp reported) and direction agreement."""
    per_cohort, S_all, strata_all = {}, {"nuis": [], "known": [], "decomp": []}, []
    signs_all = {}
    for a in cohorts:
        t = transform_cohort(a)
        units = np.array([f"{a.key}|{p}" for p in t["patient"]])
        fS = ff.rel_nuis.predict(t["Z"])
        keys, Sn = unit_stats(t["y"], t["base_nuis"], t["base_nuis"] + fS, units)
        rec = {"n_rows": int(len(t["y"])), "n_patients": int(len(keys)), "gain_nuis": pooled_gain(Sn)}
        S_all["nuis"].append(Sn)
        strata_all += [a.key] * len(keys)
        if ff.rel_known is not None and "base_known" in t:
            _, Sk = unit_stats(t["y"], t["base_known"], t["base_known"] + ff.rel_known.predict(t["Z"]), units)
            rec["gain_known"] = pooled_gain(Sk)
            S_all["known"].append(Sk)
        if ff.rel_decomp is not None:
            _, Sd = unit_stats(t["y"], t["base_nuis"] + ff.rel_decomp.predict(t["Z"]), t["base_nuis"] + fS, units)
            rec["gain_decomp"] = pooled_gain(Sd)
            S_all["decomp"].append(Sd)
        r = t["y"] - t["base_nuis"]
        sg = per_patient_partial_signs(t["Z"], r, t["patient"], ff.dominant["index"], MIN_PATIENT_ROWS)
        rec["patient_signs"] = sg
        rec["direction_agreement"] = (float(np.mean([s == ff.dominant["sign"] for s in sg.values()]))
                                      if sg else None)
        signs_all.update({f"{a.key}|{p}": s for p, s in sg.items()})
        per_cohort[a.key] = rec
    out = {"per_cohort": per_cohort, "pooled": {}}
    if not S_all["nuis"]:
        out["pooled"] = {"status": "not testable in any external cohort"}
        return out
    strata = np.array(strata_all)
    for base, Ss in S_all.items():
        if not Ss or len(Ss) != len(S_all["nuis"]):
            out["pooled"][base] = None
            continue
        S = np.vstack(Ss)
        out["pooled"][base] = {"gain": pooled_gain(S), **bootstrap(S, strata, n_boot)}
    agree = [s == ff.dominant["sign"] for s in signs_all.values()]
    out["direction"] = {"dominant_candidate": ff.dominant["candidate"], "discovery_sign": ff.dominant["sign"],
                        "n_patients": len(agree), "fraction_agree": float(np.mean(agree)) if agree else None}
    return out


def decide(results: dict) -> dict:
    """BH within the family on the pooled B_nuis bootstrap p, then the declared success rule."""
    ids = [f for f, r in results.items() if r["pooled"].get("nuis")]
    q = bh([results[f]["pooled"]["nuis"]["p_boot"] for f in ids])
    for f, qq in zip(ids, q):
        r = results[f]
        g = r["pooled"]["nuis"]
        fa = r["direction"]["fraction_agree"]
        r["bh_q"] = float(qq)
        r["criteria"] = {"ci_excludes_zero": bool(g["ci"][0] > 0),
                         "direction_agreement_ge_0.60": bool(fa is not None and fa >= SIGN_FRACTION),
                         "bh_q_le_0.05": bool(qq <= Q_LEVEL)}
        r["externally_replicated"] = all(r["criteria"].values())
    for f, r in results.items():
        if f not in ids:
            r.update(bh_q=None, criteria=None, externally_replicated=False)
    return results


# ----------------------------------------------------------------------------- rule file
def rule_file(ff: FrozenFinding, *, protocol: dict, cohorts: Sequence[str], family: Sequence[str],
              transfer_notes: dict) -> dict:
    return {"schema": "morpheus_replication_rule/v3", "protocol_id": protocol["doc"]["protocol_id"],
            "protocol_canonical_sha256": protocol["canonical_sha256"], "finding_id": ff.finding_id,
            "finding_hash": ff.meta.get("finding_hash"), "target": ff.rel_nuis.target,
            "candidates": ff.rel_nuis.candidates,
            "relation_params": {"nuis": ff.rel_nuis.to_json(),
                                "known": ff.rel_known.to_json() if ff.rel_known else None,
                                "decomp": ff.rel_decomp.to_json() if ff.rel_decomp else None},
            "relation_params_hash": ff.rel_nuis.params_hash(), "direction": ff.dominant,
            "baseline": "B_nuis (gating); B_known, B_decomp reported", "external_cohorts": list(cohorts),
            "platform_transformation": protocol["doc"]["platform_transformation"],
            "platform_notes": transfer_notes, "success_criterion": protocol["doc"]["success_criterion"],
            "multiplicity_family": sorted(family)}


# ----------------------------------------------------------------------------- real cohorts
def known_columns(coh, target_set: str | None, own_sets: set, catalogue: dict | None) -> tuple[np.ndarray, list]:
    """Known-biology covariates of an external cohort: 50 Hallmark + 41 Gavish own-spot scores (minus the
    target's set and the candidates' own sets) and the B_known catalogue gene-set scores."""
    prog = np.asarray(coh.raw.programmes, float)
    names = [str(n) for n in coh.raw.programme_names]
    keep = [i for i, n in enumerate(names) if n != target_set and n not in own_sets and coh.raw.programme_matched[i] > 0]
    cols, cn = [prog[:, keep]], [names[i] for i in keep]
    for nm, genes in (catalogue or {}).items():
        idx = [coh.panel[g] for g in genes if g in coh.panel]
        if idx:
            cols.append(np.asarray(coh.logcpm[:, idx].mean(axis=1)).reshape(-1, 1))
            cn.append(f"catalogue:{nm}")
    return np.column_stack(cols), cn


def load_external(ff: FrozenFinding, source_keys: Sequence[str], catalogue: dict | None = None,
                  residual_ctx: dict | None = None) -> tuple[list, dict]:
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.scripts.external_replication import SOURCES, load_cohort, load_discovery
    from .values import NotTestable, cohort_values, parse_candidate

    sets = _programme_sets()
    disc_panel = load_discovery().panel
    arrays, notes = [], {}
    tc = parse_candidate(ff.rel_nuis.target)
    own_sets = {parse_candidate(c).get("set") for c in ff.rel_nuis.candidates}
    for key in source_keys:
        data_dir, chem = SOURCES[key]
        coh, _ = load_cohort(key, data_dir, chem)
        try:
            Z = np.column_stack([cohort_values(coh, c, sets, disc_panel, residual_ctx) for c in ff.rel_nuis.candidates])
            y = cohort_values(coh, ff.rel_nuis.target, sets, disc_panel, residual_ctx)
        except NotTestable as e:
            notes[key] = {"testable": False, "reason": str(e)}
            continue
        K, kn = known_columns(coh, tc.get("set"), own_sets, catalogue)
        depth, patient = np.asarray(coh.raw.depth, float), np.asarray(coh.patient)
        # Registered discovery dropped, never imputed, rows where a spatial operator is undefined (no neighbour of the
        # class); the same rule applies to the candidate operators on an external section.  No-op without NaN.
        finite = np.isfinite(Z).all(1) & np.isfinite(y)
        n_dropped = int((~finite).sum())
        if n_dropped:
            Z, y, depth, patient = Z[finite], y[finite], depth[finite], patient[finite]
            K = None if K is None else np.asarray(K)[finite]
        arrays.append(ExternalArrays(key, Z, y, depth, patient, K, kn,
                                     {"chemistry": chem, "n_sections": int(len(set(coh.section))),
                                      "rows_undefined_operator_dropped": n_dropped}))
        notes[key] = {"testable": True, "chemistry": chem, "n_patients": int(len(set(coh.patient))),
                      "n_spots": int(coh.raw.n), "n_spots_dropped_undefined_operator": n_dropped,
                      "neighbours": "same neighbour counts (6, 18) within section as discovery, not the same radius"}
    return arrays, notes


def run_family(findings: Sequence[FrozenFinding], root: str, source_keys: Sequence[str], *, catalogue=None,
               residual_ctx=None, n_boot=N_BOOT, allow_violation=False) -> dict:
    """Write and lock every rule, then open the external data, evaluate, decide, record."""
    prot = verify_protocol()
    family = [f.finding_id for f in findings]
    rules = {}
    for ff in findings:
        rule = rule_file(ff, protocol=prot, cohorts=source_keys, family=family,
                         transfer_notes={"declared_before_external_values": True})
        rules[ff.finding_id] = rule
        locks.write_lock(root, ff.finding_id, "external_replication",
                         inputs={"rule_hash": locks.canonical_hash(rule), "protocol": PROTOCOL},
                         declared_analysis={"rule": rule}, allow_violation=allow_violation)
    results = {}
    for ff in findings:
        locks.require_lock(root, ff.finding_id, "external_replication")
        arrays, notes = load_external(ff, source_keys, catalogue, residual_ctx)
        res = evaluate_finding(ff, arrays, n_boot) if arrays else {"per_cohort": {}, "pooled": {},
                                                                  "direction": {"fraction_agree": None}}
        res["testability"] = notes
        res["rule_hash"] = locks.canonical_hash(rules[ff.finding_id])
        results[ff.finding_id] = res
    decide(results)
    for fid, res in results.items():
        locks.write_record(root, fid, "external_replication", res)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        p = verify_protocol()
        print(json.dumps({k: v for k, v in p.items() if k != "doc"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
