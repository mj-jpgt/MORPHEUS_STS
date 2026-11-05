"""Discovery-cohort arrays and relation freezing for findings that arrive without frozen parameters
(legacy kidney relations, the dry run). A learned-search registry entry that already carries frozen
parameters is read with ``FrozenRelation.from_json`` instead and never passes through here."""
from __future__ import annotations

import numpy as np

from .relation import (dominant_term, fit_baseline, fit_decomposition, fit_relation, standardise_within)


# Blind candidates in a registered family share the discovery cohort and
# programme catalogue.  These inputs are immutable: caching them avoids a
# repeated multi-gigabyte reload per frozen candidate without changing any
# relation fit, split, or statistic.
_DISCOVERY_COHORT = None
_PROGRAMME_SETS = None


def discovery_arrays(candidates, target, catalogue_sets: dict | None = None, residual_ctx=None,
                     operator_views=None) -> dict:
    from morpheus.tier2.substrates.hest_multi import _programme_sets

    from dbench.scripts.external_replication import load_discovery
    from .replication_v03 import known_columns
    from .values import cohort_values, parse_candidate

    global _DISCOVERY_COHORT, _PROGRAMME_SETS
    if _DISCOVERY_COHORT is None:
        _DISCOVERY_COHORT = load_discovery()
    if _PROGRAMME_SETS is None:
        _PROGRAMME_SETS = _programme_sets()
    coh, sets = _DISCOVERY_COHORT, _PROGRAMME_SETS
    Z = np.column_stack([cohort_values(coh, c, sets, coh.panel, residual_ctx) for c in candidates])
    y = cohort_values(coh, target, sets, coh.panel, residual_ctx)
    own = {parse_candidate(c).get("set") for c in candidates}
    K, kn = known_columns(coh, parse_candidate(target).get("set"), own, catalogue_sets)
    mask = None
    if operator_views:
        from .values import discovery_row_mask
        mask = discovery_row_mask(coh, operator_views, sets, coh.panel, residual_ctx)
    return {"cohort": coh, "Z": Z, "y": y, "depth": np.asarray(coh.raw.depth, float), "patient": coh.patient,
            "section": coh.section, "known": K, "known_names": kn, "row_mask": mask}


def apply_row_mask(d: dict) -> dict:
    """Restrict every row-aligned array to the rows registered discovery kept (no-op without a mask)."""
    m = d.get("row_mask")
    if m is None:
        return d
    out = dict(d)
    for k in ("Z", "y", "depth", "patient", "section", "known"):
        if out.get(k) is not None:
            out[k] = np.asarray(out[k])[m]
    out["n_rows_kept"], out["n_rows_total"] = int(m.sum()), int(len(m))
    return out


def freeze_finding(finding_id, candidates, target, d: dict, meta: dict | None = None):
    """Fit f_S under B_nuis (gating) and under B_known, the B_decomp partition (|S| >= 2) and the dominant
    term on ALL discovery patients, and freeze them."""
    from .replication_v03 import FrozenFinding

    pat = np.asarray(d["patient"]).astype(str)
    Zs, _ = standardise_within(d["Z"], pat)
    Zs = Zs.reshape(len(pat), -1)
    ys, _ = standardise_within(d["y"], pat)
    ys = ys.ravel()
    dep, _ = standardise_within(np.log1p(d["depth"]), pat)
    dep = dep.reshape(-1, 1)
    rn = ys - fit_baseline(dep, ys, pat, ["log1p_depth"]).predict(dep)
    rel_n = fit_relation(Zs, rn, pat, list(candidates), target, "nuis")
    rel_k = None
    if d.get("known") is not None:
        K, _ = standardise_within(d["known"], pat)
        K = np.column_stack([dep, K.reshape(len(pat), -1)])
        rk = ys - fit_baseline(K, ys, pat, ["log1p_depth"] + list(d["known_names"])).predict(K)
        rel_k = fit_relation(Zs, rk, pat, list(candidates), target, "known")
    rel_d = fit_decomposition(Zs, rn, pat, list(candidates), target, "nuis") if Zs.shape[1] >= 2 else None
    dom = dominant_term(rel_n, Zs, rn)
    ff = FrozenFinding(finding_id, rel_n, rel_k, rel_d, dom, dict(meta or {}))
    return ff, {"Zs": Zs, "ys": ys, "r_nuis": rn, "patient": pat}
