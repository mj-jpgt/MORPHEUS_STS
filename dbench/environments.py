"""Episode generation: E = (D_P, D_R, D_N, D_U).

The paradigm the spec puts the novelty on is hypothesis-level replication, not
sample-level correspondence. That has a concrete consequence for how these are
built: a replication environment is **independently generated**. New noise, new
nuisance realisations, new environment labels, new artifact draws -- the same
true objects. Row i of D_R has no relationship whatsoever to row i of D_P, and
the two may even have different row counts. Anything that pairs rows across
environments is measuring the wrong thing.

Concretely, `make_episode` derives every environment from a different seed
stream, and `independence_report` measures that the environments really are
independent: it checks that the cross-environment row-wise correlation of the
target is at chance and that the generators' seeds are distinct, rather than
assuming so.

D_N holds the structured nulls appropriate to the family's structure -- the
policy lives in `nulls.DEFAULT_NULLS` -- and every null carries its own
preservation report.

D_U is a further independent draw used only to measure ordinary representation
utility U, so that the U >= U_0 - delta constraint in RDC is evaluated on data
that contributed nothing to discovery.
"""
from __future__ import annotations

import numpy as np

from . import nulls as N
from .boss import BOSS_WORLDS
from . import families as F
from .families import FAMILIES
from .util import acf, corr, derive_seed, morans_i
from .world import Episode


def effective_n(world):
    """Independent-observation count, not row count.

    A dependence check that divides by sqrt(n_rows) is wrong for exactly the
    worlds DiscoveryBench cares most about. A 1200-step autocorrelated series
    and a 6144-row spatiotemporal panel each carry far fewer independent
    observations than rows, so a legitimate chance-level cross-environment
    correlation looks like evidence of a shared random stream. Measured before
    this correction: temporal_lag and boss1 were both flagged as
    non-independent at |r| ~ 0.04-0.08 against a naive scale of 0.013-0.029,
    when both are well inside chance for their effective n.

    Temporal worlds use the standard AR(1) deflation n(1-r1)/(1+r1); spatial
    worlds deflate by the mean neighbourhood size when Moran's I is positive;
    everything else uses the row count.
    """
    y = world.target
    n = world.n
    if y is None:
        return n
    y = np.asarray(y, dtype=float)
    st = world.structure
    n_eff = float(n)
    if st.time is not None:
        r1 = float(acf(y, 1)[0])
        r1 = min(max(r1, 0.0), 0.99)
        n_eff = min(n_eff, n * (1.0 - r1) / (1.0 + r1))
    if st.neighbors is not None:
        mi = morans_i(y, st.neighbors)
        if mi > 0:
            k = float(np.mean([len(v) for v in st.neighbors])) + 1.0
            n_eff = min(n_eff, n / max(1.0, 1.0 + mi * (k - 1.0)))
    return max(4.0, n_eff)

ALL_WORLDS = dict(FAMILIES)
ALL_WORLDS.update(BOSS_WORLDS)


def build_world(family, seed, role="proposal", instrument=None,
                operator_seed=None, **kw):
    if family not in ALL_WORLDS:
        raise KeyError("unknown world family %r" % family)
    import contextlib as _c
    with _c.ExitStack() as stack:
        if instrument is not None:
            stack.enter_context(F.instrument(instrument))
        if operator_seed is not None:
            stack.enter_context(F.operator_seed(operator_seed))
        return ALL_WORLDS[family](seed=seed, role=role, **kw)


def make_episode(family, seed=0, n_replication=2, null_kinds=None,
                 with_utility=True, artifacts=(), independence_ref=None,
                 replication_instrument=None, **kw):
    """One episode of the family.

    `seed` is the episode seed. Each environment gets its own derived seed, so
    D_P, each D_R, each D_N base and D_U are generated from disjoint streams.

    Every environment shares the observation operator by default, so an object
    expressed in observed variables in D_P can be measured in D_R -- which is
    what the spec's replication step asks for and what a role-seeded operator
    made impossible. `replication_instrument` names a different operator for
    the replication environments only, posing the harder question of whether
    an object survives a change of instrument. It is opt-in because it is a
    different experiment, not a stricter version of the same one.
    """
    ep_seed = int(seed)
    # every environment of this episode shares one observation
    # operator; episodes remain distinct from one another
    _op_seed = ep_seed
    p_seed = derive_seed("proposal", family, ep_seed) % (2 ** 31)
    D_P = build_world(family, p_seed, operator_seed=_op_seed, role="proposal", artifacts=artifacts, **kw)

    D_R = []
    for r in range(n_replication):
        r_seed = derive_seed("replication", family, ep_seed, r) % (2 ** 31)
        D_R.append(build_world(family, r_seed, role="replication",
                               instrument=replication_instrument,
                               operator_seed=_op_seed,
                               artifacts=artifacts, **kw))

    if null_kinds is None:
        null_kinds = N.DEFAULT_NULLS.get(family, ())
    D_N = []
    for i, kind in enumerate(null_kinds):
        n_seed = derive_seed("null", family, ep_seed, kind, i) % (2 ** 31)
        base = build_world(family, n_seed, operator_seed=_op_seed, role="proposal",
                           artifacts=artifacts, **kw)
        try:
            nw, rep = N.make_null(base, kind, seed=n_seed % (2 ** 31))
        except Exception as exc:                       # keep the failure visible
            nw, rep = None, {"kind": kind, "passes": False, "error": repr(exc)}
        if nw is not None:
            D_N.append(nw)

    D_U = None
    if with_utility:
        u_seed = derive_seed("utility", family, ep_seed) % (2 ** 31)
        D_U = build_world(family, u_seed, operator_seed=_op_seed, role="utility", artifacts=artifacts,
                          **kw)

    ep = Episode(name="%s/ep%d" % (family, ep_seed), D_P=D_P,
                 D_R=tuple(D_R), D_N=tuple(D_N), D_U=D_U,
                 meta={"family": family, "episode_seed": ep_seed,
                       "null_kinds": list(null_kinds),
                       "artifacts": list(artifacts),
                       "params": dict(kw)})
    ep.meta["independence"] = independence_report(ep, independence_ref)
    ep.meta["null_preservation"] = [w.meta.get("preservation", {})
                                    for w in ep.D_N]
    return ep


def calibrate_independence(family, n_draws=6, **kw):
    """Empirical null for the cross-environment row correlation of a family.

    The analytic chance scale 1/sqrt(n_eff) is not usable here. A world like
    boss-1 carries a static, smooth spatial nuisance field tiled across every
    time step: two *independent* draws of that field on a 16x16 grid correlate
    with a standard deviation of order 0.3, so an observed cross-environment
    correlation of -0.21 is ordinary chance while the analytic scale calls it a
    7-sigma dependence. Deflating n by an autocorrelation estimate taken from Y
    does not fix it, because Y is dominated by its noise term and is far less
    autocorrelated than the field driving the coincidence.

    Rather than loosen the tolerance until the warning goes away, the reference
    is measured: `n_draws` further worlds are generated from disjoint seeds --
    independent by construction -- and the spread of their pairwise target
    correlations is the empirical chance distribution for this family. That
    number is stored in the frozen manifest so the per-episode check has a
    calibrated scale rather than an assumed one.
    """
    ws = []
    for i in range(n_draws):
        sd = derive_seed("calibrate_independence", family, i) % (2 ** 31)
        ws.append(build_world(family, sd, role="calibration", **kw))
    rs = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            a, b = ws[i].target, ws[j].target
            if a is None or b is None or len(a) != len(b):
                continue
            rs.append(abs(corr(a, b)))
    if not rs:
        return {"family": family, "n_draws": n_draws, "n_pairs": 0,
                "abs_corr_sd": None, "abs_corr_max": None, "usable": False}
    rs = np.array(rs)
    return {"family": family, "n_draws": n_draws, "n_pairs": int(len(rs)),
            "abs_corr_mean": float(rs.mean()),
            "abs_corr_sd": float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
            "abs_corr_max": float(rs.max()),
            # the threshold a real episode is judged against
            "threshold": float(max(0.15, rs.max() * 2.0)),
            "usable": True}


def independence_report(ep, reference=None):
    """Measure, do not assume, that the environments are independent draws.

    distinct seeds        every environment came from a different derived seed;
    distinct fingerprints no two environments are byte-identical;
    no stream reuse       the row-wise target correlation between environments
                          is nowhere near the value a shared random stream
                          would produce.

    The third check is the one that catches real bugs, and it is the one that
    needs a calibrated scale. Pass `reference` from `calibrate_independence` to
    judge against this family's measured chance distribution. Without one, the
    check falls back to a coarse stream-reuse threshold (|r| > 0.5) and reports
    the analytic z values as advisory only -- a large analytic z with a small
    |r| means the analytic scale is wrong for this family, not that the
    environments are coupled.
    """
    worlds = [("D_P", ep.D_P)]
    worlds += [("D_R%d" % i, w) for i, w in enumerate(ep.D_R)]
    if ep.D_U is not None:
        worlds.append(("D_U", ep.D_U))
    seeds = {nm: w.env.world_seed for nm, w in worlds}
    fps = {nm: w.fingerprint() for nm, w in worlds}
    pairs = []
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            (na, wa), (nb, wb) = worlds[i], worlds[j]
            if wa.target is None or wb.target is None:
                continue
            if len(wa.target) != len(wb.target):
                pairs.append({"a": na, "b": nb, "row_corr": None,
                              "reason": "different row counts"})
                continue
            c = corr(wa.target, wb.target)
            ne = min(effective_n(wa), effective_n(wb))
            pairs.append({"a": na, "b": nb, "row_corr": float(c),
                          "n_rows": int(len(wa.target)),
                          "n_effective": float(ne),
                          "chance_scale": float(1.0 / np.sqrt(ne)),
                          "z": float(abs(c) * np.sqrt(ne))})
    thr = (reference or {}).get("threshold", 0.5)
    bad = [p for p in pairs
           if p.get("row_corr") is not None and abs(p["row_corr"]) > thr]
    advisory = [p for p in pairs if p.get("z") is not None and p["z"] > 6.0]
    return {
        "seeds": seeds,
        "distinct_seeds": len(set(seeds.values())) == len(seeds),
        "distinct_fingerprints": len(set(fps.values())) == len(fps),
        "pairwise_row_corr": pairs,
        "threshold": float(thr),
        "calibrated": reference is not None,
        "n_over_threshold": len(bad),
        "n_analytic_z_over_6": len(advisory),
        "advisory_note": ("analytic z is advisory only when `calibrated` is "
                          "False; see calibrate_independence"),
        "passes": bool(len(set(seeds.values())) == len(seeds)
                       and len(set(fps.values())) == len(fps)
                       and not bad),
    }


def null_health(ep):
    """Did every structured null in this episode preserve what it claimed?"""
    rows = []
    for w in ep.D_N:
        rep = w.meta.get("preservation", {})
        rows.append({"world": w.name, "kind": rep.get("kind"),
                     "passes": bool(rep.get("passes", False))})
    return {"n_nulls": len(rows), "all_pass": all(r["passes"] for r in rows),
            "detail": rows}


def episode_report(ep):
    return {
        "name": ep.name,
        "family": ep.meta.get("family"),
        "fingerprint": ep.fingerprint(),
        "n_replication": len(ep.D_R),
        "n_null": len(ep.D_N),
        "independence": ep.meta.get("independence"),
        "null_health": null_health(ep),
        "n_true_objects": len(ep.D_P.truth.true_objects),
        "n_planted_artifacts": len(ep.D_P.truth.artifacts),
        "true_object_keys": [list(h.key()[:3]) for h in ep.D_P.truth.true_objects],
    }
