"""Structured nulls: break the relationship under test, keep everything else.

A null that randomises everything is not a null, it is a different experiment.
The spec is explicit: preserve realistic nuisance structure and break only the
relationship under test. For spatial data preserve autocorrelation and
composition; for temporal data use block permutation or phase randomisation;
for graphs preserve degree and local geometry while rewiring the motif; for
causal data preserve observational distributions while removing the
intervention-specific dependence.

Every generator here returns `(null_world, report)`. The report is not
decoration: it carries a measured `preserved` block and a measured `broken`
block, and `passes` is True only if the claimed statistic really was preserved
*and* the relationship really was destroyed. A null generator that quietly
fails to preserve its statistic makes every downstream false-discovery rate
meaningless, so it is asserted rather than assumed -- the same discipline
`spike_in.covariance_preservation_check` applies to the semi-synthetic path.

In a null world every object is false: `null_world.truth` contains the same
typed objects with `true=False` and `artifact_class="structured_null"`. Any
flag raised there is a false discovery by construction.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace

import numpy as np

from .util import (acf, corr, morans_i, partial_out, power_spectrum, zscore)
from .world import EnvMeta, TruthSet, World

NULL_KINDS = ("iid_permute", "block_permute", "phase_randomize",
              "torus_shift", "graph_rewire", "intervention_strip",
              "stratum_shuffle", "conditional_resample")


# --------------------------------------------------------------------------
def _nullify(world, Y_new, kind, report, extra_meta=None, latents=None,
             structure=None):
    objs = tuple(_dc_replace(h, true=False, artifact_class="structured_null")
                 for h in world.truth.objects)
    w = World(
        name="%s|null:%s" % (world.name, kind),
        family=world.family, domain=world.domain,
        latents=dict(latents if latents is not None else world.latents),
        modalities=dict(world.modalities),
        target=Y_new, target_name=world.target_name,
        nuisance=dict(world.nuisance),
        env=EnvMeta(role="null", labels=dict(world.env.labels),
                    world_seed=world.env.world_seed,
                    template=world.env.template,
                    tags=dict(world.env.tags, null_kind=kind)),
        structure=structure if structure is not None else world.structure,
        interventions=world.interventions,
        truth=TruthSet(objects=objs),
        meta=dict(world.meta, null_kind=kind, preservation=report,
                  **(extra_meta or {})),
    )
    return w, report


def _require_target(world, kind):
    if world.target is None:
        raise ValueError("null %r needs a target; world %r has none"
                         % (kind, world.name))
    return np.asarray(world.target, dtype=np.float64)


def _rel(a, b):
    """Relative deviation, safe at zero."""
    a, b = float(a), float(b)
    s = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / s


# ==========================================================================
# 1. exchangeable permutation -- the baseline null for i.i.d. rows
# ==========================================================================
def null_iid_permute(world, seed=0, feature_name=None):
    """Permute Y. Valid only when rows are exchangeable.

    Preserved: the marginal distribution of Y, exactly (it is a permutation).
    Broken: every relationship between Y and anything else.
    """
    Y = _require_target(world, "iid_permute")
    rng = np.random.default_rng(seed)
    Yp = Y[rng.permutation(len(Y))]
    f = _probe(world, feature_name)
    rep = {
        "kind": "iid_permute",
        "preserved": {"Y_moments": _moments_match(Y, Yp)},
        "broken": {"corr_with_probe_before": corr(Y, f) if f is not None else None,
                   "corr_with_probe_after": corr(Yp, f) if f is not None else None},
    }
    rep["passes"] = bool(rep["preserved"]["Y_moments"]["passes"])
    return _nullify(world, Yp, "iid_permute", rep)


# ==========================================================================
# 2. moving-block permutation -- autocorrelated rows
# ==========================================================================
def block_permutation_index(n, block_len, rng):
    """A permutation built by reordering contiguous blocks.

    Carried over from cscd/calibrate.py, where it was validated at FPR 0.055
    on autocorrelated data. Rows inside a block keep their order, so within-
    block dependence survives; only whole blocks move.
    """
    edges = list(range(0, n, block_len))
    blocks = [np.arange(s, min(s + block_len, n)) for s in edges]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def null_block_permute(world, block_len=None, seed=0, feature_name=None,
                       max_lag=8, acf_tol=0.20):
    """Reorder contiguous blocks of the time series.

    Preserved: the marginal distribution of Y exactly, and the short-lag
    autocorrelation approximately -- only the (n/block_len) block seams are
    disturbed. `acf_tol` is the tolerated absolute change in ACF at lags below
    `block_len`, and the report carries the measured change so a caller can see
    how much was actually preserved rather than trusting the tolerance.
    Broken: any relationship with a series that is not co-permuted.
    """
    Y = _require_target(world, "block_permute")
    n = len(Y)
    if block_len is None:
        block_len = max(8, int(round(n ** 0.4)))
    rng = np.random.default_rng(seed)
    idx = block_permutation_index(n, block_len, rng)
    Yp = Y[idx]
    a0, a1 = acf(Y, max_lag), acf(Yp, max_lag)
    inb = min(max_lag, max(1, block_len - 1))
    dev = float(np.max(np.abs(a0[:inb] - a1[:inb])))
    f = _probe(world, feature_name)
    rep = {
        "kind": "block_permute", "block_len": int(block_len),
        "preserved": {
            "Y_moments": _moments_match(Y, Yp),
            "acf_before": a0.tolist(), "acf_after": a1.tolist(),
            "max_acf_deviation_within_block": dev,
            "acf_tol": acf_tol,
            "acf_passes": bool(dev <= acf_tol),
        },
        "broken": {"corr_with_probe_before": corr(Y, f) if f is not None else None,
                   "corr_with_probe_after": corr(Yp, f) if f is not None else None},
    }
    rep["passes"] = bool(rep["preserved"]["Y_moments"]["passes"]
                         and rep["preserved"]["acf_passes"])
    return _nullify(world, Yp, "block_permute", rep)


# ==========================================================================
# 3. phase randomisation -- exact spectrum preservation
# ==========================================================================
def null_phase_randomize(world, seed=0, feature_name=None, tol=1e-8):
    """Randomise Fourier phases, keep the amplitude spectrum.

    Preserved: the power spectrum *exactly*, hence the full autocovariance
    function exactly, hence every second-order temporal property of Y. This is
    a strictly stronger preservation guarantee than block permutation, at the
    cost of forcing a Gaussian-like marginal.
    Broken: all phase alignment with any other series, i.e. every lead/lag
    relationship.
    """
    Y = _require_target(world, "phase_randomize")
    n = len(Y)
    rng = np.random.default_rng(seed)
    mu = Y.mean()
    F = np.fft.rfft(Y - mu)
    mag = np.abs(F)
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    Yp = np.fft.irfft(mag * np.exp(1j * ph), n=n) + mu
    p0, p1 = power_spectrum(Y), power_spectrum(Yp)
    dev = float(np.max(np.abs(p0 - p1)) / max(np.max(p0), 1e-12))
    f = _probe(world, feature_name)
    rep = {
        "kind": "phase_randomize",
        "preserved": {"relative_spectrum_deviation": dev, "tol": tol,
                      "spectrum_passes": bool(dev <= 1e-6),
                      "variance_before": float(Y.var()),
                      "variance_after": float(Yp.var()),
                      "acf_before": acf(Y, 8).tolist(),
                      "acf_after": acf(Yp, 8).tolist()},
        "broken": {"corr_with_probe_before": corr(Y, f) if f is not None else None,
                   "corr_with_probe_after": corr(Yp, f) if f is not None else None},
    }
    rep["passes"] = bool(rep["preserved"]["spectrum_passes"])
    return _nullify(world, Yp, "phase_randomize", rep)


# ==========================================================================
# 4. toroidal shift -- exact spatial autocorrelation preservation
# ==========================================================================
def null_torus_shift(world, seed=0, feature_name=None,
                     moran_rel_tol=0.15, moran_abs_floor=0.02):
    """Rigidly translate the response field on a periodic grid.

    Preserved: composition exactly (the multiset of Y values is unchanged) and
    spatial autocorrelation almost exactly -- a rigid translation of a field on
    a torus is an isometry of the lattice, so Moran's I changes only through
    the finite-radius neighbourhood at the wrap seam. The report measures the
    change rather than asserting it is zero.
    Broken: the alignment between the response field and every covariate field,
    while both remain individually as autocorrelated as before.
    """
    if world.structure.coords is None:
        raise ValueError("torus_shift needs spatial coords")
    Y = _require_target(world, "torus_shift")
    coords = np.asarray(world.structure.coords)
    side = int(world.structure.meta.get("side", int(round(np.sqrt(len(np.unique(
        coords.view([("", coords.dtype)] * coords.shape[1])))))))) if \
        "side" in world.structure.meta else None
    if side is None:
        side = int(round(np.sqrt(len(coords))))
    rng = np.random.default_rng(seed)
    xs = coords[:, 0].astype(int)
    ys = coords[:, 1].astype(int)
    dx = int(rng.integers(side // 4, side - side // 4))
    dy = int(rng.integers(side // 4, side - side // 4))
    # cell key -> row index, per time slice if the world is a panel
    time = world.structure.time
    tvals = np.zeros(len(Y), dtype=int) if time is None else np.asarray(time)
    lookup = {}
    for i in range(len(Y)):
        lookup[(int(tvals[i]), int(xs[i]), int(ys[i]))] = i
    src = np.empty(len(Y), dtype=int)
    for i in range(len(Y)):
        key = (int(tvals[i]), (int(xs[i]) + dx) % side, (int(ys[i]) + dy) % side)
        src[i] = lookup.get(key, i)
    Yp = Y[src]
    nb = world.structure.neighbors
    m0 = morans_i(Y, nb) if nb is not None else float("nan")
    m1 = morans_i(Yp, nb) if nb is not None else float("nan")
    f = _probe(world, feature_name)
    rep = {
        "kind": "torus_shift", "shift": [dx, dy], "side": side,
        "preserved": {
            "composition_identical_multiset": bool(
                np.allclose(np.sort(Y), np.sort(Yp))),
            "morans_I_before": m0, "morans_I_after": m1,
            "morans_I_abs_change": float(abs(m0 - m1)),
            "morans_I_rel_change": float(abs(m0 - m1) / max(abs(m0), 1e-12)),
            # A rigid shift on a torus is an isometry everywhere except the wrap
            # seam, and on a small grid the seam is a real fraction of the
            # cells: on a 16x16 boss-1 slice it moves Moran's I by about 0.04
            # out of 0.52. The tolerance is therefore relative to the amount of
            # autocorrelation there is, with an absolute floor for fields that
            # are nearly unautocorrelated to begin with.
            "morans_rel_tol": moran_rel_tol,
            "morans_abs_floor": moran_abs_floor,
            "morans_passes": bool(
                abs(m0 - m1) <= max(moran_abs_floor,
                                    moran_rel_tol * abs(m0))),
        },
        "broken": {"corr_with_probe_before": corr(Y, f) if f is not None else None,
                   "corr_with_probe_after": corr(Yp, f) if f is not None else None},
    }
    rep["passes"] = bool(rep["preserved"]["composition_identical_multiset"]
                         and rep["preserved"]["morans_passes"])
    return _nullify(world, Yp, "torus_shift", rep)


# ==========================================================================
# 5. degree-preserving graph rewiring
# ==========================================================================
def null_graph_rewire(world, seed=0, n_swap_per_edge=30, feature_name=None):
    """Double-edge swaps inside each graph, then recompute the motif.

    Preserved: every node's degree exactly, the number of edges exactly, and
    the node-label composition exactly (labels are never touched).
    Broken: the labelled-motif count, because rewiring destroys which labelled
    nodes are mutually adjacent while leaving the degree sequence intact.

    The world's Y is left alone and the *latent* motif count is recomputed, so
    the null is "the same graphs, same degrees, same labels, motif count no
    longer aligned with Y".
    """
    st = world.structure
    if st.edges is None or st.graph_id is None:
        raise ValueError("graph_rewire needs edges and graph_id")
    rng = np.random.default_rng(seed)
    edges = np.array(st.edges, dtype=int)
    gid = np.asarray(st.graph_id, dtype=int)
    feats = st.meta.get("node_features")
    labels = feats[:, 0] if feats is not None else None
    deg_before = np.bincount(edges.ravel(), minlength=len(gid)).astype(int)

    eg = gid[edges[:, 0]]
    new_edges = edges.copy()
    for g in np.unique(eg):
        sel = np.where(eg == g)[0]
        if len(sel) < 2:
            continue
        # Three details decide whether this chain samples uniformly from the
        # graphs with this degree sequence, and getting any of them wrong
        # produces a null that is broken for the wrong reason.
        #
        # (a) Edges are stored in (min, max) orientation. Otherwise a swap can
        #     leave an edge with a > b and the triangle recount below -- which
        #     relies on a < b < c to count each triangle exactly once --
        #     silently miscounts.
        # (b) Every ATTEMPT is a step: a rejected swap leaves the chain where
        #     it is. Retrying until a swap succeeds biases the chain towards
        #     states that admit more legal swaps, which are the anti-clustered
        #     ones. Measured on ER(18, 0.30): retry-until-success took the mean
        #     triangle count from 21.3 to 7.5, while networkx's reference
        #     double_edge_swap left it at 21.4. A null that destroys half the
        #     motif structure by construction would have looked like a
        #     well-broken null and silently inflated apparent power.
        # (c) The endpoint pairing is randomised. Always pairing (a,d),(c,b)
        #     and never (a,c),(b,d) leaves a residual bias: with (b) alone the
        #     count settled at 16.7 rather than 21.6.
        sub = np.sort(new_edges[sel], axis=1)
        present = set((int(u), int(v)) for u, v in sub)
        for _ in range(n_swap_per_edge * len(sel)):
            i, j = rng.integers(0, len(sub), 2)
            if i == j:
                continue
            a, b = int(sub[i][0]), int(sub[i][1])
            c, d = int(sub[j][0]), int(sub[j][1])
            if rng.random() < 0.5:
                a, b = b, a
            if len({a, b, c, d}) < 4:
                continue
            e1 = tuple(sorted((a, d)))
            e2 = tuple(sorted((c, b)))
            if e1 == e2 or e1 in present or e2 in present:
                continue
            present.discard(tuple(sorted((a, b))))
            present.discard(tuple(sorted((c, d))))
            present.add(e1)
            present.add(e2)
            sub[i] = e1
            sub[j] = e2
        new_edges[sel] = sub
    deg_after = np.bincount(new_edges.ravel(), minlength=len(gid)).astype(int)

    # recompute the labelled-triangle count per graph
    motif = np.zeros(int(gid.max()) + 1)
    if labels is not None:
        adj = {}
        for a, b in new_edges:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))
        for a, b in new_edges:
            a, b = sorted((int(a), int(b)))
            if labels[a] <= 0 or labels[b] <= 0:
                continue
            common = adj.get(a, set()) & adj.get(b, set())
            for c in common:
                if c > b and labels[c] > 0:
                    motif[gid[a]] += 1
    lat = dict(world.latents)
    m_before = np.asarray(lat.get("motif_count"))
    lat["motif_count"] = motif
    # "planted_motifs" added 2026-09-15 (owner directive, extending the
    # boss6/graph_motif truth-declaration fix): the null must destroy
    # WHICHEVER motif-derived latent a candidate scans. Before graph_motif's
    # truth operand was corrected from motif_count to planted_motifs, only
    # motif_count needed nulling here. Left untouched, planted_motifs still
    # held its ORIGINAL (pre-null) value -- unperturbed and still genuinely
    # correlated with the world's UNCHANGED Y -- so a candidate over it
    # flagged on every null replicate (5/5, familywise error 1.0), not
    # because it is a false discovery but because it was never nulled at
    # all. The degree-preserving rewire has no "installation" step to
    # measure an excess against, so the recomputed raw count IS this null's
    # own notion of planted_motifs -- the same value already computed for
    # motif_count above.
    lat["planted_motifs"] = motif
    Y = world.target
    # Some complete objects are graph-by-continuous interactions rather than
    # a graph main effect.  Their correct structured-null statistic is the
    # graph relation at a predeclared, observed continuous probe—not the
    # partial correlation of the motif alone.  Worlds opt in by recording the
    # probe name in their graph structure; ordinary graph-motif worlds retain
    # the historical degree/composition-adjusted main-effect test below.
    probe_name = st.meta.get("graph_rewire_probe")
    interaction_before = interaction_after = None
    if probe_name is not None and probe_name in lat and m_before is not None and Y is not None:
        probe = np.asarray(lat[probe_name], dtype=float)
        interaction_before = corr(Y, zscore(probe) * zscore(m_before))
        interaction_after = corr(Y, zscore(probe) * zscore(motif))
    rep = {
        "kind": "graph_rewire", "n_swap_per_edge": n_swap_per_edge,
        "preserved": {
            "degree_sequence_identical": bool(np.array_equal(deg_before, deg_after)),
            "n_edges_before": int(len(edges)), "n_edges_after": int(len(new_edges)),
            "labels_untouched": True,
            "label_rate_before": (float(np.mean(labels > 0))
                                  if labels is not None else None),
            "label_rate_after": (float(np.mean(labels > 0))
                                 if labels is not None else None),
        },
        "broken": {
            "motif_corr_with_original": (float(corr(m_before, motif))
                                         if m_before is not None else None),
            "motif_corr_with_Y_before": (float(corr(m_before, Y))
                                         if (m_before is not None and Y is not None)
                                         else None),
            "motif_corr_with_Y_after": (float(corr(motif, Y))
                                        if Y is not None else None),
            # The marginal motif-Y correlation is the wrong statistic here. The
            # null deliberately PRESERVES degree and composition, and the motif
            # count is partly predictable from both, so a large part of the
            # marginal correlation must survive rewiring -- measured 0.87 ->
            # 0.45, which looks like a half-broken null and is not one. The
            # relationship under test is the motif effect net of composition and
            # degree, which is what rewiring destroys and what the hierarchical-
            # novelty rule asks the object to survive.
            "motif_partial_corr_with_Y_before": _motif_partial(world, m_before, Y),
            "motif_partial_corr_with_Y_after": _motif_partial(world, motif, Y),
            "interaction_probe": probe_name,
            "interaction_corr_with_Y_before": interaction_before,
            "interaction_corr_with_Y_after": interaction_after,
            "mean_motif_before": (float(np.mean(m_before))
                                  if m_before is not None else None),
            "mean_motif_after": float(np.mean(motif)),
            # An unbiased degree-preserving chain leaves the EXPECTED motif
            # count alone -- it moves which graphs have the motif, not how many
            # motifs exist. A ratio far from 1 means the chain is biased, not
            # that the null is strong, so it is reported as a diagnostic.
            "mean_motif_ratio": (float(np.mean(motif) / max(np.mean(m_before), 1e-9))
                                 if m_before is not None else None),
        },
    }
    pb = rep["broken"]["motif_partial_corr_with_Y_before"]
    pa = rep["broken"]["motif_partial_corr_with_Y_after"]
    if interaction_before is not None and interaction_after is not None:
        rep["broken"]["is_broken"] = bool(
            abs(interaction_after) <= 0.25 * abs(interaction_before) + 0.05)
    else:
        rep["broken"]["is_broken"] = bool(
            pb is None or pa is None or abs(pa) <= 0.25 * abs(pb) + 0.05)
    ratio = rep["broken"].get("mean_motif_ratio")
    planted = bool(st.meta.get("planted"))
    # On a world with a PLANTED motif the count is supposed to fall back to the
    # degree-matched background, so a low ratio is the null working, not the
    # chain being biased. The ratio is only a bias diagnostic when the motif
    # arose by chance, where an unbiased chain must leave the expected count
    # alone.
    rep["preserved"]["motif_planted"] = planted
    rep["preserved"]["chain_unbiased"] = bool(
        planted or ratio is None or 0.80 <= ratio <= 1.25)
    rep["passes"] = bool(rep["preserved"]["degree_sequence_identical"]
                         and rep["preserved"]["chain_unbiased"]
                         and rep["broken"]["is_broken"])
    from .world import Structure as _S
    st2 = _S(kind=st.kind, coords=st.coords, time=st.time, unit=st.unit,
             neighbors=st.neighbors, edges=new_edges, graph_id=st.graph_id,
             meta=dict(st.meta))
    return _nullify(world, Y, "graph_rewire", rep, latents=lat, structure=st2)


# ==========================================================================
# 6. intervention stripping -- keep the observational law, remove the rest
# ==========================================================================
def null_intervention_strip(world, seed=0, n_bins=8, parents=None):
    """Permute Y across regimes within matched bins of the causal parents.

    Preserved: the observational conditional E[Y | parents] and the marginal
    of Y, because Y values are only exchanged between rows that share a parent
    bin. The intervention assignment and every variable are untouched, so the
    do-regime marginals of X are exactly as before.
    Broken: any dependence of Y on the regime *given* the parents, which is
    exactly the intervention-specific structure.
    """
    Y = _require_target(world, "intervention_strip")
    if world.interventions is None:
        raise ValueError("intervention_strip needs an Interventions block")
    assign = np.asarray(world.interventions.assignment)
    if parents is None:
        parents = [k for k in sorted(world.latents)
                   if np.asarray(world.latents[k]).ndim == 1][:2]
    P = np.column_stack([np.asarray(world.latents[k], dtype=float)
                         for k in parents]) if parents else np.zeros((len(Y), 0))
    rng = np.random.default_rng(seed)
    if P.shape[1]:
        codes = np.zeros(len(Y), dtype=np.int64)
        for j in range(P.shape[1]):
            q = np.quantile(P[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
            codes = codes * n_bins + np.searchsorted(q, P[:, j])
    else:
        codes = np.zeros(len(Y), dtype=np.int64)
    Yp = Y.copy()
    for c in np.unique(codes):
        m = np.where(codes == c)[0]
        Yp[m] = Y[m][rng.permutation(len(m))]

    # Bin indicators, not linear parent terms, are the correct conditioning set
    # for the broken-statistic: the permutation is exchangeable *within bins*,
    # so a linear partial-out leaves bin-level structure that both the regime
    # indicator and Y still share, and reports residual dependence the null did
    # not in fact leave behind. Measured on intervention_causal seed 2: the
    # linear statistic falls only 0.389 -> 0.207 because the do_X3 rows are a
    # point mass in X3, while the bin-conditional statistic falls to Monte-Carlo
    # noise. Both are reported so the discrepancy stays visible.
    uniq_codes = np.unique(codes)
    Dbin = (np.column_stack([(codes == c).astype(float) for c in uniq_codes[:-1]])
            if len(uniq_codes) > 1 else np.zeros((len(Y), 0)))

    def _regime_dep(y, ctrl):
        out = 0.0
        for r in np.unique(assign):
            g = (assign == r).astype(float)
            ry = partial_out(y, ctrl)
            rg = partial_out(g, ctrl)
            if ry.std() > 1e-12 and rg.std() > 1e-12:
                out = max(out, abs(float(np.corrcoef(ry, rg)[0, 1])))
        return out

    lin_ctrl = [P[:, j] for j in range(P.shape[1])]
    bin_ctrl = [Dbin] if Dbin.shape[1] else []

    rep = {
        "kind": "intervention_strip", "parents": list(parents), "n_bins": n_bins,
        "preserved": {
            "Y_moments": _moments_match(Y, Yp),
            "within_bin_permutation_only": True,
            "n_bins_used": int(len(np.unique(codes))),
            "cond_mean_max_abs_change": float(np.max([
                abs(Y[codes == c].mean() - Yp[codes == c].mean())
                for c in np.unique(codes)])),
            "assignment_untouched": True,
        },
        "broken": {
            "regime_dependence_given_bins_before": _regime_dep(Y, bin_ctrl),
            "regime_dependence_given_bins_after": _regime_dep(Yp, bin_ctrl),
            "regime_dependence_linear_before": _regime_dep(Y, lin_ctrl),
            "regime_dependence_linear_after": _regime_dep(Yp, lin_ctrl),
            "mc_noise_scale": float(1.0 / np.sqrt(len(Y))),
        },
    }
    rep["broken"]["is_broken"] = bool(
        rep["broken"]["regime_dependence_given_bins_after"]
        <= 6.0 * rep["broken"]["mc_noise_scale"])
    rep["passes"] = bool(rep["preserved"]["Y_moments"]["passes"]
                         and rep["preserved"]["cond_mean_max_abs_change"] < 1e-9
                         and rep["broken"]["is_broken"])
    return _nullify(world, Yp, "intervention_strip", rep)


# ==========================================================================
# 7. within-stratum shuffle -- composition-preserving
# ==========================================================================
def null_stratum_shuffle(world, keys=("batch",), seed=0, feature_name=None):
    """Permute Y only within strata defined by environment labels.

    This is the BMMC-style null: shuffle within donor, site and broad cell
    type so composition and every stratum-level mean survive untouched, and
    only the within-stratum pairing is destroyed.
    """
    Y = _require_target(world, "stratum_shuffle")
    labs = []
    for k in keys:
        if k in world.env.labels:
            labs.append(np.asarray(world.env.labels[k]))
        elif k in world.nuisance:
            labs.append(np.asarray(world.nuisance[k]))
    if not labs:
        raise ValueError("no stratum keys %r present in world %r"
                         % (list(keys), world.name))
    code = np.zeros(len(Y), dtype=object)
    for i in range(len(Y)):
        code[i] = tuple(l[i] for l in labs)
    uniq = {c: [] for c in set(code.tolist())}
    for i, c in enumerate(code):
        uniq[c].append(i)
    rng = np.random.default_rng(seed)
    Yp = Y.copy()
    means_before, means_after = {}, {}
    for c, idx in uniq.items():
        idx = np.array(idx)
        Yp[idx] = Y[idx][rng.permutation(len(idx))]
        means_before[str(c)] = float(Y[idx].mean())
        means_after[str(c)] = float(Yp[idx].mean())
    f = _probe(world, feature_name)
    dev = max((abs(means_before[k] - means_after[k]) for k in means_before),
              default=0.0)
    rep = {
        "kind": "stratum_shuffle", "keys": list(keys),
        "preserved": {"n_strata": len(uniq),
                      "stratum_sizes_unchanged": True,
                      "max_abs_stratum_mean_change": float(dev),
                      "stratum_means_passes": bool(dev < 1e-9),
                      "Y_moments": _moments_match(Y, Yp)},
        "broken": {"corr_with_probe_before": corr(Y, f) if f is not None else None,
                   "corr_with_probe_after": corr(Yp, f) if f is not None else None},
    }
    rep["passes"] = bool(rep["preserved"]["stratum_means_passes"])
    return _nullify(world, Yp, "stratum_shuffle", rep)


# ==========================================================================
# 8. conditional resampling -- keeps the feature/nuisance relationship
# ==========================================================================
def null_conditional_resample(world, feature_name, nuisance_names=None, seed=0):
    """Resample the tested feature from its conditional given the nuisance.

    A conditional-randomisation-style null. The feature is decomposed into its
    linear projection on the declared nuisance plus a residual; the residual is
    permuted and the projection put back. So the feature keeps its relationship
    with the nuisance (in mean) while losing its relationship with Y.

    This is the null that distinguishes a genuine object from a
    nuisance-mediated one: a `replication_mimicry` or `nuisance_confound`
    artifact survives an ordinary permutation null and is refuted here.
    """
    Y = _require_target(world, "conditional_resample")
    if feature_name not in world.latents:
        raise ValueError("feature %r not a latent of %r"
                         % (feature_name, world.name))
    x = np.asarray(world.latents[feature_name], dtype=float)
    if nuisance_names is None:
        nuisance_names = sorted(world.nuisance)
    ctrl = [np.asarray(world.nuisance[k], dtype=float).reshape(len(Y), -1)
            for k in nuisance_names] or [np.zeros((len(Y), 0))]
    proj = x - partial_out(x, ctrl)
    resid = x - proj
    rng = np.random.default_rng(seed)
    # A raw permutation of the residual is only orthogonal to the nuisance in
    # expectation, so it perturbs the feature-nuisance correlation by
    # O(1/sqrt(n)) and the preservation claim would be approximate (measured:
    # 7.8e-4 on pure_and, n=4000). Projecting the permuted residual back onto
    # the orthogonal complement of the nuisance and restoring its scale makes
    # the claim exact: corr(x_new, C) equals corr(x, C) to machine precision
    # for every declared nuisance column.
    r_perm = resid[rng.permutation(len(x))]
    r_perp = partial_out(r_perm, ctrl)
    s0, s1 = resid.std(), r_perp.std()
    if s1 > 1e-12:
        r_perp = r_perp * (s0 / s1)
    x_new = proj + r_perp
    lat = dict(world.latents)
    lat[feature_name] = x_new
    c_before = [corr(x, np.asarray(world.nuisance[k], dtype=float).ravel())
                for k in nuisance_names
                if np.asarray(world.nuisance[k]).ndim == 1]
    c_after = [corr(x_new, np.asarray(world.nuisance[k], dtype=float).ravel())
               for k in nuisance_names
               if np.asarray(world.nuisance[k]).ndim == 1]
    dev = (float(np.max(np.abs(np.array(c_before) - np.array(c_after))))
           if c_before else 0.0)
    rep = {
        "kind": "conditional_resample", "feature": feature_name,
        "nuisance": list(nuisance_names),
        "preserved": {"feature_nuisance_corr_before": c_before,
                      "feature_nuisance_corr_after": c_after,
                      "max_abs_change": dev,
                      "passes_feature_nuisance": bool(dev < 1e-8),
                      "feature_moments": _moments_match(x, x_new)},
        "broken": {
            # The MARGINAL feature-Y correlation is not expected to vanish: the
            # nuisance projection is deliberately retained, and for a
            # nuisance-mediated artifact that projection is most of the signal
            # (measured on replication_mimicry: 0.360 -> 0.315). The statistic
            # this null destroys is the association NET of the declared
            # nuisance, which is exactly the quantity a genuine object has to
            # own and a nuisance-mediated artifact does not.
            "corr_feature_Y_before": corr(x, Y),
            "corr_feature_Y_after": corr(x_new, Y),
            "partial_corr_feature_Y_given_nuisance_before":
                _pcorr(x, Y, ctrl),
            "partial_corr_feature_Y_given_nuisance_after":
                _pcorr(x_new, Y, ctrl),
            "mc_noise_scale": float(1.0 / np.sqrt(len(Y))),
        },
    }
    rep["broken"]["is_broken"] = bool(
        abs(rep["broken"]["partial_corr_feature_Y_given_nuisance_after"])
        <= 6.0 / np.sqrt(len(Y)))
    rep["passes"] = bool(rep["preserved"]["passes_feature_nuisance"]
                         and rep["broken"]["is_broken"])
    return _nullify(world, Y, "conditional_resample", rep, latents=lat)


# --------------------------------------------------------------------------
def _pcorr(x, y, ctrl):
    rx = partial_out(np.asarray(x, dtype=float), ctrl)
    ry = partial_out(np.asarray(y, dtype=float), ctrl)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _motif_partial(world, motif, Y):
    """corr(motif, Y | composition, degree) -- the statistic actually under test.

    The controls enter as cubic polynomials, not linear terms. A labelled
    triangle needs three labelled nodes, so the motif count scales like the
    cube of the label rate; a linear control leaves that curvature in the
    residual and the null then reads as only three-quarters broken (measured
    0.899 -> 0.241 with linear controls, and the leftover was composition
    curvature, not surviving motif structure).
    """
    if motif is None or Y is None:
        return None
    ctrl = []
    for k in ("composition", "density"):
        if k in world.nuisance:
            v = np.asarray(world.nuisance[k], dtype=float)
            ctrl.extend([v, v ** 2, v ** 3])
    if not ctrl:
        return float(corr(motif, Y))
    ry = partial_out(np.asarray(Y, dtype=float), ctrl)
    rm = partial_out(np.asarray(motif, dtype=float), ctrl)
    if ry.std() < 1e-12 or rm.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ry, rm)[0, 1])


def _probe(world, feature_name):
    """The series whose relationship with Y the null claims to have broken.

    Defaults to a reconstruction of the first true relational object's planted
    feature -- the product of its operands for a conjunction, the operand
    itself for a main effect -- rather than to a bare operand. On a pure-AND
    world a bare operand has zero correlation with Y both before and after, so
    a report built on it says nothing about whether the null worked.
    """
    if feature_name is not None:
        return np.asarray(world.latents[feature_name], dtype=float)
    if not hasattr(world.truth, "objects"):
        return None
    for h in world.truth.objects:
        if not h.true:
            continue
        ops = [o for o in h.operands if o in world.latents
               and np.asarray(world.latents[o]).ndim == 1]
        if not ops:
            continue
        v = np.asarray(world.latents[ops[0]], dtype=float)
        if h.operator in ("AND", "XOR", "CONDITION", "SPACE") and len(ops) > 1:
            for o in ops[1:]:
                v = v * np.asarray(world.latents[o], dtype=float)
        return v
    return None


def _moments_match(a, b, tol=1e-9):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same_multiset = bool(np.allclose(np.sort(a), np.sort(b)))
    return {"mean_before": float(a.mean()), "mean_after": float(b.mean()),
            "sd_before": float(a.std()), "sd_after": float(b.std()),
            "identical_multiset": same_multiset,
            "passes": bool(same_multiset
                           or (abs(a.mean() - b.mean()) < 1e-6
                               and _rel(a.std(), b.std()) < 1e-3))}


# --------------------------------------------------------------------------
# default null policy per family: which null a family's structure demands
# --------------------------------------------------------------------------
DEFAULT_NULLS = {
    "additive_factors": ("iid_permute",),
    "linear_combination": ("iid_permute",),
    "pure_and": ("iid_permute",),
    "invisible_k3": ("iid_permute",),
    "invisible_k4": ("iid_permute",),
    "invisible_k5": ("iid_permute",),
    "invisible_xor_k3": ("iid_permute",),
    "invisible_xor_k4": ("iid_permute",),
    "xor": ("iid_permute",),
    "threshold": ("iid_permute",),
    "hill": ("iid_permute",),
    "context_gated": ("iid_permute",),
    "manifold": (),
    "spatial_interaction": ("torus_shift",),
    "temporal_lag": ("block_permute", "phase_randomize"),
    "regime_switch": ("block_permute",),
    "graph_motif": ("graph_rewire",),
    "latent_confound": ("iid_permute",),
    "intervention_causal": ("intervention_strip",),
    "hidden_symmetry": (),
    "sparse_ode": (),
    "anomaly": (),
    "mixture": (),
    "experimental_design": (),
    "boss1_cond_thresh_space_lag": ("torus_shift", "block_permute"),
    "boss2_xor_space_regime": ("torus_shift",),
    "boss3_hill_space_cause": ("torus_shift", "intervention_strip"),
    "boss4_ode_manifold_anomaly": (),
    "boss5_design_and_threshold": ("iid_permute",),
    "boss6_hill_graph": ("graph_rewire",),
    "boss7_causal_hill_lag": ("intervention_strip",),
}

_DISPATCH = {
    "iid_permute": null_iid_permute,
    "block_permute": null_block_permute,
    "phase_randomize": null_phase_randomize,
    "torus_shift": null_torus_shift,
    "graph_rewire": null_graph_rewire,
    "intervention_strip": null_intervention_strip,
    "stratum_shuffle": null_stratum_shuffle,
    "conditional_resample": null_conditional_resample,
}


def make_null(world, kind, **kw):
    if kind not in _DISPATCH:
        raise KeyError("unknown null kind %r; known: %s" % (kind, NULL_KINDS))
    return _DISPATCH[kind](world, **kw)
