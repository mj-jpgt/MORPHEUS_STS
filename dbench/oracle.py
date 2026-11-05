"""The oracle gate: can a specialist, handed the true latents, recover each family?

This is the first scientific gate in the build plan. It does not test MORPHEUS.
It tests DiscoveryBench: if an appropriate specialist method cannot recover a
family's planted object when it is handed the ground-truth latent variables,
then a failure by any learned representation on that family says nothing, and
the family has to be fixed before it can measure anything.

Two disciplines make the gate falsifiable rather than decorative.

**The specialists search; they are not told the answer.** `features.build_feature`
refuses to supply a threshold location or a lag from ground truth, so the
threshold specialist scans a tau grid and pays a multiplicity price for it, the
lag specialist scans every lag over every latent, and the conjunction
specialist scans every pair. Distractor latents are in the scan, so precision
can fall.

**Every specialist is also run on a matched structured null.** A gate that
could only ever return "recovered" is not evidence. `run_gate` reports, for
every family, both the recovery on the real world and the number of flags the
same specialist raises on a null world where nothing is true. A family passes
only if it recovers the planted object AND stays quiet on the null.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor

from . import nulls as N
from .features import (FeatureError, build_composite, build_feature,
                       composite_operands, composite_parent_columns,
                       lag_column, normalize_term, parent_columns, term_label)
from .metrics import Claim, match_claims
from .util import (benjamini_hochberg, derive_seed, hill,
                   package_source_sha256, partial_out,
                   ridge_r2, soft_threshold, zscore)

ALPHA = 0.05


# --------------------------------------------------------------------------
# common evidence statistic
# --------------------------------------------------------------------------
def partial_evidence(y, f, controls=()):
    """Partial F-test of adding f to a model that already contains controls.

    Returns the standardised effect (partial correlation), the signed
    likelihood-ratio-style evidence s = sign(beta) sqrt(2 (l1 - l0)), and a
    p-value. Under Gaussian errors the F-test and that likelihood ratio are
    monotone in each other, so the same number serves the compiler's
    differentiable reward and the gate's classical test.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    f = np.asarray(f, dtype=np.float64).ravel()
    n = len(y)
    ctrl = [np.asarray(c, dtype=np.float64) for c in controls]
    X0 = np.column_stack([np.ones(n)] + ctrl) if ctrl else np.ones((n, 1))
    X1 = np.column_stack([X0, f])
    if np.std(f) < 1e-12:
        return {"effect": 0.0, "s": 0.0, "p": 1.0, "beta": 0.0}
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r0 = y - X0 @ b0
    r1 = y - X1 @ b1
    rss0, rss1 = float(r0 @ r0), float(r1 @ r1)
    df1 = n - X1.shape[1]
    if df1 <= 1 or rss1 <= 0:
        return {"effect": 0.0, "s": 0.0, "p": 1.0, "beta": 0.0}
    F = (rss0 - rss1) / max(rss1 / df1, 1e-300)
    p = float(stats.f.sf(max(F, 0.0), 1, df1))
    beta = float(b1[-1])
    fr = partial_out(f, ctrl) if ctrl else f - f.mean()
    pc = (float(np.corrcoef(r0, fr)[0, 1])
          if fr.std() > 1e-12 and r0.std() > 1e-12 else 0.0)
    s = float(np.sign(beta) * np.sqrt(max(n * np.log(rss0 / max(rss1, 1e-300)),
                                          0.0)))
    return {"effect": pc, "s": s, "p": p, "beta": beta, "F": float(F)}


def _grid_argbest(evidences, anchors):
    """Pick the best grid point by F, never by loop/scan order.

    `evidences[i]` is a `partial_evidence` dict for grid point `i`; `anchors[i]`
    is that point's own parameter value (a weight, a tau, a K, ...). Three
    scans (`scan_lincomb`, `scan_threshold`, `scan_saturation`) used to keep a
    running "best so far" and replace it only on a *strict* `p <` improvement.
    That is unsound once the grid is wide or the signal strong enough that
    many points underflow `stats.f.sf` to exactly 0.0: the survival function
    saturates long before F stops discriminating between points, so a run of
    consecutive grid points all reads `p == 0.0`, the strict-`<` rule never
    fires again once it is first hit, and the claim is handed to whichever
    point the loop reached first in that run -- an artefact of iteration
    order, not of the data. On one measured case (`linear_combination`,
    seed 11) 71 of 127 weight points underflowed this way and the strict-`<`
    winner was 30 grid steps from the true optimum.

    `F` is a strictly decreasing function of `p` for fixed degrees of freedom
    (`stats.f.sf` is continuous and monotone in its first argument), so
    ranking by F descending is exactly ranking by p ascending, with no
    resolution lost -- F is a ratio of sums of squares and does not underflow
    the way a right-tail probability does.

    Residual ties (F identical to float precision -- reachable only by
    construction, e.g. a hand-built test grid) are broken by a rule declared
    here and nowhere else: prefer the point whose own parameter value is
    closest to the mean of every anchor value searched; if that is still
    tied, prefer the smaller anchor value. Both tie-breaks are properties of
    the tied points' *values*, not of where the loop visited them, so
    reversing or shuffling the scan order cannot change the winner.

    Returns the winning index, or None if no point had a finite F.
    """
    finite = [i for i in range(len(evidences))
              if np.isfinite(evidences[i].get("F", np.nan))]
    if not finite:
        return None
    center = float(np.mean([anchors[i] for i in finite]))
    best_F = max(evidences[i]["F"] for i in finite)
    tied = [i for i in finite if evidences[i]["F"] == best_F]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda i: (abs(float(anchors[i]) - center), float(anchors[i])))
    return tied[0]


def permutation_p(world, y, f, controls, kind, n_perm=199, seed=0):
    """Structure-respecting permutation p-value for a partial effect.

    OLS p-values are anti-conservative on autocorrelated and spatially
    dependent worlds, so on those the residualised feature is permuted with the
    same machinery the structured nulls use -- moving-block for time, toroidal
    shift for space -- and the observed statistic is compared to that
    distribution.
    """
    rng = np.random.default_rng(seed)
    obs = abs(partial_evidence(y, f, controls)["s"])
    fr = partial_out(np.asarray(f, dtype=float), list(controls)) if controls \
        else np.asarray(f, dtype=float)
    n = len(fr)
    cnt = 0
    for b in range(n_perm):
        if kind == "block":
            idx = N.block_permutation_index(n, max(8, int(n ** 0.4)), rng)
        elif kind == "torus":
            idx = _torus_index(world, rng)
        else:
            idx = rng.permutation(n)
        cnt += int(abs(partial_evidence(y, fr[idx], controls)["s"]) >= obs)
    return (cnt + 1.0) / (n_perm + 1.0)


def _torus_index(world, rng, min_shift=None):
    """A rigid toroidal translation of the lattice, drawn independently per
    time slice.

    Two design choices, both forced by what went wrong without them.

    The shift must exceed the neighbourhood radius. A one-cell shift maps a
    point onto its own neighbour, so the resampled response is still explained
    by the neighbourhood aggregate and the permutation leaves intact the very
    relationship it exists to destroy.

    The shift is one rigid translation of the whole panel, not an independent
    draw per time slice. Per-slice shifts look attractive -- they turn a few
    hundred distinct resamplings into effectively unbounded ones -- and they
    are wrong here: a panel feature with a persistent per-cell component
    (BOSS-1's gate C and the neighbourhood mean of C are static per cell) has
    that persistence averaged away by independent per-slice shifts, so the null
    distribution is tighter than the truth and the test becomes
    anti-conservative in exactly the direction that matters. Measured on the
    BOSS-1 torus-shift null, the per-slice reference put the 95th percentile at
    0.027 against a global-shift 0.042 for the same statistic. A rigid
    translation preserves the feature's spatial autocorrelation, its temporal
    persistence and its cross-slice alignment all at once, which is the whole
    reason it is the right isometry.
    """
    st = world.structure
    coords = np.asarray(st.coords)
    side = int(st.meta.get("side", int(round(np.sqrt(len(coords))))))
    radius = float(st.meta.get("radius", 1.5))
    lo = int(min_shift if min_shift is not None else max(1, int(np.ceil(radius))))
    time = st.time
    tv = np.zeros(len(coords), dtype=int) if time is None else np.asarray(time)
    xs, ys = coords[:, 0].astype(int), coords[:, 1].astype(int)
    # The (t, x, y) -> row lookup depends only on the lattice, so it is built
    # once per world as a dense array rather than once per permutation as a
    # dict: a max-statistic null over a search grid draws tens of thousands of
    # shifts, and a per-row Python loop there costs more than every model fit
    # in the gate put together.
    lut = getattr(st, "_torus_lut", None)
    if lut is None:
        T = int(tv.max()) + 1
        lut = np.arange(len(coords), dtype=np.int64)[None, None, :] * 0 - 1
        lut = np.full((T, side, side), -1, dtype=np.int64)
        lut[tv, xs % side, ys % side] = np.arange(len(coords), dtype=np.int64)
        object.__setattr__(st, "_torus_lut", lut)
        object.__setattr__(st, "_torus_txy", (tv, xs % side, ys % side))
    tv, xs, ys = st._torus_txy
    while True:
        dx = int(rng.integers(0, side))
        dy = int(rng.integers(0, side))
        if min(dx, side - dx) > lo or min(dy, side - dy) > lo:
            break
    out = lut[tv, (xs + dx) % side, (ys + dy) % side]
    return np.where(out < 0, np.arange(len(out), dtype=np.int64), out)


def _perm_kind(world):
    st = world.structure
    if st.coords is not None:
        return "torus"
    if st.time is not None:
        return "block"
    return "iid"


def _scan_names(world, max_names=14, include_derived=False):
    """Candidate latent names: 1-D, non-degenerate, not operator-derived.

    Derived latents (`aggB`, `B_lag3`, `thrA`, `majB`, `hillD`, ...) are an
    operator already applied at the world's true parameter. Leaving them in the
    candidate set would let a lag scan pass without finding the lag and a
    threshold scan pass without finding tau, which is the difference between a
    gate and a lookup. They are declared by the generator in
    `world.meta["derived_latents"]` and excluded here by default.
    """
    derived = set() if include_derived else set(
        world.meta.get("derived_latents", ()) or ())
    out = []
    for k in sorted(world.latents):
        if k in derived:
            continue
        v = np.asarray(world.latents[k])
        if v.ndim == 1 and np.std(v) > 1e-9:
            out.append(k)
    return out[:max_names]


# --------------------------------------------------------------------------
# relational specialists
# --------------------------------------------------------------------------
def _mk(world, hid, operator, operands, params, lag, y, f, mode,
        n_perm=None, seed=0, extra_controls=()):
    """One candidate claim, tested against its parent family.

    `n_perm=None` means "let the world decide": on a spatially or temporally
    dependent world the analytic partial-F p-value assumes independent rows and
    is anti-conservative, so a structure-respecting permutation is used
    instead. This is not a precaution -- the calibration control measured a
    family-wise error rate of 0.375 for the gated scan on `regime_switch`
    block-permutation nulls with the analytic p, against a nominal 0.05. On an
    i.i.d. world the analytic test is exact and the permutation buys nothing
    but time.
    """
    ctrl, _ = parent_columns(world, operator, operands, params, lag)
    ctrl = list(ctrl) + list(extra_controls)
    ev = partial_evidence(y, f, ctrl)
    p = ev["p"]
    if n_perm is None:
        n_perm = 0 if _perm_kind(world) == "iid" else 199
    if n_perm:
        p = permutation_p(world, y, f, ctrl, _perm_kind(world),
                          n_perm=n_perm, seed=seed)
    return Claim(id=hid, operator=operator, operands=tuple(operands),
                 target=world.target_name, lag=lag,
                 context=({"gate_var": params["gate_var"]}
                          if "gate_var" in params else {}),
                 effect=ev["effect"], p_value=float(p),
                 complexity=1 + len(operands) + (1 if lag else 0),
                 discovery_mode=mode, params=dict(params))


def scan_main_effects(world, names=None):
    y = world.target
    names = names or _scan_names(world)
    out = []
    for k in names:
        f = np.asarray(world.latents[k], dtype=float)
        out.append(_mk(world, "ADD:%s" % k, "ADD", (k,), {}, None, y, f,
                       "relational"))
    return out


def scan_pairs(world, names=None, operator="AND", n_perm=None):
    y = world.target
    names = names or _scan_names(world)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            f = zscore(world.latents[a]) * zscore(world.latents[b])
            out.append(_mk(world, "AND:%s*%s" % (a, b), "AND", (a, b), {},
                           None, y, f, "relational", n_perm=n_perm,
                           seed=derive_seed("pairs", a, b) % 10 ** 6))
    return out


def scan_lincomb(world, names=None, grid=None):
    """Search every latent pair over the weight grid the compiler prices.

    One claim per pair, at the pair's best weight, with the p-value
    Bonferroni-corrected by the grid size -- the same accounting
    `scan_threshold` applies to its tau search, and for the same reason: the
    best of `G` weights is a maximum of `G` tests and without the correction
    the scan would find a combination in noise.

    Weights are taken from `{k/G}` with `w_b = 1 - w_a`, which is the grid the
    truth is planted on. Searching a continuous direction instead would make
    the specialist strictly stronger and the object trivially recoverable, and
    would no longer be the search the compiler's ranking has to perform.
    """
    from .families import LINCOMB_GRID
    G = int(grid or LINCOMB_GRID)
    y = world.target
    names = names or _scan_names(world)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            za = zscore(np.asarray(world.latents[a], dtype=float))
            zb = zscore(np.asarray(world.latents[b], dtype=float))
            evs, was, fs = [], [], []
            for k in range(1, G):
                wa = k / float(G)
                f = wa * za + (1.0 - wa) * zb
                if np.std(f) < 1e-9:
                    continue
                evs.append(partial_evidence(y, f, []))
                was.append(wa)
                fs.append(f)
            idx = _grid_argbest(evs, was)
            if idx is None:
                continue
            ev, wa, f = evs[idx], was[idx], fs[idx]
            c = _mk(world, "LINCOMB:%s,%s" % (a, b), "LINCOMB", (a, b),
                    {"weights": (float(wa), float(1.0 - wa)), "grid": G},
                    None, y, f, "relational")
            c.p_value = float(min(1.0, ev["p"] * (G - 1)))
            c.params["weight_grid_size"] = int(G - 1)
            out.append(c)
    return out


def scan_xor(world, names=None):
    y = world.target
    names = names or _scan_names(world)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            f = build_feature(world, "XOR", (a, b))
            out.append(_mk(world, "XOR:%s^%s" % (a, b), "XOR", (a, b), {},
                           None, y, f, "relational"))
    return out


def scan_threshold(world, names=None, taus=None):
    """Profile a threshold over a tau grid, controlling for the linear parent.

    The best tau over a grid of size G is a maximum of G tests, so the reported
    p-value is Bonferroni-corrected by G. Without that the scan would find a
    threshold in pure noise.
    """
    y = world.target
    names = names or _scan_names(world)
    taus = taus if taus is not None else np.linspace(-1.8, 1.8, 19)
    out = []
    for k in names:
        x = np.asarray(world.latents[k], dtype=float)
        evs, tas, fs = [], [], []
        for tau in taus:
            f = soft_threshold(x, tau, 0.1)
            if np.std(f) < 1e-9:
                continue
            evs.append(partial_evidence(y, f, [x]))
            tas.append(float(tau))
            fs.append(f)
        idx = _grid_argbest(evs, tas)
        if idx is None:
            continue
        ev, tau, f = evs[idx], tas[idx], fs[idx]
        c = _mk(world, "THRESH:%s" % k, "THRESHOLD", (k,),
                {"tau": tau, "temp": 0.1}, None, y, f, "relational")
        c.p_value = float(min(1.0, ev["p"] * len(taus)))
        c.params["tau_grid_size"] = int(len(taus))
        out.append(c)
    return out


def scan_saturation(world, names=None, Ks=None, nexps=(2.0, 4.0, 8.0)):
    y = world.target
    names = names or _scan_names(world)
    Ks = Ks if Ks is not None else np.geomspace(0.25, 4.0, 9)
    out = []
    for k in names:
        x = np.asarray(world.latents[k], dtype=float)
        xp = x if x.min() >= 0 else np.exp(0.6 * x)
        evs, Ks_seen, nes_seen, fs = [], [], [], []
        ntest = 0
        for K in Ks:
            for ne in nexps:
                f = hill(xp, K=float(K), nexp=float(ne))
                if np.std(f) < 1e-9:
                    continue
                ntest += 1
                evs.append(partial_evidence(y, f, [x]))
                Ks_seen.append(float(K))
                nes_seen.append(float(ne))
                fs.append(f)
        # `_grid_argbest`'s declared tie-break anchors on `K` alone: `nexp`
        # only ever takes 3 values and a genuine (K, nexp) double-tie in F is
        # not a case this grid produces, but K's own tie-break stays a
        # property of K's value, never of scan position, either way.
        idx = _grid_argbest(evs, Ks_seen)
        if idx is None:
            continue
        ev, K, ne, f = evs[idx], Ks_seen[idx], nes_seen[idx], fs[idx]
        c = _mk(world, "SAT:%s" % k, "SATURATE", (k,), {"K": K, "n": ne},
                None, y, f, "relational")
        c.p_value = float(min(1.0, ev["p"] * max(ntest, 1)))
        out.append(c)
    return out


def scan_gated(world, names=None, max_levels=3):
    """Context-gated interactions: every low-cardinality latent is a gate.

    The gate indicator is built by enumerating the gate's levels, not by a
    median split. A median split silently degenerates whenever the gate is
    unbalanced -- for a binary gate with prevalence above one half the median
    equals the gate's maximum, `g > median` is false everywhere, and the design
    column collapses to a constant times the operand, so every candidate
    returns exactly zero evidence. Enumerating levels also handles three-level
    gates and string-valued regime labels, and it is the same
    `params["gate_value"]` contract `features.build_feature` uses, so the
    claim the scan emits can be rebuilt in a replication environment.
    """
    y = world.target
    names = names or _scan_names(world)
    gates = [k for k in names
             if len(np.unique(np.asarray(world.latents[k]))) <= max_levels]
    out = []
    for g in gates:
        gv = np.asarray(world.latents[g], dtype=float)
        levels = np.unique(gv)
        # A two-level gate is fully described by one contrast; a k-level gate
        # needs one indicator per level, each tested against the others.
        probe = levels[1:] if len(levels) == 2 else levels
        for lev in probe:
            ind = (gv == lev).astype(float)
            if ind.std() < 1e-9:
                continue
            gs = 2.0 * ind - 1.0
            for k in names:
                if k == g:
                    continue
                x = np.asarray(world.latents[k], dtype=float)
                f = gs * zscore(x)
                c = _mk(world, "COND:%s|%s=%g" % (k, g, lev), "CONDITION",
                        (k, g), {"gate_var": g, "gate_level": float(lev)},
                        None, y, f, "relational")
                out.append(c)
    return out


def scan_lags(world, names=None, lags=(0, 1, 2, 3, 4, 6, 8, 12), n_perm=199):
    """Search every latent at every lag, controlling for the contemporaneous
    term, with block-permutation p-values."""
    y = world.target
    names = names or _scan_names(world)
    out = []
    for k in names:
        x = np.asarray(world.latents[k], dtype=float)
        for L in lags:
            if L == 0:
                continue
            f = lag_column(world, x, L)
            ctrl = [x] + [np.asarray(v, dtype=float)
                          for v in world.nuisance.values()
                          if np.asarray(v).ndim == 1]
            p = permutation_p(world, y, f, ctrl, "block", n_perm=n_perm,
                              seed=derive_seed("lag", k, L) % 10 ** 6)
            ev = partial_evidence(y, f, ctrl)
            c = Claim(id="LAG:%s@%d" % (k, L), operator="LAG", operands=(k,),
                      target=world.target_name, lag=int(L),
                      effect=ev["effect"], p_value=float(p), complexity=2,
                      discovery_mode="relational", params={"lag": int(L)})
            out.append(c)
    return out


def scan_lagged_pairs(world, names=None, lags=(1, 2, 3, 4, 6, 8, 12),
                      top_k=3, n_perm=None):
    """`own(a) x LAG_l(b)`, a != b -- the shape every LAG composition plants.

    The gap this closes, measured. Two of the twenty-two members of the
    Headline-2 bank take a lag argument at all: `scan_lags`, which is
    univariate and emits `LAG:<var>@<l>` only, and `scan_boss1`, whose lag
    lives inside a `SPACE` term and therefore yields zero candidates on a
    world with no neighbourhood graph. Nothing in the bank could form a lagged
    *pair*. On the twenty-six pure-temporal LAG compositions the best candidate
    the whole bank can express carries a median |partial correlation| of 0.07
    against an oracle 0.94, and the best *matchable* one carries 0.04 -- while
    handing the same bank the generator's own derived `B_lag2` column makes
    `A x B_lag2` carry 0.600. The signal was never missing; the vocabulary was.

    The lag is searched, not looked up. `_scan_names` excludes the generator's
    declared `derived_latents`, so `B_lag2` is unreachable and the only way to
    the object is through this grid -- the same grid `scan_lags` already uses.

    No new statistic. Each candidate is a two-term composite program and the
    scoring is `_search_programs`, exactly as for the four BOSS combo scans.
    What is new is that `n_perm` is chosen from the **substrate** rather than
    from a constant, because the two available statistics fail on opposite
    substrates and the audit measured both:

      * pure temporal (time, no coords) -> `n_perm = 0`, i.e. the analytic
        partial-F p times the number of programs searched. `compose.py`'s
        temporal latents are drawn i.i.d. per timestep, so the independence
        assumption that makes that p anti-conservative elsewhere is *true*
        here, and the realised rate on matched `block_permute` null worlds was
        0 of 225 draws over 45 worlds.
      * spatiotemporal (coords and time) -> `n_perm = 199`, i.e.
        `maxT_permutation` under rigid toroidal translation. The analytic p is
        inadmissible here and not merely suspect: the same scan scored by it
        flagged 9 of 10 spatiotemporal null worlds. Bonferroni-correcting a
        *permutation* p instead is not the alternative -- at ~140 programs it
        would need p <= 3.6e-4 against a floor of 1/(n_perm+1) = 5e-3 and could
        never reject however strong the object. The max statistic pays for
        multiplicity inside the null distribution, so its resolution stays
        1/(n_perm+1) at any grid width, which is the property this scan needs.

    Operands are **base names**. A claim assembled on a derived column would
    name `(A, B_lag2)` against a truth's `(A, B)` -- operand Jaccard 0.25-0.33,
    under `metrics.OPERAND_FLOOR` -- so a correct lag scan would go uncredited.
    `composite_operands` returns the leaf names, and `Claim.lag` carries the
    lag, which `metrics.structural_conflict` gates on unconditionally.

    Two-term by construction, and the misses are the point. A world whose gate
    is `CONDITION` (`compose.py` builds it as `f * (2*Cctx - 1)`: sign-flipping
    and zero-marginal) has no marginal association for a two-term product to
    find; those need a three-term `gate x own x LAG` program and this scan is
    expected to miss them. A version of this that recovered everything would be
    reading the answer.
    """
    st = world.structure
    if st.time is None:
        return []
    y = world.target
    names = names or _scan_names(world)
    if n_perm is None:
        n_perm = 0 if st.coords is None else 199
    cand = []
    for a in names:
        for b in names:
            if a == b:
                continue
            for L in lags:
                cand.append([{"op": "ADD", "operands": [a]},
                             {"op": "LAG", "operands": [b], "lag": int(L)}])
    if not cand:
        return []
    hits, n = _search_programs(world, cand, y, top_k=top_k, n_perm=n_perm,
                               seed=derive_seed("lagpair", world.family) % 10 ** 6)
    out = []
    for r in hits:
        L = normalize_term(r["terms"][1])[3]
        params = {"lag": int(L), "n_programs_searched": int(n),
                  "n_structural_programs": r.get("n_structural_programs"),
                  "p_before_search_correction": r["p_raw"],
                  "program": r["label"]}
        if n_perm:
            # a max-statistic permutation p; its floor is 1/(n_perm+1) and
            # `claim_resolution` must be able to say so
            params["n_perm"] = int(n_perm)
            params["decision_statistic"] = "maxT_permutation:%s" % _perm_kind(world)
        else:
            # analytic partial-F times the search size; no floor. Deliberately
            # NOT stored under the key `p_bonferroni`, which `claim_resolution`
            # reads as evidence of an unreported permutation count and would
            # then report a floor of 1/200 that this statistic does not have.
            params["p_analytic_times_search"] = r.get("p_bonferroni")
            params["decision_statistic"] = "analytic_partial_F_x_n_searched"
        out.append(Claim(
            id="ANDLAG:%s@%d" % (r["label"], int(L)),
            operator="AND", operands=composite_operands(r["terms"]),
            target=world.target_name, lag=int(L),
            effect=r["evidence"]["effect"], p_value=float(r["p_searched"]),
            complexity=4, discovery_mode="relational", params=params))
    return out


def scan_spatial(world, names=None, n_perm=499):
    """Own-cell state times neighbourhood aggregate of a partner state."""
    y = world.target
    names = names or _scan_names(world)
    names = [k for k in names if not k.startswith("agg")]
    out = []
    for a in names:
        for b in names:
            f = build_feature(world, "SPACE", (a, b), {"agg": "mean"})
            ctrl, _ = parent_columns(world, "SPACE", (a, b), {"agg": "mean"})
            p = permutation_p(world, y, f, ctrl, "torus", n_perm=n_perm,
                              seed=derive_seed("pairs", a, b) % 10 ** 6)
            ev = partial_evidence(y, f, ctrl)
            out.append(Claim(id="SPACE:%s*agg(%s)" % (a, b), operator="SPACE",
                             operands=(a, b), target=world.target_name,
                             effect=ev["effect"], p_value=float(p),
                             complexity=3, discovery_mode="relational",
                             params={"agg": "mean"}))
    return out


def scan_anomaly(world, names=None, max_levels=3):
    """Native ANOMALY specialist -- a physics-native residual-anomaly
    detector, two engines, both scored through the same claims/BH/RDC path
    every other RELATIONAL_BANK member uses (so recall, flag rate, FCR and
    matched-budget recall are computed identically, not bolted on).

    compose.py's ANOMALY gate multiplies the composition's core signal by a
    rare-regime indicator (`world.lat["regime"]`, or any low-cardinality
    latent playing that role): under the regime the parent-adjusted residual
    of Y shifts location; outside it, it does not. Two independent engines
    look for exactly that:

    Engine 1, robust z-score: for each low-cardinality candidate gate `g` and
    level, adjust Y for every OTHER candidate latent by ordinary least
    squares (the same "parent" set `scan_gated`/`scan_threshold` control
    for), then compare the residual's median inside vs outside the candidate
    group using a robust (MAD-scaled) z-score -- robust because a handful of
    genuine outliers must not be allowed to inflate the very MAD used to
    judge them.

    Engine 2, isolation forest: fits `sklearn.ensemble.IsolationForest`
    (fixed hyperparameters: n_estimators=100, contamination="auto", one
    fixed seed) on the residual (a single column, reshaped) and reports each
    candidate group's mean anomaly score as the claim's effect -- a
    genuinely different detection principle (partitioning-based isolation
    depth, not a location-shift test), so the two engines can disagree.

    Both engines emit ordinary `Claim`s (operator="ANOMALY", operands=(g,),
    context={"gate_var": g, "gate_level": lev}), so `_search_programs`,
    `benjamini_hochberg`, `metrics.rdc` and replication attachment all apply
    unmodified -- this specialist is a candidate generator, not a separate
    scoring path.
    """
    from sklearn.ensemble import IsolationForest

    y = world.target
    names = names or _scan_names(world)
    gates = [k for k in names
            if len(np.unique(np.asarray(world.latents[k]))) <= max_levels]
    out = []
    for g in gates:
        gv = np.asarray(world.latents[g], dtype=float)
        levels = np.unique(gv)
        probe = levels[1:] if len(levels) == 2 else levels
        others = [k for k in names if k != g][:6]
        ctrl_cols = [np.asarray(world.latents[k], dtype=float) for k in others]
        if ctrl_cols:
            Xc = np.column_stack(ctrl_cols)
            beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
            resid = y - Xc @ beta
        else:
            resid = y - y.mean()
        for lev in probe:
            ind = gv == lev
            if ind.sum() < 5 or (~ind).sum() < 5:
                continue
            # engine 1: robust (median/MAD) location-shift z-score
            med_in, med_out = np.median(resid[ind]), np.median(resid[~ind])
            mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-9
            robust_z = float((med_in - med_out) / mad)
            ev1 = partial_evidence(y, (ind.astype(float) * 2 - 1), ctrl_cols)
            out.append(Claim(
                id="ANOMALY:%s=%g|zscore" % (g, lev), operator="ANOMALY",
                operands=(g,), target=world.target_name,
                context={"gate_var": g, "gate_level": float(lev)},
                effect=ev1["effect"], p_value=ev1["p"], complexity=2,
                discovery_mode="anomaly",
                params={"engine": "robust_zscore", "robust_z": robust_z}))
            # engine 2: isolation forest anomaly score on the residual
            iso = IsolationForest(n_estimators=100, contamination="auto",
                                  random_state=0)
            scores = -iso.fit(resid.reshape(-1, 1)).score_samples(
                resid.reshape(-1, 1))     # higher = more anomalous
            ev2 = partial_evidence(y, scores * (ind.astype(float) * 2 - 1),
                                   ctrl_cols)
            out.append(Claim(
                id="ANOMALY:%s=%g|isoforest" % (g, lev), operator="ANOMALY",
                operands=(g,), target=world.target_name,
                context={"gate_var": g, "gate_level": float(lev)},
                effect=ev2["effect"], p_value=ev2["p"], complexity=2,
                discovery_mode="anomaly",
                params={"engine": "isolation_forest",
                       "mean_score_in_group": float(scores[ind].mean()),
                       "mean_score_out_group": float(scores[~ind].mean())}))
    return out


def scan_graph_motif(world):
    y = world.target
    out = []
    # "planted_motifs" added 2026-09-15 (owner directive, extending the
    # boss6/graph_motif truth-declaration fix, paired with the
    # null_graph_rewire fix in nulls.py): graph_motif declares its true
    # object over planted_motifs (the achieved excess after degree-
    # preserving rewiring), not motif_count (the raw count) -- this scanner
    # never tested it. oracle.py is a scorer, not a generator (see
    # MANIFEST.frozen.json supersession chain), so this does not move any
    # world fingerprint by itself.
    for k in ("planted_motifs", "motif_count", "label_rate", "mean_degree", "D0"):
        if k not in world.latents:
            continue
        f = np.asarray(world.latents[k], dtype=float)
        ctrl = []
        for nk in ("composition", "density"):
            if nk in world.nuisance:
                v = np.asarray(world.nuisance[nk], dtype=float)
                ctrl.extend([v, v ** 2, v ** 3])
        ev = partial_evidence(y, f, ctrl)
        out.append(Claim(id="GRAPH:%s" % k, operator="GRAPH", operands=(k,),
                         target=world.target_name, effect=ev["effect"],
                         p_value=ev["p"], complexity=2,
                         discovery_mode="relational"))
    return out


def scan_intervention_edges(world):
    """Regime-conditional edges: is X -> Y present only under some regime?"""
    y = world.target
    if world.interventions is None:
        return []
    assign = np.asarray(world.interventions.assignment)
    names = _scan_names(world)
    out = []
    for r in sorted(set(assign.tolist())):
        g = (assign == r).astype(float)
        for k in names:
            x = np.asarray(world.latents[k], dtype=float)
            f = g * zscore(x)
            ctrl = [x, g] + [np.asarray(world.latents[o], dtype=float)
                             for o in names if o != k][:4]
            ev = partial_evidence(y, f, ctrl)
            out.append(Claim(id="CAUSE:%s|%s" % (k, r), operator="CAUSE",
                             operands=(k,), target=world.target_name,
                             context={"regime": r}, effect=ev["effect"],
                             p_value=ev["p"], complexity=3,
                             discovery_mode="mechanism"))
    return out


# --------------------------------------------------------------------------
# non-relational specialists -- these return a recovery dict, not claims
# --------------------------------------------------------------------------
def specialist_manifold(world, seed=0, n_neighbors=12):
    """Recover the chart coordinate and the branch label from the OBSERVED
    modality, not from the latent that defines them.

    The branch label of a branching manifold is a *connectivity* property, not
    a centroid property: the three spiral arms interleave in radius, so a
    Euclidean centroid method partitions the embedding by radius and recovers
    almost nothing (ARI ~ 0.19 on the reference world). The appropriate
    specialist is a graph-Laplacian method on a kNN affinity, which follows the
    manifold rather than cutting across it. The centroid result is kept in the
    report as `branch_ari_kmeans`: the gap between the two is the evidence that
    this family is genuinely curved rather than a linearly separable mixture
    wearing a manifold label, and a family where the two agree would not be
    testing what it claims to test.
    """
    X = world.modalities["X"]
    t = np.asarray(world.latents["t"], dtype=float)
    br = np.asarray(world.latents["branch"], dtype=int)
    n = len(t)
    K = int(br.max()) + 1
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    knn = KNeighborsRegressor(n_neighbors=10).fit(X[tr], t[tr])
    pred = knn.predict(X[te])
    ss = 1.0 - float(((t[te] - pred) ** 2).sum()
                     / max(((t[te] - t[te].mean()) ** 2).sum(), 1e-12))
    lin = ridge_r2(X, t, alpha=1.0, seed=seed)
    km_ari = float(adjusted_rand_score(
        br, KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(X)))
    sc = SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                            n_neighbors=n_neighbors, random_state=seed,
                            assign_labels="kmeans")
    ari = float(adjusted_rand_score(br, sc.fit_predict(X)))
    return {"chart_r2_knn": ss, "chart_r2_linear": lin, "branch_ari": ari,
            "branch_ari_kmeans": km_ari, "n_neighbors": n_neighbors,
            "recovered": bool(ss >= 0.7 and ari >= 0.5),
            "criterion": "kNN R2 on t >= 0.7 and spectral branch ARI >= 0.5"}


def specialist_anomaly(world, seed=0):
    X = world.modalities["X"]
    lab = np.asarray(world.latents["is_anomaly"], dtype=int)
    mu = X.mean(0)
    C = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])
    Ci = np.linalg.inv(C)
    d = np.einsum("ij,jk,ik->i", X - mu, Ci, X - mu)
    auc = float(roc_auc_score(lab, d)) if lab.sum() else float("nan")
    return {"mahalanobis_auc": auc, "prevalence": float(lab.mean()),
            "recovered": bool(auc >= 0.90),
            "criterion": "Mahalanobis AUC >= 0.90"}


def specialist_mixture(world, seed=0):
    X = world.modalities["X"]
    lab = np.asarray(world.latents["component"], dtype=int)
    K = int(world.meta.get("K", lab.max() + 1))
    gm = GaussianMixture(n_components=K, covariance_type="diag", n_init=3,
                         random_state=seed).fit(X)
    ari = float(adjusted_rand_score(lab, gm.predict(X)))
    bics = {k: float(GaussianMixture(n_components=k, covariance_type="diag",
                                     n_init=2, random_state=seed).fit(X).bic(X))
            for k in range(max(2, K - 2), K + 3)}
    k_bic = int(min(bics, key=bics.get))
    return {"ari_at_true_K": ari, "K_true": K, "K_selected_by_bic": k_bic,
            "bic": bics, "recovered": bool(ari >= 0.60),
            "criterion": "ARI at the true K >= 0.60"}


def specialist_symmetry_oracle(world, seed=0):
    """Is the conserved quantity recoverable and the group parameter not?

    ORACLE. This specialist is handed the answer, not a search space: it reads
    `world.latents["r"]` -- the invariant itself -- and `world.latents["theta"]`
    -- the group parameter -- and then asks whether the observed modality
    predicts the invariant it was already given. A symmetry-discovery method
    (LaLiGAN and its kin) has to *find* the invariant and the group action; this
    one is told both, and only its regression is a search.

    It is therefore excluded from `max_j` by default and reported only in the
    `max_j_including_oracles` column: a specialist handed the answer must not
    set the bar MORPHEUS is compared against. See
    `dbench/PREDECLARATION.external_specialists.md`. The old name
    `specialist_symmetry` is kept as an alias so no existing caller breaks.
    """
    X = world.modalities["X"]
    r = np.asarray(world.latents["r"], dtype=float)
    th = np.asarray(world.latents["theta"], dtype=float)
    y = world.target
    r2_r = ridge_r2(X, r, alpha=1.0, seed=seed)
    n = len(r)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    knn = KNeighborsRegressor(n_neighbors=15).fit(X[tr], r[tr])
    pr = knn.predict(X[te])
    r2_knn = 1.0 - float(((r[te] - pr) ** 2).sum()
                         / max(((r[te] - r[te].mean()) ** 2).sum(), 1e-12))
    # Y should be explained by the invariant and not by the group parameter
    ev_r = partial_evidence(y, r ** 2, [])
    ev_th = partial_evidence(y, np.cos(th), [r ** 2])
    return {"conserved_r2_linear": r2_r, "conserved_r2_knn": r2_knn,
            "p_invariant": ev_r["p"], "p_group_parameter": ev_th["p"],
            "recovered": bool(r2_knn >= 0.6 and ev_r["p"] < 1e-6
                              and ev_th["p"] > 0.01),
            "is_oracle": True,
            "oracle_reason": "reads world.latents['r'] (the invariant) and "
                             "world.latents['theta'] (the group parameter); a "
                             "symmetry-discovery method has to find both",
            "excluded_from_max_j": True,
            "criterion": "kNN R2 on ||z|| >= 0.6, invariant significant, "
                         "group parameter not"}


#: Deprecated alias. `specialist_symmetry` is `specialist_symmetry_oracle`;
#: the name changed so that a reader of a results table cannot mistake an
#: answer-fed control for a discovery method.
specialist_symmetry = specialist_symmetry_oracle


def specialist_ode_oracle(world, threshold=0.05, n_iter=12):
    """Sequentially thresholded least squares on a cubic library (SINDy-style).

    ORACLE. This is SINDy handed its coordinates, not SINDy discovering them:
    it reads `world.latents["x"]` and `["y"]` -- the true state -- and fits a
    sparse regression in coordinates it never had to find. A real SINDy-AE
    (Champion et al. 2019) has to identify the coordinates from the observed
    modality first, and that is the hard half of the problem.

    Kept as a control -- it says what the sparse regression achieves once the
    coordinate problem is solved, which is the right upper reference for
    `specialists_external.specialist_sindy_ae` -- but excluded from `max_j` by
    default and reported only in the `max_j_including_oracles` column. The old
    name `specialist_ode` is kept as an alias so no existing caller breaks.
    """
    x = np.asarray(world.latents["x"], dtype=float)
    y = np.asarray(world.latents["y"], dtype=float)
    dt = world.meta["dt"]
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    lib, names = _poly_library(x, y)
    out = {}
    for tgt, d in (("dx", dx), ("dy", dy)):
        xi = _stlsq(lib, d, threshold, n_iter)
        active = {names[i]: float(xi[i]) for i in range(len(names))
                  if abs(xi[i]) > 1e-8}
        out[tgt] = active
    truth = world.meta["coefs"]
    ok, detail = True, {}
    for state, terms in truth.items():
        tgt = "d" + state
        got = out[tgt]
        want = set(terms)
        detail[tgt] = {"recovered_terms": sorted(got), "true_terms": sorted(want),
                       "coef_error": {t: (abs(got.get(t, 0.0) - terms[t]))
                                      for t in terms}}
        ok = ok and (set(got) == want) and all(
            abs(got.get(t, 0.0) - terms[t]) <= 0.15 * abs(terms[t]) + 0.05
            for t in terms)
    return {"equations": out, "detail": detail, "recovered": bool(ok),
            "is_oracle": True,
            "oracle_reason": "reads world.latents['x'] and ['y'] -- the true "
                             "state; SINDy handed its coordinates rather than "
                             "SINDy-AE discovering them",
            "excluded_from_max_j": True,
            "criterion": "exact sparsity pattern and coefficients within "
                         "15% + 0.05"}


#: Deprecated alias. `specialist_ode` is `specialist_ode_oracle`; the name
#: changed so that a reader of a results table cannot mistake an answer-fed
#: control for a discovery method.
specialist_ode = specialist_ode_oracle


def _poly_library(x, y):
    cols = [np.ones_like(x), x, y, x * x, x * y, y * y,
            x ** 3, x ** 2 * y, x * y ** 2, y ** 3]
    names = ["1", "x", "y", "x^2", "xy", "y^2", "x^3", "x^2y", "xy^2", "y^3"]
    return np.column_stack(cols), names


def _stlsq(A, b, threshold, n_iter):
    xi = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(n_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0.0
        big = ~small
        if not big.any():
            break
        xi[big] = np.linalg.lstsq(A[:, big], b, rcond=None)[0]
    return xi


def specialist_design(world, n_mc=40000, seed=0):
    """Estimate the expected information gain by Monte Carlo and compare with
    the quadrature answer stored as ground truth."""
    m = world.meta
    mu = np.array(m["mu"])
    sd = float(m["sd"])
    prior = np.array(m["prior"])
    rng = np.random.default_rng(seed)
    n_hyp, n_act = mu.shape
    eig = np.zeros(n_act)
    for a in range(n_act):
        h = rng.choice(n_hyp, size=n_mc, p=prior)
        yv = mu[h, a] + sd * rng.standard_normal(n_mc)
        comp = np.exp(-0.5 * ((yv[:, None] - mu[None, :, a]) / sd) ** 2)
        marg = (comp * prior[None, :]).sum(1) / (sd * np.sqrt(2 * np.pi))
        cond = comp[np.arange(n_mc), h] / (sd * np.sqrt(2 * np.pi))
        eig[a] = float(np.mean(np.log(np.maximum(cond, 1e-300))
                               - np.log(np.maximum(marg, 1e-300))))
    a_mc = int(np.argmax(eig))
    a_true = int(m["a_star"])
    truth_eig = np.array(m["eig"])
    order = np.argsort(-truth_eig)
    gap = float(truth_eig[order[0]] - truth_eig[order[1]])
    return {"eig_monte_carlo": eig.tolist(), "eig_quadrature": m["eig"],
            "a_star_monte_carlo": a_mc, "a_star_truth": a_true,
            "best_minus_second_eig": gap,
            "recovered": bool(a_mc == a_true and gap > 0.02),
            "criterion": "MC argmax equals quadrature argmax and the design "
                         "problem is not a near-tie (gap > 0.02 nats)"}


def specialist_confound(world):
    """Recovery here is a refusal as well as a discovery.

    The confounder U must be reported and the confounded variable A must not
    be, once U is conditioned on.
    """
    y = world.target
    U = np.asarray(world.latents["U"], dtype=float)
    A = np.asarray(world.latents["A"], dtype=float)
    W = np.asarray(world.latents["W"], dtype=float)
    ev_u = partial_evidence(y, U, [A, W])
    ev_w = partial_evidence(y, W, [A, U])
    ev_a_marg = partial_evidence(y, A, [])
    ev_a_adj = partial_evidence(y, A, [U])
    return {"p_U_given_A": ev_u["p"], "p_W": ev_w["p"],
            "p_A_marginal": ev_a_marg["p"], "p_A_given_U": ev_a_adj["p"],
            "recovered": bool(ev_u["p"] < 1e-4 and ev_w["p"] < 1e-4
                              and ev_a_marg["p"] < 0.01
                              and ev_a_adj["p"] > 0.01),
            "criterion": "U and W significant; A significant marginally and "
                         "NOT significant given U"}


# ==========================================================================
# composite (boss-world) specialists
# ==========================================================================
# A boss world composes three or four primitives, so no single-operator scan
# can reach its object: the gate hides it from a marginal scan, the threshold
# hides it from a linear one, and the neighbourhood aggregate and the lag both
# have to be found before the conjunction is even visible. What the specialist
# is given is the *template* -- which operators the world composes -- and
# nothing else. Every free parameter (which variable fills each slot, the
# threshold location, the Hill constants, the aggregation kind, the lag, the
# gate level) is searched, and the reported p-value is Bonferroni-corrected by
# the number of programs the search evaluated, so a wider search costs what it
# should. That is the same information the compositional-OOD split gives a
# learned system: the composition is the question, not the answer.
#
# The search is deliberately not clever. A staged search that picks the
# aggregate first by its own marginal would fail on BOSS-2, where the parity
# term has exactly zero marginal signal by construction; a specialist that only
# worked where a marginal shortcut exists would certify the wrong thing.


def _gate_candidates(world, names, max_levels=3):
    """(name, level) pairs a gate term could take, from the latents."""
    out = []
    for k in names:
        v = np.asarray(world.latents[k], dtype=float)
        lv = np.unique(v)
        if 2 <= len(lv) <= max_levels:
            out.extend((k, float(l)) for l in (lv[1:] if len(lv) == 2 else lv))
    return out


def _perm_index(world, kind, rng):
    """One structure-respecting resampling index."""
    n = world.n
    if kind == "block":
        return N.block_permutation_index(n, max(8, int(n ** 0.4)), rng)
    if kind == "torus":
        return _torus_index(world, rng)
    return rng.permutation(n)


def _unit_resid(y, cols):
    """Residual of y on `cols`, scaled to unit norm (zero vector if constant)."""
    r = partial_out(np.asarray(y, dtype=np.float64), list(cols)) if cols else \
        np.asarray(y, dtype=np.float64) - np.mean(y)
    nrm = float(np.linalg.norm(r))
    return r / nrm if nrm > 1e-12 else np.zeros_like(r)


def maxT_permutation(world, y, programs, n_perm=999, seed=0):
    """Westfall-Young max-statistic permutation over a whole search grid.

    Returns, for each program, a family-wise-error-controlled p-value: the
    fraction of resamplings in which the *largest* partial correlation anywhere
    in the grid was at least as extreme as this program's observed one.

    Two properties matter here and neither is available from the alternative.

    Resolution. A Bonferroni-corrected permutation p cannot go below
    `n_programs / (n_perm + 1)`, so a search over a few hundred programs with a
    few hundred permutations can never reject, however strong the object is.
    The max statistic pays for multiplicity inside the null distribution
    instead of multiplying the p-value, so its resolution stays 1/(n_perm+1)
    no matter how wide the search.

    Validity under dependence. The analytic partial-F p-value assumes
    independent rows. On a spatiotemporal world two autocorrelated fields are
    correlated far beyond what that test allows: on BOSS-1 it certified a
    spurious program on a torus-shifted null world at p = 4e-4. Permuting with
    the same operation that defines the world's structured null -- toroidal
    shift in space, moving blocks in time -- keeps the feature's own
    autocorrelation intact under the null, so the reference distribution is the
    one the test actually needs.

    The statistic is the partial correlation between the response and the
    feature after both have been residualised on the program's parent family
    (Freedman-Lane / ter Braak): residualise once, then permute, which is what
    makes a grid-wide null affordable.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    kind = _perm_kind(world)
    FR, RY, obs = [], [], []
    for r in programs:
        ctrl, _ = composite_parent_columns(world, r["terms"])
        fr = _unit_resid(r["feature"], ctrl)
        ry = _unit_resid(y, ctrl)
        FR.append(fr)
        RY.append(ry)
        obs.append(abs(float(fr @ ry)))
    if not FR:
        return [], {}
    FR = np.column_stack(FR)
    RY = np.column_stack(RY)
    obs = np.asarray(obs)
    rng = np.random.default_rng(seed)
    maxnull = np.empty(n_perm)
    for b in range(n_perm):
        idx = _perm_index(world, kind, rng)
        maxnull[b] = float(np.max(np.abs((FR[idx] * RY).sum(axis=0))))
    p = (1.0 + (maxnull[None, :] >= obs[:, None]).sum(axis=1)) / (n_perm + 1.0)
    info = {"kind": kind, "n_perm": int(n_perm), "n_programs": len(obs),
            "max_null_q95": float(np.quantile(maxnull, 0.95)),
            "observed_max": float(obs.max())}
    return [float(v) for v in p], info


def _structural_key(terms):
    """A program's identity up to its continuous parameters.

    Two programs share a key when they have the same operator skeleton on the
    same variables with the same discrete choices (aggregation, lag, gate
    level). tau, K and the Hill exponent are searched parameters, not object
    identity.
    """
    def walk(t):
        op, ops, params, lag = normalize_term(t)
        disc = (params.get("agg"), params.get("gate_level",
                                              params.get("gate_value")), lag)
        return (op, disc, tuple(o if isinstance(o, str) else walk(o)
                                for o in ops))
    return tuple(walk(t) for t in terms)


def _fit_program(world, terms, y, n_perm=0, seed=0):
    """Evidence for one composite program against its full parent family."""
    f = build_composite(world, terms)
    if np.std(f) < 1e-10:
        return None
    ctrl, cnames = composite_parent_columns(world, terms)
    ev = partial_evidence(y, f, ctrl)
    if n_perm:
        ev = dict(ev, p=permutation_p(world, y, f, ctrl, _perm_kind(world),
                                      n_perm=n_perm, seed=seed))
    return {"terms": terms, "feature": f, "evidence": ev,
            "parents": cnames, "label": " * ".join(term_label(t) for t in terms)}


def _search_programs(world, candidates, y, top_k=3, n_perm=199, seed=0):
    """Score every candidate program and charge the search a Bonferroni price.

    The decision statistic is the analytic partial-F p-value against the
    program's full parent family, multiplied by the number of programs the
    search actually evaluated. A permutation p-value is reported alongside for
    the best program, but it is not the decision statistic: its resolution is
    floored at 1/(n_perm+1), and a composite search evaluates hundreds to
    thousands of programs, so a Bonferroni-corrected permutation p can never
    fall below alpha no matter how strong the object is. Raising the permutation
    count until it could would cost more compute than the whole gate.

    What replaces it is the matched structured null. If the analytic p were
    anti-conservative under this world's spatial or temporal dependence, the
    identical search on a null world -- same nuisance, same autocorrelation,
    same neighbourhood graph, relationship destroyed -- would flag programs.
    `run_null_control` runs exactly that, and a family passes the gate only if
    it recovers the object AND the null stays quiet. Calibration is therefore
    demonstrated on the null rather than asserted from the test's assumptions.
    """
    scored = []
    for terms in candidates:
        try:
            r = _fit_program(world, terms, y)
        except (FeatureError, ValueError):
            continue
        if r is not None:
            scored.append(r)
    if not scored:
        return [], 0
    scored.sort(key=lambda r: r["evidence"]["p"])
    n_tested = len(scored)
    # Keep only the best parameterisation of each *structural* program. The
    # grid contains many near-duplicates -- the same operator skeleton on the
    # same variables at three adjacent thresholds -- and reporting them as
    # separate claims would depress precision for a reason that has nothing to
    # do with discovery. Continuous parameters are searched; they are not
    # separate objects.
    seen, distinct = set(), []
    for r in scored:
        k = _structural_key(r["terms"])
        if k in seen:
            continue
        seen.add(k)
        distinct.append(r)
    scored = distinct
    n_structural = len(scored)
    pw, info = maxT_permutation(world, y, scored, n_perm=n_perm, seed=seed)         if n_perm else ([], {})
    out = []
    for rank, r in enumerate(scored[:top_k]):
        p_analytic = float(r["evidence"]["p"])
        rec = dict(r, p_raw=p_analytic,
                   p_bonferroni=float(min(1.0, p_analytic * n_tested)),
                   n_structural_programs=n_structural, maxT=dict(info))
        rec["p_searched"] = float(pw[rank]) if pw else rec["p_bonferroni"]
        out.append(rec)
    return out, n_tested


def _composite_claim(world, r, operator, mode, lag=None, context=None,
                     params=None):
    return Claim(id="COMPOSITE:" + r["label"], operator=operator,
                 operands=composite_operands(r["terms"]),
                 target=world.target_name, lag=lag,
                 context=dict(context or {}),
                 effect=r["evidence"]["effect"], p_value=r["p_searched"],
                 complexity=2 * len(r["terms"]),
                 discovery_mode=mode,
                 params=dict(params or {}, program=r["label"],
                             p_before_search_correction=r["p_raw"],
                             p_bonferroni=r.get("p_bonferroni"),
                             n_structural_programs=r.get(
                                 "n_structural_programs")))


def scan_boss1(world, taus=None, lags=(1, 2, 3, 4), aggs=("mean", "sum", "max"),
              n_perm=199):
    """CONDITION x THRESHOLD x SPACE x LAG.

    Searches gate variable and level, threshold variable and location,
    neighbourhood partner, aggregation and lag.

    The aggs argument used to be hardcoded to mean despite this docstrings own claim
    that aggregation is searched -- so a program that had the right partner
    variable but the wrong neighbourhood summary could never be told apart
    from one that had both right, and neighbourhood aggregation was not a
    scoreable component of composition recovery (dbench.compose_scoring).
    Fixed to actually search it, at the cost of a 3x wider grid.
    """
    y = world.target
    names = _scan_names(world)
    taus = np.linspace(-1.5, 1.5, 13) if taus is None else np.asarray(taus)
    gates = _gate_candidates(world, names)
    cand = []
    for g, lvl in gates:
        for a in names:
            if a == g:
                continue
            for b in names:
                for L in lags:
                    for tau in taus:
                        for agg in aggs:
                            cand.append([
                                {"op": "CONDITION", "operands": [g],
                                 "params": {"gate_level": lvl}},
                                {"op": "THRESHOLD", "operands": [a],
                                 "params": {"tau": float(tau), "temp": 0.12}},
                                {"op": "SPACE", "operands": [b],
                                 "params": {"agg": agg}, "lag": int(L)}])
    hits, n = _search_programs(world, cand, y, n_perm=n_perm)
    out = []
    for r in hits:
        gl = normalize_term(r["terms"][0])
        space_params = normalize_term(r["terms"][2])[2]
        out.append(_composite_claim(
            world, r, "CONDITION", "relational",
            lag=normalize_term(r["terms"][2])[3],
            context={"gate_var": gl[1][0],
                     "gate_value": int(gl[2]["gate_level"])},
            params={"tau": normalize_term(r["terms"][1])[2]["tau"],
                    "agg": space_params.get("agg", "mean"),
                    "n_programs_searched": n}))
    return out


def scan_boss2(world, n_perm=199):
    """XOR x SPACE(majority) x regime gate."""
    y = world.target
    names = _scan_names(world)
    gates = _gate_candidates(world, names)
    cand = []
    for g, lvl in gates:
        for a in names:
            for b in names:
                if a == g or b == g:
                    continue
                for agg in ("majority", "mean"):
                    cand.append([
                        {"op": "CONDITION", "operands": [g],
                         "params": {"gate_level": lvl}},
                        {"op": "XOR", "operands": [
                            a, {"op": "SPACE", "operands": [b],
                                "params": {"agg": agg}}]}])
    hits, n = _search_programs(world, cand, y, n_perm=n_perm)
    out = []
    for r in hits:
        gl = normalize_term(r["terms"][0])
        out.append(_composite_claim(
            world, r, "XOR", "relational",
            context={"gate_var": gl[1][0],
                     "gate_value": int(gl[2]["gate_level"])},
            params={"n_programs_searched": n}))
    return out


def scan_boss3(world, Ks=None, nexps=(2.0, 4.0, 8.0), n_perm=199):
    """SATURATE x SPACE under an intervention regime.

    The gate is a regime of `world.interventions`, not a latent: that is the
    mechanism half of the object, and reading the assignment vector is exactly
    what an intervention-aware method is allowed to do.
    """
    y = world.target
    names = _scan_names(world)
    Ks = np.geomspace(0.3, 3.0, 9) if Ks is None else np.asarray(Ks)
    if world.interventions is None:
        return []
    assign = np.asarray(world.interventions.assignment)
    cand, regime_of = [], {}
    for r in world.interventions.regimes:
        key = "regime__" + str(r)
        regime_of[key] = r
        for d in names:
            for K in Ks:
                for ne in nexps:
                    cand.append([
                        {"op": "CAUSE", "operands": [key],
                         "params": {"gate_level": 1.0}},
                        {"op": "SATURATE", "operands": [
                            {"op": "SPACE", "operands": [d],
                             "params": {"agg": "mean"}}],
                         "params": {"K": float(K), "n": float(ne)}}])
    # the regime indicators are handed to the evaluator as ordinary columns
    w = _with_extra_latents(world, {k: (assign == v).astype(float)
                                    for k, v in regime_of.items()})
    hits, n = _search_programs(w, cand, y, n_perm=n_perm)
    out = []
    for r in hits:
        key = normalize_term(r["terms"][0])[1][0]
        ops = tuple(sorted(set(composite_operands(r["terms"])) - {key}))
        sat = normalize_term(r["terms"][1])[2]
        out.append(Claim(
            id="COMPOSITE:" + r["label"], operator="CAUSE",
            operands=ops + ("treat",) if "treat" in world.latents else ops,
            target=world.target_name,
            context={"regime": regime_of[key]},
            effect=r["evidence"]["effect"], p_value=r["p_searched"],
            complexity=6, discovery_mode="mechanism",
            params={"K": sat["K"], "n": sat["n"], "agg": "mean",
                    "program": r["label"], "n_programs_searched": n,
                    "p_before_search_correction": r["p_raw"],
                    "p_bonferroni": r.get("p_bonferroni"),
                    "n_structural_programs": r.get("n_structural_programs")}))
    return out


def _with_extra_latents(world, extra):
    """A shallow view of `world` with extra oracle columns visible.

    Used only to expose an intervention-regime indicator as a named column so
    the shared feature evaluator can build a gate term from it. Ground truth is
    carried through unchanged; nothing about the object set is altered.
    """
    from dataclasses import replace as _replace
    return _replace(world, latents=dict(world.latents, **extra))


def scan_boss5(world, taus=None, n_perm=199):
    """AND x THRESHOLD -- the relational half of BOSS-5."""
    y = world.target
    names = _scan_names(world)
    taus = np.linspace(-1.5, 1.5, 19) if taus is None else np.asarray(taus)
    cand = []
    for a in names:
        for b in names:
            if a == b:
                continue
            for tau in taus:
                cand.append([
                    {"op": "THRESHOLD", "operands": [a],
                     "params": {"tau": float(tau), "temp": 0.1}},
                    {"op": "ADD", "operands": [b]}])
    hits, n = _search_programs(world, cand, y, n_perm=n_perm)
    out = []
    for r in hits:
        out.append(_composite_claim(
            world, r, "AND", "relational",
            params={"tau": normalize_term(r["terms"][0])[2]["tau"],
                    "n_programs_searched": n}))
    return out


def specialist_boss4(world, seed=0, n_neighbors=15):
    """ODE x MANIFOLD x ANOMALY, in that dependency order.

    The law lives in chart coordinates, so the chart has to be recovered from
    the observed embedding first; the coefficient flips inside a rare regime,
    so the regime has to be detected before either equation is identifiable;
    and only then does sequentially thresholded least squares return the right
    sparsity pattern. The report separates the three so a failure names which
    stage failed, and it also fits a single pooled SINDy model -- the pooled
    fit is what a system that missed the regime would report, and the gap
    between the two is the evidence that this world tests composition rather
    than three independent problems.
    """
    X = world.modalities["X"]
    x = np.asarray(world.latents["x"], dtype=float)
    yv = np.asarray(world.latents["y"], dtype=float)
    reg = np.asarray(world.latents["regime"], dtype=int)
    dt = float(world.meta["dt"])
    n = len(x)

    # -- stage 1: chart ---------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    chart = {}
    for nm, v in (("x", x), ("y", yv)):
        knn = KNeighborsRegressor(n_neighbors=10).fit(X[tr], v[tr])
        pr = knn.predict(X[te])
        chart[nm] = 1.0 - float(((v[te] - pr) ** 2).sum()
                                / max(((v[te] - v[te].mean()) ** 2).sum(), 1e-12))
    chart_ok = min(chart.values()) >= 0.7

    # -- stage 2: regime, from the residual of a pooled law ---------------
    dx, dy = np.gradient(x, dt), np.gradient(yv, dt)
    lib, lib_names = _poly_library(x, yv)
    pooled = {}
    resid = np.zeros(n)
    for tgt, d in (("dx", dx), ("dy", dy)):
        xi = _stlsq(lib, d, 0.05, 12)
        pooled[tgt] = {lib_names[i]: float(xi[i]) for i in range(len(lib_names))
                       if abs(xi[i]) > 1e-8}
        resid = resid + (d - lib @ xi) ** 2
    auc = float(roc_auc_score(reg, resid)) if reg.sum() else float("nan")
    thr = np.quantile(resid, 1.0 - max(float(reg.mean()), 1e-4))
    detected = (resid > thr).astype(int)
    regime_ok = auc >= 0.90

    # -- stage 3: the law, inside and outside the detected regime ---------
    truth = world.meta["coefs"]
    eqs, detail, ok = {}, {}, True
    base_mask = detected == 0
    for tgt, d in (("dx", dx), ("dy", dy)):
        xi = _stlsq(lib[base_mask], d[base_mask], 0.05, 12)
        got = {lib_names[i]: float(xi[i]) for i in range(len(lib_names))
               if abs(xi[i]) > 1e-8}
        eqs[tgt] = got
        want = ({"x": truth["a1"], "y": truth["a2"]} if tgt == "dx"
                else {"x": truth["a3"], "y": truth["a4"]})
        detail[tgt] = {"recovered": sorted(got), "true": sorted(want),
                       "coef_error": {t: abs(got.get(t, 0.0) - want[t])
                                      for t in want}}
        ok = ok and set(got) == set(want) and all(
            abs(got.get(t, 0.0) - want[t]) <= 0.15 * abs(want[t]) + 0.05
            for t in want)
    # the regime-specific coefficient must come back with the opposite sign
    flip = None
    if detected.sum() > 30:
        xi = _stlsq(lib[detected == 1], dx[detected == 1], 0.05, 12)
        flip = float(xi[lib_names.index("y")])
    flip_ok = flip is not None and flip * truth["a2"] < 0
    return {"chart_r2_knn": chart, "regime_auc": auc,
            "regime_prevalence": float(reg.mean()),
            "equations_outside_regime": eqs,
            "pooled_equations_ignoring_regime": pooled,
            "regime_coefficient_dx_y": flip,
            "true_coefficient_dx_y": float(truth["a2"]),
            "detail": detail,
            "stages": {"chart": bool(chart_ok), "regime": bool(regime_ok),
                       "law": bool(ok), "sign_flip": bool(flip_ok)},
            "recovered": bool(chart_ok and regime_ok and ok and flip_ok),
            "criterion": "chart kNN R2 >= 0.7 on both coordinates, regime AUC "
                         ">= 0.90 from the pooled-law residual, exact sparsity "
                         "and coefficients outside the regime, and the "
                         "regime-specific coefficient recovered with the "
                         "opposite sign"}


def specialist_boss5_design(world, n_mc=20000, seed=0):
    """Which action best separates the fitted conjunction from its parents.

    The expected information gain is computed from parameters *estimated by the
    relational search*, not from the world's stored tau and beta. That is the
    difference between doing the design calculation and reading the answer: if
    the threshold search returns the wrong tau, the design ranking is allowed
    to be wrong with it.
    """
    m = world.meta
    actions, hyps = list(m["actions"]), list(m["hypotheses"])
    rel = scan_boss5(world, n_perm=0)
    if not rel:
        return {"recovered": False, "why": "relational stage found nothing"}
    # The specialist has no access to ground truth, so it must use its own
    # best claim -- that is legitimate. What is not legitimate is *reporting*
    # the result as though the claim it used were the matched one. The claim
    # id is recorded here and cross-checked against the matcher in `run_boss5`,
    # because reading a claim's fields by minimum p-value and presenting them
    # as the recovered object is how an unmatched claim gets credited (the
    # same shape as the winner_operands defect found in the Headline-2 table).
    best = rel[0]
    a_name, b_name = best.operands[0], best.operands[-1]
    tau = float(best.params["tau"])
    A = np.asarray(world.latents[a_name], dtype=float)
    B = np.asarray(world.latents[b_name], dtype=float)
    f = soft_threshold(A, tau, 0.1) * B
    beta = float(np.linalg.lstsq(np.column_stack([np.ones(len(f)), f]),
                                 world.target, rcond=None)[0][1])
    noise = float(np.std(world.target - beta * f))

    rng = np.random.default_rng(seed)
    Aq, Bq = rng.standard_normal(n_mc), rng.standard_normal(n_mc)

    def gen(h, a):
        Aa, Bb = Aq.copy(), Bq.copy()
        if a == "clamp_A_high":
            Aa = np.full(n_mc, tau + 1.0)
        elif a == "clamp_A_low":
            Aa = np.full(n_mc, tau - 1.0)
        elif a == "clamp_B":
            Bb = np.full(n_mc, 1.0)
        if h == "and_threshold":
            return beta * soft_threshold(Aa, tau, 0.1) * Bb
        if h == "add_A":
            return beta * 0.5 * Aa
        return beta * 0.5 * Bb

    eig = {}
    for a in actions:
        means = np.array([gen(h, a).mean() for h in hyps])
        sds = np.array([gen(h, a).std() + noise for h in hyps])
        grid = np.linspace(means.min() - 6 * sds.max(),
                           means.max() + 6 * sds.max(), 1501)
        dg = grid[1] - grid[0]
        comp = np.exp(-0.5 * ((grid[None, :] - means[:, None])
                              / sds[:, None]) ** 2)
        comp /= (sds[:, None] * np.sqrt(2 * np.pi))
        pri = np.full(len(hyps), 1.0 / len(hyps))
        marg = (pri[:, None] * comp).sum(0)
        h_marg = -np.sum(np.where(marg > 0, marg * np.log(marg), 0.0)) * dg
        h_cond = float((pri * (0.5 * np.log(2 * np.pi * np.e * sds ** 2))).sum())
        eig[a] = float(h_marg - h_cond)
    a_hat = max(eig, key=eig.get)
    truth_eig = m["eig"]
    order = sorted(truth_eig, key=truth_eig.get, reverse=True)
    gap = float(truth_eig[order[0]] - truth_eig[order[1]])
    return {"claim_used": best.id, "claim_used_operands": list(best.operands),
            "a_star_estimated": a_hat, "a_star_truth": m["a_star"],
            "eig_estimated": eig, "eig_truth": truth_eig,
            "tau_estimated": tau, "tau_truth": float(m["tau"]),
            "beta_estimated": beta,
            "best_minus_second_eig": gap,
            "recovered": bool(a_hat == m["a_star"] and gap > 0.02),
            "criterion": "argmax EIG under the *estimated* threshold equals "
                         "the world's argmax, and the design problem is not a "
                         "near-tie (gap > 0.02 nats)"}


def run_boss5(world, alpha=ALPHA):
    """BOSS-5 needs both halves: the relational object and the design object."""
    claims = scan_boss5(world)
    p = np.array([c.p_value for c in claims]) if claims else np.array([1.0])
    rej, _ = benjamini_hochberg(p, alpha=alpha)
    flagged = [c for c, r in zip(claims, rej) if r]
    m = match_claims(flagged, world.truth)
    rel_ok = any(d["true"] for d in m["matched"])
    des = specialist_boss5_design(world)
    matched_ids = {d["claim_id"] for d in m["matched"] if d["true"]}
    used = des.get("claim_used")
    des["claim_used_was_matched"] = (
        None if used is None else bool(used in matched_ids))
    des["matched_claim_ids"] = sorted(matched_ids)
    if used is not None and matched_ids and used not in matched_ids:
        des["provenance_warning"] = (
            "the design stage estimated its parameters from claim %r, which "
            "the matcher did not pair with any true object; the design result "
            "is therefore not evidence about the planted object" % used)
    return {"kind": "composite specialist", "relational": {
        "n_flagged": len(flagged), "matched": [d["object_id"]
                                               for d in m["matched"]],
        "recovered": bool(rel_ok),
        "claims": [(c.id, float(c.p_value)) for c in claims[:3]]},
        "design": des,
        "recovered": bool(rel_ok and des.get("recovered")),
        "criterion": "the thresholded conjunction survives BH against its full "
                     "parent family AND the design object is recovered from "
                     "the estimated threshold"}


BOSS_RELATIONAL = {
    "boss1_cond_thresh_space_lag": scan_boss1,
    "boss2_xor_space_regime": scan_boss2,
    "boss3_hill_space_cause": scan_boss3,
    # boss6/boss7 use the same native scanner the oracle single-specialist
    # router already selects for these families (specialist_frontier.py's
    # ROUTER_SCANNER: "graph" / "cause"), so gate coverage and the router's
    # own admitted upper bound stay the same specialist.
    "boss6_hill_graph": scan_graph_motif,
    "boss7_causal_hill_lag": scan_intervention_edges,
}


# --------------------------------------------------------------------------
# family -> specialist
# --------------------------------------------------------------------------
def relational_claims(world, family):
    f = family
    if f == "additive_factors":
        return scan_main_effects(world)
    if f in ("pure_and",):
        return scan_main_effects(world) + scan_pairs(world)
    if f in ("invisible_k3", "invisible_k4", "invisible_k5"):
        # The pairwise/main-effect specialist is the correct registration,
        # not a stronger one invented for coverage: LOCKINv2's invisible-
        # k-way instrument is designed so every non-empty proper product is
        # population-invisible to Y, so this specialist is expected to find
        # nothing on these worlds. Its candidates keep it inside the gate's
        # coverage/multiplicity accounting rather than silently excluded.
        return scan_main_effects(world) + scan_pairs(world)
    if f == "linear_combination":
        return scan_lincomb(world)
    if f == "xor":
        return scan_xor(world)
    if f == "threshold":
        return scan_threshold(world)
    if f == "hill":
        return scan_saturation(world)
    if f in ("context_gated", "regime_switch"):
        return scan_gated(world)
    if f == "spatial_interaction":
        return scan_spatial(world)
    if f == "temporal_lag":
        return scan_lags(world)
    if f == "graph_motif":
        return scan_graph_motif(world)
    if f == "intervention_causal":
        return scan_intervention_edges(world)
    if f in BOSS_RELATIONAL:
        return BOSS_RELATIONAL[f](world)
    return []


NONRELATIONAL = {
    "manifold": specialist_manifold,
    "anomaly": specialist_anomaly,
    "mixture": specialist_mixture,
    "hidden_symmetry": specialist_symmetry_oracle,
    "sparse_ode": specialist_ode_oracle,
    "experimental_design": specialist_design,
    "latent_confound": specialist_confound,
    "boss4_ode_manifold_anomaly": specialist_boss4,
}

#: Bank members that are handed the ANSWER rather than a search space, and are
#: therefore excluded from `max_j` by default.
#:
#: The distinction this set encodes: being handed the true *latents* is what
#: every specialist in this module gets, and it is fair -- the specialist still
#: has to find which pair, which lag, which tau, with distractors in the scan.
#: Being handed the solved *coordinate* is different. `specialist_ode_oracle`
#: reads the true state, so the coordinate-discovery half of SINDy never
#: happens; `specialist_symmetry_oracle` reads the invariant and the group
#: parameter, so the symmetry-discovery half never happens. In both cases what
#: remains is a regression, and a regression handed the answer must not set the
#: bar MORPHEUS is compared against.
#:
#: `scan_intervention_edges` is deliberately NOT in this set. It searches over
#: already-named latents, which makes it narrow, but it is not told which edge
#: is present or in which regime -- so it is a fair-but-narrow specialist, not
#: an oracle. A representative interventional CRL (CITRIS/BISCUIT) would be the
#: named comparator there; it is not built.
#:
#: Every member also returns `is_oracle: True` in its own report, so a consumer
#: that never imports this set still sees the flag on the result.
ORACLE_SPECIALISTS = frozenset({
    "hidden_symmetry",       # -> specialist_symmetry_oracle
    "sparse_ode",            # -> specialist_ode_oracle
})

#: The same two, keyed by the name the Headline-2 bank uses for them
#: (`dbench/tests/_h2_specialists.py:NONREL_BANK`), so `max_j` can be filtered
#: on either spelling without hard-coding it at the call site.
ORACLE_BANK_NAMES = frozenset({"symmetry", "sparse_ode"})


def is_oracle_specialist(name_or_result):
    """True if this bank member is handed the answer.

    Accepts a family name, an H2 bank name, or a specialist's own result dict
    (which carries `is_oracle`). Use it rather than a hard-coded name list: a
    specialist that later stops reading the answer should stop being excluded,
    and the flag lives with the specialist.
    """
    if isinstance(name_or_result, dict):
        return bool(name_or_result.get("is_oracle"))
    n = str(name_or_result)
    return n in ORACLE_SPECIALISTS or n in ORACLE_BANK_NAMES


#: The smallest p-value each statistic in the bank can return. An analytic
#: partial-F p has no floor; a permutation p is floored at 1/(n_perm+1).
ANALYTIC_RESOLUTION = 0.0


def claim_resolution(claim) -> float:
    """The smallest p-value this claim's statistic could have returned.

    Reported so that a ranking which crosses specialist kinds can tell the
    difference between "this claim is stronger" and "the other claim's
    statistic saturated". A composite scan's Westfall-Young p is floored at
    1/(n_perm+1) = 0.005 at the default grid, while an analytic partial-F p
    routinely reaches 1e-140, so a raw-p ordering across kinds is not an
    ordering by evidence. On `boss3_hill_space_cause` that is not hypothetical:
    the intervention-edge scan outranks the composite claim on the floor alone.

    This is additive. It changes no verdict: BH and every recovery decision
    continue to use the raw p, which is the right statistic within a kind.
    """
    params = getattr(claim, "params", {}) or {}
    n_perm = params.get("n_perm")
    if n_perm is None:
        mt = params.get("maxT") or {}
        n_perm = mt.get("n_perm") if isinstance(mt, dict) else None
    if n_perm:
        return 1.0 / (float(n_perm) + 1.0)
    if params.get("p_bonferroni") is not None:
        # a max-statistic search reported without its permutation count
        return 1.0 / (199.0 + 1.0)
    return ANALYTIC_RESOLUTION


def run_family(world, family, alpha=ALPHA):
    """Run the appropriate specialist and score it against ground truth."""
    if family == "boss5_design_and_threshold":
        rep = run_boss5(world, alpha=alpha)
        rep["family"] = family
        return rep
    if family in NONRELATIONAL:
        rep = NONRELATIONAL[family](world)
        rep["kind"] = "non-relational specialist"
        rep["family"] = family
        return rep
    claims = relational_claims(world, family)
    if not claims:
        return {"family": family, "kind": "no specialist", "recovered": None}
    p = np.array([c.p_value for c in claims])
    rej, q = benjamini_hochberg(p, alpha=alpha)
    flagged = [c for c, r in zip(claims, rej) if r]
    m = match_claims(flagged, world.truth)
    n_true_hit = sum(1 for d in m["matched"] if d["true"])
    n_true = len([h for h in world.truth.true_objects
                  if h.operator not in ("MANIFOLD", "MIXTURE", "ANOMALY",
                                        "SYMMETRY", "ODE", "DESIGN")])
    return {
        "family": family, "kind": "relational scan",
        "n_candidates": len(claims), "n_flagged": len(flagged),
        "n_true_objects_in_scope": n_true,
        "n_true_recovered": n_true_hit,
        "recall": (n_true_hit / n_true) if n_true else float("nan"),
        "precision": (n_true_hit / len(flagged)) if flagged else float("nan"),
        "flagged_ids": [c.id for c in flagged][:20],
        "recovered": bool(n_true and n_true_hit >= n_true),
        "criterion": "every in-scope planted object flagged after BH at "
                     "alpha=%.3f" % alpha,
    }


def run_null_control(world, family, alpha=ALPHA, n_reps=5, seed0=17):
    """The same specialist on repeated structured nulls: a *rate*, not a draw.

    A single null world cannot decide whether a procedure is calibrated. A test
    that is exactly right at alpha = 0.05 raises at least one flag on one null
    world in twenty; scoring "quiet" on one draw therefore fails a correct
    procedure five per cent of the time and, worse, passes a badly
    anti-conservative one whenever the draw is kind. What the control reports
    instead is the family-wise error rate over `n_reps` independently seeded
    null worlds, together with the exact binomial probability of seeing at
    least that many errors if the procedure really operated at alpha. The
    family passes when that probability does not reject calibration.

    The report deliberately states the power of the check as well: with five
    replicates a procedure would have to be operating near alpha = 0.45 before
    this control could distinguish it from alpha = 0.05, so `n_reps` is raised
    for the freeze run rather than left at the interactive default.
    """
    kinds = N.DEFAULT_NULLS.get(family, ("iid_permute",))
    kind = kinds[0] if kinds else "iid_permute"
    if family in NONRELATIONAL:
        return {"null_kind": kind, "n_flagged": None, "calibrated": None,
                "note": "non-relational specialists are gated by their own "
                        "criterion, not by a flag count"}
    per_rep, errors = [], 0
    for r in range(n_reps):
        try:
            nw, _ = N.make_null(world, kind, seed=seed0 + 101 * r)
        except Exception as exc:
            per_rep.append({"seed": seed0 + 101 * r, "error": repr(exc)})
            continue
        claims = relational_claims(nw, family)
        if not claims:
            per_rep.append({"seed": seed0 + 101 * r, "n_candidates": 0,
                            "n_flagged": 0})
            continue
        p = np.array([c.p_value for c in claims])
        rej, _ = benjamini_hochberg(p, alpha=alpha)
        k = int(rej.sum())
        errors += int(k > 0)
        per_rep.append({"seed": seed0 + 101 * r, "n_candidates": len(claims),
                        "n_flagged": k,
                        "flagged": [c.id for c, x in zip(claims, rej) if x][:3]})
    done = [r for r in per_rep if "error" not in r]
    R = len(done)
    p_binom = float(stats.binom.sf(errors - 1, R, alpha)) if R and errors else 1.0
    return {
        "null_kind": kind, "n_reps": R,
        "n_reps_with_a_flag": errors,
        "familywise_error_rate": (errors / R) if R else None,
        "alpha": alpha,
        "p_binomial_vs_alpha": p_binom,
        "calibrated": bool(p_binom > 0.05) if R else None,
        "n_flagged": int(sum(r.get("n_flagged", 0) for r in done)),
        "per_replicate": per_rep,
        "criterion": "P(Bin(n_reps, alpha) >= n_reps_with_a_flag) > 0.05",
    }


def run_gate(families=None, seed=11, snr="high", verbose=True,
             n_null_reps=5, **kw):
    """The full gate: every high-SNR family, with its null control."""
    from .environments import ALL_WORLDS, build_world
    families = families or list(ALL_WORLDS)
    out = {}
    for fam in families:
        w = build_world(fam, seed, role="proposal", snr=snr, **kw)
        rep = run_family(w, fam)
        rep["null_control"] = run_null_control(w, fam, n_reps=n_null_reps)
        rep["n"] = w.n
        rep["passes"] = bool(rep.get("recovered")) and (
            rep["null_control"].get("calibrated") is not False)
        out[fam] = rep
        if verbose:
            nc = rep["null_control"]
            print("%-30s recovered=%-5s null_FWER=%-6s calibrated=%s"
                  % (fam, rep.get("recovered"),
                     nc.get("familywise_error_rate"), nc.get("calibrated")))
    n_eval = [r for r in out.values() if r.get("recovered") is not None]
    return {
        "seed": seed, "snr": snr, "n_null_reps": n_null_reps,
        # what this gate actually ran against, so a manifest cannot bind it to
        # a different generator set without the mismatch being visible
        "generators_sha256": package_source_sha256(),
        "n_families": len(out),
        "n_evaluated": len(n_eval),
        "n_recovered": sum(1 for r in n_eval if r["recovered"]),
        "n_passing": sum(1 for r in n_eval if r.get("passes")),
        "families_failing": sorted(f for f, r in out.items()
                                   if r.get("recovered") is False),
        "families_miscalibrated_on_null": sorted(
            f for f, r in out.items()
            if r.get("null_control", {}).get("calibrated") is False),
        "families_without_specialist": sorted(
            f for f, r in out.items() if r.get("recovered") is None),
        "per_family": out,
    }
