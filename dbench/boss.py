"""Composite boss worlds: several primitives at once, plus a confounder.

The spec's argument for these is the load-bearing one: "A system that wins only
ten independent toy tasks is not a universal discovery learner. A system that
solves unseen compositions becomes interesting."

Each boss world declares its `composition` -- the sorted tuple of primitive
operators it uses. That tuple is the unit the compositional-OOD split holds
out, so the held-out set is defined over compositions rather than over worlds,
and a new seed of a held-out composition is still held out.

BOSS-1 is the spec's own example, implemented literally:

    Y_{i,t} = 1[C_i = 1] * sigma(A_{i,t} - tau) * Agg_{j in N(i)} B_{j,t-2} + eps

with a nuisance Q that induces an artificial A <-> Y relationship. It tests
conditionality, thresholding, space, time, higher-order interaction and
confounding simultaneously.
"""
from __future__ import annotations

import numpy as np

from . import artifacts as art
from .families import (_B, _H, _finish, _count_labelled_triangles,
                       _install_edge)
from .util import (ar1, gp_field, grid_coords, hill, radius_neighbors,
                   soft_threshold, zscore)
from .world import Hypothesis, Interventions, Structure

# --------------------------------------------------------------------------
# panel plumbing shared by the spatiotemporal bosses
# --------------------------------------------------------------------------


def _panel(side, T, radius):
    """Rows indexed (t, cell). Returns coords, time, unit and row neighbours.

    A row's neighbourhood is its spatial neighbours *at the same time step*,
    so SPACE and LAG stay separable operators rather than one blurred
    spacetime kernel.
    """
    cell_xy = grid_coords(side)
    n_cells = len(cell_xy)
    cell_nb = radius_neighbors(cell_xy, radius)
    n = n_cells * T
    coords = np.tile(cell_xy, (T, 1))
    time = np.repeat(np.arange(T), n_cells)
    unit = np.tile(np.arange(n_cells), T)
    row_nb = []
    for t in range(T):
        base = t * n_cells
        for c in range(n_cells):
            row_nb.append(cell_nb[c] + base)
    return coords, time, unit, row_nb, n_cells


def _lag_by_unit(x, unit, time, n_cells, lag):
    """x at (unit, time - lag), zero before the series starts."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    T = int(time.max()) + 1
    for t in range(lag, T):
        out[t * n_cells:(t + 1) * n_cells] = x[(t - lag) * n_cells:(t - lag + 1) * n_cells]
    return out


# ==========================================================================
# BOSS-1: CONDITION x THRESHOLD x SPACE x LAG, with a confounder
# ==========================================================================
def make_boss1(seed=0, role="proposal", side=16, T=24, snr="high", tau=0.6,
               temp=0.12, lag=2, radius=1.5, artifacts=(), d_obs=48,
               noise=0.6, q_strength=0.9, n=None):
    """Y_{i,t} = 1[C_i=1] * sigma((A_{i,t}-tau)/T) * mean_{j in N(i)} B_{j,t-2}.

    Q is a static per-cell nuisance that raises both A and Y, so A and Y are
    marginally associated for a reason that has nothing to do with the object.
    The object is only visible after conditioning on Q, on C, on A, and on the
    lagged neighbourhood mean of B separately.
    """
    coords, time, unit, row_nb, n_cells = _panel(side, T, radius)
    nrows = len(coords)
    b = _B("boss1_cond_thresh_space_lag", seed, role, nrows, "spatial", snr)
    b.structure = Structure(kind="spatiotemporal", coords=coords, time=time,
                            unit=unit, neighbors=row_nb,
                            meta={"side": side, "T": T, "radius": radius})

    cell_xy = coords[:n_cells]
    C_cell = (b.rs.random(n_cells) < 0.5).astype(float)
    Q_cell = zscore(gp_field(cell_xy, b.rn, length_scale=3.0))
    C = np.tile(C_cell, T)
    Q = np.tile(Q_cell, T)

    A = np.concatenate([zscore(gp_field(cell_xy, b.rs, length_scale=2.0))
                        for _ in range(T)])
    A = A + q_strength * Q                      # Q -> A
    B = np.concatenate([zscore(gp_field(cell_xy, b.rs, length_scale=2.0))
                        for _ in range(T)])

    aggB = np.zeros(nrows)
    for i, nb in enumerate(row_nb):
        aggB[i] = float(B[nb].mean()) if len(nb) else 0.0
    aggB_lag = _lag_by_unit(aggB, unit, time, n_cells, lag)

    thr = soft_threshold(A, tau, temp)
    f = C * thr * aggB_lag
    beta = 2.2 * b.scale
    valid = time >= lag
    Y = beta * f + q_strength * Q + noise * b.rn.standard_normal(nrows)
    Y[~valid] = q_strength * Q[~valid] + noise * b.rn.standard_normal(int((~valid).sum()))

    b.lat.update(A=A, B=B, C=C)
    for _nm, _v in (("aggB", aggB), ("aggB_lag", aggB_lag), ("thrA", thr)):
        b.derive(_nm, _v)
    b.lat["D0"] = b.rs.standard_normal(nrows)
    b.lat["D1"] = np.concatenate([zscore(gp_field(cell_xy, b.rs, length_scale=2.0))
                                  for _ in range(T)])
    b.nui["Q"] = Q
    b.nui["density"] = zscore(np.array([len(v) for v in row_nb], dtype=float))
    b.nui["t_index"] = zscore(time.astype(float))
    b.env_labels["unit"] = unit
    b.env_labels["time"] = time
    b.env_labels["valid"] = valid.astype(int)
    b.Y = Y

    b.add(_H("BOSS1:C&thr(A)&aggB_lag", "relational", "CONDITION",
             ("A", "B", "C"), lag=lag,
             params={"tau": tau, "temp": temp, "agg": "mean", "radius": radius,
                     "lag": lag, "beta": float(beta),
                     "composition": ["CONDITION", "THRESHOLD", "SPACE", "LAG"],
                     # Composite reconstruction (dbench.features.build_composite):
                     # `build_feature`'s flat CONDITION path (gate * z(A) * z(B))
                     # cannot represent this object -- it never applies the
                     # THRESHOLD/SPACE/LAG chain named in `composition` above,
                     # which is descriptive metadata only. This `terms` list is
                     # the machine-readable version, verified to reproduce the
                     # stored generative signal (`C * thrA * aggB_lag`) at
                     # abs_corr = 1.000000 (2026-09-15, Bench-Owner + Arch-Owner).
                     "terms": [
                         {"op": "CONDITION", "operands": ["C"],
                          "params": {"gate_value": 1}},
                         {"op": "THRESHOLD", "operands": ["A"],
                          "params": {"tau": tau, "temp": temp}},
                         {"op": "SPACE", "operands": ["B"],
                          "params": {"agg": "mean"}, "lag": lag},
                     ]},
             context={"gate_var": "C", "gate_value": 1},
             parents=[("A",), ("B",), ("C",), ("aggB",), ("aggB_lag",),
                      ("thrA",), ("A", "C"), ("A", "B"), ("B", "C")],
             notes="conditional thresholded lagged spatial interaction"),
          feature=f,
          parent_features=[A, aggB_lag, C, thr, b.nui["Q"], b.nui["density"]])
    b.objs.append(Hypothesis(
        id="ART:boss1_Q_induces_A", mode="mechanism", operator="CONFOUND",
        operands=("A",), target="Y", true=False,
        artifact_class="nuisance_confound", parents=(("Q",),),
        params={"confounder": "Q", "q_strength": q_strength},
        notes="Q raises both A and Y; the marginal A~Y association is real and "
              "is not an effect of A"))
    b.meta.update(composition=("CONDITION", "LAG", "SPACE", "THRESHOLD"),
                  side=side, T=T, tau=tau, lag=lag, n_valid=int(valid.sum()))
    return _finish(b, artifacts, d_obs=d_obs,
                   name="boss1/%s/s%d" % (role, seed))


# ==========================================================================
# BOSS-2: XOR x SPACE x ANOMALY(regime)
# ==========================================================================
def make_boss2(seed=0, role="proposal", side=14, T=30, snr="high",
               radius=1.5, p_stay=0.9, artifacts=(), d_obs=48, noise=0.6,
               n=None):
    """A spatial XOR that is switched on only inside a latent temporal regime.

    Y_{i,t} = 1[s_t = 1] * XOR(A_i, majority_{j in N(i)} B_j)

    Both constituents are exactly marginally uninformative -- A is a fair coin
    and the neighbourhood channel is a median split, so the parity term is
    orthogonal to each of them -- and the regime occupies a minority of time
    steps, so pooling across time dilutes an already-zero marginal further.
    """
    coords, time, unit, row_nb, n_cells = _panel(side, T, radius)
    nrows = len(coords)
    b = _B("boss2_xor_space_regime", seed, role, nrows, "spatial", snr)
    b.structure = Structure(kind="spatiotemporal", coords=coords, time=time,
                            unit=unit, neighbors=row_nb,
                            meta={"side": side, "T": T, "radius": radius})
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = s[t - 1] if b.rs.random() < p_stay else 1 - s[t - 1]
    S = np.repeat(s, n_cells).astype(float)
    A = (b.rs.random(nrows) < 0.5).astype(float)
    # B is continuous and the neighbourhood channel is binarised at its own
    # median. A Bernoulli B with a `mean > 0.5` majority rule is *not*
    # marginally free: neighbourhoods on a radius-1.5 lattice have two, three
    # or four members, ties on the even sizes all fall to zero, and the
    # resulting P(majB = 1) sits below one half. That imbalance leaks straight
    # through the parity term and gave A a marginal |r| of 0.08 against the
    # response -- small, and quite enough for a screening method to find the
    # world's own constituent without ever composing anything. A median split
    # of a continuous aggregate is balanced by construction, and it is exactly
    # the rule `features._binarize` applies, so the specialist reproduces the
    # channel instead of approximating it.
    B = b.rs.standard_normal(nrows)
    aggB = np.array([B[nb].mean() if len(nb) else 0.0 for nb in row_nb])
    majB = (aggB > np.median(aggB)).astype(float)
    x = (A != majB).astype(float)
    f = S * (x - 0.5)
    beta = 2.4 * b.scale
    b.lat.update(A=A, B=B, regime=S)
    b.derive("aggB", aggB)
    b.derive("majB", majB)
    b.lat["D0"] = b.rs.standard_normal(nrows)
    b.nui["density"] = zscore(np.array([len(v) for v in row_nb], dtype=float))
    b.nui["t_index"] = zscore(time.astype(float))
    b.env_labels["unit"] = unit
    b.env_labels["time"] = time
    b.Y = beta * f + noise * b.rn.standard_normal(nrows)
    b.add(_H("REGIME:s", "anomaly", "ANOMALY", ("regime",),
             params={"n_states": 2, "p_stay": p_stay},
             notes="latent temporal regime gating the whole relation"))
    b.add(_H("BOSS2:xor(A,majB)|regime", "relational", "XOR",
             ("A", "B", "regime"),
             params={"agg": "mean", "binarize": "median", "radius": radius,
                     "beta": float(beta),
                     "composition": ["XOR", "SPACE", "CONDITION", "ANOMALY"],
                     # See BOSS1's own `terms` comment: `build_feature`'s flat
                     # XOR path takes exactly two operands and cannot read the
                     # declared 3-operand (A, B, regime) composite at all.
                     # Verified abs_corr = 1.000000 against the stored
                     # `regime * (XOR(A, majB) - 0.5)` signal.
                     "terms": [
                         {"op": "CONDITION", "operands": ["regime"],
                          "params": {"gate_value": 1}},
                         {"op": "XOR", "operands": ["A", "majB"]},
                     ]},
             context={"gate_var": "regime", "gate_value": 1},
             parents=[("A",), ("B",), ("majB",), ("regime",), ("A", "B"),
                      ("A", "regime")],
             notes="spatial parity active only inside the regime"),
          feature=f, parent_features=[A, majB, S, b.nui["density"]])
    b.meta.update(composition=("ANOMALY", "CONDITION", "SPACE", "XOR"),
                  regime_fraction=float(s.mean()), side=side, T=T)
    return _finish(b, artifacts, d_obs=d_obs,
                   name="boss2/%s/s%d" % (role, seed))


# ==========================================================================
# BOSS-3: SATURATE x SPACE x CAUSE (intervention-specific)
# ==========================================================================
def make_boss3(seed=0, role="proposal", side=34, snr="high", radius=2.1,
               K=0.9, nexp=4.0, artifacts=(), d_obs=48, noise=0.5, n=None):
    """A Hill response to a neighbourhood dose, active only under do(treat).

    Y_i = 1[regime = do_treat] * Hill(mean_{j in N(i)} D_j) * beta

    In the observational regime the same neighbourhood dose has no effect, so
    an environment-pooled fit sees a diluted, misshapen curve. Recovery
    requires the intervention context and the saturating form together.
    """
    coords = grid_coords(side)
    nrows = len(coords)
    b = _B("boss3_hill_space_cause", seed, role, nrows, "spatial", snr)
    nb = radius_neighbors(coords, radius)
    b.structure = Structure(kind="spatial", coords=coords, neighbors=nb,
                            meta={"side": side, "radius": radius})
    assign = np.where(b.rs.random(nrows) < 0.5, "do_treat", "obs")
    D = np.exp(0.6 * zscore(gp_field(coords, b.rs, length_scale=2.5)))
    aggD = b.structure.aggregate(D, "mean")
    gate = (assign == "do_treat").astype(float)
    h = hill(aggD, K=K, nexp=nexp)
    f = gate * h
    beta = 2.4 * b.scale
    b.lat.update(D=D)
    b.derive("aggD", aggD)
    b.derive("hillD", h)
    b.lat["treat"] = gate
    b.lat["D0"] = b.rs.standard_normal(nrows)
    b.nui["spatial_nuisance"] = zscore(gp_field(coords, b.rn, length_scale=5.0))
    b.nui["density"] = zscore(b.structure.n_neighbors().astype(float))
    b.interventions = Interventions(
        assignment=assign, targets={"do_treat": ("treat",), "obs": ()},
        kind={"do_treat": "do", "obs": "none"},
        values={"do_treat": {"treat": 1.0}})
    b.env_labels["regime"] = assign
    b.Y = (beta * f + 0.7 * b.nui["spatial_nuisance"]
           + noise * b.rn.standard_normal(nrows))
    b.add(_H("BOSS3:hill(aggD)|do_treat", "mechanism", "CAUSE",
             ("D", "treat"),
             params={"K": K, "n": nexp, "agg": "mean", "radius": radius,
                     "beta": float(beta),
                     "composition": ["CAUSE", "SATURATE", "SPACE"],
                     # See BOSS1's own `terms` comment. Verified abs_corr =
                     # 1.000000 against the stored `treat * hillD` signal
                     # (hillD = hill(aggD, K, nexp)).
                     "terms": [
                         {"op": "CONDITION", "operands": ["treat"],
                          "params": {"gate_value": 1}},
                         {"op": "SATURATE",
                          "operands": [{"op": "SPACE", "operands": ["D"],
                                       "params": {"agg": "mean"}}],
                          "params": {"K": K, "n": nexp}},
                     ]},
             context={"regime": "do_treat"},
             parents=[("D",), ("aggD",), ("treat",), ("D", "treat")],
             notes="saturating response to a neighbourhood dose, present only "
                   "under intervention"),
          feature=f, parent_features=[aggD, gate, b.nui["density"],
                                      b.nui["spatial_nuisance"]])
    b.meta.update(composition=("CAUSE", "SATURATE", "SPACE"), side=side, K=K,
                  nexp=nexp)
    return _finish(b, artifacts, d_obs=d_obs,
                   name="boss3/%s/s%d" % (role, seed))


# ==========================================================================
# BOSS-4: ODE x MANIFOLD x ANOMALY
# ==========================================================================
def make_boss4(seed=0, role="proposal", n=6000, snr="high", dt=0.004,
               artifacts=(), d_obs=48, n_switch=3):
    """A sparse ODE whose coefficient switches inside rare regimes, observed
    only through a nonlinear embedding of the state.

    The state (x, y) follows dx = a1 x + a2 y, dy = a3 x + a4 y with a rare
    regime in which a2 changes sign. Observations are a curved embedding of
    (x, y), so the law is only visible after the manifold chart is recovered.
    Three objects: the chart, the regime, and the governing equation.
    """
    b = _B("boss4_ode_manifold_anomaly", seed, role, n, "temporal", snr)
    base = {"a1": -0.2, "a2": 1.6, "a3": -1.6, "a4": -0.2}
    reg = np.zeros(n, dtype=int)
    starts = np.sort(b.rs.choice(np.arange(n // 10, n - n // 10),
                                 size=n_switch, replace=False))
    width = max(60, n // 40)
    for s0 in starts:
        reg[s0:s0 + width] = 1
    x = np.empty(n)
    y = np.empty(n)
    x[0], y[0] = 1.0, 0.2
    for i in range(n - 1):
        a2 = base["a2"] * (-1.0 if reg[i] else 1.0)
        dx = base["a1"] * x[i] + a2 * y[i]
        dy = base["a3"] * x[i] + base["a4"] * y[i]
        x[i + 1] = x[i] + dt * dx
        y[i + 1] = y[i] + dt * dy
    # curved embedding: the manifold chart
    E = np.column_stack([x, y, 0.6 * (x ** 2 - y ** 2), 0.8 * x * y])
    b.structure = Structure(kind="temporal", time=np.arange(n),
                            meta={"dt": dt})
    b.lat.update(x=x, y=y, chart=E)
    b.lat["regime"] = reg.astype(float)
    b.lat["dx"] = np.gradient(x, dt)
    b.lat["dy"] = np.gradient(y, dt)
    b.lat["D0"] = b.rs.standard_normal(n)
    b.Y = None
    b.add(_H("MANIFOLD:xy", "entity", "MANIFOLD", ("x", "y"), target="X",
             params={"intrinsic_dim": 2, "embedding": "quadratic"},
             notes="2-D chart; the law lives in these coordinates"))
    b.add(_H("REGIME:switch", "anomaly", "ANOMALY", ("regime",), target="X",
             params={"n_regimes": 2, "n_episodes": n_switch,
                     "prevalence": float(reg.mean())},
             notes="rare regime in which a coefficient of the law flips sign"))
    b.add(_H("ODE:dx", "law", "ODE", ("x", "y"), target="dx",
             params={"a1": base["a1"], "a2": base["a2"], "dt": dt,
                     "regime_effect": {"a2": -base["a2"]},
                     "composition": ["ODE", "MANIFOLD", "ANOMALY"]},
             notes="linear law in chart coordinates with a regime-dependent "
                   "coefficient"))
    b.add(_H("ODE:dy", "law", "ODE", ("x", "y"), target="dy",
             params={"a3": base["a3"], "a4": base["a4"], "dt": dt},
             notes="regime-invariant second equation"))
    b.meta.update(composition=("ANOMALY", "MANIFOLD", "ODE"), coefs=base,
                  dt=dt, regime_prevalence=float(reg.mean()))
    return _finish(b, artifacts, d_obs=d_obs, obs_noise=0.02,
                   name="boss4/%s/s%d" % (role, seed))


# ==========================================================================
# BOSS-5: DESIGN over an AND x THRESHOLD world
# ==========================================================================
def make_boss5(seed=0, role="proposal", n=3000, snr="high", tau=0.8,
               temp=0.1, artifacts=(), d_obs=48, noise=0.8, n_actions=4):
    """Experimental discovery on top of a relational world.

    Y = beta * sigma((A - tau)/T) * B: a thresholded conjunction, so both
    constituents are near-marginally uninformative. Four candidate
    interventions are available -- clamp A high, clamp A low, clamp B, or
    observe -- and the ground-truth object is which one best discriminates
    the AND-with-threshold hypothesis from its two-way parents.

    The expected information gain is computed by enumeration over the
    hypothesis set at generation time, so a*(the answer) is exact.
    """
    b = _B("boss5_design_and_threshold", seed, role, n, "tabular", snr)
    A = b.rs.standard_normal(n)
    B = b.rs.standard_normal(n)
    thr = soft_threshold(A, tau, temp)
    f = thr * B
    beta = 2.4 * b.scale
    b.lat.update(A=A, B=B)
    b.derive("thrA", thr)
    b.lat["D0"] = b.rs.standard_normal(n)
    b.lat["D1"] = b.rs.standard_normal(n)
    b.Y = beta * f + noise * b.rn.standard_normal(n)

    # --- exact design calculation over a discrete hypothesis set ----------
    actions = ["observe", "clamp_A_high", "clamp_A_low", "clamp_B"][:n_actions]
    hyps = ["and_threshold", "add_A", "add_B"]
    m = 4000
    ra = np.random.default_rng(12345)
    Aq, Bq = ra.standard_normal(m), ra.standard_normal(m)

    def gen(h, a):
        Aa, Bb = Aq.copy(), Bq.copy()
        if a == "clamp_A_high":
            Aa = np.full(m, tau + 1.0)
        elif a == "clamp_A_low":
            Aa = np.full(m, tau - 1.0)
        elif a == "clamp_B":
            Bb = np.full(m, 1.0)
        if h == "and_threshold":
            return beta * soft_threshold(Aa, tau, temp) * Bb
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
        comp = np.exp(-0.5 * ((grid[None, :] - means[:, None]) / sds[:, None]) ** 2)
        comp /= (sds[:, None] * np.sqrt(2 * np.pi))
        pri = np.full(len(hyps), 1.0 / len(hyps))
        marg = (pri[:, None] * comp).sum(0)
        h_marg = -np.sum(np.where(marg > 0, marg * np.log(marg), 0.0)) * dg
        h_cond = float((pri * (0.5 * np.log(2 * np.pi * np.e * sds ** 2))).sum())
        eig[a] = float(h_marg - h_cond)
    a_star = max(eig, key=eig.get)

    b.add(_H("BOSS5:thr(A)*B", "relational", "AND", ("A", "B"),
             params={"tau": tau, "temp": temp, "beta": float(beta),
                     "composition": ["AND", "THRESHOLD"]},
             parents=[("A",), ("B",), ("thrA",)],
             notes="thresholded conjunction"),
          feature=f, parent_features=[A, B, thr])
    b.add(_H("DESIGN:boss5", "experimental", "DESIGN", ("action",),
             target="H", params={"a_star": a_star, "eig": eig,
                                 "actions": actions, "hypotheses": hyps,
                                 "composition": ["DESIGN", "AND", "THRESHOLD"]},
             notes="which intervention most reduces uncertainty between the "
                   "thresholded conjunction and its additive parents"))
    b.meta.update(composition=("AND", "DESIGN", "THRESHOLD"), a_star=a_star,
                  eig=eig, actions=actions, hypotheses=hyps, tau=tau)
    return _finish(b, artifacts, d_obs=d_obs,
                   name="boss5/%s/s%d" % (role, seed))


# ===========================================================================
# BOSS-6: nonlinear response x graph-motif enrichment
# ===========================================================================
def make_boss6(seed=0, role="proposal", n=900, snr="high",
               nodes_per_graph=14, p_edge=0.19, K=0.95, nexp=3.0,
               artifacts=(), d_obs=48, noise=0.65, max_planted=5):
    """A nonlinear dose response whose effect is conditional on graph wiring.

    Each row is one independently sampled graph.  ``planted_motifs`` is the
    achieved excess of labelled triangles after degree-preserving edge swaps;
    it is therefore not reducible to graph density or label abundance.  The
    response is::

        Y_g = beta * z(Hill(A_g)) * z(excess-labelled-triangles_g) + Q_g + e.

    The complete scientific object needs both the continuous nonlinear dose
    relation and the graph-motif enrichment.  A response to A alone, a raw
    graph-density association, or an unconstrained triangle count is an
    incomplete/proxy answer.  This is deliberately a graph-native world, not
    a tabular interaction labelled "graph" after the fact.
    """
    b = _B("boss6_hill_graph", seed, role, n, "graph", snr)
    edges, gid, labels, feats, motif, planted, degree = [], [], [], [], [], [], []
    off = 0
    for g in range(n):
        m = int(nodes_per_graph)
        adj = (b.rs.random((m, m)) < p_edge)
        adj = np.triu(adj, 1)
        adj = adj | adj.T
        lab = (b.rs.random(m) < 0.45).astype(int)
        labelled = np.where(lab > 0)[0]
        before = _count_labelled_triangles(adj, lab)
        for _ in range(int(b.rs.integers(0, max_planted + 1))):
            if len(labelled) < 3:
                break
            u, v, w = (int(x) for x in b.rs.choice(labelled, size=3,
                                                    replace=False))
            for a, c in ((u, v), (v, w), (u, w)):
                _install_edge(adj, a, c, {u, v, w}, b.rs)
        count = _count_labelled_triangles(adj, lab)
        motif.append(float(count))
        planted.append(float(count - before))
        labels.append(float(lab.mean()))
        degree.append(float(adj.sum()) / m)
        ii, jj = np.where(np.triu(adj, 1))
        edges.append(np.column_stack([ii + off, jj + off]))
        gid.append(np.full(m, g))
        feats.append(np.column_stack([lab.astype(float),
                                      b.rs.standard_normal(m)]))
        off += m

    planted = np.asarray(planted, dtype=float)
    label_rate = np.asarray(labels, dtype=float)
    mean_degree = np.asarray(degree, dtype=float)
    A = np.exp(0.55 * b.rs.standard_normal(n))
    Q = b.rs.standard_normal(n)
    A = A + 0.22 * Q  # a genuine but incomplete nuisance-mediated shortcut
    hA = hill(np.maximum(A, 1e-6), K=K, nexp=nexp)
    f = zscore(hA) * zscore(planted)
    beta = 2.0 * b.scale
    b.structure = Structure(
        kind="graph", edges=np.vstack(edges), graph_id=np.concatenate(gid),
        meta={"nodes_per_graph": int(nodes_per_graph),
              "node_features": np.vstack(feats), "motif": "labelled_triangle",
              "planted": True, "graph_rewire_probe": "hillA"})
    b.lat.update(A=A, motif_count=np.asarray(motif, dtype=float),
                 planted_motifs=planted, label_rate=label_rate,
                 mean_degree=mean_degree)
    b.derive("hillA", hA)
    # Clipped A, machine-readable for `terms` below: `term_column`'s SATURATE
    # branch chooses between clipping (x>=0 already) and an exp(0.6x) fallback
    # based on the OPERAND's own min(), so passing raw "A" (which dips
    # slightly negative here) silently takes the exp() branch instead of
    # this world's own np.maximum(A, 1e-6) clip -- a different transform
    # (measured abs_corr 0.96-0.97 against the true `hillA`, not 1.0).
    # Deriving the already-clipped column once and referencing THAT in
    # `terms` makes the evaluator take the correct branch by construction.
    b.derive("A_clip", np.maximum(A, 1e-6))
    b.lat["D0"] = b.rs.standard_normal(n)
    b.nui["Q"] = Q
    b.nui["density"] = zscore(mean_degree)
    b.nui["composition"] = zscore(label_rate)
    b.Y = (beta * f + 0.35 * Q + 0.45 * b.nui["density"]
           + 0.35 * b.nui["composition"] + noise * b.rn.standard_normal(n))
    b.add(_H("BOSS6:hill(A)*labelled_triangle_excess", "relational", "SATURATE",
             ("A", "planted_motifs"),
             params={"K": K, "n": nexp, "motif": "labelled_triangle",
                     "graph_null": "degree_preserving_rewire",
                     "beta": float(beta),
                     "composition": ["SATURATE", "GRAPH"],
                     # Truth-declaration correctness fix (2026-09-15, owner
                     # directive): the generative feature multiplies by
                     # `planted_motifs` (the post-rewiring excess), never by
                     # `motif_count` (the raw, un-rewired count) -- the
                     # world's own docstring already said as much ("not
                     # reducible to ... an unconstrained triangle count").
                     # `motif_count` remains in `parents` below as a real,
                     # simpler alternative explanation to control against,
                     # not as the object's own operand. See BOSS1's own
                     # `terms` comment for the general mechanism; verified
                     # abs_corr = 1.000000 against the stored
                     # `zscore(hillA) * zscore(planted_motifs)` signal.
                     "terms": [
                         {"op": "IDENTITY",
                          "operands": [{"op": "SATURATE", "operands": ["A_clip"],
                                       "params": {"K": K, "n": nexp}}]},
                         {"op": "IDENTITY", "operands": ["planted_motifs"]},
                     ]},
             parents=[("A",), ("hillA",), ("motif_count",),
                      ("label_rate",), ("mean_degree",)],
             notes="saturating dose response gated by degree- and composition-"
                   "matched graph-motif enrichment"),
          feature=f, parent_features=[hA, planted, b.nui["Q"],
                                      b.nui["density"], b.nui["composition"]])
    b.meta.update(composition=("GRAPH", "SATURATE"), K=K, nexp=nexp,
                  mean_planted=float(planted.mean()),
                  graph_n=int(n), nodes_per_graph=int(nodes_per_graph))
    return _finish(b, artifacts, d_obs=d_obs,
                   name="boss6/%s/s%d" % (role, seed))


# ===========================================================================
# BOSS-7: intervention x nonlinear response x temporal lag
# ===========================================================================
def make_boss7(seed=0, role="proposal", n=4800, snr="high", lag=3,
               K=0.9, nexp=3.0, p_stay=0.92, artifacts=(), d_obs=48,
               noise=0.60):
    """A lagged nonlinear mechanism visible only under a registered do regime.

    The observational and intervention rows share the same autocorrelated
    exposure series.  Treatment changes whether its *lagged, saturating*
    response reaches Y; it does not simply add a treatment main effect.  This
    separates a temporal association from an intervention-supported causal
    scientific statement:

        Y_t = 1[do_treat_t] Hill(A_{t-lag}) + seasonal nuisance + noise.
    """
    b = _B("boss7_causal_hill_lag", seed, role, n, "temporal", snr)
    t = np.arange(n)
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        state[i] = state[i - 1] if b.rs.random() < p_stay else 1 - state[i - 1]
    gate = state.astype(float)
    A0 = ar1(n, rho=0.82, rng=b.rs)
    season = np.sin(2 * np.pi * t / 64.0)
    A = np.exp(0.45 * A0) + 0.16 * season
    Alag = np.zeros(n, dtype=float)
    Alag[lag:] = A[:-lag]
    hA = hill(np.maximum(Alag, 1e-6), K=K, nexp=nexp)
    valid = t >= lag
    f = gate * hA
    beta = 2.2 * b.scale
    b.structure = Structure(kind="temporal", time=t,
                            meta={"lag": int(lag), "period": 64})
    b.lat.update(A=A, treat=gate)
    b.derive("A_lag%d" % lag, Alag)
    b.derive("hillA_lag%d" % lag, hA)
    b.lat["D0"] = b.rs.standard_normal(n)
    b.nui["season"] = season
    b.nui["t_index"] = zscore(t.astype(float))
    assignment = np.where(gate > 0, "do_treat", "obs")
    b.interventions = Interventions(
        assignment=assignment, targets={"do_treat": ("treat",), "obs": ()},
        kind={"do_treat": "do", "obs": "none"},
        values={"do_treat": {"treat": 1.0}})
    b.env_labels["time"] = t
    b.env_labels["regime"] = assignment
    b.env_labels["valid"] = valid.astype(int)
    b.Y = beta * f + 0.38 * season + noise * b.rn.standard_normal(n)
    b.Y[~valid] = (0.38 * season[~valid]
                   + noise * b.rn.standard_normal(int((~valid).sum())))
    b.add(_H("BOSS7:do_treat*hill(A_lag)", "mechanism", "CAUSE",
             ("A", "treat"), lag=int(lag),
             params={"K": K, "n": nexp, "lag": int(lag),
                     "beta": float(beta),
                     "composition": ["CAUSE", "SATURATE", "LAG"],
                     # See BOSS1's own `terms` comment. Verified abs_corr =
                     # 1.000000 against the stored `treat * hillA_lag{lag}`
                     # signal (hillA_lag{lag} = hill(LAG(A, lag), K, nexp)).
                     "terms": [
                         {"op": "CONDITION", "operands": ["treat"],
                          "params": {"gate_value": 1}},
                         {"op": "SATURATE",
                          "operands": [{"op": "LAG", "operands": ["A"],
                                       "lag": int(lag)}],
                          "params": {"K": K, "n": nexp}},
                     ]},
             context={"regime": "do_treat"},
             parents=[("A",), ("A_lag%d" % lag,),
                      ("hillA_lag%d" % lag,), ("treat",)],
             notes="intervention-specific saturating response at a fixed "
                   "temporal lag"),
          feature=f, parent_features=[Alag, hA, gate, season])
    b.meta.update(composition=("CAUSE", "LAG", "SATURATE"), lag=int(lag),
                  K=K, nexp=nexp, intervention_fraction=float(gate.mean()),
                  n_valid=int(valid.sum()))
    return _finish(b, artifacts, d_obs=d_obs,
                   name="boss7/%s/s%d" % (role, seed))


BOSS_WORLDS = {
    "boss1_cond_thresh_space_lag": make_boss1,
    "boss2_xor_space_regime": make_boss2,
    "boss3_hill_space_cause": make_boss3,
    "boss4_ode_manifold_anomaly": make_boss4,
    "boss5_design_and_threshold": make_boss5,
    "boss6_hill_graph": make_boss6,
    "boss7_causal_hill_lag": make_boss7,
}

BOSS_COMPOSITION = {
    "boss1_cond_thresh_space_lag": ("CONDITION", "LAG", "SPACE", "THRESHOLD"),
    "boss2_xor_space_regime": ("ANOMALY", "CONDITION", "SPACE", "XOR"),
    "boss3_hill_space_cause": ("CAUSE", "SATURATE", "SPACE"),
    "boss4_ode_manifold_anomaly": ("ANOMALY", "MANIFOLD", "ODE"),
    "boss5_design_and_threshold": ("AND", "DESIGN", "THRESHOLD"),
    "boss6_hill_graph": ("GRAPH", "SATURATE"),
    "boss7_causal_hill_lag": ("CAUSE", "LAG", "SATURATE"),
}
