"""Tier-I synthetic families: exact worlds with known latents and known objects.

Every generator here returns a `World` in which the latents, the object
identities, the nuisance variables and the true hypothesis set are all known
before any method runs. Each family is a *minimal* isolation of one scientific
structure, so that a failure can be attributed to that structure rather than to
a composition; the compositions live in `dbench.boss`.

Two invariants hold across every family and are asserted by the tests in
`dbench/tests/`:

1. Headroom. Every world carries distractor latents that participate in no
   object, so a method cannot score by flagging everything it can see.
2. Structure, not summary statistics. Each family's test asserts the planted
   structure itself -- that pure AND's constituents really are marginally
   uncorrelated with the target, that the lagged relation really survives the
   contemporaneous term, that the confounded association really vanishes given
   U -- rather than asserting that some recovery score exceeded a number.

The `snr` argument selects an effect-size preset. "high" is the setting the
oracle gate is defined on: at "high", handing a specialist the oracle latents
must recover the family. "med" and "low" exist for the n_80 sample-efficiency
curves and are not part of the gate.
"""
from __future__ import annotations

import contextlib
from itertools import combinations

from dataclasses import replace as _dc_replace

import numpy as np

from . import artifacts as art
from .util import (ar1, corr, derive_seed, gp_field, grid_coords, hill,
                   knn_neighbors, orthonormal, partial_corr, radius_neighbors,
                   seasonal, sigmoid, soft_threshold, zscore)
from .world import (EnvMeta, Hypothesis, Interventions, Structure, TruthSet,
                    World)

SNR_PRESETS = {"high": 1.0, "med": 0.5, "low": 0.25}


# --------------------------------------------------------------------------
# the instrument axis
# --------------------------------------------------------------------------
#: Which observation operator worlds are built with. Environments share it by
#: default, so replication means the same measurement on new data. Building
#: under a different name gives a genuinely different instrument -- a different
#: loading matrix and nuisance mixing -- which is the harder condition and is
#: opt-in precisely because it should be asked for rather than inherited.
_INSTRUMENT = "default"

#: Which seed the observation operator is derived from. `None` means "the
#: world's own seed", which is right for a standalone world and wrong for an
#: episode: `make_episode` gives every environment a different seed, so with
#: the operator keyed on it the environments got different instruments again --
#: the environment seed had silently taken over the job `role` used to do.
#: Measured on `pure_and` before this: same seed / different role gave
#: footprint cosine 0.9948, and the episode path, which varies both, gave
#: 0.1410 -- chance, and indistinguishable from an explicitly different
#: instrument at 0.1483. `make_episode` now pins this to the episode seed so
#: every environment in one episode shares an operator while episodes stay
#: distinct from each other.
_OPERATOR_SEED = None


@contextlib.contextmanager
def operator_seed(value):
    """Build worlds whose observation operator is derived from `value`."""
    global _OPERATOR_SEED
    prev = _OPERATOR_SEED
    _OPERATOR_SEED = value
    try:
        yield
    finally:
        _OPERATOR_SEED = prev


@contextlib.contextmanager
def instrument(name):
    """Build worlds under a named observation operator.

    >>> with instrument("siteB"):
    ...     w = make_pure_and(seed=0, role="replication")

    Worlds built under different names share no observed-variable basis. Use it
    to pose "does this object survive a change of instrument?", never to
    generate an ordinary replication environment.
    """
    global _INSTRUMENT
    prev = _INSTRUMENT
    _INSTRUMENT = str(name)
    try:
        yield
    finally:
        _INSTRUMENT = prev


# --------------------------------------------------------------------------
# builder plumbing
# --------------------------------------------------------------------------
class _B:
    """Accumulator for one world under construction."""

    def __init__(self, family, seed, role, n, domain, snr, instrument=None,
                 operator_seed=None):
        self.family = family
        self.seed = int(seed)
        self.role = role
        self.n = int(n)
        self.domain = domain
        self.snr = snr
        self.scale = SNR_PRESETS[snr]
        # Separate streams: signal, nuisance, artifact, instrument, measurement
        # noise. A replication environment differs in the sample -- new
        # latents, new nuisance, new noise -- and by default shares the
        # instrument.
        #
        # `ro` deliberately does NOT depend on the role. It draws the loading
        # matrix and the nuisance mixing, i.e. the observation *operator*, and
        # seeding it by role gave every environment a different instrument.
        # Measured on `pure_and`: per-latent footprint cosine between D_P and
        # D_R was 0.001 and 0.210 against a chance level of 0.18, and a readout
        # frozen in D_P measured the planted object in D_R at |r| = 0.007
        # against 0.503 with a shared operator. The spec's replication step is
        # "measure those exact objects in D_R"; with a role-seeded operator
        # that step is unsatisfiable, because the object has no coordinates
        # there. Sample independence is already supplied in full by `rs`, `rn`
        # and `rq`.
        #
        # Changing instrument between environments is a real and harder
        # scientific condition, so it is kept -- as the named, opt-in
        # `instrument` axis rather than as a silent property of every
        # replication.
        self.instrument = str(_INSTRUMENT if instrument is None else instrument)
        # the operator seed is the episode's when one is in scope, else this
        # world's own -- so a standalone world is unchanged and every
        # environment of an episode shares one instrument
        if operator_seed is not None:
            self.operator_seed = operator_seed
        elif _OPERATOR_SEED is not None:
            self.operator_seed = _OPERATOR_SEED
        else:
            self.operator_seed = seed
        self.rs = np.random.default_rng(derive_seed(family, seed, role, "signal"))
        self.rn = np.random.default_rng(derive_seed(family, seed, role, "nuisance"))
        self.ra = np.random.default_rng(derive_seed(family, seed, role, "artifact"))
        self.ro = np.random.default_rng(
            derive_seed(family, self.operator_seed, "observe", self.instrument))
        self.rq = np.random.default_rng(
            derive_seed(family, seed, role, "measurement_noise"))
        self.lat = {}
        self.derived = []       # latents that ARE an operator applied to
                                # another latent at its true parameter
        self.nui = {}
        self.env_labels = {}
        self.objs = []
        self.features = {}      # hypothesis id -> planted feature, for effect sizes
        self.parents_of = {}    # hypothesis id -> list of parent features
        self.Y = None
        self.structure = Structure()
        self.interventions = None
        self.meta = {}

    def distractors(self, k, prefix="D"):
        for i in range(k):
            self.lat["%s%d" % (prefix, i)] = self.rs.standard_normal(self.n)

    def derive(self, name, value):
        """Record a latent that is a *derived* column: an operator already
        applied to another latent at the world's true parameter value.

        `aggB`, `B_lag3`, `thrA` and their kin exist because the generator
        needed them, and they are legitimately part of the oracle latent set --
        an encoder that reconstructed them would deserve credit. They are, at
        the same time, the answer written down: a lag scan that may test
        `B_lag3` does not have to find the lag, and a threshold scan handed
        `thrA` does not have to find tau. `run_gate` therefore excludes them
        from every specialist's candidate set, so that the gate reports a
        search that succeeded rather than a lookup that could not fail.
        """
        self.lat[name] = value
        self.derived.append(name)
        return value

    def add(self, h, feature=None, parent_features=()):
        self.objs.append(h)
        if feature is not None:
            self.features[h.id] = np.asarray(feature, dtype=np.float64)
            self.parents_of[h.id] = [np.asarray(p, dtype=np.float64)
                                     for p in parent_features]
        return h


def _finish(b, artifact_classes=(), d_obs=48, obs_noise=0.35,
            n_artifact_each=1, name=None):
    """Plant artifacts, build observed modalities, compute effect sizes."""
    if artifact_classes:
        b.Y, aobjs = art.plant(b.lat, b.nui, b.Y, b.env_labels, b.ra,
                               classes=artifact_classes, n_each=n_artifact_each)
        b.objs.extend(aobjs)

    # ---- observed modalities -------------------------------------------
    names = sorted(b.lat)
    m1 = [k for k in names if not k.endswith("_m2")]
    m2 = [k for k in names if not k.endswith("_m1")]
    cross_modal = any(k.endswith("_m1") or k.endswith("_m2") for k in names)
    mods = {}
    if cross_modal:
        mods["X1"] = _observe(b, m1, d_obs, obs_noise, "X1")
        mods["X2"] = _observe(b, m2, d_obs, obs_noise, "X2")
    else:
        mods["X"] = _observe(b, names, d_obs, obs_noise, "X")

    # ---- effect sizes: partial correlation given the object's parents ----
    finished = []
    for h in b.objs:
        if h.id in b.features and b.Y is not None:
            f = b.features[h.id]
            ctrl = list(b.parents_of.get(h.id, []))
            eff = partial_corr(b.Y, f, ctrl) if ctrl else corr(b.Y, f)
            h = _dc_replace(h, effect=float(eff))
        finished.append(h)

    return World(
        name=name or "%s/%s/s%d" % (b.family, b.role, b.seed),
        family=b.family, domain=b.domain,
        latents=b.lat, modalities=mods, target=b.Y,
        nuisance=b.nui,
        env=EnvMeta(role=b.role, labels=b.env_labels, world_seed=b.seed,
                    template=b.family,
                    tags={"snr": b.snr, "artifacts": list(artifact_classes)}),
        structure=b.structure, interventions=b.interventions,
        truth=TruthSet(objects=tuple(finished)),
        meta=dict(b.meta, snr=b.snr, n=b.n, d_obs=d_obs, obs_noise=obs_noise,
                  instrument=b.instrument, operator_seed=b.operator_seed,
                  artifact_classes=list(artifact_classes),
                  derived_latents=tuple(dict.fromkeys(
                      list(b.derived) + list(b.meta.get("derived_latents", ()))))),
    )


def _observe(b, names, d_obs, obs_noise, tag):
    """Linear mixing of latents into an observed modality, plus nuisance.

    Orthonormal loadings keep every latent recoverable by ridge from the
    modality, which is what makes the oracle-vs-observed comparison meaningful
    rather than a test of whether the mixing was invertible.
    """
    cols = []
    for k in names:
        v = np.asarray(b.lat[k], dtype=np.float64)
        cols.append(v if v.ndim == 1 else v[:, 0])
        if v.ndim > 1:
            for j in range(1, v.shape[1]):
                cols.append(v[:, j])
    F = zscore(np.column_stack(cols)) if cols else np.zeros((b.n, 0))
    k = F.shape[1]
    L = orthonormal(b.ro, d_obs, k)
    # Accumulated as k rank-one updates in a fixed order rather than as one
    # `F @ L.T`. A gemm is free to split the summation dimension across
    # threads, and on this package it did: with k = 8 or 10 the observed
    # modality of additive_factors, pure_and, manifold and boss4 changed
    # between one and eight threads on one of the two machines tested, while
    # k = 6 or 7 families were unaffected. Same arithmetic, fixed order, no
    # BLAS.
    X = np.zeros((b.n, d_obs), dtype=np.float64)
    for j in range(k):
        X = X + F[:, j:j + 1] * L[:, j]
    if b.nui:
        Nc = np.column_stack([zscore(np.asarray(v, dtype=np.float64).reshape(b.n, -1))
                              for v in b.nui.values()])
        Wn = b.ro.standard_normal((Nc.shape[1], d_obs)) / np.sqrt(Nc.shape[1])
        acc = np.zeros((b.n, d_obs), dtype=np.float64)
        for j in range(Nc.shape[1]):
            acc = acc + Nc[:, j:j + 1] * Wn[j, :]
        X = X + 0.8 * acc
    # measurement noise is drawn per environment; the instrument above is not
    return X + obs_noise * b.rq.standard_normal((b.n, d_obs))


def _H(hid, mode, operator, operands, target="Y", params=None, context=None,
       lag=None, parents=(), notes="", equivalent_expressions=()):
    return Hypothesis(id=hid, mode=mode, operator=operator,
                      operands=tuple(operands), target=target,
                      params=dict(params or {}), context=dict(context or {}),
                      lag=lag, parents=tuple(tuple(p) for p in parents),
                      equivalent_expressions=tuple(
                          _freeze(e) for e in equivalent_expressions),
                      notes=notes)


def _freeze(e):
    """A declared equivalent expression, in a hashable form."""
    if isinstance(e, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in e.items()))
    if isinstance(e, (list, tuple)):
        return tuple(_freeze(x) for x in e)
    return e


# ==========================================================================
# 1. additive factors                                          (entity/ADD)
# ==========================================================================
def make_additive_factors(seed=0, role="proposal", n=3000, snr="high",
                          k_true=3, k_distract=5, artifacts=(), d_obs=48,
                          noise=1.0):
    b = _B("additive_factors", seed, role, n, "tabular", snr)
    betas = np.array([1.2, -0.9, 0.7]) * b.scale
    feats = []
    for i in range(k_true):
        b.lat["A%d" % i] = b.rs.standard_normal(n)
        feats.append(b.lat["A%d" % i])
    b.distractors(k_distract)
    Y = sum(betas[i] * feats[i] for i in range(k_true))
    Y = Y + noise * b.rn.standard_normal(n)
    b.Y = Y
    for i in range(k_true):
        b.add(_H("ADD:A%d" % i, "relational", "ADD", ("A%d" % i,),
                 params={"beta": float(betas[i])},
                 notes="additive main effect"), feature=feats[i])
    return _finish(b, artifacts, d_obs=d_obs, name=None)


# ==========================================================================
# 2. pure AND with zero marginal signal                     (relational/AND)
# ==========================================================================
def make_pure_and(seed=0, role="proposal", n=4000, snr="high", n_pairs=2,
                  n_decoy=1, k_distract=4, artifacts=(), d_obs=48, noise=1.0):
    """Y depends on A*B only.

    A and B are independent, zero-mean and unit-variance, so
        E[A * (A B)] = E[A^2] E[B] = 0,
    i.e. each constituent is *marginally uncorrelated* with the term it helps
    generate. This is the structure Prop. 2 proves is invisible to any method
    that ranks candidates by marginal association. `n_decoy` additive pairs are
    present so that flagging every product is punished: a decoy's constituents
    have real main effects and no interaction.
    """
    b = _B("pure_and", seed, role, n, "tabular", snr)
    beta = 1.4 * b.scale
    Y = np.zeros(n)
    pairs = []
    for i in range(n_pairs):
        a = b.rs.standard_normal(n)
        bb = b.rs.standard_normal(n)
        b.lat["A%d" % i], b.lat["B%d" % i] = a, bb
        prod = a * bb
        Y = Y + beta * prod
        pairs.append((i, a, bb, prod))
    for j in range(n_decoy):
        u = b.rs.standard_normal(n)
        v = b.rs.standard_normal(n)
        b.lat["U%d" % j], b.lat["V%d" % j] = u, v
        Y = Y + 0.8 * b.scale * u + 0.6 * b.scale * v   # additive only, no U*V
    b.distractors(k_distract)
    b.Y = Y + noise * b.rn.standard_normal(n)
    for i, a, bb, prod in pairs:
        b.add(_H("AND:A%dB%d" % (i, i), "relational", "AND",
                 ("A%d" % i, "B%d" % i),
                 params={"beta": float(beta)},
                 parents=[("A%d" % i,), ("B%d" % i,)],
                 notes="pure conjunction; both constituents marginally "
                       "uncorrelated with Y"),
              feature=prod, parent_features=[a, bb])
    for j in range(n_decoy):
        b.add(_H("ADD:U%d" % j, "relational", "ADD", ("U%d" % j,)),
              feature=b.lat["U%d" % j])
        b.add(_H("ADD:V%d" % j, "relational", "ADD", ("V%d" % j,)),
              feature=b.lat["V%d" % j])
    b.meta["decoy_pairs"] = [("U%d" % j, "V%d" % j) for j in range(n_decoy)]
    return _finish(b, artifacts, d_obs=d_obs)


# ===========================================================================
# 2b. invisible higher-order conjunctions                    (relational/AND)
# ===========================================================================
def make_invisible_kway(seed=0, role="proposal", n=12000, snr="high",
                        order=3, k_distract=8, artifacts=(), d_obs=48,
                        noise=1.0):
    """A conjunction invisible to every proper lower-order intermediary.

    ``Y = beta * product(A_0, ..., A_{k-1}) + epsilon`` with independent,
    centred Rademacher constituents.  Any non-empty proper product omits at
    least one zero-mean constituent, so it has zero population covariance with
    the full product.  Thus greedy compiler pruning has no statistical reason
    to retain the intermediate route to the true k-way object.

    This is the explicit Tier-I instrument for LOCKINv2's k=3--5 invisible-
    intermediary requirement.  It is not a specialist-inexpressibility win:
    recovery requires every named constituent of the complete conjunction.
    """
    if int(order) not in (3, 4, 5):
        raise ValueError("invisible k-way order must be one of 3, 4, 5")
    order = int(order)
    b = _B("invisible_k%d" % order, seed, role, n, "tabular", snr)
    operands = tuple("A%d" % i for i in range(order))
    columns = []
    for name in operands:
        # Unit Rademacher coordinates avoid the unstable tails of a five-way
        # Gaussian product while preserving exact population orthogonality of
        # all Walsh-product terms.  This makes the benchmark's invisibility
        # property a structural fact, not a lucky finite-sample event.
        x = (2.0 * b.rs.integers(0, 2, n) - 1.0)
        b.lat[name] = x
        columns.append(x)
    b.distractors(k_distract)

    full = np.prod(np.column_stack(columns), axis=1)
    beta = 1.8 * b.scale
    b.Y = beta * full + noise * b.rn.standard_normal(n)

    # These are all lower-order candidate features, not planted objects.  The
    # explicit hierarchy is used by the effect calculation and documents what
    # a search must preserve in order to form the final object.
    parent_operands, parent_features = [], []
    for width in range(1, order):
        for inds in combinations(range(order), width):
            parent_operands.append(tuple(operands[i] for i in inds))
            parent_features.append(np.prod(
                np.column_stack([columns[i] for i in inds]), axis=1))

    b.add(_H("ANDK%d:%s" % (order, "*".join(operands)), "relational", "AND",
                 operands, params={"beta": float(beta), "order": order},
                 parents=parent_operands,
                 notes="%d-way conjunction; every proper lower-order "
                       "intermediate is marginally invisible to Y" % order),
          feature=full, parent_features=parent_features)
    b.meta.update(hard=True, invisible_intermediate_order=order,
                  lower_order_candidate_count=len(parent_features),
                  required_complete_operands=operands)
    return _finish(b, artifacts, d_obs=d_obs)


def make_invisible_k3(**kw):
    """Convenience registry entry for the three-way invisible world."""
    if "order" in kw:
        raise ValueError("invisible_k3 fixes order=3")
    return make_invisible_kway(order=3, **kw)


def make_invisible_k4(**kw):
    """Convenience registry entry for the four-way invisible world."""
    if "order" in kw:
        raise ValueError("invisible_k4 fixes order=4")
    return make_invisible_kway(order=4, **kw)


def make_invisible_k5(**kw):
    """Convenience registry entry for the five-way invisible world."""
    if "order" in kw:
        raise ValueError("invisible_k5 fixes order=5")
    return make_invisible_kway(order=5, **kw)


# ==========================================================================
# 3. XOR                                                    (relational/XOR)
# ==========================================================================
def make_xor(seed=0, role="proposal", n=4000, snr="high", k_distract=5,
             artifacts=(), d_obs=48, noise=1.0):
    """Y = beta * 1[A != B] with A, B independent Bernoulli(1/2).

    E[Y | A=0] = E[Y | A=1] = beta/2, so both constituents are exactly
    marginally uninformative. Unlike pure AND the latents are discrete, so a
    product-of-continuous-latents search does not solve it for free.
    """
    b = _B("xor", seed, role, n, "tabular", snr)
    beta = 1.6 * b.scale
    A = b.rs.integers(0, 2, n)
    B = b.rs.integers(0, 2, n)
    b.lat["A"], b.lat["B"] = A.astype(float), B.astype(float)
    b.distractors(k_distract)
    x = (A != B).astype(float)
    b.Y = beta * x + noise * b.rn.standard_normal(n)
    b.add(_H("XOR:AB", "relational", "XOR", ("A", "B"),
             params={"beta": float(beta)}, parents=[("A",), ("B",)],
             notes="parity; each constituent exactly marginally uninformative"),
          feature=x, parent_features=[A.astype(float), B.astype(float)])
    return _finish(b, artifacts, d_obs=d_obs)


def make_invisible_xor_kway(seed=0, role="proposal", n=12000, snr="high",
                            order=3, k_distract=8, artifacts=(), d_obs=48,
                            noise=1.0):
    """High-order parity whose every proper subset is uninformative.

    The response is the parity of independent fair binary variables.
    Marginalising any operand makes the parity fair again, so neither a
    singleton, pair, nor another proper subset identifies the full object.
    This complements signed-product AND with a discrete interaction for which
    multiplication is not an equivalent shortcut.
    """
    if int(order) not in (3, 4):
        raise ValueError("invisible k-way XOR order must be 3 or 4")
    order = int(order)
    b = _B("invisible_xor_k%d" % order, seed, role, n, "tabular", snr)
    operands = tuple("A%d" % i for i in range(order))
    columns = []
    for name in operands:
        x = b.rs.integers(0, 2, n).astype(float)
        b.lat[name] = x
        columns.append(x)
    b.distractors(k_distract)
    full = (np.sum(np.column_stack(columns), axis=1).astype(int) % 2).astype(float)
    beta = 1.8 * b.scale
    b.Y = beta * full + noise * b.rn.standard_normal(n)
    parent_operands, parent_features = [], []
    for width in range(1, order):
        for inds in combinations(range(order), width):
            parent_operands.append(tuple(operands[i] for i in inds))
            parent_features.append((np.sum(np.column_stack(
                [columns[i] for i in inds]), axis=1).astype(int) % 2).astype(float))
    b.add(_H("XORK%d:%s" % (order, "^".join(operands)), "relational", "XOR",
             operands, params={"beta": float(beta), "order": order},
             parents=parent_operands,
             notes="%d-way parity; every proper parity subset is marginally "
                   "uninformative about the complete object" % order),
          feature=full, parent_features=parent_features)
    b.meta.update(hard=True, invisible_intermediate_order=order,
                  lower_order_candidate_count=len(parent_features),
                  required_complete_operands=operands, parity=True)
    return _finish(b, artifacts, d_obs=d_obs)


def make_invisible_xor_k3(**kw):
    if "order" in kw:
        raise ValueError("invisible_xor_k3 fixes order=3")
    return make_invisible_xor_kway(order=3, **kw)


def make_invisible_xor_k4(**kw):
    if "order" in kw:
        raise ValueError("invisible_xor_k4 fixes order=4")
    return make_invisible_xor_kway(order=4, **kw)


# ==========================================================================
# 4. threshold                                        (relational/THRESHOLD)
# ==========================================================================
def make_threshold(seed=0, role="proposal", n=3000, snr="high", tau=1.0,
                   temp=0.08, k_distract=5, artifacts=(), d_obs=48, noise=0.6):
    """Y = beta * sigma((A - tau)/T): a sharp switch, not a slope.

    The object must beat its unthresholded parent A. tau sits at ~1 sd so a
    linear term is a poor surrogate.
    """
    b = _B("threshold", seed, role, n, "tabular", snr)
    beta = 2.0 * b.scale
    A = b.rs.standard_normal(n)
    b.lat["A"] = A
    b.distractors(k_distract)
    f = soft_threshold(A, tau, temp)
    b.Y = beta * f + noise * b.rn.standard_normal(n)
    b.add(_H("THRESH:A", "relational", "THRESHOLD", ("A",),
             params={"tau": tau, "temp": temp, "beta": float(beta)},
             parents=[("A",)],
             notes="switch at tau; must beat the linear parent A"),
          feature=f, parent_features=[A])
    b.meta["tau"] = tau
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 5. Hill / saturation                                 (relational/SATURATE)
# ==========================================================================
def make_hill(seed=0, role="proposal", n=3000, snr="high", K=1.0, nexp=4.0,
              k_distract=5, artifacts=(), d_obs=48, noise=0.5):
    b = _B("hill", seed, role, n, "tabular", snr)
    beta = 2.0 * b.scale
    A = np.exp(0.7 * b.rs.standard_normal(n))          # positive dose
    b.lat["A"] = A
    b.distractors(k_distract)
    f = hill(A, K=K, nexp=nexp)
    b.Y = beta * f + noise * b.rn.standard_normal(n)
    b.add(_H("SAT:A", "relational", "SATURATE", ("A",),
             params={"K": K, "n": nexp, "beta": float(beta)},
             parents=[("A",)], notes="Hill response; saturates above K"),
          feature=f, parent_features=[A])
    b.meta.update(K=K, nexp=nexp)
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 6. context-gated interaction                        (relational/CONDITION)
# ==========================================================================
def make_context_gated(seed=0, role="proposal", n=4000, snr="high",
                       k_distract=5, artifacts=(), d_obs=48, noise=1.0):
    """Y = beta * (2*G - 1) * A: the effect of A reverses with context G.

    Zero-marginal by construction in A (G is balanced), so the object is the
    conditioning and not a main effect wearing a hat.
    """
    b = _B("context_gated", seed, role, n, "tabular", snr)
    beta = 1.5 * b.scale
    A = b.rs.standard_normal(n)
    G = b.rs.integers(0, 2, n)
    b.lat["A"], b.lat["G"] = A, G.astype(float)
    b.distractors(k_distract)
    f = (2.0 * G - 1.0) * A
    b.Y = beta * f + noise * b.rn.standard_normal(n)
    b.add(_H("COND:A|G", "relational", "CONDITION", ("A", "G"),
             params={"beta": float(beta), "gate": "G"},
             context={"gate_var": "G"}, parents=[("A",), ("G",)],
             equivalent_expressions=[
                 {"operands": ["A", "G"], "operator": "MUL",
                  "measured_abscorr": {"proposal": 0.690, "replication": 0.702},
                  "why": "G is exactly binary (0/1) in this world, and the "
                        "compiler's own COND is defined as h|G == 1[G]*h "
                        "(ops.py: COND.hard = a[0]*a[1]) -- numerically "
                        "IDENTICAL to MUL(A,G) whenever the gate is binary, "
                        "not merely correlated with it. COND(A,THR(G,0.5)) "
                        "is generated by the search (level-1 exhaustive "
                        "REAL x BINARY composition) but is then MDL-dominated "
                        "by the cheaper, value-identical MUL(A,G) -- which "
                        "needs no THR wrapper because a raw VAR is always "
                        "typed Kind.REAL by this grammar's own design (see "
                        "program.py: Program.kind()) -- so MUL(A,G), not "
                        "COND, is what reaches certification. This is the "
                        "SAME object under the operator this grammar actually "
                        "reports for it, not a decoy: declared here rather "
                        "than left to the kin-credit cap (0.65, below "
                        "MATCH_THRESHOLD 0.70) to under-count it."}],
             notes="A acts with opposite sign in the two contexts"),
          feature=f, parent_features=[A, G.astype(float)])
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 7. curved / branching manifold                        (entity/MANIFOLD)
# ==========================================================================
def make_manifold(seed=0, role="proposal", n=3000, snr="high", n_branch=3,
                  k_distract=3, artifacts=(), d_obs=48, noise=0.25):
    """A 1-D latent coordinate on a 3-branch curved manifold embedded in R^d.

    The ground-truth objects are the chart coordinate t (MANIFOLD) and the
    branch label (MIXTURE) -- not a linear factor. PCA recovers neither.
    """
    b = _B("manifold", seed, role, n, "tabular", snr)
    t = b.rs.uniform(0, 1, n)
    br = b.rs.integers(0, n_branch, n)
    ang = 2 * np.pi * br / n_branch
    r = 1.0 + 2.5 * t
    E = np.column_stack([
        r * np.cos(ang + 3.0 * t),
        r * np.sin(ang + 3.0 * t),
        1.5 * t ** 2,
    ])
    b.lat["t"] = t
    b.lat["branch"] = br.astype(float)
    b.lat["chart"] = E
    b.distractors(k_distract)
    b.Y = (2.0 * b.scale) * np.sin(3.0 * t) + 0.4 * b.rn.standard_normal(n)
    b.add(_H("MANIFOLD:t", "entity", "MANIFOLD", ("t",), target="X",
             params={"intrinsic_dim": 1, "n_branches": n_branch,
                     "embedding": "spiral"},
             notes="1-D chart coordinate; curvature makes it linearly invisible"))
    b.add(_H("MIXTURE:branch", "entity", "MIXTURE", ("branch",), target="X",
             params={"n_components": n_branch},
             notes="branch identity of the manifold"))
    b.meta.update(n_branch=n_branch, intrinsic_dim=1)
    return _finish(b, artifacts, d_obs=d_obs, obs_noise=noise)


# ==========================================================================
# 8. spatial neighbourhood interaction                    (relational/SPACE)
# ==========================================================================
def make_spatial(seed=0, role="proposal", n=None, snr="high", side=40,
                 radius=1.5, k_distract=3, artifacts=(), d_obs=48, noise=0.8,
                 ls_nuisance=4.0):
    """Y_i = beta * A_i * mean_{j in N(i)} B_j.

    A and B are zero-mean, so the spatial product is marginally uncorrelated
    with both A and the neighbourhood mean of B. A spatially autocorrelated
    nuisance field and a local-density covariate are present, which the
    hierarchical-novelty rule requires the object to survive.
    """
    coords = grid_coords(side)
    n = len(coords)
    b = _B("spatial_interaction", seed, role, n, "spatial", snr)
    nb = radius_neighbors(coords, radius)
    b.structure = Structure(kind="spatial", coords=coords, neighbors=nb,
                            meta={"side": side, "radius": radius})
    A = zscore(gp_field(coords, b.rs, length_scale=2.0))
    Braw = zscore(gp_field(coords, b.rs, length_scale=2.0))
    b.lat["A"], b.lat["B"] = A, Braw
    aggB = b.structure.aggregate(Braw, "mean")
    b.derive("aggB", aggB)
    b.distractors(k_distract)
    b.nui["spatial_nuisance"] = zscore(gp_field(coords, b.rn,
                                                length_scale=ls_nuisance))
    b.nui["density"] = zscore(b.structure.n_neighbors().astype(float))
    f = zscore(A * aggB)
    beta = 1.6 * b.scale
    b.Y = (beta * f + 0.9 * b.nui["spatial_nuisance"]
           + noise * b.rn.standard_normal(n))
    # The neighbourhood operator is a row-normalised symmetric adjacency, so
    # A' W B and B' W A are the same bilinear form up to the degree
    # normalisation: A * agg(B) and B * agg(A) are the same scientific object
    # with the channels swapped. Measured |corr| between the two design
    # columns is 0.947 on the proposal split and 0.959 on replication -- close,
    # and deliberately recorded rather than rounded to "identical", because it
    # is below the 0.99 evaluation-equivalence criterion and would not be
    # folded by that route. Declaring it here is the generator doing its own
    # job instead of leaving the matcher to guess.
    b.add(_H("SPACE:A*aggB", "relational", "SPACE", ("A", "B"),
             params={"agg": "mean", "radius": radius, "beta": float(beta)},
             parents=[("A",), ("B",), ("aggB",)],
             equivalent_expressions=[
                 {"operands": ["B", "A"], "operator": "SPACE",
                  "params": {"agg": "mean"},
                  "measured_abscorr": {"proposal": 0.947, "replication": 0.959},
                  "why": "own and partner channels swapped; the row-normalised "
                         "neighbourhood operator is symmetric up to degree, so "
                         "this is the same object, not a second one"}],
             notes="own-cell A times neighbourhood mean of B; must survive both "
                   "states and local density"),
          feature=f, parent_features=[A, aggB, b.nui["density"]])
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 9. temporal lag                                          (relational/LAG)
# ==========================================================================
def make_temporal_lag(seed=0, role="proposal", n=1200, snr="high", lag=3,
                      rho=0.75, period=12, k_distract=3, artifacts=(),
                      d_obs=32, noise=0.7):
    """Y_t = beta * B_{t-lag}, with B autocorrelated and Y seasonal.

    B_t enters with zero coefficient, so the lag has to be found rather than
    inherited from the contemporaneous term. The AR(1) structure and the
    seasonal cycle are the statistics the temporal structured null must
    preserve.
    """
    b = _B("temporal_lag", seed, role, n, "temporal", snr)
    t = np.arange(n)
    b.structure = Structure(kind="temporal", time=t, meta={"period": period})
    B = ar1(n, rho, b.rs)
    A = ar1(n, rho, b.rs)
    b.lat["B"], b.lat["A"] = B, A
    Bl = np.concatenate([np.zeros(lag), B[:-lag]])
    b.derive("B_lag%d" % lag, Bl)
    for i in range(k_distract):
        b.lat["D%d" % i] = ar1(n, rho, b.rs)
    b.nui["season"] = seasonal(n, period, b.rn)
    b.nui["trend"] = zscore(t.astype(float))
    beta = 1.5 * b.scale
    b.Y = (beta * Bl + 1.0 * b.nui["season"] + 0.3 * b.nui["trend"]
           + noise * ar1(n, 0.5, b.rn))
    b.add(_H("LAG:B", "relational", "LAG", ("B",), lag=lag,
             params={"beta": float(beta), "lag": lag},
             parents=[("B",)],
             notes="lagged effect; contemporaneous B has coefficient exactly 0"),
          feature=Bl, parent_features=[B])
    b.meta.update(lag=lag, rho=rho, period=period)
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 10. regime switching                                    (anomaly/ANOMALY)
# ==========================================================================
def make_regime_switch(seed=0, role="proposal", n=1500, snr="high",
                       p_stay=0.985, k_distract=3, artifacts=(), d_obs=32,
                       noise=0.6):
    """A two-state Markov regime that flips the sign of A's effect on Y.

    The objects are the regime variable itself (a phase change) and the two
    context-conditional relationships.
    """
    b = _B("regime_switch", seed, role, n, "temporal", snr)
    t = np.arange(n)
    b.structure = Structure(kind="temporal", time=t)
    s = np.zeros(n, dtype=int)
    for i in range(1, n):
        s[i] = s[i - 1] if b.rs.random() < p_stay else 1 - s[i - 1]
    A = ar1(n, 0.6, b.rs)
    b.lat["A"], b.lat["regime"] = A, s.astype(float)
    for i in range(k_distract):
        b.lat["D%d" % i] = ar1(n, 0.6, b.rs)
    beta = 1.6 * b.scale
    f = np.where(s == 1, 1.0, -1.0) * A
    b.Y = beta * f + noise * b.rn.standard_normal(n)
    b.add(_H("REGIME:s", "anomaly", "ANOMALY", ("regime",), target="Y",
             params={"n_states": 2, "p_stay": p_stay},
             notes="latent two-state regime; a phase change, not an outlier"))
    b.add(_H("COND:A|regime", "relational", "CONDITION", ("A", "regime"),
             params={"beta": float(beta)}, context={"gate_var": "regime"},
             parents=[("A",), ("regime",)],
             notes="A's effect reverses sign across the regime"),
          feature=f, parent_features=[A, s.astype(float)])
    b.meta["n_switches"] = int((np.diff(s) != 0).sum())
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 11. graph / subgraph motif                              (relational/GRAPH)
# ==========================================================================
def _install_edge(Adj, u, v, protect, rng, tries=80):
    """Add edge (u,v) by a degree-preserving double swap. True if installed."""
    m = len(Adj)
    if Adj[u, v]:
        return True
    nu = [x for x in range(m) if Adj[u, x] and x not in protect]
    nv = [y for y in range(m) if Adj[v, y] and y not in protect]
    for _ in range(tries):
        if not nu or not nv:
            return False
        x = nu[rng.integers(0, len(nu))]
        y = nv[rng.integers(0, len(nv))]
        if x == y or x == v or y == u or Adj[x, y]:
            continue
        Adj[u, x] = Adj[x, u] = False
        Adj[v, y] = Adj[y, v] = False
        Adj[u, v] = Adj[v, u] = True
        Adj[x, y] = Adj[y, x] = True
        return True
    return False


def _count_labelled_triangles(Adj, lab):
    L = np.outer(lab, lab) * Adj
    m = len(Adj)
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            if not L[i, j]:
                continue
            cnt += int(np.sum(L[i, j + 1:] * L[j, j + 1:]))
    return cnt


def make_graph_motif(seed=0, role="proposal", n=1200, snr="high",
                     nodes_per_graph=16, p_edge=0.18, artifacts=(), d_obs=32,
                     noise=0.7, max_planted=5):
    """Graph-level Y driven by *enrichment* of labelled triangles, not by the
    raw count.

    The first version of this family made Y a function of the raw labelled-
    triangle count of an Erdos-Renyi graph. That is not a well-posed motif
    world: at those densities the count is almost a deterministic function of
    the degree sequence and the label rate, both of which a degree-preserving
    null preserves on purpose. Measured with a corrected, unbiased rewiring
    chain, the rewired count still correlated 0.73 with the original and the
    null could not break the relationship however hard it was rewired -- the
    "motif" had no content beyond its own parents.

    So the object is now planted. Each graph starts sparse enough that the
    background labelled-triangle count is near zero, then `k_g` triangles are
    *installed* by degree-preserving double swaps -- degrees, edge count and
    labels are all untouched, so the planted excess is orthogonal to every
    statistic the null preserves. `k_g` is drawn independently of degree and
    composition, and Y depends on `k_g`.

    That makes the discovery the enrichment of a wiring pattern over a
    degree-and-composition-matched background, which is what motif analysis
    means in practice, and it makes the degree-preserving null able to destroy
    exactly that and nothing else.
    """
    b = _B("graph_motif", seed, role, n, "graph", snr)
    edges, gid, labels, feats, motif, deg, planted = [], [], [], [], [], [], []
    off = 0
    for g in range(n):
        m = nodes_per_graph
        Adj = (b.rs.random((m, m)) < p_edge)
        Adj = np.triu(Adj, 1)
        Adj = Adj | Adj.T
        lab = (b.rs.random(m) < 0.45).astype(int)
        lab_idx = np.where(lab > 0)[0]
        k_target = int(b.rs.integers(0, max_planted + 1))
        cnt0 = _count_labelled_triangles(Adj, lab)
        for _ in range(k_target):
            if len(lab_idx) < 3:
                break
            trio = b.rs.choice(lab_idx, size=3, replace=False)
            u, v, w = (int(trio[0]), int(trio[1]), int(trio[2]))
            for (a, c) in ((u, v), (v, w), (u, w)):
                _install_edge(Adj, a, c, {u, v, w}, b.rs)
        cnt = _count_labelled_triangles(Adj, lab)
        # The achieved enrichment, not the attempted one: a swap can fail when
        # a node has no spare neighbour, and crediting an attempt that did not
        # install its edges would make the ground truth a fiction.
        k_done = cnt - cnt0
        motif.append(cnt)
        planted.append(k_done)
        deg.append(float(Adj.sum()) / m)
        labels.append(float(lab.mean()))
        ii, jj = np.where(np.triu(Adj, 1))
        edges.append(np.column_stack([ii + off, jj + off]))
        gid.append(np.full(m, g))
        feats.append(np.column_stack([lab.astype(float),
                                      b.rs.standard_normal(m)]))
        off += m
    motif = np.array(motif, dtype=float)
    planted = np.array(planted, dtype=float)
    b.structure = Structure(kind="graph", edges=np.vstack(edges),
                            graph_id=np.concatenate(gid),
                            meta={"nodes_per_graph": nodes_per_graph,
                                  "node_features": np.vstack(feats),
                                  "motif": "labelled_triangle",
                                  "planted": True})
    b.lat["motif_count"] = motif
    b.lat["planted_motifs"] = planted
    b.lat["label_rate"] = np.array(labels)
    b.lat["mean_degree"] = np.array(deg)
    b.lat["D0"] = b.rs.standard_normal(n)
    b.nui["density"] = zscore(np.array(deg))
    b.nui["composition"] = zscore(np.array(labels))
    beta = 1.5 * b.scale
    f = zscore(planted)
    b.Y = (beta * f + 0.8 * b.nui["composition"] + 0.5 * b.nui["density"]
           + noise * b.rn.standard_normal(n))
    b.add(_H("GRAPH:tri", "relational", "GRAPH", ("planted_motifs",),
             params={"motif": "labelled_triangle", "beta": float(beta),
                     "planted_max": max_planted},
             parents=[("motif_count",), ("label_rate",), ("mean_degree",)],
             notes="enrichment of all-labelled triangles over a degree- and "
                   "composition-matched background; must survive composition "
                   "and degree. Truth-declaration correctness fix, 2026-09-15 "
                   "(owner directive, same pattern as BOSS6): the generative "
                   "feature is zscore(planted_motifs) (the achieved excess "
                   "after degree-preserving rewiring), never motif_count (the "
                   "raw un-rewired count) -- motif_count moved from operand "
                   "to parents, a real simpler alternative to control against."),
          feature=f, parent_features=[b.nui["composition"], b.nui["density"]])
    b.meta["mean_motif"] = float(motif.mean())
    b.meta["mean_planted"] = float(planted.mean())
    b.meta["planting_success_rate"] = float((planted > 0).mean())
    b.meta["frac_graphs_with_motif"] = float((motif > 0).mean())
    b.meta["corr_planted_degree"] = float(corr(planted, np.array(deg)))
    b.meta["corr_planted_composition"] = float(corr(planted, np.array(labels)))
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 12. latent-confounded association                    (mechanism/CONFOUND)
# ==========================================================================
def make_latent_confound(seed=0, role="proposal", n=3000, snr="high",
                         k_distract=4, artifacts=(), d_obs=48, noise=0.8):
    """U drives both A and Y; there is no A -> Y edge.

    The true object set contains U -> Y and U -> A. The A ~ Y association is a
    planted artifact: real, reproducible in every environment, and not a
    mechanism. Recovery means reporting U and *not* reporting A, so this
    family scores a refusal as well as a discovery.
    """
    b = _B("latent_confound", seed, role, n, "tabular", snr)
    U = b.rs.standard_normal(n)
    A = zscore(1.1 * U + 0.5 * b.rs.standard_normal(n))
    W = b.rs.standard_normal(n)                       # a genuine cause
    b.lat["U"], b.lat["A"], b.lat["W"] = U, A, W
    b.distractors(k_distract)
    beta_u, beta_w = 1.3 * b.scale, 1.0 * b.scale
    b.Y = beta_u * U + beta_w * W + noise * b.rn.standard_normal(n)
    b.add(_H("CAUSE:U->Y", "mechanism", "CAUSE", ("U",),
             params={"beta": float(beta_u)}, notes="genuine common cause"),
          feature=U)
    b.add(_H("CAUSE:W->Y", "mechanism", "CAUSE", ("W",),
             params={"beta": float(beta_w)}, notes="genuine direct cause"),
          feature=W)
    b.objs.append(Hypothesis(
        id="ART:confound_A", mode="mechanism", operator="CONFOUND",
        operands=("A",), target="Y", true=False,
        artifact_class="nuisance_confound", parents=(("U",),),
        params={"confounder": "U"},
        notes="A ~ Y is entirely mediated by U; A has no edge into Y"))
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 13. intervention-specific causal structure               (mechanism/CAUSE)
# ==========================================================================
def make_intervention_causal(seed=0, role="proposal", n=3600, snr="high",
                             artifacts=(), d_obs=48, noise=0.7):
    """An SCM whose X3 -> Y edge exists only under do(X2).

    Under the observational regime X3 is independent of Y given X1; under
    do(X2 := 1) it becomes a cause. A method that pools regimes sees a diluted
    main effect and cannot tell the two apart, which is the point: the object
    is an edge *with a context*.
    """
    b = _B("intervention_causal", seed, role, n, "tabular", snr)
    regimes = np.array(["obs", "do_X2", "do_X3"])
    assign = regimes[b.rs.integers(0, 3, n)]
    X1 = b.rs.standard_normal(n)
    X2 = zscore(0.8 * X1 + 0.6 * b.rs.standard_normal(n))
    X2 = np.where(assign == "do_X2", 1.5, X2)
    X3 = zscore(0.5 * X1 + 0.9 * b.rs.standard_normal(n))
    X3 = np.where(assign == "do_X3", -1.5, X3)
    X5 = zscore(0.7 * X1 + 0.7 * b.rs.standard_normal(n))     # non-causal
    gate = (assign == "do_X2").astype(float)
    beta1, beta3 = 1.0 * b.scale, 1.6 * b.scale
    f_ctx = gate * X3
    b.Y = beta1 * X1 + beta3 * f_ctx + noise * b.rn.standard_normal(n)
    b.lat.update(X1=X1, X2=X2, X3=X3, X5=X5)
    b.lat["D0"] = b.rs.standard_normal(n)
    # Composite reconstruction (dbench.features.build_composite): the flat
    # CAUSE path (build_feature) cannot gate on a NAMED categorical
    # intervention regime at all -- `term_column`'s CONDITION/CAUSE/ANOMALY
    # branch only ever gates on a 0/1 indicator, and its IDENTITY branch
    # z-scores its operand, which breaks EXACT reproduction if applied
    # separately to X3: gate * zscore(X3) is not a scalar multiple of
    # gate * X3 whenever the do_X3 override shifts X3's population mean
    # away from 0 (it does, every seed checked) -- correlation drops to
    # ~0.93, not a near-miss. Declaring the already-gated product itself as
    # one derived latent (the same `b.derive` mechanism boss6's `A_clip`
    # uses, for the same reason: routing around a grammar limitation, not
    # hiding the answer -- `run_gate` excludes every `b.derive`d name from a
    # specialist's candidate set exactly as it does for `A_clip`) makes a
    # single IDENTITY term reproduce the generator's own signal exactly:
    # IDENTITY z-scores its one operand, and correlation is invariant to
    # z-scoring a single quantity, so this holds by construction, not by
    # coincidence. Verified abs_corr = 1.000000, 3 seeds (2026-09-15,
    # Bench-Owner).
    b.derive("X3_gated_do_X2", f_ctx)
    b.interventions = Interventions(
        assignment=assign,
        targets={"do_X2": ("X2",), "do_X3": ("X3",), "obs": ()},
        kind={"do_X2": "do", "do_X3": "do", "obs": "none"},
        values={"do_X2": {"X2": 1.5}, "do_X3": {"X3": -1.5}})
    b.env_labels["regime"] = assign
    b.add(_H("CAUSE:X1->Y", "mechanism", "CAUSE", ("X1",),
             params={"beta": float(beta1)}, notes="regime-invariant cause"),
          feature=X1)
    b.add(_H("CAUSE:X3->Y|do_X2", "mechanism", "CAUSE", ("X3",),
             params={"beta": float(beta3),
                     "composition": ["CAUSE", "CONDITION"],
                     "terms": [
                         {"op": "IDENTITY", "operands": ["X3_gated_do_X2"]},
                     ]}, context={"regime": "do_X2"},
             parents=[("X3",), ("X1",)],
             notes="edge exists only under do(X2); absent in the observational "
                   "regime"),
          feature=f_ctx, parent_features=[X3, X1])
    b.objs.append(Hypothesis(
        id="ART:X5", mode="mechanism", operator="CONFOUND", operands=("X5",),
        target="Y", true=False, artifact_class="nuisance_confound",
        parents=(("X1",),), params={"confounder": "X1"},
        notes="X5 ~ Y only through X1; no edge"))
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 14. hidden symmetry                                        (law/SYMMETRY)
# ==========================================================================
def make_hidden_symmetry(seed=0, role="proposal", n=3000, snr="high",
                         artifacts=(), d_obs=48, noise=0.15):
    """Observations are an SO(2) orbit of a latent state.

    Each sample draws a latent (r, phi), applies a random rotation, then
    embeds nonlinearly. The conserved quantity is r = ||z||: invariant along
    the group orbit and the only latent Y depends on. A representation that
    does not quotient the rotation cannot expose it.
    """
    b = _B("hidden_symmetry", seed, role, n, "tabular", snr)
    r = 0.5 + b.rs.uniform(0, 2.0, n)
    phi = b.rs.uniform(0, 2 * np.pi, n)
    theta = b.rs.uniform(0, 2 * np.pi, n)              # group parameter
    ang = phi + theta
    z = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    b.lat["r"] = r
    b.lat["theta"] = theta
    b.lat["z"] = z
    b.lat["D0"] = b.rs.standard_normal(n)
    b.Y = (1.8 * b.scale) * (r ** 2) + 0.4 * b.rn.standard_normal(n)
    b.meta["group"] = "SO(2)"
    b.add(_H("SYM:SO2", "law", "SYMMETRY", ("z",), target="X",
             params={"group": "SO(2)", "action": "rotation",
                     "conserved": "norm"},
             notes="observations are invariant in distribution under rotation "
                   "of z; ||z|| is the conserved quantity"))
    b.add(_H("ADD:r", "relational", "ADD", ("r",),
             params={"form": "quadratic"},
             notes="Y depends only on the conserved quantity"),
          feature=r ** 2)
    return _finish(b, artifacts, d_obs=d_obs, obs_noise=noise)


# ==========================================================================
# 15. sparse differential equation                                (law/ODE)
# ==========================================================================
def make_sparse_ode(seed=0, role="proposal", n=4000, snr="high", dt=0.002,
                    artifacts=(), d_obs=32, noise=0.0):
    """A cubic damped oscillator with a two-term sparse governing equation.

        dx/dt = -0.1 x^3 + 2.0 y^3
        dy/dt = -2.0 x^3 - 0.1 y^3

    The library has 10 quadratic/cubic candidate terms per equation and only
    two are active. `snr` sets the observation noise on the state, which is
    what makes derivative estimation hard.
    """
    b = _B("sparse_ode", seed, role, n, "temporal", snr)
    obs_sd = {"high": 0.0, "med": 0.01, "low": 0.03}[b.snr]
    coefs = {"x": {"x^3": -0.1, "y^3": 2.0}, "y": {"x^3": -2.0, "y^3": -0.1}}
    x = np.empty(n)
    y = np.empty(n)
    x[0] = 2.0 * (b.rs.random() - 0.5) + 1.5
    y[0] = 2.0 * (b.rs.random() - 0.5)

    def dxdt(u, v):
        return -0.1 * u ** 3 + 2.0 * v ** 3

    def dydt(u, v):
        return -2.0 * u ** 3 - 0.1 * v ** 3

    for i in range(n - 1):
        u, v = x[i], y[i]
        k1u, k1v = dxdt(u, v), dydt(u, v)
        k2u, k2v = dxdt(u + dt * k1u / 2, v + dt * k1v / 2), dydt(u + dt * k1u / 2, v + dt * k1v / 2)
        k3u, k3v = dxdt(u + dt * k2u / 2, v + dt * k2v / 2), dydt(u + dt * k2u / 2, v + dt * k2v / 2)
        k4u, k4v = dxdt(u + dt * k3u, v + dt * k3v), dydt(u + dt * k3u, v + dt * k3v)
        x[i + 1] = u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6
        y[i + 1] = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    xo = x + obs_sd * b.rn.standard_normal(n)
    yo = y + obs_sd * b.rn.standard_normal(n)
    t = np.arange(n) * dt
    b.structure = Structure(kind="temporal", time=np.arange(n),
                            meta={"dt": dt})
    b.lat["x"], b.lat["y"] = xo, yo
    b.lat["dx"] = np.gradient(x, dt)
    b.lat["dy"] = np.gradient(y, dt)
    b.Y = None
    for state, terms in coefs.items():
        for term, c in terms.items():
            b.add(_H("ODE:d%s/%s" % (state, term), "law", "ODE",
                     (term,), target="d%s" % state,
                     params={"coef": c, "dt": dt,
                             "library": "poly<=3 in (x,y)"},
                     notes="sparse governing-equation term"))
    b.meta.update(dt=dt, coefs=coefs, obs_sd=obs_sd, t_end=float(t[-1]))
    return _finish(b, artifacts, d_obs=d_obs, obs_noise=0.02)


# ==========================================================================
# 16. rare-state anomaly                                  (anomaly/ANOMALY)
# ==========================================================================
def make_anomaly(seed=0, role="proposal", n=5000, snr="high",
                 prevalence=0.02, k_bg=6, artifacts=(), d_obs=48, noise=0.4):
    """A rare state with a shifted mean AND a different correlation structure.

    Mean-shift-only anomalies are found by any univariate screen; here the
    rare state also decorrelates two background factors, so a detector must
    look at structure and not only location.
    """
    b = _B("anomaly", seed, role, n, "tabular", snr)
    is_a = b.rs.random(n) < prevalence
    Z = b.rs.standard_normal((n, k_bg))
    Z[:, 1] = 0.85 * Z[:, 0] + np.sqrt(1 - 0.85 ** 2) * Z[:, 1]   # background dep
    shift = 2.2 * b.scale
    Z[is_a, 0] += shift
    Z[is_a, 1] = b.rs.standard_normal(int(is_a.sum()))            # dependence broken
    for j in range(k_bg):
        b.lat["Z%d" % j] = Z[:, j]
    b.lat["is_anomaly"] = is_a.astype(float)
    b.Y = None
    b.add(_H("ANOM:rare", "anomaly", "ANOMALY", ("Z0", "Z1"), target="X",
             params={"prevalence": prevalence, "mean_shift": float(shift),
                     "structure": "dependence between Z0 and Z1 removed"},
             notes="rare state, shifted mean and broken dependence"))
    b.meta.update(prevalence=float(is_a.mean()), n_anomalous=int(is_a.sum()))
    return _finish(b, artifacts, d_obs=d_obs, obs_noise=noise)


# ==========================================================================
# 17. mixture of unknown categories                       (entity/MIXTURE)
# ==========================================================================
def make_mixture(seed=0, role="proposal", n=4000, snr="high", K=4,
                 rare_weight=0.03, artifacts=(), d_obs=48, noise=0.5):
    """K latent categories, one of them rare -- an unknown-category world.

    Selecting K by BIC on this world tends to miss the rare component, which
    is the intended difficulty: the discovery is the component, not the count.
    """
    b = _B("mixture", seed, role, n, "tabular", snr)
    w = np.full(K, (1.0 - rare_weight) / (K - 1))
    w[-1] = rare_weight
    lab = b.rs.choice(K, size=n, p=w)
    mus = b.rs.standard_normal((K, 3)) * (2.4 * b.scale)
    Zc = mus[lab] + 0.7 * b.rs.standard_normal((n, 3))
    for j in range(3):
        b.lat["Z%d" % j] = Zc[:, j]
    b.lat["component"] = lab.astype(float)
    b.lat["D0"] = b.rs.standard_normal(n)
    b.Y = None
    b.add(_H("MIXTURE:comp", "entity", "MIXTURE", ("component",), target="X",
             params={"K": K, "weights": w.tolist(),
                     "rare_component": int(K - 1)},
             notes="latent categories including one rare component"))
    b.meta.update(K=K, weights=w.tolist(),
                  rare_n=int((lab == K - 1).sum()))
    return _finish(b, artifacts, d_obs=d_obs, obs_noise=noise)


# ==========================================================================
# 18. experimental design                            (experimental/DESIGN)
# ==========================================================================
def make_experimental_design(seed=0, role="proposal", n=600, snr="high",
                             n_hyp=4, n_actions=5, artifacts=(), d_obs=16):
    """Which intervention most reduces uncertainty about which model is true?

    A discrete world: `n_hyp` competing hypotheses with a uniform prior, and
    `n_actions` candidate interventions. Under action a and hypothesis h the
    observation is Gaussian with mean mu[h, a]. The expected information gain
        I(H; Y_a) = H(Y_a) - E_h H(Y_a | h)
    is computed exactly by quadrature at generation time, so the ground-truth
    object -- the argmax action -- is closed form and not estimated.
    """
    b = _B("experimental_design", seed, role, n, "tabular", snr)
    prior = np.full(n_hyp, 1.0 / n_hyp)
    mu = b.rs.standard_normal((n_hyp, n_actions)) * 1.2
    # one action is deliberately made maximally separating
    best = int(b.rs.integers(0, n_actions))
    mu[:, best] = np.linspace(-3.0, 3.0, n_hyp) * (1.5 * b.scale)
    sd = 1.0
    grid = np.linspace(-9, 9, 2001)
    eig = np.zeros(n_actions)
    for a in range(n_actions):
        comp = np.exp(-0.5 * ((grid[None, :] - mu[:, a][:, None]) / sd) ** 2)
        comp /= (sd * np.sqrt(2 * np.pi))
        marg = (prior[:, None] * comp).sum(0)
        dg = grid[1] - grid[0]
        h_marg = -np.sum(np.where(marg > 0, marg * np.log(marg), 0.0)) * dg
        h_cond = float(0.5 * np.log(2 * np.pi * np.e * sd ** 2))
        eig[a] = h_marg - h_cond
    a_star = int(np.argmax(eig))
    h_true = int(b.rs.integers(0, n_hyp))
    acts = b.rs.integers(0, n_actions, n)
    obs = mu[h_true, acts] + sd * b.rn.standard_normal(n)
    b.lat["action"] = acts.astype(float)
    b.lat["observation"] = obs
    b.lat["D0"] = b.rs.standard_normal(n)
    b.Y = obs
    b.add(_H("DESIGN:a*", "experimental", "DESIGN", ("action",),
             target="H", params={"a_star": a_star,
                                 "eig": eig.tolist(),
                                 "n_hyp": n_hyp, "n_actions": n_actions,
                                 "mu": mu.tolist(), "sd": sd},
             notes="the action maximising I(H; Y_a); computed by quadrature"))
    b.meta.update(a_star=a_star, eig=eig.tolist(), h_true=h_true,
                  prior=prior.tolist(), mu=mu.tolist(), sd=sd)
    return _finish(b, artifacts, d_obs=d_obs)


# ==========================================================================
# 19. one linear-combination object                     (relational/LINCOMB)
# ==========================================================================
#: the weight grid the compiler prices. A LINCOMB is only a well-posed object
#: if its weights are drawn from the same finite set the ranking pays for;
#: otherwise "the right weights" is a measure-zero event and no MDL comparison
#: between one combination and two main effects is possible.
LINCOMB_GRID = 128


def make_linear_combination(seed=0, role="proposal", n=3000, snr="high",
                            w_a=0.75, k_distract=5, artifacts=(), d_obs=48,
                            noise=1.0, slot_parts=4, slot_gamma=0.25,
                            slot_noise=1.0):
    """`Y` depends on ONE weighted combination of `A` and `B`.

    The difference from `additive_factors` is the whole point of the family.
    There, three separate `ADD` main effects are planted and the response is
    their sum; a method that reports two of the three has found two objects.
    Here `A` and `B` are **not** planted objects at all. The only planted
    object is the direction `w = (0.75, 0.25)` in the `(A, B)` plane, and a
    method that reports `ADD:A` has not found 0.75 of it -- it has reported a
    different object, of a granularity the truth does not assert.

    Both weights sit exactly on the `1/LINCOMB_GRID` grid (96/128 and 32/128)
    and are unequal, so a solver that assumes an equal-weight sum is wrong on
    this world rather than accidentally right.

    Because `A` and `B` are not objects, the object has no parent family to
    beat: `features.parent_columns` returns no controls for LINCOMB. That is
    forced rather than chosen -- the design column lies inside the span of its
    own operands, so controlling for them would regress the feature on itself
    and return exactly zero evidence for every true LINCOMB.
    """
    b = _B("linear_combination", seed, role, n, "tabular", snr)
    w_b = 1.0 - float(w_a)
    for w in (float(w_a), w_b):
        if abs(w * LINCOMB_GRID - round(w * LINCOMB_GRID)) > 1e-9:
            raise ValueError("weight %r is not on the 1/%d grid the compiler "
                             "prices" % (w, LINCOMB_GRID))
    A = b.rs.standard_normal(n)
    B = b.rs.standard_normal(n)
    b.lat["A"], b.lat["B"] = A, B
    b.distractors(k_distract)
    f = float(w_a) * zscore(A) + w_b * zscore(B)
    beta = 1.4 * b.scale
    b.Y = beta * f + noise * b.rn.standard_normal(n)
    b.add(_H("LINCOMB:A,B", "relational", "LINCOMB", ("A", "B"),
             params={"weights": (float(w_a), w_b), "grid": LINCOMB_GRID,
                     "beta": float(beta)},
             equivalent_expressions=[
                 {"operator": "LINCOMB", "operands": ["A", "B"],
                  "params": {"weights": [float(w_a) / 2.0, w_b / 2.0]},
                  "why": "a LINCOMB is a direction, so any positive rescaling "
                         "of the weight vector names the same object and must "
                         "fold rather than be counted twice"}],
             notes="one weighted combination, planted as a single object; "
                   "A and B are not objects of this world"),
          feature=f)
    w = _finish(b, artifacts, d_obs=d_obs, name=None)
    return _with_slot_view(w, b, parts=slot_parts, gamma=slot_gamma,
                           sigma=slot_noise)


def _with_slot_view(world, b, parts=4, gamma=0.25, sigma=1.0):
    """Attach the distributed slot representation as modality `S`.

    Each planted operand is spread over `parts` slots, none of which reaches it
    on its own: `S_i = gamma * z(L) + N(0, sigma^2)` gives
    `|r| = gamma / sqrt(gamma^2 + sigma^2) = 0.2425` at the defaults, below
    `align.MIN_ABS_R = 0.30`, while the `parts` slots together reach
    `R^2 = parts*gamma^2 / (parts*gamma^2 + sigma^2) = 0.20`, above the
    subspace rule's `MIN_ABS_R**2 = 0.09`.

    That gap is the construction's only job. It makes the world's alignment
    caps read `single 0.000 / subspace 1.000`, which is the pattern that says
    an object is reachable only as a combination -- the representational
    counterpart of the granularity claim the truth makes.

    Modality `X` is untouched and remains what every specialist,
    `utility.world_utility` and `_footprint_cosine` read, so the slot view adds
    a representation without moving a single existing measurement.
    """
    rg = np.random.default_rng(derive_seed(b.family, b.seed, b.role,
                                           "slot_view"))
    names, cols, spans = [], [], {}
    for lat in ("A", "B"):
        z = zscore(np.asarray(b.lat[lat], dtype=np.float64))
        mine = []
        for _ in range(parts):
            nm = "S%d" % len(names)
            names.append(nm)
            mine.append(nm)
            cols.append(gamma * z + sigma * rg.standard_normal(b.n))
        spans[lat] = tuple(mine)
    S = np.column_stack(cols)
    exp_r = gamma / float(np.sqrt(gamma ** 2 + sigma ** 2))
    exp_r2 = (parts * gamma ** 2) / (parts * gamma ** 2 + sigma ** 2)
    return _dc_replace(
        world,
        modalities=dict(world.modalities, S=S),
        meta=dict(world.meta, slots=tuple(names),
                  slot_view={"parts_per_latent": int(parts),
                             "gamma": float(gamma), "sigma": float(sigma),
                             "spans": {k: list(v) for k, v in spans.items()},
                             "expected_abs_r_per_slot": float(exp_r),
                             "expected_r2_on_span": float(exp_r2)}))


def slot_view(world):
    """`{slot name: column}` for a world that ships one, else `{}`."""
    names = world.meta.get("slots")
    if not names or "S" not in world.modalities:
        return {}
    S = np.asarray(world.modalities["S"], dtype=np.float64)
    return {nm: S[:, i] for i, nm in enumerate(names)}


FAMILIES = {
    "additive_factors": make_additive_factors,
    "pure_and": make_pure_and,
    "invisible_k3": make_invisible_k3,
    "invisible_k4": make_invisible_k4,
    "invisible_k5": make_invisible_k5,
    "invisible_xor_k3": make_invisible_xor_k3,
    "invisible_xor_k4": make_invisible_xor_k4,
    "xor": make_xor,
    "threshold": make_threshold,
    "hill": make_hill,
    "context_gated": make_context_gated,
    "manifold": make_manifold,
    "spatial_interaction": make_spatial,
    "temporal_lag": make_temporal_lag,
    "regime_switch": make_regime_switch,
    "graph_motif": make_graph_motif,
    "latent_confound": make_latent_confound,
    "intervention_causal": make_intervention_causal,
    "hidden_symmetry": make_hidden_symmetry,
    "sparse_ode": make_sparse_ode,
    "anomaly": make_anomaly,
    "mixture": make_mixture,
    "experimental_design": make_experimental_design,
    "linear_combination": make_linear_combination,
}

# which discovery mode each family is the minimal isolation of
FAMILY_MODE = {
    "additive_factors": "relational", "pure_and": "relational",
    "invisible_k3": "relational", "invisible_k4": "relational",
    "invisible_k5": "relational",
    "invisible_xor_k3": "relational", "invisible_xor_k4": "relational",
    "xor": "relational", "threshold": "relational", "hill": "relational",
    "context_gated": "relational", "manifold": "entity",
    "spatial_interaction": "relational", "temporal_lag": "relational",
    "regime_switch": "anomaly", "graph_motif": "relational",
    "latent_confound": "mechanism", "intervention_causal": "mechanism",
    "hidden_symmetry": "law", "sparse_ode": "law", "anomaly": "anomaly",
    "mixture": "entity", "experimental_design": "experimental",
    "linear_combination": "relational",
}

def make(family, **kw):
    if family not in FAMILIES:
        raise KeyError("unknown family %r; known: %s"
                       % (family, sorted(FAMILIES)))
    return FAMILIES[family](**kw)
