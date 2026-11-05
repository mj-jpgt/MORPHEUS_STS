"""MORPHEUS DiscoveryBench -- the world interface.

Everything in DiscoveryBench is a `World`. A world is a fully specified
scientific micro-universe in which the answer is known before any method runs:
which latent variables exist, how they were combined, which observed modality
carries them, what nuisance structure sits on top, and which candidate objects
are true versus planted-false.

The interface is deliberately wide, because the six discovery modes need
different things:

  entity        needs latents and a modality
  relational    needs latents, a target and typed operators
  mechanism     needs environments and interventions
  law           needs time, state derivatives and a group action
  anomaly       needs a regime/label channel
  experimental  needs a candidate action set and a posterior over hypotheses

A generator that cannot fill a field leaves it None. Consumers must check.

Contract with the Hypothesis Compiler
-------------------------------------
A world exposes exactly seven things the compiler may read at proposal time:

    world.latents        oracle latent variables            dict[str, (n,) or (n,d)]
    world.modalities     observed data                      dict[str, (n, d_m)]
    world.env            environment metadata E             EnvMeta
    world.nuisance       known nuisance covariates C        dict[str, (n,) or (n,d)]
    world.structure      spatial / temporal / graph G       Structure
    world.interventions  interventions I                    Interventions | None
    world.target         the response, when the family has one   (n,) | None

and exactly one thing it may NOT read at proposal time:

    world.truth          the typed ground-truth object set  TruthSet

`World.blind()` returns a copy with `truth` replaced by a tombstone that raises
on access, so a compiler cannot accidentally peek. Scoring calls `world.truth`
on the unblinded handle held by the harness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------
# operator grammar -- shared vocabulary with the Hypothesis Compiler.
# These strings are the only legal values of Hypothesis.operator.
# --------------------------------------------------------------------------
OPERATORS = (
    "ADD",        # additive main effect
    "AND",        # multiplicative conjunction (pure AND: zero marginal signal)
    "XOR",        # parity / exclusive-or
    "THRESHOLD",  # sigma((A - tau)/T)
    "SATURATE",   # Hill A^n / (K^n + A^n)
    "CONDITION",  # 1[C = c] * inner
    "RATIO",
    "DIFF",
    "SPACE",      # Agg over the neighbourhood N(i)
    "LAG",        # B at time t - l
    "GRAPH",      # subgraph-motif indicator
    "CAUSE",      # intervention-sensitive directed edge
    "SYMMETRY",   # invariance under a group action
    "ODE",        # sparse governing-equation term
    "ANOMALY",    # rare-state / regime membership
    "MANIFOLD",   # chart coordinate on a curved/branching manifold
    "MIXTURE",    # latent category membership
    "CONFOUND",   # nuisance-induced association (always a false object)
    "DESIGN",     # experimental: an action's expected information gain
    "IDENTITY",   # a bare latent state, used as an entity-mode object
    "LINCOMB",    # one weighted combination of several latents, planted as a
                  # SINGLE object -- not the sum of several ADDs
)

DISCOVERY_MODES = (
    "entity",
    "relational",
    "mechanism",
    "law",
    "anomaly",
    "experimental",
)

DOMAINS = (
    "molecular",     # cell x feature tables, sparse, batch-structured
    "spatial",       # spots on a tissue, autocorrelated, composition-structured
    "temporal",      # time series / reanalysis fields
    "graph",         # graphs with node features (materials / catalysts)
    "particle",      # unordered sets of objects per event
    "tabular",       # generic i.i.d. design matrices
)


# --------------------------------------------------------------------------
# structure G
# --------------------------------------------------------------------------
@dataclass
class Structure:
    """Spatial / temporal / graph scaffolding shared by all modalities.

    `kind` is the primary scaffolding. A world may carry more than one (a
    spatiotemporal world sets kind="spatiotemporal" and fills coords, time and
    unit), so consumers test the field they need rather than `kind`.
    """
    kind: str = "iid"                      # iid|spatial|temporal|spatiotemporal|graph
    coords: np.ndarray | None = None       # (n, 2) spatial positions
    time: np.ndarray | None = None         # (n,) integer time index
    unit: np.ndarray | None = None         # (n,) unit id, for panel/repeat data
    neighbors: list | None = None          # per-row neighbour row-indices
    edges: np.ndarray | None = None        # (n_edges, 2) for graph worlds
    graph_id: np.ndarray | None = None     # (n,) which graph each node is in
    meta: dict = field(default_factory=dict)

    def n_neighbors(self) -> np.ndarray:
        if self.neighbors is None:
            raise ValueError("world has no neighbourhood structure")
        return np.array([len(v) for v in self.neighbors])

    def _incidence(self):
        """Row-major (row, neighbour) index pair, built once and cached.

        The SPACE operator is evaluated once per candidate program, and a
        composite search evaluates thousands, so a per-row Python loop over
        neighbourhoods dominates the whole gate. Flattening the neighbour lists
        once turns every aggregation into a single `np.add.reduceat`.
        """
        cached = getattr(self, "_inc_cache", None)
        if cached is None:
            counts = np.array([len(v) for v in self.neighbors], dtype=np.int64)
            flat = (np.concatenate([np.asarray(v, dtype=np.int64)
                                    for v in self.neighbors])
                    if counts.sum() else np.zeros(0, dtype=np.int64))
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            cached = (flat, counts, starts)
            object.__setattr__(self, "_inc_cache", cached)
        return cached

    def aggregate(self, x: np.ndarray, how: str = "mean") -> np.ndarray:
        """Agg over N(i) -- the kernel of the SPACE and GRAPH operators."""
        if self.neighbors is None:
            raise ValueError("world has no neighbourhood structure")
        x = np.asarray(x, dtype=np.float64)
        if how == "max":
            out = np.zeros(len(x), dtype=np.float64)
            for i, nb in enumerate(self.neighbors):
                out[i] = float(np.max(x[nb])) if len(nb) else 0.0
            return out
        if how not in ("mean", "sum", "majority"):
            raise ValueError(how)
        flat, counts, starts = self._incidence()
        out = np.zeros(len(x), dtype=np.float64)
        nz = counts > 0
        if flat.size:
            sums = np.add.reduceat(x[flat], starts)
            out[nz] = sums[nz]
        if how == "sum":
            return out
        out[nz] = out[nz] / counts[nz]
        return (out > 0.5).astype(np.float64) if how == "majority" else out

    #: aggregation kinds the SPACE and GRAPH operators may use. Part of the
    #: shared grammar: a specialist or the Hypothesis Compiler searches over
    #: these, so adding one widens the search space for every consumer.
    AGGREGATIONS = ("mean", "sum", "max", "majority")


# --------------------------------------------------------------------------
# interventions I
# --------------------------------------------------------------------------
@dataclass
class Interventions:
    """Which rows were intervened on, and where.

    `assignment[i]` names the intervention applied to row i ("obs" for the
    observational regime). `targets[name]` lists the latent variable names the
    intervention sets, and `kind[name]` is "do" (hard) or "shift" (soft).
    """
    assignment: np.ndarray                        # (n,) str
    targets: dict = field(default_factory=dict)
    kind: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)

    @property
    def regimes(self) -> tuple:
        return tuple(sorted(set(np.asarray(self.assignment).tolist())))

    def mask(self, name: str) -> np.ndarray:
        return np.asarray(self.assignment) == name


# --------------------------------------------------------------------------
# environment metadata E
# --------------------------------------------------------------------------
@dataclass
class EnvMeta:
    """Per-row environment labels plus world-level provenance.

    `labels` holds the grouping variables that define independent replication:
    donor, site, patient, year, run, generator seed. Replication environments
    are generated with different values here and a different noise draw, but
    the same true objects.
    """
    role: str = "proposal"                 # proposal|replication|null|utility
    labels: dict = field(default_factory=dict)
    world_seed: int = 0
    template: str = ""
    tags: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# typed ground-truth objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Hypothesis:
    """One typed scientific object -- a node/edge/hyperedge of the object graph.

    This is the object, not its class label. Scoring in DiscoveryBench is
    against (operator, operands, target, context), never against a predicted
    discovery-type string: a system that says "this is a relational discovery"
    without naming the operands has recovered nothing.
    """
    id: str
    mode: str                              # one of DISCOVERY_MODES
    operator: str                          # one of OPERATORS
    operands: tuple                        # latent names, order-insensitive
    target: str                            # latent or observed name
    params: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)     # the Gamma conditions
    lag: int | None = None
    effect: float = float("nan")           # true standardised effect size
    true: bool = True                      # False => planted artifact
    artifact_class: str | None = None
    parents: tuple = ()                    # P(h): operand subsets
    equivalent_expressions: tuple = ()     # other operand orders / forms that
                                           # denote this same object
    notes: str = ""

    def key(self) -> tuple:
        """Canonical identity used for matching. Operand order does not
        matter; operator, target, context and lag do."""
        return (
            self.operator,
            tuple(sorted(self.operands)),
            self.target,
            self.lag,
            tuple(sorted((str(k), str(v)) for k, v in self.context.items())),
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "mode": self.mode, "operator": self.operator,
            "operands": list(self.operands), "target": self.target,
            "params": _jsonable(self.params), "context": _jsonable(self.context),
            "lag": self.lag, "effect": _num(self.effect), "true": self.true,
            "artifact_class": self.artifact_class,
            "parents": [list(p) for p in self.parents], "notes": self.notes,
        }
        # Included only when non-empty, deliberately: `to_dict` feeds the world
        # fingerprint, so emitting an empty field for every object would move
        # every world's hash to record nothing. Only worlds that actually
        # declare an equivalence move.
        if self.equivalent_expressions:
            d["equivalent_expressions"] = _jsonable(
                list(self.equivalent_expressions))
        return d


@dataclass
class TruthSet:
    """Everything a scorer is allowed to know and a discoverer is not."""
    objects: tuple = ()

    @property
    def true_objects(self) -> tuple:
        return tuple(h for h in self.objects if h.true)

    @property
    def artifacts(self) -> tuple:
        return tuple(h for h in self.objects if not h.true)

    def by_mode(self, mode: str) -> tuple:
        return tuple(h for h in self.objects if h.mode == mode)

    def by_id(self, hid: str):
        for h in self.objects:
            if h.id == hid:
                return h
        raise KeyError(hid)

    def keys(self) -> set:
        return {h.key() for h in self.true_objects}

    def to_list(self) -> list:
        return [h.to_dict() for h in self.objects]


class _SealedTruth:
    """Tombstone left in place of TruthSet on a blinded world."""

    def __getattr__(self, item):
        raise PermissionError(
            "this world is blinded: ground truth is held by the harness, not "
            "by the discovery method. Score through dbench.metrics instead.")

    def __repr__(self):
        return "<sealed truth>"


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------
@dataclass
class World:
    name: str
    family: str
    domain: str
    latents: dict = field(default_factory=dict)
    modalities: dict = field(default_factory=dict)
    target: np.ndarray | None = None
    target_name: str = "Y"
    nuisance: dict = field(default_factory=dict)
    env: EnvMeta = field(default_factory=EnvMeta)
    structure: Structure = field(default_factory=Structure)
    interventions: Interventions | None = None
    truth: Any = field(default_factory=TruthSet)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        if self.target is not None:
            return len(self.target)
        for v in self.latents.values():
            return len(v)
        for v in self.modalities.values():
            return len(v)
        raise ValueError("empty world")

    def oracle_matrix(self, names: Sequence | None = None):
        """Oracle latents as a design matrix, with the column-name list.

        This is what step 2 of the build plan (Hypothesis Compiler on oracle
        features) consumes. 2-D latents are expanded to one column per
        component with a name[j] label.
        """
        names = list(self.latents) if names is None else list(names)
        cols, labels = [], []
        for nm in names:
            v = np.asarray(self.latents[nm])
            if v.ndim == 1:
                cols.append(v.astype(np.float64))
                labels.append(nm)
            else:
                for j in range(v.shape[1]):
                    cols.append(v[:, j].astype(np.float64))
                    labels.append("%s[%d]" % (nm, j))
        return (np.column_stack(cols) if cols else np.zeros((self.n, 0))), labels

    def blind(self) -> "World":
        return replace(self, truth=_SealedTruth())

    def fingerprint(self) -> str:
        """Content hash of the world's data and its truth set.

        Two worlds with the same fingerprint are byte-identical in everything a
        method or a scorer can see. Used to freeze manifests.
        """
        h = hashlib.sha256()
        h.update(("%s|%s|%s" % (self.name, self.family, self.domain)).encode())
        for d, tag in ((self.latents, "L"), (self.modalities, "M"),
                       (self.nuisance, "C")):
            for k in sorted(d):
                h.update(("%s:%s" % (tag, k)).encode())
                h.update(np.ascontiguousarray(
                    np.asarray(d[k], dtype=np.float64)).tobytes())
        if self.target is not None:
            h.update(b"Y")
            h.update(np.ascontiguousarray(
                np.asarray(self.target, dtype=np.float64)).tobytes())
        for f in ("coords", "time", "unit", "edges", "graph_id"):
            v = getattr(self.structure, f)
            if v is not None:
                h.update(f.encode())
                h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
        if self.interventions is not None:
            h.update(b"I")
            h.update("|".join(
                np.asarray(self.interventions.assignment).astype(str)).encode())
        if isinstance(self.truth, TruthSet):
            h.update(json.dumps(self.truth.to_list(), sort_keys=True).encode())
        return h.hexdigest()

    def portable_digest(self, sig: int = 6) -> str:
        """A machine-independent identity for this world.

        `fingerprint` hashes raw float64 and is the right check *within* one
        environment: it sees any bit change, which is what caught the
        thread-dependent BLAS calls. It cannot travel. Measured between two
        machines at one thread each -- Windows/numpy 2.4.3 and Linux/numpy
        2.2.6 -- 22 of 180 generated arrays disagreed, every one of them by one
        or two units in the last place, at most 2.8e-16 relative. The cause is
        not a defect and cannot be engineered away in this package: IEEE-754
        mandates correct rounding for arithmetic and `sqrt`, and not for `exp`,
        `log`, `pow`, `sin` or `cos`, so two conforming platforms may return
        adjacent doubles. `gp_field`'s Gaussian taps, `soft_threshold`'s
        sigmoid and `hill`'s powers all go through those functions.

        This digest is therefore deliberately *not* byte-exact. It commits to
        the structure -- names, shapes, the typed object set, the intervention
        regimes -- exactly, and to the numerical content only through summary
        statistics formatted to `sig` significant digits. With `sig = 6` the
        quantisation is ten orders of magnitude coarser than the observed
        platform noise, so it is stable in practice, and it still separates any
        world whose seed, parameters, object set or sampling changed.

        What it does *not* detect is sub-part-per-million numerical drift; that
        is `fingerprint`'s job, on the machine that produced it. Report both:
        `portable_digest` says a consumer received the worlds that were frozen,
        `fingerprint` says nothing has moved since.
        """
        fmt = "%%.%dg" % sig
        quantum = 10 ** sig

        def _qz(x, scale):
            """Fixed-point quantisation *relative to the array's scale*.

            Significant-digit rounding is the wrong tool here and quietly
            reintroduces the problem this digest exists to avoid. A z-scored
            nuisance has a mean of order 1e-16 -- exactly zero, numerically --
            and "%.6g" of -2.406e-16 and of -2.411e-16 are different strings,
            so two machines agreeing to the last bit on every value would
            still disagree on the digest. That is not hypothetical: it was
            what made temporal_lag look like a structural change between the
            two machines when nothing about it had moved. Dividing by the
            array's scale first and rounding onto a fixed grid maps anything
            below half a millionth of the scale to the same integer, which is
            what "agrees to six significant figures of the data" should mean.
            """
            if not np.isfinite(x):
                return "nan"
            return int(np.rint(float(x) / scale * quantum))

        def q(a):
            a = np.asarray(a, dtype=np.float64).ravel()
            if a.size == 0:
                return "empty"
            fin = np.isfinite(a)
            b = a[fin]
            if b.size == 0:
                return "nonfinite:%d" % a.size
            scale = float(np.max(np.abs(b)))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            st = [b.mean(), b.std(), b.min(), b.max(),
                  np.quantile(b, 0.25), np.quantile(b, 0.5),
                  np.quantile(b, 0.75)]
            exp = int(np.floor(np.log10(scale)))
            mant = int(np.rint(scale / (10.0 ** exp) * quantum))
            return "|".join(str(_qz(v, scale)) for v in st) + (
                "|s=%de%d|n=%d|nf=%d" % (mant, exp, a.size, int((~fin).sum())))

        h = hashlib.sha256()
        h.update(("W|%s|%s|%s|%d" % (self.name, self.family, self.domain,
                                     self.n)).encode())
        for d, tag in ((self.latents, "L"), (self.modalities, "M"),
                       (self.nuisance, "C")):
            for k in sorted(d):
                v = np.asarray(d[k])
                h.update(("%s:%s:%s:" % (tag, k, "x".join(map(str, v.shape))
                                         )).encode())
                h.update(q(v).encode())
        if self.target is not None:
            h.update(b"Y:")
            h.update(q(self.target).encode())
        st = self.structure
        h.update(("G|%s" % st.kind).encode())
        for f in ("coords", "time", "unit", "edges", "graph_id"):
            v = getattr(st, f)
            h.update(("|%s:%s" % (f, "none" if v is None else
                                  "x".join(map(str, np.shape(v))))).encode())
        if st.neighbors is not None:
            deg = np.bincount(np.array([len(x) for x in st.neighbors]))
            h.update(("|deg:" + ",".join(map(str, deg.tolist()))).encode())
        if self.interventions is not None:
            a = np.asarray(self.interventions.assignment).astype(str)
            for r in sorted(set(a.tolist())):
                h.update(("|I:%s=%d" % (r, int((a == r).sum()))).encode())
        h.update(b"|E:" + ("%s|%s" % (self.env.role,
                                      ",".join(sorted(self.env.labels)))).encode())
        if isinstance(self.truth, TruthSet):
            rows = []
            for o in self.truth.to_list():
                o = dict(o)
                e = o.get("effect")
                # effects are correlations on [-1, 1] and are *computed*, so
                # they get the same fixed-point treatment: an effect of 3e-17
                # and one of 8e-17 are both zero.
                if isinstance(e, float) and np.isfinite(e):
                    o["effect"] = int(np.rint(e * quantum))
                elif isinstance(e, float):
                    o["effect"] = "nan"
                o["params"] = _round_floats(o.get("params"), sig)
                rows.append(o)
            h.update(json.dumps(rows, sort_keys=True).encode())
        h.update(("|derived:" + ",".join(
            sorted(self.meta.get("derived_latents", ()) or ()))).encode())
        return h.hexdigest()

    def summary(self) -> dict:
        return {
            "name": self.name, "family": self.family, "domain": self.domain,
            "n": self.n,
            "latents": {k: list(np.shape(v)) for k, v in self.latents.items()},
            "modalities": {k: list(np.shape(v)) for k, v in self.modalities.items()},
            "nuisance": sorted(self.nuisance),
            "structure": self.structure.kind,
            "env_role": self.env.role,
            "env_labels": sorted(self.env.labels),
            "interventions": (list(self.interventions.regimes)
                              if self.interventions else None),
            "n_true": (len(self.truth.true_objects)
                       if isinstance(self.truth, TruthSet) else None),
            "n_artifact": (len(self.truth.artifacts)
                           if isinstance(self.truth, TruthSet) else None),
            "fingerprint": self.fingerprint(),
            "portable_digest": self.portable_digest(),
        }


# --------------------------------------------------------------------------
# episodes: E = (D_P, D_R, D_N, D_U)
# --------------------------------------------------------------------------
@dataclass
class Episode:
    """One scientific episode.

    D_P   proposal environment
    D_R   one or more replication environments, INDEPENDENTLY GENERATED:
          new noise, new nuisance realisations, new environment labels, same
          true objects. Row i of D_R has no correspondence to row i of D_P.
    D_N   structured-null environments: realistic nuisance preserved, only the
          relationship under test destroyed.
    D_U   utility environment, for the U >= U_0 - delta constraint.
    """
    name: str
    D_P: World
    D_R: tuple
    D_N: tuple
    D_U: World | None = None
    meta: dict = field(default_factory=dict)

    def blind(self) -> "Episode":
        return Episode(
            name=self.name, D_P=self.D_P.blind(),
            D_R=tuple(w.blind() for w in self.D_R),
            D_N=tuple(w.blind() for w in self.D_N),
            D_U=self.D_U.blind() if self.D_U is not None else None,
            meta=dict(self.meta))

    def truth(self) -> TruthSet:
        return self.D_P.truth

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode())
        worlds = (self.D_P,) + tuple(self.D_R) + tuple(self.D_N)
        if self.D_U is not None:
            worlds = worlds + (self.D_U,)
        for w in worlds:
            h.update(w.fingerprint().encode())
        return h.hexdigest()

    def summary(self) -> dict:
        return {
            "name": self.name,
            "D_P": self.D_P.summary(),
            "D_R": [w.summary() for w in self.D_R],
            "D_N": [w.summary() for w in self.D_N],
            "D_U": self.D_U.summary() if self.D_U else None,
            "fingerprint": self.fingerprint(),
        }


# --------------------------------------------------------------------------
# world specs -- the frozen, hashable description a world is generated from
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldSpec:
    family: str
    params: tuple                          # sorted (key, value) pairs
    seed: int
    role: str = "proposal"

    @staticmethod
    def make(family: str, seed: int, role: str = "proposal", **params) -> "WorldSpec":
        return WorldSpec(
            family=family,
            params=tuple(sorted((k, _hashable(v)) for k, v in params.items())),
            seed=seed, role=role)

    def as_dict(self) -> dict:
        return dict(self.params)

    def with_(self, **over) -> "WorldSpec":
        d = self.as_dict()
        seed = over.pop("seed", self.seed)
        role = over.pop("role", self.role)
        d.update(over)
        return WorldSpec.make(self.family, seed, role, **d)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            {"family": self.family, "params": [list(p) for p in self.params],
             "seed": self.seed, "role": self.role},
            sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"family": self.family, "params": {k: v for k, v in self.params},
                "seed": self.seed, "role": self.role, "digest": self.digest()}


def _round_floats(o, sig):
    """Recursively format floats to `sig` significant digits, for the portable
    digest. Ints, strings and structure are left alone."""
    fmt = "%%.%dg" % sig
    if isinstance(o, dict):
        return {str(k): _round_floats(v, sig) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_round_floats(v, sig) for v in o]
    if isinstance(o, bool) or isinstance(o, int):
        return o
    if isinstance(o, float):
        return fmt % o
    if isinstance(o, np.floating):
        return fmt % float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _round_floats(o.tolist(), sig)
    return o


def _hashable(v):
    if isinstance(v, (list, np.ndarray)):
        return tuple(np.asarray(v).ravel().tolist())
    return v


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _num(x):
    try:
        f = float(x)
    except Exception:
        return None
    return None if not np.isfinite(f) else f
