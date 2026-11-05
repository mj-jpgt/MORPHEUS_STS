"""The typed Scientific Object Hypergraph, G_sci = (V, E, H, Gamma).

The output of the encoder is deliberately not a vector in R^d. It is a typed
object graph, because a discovered AND-state should be *literally a hyperedge*
rather than a hidden property of a coordinate that someone later has to go
looking for.

    V      atomic scientific objects: states, programmes, factors, latent
           variables, charts, regimes
    E      pairwise relations, optionally directed and optionally lagged
    H      higher-order hyperedges: AND, XOR, synergy, inhibition, conditional,
           and unclassified higher-order candidates
    Gamma  conditions -- treatment, environment, time, space, intervention --
           attached by id to any node, edge or hyperedge

Interfaces this file is the contract for
----------------------------------------
1. The **discovery registry** (append-only log). `canonical_json()` is the
   agreed serialisation: UTF-8, sorted keys, no insignificant whitespace,
   floats rounded to `FLOAT_NDIGITS`, and a `schema_version` string at the top
   level. `content_hash()` is sha256 over exactly those bytes, so a registry
   row can be checked against a graph without re-serialising it a second way.
   `SCHEMA_VERSION` is bumped on any change that alters those bytes for an
   unchanged graph.

2. The **Hypothesis Compiler** (`morpheus_compiler`) and **DiscoveryBench**
   (`dbench`). `to_hypotheses()` emits dicts in `dbench.world.Hypothesis`
   field layout, using the shared `OPERATORS` vocabulary, so that a proposed
   object and a ground-truth object are comparable by `key()` and never by a
   discovery-type string. The vocabulary is mirrored below rather than
   imported, so this module has no hard dependency on a package that is still
   being written; `check_vocabulary_agreement()` asserts they still match and
   is called by the tests.

Nothing here is a model. It is a data structure with a serialisation and a
validator, and it is deliberately free of torch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "morpheus.sci_hypergraph/1.0.0"
FLOAT_NDIGITS = 12

# --------------------------------------------------------------------------
# vocabularies. OPERATORS mirrors dbench.world.OPERATORS on purpose -- see the
# module docstring. RELATION_TYPES and HYPEREDGE_TYPES partition it by arity.
# --------------------------------------------------------------------------
OPERATORS = (
    "ADD", "AND", "XOR", "THRESHOLD", "SATURATE", "CONDITION", "RATIO",
    "DIFF", "SPACE", "LAG", "GRAPH", "CAUSE", "SYMMETRY", "ODE", "ANOMALY",
    "MANIFOLD", "MIXTURE", "CONFOUND", "DESIGN", "IDENTITY", "LINCOMB",
)

OBJECT_KINDS = (
    "state",      # a latent state a row either occupies or does not
    "program",    # an additive/non-negative programme (factor-expert output)
    "factor",     # a signed linear factor
    "latent",     # a bare latent coordinate with no further commitment
    "chart",      # a manifold chart coordinate
    "regime",     # a discrete regime / branch label
    "anomaly",    # a rare-state indicator
    "motif",      # a graph/subgraph motif
)

RELATION_TYPES = (
    "ADD", "RATIO", "DIFF", "SPACE", "LAG", "GRAPH", "CAUSE", "SYMMETRY",
    "MANIFOLD",
)

HYPEREDGE_TYPES = (
    "AND", "XOR", "SYNERGY", "INHIBIT", "CONDITION", "THRESHOLD", "SATURATE",
    "ODE", "MIXTURE", "ANOMALY", "HIGHER_ORDER", "LINCOMB",
)

CONDITION_KINDS = ("treatment", "environment", "time", "space", "intervention")

# Which OPERATORS string a hyperedge/relation type compiles to. HIGHER_ORDER
# has no single operator: an unclassified higher-order candidate compiles to
# AND, which is the weakest typed claim consistent with "these k objects act
# jointly", and the compiler is free to refine it.
_HYPEREDGE_TO_OPERATOR = {
    "AND": "AND", "XOR": "XOR", "SYNERGY": "AND", "INHIBIT": "AND",
    "CONDITION": "CONDITION", "THRESHOLD": "THRESHOLD",
    "SATURATE": "SATURATE", "ODE": "ODE", "MIXTURE": "MIXTURE",
    "ANOMALY": "ANOMALY", "HIGHER_ORDER": "AND", "LINCOMB": "LINCOMB",
}


class SchemaError(ValueError):
    """Raised by `SciHypergraph.validate` on a structurally invalid graph."""


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------
def _canon(o: Any) -> Any:
    """Recursively make a value canonically serialisable.

    Floats are rounded so that a graph reconstructed from JSON hashes to the
    same bytes as the graph it came from; without this a registry row and a
    freshly re-serialised graph disagree in the 17th digit and the provenance
    check fails for no scientific reason.
    """
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            return str(o)
        return round(o, FLOAT_NDIGITS) + 0.0
    if isinstance(o, bool) or isinstance(o, int) or o is None or isinstance(o, str):
        return o
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if hasattr(o, "tolist"):          # numpy / torch, kept out of the schema
        return _canon(o.tolist())
    if hasattr(o, "item"):
        return _canon(o.item())
    return str(o)


# --------------------------------------------------------------------------
# V, E, H, Gamma
# --------------------------------------------------------------------------
@dataclass
class ObjectNode:
    """One atomic scientific object.

    `slot` and `expert` are provenance, not identity: the same object found in
    a replication environment will generally land in a different slot index,
    which is exactly why step 5 matches objects on semantics rather than index.
    `modality_gates` is the object's learned modality signature g_km -- the
    thing that makes "a transcriptional programme uses only RNA" a readable
    property of the object rather than a global constant.
    """
    id: str
    kind: str
    slot: int | None = None
    expert: str | None = None
    modality_gates: dict = field(default_factory=dict)
    loading: dict = field(default_factory=dict)     # summary stats, not the data
    params: dict = field(default_factory=dict)
    conditions: tuple = ()                          # Condition ids
    score: float = float("nan")
    notes: str = ""

    def to_dict(self) -> dict:
        return _canon({
            "id": self.id, "kind": self.kind, "slot": self.slot,
            "expert": self.expert, "modality_gates": self.modality_gates,
            "loading": self.loading, "params": self.params,
            "conditions": list(self.conditions), "score": self.score,
            "notes": self.notes,
        })

    @staticmethod
    def from_dict(d: dict) -> "ObjectNode":
        return ObjectNode(
            id=d["id"], kind=d["kind"], slot=d.get("slot"),
            expert=d.get("expert"), modality_gates=dict(d.get("modality_gates", {})),
            loading=dict(d.get("loading", {})), params=dict(d.get("params", {})),
            conditions=tuple(d.get("conditions", ())),
            score=_float(d.get("score")), notes=d.get("notes", ""))


@dataclass
class Relation:
    """A pairwise relation between two objects."""
    id: str
    type: str
    source: str
    target: str
    directed: bool = False
    lag: int | None = None
    weight: float = float("nan")
    score: float = float("nan")
    params: dict = field(default_factory=dict)
    conditions: tuple = ()
    notes: str = ""

    def to_dict(self) -> dict:
        return _canon({
            "id": self.id, "type": self.type, "source": self.source,
            "target": self.target, "directed": self.directed, "lag": self.lag,
            "weight": self.weight, "score": self.score, "params": self.params,
            "conditions": list(self.conditions), "notes": self.notes,
        })

    @staticmethod
    def from_dict(d: dict) -> "Relation":
        return Relation(
            id=d["id"], type=d["type"], source=d["source"], target=d["target"],
            directed=bool(d.get("directed", False)), lag=d.get("lag"),
            weight=_float(d.get("weight")), score=_float(d.get("score")),
            params=dict(d.get("params", {})),
            conditions=tuple(d.get("conditions", ())), notes=d.get("notes", ""))


@dataclass
class Hyperedge:
    """A higher-order object over >= 2 members.

    `members` is stored as given and compared as a sorted tuple, because
    "A AND B" and "B AND A" are the same scientific claim. `order` is
    `len(members)` and is stored explicitly so that a reader can filter by
    order without parsing the member list.
    """
    id: str
    type: str
    members: tuple
    score: float = float("nan")
    effect: float = float("nan")
    params: dict = field(default_factory=dict)
    conditions: tuple = ()
    parents: tuple = ()          # P(h): ids of the simpler objects it must beat
    notes: str = ""

    @property
    def order(self) -> int:
        return len(self.members)

    def key(self) -> tuple:
        return (self.type, tuple(sorted(self.members)), tuple(sorted(self.conditions)))

    def to_dict(self) -> dict:
        return _canon({
            "id": self.id, "type": self.type, "members": list(self.members),
            "order": self.order, "score": self.score, "effect": self.effect,
            "params": self.params, "conditions": list(self.conditions),
            "parents": [list(p) if isinstance(p, (list, tuple)) else p
                        for p in self.parents],
            "notes": self.notes,
        })

    @staticmethod
    def from_dict(d: dict) -> "Hyperedge":
        return Hyperedge(
            id=d["id"], type=d["type"], members=tuple(d["members"]),
            score=_float(d.get("score")), effect=_float(d.get("effect")),
            params=dict(d.get("params", {})),
            conditions=tuple(d.get("conditions", ())),
            parents=tuple(tuple(p) if isinstance(p, list) else p
                          for p in d.get("parents", ())),
            notes=d.get("notes", ""))


@dataclass
class Condition:
    """One element of Gamma: the context a claim is made *in*."""
    id: str
    kind: str
    key: str
    value: Any = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _canon({"id": self.id, "kind": self.kind, "key": self.key,
                       "value": self.value, "meta": self.meta})

    @staticmethod
    def from_dict(d: dict) -> "Condition":
        return Condition(id=d["id"], kind=d["kind"], key=d["key"],
                         value=d.get("value"), meta=dict(d.get("meta", {})))


def _float(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, str):
        return float(v)
    return float(v)


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------
@dataclass
class SciHypergraph:
    nodes: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    hyperedges: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    operating_point: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # -- shorthand matching the paper's notation ---------------------------
    @property
    def V(self) -> list:
        return self.nodes

    @property
    def E(self) -> list:
        return self.relations

    @property
    def H(self) -> list:
        return self.hyperedges

    @property
    def Gamma(self) -> list:
        return self.conditions

    def node(self, nid: str) -> ObjectNode:
        for v in self.nodes:
            if v.id == nid:
                return v
        raise KeyError(nid)

    def by_expert(self, expert: str) -> list:
        return [v for v in self.nodes if v.expert == expert]

    def by_order(self, order: int) -> list:
        return [h for h in self.hyperedges if h.order == order]

    # -- validation ---------------------------------------------------------
    def validate(self) -> "SciHypergraph":
        ids = [v.id for v in self.nodes]
        if len(set(ids)) != len(ids):
            raise SchemaError("duplicate node ids: %s" % _dups(ids))
        node_ids = set(ids)
        cond_ids = {c.id for c in self.conditions}
        if len(cond_ids) != len(self.conditions):
            raise SchemaError("duplicate condition ids")
        for v in self.nodes:
            if v.kind not in OBJECT_KINDS:
                raise SchemaError("node %s has unknown kind %r" % (v.id, v.kind))
            _check_conds(v.id, v.conditions, cond_ids)
        rel_ids = [e.id for e in self.relations]
        if len(set(rel_ids)) != len(rel_ids):
            raise SchemaError("duplicate relation ids: %s" % _dups(rel_ids))
        for e in self.relations:
            if e.type not in RELATION_TYPES:
                raise SchemaError("relation %s has unknown type %r" % (e.id, e.type))
            for endpoint in (e.source, e.target):
                if endpoint not in node_ids:
                    raise SchemaError(
                        "relation %s references unknown node %r" % (e.id, endpoint))
            if e.lag is not None and not e.directed:
                raise SchemaError(
                    "relation %s carries a lag but is undirected; a lagged "
                    "relation without a direction is not a scientific claim"
                    % e.id)
            _check_conds(e.id, e.conditions, cond_ids)
        h_ids = [h.id for h in self.hyperedges]
        if len(set(h_ids)) != len(h_ids):
            raise SchemaError("duplicate hyperedge ids: %s" % _dups(h_ids))
        for h in self.hyperedges:
            if h.type not in HYPEREDGE_TYPES:
                raise SchemaError("hyperedge %s has unknown type %r" % (h.id, h.type))
            if h.order < 2:
                raise SchemaError(
                    "hyperedge %s has order %d; an order-1 object is a node"
                    % (h.id, h.order))
            for m in h.members:
                if m not in node_ids:
                    raise SchemaError(
                        "hyperedge %s references unknown node %r" % (h.id, m))
            if len(set(h.members)) != len(h.members):
                raise SchemaError("hyperedge %s repeats a member" % h.id)
            _check_conds(h.id, h.conditions, cond_ids)
        for c in self.conditions:
            if c.kind not in CONDITION_KINDS:
                raise SchemaError("condition %s has unknown kind %r" % (c.id, c.kind))
        return self

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "nodes": [v.to_dict() for v in self.nodes],
            "relations": [e.to_dict() for e in self.relations],
            "hyperedges": [h.to_dict() for h in self.hyperedges],
            "conditions": [c.to_dict() for c in self.conditions],
            "operating_point": _canon(self.operating_point),
            "meta": _canon(self.meta),
        }

    @staticmethod
    def from_dict(d: dict) -> "SciHypergraph":
        sv = d.get("schema_version", "")
        if sv.split("/")[0] != SCHEMA_VERSION.split("/")[0]:
            raise SchemaError("not a scientific hypergraph document: %r" % sv)
        if _major(sv) != _major(SCHEMA_VERSION):
            raise SchemaError(
                "schema major version mismatch: document %r, reader %r"
                % (sv, SCHEMA_VERSION))
        return SciHypergraph(
            nodes=[ObjectNode.from_dict(x) for x in d.get("nodes", [])],
            relations=[Relation.from_dict(x) for x in d.get("relations", [])],
            hyperedges=[Hyperedge.from_dict(x) for x in d.get("hyperedges", [])],
            conditions=[Condition.from_dict(x) for x in d.get("conditions", [])],
            operating_point=dict(d.get("operating_point", {})),
            meta=dict(d.get("meta", {})),
            schema_version=sv)

    def canonical_json(self) -> str:
        """The agreed registry bytes: sorted keys, no padding, rounded floats."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=True)

    def to_json(self, indent: int | None = 2) -> str:
        if indent is None:
            return self.canonical_json()
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          indent=indent, allow_nan=True)

    @staticmethod
    def from_json(s: str) -> "SciHypergraph":
        return SciHypergraph.from_dict(json.loads(s))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # -- handoff to the compiler / registry ---------------------------------
    def to_hypotheses(self, target: str = "Y") -> list:
        """Emit `dbench.world.Hypothesis` field dicts.

        Nodes become IDENTITY objects, relations their own operator, and
        hyperedges the operator named by `_HYPEREDGE_TO_OPERATOR`. Every emitted
        record names its operands, because scoring is against
        (operator, operands, target, context) and a record that says only
        "this is a relational discovery" has recovered nothing.
        """
        out = []
        cmap = {c.id: c for c in self.conditions}

        def ctx(ids: Iterable[str]) -> dict:
            return {cmap[i].key: cmap[i].value for i in ids if i in cmap}

        for v in self.nodes:
            out.append({
                "id": v.id, "mode": "entity", "operator": "IDENTITY",
                "operands": [v.id], "target": target, "params": dict(v.params),
                "context": ctx(v.conditions), "lag": None,
                "effect": v.score, "true": True, "artifact_class": None,
                "parents": [], "notes": v.notes,
            })
        for e in self.relations:
            out.append({
                "id": e.id, "mode": "relational", "operator": e.type,
                "operands": [e.source, e.target], "target": target,
                "params": dict(e.params), "context": ctx(e.conditions),
                "lag": e.lag, "effect": e.weight, "true": True,
                "artifact_class": None, "parents": [[e.source], [e.target]],
                "notes": e.notes,
            })
        for h in self.hyperedges:
            out.append({
                "id": h.id, "mode": "relational",
                "operator": _HYPEREDGE_TO_OPERATOR[h.type],
                "operands": list(h.members), "target": target,
                "params": dict(h.params, hyperedge_type=h.type),
                "context": ctx(h.conditions), "lag": None, "effect": h.effect,
                "true": True, "artifact_class": None,
                "parents": [list(p) for p in h.parents] or
                           [[m] for m in h.members],
                "notes": h.notes,
            })
        return out

    def to_registry_record(self, **provenance) -> dict:
        """One append-only discovery-registry row.

        The registry owns the provenance fields (checkpoint hash, discovery
        dataset and split, pre-validation rank, named external dataset, result).
        This side owns the object itself and its content hash. Passing the
        provenance through here rather than merging it into `meta` keeps the
        content hash a hash of the *object*, so the same object proposed twice
        under different provenance still hashes identically.
        """
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash(),
            "graph": self.to_dict(),
            "provenance": _canon(provenance),
        }

    def to_registry_candidates(self, target: str = "Y", **common) -> list:
        """`add_candidate(**kw)` keyword dicts, one per emitted hypothesis.

        `morpheus_registry` stores each candidate's `h` as a loose JSON dict
        tagged with an `h_schema_version`, deliberately, because the Hypothesis
        Compiler's typed relation grammar is being defined in parallel. That
        tag is this package's `SCHEMA_VERSION`, so a stored candidate names the
        exact object grammar it was written under and a later reader can tell
        a 1.x object from a 2.x one instead of guessing.

        Registry-owned provenance (freeze_id, discovery_dataset, split,
        statistical_family, ...) is passed through `common` rather than
        invented here -- this side does not know the freeze it is being run
        under, and a record that guesses its own provenance is worse than one
        that refuses to write.
        """
        hs = self.to_hypotheses(target=target)
        out = []
        for i, h in enumerate(sorted(hs, key=lambda d: str(d.get("id")))):
            kw = dict(common)
            kw["h"] = h
            kw["h_schema_version"] = self.schema_version
            kw.setdefault("candidate_id", "%s:%s" % (self.content_hash()[:16],
                                                     h.get("id")))
            kw.setdefault("rank_before_validation", i + 1)
            kw.setdefault("parent_hypothesis_family", h.get("operator"))
            out.append(kw)
        return out

    def summary(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "n_nodes": len(self.nodes), "n_relations": len(self.relations),
            "n_hyperedges": len(self.hyperedges),
            "n_conditions": len(self.conditions),
            "node_kinds": _counts(v.kind for v in self.nodes),
            "relation_types": _counts(e.type for e in self.relations),
            "hyperedge_types": _counts(h.type for h in self.hyperedges),
            "hyperedge_orders": _counts(h.order for h in self.hyperedges),
            "content_hash": self.content_hash(),
        }


def _major(v: str) -> str:
    return v.split("/")[-1].split(".")[0]


def _dups(xs: Sequence) -> list:
    seen, out = set(), []
    for x in xs:
        if x in seen and x not in out:
            out.append(x)
        seen.add(x)
    return out


def _counts(xs: Iterable) -> dict:
    d: dict = {}
    for x in xs:
        d[str(x)] = d.get(str(x), 0) + 1
    return dict(sorted(d.items()))


def _check_conds(owner: str, conds, cond_ids: set):
    for c in conds:
        if c not in cond_ids:
            raise SchemaError("%s references unknown condition %r" % (owner, c))


def check_vocabulary_agreement() -> dict:
    """Assert this module's OPERATORS still equals `dbench.world.OPERATORS`.

    Called by the tests. If DiscoveryBench adds an operator, this fails loudly
    here rather than silently emitting a hypothesis the scorer cannot match.
    """
    try:
        from dbench.world import OPERATORS as BENCH_OPERATORS
    except Exception as exc:                      # pragma: no cover
        return {"checked": False, "reason": repr(exc)}
    mine, theirs = set(OPERATORS), set(BENCH_OPERATORS)
    return {
        "checked": True,
        "agree": mine == theirs,
        "only_here": sorted(mine - theirs),
        "only_in_dbench": sorted(theirs - mine),
    }
