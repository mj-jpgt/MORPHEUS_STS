"""Turning a typed object into a column of numbers.

`build_feature(world, operator, operands, params, lag)` evaluates a hypothesis
against a world and returns the design column it implies, plus the columns of
its parent family. This is the one piece of machinery the benchmark and the
Hypothesis Compiler must agree on: a claim is only re-testable in a replication
environment if both sides build the same column from the same typed object.

It is deliberately parameterised rather than memorised. `THRESHOLD` needs a
`tau`; if the caller does not supply one, the evaluator does not reach into the
world's ground truth for it -- it raises. A specialist has to search for tau,
which is what makes the oracle gate falsifiable rather than a lookup.
"""
from __future__ import annotations

import numpy as np

from .util import hill, partial_out, soft_threshold, zscore


class FeatureError(ValueError):
    pass


#: operators whose design column is exactly the operand column. Their parent
#: family therefore excludes the operand -- see `parent_columns`.
IDENTITY_OPS = ("ADD", "IDENTITY", "GRAPH", "MIXTURE")

#: operators `build_feature` can evaluate into a numeric design column. The
#: remainder (SYMMETRY, ODE, DESIGN, MANIFOLD, CONFOUND) are structural: they
#: are scored by their own specialists, not by a partial F-test on a column.
EVALUABLE = IDENTITY_OPS + (
    "AND", "XOR", "THRESHOLD", "SATURATE", "CONDITION", "CAUSE", "ANOMALY",
    "SPACE", "LAG", "RATIO", "DIFF", "LINCOMB")

STRUCTURAL = ("SYMMETRY", "ODE", "DESIGN", "MANIFOLD", "CONFOUND")


def _col(world, name):
    if name in world.latents:
        v = np.asarray(world.latents[name], dtype=np.float64)
        return v if v.ndim == 1 else v[:, 0]
    if name in world.nuisance:
        v = np.asarray(world.nuisance[name], dtype=np.float64)
        return v if v.ndim == 1 else v[:, 0]
    if name in world.env.labels:
        v = np.asarray(world.env.labels[name])
        if v.dtype.kind in "USO":
            uniq = sorted(set(v.tolist()))
            return np.array([uniq.index(x) for x in v], dtype=np.float64)
        return v.astype(np.float64)
    raise FeatureError("no variable %r in world %r" % (name, world.name))


def lag_column(world, x, lag):
    """x at (unit, t - lag). Panel-aware: lags within unit, not across rows."""
    x = np.asarray(x, dtype=np.float64)
    st = world.structure
    if lag in (None, 0):
        return x
    if st.unit is not None and st.time is not None:
        out = np.zeros_like(x)
        unit = np.asarray(st.unit)
        time = np.asarray(st.time)
        for u in np.unique(unit):
            m = np.where(unit == u)[0]
            order = m[np.argsort(time[m])]
            v = x[order]
            out[order[lag:]] = v[:-lag]
        return out
    if st.time is not None:
        return np.concatenate([np.zeros(lag), x[:-lag]])
    raise FeatureError("lag requested on a world with no time structure")


def build_feature(world, operator, operands, params=None, lag=None):
    """The design column implied by a typed object.

    Supported: ADD/IDENTITY, AND, XOR, THRESHOLD, SATURATE, CONDITION, SPACE,
    LAG, GRAPH, CAUSE, ANOMALY, MIXTURE, RATIO, DIFF.
    """
    params = dict(params or {})
    ops = list(operands)
    if operator == "LINCOMB":
        if len(ops) < 2:
            raise FeatureError("LINCOMB needs at least two operands")
        w = params.get("weights")
        if w is None:
            raise FeatureError("LINCOMB needs weights; the evaluator will not "
                               "read them from ground truth")
        w = [float(x) for x in w]
        if len(w) != len(ops):
            raise FeatureError("LINCOMB needs one weight per operand")
        f = np.zeros(world.n)
        for wi, o in zip(w, ops):
            f = f + wi * zscore(_col(world, o))
        return f
    if operator in ("ADD", "IDENTITY", "GRAPH", "MIXTURE"):
        if len(ops) != 1:
            raise FeatureError("%s takes exactly one operand" % operator)
        return _col(world, ops[0])
    if operator in ("AND",):
        if len(ops) < 2:
            raise FeatureError("AND needs at least two operands")
        f = np.ones(world.n)
        for o in ops:
            f = f * zscore(_col(world, o))
        return f
    if operator == "XOR":
        if len(ops) != 2:
            raise FeatureError("XOR needs exactly two operands")
        a = _binarize(_col(world, ops[0]))
        b = _binarize(_col(world, ops[1]))
        return (a != b).astype(float) - 0.5
    if operator == "THRESHOLD":
        if "tau" not in params:
            raise FeatureError("THRESHOLD needs a tau; the evaluator will not "
                               "read one from ground truth")
        return soft_threshold(_col(world, ops[0]), params["tau"],
                              params.get("temp", 0.1))
    if operator == "SATURATE":
        if "K" not in params:
            raise FeatureError("SATURATE needs a K")
        x = _col(world, ops[0])
        return hill(np.maximum(x, 0.0) if x.min() >= 0 else np.exp(0.6 * x),
                    K=params["K"], nexp=params.get("n", 4.0))
    if operator in ("CONDITION", "CAUSE", "ANOMALY"):
        gate_var = params.get("gate_var") or params.get("gate")
        if gate_var is None and len(ops) == 1:
            # A single-operand CONDITION/CAUSE/ANOMALY with no declared gate
            # is a plain mechanism effect, not a self-gate. Fixed 2026-09-15
            # (owner directive, extending the boss6/graph_motif audit):
            # defaulting gate_var to the sole operand made every such claim
            # (e.g. latent_confound's CAUSE:U->Y, intervention_causal's
            # CAUSE:X1->Y) a median-split-SIGNED version of itself instead
            # of the direct effect the generator actually planted -- the
            # same failure class this module's own docstring already names
            # ("the gate indicator was a median split, which is constant
            # whenever a binary gate is unbalanced"), just not previously
            # caught for a single-operand CAUSE. Verified against the
            # generator's own stored feature: latent_confound raw corr(Y,U)
            # 0.716 vs the pre-fix reconstruction's 0.576.
            return zscore(_col(world, ops[0]))
        if gate_var is None:
            gate_var = ops[-1]
        gate = _col(world, gate_var)
        gv = params.get("gate_value")
        g = (gate == gv).astype(float) if gv is not None else (
            2.0 * (gate > np.median(gate)) - 1.0)
        inner = [o for o in ops if o != gate_var]
        f = np.ones(world.n)
        for o in inner:
            f = f * zscore(_col(world, o))
        return g * f
    if operator == "SPACE":
        if world.structure.neighbors is None:
            raise FeatureError("SPACE needs neighbourhood structure")
        own, partner = ops[0], (ops[1] if len(ops) > 1 else ops[0])
        w = _col(world, partner)
        if lag:
            w = lag_column(world, w, lag)
        agg = world.structure.aggregate(w, params.get("agg", "mean"))
        return zscore(_col(world, own)) * zscore(agg)
    if operator == "LAG":
        return lag_column(world, _col(world, ops[0]), lag or params.get("lag"))
    if operator == "RATIO":
        a, b = _col(world, ops[0]), _col(world, ops[1])
        return a / np.where(np.abs(b) < 1e-6, np.sign(b) * 1e-6 + 1e-6, b)
    if operator == "DIFF":
        return _col(world, ops[0]) - _col(world, ops[1])
    raise FeatureError("operator %r has no feature evaluator; structural "
                       "operators (SYMMETRY, ODE, DESIGN) are scored by their "
                       "own specialists" % operator)


def parent_columns(world, operator, operands, params=None, lag=None):
    """Columns of the parent family P(h) that the object must beat.

    For a conjunction the parents are its constituents; for a threshold, the
    unthresholded variable; for a spatial interaction, both states, the
    neighbourhood aggregate and the local density; for a lag, the
    contemporaneous term. Declared nuisance covariates are always included --
    the hierarchical-novelty rule conditions on them, and the artifact classes
    that attack the replication gate are only refutable there.
    """
    params = dict(params or {})
    ops = list(operands)
    cols, names = [], []

    def push(nm, v):
        v = np.asarray(v, dtype=np.float64)
        if v.std() > 1e-12 and nm not in names:
            cols.append(v)
            names.append(nm)

    # An object is never its own parent. For the identity-like operators the
    # design column IS the operand column, so pushing the operand would
    # regress the feature on itself and return zero evidence for every true
    # main effect. `LAG` names its contemporaneous parent explicitly below.
    # LINCOMB joins the identity-like operators here for the same reason they
    # are here: its design column lies inside the span of its own operands, so
    # pushing them would regress the feature on itself and return exactly zero
    # evidence for every true LINCOMB. The hierarchical-novelty rule asks an
    # object to beat its parent *family*, and in a world that plants a
    # combination the constituents are not objects.
    if operator not in IDENTITY_OPS and operator not in ("LAG", "LINCOMB"):
        for o in ops:
            try:
                push(o, _col(world, o))
            except FeatureError:
                pass
    if operator == "THRESHOLD":
        pass                                   # the operand itself is the parent
    if operator == "SATURATE":
        push(ops[0] + "^2", _col(world, ops[0]) ** 2)
    if operator == "SPACE" and world.structure.neighbors is not None:
        partner = ops[1] if len(ops) > 1 else ops[0]
        w = _col(world, partner)
        if lag:
            w = lag_column(world, w, lag)
        push("agg(" + partner + ")",
             world.structure.aggregate(w, params.get("agg", "mean")))
        push("density", world.structure.n_neighbors().astype(float))
    if operator == "LAG":
        push(ops[0] + "@t", _col(world, ops[0]))
    if operator in ("AND", "XOR", "CONDITION") and len(ops) > 2:
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                push("%s*%s" % (ops[i], ops[j]),
                     zscore(_col(world, ops[i])) * zscore(_col(world, ops[j])))
    for k, v in world.nuisance.items():
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            push("nuisance:" + k, v)
        else:
            for j in range(v.shape[1]):
                push("nuisance:%s[%d]" % (k, j), v[:, j])
    return cols, names


def _binarize(x):
    x = np.asarray(x, dtype=np.float64)
    u = np.unique(x)
    if len(u) <= 2:
        return x > u.min()
    return x > np.median(x)


def residualise(y, cols):
    return partial_out(np.asarray(y, dtype=np.float64), cols) if cols else \
        np.asarray(y, dtype=np.float64) - np.mean(y)


# ==========================================================================
# composite programs -- the grammar the Hypothesis Compiler consumes
# ==========================================================================
# A composite object is a product of typed *terms*. A term is
#
#     {"op": OPERATOR, "operands": [...], "params": {...}, "lag": int|None}
#
# and an entry of `operands` is either a variable name or a nested term, so
# the grammar is a program tree rather than a flat list: BOSS-2's object is
# XOR(A, majority-aggregate of B) gated by a regime, which no flat product can
# express. `build_composite` evaluates the tree; `composite_parent_columns`
# builds the hierarchical-novelty parent family the tree must beat, which is
# every strict sub-product together with each term's own operand columns.
#
# The single-object evaluator `build_feature` above stays the authority for
# one-operator objects; `term_column` agrees with it on every operator they
# share (asserted in the test suite) and adds the three things a composite
# needs and a single object does not: a gate as a standalone factor, a
# neighbourhood aggregate as a standalone factor, and nested operands.

GATE_OPS = ("CONDITION", "CAUSE", "ANOMALY")


def normalize_term(term):
    """(op, operands, params, lag) from a dict or a tuple."""
    if isinstance(term, dict):
        return (term["op"], tuple(term.get("operands", ())),
                dict(term.get("params", {})), term.get("lag"))
    if isinstance(term, (tuple, list)):
        t = list(term) + [None] * (4 - len(term))
        return (t[0], tuple(t[1] or ()), dict(t[2] or {}), t[3])
    raise FeatureError("cannot read term %r" % (term,))


def gate_indicator(x, params):
    """The 0/1 indicator a gate term contributes.

    `gate_level` / `gate_value` name the level explicitly. With neither, a
    two-level gate uses its upper level -- never a median split, which silently
    returns a constant whenever the gate is unbalanced and turns every gated
    design column into a rescaled copy of its inner term.
    """
    x = np.asarray(x, dtype=np.float64)
    v = params.get("gate_level", params.get("gate_value"))
    if v is not None:
        return (x == float(v)).astype(np.float64)
    lv = np.unique(x)
    if len(lv) > 12:
        raise FeatureError("a gate term needs a low-cardinality gate or an "
                           "explicit gate_level; got %d levels" % len(lv))
    return (x == lv[-1]).astype(np.float64)


def _operand_column(world, o):
    return _col(world, o) if isinstance(o, str) else term_column(world, o)


def term_column(world, term):
    """One factor of a composite program."""
    op, ops, params, lag = normalize_term(term)
    cols = [_operand_column(world, o) for o in ops]
    if op in ("ADD", "IDENTITY", "GRAPH", "MIXTURE"):
        return zscore(cols[0])
    if op == "AND":
        f = np.ones(world.n)
        for c in cols:
            f = f * zscore(c)
        return f
    if op == "XOR":
        return (_binarize(cols[0]) != _binarize(cols[1])).astype(float) - 0.5
    if op == "THRESHOLD":
        if "tau" not in params:
            raise FeatureError("THRESHOLD needs a tau")
        return soft_threshold(cols[0], params["tau"], params.get("temp", 0.1))
    if op == "SATURATE":
        if "K" not in params:
            raise FeatureError("SATURATE needs a K")
        x = cols[0]
        return hill(x if x.min() >= 0 else np.exp(0.6 * x),
                    K=params["K"], nexp=params.get("n", 4.0))
    if op in GATE_OPS:
        return gate_indicator(cols[0], params)
    if op == "SPACE":
        if world.structure.neighbors is None:
            raise FeatureError("SPACE needs neighbourhood structure")
        x = cols[0]
        if lag:
            x = lag_column(world, x, lag)
        return world.structure.aggregate(x, params.get("agg", "mean"))
    if op == "LAG":
        return lag_column(world, cols[0], lag if lag else params.get("lag"))
    if op == "RATIO":
        b = cols[1]
        return cols[0] / np.where(np.abs(b) < 1e-6, np.sign(b) * 1e-6 + 1e-6, b)
    if op == "DIFF":
        return cols[0] - cols[1]
    raise FeatureError("operator %r is structural and has no design column; "
                       "it is scored by its own specialist" % op)


def build_composite(world, terms):
    """Design column of a composite program: the product of its terms."""
    f = np.ones(world.n, dtype=np.float64)
    for t in terms:
        f = f * term_column(world, t)
    return f


def _term_leaf_names(term):
    _, ops, _, _ = normalize_term(term)
    out = []
    for o in ops:
        out.extend([o] if isinstance(o, str) else _term_leaf_names(o))
    return out


def composite_operands(terms):
    """Every distinct variable the program reads, in sorted order."""
    names = []
    for t in terms:
        names.extend(_term_leaf_names(t))
    return tuple(sorted(dict.fromkeys(names)))


def term_label(term):
    op, ops, params, lag = normalize_term(term)
    inner = ",".join(o if isinstance(o, str) else term_label(o) for o in ops)
    tag = op
    if op in GATE_OPS and ("gate_level" in params or "gate_value" in params):
        tag = "%s=%s" % (op, params.get("gate_level", params.get("gate_value")))
    if op == "THRESHOLD" and "tau" in params:
        tag = "THRESHOLD@%.3g" % params["tau"]
    if op == "SATURATE" and "K" in params:
        tag = "SATURATE@%.3g" % params["K"]
    if op == "SPACE":
        tag = "SPACE_%s" % params.get("agg", "mean")
    if lag:
        tag = "%s[-%d]" % (tag, lag)
    return "%s(%s)" % (tag, inner)


def composite_parent_columns(world, terms, include_nuisance=True):
    """The parent family P(h) of a composite: every strictly simpler program.

    Three groups, and all three are load-bearing:

      * every strict sub-product of the terms -- a three-term object must beat
        each of its two-term and one-term restrictions, which is what stops a
        composite from being credited for a relationship its own gate already
        explains;
      * each leaf variable in its raw form -- so a thresholded term must beat
        the untransformed variable;
      * the declared nuisance covariates and, on a structured world, the local
        density, because the adversarial artifact classes are refutable only
        by conditioning on those.
    """
    terms = list(terms)
    cols, names = [], []

    def push(nm, v):
        v = np.asarray(v, dtype=np.float64)
        if np.std(v) > 1e-12 and nm not in names:
            cols.append(v)
            names.append(nm)

    factors = [term_column(world, t) for t in terms]
    labels = [term_label(t) for t in terms]
    k = len(terms)
    for mask in range(1, (1 << k) - 1):          # proper non-empty subsets
        f = np.ones(world.n)
        lab = []
        for i in range(k):
            if mask >> i & 1:
                f = f * factors[i]
                lab.append(labels[i])
        push("*".join(lab), f)
    for nm in composite_operands(terms):
        try:
            push(nm, _col(world, nm))
        except FeatureError:
            pass
    if include_nuisance:
        if world.structure.neighbors is not None:
            push("density", world.structure.n_neighbors().astype(float))
        for k2, v in world.nuisance.items():
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 1:
                push("nuisance:" + k2, v)
            else:
                for j in range(v.shape[1]):
                    push("nuisance:%s[%d]" % (k2, j), v[:, j])
    return cols, names


def program_signature(terms):
    """Operator multiset of a program -- the unit the compositional-OOD split
    holds out. Order-insensitive and parameter-free by design."""
    ops = []

    def walk(t):
        op, o, _, _ = normalize_term(t)
        ops.append(op)
        for x in o:
            if not isinstance(x, str):
                walk(x)
    for t in terms:
        walk(t)
    return tuple(sorted(ops))
