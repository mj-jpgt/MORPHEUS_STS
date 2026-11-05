"""Scoring: hidden-object recovery, RDC, and the secondary metrics.

The scoring rule the spec is emphatic about: **score hidden-object recovery,
not discovery-type classification.** A system that reports "this dataset
contains a relational discovery" has recovered nothing. A claim is credited
only when it names the object -- the operands, the operator, the target, the
context and the lag -- and it is matched against ground truth on those fields.
`Claim.discovery_mode` is carried for reporting and contributes zero to the
match score; `mode_confusion` reports how often the mode label was right, as a
diagnostic that is deliberately not part of any headline number.

Matching is a global assignment, not a greedy scan. A greedy matcher will
happily credit one true object twice under two slightly different claims and
inflate recall; the same shortcut has already retracted a claim on this
project once. `match_claims` solves a rectangular assignment problem
(scipy.optimize.linear_sum_assignment), so each true object is matched at most
once and each claim is matched at most once, and the assignment maximises total
similarity rather than whichever pair happened to be examined first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dataclasses import replace as _dc_replace

from scipy.optimize import linear_sum_assignment

from .align import MIN_ABS_R, align_claim, structural_cap
from .opvocab import to_canonical
from .util import benjamini_hochberg

MATCH_THRESHOLD = 0.70
OPERAND_FLOOR = 0.50

#: Evaluation-equivalence threshold for "these two claims express the same
#: object". Mirrors `morpheus.rotation.engine.SEMANTIC_CORR_MIN`, which is the
#: canonical definition; it is restated here only because `dbench` is the
#: lower-level package and `rotation` imports it, not the other way round, so a
#: runtime import would invert the dependency. The mirroring is *enforced* by
#: `tests/test_duplicate_expressions.py`, in the same way
#: `morpheus_encoder.hypergraph` enforces its copy of `OPERATORS`. If the
#: rotation harness moves this number, that test fails.
SEMANTIC_CORR_MIN = 0.99

#: A kin match is a near-miss, not a recovery. Perfect operands alone score
#: 0.800 even with zero operator credit, above MATCH_THRESHOLD, so kin cannot
#: be held below the threshold by reweighting -- the cap is explicit. It sits
#: strictly below MATCH_THRESHOLD so a kin pair can never be assigned as a
#: recovered object, while the similarity stays visible and the pair is counted
#: under `kin_matches`.
#:
#: The kin table is heterogeneous, which is what forces this. Only
#: {ADD, IDENTITY} share a design column; {AND, CONDITION, XOR} are different
#: combiners, {THRESHOLD, SATURATE} different functions, {ANOMALY, MIXTURE}
#: different entity types, and {CAUSE, CONFOUND} are opposite claims -- a
#: CONFOUND object is false by construction, so crediting a CAUSE claim against
#: one as recovery would credit exactly the confusion this benchmark exists to
#: detect.
KIN_SIMILARITY_CAP = 0.65


def abscorr(a, b) -> float:
    """|Pearson r|, matching `morpheus.rotation.engine._abscorr` exactly."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 8:
        return float("nan")
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    da, db = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(abs(a @ b) / (da * db))


#: How far below a declaration's own measured correlation an observed one may
#: fall and still count as that expression.
DECLARED_ABSCORR_SLACK = 0.02


def declares_expression(hyp, claim):
    """The declaration matching `claim`'s form, or None.

    The generator knows its own symmetries and the matcher should not have to
    infer them: `spatial_interaction` declares that A * agg(B) and B * agg(A)
    are the same object, which matters because their measured |corr| is 0.959,
    below `SEMANTIC_CORR_MIN`, so the evaluation route alone would not fold
    them.

    It returns the declaration rather than a bare True because a declaration is
    about a *form*, and a claim asserting that form still has to be it. dbench
    claims carry only symbolic fields, so two claims with identical operator
    and operands can evaluate to entirely different columns -- which is not
    hypothetical: a decoy tuned to |corr| 0.85 with the true column, carrying
    the same operator and operand set, was folded when this returned a boolean
    and the caller skipped evaluation on the strength of it. The declaration
    lowers the bar to the correlation the generator measured; it does not
    remove it.
    """
    ops = set(claim.operands)
    for e in getattr(hyp, "equivalent_expressions", ()) or ():
        d = dict(e) if isinstance(e, tuple) else dict(e or {})
        want = d.get("operands")
        if want is None or set(want) != ops:
            continue
        op = d.get("operator")
        if op is not None and to_canonical(op) != to_canonical(claim.operator):
            continue
        return d
    return None


def declared_abscorr_floor(decl):
    """The correlation a declared expression must still reach."""
    m = decl.get("measured_abscorr")
    if isinstance(m, tuple):
        m = dict(m)
    if isinstance(m, dict) and m:
        return max(0.0, min(float(v) for v in m.values())
                   - DECLARED_ABSCORR_SLACK)
    if isinstance(m, (int, float)):
        return max(0.0, float(m) - DECLARED_ABSCORR_SLACK)
    return SEMANTIC_CORR_MIN


# --------------------------------------------------------------------------
@dataclass
class Claim:
    """What a discovery system reports about one candidate object."""
    id: str
    operator: str
    operands: tuple
    target: str = "Y"
    context: dict = field(default_factory=dict)
    lag: int | None = None
    effect: float = float("nan")          # proposal-environment effect size
    p_value: float = float("nan")         # proposal-environment p
    replication: dict = field(default_factory=dict)   # env name -> (effect, p)
    complexity: int = 1                   # program length / MDL proxy
    discovery_mode: str | None = None     # reported for diagnosis only
    params: dict = field(default_factory=dict)

    def key(self):
        return (self.operator, tuple(sorted(self.operands)), self.target,
                self.lag,
                tuple(sorted((str(k), str(v)) for k, v in self.context.items())))


# --------------------------------------------------------------------------
#: Parameters that change *what the object is*. A claim that disagrees with the
#: truth on any of these is a claim about a different object, however well its
#: operands line up, so a mismatch sets similarity to zero rather than docking
#: a few points. `lag` and `target` are always compared; the rest gate only
#: when both sides state them, because an unreported parameter is not a wrong
#: one -- except `lag`, where "no lag" is itself an assertion.
IDENTITY_PARAMS = ("operator", "lag", "target", "agg", "gate_level")

#: `composition` was considered for this list and rejected. A composite truth
#: object declares the operator multiset of the *world*, not of itself:
#: BOSS-2's relational object lists ANOMALY because the world also contains a
#: separate regime object, and BOSS-1 lists LAG although the lag is carried on
#: its SPACE term rather than as a term of its own. Gating on it would have
#: turned three boss-world recoveries off for a bookkeeping mismatch rather
#: than a scientific one, which is the over-correction this fix is supposed to
#: avoid. The operator field, weighted with its kinship rule, is what carries
#: that information.

#: Parameters that change how *strongly* or *where* an object holds. These stay
#: on the weighted side: a specialist that finds the right conjunction at
#: tau = 0.5 when the truth is 0.6 has found the object.
MAGNITUDE_PARAMS = ("tau", "temp", "K", "n", "beta", "nexp", "radius")


def _weight_key(weights, places=6):
    """A LINCOMB weight vector reduced to the direction it names.

    Unit L1 with the leading non-zero made positive, then rounded. Positive
    rescalings and an overall sign flip collapse to one key; a genuinely
    different mixture does not.
    """
    w = [float(x) for x in weights]
    s = sum(abs(x) for x in w)
    if s <= 0.0:
        return tuple(0.0 for _ in w)
    w = [x / s for x in w]
    lead = next((x for x in w if abs(x) > 1e-12), 0.0)
    if lead < 0:
        w = [-x for x in w]
    return tuple(round(x, places) + 0.0 for x in w)


def structural_key(obj):
    """The identity-defining part of a claim or a ground-truth object."""
    params = dict(getattr(obj, "params", None) or {})
    context = dict(getattr(obj, "context", None) or {})
    # `operator` is deliberately NOT here. It is gated at the top of
    # `structural_conflict` by a kin-aware check, and putting it in this dict
    # too made it compared a second time in the both-declared loop, where the
    # kin table has no say -- which silently gated THRESHOLD against its own
    # declared kin SATURATE.
    out = {"lag": getattr(obj, "lag", None),
           "target": getattr(obj, "target", None)}
    if "agg" in params:
        out["agg"] = params["agg"]
    # The weight vector of a LINCOMB is operator-defining in exactly the sense
    # `lag` is: the compiler prices weights on a finite grid, and a claim at a
    # different point of that grid is a different object, not a noisier
    # reading of the same one. Compared only when both sides declare it, so no
    # specialist is punished for making no weight claim, and no other family
    # declares weights.
    #
    # Normalised to unit L1 with the leading sign made positive, because a
    # LINCOMB is a *direction*: `(0.375, 0.125)` and `(0.75, 0.25)` are the
    # same object and the generator declares them equivalent, so the identity
    # key must not gate one against the other. `(0.5, 0.5)` still differs and
    # is still gated, which is the discrimination the grid buys.
    if params.get("weights") is not None:
        out["weights"] = _weight_key(params["weights"])
    for src in (context, params):
        for k in ("gate_value", "gate_level"):
            if src.get(k) is not None and "gate_level" not in out:
                out["gate_level"] = float(src[k])
    return out


def structural_conflict(claim, hyp):
    """The first identity-defining parameter the two disagree on, or None.

    `lag` and `target` are compared unconditionally. The others are compared
    only when both sides declare them, so a specialist is never punished for
    declining to report a parameter it made no claim about.
    """
    # The operator gates. Naming the right single operand with the wrong
    # relation used to score exactly 0.8000 -- 0.55 operand + 0.10 target +
    # 0.075 lag + 0.075 context -- and clear the 0.70 threshold, so ADD(A),
    # AND(A) and XOR(A) were each credited with recovering a planted
    # THRESHOLD(A). Measured on the external bank, six interaction detectors
    # read recall 1.000 on nine of eleven relational families they cannot
    # express at all; the single 0.000 was temporal_lag, because `lag` already
    # gated. A THRESHOLD(A) object is not an ADD(A) object.
    #
    # Kin remains partial credit, because it is declared: the pairs in `_KIN`
    # are relations that share a design column or a role, and a method that
    # picks the wrong member of such a pair has the right variable and the
    # wrong claim. Everything outside the declared kin table is a different
    # object and is gated.
    ca, cb = to_canonical(claim.operator), to_canonical(hyp.operator)
    if ca != cb and not _operator_kin(ca, cb):
        return "operator"
    a, b = structural_key(claim), structural_key(hyp)
    for k in ("lag", "target"):
        if a.get(k) != b.get(k):
            return k
    for k in sorted((set(a) & set(b)) - {"lag", "target"}):
        if a[k] != b[k]:
            return k
    return None


def object_similarity(claim, hyp, operand_weight=0.55, operator_weight=0.20,
                      target_weight=0.10, lag_weight=0.075,
                      context_weight=0.075):
    """Graded similarity in [0, 1] between a claim and a ground-truth object.

    Operand identity carries most of the weight because it is the object. The
    operator and context refine it. The discovery-mode label is absent on
    purpose.

    Identity-defining parameters gate rather than weight. They used to be
    weighted, and `lag` carried 7.5% of the total, which was not enough to
    matter: on the `LAG+THRESHOLD` and `LAG+XOR` compositions a claim that
    modelled raw unlagged `B` scored 0.9250 against a truth object with
    `lag = 2` and was credited with recovering it -- while `corr(B, B_lag2)`
    was about -0.01 and the proxy's correlation with the response was -0.018,
    against the true object's 0.943. A specialist was collecting recovery
    credit for a feature uncorrelated with both the partner it named and the
    response. Three of the twenty-six bank-inexpressible worlds read as
    recovered for that reason.

    A lag is not a gradation of similarity. Neither is which neighbourhood
    aggregation was used, nor which level of a gate the object lives on, nor
    which operators a composite is made of. Those set similarity to zero on
    mismatch. A threshold's `tau`, a Hill's `K` and `n`, an effect's `beta`
    stay weighted, because they change how strongly an object holds and not
    which object it is -- so a correct conjunction found at a slightly wrong
    tau still matches, which `MAGNITUDE_PARAMS` and its test pin down.

    `lag` and `target` keep their weights only so the arithmetic is unchanged
    for everything that passes the gate; both terms are now tautologically 1.
    """
    conflict = structural_conflict(claim, hyp)
    a = set(claim.operands)
    b = set(hyp.operands)
    jac = len(a & b) / max(len(a | b), 1)
    if conflict is not None:
        return 0.0, jac
    # Operator names are normalised through the shared table before they are
    # compared. A claim built from a compiler program spells the same concepts
    # MUL, THR, COND where dbench spells them AND, THRESHOLD, CONDITION, and
    # comparing the raw strings forfeited the entire 0.20 operator weight for
    # every such claim -- similarity 0.7999 where 1.0 was available, which put
    # anything whose true similarity was under 0.90 below the 0.70 match
    # threshold. Normalising can only turn an unequal comparison into an equal
    # one, so it can only raise a similarity.
    c_op = to_canonical(claim.operator)
    h_op = to_canonical(hyp.operator)
    is_kin = (c_op != h_op) and _operator_kin(c_op, h_op)
    # A generator-declared equivalent expression (`declares_expression`,
    # already used by `_same_object` to fold decoys) names a SPECIFIC
    # (operator, operand-set) form the generator itself has measured as the
    # same claim -- not merely a related one. `context_gated`'s COND:A|G is
    # the case this closes: the compiler's own `COND` is defined as
    # `1[G]*h`, numerically IDENTICAL to `MUL(A,G)` whenever the gate is
    # already binary (not just correlated with it), so a MUL(A,G) claim here
    # is the SAME object under the operator this grammar's own MDL dominance
    # actually reports, not a decoy holding the right variable and the wrong
    # claim -- which is what the kin cap exists to price. A declaration is
    # required (not inferred from `is_kin` generally) so this stays scoped to
    # forms the generator has explicitly named and measured, exactly as
    # `_same_object`'s existing `declares_expression` route already is.
    declared = (c_op != h_op) and (declares_expression(hyp, claim) is not None)
    if declared:
        is_kin = False
        op = 1.0
    else:
        op = 1.0 if c_op == h_op else (0.5 if is_kin else 0.0)
    tg = 1.0 if claim.target == hyp.target else 0.0
    lg = 1.0 if claim.lag == hyp.lag else 0.0
    ck, hk = claim.context, hyp.context
    if not ck and not hk:
        ctx = 1.0
    else:
        keys = set(map(str, ck)) | set(map(str, hk))
        ctx = sum(1.0 for k in keys
                  if str(ck.get(k)) == str(hk.get(k))) / max(len(keys), 1)
    sim = (operand_weight * jac + operator_weight * op + target_weight * tg
           + lag_weight * lg + context_weight * ctx)
    if is_kin:
        sim = min(sim, KIN_SIMILARITY_CAP)
    return sim, jac


_KIN = [
    {"AND", "CONDITION", "XOR"},          # all higher-order combiners
    {"THRESHOLD", "SATURATE"},            # both saturating own-channel maps
    # LINCOMB is kin to ADD, never equal to it. A planted LINCOMB asserts one
    # weighted combination; an ADD claim over one of its operands has the
    # right variable and the wrong granularity, which is what the kin cap
    # prices. Equality here would let a bank of single-operand detectors read
    # full recall on `linear_combination`, an object none of them can express.
    {"ADD", "IDENTITY", "LINCOMB"},
    {"CAUSE", "CONFOUND"},
    {"ANOMALY", "MIXTURE"},
]


def _operator_kin(a, b):
    return any(a in g and b in g for g in _KIN)


# --------------------------------------------------------------------------
# Complete-object fidelity / capability-frontier metrics
# --------------------------------------------------------------------------
def _pair_f1(left, right):
    """Set F1, with an empty/empty field counted as fully agreed."""
    left, right = set(left), set(right)
    if not left and not right:
        return 1.0
    return 2.0 * len(left & right) / max(len(left) + len(right), 1)


def _context_pairs(obj):
    return {(str(k), str(v)) for k, v in dict(getattr(obj, "context", {}) or {}).items()}


def _parameter_fidelity(claim, hyp):
    """Agreement on declared magnitude parameters without inventing precision.

    Identity-defining disagreements are already scientific-object disagreements
    under ``structural_conflict`` and receive zero. For continuous magnitudes,
    absence of a parameter is not punished: a method that names the correct
    relation but does not estimate its Hill exponent has made an incomplete
    measurement, not a different claim. When both sides declare a finite
    magnitude, a one-scale relative error produces a bounded score.
    """
    if structural_conflict(claim, hyp) is not None:
        return 0.0
    cp = dict(getattr(claim, "params", {}) or {})
    hp = dict(getattr(hyp, "params", {}) or {})
    scores = []
    # A raw threshold value is not comparable across an ICA coordinate, a
    # learned slot, and a measured feature.  When both sides provide the
    # proposal-distribution percentile of the *non-context* threshold instead,
    # compare that scale-invariant semantic quantity on the predeclared grid
    # resolution.  This is intentionally all-or-nothing: a threshold in the
    # neighbouring grid cell is a near miss, not a complete object recovery.
    semantic_thresholds = (cp.get("threshold_quantiles"), hp.get("threshold_quantiles"))
    if all(v is not None for v in semantic_thresholds):
        try:
            left = tuple(sorted(float(v) for v in semantic_thresholds[0]))
            right = tuple(sorted(float(v) for v in semantic_thresholds[1]))
            if len(left) != len(right) or any(not np.isfinite(v) for v in left + right):
                return 0.0
            if any(abs(a - b) > 0.15 for a, b in zip(left, right)):
                return 0.0
        except (TypeError, ValueError):
            return 0.0
    for key in MAGNITUDE_PARAMS:
        if key == "tau" and all(v is not None for v in semantic_thresholds):
            continue
        if key not in cp or key not in hp:
            continue
        try:
            a, b = float(cp[key]), float(hp[key])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        scores.append(max(0.0, 1.0 - abs(a - b) / max(abs(b), 1.0)))
    return float(np.mean(scores)) if scores else 1.0


def object_fidelity(claim, hyp):
    """Measure whether a claim states the *complete* scientific object.

    Unlike ordinary graded similarity, this is intentionally conjunctive.
    A strict ancestor such as ``A`` for truth ``A AND B | C`` can have useful
    overlap, but it cannot receive near-full Object Fidelity: missing operands
    reduce the score and a wrong relation, context, lag, or identity parameter
    makes the relevant component zero.
    """
    operand = _pair_f1(getattr(claim, "operands", ()), getattr(hyp, "operands", ()))
    op = float(to_canonical(claim.operator) == to_canonical(hyp.operator))
    context = _pair_f1(_context_pairs(claim), _context_pairs(hyp))
    lag = float(getattr(claim, "lag", None) == getattr(hyp, "lag", None))
    parameter = _parameter_fidelity(claim, hyp)
    target = float(getattr(claim, "target", "Y") == getattr(hyp, "target", "Y"))
    score = operand * op * context * lag * parameter * target
    return {"object_fidelity": float(score), "operand_f1": float(operand),
            "operator_exact": bool(op), "context_f1": float(context),
            "lag_exact": bool(lag), "parameter_fidelity": float(parameter),
            "target_exact": bool(target)}


def object_error_class(claim, hyp):
    """A mutually exclusive, interpretable nearest-truth error taxonomy."""
    f = object_fidelity(claim, hyp)
    cset, hset = set(claim.operands), set(hyp.operands)
    if f["object_fidelity"] == 1.0:
        return "full_object"
    if cset and cset < hset:
        return "strict_ancestor"
    if cset == hset and not f["operator_exact"]:
        return "correct_operands_wrong_operator"
    if cset == hset and f["operator_exact"] and (not f["context_f1"] or not f["lag_exact"]):
        return "correct_relation_wrong_context"
    if cset & hset:
        return "proxy_or_wrong_constituent"
    return "unrelated"


def object_fidelity_report(claims, truth):
    """Per-truth best fidelity and taxonomy, without hiding partial claims.

    This is separate from RDC's globally one-to-one adjudication. Its purpose
    is diagnostic: it asks what the closest claim said about every true object,
    including cases where a partial claim correctly fails ordinary matching.
    """
    claims = list(claims)
    rows = []
    for hyp in (h for h in truth.objects if h.true):
        scored = [(object_fidelity(c, hyp), c) for c in claims]
        if not scored:
            rows.append({"object_id": hyp.id, "best_claim_id": None,
                         "error_class": "missed", "object_fidelity": 0.0})
            continue
        fidelity, claim = max(scored, key=lambda x: (x[0]["object_fidelity"], x[0]["operand_f1"]))
        rows.append({"object_id": hyp.id, "best_claim_id": claim.id,
                     "error_class": object_error_class(claim, hyp), **fidelity})
    classes = {}
    for row in rows:
        classes[row["error_class"]] = classes.get(row["error_class"], 0) + 1
    return {"per_object": rows, "n_true": len(rows), "counts": classes,
            "mean_object_fidelity": float(np.mean([r["object_fidelity"] for r in rows])) if rows else float("nan"),
            "complete_recovery_rate": (classes.get("full_object", 0) / len(rows) if rows else float("nan"))}


def hard_solve_rate(object_reports):
    """Complete-object recovery on a prespecified capability-frontier set.

    Each item must explicitly provide ``hard=True`` and a completed
    ``object_fidelity_report``; silently treating ordinary worlds as hard is
    refused. A hard object counts only when its error class is ``full_object``.
    """
    hard = [x for x in object_reports if x.get("hard") is True]
    if not hard:
        raise ValueError("HardSolveRate requires explicitly labelled hard worlds")
    rows = [r for x in hard for r in x["object_fidelity_report"]["per_object"]]
    return {"n_hard_worlds": len(hard), "n_hard_objects": len(rows),
            "n_complete": sum(r["error_class"] == "full_object" for r in rows),
            "HardSolveRate": (sum(r["error_class"] == "full_object" for r in rows) / len(rows) if rows else float("nan"))}


def b80(recovery_by_budget, target=0.80):
    """Search-budget analogue of :func:`n80`, with no extrapolation."""
    out = n80(recovery_by_budget, target=target)
    return {"B80": out.get("n80"), "target": target, "curve": out["curve"],
            "reached": out["reached"],
            **({k: v for k, v in out.items() if k in ("max_observed_rate", "max_n_tried")})}


def align_claims(claims, slots, latents, rule="single", min_abs_r=MIN_ABS_R):
    """Rewrite anonymous-slot operands into latent names, where they align.

    A claim whose operands are already latent names is returned untouched, so
    an oracle-latent claim matches identically before and after. A claim whose
    slots cannot be aligned keeps its slot names and will simply fail to match
    -- it is not credited on the strength of having been offered.
    """
    out, report = [], []
    for c in claims:
        a = align_claim(c, slots, latents, rule=rule, min_abs_r=min_abs_r)
        if a is None:
            out.append(c)
            report.append({"claim_id": c.id, "aligned": False,
                           "operands": list(c.operands)})
            continue
        # Context variables are leaves too. Rewriting only operands left a
        # compiled ``COND(..., THR(slot7))`` with operands aligned to A/B/C
        # but context ``gate_var=slot7``; a correct context-conditioned
        # formula then failed exact-object scoring solely because the adapter
        # stopped one field short of its own alignment map.
        context = dict(c.context)
        detail = a.get("detail") or {}
        assignment = detail if isinstance(detail, dict) else {}
        gate = context.get("gate_var")
        if gate in assignment:
            context["gate_var"] = assignment[gate]
        out.append(_dc_replace(c, operands=tuple(a["operands"]), context=context))
        report.append({"claim_id": c.id, "aligned": True, "route": a["route"],
                       "from": list(c.operands), "to": list(a["operands"]),
                       "detail": a.get("detail")})
    return out, report


def match_claims(claims, truth, threshold=MATCH_THRESHOLD,
                 operand_floor=OPERAND_FLOOR):
    """Globally optimal one-to-one matching of claims to ground-truth objects.

    Returns a dict with the matched pairs, the unmatched claims (false
    discoveries, attributed to a planted artifact class where one applies) and
    the unmatched true objects (misses).
    """
    claims = list(claims)
    objs = list(truth.objects)
    if not claims or not objs:
        return {"matched": [], "unmatched_claims": list(range(len(claims))),
                "missed_true": [i for i, h in enumerate(objs) if h.true],
                "n_true": sum(1 for h in objs if h.true)}
    S = np.zeros((len(claims), len(objs)))
    J = np.zeros_like(S)
    blocked = {}
    kin_near = {}
    for i, c in enumerate(claims):
        for j, h in enumerate(objs):
            S[i, j], J[i, j] = object_similarity(c, h)
            if (J[i, j] >= operand_floor
                    and structural_conflict(c, h) is None
                    and to_canonical(c.operator) != to_canonical(h.operator)
                    and _operator_kin(to_canonical(c.operator),
                                      to_canonical(h.operator))):
                kin_near.setdefault(claims[i].id, []).append(
                    {"object_id": objs[j].id, "similarity": float(S[i, j]),
                     "claim_operator": c.operator,
                     "truth_operator": h.operator,
                     "operand_jaccard": float(J[i, j])})
            if S[i, j] == 0.0 and J[i, j] >= operand_floor:
                # operands lined up but an identity-defining parameter did not:
                # worth reporting, because this is precisely the case that used
                # to be scored as a match
                blocked.setdefault(claims[i].id, []).append(
                    {"object_id": objs[j].id,
                     "on": structural_conflict(c, h),
                     "operand_jaccard": float(J[i, j])})
    ok = (S >= threshold) & (J >= operand_floor)
    cost = np.where(ok, 1.0 - S, 1e6)
    ri, ci = linear_sum_assignment(cost)
    matched = []
    used_c, used_o = set(), set()
    for i, j in zip(ri, ci):
        if not ok[i, j]:
            continue
        matched.append({"claim": int(i), "object": int(j),
                        "similarity": float(S[i, j]),
                        "operand_jaccard": float(J[i, j]),
                        "true": bool(objs[j].true),
                        "artifact_class": objs[j].artifact_class,
                        "claim_id": claims[i].id, "object_id": objs[j].id})
        used_c.add(i)
        used_o.add(j)
    # An unmatched claim whose canonical key equals a matched claim's is not a
    # false discovery -- it is the same object expressed differently, and the
    # one-to-one assignment simply had to pick one. `spatial_interaction` makes
    # this concrete: the declared object is A * agg(B), the scan also finds
    # B * agg(A), the two design columns correlate 0.947, and on a symmetric
    # neighbourhood graph they are near-identical bilinear forms. Counting the
    # second as a spurious discovery understates precision for behaviour that
    # is correct. It is reported here rather than silently reclassified: what
    # counts as a false discovery is a scoring policy, and changing it would
    # move every recorded FDR.
    matched_keys = {claims[d["claim"]].key() for d in matched}
    duplicates = {claims[i].id: claims[i].key()
                  for i in range(len(claims))
                  if i not in used_c and claims[i].key() in matched_keys}
    return {
        "matched": matched,
        "unmatched_claims": [i for i in range(len(claims)) if i not in used_c],
        "duplicate_expressions": sorted(duplicates),
        "missed_true": [j for j, h in enumerate(objs)
                        if h.true and j not in used_o],
        "n_true": sum(1 for h in objs if h.true),
        "n_claims": len(claims),
        "threshold": threshold, "operand_floor": operand_floor,
        "blocked_by_structural_mismatch": blocked,
        "kin_matches": kin_near,
        "n_kin_matches": len(kin_near),
        "kin_similarity_cap": KIN_SIMILARITY_CAP,
    }


# --------------------------------------------------------------------------
def replicated(claim, alpha_rep=0.05, min_envs=1, require_sign=True):
    """Did the claim reproduce in independently generated environments?

    Sign agreement with the proposal effect is required by default: a
    relationship that flips direction between environments has not replicated,
    it has contradicted itself. This is the check the sign_reversing artifact
    class exists to be caught by.
    """
    n_ok, detail = 0, []
    for env, val in claim.replication.items():
        eff, p = (val if isinstance(val, (tuple, list)) else
                  (val.get("effect"), val.get("p")))
        sign_ok = (not require_sign) or (
            np.isfinite(eff) and np.isfinite(claim.effect)
            and np.sign(eff) == np.sign(claim.effect))
        ok = bool(np.isfinite(p) and p <= alpha_rep and sign_ok)
        detail.append({"env": env, "effect": float(eff) if eff is not None else None,
                       "p": float(p) if p is not None else None,
                       "sign_agrees": bool(sign_ok), "passes": ok})
        n_ok += int(ok)
    return {"n_environments": len(claim.replication), "n_replicated": n_ok,
            "passes": n_ok >= min_envs, "detail": detail}


def _same_object(claim, other, hyp, evaluate):
    """Is `claim` another expression of the object `other` already matched to?

    Two gates, both required. The identity gate first: no structural conflict
    against the object, the same operand set, the same identity parameters --
    the same test `match_claims` uses, not a looser one. Then evaluation
    equivalence: either the object itself declares this form, or the two design
    columns agree to |corr| >= SEMANTIC_CORR_MIN on the split `evaluate` reads.

    A claim that fails either is a distinct claim and is scored on its own
    merits. That is what stops a genuinely different object being absorbed:
    sharing an operand set is not enough, and a decoy correlating 0.85 with the
    truth is not folded.
    """
    if structural_conflict(claim, hyp) is not None:
        return None
    if set(claim.operands) != set(other.operands):
        return None
    if structural_conflict(claim, other) is not None:
        return None
    decl = declares_expression(hyp, claim)
    if evaluate is None:
        # Without an evaluator neither route can be completed. Folding on the
        # symbolic form alone is precisely how a decoy gets absorbed, so the
        # conservative answer is to fold nothing and say so.
        return None
    try:
        r = abscorr(evaluate(claim), evaluate(other))
    except Exception:
        return None
    if not np.isfinite(r):
        return None
    if decl is not None and r >= declared_abscorr_floor(decl):
        return {"route": "declared_by_the_generator", "abscorr": float(r),
                "floor": declared_abscorr_floor(decl)}
    if r >= SEMANTIC_CORR_MIN:
        return {"route": "evaluation_equivalence", "abscorr": float(r),
                "floor": SEMANTIC_CORR_MIN}
    return None


def rdc(claims, truth, alpha=0.05, delta=0.02, utility=None,
        utility_baseline=None, alpha_rep=0.05, min_envs=1,
        evaluate=None, fold_duplicates=True,
        slots=None, latents=None, min_abs_r=MIN_ABS_R):
    """RDC_{alpha,delta}: independently replicated discoveries at FDR alpha,
    subject to retaining ordinary utility.

    Three gates, in order:
      1. multiplicity   BH at level alpha over the proposal p-values;
      2. replication    survives in at least `min_envs` independent
                        environments with the same sign;
      3. utility        U >= U_0 - delta. If this fails, RDC is zero however
                        many objects survived -- the constraint is not a
                        tiebreak, it is a precondition.

    Alongside the headline count the report separates claims that matched a
    true object from those that matched a planted artifact and those that
    matched nothing, because the count on its own cannot distinguish a system
    that found ten objects from one that manufactured ten.
    """
    claims = list(claims)
    # When the claims name anonymous representation slots, score under BOTH
    # alignment rules and report both, plus each rule's structural cap. An
    # encoder arm scored under one rule silently would have a primary endpoint
    # decided by a scoring choice nobody made explicitly.
    if slots is not None and latents is not None:
        both, caps = {}, {}
        for rule in ("single", "subspace"):
            aligned, report = align_claims(claims, slots, latents, rule=rule,
                                           min_abs_r=min_abs_r)
            sub = rdc(aligned, truth, alpha=alpha, delta=delta,
                      utility=utility, utility_baseline=utility_baseline,
                      alpha_rep=alpha_rep, min_envs=min_envs,
                      evaluate=evaluate, fold_duplicates=fold_duplicates)
            sub["alignment"] = report
            both[rule] = sub
            caps[rule] = structural_cap(truth, slots, latents, rule=rule,
                                        min_abs_r=min_abs_r)
        out = dict(both["single"])
        out.update({
            "rdc_true_single": both["single"]["RDC_true"],
            "rdc_true_subspace": both["subspace"]["RDC_true"],
            "rdc_all_single": both["single"]["RDC_all"],
            "rdc_all_subspace": both["subspace"]["RDC_all"],
            "structural_cap": caps,
            "by_rule": both,
            "alignment_rules_reported": ["single", "subspace"],
            "primary_rule_is_a_scoring_ruling": (
                "both are reported and neither is declared primary; the "
                "choice has the standing of the p-resolution and "
                "duplicate-expression rulings"),
            "direction_guarantee": (
                "rdc_true_single <= rdc_true_subspace by construction: the "
                "subspace rule takes single-atom credit first and only then "
                "consults the joint span, at the same threshold in R^2 terms, "
                "so it is a strict superset. A cell where single exceeds "
                "subspace is a bug."),
        })
        return out

    p = np.array([c.p_value for c in claims], dtype=float)
    rej, q = benjamini_hochberg(p, alpha=alpha)
    m = match_claims(claims, truth)
    match_of = {d["claim"]: d for d in m["matched"]}

    surviving, rows = [], []
    claim_by_id = {c.id: c for c in claims}
    for i, c in enumerate(claims):
        r = replicated(c, alpha_rep=alpha_rep, min_envs=min_envs)
        d = match_of.get(i)
        row = {"claim_id": c.id, "operator": c.operator,
               "operands": list(c.operands), "p": float(c.p_value),
               "q": float(q[i]) if np.isfinite(q[i]) else None,
               "bh_rejected": bool(rej[i]),
               "replicated": r["passes"], "n_replicated": r["n_replicated"],
               "matched_object": d["object_id"] if d else None,
               "matched_true": bool(d["true"]) if d else False,
               "artifact_class": d["artifact_class"] if d else None,
               "complexity": c.complexity,
               "reported_mode": c.discovery_mode}
        row["counts_toward_rdc"] = bool(rej[i] and r["passes"])
        rows.append(row)
        if row["counts_toward_rdc"]:
            surviving.append(row)

    util_ok, util_gap = True, None
    if utility is not None and utility_baseline is not None:
        util_gap = float(utility - (utility_baseline - delta))
        util_ok = bool(utility >= utility_baseline - delta)

    # ---- fold duplicate expressions ------------------------------------
    # A duplicate expression of a planted object is that object, so it counts
    # once toward RDC and not at all toward the false-discovery numerator. The
    # expressions are recorded beneath the object so the multiplicity stays
    # visible without inflating the count.
    obj_of_claim, claim_by_i = {}, {i: c for i, c in enumerate(claims)}
    for i, c in enumerate(claims):
        d = match_of.get(i)
        if d and d["true"]:
            obj_of_claim[c.id] = d["object_id"]
    objs_by_id = {h.id: h for h in truth.objects}
    folded = {}
    if fold_duplicates:
        for i, c in enumerate(claims):
            if c.id in obj_of_claim or i not in set(m["unmatched_claims"]):
                continue
            for j, d in ((d["claim"], d) for d in m["matched"] if d["true"]):
                hyp = objs_by_id.get(d["object_id"])
                if hyp is None:
                    continue
                verdict = _same_object(c, claim_by_i[j], hyp, evaluate)
                if verdict:
                    folded[c.id] = dict(verdict, object_id=d["object_id"])
                    break
    for r in rows:
        f = folded.get(r["claim_id"])
        r["folded_into_object"] = f["object_id"] if f else None
        r["fold_route"] = f["route"] if f else None
        r["fold_abscorr"] = f["abscorr"] if f else None

    expressions = {}
    for r in surviving:
        oid = (r["matched_object"] if r["matched_true"]
               else r.get("folded_into_object"))
        if oid:
            expressions.setdefault(oid, []).append(
                {"claim_id": r["claim_id"],
                 "key": [str(x) for x in claim_by_id[r["claim_id"]].key()],
                 "route": ("matched" if r["matched_true"] else r["fold_route"]),
                 "abscorr": r.get("fold_abscorr")})

    n_objects = len(expressions)
    n_art = sum(1 for r in surviving
                if (not r["matched_true"]) and r["artifact_class"]
                and not r.get("folded_into_object"))
    n_none = sum(1 for r in surviving
                 if (not r["matched_true"]) and not r["artifact_class"]
                 and not r.get("folded_into_object"))
    n_true = n_objects
    dupes = set(m.get("duplicate_expressions", ()))
    n_dupe = sum(1 for r in surviving
                 if (not r["matched_true"]) and not r["artifact_class"]
                 and r["claim_id"] in dupes)
    return {
        "alpha": alpha, "delta": delta, "alpha_replication": alpha_rep,
        # The spec is RDC_{alpha,delta} = #{h: true, replicated, q <= alpha,
        # U >= U_0 - delta}. The truth clause was missing: `counts_toward_rdc`
        # was `bh_rejected AND replicated` and matching played no part, so
        # every RDC ever reported from here was the *replicated* count and
        # credited replicated false discoveries. `RDC` is now the spec's
        # quantity; the old one is kept under a name it cannot be mistaken for.
        "RDC": int(n_objects) if util_ok else 0,
        "RDC_true": int(n_objects) if util_ok else 0,
        "RDC_all": int(n_objects + n_art + n_none) if util_ok else 0,
        "RDC_true_uncapped": int(n_objects),
        "RDC_all_uncapped": int(n_objects + n_art + n_none),
        "FCR": (float((n_art + n_none) / (n_objects + n_art + n_none))
                if (n_objects + n_art + n_none) else 0.0),
        "RDC_definition": (
            "RDC == RDC_true == #{true objects with a certified, replicated "
            "expression}; RDC_all additionally counts surviving claims that "
            "matched no true object. FCR = (RDC_all - RDC_true) / RDC_all."),
        "RDC_counts_objects_not_claims": (
            "an object counts once however many of its expressions are "
            "certified; the expressions are listed under `expressions`"),
        "n_surviving_rows": int(len(surviving)),
        "utility_constraint_met": util_ok,
        "utility": utility, "utility_baseline": utility_baseline,
        "utility_slack": util_gap,
        "n_surviving_true": n_true,
        "n_surviving_planted_artifact": n_art,
        "n_surviving_unmatched": n_none,
        "n_surviving_duplicate_expression": n_dupe,
        "n_folded_duplicate_expressions": len(
            [r for r in surviving if r.get("folded_into_object")]),
        "n_objects_with_multiple_expressions": sum(
            1 for v in expressions.values() if len(v) > 1),
        "expressions": expressions,
        "semantic_corr_min": SEMANTIC_CORR_MIN,
        "fold_evaluator_supplied": evaluate is not None,
        "n_surviving_unexplained": n_none - n_dupe,
        "duplicate_expression_note": (
            "an unmatched survivor whose canonical key equals a matched "
            "claim's is the same object expressed differently, not a "
            "discovery that corresponds to nothing; RDC and the false "
            "discovery proportion are unchanged by this accounting and it is "
            "reported so the composition is legible"),
        "false_discovery_proportion": (
            float((n_art + n_none) / (n_objects + n_art + n_none))
            if (n_objects + n_art + n_none) else 0.0),
        "n_true_objects": m["n_true"],
        "recall": float(n_true / m["n_true"]) if m["n_true"] else float("nan"),
        "per_claim": rows,
        "artifact_susceptibility": artifact_susceptibility(rows),
    }


def comparable_p(claims, floor=None):
    """p-values put on a common resolution, for cross-kind ranking only.

    Every claim is floored at the coarsest resolution present in the bank, so
    two claims that both saturate their statistic tie instead of being ordered
    by an artefact of how many permutations each one happened to run. Ties at
    the floor are exactly the case where the standardised effect, which every
    specialist reports on the same scale, is the ordering key.

    Returns (comparable_p_list, report). BH, FDR control and every recovery
    verdict continue to take the raw `claim.p_value`; nothing here re-scores
    anything, and the ruling recorded in the frozen manifest says so.
    """
    from .oracle import claim_resolution
    claims = list(claims)
    res = [float(claim_resolution(c)) for c in claims]
    lim = float(max(res)) if floor is None else float(floor)
    out = [float(max(float(c.p_value), lim)) for c in claims]
    n_at_floor = sum(1 for v in out if v <= lim)
    return out, {
        "resolution_floor": lim,
        "per_claim_resolution": res,
        "n_claims": len(claims),
        "n_at_floor": n_at_floor,
        "n_distinguishable": len(claims) - n_at_floor,
        "tiebreak": "standardised effect (Claim.effect)",
        "note": ("cross-kind ranking only; BH and recovery verdicts use the "
                 "raw p-value"),
    }


def artifact_susceptibility(rows):
    """Which planted artifact class got through, per class.

    Kept separate from the pooled false-discovery rate because the classes are
    not interchangeable: `replication_mimicry` slipping through means the
    replication gate is doing what it is supposed to and is not sufficient,
    while `sign_reversing` slipping through means the replication gate is
    broken. Attribution decides which of those it is.
    """
    out = {}
    for r in rows:
        cls = r.get("artifact_class")
        if not cls:
            continue
        d = out.setdefault(cls, {"n_claimed": 0, "n_survived_rdc": 0})
        d["n_claimed"] += 1
        d["n_survived_rdc"] += int(r["counts_toward_rdc"])
    return out


# --------------------------------------------------------------------------
# secondary metrics
# --------------------------------------------------------------------------
def per_family_recall(results_by_family):
    """{family: rdc_report} -> per-family recall, precision and FDR."""
    out = {}
    for fam, rep in results_by_family.items():
        surviving = [r for r in rep["per_claim"] if r["counts_toward_rdc"]]
        tp = sum(1 for r in surviving if r["matched_true"])
        fp = len(surviving) - tp
        out[fam] = {
            "n_true_objects": rep["n_true_objects"],
            "n_recovered": tp,
            "recall": (tp / rep["n_true_objects"]
                       if rep["n_true_objects"] else float("nan")),
            "precision": (tp / len(surviving)) if surviving else float("nan"),
            "fdr": (fp / len(surviving)) if surviving else 0.0,
            "n_false": fp,
        }
    return out


def effect_calibration(claims, truth, match=None):
    """Are claimed effect sizes calibrated against the planted ones?

    Reports the OLS slope of claimed on true effect (1.0 is calibrated), the
    intercept, the correlation and the mean absolute error, over matched true
    objects only.
    """
    claims = list(claims)
    objs = list(truth.objects)
    match = match or match_claims(claims, truth)
    xs, ys, ids = [], [], []
    for d in match["matched"]:
        if not d["true"]:
            continue
        te = objs[d["object"]].effect
        ce = claims[d["claim"]].effect
        if np.isfinite(te) and np.isfinite(ce):
            xs.append(float(te))
            ys.append(float(ce))
            ids.append(objs[d["object"]].id)
    if len(xs) < 2:
        return {"n": len(xs), "slope": None, "intercept": None,
                "corr": None, "mae": None,
                "note": "fewer than two matched objects with finite effects"}
    x = np.array(xs)
    y = np.array(ys)
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"n": len(xs), "intercept": float(beta[0]), "slope": float(beta[1]),
            "corr": float(np.corrcoef(x, y)[0, 1]),
            "mae": float(np.mean(np.abs(y - x))),
            "objects": ids, "true_effects": xs, "claimed_effects": ys}


def n80(recovery_by_n, target=0.80):
    """n_80(h) = min{n : P(recover h) >= 0.80}.

    `recovery_by_n` maps a sample size to the fraction of seeds in which the
    object was recovered. Returns None when the target is never reached, with
    the observed maximum, rather than extrapolating: reporting an n_80 the grid
    never demonstrated would be a fabricated number.
    """
    items = sorted((int(n), float(p)) for n, p in recovery_by_n.items())
    for n, p in items:
        if p >= target:
            return {"n80": int(n), "target": target, "curve": items,
                    "reached": True}
    return {"n80": None, "target": target, "curve": items, "reached": False,
            "max_observed_rate": (max(p for _, p in items) if items else None),
            "max_n_tried": (items[-1][0] if items else None)}


def object_stability(claim_sets):
    """Jaccard stability of the recovered object set across repeated runs.

    `claim_sets` is a list of iterables of Claim. Objects are compared by
    `Claim.key()`, i.e. by identity, not by slot index -- the seed-instability
    failure mode of the earlier DISCO work came from comparing atom indices.
    """
    keys = [set(c.key() for c in cs) for cs in claim_sets]
    if len(keys) < 2:
        return {"n_runs": len(keys), "mean_jaccard": None,
                "core_fraction": None}
    js = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            u = keys[i] | keys[j]
            js.append(len(keys[i] & keys[j]) / max(len(u), 1))
    allk = set().union(*keys)
    core = set.intersection(*keys) if keys else set()
    counts = {k: sum(1 for s in keys if k in s) for k in allk}
    return {"n_runs": len(keys), "mean_jaccard": float(np.mean(js)),
            "core_fraction": len(core) / max(len(allk), 1),
            "n_union": len(allk), "n_core": len(core),
            "selection_frequency": {str(k): c / len(keys)
                                    for k, c in counts.items()}}


#: Nonlinear/structural operator vocabulary used by `hypothesis_complexity`'s
#: functional-complexity term. Mirrors the set trainprotocol Sec.12's
#: prototype used, kept here as the single canonical copy.
_NONLINEAR_COMPLEXITY_OPS = frozenset({
    "THRESHOLD", "HILL", "SATURATE", "XOR", "CONDITION", "MUL", "AND",
    "SPACE", "LAG", "ANOMALY", "REGIME",
})


def hypothesis_complexity(h, weights=(1, 1, 1, 1, 1)) -> dict:
    """Predeclared scientific-object complexity score C(h) (trainprotocol Sec.12).

    C(h) = w1*(#operands) + w2*(operator depth) + w3*(#contexts)
         + w4*(spatial/temporal dependencies) + w5*(functional complexity)

    Weights are predeclared as (1, 1, 1, 1, 1) -- equal, unit weights, fixed
    before any Delta-RDC-vs-C fit is run, per Sec.12's instruction to define
    complexity before running the experiment. Scores a ground-truth
    `Hypothesis` (`h`), not a compiler `Claim` -- see `program_complexity`
    for the analogous score over claims.
    """
    composition = list(h.params.get("composition", [])) or [h.operator]
    n_operands = len(h.operands)
    operator_depth = len(composition) if h.params.get("composition") else 1
    n_contexts = len(h.context)
    has_spatial = int(any(op == "SPACE" for op in composition)
                       or "radius" in h.params)
    has_temporal = int(h.lag is not None or any(op == "LAG" for op in composition))
    spatiotemporal = has_spatial + has_temporal
    functional = sum(1 for op in composition if op in _NONLINEAR_COMPLEXITY_OPS)
    w1, w2, w3, w4, w5 = weights
    C = (w1 * n_operands + w2 * operator_depth + w3 * n_contexts
         + w4 * spatiotemporal + w5 * functional)
    return {
        "id": h.id, "operator": h.operator, "composition": composition,
        "n_operands": n_operands, "operator_depth": operator_depth,
        "n_contexts": n_contexts, "has_spatial": has_spatial,
        "has_temporal": has_temporal, "functional_complexity": functional,
        "C": C,
    }


def program_complexity(claims):
    c = [int(x.complexity) for x in claims]
    return {"n": len(c), "mean": float(np.mean(c)) if c else None,
            "median": float(np.median(c)) if c else None,
            "max": int(max(c)) if c else None, "total": int(sum(c))}


def compositional_ood_accuracy(results_by_composition, heldout):
    """Recovery on held-out compositions versus development compositions."""
    held = {tuple(c) for c in heldout}
    dev_r, held_r = [], []
    for comp, rep in results_by_composition.items():
        r = rep.get("recall")
        if r is None or not np.isfinite(r):
            continue
        (held_r if tuple(comp) in held else dev_r).append(float(r))
    return {
        "n_dev": len(dev_r), "n_heldout": len(held_r),
        "dev_recall": float(np.mean(dev_r)) if dev_r else None,
        "heldout_recall": float(np.mean(held_r)) if held_r else None,
        "gap": (float(np.mean(dev_r) - np.mean(held_r))
                if dev_r and held_r else None),
    }


def mode_confusion(claims, truth, match=None):
    """Diagnostic only: how often was the reported discovery-mode label right?

    Deliberately excluded from every headline metric. A system may recover an
    object perfectly and call it the wrong kind of discovery; that is a
    labelling error, not a failure of discovery, and the benchmark is not a
    classifier benchmark.
    """
    claims = list(claims)
    objs = list(truth.objects)
    match = match or match_claims(claims, truth)
    tab = {}
    n_ok = n_tot = 0
    for d in match["matched"]:
        if not d["true"]:
            continue
        rep = claims[d["claim"]].discovery_mode
        act = objs[d["object"]].mode
        if rep is None:
            continue
        tab.setdefault(act, {}).setdefault(rep, 0)
        tab[act][rep] += 1
        n_tot += 1
        n_ok += int(rep == act)
    return {"n_labelled": n_tot,
            "mode_accuracy": (n_ok / n_tot) if n_tot else None,
            "confusion": tab,
            "note": "diagnostic only; contributes nothing to RDC or recall"}
