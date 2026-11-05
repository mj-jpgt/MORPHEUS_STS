"""Aligning anonymous representation slots to planted latents, before matching.

An encoder names its objects `slot0`, `slot7`; the truth names them `A`, `B`.
Operand Jaccard is then 0 whatever the operator spelling, so `n_surviving_true`
was 0 in all twelve rotation arms -- not because nothing was discovered but
because nothing could be credited. Operand naming binds before vocabulary.

Two alignment rules are provided and **both** are always reported. Choosing one
silently would decide a scoring question that belongs to the owner.

`single`
    Each operand slot aligns to the one latent it best tracks, at
    |r| >= `MIN_ABS_R`, and the latents assigned within a claim must be
    distinct. This is `cscd.matching.align_atoms_to_factors`, the rule the
    step-3 baseline matrix used.

`subspace`
    An operand's slots may *jointly* span a latent: a latent is credited to a
    slot set when its multiple correlation on that set's span clears the same
    threshold in R^2 terms, and credited sets must be disjoint. This is
    `cscd.matching.align_tuples_to_factors_subspace`.

The threshold for the subspace rule is `MIN_ABS_R ** 2`, not the 0.50 that was
suggested. 0.50 would make the subspace rule *stricter* per latent than the
single-atom rule -- a latent tracked by one slot at r = 0.35 (R^2 = 0.12) would
pass `single` and fail `subspace` -- and the predeclared direction
`rdc_true_single <= rdc_true_subspace` would be violated by construction rather
than by any discovery. At `MIN_ABS_R ** 2`, with single-atom credit taken
first, the subspace rule is a strict superset and the direction holds as an
identity. That is also exactly how `cscd` builds it.

Structural caps
---------------
A cap is what the rule can credit *before any claim exists*: the fraction of
planted true objects whose operand latents are reachable at all under the rule,
given this representation. The step-3 leakage audit measured PCA capped at
0.222 and four self-supervised representations at 0.000 on every seed. A capped
world must read as "capped by the rule", never as "the method found nothing",
so `structural_cap` is computed from the latents and the slots alone and is
reported beside every result.
"""
from __future__ import annotations

import numpy as np

#: correlation floor for crediting a slot with a latent. Mirrors
#: `cscd.matching.align_atoms_to_factors`'s default; the mirroring is enforced
#: by `tests/test_slot_alignment.py`.
MIN_ABS_R = 0.30


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    da, db = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(a @ b / (da * db))


def _r2_on_span(cols, y, ridge=1e-8):
    """R^2 of `y` regressed on the span of `cols`."""
    if not cols:
        return 0.0
    X = np.column_stack([np.asarray(c, dtype=np.float64).ravel() for c in cols])
    X = X - X.mean(0, keepdims=True)
    y = np.asarray(y, dtype=np.float64).ravel()
    yc = y - y.mean()
    denom = float(yc @ yc)
    if denom < 1e-12:
        return 0.0
    A = X.T @ X + ridge * np.eye(X.shape[1])
    try:
        beta = np.linalg.solve(A, X.T @ yc)
    except np.linalg.LinAlgError:                            # pragma: no cover
        return 0.0
    resid = yc - X @ beta
    return float(max(0.0, 1.0 - (resid @ resid) / denom))


def is_slot_name(name, slots):
    return name in slots


def align_single(slot_names, slots, latents, min_abs_r=MIN_ABS_R):
    """{slot: latent} under the single-atom rule, or None if it cannot align.

    Every slot must reach some latent at |r| >= min_abs_r and the assignment
    must be injective -- two slots collapsing onto one latent is not an
    alignment, it is one latent reported twice.
    """
    lat = sorted(latents)
    assign, used = {}, set()
    for s in slot_names:
        rs = [(abs(_pearson(slots[s], latents[nm])), nm) for nm in lat]
        rs.sort(reverse=True)
        pick = next(((r, nm) for r, nm in rs if r >= min_abs_r and nm not in used),
                    None)
        if pick is None:
            return None
        used.add(pick[1])
        assign[s] = pick[1]
    return assign


def align_subspace(slot_names, slots, latents, min_abs_r=MIN_ABS_R):
    """{slot-set: latent-set} under the subspace rule, or None.

    Single-atom credit is taken first so this rule is a strict superset of
    `align_single`; only if that fails is the joint span consulted.
    """
    single = align_single(slot_names, slots, latents, min_abs_r=min_abs_r)
    if single is not None:
        return {"assignment": single, "route": "single_atom",
                "r2": None}
    # Distinctness is a property of the alignment, not of the route. If two or
    # more slots individually track the SAME latent above the floor, that is
    # one latent reported twice and the alignment fails however well the span
    # behaves. Without this the subspace route quietly credited two A-tracking
    # slots with {A}, turning a two-operand claim into a one-operand one that
    # then collected partial credit against a two-operand truth.
    best = {}
    for sname in slot_names:
        rs = sorted(((abs(_pearson(slots[sname], latents[nm])), nm)
                     for nm in latents), reverse=True)
        if rs and rs[0][0] >= min_abs_r:
            best.setdefault(rs[0][1], []).append(sname)
    if any(len(v) > 1 for v in best.values()):
        return None
    thr = float(min_abs_r) ** 2
    cols = [slots[s] for s in slot_names]
    # Credit every latent the span recovers, not exactly as many latents as
    # there are slots. Several slots jointly representing ONE latent is the
    # case the subspace rule exists for -- a LINCOMB over four slots spanning
    # `A` -- and requiring a latent per slot made exactly that case
    # uncreditable, which is the failure this rule was added to remove.
    scored = sorted(((_r2_on_span(cols, latents[nm]), nm) for nm in latents),
                    reverse=True)
    chosen = [(r2, nm) for r2, nm in scored if r2 >= thr]
    if not chosen:
        return None
    # more latents than slots cannot be a faithful reading of the span
    chosen = chosen[:len(slot_names)]
    return {"assignment": None, "route": "subspace",
            "latents": sorted(nm for _, nm in chosen),
            "r2": [float(r) for r, _ in chosen]}


def align_claim(claim, slots, latents, rule="single", min_abs_r=MIN_ABS_R):
    """The latent names this claim's operands stand for, or None."""
    ops = list(claim.operands)
    if not ops or not all(o in slots for o in ops):
        return None                       # already named, or partly named
    if rule == "single":
        a = align_single(ops, slots, latents, min_abs_r=min_abs_r)
        return None if a is None else {"operands": tuple(a[o] for o in ops),
                                       "route": "single_atom", "detail": a}
    res = align_subspace(ops, slots, latents, min_abs_r=min_abs_r)
    if res is None:
        return None
    if res["route"] == "single_atom":
        a = res["assignment"]
        return {"operands": tuple(a[o] for o in ops), "route": "single_atom",
                "detail": a}
    return {"operands": tuple(res["latents"]), "route": "subspace",
            "detail": {"r2": res["r2"]}}


def structural_cap(truth, slots, latents, rule="single", min_abs_r=MIN_ABS_R):
    """What fraction of planted objects this rule could credit at all.

    Computed from the latents and the slots, with no claims involved: it is a
    property of the representation and the rule, not of any method. Reported
    beside every result so a capped world is never read as a failed detector.
    """
    objs = [h for h in truth.objects if h.true]
    attainable = 0
    detail = []
    for h in objs:
        want = [o for o in h.operands if o in latents]
        if not want:
            detail.append({"object_id": h.id, "attainable": None,
                           "why": "operands are not latents of this world"})
            continue
        ok = _object_attainable(want, slots, latents, rule, min_abs_r)
        attainable += int(ok)
        detail.append({"object_id": h.id, "attainable": bool(ok)})
    n = len([d for d in detail if d["attainable"] is not None])
    return {
        "rule": rule, "min_abs_r": min_abs_r,
        "n_true_objects": len(objs),
        "n_scored": n,
        "n_attainable": attainable,
        "cap": (attainable / n) if n else float("nan"),
        "n_slots": len(slots),
        "detail": detail,
        "note": ("the fraction of planted objects reachable under this rule "
                 "before any claim exists; a low cap is a property of the "
                 "representation, not evidence about a method"),
    }


def _object_attainable(want, slots, latents, rule, min_abs_r):
    """Could *some* set of slots align to this object's latents?"""
    names = list(slots)
    if rule == "single":
        used = set()
        for nm in want:
            hit = None
            for s in names:
                if s in used:
                    continue
                if abs(_pearson(slots[s], latents[nm])) >= min_abs_r:
                    hit = s
                    break
            if hit is None:
                return False
            used.add(hit)
        return True
    thr = float(min_abs_r) ** 2
    if _object_attainable(want, slots, latents, "single", min_abs_r):
        return True
    cols = [slots[s] for s in names]
    return all(_r2_on_span(cols, latents[nm]) >= thr for nm in want)
