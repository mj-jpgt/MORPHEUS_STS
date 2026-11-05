"""Slot->latent alignment, both rules, and the caps that keep them honest.

Encoder claims name `slot0`, `slot7`; truth names `A`, `B`. Operand Jaccard is
0 whatever the operator spelling, so `n_surviving_true` was 0 in all twelve
rotation arms -- by the scorer, not by discovery. Operand naming binds before
vocabulary.

Both rules are reported and neither is declared primary; that choice is a
scoring ruling. What is pinned here is that each rule does what it says, that
the direction between them holds by construction, and that a representation's
structural cap is reported so a capped world never reads as a failed detector.
"""
import numpy as np
import pytest

from dbench.align import (MIN_ABS_R, _pearson, _r2_on_span, align_claim,
                          align_single, align_subspace, structural_cap)
from dbench.environments import build_world
from dbench.metrics import Claim, rdc

REP = {"R0": (0.8, 1e-6)}


def _latents(w):
    return {k: np.asarray(v, float) for k, v in w.latents.items()
            if np.asarray(v).ndim == 1}


def _claim(ops, operator="AND", cid="c"):
    return Claim(id=cid, operator=operator, operands=tuple(ops), target="Y",
                 effect=0.9, p_value=1e-10, replication=dict(REP))


def _noisy(col, r_target, rng):
    """A slot correlating about `r_target` with `col`."""
    z = (col - col.mean()) / col.std()
    e = rng.standard_normal(len(z))
    e = e - e.mean()
    e /= e.std()
    a = r_target / np.sqrt(max(1e-12, 1.0 - r_target ** 2))
    return a * z + e


# --------------------------------------------------------------------------
# the rule is imported, not reinvented
# --------------------------------------------------------------------------
def test_the_correlation_floor_matches_the_step3_rule():
    try:
        import inspect

        from cscd.matching import align_atoms_to_factors
    except Exception as exc:                                 # pragma: no cover
        pytest.skip("cscd not importable here: %r" % exc)
    default = inspect.signature(align_atoms_to_factors).parameters[
        "min_abs_r"].default
    assert MIN_ABS_R == default


def test_the_subspace_threshold_is_the_square_of_the_single_atom_floor():
    """0.50 was suggested and is not used. At 0.50 a latent tracked by one slot
    at r = 0.35 (R^2 = 0.12) would pass `single` and fail `subspace`, so the
    predeclared direction would be violated by construction. At MIN_ABS_R ** 2,
    with single-atom credit taken first, subspace is a strict superset."""
    assert (0.35 ** 2) < 0.50
    assert (0.35 ** 2) > MIN_ABS_R ** 2


# --------------------------------------------------------------------------
# the four known answers
# --------------------------------------------------------------------------
def test_an_oracle_latent_claim_matches_identically_before_and_after():
    w = build_world("pure_and", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "AND"][0]
    rng = np.random.default_rng(0)
    slots = {"slot0": _noisy(np.asarray(w.latents[h.operands[0]], float), 0.99, rng),
             "slot1": _noisy(np.asarray(w.latents[h.operands[1]], float), 0.99, rng)}
    plain = rdc([_claim(h.operands)], w.truth)
    aligned = rdc([_claim(h.operands)], w.truth, slots=slots,
                  latents=_latents(w))
    assert plain["RDC_true"] == 1
    assert aligned["rdc_true_single"] == aligned["rdc_true_subspace"] == 1
    assert not any(a["aligned"] for a in aligned["by_rule"]["single"]["alignment"]), (
        "a claim already naming latents must not be rewritten")


def test_two_slots_at_099_match_the_conjunction():
    w = build_world("pure_and", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "AND"][0]
    rng = np.random.default_rng(0)
    A = np.asarray(w.latents[h.operands[0]], float)
    B = np.asarray(w.latents[h.operands[1]], float)
    slots = {"slot0": _noisy(A, 0.99, rng), "slot1": _noisy(B, 0.99, rng)}
    assert abs(_pearson(slots["slot0"], A)) > 0.95
    r = rdc([_claim(("slot0", "slot1"))], w.truth, slots=slots,
            latents=_latents(w))
    assert r["rdc_true_single"] == 1
    assert r["rdc_true_subspace"] == 1


def test_two_slots_aligning_to_the_same_latent_fail_distinctness():
    """One latent reported twice is not two operands."""
    w = build_world("pure_and", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "AND"][0]
    rng = np.random.default_rng(0)
    A = np.asarray(w.latents[h.operands[0]], float)
    slots = {"slot0": _noisy(A, 0.95, rng), "slot1": _noisy(A, 0.95, rng)}
    assert align_single(["slot0", "slot1"], slots, _latents(w)) is None
    r = rdc([_claim(("slot0", "slot1"))], w.truth, slots=slots,
            latents=_latents(w))
    assert r["rdc_true_single"] == 0
    assert r["rdc_true_subspace"] == 0


def test_a_distributed_representation_is_credited_only_by_the_subspace_rule():
    """Four slots jointly spanning a latent, none of them individually above
    the single-atom floor. This is the case a distributed representation is in,
    and the case a single-atom rule scores as zero discoveries."""
    w = build_world("additive_factors", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "ADD"][0]
    nm = h.operands[0]
    A = np.asarray(w.latents[nm], float)
    rng = np.random.default_rng(1)
    An = (A - A.mean()) / A.std()
    E = np.linalg.qr(rng.standard_normal((len(A), 4)))[0] * np.sqrt(len(A))
    slots = {"s%d" % i: 0.25 * An + E[:, i] for i in range(4)}

    assert max(abs(_pearson(v, A)) for v in slots.values()) < MIN_ABS_R
    assert _r2_on_span(list(slots.values()), A) >= MIN_ABS_R ** 2

    r = rdc([_claim(tuple(slots), operator="ADD")], w.truth, slots=slots,
            latents=_latents(w))
    assert r["rdc_true_single"] == 0
    assert r["rdc_true_subspace"] == 1
    al = [a for a in r["by_rule"]["subspace"]["alignment"] if a["aligned"]]
    assert al and al[0]["route"] == "subspace" and al[0]["to"] == [nm]


# --------------------------------------------------------------------------
# the direction guarantee
# --------------------------------------------------------------------------
@pytest.mark.parametrize("r_target", (0.10, 0.25, 0.35, 0.60, 0.95))
def test_single_never_exceeds_subspace(r_target):
    w = build_world("pure_and", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "AND"][0]
    rng = np.random.default_rng(2)
    A = np.asarray(w.latents[h.operands[0]], float)
    B = np.asarray(w.latents[h.operands[1]], float)
    slots = {"slot0": _noisy(A, r_target, rng), "slot1": _noisy(B, r_target, rng)}
    r = rdc([_claim(("slot0", "slot1"))], w.truth, slots=slots,
            latents=_latents(w))
    assert r["rdc_true_single"] <= r["rdc_true_subspace"], r_target


def test_the_subspace_rule_takes_single_atom_credit_first():
    """That is what makes it a superset rather than a different rule."""
    w = build_world("pure_and", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "AND"][0]
    rng = np.random.default_rng(3)
    slots = {"slot0": _noisy(np.asarray(w.latents[h.operands[0]], float), 0.9, rng),
             "slot1": _noisy(np.asarray(w.latents[h.operands[1]], float), 0.9, rng)}
    res = align_subspace(["slot0", "slot1"], slots, _latents(w))
    assert res["route"] == "single_atom"


# --------------------------------------------------------------------------
# structural caps
# --------------------------------------------------------------------------
def test_the_cap_is_computed_without_any_claims():
    w = build_world("pure_and", 0, n=4000)
    rng = np.random.default_rng(4)
    slots = {"s0": rng.standard_normal(w.n), "s1": rng.standard_normal(w.n)}
    cap = structural_cap(w.truth, slots, _latents(w), rule="single")
    assert cap["cap"] == 0.0, "pure noise slots can credit nothing"
    assert cap["n_true_objects"] > 0 and cap["n_slots"] == 2
    assert "before any claim exists" in cap["note"]


def test_a_capped_representation_reports_its_cap_beside_the_result():
    w = build_world("additive_factors", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "ADD"][0]
    A = np.asarray(w.latents[h.operands[0]], float)
    rng = np.random.default_rng(1)
    An = (A - A.mean()) / A.std()
    E = np.linalg.qr(rng.standard_normal((len(A), 4)))[0] * np.sqrt(len(A))
    slots = {"s%d" % i: 0.25 * An + E[:, i] for i in range(4)}
    r = rdc([_claim(tuple(slots), operator="ADD")], w.truth, slots=slots,
            latents=_latents(w))
    caps = r["structural_cap"]
    assert caps["single"]["cap"] == 0.0
    assert caps["subspace"]["cap"] > 0.0
    assert caps["single"]["cap"] <= caps["subspace"]["cap"], (
        "a stricter rule cannot have a higher cap")


def test_a_perfect_representation_is_not_capped():
    w = build_world("pure_and", 0, n=4000)
    lat = _latents(w)
    slots = {"slot_%s" % k: v.copy() for k, v in lat.items()}
    for rule in ("single", "subspace"):
        cap = structural_cap(w.truth, slots, lat, rule=rule)
        assert cap["cap"] == 1.0, (rule, cap["cap"])


# --------------------------------------------------------------------------
# unalignable claims are not credited
# --------------------------------------------------------------------------
def test_an_unalignable_claim_keeps_its_slot_names_and_does_not_match():
    w = build_world("pure_and", 0, n=4000)
    rng = np.random.default_rng(5)
    slots = {"slot0": rng.standard_normal(w.n), "slot1": rng.standard_normal(w.n)}
    r = rdc([_claim(("slot0", "slot1"))], w.truth, slots=slots,
            latents=_latents(w))
    assert r["rdc_true_single"] == 0 and r["rdc_true_subspace"] == 0
    assert r["rdc_all_single"] == 1, "it still counts as a replicated claim"


def test_a_partly_named_claim_is_left_alone():
    w = build_world("pure_and", 0, n=4000)
    h = [x for x in w.truth.true_objects if x.operator == "AND"][0]
    rng = np.random.default_rng(6)
    slots = {"slot0": _noisy(np.asarray(w.latents[h.operands[0]], float), 0.99, rng)}
    c = _claim(("slot0", h.operands[1]))
    assert align_claim(c, slots, _latents(w), rule="single") is None
