"""One operator table, two spellings.

`dbench` names its operators `ADD`, `AND`, `THRESHOLD`, `CONDITION`;
`morpheus_compiler` names the same concepts `ADD`, `MUL`, `THR`, `COND`. Both
packages were written against the same concept list -- the compiler's
`hgrammar` mirrors `dbench.world.OPERATORS` on purpose -- but nothing enforced
the mirroring, and the spellings drifted.

The consequence was not cosmetic. `metrics.object_similarity` compares operator
names for equality, so a claim built from a compiler program forfeited the whole
0.20 operator weight against every dbench truth object: measured similarity
0.7999 where 1.0 was available. A uniform penalty does not reorder arms, but the
match threshold is 0.70, so a claim whose true similarity is 0.85 read 0.65 and
was not credited at all. RDC was therefore not merely biased low, it was
truncated by an amount that depends on where each arm's similarity distribution
sits relative to the threshold -- and the specialist bank, whose claims are
built in dbench's own spelling, paid nothing. The comparison was tilted toward
the specialists by construction.

This module is the single table. It lives in `dbench` because `dbench` is the
frozen artefact and the lower-level package: the compiler may import it, and it
imports nothing. `metrics` normalises both sides through `to_canonical` before
comparing, so a consumer who forgets to translate is not silently penalised.

Not every name has a counterpart, and the table says so rather than guessing.
`MANIFOLD`, `MIXTURE`, `CONFOUND` and `DESIGN` are dbench-only; `OR`, `NAND`,
`NOR`, `XNOR` and `DERIV` are compiler-only. Those map to themselves and are
reported by `coverage()`. `LINCOMB` used to be in that compiler-only list and
is now shared, because `linear_combination` plants it; the ruling is recorded
beside its entry in `DBENCH_TO_COMPILER`.
"""
from __future__ import annotations

#: canonical dbench name -> the compiler's primary spelling of the same concept
DBENCH_TO_COMPILER = {
    # RULING: the compiler's bare-variable leaf is dbench's additive main
    # effect. In a program whose target is the response, a bare VAR at the
    # root asserts "this variable has an effect on Y", and that is exactly
    # what dbench plants as ADD. The evidence is that no dbench world plants
    # IDENTITY as a *true* object at all -- it appears only on two planted
    # artifacts (gate_capture, batch_aligned), where it names an entity-mode
    # object whose target is a modality rather than the response. Sending VAR
    # to IDENTITY therefore gave a correct main-effect claim kin credit
    # instead of exact, which moved tabular's oracle ceiling from exact 2 to
    # exact 1.
    #
    # ADD and IDENTITY remain distinct and remain kin, not equal. They share a
    # design column -- `features.IDENTITY_OPS` treats them identically -- and
    # differ in what they assert: ADD is a relational claim about the
    # response, IDENTITY an entity-mode claim that a state is an object in its
    # own right. A method that says "this is a thing" when the truth says
    # "this drives Y" has found the right variable and the wrong claim, which
    # is what kin credit is for.
    "ADD": "VAR",
    "AND": "MUL",
    "THRESHOLD": "THR",
    "SATURATE": "SAT",
    "CONDITION": "COND",
    "ANOMALY": "ANOM",
    # identical in both packages
    "DIFF": "DIFF",
    "RATIO": "RATIO",
    "XOR": "XOR",
    "SPACE": "SPACE",
    "LAG": "LAG",
    "GRAPH": "GRAPH",
    "CAUSE": "CAUSE",
    "SYMMETRY": "SYMMETRY",
    "ODE": "ODE",
    # RULING: LINCOMB is no longer compiler-only. `linear_combination` plants
    # it as a true dbench object -- one weighted combination, declared as a
    # single object rather than as several ADDs -- so the old alias
    # `LINCOMB -> ADD` became wrong in a way that mattered: it made
    # `to_canonical("LINCOMB") == "ADD"`, and a one-operand ADD claim was then
    # operator-*equal* to a two-operand planted LINCOMB, scoring
    # 0.55*0.5 + 0.20 + 0.10 + 0.075 + 0.075 = 0.725, above MATCH_THRESHOLD.
    # A bank of single-operand detectors would have read recall 1.000 on an
    # object it cannot express, which is the exact defect the operator ruling
    # was written to close.
    #
    # LINCOMB and ADD stay *kin* in `metrics._KIN`, with the kin cap applying:
    # a method that reports one constituent of a planted combination has the
    # right variable and the wrong granularity, which is what kin credit is
    # for. Consequence to broadcast: a compiler claim spelled LINCOMB against a
    # planted ADD truth used to score exact and now scores kin.
    "LINCOMB": "LINCOMB",
}

#: extra compiler spellings that resolve to a dbench name but are not the
#: primary one, so the mapping back is unambiguous. `AND` is the compiler's
#: boolean conjunction; dbench has a single conjunction operator and the
#: kinship rule in `metrics` already treats the combiners as related, so it
#: resolves there rather than being left unmatched.
COMPILER_ALIASES = {
    "AND": "AND",
    # the compiler has no separate spelling for the entity-mode object; a bare
    # leaf is VAR whichever is meant, and VAR resolves to ADD above
    "ADD": "ADD",
}

#: names with no counterpart in the other package. Recorded rather than mapped
#: to something approximate: a wrong match is worse than an honest miss.
DBENCH_ONLY = ("MANIFOLD", "MIXTURE", "CONFOUND", "DESIGN",
               "IDENTITY")
COMPILER_ONLY = ("OR", "NAND", "NOR", "XNOR", "DERIV")   # LINCOMB left this
#: set when `linear_combination` began planting it; see the ruling above.

_COMPILER_TO_DBENCH = {v: k for k, v in DBENCH_TO_COMPILER.items()}
_COMPILER_TO_DBENCH.update(COMPILER_ALIASES)


def to_canonical(name):
    """The dbench spelling of `name`, whichever package it came from.

    Unknown names pass through unchanged. A name this table does not know is
    not evidence that it is wrong -- it may be a new operator -- and silently
    rewriting it would be worse than leaving it alone.
    """
    if name is None:
        return None
    n = str(name).upper()
    return _COMPILER_TO_DBENCH.get(n, n)


def to_compiler(name):
    """The compiler spelling of a dbench operator name."""
    if name is None:
        return None
    n = str(name).upper()
    return DBENCH_TO_COMPILER.get(n, n)


def coverage():
    """Which names each package has, and which cross the seam."""
    from .world import OPERATORS
    dbench = set(OPERATORS)
    mapped = set(DBENCH_TO_COMPILER)
    return {
        "n_dbench_operators": len(dbench),
        "n_mapped": len(mapped),
        "dbench_only": sorted(dbench - mapped),
        "compiler_only": sorted(COMPILER_ONLY),
        "aliases": dict(COMPILER_ALIASES),
        "unmapped_dbench_names_are_expected": sorted(DBENCH_ONLY),
        "consistent": sorted(dbench - mapped) == sorted(DBENCH_ONLY),
    }


def check_round_trip():
    """Every dbench name survives a round trip; every compiler name resolves."""
    bad = []
    for d in DBENCH_TO_COMPILER:
        if to_canonical(to_compiler(d)) != d:
            bad.append(("round_trip", d))
    for c, d in _COMPILER_TO_DBENCH.items():
        if to_canonical(c) != d:
            bad.append(("resolve", c))
    for n in DBENCH_ONLY + COMPILER_ONLY:
        if to_canonical(n) != n and n not in _COMPILER_TO_DBENCH:
            bad.append(("passthrough", n))
    return {"problems": bad, "passes": not bad}
