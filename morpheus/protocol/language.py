"""Claim-language discipline, enforced in code on every free-text field the
runner writes into the registry.

The rule this implements (MEMORY: "Claim language & discovery object"):

* A computational ablation establishes that a predictor **depends on** a
  representation. It does not establish that the representation is
  *necessary*, *sufficient*, or that it *causes* the outcome. Removing a
  feature and watching a model get worse is a statement about the model.
* A spatial gradient is a **gradient**. It is not a *trajectory*, a
  *pseudotime*, or a *lineage*: a static section has no time axis, and
  ordering cells along an axis of expression does not make the axis temporal.

Why this is a code check and not a style note
---------------------------------------------
The registry is append-only. A sentence with "necessary" in it, once on the
chain, is on the chain -- it can be amended by a `followup` but never edited
out. The cheapest place to stop an overclaim is before `_append` runs, so
`check` is called on every `notes`, `description`, `measured` and `mechanism`
string the runner passes down, and `LanguageDisciplineError` is raised rather
than warned.

Escape hatch
------------
Verbatim quotation from a cited source is not the runner's own claim, so text
between ``{{quote: ...}}`` markers is exempt. It is the *only* exemption, it
is visible in the record itself, and it is deliberately ugly so that using it
is a decision rather than a habit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Words that assert causal necessity/sufficiency from an associational or
#: ablation-based result.
CAUSAL_WORDS: Tuple[str, ...] = (
    "necessary", "necessity", "sufficient", "sufficiency", "indispensable",
    "essential", "requires", "required for", "causes", "caused", "causing",
    "causal", "causally", "causation", "proves", "proven", "proof that",
    "establishes that", "demonstrates that",
)

#: Words that assert a temporal ordering from static or spatial data.
TEMPORAL_WORDS: Tuple[str, ...] = (
    "trajectory", "trajectories", "pseudotime", "pseudo-time",
    "lineage progression", "developmental path", "differentiation path",
    "progresses over time", "evolves into", "precursor state of",
)

CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "causal": CAUSAL_WORDS,
    "temporal": TEMPORAL_WORDS,
}

#: What to say instead. Reported back with every violation so the fix is
#: obvious and does not require reading this module.
SANCTIONED_PHRASING: Dict[str, str] = {
    "causal": (
        "say what was measured: 'the predictor depends on <X>', "
        "'removing <X> reduces <metric> by <amount>', "
        "'<X> carries information about <Y> beyond <parents>'"
    ),
    "temporal": (
        "say what was measured: 'a spatial gradient in <X>', "
        "'<X> varies monotonically with distance from <Y>' -- "
        "a gradient is not a trajectory unless time was observed"
    ),
}

_QUOTE_RE = re.compile(r"\{\{quote:.*?\}\}", re.DOTALL | re.IGNORECASE)


class LanguageDisciplineError(ValueError):
    """Raised when a free-text field would put an overclaim on the chain."""


@dataclass(frozen=True)
class Violation:
    field: str
    category: str
    word: str
    context: str
    remedy: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (f"{self.field}: forbidden {self.category} term {self.word!r} "
                f"in ...{self.context}... -- {self.remedy}")


def strip_quotations(text: str) -> str:
    """Blank out ``{{quote: ...}}`` spans, preserving length so offsets in
    reported context stay meaningful."""
    return _QUOTE_RE.sub(lambda m: " " * len(m.group(0)), text)


def _pattern(word: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(word).replace(r"\ ", r"\s+")
                      + r"(?![\w-])", re.IGNORECASE)


_COMPILED = {cat: [(w, _pattern(w)) for w in words]
             for cat, words in CATEGORIES.items()}


def check(text: object, *, field: str = "text",
          categories: Sequence[str] = ("causal", "temporal")) -> List[Violation]:
    """Every violation in `text`. Never raises; returns a (possibly empty) list."""
    if text is None:
        return []
    s = strip_quotations(str(text))
    out: List[Violation] = []
    for cat in categories:
        for word, pat in _COMPILED[cat]:
            for m in pat.finditer(s):
                lo, hi = max(0, m.start() - 30), min(len(s), m.end() + 30)
                out.append(Violation(field=field, category=cat, word=word,
                                     context=s[lo:hi].replace("\n", " "),
                                     remedy=SANCTIONED_PHRASING[cat]))
    return out


def assert_clean(fields: Dict[str, object],
                 categories: Sequence[str] = ("causal", "temporal")) -> None:
    """Raise `LanguageDisciplineError` if any field carries an overclaim.

    `fields` is `{field_name: text}` so the error names the field, not just
    the offending word.
    """
    bad: List[Violation] = []
    for name, text in fields.items():
        bad.extend(check(text, field=name, categories=categories))
    if bad:
        raise LanguageDisciplineError(
            "claim-language discipline refused %d phrase(s) before they "
            "reached the append-only chain:\n  - %s"
            % (len(bad), "\n  - ".join(str(v) for v in bad))
        )
