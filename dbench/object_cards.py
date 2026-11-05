"""Deterministic, evidence-bound object cards for MORPHEUS discoveries.

An object card is deliberately not a prose generator.  It is the portable
record that connects a frozen scientific candidate to all of the evidence that
can characterize it: its complete statement, lower-order alternatives,
response surface, spatial/temporal scale, ablations, stability, uncertainty,
known analogues, and the next discriminating experiment.  Missing evidence is
represented explicitly as ``None``/``[]`` rather than filled with a plausible
story.  This is essential for prospective discovery, where a candidate may be
valid before protein, clinical, or experimental evidence exists.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .metrics import Claim, object_fidelity

SCHEMA_VERSION = "morpheus-object-card/v2"


def _get_language_module():
    """`morpheus.protocol.language` is a leaf module with no real
    dependencies of its own, but importing it the normal way runs
    `morpheus/protocol/__init__.py`, which pulls in `morpheus_compiler` and
    `morpheus_registry` -- separate packages this card-builder has no other
    reason to need and that are not guaranteed to be installed in every
    environment this checker runs in (e.g. a lean receipt/card-only box).
    Try the normal import first; only on failure, load the file directly by
    path, bypassing the package `__init__.py` entirely, and cache the
    result so this only happens once per process."""
    global _LANGUAGE_MODULE
    if _LANGUAGE_MODULE is not None:
        return _LANGUAGE_MODULE
    try:
        from morpheus.protocol import language as _lang
        _LANGUAGE_MODULE = _lang
        return _lang
    except Exception:
        import importlib.util
        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, os.pardir, "morpheus",
                                             "protocol", "language.py"))
        modname = "_morpheus_protocol_language_standalone"
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        # Register before exec: `language.py` decorates a dataclass, and
        # `dataclasses` resolves string annotations via
        # `sys.modules[cls.__module__]`, which does not exist yet unless
        # this module is in `sys.modules` before its body runs.
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        _LANGUAGE_MODULE = mod
        return mod


_LANGUAGE_MODULE = None

#: D4 replication-level ladder (`.claude/discoadd.md` "Replication needs
#: explicit evidence levels"): 0 internal holdout, 1 new samples/donors under
#: the same acquisition setting (HEST held-out fold), 2 an independent
#: cohort/site (CPTAC), 3 an orthogonal modality (protein against RNA), 4 a
#: prospective perturbation/intervention.
REPLICATION_LEVELS = (0, 1, 2, 3, 4)

#: D6 causal-language evidence tiers, weakest to strongest.
CAUSAL_TIERS = ("association", "invariant_association", "directional",
               "perturbational", "mechanistic_triangulation", "causal")

#: E2 / discoadd "Scientific importance should be distinct from novelty":
#: the per-dimension importance table `score_importance` reports. Kept in
#: sync with the analogous `SCALE_FIELDS` naming convention in
#: `morpheus_encoder.discovery_episode`.
IMPORTANCE_DIMENSIONS = (
    "explanatory_gain", "robustness", "effect_magnitude",
    "connection_to_important_quantities", "mechanistic_discrimination",
    "breadth_specificity", "reproducibility", "potential_consequences",
)

#: E2: kept in sync with `morpheus_encoder.discovery_episode.SCALE_FIELDS`.
SCALE_FIELDS = ("entity_type", "spatial_support", "temporal_support",
               "modality", "biological_scale", "containment")

#: Tier-matched claim templates for `evidence_statement`'s auto-generated
#: wording. Only "causal" itself is licensed to say "causes"; every other
#: template is written to already pass `morpheus.protocol.language.check`.
_TIER_STATEMENT_TEMPLATES = {
    "association": "{operands} is associated with {target}.",
    "invariant_association": "{operands} is associated with {target}, stably across the tested environments.",
    "directional": "{operands} shows a directional/temporal relationship with {target}.",
    "perturbational": "Perturbing {operands} changes {target} in held-out perturbational evidence.",
    "mechanistic_triangulation": "{operands} is mechanistically implicated in {target}, triangulated across evidence.",
    "causal": "{operands} causes {target}.",
}


def _plain(value: Any) -> Any:
    """Convert ordinary scientific Python values into JSON-ready primitives."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    if isinstance(value, list):
        return [_plain(x) for x in value]
    # numpy scalar support without importing numpy into the public contract.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            pass
    return value


def _claim_statement(claim: Claim) -> dict:
    return {
        "id": str(claim.id),
        "operator": str(claim.operator),
        "operands": sorted(map(str, claim.operands)),
        "target": str(claim.target),
        "context": _plain(claim.context),
        "lag": None if claim.lag is None else int(claim.lag),
        "parameters": _plain(claim.params),
        "proposal_effect": _plain(claim.effect),
        "proposal_p_value": _plain(claim.p_value),
        "replication": _plain(claim.replication),
        "complexity": int(claim.complexity),
        "reported_mode": claim.discovery_mode,
    }


def _resolve_causal_tier(causal_tier: str | None, statement: str | None):
    """Validate `causal_tier` and auto-downgrade to `association` when
    `morpheus.protocol.language.check` flags a causal word in `statement`
    (D6: "auto-downgrade to association when language.check flags a causal
    word in the statement"). Returns `(resolved_tier, downgraded, violations)`.
    """
    tier = causal_tier if causal_tier is not None else "association"
    if tier not in CAUSAL_TIERS:
        raise ValueError("causal_tier must be one of %r, got %r"
                         % (CAUSAL_TIERS, causal_tier))
    if statement is None or tier == "association":
        return tier, False, []
    language = _get_language_module()
    violations = language.check(statement, field="statement", categories=("causal",))
    if violations:
        return "association", True, [str(v) for v in violations]
    return tier, False, []


def build_object_card(
    claim: Claim,
    *,
    truth=None,
    lower_order_effects: Mapping[str, Any] | None = None,
    unique_higher_order_effect: Mapping[str, Any] | None = None,
    response_surface: Mapping[str, Any] | None = None,
    constituent_ablations: Mapping[str, Any] | None = None,
    nuisance_sensitivity: Mapping[str, Any] | None = None,
    stability: Mapping[str, Any] | None = None,
    nearest_simpler_explanation: Mapping[str, Any] | None = None,
    nearest_known_analogues: list[Mapping[str, Any]] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
    suggested_discriminating_experiment: Mapping[str, Any] | None = None,
    external_validation: Mapping[str, Any] | None = None,
    translational_evidence: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    replication_level: int | None = None,
    structured_nulls: list[Mapping[str, Any]] | None = None,
    novelty: Mapping[str, Any] | None = None,
    causal_tier: str = "association",
    statement: str | None = None,
    failure_conditions: list[Any] | None = None,
    compression: Mapping[str, Any] | None = None,
    scale: Mapping[str, Any] | None = None,
    human_involvement: list[Mapping[str, Any]] | None = None,
    explanation: Mapping[str, Any] | None = None,
) -> dict:
    """Create one complete, non-speculative object-card record (schema v2).

    ``truth`` is evaluation-only and optional: real prospective objects will
    not have it.  When supplied (Tier-I/II), the card records object fidelity;
    it never changes the candidate statement or any evidence field.  Every
    optional characterization block has an explicit empty value, making a
    card auditable without silently implying that an analysis was performed.

    v2 additions (D6, every v1 key kept unchanged):

    ``replication_level``   int 0-4 on the D4 ladder, or ``None`` if not yet
                             assessed.
    ``structured_nulls``    ``[{"kind": ..., "p": ..., "n": ...}, ...]``.
    ``novelty``              the D5 literature-novelty record (canonical
                             claim + compositional label), or ``{}``.
    ``causal_tier``         one of `CAUSAL_TIERS`; auto-downgraded to
                             ``"association"`` when ``statement`` carries a
                             causal word (`morpheus.protocol.language.check`)
                             -- the downgrade and its reason are recorded on
                             the card, never silently applied.
    ``failure_conditions``  what would falsify or invalidate this candidate.

    E2 additions (every v2 key kept unchanged):

    ``compression``         `morpheus_encoder.general_discovery.compress_relation`
                             output (or ``{}`` if compression was not
                             attempted) -- interpretability AFTER evidence.
    ``scale``                the discoadd scale-graph record: entity type,
                             spatial/temporal support, modality, biological
                             scale, containment mappings (``SCALE_FIELDS``),
                             or ``{}`` if not supplied.
    ``human_involvement``   ``[{"who":..., "what":..., "why":..., "when":...},
                             ...]`` logged interventions; ``[]`` means none.

    Explanation addition (every earlier key kept unchanged):

    ``explanation``         `morpheus_encoder.general_discovery.decompose_explanation`
                             output -- the Level A-D classification of how a
                             validated relation explains its target, with
                             E/F only from evidence flags; ``{}`` if the
                             decomposition was not run. It never changes an
                             acceptance decision.
    """
    if not isinstance(claim, Claim):
        raise TypeError("claim must be a dbench.metrics.Claim")
    fidelity = None
    if truth is not None:
        f = object_fidelity(claim, truth)
        fidelity = _plain(f)
    spatial_radius = claim.params.get("radius") if isinstance(claim.params, Mapping) else None
    if replication_level is not None and int(replication_level) not in REPLICATION_LEVELS:
        raise ValueError("replication_level must be one of %r, got %r"
                         % (REPLICATION_LEVELS, replication_level))
    resolved_tier, downgraded, tier_violations = _resolve_causal_tier(causal_tier, statement)
    card = {
        "schema_version": SCHEMA_VERSION,
        "object": _claim_statement(claim),
        "evaluation_only": {"object_fidelity": fidelity},
        "physical_scale": {
            "spatial_radius": _plain(spatial_radius),
            "temporal_lag": None if claim.lag is None else int(claim.lag),
            "context": _plain(claim.context),
        },
        "lower_order_effects": _plain(lower_order_effects) if lower_order_effects is not None else {},
        "unique_higher_order_effect": _plain(unique_higher_order_effect) if unique_higher_order_effect is not None else {},
        "functional_response_surface": _plain(response_surface) if response_surface is not None else {},
        "constituent_ablations": _plain(constituent_ablations) if constituent_ablations is not None else {},
        "nuisance_sensitivity": _plain(nuisance_sensitivity) if nuisance_sensitivity is not None else {},
        "stability": _plain(stability) if stability is not None else {},
        "nearest_simpler_explanation": _plain(nearest_simpler_explanation) if nearest_simpler_explanation is not None else {},
        "nearest_known_analogues": _plain(nearest_known_analogues) if nearest_known_analogues is not None else [],
        "uncertainty": _plain(uncertainty) if uncertainty is not None else {},
        "suggested_discriminating_experiment": _plain(suggested_discriminating_experiment) if suggested_discriminating_experiment is not None else {},
        "external_validation": _plain(external_validation) if external_validation is not None else {},
        "translational_evidence": _plain(translational_evidence) if translational_evidence is not None else {},
        "provenance": _plain(provenance) if provenance is not None else {},
        "replication_level": None if replication_level is None else int(replication_level),
        "structured_nulls": _plain(structured_nulls) if structured_nulls is not None else [],
        "novelty": _plain(novelty) if novelty is not None else {},
        "causal_tier": resolved_tier,
        "causal_tier_requested": str(causal_tier),
        "causal_tier_downgraded": downgraded,
        "causal_tier_downgrade_reason": tier_violations,
        "failure_conditions": _plain(failure_conditions) if failure_conditions is not None else [],
        "compression": _plain(compression) if compression is not None else {},
        "scale": _plain(scale) if scale is not None else {},
        "human_involvement": _plain(human_involvement) if human_involvement is not None else [],
        "explanation": _plain(explanation) if explanation is not None else {},
    }
    return card


# --------------------------------------------------------------------------
# E2: importance ranking (distinct from novelty), read only AFTER validity
# and novelty are already established.
# --------------------------------------------------------------------------

def score_importance(card_inputs: Mapping[str, Any]) -> dict:
    """Build the discoadd "scientific importance" per-dimension table.

    ``card_inputs`` is a flat mapping the caller assembles once validity
    and literature novelty are already known (discoadd: "This ranking
    occurs only after statistical validity and literature novelty are
    assessed"). Every dimension is reported separately as ``{"assessed":
    bool, "value": ..., "evidence": "<what this dimension means>"}`` --
    this function NEVER collapses the table into one opaque scalar; there
    is no "importance_score" key anywhere in the return. A dimension whose
    input was not supplied is reported as ``"assessed": False`` rather than
    guessed.

    Recognised keys on ``card_inputs`` (all optional):

    ``explanatory_gain``       float, e.g. the compression/flexible-relation
                                held-out incremental gain over the best
                                simpler explanation.
    ``robustness``             mapping ``{environment_or_fold: gain}``.
    ``effect_magnitude``       float, the candidate's own effect size.
    ``important_quantity_tags`` caller-supplied ``list[str]`` naming which
                                scientifically important quantities this
                                candidate connects to; ``[]`` is a valid,
                                explicit "none named".
    ``competing_explanations``  caller-supplied description of which rival
                                mechanisms this candidate discriminates
                                against (mechanistic discrimination).
    ``regime_description``     caller-supplied description of how broad or
                                narrow the validated regime/context is.
    ``replication_level``      int 0-4 (D4 ladder); reproducibility is read
                                directly off this, never re-derived.
    ``potential_consequences``  caller-supplied ``list[str]`` naming
                                downstream consequences if the candidate is
                                true.
    """
    robustness = card_inputs.get("robustness")
    robustness_summary = None
    if isinstance(robustness, Mapping) and robustness:
        values = [float(v) for v in robustness.values()]
        robustness_summary = {"n_environments": len(values), "min": min(values),
                              "max": max(values), "mean": sum(values) / len(values)}
    replication_level = card_inputs.get("replication_level")
    table = {
        "explanatory_gain": {
            "assessed": card_inputs.get("explanatory_gain") is not None,
            "value": _plain(card_inputs.get("explanatory_gain")),
            "evidence": "held-out incremental gain over the best simpler explanation",
        },
        "robustness": {
            "assessed": robustness_summary is not None,
            "value": robustness_summary,
            "evidence": "gain summarised across environments/folds",
        },
        "effect_magnitude": {
            "assessed": card_inputs.get("effect_magnitude") is not None,
            "value": _plain(card_inputs.get("effect_magnitude")),
            "evidence": "candidate's own effect size",
        },
        "connection_to_important_quantities": {
            "assessed": card_inputs.get("important_quantity_tags") is not None,
            "value": list(card_inputs.get("important_quantity_tags") or []),
            "evidence": "caller-supplied tags naming connected scientific quantities",
        },
        "mechanistic_discrimination": {
            "assessed": card_inputs.get("competing_explanations") is not None,
            "value": _plain(card_inputs.get("competing_explanations") or []),
            "evidence": "competing explanations this candidate discriminates against",
        },
        "breadth_specificity": {
            "assessed": card_inputs.get("regime_description") is not None,
            "value": _plain(card_inputs.get("regime_description")),
            "evidence": "how broad or narrow the validated regime/context is",
        },
        "reproducibility": {
            "assessed": replication_level is not None,
            "value": None if replication_level is None else int(replication_level),
            "evidence": "D4 replication-level ladder (0-4)",
        },
        "potential_consequences": {
            "assessed": card_inputs.get("potential_consequences") is not None,
            "value": list(card_inputs.get("potential_consequences") or []),
            "evidence": "caller-supplied downstream consequences if the candidate is true",
        },
    }
    assert set(table) == set(IMPORTANCE_DIMENSIONS)
    return table


def rank_by_importance(cards: list[Mapping[str, Any]]) -> list[dict]:
    """Order validated, novel candidates for expensive characterization.

    Refuses any card missing a `novelty` assessment (falsy/empty counts as
    missing -- `build_object_card`'s own default is `{}`) or a
    `replication_level`: importance is assessed only AFTER validity and
    novelty, never as a substitute for them (discoadd: "Do not tune the
    discovery algorithm directly to an opaque importance score; use it to
    prioritize expensive characterization").

    Returns ``[{"card": card, "rank_key": {...}}, ...]`` ordered highest-
    priority first. ``rank_key`` is an explicit, inspectable multi-key sort
    tuple (dimensions assessed, replication level, explanatory gain) --
    there is deliberately no single opaque priority scalar here either.
    """
    cards = list(cards)
    for card in cards:
        if not card.get("novelty"):
            raise ValueError(
                "rank_by_importance refuses a card with no `novelty` "
                "assessment -- importance ranking requires validity and "
                "literature novelty to be assessed first")
        if card.get("replication_level") is None:
            raise ValueError(
                "rank_by_importance refuses a card with no "
                "`replication_level` -- importance ranking requires "
                "validity and novelty to be assessed first")
    scored = []
    for card in cards:
        table = card.get("importance") or {}
        n_assessed = sum(1 for d in IMPORTANCE_DIMENSIONS if (table.get(d) or {}).get("assessed"))
        gain = (table.get("explanatory_gain") or {}).get("value")
        gain = float(gain) if isinstance(gain, (int, float)) else 0.0
        rank_key = {"n_dimensions_assessed": n_assessed,
                   "replication_level": int(card["replication_level"]),
                   "explanatory_gain": gain}
        scored.append({"card": card, "rank_key": rank_key})
    scored.sort(key=lambda item: (item["rank_key"]["n_dimensions_assessed"],
                                  item["rank_key"]["replication_level"],
                                  item["rank_key"]["explanatory_gain"]),
               reverse=True)
    return scored


def evidence_statement(card: Mapping[str, Any], statement: str | None = None) -> str:
    """Render the card's claim in language matched to its recorded
    `causal_tier` (discoadd: "Causal and mechanistic language must have
    evidence tiers").

    If `statement` is omitted, a plain templated statement is generated
    from the card's own `object` block (operands/target) using a template
    already written to match that tier. Either way, the rendered text is
    checked with `morpheus.protocol.language.check` for causal wording
    UNLESS the card's tier is `mechanistic_triangulation` or `causal` (the
    only tiers licensed to use it); a mismatch raises
    `LanguageDisciplineError` rather than silently emitting overclaiming
    language the card's own recorded evidence does not support.
    """
    tier = card.get("causal_tier", "association")
    if tier not in CAUSAL_TIERS:
        raise ValueError("card causal_tier must be one of %r, got %r" % (CAUSAL_TIERS, tier))
    obj = card.get("object") or {}
    operands = ", ".join(str(o) for o in (obj.get("operands") or ()))
    target = obj.get("target")
    if statement is None:
        template = _TIER_STATEMENT_TEMPLATES[tier]
        statement = template.format(operands=operands or "the candidate", target=target)
    language = _get_language_module()
    categories = () if tier in ("mechanistic_triangulation", "causal") else ("causal",)
    violations = language.check(statement, field="evidence_statement", categories=categories)
    if violations:
        raise language.LanguageDisciplineError(
            "evidence_statement wording exceeds the card's causal_tier %r: %s"
            % (tier, "; ".join(str(v) for v in violations)))
    return statement


# --------------------------------------------------------------------------
# card_from_receipt: build a card directly from a general-discovery receipt
# row (D6), optionally joined with the ledger event recorded for the same
# candidate (whatever `dbench.ledger.record_from_receipt` wrote).
# --------------------------------------------------------------------------

#: `morpheus_encoder.general_discovery.Candidate.feature_form` (v1 receipts)
#: to the D2/D3 flexible-relation operator names (v2 receipts may already
#: carry `operator` directly, in which case this mapping is not consulted).
_FEATURE_FORM_TO_OPERATOR = {
    "product": "flexible_product",
    "linear_subspace": "flexible_subspace",
}


def _operator_from_row(row: Mapping[str, Any]) -> str:
    op = row.get("operator")
    if op in ("flexible_product", "flexible_subspace"):
        return str(op)
    return _FEATURE_FORM_TO_OPERATOR.get(row.get("feature_form"), "flexible_subspace")


def _infer_replication_level(row: Mapping[str, Any]) -> int:
    """D4: a receipt naming a held-out HEST fold is L1 evidence (new samples
    under the same acquisition setting) until something stronger is recorded
    on top of it. A row with no fold/replication signal at all is L0 (an
    internal holdout within one dataset)."""
    if row.get("fold") is not None or row.get("held_out_fold") is not None:
        return 1
    episode_identity = row.get("episode_identity")
    if isinstance(episode_identity, Mapping) and episode_identity.get("fold") is not None:
        return 1
    if row.get("fold_gains") is not None or row.get("replication_gain") is not None:
        return 1
    return 0


def _structured_nulls_from_row(row: Mapping[str, Any]) -> list[dict]:
    nulls = row.get("nulls")
    if isinstance(nulls, Mapping) and nulls:
        out = []
        for kind, v in nulls.items():
            if isinstance(v, Mapping):
                out.append({"kind": str(kind), "p": v.get("p"), "n": v.get("n")})
            else:
                out.append({"kind": str(kind), "p": v, "n": None})
        return out
    if row.get("null_p_value") is not None:
        return [{"kind": "response_permutation", "p": row.get("null_p_value"),
                "n": len(row.get("null_gains") or ())}]
    return []


def _replication_from_row(row: Mapping[str, Any]) -> dict:
    fold_gains = row.get("fold_gains")
    if isinstance(fold_gains, Mapping):
        return {str(k): {"gain": v} for k, v in fold_gains.items()}
    if isinstance(fold_gains, (list, tuple)):
        return {"fold_%d" % i: {"gain": g} for i, g in enumerate(fold_gains)}
    if row.get("replication_gain") is not None:
        return {"replication": {"gain": row["replication_gain"]}}
    return {}


def _scale_from_row(row: Mapping[str, Any], episode_identity: Any = None) -> dict:
    """Read the discoadd scale-graph record (`SCALE_FIELDS`) off whichever
    of `row["scale"]`, `row["context"]["scale"]`, or an `episode_identity`
    mapping's own `"scale"` key actually carries it. Missing entirely ->
    `{}` (not a guess); a partially-populated source keeps exactly the
    fields it names, each other field left absent rather than fabricated.
    """
    candidates = (row.get("scale"), (row.get("context") or {}).get("scale"),
                 (episode_identity or {}).get("scale") if isinstance(episode_identity, Mapping) else None)
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            return {f: candidate[f] for f in SCALE_FIELDS if f in candidate}
    return {}


def card_from_receipt(
    row: Mapping[str, Any],
    ledger_rec: Mapping[str, Any] | None = None,
    *,
    target: str = "Y",
    statement: str | None = None,
    causal_tier: str = "association",
    replication_level: int | None = None,
    **extra: Any,
) -> dict:
    """Build a v2 object card straight from one general-discovery receipt row
    (`morpheus_encoder.general_discovery` / the runner's JSON receipt),
    optionally joined with the ledger event recorded for the same candidate.

    `row` is read defensively because the receipt schema is mid-migration
    (plan `calm-mixing-candy.md`, D1-D6): a v1 row carries integer
    ``variables`` and ``feature_form in {"product","linear_subspace"}``; a
    v2 row is expected to carry ``operator in {"flexible_product",
    "flexible_subspace"}`` and gene symbols directly under ``gene_symbols``.
    Both are accepted, and ``gene_symbols`` is preferred over ``variables``
    when both are present.
    """
    row = dict(row or {})
    ledger_rec = dict(ledger_rec or {}) if ledger_rec is not None else {}
    operator = _operator_from_row(row)
    operands_raw = row.get("gene_symbols") or row.get("variables") or row.get("operands") or ()
    operands = tuple(str(o) for o in operands_raw)
    cid = str(row.get("candidate_hash") or row.get("candidate_id")
              or ledger_rec.get("candidate_id") or "candidate")
    context = dict(row.get("context") or {})
    episode_identity = row.get("episode_identity")
    if episode_identity:
        context.setdefault("episode_identity", episode_identity)
    params = dict(row.get("parameters") or {})
    if row.get("baseline_width") is not None:
        params.setdefault("baseline_width", row["baseline_width"])
    effect = row.get("proposal_gain", row.get("effect"))
    p_value = row.get("null_p_value", row.get("p_value"))

    claim = Claim(
        id=cid, operator=operator, operands=operands, target=str(target),
        context=context, lag=row.get("lag"),
        effect=float(effect) if effect is not None else float("nan"),
        p_value=float(p_value) if p_value is not None else float("nan"),
        replication=_replication_from_row(row),
        complexity=int(row.get("complexity", max(len(operands), 1))),
        discovery_mode=row.get("proposal_family"),
        params=params,
    )

    if replication_level is None:
        replication_level = _infer_replication_level(row)

    structured_nulls = extra.pop("structured_nulls", None)
    if structured_nulls is None:
        structured_nulls = _structured_nulls_from_row(row)

    novelty = extra.pop("novelty", None)
    if novelty is None:
        evidence = ledger_rec.get("evidence")
        novelty = evidence.get("novelty") if isinstance(evidence, Mapping) else None

    failure_conditions = extra.pop("failure_conditions", None)
    if failure_conditions is None:
        failure_conditions = ([{"condition": "stop_reason", "detail": row["stop_reason"]}]
                              if row.get("stop_reason") else [])

    provenance = extra.pop("provenance", None)
    if provenance is None:
        provenance = {
            "candidate_hash": row.get("candidate_hash"),
            "episode_identity": episode_identity,
            "representation_checkpoint": row.get("representation_checkpoint"),
            "proposal_mechanism": row.get("proposal_family"),
            "search_budget": row.get("search_budget"),
            "ledger_stage": ledger_rec.get("stage"),
        }

    scale = extra.pop("scale", None)
    if scale is None:
        scale = _scale_from_row(row, episode_identity)

    # The explanation decomposition travels either as an explicit keyword or
    # on the receipt row itself (a classification block, never a gate).
    explanation = extra.pop("explanation", None)
    if explanation is None and isinstance(row.get("explanation"), Mapping):
        explanation = row["explanation"]

    return build_object_card(
        claim,
        replication_level=replication_level,
        structured_nulls=structured_nulls,
        novelty=novelty,
        causal_tier=causal_tier,
        statement=statement,
        failure_conditions=failure_conditions,
        provenance=provenance,
        scale=scale,
        explanation=explanation,
        **extra,
    )

