"""Machine-checkable MORPHEUS Unified Biological Discovery criteria v1.0."""
from __future__ import annotations

from typing import Any

DISCOVERY_TYPES = {
    "D1_molecular_state", "D2_morphology_first", "D3_morpho_molecular_correspondence",
    "D4_spatial_multiscale_dependency", "D5_context_dependent_modulation",
    "D6_nonadditive_interaction", "D7_irreducible_higher_order_interaction",
    "D8_conserved_biological_organization",
}
NOVELTY = {
    "N0_known_biology_recovered", "N1_context_extension", "N2_spatial_extension",
    "N3_compositional_novelty", "N4_new_molecular_state_or_association",
    "N5_new_functional_mechanism",
}
TIERS = {"Candidate", "Internally supported", "Externally confirmed", "Functionally supported"}
UTILITY = {"Biological", "Assay-translatable", "Clinically associated", "Clinically predictive", "Clinically actionable"}
STATUS = {"pass", "fail", "not_assessed", "not_applicable", "inherited"}
UNIVERSAL = ("frozen_definition", "patient_disjoint_evaluation", "statistical_validity",
             "confounder_robustness", "patient_level_reproducibility", "detection_power")
TYPE_REQUIREMENTS = {
    "D1_molecular_state": ("stable_structure", "known_axis_distinction", "independent_molecular_validation"),
    "D2_morphology_first": ("frozen_morphology", "heldout_molecular_correspondence", "imaging_composition_controls"),
    "D3_morpho_molecular_correspondence": ("heldout_cross_modal_association", "molecular_tissue_nulls", "abundance_controls"),
    "D4_spatial_multiscale_dependency": ("claim_matched_spatial_ablation", "spatial_calibrated_null", "external_replication"),
    "D5_context_dependent_modulation": ("effect_modification_over_additive", "compartment_matched_controls", "replicated_conditional_effect"),
    "D6_nonadditive_interaction": ("matched_additive_decomposition", "calibrated_uncertainty_null"),
    "D7_irreducible_higher_order_interaction": ("complete_lower_order_decomposition", "corrected_inference", "external_replication"),
    "D8_conserved_biological_organization": ("frozen_no_refit_transfer", "preserved_direction_interpretation", "matched_transfer_null"),
}


def _status(value: Any) -> str:
    return value if isinstance(value, str) and value in STATUS else "not_assessed"


def _evidenced(value: Any) -> bool:
    """``inherited`` is allowed only for a transparently mapped legacy receipt."""
    return _status(value) in {"pass", "inherited"}


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return errors; deliberately accepts legacy records with not_assessed fields."""
    errors: list[str] = []
    if not record.get("object_id"):
        errors.append("object_id is required")
    types = set(record.get("discovery_types") or [])
    if not types or not types <= DISCOVERY_TYPES:
        errors.append("discovery_types must be a nonempty controlled-vocabulary subset")
    if record.get("primary_novelty") not in NOVELTY:
        errors.append("primary_novelty is invalid")
    if record.get("evidence_tier") not in TIERS:
        errors.append("evidence_tier is invalid")
    if not set(record.get("utility_labels") or []) <= UTILITY:
        errors.append("utility_labels contains an invalid label")
    assessments = record.get("assessments") or {}
    for key, value in assessments.items():
        if _status(value) != value:
            errors.append(f"assessment {key!r} has invalid status")
    if record.get("primary_novelty") == "N4_new_molecular_state_or_association" and record.get("novelty_adjudication") not in {"provisional", "adjudicated"}:
        errors.append("N4 must state whether targeted adjudication is provisional or complete")
    if record.get("primary_novelty") == "N5_new_functional_mechanism" and _status(assessments.get("causal_functional_evidence")) != "pass":
        errors.append("N5 requires causal_functional_evidence=pass")
    if "D7_irreducible_higher_order_interaction" in types:
        for field in TYPE_REQUIREMENTS["D7_irreducible_higher_order_interaction"]:
            if _status(assessments.get(field)) != "pass":
                errors.append(f"D7 cannot be claimed without {field}=pass")
    if "Clinically actionable" in set(record.get("utility_labels") or []) and _status(assessments.get("clinical_decision_benefit")) != "pass":
        errors.append("Clinically actionable requires clinical_decision_benefit=pass")
    utility_requirements = {
        "Biological": "independent_phenotype_information",
        "Assay-translatable": "external_assay_fidelity",
        "Clinically associated": "independent_outcome_association",
        "Clinically predictive": "validated_incremental_outcome_prediction",
    }
    for label, requirement in utility_requirements.items():
        if label in set(record.get("utility_labels") or []) and not _evidenced(assessments.get(requirement)):
            errors.append(f"{label} requires {requirement}=pass")
    if record.get("evidence_tier") == "Externally confirmed" and not _evidenced(assessments.get("external_replication")):
        errors.append("Externally confirmed requires external_replication=pass")
    if record.get("highlighted"):
        for field in ("specific_novelty_claim", "known_biology_alternative", "decisive_test"):
            if not record.get(field) or record[field] == "not_assessed":
                errors.append(f"highlighted record requires {field}")
        for field in UNIVERSAL:
            if not _evidenced(assessments.get(field)):
                errors.append(f"highlighted record requires {field}=pass")
    return errors


def map_legacy_novelty(label: str | None) -> tuple[str, str]:
    mapping = {
        "No novelty identified": ("N0_known_biology_recovered", "adjudicated"),
        "Context extension": ("N1_context_extension", "adjudicated"),
        "Spatial extension": ("N2_spatial_extension", "adjudicated"),
        "Compositional novelty": ("N3_compositional_novelty", "adjudicated"),
        "New association": ("N4_new_molecular_state_or_association", "provisional"),
    }
    return mapping.get(label or "", ("N0_known_biology_recovered", "not_assessed"))


def evidence_tier(source: dict[str, Any]) -> str:
    """Map existing evidence without inventing an unperformed assessment."""
    external = (source.get("external_replication") or {}).get("pooled")
    if external:
        return "Externally confirmed"
    if source.get("tier") in {"Confirmed", "Conserved"}:
        return "Internally supported"
    return "Candidate"


def legacy_record(source: dict[str, Any]) -> dict[str, Any]:
    novelty, novelty_status = map_legacy_novelty(source.get("novelty_label"))
    types = ["D4_spatial_multiscale_dependency"]
    if source.get("higher_order"):
        # Legacy higher_order is a candidate label, never a D7 claim unless
        # a complete lower-order receipt is attached.
        types.append("D6_nonadditive_interaction")
    if source.get("conserved"):
        types.append("D8_conserved_biological_organization")
    tier = evidence_tier(source)
    assessments = {field: "inherited" if field in {"frozen_definition", "patient_disjoint_evaluation"} else "not_assessed" for field in UNIVERSAL}
    assessments.update({field: "not_assessed" for typ in types for field in TYPE_REQUIREMENTS[typ]})
    if (source.get("external_replication") or {}).get("pooled"):
        assessments["external_replication"] = "inherited"
    return {
        "object_id": source["relation_id"], "source_relation_id": source["relation_id"],
        "discovery_types": types, "primary_novelty": novelty,
        "novelty_adjudication": novelty_status, "evidence_tier": tier,
        "utility_labels": [], "assessments": assessments,
        "known_biology_alternative": "not_assessed", "specific_novelty_claim": "not_assessed",
        "decisive_test": "inherited legacy receipt; claim-specific v1.0 tests not retrospectively imputed",
        "legacy_mapping": True, "highlighted": False,
    }
