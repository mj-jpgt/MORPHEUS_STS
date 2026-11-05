"""Prespecification locks and the state card (downstream spec Sections 0 and 10.1).

A lock is a write-once JSON file holding the inputs, the declared analysis, the hashes of the frozen
objects it depends on and a UTC timestamp. It is written BEFORE the analysis opens its data; the
analysis then calls :func:`require_lock` (which re-verifies the lock's own hash) and, when it has
finished, :func:`write_record` stores the result with its hash. The next step's lock depends on that
record hash, so the chain of locks fixes the order of Section 0.

Stage order (Section 0): registry_freeze -> internal_validation -> external_replication ->
characterisation -> literature -> prioritisation -> {orthogonal, therapeutic, distillation, clinical}
-> state_card. A stage opened before its prerequisites have records raises :class:`OrderViolation`
unless the caller passes ``allow_violation=True``, in which case the violation is written into the
lock and the state card moves the finding to the supplement.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_LOCK = "morpheus_downstream_lock/v1"
SCHEMA_RECORD = "morpheus_downstream_record/v1"
SCHEMA_CARD = "morpheus_state_card/v1"

STAGES = ("registry_freeze", "internal_validation", "external_replication", "characterisation",
          "literature", "prioritisation", "orthogonal", "therapeutic", "distillation", "clinical",
          "state_card")
PREREQ = {
    "registry_freeze": (),
    "internal_validation": ("registry_freeze",),
    "external_replication": ("internal_validation",),
    "characterisation": ("external_replication",),
    "literature": ("characterisation",),
    "prioritisation": ("literature",),
    "orthogonal": ("prioritisation",),
    "therapeutic": ("prioritisation",),
    "distillation": ("prioritisation",),
    "clinical": ("prioritisation",),
    "state_card": ("prioritisation",),
}


class OrderViolation(RuntimeError):
    """A downstream step was opened before the records it depends on exist."""


class LockError(RuntimeError):
    """Missing, altered or duplicate lock."""


def canonical_hash(payload: Any) -> str:
    """Identical to ``dbench.prospective_campaign.canonical_hash``."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def software_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "statsmodels", "squidpy", "liana", "torch"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", "?")
    return out


def _stage_dir(root, finding_id: str) -> Path:
    return Path(root) / _safe(finding_id)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def lock_path(root, finding_id: str, stage: str) -> Path:
    return _stage_dir(root, finding_id) / "locks" / f"{STAGES.index(stage):02d}_{stage}.lock.json"


def record_path(root, finding_id: str, stage: str) -> Path:
    return _stage_dir(root, finding_id) / "records" / f"{STAGES.index(stage):02d}_{stage}.record.json"


def _write_once(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LockError(f"{path} exists: locks and records are write-once (a change is a new finding id)")
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _check_self_hash(doc: dict, key: str) -> None:
    body = {k: v for k, v in doc.items() if k != key}
    if canonical_hash(body) != doc.get(key):
        raise LockError(f"{doc.get('stage')} {key} does not match its contents (altered after writing)")


def read_record(root, finding_id: str, stage: str) -> dict | None:
    p = record_path(root, finding_id, stage)
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    _check_self_hash(doc, "record_hash")
    return doc


def write_lock(root, finding_id: str, stage: str, *, inputs: Mapping[str, Any],
               declared_analysis: Mapping[str, Any], depends_on: Mapping[str, str] | None = None,
               allow_violation: bool = False) -> dict:
    """Write the stage's lock before the analysis opens any data. Prerequisite records are
    added to ``depends_on`` automatically (their record hashes)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {STAGES}")
    deps = dict(depends_on or {})
    missing = []
    for pre in PREREQ[stage]:
        rec = read_record(root, finding_id, pre)
        if rec is None:
            missing.append(pre)
        else:
            deps[f"record:{pre}"] = rec["record_hash"]
    if missing and not allow_violation:
        raise OrderViolation(f"{finding_id}: stage {stage!r} opened before {missing} have records")
    doc = {"schema": SCHEMA_LOCK, "finding_id": finding_id, "stage": stage, "inputs": dict(inputs),
           "declared_analysis": dict(declared_analysis), "depends_on": deps, "created_utc": utc_now(),
           "software": software_versions(),
           "order_violation": ({"missing_prerequisites": missing,
                                "consequence": "finding moves to the supplement for this result"}
                               if missing else None)}
    doc["lock_hash"] = canonical_hash(doc)
    _write_once(lock_path(root, finding_id, stage), doc)
    return doc


def require_lock(root, finding_id: str, stage: str) -> dict:
    """Called by an analysis before it opens data: the lock must exist and be unaltered."""
    p = lock_path(root, finding_id, stage)
    if not p.exists():
        raise LockError(f"{finding_id}: no lock for stage {stage!r}; write it before opening data")
    doc = json.loads(p.read_text())
    _check_self_hash(doc, "lock_hash")
    return doc


def write_record(root, finding_id: str, stage: str, result: Mapping[str, Any]) -> dict:
    """Store the stage's result, chained to its lock."""
    lock = require_lock(root, finding_id, stage)
    doc = {"schema": SCHEMA_RECORD, "finding_id": finding_id, "stage": stage, "lock_hash": lock["lock_hash"],
           "lock_created_utc": lock["created_utc"], "created_utc": utc_now(), "result": dict(result)}
    if doc["created_utc"] < lock["created_utc"]:
        raise LockError("record timestamp precedes its lock")
    doc["record_hash"] = canonical_hash(doc)
    _write_once(record_path(root, finding_id, stage), doc)
    return doc


# ----------------------------------------------------------------------------- state card
CARD_SECTIONS = {
    "identity": ("finding_id", "cohort", "target", "candidates", "relation_params_hash", "registry_hash"),
    "validation": ("gain_nuis", "gain_known", "gain_decomp", "discovery_labels", "null_yield_at_rank"),
    "replication": ("per_cohort", "pooled", "direction_agreement", "q", "platform_transformation",
                    "rule_hash"),
    # The blinded pathologist read (1.7) was scratched by the owner on 2026-10-21: not required; a card is
    # complete without it, and a 'pathologist_read' entry, if present, is carried as an optional extra.
    "characterization": ("constituents", "enrichment", "localization", "scale_decomposition", "squidpy",
                         "liana", "centre_labels"),
    "novelty": ("query_log", "record_ids", "grade", "secondary_tags", "reviewer_agreement",
                "literature_lock_hash"),
    "orthogonal": ("cptac_rna_protein", "tcga_bulk_score"),
    "therapeutic": ("depmap", "open_targets", "depmap_release", "platform_release", "relevance_statement"),
    "distillation": ("per_model", "success", "site_decomposition"),     # 7a primary (protocol v02 item 7a)
    "clinical": ("lock", "hazard_ratios", "q", "incremental_discrimination", "known_programme_correlations"),
    "provenance": ("locks", "software", "partition_manifest_version"),
}
# Reported-never-gated fields of assay distillation protocol v02 (dbench/protocols/assay_distillation_v02.json):
# written into the distillation section when present; a card without them states 'not assessed'.
CARD_OPTIONAL = {"distillation": ("primary_7a", "diagnosis_7b", "bulk_anchored_7c")}
STAGE_OF_SECTION = {"replication": "external_replication", "characterization": "characterisation",
                    "novelty": "literature", "orthogonal": "orthogonal", "therapeutic": "therapeutic",
                    "distillation": "distillation", "clinical": "clinical", "validation": "internal_validation",
                    "identity": "registry_freeze"}


class StateCard:
    """One JSON document per finding, written incrementally; every section update records the
    lock it was produced under and the section hash. Missing required fields must be present with an
    explicit status (for example ``{"status": "not assessed", "reason": ...}``), never omitted."""

    def __init__(self, root, finding_id: str):
        self.root, self.finding_id = Path(root), finding_id
        self.path = _stage_dir(root, finding_id) / "state_card.json"
        if self.path.exists():
            self.doc = json.loads(self.path.read_text())
        else:
            self.doc = {"schema": SCHEMA_CARD, "finding_id": finding_id, "created_utc": utc_now(),
                        "sections": {}, "section_hashes": {},
                        "provenance": {"locks": [], "software": software_versions(),
                                       "partition_manifest_version": None},
                        "headline_eligible": True, "violations": []}

    def update(self, section: str, payload: Mapping[str, Any], *, stage: str | None = None) -> dict:
        if section not in CARD_SECTIONS or section == "provenance":
            raise ValueError(f"unknown section {section!r}")
        missing = [k for k in CARD_SECTIONS[section] if k not in payload]
        if missing:
            raise ValueError(f"section {section!r} is missing required fields {missing} "
                             "(use an explicit 'not assessed' status instead of omitting)")
        unknown_7x = [k for k in payload if k.startswith(("primary_7", "diagnosis_7", "bulk_anchored_7"))
                      and k not in CARD_OPTIONAL.get(section, ())]
        if unknown_7x:
            raise ValueError(f"unknown 7.x fields {unknown_7x} in section {section!r}")
        sd = self.doc.get("state_definitions")
        if sd and (section == "clinical" or (section == "distillation" and "per_model" in payload)):
            used = payload.get("state_definition_hash")
            if used == sd["within_patient"]["hash"]:
                raise ValueError(f"{section}: 7a/7c and Section 8 results must use the between-patient-inclusive "
                                 "state (b), not the within-patient state (a) (assay_distillation_v03)")
            if used != sd["between_patient_inclusive"]["hash"]:
                raise ValueError(f"{section}: payload must carry state_definition_hash of state (b)")
        stage = stage or STAGE_OF_SECTION[section]
        lock = require_lock(self.root, self.finding_id, stage)
        if lock.get("order_violation"):
            self.doc["headline_eligible"] = False
            self.doc["violations"].append({"stage": stage, **lock["order_violation"]})
        self.doc["sections"][section] = dict(payload)
        self.doc["section_hashes"][section] = canonical_hash(dict(payload))
        self.doc["provenance"]["locks"].append({"section": section, "stage": stage,
                                                "lock_hash": lock["lock_hash"],
                                                "lock_created_utc": lock["created_utc"],
                                                "written_utc": utc_now()})
        self.save()
        return self.doc

    def set_state_definitions(self, within_patient: dict, between_patient_inclusive: dict) -> None:
        """assay_distillation_v03: both state definitions frozen at the characterisation lock. Each dict carries
        'definition' and 'hash' (canonical hash of the definition and its frozen parameters)."""
        require_lock(self.root, self.finding_id, "characterisation")
        if self.doc.get("state_definitions"):
            raise LockError("state definitions are frozen once")
        self.doc["state_definitions"] = {"within_patient": dict(within_patient),
                                         "between_patient_inclusive": dict(between_patient_inclusive),
                                         "frozen_utc": utc_now()}
        self.save()

    def set_partition_manifest(self, version: str) -> None:
        self.doc["provenance"]["partition_manifest_version"] = version
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {k: v for k, v in self.doc.items() if k != "card_hash"}
        self.doc["card_hash"] = canonical_hash(body)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.doc, indent=1, sort_keys=True, default=str) + "\n")
        os.replace(tmp, self.path)

    def completeness(self) -> dict:
        return {s: (s in self.doc["sections"]) for s in CARD_SECTIONS if s != "provenance"}


def validate_card(doc: dict) -> dict:
    """Completeness of a written card against the current schema: per section, present and missing fields."""
    out = {}
    for sec, req in CARD_SECTIONS.items():
        if sec == "provenance":
            continue
        payload = doc.get("sections", {}).get(sec)
        out[sec] = {"present": payload is not None,
                    "missing_fields": [k for k in req if payload is None or k not in payload]}
    out["complete"] = all(v["present"] and not v["missing_fields"] for k, v in out.items() if isinstance(v, dict))
    return out


def not_assessed(reason: str) -> dict:
    return {"status": "not assessed", "reason": reason}
