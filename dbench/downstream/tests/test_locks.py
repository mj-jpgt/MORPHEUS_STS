import json

import pytest

from dbench.downstream import locks


def _chain(root, fid, upto):
    for st in locks.STAGES[: locks.STAGES.index(upto) + 1]:
        if st in ("orthogonal", "therapeutic", "distillation", "clinical", "state_card"):
            continue
        locks.write_lock(root, fid, st, inputs={"x": st}, declared_analysis={"a": st})
        locks.write_record(root, fid, st, {"done": st})


def test_canonical_hash_matches_prospective_campaign():
    from dbench.prospective_campaign import canonical_hash

    obj = {"b": [1, 2.5, "x"], "a": {"z": None, "y": True}}
    assert locks.canonical_hash(obj) == canonical_hash(obj)


def test_lock_record_chain_and_dependencies(tmp_path):
    _chain(tmp_path, "F1", "literature")
    lk = locks.write_lock(tmp_path, "F1", "prioritisation", inputs={}, declared_analysis={"rule": "sec4"})
    rec = locks.read_record(tmp_path, "F1", "literature")
    assert lk["depends_on"]["record:literature"] == rec["record_hash"]
    assert lk["order_violation"] is None
    r = locks.write_record(tmp_path, "F1", "prioritisation", {"selected": True})
    assert r["lock_hash"] == lk["lock_hash"] and r["created_utc"] >= lk["created_utc"]


def test_out_of_order_raises_and_violation_is_recorded(tmp_path):
    with pytest.raises(locks.OrderViolation):
        locks.write_lock(tmp_path, "F2", "characterisation", inputs={}, declared_analysis={})
    lk = locks.write_lock(tmp_path, "F2", "characterisation", inputs={}, declared_analysis={}, allow_violation=True)
    assert lk["order_violation"]["missing_prerequisites"] == ["external_replication"]
    card = locks.StateCard(tmp_path, "F2")
    card.update("characterization", {k: locks.not_assessed("test") for k in locks.CARD_SECTIONS["characterization"]})
    assert card.doc["headline_eligible"] is False and card.doc["violations"]


def test_write_once_and_tamper_detection(tmp_path):
    locks.write_lock(tmp_path, "F3", "registry_freeze", inputs={"h": 1}, declared_analysis={})
    with pytest.raises(locks.LockError):
        locks.write_lock(tmp_path, "F3", "registry_freeze", inputs={"h": 2}, declared_analysis={})
    p = locks.lock_path(tmp_path, "F3", "registry_freeze")
    d = json.loads(p.read_text())
    d["inputs"]["h"] = 99
    p.write_text(json.dumps(d))
    with pytest.raises(locks.LockError):
        locks.require_lock(tmp_path, "F3", "registry_freeze")


def test_record_needs_lock_and_card_needs_fields(tmp_path):
    with pytest.raises(locks.LockError):
        locks.write_record(tmp_path, "F4", "registry_freeze", {"x": 1})
    _chain(tmp_path, "F4", "registry_freeze")
    card = locks.StateCard(tmp_path, "F4")
    with pytest.raises(ValueError):
        card.update("identity", {"finding_id": "F4"})
    card.update("identity", {"finding_id": "F4", "cohort": "c", "target": "t", "candidates": ["a@v0"],
                             "relation_params_hash": "h", "registry_hash": "r"})
    again = locks.StateCard(tmp_path, "F4")
    assert again.doc["section_hashes"]["identity"] == locks.canonical_hash(card.doc["sections"]["identity"])
    assert again.doc["provenance"]["locks"][0]["stage"] == "registry_freeze"
    assert again.completeness()["identity"] and not again.completeness()["clinical"]
