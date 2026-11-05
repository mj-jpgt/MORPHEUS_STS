"""Sprint data partitions (`dbench/protocols/sprint_partitions_v01.json`): roles are
exclusive, untouched studies carry evidence, and every id exists in the catalogue
and on disk."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARTITIONS = os.path.join(REPO, "dbench", "protocols", "sprint_partitions_v01.json")
PARTITIONS_SHA256 = "06301b22e72a3a3032fef873c170f3e5f4d1f0ee7b5412d1a31d93b0928b1ed1"
ROLES = {"meta_training", "development", "historical", "blind", "external_replication", "reserve"}


@pytest.fixture(scope="module")
def doc():
    with open(PARTITIONS) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def catalogue(doc):
    if not os.path.exists(doc["catalogue"]):
        pytest.skip("HEST catalogue not on this machine")
    import pandas as pd
    return pd.read_csv(doc["catalogue"])


def test_every_study_has_exactly_one_role(doc):
    seen = {}
    for role, keys in doc["roles"].items():
        if role == "sanity":
            continue
        assert role in ROLES, role
        for k in keys:
            assert k not in seen, f"{k} in {seen[k]} and {role}"
            seen[k] = role
    assert set(seen) == set(doc["studies"])
    for k, s in doc["studies"].items():
        assert s["role"] == seen[k]
    # the one allowed overlap
    assert doc["roles"].get("sanity", []) == ["meylan2022_ccrcc"]
    assert doc["studies"]["meylan2022_ccrcc"]["role"] == "blind"


def test_no_dataset_title_or_patient_or_id_shared_across_studies(doc):
    owner_title, owner_id, owner_patient = {}, {}, {}
    for k, s in doc["studies"].items():
        for t in s.get("dataset_titles", []):
            assert t not in owner_title, f"title {t!r} in {owner_title.get(t)} and {k}"
            owner_title[t] = k
        for i in s.get("hest_ids", []):
            assert i not in owner_id, f"id {i} in {owner_id.get(i)} and {k}"
            owner_id[i] = k
        for p in set(s.get("patient_of", {}).values()):
            assert p.startswith(k + ":") and p not in owner_patient
            owner_patient[p] = k


def test_plan_role_assignment(doc):
    r = {k: s["role"] for k, s in doc["studies"].items()}
    for k in ("he2020_breast_idc", "andersson2021_her2", "lymphnode_idc", "bladder_epithelial", "bladder_batf3",
              "hcc_trem1"):
        assert r[k] == "meta_training"
    assert r["kidney_injury"] == "development"
    assert {k for k, v in r.items() if v == "historical"} == {"ji2020_cscc", "valdeolivas2024_crc",
                                                              "kiviaho2024_prostate"}
    assert {k for k, v in r.items() if v == "blind"} == {"meylan2022_ccrcc", "ependymoma_pfa"}
    assert {k for k, v in r.items() if v == "external_replication"} == {"li2022_ccrcc", "gse250163",
                                                                        "zenodo14620362"}
    for k in ("kiviaho2024_prostate", "li2022_ccrcc", "gse250163", "zenodo14620362"):
        s = doc["studies"][k]
        assert s["pending_conversion"] is False and s["layout"] == "flat" and s["hest_ids"]
        assert s["source_kind"] == "external_converted"
    kiv = doc["studies"]["kiviaho2024_prostate"]
    assert kiv["n_sections"] == 48 and kiv["n_patients"] == 48
    assert sorted(kiv["excluded_ids"]) == [f"Kiviaho2024_MET_{c}" for c in "ABCD"]
    assert not any(i.startswith("Kiviaho2024_MET_") for i in kiv["hest_ids"])
    assert len(doc["studies"]["li2022_ccrcc"]["hest_ids"]) == 16
    assert doc["studies"]["li2022_ccrcc"]["n_patients"] == 8
    assert len(doc["studies"]["gse250163"]["hest_ids"]) == 5
    assert doc["studies"]["zenodo14620362"]["meets_0p5_um_per_px_rule"] is False
    ji = doc["studies"]["ji2020_cscc"]
    for i in ("MEND37", "MEND38", "MEND39", "MEND40"):
        assert i in ji["hest_ids"] and ji["platform_of"][i] == "visium"
    assert ji["platform_of"]["NCBI759"] == "st_100um"


def test_revision_and_pinned_hash(doc):
    assert doc["revision"] == 2
    with open(PARTITIONS, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    assert sha == PARTITIONS_SHA256
    receipt = os.path.join(REPO, "runs", "development", "sprint_v06", "partitions_sha256.txt")
    if os.path.exists(receipt):
        with open(receipt) as fh:
            assert fh.read().split()[0] == sha


def test_historical_studies_are_untouched_with_evidence(doc):
    for k in doc["roles"]["historical"]:
        s = doc["studies"][k]
        assert s["previously_influenced_morpheus"] is False
        assert isinstance(s["evidence"], str) and len(s["evidence"]) > 20
    for k in ("ji2020_cscc", "valdeolivas2024_crc"):
        assert "2026-08-09" in doc["studies"][k]["evidence"] and "embedding only" in doc["studies"][k]["evidence"]


def test_catalogue_hash_and_counts_match(doc, catalogue):
    with open(doc["catalogue"], "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == doc["sha256_of_catalogue"]
    by_id = catalogue.set_index("id")
    for k, s in doc["studies"].items():
        if s.get("pending_conversion") or s.get("source_kind") == "external_converted":
            continue
        ids = s["hest_ids"]
        assert ids and s["n_sections"] == len(ids)
        assert s["n_patients"] == len(set(s["patient_of"].values()))
        assert set(s["patient_of"]) == set(ids)
        for i in ids:
            assert i in by_id.index, f"{k}: {i} not in catalogue"
            assert by_id.loc[i, "dataset_title"] in s["dataset_titles"]
        if len(s["dataset_titles"]) == 1 and s["role"] != "reserve" and k != "tenx_demo_bowel":
            n_cat = int((catalogue.dataset_title == s["dataset_titles"][0]).sum())
            assert len(ids) <= n_cat
    counts = {}
    for role, keys in doc["roles"].items():
        counts[role] = {"n_studies": len(keys),
                        "n_sections": sum(doc["studies"][k]["n_sections"] or 0 for k in keys),
                        "n_patients": sum(doc["studies"][k]["n_patients"] or 0 for k in keys)}
    assert counts == doc["counts"]


def test_every_id_on_disk(doc):
    for k, s in doc["studies"].items():
        if s.get("pending_conversion"):
            continue
        if not os.path.isdir(s["data_dir"]):
            pytest.skip("store not mounted")
        for i in s["hest_ids"]:
            if s["layout"] == "mirror":
                paths = [f"{s['data_dir']}/{i}_st.h5ad", f"{s['data_dir']}/{i}_metadata.json"]
            else:
                paths = [f"{s['data_dir']}/st/{i}.h5ad", f"{s['data_dir']}/metadata/{i}.json"]
            for p in paths:
                assert os.path.exists(p), p
