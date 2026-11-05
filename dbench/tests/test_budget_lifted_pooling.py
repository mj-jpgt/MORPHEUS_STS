"""Budget-lifted finding stage (pooling.finding_stage = budget_lifted): native in _aggregate_families.
Known-answer: on the registered Meylan (renal-spatial-v07) and ependymoma fold receipts the native mode reproduces the
supplementary v2 counts of runs/development/supplementary_pooling_v2 (Agent R). Skipped when the registered run
directories are absent (MORPHEUS_REGISTERED_RUNS, default ~/morpheus_work_v06/runs/prospective)."""
import copy
import json
import os
from pathlib import Path

import pytest

from dbench import prospective_campaign as pc

ROOT = Path(os.environ.get("MORPHEUS_REGISTERED_RUNS", "~/morpheus_work_v06/runs/prospective")).expanduser()
EXPECTED = {
    "renal-spatial-v07-20261022": {"entrants": 2234, "B_nuis": 1953, "B_known": 244, "B_decomp": 11, "both": 1,
                                   "families": {"context_conditioned_programme": (925, 835, 146, 11, 1),
                                                "cross_modal_sets": (815, 684, 96, 0, 0),
                                                "morphology_from_molecular": (437, 378, 2, 0, 0),
                                                "programme_from_morphology": (57, 56, 0, 0, 0)}},
    "ependymoma-spatial-v01-20261022": {"entrants": 2226, "B_nuis": 1679, "B_known": 579, "B_decomp": 63, "both": 25,
                                        "families": {"context_conditioned_programme": (956, 787, 316, 41, 19),
                                                     "cross_modal_sets": (926, 625, 194, 16, 6),
                                                     "morphology_from_molecular": (314, 253, 69, 6, 0),
                                                     "programme_from_morphology": (30, 14, 0, 0, 0)}},
}


def _lifted(proto):
    p = copy.deepcopy(proto)
    p["discovery"]["pooling"] = dict(p["discovery"].get("pooling") or {"mode": "all_folds"}, finding_stage="budget_lifted")
    if isinstance(p["discovery"]["pooling"].get("mode"), str) is False:
        p["discovery"]["pooling"]["mode"] = "all_folds"
    p["discovery"]["candidate_budget"] = None
    for b in p["discovery"]["relationship_families"].values():
        b.pop("budget", None)
    return p


def test_validation_accepts_null_budget_only_when_lifted():
    base = {"schema": "morpheus_prospective_campaign/v1", "campaign_id": "x", "release": {}, "cohorts": [], "ledger": {},
            "anti_leakage": {"literature_before_internal_freeze": False},
            "discovery": {"candidate_budget": None, "n_folds": 5, "advance_rule": {"min_gain": 0.01},
                          "relationship_families": {"f": {"targets": ["gene:A"]}}}}
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(base, fh)
    with pytest.raises(pc.CampaignProtocolError):
        pc.load_protocol(fh.name)
    base["discovery"]["pooling"] = {"mode": "all_folds", "finding_stage": "budget_lifted"}
    with open(fh.name, "w") as f2:
        json.dump(base, f2)
    assert pc.load_protocol(fh.name)["discovery"]["candidate_budget"] is None


@pytest.mark.parametrize("campaign", sorted(EXPECTED))
def test_native_budget_lifted_reproduces_supplementary_v2(campaign):
    rd = ROOT / campaign
    if not (rd / "campaign_manifest.json").exists():
        pytest.skip("registered run directory not available")
    man = json.loads((rd / "campaign_manifest.json").read_text())
    proto = _lifted(man["protocol"])
    pc._validate_families(proto["discovery"])
    out = pc._aggregate_families(proto, man, rd, None, dry_run=True, allow_partial=False)
    exp = EXPECTED[campaign]
    fam = out["families"]
    got = {k: (v["n_frozen"], v["n_internal"], v["n_passes_B_known"], v["n_passes_B_decomp"], v["n_passes_B_known_and_B_decomp"])
           for k, v in fam.items()}
    assert got == exp["families"]
    assert sum(v[0] for v in got.values()) == exp["entrants"]
    assert sum(v[1] for v in got.values()) == exp["B_nuis"]
    assert sum(v[2] for v in got.values()) == exp["B_known"]
    assert sum(v[3] for v in got.values()) == exp["B_decomp"]
    assert sum(v[4] for v in got.values()) == exp["both"]
