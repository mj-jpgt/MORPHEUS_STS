from dbench.discovery_panel import (DISCOVERY_CAPABILITY_PANEL_V1,
                                     MAIN_PACKETS_PER_CONDITION,
                                     PANEL_ARMS, panel_manifest,
                                     validate_panel_contract)
from dbench.environments import build_world


def test_panel_is_powered_and_has_matched_arms():
    assert MAIN_PACKETS_PER_CONDITION >= 30
    assert PANEL_ARMS == ("source", "reconstruction", "ssl", "morpheus")
    manifest = panel_manifest()
    assert len(manifest["contract_sha256"]) == 64
    assert len(manifest["conditions"]) >= 9


def test_high_order_xor_has_a_complete_typed_truth_object():
    world = build_world("invisible_xor_k4", seed=8, role="proposal")
    truth = [h for h in world.truth.objects if h.true]
    assert len(truth) == 1
    assert truth[0].operator == "XOR"
    assert len(truth[0].operands) == 4


def test_panel_contract_covers_its_tier_one_tasks():
    report = validate_panel_contract()
    assert report["passes"]
    assert {x.id for x in DISCOVERY_CAPABILITY_PANEL_V1} >= {"and_k3", "xor_k4", "hest_spatial"}
