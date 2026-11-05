"""Dry-mode output of the family sweep branches and the v0.5 queue (box only: the scripts run from
~/morpheus_work with ~/morpheus_venv)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

WORK = Path.home() / "morpheus_work"
pytestmark = pytest.mark.skipif(not (WORK / "dbench/scripts/k3_sweep.sh").exists()
                                or not (Path.home() / "morpheus_venv/bin/python").exists(),
                                reason="needs the box layout (~/morpheus_work, ~/morpheus_venv)")


def _dry(campaign, protocol, run=None, folds="0"):
    env = dict(os.environ, CAMPAIGN=campaign, PROTO=str(protocol), FOLDS=folds)
    env.pop("RUN", None)
    if run is not None:
        env["RUN"] = str(run)
    out = subprocess.run(["bash", str(WORK / "dbench/scripts/k3_sweep.sh"), "all", "4", "--dry"], env=env,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [line for line in out if "run_general_discovery_packet" in line], out[-1]


def _protocol(tmp_path, campaign_id, families):
    p = tmp_path / f"{campaign_id}.json"
    p.write_text(json.dumps({"campaign_id": campaign_id, "discovery": {"relationship_families": families}}))
    return p


def test_renal5_and_bmmc4_dry_commands(tmp_path):
    renal = _protocol(tmp_path, "renal-spatial-v05-test", {"famA": {"targets": ["morph_pc1", "cv_count"]},
                                                           "famB": {"targets": ["own_mp1"]},
                                                           "deferred": {"status": "later"}})
    run = tmp_path / "run5"
    cmds, last = _dry("renal5", renal, run, folds="0 3")
    assert len(cmds) == 6 and last.startswith("K3_renal5_all_COMPLETE")
    first = next(c for c in cmds if "f0_morph_pc1" in c)     # xargs -P: output order is not fixed
    for piece in ("--episode hest_real", "--proposer all", "--device cuda", "--arm source", "--family famA",
                  "--embed-root ", f"--output {run}/receipts/famA/f0_morph_pc1.json"):
        assert piece in first
    assert any(f"{run}/receipts/famB/f3_own_mp1.json" in c and "--seed 20260929" in c for c in cmds)
    assert not run.exists()                                     # dry mode creates nothing

    bmmc = _protocol(tmp_path, "bmmc-cite-seq-v04-test", {"molecular_to_protein": {"targets": ["CD3", "CD19"]},
                                                          "molecular_x_donor_context_to_protein": {"targets": []}})
    cmds, last = _dry("bmmc4", bmmc)                            # RUN unset: runs/prospective/<campaign_id>
    assert len(cmds) == 2 and last.startswith("K3_bmmc4_all_COMPLETE")
    cd3 = next(c for c in cmds if "f0_CD3.json" in c)
    assert "--episode bmmc_real" in cd3 and "--embed-root" not in cd3 and "--family molecular_to_protein" in cd3
    assert "--output runs/prospective/bmmc-cite-seq-v04-test/receipts/molecular_to_protein/f0_CD3.json" in cd3


def test_renal4_dry_unchanged_and_queue_dry_writes_nothing(tmp_path):
    cmds, _ = _dry("renal4", WORK / "dbench/protocols/renal_spatial_v04.json", tmp_path / "r4")
    assert len(cmds) == 40 and any(c.endswith(
        f"--output {tmp_path}/r4/receipts/molecular_to_morphology/f0_morph_pc1.json --family molecular_to_morphology"
        f" > {tmp_path}/r4/logs/molecular_to_morphology_f0_morph_pc1.log 2>&1") for c in cmds)
    q = tmp_path / "queue"
    env = dict(os.environ, QUEUE_DIR=str(q), PAUSE_FILE=str(tmp_path / "PAUSE"), K_LOGS=str(tmp_path / "k"))
    out = subprocess.run(["bash", str(WORK / "dbench/scripts/queue_v05.sh"), "--dry"], env=env,
                         capture_output=True, text=True, check=True).stdout
    for step in ("1 repro", "2 merge", "3 kidney", "4 bmmc", "5 finish"):
        assert step in out
    assert "relaunch embed: setsid nohup " in out and "relaunch cellvit: setsid nohup " in out
    assert not q.exists() and not (tmp_path / "PAUSE").exists()
