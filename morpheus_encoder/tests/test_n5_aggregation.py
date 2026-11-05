"""The N5 report must fail closed unless its full 45-cell contract exists."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "aggregate_n100_joint.py"
    spec = importlib.util.spec_from_file_location("n5_aggregate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _receipt(world, arm, seed):
    row = {"world": world, "arm": arm, "seed": seed,
           "training_uses_truth_or_latents": False,
           "steps": 400, "batch": 512, "recompile_every": 25,
           "objective_mode": "parent_essentiality",
           "incremental_requires_baseline": True,
           "benchmark_status": "qualifying",
           "benchmark_manifest": {"passes_strict": True}}
    if arm != "ssl_continued_control":
        event = {"object_members": [], "object_type": [],
                 "incremental_requires_baseline": True,
                 "items": [{"complete_essentiality": 0.5,
                            "ablations": {"P:0": {"essentiality": 0.5}}}],
                 "rep_to_ssl": 0.5}
        if arm == "parent_essentiality_orthogonal":
            event.update({"ssl_grad_norm": 1.0, "rep_grad_added_norm": 0.5,
                          "post_projection_dot": 0.0})
        row["recompile_events"] = [event]
    return row


def test_n5_aggregation_requires_exact_nonduplicated_grid(tmp_path):
    agg = _module()
    arms, _ = agg.grid_spec("n5")
    for world in agg.WORLDS:
        for arm in arms:
            for seed in agg.SEEDS:
                (tmp_path / f"n5__{world}__{arm}__fixed__s{seed}.json").write_text(
                    json.dumps(_receipt(world, arm, seed)))
    assert len(agg.load(tmp_path, "n5")) == 45
    (tmp_path / "n5__pure_and__ssl_continued_control__fixed__s21.json").unlink()
    with pytest.raises(agg.CompletionError, match="44/45"):
        agg.load(tmp_path, "n5")
