import json

import pytest

from dbench.prospective_campaign import CampaignProtocolError, initialise, load_protocol, refresh_preflight


def _protocol(tmp_path):
    contract, sealed, data = (tmp_path / "contract.json", tmp_path / "sealed.json", tmp_path / "data")
    contract.write_text("contract")
    sealed.write_text("sealed")
    data.mkdir()
    import hashlib
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "schema": "morpheus_prospective_campaign/v1", "campaign_id": "test",
        "release": {"contract": {"path": str(contract), "sha256": h(contract)},
                    "sealed_evidence": {"path": str(sealed), "sha256": h(sealed)}},
        "discovery": {"candidate_budget": 50},
        "cohorts": [{"id": "x", "gates": [{"id": "data", "path": str(data)}]}],
        "ledger": {"filename": "ledger.jsonl"},
        "anti_leakage": {"literature_before_internal_freeze": False},
    }


def test_initialise_writes_stable_manifest_and_empty_ledger(tmp_path):
    protocol = _protocol(tmp_path)
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    first = initialise(p, tmp_path / "run")
    second = initialise(p, tmp_path / "run")
    assert first["ready_for_discovery"] and first == second


def test_refuses_literature_leakage(tmp_path):
    protocol = _protocol(tmp_path)
    protocol["anti_leakage"]["literature_before_internal_freeze"] = True
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    with pytest.raises(CampaignProtocolError):
        load_protocol(p)


def test_preflight_refresh_keeps_registration_immutable(tmp_path):
    protocol = _protocol(tmp_path)
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    before = (run / "campaign_manifest.json").read_bytes()
    refreshed = refresh_preflight(p, run)
    assert refreshed["ready_for_discovery"]
    assert (run / "campaign_manifest.json").read_bytes() == before
    assert (run / "preflight_history.jsonl").exists()


def _v02_protocol(tmp_path):
    protocol = _protocol(tmp_path)
    protocol["discovery"].update({
        "arm_split": {"source": 2, "morpheus_factorized": 1}, "n_folds": 5, "proposals_per_target": 4,
        "targets": {"morphology": {"cv_count": "x"}, "spatial_organisation": {"nbr_mp1..nbr_mp2": "y"}},
    })
    return protocol


def _receipt(fold, target, arm, hash_, rows):
    return {"schema": "morpheus_general_discovery_packet/v2", "protocol_sha256": hash_, "fold": fold,
            "target": target, "arm": arm, "candidates": rows}


def _row(symbols, gain, p, rank=1, form="product"):
    return {"gene_symbols": symbols, "feature_form": form, "replication_gain": gain, "proposal_gain": gain,
            "proposal_family": "forward_screen", "baseline_width": 47, "rank": rank, "candidate_hash": "h" + "".join(symbols),
            "nulls": {"spatial_block": {"p": p, "n": 199}, "within_patient_perm": {"p": p, "n": 199},
                      "interaction_break": {"p": p, "n": 199}}}


def _write_matrix(run, hash_, folds=range(5)):
    from dbench.prospective_campaign import aggregate_funnel  # noqa: F401
    for arm in ("source", "morpheus_factorized"):
        d = run / "receipts" / arm
        d.mkdir(parents=True, exist_ok=True)
        for f in folds:
            for t in ("cv_count", "nbr_mp1", "nbr_mp2"):
                rows = [_row(["GENEA", "GENEB"], 0.05, 0.005)]            # recurs in every fold, passes everything
                rows.append(_row(["GENEC", "GENED"], 0.04 if f < 2 else -0.01, 0.005, rank=2))  # gain fails in later folds
                rows.append(_row([f"NOISE{f}", "GENEE"], 0.9, 0.5, rank=3))  # never recurs (fold-specific symbol)
                (d / f"f{f}_{t}.json").write_text(json.dumps(_receipt(f, t, arm, hash_, rows)))


def test_aggregate_funnel_applies_frozen_advance_rule(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel, canonical_hash
    protocol = _v02_protocol(tmp_path)
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    _write_matrix(run, canonical_hash(load_protocol(p)))
    summary = aggregate_funnel(p, run)
    by_stage = {(e["arm"], tuple(e["gene_symbols"])): e for e in summary["table"]}
    assert summary["n_frozen"] == 3 and summary["n_internal"] == 3          # 2 source + 1 morpheus budget; GENEA/GENEB wins each
    assert by_stage[("source", ("GENEA", "GENEB"))]["stage"] == "internal"
    lines = [json.loads(l) for l in (run / "ledger.jsonl").read_text().splitlines()]
    assert {l["stage"] for l in lines} == {"proposed", "internal"}
    assert all(l["evidence"]["replication_level"] == 1 for l in lines)


def test_aggregate_funnel_stops_failed_candidates_with_reason(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel, canonical_hash
    protocol = _v02_protocol(tmp_path)
    protocol["discovery"]["arm_split"] = {"source": 6, "morpheus_factorized": 1}  # 3 targets x 2 recurring candidates
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    _write_matrix(run, canonical_hash(load_protocol(p)))
    summary = aggregate_funnel(p, run, dry_run=True)
    failed = [e for e in summary["table"] if e["gene_symbols"] == ["GENEC", "GENED"] and e["arm"] == "source"]
    assert failed and failed[0]["stage"] == "proposed" and "replication gain" in failed[0]["stop_reason"]
    assert not (run / "ledger.jsonl").read_text().strip()                  # dry run writes nothing


def test_aggregate_funnel_records_one_terminal_proposed_event(tmp_path):
    """A failed candidate remains in the denominator, not duplicated in it."""
    from dbench.prospective_campaign import aggregate_funnel, canonical_hash
    protocol = _v02_protocol(tmp_path)
    protocol["discovery"]["arm_split"] = {"source": 6, "morpheus_factorized": 1}
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    _write_matrix(run, canonical_hash(load_protocol(p)))
    aggregate_funnel(p, run)
    lines = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
    failed = [line for line in lines if line["object"]["operands"] == ["GENEC", "GENED"]]
    assert len(failed) == 3  # one candidate per target, each recorded exactly once
    assert all(line["stage"] == "proposed" and line["stop_reason"] for line in failed)


def test_declared_channel_targets_support_citeseq_campaigns(tmp_path):
    """Protein-channel cohorts use the same frozen funnel as HEST packets."""
    from dbench.prospective_campaign import _declared_targets
    protocol = _v02_protocol(tmp_path)
    protocol["discovery"]["targets"] = {"channels": ["CD3", "CD19", "HLA_DR"]}
    assert _declared_targets(protocol["discovery"]) == ["CD19", "CD3", "HLA_DR"]


def test_aggregate_funnel_refuses_partial_matrix_and_foreign_hash(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel, canonical_hash
    protocol = _v02_protocol(tmp_path)
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    _write_matrix(run, canonical_hash(load_protocol(p)), folds=range(3))
    with pytest.raises(CampaignProtocolError):
        aggregate_funnel(p, run)
    assert aggregate_funnel(p, run, dry_run=True, allow_partial=True)["missing_cells"]["source"]
    (run / "receipts" / "source" / "f0_cv_count.json").write_text(json.dumps(_receipt(0, "cv_count", "source", "deadbeef", [])))
    with pytest.raises(CampaignProtocolError):
        aggregate_funnel(p, run, dry_run=True, allow_partial=True)


# ---------------------------------------------------------------- v0.4 path
# Relationship families, both proposers pooled, recurrence = PASSING folds.

def _v04_protocol(tmp_path):
    protocol = _protocol(tmp_path)
    del protocol["release"]["contract"], protocol["release"]["sealed_evidence"]   # v0.4 has no checkpoint
    protocol["discovery"].update({
        "n_folds": 5, "proposals_per_target": 32, "advance_rule": {"min_gain": 0.01},
        "targets": {"morphology": {"cv_count": "x"}},
        "relationship_families": {
            "famA": {"targets": ["cv_count"], "budget": 50},
            "famB": {"targets": ["own_mp1"], "budget": 50},
            "deferred": {"status": "external stage"}},
    })
    protocol["legacy_leads"] = {"genes": ["PFKP", "ENPEP"]}
    return protocol


def _v04_row(names, gain, *, r2=0.2, ci_lo=0.005, p=0.005, proposer="enumeration", validated=True, form="product"):
    return {"variable_names": list(names), "gene_symbols": [n.split(":")[-1] for n in names], "feature_form": form,
            "replication_gain": gain, "replication_gain_ci": [ci_lo, gain + 0.01], "replication_r2_full": r2,
            "validated": validated, "proposer": proposer, "also_proposed_by": [], "provenance": {"by": proposer},
            "recurrent_in_generator": None if proposer == "enumeration" else True, "candidate_hash": "h",
            "nulls": {"spatial_block": {"p": p, "n": 199}, "within_patient_perm": {"p": p, "n": 199}}}


def _v04_matrix(run, hash_, folds=range(5)):
    for f in folds:
        a = [
            _v04_row(["raw_genes:PFKP", "raw_genes:G2"], 0.05),                                   # passes all 5
            _v04_row(["raw_genes:G3", "raw_genes:G4"], 0.04, ci_lo=0.004 if f < 2 else -0.01),   # 5 proposed, 2 pass
            _v04_row(["raw_genes:G5"], 0.30, r2=0.1 if f < 2 else -0.2, form="linear_subspace"),  # r2<0 in 3 folds
            _v04_row(["raw_genes:G6"], 0.08, validated=(f == 0), form="linear_subspace"),         # 1 validated fold
            # the same hypothesis from both proposers, variables in a different order
            _v04_row(["raw_genes:G8", "pca:pc3"], 0.03, proposer="enumeration"),
            _v04_row(["pca:pc3", "raw_genes:G8"], 0.03, proposer="residual_generator"),
        ]
        a += [_v04_row([f"nmf:k{i}", "raw_genes:G9"], 0.02, p=0.5, ci_lo=0.001 if f < 2 else -1) for i in range(6)]
        b = [_v04_row(["cellvit_structure:frac_dead", "context_neighbour_mean:nbr_x"], 0.015, p=0.04,
                      ci_lo=0.002 if f < 3 else -0.01)]
        for family, target, rows, extra in (("famA", "cv_count", a, {}),
                                            ("famB", "own_mp1", b, {"own_mp_program": {"program_name": "Hypoxia"}})):
            d = run / "receipts" / family
            d.mkdir(parents=True, exist_ok=True)
            receipt = {"schema": "morpheus_general_discovery_packet/v2", "protocol_sha256": hash_, "fold": f,
                       "target": target, "family": family, "proposer": "all", "candidates": rows, **extra}
            (d / f"f{f}_{target}.json").write_text(json.dumps(receipt))


def _v04_run(tmp_path, folds=range(5)):
    from dbench.prospective_campaign import canonical_hash
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(_v04_protocol(tmp_path)))
    run = tmp_path / "run"
    initialise(p, run)
    _v04_matrix(run, canonical_hash(load_protocol(p)), folds)
    return p, run


def test_v04_counts_passing_folds_not_proposed_folds(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel
    p, run = _v04_run(tmp_path)
    s = aggregate_funnel(p, run, dry_run=True)
    assert s["schema"] == "morpheus_funnel_aggregate/v2"
    by = {tuple(e["variable_names"]): e for e in s["table"]}
    top = by[("raw_genes:G2", "raw_genes:PFKP")]
    assert top["stage"] == "internal" and top["passing_folds"] == [0, 1, 2, 3, 4]
    two = by[("raw_genes:G3", "raw_genes:G4")]                       # proposed 5x, passing 2x: frozen, not internal
    assert two["n_proposed_folds"] == 5 and two["passing_folds"] == [0, 1] and two["stage"] == "proposed"
    assert two["stop_reason"].startswith("passing folds 2/5 < 3")
    assert by[("raw_genes:G5",)]["passing_folds"] == [0, 1]          # r2 < 0 folds never count
    assert ("raw_genes:G6",) not in by                               # 1 passing (validated) fold: not frozen
    assert s["families"]["famA"]["below_recurrence"] == 1
    assert "deferred" not in s["families"] and not (run / "ledger.jsonl").read_text().strip()


def test_v04_merges_proposers_and_applies_bh_within_family(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel
    p, run = _v04_run(tmp_path)
    s = aggregate_funnel(p, run, dry_run=True)
    merged = [e for e in s["table"] if e["variable_names"] == ["pca:pc3", "raw_genes:G8"]]
    assert len(merged) == 1 and merged[0]["proposers"] == ["enumeration", "residual_generator"]
    assert merged[0]["stage"] == "internal"
    # famB alone: q = 0.04 <= 0.10 and mean gain 0.015 >= protocol min_gain 0.01. Pooled with the famA
    # frozen set (m = 11) the same p would give q = 0.04 * 11 / 3 > 0.10.
    b = [e for e in s["table"] if e["family"] == "famB"]
    assert len(b) == 1 and b[0]["target"] == "own:Hypoxia" and b[0]["stage"] == "internal"
    assert abs(b[0]["q_value"] - 0.04) < 1e-9
    assert s["families"]["famA"]["n_frozen"] == 10 and s["families"]["famB"]["n_internal"] == 1
    assert s["legacy_lead_check"]["genes"]["PFKP"]["frozen"][0]["n_passing_folds"] == 5
    assert s["legacy_lead_check"]["genes"]["ENPEP"] == {"n_groups_mentioning": 0, "frozen": []}


def test_v04_ledger_evidence_and_partial_matrix(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel
    p, run = _v04_run(tmp_path, folds=range(3))
    with pytest.raises(CampaignProtocolError):
        aggregate_funnel(p, run)
    assert aggregate_funnel(p, run, dry_run=True, allow_partial=True)["missing_cells"]["famB"]
    _v04_matrix(run, json.loads((run / "campaign_manifest.json").read_text())["protocol_sha256"], range(3, 5))
    s = aggregate_funnel(p, run)
    lines = [json.loads(l) for l in (run / "ledger.jsonl").read_text().splitlines()]
    assert len([l for l in lines if l["stage"] == "proposed"]) == s["n_frozen"]
    ev = lines[0]["evidence"]
    for key in ("passing_folds", "n_proposed_folds", "proposers", "family", "variable_names",
                "provenance_by_fold", "explanation_level"):
        assert key in ev
    assert ev["search_budget"] == 32 and ev["explanation_level"] == "pending"
    assert lines[0]["candidate_id"].startswith("test-fam")
    assert json.loads((run / "funnel_aggregate.json").read_text())["schema"] == "morpheus_funnel_aggregate/v2"


def test_v04_protocol_loading(tmp_path):
    from dbench.prospective_campaign import _declared_targets
    protocol = _v04_protocol(tmp_path)
    assert _declared_targets(protocol["discovery"]) == ["cv_count", "own_mp1"]
    for broken in ({"advance_rule": {}}, {"arm_split": {"source": 50}}):
        bad = json.loads(json.dumps(protocol)); bad["discovery"].update(broken)
        p = tmp_path / "bad.json"; p.write_text(json.dumps(bad))
        with pytest.raises(CampaignProtocolError):
            load_protocol(p)


def test_v03_sealed_funnel_reaggregates_exactly():
    """The v0.3 path must reproduce the sealed renal funnel from its receipts (dry run, read-only)."""
    from pathlib import Path
    from dbench.prospective_campaign import aggregate_funnel
    root = Path(__file__).resolve().parents[2]
    run = root / "runs" / "prospective" / "renal-spatial-v03-20260926"
    if not (run / "funnel_aggregate.json").exists():
        pytest.skip("sealed v0.3 run not present on this machine")
    sealed = json.loads((run / "funnel_aggregate.json").read_text())
    again = aggregate_funnel(root / "dbench" / "protocols" / "renal_spatial_v03.json", run, dry_run=True)
    assert [e["candidate_id"] for e in again["table"]] == [e["candidate_id"] for e in sealed["table"]]
    assert len(again["table"]) == 50 and again["n_internal"] == sealed["n_internal"] == 5
    assert again["n_groups"] == sealed["n_groups"]
    for a, b in zip(again["table"], sealed["table"]):
        assert a["stage"] == b["stage"] and a["stop_reason"] == b["stop_reason"]
        assert abs(a["mean_replication_gain"] - b["mean_replication_gain"]) < 1e-9


def test_v04_positive_control_check(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel, canonical_hash
    protocol = _v04_protocol(tmp_path)
    protocol["discovery"]["relationship_families"]["famC"] = {"targets": ["cv_frac_Inflammatory"], "budget": 50}
    protocol["positive_controls"] = {"controls": [
        {"name": "immunoglobulin_to_inflammatory_structure", "genes": ["IGKC"], "family": ["famC"], "expected": "x"},
        {"name": "chemokine_niche_context", "genes": ["CXCL9"], "family": ["famC"], "expected": "y"},
        {"name": "planted_interaction_recovery", "spec": "dev only", "expected": "z"}]}
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    h = canonical_hash(load_protocol(p))
    _v04_matrix(run, h)
    d = run / "receipts" / "famC"; d.mkdir(parents=True)
    for f in range(5):
        rows = [_v04_row(["IGKC"], 0.03, ci_lo=0.002 if f < 2 else -0.01, form="linear_subspace"),
                _v04_row(["CXCL9", "context_neighbour_mean:nbr_cv_frac_Inflammatory"], 0.01, ci_lo=-0.01)]
        (d / f"f{f}_cv_frac_Inflammatory.json").write_text(json.dumps({
            "protocol_sha256": h, "fold": f, "target": "cv_frac_Inflammatory", "family": "famC", "candidates": rows}))
    s = aggregate_funnel(p, run, dry_run=True)
    c = s["positive_control_check"]["controls"]
    ig = c["immunoglobulin_to_inflammatory_structure"]
    assert ig["expected_met"] is True and ig["hits"][0]["n_passing_folds"] == 2 and ig["hits"][0]["stage"] == "proposed"
    chem = c["chemokine_niche_context"]
    assert chem["expected_met"] is True and chem["hits"][0]["stage"] == "not_frozen"   # proposed, never passing
    assert c["planted_interaction_recovery"]["expected_met"] == "not_evaluated_here"
    assert s["families"]["famC"]["n_internal"] == 0                                    # changes no stage


def test_v04_required_nulls_gate_internal_and_others_are_diagnostic(tmp_path):
    from dbench.prospective_campaign import aggregate_funnel, canonical_hash
    protocol = _v04_protocol(tmp_path)
    protocol["discovery"]["null_suite"] = {"required_for_internal": ["within_patient_perm"]}
    p = tmp_path / "protocol.json"; p.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    initialise(p, run)
    h = canonical_hash(load_protocol(p))
    d = run / "receipts" / "famA"; d.mkdir(parents=True)
    for f in range(5):
        inv = _v04_row(["raw_genes:A1"], 0.05, form="linear_subspace")
        inv["nulls"]["spatial_block"]["p"] = 1.0                 # invariance diagnostic near 1: must not block
        dep = _v04_row(["raw_genes:B1"], 0.04, form="linear_subspace")
        dep["nulls"]["spatial_block"]["p"] = 0.01                # gain depends on that structure: flagged only
        bad = _v04_row(["raw_genes:C1"], 0.06, form="linear_subspace")
        bad["nulls"]["within_patient_perm"]["p"] = 0.3           # required null fails
        (d / f"f{f}_cv_count.json").write_text(json.dumps({"protocol_sha256": h, "fold": f, "target": "cv_count",
                                                           "family": "famA", "candidates": [inv, dep, bad]}))
    (run / "receipts" / "famB").mkdir()
    s = aggregate_funnel(p, run, dry_run=True, allow_partial=True)
    by = {e["variable_names"][0]: e for e in s["table"]}
    assert by["raw_genes:A1"]["stage"] == "internal" and by["raw_genes:A1"]["environment_dependent"] is False
    assert by["raw_genes:A1"]["diagnostic_nulls"]["spatial_block"]["0"] == 1.0
    assert by["raw_genes:B1"]["stage"] == "internal" and by["raw_genes:B1"]["environment_dependent"] is True
    assert by["raw_genes:C1"]["stage"] == "proposed" and "structured null p 0.300" in by["raw_genes:C1"]["stop_reason"]
    # without the field every kind gates, exactly as before
    protocol["discovery"].pop("null_suite")
    p.write_text(json.dumps(protocol))
    run2 = tmp_path / "run2"
    initialise(p, run2)
    h2 = canonical_hash(load_protocol(p))
    (run2 / "receipts").mkdir()
    import shutil
    shutil.copytree(run / "receipts", run2 / "receipts", dirs_exist_ok=True)
    for path in (run2 / "receipts" / "famA").glob("*.json"):
        r = json.loads(path.read_text()); r["protocol_sha256"] = h2; path.write_text(json.dumps(r))
    s2 = aggregate_funnel(p, run2, dry_run=True, allow_partial=True)
    by2 = {e["variable_names"][0]: e for e in s2["table"]}
    assert by2["raw_genes:A1"]["stage"] == "proposed" and "diagnostic_nulls" not in by2["raw_genes:A1"]


def test_v04_registered_renal_funnel_reaggregates_exactly():
    """With no required_for_internal field the registered kidney v0.4 pooling must reproduce exactly."""
    from pathlib import Path
    from dbench.prospective_campaign import aggregate_funnel
    root = Path(__file__).resolve().parents[2]
    run = root / "runs" / "prospective" / "renal-spatial-v04-20261011"
    if not (run / "funnel_aggregate.json").exists():
        pytest.skip("registered v0.4 run not present on this machine")
    sealed = json.loads((run / "funnel_aggregate.json").read_text())
    again = aggregate_funnel(root / "dbench" / "protocols" / "renal_spatial_v04.json", run, dry_run=True,
                             allow_partial=bool(any(sealed["missing_cells"].values())))
    assert [e["candidate_id"] for e in again["table"]] == [e["candidate_id"] for e in sealed["table"]]
    assert again["n_groups"] == sealed["n_groups"] and again["families"] == sealed["families"]
    for a, b in zip(again["table"], sealed["table"]):
        assert a == b                                   # same id, stage, stop reason, passing folds, q
        assert abs(a["mean_replication_gain"] - b["mean_replication_gain"]) < 1e-9


def test_v03_rff_four_baseline_pass_counts_are_reported_not_gated(tmp_path):
    """Rows carrying gain_B_known_ci (relation v03_rff): the funnel reports B_nuis / B_known / B_decomp /
    B_known and B_decomp passes per family and per cohort; stages are unchanged."""
    from dbench.prospective_campaign import aggregate_funnel
    p, run = _v04_run(tmp_path)
    before = {tuple(e["variable_names"]): e["stage"] for e in aggregate_funnel(p, run, dry_run=True)["table"]}
    assert "baseline_pass_counts" not in aggregate_funnel(p, run, dry_run=True)     # older receipts: unchanged
    for path in sorted((run / "receipts").glob("*/f*_*.json")):
        r = json.loads(path.read_text())
        f = r["fold"]
        for row in r["candidates"]:
            top = "raw_genes:PFKP" in row["variable_names"]
            row["gain_B_known_ci"] = [0.002, 0.03] if (top and f < 4) else [-0.01, 0.02]
            row["higher_order"] = bool(top and f < 3)
        path.write_text(json.dumps(r))
    s = aggregate_funnel(p, run, dry_run=True)
    by = {tuple(e["variable_names"]): e for e in s["table"]}
    assert {k: e["stage"] for k, e in by.items()} == before
    top = by[("raw_genes:G2", "raw_genes:PFKP")]
    assert top["known_folds"] == [0, 1, 2, 3] and top["decomp_folds"] == [0, 1, 2]
    assert top["passes_B_nuis"] and top["passes_B_known"] and top["passes_B_decomp"] and top["passes_B_known_and_B_decomp"]
    merged = by[("pca:pc3", "raw_genes:G8")]
    assert merged["passes_B_nuis"] and not merged["passes_B_known"] and not merged["passes_B_decomp"]
    c = s["baseline_pass_counts"]
    assert c["cohort"]["n_passes_B_known"] == 1 and c["cohort"]["n_passes_B_known_and_B_decomp"] == 1
    assert c["cohort"]["n_passes_B_nuis"] == sum(e["stage"] == "internal" for e in s["table"])
    assert set(c["by_family"]) == {"famA", "famB"}
