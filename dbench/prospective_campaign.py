"""Frozen, literature-blind campaign protocol for prospective MORPHEUS work.

This module deliberately owns campaign identity, data gates and immutable
ledger initialisation; it does not propose a biological result.  Discovery
workers must receive the manifest it emits and may advance a candidate only
through :class:`dbench.ledger.CandidateLedger`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .ledger import CandidateLedger


class CampaignProtocolError(RuntimeError):
    pass


def canonical_hash(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != "morpheus_prospective_campaign/v1":
        raise CampaignProtocolError("unsupported prospective campaign schema")
    required = {"campaign_id", "release", "discovery", "cohorts", "ledger", "anti_leakage"}
    missing = sorted(required - set(protocol))
    if missing:
        raise CampaignProtocolError("protocol missing %s" % ", ".join(missing))
    if protocol["anti_leakage"].get("literature_before_internal_freeze") is not False:
        raise CampaignProtocolError("literature access must remain disabled before internal freeze")
    if protocol["discovery"].get("candidate_budget") != 50 and not (
            protocol["discovery"].get("candidate_budget") is None and budget_lifted(protocol["discovery"])):
        raise CampaignProtocolError("renal campaign candidate budget must remain 50 (null only with pooling.finding_stage = budget_lifted)")
    if "relationship_families" in protocol["discovery"]:
        _validate_families(protocol["discovery"])
    return protocol


def _validate_families(discovery: Mapping[str, Any]) -> None:
    """v0.4 relationship-family protocols: per-family targets, budget 50 each,
    and a numeric advance-rule minimum gain (the internal clause is frozen at
    registration, so it must be machine-readable, not only prose)."""
    if "arm_split" in discovery:
        raise CampaignProtocolError("a protocol declares either arm_split (v0.3) or relationship_families (v0.4), not both")
    families = _family_targets(discovery)
    if not families:
        raise CampaignProtocolError("relationship_families declares no family with targets")
    for name in families:
        if discovery["relationship_families"][name].get("budget", discovery["candidate_budget"]) != discovery["candidate_budget"]:
            raise CampaignProtocolError(f"family {name} budget must equal candidate_budget (50)")
    min_gain = (discovery.get("advance_rule") or {}).get("min_gain")
    if not isinstance(min_gain, (int, float)) or isinstance(min_gain, bool):
        raise CampaignProtocolError("relationship-family protocol needs numeric discovery.advance_rule.min_gain")


def preflight(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Verify only declared paths and fingerprints; never inspect outcomes."""
    checks: list[dict[str, Any]] = []
    release = protocol["release"]
    for key in ("contract", "sealed_evidence"):
        if key not in release:   # v0.4: no representation checkpoint to fingerprint
            continue
        path = Path(release[key]["path"])
        expected = release[key]["sha256"]
        found = path.exists()
        actual = None
        if found:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({"kind": key, "path": str(path), "exists": found,
                       "hash_matches": actual == expected if found else False})
    for cohort in protocol["cohorts"]:
        for gate in cohort["gates"]:
            path = Path(gate["path"])
            checks.append({"kind": gate["id"], "path": str(path), "exists": path.exists(),
                           "required": bool(gate.get("required", True))})
    required_ok = all(row["exists"] and row.get("hash_matches", True)
                      for row in checks if row.get("required", True))
    return {"campaign_id": protocol["campaign_id"], "protocol_sha256": canonical_hash(protocol),
            "checks": checks, "ready_for_discovery": required_ok,
            "reason": None if required_ok else "one or more required data/provenance gates failed"}


def initialise(protocol_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Write immutable campaign metadata and initialise its append-only ledger.

    The operation refuses a conflicting manifest on re-run.  It intentionally
    does not create a candidate or query literature.
    """
    protocol = load_protocol(protocol_path)
    result = preflight(protocol)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "morpheus_prospective_campaign_manifest/v1", **result,
                "protocol": protocol}
    target = root / "campaign_manifest.json"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise CampaignProtocolError("existing campaign manifest differs; start a new campaign ID")
    target.write_text(text, encoding="utf-8")
    ledger = CandidateLedger(str(root / protocol["ledger"]["filename"]))
    return {"manifest": str(target), "ledger": ledger.path, **result}


def refresh_preflight(protocol_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Append a new intake check without mutating the frozen registration.

    A data/provenance receipt can be completed after campaign registration;
    candidate definitions and the campaign protocol cannot.  This distinction
    keeps the protocol immutable while preserving an auditable gate history.
    """
    protocol = load_protocol(protocol_path)
    root = Path(run_dir)
    manifest = root / "campaign_manifest.json"
    if not manifest.exists():
        raise CampaignProtocolError("campaign must be initialised before preflight refresh")
    registered = json.loads(manifest.read_text(encoding="utf-8"))
    if registered.get("protocol_sha256") != canonical_hash(protocol):
        raise CampaignProtocolError("protocol differs from registered campaign")
    result = preflight(protocol)
    event = {
        "event": "preflight_refresh",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **result,
    }
    with (root / "preflight_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    (root / "preflight_latest.json").write_text(
        json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _bh_q(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    q, running = {}, 1.0
    for rank in range(m, 0, -1):
        key, p = items[rank - 1]
        running = min(running, p * m / rank)
        q[key] = min(1.0, running)
    return q


def _declared_targets(discovery: Mapping[str, Any]) -> list[str]:
    """Return the frozen target names without assuming a particular modality.

    HEST packets declare morphology and spatial-organisation targets, whereas
    CITE-seq packets declare measured protein channels.  Both are valid
    prospective campaigns: the aggregator must not silently exclude a cohort
    merely because its target vocabulary differs from the first campaign.
    Ranges such as ``nbr_mp1..nbr_mp8`` are expanded deterministically.
    """
    target_spec = discovery.get("targets", {})
    targets: list[str] = []
    for section in ("morphology", "spatial_organisation"):
        for key in target_spec.get(section, {}):
            targets.extend(_expand_target(str(key)))
    # A source-only BMMC campaign is deliberately protein-channel based.
    # The list, rather than the values of a mapping, is part of its frozen
    # protocol and has no outcome information in it.
    targets.extend(str(channel) for channel in target_spec.get("channels", ()))
    # v0.4: relationship families name their own targets (e.g. own_mp1..8).
    for family_targets in _family_targets(discovery).values():
        targets.extend(family_targets)
    if not targets:
        raise CampaignProtocolError("discovery.targets declares no aggregatable targets")
    return sorted(set(targets))


def _expand_target(key: str) -> list[str]:
    if ".." not in key:
        return [key]
    head, tail = key.split("..", 1)
    prefix = head.rstrip("0123456789")
    start, stop = head[len(prefix):], tail[len(prefix):]
    if not prefix or not start.isdigit() or not stop.isdigit():
        raise CampaignProtocolError(f"invalid frozen target range: {key}")
    return [f"{prefix}{i}" for i in range(int(start), int(stop) + 1)]


def _family_targets(discovery: Mapping[str, Any]) -> dict[str, list[str]]:
    """Families that carry a target list (a deferred family, e.g. a
    tissue_to_patient block with only a status, is not part of the sweep)."""
    out = {}
    for name, block in (discovery.get("relationship_families") or {}).items():
        if isinstance(block, Mapping) and block.get("targets"):
            out[str(name)] = sorted({t for key in block["targets"] for t in _expand_target(str(key))})
    return out


def aggregate_funnel(protocol_path: str | Path, run_dir: str | Path, receipts_dir: str | Path | None = None,
                     *, dry_run: bool = False, allow_partial: bool = False) -> dict[str, Any]:
    """Fold receipts -> frozen candidates -> ledger, under the frozen advance rule.

    This is the ONLY path from receipts into the campaign ledger. Identity of a
    candidate across folds is (arm, target, operator, sorted gene symbols); a
    candidate "recurs" when it is frozen in >= 2 folds. Per arm, the top
    `arm_split[arm]` recurring candidates by mean replication gain over their
    recurring folds enter the ledger at `proposed` (the 50-candidate budget).
    `internal` requires, per the protocol: recurrence in >= 3 of n_folds,
    replication gain >= min_gain in each recurring fold, every structured null
    p <= 0.05 in each recurring fold, and BH q <= 0.10 over the frozen 50 using
    the candidate's worst (max) null p across its recurring folds and kinds.
    Candidates that fail carry a `stop_reason` naming the first failed clause.
    Nothing here reads literature or outcomes; nothing is written in dry-run.
    """
    protocol = load_protocol(protocol_path)
    run_dir = Path(run_dir)
    manifest_path = run_dir / "campaign_manifest.json"
    if not manifest_path.exists():
        raise CampaignProtocolError("campaign must be initialised before aggregation")
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol_sha256"] != canonical_hash(protocol):
        raise CampaignProtocolError("protocol differs from registered campaign")
    discovery = protocol["discovery"]
    if "relationship_families" in discovery and "arm_split" not in discovery:
        return _aggregate_families(protocol, manifest, run_dir, receipts_dir, dry_run=dry_run,
                                   allow_partial=allow_partial)
    arm_split: dict[str, int] = dict(discovery["arm_split"])
    n_folds = int(discovery["n_folds"])
    targets = _declared_targets(discovery)
    min_gain, fdr = 0.02, 0.10
    receipts_dir = Path(receipts_dir) if receipts_dir else run_dir / "receipts"

    groups: dict[tuple, dict[str, Any]] = {}
    seen: dict[str, set[tuple[int, str]]] = {arm: set() for arm in arm_split}
    for arm in arm_split:
        for path in sorted((receipts_dir / arm).glob("f*_*.json")):
            receipt = json.loads(path.read_text())
            if receipt.get("protocol_sha256") != manifest["protocol_sha256"]:
                raise CampaignProtocolError(f"receipt {path.name} carries a foreign protocol hash")
            fold, target = int(receipt["fold"]), str(receipt["target"])
            seen[arm].add((fold, target))
            # A neighbour-program target is named by its D_P Moran's-I rank, which
            # can pick a different Gavish program in different folds; candidate
            # identity across folds is therefore keyed on the PROGRAM the fold
            # actually used (recorded in the receipt), not on the rank label.
            info = receipt.get("nbr_mp_program") or {}
            if info.get("program_name"):
                target = f"nbr:{info['program_name']}"
            for row in receipt["candidates"]:
                operator = "flexible_product" if row.get("feature_form") == "product" else "flexible_subspace"
                symbols = tuple(sorted(str(s) for s in row.get("gene_symbols") or ()))
                if not symbols:
                    continue
                key = (arm, target, operator, symbols)
                g = groups.setdefault(key, {"arm": arm, "target": target, "operator": operator,
                                            "gene_symbols": list(symbols), "folds": {}, "proposal_family": row.get("proposal_family"),
                                            "baseline_width": row.get("baseline_width"), "receipts": []})
                g["folds"][fold] = {"replication_gain": float(row["replication_gain"]),
                                    "proposal_gain": float(row.get("proposal_gain", 0.0)),
                                    "nulls": {k: v.get("p") for k, v in (row.get("nulls") or {}).items()},
                                    "receipt_rank": row.get("rank"), "candidate_hash": row.get("candidate_hash"),
                                    "observed_coordinate_links": row.get("observed_coordinate_links")}
                g["receipts"].append(str(path))
    expected = {(f, t) for f in range(n_folds) for t in targets}
    missing = {arm: sorted(expected - seen[arm]) for arm in arm_split}
    if any(missing.values()) and not allow_partial:
        raise CampaignProtocolError("receipt matrix incomplete: " + json.dumps({a: len(m) for a, m in missing.items()}))

    frozen: list[dict[str, Any]] = []
    for arm, budget in arm_split.items():
        recurring = [g for g in groups.values() if g["arm"] == arm and len(g["folds"]) >= 2]
        for g in recurring:
            gains = [f["replication_gain"] for f in g["folds"].values()]
            g["recurrence"] = len(g["folds"])
            g["mean_replication_gain"] = sum(gains) / len(gains)
            g["min_replication_gain"] = min(gains)
            ps = [p for f in g["folds"].values() for p in f["nulls"].values() if p is not None]
            g["worst_null_p"] = max(ps) if ps else 1.0
            g["all_nulls_present"] = all(f["nulls"] for f in g["folds"].values())
        recurring.sort(key=lambda g: (-g["mean_replication_gain"], -g["recurrence"], g["gene_symbols"]))
        frozen.extend(recurring[:budget])
    q_values = _bh_q({_cid(manifest["campaign_id"], g): g["worst_null_p"] for g in frozen}) if frozen else {}

    ledger = None if dry_run else CandidateLedger(str(run_dir / protocol["ledger"]["filename"]))
    table = []
    for g in frozen:
        cid = _cid(manifest["campaign_id"], g)
        g["q_value"] = q_values[cid]
        reasons = []
        if g["recurrence"] < 3:
            reasons.append(f"recurrence {g['recurrence']}/{n_folds} < 3")
        if g["min_replication_gain"] < min_gain:
            reasons.append(f"replication gain {g['min_replication_gain']:.4f} < {min_gain} in a recurring fold")
        if not g["all_nulls_present"] or g["worst_null_p"] > 0.05:
            reasons.append(f"structured null p {g['worst_null_p']:.3f} > 0.05")
        if g["q_value"] > fdr:
            reasons.append(f"BH q {g['q_value']:.3f} > {fdr}")
        stage = "internal" if not reasons else "proposed"
        evidence = {"fold_gains": {str(k): v["replication_gain"] for k, v in sorted(g["folds"].items())},
                    "null_p_by_fold": {str(k): v["nulls"] for k, v in sorted(g["folds"].items())},
                    "recurrence": g["recurrence"], "n_folds": n_folds, "mean_replication_gain": g["mean_replication_gain"],
                    "worst_null_p": g["worst_null_p"], "q": g["q_value"], "arm": g["arm"], "target": g["target"],
                    "proposal_mechanism": g["proposal_family"], "baseline_width": g["baseline_width"],
                    "search_budget": int(discovery["proposals_per_target"]), "protocol_sha256": manifest["protocol_sha256"],
                    "replication_level": 1, "replication_level_meaning": "held-out HEST patients (same acquisition)"}
        row = {"operator": g["operator"], "gene_symbols": g["gene_symbols"], "baseline_width": g["baseline_width"],
               "context": {"target": g["target"], "cohort": "hest_ccrcc"}}
        entry = {"candidate_id": cid, "stage": stage, "arm": g["arm"], "target": g["target"], "operator": g["operator"],
                 "gene_symbols": g["gene_symbols"], "recurrence": g["recurrence"],
                 "mean_replication_gain": round(g["mean_replication_gain"], 5), "worst_null_p": g["worst_null_p"],
                 "q_value": round(g["q_value"], 4), "stop_reason": "; ".join(reasons) or None}
        table.append(entry)
        if ledger is not None:
            from .ledger import record_from_receipt
            record_from_receipt(ledger, cid, "proposed", row, evidence=evidence, data_ref=";".join(g["receipts"]),
                                stop_reason=entry["stop_reason"],
                                notes="frozen by aggregate_funnel under the registered protocol")
            if stage == "internal":
                record_from_receipt(ledger, cid, "internal", row, evidence=evidence, data_ref=";".join(g["receipts"]),
                                    notes="advance rule satisfied: recurrence, gain, nulls, BH q")
    summary = {"schema": "morpheus_funnel_aggregate/v1", "campaign_id": manifest["campaign_id"],
               "protocol_sha256": manifest["protocol_sha256"], "receipts_seen": {a: len(s) for a, s in seen.items()},
               "missing_cells": missing, "n_groups": len(groups), "n_frozen": len(frozen),
               "n_internal": sum(1 for e in table if e["stage"] == "internal"), "dry_run": dry_run,
               "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "table": table}
    if not dry_run:
        (run_dir / "funnel_aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _cid(campaign_id: str, g: Mapping[str, Any]) -> str:
    blob = json.dumps([g["arm"], g["target"], g["operator"], g["gene_symbols"]], sort_keys=True).encode()
    return f"{campaign_id}-{g['arm']}-{hashlib.sha256(blob).hexdigest()[:12]}"


_PROGRAM_TARGETS = (("nbr_mp_program", "nbr"), ("own_mp_program", "own"))


def budget_lifted(discovery: Mapping[str, Any]) -> bool:
    """``pooling.finding_stage == "budget_lifted"``: no per-family cut at the finding stage.  Every hypothesis
    group passing the fold rule in >= 3 folds enters (``candidate_budget: null``), BH q is taken within family
    over all entrants, and the ledger freezes every entrant.  Near-duplicate collapse and the
    second_catalogue_known / target_leak flags are a reporting layer applied afterwards
    (dbench/downstream/scripts/bknown_semantics.py)."""
    pooling = discovery.get("pooling")
    return isinstance(pooling, Mapping) and pooling.get("finding_stage") == "budget_lifted"


def pooling_mode(discovery: Mapping[str, Any]) -> str:
    """``"all_folds"`` when the protocol's pooling step scores every frozen hypothesis in every
    fold (``pooling: "all_folds"``, ``pooling.mode``, or a pooling rule that includes "the folds
    in which it was not frozen"); otherwise ``"frozen_folds"`` (pool over folds where frozen)."""
    pooling = discovery.get("pooling")
    if pooling == "all_folds":
        return "all_folds"
    if isinstance(pooling, Mapping):
        if pooling.get("mode") == "all_folds":
            return "all_folds"
        if "the folds in which it was not frozen" in str(pooling.get("rule", "")):
            return "all_folds"
    return "frozen_folds"


def _safe_name(target: str) -> str:
    """File-name form of a target: '/' and ' ' -> '_' (e.g. programme:mp:Interferon/MHC-II (I)); the
    receipt and the request carry the exact target, so only the path changes."""
    return str(target).replace("/", "_").replace(" ", "_")


def crossfold_requests(protocol_path: str | Path, run_dir: str | Path, receipts_dir: str | Path | None = None,
                       write: bool = True) -> dict[str, Any]:
    """The cross-fold scoring pass of an all-fold-pooling protocol: for every family, fold and
    target, the hypotheses frozen in OTHER folds (same resolved target) that this fold has not
    scored.  Each cell names the fold receipt it extends (same episode, study, target) and the
    runner output path ``receipts/<family>/crossfold/f<fold>_<target>.json``; the hypothesis
    list for ``--score-frozen`` is written under ``crossfold_requests/``.  Hypotheses whose
    resolved target has no receipt in a fold are listed as not scoreable there."""
    protocol = load_protocol(protocol_path)
    discovery = protocol["discovery"]
    run_dir = Path(run_dir)
    receipts_dir = Path(receipts_dir) if receipts_dir else run_dir / "receipts"
    n_folds = int(discovery["n_folds"])
    cells, unmatched = [], []
    for family in _family_targets(discovery):
        by_cell: dict[tuple, dict] = {}
        hyps: dict[tuple, dict] = {}
        for path in sorted((receipts_dir / family).glob("f*_*.json")):
            receipt = json.loads(path.read_text())
            fold, target = int(receipt["fold"]), _resolved_target(receipt)
            by_cell[(fold, target)] = {"receipt": str(path), "target": receipt["target"],
                                       "episode": receipt.get("episode"), "study": receipt.get("study"),
                                       "loader_target": receipt.get("loader_target")}
            for row in receipt["candidates"]:
                names = tuple(sorted(str(v) for v in row.get("variable_names") or ()))
                if not names:
                    continue
                h = hyps.setdefault((target, str(row.get("feature_form")), names), {
                    "variable_names": list(names), "feature_form": str(row.get("feature_form")),
                    "frozen_in_folds": set(), "proposers": set()})
                h["frozen_in_folds"].add(fold)
                h["proposers"].add(str(row.get("proposer")))
        for (target, form, names), h in sorted(hyps.items()):
            for fold in range(n_folds):
                if fold in h["frozen_in_folds"]:
                    continue
                cell = by_cell.get((fold, target))
                if cell is None:
                    unmatched.append({"family": family, "fold": fold, "resolved_target": target,
                                      "variable_names": list(names), "reason": "no receipt of this fold has the target"})
                    continue
                key = (family, fold, cell["target"])
                found = next((c for c in cells if (c["family"], c["fold"], c["target"]) == key), None)
                if found is None:
                    found = {"family": family, "fold": fold, **cell, "resolved_target": target,
                             "output": str(receipts_dir / family / "crossfold" / f"f{fold}_{_safe_name(cell['target'])}.json"),
                             "hypotheses": []}
                    cells.append(found)
                found["hypotheses"].append({"variable_names": list(names), "feature_form": form,
                                            "frozen_in_folds": sorted(h["frozen_in_folds"]),
                                            "proposers": sorted(h["proposers"])})
    if write:
        req = run_dir / "crossfold_requests"
        req.mkdir(parents=True, exist_ok=True)
        for c in cells:
            c["score_frozen"] = str(req / f"{c['family']}_f{c['fold']}_{_safe_name(c['target'])}.json")
            Path(c["score_frozen"]).write_text(json.dumps(c["hypotheses"], indent=1))
    return {"pooling": pooling_mode(discovery), "n_cells": len(cells),
            "n_hypothesis_fold_pairs": sum(len(c["hypotheses"]) for c in cells), "cells": cells,
            "not_scoreable": unmatched}


def _resolved_target(receipt: Mapping[str, Any]) -> str:
    # Rank-labelled programme targets are keyed on the programme the fold used.
    for field, tag in _PROGRAM_TARGETS:
        info = receipt.get(field) or {}
        if info.get("program_name"):
            return f"{tag}:{info['program_name']}"
    return str(receipt["target"])


BASELINE_PASS_FIELDS = ("known_folds", "decomp_folds", "passes_B_nuis", "passes_B_known", "passes_B_decomp",
                        "passes_B_known_and_B_decomp")
BASELINE_COUNT_KEYS = ("n_passes_B_nuis", "n_passes_B_known", "n_passes_B_decomp", "n_passes_B_known_and_B_decomp")
BASELINE_PASS_RULE = ("reported, never gated: B_nuis = the internal rule; for B_known (B_decomp) the same rule with "
                      "the passing folds restricted to folds whose gain_B_known CI lower bound is > 0 (whose row is "
                      "higher_order): >= 3 such folds of n_folds, every required structured null p <= 0.05 in them, "
                      "BH q <= fdr within family (the same q); the mean-gain floor applies to the B_nuis gain only")


def _baseline_passes(g: dict, n_folds: int, fdr: float, fam: dict, internal: bool) -> None:
    passing = [k for k, f in g["folds"].items() if _fold_passes(f)]
    g["known_folds"] = sorted(k for k in passing if (g["folds"][k].get("gain_known_ci_lo") or 0) > 0)
    g["decomp_folds"] = sorted(k for k in passing if g["folds"][k].get("higher_order") is True)
    nulls_ok = g.get("all_nulls_present") and g["worst_null_p"] <= 0.05
    base_ok = bool(nulls_ok) and g["q_value"] <= fdr
    g["passes_B_nuis"] = bool(internal)
    g["passes_B_known"] = base_ok and len(g["known_folds"]) >= 3
    g["passes_B_decomp"] = base_ok and len(g["decomp_folds"]) >= 3
    g["passes_B_known_and_B_decomp"] = g["passes_B_known"] and g["passes_B_decomp"]
    for k, field in zip(BASELINE_COUNT_KEYS, BASELINE_PASS_FIELDS[2:]):
        fam[k] = fam.get(k, 0) + int(g[field])


def _fold_record(row: Mapping[str, Any]) -> dict[str, Any]:
    r2_full = row.get("replication_r2_full")
    if r2_full is None:
        r2_full = (row.get("r2") or {}).get("replication_full")
    ci = row.get("replication_gain_ci")
    ci_lo = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None else None
    proposer = str(row.get("proposer") or "unknown")
    return {"replication_gain": float(row["replication_gain"]),
            "replication_r2_full": None if r2_full is None else float(r2_full),
            "gain_ci_lo": ci_lo, "validated": row.get("validated"),
            "nulls": {k: (v.get("p") if isinstance(v, Mapping) else v) for k, v in (row.get("nulls") or {}).items()},
            "proposers": {proposer, *map(str, row.get("also_proposed_by") or ())},
            "provenance": {proposer: row.get("provenance")},
            "recurrent_in_generator": row.get("recurrent_in_generator"),
            "explanation_level": (row.get("explanation") or {}).get("level"),
            "ladder": row.get("ladder"),
            **({"gain_known_ci_lo": _ci_lo(row.get("gain_B_known_ci"))} if "gain_B_known_ci" in row else {})}


def _ci_lo(ci):
    return float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None else None


def _merge_fold(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two rows for the same hypothesis in one fold (one per proposer): keep the
    least favourable numbers so a duplicate can never manufacture a pass."""
    def low(x, y):
        return None if x is None or y is None else min(x, y)
    nulls = {k: max([p for p in (a["nulls"].get(k), b["nulls"].get(k)) if p is not None], default=None)
             for k in sorted(set(a["nulls"]) | set(b["nulls"]))}
    validated = (None if a["validated"] is None and b["validated"] is None
                 else a["validated"] is not False and b["validated"] is not False)
    known = ({"gain_known_ci_lo": low(a.get("gain_known_ci_lo"), b.get("gain_known_ci_lo"))}
             if "gain_known_ci_lo" in a or "gain_known_ci_lo" in b else {})
    return {**a, **known, "replication_gain": min(a["replication_gain"], b["replication_gain"]),
            "replication_r2_full": low(a["replication_r2_full"], b["replication_r2_full"]),
            "gain_ci_lo": low(a["gain_ci_lo"], b["gain_ci_lo"]), "validated": validated, "nulls": nulls,
            "proposers": a["proposers"] | b["proposers"], "provenance": {**a["provenance"], **b["provenance"]},
            "explanation_level": a["explanation_level"] or b["explanation_level"],
            "ladder": _least_ladder(a.get("ladder"), b.get("ladder"))}


def _least_ladder(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Of two ladder blocks for one hypothesis in one fold, the one with the smaller gain at N."""
    if a is None or b is None:
        return a if b is None else b
    return a if a["levels"]["N"]["gain"] <= b["levels"]["N"]["gain"] else b


#: Tiers that can reach ``internal`` under a ladder protocol (simple associations are reported
#: at their own tier and enter the ledger at ``proposed``; nothing is dropped).
LADDER_INTERNAL_TIERS = ("irreducible", "partly absorbed")
_TIER_ORDER = ("irreducible", "partly absorbed", "simple association (novelty unchecked)", "none")


def _pool_ladder(folds: Mapping[int, Mapping[str, Any]], ladder_cfg: Mapping[str, Any], seed_key: str) -> dict:
    """Pooled ladder of one hypothesis over the folds whose receipts carry its ladder block.

    As ``dbench.scripts.decomposition_ladder pool``: each fold's held-out patients are
    predicted once, so the pooled Delta_k is the R^2 difference of the stacked out-of-fold
    predictions (full model vs competitor at level k) and its CI resamples whole patients
    (``dbench.decomposition.pool_folds``). Every null is pooled the same way: draw b's pooled
    gain is the sum over folds of (fold null gain x fold SST) / pooled SST -- exact for nulls
    that keep each patient's response values (within-patient permutation, within-patient block
    swap, constituent break). The tier follows the protocol's tier rules on the pooled passes;
    the folds passing at each level are reported beside it, never as a gate."""
    import numpy as np
    from . import decomposition as dc

    results = {int(f): r["ladder"] for f, r in folds.items() if r.get("ladder")}
    levels = [lv for lv in dc.LEVELS if all(lv in res["levels"] for res in results.values())]
    pooled = dc.pool_folds(results, levels=levels, n_boot=int(ladder_cfg.get("pooled_bootstrap_draws", 1000)),
                           seed=dc.hypothesis_seed(dc.content_hash("pool", seed_key)))

    def sst(stats):
        n = sum(v["n"] for v in stats); sy = sum(v["sy"] for v in stats); yy = sum(v["yy"] for v in stats)
        return yy - sy * sy / n if n else 0.0
    fold_sst = {f: sst(res["y_by_patient"].values()) for f, res in results.items()}
    total_sst = sst([v for res in results.values() for v in res["y_by_patient"].values()])
    rule = dict(dc.DEFAULT_PASS_RULE, **(ladder_cfg.get("pass_rule") or {}))
    out_levels = {}
    for lv in levels:
        e = pooled["levels"][lv]
        kinds = set.intersection(*(set(res["levels"][lv]["nulls"]) for res in results.values()))
        nulls = {}
        for kind in sorted(kinds):
            draws = [res["levels"][lv]["nulls"][kind].get("gains") for res in results.values()]
            if any(d is None for d in draws) or total_sst <= 1e-15:
                continue
            b = min(len(d) for d in draws)
            null = sum(np.asarray(res["levels"][lv]["nulls"][kind]["gains"][:b], float) * fold_sst[f]
                       for f, res in results.items()) / total_sst
            nulls[kind] = {"p": float((1.0 + np.sum(null >= e["delta"])) / (b + 1.0)), "n": int(b)}
        entry = {"gain": e["delta"], "ci": e["delta_ci95"], "r2_heldout": e["r2_with_H"],
                 "r2_competitor_heldout": e["r2_competitor"],
                 "linear_part": {"gain": e["delta_linear"], "ci": e["delta_linear_ci95"]},
                 "beyond_linear": {"gain": e["delta_beyond_linear"], "ci": e["delta_beyond_linear_ci95"]},
                 "shape": dc.shape_label(e["delta_linear_ci95"], e["delta_beyond_linear_ci95"]), "nulls": nulls,
                 "folds_passing": sorted(f for f, res in results.items() if res["levels"][lv].get("pass"))}
        decision = dc.level_passes(entry, rule)
        entry["pass"], entry["pass_checks"] = decision["pass"], decision["checks"]
        entry["n_folds_passing"] = len(entry["folds_passing"])
        out_levels[lv] = entry
    tier = dc.tier_from_levels(out_levels, ladder_cfg.get("tiers"))
    flags = [res.get("flags") or {} for res in results.values()]
    overlap = [f.get("target_programme_overlap") for f in flags if f.get("target_programme_overlap") is not None]
    leak = [f.get("leak_guard_max_abs_r") for f in flags if f.get("leak_guard_max_abs_r") is not None]
    return {"folds": sorted(results), "n_patients": pooled["n_patients"], "n_boot": pooled["n_boot"],
            "levels": out_levels, **tier,
            "recurrence_note": "folds passing per level reported beside the pooled tier; not a gate",
            "flags": {"target_programme_overlap": any(overlap) if overlap else None,
                      "leak_guard_max_abs_r": max(leak) if leak else None,
                      "leak_guarded_in_v05": (max(leak) >= 0.9) if leak else None}}


def _ladder_rank(g: Mapping[str, Any]) -> tuple:
    """Tier order, then the pooled gain at the highest passing level (B3 > B2 > B1 > N)."""
    pl = g["pooled_ladder"]
    passing = [lv for lv in ("B3", "B2", "B1", "N") if lv in pl["levels"] and pl["levels"][lv]["pass"]]
    gain = pl["levels"][passing[0]]["gain"] if passing else float("-inf")
    return (_TIER_ORDER.index(pl["tier"]), -gain, g["variable_names"], g["target"], g["feature_form"])


def _fold_passes(f: Mapping[str, Any]) -> bool:
    """Held-out full-model R2 > 0 AND gain CI lower bound > 0 (AND validated,
    when the receipt states it). A missing R2 or CI is not a pass."""
    return (f["replication_r2_full"] is not None and f["replication_r2_full"] > 0
            and f["gain_ci_lo"] is not None and f["gain_ci_lo"] > 0 and f["validated"] is not False)


def _cid_v04(campaign_id: str, g: Mapping[str, Any]) -> str:
    blob = json.dumps([g["family"], g["target"], g["feature_form"], g["variable_names"]], sort_keys=True).encode()
    return f"{campaign_id}-{g['family']}-{hashlib.sha256(blob).hexdigest()[:12]}"


def _aggregate_families(protocol: Mapping[str, Any], manifest: Mapping[str, Any], run_dir: Path,
                        receipts_dir: str | Path | None, *, dry_run: bool, allow_partial: bool) -> dict[str, Any]:
    """v0.4 funnel: relationship families, both proposers pooled, PASSING folds.

    Identity is (family, resolved target, feature_form, sorted variable_names);
    rows from every proposer merge into one hypothesis. A fold counts toward
    recurrence only if it passes (held-out full R2 > 0, gain CI > 0, validated
    when stated). Per family the top `budget` hypotheses with >= 2 passing
    folds by mean gain over passing folds are frozen at `proposed`;
    `internal` needs >= 3 passing folds, mean passing gain >= advance_rule.min_gain,
    every structured null p <= 0.05 in every passing fold, and BH q <= 0.10
    within the family's frozen set on the worst null p. Nothing else gates.
    """
    discovery = protocol["discovery"]
    families = _family_targets(discovery)
    n_folds = int(discovery["n_folds"])
    min_gain, fdr = float(discovery["advance_rule"]["min_gain"]), 0.10
    lifted = budget_lifted(discovery)
    budgets = {f: (None if lifted else int(discovery["relationship_families"][f].get("budget", discovery["candidate_budget"])))
               for f in families}
    search_budget = discovery.get("max_candidates", discovery.get("proposals_per_target"))
    # Optional: only these null kinds gate `internal`; every other recorded kind
    # is a per-fold diagnostic (e.g. an invariance test whose p sits near 1 for
    # exactly the site-independent relations we want). Absent -> all kinds gate.
    required = (discovery.get("null_suite") or {}).get("required_for_internal")
    required = None if required is None else {str(k) for k in required}
    receipts_dir = Path(receipts_dir) if receipts_dir else run_dir / "receipts"
    ladder_cfg = discovery.get("ladder") or {}

    groups: dict[tuple, dict[str, Any]] = {}
    seen: dict[str, set[tuple[int, str]]] = {f: set() for f in families}
    rows_without_names = 0
    episodes: set[str] = set()
    # Relation v07_ho rows carry ``higher_order`` (G_HO interval above 0); tallied beside
    # ``validated`` only when some row carries it, so earlier campaigns pool byte-identically.
    any_higher_order = False
    # v03_rff rows carry gain_B_known (+CI): the four baseline pass counts are then reported (never gated)
    any_known = False
    for family in families:
        for path in sorted((receipts_dir / family).glob("f*_*.json")):
            receipt = json.loads(path.read_text())
            if receipt.get("protocol_sha256") != manifest["protocol_sha256"]:
                raise CampaignProtocolError(f"receipt {path.name} carries a foreign protocol hash")
            if receipt.get("family") not in (None, family):
                raise CampaignProtocolError(f"receipt {path} declares family {receipt.get('family')!r}")
            fold = int(receipt["fold"])
            episodes.add(str(receipt.get("episode")))
            seen[family].add((fold, str(receipt["target"])))
            target = _resolved_target(receipt)
            for row in receipt["candidates"]:
                names = tuple(sorted(str(v) for v in row.get("variable_names") or ()))
                if not names:
                    rows_without_names += 1
                    continue
                form = str(row.get("feature_form"))
                g = groups.setdefault((family, target, form, names), {
                    "family": family, "target": target, "feature_form": form, "variable_names": list(names),
                    "gene_symbols": set(), "folds": {}, "receipts": set(), "baseline_width": row.get("baseline_width")})
                g["gene_symbols"].update(str(s) for s in row.get("gene_symbols") or ())
                g["receipts"].add(str(path))
                rec = _fold_record(row)
                any_known = any_known or "gain_known_ci_lo" in rec
                if "higher_order" in row:
                    any_higher_order = True
                    rec["higher_order"] = bool(row["higher_order"])
                if fold in g["folds"]:
                    merged = _merge_fold(g["folds"][fold], rec)
                    if "higher_order" in rec or "higher_order" in g["folds"][fold]:
                        # least favourable, like every other merged field
                        merged["higher_order"] = (bool(g["folds"][fold].get("higher_order"))
                                                  and bool(rec.get("higher_order")))
                    rec = merged
                g["folds"][fold] = rec
    all_folds = pooling_mode(discovery) == "all_folds"
    crossfold_seen = 0
    if all_folds:
        # The registered cross-fold scoring pass: rows of frozen hypotheses scored in the folds
        # where they were not frozen (receipts/<family>/crossfold/), same relation and design.
        for g in groups.values():
            for f in g["folds"].values():
                f["frozen_in_fold"] = True
        for family in families:
            for path in sorted((receipts_dir / family / "crossfold").glob("f*_*.json")):
                receipt = json.loads(path.read_text())
                if receipt.get("protocol_sha256") != manifest["protocol_sha256"]:
                    raise CampaignProtocolError(f"crossfold receipt {path.name} carries a foreign protocol hash")
                crossfold_seen += 1
                fold = int(receipt["fold"])
                target = _resolved_target(receipt)
                for row in receipt["candidates"]:
                    names = tuple(sorted(str(v) for v in row.get("variable_names") or ()))
                    g = groups.get((family, target, str(row.get("feature_form")), names))
                    if g is None or fold in g["folds"]:
                        continue  # frozen in that fold already: its own receipt row is the record
                    rec = _fold_record(row)
                    rec["frozen_in_fold"] = False
                    if "higher_order" in row:
                        rec["higher_order"] = bool(row["higher_order"])
                    g["folds"][fold] = rec
                    g["receipts"].add(str(path))
    expected = {f: {(k, t) for k in range(n_folds) for t in ts} for f, ts in families.items()}
    missing = {f: sorted(expected[f] - seen[f]) for f in families}
    if any(missing.values()) and not allow_partial:
        raise CampaignProtocolError("receipt matrix incomplete: " + json.dumps({f: len(m) for f, m in missing.items()}))

    for g in groups.values():
        g["gene_symbols"] = sorted(g["gene_symbols"])
        g["proposers"] = sorted(set().union(*(f["proposers"] for f in g["folds"].values())))
        g["n_proposed_folds"] = len(g["folds"])
        passing = {k: f for k, f in g["folds"].items() if _fold_passes(f)}
        g["passing_folds"] = sorted(passing)
        g["n_passing_folds"] = len(passing)
        gains = [f["replication_gain"] for f in passing.values()]
        g["mean_replication_gain"] = sum(gains) / len(gains) if gains else None
        if required is None:
            ps = [p for f in passing.values() for p in f["nulls"].values() if p is not None]
            g["worst_null_p"] = max(ps) if ps else 1.0
            g["all_nulls_present"] = bool(passing) and all(f["nulls"] for f in passing.values())
        else:
            # every required kind the fold recorded must be <= 0.05; at least one must be recorded
            ps = [p for f in passing.values() for k, p in f["nulls"].items() if k in required and p is not None]
            g["worst_null_p"] = max(ps) if ps else 1.0
            g["all_nulls_present"] = bool(passing) and all(
                any(f["nulls"].get(k) is not None for k in required) for f in passing.values())
            kinds = sorted({k for f in g["folds"].values() for k in f["nulls"]} - required)
            g["diagnostic_nulls"] = {k: {str(fold): f["nulls"].get(k) for fold, f in sorted(g["folds"].items())}
                                     for k in kinds}
            # gain changes when that structure is broken: a Level D hint, never a drop
            g["environment_dependent"] = any(f["nulls"].get(k) is not None and f["nulls"][k] <= 0.05
                                             for f in passing.values() for k in kinds)
        if any_higher_order:
            g["higher_order_folds"] = sorted(k for k, f in passing.items() if f.get("higher_order") is True)
        levels = sorted({str(f["explanation_level"]) for f in passing.values() if f["explanation_level"]})
        g["explanation_level"] = "/".join(levels) or "pending"
        if any(f.get("ladder") for f in g["folds"].values()):
            g["pooled_ladder"] = _pool_ladder(g["folds"], ladder_cfg, _cid_v04(manifest["campaign_id"], g))
            b3 = g["pooled_ladder"]["levels"].get("B3", {}).get("nulls", {})
            ps = [v["p"] for k, v in b3.items() if required is None or k in required]
            g["ladder_worst_b3_null_p"] = max(ps) if ps else 1.0

    # Patient-level CPTAC receipts are labelled as such; every other campaign keeps the HEST labels.
    patient_level = episodes == {"cptac_patient"}
    cohort_label = "cptac_ccrcc" if patient_level else "hest_ccrcc"
    level_meaning = ("held-out CPTAC patients (patient level, same acquisition)" if patient_level
                     else "held-out HEST patients (same acquisition)")
    frozen: list[dict[str, Any]] = []
    per_family: dict[str, dict[str, Any]] = {}
    q_values: dict[str, float] = {}
    has_ladder = any("pooled_ladder" in g for g in groups.values())
    for family, budget in budgets.items():
        fam = [g for g in groups.values() if g["family"] == family]
        if has_ladder:
            # v0.6: frozen = every hypothesis with a pooled tier other than "none", tier first,
            # then pooled gain at its highest passing level; the full ladder of EVERY hypothesis
            # (tier "none" included) is reported under summary["ladder"].
            recurring = [g for g in fam if "pooled_ladder" in g and g["pooled_ladder"]["tier"] != "none"]
            recurring.sort(key=_ladder_rank)
        else:
            recurring = [g for g in fam if g["n_passing_folds"] >= (3 if lifted else 2)]
            recurring.sort(key=lambda g: (-g["mean_replication_gain"], -g["n_passing_folds"], g["variable_names"],
                                          g["target"], g["feature_form"]))
        chosen = recurring if lifted else recurring[:budget]
        if chosen:   # BH within family, over that family's frozen set
            null_key = "ladder_worst_b3_null_p" if has_ladder else "worst_null_p"
            q_values.update(_bh_q({_cid_v04(manifest["campaign_id"], g): g[null_key] for g in chosen}))
        frozen.extend(chosen)
        per_family[family] = {"n_groups": len(fam), "n_frozen": len(chosen), "n_internal": 0,
                              "below_recurrence": len(fam) - len(recurring), "budget": budget}
        if any_higher_order:
            # General discoveries (validated) and higher-order discoveries (validated AND the
            # G_HO interval above 0) side by side: fold rows, then frozen hypotheses whose
            # passing folds include >= 2 higher-order folds (the freezing recurrence rule).
            rows_f = [f for g in fam for f in g["folds"].values()]
            per_family[family].update({
                "n_fold_rows": len(rows_f),
                "n_fold_rows_validated": sum(f["validated"] is True for f in rows_f),
                "n_fold_rows_higher_order": sum(f["validated"] is True and f.get("higher_order") is True
                                                for f in rows_f),
                "n_frozen_higher_order": sum(len(g["higher_order_folds"]) >= 2 for g in chosen)})

    ledger = None if dry_run else CandidateLedger(str(run_dir / protocol["ledger"]["filename"]))
    table = []
    for g in frozen:
        cid = _cid_v04(manifest["campaign_id"], g)
        g["q_value"] = q_values[cid]
        reasons = []
        if has_ladder:
            tier = g["pooled_ladder"]["tier"]
            if tier not in LADDER_INTERNAL_TIERS:
                reasons.append(f"pooled tier {tier!r} (reported as a discovery of its own tier)")
            if g["ladder_worst_b3_null_p"] > 0.05:
                reasons.append(f"pooled B3 structured null p {g['ladder_worst_b3_null_p']:.3f} > 0.05")
        else:
            if g["n_passing_folds"] < 3:
                reasons.append(f"passing folds {g['n_passing_folds']}/{n_folds} < 3")
            if g["mean_replication_gain"] < min_gain:
                reasons.append(f"mean passing-fold gain {g['mean_replication_gain']:.4f} < {min_gain}")
            if not g["all_nulls_present"] or g["worst_null_p"] > 0.05:
                reasons.append(f"structured null p {g['worst_null_p']:.3f} > 0.05")
        if g["q_value"] > fdr:
            reasons.append(f"BH q {g['q_value']:.3f} > {fdr}")
        stage = "internal" if not reasons else "proposed"
        per_family[g["family"]]["n_internal"] += stage == "internal"
        if any_known:
            _baseline_passes(g, n_folds, fdr, per_family[g["family"]], stage == "internal")
        folds = sorted(g["folds"].items())
        evidence = {"fold_gains": {str(k): v["replication_gain"] for k, v in folds},
                    "fold_r2_full": {str(k): v["replication_r2_full"] for k, v in folds},
                    "fold_gain_ci_lo": {str(k): v["gain_ci_lo"] for k, v in folds},
                    "null_p_by_fold": {str(k): v["nulls"] for k, v in folds},
                    "passing_folds": g["passing_folds"], "n_proposed_folds": g["n_proposed_folds"],
                    "recurrence": g["n_passing_folds"], "n_folds": n_folds,
                    "mean_replication_gain": g["mean_replication_gain"], "worst_null_p": g["worst_null_p"],
                    "q": g["q_value"], "family": g["family"], "target": g["target"],
                    "feature_form": g["feature_form"], "variable_names": g["variable_names"],
                    "gene_symbols": g["gene_symbols"], "proposers": g["proposers"],
                    "proposal_mechanism": "+".join(g["proposers"]),
                    "provenance_by_fold": {str(k): v["provenance"] for k, v in folds},
                    **({"pooling": "all_folds",
                        "frozen_folds": sorted(k for k, v in folds if v.get("frozen_in_fold", True))}
                       if all_folds else {}),
                    "recurrent_in_generator_by_fold": {str(k): v["recurrent_in_generator"] for k, v in folds},
                    "explanation_level": g["explanation_level"], "baseline_width": g["baseline_width"],
                    **({} if required is None else {"required_nulls": sorted(required),
                                                    "diagnostic_nulls": g["diagnostic_nulls"],
                                                    "environment_dependent": g["environment_dependent"]}),
                    "search_budget": None if search_budget is None else int(search_budget),
                    "protocol_sha256": manifest["protocol_sha256"], "replication_level": 1,
                    "replication_level_meaning": level_meaning}
        row = {"operator": "flexible_product" if g["feature_form"] == "product" else "flexible_subspace",
               "variables": g["variable_names"], "parameters": {"feature_form": g["feature_form"]},
               "baseline_width": g["baseline_width"],
               "context": {"target": g["target"], "family": g["family"], "cohort": cohort_label}}
        if all_folds:
            g["frozen_folds"] = sorted(k for k, v in g["folds"].items() if v.get("frozen_in_fold", True))
        entry = {"candidate_id": cid, "stage": stage, "family": g["family"], "target": g["target"],
                 "feature_form": g["feature_form"], "variable_names": g["variable_names"],
                 "gene_symbols": g["gene_symbols"], "proposers": g["proposers"],
                 "passing_folds": g["passing_folds"], "n_proposed_folds": g["n_proposed_folds"],
                 "mean_replication_gain": (None if g["mean_replication_gain"] is None
                                           else round(g["mean_replication_gain"], 5)),
                 "worst_null_p": g["worst_null_p"],
                 "q_value": round(g["q_value"], 4), "explanation_level": g["explanation_level"],
                 "stop_reason": "; ".join(reasons) or None,
                 **({"frozen_folds": g["frozen_folds"], "n_scored_folds": len(g["folds"])} if all_folds else {})}
        if any_known:
            entry.update({k: g[k] for k in BASELINE_PASS_FIELDS})
            evidence["baseline_passes"] = {k: g[k] for k in BASELINE_PASS_FIELDS}
        if any_higher_order:
            entry["higher_order_folds"] = g["higher_order_folds"]
            entry["higher_order"] = len(g["higher_order_folds"]) >= 2
            evidence["higher_order_folds"] = g["higher_order_folds"]
        if has_ladder:
            entry["ladder_tier"] = g["pooled_ladder"]["tier"]
            entry["pooled_ladder"] = g["pooled_ladder"]
            evidence["pooled_ladder"] = g["pooled_ladder"]
        if required is not None:
            entry.update(diagnostic_nulls=g["diagnostic_nulls"], environment_dependent=g["environment_dependent"])
        table.append(entry)
        if ledger is not None:
            from .ledger import record_from_receipt
            data_ref = ";".join(sorted(g["receipts"]))
            record_from_receipt(ledger, cid, "proposed", row, evidence=evidence, data_ref=data_ref,
                                stop_reason=entry["stop_reason"],
                                notes="frozen by aggregate_funnel (v0.4 passing-fold rule) under the registered protocol")
            if stage == "internal":
                record_from_receipt(ledger, cid, "internal", row, evidence=evidence, data_ref=data_ref,
                                    notes="advance rule satisfied: passing folds, mean gain, nulls, BH q within family")

    def mentions(g, gene):
        return gene in g["gene_symbols"] or any(v == gene or v.endswith(":" + gene) for v in g["variable_names"])
    legacy = {}
    for gene in map(str, (protocol.get("legacy_leads") or {}).get("genes", ())):
        legacy[gene] = {"n_groups_mentioning": sum(1 for g in groups.values() if mentions(g, gene)),
                        "frozen": [{"candidate_id": e["candidate_id"], "family": e["family"], "target": e["target"],
                                    "n_passing_folds": len(e["passing_folds"]),
                                    "mean_replication_gain": e["mean_replication_gain"], "stage": e["stage"]}
                                   for g, e in zip(frozen, table) if mentions(g, gene)]}
    controls = _positive_control_check(protocol, groups, {_cid_v04(manifest["campaign_id"], g): e["stage"]
                                                         for g, e in zip(frozen, table)}, manifest["campaign_id"])
    summary = {"schema": "morpheus_funnel_aggregate/v2", "campaign_id": manifest["campaign_id"],
               "protocol_sha256": manifest["protocol_sha256"], "receipts_seen": {f: len(s) for f, s in seen.items()},
               "missing_cells": missing, "families": per_family, "n_groups": len(groups), "n_frozen": len(frozen),
               "n_internal": sum(1 for e in table if e["stage"] == "internal"), "min_gain": min_gain, "fdr": fdr,
               "rows_without_variable_names": rows_without_names,
               **({"pooling": {"mode": "all_folds", "crossfold_receipts_seen": crossfold_seen,
                               "frozen_groups_scored_in_every_fold": sum(len(g["folds"]) == n_folds for g in frozen),
                               "n_frozen": len(frozen)}} if all_folds else {}),
               "legacy_lead_check": {"note": "informational only; nothing is excluded", "genes": legacy},
               "positive_control_check": controls,
               **({"ladder": _ladder_summary(groups, frozen, table, manifest["campaign_id"])} if has_ladder else {}),
               **({"baseline_pass_counts": {
                   "rule": BASELINE_PASS_RULE,
                   "cohort": {k: sum(per_family[f].get(k, 0) for f in per_family) for k in BASELINE_COUNT_KEYS},
                   "by_family": {f: {k: per_family[f].get(k, 0) for k in BASELINE_COUNT_KEYS} for f in per_family}}}
                  if any_known else {}),
               "dry_run": dry_run, "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "table": table}
    if not dry_run:
        (run_dir / "funnel_aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _ladder_summary(groups: Mapping[tuple, Mapping[str, Any]], frozen: list, table: list, campaign_id: str) -> dict:
    """Every hypothesis's pooled ladder (frozen or not, tier "none" included) and tier counts."""
    stage = {e["candidate_id"]: e["stage"] for e in table}
    rows, counts = [], {}
    for g in sorted(groups.values(), key=lambda g: (g["family"], g["target"], g["variable_names"], g["feature_form"])):
        if "pooled_ladder" not in g:
            continue
        cid = _cid_v04(campaign_id, g)
        pl = g["pooled_ladder"]
        rows.append({"candidate_id": cid, "family": g["family"], "target": g["target"],
                     "feature_form": g["feature_form"], "variable_names": g["variable_names"],
                     "proposers": g["proposers"], "stage": stage.get(cid, "reported (not in ledger)"),
                     "tier": pl["tier"], "absorbing_level": pl["absorbing_level"],
                     "absorbing_explanation": pl["absorbing_explanation"], "folds": pl["folds"],
                     "n_patients": pl["n_patients"],
                     "folds_passing": {lv: e["folds_passing"] for lv, e in pl["levels"].items()},
                     "flags": pl["flags"], "pooled_ladder": pl})
        fam = counts.setdefault(g["family"], {"total": {}, "by_target": {}})
        fam["total"][pl["tier"]] = fam["total"].get(pl["tier"], 0) + 1
        tgt = fam["by_target"].setdefault(g["target"], {})
        tgt[pl["tier"]] = tgt.get(pl["tier"], 0) + 1
    return {"pooling": ("pooled out-of-fold Delta_k per level over the folds carrying the hypothesis, patient "
                        "bootstrap; nulls pooled the same way; tier from the protocol's tier rules on the pooled "
                        "passes; folds passing per level reported, not gating"),
            "internal_tiers": list(LADDER_INTERNAL_TIERS), "tier_counts": counts, "hypotheses": rows}


# Encoded readings of the protocol's positive-control `expected` text, by
# control name. Informational only: no stage depends on them.
_CONTROL_RULES = {
    # >= 1 IG-containing hypothesis passing in >= 2 folds for cv_frac_Inflammatory or a morph_pc target
    "immunoglobulin_to_inflammatory_structure": lambda hits: any(
        h["n_passing_folds"] >= 2 and (h["target"] == "cv_frac_Inflammatory" or h["target"].startswith("morph_pc"))
        for h in hits),
    # >= 1 proposed hypothesis carrying an Inflammatory(-fraction or neighbour) context variable
    "chemokine_niche_context": lambda hits: any(
        any("Inflammatory" in v for v in h["variable_names"]) for h in hits),
}


def _positive_control_check(protocol: Mapping[str, Any], groups: Mapping[tuple, Mapping[str, Any]],
                            stage_by_cid: Mapping[str, str], campaign_id: str) -> dict[str, Any]:
    block = protocol.get("positive_controls") or {}
    out = {}
    for control in block.get("controls", ()):
        name = str(control.get("name"))
        genes = {str(g) for g in control.get("genes") or ()}
        if not genes:   # e.g. a planted, development-only control
            out[name] = {"expected": control.get("expected"), "expected_met": "not_evaluated_here", "hits": []}
            continue
        families = set(control.get("family") or ())
        hits = []
        for g in groups.values():
            if (families and g["family"] not in families) or not genes & set(g["gene_symbols"]):
                continue
            cid = _cid_v04(campaign_id, g)
            hits.append({"candidate_id": cid, "family": g["family"], "target": g["target"],
                         "variable_names": g["variable_names"], "proposers": g["proposers"],
                         "n_proposed_folds": g["n_proposed_folds"], "n_passing_folds": g["n_passing_folds"],
                         "mean_replication_gain": g["mean_replication_gain"],
                         "stage": stage_by_cid.get(cid, "not_frozen")})
        hits.sort(key=lambda h: (-h["n_passing_folds"], h["family"], h["target"], h["variable_names"]))
        rule = _CONTROL_RULES.get(name)
        out[name] = {"expected": control.get("expected"), "hits": hits,
                     "expected_met": rule(hits) if rule else "not_encoded"}
    return {"note": "informational only; changes no stage", "controls": out}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--refresh-preflight", action="store_true")
    parser.add_argument("--aggregate", action="store_true", help="fold receipts -> frozen candidates -> ledger")
    parser.add_argument("--receipts", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="dry-run only: aggregate an incomplete matrix")
    parser.add_argument("--crossfold-requests", action="store_true",
                        help="all-fold pooling: write the cross-fold scoring requests (frozen hypotheses x folds)")
    args = parser.parse_args()
    if args.crossfold_requests:
        result = crossfold_requests(args.protocol, args.run_dir, args.receipts)
        result = {k: v for k, v in result.items() if k != "cells"} | {
            "cells": [{k: v for k, v in c.items() if k != "hypotheses"} for c in result["cells"]]}
    elif args.aggregate:
        if args.allow_partial and not args.dry_run:
            parser.error("--allow-partial is permitted only with --dry-run")
        result = aggregate_funnel(args.protocol, args.run_dir, args.receipts, dry_run=args.dry_run,
                                  allow_partial=args.allow_partial)
    else:
        operation = refresh_preflight if args.refresh_preflight else initialise
        result = operation(args.protocol, args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
