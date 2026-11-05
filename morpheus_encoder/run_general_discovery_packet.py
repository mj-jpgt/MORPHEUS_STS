"""Run one observed-only MORPHEUS/SSL-Scientist discovery packet.

This is the v2 execution seam: every arm shares the same contextual observed
features, proposal budget, flexible relation family, proposal/replication/null
split, and candidate receipt format.  The runner does not open benchmark truth
and therefore cannot score object fidelity itself; a separate sealed evaluator
may attach that field after candidates have been frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from dbench.discovery_panel import build_panel_episode, panel_condition
from morpheus_encoder.capability_metrics import adjust_family_pvalues
from morpheus_encoder.discovery_representation import (
    IdentityPreservingEncoder, RepresentationConfig, fit_response_free, represent,
)
from morpheus_encoder.general_discovery import (
    contextual_feature_matrix, evaluate_frozen_candidate, evaluate_response_permutation_nulls,
    evaluate_structured_nulls, propose_candidates,
)


def _neighbour_array(world) -> list[np.ndarray] | None:
    values = getattr(world.structure, "neighbors", None)
    return None if values is None else [np.asarray(row, dtype=int) for row in values]


def _context(world) -> np.ndarray | None:
    pieces = []
    for value in (getattr(world.structure, "coords", None), getattr(world.structure, "time", None)):
        if value is not None:
            pieces.append(np.asarray(value, dtype=float).reshape(len(world.target), -1))
    return None if not pieces else np.column_stack(pieces)


def _own_observed_features(world) -> tuple[np.ndarray, tuple[dict, ...]]:
    """Return only an observation's own measurements to the encoder.

    Contextual transforms deliberately happen *after* representation.  Feeding
    precomputed neighbour features into an encoder and also asking it to learn
    a neighbourhood-prediction objective made the old v2 contextual arm
    partly tautological.
    """
    if "X" not in world.modalities:
        raise ValueError("v2 generic runner requires an observed X modality")
    value = np.asarray(world.modalities["X"], dtype=float)
    return value, tuple({"source": "observed", "feature": int(j)} for j in range(value.shape[1]))


def _standardise_train_apply(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True)
    sd[sd < 1e-7] = 1.0
    return tuple((value - mean) / sd for value in (train, *others))


def _pca_train_apply(train: np.ndarray, *others: np.ndarray, output_width: int) -> tuple[np.ndarray, ...]:
    """Deterministic capacity-matched PCA reference fit on proposal data only."""
    if output_width < 1:
        raise ValueError("output_width must be positive")
    centered = train - train.mean(0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:min(output_width, vt.shape[0])].T
    mean = train.mean(0, keepdims=True)
    return tuple((value - mean) @ basis for value in (train, *others))


def _contextualise_representation(z: np.ndarray, world) -> tuple[np.ndarray, tuple[dict, ...]]:
    """Derive observed spatial/temporal views only after encoding own features."""
    return contextual_feature_matrix(z, neighbours=_neighbour_array(world),
                                     time=getattr(world.structure, "time", None),
                                     unit=getattr(world.structure, "unit", None))


def _partition_neighbours(global_neighbours, rows: np.ndarray, n_rows: int,
                          *, device: torch.device) -> torch.Tensor | None:
    """Map a full-episode neighbour list into a proposal-partition tensor.

    The representation objective may use only proposal observations.  This
    remapping preserves real within-patient edges while replacing every edge
    to a replication observation with ``-1``; it prevents a contextual loss
    from reading held-out patients through the graph.  A missing graph is a
    legitimate non-spatial input and returns ``None``.
    """
    if global_neighbours is None:
        return None
    rows = np.asarray(rows, dtype=int)
    lookup = np.full(int(n_rows), -1, dtype=int)
    lookup[rows] = np.arange(len(rows), dtype=int)
    width = max((len(v) for v in global_neighbours), default=0)
    if width == 0:
        return None
    local = np.full((len(rows), width), -1, dtype=np.int64)
    for local_i, global_i in enumerate(rows):
        values = np.asarray(global_neighbours[int(global_i)], dtype=int)
        mapped = lookup[values]
        local[local_i, :len(mapped)] = mapped
    return torch.as_tensor(local, dtype=torch.long, device=device)


def _top_observed_links(representation: np.ndarray, observed: np.ndarray,
                        coordinates: tuple[int, ...], provenance: tuple[dict, ...],
                        *, top_k: int = 5) -> list[dict]:
    """Describe selected representation coordinates using observed data only.

    This lightweight decoder is not used to choose or validate candidates. It
    is written into an already-frozen receipt so a sealed evaluator and later
    biological characterization can identify which measured quantities each
    coordinate retained, without requiring relation labels or oracle latents.
    """
    centered_observed = observed - observed.mean(0, keepdims=True)
    observed_norm = np.sqrt(np.sum(centered_observed ** 2, axis=0)).clip(min=1e-12)
    links = []
    for coordinate in coordinates:
        value = representation[:, coordinate] - representation[:, coordinate].mean()
        denom = np.sqrt(np.sum(value ** 2)) * observed_norm
        corr = (value @ centered_observed) / denom
        order = np.argsort(-np.abs(corr), kind="stable")[:top_k]
        links.append({"representation_coordinate": int(coordinate), "top_observed": [
            {"observed_coordinate": int(index), "correlation": float(corr[index]),
             "provenance": dict(provenance[index])} for index in order
        ]})
    return links


def run_episode(candidate_X: np.ndarray, candidate_y: np.ndarray,
                replication_X: np.ndarray, replication_y: np.ndarray, *,
                max_candidates: int, model: str = "ridge",
                proposal_context: np.ndarray | None = None, replication_context: np.ndarray | None = None,
                proposal_baseline: np.ndarray | None = None, replication_baseline: np.ndarray | None = None,
                proposal_groups: np.ndarray | None = None,
                ) -> list[tuple]:
    """Array-level discovery core, shared by every arm and every episode kind.

    Proposes a bounded candidate family over ``(candidate_X, candidate_y)``
    and freezes/evaluates each one's independent incremental gain on
    ``(replication_X, replication_y)`` against a fixed nuisance ``baseline``.
    Returns ``[(candidate, receipt), ...]`` in proposal order (unranked,
    un-null-tested): a null suite is domain-owned (the panel wrapper keeps
    its own fast response-permutation null; the real-episode path attaches
    ``hest.episode_null_generators``), so this core stays agnostic to which
    one a caller needs.
    """
    candidates = propose_candidates(candidate_X, candidate_y, context=proposal_context,
                                    max_candidates=max_candidates, max_pairs=max_candidates // 2,
                                    max_subspaces=max_candidates // 4)
    return [(candidate, evaluate_frozen_candidate(
        candidate, candidate_X, candidate_y, replication_X, replication_y,
        proposal_context=proposal_context, replication_context=replication_context,
        proposal_baseline=proposal_baseline, replication_baseline=replication_baseline, model=model,
        proposal_groups=proposal_groups))
        for candidate in candidates]


def run_packet(*, condition_id: str, seed: int, arm: str, device: str,
               steps: int, output_width: int, max_candidates: int, n_null: int, min_gain: float,
               fdr: float, sidecar_path: Path | None = None) -> dict:
    """Execute one arm without inspecting a controlled packet's truth handle."""
    if arm not in {"source", "pca", "reconstruction", "ssl", "predictive", "morpheus_legacy", "morpheus_factorized", "morpheus"}:
        raise ValueError("unknown arm")
    condition = panel_condition(condition_id)
    episode = build_panel_episode(condition, seed)
    proposal = episode.D_P.blind()
    replication = episode.D_R[0].blind()
    null = episode.D_N[0].blind()
    p_raw, own_provenance = _own_observed_features(proposal)
    r_raw, _ = _own_observed_features(replication)
    n_raw, _ = _own_observed_features(null)
    p_x, r_x, n_x = _standardise_train_apply(p_raw, r_raw, n_raw)
    p_context, r_context, n_context = _context(proposal), _context(replication), _context(null)
    torch.manual_seed(seed)
    learned_width = min(int(output_width), p_x.shape[1])
    if learned_width < 1:
        raise ValueError("output_width must be positive")
    model = IdentityPreservingEncoder(RepresentationConfig(d_input=p_x.shape[1], d_output=learned_width,
                                                            mask_probability=0.15)).to(device)
    p_tensor = torch.as_tensor(p_x, dtype=torch.float32, device=device)
    neighbours = _neighbour_array(proposal)
    neighbour_tensor = None
    if neighbours is not None:
        width = max((len(row) for row in neighbours), default=0)
        neighbour_tensor = torch.full((len(neighbours), max(width, 1)), -1, dtype=torch.long, device=device)
        for row, ids in enumerate(neighbours):
            if len(ids):
                neighbour_tensor[row, :len(ids)] = torch.as_tensor(ids, device=device)
    started = time.time()
    if arm == "pca":
        z_own_p, z_own_r, z_own_n = _pca_train_apply(p_x, r_x, n_x, output_width=learned_width)
        losses = []
    else:
        losses = fit_response_free(model, p_tensor, arm, neighbours=neighbour_tensor, steps=steps) if arm != "source" else []
        with torch.no_grad():
            z_own_p = represent(model, p_tensor, arm).detach().cpu().numpy()
            z_own_r = represent(model, torch.as_tensor(r_x, dtype=torch.float32, device=device), arm).detach().cpu().numpy()
            z_own_n = represent(model, torch.as_tensor(n_x, dtype=torch.float32, device=device), arm).detach().cpu().numpy()
    # The discovery stack sees matching derived views for every arm.
    z_p, provenance = _contextualise_representation(z_own_p, proposal)
    z_r, _ = _contextualise_representation(z_own_r, replication)
    z_n, _ = _contextualise_representation(z_own_n, null)
    # This post-freeze decoder compares contextual learned coordinates to the
    # same contextual measured views; it is not used in fitting or ranking.
    p_observed, observed_provenance = contextual_feature_matrix(p_x, neighbours=_neighbour_array(proposal),
                                                                  time=getattr(proposal.structure, "time", None),
                                                                  unit=getattr(proposal.structure, "unit", None))
    r_observed, _ = contextual_feature_matrix(r_x, neighbours=_neighbour_array(replication),
                                               time=getattr(replication.structure, "time", None),
                                               unit=getattr(replication.structure, "unit", None))
    y_p, y_r, y_n = (np.asarray(proposal.target, dtype=float), np.asarray(replication.target, dtype=float),
                     np.asarray(null.target, dtype=float))
    rows = []
    for candidate, receipt in run_episode(z_p, y_p, z_r, y_r, max_candidates=max_candidates,
                                          model="rbf", proposal_context=p_context, replication_context=r_context):
        declared_null = evaluate_frozen_candidate(candidate, z_p, y_p, z_n, y_n,
                                                   proposal_context=p_context, replication_context=n_context,
                                                   model="rbf")
        null_receipt = evaluate_response_permutation_nulls(
            receipt, z_p, y_p, z_r, y_r, proposal_context=p_context, null_context=r_context,
            n_null=n_null, seed=seed * 100_003 + int(candidate.candidate_hash[:8], 16),
        )
        rows.append({
            "candidate_hash": candidate.candidate_hash,
            "variables": list(candidate.variables), "context_variables": list(candidate.context_variables),
            "feature_form": candidate.feature_form, "proposal_family": candidate.proposal_family,
            "proposal_score": candidate.proposal_score, "proposal_gain": receipt.proposal_gain,
            "replication_gain": receipt.replication_gain,
            "declared_null_gain": declared_null.replication_gain,
            "null_p_value": null_receipt.null_p_value,
            "null_gains": list(null_receipt.null_gains),
            "observed_coordinate_links": _top_observed_links(z_r, r_observed, candidate.variables, observed_provenance),
        })
    q_values = adjust_family_pvalues({row["candidate_hash"]: row["null_p_value"] for row in rows})
    ranked = sorted(rows, key=lambda row: (-row["proposal_gain"], row["candidate_hash"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["q_value"] = q_values[row["candidate_hash"]]
        row["validated"] = bool(row["replication_gain"] >= min_gain and row["q_value"] <= fdr)
        row["object_fidelity"] = None  # sealed evaluator supplies controlled-truth fidelity later.
    sidecar = None
    if sidecar_path is not None:
        # The sidecar contains only the independently generated replication
        # values of coordinates which the already-frozen candidate list uses.
        # It is never read by the runner itself.  Keeping this numeric payload
        # out of JSON makes exact post-freeze latent/component scoring feasible
        # for the controlled Tier-I worlds without turning receipts into giant
        # text files.
        used_coordinates = np.asarray(sorted({int(i) for row in ranked
                                               for i in tuple(row["variables"]) + tuple(row["context_variables"])}),
                                      dtype=np.int32)
        values = np.asarray(z_r[:, used_coordinates], dtype=np.float32)
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sidecar_path, coordinates=used_coordinates, values=values)
        digest = sha256(sidecar_path.read_bytes()).hexdigest()
        sidecar = {"path": str(sidecar_path), "sha256": digest,
                   "coordinates": int(len(used_coordinates)),
                   "rows": int(values.shape[0]), "kind": "replication_candidate_coordinate_values"}
    return {
        "schema": "morpheus_general_discovery_packet/v1",
        "condition": asdict(condition), "packet_id": f"{condition_id}/seed{seed:04d}",
        "arm": arm, "seed": int(seed), "method_truth_accessed": False,
        "representation": {"arm": arm, "steps": int(steps), "loss_trace": losses,
                             "input_width": int(p_x.shape[1]), "output_width": int(z_own_p.shape[1]),
                             "contextual_output_width": int(z_p.shape[1]),
                             "context_applied_after_encoding": True,
                             "all_coordinate_observed_links": _top_observed_links(
                                 z_r, r_observed, tuple(range(z_r.shape[1])), observed_provenance)},
        "proposal_budget": int(max_candidates), "null": {"kind": "predeclared_response_permutation",
                                                              "n_null": int(n_null), "fdr": float(fdr)},
        "validation_rule": {"min_replication_incremental_gain": float(min_gain), "q_value_max": float(fdr)},
        "feature_provenance": list(provenance), "own_feature_provenance": list(own_provenance), "candidates": ranked,
        "sealed_scoring_sidecar": sidecar,
        "elapsed_seconds": time.time() - started,
    }


def run_hest_episode(*, fold: int, n_folds: int, seed: int, target: str, arm: str, device: str,
                     protocol_path: Path | None, embed_root: str | None, data_dir: str | None,
                     sample_ids: list[int] | None, cohort: str, max_candidates: int, n_null: int,
                     min_gain: float, fdr: float, steps: int, n_hvg: int = 300, min_depth: int = 200,
                     program_sets: list | None = None) -> dict:
    """Execute one real ccRCC HEST discovery packet (D1-D4).

    A leak-proof grouped-patient fold (`hest.real_episode`), one predeclared
    target, the `source`/`morpheus_factorized` representation arm, and the
    hest-owned structured null suite (`hest.episode_null_generators`) --
    receipt schema v2: `episode_identity`, `gene_symbols`, `nulls{kind:{p,n}}`,
    `baseline_width`. Recurrence-across-folds and the ledger admission gate
    are the funnel aggregator's job, not this per-fold runner's.
    """
    from morpheus.tier2.substrates import hest

    protocol: dict = {}
    protocol_sha256 = None
    protocol_file_sha256 = None
    if protocol_path is not None:
        from dbench.prospective_campaign import canonical_hash
        protocol_bytes = Path(protocol_path).read_bytes()
        protocol_file_sha256 = sha256(protocol_bytes).hexdigest()
        protocol = json.loads(protocol_bytes)
        # The campaign manifest hashes the canonical JSON payload, not the file
        # bytes; receipts carry the same canonical hash so they lock to the
        # registered campaign (file hash kept beside it for provenance).
        protocol_sha256 = canonical_hash(protocol)
    # v0.2's schema nests everything under "discovery"; `advance_rule` stays
    # prose (min_gain=0.02/fdr=0.10 already match it by convention, not by
    # parsing English), so only the machine-readable fields are threaded.
    discovery_cfg = protocol.get("discovery", {})
    max_candidates = int(discovery_cfg.get("proposals_per_target", max_candidates))
    n_null = int(discovery_cfg.get("null_suite", {}).get("draws", n_null))
    n_folds = int(discovery_cfg.get("n_folds", n_folds))
    # `fold_seed` locks the PATIENT-FOLD assignment across every target/arm
    # this protocol runs; `seed` (the CLI/run argument) still seeds the
    # representation arm (e.g. `morpheus_factorized`'s encoder init).
    fold_seed = int(discovery_cfg.get("fold_seed", seed))
    n_hvg = int(protocol.get("n_hvg", n_hvg))
    min_depth = int(protocol.get("min_depth", min_depth))

    episode = hest.real_episode(fold=fold, n_folds=n_folds, seed=fold_seed, target=target,
                                embed_root=embed_root, sample_ids=sample_ids, cohort=cohort,
                                data_dir=data_dir, n_hvg=n_hvg, min_depth=min_depth,
                                program_sets=program_sets)
    proposal, replication = episode.partitions.proposal, episode.partitions.replication
    X = episode.observed_matrix()
    y = np.asarray(episode.response, dtype=float)
    baseline = np.asarray(episode.context["baseline"], dtype=float)
    p_x, r_x = X[proposal], X[replication]
    p_y, r_y = y[proposal], y[replication]
    p_b, r_b = baseline[proposal], baseline[replication]

    started = time.time()
    if arm == "source":
        z_p, z_r, losses = p_x, r_x, []
    elif arm in ("morpheus_factorized", "ssl", "predictive", "morpheus"):
        # `ssl` is the SSL-Scientist comparison arm (identical machinery, contrastive
        # objective); it runs on the same folds/targets for the representation-level
        # comparison and never enters the campaign ledger (arm_split names only
        # source and morpheus_factorized).
        torch.manual_seed(seed)
        p_std, r_std = _standardise_train_apply(p_x, r_x)
        model = IdentityPreservingEncoder(RepresentationConfig(d_input=p_std.shape[1],
                                                                d_output=p_std.shape[1])).to(device)
        p_tensor = torch.as_tensor(p_std, dtype=torch.float32, device=device)
        neighbour_tensor = _partition_neighbours(episode.context.get("neighbors"), proposal, X.shape[0],
                                                   device=torch.device(device))
        losses = fit_response_free(model, p_tensor, arm, neighbours=neighbour_tensor, steps=steps)
        with torch.no_grad():
            z_p = represent(model, p_tensor, arm).detach().cpu().numpy()
            z_r = represent(model, torch.as_tensor(r_std, dtype=torch.float32, device=device),
                            arm).detach().cpu().numpy()
    else:
        raise ValueError("--episode hest_real supports source, ssl, predictive, morpheus_factorized and morpheus arms")

    gene_symbols = list(episode.metadata["gene_symbols"])
    nulls = hest.episode_null_generators(episode)
    # For a learned arm the candidate indexes representation coordinates; the
    # identity-preserving encoder keeps coordinate i anchored to gene i, and
    # the response-free links below (D_R only, written after the candidate is
    # frozen) let a reader verify that anchoring per coordinate.
    gene_provenance = tuple({"feature": symbol, "view": "observed"} for symbol in gene_symbols)
    rows = []
    # Flexible, capacity-controlled relation family (24 random Fourier features
    # over [simpler design, candidate feature]; baseline appended linearly), so a
    # single-variable candidate can carry a nonlinear (threshold/saturating)
    # relation and is not identically zero against its own linear term.
    group_key = "patient_codes" if "patient_codes" in episode.context else "donor_codes"
    p_groups = np.asarray(episode.context[group_key])[proposal] if group_key in episode.context else None
    for candidate, receipt in run_episode(z_p, p_y, z_r, r_y, max_candidates=max_candidates, model="rbf",
                                          proposal_baseline=p_b, replication_baseline=r_b,
                                          proposal_groups=p_groups):
        null_summary = {}
        for kind, generator in nulls.items():
            if kind == "interaction_break":
                # Breaking one constituent tests JOINT structure; a linear
                # subspace has none (its other columns carry the same additive
                # signal), so this null applies to product candidates only.
                if len(candidate.variables) < 2 or candidate.feature_form != "product":
                    continue
                generator = generator(candidate)
            null_receipt = evaluate_structured_nulls(
                receipt, z_p, p_y, proposal_baseline=p_b, replication_baseline=r_b,
                null_generator=generator, n_null=n_null)
            null_summary[kind] = {"p": null_receipt.null_p_value, "n": int(n_null)}
        combined_p = max((v["p"] for v in null_summary.values() if v["p"] is not None), default=None)
        rows.append({
            "candidate_hash": candidate.candidate_hash, "variables": list(candidate.variables),
            "gene_symbols": [gene_symbols[i] for i in candidate.variables if i < len(gene_symbols)],
            "feature_form": candidate.feature_form, "proposal_family": candidate.proposal_family,
            "proposal_score": candidate.proposal_score, "proposal_gain": receipt.proposal_gain,
            "replication_gain": receipt.replication_gain, "baseline_width": receipt.baseline_width,
            "r2": {"proposal_simple": receipt.proposal_r2_simple, "proposal_full": receipt.proposal_r2_full,
                   "replication_simple": receipt.replication_r2_simple, "replication_full": receipt.replication_r2_full},
            "nulls": null_summary, "_combined_null_p": combined_p,
            "observed_coordinate_links": (None if arm == "source" else _top_observed_links(
                z_r, r_x, tuple(candidate.variables), gene_provenance)),
        })
    # BH needs a finite p in [0, 1] for every candidate; one with no
    # applicable null (order-1, so `interaction_break` was skipped) is
    # scored at the least favourable p=1 rather than dropped from the family.
    q_values = adjust_family_pvalues({row["candidate_hash"]: (1.0 if row["_combined_null_p"] is None
                                                              else row["_combined_null_p"]) for row in rows}) if rows else {}
    ranked = sorted(rows, key=lambda row: (-row["proposal_gain"], row["candidate_hash"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["q_value"] = q_values.get(row["candidate_hash"])
        nulls_ok = bool(row["nulls"]) and all(v["p"] is not None and v["p"] <= 0.05 for v in row["nulls"].values())
        # A gain only counts when the full relation itself beats the held-out
        # mean: a less-negative R^2 than the competitor is a transfer failure,
        # not reproducible held-out improvement.
        row["validated"] = bool(row["replication_gain"] >= min_gain and row["q_value"] is not None
                                and row["q_value"] <= fdr and nulls_ok
                                and row["r2"]["replication_full"] > 0.0)
        row.pop("_combined_null_p", None)

    return {
        "schema": "morpheus_general_discovery_packet/v2",
        "episode": "hest_real", "episode_identity": episode.receipt_identity(),
        "protocol_sha256": protocol_sha256, "protocol_file_sha256": protocol_file_sha256,
        "fold": int(fold), "n_folds": int(n_folds), "target": target, "arm": arm, "seed": int(seed),
        "fold_seed": int(fold_seed), "cohort": cohort, "method_truth_accessed": False,
        "representation": {"arm": arm, "steps": int(steps), "loss_trace": losses, "input_width": int(p_x.shape[1])},
        "gene_symbols": gene_symbols, "baseline_width": int(baseline.shape[1]),
        "nbr_mp_program": episode.metadata.get("nbr_mp_program"),
        "fit_patients": list(episode.metadata["fit_patients"]), "held_patients": list(episode.metadata["held_patients"]),
        "proposal_budget": int(max_candidates), "null_kinds": sorted(nulls),
        "validation_rule": {"min_replication_incremental_gain": float(min_gain), "q_value_max": float(fdr),
                           "recurrence_note": "recurrence-across-folds gate applied by the funnel aggregator, not here"},
        "candidates": ranked,
        "elapsed_seconds": time.time() - started,
    }


def run_bmmc_episode(*, fold: int, n_folds: int, seed: int, target: str, arm: str, device: str,
                     protocol_path: Path | None, h5ad_path: str | None, max_candidates: int, n_null: int,
                     min_gain: float, fdr: float, steps: int, n_hvg: int = 300, min_rna_depth: int = 500,
                     min_adt_depth: int = 200, min_genes: int = 200, max_cells: int | None = None) -> dict:
    """Execute one real BMMC CITE-seq discovery packet (second campaign).

    Mirrors `run_hest_episode` exactly (receipt schema v2): a leak-proof
    donor-grouped fold (`bmmc.real_episode`), one predeclared ADT surface-
    protein target, the `source` representation arm (`morpheus_factorized`
    stays reserved/blocked per `bmmc_cite_seq_v01.json`'s `release` block
    until a checkpoint is trained for this substrate), and the bmmc-owned
    structured null suite (`bmmc.episode_null_generators`). Recurrence-
    across-folds and the ledger admission gate remain the funnel
    aggregator's job, not this per-fold runner's.
    """
    from morpheus.tier2.substrates import bmmc

    protocol: dict = {}
    protocol_sha256 = None
    protocol_file_sha256 = None
    if protocol_path is not None:
        from dbench.prospective_campaign import canonical_hash
        protocol_bytes = Path(protocol_path).read_bytes()
        protocol_file_sha256 = sha256(protocol_bytes).hexdigest()
        protocol = json.loads(protocol_bytes)
        protocol_sha256 = canonical_hash(protocol)
    discovery_cfg = protocol.get("discovery", {})
    max_candidates = int(discovery_cfg.get("proposals_per_target", max_candidates))
    n_null = int(discovery_cfg.get("null_suite", {}).get("draws", n_null))
    n_folds = int(discovery_cfg.get("n_folds", n_folds))
    # `fold_seed` locks the DONOR-FOLD assignment across every target/arm this
    # protocol runs; `seed` (the CLI/run argument) still seeds the
    # representation arm, exactly as in `run_hest_episode`.
    fold_seed = int(discovery_cfg.get("fold_seed", seed))

    episode_kwargs = dict(fold=fold, n_folds=n_folds, seed=fold_seed, target=target, n_hvg=n_hvg,
                          min_rna_depth=min_rna_depth, min_adt_depth=min_adt_depth, min_genes=min_genes,
                          max_cells=max_cells)
    if h5ad_path is not None:
        episode_kwargs["h5ad_path"] = h5ad_path
    episode = bmmc.real_episode(**episode_kwargs)

    proposal, replication = episode.partitions.proposal, episode.partitions.replication
    X = episode.observed_matrix()
    y = np.asarray(episode.response, dtype=float)
    baseline = np.asarray(episode.context["baseline"], dtype=float)
    p_x, r_x = X[proposal], X[replication]
    p_y, r_y = y[proposal], y[replication]
    p_b, r_b = baseline[proposal], baseline[replication]

    started = time.time()
    if arm == "source":
        z_p, z_r, losses = p_x, r_x, []
    elif arm in ("morpheus_factorized", "ssl", "predictive"):
        # `ssl` is the SSL-Scientist comparison arm; `morpheus_factorized` on
        # BMMC is predeclared-but-blocked until a representation checkpoint
        # is trained for this substrate (see the campaign protocol's
        # `release` block) -- executable here for development/testing, but
        # not yet an admissible campaign arm.
        torch.manual_seed(seed)
        p_std, r_std = _standardise_train_apply(p_x, r_x)
        model = IdentityPreservingEncoder(RepresentationConfig(d_input=p_std.shape[1],
                                                                d_output=p_std.shape[1])).to(device)
        p_tensor = torch.as_tensor(p_std, dtype=torch.float32, device=device)
        losses = fit_response_free(model, p_tensor, arm, steps=steps)
        with torch.no_grad():
            z_p = represent(model, p_tensor, arm).detach().cpu().numpy()
            z_r = represent(model, torch.as_tensor(r_std, dtype=torch.float32, device=device),
                            arm).detach().cpu().numpy()
    else:
        raise ValueError("--episode bmmc_real supports the source, morpheus_factorized, ssl and predictive arms")

    gene_symbols = list(episode.metadata["gene_symbols"])
    nulls = bmmc.episode_null_generators(episode)
    gene_provenance = tuple({"feature": symbol, "view": "observed"} for symbol in gene_symbols)
    rows = []
    group_key = "patient_codes" if "patient_codes" in episode.context else "donor_codes"
    p_groups = np.asarray(episode.context[group_key])[proposal] if group_key in episode.context else None
    for candidate, receipt in run_episode(z_p, p_y, z_r, r_y, max_candidates=max_candidates, model="rbf",
                                          proposal_baseline=p_b, replication_baseline=r_b,
                                          proposal_groups=p_groups):
        null_summary = {}
        for kind, generator in nulls.items():
            if kind == "interaction_break":
                # Breaking one constituent tests JOINT structure; a linear
                # subspace has none (its other columns carry the same additive
                # signal), so this null applies to product candidates only.
                if len(candidate.variables) < 2 or candidate.feature_form != "product":
                    continue
                generator = generator(candidate)
            null_receipt = evaluate_structured_nulls(
                receipt, z_p, p_y, proposal_baseline=p_b, replication_baseline=r_b,
                null_generator=generator, n_null=n_null)
            null_summary[kind] = {"p": null_receipt.null_p_value, "n": int(n_null)}
        combined_p = max((v["p"] for v in null_summary.values() if v["p"] is not None), default=None)
        rows.append({
            "candidate_hash": candidate.candidate_hash, "variables": list(candidate.variables),
            "gene_symbols": [gene_symbols[i] for i in candidate.variables if i < len(gene_symbols)],
            "feature_form": candidate.feature_form, "proposal_family": candidate.proposal_family,
            "proposal_score": candidate.proposal_score, "proposal_gain": receipt.proposal_gain,
            "replication_gain": receipt.replication_gain, "baseline_width": receipt.baseline_width,
            "r2": {"proposal_simple": receipt.proposal_r2_simple, "proposal_full": receipt.proposal_r2_full,
                   "replication_simple": receipt.replication_r2_simple, "replication_full": receipt.replication_r2_full},
            "nulls": null_summary, "_combined_null_p": combined_p,
            "observed_coordinate_links": (None if arm == "source" else _top_observed_links(
                z_r, r_x, tuple(candidate.variables), gene_provenance)),
        })
    q_values = adjust_family_pvalues({row["candidate_hash"]: (1.0 if row["_combined_null_p"] is None
                                                              else row["_combined_null_p"]) for row in rows}) if rows else {}
    ranked = sorted(rows, key=lambda row: (-row["proposal_gain"], row["candidate_hash"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["q_value"] = q_values.get(row["candidate_hash"])
        nulls_ok = bool(row["nulls"]) and all(v["p"] is not None and v["p"] <= 0.05 for v in row["nulls"].values())
        # A gain only counts when the full relation itself beats the held-out
        # mean: a less-negative R^2 than the competitor is a transfer failure,
        # not reproducible held-out improvement.
        row["validated"] = bool(row["replication_gain"] >= min_gain and row["q_value"] is not None
                                and row["q_value"] <= fdr and nulls_ok
                                and row["r2"]["replication_full"] > 0.0)
        row.pop("_combined_null_p", None)

    return {
        "schema": "morpheus_general_discovery_packet/v2",
        "episode": "bmmc_real", "episode_identity": episode.receipt_identity(),
        "protocol_sha256": protocol_sha256, "protocol_file_sha256": protocol_file_sha256,
        "fold": int(fold), "n_folds": int(n_folds), "target": target, "arm": arm, "seed": int(seed),
        "fold_seed": int(fold_seed), "method_truth_accessed": False,
        "representation": {"arm": arm, "steps": int(steps), "loss_trace": losses, "input_width": int(p_x.shape[1])},
        "gene_symbols": gene_symbols, "baseline_width": int(baseline.shape[1]),
        "baseline_columns": list(episode.metadata["baseline_columns"]),
        "fit_donors": list(episode.metadata["fit_donors"]), "held_donors": list(episode.metadata["held_donors"]),
        "proposal_budget": int(max_candidates), "null_kinds": sorted(nulls),
        "validation_rule": {"min_replication_incremental_gain": float(min_gain), "q_value_max": float(fdr),
                           "recurrence_note": "recurrence-across-folds gate applied by the funnel aggregator, not here"},
        "candidates": ranked,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", choices=["panel", "hest_real", "bmmc_real"], default="panel")
    parser.add_argument("--condition", help="required for --episode panel")
    parser.add_argument("--protocol", type=Path, help="hest_real: dbench/protocols/renal_spatial_v0*.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--target", help="required for --episode hest_real")
    parser.add_argument("--embed-root")
    parser.add_argument("--data-dir")
    parser.add_argument("--cohort", default="ccrcc")
    parser.add_argument("--smoke-samples", type=int,
                        help="hest_real: restrict to the first N ccRCC patients, for a fast smoke run")
    parser.add_argument("--smoke-cells", type=int,
                        help="bmmc_real: donor-stratified subsample to N cells, for a fast smoke run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output-width", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--n-null", type=int, default=99)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--fdr", type=float, default=0.10)
    parser.add_argument("--sidecar", type=Path,
                        help="optional compressed frozen replication-coordinate payload for the sealed evaluator")
    args = parser.parse_args()

    if args.episode == "hest_real":
        if not args.target:
            parser.error("--target is required for --episode hest_real")
        sample_ids = list(range(1, args.smoke_samples + 1)) if args.smoke_samples else None
        result = run_hest_episode(fold=args.fold, n_folds=args.n_folds, seed=args.seed, target=args.target,
                                  arm=args.arm, device=args.device, protocol_path=args.protocol,
                                  embed_root=args.embed_root, data_dir=args.data_dir, sample_ids=sample_ids,
                                  cohort=args.cohort, max_candidates=args.max_candidates, n_null=args.n_null,
                                  min_gain=args.min_gain, fdr=args.fdr, steps=args.steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"episode": "hest_real", "fold": result["fold"], "target": result["target"],
                          "arm": result["arm"], "n_candidates": len(result["candidates"]),
                          "n_validated": sum(row["validated"] for row in result["candidates"]),
                          "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return

    if args.episode == "bmmc_real":
        if not args.target:
            parser.error("--target is required for --episode bmmc_real")
        # `--data-dir` is reused here as the BMMC .h5ad file path (rather than
        # a directory, as it is for hest_real) so no new CLI surface is added
        # for the hest path; unset means `bmmc.real_episode`'s own default.
        result = run_bmmc_episode(fold=args.fold, n_folds=args.n_folds, seed=args.seed, target=args.target,
                                  arm=args.arm, device=args.device, protocol_path=args.protocol,
                                  h5ad_path=args.data_dir, max_candidates=args.max_candidates,
                                  n_null=args.n_null, min_gain=args.min_gain, fdr=args.fdr, steps=args.steps,
                                  max_cells=args.smoke_cells)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"episode": "bmmc_real", "fold": result["fold"], "target": result["target"],
                          "arm": result["arm"], "n_candidates": len(result["candidates"]),
                          "n_validated": sum(row["validated"] for row in result["candidates"]),
                          "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return

    if not args.condition:
        parser.error("--condition is required for --episode panel")
    result = run_packet(condition_id=args.condition, seed=args.seed, arm=args.arm,
                        device=args.device, steps=args.steps, output_width=args.output_width,
                        max_candidates=args.max_candidates,
                        n_null=args.n_null, min_gain=args.min_gain, fdr=args.fdr,
                        sidecar_path=args.sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"packet_id": result["packet_id"], "arm": result["arm"],
                      "n_validated": sum(row["validated"] for row in result["candidates"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
