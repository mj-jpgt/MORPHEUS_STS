"""Run one observed-only MORPHEUS/SSL-Scientist discovery packet.

This is the v2 execution seam: every arm shares the same contextual observed
features, proposal budget, flexible relation family, proposal/replication/null
split, and candidate receipt format.  The runner does not open benchmark truth
and therefore cannot score object fidelity itself; a separate sealed evaluator
may attach that field after candidates have been frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
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
    VALIDATION_ALPHA_GRID, FittedRelation, PROPOSERS, RELATIONS, _cv_alpha, _r2, _RidgeFit, contextual_feature_matrix, evaluate_frozen_candidate,
    evaluate_response_permutation_nulls, evaluate_structured_nulls, propose_candidates, propose_candidates_by,
    V06_RELATIONS, CV_RELATIONS, HO_RELATIONS, _FITTED_CACHE, _fit_key, _patient_bootstrap,
)


#: Proposers sharing the per-target budget under ``--proposer all`` when the protocol names none.
DEFAULT_PROPOSERS_IN_ALL: tuple = ("enumeration", "residual_generator")


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
                proposal_groups: np.ndarray | None = None, candidates: tuple | None = None,
                alpha_grid: tuple | None = None, relation: str = "v04",
                replication_groups: np.ndarray | None = None,
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

    ``candidates`` (already frozen by another proposer, e.g. the residual
    generator) replaces the enumeration proposal; the freeze/evaluate path
    below is the same for every source.

    ``relation`` (``"v04"`` default, or ``"v05"``) is passed to
    ``evaluate_frozen_candidate``; with ``replication_groups`` a v0.5 receipt
    carries the linear decomposition with patient-bootstrap intervals.
    """
    if candidates is None:
        candidates = propose_candidates(candidate_X, candidate_y, context=proposal_context,
                                        max_candidates=max_candidates, max_pairs=max_candidates // 2,
                                        max_subspaces=max_candidates // 4)
    return [(candidate, evaluate_frozen_candidate(
        candidate, candidate_X, candidate_y, replication_X, replication_y,
        proposal_context=proposal_context, replication_context=replication_context,
        proposal_baseline=proposal_baseline, replication_baseline=replication_baseline, model=model,
        proposal_groups=proposal_groups, alpha_grid=alpha_grid, relation=relation,
        replication_groups=replication_groups))
        for candidate in candidates]


def _extend_null_generators(nulls: dict, extras_r: np.ndarray, y_r: np.ndarray, n_base: int,
                            break_column) -> dict:
    """Carry the generator's extra design columns through the substrate nulls.

    Response nulls keep every design column fixed (the same rule they apply
    to the candidate matrix), so they return one fixed extended matrix.
    ``interaction_break`` breaks ``candidate.variables[0]`` with the response
    unchanged: a candidate-matrix column is broken by the substrate's own
    closure; an extra column (only for a candidate with no candidate-matrix
    column, since variables are sorted) by ``break_column(index, column)``,
    the substrate's own break rule applied to that column.
    """
    response_kind = next(kind for kind in nulls
                         if kind != "interaction_break" and not hasattr(nulls[kind], "row_permutation"))
    base_r = nulls[response_kind](0, None, None, None)[0]
    full_r = np.column_stack((base_r, extras_r))
    broken: dict = {}
    out = {}
    for kind, generator in nulls.items():
        if kind != "interaction_break":
            permutation = getattr(generator, "row_permutation", None)
            if permutation is not None:
                # A design-row null (e.g. slide descriptors permuted among held-out
                # patients) moves the extra design columns with the same rows.
                def extended(index, X, y, context, _generator=generator, _perm=permutation):
                    X_null, y_null, c_null = _generator(index, X, y, context)
                    return np.column_stack((X_null, extras_r[_perm(index)])), y_null, c_null
            else:
                def extended(index, X, y, context, _generator=generator):
                    _, y_null, c_null = _generator(index, X, y, context)
                    return full_r, y_null, c_null
            out[kind] = extended
            continue

        def factory(candidate, _factory=generator):
            column = candidate.variables[0]
            if column < n_base:
                inner = _factory(candidate)

                def gene_break(index, X, y, context):
                    X_null, y_null, c_null = inner(index, X, y, context)
                    return np.column_stack((X_null, extras_r)), y_null, c_null
                return gene_break

            def extra_break(index, X, y, context):
                key = (column, index)
                if key not in broken:
                    broken[key] = break_column(index, extras_r[:, column - n_base])
                X_null = full_r.copy()
                X_null[:, column] = broken[key]
                return X_null, y_r, None
            return extra_break
        out[kind] = factory
    return out


def family_config(protocol: dict, family: str | None, target: str) -> dict | None:
    """The protocol's ``discovery.relationship_families[family]`` (``None`` without a family)."""
    if family is None:
        return None
    families = protocol.get("discovery", {}).get("relationship_families", {})
    if family not in families or "candidate_views" not in families[family]:
        raise ValueError(f"--family {family!r} is not a relationship family with candidate_views in the protocol")
    cfg = dict(families[family])
    if target not in cfg.get("targets", []):
        raise ValueError(f"target {target!r} is not declared for family {family!r}")
    return cfg


@dataclass
class Proposal:
    """Frozen candidates plus the matrices and nulls they are evaluated on."""

    candidates: tuple | None
    z_p: np.ndarray
    z_r: np.ndarray
    nulls: dict
    column_names: list | None = None
    design: dict | None = None
    report: dict | None = None
    also: dict = field(default_factory=dict)
    views: list | None = None
    ranking: dict = field(default_factory=dict)


def _proposal_cv_gain(candidate, z_p, p_y, p_b, p_groups, n_splits: int = 5, alpha_grid=None,
                      relation: str = "v04") -> float:
    """Patient-grouped cross-validated proposal gain (proposal rows only).

    The relation and its competitor are fit on four groups of proposal
    patients and scored on the fifth; the mean over folds ranks the shared
    budget.  An in-sample gain rewards whichever proposer screened on the
    same rows (enumeration took 30 of 32 slots in the first smoke)."""
    from morpheus_encoder.hypothesis_generator import grouped_folds

    folds = grouped_folds(p_groups, len(p_y), n_splits, 0)
    gains = []
    for k in np.unique(folds):
        test = folds == k
        if test.sum() < 2 or (~test).sum() < 2:
            continue
        fitted = FittedRelation(candidate, z_p[~test], p_y[~test], context=None, baseline=p_b[~test],
                                model="rbf", ridge_alpha=1.0, rbf_features=24,
                                groups=None if p_groups is None else np.asarray(p_groups)[~test],
                                alpha_grid=alpha_grid, relation=relation)
        gains.append(fitted.gain(z_p[test], p_y[test], context=None, baseline=p_b[test])[0])
    return float(np.mean(gains)) if gains else float("nan")


def _propose(episode, z_p: np.ndarray, z_r: np.ndarray, nulls: dict, *, p_y, p_b, p_groups, target: str,
             proposer: str, family_cfg: dict | None, seed: int, device: str, break_column,
             max_candidates: int, relation: str = "v04", baseline_names: list | None = None,
             in_all: tuple = DEFAULT_PROPOSERS_IN_ALL, learned_checkpoint: str | None = None,
             merge: str = "cv_gain", design=None, fixed_hypotheses: list | None = None) -> Proposal:
    """Run the requested proposer(s) on proposal rows and freeze the candidates.

    Without a family and with ``enumeration`` this is the historical path
    (``candidates=None`` -> enumerate over the candidate matrix).  Otherwise
    the episode's multi-view design is built (a family restricts it to its
    ``candidate_views``); enumeration draws singles and low-order products
    from those views (``hoptimus`` only through the generator; molecular x
    context products when the family requires context); the generator
    searches the same design (graph branch only for a context family) and
    obeys the same context requirement.  Under ``all`` both pools share one
    budget: duplicates are merged (``also_proposed_by``) and the top
    ``max_candidates`` by proposal gain are frozen.  Candidates are compacted
    to the candidate matrix plus the extra columns they use.

    ``in_all`` (protocol ``discovery.proposers_in_all``; default enumeration +
    residual_generator, the v0.5 pool) names the proposers that share the
    budget under ``all``; each emits up to ``max_candidates`` (equal budgets)
    before the joint ranking.  ``residual_bank`` (alone: a development mode)
    obeys a family's context requirement by keeping only admissible
    hypotheses; emitted / admissible counts go to its report.

    ``learned_search`` (alone: development mode) calls
    ``morpheus_encoder.learned_search.propose.propose`` on the proposal rows of
    the same design (proposal-local within-patient neighbours); the model comes
    from ``learned_checkpoint`` or env ``MORPHEUS_LEARNED_CHECKPOINT`` (absent:
    the marginal-screen stub, recorded as ``report["stub"]``).  Its transformed
    variables (``nbr6:`` / ``nbr18:`` / ``contrast:`` / ``concept:``) are rebuilt on
    every row with ``transform_columns`` and appended to the design as extra
    columns, compacted like the generator's; a neighbour transform counts as a
    ``context_neighbour_mean`` variable for a family's context requirement.

    A family with ``transforms`` (protocol v0.7 grammar) states its context requirement
    in transforms: a hypothesis needs at least one own-spot variable and at least one
    ``nbr6`` / ``nbr18`` / ``contrast`` variable (no CellViT context view is added).
    ``merge="interleaved_rank"`` (protocol merge rule) interleaves the proposers' ranked
    lists under ``all`` (rank 1 of each in ``in_all`` order, then rank 2, ...), deduplicated,
    capped at ``max_candidates``, with no cross-validated re-ranking.  ``design`` is a
    prebuilt design (the hest_multi path); ``fixed_hypotheses`` (``proposer="fixed"``) scores
    already frozen hypotheses named by their receipt ``variable_names`` (all-fold pooling).
    """
    if proposer == "fixed":
        if fixed_hypotheses is None:
            raise ValueError("proposer 'fixed' needs fixed_hypotheses")
    elif proposer not in PROPOSERS:
        raise ValueError(f"--proposer must be one of {PROPOSERS}")
    if family_cfg is None and proposer == "enumeration":
        return Proposal(None, z_p, z_r, nulls)
    from morpheus_encoder.general_discovery import Candidate
    from morpheus_encoder.hypothesis_generator import (
        CONTEXT_VIEWS, MOLECULAR_VIEWS, GeneratorConfig, enumerate_design, episode_design, proposal_neighbours,
    )

    proposal, replication = episode.partitions.proposal, episode.partitions.replication
    y = np.asarray(episode.response, dtype=float)
    require_context = bool(family_cfg and "requirement" in family_cfg)
    transform_family = bool(family_cfg and family_cfg.get("transforms"))
    # explicit protocol fields (r7): family.admissibility = {"context": bool, "cross_modal": bool}
    adm_cfg = (family_cfg or {}).get("admissibility")
    if isinstance(adm_cfg, dict):
        require_context = bool(adm_cfg.get("context") or adm_cfg.get("cross_modal"))
    family_transforms = tuple((family_cfg or {}).get("transforms") or ())
    # A family may name its own variable / context views (patient-level
    # families: slide structure x clinical context); absent -> the v0.4
    # molecular x tissue-context convention, unchanged.
    variable_views = tuple((family_cfg or {}).get("variable_views", MOLECULAR_VIEWS))
    context_role_views = tuple((family_cfg or {}).get("context_views", CONTEXT_VIEWS))
    if design is None:
        z_full = np.empty((episode.n_rows, z_p.shape[1]))
        z_full[proposal], z_full[replication] = z_p, z_r
        design = episode_design(episode, z_full, target=target, response=y,
                                candidate_views=None if family_cfg is None else family_cfg["candidate_views"],
                                context_views=(tuple((family_cfg or {}).get("context_views", ("cellvit_structure",)))
                                               if require_context and not transform_family else ()))
        del z_full
    n_base = design.n_base
    view_of = {int(c): name for name, cols in design.views.items() for c in cols}
    # The design every proposer's column indices point into; learned_search appends its
    # transformed variables (other proposers leave it as the episode design).
    design_X, design_names = design.X, list(design.column_names)

    transformed: set = set()  # indices of nbr6 / nbr18 / contrast variables (shared resolver)

    cross_modal = (bool(adm_cfg.get("cross_modal")) if isinstance(adm_cfg, dict)
                   else family_requires_cross_modal(family_cfg))

    def admissible(cols) -> bool:
        if transform_family and cross_modal:   # "at least one molecular variable and one morphology variable"
            views = {view_of.get(int(c)) for c in cols}
            return bool(views & set(MOLECULAR_VIEW_KEYS)) and bool(views & set(MORPHOLOGY_VIEW_KEYS))
        if transform_family:
            return (any(int(c) in transformed for c in cols)
                    and any(int(c) not in transformed for c in cols))
        views = {view_of.get(int(c)) for c in cols}
        return bool(views & set(variable_views)) and bool(views & set(context_role_views))

    pool: list = []
    report = None
    def runs(name: str) -> bool:
        return proposer == name or (proposer == "all" and name in in_all)

    if runs("enumeration"):
        if family_cfg is None:
            pool += list(propose_candidates(z_p, p_y, max_candidates=max_candidates,
                                            max_pairs=max_candidates // 2, max_subspaces=max_candidates // 4))
        else:
            enum_views = [v for v in family_cfg["candidate_views"] if v != "hoptimus"]
            pool += list(enumerate_design(design.X[proposal], p_y, views=design.views, enumerate_views=enum_views,
                                          max_candidates=max_candidates, require_context=require_context,
                                          molecular_views=variable_views, context_views=context_role_views))
    if runs("residual_generator"):
        neighbours = episode.context.get("neighbors")
        coords = episode.context.get("coords")
        generator_kwargs = {
            "views": design.views, "roles": design.roles, "column_names": design.column_names,
            "neighbours": None if neighbours is None else proposal_neighbours(list(neighbours), proposal,
                                                                               episode.n_rows),
            "coords": None if coords is None else np.asarray(coords, dtype=float)[proposal],
            "neighbour_of": design.neighbour_of, "seed": int(seed), "device": device,
        }
        if family_cfg is not None:
            generator_kwargs["config"] = GeneratorConfig(graph=require_context and bool(family_cfg.get("graph", True)))
            if require_context:
                generator_kwargs["admissible"] = admissible
        conventional_p = None if design.conventional is None else design.conventional[proposal]
        generated, report = propose_candidates_by(
            "residual_generator", design.X[proposal], p_y, groups=p_groups, baseline=conventional_p,
            max_candidates=max_candidates, generator_kwargs=generator_kwargs)
        pool += list(generated)
    if runs("residual_bank"):
        # One head per ladder level, trained on the cross-fitted residuals r(k) of the
        # proposal rows (contract documented above general_discovery.PROPOSERS).
        if baseline_names is None:
            raise ValueError("proposer 'residual_bank' needs the comparison model's column names "
                             "(episode.metadata['baseline_columns']) to build the ladder levels")
        from dbench.decomposition import ladder_levels
        generated, bank_report = propose_candidates_by(
            "residual_bank", design.X[proposal], p_y, groups=p_groups, baseline=p_b, max_candidates=max_candidates,
            generator_kwargs={"levels": ladder_levels(list(baseline_names))["levels"], "views": design.views,
                              "roles": design.roles, "column_names": design.column_names,
                              "seed": int(seed), "device": device})
        kept = [c for c in generated if not require_context or admissible(c.variables)]
        bank_report = {**(bank_report or {}), "emitted": len(generated), "admissible": len(kept),
                       "context_requirement_applied": require_context}
        report = bank_report if report is None else {**report, "residual_bank": bank_report}
        pool += kept
    if runs("learned_search") or runs("classical_pair") or proposer == "fixed":
        # Both name their variables (``own:`` / ``nbr6:`` / ``nbr18:`` / ``contrast:`` transforms of
        # design columns, or learned concepts); one shared resolver maps names to design indices and
        # appends every transformed variable once, rebuilt on all rows (within-patient neighbours).
        from morpheus_encoder.learned_search.propose import NameResolver
        shared = NameResolver(design.X, design.column_names)
        learned_model = learned_embed = None
        named: list = []
        if runs("learned_search"):
            learned, learned_report, learned_model, learned_embed = _learned_search_pool(
                episode, design, proposal, p_y, p_groups, max_candidates=max_candidates, seed=seed, device=device,
                checkpoint=learned_checkpoint,
                family={"context": bool(require_context and transform_family and not cross_modal),
                        "cross_modal": bool(require_context and cross_modal)} if require_context else None)
            named.append(("learned_search", learned, learned_report))
        if runs("classical_pair"):
            classical, classical_report = _classical_pair_pool(
                design, proposal, p_y, p_b, p_groups, episode, require_context=require_context,
                max_candidates=max_candidates, seed=seed, device=device,
                transforms=family_transforms if transform_family else None)
            named.append(("classical_pair", classical, classical_report))
        if proposer == "fixed":
            fixed = [_fixed_named_candidate(h, design) for h in fixed_hypotheses]
            named.append(("fixed", fixed, {"n_fixed": len(fixed), "purpose": "all-fold pooling: frozen "
                                           "hypotheses of other folds scored on this fold"}))
        resolved = {name: [_candidate_from_names(c, shared) for c in cands] for name, cands, _ in named}
        if shared.specs:
            extra = _transformed_columns(episode, design, shared.specs, model=learned_model, embed=learned_embed)
            design_X = np.column_stack((design_X, extra))
            design_names = design_names + [spec["name"] for spec in shared.specs]
            for j, spec in enumerate(shared.specs):
                base_view = "hoptimus" if "concept" in spec else view_of.get(int(spec["column"]))
                view_of[design.X.shape[1] + j] = ("context_neighbour_mean" if spec["transform"] != "own"
                                                  and not cross_modal else base_view)
                if spec["transform"] != "own":
                    transformed.add(design.X.shape[1] + j)
        for name, cands, sub_report in named:
            kept = [c for c in resolved[name]
                    if name == "fixed" or not require_context or admissible(c.variables)]
            sub_report = {**sub_report, "emitted": len(cands), "admissible": len(kept),
                          "context_requirement_applied": require_context}
            report = (sub_report if report is None and proposer != "all"
                      else {**(report or {}), name: sub_report})
            pool += kept
    used = sorted({i for c in pool for i in c.variables if i >= n_base})
    remap = {old: n_base + k for k, old in enumerate(used)}
    pool = [Candidate(tuple(remap.get(i, i) for i in c.variables), c.context_variables, weights=c.weights,
                      feature_form=c.feature_form, proposal_family=c.proposal_family,
                      proposal_score=c.proposal_score, metadata=dict(c.metadata)) for c in pool]
    column_names = design_names[:n_base] + [design_names[i] for i in used]
    if used:
        extras = np.asarray(design_X[:, used], dtype=float)
        nulls = _extend_null_generators(nulls, extras[replication], y[replication], n_base, break_column)
        z_p = np.column_stack((z_p, extras[proposal]))
        z_r = np.column_stack((z_r, extras[replication]))
    # Merge the same relation nominated by both proposers.
    merged: dict = {}
    also: dict = {}
    if proposer == "all" and merge == "interleaved_rank":
        return _interleaved_proposal(pool, in_all, max_candidates, z_p, z_r, nulls, column_names, design,
                                     design_names, used, report)
    for c in pool:
        key = (c.variables, c.feature_form, c.context_variables)
        who = c.metadata.get("proposer", "enumeration")
        if key in merged:
            first = merged[key]
            if who != first.metadata.get("proposer", "enumeration"):
                also.setdefault(first.candidate_hash, set()).add(who)
            continue
        merged[key] = c
    candidates = list(merged.values())
    shared = None
    if proposer == "all":
        grid = None if family_cfg is None else VALIDATION_ALPHA_GRID
        gains = {c.candidate_hash: _proposal_cv_gain(c, z_p, p_y, p_b, p_groups, alpha_grid=grid,
                                                     relation=relation)
                 for c in candidates}
        candidates.sort(key=lambda c: (-np.nan_to_num(gains[c.candidate_hash], nan=-np.inf), c.candidate_hash))
        shared = {"pool": len(pool), "unique": len(candidates), "frozen": min(len(candidates), max_candidates),
                  "ranking": "patient-grouped 5-fold cross-validated proposal gain (proposal rows only)",
                  "pool_by_proposer": {},
                  "frozen_by_proposer": {}}
        for c in candidates:
            who = c.metadata.get("proposer", "enumeration")
            shared["pool_by_proposer"][who] = shared["pool_by_proposer"].get(who, 0) + 1
        candidates = candidates[:max_candidates]
        for c in candidates:
            who = c.metadata.get("proposer", "enumeration")
            shared["frozen_by_proposer"][who] = shared["frozen_by_proposer"].get(who, 0) + 1
    summary = design.summary()
    summary["compacted_extra_columns"] = [design_names[i] for i in used]
    if shared is not None:
        summary["shared_budget"] = shared
    return Proposal(tuple(candidates), z_p, z_r, nulls, column_names, summary, report,
                    {h: sorted(v) for h, v in also.items()}, sorted(design.views),
                    {} if shared is None else gains)


def _learned_embedding(episode, proposal) -> np.ndarray | None:
    """The episode's frozen H-Optimus-0 spot embedding (bank view ``hoptimus``), uncovered
    rows filled with the proposal-row column means; ``None`` when the bank has no such view."""
    view = ((episode.metadata or {}).get("bank") or {}).get("hoptimus")
    if not isinstance(view, dict) or view.get("X") is None:
        return None
    E = np.asarray(view["X"], dtype=np.float32).copy()
    fill = np.nanmean(E[proposal], axis=0)
    rows, cols = np.nonzero(~np.isfinite(E))
    E[rows, cols] = np.nan_to_num(fill[cols])
    return E


def _transformed_columns(episode, design, specs, *, model=None, embed=None) -> np.ndarray:
    """Every named transformed variable on ALL rows (neighbours never cross a patient, so the
    proposal rows equal a proposal-only build)."""
    from morpheus_encoder.learned_search.propose import transform_columns
    neighbours = episode.context.get("neighbors")
    coords = episode.context.get("coords")
    groups = np.asarray(episode.context.get("patient_codes", episode.context.get("donor_codes")))
    if getattr(model, "kind", "") == "gain_v03":   # v03: nbr18 = two-hop ring of nbr6, as in training and the pool
        coords = None
    return transform_columns(specs, design.X, None if neighbours is None else list(neighbours), embed=embed,
                             model=model, coords=None if coords is None else np.asarray(coords, dtype=float),
                             groups=None if coords is None else groups)


def _candidate_from_names(candidate, resolver):
    """Re-index a named candidate (``provenance["transforms"]``) into [design | shared extras]."""
    from morpheus_encoder.general_discovery import Candidate
    names = [t["name"] for t in candidate.metadata["provenance"]["transforms"]]
    variables = tuple(sorted(resolver.resolve(n)[0] for n in names))
    return Candidate(variables, candidate.context_variables, weights=candidate.weights,
                     feature_form=candidate.feature_form, proposal_family=candidate.proposal_family,
                     proposal_score=candidate.proposal_score, metadata=dict(candidate.metadata))


def _learned_embedding(episode, proposal) -> np.ndarray | None:
    """The episode's frozen H-Optimus-0 spot embedding (bank view ``hoptimus``), uncovered
    rows filled with the proposal-row column means; ``None`` when the bank has no such view."""
    view = ((episode.metadata or {}).get("bank") or {}).get("hoptimus")
    if not isinstance(view, dict) or view.get("X") is None:
        return None
    E = np.asarray(view["X"], dtype=np.float32).copy()
    fill = np.nanmean(E[proposal], axis=0)
    rows, cols = np.nonzero(~np.isfinite(E))
    E[rows, cols] = np.nan_to_num(fill[cols])
    return E


def _learned_search_pool(episode, design, proposal, p_y, p_groups, *, max_candidates: int, seed: int, device: str,
                         checkpoint: str | None, family: dict | None = None):
    """``learned_search.propose`` on the proposal rows of ``design`` (proposal-local neighbours).
    Returns (named candidates, report, model, all-row embedding).

    v03 checkpoints additionally receive the un-centred proposal-row inputs (``design.v03_raw``:
    design columns, target, log depth, composition, sections; built by the hest_multi loader) so
    the gain predictor sees the training-episode construction, and the family requirement
    (``family``), enforced at candidate generation."""
    from morpheus_encoder.hypothesis_generator import proposal_neighbours
    from morpheus_encoder.learned_search.propose import load_model, propose

    checkpoint = checkpoint or os.environ.get("MORPHEUS_LEARNED_CHECKPOINT") or None
    model = load_model(checkpoint, device=device) if checkpoint else None
    neighbours = episode.context.get("neighbors")
    nb_p = None if neighbours is None else proposal_neighbours(list(neighbours), proposal, episode.n_rows)
    coords = episode.context.get("coords")
    embed = _learned_embedding(episode, proposal) if model is not None else None
    v03 = getattr(model, "kind", "") == "gain_v03"
    raw = getattr(design, "v03_raw", None) if v03 else None
    extra = {}
    if v03:
        if raw is not None:
            extra["raw"] = {k: (None if v is None else np.asarray(v)[proposal]) for k, v in raw.items()
                            if k != "raw_rows"}
        extra["family"] = family
        embed = None   # v03 morphology enters as the design view morph_dictionary_v03, not a model layer
    result = propose(model, design.X[proposal], p_y, p_groups, nb_p,
                     None if coords is None else np.asarray(coords, dtype=float)[proposal],
                     list(design.column_names), design.views,
                     embed=None if embed is None else embed[proposal], budget=max_candidates, seed=int(seed),
                     device=device, **extra)
    report = {**result.report, "checkpoint": checkpoint, "n_relationships_tested": result.n_relationships_tested,
              "extra_names": list(result.extra_names),
              **({"v03_inputs": "un-centred training construction" if raw is not None else "standardised design",
                  "family_constraint": family} if v03 else {})}
    if result.extra_specs:  # proposal-only and all-row builds must agree (no neighbour crosses the split)
        full = _transformed_columns(episode, design, result.extra_specs, model=model, embed=embed)
        if not np.allclose(full[proposal], result.extra_columns, rtol=0, atol=1e-9):
            raise RuntimeError("learned_search extra columns differ between proposal-only and all-row builds")
    return list(result.candidates), report, model, embed


#: classical pair search inside a registered run (blind-discovery protocols): the benchmark's
#: explainable boosting machine (mains + pairwise interactions) and gradient boosting with the
#: Friedman H-statistic, on the proposal rows of the same design.
CLASSICAL_PAIR_METHODS = ("ebm_ga2m", "gbm_interaction")


#: base columns whose nbr6 / nbr18 / contrast transforms enter the classical searches (top by
#: within-patient |r| with the response on proposal rows, maximum over transforms)
CLASSICAL_TRANSFORM_SHORTLIST = 64


def _classical_transformed_block(design, proposal, p_y, p_groups, episode, transforms, shortlist: int):
    """Transformed columns for the classical searches: for every base column the requested
    transforms on proposal rows (proposal-local within-patient neighbours; nbr18 = 18 nearest
    same-patient spots), then the ``shortlist`` base columns with the largest within-patient
    |r| with the response (maximum over transforms).  Returns (block, names, info)."""
    from morpheus_encoder.hypothesis_generator import proposal_neighbours
    from morpheus_encoder.learned_search.propose import neighbour_operator, neighbours18, pad_neighbours

    kinds = [t for t in transforms if t != "own"]
    n_p = len(proposal)
    neighbours = episode.context.get("neighbors")
    if not kinds or neighbours is None:
        return np.zeros((n_p, 0)), [], {"transforms": kinds, "reason": "no transforms or no neighbours"}
    nb6 = pad_neighbours(proposal_neighbours(list(neighbours), proposal, episode.n_rows), n_p)
    groups = np.asarray(p_groups) if p_groups is not None else np.zeros(n_p, dtype=int)
    coords = episode.context.get("coords")
    base_cols = np.sort(np.concatenate([np.asarray(c, dtype=int) for c in design.views.values() if len(c)]))
    Xb = np.asarray(design.X[proposal][:, base_cols], dtype=float)
    mats = {}
    if "nbr6" in kinds or "contrast" in kinds:
        lag6 = np.asarray(neighbour_operator(nb6) @ Xb)
        if "nbr6" in kinds:
            mats["nbr6"] = lag6
        if "contrast" in kinds:
            mats["contrast"] = Xb - lag6
    if "nbr18" in kinds:
        nb18 = neighbours18(nb6, None if coords is None else np.asarray(coords, dtype=float)[proposal],
                            None if coords is None else groups)
        mats["nbr18"] = np.asarray(neighbour_operator(nb18) @ Xb)

    def centred(a):
        a = np.array(a, dtype=float, copy=True)
        for g in np.unique(groups):
            r = groups == g
            a[r] -= a[r].mean(0)
        return a

    yc = centred(np.asarray(p_y, dtype=float).reshape(-1, 1)).ravel()
    score = np.zeros(len(base_cols))
    for M in mats.values():
        Mc = centred(M)
        den = np.sqrt((Mc ** 2).sum(0) * (yc ** 2).sum())
        score = np.maximum(score, np.abs(Mc.T @ yc) / np.maximum(den, 1e-12))
    keep = np.argsort(-score, kind="stable")[:int(shortlist)]
    blocks, names = [], []
    for t in kinds:
        blocks.append(mats[t][:, keep])
        names += [f"{t}:{design.column_names[int(base_cols[j])]}" for j in keep]
    return np.column_stack(blocks), names, {"transforms": kinds, "shortlist": int(len(keep)),
                                            "rule": "top base columns by within-patient |r| with the response on "
                                                    "proposal rows, maximum over transforms"}


def _classical_pair_pool(design, proposal, p_y, p_b, p_groups, episode, *, require_context: bool,
                         max_candidates: int, seed: int, device: str, methods: tuple = CLASSICAL_PAIR_METHODS,
                         transforms: tuple | None = None, shortlist: int = CLASSICAL_TRANSFORM_SHORTLIST):
    """``classical_pair``: ``dbench.benchmark.methods`` ``ebm_ga2m`` and ``gbm_interaction`` on a
    ``MethodInput`` built from the proposal rows of the episode design (budget = the per-target
    count each); their ``Hypothesis`` lists become named ``Candidate`` objects, interleaved by
    rank and deduplicated, at most ``max_candidates``.  Without interpret-core the EBM is
    recorded as unavailable and the GBM runs alone."""
    from dbench.benchmark.methods import METHODS, MethodInput
    from morpheus_encoder.hypothesis_generator import proposal_neighbours
    from morpheus_encoder.learned_search.propose import pad_neighbours

    n_p = int(len(proposal))
    neighbours = episode.context.get("neighbors")
    nb = (-np.ones((n_p, 1), dtype=np.int64) if neighbours is None else
          pad_neighbours(proposal_neighbours(list(neighbours), proposal, episode.n_rows), n_p))
    coords = episode.context.get("coords")
    views = {v: np.asarray(c, dtype=int) for v, c in design.views.items() if len(c)}
    roles = {v: design.roles.get(v, "variable") for v in views}
    X_in, names_in = np.asarray(design.X[proposal], dtype=float), list(design.column_names)
    transform_info = None
    if transforms:
        T, t_names, transform_info = _classical_transformed_block(design, proposal, p_y, p_groups, episode,
                                                                  transforms, shortlist)
        if T.shape[1]:
            views["context_neighbour_mean"] = np.arange(X_in.shape[1], X_in.shape[1] + T.shape[1])
            roles["context_neighbour_mean"] = "context"
            X_in = np.column_stack((X_in, T))
            names_in = names_in + t_names
    inp = MethodInput(X=X_in, y=np.asarray(p_y, dtype=float),
                      comparison=np.asarray(p_b, dtype=float),
                      groups=np.asarray(p_groups) if p_groups is not None else np.zeros(n_p, dtype=int),
                      neighbours=nb, coords=None if coords is None else np.asarray(coords, dtype=float)[proposal],
                      names=names_in, views=views, roles=roles,
                      family="context" if require_context else "molecular", seed=int(seed),
                      device="cpu" if device is None else str(device))
    report: dict = {"methods": {}, "budget_each": int(max_candidates), "transformed_columns": transform_info}
    lists = []
    for name in methods:
        method = METHODS[name]()
        out = method.run(inp, int(max_candidates))
        if not out.available:
            report["methods"][name] = {"available": False, "reason": out.info.get("unavailable")}
            continue
        report["methods"][name] = {"available": True, "n_hypotheses": len(out.hypotheses),
                                   "n_relationships_tested": out.info.get("n_relationships_tested"),
                                   "seconds": round(out.seconds, 2)}
        lists.append((name, out.hypotheses[:int(max_candidates)]))
    if not lists:
        raise RuntimeError(f"classical_pair: no classical method available ({report['methods']})")
    report["fallback"] = None if len(lists) == len(methods) else f"ran {[n for n, _ in lists]} only"
    report["n_relationships_tested"] = int(sum(v.get("n_relationships_tested") or 0
                                               for v in report["methods"].values()))
    out_c, seen = [], set()
    for rank in range(max((len(h) for _, h in lists), default=0)):
        for name, hyps in lists:
            if rank >= len(hyps) or len(out_c) >= max_candidates:
                continue
            c = _hypothesis_to_named_candidate(hyps[rank], name, rank)
            key = tuple(sorted(t["name"] for t in c.metadata["provenance"]["transforms"])) + (c.feature_form,)
            if key not in seen:
                seen.add(key)
                out_c.append(c)
    return out_c, report


def _hypothesis_to_named_candidate(h, source: str, rank: int):
    """A benchmark ``Hypothesis`` -> a named ``Candidate`` (variables re-indexed by the caller).
    ``spatial_lag`` names a bare column evaluated on its neighbour mean; ``additive`` pairs are a
    linear subspace; product / context / higher-order sets are products."""
    from morpheus_encoder.general_discovery import Candidate
    names = []
    for v in h.variables:
        head, _, rest = str(v).partition(":")
        if h.relation == "spatial_lag":
            names.append(f"nbr6:{v}")
        elif head in ("nbr6", "nbr18", "contrast") and rest:
            names.append(str(v))
        else:
            names.append(f"own:{v}")
    k = len(names)
    linear = k == 1 or h.relation == "additive"
    score = float(h.score) if np.isfinite(float(h.score)) else float("nan")
    return Candidate(tuple(range(k)), feature_form="linear_subspace" if linear else "product",
                     weights=(1.0,) * k if linear else (), proposal_family="classical_pair", proposal_score=score,
                     metadata={"proposer": "classical_pair", "provenance": {
                         "hypothesis_kind": h.relation, "source_method": source, "source_score": score,
                         "source_rank": int(rank), "recurrent_in_generator": None,
                         "transforms": [{"name": n, "transform": n.split(":", 1)[0], "base": n.split(":", 1)[1]}
                                        for n in names]}})


def _interleaved_proposal(pool, in_all, max_candidates, z_p, z_r, nulls, column_names, design, design_names,
                          used, report) -> "Proposal":
    """Protocol merge rule ``interleaved_rank``: each proposer's own ranked list (emission
    order), interleaved by rank in ``in_all`` order; a hypothesis nominated twice keeps its
    first position and records the other proposer; the first ``max_candidates`` are frozen."""
    lists = {name: [c for c in pool if c.metadata.get("proposer", "enumeration") == name] for name in in_all}
    merged, also, order = {}, {}, []
    for rank in range(max((len(v) for v in lists.values()), default=0)):
        for name in in_all:
            if rank >= len(lists[name]):
                continue
            c = lists[name][rank]
            key = (c.variables, c.feature_form, c.context_variables)
            if key in merged:
                first = merged[key]
                if name != first.metadata.get("proposer", "enumeration"):
                    also.setdefault(first.candidate_hash, set()).add(name)
                continue
            merged[key] = c
            order.append(c)
    candidates = order[:max_candidates]
    shared = {"pool": len(pool), "unique": len(order), "frozen": len(candidates),
              "ranking": "interleaved rank across proposers (protocol merge rule); no cross-validated re-ranking",
              "pool_by_proposer": {n: len(v) for n, v in lists.items()},
              "frozen_by_proposer": {}}
    for c in candidates:
        who = c.metadata.get("proposer", "enumeration")
        shared["frozen_by_proposer"][who] = shared["frozen_by_proposer"].get(who, 0) + 1
    summary = design.summary()
    summary["compacted_extra_columns"] = [design_names[i] for i in used]
    summary["shared_budget"] = shared
    return Proposal(tuple(candidates), z_p, z_r, nulls, column_names, summary, report,
                    {h: sorted(v) for h, v in also.items()}, sorted(design.views), {})


def _fixed_named_candidate(h: dict, design):
    """A frozen hypothesis from another fold's receipt (``variable_names``, ``feature_form``)
    as a named candidate on this fold's design (resolved by the shared name resolver)."""
    from morpheus_encoder.general_discovery import Candidate
    index = set(map(str, design.column_names))
    names = []
    for v in h["variable_names"]:
        v = str(v)
        head, _, rest = v.partition(":")
        if head in ("nbr6", "nbr18", "contrast") and (rest in index or rest.startswith("concept:")):
            names.append(v)
        elif v in index:
            names.append(f"own:{v}")
        elif head == "candidate_matrix" and rest in index:
            names.append(f"own:{rest}")
        else:
            raise KeyError(f"frozen variable {v!r} is not a column (or transform of a column) of this fold's design")
    k = len(names)
    form = str(h.get("feature_form", "product" if k >= 2 else "linear_subspace"))
    linear = form == "linear_subspace"
    return Candidate(tuple(range(k)), feature_form=form, weights=(1.0,) * k if linear else (),
                     proposal_family="fixed", proposal_score=float("nan"),
                     metadata={"proposer": "fixed", "provenance": {
                         "hypothesis_kind": "frozen_elsewhere", "frozen_by": list(h.get("proposers") or []),
                         "frozen_in_folds": list(h.get("frozen_in_folds") or []), "recurrent_in_generator": None,
                         "transforms": [{"name": n, "transform": n.split(":", 1)[0], "base": n.split(":", 1)[1]}
                                        for n in names]}})


def _proposer_fields(candidate, column_names, also: dict, gene_symbols: list) -> dict:
    """Additive receipt fields naming the hypothesis source of a candidate.

    ``variable_names`` are always ``view:name`` (candidate-matrix genes as
    ``candidate_matrix:SYMBOL``) for both proposers; ``gene_symbols`` stay
    bare symbols for gene variables.
    """
    provenance = candidate.metadata.get("provenance")
    if column_names is None:
        names = [f"candidate_matrix:{gene_symbols[i]}" if i < len(gene_symbols) else f"candidate_matrix:x{i}"
                 for i in candidate.variables]
    else:
        names = [n if ":" in n else f"candidate_matrix:{n}"
                 for n in (column_names[i] for i in candidate.variables)]
    fields = {"proposer": candidate.metadata.get("proposer", "enumeration"),
              "also_proposed_by": list(also.get(candidate.candidate_hash, [])),
              "provenance": None if provenance is None else json.loads(json.dumps(dict(provenance))),
              "recurrent_in_generator": None if provenance is None else provenance.get("recurrent_in_generator"),
              "variable_names": names,
              "gene_symbols": [n.split(":", 1)[1] for n in names
                               if n.startswith(("raw_genes:", "candidate_matrix:"))]}
    return fields


def _gain_interval(candidate, z_p, p_y, z_r, r_y, p_b, r_b, p_groups, r_groups, n_boot: int = 200,
                   fixed_alphas=None, relation: str = "v04", alpha_grid=None) -> list:
    """Held-out gain interval: the frozen proposal fit (the receipt's own
    ridge strengths), held-out patients resampled with replacement
    (2.5/97.5 percentiles).  A v0.6 relation ignores ``fixed_alphas`` (its
    penalties come from inner CV), so its interval uses the receipt's own
    cached fit, else a refit on the receipt's grid."""
    fitted = None
    if relation in CV_RELATIONS:
        fitted = _FITTED_CACHE.get(_fit_key(candidate.candidate_hash, z_p, p_y, None, p_b))
        if fitted is None or fitted.relation != relation:
            fitted = FittedRelation(candidate, z_p, p_y, context=None, baseline=p_b, model="rbf", ridge_alpha=1.0,
                                    rbf_features=24, groups=p_groups, alpha_grid=alpha_grid, relation=relation)
    if fitted is None:
        fitted = FittedRelation(candidate, z_p, p_y, context=None, baseline=p_b, model="rbf", ridge_alpha=1.0,
                                rbf_features=24, groups=p_groups, fixed_alphas=fixed_alphas, relation=relation)
    simple, full = fitted.predictions(z_r, context=None, baseline=r_b)
    rng = np.random.default_rng(int(candidate.candidate_hash[:8], 16))
    groups = np.unique(r_groups)
    members = {g: np.flatnonzero(r_groups == g) for g in groups}
    gains = []
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.choice(groups, size=len(groups), replace=True)])
        gains.append(_r2(r_y[rows], full[rows]) - _r2(r_y[rows], simple[rows]))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def proposers_in_all(discovery_cfg: dict) -> tuple:
    """Proposers sharing the budget under ``--proposer all`` (``discovery.proposers_in_all``;
    absent -> the v0.5 pair, so earlier protocols run unchanged)."""
    names = discovery_cfg.get("proposers_in_all", DEFAULT_PROPOSERS_IN_ALL)
    if isinstance(names, dict):
        names = names.get("proposers", DEFAULT_PROPOSERS_IN_ALL)
    names = tuple(names)
    bad = [n for n in names if n not in PROPOSERS or n == "all"]
    if bad:
        raise ValueError(f"discovery.proposers_in_all has unknown proposers {bad}")
    return names


def proposers_merge(discovery_cfg: dict) -> str:
    """How ``--proposer all`` merges the proposers' lists: ``"interleaved_rank"`` when the
    protocol says so (``proposers_in_all: {"proposers": [...], "merge": "interleaved_rank"}``,
    ``discovery.proposers.merge_rule``, or a ``discovery.proposers.merge`` rule that begins
    "interleaved rank"); otherwise ``"cv_gain"`` (the v0.5 / v0.6 cross-validated ranking)."""
    names = discovery_cfg.get("proposers_in_all")
    if isinstance(names, dict) and names.get("merge"):
        return str(names["merge"])
    props = discovery_cfg.get("proposers") or {}
    if isinstance(props, dict):
        if props.get("merge_rule"):
            return str(props["merge_rule"])
        if str(props.get("merge", "")).strip().lower().startswith("interleaved rank"):
            return "interleaved_rank"
    return "cv_gain"


def ladder_config(discovery_cfg: dict, relation: str, baseline_names, null_kinds, default_draws: int) -> dict | None:
    """The protocol's ``discovery.ladder`` resolved against this episode (``None`` without one).

    The levels are ``dbench.decomposition.ladder_levels`` of the registered comparison model's
    column names, so they are the ladder script's levels by construction. Defaults: the
    within-patient permutation null at every level, every registered null at B3, the null
    suite's draws at every level, Agent L's pass rule, Agent D's tier names."""
    cfg = discovery_cfg.get("ladder")
    if cfg is None:
        return None
    if relation != "v06":
        raise ValueError(f"discovery.ladder is defined under relation 'v06' (protocol declares {relation!r})")
    if baseline_names is None:
        raise ValueError("discovery.ladder needs the comparison model's column names (episode baseline_columns)")
    from dbench import decomposition as dc
    info = dc.ladder_levels(list(baseline_names))
    names = [str(lv) for lv in cfg.get("levels", dc.LEVELS)]
    unknown = [lv for lv in names if lv not in dc.LEVELS]
    if unknown:
        raise ValueError(f"unknown ladder levels {unknown}; known {dc.LEVELS}")
    perm = dict(dc.DEFAULT_PASS_RULE, **cfg.get("pass_rule", {}))["permutation_null"]
    nulls = {lv: list(cfg.get("nulls", {}).get(lv, sorted(null_kinds) if lv == "B3" else [perm])) for lv in names}
    missing = sorted({k for ks in nulls.values() for k in ks} - set(null_kinds))
    if missing:
        raise ValueError(f"ladder nulls {missing} are not among this episode's nulls {sorted(null_kinds)}")
    draws = {lv: int(cfg.get("draws", {}).get(lv, default_draws)) for lv in names}
    return {"levels": names, "columns": {lv: [int(i) for i in info["levels"][lv]] for lv in names},
            "level_widths": {lv: info["level_widths"][lv] for lv in names}, "group_counts": info["group_counts"],
            "nulls": nulls, "draws": draws, "bootstrap_draws": int(cfg.get("bootstrap_draws", 200)),
            "pass_rule": dict(dc.DEFAULT_PASS_RULE, **cfg.get("pass_rule", {})),
            "tiers": list(cfg.get("tiers", dc.DEFAULT_TIER_RULES)), "budget": dict(cfg.get("budget", {}))}


def _ladder_budget(discovery_cfg: dict, max_candidates: int) -> int:
    """``ladder.budget.hypotheses_per_target`` frozen hypotheses per target, each validated at every
    level (must agree with ``proposals_per_target`` when both are declared)."""
    budget = (discovery_cfg.get("ladder") or {}).get("budget", {}).get("hypotheses_per_target")
    if budget is None:
        return max_candidates
    declared = discovery_cfg.get("proposals_per_target")
    if declared is not None and int(declared) != int(budget):
        raise ValueError(f"ladder.budget.hypotheses_per_target={budget} disagrees with proposals_per_target={declared}")
    return int(budget)


def _ladder_timing(rows: list, cfg: dict) -> dict:
    """Seconds per level summed over the frozen hypotheses, and per hypothesis (mean, max); B3's
    include the row's own fit share, CI and nulls, which the ladder reuses."""
    out = {}
    for lv in cfg["levels"]:
        # B3 reuses the row's own receipt, so its cost is the row's validation time.
        secs = [row["ladder"]["levels"][lv]["seconds"]["total"]
                + (sum(row["validation_seconds"].values()) if row["ladder"]["levels"][lv].get("reuses_row") else 0.0)
                for row in rows if "ladder" in row]
        out[lv] = {"total": round(float(np.sum(secs)), 2) if secs else 0.0,
                   "per_hypothesis_mean": round(float(np.mean(secs)), 3) if secs else None,
                   "per_hypothesis_max": round(float(np.max(secs)), 3) if secs else None,
                   "n_hypotheses": len(secs), "nulls": cfg["nulls"][lv], "draws": cfg["draws"][lv]}
    return out


def _hest_ladder_flags(episode, target: str, z_p, p_groups, column_names, gene_symbols):
    """Agent D's two flags for a kidney hypothesis: ``target_programme_overlap`` (programme targets:
    the variable is, shares >= 20 % of genes with, or is the neighbour mean of the target programme)
    and ``leak_guarded_in_v05`` (own_mp targets: within-patient |r| >= 0.9 to the target's own k=6
    neighbour mean on proposal rows). Both ``None`` where they do not apply."""
    from dbench import decomposition as dc
    from dbench.scripts.decomposition_ladder import _own_target_reference, bank_gene_sets, resolved_target

    resolved = resolved_target({"target": target, "own_mp_program": episode.metadata.get("own_mp_program"),
                                "nbr_mp_program": episode.metadata.get("nbr_mp_program")})
    gene_sets = bank_gene_sets() if resolved.startswith(("own:", "nbr:")) else None
    reference = _own_target_reference(episode) if str(target).startswith("own_mp") else None
    if reference is not None:
        reference = reference[episode.partitions.proposal]

    def flags(candidate) -> dict:
        names = _proposer_fields(candidate, column_names, {}, gene_symbols)["variable_names"]
        overlap = None if gene_sets is None else dc.target_programme_overlap(names, resolved, gene_sets)
        out = {"resolved_target": resolved,
               "target_programme_overlap": None if overlap is None else overlap["target_programme_overlap"],
               "target_programme_overlap_detail": overlap, "leak_guard_max_abs_r": None, "leak_guarded_in_v05": None}
        if reference is not None:
            rs = [abs(dc.within_patient_r(z_p[:, j], reference, np.asarray(p_groups))) for j in candidate.variables]
            out["leak_guard_max_abs_r"] = max(rs)
            out["leak_guarded_in_v05"] = bool(max(rs) >= 0.9)
        return out
    return flags


def _ladder_block(candidate, receipt, main_nulls: dict, *, z_p, p_y, z_r, r_y, p_b, r_b, p_groups, r_groups,
                  nulls: dict, n_null: int, cfg: dict, main_null_gains: dict | None = None) -> dict:
    """Validate one frozen hypothesis at every ladder level under relation v06 (proposal fits only).

    Per level: held-out gain, its patient-bootstrap CI (the runner's ``_gain_interval`` draws),
    held-out R^2 of the full model and of the competitor, the level's nulls, both ridge
    penalties, the linear part (and, for one variable, the part beyond linear) with CIs, the
    shape label, the pass checks and the seconds spent. A level whose columns are the whole
    comparison model (B3) reuses the row's own receipt and the nulls it already ran.

    For pooling across folds (out-of-fold predictions stacked, patient bootstrap) every level
    also records the held-out squared error per patient of the competitor (``ref``), the
    linear model (``lin``) and the full model, and every null its per-draw held-out gains;
    the block records each held-out patient's n, sum y, sum y^2."""
    from dbench import decomposition as dc
    width = p_b.shape[1]
    seed = int(candidate.candidate_hash[:8], 16)
    held = {str(q): np.flatnonzero(r_groups == q) for q in np.unique(r_groups)}
    levels = {}
    for lv in cfg["levels"]:
        cols = np.asarray(cfg["columns"][lv], dtype=int)
        started = time.time()
        whole = len(cols) == width and np.array_equal(cols, np.arange(width))
        pb, rb = (p_b, r_b) if whole else (p_b[:, cols], r_b[:, cols])
        rec = receipt if whole else evaluate_frozen_candidate(
            candidate, z_p, p_y, z_r, r_y, proposal_baseline=pb, replication_baseline=rb, model="rbf",
            proposal_groups=p_groups, alpha_grid=VALIDATION_ALPHA_GRID, relation="v06", replication_groups=r_groups)
        fitted = _FITTED_CACHE.get(_fit_key(candidate.candidate_hash, z_p, p_y, None, pb))
        if fitted is None or fitted.relation != "v06":
            fitted = FittedRelation(candidate, z_p, p_y, context=None, baseline=pb, model="rbf", ridge_alpha=1.0,
                                    rbf_features=24, groups=p_groups, alpha_grid=VALIDATION_ALPHA_GRID,
                                    relation="v06")
        simple, full = fitted.predictions(z_r, context=None, baseline=rb)
        ci = _patient_bootstrap(r_y, {"simple": simple, "full": full}, {"gain": ("full", "simple")},
                                r_groups, cfg["bootstrap_draws"], seed)["gain"]
        lin = fitted.linear_fit.predict(fitted._v06_design(z_r, None, rb, "linear"))
        sse = {m: {q: float(np.sum((r_y[ix] - pred[ix]) ** 2)) for q, ix in held.items()}
               for m, pred in (("ref", simple), ("lin", lin), ("full", full))}
        fit_seconds = time.time() - started
        null_out, null_seconds = {}, {}
        for kind in cfg["nulls"][lv]:
            draws = cfg["draws"][lv]
            if kind == "interaction_break" and (len(candidate.variables) < 2 or candidate.feature_form != "product"):
                continue
            t0 = time.time()
            if whole and kind in main_nulls and draws == n_null and kind in (main_null_gains or {}):
                null_out[kind] = dict(main_nulls[kind], reused_from_row=True,
                                      gains=list(map(float, main_null_gains[kind])))
            else:
                generator = nulls[kind](candidate) if kind == "interaction_break" else nulls[kind]
                null_receipt = evaluate_structured_nulls(rec, z_p, p_y, proposal_baseline=pb, replication_baseline=rb,
                                                         null_generator=generator, n_null=draws)
                null_out[kind] = {"p": null_receipt.null_p_value, "n": int(draws),
                                  "gains": list(map(float, null_receipt.null_gains))}
            null_seconds[kind] = round(time.time() - t0, 3)
        d = rec.decomposition or {}
        linear_key = "gain_linear" if len(candidate.variables) == 1 else "gain_linear_additive"
        entry = {
            "width": int(len(cols)), "gain": float(rec.replication_gain), "ci": ci,
            "r2_heldout": float(rec.replication_r2_full), "r2_competitor_heldout": float(rec.replication_r2_simple),
            "nulls": null_out, "sse_by_patient": sse,
            "zero_block_models": d.get("zero_block_models", []),
            "penalties": {"comparison": d.get("alpha_pair_full", [None])[0],
                          "block": d.get("alpha_pair_full", [None, None])[1],
                          "comparison_alone": d.get("alpha_comparison"),
                          "competitor_pair": d.get("alpha_pair_simple")},
            "linear_part": {"gain": d.get(linear_key), "ci": d.get(f"{linear_key}_ci"), "kind": linear_key},
            "shape": d.get("shape"),
        }
        if len(candidate.variables) == 1:
            entry["beyond_linear"] = {"gain": d.get("gain_nonlinear_beyond_linear"),
                                      "ci": d.get("gain_nonlinear_beyond_linear_ci")}
        decision = dc.level_passes(entry, cfg["pass_rule"])
        entry["pass"], entry["pass_checks"] = decision["pass"], decision["checks"]
        entry["seconds"] = {"fit_and_ci": round(fit_seconds, 3), "nulls": null_seconds,
                            "total": round(time.time() - started, 3)}
        if whole:
            entry["reuses_row"] = True  # gain, R2, penalties and matching nulls are the row's own
        levels[lv] = _json_safe(entry)
    y_by_patient = {q: {"n": int(len(ix)), "sy": float(r_y[ix].sum()), "yy": float(r_y[ix] @ r_y[ix])}
                    for q, ix in held.items()}
    return {"relation": "v06", "levels": levels, "y_by_patient": y_by_patient,
            **dc.tier_from_levels(levels, cfg["tiers"])}


def _json_safe(value):
    """Plain JSON values; a non-finite float (the zero model's infinite penalty) becomes a string."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else str(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _evaluate_rows(episode, proposed: Proposal, *, p_y, r_y, p_b, r_b, p_groups, n_null: int, min_gain: float,
                   fdr: float, arm: str, r_x, gene_symbols: list, max_candidates: int, v04: bool,
                   relation: str = "v04", ladder: dict | None = None, flags_fn=None) -> list:
    """Freeze -> held-out fit -> structured nulls -> BH -> validation, one row per candidate.

    With ``v04`` (a relationship family) ``validated`` means exactly: held-out
    full-model R^2 > 0 and gain-interval lower bound > 0 (the per-fold gain
    and per-target BH thresholds move to the family pooling step); validated
    rows carry the A-D explanation decomposition.  Without ``v04`` the v0.3
    rule is unchanged.

    ``ladder`` (protocol v0.6, :func:`ladder_config`): every frozen row also
    carries ``ladder`` -- the hypothesis validated at each nested comparison
    level with its pass checks and tier (none dropped) -- and ``flags_fn(candidate)``
    adds the substrate's flags beside it.
    """
    if relation == V03_RELATION:   # the governing spec's family, batched on the GPU
        return _evaluate_rows_v03(episode, proposed, p_y=p_y, r_y=r_y, p_b=p_b, r_b=r_b, p_groups=p_groups,
                                  n_null=n_null, fdr=fdr, arm=arm, r_x=r_x, gene_symbols=gene_symbols,
                                  max_candidates=max_candidates)
    z_p, z_r, nulls = proposed.z_p, proposed.z_r, proposed.nulls
    group_key = "patient_codes" if "patient_codes" in episode.context else "donor_codes"
    groups = np.asarray(episode.context[group_key])
    r_groups = groups[episode.partitions.replication]
    gene_provenance = tuple({"feature": symbol, "view": "observed"} for symbol in gene_symbols)
    rows = []
    fit_started = time.time()
    fitted_rows = run_episode(z_p, p_y, z_r, r_y, max_candidates=max_candidates, model="rbf",
                                          proposal_baseline=p_b, replication_baseline=r_b,
                                          proposal_groups=p_groups, candidates=proposed.candidates,
                                          alpha_grid=VALIDATION_ALPHA_GRID if v04 else None,
                                          relation=relation,
                                          replication_groups=r_groups if relation != "v04" else None)
    fit_share = (time.time() - fit_started) / max(len(fitted_rows), 1)
    for candidate, receipt in fitted_rows:
        row_started = time.time()
        null_summary, row_null_gains = {}, {}
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
            row_null_gains[kind] = null_receipt.null_gains
        combined_p = max((v["p"] for v in null_summary.values() if v["p"] is not None), default=None)
        rows.append({
            "candidate_hash": candidate.candidate_hash, "variables": list(candidate.variables),
            "gene_symbols": [gene_symbols[i] for i in candidate.variables if i < len(gene_symbols)],
            "feature_form": candidate.feature_form, "proposal_family": candidate.proposal_family,
            "proposal_score": candidate.proposal_score, "proposal_gain": receipt.proposal_gain,
            "replication_gain": receipt.replication_gain, "baseline_width": receipt.baseline_width,
            "replication_gain_ci": _gain_interval(candidate, z_p, p_y, z_r, r_y, p_b, r_b, p_groups, r_groups,
                                                  fixed_alphas=(receipt.alpha_simple, receipt.alpha_full),
                                                  relation=relation,
                                                  alpha_grid=VALIDATION_ALPHA_GRID if v04 else None),
            "competitor_alpha": _alpha_value(receipt.alpha_simple), "full_alpha": _alpha_value(receipt.alpha_full),
            "competitor_r2_replication": receipt.replication_r2_simple,
            "full_r2_replication": receipt.replication_r2_full,
            "replication_r2_full": receipt.replication_r2_full,
            "r2": {"proposal_simple": receipt.proposal_r2_simple, "proposal_full": receipt.proposal_r2_full,
                   "replication_simple": receipt.replication_r2_simple, "replication_full": receipt.replication_r2_full},
            "nulls": null_summary, "_combined_null_p": combined_p,
            **({} if relation == "v04" else {"relation": relation, "decomposition": receipt.decomposition}),
            **({"higher_order": _higher_order(receipt.decomposition, receipt.replication_r2_full)}
               if relation in HO_RELATIONS else {}),
            **_proposer_fields(candidate, proposed.column_names, proposed.also, gene_symbols),
            "proposal_cv_gain": proposed.ranking.get(candidate.candidate_hash),
            "observed_coordinate_links": (None if arm == "source" else _top_observed_links(
                z_r, r_x, tuple(candidate.variables), gene_provenance)),
            "_candidate": candidate,
        })
        if ladder is not None:
            # The row's own fit, CI and nulls are the B3 level's cost (the ladder reuses them).
            rows[-1]["validation_seconds"] = {"fit_share": round(fit_share, 3),
                                              "ci_and_nulls": round(time.time() - row_started, 3)}
            rows[-1]["ladder"] = _ladder_block(candidate, receipt, null_summary, z_p=z_p, p_y=p_y, z_r=z_r, r_y=r_y,
                                               p_b=p_b, r_b=r_b, p_groups=p_groups, r_groups=r_groups, nulls=nulls,
                                               n_null=n_null, cfg=ladder, main_null_gains=row_null_gains)
            if flags_fn is not None:
                rows[-1]["ladder"]["flags"] = _json_safe(flags_fn(candidate))
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
        if v04:
            row["validated"] = bool(row["replication_r2_full"] > 0.0 and row["replication_gain_ci"][0] > 0.0)
        else:
            row["validated"] = bool(row["replication_gain"] >= min_gain and row["q_value"] is not None
                                    and row["q_value"] <= fdr and nulls_ok
                                    and row["r2"]["replication_full"] > 0.0)
        candidate = row.pop("_candidate")
        row.pop("_combined_null_p", None)
        if v04 and row["validated"]:
            row["explanation"] = _explanation(candidate, episode, z_p, z_r, p_y, r_y, p_b, r_b, groups)
    return ranked


def _higher_order(decomposition, replication_r2_full: float | None = None) -> bool:
    """v07_ho row flag: the G_HO interval's lower bound > 0 AND the full relation beats the
    held-out mean (R^2 > 0; a joint model that is merely less negative than its decompositions is
    a transfer failure, not joint information).  False for one variable."""
    ci = (decomposition or {}).get("G_HO_ci")
    positive = replication_r2_full is None or replication_r2_full > 0.0
    return bool(ci is not None and ci[0] is not None and ci[0] > 0 and positive)


def _alpha_value(alpha):
    """JSON-safe ridge strength: the zero model (infinite shrinkage) is "inf"."""
    return None if alpha is None else (float(alpha) if np.isfinite(alpha) else "inf")


def competitor_alone(p_b, p_y, r_b, r_y, p_groups) -> dict:
    """Held-out R^2 of the competitor with no candidate: the v0.3 default
    grid versus the v0.4 inner-CV grid that includes the zero model."""
    out = {}
    for label, grid in (("default_grid", FittedRelation.ALPHA_GRID), ("inner_cv_with_zero", VALIDATION_ALPHA_GRID)):
        alpha = _cv_alpha(p_b, p_y, p_groups, grid, 0)
        out[label] = {"alpha": _alpha_value(alpha), "r2_replication": _r2(r_y, _RidgeFit(p_b, p_y, alpha).predict(r_b))}
    return out


def _explanation(candidate, episode, z_p, z_r, p_y, r_y, p_b, r_b, groups) -> dict:
    """Agent T's A-D decomposition of a validated relation (fits on proposal rows only)."""
    from morpheus_encoder.general_discovery import decompose_explanation

    proposal, replication = episode.partitions.proposal, episode.partitions.replication
    n = episode.n_rows
    X = np.empty((n, z_p.shape[1]))
    X[proposal], X[replication] = z_p, z_r
    y = np.empty(n)
    y[proposal], y[replication] = p_y, r_y
    B = np.empty((n, p_b.shape[1]))
    B[proposal], B[replication] = p_b, r_b
    result = decompose_explanation(candidate, X, y, groups=groups, proposal=proposal, replication=replication,
                                   baseline=B)
    return json.loads(json.dumps(result, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))


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
                     program_sets: list | None = None, proposer: str = "enumeration",
                     family: str | None = None, relation_override: str | None = None,
                     learned_checkpoint: str | None = None, fixed_hypotheses: list | None = None) -> dict:
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
    # v0.5 substrate (wider competitor, leak guard, nuclear targets, aligned PCA/NMF) is
    # selected by the protocol; absent key = the registered v0.4 substrate, byte-identical.
    substrate_version = str(discovery_cfg.get("substrate_version", "v04"))
    # Protocol v0.5 may declare ``discovery.relation = "v05"``; absent -> v04.
    relation = str(discovery_cfg.get("relation", "v04"))
    if relation_override is not None:  # development runs only (--relation); recorded in the output
        relation = str(relation_override)
    # A relationship family (protocol v0.4) fixes the candidate views and
    # always attaches the bank, so every proposer sees identical rows.
    family_cfg = family_config(protocol, family, target)
    max_candidates = _ladder_budget(discovery_cfg, max_candidates)
    n_hvg = int(protocol.get("n_hvg", n_hvg))
    min_depth = int(protocol.get("min_depth", min_depth))

    load_started = time.time()
    episode = hest.real_episode(fold=fold, n_folds=n_folds, seed=fold_seed, target=target,
                                embed_root=embed_root, sample_ids=sample_ids, cohort=cohort,
                                data_dir=data_dir, n_hvg=n_hvg, min_depth=min_depth,
                                program_sets=program_sets,
                                bank=proposer != "enumeration" or family_cfg is not None,
                                substrate_version=substrate_version)
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
    null_sub = hest._NullSubstrate(np.asarray(episode.context["coords"])[replication],
                                   np.asarray(episode.context["patient_codes"])[replication])
    group_key = "patient_codes" if "patient_codes" in episode.context else "donor_codes"
    p_groups = np.asarray(episode.context[group_key])[proposal] if group_key in episode.context else None
    load_seconds = time.time() - load_started
    baseline_names = episode.metadata.get("baseline_columns")
    ladder = ladder_config(discovery_cfg, relation, baseline_names, nulls, n_null)
    in_all = proposers_in_all(discovery_cfg)
    proposal_started = time.time()
    proposed = _propose(episode, z_p, z_r, nulls, p_y=p_y, p_b=p_b, p_groups=p_groups, target=target,
                        proposer=proposer, family_cfg=family_cfg, seed=seed, device=device, relation=relation,
                        break_column=lambda index, column: hest.null_hest(column, null_sub, seed=index, nx=6, ny=6), max_candidates=max_candidates,
                        baseline_names=baseline_names, in_all=in_all, learned_checkpoint=learned_checkpoint,
                        merge=proposers_merge(discovery_cfg),
                        fixed_hypotheses=fixed_hypotheses)
    nulls = proposed.nulls
    proposal_seconds = time.time() - proposal_started
    validation_started = time.time()
    ranked = _evaluate_rows(episode, proposed, p_y=p_y, r_y=r_y, p_b=p_b, r_b=r_b, p_groups=p_groups,
                            n_null=n_null, min_gain=min_gain, fdr=fdr, arm=arm, r_x=r_x,
                            gene_symbols=gene_symbols, max_candidates=max_candidates, v04=family_cfg is not None,
                            relation=relation, ladder=ladder,
                            flags_fn=None if ladder is None else _hest_ladder_flags(
                                episode, target, proposed.z_p, p_groups, proposed.column_names, gene_symbols))
    ladder_fields = {} if ladder is None else {"relation": relation, "ladder_config": ladder,
                                               "ladder_timing_seconds": _ladder_timing(ranked, ladder)}
    if relation_override is not None:
        ladder_fields = {**ladder_fields, "relation": relation, "relation_override": True}
    if fixed_hypotheses is not None:
        for row in ranked:
            row["frozen_in_fold"] = False
        ladder_fields = {**ladder_fields, "crossfold": True}

    return {
        **ladder_fields,
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
        "timing_seconds": {"load": round(load_seconds, 1), "proposal": round(proposal_seconds, 1),
                           "validation": round(time.time() - validation_started, 1)},
        "substrate_version": substrate_version, "proposer": proposer, "proposer_report": proposed.report,
        "competitor_alone": None if family_cfg is None else competitor_alone(p_b, p_y, r_b, r_y, p_groups),
        "design": proposed.design, "family": family, "candidate_views_resolved": proposed.views,
        "own_mp_program": episode.metadata.get("own_mp_program"),
        "elapsed_seconds": time.time() - started,
    }


# --------------------------------------------------------------------------
# hest_multi episodes (blind-discovery protocols renal_spatial_v07 / ependymoma_spatial_v01):
# any study of sprint_partitions_v01 through morpheus.tier2.substrates.hest_multi.  The
# hest_real path above is untouched.
# --------------------------------------------------------------------------

#: protocol candidate view -> hest_multi source ("concepts" = the learned concept layer)
HEST_MULTI_VIEWS = {"raw_genes": "genes", "programmes": "programmes", "morph_concepts": "concepts",
                    "morph_dictionary_v03": "morph_v03"}
#: view key of the frozen v03 morphology dictionary: cosine similarities of the frozen H-Optimus-0
#: spot embedding, standardised per dimension on the fold's covered support (D_P) spots, to the
#: kept centres of runs/development/learned_search_v03/morphology_dictionary (columns
#: ``morph_dictionary_v03:morph:kNN``, NN = centre index); targets ``morph:kNN`` are the same
#: similarity, within-patient standardised (the morphology view then drops the target and every
#: column with within-patient |r| >= 0.9 to it on support rows).
MORPH_V03_VIEW = "morph_dictionary_v03"
MOLECULAR_VIEW_KEYS = ("raw_genes", "programmes", "scimilarity", "scgpt")
MORPHOLOGY_VIEW_KEYS = ("morph_concepts", "morph_dictionary_v03", "hoptimus", "cellvit_structure")


def family_requires_cross_modal(family_cfg: dict | None) -> bool:
    """A family whose requirement asks for a molecular and a morphology variable (cross-modal sets)."""
    req = str((family_cfg or {}).get("requirement", "")).lower()
    return "morpholog" in req and "molecular" in req
N_CONCEPTS = 64
N_STUB_CONCEPT_TARGETS = 16


def hest_multi_folds(patients, n_folds: int, fold: int, fold_seed: int) -> tuple[list, list]:
    """Deterministic grouped-patient folds: patients sorted, shuffled with ``fold_seed``, dealt
    round-robin into ``min(n_folds, n_patients)`` folds; fold ``fold`` is the held-out set."""
    patients = sorted(map(str, patients))
    n_f = min(int(n_folds), len(patients))
    if not 0 <= fold < n_f:
        raise ValueError(f"fold {fold} outside 0..{n_f - 1} ({len(patients)} patients)")
    perm = np.random.default_rng(int(fold_seed)).permutation(len(patients))
    fold_of = {patients[j]: i % n_f for i, j in enumerate(perm)}
    query = [p for p in patients if fold_of[p] == fold]
    return [p for p in patients if fold_of[p] != fold], query


def _within_standardise(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    v = np.array(values, dtype=float, copy=True)
    v = v.reshape(len(v), -1)
    for g in np.unique(groups):
        r = groups == g
        v[r] -= v[r].mean(0)
        sd = v[r].std(0)
        v[r] /= np.where(sd > 1e-12, sd, 1.0)
    return v


def concept_matrix(hoptimus: np.ndarray, covered: np.ndarray, support_rows: np.ndarray, *, model=None,
                   n: int = N_CONCEPTS) -> tuple[np.ndarray, dict]:
    """Concept coordinates of every row: the frozen model's concept layer on the H-Optimus-0
    embedding, or (stub, no checkpoint) the first ``n`` principal components of the embedding
    fitted on covered support rows.  Uncovered rows are returned but never used."""
    E = np.asarray(hoptimus, dtype=np.float32)
    if model is not None:
        C = np.asarray(model.concept_values(E), dtype=float)[:, :n]
        return C, {"source": "learned concept layer", "checkpoint_meta": dict(getattr(model, "meta", {}) or {})}
    fit = np.intersect1d(np.asarray(support_rows), np.flatnonzero(covered))
    if len(fit) > 20000:
        fit = np.sort(np.random.default_rng(0).choice(fit, 20000, replace=False))
    mu = E[fit].astype(np.float64).mean(0)
    _, _, vt = np.linalg.svd(E[fit].astype(np.float64) - mu, full_matrices=False)
    C = (E.astype(np.float64) - mu) @ vt[:n].T
    return C, {"source": "stub: H-Optimus-0 principal components fitted on covered support rows "
                         "(no learned-search checkpoint)", "stub": True}


#: fixed blind programme targets per study (Agent D: top 8 Gavish by median within-section
#: Moran's I over all sections, computed once, response-free)
BLIND_TARGETS = "runs/development/sprint_v06/blind_targets_{study}.json"


def _moran_rank(study: str, rank: int, path: str | None = None) -> tuple[str, dict]:
    """``own_mp<rank>`` -> the study's fixed blind programme target (``BLIND_TARGETS``; the same
    programme in every fold).  A missing file is an error, never a per-fold re-ranking."""
    root = Path(__file__).resolve().parents[2]
    f = Path(path) if path else root / BLIND_TARGETS.format(study=study)
    if not f.exists():
        raise FileNotFoundError(f"own_mp targets of {study} need the fixed blind-target file {f} (Agent D)")
    raw_bytes = f.read_bytes()
    spec = json.loads(raw_bytes)
    entry = next((t for t in spec["targets"] if int(t["rank"]) == int(rank)), None)
    if entry is None:
        raise ValueError(f"{f.name} has no rank {rank}")
    name = str(entry["programme"])
    return name, {"rank": int(rank), "program_name": name[len("mp:"):] if name.startswith("mp:") else name,
                  "loader_target": entry["loader_target"], "median_moran_i": entry.get("median_moran_i"),
                  "source": str(f.relative_to(root)) if f.is_relative_to(root) else str(f),
                  "source_sha256": sha256(raw_bytes).hexdigest(), "rule": spec.get("rule")}


def _v03_raw_inputs(spec, ep, keep, names, *, target, loader_target, morph_target, M_v03, morph_cols, study,
                    concepts=None):
    """Un-centred values of the design columns and the target on the kept rows, with log depth,
    the composition variables (CellViT fractions or programme axes, as the training studies) and
    sections: what a v03 gain predictor needs to see the training-episode construction."""
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.episodes_v03 import composition

    raw = hm.load_study(spec)
    index = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(raw.sample_id, raw.barcode))}
    rows = np.array([index[(str(a), str(b))] for a, b in zip(np.asarray(ep.sample_id)[keep],
                                                              np.asarray(ep.barcode)[keep])])
    L = raw.logcpm(rows).tocsc()
    gidx = {str(g): j for j, g in enumerate(raw.genes)}
    pidx = {str(c): j for j, c in enumerate(raw.programme_names)}
    P = np.asarray(raw.programmes, dtype=np.float64)[rows]
    mcol = {c: j for j, c in enumerate(morph_cols)}

    def column(name):
        view, col = name.split(":", 1)
        if view == "raw_genes":
            return np.asarray(L[:, gidx[col]].todense()).ravel()
        if view == "programmes":
            return P[:, pidx[col]]
        if view == MORPH_V03_VIEW:
            return M_v03[keep][:, mcol[col]]
        if view == "morph_concepts" and concepts is not None:   # the checkpoint's own morphology values
            return np.asarray(concepts)[keep][:, int(col)]
        raise ValueError(f"no un-centred source for {name}")

    X = np.column_stack([column(n) for n in names])
    if morph_target is not None:
        y = M_v03[keep][:, mcol[morph_target]]
    else:
        lt = loader_target.split(":", 1)
        y = P[:, pidx[lt[1]]] if lt[0] == "programme" else np.asarray(L[:, gidx[lt[1]]].todense()).ravel()
    # composition exactly as the training studies: computed over the study's covered rows in
    # loader order (episodes_v03.study_v03_from_raw), then indexed for these rows
    keep_all = (np.flatnonzero(np.asarray(raw.covered, dtype=bool)) if raw.hoptimus is not None
                else np.arange(raw.n))
    comp_all, _ = composition(raw, spec, keep_all, np.asarray(raw.programmes, dtype=np.float64)[keep_all], name=study)
    pos = -np.ones(raw.n, dtype=np.int64)
    pos[keep_all] = np.arange(len(keep_all))
    comp = comp_all[pos[rows]]
    return {"X": X, "y": y, "depth": np.log1p(np.asarray(raw.depth, dtype=np.float64)[rows]),
            "composition": np.asarray(comp, dtype=np.float64), "sections": np.asarray(raw.sample_id).astype(str)[rows],
            "raw_rows": rows}


def build_hest_multi_episode(*, study: str, fold: int, n_folds: int, fold_seed: int, target: str,
                             family_cfg: dict, seed: int, smoke_samples: int | None = None,
                             checkpoint: str | None = None, device: str = "cpu"):
    """(DiscoveryEpisode, HypothesisDesign, info) for one fold and protocol target.

    Targets: ``own_mp<k>`` (k-th Moran-ranked Gavish programme -> loader ``programme:mp:<name>``),
    ``morph_concept_<k>`` (concept k, within-patient standardised, covered spots only), or a
    loader target (``programme:...``, ``gene:...``).  Candidate views: ``raw_genes``,
    ``programmes``, ``morph_concepts``; views the study lacks are recorded as unavailable."""
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus.tier2.substrates.hest import _within_patient_standardise
    from morpheus_encoder.hypothesis_generator import HypothesisDesign
    from morpheus_encoder.learned_search.propose import load_model

    spec = hm.study_specs()[study]
    patients = sorted(set(map(str, spec.patient_of.values())))
    if smoke_samples:
        patients = patients[:int(smoke_samples)]
    support, query = hest_multi_folds(patients, n_folds, fold, fold_seed)
    model = load_model(checkpoint, device=device) if checkpoint else None
    info: dict = {"study": study, "support_patients": support, "query_patients": query,
                  "n_patients_used": len(patients), "smoke_samples": smoke_samples}
    concept_k = None
    morph_target = None
    if target.startswith("morph:k"):
        morph_target = target
        loader_target = "concept:0"   # hest_multi serves no response for it; the similarity is set below
    elif target.startswith("own_mp"):
        _, program = _moran_rank(study, int(target[len("own_mp"):]))
        loader_target = program["loader_target"]
        info["own_mp_program"] = program
    elif target.startswith("morph_concept_"):
        concept_k = int(target[len("morph_concept_"):])
        loader_target = f"concept:{concept_k}"
    elif morph_target is None:  # a named loader target, e.g. programme:mp:<name> (resolved directly by name)
        loader_target = target
    info["loader_target"] = loader_target
    ep = hm.study_episode(spec, target=loader_target, support_patients=support, query_patients=query, seed=seed,
                          with_morphology=True)
    groups_all = np.unique(ep.patient, return_inverse=True)[1]
    M_v03, morph_cols_v03 = None, []
    if morph_target is not None or MORPH_V03_VIEW in family_cfg["candidate_views"]:
        if ep.views.get("hoptimus") is None:
            raise ValueError(f"{study}: no H-Optimus-0 embeddings, so the v03 morphology view is unavailable")
        from morpheus_encoder.learned_search.morphology_dictionary import (load_dictionary_v03, similarities_v03,
                                                                            standardise_stats)
        dct = load_dictionary_v03()
        E = np.asarray(ep.views["hoptimus"], dtype=np.float32)
        fit_rows = np.intersect1d(np.asarray(ep.support_rows), np.flatnonzero(ep.covered))
        mu_e, sd_e = standardise_stats(E[fit_rows])
        M_v03 = similarities_v03(E, dct["centres"], mu_e, sd_e).astype(np.float64)
        morph_cols_v03 = [f"morph:k{j:02d}" for j in dct["kept"]]
        info["morph_dictionary_v03"] = {"kept_centres": len(morph_cols_v03), "standardisation": "covered support rows",
                                        "n_fit_rows": int(len(fit_rows))}
    if morph_target is not None:
        if morph_target not in morph_cols_v03:
            raise ValueError(f"{morph_target} is not a kept v03 dictionary centre")
        jt = morph_cols_v03.index(morph_target)
        ep.y = _within_patient_standardise(M_v03[:, [jt]], groups_all).ravel().astype(np.float32)
        info["morph_target"] = {"name": morph_target, "dictionary": "v03"}
    need_concepts = concept_k is not None or "morph_concepts" in family_cfg["candidate_views"]
    C = None
    if need_concepts:
        if ep.views.get("hoptimus") is None:
            raise ValueError(f"{study}: no H-Optimus-0 embeddings, so concept targets / morph_concepts are unavailable")
        C, info["concepts"] = concept_matrix(ep.views["hoptimus"], ep.covered, ep.support_rows, model=model)
    if concept_k is not None:
        if model is None and concept_k >= N_STUB_CONCEPT_TARGETS:
            raise ValueError(f"stub concept targets are the first {N_STUB_CONCEPT_TARGETS} principal components")
        ep.y = _within_patient_standardise(C[:, [concept_k]], groups_all).ravel().astype(np.float32)
        info["concept_target"] = {"index": concept_k, **info["concepts"]}
    dep = ep.to_discovery_episode()
    keep = (np.flatnonzero(ep.covered) if ep.views.get("hoptimus") is not None and not ep.covered.all()
            else np.arange(ep.n))
    proposal = dep.partitions.proposal
    groups = np.asarray(dep.context["patient_codes"])
    # design: requested views, within-patient centred, z-scored on support rows
    blocks, names, views, unavailable, start = [], [], {}, {}, 0
    for view in family_cfg["candidate_views"]:
        src = HEST_MULTI_VIEWS.get(view)
        if src is None:
            unavailable[view] = "not served by hest_multi"
            continue
        if src == "morph_v03":
            M, cols = M_v03[keep], [f"{view}:{c}" for c in morph_cols_v03]
        elif src == "concepts":
            if concept_k is not None:
                unavailable[view] = "excluded: the target is a function of the same embedding"
                continue
            M, cols = C[keep], [f"morph_concepts:{j}" for j in range(C.shape[1])]
        else:
            M = dep.modalities.get(src)
            if M is None:
                unavailable[view] = "absent for this study"
                continue
            cols = [f"{view}:{c}" for c in dep.metadata["columns"][src]]
        M = _within_standardise(np.asarray(M, dtype=float), groups)
        sd = M[proposal].std(0)
        M = (M - M[proposal].mean(0)) / np.where(sd > 1e-8, sd, 1.0)
        blocks.append(M)
        names += cols
        views[view] = np.arange(start, start + M.shape[1])
        start += M.shape[1]
    if not blocks:
        raise ValueError(f"no candidate view available for {study} ({unavailable})")
    X = np.column_stack(blocks)
    # leak guard (own_mp targets): drop columns with within-patient |r| >= 0.9 to the target's own
    # k=6 neighbour mean on support rows (the target itself is already excluded by hest_multi)
    dropped = []
    if target.startswith(("own_mp", "programme:")):
        from dbench.benchmark.substrate import neighbour_mean
        from morpheus_encoder.learned_search.propose import pad_neighbours
        nb = pad_neighbours(dep.context["neighbors"], dep.n_rows)
        lag = _within_standardise(neighbour_mean(np.asarray(dep.response, dtype=float), nb), groups).ravel()
        Xc = X[proposal]
        r = np.abs((Xc * lag[proposal, None]).mean(0) / np.maximum(Xc.std(0) * lag[proposal].std(), 1e-12))
        bad = np.flatnonzero(r >= 0.9)
        if len(bad):
            dropped = [names[j] for j in bad]
            keep_cols = np.setdiff1d(np.arange(X.shape[1]), bad)
            remap = -np.ones(X.shape[1], dtype=int)
            remap[keep_cols] = np.arange(len(keep_cols))
            X = X[:, keep_cols]
            names = [names[j] for j in keep_cols]
            views = {v: remap[c][remap[c] >= 0] for v, c in views.items()}
    if morph_target is not None:   # the training proxy rule: drop the target and |r| >= 0.9 morphology columns
        yt = np.asarray(dep.response, dtype=float)
        Xc = X[proposal]
        r = np.abs((Xc * yt[proposal, None]).mean(0) / np.maximum(Xc.std(0) * yt[proposal].std(), 1e-12))
        bad = [j for j in np.flatnonzero(r >= 0.9) if names[j].split(":", 1)[0] in MORPHOLOGY_VIEW_KEYS]
        if bad:
            dropped += [names[j] for j in bad]
            keep_cols = np.setdiff1d(np.arange(X.shape[1]), bad)
            remap = -np.ones(X.shape[1], dtype=int)
            remap[keep_cols] = np.arange(len(keep_cols))
            X = X[:, keep_cols]
            names = [names[j] for j in keep_cols]
            views = {v: remap[c][remap[c] >= 0] for v, c in views.items()}
    design = HypothesisDesign(X=X, views=views, roles={v: "variable" for v in views},
                              modalities={v: v for v in views}, column_names=names, n_base=X.shape[1])
    if model is not None and getattr(model, "kind", "") == "gain_v03":
        design.v03_raw = _v03_raw_inputs(spec, ep, keep, names, target=target, loader_target=loader_target,
                                         morph_target=morph_target, M_v03=M_v03, morph_cols=morph_cols_v03,
                                         study=study, concepts=C)
    info.update({"unavailable_views": unavailable, "leak_guard_dropped": dropped,
                 "design_width": int(X.shape[1]), "rows_kept": int(dep.n_rows),
                 "coverage_rule": dep.metadata.get("coverage_rule"), "exclusions": dep.metadata.get("exclusions")})
    try:   # composition variables of the kept rows (B_known under relation v03_rff), as the training studies
        dep.context["composition_v03"], info["composition_source"] = _composition_of_rows(spec, ep, keep, study)
        # every programme score (before the target exclusions) with its names and scoring panel, so B_known
        # is built by relation_v03.known_covariates exactly as the harness and training tasks
        dep.context["programmes_all_v03"] = np.asarray(ep.views["programmes"], dtype=np.float64)[keep]
        dep.context["programme_names_v03"] = list(ep.columns["programmes"])
        dep.context["panel_v03"] = [str(g) for g in np.asarray(hm.load_study(spec).genes)]
    except Exception as exc:  # recorded, never imputed
        info["composition_source"] = f"unavailable: {type(exc).__name__}: {str(exc)[:120]}"
    return dep, design, info


def _composition_of_rows(spec, ep, keep, study):
    """(n_kept, 3) composition (CellViT PanNuke fractions or programme axes) computed over the study's
    covered rows in loader order exactly as ``episodes_v03.study_v03_from_raw``, indexed for these rows."""
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.episodes_v03 import composition

    raw = hm.load_study(spec)
    index = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(raw.sample_id, raw.barcode))}
    rows = np.array([index[(str(a), str(b))] for a, b in zip(np.asarray(ep.sample_id)[keep],
                                                              np.asarray(ep.barcode)[keep])])
    keep_all = (np.flatnonzero(np.asarray(raw.covered, dtype=bool)) if raw.hoptimus is not None
                else np.arange(raw.n))
    comp_all, source = composition(raw, spec, keep_all, np.asarray(raw.programmes, dtype=np.float64)[keep_all],
                                   name=study)
    pos = -np.ones(raw.n, dtype=np.int64)
    pos[keep_all] = np.arange(len(keep_all))
    return np.asarray(comp_all, dtype=np.float64)[pos[rows]], source


def run_hest_multi_episode(*, study: str, fold: int, target: str, family: str, protocol_path: Path, seed: int,
                           device: str, proposer: str, max_candidates: int, n_null: int, min_gain: float, fdr: float,
                           n_folds: int = 5, smoke_samples: int | None = None, relation_override: str | None = None,
                           learned_checkpoint: str | None = None, fixed_hypotheses: list | None = None) -> dict:
    """One blind-discovery packet on a hest_multi study (receipt schema as hest_real)."""
    from types import SimpleNamespace

    from dbench.prospective_campaign import canonical_hash
    from morpheus.tier2.substrates import hest

    started = time.time()
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes)
    discovery_cfg = protocol.get("discovery", {})
    max_candidates = int(discovery_cfg.get("proposals_per_target", max_candidates))
    n_null = int(discovery_cfg.get("null_suite", {}).get("draws", n_null))
    n_folds = int(discovery_cfg.get("n_folds", n_folds))
    fold_seed = int(discovery_cfg.get("fold_seed", seed))
    relation = str(relation_override or discovery_cfg.get("relation", "v04"))
    family_cfg = family_config(protocol, family, target)
    checkpoint = learned_checkpoint or os.environ.get("MORPHEUS_LEARNED_CHECKPOINT") or None
    episode, design, info = build_hest_multi_episode(study=study, fold=fold, n_folds=n_folds, fold_seed=fold_seed,
                                                     target=target, family_cfg=family_cfg, seed=seed,
                                                     smoke_samples=smoke_samples, checkpoint=checkpoint,
                                                     device=device)
    load_seconds = time.time() - started
    proposal, replication = episode.partitions.proposal, episode.partitions.replication
    y = np.asarray(episode.response, dtype=float)
    groups = np.asarray(episode.context["patient_codes"])
    depth = _within_standardise(np.asarray(episode.context["baseline"], dtype=float), groups)
    p_y, r_y, p_b, r_b = y[proposal], y[replication], depth[proposal], depth[replication]
    p_groups = groups[proposal]
    z_p, z_r = design.X[proposal], design.X[replication]
    shim = SimpleNamespace(partitions=episode.partitions, context=episode.context, response=episode.response,
                           observed_matrix=lambda: design.X)
    nulls = hest.episode_null_generators(shim)
    null_sub = hest._NullSubstrate(np.asarray(episode.context["coords"])[replication], groups[replication])
    proposal_started = time.time()
    proposed = _propose(episode, z_p, z_r, nulls, p_y=p_y, p_b=p_b, p_groups=p_groups, target=target,
                        proposer=proposer, family_cfg=family_cfg, seed=seed, device=device, relation=relation,
                        break_column=lambda index, column: hest.null_hest(column, null_sub, seed=index, nx=6, ny=6),
                        max_candidates=max_candidates, in_all=proposers_in_all(discovery_cfg),
                        learned_checkpoint=checkpoint, merge=proposers_merge(discovery_cfg), design=design,
                        fixed_hypotheses=fixed_hypotheses)
    proposal_seconds = time.time() - proposal_started
    validation_started = time.time()
    ranked = _evaluate_rows(episode, proposed, p_y=p_y, r_y=r_y, p_b=p_b, r_b=r_b, p_groups=p_groups,
                            n_null=n_null, min_gain=min_gain, fdr=fdr, arm="source", r_x=z_r, gene_symbols=[],
                            max_candidates=max_candidates, v04=True, relation=relation)
    if fixed_hypotheses is not None:
        for row in ranked:
            row["frozen_in_fold"] = False
    return {
        "schema": "morpheus_general_discovery_packet/v2", "episode": "hest_multi", "study": study,
        "episode_identity": episode.receipt_identity(),
        "protocol_sha256": canonical_hash(protocol), "protocol_file_sha256": sha256(protocol_bytes).hexdigest(),
        "fold": int(fold), "n_folds": int(n_folds), "target": target, "loader_target": info["loader_target"],
        "arm": "source", "seed": int(seed), "fold_seed": int(fold_seed), "cohort": f"hest_multi:{study}",
        "method_truth_accessed": False, "relation": relation,
        **({"relation_override": True} if relation_override else {}),
        **({"crossfold": True} if fixed_hypotheses is not None else {}),
        "own_mp_program": info.get("own_mp_program"), "episode_info": _json_safe(info),
        "baseline_width": 1, "baseline_columns": ["depth_log1p (within-patient standardised)"],
        "fit_patients": info["support_patients"], "held_patients": info["query_patients"],
        "proposal_budget": int(max_candidates), "null_kinds": sorted(proposed.nulls),
        "validation_rule": {"min_replication_incremental_gain": float(min_gain), "q_value_max": float(fdr),
                            "rule": "held-out full R2 > 0 and gain CI lower bound > 0 (v0.4 family rule)"},
        "candidates": ranked, "proposer": proposer, "proposer_report": _json_safe(proposed.report),
        "proposers_merge": proposers_merge(discovery_cfg),
        "competitor_alone": competitor_alone(p_b, p_y, r_b, r_y, p_groups),
        "design": _json_safe(proposed.design), "family": family, "candidate_views_resolved": proposed.views,
        "timing_seconds": {"load": round(load_seconds, 1), "proposal": round(proposal_seconds, 1),
                           "validation": round(time.time() - validation_started, 1)},
        "elapsed_seconds": time.time() - started,
    }


def run_bmmc_episode(*, fold: int, n_folds: int, seed: int, target: str, arm: str, device: str,
                     protocol_path: Path | None, h5ad_path: str | None, max_candidates: int, n_null: int,
                     min_gain: float, fdr: float, steps: int, n_hvg: int = 300, min_rna_depth: int = 500,
                     min_adt_depth: int = 200, min_genes: int = 200, max_cells: int | None = None,
                     proposer: str = "enumeration", family: str | None = None) -> dict:
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
    # Protocol v0.5 may declare ``discovery.relation = "v05"``; absent -> v04.
    relation = str(discovery_cfg.get("relation", "v04"))
    if "ladder" in discovery_cfg:
        raise NotImplementedError("discovery.ladder is wired for --episode hest_real only")
    family_cfg = family_config(protocol, family, target)

    episode_kwargs = dict(fold=fold, n_folds=n_folds, seed=fold_seed, target=target, n_hvg=n_hvg,
                          min_rna_depth=min_rna_depth, min_adt_depth=min_adt_depth, min_genes=min_genes,
                          max_cells=max_cells, bank=proposer != "enumeration" or family_cfg is not None)
    if h5ad_path is not None:
        episode_kwargs["h5ad_path"] = h5ad_path
    load_started = time.time()
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
    donors_r = np.asarray(episode.context["donor_codes"])[replication]

    def _within_donor_break(index, column):
        # bmmc's own interaction_break rule (within-donor permutation, rng(index)).
        rng = np.random.default_rng(index)
        out = column.copy()
        for g in np.unique(donors_r):
            idx = np.where(donors_r == g)[0]
            out[idx] = column[idx[rng.permutation(len(idx))]]
        return out
    group_key = "patient_codes" if "patient_codes" in episode.context else "donor_codes"
    p_groups = np.asarray(episode.context[group_key])[proposal] if group_key in episode.context else None
    load_seconds = time.time() - load_started
    proposal_started = time.time()
    proposed = _propose(episode, z_p, z_r, nulls, p_y=p_y, p_b=p_b, p_groups=p_groups, target=target,
                        proposer=proposer, family_cfg=family_cfg, seed=seed, device=device, relation=relation,
                        break_column=_within_donor_break, max_candidates=max_candidates)
    nulls = proposed.nulls
    proposal_seconds = time.time() - proposal_started
    validation_started = time.time()
    ranked = _evaluate_rows(episode, proposed, p_y=p_y, r_y=r_y, p_b=p_b, r_b=r_b, p_groups=p_groups,
                            n_null=n_null, min_gain=min_gain, fdr=fdr, arm=arm, r_x=r_x,
                            gene_symbols=gene_symbols, max_candidates=max_candidates, v04=family_cfg is not None,
                            relation=relation)

    return {
        "schema": "morpheus_general_discovery_packet/v2",
        "episode": "bmmc_real", "episode_identity": episode.receipt_identity(),
        "protocol_sha256": protocol_sha256, "protocol_file_sha256": protocol_file_sha256,
        "fold": int(fold), "n_folds": int(n_folds), "target": target, "arm": arm, "seed": int(seed),
        "fold_seed": int(fold_seed), "method_truth_accessed": False,
        "representation": {"arm": arm, "steps": int(steps), "loss_trace": losses, "input_width": int(p_x.shape[1])},
        "gene_symbols": gene_symbols, "baseline_width": int(baseline.shape[1]),
        "baseline_columns": list(episode.metadata["baseline_columns"]),
        "nbr_mp_program": episode.metadata.get("nbr_mp_program"),
        "fit_donors": list(episode.metadata["fit_donors"]), "held_donors": list(episode.metadata["held_donors"]),
        "proposal_budget": int(max_candidates), "null_kinds": sorted(nulls),
        "validation_rule": {"min_replication_incremental_gain": float(min_gain), "q_value_max": float(fdr),
                           "recurrence_note": "recurrence-across-folds gate applied by the funnel aggregator, not here"},
        "candidates": ranked,
        "timing_seconds": {"load": round(load_seconds, 1), "proposal": round(proposal_seconds, 1),
                           "validation": round(time.time() - validation_started, 1)},
        "proposer": proposer, "proposer_report": proposed.report,
        "competitor_alone": None if family_cfg is None else competitor_alone(p_b, p_y, r_b, r_y, p_groups),
        "design": proposed.design, "family": family, "candidate_views_resolved": proposed.views,
        "own_mp_program": episode.metadata.get("own_mp_program"),
        "elapsed_seconds": time.time() - started,
    }


def run_cptac_episode(*, fold: int, target: str, family: str, protocol_path: Path | None, seed: int = 0,
                      device: str = "cpu", proposer: str = "all", max_candidates: int = 32, n_null: int = 199,
                      n_folds: int = 5, min_gain: float = 0.01, fdr: float = 0.10, table_loader=None,
                      cohort: str = "cptac_ccrcc") -> dict:
    """One CPTAC ccRCC patient-level (tissue_to_patient) discovery packet.

    Same evaluation path as `run_hest_episode` / `run_bmmc_episode`: the
    episode is `dbench.cohorts.cptac_patient_episode` (one row per patient,
    stage-stratified patient folds, competitor = clinical + deconvolution +
    depth proxy, no spatial graph), the nulls are
    `cptac_episode_null_generators` (patient permutation, slide-descriptor
    permutation within stage, interaction break within stage), the family
    fixes the candidate views, and the relation / alpha grid / validated rule
    come from the protocol. Source arm only (the slide descriptors are frozen
    H-Optimus-0 / CellViT measurements). The receipt also carries the
    informational competitor (clinical + depth proxy) alone. ``cohort``
    (``cptac_<type>``, default ``cptac_ccrcc``) selects the CPTAC tumour type
    (``dbench.cohorts.cptac_patient_episode``)."""
    from dbench import cohorts

    protocol: dict = {}
    protocol_sha256 = protocol_file_sha256 = None
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
    fold_seed = int(discovery_cfg.get("fold_seed", seed))
    relation = str(discovery_cfg.get("relation", "v04"))
    if "ladder" in discovery_cfg:
        raise NotImplementedError("discovery.ladder is wired for --episode hest_real only")
    family_cfg = family_config(protocol, family, target) if protocol else None

    load_started = time.time()
    episode = cohorts.cptac_patient_episode(target, fold, family=family, n_folds=n_folds, seed=fold_seed,
                                            table_loader=table_loader, cohort=cohort)
    proposal, replication = episode.partitions.proposal, episode.partitions.replication
    X = episode.observed_matrix()
    y = np.asarray(episode.response, dtype=float)
    baseline = np.asarray(episode.context["baseline"], dtype=float)
    p_x, r_x = X[proposal], X[replication]
    p_y, r_y = y[proposal], y[replication]
    p_b, r_b = baseline[proposal], baseline[replication]
    gene_symbols = list(episode.metadata["gene_symbols"])
    nulls = cohorts.cptac_episode_null_generators(episode)
    stage_r = np.asarray(episode.context["stage"])[replication]
    patients = np.asarray(episode.context["patient_codes"])
    p_groups = patients[proposal]

    def _within_stage_break(index, column):
        rng = np.random.default_rng(index)
        out = column.copy()
        for s in np.unique(stage_r):
            idx = np.flatnonzero(stage_r == s)
            out[idx] = column[idx[rng.permutation(len(idx))]]
        return out

    started = time.time()
    load_seconds = started - load_started
    proposed = _propose(episode, p_x, r_x, nulls, p_y=p_y, p_b=p_b, p_groups=p_groups, target=target,
                        proposer=proposer, family_cfg=family_cfg, seed=seed, device=device, relation=relation,
                        break_column=_within_stage_break, max_candidates=max_candidates)
    proposal_seconds = time.time() - started
    validation_started = time.time()
    ranked = _evaluate_rows(episode, proposed, p_y=p_y, r_y=r_y, p_b=p_b, r_b=r_b, p_groups=p_groups,
                            n_null=n_null, min_gain=min_gain, fdr=fdr, arm="source", r_x=r_x,
                            gene_symbols=gene_symbols, max_candidates=max_candidates, v04=family_cfg is not None,
                            relation=relation)
    names = list(episode.metadata["baseline_columns"])
    info_cols = [j for j, c in enumerate(names) if not c.startswith(("estimate_", "cibersort_", "xcell_"))]
    return {
        "schema": "morpheus_general_discovery_packet/v2",
        "episode": "cptac_patient", "episode_identity": episode.receipt_identity(),
        "protocol_sha256": protocol_sha256, "protocol_file_sha256": protocol_file_sha256,
        "fold": int(fold), "n_folds": int(n_folds), "target": target, "arm": "source", "seed": int(seed),
        "fold_seed": int(fold_seed), "method_truth_accessed": False, "outcome_accessed": False,
        **({} if cohort == "cptac_ccrcc" else {"cohort": cohort}),
        "gene_symbols": gene_symbols, "baseline_width": int(baseline.shape[1]), "baseline_columns": names,
        "fit_patients": list(episode.metadata["fit_patients"]),
        "held_patients": list(episode.metadata["held_patients"]),
        "dropped_rows": list(episode.metadata["dropped_rows"]),
        "dropped_columns": list(episode.metadata["dropped_columns"]),
        "programme_genes": list(episode.metadata["programme_genes"]),
        "provenance": dict(episode.metadata["provenance"]),
        "proposal_budget": int(max_candidates), "null_kinds": sorted(proposed.nulls),
        "relation": relation, "substrate_version": "not_applicable",
        "validation_rule": {"min_replication_incremental_gain": float(min_gain), "q_value_max": float(fdr),
                            "recurrence_note": "recurrence-across-folds gate applied by the funnel aggregator"},
        "candidates": ranked,
        "timing_seconds": {"load": round(load_seconds, 1), "proposal": round(proposal_seconds, 1),
                           "validation": round(time.time() - validation_started, 1)},
        "proposer": proposer, "proposer_report": proposed.report,
        "competitor_alone": None if family_cfg is None else competitor_alone(p_b, p_y, r_b, r_y, p_groups),
        "competitor_informational": competitor_alone(p_b[:, info_cols], p_y, r_b[:, info_cols], r_y, p_groups),
        "design": proposed.design, "family": family, "candidate_views_resolved": proposed.views,
        "elapsed_seconds": time.time() - load_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", choices=["panel", "hest_real", "hest_multi", "bmmc_real", "cptac_patient"],
                        default="panel")
    parser.add_argument("--study", help="hest_multi: study key of sprint_partitions_v01 (e.g. ependymoma_pfa)")
    parser.add_argument("--score-frozen", type=Path,
                        help="all-fold pooling: JSON list of frozen hypotheses ({variable_names, feature_form}) "
                             "to score on this fold instead of proposing (proposer 'fixed')")
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
    parser.add_argument("--proposer", choices=list(PROPOSERS), default="enumeration",
                        help="hest_real/bmmc_real hypothesis source: enumeration (default), "
                             "residual_generator, all, or residual_bank (hest_real; needs "
                             "morpheus_encoder.residual_bank_generator)")
    parser.add_argument("--relation", choices=list(RELATIONS) + ["v03_rff"],
                        help="hest_real development override of discovery.relation (recorded as relation_override)")
    parser.add_argument("--learned-checkpoint",
                        help="proposer learned_search: model checkpoint (else env MORPHEUS_LEARNED_CHECKPOINT; "
                             "absent: the marginal-screen stub)")
    parser.add_argument("--family",
                        help="protocol v0.4 relationship family (discovery.relationship_families); "
                             "sets the candidate views and attaches the measurement bank")
    parser.add_argument("--sidecar", type=Path,
                        help="optional compressed frozen replication-coordinate payload for the sealed evaluator")
    args = parser.parse_args()
    fixed = None if args.score_frozen is None else json.loads(args.score_frozen.read_text())
    proposer = "fixed" if fixed is not None else args.proposer

    if args.episode == "hest_multi":
        if not (args.study and args.target and args.family and args.protocol):
            parser.error("--study, --target, --family and --protocol are required for --episode hest_multi")
        result = run_hest_multi_episode(study=args.study, fold=args.fold, target=args.target, family=args.family,
                                        protocol_path=args.protocol, seed=args.seed, device=args.device,
                                        proposer=proposer, max_candidates=args.max_candidates, n_null=args.n_null,
                                        min_gain=args.min_gain, fdr=args.fdr, n_folds=args.n_folds,
                                        smoke_samples=args.smoke_samples, relation_override=args.relation,
                                        learned_checkpoint=args.learned_checkpoint, fixed_hypotheses=fixed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"episode": "hest_multi", "study": args.study, "fold": result["fold"],
                          "target": result["target"], "loader_target": result["loader_target"],
                          "proposer": proposer, "family": args.family, "n_candidates": len(result["candidates"]),
                          "n_validated": sum(row["validated"] for row in result["candidates"]),
                          "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return

    if args.episode == "hest_real":
        if not args.target:
            parser.error("--target is required for --episode hest_real")
        sample_ids = list(range(1, args.smoke_samples + 1)) if args.smoke_samples else None
        result = run_hest_episode(fold=args.fold, n_folds=args.n_folds, seed=args.seed, target=args.target,
                                  arm=args.arm, device=args.device, protocol_path=args.protocol,
                                  embed_root=args.embed_root, data_dir=args.data_dir, sample_ids=sample_ids,
                                  cohort=args.cohort, max_candidates=args.max_candidates, n_null=args.n_null,
                                  min_gain=args.min_gain, fdr=args.fdr, steps=args.steps,
                                  proposer=proposer, family=args.family, relation_override=args.relation,
                                  learned_checkpoint=args.learned_checkpoint, fixed_hypotheses=fixed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"episode": "hest_real", "fold": result["fold"], "target": result["target"],
                          "arm": result["arm"], "proposer": result["proposer"], "family": result["family"],
                          "n_candidates": len(result["candidates"]),
                          "n_validated": sum(row["validated"] for row in result["candidates"]),
                          "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return

    if args.episode == "cptac_patient":
        if not args.target or not args.family:
            parser.error("--target and --family are required for --episode cptac_patient")
        result = run_cptac_episode(fold=args.fold, target=args.target, family=args.family,
                                   protocol_path=args.protocol, seed=args.seed, device=args.device,
                                   proposer=args.proposer, max_candidates=args.max_candidates, n_null=args.n_null,
                                   n_folds=args.n_folds, min_gain=args.min_gain, fdr=args.fdr,
                                   # --cohort cptac_<type>; the hest default "ccrcc" means cptac_ccrcc
                                   cohort=args.cohort if args.cohort.startswith("cptac_") else "cptac_ccrcc")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"episode": "cptac_patient", "fold": result["fold"], "target": result["target"],
                          "family": result["family"], "n_candidates": len(result["candidates"]),
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
                                  max_cells=args.smoke_cells, proposer=args.proposer, family=args.family)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"episode": "bmmc_real", "fold": result["fold"], "target": result["target"],
                          "arm": result["arm"], "proposer": result["proposer"], "family": result["family"],
                          "n_candidates": len(result["candidates"]),
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



# ----------------------------------------------------------- relation v03 (GPU)
#: relation name of the governing spec's fixed family (0.7) for the discovery packets: kernel ridge on
#: 256 random Fourier features (median-heuristic bandwidth) + linear terms of the set, penalty from
#: {0.01, 0.1, 1, 10} by patient-grouped (leave-one-patient-out) CV on proposal rows, fitted to the
#: response after the nuisance ridge (log depth; per-patient intercepts are the within-patient
#: standardisation); decomposition family = every proper set partition, each block its own RFF map,
#: fitted jointly (``morpheus_encoder.learned_search.relation_v03``).  Batched on the GPU.
V03_RELATION = "v03_rff"
V03_N_BOOT = 200


def _partitions_of(k: int) -> list:
    """Every set partition of range(k) except the one-block partition (the proper decompositions)."""
    def parts(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for p in parts(rest):
            yield [[first]] + p
            for i in range(len(p)):
                yield p[:i] + [[first] + p[i]] + p[i + 1:]
    return [sorted(sorted(b) for b in p) for p in parts(list(range(k))) if len(p) > 1]


def _partition_name(part) -> str:
    return "+".join("".join("ABCD"[i] for i in b) for b in part)


def _boot_draws(groups_q: np.ndarray, seed: int, n: int = V03_N_BOOT) -> np.ndarray:
    """(n, Pq) multiplicities of held-out patients resampled with replacement."""
    pq = len(np.unique(groups_q))
    idx = np.random.default_rng(seed).integers(pq, size=(n, pq))
    W = np.zeros((n, pq))
    for d in range(n):
        np.add.at(W[d], idx[d], 1.0)
    return W


def _evaluate_rows_v03(episode, proposed, *, p_y, r_y, p_b, r_b, p_groups, n_null: int, fdr: float, arm: str,
                       r_x, gene_symbols: list, max_candidates: int, device: str = "cuda") -> list:
    """``_evaluate_rows`` under relation ``v03_rff``: the same receipt fields, the same null definitions
    and draw counts (spatial_block and within_patient_perm re-score fixed predictions against nulled
    held-out responses; interaction_break re-predicts with the first constituent nulled), the v0.4
    family validation rule (held-out full R^2 > 0 and gain-interval lower bound > 0), BH over the
    combined null p; every fit, prediction and null score is a batched GPU solve."""
    import torch
    from morpheus.tier2.substrates import hest
    from morpheus_encoder.learned_search import relation_v03 as rv

    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    z_p, z_r = proposed.z_p, proposed.z_r
    cands = list(proposed.candidates or [])[:max_candidates]
    groups = np.asarray(episode.context["patient_codes"])
    r_groups = groups[episode.partitions.replication]
    t0 = time.time()
    base = rv.fit_baseline(p_y, p_groups, np.asarray(p_b, dtype=float).reshape(len(p_y), -1), yq=r_y,
                           C_q=np.asarray(r_b, dtype=float).reshape(len(r_y), -1))
    pred_base = np.asarray(r_y, dtype=float) - base.r_q
    sst_tot = float(np.sum((r_y - np.mean(r_y)) ** 2))
    _, qinv = np.unique(np.asarray(r_groups).astype(str), return_inverse=True)
    Pq = int(qinv.max()) + 1
    res = {}
    by_k: dict = {}
    for i, c in enumerate(cands):
        by_k.setdefault(len(c.variables), []).append(i)
    for k, ids in by_k.items():
        sets = np.asarray([list(cands[i].variables) for i in ids], dtype=np.int64)
        Vs, Vq = z_p[:, sets], z_r[:, sets]
        full = rv.fit_sets(Vs, base, p_y, p_groups, Vq=Vq, yq=r_y, groups_q=r_groups, device=dev, return_beta=True)
        decs = {}
        for part in (_partitions_of(k) if k >= 2 else []):
            decs[_partition_name(part)] = rv.fit_sets(Vs, base, p_y, p_groups, Vq=Vq, yq=r_y, groups_q=r_groups,
                                                      blocks=part, device=dev)
        M = full["model"]
        for j, i in enumerate(ids):
            res[i] = {"full": {"g_sup": full["g_sup"][j], "alpha": full["alpha"][j], "sse_q": full["sse_q"][j],
                               "gain_qp": full["gain_qp"][:, j], "sst_qp": full["sst_qp"]},
                      "model": {"mu": M["mu"][j:j + 1], "sd": M["sd"][j:j + 1], "beta": M["beta"][j:j + 1],
                                "bw": M["bw"][j:j + 1] if M["blocks"] is None else M["bw"][:, j:j + 1],
                                "blocks": M["blocks"]},
                      "pred": full["pred_q"][:, j],
                      "decs": {n: {"g_sup": d["g_sup"][j], "gain_qp": d["gain_qp"][:, j]} for n, d in decs.items()}}
    # B_known (spec 0.8): nuisance + the programme scores (the target and gene-set-overlapping programmes
    # excluded, as for the candidate design) + the composition variables, fitted on D_P
    # B_known (spec 0.8) through relation_v03.known_covariates / baseline_for, the definition the harness
    # and the training tasks use: nuisance + programme scores without the target and its overlapping
    # programmes (gene targets: without the programmes containing the gene) + the composition variables
    prog = episode.context.get("programmes_all_v03")
    comp = episode.context.get("composition_v03")
    base_known, known_cols, known_dropped = None, [], []
    if prog is not None or comp is not None:
        parts, names = [], []
        if prog is not None:
            parts.append(np.asarray(prog, dtype=float))
            names += list(episode.context["programme_names_v03"])
        if comp is not None:
            parts.append(np.asarray(comp, dtype=float))
            names += [f"composition:{j}" for j in range(np.asarray(comp).shape[1])]
        Kn = _within_standardise(np.column_stack(parts), groups)
        prop, repl = episode.partitions.proposal, episode.partitions.replication
        ktask = rv.Task(X=np.zeros((len(p_y) + len(r_y), 1)), y=np.concatenate([p_y, r_y]), n_support=len(p_y),
                        groups=np.concatenate([np.asarray(p_groups), np.asarray(r_groups)]),
                        depth=np.concatenate([np.asarray(p_b, dtype=float).reshape(len(p_y), -1)[:, 0],
                                              np.asarray(r_b, dtype=float).reshape(len(r_y), -1)[:, 0]]),
                        known=np.concatenate([Kn[prop], Kn[repl]]), composition=None, known_names=names,
                        target_name=str((episode.metadata or {}).get("target") or ""),
                        panel=episode.context.get("panel_v03"))
        _, known_dropped = rv.known_covariates(ktask)
        base_known = rv.baseline_for(ktask, "known")
        known_cols = (["programmes"] if prog is not None else []) + (["composition"] if comp is not None else [])
        for k, ids in by_k.items():
            sets = np.asarray([list(cands[i].variables) for i in ids], dtype=np.int64)
            kn = rv.fit_sets(z_p[:, sets], base_known, p_y, p_groups, Vq=z_r[:, sets], yq=r_y, groups_q=r_groups,
                             device=dev)
            for j, i in enumerate(ids):
                res[i]["known"] = {"gain_qp": kn["gain_qp"][:, j], "g_sup": kn["g_sup"][j]}
    fit_seconds = time.time() - t0
    # ---- nulls: identical definitions, batched scoring
    t1 = time.time()
    shim_nulls = proposed.nulls
    coords_r = np.asarray(episode.context["coords"])[episode.partitions.replication]
    fake_sub = hest._NullSubstrate(coords_r, np.asarray(r_groups))
    Ynull = {kind: np.stack([shim_nulls[kind](d, None, None, None)[1] for d in range(n_null)])
             for kind in ("spatial_block", "within_patient_perm") if kind in shim_nulls}
    tdev = torch.device(dev)
    P_full = torch.as_tensor(np.stack([pred_base + res[i]["pred"] for i in range(len(cands))], 1), device=tdev)
    pb_t = torch.as_tensor(pred_base, device=tdev)

    def null_gains(Y):   # (D, n) nulled responses -> (D, m) gains with the fixed predictions
        Yt = torch.as_tensor(Y, device=tdev, dtype=P_full.dtype)
        sst = ((Yt - Yt.mean(1, keepdim=True)) ** 2).sum(1, keepdim=True)
        sse_b = ((Yt - pb_t) ** 2).sum(1, keepdim=True)
        sse_f = (Yt ** 2).sum(1, keepdim=True) - 2 * Yt @ P_full + (P_full ** 2).sum(0, keepdim=True)
        return ((sse_b - sse_f) / sst).cpu().numpy()

    G_null = {kind: null_gains(Y) for kind, Y in Ynull.items()}
    break_cache: dict = {}
    G_break = {}
    if "interaction_break" in shim_nulls:
        for i, c in enumerate(cands):
            if len(c.variables) < 2 or c.feature_form != "product":
                continue
            col = int(c.variables[0])
            if col not in break_cache:
                break_cache[col] = np.stack([hest.null_hest(z_r[:, col], fake_sub, seed=d, nx=6, ny=6)
                                             for d in range(n_null)])          # (D, n)
            V = np.repeat(z_r[:, list(c.variables)][:, None, :], n_null, axis=1)  # (n, D, k)
            V[:, :, 0] = break_cache[col].T
            mdl = res[i]["model"]
            rep_model = {kk: (np.repeat(v, n_null, axis=0) if kk in ("mu", "sd", "beta") else
                              (np.repeat(v, n_null, axis=0) if mdl["blocks"] is None else np.repeat(v, n_null, axis=1))
                              if kk == "bw" else v) for kk, v in mdl.items()}
            pred = pred_base[:, None] + rv.predict(rep_model, V, device=dev)            # (n, D)
            sse_f = ((r_y[:, None] - pred) ** 2).sum(0)
            G_break[i] = (np.sum((r_y - pred_base) ** 2) - sse_f) / sst_tot
    null_seconds = time.time() - t1
    # ---- rows
    rows = []
    for i, c in enumerate(cands):
        f = res[i]["full"]
        gain_qp, sst_qp = np.asarray(f["gain_qp"]), np.asarray(f["sst_qp"])
        g_obs = float(gain_qp.sum() / sst_qp.sum())
        W = _boot_draws(r_groups, int(c.candidate_hash[:8], 16))
        boot = (W @ gain_qp) / (W @ sst_qp)
        r2_full = 1.0 - float(f["sse_q"]) / sst_tot
        r2_base = 1.0 - float(np.sum(base.r_q ** 2)) / sst_tot
        null_summary = {}
        for kind, G in G_null.items():
            null_summary[kind] = {"p": float((1 + np.sum(G[:, i] >= g_obs)) / (n_null + 1)), "n": int(n_null)}
        if i in G_break:
            null_summary["interaction_break"] = {"p": float((1 + np.sum(G_break[i] >= g_obs)) / (n_null + 1)),
                                                 "n": int(n_null)}
        decomposition = {"gain_general": g_obs, "gain_general_ci": [float(np.percentile(boot, 2.5)),
                                                                    float(np.percentile(boot, 97.5))]}
        if "known" in res[i]:
            kq = np.asarray(res[i]["known"]["gain_qp"])
            boot_k = (W @ kq) / (W @ sst_qp)
            decomposition.update({"gain_known": float(kq.sum() / sst_qp.sum()),
                                  "gain_known_ci": [float(np.percentile(boot_k, 2.5)), float(np.percentile(boot_k, 97.5))],
                                  "gain_known_proposal": float(res[i]["known"]["g_sup"]),
                                  "known_covariates": ["depth_log1p"] + known_cols,
                                  "known_dropped": known_dropped})
        if res[i]["decs"]:
            names = list(res[i]["decs"])
            dq = np.stack([res[i]["decs"][n]["gain_qp"] for n in names], 1)          # (Pq, n_parts)
            g_dec = dq.sum(0) / sst_qp.sum()
            best = int(np.argmax(g_dec))
            boot_ho = boot - ((W @ dq) / (W @ sst_qp)[:, None]).max(1)            # max inside every draw
            sup_dec = np.array([res[i]["decs"][n]["g_sup"] for n in names])
            decomposition.update({
                "G_HO": float(g_obs - g_dec[best]), "G_HO_ci": [float(np.percentile(boot_ho, 2.5)),
                                                                float(np.percentile(boot_ho, 97.5))],
                "G_HO_proposal": float(f["g_sup"] - sup_dec.max()), "best_decomposition": names[best],
                "decompositions": names, "gain_decompositions": {n: float(v) for n, v in zip(names, g_dec)},
                "gain_joint_vs_additive": float(g_obs - g_dec[names.index("+".join("ABCD"[j] for j in range(len(c.variables))))])
                if len(c.variables) >= 2 else None})
        combined_p = max((v["p"] for v in null_summary.values()), default=None)
        rows.append({
            "candidate_hash": c.candidate_hash, "variables": list(c.variables),
            "gene_symbols": [gene_symbols[v] for v in c.variables if v < len(gene_symbols)],
            "feature_form": c.feature_form, "proposal_family": c.proposal_family, "proposal_score": c.proposal_score,
            "proposal_gain": float(f["g_sup"]), "replication_gain": g_obs, "baseline_width": int(np.asarray(p_b).reshape(len(p_y), -1).shape[1]),
            "replication_gain_ci": decomposition["gain_general_ci"], "competitor_alpha": float(base.alpha),
            # the spec's three baselines per row (0.8): general, novel (beyond known biology), higher-order
            "gain_B_nuis": g_obs, "gain_B_nuis_ci": decomposition["gain_general_ci"],
            "gain_B_known": decomposition.get("gain_known"), "gain_B_known_ci": decomposition.get("gain_known_ci"),
            "G_HO_B_decomp": decomposition.get("G_HO"), "G_HO_B_decomp_ci": decomposition.get("G_HO_ci"),
            "best_decomposition": decomposition.get("best_decomposition"),
            "full_alpha": float(f["alpha"]), "competitor_r2_replication": r2_base, "full_r2_replication": r2_full,
            "replication_r2_full": r2_full,
            "r2": {"proposal_simple": None, "proposal_full": None, "replication_simple": r2_base,
                   "replication_full": r2_full},
            "nulls": null_summary, "_combined_null_p": combined_p, "relation": V03_RELATION,
            "decomposition": decomposition,
            "higher_order": bool(decomposition.get("G_HO_ci", [None])[0] is not None
                                 and decomposition["G_HO_ci"][0] > 0 and r2_full > 0),
            **_proposer_fields(c, proposed.column_names, proposed.also, gene_symbols),
            "proposal_cv_gain": proposed.ranking.get(c.candidate_hash), "observed_coordinate_links": None,
        })
    q_values = adjust_family_pvalues({row["candidate_hash"]: (1.0 if row["_combined_null_p"] is None
                                                              else row["_combined_null_p"]) for row in rows}) if rows else {}
    ranked = sorted(rows, key=lambda row: (-row["proposal_gain"], row["candidate_hash"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["q_value"] = q_values.get(row["candidate_hash"])
        row["validated"] = bool(row["replication_r2_full"] > 0.0 and row["replication_gain_ci"][0] > 0.0)
        row.pop("_combined_null_p", None)
        row["validation_seconds"] = {"fits": round(fit_seconds / max(1, len(rows)), 3),
                                     "nulls": round(null_seconds / max(1, len(rows)), 3)}
    return ranked

if __name__ == "__main__":
    main()
