"""Strict, query-side audit and rescoring for historical rediscovery.

This module leaves historical v01/v02 untouched so their receipts remain
reproducible. It addresses query-calibrated proxy ceilings, negligible target
associations, and silently unscorable composed spatial transforms.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dbench import historical as H


V03_NAME = "v03_query_validated"
DEFAULTS = {"top_k": 20, "min_abs_answer_r": 0.50, "min_abs_target_r": 0.20,
            "min_query_patients": 3, "n_boot": 2000}


def _rows_for(episode, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Support/query positions within a task's row-restricted arrays."""
    rows = np.asarray(rows, dtype=int)
    support = np.flatnonzero(np.isin(rows, np.asarray(episode.support_rows, dtype=int)))
    query = np.flatnonzero(np.isin(rows, np.asarray(episode.query_rows, dtype=int)))
    if not len(support) or not len(query):
        raise ValueError("historical task has an empty support or query side")
    return support, query


def _patient_r(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One within-patient Pearson r per patient, dropping degenerate patients."""
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    values, patients = [], []
    for p in np.unique(groups):
        ix = np.flatnonzero(groups == p)
        if len(ix) < 3:
            continue
        x, y = a[ix] - a[ix].mean(), b[ix] - b[ix].mean()
        den = math.sqrt(float(x @ x) * float(y @ y))
        if den > 1e-12:
            values.append(float((x @ y) / den))
            patients.append(str(p))
    return np.asarray(values, float), np.asarray(patients, str)


def _summary(values: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, float)
    if not len(values):
        return {"n_patients": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(int(n_boot), len(values)))].mean(1)
    return {"n_patients": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "positive_patients": int((values > 0).sum()), "negative_patients": int((values < 0).sum())}


def _direct_transform(name: str) -> tuple[str, str, bool]:
    """Return transform/base/is_direct; nested transforms are never silently credited."""
    transform, base = H.parse_variable(str(name))
    return transform, base, transform in H.TRANSFORMS


def direct_base_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Own-level catalogue for a proposer that constructs spatial transforms exactly once.

    Historical v02 passed a catalogue already containing nbr6/nbr18/contrast
    columns into a proposer that itself emits transformed columns.  The result
    was a composed transform such as ``nbr18:nbr18:gene`` that no direct
    historical relation can match.  This helper keeps every own-level column,
    rebases views, and preserves the support-standardised values unchanged.
    """
    names = list(design["names"])
    keep = [i for i, name in enumerate(names) if H.parse_variable(name)[0] == "own"]
    if not keep:
        raise ValueError("historical design contains no own-level variables")
    rebase = {old: new for new, old in enumerate(keep)}
    views = {str(k): [rebase[int(i)] for i in v if int(i) in rebase]
             for k, v in dict(design.get("views") or {}).items()}
    views = {k: v for k, v in views.items() if v}
    return {"X": np.asarray(design["X"], float)[:, keep], "names": [names[i] for i in keep],
            "views": views, "roles": dict(design.get("roles") or {}), "kept_indices": keep}


def learned_direct_grammar_ranked(task: Mapping[str, Any], episode, model, *, gene_matrix=None,
                                  budget: int = 64, seed: int = 0) -> tuple[list[dict], dict]:
    """Support-only learned proposal replay with a non-composing input grammar."""
    from morpheus_encoder.learned_search.propose import propose, transform_columns

    scores = H.proxy_scores(episode, task, gene_matrix=gene_matrix)
    rows = H.task_rows(episode, task, scores)
    y = H.target_values(episode, task, scores, rows)
    design = H.build_design(episode, task, rows, y)
    direct = direct_base_design(design)
    support, _ = _rows_for(episode, rows)
    groups = H._groups(episode)[rows]
    nb6 = H._remap(episode.neighbours[6], rows, episode.n)
    coords = None if episode.coords is None else np.asarray(episode.coords, dtype=float)[rows]
    result = propose(model, direct["X"][support], y[support], groups[support],
                     H._remap(nb6, support, len(rows)), None if coords is None else coords[support],
                     direct["names"], direct["views"], budget=int(budget), seed=int(seed))
    all_names = list(direct["names"]) + list(result.extra_names)
    extra = transform_columns(result.extra_specs, direct["X"], nb6, model=model,
                              coords=coords, groups=groups) if result.extra_specs else np.zeros((len(rows), 0))
    ranked: list[dict] = []
    for candidate in result.candidates:
        idx = list(candidate.variables)
        values = np.column_stack([direct["X"][:, i] if i < direct["X"].shape[1]
                                  else extra[:, i - direct["X"].shape[1]] for i in idx])
        ranked.append({"names": [all_names[i] for i in idx], "values": values,
                       "score": float(candidate.proposal_score)})
    report = {"input_grammar": "own_columns_plus_one_proposer_transform", "n_input_columns": len(direct["names"]),
              "n_sets_returned": len(ranked), "n_relationships_tested": int(result.n_relationships_tested),
              "proposer_report": result.report}
    return ranked, report


def _source_exact(task: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Flag proxy definitions invented or widened by the protocol rather than supplied by a paper."""
    flags: list[str] = []
    for name, spec in task.get("proxy_sets", {}).items():
        source = str(spec.get("source", "")).lower()
        if any(token in source for token in ("protocol marker", "canonical", "added by this protocol", "not verified")):
            flags.append(f"{name}:protocol_derived_proxy")
    rows = str((task.get("given_to_morpheus") or {}).get("rows", {}).get("note", "")).lower()
    if "stands in" in rows or "stand in" in rows:
        flags.append("rows:protocol_derived_region_rule")
    return not flags, flags


def _candidate_values(entry: Mapping[str, Any], j: int, name: str, design: Mapping[str, Any], episode,
                      rows: np.ndarray) -> np.ndarray:
    values = entry.get("values")
    if values is not None:
        return np.asarray(values, dtype=float)[:, j]
    return H._resolve_values(name, design, episode, rows)


def _candidate_query_stats(v: np.ndarray, y: np.ndarray, proxy: np.ndarray, groups: np.ndarray,
                           direction: int, n_boot: int, seed: int) -> dict[str, Any]:
    rt, patients = _patient_r(v, y, groups)
    ra, patients_a = _patient_r(v, proxy, groups)
    common = sorted(set(patients).intersection(patients_a))
    if not common:
        return {"n_patients": 0, "valid": False}
    rt_map = dict(zip(patients.tolist(), rt.tolist()))
    ra_map = dict(zip(patients_a.tolist(), ra.tolist()))
    target = np.asarray([rt_map[p] for p in common], float)
    answer = np.asarray([ra_map[p] for p in common], float)
    return {"n_patients": len(common), "target_abs": _summary(np.abs(target), n_boot, seed + 1),
            "answer_abs": _summary(np.abs(answer), n_boot, seed + 2),
            "signed_product": _summary(int(direction) * target * answer, n_boot, seed + 3),
            "patient_ids": common}


def audit_task_feasibility(task: Mapping[str, Any], episode, *, gene_matrix=None, n_boot: int = 2000) -> dict[str, Any]:
    """Check whether the published relation is observable under the frozen patient split."""
    scores = H.proxy_scores(episode, task, gene_matrix=gene_matrix)
    rows = H.task_rows(episode, task, scores)
    y = H.target_values(episode, task, scores, rows)
    groups = H._groups(episode)[rows]
    support, query = _rows_for(episode, rows)
    relation, scoring = task["relation"], task.get("scoring") or {}
    direct: list[dict[str, Any]] = []
    for answer in scoring.get("answer_proxies") or relation["answer_variables"]:
        for transform in scoring.get("answer_transforms") or H._as_list(relation["transform"]):
            p = np.asarray(scores[answer if transform == "own" else f"{answer}@{transform}"], float)[rows]
            r, patients = _patient_r(y[query], p[query], groups[query])
            seed = int(hashlib.sha256(f"{task['task_id']}|{answer}|{transform}".encode()).hexdigest()[:8], 16)
            summary = _summary(int(relation["direction"]) * r, n_boot, seed)
            direct.append({"answer_proxy": answer, "transform": transform, "query_relation": summary,
                           "patient_ids": patients.tolist(),
                           "positive_control_pass": bool(summary["n_patients"] >= DEFAULTS["min_query_patients"] and
                                                         summary["ci95"][0] is not None and summary["ci95"][0] > 0.0)})
    source_exact, source_flags = _source_exact(task)
    return {"historical_repair_version": V03_NAME, "task_id": task["task_id"], "study_key": task["study_key"],
            "source_exact": source_exact, "source_flags": source_flags, "n_rows": int(len(rows)),
            "n_support_patients": int(len(np.unique(groups[support]))),
            "n_query_patients": int(len(np.unique(groups[query]))), "proxy_genes_present": scores.get("_present", {}),
            "direct_query_controls": direct,
            "primary_eligible": bool(source_exact and any(x["positive_control_pass"] for x in direct))}


def score_ranked_list_v03(ranked_sets: Sequence[Any], episode, task: Mapping[str, Any], *,
                          top_k: int | None = None, scores: Mapping[str, Any] | None = None,
                          rows: np.ndarray | None = None, y: np.ndarray | None = None,
                          design: Mapping[str, Any] | None = None, gene_matrix=None,
                          n_boot: int | None = None) -> dict[str, Any]:
    """Query-only historical score with fixed thresholds and a non-negligible target relation."""
    cfg = {**DEFAULTS, **dict(task.get("scoring") or {})}
    k, n_boot = int(top_k or cfg["top_k"]), int(n_boot or cfg["n_boot"])
    if scores is None:
        scores = H.proxy_scores(episode, task, gene_matrix=gene_matrix)
    if rows is None:
        rows = H.task_rows(episode, task, scores)
    if y is None:
        y = H.target_values(episode, task, scores, rows)
    if design is None:
        design = H.build_design(episode, task, rows, y)
    _, query = _rows_for(episode, rows)
    groups = H._groups(episode)[rows]
    relation = task["relation"]
    answers = list(cfg.get("answer_proxies") or relation["answer_variables"])
    allowed = set(cfg.get("answer_transforms") or H._as_list(relation["transform"]))
    details, first = [], None
    for rank, raw in enumerate(list(ranked_sets)[:k], 1):
        entry = raw if isinstance(raw, Mapping) else {"names": list(raw)}
        names, candidates = list(entry["names"]), []
        for j, name in enumerate(names):
            transform, base, direct = _direct_transform(name)
            if not direct:
                candidates.append({"variable": name, "status": "composed_transform_not_in_historical_grammar",
                                   "transform": transform, "base": base, "valid": False})
                continue
            if transform not in allowed:
                candidates.append({"variable": name, "status": "transform_not_stated_by_task",
                                   "transform": transform, "base": base, "valid": False})
                continue
            v = _candidate_values(entry, j, name, design, episode, rows)[query]
            for answer in answers:
                p = np.asarray(scores[answer if transform == "own" else f"{answer}@{transform}"], float)[rows][query]
                seed = int(hashlib.sha256(f"{task['task_id']}|{rank}|{name}|{answer}".encode()).hexdigest()[:8], 16)
                stat = _candidate_query_stats(v, np.asarray(y, float)[query], p, groups[query],
                                              int(relation["direction"]), n_boot, seed)
                qa, qt, prod = stat.get("answer_abs", {}), stat.get("target_abs", {}), stat.get("signed_product", {})
                valid = bool(stat.get("n_patients", 0) >= int(cfg["min_query_patients"]) and
                             qa.get("ci95", [None])[0] is not None and qa["ci95"][0] >= float(cfg["min_abs_answer_r"]) and
                             qt.get("ci95", [None])[0] is not None and qt["ci95"][0] >= float(cfg["min_abs_target_r"]) and
                             prod.get("ci95", [None])[0] is not None and prod["ci95"][0] > 0.0)
                candidates.append({"variable": name, "base": base, "transform": transform, "answer_proxy": answer,
                                   "status": "query_validated" if valid else "query_not_validated", "valid": valid, **stat})
        best = next((x for x in candidates if x.get("valid")), None)
        details.append({"rank": rank, "names": names, "candidates": candidates, "hit": best})
        if first is None and best is not None:
            first = (rank, best)
    return {"historical_repair_version": V03_NAME, "rank_of_first_query_validated_hit": first[0] if first else math.inf,
            "hit": first[1] if first else None, "n_evaluated": len(details), "top_k": k,
            "thresholds": {x: cfg[x] for x in ("min_abs_answer_r", "min_abs_target_r", "min_query_patients")},
            "details": details}


def rescore_saved_v03(result: Mapping[str, Any], task: Mapping[str, Any], episode, *, gene_matrix=None,
                      scores: Mapping[str, Any] | None = None, rows: np.ndarray | None = None,
                      y: np.ndarray | None = None, design: Mapping[str, Any] | None = None,
                      n_boot: int = 2000) -> dict[str, Any]:
    """Rescore saved lists without rerunning a proposer or changing its receipt."""
    ranked = [{"names": x["names"]} for x in result.get("top", [])]
    score = score_ranked_list_v03(ranked, episode, task, gene_matrix=gene_matrix, scores=scores, rows=rows,
                                  y=y, design=design, n_boot=n_boot)
    return {"task_id": task["task_id"], "study_key": task["study_key"], "method": result.get("method"),
            "source_result": {"path_hash": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
                              "protocol_sha256": result.get("protocol_sha256")}, **score}
