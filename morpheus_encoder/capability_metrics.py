"""Packet-level metrics for paper-ready MORPHEUS capability comparisons.

The independent experimental unit in a capability panel is a *packet*, not a
spot, cell, patch, time point, candidate expression, or bootstrap row.  This
module deliberately accepts one receipt per ``(arm, family, packet_id)`` and
rejects duplicate receipts.  It provides descriptive recovery measures in
addition to strict validation so a near miss is not collapsed into ``0/1``.

It is intentionally independent of the world adapter and discovery core.
Callers supply frozen per-packet receipts and, for object fidelity, mappings
that describe the truth and recovered object.  Nothing here uses truth during
training or proposal selection.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_COMPONENTS: dict[str, dict[str, Any]] = {
    "variables": {"truth_key": "variables", "candidate_key": "variables", "weight": 1.0},
    "subspace": {"truth_key": "subspace", "candidate_key": "subspace", "weight": 1.0},
    "context": {"truth_key": "context", "candidate_key": "context", "weight": 1.0},
    "relation": {"truth_key": "relation", "candidate_key": "relation", "weight": 1.0},
    "environment": {"truth_key": "environment", "candidate_key": "environment", "weight": 1.0},
}


def _as_set(value: Any) -> set[Any] | None:
    if isinstance(value, (str, bytes, Mapping)) or value is None:
        return None
    if isinstance(value, (set, frozenset, list, tuple)):
        return set(value)
    return None


def _value_similarity(truth: Any, candidate: Any) -> float:
    """A conservative, interpretable similarity for an object component.

    Sets (including lists/tuples) use Jaccard overlap; mappings use the mean
    similarity of their union of keys; scalar values use exact agreement; and
    numeric vectors use signed cosine similarity clipped to ``[0, 1]``.
    Missing a declared component earns zero rather than being silently
    excluded from the denominator.
    """
    if truth is None or candidate is None:
        return 0.0
    if isinstance(truth, Mapping) and isinstance(candidate, Mapping):
        keys = set(truth) | set(candidate)
        if not keys:
            return 1.0
        return float(np.mean([_value_similarity(truth.get(key), candidate.get(key))
                              for key in keys]))
    truth_set, candidate_set = _as_set(truth), _as_set(candidate)
    if truth_set is not None and candidate_set is not None:
        union = truth_set | candidate_set
        return 1.0 if not union else len(truth_set & candidate_set) / len(union)
    try:
        a, b = np.asarray(truth, dtype=float), np.asarray(candidate, dtype=float)
        if a.ndim == 1 and b.ndim == 1 and a.size > 1 and a.shape == b.shape:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 0:
                return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))
    except (TypeError, ValueError):
        pass
    return float(truth == candidate)


def structured_object_fidelity(
    truth: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    components: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score a recovered object without hiding partial recovery.

    ``components`` maps a report name to ``truth_key``, ``candidate_key`` and
    optional positive ``weight``.  By default the score uses whichever of
    variable/subspace/context/relation/environment keys the truth declares.
    The returned weighted score is in ``[0, 1]`` and is accompanied by every
    component score, allowing reports to say *what* was recovered.
    """
    if not isinstance(truth, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("truth and candidate must be mappings")
    spec = components or DEFAULT_COMPONENTS
    selected = []
    for name, config in spec.items():
        if not isinstance(config, Mapping):
            raise TypeError("each component specification must be a mapping")
        truth_key = str(config.get("truth_key", name))
        candidate_key = str(config.get("candidate_key", truth_key))
        # An undeclared truth component is not an identity condition.
        if truth_key not in truth:
            continue
        weight = float(config.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("component weights must be finite and positive")
        selected.append((name, truth_key, candidate_key, weight))
    if not selected:
        raise ValueError("truth declares none of the requested fidelity components")
    values = {
        name: _value_similarity(truth[truth_key], candidate.get(candidate_key))
        for name, truth_key, candidate_key, _ in selected
    }
    weights = {name: weight for name, _, _, weight in selected}
    total = sum(weights.values())
    return {
        "object_fidelity": float(sum(values[name] * weights[name] for name in values) / total),
        "components": values,
        "weights": weights,
        "declared_components": list(values),
    }


def _required(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise ValueError("packet receipt is missing required field %r" % key)
    return record[key]


def validate_packet_receipts(
    receipts: Iterable[Mapping[str, Any]], *, require_complete_pairs: bool = False,
    arms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate the one-receipt-per-arm-per-packet contract.

    A receipt may carry arbitrarily many candidate rows under ``candidates``,
    but the outer iterable must contain one summary receipt per arm and
    independent packet.  This prevents pseudo-replication from spatial rows
    or repeated candidate expressions.
    """
    checked: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise TypeError("each packet receipt must be a mapping")
        row = dict(raw)
        arm = str(_required(row, "arm"))
        family = str(_required(row, "family"))
        packet_id = str(_required(row, "packet_id"))
        key = (arm, family, packet_id)
        if key in seen:
            raise ValueError("duplicate receipt for arm/family/packet %r" % (key,))
        seen.add(key)
        checked.append(row)
    if not checked:
        raise ValueError("at least one independent packet receipt is required")
    if arms is not None:
        unknown = {str(row["arm"]) for row in checked} - set(arms)
        if unknown:
            raise ValueError("unexpected arms: %s" % sorted(unknown))
    if require_complete_pairs:
        if not arms or len(arms) < 2:
            raise ValueError("complete pairing requires at least two declared arms")
        expected = set(map(str, arms))
        by_packet: dict[tuple[str, str], set[str]] = {}
        for row in checked:
            by_packet.setdefault((str(row["family"]), str(row["packet_id"])), set()).add(str(row["arm"]))
        incomplete = {key: sorted(expected - available) for key, available in by_packet.items()
                      if available != expected}
        if incomplete:
            raise ValueError("incomplete paired packet receipts: %r" % incomplete)
    return checked


def _candidate_at_budget(receipt: Mapping[str, Any], budget: int) -> list[Mapping[str, Any]]:
    if budget < 1:
        raise ValueError("budget must be at least one")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("each packet receipt must include a candidates sequence")
    out = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate entries must be mappings")
        rank = int(_required(candidate, "rank"))
        if rank < 1:
            raise ValueError("candidate ranks must be positive")
        if rank <= budget:
            out.append(candidate)
    return out


def validated_recovery_at_budget(
    receipts: Iterable[Mapping[str, Any]], *, budget: int,
    min_fidelity: float = 1.0,
    validation_key: str = "validated",
    fidelity_key: str = "object_fidelity",
) -> dict[str, Any]:
    """Compute packet-level validated recovery under a fixed candidate budget."""
    rows = validate_packet_receipts(receipts)
    if not 0.0 <= min_fidelity <= 1.0:
        raise ValueError("min_fidelity must be in [0, 1]")
    per_packet = []
    for row in rows:
        eligible = _candidate_at_budget(row, budget)
        recovered = any(bool(c.get(validation_key, False))
                        and float(c.get(fidelity_key, 0.0)) >= min_fidelity for c in eligible)
        best_fidelity = max((float(c.get(fidelity_key, 0.0)) for c in eligible), default=0.0)
        per_packet.append({"arm": str(row["arm"]), "family": str(row["family"]),
                           "packet_id": str(row["packet_id"]), "recovered": int(recovered),
                           "best_fidelity": best_fidelity, "n_candidates_within_budget": len(eligible)})
    return {"budget": int(budget), "min_fidelity": float(min_fidelity),
            "unit_of_inference": "independent packet", "n_packets": len(per_packet),
            "recovery_rate": float(np.mean([x["recovered"] for x in per_packet])),
            "mean_best_fidelity": float(np.mean([x["best_fidelity"] for x in per_packet])),
            "per_packet": per_packet}


def discovery_vs_budget(
    receipts: Iterable[Mapping[str, Any]], *, max_budget: int,
    min_fidelity: float = 1.0,
) -> dict[str, Any]:
    """Return a fixed-budget recovery curve and its normalized discrete AUC."""
    if max_budget < 1:
        raise ValueError("max_budget must be at least one")
    rows = validate_packet_receipts(receipts)
    curve = [validated_recovery_at_budget(rows, budget=b, min_fidelity=min_fidelity)["recovery_rate"]
             for b in range(1, max_budget + 1)]
    # Mean over the predeclared integer budgets: 1.0 is perfect recovery at
    # every permitted search budget, 0.0 is no recovery.
    return {"max_budget": int(max_budget), "min_fidelity": float(min_fidelity),
            "budgets": list(range(1, max_budget + 1)), "recovery_curve": curve,
            "discovery_auc": float(np.mean(curve)), "unit_of_inference": "independent packet"}


def budget_to_80(curve: Mapping[str, Any], *, target: float = 0.80) -> dict[str, Any]:
    """Return the first budget reaching an absolute recovery target, if any."""
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    budgets, values = list(curve.get("budgets", [])), list(curve.get("recovery_curve", []))
    if len(budgets) != len(values) or not budgets:
        return {"available": False, "reason": "empty_or_malformed_curve", "budget": None, "target": target}
    for budget, value in zip(budgets, values):
        if float(value) >= target:
            return {"available": True, "budget": int(budget), "target": target,
                    "recovery_at_budget": float(value)}
    return {"available": False, "reason": "target_not_reached", "budget": None,
            "target": target, "maximum_recovery": float(max(values))}


def _paired_values(
    receipts: Iterable[Mapping[str, Any]], *, candidate_arm: str, reference_arm: str,
    metric: str | Callable[[Mapping[str, Any]], float],
) -> list[dict[str, Any]]:
    rows = validate_packet_receipts(receipts, require_complete_pairs=True,
                                    arms=[reference_arm, candidate_arm])
    getter = metric if callable(metric) else lambda row: float(_required(row, str(metric)))
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["packet_id"])), {})[str(row["arm"])] = row
    pairs = []
    for (family, packet_id), by_arm in sorted(grouped.items()):
        candidate, reference = float(getter(by_arm[candidate_arm])), float(getter(by_arm[reference_arm]))
        if not (np.isfinite(candidate) and np.isfinite(reference)):
            raise ValueError("paired metric must be finite for every packet")
        pairs.append({"family": family, "packet_id": packet_id, "candidate": candidate,
                      "reference": reference, "difference": candidate - reference})
    return pairs


def paired_bootstrap_ci(
    receipts: Iterable[Mapping[str, Any]], *, candidate_arm: str, reference_arm: str,
    metric: str | Callable[[Mapping[str, Any]], float], n_bootstrap: int = 4000,
    seed: int = 0, hierarchical: bool = True,
) -> dict[str, Any]:
    """Paired CI, optionally resampling families then packets within family."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    pairs = _paired_values(receipts, candidate_arm=candidate_arm, reference_arm=reference_arm, metric=metric)
    diffs = np.asarray([x["difference"] for x in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    if hierarchical and len({x["family"] for x in pairs}) > 1:
        by_family: dict[str, np.ndarray] = {}
        for family in sorted({x["family"] for x in pairs}):
            by_family[family] = np.asarray([x["difference"] for x in pairs if x["family"] == family])
        families = list(by_family)
        draws = np.empty(n_bootstrap, dtype=float)
        for draw in range(n_bootstrap):
            selected = rng.integers(0, len(families), size=len(families))
            sampled = [by_family[families[index]][rng.integers(0, len(by_family[families[index]]),
                                                               size=len(by_family[families[index]]) )]
                       for index in selected]
            draws[draw] = float(np.mean(np.concatenate(sampled)))
        method = "paired hierarchical family-then-packet bootstrap"
    else:
        indices = rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))
        draws = diffs[indices].mean(axis=1)
        method = "paired packet bootstrap"
    return {"candidate_arm": candidate_arm, "reference_arm": reference_arm,
            "n_paired_packets": len(pairs), "mean_difference": float(diffs.mean()),
            "bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "bootstrap_method": method, "per_packet": pairs}


def paired_sign_permutation_test(
    receipts: Iterable[Mapping[str, Any]], *, candidate_arm: str, reference_arm: str,
    metric: str | Callable[[Mapping[str, Any]], float], n_permutations: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided paired sign-flip test on independent packet differences."""
    pairs = _paired_values(receipts, candidate_arm=candidate_arm, reference_arm=reference_arm, metric=metric)
    diffs = np.asarray([x["difference"] for x in pairs], dtype=float)
    observed = float(diffs.mean())
    n = len(diffs)
    if n <= 16:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=n)), dtype=float)
        null = (signs * diffs).mean(axis=1)
        exact = True
    else:
        if n_permutations < 1:
            raise ValueError("n_permutations must be positive")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
        null = (signs * diffs).mean(axis=1)
        exact = False
    p_value = float((np.count_nonzero(np.abs(null) >= abs(observed)) + (0 if exact else 1)) /
                    (len(null) + (0 if exact else 1)))
    return {"n_paired_packets": n, "mean_difference": observed, "p_value_two_sided": p_value,
            "exact": exact, "n_null_draws": int(len(null)), "unit_of_inference": "independent packet"}


def adjust_family_pvalues(pvalues: Mapping[str, float], *, method: str = "bh") -> dict[str, float]:
    """Benjamini--Hochberg adjustment across a predeclared family list."""
    if method.lower() != "bh":
        raise ValueError("only Benjamini-Hochberg ('bh') adjustment is supported")
    keys = list(pvalues)
    raw = np.asarray([float(pvalues[key]) for key in keys], dtype=float)
    if raw.size == 0:
        return {}
    if not np.all(np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
        raise ValueError("family p-values must be finite values in [0, 1]")
    order = np.argsort(raw)
    adjusted_sorted = np.empty_like(raw)
    running = 1.0
    for position in range(len(raw) - 1, -1, -1):
        index = order[position]
        running = min(running, raw[index] * len(raw) / (position + 1))
        adjusted_sorted[index] = running
    return {key: float(adjusted_sorted[index]) for index, key in enumerate(keys)}


# ---------------------------------------------------------------------------
# Sealed stagewise receipts
# ---------------------------------------------------------------------------
def _sealed_stage_components(receipt: Mapping[str, Any], *, stage_key: str) -> dict[str, Any]:
    """Read already-computed stage outputs without inspecting a truth object.

    The runner/scorer is responsible for deciding what the hidden target is and
    for writing its frozen stage receipt.  This function deliberately has no
    ``truth`` argument and does not read a truth-like field from ``receipt``.
    It merely aggregates a small, auditable summary emitted before reporting.
    """
    stages = receipt.get(stage_key)
    if not isinstance(stages, Mapping):
        raise ValueError("each packet receipt must contain a %r mapping" % stage_key)
    required = ("accessibility", "candidate_exposed", "candidate_rank",
                "relation_validated", "null_survived", "decoy_promoted")
    missing = [key for key in required if key not in stages]
    if missing:
        raise ValueError("sealed stage receipt is missing fields: %s" % missing)
    accessibility = float(stages["accessibility"])
    if not np.isfinite(accessibility) or not 0.0 <= accessibility <= 1.0:
        raise ValueError("sealed accessibility must be a finite value in [0, 1]")
    exposed = stages["candidate_exposed"]
    if not isinstance(exposed, (bool, np.bool_)):
        raise TypeError("candidate_exposed must be boolean")
    rank = stages["candidate_rank"]
    if rank is not None:
        if isinstance(rank, bool) or int(rank) != rank or int(rank) < 1:
            raise ValueError("candidate_rank must be a positive integer or None")
        rank = int(rank)
    if bool(exposed) != (rank is not None):
        raise ValueError("candidate_exposed and candidate_rank must agree")
    flags = {}
    for key in ("relation_validated", "null_survived", "decoy_promoted"):
        if not isinstance(stages[key], (bool, np.bool_)):
            raise TypeError("%s must be boolean" % key)
        flags[key] = int(bool(stages[key]))
    return {
        "accessibility": accessibility,
        "candidate_exposed": int(bool(exposed)),
        "candidate_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        **flags,
        # This is a descriptive progression through frozen stages, not a new
        # certification claim.  A promoted decoy invalidates a clean passage.
        "clean_stagewise_pass": int(bool(exposed) and bool(stages["relation_validated"])
                                    and bool(stages["null_survived"])
                                    and not bool(stages["decoy_promoted"])),
    }


def sealed_stagewise_summary(
    receipts: Iterable[Mapping[str, Any]], *, stage_key: str = "sealed_stages",
) -> dict[str, Any]:
    """Summarise frozen accessibility-to-null stages at independent-packet level.

    It accepts exactly one receipt per arm/family/packet, making it impossible
    to inflate an arm through cells, spots, candidates, or repeated rows.  The
    output retains one row per packet and aggregates only after that boundary.
    """
    rows = validate_packet_receipts(receipts)
    per_packet = []
    for row in rows:
        components = _sealed_stage_components(row, stage_key=stage_key)
        per_packet.append({"arm": str(row["arm"]), "family": str(row["family"]),
                           "packet_id": str(row["packet_id"]), **components})
    by_arm_family: dict[str, dict[str, Any]] = {}
    for arm, family in sorted({(row["arm"], row["family"]) for row in per_packet}):
        values = [row for row in per_packet if row["arm"] == arm and row["family"] == family]
        exposed = [row for row in values if row["candidate_exposed"]]
        relation_validated = [row for row in exposed if row["relation_validated"]]
        label = "%s::%s" % (arm, family)
        by_arm_family[label] = {
            "arm": arm,
            "family": family,
            "n_packets": len(values),
            "mean_accessibility": float(np.mean([row["accessibility"] for row in values])),
            "candidate_exposure_rate": float(np.mean([row["candidate_exposed"] for row in values])),
            "mean_reciprocal_rank": float(np.mean([row["reciprocal_rank"] for row in values])),
            "median_exposed_rank": (float(np.median([row["candidate_rank"] for row in exposed]))
                                    if exposed else None),
            "relation_validation_rate": float(np.mean([row["relation_validated"] for row in values])),
            "relation_validation_given_exposure": (float(np.mean([row["relation_validated"] for row in exposed]))
                                                    if exposed else None),
            "null_survival_rate": float(np.mean([row["null_survived"] for row in values])),
            "null_survival_given_validation": (float(np.mean([row["null_survived"] for row in relation_validated]))
                                                if relation_validated else None),
            "decoy_promotion_rate": float(np.mean([row["decoy_promoted"] for row in values])),
            "clean_stagewise_pass_rate": float(np.mean([row["clean_stagewise_pass"] for row in values])),
        }
    return {"unit_of_inference": "independent packet",
            "stage_source": "already-frozen receipt summaries; no truth inspected",
            "n_packets": len(per_packet), "by_arm_family": by_arm_family,
            "per_packet": per_packet}


def _candidate_pvalue_entries(receipt: Mapping[str, Any], *, pvalues_key: str,
                              candidate_id_key: str, pvalue_key: str) -> list[tuple[str, float]]:
    """Normalise a frozen per-packet candidate p-value list or mapping."""
    raw = receipt.get(pvalues_key)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        entries = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise TypeError("candidate p-value entries must be mappings")
            entries.append((_required(item, candidate_id_key), _required(item, pvalue_key)))
    else:
        raise TypeError("candidate p-values must be a mapping or sequence of mappings")
    normalised = []
    for candidate_id, pvalue in entries:
        pvalue = float(pvalue)
        if not np.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise ValueError("candidate p-values must be finite values in [0, 1]")
        normalised.append((str(candidate_id), pvalue))
    return normalised


def adjust_candidate_pvalues_arm_family(
    receipts: Iterable[Mapping[str, Any]], *, pvalues_key: str = "candidate_pvalues",
    candidate_id_key: str = "candidate_id", pvalue_key: str = "p_value",
) -> dict[str, Any]:
    """Apply BH once across every frozen candidate in each arm×family panel.

    In particular, it does **not** adjust candidate p-values separately for
    every packet.  Receipts are still validated at the packet boundary, while
    multiplicity is defined by the predeclared arm/family candidate family.
    The function returns new records and never mutates frozen receipts.
    """
    rows = validate_packet_receipts(receipts)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        arm, family, packet_id = str(row["arm"]), str(row["family"]), str(row["packet_id"])
        for candidate_id, pvalue in _candidate_pvalue_entries(
                row, pvalues_key=pvalues_key, candidate_id_key=candidate_id_key, pvalue_key=pvalue_key):
            grouped.setdefault((arm, family), []).append({"packet_id": packet_id,
                                                            "candidate_id": candidate_id,
                                                            "p_value": pvalue})
    adjusted: dict[str, Any] = {}
    for (arm, family), entries in sorted(grouped.items()):
        seen = set()
        for entry in entries:
            key = (entry["packet_id"], entry["candidate_id"])
            if key in seen:
                raise ValueError("duplicate candidate p-value within packet: %r" % (key,))
            seen.add(key)
        qs = adjust_family_pvalues({str(index): entry["p_value"]
                                    for index, entry in enumerate(entries)})
        output = [{**entry, "q_value": qs[str(index)]} for index, entry in enumerate(entries)]
        adjusted["%s::%s" % (arm, family)] = {"arm": arm, "family": family,
                                                "n_candidates": len(output),
                                                "multiplicity_family": "all frozen candidates in arm×family",
                                                "candidates": output}
    return {"method": "Benjamini-Hochberg", "unit_of_multiplicity": "frozen candidate",
            "grouping": "arm×family", "by_arm_family": adjusted}
