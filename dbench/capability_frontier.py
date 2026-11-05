"""Frozen capability-frontier task registry for the MORPHEUS programme.

This is deliberately a registry, not another scoring scheme.  It names the
hard scientific objects against which the same representation, compiler and
candidate budget will later be compared.  A task only enters this registry if
the Tier-I generator already exposes its complete typed truth and its claimed
components can be checked mechanically.  That keeps a convenient/easy family
from silently standing in for a required compositional capability.
"""
from __future__ import annotations

from dataclasses import dataclass

from .environments import build_world


@dataclass(frozen=True)
class FrontierTask:
    """One predeclared hard scientific-object family."""

    id: str
    family: str
    required_components: tuple[str, ...]
    purpose: str
    hard: bool = True


# The task IDs, component labels and purposes are part of the experimental
# contract.  They are selected before method outcomes, not after inspecting
# which families flatter a representation.
CAPABILITY_FRONTIER = (
    FrontierTask(
        "k3_invisible", "invisible_k3", ("AND",),
        "three-way object whose singleton and pair intermediates are null"),
    FrontierTask(
        "k4_invisible", "invisible_k4", ("AND",),
        "four-way object whose every proper intermediate is null"),
    FrontierTask(
        "k5_invisible", "invisible_k5", ("AND",),
        "five-way object whose every proper intermediate is null"),
    FrontierTask(
        "xor_k3_invisible", "invisible_xor_k3", ("XOR",),
        "three-way parity whose singleton and pair intermediates are null"),
    FrontierTask(
        "xor_k4_invisible", "invisible_xor_k4", ("XOR",),
        "four-way parity whose every proper intermediate is null"),
    FrontierTask(
        "context_threshold_space_lag", "boss1_cond_thresh_space_lag",
        ("CONDITION", "THRESHOLD", "SPACE", "LAG"),
        "context-specific, thresholded, lagged neighbourhood mechanism with a confound"),
    FrontierTask(
        "regime_xor_space", "boss2_xor_space_regime",
        ("ANOMALY", "CONDITION", "SPACE", "XOR"),
        "spatial parity visible only in a temporal regime"),
    FrontierTask(
        "causal_nonlinear_space", "boss3_hill_space_cause",
        ("CAUSE", "SATURATE", "SPACE"),
        "intervention-specific nonlinear spatial dose-response"),
    FrontierTask(
        "nonlinear_graph", "boss6_hill_graph", ("SATURATE", "GRAPH"),
        "nonlinear dose response that is specific to graph-motif enrichment"),
    FrontierTask(
        "causal_nonlinear_temporal", "boss7_causal_hill_lag",
        ("CAUSE", "SATURATE", "LAG"),
        "intervention-specific nonlinear response at a fixed temporal lag"),
    FrontierTask(
        "dynamics_manifold_regime", "boss4_ode_manifold_anomaly",
        ("ODE", "MANIFOLD", "ANOMALY"),
        "regime-specific law visible through a nonlinear chart"),
    FrontierTask(
        "design_interaction_threshold", "boss5_design_and_threshold",
        ("DESIGN", "AND", "THRESHOLD"),
        "experimental-design object composed with a thresholded interaction"),
    FrontierTask(
        "continuous_combination", "linear_combination", ("LINCOMB",),
        "continuous composition that cannot be replaced by separate main effects"),
)


def task_by_id(task_id: str) -> FrontierTask:
    for task in CAPABILITY_FRONTIER:
        if task.id == task_id:
            return task
    raise KeyError("unknown capability-frontier task %r" % task_id)


def task_components(world) -> set[str]:
    """Components named by a world without reading any method output."""
    out = set()
    for h in world.truth.objects:
        out.add(str(h.operator))
        params = dict(getattr(h, "params", {}) or {})
        out.update(str(x) for x in params.get("composition", ()) or ())
    return out


def validate_frontier(seed: int = 0) -> dict:
    """Build every task and prove its declared component coverage.

    This validates the benchmark instrument only.  It cannot return an
    efficacy result because it does not accept a representation or a method.
    """
    rows = []
    for task in CAPABILITY_FRONTIER:
        world = build_world(task.family, seed=seed, role="proposal")
        observed = task_components(world)
        missing = sorted(set(task.required_components) - observed)
        rows.append({
            "id": task.id,
            "family": task.family,
            "n_truth": len(world.truth.objects),
            "components": sorted(observed),
            "missing": missing,
            "passes": bool(world.truth.objects) and not missing,
        })
    return {"tasks": rows, "passes": all(x["passes"] for x in rows)}
