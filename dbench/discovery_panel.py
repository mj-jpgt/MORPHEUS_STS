"""Frozen, biology-centred capability panel for MORPHEUS.

This module is deliberately a compact experiment contract rather than a new
scoring system.  It says which scientific objects are tested, how many
independent packets comprise a condition, which conventional representation
arms are compared, and where a native specialist is eligible.  Recovery is
scored by the common representation/compiler evaluator; this module never
exposes a truth object to a method.

The Tier-II HEST conditions use a measured expression-and-geometry substrate
with a controlled response.  They are therefore real-substrate packets, not
claims that a tissue simulator is biology.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from .capability_frontier import task_by_id
from .environments import make_episode
from .world import Episode


@dataclass(frozen=True)
class PanelCondition:
    id: str
    family: str
    task_id: str | None
    tier: str
    required_components: tuple[str, ...]
    purpose: str
    specialist_eligible: bool
    specialist_reason: str


PANEL_ARMS = ("source", "reconstruction", "ssl", "morpheus")
SMOKE_PACKETS_PER_CONDITION = 5
MAIN_PACKETS_PER_CONDITION = 30

# Conditions are chosen before inspecting the result of any arm.  A
# specialist is marked eligible only when a normal native formulation can
# express the exact object; a bespoke scanner that encodes a boss generator is
# intentionally not treated as a scientific baseline.
DISCOVERY_CAPABILITY_PANEL_V1 = (
    PanelCondition("and_k3", "invisible_k3", "k3_invisible", "I", ("AND",),
                   "three-way interaction with null lower-order intermediates", False,
                   "pairwise interaction methods cannot express a three-way object"),
    PanelCondition("and_k4", "invisible_k4", "k4_invisible", "I", ("AND",),
                   "four-way interaction with null lower-order intermediates", False,
                   "pairwise interaction methods cannot express a four-way object"),
    PanelCondition("and_k5", "invisible_k5", "k5_invisible", "I", ("AND",),
                   "five-way interaction with null lower-order intermediates", False,
                   "pairwise interaction methods cannot express a five-way object"),
    PanelCondition("xor_k3", "invisible_xor_k3", "xor_k3_invisible", "I", ("XOR",),
                   "three-way parity with no useful proper subset", False,
                   "the native pairwise XOR scanner is not an exact formulation"),
    PanelCondition("xor_k4", "invisible_xor_k4", "xor_k4_invisible", "I", ("XOR",),
                   "four-way parity with no useful proper subset", False,
                   "the native pairwise XOR scanner is not an exact formulation"),
    PanelCondition("context_threshold_space_lag", "boss1_cond_thresh_space_lag",
                   "context_threshold_space_lag", "I",
                   ("CONDITION", "THRESHOLD", "SPACE", "LAG"),
                   "a thresholded lagged neighbourhood mechanism under context and confounding", False,
                   "a generator-specific boss scanner is not a native general specialist"),
    PanelCondition("regime_xor_space", "boss2_xor_space_regime", "regime_xor_space", "I",
                   ("ANOMALY", "CONDITION", "SPACE", "XOR"),
                   "a spatial parity mechanism active only in a temporal regime", False,
                   "a generator-specific boss scanner is not a native general specialist"),
    PanelCondition("hest_spatial", "hest_spatial", None, "II", ("SPACE",),
                   "neighbourhood signal on measured HEST expression and tissue geometry", False,
                   "the available spatial scanner is an own-cell-by-neighbour pair model, not this unary aggregate"),
    PanelCondition("hest_context_threshold_space", "hest_context_threshold_space", None, "II",
                   ("CONDITION", "THRESHOLD", "SPACE"),
                   "context-by-thresholded-neighbourhood signal on measured HEST tissue", False,
                   "no available native specialist expresses the complete controlled object"),
)


def panel_condition(identifier: str) -> PanelCondition:
    for condition in DISCOVERY_CAPABILITY_PANEL_V1:
        if condition.id == identifier:
            return condition
    raise KeyError("unknown panel condition %r" % identifier)


def panel_manifest(*, packets_per_condition: int = MAIN_PACKETS_PER_CONDITION) -> dict:
    """Return a serialisable contract and deterministic content hash."""
    if packets_per_condition < 1:
        raise ValueError("packets_per_condition must be positive")
    payload = {
        "schema": "morpheus_discovery_capability_panel/v1",
        "arms": list(PANEL_ARMS),
        "packets_per_condition": int(packets_per_condition),
        "conditions": [asdict(x) for x in DISCOVERY_CAPABILITY_PANEL_V1],
        "primary_comparison": "morpheus_minus_ssl",
        "tier2_interpretation": "real-substrate controlled-signal stress test; not external clinical validation",
        "specialist_policy": "only native exact-expression specialists are compared; inexpressibility is reported separately and never scored as a MORPHEUS win",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "contract_sha256": sha256(canonical.encode("utf-8")).hexdigest()}


def write_panel_manifest(path: str | Path, *, packets_per_condition: int = MAIN_PACKETS_PER_CONDITION) -> dict:
    manifest = panel_manifest(packets_per_condition=packets_per_condition)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_panel_episode(condition: PanelCondition | str, seed: int):
    """Build a blinded-method episode while retaining a held scoring handle.

    Tier I uses DiscoveryBench's normal independently generated proposal,
    replication and matched-null environments.  Tier II re-samples independent
    spot packets from the fixed measured slide and uses its explicitly
    calibrated target permutation.  Tier II is useful evidence about search in
    real covariance/geometry, but is not presented as patient-level transport.
    """
    condition = panel_condition(condition) if isinstance(condition, str) else condition
    if condition.tier == "I":
        return make_episode(condition.family, seed=seed, n_replication=1, with_utility=True)
    from . import spike_in
    factory = spike_in.TIER2_FAMILIES_HEST[condition.family]
    proposal = factory(seed=seed, role="proposal")
    replication = factory(seed=seed + 100_003, role="replication")
    null_world, preservation = spike_in.hest_iid_permute_null(proposal, seed=seed + 200_003)
    return Episode(name="%s/ep%d" % (condition.family, seed), D_P=proposal,
                   D_R=(replication,), D_N=(null_world,), D_U=None,
                   meta={"family": condition.family, "episode_seed": int(seed),
                         "tier": "II", "null_preservation": [preservation],
                         "real_substrate": True})


def validate_panel_contract(seed: int = 0) -> dict:
    """Validate declared Tier-I task coverage without running a method."""
    rows = []
    for condition in DISCOVERY_CAPABILITY_PANEL_V1:
        if condition.task_id is None:
            rows.append({"id": condition.id, "tier": condition.tier,
                         "passes": True, "reason": "Tier-II data opened only in a run environment"})
            continue
        task = task_by_id(condition.task_id)
        rows.append({"id": condition.id, "tier": condition.tier,
                     "family": task.family,
                     "passes": tuple(task.required_components) == condition.required_components})
    return {"manifest": panel_manifest(), "rows": rows,
            "passes": all(row["passes"] for row in rows)}
