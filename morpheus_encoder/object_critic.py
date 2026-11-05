"""Differentiable, truth-free critic for proposed scientific objects.

The critic is intentionally smaller than the symbolic compiler.  A proposer
can emit a soft bank of candidate objects in a discovery environment and the
same frozen-form candidates in independent replication and structured-null
environments.  This module turns their *observed, candidate-specific
evidence* into one continuous explanatory-compression value:

``replication agreement - structured-null evidence - parent explanation -
description complexity``.

It neither receives planted object identities nor knows relation names such as
AND or XOR.  The calling harness is responsible for constructing evidence
from measurements (for example, a differentiable residual association between
an object value and an observed outcome).  The exact compiler remains the
post-training mechanism for rendering and certifying a scientific claim.

The purpose is to give the residual core/object proposer a dense, general
gradient during training without requiring a discrete compiler success at
every optimisation step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ObjectCriticOutput:
    """Per-candidate terms returned by :class:`ReplicationObjectCritic`.

    All fields have shape ``(n_candidates,)``.  ``value`` is maximised by a
    training loop.  A positive ``agreement`` requires discovery and
    replication evidence to have the same direction; a large ``null_penalty``
    means structurally matched null environments exhibit comparable evidence.
    """

    value: torch.Tensor
    agreement: torch.Tensor
    null_penalty: torch.Tensor
    parent_novelty_penalty: torch.Tensor
    complexity_penalty: torch.Tensor
    discovery_strength: torch.Tensor
    replication_strength: torch.Tensor
    parent_explanatory_strength: torch.Tensor


class ReplicationObjectCritic(nn.Module):
    """Score soft candidate objects by replicable explanatory compression.

    Parameters
    ----------
    temperature:
        Positive scale converting a signed aggregate evidence value into a
        bounded soft strength by ``tanh(evidence / temperature)``.  It is a
        declared numerical scale, not fitted to planted truth.
    null_weight, parent_weight, complexity_weight:
        Non-negative coefficients for structured-null evidence and
        candidate-specific lower-complexity parent evidence and
        ``log(1 + complexity)`` respectively.  Keeping these explicit makes
        an operating point auditable and avoids relation-specific rewards.

    Inputs to :meth:`forward`
    -------------------------
    discovery_evidence, replication_evidence:
        Float tensors of shape ``(observations, candidates)``.  Each entry is
        a differentiable, signed observed-evidence contribution for a soft
        candidate object; it can be formed from any modality or relation.
    null_evidence:
        Either ``(observations, candidates)`` for one structured-null
        environment or ``(null_environments, observations, candidates)`` for
        several.  Null environments are averaged only after each environment
        has independently aggregated its observations, so a longer null list
        cannot change the scale merely by duplicating rows.
    complexity:
        A non-negative tensor broadcastable to ``(candidates,)``.  It may be
        a soft expected description length from the proposer (operands,
        operators, thresholds, lags, etc.).
    parent_discovery_evidence, parent_replication_evidence:
        Optional floating tensors of shape ``(parents, observations,
        candidates)``.  Each parent is a predeclared, lower-complexity
        candidate-specific explanation, opened independently in proposal and
        replication data.  The critic penalizes the strongest *replicated*
        parent explanation for each child candidate.  It does not infer a
        parent from relation labels or benchmark truth.
    parent_complexity:
        Required with parent evidence.  A non-negative tensor broadcastable
        to ``(parents, candidates)`` that is strictly smaller than the child
        complexity at every active parent-child entry.  This is a contract
        check, not a learned feature: callers must declare the lower-order
        alternatives before examining held-out outcomes.
    parent_mask:
        Optional boolean tensor of shape ``(parents, candidates)``.  ``True``
        declares an active parent-child comparison; a candidate can have no
        active parents.  Inactive entries cannot affect either the value or
        the lower-complexity contract.

    Notes
    -----
    Evidence is deliberately an input rather than a hidden response argument.
    This makes it impossible for the critic to access benchmark truth.  The
    training harness should use only discovery/replication measurements to
    construct these tensors and freeze candidate semantics before a final
    compiler certification.
    """

    def __init__(self, *, temperature: float = 1.0, null_weight: float = 1.0,
                 parent_weight: float = 1.0, complexity_weight: float = 0.05):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if null_weight < 0 or parent_weight < 0 or complexity_weight < 0:
            raise ValueError("penalty weights must be non-negative")
        # Buffers travel with a checkpoint but are not silently trainable.
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("null_weight", torch.tensor(float(null_weight)))
        self.register_buffer("parent_weight", torch.tensor(float(parent_weight)))
        self.register_buffer("complexity_weight", torch.tensor(float(complexity_weight)))

    @staticmethod
    def _environment_strength(evidence: torch.Tensor, *, name: str) -> torch.Tensor:
        """Aggregate a single environment's observations into signed evidence."""
        if evidence.ndim != 2:
            raise ValueError(f"{name} must have shape (observations, candidates), "
                             f"got {tuple(evidence.shape)}")
        if evidence.shape[0] < 1 or evidence.shape[1] < 1:
            raise ValueError(f"{name} must contain at least one observation and candidate")
        if not evidence.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        return evidence.mean(dim=0)

    def _soft_strength(self, aggregate_evidence: torch.Tensor) -> torch.Tensor:
        return torch.tanh(aggregate_evidence / self.temperature.to(aggregate_evidence))

    def _null_penalty(self, null_evidence: torch.Tensor, n_candidates: int) -> torch.Tensor:
        if null_evidence.ndim == 2:
            if null_evidence.shape[1] != n_candidates:
                raise ValueError("null_evidence candidate width must match discovery evidence")
            per_null = self._environment_strength(null_evidence, name="null_evidence").unsqueeze(0)
        elif null_evidence.ndim == 3:
            if null_evidence.shape[0] < 1 or null_evidence.shape[1] < 1:
                raise ValueError("null_evidence must contain at least one environment and observation")
            if null_evidence.shape[2] != n_candidates:
                raise ValueError("null_evidence candidate width must match discovery evidence")
            if not null_evidence.is_floating_point():
                raise TypeError("null_evidence must be floating point")
            per_null = null_evidence.mean(dim=1)
        else:
            raise ValueError("null_evidence must have shape (observations, candidates) or "
                             "(null_environments, observations, candidates)")
        # A null can be compelling in either direction.  Its signed mean is
        # converted to the same bounded scale as discovery/replication before
        # environments are averaged.
        return self._soft_strength(per_null).abs().mean(dim=0)

    def _parent_novelty_penalty(
            self, parent_discovery_evidence: torch.Tensor | None,
            parent_replication_evidence: torch.Tensor | None,
            parent_complexity: torch.Tensor | None, parent_mask: torch.Tensor | None,
            child_complexity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return strongest valid replicated parent evidence per candidate.

        A max, rather than a sum or average, has two useful properties here:
        it asks whether *any* simpler declared explanation already captures a
        child's evidence, and duplicating an identical parent cannot increase
        the penalty.  Parent identities remain entirely caller supplied.
        """
        supplied = (parent_discovery_evidence, parent_replication_evidence,
                    parent_complexity)
        if all(x is None for x in supplied):
            if parent_mask is not None:
                raise ValueError("parent_mask requires parent evidence and parent_complexity")
            zero = torch.zeros_like(child_complexity)
            return zero, zero
        if any(x is None for x in supplied):
            raise ValueError(
                "parent_discovery_evidence, parent_replication_evidence, and "
                "parent_complexity must be supplied together")
        assert parent_discovery_evidence is not None
        assert parent_replication_evidence is not None
        assert parent_complexity is not None

        def check_parent_evidence(evidence: torch.Tensor, name: str) -> torch.Tensor:
            if not isinstance(evidence, torch.Tensor) or evidence.ndim != 3:
                raise ValueError(
                    f"{name} must have shape (parents, observations, candidates)")
            if not evidence.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if evidence.shape[0] < 1 or evidence.shape[1] < 1:
                raise ValueError(f"{name} must contain at least one parent and observation")
            if evidence.shape[2] != child_complexity.numel():
                raise ValueError(f"{name} candidate width must match discovery evidence")
            return evidence.to(device=child_complexity.device, dtype=child_complexity.dtype)

        parent_d = check_parent_evidence(parent_discovery_evidence,
                                         "parent_discovery_evidence")
        parent_r = check_parent_evidence(parent_replication_evidence,
                                         "parent_replication_evidence")
        if parent_d.shape[0] != parent_r.shape[0]:
            raise ValueError("parent discovery and replication stacks must have the same parent count")
        n_parents, _, n_candidates = parent_d.shape
        try:
            parent_complexity = torch.broadcast_to(
                torch.as_tensor(parent_complexity, device=child_complexity.device,
                                dtype=child_complexity.dtype),
                (n_parents, n_candidates),
            )
        except RuntimeError as exc:
            raise ValueError(
                "parent_complexity must be broadcastable to (parents, candidates)") from exc
        if parent_mask is None:
            mask = torch.ones((n_parents, n_candidates), dtype=torch.bool,
                              device=child_complexity.device)
        else:
            if (not isinstance(parent_mask, torch.Tensor)
                    or parent_mask.dtype != torch.bool
                    or tuple(parent_mask.shape) != (n_parents, n_candidates)):
                raise ValueError("parent_mask must be boolean with shape (parents, candidates)")
            mask = parent_mask.to(device=child_complexity.device)

        if bool((parent_complexity.detach()[mask] < 0).any()):
            raise ValueError("active parent_complexity entries must be non-negative")
        child_for_parent = child_complexity.unsqueeze(0).expand_as(parent_complexity)
        if bool((parent_complexity.detach()[mask] >= child_for_parent.detach()[mask]).any()):
            raise ValueError(
                "every active parent must have strictly lower declared complexity than its child")

        # Parent evidence may be signed; agreement in either direction is an
        # explanatory alternative.  Non-replicating or direction-conflicting
        # parent evidence cannot subtract novelty.
        parent_agreement = (
            self._soft_strength(parent_d.mean(dim=1))
            * self._soft_strength(parent_r.mean(dim=1))
        ).clamp_min(0.0)
        any_parent = mask.any(dim=0)
        strongest = parent_agreement.masked_fill(~mask, float("-inf")).max(dim=0).values
        explanatory_strength = torch.where(any_parent, strongest,
                                           torch.zeros_like(strongest))
        return explanatory_strength, explanatory_strength

    def forward(self, discovery_evidence: torch.Tensor,
                replication_evidence: torch.Tensor,
                null_evidence: torch.Tensor, complexity: torch.Tensor, *,
                parent_discovery_evidence: torch.Tensor | None = None,
                parent_replication_evidence: torch.Tensor | None = None,
                parent_complexity: torch.Tensor | None = None,
                parent_mask: torch.Tensor | None = None) -> ObjectCriticOutput:
        d_raw = self._environment_strength(discovery_evidence, name="discovery_evidence")
        r_raw = self._environment_strength(replication_evidence, name="replication_evidence")
        if r_raw.shape != d_raw.shape:
            raise ValueError("discovery and replication candidate widths must match")
        if not torch.is_tensor(complexity):
            complexity = torch.as_tensor(complexity, dtype=d_raw.dtype, device=d_raw.device)
        complexity = complexity.to(device=d_raw.device, dtype=d_raw.dtype)
        try:
            complexity = torch.broadcast_to(complexity, d_raw.shape)
        except RuntimeError as exc:
            raise ValueError("complexity must be broadcastable to (candidates,)") from exc
        if bool((complexity.detach() < 0).any()):
            raise ValueError("complexity must be non-negative")

        d = self._soft_strength(d_raw)
        r = self._soft_strength(r_raw)
        agreement = d * r
        null_penalty = self._null_penalty(null_evidence, d.numel())
        parent_explanatory_strength, parent_novelty_penalty = self._parent_novelty_penalty(
            parent_discovery_evidence, parent_replication_evidence,
            parent_complexity, parent_mask, complexity)
        complexity_penalty = torch.log1p(complexity)
        value = (agreement
                 - self.null_weight.to(d) * null_penalty
                 - self.parent_weight.to(d) * parent_novelty_penalty
                 - self.complexity_weight.to(d) * complexity_penalty)
        return ObjectCriticOutput(value=value, agreement=agreement,
                                  null_penalty=null_penalty,
                                  parent_novelty_penalty=parent_novelty_penalty,
                                  complexity_penalty=complexity_penalty,
                                  discovery_strength=d,
                                  replication_strength=r,
                                  parent_explanatory_strength=parent_explanatory_strength)
