"""Universal continuous relations over unordered scientific-object slots.

This module deliberately lives *before* the symbolic compiler.  It gives the
encoder a compact, differentiable way to represent a k-ary relationship when
the correct scientific expression is not yet known.  It is a DeepSets-style
set function: member order cannot change its value, and the representation is
not restricted to AND, XOR, or any compiler opcode.

``ContinuousRelationBank`` never receives a response, planted truth, program
string, or compiler score.  A later proposer/certifier may decide whether a
continuous representation warrants a symbolic hypothesis.  Until then, every
output explicitly identifies itself as a *non-symbolic proposal*.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2

import torch
import torch.nn as nn


@dataclass
class ContinuousRelationOutput:
    """Continuous relation states and their auditable compactness metadata.

    ``relation_state`` is a learned continuous representation, not a program.
    ``approx_description_bits`` is intentionally only an MDL-relevant routing
    cost (member-address and arity cost); it must not be presented as the final
    compiler MDL of a scientific claim.
    """

    relation_state: torch.Tensor          # (B, H, E)
    value: torch.Tensor                   # (B, H), optional scalar readout
    membership: torch.Tensor              # (B, H, K), soft member weights
    arity_probabilities: torch.Tensor     # (H, A), arities 2..max_order
    expected_arity: torch.Tensor          # (H,)
    effective_member_count: torch.Tensor  # (B, H), exp(entropy)
    approx_description_bits: torch.Tensor # (H,)
    proposal_names: tuple[str, ...]
    relation_family: str = "continuous_set_relation"
    is_symbolic_program: bool = False

    def proposal_records(self) -> tuple[dict, ...]:
        """Return readable non-symbolic proposal records.

        The records intentionally do not contain an expression, opcode, or
        claim type.  They can be ranked or handed to a later compiler, but
        cannot be mistaken for a certified symbolic discovery.
        """
        return tuple({
            "id": name,
            "representation": self.relation_family,
            "is_symbolic_program": False,
            "status": "non_symbolic_continuous_proposal",
            "expected_arity": float(self.expected_arity[i].detach()),
            "approx_description_bits": float(self.approx_description_bits[i].detach()),
            "requires": "separate compiler/certification before any symbolic claim",
        } for i, name in enumerate(self.proposal_names))

    def as_symbolic_program(self):
        """Refuse an invalid shortcut from representation to scientific claim."""
        raise TypeError(
            "A continuous relation proposal is not a symbolic program; "
            "use a separate compiler and certification step."
        )


class ContinuousRelationBank(nn.Module):
    """A bounded bank of permutation-invariant continuous k-ary relations.

    A query softly selects slots, a declared arity mixture retains only the
    top-k selected members for each legal ``k``, and a shared DeepSets map
    evaluates weighted first and second moments.  Thus it can express a broad
    family of nonlinear set relations while keeping arity and member-address
    costs explicit.  Top-k determines a bounded *forward* support; gradients
    flow through the selected weights and all neural relation parameters.

    This is not a replacement for :class:`DifferentiableObjectFormation`.
    That class produces typed AND/XOR candidates for compiler evaluation;
    this class retains an intentionally non-symbolic universal route.
    """

    def __init__(self, d_slot: int, n_slots: int, *, n_relations: int = 8,
                 d_relation: int | None = None, max_order: int = 4):
        super().__init__()
        if d_slot < 1 or n_slots < 2 or n_relations < 1:
            raise ValueError("d_slot>=1, n_slots>=2 and n_relations>=1 are required")
        if max_order < 2:
            raise ValueError("max_order must be at least two")
        self.d_slot = int(d_slot)
        self.n_slots = int(n_slots)
        self.n_relations = int(n_relations)
        self.max_order = min(int(max_order), self.n_slots)
        self.d_relation = int(d_relation or d_slot)
        self.member_queries = nn.Parameter(torch.randn(n_relations, d_slot) * 0.02)
        # A fixed candidate-level prior avoids reading target/outcome data to
        # decide a relation's cardinality.  The slight two-member preference
        # implements compactness before any later MDL ranking.
        self.arity_logits = nn.Parameter(torch.zeros(n_relations, self.max_order - 1))
        with torch.no_grad():
            self.arity_logits[:, 0] = 0.1
        self.phi = nn.Sequential(
            nn.Linear(d_slot, self.d_relation), nn.GELU(),
            nn.Linear(self.d_relation, self.d_relation),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * self.d_relation, self.d_relation), nn.GELU(),
            nn.Linear(self.d_relation, self.d_relation),
        )
        self.value_head = nn.Linear(self.d_relation, 1)

    def _validate(self, slots: torch.Tensor, slot_mask: torch.Tensor | None) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[1:] != (self.n_slots, self.d_slot):
            raise ValueError("slots must have shape (B,%d,%d), got %r" %
                             (self.n_slots, self.d_slot, tuple(slots.shape)))
        if not slots.is_floating_point():
            raise ValueError("slots must be floating point")
        if slot_mask is None:
            return torch.ones(slots.shape[:2], dtype=torch.bool, device=slots.device)
        if slot_mask.shape != slots.shape[:2] or slot_mask.dtype is not torch.bool:
            raise ValueError("slot_mask must be bool with shape (B,n_slots)")
        if int(slot_mask.sum(dim=-1).min()) < 2:
            raise ValueError("each observation must contain at least two valid slots")
        return slot_mask

    @staticmethod
    def _weighted_moments(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant first/second moments for one selected set."""
        # features B,K,E; weights B,H,K -> B,H,E
        mean = torch.einsum("bhk,bke->bhe", weights, features)
        second = torch.einsum("bhk,bke->bhe", weights, features.square())
        variance = (second - mean.square()).clamp_min(0.0)
        return torch.cat((mean, variance), dim=-1)

    def forward(self, slots: torch.Tensor, *, slot_mask: torch.Tensor | None = None) -> ContinuousRelationOutput:
        mask = self._validate(slots, slot_mask)
        bsz = slots.shape[0]
        # Dot-product membership follows each slot's state.  Permuting the
        # slots permutes these logits/weights and leaves the set moments below
        # unchanged.
        logits = torch.einsum("h d,b k d->b h k", self.member_queries, slots)
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        membership = torch.softmax(logits, dim=-1)
        arity_probabilities = torch.softmax(self.arity_logits, dim=-1)
        features = self.phi(slots)
        states = []
        for arity in range(2, self.max_order + 1):
            # A padded observation may have fewer than this maximum requested
            # number of valid slots.  Its invalid entries have -inf logits, so
            # they receive zero weight after normalisation; the relation stays
            # defined rather than inventing a fake member.
            top = torch.topk(membership, k=arity, dim=-1).indices
            selected = torch.zeros_like(membership).scatter_(-1, top, 1.0)
            selected = selected * mask[:, None, :]
            weights = membership * selected
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            states.append(self.rho(self._weighted_moments(features, weights)))
        by_arity = torch.stack(states, dim=2)  # B,H,A,E
        relation_state = (by_arity * arity_probabilities[None, :, :, None]).sum(dim=2)
        value = self.value_head(relation_state).squeeze(-1)
        expected_arity = (arity_probabilities * torch.arange(
            2, self.max_order + 1, device=slots.device, dtype=slots.dtype)[None]).sum(dim=-1)
        entropy = -(membership * membership.clamp_min(1e-12).log()).sum(dim=-1)
        effective_member_count = entropy.exp()
        # This cost is intentionally a routing/representation proxy, not a
        # symbolic expression MDL.  It makes larger addressed sets more costly
        # without pretending a universal neural relation has been explained.
        arity_bits = log2(float(self.max_order - 1)) if self.max_order > 2 else 0.0
        approx_description_bits = expected_arity * log2(float(self.n_slots)) + arity_bits + 1.0
        names = tuple("continuous_relation:%03d" % i for i in range(self.n_relations))
        return ContinuousRelationOutput(
            relation_state=relation_state, value=value, membership=membership,
            arity_probabilities=arity_probabilities, expected_arity=expected_arity,
            effective_member_count=effective_member_count,
            approx_description_bits=approx_description_bits, proposal_names=names,
        )
