"""Joint differentiable formation of scientific objects.

This is the N+100 architectural path in LOCKINv2.  A representation alone is
not asked to hope that a later compiler will reconstruct the right object.
Instead a bounded bank jointly learns an operand set S and a relation rho:

    o_k = sum_(S,rho) p_theta(S,rho) rho(z_S).

The module deliberately receives only slot states.  It has no target, truth,
latent, compiler score, or candidate-ranking input; a training harness may
evaluate its objects against response data, but cannot leak the answer into
object formation.  The same learned object queries can be applied to proposal
and independent replication environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn as nn

from .sparse import EntmaxGate


RELATIONS = ("AND", "XOR", "SYNERGY")
_COMPILABLE = {"AND", "XOR"}


@dataclass(frozen=True)
class FrozenObjectSelection:
    """A proposal-environment object dictionary, detached at the boundary."""

    membership: torch.Tensor              # (H,K), soft but detached
    type_index: torch.Tensor              # (H,), detached integer semantics
    arity: torch.Tensor                   # (H,), detached integer support size
    members: tuple[tuple[int, ...], ...]  # readable support per object
    confidence: torch.Tensor              # (H,), detached


@dataclass
class JointObjectOutput:
    """Differentiable values plus the object dictionary that produced them."""

    values: torch.Tensor                  # (B,H)
    membership: torch.Tensor              # (H,K)
    type_probabilities: torch.Tensor      # (H,R)
    arity_probabilities: torch.Tensor     # (H,max_order-1), arities 2..max_order
    relation_values: torch.Tensor         # (B,H,R)
    relation_values_by_arity: torch.Tensor  # (B,H,A,R)
    slot_activity: torch.Tensor           # (B,K)
    query_states: torch.Tensor            # (H,D)


class DifferentiableObjectFormation(nn.Module):
    """Learn a bounded, typed bank of compositional scientific objects.

    Membership is sparse entmax over scientific object slots.  Relation type
    is a small soft distribution during optimization and becomes a declared
    semantic type only when frozen.  The continuous SYNERGY route stays inside
    the representation; it is intentionally not laundered into an AND/XOR
    compiler claim unless a later typed mechanism supports it.
    """

    def __init__(self, d_slot: int, n_slots: int, n_objects: int = 12,
                 membership_alpha: float = 1.5, max_order: int = 6):
        super().__init__()
        if n_objects < 1 or n_slots < 2 or max_order < 2:
            raise ValueError("need n_objects>=1, n_slots>=2 and max_order>=2")
        self.d_slot = int(d_slot)
        self.n_slots = int(n_slots)
        self.n_objects = int(n_objects)
        # ``d_slot`` is the dimension of one scientific value, which can
        # legitimately be one for standardized source coordinates.  It is not
        # an adequate address space for selecting *which* coordinate belongs
        # to an object: the old mean-only scalar selector tied at zero and
        # deterministically froze every object onto columns 0/1.  Separate
        # response-free identity keys make distinct slots addressable without
        # changing the value space used for the scientific relation itself.
        self.d_query = max(8, self.d_slot)
        # A requested sixth-order object cannot exist in a four-slot encoder.
        # Clamp once at construction so both the arity head and frozen
        # dictionary share a real, executable arity range.
        self.max_order = min(int(max_order), int(n_slots))
        self.activity = nn.Linear(d_slot, 1)
        self.queries = nn.Parameter(torch.randn(n_objects, self.d_query) * 0.02)
        self.slot_keys = nn.Parameter(torch.randn(n_slots, self.d_query) * 0.02)
        # A learned data-dependent residual can refine the identity keys once
        # evidence supports it.  Beginning at zero avoids reintroducing the
        # standardized-mean artefact before any P/R/N objective is applied.
        self.slot_state_key = nn.Linear(self.d_slot, self.d_query, bias=False)
        nn.init.zeros_(self.slot_state_key.weight)
        self.membership_gate = EntmaxGate(alpha=membership_alpha)
        self.type_head = nn.Sequential(
            nn.Linear(d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, len(RELATIONS)))
        # Relation identity is primarily a property of a learned object, not
        # something that should flip solely because the same object is opened
        # on a new unlabeled episode.  The context head remains as a small,
        # learnable residual for genuinely context-conditioned structures.
        self.type_prior = nn.Parameter(torch.zeros(n_objects, len(RELATIONS)))
        self.context_type_scale = nn.Parameter(torch.tensor(0.01))
        # A relation type without a declared arity is not a complete
        # scientific object.  The head is trained through the soft mixture of
        # the bounded top-k relation values below, then frozen to one readable
        # cardinality at the proposal/replication boundary.
        self.arity_head = nn.Sequential(
            nn.Linear(d_slot, d_slot), nn.GELU(),
            nn.Linear(d_slot, self.max_order - 1))
        self.arity_prior = nn.Parameter(torch.zeros(n_objects, self.max_order - 1))
        self.context_arity_scale = nn.Parameter(torch.tensor(0.01))
        # Break arbitrary argmax ties toward the simplest complete typed
        # object. Replication gradients may move either prior; an
        # episode-specific random head may not redefine the object before it
        # can be tested.
        with torch.no_grad():
            self.type_prior[:, 0] = 0.1       # AND
            self.arity_prior[:, 0] = 0.1      # arity two

    def _membership(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if u.ndim != 3 or u.shape[1:] != (self.n_slots, self.d_slot):
            raise ValueError("u must have shape (B,%d,%d), got %r"
                             % (self.n_slots, self.d_slot, tuple(u.shape)))
        slot_state = u.mean(dim=0)                 # (K,D), environment-local
        key = self.slot_keys + self.slot_state_key(slot_state)
        logits = self.queries @ key.t()            # (H,K)
        membership = self.membership_gate(logits, dim=-1)
        query_state = membership @ slot_state      # (H,D)
        return membership, query_state

    @staticmethod
    def _relation_values(activity: torch.Tensor,
                         membership: torch.Tensor) -> torch.Tensor:
        """Continuous forms of AND, XOR and centred multiway synergy."""
        # A non-member has multiplicative identity 1.  Fractional entmax
        # memberships remain differentiable and become exact support at freeze.
        a = activity[:, None, :]                   # (B,1,K)
        m = membership[None, :, :]                 # (1,H,K)
        and_value = torch.prod(1.0 - m + m * a, dim=-1)
        # n-ary parity: non-members contribute 1 to the signed product.
        xor_value = 0.5 * (1.0 - torch.prod(1.0 - 2.0 * m * a, dim=-1))
        centered = a - a.mean(dim=0, keepdim=True)
        # A centred multiplicative residual. Adding 1 makes non-members an
        # identity and keeps the value defined under sparse soft support.
        synergy_value = torch.prod(1.0 + m * centered, dim=-1) - 1.0
        return torch.stack((and_value, xor_value, synergy_value), dim=-1)

    def _bounded_relation_values(self, activity: torch.Tensor,
                                 membership: torch.Tensor) -> torch.Tensor:
        """Relations for every legal arity with a top-k support.

        Raw entmax can legitimately retain more than ``max_order`` non-zero
        coordinates early in training.  Earlier ``freeze`` simply discarded
        those rows, so the joint bank emitted no readable typed object at all.
        Here every arity from 2 through ``max_order`` receives a bounded,
        renormalised top-k support.  Top-k indices are a proposal-side routing
        decision; gradients still flow through the selected membership weights
        and through the differentiable arity mixture.
        """
        values = []
        for arity in range(2, self.max_order + 1):
            top = torch.topk(membership, k=arity, dim=-1).indices
            mask = torch.zeros_like(membership).scatter_(-1, top, 1.0)
            sparse = membership * mask
            sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            values.append(self._relation_values(activity, sparse))
        return torch.stack(values, dim=2)  # (B,H,A,R)

    def forward(self, u: torch.Tensor) -> JointObjectOutput:
        membership, query_state = self._membership(u)
        activity = torch.sigmoid(self.activity(u).squeeze(-1))
        relation_values = self._relation_values(activity, membership)
        relation_values_by_arity = self._bounded_relation_values(activity, membership)
        type_logits = (self.type_prior
                       + self.context_type_scale * self.type_head(query_state))
        arity_logits = (self.arity_prior
                        + self.context_arity_scale * self.arity_head(query_state))
        type_probabilities = torch.softmax(type_logits, dim=-1)
        arity_probabilities = torch.softmax(arity_logits, dim=-1)
        values = (relation_values_by_arity
                  * type_probabilities[None, :, None, :]
                  * arity_probabilities[None, :, :, None]).sum(dim=(-1, -2))
        return JointObjectOutput(values=values, membership=membership,
                                 type_probabilities=type_probabilities,
                                 arity_probabilities=arity_probabilities,
                                 relation_values=relation_values,
                                 relation_values_by_arity=relation_values_by_arity,
                                 slot_activity=activity,
                                 query_states=query_state)

    def freeze(self, output: JointObjectOutput,
               support_threshold: float = 1e-6) -> FrozenObjectSelection:
        """Freeze a proposal dictionary for independent-environment testing.

        No response statistic enters this method.  Low-confidence semantic
        assignments remain ``HIGHER_ORDER`` in :meth:`proposals`, rather than
        being silently promoted to a more specific claim.
        """
        if not 0.0 <= support_threshold < 1.0:
            raise ValueError("support_threshold must lie in [0,1)")
        raw_membership = output.membership.detach().clone()
        probs = output.type_probabilities.detach()
        confidence, type_index = probs.max(dim=-1)
        arity = output.arity_probabilities.detach().argmax(dim=-1).to(torch.long) + 2
        membership = torch.zeros_like(raw_membership)
        members = []
        for h in range(self.n_objects):
            # A bounded support is required even when pre-training entmax is
            # dense.  Dropping overfull rows creates an empty object bank and
            # makes the later compiler reward atomic proxies by construction.
            k = int(arity[h])
            _, idx = torch.topk(raw_membership[h], k=k)
            weights = raw_membership[h, idx]
            membership[h, idx] = weights / weights.sum().clamp_min(1e-12)
            support = tuple(sorted(int(i) for i in idx.tolist()))
            members.append(support)
        return FrozenObjectSelection(membership=membership,
                                     type_index=type_index.detach().clone(),
                                     arity=arity.detach().clone(),
                                     members=tuple(members),
                                     confidence=confidence.detach().clone())

    def evaluate_frozen(self, u: torch.Tensor,
                        selection: FrozenObjectSelection,
                        gradient_source: JointObjectOutput | None = None) -> torch.Tensor:
        """Evaluate a frozen proposal dictionary on a new environment.

        When ``gradient_source`` is supplied, the forward value remains the
        exact frozen dictionary while the backward path follows the current
        differentiable membership/type distribution.  This is the important
        N+100 separation: the compiler cannot see a moving candidate syntax,
        but representation learning and object formation are still trained
        jointly rather than freezing S and rho permanently at recompilation.
        """
        if selection.membership.shape != (self.n_objects, self.n_slots):
            raise ValueError("selection belongs to a different object bank")
        if selection.arity.shape != (self.n_objects,):
            raise ValueError("selection arity belongs to a different object bank")
        activity = torch.sigmoid(self.activity(u).squeeze(-1))
        membership = selection.membership
        one_hot = torch.nn.functional.one_hot(selection.type_index,
                                              num_classes=len(RELATIONS))
        type_weights = one_hot.to(activity.dtype)
        if gradient_source is not None:
            if (gradient_source.membership.shape != membership.shape
                    or gradient_source.type_probabilities.shape != type_weights.shape):
                raise ValueError("gradient source belongs to a different object bank")
            # Exact frozen values forward; current differentiable objects
            # backward.  Both environments therefore train one object bank
            # without letting either rewrite the candidate list mid-cadence.
            membership = membership + (gradient_source.membership
                                       - gradient_source.membership.detach())
            type_weights = type_weights + (gradient_source.type_probabilities
                                           - gradient_source.type_probabilities.detach())
        relation_values = self._relation_values(activity, membership)
        return (relation_values * type_weights[None]).sum(dim=-1)

    def proposals(self, selection: FrozenObjectSelection,
                  min_confidence: float = 0.34) -> tuple[dict, ...]:
        """Readable bounded proposals without response-fitted semantics.

        With three relation types, an argmax relation is already a declared
        typed hypothesis once it clears the non-uniform floor (1/3).  A 0.50
        cutoff had a destructive side effect on independent episodes: it
        converted every otherwise valid frozen AND/XOR proposal into the
        untyped channel before the compiler could test it.  Confidence remains
        attached to each proposal for later ranking/characterisation; ordinary
        evidence and replication, not an arbitrary pre-test confidence gate,
        decide whether the formula survives.
        """
        out = []
        for h, members in enumerate(selection.members):
            if not 2 <= len(members) <= self.max_order:
                continue
            conf = float(selection.confidence[h])
            relation = RELATIONS[int(selection.type_index[h])]
            kind = relation if conf >= min_confidence and relation in _COMPILABLE else "HIGHER_ORDER"
            out.append({"id": "joint:%d" % h, "type": kind,
                        "members": members, "relation": relation,
                        "arity": int(selection.arity[h]),
                        "type_confidence": conf})
        return tuple(out)

    def coverage_proposals(self, output: JointObjectOutput, *,
                           n_query_objects: int = 4,
                           arities: tuple[int, ...] | None = None,
                           relations: tuple[str, ...] = ("AND", "XOR"),
                           proposal_query_scores: torch.Tensor | None = None,
                           support_slack: int = 0,
                           variants_per_arity: int = 1) -> tuple[dict, ...]:
        """Emit a bounded, truth-free typed coverage dictionary.

        The learned bank decides *which slots belong together*.  It must not,
        however, lose a scientifically plausible support merely because an
        early soft type/arity head makes one brittle argmax declaration.  This
        method therefore opens the top learned supports at a small, fixed
        grammar cover (declared arities x compilable relations).  The compiler
        still tests/ranks every object using proposal data; no response or
        benchmark truth enters this admission step.

        It is intentionally bounded and query-routed rather than an all-
        subsets enumeration.  Callers must record its cap and give every arm
        the identical cover.  The ordinary :meth:`freeze` API remains the
        compact final dictionary used during differentiable training.
        """
        if not 1 <= int(n_query_objects) <= self.n_objects:
            raise ValueError("n_query_objects must lie in [1, n_objects]")
        legal = tuple(range(2, self.max_order + 1)) if arities is None else tuple(int(v) for v in arities)
        if not legal or any(v < 2 or v > self.max_order for v in legal):
            raise ValueError("arities must be within the configured executable range")
        rels = tuple(str(v) for v in relations)
        if not rels or set(rels) - _COMPILABLE:
            raise ValueError("coverage relations must be non-empty compilable relations")

        if support_slack < 0 or variants_per_arity < 1:
            raise ValueError("support_slack must be non-negative and variants_per_arity positive")
        membership = output.membership.detach()
        # Candidate syntax remains response-free, but *which frozen learned
        # queries enter the bounded proposal family may use proposal-only
        # evidence just as an ordinary compiler screen does.  Ranking only by
        # membership concentration picked visually sparse but scientifically
        # irrelevant queries.  The score is one scalar per query, supplied by
        # the caller before R/N are opened; it cannot select an operand,
        # operator, arity or answer from hidden truth.
        top2 = torch.topk(membership, k=2, dim=-1).values.mean(dim=-1)
        if proposal_query_scores is None:
            rank = top2
            ranking = "membership_concentration"
        else:
            scores = torch.as_tensor(proposal_query_scores, device=membership.device,
                                     dtype=membership.dtype).reshape(-1)
            if scores.shape[0] != self.n_objects or not bool(torch.isfinite(scores).all()):
                raise ValueError("proposal_query_scores must be finite with one value per object query")
            rank = scores + top2 * 1e-8
            ranking = "proposal_evidence_then_membership_tiebreak"
        selected = torch.argsort(rank, descending=True)[:int(n_query_objects)]
        result = []
        for h in selected.tolist():
            for arity in legal:
                # One top-k support is brittle near a membership tie: a query
                # that has identified three of four genuine constituents can
                # silently lock a fourth unrelated coordinate into every
                # later compiler test. Open a bounded, response-free local
                # neighbourhood around that learned support instead. The
                # membership weights—not truth or response—rank variants.
                pool_width = min(self.n_slots, int(arity) + int(support_slack))
                pool = torch.topk(membership[h], k=pool_width).indices.tolist()
                supports = sorted(
                    (tuple(sorted(int(v) for v in group))
                     for group in combinations(pool, int(arity))),
                    key=lambda group: (-float(membership[h, list(group)].sum()), group),
                )[:int(variants_per_arity)]
                confidence = float(top2[h])
                for support_rank, support in enumerate(supports):
                    for relation in rels:
                        admission = ("typed_coverage" if support_slack == 0
                                     and variants_per_arity == 1
                                     else "typed_coverage_local_neighbourhood")
                        result.append({"id": "cover:%d:%s:%d:%d" %
                                       (h, relation, arity, support_rank),
                                       "type": relation, "relation": relation,
                                       "members": support, "arity": int(arity),
                                       "type_confidence": confidence,
                                       "admission": admission,
                                       "query_ranking": ranking,
                                       "support_slack": int(support_slack),
                                       "support_rank": int(support_rank)})
        return tuple(result)
