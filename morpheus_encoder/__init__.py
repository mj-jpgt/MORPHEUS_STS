"""MORPHEUS Scientific Object Encoder -- build-plan step 4.

Scope of this package
---------------------
The *architecture* of the Scientific Object Encoder, and nothing else. No
discovery training, no replication-grounded objective, no Pareto training, and
no acceptance gate is run from here. Step 4's gate ("the untrained hybrid
approximately matches cNMF on additive worlds, nonlinear methods on manifold
worlds, and standard models on utility") is left as a runnable script in
`scripts/gate_step4.py` and is deliberately NOT executed until DiscoveryBench
(step 1) and the baseline matrix (step 3) report. The benchmark freeze forbids
tuning this model against the benchmark; it does not forbid building the
skeleton first.

Layout
------
    sparse.py            alpha-entmax with exact zeros, sparsemax, sparsity stats
    sae.py               JumpReLU / TopK sparse heads with dead-atom revival
                         (ported from cscd/dictionary.py)
    operating_point.py   p = (alpha, delta, B, kappa) conditioning token
    adapters.py          T^(m) = A_m(X^(m)), one per domain
    slots.py             K object slots + object-specific modality routing
    experts.py           Mixture of Discovery Experts
    hypergraph.py        typed Scientific Object Hypergraph G_sci = (V,E,H,Gamma)
    encoder.py           the assembled stack
    world_adapter.py     dbench.World -> encoder inputs (soft dependency)

    scripts/router_study.py  reproducible receipt for the routing layer
    scripts/gpu_smoke.py     cost measurement at K=64, M=3, n=12000
    scripts/gate_step4.py    the step-4 acceptance gate (refuses until step 1
                             and step 3 report; supply the inputs, do not
                             weaken the refusal)
    runs/                    receipts: router_study, gpu_smoke_a100,
                             gate_step4 readiness, provenance (sha256+mtime,
                             since .git in this tree is empty)

Why the package is `morpheus_encoder/` and not `morpheus/`
----------------------------------------------------------
`morpheus/` in this working tree is an *existing, separate* project with its
own nested `.git`. Writing a new package into it would mix two repositories.
The parallel step-1 and step-2 agents took the same route (`dbench/`,
`morpheus_compiler/`), so this package sits beside them at the tree root.
"""

__version__ = "0.1.0"

from .hypergraph import (  # noqa: F401
    SCHEMA_VERSION, SciHypergraph, ObjectNode, Relation, Hyperedge, Condition,
)
from .operating_point import OperatingPoint  # noqa: F401
from .joint_objects import (  # noqa: F401
    DifferentiableObjectFormation, FrozenObjectSelection, JointObjectOutput,
)
from .residual_core import ResidualCoreOutput, ResidualStructuralCore  # noqa: F401
from .continuous_relations import (  # noqa: F401
    ContinuousRelationBank, ContinuousRelationOutput,
)
from .object_critic import ObjectCriticOutput, ReplicationObjectCritic  # noqa: F401
from .joint_training import (  # noqa: F401
    JointReplicationTrainingHarness, JointTrainingOutput, observed_signed_evidence,
)
from .discovery_episode import DiscoveryEpisode, EpisodePartitions, world_to_episode  # noqa: F401
from .discovery_representation import (  # noqa: F401
    IdentityPreservingEncoder, RepresentationConfig, fit_response_free, represent,
)
from .discovery_episode import (  # noqa: F401
    DiscoveryEpisode, EpisodePartitions, world_to_episode,
)
from .sparse import (  # noqa: F401
    entmax, entmax15, sparsemax, sparsity_stats, topk_gate,
    gate_sparsity_loss, leak_schedule, EntmaxGate,
)
