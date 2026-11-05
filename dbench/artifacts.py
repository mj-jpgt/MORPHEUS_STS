"""Planted artifacts: shortcut structures a well-behaved discoverer must not report.

Every artifact here is a *typed object* that a discovery system can genuinely
construct and score highly on, and that is false by construction. They are
planted into the same latent space as the true objects, so an oracle-latent
method is exposed to them too -- an artifact only visible to a learned encoder
would let the oracle gate pass vacuously.

Classes fall into two groups.

Classical (inherited and extended from cscd/certify_d.py and cscd/spike_in.py)
-----------------------------------------------------------------------------
sign_reversing    effect present on the discovery split and reversed on the
                  held-out split. Flagging it is a false discovery and no
                  held-out replication can rescue it.
discovery_only    effect on the discovery split, nothing to replicate into.
pure_null         independent noise; no dependence on anything.
nuisance_confound Q drives both A and Y with no A->Y edge. The A~Y association
                  is real and reproducible and still not a mechanism.
batch_aligned     cross-modal correlation carried entirely within batch, with
                  the sign flipped between batches.

Adversarial against MORPHEUS as specified
-----------------------------------------
The architecture in the spec has four load-bearing mechanisms: object slots,
object-specific modality gates, expert routing, and the replication gate. Each
class below is aimed at exactly one of them, and each is refutable only by a
mechanism the architecture has *in addition* to the one it attacks. The point
of the group is that DiscoveryBench should not flatter its own author: a
system that scores well by replication alone fails here.

replication_mimicry  -> attacks THE REPLICATION GATE.
    An instrument/measurement covariate whose relationship to Y is a property
    of the generator, not of the world: the same functional form and the same
    sign in every independently generated environment. It replicates perfectly
    and forever. The outer gradient of replication-grounded training therefore
    rewards it. It is refutable only by conditioning on the declared nuisance
    covariate it is a deterministic function of -- i.e. by the hierarchical-
    novelty machinery, not by replication.

gate_capture         -> attacks OBJECT-SPECIFIC MODALITY GATES.
    A cross-modal direction shared by two modalities with alignment stronger
    than any true object's, driven entirely by a declared acquisition
    covariate, and carrying exactly zero information about Y in every
    environment. An objective that rewards cross-modal coupling opens both
    gates on it; the spec's own rule ("a cross-modal relationship receives no
    reward merely for being aligned") is precisely what this tests.

expert_mimicry       -> attacks EXPERT ROUTING.
    A relationship whose functional form is an exact match for one expert's
    inductive bias -- a sharp threshold -- where the threshold location is a
    batch boundary. The threshold expert fits it with a large likelihood gain
    over any linear parent, so routing endorses it. Conditioning on the batch
    covariate removes it entirely.

slot_splitting       -> attacks OBJECT SLOTS + per-object replication.
    A nuisance direction split into K pieces, each of which has a strong
    effect inside exactly one environment stratum, with balanced signs so that
    the pooled sum of the pieces carries no marginal association at all. K
    slots can each capture one piece and each piece looks like a clean
    context-gated object. The piece-to-stratum assignment and the signs are
    redrawn independently in every environment, so no piece survives
    cross-environment replication. It punishes a learner that allocates
    capacity per stratum and then replicates within, rather than across,
    environments.
"""
from __future__ import annotations

import numpy as np

from .util import zscore
from .world import Hypothesis

CLASSICAL = ("sign_reversing", "discovery_only", "pure_null",
             "nuisance_confound", "batch_aligned")
ADVERSARIAL = ("replication_mimicry", "gate_capture", "expert_mimicry",
               "slot_splitting")
ALL_CLASSES = CLASSICAL + ADVERSARIAL

# The instrument response is a property of the measurement process, so it is
# fixed once for the whole benchmark and is deliberately NOT derived from a
# world seed. That is the entire mechanism by which replication_mimicry
# survives independent regeneration.
_INSTRUMENT_SEED = 0x4D4F525048455553 % (2 ** 63)


def instrument_response():
    """The frozen instrument transfer function used by replication_mimicry.

    Returned as its coefficients so a test can assert that two independently
    generated environments really do share it.
    """
    r = np.random.default_rng(_INSTRUMENT_SEED)
    return {"beta": 0.9, "a1": float(r.uniform(0.8, 1.2)),
            "a2": float(r.uniform(-0.6, -0.3)), "a3": float(r.uniform(0.2, 0.4))}


def _instr_transform(q, c):
    q = zscore(q)
    return zscore(c["a1"] * q + c["a2"] * np.tanh(q) + c["a3"] * (q ** 2 - 1.0))


def plant(latents, nuisance, Y, env_labels, rng, classes=(), n_each=1,
          strength=0.8, n_strata=4):
    """Plant artifacts into an in-progress world.

    Mutates `latents` and `nuisance` in place, returns (Y, [Hypothesis, ...]).
    `Y` may be None for worlds with no response, in which case only the
    classes that do not touch Y (pure_null, gate_capture, batch_aligned) are
    planted.
    """
    n = len(next(iter(latents.values())))
    objs = []
    has_Y = Y is not None
    if has_Y:
        Y = np.asarray(Y, dtype=np.float64).copy()
    split = env_labels.get("split")
    if split is None:
        split = (rng.random(n) < 0.5).astype(int)
        env_labels["split"] = split
    batch = env_labels.get("batch")
    if batch is None:
        batch = rng.integers(0, 2, n)
        env_labels["batch"] = batch

    for cls in classes:
        for j in range(n_each):
            tag = "%s_%d" % (cls, j)

            # ---------------- classical -----------------------------------
            if cls == "pure_null":
                latents["Zart_" + tag] = rng.standard_normal(n)
                objs.append(_art(tag, cls, ("Zart_" + tag,), "Y",
                                 "independent noise; no dependence on anything",
                                 mode="relational", operator="ADD"))

            elif cls == "sign_reversing":
                z = rng.standard_normal(n)
                latents["Zart_" + tag] = z
                if has_Y:
                    Y = Y + strength * np.where(split == 0, 1.0, -1.0) * z
                objs.append(_art(tag, cls, ("Zart_" + tag,), "Y",
                                 "effect reverses sign between splits",
                                 mode="relational", operator="ADD"))

            elif cls == "discovery_only":
                z = rng.standard_normal(n)
                latents["Zart_" + tag] = z
                if has_Y:
                    Y = Y + strength * (split == 0) * z
                objs.append(_art(tag, cls, ("Zart_" + tag,), "Y",
                                 "effect exists only on the discovery split",
                                 mode="relational", operator="ADD"))

            elif cls == "nuisance_confound":
                q = rng.standard_normal(n)
                a = zscore(0.9 * q + 0.45 * rng.standard_normal(n))
                latents["Zart_" + tag] = a
                nuisance["Q_" + tag] = q
                if has_Y:
                    Y = Y + strength * q
                objs.append(_art(tag, cls, ("Zart_" + tag,), "Y",
                                 "Q drives both the state and Y; no state->Y edge",
                                 mode="mechanism", operator="CONFOUND",
                                 params={"nuisance": "Q_" + tag}))

            elif cls == "batch_aligned":
                u = rng.standard_normal(n)
                s = np.where(batch == 0, 1.0, -1.0)
                latents["Zart_%s_m1" % tag] = zscore(u + 0.3 * rng.standard_normal(n))
                latents["Zart_%s_m2" % tag] = zscore(s * u + 0.3 * rng.standard_normal(n))
                objs.append(_art(tag, cls,
                                 ("Zart_%s_m1" % tag, "Zart_%s_m2" % tag),
                                 "cross_modal",
                                 "cross-modal correlation carried within batch, "
                                 "sign-flipped between batches",
                                 mode="entity", operator="IDENTITY",
                                 params={"nuisance": "batch"}))

            # ---------------- adversarial ---------------------------------
            elif cls == "replication_mimicry":
                c = instrument_response()
                q = rng.standard_normal(n)          # fresh draw per environment
                z = _instr_transform(q, c)          # frozen transfer function
                latents["Zart_" + tag] = z
                nuisance["Q_instrument_" + tag] = q
                if has_Y:
                    Y = Y + c["beta"] * z
                objs.append(_art(
                    tag, cls, ("Zart_" + tag,), "Y",
                    "instrument response: identical functional form and sign in "
                    "every independently generated environment, so it replicates "
                    "by construction; a deterministic function of the declared "
                    "nuisance covariate Q_instrument_" + tag,
                    mode="relational", operator="ADD",
                    params={"nuisance": "Q_instrument_" + tag,
                            "beta": c["beta"], "attacks": "replication_gate"}))

            elif cls == "gate_capture":
                q = rng.standard_normal(n)          # declared acquisition covariate
                nuisance["Q_acquisition_" + tag] = q
                latents["Zart_%s_m1" % tag] = zscore(1.4 * q + 0.15 * rng.standard_normal(n))
                latents["Zart_%s_m2" % tag] = zscore(1.4 * q + 0.15 * rng.standard_normal(n))
                objs.append(_art(
                    tag, cls, ("Zart_%s_m1" % tag, "Zart_%s_m2" % tag),
                    "cross_modal",
                    "cross-modal alignment stronger than any true object's, "
                    "entirely explained by the declared acquisition covariate, "
                    "and independent of Y in every environment",
                    mode="entity", operator="IDENTITY",
                    params={"nuisance": "Q_acquisition_" + tag,
                            "attacks": "modality_gates"}))

            elif cls == "expert_mimicry":
                # A sharp threshold whose location is a batch boundary. Any
                # threshold/saturation expert fits it far better than a linear
                # parent; conditioning on batch removes all of it.
                nb = int(np.max(batch)) + 1
                edges = np.linspace(-2.0, 2.0, nb + 1)
                z = np.empty(n)
                for b in range(nb):
                    m = batch == b
                    z[m] = rng.uniform(edges[b], edges[b + 1], int(m.sum()))
                latents["Zart_" + tag] = z
                if has_Y:
                    Y = Y + strength * 1.3 * (batch >= nb / 2.0).astype(float)
                objs.append(_art(
                    tag, cls, ("Zart_" + tag,), "Y",
                    "apparent sharp threshold in Zart whose location is a batch "
                    "boundary; the response is a step in batch, not in Zart",
                    mode="relational", operator="THRESHOLD",
                    params={"nuisance": "batch", "attacks": "expert_routing",
                            "apparent_tau": float(edges[int(nb / 2)])}))

            elif cls == "slot_splitting":
                stratum = env_labels.get("stratum")
                if stratum is None:
                    stratum = rng.integers(0, n_strata, n)
                    env_labels["stratum"] = stratum
                k = int(np.max(stratum)) + 1
                # piece -> stratum assignment redrawn per environment
                assign = rng.permutation(k)
                # Balanced signs. Without them every piece pushes Y the same
                # way inside its own stratum, so the pooled sum of the pieces
                # carries a real marginal association (|r| = 0.22 at k = 4) and
                # an ordinary additive method finds the artifact without ever
                # allocating a slot per stratum. Balanced signs make the pooled
                # sum null in expectation, which is what the class claims and
                # what forces a learner to split it across slots before it can
                # be rewarded for it.
                signs = np.array([1.0] * (k // 2) + [-1.0] * (k - k // 2))
                signs = signs[rng.permutation(k)]
                names = []
                for p in range(k):
                    zp = rng.standard_normal(n)
                    nm = "Zart_%s_p%d" % (tag, p)
                    latents[nm] = zp
                    names.append(nm)
                    if has_Y:
                        Y = Y + strength * 1.2 * signs[p] * (stratum == assign[p]) * zp
                objs.append(_art(
                    tag, cls, tuple(names), "Y",
                    "K pieces each active in exactly one environment stratum; "
                    "the piece-to-stratum map is redrawn independently in every "
                    "environment, so no piece replicates across environments and "
                    "the pooled sum is null",
                    mode="relational", operator="CONDITION",
                    params={"strata": k, "assignment": assign.tolist(),
                            "signs": signs.tolist(),
                            "attacks": "object_slots+replication"}))
            else:
                raise ValueError("unknown artifact class: %s" % cls)

    return (Y if has_Y else None), objs


def _art(tag, cls, operands, target, notes, mode="relational", operator="ADD",
         params=None):
    return Hypothesis(
        id="ART:%s" % tag, mode=mode, operator=operator, operands=tuple(operands),
        target=target, params=dict(params or {}), true=False,
        artifact_class=cls, notes=notes,
        parents=tuple((o,) for o in operands))


def refutation_route(cls: str) -> str:
    """Which machinery is *supposed* to reject each class.

    Recorded so that a failure can be attributed: an artifact that slips past
    is a failure of the named mechanism, not a generic false discovery.
    """
    return {
        "sign_reversing": "held-out replication (sign agreement)",
        "discovery_only": "held-out replication (presence)",
        "pure_null": "the primary test itself / FDR control",
        "nuisance_confound": "conditioning on the declared nuisance Q",
        "batch_aligned": "cross-batch replication of the cross-modal object",
        "replication_mimicry": "conditioning on the declared instrument covariate "
                               "-- replication CANNOT reject it",
        "gate_capture": "requiring cross-modal objects to carry conditional "
                        "information beyond the acquisition covariate",
        "expert_mimicry": "parent/nuisance residualisation before fitting the "
                          "threshold expert",
        "slot_splitting": "replication across environments rather than within",
    }[cls]
