"""Modality identity as *metadata*, not as a name the core looks up.

This module exists because of one measured gap. The leave-one-domain-out
harness reported, verbatim:

    "the gap appears at the real-domain rotation the headline is actually
    about -- attaching a genuinely new modality name requires new core-side
    parameters (`ObjectSlots.attn[m]`, and the `n_modalities` axis of
    `ModalityRouter.mod_embed` and `slot_bias`), so 'only a new modality
    adapter' holds only where the new domain reuses an existing modality slot
    and token width."

A core that has to grow a parameter to meet a new modality is not a frozen
core, and Headline 3's whole claim is that only a new *adapter* is fitted. So
modality identity stops being a dictionary key and becomes a vector:

    e_m = normalize( normalize(features(descriptor) @ MODALITY_BASIS) + delta_m )

`features` is deterministic and parameter-free -- it reads what the observation
*is* (kind, units, token width, token count, input width, whether it is a dense
matrix / an unordered set / a frozen upstream embedding), never what it is
called. `MODALITY_BASIS` is one fixed matrix shared by every adapter and every
modality; it is a constant of this module, not a parameter of anything.
`delta_m` is the only learned part and it belongs to the adapter.

Why the deterministic half is worth having (rather than just a free learned
vector per adapter): two adapters of the *same kind* start close together in
modality space. A frozen core that learned how to read a `spatial` token stream
meets a new spatial modality already pointing in roughly the right direction.
The name lookup it replaces had no such structure -- every new name was an
independent random row.

Why `delta_m` exists at all: without it, two `molecular` adapters with the same
token geometry would be *identical* in modality space and the router could not
tell them apart. Its init is a deterministic hash of the adapter's identity, at
0.35/sqrt(d) per component, so the kind stays the dominant direction and the
identity is a perturbation of it.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

#: Closed vocabulary of observation KINDS. These are physical kinds of
#: measurement, not modality names: `DomainAdapter.domain` already ranges over
#: them. A kind outside the vocabulary maps to `other` and is separated from
#: other unknown kinds by the identity residual, so an unseen kind is
#: well-defined rather than an error.
MODALITY_KINDS = ("tabular", "molecular", "spatial", "temporal", "particle",
                  "graph", "image", "sequence", "other")

#: Closed vocabulary of measurement UNITS. What the numbers are, which is a
#: different question from what instrument produced them: raw counts and
#: log-counts of the same assay route differently and should say so.
MODALITY_UNITS = ("counts", "logcounts", "zscore", "embedding", "field",
                  "kinematic", "geometry", "unknown")

#: 6 scalars: log token width, log token count, log input width, and three
#: structural flags (dense matrix / unordered set / frozen upstream embedding).
N_MODALITY_SCALARS = 6

D_MODALITY_META = len(MODALITY_KINDS) + len(MODALITY_UNITS) + N_MODALITY_SCALARS

#: Width of e_m. The core's three consumers of e_m are all (something, d_mod)
#: linear maps, so this number -- not the number of modalities -- is what fixes
#: the core's shape.
D_MODALITY_EMBED = 32

#: Relative weights of the feature blocks. NOT free parameters and not tuned
#: to a test: the first version of this module gave every block weight 1, and
#: the consequence was measured (`scripts/h3_props.py` P8). Two modalities of
#: DIFFERENT kinds that happen to share their units, their structural flags and
#: roughly their token geometry -- the normal case -- then sat at mean pairwise
#: cosine 0.679, because the blocks they share outweighed the one block that
#: distinguishes them. The pre-redesign `mod_embed` table held i.i.d. rows at
#: cosine ~0, and the router's unit-spread calibration is stated against
#: near-orthogonal modality directions, so 0.679 cost the object-level gate its
#: sparsity: signature exact zeros fell from 0.446 (pre-redesign, 5 seeds) to
#: 0.325.
#:
#: The criterion these weights are set by, stated before they were chosen:
#:   (a) two modalities of DIFFERENT kinds must be near-orthogonal -- mean
#:       pairwise cosine at most 0.25, the level at which the router's
#:       calibration still holds to within about 13% of the orthogonal case,
#:       since the informative spread goes as sqrt(1 - cos);
#:   (b) two modalities of the SAME kind must remain far closer to each other
#:       than to any other kind, because that is the entire transfer property a
#:       metadata embedding buys over a name lookup.
#: Raising the identity residual instead satisfies (a) and destroys (b): at
#: residual 1.5 the P8 table gives different-kind cosine 0.222 but same-kind
#: collapses with it. Weighting the blocks separates the two questions.
KIND_WEIGHT = 2.0
UNITS_WEIGHT = 0.6
SCALAR_WEIGHT = 0.5
FLAG_WEIGHT = 0.4

#: Seed for the fixed metadata basis. One number, in one place, shared by every
#: adapter ever built. Changing it changes every modality embedding at once,
#: which is why it is a module constant with a recorded value rather than a
#: constructor argument that could drift per adapter.
MODALITY_BASIS_SEED = 20260907

#: Versioned value pin for the module-level basis.  This is deliberately
#: separate from the seed: the seed documents how the original constant was
#: made, while the digest makes a silent change of its numerical value fail
#: loudly before any adapter-only transfer run can consume it.
MODALITY_BASIS_VERSION = "2026-09-13"
MODALITY_BASIS_SHA256 = (
    "e88cc5fe8a33b74e6dd5bbbe66f3fbba8b7b93440bdc872b5017fa25ff081a00")


def _basis() -> torch.Tensor:
    g = torch.Generator().manual_seed(MODALITY_BASIS_SEED)
    return torch.randn(D_MODALITY_META, D_MODALITY_EMBED, generator=g)


#: (d_meta, d_mod). A constant, deliberately NOT an nn.Parameter and NOT a
#: registered buffer of any module: it must not appear in any state_dict, must
#: not be trainable, and must be identical in every process that imports this
#: module. `test_h3_interface.py` pins its sha256.
MODALITY_BASIS = _basis()


def basis_sha256() -> str:
    return hashlib.sha256(
        MODALITY_BASIS.double().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class ModalityDescriptor:
    """What an observation *is*. Never what it is called.

    `name` is carried only so that the adapter can derive a deterministic
    identity residual from it and so that error messages can be read. It does
    not enter `features()` -- assert this, do not assume it: see
    `test_h3_interface.py::test_descriptor_features_ignore_the_modality_name`.
    """
    name: str = ""
    kind: str = "other"
    units: str = "unknown"
    d_token: int = 0
    n_tokens: int = 0
    d_input: int = 0
    is_dense: bool = True
    is_set: bool = False
    is_frozen_embedding: bool = False

    def features(self) -> torch.Tensor:
        """(D_MODALITY_META,) deterministic, parameter-free featurisation."""
        f = torch.zeros(D_MODALITY_META)
        k = self.kind if self.kind in MODALITY_KINDS else "other"
        f[MODALITY_KINDS.index(k)] = KIND_WEIGHT
        u = self.units if self.units in MODALITY_UNITS else "unknown"
        f[len(MODALITY_KINDS) + MODALITY_UNITS.index(u)] = UNITS_WEIGHT
        o = len(MODALITY_KINDS) + len(MODALITY_UNITS)
        # log-scaled so a 2000-gene panel and a 9-column design matrix differ
        # by an O(1) amount rather than by three orders of magnitude
        f[o + 0] = SCALAR_WEIGHT * math.log1p(max(int(self.d_token), 0)) / 8.0
        f[o + 1] = SCALAR_WEIGHT * math.log1p(max(int(self.n_tokens), 0)) / 8.0
        f[o + 2] = SCALAR_WEIGHT * math.log1p(max(int(self.d_input), 0)) / 16.0
        f[o + 3] = FLAG_WEIGHT if self.is_dense else 0.0
        f[o + 4] = FLAG_WEIGHT if self.is_set else 0.0
        f[o + 5] = FLAG_WEIGHT if self.is_frozen_embedding else 0.0
        return f

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm().clamp_min(eps)


def metadata_embedding(desc: ModalityDescriptor) -> torch.Tensor:
    """(D_MODALITY_EMBED,) the learned-free part of e_m: kind and geometry."""
    return _unit(desc.features() @ MODALITY_BASIS)


def identity_residual_init(desc: ModalityDescriptor, scale: float = 0.35
                           ) -> torch.Tensor:
    """A small, deterministic per-adapter residual.

    Seeded from a hash of (name, kind, units, d_token) so that rebuilding the
    same adapter gives the same modality embedding on any machine, and two
    same-kind adapters are distinguishable. `scale` is the target L2 norm of
    the residual against a unit-norm metadata direction, so the kind remains
    the dominant term (0.35 puts two same-kind adapters about 20 degrees apart).
    """
    key = "|".join((desc.name, desc.kind, desc.units, str(desc.d_token)))
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    g = torch.Generator().manual_seed(seed % (2 ** 63 - 1))
    v = torch.randn(D_MODALITY_EMBED, generator=g)
    return _unit(v) * float(scale)


def modality_embedding(desc: ModalityDescriptor,
                       delta: torch.Tensor | None = None) -> torch.Tensor:
    """(D_MODALITY_EMBED,) unit-norm e_m.

    Unit norm is load-bearing, not cosmetic: the router's calibration
    (`ModalityRouter.mod_proj` initialised at 1/sqrt(hidden)) assumes
    ||e_m|| = 1 in order to put the pre-normalisation logits at unit standard
    deviation, which is the whole reason `temperature = 1` means something.
    An adapter that emitted an unnormalised e_m would silently rescale every
    modality logit.
    """
    e = metadata_embedding(desc)
    if delta is not None:
        e = e + delta
    return _unit(e)
