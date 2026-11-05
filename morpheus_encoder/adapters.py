"""Domain adapters: T^(m) = A_m(X^(m)).

One encoder per kind of observation, not one shared encoder pretending that
pixels, RNA counts and particle sets are alike. Every adapter is small,
swappable, and returns the same thing: a token tensor in the model's common
width plus a validity mask.

Batch convention (assumption, stated once and used everywhere)
-------------------------------------------------------------
The batch dimension is the **scientific unit** -- a cell, a spot, an event, a
grid cell-time, a graph. A world with n = 12000 rows gives B = 12000. Tokens
within a row are *features* of that unit (gene groups, image patches, particles,
grid patches). The object slots then attend across those feature tokens, and a
slot's per-row activation is its loading, which is what the factor expert needs
to look like NMF and what the compiler needs as a candidate variable.

This is a choice, and it is the one that makes discovery objects dataset-level
while keeping attention cost K x L rather than K x n.

Every adapter exposes:

    adapter.modality      the key in X = {X^(m)}
    adapter.domain        one of dbench.world.DOMAINS
    adapter.d_token       the adapter's NATIVE token width
    adapter.d_core        the width the shared core reads (== d_token unless the
                          adapter harmonises)
    adapter.n_tokens      L, tokens emitted per unit (None if data-dependent)
    adapter.descriptor()  -> ModalityDescriptor, what this observation IS
    adapter.modality_embedding() -> (D_MODALITY_EMBED,) unit-norm e_m
    adapter(x, ...)   ->  AdapterOutput(tokens (B, L, d_core), mask (B, L) bool,
                                        mod_embed (d_mod,), aux)

Adapter, not core, owns modality identity (H3 redesign)
-------------------------------------------------------
The core used to key its cross-attention and its router on the modality NAME,
so attaching a genuinely new modality meant new core parameters and Headline
3's "only a new adapter is fitted" held only where the new domain reused an
existing name and token width. Two things moved here to close that:

    e_m         the modality embedding. Produced by the adapter from metadata
                (`modality.modality_embedding`), consumed by the core through
                fixed-shape linear maps that have no modality axis to grow.
    core_proj   width harmonisation. The shared cross-attention has ONE key and
                value projection at width `d_core`, so an adapter whose native
                token width differs carries the projection itself. It is
                `nn.Identity()` -- literally zero parameters and zero
                arithmetic -- in the common case `d_core == d_token`, which is
                what keeps the single-modality path bitwise unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modality import (D_MODALITY_EMBED, ModalityDescriptor,
                       identity_residual_init, modality_embedding)


@dataclass
class AdapterOutput:
    tokens: torch.Tensor              # (B, L, d_core)
    mask: torch.Tensor                # (B, L) bool, True = real token
    mod_embed: torch.Tensor = None    # (D_MODALITY_EMBED,) unit-norm e_m
    aux: dict = field(default_factory=dict)

    def shapes(self) -> dict:
        return {"tokens": tuple(self.tokens.shape), "mask": tuple(self.mask.shape)}


class DomainAdapter(nn.Module):
    """Base class. Subclasses implement `encode`; `forward` handles the mask."""

    domain: str = "tabular"
    #: what the numbers are, from `modality.MODALITY_UNITS`
    units: str = "unknown"
    #: structural facts the modality descriptor reads
    is_dense: bool = True
    is_set: bool = False
    is_frozen_embedding: bool = False

    def __init__(self, modality: str, d_token: int, n_tokens: int | None,
                 d_core: int | None = None, units: str | None = None,
                 d_input: int = 0):
        super().__init__()
        self.modality = modality
        self.d_token = int(d_token)
        self.d_core = self.d_token if d_core is None else int(d_core)
        self.n_tokens = n_tokens
        self.d_input = int(d_input)
        if units is not None:
            self.units = str(units)
        self.out_norm = nn.LayerNorm(d_token)
        # Width harmonisation lives on the adapter side of the freeze
        # boundary. Identity when the widths already agree -- not a Linear with
        # an identity init, an actual `nn.Identity`, so the parameter count and
        # the arithmetic are both unchanged in that case.
        self.core_proj = (nn.Identity() if self.d_core == self.d_token
                          else nn.Linear(self.d_token, self.d_core, bias=False))
        # the adapter's own modality-identity residual; see modality.py
        self.mod = nn.Module()
        self.mod.delta = nn.Parameter(
            identity_residual_init(self._descriptor_for_init()))

    # -- modality identity --------------------------------------------------
    def _descriptor_for_init(self) -> ModalityDescriptor:
        """The descriptor as known at construction time.

        Separate from `descriptor()` because an adapter may only learn its
        token count while building (the grid adapter computes `n_tokens` from
        the patch grid). The residual's seed must not depend on a field that is
        still None at `__init__`, or the same adapter would hash differently
        depending on when it was asked.
        """
        return ModalityDescriptor(
            name=self.modality, kind=self.domain, units=self.units,
            d_token=self.d_token, n_tokens=0, d_input=self.d_input,
            is_dense=self.is_dense, is_set=self.is_set,
            is_frozen_embedding=self.is_frozen_embedding)

    def descriptor(self) -> ModalityDescriptor:
        """What this observation is. Read by nothing that stores a name."""
        return ModalityDescriptor(
            name=self.modality, kind=self.domain, units=self.units,
            d_token=self.d_core, n_tokens=int(self.n_tokens or 0),
            d_input=self.d_input, is_dense=self.is_dense, is_set=self.is_set,
            is_frozen_embedding=self.is_frozen_embedding)

    def modality_embedding(self) -> torch.Tensor:
        """(D_MODALITY_EMBED,) unit-norm e_m. The only thing the core learns
        this modality's identity from."""
        return modality_embedding(self.descriptor(), self.mod.delta)

    def encode(self, x, **kw) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x, mask: torch.Tensor | None = None, **kw) -> AdapterOutput:
        t = self.encode(x, mask=mask, **kw)
        if t.dim() != 3:
            raise ValueError(
                "%s.encode must return (B, L, d_token), got %s"
                % (type(self).__name__, tuple(t.shape)))
        if t.shape[-1] != self.d_token:
            raise ValueError(
                "%s emitted width %d, expected d_token=%d"
                % (type(self).__name__, t.shape[-1], self.d_token))
        t = self.core_proj(self.out_norm(t))
        m = torch.ones(t.shape[:2], dtype=torch.bool, device=t.device)
        return AdapterOutput(tokens=t, mask=m,
                             mod_embed=self.modality_embedding(),
                             aux={"modality": self.modality,
                                  "domain": self.domain,
                                  "units": self.units,
                                  "d_core": self.d_core})

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# 1. set / gene-token adapter -- BMMC, HEST expression, Norman
# --------------------------------------------------------------------------
class GeneTokenAdapter(DomainAdapter):
    """Expression row -> L gene-set tokens.

    Genes are the set elements. Each gene carries a learned embedding, and a
    learned soft assignment sends it to one or more of L token groups; a token
    is the expression-weighted sum of its group's gene embeddings. Written this
    way rather than as a plain MLP because the assignment matrix is the object
    a later analysis actually wants to read: it is a gene-programme membership,
    and `gene_assignment()` returns it.

    The assignment uses entmax when `sparse_assignment=True`, so a gene belongs
    to an exact subset of groups rather than a little bit of all of them.
    """

    domain = "molecular"
    units = "counts"

    def __init__(self, modality: str, n_genes: int, d_token: int = 128,
                 n_tokens: int = 16, log1p: bool = False,
                 sparse_assignment: bool = True, alpha: float = 1.5,
                 seed: int = 0, d_core: int | None = None):
        super().__init__(modality, d_token, n_tokens, d_core=d_core,
                         units="logcounts" if log1p else "counts",
                         d_input=n_genes)
        g = torch.Generator().manual_seed(seed)
        self.n_genes = int(n_genes)
        self.log1p = bool(log1p)
        self.sparse_assignment = bool(sparse_assignment)
        self.alpha = float(alpha)
        self.gene_embed = nn.Parameter(
            torch.randn(n_genes, d_token, generator=g) / d_token ** 0.5)
        self.assign_logits = nn.Parameter(
            torch.randn(n_genes, n_tokens, generator=g) * 0.1)
        self.gene_scale = nn.Parameter(torch.zeros(n_genes))
        self.token_bias = nn.Parameter(torch.zeros(n_tokens, d_token))

    def gene_assignment(self) -> torch.Tensor:
        if self.sparse_assignment:
            from .sparse import entmax
            return entmax(self.assign_logits, alpha=self.alpha, dim=-1)
        return torch.softmax(self.assign_logits, dim=-1)

    def encode(self, x, mask=None, **kw) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.dim() != 2 or x.shape[-1] != self.n_genes:
            raise ValueError("expected (B, %d) expression, got %s"
                             % (self.n_genes, tuple(x.shape)))
        x = x.to(self.gene_embed.dtype)
        if self.log1p:
            x = torch.log1p(x.clamp_min(0))
        x = x * F.softplus(self.gene_scale + 0.5413)     # softplus(0.5413) ~ 1
        A = self.gene_assignment()                       # (G, L)
        # (B,G) x (G,L) x (G,D) -> (B,L,D). Contracted explicitly as one GEMM
        # against a (G, L*D) matrix rather than left to einsum's path chooser:
        # the shared `g` index sits in all three operands, and the order einsum
        # picked materialised a (B,G,L) intermediate that is 12000x2000x16 at
        # the target size.
        W = (A[:, :, None] * self.gene_embed[:, None, :]).reshape(
            self.n_genes, self.n_tokens * self.d_token)
        t = (x @ W).view(x.shape[0], self.n_tokens, self.d_token)
        return t / (A.sum(0).clamp_min(1e-6)[None, :, None] ** 0.5) + self.token_bias


# --------------------------------------------------------------------------
# 2. generic tabular / MLP adapter -- planted Tier-0 worlds
# --------------------------------------------------------------------------
class TabularAdapter(DomainAdapter):
    """(B, d) design matrix -> L tokens. The default for planted worlds."""

    domain = "tabular"
    units = "zscore"

    def __init__(self, modality: str, d_in: int, d_token: int = 128,
                 n_tokens: int = 8, hidden: int = 256, standardize: bool = True,
                 d_core: int | None = None):
        super().__init__(modality, d_token, n_tokens, d_core=d_core,
                         units="zscore" if standardize else "unknown",
                         d_input=d_in)
        self.d_in = int(d_in)
        self.standardize = bool(standardize)
        self.in_norm = nn.LayerNorm(d_in) if standardize else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, n_tokens * d_token))
        self.token_pos = nn.Parameter(torch.zeros(n_tokens, d_token))

    def encode(self, x, mask=None, **kw) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.token_pos.dtype)
        if x.dim() != 2 or x.shape[-1] != self.d_in:
            raise ValueError("expected (B, %d), got %s" % (self.d_in, tuple(x.shape)))
        h = self.net(self.in_norm(x))
        return h.view(x.shape[0], self.n_tokens, self.d_token) + self.token_pos


# --------------------------------------------------------------------------
# 3. spatiotemporal grid adapter -- climate (NCEP / ERA5)
# --------------------------------------------------------------------------
class SpatioTemporalGridAdapter(DomainAdapter):
    """(B, C, H, W) or (B, C, T, H, W) -> patch tokens with a learned position.

    Position is a *learned* embedding over the patch grid rather than a fixed
    sinusoid: a reanalysis field is not translation invariant (the tropics are
    not the pole), and pretending it is throws away the one piece of structure
    a climate object is usually made of.
    """

    domain = "temporal"
    units = "field"

    def __init__(self, modality: str, in_channels: int, grid: tuple,
                 patch: tuple, d_token: int = 128, d_core: int | None = None):
        super().__init__(modality, d_token, None, d_core=d_core,
                         units="field", d_input=in_channels)
        self.in_channels = int(in_channels)
        self.grid = tuple(int(g) for g in grid)
        self.patch = tuple(int(p) for p in patch)
        if len(self.grid) != len(self.patch):
            raise ValueError("grid and patch must have the same rank")
        self.rank = len(self.grid)
        for g, p in zip(self.grid, self.patch):
            if g % p:
                raise ValueError("grid %s not divisible by patch %s"
                                 % (self.grid, self.patch))
        n_tok = 1
        dims = []
        for g, p in zip(self.grid, self.patch):
            dims.append(g // p)
            n_tok *= g // p
        self.patch_dims = tuple(dims)
        self.n_tokens = n_tok
        conv = nn.Conv2d if self.rank == 2 else nn.Conv3d
        self.proj = conv(in_channels, d_token, kernel_size=self.patch,
                         stride=self.patch)
        self.pos = nn.Parameter(torch.zeros(n_tok, d_token))

    def encode(self, x, mask=None, **kw) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.pos.dtype)
        want = (self.in_channels,) + self.grid
        if tuple(x.shape[1:]) != want:
            raise ValueError("expected (B,%s), got %s"
                             % (",".join(str(w) for w in want), tuple(x.shape)))
        h = self.proj(x)                                  # (B, D, *dims)
        h = h.flatten(2).transpose(1, 2)                  # (B, L, D)
        return h + self.pos


# --------------------------------------------------------------------------
# 4. particle-set Transformer -- LHCO
# --------------------------------------------------------------------------
class ParticleSetAdapter(DomainAdapter):
    """(B, P, F) unordered particles (+ (B, P) mask) -> L pooled tokens.

    Permutation *equivariant* self-attention over particles, then permutation
    *invariant* pooling by learned queries. An event has no particle ordering,
    so the adapter must not have one either; `test_adapters.py` asserts this
    against a random permutation rather than assuming it.

    Marked a stub in the build plan: the interface and the invariance are real,
    the physics inductive bias (pairwise Lorentz-invariant features, jet
    clustering) is not here yet and is where LHCO work will start.
    """

    domain = "particle"
    units = "kinematic"
    is_dense = False
    is_set = True

    def __init__(self, modality: str, n_features: int, d_token: int = 128,
                 n_tokens: int = 8, n_heads: int = 4, n_layers: int = 2,
                 d_core: int | None = None):
        super().__init__(modality, d_token, n_tokens, d_core=d_core,
                         units="kinematic", d_input=n_features)
        self.n_features = int(n_features)
        self.embed = nn.Sequential(
            nn.Linear(n_features, d_token), nn.GELU(),
            nn.Linear(d_token, d_token))
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_token, n_heads, batch_first=True)
            for _ in range(n_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.pool_q = nn.Parameter(torch.randn(n_tokens, d_token) * 0.02)
        self.pool = nn.MultiheadAttention(d_token, n_heads, batch_first=True)

    def encode(self, x, mask=None, **kw) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.pool_q.dtype)
        if x.dim() != 3 or x.shape[-1] != self.n_features:
            raise ValueError("expected (B, P, %d), got %s"
                             % (self.n_features, tuple(x.shape)))
        pad = None if mask is None else ~torch.as_tensor(mask, dtype=torch.bool)
        h = self.embed(x)
        for attn, ln in zip(self.layers, self.lns):
            a, _ = attn(h, h, h, key_padding_mask=pad, need_weights=False)
            h = ln(h + a)
        q = self.pool_q.unsqueeze(0).expand(x.shape[0], -1, -1)
        out, _ = self.pool(q, h, h, key_padding_mask=pad, need_weights=False)
        return out


# --------------------------------------------------------------------------
# 5. equivariant graph adapter -- OC20
# --------------------------------------------------------------------------
class EquivariantGraphAdapter(DomainAdapter):
    """Batched graphs with 3-D positions -> L invariant tokens per graph.

    E(3)-*invariant* by construction: messages are functions of interatomic
    distances only, so a rigid rotation or translation of the coordinates
    leaves every token identical. Node outputs are scalars-on-nodes, which is
    the invariant branch of an EGNN; the equivariant vector branch (forces) is
    the part that is stubbed, since a discovery object over OC20 structures is
    a scalar motif claim, not a force field.

    Inputs
        node_feat   (N, F)      per-node scalar features
        pos         (N, 3)      coordinates
        edge_index  (2, Ne)     directed message edges (both directions listed)
        batch       (N,)        which graph each node belongs to, 0..B-1
    """

    domain = "graph"
    units = "geometry"
    is_dense = False

    def __init__(self, modality: str, n_node_features: int, d_token: int = 128,
                 n_tokens: int = 4, n_layers: int = 2, n_rbf: int = 16,
                 cutoff: float = 6.0, d_core: int | None = None):
        super().__init__(modality, d_token, n_tokens, d_core=d_core,
                         units="geometry", d_input=n_node_features)
        self.n_node_features = int(n_node_features)
        self.n_rbf = int(n_rbf)
        self.cutoff = float(cutoff)
        self.embed = nn.Linear(n_node_features, d_token)
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d_token + n_rbf, d_token), nn.SiLU(),
                          nn.Linear(d_token, d_token))
            for _ in range(n_layers)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d_token, d_token), nn.SiLU(),
                          nn.Linear(d_token, d_token))
            for _ in range(n_layers)])
        self.readout_w = nn.Parameter(torch.ones(n_tokens, d_token))
        self.readout_b = nn.Parameter(torch.zeros(n_tokens, d_token))
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, n_rbf))

    def _rbf(self, d: torch.Tensor) -> torch.Tensor:
        gamma = (self.cutoff / max(self.n_rbf - 1, 1)) ** -2
        return torch.exp(-gamma * (d[:, None] - self.rbf_centers[None, :]) ** 2)

    def encode(self, x, mask=None, pos=None, edge_index=None, batch=None, **kw):
        if isinstance(x, dict):
            pos, edge_index, batch = x["pos"], x["edge_index"], x.get("batch", batch)
            x = x["node_feat"]
        if pos is None or edge_index is None:
            raise ValueError("EquivariantGraphAdapter needs pos and edge_index")
        x = torch.as_tensor(x, dtype=self.readout_w.dtype)
        pos = torch.as_tensor(pos, dtype=self.readout_w.dtype)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        n = x.shape[0]
        batch = (torch.zeros(n, dtype=torch.long, device=x.device)
                 if batch is None else torch.as_tensor(batch, dtype=torch.long))
        src, dst = edge_index[0], edge_index[1]
        d = (pos[src] - pos[dst]).norm(dim=-1)          # the invariant
        rbf = self._rbf(d)
        h = self.embed(x)
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([h[src], h[dst], rbf], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = h + upd(torch.cat([h, agg], dim=-1))
        n_graphs = int(batch.max()) + 1 if n else 1
        # Mean pool per graph, then expand into n_tokens learned readouts. The
        # readout is a per-token affine of the pooled graph vector, so each
        # token can specialise while remaining invariant.
        pooled = torch.zeros(n_graphs, h.shape[-1], dtype=h.dtype, device=h.device)
        pooled.index_add_(0, batch, h)
        cnt = torch.zeros(n_graphs, dtype=h.dtype, device=h.device)
        cnt.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        pooled = pooled / cnt.clamp_min(1.0)[:, None]
        # per-token FiLM on the invariant graph vector: each readout token can
        # specialise while every one of them stays a function of distances only
        return pooled[:, None, :] * self.readout_w[None] + self.readout_b[None]


# --------------------------------------------------------------------------
# 6. frozen pathology foundation-model adapter -- HEST H&E
# --------------------------------------------------------------------------
#: Backbones whose per-spot embeddings already exist on the store, written by
#: `cscd/w3_embed_ccrcc_*.py` / `cscd/prop1_fm.py` as per-sample .npz shards
#: with keys `embeddings` (n_spots, d), `barcodes`, `sample_id`, keyed
#: `HEST-{sample_id}-{barcode}`. This adapter NEVER loads a foundation model;
#: it consumes those arrays. Re-running an FM here would both cost GPU time
#: already spent and silently change the features under a frozen benchmark.
PATHOLOGY_EMBED_DIMS = {
    "h_optimus_0": 1536,
    "dinov2_vitl14": 1024,
    "prov_gigapath": 1536,
}


class FrozenPathologyAdapter(DomainAdapter):
    """Precomputed patch/spot embeddings -> L tokens.

    Accepts (B, d_embed) -- one embedding per spot, the HEST case -- or
    (B, P, d_embed) for multi-patch spots, in which case patches are pooled by
    learned queries the way `ParticleSetAdapter` pools particles.
    """

    domain = "spatial"
    units = "embedding"
    is_frozen_embedding = True

    def __init__(self, modality: str, backbone: str = "h_optimus_0",
                 d_embed: int | None = None, d_token: int = 128,
                 n_tokens: int = 8, n_heads: int = 4, freeze_input: bool = True,
                 d_core: int | None = None):
        d_embed = PATHOLOGY_EMBED_DIMS.get(backbone) if d_embed is None else d_embed
        if d_embed is None:
            raise ValueError(
                "unknown pathology backbone %r; pass d_embed explicitly or add "
                "it to PATHOLOGY_EMBED_DIMS" % (backbone,))
        super().__init__(modality, d_token, n_tokens, d_core=d_core,
                         units="embedding", d_input=int(d_embed))
        self.backbone = backbone
        self.d_embed = int(d_embed)
        self.freeze_input = bool(freeze_input)
        self.in_norm = nn.LayerNorm(d_embed)
        self.proj = nn.Linear(d_embed, d_token)
        self.token_q = nn.Parameter(torch.randn(n_tokens, d_token) * 0.02)
        self.patch_pool = nn.MultiheadAttention(d_token, n_heads, batch_first=True)

    def encode(self, x, mask=None, **kw) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.token_q.dtype)
        if self.freeze_input:
            x = x.detach()
        if x.dim() == 2:
            if x.shape[-1] != self.d_embed:
                raise ValueError("expected (B, %d) embeddings, got %s"
                                 % (self.d_embed, tuple(x.shape)))
            h = self.proj(self.in_norm(x))                  # (B, D)
            return h[:, None, :] + self.token_q[None, :, :]
        if x.dim() == 3:
            if x.shape[-1] != self.d_embed:
                raise ValueError("expected (B, P, %d) embeddings, got %s"
                                 % (self.d_embed, tuple(x.shape)))
            h = self.proj(self.in_norm(x))
            q = self.token_q.unsqueeze(0).expand(x.shape[0], -1, -1)
            pad = None if mask is None else ~torch.as_tensor(mask, dtype=torch.bool)
            out, _ = self.patch_pool(q, h, h, key_padding_mask=pad,
                                     need_weights=False)
            return out
        raise ValueError("expected (B, d) or (B, P, d), got %s" % (tuple(x.shape),))


ADAPTER_REGISTRY = {
    "gene": GeneTokenAdapter,
    "tabular": TabularAdapter,
    "grid": SpatioTemporalGridAdapter,
    "particle": ParticleSetAdapter,
    "graph": EquivariantGraphAdapter,
    "pathology": FrozenPathologyAdapter,
}
