"""Domain adapters: shapes, gradients, and the invariances each one claims.

Where an adapter claims an invariance the test asserts it against a random
transformation rather than trusting the construction. The particle adapter
claims permutation invariance and the graph adapter claims E(3) invariance;
both are checked numerically.
"""
import torch

from morpheus_encoder.adapters import (EquivariantGraphAdapter,
                                       FrozenPathologyAdapter,
                                       GeneTokenAdapter, ParticleSetAdapter,
                                       PATHOLOGY_EMBED_DIMS,
                                       SpatioTemporalGridAdapter,
                                       TabularAdapter)
from morpheus_encoder.sparse import sparsity_stats


def _fwd_bwd(out, params):
    out.tokens.pow(2).mean().backward()
    grads = [p for p in params if p.grad is not None and p.grad.abs().sum() > 0]
    return len(grads)


def test_gene_token_adapter_shapes_and_grads():
    a = GeneTokenAdapter("rna", n_genes=300, d_token=64, n_tokens=12)
    x = torch.rand(40, 300) * 8
    o = a(x)
    assert o.tokens.shape == (40, 12, 64)
    assert o.mask.shape == (40, 12) and o.mask.all()
    assert _fwd_bwd(o, list(a.parameters())) > 0


def test_gene_assignment_is_sparse_and_a_readable_membership():
    a = GeneTokenAdapter("rna", n_genes=200, d_token=32, n_tokens=8,
                         sparse_assignment=True)
    with torch.no_grad():
        a.assign_logits.mul_(20.0)     # separate the logits, as training would
    A = a.gene_assignment()
    assert A.shape == (200, 8)
    assert (A.sum(-1) - 1).abs().max() < 1e-5
    s = sparsity_stats(A)
    # `max_offsupport_mass` used to be arithmetically pinned at 0.0 and is now
    # `max {|g| : 0 < |g| <= tol}` (see `sparse.sparsity_stats`). The live
    # property for a hard assignment is that the zeros are *exact* and there is
    # essentially nothing in the "nonzero but negligible" band.
    assert s["frac_exact_zero"] > 0.5
    assert s["frac_subtol_nonzero"] < 0.05, (
        "the assignment is leaking sub-tolerance mass: %.4f of coordinates are "
        "nonzero but <= %g" % (s["frac_subtol_nonzero"], s["offsupport_tol"]))
    assert s["mean_active"] < 8


def test_gene_adapter_rejects_wrong_width():
    a = GeneTokenAdapter("rna", n_genes=50, d_token=16, n_tokens=4)
    try:
        a(torch.rand(5, 49))
    except ValueError as exc:
        assert "50" in str(exc)
    else:
        raise AssertionError("wrong gene count must not be silently accepted")


def test_tabular_adapter():
    a = TabularAdapter("tab", d_in=17, d_token=48, n_tokens=6)
    o = a(torch.randn(33, 17))
    assert o.tokens.shape == (33, 6, 48)
    assert _fwd_bwd(o, list(a.parameters())) > 0


def test_grid_adapter_2d_and_3d():
    a2 = SpatioTemporalGridAdapter("clim", in_channels=3, grid=(16, 32),
                                   patch=(4, 8), d_token=32)
    o = a2(torch.randn(7, 3, 16, 32))
    assert o.tokens.shape == (7, 4 * 4, 32)
    a3 = SpatioTemporalGridAdapter("clim3", in_channels=2, grid=(4, 8, 8),
                                   patch=(2, 4, 4), d_token=32)
    o3 = a3(torch.randn(5, 2, 4, 8, 8))
    assert o3.tokens.shape == (5, 2 * 2 * 2, 32)
    assert _fwd_bwd(o3, list(a3.parameters())) > 0


def test_grid_adapter_refuses_indivisible_patch():
    try:
        SpatioTemporalGridAdapter("c", 1, grid=(10, 10), patch=(3, 3), d_token=8)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("an indivisible patch grid must be refused")


def test_particle_adapter_is_permutation_invariant():
    torch.manual_seed(0)
    a = ParticleSetAdapter("jets", n_features=5, d_token=32, n_tokens=4,
                           n_heads=4, n_layers=2).eval()
    x = torch.randn(6, 20, 5)
    perm = torch.randperm(20)
    with torch.no_grad():
        t0 = a(x).tokens
        t1 = a(x[:, perm]).tokens
    assert (t0 - t1).abs().max() < 1e-5, (
        "an event has no particle ordering; the adapter must not have one")


def test_particle_adapter_respects_the_padding_mask():
    torch.manual_seed(1)
    a = ParticleSetAdapter("jets", n_features=4, d_token=32, n_tokens=3).eval()
    x = torch.randn(3, 10, 4)
    m = torch.ones(3, 10, dtype=torch.bool)
    m[:, 6:] = False
    y = x.clone()
    y[:, 6:] = 99.0                      # garbage behind the mask
    with torch.no_grad():
        assert (a(x, mask=m).tokens - a(y, mask=m).tokens).abs().max() < 1e-5


def _random_graph(n_nodes=12, n_feat=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, n_feat, generator=g)
    pos = torch.randn(n_nodes, 3, generator=g)
    src = torch.arange(n_nodes).repeat_interleave(3)
    dst = torch.randint(0, n_nodes, (n_nodes * 3,), generator=g)
    return x, pos, torch.stack([src, dst]), torch.zeros(n_nodes, dtype=torch.long)


def test_graph_adapter_is_e3_invariant():
    torch.manual_seed(0)
    a = EquivariantGraphAdapter("cat", n_node_features=6, d_token=32,
                                n_tokens=4, n_layers=2).eval()
    x, pos, ei, b = _random_graph()
    # random rotation via QR, plus a translation
    q, r = torch.linalg.qr(torch.randn(3, 3))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    pos2 = pos @ q.t() + torch.tensor([3.0, -1.5, 0.25])
    with torch.no_grad():
        t0 = a(x, pos=pos, edge_index=ei, batch=b).tokens
        t1 = a(x, pos=pos2, edge_index=ei, batch=b).tokens
    assert (t0 - t1).abs().max() < 1e-4, (
        "graph tokens changed under a rigid motion; the adapter is not "
        "E(3)-invariant and an OC20 motif claim would depend on the frame")


def test_graph_adapter_batches_and_backprops():
    a = EquivariantGraphAdapter("cat", n_node_features=6, d_token=32, n_tokens=4)
    x0, p0, e0, _ = _random_graph(seed=0)
    x1, p1, e1, _ = _random_graph(n_nodes=9, seed=1)
    x = torch.cat([x0, x1])
    pos = torch.cat([p0, p1])
    ei = torch.cat([e0, e1 + 12], dim=1)
    batch = torch.cat([torch.zeros(12, dtype=torch.long),
                       torch.ones(9, dtype=torch.long)])
    o = a({"node_feat": x, "pos": pos, "edge_index": ei, "batch": batch})
    assert o.tokens.shape == (2, 4, 32)
    assert _fwd_bwd(o, list(a.parameters())) > 0


def test_pathology_adapter_consumes_precomputed_embeddings():
    for bb, d in PATHOLOGY_EMBED_DIMS.items():
        a = FrozenPathologyAdapter("he", backbone=bb, d_token=32, n_tokens=4)
        assert a.d_embed == d
        o = a(torch.randn(9, d))
        assert o.tokens.shape == (9, 4, 32)
    a = FrozenPathologyAdapter("he", backbone="h_optimus_0", d_token=32,
                               n_tokens=4)
    o = a(torch.randn(5, 7, 1536))         # multi-patch spot
    assert o.tokens.shape == (5, 4, 32)
    assert _fwd_bwd(o, list(a.parameters())) > 0


def test_pathology_adapter_does_not_backprop_into_the_frozen_features():
    a = FrozenPathologyAdapter("he", backbone="dinov2_vitl14", d_token=16,
                               n_tokens=2, freeze_input=True)
    x = torch.randn(4, 1024, requires_grad=True)
    a(x).tokens.pow(2).mean().backward()
    assert x.grad is None, (
        "the pathology features are frozen artefacts already on the store; a "
        "gradient reaching them means the benchmark's inputs are moving")


def test_unknown_backbone_is_refused():
    try:
        FrozenPathologyAdapter("he", backbone="not_a_real_fm")
    except ValueError as exc:
        assert "unknown pathology backbone" in str(exc)
    else:
        raise AssertionError("an unknown backbone must not get a guessed width")
