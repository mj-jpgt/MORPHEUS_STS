"""Focused contracts for the information-preserving residual structural core."""
import torch

from morpheus_encoder.residual_core import ResidualStructuralCore


def test_source_path_is_exact_and_residuals_start_neutral():
    torch.manual_seed(3)
    F = torch.randn(2, 4, 6, requires_grad=True)
    core = ResidualStructuralCore(d_source=6, d_residual=5, coordinate_dim=2)
    out = core(F)

    assert out.source is F
    assert torch.equal(out.source, F)
    assert out.combined.shape == (2, 4, 21)
    assert torch.equal(out.combined[..., :6], F)
    assert torch.count_nonzero(out.atomic) == 0
    assert torch.count_nonzero(out.relational) == 0
    assert torch.count_nonzero(out.continuous) == 0


def test_structural_inputs_and_gradients_work_without_raw_domain_data():
    torch.manual_seed(4)
    F = torch.randn(2, 3, 4, requires_grad=True)
    neighbors = torch.tensor([[[1, 2], [0, -1], [0, 1]],
                              [[1, -1], [0, 2], [1, -1]]])
    coordinates = torch.randn(2, 3, 3)
    core = ResidualStructuralCore(d_source=4, d_residual=3, coordinate_dim=3)

    out = core(F, neighbors=neighbors, coordinates=coordinates)
    # Source loss proves an exact direct backward path; branch loss proves the
    # trainable output projections are reachable on the first update.
    loss = out.source.square().sum() + out.combined[..., 4:].sum()
    loss.backward()

    assert F.grad is not None and torch.allclose(F.grad, 2 * F.detach())
    for branch in (core.atomic_branch, core.relational_branch, core.continuous_branch):
        assert branch.net[-1].weight.grad is not None
        assert float(branch.net[-1].weight.grad.abs().sum()) > 0


def test_neighbor_permutations_do_not_change_relational_channel_after_update():
    torch.manual_seed(5)
    F = torch.randn(1, 4, 3)
    core = ResidualStructuralCore(d_source=3, d_residual=2, coordinate_dim=0)
    # Make the residual visible; identity initialisation is independently
    # tested above, while this test checks graph aggregation semantics.
    with torch.no_grad():
        core.relational_branch.net[-1].weight.normal_(std=0.1)
    a = torch.tensor([[[1, 2], [0, 2], [0, 1], [-1, -1]]])
    b = torch.tensor([[[2, 1], [2, 0], [1, 0], [-1, -1]]])
    assert torch.allclose(core(F, neighbors=a).relational,
                          core(F, neighbors=b).relational)


def test_lagged_neighbourhoods_are_coordinate_native_and_padded_safely():
    # Row 2 reads row 1's neighbourhood mean at lag 1; missing predecessors
    # are exactly zero rather than fabricated self-links.
    F = torch.tensor([[[1.0], [3.0], [9.0]]])
    neighbours = torch.tensor([[[1], [0], [1]]])
    lags = torch.tensor([[[-1, -1, -1], [0, -1, -1], [1, 0, -1]]])
    core = ResidualStructuralCore(d_source=1, d_residual=1,
                                  coordinate_dim=0, temporal_lags=3)
    mean = core._neighbor_mean(F, neighbours)
    got = core._lagged_neighborhoods(mean, lags)
    # neighbourhood means are [3, 1, 3]; at row 2 lags are [row1,row0,none].
    assert torch.allclose(got[0, 2], torch.tensor([1.0, 3.0, 0.0]))
    assert torch.allclose(got[0, 0], torch.zeros(3))
