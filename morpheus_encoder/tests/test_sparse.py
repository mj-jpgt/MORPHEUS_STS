"""entmax: exact zeros, exact simplex, exact gradients, controllable sparsity.

These are the tests the whole routing story rests on. If a gate's "zero" is
1e-9 rather than 0, then "this object does not use RNA" is false and every
downstream claim about object-specific coupling is a claim about a small number
instead of about an absence.
"""
import torch

from morpheus_encoder.sparse import (EntmaxGate, entmax, entmax15,
                                     gate_sparsity_loss, leak_schedule,
                                     sparsemax, sparsity_stats, topk_gate)


def test_entmax_is_on_the_simplex():
    x = torch.randn(32, 11, dtype=torch.double) * 3
    for a in (1.2, 1.5, 2.0, 3.0):
        y = entmax(x, alpha=a)
        assert torch.all(y >= 0)
        assert (y.sum(-1) - 1).abs().max() < 1e-12


def test_entmax_produces_exact_zeros_not_small_values():
    x = torch.randn(64, 16, dtype=torch.double) * 4
    y = entmax15(x)
    zero = y == 0
    assert zero.any(), "entmax produced no zeros at all on well-separated logits"
    # the load-bearing assertion: off-support mass is identically zero
    assert float(y[zero].abs().max() if zero.any() else 0.0) == 0.0
    stats = sparsity_stats(y)
    assert stats["frac_exact_zero"] > 0.5
    assert stats["mean_active"] < y.shape[-1]


def test_offsupport_mass_separates_a_sparse_gate_from_a_dense_one():
    """The repaired `max_offsupport_mass`, pinned on both sides of the question.

    The statistic this replaces could only return 0.0: it took `max |g|` over
    `{|g| <= 0.0}`. It returned 0.0 for entmax, softmax, uniform noise,
    all-ones, a constant and a uniformly-1e-9 tensor alike, so asserting
    `== 0.0` on one input tested nothing. The repaired statistic is
    `max {|g| : 0 < |g| <= tol}` with the fraction of coordinates in that band
    beside it, and this test asserts **values on a known-sparse input and on a
    known-dense one** and fails if they agree.

    Values measured 2026-09-06 at `tol = 1e-3`, torch.manual_seed(0),
    float64, shape (64, 16), logits `randn * 4`.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 16, dtype=torch.double) * 4
    sparse = sparsity_stats(entmax15(x))
    dense = sparsity_stats(torch.softmax(x, dim=-1))

    # known-sparse: 87.89% of the coordinates are *exactly* zero and only
    # 0.20% sit in the "nonzero but small enough that calling it zero would be
    # a lie" band.
    assert abs(sparse["frac_exact_zero"] - 0.878906) < 1e-5
    assert abs(sparse["frac_subtol_nonzero"] - 0.001953) < 1e-5
    assert abs(sparse["max_offsupport_mass"] - 6.43096e-4) < 1e-8

    # known-dense: not one exact zero, and **55.7% of the tensor** is sub-tolerance
    # mass -- the thing the old statistic reported as 0.0.
    assert dense["frac_exact_zero"] == 0.0
    assert abs(dense["frac_subtol_nonzero"] - 0.556641) < 1e-5
    assert abs(dense["max_offsupport_mass"] - 9.76603e-4) < 1e-8

    # the property the old assertion could not have: the two disagree.
    assert sparse["max_offsupport_mass"] != dense["max_offsupport_mass"]
    assert dense["frac_subtol_nonzero"] == 285.0 * sparse["frac_subtol_nonzero"]


def test_offsupport_mass_catches_the_tensor_the_dead_statistic_missed():
    """A uniformly 1e-9 tensor is *entirely* off-support mass.

    The old statistic returned 0.0 for it, which is the exact failure mode the
    repair exists for: "this object does not use RNA" must not be reported as
    an absence when it is a small number.
    """
    tiny = torch.full((64, 16), 1e-9, dtype=torch.double)
    s = sparsity_stats(tiny)
    assert s["max_offsupport_mass"] == 1e-9
    assert s["frac_subtol_nonzero"] == 1.0
    assert s["frac_exact_zero"] == 0.0
    # and a tensor with no sub-tolerance mass at all still returns exactly 0.0,
    # so 0.0 remains a reachable and meaningful value rather than the only one.
    ones = sparsity_stats(torch.ones(64, 16, dtype=torch.double))
    assert ones["max_offsupport_mass"] == 0.0
    assert ones["frac_subtol_nonzero"] == 0.0


def test_softmax_alpha_is_refused():
    try:
        entmax(torch.randn(4, 5), alpha=1.0)
    except ValueError as exc:
        assert "softmax" in str(exc)
    else:
        raise AssertionError("alpha=1 should be refused, it has no exact zeros")


def test_entmax_gradcheck_all_alphas():
    x = torch.randn(6, 9, dtype=torch.double, requires_grad=True)
    for a in (1.2, 1.5, 2.0, 3.0):
        assert torch.autograd.gradcheck(
            lambda z, a=a: entmax(z * 2.0, alpha=a), (x,), eps=1e-6, atol=1e-7)


def test_gradient_is_exactly_zero_off_support_and_nonzero_on_it():
    """Gradients must flow through the open gates and only through them."""
    x = (torch.randn(8, 12, dtype=torch.double) * 4).requires_grad_(True)
    y = entmax15(x)
    y.sum().backward()          # sum is constant, so this alone gives zero
    x.grad = None
    y = entmax15(x)
    w = torch.randn_like(y)
    (y * w).sum().backward()
    zero = (y == 0).detach()
    assert float(x.grad[zero].abs().max()) == 0.0
    assert float(x.grad[~zero].abs().max()) > 0.0


def test_sparsemax_equals_entmax_alpha_two():
    x = torch.randn(20, 13, dtype=torch.double) * 2
    assert (sparsemax(x) - entmax(x, alpha=2.0)).abs().max() < 1e-12


def test_sparsity_is_controllable_by_temperature():
    """Sparsity is a knob, not an accident of initialisation."""
    torch.manual_seed(0)
    logits = torch.randn(256, 12)
    gate = EntmaxGate(alpha=1.5, temperature=1.0)
    prev = None
    for t in (4.0, 2.0, 1.0, 0.5, 0.25, 0.1):
        gate.set_temperature(t)
        g = gate(logits)
        act = sparsity_stats(g)["mean_active"]
        if prev is not None:
            assert act <= prev + 1e-9, (
                "lowering the temperature must not make the gate denser: "
                "t=%s gave %s active, previous %s" % (t, act, prev))
        prev = act
    assert prev < 12, "the sparsest setting is still fully dense"


def test_mask_forces_exact_zero_on_disabled_coordinates():
    gate = EntmaxGate(alpha=1.5, temperature=1.0)
    logits = torch.randn(10, 6)
    mask = torch.ones(10, 6, dtype=torch.bool)
    mask[:, 2] = False
    g = gate(logits, mask=mask)
    assert float(g[:, 2].abs().max()) == 0.0
    assert (g.sum(-1) - 1).abs().max() < 1e-6


def test_gate_temperature_is_learnable():
    gate = EntmaxGate(alpha=1.5, temperature=1.0, learn_temperature=True)
    g = gate(torch.randn(8, 7) * 3)
    g.sum(-1).mean().backward()
    assert gate.log_temp.grad is not None


def test_entmax_backward_matches_finite_differences():
    """A known-answer test on the *value* of the gradient, not just its support.

    The hand-written backward is a closed form, so every sign/support test can
    pass while the magnitude is wrong by a constant factor -- an errant
    `(alpha - 1)` chain factor scaled it by 0.5 at alpha=1.5 and 0.25 at
    alpha=1.25 and broke nothing else. Central differences pin the value.
    """
    torch.manual_seed(0)
    for alpha in (1.25, 1.5, 1.8, 2.0):
        x = torch.randn(1, 6, dtype=torch.double)
        w = torch.randn(1, 6, dtype=torch.double)

        def f(z):
            return (entmax(z, alpha=alpha, n_iter=200) * w).sum()

        xg = x.clone().requires_grad_(True)
        f(xg).backward()
        num = torch.zeros_like(x)
        h = 1e-6
        for i in range(x.shape[-1]):
            xp, xm = x.clone(), x.clone()
            xp[0, i] += h
            xm[0, i] -= h
            num[0, i] = (f(xp) - f(xm)) / (2 * h)
        assert float((xg.grad - num).abs().max()) < 1e-7, (
            "alpha=%s: analytic gradient disagrees with central differences"
            % alpha)


def test_leak_does_not_change_the_forward_pass():
    """The leak is a backward-only device. Exact zeros must survive it."""
    torch.manual_seed(1)
    x = torch.randn(50, 7) * 3
    exact = entmax(x, leak_beta=None)
    leaky = entmax(x, leak_beta=1.0)
    assert float((exact - leaky).abs().max()) == 0.0
    assert int((exact == 0).sum()) == int((leaky == 0).sum()) > 0


def test_leak_gives_closed_coordinates_a_gradient_and_exact_mode_does_not():
    x = torch.tensor([[3.0, 0.0, -3.0]], requires_grad=True)
    y = entmax(x, leak_beta=None)
    (y * torch.tensor([[1.0, 2.0, 5.0]])).sum().backward()
    assert float(x.grad.abs().max()) == 0.0, (
        "the exact Jacobian is expected to be identically zero here -- that is "
        "the pathology the leak exists to fix")
    x2 = torch.tensor([[3.0, 0.0, -3.0]], requires_grad=True)
    y2 = entmax(x2, leak_beta=1.0)
    assert float(y2[0, 2]) == 0.0, "the forward must still close the gate"
    (y2 * torch.tensor([[1.0, 2.0, 5.0]])).sum().backward()
    assert float(x2.grad[0, 2].abs()) > 0.0, "closed coordinate got no gradient"


def _closed_coord_grad(gap: float, beta: float) -> float:
    """|grad| on a coordinate that entmax has closed, `gap` below the winner.

    At alpha=1.5 with two coordinates the loser closes once gap >= 2.
    """
    x = torch.tensor([[2.0, 2.0 - gap]], requires_grad=True)
    y = entmax(x, leak_beta=beta)
    assert float(y[0, 1]) == 0.0, "gap=%s did not close the coordinate" % gap
    (y * torch.tensor([[0.0, 1.0]])).sum().backward()
    return float(x.grad[0, 1].abs())


def test_leak_is_proximity_weighted_not_a_flat_bribe():
    """A near-miss modality should be recoverable, a decisive rejection less so.

    The leak is not an unconditional bribe to every closed coordinate: it
    decays with how far below the entmax threshold the coordinate sits, and
    `leak_beta` sets how fast. That is what makes it an exploration knob rather
    than a way of quietly turning the gate back into a softmax.
    """
    grads = [_closed_coord_grad(g, 1.0) for g in (2.2, 3.0, 6.0)]
    assert grads[0] > grads[1] > grads[2], (
        "leak must decay with distance below the threshold, got %s" % grads)

    # and beta controls the decay rate: larger beta is more faithful to the
    # exact Jacobian, i.e. it withdraws the leak faster.
    far_soft = _closed_coord_grad(6.0, 1.0)
    far_sharp = _closed_coord_grad(6.0, 10.0)
    assert far_sharp < far_soft / 100, (
        "beta must control how quickly the leak vanishes: %.3e vs %.3e"
        % (far_sharp, far_soft))


def test_leak_beta_is_refused_when_nonpositive():
    for bad in (0.0, -1.0):
        try:
            entmax(torch.randn(3, 4), leak_beta=bad)
        except ValueError as exc:
            assert "leak_beta" in str(exc)
        else:
            raise AssertionError("leak_beta=%r should be refused" % bad)


def test_leak_schedule_is_monotone_and_ends_faithful():
    vals = [leak_schedule(f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)
    assert vals[0] == 1.0 and vals[-1] == 20.0
    assert leak_schedule(-5.0) == 1.0 and leak_schedule(9.0) == 20.0


def test_gate_set_leak_beta_round_trips():
    g = EntmaxGate()
    assert g.leak_beta == 1.0
    g.set_leak_beta(7.5)
    assert g.leak_beta == 7.5
    g.set_leak_beta(None)
    assert g.leak_beta is None
    x = torch.randn(4, 5, requires_grad=True) * 3
    g(x).sum().backward()


def test_topk_gate_is_exact_and_straight_through():
    torch.manual_seed(0)
    x = torch.randn(16, 5)
    for k in (1, 2, 3):
        y = topk_gate(x, k=k)
        assert (y.sum(-1) - 1).abs().max() < 1e-5
        assert int((y > 0).sum(-1).max()) == k
        assert float(y[y == 0].abs().max() if (y == 0).any() else 0.0) == 0.0
    xg = x.clone().requires_grad_(True)
    y = topk_gate(xg, k=1)
    (y * torch.randn_like(y)).sum().backward()
    closed = (y == 0).detach()
    assert float(xg.grad[closed].abs().max()) > 0.0, (
        "the straight-through estimator must reach masked coordinates")


def test_topk_gate_refuses_an_out_of_range_budget():
    for k in (0, 6):
        try:
            topk_gate(torch.randn(3, 5), k=k)
        except ValueError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("k=%d should be refused" % k)


def test_gate_sparsity_loss_tracks_effective_support():
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    two = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    flat = torch.full((1, 4), 0.25)
    assert abs(float(gate_sparsity_loss(one_hot)) - 1.0) < 1e-5
    assert abs(float(gate_sparsity_loss(two)) - 2.0) < 1e-5
    assert abs(float(gate_sparsity_loss(flat)) - 4.0) < 1e-5
    # two-sided form is zero exactly at the requested operating point
    assert float(gate_sparsity_loss(two, target_l0=2.0)) < 1e-8
    assert float(gate_sparsity_loss(flat, target_l0=2.0)) > 1.0
    # and it is differentiable, which L0 itself is not
    g = torch.tensor([[0.6, 0.4, 0.0, 0.0]], requires_grad=True)
    gate_sparsity_loss(g, target_l0=1.0).backward()
    assert float(g.grad.abs().sum()) > 0
