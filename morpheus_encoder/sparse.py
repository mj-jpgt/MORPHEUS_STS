"""alpha-entmax, implemented here rather than pulled in as a dependency.

Why implement it
----------------
The whole point of the modality gate `g_km = entmax(R_rho(q_k, m, E))` and the
expert gate `a_kr = entmax(R_psi(u_k))` is that a slot may attend to *no* RNA
at all -- an exact zero, not a small number that a downstream sum still mixes
in. Softmax cannot do that. `entmax_alpha` for alpha > 1 can, because the
solution has the form

    p_i = [ (alpha - 1) x_i - tau ]_+ ^ (1 / (alpha - 1))

and the hinge is a real hinge. alpha = 2 is sparsemax; alpha = 1.5 is the usual
entmax; alpha -> 1 recovers softmax (and loses the zeros, which is why alpha is
clamped away from 1 here).

tau is found by bisection on the normalisation g(tau) = sum_i p_i(tau) - 1,
which is continuous and strictly decreasing on the bracket used below, so
bisection converges monotonically. The backward pass is the closed-form entmax
Jacobian-vector product rather than autograd through the bisection loop: with
s_i = p_i^(2 - alpha) and S = support,

    (J dY)_i = s_i * (dY_i - sum_j s_j dY_j / sum_j s_j)

which is exactly zero off the support -- gradients flow only through the gates
that are actually open, which is the property the routing tests assert.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _bisect(Z: torch.Tensor, alpha: float, dim: int, n_iter: int) -> torch.Tensor:
    """Return tau such that sum_i [Z_i - tau]_+^(1/(alpha-1)) == 1.

    Z is the pre-scaled logit tensor (alpha - 1) * X.
    """
    p = 1.0 / (alpha - 1.0)
    d = Z.shape[dim]
    zmax = Z.max(dim=dim, keepdim=True).values
    # At tau = zmax - 1 the largest term alone is 1, so g >= 0.
    tau_lo = zmax - 1.0
    # At tau = zmax - d^{-(alpha-1)} every term is at most d^{-1} ... g <= 0.
    tau_hi = zmax - float(d) ** (-(alpha - 1.0))
    for _ in range(n_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        g = torch.clamp(Z - tau, min=0.0).pow(p).sum(dim=dim, keepdim=True) - 1.0
        # g decreasing in tau: g > 0 => tau too small.
        too_small = g > 0
        tau_lo = torch.where(too_small, tau, tau_lo)
        tau_hi = torch.where(too_small, tau_hi, tau)
    return 0.5 * (tau_lo + tau_hi)


class _EntmaxBisect(torch.autograd.Function):
    """Forward: exact alpha-entmax. Backward: Jacobian of a beta-smoothed entmax.

    `leak_beta=None` reproduces the exact entmax Jacobian, which is identically
    zero off the support. That exactness is a measured liability, not a virtue:
    a modality gate that reaches exact zero then receives no gradient ever
    again, so its logit cannot move and the gate cannot reopen. Driving a
    single logit vector towards the target [0, 0, 1] from a start of
    [3, 0, -3] leaves the loss at 2.0 after 300 Adam steps -- the closed
    coordinate is frozen, and which modalities an object uses ends up decided
    by initialisation rather than learned.

    A finite `leak_beta` replaces the hinge [r]_+ by softplus(beta*r)/beta in
    the *backward pass only*. Since the entmax Jacobian

        J = diag(s) - s s^T / sum(s),      s_i = [r_i]_+^{(2-alpha)/(alpha-1)}

    is symmetric PSD for any nonnegative s, substituting the smoothed s keeps
    it a well-formed Jacobian -- specifically it is the exact Jacobian of the
    beta-smoothed problem, whose solution converges to this one as beta -> inf.
    The leak is therefore *proximity-aware*: it decays like exp(beta * r), so a
    modality that only just lost the competition can reopen while one that was
    decisively rejected gets essentially nothing. The forward pass is bitwise
    the hard entmax either way, so exact zeros survive.
    """

    @staticmethod
    def forward(ctx, X, alpha, dim, n_iter, leak_beta):
        Z = (alpha - 1.0) * X
        tau = _bisect(Z, alpha, dim, n_iter)
        R = Z - tau
        Y = torch.clamp(R, min=0.0).pow(1.0 / (alpha - 1.0))
        # Renormalise against bisection residual. This is a rescale of an
        # already-correct support, so it moves no coordinate off zero.
        Y = Y / Y.sum(dim=dim, keepdim=True).clamp_min(1e-30)
        ctx.save_for_backward(Y, R)
        ctx.alpha, ctx.dim, ctx.leak_beta = alpha, dim, leak_beta
        return Y

    @staticmethod
    def backward(ctx, dY):
        Y, R = ctx.saved_tensors
        alpha, dim, beta = ctx.alpha, ctx.dim, ctx.leak_beta
        pw = (2.0 - alpha) / (alpha - 1.0)
        if beta is None:
            # s must be the *support-masked* power. At alpha == 2, Y ** 0 == 1
            # even where Y == 0, which silently puts every coordinate back into
            # the support and produced a dense (and wrong) Jacobian; the mask is
            # what makes the gradient vanish off-support for every alpha.
            s = torch.where(Y > 0, Y.pow(2.0 - alpha), torch.zeros_like(Y))
        else:
            # softplus(beta*r)/beta -> [r]_+ as beta -> inf, and is strictly
            # positive below the threshold, which is the whole point.
            s = torch.nn.functional.softplus(beta * R).div(beta).clamp_min(1e-30).pow(pw)
        dX = dY * s
        q = dX.sum(dim=dim, keepdim=True) / s.sum(dim=dim, keepdim=True).clamp_min(1e-30)
        # No (alpha-1) chain factor here: the standard entmax Jacobian is
        # already stated with respect to x, not to Z = (alpha-1)x. A
        # finite-difference check (test_sparse.py) pins this down -- adding
        # the factor scales the gradient by (alpha-1) and still passes every
        # sign/support test, because it is a positive rescale.
        dX = dX - q * s
        return dX, None, None, None, None


def entmax(x: torch.Tensor, alpha: float = 1.5, dim: int = -1,
           n_iter: int = 50, leak_beta: float | None = None) -> torch.Tensor:
    """alpha-entmax. Returns a simplex vector with exact zeros for alpha > 1.

    `leak_beta` finite gives closed coordinates a proximity-weighted gradient
    so they can reopen; see `_EntmaxBisect`. It does not change the forward.
    """
    if alpha <= 1.0 + 1e-6:
        raise ValueError(
            "alpha must be > 1 for entmax to produce exact zeros; alpha=%r "
            "is softmax, which never does. Use torch.softmax explicitly if "
            "that is what you want." % (alpha,))
    if leak_beta is not None and float(leak_beta) <= 0:
        raise ValueError("leak_beta must be positive or None, got %r" % (leak_beta,))
    return _EntmaxBisect.apply(x, float(alpha), dim, int(n_iter),
                               None if leak_beta is None else float(leak_beta))


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return entmax(x, alpha=1.5, dim=dim)


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """alpha = 2. Exact via sorting rather than bisection."""
    x = x.transpose(dim, -1)
    srt, _ = torch.sort(x, dim=-1, descending=True)
    cum = srt.cumsum(-1)
    k = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    support = (1.0 + k * srt) > cum
    ksz = support.to(x.dtype).sum(-1, keepdim=True)
    tau = (cum.gather(-1, ksz.long() - 1) - 1.0) / ksz
    return torch.clamp(x - tau, min=0.0).transpose(dim, -1)


class EntmaxGate(nn.Module):
    """entmax with a learned temperature, so sparsity is *controllable*.

    Gate sparsity is not a fixed property of the architecture: a smaller
    temperature concentrates the logits and drives more coordinates to exact
    zero. `log_temp` is a parameter so that later training stages (step 6
    onwards) can move the operating point of the routing itself, and
    `set_temperature` lets a test sweep it deterministically.
    """

    def __init__(self, alpha: float = 1.5, temperature: float = 1.0,
                 learn_temperature: bool = True, n_iter: int = 50,
                 leak_beta: float | None = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        # Finite by default: an exactly-zero gate with an exactly-zero gradient
        # is a gate that can never reopen, so routing would be fixed by
        # initialisation. Set None for a strictly exact Jacobian at eval time.
        #
        # Why 1.0. The test that discriminates is an *adversarial* one: start
        # every object hard-committed to modality 0 (24 of 36 gates already at
        # exact zero), then require 8 of 12 objects to move to a modality whose
        # gate is closed. Hits out of 12, 3 seeds, judged afterwards under the
        # exact Jacobian:
        #
        #   leak_beta   None   0.5   1.0   2.0   5.0   10.0
        #   hits         4.0  12.0  12.0  12.0  12.0    4.0
        #
        # 4/12 is exactly the set of objects that already happened to want
        # modality 0: with the exact Jacobian not one closed gate ever reopened.
        # beta=10 fails for the opposite reason -- the leak decays so fast that
        # a decisively closed gate gets nothing. The usable band is about
        # [0.5, 5].
        #
        # The cost is gradient fidelity on the support: deviation from the
        # exact Jacobian is 123% at beta=1, 50% at beta=2, 12% at beta=5 and
        # 3.2% at beta=10. So beta trades a faithful gradient against the
        # ability to revise a decision, exactly like an exploration
        # temperature, and like one it should be annealed -- `leak_schedule`
        # is that schedule and `set_leak_beta` the hook. The default sits at
        # the exploratory end because the failure it prevents is silent:
        # without the leak, which modalities an object uses is settled by
        # whatever closed first.
        self.leak_beta = None if leak_beta is None else float(leak_beta)
        t = torch.tensor(float(temperature)).log()
        self.log_temp = nn.Parameter(t) if learn_temperature else nn.Parameter(t, requires_grad=False)

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.log_temp.copy_(torch.tensor(float(t)).log())

    @property
    def temperature(self) -> float:
        return float(self.log_temp.exp())

    def set_leak_beta(self, beta: float | None):
        """Move the backward-pass smoothing. `None` = exact Jacobian."""
        if beta is not None and float(beta) <= 0:
            raise ValueError("leak_beta must be positive or None, got %r" % (beta,))
        self.leak_beta = None if beta is None else float(beta)

    def forward(self, logits: torch.Tensor, dim: int = -1,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        z = logits / self.log_temp.exp().clamp_min(1e-4)
        if mask is not None:
            # Disabled coordinates get -inf so entmax assigns them exact zero
            # without their logits participating in tau at all.
            z = z.masked_fill(~mask, float("-inf"))
            # -inf * 0 is nan inside the bisection; use a large finite floor.
            z = torch.nan_to_num(z, neginf=-1e9)
        return entmax(z, alpha=self.alpha, dim=dim, n_iter=self.n_iter,
                      leak_beta=self.leak_beta)


@torch.no_grad()
def sparsity_stats(g: torch.Tensor, dim: int = -1, tol: float = 1e-3) -> dict:
    """Measured, not asserted, and every field here can return more than one value.

    Under no_grad: these are diagnostics that end up in a JSON receipt, and a
    reported statistic must never be a node someone can accidentally
    backpropagate through.

    **`max_offsupport_mass` was a dead statistic and this is the repair.** It
    used to compute `zero = (g.abs() <= tol)` at `tol = 0.0` and then take
    `max |g|` over that set -- the maximum of a quantity over the set on which
    it is at most zero -- so it could only ever return 0.0, on softmax, on
    uniform noise, on all-ones, on a constant, on a genuinely sparse tensor and
    on a uniformly-1e-9 tensor alike (`step3/AUDIT_FINDINGS.md` section 3,
    `PREDECLARATION_STEP4B.md` section 9 A3 F1). It is redefined as

        max { |g_i| : 0 < |g_i| <= tol }        (0.0 if that set is empty)

    the largest magnitude that is **nonzero but small enough that calling it
    zero would be a lie**. That number *can* fail: it is 0.0 for a gate whose
    zeros are exact and whose live coordinates are all above `tol`, and it is
    ~1e-3 for a softmax, ~1e-9 for the uniformly-1e-9 tensor the old statistic
    reported as 0.0. `frac_subtol_nonzero` is its support: how much of the
    tensor sits in that band.

    `tol` is the magnitude below which a nonzero coordinate is called
    off-support. It is **not** used by `frac_exact_zero`, `mean_active`,
    `min_active`, `max_active` or `l0_over_n`, all of which count **exact**
    zeros at any `tol` and therefore return exactly what they returned before
    this repair -- which is why changing the default from 0.0 to 1e-3 moves no
    number that any prior receipt or claim depends on.
    """
    g = g.detach()
    absg = g.abs()
    exact_zero = (g == 0)
    subtol = (absg > 0) & (absg <= tol)
    n = g.shape[dim]
    active = (~exact_zero).to(g.dtype).sum(dim=dim)
    return {
        "n_coords": int(n),
        "offsupport_tol": float(tol),
        "frac_exact_zero": float(exact_zero.to(g.dtype).mean()),
        "mean_active": float(active.mean()),
        "min_active": float(active.min()),
        "max_active": float(active.max()),
        "max_offsupport_mass": float(absg.masked_fill(~subtol, 0.0).max()),
        "frac_subtol_nonzero": float(subtol.to(g.dtype).mean()),
        "l0_over_n": float(active.mean() / n),
    }


# ---------------------------------------------------------------------------
# alternatives measured against entmax in the router study; kept because the
# comparison is part of the record and because a hard budget is occasionally
# the thing you want (an ablation that forces "exactly one modality").
# ---------------------------------------------------------------------------

class _TopKST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, k, dim):
        idx = X.topk(k, dim=dim).indices
        mask = torch.zeros_like(X, dtype=torch.bool).scatter_(dim, idx, True)
        Z = X.masked_fill(~mask, float("-inf"))
        Y = torch.softmax(Z, dim=dim)
        ctx.save_for_backward(torch.softmax(X, dim=dim))
        ctx.dim = dim
        return Y

    @staticmethod
    def backward(ctx, dY):
        (P,) = ctx.saved_tensors
        dim = ctx.dim
        # Straight-through: pretend the forward was the dense softmax, whose
        # Jacobian is dense, so masked-out coordinates still get gradient.
        dX = P * (dY - (dY * P).sum(dim=dim, keepdim=True))
        return dX, None, None


def topk_gate(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Hard top-k over modalities with a straight-through softmax gradient.

    Exact zeros are guaranteed by construction rather than earned, which is
    both the appeal and the objection: `k` is a global budget the objects do
    not choose, i.e. exactly the DISCO-style single knob this encoder is meant
    to replace. Useful as an ablation and as a hard cap, not as the default.
    """
    if not 1 <= k <= x.shape[dim]:
        raise ValueError("k=%d out of range for %d coordinates" % (k, x.shape[dim]))
    return _TopKST.apply(x, int(k), dim)


def gate_sparsity_loss(g: torch.Tensor, target_l0: float | None = None,
                       dim: int = -1) -> torch.Tensor:
    """Differentiable pressure on how many modalities an object consumes.

    L0 itself has no gradient, so the surrogate is the simplex's Gini-style
    mass concentration: sum_i g_i^2 is 1 when one gate is open and 1/M when all
    are equally open, so its reciprocal is a smooth stand-in for the number of
    active coordinates. With `target_l0` this is a two-sided penalty (an
    operating point), without it a one-sided push towards sparsity.

    Note this is pressure on the gate *values*, so it acts only through
    coordinates already on the support; it complements the entmax leak rather
    than substituting for it.
    """
    eff = 1.0 / g.pow(2).sum(dim=dim).clamp_min(1e-12)     # ~ number active
    if target_l0 is None:
        return eff.mean()
    return (eff - float(target_l0)).pow(2).mean()


def leak_schedule(frac: float, beta0: float = 1.0, beta1: float = 20.0) -> float:
    """Geometric leak anneal, `frac` in [0, 1] over the training run.

    Measured (see `EntmaxGate`): fixed beta=1 recovers an assigned modality
    signature on every seed but distorts the on-support gradient by ~123%;
    fixed beta=10 is faithful to ~4% but recovers nothing above chance. The
    geometric anneal gets both -- full recovery, and a final Jacobian close to
    exact. Apply with `gate.set_leak_beta(leak_schedule(step / total))`, and
    set `None` for evaluation.
    """
    f = min(max(float(frac), 0.0), 1.0)
    return float(beta0) * (float(beta1) / float(beta0)) ** f
