"""The pure-AND construction must be genuinely marginal-free.

This is the benchmark's most load-bearing family. Proposition 2 says a
conjunction whose constituents are independent and zero-mean is invisible to
any method that ranks candidates by marginal association, because

    E[A * (A B)] = E[A^2] E[B] = 0

when B is zero-mean. Everything downstream -- the observability hierarchy, the
claim that linear representations cannot reach this object, the ablation that
removes the relational expert -- rests on the generator actually having that
property rather than on the docstring saying it does.

So it is asserted here, and asserted structurally: not "the correlation is
small" but "a marginal ranker cannot separate the conjunction's constituents
from pure noise, while the same ranker does find the additive decoys planted in
the same world." A world where the second half failed would be a world where
the test cannot fail, and a test that cannot fail certifies nothing.
"""
import numpy as np
import pytest

from dbench.environments import build_world
from dbench.families import make_pure_and
from dbench.util import corr

SEEDS = (0, 1, 2, 3, 7, 11, 23)


def _marg(y, x):
    return abs(float(corr(np.asarray(y, float), np.asarray(x, float))))


def _null_band(n, z=4.0):
    """Sampling band for a zero population correlation at n observations."""
    return z / np.sqrt(n)


# --------------------------------------------------------------------------
# 1. the algebraic identity, at the population level
# --------------------------------------------------------------------------
def test_identity_holds_for_independent_zero_mean_constituents():
    """E[a * a b] = E[a^2] E[b], which is zero exactly when b is zero-mean.

    Checked directly on the generator's own assumption rather than on a world,
    so a failure here localises to the algebra and not to the plumbing.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    prod = a * b
    assert abs(float(np.mean(a * prod))) < 0.02
    assert abs(float(np.mean(b * prod))) < 0.02
    # and the identity is not a coincidence of symmetry: shifting b off zero
    # mean restores the marginal, so the zero really is carried by E[b] = 0.
    b_shift = b + 1.0
    assert abs(float(np.mean(a * (a * b_shift)))) > 0.5


# --------------------------------------------------------------------------
# 2. the world: constituents carry no marginal signal
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_and_constituents_are_marginally_uncorrelated_with_target(seed):
    w = make_pure_and(seed=seed, n=8000)
    band = _null_band(w.n)
    names = [h.operands for h in w.truth.true_objects if h.operator == "AND"]
    assert names, "the world must plant at least one conjunction"
    for operands in names:
        for nm in operands:
            r = _marg(w.target, w.latents[nm])
            assert r < band, (
                "constituent %s of a pure AND carries marginal signal "
                "|r| = %.4f against a sampling band of %.4f; the family is not "
                "marginal-free and Prop. 2 does not apply to it" % (nm, r, band))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_conjunction_itself_is_strongly_associated(seed):
    """Otherwise the family is marginal-free for the trivial reason that
    nothing is there at all."""
    w = make_pure_and(seed=seed, n=8000)
    for h in w.truth.true_objects:
        if h.operator != "AND":
            continue
        a, b = (np.asarray(w.latents[o], float) for o in h.operands)
        r = _marg(w.target, a * b)
        assert r > 10 * _null_band(w.n), (
            "the product %s carries |r| = %.4f, which is not a detectable "
            "object" % (h.operands, r))


@pytest.mark.parametrize("seed", SEEDS)
def test_decoy_additive_terms_do_carry_marginal_signal(seed):
    """The falsifiability half: the same world contains additive objects a
    marginal ranker must find, so 'nothing is marginally visible' is a claim
    about the conjunction and not about the world."""
    w = make_pure_and(seed=seed, n=8000)
    band = _null_band(w.n)
    adds = [h.operands[0] for h in w.truth.true_objects if h.operator == "ADD"]
    assert adds, "pure_and must plant additive decoys"
    for nm in adds:
        assert _marg(w.target, w.latents[nm]) > 5 * band


# --------------------------------------------------------------------------
# 3. the structural claim: a marginal ranker cannot separate constituents
#    from noise, and this is what a linear method is restricted to
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_marginal_ranker_puts_the_additive_decoys_on_top(seed):
    """Rank every latent by |corr(latent, Y)| and read off the top of the list.

    The assertion is on the ranking -- the structure a screening method acts
    on -- not on a correlation value. A marginal ranker should recover exactly
    the additive objects and nothing else.
    """
    w = make_pure_and(seed=seed, n=8000)
    add_ops = {h.operands[0] for h in w.truth.true_objects
               if h.operator == "ADD"}
    scores = {k: _marg(w.target, v) for k, v in w.latents.items()}
    order = sorted(scores, key=lambda k: -scores[k])
    assert set(order[:len(add_ops)]) == add_ops, (
        "a marginal ranker should recover exactly the additive objects first; "
        "got %s" % sorted(order[:len(add_ops)]))


def test_and_constituents_are_rank_exchangeable_with_pure_noise():
    """The constituents must be *indistinguishable* from noise to a marginal
    ranker, not merely small.

    Per seed this cannot be asserted -- with four distractors a null variable
    tops the pool one time in five by chance, and a test that demanded
    otherwise would be measuring luck. What is assertable, and what the claim
    actually needs, is exchangeability: pooled over seeds, a constituent's rank
    inside {itself} u {distractors} is uniform. Both tails are checked, so a
    generator that made constituents systematically *quieter* than noise --
    which would be its own kind of tell -- fails too.
    """
    ranks, n_pool = [], None
    for seed in range(40):
        w = make_pure_and(seed=seed, n=8000)
        scores = {k: _marg(w.target, v) for k, v in w.latents.items()}
        distract = [k for k in w.latents if k.startswith("D")]
        n_pool = len(distract) + 1
        for h in w.truth.true_objects:
            if h.operator != "AND":
                continue
            for nm in h.operands:
                pool = sorted([nm] + distract, key=lambda k: -scores[k])
                ranks.append(pool.index(nm) + 1)          # 1 = most correlated
    ranks = np.asarray(ranks, dtype=float)
    expected = (n_pool + 1) / 2.0
    sd = np.sqrt((n_pool ** 2 - 1) / 12.0 / len(ranks))
    z = (ranks.mean() - expected) / sd
    assert abs(z) < 3.5, (
        "mean marginal rank of an AND constituent among %d noise variables is "
        "%.3f against %.3f expected under exchangeability (z = %.2f, %d "
        "observations): the constituents are not marginally invisible"
        % (n_pool - 1, ranks.mean(), expected, z, len(ranks)))
    top_rate = float((ranks == 1).mean())
    assert 0.5 / n_pool < top_rate < 2.0 / n_pool, (
        "an AND constituent is the most-correlated variable in the pool "
        "%.1f%% of the time against %.1f%% expected" % (100 * top_rate,
                                                        100.0 / n_pool))


@pytest.mark.parametrize("seed", (0, 5, 9))
def test_linear_model_on_all_latents_gains_nothing_from_constituents(seed):
    """A least-squares fit on every latent -- the strongest purely linear use
    of the oracle variables -- explains no more of Y than the additive decoys
    alone do."""
    w = make_pure_and(seed=seed, n=8000)
    add_ops = sorted(h.operands[0] for h in w.truth.true_objects
                     if h.operator == "ADD")
    y = np.asarray(w.target, float)

    def r2(cols):
        X = np.column_stack([np.ones(len(y))] + list(cols))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        return 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())

    r2_adds = r2([w.latents[k] for k in add_ops])
    r2_all = r2([np.asarray(v, float) for v in w.latents.values()])
    # every extra column can only help in-sample; the point is that it barely
    # does, and specifically by less than the additive terms contributed.
    assert r2_all - r2_adds < 0.25 * r2_adds, (
        "linear fit on all latents R2 = %.4f vs additive-only %.4f: the AND "
        "constituents are contributing linearly, which they must not"
        % (r2_all, r2_adds))


# --------------------------------------------------------------------------
# 4. the property must survive the machinery the benchmark wraps around it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("role", ("proposal", "replication", "null"))
def test_marginal_freedom_survives_every_environment_role(role):
    w = build_world("pure_and", seed=4, role=role, snr="high")
    band = _null_band(w.n)
    for h in w.truth.true_objects:
        if h.operator != "AND":
            continue
        for nm in h.operands:
            assert _marg(w.target, w.latents[nm]) < band


@pytest.mark.parametrize("snr", ("high", "med", "low"))
def test_marginal_freedom_is_not_an_artefact_of_the_snr_preset(snr):
    w = build_world("pure_and", seed=6, snr=snr)
    band = _null_band(w.n)
    for h in w.truth.true_objects:
        if h.operator != "AND":
            continue
        for nm in h.operands:
            assert _marg(w.target, w.latents[nm]) < band


def test_planted_artifacts_do_not_reintroduce_a_marginal():
    """Artifact planting rewrites Y. It must not hand the conjunction's
    constituents a marginal association through the back door."""
    from dbench.artifacts import ALL_CLASSES
    w = make_pure_and(seed=3, n=8000, artifacts=ALL_CLASSES)
    band = _null_band(w.n)
    for h in w.truth.true_objects:
        if h.operator != "AND":
            continue
        for nm in h.operands:
            r = _marg(w.target, w.latents[nm])
            assert r < band, (
                "planting %s gave AND constituent %s a marginal |r| = %.4f"
                % (sorted(ALL_CLASSES), nm, r))


def test_observed_modality_does_not_leak_the_conjunction_linearly():
    """The observed matrix is a linear mixing of the latents, so it must not
    make the product linearly available either."""
    w = make_pure_and(seed=8, n=8000)
    X = w.modalities["X"]
    y = np.asarray(w.target, float)
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    res = y - Xc @ beta
    r2_lin = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    # the true product terms explain a large share; a linear read of the
    # modality must not get there.
    prods = []
    for h in w.truth.true_objects:
        if h.operator == "AND":
            a, b = (np.asarray(w.latents[o], float) for o in h.operands)
            prods.append(a * b)
    Xp = np.column_stack([np.ones(len(y))] + prods)
    bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    rp = y - Xp @ bp
    r2_prod = 1.0 - float(rp @ rp) / float(((y - y.mean()) ** 2).sum())
    assert r2_prod > 0.2
    assert r2_lin < r2_prod, (
        "a linear model on the observed modality reaches R2 = %.3f against "
        "R2 = %.3f for the true products; the mixing is leaking the "
        "conjunction" % (r2_lin, r2_prod))
