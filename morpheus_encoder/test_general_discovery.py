import numpy as np

from morpheus_encoder.general_discovery import (
    Candidate, evaluate_episode_candidate, evaluate_frozen_candidate, evaluate_structured_nulls,
    evaluate_response_permutation_nulls, propose_candidates, contextual_feature_matrix,
    compress_relation,
)


def test_candidate_that_is_a_linear_function_of_the_baseline_gains_nothing():
    """D3: a candidate that just re-expresses an already-declared nuisance
    baseline column must show ~zero incremental gain once B includes that
    baseline -- otherwise "beats B" would trivially reward rediscovering a
    known covariate."""
    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=(n, 3))
    baseline = (2.0 * x[:, 0] - x[:, 1]).reshape(-1, 1)
    candidate = Candidate((0,), feature_form="linear_subspace", weights=(1.0,))
    # y IS the baseline column (plus tiny noise): a candidate correlated
    # with it should add nothing once the baseline already explains it.
    y = baseline.ravel() + 0.01 * rng.normal(size=n)
    receipt = evaluate_frozen_candidate(candidate, x[:150], y[:150], x[150:], y[150:],
                                        proposal_baseline=baseline[:150], replication_baseline=baseline[150:])
    assert receipt.baseline_width == 1
    assert abs(receipt.replication_gain) < 0.05


def test_order_three_candidate_with_only_pairwise_signal_gains_nothing():
    """D3: B for an order->=3 candidate already contains every pairwise
    product among its constituents, so a candidate whose true structure is
    just a 2-way interaction (plus an inert third variable) must show ~zero
    incremental gain over that richer B."""
    rng = np.random.default_rng(6)
    n = 400
    x = rng.normal(size=(n, 4))
    y = 1.5 * x[:, 0] * x[:, 1] + 0.05 * rng.normal(size=n)
    candidate = Candidate((0, 1, 2), feature_form="product")
    receipt = evaluate_frozen_candidate(candidate, x[:240], y[:240], x[240:], y[240:])
    assert abs(receipt.replication_gain) < 0.05


def _interaction_data(seed, n=240):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = 1.8 * x[:, 0] * x[:, 1] + 0.12 * rng.normal(size=n)
    return x, y


def test_candidate_freeze_hash_is_stable_and_metadata_order_independent():
    first = Candidate((0, 2), weights=(1.0, -0.5), feature_form="linear_subspace",
                      metadata={"b": 2, "a": 1})
    second = Candidate((0, 2), weights=(1.0, -0.5), feature_form="linear_subspace",
                       metadata={"a": 1, "b": 2})
    assert first.candidate_hash == second.candidate_hash
    assert first.candidate_hash == first.candidate_hash


def test_held_out_incremental_gain_exceeds_simpler_constituent_model():
    x, y = _interaction_data(3)
    candidate = Candidate((0, 1), feature_form="product")
    receipt = evaluate_frozen_candidate(candidate, x[:120], y[:120], x[120:], y[120:])
    assert receipt.replication_gain > 0.45
    assert receipt.replication_r2_full > receipt.replication_r2_simple


def test_replication_response_cannot_change_frozen_candidate_or_its_hash():
    x, y = _interaction_data(9)
    candidates = propose_candidates(x[:120], y[:120], max_candidates=12, max_pairs=8)
    candidate = next(c for c in candidates if c.feature_form == "product")
    frozen_hash = candidate.candidate_hash
    altered = -y[120:]
    _ = evaluate_frozen_candidate(candidate, x[:120], y[:120], x[120:], altered)
    assert candidate.candidate_hash == frozen_hash
    assert candidate.variables == tuple(candidate.variables)


def test_structured_null_hook_returns_empirical_tail_probability():
    x, y = _interaction_data(7)
    candidate = Candidate((0, 1), feature_form="product")
    receipt = evaluate_frozen_candidate(candidate, x[:120], y[:120], x[120:], y[120:])

    def response_shift(index, X, response, context):
        return X, np.roll(response, index + 1), context

    null_receipt = evaluate_structured_nulls(receipt, x[:120], y[:120],
                                             null_generator=response_shift, n_null=20)
    assert len(null_receipt.null_gains) == 20
    assert 0.0 < null_receipt.null_p_value <= 1.0
    assert null_receipt.candidate_hash == receipt.candidate_hash


def test_fixed_environment_response_nulls_need_no_repeated_refit():
    x, y = _interaction_data(19)
    receipt = evaluate_frozen_candidate(Candidate((0, 1), feature_form="product"),
                                        x[:120], y[:120], x[120:], y[120:], model="rbf")
    null_receipt = evaluate_response_permutation_nulls(
        receipt, x[:120], y[:120], x[120:], y[120:], n_null=31, seed=3)
    assert len(null_receipt.null_gains) == 31
    assert 0.0 < null_receipt.null_p_value <= 1.0


def test_duck_typed_episode_adapter_accepts_attributes_without_base_class():
    x, y = _interaction_data(11)
    episode_type = type("Episode", (), {})
    proposal, replication = episode_type(), episode_type()
    proposal.X, proposal.y = x[:120], y[:120]
    replication.X, replication.y = x[120:], y[120:]
    receipt = evaluate_episode_candidate(Candidate((0, 1), feature_form="product"), proposal, replication)
    assert receipt.replication_gain > 0.4


def test_contextual_views_retain_observed_and_context_provenance():
    x = np.arange(18, dtype=float).reshape(6, 3)
    features, provenance = contextual_feature_matrix(
        x, neighbours=[[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]],
        time=np.arange(6), unit=np.zeros(6), lags=(1,))
    assert features.shape == (6, 12)
    assert provenance[0]["source"] == "observed"
    assert provenance[3]["source"] == "neighbour_mean"
    assert provenance[6] == {"source": "lag", "lag": 1, "feature": 0}
    assert provenance[9] == {"source": "neighbour_mean_lag", "lag": 1, "feature": 0}


def test_forward_expansion_emits_triplet_candidates_without_relation_labels():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(400, 5))
    y = 2.0 * x[:, 0] * x[:, 1] * x[:, 2] + 0.03 * rng.normal(size=len(x))
    candidates = propose_candidates(x, y, max_candidates=64, max_pairs=16,
                                    max_order=3, max_forward_per_order=16)
    assert any(c.variables == (0, 1, 2) and c.feature_form == "product" for c in candidates)
    assert any(len(c.variables) == 3 for c in candidates)


def test_forward_expansion_respects_candidate_and_order_budgets():
    rng = np.random.default_rng(45)
    x = rng.normal(size=(120, 9))
    y = rng.normal(size=120)
    candidates = propose_candidates(x, y, max_candidates=9, max_pairs=2,
                                    max_subspaces=3, max_order=2,
                                    max_forward_per_order=1)
    assert len(candidates) <= 9
    assert all(len(c.variables) <= 2 for c in candidates)


# ==========================================================================
# E1: `compress_relation` -- interpretability AFTER evidence. Known-answer
# structural tests: a planted step function must compress to a threshold
# form within tolerance of the flexible relation's own held-out gain; a
# smooth (non-monotonic) quadratic must NOT compress to a step, even though
# it may separately compress to a sparse polynomial.
# ==========================================================================

def test_planted_step_compresses_to_threshold_within_tolerance():
    rng = np.random.default_rng(0)
    n = 1200
    x = rng.normal(size=(n, 3))
    y = np.where(x[:, 0] > 0.0, 3.0, -3.0) + 0.02 * rng.normal(size=n)
    candidate = Candidate((0,), feature_form="linear_subspace", weights=(1.0,))
    Xp, yp, Xr, yr = x[:800], y[:800], x[800:], y[800:]

    result = compress_relation(candidate, Xp, yp, Xr, yr, tolerance=0.1, complexity_budget=6)

    assert result["compressed"] is True
    accepted_forms = {f["form"] for f in result["accepted"]}
    assert "threshold_step" in accepted_forms
    threshold_entry = next(f for f in result["forms"] if f["form"] == "threshold_step")
    assert threshold_entry["complexity"] == 2
    assert abs(threshold_entry["detail"]["threshold"]) < 0.5


def test_smooth_nonmonotonic_quadratic_does_not_compress_to_a_step():
    rng = np.random.default_rng(100)
    n = 1200
    x = rng.normal(size=(n, 3))
    y = x[:, 0] ** 2 + 0.05 * rng.normal(size=n)
    candidate = Candidate((0,), feature_form="linear_subspace", weights=(1.0,))
    Xp, yp, Xr, yr = x[:800], y[:800], x[800:], y[800:]

    result = compress_relation(candidate, Xp, yp, Xr, yr, tolerance=0.1, complexity_budget=6)

    accepted_forms = {f["form"] for f in result["accepted"]}
    assert "threshold_step" not in accepted_forms
    # A degree<=2 polynomial legitimately recovers x**2 -- compression is
    # decomposed by form, not vetoed wholesale just because one form fails.
    assert "sparse_polynomial" in accepted_forms


def test_compress_relation_never_fabricates_a_formula_when_nothing_fits():
    """An impossibly tight complexity budget (every form needs >=2
    parameters) must return an explicit `compressed: False` with every
    attempted form on record, never a fabricated formula -- even though the
    underlying relation is a genuine, strongly validated step function."""
    rng = np.random.default_rng(0)
    n = 1200
    x = rng.normal(size=(n, 3))
    y = np.where(x[:, 0] > 0.0, 3.0, -3.0) + 0.02 * rng.normal(size=n)
    candidate = Candidate((0,), feature_form="linear_subspace", weights=(1.0,))
    Xp, yp, Xr, yr = x[:800], y[:800], x[800:], y[800:]

    result = compress_relation(candidate, Xp, yp, Xr, yr, tolerance=0.1, complexity_budget=1)

    assert result["accepted"] == []
    assert result["compressed"] is False
    assert isinstance(result["reason"], str) and result["reason"]
    assert {f["form"] for f in result["forms"]} >= {"threshold_step", "sparse_polynomial", "piecewise_regime"}


def test_forward_expansion_hashes_and_order_are_deterministic():
    rng = np.random.default_rng(46)
    x = rng.normal(size=(180, 6))
    y = x[:, 0] * x[:, 1] * x[:, 2] + rng.normal(scale=0.1, size=180)
    kwargs = dict(max_candidates=40, max_pairs=12, max_order=4,
                  forward_beam_width=8, max_forward_per_order=6)
    first = propose_candidates(x, y, **kwargs)
    second = propose_candidates(x, y, **kwargs)
    assert [c.candidate_hash for c in first] == [c.candidate_hash for c in second]


def _planted(seed=0, n=1500, interaction=True):
    import numpy as np
    from morpheus_encoder.general_discovery import Candidate
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    depth = rng.normal(size=n)
    baseline = np.column_stack((depth, rng.normal(size=(n, 2))))
    y = 0.5 * X[:, 0] + 0.5 * X[:, 1] + 0.8 * depth ** 2 + 0.2 * rng.normal(size=n)
    if interaction:
        y = y + 1.2 * X[:, 0] * X[:, 1]
    cand = Candidate(variables=(0, 1), context_variables=(), weights=(1.0, 1.0), feature_form="product",
                     proposal_family="test", proposal_score=1.0)
    return X, y, baseline, cand


def test_fit_once_null_path_matches_refit_path():
    import numpy as np
    from morpheus_encoder.general_discovery import FittedRelation, evaluate_frozen_candidate
    X, y, b, cand = _planted()
    half = len(y) // 2
    fitted = FittedRelation(cand, X[:half], y[:half], context=None, baseline=b[:half], model="rbf",
                            ridge_alpha=1.0, rbf_features=24)
    rng = np.random.default_rng(1)
    for _ in range(3):
        perm = rng.permutation(len(y) - half)
        yn = y[half:][perm]
        via_fitted = fitted.gain(X[half:], yn, context=None, baseline=b[half:])[0]
        via_refit = evaluate_frozen_candidate(cand, X[:half], y[:half], X[half:], yn, proposal_baseline=b[:half],
                                              replication_baseline=b[half:], model="rbf").replication_gain
        assert abs(via_fitted - via_refit) < 1e-9


def test_interaction_gains_and_baseline_nonlinearity_does_not():
    from morpheus_encoder.general_discovery import evaluate_frozen_candidate
    X, y, b, cand = _planted(interaction=True)
    half = len(y) // 2
    with_int = evaluate_frozen_candidate(cand, X[:half], y[:half], X[half:], y[half:], proposal_baseline=b[:half],
                                         replication_baseline=b[half:], model="rbf").replication_gain
    X2, y2, b2, cand2 = _planted(interaction=False)
    without = evaluate_frozen_candidate(cand2, X2[:half], y2[:half], X2[half:], y2[half:], proposal_baseline=b2[:half],
                                        replication_baseline=b2[half:], model="rbf").replication_gain
    assert with_int > 0.05                      # joint structure beyond additive flexible constituents
    assert without < 0.02                       # never credited (negative = joint map generalises worse than additive, as expected)


def test_single_variable_nonlinearity_is_credited_against_linear_competitor():
    import numpy as np
    from morpheus_encoder.general_discovery import Candidate, evaluate_frozen_candidate
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 3))
    y = np.where(X[:, 0] > 0.5, 1.0, 0.0) + 0.1 * rng.normal(size=1500)   # threshold on one gene
    cand = Candidate(variables=(0,), context_variables=(), weights=(1.0,), feature_form="linear_subspace",
                     proposal_family="test", proposal_score=1.0)
    half = 750
    gain = evaluate_frozen_candidate(cand, X[:half], y[:half], X[half:], y[half:], model="rbf").replication_gain
    assert gain > 0.05
def test_context_proposal_family_reserves_cross_view_candidates():
    """Marginal candidates may not starve the declared cross-view grammar."""
    rng = np.random.default_rng(991)
    x = rng.normal(size=(160, 10))
    context = rng.normal(size=(160, 10))
    # Strong marginals intentionally outrank most interactions in a naive
    # global sort; the fixed context reserve remains response-selected but
    # guarantees that the cross family is actually evaluated.
    y = 4 * x[:, 0] + x[:, 2] * context[:, 3] + .1 * rng.normal(size=160)
    candidates = propose_candidates(x, y, context=context, max_candidates=12,
                                    min_context_candidates=4)
    assert len(candidates) == 12
    assert sum(bool(c.context_variables) for c in candidates) >= 4
