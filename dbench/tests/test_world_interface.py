"""The world interface: the contract the Hypothesis Compiler and the encoder read.

Every assertion here is on structure -- which fields exist, what they contain,
which names line up -- rather than on a summary statistic, because the interface
is what the two parallel builds are coding against. A world that quietly stopped
exposing `interventions` on a mechanism family, or that renamed a latent between
the proposal and replication environments, would break them silently and would
show up in no accuracy number.
"""
import re

import numpy as np
import pytest

from dbench.environments import ALL_WORLDS, build_world, make_episode
from dbench.families import FAMILIES, FAMILY_MODE
from dbench.world import (DISCOVERY_MODES, DOMAINS, OPERATORS, EnvMeta,
                          Episode, Hypothesis, Interventions, Structure,
                          TruthSet, World, WorldSpec)

CHEAP = ("additive_factors", "pure_and", "xor", "threshold", "hill",
         "context_gated", "manifold", "spatial_interaction", "temporal_lag",
         "regime_switch", "graph_motif", "latent_confound",
         "intervention_causal", "hidden_symmetry", "sparse_ode", "anomaly",
         "mixture", "experimental_design")


# --------------------------------------------------------------------------
# the seven readable fields, and the one that is not readable
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", CHEAP)
def test_world_exposes_the_seven_proposal_time_fields(family):
    w = build_world(family, 0)
    assert isinstance(w.latents, dict) and w.latents
    assert isinstance(w.modalities, dict) and w.modalities
    assert isinstance(w.env, EnvMeta)
    assert isinstance(w.nuisance, dict)
    assert isinstance(w.structure, Structure)
    assert w.interventions is None or isinstance(w.interventions, Interventions)
    assert w.target is None or len(w.target) == w.n
    assert isinstance(w.truth, TruthSet)


@pytest.mark.parametrize("family", CHEAP)
def test_every_array_agrees_on_the_row_count(family):
    w = build_world(family, 0)
    n = w.n
    for k, v in w.latents.items():
        assert len(np.asarray(v)) == n, "latent %s has %d rows, world has %d" % (
            k, len(np.asarray(v)), n)
    for k, v in w.modalities.items():
        assert np.asarray(v).shape[0] == n
    for k, v in w.nuisance.items():
        assert len(np.asarray(v)) == n
    for k, v in w.env.labels.items():
        assert len(np.asarray(v)) == n
    st = w.structure
    for f in ("coords", "time", "unit"):
        if getattr(st, f) is not None:
            assert len(np.asarray(getattr(st, f))) == n
    if st.neighbors is not None:
        assert len(st.neighbors) == n
    if w.interventions is not None:
        assert len(np.asarray(w.interventions.assignment)) == n


def test_blinding_seals_the_truth_set():
    w = build_world("pure_and", 0)
    b = w.blind()
    assert b.latents is w.latents          # data is still readable
    with pytest.raises(PermissionError):
        _ = b.truth.objects
    with pytest.raises(PermissionError):
        _ = b.truth.true_objects
    # and the unblinded handle the harness holds is untouched
    assert len(w.truth.true_objects) > 0


def test_blinding_an_episode_seals_every_world_in_it():
    ep = make_episode("threshold", seed=0, n_replication=1)
    b = ep.blind()
    for w in (b.D_P,) + tuple(b.D_R) + tuple(b.D_N):
        with pytest.raises(PermissionError):
            _ = w.truth.objects


# --------------------------------------------------------------------------
# typed objects
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", CHEAP)
def test_every_ground_truth_object_is_typed_from_the_shared_vocabulary(family):
    w = build_world(family, 0)
    assert w.truth.objects, "a world with no typed objects measures nothing"
    for h in w.truth.objects:
        assert isinstance(h, Hypothesis)
        assert h.operator in OPERATORS, h.operator
        assert h.mode in DISCOVERY_MODES, h.mode
        assert h.operands, "object %s names no operands" % h.id
        assert isinstance(h.true, bool)
        if not h.true:
            assert h.artifact_class is not None


def _resolvable(name, known):
    """Is `name` a variable the world has, or a library term over such?

    `ODE` objects name terms of a governing-equation library -- `x^3`, `xy` --
    rather than bare latents, because the term *is* the object: a law
    discoverer that reported the variable `x` when the law contains `x^3` has
    not recovered it. The term still has to resolve to variables the world
    holds, which is what is checked.
    """
    if name in known:
        return True
    syms = [p for p in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", name)]
    return bool(syms) and all(s in known for s in syms)


@pytest.mark.parametrize("family", CHEAP)
def test_object_operands_name_variables_the_world_actually_has(family):
    w = build_world(family, 0)
    known = set(w.latents) | set(w.nuisance) | set(w.env.labels)
    for h in w.truth.objects:
        for o in h.operands:
            assert _resolvable(o, known), (
                "object %s names operand %r which resolves to no variable in "
                "the latents, the nuisance covariates or the environment "
                "labels" % (h.id, o))


def test_ode_operands_are_library_terms_over_real_state_variables():
    """The relaxation above must not become a hole: an ODE operand has to be a
    polynomial term whose symbols are genuine state latents."""
    w = build_world("sparse_ode", 0)
    ode = [h for h in w.truth.true_objects if h.operator == "ODE"]
    assert ode
    for h in ode:
        for o in h.operands:
            syms = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", o))
            assert syms and syms <= set(w.latents), (h.id, o)
            assert h.target in w.latents


def test_hypothesis_key_is_operand_order_insensitive():
    a = Hypothesis(id="a", mode="relational", operator="AND",
                   operands=("A", "B"), target="Y")
    b = Hypothesis(id="b", mode="relational", operator="AND",
                   operands=("B", "A"), target="Y")
    c = Hypothesis(id="c", mode="relational", operator="AND",
                   operands=("A", "C"), target="Y")
    assert a.key() == b.key()
    assert a.key() != c.key()


def test_hypothesis_key_separates_lag_and_context():
    base = dict(mode="relational", operator="LAG", operands=("B",), target="Y")
    assert (Hypothesis(id="1", lag=2, **base).key()
            != Hypothesis(id="2", lag=3, **base).key())
    assert (Hypothesis(id="3", context={"g": 1}, **base).key()
            != Hypothesis(id="4", context={"g": 0}, **base).key())


@pytest.mark.parametrize("family", CHEAP)
def test_family_mode_matches_the_mode_of_its_own_objects(family):
    w = build_world(family, 0)
    modes = {h.mode for h in w.truth.true_objects}
    assert FAMILY_MODE[family] in modes, (
        "%s is registered as isolating the %s mode but plants only %s objects"
        % (family, FAMILY_MODE[family], sorted(modes)))


# --------------------------------------------------------------------------
# structure, environments, interventions
# --------------------------------------------------------------------------
def test_mechanism_families_carry_interventions():
    w = build_world("intervention_causal", 0)
    assert w.interventions is not None
    assert len(w.interventions.regimes) >= 2
    for r in w.interventions.regimes:
        assert w.interventions.mask(r).sum() > 0


def test_structure_kind_agrees_with_the_fields_that_are_filled():
    checks = {"spatial_interaction": ("coords", "neighbors"),
              "temporal_lag": ("time",),
              "regime_switch": ("time",),
              "graph_motif": ("graph_id",)}
    for fam, fields in checks.items():
        st = build_world(fam, 0).structure
        for f in fields:
            assert getattr(st, f) is not None, "%s is missing %s" % (fam, f)


def test_neighbourhood_aggregate_matches_a_direct_computation():
    w = build_world("spatial_interaction", 0)
    x = np.asarray(w.latents["B"], dtype=float)
    agg = w.structure.aggregate(x, "mean")
    for i in (0, 7, 101, w.n - 1):
        nb = w.structure.neighbors[i]
        assert np.isclose(agg[i], x[nb].mean() if len(nb) else 0.0)


def test_majority_aggregate_is_a_vote_not_a_mean():
    w = build_world("spatial_interaction", 0)
    x = (np.asarray(w.latents["B"], dtype=float) > 0).astype(float)
    maj = w.structure.aggregate(x, "majority")
    assert set(np.unique(maj)) <= {0.0, 1.0}
    for i in (3, 55, 400):
        nb = w.structure.neighbors[i]
        assert maj[i] == float(x[nb].mean() > 0.5)


@pytest.mark.parametrize("family", CHEAP)
def test_domain_is_from_the_declared_vocabulary(family):
    assert build_world(family, 0).domain in DOMAINS


# --------------------------------------------------------------------------
# derived latents: declared, and genuinely derived
# --------------------------------------------------------------------------
def test_derived_latents_are_declared_where_they_exist():
    """A latent that is an operator already applied at the true parameter must
    be declared, or a specialist scan can pass by lookup instead of search."""
    expect = {"spatial_interaction": {"aggB"},
              "temporal_lag": {"B_lag3"},
              "boss1_cond_thresh_space_lag": {"aggB", "aggB_lag", "thrA"},
              "boss2_xor_space_regime": {"majB"},
              "boss3_hill_space_cause": {"aggD", "hillD"},
              "boss5_design_and_threshold": {"thrA"}}
    for fam, names in expect.items():
        w = build_world(fam, 0)
        declared = set(w.meta.get("derived_latents", ()) or ())
        assert names <= declared, (
            "%s exposes %s without declaring them derived" % (
                fam, sorted(names - declared)))
        assert names <= set(w.latents)


def test_declared_derived_latents_really_are_functions_of_other_latents():
    """The declaration must not be decorative: `aggB` must equal the
    neighbourhood mean of `B`, and `B_lag3` must equal `B` shifted by three."""
    w = build_world("spatial_interaction", 0)
    assert np.allclose(w.latents["aggB"],
                       w.structure.aggregate(w.latents["B"], "mean"))
    w = build_world("temporal_lag", 0)
    B = np.asarray(w.latents["B"], dtype=float)
    assert np.allclose(w.latents["B_lag3"][3:], B[:-3])


# --------------------------------------------------------------------------
# episodes and replication
# --------------------------------------------------------------------------
def test_episode_replication_is_independent_and_carries_the_same_objects():
    ep = make_episode("pure_and", seed=0, n_replication=2)
    assert isinstance(ep, Episode)
    assert len(ep.D_R) == 2
    keys_p = {h.key() for h in ep.D_P.truth.true_objects}
    for r in ep.D_R:
        assert {h.key() for h in r.truth.true_objects} == keys_p, (
            "a replication environment must carry the same true objects")
        assert r.fingerprint() != ep.D_P.fingerprint(), (
            "a replication environment must be independently generated")
        assert set(r.latents) == set(ep.D_P.latents)


def test_null_environments_keep_the_covariates_and_lose_the_relationship():
    ep = make_episode("pure_and", seed=0, n_replication=1)
    assert ep.D_N
    for nw in ep.D_N:
        assert set(nw.latents) == set(ep.D_P.latents)
        assert nw.n == ep.D_P.n


# --------------------------------------------------------------------------
# fingerprints and specs
# --------------------------------------------------------------------------
def test_fingerprint_is_stable_and_sensitive():
    a = build_world("threshold", 3)
    b = build_world("threshold", 3)
    c = build_world("threshold", 4)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_fingerprint_moves_when_only_the_truth_set_moves():
    """Two worlds with identical data but a different object set must not
    share a fingerprint, or a manifest cannot detect a silently relabelled
    ground truth."""
    import dataclasses
    w = build_world("threshold", 3)
    objs = list(w.truth.objects)
    moved = dataclasses.replace(objs[0], operands=("D0",))
    w2 = dataclasses.replace(w, truth=TruthSet(objects=tuple([moved] + objs[1:])))
    assert w.fingerprint() != w2.fingerprint()


def test_world_spec_digest_is_parameter_sensitive_and_order_insensitive():
    a = WorldSpec.make("threshold", 0, tau=1.0, temp=0.1)
    b = WorldSpec.make("threshold", 0, temp=0.1, tau=1.0)
    c = WorldSpec.make("threshold", 0, tau=1.2, temp=0.1)
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_oracle_matrix_expands_multidimensional_latents_with_labels():
    w = build_world("manifold", 0)
    X, names = w.oracle_matrix()
    assert X.shape == (w.n, len(names))
    assert any(nm.startswith("chart[") for nm in names)
    assert len(set(names)) == len(names)


def test_all_registered_families_build():
    for fam in sorted(FAMILIES):
        w = build_world(fam, 0)
        assert w.n > 0 and w.truth.objects
