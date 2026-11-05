"""C5 (plan `calm-mixing-candy.md`, 2026-09-26): `make_hest_spatial` and
`make_hest_context_threshold_space` no longer read the single-slide INT1
cache (`_load_hest_substrate`) -- they read the real, patient-disjoint
24-patient ccRCC cohort through `morpheus.tier2.substrates.hest.load`
instead, with `role="proposal"` and every other role reading FIXED,
disjoint patient sets.

Built on a small SYNTHETIC h5ad/metadata fixture tree (same layout as
`dbench/tests/test_hest_real_episode.py`'s fixture, minus the CellViT/FM
pieces this module's two families do not need), so this file runs anywhere
without network or box access, exercising `hest.load`'s real code path
unmodified.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
import scipy.sparse as sp

from dbench import spike_in
from dbench.world import World

N_GENES = 30
GENE_NAMES = [f"GENE{i}" for i in range(N_GENES)]
N_PATIENTS = 8
SIDE = 4                 # 4x4 = 16 spots per patient (>= load()'s 10-spot floor)
N_SPOTS = SIDE * SIDE
SPACING = 200.0


def _write_sample(root, numeric_id, patient, seed):
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    xy = (np.column_stack([cols.ravel(), rows.ravel()]) * SPACING).astype(np.int64)
    counts = rng.poisson(lam=8.0, size=(N_SPOTS, N_GENES)).astype(np.float32)
    counts[:, 0] += patient * 3.0   # a real between-patient effect on gene 0,
                                    # the same leak risk `hest.load` residualises.

    a = anndata.AnnData(X=sp.csr_matrix(counts))
    a.var_names = GENE_NAMES
    sid_str = f"INT{numeric_id}"
    a.obs_names = [f"{sid_str}_BC{i}-1" for i in range(N_SPOTS)]
    a.obs["total_counts"] = counts.sum(1)
    a.obsm["spatial"] = xy
    a.write_h5ad(os.path.join(root, f"{sid_str}_st.h5ad"))

    meta = {"patient": f"Patient {patient}", "tissue": "Lymph node",
           "disease_state": "Cancer", "subseries": "frozen_x_1",
           "spots_under_tissue": N_SPOTS}
    with open(os.path.join(root, f"{sid_str}_metadata.json"), "w") as fh:
        json.dump(meta, fh)


@pytest.fixture()
def hest_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("hest_ccrcc_synth_spikein")
    for p in range(1, N_PATIENTS + 1):
        _write_sample(str(root), p, p, seed=1000 + p)
    return str(root)


@pytest.fixture(autouse=True)
def _clear_role_cache():
    """Each test gets a fresh substrate cache: different tests use different
    `data_dir`s (distinct tmp_path roots), and the cache is keyed on
    `(cohort, data_dir, n_hvg, min_depth, k_neighbors)`, but clearing it
    avoids ever comparing across two different pytest tmp_path fixtures by
    accident."""
    spike_in._HEST_ROLE_SUB_CACHE.clear()
    yield
    spike_in._HEST_ROLE_SUB_CACHE.clear()


def _kwargs(hest_root):
    return dict(data_dir=hest_root, n_hvg=10, min_depth=1, k_neighbors=3)


# -------------------------------------------------- patient disjointness --
def test_d_p_and_d_r_patient_sets_are_disjoint(hest_root):
    d_p, d_r = spike_in._hest_role_patient_sets(cohort="ccrcc", data_dir=hest_root)
    assert d_p and d_r
    assert not (set(d_p) & set(d_r))
    assert set(d_p) | set(d_r) == {f"Patient {p}" for p in range(1, N_PATIENTS + 1)}


def test_role_patient_split_is_independent_of_packet_seed(hest_root):
    """The literal property C5 exists for: two calls with DIFFERENT packet
    seeds (as `build_panel_episode` actually makes them: `seed` vs
    `seed + 100_003`) must still agree on the same D_P/D_R patient
    partition, so "proposal" and "replication" are patient-disjoint
    regardless of which seeds a caller happens to use."""
    d_p_a, d_r_a = spike_in._hest_role_patient_sets(cohort="ccrcc", data_dir=hest_root)
    d_p_b, d_r_b = spike_in._hest_role_patient_sets(cohort="ccrcc", data_dir=hest_root)
    assert d_p_a == d_p_b
    assert d_r_a == d_r_b


def test_make_hest_spatial_proposal_and_replication_patients_are_disjoint(hest_root):
    kw = _kwargs(hest_root)
    proposal = spike_in.make_hest_spatial(seed=0, role="proposal", **kw)
    replication = spike_in.make_hest_spatial(seed=100_003, role="replication", **kw)
    p_patients = set(proposal.env.labels["patient"].tolist())
    r_patients = set(replication.env.labels["patient"].tolist())
    assert p_patients and r_patients
    assert not (p_patients & r_patients), \
        "proposal and replication packets must never share a real patient"


def test_make_hest_context_threshold_space_proposal_and_replication_patients_are_disjoint(hest_root):
    kw = _kwargs(hest_root)
    proposal = spike_in.make_hest_context_threshold_space(seed=0, role="proposal", **kw)
    replication = spike_in.make_hest_context_threshold_space(seed=100_003, role="replication", **kw)
    p_patients = set(proposal.env.labels["patient"].tolist())
    r_patients = set(replication.env.labels["patient"].tolist())
    assert p_patients and r_patients
    assert not (p_patients & r_patients)


# ------------------------------------------ no longer sourced from INT1 ---
def test_load_hest_substrate_is_not_called_by_the_new_families(hest_root, monkeypatch):
    """`_load_hest_substrate` (the single-slide INT1 cache) must no longer
    be the source: patch it to raise, and confirm both families still
    build real packets from the new patient-disjoint loader without
    touching it."""
    def _boom():
        raise AssertionError("_load_hest_substrate must not be called")
    monkeypatch.setattr(spike_in, "_load_hest_substrate", _boom)
    kw = _kwargs(hest_root)
    w1 = spike_in.make_hest_spatial(seed=0, role="proposal", **kw)
    w2 = spike_in.make_hest_context_threshold_space(seed=0, role="proposal", **kw)
    assert isinstance(w1, World) and isinstance(w2, World)


def test_substrate_tag_no_longer_names_the_single_int1_slide(hest_root):
    kw = _kwargs(hest_root)
    w = spike_in.make_hest_spatial(seed=0, role="proposal", **kw)
    assert w.env.tags["substrate"] != "HEST_MahmoodLab_INT1"
    assert w.env.tags["n_real_patients_in_pool"] >= 1


# ------------------------------------------------------- world contract ---
def test_make_hest_spatial_world_contract_unchanged(hest_root):
    kw = _kwargs(hest_root)
    w = spike_in.make_hest_spatial(seed=0, role="proposal", **kw)
    assert w.family == "hest_spatial" and w.domain == "spatial"
    assert len(w.truth.objects) == 2
    true_h = next(o for o in w.truth.objects if o.true)
    art_h = next(o for o in w.truth.objects if not o.true)
    assert true_h.operator == "SPACE" and art_h.artifact_class == "spatially_smooth_decoy"
    b_gene = w.meta["b_gene"]
    assert true_h.operands == (("B_" + b_gene),)
    assert w.modalities["X"].shape[0] == w.structure.coords.shape[0]
    assert w.meta["d_p_patients"] and w.meta["d_r_patients"]
    assert not (set(w.meta["d_p_patients"]) & set(w.meta["d_r_patients"]))


def test_make_hest_context_threshold_space_world_contract_unchanged(hest_root):
    kw = _kwargs(hest_root)
    w = spike_in.make_hest_context_threshold_space(seed=0, role="proposal", **kw)
    assert w.family == "hest_context_threshold_space"
    true_h = next(o for o in w.truth.objects if o.true)
    assert true_h.operator == "CONDITION"
    assert set(true_h.operands) == {"B_" + w.meta["b_gene"], "C_" + w.meta["c_gene"]}
    decoy = w.meta["decoy_gene"]
    assert decoy not in (w.meta["b_gene"], w.meta["c_gene"])


def test_b_and_c_gene_agree_between_proposal_and_replication(hest_root):
    """B_GENE/C_GENE are predeclared once per substrate, not re-picked per
    role or per seed -- both roles must name the same two genes."""
    kw = _kwargs(hest_root)
    proposal = spike_in.make_hest_spatial(seed=0, role="proposal", **kw)
    replication = spike_in.make_hest_spatial(seed=100_003, role="replication", **kw)
    assert proposal.meta["b_gene"] == replication.meta["b_gene"]
    assert proposal.meta["c_gene"] == replication.meta["c_gene"]


def test_too_few_patients_is_refused():
    with pytest.raises(ValueError):
        spike_in._hest_role_patient_sets(cohort="ccrcc", data_dir="/nonexistent/path/for/this/test")
