"""Planted known answers for the external replication runner (dbench/scripts/external_replication.py)."""
from __future__ import annotations

import numpy as np
import pytest

from dbench.scripts import external_replication as er


def _cohort(seed=0, n_pat=8, rows=180, beta=0.6, planted_sign=+1):
    rng = np.random.default_rng(seed)
    n = n_pat * rows
    patient = np.repeat([f"P{i}" for i in range(n_pat)], rows)
    section = np.char.add(patient, "_s")
    coords = rng.uniform(0, 1000, size=(n, 2))
    B = rng.normal(size=(n, 6))                       # response-free competitor columns
    x = 0.5 * B[:, 0] + rng.normal(size=n)           # the frozen relation's feature (correlated with B)
    y = 0.4 * B[:, 1] + planted_sign * beta * x + rng.normal(size=n)
    return x, y, B, patient, section, coords


def _signs(x, y, patient):
    return er.within_sign(x, y, patient)[0]


@pytest.mark.parametrize("few", [False, True])
def test_planted_relation_passes(few):
    x, y, B, patient, section, coords = _cohort(n_pat=3 if few else 8)
    res = er.evaluate_relation(x, y, B, patient, section, coords, relation="v05", few=few, n_boot=200, n_perm=99,
                               supp_draws=19)
    assert res["level"] == ("section" if few else "patient")
    assert res["gain"] > 0.05 and res["gain_ci"][0] > 0 and res["perm_p"] <= 0.05
    assert res["supplementary_refit_feature_perm_p"] <= 0.05
    assert _signs(x, y, patient) == +1


def test_sign_flipped_relation_fails_on_sign():
    """Copied with the opposite sign: the gain is real but the direction disagrees with discovery (+1)."""
    x, y, B, patient, section, coords = _cohort(planted_sign=-1)
    res = er.evaluate_relation(x, y, B, patient, section, coords, relation="v05", n_boot=200, n_perm=99)
    discovery_sign = +1
    external_sign = _signs(x, y, patient)
    passed = res["gain"] > 0 and res["gain_ci"][0] > 0 and res["perm_p"] <= 0.05 and external_sign == discovery_sign
    assert external_sign == -1 and not passed


def test_absent_relation_fails():
    x, y, B, patient, section, coords = _cohort(beta=0.0)
    res = er.evaluate_relation(x, y, B, patient, section, coords, relation="v05", n_boot=200, n_perm=99,
                               supp_draws=19)
    assert not (res["gain_ci"][0] > 0 and res["perm_p"] <= 0.05)
    assert res["supplementary_refit_feature_perm_p"] > 0.05


def test_parsing_and_transfer_rules():
    assert er.parse_variable("context_neighbour_mean:nbr:hallmark:HALLMARK_HYPOXIA") == {
        "name": "context_neighbour_mean:nbr:hallmark:HALLMARK_HYPOXIA", "kind": "programme",
        "set": "hallmark:HALLMARK_HYPOXIA", "nbr": 6}
    assert er.parse_variable("raw_genes:SPINK13")["gene"] == "SPINK13"
    for v in ("context_neighbour_mean:nbr:PC2", "pca:PC1", "scgpt:12", "scimilarity:60", "nmf:NMF18"):
        assert er.parse_variable(v)["kind"] == "untransferable", v
    assert er.parse_target("own:IG") == {"name": "own:IG", "kind": "programme", "set": "mp:IG", "nbr": 0}
    assert er.parse_target("nbr:Hypoxia")["nbr"] == 6
    assert er.parse_target("morph_pc4")["kind"] == "morphology"
    assert er.parse_target("nuc_neoplastic_area")["kind"] == "morphology"


def test_rank_normal_within_section():
    x = np.r_[np.arange(5.0), 100 + np.arange(5.0) * 7]
    s = np.r_[np.zeros(5), np.ones(5)].astype(str)
    z = er.rank_normal_within(x, s)
    np.testing.assert_allclose(z[:5], z[5:])            # same ranks -> same scores in each section
    assert abs(z[:5].mean()) < 1e-12 and z[4] > 0 > z[0]


def test_bh_known_answer():
    np.testing.assert_allclose(er.bh([0.01, 0.04, 0.03, 0.5]), [0.04, 0.16 / 3, 0.16 / 3, 0.5])


class _FakeRaw:
    """Minimal hest_multi.StudyRaw stand-in: smooth gene fields on 4 sections (1 patient each)."""

    def __init__(self, seed=0, n_genes=160, side=18):
        import scipy.sparse as sp
        from scipy.ndimage import gaussian_filter

        from morpheus.tier2.substrates.hest_multi import knn_neighbors_kdtree

        rng = np.random.default_rng(seed)
        secs, coords, cols = [], [], []
        for s in range(4):
            g = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2).astype(float)
            coords.append(g * 100.0)
            secs += [f"S{s}"] * len(g)
        coords = np.vstack(coords)
        self.sample_id = np.asarray(secs)
        self.patient = np.char.add("P_", self.sample_id)
        X = np.empty((len(coords), n_genes))
        for j in range(n_genes):
            sig = rng.uniform(0.5, 4.0)
            for s in range(4):
                m = self.sample_id == f"S{s}"
                X[m, j] = gaussian_filter(rng.normal(size=(side, side)), sig).ravel()
            X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-9) + rng.normal(0, 0.3, len(coords))
            X[:, j] = np.exp(0.3 * X[:, j]) * rng.uniform(0.5, 3.0)
        self._X = sp.csr_matrix(X)
        self.genes = np.asarray([f"G{j}" for j in range(n_genes)])
        self.coords = coords
        self.neighbours = {6: knn_neighbors_kdtree(coords, 6, self.sample_id)}
        self.n = len(coords)

    def logcpm(self):
        return self._X


def test_matched_null_planted_passes_and_moran_matched_decoy_fails():
    raw = _FakeRaw()
    coh = er.Cohort("synthetic", raw, "fresh_frozen_whole_transcriptome", None, few=False)
    v = {"name": "programmes:hallmark:SYN", "kind": "programme", "genes": [f"G{j}" for j in range(10)], "nbr": 0}
    tr = {"variables": [v]}
    rng = np.random.default_rng(1)
    f = coh.value(v)
    smooth_y = coh.value({"name": "y", "kind": "programme", "genes": ["G150", "G151", "G152"], "nbr": 0})
    B = coh.transform(np.column_stack([rng.normal(size=raw.n), coh.value(
        {"name": "b", "kind": "gene", "genes": ["G140"], "nbr": 0})]), response=False, rank=False)
    F = coh.transform(f.reshape(-1, 1), response=False, rank=False)
    fz = (F[:, 0] - F[:, 0].mean()) / F[:, 0].std()
    sy = (smooth_y - smooth_y.mean()) / smooth_y.std()
    y_planted = coh.transform(0.7 * fz + sy + rng.normal(0, 1, raw.n), response=True, rank=False).ravel()
    res = er.evaluate_relation(F, y_planted, B, coh.patient, coh.section, coh.coords, relation="v05",
                               n_boot=200, n_perm=0)
    mn = er.matched_null(coh, tr, F, y_planted, B, res, rank=False, relation="v05", seed=3, n_draws=39)
    assert res["gain"] > 0 and res["gain_ci"][0] > 0
    assert mn["status"] == "computed" and mn["matched_p"] <= 0.05, mn
    # decoy: a Moran-matched random programme, unrelated to a smooth response
    y_smooth = coh.transform(sy + rng.normal(0, 1, raw.n), response=True, rank=False).ravel()
    mp = er.matched_programmes(coh, v, seed=11, n_accept=1)
    D = coh.transform(np.asarray(mp["values"][:, :1], float), response=False, rank=False)
    vd = {**v, "name": "decoy"}
    resd = er.evaluate_relation(D, y_smooth, B, coh.patient, coh.section, coh.coords, relation="v05",
                                n_boot=200, n_perm=0)
    mnd = er.matched_null(coh, {"variables": [vd]}, D, y_smooth, B, resd, rank=False, relation="v05", seed=5,
                          n_draws=39)
    assert mnd["matched_p"] > 0.05, (resd["gain"], mnd)
