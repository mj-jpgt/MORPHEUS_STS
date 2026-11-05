import numpy as np

from dbench.downstream import bknown
from dbench.downstream import characterise as C


def grid(n_sec=4, side=20):
    xs, secs, pats = [], [], []
    for s in range(n_sec):
        g = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2) * 100.0
        xs.append(g)
        secs += [f"S{s}"] * len(g)
        pats += [f"P{s}"] * len(g)
    return np.vstack(xs), np.array(secs), np.array(pats)


def knn6(coords, sections):
    from scipy.spatial import cKDTree

    nb = -np.ones((len(coords), 6), dtype=np.int64)
    for s in np.unique(sections):
        ix = np.flatnonzero(sections == s)
        _, j = cKDTree(coords[ix]).query(coords[ix], k=7)
        nb[ix] = ix[j[:, 1:]]
    return nb


def test_ora_planted_set_is_top():
    universe = [f"G{i}" for i in range(2000)]
    refs = {"hallmark": {"PLANT": universe[:40], "OTHER": universe[500:540]}, "gavish": {}, "reactome": {}}
    res = C.ora(universe[:20] + universe[1000:1005], universe, refs)
    assert res[0]["set"] == "PLANT" and res[0]["overlap"] == 20 and res[0]["q"] < 1e-10
    assert next(r for r in res if r["set"] == "OTHER")["overlap"] == 0


def test_composition_strata_rules():
    F = np.array([[0.6, 0.2, 0, 0.2, 0], [0.1, 0.5, 0, 0.4, 0], [0.3, 0.3, 0.1, 0.2, 0.1], [0.55, 0, 0, 0.45, 0]])
    lab = C.composition_strata(F, covered=[True, True, True, False])
    assert list(lab) == ["tumour-rich", "immune-rich", "mixed", "no nuclei"]


def test_localisation_moran_and_diffuse_flag():
    coords, secs, pats = grid()
    nb = knn6(coords, secs)
    smooth = np.sin(coords[:, 0] / 400.0)
    rnd = np.random.default_rng(0).standard_normal(len(coords))
    strata = np.where(smooth > 0, "tumour-rich", "stroma-rich")
    a = C.localisation(smooth, smooth, secs, nb, strata)
    b = C.localisation(rnd, rnd, secs, nb, strata)
    assert a["moran_median"] > 0.5 and not a["spatially_diffuse"]
    assert b["spatially_diffuse"] and a["top_decile_occupancy"]["tumour-rich"] == 1.0


def test_matched_regions_respect_distance_and_structure():
    coords, secs, pats = grid()
    rng = np.random.default_rng(1)
    F = rng.dirichlet(np.ones(5), len(coords))
    state = rng.standard_normal(len(coords))
    reg = C.regions(coords, secs, state, F, size_um=250.0)
    m = C.matched_high_low(reg, n=16)
    assert m["n_pairs"] <= 16 and all(p["mahalanobis"] < 1.0 for p in m["pairs"])
    hi = {p["high"] for p in m["pairs"]}
    lo = {p["low"] for p in m["pairs"]}
    assert not hi & lo and len(hi) == m["n_pairs"] == len(lo)


def test_organisation_beyond_abundance_detects_arrangement():
    coords, secs, pats = grid(n_sec=6, side=16)
    nb = knn6(coords, secs)
    rng = np.random.default_rng(2)
    F = rng.dirichlet(np.ones(5) * 0.5, len(coords))
    counts = rng.poisson(20, len(coords)).astype(float)
    areas = rng.normal(50, 5, len(coords))
    O, _ = C.organisation_descriptors(F, counts, nb)
    state = 2.0 * O[:, 3] + 0.3 * rng.standard_normal(len(coords))      # neighbour inflammatory mean
    res = C.organisation_beyond_abundance(state, F, counts, areas, nb, secs, pats, n_null=9)
    assert res["delta_r2"] > 0.1 and res["delta_r2"] > res["null_q95"]


def test_neighbourhood_runs_and_skips():
    coords, secs, pats = grid(n_sec=3, side=15)
    smooth = np.sin(coords[:, 0] / 300.0)
    strata = np.where(smooth > 0.3, "tumour-rich", np.where(smooth < -0.3, "stroma-rich", "mixed"))
    # state-high spots scattered inside the tumour-rich field: their neighbours are tumour-rich, not stroma
    state = (strata == "tumour-rich") + 0.5 * np.random.default_rng(3).standard_normal(len(coords))
    out = C.neighbourhood(state, strata, coords, secs, moran_median=0.8, involves_space=True, n_perms=50)
    z = out["state_high_vs_strata_z"]
    assert out["status"] == "run" and z["tumour-rich"] > 2 and z["stroma-rich"] < -2
    assert len(out["co_occurrence_interval_um"]) == 10 and set(out["state_high_co_occurrence"]) >= {"stroma-rich"}
    assert C.neighbourhood(smooth, strata, coords, secs, moran_median=0.05, involves_space=True)["status"] == "skipped"


def test_bknown_catalogue_excludes_historical_targets():
    doc = bknown.load_catalogue()
    sets = bknown.covariate_sets(doc)
    assert "tls_plasma_cell_propagation:tls" in sets
    emt = {"CAV1", "TGM2", "FSTL3", "SERPINE1", "FLNA", "KRT19", "C3", "S100A6", "EMP3", "MT2A"}
    assert not any(len(emt & set(g)) >= 3 for g in sets.values())
    assert {e["task_id"] for e in doc["excluded_historical_rediscovery"]} == {"H1", "H2", "H3", "H3b"}
    X = np.arange(12, dtype=float).reshape(3, 4)
    S, names, dropped = bknown.score_sets(X, ["CP", "COL4A1", "A", "B"], {"x": ["CP", "COL4A1"], "y": ["ZZZ"]})
    assert names == ["x"] and np.allclose(S[:, 0], [0.5, 4.5, 8.5]) and "y" in dropped
