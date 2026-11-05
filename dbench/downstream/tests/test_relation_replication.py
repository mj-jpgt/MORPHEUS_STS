import json

import numpy as np
import pytest

from dbench.downstream import relation as R
from dbench.downstream import replication_v03 as V


def world(seed, n_pat=10, per=150, k=2, law="additive", sign=1.0, noise=0.6, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    pat = np.repeat([f"p{i}" for i in range(n_pat)], per)
    Z = rng.standard_normal((len(pat), k)) * scale + offset
    pm = rng.standard_normal(n_pat)[np.unique(pat, return_inverse=True)[1]]
    if law == "additive":
        f = np.sin(Z[:, 0]) + 0.5 * Z[:, 1]
    elif law == "product":
        f = Z[:, 0] * Z[:, 1]
    else:
        f = np.zeros(len(pat))
    depth = np.exp(rng.normal(8, 0.3, len(pat)))
    y = sign * f + 0.3 * np.log(depth) + pm + noise * rng.standard_normal(len(pat))
    return Z, y, depth, pat


def freeze(Z, y, depth, pat, cands=("a@v0", "b@v0")):
    Zs, _ = R.standardise_within(Z, pat)
    ys, _ = R.standardise_within(y, pat)
    dep, _ = R.standardise_within(np.log1p(depth), pat)
    base = R.fit_baseline(dep.reshape(-1, 1), ys, pat, ["d"])
    r = ys - base.predict(dep.reshape(-1, 1))
    rel = R.fit_relation(Zs, r, pat, list(cands), "y@v0", "nuis")
    dec = R.fit_decomposition(Zs, r, pat, list(cands), "y@v0", "nuis")
    return V.FrozenFinding("F", rel, None, dec, R.dominant_term(rel, Zs, r)), Zs, r


def test_partitions_counts():
    assert len(R.proper_partitions(2)) == 1
    assert len(R.proper_partitions(3)) == 4
    assert len(R.proper_partitions(4)) == 14


def test_frozen_roundtrip_and_gradient():
    Z, y, d, p = world(0)
    ff, Zs, r = freeze(Z, y, d, p)
    rel2 = R.FrozenRelation.from_json(json.loads(json.dumps(ff.rel_nuis.to_json())))
    assert np.allclose(rel2.predict(Zs), ff.rel_nuis.predict(Zs))
    assert rel2.params_hash() == ff.rel_nuis.params_hash()
    eps = 1e-5
    E = np.zeros_like(Zs[:5])
    E[:, 1] = eps
    num = (ff.rel_nuis.predict(Zs[:5] + E) - ff.rel_nuis.predict(Zs[:5] - E)) / (2 * eps)
    assert np.allclose(num, ff.rel_nuis.gradient(Zs[:5])[:, 1], atol=1e-5)


def test_same_law_replicates_and_flipped_sign_fails_direction():
    ff, _, _ = freeze(*world(1))
    ext = [V.ExternalArrays("c1", *world(2, n_pat=6)), V.ExternalArrays("c2", *world(3, n_pat=5, scale=2.0, offset=1.0))]
    res = V.decide({"F": V.evaluate_finding(ff, ext, n_boot=300)})["F"]
    assert res["pooled"]["nuis"]["ci"][0] > 0 and res["direction"]["fraction_agree"] >= 0.6
    assert res["externally_replicated"]
    flip = [V.ExternalArrays("c1", *world(4, n_pat=6, sign=-1.0))]
    res2 = V.decide({"F": V.evaluate_finding(ff, flip, n_boot=300)})["F"]
    assert res2["direction"]["fraction_agree"] < 0.6 and not res2["externally_replicated"]


def test_null_world_does_not_replicate():
    ff, _, _ = freeze(*world(5))
    ext = [V.ExternalArrays("c1", *world(6, n_pat=8, law="null"))]
    res = V.decide({"F": V.evaluate_finding(ff, ext, n_boot=300)})["F"]
    assert res["pooled"]["nuis"]["ci"][0] <= 0 and not res["externally_replicated"]


def test_decomposition_gain_separates_product_from_additive():
    ffp, _, _ = freeze(*world(7, law="product", noise=0.3))
    extp = [V.ExternalArrays("c", *world(8, law="product", noise=0.3))]
    g_prod = V.evaluate_finding(ffp, extp, n_boot=200)["pooled"]["decomp"]
    ffa, _, _ = freeze(*world(9, law="additive", noise=0.3))
    exta = [V.ExternalArrays("c", *world(10, law="additive", noise=0.3))]
    g_add = V.evaluate_finding(ffa, exta, n_boot=200)["pooled"]["decomp"]
    assert g_prod["ci"][0] > 0.05 and g_add["gain"] < 0.02


def test_bh_and_bootstrap_structure():
    q = V.bh([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)
    S = np.tile([10, 0, 10, 10, 5], (8, 1)).astype(float)
    out = V.bootstrap(S, np.array(["a"] * 8), n_boot=50)
    assert out["ci"][0] == pytest.approx(0.5) and out["p_boot"] == pytest.approx(1 / 51)


def test_protocol_hash_is_recorded():
    p = V.verify_protocol()
    assert p["doc"]["protocol_id"] == "ccrcc-external-replication-v03"
    assert p["doc"]["supersedes"]["protocol_id"] == "ccrcc-external-replication-v02"
