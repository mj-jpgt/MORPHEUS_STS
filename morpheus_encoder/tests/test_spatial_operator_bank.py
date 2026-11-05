import numpy as np
from morpheus_encoder.spatial_operator_bank import annular_mean, build_operator_bank


def test_annulus_never_crosses_patient_or_section():
    coords = np.array([[0, 0], [1, 0], [0, 0], [1, 0]], float)
    values = np.array([1., 3., 100., 200.])
    out = annular_mean(values, coords, ["a", "a", "b", "b"], ["s", "s", "s", "s"], inner=.1, outer=2)
    assert np.allclose(out, [3., 1., 200., 100.])


def test_bank_has_traceable_operator_provenance():
    x = np.arange(8., dtype=float).reshape(4, 2)
    bank, cols = build_operator_bank(x, ["g1", "g2"], [[0,0],[1,0],[0,0],[1,0]], ["a","a","b","b"],
                                     ["s","s","s","s"], annuli=(0., 2.))
    assert bank.shape[1] == 6
    assert all(c.provenance["section_safe"] for c in cols)
