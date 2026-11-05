"""Tests for the LOCKINv2 invisible-intermediate capability worlds."""
from itertools import combinations

import numpy as np
import pytest

from dbench.families import make
from dbench.util import corr


@pytest.mark.parametrize("order", [3, 4, 5])
def test_full_kway_signal_is_strong_but_every_proper_product_is_invisible(order):
    world = make("invisible_k%d" % order, seed=19, role="proposal", n=18000)
    truth = world.truth.objects
    assert len(truth) == 1
    h = truth[0]
    assert h.operator == "AND"
    assert len(h.operands) == order
    assert h.effect is not None and h.effect > 0.70
    assert world.meta["hard"] is True
    assert world.meta["invisible_intermediate_order"] == order

    columns = [world.latents[name] for name in h.operands]
    full = np.prod(np.column_stack(columns), axis=1)
    assert corr(world.target, full) > 0.70
    # Each proper product omits an independent centred constituent.  The
    # tolerance is intentionally finite-sample rather than a recovery score.
    for width in range(1, order):
        for inds in combinations(range(order), width):
            partial = np.prod(np.column_stack([columns[i] for i in inds]), axis=1)
            assert abs(corr(world.target, partial)) < 0.06


def test_orders_outside_lockin_protocol_are_rejected():
    with pytest.raises(ValueError, match="fixes order=3"):
        make("invisible_k3", order=2)
