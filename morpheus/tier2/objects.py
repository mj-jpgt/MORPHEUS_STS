"""Tier II semi-synthetic injection: typed objects and the generic Substrate
container they are injected into.

This generalises `cscd/spike_in.py` (patient_split, the five kinds, exact
covariance preservation via append-only construction) from "one real
molecular matrix A" to any real substrate that carries the auxiliary
structure a given object kind needs: an entity-aware discovery/held-out
split, spatial coordinates (+ groups so neighbours never cross patients),
a time order (+ groups so lags never cross an unrelated series), and named
per-row conditioning variables.

`inject.py` is the dispatcher that consumes `Substrate` + a list of
`InjectObject` specs and is otherwise substrate-agnostic: it knows nothing
about "donor" or "patient" or "month" by name, only the generic arrays a
`Substrate` exposes. All domain wiring (which column of `obs` is the split
unit, which columns are coordinates, what "conditional on" means) lives in
`substrates/{bmmc,hest,climate}.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The five kinds spike_in.py already has, unchanged in spirit.
LEGACY_TRUE_KINDS = ("marginal", "conjunctive")
LEGACY_ARTIFACT_KINDS = ("sign_reversing", "discovery_only", "pure_null")
# The five new relation kinds this module adds.
NEW_KINDS = ("threshold", "saturation", "conditional",
             "spatial_neighborhood", "lagged")
ALL_KINDS = LEGACY_TRUE_KINDS + LEGACY_ARTIFACT_KINDS + NEW_KINDS


@dataclass
class Substrate:
    """One real substrate, ready for injection.

    `A` is (n, d) and is NEVER mutated by anything in this package -- every
    spike-in is a new column built from separately-allocated arrays, exactly
    as spike_in.py's docstring promises for its own five kinds. Everything
    else here is auxiliary structure some object kinds need to be generated
    correctly (and that the preservation checks assert is untouched).
    """
    name: str
    A: np.ndarray                              # (n, d) real, untouched
    split: np.ndarray                           # (n,) 0 discovery / 1 held out
    coords: np.ndarray | None = None            # (n, 2) spatial coords
    coord_groups: np.ndarray | None = None      # (n,) e.g. patient id, int-coded
    time_order: np.ndarray | None = None        # (n,) integer time index
    time_groups: np.ndarray | None = None       # (n,) series id, int-coded
    cond_vars: dict = field(default_factory=dict)   # name -> (n,) array
    meta: dict = field(default_factory=dict)    # row_meta for preservation checks

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def d(self):
        return self.A.shape[1]


@dataclass
class InjectObject:
    kind: str
    a: tuple                       # A-column index/indices this object reads
    params: dict = field(default_factory=dict)
    true: bool = True              # real relationship vs. planted artifact
    tag: str = ""                  # free-form label (e.g. "donor_confound")

    def __post_init__(self):
        if self.kind not in ALL_KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if not isinstance(self.a, tuple):
            self.a = (self.a,)


# ---------------------------------------------------------------- factories
# Thin constructors so calling code reads like the relation it is planting,
# and so every kind's parameter contract lives in one place.

def marginal(a):
    return InjectObject("marginal", (a,), true=True)


def conjunctive(a1, a2):
    return InjectObject("conjunctive", (a1, a2), true=True)


def sign_reversing(a):
    return InjectObject("sign_reversing", (a,), true=False)


def discovery_only(a):
    return InjectObject("discovery_only", (a,), true=False)


def pure_null(anchor):
    return InjectObject("pure_null", (anchor,), true=False)


def threshold(a, tau=0.0):
    """Step relation: y depends on 1[A[:,a] > tau], not on A[:,a] linearly.
    Tests whether a discovery engine finds a non-monotone/step object rather
    than only linear correspondence."""
    return InjectObject("threshold", (a,), params={"tau": float(tau)}, true=True)


def saturation(a, tau=0.0, gain=3.0):
    """Saturating dose-response: y ~ sigmoid(gain * (A[:,a] - tau))."""
    return InjectObject("saturation", (a,),
                        params={"tau": float(tau), "gain": float(gain)}, true=True)


def conditional(a, cond_name, cond_value):
    """y depends on A[:,a] only where substrate.cond_vars[cond_name] ==
    cond_value; zero/noise elsewhere. The boss-world nuisance-selection
    primitive (1[C_i=1] * f(A))."""
    return InjectObject("conditional", (a,),
                        params={"cond_name": cond_name, "cond_value": cond_value},
                        true=True)


def spatial_neighborhood(a, k=6, agg="mean"):
    """y depends on the aggregate of A[:,a] over the k spatial neighbours of
    each row (never crossing a coord_group boundary), not on the row's own
    value. Requires substrate.coords/coord_groups."""
    return InjectObject("spatial_neighborhood", (a,),
                        params={"k": int(k), "agg": agg}, true=True)


def lagged(a, lag=2):
    """y[t] depends on A[t-lag, a] within the same time_group (circular
    wrap at the group boundary, so every row gets a well-defined value
    rather than being dropped). Requires substrate.time_order/time_groups."""
    return InjectObject("lagged", (a,), params={"lag": int(lag)}, true=True)
