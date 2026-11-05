"""W2: semi-synthetic calibration copies via spike-in correspondence controls.

The problem this exists to fix, measured (`FINDINGS_P1.md` T1.3): a coupling
budget calibrated on synthetic Tier-0 data (eps* = 2, artifact susceptibility
0.11) admitted **four times** the artifact rate (0.44) when the identical
nominal budget was measured on a real molecular substrate. A budget picked on
synthetic data does not transport. The fix has to be a budget picked on the
substrate it will be used on.

This module builds that calibration copy. Given any real matrix `A` (n x d,
one row per spot/sample, one column per real, already-fitted molecular or
morphological state) and a `patient_id` array of the same length, it injects
five kinds of B-side spike-in, all constructed FROM A's own real columns plus
noise so the calibration copy is answerable on the substrate's own
covariance, sparsity and sample size rather than on a generator's:

  marginal        stable: A[:, a] + noise, identical role in every split.
  conjunctive     genuine AND: A[:, a1] * A[:, a2] + noise, marginally
                  uncorrelated with either constituent (E[a1] = E[a2] = 0),
                  present and stable in every split.
  sign_reversing  A[:, a] on the discovery split, -A[:, a] on the held-out
                  split (plus noise everywhere): correlated where it is
                  fitted, anti-correlated where it is checked. Flagging one
                  is a false discovery by construction, and no held-out
                  replication can rescue it.
  discovery_only  A[:, a] + noise on the discovery split, pure independent
                  noise on the held-out split: present exactly where the
                  method is fit, gone exactly where it would be checked. A
                  distinct failure mode from sign-reversal -- nothing to
                  reverse into, just nothing there.
  pure_null       independent noise in every split, with no dependence on A
                  at all. Assigned an "anchor" A-column purely so the
                  (a_tuple, b_index) bookkeeping the rest of the pipeline
                  already uses stays uniform across all five kinds; the
                  anchor is not a generator and the column has zero
                  functional dependence on it.

`A` itself is never modified -- every spike-in is a new column appended to a
separately allocated B, so the calibration copy is provably running on the
substrate's own real covariance among the A columns. `covariance_preservation
_check` makes that an explicit, reported number rather than an assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ARTIFACT_KINDS = ("sign_reversing", "discovery_only", "pure_null")
TRUE_KINDS = ("marginal", "conjunctive")


def patient_split(patient_ids, seed=0, frac_discovery=0.5):
    """Patient-level 0/1 split: 0 = discovery, 1 = held out.

    Never split by spot/row -- a spot from a held-out patient must never
    leak into discovery, which is the whole point of planting artifacts that
    depend on which side of the split a row falls.
    """
    rng = np.random.default_rng(seed)
    pats = np.unique(patient_ids)
    n_disc = max(1, int(round(len(pats) * frac_discovery)))
    disc_pats = set(rng.permutation(pats)[:n_disc])
    return np.array([0 if p in disc_pats else 1 for p in patient_ids])


@dataclass
class SpikeInCopy:
    A: np.ndarray                  # (n, d) real substrate, untouched
    B: np.ndarray                  # (n, m) semi-synthetic, all spike-ins
    split: np.ndarray              # (n,) 0 discovery / 1 held out
    truth: list                    # list of {"kind","a":(idx,),"b":idx,"true":bool}
    meta: dict = field(default_factory=dict)

    def as_marginal_list(self):
        return [(t["a"][0], t["b"]) for t in self.truth if t["kind"] == "marginal"]

    def as_conjunctive_list(self):
        return [(tuple(t["a"]), t["b"]) for t in self.truth if t["kind"] == "conjunctive"]

    def as_artifact_list(self):
        """All three false-correspondence kinds, pooled -- what the coupling
        budget rule is calibrated against, matching how `select_epsilon.py`
        and `tier1_real.py` already use a flat `.artifact` list."""
        return [(t["a"][0], t["b"]) for t in self.truth if t["kind"] in ARTIFACT_KINDS]

    def truth_by_key(self):
        """(frozenset(a), b) -> kind, for post-hoc sub-kind attribution against
        a run's ledger without re-deriving the mapping."""
        return {(frozenset(t["a"]), t["b"]): t["kind"] for t in self.truth}


def build_spike_in_copy(A, patient_ids, seed=0,
                         n_marginal=3, n_conjunctive=3,
                         n_sign_reversing=3, n_discovery_only=3, n_null=3,
                         noise=0.35, frac_discovery=0.5):
    """One semi-synthetic calibration copy of a real substrate `A`.

    `A` is (n, d) and is never mutated. Returns a `SpikeInCopy` whose `.B`
    holds every spike-in column and whose `.truth` labels each by kind.
    """
    A = np.asarray(A, dtype=np.float64)
    n, d = A.shape
    need = n_marginal + 2 * n_conjunctive + n_sign_reversing + n_discovery_only + n_null
    if need > d:
        raise ValueError(f"need {need} distinct A columns, substrate has {d}")

    rng = np.random.default_rng(seed)
    split = patient_split(patient_ids, seed=seed, frac_discovery=frac_discovery)
    disc = split == 0

    pool = list(rng.permutation(d))
    cols, truth = [], []

    def _std(x):
        s = x.std()
        return x / s if s > 1e-12 else x

    for _ in range(n_marginal):
        a = pool.pop()
        x = _std(A[:, a])
        cols.append(x + noise * rng.standard_normal(n))
        truth.append({"kind": "marginal", "a": (a,), "true": True})

    for _ in range(n_conjunctive):
        a1, a2 = pool.pop(), pool.pop()
        x = _std(A[:, a1]) * _std(A[:, a2])
        x = _std(x)
        cols.append(x + noise * rng.standard_normal(n))
        truth.append({"kind": "conjunctive", "a": (a1, a2), "true": True})

    for _ in range(n_sign_reversing):
        a = pool.pop()
        x = _std(A[:, a])
        sign = np.where(disc, 1.0, -1.0)
        cols.append(sign * x + noise * rng.standard_normal(n))
        truth.append({"kind": "sign_reversing", "a": (a,), "true": False})

    for _ in range(n_discovery_only):
        a = pool.pop()
        x = _std(A[:, a])
        col = np.where(disc, x, rng.standard_normal(n))
        cols.append(col + noise * rng.standard_normal(n))
        truth.append({"kind": "discovery_only", "a": (a,), "true": False})

    for _ in range(n_null):
        anchor = pool.pop()          # bookkeeping only, not a generator
        cols.append(rng.standard_normal(n))
        truth.append({"kind": "pure_null", "a": (anchor,), "true": False})

    B = np.column_stack(cols)
    for j, t in enumerate(truth):
        t["b"] = j
    B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-9)

    return SpikeInCopy(A=A, B=B, split=split, truth=truth,
                       meta=dict(n=n, d=d, seed=seed, noise=noise,
                                 frac_discovery=frac_discovery,
                                 n_marginal=n_marginal, n_conjunctive=n_conjunctive,
                                 n_sign_reversing=n_sign_reversing,
                                 n_discovery_only=n_discovery_only, n_null=n_null))


def covariance_preservation_check(A_before, A_after):
    """Does the calibration copy alter the substrate's own covariance?

    `A_after` is whatever the downstream pipeline actually consumed as the
    A-side matrix (after any row subsetting/reordering the caller did, but
    before any injection). This asserts and reports, rather than assumes,
    that constructing B did not perturb A: the two covariance matrices must
    be identical up to float roundoff, because B is built from separately
    allocated columns and A is never written to.

    Returns a dict with the max absolute deviation, which should be at
    machine-epsilon scale; anything else means an injection routine has a
    bug that leaks into the substrate it is supposed to leave alone.
    """
    A0 = np.asarray(A_before, dtype=np.float64)
    A1 = np.asarray(A_after, dtype=np.float64)
    if A0.shape != A1.shape:
        return {"passes": False, "reason": f"shape mismatch {A0.shape} vs {A1.shape}"}
    ident = bool(np.array_equal(A0, A1))
    C0 = np.cov(A0, rowvar=False)
    C1 = np.cov(A1, rowvar=False)
    max_abs_dev = float(np.max(np.abs(C0 - C1)))
    scale = float(np.max(np.abs(C0))) or 1.0
    rel_dev = max_abs_dev / scale
    m0 = A0.mean(0)
    m1 = A1.mean(0)
    max_mean_dev = float(np.max(np.abs(m0 - m1)))
    return {
        "identical_arrays": ident,
        "max_abs_covariance_deviation": max_abs_dev,
        "relative_covariance_deviation": rel_dev,
        "max_abs_mean_deviation": max_mean_dev,
        "passes": bool(ident and max_abs_dev < 1e-8),
    }
