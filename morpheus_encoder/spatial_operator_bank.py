"""Measured, patient-safe multiscale spatial operators for MORPHEUS discovery.

Operators are measurements, not biological claims.  They never infer a tissue
boundary from a discovery target and never connect observations across a
patient or section.  Each returned column carries a provenance record so a
later candidate can be translated back to distance, scale and source values.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class OperatorColumn:
    name: str
    values: np.ndarray
    provenance: dict


def _validate(values, coords, patients, sections):
    x, xy = np.asarray(values, float), np.asarray(coords, float)
    p, s = np.asarray(patients).astype(str), np.asarray(sections).astype(str)
    if x.ndim not in (1, 2) or xy.shape != (len(x), 2) or len(p) != len(x) or len(s) != len(x):
        raise ValueError("values, coordinates, patient ids and section ids must align")
    if not np.isfinite(x).all() or not np.isfinite(xy).all():
        raise ValueError("operators require finite measured values and coordinates")
    return x, xy, p, s


def annular_mean(values, coords, patients, sections, *, inner: float, outer: float) -> np.ndarray:
    """Mean measured value in a physical-distance annulus, within section only."""
    if not 0 <= inner < outer: raise ValueError("require 0 <= inner < outer")
    x, xy, p, sec = _validate(values, coords, patients, sections)
    out = np.full_like(x, np.nan, dtype=float)
    for patient in np.unique(p):
        for section in np.unique(sec[p == patient]):
            rows = np.flatnonzero((p == patient) & (sec == section))
            d = np.sqrt(((xy[rows, None] - xy[rows][None]) ** 2).sum(-1))
            mask = (d >= inner) & (d < outer)
            if x.ndim == 1:
                count = mask.sum(1); out[rows] = np.where(count, mask @ x[rows] / np.maximum(count, 1), x[rows])
            else:
                count = mask.sum(1)[:, None]; out[rows] = np.where(count, mask @ x[rows] / np.maximum(count, 1), x[rows])
    return out


def local_heterogeneity(values, coords, patients, sections, *, radius: float) -> np.ndarray:
    """Within-annulus variance, a measured local diversity rather than a latent state."""
    x, xy, p, sec = _validate(values, coords, patients, sections)
    if radius <= 0: raise ValueError("radius must be positive")
    out = np.zeros_like(x, dtype=float)
    for patient in np.unique(p):
        for section in np.unique(sec[p == patient]):
            rows = np.flatnonzero((p == patient) & (sec == section)); v = x[rows]
            d = np.sqrt(((xy[rows, None] - xy[rows][None]) ** 2).sum(-1)); mask = (d > 0) & (d <= radius)
            for i in range(len(rows)):
                nbr = v[mask[i]]
                out[rows[i]] = np.var(nbr, axis=0) if len(nbr) else 0.
    return out


def directional_gradient(values, coords, patients, sections, *, radius: float) -> np.ndarray:
    """Magnitude of a local least-squares spatial gradient within each section."""
    x, xy, p, sec = _validate(values, coords, patients, sections)
    if x.ndim != 1: raise ValueError("gradient currently accepts one measured column")
    out = np.zeros(len(x), float)
    for patient in np.unique(p):
        for section in np.unique(sec[p == patient]):
            rows = np.flatnonzero((p == patient) & (sec == section)); d = np.sqrt(((xy[rows,None]-xy[rows][None])**2).sum(-1))
            for i, row in enumerate(rows):
                near = np.flatnonzero((d[i] > 0) & (d[i] <= radius))
                if len(near) < 3: continue
                A = np.column_stack([xy[rows[near]] - xy[row], np.ones(len(near))])
                beta = np.linalg.lstsq(A, x[rows[near]], rcond=None)[0][:2]
                out[row] = np.linalg.norm(beta)
    return out


def composition_entropy(composition, *, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy of already measured composition estimates per location."""
    x = np.asarray(composition, float)
    if x.ndim != 2 or np.any(x < 0) or not np.isfinite(x).all():
        raise ValueError("composition must be a finite non-negative matrix")
    p = x / np.maximum(x.sum(1, keepdims=True), eps)
    return -(p * np.log(np.maximum(p, eps))).sum(1)


def signed_boundary_distance(coords, patients, sections, compartment) -> np.ndarray:
    """Distance to the independently supplied opposite compartment boundary.

    Positive values are inside the lexically first compartment and negative
    values inside the other.  The caller must supply compartment labels that
    were not defined from the discovery target.
    """
    xy = np.asarray(coords, float); p = np.asarray(patients).astype(str); sec = np.asarray(sections).astype(str)
    comp = np.asarray(compartment).astype(str)
    if xy.shape != (len(comp), 2) or len(p) != len(comp) or len(sec) != len(comp):
        raise ValueError("coordinates, patient, section, and compartment must align")
    out = np.full(len(comp), np.nan)
    for patient in np.unique(p):
        for section in np.unique(sec[p == patient]):
            rows = np.flatnonzero((p == patient) & (sec == section))
            labels = np.unique(comp[rows])
            if len(labels) != 2:
                continue
            sign = {labels[0]: 1., labels[1]: -1.}
            for label in labels:
                own, other = rows[comp[rows] == label], rows[comp[rows] != label]
                d = np.sqrt(((xy[own, None] - xy[other][None]) ** 2).sum(-1)).min(1)
                out[own] = sign[label] * d
    return out


def directional_contrast(values, coords, patients, sections, *, radius: float, direction=(1., 0.)) -> np.ndarray:
    """Forward-minus-backward local contrast along a declared physical axis."""
    x, xy, p, sec = _validate(values, coords, patients, sections)
    if x.ndim != 1 or radius <= 0: raise ValueError("one value column and positive radius required")
    direction = np.asarray(direction, float)
    if direction.shape != (2,) or np.linalg.norm(direction) == 0: raise ValueError("direction must be nonzero 2-vector")
    direction /= np.linalg.norm(direction)
    out = np.zeros(len(x))
    for patient in np.unique(p):
        for section in np.unique(sec[p == patient]):
            rows = np.flatnonzero((p == patient) & (sec == section))
            delta = xy[rows, None] - xy[rows][None]
            dist = np.sqrt((delta ** 2).sum(-1)); projection = delta @ direction
            for i, row in enumerate(rows):
                forward = x[rows][(dist[i] > 0) & (dist[i] <= radius) & (projection[i] > 0)]
                backward = x[rows][(dist[i] > 0) & (dist[i] <= radius) & (projection[i] < 0)]
                out[row] = (forward.mean() if len(forward) else x[row]) - (backward.mean() if len(backward) else x[row])
    return out


def local_connectivity(labels, neighbours) -> np.ndarray:
    """Fraction of valid same-label graph neighbors, preserving supplied graph."""
    label = np.asarray(labels).astype(str)
    if len(neighbours) != len(label): raise ValueError("neighbours must align with labels")
    out = np.zeros(len(label))
    for i, nbr in enumerate(neighbours):
        nbr = np.asarray(nbr, int)
        if len(nbr):
            if nbr.min() < 0 or nbr.max() >= len(label): raise ValueError("neighbour index out of bounds")
            out[i] = np.mean(label[nbr] == label[i])
    return out


def build_operator_bank(values, names: Iterable[str], coords, patients, sections, *, annuli=(0., 100., 250.),
                        heterogeneity_radius=100., gradient_radius=150.) -> tuple[np.ndarray, list[OperatorColumn]]:
    """Create a provenance-rich bank from frozen measured columns only."""
    x, names = np.asarray(values, float), list(names)
    if x.ndim != 2 or x.shape[1] != len(names): raise ValueError("values must be [rows, measurements]")
    columns = []
    for j, name in enumerate(names):
        for a, b in zip(annuli[:-1], annuli[1:]):
            columns.append(OperatorColumn(f"annulus[{a:g},{b:g}):{name}", annular_mean(x[:,j], coords, patients, sections, inner=a, outer=b),
                {"operator":"annular_mean","source":name,"inner_distance":a,"outer_distance":b,"section_safe":True}))
        columns.append(OperatorColumn(f"heterogeneity[{heterogeneity_radius:g}]:{name}", local_heterogeneity(x[:,j], coords, patients, sections, radius=heterogeneity_radius),
            {"operator":"local_variance","source":name,"radius":heterogeneity_radius,"section_safe":True}))
        columns.append(OperatorColumn(f"gradient[{gradient_radius:g}]:{name}", directional_gradient(x[:,j], coords, patients, sections, radius=gradient_radius),
            {"operator":"gradient_magnitude","source":name,"radius":gradient_radius,"section_safe":True}))
    return np.column_stack([c.values for c in columns]), columns
