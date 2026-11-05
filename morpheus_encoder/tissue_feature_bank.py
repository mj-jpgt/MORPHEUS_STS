"""Frozen, multiscale tissue-only representations for cross-view discovery.

The bank is deliberately blind to a molecular programme or response.  It
learns normalisation and image-architecture directions on a discovery
partition, then applies exactly those transforms to held-out patients.  This
keeps tissue-state discovery independent from molecular-state discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


TISSUE_SCALES = ("nuclear", "composition", "graph", "boundary", "architecture")


def _array(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or out.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def derive_cell_graph_features(*, cell_xy: np.ndarray, cell_class: Sequence[object], cell_area: np.ndarray,
                               cell_groups: Sequence[object], anchor_xy: np.ndarray,
                               anchor_groups: Sequence[object], anchor_radius: np.ndarray,
                               classes: Sequence[str] = ("Neoplastic", "Connective", "Epithelial",
                                                         "Inflammatory", "Dead"),
                               k_neighbours: int = 6) -> dict[str, np.ndarray]:
    """Create measured nucleus, composition, graph, and boundary descriptors.

    Nuclei are assigned only to an anchor in the same sample and only when
    they fall within that anchor's declared radius.  Cell graph edges are
    k-nearest-neighbour edges within sample.  The returned descriptors are
    not fitted or response-selected; ``TissueFeatureBank.fit`` performs the
    later D_P-only dictionary fit.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(cell_xy, dtype=float)
    area = np.asarray(cell_area, dtype=float).reshape(-1)
    axy = np.asarray(anchor_xy, dtype=float)
    radius = np.asarray(anchor_radius, dtype=float).reshape(-1)
    groups = np.asarray(cell_groups).reshape(-1)
    agroups = np.asarray(anchor_groups).reshape(-1)
    labels = np.asarray(cell_class).astype(str).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] < 2 or axy.ndim != 2 or axy.shape[1] < 2:
        raise ValueError("cell_xy and anchor_xy must be (n, >=2)")
    if len(xy) != len(area) or len(xy) != len(groups) or len(xy) != len(labels):
        raise ValueError("cell arrays must align")
    if len(axy) != len(radius) or len(axy) != len(agroups):
        raise ValueError("anchor arrays must align")
    if not np.isfinite(xy).all() or not np.isfinite(axy).all() or not np.isfinite(area).all():
        raise ValueError("geometry and area must be finite")
    if not np.isfinite(radius).all() or np.any(radius <= 0) or k_neighbours < 1:
        raise ValueError("anchor_radius and k_neighbours must be positive")
    classes = tuple(map(str, classes))
    if not classes or any(label not in classes for label in labels):
        unknown = sorted(set(labels) - set(classes))
        raise ValueError(f"unknown CellViT classes: {unknown}")

    n_anchor, n_class = len(axy), len(classes)
    assigned: list[list[int]] = [[] for _ in range(n_anchor)]
    same = np.zeros(len(xy)); hetero = np.zeros(len(xy)); mean_dist = np.zeros(len(xy)); degree = np.zeros(len(xy))
    tumor_connective = np.zeros(len(xy)); tumor_inflammatory = np.zeros(len(xy)); tumor_epithelial = np.zeros(len(xy))
    for group in np.unique(agroups):
        anchor_ids = np.flatnonzero(agroups == group)
        cell_ids = np.flatnonzero(groups == group)
        if not len(cell_ids):
            continue
        tree = cKDTree(axy[anchor_ids, :2])
        distance, local = tree.query(xy[cell_ids, :2], k=1)
        for ci, dist, ai in zip(cell_ids, np.asarray(distance), np.asarray(local)):
            anchor = int(anchor_ids[int(ai)])
            if dist <= radius[anchor]:
                assigned[anchor].append(int(ci))
        if len(cell_ids) < 2:
            continue
        ctree = cKDTree(xy[cell_ids, :2])
        dists, local_nb = ctree.query(xy[cell_ids, :2], k=min(k_neighbours + 1, len(cell_ids)))
        dists = np.atleast_2d(dists); local_nb = np.atleast_2d(local_nb)
        for pos, ci in enumerate(cell_ids):
            nb_local = np.asarray(local_nb[pos]).reshape(-1)[1:]
            ds = np.asarray(dists[pos]).reshape(-1)[1:]
            if not len(nb_local):
                continue
            nb = cell_ids[nb_local]
            degree[ci] = len(nb)
            mean_dist[ci] = float(np.mean(ds))
            same[ci] = float(np.mean(labels[nb] == labels[ci]))
            hetero[ci] = 1.0 - same[ci]
            pairs = {(labels[ci], labels[j]) for j in nb}
            tumor_connective[ci] = float(any({"Neoplastic", "Connective"} == set(pair) for pair in pairs))
            tumor_inflammatory[ci] = float(any({"Neoplastic", "Inflammatory"} == set(pair) for pair in pairs))
            tumor_epithelial[ci] = float(any({"Neoplastic", "Epithelial"} == set(pair) for pair in pairs))

    nuclear = np.zeros((n_anchor, 3)); composition = np.zeros((n_anchor, n_class + 1))
    graph = np.zeros((n_anchor, 4)); boundary = np.zeros((n_anchor, 3)); covered = np.zeros(n_anchor, dtype=bool)
    for anchor, ids in enumerate(assigned):
        if not ids:
            continue
        ids = np.asarray(ids, dtype=int); covered[anchor] = True
        a = area[ids]
        nuclear[anchor] = (np.log1p(len(ids)), np.log1p(float(np.mean(a))),
                           float(np.std(a) / max(np.mean(a), 1e-8)))
        fractions = np.asarray([np.mean(labels[ids] == cls) for cls in classes])
        composition[anchor, :n_class] = fractions
        composition[anchor, -1] = float(-np.sum(fractions[fractions > 0] * np.log(fractions[fractions > 0])))
        graph[anchor] = (float(np.mean(same[ids])), float(np.mean(hetero[ids])),
                         float(np.mean(degree[ids])), float(np.mean(mean_dist[ids])))
        boundary[anchor] = (float(np.mean(tumor_connective[ids])), float(np.mean(tumor_inflammatory[ids])),
                            float(np.mean(tumor_epithelial[ids])))
    return {"nuclear": nuclear, "composition": composition, "graph": graph,
            "boundary": boundary, "covered": covered}


@dataclass(frozen=True)
class TissueFeatureMatrix:
    """One frozen tissue dictionary evaluated at locations/anchors."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    scale_slices: Mapping[str, tuple[int, int]]
    provenance: Mapping[str, object]

    def scale(self, name: str) -> np.ndarray:
        start, stop = self.scale_slices[name]
        return self.values[:, start:stop]


@dataclass(frozen=True)
class TissueFeatureBank:
    """Fitted transforms for independent tissue-state discovery.

    ``nuclear``, ``composition``, ``graph`` and ``boundary`` are measured
    descriptors. ``architecture`` can be a high-dimensional frozen pathology
    foundation embedding; it is reduced by a D_P-only PCA dictionary.  The
    bank cannot receive molecular scores or a response, by API design.
    """

    means: Mapping[str, np.ndarray]
    scales: Mapping[str, np.ndarray]
    architecture_basis: np.ndarray
    architecture_components: int
    feature_names: tuple[str, ...]
    scale_slices: Mapping[str, tuple[int, int]]
    fit_count: int
    dictionary_hash: str

    @classmethod
    def fit(cls, *, nuclear: np.ndarray, composition: np.ndarray, graph: np.ndarray,
            boundary: np.ndarray, architecture: np.ndarray, fit_rows: Sequence[int],
            architecture_components: int = 8) -> "TissueFeatureBank":
        raw = {
            "nuclear": _array(nuclear, "nuclear"),
            "composition": _array(composition, "composition"),
            "graph": _array(graph, "graph"),
            "boundary": _array(boundary, "boundary"),
            "architecture": _array(architecture, "architecture"),
        }
        n = raw["nuclear"].shape[0]
        if any(value.shape[0] != n for value in raw.values()):
            raise ValueError("all tissue components must have the same row count")
        fit = np.unique(np.asarray(fit_rows, dtype=int))
        if len(fit) < 3 or fit.min() < 0 or fit.max() >= n:
            raise ValueError("fit_rows must name at least three in-range rows")
        if architecture_components < 0:
            raise ValueError("architecture_components must be non-negative")

        means: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        standard: dict[str, np.ndarray] = {}
        for name, values in raw.items():
            mean, scale = _standardizer(values[fit])
            means[name], scales[name] = mean, scale
            standard[name] = (values - mean) / scale
        k = min(int(architecture_components), standard["architecture"].shape[1], len(fit))
        # H-Optimus is deliberately high-dimensional.  A dense full SVD of
        # every D_P spot by every embedding coordinate is needlessly expensive
        # when the dictionary retains only a handful of directions (normally
        # eight).  The randomized decomposition is deterministic, fits only
        # D_P, and has no access to an outcome; it changes compute, not the
        # scientific definition of the tissue dictionary.  Keeping a small
        # exact-data path makes tiny fixtures straightforward to inspect.
        fit_architecture = standard["architecture"][fit]
        if k == 0:
            basis = np.zeros((0, fit_architecture.shape[1]), dtype=float)
        elif min(fit_architecture.shape) <= 128 or k >= min(fit_architecture.shape):
            _, _, vt = np.linalg.svd(fit_architecture, full_matrices=False)
            basis = vt[:k].copy()
        else:
            from sklearn.utils.extmath import randomized_svd
            _, _, vt = randomized_svd(
                fit_architecture, n_components=k, n_oversamples=min(16, max(4, k)),
                n_iter=5, random_state=20260930, flip_sign=True,
            )
            basis = vt.copy()

        names: list[str] = []
        slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name in TISSUE_SCALES:
            width = k if name == "architecture" else standard[name].shape[1]
            slices[name] = (cursor, cursor + width)
            names.extend(f"{name}:{i}" for i in range(width))
            cursor += width
        identity = {
            "fit_count": int(len(fit)), "architecture_components": int(k),
            "component_widths": {name: int(value.shape[1]) for name, value in raw.items()},
            "feature_names": names,
        }
        return cls(means=MappingProxyType({k: v.copy() for k, v in means.items()}),
                   scales=MappingProxyType({k: v.copy() for k, v in scales.items()}),
                   architecture_basis=basis, architecture_components=k,
                   feature_names=tuple(names), scale_slices=MappingProxyType(dict(slices)),
                   fit_count=int(len(fit)), dictionary_hash=sha256(_canonical(identity).encode()).hexdigest())

    def transform(self, *, nuclear: np.ndarray, composition: np.ndarray, graph: np.ndarray,
                  boundary: np.ndarray, architecture: np.ndarray) -> TissueFeatureMatrix:
        raw = {
            "nuclear": _array(nuclear, "nuclear"),
            "composition": _array(composition, "composition"),
            "graph": _array(graph, "graph"),
            "boundary": _array(boundary, "boundary"),
            "architecture": _array(architecture, "architecture"),
        }
        n = raw["nuclear"].shape[0]
        if any(value.shape[0] != n for value in raw.values()):
            raise ValueError("all tissue components must have the same row count")
        pieces = []
        for name in TISSUE_SCALES:
            if raw[name].shape[1] != len(self.means[name]):
                raise ValueError(f"{name} width differs from the frozen dictionary")
            z = (raw[name] - self.means[name]) / self.scales[name]
            pieces.append(z @ self.architecture_basis.T if name == "architecture" else z)
        values = np.column_stack(pieces)
        return TissueFeatureMatrix(values=values, feature_names=self.feature_names,
                                   scale_slices=self.scale_slices,
                                   provenance=MappingProxyType({
                                       "dictionary_hash": self.dictionary_hash,
                                       "fit_count": self.fit_count,
                                       "scales": list(TISSUE_SCALES),
                                       "architecture_components": self.architecture_components,
                                       "response_used": False,
                                   }))
