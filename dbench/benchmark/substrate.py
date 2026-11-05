"""Benchmark substrate: one search design, one comparison ladder, one patient split.

The real substrate is the v0.5 kidney measurement bank
(``hest.real_episode(fold, substrate_version="v05", bank=True)``) passed through
the runner's own ``episode_design`` adapter, so every search method sees the
exact columns the registered proposers see (within-patient centred, z-scored on
proposal rows).  Nothing here edits or wraps runner code; it only calls it.

Comparison ladder (ideas.md v0.6).  Agent D's ``dbench/decomposition.py`` had
not landed when this was written, so the levels are defined here from the v0.5
competitor's column names and recorded in every output:

* ``N``  = sequencing depth (patient is removed by the within-patient
  standardisation of response and columns);
* ``B1`` = N + CellViT class fractions (cell composition);
* ``B2`` = B1 + the 41 Gavish and 50 Hallmark own-spot programme scores;
* ``B3`` = B2 + neighbour means (k = 6 ``nbr:`` and k = 18 ``nbr18:``) of the
  programmes and PCA columns -- the full registered v0.5 competitor.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STORE = os.environ.get("MORPHEUS_STORE", "/lambda/nfs/geeg/biorag3_persistent_20260711")
DATA_DIR = f"{STORE}/morpheus_substrate_mirror/hest/ccrcc"
EMBED_ROOT = f"{STORE}/external/hest_fm_embeddings_20260906/ccrcc/embeddings/hoptimus"
FOLD_SEED = 20260926  # renal_spatial_v05.json discovery.fold_seed

#: Views of the registered v0.5 families (``renal_spatial_v05.json``).
MOLECULAR_FAMILY = ("raw_genes", "pca", "nmf", "programmes", "scimilarity", "scgpt")
CONTEXT_FAMILY = ("raw_genes", "programmes", "scimilarity", "context_neighbour_mean", "cellvit_structure")
CONTEXT_ROLE_VIEWS = ("context_neighbour_mean", "cellvit_structure")
FM_VIEWS = ("scimilarity", "scgpt")
GENE_VIEWS = ("raw_genes",)
LEVELS = ("N", "B1", "B2", "B3")
#: generic spatial transforms of a scalar column; a hypothesis variable name carries the
#: prefix ``nbr6:`` / ``nbr18:`` / ``contrast:`` (``own`` is the bare design-column name)
TRANSFORMS = ("own", "nbr6", "nbr18", "contrast")
N_NEIGHBOURS_WIDE = 18


def variable_name(name: str, kind: str) -> str:
    """Hypothesis-side name of design column ``name`` under transform ``kind``."""
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown transform {kind!r}")
    return str(name) if kind == "own" else f"{kind}:{name}"


def split_variable(name: str, index: dict) -> tuple[str, str]:
    """Inverse of ``variable_name``: (design-column name, transform).  A name that is itself a
    design column is always ``own`` (so a column named e.g. ``nbr18:x`` is never re-parsed)."""
    name = str(name)
    if name in index:
        return name, "own"
    head, _, rest = name.partition(":")
    if head in TRANSFORMS[1:] and rest in index:
        return rest, head
    raise KeyError(f"{name!r} is neither a design column nor a transform of one")


def level_of_column(name: str) -> str:
    """Ladder level at which a v0.5 competitor column first enters."""
    if name == "depth_log1p":
        return "N"
    if name.startswith("cv_frac_"):
        return "B1"
    if name.startswith(("mp:", "hallmark:")):
        return "B2"
    if name.startswith(("nbr:", "nbr18:")):
        return "B3"
    raise ValueError(f"competitor column {name!r} has no ladder level")


@dataclass
class BenchSubstrate:
    """Everything a world and a method may use, in one place.

    ``X`` holds every searchable column (both families' views); ``views`` maps
    a view name to its column indices; ``roles`` gives each view's role
    (``variable`` or ``context``).  ``neighbours`` is an (n, k) index array,
    -1 padded, of within-patient spatial neighbours (never crossing a patient,
    so never crossing the proposal / held-out split).  ``comparison`` is the
    full B3 competitor; ``level_columns[level]`` indexes its columns."""

    X: np.ndarray
    names: list
    views: dict
    roles: dict
    groups: np.ndarray
    proposal: np.ndarray
    heldout: np.ndarray
    neighbours: np.ndarray
    y_real: np.ndarray
    comparison: np.ndarray
    comparison_names: list
    coords: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    _neighbours18: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.index = {str(c): i for i, c in enumerate(self.names)}
        lv = np.array([LEVELS.index(level_of_column(c)) for c in self.comparison_names])
        self.level_columns = {lev: np.flatnonzero(lv <= k) for k, lev in enumerate(LEVELS)}
        self.view_of = np.empty(len(self.names), dtype=object)
        for v, cols in self.views.items():
            self.view_of[np.asarray(cols, dtype=int)] = v

    @property
    def n(self) -> int:
        return len(self.y_real)

    def level_matrix(self, level: str) -> np.ndarray:
        return self.comparison[:, self.level_columns[level]]

    def family_views(self, family: str, restrict: tuple | None = None) -> dict:
        base = MOLECULAR_FAMILY if family == "molecular" else CONTEXT_FAMILY
        keep = [v for v in base if v in self.views and (restrict is None or v in restrict
                                                        or v in CONTEXT_ROLE_VIEWS and family == "context")]
        return {v: np.asarray(self.views[v], dtype=int) for v in keep}

    def lag(self, col: int | np.ndarray) -> np.ndarray:
        """Within-patient neighbour mean of a column (or of each column of a matrix)."""
        values = self.X[:, col] if np.ndim(col) == 0 else np.asarray(col, dtype=float)
        return neighbour_mean(np.asarray(values, dtype=float), self.neighbours)

    @property
    def neighbours18(self) -> np.ndarray:
        """(n, 18) -1-padded within-patient nearest neighbours (KD-tree per patient, never dense n^2)."""
        if self._neighbours18 is None:
            self._neighbours18 = knn_within_groups(self.coords, self.groups, N_NEIGHBOURS_WIDE)
        return self._neighbours18

    def transform(self, col: int, kind: str) -> np.ndarray:
        """Values of design column ``col`` under a spatial transform (``TRANSFORMS``)."""
        x = np.asarray(self.X[:, int(col)], dtype=float)
        if kind == "own":
            return x
        if kind == "nbr6":
            return neighbour_mean(x, self.neighbours)
        if kind == "nbr18":
            return neighbour_mean(x, self.neighbours18)
        if kind == "contrast":
            return x - neighbour_mean(x, self.neighbours)
        raise ValueError(f"unknown transform {kind!r}")

    def transform_matrix(self, cols, kinds) -> np.ndarray:
        cols, kinds = list(cols), list(kinds)
        if len(cols) != len(kinds):
            raise ValueError("one transform per column")
        return np.column_stack([self.transform(c, k) for c, k in zip(cols, kinds)])

    def resolve(self, name: str) -> np.ndarray:
        """Values of a hypothesis variable name (bare design column or ``<transform>:<column>``)."""
        base, kind = split_variable(name, self.index)
        return self.transform(self.index[base], kind)

    def ladder_agreement(self) -> dict:
        """Compare these levels with Agent D's ``dbench.decomposition.ladder_levels`` when importable."""
        try:
            from dbench.decomposition import ladder_levels
            theirs = ladder_levels(self.comparison_names)
            theirs = theirs.get("levels", theirs)
            same = {lev: sorted(map(int, theirs[lev])) == sorted(map(int, self.level_columns[lev]))
                    for lev in LEVELS}
            return {"available": True, "identical": all(same.values()), "per_level": same}
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.X[:: max(1, self.n // 997)], self.y_real, self.groups, self.proposal, self.heldout):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]


def neighbour_mean(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Mean over the -1-padded neighbour index array (row's own value if it has none)."""
    values = np.asarray(values, dtype=float)
    valid = neighbours >= 0
    safe = np.where(valid, neighbours, 0)
    gathered = values[safe]  # (n, k) or (n, k, p)
    mask = valid if values.ndim == 1 else valid[:, :, None]
    total = (gathered * mask).sum(1)
    count = valid.sum(1) if values.ndim == 1 else valid.sum(1)[:, None]
    return np.where(count > 0, total / np.maximum(count, 1), values)


def knn_within_groups(coords: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """(n, k) -1-padded indices of each row's k nearest rows of the same group (self excluded)."""
    from scipy.spatial import cKDTree

    if coords is None:
        raise ValueError("wide neighbourhoods need spot coordinates")
    coords = np.asarray(coords, dtype=float)
    out = -np.ones((len(coords), k), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        m = min(k, len(idx) - 1)
        if m < 1:
            continue
        _, nn = cKDTree(coords[idx]).query(coords[idx], k=m + 1)
        nn = np.asarray(nn).reshape(len(idx), m + 1)
        own = np.arange(len(idx))[:, None]
        rows = []
        for i in range(len(idx)):  # drop self (normally column 0; ties at distance 0 can reorder)
            r = nn[i][nn[i] != own[i, 0]][:m]
            rows.append(r)
        out[idx, :m] = idx[np.asarray(rows)]
    return out


def _pad_neighbours(neighbours) -> np.ndarray:
    k = max((len(v) for v in neighbours), default=0)
    out = -np.ones((len(neighbours), max(k, 1)), dtype=np.int64)
    for i, idx in enumerate(neighbours):
        out[i, :len(idx)] = np.asarray(idx, dtype=np.int64)
    return out


def load_hest_substrate(fold: int = 0, target: str = "nuc_neoplastic_area", *, cache_dir: str | Path | None = None,
                        data_dir: str = DATA_DIR, embed_root: str = EMBED_ROOT) -> BenchSubstrate:
    """Build (or read from ``cache_dir``) the real v0.5 kidney benchmark substrate."""
    cache = None if cache_dir is None else Path(cache_dir) / f"hest_v05_fold{fold}_{target}"
    if cache is not None and (cache.with_suffix(".npz")).exists():
        return _read_cache(cache)
    from morpheus.tier2.substrates import hest
    from morpheus_encoder.hypothesis_generator import episode_design

    episode = hest.real_episode(fold, target=target, bank=True, substrate_version="v05", seed=FOLD_SEED,
                                data_dir=data_dir, embed_root=embed_root)
    views = tuple(dict.fromkeys(MOLECULAR_FAMILY + CONTEXT_FAMILY))
    # response=None: the design is response-free and shared by every planted world.
    design = episode_design(episode, episode.observed_matrix(), target=target, response=None,
                            candidate_views=views, context_views=CONTEXT_ROLE_VIEWS)
    md = episode.metadata
    groups = np.asarray(episode.context["patient_codes"])
    sub = BenchSubstrate(
        X=np.asarray(design.X, dtype=np.float32), names=list(design.column_names),
        views={k: np.asarray(v, dtype=int) for k, v in design.views.items()},
        roles={k: ("context" if k in CONTEXT_ROLE_VIEWS else "variable") for k in design.views},
        groups=groups, proposal=np.asarray(episode.partitions.proposal, dtype=np.int64),
        heldout=np.asarray(episode.partitions.replication, dtype=np.int64),
        neighbours=_pad_neighbours(episode.context["neighbors"]),
        y_real=np.asarray(episode.response, dtype=float),
        comparison=np.asarray(episode.context["baseline"], dtype=np.float32),
        comparison_names=list(md["baseline_columns"]),
        coords=np.asarray(episode.context["coords"], dtype=float),
        meta={"source": "hest.real_episode", "fold": int(fold), "target": target, "substrate_version": "v05",
              "fold_seed": FOLD_SEED, "fit_patients": list(md["fit_patients"]),
              "held_patients": list(md["held_patients"]), "n_rows": int(episode.n_rows),
              "design_excluded": design.excluded, "ladder": "defined in dbench.benchmark.substrate "
              "(Agent D's decomposition.py not yet available)"})
    if cache is not None:
        _write_cache(sub, cache)
    return sub


def _write_cache(sub: BenchSubstrate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), X=sub.X, groups=sub.groups, proposal=sub.proposal, heldout=sub.heldout,
             neighbours=sub.neighbours, y_real=sub.y_real, comparison=sub.comparison,
             coords=sub.coords if sub.coords is not None else np.zeros((0, 2)))
    path.with_suffix(".json").write_text(json.dumps(
        {"names": sub.names, "views": {k: v.tolist() for k, v in sub.views.items()}, "roles": sub.roles,
         "comparison_names": sub.comparison_names, "meta": sub.meta}, default=str))


def _read_cache(path: Path) -> BenchSubstrate:
    z = np.load(path.with_suffix(".npz"))
    info = json.loads(path.with_suffix(".json").read_text())
    coords = z["coords"]
    return BenchSubstrate(X=z["X"], names=info["names"],
                          views={k: np.asarray(v, dtype=int) for k, v in info["views"].items()},
                          roles=info["roles"], groups=z["groups"], proposal=z["proposal"], heldout=z["heldout"],
                          neighbours=z["neighbours"], y_real=z["y_real"], comparison=z["comparison"],
                          comparison_names=info["comparison_names"], coords=coords if len(coords) else None,
                          meta={**info["meta"], "cache": str(path)})


def synthetic_substrate(*, n_patients: int = 10, rows_per_patient: int = 120, n_genes: int = 40, n_fm: int = 12,
                        n_context: int = 6, seed: int = 0) -> BenchSubstrate:
    """Small fully synthetic substrate with the real one's structure (tests only)."""
    rng = np.random.default_rng(seed)
    n = n_patients * rows_per_patient
    groups = np.repeat(np.arange(n_patients), rows_per_patient)
    side = int(np.ceil(np.sqrt(rows_per_patient)))
    grid = np.array([(i % side, i // side) for i in range(rows_per_patient)], dtype=float)
    coords = np.tile(grid, (n_patients, 1))
    neighbours = []
    for p in range(n_patients):
        idx = np.flatnonzero(groups == p)
        d2 = ((coords[idx, None, :] - coords[None, idx, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        neighbours += [idx[np.argsort(row)[:6]] for row in d2]
    nb = _pad_neighbours(neighbours)
    widths = {"raw_genes": n_genes, "scimilarity": n_fm // 2, "scgpt": n_fm - n_fm // 2, "pca": 4,
              "programmes": 5, "context_neighbour_mean": n_context, "cellvit_structure": 3}
    blocks, names, views, start = [], [], {}, 0
    for v, w in widths.items():
        raw = rng.standard_normal((n, w))
        if v == "context_neighbour_mean":
            raw = neighbour_mean(rng.standard_normal((n, w)), nb)
        blocks.append(raw)
        prefix = "nbr:gavish:prog" if v == "context_neighbour_mean" else ("g" if v == "raw_genes" else "")
        names += [f"{v}:{prefix}{j}" for j in range(w)]
        views[v] = np.arange(start, start + w)
        start += w
    X = np.column_stack(blocks)
    for p in range(n_patients):  # within-patient centring, z-score on proposal rows below
        rows = groups == p
        X[rows] -= X[rows].mean(0)
    held = np.arange(n_patients) >= n_patients - max(2, n_patients // 5)
    heldout = np.flatnonzero(held[groups])
    proposal = np.flatnonzero(~held[groups])
    X = (X - X[proposal].mean(0)) / X[proposal].std(0)
    comp_names = (["depth_log1p", "cv_frac_A", "cv_frac_B"] + [f"mp:prog{j}" for j in range(4)]
                  + [f"hallmark:h{j}" for j in range(3)] + [f"nbr:prog{j}" for j in range(3)]
                  + [f"nbr18:prog{j}" for j in range(2)])
    comparison = rng.standard_normal((n, len(comp_names)))
    y = comparison[:, :6] @ rng.normal(0, 0.3, 6) + rng.standard_normal(n)
    for p in range(n_patients):
        rows = groups == p
        comparison[rows] -= comparison[rows].mean(0)
        y[rows] = (y[rows] - y[rows].mean()) / y[rows].std()
    return BenchSubstrate(X=X.astype(np.float32), names=names, views=views,
                          roles={v: ("context" if v in CONTEXT_ROLE_VIEWS else "variable") for v in views},
                          groups=groups, proposal=proposal, heldout=heldout, neighbours=nb, y_real=y,
                          comparison=comparison.astype(np.float32), comparison_names=comp_names, coords=coords,
                          meta={"source": "synthetic", "seed": seed})
