"""Multi-study HEST loader: catalogue-driven studies, a per-study cache, and
patient-disjoint `StudyEpisode`s for episodic training of the learned search model.

Sibling of `hest.py`, which stays untouched (its sha256 is pinned in
`tests/test_hest_multi.py`). The arithmetic is imported from `hest.py`, not
copied: log1p CPM at 1e4, programme scores on the full common panel BEFORE the
HVG cut, HVG = top variance on fit (support) rows, `_residualize_against_onehot`
for patient centring, `_within_patient_standardise` for responses.

Layouts
* ``"flat"``     -- HEST-1k layout ``<data_dir>/{st,metadata,patches}/<id>.{h5ad,json,h5}``
* ``"mirror"``   -- MORPHEUS substrate mirror ``<data_dir>/<id>_{st.h5ad,metadata.json,patches.h5}``
* ``"external"`` -- converted external cohorts (flat layout once converted;
  ``pending_conversion`` studies are skipped until their ids are filled).

Contract (published for Agents B, F, G): `StudyEpisode`, `StudySpec`, `StudyRaw`
and the functions below. Episode rows are the rows of the support and query
patients only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
CATALOGUE = f"{STORE}/spatial/hest_meta/HEST_v1_3_0.csv"
HEST_DATA = f"{STORE}/spatial/hest_data"
EMBED_ROOTS = (f"{STORE}/external/hest_fm_embeddings_20260906",
               f"{STORE}/external/hest_fm_embeddings_20261016")
TARGET_FAMILIES = ("gene", "programme", "concept", "nbr_programme")
PLATFORMS = ("st_100um", "visium")


@dataclass
class StudySpec:
    key: str
    dataset_title: str
    role: str
    layout: str                      # "flat" | "mirror" | "external"
    sample_ids: tuple
    patient_of: dict                 # sample id -> patient key
    organ: str
    st_technology: str
    platform: str                    # "st_100um" | "visium"
    embed_roots: tuple               # directories holding <id>.npz (hoptimus), searched in order
    data_dir: str
    dataset_titles: tuple = ()       # member HEST titles when a study pools several
    pending_conversion: bool = False
    partitions_sha256: str = ""
    platform_of: dict = field(default_factory=dict)  # sample id -> platform when a study mixes platforms


@dataclass
class StudyRaw:
    """One study's patient-independent arrays (cached once, reused by every episode)."""
    key: str
    sample_id: np.ndarray            # (n,) str
    barcode: np.ndarray              # (n,) str
    patient: np.ndarray              # (n,) str
    coords: np.ndarray               # (n,2) f64 full-res pixels
    depth: np.ndarray                # (n,) f64 total counts over the common panel (>= 1)
    counts: Any                      # scipy CSR (n, G_all) raw counts over the common panel
    genes: np.ndarray                # (G_all,) symbols
    programmes: np.ndarray           # (n,91) f64 mean log1p-CPM per set, full panel
    programme_names: list
    programme_matched: list
    neighbours: dict                 # {6: (n,k6), 18: (n,k18)} int64, -1 padded, within sample
    hoptimus: np.ndarray | None      # (n,1536) f32 or None
    covered: np.ndarray              # (n,) bool
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sample_id)

    def logcpm(self, rows=None):
        """log1p CPM (1e4) as float64 CSR, exactly as `hest._load_uncached`
        (depth = row sum over the common panel, floored at 1)."""
        full = self.meta.get("_logcpm_full")
        if full is None:  # computed once per study in process (row-wise op, so slicing is exact)
            full = _logcpm_from_counts(self.counts, self.depth)
            self.meta["_logcpm_full"] = full
        return full if rows is None else full[np.asarray(rows)]


@dataclass
class StudyEpisode:
    study: str
    target: str
    target_family: str               # "gene" | "programme" | "concept" | "nbr_programme"
    y: np.ndarray | None             # (n,) f32 within-patient standardised; None for concept targets
    target_concept: int | None
    patient: np.ndarray              # (n,) str
    sample_id: np.ndarray            # (n,) str
    barcode: np.ndarray              # (n,) str
    coords: np.ndarray               # (n,2) f32 full-res pixels
    platform: str                    # "st_100um" | "visium" (per row: meta["platform_of_row"])
    neighbours: dict                 # {6: (n,k), 18: (n,k)} int64, -1 padded, within sample only
    views: dict                      # "genes" (n,G) f32 within-patient centred log1p-CPM;
                                     # "programmes" (n,91) f32 within-patient centred;
                                     # "hoptimus" (n,1536) f32 raw or None; "depth" (n,1) log1p
    columns: dict                    # "genes": symbols; "programmes": "mp:<name>" / "hallmark:<name>"
    covered: np.ndarray              # (n,) bool morphology coverage (never imputed; uncovered rows are 0)
    support_rows: np.ndarray         # int64, patient-disjoint from query_rows
    query_rows: np.ndarray
    exclusions: dict                 # view -> column indices excluded for this target
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.patient)

    def effective_exclusions(self, gene_set_overlap: bool = True) -> dict:
        """`exclusions` (target, member genes, |r| >= 0.9 proxies) plus, when
        `gene_set_overlap`, the programme scores whose gene set overlaps the target's
        (overlap coefficient >= 0.2; programme targets only)."""
        out = {k: set(int(j) for j in v) for k, v in self.exclusions.items()}
        if gene_set_overlap:
            for view, cols in (self.meta.get("overlap_exclusions") or {}).items():
                out.setdefault(view, set()).update(int(j) for j in cols)
        return {k: sorted(v) for k, v in out.items() if v}

    def to_discovery_episode(self, *, gene_set_overlap: bool = True):
        """`DiscoveryEpisode` with proposal = support rows, replication = query rows,
        an empty null partition (`null_source="replication_structured"`), excluded
        columns removed, and coords / patient codes / k=6 neighbours as context.
        With a morphology view, rows are restricted to covered spots (nothing imputed)."""
        from morpheus_encoder.discovery_episode import DiscoveryEpisode, EpisodePartitions

        if self.y is None:
            raise ValueError("concept targets have no measured response; the model side computes them")
        excl = self.effective_exclusions(gene_set_overlap)
        keep = np.arange(self.n)
        if self.views.get("hoptimus") is not None and not self.covered.all():
            keep = np.flatnonzero(self.covered)
        remap = -np.ones(self.n, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        support = np.sort(remap[np.intersect1d(keep, self.support_rows)])
        query = np.sort(remap[np.intersect1d(keep, self.query_rows)])
        modalities, kept_columns = {}, {}
        for view in ("genes", "programmes", "hoptimus"):
            X = self.views.get(view)
            if X is None:
                continue
            cols = _kept_columns(X.shape[1], excl.get(view, ()))
            if len(cols):
                modalities[view] = np.asarray(X[np.ix_(keep, cols)], dtype=np.float64)
                names = self.columns.get(view) or [f"{view}{j}" for j in range(X.shape[1])]
                kept_columns[view] = [str(names[j]) for j in cols]
        nb6 = _remap_neighbours(self.neighbours[6], keep, remap)
        patient_codes = np.unique(self.patient[keep], return_inverse=True)[1].astype(np.int64)
        sample_codes = np.unique(self.sample_id[keep], return_inverse=True)[1].astype(np.int64)
        depth = np.asarray(self.views["depth"], dtype=np.float64).reshape(-1)[keep]
        context = {"coords": np.asarray(self.coords, dtype=np.float64)[keep], "patient_codes": patient_codes,
                   "sample_codes": sample_codes, "neighbors": [r[r >= 0] for r in nb6],
                   "depth_log1p": depth, "baseline": depth.reshape(-1, 1)}
        metadata = {"study": self.study, "target": self.target, "target_family": self.target_family,
                    "null_source": "replication_structured", "source": "hest_multi.StudyEpisode",
                    "gene_symbols": kept_columns.get("genes", []), "columns": kept_columns,
                    "baseline_columns": ["depth_log1p"], "n_rows_total": int(self.n),
                    "n_rows_kept": int(len(keep)), "coverage_fraction": float(len(keep) / max(self.n, 1)),
                    "coverage_rule": ("hoptimus-covered spots only" if len(keep) < self.n else "all rows"),
                    "support_patients": sorted(set(self.patient[self.support_rows].tolist())),
                    "query_patients": sorted(set(self.patient[self.query_rows].tolist())),
                    "exclusions": excl, "gene_set_overlap_applied": bool(gene_set_overlap),
                    "exclusion_report": {**(self.meta.get("exclusion_report") or {}),
                                         **({} if gene_set_overlap else {"gene_set_overlap": {}})},
                    "episode_meta": {k: v for k, v in self.meta.items()
                                     if isinstance(v, (str, int, float, bool))}}
        partitions = EpisodePartitions(support.astype(np.int64), query.astype(np.int64),
                                       np.array([], dtype=np.int64))
        return DiscoveryEpisode(modalities=modalities, response=np.asarray(self.y, dtype=np.float64)[keep],
                                partitions=partitions, context=context, metadata=metadata)

    def to_bench_substrate(self, *, comparison: str = "nuisance", gene_set_overlap: bool = False):
        """`dbench.benchmark.substrate.BenchSubstrate`: searchable views centred within
        patient and z-scored on support rows (excluded columns dropped); comparison =
        depth (+ platform one-hot when it varies and the ladder knows the column)
        (+ the 91 programmes when ``comparison="programmes"``)."""
        from dbench.benchmark.substrate import BenchSubstrate, level_of_column

        if self.y is None:
            raise ValueError("concept targets have no measured response; the model side computes them")
        if comparison not in ("nuisance", "programmes"):
            raise ValueError("comparison must be 'nuisance' or 'programmes'")
        excl = self.effective_exclusions(gene_set_overlap)
        groups = np.unique(self.patient, return_inverse=True)[1].astype(np.int64)
        support = np.asarray(self.support_rows, dtype=np.int64)
        blocks, names, views, start = [], [], {}, 0
        for view, bench_name in (("genes", "raw_genes"), ("programmes", "programmes"), ("hoptimus", "hoptimus")):
            X = self.views.get(view)
            if X is None:
                continue
            cols = _kept_columns(X.shape[1], excl.get(view, ()))
            if not len(cols):
                continue
            block = _centre_within(np.asarray(X[:, cols], dtype=np.float64), groups)
            sd = block[support].std(0)
            block = (block - block[support].mean(0)) / np.where(sd > 1e-8, sd, 1.0)
            blocks.append(block)
            col_names = self.columns.get(view) or [f"{view}{j}" for j in range(X.shape[1])]
            names += [f"{bench_name}:{col_names[j]}" for j in cols]
            views[bench_name] = np.arange(start, start + len(cols))
            start += len(cols)
        # Context candidates are independently available neighbour means of
        # programme activity. They are derived from covariates only and use
        # the within-sample graph supplied by HEST.
        prog = self.views.get("programmes")
        if prog is not None:
            cols = _kept_columns(prog.shape[1], excl.get("programmes", ()))
            if len(cols):
                context = _neighbour_mean(np.asarray(prog[:, cols], dtype=np.float64), self.neighbours[6])
                context = _centre_within(context, groups)
                sd = context[support].std(0)
                context = (context - context[support].mean(0)) / np.where(sd > 1e-8, sd, 1.0)
                blocks.append(context)
                pnames = self.columns.get("programmes") or [f"programmes{j}" for j in range(prog.shape[1])]
                names += [f"context_neighbour_mean:nbr:{pnames[j]}" for j in cols]
                views["context_neighbour_mean"] = np.arange(start, start + len(cols))
                start += len(cols)
        comp = [_centre_within(np.asarray(self.views["depth"], dtype=np.float64).reshape(-1, 1), groups)]
        comp_names = ["depth_log1p"]
        omitted = []
        platform_of_row = np.asarray(self.meta.get("platform_of_row") or [self.platform] * self.n)
        if len(set(platform_of_row.tolist())) > 1:
            for p in sorted(set(platform_of_row.tolist())):
                name = f"platform:{p}"
                try:
                    level_of_column(name)
                except ValueError:
                    omitted.append(name)
                    continue
                comp.append((platform_of_row == p).astype(np.float64).reshape(-1, 1))
                comp_names.append(name)
        if comparison == "programmes":
            comp.append(_centre_within(np.asarray(self.views["programmes"], dtype=np.float64), groups))
            comp_names += list(self.columns["programmes"])
        return BenchSubstrate(
            X=np.column_stack(blocks).astype(np.float32), names=names, views=views,
            roles={v: ("context" if v == "context_neighbour_mean" else "variable") for v in views}, groups=groups, proposal=support,
            heldout=np.asarray(self.query_rows, dtype=np.int64),
            neighbours=np.asarray(self.neighbours[6], dtype=np.int64),
            y_real=np.asarray(self.y, dtype=float), comparison=np.column_stack(comp).astype(np.float32),
            comparison_names=comp_names, coords=np.asarray(self.coords, dtype=float),
            meta={"source": "hest_multi.StudyEpisode", "study": self.study, "target": self.target,
                  "target_family": self.target_family, "comparison": comparison,
                  "comparison_omitted_no_ladder_level": omitted, "exclusions": excl,
                  "gene_set_overlap_applied": bool(gene_set_overlap),
                  "support_patients": sorted(set(self.patient[support].tolist())),
                  "query_patients": sorted(set(self.patient[self.query_rows].tolist()))})


# ------------------------------------------------------------------ helpers --
def _kept_columns(width: int, excluded) -> np.ndarray:
    mask = np.ones(width, dtype=bool)
    ex = np.asarray(list(excluded), dtype=np.int64)
    if ex.size:
        mask[ex] = False
    return np.flatnonzero(mask)


def _centre_within(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Exact within-group centring (each column minus its group mean)."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = X[m] - X[m].mean(0)
    return out


def _remap_neighbours(nb: np.ndarray, keep: np.ndarray, remap: np.ndarray) -> np.ndarray:
    """Restrict a -1 padded neighbour array to `keep` rows, remapping indices and
    dropping neighbours that were removed (left-packed, -1 padded)."""
    sub = np.asarray(nb)[keep]
    mapped = np.where(sub >= 0, remap[np.maximum(sub, 0)], -1)
    out = -np.ones_like(mapped)
    for i, row in enumerate(mapped):
        v = row[row >= 0]
        out[i, :len(v)] = v
    return out


def _neighbour_mean(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Mean over -1 padded neighbours (a row with none keeps its own value)."""
    values = np.asarray(values, dtype=np.float64)
    valid = neighbours >= 0
    gathered = values[np.where(valid, neighbours, 0)]
    mask = valid if values.ndim == 1 else valid[:, :, None]
    total = (gathered * mask).sum(1)
    count = valid.sum(1) if values.ndim == 1 else valid.sum(1)[:, None]
    return np.where(count > 0, total / np.maximum(count, 1), values)


def _proxy_columns(y: np.ndarray, X: np.ndarray, rows: np.ndarray, groups: np.ndarray,
                   threshold: float = 0.9) -> list:
    """Columns whose within-patient |r| with `y` on `rows` is >= threshold."""
    if X is None or X.shape[1] == 0:
        return []
    g = np.asarray(groups)[rows]
    yc = _centre_within(np.asarray(y, dtype=np.float64)[rows].reshape(-1, 1), g).ravel()
    Xc = _centre_within(np.asarray(X, dtype=np.float64)[rows], g)
    r = (Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)
    return [int(j) for j in np.flatnonzero(np.abs(r) >= threshold)]


def _logcpm_from_counts(counts, depth):
    """The `hest._load_uncached` normalisation, in the same dtypes: counts scaled by
    1e4 / depth, log1p on the stored entries, then float64."""
    import scipy.sparse as sp

    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    cpm = counts.tocsr().multiply(1e4 / depth[:, None]).tocsr()
    cpm.data = np.log1p(cpm.data)
    return cpm.astype(np.float64)


# ---------------------------------------------------------------- catalogue --
PARTITIONS_JSON = "dbench/protocols/sprint_partitions_v01.json"
CACHE_VERSION = "hest_multi_v1_20261015"
DEFAULT_CACHE = "~/cache/hest_multi"
GENE_MAP = f"{STORE}/external/gene_annotation/gencode_v47_ensg_to_symbol.tsv"
GENCODE_GTF = f"{STORE}/reference/gencode.v47.annotation.gtf.gz"
MIN_MAPPED_FRACTION = 0.5


def read_catalogue(path: str = CATALOGUE):
    import pandas as pd

    return pd.read_csv(path)


def study_specs(partitions_json: str = PARTITIONS_JSON, roles: Sequence[str] | None = None) -> dict:
    """`StudySpec` per study of the frozen partitions file (optionally restricted to
    `roles`; "sanity" selects Meylan). Pending external studies are returned with
    `pending_conversion=True` and no sample ids."""
    import hashlib
    import json
    import os

    path = partitions_json
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), partitions_json)
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)
    wanted = None
    if roles is not None:
        wanted = {k for r in roles for k in doc["roles"].get(r, [])}
    out = {}
    for key, s in doc["studies"].items():
        if wanted is not None and key not in wanted:
            continue
        platform = s.get("platform") or "visium"
        if isinstance(platform, list):
            platform = platform[0]
        out[key] = StudySpec(
            key=key, dataset_title=str(s.get("dataset_title") or key), role=s["role"],
            layout=s["layout"], sample_ids=tuple(s.get("hest_ids") or ()),
            patient_of=dict(s.get("patient_of") or {}),
            organ=",".join(s.get("organ") or []) if isinstance(s.get("organ"), list) else str(s.get("organ", "")),
            st_technology=",".join(s.get("st_technology") or []) if isinstance(s.get("st_technology"), list)
            else str(s.get("st_technology", "")),
            platform=platform, embed_roots=tuple(s.get("embed_roots") or ()), data_dir=s.get("data_dir", HEST_DATA),
            dataset_titles=tuple(s.get("dataset_titles") or ()), pending_conversion=bool(s.get("pending_conversion")),
            partitions_sha256=sha, platform_of=dict(s.get("platform_of") or {}))
    return out


def _paths(spec: StudySpec, sid: str) -> dict:
    if spec.layout == "mirror":
        d = spec.data_dir
        return {"st": f"{d}/{sid}_st.h5ad", "metadata": f"{d}/{sid}_metadata.json"}
    return {"st": f"{spec.data_dir}/st/{sid}.h5ad", "metadata": f"{spec.data_dir}/metadata/{sid}.json"}


def classify_gene_id_convention(names) -> str:
    """'ensembl' when >= 50 % of identifiers look like Ensembl gene ids, else 'symbol'
    (re-implementation of rpd_scale_substrate.classify_gene_id_convention, which is
    not importable from this clone)."""
    import re

    names = [str(n) for n in names]
    if not names:
        return "symbol"
    pat = re.compile(r"^ENS[A-Z]*G\d{6,}")
    return "ensembl" if np.mean([bool(pat.match(n)) for n in names]) >= 0.5 else "symbol"


_ENSG_MAP = None


def ensg_to_symbol() -> dict:
    """GENCODE v47 Ensembl gene id (version stripped) -> symbol, 1:1 only
    (`cscd.build_breast_pooled.load_ensg_to_symbol`, cached as a TSV under
    `S/external/gene_annotation/`)."""
    global _ENSG_MAP
    if _ENSG_MAP is None:
        from cscd.build_breast_pooled import load_ensg_to_symbol

        _ENSG_MAP = load_ensg_to_symbol(cache_path=GENE_MAP, gtf_path=GENCODE_GTF)
    return _ENSG_MAP


_SETS_MEMO: dict = {}


def _programme_sets():
    if "sets" not in _SETS_MEMO:
        from morpheus.tier2.substrates.hest import load_gavish_programs
        from morpheus_encoder.measurement_bank import load_hallmark_sets

        _SETS_MEMO["sets"] = {"mp": load_gavish_programs(), "hallmark": load_hallmark_sets()}
    return _SETS_MEMO["sets"]


def _file_identity(path):
    import os

    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return [os.path.basename(path), int(st.st_size), int(st.st_mtime_ns)]


def _embedding_path(spec: StudySpec, sid: str):
    import os

    for root in spec.embed_roots:
        p = os.path.join(root, f"{sid}.npz")
        if os.path.exists(p):
            return p
    return None


def _cache_key(spec: StudySpec, min_depth: float, sets: dict):
    import hashlib
    import json

    files = []
    for sid in spec.sample_ids:
        p = _paths(spec, sid)
        files.append([sid, _file_identity(p["st"]), _file_identity(p["metadata"]),
                      _file_identity(_embedding_path(spec, sid) or "")])
    record = {"study": spec.key, "layout": spec.layout, "data_dir": spec.data_dir, "files": files,
              "patient_of": spec.patient_of, "platform": spec.platform, "min_depth": float(min_depth),
              "code_version": CACHE_VERSION,
              "sets": {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16]
                       for k, v in sets.items()}}
    mixed = {k: v for k, v in spec.platform_of.items() if v != spec.platform}
    if mixed:  # only mixed-platform studies carry it, so single-platform cache keys are unchanged
        record["platform_of"] = dict(sorted(spec.platform_of.items()))
    blob = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20], record


def ring_neighbours(coords, groups, rings: int) -> tuple[np.ndarray, float]:
    """Neighbourhoods on a square legacy ST grid: all spots within `rings` grid
    steps (Chebyshev rings, radius (rings + 0.45) x the sample's median
    nearest-neighbour spacing, so ring 1 = up to 8 and ring 2 = up to 24 spots),
    within each group, self excluded, sorted by distance, -1 padded to the widest
    row. Returns (array, mean neighbour count)."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    groups = np.asarray(groups)
    lists = [np.array([], dtype=np.int64)] * len(coords)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        if len(rows) < 2:
            continue
        tree = cKDTree(coords[rows])
        d, _ = tree.query(coords[rows], k=2)
        spacing = float(np.median(d[:, 1]))
        radius = (rings * np.sqrt(2.0) + 0.3) * spacing if rings > 0 else 0.0
        for i, nb in enumerate(tree.query_ball_point(coords[rows], r=radius)):
            nb = np.asarray([j for j in nb if j != i], dtype=np.int64)
            if nb.size:
                delta = np.abs(coords[rows][nb] - coords[rows][i]) / spacing
                nb = nb[np.max(delta, axis=1) <= rings + 0.45]
                dist = np.hypot(*(coords[rows][nb] - coords[rows][i]).T)
                nb = nb[np.argsort(dist, kind="stable")]
            lists[rows[i]] = rows[nb]
    width = max(1, max(len(v) for v in lists))
    out = -np.ones((len(coords), width), dtype=np.int64)
    for i, v in enumerate(lists):
        out[i, :len(v)] = v
    return out, float(np.mean([len(v) for v in lists]))


def _read_sample(spec: StudySpec, sid: str, min_depth: float):
    """One sample after the hest depth filter, var names harmonised to symbols.
    Returns (AnnData, report) or (None, reason)."""
    import anndata as ad

    p = _paths(spec, sid)
    a = ad.read_h5ad(p["st"])
    if "total_counts" in a.obs:
        depth = np.asarray(a.obs["total_counts"].to_numpy(), dtype=np.float64)
    else:
        depth = np.asarray(a.X.sum(1)).ravel().astype(np.float64)
    keep = depth >= min_depth
    if keep.sum() < 10:
        return None, f"fewer than 10 spots with depth >= {min_depth}"
    a = a[keep].copy()
    convention = classify_gene_id_convention(a.var_names)
    report = {"convention": convention, "n_var": int(a.n_vars)}
    if convention == "ensembl":
        mapping = ensg_to_symbol()
        ids = [str(v).split(".")[0] for v in a.var_names]
        sym = [mapping.get(i) for i in ids]
        ok = np.array([s is not None for s in sym])
        report["mapped_fraction"] = float(ok.mean())
        if ok.mean() < MIN_MAPPED_FRACTION:
            return None, f"Ensembl ids, only {ok.mean():.1%} map to GENCODE v47 symbols"
        a = a[:, ok].copy()
        a.var_names = [s for s in sym if s is not None]
    if not a.var_names.is_unique:
        dup = np.asarray(a.var_names.duplicated(keep=False))
        report["duplicate_symbols_dropped"] = int(dup.sum())
        a = a[:, ~dup].copy()
    return a, report


def _load_study_uncached(spec: StudySpec, min_depth: float, sets: dict) -> StudyRaw:
    import scipy.sparse as sp

    from morpheus.tier2.substrates.hest import load_image_embeddings

    ads, sids, dropped, reports = [], [], {}, {}
    for sid in spec.sample_ids:
        a, rep = _read_sample(spec, sid, min_depth)
        if a is None:
            dropped[sid] = rep
            continue
        ads.append(a)
        sids.append(sid)
        reports[sid] = rep
    if not ads:
        raise ValueError(f"{spec.key}: no loadable samples ({dropped})")
    common = set(ads[0].var_names)
    for a in ads[1:]:
        common &= set(a.var_names)
    common = sorted(common)
    blocks, depths, coords, sample_id, barcode, patient = [], [], [], [], [], []
    for sid, a in zip(sids, ads):
        a = a[:, common]
        counts = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
        counts = counts.tocsr()
        d = np.asarray(counts.sum(1)).ravel()
        d = np.maximum(d, 1.0)
        blocks.append(counts)
        depths.append(d)
        coords.append(np.asarray(a.obsm["spatial"], dtype=np.float64))
        sample_id += [sid] * a.n_obs
        barcode += [str(b) for b in a.obs_names]
        patient += [spec.patient_of.get(sid, f"{spec.key}:{sid}")] * a.n_obs
    del ads
    counts = sp.vstack(blocks).tocsr()
    depth = np.concatenate(depths)
    coords = np.vstack(coords)
    sample_id, barcode, patient = np.asarray(sample_id), np.asarray(barcode), np.asarray(patient)
    X = _logcpm_from_counts(counts, depth)
    pos = {g: i for i, g in enumerate(common)}
    prog, names, matched = [], [], []
    for prefix, gene_sets in sets.items():
        block = np.zeros((X.shape[0], len(gene_sets)), dtype=np.float64)
        for j, gs in enumerate(gene_sets):
            cols = [pos[g] for g in gs.get("genes", []) if g in pos]  # hest order (no duplicates in these sets)
            names.append(f"{prefix}:{gs.get('name', gs.get('mp_id', j))}")
            matched.append(len(cols))
            if cols:
                block[:, j] = np.asarray(X[:, cols].mean(axis=1)).ravel()
        prog.append(block)
    programmes = np.column_stack(prog)
    del X
    sample_codes = np.unique(sample_id, return_inverse=True)[1]
    row_platform = np.asarray([spec.platform_of.get(s, spec.platform) for s in sample_id])
    parts6, parts18, rules = [], [], []
    for plat in sorted(set(row_platform.tolist())):
        rows = np.flatnonzero(row_platform == plat)
        if plat == "st_100um":
            a, _ = ring_neighbours(coords[rows], sample_codes[rows], 1)
            b, _ = ring_neighbours(coords[rows], sample_codes[rows], 2)
            rules.append("st_100um: square-grid rings by median spacing (ring 1 -> key 6, rings 1-2 -> key 18)")
        else:
            a = knn_neighbors_kdtree(coords[rows], 6, sample_codes[rows])
            b = knn_neighbors_kdtree(coords[rows], 18, sample_codes[rows])
            rules.append(f"{plat}: KD-tree k nearest within sample")
        parts6.append((rows, np.where(a >= 0, rows[np.maximum(a, 0)], -1)))
        parts18.append((rows, np.where(b >= 0, rows[np.maximum(b, 0)], -1)))

    def _merge(parts):
        out = -np.ones((len(sample_id), max(p[1].shape[1] for p in parts)), dtype=np.int64)
        for rows, nb in parts:
            out[rows, :nb.shape[1]] = nb
        return out

    nb6, nb18 = _merge(parts6), _merge(parts18)
    k6 = float((nb6 >= 0).sum(1).mean())
    k18 = float((nb18 >= 0).sum(1).mean())
    rule = "; ".join(rules)
    hoptimus, covered, emb_report = None, np.zeros(len(sample_id), dtype=bool), {}
    by_root = {}
    for sid in sids:
        p = _embedding_path(spec, sid)
        if p is not None:
            by_root.setdefault(p.rsplit("/", 1)[0], []).append(sid)
    if by_root:
        import types
        for root, members in by_root.items():
            rows = np.flatnonzero(np.isin(sample_id, members))
            view = types.SimpleNamespace(n=len(rows), meta={"sample_id": sample_id[rows], "barcode": barcode[rows],
                                                            "cohort": spec.key})
            E, cov = load_image_embeddings(view, "hoptimus", cohort=spec.key, embed_root=root, min_coverage=0.0)
            if hoptimus is None:
                hoptimus = np.zeros((len(sample_id), E.shape[1]), dtype=np.float32)
            hoptimus[rows] = E
            covered[rows] = cov
            emb_report[root] = members
    missing = [s for s in sids if not any(s in m for m in emb_report.values())]
    meta = {"genes_convention": {s: r["convention"] for s, r in reports.items()},
            "sample_reports": reports, "dropped_samples": dropped, "n_genes_common": len(common),
            "neighbour_rule": rule, "neighbour_k": {6: k6, 18: k18},
            "embeddings": {"roots": emb_report, "samples_without": missing,
                           "coverage": float(covered.mean()) if hoptimus is not None else 0.0},
            "platform": spec.platform, "min_depth": float(min_depth)}
    return StudyRaw(key=spec.key, sample_id=sample_id, barcode=barcode, patient=patient, coords=coords,
                    depth=depth, counts=counts, genes=np.asarray(common), programmes=programmes,
                    programme_names=names, programme_matched=matched, neighbours={6: nb6, 18: nb18},
                    hoptimus=hoptimus, covered=covered, meta=meta)


def _raw_to_arrays(raw: StudyRaw):
    import json

    arrays = {"sample_id": raw.sample_id.astype(str), "barcode": raw.barcode.astype(str),
              "patient": raw.patient.astype(str), "coords": raw.coords, "depth": raw.depth,
              "counts_data": raw.counts.data, "counts_indices": raw.counts.indices,
              "counts_indptr": raw.counts.indptr, "genes": raw.genes.astype(str), "programmes": raw.programmes,
              "nb6": raw.neighbours[6], "nb18": raw.neighbours[18], "covered": raw.covered}
    if raw.hoptimus is not None:
        arrays["hoptimus"] = raw.hoptimus
    header = {"key": raw.key, "shape": list(raw.counts.shape), "programme_names": raw.programme_names,
              "programme_matched": raw.programme_matched,
              "meta": json.loads(json.dumps({k: v for k, v in raw.meta.items() if not k.startswith("_")},
                                            default=str))}
    return arrays, header


def _raw_from_arrays(z, header) -> StudyRaw:
    import scipy.sparse as sp

    counts = sp.csr_matrix((z["counts_data"], z["counts_indices"], z["counts_indptr"]),
                           shape=tuple(header["shape"]))
    meta = dict(header["meta"])
    meta["neighbour_k"] = {int(k): v for k, v in meta.get("neighbour_k", {}).items()}
    return StudyRaw(key=header["key"], sample_id=z["sample_id"], barcode=z["barcode"], patient=z["patient"],
                    coords=z["coords"], depth=z["depth"], counts=counts, genes=z["genes"],
                    programmes=z["programmes"], programme_names=list(header["programme_names"]),
                    programme_matched=list(header["programme_matched"]),
                    neighbours={6: z["nb6"], 18: z["nb18"]},
                    hoptimus=z["hoptimus"] if "hoptimus" in z.files else None, covered=z["covered"], meta=meta)


_RAW_MEMO: dict = {}


def patient_fixed_effect_residual(X, group_codes, alpha: float = 1.0) -> np.ndarray:
    """Closed form of `hest._residualize_against_onehot`: its one-hot normal
    equations are diagonal (n_g + alpha), so beta_g = sum_g / (n_g + alpha) and the
    residual is X minus that per-group shrunken mean. Equal to the hest function to
    float rounding (pinned in the tests), without the dense one-hot products."""
    X = np.asarray(X, dtype=np.float64)
    codes = np.asarray(group_codes)
    n_groups = int(codes.max()) + 1 if codes.size else 0
    sums = np.zeros((n_groups, X.shape[1]))
    np.add.at(sums, codes, X)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    beta = sums / (counts + alpha)[:, None]
    return X - beta[codes]


def _logcpm_csc(raw: StudyRaw):
    csc = raw.meta.get("_logcpm_csc")
    if csc is None:
        csc = raw.logcpm().tocsc()
        raw.meta["_logcpm_csc"] = csc
    return csc


def _patient_stats(raw: StudyRaw):
    """Per-patient sum and sum of squares of log1p-CPM per gene (cached in process),
    so support-row HVG variance needs no pass over the spot matrix."""
    st = raw.meta.get("_patient_stats")
    if st is None:
        import scipy.sparse as sp

        names, codes = np.unique(raw.patient, return_inverse=True)
        ind = sp.csr_matrix((np.ones(raw.n), (codes, np.arange(raw.n))), shape=(len(names), raw.n))
        L = raw.logcpm()
        s1 = np.asarray((ind @ L).todense())
        s2 = np.asarray((ind @ L.multiply(L)).todense())
        st = {"names": names, "index": {p: i for i, p in enumerate(names)},
              "n": np.bincount(codes, minlength=len(names)).astype(np.float64), "s1": s1, "s2": s2}
        raw.meta["_patient_stats"] = st
    return st


def _support_hvg(parts_cols, n_hvg: int) -> np.ndarray:
    """Top-`n_hvg` variance on support rows from per-patient sums (same
    E[x^2] - E[x]^2 arithmetic as `select_hvg`; summation order differs only)."""
    tot1 = tot2 = None
    n = 0.0
    for raw, cols, sup in parts_cols:
        if not len(sup):
            continue
        st = _patient_stats(raw)
        idx = [st["index"][p] for p in sup]
        a, b = st["s1"][idx][:, cols].sum(0), st["s2"][idx][:, cols].sum(0)
        tot1 = a if tot1 is None else tot1 + a
        tot2 = b if tot2 is None else tot2 + b
        n += st["n"][idx].sum()
    mean = tot1 / n
    var = tot2 / n - mean ** 2
    return np.argsort(-var)[:n_hvg]


def load_study(spec: StudySpec, *, min_depth: float = 200, cache_root: str | None = "default") -> StudyRaw:
    """Patient-independent arrays of one study behind an exact on-disk cache:
    `<cache_root>/<study>/<key>.npz` plus `MANIFEST.json` (params, header and the
    file's sha256, verified on every read; a mismatch recomputes). The key covers
    every input file's identity (st, metadata, embedding npz), patient keys,
    platform, min_depth, programme sets and the code version. Also memoised in
    process, so drawing many episodes reads each study once."""
    import json
    import os

    from morpheus.tier2.substrates.hest import _read_multiscale_manifest, _sha256_file, _update_multiscale_manifest

    if spec.pending_conversion or not spec.sample_ids:
        raise ValueError(f"{spec.key}: pending conversion, no samples to load")
    sets = _programme_sets()
    key, record = _cache_key(spec, min_depth, sets)
    if (spec.key, key) in _RAW_MEMO:
        return _RAW_MEMO[(spec.key, key)]
    if cache_root == "default":
        cache_root = os.path.expanduser(DEFAULT_CACHE)
    if cache_root is None:
        raw = _load_study_uncached(spec, min_depth, sets)
    else:
        cache_dir = os.path.join(cache_root, spec.key)
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{key}.npz")
        name = os.path.basename(path)
        entry = _read_multiscale_manifest(cache_dir).get(name)
        if entry and os.path.exists(path) and _sha256_file(path) == entry.get("sha256"):
            with np.load(path, allow_pickle=False) as z:
                raw = _raw_from_arrays(z, entry["header"])
        else:
            raw = _load_study_uncached(spec, min_depth, sets)
            arrays, header = _raw_to_arrays(raw)
            tmp = f"{path}.{os.getpid()}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            _update_multiscale_manifest(cache_dir, name, {"params": json.loads(json.dumps(record, default=str)),
                                                          "header": header, "sha256": _sha256_file(path)},
                                        version=CACHE_VERSION)
            with np.load(path, allow_pickle=False) as z:  # serve what was written
                raw = _raw_from_arrays(z, header)
        raw.meta["cache_file"] = path
    raw.meta["cache_key"] = key
    _RAW_MEMO[(spec.key, key)] = raw
    return raw


def select_hvg(logcpm, fit_rows, n_hvg: int = 768) -> np.ndarray:
    """Top-`n_hvg` variance genes on `fit_rows` (float64 E[x^2] - E[x]^2, argsort of
    -var), the arithmetic of `hest._load_uncached`."""
    import scipy.sparse as sp

    X_fit = logcpm[np.asarray(fit_rows)]
    if sp.issparse(X_fit):
        mean = np.asarray(X_fit.mean(axis=0)).ravel()
        var = np.asarray(X_fit.multiply(X_fit).mean(axis=0)).ravel() - mean ** 2
    else:
        X_fit = np.asarray(X_fit, dtype=np.float64)
        mean = X_fit.mean(axis=0)
        var = (X_fit * X_fit).mean(axis=0) - mean ** 2
    return np.argsort(-var)[:n_hvg]


def knn_neighbors_kdtree(coords, k: int, groups) -> np.ndarray:
    """(n, k) global row indices of the k nearest neighbours within each group
    (cKDTree per group, self excluded, ascending distance), -1 padded when a group
    has fewer than k + 1 rows."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    groups = np.asarray(groups)
    if len(groups) != len(coords):
        raise ValueError("groups must align with coords")
    out = -np.ones((len(coords), k), dtype=np.int64)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        kk = min(k, len(rows) - 1)
        if kk <= 0:
            continue
        _, idx = cKDTree(coords[rows]).query(coords[rows], k=kk + 1)
        idx = np.asarray(idx).reshape(len(rows), kk + 1)
        for i in range(len(rows)):  # drop self wherever it sits (duplicate coordinates)
            row = idx[i][idx[i] != i][:kk]
            out[rows[i], :len(row)] = rows[row]
    return out


class DegenerateTarget(ValueError):
    """The requested target has (almost) no within-patient variation on support rows."""


def parse_target(target: str) -> tuple[str, dict]:
    """`gene:<SYM>`, `programme:<prefix>:<name>`, `nbr6:programme:<prefix>:<name>`,
    `nbr18:programme:...`, `concept:<k>` -> (family, details)."""
    if target.startswith("gene:"):
        return "gene", {"symbol": target[len("gene:"):]}
    if target.startswith("programme:"):
        return "programme", {"column": target[len("programme:"):]}
    for k in (6, 18):
        pre = f"nbr{k}:programme:"
        if target.startswith(pre):
            return "nbr_programme", {"column": target[len(pre):], "k": k}
    if target.startswith("concept:"):
        return "concept", {"concept": int(target[len("concept:"):])}
    raise ValueError(f"unknown target {target!r}")


def _programme_members(column: str) -> list:
    prefix, name = column.split(":", 1)
    for gs in _programme_sets().get(prefix, []):
        if str(gs.get("name", gs.get("mp_id"))) == name:
            return list(gs.get("genes", []))
    return []


OVERLAP_THRESHOLD = 0.2


def gene_set_overlap_columns(target_column: str, programme_names, panel, *, threshold: float = OVERLAP_THRESHOLD) -> dict:
    """{programme index: overlap coefficient} for every programme score whose gene set
    (restricted to genes in `panel`, the genes its score is computed from) has
    |A & B| / min(|A|, |B|) >= threshold with the target programme's set. Includes the
    target itself (coefficient 1)."""
    panel = set(panel)
    a = set(_programme_members(target_column)) & panel
    out = {}
    if not a:
        return out
    for k, name in enumerate(programme_names):
        b = set(_programme_members(name)) & panel
        if not b:
            continue
        c = len(a & b) / min(len(a), len(b))
        if c >= threshold:
            out[k] = float(c)
    return out


def _build_episode(parts, *, target: str, n_hvg: int, with_morphology: bool, seed: int = 0) -> StudyEpisode:
    """parts: [(spec, raw, support_patients, query_patients)], one per study. Rows of
    every listed patient; HVGs, proxies and exclusions on support rows only."""
    import time

    from morpheus.tier2.substrates.hest import _within_patient_standardise

    t0 = time.time()
    family, info = parse_target(target)
    segs, all_sup, all_q = [], set(), set()
    for spec, raw, sup, qry in parts:
        sup, qry = {str(p) for p in sup}, {str(p) for p in qry}
        missing = (sup | qry) - set(np.unique(raw.patient).tolist())
        if missing:
            raise ValueError(f"{spec.key}: patients not loaded: {sorted(missing)}")
        all_sup |= sup
        all_q |= qry
        rows = np.flatnonzero(np.isin(raw.patient, sorted(sup | qry)))
        segs.append((spec, raw, rows, np.isin(raw.patient[rows], sorted(sup))))
    if all_sup & all_q:
        raise ValueError(f"support and query share patients: {sorted(all_sup & all_q)}")
    if not all_sup or not all_q:
        raise ValueError("support and query need at least one patient each")
    common = segs[0][1].genes
    for _, raw, _, _ in segs[1:]:
        common = np.asarray(sorted(set(common.tolist()) & set(raw.genes.tolist())))
    blocks, prog, depth, coords, patient, sample_id, barcode, is_sup, plat = [], [], [], [], [], [], [], [], []
    hop, covered, nbs, offset = [], [], {6: [], 18: []}, 0
    morph_ok = with_morphology and all(raw.hoptimus is not None for _, raw, _, _ in segs)
    for spec, raw, rows, sup_mask in segs:
        pos = {g: i for i, g in enumerate(raw.genes.tolist())}
        cols = np.asarray([pos[g] for g in common.tolist()], dtype=np.int64)
        blocks.append((raw, rows, cols, sorted(set(raw.patient[rows][sup_mask].tolist()))))
        prog.append(raw.programmes[rows])
        depth.append(np.asarray(raw.depth, dtype=np.float64)[rows])
        coords.append(raw.coords[rows])
        patient.append(raw.patient[rows])
        sample_id.append(raw.sample_id[rows])
        barcode.append(raw.barcode[rows])
        is_sup.append(sup_mask)
        plat += [spec.platform_of.get(x, spec.platform) for x in raw.sample_id[rows].tolist()]
        remap = -np.ones(raw.n, dtype=np.int64)
        remap[rows] = offset + np.arange(len(rows))
        for k in (6, 18):
            nb = raw.neighbours[k][rows]
            mapped = np.where(nb >= 0, remap[np.maximum(nb, 0)], -1)
            if (mapped[nb >= 0] < 0).any():
                raise AssertionError("a neighbour fell outside the episode rows (crossed a sample)")
            nbs[k].append(mapped)
        if morph_ok:
            hop.append(raw.hoptimus[rows])
            covered.append(raw.covered[rows])
        else:
            covered.append(np.zeros(len(rows), dtype=bool))
        offset += len(rows)
    patient, sample_id, barcode = np.concatenate(patient), np.concatenate(sample_id), np.concatenate(barcode)
    is_sup = np.concatenate(is_sup)
    support_rows = np.flatnonzero(is_sup).astype(np.int64)
    query_rows = np.flatnonzero(~is_sup).astype(np.int64)
    neighbours = {}
    for k in (6, 18):
        width = max(b.shape[1] for b in nbs[k])
        neighbours[k] = np.vstack([np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=-1)
                                   for b in nbs[k]]).astype(np.int64)
    pcodes = np.unique(patient, return_inverse=True)[1]
    hvg = _support_hvg([(raw, cols, sup) for raw, _, cols, sup in blocks], n_hvg)
    hvg_genes = [str(g) for g in common[hvg]]

    def gene_columns(idx):
        return np.vstack([_logcpm_csc(raw)[:, cols[idx]].toarray()[rows] for raw, rows, cols, _ in blocks])

    genes_view = patient_fixed_effect_residual(gene_columns(hvg), pcodes)
    prog_raw = np.vstack(prog)
    prog_view = patient_fixed_effect_residual(prog_raw, pcodes)
    prog_names = list(segs[0][1].programme_names)
    hoptimus = np.vstack(hop).astype(np.float32) if morph_ok else None
    covered = np.concatenate(covered)

    y, concept, y_raw = None, None, None
    exclusions = {"genes": set(), "programmes": set(), "hoptimus": set()}
    overlap_ex = {"programmes": []}
    report = {"target": [], "member_genes": [], "gene_set_overlap": {}, "proxy_abs_r_ge_0.9": {}}
    if family == "gene":
        sym = info["symbol"]
        where = np.flatnonzero(common == sym)
        if not where.size:
            raise ValueError(f"gene {sym} not in the common panel of {[s.key for s, *_ in segs]}")
        y_raw = gene_columns(np.asarray([int(where[0])])).ravel()
        if sym in hvg_genes:
            exclusions["genes"].add(hvg_genes.index(sym))
        report["target"] = [sym]
    elif family in ("programme", "nbr_programme"):
        col = info["column"]
        if col not in prog_names:
            raise ValueError(f"programme {col!r} unknown (expected 'mp:<name>' or 'hallmark:<name>')")
        j = prog_names.index(col)
        y_raw = prog_raw[:, j]
        if family == "nbr_programme":
            y_raw = _neighbour_mean(y_raw, neighbours[info["k"]])
        exclusions["programmes"].add(j)
        members = set(_programme_members(col))
        exclusions["genes"] |= {i for i, g in enumerate(hvg_genes) if g in members}
        report["target"] = [col]
        report["member_genes"] = sorted(g for g in hvg_genes if g in members)
        panel = set(common.tolist())
        overlap = gene_set_overlap_columns(col, prog_names, panel, threshold=OVERLAP_THRESHOLD)
        overlap_ex["programmes"] = sorted(k for k in overlap if k != j)
        report["gene_set_overlap"] = {prog_names[k]: round(overlap[k], 3) for k in overlap_ex["programmes"]}
    else:
        concept = info["concept"]
        if hoptimus is None:
            raise ValueError("concept targets need the morphology view (with_morphology=True and embeddings)")
        exclusions["hoptimus"] = set(range(hoptimus.shape[1]))
    meta_target = {}
    if y_raw is not None:
        cen = _centre_within(y_raw[support_rows].reshape(-1, 1), pcodes[support_rows]).ravel()
        if float(cen.std()) < 1e-6:
            raise DegenerateTarget(f"{target}: pooled within-patient sd on support rows {cen.std():.2e}")
        y = _within_patient_standardise(y_raw.reshape(-1, 1), pcodes).ravel().astype(np.float32)
        px = {"genes": _proxy_columns(y, genes_view, support_rows, pcodes),
              "programmes": _proxy_columns(y, prog_view, support_rows, pcodes)}
        if hoptimus is not None:
            px["hoptimus"] = _proxy_columns(y, hoptimus, support_rows, pcodes)
        for view, cols in px.items():
            exclusions[view] |= set(cols)
        report["proxy_abs_r_ge_0.9"] = {
            "genes": [hvg_genes[i] for i in px["genes"]], "programmes": [prog_names[i] for i in px["programmes"]],
            "hoptimus": [f"hoptimus{i}" for i in px.get("hoptimus", [])]}
        meta_target = {"response_zero_fraction": float(np.mean(y_raw == 0)),
                       "response_within_patient_sd_support": float(cen.std())}
    keys = [s.key for s, *_ in segs]
    platform = segs[0][0].platform
    meta = {
        "studies": keys, "cross_study": len(segs) > 1, "dataset_title": segs[0][0].dataset_title,
        "platform": platform, "platform_of_row": plat,
        "study_of_row_counts": {s.key: int(len(r)) for s, _, r, _ in segs},
        "gene_convention": {s.key: sorted(set(raw.meta.get("genes_convention", {}).values())) for s, raw, _, _ in segs},
        "hvg_rule": f"top {n_hvg} log1p-CPM variance on support rows over the common panel "
                    f"({len(common)} genes); target-free", "n_genes_common": int(len(common)),
        "min_depth": float(segs[0][1].meta.get("min_depth", 200)),
        "partitions_sha256": segs[0][0].partitions_sha256,
        "neighbour_rule": {s.key: raw.meta.get("neighbour_rule") for s, raw, _, _ in segs},
        "neighbour_k": {k: float((neighbours[k] >= 0).sum(1).mean()) for k in (6, 18)},
        "view_scaling": "genes/programmes: patient fixed-effect residual (closed form of "
                        "hest._residualize_against_onehot, ridge 1); "
                        "hoptimus raw; depth log1p",
        "morphology": ("hoptimus" if hoptimus is not None else
                       ("not requested" if not with_morphology else "embeddings missing for a study")),
        "morphology_coverage": float(covered.mean()) if hoptimus is not None else 0.0,
        "support_patients": sorted(all_sup), "query_patients": sorted(all_q),
        "seed": int(seed), "build_seconds": round(time.time() - t0, 3), **meta_target,
        "overlap_exclusions": overlap_ex, "exclusion_report": report,
        "overlap_rule": f"programme targets: candidate programme scores whose panel-present gene set has overlap "
                        f"coefficient |A&B|/min(|A|,|B|) >= {OVERLAP_THRESHOLD} with the target's are dropped "
                        f"(to_discovery_episode default; to_bench_substrate off unless gene_set_overlap=True)",
    }
    return StudyEpisode(
        study="->".join(keys) if len(keys) > 1 else keys[0], target=target, target_family=family, y=y,
        target_concept=concept, patient=patient, sample_id=sample_id, barcode=barcode,
        coords=np.vstack(coords).astype(np.float32), platform=platform, neighbours=neighbours,
        views={"genes": genes_view.astype(np.float32), "programmes": prog_view.astype(np.float32),
               "hoptimus": hoptimus, "depth": np.log1p(np.concatenate(depth)).reshape(-1, 1).astype(np.float32)},
        columns={"genes": hvg_genes, "programmes": prog_names}, covered=covered,
        support_rows=support_rows, query_rows=query_rows,
        exclusions={k: sorted(int(j) for j in v) for k, v in exclusions.items() if v}, meta=meta)


def study_episode(spec: StudySpec, *, target: str, support_patients, query_patients, n_hvg: int = 768,
                  seed: int = 0, min_depth: float = 200, with_morphology: bool = True) -> StudyEpisode:
    """One patient-disjoint episode inside one study (rows = the listed patients)."""
    raw = load_study(spec, min_depth=min_depth)
    return _build_episode([(spec, raw, support_patients, query_patients)], target=target, n_hvg=n_hvg,
                          with_morphology=with_morphology, seed=seed)


def cross_study_episode(support_spec: StudySpec, query_spec: StudySpec, *, target: str, support_patients,
                        query_patients, n_hvg: int = 768, seed: int = 0, min_depth: float = 200,
                        with_morphology: bool = True) -> StudyEpisode:
    """Support patients from one study, query patients from another (genes = the
    intersection of both panels; HVGs on support rows)."""
    a = load_study(support_spec, min_depth=min_depth)
    b = load_study(query_spec, min_depth=min_depth)
    return _build_episode([(support_spec, a, support_patients, ()), (query_spec, b, (), query_patients)],
                          target=target, n_hvg=n_hvg, with_morphology=with_morphology, seed=seed)


def _random_target(rng, family, raw: StudyRaw, support_patients, n_top: int = 1000):
    if family == "gene":
        st = _patient_stats(raw)
        idx = [st["index"][p] for p in support_patients]
        hvg = _support_hvg([(raw, np.arange(len(raw.genes)), list(support_patients))], n_top)
        mean = st["s1"][idx].sum(0) / st["n"][idx].sum()
        pool = hvg[mean[hvg] >= 0.1]
        pool = pool if pool.size else hvg
        return f"gene:{raw.genes[int(rng.choice(pool))]}"
    pool = [n for n, m in zip(raw.programme_names, raw.programme_matched) if m >= 5]
    col = pool[int(rng.integers(len(pool)))]
    if family == "programme":
        return f"programme:{col}"
    if family == "nbr_programme":
        return f"nbr{int(rng.choice([6, 18]))}:programme:{col}"
    return f"concept:{int(rng.integers(64))}"


def sample_episode(specs: Mapping[str, StudySpec], rng, *, target_families=TARGET_FAMILIES,
                   n_support_patients: int, n_query_patients: int, cross_study_prob: float = 0.3,
                   n_hvg: int = 768, min_depth: float = 200, max_tries: int = 20) -> StudyEpisode:
    """Draw one training episode. A study is drawn uniformly; with probability
    `cross_study_prob` (or when the study alone has too few patients) the query
    patients come from another study of the same organ. Support and query never
    share a patient. A study with too few patients and no same-organ partner uses
    a smaller patient-disjoint split (at least one query patient). Degenerate
    targets are redrawn."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    usable = {k: s for k, s in specs.items() if not s.pending_conversion and s.sample_ids}
    if not usable:
        raise ValueError("no loadable studies")
    keys = sorted(usable)
    last_error = None
    for _ in range(max_tries):
        spec = usable[keys[int(rng.integers(len(keys)))]]
        raw = load_study(spec, min_depth=min_depth)
        pats = np.unique(raw.patient)
        organs = set(spec.organ.split(","))
        partners = [usable[k] for k in keys if k != spec.key and organs & set(usable[k].organ.split(","))]
        want_cross = bool(partners) and (rng.random() < cross_study_prob or len(pats) < n_support_patients + 1)
        if want_cross:
            qspec = partners[int(rng.integers(len(partners)))]
            qraw = load_study(qspec, min_depth=min_depth)
            qpats = np.unique(qraw.patient)
            sup = rng.choice(pats, size=min(n_support_patients, len(pats)), replace=False)
            qry = rng.choice(qpats, size=min(n_query_patients, len(qpats)), replace=False)
            parts = [(spec, raw, sup, ()), (qspec, qraw, (), qry)]
        else:
            if len(pats) < 2:
                last_error = f"{spec.key}: one patient and no same-organ partner"
                continue
            perm = rng.permutation(pats)
            n_q = min(n_query_patients, max(1, len(pats) - n_support_patients), len(pats) - 1)
            n_s = min(n_support_patients, len(pats) - n_q)
            sup, qry = perm[:n_s], perm[n_s:n_s + n_q]
            parts = [(spec, raw, sup, qry)]
        morph = all(r.hoptimus is not None and r.covered.mean() > 0.5 for _, r, _, _ in parts)
        fams = [f for f in target_families if f != "concept" or morph]
        family = fams[int(rng.integers(len(fams)))]
        target = _random_target(rng, family, raw, [str(p) for p in sup])
        try:
            return _build_episode(parts, target=target, n_hvg=n_hvg, with_morphology=morph,
                                  seed=int(rng.integers(2 ** 31)))
        except (DegenerateTarget, ValueError) as exc:  # e.g. gene absent from the partner's panel
            last_error = str(exc)
            continue
    raise RuntimeError(f"sample_episode: no valid episode in {max_tries} draws; last: {last_error}")


def synthetic_study_episode(seed: int = 0, n_patients: int = 8, rows: int = 200, n_genes: int = 60,
                            d_embed: int = 32) -> StudyEpisode:
    """Small fully synthetic episode with the real contract (one sample per patient,
    hex grid, Poisson counts -> log1p CPM, 91 programme columns, a d_embed
    morphology view, ~5 % uncovered spots). Target ``gene:G0`` is planted as
    ``tanh(G1 * G2) + 0.8 * nbr6(G3) + noise`` on within-patient standardised genes;
    the planted set is in ``meta["planted"]``. Support = first half of the patients."""
    from morpheus.tier2.substrates.hest import _within_patient_standardise

    rng = np.random.default_rng(seed)
    n = n_patients * rows
    side = int(np.ceil(np.sqrt(rows)))
    grid = np.array([((i % side) + 0.5 * ((i // side) % 2), (i // side) * np.sqrt(3) / 2)
                     for i in range(rows)], dtype=np.float64) * 100.0
    patient = np.repeat([f"P{p}" for p in range(n_patients)], rows)
    sample_id = np.repeat([f"S{p}" for p in range(n_patients)], rows)
    barcode = np.array([f"bc{i}-1" for i in range(rows)] * n_patients)
    coords = np.tile(grid, (n_patients, 1))
    groups = np.repeat(np.arange(n_patients), rows)
    nb6 = knn_neighbors_kdtree(coords, 6, groups)
    nb18 = knn_neighbors_kdtree(coords, 18, groups)
    rate = np.exp(rng.normal(0.0, 1.0, n_genes))[None, :] * np.exp(rng.normal(0, 0.3, (n, n_genes)))
    rate = rate * np.exp(rng.normal(0, 0.5, (n_patients, n_genes)))[groups]
    counts = rng.poisson(rate * 5.0).astype(np.float64)
    depth = np.maximum(counts.sum(1), 1.0)
    logcpm = np.log1p(counts * (1e4 / depth[:, None]))
    genes = [f"G{j}" for j in range(n_genes)]
    gz = _centre_within(logcpm, groups)
    gz = gz / np.maximum(gz.std(0), 1e-8)
    signal = np.tanh(gz[:, 1] * gz[:, 2]) + 0.8 * _neighbour_mean(gz[:, 3], nb6)
    target_raw = signal + 0.5 * rng.standard_normal(n)
    y = _within_patient_standardise(target_raw.reshape(-1, 1), groups).ravel().astype(np.float32)
    logcpm[:, 0] = target_raw  # the target gene column carries the planted value
    sets = [rng.choice(np.arange(1, n_genes), size=8, replace=False) for _ in range(91)]
    prog = np.column_stack([logcpm[:, s].mean(1) for s in sets])
    prog_names = [f"mp:synthetic_mp{j}" for j in range(41)] + [f"hallmark:SYNTHETIC_{j}" for j in range(50)]
    emb = rng.standard_normal((n, d_embed)).astype(np.float32)
    covered = rng.random(n) >= 0.05
    emb[~covered] = 0.0
    half = n_patients // 2
    support = np.flatnonzero(groups < half).astype(np.int64)
    query = np.flatnonzero(groups >= half).astype(np.int64)
    genes_c = _centre_within(logcpm, groups).astype(np.float32)
    prog_c = _centre_within(prog, groups).astype(np.float32)
    gene_ex = sorted({0} | set(_proxy_columns(y, genes_c, support, groups)))
    exclusions = {"genes": gene_ex, "programmes": _proxy_columns(y, prog_c, support, groups)}
    return StudyEpisode(
        study="synthetic", target="gene:G0", target_family="gene", y=y, target_concept=None,
        patient=patient, sample_id=sample_id, barcode=barcode, coords=coords.astype(np.float32),
        platform="visium", neighbours={6: nb6, 18: nb18},
        views={"genes": genes_c, "programmes": prog_c, "hoptimus": emb,
               "depth": np.log1p(depth).reshape(-1, 1).astype(np.float32)},
        columns={"genes": genes, "programmes": prog_names}, covered=covered,
        support_rows=support, query_rows=query, exclusions=exclusions,
        meta={"synthetic": True, "seed": int(seed), "planted": {"own": ["G1", "G2"], "nbr6": ["G3"]},
              "neighbour_k": {6: 6, 18: 18}, "platform_of_row": ["visium"] * n})


def warm_cache(roles=("meta_training", "development"), partitions_json: str = PARTITIONS_JSON,
               min_depth: float = 200) -> dict:
    """Build the on-disk study cache for every loadable study of `roles`; returns
    {study: {"n_spots", "seconds", "cache_file"} or {"error"}}."""
    import time

    out = {}
    for key, spec in sorted(study_specs(partitions_json, roles=roles).items()):
        if spec.pending_conversion or not spec.sample_ids:
            out[key] = {"skipped": "pending conversion"}
            continue
        t = time.time()
        try:
            raw = load_study(spec, min_depth=min_depth)
            out[key] = {"n_spots": raw.n, "seconds": round(time.time() - t, 1),
                        "cache_file": raw.meta.get("cache_file")}
        except Exception as exc:  # report and continue with the other studies
            out[key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        _RAW_MEMO.clear()
    return out


def section_moran_i(values, neighbours, sections) -> dict:
    """Moran's I of `values` within each section over the row-standardised neighbour
    graph `neighbours` (-1 padded, within section): the per-block statistic of
    `preservation.moran_i`, I = (n / S0) * sum_ij w_ij z_i z_j / sum_i z_i^2."""
    values = np.asarray(values, dtype=np.float64)
    out = {}
    for sec in np.unique(sections):
        rows = np.flatnonzero(sections == sec)
        if len(rows) < 7:
            continue
        local = -np.ones(len(values), dtype=np.int64)
        local[rows] = np.arange(len(rows))
        nb = neighbours[rows]
        nb = np.where(nb >= 0, local[np.maximum(nb, 0)], -1)
        valid = nb >= 0
        cnt = valid.sum(1)
        z = values[rows] - values[rows].mean()
        den = float((z ** 2).sum())
        if den <= 0:
            continue
        lag = np.where(cnt > 0, (np.where(valid, z[np.maximum(nb, 0)], 0.0)).sum(1) / np.maximum(cnt, 1), 0.0)
        s0 = float((cnt > 0).sum())
        out[str(sec)] = float((len(rows) / s0) * float((z * lag).sum()) / den)
    return out


def blind_programme_targets(spec: StudySpec, n_top: int = 8, k: int = 6) -> dict:
    """The `n_top` Gavish programmes with the highest median within-section Moran's I
    over ALL sections of the study (k-nearest neighbours within section; target choice
    only, no response or hypothesis information). Fixed per study, so a programme
    target names the same programme in every fold."""
    raw = load_study(spec)
    sections = np.asarray(raw.sample_id)
    mp = [j for j, n in enumerate(raw.programme_names) if str(n).startswith("mp:")]
    nb = knn_neighbors_kdtree(raw.coords, k, np.unique(sections, return_inverse=True)[1])
    per = {}
    for j in mp:
        per[raw.programme_names[j]] = section_moran_i(raw.programmes[:, j], nb, sections)
    med = {n: float(np.median(list(v.values()))) if v else float("nan") for n, v in per.items()}
    order = sorted((n for n in med if np.isfinite(med[n])), key=lambda n: -med[n])
    top = order[:n_top]
    return {"schema": "morpheus_blind_programme_targets/v1", "study": spec.key,
            "rule": f"Gavish meta-programmes ranked by median within-section Moran's I (k = {k} nearest spots "
                    "within section, row-standardised), computed once on all sections of the study; target "
                    "choice only (no response, no hypothesis information)",
            "partitions_sha256": spec.partitions_sha256, "cache_key": raw.meta.get("cache_key"),
            "sections_used": sorted(set(sections.tolist())), "n_spots": int(raw.n),
            "targets": [{"rank": i + 1, "own_mp": f"own_mp{i + 1}", "programme": n,
                         "loader_target": f"programme:{n}", "median_moran_i": round(med[n], 5),
                         "per_section_moran_i": {s: round(v, 5) for s, v in sorted(per[n].items())}}
                        for i, n in enumerate(top)],
            "all_programmes_median_moran_i": {n: round(med[n], 5) for n in order}}


if __name__ == "__main__":
    import json
    import sys

    if sys.argv[1:2] == ["--blind-targets"]:
        import os

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        specs = study_specs()
        for key in sys.argv[2:]:
            doc = blind_programme_targets(specs[key])
            path = os.path.join(root, "runs", "development", "sprint_v06", f"blind_targets_{key}.json")
            with open(path, "w") as fh:
                json.dump(doc, fh, indent=1)
                fh.write("\n")
            print(key, [t["programme"] for t in doc["targets"]], path)
    else:
        print(json.dumps(warm_cache(tuple(sys.argv[1:]) or ("meta_training", "development")), indent=1))
