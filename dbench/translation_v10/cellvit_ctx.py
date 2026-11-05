"""Frozen CellViT adapter for v08 external residual and compartment candidates.

Only the Zenodo ccRCC cohort is admitted: its retained full-resolution H&E
slides can be mapped to the original Visium coordinate frame.  Li 2022 and
GSE250163 lack such images/segmentations, so this adapter raises
``NotTestable`` for them rather than manufacturing composition labels from
RNA.  Source-study labels are reconstructed by the registered CellViT cache.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from dbench.translation_v10.values_v06_eaca09b import NotTestable


SOURCE_RESIDUAL = Path("/home/ubuntu/cache/hest_multi/resid_targets/"
                       "meylan2022_ccrcc_c3bf14aa1d01c9f0379a_resid_targets_v1_20261026.npz")
SOURCE_ARCHITECTURE = Path("/home/ubuntu/cache/hest_multi/architecture/"
                           "meylan2022_ccrcc_c3bf14aa1d01c9f0379a_architecture_v1_20261023.npz")


def _programme_covariates(coh, comp: np.ndarray) -> np.ndarray:
    names = [str(x) for x in coh.raw.programme_names]
    gav = [i for i, name in enumerate(names) if not name.startswith("hallmark")]
    return np.column_stack([np.log1p(np.asarray(coh.raw.depth, float)), comp,
                            np.asarray(coh.raw.programmes, float)[:, gav]])


def _fill_uncovered_by_section(comp: np.ndarray, sample_id: np.ndarray) -> np.ndarray:
    """Match the registered composition fallback for spots without nuclei."""
    out = np.asarray(comp, float).copy()
    for sid in np.unique(np.asarray(sample_id).astype(str)):
        rows = np.flatnonzero(np.asarray(sample_id).astype(str) == sid)
        covered = np.isfinite(out[rows]).all(1)
        out[rows[~covered]] = out[rows[covered]].mean(0) if covered.any() else 0.0
    return out


def v08_external_cellvit_context(root: str | Path) -> dict:
    root = Path(root)
    manifest = {r["sample"]: r for r in json.loads((root / "wsis" / "adapter_manifest.json").read_text())}

    @lru_cache(maxsize=3)
    def nuclei(sample: str):
        path = root / sample / "cells.json"
        if not path.exists():
            raise NotTestable(f"{sample}: frozen external CellViT output is not available")
        doc = json.loads(path.read_text())
        mapping = {int(k): str(v) for k, v in doc["type_map"].items()}
        cells = doc["cells"]
        xy = np.asarray([x["centroid"] for x in cells], float)
        cls = np.asarray([mapping.get(int(x["type"]), "unknown") for x in cells], object)
        return xy, cls

    cache = {}

    def external_labels(coh):
        key = id(coh)
        if key in cache:
            return cache[key]
        if coh.key == "meylan2022_ccrcc":
            # The source campaign used this frozen architecture cache.  The
            # deployed generic loader intentionally does not expose the v08
            # architecture builder, so read the exact registered cache rather
            # than recomputing it with a different module version.
            if not SOURCE_ARCHITECTURE.exists():
                raise NotTestable("registered source CellViT architecture cache is unavailable")
            with np.load(SOURCE_ARCHITECTURE, allow_pickle=False) as z:
                table = {k: np.asarray(z[k]) for k in z.files if k != "_meta"}
            if len(table["neoplastic_fraction"]) != len(coh.raw.sample_id):
                raise NotTestable("registered source CellViT cache does not match source spot rows")
            comp = _fill_uncovered_by_section(np.column_stack([table["neoplastic_fraction"],
                                                                 table["inflammatory_fraction"],
                                                                 table["connective_fraction"]]), coh.raw.sample_id)
            out = {"S": np.asarray(table["stromal_immune_dominant"], bool),
                   "E": np.asarray(table["epithelial_dominant"], bool),
                   "composition": comp}
            cache[key] = out
            return out
        if coh.key != "zenodo14620362":
            raise NotTestable(f"{coh.key}: no full-resolution external H&E / frozen CellViT segmentation")
        from scipy.spatial import cKDTree
        n = len(coh.raw.sample_id)
        comp = np.full((n, 3), np.nan)
        for sid in np.unique(np.asarray(coh.raw.sample_id).astype(str)):
            # Loader IDs are Z14620362_KC1 etc.; CellViT outputs use KC1.
            sample = sid.rsplit("_", 1)[-1]
            if sample not in manifest:
                raise NotTestable(f"{sid}: no coordinate-transform record for external CellViT output")
            rows = np.flatnonzero(np.asarray(coh.raw.sample_id).astype(str) == sid)
            xy, cls = nuclei(sample)
            item = manifest[sample]
            scale = float(item["source_to_wsi_coordinate_scale"])
            radius = float(item["spot_diameter_fullres_px"]) * scale / 2.0
            found = cKDTree(xy).query_ball_point(np.asarray(coh.raw.coords, float)[rows] * scale, radius)
            for local, hit in enumerate(found):
                if not hit:
                    continue
                labels = cls[np.asarray(hit, int)]
                comp[rows[local]] = [(labels == "Neoplastic").mean(),
                                    (labels == "Inflammatory").mean(),
                                    (labels == "Connective").mean()]
        covered = np.isfinite(comp).all(1)
        safe = np.nan_to_num(comp, nan=-1.0)
        out = {"S": covered & (safe[:, 1:].max(1) > safe[:, 0]),
               "E": covered & (safe[:, 0] > 0) & (safe[:, 0] >= safe[:, 1:].max(1)),
               "composition": _fill_uncovered_by_section(comp, coh.raw.sample_id)}
        cache[key] = out
        return out

    def covariates(coh):
        labels = external_labels(coh)
        return _programme_covariates(coh, labels["composition"])

    with np.load(SOURCE_RESIDUAL, allow_pickle=False) as z:
        hvg, loadings = np.asarray(z["hvg"]).astype(str), np.asarray(z["loadings"], float)
    return {"loadings": {f"G{i + 1}": (hvg, loadings[i]) for i in range(loadings.shape[0])},
            "covariates": covariates, "compartments": external_labels,
            "adapter": {"kind": "v08_frozen_cellvit_external", "root": str(root),
                        "source_residual": str(SOURCE_RESIDUAL),
                        "source_architecture": str(SOURCE_ARCHITECTURE)}}
