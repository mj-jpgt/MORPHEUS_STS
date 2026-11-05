"""The ccRCC B_known catalogue (downstream spec Section 9) as covariate gene sets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .locks import canonical_hash

CATALOGUE = Path(__file__).with_name("data") / "bknown_ccrcc_v01.json"


def load_catalogue(path=CATALOGUE) -> dict:
    doc = json.loads(Path(path).read_text())
    doc["_canonical_sha256"] = canonical_hash({k: v for k, v in doc.items() if not k.startswith("_")})
    return doc


def covariate_sets(doc: dict | None = None) -> dict:
    """{'<entry>:<set>': [genes]} for every scorable entry."""
    doc = doc or load_catalogue()
    return {f"{e['key']}:{nm}": list(g) for e in doc["entries"] for nm, g in e.get("gene_sets", {}).items()}


def score_sets(X, genes, sets: dict) -> tuple[np.ndarray, list, dict]:
    """Mean of ``X`` (rows x genes, e.g. log1p CPM) over each set's present genes."""
    gi = {g: i for i, g in enumerate(genes)}
    cols, names, dropped = [], [], {}
    for nm, gs in sets.items():
        idx = [gi[g] for g in gs if g in gi]
        if not idx:
            dropped[nm] = "no gene of the set is measured"
            continue
        v = X[:, idx].mean(axis=1)
        cols.append(np.asarray(v).ravel())
        names.append(nm)
    return (np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))), names, dropped
