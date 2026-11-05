"""Protected measurement bank: interpretable measurements beside frozen views.

Every discovery episode can expose, side by side, the original measurements
(normalised genes, programme scores, measured tissue structure, coordinates)
and several frozen pretrained views (H-Optimus-0 spot embeddings, SCimilarity
and scGPT expression embeddings).  Nothing is compressed into one embedding;
the hypothesis generator decides which view carries signal.

Bank schema (``episode.metadata["bank"]``)::

    view_name -> {"X": (n_rows, d) array aligned to the episode's row order,
                  "columns": [str, ...], "modality": str, "scale": str,
                  "fitted_on": "proposal" | "frozen", "response_free": True,
                  "source": str, "sha256": str | None, ...optional extras}

Optional extras: ``covered`` (boolean row mask; uncovered rows are NaN, never
imputed), ``caveat``, ``excluded_columns`` and view-specific provenance.

Leakage rules enforced here:

* Every fitted statistic (raw-gene selection, standardisation, PCA, NMF) is
  fitted on ``episode.partitions.proposal`` rows only and then applied to all
  rows (``fitted_on="proposal"``).
* No view may contain the response.  The substrate loader names
  target-derived exclusions (``sources["exclude"]``) because only it knows the
  target's construction; this module applies them and records what was
  removed and why.
* The loader in this module never imports a model package: frozen foundation
  model views are read from ``.npz`` caches written by
  ``dbench/scripts/molecular_fm_embed.py``.

This module depends on numpy at import time only (scikit-learn is imported
lazily), so the extraction script can load it by file path from a separate
virtual environment.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

BANK_SCHEMA = "measurement_bank/v1"
FITTED_ON = ("proposal", "frozen")
DEFAULT_CACHE_ROOT = ("/lambda/nfs/geeg/biorag3_persistent_20260711/external/"
                      "molecular_fm_embeddings_20261010")
FM_MODELS = ("scimilarity", "scgpt")
HEST_VIEWS = ("raw_genes", "pca", "nmf", "programmes", "hoptimus", "cellvit_structure",
              "context_neighbour_mean", "coords", "patient", "scimilarity", "scgpt")
BMMC_VIEWS = ("raw_genes", "pca", "programmes", "scimilarity", "scgpt", "donor", "site")
SPOT_CAVEAT = "Visium spot, not a single cell"


# ------------------------------------------------------------- primitives --
def array_sha256(x: np.ndarray) -> str:
    """sha256 of the C-contiguous bytes of ``x`` (dtype and shape are part of
    the cached file, so a byte-identical array reproduces this digest)."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes(order="C")).hexdigest()


def make_view(X, columns: Sequence[str], *, modality: str, scale: str, fitted_on: str,
              source: str, sha256: str | None = None, **extra) -> dict:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("a bank view must be two-dimensional")
    columns = [str(c) for c in columns]
    if len(columns) != X.shape[1]:
        raise ValueError(f"{len(columns)} column names for {X.shape[1]} columns")
    if fitted_on not in FITTED_ON:
        raise ValueError(f"fitted_on must be one of {FITTED_ON}")
    view = {"X": X, "columns": columns, "modality": str(modality), "scale": str(scale),
            "fitted_on": fitted_on, "response_free": True, "source": str(source),
            "sha256": sha256}
    view.update(extra)
    return view


def write_embedding_npz(path: str, embeddings: np.ndarray, barcodes: Sequence[str]) -> str:
    """Write one cached embedding file and return the sha256 of the array.

    The digest is stored inside the file and checked on every read."""
    emb = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    bc = np.asarray([str(b) for b in barcodes])
    if emb.ndim != 2 or emb.shape[0] != len(bc):
        raise ValueError("embeddings must be (n, d) aligned with barcodes")
    digest = array_sha256(emb)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, embeddings=emb, barcodes=bc, sha256=np.asarray(digest))
    os.replace(tmp, path)
    return digest


def read_embedding_npz(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as d:
        emb, bc, stored = d["embeddings"], d["barcodes"], str(d["sha256"])
    digest = array_sha256(emb)
    if digest != stored:
        raise ValueError(f"{path}: array sha256 {digest} != stored {stored}")
    return emb, bc, digest


def read_manifest(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, "MANIFEST.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fm_view_from_cache(model_dir: str, model: str, row_groups: Sequence[str],
                       row_barcodes: Sequence[str], *, scale: str, single_file: str | None = None,
                       caveat: str | None = None) -> dict:
    """Align a cached frozen embedding to the episode rows by (group, barcode).

    ``row_groups`` is the per-row cache file stem (HEST sample id); with
    ``single_file`` every row is looked up in that one file (BMMC).  Rows
    absent from the cache are NaN and ``covered`` is False -- never imputed.
    """
    groups = np.asarray([str(g) for g in row_groups])
    barcodes = np.asarray([str(b) for b in row_barcodes])
    n = len(barcodes)
    X = None
    covered = np.zeros(n, dtype=bool)
    digests = {}
    stems = [single_file] if single_file else list(np.unique(groups))
    for stem in stems:
        path = os.path.join(model_dir, f"{stem}.npz")
        if not os.path.exists(path):
            continue
        emb, bc, digest = read_embedding_npz(path)
        digests[str(stem)] = digest
        if X is None:
            X = np.full((n, emb.shape[1]), np.nan, dtype=np.float32)
        lookup = {str(b): i for i, b in enumerate(bc)}
        rows = np.arange(n) if single_file else np.flatnonzero(groups == stem)
        for r in rows:
            j = lookup.get(barcodes[r])
            if j is not None:
                X[r] = emb[j]
                covered[r] = True
    if X is None:
        raise FileNotFoundError(f"no cached {model} embeddings under {model_dir}")
    manifest = read_manifest(model_dir) or {}
    extra = {"covered": covered, "coverage_fraction": float(covered.mean()),
             "file_sha256": digests, "model_version": manifest.get("model_version"),
             "gene_space": manifest.get("gene_space")}
    if caveat:
        extra["caveat"] = caveat
    sha = digests[stems[0]] if single_file and len(digests) == 1 else None
    return make_view(X, [f"{model}:{i}" for i in range(X.shape[1])], modality="rna_fm",
                     scale=scale, fitted_on="frozen", source=f"{model} frozen embedding ({model_dir})",
                     sha256=sha, **extra)


# ------------------------------------------------------------ view makers --
def _proposal_rows(episode) -> np.ndarray:
    return np.asarray(episode.partitions.proposal, dtype=np.int64)


def _select_raw_genes(logcpm, genes, proposal, n_raw):
    """Top-``n_raw`` genes by log1p-CPM variance on proposal rows only."""
    import scipy.sparse as sp

    Xp = logcpm[proposal]
    if sp.issparse(Xp):
        mean = np.asarray(Xp.mean(axis=0)).ravel()
        var = np.asarray(Xp.multiply(Xp).mean(axis=0)).ravel() - mean ** 2
    else:
        var = np.asarray(Xp, dtype=np.float64).var(axis=0)
    keep = np.sort(np.argsort(-var, kind="stable")[:min(n_raw, len(var))])
    sub = logcpm[:, keep]
    X = np.asarray(sub.todense() if sp.issparse(sub) else sub, dtype=np.float32)
    return X, [str(genes[i]) for i in keep]


def _standardise(X, proposal):
    mu = X[proposal].mean(axis=0)
    sd = X[proposal].std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def _pca_view(raw, proposal, n_components, seed):
    from sklearn.utils.extmath import randomized_svd

    Z = _standardise(raw.astype(np.float64), proposal)
    k = int(min(n_components, len(proposal) - 1, Z.shape[1]))
    _, s, vt = randomized_svd(Z[proposal], n_components=k, n_iter=5, random_state=seed,
                              flip_sign=True)
    scores = (Z @ vt.T).astype(np.float32)
    var_ratio = (s ** 2) / max(float((Z[proposal] ** 2).sum()), 1e-12)
    return scores, var_ratio, vt


def _nmf_view(raw, proposal, n_components, seed, genes):
    from sklearn.decomposition import NMF

    k = int(min(n_components, len(proposal) - 1, raw.shape[1]))
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400, tol=1e-4)
    nmf.fit(np.maximum(raw[proposal], 0.0))
    W = nmf.transform(np.maximum(raw, 0.0)).astype(np.float32)
    top = [[genes[j] for j in np.argsort(-row)[:10]] for row in nmf.components_]
    return W, top, nmf.components_


def score_gene_sets(logcpm, genes, gene_sets: Sequence[Mapping[str, Any]], prefix: str):
    """Mean log1p-CPM over each set's genes present in the full panel (the
    same scoring ``hest.load`` uses for the Gavish metaprograms)."""
    import scipy.sparse as sp

    pos = {str(g): i for i, g in enumerate(genes)}
    out = np.zeros((logcpm.shape[0], len(gene_sets)), dtype=np.float32)
    names, matched = [], []
    for j, gs in enumerate(gene_sets):
        cols = sorted({pos[g] for g in gs.get("genes", []) if g in pos})
        names.append(f"{prefix}:{gs.get('name', gs.get('mp_id', j))}")
        matched.append(len(cols))
        if cols:
            block = logcpm[:, cols]
            out[:, j] = np.asarray(block.mean(axis=1)).ravel() if sp.issparse(block) else block.mean(axis=1)
    return out, names, matched


def read_gmt(path: str, prefix_filter: str | None = None) -> list[dict]:
    sets = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or (prefix_filter and not parts[0].startswith(prefix_filter)):
                continue
            sets.append({"name": parts[0], "genes": [g for g in parts[2:] if g]})
    return sets


def load_hallmark_sets(path: str | None = None) -> list[dict]:
    """The 50 MSigDB Hallmark gene sets (public GMT), or [] if no copy is found.

    Looks at ``path``, ``$MORPHEUS_HALLMARK_GMT``, the store copy next to the
    embedding caches, then the repository's MSigDB discovery GMT."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [path, os.environ.get("MORPHEUS_HALLMARK_GMT"),
                  os.path.join(DEFAULT_CACHE_ROOT, "genesets", "msigdb_hallmark_2024.1.Hs.gmt"),
                  os.path.join(here, "cscd", "w5_data", "msigdb_discovery_2024.1.Hs.gmt")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return read_gmt(candidate, prefix_filter="HALLMARK_")
    return []


def _neighbour_mean(values, neighbours):
    out = np.empty_like(values, dtype=np.float32)
    for i, idx in enumerate(neighbours):
        idx = np.asarray(idx, dtype=np.int64)
        out[i] = values[idx].mean(axis=0) if idx.size else values[i]
    return out


def _keep_columns(view, keep):
    keep = np.asarray(keep, dtype=np.int64)
    view["X"] = view["X"][:, keep]
    view["columns"] = [view["columns"][i] for i in keep]
    if "match_abs_r" in view:
        view["match_abs_r"] = [view["match_abs_r"][i] for i in keep]
    if "top_genes" in view and len(view["top_genes"]) > int(keep.max(initial=-1)):
        view["top_genes"] = [view["top_genes"][i] for i in keep]
    if view.get("sha256") is not None:
        view["sha256"] = None


def _onehot(labels, prefix):
    labels = np.asarray([str(x) for x in labels])
    cats = sorted(set(labels.tolist()))
    X = (labels[:, None] == np.asarray(cats)[None, :]).astype(np.float32)
    return X, [f"{prefix}:{c}" for c in cats]


# ------------------------------------------- v0.5: identity, pruning, guard --
DECOMPOSITION_REFERENCE_VERSION = "decomposition_reference_v1_20261013"
MATCH_MIN_ABS_R = 0.6


def align_components(scores, loadings, genes, ref_loadings, ref_genes, *, prefix, signed=True,
                     min_abs_r=MATCH_MIN_ABS_R):
    """Align one fold's decomposition to a reference fit.

    ``loadings`` (k x G) over ``genes`` and ``ref_loadings`` (k_ref x G_ref)
    over ``ref_genes`` are placed on the union gene axis (absent genes = 0)
    and matched one-to-one by maximum |correlation| (Hungarian assignment);
    with ``signed`` a negatively matched component is sign-flipped (scores and
    loading). Columns are returned in reference order and named
    ``<prefix><ref index + 1>``; a match with |r| < ``min_abs_r`` gets the
    suffix ``_unmatched``, and a component left without a reference partner is
    named by its own index with that suffix. Returns ``(scores, names,
    quality, source)`` where ``quality[j]`` is the matched |r| and
    ``source[j]`` the fold's own component index now in position ``j``.
    """
    from scipy.optimize import linear_sum_assignment

    union = sorted(set(map(str, genes)) | set(map(str, ref_genes)))
    pos = {g: i for i, g in enumerate(union)}

    def _embed(L, gs):
        out = np.zeros((L.shape[0], len(union)))
        out[:, [pos[str(g)] for g in gs]] = L
        return out

    cur, ref = _embed(np.asarray(loadings, float), genes), _embed(np.asarray(ref_loadings, float), ref_genes)
    cur_c = cur - cur.mean(1, keepdims=True)
    ref_c = ref - ref.mean(1, keepdims=True)
    denom = np.outer(np.linalg.norm(cur_c, axis=1), np.linalg.norm(ref_c, axis=1))
    r = (cur_c @ ref_c.T) / np.maximum(denom, 1e-300)
    score = np.abs(r) if signed else r
    rows, cols = linear_sum_assignment(-score)
    scores = np.asarray(scores)
    out_cols, names, quality, order, src = [], [], [], [], []
    for i, j in zip(rows, cols):
        sign = -1.0 if (signed and r[i, j] < 0) else 1.0
        q = float(abs(r[i, j]))
        order.append(j)
        out_cols.append(sign * scores[:, i])
        names.append(f"{prefix}{j + 1}" + ("" if q >= min_abs_r else "_unmatched"))
        quality.append(q)
        src.append(int(i))
    for i in sorted(set(range(cur.shape[0])) - set(rows.tolist())):
        order.append(10 ** 6 + i)
        out_cols.append(scores[:, i])
        names.append(f"{prefix}{i + 1}_unmatched")
        quality.append(0.0)
        src.append(int(i))
    idx = np.argsort(order, kind="stable")
    X = np.column_stack([out_cols[i] for i in idx]).astype(scores.dtype, copy=False)
    return X, [names[i] for i in idx], [quality[i] for i in idx], [src[i] for i in idx]


def _decomposition_reference(kind, sources, cache_root, cohort, *, n_raw_genes, n_components, seed):
    """Fold-0 reference fit (``sources["reference_proposal_rows"]``) of the PCA or
    NMF decomposition, cached as ``<cache_root>/<cohort>/decomposition_reference/
    <key>__<kind>.npz`` with a sidecar sha256 (verified on read; a mismatch
    refits). The key covers the reference rows' identity
    (``sources["reference_key"]``), the decomposition parameters and the code
    version, so every fold of a run aligns to the same reference fit."""
    ref_rows = np.asarray(sources["reference_proposal_rows"], dtype=np.int64)
    record = {"kind": kind, "cohort": cohort, "reference_key": str(sources["reference_key"]),
              "n_raw_genes": int(n_raw_genes), "n_components": int(n_components), "seed": int(seed),
              "version": DECOMPOSITION_REFERENCE_VERSION}
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]
    folder = os.path.join(cache_root, cohort, "decomposition_reference")
    path = os.path.join(folder, f"{key}__{kind}.npz")
    side = path + ".json"
    if os.path.exists(path) and os.path.exists(side):
        with open(side) as fh:
            meta = json.load(fh)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest == meta.get("sha256"):
            with np.load(path, allow_pickle=False) as z:
                return z["loadings"], [str(g) for g in z["genes"]], key
    raw, genes = _select_raw_genes(sources["logcpm"], sources["genes"], ref_rows, n_raw_genes)
    if kind == "pca":
        _, _, loadings = _pca_view(raw, ref_rows, n_components, seed)
    else:
        _, _, loadings = _nmf_view(raw, ref_rows, n_components, seed, genes)
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, loadings=np.asarray(loadings, dtype=np.float64), genes=np.asarray(genes))
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(f"{side}.{os.getpid()}.tmp", "w") as fh:
        json.dump({**record, "sha256": digest}, fh, indent=1, sort_keys=True)
    os.replace(f"{side}.{os.getpid()}.tmp", side)
    return np.asarray(loadings, dtype=np.float64), list(genes), key


def within_group_sd(X, groups, rows):
    """Pooled within-group sd of every column over ``rows`` (NaN-aware)."""
    X = np.asarray(X, dtype=np.float64)[rows]
    g = np.asarray(groups)[rows]
    ss = np.zeros(X.shape[1]); n = np.zeros(X.shape[1])
    for code in np.unique(g):
        block = X[g == code]
        ok = np.isfinite(block)
        cnt = ok.sum(0)
        mean = np.where(cnt > 0, np.nansum(block, 0) / np.maximum(cnt, 1), 0.0)
        ss += np.nansum((block - mean) ** 2, 0)
        n += cnt
    return np.sqrt(ss / np.maximum(n, 1))


def within_group_corr(X, ref, groups, rows):
    """Within-group (patient) correlation of every column with ``ref`` over
    ``rows``: both sides standardised within group, then averaged products."""
    X = np.asarray(X, dtype=np.float64)[rows]
    y = np.asarray(ref, dtype=np.float64)[rows]
    g = np.asarray(groups)[rows]
    num = np.zeros(X.shape[1]); n = 0
    for code in np.unique(g):
        m = g == code
        yb = y[m] - y[m].mean()
        ysd = yb.std()
        if ysd < 1e-12:
            continue
        xb = np.nan_to_num(X[m] - np.nanmean(X[m], 0))
        xsd = xb.std(0)
        num += (xb * (yb / ysd)[:, None]).sum(0) / np.where(xsd < 1e-12, np.inf, xsd)
        n += int(m.sum())
    return num / max(n, 1)


# ---------------------------------------------------------------- builder --
def build_bank(episode, views: Sequence[str] | None = None, cache_root: str | None = None,
               sources: Mapping[str, Any] | None = None, **kwargs) -> dict:
    """The bank dict only; see :func:`build_bank_with_report`."""
    return build_bank_with_report(episode, views=views, cache_root=cache_root,
                                  sources=sources, **kwargs)[0]


def build_bank_with_report(episode, views: Sequence[str] | None = None,
                           cache_root: str | None = None, sources: Mapping[str, Any] | None = None,
                           *, n_raw_genes: int = 2000, n_pca: int = 50, n_nmf: int = 20,
                           seed: int = 0, align_decompositions: bool = False,
                           near_constant_sd: float | None = None) -> tuple[dict, dict]:
    """Build the protected measurement bank for one episode.

    ``sources`` holds row-aligned raw inputs supplied by the substrate loader
    (all arrays already restricted to the episode's rows, in its order):

    ``cohort`` ("hest_ccrcc" | "bmmc"), ``logcpm`` (n x G sparse/dense full
    gene panel, log1p CPM), ``genes``, ``gene_sets`` ({prefix: [set, ...]}),
    ``fm_groups``/``fm_barcodes`` (cache keys), ``fm_single_file`` (BMMC),
    ``fm_caveat``, ``fm_scale``, ``hoptimus`` ((E, covered)), ``cellvit``
    (the ``load_cellvit_multiscale_features`` dict), ``coords``, ``patient``,
    ``donor``, ``site``, ``neighbours``, and ``exclude`` ({view: {"columns":
    [...] or None (whole view), "reason": str}}).

    ``views=None`` requests the cohort's default set; a default view whose
    source or cache is absent is reported as unavailable, while an explicitly
    requested one raises.

    v0.5 options (all off by default, so a v0.4 bank is unchanged):
    ``align_decompositions`` aligns the PCA and NMF views (and the PCA part of
    the neighbour means) to a fold-0 reference fit (:func:`align_components`;
    needs ``sources["reference_proposal_rows"]``/``["reference_key"]``) and
    records per-column match quality; ``near_constant_sd`` drops every
    non-identity view column whose pooled within-group sd on proposal rows is
    below it (``sources["groups"]``), counted per view in
    ``report["near_constant_dropped"]``; ``sources["leak_guard"]`` ({"reference":
    (n,), "threshold": float, "views": [...] or None, "reason": str}) drops
    every column whose within-group |r| with the reference on proposal rows
    reaches the threshold, listed with its r in ``report["leak_guard"]``;
    ``sources["programmes_precomputed"]`` ((X, names, matched)) reuses the
    loader's own gene-set scores.
    """
    sources = dict(sources or {})
    cohort = sources.get("cohort", "hest_ccrcc")
    explicit = views is not None
    views = tuple(views) if explicit else (BMMC_VIEWS if cohort == "bmmc" else HEST_VIEWS)
    cache_root = cache_root or DEFAULT_CACHE_ROOT
    n = episode.n_rows
    proposal = _proposal_rows(episode)
    exclude = dict(sources.get("exclude") or {})
    bank: dict[str, dict] = {}
    report = {"schema": BANK_SCHEMA, "cohort": cohort, "unavailable": {}, "excluded": {},
              "n_rows": int(n), "n_proposal_rows": int(len(proposal)), "cache_root": cache_root}

    def _need(key):
        if key not in sources or sources[key] is None:
            raise KeyError(f"bank source {key!r} not supplied by the substrate loader")
        return sources[key]

    raw = raw_cols = None
    pca = None
    prog = None

    def _raw():
        nonlocal raw, raw_cols
        if raw is None:
            raw, raw_cols = _select_raw_genes(_need("logcpm"), _need("genes"), proposal, n_raw_genes)
        return raw, raw_cols

    def _pca():
        """(scores, var_ratio, names, match quality or None)."""
        nonlocal pca
        if pca is None:
            r, genes = _raw()
            scores, var_ratio, vt = _pca_view(r, proposal, n_pca, seed)
            names, quality = [f"PC{i + 1}" for i in range(scores.shape[1])], None
            if align_decompositions:
                ref, ref_genes, key = _decomposition_reference(
                    "pca", sources, cache_root, cohort, n_raw_genes=n_raw_genes, n_components=n_pca,
                    seed=seed)
                scores, names, quality, source = align_components(scores, vt, genes, ref, ref_genes,
                                                                  prefix="PC")
                var_ratio = np.asarray(var_ratio)[source]
                report.setdefault("decomposition_reference", {})["pca"] = key
            pca = (scores, var_ratio, names, quality)
        return pca

    def _prog():
        nonlocal prog
        if prog is None and sources.get("programmes_precomputed") is not None:
            X, names, matched = sources["programmes_precomputed"]
            prog = (np.asarray(X, dtype=np.float32), list(names), list(matched))
        if prog is None:
            blocks, names, matched = [], [], []
            for prefix, sets in (_need("gene_sets") or {}).items():
                X, nm, mt = score_gene_sets(_need("logcpm"), _need("genes"), sets, prefix)
                blocks.append(X); names += nm; matched += mt
            if not blocks:
                raise KeyError("no gene sets supplied for programme scores")
            prog = (np.column_stack(blocks).astype(np.float32), names, matched)
        return prog

    for name in views:
        try:
            if name == "raw_genes":
                X, cols = _raw()
                view = make_view(X, cols, modality="rna", scale=sources.get("fm_scale", "spot"),
                                 fitted_on="proposal",
                                 source=f"log1p CPM (1e4); top {len(cols)} genes by proposal-row variance")
            elif name == "pca":
                scores, var_ratio, pc_names, quality = _pca()
                extra = {} if quality is None else {"match_abs_r": [float(q) for q in quality],
                                                    "aligned_to": "fold-0 reference fit"}
                view = make_view(scores, pc_names,
                                 modality="rna", scale=sources.get("fm_scale", "spot"), fitted_on="proposal",
                                 source="PCA of proposal-standardised raw_genes, fitted on proposal rows",
                                 explained_variance_ratio=[float(v) for v in var_ratio], **extra)
            elif name == "nmf":
                r, cols = _raw()
                W, top, comps = _nmf_view(r, proposal, n_nmf, seed, cols)
                nmf_names, extra = [f"NMF{i + 1}" for i in range(W.shape[1])], {}
                if align_decompositions:
                    ref, ref_genes, key = _decomposition_reference(
                        "nmf", sources, cache_root, cohort, n_raw_genes=n_raw_genes, n_components=n_nmf,
                        seed=seed)
                    W, nmf_names, quality, source = align_components(W, comps, cols, ref, ref_genes,
                                                                     prefix="NMF", signed=False)
                    top = [top[i] for i in source]  # top genes follow their component
                    extra = {"match_abs_r": [float(q) for q in quality], "aligned_to": "fold-0 reference fit"}
                    report.setdefault("decomposition_reference", {})["nmf"] = key
                view = make_view(W, nmf_names, modality="rna",
                                 scale=sources.get("fm_scale", "spot"), fitted_on="proposal",
                                 source="NMF of raw_genes log1p CPM, fitted on proposal rows",
                                 top_genes=top, **extra)
            elif name == "programmes":
                X, names, matched = _prog()
                view = make_view(X, names, modality="rna", scale=sources.get("fm_scale", "spot"),
                                 fitted_on="frozen",
                                 source="mean log1p CPM over published gene sets (full gene panel)",
                                 n_genes_matched=matched)
            elif name == "hoptimus":
                E, cov = _need("hoptimus")
                X = np.asarray(E, dtype=np.float32).copy()
                cov = np.asarray(cov, dtype=bool)
                X[~cov] = np.nan
                view = make_view(X, [f"hoptimus:{i}" for i in range(X.shape[1])], modality="he_fm",
                                 scale="spot", fitted_on="frozen",
                                 source="H-Optimus-0 spot-patch embedding (cached)",
                                 covered=cov, coverage_fraction=float(cov.mean()))
            elif name == "cellvit_structure":
                cv = _need("cellvit")
                classes = list(sources.get("cellvit_classes") or
                               [f"class{i}" for i in range(cv["composition"].shape[1] - 1)])
                cols = (["nuclear:log1p_count", "nuclear:log1p_mean_area", "nuclear:area_cv"]
                        + [f"composition:frac_{c}" for c in classes] + ["composition:class_entropy"]
                        + ["graph:same_class_fraction", "graph:heterotypic_fraction",
                           "graph:mean_degree", "graph:mean_neighbour_distance"]
                        + ["boundary:neoplastic_connective", "boundary:neoplastic_inflammatory",
                           "boundary:neoplastic_epithelial"])
                X = np.column_stack([cv["nuclear"], cv["composition"], cv["graph"],
                                     cv["boundary"]]).astype(np.float32)
                cov = np.asarray(cv["covered"], dtype=bool)
                X[~cov] = np.nan
                view = make_view(X, cols, modality="he_structure", scale="spot", fitted_on="frozen",
                                 source="CellViT nuclei within each spot radius; sample-local kNN cell graph",
                                 covered=cov, coverage_fraction=float(cov.mean()))
            elif name == "context_neighbour_mean":
                nb = _need("neighbours")
                pieces, cols = [], []
                if sources.get("gene_sets"):
                    X, names, _ = _prog()
                    pieces.append(X); cols += names
                if "logcpm" in sources:
                    scores, _, pc_names, _ = _pca()
                    pieces.append(scores); cols += list(pc_names)
                base = np.column_stack(pieces)
                X = _neighbour_mean(base, nb)
                view = make_view(X, [f"nbr:{c}" for c in cols], modality="spatial_context",
                                 scale="neighbourhood", fitted_on="proposal",
                                 source="mean over within-patient kNN neighbours (self excluded) of programmes and PCA")
            elif name == "spatial_operators":
                # The MDE physical operator bank is opt-in: older MORPHEUS
                # experiments retain their exact candidate bank, while a new
                # registered campaign can request it explicitly.  Inputs are
                # measured columns only and may never be defined from y.
                from .spatial_operator_bank import build_operator_bank
                values = np.asarray(_need("spatial_operator_values"), dtype=float)
                names = list(_need("spatial_operator_names"))
                patient = sources.get("patient", sources.get("donor"))
                if patient is None:
                    raise KeyError("spatial_operators requires patient or donor ids")
                section = sources.get("section", sources.get("site"))
                if section is None:
                    # Patient is the conservative fallback: never connect
                    # observations from different patients, even if section
                    # identifiers have not yet been loaded.
                    section = patient
                annuli = tuple(sources.get("spatial_operator_annuli", (0., 100., 250.)))
                X, operator_columns = build_operator_bank(
                    values, names, _need("coords"), patient, section, annuli=annuli,
                    heterogeneity_radius=float(sources.get("spatial_heterogeneity_radius", 100.)),
                    gradient_radius=float(sources.get("spatial_gradient_radius", 150.)),
                )
                cols = [x.name for x in operator_columns]
                view = make_view(X, cols, modality="spatial_context", scale="physical_distance",
                                 fitted_on="frozen",
                                 source="patient- and section-safe measured physical spatial operators",
                                 operator_provenance=[x.provenance for x in operator_columns])
            elif name == "coords":
                X = np.asarray(_need("coords"), dtype=np.float32)
                view = make_view(X, ["coord_x_px", "coord_y_px"][:X.shape[1]], modality="spatial_context",
                                 scale="spot", fitted_on="frozen", source="full-resolution pixel coordinates")
            elif name in ("patient", "donor", "site"):
                X, cols = _onehot(_need(name), name)
                view = make_view(X, cols, modality="context", scale="sample", fitted_on="frozen",
                                 source=f"{name} one-hot")
            elif name in FM_MODELS:
                model_dir = os.path.join(cache_root, cohort, name)
                view = fm_view_from_cache(model_dir, name, _need("fm_groups"), _need("fm_barcodes"),
                                          scale=sources.get("fm_scale", "spot"),
                                          single_file=sources.get("fm_single_file"),
                                          caveat=sources.get("fm_caveat"))
            else:
                raise ValueError(f"unknown bank view {name!r}")
        except (KeyError, FileNotFoundError) as exc:
            if explicit:
                raise
            report["unavailable"][name] = str(exc)
            continue

        rule = exclude.get(name)
        if rule is not None:
            drop = rule.get("columns")
            if drop is None:
                report["excluded"][name] = {"columns": "all", "reason": rule.get("reason", "")}
                continue
            keep = [i for i, c in enumerate(view["columns"]) if c not in set(drop)]
            removed = [c for c in view["columns"] if c in set(drop)]
            view["X"] = view["X"][:, keep]
            view["columns"] = [view["columns"][i] for i in keep]
            view["excluded_columns"] = removed
            if view.get("sha256") is not None:
                view["sha256"] = None  # no longer the cached array
            report["excluded"][name] = {"columns": removed, "reason": rule.get("reason", "")}
        if near_constant_sd is not None and name not in ("patient", "donor", "site"):
            sd = within_group_sd(view["X"], _need("groups"), proposal)
            keep = np.flatnonzero(sd >= near_constant_sd)
            dropped = [view["columns"][i] for i in np.flatnonzero(sd < near_constant_sd)]
            report.setdefault("near_constant_dropped", {})[name] = {
                "count": len(dropped), "of": len(view["columns"]), "columns": dropped,
                "rule": f"pooled within-group sd on proposal rows < {near_constant_sd}"}
            if dropped:
                _keep_columns(view, keep)
        guard = sources.get("leak_guard")
        if guard is not None and (guard.get("views") is None or name in guard["views"]) \
                and name not in ("patient", "donor", "site"):
            r = within_group_corr(view["X"], guard["reference"], _need("groups"), proposal)
            hit = np.flatnonzero(np.abs(r) >= float(guard["threshold"]))
            if hit.size:
                report.setdefault("leak_guard", {})[name] = {
                    "columns": {view["columns"][i]: round(float(r[i]), 4) for i in hit},
                    "threshold": float(guard["threshold"]), "reason": guard.get("reason", "")}
                _keep_columns(view, np.setdiff1d(np.arange(len(view["columns"])), hit))
        if view["X"].shape[0] != n:
            raise AssertionError(f"view {name} has {view['X'].shape[0]} rows; episode has {n}")
        bank[name] = view
    report["views"] = {k: [int(v["X"].shape[0]), int(v["X"].shape[1])] for k, v in bank.items()}
    report["nbytes"] = int(sum(v["X"].nbytes for v in bank.values()))
    return bank, report
