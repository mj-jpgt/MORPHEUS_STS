"""Episodes for the learned search model.

A *study* (:class:`Study`) holds response-free per-spot scalar columns (genes,
programme scores, nuclear morphology, ...), an optional frozen H-Optimus-0
spot embedding, patient codes and within-sample spatial neighbours.  An
*episode* (:class:`Episode`) is one scientific task drawn from studies:
support patients (seen by the search model) and query patients (scored only),
a target ``y`` and the candidate variables.

Candidate variables are four generic spatial transforms of every base scalar:

* ``own``       the spot's value;
* ``nbr6``      mean over the (up to) 6 nearest spots of the same sample;
* ``nbr18``     mean over the two-hop ring of the 6-neighbour graph (up to 18 spots);
* ``contrast``  own - nbr6.

Token names are ``<transform>:<base column name>``; concept tokens
(``<transform>:concept:<j>``) are built inside the model from the embedding.

Two kinds of episode, mixed 50/50 by :class:`EpisodeSampler`:

* ``real``        the target is a measured column (gene, programme, morphology)
                  or a morphology concept; candidates exclude the target and
                  columns built from it (e.g. a programme's member genes);
* ``controlled``  the target is ``r + g(V)``: ``r`` the cross-fitted
                  within-patient ridge residual of a real column on the
                  programme scores and their nbr6 / nbr18 means (the B3-style
                  comparison design; :func:`episode_residual`), ``g`` a random
                  relation over 1-4 true tokens from
                  ``dbench.benchmark.generators`` scaled to 0.5/1/2/5 % of
                  var(r) (``tasks.calibrate``).

Support and query patients are disjoint (:class:`EpisodeSampler`); with
probability ``cross_study_prob`` the query patients come from another study
that shares the candidate columns.  :class:`RowSplitSampler` (the ablation)
draws support and query rows from the same patients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

TRANSFORMS: tuple[str, ...] = ("own", "nbr6", "nbr18", "contrast")
VIEW_VOCAB: tuple[str, ...] = ("gene", "programme", "concept", "morphology", "fm", "other")
FRACTIONS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)   # development grid
TRAIN_FRACTION_RANGE: tuple[float, float] = (0.005, 0.20)              # training: log-uniform


def view_id(view_name: str) -> int:
    """Map any view name onto the small view vocabulary the model embeds."""
    v = str(view_name).lower()
    if "concept" in v:
        return VIEW_VOCAB.index("concept")
    if any(s in v for s in ("programme", "program", "gavish", "hallmark", "mp")):
        return VIEW_VOCAB.index("programme")
    if "gene" in v:
        return VIEW_VOCAB.index("gene")
    if any(s in v for s in ("cellvit", "nuclear", "morph", "nuc_")):
        return VIEW_VOCAB.index("morphology")
    if any(s in v for s in ("scgpt", "scimilarity", "fm", "pca", "nmf", "hoptimus")):
        return VIEW_VOCAB.index("fm")
    return VIEW_VOCAB.index("other")


# ------------------------------------------------------------------ geometry
def neighbour_operators(neighbours6: np.ndarray):
    """Row-normalised sparse (n, n) operators for nbr6 and the two-hop ring nbr18.

    A row with no neighbour keeps its own value (as ``substrate.neighbour_mean``)."""
    from scipy import sparse

    nb = np.asarray(neighbours6, dtype=np.int64)
    n = len(nb)
    valid = nb >= 0
    rows = np.repeat(np.arange(n), valid.sum(1))
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, nb[valid])), shape=(n, n))
    B.data[:] = 1.0
    R2 = (B + B @ B).tocsr()
    R2.setdiag(0)
    R2.eliminate_zeros()
    R2.data[:] = 1.0

    def normalise(M):
        M = M.tocsr().astype(np.float64)
        cnt = np.asarray(M.sum(1)).ravel()
        lonely = np.flatnonzero(cnt == 0)
        M = sparse.diags(1.0 / np.maximum(cnt, 1)) @ M
        if len(lonely):
            M = M + sparse.csr_matrix((np.ones(len(lonely)), (lonely, lonely)), shape=(n, n))
        return M.tocsr()

    return normalise(B), normalise(R2)


def ring2_indices(neighbours6: np.ndarray, width: int = 18) -> np.ndarray:
    """(n, width) -1-padded two-hop ring indices (never the row itself)."""
    _, A18 = neighbour_operators(neighbours6)
    n = A18.shape[0]
    out = -np.ones((n, width), dtype=np.int64)
    for i in range(n):
        idx = A18.indices[A18.indptr[i]:A18.indptr[i + 1]]
        idx = idx[idx != i][:width]
        out[i, :len(idx)] = idx
    return out


def knn_neighbours(coords: np.ndarray, groups: np.ndarray, k: int = 6) -> np.ndarray:
    """Within-group k-nearest neighbours by KD-tree, -1 padded."""
    from scipy.spatial import cKDTree

    out = -np.ones((len(coords), k), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        _, nn = cKDTree(coords[idx]).query(coords[idx], k=kk + 1)
        out[idx, :kk] = idx[nn[:, 1:]]
    return out


def transform_block(base_rows_ctx: np.ndarray, A6, A18, rows_local: np.ndarray) -> np.ndarray:
    """(r, 4m) [own | nbr6 | nbr18 | contrast] for ``rows_local`` of a context block.

    ``base_rows_ctx`` (c, m) holds the base columns on the context rows (every
    row of the chosen patients) and ``A6``/``A18`` the operators restricted to
    those rows, so neighbour means are exact."""
    own = base_rows_ctx[rows_local]
    n6 = np.asarray(A6[rows_local] @ base_rows_ctx)
    n18 = np.asarray(A18[rows_local] @ base_rows_ctx)
    return np.concatenate([own, n6, n18, own - n6], axis=1).astype(np.float32)


def centre_scale(M: np.ndarray, groups: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Within-patient centring (patient means from support rows where the patient has
    any, else from its own rows), then division by the support-row sd."""
    from scipy import sparse

    M = np.asarray(M)
    one = M.ndim == 1
    A = (M[:, None] if one else M).astype(np.float64 if one else np.float32, copy=True)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    n_g = int(inv.max()) + 1
    support = np.asarray(support, dtype=bool)
    has_sup = np.bincount(inv, weights=support.astype(float), minlength=n_g) > 0
    ref = support | ~has_sup[inv]                       # rows defining each patient's mean
    R = sparse.csr_matrix((np.ones(int(ref.sum()), dtype=A.dtype), (inv[ref], np.flatnonzero(ref))),
                          shape=(n_g, len(A)))
    cnt = np.maximum(np.asarray(R.sum(1)).ravel(), 1)
    means = np.asarray(R @ A) / cnt[:, None]
    A -= means[inv].astype(A.dtype)
    sd = A[support].std(0)
    sd[sd < 1e-8] = 1.0
    A /= sd.astype(A.dtype)
    return (A[:, 0] if one else A).astype(np.float32)


# ------------------------------------------------------------------- studies
@dataclass
class Study:
    """Response-free columns of one source study (one platform)."""

    name: str
    base: np.ndarray                 # (n, q) float32, within-patient centred
    names: list                      # q full column names (e.g. "raw_genes:COL1A1")
    views: dict                      # view name -> base column indices
    groups: np.ndarray               # (n,) patient codes (str or int)
    neighbours6: np.ndarray          # (n, 6) int64, -1 padded, within sample
    coords: np.ndarray | None = None
    embed: np.ndarray | None = None  # (n, D) float16/32 frozen H-Optimus-0 spot embedding
    platform: str = "visium"
    target_cols: list = field(default_factory=list)       # base columns usable as real targets
    extra_targets: dict = field(default_factory=dict)     # name -> (n,) measured target outside base
    extra_target_excluded_views: dict = field(default_factory=dict)  # name -> views excluded for it
    exclusions: dict = field(default_factory=dict)        # target col -> [base cols built from it]
    residual_cols: list = field(default_factory=list)     # programme columns for episode_residual
    subsample_views: dict = field(default_factory=dict)   # view -> max columns kept per episode (random)
    hvg_views: dict = field(default_factory=dict)         # view -> columns kept per episode (top support variance)
    train_patients: np.ndarray | None = None
    dev_patients: np.ndarray | None = None
    depth: np.ndarray | None = None  # (n,) log1p total counts (the nuisance covariate), when known

    def __post_init__(self):
        self.base = np.asarray(self.base, dtype=np.float32)
        self.groups = np.asarray(self.groups).astype(str)
        self.patients = np.unique(self.groups)
        if self.train_patients is None:
            self.train_patients = self.patients
        self.train_patients = np.asarray(self.train_patients).astype(str)
        if self.dev_patients is not None:
            self.dev_patients = np.asarray(self.dev_patients).astype(str)
        self.A6, self.A18 = neighbour_operators(self.neighbours6)
        self.index = {c: i for i, c in enumerate(self.names)}
        self.view_of = np.empty(len(self.names), dtype=object)
        for v, cols in self.views.items():
            self.view_of[np.asarray(cols, dtype=int)] = v
        self.rows_of = {p: np.flatnonzero(self.groups == p) for p in self.patients}
        if self.coords is None:
            self.coords = np.zeros((len(self.groups), 2), dtype=np.float32)

    @property
    def n(self) -> int:
        return len(self.groups)


def _hex_grid(m: int) -> np.ndarray:
    side = int(math.ceil(math.sqrt(m)))
    pts = [(i + 0.5 * (j % 2), j * math.sqrt(3) / 2) for j in range(side) for i in range(side)]
    return np.asarray(pts[:m], dtype=float)


def synthetic_study(seed: int = 0, *, n_patients: int = 12, rows_per_patient: int = 256, n_genes: int = 48,
                    n_programmes: int = 16, embed_dim: int = 64, n_factors: int = 6, name: str | None = None,
                    vocabulary_seed: int = 0, platform: str = "visium") -> Study:
    """A study with real-like structure: spatially smooth latent factors drive genes,
    programme scores (member-gene means) and a morphology embedding.

    Studies with the same ``vocabulary_seed`` share gene names, gene loadings and
    programme memberships, so cross-study episodes are meaningful."""
    vr = np.random.default_rng(10_000 + vocabulary_seed)
    W = vr.standard_normal((n_factors, n_genes)) * (vr.random((n_factors, n_genes)) < 0.35)
    members = [np.sort(vr.choice(n_genes, 6, replace=False)) for _ in range(n_programmes)]
    M1 = vr.standard_normal((n_factors, 32)) / np.sqrt(n_factors)
    M2 = vr.standard_normal((32, embed_dim)) / np.sqrt(32)
    rng = np.random.default_rng(seed)
    grid = _hex_grid(rows_per_patient)
    coords = np.concatenate([grid + 100.0 * p for p in range(n_patients)])
    groups = np.repeat(np.arange(n_patients), rows_per_patient).astype(str)
    nb6 = knn_neighbours(coords, groups, 6)
    A6, _ = neighbour_operators(nb6)
    F = rng.standard_normal((len(groups), n_factors))
    for _ in range(3):  # spatial smoothing: latent factors vary smoothly across the section
        F = 0.5 * F + 0.5 * np.asarray(A6 @ F)
    F /= F.std(0)
    G = F @ W + 0.7 * rng.standard_normal((len(groups), n_genes))
    G = 0.6 * G + 0.4 * np.asarray(A6 @ G)
    P = np.column_stack([G[:, m].mean(1) for m in members])
    E = np.tanh(F @ M1) @ M2 + 0.3 * rng.standard_normal((len(groups), embed_dim))
    base = np.column_stack([G, P])
    for g in np.unique(groups):
        r = groups == g
        base[r] -= base[r].mean(0)
    base /= base.std(0)
    gnames = [f"gene:G{j}" for j in range(n_genes)]
    pnames = [f"programme:P{j}" for j in range(n_programmes)]
    gene_cols = list(range(n_genes))
    prog_cols = list(range(n_genes, n_genes + n_programmes))
    excl = {n_genes + j: [int(c) for c in members[j]] for j in range(n_programmes)}
    return Study(name=name or f"synthetic{seed}", base=base, names=gnames + pnames,
                 views={"gene": gene_cols, "programme": prog_cols}, groups=groups, neighbours6=nb6,
                 coords=coords.astype(np.float32), embed=E.astype(np.float32), platform=platform,
                 target_cols=gene_cols + prog_cols, exclusions=excl, residual_cols=prog_cols)


def add_morph_view(study: Study, centres: np.ndarray) -> Study:
    """A copy of ``study`` with the ``morph`` view: cosine similarities of its embedding to
    fixed ``centres`` (within-patient centred), plus a synthetic log depth when none is known."""
    from .morphology_dictionary import similarities

    M = similarities(np.asarray(study.embed, dtype=np.float32), centres)
    for p in study.patients:
        r = study.groups == p
        M[r] -= M[r].mean(0)
    q = study.base.shape[1]
    views = {**study.views, "morph": list(range(q, q + M.shape[1]))}
    depth = study.depth
    if depth is None:
        depth = np.log1p(np.random.default_rng(len(study.groups)).gamma(4.0, 500.0, size=len(study.groups)))
    return Study(name=study.name, base=np.concatenate([study.base, M], 1),
                 names=list(study.names) + [f"morph:k{j:02d}" for j in range(M.shape[1])], views=views,
                 groups=study.groups, neighbours6=study.neighbours6, coords=study.coords, embed=None,
                 platform=study.platform, target_cols=list(study.target_cols), extra_targets=study.extra_targets,
                 extra_target_excluded_views=study.extra_target_excluded_views, exclusions=study.exclusions,
                 residual_cols=study.residual_cols, subsample_views=study.subsample_views,
                 hvg_views=study.hvg_views, train_patients=study.train_patients, dev_patients=study.dev_patients,
                 depth=depth)


def _programme_member_exclusions(names: Sequence[str], prog_cols: Sequence[int], gene_prefix: str) -> dict:
    """Programme column -> indices of its member genes present as base columns."""
    sets: dict[str, list] = {}
    try:
        from morpheus.tier2.substrates import hest
        for p in hest.load_gavish_programs():
            sets[f"gavish:{p['name']}"] = list(p["genes"])
    except Exception:  # pragma: no cover - box-only data
        pass
    try:
        from morpheus_encoder.measurement_bank import load_hallmark_sets
        for h in load_hallmark_sets():
            sets[f"hallmark:{h['name']}"] = list(h["genes"])
    except Exception:  # pragma: no cover
        pass
    index = {c: i for i, c in enumerate(names)}
    out = {}
    for c in prog_cols:
        key = str(names[c]).split(":", 1)[1] if ":" in str(names[c]) else str(names[c])
        genes = sets.get(key, [])
        out[int(c)] = [index[f"{gene_prefix}{g}"] for g in genes if f"{gene_prefix}{g}" in index]
    return out


def study_from_bench_substrate(sub, target_cols: Sequence[str] | None = None, *, name: str = "kidney_fold0",
                               n_hvg: int = 768) -> Study:
    """The v0.5 kidney benchmark substrate as a :class:`Study` (no embedding in the cache).

    Base columns: ``raw_genes`` (subsampled to ``n_hvg`` per episode), the 91
    ``programmes`` and the 14 ``cellvit_structure`` columns.  Real targets: the
    substrate's own target (``nuc_neoplastic_area``; the CellViT view is excluded
    for it), the programmes (member genes excluded) and the CellViT columns.
    Training patients are the substrate's proposal patients, development
    patients its held-out patients."""
    keep_views = [v for v in ("raw_genes", "programmes", "cellvit_structure") if v in sub.views]
    cols = np.concatenate([np.asarray(sub.views[v], dtype=int) for v in keep_views])
    names = [str(sub.names[c]) for c in cols]
    views, start = {}, 0
    for v in keep_views:
        w = len(sub.views[v])
        views[v] = list(range(start, start + w))
        start += w
    base = np.asarray(sub.X[:, cols], dtype=np.float32)
    groups = np.asarray(sub.groups).astype(str)
    if target_cols is None:
        tcols = views.get("programmes", []) + views.get("cellvit_structure", [])
    else:
        idx = {c: i for i, c in enumerate(names)}
        tcols = [idx[c] for c in target_cols]
    excl = _programme_member_exclusions(names, views.get("programmes", []), "raw_genes:")
    target_name = str(sub.meta.get("target", "target"))
    return Study(name=name, base=base, names=names, views=views, groups=groups,
                 neighbours6=np.asarray(sub.neighbours, dtype=np.int64),
                 coords=None if sub.coords is None else np.asarray(sub.coords, dtype=np.float32),
                 embed=None, platform="visium", target_cols=list(tcols),
                 extra_targets={target_name: np.asarray(sub.y_real, dtype=np.float32)},
                 extra_target_excluded_views={target_name: ["cellvit_structure"]},
                 exclusions=excl, residual_cols=list(views.get("programmes", [])),
                 subsample_views={"raw_genes": n_hvg},
                 train_patients=np.unique(groups[np.asarray(sub.proposal)]),
                 dev_patients=np.unique(groups[np.asarray(sub.heldout)]))


def study_from_arrays(X: np.ndarray, groups: np.ndarray, neighbours6: np.ndarray | None, names: Sequence[str],
                      views: dict | None, *, embed: np.ndarray | None = None, y: np.ndarray | None = None,
                      coords: np.ndarray | None = None, name: str = "task") -> Study:
    """A single-task study from raw arrays (``propose_sets`` and the per-task ablation)."""
    X = np.asarray(X, dtype=np.float32)
    groups = np.asarray(groups).astype(str)
    if neighbours6 is None:
        neighbours6 = -np.ones((len(X), 6), dtype=np.int64)
    if not views:
        views = {"other": list(range(X.shape[1]))}
    base = X.copy()
    for g in np.unique(groups):
        r = groups == g
        base[r] -= base[r].mean(0)
    return Study(name=name, base=base, names=[str(c) for c in names],
                 views={k: [int(i) for i in v] for k, v in views.items()}, groups=groups,
                 neighbours6=np.asarray(neighbours6, dtype=np.int64), coords=coords, embed=embed,
                 extra_targets={} if y is None else {"task": np.asarray(y, dtype=np.float32)})


def study_from_hest_raw(raw, *, name: str, platform: str = "visium", n_gene_pool: int = 2000,
                        n_hvg: int = 768, n_gene_targets: int = 200, morph_dictionary: str | None = None) -> Study:
    """A ``hest_multi.StudyRaw`` (Agent D's per-study cache) as a :class:`Study`.

    Base columns: a response-free pool of the ``n_gene_pool`` most variable genes
    (within-patient variance of log1p CPM over the study), of which each episode
    keeps the ``n_hvg`` most variable on its SUPPORT rows (target excluded), plus
    the 91 programme scores (full panel).  Real targets: the programmes (member
    genes excluded) and the ``n_gene_targets`` most variable pool genes.

    With ``morph_dictionary`` (learned search v02) and an embedding, rows are
    restricted to embedding-covered spots (nothing imputed) and the 64
    ``morph:kNN`` cosine similarities to the frozen centres are added as the
    ``morph`` view (candidates; targets via ``morph_target_prob``); the raw
    embedding is then not carried (no concept layer)."""
    from morpheus.tier2.substrates import hest_multi as hm

    keep = None
    if morph_dictionary is not None and raw.hoptimus is not None:
        keep = np.flatnonzero(np.asarray(raw.covered, dtype=bool))
    groups = np.asarray(raw.patient).astype(str)
    L = raw.logcpm()
    if keep is not None:
        groups, L = groups[keep], L[keep]
    pats = np.unique(groups)
    ss = np.zeros(L.shape[1])
    for p in pats:
        rows = np.flatnonzero(groups == p)
        sub = L[rows]
        m = np.asarray(sub.mean(0)).ravel()
        ss += np.asarray(sub.multiply(sub).sum(0)).ravel() - len(rows) * m * m
    var = ss / max(len(groups), 1)
    pool = np.sort(np.argsort(-var, kind="stable")[:n_gene_pool])
    G = np.asarray(L[:, pool].todense(), dtype=np.float32)
    rows = slice(None) if keep is None else keep
    P = np.asarray(raw.programmes, dtype=np.float32)[rows]
    genes = [str(g) for g in np.asarray(raw.genes)[pool]]
    names = [f"gene:{g}" for g in genes] + [f"programme:{c}" for c in raw.programme_names]
    blocks = [G, P]
    views = {"gene": list(range(len(genes))), "programme": list(range(len(genes), len(names)))}
    if keep is not None:
        from .morphology_dictionary import morph_view
        M, mnames = morph_view(np.asarray(raw.hoptimus, dtype=np.float32)[keep], morph_dictionary)
        blocks.append(M)
        views["morph"] = list(range(len(names), len(names) + len(mnames)))
        names += mnames
    base = np.concatenate(blocks, axis=1)
    for p in pats:
        r = groups == p
        base[r] -= base[r].mean(0)
    gcols, pcols = views["gene"], views["programme"]
    index = {c: i for i, c in enumerate(names)}
    excl = {}
    for j, c in zip(pcols, raw.programme_names):
        excl[j] = [index[f"gene:{g}"] for g in hm._programme_members(str(c)) if f"gene:{g}" in index]
    order = np.argsort(-var[pool], kind="stable")[:n_gene_targets]
    nb6 = np.asarray(raw.neighbours[6], dtype=np.int64)[:, :6]
    E = None if raw.hoptimus is None else np.asarray(raw.hoptimus, dtype=np.float16)
    if keep is not None:
        remap = -np.ones(raw.n, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        nb6, E = hm._remap_neighbours(nb6, keep, remap), None
    return Study(name=name, base=base, names=names, views=views, groups=groups, neighbours6=nb6,
                 coords=np.asarray(raw.coords, dtype=np.float32)[rows], embed=E, platform=platform,
                 target_cols=pcols + [int(gcols[i]) for i in order], exclusions=excl, residual_cols=pcols,
                 hvg_views={"gene": n_hvg}, depth=np.log1p(np.asarray(raw.depth, dtype=np.float64))[rows])


def studies_from_partitions(partitions_json: str, *, min_patients: int = 2, only: set | None = None,
                            **kw) -> tuple[list, list, dict]:
    """Meta-training studies (train) and development studies (dev) of the frozen
    partitions file through ``hest_multi``; studies with fewer than
    ``min_patients`` patients or that fail to load are skipped and listed."""
    from morpheus.tier2.substrates import hest_multi as hm

    info: dict = {"partitions": partitions_json, "loaded": {}, "skipped": {}}
    out = {"meta_training": [], "development": []}
    for role in out:
        for key, spec in sorted(hm.study_specs(partitions_json, roles=[role]).items()):
            if only is not None and role == "meta_training" and key not in only:
                continue
            try:
                raw = hm.load_study(spec)
                n_pat = len(np.unique(raw.patient))
                if n_pat < min_patients:
                    info["skipped"][key] = f"{n_pat} patient(s)"
                    continue
                st = study_from_hest_raw(raw, name=key, platform=spec.platform, **kw)
                out[role].append(st)
                info["loaded"][key] = {"role": role, "n_rows": st.n, "n_patients": n_pat,
                                       "embedding": st.embed is not None, "platform": spec.platform}
            except Exception as exc:  # report and continue
                info["skipped"][key] = f"{type(exc).__name__}: {exc}"[:200]
            hm._RAW_MEMO.clear()
    return out["meta_training"], out["development"], info


# ------------------------------------------------------------------ episodes
@dataclass
class TokenMeta:
    names: list                 # token names "<transform>:<base name>"
    transform: np.ndarray       # (p,) int64 index into TRANSFORMS
    view: np.ndarray            # (p,) int64 index into VIEW_VOCAB


@dataclass
class Episode:
    X: np.ndarray               # (n, p) float32 candidate tokens' values, support rows first
    y: np.ndarray | None        # (n,) float32; None when the target is a concept (built in the model)
    n_support: int
    groups: np.ndarray          # (n,) int64 patient codes (episode-local)
    coords: np.ndarray          # (n, 2) float32, per-sample standardised
    meta: TokenMeta
    kind: str = "real"          # "real" | "controlled"
    family: str | None = None
    fraction: float | None = None
    true_idx: tuple | None = None   # token indices of the planted set (controlled)
    concept: dict | None = None     # {"E": (c, D), "own": (n,), "nb6": (n, 6), "nb18": (n, 18)} local indices
    concept_target: int | None = None
    target_name: str = ""
    study: str = ""
    query_study: str = ""
    support_patients: tuple = ()
    query_patients: tuple = ()
    info: dict = field(default_factory=dict)
    nb6: np.ndarray | None = None       # (n, 6) episode-local neighbours (-1 padded; kept rows only)
    nuisance: np.ndarray | None = None  # (n,) log depth, centred within patient

    @property
    def n(self) -> int:
        return len(self.groups)

    @property
    def support(self) -> np.ndarray:
        return np.arange(self.n_support)

    @property
    def query(self) -> np.ndarray:
        return np.arange(self.n_support, self.n)

    def true_names(self) -> tuple:
        return tuple(sorted(self.meta.names[i] for i in (self.true_idx or ())))


def episode_residual(y: np.ndarray, D: np.ndarray, groups: np.ndarray, *, alpha: float = 10.0,
                     n_folds: int = 5) -> np.ndarray:
    """Cross-fitted within-patient ridge residual of ``y`` on ``D``.

    Rows and columns are centred within patient; folds are whole patients; each
    fold's rows are predicted by the ridge fit on the other folds."""
    y = np.asarray(y, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    yc, Dc = y.copy(), D.copy()
    pats = np.unique(groups)
    for g in pats:
        r = groups == g
        yc[r] -= yc[r].mean()
        Dc[r] -= Dc[r].mean(0)
    if D.shape[1] == 0 or len(pats) < 2:
        return yc
    fold_of = {g: i % min(n_folds, len(pats)) for i, g in enumerate(pats)}
    fold = np.array([fold_of[g] for g in groups])
    out = np.empty_like(yc)
    for k in np.unique(fold):
        te = fold == k
        A = Dc[~te]
        sd = A.std(0)
        sd[sd < 1e-12] = 1.0
        A = A / sd
        beta = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ yc[~te])
        out[te] = yc[te] - (Dc[te] / sd) @ beta
    return out


def _coords_std(coords: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float64).copy()
    for g in np.unique(groups):
        r = groups == g
        out[r] -= out[r].mean(0)
        s = out[r].std()
        out[r] /= s if s > 1e-12 else 1.0
    return out.astype(np.float32)


@dataclass
class _Block:
    X: np.ndarray        # (r, 4m)
    target: np.ndarray | None
    resid_design: np.ndarray
    groups: np.ndarray   # str codes
    coords: np.ndarray
    rows: np.ndarray     # study row indices
    E: np.ndarray | None
    own: np.ndarray | None
    nb6: np.ndarray | None
    nb18: np.ndarray | None
    nbk: np.ndarray | None = None    # (r, 6) neighbours among the kept rows (block-local), -1 padded
    depth: np.ndarray | None = None  # (r,) log depth


def _kept_neighbours(study: "Study", rows: np.ndarray) -> np.ndarray:
    pos = -np.ones(study.n, dtype=np.int64)
    pos[rows] = np.arange(len(rows))
    nb = study.neighbours6[rows]
    return np.where(nb >= 0, pos[np.maximum(nb, 0)], -1)


def _block(study: Study, patients: Sequence[str], cols: np.ndarray, rng: np.random.Generator, *,
           target: np.ndarray | None, resid_cols: np.ndarray, max_rows_per_patient: int,
           with_concepts: bool) -> _Block:
    """Token values for the chosen patients: neighbour means over every row of
    those patients, then rows subsampled per patient."""
    ctx = np.concatenate([study.rows_of[p] for p in patients])
    keep = []
    offset = 0
    for p in patients:
        m = len(study.rows_of[p])
        local = np.arange(offset, offset + m)
        if m > max_rows_per_patient:
            local = np.sort(rng.choice(local, max_rows_per_patient, replace=False))
        keep.append(local)
        offset += m
    keep = np.concatenate(keep)
    A6 = study.A6[ctx][:, ctx]
    A18 = study.A18[ctx][:, ctx]
    base_ctx = study.base[ctx][:, cols]
    X = transform_block(base_ctx, A6, A18, keep)
    rows = ctx[keep]
    E = own = nb6 = nb18 = None
    if with_concepts and study.embed is not None:
        E = np.asarray(study.embed[ctx], dtype=np.float16)
        own = keep.astype(np.int64)
        pos = -np.ones(study.n, dtype=np.int64)
        pos[ctx] = np.arange(len(ctx))
        nb = study.neighbours6[rows]
        nb6 = np.where(nb >= 0, pos[np.maximum(nb, 0)], -1)
        nb18 = -np.ones((len(rows), 18), dtype=np.int64)
        for i, li in enumerate(keep):
            idx = A18.indices[A18.indptr[li]:A18.indptr[li + 1]]
            idx = idx[idx != li][:18]
            nb18[i, :len(idx)] = idx
    if len(resid_cols):  # own, nbr6 and nbr18 programme values (the B3-style comparison design)
        R = transform_block(study.base[ctx][:, resid_cols], A6, A18, keep)[:, :3 * len(resid_cols)]
    else:
        R = np.zeros((len(rows), 0), dtype=np.float32)
    return _Block(X=X, target=None if target is None else np.asarray(target)[rows],
                  resid_design=R,
                  groups=study.groups[rows], coords=study.coords[rows], rows=rows, E=E, own=own, nb6=nb6, nb18=nb18,
                  nbk=_kept_neighbours(study, rows), depth=None if study.depth is None else study.depth[rows])


class EpisodeSampler:
    """Patient-disjoint support/query episodes from one or more studies.

    ``controlled_prob`` of the episodes are controlled (planted) episodes,
    families from ``families`` (``generators.TRAIN_FAMILIES`` by default)."""

    rowsplit = False

    def __init__(self, studies: Sequence[Study], *, families: Sequence[str] | None = None,
                 fractions: Sequence[float] = FRACTIONS, controlled_prob: float = 0.5,
                 fraction_range: tuple | None = TRAIN_FRACTION_RANGE,
                 cross_study_prob: float = 0.3, n_support_patients: int = 10, n_query_patients: int = 3,
                 min_support_patients: int = 3, max_rows_per_patient: int = 600, concept_target_prob: float = 0.25,
                 concepts: bool = True, transform_prob: float = 0.25, split: str = "train", seed: int = 0,
                 morph_target_prob: float = 0.0, proxy_threshold: float = 1.0):
        from dbench.benchmark import generators as gen

        self.gen = gen
        self.studies = list(studies)
        self.families = tuple(families) if families is not None else tuple(gen.TRAIN_FAMILIES)
        self.fractions = tuple(fractions)
        # an explicit fraction list (anything but the default grid) means draws from that list
        self.fraction_range = fraction_range if tuple(fractions) == FRACTIONS else None
        self.controlled_prob = controlled_prob
        self.cross_study_prob = cross_study_prob
        self.n_sup, self.n_q = n_support_patients, n_query_patients
        self.min_sup = min(min_support_patients, n_support_patients)
        self.max_rows = max_rows_per_patient
        self.concept_target_prob = concept_target_prob
        self.concepts = concepts
        self.transform_prob = transform_prob
        self.split = split
        self.morph_target_prob = morph_target_prob
        self.proxy_threshold = proxy_threshold
        self.rng = np.random.default_rng(seed)

    # -- patients
    def _pool(self, study: Study) -> np.ndarray:
        if self.split in ("dev", "dev_mixed") and study.dev_patients is not None:
            return study.dev_patients
        return study.train_patients

    def _patients(self, study: Study, rng, n_sup: int, n_q: int, query_study: Study | None):
        n_sup = int(rng.integers(self.min_sup, n_sup + 1)) if n_sup > self.min_sup else n_sup
        if self.split == "dev_mixed" and study.dev_patients is not None and query_study is None:
            # development episodes: support from training patients, query from held-out patients
            pool = rng.permutation(study.train_patients)
            qpool = rng.permutation(study.dev_patients)
            return list(pool[:min(n_sup, len(pool))]), list(qpool[:min(n_q, len(qpool))])
        pool = rng.permutation(self._pool(study))
        if query_study is not None:
            qpool = rng.permutation(self._pool(query_study))
            ns = min(n_sup, len(pool))
            return list(pool[:ns]), list(qpool[:min(n_q, len(qpool))])
        nq = min(n_q, max(1, len(pool) // 3))
        ns = min(n_sup, len(pool) - nq)
        if ns < 1:
            raise ValueError(f"study {study.name} has too few patients for a patient-disjoint episode")
        return list(pool[:ns]), list(pool[ns:ns + nq])

    # -- columns
    def _target(self, study: Study, rng):
        """(name, values (n,), excluded base cols, excluded views)."""
        morph = [int(c) for c in study.views.get("morph", [])]
        if morph and rng.random() < self.morph_target_prob:   # a morphology-similarity target
            c = morph[int(rng.integers(len(morph)))]
            return study.names[c], study.base[:, c], [c], []
        names = list(study.extra_targets)
        n_opts = len(study.target_cols) + len(names)
        if n_opts == 0:
            raise ValueError(f"study {study.name} has no targets")
        k = int(rng.integers(n_opts))
        if k < len(names):
            nm = names[k]
            return nm, study.extra_targets[nm], [], list(study.extra_target_excluded_views.get(nm, []))
        c = int(study.target_cols[k - len(names)])
        return study.names[c], study.base[:, c], [c] + list(study.exclusions.get(c, [])), []

    def _columns(self, study: Study, rng, excluded: Sequence[int], excluded_views: Sequence[str],
                 shared: set | None, support_patients: Sequence[str] = ()) -> np.ndarray:
        cols = []
        ex0 = set(int(c) for c in excluded)
        for v, vc in study.views.items():
            if v in excluded_views:
                continue
            vc = np.asarray(vc, dtype=int)
            cap = study.subsample_views.get(v)
            if cap is not None and len(vc) > cap:
                vc = np.sort(rng.choice(vc, cap, replace=False))
            hv = study.hvg_views.get(v)
            if hv is not None and len(vc) > hv and len(support_patients):
                # highly variable on SUPPORT rows (base is within-patient centred), target excluded
                rows = np.concatenate([study.rows_of[p] for p in support_patients])
                vc = np.array([c for c in vc if c not in ex0], dtype=int)
                var = np.square(study.base[rows][:, vc]).mean(0)
                vc = np.sort(vc[np.argsort(-var, kind="stable")[:hv]])
            cols.append(vc)
        cols = np.concatenate(cols) if cols else np.zeros(0, dtype=int)
        ex = set(int(c) for c in excluded)
        cols = np.array([c for c in cols if c not in ex and (shared is None or study.names[c] in shared)], dtype=int)
        return np.sort(cols)

    def _drop_proxies(self, study: Study, cols: np.ndarray, tvals: np.ndarray, support_patients) -> np.ndarray:
        """Drop base columns whose within-patient |r| with the target on support rows is
        >= ``proxy_threshold`` (the ``hest_multi`` proxy rule; e.g. near-duplicate
        morphology similarities of a morphology target)."""
        rows = np.concatenate([study.rows_of[p] for p in support_patients])
        B = study.base[rows][:, cols].astype(np.float64)        # base is within-patient centred
        t = np.asarray(tvals, dtype=np.float64)[rows].copy()
        for p in support_patients:
            m = study.groups[rows] == p
            t[m] -= t[m].mean()
        r =(B.T @ t) / (np.sqrt((B ** 2).sum(0) * (t ** 2).sum()) + 1e-12)
        return cols[np.abs(r) < self.proxy_threshold]

    def _query_study(self, study: Study, rng) -> Study | None:
        if len(self.studies) < 2 or rng.random() >= self.cross_study_prob:
            return None
        others = [s for s in self.studies if s is not study and len(self._pool(s)) >= 1
                  and len(set(s.names) & set(study.names)) >= 0.4 * len(study.names)]
        return others[int(rng.integers(len(others)))] if others else None

    # -- the episode
    def sample(self, rng: np.random.Generator | None = None, *, kind: str | None = None,
               family: str | None = None, fraction: float | None = None, study: Study | None = None) -> Episode:
        rng = self.rng if rng is None else rng
        for attempt in range(20):
            try:
                return self._sample(rng, kind=kind, family=family, fraction=fraction, study=study)
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:  # degenerate draw: redraw
                last = exc
        raise RuntimeError(f"episode sampling failed 20 times: {last}")

    def _sample(self, rng, *, kind, family, fraction, study) -> Episode:
        study = study if study is not None else self.studies[int(rng.integers(len(self.studies)))]
        kind = kind or ("controlled" if rng.random() < self.controlled_prob else "real")
        qstudy = self._query_study(study, rng)
        sup_p, q_p = self._patients(study, rng, self.n_sup, self.n_q, qstudy)
        use_concepts = self.concepts and study.embed is not None and (qstudy is None or qstudy.embed is not None)
        concept_target = None
        if kind == "real" and use_concepts and rng.random() < self.concept_target_prob:
            concept_target = int(rng.integers(1 << 30))  # reduced mod n_concepts in the model
            tname, tvals, excl, exv = "concept", None, [], []
        else:
            tname, tvals, excl, exv = self._target(study, rng)
        shared = None if qstudy is None else set(qstudy.names)
        cols = self._columns(study, rng, excl, exv, shared, sup_p)
        if qstudy is not None and tname != "concept" and tname not in qstudy.index and tname not in qstudy.extra_targets:
            qstudy = None
            sup_p, q_p = self._patients(study, rng, self.n_sup, self.n_q, None)
            cols = self._columns(study, rng, excl, exv, None, sup_p)
        if tvals is not None and self.proxy_threshold < 1.0:
            cols = self._drop_proxies(study, cols, tvals, sup_p)
        if len(cols) < 4:
            raise ValueError("too few candidate columns")
        resid_cols = np.array([c for c in study.residual_cols if c not in set(excl)], dtype=int)
        blocks = self._make_blocks(study, qstudy, sup_p, q_p, cols, rng, tname, tvals, resid_cols, use_concepts)
        return self._assemble(study, qstudy, blocks, cols, rng, kind=kind, family=family, fraction=fraction,
                              tname=tname, concept_target=concept_target, sup_p=sup_p, q_p=q_p)

    def _make_blocks(self, study, qstudy, sup_p, q_p, cols, rng, tname, tvals, resid_cols, use_concepts):
        sb = _block(study, sup_p, cols, rng, target=tvals, resid_cols=resid_cols,
                    max_rows_per_patient=self.max_rows, with_concepts=use_concepts)
        if qstudy is None:
            qb = _block(study, q_p, cols, rng, target=tvals, resid_cols=resid_cols,
                        max_rows_per_patient=self.max_rows, with_concepts=use_concepts)
            return [sb, qb], [study.name + ":" + p for p in sup_p], [study.name + ":" + p for p in q_p]
        qcols = np.array([qstudy.index[study.names[c]] for c in cols], dtype=int)
        qres = np.array([qstudy.index[study.names[c]] for c in resid_cols if study.names[c] in qstudy.index],
                        dtype=int)
        if len(qres) != len(resid_cols):
            ok = np.array([study.names[c] in qstudy.index for c in resid_cols])
            sb.resid_design = sb.resid_design[:, np.tile(ok, 3)]
        if tname == "concept":
            qt = None
        elif tname in qstudy.extra_targets:
            qt = qstudy.extra_targets[tname]
        else:
            qt = qstudy.base[:, qstudy.index[tname]]
        qb = _block(qstudy, q_p, qcols, rng, target=qt, resid_cols=qres,
                    max_rows_per_patient=self.max_rows, with_concepts=use_concepts)
        return [sb, qb], [study.name + ":" + p for p in sup_p], [qstudy.name + ":" + p for p in q_p]

    def _assemble(self, study, qstudy, blocks_info, cols, rng, *, kind, family, fraction, tname,
                  concept_target, sup_p, q_p) -> Episode:
        (sb, qb), sup_codes, q_codes = blocks_info
        n_s = len(sb.rows)
        X = np.concatenate([sb.X, qb.X])
        gstr = np.concatenate([np.char.add(sup_codes[0].split(":")[0] + ":", sb.groups.astype(str)),
                               np.char.add((q_codes[0].split(":")[0]) + ":", qb.groups.astype(str))])
        _, groups = np.unique(gstr, return_inverse=True)
        support = np.zeros(len(groups), dtype=bool)
        support[:n_s] = True
        X = centre_scale(X, groups, support)
        coords = _coords_std(np.concatenate([sb.coords, qb.coords]), groups)
        m = len(cols)
        names = [f"{t}:{study.names[c]}" for t in TRANSFORMS for c in cols]
        transform = np.repeat(np.arange(4), m)
        view = np.tile(np.array([view_id(study.view_of[c]) for c in cols]), 4)
        meta = TokenMeta(names, transform.astype(np.int64), view.astype(np.int64))
        concept = None
        if sb.E is not None and qb.E is not None:
            off = len(sb.E)
            concept = {"E": np.concatenate([sb.E, qb.E]),
                       "own": np.concatenate([sb.own, qb.own + off]),
                       "nb6": np.concatenate([sb.nb6, np.where(qb.nb6 >= 0, qb.nb6 + off, -1)]),
                       "nb18": np.concatenate([sb.nb18, np.where(qb.nb18 >= 0, qb.nb18 + off, -1)])}
        nb6 = None
        if sb.nbk is not None and qb.nbk is not None:
            nb6 = np.concatenate([sb.nbk, np.where(qb.nbk >= 0, qb.nbk + n_s, -1)])
        nuisance = None
        if sb.depth is not None and qb.depth is not None:
            nuisance = centre_scale(np.concatenate([sb.depth, qb.depth]).astype(np.float64), groups, support)
        ep = Episode(X=X, y=None, n_support=n_s, groups=groups.astype(np.int64), coords=coords, meta=meta,
                     kind=kind, concept=concept, concept_target=concept_target, target_name=tname,
                     study=study.name, query_study=(qstudy or study).name,
                     support_patients=tuple(sup_codes), query_patients=tuple(q_codes), nb6=nb6, nuisance=nuisance)
        if tname == "concept":
            if kind != "real":
                raise ValueError("concept targets are real episodes only")
            return ep
        ybase = np.concatenate([sb.target, qb.target]).astype(np.float64)
        if kind == "real":
            ep.y = centre_scale(ybase, groups, support)
            return ep
        # controlled: residual of the real column on the programme scores, plus g(V)
        D = np.concatenate([sb.resid_design, qb.resid_design]) if sb.resid_design.shape[1] else np.zeros((len(groups), 0))
        r = episode_residual(ybase, D, groups)
        family = family or self.families[int(rng.integers(len(self.families)))]
        if fraction is None and self.fraction_range is not None:
            lo, hi = self.fraction_range
            fraction = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        fraction = float(fraction if fraction is not None else self.fractions[int(rng.integers(len(self.fractions)))])
        true_idx = self._draw_truth(X, n_s, m, family, rng)
        V = X[:, list(true_idx)].astype(np.float64)
        g, ginfo = self.gen.plant(V, family, rng, groups=groups)
        from dbench.benchmark.tasks import calibrate
        gs, scale = calibrate(g, r, fraction)
        ep.y = centre_scale(r + gs, groups, support)
        ep.kind, ep.family, ep.fraction, ep.true_idx = "controlled", family, fraction, tuple(int(i) for i in true_idx)
        tot = r + gs
        ep.info = {"scale": float(scale), "generator": {k: v for k, v in ginfo.items() if k not in ("W1",)},
                   "true_names": ep.true_names(),
                   "signal_share_support": float(np.var(gs[:n_s]) / max(np.var(tot[:n_s]), 1e-12)),
                   "signal_share_query": float(np.var(gs[n_s:]) / max(np.var(tot[n_s:]), 1e-12))}
        return ep

    def _draw_truth(self, X: np.ndarray, n_s: int, m: int, family: str, rng) -> tuple:
        """True token indices: own tokens by default, a neighbour transform with
        probability ``transform_prob`` each; spatial families take (own_a, nbr6_b) or
        (own_a, contrast_a).  Pairs are weakly correlated (|r| < 0.3) on support rows."""
        k = self.gen.arity(family)
        Xs = X[:n_s]
        ok = np.flatnonzero(Xs[:, :m].std(0) > 0.5)
        if len(ok) < k:
            raise ValueError("too few usable columns")
        spatial = family in getattr(self.gen, "SPATIAL_FAMILIES", ()) or family in ("spatial_lag",)
        for _ in range(200):
            base = rng.choice(ok, k, replace=False)
            if spatial and k == 2:
                if rng.random() < 0.5:
                    idx = (int(base[0]), int(base[1]) + m)          # own_a, nbr6_b
                else:
                    idx = (int(base[0]), int(base[0]) + 3 * m)      # own_a, contrast_a
            else:
                idx = tuple(int(b) + (int(rng.integers(1, 3)) * m if rng.random() < self.transform_prob else 0)
                            for b in base)
            if any(Xs[:, i].std() < 0.5 for i in idx):
                continue
            C = np.corrcoef(Xs[:, list(idx)].T) if k > 1 else np.ones((1, 1))
            if k == 1 or np.max(np.abs(C[np.triu_indices(k, 1)])) < 0.3:
                return idx
        raise ValueError("no weakly correlated set found")


class RowSplitSampler(EpisodeSampler):
    """The ablation: support and query rows from the SAME patients (60/40 row split)."""

    rowsplit = True

    def _patients(self, study, rng, n_sup, n_q, query_study):
        pool = rng.permutation(self._pool(study))
        pats = list(pool[:min(n_sup + n_q, len(pool))])
        return pats, pats

    def _query_study(self, study, rng):
        return None

    def _make_blocks(self, study, qstudy, sup_p, q_p, cols, rng, tname, tvals, resid_cols, use_concepts):
        full = _block(study, sup_p, cols, rng, target=tvals, resid_cols=resid_cols,
                      max_rows_per_patient=self.max_rows, with_concepts=use_concepts)
        sel = np.zeros(len(full.rows), dtype=bool)
        for g in np.unique(full.groups):
            idx = np.flatnonzero(full.groups == g)
            sel[rng.choice(idx, int(round(0.6 * len(idx))), replace=False)] = True

        def part(mask):
            pos = np.cumsum(mask) - 1
            nbk = None
            if full.nbk is not None:
                nb = full.nbk[mask]
                ok = (nb >= 0) & mask[np.maximum(nb, 0)]
                nbk = np.where(ok, pos[np.maximum(nb, 0)], -1)
            return _Block(X=full.X[mask], target=None if full.target is None else full.target[mask],
                          resid_design=full.resid_design[mask], groups=full.groups[mask], coords=full.coords[mask],
                          rows=full.rows[mask], E=full.E, own=None if full.own is None else full.own[mask],
                          nb6=None if full.nb6 is None else full.nb6[mask],
                          nb18=None if full.nb18 is None else full.nb18[mask], nbk=nbk,
                          depth=None if full.depth is None else full.depth[mask])

        sb, qb = part(sel), part(~sel)
        if qb.E is not None:  # both halves index one shared context block
            qb.E = qb.E[:0]
            qb.own, qb.nb6, qb.nb18 = qb.own - len(full.E), np.where(qb.nb6 >= 0, qb.nb6 - len(full.E), -1), \
                np.where(qb.nb18 >= 0, qb.nb18 - len(full.E), -1)
        codes = [study.name + ":" + p for p in sup_p]
        return [sb, qb], codes, codes


# ------------------------------------------------------------------ loading
def collate(batch):
    """Episodes are processed one at a time by the model; a batch is a list."""
    return list(batch)


class _BatchStream:
    def __init__(self, sampler: EpisodeSampler, n_real: int, n_controlled: int, seed: int):
        self.sampler, self.n_real, self.n_controlled, self.seed = sampler, n_real, n_controlled, seed

    def __iter__(self) -> Iterator[list]:
        import torch.utils.data as tud

        info = tud.get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield ([self.sampler.sample(rng, kind="real") for _ in range(self.n_real)]
                   + [self.sampler.sample(rng, kind="controlled") for _ in range(self.n_controlled)])


def make_loader(sampler: EpisodeSampler, *, n_real: int = 4, n_controlled: int = 4, workers: int = 8,
                seed: int = 0, prefetch: int = 2):
    """Endless loader of batches (lists of ``n_real`` real + ``n_controlled`` controlled episodes)."""
    import torch.utils.data as tud

    class _DS(tud.IterableDataset):
        def __iter__(self_inner):
            return iter(_BatchStream(sampler, n_real, n_controlled, seed))

    kw = dict(batch_size=None, collate_fn=None, num_workers=workers)
    if workers > 0:
        kw.update(prefetch_factor=prefetch, persistent_workers=True)
    return tud.DataLoader(_DS(), **kw)


def fixed_dev_set(studies: Sequence[Study], *, n_controlled: int = 64, n_real: int = 64, families=None,
                  seed: int = 12345, **sampler_kw) -> list:
    """Deterministic development episodes.

    Controlled episodes use ``families`` (``generators.HELDOUT_FAMILIES`` by
    default), cycling over the four fractions; where a study defines
    development patients, their support rows come from training patients
    (planted functions and families are new, so nothing is memorisable) and
    their query rows from the development patients.  Real tasks use
    development patients only (support and query)."""
    from dbench.benchmark import generators as gen

    fams = tuple(families) if families is not None else tuple(gen.HELDOUT_FAMILIES)
    fr = tuple(sampler_kw.pop("fractions", FRACTIONS))
    sc = EpisodeSampler(studies, families=fams, split="dev_mixed", cross_study_prob=0.0, fractions=fr,
                        n_support_patients=sampler_kw.get("n_support_patients", 10),
                        min_support_patients=sampler_kw.get("n_support_patients", 10),
                        n_query_patients=sampler_kw.get("n_query_patients", 3), seed=seed,
                        **{k: v for k, v in sampler_kw.items() if k not in ("n_support_patients", "n_query_patients")})
    sr = EpisodeSampler(studies, split="dev", cross_study_prob=0.0, n_support_patients=3, min_support_patients=3,
                        n_query_patients=2, seed=seed,
                        **{k: v for k, v in sampler_kw.items() if k not in ("n_support_patients", "n_query_patients")})
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_controlled):
        out.append(sc.sample(rng, kind="controlled", family=fams[i % len(fams)],
                             fraction=fr[(i // len(fams)) % len(fr)]))
    for _ in range(n_real):
        out.append(sr.sample(rng, kind="real"))
    return out
