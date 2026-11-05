"""Orthogonal molecular support (downstream spec Section 5): the finding's molecular score per patient in
CPTAC ccRCC (RNA and protein, ~110 tumours) and TCGA-KIRC (RNA, ~510 tumours).

* gene set: mean of per-gene standardised (log) expression over the set's measured genes;
* residual programme: the frozen loading applied to the bulk residual after the covariate adjustment at
  bulk level: each gene regressed (ridge, intercept free) on the declared covariates, composition replaced
  by deconvolution proxies and depth by library size, residual standardised per gene, dot the loading.

Composition proxies are fixed marker scores identical in every bulk cohort (immune, stromal, endothelial,
tumour-epithelial; ``COMPOSITION_MARKERS``) plus the 41 Gavish meta-programme scores; library size is
used where the matrix carries it and recorded as unavailable otherwise (TCGA RSEM-normalised and CPTAC
log2 matrices do not). The legacy G18 definition (rank-z over tumours, dot the loading; the 0.743 RNA-
protein Spearman of the original paper) is kept as ``legacy_rankz_score`` for the reproduction check.
The output is patient-level support for the molecular half of a finding; it is never called replication
of the spatial relationship, and a morphology constituent is reported as not carried by bulk data.
"""
from __future__ import annotations

import numpy as np

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
CPTAC = f"{STORE}/external/hf_tcga_cptac_cgga/cptac/tables/ccrcc"
TCGA = f"{STORE}/external/tcga_kirc_20260925"
LOADINGS = f"{STORE}/code/ws_aa_20260825/frozen_consensus_loadings.npz"
COMPOSITION_MARKERS = {
    "immune": ["PTPRC", "CD3E", "CD2", "CD247", "CD53", "LCP2", "CORO1A", "CD48", "LAPTM5", "CD52"],
    "stromal": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "FAP", "COL6A3", "SPARC", "THY1"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "ESAM", "EMCN", "PLVAP", "CLDN5", "TIE1", "ROBO4"],
    "tumour_epithelial": ["CA9", "NDUFA4L2", "ANGPTL4", "VIM", "PAX8", "PAX2", "KRT8", "KRT18", "EPCAM", "CDH16"],
}


def strip_columns(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            out.append(str(c[0]))
        elif isinstance(c, str) and c.startswith("("):
            out.append(c.split("'")[1])
        else:
            out.append(str(c))
    return out


def load_matrix(path):
    import pandas as pd

    df = pd.read_parquet(path)
    df.columns = strip_columns(df.columns)
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def rankz(df):
    z = df.rank(axis=0)
    return (z - z.mean()) / z.std(ddof=0)


def frozen_loading(programme: str, path: str = LOADINGS, organ: str = "ccRCC"):
    z = np.load(path, allow_pickle=True)
    return [str(g) for g in z[f"{organ}_{programme}__genes"]], np.asarray(z[f"{organ}_{programme}__loading"], float)


def legacy_rankz_score(expr, genes, weights):
    """Original-paper arithmetic: rank-z per gene across tumours, dot the frozen loading over shared genes."""
    import pandas as pd

    w = pd.Series(weights, index=genes)
    w = w[~w.index.duplicated()]
    fin = np.isfinite(expr.to_numpy(float)).all(0)
    sd = expr.std().to_numpy()
    shared = [g for g, f, v in zip(expr.columns, fin, sd) if g in w.index and f and v > 1e-9]
    Z = rankz(expr[shared])
    return pd.Series(Z.to_numpy() @ w[shared].to_numpy(), index=expr.index), len(shared)


def cptac_tables():
    """CPTAC ccRCC tumour RNA (log2) and protein, filtered exactly as the original G18 RNA-protein test."""
    import pandas as pd

    tp = pd.read_parquet(f"{CPTAC}/tumor_purity__washu.parquet")
    rna = load_matrix(f"{CPTAC}/transcriptomics__bcm.parquet")
    prot = load_matrix(f"{CPTAC}/proteomics__umich.parquet")
    tset = set(tp.index)
    R = rna.loc[[i for i in rna.index if i in tset]].astype(float)
    R = R.loc[:, R.notna().mean() > 0.8]
    R = R.loc[:, R.std() > 0]
    R = R.loc[:, rankz(R).notna().all()]
    P = prot.loc[[i for i in prot.index if i in tset]].astype(float)
    P = P.loc[:, P.notna().mean() > 0.7]
    P = P.fillna(P.median())
    P = P.loc[:, P.std() > 0]
    clin = pd.read_csv(f"{CPTAC}/clinical__mssm.csv", index_col=0, low_memory=False)
    return {"rna": R, "protein": P, "purity": tp, "clinical": clin}


def tcga_tables():
    import pandas as pd

    r = pd.read_parquet(f"{TCGA}/rna_seq_v2_mrna.parquet").set_index("patient_id")
    r = r.loc[~r.index.duplicated()].astype(float)
    r = np.log2(r.clip(lower=0) + 1)
    r = r.loc[:, (r.std() > 0) & r.notna().all()]
    clin = pd.read_parquet(f"{TCGA}/clinical.parquet").set_index("patient_id")
    return {"rna": r, "clinical": clin}


def zscore_cols(df):
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1)


def set_score(expr, genes):
    g = [x for x in dict.fromkeys(genes) if x in expr.columns]
    return (zscore_cols(expr[g]).mean(1) if g else None), len(g)


def bulk_covariates(expr, sets_mp: dict | None = None, library_size=None):
    """Composition proxies (+ Gavish MP scores, + log library size when available) per sample."""
    import pandas as pd

    cols, rec = {}, {}
    for nm, genes in COMPOSITION_MARKERS.items():
        s, n = set_score(expr, genes)
        rec[f"comp:{nm}"] = n
        if s is not None:
            cols[f"comp:{nm}"] = s
    for nm, genes in (sets_mp or {}).items():
        s, n = set_score(expr, genes)
        if s is not None and n >= 3:
            cols[f"mp:{nm}"] = s
    if library_size is not None:
        cols["log_library_size"] = np.log(np.asarray(library_size, float))
    rec["library_size"] = "used" if library_size is not None else "unavailable in this matrix (normalised values)"
    return pd.DataFrame(cols, index=expr.index), rec


def residual_score(expr, covariates, genes, weights, alpha: float = 1.0):
    """Frozen loading on the covariate-adjusted bulk residual (per-gene standardised)."""
    import pandas as pd

    w = pd.Series(weights, index=genes)
    w = w[~w.index.duplicated()]
    shared = [g for g in expr.columns if g in w.index]
    X = zscore_cols(expr[shared]).to_numpy()
    C = zscore_cols(covariates).to_numpy()
    C = np.column_stack([np.ones(len(C)), C])
    pen = np.diag([0.0] + [alpha] * (C.shape[1] - 1))
    beta = np.linalg.solve(C.T @ C + pen, C.T @ X)
    Rz = X - C @ beta
    sd = Rz.std(0)
    Rz = Rz / np.where(sd > 1e-12, sd, 1.0)
    wv = w[shared].to_numpy()
    tot = float((w ** 2).sum())
    return pd.Series(Rz @ wv, index=expr.index), {"n_genes": len(shared),
                                                   "loading_norm_coverage": float((wv ** 2).sum() / tot)}


def spearman(a, b) -> dict:
    from scipy.stats import spearmanr

    common = a.index.intersection(b.index)
    r, p = spearmanr(a[common], b[common])
    return {"rho": float(r), "p": float(p), "n": int(len(common))}


def bootstrap_spearman(a, b, n_boot: int = 2000, seed: int = 20261021) -> list:
    from scipy.stats import spearmanr

    common = a.index.intersection(b.index)
    x, y = a[common].to_numpy(), b[common].to_numpy()
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        v.append(spearmanr(x[i], y[i])[0])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def finding_bulk_support(kind: str, *, genes=None, loading=None, sets_mp=None, has_morphology=False) -> dict:
    """Section 5 for one finding: CPTAC RNA score, protein score and their Spearman (with a patient
    bootstrap interval); TCGA RNA score availability. ``kind`` in {'gene_set', 'residual'}."""
    out = {"note": "patient-level support for the molecular half of the finding; never replication of the "
                   "spatial relationship"}
    if has_morphology:
        out["morphology"] = "the finding has a morphology constituent, which bulk data cannot carry"
    cp = cptac_tables()
    tc = tcga_tables()
    scores = {}
    for nm, expr in (("cptac_rna", cp["rna"]), ("cptac_protein", cp["protein"]), ("tcga_rna", tc["rna"])):
        if kind == "gene_set":
            s, n = set_score(expr, genes)
            scores[nm], info = s, {"n_genes": n}
        else:
            base = cp["rna"] if nm == "cptac_protein" else expr
            cov, crec = bulk_covariates(base, sets_mp)
            cov = cov.loc[expr.index.intersection(cov.index)]
            e = expr.loc[cov.index]
            s, info = residual_score(e, cov, loading[0], loading[1])
            info["covariates"] = crec
            scores[nm] = s
        out[nm] = {"available": scores[nm] is not None, **info}
    if scores["cptac_rna"] is not None and scores["cptac_protein"] is not None:
        out["cptac_rna_protein"] = {**spearman(scores["cptac_rna"], scores["cptac_protein"]),
                                    "ci95": bootstrap_spearman(scores["cptac_rna"], scores["cptac_protein"])}
    out["_scores"] = scores
    return out
