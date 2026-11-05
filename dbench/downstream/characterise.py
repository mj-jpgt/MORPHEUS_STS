"""Characterisation procedures (downstream spec Section 1), fixed functions run with the literature closed.

1.1 ``constituent_table``        constituents, partial-association sign, held-out standardised contribution,
                                 top +/-25 loaded genes of residual programmes, morphology centre panels
1.2 ``enrichment``               hypergeometric over-representation vs Hallmark, Reactome, Gavish (cohort gene
                                 universe, BH within the finding, residual +/- lists separately) and the
                                 Spearman of the finding score with every Hallmark / Gavish score
1.3 ``localisation``             per-section maps, PanNuke composition strata, top-decile occupancy, Moran's I
1.4 ``scale_decomposition``      own-version refit loss per candidate; organisation-beyond-abundance
                                 dR2 = R2(A u O) - R2(A) on held-out patients with the scrambled-configuration null
1.5 ``neighbourhood``            Squidpy neighbourhood enrichment (6-NN graph, 1000 permutations) and
                                 co-occurrence (50-500 um), conditional
1.6 ``liana_bivariate``          LIANA+ spatially weighted cosine over the consensus resource, conditional, one
                                 repair attempt, else 'not assessed'
1.7 ``pathologist_packet``       OPTIONAL, off by default (the blinded read was scratched by the owner on
                                 2026-10-21; code kept, packets not generated): 16 state-high / 16 state-low
                                 250 um regions matched on composition (Mahalanobis < 1.0, optimal assignment)
Each function returns a dict with ``status`` in {run, skipped} and a reason when skipped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .relation import (FrozenRelation, fit_relation, patient_folds, r2, standardised_contributions)

PANNUKE = ("Neoplastic", "Connective", "Epithelial", "Inflammatory", "Dead")
STRATUM_RULE = {"immune-rich": ("Inflammatory", 0.30), "tumour-rich": ("Neoplastic", 0.50),
                "stroma-rich": ("Connective", 0.40)}
REACTOME_GMT = "/lambda/nfs/geeg/biorag3_persistent_20260711/external/reactome_gmt_20261021/ReactomePathways.gmt"
MORAN_DIFFUSE = 0.1


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "reason": reason}


# ----------------------------------------------------------------------------- 1.1
def candidate_class(name: str) -> str:
    n = name.lower()
    if n.startswith("residual:") or "pca:" in n or n.startswith("g") and n[1:3].isdigit():
        return "residual programme"
    if n.startswith(("own:", "nbr:")) or any(t in n for t in ("hallmark:", "gavish:", "programmes:")):
        return "programme score"
    if n.startswith("gene:") or "raw_genes:" in n:
        return "gene"
    if "morph" in n or "centre" in n or "hoptimus" in n:
        return "morphology similarity"
    if "cellvit" in n or n.startswith(("cv_", "nuc_")):
        return "CellViT descriptor"
    return "other"


def candidate_version(name: str) -> str:
    if "@" in name:
        return name.rsplit("@", 1)[1]
    if "nbr18" in name:
        return "v18"
    if "context_neighbour_mean:nbr:" in name or name.startswith("nbr:"):
        return "v6"
    return "v0"


def heldout_contributions(Z, r, groups, rel_template: FrozenRelation) -> list[float]:
    """Standardised contribution on held-out patients: refit the same family on the other patients
    (patient folds), evaluate the R2 drop on the held-out fold, pool the folds."""
    Z = np.asarray(Z, float).reshape(len(r), -1)
    groups = np.asarray(groups).astype(str)
    pred_full, pred_drop = np.zeros(len(r)), np.zeros((Z.shape[1], len(r)))
    for held in patient_folds(groups, max_folds=6, seed=0):
        h = np.isin(groups, held)
        rel = fit_relation(Z[~h], r[~h], groups[~h], rel_template.candidates, rel_template.target,
                           rel_template.baseline, blocks=rel_template.blocks)
        pred_full[h] = rel.predict(Z[h])
        for j in range(Z.shape[1]):
            Zj = Z[h].copy()
            Zj[:, j] = 0.0
            pred_drop[j, h] = rel.predict(Zj)
    full = r2(r, pred_full)
    return [float(full - r2(r, pred_drop[j])) for j in range(Z.shape[1])]


def top_loaded(genes, loading, n: int = 25) -> dict:
    w = np.asarray(loading, float)
    o = np.argsort(w)
    return {"positive": [(str(genes[i]), float(w[i])) for i in o[::-1][:n]],
            "negative": [(str(genes[i]), float(w[i])) for i in o[:n]]}


def centre_panel(centre, patch_embeddings, patch_ids, cellvit_profiles=None, n: int = 32) -> dict:
    """1.1 morphology centre interpretation panel: the n nearest meta-training patches by cosine and the
    centre's mean CellViT descriptor profile over those patches. No label is assigned."""
    E = np.asarray(patch_embeddings, float)
    c = np.asarray(centre, float)
    cos = E @ c / (np.linalg.norm(E, axis=1) * np.linalg.norm(c) + 1e-12)
    idx = np.argsort(-cos)[:n]
    out = {"nearest_patches": [str(patch_ids[i]) for i in idx], "cosine": [float(cos[i]) for i in idx],
           "label": None}
    if cellvit_profiles is not None:
        out["mean_cellvit_profile"] = np.asarray(cellvit_profiles, float)[idx].mean(0).tolist()
    return out


def constituent_table(rel: FrozenRelation, Z, r, groups, *, residual_loadings: dict | None = None,
                      morphology_panels: dict | None = None) -> dict:
    Z = np.asarray(Z, float).reshape(len(r), -1)
    grad = rel.gradient(Z).mean(0)
    contrib_ho = heldout_contributions(Z, r, groups, rel)
    contrib_in = standardised_contributions(rel, Z, r)
    rows = []
    for j, c in enumerate(rel.candidates):
        row = {"candidate": c, "class": candidate_class(c), "spatial_version": candidate_version(c),
               "partial_association_sign": int(np.sign(grad[j])), "mean_partial_derivative": float(grad[j]),
               "standardised_contribution_heldout": contrib_ho[j],
               "standardised_contribution_support": float(contrib_in[j])}
        if residual_loadings and c in residual_loadings:
            row["loaded_genes"] = top_loaded(*residual_loadings[c])
        if morphology_panels and c in morphology_panels:
            row["interpretation_panel"] = morphology_panels[c]
        rows.append(row)
    return {"status": "run", "rows": rows}


# ----------------------------------------------------------------------------- 1.2
def load_gmt(path) -> dict:
    out = {}
    for ln in Path(path).read_text().splitlines():
        p = ln.split("\t")
        if len(p) > 2:
            out[p[0]] = [g for g in p[2:] if g]
    return out


def reference_sets(sets: dict | None = None, reactome_gmt: str = REACTOME_GMT) -> dict:
    """{'hallmark': {name: genes}, 'gavish': {...}, 'reactome': {...}}."""
    if sets is None:
        from morpheus.tier2.substrates.hest_multi import _programme_sets

        sets = _programme_sets()
    out = {"hallmark": {s["name"]: list(s["genes"]) for s in sets["hallmark"]},
           "gavish": {s["name"]: list(s["genes"]) for s in sets["mp"]}}
    out["reactome"] = load_gmt(reactome_gmt) if reactome_gmt and os.path.exists(reactome_gmt) else {}
    return out


def bh(p) -> np.ndarray:
    from .replication_v03 import bh as _bh

    return _bh(p)


def ora(query: Sequence[str], universe: Sequence[str], refs: dict, min_size: int = 5) -> list[dict]:
    """Hypergeometric over-representation of ``query`` within ``universe``; BH over every tested set
    of every collection (within the finding)."""
    from scipy.stats import hypergeom

    U = set(universe)
    q = set(query) & U
    rows = []
    for coll, d in refs.items():
        for nm, genes in d.items():
            G = set(genes) & U
            if len(G) < min_size or not q:
                continue
            k = len(q & G)
            p = float(hypergeom.sf(k - 1, len(U), len(G), len(q))) if k else 1.0
            rows.append({"collection": coll, "set": nm, "set_size_in_universe": len(G), "overlap": k,
                         "query_size": len(q), "p": p, "genes": sorted(q & G)})
    for row, qq in zip(rows, bh([x["p"] for x in rows])):
        row["q"] = float(qq)
    return sorted(rows, key=lambda x: x["p"])


def enrichment(gene_lists: dict, universe, refs: dict, score=None, programme_scores=None,
               programme_names=None, top: int = 20) -> dict:
    """``gene_lists``: {'constituents': [...]} or {'positive': [...], 'negative': [...]}; each list tested
    separately."""
    from scipy.stats import spearmanr

    out = {"status": "run", "universe_size": len(set(universe)), "tests": {}}
    for nm, gl in gene_lists.items():
        res = ora(gl, universe, refs)
        out["tests"][nm] = {"n_query": len(set(gl) & set(universe)), "n_sets_tested": len(res),
                            "top": res[:top], "n_q_le_0.05": int(sum(r["q"] <= 0.05 for r in res))}
    if score is not None and programme_scores is not None:
        P = np.asarray(programme_scores, float)
        rho = [float(spearmanr(score, P[:, i])[0]) for i in range(P.shape[1])]
        o = np.argsort(-np.abs(np.nan_to_num(rho)))
        out["adjacent_known_biology"] = [{"programme": programme_names[i], "spearman": rho[i]} for i in o]
    return out


# ----------------------------------------------------------------------------- 1.3
def composition_strata(fractions, covered=None) -> np.ndarray:
    """Fixed thresholds on PanNuke class fractions (columns in ``PANNUKE`` order): immune-rich
    (Inflammatory >= 0.30), tumour-rich (Neoplastic >= 0.50), stroma-rich (Connective >= 0.40); when
    several apply, the largest margin above threshold; otherwise mixed; spots without nuclei 'no nuclei'."""
    F = np.asarray(fractions, float)
    lab = np.array(["mixed"] * len(F), dtype=object)
    margins = np.column_stack([F[:, PANNUKE.index(c)] - t for c, t in STRATUM_RULE.values()])
    best = np.argmax(margins, 1)
    names = list(STRATUM_RULE)
    ok = margins.max(1) >= 0
    lab[ok] = np.array(names, dtype=object)[best[ok]]
    if covered is not None:
        lab[~np.asarray(covered, bool)] = "no nuclei"
    return lab.astype(str)


def moran_by_section(values, neighbours6, sections) -> dict:
    from morpheus.tier2.substrates.hest_multi import section_moran_i

    return section_moran_i(np.asarray(values, float), neighbours6, np.asarray(sections).astype(str))


def localisation(state, target, sections, neighbours6, strata, coords=None, out_dir=None,
                 annotations=None) -> dict:
    state, target = np.asarray(state, float), np.asarray(target, float)
    sections = np.asarray(sections).astype(str)
    mi = moran_by_section(state, neighbours6, sections)
    vals = np.array(list(mi.values()))
    top = state >= np.quantile(state, 0.9)
    region = annotations if annotations is not None else strata
    occ = {str(s): float(np.mean(region[top] == s)) for s in np.unique(region)}
    base = {str(s): float(np.mean(region == s)) for s in np.unique(region)}
    diffuse = bool(np.mean(vals < MORAN_DIFFUSE) > 0.5) if len(vals) else True
    out = {"status": "run", "moran_by_section": mi,
           "moran_median": float(np.median(vals)) if len(vals) else None,
           "moran_range": [float(vals.min()), float(vals.max())] if len(vals) else None,
           "spatially_diffuse": diffuse, "top_decile_occupancy": occ, "all_spot_occupancy": base,
           "region_source": "pathologist annotation" if annotations is not None else "CellViT composition strata"}
    if out_dir is not None and coords is not None:
        out["maps"] = section_maps(state, target, sections, coords, out_dir)
    return out


def section_maps(state, target, sections, coords, out_dir, he_thumbs: dict | None = None) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    resid = np.abs(target - state)
    paths = []
    for s in np.unique(sections):
        m = sections == s
        n_panels = 4 if he_thumbs and s in he_thumbs else 3
        fig, ax = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        k = 0
        if n_panels == 4:
            ax[0].imshow(he_thumbs[s])
            ax[0].set_title("H&E")
            k = 1
        for a, v, t in zip(ax[k:], (state, target, resid), ("state score", "target", "|residual|")):
            sc = a.scatter(coords[m, 0], -coords[m, 1], c=v[m], s=4, cmap="viridis")
            a.set_title(f"{s}: {t}")
            a.set_aspect("equal")
            a.axis("off")
            fig.colorbar(sc, ax=a, shrink=0.7)
        p = os.path.join(out_dir, f"{s}_maps.png")
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------- 1.4
def own_version_loss(rel: FrozenRelation, Z, r, groups, own_values: dict) -> dict:
    """Refit with each candidate forced to its own-value version; held-out gain lost."""
    from .relation import fit_relation as _fit

    Z = np.asarray(Z, float).reshape(len(r), -1)
    groups = np.asarray(groups).astype(str)

    def heldout_r2(ZZ):
        pred = np.zeros(len(r))
        for held in patient_folds(groups, max_folds=6, seed=0):
            h = np.isin(groups, held)
            pred[h] = _fit(ZZ[~h], r[~h], groups[~h], rel.candidates, rel.target, rel.baseline).predict(ZZ[h])
        return r2(r, pred)

    full = heldout_r2(Z)
    rows = {}
    for j, c in enumerate(rel.candidates):
        if candidate_version(c) == "v0" or c not in own_values:
            continue
        Zj = Z.copy()
        Zj[:, j] = own_values[c]
        rows[c] = {"heldout_r2_full": full, "heldout_r2_own_version": heldout_r2(Zj)}
        rows[c]["gain_lost"] = rows[c]["heldout_r2_full"] - rows[c]["heldout_r2_own_version"]
    return {"status": "run" if rows else "skipped", "rows": rows,
            **({} if rows else {"reason": "no candidate has a neighbour or contrast version"})}


def organisation_descriptors(fractions, counts, neighbours6) -> tuple[np.ndarray, list]:
    """O = graph and interface descriptors from the arrangement of composition: 6-NN mean class
    fractions, own - neighbour contrast, interface indicator (own stratum differs from the neighbour
    majority), neighbour stratum entropy, density contrast."""
    from .values import neighbour_mean

    F = np.asarray(fractions, float)
    nbF = np.column_stack([neighbour_mean(F[:, i], neighbours6) for i in range(F.shape[1])])
    lab = composition_strata(F)
    codes = np.unique(lab, return_inverse=True)[1]
    nb = np.asarray(neighbours6)
    valid = nb >= 0
    nbl = np.where(valid, codes[np.maximum(nb, 0)], -1)
    K = codes.max() + 1
    cnt = np.stack([(nbl == c).sum(1) for c in range(K)], 1).astype(float)
    tot = np.maximum(cnt.sum(1, keepdims=True), 1)
    p = cnt / tot
    ent = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)
    interface = (np.argmax(cnt, 1) != codes).astype(float)
    dens = np.asarray(counts, float)
    dcon = dens - neighbour_mean(dens, neighbours6)
    O = np.column_stack([nbF, F - nbF, interface, ent, dcon])
    names = [f"nbr6_frac_{c}" for c in PANNUKE] + [f"contrast_frac_{c}" for c in PANNUKE] + \
        ["interface", "nbr_stratum_entropy", "density_contrast"]
    return O, names


def _heldout_ridge_r2(X, y, groups) -> float:
    from .relation import fit_baseline, standardise_within

    groups = np.asarray(groups).astype(str)
    Xs, _ = standardise_within(X, groups)
    ys, _ = standardise_within(y, groups)
    pred = np.zeros(len(y))
    for held in patient_folds(groups, max_folds=6, seed=0):
        h = np.isin(groups, held)
        pred[h] = fit_baseline(Xs[~h], ys[~h], groups[~h], list(range(X.shape[1]))).predict(Xs[h])
    return r2(ys, pred)


def organisation_beyond_abundance(state, fractions, counts, areas, neighbours6, sections, groups,
                                  n_null: int = 99, seed: int = 0) -> dict:
    """dR2 = R2(A u O) - R2(A) on held-out patients; null: composition vectors scrambled across spots
    within section (abundance kept, arrangement destroyed), O recomputed, dR2 recomputed."""
    F = np.asarray(fractions, float)
    A = np.column_stack([F, np.asarray(counts, float), np.asarray(areas, float)])
    O, onames = organisation_descriptors(F, counts, neighbours6)
    rA = _heldout_ridge_r2(A, state, groups)
    obs = _heldout_ridge_r2(np.column_stack([A, O]), state, groups) - rA
    rng = np.random.default_rng(seed)
    sections = np.asarray(sections).astype(str)
    null = []
    for _ in range(n_null):
        perm = np.arange(len(F))
        for s in np.unique(sections):
            ix = np.flatnonzero(sections == s)
            perm[ix] = rng.permutation(ix)
        Fp, cp = F[perm], np.asarray(counts, float)[perm]
        Ap = np.column_stack([Fp, cp, np.asarray(areas, float)[perm]])
        Op, _ = organisation_descriptors(Fp, cp, neighbours6)
        # the state stays in place; abundance A is the scrambled spots' own, so A u O vs A isolates arrangement
        null.append(_heldout_ridge_r2(np.column_stack([Ap, Op]), state, groups) - _heldout_ridge_r2(Ap, state, groups))
    null = np.asarray(null)
    return {"status": "run", "r2_A": rA, "delta_r2": float(obs), "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)), "p": float((1 + np.sum(null >= obs)) / (n_null + 1)),
            "n_null": int(n_null), "O_descriptors": onames}


# ----------------------------------------------------------------------------- 1.5
def _anndata(coords_um, sections, labels, X=None, genes=None):
    import anndata as ad
    import pandas as pd

    sections = np.asarray(sections).astype(str)
    C = np.asarray(coords_um, float).copy()
    # separate sections in space so no neighbour or distance crosses a section
    for i, s in enumerate(np.unique(sections)):
        m = sections == s
        C[m] -= C[m].min(0)
        C[m, 0] += i * 1e6
    obs = pd.DataFrame({"section": pd.Categorical(sections), "group": pd.Categorical(np.asarray(labels).astype(str))})
    obs.index = obs.index.astype(str)
    a = ad.AnnData(X=np.zeros((len(C), 1)) if X is None else X, obs=obs)
    if genes is not None:
        a.var_names = list(genes)
    a.obsm["spatial"] = C
    return a


def neighbourhood_labels(state, strata) -> np.ndarray:
    q20, q80 = np.quantile(state, [0.2, 0.8])
    lab = np.asarray(strata).astype(object).copy()
    lab[state >= q80] = "state_high"
    lab[state <= q20] = "state_low"
    return lab.astype(str)


def neighbourhood(state, strata, coords_um, sections, *, moran_median, involves_space: bool,
                  n_perms: int = 1000, seed: int = 0) -> dict:
    if moran_median is None or moran_median < MORAN_DIFFUSE:
        return _skipped(f"state score Moran's I median {moran_median} < {MORAN_DIFFUSE} (1.3: spatially diffuse)")
    if not involves_space:
        return _skipped("no neighbour/contrast version and no morphology candidate in the finding")
    import squidpy as sq

    a = _anndata(coords_um, sections, neighbourhood_labels(np.asarray(state, float), strata))
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    sq.gr.nhood_enrichment(a, cluster_key="group", n_perms=n_perms, seed=seed, show_progress_bar=False)
    cats = list(a.obs["group"].cat.categories)
    z = a.uns["group_nhood_enrichment"]["zscore"]
    interval = np.linspace(50, 500, 10)
    sq.gr.co_occurrence(a, cluster_key="group", interval=interval, show_progress_bar=False)
    occ = a.uns["group_co_occurrence"]["occ"]          # (cats, cats, len(interval) - 1)
    hi = cats.index("state_high")
    curves = {c: occ[hi, j].tolist() for j, c in enumerate(cats) if not c.startswith("state_")}
    return {"status": "run", "categories": cats, "zscore": np.asarray(z).tolist(),
            "state_high_vs_strata_z": {c: float(z[hi, j]) for j, c in enumerate(cats)},
            "co_occurrence_interval_um": interval.tolist(), "state_high_co_occurrence": curves,
            "n_perms": n_perms, "graph": "6 nearest spots within section (squidpy generic)"}


# ----------------------------------------------------------------------------- 1.6
def lr_genes_consensus() -> set:
    import liana as li

    res = li.rs.select_resource("consensus")
    out = set()
    for col in ("ligand", "receptor"):
        for x in res[col].astype(str):
            out.update(x.split("_"))
    return out


def liana_condition(constituent_genes: Sequence[str], nhood: dict | None) -> tuple[bool, str]:
    lr = lr_genes_consensus()
    hit = sorted(set(constituent_genes) & lr)
    if hit:
        return True, f"constituent genes annotated as ligand/receptor in the consensus resource: {hit[:10]}"
    if nhood and nhood.get("status") == "run":
        enr = [c for c, zz in nhood["state_high_vs_strata_z"].items() if not c.startswith("state_") and zz > 1.96]
        if len(enr) >= 2:
            return True, f"state-high regions enriched for two strata: {enr}"
    return False, "no ligand/receptor constituent and no two-stratum enrichment in 1.5"


def liana_bivariate(logcpm, genes, state, coords_um, sections, *, condition: tuple[bool, str],
                    n_perm: int = 999, seed: int = 0, bandwidth: float = 150.0) -> dict:
    if not condition[0]:
        return _skipped(condition[1])
    import liana as li

    state = np.asarray(state, float)
    q20, q80 = np.quantile(state, [0.2, 0.8])
    keep = (state >= q80) | (state <= q20)
    sections = np.asarray(sections).astype(str)

    def _run(bw):
        import scipy.sparse as sp

        X = logcpm[keep] if not sp.issparse(logcpm) else logcpm.tocsr()[np.flatnonzero(keep)]
        a = _anndata(np.asarray(coords_um)[keep], sections[keep], np.where(state[keep] >= q80, "high", "low"),
                     X=sp.csr_matrix(X), genes=genes)
        li.ut.spatial_neighbors(a, bandwidth=bw, cutoff=0.1, set_diag=False)
        lr = li.mt.bivariate(a, local_name="cosine", resource_name="consensus", n_perms=None, use_raw=False,
                             verbose=False)
        return a, lr

    try:
        a, lr = _run(bandwidth)
        repair = None
    except Exception as e1:  # one repair attempt (wider kernel), then not assessed
        try:
            a, lr = _run(bandwidth * 2)
            repair = f"first attempt failed ({type(e1).__name__}: {e1}); bandwidth doubled"
        except Exception as e2:
            return {"status": "not assessed", "reason": f"LIANA+ failed after one repair attempt: {e2}"}
    L = lr.X.toarray() if hasattr(lr.X, "toarray") else np.asarray(lr.X)
    grp = np.asarray(a.obs["group"]).astype(str)
    sec = np.asarray(a.obs["section"]).astype(str)
    secs = np.unique(sec)
    # per-section difference of mean local score (high - low); permutation over sections = sign flips
    diffs = []
    for s in secs:
        m = sec == s
        if (grp[m] == "high").sum() and (grp[m] == "low").sum():
            diffs.append(L[m & (grp == "high")].mean(0) - L[m & (grp == "low")].mean(0))
    if len(diffs) < 3:
        return {"status": "not assessed", "reason": f"only {len(diffs)} sections carry both quintiles"}
    D = np.vstack(diffs)
    obs = D.mean(0)
    rng = np.random.default_rng(seed)
    null = np.vstack([(D * rng.choice([-1, 1], size=(len(D), 1))).mean(0) for _ in range(n_perm)])
    p = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (n_perm + 1)
    from .replication_v03 import bh as _bh

    q = _bh(p)
    names = list(lr.var_names)
    o = np.argsort(p)
    return {"status": "run", "repair": repair, "n_pairs": len(names), "n_sections": len(D),
            "top_pairs": [{"pair": names[i], "mean_diff_high_minus_low": float(obs[i]), "p": float(p[i]),
                           "q": float(q[i])} for i in o[:25]],
            "n_q_le_0.05": int((q <= 0.05).sum()),
            "metric": "LIANA+ spatially weighted cosine, consensus resource",
            "test": "per-section mean difference, sign-flip permutation over sections"}


# ----------------------------------------------------------------------------- 1.7
def regions(coords_um, sections, state, fractions, size_um: float = 250.0) -> dict:
    """Non-overlapping 250 um square windows per section with >= 3 spots: mean state, mean fractions."""
    out = {"section": [], "centre": [], "state": [], "fractions": [], "spot_rows": []}
    sections = np.asarray(sections).astype(str)
    C = np.asarray(coords_um, float)
    for s in np.unique(sections):
        ix = np.flatnonzero(sections == s)
        ij = np.floor((C[ix] - C[ix].min(0)) / size_um).astype(int)
        keys = {}
        for r, k in zip(ix, map(tuple, ij)):
            keys.setdefault(k, []).append(r)
        for k, rows in keys.items():
            if len(rows) < 3:
                continue
            rows = np.asarray(rows)
            out["section"].append(s)
            out["centre"].append(C[rows].mean(0).tolist())
            out["state"].append(float(np.mean(state[rows])))
            out["fractions"].append(np.asarray(fractions)[rows].mean(0).tolist())
            out["spot_rows"].append(rows.tolist())
    return out


def matched_high_low(reg: dict, n: int = 16, max_dist: float = 1.0, quantile: float = 0.2) -> dict:
    """State-high (top quintile) vs state-low (bottom quintile) regions matched on composition by
    optimal assignment on the Mahalanobis distance (pooled covariance of region fractions), pairs kept
    when the distance is < ``max_dist``; the n closest pairs are returned."""
    from scipy.optimize import linear_sum_assignment

    st = np.asarray(reg["state"])
    F = np.asarray(reg["fractions"])
    hi = np.flatnonzero(st >= np.quantile(st, 1 - quantile))
    lo = np.flatnonzero(st <= np.quantile(st, quantile))
    VI = np.linalg.pinv(np.cov(F.T) + 1e-6 * np.eye(F.shape[1]))
    d = F[hi][:, None, :] - F[lo][None, :, :]
    D = np.sqrt(np.einsum("ijk,kl,ijl->ij", d, VI, d))
    rr, cc = linear_sum_assignment(np.where(D < max_dist, D, 1e6))
    pairs = [(hi[a], lo[b], D[a, b]) for a, b in zip(rr, cc) if D[a, b] < max_dist]
    pairs.sort(key=lambda t: t[2])
    pairs = pairs[:n]
    return {"pairs": [{"high": int(a), "low": int(b), "mahalanobis": float(x)} for a, b, x in pairs],
            "n_pairs": len(pairs), "n_requested": n,
            "status": "run" if len(pairs) == n else "run (fewer matched pairs than requested)"}


def pathologist_packet(reg: dict, match: dict, out_dir, patch_loader: Callable | None = None,
                       seed: int = 20261021) -> dict:
    """Blinded packet: shuffled region ids with montages (when ``patch_loader(section, spot_rows)``
    returns an RGB array), a free-text + forced-choice form, and a separately stored, hashed key."""
    from .locks import canonical_hash

    os.makedirs(out_dir, exist_ok=True)
    items = [(m["high"], "high") for m in match["pairs"]] + [(m["low"], "low") for m in match["pairs"]]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    form, key = [], {}
    for k, i in enumerate(order):
        rid, lab = items[i]
        bid = f"R{k + 1:02d}"
        entry = {"blinded_id": bid, "section": reg["section"][rid], "centre_um": reg["centre"][rid],
                 "descriptors": "", "set_choice": ""}
        if patch_loader is not None:
            try:
                img = patch_loader(reg["section"][rid], reg["spot_rows"][rid])
                if img is not None:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    p = os.path.join(out_dir, f"{bid}.png")
                    plt.imsave(p, img)
                    entry["image"] = os.path.basename(p)
            except Exception as e:
                entry["image_error"] = str(e)
        form.append(entry)
        key[bid] = lab
    key_doc = {"key": key, "pairs": match["pairs"]}
    Path(out_dir, "form.json").write_text(json.dumps({
        "instructions": "For each region record free-text morphological descriptors (descriptive, never a "
                        "diagnosis). Then assign every region to set A or set B so that each set holds the "
                        "regions you judge to share a state. The state score is hidden.",
        "regions": form}, indent=1))
    kp = Path(out_dir).parent / (Path(out_dir).name + "_KEY_do_not_open.json")
    kp.write_text(json.dumps(key_doc, indent=1))
    return {"status": "run", "n_regions": len(form), "form": str(Path(out_dir, "form.json")),
            "key_sha256": canonical_hash(key_doc), "key_path": str(kp)}


def score_read(form_answers: dict, key: dict) -> dict:
    """Forced-choice agreement: the better of the two set-to-label assignments."""
    ids = [b for b in key if b in form_answers]
    a = np.mean([(form_answers[b] == "A") == (key[b] == "high") for b in ids])
    return {"n": len(ids), "agreement": float(max(a, 1 - a))}
