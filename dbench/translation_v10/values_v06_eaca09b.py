"""Candidate and target values on a spatial cohort (training spec 0.4, 0.6; replication rule 2.3).

Names accepted:

* the spec form ``<kind>:<name>@<version>`` with kind in {gene, hallmark, gavish, residual} and version in
  {v0, v6, v18, c} (own, 6-neighbour mean, 18-neighbour mean, contrast own - 6-neighbour mean);
* the legacy kidney-campaign form read by ``dbench.scripts.external_replication.parse_variable``
  (``raw_genes:X``, ``programmes:hallmark:X``, ``context_neighbour_mean:nbr:...``) and the legacy
  target form ``own:<Gavish MP>``.

Arithmetic is identical on every cohort: log1p CPM at 1e4 over the section's panel, gene-set score =
mean over the set's genes that entered the discovery computation, neighbours = the same counts (6, 18)
within section, never the same physical radius. A missing gene raises :class:`NotTestable`; nothing is
substituted. A residual programme is the frozen loading applied to the cohort's covariate-adjusted
expression; it is testable when the genes present carry at least ``RESIDUAL_COVERAGE`` of the
loading's squared norm (declared in the v03 protocol).
"""
from __future__ import annotations

import re

import numpy as np

RESIDUAL_COVERAGE = 0.95
VERSIONS = {"v0": 0, "v6": 6, "v18": 18, "c": "c"}


class NotTestable(Exception):
    """A constituent cannot be computed on this cohort without substitution."""


V07_TRANSFORMS = {"own": "v0", "nbr6": "v6", "nbr18": "v18", "contrast": "c"}


# These are deliberately local copies of the v08 discovery operators.  The
# confirmation worktree is on an older MORPHEUS checkout, so importing them
# from hest_multi would silently make confirmation depend on checkout version.
# Keeping the exact definitions here makes the external adapter self-contained
# and auditable.
def _adjacency(neighbours, n: int):
    import scipy.sparse as sp
    nb = np.asarray(neighbours)
    r, c = np.nonzero(nb >= 0)
    return sp.csr_matrix((np.ones(len(r), dtype=bool), (r, nb[r, c])), shape=(n, n))


def graph_rings(neighbours6, max_ring: int = 3) -> list:
    import scipy.sparse as sp
    n = len(neighbours6)
    A = _adjacency(neighbours6, n).astype(np.int32)
    seen = sp.identity(n, dtype=np.int32, format="csr")
    frontier, rings = seen, []
    for _ in range(int(max_ring)):
        reach = (frontier @ A + frontier).astype(bool).astype(np.int32)
        ring = (reach - reach.multiply(seen.astype(bool))).astype(bool)
        rings.append(ring.tocsr())
        seen = (seen + reach).astype(bool).astype(np.int32)
        frontier = ring.astype(np.int32)
    return rings


def compartment_mean(values: np.ndarray, adjacency, member: np.ndarray) -> tuple:
    import scipy.sparse as sp
    v = np.asarray(values, dtype=np.float64)
    v = v[:, None] if v.ndim == 1 else v
    # Cohort loaders expose k-nearest-neighbour indices as an (n, k)
    # rectangular array, while graph rings are square sparse matrices.  The
    # earlier conversion treated the former as matrix coordinates and could
    # silently fail shape checks.  Evaluate the indexed neighbour relation
    # directly, preserving the exact neighbour list and member mask.
    nb = np.asarray(adjacency)
    if nb.ndim == 2 and nb.shape[0] == len(v) and nb.shape[1] != len(v):
        valid = nb >= 0
        idx = np.where(valid, nb, 0).astype(np.int64, copy=False)
        eligible = valid & np.asarray(member, dtype=bool)[idx]
        count = eligible.sum(1).astype(float)
        gathered = v[idx]
        total = (gathered * eligible[..., None]).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count[:, None]
        mean[count == 0] = np.nan
        return mean, count
    a = sp.csr_matrix(adjacency).multiply(np.asarray(member, dtype=bool)[None, :]).tocsr().astype(float)
    count = np.asarray(a.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.asarray(a @ v) / count[:, None]
    mean[count == 0] = np.nan
    return mean, count


def neighbour_moments(values: np.ndarray, adjacency) -> tuple:
    import scipy.sparse as sp
    v = np.asarray(values, dtype=np.float64)
    v = v[:, None] if v.ndim == 1 else v
    a = sp.csr_matrix(adjacency).astype(float)
    count = np.asarray(a.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.asarray(a @ v) / count[:, None]
        mean_sq = np.asarray(a @ (v * v)) / count[:, None]
    mean[count == 0] = np.nan
    var = np.maximum(mean_sq - mean * mean, 0.0)
    var[count == 0] = np.nan
    return mean, var


def gradient_operator(coords: np.ndarray, neighbours: np.ndarray, min_neighbours: int = 3) -> tuple:
    import scipy.sparse as sp
    xy, nb, n = np.asarray(coords, dtype=float), np.asarray(neighbours), len(coords)
    ri, ci, wx, wy, ok = [], [], [], [], np.zeros(n, dtype=bool)
    for i in range(n):
        j = nb[i][nb[i] >= 0]
        if len(j) < min_neighbours:
            continue
        design = np.column_stack([xy[j] - xy[i], np.ones(len(j))])
        pinv = np.linalg.pinv(design)
        ri += [i] * len(j)
        ci += j.tolist()
        wx += pinv[0].tolist()
        wy += pinv[1].tolist()
        ok[i] = True
    return (sp.csr_matrix((wx, (ri, ci)), shape=(n, n)),
            sp.csr_matrix((wy, (ri, ci)), shape=(n, n)), ok)


#: Operator views of the registered v08 multiscale discovery (hest_multi.OPERATOR_VIEWS in the v08 worktree).
OPERATOR_VIEWS = {
    "compartment_neighbours": ("nbrS6", "nbrS18", "nbrE6", "nbrE18"),
    "rings": ("ring1", "ring2", "ring3", "ringS1", "ringS2", "ringS3", "ringE1", "ringE2", "ringE3"),
    "heterogeneity_gradient": ("het", "grad"),
}


def discovery_row_mask(coh, views, sets, discovery_panel=None, residual_ctx=None) -> np.ndarray:
    """Rows the registered discovery kept: it dropped (never imputed) every row where ANY operator of ANY
    operator view of the family is undefined (run_general_discovery_packet.py, undefined_op).  Definedness
    depends only on graph structure and compartment labels, so it is evaluated on one finite base column."""
    keep = np.ones(len(coh.patient), dtype=bool)
    for view in views:
        for op in OPERATOR_VIEWS[view]:
            v = cohort_values(coh, f"{op}:programmes:mp:EMT-I", sets, discovery_panel, residual_ctx)
            keep &= np.isfinite(np.asarray(v, float))
    return keep


def parse_candidate(name: str) -> dict:
    # blind-discovery (hest_multi / learned_search) names: <transform>:<view>:<column>
    # the own transform may be written without its prefix (receipts: "raw_genes:FOS", "programmes:mp:EMT-II")
    # v08 spatial-operator grammar.  These are not aliases for an own-spot
    # value: their operator must be rebuilt on each external section using the
    # same graph construction as discovery.  Compartment-specific operators
    # deliberately remain untestable until compatible compartment labels are
    # supplied; they must never fall back to an unlabeled ring mean.
    m = re.match(r"^(ring[123]|ring[SE][123]|het|grad):(raw_genes|programmes?|programme):(.+)$", name)
    if m:
        op, view, col = m.groups()
        base = ("raw_genes:" if view == "raw_genes" else
                ("programme:" if view == "programme" else "programmes:")) + col
        return {"name": name, "kind": "operator", "operator": op,
                "base": parse_candidate(base)}
    m = re.match(r"^(nbr[SE](?:6|18)):(raw_genes|programmes?):(.+)$", name)
    if m:
        op, view, col = m.groups()
        base = ("raw_genes:" if view == "raw_genes" else
                ("programme:" if view == "programme" else "programmes:")) + col
        return {"name": name, "kind": "operator", "operator": op,
                "base": parse_candidate(base)}
    m = re.match(r"^(?:(own|nbr6|nbr18|contrast):)?(raw_genes|programmes|morph_concepts|morph_dictionary_v03):(.+)$",
                 name)
    if m:
        t, view, col = m.groups()
        ver = V07_TRANSFORMS[t or "own"]
        if view == "morph_dictionary_v03":   # parsed (so a family can still be assembled); values raise
            return {"name": name, "kind": "morphology", "version": "v0",
                    "reason": f"{name}: v03 morphology dictionary centre (needs frozen H-Optimus-0 embeddings of the "
                              "external section: none are on the store; Zenodo carries spot patches they could be "
                              "extracted from, Li 2022 and GSE250163 carry no full-resolution image)"}
        if view == "raw_genes":
            return {"name": name, "kind": "gene", "genes_key": col, "version": ver}
        if view == "programmes":
            return {"name": name, "kind": "programme", "set": col, "version": ver}
        raise NotTestable(f"{name}: learned morphology concept (needs the frozen concept layer on a <= 0.5 um/px "
                          "image of the external section)")
    m = re.match(r"^(?:(own|nbr6|nbr18|contrast):)?(gene|programme):(.+)$", name)   # targets, e.g. contrast:gene:LDHA
    if m and "@" not in name:
        t, kind, col = m.groups()
        ver = V07_TRANSFORMS[t or "own"]
        if kind == "gene":
            return {"name": name, "kind": "gene", "genes_key": col, "version": ver}
        return {"name": name, "kind": "programme", "set": col, "version": ver}
    if re.match(r"^morph:k\d+$", name):   # v03 morphology dictionary centre as a target
        return {"name": name, "kind": "morphology", "version": "v0",
                "reason": f"{name}: v03 morphology dictionary centre target (needs H-Optimus-0 embeddings of an external "
                          "image <= 0.5 um/px; none available)"}
    # v08 residual targets are fixed source-space components.  The loading is
    # supplied separately, never refit on the external cohort.
    m = re.match(r"^resid:(G[1-9]\d*)$", name)
    if m:
        return {"name": name, "kind": "residual", "programme": m.group(1), "version": "v0"}
    m = re.match(r"^programme:(mp|hallmark):(.+)$", name)
    if m:
        return {"name": name, "kind": "programme", "set": f"{m.group(1)}:{m.group(2)}", "version": "v0"}
    if re.match(r"^morph_concept_\d+$", name):
        raise NotTestable(f"{name}: morphology-concept target (frozen concept layer on external images)")
    m = re.match(r"^(gene|hallmark|gavish|residual):(.+)@(v0|v6|v18|c)$", name)
    if m:
        kind, key, ver = m.groups()
        if kind == "gene":
            return {"name": name, "kind": "gene", "genes_key": key, "version": ver}
        if kind == "residual":
            return {"name": name, "kind": "residual", "programme": key, "version": ver}
        return {"name": name, "kind": "programme", "set": f"{'mp' if kind == 'gavish' else 'hallmark'}:{key}",
                "version": ver}
    m = re.match(r"^(own|nbr|nbr18):(.+)$", name)
    if m:
        return {"name": name, "kind": "programme", "set": f"mp:{m.group(2)}",
                "version": {"own": "v0", "nbr": "v6", "nbr18": "v18"}[m.group(1)]}
    from dbench.scripts.external_replication import parse_variable

    v = parse_variable(name)
    ver = {0: "v0", 6: "v6", 18: "v18"}[int(v.get("nbr", 0))]
    if v["kind"] == "gene":
        return {"name": name, "kind": "gene", "genes_key": v["gene"], "version": ver}
    if v["kind"] == "programme":
        return {"name": name, "kind": "programme", "set": v["set"], "version": ver}
    raise NotTestable(f"{name}: {v.get('reason', 'unsupported variable kind')}")


def set_genes(sets: dict, key: str) -> list:
    kind, nm = key.split(":", 1)
    for s in sets[kind]:
        if s["name"] == nm:
            return list(dict.fromkeys(s["genes"]))
    raise NotTestable(f"programme {key} not found")


def candidate_genes(c: dict, sets: dict, discovery_panel=None) -> list:
    if c["kind"] == "gene":
        return [c["genes_key"]]
    if c["kind"] == "programme":
        g = set_genes(sets, c["set"])
        return [x for x in g if discovery_panel is None or x in discovery_panel]
    raise ValueError(c["kind"])


def neighbour_mean(v, nb) -> np.ndarray:
    from morpheus.tier2.substrates.hest_multi import _neighbour_mean

    return _neighbour_mean(np.asarray(v, float), nb)


def apply_version(v, version: str, neighbours: dict) -> np.ndarray:
    if version == "v0":
        return np.asarray(v, float)
    if version == "c":
        return np.asarray(v, float) - neighbour_mean(v, neighbours[6])
    return neighbour_mean(v, neighbours[VERSIONS[version]])


def residual_projection(logcpm, genes, covariates, loading_genes, loading, alpha: float = 1.0, groups=None,
                        chunk: int = 1000):
    """Frozen loading applied to the covariate-adjusted expression: each gene regressed (ridge, intercept
    free) on the declared covariates, after within-patient centring of genes and covariates when ``groups``
    is given (per-patient intercepts), residual standardised per gene, dot the loading over present genes.
    Genes are processed in chunks so a sparse whole-transcriptome matrix never becomes dense at once.
    Returns (score, coverage)."""
    from .relation import within_centre

    gi = {g: i for i, g in enumerate(genes)}
    w = dict(zip(map(str, loading_genes), np.asarray(loading, float)))
    present = [g for g in w if g in gi]
    tot = float(sum(x * x for x in w.values()))
    cov = float(sum(w[g] ** 2 for g in present)) / tot if tot > 0 else 0.0
    C = np.asarray(covariates, float).reshape(logcpm.shape[0], -1)
    C = within_centre(C, groups) if groups is not None else C - C.mean(0)
    C = C / np.where(C.std(0) > 1e-12, C.std(0), 1.0)
    P = np.linalg.solve(C.T @ C + alpha * np.eye(C.shape[1]), C.T)
    score = np.zeros(logcpm.shape[0])
    for i in range(0, len(present), chunk):
        gs = present[i:i + chunk]
        X = logcpm[:, [gi[g] for g in gs]]
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, float)
        X = within_centre(X, groups) if groups is not None else X - X.mean(0)
        R = X - C @ (P @ X)
        sd = R.std(0)
        R /= np.where(sd > 1e-12, sd, 1.0)
        score += R @ np.array([w[g] for g in gs])
    return score, cov


def cohort_values(coh, name: str, sets: dict, discovery_panel=None, residual_ctx: dict | None = None):
    """Raw (untransformed) values of one candidate on a loaded ``external_replication.Cohort``."""
    c = parse_candidate(name)
    if c["kind"] == "operator":
        op, base = str(c["operator"]), c["base"]
        if (op.startswith(("ringS", "ringE", "nbrS", "nbrE")) and
                (not residual_ctx or "compartments" not in residual_ctx)):
            raise NotTestable(f"{name}: requires frozen external stromal/epithelial labels")
        # Evaluate the constituent at the own location, then apply the exact
        # graph operator per section.  The cohort loader's neighbour arrays
        # are section-local, matching the discovery contract.
        base_name = str(base["name"])
        v = cohort_values(coh, base_name, sets, discovery_panel, residual_ctx)
        n = len(v)
        if op.startswith("ring"):
            rings = graph_rings(np.asarray(coh.raw.neighbours[6], dtype=np.int64), 3)
            k = int(op[-1])
            if op.startswith(("ringS", "ringE")):
                if not residual_ctx or "compartments" not in residual_ctx:
                    raise NotTestable(f"{name}: requires frozen external stromal/epithelial labels")
                labels = residual_ctx["compartments"](coh)
                member = np.asarray(labels[op[4]], dtype=bool)
            else:
                member = np.ones(n, dtype=bool)
            M, _ = compartment_mean(np.asarray(v, float)[:, None], rings[k - 1], member)
            return np.asarray(M[:, 0], float)
        if op.startswith("nbr"):
            if not residual_ctx or "compartments" not in residual_ctx:
                raise NotTestable(f"{name}: requires frozen external stromal/epithelial labels")
            labels = residual_ctx["compartments"](coh)
            member = np.asarray(labels[op[3]], dtype=bool)
            k = int(op[4:])
            M, _ = compartment_mean(np.asarray(v, float)[:, None], coh.raw.neighbours[k], member)
            return np.asarray(M[:, 0], float)
        if op == "het":
            _, var = neighbour_moments(np.asarray(v, float)[:, None],
                                       _adjacency(np.asarray(coh.raw.neighbours[18], dtype=np.int64), n))
            return np.asarray(var[:, 0], float)
        if op == "grad":
            wx, wy, ok = gradient_operator(np.asarray(coh.raw.coords, dtype=float),
                                            np.asarray(coh.raw.neighbours[18], dtype=np.int64))
            out = np.sqrt(np.asarray(wx @ np.asarray(v, float)) ** 2 + np.asarray(wy @ np.asarray(v, float)) ** 2)
            out[~ok] = np.nan
            return out
        raise NotTestable(f"{name}: unsupported spatial operator {op}")
    if c["kind"] == "morphology":
        raise NotTestable(c["reason"])
    if c["kind"] == "residual":
        if not residual_ctx or c["programme"] not in residual_ctx.get("loadings", {}):
            raise NotTestable(f"{name}: no frozen loading supplied")
        lg, lw = residual_ctx["loadings"][c["programme"]]
        s, cov = residual_projection(coh.logcpm, list(coh.raw.genes), residual_ctx["covariates"](coh), lg, lw,
                                     groups=coh.patient)
        if cov < RESIDUAL_COVERAGE:
            raise NotTestable(f"{name}: present genes carry {cov:.3f} of the loading norm (< {RESIDUAL_COVERAGE})")
        return apply_version(s, c["version"], coh.raw.neighbours)
    genes = candidate_genes(c, sets, discovery_panel)
    miss = [g for g in genes if g not in coh.panel]
    if miss or not genes:
        raise NotTestable(f"{name}: {len(miss)} discovery gene(s) absent from the panel: {miss[:10]}")
    idx = [coh.panel[g] for g in genes]
    v = np.asarray(coh.logcpm[:, idx].mean(axis=1)).ravel()
    return apply_version(v, c["version"], coh.raw.neighbours)
