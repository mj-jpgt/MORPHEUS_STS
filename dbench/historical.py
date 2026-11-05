"""Historical rediscovery: published spatial findings as known-answer tests.

The frozen protocol (``dbench/protocols/historical_rediscovery_v01.json``)
names, per task, a target gene set, answer proxy gene sets, the transforms under
which the answer is stated (own / nbr6 / nbr18 / contrast) and its direction.
Every method receives the same design built from a ``StudyEpisode`` (support
rows only, answer variables never named) and returns ranked variable sets; the
top 20 are scored by one rule, identical for the learned search and every
baseline:

    a hypothesis is a hit if any of its variables carries one of the task's
    stated transforms and has within-patient |r| >= 0.9 with the answer proxy
    score under that transform (same rows), and sign(r_proxy) x sign(r_target)
    equals the stated direction.

The first hit is then frozen and evaluated on the held-out (query) patients
with ``general_discovery.evaluate_frozen_candidate`` (relation ``v07_ho`` when
available, else ``v06``) against the nuisance comparison model (depth,
platform).

Names.  Design columns are ``<view>:<column>`` (own) and
``<transform>:<view>:<column>`` for nbr6 / nbr18 / contrast, the grammar of
``learned_search.propose`` and of the benchmark baselines' ``nbr6:`` names.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TRANSFORMS: tuple[str, ...] = ("own", "nbr6", "nbr18", "contrast")
VIEW_ALIASES = {"genes": "genes", "raw_genes": "genes", "programmes": "programmes"}
R_THRESHOLD = 0.9
TOP_K = 20


# ----------------------------------------------------------------- protocol
def load_tasks(path: str | Path) -> list[dict]:
    """Tasks of a frozen historical protocol, each carrying the protocol's id,
    hash and scoring block (task-level scoring keys override the defaults)."""
    from dbench.prospective_campaign import canonical_hash

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != "morpheus_historical_rediscovery/v1":
        raise ValueError("unsupported historical protocol schema")
    sha = canonical_hash(doc)
    defaults = dict(doc.get("scoring") or {})
    pool_defaults = dict(doc.get("given_to_morpheus_defaults") or {})
    out = []
    for task in doc["tasks"]:
        for key in ("task_id", "study_key", "publication", "finding_text", "relation", "given_to_morpheus",
                    "proxy_sets", "scoring", "modality"):
            if key not in task:
                raise ValueError(f"task {task.get('task_id')}: missing {key}")
        rel = task["relation"]
        if rel["target"] not in task["proxy_sets"]:
            raise ValueError(f"task {task['task_id']}: target {rel['target']} has no proxy set")
        for a in rel["answer_variables"]:
            if a not in task["proxy_sets"]:
                raise ValueError(f"task {task['task_id']}: answer {a} has no proxy set")
        if any(t not in TRANSFORMS for t in _as_list(rel["transform"])):
            raise ValueError(f"task {task['task_id']}: unknown transform in {rel['transform']}")
        if rel["direction"] not in (1, -1):
            raise ValueError(f"task {task['task_id']}: direction must be +1 or -1")
        t = json.loads(json.dumps(task))
        t["scoring"] = {**defaults, **t["scoring"]}
        t["given_to_morpheus"] = {**pool_defaults, **t["given_to_morpheus"]}
        t["protocol_id"] = doc.get("protocol_id")
        t["protocol_sha256"] = sha
        out.append(t)
    return out


def before_date(task: Mapping[str, Any]) -> str:
    """Literature cutoff of a task: the earliest public date of its finding."""
    pub = task["publication"]
    return str(pub.get("before_date") or pub["first_published"])


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


# ------------------------------------------------------------ arithmetic
def _groups(episode) -> np.ndarray:
    return np.unique(np.asarray(episode.patient), return_inverse=True)[1].astype(np.int64)


def _centre(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for g in np.unique(groups):
        m = groups == g
        out[m] = values[m] - values[m].mean(0)
    return out


def _standardise(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    c = _centre(values, groups)
    out = np.empty_like(c)
    for g in np.unique(groups):
        m = groups == g
        sd = c[m].std(0)
        out[m] = c[m] / np.where(sd > 1e-12, sd, 1.0)
    return out


def neighbour_mean(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Mean over -1 padded neighbours (a row without neighbours keeps its own value)."""
    from dbench.benchmark.substrate import neighbour_mean as _nm
    return _nm(np.asarray(values, dtype=np.float64), np.asarray(neighbours, dtype=np.int64))


def apply_transform(values: np.ndarray, transform: str, neighbours: Mapping[int, np.ndarray]) -> np.ndarray:
    """own / nbr6 / nbr18 / contrast (= own - nbr6) of a column or matrix on episode rows."""
    values = np.asarray(values, dtype=np.float64)
    if transform == "own":
        return values
    if transform == "nbr6":
        return neighbour_mean(values, neighbours[6])
    if transform == "nbr18":
        return neighbour_mean(values, neighbours[18])
    if transform == "contrast":
        return values - neighbour_mean(values, neighbours[6])
    raise ValueError(f"unknown transform {transform!r}")


def within_patient_r(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> float:
    """Pearson r of within-patient-centred values (pooled over patients)."""
    a = _centre(np.asarray(a, dtype=np.float64).reshape(-1, 1), groups).ravel()
    b = _centre(np.asarray(b, dtype=np.float64).reshape(-1, 1), groups).ravel()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return 0.0 if den <= 1e-15 else float((a * b).sum() / den)


def parse_variable(name: str) -> tuple[str, str]:
    """``"nbr6:genes:COL1A1"`` -> ``("nbr6", "genes:COL1A1")``; ``"own:x"`` and bare names are own.
    Nested prefixes (a method's own neighbour mean of a precomputed column) compose with '.'"""
    head, _, rest = str(name).partition(":")
    if head in TRANSFORMS and rest:
        inner_t, base = parse_variable(rest)
        if head == "own":
            return inner_t, base
        return (head if inner_t == "own" else f"{head}.{inner_t}"), base
    return "own", str(name)


# --------------------------------------------------------------- gene panel
def _gene_matrix(episode, gene_matrix=None) -> tuple[np.ndarray, list]:
    """(n, G) within-patient-centred log1p-CPM and symbols: the full panel when
    given (``gene_matrix=(X, symbols)``, rows aligned to the episode), else the
    episode's genes view (already within-patient centred)."""
    groups = _groups(episode)
    if gene_matrix is not None:
        X, symbols = gene_matrix
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float64)
        if len(X) != episode.n:
            raise ValueError("gene_matrix rows must align with the episode rows")
        return _centre(X, groups), [str(s) for s in symbols]
    return np.asarray(episode.views["genes"], dtype=np.float64), [str(s) for s in episode.columns["genes"]]


def _set_genes(task: Mapping[str, Any], name: str) -> list[str]:
    drop = set(task.get("shared_genes_removed_from_both_scores") or [])
    return [g for g in task["proxy_sets"][name]["genes"] if g not in drop]


def set_score(episode, genes: Sequence[str], *, gene_matrix=None) -> tuple[np.ndarray, list[str]]:
    """Mean within-patient-centred log1p-CPM over the set's present genes, and the present genes."""
    X, symbols = _gene_matrix(episode, gene_matrix)
    index = {s: j for j, s in enumerate(symbols)}
    present = [g for g in genes if g in index]
    if not present:
        return np.full(episode.n, np.nan), []
    return X[:, [index[g] for g in present]].mean(1), present


def proxy_scores(episode, task: Mapping[str, Any], *, gene_matrix=None) -> dict[str, np.ndarray]:
    """Every proxy set's score on all episode rows under every transform:
    ``"<set>"`` (own) and ``"<set>@<transform>"``; present-gene lists under
    ``"_present"`` (a dict, not an array) for reporting."""
    out: dict[str, Any] = {}
    present: dict[str, list] = {}
    groups = _groups(episode)
    for name in task["proxy_sets"]:
        own, genes = set_score(episode, _set_genes(task, name), gene_matrix=gene_matrix)
        present[name] = genes
        out[name] = own
        for t in TRANSFORMS[1:]:
            out[f"{name}@{t}"] = (np.full(episode.n, np.nan) if not genes
                                  else _centre(apply_transform(own, t, episode.neighbours).reshape(-1, 1),
                                               groups).ravel())
    out["_present"] = present
    return out


def task_rows(episode, task: Mapping[str, Any], scores: Mapping[str, Any]) -> np.ndarray:
    """Rows the task is defined on (all, or a declared own-score quantile rule within sample)."""
    rule = (task["given_to_morpheus"].get("rows") or {"rule": "all"})
    if rule.get("rule", "all") == "all":
        return np.arange(episode.n)
    if rule["rule"] == "own_score_at_or_below_within_sample_quantile":
        s = np.asarray(scores[rule["proxy"]], dtype=np.float64)
        keep = np.zeros(episode.n, dtype=bool)
        samples = np.asarray(episode.sample_id)
        for sid in np.unique(samples):
            m = samples == sid
            keep[m] = s[m] <= np.quantile(s[m], float(rule["quantile"]))
        return np.flatnonzero(keep)
    raise ValueError(f"unknown row rule {rule['rule']!r}")


def target_values(episode, task: Mapping[str, Any], scores: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    """The task target (own-spot target-set score) within-patient standardised on ``rows``."""
    raw = np.asarray(scores[task["relation"]["target"]], dtype=np.float64)[rows]
    return _standardise(raw.reshape(-1, 1), _groups(episode)[rows]).ravel()


# ------------------------------------------------------------------- design
def build_design(episode, task: Mapping[str, Any], rows: np.ndarray, y: np.ndarray) -> dict:
    """The design every method receives: pool views under every transform
    (computed on all episode rows, then restricted to ``rows``), within-patient
    centred and z-scored on support rows; target genes and leak-guarded own
    columns removed."""
    groups_all = _groups(episode)
    gm = task["given_to_morpheus"]
    pools = [VIEW_ALIASES.get(v, v) for v in gm.get("variable_pools", ["genes", "programmes"])]
    target_genes = set(task["proxy_sets"][task["relation"]["target"]]["genes"])
    transforms = [t for t in gm.get("transforms", TRANSFORMS) if t in TRANSFORMS]
    support_mask = np.isin(rows, np.asarray(episode.support_rows))
    g = groups_all[rows]
    blocks, names, views, roles = [], [], {}, {}
    excluded: dict[str, list] = {"target_genes": [], "leak_guard": [], "episode_exclusions": []}
    start = 0
    yc = _centre(np.asarray(y, dtype=np.float64).reshape(-1, 1), g).ravel()
    for view in pools:
        X = episode.views.get(view)
        if X is None:
            continue
        cols = list(episode.columns.get(view) or [f"{view}{j}" for j in range(X.shape[1])])
        ep_ex = set(int(j) for j in (episode.exclusions or {}).get(view, ()))
        keep = []
        for j, c in enumerate(cols):
            if view == "genes" and str(c) in target_genes:
                excluded["target_genes"].append(f"{view}:{c}")
            elif j in ep_ex:
                excluded["episode_exclusions"].append(f"{view}:{c}")
            else:
                keep.append(j)
        Xv = np.asarray(X, dtype=np.float64)[:, keep]
        base_names = [f"{view}:{cols[j]}" for j in keep]
        # leak guard on own columns (support rows)
        own = _centre(Xv[rows], g)
        s = support_mask
        num = own[s].T @ yc[s]
        den = np.sqrt((own[s] ** 2).sum(0) * (yc[s] ** 2).sum()) + 1e-12
        guard = np.abs(num / den) >= R_THRESHOLD
        excluded["leak_guard"] += [base_names[j] for j in np.flatnonzero(guard)]
        for t in transforms:
            sel = ~guard if t == "own" else np.ones(len(base_names), dtype=bool)
            M = apply_transform(Xv[:, sel], t, episode.neighbours)[rows]
            M = _centre(M, g)
            sd = M[support_mask].std(0)
            M = (M - M[support_mask].mean(0)) / np.where(sd > 1e-8, sd, 1.0)
            key = view if t == "own" else f"{t}_{view}"
            kept = [n for n, k in zip(base_names, sel) if k]
            blocks.append(M.astype(np.float32))
            names += [n if t == "own" else f"{t}:{n}" for n in kept]
            views[key] = np.arange(start, start + len(kept))
            roles[key] = "variable" if t == "own" else "context"
            start += len(kept)
    X = np.column_stack(blocks) if blocks else np.zeros((len(rows), 0), dtype=np.float32)
    return {"X": X, "names": names, "views": views, "roles": roles, "excluded": excluded}


def _remap(nb: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    remap = -np.ones(n, dtype=np.int64)
    remap[rows] = np.arange(len(rows))
    sub = np.asarray(nb, dtype=np.int64)[rows]
    mapped = np.where(sub >= 0, remap[np.maximum(sub, 0)], -1)
    out = -np.ones_like(mapped)
    for i, row in enumerate(mapped):
        v = row[row >= 0]
        out[i, :len(v)] = v
    return out


def _nuisance(episode, rows: np.ndarray) -> tuple[np.ndarray, list]:
    g = _groups(episode)[rows]
    cols = [_centre(np.asarray(episode.views["depth"], dtype=np.float64).reshape(-1, 1)[rows], g)]
    names = ["depth_log1p"]
    plat = np.asarray(episode.meta.get("platform_of_row") or [episode.platform] * episode.n)[rows]
    if len(set(plat.tolist())) > 1:
        for p in sorted(set(plat.tolist()))[1:]:
            cols.append((plat == p).astype(np.float64).reshape(-1, 1))
            names.append(f"platform:{p}")
    return np.column_stack(cols), names


# ------------------------------------------------------------------ scoring
def _resolve_values(name: str, design: Mapping[str, Any] | None, episode, rows: np.ndarray) -> np.ndarray:
    """Values of a named variable on ``rows``: a design column, else built from the
    episode (base column of a pool view under the parsed transform chain)."""
    if design is not None:
        idx = design.get("_index")
        if idx is None:
            idx = design["_index"] = {n: i for i, n in enumerate(design["names"])}
        if name in idx:
            return np.asarray(design["X"][:, idx[name]], dtype=np.float64)
    t, base = parse_variable(name)
    view, _, col = base.partition(":")
    view = VIEW_ALIASES.get(view, view)
    cols = [str(c) for c in (episode.columns.get(view) or [])]
    if view not in episode.views or col not in cols:
        raise KeyError(f"cannot resolve variable {name!r}")
    v = np.asarray(episode.views[view], dtype=np.float64)[:, cols.index(col)]
    for step in reversed(t.split(".")):
        v = apply_transform(v, step, episode.neighbours)
    return v[rows]


def answer_ceiling(design: Mapping[str, Any], scores: Mapping[str, Any], rows: np.ndarray, groups: np.ndarray,
                   answers: Sequence[str], allowed: Sequence[str], n_top: int = 10) -> dict:
    """Per answer proxy and stated transform: the design columns under that transform ranked by
    within-patient |r| with the proxy score; returns the top ``n_top`` (name, r), the |r| of the
    n_top-th column (the representative threshold) and the ceiling (largest |r|); ``"task_ceiling"``
    is the maximum over every (proxy, transform)."""
    X = _centre(np.asarray(design["X"], dtype=np.float64), groups)
    norms = np.sqrt((X ** 2).sum(0))
    out: dict[str, Any] = {}
    task_ceiling = 0.0
    for a in answers:
        for t in allowed:
            p = np.asarray(scores[a if t == "own" else f"{a}@{t}"], dtype=np.float64)[rows]
            if not np.isfinite(p).all():
                continue
            pc = _centre(p.reshape(-1, 1), groups).ravel()
            cols = [i for i, nm in enumerate(design["names"]) if parse_variable(nm)[0] == t]
            if not cols:
                continue
            r = (X[:, cols].T @ pc) / (norms[cols] * np.sqrt((pc ** 2).sum()) + 1e-12)
            order = np.argsort(-np.abs(r), kind="stable")
            top = [(design["names"][cols[i]], float(r[i])) for i in order[:n_top]]
            out[f"{a}@{t}"] = {"top": top, "threshold": float(abs(top[-1][1])), "ceiling": float(abs(top[0][1]))}
            task_ceiling = max(task_ceiling, float(abs(top[0][1])))
    out["task_ceiling"] = task_ceiling
    return out


def score_ranked_list(ranked_sets: Sequence[Any], episode, task: Mapping[str, Any], top_k: int = TOP_K, *,
                      scores: Mapping[str, Any] | None = None, rows: np.ndarray | None = None,
                      y: np.ndarray | None = None, design: Mapping[str, Any] | None = None,
                      gene_matrix=None) -> dict:
    """Apply the protocol hit rule to the first ``top_k`` ranked variable sets.

    Each entry is a sequence of variable names, or a dict ``{"names": [...],
    "values": (len(rows), k) array}`` carrying the variables' values on the task
    rows.  Returns ``rank_of_first_hit`` (1-based; ``math.inf`` if none),
    ``hit_variable``, ``hit_proxy``, ``hit_transform``, ``r`` (with the proxy),
    ``r_target``, ``n_evaluated`` (sets scored) and per-rank details."""
    if scores is None:
        scores = proxy_scores(episode, task, gene_matrix=gene_matrix)
    if rows is None:
        rows = task_rows(episode, task, scores)
    if y is None:
        y = target_values(episode, task, scores, rows)
    groups = _groups(episode)[rows]
    rel, sc = task["relation"], task.get("scoring") or {}
    k = int(sc.get("top_k", top_k) if top_k is None else top_k)
    thr = float(sc.get("r_threshold", R_THRESHOLD))
    answers = list(sc.get("answer_proxies") or rel["answer_variables"])
    allowed = list(sc.get("answer_transforms") or _as_list(rel["transform"]))
    direction = int(rel["direction"])
    v02 = sc.get("hit_rule_version") == "v02_top10_representative"
    ceiling = None
    if v02:
        if design is None:
            design = build_design(episode, task, rows, y)
        ceiling = answer_ceiling(design, scores, rows, groups, answers, allowed, int(sc.get("representative_top_n", 10)))
        min_r = float(sc.get("representative_min_abs_r", 0.5))
    best_signed = None  # secondary metric: largest proxy |r| with the stated sign among top-k variables
    details, first = [], None
    for rank, entry in enumerate(list(ranked_sets)[:k], start=1):
        names = list(entry["names"]) if isinstance(entry, Mapping) else [str(v) for v in entry]
        values = None if not isinstance(entry, Mapping) else entry.get("values")
        best = None
        for j, name in enumerate(names):
            t = (entry.get("transforms") or [None] * len(names))[j] if isinstance(entry, Mapping) else None
            t = t or parse_variable(name)[0]
            if t not in allowed:
                continue
            v = (np.asarray(values, dtype=np.float64)[:, j] if values is not None
                 else _resolve_values(name, design, episode, rows))
            r_t = within_patient_r(v, y, groups)
            for a in answers:
                p = np.asarray(scores[a if t == "own" else f"{a}@{t}"], dtype=np.float64)[rows]
                if not np.isfinite(p).all():
                    continue
                r = within_patient_r(v, p, groups)
                sign_ok = int(np.sign(r) * np.sign(r_t)) == direction
                if v02:
                    c = ceiling.get(f"{a}@{t}")
                    hit = c is not None and sign_ok and abs(r) >= max(min_r, c["threshold"] - 1e-12)
                    if sign_ok and (best_signed is None or abs(r) > abs(best_signed["r"])):
                        best_signed = {"variable": name, "transform": t, "proxy": a, "r": r, "rank": rank}
                else:
                    hit = abs(r) >= thr and sign_ok
                cand = {"variable": name, "transform": t, "proxy": a, "r": r, "r_target": r_t, "hit": bool(hit)}
                if best is None or (hit, abs(r)) > (best["hit"], abs(best["r"])):
                    best = cand
        details.append({"rank": rank, "names": names, "best": best})
        if first is None and best is not None and best["hit"]:
            first = (rank, best)
    out = {"rank_of_first_hit": first[0] if first else math.inf, "n_evaluated": len(details),
           "top_k": k, "r_threshold": thr, "answer_proxies": answers, "answer_transforms": allowed,
           "hit_variable": first[1]["variable"] if first else None,
           "hit_proxy": first[1]["proxy"] if first else None,
           "hit_transform": first[1]["transform"] if first else None,
           "r": first[1]["r"] if first else None, "r_target": first[1]["r_target"] if first else None,
           "hit_set": details[first[0] - 1]["names"] if first else None, "details": details}
    if v02:
        tc = ceiling["task_ceiling"]
        out.update({"hit_rule_version": "v02_top10_representative", "representative_min_abs_r": min_r,
                    "task_ceiling": tc,
                    "ceiling": {kk: vv for kk, vv in ceiling.items() if kk != "task_ceiling"},
                    "hit_abs_r": abs(first[1]["r"]) if first else None,
                    "best_signed_abs_r": None if best_signed is None else abs(best_signed["r"]),
                    "best_signed_variable": None if best_signed is None else best_signed["variable"],
                    "best_signed_over_ceiling": (None if best_signed is None or tc <= 0
                                                 else abs(best_signed["r"]) / tc)})
    return out


# ------------------------------------------------------------------ methods
def _learned_sets(design, y, groups, support, nb6_rows, coords, *, budget, seed, model, embed=None):
    from morpheus_encoder.learned_search.propose import propose, transform_columns

    Xs = np.asarray(design["X"][support], dtype=np.float64)
    nb = None if model is None else _remap(nb6_rows, support, len(design["X"]))
    res = propose(model, Xs, y[support], groups[support], nb, None if coords is None else coords[support],
                  list(design["names"]), {k: list(map(int, v)) for k, v in design["views"].items()},
                  embed=embed, budget=int(budget), seed=int(seed))
    all_names = list(design["names"]) + list(res.extra_names)
    extra_all = (transform_columns(res.extra_specs, np.asarray(design["X"], dtype=np.float64), nb6_rows,
                                   model=model) if res.extra_specs else np.zeros((len(design["X"]), 0)))
    ranked = []
    for c in res.candidates:
        idx = list(c.variables)
        vals = np.column_stack([design["X"][:, i] if i < design["X"].shape[1]
                                else extra_all[:, i - design["X"].shape[1]] for i in idx])
        ranked.append({"names": [all_names[i] for i in idx], "values": vals, "score": float(c.proposal_score)})
    return ranked, int(res.n_relationships_tested), {"report": res.report, "extra_all": extra_all,
                                                     "all_names": all_names}


def _baseline_sets(method, design, y, groups, support, nb6_rows, coords, *, budget, seed, n_threads, comparison):
    from dbench.benchmark.methods import MethodInput

    inp = MethodInput(X=np.asarray(design["X"][support], dtype=np.float64), y=y[support],
                      comparison=comparison[support], groups=groups[support],
                      neighbours=_remap(nb6_rows, support, len(design["X"])),
                      coords=None if coords is None else coords[support], names=list(design["names"]),
                      views={k: np.asarray(v, dtype=int) for k, v in design["views"].items()},
                      roles=dict(design["roles"]), family="molecular", seed=int(seed), n_threads=int(n_threads))
    out = method.run(inp, int(budget))
    ranked = []
    for h in out.hypotheses:
        names = list(h.variables)
        if h.relation == "spatial_lag" and len(names) == 1 and not names[0].startswith("nbr6:"):
            names = ["nbr6:" + names[0]]
        ranked.append({"names": names, "score": float(h.score), "relation": h.relation})
    tested = int((out.info or {}).get("n_relationships_tested") or len(ranked))
    return ranked, tested, {"available": bool(out.available), "info": _jsonable(out.info),
                            "seconds_method": float(out.seconds)}


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def _heldout(hit_values: np.ndarray, y: np.ndarray, groups: np.ndarray, support: np.ndarray, query: np.ndarray,
             comparison: np.ndarray, relation: str | None, n_boot: int) -> dict:
    from morpheus_encoder import general_discovery as gd

    if relation is None:
        relation = "v07_ho" if "v07_ho" in gd.RELATIONS else "v06"
    V = np.asarray(hit_values, dtype=np.float64)
    k = V.shape[1]
    cand = gd.Candidate(tuple(range(k)), feature_form="product" if k >= 2 else "linear_subspace",
                        weights=() if k >= 2 else (1.0,), proposal_family="historical")
    rec = gd.evaluate_frozen_candidate(cand, V[support], y[support], V[query], y[query],
                                       proposal_baseline=comparison[support], replication_baseline=comparison[query],
                                       proposal_groups=groups[support], replication_groups=groups[query],
                                       relation=relation, n_boot=int(n_boot))
    dec = dict(rec.decomposition or {})
    return {"relation": relation, "replication_gain": float(rec.replication_gain),
            "gain_general": dec.get("gain_general"), "gain_general_ci": dec.get("gain_general_ci"),
            "G_HO": dec.get("G_HO"), "G_HO_ci": dec.get("G_HO_ci"),
            "best_decomposition": dec.get("best_decomposition"),
            "replication_r2_full": float(rec.replication_r2_full),
            "replication_r2_simple": float(rec.replication_r2_simple),
            "n_proposal": int(rec.n_proposal), "n_replication": int(rec.n_replication)}


def run_task(task: Mapping[str, Any], episode, method_name: str, *, budget: int | None = None, model=None,
             gene_matrix=None, seed: int = 0, relation: str | None = None, n_threads: int = 4,
             n_boot: int = 200, methods: Mapping[str, Callable[[], Any]] | None = None,
             method_kwargs: Mapping[str, Any] | None = None) -> dict:
    """One task x one method: design from support rows, ranked sets, hit rule on the
    top ``top_k``, and the held-out general gain / G_HO of the first hit."""
    t0 = time.time()
    sc = task.get("scoring") or {}
    budget = int(budget if budget is not None else sc.get("budget", 128))
    top_k = int(sc.get("top_k", TOP_K))
    scores = proxy_scores(episode, task, gene_matrix=gene_matrix)
    rows = task_rows(episode, task, scores)
    y = target_values(episode, task, scores, rows)
    groups = _groups(episode)[rows]
    support = np.flatnonzero(np.isin(rows, np.asarray(episode.support_rows)))
    query = np.flatnonzero(np.isin(rows, np.asarray(episode.query_rows)))
    design = build_design(episode, task, rows, y)
    nb6_rows = _remap(episode.neighbours[6], rows, episode.n)
    coords = None if episode.coords is None else np.asarray(episode.coords, dtype=np.float64)[rows]
    comparison, comparison_names = _nuisance(episode, rows)
    base = {"task_id": task["task_id"], "study_key": task["study_key"], "method": method_name,
            "protocol_id": task.get("protocol_id"), "protocol_sha256": task.get("protocol_sha256"),
            "before_date": before_date(task), "untouched": bool(task.get("untouched", True)),
            "label": task.get("label"), "publication_doi": task["publication"].get("doi"),
            "relation": task["relation"], "budget": budget, "top_k": top_k, "seed": int(seed),
            "n_rows": int(len(rows)), "n_support_rows": int(len(support)), "n_query_rows": int(len(query)),
            "n_support_patients": int(len(np.unique(groups[support]))),
            "n_query_patients": int(len(np.unique(groups[query]))),
            "n_design_columns": int(design["X"].shape[1]), "excluded": design["excluded"],
            "proxy_genes_present": scores["_present"],
            "proxy_genes_declared": {k: len(_set_genes(task, k)) for k in task["proxy_sets"]},
            "comparison_columns": comparison_names, "study": getattr(episode, "study", None)}
    extra_all, all_names = None, list(design["names"])
    if method_name == "learned_search":
        ranked, tested, info = _learned_sets(design, y, groups, support, nb6_rows, coords, budget=budget,
                                             seed=seed, model=model)
        extra_all, all_names = info.pop("extra_all"), info.pop("all_names")
        info = {"available": True, "report": _jsonable(info["report"]), "model": None if model is None
                else type(model).__name__}
    else:
        registry = methods
        if registry is None:
            from dbench.benchmark.methods import METHODS as registry  # noqa: N811
        if method_name not in registry:
            return {**base, "available": False, "unavailable": f"method {method_name!r} not registered",
                    "rank_of_first_hit": "inf", "seconds": time.time() - t0}
        method = registry[method_name](**dict(method_kwargs or {}))
        ranked, tested, info = _baseline_sets(method, design, y, groups, support, nb6_rows, coords, budget=budget,
                                              seed=seed, n_threads=n_threads, comparison=comparison)
        if not info["available"]:
            return {**base, **info, "rank_of_first_hit": "inf", "n_relationships_tested": 0,
                    "seconds": time.time() - t0}
    ranked = list(ranked)[:budget]
    hit = score_ranked_list(ranked, episode, task, top_k, scores=scores, rows=rows, y=y, design=design)
    heldout = None
    if hit["hit_set"] is not None:
        entry = ranked[hit["rank_of_first_hit"] - 1]
        if entry.get("values") is not None:
            V = np.asarray(entry["values"], dtype=np.float64)
        else:
            V = np.column_stack([_resolve_values(n, design, episode, rows) for n in entry["names"]])
        try:
            heldout = _heldout(V, y, groups, support, query, comparison, relation, n_boot)
        except Exception as exc:  # reported, never hidden
            heldout = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    top = [{"rank": d["rank"], "names": d["names"], "best_match": d["best"]} for d in hit.pop("details")]
    rank = hit.pop("rank_of_first_hit")
    return _jsonable({**base, **info, "available": True, "n_relationships_tested": int(tested),
                      "n_sets_returned": len(ranked), "rank_of_first_hit": "inf" if math.isinf(rank) else int(rank),
                      **hit, "top": top, "heldout_hit": heldout, "seconds": time.time() - t0})


_RESCORE_CACHE: dict = {}


def rescore_saved(result: Mapping[str, Any], task: Mapping[str, Any], episode, *, gene_matrix=None,
                  relation: str | None = None, n_boot: int = 200) -> dict:
    """Re-score a saved run's top-k ranked lists (``result["top"]``) under ``task``'s hit rule
    without re-running the method; the held-out general gain / G_HO of a hit is evaluated afresh."""
    t0 = time.time()
    key = (id(episode), task.get("protocol_sha256"), task["task_id"])
    if key not in _RESCORE_CACHE:
        _RESCORE_CACHE.clear()
        scores = proxy_scores(episode, task, gene_matrix=gene_matrix)
        rows = task_rows(episode, task, scores)
        y = target_values(episode, task, scores, rows)
        _RESCORE_CACHE[key] = (scores, rows, y, build_design(episode, task, rows, y), _nuisance(episode, rows)[0])
    scores, rows, y, design, comparison = _RESCORE_CACHE[key]
    groups = _groups(episode)[rows]
    support = np.flatnonzero(np.isin(rows, np.asarray(episode.support_rows)))
    query = np.flatnonzero(np.isin(rows, np.asarray(episode.query_rows)))
    ranked = [list(x["names"]) for x in result["top"]]
    top_k = int((task.get("scoring") or {}).get("top_k", TOP_K))
    hit = score_ranked_list(ranked, episode, task, top_k, scores=scores, rows=rows, y=y, design=design)
    heldout = None
    if hit["hit_set"] is not None:
        V = np.column_stack([_resolve_values(nm, design, episode, rows) for nm in hit["hit_set"]])
        try:
            heldout = _heldout(V, y, groups, support, query, comparison, relation, n_boot)
        except Exception as exc:
            heldout = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    hit.pop("details")
    rank = hit.pop("rank_of_first_hit")
    keep = ("task_id", "study_key", "method", "before_date", "n_relationships_tested", "n_sets_returned",
            "budget", "top_k", "seconds", "top")
    return _jsonable({**{k: result.get(k) for k in keep}, "seconds_original_run": result.get("seconds"),
                      "rescored_from_protocol": result.get("protocol_sha256"),
                      "protocol_id": task.get("protocol_id"), "protocol_sha256": task.get("protocol_sha256"),
                      **hit, "rank_of_first_hit": "inf" if math.isinf(rank) else int(rank),
                      "heldout_hit": heldout, "available": True, "seconds_rescore": time.time() - t0})
