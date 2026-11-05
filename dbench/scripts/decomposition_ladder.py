"""Nested explanation ladder (N, B1, B2, B3) for the frozen hypotheses of a kidney run.

    python -m dbench.scripts.decomposition_ladder run  --run-dir R --out O [--screen pooled.json --screen-top 40]
                                                       [--shard i --n-shards n]
    python -m dbench.scripts.decomposition_ladder pool --run-dir R --out O

`run` scores every (hypothesis, registered fold) and writes one JSON per (family, receipt target,
fold) under O/cells/ (resumable: existing cells are skipped). `pool` pools folds per hypothesis
(patient bootstrap), assigns tiers and writes O/ladder.json, O/ladder.md and O/REPORT.md.

Hypotheses: every entry of the run's funnel_aggregate.json (frozen hypotheses incl. internal
objects) and, with --screen, the top columns of a Level-A linear screen (development only,
labelled so). Hypothesis values come from the run's own substrate (v0.4 run -> v0.4 episode and
family design, exactly as registered); the comparison ladder is always the v0.5 bank's registered
competitor (`hest.real_episode(substrate_version="v05", bank=True)`) on the same rows. For a v0.4
run the v0.4 competitor is scored as an extra level ("v04") and the v0.4 relation's gain there is
checked against the receipt's replication_gain. For a v0.5 run the B3 full-model R^2 is checked
against the receipt's replication_r2_full. See dbench/decomposition.py for the method.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from dbench import decomposition as dc

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
EMBED = f"{STORE}/external/hest_fm_embeddings_20260906/ccrcc/embeddings/hoptimus"
DATA = f"{STORE}/morpheus_substrate_mirror/hest/ccrcc"
PROTOCOL_V04 = "dbench/protocols/renal_spatial_v04.json"
PROTOCOL_V05 = "dbench/protocols/renal_spatial_v05.json"
#: the v0.5 protocol does not declare the neighbourhood-programme family; its views are v0.4's
LADDER_SUBSTRATE = "v05"


def read_protocol(path) -> dict:
    return json.loads(Path(path).read_text())


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


#: bank views the ladder episode needs: the v0.5 competitor reads only the PCA view (neighbour means of
#: the aligned PCs) and, for nbr_mp targets, the PCA and neighbour-mean views
LADDER_BANK_VIEWS = ("pca", "context_neighbour_mean")


def hypothesis_views(specs, family_cfg: dict, target_id: str) -> tuple:
    """Bank views needed to rebuild the hypotheses' columns (the view is the name's first field), the
    family's context view when it has a requirement, and the nbr_mp competitor's PCA views. Views are
    built independently from shared, response-free inputs, so restricting them changes no column (the
    receipt check confirms this on every frozen row)."""
    views = {str(v).split(":", 1)[0] for s in specs for v in s["variable_names"]}
    if "requirement" in family_cfg:
        views.add("cellvit_structure")
    if target_id.startswith("nbr_mp"):
        views |= {"pca", "context_neighbour_mean"}
    return tuple(sorted(views))


def install_moran_cache(cache_dir) -> None:
    """Memoise ``hest.moran_i`` in THIS process on disk, keyed by the sha256 of its exact inputs.

    The own_mp / nbr_mp targets rank the 41 programmes by within-patient Moran's I on the proposal
    rows (~14 min per episode on the shared box); the ranking is response-free and identical for every
    target of a fold and for the v0.4 and v0.5 episodes, so each value is computed once. A cached value
    is the float the function returned for byte-identical inputs. No module is edited: the name is
    rebound in this process only."""
    import hashlib
    from morpheus.tier2.substrates import hest
    if getattr(hest.moran_i, "_ladder_cached", False):
        return
    original = hest.moran_i
    os.makedirs(cache_dir, exist_ok=True)

    def cached(values, coords, groups=None, k=6, **kw):
        h = hashlib.sha256()
        for a in (values, coords, groups):
            if a is None:
                h.update(b"None"); continue
            a = np.ascontiguousarray(a)
            h.update(("%s%s" % (a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        h.update(("k=%r;%r" % (k, sorted(kw.items()))).encode())
        path = os.path.join(cache_dir, h.hexdigest() + ".json")
        if os.path.exists(path):
            return json.loads(Path(path).read_text())["value"]
        value = original(values, coords, groups, k=k, **kw)
        tmp = path + ".%d.tmp" % os.getpid()
        Path(tmp).write_text(json.dumps({"value": float(value)}))
        os.replace(tmp, path)
        return value
    cached._ladder_cached = True
    hest.moran_i = cached


def load_episode(substrate: str, fold: int, target_id: str, bank_views=None):
    from morpheus.tier2.substrates import hest
    proto = read_protocol(PROTOCOL_V05 if substrate == "v05" else PROTOCOL_V04)
    disc = proto["discovery"]
    return hest.real_episode(fold=fold, n_folds=int(disc["n_folds"]), seed=int(disc["fold_seed"]), target=target_id,
                             embed_root=EMBED, sample_ids=None, cohort="ccrcc", data_dir=DATA,
                             n_hvg=int(proto.get("n_hvg", 300)), min_depth=int(proto.get("min_depth", 200)),
                             program_sets=None, bank=True, substrate_version=substrate,
                             bank_views=None if bank_views is None else tuple(bank_views))


def family_design(ep, protocol: dict, family: str, target_id: str, views=None):
    from morpheus_encoder.hypothesis_generator import episode_design
    fam = protocol["discovery"]["relationship_families"][family]
    require_context = "requirement" in fam
    cand = list(fam["candidate_views"]) if views is None else [v for v in fam["candidate_views"] if v in views]
    return episode_design(ep, ep.observed_matrix(), target=target_id, response=np.asarray(ep.response, float),
                          candidate_views=cand,
                          context_views=("cellvit_structure",) if require_context else ())


def design_columns(d, variable_names) -> np.ndarray:
    names = list(d.column_names)
    idx = []
    for v in variable_names:
        if v in names:
            idx.append(names.index(v))
        elif v.split(":", 1)[-1] in names:
            idx.append(names.index(v.split(":", 1)[-1]))
        else:
            raise KeyError("variable %r not in the design" % v)
    return np.asarray(d.X[:, idx], float)


# ------------------------------------------------------------------ inputs --

def resolved_target(r: dict) -> str:
    for fld, kind in (("own_mp_program", "own"), ("nbr_mp_program", "nbr")):
        info = r.get(fld) or {}
        if info.get("program_name"):
            return "%s:%s" % (kind, info["program_name"])
    return str(r["target"])


def receipt_index(run_dir) -> dict:
    """{(family, fold, resolved target): {"target_id", "substrate", "rows": {tuple(vars): row}}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "receipts", "*", "f*_*.json"))):
        r = json.loads(Path(path).read_text())
        fam = r.get("family") or Path(path).parent.name
        rows = {}
        for c in r.get("candidates", []):
            rows[tuple(c.get("variable_names") or [])] = {
                k: c.get(k) for k in ("candidate_hash", "replication_gain", "replication_r2_full",
                                      "competitor_r2_replication", "decomposition", "feature_form", "validated")}
        out[(fam, int(r["fold"]), resolved_target(r))] = {
            "target_id": str(r["target"]), "substrate": r.get("substrate_version") or "v04", "rows": rows,
            "receipt": str(path)}
    return out


def screen_specs(path, top: int) -> tuple[list, dict]:
    """Top `top` columns of the Level-A linear screen: recurring_all minus near-duplicates of a
    competitor column, ordered by passing folds then mean linear gain over all folds."""
    pooled = json.loads(Path(path).read_text())
    pool = [e for e in pooled["recurring_all"] if not e.get("near_duplicate_of_competitor")]
    pool.sort(key=lambda e: (-int(e["n_passing"]), -float(e.get("mean_gain_all") or 0.0), e["target"], e["column"]))
    specs = []
    for e in pool[:top]:
        specs.append({"hypothesis_id": "levelA:%s:%s:%s" % (e["family"], e["target"], e["column"]),
                      "source": "Level-A linear screen (development only, not registered)",
                      "family": e["family"], "target": e["target"], "variable_names": [e["column"]],
                      "feature_form": "linear_subspace", "screen_passing_folds": e["passing_folds"],
                      "screen_mean_gain_all": e.get("mean_gain_all"),
                      "in_registered_pooled_frozen": e.get("in_registered_pooled_frozen")})
    rule = {"source": str(path), "pool": "recurring_all minus near_duplicate_of_competitor (%d)" % len(pool),
            "order": "n_passing desc, mean_gain_all desc", "top": int(top)}
    return specs, rule


def frozen_specs(run_dir) -> list:
    funnel = json.loads(Path(run_dir, "funnel_aggregate.json").read_text())
    out = []
    for e in funnel["table"]:
        out.append({"hypothesis_id": e["candidate_id"], "source": "registered run (frozen, stage %s)" % e["stage"],
                    "family": e["family"], "target": e["target"], "variable_names": list(e["variable_names"]),
                    "feature_form": e.get("feature_form", "linear_subspace"), "stage": e["stage"],
                    "registered_passing_folds": e.get("passing_folds"),
                    "registered_mean_gain": e.get("mean_replication_gain"),
                    "explanation_level": e.get("explanation_level")})
    return out


def receipt_specs(run_dir) -> list:
    """Before a run is pooled (no funnel_aggregate.json): every validated receipt row, one spec per
    (family, target, variables), labelled development (not pooled)."""
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(run_dir, "receipts", "*", "f*_*.json"))):
        r = json.loads(Path(path).read_text())
        fam = r.get("family") or Path(path).parent.name
        for c in r.get("candidates", []):
            if not c.get("validated"):
                continue
            key = (fam, resolved_target(r), tuple(c.get("variable_names") or []))
            if key in seen:
                continue
            seen.add(key)
            out.append({"hypothesis_id": "dev-receipt:%s:%s:%s" % (fam, key[1], ",".join(key[2])),
                        "source": "validated receipt row, run not pooled yet (development)", "family": fam,
                        "target": key[1], "variable_names": list(key[2]),
                        "feature_form": c.get("feature_form", "linear_subspace"), "stage": "development (not pooled)"})
    return out


def all_specs(run_dir, screen=None, screen_top=40) -> tuple[list, dict | None]:
    pooled = os.path.exists(os.path.join(run_dir, "funnel_aggregate.json"))
    specs = frozen_specs(run_dir) if pooled else receipt_specs(run_dir)
    rule = None
    if screen:
        extra, rule = screen_specs(screen, screen_top)
        seen = {(s["family"], s["target"], tuple(s["variable_names"])): s for s in specs}
        for s in extra:
            key = (s["family"], s["target"], tuple(s["variable_names"]))
            if key in seen:
                seen[key].setdefault("also", []).append(s["hypothesis_id"])
                rule.setdefault("overlap_with_frozen", []).append(s["hypothesis_id"])
                continue
            specs.append(s)
    return specs, rule


def plan(run_dir, specs, n_folds=5) -> tuple[list, list]:
    idx = receipt_index(run_dir)
    tasks: dict = {}
    missing = []
    for s in specs:
        for f in range(n_folds):
            hit = idx.get((s["family"], f, s["target"]))
            if hit is None:
                missing.append({"hypothesis_id": s["hypothesis_id"], "fold": f,
                                "reason": "no receipt of this family resolves to %s in this fold" % s["target"]})
                continue
            key = (s["family"], hit["target_id"], f)
            tasks.setdefault(key, {"family": s["family"], "target_id": hit["target_id"], "fold": f,
                                   "target": s["target"], "substrate": hit["substrate"], "receipt": hit["receipt"],
                                   "rows": hit["rows"], "specs": []})["specs"].append(s)
    return [tasks[k] for k in sorted(tasks)], missing


def cell_path(out, t) -> Path:
    safe = t["target_id"].replace("/", "_")
    return Path(out, "cells", "%s__%s__f%d.json" % (t["family"], safe, t["fold"]))


# --------------------------------------------------------------------- run --

def _aligned(ep_a, ep_b) -> bool:
    return (ep_a.n_rows == ep_b.n_rows
            and np.array_equal(np.asarray(ep_a.context["coords"]), np.asarray(ep_b.context["coords"]))
            and np.array_equal(np.asarray(ep_a.context["patient_codes"]), np.asarray(ep_b.context["patient_codes"]))
            and np.array_equal(np.asarray(ep_a.partitions.proposal), np.asarray(ep_b.partitions.proposal)))


def _own_target_reference(ep):
    """The own_mp target programme's own k=6 neighbour mean (the v0.5 leak guard's reference)."""
    info = ep.metadata.get("own_mp_program") or {}
    if "program_index" not in info:
        return None
    from morpheus_encoder.measurement_bank import _neighbour_mean
    j = int(info["program_index"])
    ps = np.asarray(ep.context["program_scores"], float)[:, [j]]
    return _neighbour_mean(ps, ep.context["neighbors"])[:, 0]


def run_task(t: dict, out: str, restrict_views: bool = True) -> dict:
    t0 = time.time()
    h_sub = t["substrate"]
    h_proto = read_protocol(PROTOCOL_V05 if h_sub == "v05" else PROTOCOL_V04)
    if t["family"] not in h_proto["discovery"]["relationship_families"] or \
            "candidate_views" not in h_proto["discovery"]["relationship_families"][t["family"]]:
        h_proto = read_protocol(PROTOCOL_V04)
    fam_cfg = h_proto["discovery"]["relationship_families"][t["family"]]
    views = hypothesis_views(t["specs"], fam_cfg, t["target_id"]) if restrict_views else None
    if views is not None and h_sub == LADDER_SUBSTRATE:
        views = tuple(sorted(set(views) | set(LADDER_BANK_VIEWS)))
    ep_h = load_episode(h_sub, t["fold"], t["target_id"], bank_views=views)
    design = family_design(ep_h, h_proto, t["family"], t["target_id"], views=views)
    ep_l = ep_h if h_sub == LADDER_SUBSTRATE else load_episode(
        LADDER_SUBSTRATE, t["fold"], t["target_id"], bank_views=LADDER_BANK_VIEWS if restrict_views else None)
    load_s = time.time() - t0
    if not _aligned(ep_h, ep_l):
        raise RuntimeError("hypothesis and ladder episodes do not share rows for %s f%d" % (t["target_id"], t["fold"]))
    y_h, y_l = np.asarray(ep_h.response, float), np.asarray(ep_l.response, float)
    names = list(ep_l.metadata["baseline_columns"])
    extra = {}
    if h_sub != LADDER_SUBSTRATE:
        extra["v04"] = np.asarray(ep_h.context["baseline"], float)
    fd = dc.FoldData(y=y_l, B=np.asarray(ep_l.context["baseline"], float), baseline_names=names,
                     groups=np.asarray(ep_l.context["patient_codes"]), proposal=np.asarray(ep_l.partitions.proposal),
                     replication=np.asarray(ep_l.partitions.replication), extra_competitors=extra)
    lv_info = dc.ladder_levels(names)
    t1 = time.time()
    ladder = dc.FoldLadder(fd, lv_info["levels"])
    shared_s = time.time() - t1
    leak_ref = _own_target_reference(ep_l) if t["target_id"].startswith("own_mp") else None
    p = np.asarray(ep_l.partitions.proposal)
    groups = np.asarray(ep_l.context["patient_codes"])
    results = []
    for s in t["specs"]:
        th = time.time()
        row = t["rows"].get(tuple(s["variable_names"]))
        try:
            V = design_columns(design, s["variable_names"])
        except KeyError as exc:
            results.append({"hypothesis_id": s["hypothesis_id"], "error": str(exc)})
            continue
        if row and row.get("candidate_hash"):
            h, hsrc = row["candidate_hash"], "receipt candidate hash (frozen in this fold)"
        else:
            h, hsrc = dc.content_hash("ladder", t["family"], s["target"], *s["variable_names"]), \
                "content hash (not frozen in this fold)"
        res = dc.FoldLadder.evaluate(ladder, V, feature_form=s["feature_form"], candidate_hash=h,
                                     extra_levels=tuple(extra), v04_check="v04" if extra else None)
        rec = {"hypothesis_id": s["hypothesis_id"], "fold": t["fold"], "target_id": t["target_id"],
               "candidate_hash": h, "seed_source": hsrc, "result": res, "seconds": round(time.time() - th, 2)}
        if row:
            rec["receipt"] = {k: row.get(k) for k in ("replication_gain", "replication_r2_full", "validated")}
            if extra:
                rec["receipt_check"] = {
                    "v04_relation_gain_abs_diff": abs(res["levels"]["v04"]["v04_relation_gain"] - row["replication_gain"]),
                    "v04_full_r2_abs_diff": abs(res["levels"]["v04"]["r2_full"] - row["replication_r2_full"])}
            else:
                rec["receipt_check"] = {"b3_full_r2_abs_diff": abs(res["levels"]["B3"]["r2_full"]
                                                                   - row["replication_r2_full"])}
                d = row.get("decomposition") or {}
                if "gain_total" in d:
                    rec["receipt_check"]["b3_delta_vs_gain_total_abs_diff"] = abs(res["levels"]["B3"]["delta"]
                                                                                   - d["gain_total"])
                if d.get("shape"):
                    rec["receipt_shape"] = d["shape"]
        if leak_ref is not None:
            rec["own_target_neighbour_mean_within_patient_r"] = [
                dc.within_patient_r(V[p, j], leak_ref[p], groups[p]) for j in range(V.shape[1])]
        results.append(rec)
    cell = {"family": t["family"], "target_id": t["target_id"], "target": t["target"], "fold": t["fold"],
            "hypothesis_substrate": h_sub, "ladder_substrate": LADDER_SUBSTRATE, "receipt": t["receipt"],
            "response_equal_across_substrates": bool(np.allclose(y_h, y_l, atol=1e-10)),
            "n_rows": int(len(y_l)), "n_proposal": int(len(fd.proposal)), "n_heldout": int(len(fd.replication)),
            "level_widths": lv_info["level_widths"], "group_counts": lv_info["group_counts"],
            "columns_by_group": {g: [n for n, gg in zip(names, lv_info["groups"]) if gg == g]
                                 for g in ("nuisance", "composition", "programmes", "context")},
            "v04_competitor_width": int(extra["v04"].shape[1]) if extra else None,
            "bank_views": sorted(k for k, v in (ep_l.metadata.get("bank") or {}).items() if isinstance(v, dict)),
            "hypothesis_bank_views": sorted(k for k, v in (ep_h.metadata.get("bank") or {}).items()
                                            if isinstance(v, dict)),
            "competitor_sha256": __import__("hashlib").sha256(np.ascontiguousarray(fd.B).tobytes()).hexdigest(),
            "timing_seconds": {"load_episodes": round(load_s, 1), "shared_competitor_stats": round(shared_s, 1),
                               "hypotheses": round(sum(r.get("seconds", 0) for r in results), 1),
                               "per_hypothesis_mean": round(float(np.mean([r["seconds"] for r in results
                                                                           if "seconds" in r] or [0])), 2)},
            "results": results}
    path = cell_path(out, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cell, default=_plain) + "\n")
    tmp.replace(path)
    return cell


# -------------------------------------------------------------------- pool --

def _f(x, n=3):
    return "-" if x is None else ("%.*f" % (n, x))


def bank_gene_sets() -> dict:
    """The bank's gene sets (Gavish programmes + MSigDB Hallmark), as `hest._bank_sources` passes them."""
    from morpheus.tier2.substrates import hest
    from morpheus_encoder.measurement_bank import load_hallmark_sets
    out = {"gavish:" + str(p["name"]): [str(g) for g in p["genes"]] for p in hest.load_gavish_programs()}
    out.update({"hallmark:" + str(p["name"]): [str(g) for g in p["genes"]] for p in (load_hallmark_sets() or [])})
    return out


def pool(run_dir, out, specs, missing, screen_rule, n_boot=1000) -> dict:
    gene_sets = bank_gene_sets()
    cells = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(os.path.join(out, "cells", "*.json")))]
    per_h: dict = {}
    meta_h: dict = {}
    for c in cells:
        for r in c["results"]:
            if "result" not in r:
                per_h.setdefault(r["hypothesis_id"], {})
                meta_h.setdefault(r["hypothesis_id"], {"errors": []})["errors"].append(r.get("error"))
                continue
            per_h.setdefault(r["hypothesis_id"], {})[int(r["fold"])] = r
    rows = []
    checks = []
    for s in specs:
        hid = s["hypothesis_id"]
        folds = per_h.get(hid) or {}
        entry = dict(s)
        entry["folds_scored"] = sorted(folds)
        entry["folds_missing"] = [m for m in missing if m["hypothesis_id"] == hid]
        ov = dc.target_programme_overlap(s["variable_names"], s["target"], gene_sets)
        entry["target_programme_overlap"] = None if ov is None else ov["target_programme_overlap"]
        if ov is not None:
            entry["target_programme_overlap_detail"] = ov
        if not folds:
            entry["tier"] = "not scored"
            rows.append(entry)
            continue
        res = {f: r["result"] for f, r in folds.items()}
        has_v04 = all("v04" in v["levels"] for v in res.values())
        levels = list(dc.LEVELS) + (["v04"] if has_v04 else [])
        pooled = dc.pool_folds(res, levels=levels, n_boot=n_boot, seed=dc.hypothesis_seed(
            dc.content_hash("pool", hid)))
        tier = dc.assign_tier(pooled)
        B3 = pooled["levels"]["B3"]
        entry.update({
            "pooled": pooled, **tier,
            "shape_at_B3": dc.shape_label(B3["delta_linear_ci95"], B3["delta_beyond_linear_ci95"]),
            "shape_source": "pooled held-out patients at B3 (v0.5 receipt rule)",
            "receipt_shapes": {f: r.get("receipt_shape") for f, r in folds.items() if r.get("receipt_shape")},
            "per_fold": {f: {"delta": {lv: r["result"]["levels"][lv]["delta"] for lv in levels},
                             "delta_linear": {lv: r["result"]["levels"][lv]["delta_linear"] for lv in levels},
                             "competitor_r2": {lv: r["result"]["levels"][lv]["r2_ref"] for lv in levels},
                             "alphas": {lv: [r["result"]["levels"][lv][k] for k in ("alpha_ref", "alpha_lin", "alpha_full")]
                                        for lv in levels},
                             "v04_relation_gain": (r["result"]["levels"]["v04"].get("v04_relation_gain")
                                                   if "v04" in r["result"]["levels"] else None),
                             "receipt": r.get("receipt"), "receipt_check": r.get("receipt_check"),
                             "seed_source": r["seed_source"], "target_id": r["target_id"], "seconds": r["seconds"],
                             "own_target_neighbour_mean_within_patient_r":
                                 r.get("own_target_neighbour_mean_within_patient_r")}
                         for f, r in sorted(folds.items())}})
        rs = [abs(x) for r in folds.values() for x in (r.get("own_target_neighbour_mean_within_patient_r") or [])]
        if rs:
            entry["leak_guard_max_abs_r"] = max(rs)
            entry["leak_guarded_in_v05"] = bool(max(rs) >= 0.9)
        for f, r in folds.items():
            if r.get("receipt_check"):
                checks.append({"hypothesis_id": hid, "fold": f, **r["receipt_check"]})
        rows.append(entry)
    timing = [c["timing_seconds"] for c in cells]
    n_h_fold = sum(len([r for r in c["results"] if "result" in r]) for c in cells)
    cost = {"cells": len(cells), "hypothesis_folds": n_h_fold,
            "episode_load_seconds_total": round(sum(t["load_episodes"] for t in timing), 1),
            "shared_competitor_seconds_total": round(sum(t["shared_competitor_stats"] for t in timing), 1),
            "hypothesis_seconds_total": round(sum(t["hypotheses"] for t in timing), 1),
            "per_hypothesis_fold_seconds_mean": round(sum(t["hypotheses"] for t in timing) / max(n_h_fold, 1), 2)}
    check_summary = {}
    for k in ("v04_relation_gain_abs_diff", "v04_full_r2_abs_diff", "b3_full_r2_abs_diff",
              "b3_delta_vs_gain_total_abs_diff"):
        v = [c[k] for c in checks if k in c]
        if v:
            check_summary[k] = {"n": len(v), "max": float(max(v)), "median": float(np.median(v)),
                                "n_above_1e-6": int(sum(x > 1e-6 for x in v))}
    out_json = {"schema": "morpheus-decomposition-ladder/v1", "run_dir": str(run_dir),
                "method": "dbench/decomposition.py (Delta_k = Q(B_k + H) - Q(B_k), held-out patients, "
                          "registered folds, relation v05 full model, pooled with a patient bootstrap)",
                "levels": {lv: {"groups": list(dc.LEVEL_GROUPS[lv]), "adds": dc.LEVEL_STEP_NAME.get(lv)}
                           for lv in dc.LEVELS},
                "nuisance_choice": NUISANCE_NOTE,
                "screen_selection": screen_rule,
                "development_only_note": "hypotheses whose source is the Level-A linear screen are development "
                                         "results (the screen chose them on the same held-out folds); they are not "
                                         "registered findings",
                "receipt_reproduction": check_summary, "cost": cost,
                "cells": [{k: c[k] for k in ("family", "target_id", "target", "fold", "n_rows", "n_heldout",
                                             "level_widths", "group_counts", "v04_competitor_width",
                                             "response_equal_across_substrates", "timing_seconds", "bank_views")}
                          for c in cells],
                "columns_by_group_example": {c["family"] + "|" + c["target_id"]: c["columns_by_group"] for c in cells
                                             if c["fold"] == 0},
                "hypotheses": rows}
    Path(out, "ladder.json").write_text(json.dumps(out_json, indent=1, default=_plain) + "\n")
    Path(out, "ladder.md").write_text(to_markdown(out_json))
    Path(out, "REPORT.md").write_text(report(out_json, notes=Path(out, "notes.md")))
    return out_json


NUISANCE_NOTE = ("N = log1p sequencing depth only (within-patient standardised like every competitor column). "
                 "The v0.5 bank exposes no spot-level RNA-quality column (no mitochondrial fraction, no detected-gene "
                 "count): its views are raw genes (2,000), PCA, NMF, programmes, SCimilarity, scGPT, H-Optimus-0, "
                 "CellViT structure and neighbour means, none of which is a QC measurement; adding one would change the "
                 "registered comparison model, so it is not added. Patient is handled by the within-patient "
                 "standardisation of the response and every column; batch = sample, nested in patient.")


def _short(s, n=60):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "~"


def to_markdown(o: dict) -> str:
    L = ["# Nested explanation ladder", "",
         "Delta_k = held-out R^2 gain of the hypothesis over comparison model B_k (N = depth; B1 = + cell "
         "composition; B2 = + programme scores; B3 = + neighbour means / PCA = the registered v0.5 comparison "
         "model). Pooled over the registered folds' held-out patients; [CI] = 95% patient bootstrap. "
         "Rows marked (dev) come from the Level-A screen and are development results only.", "",
         "| hypothesis | target | Delta_N | Delta_B1 | Delta_B2 | Delta_B3 | Delta at v0.4 model | competitor R^2 N/B1/B2/B3 "
         "| tier | absorbed by | shape at B3 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    # tier cell carries "[overlap]" when the hypothesis is the target programme, a set sharing >= 20 % of its
    # genes, the neighbour mean of such a set or a raw gene of it (expected by construction, not a finding);
    # "+adj" when the gain is resolved only after adjustment (Delta_B3 CI > 0, Delta_N CI not)
    for h in o["hypotheses"]:
        if "pooled" not in h:
            L.append("| %s | %s | not scored | | | | | | | | |" % (_short(", ".join(h["variable_names"])), h["target"]))
            continue
        P = h["pooled"]["levels"]
        cell = lambda lv: "%s [%s, %s]" % (_f(P[lv]["delta"]), _f(P[lv]["delta_ci95"][0]), _f(P[lv]["delta_ci95"][1]))
        dev = " (dev)" if h["hypothesis_id"].startswith("levelA:") else ""
        stage = "" if dev else " (%s)" % h.get("stage", "")
        L.append("| %s%s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            _short(", ".join(h["variable_names"])), dev, stage, h["target"], cell("N"), cell("B1"), cell("B2"), cell("B3"),
            cell("v04") if "v04" in P else "-",
            "/".join(_f(P[lv]["r2_competitor"]) for lv in dc.LEVELS),
            h["tier"] + (" +adj" if h.get("resolved_only_after_adjustment") else "")
            + (" [overlap]" if h.get("target_programme_overlap") else ""),
            {"B1": "composition", "B2": "programmes", "B3": "spatial context"}.get(h.get("absorbing_level"), "-"),
            h["shape_at_B3"]))
    return "\n".join(L) + "\n"


def report(o: dict, notes: Path | None = None) -> str:
    hs = [h for h in o["hypotheses"] if "pooled" in h]
    frozen = [h for h in hs if not h["hypothesis_id"].startswith("levelA:")]
    dev = [h for h in hs if h["hypothesis_id"].startswith("levelA:")]

    def count(rows):
        c = {t: 0 for t in dc.TIERS}
        for h in rows:
            c[h["tier"]] = c.get(h["tier"], 0) + 1
        return c

    def absorb(rows):
        c = {}
        for h in rows:
            k = h.get("absorbing_level") or "none"
            c[k] = c.get(k, 0) + 1
        return c

    internal = [h for h in frozen if h.get("stage") == "internal"]
    lines = ["# Nested explanation ladder - report", "",
             "Run: `%s`. Method: `dbench/decomposition.py`; numbers: `ladder.json`, table: `ladder.md`." % o["run_dir"],
             "", "## Counts", "",
             "Frozen hypotheses scored: %d of %d." % (len(frozen), len([h for h in o["hypotheses"]
                                                                       if not h["hypothesis_id"].startswith("levelA:")])),
             "", "Tiers (pooled patient-bootstrap CIs): simple association (novelty unchecked) = Delta_N CI > 0 and Delta_B3 CI "
             "includes 0; partly absorbed = Delta_N and Delta_B3 CIs > 0 with Delta_B3/Delta_N < 0.5; irreducible = "
             "the same with ratio >= 0.5; none = otherwise.",
             "", "| tier | frozen | Level-A screen (development) |", "|---|---|---|"]
    cf, cd = count(frozen), count(dev)
    clean = lambda rows: [h for h in rows if not h.get("target_programme_overlap")]
    cfc, cdc = count(clean(frozen)), count(clean(dev))
    lines[-2] = ("| tier | frozen | frozen without overlap-flagged rows | Level-A screen (development) "
                 "| Level-A without overlap-flagged rows |")
    lines[-1] = "|---|---|---|---|---|"
    for t in dc.TIERS:
        lines.append("| %s | %d | %d | %d | %d |" % (t, cf.get(t, 0), cfc.get(t, 0), cd.get(t, 0), cdc.get(t, 0)))
    lines += ["", "Overlap-flagged rows (programme targets whose hypothesis is the target programme, a Gavish/Hallmark "
              "set sharing >= 20 %% of genes with it, the neighbour mean of such a set, or a raw gene of it: expected "
              "by construction, not findings): frozen %d, Level-A %d." % (
                  sum(1 for h in frozen if h.get("target_programme_overlap")),
                  sum(1 for h in dev if h.get("target_programme_overlap")))]
    n_d0 = sum(1 for h in frozen if h["pooled"]["levels"]["N"]["delta_ci95"][0] > 0)
    lines += ["", "Frozen hypotheses with Delta_N CI above 0 (a held-out association beyond depth): %d." % n_d0,
              "Frozen hypotheses resolved only after adjustment (Delta_N CI includes 0, Delta_B3 CI above 0; "
              "not a tier): %d." % sum(1 for h in frozen if h.get("resolved_only_after_adjustment")),
              "Absorbing level among frozen hypotheses (first step, in the order N -> B1 -> B2 -> B3, "
              "with the largest drop): %s." % json.dumps(absorb(frozen)), ""]
    for h in internal:
        P = h["pooled"]["levels"]
        lines += ["## Internal object: %s <- %s" % (h["target"], ", ".join(h["variable_names"])), "",
                  "Delta N/B1/B2/B3 = %s; tier %s; largest drop at %s (%s, %s of Delta_N); shape at B3 %s." % (
                      " / ".join("%s [%s, %s]" % (_f(P[lv]["delta"], 4), _f(P[lv]["delta_ci95"][0], 4),
                                                  _f(P[lv]["delta_ci95"][1], 4)) for lv in dc.LEVELS),
                      h["tier"], h.get("absorbing_level"), h.get("absorbing_explanation"),
                      _f(h.get("absorbed_fraction"), 2), h["shape_at_B3"]), ""]
    rr = o.get("receipt_reproduction") or {}
    lines += ["## Reproduction of registered receipts", "", "```", json.dumps(rr, indent=1), "```", "",
              "## Cost", "", "```", json.dumps(o["cost"], indent=1), "```", "",
              "## Nuisance level", "", o["nuisance_choice"], ""]
    if notes is not None and Path(notes).exists():  # hand-written reading of these numbers
        lines += [Path(notes).read_text(encoding="utf-8")]
    return "\n".join(lines)


# -------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("plan", "run", "pool"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--screen", help="Level-A screen pooled.json (development)")
    ap.add_argument("--screen-top", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--only-target", help="restrict to one receipt target id (smoke)")
    ap.add_argument("--only-fold", type=int)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--all-views", action="store_true", help="build every bank view (slow; default: only the "
                                                             "views the hypotheses and the competitor need)")
    a = ap.parse_args()
    specs, rule = all_specs(a.run_dir, a.screen, a.screen_top)
    tasks, missing = plan(a.run_dir, specs)
    os.makedirs(a.out, exist_ok=True)
    if a.command == "plan":
        print(json.dumps({"specs": len(specs), "tasks": len(tasks), "missing_folds": len(missing),
                          "tasks_list": [(t["family"], t["target_id"], t["fold"], len(t["specs"])) for t in tasks]},
                         indent=1))
        return
    if a.command == "run":
        install_moran_cache(os.path.join(a.out, "_moran_cache"))
        mine = [t for i, t in enumerate(tasks) if i % a.n_shards == a.shard]
        if a.only_target:
            mine = [t for t in mine if t["target_id"] == a.only_target]
        if a.only_fold is not None:
            mine = [t for t in mine if t["fold"] == a.only_fold]
        for t in mine:
            if cell_path(a.out, t).exists():
                continue
            t0 = time.time()
            cell = run_task(t, a.out, restrict_views=not a.all_views)
            print(json.dumps({"cell": str(cell_path(a.out, t)), "hypotheses": len(cell["results"]),
                              "seconds": round(time.time() - t0, 1), "timing": cell["timing_seconds"]}), flush=True)
        return
    o = pool(a.run_dir, a.out, specs, missing, rule, n_boot=a.n_boot)
    print(json.dumps({"hypotheses": len(o["hypotheses"]), "cost": o["cost"],
                      "receipt_reproduction": o["receipt_reproduction"]}, indent=1))


if __name__ == "__main__":
    main()
