"""Translational evidence branch (ideas.md: "Translational discovery must be
its own systematic research branch", "Create a translational evidence record").

Public sources only, no accounts, no licence acceptance:

* Open Targets Platform GraphQL API (no key) -- disease association and
  datatype scores, human-genetics and somatic evidence counts, tractability
  by modality, drugs/clinical candidates with their maximum clinical stage,
  safety liabilities;
* DepMap public CRISPR gene effect (Chronos) + Model annotation, read from
  the persistent store (`S/external/depmap_20261010`, figshare DepMap 24Q4
  Public release);
* a predeclared Cox proportional-hazards survival analysis whose covariate
  specification must be written to disk BEFORE any outcome is loaded.

Every network response goes through a JSON cache (`cache_dir`), so a re-run
is offline and byte-reproducible.  Every record field carries its source and
an explicit ``"not assessed"`` status when no evidence was gathered: absence
of an analysis is never presented as absence of an effect.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_WEB = "https://platform.opentargets.org"
STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
DEPMAP_DIR = os.path.join(STORE, "external", "depmap_20261010")
DEPMAP_GENE_EFFECT = os.path.join(DEPMAP_DIR, "CRISPRGeneEffect.csv")
DEPMAP_MODEL = os.path.join(DEPMAP_DIR, "Model.csv")
DEPMAP_RELEASE = "DepMap 24Q4 Public (figshare 10.25452/figshare.plus.27993248.v1)"
DEPMAP_URL = "https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248"

#: Germline human-genetics datasources (Open Targets ids) -- causal-support class.
OT_GERMLINE_DATASOURCES = ("gwas_credible_sets", "gene_burden", "eva", "genomics_england",
                           "gene2phenotype", "uniprot_literature", "uniprot_variants",
                           "orphanet", "clingen")
#: Somatic cancer datasources -- disease relevance, not human-genetic causality.
OT_SOMATIC_DATASOURCES = ("cancer_gene_census", "intogen", "eva_somatic", "cancer_biomarkers")

RECORD_FIELDS = (
    "independent_biological_replication", "orthogonal_modality_support", "perturbational_support",
    "human_genetic_evidence", "druggability_tractability", "dependency_evidence", "direction_of_effect",
    "safety_liabilities", "existing_compounds", "biomarker_measurability", "outcome_association",
    "incremental_clinical_value", "prospective_evidence",
)
#: Conventional biomarker-development continuum (sequential stages).
BIOMARKER_CONTINUUM = ("discovery", "analytical_validation", "clinical_qualification", "clinical_utility")
NOT_ASSESSED = "not assessed"
LEGACY_LABEL = "legacy source-baseline lead (pre-generator workflow)"


def _today() -> str:
    return _dt.date.today().isoformat()


# --------------------------------------------------------------------------
# JSON cache + HTTP (urllib only; tests mock `urllib.request.urlopen`)
# --------------------------------------------------------------------------

def _cache_file(cache_dir: str | os.PathLike, source: str, key: Any) -> Path:
    blob = json.dumps({"source": source, "key": key}, sort_keys=True, default=str).encode()
    return Path(cache_dir) / source / (hashlib.sha256(blob).hexdigest() + ".json")


def cached_fetch(cache_dir: str | os.PathLike | None, source: str, key: Any,
                 fetch: Callable[[], Any], *, offline: bool = False) -> tuple[Any, dict]:
    """Return ``(payload, provenance)``; a cached payload is never refetched.
    ``offline=True`` with no cached payload raises `LookupError` (the caller
    records the field as not assessed instead of inventing a value)."""
    path = None if cache_dir is None else _cache_file(cache_dir, source, key)
    if path is not None and path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        return stored["payload"], {"cached": True, "retrieved": stored.get("retrieved"), "cache_file": str(path)}
    if offline:
        raise LookupError("offline and no cached %s response for %r" % (source, key))
    payload = fetch()
    retrieved = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": source, "key": key, "retrieved": retrieved, "payload": payload},
                                   sort_keys=True, indent=1), encoding="utf-8")
    return payload, {"cached": False, "retrieved": retrieved, "cache_file": None if path is None else str(path)}


def _post_json(url: str, body: Mapping[str, Any], timeout: float = 60) -> Any:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "morpheus-translational/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _graphql(query: str, variables: Mapping[str, Any], *, cache_dir, offline: bool, timeout: float = 60):
    key = {"url": OPEN_TARGETS_URL, "query": " ".join(query.split()), "variables": dict(variables)}
    payload, prov = cached_fetch(cache_dir, "opentargets", key,
                                 lambda: _post_json(OPEN_TARGETS_URL, {"query": query, "variables": dict(variables)},
                                                    timeout), offline=offline)
    if payload.get("errors"):
        raise RuntimeError("Open Targets GraphQL error: %s" % payload["errors"])
    return payload["data"], prov


# --------------------------------------------------------------------------
# Open Targets
# --------------------------------------------------------------------------

_Q_META = "query { meta { apiVersion { x y z } dataVersion { year month iteration } } }"
_Q_SEARCH = ("query($s:String!, $e:[String!]!){ search(queryString:$s, entityNames:$e, page:{index:0,size:10})"
             "{ hits { id name entity } } }")
_Q_TARGET = """query($t:String!){ target(ensemblId:$t){ id approvedSymbol approvedName biotype
  tractability { label modality value }
  safetyLiabilities { event eventId datasource literature url effects { direction dosing } }
  drugAndClinicalCandidates { count rows { id maxClinicalStage
     drug { id name drugType mechanismsOfAction { rows { mechanismOfAction actionType } } }
     diseases { diseaseFromSource disease { id name } } } } } }"""
_Q_DISEASE = """query($d:String!, $t:String!, $germ:[String!]!, $som:[String!]!){ disease(efoId:$d){ id name
  assoc: associatedTargets(Bs:[$t], enableIndirect:true){ count rows { score
     datatypeScores { id score } datasourceScores { id score } target { id } } }
  germline: evidences(ensemblIds:[$t], enableIndirect:true, datasourceIds:$germ, size:50){ count rows {
     datasourceId score studyId variantRsId literature directionOnTrait disease { id name } } }
  somatic: evidences(ensemblIds:[$t], enableIndirect:true, datasourceIds:$som, size:50){ count rows {
     datasourceId score literature disease { id name } } } } }"""


def _resolve(term: str, entity: str, *, cache_dir, offline) -> tuple[str | None, str | None]:
    data, _ = _graphql(_Q_SEARCH, {"s": term, "e": [entity]}, cache_dir=cache_dir, offline=offline)
    hits = [h for h in (data.get("search") or {}).get("hits", []) if h.get("entity") == entity]
    exact = [h for h in hits if str(h.get("name", "")).lower() == term.lower()]
    best = (exact or hits or [None])[0]
    return (best["id"], best["name"]) if best else (None, None)


_STAGE_RANK = {"APPROVAL": 5, "PHASE_4": 5, "PHASE_3": 4, "PHASE_2_3": 3.5, "PHASE_2": 3, "PHASE_1_2": 2.5,
               "PHASE_1": 2, "EARLY_PHASE_1": 1.5, "PRECLINICAL": 1, "UNKNOWN": 0}


def open_targets_evidence(gene_symbol: str, disease: str = "renal cell carcinoma", *,
                          disease_id: str | None = None, cache_dir: str | os.PathLike | None = None,
                          offline: bool = False, timeout: float = 60) -> dict:
    """Open Targets evidence for one gene against one disease (descendant
    diseases included for association and evidence counts)."""
    out: dict[str, Any] = {"source": "Open Targets Platform GraphQL API", "api_url": OPEN_TARGETS_URL,
                           "gene_symbol": gene_symbol, "disease_query": disease, "status": NOT_ASSESSED}
    try:
        meta, prov = _graphql(_Q_META, {}, cache_dir=cache_dir, offline=offline)
        out["retrieved"] = prov["retrieved"]
        api, data = meta["meta"]["apiVersion"], meta["meta"]["dataVersion"]
        out["api_version"] = "%s.%s.%s" % (api["x"], api["y"], api["z"])
        out["data_version"] = "%s.%s" % (data["year"], data["month"])
        ensg, symbol = _resolve(gene_symbol, "target", cache_dir=cache_dir, offline=offline)
        if ensg is None or str(symbol).upper() != gene_symbol.upper():
            out["reason"] = "gene symbol did not resolve to an exact Open Targets target"
            return out
        if disease_id is None:
            disease_id, disease_name = _resolve(disease, "disease", cache_dir=cache_dir, offline=offline)
        else:
            disease_name = None
        target_data, _ = _graphql(_Q_TARGET, {"t": ensg}, cache_dir=cache_dir, offline=offline)
        disease_data, _ = _graphql(_Q_DISEASE, {"d": disease_id, "t": ensg, "germ": list(OT_GERMLINE_DATASOURCES),
                                                "som": list(OT_SOMATIC_DATASOURCES)},
                                   cache_dir=cache_dir, offline=offline)
    except LookupError as exc:
        out["reason"] = str(exc)
        return out
    target = target_data.get("target") or {}
    dis = disease_data.get("disease") or {}
    rows = [r for r in ((dis.get("assoc") or {}).get("rows") or []) if (r.get("target") or {}).get("id") == ensg]
    association = None
    if rows:
        association = {"overall_score": rows[0]["score"],
                       "datatype_scores": {d["id"]: d["score"] for d in rows[0]["datatypeScores"]},
                       "datasource_scores": {d["id"]: d["score"] for d in rows[0]["datasourceScores"]}}
    tract: dict[str, list[str]] = {}
    for t in target.get("tractability") or []:
        if t.get("value"):
            tract.setdefault(t["modality"], []).append(t["label"])
    candidates = []
    for row in ((target.get("drugAndClinicalCandidates") or {}).get("rows") or []):
        drug = row.get("drug") or {}
        moa = ((drug.get("mechanismsOfAction") or {}).get("rows") or [])
        diseases = [(d.get("disease") or {}).get("name") or d.get("diseaseFromSource") for d in row.get("diseases") or []]
        candidates.append({"drug": drug.get("name"), "drug_id": drug.get("id"), "drug_type": drug.get("drugType"),
                           "max_clinical_stage": row.get("maxClinicalStage"),
                           "mechanisms": sorted({m.get("mechanismOfAction") for m in moa if m.get("mechanismOfAction")}),
                           "action_types": sorted({m.get("actionType") for m in moa if m.get("actionType")}),
                           "diseases": sorted({d for d in diseases if d})})
    candidates.sort(key=lambda c: -_STAGE_RANK.get(str(c["max_clinical_stage"]), 0))
    renal = [c for c in candidates if any("renal" in d.lower() or "kidney" in d.lower() for d in c["diseases"])]
    germ, som = dis.get("germline") or {}, dis.get("somatic") or {}
    out.update({
        "status": "assessed", "ensembl_id": ensg, "approved_name": target.get("approvedName"),
        "disease": {"id": disease_id, "name": dis.get("name") or disease_name},
        "association": association,
        "association_note": None if association else "no Open Targets association between this target and the disease (or its descendants)",
        "human_genetics_germline": {"count": int(germ.get("count") or 0),
                                    "datasources": sorted({r["datasourceId"] for r in germ.get("rows") or []}),
                                    "rows": (germ.get("rows") or [])[:10]},
        "somatic_evidence": {"count": int(som.get("count") or 0),
                             "datasources": sorted({r["datasourceId"] for r in som.get("rows") or []})},
        "tractability": tract,
        "clinical_candidates": {"count": int(((target.get("drugAndClinicalCandidates") or {}).get("count")) or 0),
                                "rows": candidates[:25], "renal_indication_rows": renal},
        "safety_liabilities": [{"event": s.get("event"), "datasource": s.get("datasource"), "url": s.get("url"),
                                "effects": s.get("effects") or []} for s in target.get("safetyLiabilities") or []],
        "urls": {"target": "%s/target/%s" % (OPEN_TARGETS_WEB, ensg),
                 "evidence": "%s/evidence/%s/%s" % (OPEN_TARGETS_WEB, ensg, disease_id)},
    })
    return out


# --------------------------------------------------------------------------
# DepMap
# --------------------------------------------------------------------------

def _depmap_column(header: Sequence[str], gene_symbol: str) -> int | None:
    for i, name in enumerate(header):
        if name.split(" (")[0] == gene_symbol:
            return i
    return None


def depmap_dependency(gene_symbol: str, lineage: str = "kidney", *, gene_effect_path: str = DEPMAP_GENE_EFFECT,
                      model_path: str = DEPMAP_MODEL, subtype_contains: str | None = "Clear Cell",
                      dependent_threshold: float = -0.5, release: str = DEPMAP_RELEASE,
                      source_url: str = DEPMAP_URL, cache_dir: str | os.PathLike | None = None) -> dict:
    """CRISPR (Chronos) gene effect of `gene_symbol` in the `lineage` cell
    lines versus all others. A gene effect of 0 is no effect and -1 is the
    median of common essential genes; `dependent_threshold` marks a line as
    dependent. Results are cached by (gene, lineage, file sizes)."""
    import pandas as pd
    out: dict[str, Any] = {"source": "DepMap CRISPR gene effect (Chronos)", "release": release, "url": source_url,
                           "gene_symbol": gene_symbol, "lineage": lineage, "status": NOT_ASSESSED}
    if not (os.path.exists(gene_effect_path) and os.path.exists(model_path)):
        out["reason"] = "DepMap gene-effect or model file absent"
        return out
    key = {"gene": gene_symbol, "lineage": lineage, "subtype": subtype_contains, "thr": dependent_threshold,
           "effect_size": os.path.getsize(gene_effect_path), "model_size": os.path.getsize(model_path)}

    def compute() -> dict:
        with open(gene_effect_path, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split(",")
        col = _depmap_column(header, gene_symbol)
        if col is None:
            return {"status": NOT_ASSESSED, "reason": "gene not screened in this release"}
        eff = pd.read_csv(gene_effect_path, usecols=[0, col])
        eff.columns = ["ModelID", "effect"]
        model = pd.read_csv(model_path, usecols=lambda c: c in ("ModelID", "OncotreeLineage", "OncotreeSubtype",
                                                                 "OncotreePrimaryDisease", "CellLineName"))
        df = eff.merge(model, on="ModelID", how="left").dropna(subset=["effect"])
        in_lin = df["OncotreeLineage"].astype(str).str.lower() == lineage.lower()
        from scipy.stats import mannwhitneyu
        lin, rest = df.loc[in_lin, "effect"].to_numpy(), df.loc[~in_lin, "effect"].to_numpy()
        res: dict[str, Any] = {"status": "assessed", "n_lines_total": int(len(df)), "n_lines_lineage": int(len(lin)),
                               "median_effect_all": float(np.median(df["effect"])),
                               "fraction_dependent_all": float(np.mean(df["effect"] < dependent_threshold)),
                               "dependent_threshold": dependent_threshold}
        if len(lin):
            res.update({"median_effect_lineage": float(np.median(lin)), "mean_effect_lineage": float(np.mean(lin)),
                        "fraction_dependent_lineage": float(np.mean(lin < dependent_threshold)),
                        "lineage_minus_rest_median": float(np.median(lin) - np.median(rest)),
                        "mannwhitney_p_lineage_vs_rest": (float(mannwhitneyu(lin, rest).pvalue)
                                                          if len(lin) >= 3 and len(rest) >= 3 else None)})
            by_lineage = df.groupby("OncotreeLineage")["effect"].median().sort_values()
            res["lineage_rank_by_median_effect"] = {"rank": int(list(by_lineage.index).index(
                df.loc[in_lin, "OncotreeLineage"].iloc[0]) + 1), "n_lineages": int(len(by_lineage))}
        if subtype_contains:
            sub = df[in_lin & df["OncotreeSubtype"].astype(str).str.contains(subtype_contains, case=False)]
            res["subtype"] = {"pattern": subtype_contains, "n_lines": int(len(sub)),
                              "median_effect": float(np.median(sub["effect"])) if len(sub) else None,
                              "lines": [{"name": r.CellLineName, "effect": float(r.effect)} for r in sub.itertuples()]}
        all_frac = res["fraction_dependent_all"]
        res["pattern"] = ("broadly essential (dependent in >=90% of all lines; not lineage-selective)" if all_frac >= 0.9
                          else "selective dependency in lineage" if len(lin) and res["fraction_dependent_lineage"] >= 0.5
                          and res["lineage_minus_rest_median"] < -0.2
                          else "no lineage dependency" if len(lin) and res["fraction_dependent_lineage"] < 0.2
                          else "partial / mixed dependency")
        return res

    try:
        payload, prov = cached_fetch(cache_dir, "depmap", key, compute)
    except Exception as exc:  # file parsing problems are reported, not hidden
        out["reason"] = "DepMap read failed: %s" % exc
        return out
    out.update(payload)
    out["retrieved"] = prov["retrieved"]
    return out


# --------------------------------------------------------------------------
# Predeclared survival (spec written BEFORE outcomes are loaded)
# --------------------------------------------------------------------------

def write_survival_spec(path: str | os.PathLike, spec: Mapping[str, Any]) -> dict:
    """Freeze the survival analysis specification to disk; returns the record
    with its sha256. Refuses to overwrite a different existing spec."""
    path = Path(path)
    body = json.dumps(dict(spec), sort_keys=True, indent=1)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != digest:
            raise ValueError("a different survival spec is already frozen at %s" % path)
        return existing
    record = {"spec": json.loads(body), "sha256": digest,
              "written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, indent=1), encoding="utf-8")
    return record


def read_survival_spec(path: str | os.PathLike) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("survival covariate specification must be written before fitting: %s" % path)
    record = json.loads(path.read_text(encoding="utf-8"))
    body = json.dumps(record["spec"], sort_keys=True, indent=1)
    if hashlib.sha256(body.encode()).hexdigest() != record["sha256"]:
        raise ValueError("survival spec hash mismatch (spec edited after freezing)")
    return record


def harrell_c(time: Sequence[float], event: Sequence[float], risk: Sequence[float]) -> float:
    """Harrell's concordance: higher risk should mean shorter survival."""
    t, e, r = (np.asarray(v, dtype=float) for v in (time, event, risk))
    num = den = 0.0
    for i in np.flatnonzero(e > 0):
        later = t > t[i]
        den += later.sum()
        num += (r[i] > r[later]).sum() + 0.5 * (r[i] == r[later]).sum()
    return float(num / den) if den else float("nan")


def _cox(time, event, design):
    from statsmodels.duration.hazard_regression import PHReg
    model = PHReg(np.asarray(time, float), np.asarray(design, float), status=np.asarray(event, float), ties="efron")
    return model.fit(disp=False)


def fit_cox_tertiles(time, event, covariates, expression, *, n_boot: int = 200, seed: int = 0) -> dict:
    """Cox PH: covariates alone vs covariates + expression tertile dummies
    (low = reference). Returns hazard ratios (95% CI), likelihood-ratio test
    for the tertile terms, Harrell C for both models and the incremental C
    with a percentile bootstrap interval (both models refit per resample).
    In-sample C is optimistic; the increment is the comparable quantity."""
    from scipy.stats import chi2, norm
    time, event = np.asarray(time, float), np.asarray(event, float)
    C = np.asarray(covariates, float).reshape(len(time), -1)
    expr = np.asarray(expression, float)
    cuts = np.quantile(expr, [1 / 3, 2 / 3])
    tert = np.searchsorted(cuts, expr, side="right")
    T = np.column_stack(((tert == 1).astype(float), (tert == 2).astype(float)))
    full_design = np.column_stack((C, T))
    base, full = _cox(time, event, C), _cox(time, event, full_design)
    params, bse = np.asarray(full.params), np.asarray(full.bse)
    hr = {}
    for name, j in (("mid_vs_low", C.shape[1]), ("high_vs_low", C.shape[1] + 1)):
        hr[name] = {"hr": float(np.exp(params[j])), "ci95": [float(np.exp(params[j] - 1.96 * bse[j])),
                                                            float(np.exp(params[j] + 1.96 * bse[j]))],
                    "p_wald": float(2 * norm.sf(abs(params[j] / bse[j])))}
    lr = 2 * (full.llf - base.llf)
    c_base = harrell_c(time, event, C @ np.asarray(base.params))
    c_full = harrell_c(time, event, full_design @ params)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(time), len(time))
        if event[idx].sum() < 3:
            continue
        try:
            b, f = _cox(time[idx], event[idx], C[idx]), _cox(time[idx], event[idx], full_design[idx])
            deltas.append(harrell_c(time[idx], event[idx], full_design[idx] @ np.asarray(f.params))
                          - harrell_c(time[idx], event[idx], C[idx] @ np.asarray(b.params)))
        except Exception:
            continue
    return {"n": int(len(time)), "n_events": int(event.sum()), "tertile_cuts": [float(x) for x in cuts],
            "tertile_sizes": [int((tert == k).sum()) for k in range(3)], "hazard_ratios": hr,
            "covariate_hazard_ratios": [float(np.exp(v)) for v in np.asarray(full.params)[:C.shape[1]]],
            "lr_test_tertiles": {"chi2": float(lr), "df": 2, "p": float(chi2.sf(lr, 2))},
            "c_index_covariates": c_base, "c_index_covariates_plus_tertiles": c_full,
            "delta_c_index": c_full - c_base, "delta_c_index_ci95_bootstrap": (
                [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))] if deltas else None),
            "n_boot_effective": len(deltas),
            "events_per_parameter": float(event.sum() / full_design.shape[1])}


def run_predeclared_survival(spec_path: str | os.PathLike, load_cohort: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                             gene_symbols: Sequence[str]) -> dict:
    """Read the frozen spec FIRST, then call `load_cohort(spec_for_cohort)`
    for each cohort in it (outcomes are loaded only after the spec exists),
    fit one Cox model per (gene, cohort) and BH-adjust the tertile LR tests
    across the whole predeclared family."""
    record = read_survival_spec(spec_path)
    spec = record["spec"]
    results = []
    for cohort_spec in spec["cohorts"]:
        data = load_cohort(cohort_spec)
        for gene in gene_symbols:
            expr = data["expression"].get(gene)
            entry = {"cohort": cohort_spec["name"], "gene": gene, "covariates": data["covariate_names"],
                     "covariates_unavailable": data.get("covariates_unavailable", [])}
            if expr is None:
                entry["status"] = NOT_ASSESSED
                entry["reason"] = "gene absent from cohort expression"
            else:
                keep = np.isfinite(expr) & np.all(np.isfinite(data["covariates"]), axis=1) \
                    & np.isfinite(data["time"]) & np.isfinite(data["event"]) & (data["time"] > 0)
                if keep.sum() < 20 or np.nansum(np.asarray(data["event"])[keep]) < 5:
                    entry.update({"status": NOT_ASSESSED, "reason": "too few patients with a finite feature "
                                  "(%d) or events in this cohort" % int(keep.sum())})
                    results.append(entry)
                    continue
                entry.update({"status": "assessed", **fit_cox_tertiles(
                    data["time"][keep], data["event"][keep], data["covariates"][keep], expr[keep],
                    n_boot=int(spec.get("n_boot", 200)), seed=int(spec.get("seed", 0)))})
            results.append(entry)
    ps = [(i, r["lr_test_tertiles"]["p"]) for i, r in enumerate(results) if r.get("status") == "assessed"]
    if ps:
        order = sorted(ps, key=lambda x: x[1])
        m, running = len(order), 1.0
        qs = {}
        for rank in range(m, 0, -1):
            i, p = order[rank - 1]
            running = min(running, p * m / rank)
            qs[i] = running
        for i, q in qs.items():
            results[i]["lr_test_tertiles"]["bh_q_family"] = float(q)
    return {"spec_sha256": record["sha256"], "spec_written_utc": record["written_utc"], "results": results}


# --------------------------------------------------------------------------
# The translational evidence record
# --------------------------------------------------------------------------

def _field(status: str = NOT_ASSESSED, summary: str = "", value: Any = None,
           sources: Sequence[Mapping[str, Any]] = ()) -> dict:
    return {"status": status, "summary": summary or ("no evidence gathered for this field" if status == NOT_ASSESSED
                                                     else ""), "value": value, "sources": list(sources)}


def empty_record_fields() -> dict:
    return {name: _field() for name in RECORD_FIELDS}


def evidence_level_flags(record: Mapping[str, Any]) -> dict:
    """Levels E/F from evidence, never from the decomposition statistics.
    E needs perturbational support OF THE RELATION (a gene's own fitness
    dependency is not that). F needs an assessed outcome association that
    agrees in direction across >= 2 independent cohorts, survives the
    family multiplicity correction in each, and adds concordance over the
    clinical covariates in each."""
    fields = record.get("fields", record)
    pert = fields.get("perturbational_support", {})
    e_ok = bool(pert.get("status") == "assessed" and (pert.get("value") or {}).get("supports_relation"))
    out_assoc = fields.get("outcome_association", {})
    f_ok = bool(out_assoc.get("status") == "assessed" and (out_assoc.get("value") or {}).get("replicated_prognostic"))
    return {"E": {"supported": e_ok, "basis": pert.get("summary", "")},
            "F": {"supported": f_ok, "basis": out_assoc.get("summary", "")}}


def biomarker_continuum_position(fields: Mapping[str, Any]) -> dict:
    """Sequential continuum: a stage counts only if every earlier one is met.
    Analytical validation needs an assay with measured reproducibility, which
    no field here supplies unless `biomarker_measurability.value` says so."""
    meas = (fields.get("biomarker_measurability") or {}).get("value") or {}
    outcome = (fields.get("outcome_association") or {}).get("value") or {}
    incr = (fields.get("incremental_clinical_value") or {}).get("value") or {}
    position, rationale = "discovery", ["held-out association in spatial data (internal validation only)"]
    if meas.get("analytically_validated_assay"):
        position = "analytical_validation"
        rationale.append("assay reproducibility measured")
        if outcome.get("replicated_prognostic") and incr.get("adds_over_clinical"):
            position = "clinical_qualification"
            rationale.append("predeclared outcome association replicated with incremental value")
            if incr.get("decision_impact_shown"):
                position = "clinical_utility"
    else:
        rationale.append("no analytically validated assay (RNA/protein measurements are research-grade)")
        if outcome.get("replicated_prognostic"):
            rationale.append("prognostic association replicated in public cohorts -- exploratory evidence toward "
                             "clinical qualification, which cannot be claimed before analytical validation")
    return {"position": position, "stages": list(BIOMARKER_CONTINUUM), "rationale": rationale}


def translational_evidence_record(candidate_id: str, genes: Sequence[str], target: str,
                                  ledger_rec: Mapping[str, Any] | None, *, fields: Mapping[str, Mapping] | None = None,
                                  explanation: Mapping[str, Any] | None = None, novelty: Mapping[str, Any] | None = None,
                                  confound: Mapping[str, Any] | None = None, label: str = LEGACY_LABEL,
                                  notes: Sequence[str] = ()) -> dict:
    """Assemble one record. `fields` supplies assessed entries (each made by
    `_field`-shaped dicts); every other field stays explicitly not assessed.
    The language guard is applied to every summary string."""
    record_fields = empty_record_fields()
    for name, value in (fields or {}).items():
        if name not in RECORD_FIELDS:
            raise KeyError("unknown translational record field %r" % name)
        merged = _field()
        merged.update(dict(value))
        record_fields[name] = merged
    violations = {}
    # Same loader the object cards use: works even where the morpheus.protocol
    # package __init__ cannot import its optional dependencies.
    from dbench.object_cards import _get_language_module
    _language_check = _get_language_module().check
    if _language_check is not None:
        for name, value in record_fields.items():
            found = _language_check(str(value.get("summary", "")), field=name)
            if found:
                violations[name] = [{"category": v.category, "word": v.word, "context": v.context} for v in found]
    ledger_rec = dict(ledger_rec or {})
    record = {
        "schema": "morpheus-translational-evidence-record/v1",
        "candidate_id": candidate_id, "genes": list(genes), "target": target, "label": label,
        "ledger": {"stage": ledger_rec.get("stage"), "object_hash": ledger_rec.get("object_hash"),
                   "evidence": ledger_rec.get("evidence")},
        "fields": record_fields,
        "treatment_response": {"status": "blocked",
                               "reason": "no public treatment-response raw data (JAVELIN raw data unavailable); "
                                         "no treatment-response claim is made"},
        "explanation": dict(explanation or {}), "literature_novelty": dict(novelty or {}),
        "confound_retest": dict(confound or {}), "language_violations": violations, "notes": list(notes),
        "assembled": _today(),
    }
    record["evidence_level_flags"] = evidence_level_flags(record)
    record["biomarker_continuum"] = biomarker_continuum_position(record_fields)
    return record


# --------------------------------------------------------------------------
# Bulk proxies and patient-level tests for coordinate / programme hypotheses
# --------------------------------------------------------------------------

def top_signed_genes(names: Sequence[str], weights: Sequence[float], n: int = 25) -> tuple[list, list]:
    """The `n` most positive and `n` most negative named weights (ties by name),
    as [(gene, weight), ...] lists; non-finite weights are skipped."""
    pairs = [(str(g), float(w)) for g, w in zip(names, weights) if np.isfinite(w)]
    up = sorted((p for p in pairs if p[1] > 0), key=lambda p: (-p[1], p[0]))[:n]
    down = sorted((p for p in pairs if p[1] < 0), key=lambda p: (p[1], p[0]))[:n]
    return up, down


def signed_proxy_score(values, symbols: Sequence[str], up: Sequence[str], down: Sequence[str], *,
                       fit_rows=None, min_genes: int = 3) -> tuple[np.ndarray, dict]:
    """Platform-portable proxy of a decoded axis: mean z-score of its up genes
    minus mean z-score of its down genes. z statistics are fitted on
    `fit_rows` (default all rows); a side with fewer than `min_genes` usable
    genes contributes nothing, and a row with neither side usable is NaN."""
    values = np.asarray(values, dtype=float)
    fit = np.arange(values.shape[0]) if fit_rows is None else np.asarray(fit_rows)
    index = {s: j for j, s in enumerate(symbols)}
    mu, sd = np.nanmean(values[fit], axis=0), np.nanstd(values[fit], axis=0)

    def side(genes):
        cols = [index[g] for g in genes if g in index and np.isfinite(sd[index[g]]) and sd[index[g]] > 1e-12]
        if len(cols) < min_genes:
            return None, []
        z = (values[:, cols] - mu[cols]) / sd[cols]
        with np.errstate(invalid="ignore"):
            return np.nanmean(z, axis=1), [symbols[j] for j in cols]

    s_up, used_up = side(up)
    s_dn, used_dn = side(down)
    if s_up is None and s_dn is None:
        return np.full(values.shape[0], np.nan), {"up_used": [], "down_used": []}
    score = (s_up if s_up is not None else 0.0) - (s_dn if s_dn is not None else 0.0)
    return np.asarray(score, dtype=float), {"up_used": used_up, "down_used": used_dn}


def _ranks(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a)


def permutation_spearman(x, y, *, n_perm: int = 9999, seed: int = 0, expected_sign: int | None = None) -> dict:
    """Spearman rho with a label-permutation p value (two-sided, and one-sided
    in `expected_sign` when given). Rows with a non-finite value are dropped."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < 8:
        return {"n": n, "rho": None, "p_two_sided": None, "p_one_sided_expected": None}
    rx, ry = _ranks(x[keep]), _ranks(y[keep])
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm))
    for start in range(0, int(n_perm), 1000):
        k = min(1000, int(n_perm) - start)
        perm = np.argsort(rng.random((k, n)), axis=1)
        null[start:start + k] = (ry[perm] * rx).mean(axis=1)
    out = {"n": n, "rho": rho, "n_perm": int(n_perm),
           "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n_perm + 1))}
    if expected_sign:
        s = 1.0 if expected_sign > 0 else -1.0
        out["expected_sign"] = int(s)
        out["sign_agrees"] = bool(np.sign(rho) == s)
        out["p_one_sided_expected"] = float((1 + np.sum(s * null >= s * rho - 1e-12)) / (n_perm + 1))
    else:
        out["p_one_sided_expected"] = None
    return out


def partial_permutation_spearman(x, y, covariates, *, n_perm: int = 9999, seed: int = 0,
                                 expected_sign: int | None = None) -> dict:
    """Spearman association of x and y after removing the covariates: ranks
    of x and y are each residualised on the covariates' ranks (least squares
    with intercept), then `permutation_spearman` is applied to the residuals.
    Rows with any non-finite value are dropped."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    Z = np.asarray(covariates, dtype=float).reshape(len(x), -1)
    keep = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if keep.sum() < Z.shape[1] + 8:
        return {"n": int(keep.sum()), "rho": None, "p_two_sided": None, "p_one_sided_expected": None}
    D = np.column_stack([np.ones(int(keep.sum()))] + [_ranks(Z[keep, j]) for j in range(Z.shape[1])])

    def resid(v):
        rv = _ranks(v[keep])
        beta, *_ = np.linalg.lstsq(D, rv, rcond=None)
        return rv - D @ beta

    out = permutation_spearman(resid(x), resid(y), n_perm=n_perm, seed=seed, expected_sign=expected_sign)
    out["n_covariates"] = int(Z.shape[1])
    return out


# --------------------------------------------------------------------------
# Patient-level linear-irreplicability and assay distillation (biomarker table)
# --------------------------------------------------------------------------

def _oof_predictions(X, y, fold_ids, *, flexible: bool = False, n_rbf: int = 24, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions: inner-CV ridge (the discovery core's own fits)
    on X, or on [X, random-Fourier map of X] when `flexible` -- the flexible
    model contains the linear one, so it can only add non-linear shape."""
    from morpheus_encoder.general_discovery import _cv_alpha, _RBFMap, _RidgeFit
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float)
    folds = np.asarray(fold_ids)
    grid = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr_, te = folds != f, folds == f
        Dtr, Dte = X[tr_], X[te]
        if flexible:
            m = _RBFMap(Dtr, n_features=n_rbf * X.shape[1], seed=seed)
            Dtr, Dte = np.column_stack((Dtr, m.apply(Dtr))), np.column_stack((Dte, m.apply(Dte)))
        alpha = _cv_alpha(Dtr, y[tr_], None, grid, seed)
        pred[te] = _RidgeFit(Dtr, y[tr_], alpha).predict(Dte)
    return pred


def _r2_of(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def _complete(*arrays):
    keep = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        a = np.asarray(a, dtype=float).reshape(len(keep), -1)
        keep &= np.all(np.isfinite(a), axis=1)
    return keep


def linear_irreplicability(x_relation, X_raw, y, fold_ids, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Held-out R^2 of the relation (flexible model of the relation's own
    variables) minus that of the best linear model on the raw measurements
    (inner-CV ridge), same patient folds; patient-bootstrap CI of the gap on
    the out-of-fold predictions. A gap <= 0 means a linear model on the raw
    measurements already does as well."""
    xr = np.asarray(x_relation, dtype=float).reshape(len(y), -1)
    Xl = np.asarray(X_raw, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float)
    keep = _complete(xr, Xl, y)
    folds = np.asarray(fold_ids)[keep]
    yk = y[keep]
    p_rel = _oof_predictions(xr[keep], yk, folds, flexible=True, seed=seed)
    p_lin = _oof_predictions(Xl[keep], yk, folds, flexible=False, seed=seed)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(int(n_boot)):
        i = rng.integers(0, len(yk), len(yk))
        gaps.append(_r2_of(yk[i], p_rel[i]) - _r2_of(yk[i], p_lin[i]))
    r_rel, r_lin = _r2_of(yk, p_rel), _r2_of(yk, p_lin)
    return {"n": int(keep.sum()), "n_raw_features": int(Xl.shape[1]), "r2_relation": r_rel, "r2_best_linear": r_lin,
            "gap": r_rel - r_lin, "gap_ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
            "linearly_replicable": bool(np.quantile(gaps, 0.025) <= 0)}


def distilled_assay(X_cheap, y_expensive, fold_ids, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """How well a cheap measurement stands in for an expensive one: out-of-fold
    inner-CV ridge from X_cheap to y_expensive; held-out R^2 and Spearman with
    patient-bootstrap CIs."""
    from scipy.stats import spearmanr
    X = np.asarray(X_cheap, dtype=float).reshape(len(y_expensive), -1)
    y = np.asarray(y_expensive, dtype=float)
    keep = _complete(X, y)
    yk = y[keep]
    p = _oof_predictions(X[keep], yk, np.asarray(fold_ids)[keep], seed=seed)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        i = rng.integers(0, len(yk), len(yk))
        boots.append((_r2_of(yk[i], p[i]), spearmanr(yk[i], p[i]).statistic))
    b = np.asarray(boots)
    return {"n": int(keep.sum()), "n_cheap_features": int(X.shape[1]), "heldout_r2": _r2_of(yk, p),
            "heldout_r2_ci95": [float(np.quantile(b[:, 0], 0.025)), float(np.quantile(b[:, 0], 0.975))],
            "heldout_spearman": float(spearmanr(yk, p).statistic),
            "heldout_spearman_ci95": [float(np.quantile(b[:, 1], 0.025)), float(np.quantile(b[:, 1], 0.975))]}


# --------------------------------------------------------------------------
# Held-out comparisons pooled over folds (patients / donors as the unit)
# --------------------------------------------------------------------------

def ridge_fit_predict(X_train, y_train, X_test, *, groups_train=None, seed: int = 0,
                      max_cv_rows: int = 20000) -> np.ndarray:
    """Inner-CV ridge (grouped when groups are given; the discovery core's own
    alpha grid and fitter). For very many rows the alpha search uses a fixed
    random subset of at most `max_cv_rows` training rows; the final fit uses all."""
    from morpheus_encoder.general_discovery import VALIDATION_ALPHA_GRID, _cv_alpha, _RidgeFit
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    y_train = np.asarray(y_train, float)
    rows = np.arange(len(y_train))
    if len(rows) > max_cv_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_cv_rows, replace=False))
    g = None if groups_train is None else np.asarray(groups_train)[rows]
    alpha = _cv_alpha(X_train[rows], y_train[rows], g, VALIDATION_ALPHA_GRID, seed)
    return _RidgeFit(X_train, y_train, alpha).predict(X_test)


def _group_resample(groups, rng):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {u: np.flatnonzero(groups == u) for u in uniq}
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([members[u] for u in pick])


def group_bootstrap_gap(y, pred_a, pred_b, groups, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Held-out R^2 of predictions a minus b on the same rows, with a bootstrap
    CI that resamples whole groups (patients / donors). With fewer than 3
    groups the rows are resampled instead and the result says so."""
    y, a, b = (np.asarray(v, float) for v in (y, pred_a, pred_b))
    groups = np.asarray(groups)
    keep = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b, groups = y[keep], a[keep], b[keep], groups[keep]
    rng = np.random.default_rng(seed)
    by_group = len(np.unique(groups)) >= 3
    gaps = []
    for _ in range(int(n_boot)):
        i = _group_resample(groups, rng) if by_group else rng.integers(0, len(y), len(y))
        gaps.append(_r2_of(y[i], a[i]) - _r2_of(y[i], b[i]))
    ra, rb = _r2_of(y, a), _r2_of(y, b)
    return {"n": int(len(y)), "n_groups": int(len(np.unique(groups))), "r2_a": ra, "r2_b": rb, "gap": ra - rb,
            "gap_ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
            "bootstrap_unit": "group" if by_group else "row (fewer than 3 groups)"}


def heldout_summary(y, pred, groups, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Held-out R^2 and Spearman of one model's pooled out-of-fold predictions,
    group-bootstrap CIs."""
    from scipy.stats import spearmanr
    y, p = np.asarray(y, float), np.asarray(pred, float)
    groups = np.asarray(groups)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p, groups = y[keep], p[keep], groups[keep]
    rng = np.random.default_rng(seed)
    by_group = len(np.unique(groups)) >= 3
    boots = []
    for _ in range(int(n_boot)):
        i = _group_resample(groups, rng) if by_group else rng.integers(0, len(y), len(y))
        boots.append((_r2_of(y[i], p[i]), spearmanr(y[i], p[i]).statistic))
    b = np.asarray(boots)
    return {"n": int(len(y)), "n_groups": int(len(np.unique(groups))), "heldout_r2": _r2_of(y, p),
            "heldout_r2_ci95": [float(np.nanquantile(b[:, 0], 0.025)), float(np.nanquantile(b[:, 0], 0.975))],
            "heldout_spearman": float(spearmanr(y, p).statistic),
            "heldout_spearman_ci95": [float(np.nanquantile(b[:, 1], 0.025)), float(np.nanquantile(b[:, 1], 0.975))],
            "bootstrap_unit": "group" if by_group else "row (fewer than 3 groups)"}


# --------------------------------------------------------------------------
# External CITE-seq cohort (bone marrow): Stuart et al. 2019, GEO GSE128639
# --------------------------------------------------------------------------

STUART2019_DIR = os.path.join(STORE, "external", "stuart2019_bmcite_20261014")
STUART2019_FILES = {"rna": "GSM3681518_MNC_RNA_counts.tsv.gz", "adt": "GSM3681519_MNC_ADT_counts.tsv.gz",
                    "hto": "GSM3681520_MNC_HTO_counts.tsv.gz"}
#: Antibody names that differ between panels for the same target.
ADT_ALIASES = {"CD8A": "CD8", "ICOS": "CD278", "CD197": "CCR7", "CD11A/CD18": "CD11A", "INTEGRIN-7": "INTEGRINB7",
               "TCR-V-7.2": "TCRVA7.2", "TCR-V-2": "TCRVD2"}


def normalise_adt_name(name: str) -> str:
    """Panel-independent antibody target name: upper case, clone suffix
    (``-1``/``-2``) removed, known aliases mapped."""
    import re
    n = re.sub(r"-\d$", "", str(name).strip().replace(".", "-")).upper()
    m = re.match(r"^(CD\d+[A-Z]?)-[A-Z]", n)  # "CD127-IL7Ra", "CD278-ICOS" -> the CD name
    if m:
        n = m.group(1)
    return ADT_ALIASES.get(n, n)


def adt_panel_overlap(ours, theirs) -> dict:
    a = {normalise_adt_name(x): x for x in ours}
    b = {normalise_adt_name(x): x for x in theirs}
    shared = sorted(set(a) & set(b))
    return {"n_ours": len(a), "n_theirs": len(b), "n_shared": len(shared),
            "shared": [[a[s], b[s]] for s in shared]}


STUART2019_RNA_CACHE = "rna_counts_cells_x_genes.npz"


def _stuart_rna_sparse(root: str):
    """The RNA TSV (genes x cells, dense) as a cells x genes CSR matrix, built
    once and cached next to the source files (a derived file, listed in the
    directory's manifest)."""
    import pandas as pd
    import scipy.sparse as sp
    cache = os.path.join(root, STUART2019_RNA_CACHE)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        M = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return M, [str(g) for g in z["genes"]], [str(c) for c in z["cells"]]
    blocks, genes = [], []
    path = os.path.join(root, STUART2019_FILES["rna"])
    cells = [str(c) for c in pd.read_csv(path, sep="	", index_col=0, nrows=0).columns]
    for chunk in pd.read_csv(path, sep="	", index_col=0, chunksize=1000):
        blocks.append(sp.csr_matrix(chunk.to_numpy(np.float32)))
        genes += [str(g) for g in chunk.index]
    M = sp.vstack(blocks).T.tocsr()
    try:
        np.savez(cache, data=M.data, indices=M.indices, indptr=M.indptr, shape=np.array(M.shape),
                 genes=np.array(genes), cells=np.array(cells))
    except OSError:
        pass
    return M, genes, cells


def load_stuart2019_bmcite(root: str = STUART2019_DIR, genes=None) -> dict:
    """Stuart et al. 2019 bone-marrow CITE-seq (GEO GSE128639, public, no
    account): RNA counts (cells x genes; restricted to `genes` when given, with
    each cell's total over ALL genes in `rna_total` for normalisation), ADT
    counts (25 antibodies), and the hashtag (HTO) with the highest count per
    cell as the sample label."""
    import pandas as pd
    adt = pd.read_csv(os.path.join(root, STUART2019_FILES["adt"]), sep="	", index_col=0)
    hto = pd.read_csv(os.path.join(root, STUART2019_FILES["hto"]), sep="	", index_col=0)
    # the hashtag file writes barcodes as "<bc>-1" where the RNA/ADT files write "<bc>.1"
    import re
    hto.columns = [re.sub(r"-(\d+)$", r".\1", str(c)) for c in hto.columns]
    M, all_genes, rna_cells = _stuart_rna_sparse(root)
    total = np.asarray(M.sum(axis=1)).ravel()
    cols = [i for i, g in enumerate(all_genes) if genes is None or g in set(genes)]
    names = [all_genes[i] for i in cols]
    cells = [c for c in rna_cells if c in adt.columns]
    ri = {c: i for i, c in enumerate(rna_cells)}
    rows = np.array([ri[c] for c in cells], dtype=np.int64)
    missing_hto = [c for c in cells if c not in hto.columns]
    if missing_hto:
        raise ValueError("%d cells have no hashtag counts (barcode mismatch?)" % len(missing_hto))
    sample = hto[cells].to_numpy().argmax(axis=0)
    return {"cells": cells, "genes": names, "rna_counts": M[rows][:, cols].toarray().astype(np.float32),
            "rna_total": total[rows], "adt_names": [str(a) for a in adt.index],
            "adt_counts": adt[cells].to_numpy(np.float32).T,
            "sample": np.asarray([str(hto.index[k]) for k in sample]), "source": "GEO GSE128639 (Stuart et al. 2019)"}


def clr_rows(counts) -> np.ndarray:
    """Centred log-ratio per cell (the usual ADT normalisation)."""
    x = np.log1p(np.asarray(counts, float))
    return x - x.mean(axis=1, keepdims=True)


def log_normalise_rows(counts, library_size=None, total: float = 1e4) -> np.ndarray:
    """log1p(counts / library size * 1e4). Pass `library_size` (each cell's
    total over all genes) whenever `counts` holds only a subset of genes."""
    c = np.asarray(counts, float)
    lib = c.sum(axis=1) if library_size is None else np.asarray(library_size, float)
    return np.log1p(c / np.maximum(lib, 1.0)[:, None] * total)


# --------------------------------------------------------------------------
# H&E-only student (assay distillation; sprint 2026-10-15, appended)
# --------------------------------------------------------------------------
#: the ridge alpha grid of ``_oof_predictions`` / ``distilled_assay``
STUDENT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def he_student_fit(E_support, state_score, groups=None, *, kind: str = "ridge", seed: int = 0,
                   alpha_cv: str = "grouped", hidden: int = 64) -> dict:
    """Fit an H&E-only student: frozen pathology features ``E_support`` (rows = spots,
    tiles or slides) -> a frozen spatial state score.

    ``kind="ridge"``: the discovery core's inner-CV ridge (``_cv_alpha`` on
    ``STUDENT_ALPHA_GRID``, ``_RidgeFit``), alpha chosen by group-wise CV over
    ``groups`` (``alpha_cv="grouped"``) or row-wise CV (``alpha_cv="rows"``, exactly
    as ``distilled_assay``).  ``kind="mlp"``: one hidden layer of ``hidden`` units on
    standardised features (scikit-learn ``MLPRegressor``, early stopping, ``seed``).
    Rows with a non-finite feature or score are dropped."""
    from morpheus_encoder.general_discovery import _cv_alpha, _RidgeFit
    y = np.asarray(state_score, dtype=float).reshape(-1)
    E = np.asarray(E_support, dtype=float).reshape(len(y), -1)
    keep = _complete(E, y)
    E, y = E[keep], y[keep]
    g = None if groups is None else np.asarray(groups)[keep]
    if kind == "ridge":
        cv_groups = g if alpha_cv == "grouped" else None
        alpha = _cv_alpha(E, y, cv_groups, STUDENT_ALPHA_GRID, seed)
        return {"kind": "ridge", "alpha": float(alpha), "fit": _RidgeFit(E, y, alpha), "n_train": int(len(y)),
                "n_features": int(E.shape[1]), "alpha_cv": alpha_cv}
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        mu, sd = E.mean(0), E.std(0)
        sd[sd < 1e-12] = 1.0
        net = MLPRegressor(hidden_layer_sizes=(int(hidden),), alpha=1e-3, early_stopping=True,
                           validation_fraction=0.15, max_iter=400, random_state=int(seed))
        net.fit((E - mu) / sd, y)
        return {"kind": "mlp", "fit": net, "mu": mu, "sd": sd, "n_train": int(len(y)),
                "n_features": int(E.shape[1])}
    raise ValueError("kind must be 'ridge' or 'mlp'")


def he_student_predict(model: Mapping[str, Any], E) -> np.ndarray:
    """Student predictions for feature rows ``E`` (NaN where a feature is non-finite)."""
    E = np.asarray(E, dtype=float)
    E = E.reshape(len(E), -1)
    out = np.full(len(E), np.nan)
    ok = np.all(np.isfinite(E), axis=1)
    if ok.any():
        if model["kind"] == "ridge":
            out[ok] = model["fit"].predict(E[ok])
        else:
            out[ok] = model["fit"].predict((E[ok] - model["mu"]) / model["sd"])
    return out


def he_student_patient_scores(model: Mapping[str, Any], E, sample_ids) -> dict:
    """Mean student prediction per sample / slide / patient id (rows of ``E`` grouped by
    ``sample_ids``), in order of first appearance.  For the ridge student the mean of
    the predictions equals the prediction of the mean-pooled features."""
    pred = he_student_predict(model, E)
    ids = np.asarray(sample_ids).astype(str)
    out: dict = {}
    for s in dict.fromkeys(ids.tolist()):
        v = pred[ids == s]
        v = v[np.isfinite(v)]
        out[s] = float(v.mean()) if len(v) else float("nan")
    return out


def _distilled_style_summary(yk: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> dict:
    """The row-bootstrap summary of ``distilled_assay`` for given out-of-fold predictions."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        i = rng.integers(0, len(yk), len(yk))
        boots.append((_r2_of(yk[i], p[i]), spearmanr(yk[i], p[i]).statistic))
    b = np.asarray(boots)
    return {"n": int(len(yk)), "heldout_r2": _r2_of(yk, p),
            "heldout_r2_ci95": [float(np.quantile(b[:, 0], 0.025)), float(np.quantile(b[:, 0], 0.975))],
            "heldout_spearman": float(spearmanr(yk, p).statistic),
            "heldout_spearman_ci95": [float(np.quantile(b[:, 1], 0.025)), float(np.quantile(b[:, 1], 0.975))]}


def he_student_evaluate(E, state_score, fold_ids, groups, *, kind: str = "ridge", seed: int = 0,
                        n_boot: int = 1000, alpha_cv: str = "grouped", with_reference: bool = False,
                        hidden: int = 64) -> dict:
    """Patient-grouped held-out evaluation of the H&E-only student.

    Out-of-fold predictions: for every fold the student is fitted on the other folds'
    rows (``he_student_fit``) and predicts the fold's rows; ``fold_ids`` must keep a
    patient's rows in one fold.  Reported: ``row_level`` (``heldout_summary`` with a
    patient bootstrap), ``patient_level`` (patient means of score and prediction,
    ``heldout_summary``) and ``row_bootstrap`` (``distilled_assay``'s own summary on the
    same predictions).  ``with_reference`` also runs ``distilled_assay`` on the same
    inputs and records how far the student's out-of-fold predictions are from it
    (identical for the ridge student with ``alpha_cv="rows"``)."""
    y = np.asarray(state_score, dtype=float).reshape(-1)
    X = np.asarray(E, dtype=float).reshape(len(y), -1)
    folds = np.asarray(fold_ids)
    g = np.asarray(groups).astype(str)
    for f in np.unique(folds):
        if set(g[folds == f]) & set(g[folds != f]):
            raise ValueError("fold_ids split a patient across folds")
    keep = _complete(X, y)
    Xk, yk, fk, gk = X[keep], y[keep], folds[keep], g[keep]
    pred = np.full(len(yk), np.nan)
    for f in np.unique(fk):
        tr_, te = fk != f, fk == f
        m = he_student_fit(Xk[tr_], yk[tr_], gk[tr_], kind=kind, seed=seed, alpha_cv=alpha_cv, hidden=hidden)
        pred[te] = he_student_predict(m, Xk[te])
    pats = list(dict.fromkeys(gk.tolist()))
    py = np.array([yk[gk == p].mean() for p in pats])
    pp = np.array([pred[gk == p].mean() for p in pats])
    out = {"kind": kind, "alpha_cv": alpha_cv if kind == "ridge" else None, "n_rows": int(len(yk)),
           "n_patients": len(pats), "n_folds": int(len(np.unique(fk))), "n_features": int(Xk.shape[1]),
           "row_level": heldout_summary(yk, pred, gk, n_boot=n_boot, seed=seed),
           "patient_level": (heldout_summary(py, pp, np.array(pats), n_boot=n_boot, seed=seed)
                             if len(pats) >= 3 else None),
           "row_bootstrap": _distilled_style_summary(yk, pred, n_boot, seed),
           "oof_predictions": pred}
    if with_reference:
        ref = distilled_assay(X, y, folds, n_boot=n_boot, seed=seed)
        ref_pred = _oof_predictions(Xk, yk, fk, seed=seed)
        out["reference_distilled_assay"] = ref
        out["max_abs_oof_difference"] = float(np.nanmax(np.abs(ref_pred - pred)))
        out["reproduces_reference"] = all(
            np.allclose(np.asarray(out["row_bootstrap"][k], float), np.asarray(ref[k], float), atol=1e-9)
            for k in ("heldout_r2", "heldout_r2_ci95", "heldout_spearman", "heldout_spearman_ci95"))
    return out


def mean_pool_tile_embeddings(npz_paths: Sequence[str]) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Slide-mean of stored fp32 per-tile H-Optimus-0 embeddings (``embeddings`` (t, d) in
    each npz, with ``patient_id`` / ``slide_id``): (patient ids, slide ids, (slides, d)
    float64 means, tile counts)."""
    pats, slides, means, counts = [], [], [], []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as z:
            e = np.asarray(z["embeddings"], dtype=np.float64)
            pats.append(str(z["patient_id"]))
            slides.append(str(z["slide_id"]))
        means.append(e.mean(0))
        counts.append(len(e))
    return pats, slides, np.vstack(means), np.asarray(counts)
