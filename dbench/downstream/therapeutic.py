"""Prioritisation (Section 4), DepMap and Open Targets evidence (Section 6.1, 6.2) and the therapeutic
relevance template (6.3). Evidence tables, never drug claims.

DepMap: the latest public Chronos gene effect obtainable without an account. The DepMap portal download
API sits behind a bot-verification page, so the release is read from the Broad DepMap figshare record
"Chronos parameters (Public 26Q1)" (figshare 31660582, ``gene_effect.csv``); model annotation (lineage,
Oncotree) from the stored DepMap 24Q4 ``Model.csv`` (models absent from it are counted and excluded);
PRISM Repurposing Public 24Q2 (figshare 25917643). The common-essential flag is DERIVED here
(Dempster 2019 90th-percentile rule, declared below) because the 26Q1 common-essential list is not in
that record. Every response is cached under a directory keyed by the release.

Open Targets: Platform GraphQL v4 through ``dbench.translational`` (its tested queries), cached under a
directory keyed by the platform data version read from the API at lock time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .locks import canonical_hash, sha256_file, utc_now

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
DEPMAP = {
    "release": "DepMap Public 26Q1 (Chronos gene effect, figshare 31660582 'Chronos parameters (Public 26Q1)')",
    "gene_effect": f"{STORE}/external/depmap_public_26q1_chronos_20261021/gene_effect.csv",
    "gene_effect_md5": "c89e3ec7e2c3682e5c3535172177a1ee",
    "model": f"{STORE}/external/depmap_20261010/Model.csv",
    "model_release": "DepMap 24Q4 Public Model.csv (figshare 27993248)",
    "prism_matrix": f"{STORE}/external/depmap_public_26q1_chronos_20261021/Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
    "prism_compounds": f"{STORE}/external/depmap_public_26q1_chronos_20261021/Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv",
    "prism_release": "PRISM Repurposing Public 24Q2 (figshare 25917643)",
}
CE_RULE = ("common essential (derived): the gene's normalised depletion rank (1 = most depleted gene of the "
           "model, scaled to 0-1) at the 90th percentile of models is <= 0.10 (Dempster 2019 rule; DepMap's own "
           "list is not in the 26Q1 figshare record)")
DEP_THR = -0.5
CLINICAL = {"Approved Drug", "Advanced Clinical", "Phase 1 Clinical"}
# Open Targets tractability precedence groups: clinical precedence (Approved Drug, Advanced Clinical, Phase 1
# Clinical) and discovery precedence (small molecule: Structure with Ligand, High-Quality Ligand; PROTAC:
# Literature); pockets, druggable family, localisation and ubiquitination evidence are 'predicted tractable'.
DISCOVERY = {"Structure with Ligand", "High-Quality Ligand", "Literature"}
STAGE_RANK = {"APPROVAL": 5, "PHASE_4": 5, "PHASE_3": 4, "PHASE_2_3": 3.5, "PHASE_2": 3, "PHASE_1_2": 2.5,
              "PHASE_1": 2, "EARLY_PHASE_1": 1.5, "PRECLINICAL": 1, "UNKNOWN": 0}
CCRCC_EFO = "EFO_0000349"


# ----------------------------------------------------------------------------- DepMap
class DepMap:
    def __init__(self, cfg: dict = DEPMAP, cache_root: str = "runs/development/downstream_v01/cache"):
        import pandas as pd

        self.cfg = cfg
        self.cache = Path(cache_root) / ("depmap_" + canonical_hash(cfg["release"])[:12])
        self.cache.mkdir(parents=True, exist_ok=True)
        with open(cfg["gene_effect"]) as fh:
            self.header = fh.readline().rstrip("\n").split(",")
        self.col = {h.split(" (")[0]: i for i, h in enumerate(self.header) if i}
        m = pd.read_csv(cfg["model"], usecols=["ModelID", "OncotreeLineage", "OncotreePrimaryDisease",
                                               "OncotreeSubtype"]).set_index("ModelID")
        self.models = m
        self._ce = None

    def version_record(self) -> dict:
        return {"release": self.cfg["release"], "gene_effect_sha256": sha256_file(self.cfg["gene_effect"]),
                "gene_effect_md5_figshare": self.cfg["gene_effect_md5"], "model_release": self.cfg["model_release"],
                "prism_release": self.cfg["prism_release"], "common_essential_rule": CE_RULE}

    def matrix(self):
        if getattr(self, "_M", None) is None:
            import pandas as pd

            M = pd.read_csv(self.cfg["gene_effect"], index_col=0)
            M.columns = [c.split(" (")[0] for c in M.columns]
            self._M = M.loc[:, ~M.columns.duplicated()]
        return self._M

    def _common_essentials(self) -> dict:
        p = self.cache / "common_essential.json"
        if p.exists():
            return json.loads(p.read_text())
        M = self.matrix()
        R = M.rank(axis=1, method="average") / M.notna().sum(1).values[:, None]
        q90 = R.quantile(0.9, axis=0)
        out = {g: bool(v <= 0.10) for g, v in q90.items() if v == v}
        p.write_text(json.dumps(out))
        return out

    def gene(self, symbol: str) -> dict:
        p = self.cache / f"gene_{symbol}.json"
        if p.exists():
            return json.loads(p.read_text())
        import pandas as pd

        if symbol not in self.col:
            out = {"gene": symbol, "status": "not screened in this release"}
        else:
            e = self.matrix()[symbol].dropna()
            ann = self.models.reindex(e.index)
            known = ann["OncotreeLineage"].notna()
            rcc = (ann["OncotreeLineage"] == "Kidney") & ann["OncotreePrimaryDisease"].astype(str).str.contains(
                "Renal Cell Carcinoma", case=False)
            cc = rcc & ann["OncotreeSubtype"].astype(str).str.contains("Clear Cell", case=False)
            if self._ce is None:
                self._ce = self._common_essentials()
            ce = bool(self._ce.get(symbol, False))
            k = e[rcc]
            out = {"gene": symbol, "status": "assessed", "n_models": int(len(e)),
                   "n_models_without_annotation": int((~known).sum()), "n_rcc": int(rcc.sum()), "n_ccrcc": int(cc.sum()),
                   "median_pan_cancer": float(e.median()),
                   "median_rcc": float(k.median()) if len(k) else None,
                   "median_ccrcc": float(e[cc].median()) if cc.sum() else None,
                   "fraction_rcc_le_-0.5": float((k <= DEP_THR).mean()) if len(k) else None,
                   "common_essential_derived": ce}
            out["context_dependency"] = bool(len(k) and out["median_rcc"] <= DEP_THR and not ce)
        out["prism"] = self.prism(symbol)
        p.write_text(json.dumps(out))
        return out

    def prism(self, symbol: str) -> dict:
        import pandas as pd

        if getattr(self, "_prism", None) is None:
            self._prism = (pd.read_csv(self.cfg["prism_compounds"]), pd.read_csv(self.cfg["prism_matrix"], index_col=0))
        cl, M = self._prism
        tgt = cl["repurposing_target"].astype(str).str.split(r"\s*,\s*")
        hit = cl[[symbol in t for t in tgt]]
        if hit.empty:
            return {"status": "no PRISM compound annotated to this target"}
        ann = self.models.reindex(M.columns)
        kid = (ann["OncotreeLineage"] == "Kidney").values
        rows = []
        for _, r in hit.drop_duplicates("IDs").iterrows():
            if r["IDs"] in M.index:
                v = M.loc[r["IDs"]]
                v = v.iloc[0] if v.ndim > 1 else v
                rows.append({"compound": r["Drug.Name"], "id": r["IDs"], "moa": r.get("MOA"),
                             "median_lfc_all": float(np.nanmedian(v.values)),
                             "median_lfc_kidney": float(np.nanmedian(v.values[kid])) if kid.any() else None,
                             "n_kidney": int(np.isfinite(v.values[kid]).sum())})
        return {"status": "assessed", "compounds": rows}


# ----------------------------------------------------------------------------- Open Targets
def open_targets_version(cache_root: str) -> dict:
    from dbench.translational import _Q_META, _graphql

    meta, _ = _graphql(_Q_META, {}, cache_dir=None, offline=False)
    api, data = meta["meta"]["apiVersion"], meta["meta"]["dataVersion"]
    ver = f"{data['year']}.{data['month']}" + (f".{data.get('iteration')}" if data.get("iteration") is not None else "")
    return {"api_version": f"{api['x']}.{api['y']}.{api['z']}", "data_version": ver,
            "cache_dir": str(Path(cache_root) / f"opentargets_{ver}"), "read_utc": utc_now()}


def resolve_mondo(cache_dir: str) -> dict:
    """The spec names EFO_0000349 'and its MONDO equivalent as resolved at lock time'. Platform releases from
    2025 index ccRCC under MONDO; the id that resolves is used and both are recorded."""
    from dbench.translational import _graphql, _resolve

    q = """query($d:String!){ disease(efoId:$d){ id name dbXRefs } }"""
    data, _ = _graphql(q, {"d": CCRCC_EFO}, cache_dir=cache_dir, offline=False)
    d = data.get("disease") or {}
    mid, mname = _resolve("clear cell renal carcinoma", "disease", cache_dir=cache_dir, offline=False)
    use = d.get("id") or mid
    return {"efo": CCRCC_EFO, "efo_resolves": bool(d.get("id")), "mondo": mid, "mondo_name": mname,
            "disease_id_used": use}


def ot_gene(symbol: str, cache_dir: str, disease_id: str = CCRCC_EFO) -> dict:
    from dbench.translational import open_targets_evidence

    ev = open_targets_evidence(symbol, "clear cell renal carcinoma", disease_id=disease_id, cache_dir=cache_dir)
    if ev.get("status") != "assessed":
        return {"gene": symbol, "status": ev.get("status"), "reason": ev.get("reason")}
    tract = ev.get("tractability") or {}
    cands = ev["clinical_candidates"]["rows"]
    best = max([STAGE_RANK.get(str(c["max_clinical_stage"]), 0) for c in cands], default=0)
    rcc = max([STAGE_RANK.get(str(c["max_clinical_stage"]), 0) for c in ev["clinical_candidates"]["renal_indication_rows"]],
              default=0)
    assoc = ev.get("association") or {}
    dts = assoc.get("datatype_scores") or {}
    labels = {m: set(v) for m, v in tract.items()}
    return {"gene": symbol, "status": "assessed", "ensembl_id": ev["ensembl_id"],
            "association_score": assoc.get("overall_score"), "datatype_scores": dts,
            "strongest_evidence_type": max(dts, key=dts.get) if dts else None,
            "highest_phase_any": best, "highest_phase_rcc": rcc,
            "tractability": {m: sorted(v) for m, v in labels.items()},
            "handle_sm_or_ab": bool(labels.get("SM") or labels.get("AB")),
            "actionable_in_principle": any(v & (CLINICAL | DISCOVERY) for v in labels.values()),
            "clinically_precedented": best >= STAGE_RANK["PHASE_2"],
            "precedented_drugs": [c["drug"] for c in cands if STAGE_RANK.get(str(c["max_clinical_stage"]), 0) >= 3][:10],
            "n_safety_flags": len(ev.get("safety_liabilities") or []), "data_version_cache": cache_dir}


# ----------------------------------------------------------------------------- Section 4 and 6.3
def prioritise(findings: Sequence[dict], max_n: int = 4) -> list[dict]:
    """findings: dicts with external_ci_lower, higher_order (bool), grade, has_handle (bool),
    external_gain. Score = sum of the four Section 4 indicators; top four, ties by external gain."""
    good = {"new association", "compositional novelty"}
    out = []
    for f in findings:
        ind = {"external_ci_lower_gt_0.01": bool(f["external_ci_lower"] is not None and f["external_ci_lower"] > 0.01),
               "higher_order": bool(f["higher_order"]),
               "literature_grade": bool(f.get("grade") in good or "mechanistic novelty" in (f.get("tags") or [])),
               "therapeutic_handle": bool(f.get("has_handle"))}
        out.append({**f, "indicators": ind, "score": int(sum(ind.values()))})
    out.sort(key=lambda f: (-f["score"], -(f.get("external_gain") or 0)))
    for i, f in enumerate(out):
        f["selected"] = i < max_n
        f["section6_eligible"] = f["selected"] and f["indicators"]["therapeutic_handle"]
    return out


def relevance_statement(finding_id: str, depmap_rows: Sequence[dict], ot_rows: Sequence[dict],
                        top_decile_occupancy: dict | None) -> str:
    dep = [r["gene"] for r in depmap_rows if r.get("context_dependency")]
    act = [r["gene"] for r in ot_rows if r.get("actionable_in_principle")]
    drugs = sorted({d for r in ot_rows if r.get("clinically_precedented") for d in r.get("precedented_drugs", [])})
    ctx = "not assessed (no composition strata)"
    if top_decile_occupancy:
        tum = top_decile_occupancy.get("tumour-rich", 0.0)
        micro = sum(v for k, v in top_decile_occupancy.items() if k in ("immune-rich", "stroma-rich"))
        ctx = ("tumour-intrinsic" if tum > micro else "microenvironmental") + \
            f" (state-high spots: {tum:.0%} tumour-rich, {micro:.0%} immune- or stroma-rich)"
    s = [f"Finding {finding_id}.",
         "Context dependencies in kidney-lineage models: " + (", ".join(dep) if dep else "none") + ".",
         "Constituents actionable in principle: " + (", ".join(act) if act else "none") + ".",
         "Clinically precedented drugs against constituents: " + (", ".join(drugs) if drugs else "none") + ".",
         f"Spatial context suggests a {ctx} mechanism."]
    if drugs:
        s.append("The finding provides a spatial rationale for investigating these agents, and nothing stronger.")
    s.append("No efficacy claim is made. Absence of a monoculture dependency is not a negative result for a "
             "spatial multicellular relationship.")
    return " ".join(s)
