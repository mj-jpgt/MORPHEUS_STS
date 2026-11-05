"""Clinical analysis (downstream spec Section 8): one prespecified test per prioritised finding per cohort.

Cohorts and availability (public downloads only, no account, no data-use agreement):
* TCGA-KIRC: stored clinical table (OS, PFS, stage, age, sex; histologic grade is NOT in the stored
  table and is recorded as unavailable in every TCGA covariate set).
* JAVELIN Renal 101: Motzer et al. Nat Med 2020 supplementary workbook (Springer ESM, public): PFS,
  arm, age, sex (S11), log2 TPM for 726 patients (S13), the published signature interactions (S6), the
  Angio (S8) and Immuno (S15 'Javelin_Renal_101_genes26') gene lists.
* CheckMate 009/010/025: Braun et al. Nat Med 2020 supplementary workbook (Springer ESM, public).
  Read here for availability; the trial analyses use the bulk molecular score (Section 5 arithmetic).
* IMmotion151: unavailable (RNA-seq under EGA controlled access EGAS00001004353); the IMmotion151
  molecular clusters and the 2026 17 consensus programmes are unavailable as gene lists here and are
  marked so in the 8.4 comparison.

Pipeline test: the published JAVELIN Immuno and Angio signature-by-arm interactions (S6) are recomputed
with this code (signature score = mean of per-gene z-scored log2 TPM, median split, Cox PFS on arm, high,
arm x high) before any new score enters.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from . import locks

STORE = "/lambda/nfs/geeg/biorag3_persistent_20260711"
TRIALS = f"{STORE}/external/trials_public_20261021"
JAVELIN_XLSX = f"{TRIALS}/javelin_motzer2020_MOESM3_ESM.xlsx"
BRAUN_XLSX = f"{TRIALS}/braun2020_MOESM2_ESM.xlsx"
DERIVED = f"{TRIALS}/derived_csv"
AVAILABILITY = {
    "TCGA-KIRC": {"status": "available", "note": "histologic grade absent from the stored clinical table"},
    "JAVELIN Renal 101": {"status": "available", "source": "Springer ESM 41591_2020_1044_MOESM3_ESM.xlsx (public)",
                          "sha256": "673c5fd66813e3267b3a514074ac22d090a258b1094b45119d592405d5f663ed"},
    "CheckMate 009/010/025": {"status": "available", "source": "Springer ESM 41591_2020_839_MOESM2_ESM.xlsx (public)",
                              "sha256": "a1f8683676d416b255291dcaa007aa0fa48c736c23e6a7a566664ba528bf205b"},
    "IMmotion151": {"status": "unavailable", "reason": "RNA-seq under EGA controlled access (EGAS00001004353); "
                                                       "no access requested"},
    "IMmotion151 molecular clusters (gene lists)": {"status": "unavailable", "reason": "not in a public table read here"},
    "17 consensus transcriptomic programmes (2026)": {"status": "unavailable",
                                                     "reason": "gene lists not located in a public download"},
}


# ----------------------------------------------------------------------------- xlsx
def _xlsx_extract(xlsx, sheet, out_csv):
    """Streaming sheet -> CSV (the expression sheets inflate to 0.5 GB of XML; openpyxl is too slow)."""
    import re
    import xml.etree.ElementTree as ET
    import zipfile

    NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(xlsx)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        with z.open("xl/sharedStrings.xml") as fh:
            for _, el in ET.iterparse(fh, events=("end",)):
                if el.tag == NS + "si":
                    ss.append("".join(t.text or "" for t in el.iter(NS + "t")))
                    el.clear()
    wb = z.read("xl/workbook.xml").decode("utf8", "replace")
    rels = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', z.read("xl/_rels/workbook.xml.rels").decode()))
    target = {m.group(1): "xl/" + rels[m.group(2)].lstrip("/")
              for m in re.finditer(r'<sheet[^>]*name="([^"]*)"[^>]*r:id="([^"]+)"', wb)}[sheet]

    def col(ref):
        n = 0
        for ch in ref:
            if not ch.isalpha():
                break
            n = n * 26 + ord(ch.upper()) - 64
        return n - 1
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tmp = out_csv + f".{os.getpid()}.tmp"
    with z.open(target) as fh, open(tmp, "w", newline="") as oh:
        w, row = csv.writer(oh), []
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == NS + "c":
                ci, v, t = col(el.get("r", "A1")), el.find(NS + "v"), el.get("t")
                if t == "inlineStr":
                    isn = el.find(NS + "is")
                    val = "".join(x.text or "" for x in isn.iter(NS + "t")) if isn is not None else ""
                else:
                    val = "" if v is None else (ss[int(v.text)] if t == "s" else v.text)
                row += [""] * (ci + 1 - len(row))
                row[ci] = val
                el.clear()
            elif el.tag == NS + "row":
                w.writerow(row)
                row = []
                el.clear()
    os.replace(tmp, out_csv)
    return out_csv


def sheet_csv(xlsx, sheet):
    out = f"{DERIVED}/{Path(xlsx).stem}__{sheet}.csv"
    return out if os.path.exists(out) else _xlsx_extract(xlsx, sheet, out)


def load_javelin() -> dict:
    import pandas as pd

    clin = pd.read_csv(sheet_csv(JAVELIN_XLSX, "S11_Clinical_data"), skiprows=1).dropna(subset=["ID"]).set_index("ID")
    expr = pd.read_csv(sheet_csv(JAVELIN_XLSX, "S13_Gene_expression_TPM"), skiprows=1, index_col=0, low_memory=False)
    expr = expr.loc[~expr.index.duplicated()].apply(pd.to_numeric, errors="coerce").T          # patients x genes
    angio = [g for row in csv.reader(open(sheet_csv(JAVELIN_XLSX, "S8_Renal101_Angio_signature")))
             for g in row if g and not g.startswith("Supplementary")]
    immuno = []
    for row in csv.reader(open(sheet_csv(JAVELIN_XLSX, "S15_Pathway_genesets"))):
        if row and row[0] == "B9991003_Javelin_Renal_101_genes26":
            immuno = [g for g in row[2:] if g]
    pub = pd.read_csv(sheet_csv(JAVELIN_XLSX, "S6_UVA_MVA"), skiprows=1).set_index("Feature")
    common = [i for i in expr.index if i in clin.index]
    d = clin.loc[common].copy()
    d["time"] = pd.to_numeric(d["PFS_P"], errors="coerce")
    d["event"] = (pd.to_numeric(d["PFS_P_CNSR"], errors="coerce") == 0).astype(float)    # CNSR = 0 is the event
    d["arm_combo"] = (d["TRT01P"].astype(str).str.contains("Avelumab")).astype(float)
    d["age"] = pd.to_numeric(d["AGE"], errors="coerce")
    d["male"] = (d["SEX"].astype(str).str.upper().str.startswith("M")).astype(float)
    return {"clinical": d, "expr": expr.loc[common], "signatures": {"JAVELIN_Angio": angio, "JAVELIN_Immuno": immuno},
            "published": pub}


def load_braun_clinical():
    import pandas as pd

    return pd.read_csv(sheet_csv(BRAUN_XLSX, "S1_Clinical_and_Immune_Data"), skiprows=1)


# ----------------------------------------------------------------------------- Cox
def cox(time, event, X, names) -> dict:
    """Cox PH (Breslow ties) via statsmodels PHReg: HR, 95% CI, p per term, log-likelihood."""
    from statsmodels.duration.hazard_regression import PHReg

    X = np.asarray(X, float).reshape(len(time), -1)
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(X).all(1)
    m = PHReg(np.asarray(time)[ok], X[ok], status=np.asarray(event)[ok], ties="breslow").fit()
    ci = np.asarray(m.conf_int())
    return {"n": int(ok.sum()), "events": int(np.asarray(event)[ok].sum()), "llf": float(m.llf),
            "terms": {nm: {"hr": float(np.exp(b)), "ci95": [float(np.exp(lo)), float(np.exp(hi))], "p": float(p)}
                      for nm, b, (lo, hi), p in zip(names, m.params, ci, m.pvalues)},
            "_lp": X[ok] @ m.params, "_ok": ok}


def harrell_c(time, event, risk) -> float:
    from dbench.translational import harrell_c as hc

    return float(hc(time, event, risk))


def zscore(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def signature_score(expr, genes):
    g = [x for x in dict.fromkeys(genes) if x in expr.columns]
    Z = (expr[g] - expr[g].mean()) / expr[g].std(ddof=0).replace(0, 1)
    return Z.mean(1), len(g)


def interaction_test(d, score, *, binary_median: bool, covariates=()) -> dict:
    s = np.asarray(score, float)
    x = (s > np.nanmedian(s)).astype(float) if binary_median else zscore(s)
    arm = d["arm_combo"].to_numpy()
    cols = [arm, x, arm * x] + [d[c].to_numpy(float) for c in covariates]
    names = ["arm", "score", "arm_x_score"] + list(covariates)
    res = cox(d["time"].to_numpy(float), d["event"].to_numpy(float), np.column_stack(cols), names)
    per_arm = {}
    for a, nm in ((1.0, "avelumab_axitinib"), (0.0, "sunitinib")):
        m = arm == a
        per_arm[nm] = cox(d["time"].to_numpy(float)[m], d["event"].to_numpy(float)[m],
                          np.column_stack([x[m]] + [d[c].to_numpy(float)[m] for c in covariates]),
                          ["score"] + list(covariates))["terms"]["score"]
    return {"interaction": res["terms"]["arm_x_score"], "per_arm": per_arm, "n": res["n"], "events": res["events"]}


def reproduce_javelin_published(j: dict | None = None) -> dict:
    """Pipeline test (10.2 item 3): recompute the published univariate signature-by-arm interactions."""
    j = j or load_javelin()
    out = {}
    for nm, feat in (("JAVELIN_Immuno", "Javelin_Renal_101_Immuno"), ("JAVELIN_Angio", "Javelin_Renal_101_Angio")):
        s, n = signature_score(j["expr"], j["signatures"][nm])
        r = interaction_test(j["clinical"], s.to_numpy(), binary_median=True)
        pub = j["published"].loc[feat]
        out[nm] = {"n_genes_scored": n, "n_genes_listed": len(j["signatures"][nm]),
                   "recomputed_hr": r["interaction"]["hr"], "recomputed_ci95": r["interaction"]["ci95"],
                   "recomputed_p": r["interaction"]["p"], "published_hr": float(pub["Interaction_term_expCoef"]),
                   "published_p": float(pub["Interaction_term_pval"]),
                   "published_ci95": [float(pub["Interaction_term_low_CI95"]), float(pub["Interaction_term_high_CI95"])],
                   "per_arm": r["per_arm"]}
        out[nm]["recomputed_inside_published_ci"] = bool(out[nm]["published_ci95"][0] <= out[nm]["recomputed_hr"]
                                                         <= out[nm]["published_ci95"][1])
    return out


# ----------------------------------------------------------------------------- TCGA
def tcga_clinical():
    from .bulk import tcga_tables

    c = tcga_tables()["clinical"].copy()
    stage = c["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.upper()
    c["stage_num"] = np.select([stage.str.contains("IV"), stage.str.contains("III"), stage.str.contains("II"),
                                stage.str.contains("I")], [4, 3, 2, 1], np.nan)
    c["age"] = c["AGE"].astype(float)
    c["male"] = c["SEX"].astype(str).str.upper().str.startswith("M").astype(float)
    for ep in ("OS", "PFS"):
        c[f"{ep}_time"] = c[f"{ep}_MONTHS"].astype(float)
        c[f"{ep}_event"] = c[f"{ep}_STATUS"].astype(str).str.startswith("1").astype(float)
    return c


TCGA_COVARIATES = ("stage_num", "age", "male")


def prognostic_test(time, event, score, covs: dict) -> dict:
    """Cox with the score per SD plus covariates; incremental discrimination (dC, LR) vs covariates."""
    from scipy.stats import chi2

    s = zscore(score)
    C = np.column_stack(list(covs.values())) if covs else np.zeros((len(s), 0))
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(s) & np.isfinite(C).all(1)
    full = cox(time[ok], event[ok], np.column_stack([s[ok], C[ok]]), ["score_per_sd"] + list(covs))
    base = cox(time[ok], event[ok], C[ok], list(covs)) if C.shape[1] else None
    lr = 2 * (full["llf"] - (base["llf"] if base else 0.0))
    out = {"hr_per_sd": full["terms"]["score_per_sd"], "n": full["n"], "events": full["events"],
           "lr_statistic": float(lr), "lr_p": float(chi2.sf(lr, 1)),
           "c_full": harrell_c(time[ok], event[ok], full["_lp"])}
    if base:
        out["c_covariates"] = harrell_c(time[ok], event[ok], base["_lp"])
        out["delta_c"] = out["c_full"] - out["c_covariates"]
    return out


# ----------------------------------------------------------------------------- lock and 8.4
def write_clinical_lock(root, finding_id, *, cohort, endpoint, score_definition: dict, covariates, model,
                        family, replication_of=None, allow_violation=False) -> dict:
    return locks.write_lock(root, finding_id, "clinical",
                            inputs={"cohort": cohort, "score_definition_hash": locks.canonical_hash(score_definition)},
                            declared_analysis={"cohort": cohort, "endpoint": endpoint, "score_definition": score_definition,
                                               "covariates": list(covariates), "model": model,
                                               "multiplicity_family": sorted(family),
                                               "success_criterion": "corrected q <= 0.05 for the main-effect or "
                                                                    "interaction hazard ratio",
                                               "replication_of": replication_of, "one_endpoint_only": True},
                            allow_violation=allow_violation)


def known_programme_comparison(score, expr, known: dict) -> dict:
    """8.4: max |Spearman| of the finding's molecular score with each published clinical programme
    scored in the same cohort; > 0.7 = recovery of a known axis."""
    from scipy.stats import spearmanr

    rows = {}
    for nm, genes in known.items():
        s, n = signature_score(expr, genes)
        if n:
            rows[nm] = {"spearman": float(spearmanr(score, s, nan_policy="omit")[0]), "n_genes": n}
    mx = max((abs(v["spearman"]) for v in rows.values()), default=None)
    return {"programmes": rows, "max_abs_spearman": mx,
            "recovers_known_axis": bool(mx is not None and mx > 0.7),
            "unavailable": {k: v for k, v in AVAILABILITY.items() if v["status"] == "unavailable" and "gene" in k}}
