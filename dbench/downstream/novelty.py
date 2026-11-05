"""Novelty grading (downstream spec Section 3): queries generated from the characterisation record, a
literature lock written before any search, retrieval of up to 200 records per query from OpenAlex and
Europe PMC (``dbench.novelty_check`` cache-key and date conventions), the fixed relevance screen, and the
3.3 decision procedure producing a DRAFT grade and a two-reviewer form. The final grade is the
reviewers'; the code never assigns it alone.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Sequence

from . import locks

SPATIAL_TERMS = ("neighbourhood", "neighborhood", "proximity", "boundary", "niche", "interface", "gradient",
                 "spatial transcriptomics")
DISEASE = {"ccRCC": {"query": "clear cell renal cell carcinoma",
                     "screen": ("clear cell renal cell carcinoma", "clear-cell renal cell carcinoma", "ccrcc",
                                "clear cell rcc", "renal cell carcinoma")}}
MAX_RECORDS = 200
GRADES = ("no novelty identified", "spatial or scale extension", "context extension", "compositional novelty",
          "new association")


# ----------------------------------------------------------------------------- labels and queries
def humanise(name: str) -> str:
    hallmark = "HALLMARK_" in name
    n = re.sub(r"^(own|nbr|nbr18):", "", name)
    n = re.sub(r"^context_neighbour_mean:(nbr18|nbr):", "", n)
    n = re.sub(r"^(programmes:)?(hallmark|gavish|gene|residual|raw_genes):", "", n)
    n = n.split("@")[0].replace("HALLMARK_", "").replace("_", " ")
    return n.lower() if hallmark else n


def constituent_labels(char_record: dict, target: str | None = None, n_loaded: int = 3) -> list[str]:
    """Gene symbols, programme names and descriptive morphology labels; a residual programme contributes
    its top ``n_loaded`` positive and negative loaded genes (declared)."""
    out = []
    for row in char_record["constituents"]["rows"]:
        if row.get("loaded_genes"):
            out += [g for g, _ in row["loaded_genes"]["positive"][:n_loaded]]
            out += [g for g, _ in row["loaded_genes"]["negative"][:n_loaded]]
        elif row.get("interpretation_panel", {}).get("label"):
            out.append(row["interpretation_panel"]["label"])
        else:
            out.append(humanise(row["candidate"]))
    if target:
        out.append(humanise(target))
    return list(dict.fromkeys(out))


def _q(term: str) -> str:
    return f'"{term}"'


def generate_queries(labels: Sequence[str], disease: str = "ccRCC") -> list[dict]:
    dz = _q(DISEASE[disease]["query"])
    sp = "(" + " OR ".join(_q(t) for t in SPATIAL_TERMS) + ")"
    qs = [{"family": "constituent", "terms": [c], "text": f"{_q(c)} AND {dz}"} for c in labels]
    qs += [{"family": "pairwise", "terms": [a, b], "text": f"{_q(a)} AND {_q(b)} AND {dz}"}
           for a, b in itertools.combinations(labels, 2)]
    qs += [{"family": "spatial_context", "terms": [c], "text": f"{_q(c)} AND {dz} AND {sp}"} for c in labels]
    allc = " AND ".join(_q(c) for c in labels)
    qs += [{"family": "full_composition", "terms": list(labels), "text": f"{allc} AND {dz}"},
           {"family": "full_composition_spatial", "terms": list(labels), "text": f"{allc} AND {dz} AND {sp}"}]
    return qs


def write_literature_lock(root, finding_id: str, char_record_hash: str, labels, queries, *, date: str,
                          before_date: str | None = None, allow_violation=False) -> dict:
    return locks.write_lock(root, finding_id, "literature",
                            inputs={"characterisation_record_hash": char_record_hash, "labels": list(labels)},
                            declared_analysis={"date": date, "before_date": before_date,
                                               "query_templates": "dbench.downstream.novelty.generate_queries",
                                               "queries": [q["text"] for q in queries],
                                               "max_records_per_query": MAX_RECORDS,
                                               "sources": ["OpenAlex works", "Europe PMC (core)"],
                                               "screen": "disease term in title/abstract AND >= 1 constituent"},
                            allow_violation=allow_violation)


# ----------------------------------------------------------------------------- retrieval
def _abstract_from_inverted(inv) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, ix in inv.items() for i in ix)
    return " ".join(w for _, w in pos)


def fetch(query: str, source: str, *, cache_dir, offline: bool, before_date: str | None = None) -> dict:
    from dbench.novelty_check import _cache_key, _check_date, _http_get_json

    before_date = _check_date(before_date)
    key = "records200::" + source
    p = Path(cache_dir) / _cache_key(key, query, before_date)
    if p.exists():
        return json.loads(p.read_text())
    if offline:
        return {"source": source, "query": query, "offline": True, "records": [], "n_total": None}
    if source == "openalex":
        url = ("https://api.openalex.org/works?per-page=200&search=" + urllib.parse.quote(query)
               + (f"&filter=to_publication_date:{before_date}" if before_date else ""))
        d = _http_get_json(url, timeout=60)
        recs = [{"id": r.get("id"), "doi": r.get("doi"), "title": r.get("display_name") or "",
                 "abstract": _abstract_from_inverted(r.get("abstract_inverted_index")),
                 "year": r.get("publication_year")} for r in d.get("results", [])[:MAX_RECORDS]]
        n = (d.get("meta") or {}).get("count")
    else:
        q = query + (f" AND FIRST_PDATE:[1800-01-01 TO {before_date}]" if before_date else "")
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&pageSize=200&query="
               + urllib.parse.quote(q))
        d = _http_get_json(url, timeout=60)
        recs = [{"id": f"{r.get('source')}:{r.get('id')}", "doi": r.get("doi"), "title": r.get("title") or "",
                 "abstract": r.get("abstractText") or "", "year": r.get("pubYear")}
                for r in ((d.get("resultList") or {}).get("result") or [])[:MAX_RECORDS]]
        n = d.get("hitCount")
    out = {"source": source, "query": query, "offline": False, "records": recs, "n_total": n,
           "before_date": before_date, "retrieved_utc": locks.utc_now()}
    os.makedirs(cache_dir, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def _has(text: str, term: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])", text.lower()) is not None


def screen(record: dict, labels: Sequence[str], disease: str = "ccRCC") -> dict:
    """Fixed relevance rule: disease term in title/abstract and >= 1 constituent in title/abstract."""
    txt = f"{record.get('title', '')} {record.get('abstract', '')}"
    dz = any(_has(txt, t) for t in DISEASE[disease]["screen"])
    hit = [c for c in labels if _has(txt, c)]
    return {"disease": dz, "constituents_present": hit, "spatial": [t for t in SPATIAL_TERMS if _has(txt, t)],
            "passes": bool(dz and hit)}


def run_queries(root, finding_id: str, queries, labels, *, cache_dir, offline=False, before_date=None,
                disease="ccRCC") -> dict:
    lock = locks.require_lock(root, finding_id, "literature")
    if [q["text"] for q in queries] != lock["declared_analysis"]["queries"]:
        raise locks.LockError("queries differ from the literature lock")
    log, kept = [], {}
    for q in queries:
        for src in ("openalex", "europepmc"):
            try:
                res = fetch(q["text"], src, cache_dir=cache_dir, offline=offline, before_date=before_date)
            except Exception as e:  # recorded, never silently dropped
                log.append({"family": q["family"], "query": q["text"], "source": src, "error": str(e)})
                continue
            n_pass = 0
            for r in res["records"]:
                s = screen(r, labels, disease)
                if s["passes"]:
                    n_pass += 1
                    key = (r.get("doi") or r["id"] or "").lower()
                    e = kept.setdefault(key, {**r, "screen": s, "families": set(), "query_terms": []})
                    e["families"].add(q["family"])
                    e["query_terms"].append(q["terms"])
            log.append({"family": q["family"], "query": q["text"], "source": src, "n_total": res.get("n_total"),
                        "n_retrieved": len(res["records"]), "n_screened_in": n_pass, "offline": res.get("offline")})
    recs = [{**v, "families": sorted(v["families"])} for v in kept.values()]
    return {"query_log": log, "records": recs, "lock_hash": lock["lock_hash"]}


# ----------------------------------------------------------------------------- grading
def record_flags(rec: dict, labels: Sequence[str], target_label: str | None) -> dict:
    s = rec["screen"]
    cons = [c for c in labels if c != target_label]
    present = set(s["constituents_present"])
    return {"full_composition": bool(cons) and set(cons) <= present,
            "spatial_context": bool(s["spatial"]), "same_disease": bool(s["disease"]),
            "constituent_with_target": sorted(c for c in cons if c in present and
                                              (target_label is None or target_label in present))}


def draft_grade(records: Sequence[dict], labels: Sequence[str], target_label: str | None, *,
                higher_order: bool, mechanism_reported_by_finding: bool = False) -> dict:
    """3.3 decision procedure, applied in order, on automatic text flags (a draft for the reviewers)."""
    cons = [c for c in labels if c != target_label]
    fl = [(r, record_flags(r, labels, target_label)) for r in records]
    support = lambda pred: [r.get("doi") or r.get("id") for r, f in fl if pred(f)]
    full_same = support(lambda f: f["full_composition"] and f["spatial_context"] and f["same_disease"])
    full_nonspatial = support(lambda f: f["full_composition"] and f["same_disease"] and not f["spatial_context"])
    indiv = {c: support(lambda f, c=c: c in f["constituent_with_target"]) for c in cons}
    if full_same:
        grade, ids = "no novelty identified", full_same
    elif full_nonspatial:
        grade, ids = "spatial or scale extension", full_nonspatial
    elif cons and all(indiv[c] for c in cons) and higher_order:
        grade, ids = "compositional novelty", sorted({i for v in indiv.values() for i in v})
    elif not any(indiv.values()):
        grade, ids = "new association", []
    else:
        grade, ids = "no grade by rule (partial constituent associations; reviewer decides)", \
            sorted({i for v in indiv.values() for i in v})
    tags = ["mechanistic novelty"] if mechanism_reported_by_finding else []
    return {"draft_grade": grade, "secondary_tags_draft": tags, "supporting_record_ids": ids,
            "constituent_support": indiv, "note": "context extension (full composition in a different "
            "disease) is assessed by the reviewers: the screen keeps only same-disease records"}


def grading_form(finding_id: str, records, labels, target_label, draft: dict) -> dict:
    blank = {"reports_full_composition": None, "same_spatial_context": None, "same_disease": None,
             "constituents_associated_with_target": [], "reports_mechanism": None}
    return {"finding_id": finding_id, "labels": list(labels), "target_label": target_label, "draft": draft,
            "procedure": list(GRADES) + ["+ mechanistic novelty (secondary tag)"],
            "records": [{"id": r.get("doi") or r.get("id"), "title": r.get("title"), "year": r.get("year"),
                         "abstract": r.get("abstract", "")[:3000], "auto_flags": record_flags(r, labels, target_label),
                         "reviewer_A": dict(blank), "reviewer_B": dict(blank)} for r in records],
            "reviewer_A_grade": None, "reviewer_B_grade": None, "resolved_grade": None, "resolution_note": None}


def reviewer_agreement(form: dict) -> dict:
    """Cohen's kappa over per-record boolean judgements of the two reviewers, and grade agreement."""
    a, b = [], []
    for r in form["records"]:
        for k in ("reports_full_composition", "same_spatial_context", "same_disease", "reports_mechanism"):
            x, y = r["reviewer_A"].get(k), r["reviewer_B"].get(k)
            if x is not None and y is not None:
                a.append(bool(x))
                b.append(bool(y))
    kappa = None
    if a:
        import numpy as np

        a, b = np.array(a), np.array(b)
        po = float(np.mean(a == b))
        pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n_judgements": len(a), "kappa": kappa,
            "grade_agreement": (form["reviewer_A_grade"] == form["reviewer_B_grade"])
            if form["reviewer_A_grade"] and form["reviewer_B_grade"] else None}
