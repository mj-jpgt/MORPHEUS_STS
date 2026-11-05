"""W1c: literature checker v2.1 (synonym expansion, Europe PMC full text) and blind-grader packets for N0-N5 grading.

Pipeline per relation (frozen target + constituents + spatial form):
  1. labels with synonyms: programme lexicon (relation_novelty_v08b.json) and gene aliases (mygene.info);
  2. query families Q1-Q5 of relation_novelty_v08b (pair, pair+spatial, full, full+spatial, pair in any cancer) against
     Europe PMC (core records; full text searched by the service for open-access papers);
  3. screen (title/abstract/full-text sentence mentions the target and >= 1 constituent), full-text sentence
     extraction for the best open-access records, per-record automatic flags;
  4. packet for two blind graders: relation structure, closest records with the sentences, decoded-axis evidence,
     checker false-known rates.  The code never assigns a grade.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUTDIR = Path("runs/development/novelty_v10")
CACHE = OUTDIR / "lit_cache"
V08B = Path("dbench/protocols/relation_novelty_v08b.json")
GRADER_NULL = Path("runs/development/grader_null/grader_v21_meylan.json")
SPATIAL_WORDS = ["neighbourhood", "neighborhood", "adjacent", "neighbouring", "neighboring", "proximity", "niche",
                 "interface", "boundary", "gradient", "spatial", "peri-", "surrounding", "heterogeneity", "co-localis",
                 "co-localiz", "compartment"]
DISEASE = '("clear cell renal cell carcinoma" OR "ccRCC" OR "renal cell carcinoma" OR "kidney cancer")'
ANY_CANCER = '("cancer" OR "tumor" OR "tumour" OR "carcinoma")'
OP_PHRASE = {"ring": "neighbouring spots (ring around the spot)", "het": "local heterogeneity (variance over the 18 nearest spots)",
             "grad": "local spatial gradient magnitude", "nbrS": "stromal neighbours", "nbrE": "epithelial neighbours",
             "ringS": "stromal cells in rings around the spot", "ringE": "epithelial cells in rings around the spot"}
HALLMARK_SYN = {
    "ANGIOGENESIS": ['"angiogenesis"', '"neovascularization"', '"VEGF"'],
    "HYPOXIA": ['"hypoxia"', '"HIF"'],
    "EPITHELIAL_MESENCHYMAL_TRANSITION": ['"epithelial-mesenchymal transition"', '"EMT"'],
    "ALLOGRAFT_REJECTION": ['"allograft rejection"', '"T cell activation"', '"immune activation"'],
    "INTERFERON_GAMMA_RESPONSE": ['"interferon gamma"', '"IFN-gamma"', '"interferon response"'],
    "GLYCOLYSIS": ['"glycolysis"', '"Warburg"'],
    "APICAL_JUNCTION": ['"apical junction"', '"tight junction"', '"cell-cell adhesion"'],
    "FATTY_ACID_METABOLISM": ['"fatty acid metabolism"', '"lipid metabolism"'],
    "OXIDATIVE_PHOSPHORYLATION": ['"oxidative phosphorylation"', '"OXPHOS"'],
    "TNFA_SIGNALING_VIA_NFKB": ['"TNF-alpha"', '"NF-kB"'],
}


def _get(url, timeout=60, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "morpheus-checker/2.1"}),
                                          timeout=timeout).read().decode("utf-8", "replace")
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def cached(key, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if p.exists():
        return json.loads(p.read_text())
    try:
        v = fn()
    except Exception as e:
        v = {"error": f"{type(e).__name__}: {e}"}
    p.write_text(json.dumps(v))
    return v


# ---------------------------------------------------------------- labels
def gene_aliases(sym):
    def f():
        d = json.loads(_get("https://mygene.info/v3/query?q=symbol:%s&species=human&fields=alias,name&size=1" % sym))
        h = (d.get("hits") or [{}])[0]
        al = h.get("alias") or []
        al = [al] if isinstance(al, str) else al
        return {"aliases": [a for a in al if len(a) >= 3][:6], "name": h.get("name")}
    r = cached("mygene:" + sym, f)
    out = [sym] + r.get("aliases", [])
    if r.get("name"):
        out.append(r["name"])
    return out


def term_group(name, lex):
    """Synonym OR-group (list of quoted strings) for one variable/target name."""
    base = name
    m = re.match(r"^(ring[123]|ring[SE][123]|nbr[SE](?:6|18)|het|grad|nbr6|nbr18|contrast|own):(.*)$", base)
    if m:
        base = m.group(2)
    view, _, col = base.partition(":")
    if view == "raw_genes" or view == "gene":
        return [f'"{a}"' for a in dict.fromkeys(gene_aliases(col))]
    col = base.split(":", 1)[1] if ":" in base else base
    if col.startswith("hallmark:"):
        key = col.split("HALLMARK_", 1)[-1]
        return HALLMARK_SYN.get(key, ['"%s"' % key.replace("_", " ").lower()])
    key = col if col.startswith("mp:") else "mp:" + col.replace("mp:", "")
    if key in lex:
        return re.findall(r'"[^"]+"', lex[key]) or [f'"{key[3:]}"']
    return [f'"{col.replace("mp:", "")}"']


def spatial_form(names, target):
    ops = sorted({re.match(r"^(ring[123]|ring[SE][123]|nbr[SE](?:6|18)|het|grad|nbr6|nbr18|contrast)", n).group(1)
                  for n in list(names) + [target] if re.match(r"^(ring[123]|ring[SE][123]|nbr[SE](?:6|18)|het|grad|nbr6|nbr18|contrast):", n)})
    return ops


def relation_labels(names, target):
    lex = json.loads(V08B.read_text())["programme_query_terms"]
    t = term_group(target, lex)
    cons = [{"name": n, "terms": term_group(n, lex), "operator": (re.match(r"^([A-Za-z]+[0-9]*):", n).group(1)
                                                                if re.match(r"^(ring[123]|ring[SE][123]|nbr[SE](?:6|18)|het|grad|nbr6|nbr18|contrast):", n) else "own")}
            for n in names]
    return {"target": {"name": target, "terms": t}, "constituents": cons, "spatial_ops": spatial_form(names, target)}


def _or(terms):
    return "(" + " OR ".join(terms) + ")"


def queries(lab):
    DISEASE = lab.get("disease_query", globals()["DISEASE"])
    T = _or(lab["target"]["terms"])
    sp = "(" + " OR ".join(f'"{w}"' for w in ["neighbourhood", "neighborhood", "adjacent", "niche", "interface", "spatial",
                                              "proximity", "gradient", "heterogeneity"]) + ")"
    qs = []
    for c in lab["constituents"]:
        C = _or(c["terms"])
        qs += [("Q1_pair", f"{T} AND {C} AND {DISEASE}"), ("Q2_pair_spatial", f"{T} AND {C} AND {DISEASE} AND {sp}"),
               ("Q5_pair_any_cancer", f"{T} AND {C} AND {ANY_CANCER}")]
    allc = " AND ".join(_or(c["terms"]) for c in lab["constituents"])
    qs += [("Q3_full", f"{T} AND {allc} AND {DISEASE}"), ("Q4_full_spatial", f"{T} AND {allc} AND {sp}")]
    return list(dict.fromkeys(qs))


# ---------------------------------------------------------------- retrieval
def epmc(q, n=100):
    def f():
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&pageSize=%d&query=%s"
               % (n, urllib.parse.quote(q)))
        d = json.loads(_get(url))
        recs = [{"id": f"{r.get('source')}:{r.get('id')}", "pmcid": r.get("pmcid"), "doi": r.get("doi"),
                 "title": r.get("title") or "", "abstract": r.get("abstractText") or "", "year": r.get("pubYear"),
                 "open_access": r.get("isOpenAccess") == "Y", "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title")}
                for r in ((d.get("resultList") or {}).get("result") or [])]
        return {"hit_count": d.get("hitCount"), "records": recs}
    return cached("epmc:%d:%s" % (n, q), f)


def fulltext_sentences(pmcid, term_sets, max_s=6):
    def f():
        x = _get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML", timeout=90)
        txt = re.sub(r"<[^>]+>", " ", x)
        txt = re.sub(r"\s+", " ", txt)
        return {"sentences": re.split(r"(?<=[.!?])\s+(?=[A-Z])", txt)}
    d = cached("ft:" + pmcid, f)
    sents = d.get("sentences") or []
    out = []
    for s in sents:
        low = s.lower()
        hits = [any(t.strip('"').lower() in low for t in ts) for ts in term_sets]
        if hits[0] and any(hits[1:]):
            out.append({"sentence": s[:500], "spatial": any(w in low for w in SPATIAL_WORDS)})
    out.sort(key=lambda o: (not o["spatial"], len(o["sentence"])))
    return out[:max_s]


def has_any(text, terms):
    low = text.lower()
    return any(t.strip('"').lower() in low for t in terms)


def retrieve(lab, max_ft=8):
    qs = queries(lab)
    T = lab["target"]["terms"]
    allt = [T] + [c["terms"] for c in lab["constituents"]]
    kept, log = {}, []
    for fam, q in qs:
        r = epmc(q)
        if "error" in r:
            log.append({"family": fam, "query": q, "error": r["error"]})
            continue
        log.append({"family": fam, "query": q, "hit_count": r["hit_count"], "n": len(r["records"])})
        for rank, rec in enumerate(r["records"]):
            txt = f"{rec['title']} {rec['abstract']}"
            tm = has_any(txt, T)
            cm = [c["name"] for c in lab["constituents"] if has_any(txt, c["terms"])]
            e = kept.setdefault(rec["id"], {**rec, "families": [], "best_rank": rank, "target_in_abstract": tm,
                                            "constituents_in_abstract": cm})
            e["families"].append(fam)
            e["best_rank"] = min(e["best_rank"], rank)
    recs = list(kept.values())
    # score: families hit (full > pair), abstract mentions; full-text sentences for the best open-access records
    def score(r):
        fam = set(r["families"])
        junk = re.search(r"congress|annual meeting|abstracts?|proceedings|poster|symposium", r["title"], re.I) is not None
        both = r["target_in_abstract"] and len(r["constituents_in_abstract"]) >= 1
        return (("Q3_full" in fam) * 4 + ("Q4_full_spatial" in fam) * 4 + len(fam) + len(r["constituents_in_abstract"])
                + (1 if r["target_in_abstract"] else 0) + 3 * both - 6 * junk - 0.001 * r["best_rank"])
    recs.sort(key=score, reverse=True)
    n_ft = 0
    for r in recs[:40]:
        if r["open_access"] and r.get("pmcid") and n_ft < max_ft:
            r["fulltext_cooccurrence"] = fulltext_sentences(r["pmcid"], allt)
            n_ft += 1
        r["score"] = score(r)
    top = recs[:12]
    for r in top:
        r["abstract"] = r["abstract"][:1500]
    return {"queries": log, "n_records_screened": len(recs), "top_records": top}


# ---------------------------------------------------------------- packets
def false_known_rates():
    d = json.loads(GRADER_NULL.read_text())
    return {"statistical_checker_v2.1_null_eval": d.get("null_eval"), "thresholds": {"T_prog": d["T_prog"], "T_comp": d["T_comp"]},
            "note": "false-known rate on random gene sets: 0.5 % (expression-matched) to 6.2 % (Moran-matched)"}


def packet(rid, names, target, family, numeric=None, decode=None):
    lab = relation_labels(names, target)
    lit = retrieve(lab)
    return {"relation_id": rid, "family": family, "target": target, "constituents": names, "labels": lab,
            "disease": lab.get("disease_name", "clear cell renal cell carcinoma (Visium)"), "literature": lit,
            "decoded_axis_evidence": decode if decode is not None else "not computed for this item",
            "numeric_evidence": numeric if numeric is not None else "not provided for this item",
            "checker_false_known_rates": false_known_rates()}


CALIBRATION = [
    # id, disease name, disease query, target terms, [(constituent display, terms, operator)], statement
    ("cal_S0_tls_plasma", "clear cell renal cell carcinoma (Visium)", DISEASE, ['"plasma cell"', '"plasma cells"', '"antibody-producing"'],
     [("nbr18:programmes:TLS", ['"tertiary lymphoid structure"', '"tertiary lymphoid structures"', '"TLS"'], "nbr18")]),
    ("cal_H1_tsk_fibroblast", "cutaneous squamous cell carcinoma (Visium)", '("cutaneous squamous cell carcinoma" OR "cSCC" OR "skin squamous cell carcinoma")',
     ['"tumor-specific keratinocyte"', '"TSK"', '"tumour-specific keratinocyte"'],
     [("nbr6:programmes:fibroblast", ['"fibroblast"', '"fibroblasts"', '"stroma"'], "nbr6"),
      ("nbr6:programmes:endothelial", ['"endothelial"', '"endothelial cells"', '"vasculature"'], "nbr6")]),
    ("cal_H2_spp1_myofibroblast", "colorectal cancer (Visium)", '("colorectal cancer" OR "colorectal carcinoma" OR "colon cancer")',
     ['"SPP1"', '"SPP1+ macrophage"', '"SPP1-positive macrophages"'],
     [("programmes:myofibroblast", ['"myofibroblast"', '"myofibroblasts"', '"fibrotic stroma"'], "own")]),
    ("cal_H3_pmnmdsc_club", "prostate cancer (Visium)", '("prostate cancer" OR "prostate adenocarcinoma")',
     ['"PMN-MDSC"', '"myeloid-derived suppressor cell"', '"MDSC"'],
     [("nbr6:programmes:club", ['"club-like cell"', '"club cell"', '"club-like cells"'], "nbr6")]),
    ("cal_H3b_emt_il1bmac", "clear cell renal cell carcinoma (Visium)", DISEASE,
     ['"epithelial-mesenchymal transition"', '"EMT"'],
     [("programmes:il1b_macrophage", ['"IL1B"', '"IL-1beta"', '"IL1B+ macrophage"', '"tissue-resident macrophage"'], "own")]),
    ("cal_C1_hypoxia", "clear cell renal cell carcinoma (Visium)", DISEASE, ['"hypoxia"', '"HIF"'],
     [("programmes:hallmark:HALLMARK_HYPOXIA", ['"hypoxia"', '"HIF"', '"hypoxia-inducible factor"'], "own")]),
    ("cal_C2_emt", "clear cell renal cell carcinoma (Visium)", DISEASE, ['"epithelial-mesenchymal transition"', '"EMT"'],
     [("programmes:hallmark:HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", ['"epithelial-mesenchymal transition"', '"EMT"'], "own")]),
    ("cal_C3_ifn", "clear cell renal cell carcinoma (Visium)", DISEASE, ['"MHC class II"', '"HLA-DR"', '"antigen presentation"', '"interferon"'],
     [("programmes:hallmark:HALLMARK_INTERFERON_GAMMA_RESPONSE", ['"interferon gamma"', '"IFN-gamma"', '"interferon response"'], "own")]),
]


def calibration_packet(item):
    rid, dname, dq, tterms, cons = item
    lab = {"target": {"name": rid, "terms": tterms},
           "constituents": [{"name": n, "terms": t, "operator": op} for n, t, op in cons],
           "spatial_ops": sorted({op for _, _, op in cons if op != "own"}), "disease_query": dq, "disease_name": dname}
    lit = retrieve(lab)
    return {"relation_id": rid, "family": "calibration", "target": "target concept: " + " / ".join(tterms[:2]),
            "constituents": [n for n, _, _ in cons], "labels": lab, "disease": dname, "literature": lit,
            "decoded_axis_evidence": "not computed for this item", "numeric_evidence": "not provided for this item",
            "checker_false_known_rates": false_known_rates()}


# ---------------------------------------------------------------- claim-specific rerun for insufficient-evidence items
SPATIAL_Q = '("spatial" OR "heterogeneity" OR "gradient" OR "neighbourhood" OR "neighborhood")'


def claim_specific(rid, names, target):
    lab = relation_labels(names, target)
    T = lab["target"]["terms"]
    out = {"relation_id": rid, "queries": [], "records": {}}
    for organ_tag, organ in (("ccRCC_kidney", '("clear cell renal cell carcinoma" OR "ccRCC" OR "kidney")'), ("no_organ", None)):
        for c in lab["constituents"]:
            q = f"{_or(T)} AND {_or(c['terms'])} AND {SPATIAL_Q}" + (f" AND {organ}" if organ else "")
            r = epmc(q)
            out["queries"].append({"organ": organ_tag, "constituent": c["name"], "query": q,
                                   "hit_count": r.get("hit_count"), "error": r.get("error")})
            for rank, rec in enumerate(r.get("records", [])):
                txt = f"{rec['title']}. {rec['abstract']}"
                sents = re.split(r"(?<=[.!?])\s+", txt)
                best, sp = 0, False
                for s_ in sents:
                    t_ = has_any(s_, T)
                    c_ = has_any(s_, c["terms"])
                    w = any(w_ in s_.lower() for w_ in SPATIAL_WORDS)
                    sc = (2 if t_ and c_ else 0) + (1 if (t_ and c_ and w) else 0) + (0.3 if t_ or c_ else 0)
                    if sc > best:
                        best, sp = sc, (t_ and c_ and w)
                e = out["records"].setdefault(rec["id"], {"id": rec["id"], "doi": rec["doi"], "title": rec["title"], "year": rec["year"],
                                                          "pmcid": rec.get("pmcid"), "open_access": rec["open_access"],
                                                          "relevance_score": 0.0, "same_sentence_target_constituent_spatial": False,
                                                          "constituents_hit": [], "organ_variants": []})
                if best > e["relevance_score"]:
                    e["relevance_score"] = best
                if best >= 2:
                    e["constituents_hit"].append(c["name"])
                e["same_sentence_target_constituent_spatial"] |= bool(sp)
                e["organ_variants"].append(organ_tag)
    recs = sorted(out["records"].values(), key=lambda r: (-r["relevance_score"], -len(set(r["constituents_hit"]))))
    top = recs[:10]
    for r in top:
        r["constituents_hit"] = sorted(set(r["constituents_hit"]))
        r["organ_variants"] = sorted(set(r["organ_variants"]))
    out["top10"] = top
    out["n_records"] = len(recs)
    out["specific_record_appears"] = any(r["same_sentence_target_constituent_spatial"] and r["relevance_score"] >= 3 for r in top)
    del out["records"]
    return out
