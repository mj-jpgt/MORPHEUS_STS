"""Relation-level novelty re-grade, retrieval stage (amendment v08b; protocol dbench/protocols/relation_novelty_v08b.json,
canonical hash d0fc44d2...). Builds the frozen queries for every relation, retrieves Europe PMC and OpenAlex records
(dbench.downstream.novelty.fetch, cached), screens them by the frozen co-mention rule and extracts, for every read
record, the sentences in which the target and a constituent co-occur (with a relational-cue flag) so the grader can
judge each record against the prior-art rule. The grade itself is assigned by the grader from the logged records.

    python -m dbench.downstream.scripts.relation_regrade RELATIONS_JSON OUT_DIR [--no-openalex]
RELATIONS_JSON: [{"id", "study", "target", "variables", "higher_order", "organs": [...], "labels": {var: label}}]
"""
import json
import re
import sys
import time
from pathlib import Path

PROTO = "dbench/protocols/relation_novelty_v08b.json"
CACHE = "runs/development/relation_regrade/lit_cache"
HALLMARK = {
    "EPITHELIAL_MESENCHYMAL_TRANSITION": '("epithelial-mesenchymal transition" OR "EMT")',
    "TNFA_SIGNALING_VIA_NFKB": '("TNF-alpha" OR "NF-kB" OR "NF-kappaB")',
    "OXIDATIVE_PHOSPHORYLATION": '("oxidative phosphorylation" OR "OXPHOS")',
    "P53_PATHWAY": '("p53")', "MYC_TARGETS_V1": '("MYC")', "MYC_TARGETS_V2": '("MYC")', "E2F_TARGETS": '("E2F")',
    "G2M_CHECKPOINT": '("G2/M" OR "proliferation")', "IL6_JAK_STAT3_SIGNALING": '("IL-6" OR "STAT3")',
    "FATTY_ACID_METABOLISM": '("fatty acid metabolism" OR "fatty acid oxidation")', "ESTROGEN_RESPONSE_LATE": '("estrogen")',
    "ESTROGEN_RESPONSE_EARLY": '("estrogen")', "MTORC1_SIGNALING": '("mTORC1" OR "mTOR")',
    "PANCREAS_BETA_CELLS": '("beta cell" OR "beta-cell")', "INTERFERON_GAMMA_RESPONSE": '("interferon gamma" OR "IFN-gamma")',
    "INTERFERON_ALPHA_RESPONSE": '("interferon alpha" OR "type I interferon")', "ALLOGRAFT_REJECTION": '("allograft rejection")',
    "COMPLEMENT": '("complement")', "ANGIOGENESIS": '("angiogenesis")', "APOPTOSIS": '("apoptosis")',
    "ADIPOGENESIS": '("adipogenesis")', "GLYCOLYSIS": '("glycolysis")', "HYPOXIA": '("hypoxia" OR "HIF")',
    "XENOBIOTIC_METABOLISM": '("xenobiotic metabolism")', "KRAS_SIGNALING_UP": '("KRAS")',
    "INFLAMMATORY_RESPONSE": '("inflammatory response" OR "inflammation")', "UV_RESPONSE_UP": '("UV response")',
    "UNFOLDED_PROTEIN_RESPONSE": '("unfolded protein response" OR "ER stress")', "COAGULATION": '("coagulation")',
    "MYOGENESIS": '("myogenesis")', "KERATINOCYTE": '("keratinocyte")', "HEME_METABOLISM": '("heme metabolism")',
    "CHOLESTEROL_HOMEOSTASIS": '("cholesterol")', "PEROXISOME": '("peroxisome")', "BILE_ACID_METABOLISM": '("bile acid")',
    "DNA_REPAIR": '("DNA repair")', "MITOTIC_SPINDLE": '("mitotic spindle")', "TGF_BETA_SIGNALING": '("TGF-beta" OR "TGF-β")',
    "WNT_BETA_CATENIN_SIGNALING": '("Wnt" OR "beta-catenin")', "NOTCH_SIGNALING": '("Notch")', "HEDGEHOG_SIGNALING": '("Hedgehog")',
    "PI3K_AKT_MTOR_SIGNALING": '("PI3K" OR "AKT")', "REACTIVE_OXYGEN_SPECIES_PATHWAY": '("reactive oxygen species" OR "ROS")',
    "APICAL_JUNCTION": '("apical junction")', "APICAL_SURFACE": '("apical surface")', "SPERMATOGENESIS": '("spermatogenesis")',
    "ANDROGEN_RESPONSE": '("androgen")', "IL2_STAT5_SIGNALING": '("IL-2" OR "STAT5")', "UV_RESPONSE_DN": '("UV response")',
    "KRAS_SIGNALING_DN": '("KRAS")', "PROTEIN_SECRETION": '("protein secretion")',
}
CUES = re.compile(r"associat|correlat|regulat|induc|promot|depend|co-?express|co-?locali|colocali|drive|mediat|link|"
                  r"predict|interact|recruit|activat|suppress|inhibit|upregulat|downregulat|enrich|adjacent|proximity|"
                  r"neighbo|niche|interface|boundary|surround|accompan|coupl|coincid", re.I)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_var(v):
    head, _, rest = v.partition(":")
    if head in ("own", "nbr", "nbr6", "nbr18", "contrast"):
        return head, rest
    return "own", v


def label_of(body, P, given=None):
    """(query OR-group, list of screen terms, kind) for one variable body or target."""
    if given:
        terms = [t.strip().strip('"') for t in re.split(r"\s+OR\s+", given.strip("()"))]
        return given if given.startswith("(") else f'"{given}"', terms, "given"
    if body.startswith("raw_genes:") or body.startswith("gene:"):
        g = body.split(":")[-1]
        if ALIASES:
            terms = hgnc_terms(g)
            return "(" + " OR ".join(f'"{t}"' for t in terms) + ")", terms, "gene"
        return f'"{g}"', [g], "gene"
    if body.startswith("programmes:") or body.startswith("programme:"):
        nm = body.split(":", 1)[1]
        if nm.startswith("hallmark:"):
            q = HALLMARK.get(nm.split("HALLMARK_", 1)[-1], '"' + nm.split("HALLMARK_", 1)[-1].replace("_", " ").lower() + '"')
        else:
            q = P["programme_query_terms"].get(nm, '"' + nm.split(":", 1)[-1] + '"')
        terms = [t.strip().strip('"') for t in re.split(r"\s+OR\s+", q.strip("()"))]
        return q, terms, "programme"
    raise KeyError(f"no frozen label for {body}: supply labels[variable] (morphology/composition/other)")


ALIASES = "--aliases" in sys.argv
_HGNC = {}


def hgnc_terms(g):
    """Declared sensitivity variant (not the frozen primary rule): symbol + HGNC alias symbols + approved name +
    alphanumeric protein tokens of the approved name (e.g. 'C1q' from 'complement C1q B chain')."""
    if g in _HGNC:
        return _HGNC[g]
    import urllib.request
    terms = [g]
    try:
        req = urllib.request.Request(f"https://rest.genenames.org/fetch/symbol/{g}", headers={"Accept": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=20).read())["response"]["docs"]
        if d:
            d = d[0]
            terms += [a for a in (d.get("alias_symbol") or []) if len(a) >= 3]
            nm = d.get("name") or ""
            if nm:
                terms.append(nm)
                terms += [t for t in re.findall(r"[A-Za-z0-9-]+", nm) if re.search(r"[A-Za-z]", t) and re.search(r"[0-9]", t)
                          and len(t) >= 3 and t.upper() != g]
    except Exception:
        pass
    _HGNC[g] = list(dict.fromkeys(terms))
    return _HGNC[g]


def has(text, term):
    if re.fullmatch(r"[A-Z0-9-]+", term) and len(term) <= 10:   # gene symbol / acronym: case-sensitive word match
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text) is not None
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])", text.lower()) is not None


def sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def main():
    from dbench.downstream.novelty import fetch

    rels = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    use_oa = "--no-openalex" not in sys.argv
    P = json.load(open(PROTO))
    SP = P["spatial_terms"]
    oa_dead = False
    allres = []
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only-prefix=")]
    for rel in rels:
        if only and not any(rel["id"].startswith(o) for o in only):
            continue
        try:
            label_check = [label_of(split_var(v)[1], P, rel.get("labels", {}).get(v)) for v in rel["variables"] + [rel["target"]]]
        except KeyError as e:
            log(rel["id"], "skipped (label pending):", e)
            continue
        dz = P["disease_terms"][rel["study"]]
        dz_terms = [t.strip().strip('"') for t in re.split(r"\s+OR\s+", dz.strip("()"))]
        lab = rel.get("labels", {})
        t_ver, t_body = split_var(rel["target"])
        tq, tterms, _ = label_of(t_body, P, lab.get(rel["target"]))
        cons = []
        for v in rel["variables"]:
            ver, body = split_var(v)
            q, terms, kind = label_of(body, P, lab.get(v))
            cons.append({"variable": v, "version": ver, "query": q, "terms": terms, "kind": kind})
        # one query term per distinct constituent label (nbr6/nbr18 versions of one variable share a label)
        uq = list(dict.fromkeys(c["query"] for c in cons if c["query"] != tq))
        queries = []
        for c in uq:
            queries += [{"family": "Q1_pair", "c": c, "text": f"{tq} AND {c} AND {dz}"},
                        {"family": "Q2_pair_spatial", "c": c, "text": f"{tq} AND {c} AND {dz} AND {SP}"},
                        {"family": "Q5_pair_any_disease", "c": c,
                         "text": f'{tq} AND {c} AND ("cancer" OR "tumor" OR "tumour" OR "tissue")'}]
        if len(uq) > 1:
            allc = " AND ".join(uq)
            queries += [{"family": "Q3_full", "c": "all", "text": f"{tq} AND {allc} AND {dz}"},
                        {"family": "Q4_full_spatial", "c": "all", "text": f"{tq} AND {allc} AND {SP}"}]
        if len(rel.get("organs") or []) >= 3:
            for c in uq:
                queries.append({"family": "Q6_conserved", "c": c,
                                "text": f'{tq} AND {c} AND ("pan-cancer" OR "across cancers" OR "multiple cancer types")'})
                for org in rel["organs"]:
                    queries.append({"family": "Q6_conserved_organ", "c": c,
                                    "text": f"{tq} AND {c} AND {P['disease_terms'][org]}"})
        qlog, recs = [], {}
        for q in queries:
            for src in (["europepmc"] + (["openalex"] if use_oa and not oa_dead else [])):
                try:
                    res = fetch(q["text"], src, cache_dir=CACHE, offline=False)
                except Exception as e:
                    qlog.append({**q, "source": src, "error": f"{type(e).__name__}: {e}"})
                    if src == "openalex":
                        oa_dead = True
                        log("OpenAlex unavailable from here on:", e)
                    continue
                qlog.append({**q, "source": src, "n_total": res.get("n_total"), "n_retrieved": len(res["records"])})
                for r in res["records"]:
                    key = (r.get("doi") or r.get("id") or r.get("title", "")[:80]).lower()
                    e = recs.setdefault(key, {**{k: r.get(k) for k in ("id", "doi", "title", "year")},
                                              "abstract": r.get("abstract") or "", "families": set(), "sources": set()})
                    e["families"].add(q["family"])
                    e["sources"].add(src)
                    if len(r.get("abstract") or "") > len(e["abstract"]):
                        e["abstract"] = r["abstract"]
        read, unread = [], 0
        for key, r in recs.items():
            txt = f"{r['title'] or ''}. {r['abstract'] or ''}"
            t_in = any(has(txt, t) for t in tterms)
            c_in = sorted({c["query"] for c in cons if any(has(txt, t) for t in c["terms"])} - {tq})
            if not (t_in and c_in):
                unread += 1
                continue
            sents = []
            for s in sentences(txt):
                if any(has(s, t) for t in tterms) and any(has(s, t) for c in cons for t in c["terms"]):
                    sents.append({"text": s[:600], "cue": bool(CUES.search(s))})
            read.append({"key": key, "id": r["id"], "doi": r["doi"], "title": r["title"], "year": r["year"],
                         "families": sorted(r["families"]), "sources": sorted(r["sources"]),
                         "constituents_mentioned": c_in, "same_disease": any(has(txt, t) for t in dz_terms),
                         "spatial_terms": [t for t in re.findall(r'"([^"]+)"', SP) if has(txt, t)],
                         "co_sentences": sents, "any_cue_sentence": any(s["cue"] for s in sents),
                         "abstract": (r["abstract"] or "")[:2500]})
        read.sort(key=lambda x: (not x["any_cue_sentence"], not x["same_disease"], -len(x["constituents_mentioned"])))
        res = {"id": rel["id"], "study": rel["study"], "target": rel["target"], "variables": rel["variables"],
               "target_query": tq, "constituent_queries": uq, "spatial_form": [c["version"] for c in cons],
               "higher_order": rel.get("higher_order"), "organs": rel.get("organs"),
               "n_queries": len(queries), "query_log": qlog, "n_records_retrieved": len(recs),
               "n_not_read_no_comention": unread, "n_read": len(read),
               "n_read_with_cue_sentence": sum(r["any_cue_sentence"] for r in read), "records_read": read}
        allres.append(res)
        (out / f"retrieval_{rel['id'].replace('/', '_')}.json").write_text(json.dumps(res, indent=1, default=str))
        log(rel["id"], "queries", len(queries), "retrieved", len(recs), "read", len(read),
            "cue", res["n_read_with_cue_sentence"])
    (out / "retrieval_index.json").write_text(json.dumps([{k: v for k, v in r.items() if k not in ("query_log", "records_read")}
                                                           for r in allres], indent=1, default=str))


if __name__ == "__main__":
    main()
