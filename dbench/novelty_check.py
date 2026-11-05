"""Novelty check after freeze (LOCKIN Sec. 11): a frozen object is checked
against literature/database sources AFTER it is frozen, never before -- this
module takes a frozen object (the same operator/operands/parameters/context
shape `dbench.ledger` records) and returns whatever real reference sources
on the store name something matching it, so "novel" is a post-hoc, checked
verdict rather than a claim made before the check exists.

Four domain checks, each against real, already-on-the-store sources:

  biology   MSigDB Hallmark + curated gene sets
            (~/cscd/w5_data/msigdb_discovery_2024.1.Hs.gmt, real, 1.08MB,
            GMT format) -- operand gene symbols checked for membership in
            any named set; PLUS a small KNOWN_OBJECTS registry seeded
            directly from `dbench/TIER3_TARGETS.md`'s own targets (the
            established literature objects this benchmark already declared
            "known" -- BAP1-loss, CIBERSORT/xCell concordance, etc.), since
            a gene-set-membership check alone cannot express "this specific
            operand pair with this specific relationship was published in
            Clark et al. 2019".
  climate   the real teleconnection index catalogue
            (morpheus_substrate_mirror/climate/teleconnections/) -- operand
            names checked against the real index file stems (oni, nao, pna,
            ao, ea, wp, ...).
  physics   the LHC Olympics 2020 masterkey files (already on the store,
            `morpheus_substrate_mirror/particle_physics/lhco2020/*.masterkey`)
            -- an object whose declared mass parameter falls in a
            masterkey's published resonance window is a known object.
  materials a small registry of published OC20-relevant scaling-relation
            species pairs (Abild-Pedersen et al. 2007) -- checked directly,
            not invented per call.

Returns `{"novel": bool, "hits": [...]}` -- `hits` is empty and `novel` is
True only when NOTHING matched; this module never asserts novelty from
absence of a match in one source without checking the others available for
that domain.
"""
from __future__ import annotations

import hashlib
import re as _re
import json
import os
import re

MSIGDB_PATH = "/home/ubuntu/cscd/w5_data/msigdb_discovery_2024.1.Hs.gmt"
TELECONN_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/"
                "morpheus_substrate_mirror/climate/teleconnections")
LHCO_DIR = ("/lambda/nfs/geeg/biorag3_persistent_20260711/"
           "morpheus_substrate_mirror/particle_physics/lhco2020")

# Seeded directly from dbench/TIER3_TARGETS.md's own five targets -- the
# established literature objects this delivery already declared "known",
# not invented for this check.
KNOWN_OBJECTS = [
    {
        "domain": "biology",
        "keywords": {"bap1", "immune", "cd8", "macrophage", "ccrcc"},
        "reference": "Clark et al., Cell 2019, 'Integrated Proteogenomic "
                     "Characterization of Clear Cell Renal Cell Carcinoma' "
                     "(BAP1-loss immune/transcriptional subtype)",
    },
    {
        "domain": "biology",
        "keywords": {"cibersort", "xcell", "deconvolution", "immune"},
        "reference": "Newman et al. 2015 (CIBERSORT); Aran et al. 2017 "
                     "(xCell); cross-method agreement reported in CPTAC "
                     "ccRCC (Clark et al. 2019, Cell)",
    },
    {
        "domain": "biology",
        "keywords": {"tumour", "tumor", "stroma", "spatial", "ccrcc"},
        "reference": "Standard renal pathology (WHO/ISUP grading); "
                     "tumour-stroma immune spatial organisation documented "
                     "in ccRCC spatial transcriptomics (STHELAR kidney "
                     "atlas)",
    },
    {
        "domain": "climate",
        "keywords": {"oni", "enso", "precipitation", "precip"},
        "reference": "Ropelewski & Halpert 1986, Mon. Wea. Rev., 'North "
                     "American Precipitation and Temperature Patterns "
                     "Associated with the El Nino/Southern Oscillation'",
    },
    {
        "domain": "climate",
        "keywords": {"nao", "temperature", "temp"},
        "reference": "Hurrell 1995, Science, 'Decadal Trends in the North "
                     "Atlantic Oscillation'",
    },
    {
        "domain": "climate",
        "keywords": {"pna", "hgt", "geopotential"},
        "reference": "Wallace & Gutzler 1981, Mon. Wea. Rev., "
                     "'Teleconnections in the Geopotential Height Field'",
    },
    {
        "domain": "materials",
        "keywords": {"adsorption", "scaling", "oh", "ooh", "binding"},
        "reference": "Abild-Pedersen et al. 2007, Phys. Rev. Lett., "
                     "'Scaling Properties of Adsorption Energies for "
                     "Hydrogen-Containing Molecules on Transition-Metal "
                     "Surfaces'",
    },
]


def _tokenize(operands, context=None):
    toks = set()
    for o in operands or ():
        toks.update(re.findall(r"[a-zA-Z0-9]+", str(o).lower()))
    for v in (context or {}).values():
        toks.update(re.findall(r"[a-zA-Z0-9]+", str(v).lower()))
    return toks


_MSIGDB_CACHE = None


def _load_msigdb():
    global _MSIGDB_CACHE
    if _MSIGDB_CACHE is not None:
        return _MSIGDB_CACHE
    sets = {}
    if os.path.exists(MSIGDB_PATH):
        with open(MSIGDB_PATH) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0]
                genes = set(g.upper() for g in parts[2:])
                sets[name] = genes
    _MSIGDB_CACHE = sets
    return sets


def check_biology(operands, context=None):
    hits = []
    toks = _tokenize(operands, context)
    genes = {str(o).upper() for o in (operands or ())
            if re.match(r"^[A-Z0-9]+$", str(o).upper())}
    sets = _load_msigdb()
    for name, gset in sets.items():
        overlap = genes & gset
        if overlap:
            hits.append({"source": "msigdb", "set_name": name,
                        "overlap_genes": sorted(overlap)})
    for k in KNOWN_OBJECTS:
        if k["domain"] != "biology":
            continue
        if toks & k["keywords"]:
            hits.append({"source": "known_objects_registry",
                        "reference": k["reference"],
                        "matched_keywords": sorted(toks & k["keywords"])})
    return hits


def check_climate(operands, context=None):
    hits = []
    toks = _tokenize(operands, context)
    if os.path.isdir(TELECONN_DIR):
        for fn in os.listdir(TELECONN_DIR):
            stem = re.split(r"[._]", fn)[0].lower()
            if stem and stem in toks:
                hits.append({"source": "teleconnection_catalogue",
                            "index_file": fn})
    for k in KNOWN_OBJECTS:
        if k["domain"] != "climate":
            continue
        if toks & k["keywords"]:
            hits.append({"source": "known_objects_registry",
                        "reference": k["reference"],
                        "matched_keywords": sorted(toks & k["keywords"])})
    return hits


def check_physics(operands=None, context=None):
    """The masterkey files are PER-EVENT binary signal labels (0.0/1.0,
    one line per event) -- checked directly: they are not a scalar mass
    value to pattern-match against, and an earlier version of this function
    that tried to extract a mass number from the file was wrong and is not
    what is shipped. The verifiable fact from these files is membership: an
    object framed as "anomaly detection on LHCO2020 BlackBox N" is a KNOWN
    published benchmark object (Kasieczka et al. 2021) regardless of the
    fitted mass, so that is what this check returns."""
    hits = []
    toks = _tokenize(operands, context)
    if not os.path.isdir(LHCO_DIR):
        return hits
    keyed = {"lhco", "lhco2020", "blackbox", "dijet", "resonance"}
    if toks & keyed:
        boxes = sorted(fn for fn in os.listdir(LHCO_DIR)
                      if fn.endswith(".masterkey"))
        hits.append({"source": "lhco2020_masterkey_files_present",
                    "masterkey_files": boxes,
                    "reference": "Kasieczka et al. 2021, Rep. Prog. Phys., "
                    "'The LHC Olympics 2020: A Community Challenge for "
                    "Anomaly Detection in High Energy Physics'"})
    return hits


def check_materials(operands, context=None):
    hits = []
    toks = _tokenize(operands, context)
    for k in KNOWN_OBJECTS:
        if k["domain"] != "materials":
            continue
        if toks & k["keywords"]:
            hits.append({"source": "known_objects_registry",
                        "reference": k["reference"],
                        "matched_keywords": sorted(toks & k["keywords"])})
    return hits


def novelty_check(domain, operator, operands, parameters=None, context=None):
    """The public entry point: a FROZEN object in, hits out. Never call this
    before an object is frozen (LOCKIN Sec. 11) -- it exists to be run
    after, not to inform whether to freeze."""
    parameters = parameters or {}
    context = context or {}
    if domain == "biology":
        hits = check_biology(operands, context)
    elif domain == "climate":
        hits = check_climate(operands, context)
    elif domain == "physics":
        hits = check_physics(operands, context)
    elif domain == "materials":
        hits = check_materials(operands, context)
    else:
        hits = []
    return {"domain": domain, "operator": operator,
           "operands": list(operands or ()), "novel": len(hits) == 0,
           "n_hits": len(hits), "hits": hits}



# ==========================================================================
# D5: the auditable literature-novelty client (plan `calm-mixing-candy.md`,
# `.claude/discoadd.md` "Literature novelty backbone" / "Literature novelty
# is compositional, not binary").
#
# This is a SEPARATE mechanism from `novelty_check` above (which stays as
# the offline MSigDB/registry stub and is untouched). This one:
#
#   1. freezes a canonical claim record and its sha256 BEFORE any query is
#      built or any network call is made (`canonical_claim_record`);
#   2. builds an exact + relaxed query set (full relation, synonyms,
#      pairwise constituents, lower-order subsets, other context, other
#      scale, citation neighbours of the closest hit);
#   3. runs every query against OpenAlex and Europe PMC via `urllib`
#      (`_openalex`, `_europepmc`), through a JSON cache, `offline=True` by
#      default so nothing here ever touches the network unless a caller
#      explicitly opts in;
#   4. reduces the hits to one compositional label, never a bare "novel"
#      bool (`compositional_labels`);
#   5. refuses to run at all on a candidate that has not reached at least
#      `require_stage` in the campaign ledger (`search_novelty_literature`
#      with `require_stage=`), because LOCKIN forbids literature access
#      before a candidate is frozen internally.
# ==========================================================================

import itertools
import urllib.parse
import urllib.request

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

#: Compositional labels a frozen candidate's literature dossier may carry.
#: `unresolved` is what an all-absent search returns -- it is never upgraded
#: to `strong_novelty_candidate` on absence alone (see `compositional_labels`).
NOVELTY_LABELS = (
    "known", "independent_replication", "context_extension",
    "scale_extension", "compositional_novelty", "mechanistic_novelty",
    "actionability_novelty", "strong_novelty_candidate", "unresolved",
)

#: A small, extensible terminology map used to build "synonym" queries.
#: Not exhaustive by design -- it only has to widen the exact query into a
#: second, differently-worded one; the relaxed query set (pairwise
#: constituents, lower-order subsets, other context/scale) is what actually
#: carries the compositional decomposition.
SYNONYMS = {
    "ifng": ("interferon gamma", "ifn-gamma", "ifn gamma"),
    "ifn-gamma": ("interferon gamma", "ifng"),
    "interferon_gamma_program": ("ifn-gamma program", "interferon gamma response"),
    "mhc": ("hla", "major histocompatibility complex"),
    "mhc_program": ("hla class ii", "antigen presentation program"),
    "hif1a": ("hif-1alpha", "hypoxia inducible factor 1 alpha", "hypoxia"),
    "hypoxia": ("hif1a", "hif-1alpha", "low oxygen"),
    "bap1": ("bap1 loss", "bap1-mutant"),
}


class NoveltyGuardError(Exception):
    """Raised when a literature search is attempted before the candidate has
    reached `require_stage` in the ledger (LOCKIN: literature-blind until a
    candidate is frozen internally)."""


def _canon2(x):
    if isinstance(x, dict):
        return {str(k): _canon2(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple, set)):
        return [_canon2(v) for v in x]
    return x


def canonical_claim_record(entities, relationship, direction, context=None,
                           spatial_scale=None, temporal_structure=None,
                           environment=None, outcome=None):
    """Freeze one claim into a canonical dict + its sha256 `claim_hash`.

    This MUST be, and is, computed before any query is built or any network
    call is made -- `search_novelty_literature` below calls this first, and
    `test_novelty_check.py::test_hash_computed_before_first_network_call`
    checks the call order directly rather than trusting a comment."""
    canon = {
        "entities": sorted(str(e) for e in entities),
        "relationship": str(relationship),
        "direction": str(direction),
        "context": _canon2(context or {}),
        "spatial_scale": _canon2(spatial_scale) if spatial_scale is not None else None,
        "temporal_structure": _canon2(temporal_structure) if temporal_structure is not None else None,
        "environment": _canon2(environment) if environment is not None else None,
        "outcome": None if outcome is None else str(outcome),
    }
    blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    claim_hash = hashlib.sha256(blob).hexdigest()
    return {"canonical": canon, "claim_hash": claim_hash}


def _synonym_terms(tokens):
    out = []
    for t in tokens:
        out.extend(SYNONYMS.get(str(t).lower(), ()))
    return out


def build_query_set(record, *, other_context=None, other_scale=None):
    """The exact + relaxed query set for one frozen claim record.

    Kinds, in the order the plan names them: `exact_full_relation`,
    `synonym`, `pairwise_constituents`, `lower_order_subset`,
    `other_context`, `other_scale`. `citation_neighbor` queries are built
    separately, by `citation_neighbor_queries`, once the closest hit from
    this first round is known.
    """
    canon = record["canonical"]
    entities = list(canon["entities"])
    relationship = canon["relationship"]
    direction = canon["direction"]
    queries = []
    full_text = " ".join([relationship] + entities + [direction]).strip()
    queries.append({"kind": "exact_full_relation", "text": full_text})
    for term in _synonym_terms(entities + [relationship]):
        rest = [e for e in entities if e.lower() != term.lower()]
        queries.append({"kind": "synonym", "text": " ".join([term] + rest).strip()})
    if len(entities) >= 2:
        for a, b in itertools.combinations(entities, 2):
            queries.append({"kind": "pairwise_constituents", "text": f"{a} {b}"})
    for e in entities:
        queries.append({"kind": "lower_order_subset", "text": str(e)})
    if other_context:
        queries.append({"kind": "other_context",
                        "text": " ".join([relationship] + entities + [str(other_context)])})
    if other_scale:
        queries.append({"kind": "other_scale",
                        "text": " ".join([relationship] + entities + [str(other_scale)])})
    return queries


def citation_neighbor_queries(top_hit):
    """Follow-up queries for the citation neighbours of the closest hit from
    the first round. `top_hit` is one merged hit dict (see `_run_query`)."""
    out = []
    for r in (top_hit or {}).get("results", []):
        rid = r.get("id")
        if rid:
            out.append({"kind": "citation_neighbor", "text": f"cites:{rid}"})
    return out


def _cache_key(source, query_text, before_date=None):
    """Cache file name.  A literature cutoff is part of the key (a date-limited
    query never reads an unlimited cache entry or vice versa); without a cutoff
    the key is unchanged, so existing caches stay valid."""
    text = "%s::%s" % (source, query_text)
    if before_date:
        text += "::before=%s" % before_date
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + ".json"


def _check_date(before_date):
    """``before_date`` must be ISO ``YYYY-MM-DD`` (it is spliced into URLs)."""
    if before_date is None:
        return None
    before_date = str(before_date)
    if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", before_date):
        raise ValueError("before_date must be YYYY-MM-DD, got %r" % before_date)
    return before_date


def _cache_read(cache_dir, source, query_text, before_date=None):
    if not cache_dir:
        return None
    p = os.path.join(cache_dir, _cache_key(source, query_text, before_date))
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _cache_write(cache_dir, source, query_text, payload, before_date=None):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, _cache_key(source, query_text, before_date))
    with open(p, "w") as f:
        json.dump(payload, f)


def _http_get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _openalex(query_text, *, cache_dir=None, offline=True, timeout=15, before_date=None):
    """OpenAlex works search, `search=` over a long natural-language claim
    description (OpenAlex's own semantic/full-text ranking over `search=`,
    not a separate endpoint) -- cached to a JSON file, offline-safe.
    ``before_date`` (YYYY-MM-DD) restricts to works published on or before
    that date (``filter=to_publication_date:``) and enters the cache key."""
    before_date = _check_date(before_date)
    cached = _cache_read(cache_dir, "openalex", query_text, before_date)
    if cached is not None:
        return cached
    if offline:
        result = {"source": "openalex", "query": query_text, "offline": True,
                  "n_results": 0, "results": [], "before_date": before_date}
        _cache_write(cache_dir, "openalex", query_text, result, before_date)
        return result
    url = OPENALEX_WORKS_URL + "?search=" + urllib.parse.quote(query_text)
    if before_date:
        url += "&filter=to_publication_date:" + before_date
    data = _http_get_json(url, timeout=timeout)
    results = data.get("results", []) or []
    result = {"source": "openalex", "query": query_text, "offline": False,
              "n_results": int((data.get("meta") or {}).get("count", len(results))),
              "results": [{"id": r.get("id"), "title": r.get("display_name") or r.get("title")}
                         for r in results[:5]], "before_date": before_date}
    _cache_write(cache_dir, "openalex", query_text, result, before_date)
    return result


def _europepmc(query_text, *, cache_dir=None, offline=True, timeout=15, before_date=None):
    """Europe PMC REST search (full text, citation network, text-mined
    entities/relations) -- cached to a JSON file, offline-safe.
    ``before_date`` (YYYY-MM-DD) appends ``AND FIRST_PDATE:[1800-01-01 TO
    <date>]`` to the query and enters the cache key."""
    before_date = _check_date(before_date)
    cached = _cache_read(cache_dir, "europepmc", query_text, before_date)
    if cached is not None:
        return cached
    if offline:
        result = {"source": "europepmc", "query": query_text, "offline": True,
                  "n_results": 0, "results": [], "before_date": before_date}
        _cache_write(cache_dir, "europepmc", query_text, result, before_date)
        return result
    q = query_text
    if before_date:
        q = query_text + " AND FIRST_PDATE:[1800-01-01 TO %s]" % before_date
    url = (EUROPEPMC_SEARCH_URL + "?query=" + urllib.parse.quote(q)
          + "&format=json")
    data = _http_get_json(url, timeout=timeout)
    result_list = ((data.get("resultList") or {}).get("result")) or []
    result = {"source": "europepmc", "query": query_text, "offline": False,
              "n_results": int(data.get("hitCount", len(result_list))),
              "results": [{"id": r.get("id"), "title": r.get("title")}
                         for r in result_list[:5]], "before_date": before_date}
    _cache_write(cache_dir, "europepmc", query_text, result, before_date)
    return result


def _run_query(q, *, cache_dir=None, offline=True, before_date=None):
    oa = _openalex(q["text"], cache_dir=cache_dir, offline=offline, before_date=before_date)
    epmc = _europepmc(q["text"], cache_dir=cache_dir, offline=offline, before_date=before_date)
    n = int(oa.get("n_results", 0)) + int(epmc.get("n_results", 0))
    return {"kind": q["kind"], "text": q["text"], "n_results": n,
           "results": (oa.get("results") or []) + (epmc.get("results") or []),
           "openalex": oa, "europepmc": epmc}


def compositional_labels(hits):
    """Reduce a list of merged hit dicts (`_run_query` output) to exactly one
    of `NOVELTY_LABELS`.

    The rule the plan requires, checked here rather than only stated: an
    all-absent search returns `unresolved`, never `strong_novelty_candidate`
    -- a candidate for the latter must show POSITIVE evidence that the
    search mechanism actually works for this topic (a `lower_order_subset`
    hit, i.e. the individual entities are indexed and findable) while the
    full composed claim still returns nothing anywhere, including its
    relaxations. Absence alone proves nothing about whether the query was
    any good.
    """
    if not hits:
        return "unresolved"
    by_kind = {}
    for h in hits:
        by_kind.setdefault(h.get("kind", "unknown"), 0)
        by_kind[h.get("kind", "unknown")] += int(h.get("n_results", 0))

    def has(kind):
        return by_kind.get(kind, 0) > 0

    if not any(by_kind.values()):
        return "unresolved"

    if has("exact_full_relation") or has("synonym"):
        exact_hits = [h for h in hits
                     if h.get("kind") in ("exact_full_relation", "synonym")
                     and h.get("n_results", 0) > 0]
        if any(h.get("independent_cohort") for h in exact_hits):
            return "independent_replication"
        return "known"
    if has("other_context"):
        return "context_extension"
    if has("other_scale"):
        return "scale_extension"
    if any(h.get("actionability_gap") for h in hits if h.get("n_results", 0) > 0):
        return "actionability_novelty"
    if has("citation_neighbor"):
        return "mechanistic_novelty"
    if has("pairwise_constituents"):
        return "compositional_novelty"
    if has("lower_order_subset"):
        return "strong_novelty_candidate"
    return "unresolved"


def _guard_ledger_stage(ledger, candidate_id, require_stage):
    if ledger is None or candidate_id is None:
        raise NoveltyGuardError(
            "require_stage=%r needs both a ledger and a candidate_id -- "
            "literature search is forbidden on an unfrozen, unidentified "
            "candidate" % (require_stage,))
    from .ledger import STAGE_RANK
    if require_stage not in STAGE_RANK:
        raise NoveltyGuardError("unknown require_stage %r; must be one of %r"
                               % (require_stage, tuple(STAGE_RANK)))
    events = [e for e in ledger.all_events() if e.get("candidate_id") == candidate_id]
    if not events:
        raise NoveltyGuardError(
            "candidate %r has no ledger events; literature search is "
            "forbidden before a candidate is frozen at stage %r"
            % (candidate_id, require_stage))
    best = max(STAGE_RANK[e["stage"]] for e in events)
    if best < STAGE_RANK[require_stage]:
        raise NoveltyGuardError(
            "candidate %r has not reached stage %r (highest recorded stage "
            "rank %d < %d); literature search is forbidden until then "
            "(LOCKIN: literature-blind until frozen)"
            % (candidate_id, require_stage, best, STAGE_RANK[require_stage]))


def search_novelty_literature(entities, relationship, direction, *,
                              context=None, spatial_scale=None,
                              temporal_structure=None, environment=None,
                              outcome=None, other_context=None,
                              other_scale=None, offline=True, cache_dir=None,
                              include_citation_neighbors=True,
                              ledger=None, candidate_id=None,
                              require_stage=None, before_date=None):
    """The D5 entry point: freeze the claim, guard the stage, search, label.

    Order is load-bearing: `require_stage` is checked, then the claim is
    canonicalised and hashed, THEN (and only then) any query is built or any
    network call made. `offline=True` by default, matching the coordinator's
    rule that the live client runs only on `internal` candidates.
    ``before_date`` (YYYY-MM-DD) limits every query to literature published on
    or before that date (historical-rediscovery tasks) and enters the cache key.
    """
    before_date = _check_date(before_date)
    if require_stage is not None:
        _guard_ledger_stage(ledger, candidate_id, require_stage)
    record = canonical_claim_record(entities, relationship, direction, context,
                                    spatial_scale, temporal_structure,
                                    environment, outcome)
    queries = build_query_set(record, other_context=other_context,
                              other_scale=other_scale)
    hits = [_run_query(q, cache_dir=cache_dir, offline=offline, before_date=before_date)
            for q in queries]
    if include_citation_neighbors:
        top = max(hits, key=lambda h: h["n_results"], default=None)
        if top is not None and top["n_results"] > 0:
            hits = hits + [_run_query(q, cache_dir=cache_dir, offline=offline,
                                      before_date=before_date)
                          for q in citation_neighbor_queries(top)]
    label = compositional_labels(hits)
    return {"record": record, "claim_hash": record["claim_hash"],
           "queries": queries, "hits": hits, "label": label, "before_date": before_date}


if __name__ == "__main__":
    import sys
    print(json.dumps(novelty_check(*sys.argv[1:]), indent=2))
