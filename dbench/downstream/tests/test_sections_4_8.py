import numpy as np
import pandas as pd
import pytest

from dbench.downstream import bulk, clinical, distill, locks, novelty, therapeutic


def _char(cands, loaded=None):
    rows = [{"candidate": c} for c in cands]
    if loaded:
        rows[0]["loaded_genes"] = loaded
    return {"constituents": {"rows": rows}}


def test_query_generation_families_and_lock(tmp_path):
    labels = novelty.constituent_labels(_char(["gene:CXCL13@v6", "hallmark:HALLMARK_HYPOXIA@v0"]), "own:Glutathione")
    assert labels == ["CXCL13", "hypoxia", "Glutathione"]
    qs = novelty.generate_queries(labels)
    fam = [q["family"] for q in qs]
    assert fam.count("constituent") == 3 and fam.count("pairwise") == 3 and fam.count("spatial_context") == 3
    assert fam[-2:] == ["full_composition", "full_composition_spatial"]
    assert all('"clear cell renal cell carcinoma"' in q["text"] for q in qs)
    with pytest.raises(locks.OrderViolation):
        novelty.write_literature_lock(tmp_path, "F", "h", labels, qs, date="2026-10-21")
    novelty.write_literature_lock(tmp_path, "F", "h", labels, qs, date="2026-10-21", allow_violation=True)
    res = novelty.run_queries(tmp_path, "F", qs, labels, cache_dir=tmp_path / "c", offline=True)
    assert len(res["query_log"]) == 2 * len(qs) and res["records"] == []
    with pytest.raises(locks.LockError):
        novelty.run_queries(tmp_path, "F", qs[:-1], labels, cache_dir=tmp_path / "c", offline=True)


def test_residual_labels_use_loaded_genes():
    lg = {"positive": [("IGKC", 1.0), ("IGHG1", 0.9), ("SPP1", 0.8), ("X", 0.1)], "negative": [("CA9", -1.0), ("NDUFA4L2", -.9), ("VIM", -.5)]}
    assert novelty.constituent_labels(_char(["residual:G18@v0"], lg)) == ["IGKC", "IGHG1", "SPP1", "CA9", "NDUFA4L2", "VIM"]


def _rec(i, text):
    return {"id": f"r{i}", "title": text, "abstract": "", "screen": novelty.screen({"title": text}, ["A1", "B2", "T"])}


def test_grading_decision_procedure_in_order():
    labels, tgt = ["A1", "B2", "T"], "T"
    full_sp = _rec(1, "A1 and B2 with T at the tumour boundary in clear cell renal cell carcinoma")
    full = _rec(2, "A1 B2 T in ccRCC")
    a_only = _rec(3, "A1 associates with T in ccRCC")
    b_only = _rec(4, "B2 associates with T in ccRCC")
    g = lambda recs, ho=True: novelty.draft_grade(recs, labels, tgt, higher_order=ho)["draft_grade"]
    assert g([full_sp, full]) == "no novelty identified"
    assert g([full, a_only]) == "spatial or scale extension"
    assert g([a_only, b_only]) == "compositional novelty"
    assert g([]) == "new association"
    assert g([a_only]).startswith("no grade by rule")
    assert not novelty.screen({"title": "A1 in breast cancer"}, labels)["passes"]
    form = novelty.grading_form("F", [a_only], labels, tgt, novelty.draft_grade([a_only], labels, tgt, higher_order=True))
    form["records"][0]["reviewer_A"].update(same_disease=True, reports_full_composition=False)
    form["records"][0]["reviewer_B"].update(same_disease=True, reports_full_composition=True)
    ag = novelty.reviewer_agreement(form)
    assert ag["n_judgements"] == 2 and ag["grade_agreement"] is None


def test_prioritisation_rule_and_statement():
    fs = [{"id": "a", "external_ci_lower": 0.02, "higher_order": True, "grade": "new association", "has_handle": True, "external_gain": 0.03},
          {"id": "b", "external_ci_lower": 0.005, "higher_order": False, "grade": "no novelty identified", "has_handle": False, "external_gain": 0.5},
          {"id": "c", "external_ci_lower": 0.02, "higher_order": False, "grade": "context extension", "has_handle": False, "external_gain": 0.04},
          {"id": "d", "external_ci_lower": 0.02, "higher_order": False, "grade": "context extension", "has_handle": False, "external_gain": 0.02},
          {"id": "e", "external_ci_lower": 0.0, "higher_order": False, "grade": None, "has_handle": False, "external_gain": 0.9}]
    out = therapeutic.prioritise(fs)
    assert [f["id"] for f in out][:3] == ["a", "c", "d"] and out[0]["score"] == 4
    assert sum(f["selected"] for f in out) == 4 and not out[-1]["selected"]
    s = therapeutic.relevance_statement("a", [{"gene": "X", "context_dependency": True}],
                                        [{"gene": "Y", "actionable_in_principle": True, "clinically_precedented": True,
                                          "precedented_drugs": ["drugZ"]}], {"tumour-rich": 0.7, "immune-rich": 0.1})
    assert "X" in s and "drugZ" in s and "tumour-intrinsic" in s and "No efficacy claim" in s


def test_bulk_residual_removes_covariate():
    rng = np.random.default_rng(0)
    n = 300
    comp = rng.standard_normal(n)
    genes = [f"g{i}" for i in range(20)]
    X = pd.DataFrame(np.outer(comp, np.ones(20)) + 0.5 * rng.standard_normal((n, 20)), columns=genes)
    s, info = bulk.residual_score(X, pd.DataFrame({"comp": comp}), genes, np.ones(20))
    assert abs(np.corrcoef(s, comp)[0, 1]) < 0.05 and info["loading_norm_coverage"] == 1.0
    raw, _ = bulk.set_score(X, genes)
    assert np.corrcoef(raw, comp)[0, 1] > 0.9


def test_cox_known_answer_and_prognostic():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.standard_normal(n)
    t = rng.exponential(1 / np.exp(0.5 * x))
    c = rng.exponential(2, n)
    time, ev = np.minimum(t, c), (t <= c).astype(float)
    r = clinical.cox(time, ev, x[:, None], ["x"])
    assert r["terms"]["x"]["hr"] == pytest.approx(np.exp(0.5), rel=0.08)
    p = clinical.prognostic_test(time, ev, x, {"noise": rng.standard_normal(n)})
    assert p["delta_c"] > 0.05 and p["lr_p"] < 1e-10


def test_ppi_mean_known_answer():
    rng = np.random.default_rng(2)
    N, n = 5000, 300
    y_all = rng.normal(1.0, 1, N + n)
    f_all = y_all + 0.5 + 0.3 * rng.standard_normal(N + n)          # biased predictor
    r = distill.ppi_mean(y_all[:n], f_all[:n], f_all[n:])
    assert r["ppi_ci"][0] < 1.0 < r["ppi_ci"][1] and not (r["naive_ci"][0] < 1.0 < r["naive_ci"][1])
    X = rng.standard_normal(N + n)
    y2 = 2 * X + rng.standard_normal(N + n)
    f2 = 1.5 * X + 0.2
    o = distill.ppi_ols(X[:n], y2[:n], f2[:n], X[n:], f2[n:])
    half = (o["ppi_ci"][1][1] - o["ppi_ci"][1][0]) / 2
    assert abs(o["ppi"][1] - 2.0) < 1.5 * half and not (o["naive_ci"][1][0] < 2.0 < o["naive_ci"][1][1])


def test_slide_eval_separates_site_offset_from_ranking():
    rng = np.random.default_rng(3)
    sites = np.repeat(list("ABCDEF"), 60)
    ref = rng.standard_normal(len(sites))
    off = {s: rng.normal(0, 3) for s in "ABCDEF"}
    pred = ref + np.array([off[s] for s in sites]) + 0.3 * rng.standard_normal(len(sites))
    e = distill.slide_cohort_eval(pred, ref, sites, n_boot=200)
    assert e["within_site_centred_spearman"] > 0.9 > e["pooled_spearman"]
    assert distill.site_of("TCGA-B0-4837") == "B0"


def test_ridge_student_heldout():
    rng = np.random.default_rng(4)
    E = rng.standard_normal((1200, 30))
    g = np.repeat([f"p{i}" for i in range(12)], 100)
    y = E[:, 0] - E[:, 1] + 0.5 * rng.standard_normal(1200)
    out = distill.heldout_spot_eval(distill.RidgeStudent, E, y, g, n_perm=2)
    assert out["tile_spearman"] > 0.8 and out["null_tile_spearman_max"] < 0.2
    assert distill.BASELINE_STATUS["STPath"]["status"] == "unavailable"


def test_spatial_residual_projection_is_chunk_invariant_and_removes_covariate():
    import scipy.sparse as sp

    from dbench.downstream.values import residual_projection

    rng = np.random.default_rng(5)
    n, G = 600, 40
    groups = np.repeat(["a", "b", "c"], 200)
    cov = rng.standard_normal(n)
    prog = rng.standard_normal(n)
    load = rng.standard_normal(G)
    X = np.outer(cov, np.ones(G)) + np.outer(prog, load) + 0.5 * rng.standard_normal((n, G)) + 5
    genes = [f"g{i}" for i in range(G)]
    s1, c1 = residual_projection(sp.csr_matrix(X), genes, cov[:, None], genes, load, groups=groups, chunk=7)
    s2, _ = residual_projection(X, genes, cov[:, None], genes, load, groups=groups, chunk=1000)
    assert np.allclose(s1, s2) and c1 == 1.0
    assert abs(np.corrcoef(s1, cov)[0, 1]) < 0.1 and np.corrcoef(s1, prog)[0, 1] > 0.9


def test_between_patient_share_and_preparation_split():
    g = np.repeat(["a", "b", "c", "d"], 50)
    within = np.tile(np.linspace(-1, 1, 50), 4)
    assert distill.between_patient_share(within, g) == pytest.approx(0.0, abs=1e-12)
    shifted = within + np.repeat([0, 10, 20, 30], 50)
    assert distill.between_patient_share(shifted, g) > 0.99
    ref = pd.Series(np.arange(10.0), index=[f"p{i}" for i in range(10)])
    pred = {f"p{i}": float(i if i < 5 else -i) for i in range(10)}
    prep = {f"p{i}": ("A" if i < 5 else "B") for i in range(10)}
    sp = distill.preparation_split(pred, ref, prep)
    assert sp["A"]["spearman"] == pytest.approx(1.0) and sp["B"]["spearman"] == pytest.approx(-1.0)


def test_card_accepts_declared_7x_fields_only(tmp_path):
    for st in ("registry_freeze", "internal_validation", "external_replication", "characterisation", "literature",
               "prioritisation"):
        locks.write_lock(tmp_path, "F", st, inputs={}, declared_analysis={})
        locks.write_record(tmp_path, "F", st, {})
    locks.write_lock(tmp_path, "F", "distillation", inputs={}, declared_analysis={})
    card = locks.StateCard(tmp_path, "F")
    base = {"per_model": {}, "success": None, "site_decomposition": None}
    card.update("distillation", {**base, "diagnosis_7b": {}, "bulk_anchored_7c": {}})
    with pytest.raises(ValueError):
        card.update("distillation", {**base, "diagnosis_7x": {}})


def test_state_definitions_gate_7a_and_clinical(tmp_path):
    for st in ("registry_freeze", "internal_validation", "external_replication", "characterisation", "literature",
               "prioritisation"):
        locks.write_lock(tmp_path, "F", st, inputs={}, declared_analysis={})
        locks.write_record(tmp_path, "F", st, {})
    for st in ("distillation", "clinical"):
        locks.write_lock(tmp_path, "F", st, inputs={}, declared_analysis={})
    card = locks.StateCard(tmp_path, "F")
    card.set_state_definitions({"definition": "a", "hash": "ha"}, {"definition": "b", "hash": "hb"})
    with pytest.raises(locks.LockError):
        card.set_state_definitions({"definition": "a", "hash": "x"}, {"definition": "b", "hash": "y"})
    d = {"per_model": {}, "success": None, "site_decomposition": None}
    with pytest.raises(ValueError):
        card.update("distillation", {**d, "state_definition_hash": "ha"})
    card.update("distillation", {**d, "state_definition_hash": "hb"})
    c = {k: None for k in locks.CARD_SECTIONS["clinical"]}
    with pytest.raises(ValueError):
        card.update("clinical", c)
    card.update("clinical", {**c, "state_definition_hash": "hb"})


def test_card_complete_without_pathologist_read(tmp_path):
    assert "pathologist_read" not in locks.CARD_SECTIONS["characterization"]
    doc = {"sections": {s: {k: None for k in req} for s, req in locks.CARD_SECTIONS.items() if s != "provenance"}}
    assert locks.validate_card(doc)["complete"]
    del doc["sections"]["novelty"]["grade"]
    v = locks.validate_card(doc)
    assert not v["complete"] and v["novelty"]["missing_fields"] == ["grade"]
