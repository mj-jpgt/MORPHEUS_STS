"""Combine two blind graders: per-relation grades, Cohen kappa, calibration accuracy (known relations must grade N0)."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from dbench.downstream import novelty_v10 as nv

CATS = ["N0_known_biology_recovered", "N1_context_extension", "N2_spatial_extension", "N3_compositional_novelty",
        "N4_new_molecular_state_or_association", "N5_new_functional_mechanism"]


def kappa(a, b):
    n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in CATS)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    stage = sys.argv[1]
    d = nv.OUTDIR
    key = json.loads((d / f"bundle_stage{stage}_KEY_not_for_graders.json").read_text())
    A = {g["item_id"]: g for g in json.loads((d / "grades" / f"stage{stage}_graderA.json").read_text())["grades"]}
    B = {g["item_id"]: g for g in json.loads((d / "grades" / f"stage{stage}_graderB.json").read_text())["grades"]}
    Cp = d / "grades" / f"stage{stage}_graderC.json"
    C = {g["item_id"]: g for g in json.loads(Cp.read_text())["grades"]} if Cp.exists() else None
    regraded = {}
    for nm, D in (("A", A), ("B", B)):
        rp = d / "grades" / f"stage{stage}_regrade_grader{nm}.json"
        if rp.exists():
            for g in json.loads(rp.read_text())["grades"]:
                regraded.setdefault(g["item_id"], {})[nm] = {"original_category": D[g["item_id"]]["category"]}
                g["regraded_after_claim_specific_rerun"] = True
                g["original_category"] = D[g["item_id"]]["category"]
                D[g["item_id"]] = g
    ids = sorted(key)
    ca = [A[i]["category"] for i in ids]
    cb = [B[i]["category"] for i in ids]
    rel_ids = [i for i in ids if key[i]["kind"] == "relation"]
    cal_ids = [i for i in ids if key[i]["kind"] == "calibration"]
    fk = nv.false_known_rates()["statistical_checker_v2.1_null_eval"]
    def resolve(cats):
        if len(cats) == 3:
            c = Counter(cats).most_common(1)[0]
            if c[1] >= 2:
                # N4 is provisional: never the resolved category when any grader gave a lower category
                if c[0].startswith("N4") and min(cats, key=CATS.index) != c[0]:
                    return min(cats, key=CATS.index)
                return c[0]
        return min(cats, key=CATS.index)
    out = {"stage": stage, "n_items": len(ids), "kappa_all_items": kappa(ca, cb),
           "kappa_relations": kappa([A[i]["category"] for i in rel_ids], [B[i]["category"] for i in rel_ids]) if rel_ids else None,
           "exact_agreement_all": float(np.mean([x == y for x, y in zip(ca, cb)])),
           "calibration": {"n": len(cal_ids), "grader_A_N0_accuracy": float(np.mean([A[i]["category"] == CATS[0] for i in cal_ids])),
                           "grader_B_N0_accuracy": float(np.mean([B[i]["category"] == CATS[0] for i in cal_ids])),
                           "items": [{"id": key[i]["relation_id"], "A": A[i]["category"], "B": B[i]["category"]} for i in cal_ids]},
           "false_known_rate_beside_every_grade": {"expression_matched": fk["expression"]["false_known_programme"],
                                                   "moran_matched": fk["moran"]["false_known_programme"]},
           "graders": "A,B" + (",C" if C else ""), "provisional_until_three_graders": C is None and stage == "3",
           "relations": []}
    if C:
        cc = [C[i]["category"] for i in ids]
        out["pairwise"] = {"kappa_AB": kappa(ca, cb), "kappa_AC": kappa(ca, cc), "kappa_BC": kappa(cb, cc),
                           "exact_AB": float(np.mean([x == y for x, y in zip(ca, cb)])),
                           "exact_AC": float(np.mean([x == y for x, y in zip(ca, cc)])),
                           "exact_BC": float(np.mean([x == y for x, y in zip(cb, cc)])),
                           "all_three_agree": float(np.mean([x == y == z for x, y, z in zip(ca, cb, cc)]))}
    for i in rel_ids:
        rid = key[i]["relation_id"]
        a, b = A[i], B[i]
        agree = a["category"] == b["category"]
        dec = nv.OUTDIR / "decode" / f"{rid}.json"
        out["relations"].append({
            "relation_id": rid, "item": i, "grader_A": a, "grader_B": b, "agree": agree,
            "grader_C": C[i] if C else None,
            "resolved_category": resolve([a["category"], b["category"]] + ([C[i]["category"]] if C else [])),
            "resolution_rule": ("majority of three, ties to the lower N" if C else
                                ("agree" if agree else "disagreement resolved to the more conservative (lower N) category")),
            "regraded_after_claim_specific_rerun": regraded.get(i),
            "rests_on_generic_reviews": bool(a.get("insufficient_evidence") or b.get("insufficient_evidence")),
            "claim_specific_rerun": (lambda p: None if not p.exists() else (lambda d: {"specific_record_appears": d["specific_record_appears"], "top10": [(x["relevance_score"], x["title"], x["year"], x["same_sentence_target_constituent_spatial"]) for x in d["top10"]]})(json.loads(p.read_text())))(nv.OUTDIR / "claim_specific" / f"{rid}.json"),
            "provisional_N4": any(g["category"].startswith("N4") for g in (a, b)),
            "needs_adjudication": not agree,
            "checker_tier": json.loads(dec.read_text()).get("checker_tier") if dec.exists() else None,
            "false_known_rate": out["false_known_rate_beside_every_grade"]})
    (d / f"grades_stage{stage}.json").write_text(json.dumps(out, indent=1))
    L = [f"## Stage {stage} novelty grades (two blind graders)", "",
         f"Cohen kappa (all items) {out['kappa_all_items']:.2f}; exact agreement {out['exact_agreement_all']:.2f}; calibration N0 accuracy "
         f"A {out['calibration']['grader_A_N0_accuracy']:.2f}, B {out['calibration']['grader_B_N0_accuracy']:.2f} (n={len(cal_ids)}). "
         f"Checker false-known rate on random sets: {fk['expression']['false_known_programme']:.3f} (expression-matched), "
         f"{fk['moran']['false_known_programme']:.3f} (Moran-matched).", "",
         "| relation | A | B | resolved | agree | novel property (A) | closest prior record (A) |", "|---|---|---|---|---|---|---|"]
    for r in out["relations"]:
        a = r["grader_A"]
        cp = (a.get("closest_prior_record") or {}).get("citation", "")
        L.append(f"| {r['relation_id'].split('-multiscale-20261026-')[-1]} | {a['category'][:2]} | {r['grader_B']['category'][:2]} | {r['resolved_category'][:2]} | "
                 f"{r['agree']} | {a['novel_property']} | {cp[:90]} |")
    L += ["", "| calibration item | A | B |", "|---|---|---|"] + [f"| {c['id']} | {c['A'][:2]} | {c['B'][:2]} |" for c in out["calibration"]["items"]]
    (d / f"grades_stage{stage}.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: out[k] for k in ("kappa_all_items", "exact_agreement_all")}), out["calibration"]["grader_A_N0_accuracy"],
          out["calibration"]["grader_B_N0_accuracy"])
    for r in out["relations"]:
        print(r["relation_id"][-30:], r["grader_A"]["category"][:2], r["grader_B"]["category"][:2], r["grader_A"]["novel_property"], "|", r["grader_B"]["novel_property"])
    for c in out["calibration"]["items"]:
        print(c)


if __name__ == "__main__":
    main()
