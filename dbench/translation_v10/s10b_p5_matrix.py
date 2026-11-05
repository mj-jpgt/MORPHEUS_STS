"""S10b: relation x cohort P5 matrix (pooled gain over nuisance baseline with CI, null p95, direction agreement, conserved flag) + earlier conservation comparator."""
import json, os
import numpy as np
from dbench.translation_v10 import common as C

COH = ["valdeolivas2024_crc", "gse294385", "ji2020_cscc", "ependymoma_pfa"]
ORGAN = {"valdeolivas2024_crc": "colorectal", "gse294385": "colorectal", "ji2020_cscc": "skin", "ependymoma_pfa": "brain"}
R = {c: (json.load(open(f"{C.OUT}/p5_{c}.json")) if os.path.exists(f"{C.OUT}/p5_{c}.json") else {}) for c in COH}
for c in COH:
    for f6 in (f"{C.OUT}/p5ctrl6_{c}.json", f"{C.OUT}/p5ctrlhalf_{c}.json", f"{C.OUT}/p5_{c}_D.json"):
        if os.path.exists(f6):
            R[c].update(json.load(open(f6)))
names = list(dict.fromkeys(["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"] + [n for c in COH for n in R[c] if n not in ("CTRL_HYPOXIA", "CTRL_HYPOXIA6", "CTRL_HYPOXIA_HALF") and not n.startswith("G")] + ["CTRL_HYPOXIA_HALF", "CTRL_HYPOXIA6", "CTRL_HYPOXIA"]))
matrix, md = {}, ["# P5 conserved rules: no-refit transfer of kidney relations (G1-G8 best members) and the hypoxia Hallmark control",
                  "", "cell = pooled gain over nuisance baseline [95% CI] | null p95 | direction agreement | transfers (CI>0 AND gain>null p95)", "",
                  "| relation | " + " | ".join(COH) + " | organs passed (non-kidney) | conserved (>=2 non-kidney organs) |", "|---|" + "---|" * (len(COH) + 2)]
for n in names:
    row, cells, passed = {}, [], set()
    for c in COH:
        r = R[c].get(n)
        if r is None:
            row[c] = {"status": "not_assessed", "reason": "no result"}; cells.append("n/a"); continue
        row[c] = r
        if r.get("status") != "ok":
            cells.append("not_assessed: " + r.get("reason", "")[:70]); continue
        nl = r.get("null", {})
        tr = r.get("transfers_v03_1")
        if tr and (r.get("direction_agreement") or 0) >= 0.6:
            passed.add(ORGAN[c])
        cells.append(f"{r['gain']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] | p95 {nl.get('null_q95', float('nan')):.3f} | dir {r['direction_agreement'] if r['direction_agreement'] is None else round(r['direction_agreement'], 2)} | {'YES' if tr else 'no'}")
    flag = len(passed) >= 2
    matrix[n] = {"cells": row, "organs_passed": sorted(passed), "conserved_flag": flag,
                 "flag_rule": "transfers_v03_1 (gain CI excludes 0 AND gain > Moran-matched null p95) with direction agreement >= 0.6 in >= 2 organs other than the source (kidney); with the source organ this is the >= 3 organs tag"}
    md.append(f"| {n} | " + " | ".join(cells) + f" | {', '.join(sorted(passed)) or '-'} | {flag} |")
# comparator: earlier conservation matrix (relations_frozen organs / conserved_8 / five_cohort_4 flags)
rf = json.load(open(f"{C.V06}/runs/development/relation_regrade/relations_frozen.json"))
comp = {x["id"]: {"organs": x["organs"], "conserved_8": x["conserved_8"], "five_cohort_4": x["five_cohort_4"], "phenomenon": x.get("phenomenon")} for x in rf if x.get("conserved_8") or x.get("five_cohort_4")}
matrix["_comparator_earlier_p5"] = comp
md += ["", "## Comparator: earlier P5 conservation matrix (relation_regrade/relations_frozen.json organ lists; legacy conserved_8 / five_cohort_4)", "",
       "| relation | organs recorded | conserved_8 | five_cohort_4 |", "|---|---|---|---|"]
for k, v in comp.items():
    md.append(f"| {k} | {', '.join(v['organs'])} | {v['conserved_8']} | {v['five_cohort_4']} |")
md += ["", "Cells labelled not_assessed name the missing variable (gene absent from the cohort panel). Control target = Hallmark hypoxia genes available in the cohort minus the predictors VEGFA and SLC2A1."]
json.dump(matrix, open(f"{C.OUT}/p5_matrix.json", "w"), indent=1, default=float)
open(f"{C.OUT}/p5_matrix.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
