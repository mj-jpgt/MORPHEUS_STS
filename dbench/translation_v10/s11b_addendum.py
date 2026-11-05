import json
from dbench.translation_v10 import common as C

V = json.load(open(f"{C.OUT}/p5_known_valdeolivas2024_crc.json"))
Ep = json.load(open(f"{C.OUT}/p5_known_ependymoma_pfa.json"))
M = json.load(open(f"{C.OUT}/p5_matrix.json"))
cells = [("G3", "valdeolivas2024_crc", V["G3"]), ("G4", "valdeolivas2024_crc", V["G4"]),
         ("heterogeneity_gradient-934c62f7ec26", "valdeolivas2024_crc", V["heterogeneity_gradient-934c62f7ec26"]),
         ("G3", "ependymoma_pfa", Ep["G3"]), ("G4", "ependymoma_pfa", Ep["G4"]),
         ("heterogeneity_gradient-ec7174547e19", "ependymoma_pfa", Ep["heterogeneity_gradient-ec7174547e19"])]
out = {"rule": "frozen f_S (no refit) added to the cross-fitted B_known baseline of the target cohort (log1p depth + own-spot Hallmark and Gavish programme scores, target and candidate own sets removed, within-patient standardised); patient bootstrap CI; Moran-matched null p95 (99 draws); catalogue gene sets of B_known not included (response-free programme columns only)",
       "cells": []}
md = ["# P5 addendum: transfer gain over known-biology baseline B_known", "",
      "| relation | cohort | gain over B_nuis (P5) | gain over B_known [95% CI] | null p95 | transfers over B_known |", "|---|---|---|---|---|---|"]
for n, c, r in cells:
    p5 = M[n]["cells"][c]
    out["cells"].append({"relation": n, "cohort": c, "gain_over_B_nuis_p5": p5.get("gain"), **r})
    nl = r.get("null", {})
    md.append("| %s | %s | %.3f | %.3f [%.3f, %.3f] | %.3f | %s |" % (n, c, p5.get("gain"), r["gain_over_B_known"], r["ci_low"], r["ci_high"], nl.get("null_q95", float("nan")), r.get("transfers_over_B_known")))
json.dump(out, open(f"{C.OUT}/p5_known_addendum.json", "w"), indent=1, default=float)
open(f"{C.OUT}/p5_known_addendum.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
