"""Blind choice of the four tumour-elevated Gavish meta-programmes for the invisible_field family (target choice only).
Rule (as the v02/v03 protocols declare): mean within-patient-standardised programme score on neoplastic-dominant spots
(CellViT epithelial_dominant row label) minus on normal_appearing spots, averaged over patients that have both sets,
all sections, response-free; the four mp: programmes with the largest difference.
Usage: PYTHONPATH=<worktree> python tumour_elevated_targets.py <study> <out.json>"""
import json
import sys

import numpy as np

from morpheus.tier2.substrates import hest_multi as hm

study, out = sys.argv[1], sys.argv[2]
spec = hm.study_specs()[study]
raw = hm.load_study(spec)
tab = hm.architecture_table(spec, raw)
tum = np.asarray(tab["epithelial_dominant"]).astype(bool)
nor = np.asarray(tab["normal_appearing"]).astype(bool)
pat = np.asarray(raw.patient).astype(str)
mp = [j for j, n in enumerate(raw.programme_names) if str(n).startswith("mp:")]
diffs = {}
used = {}
for p in np.unique(pat):
    r = pat == p
    if (tum & r).sum() < 20 or (nor & r).sum() < 20:
        continue
    used[p] = {"neoplastic_dominant_spots": int((tum & r).sum()), "normal_appearing_spots": int((nor & r).sum())}
    X = np.asarray(raw.programmes, dtype=float)[r][:, mp]
    Z = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    t, n = tum[r], nor[r]
    for k, j in enumerate(mp):
        diffs.setdefault(raw.programme_names[j], {})[p] = float(Z[t, k].mean() - Z[n, k].mean())
mean = {n: float(np.mean(list(v.values()))) for n, v in diffs.items()}
order = sorted(mean, key=lambda n: -mean[n])
doc = {"schema": "morpheus_tumour_elevated_targets/v1", "study": study,
       "rule": "mean over patients of (mean within-patient-standardised score on epithelial_dominant spots minus on normal_appearing spots); patients with >= 20 spots in both sets; mp: programmes; target choice only",
       "patients_used": used, "partitions_sha256": spec.partitions_sha256, "cache_key": raw.meta.get("cache_key"),
       "targets": [{"rank": i + 1, "programme": n, "loader_target": f"programme:{n}", "mean_difference": round(mean[n], 4),
                    "per_patient": {p: round(v, 4) for p, v in diffs[n].items()}} for i, n in enumerate(order[:4])],
       "all_programmes_mean_difference": {n: round(mean[n], 4) for n in order}}
json.dump(doc, open(out, "w"), indent=1)
print(study, len(used), "patients", [t["programme"] for t in doc["targets"]])
