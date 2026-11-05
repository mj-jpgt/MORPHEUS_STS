"""Operator and morphology leave-out ablation, computed from the registered v08 funnel (read only): the number of internally supported
and beyond-known-biology relations, and of externally replicated relations, that remain when one operator family (ring, heterogeneity /
gradient, compartment context) or all morphology-involving candidates are removed.  A leave-out count, not a refit: it reports how
much of the catalogue each operator contributes.
"""
import json
import re
from pathlib import Path

H = Path.home()
RUN = H / "morpheus_work_v08/runs/registered_v08/renal-spatial-v08-multiscale-20261026"
agg = json.load(open(RUN / "funnel_aggregate_receipt_matrix_repaired.json"))
rows = agg["table"]
rep = []
for p in (H / "morpheus_work_v06/runs/confirmation/v08_downstream_strict_spatial_v08/replication_summary.json",):
    rep += json.load(open(p))["replicated"]
rep_ids = set(rep)


def morph(r):
    names = list(r.get("variable_names", [])) + [str(r.get("target", ""))]
    return any(re.search(r"(^|:)morph:|centre|cellvit|hoptimus", str(n), re.I) for n in names)


fam = sorted({r["family"] for r in rows})
internal = [r for r in rows if r["stage"] == "internal" or r.get("passes_B_nuis")]
known = [r for r in internal if r.get("passes_B_known")]
repl = [r for r in rows if r["candidate_id"] in rep_ids]


def count(sel, drop=None, drop_morph=False):
    return sum(1 for r in sel if (drop is None or r["family"] != drop) and not (drop_morph and morph(r)))


n_morph = sum(1 for r in rows if morph(r))
lines = ["### Operator and morphology leave-out ablation on the registered v08 catalogue (renal, 200 frozen candidates; counts of relations that remain)", "",
         f"HEADER NOTE: {n_morph} of the 200 frozen candidates carry an H-Optimus concept or morphology-dictionary variable (every variable and target is a gene, a programme score, a residual-programme target, or a ring / neighbour / contrast / heterogeneity / gradient transform of those). The morphology row therefore removes nothing and must not be read as morphology contributing nothing: the v08 registered catalogue contains no morphology candidates, so the morphology contribution is not assessed here. compartment_context restricts rows to operator-defined interface spots whose compartment labels come from the deconvolved compartments (registered_deconvolution v3.1); it is counted as a spatial-context operator, not as morphology, and has its own row.", "",
         "| removed | internally supported (B_nuis rule) | beyond known biology (B_known) | externally replicated (W1 summary at read time) |", "|---|---|---|---|",
         f"| none | {count(internal)} | {count(known)} | {count(repl)} |"]
for f in fam:
    lines.append(f"| family {f} | {count(internal, f)} | {count(known, f)} | {count(repl, f)} |")
lines.append(f"| all morphology-involving candidates (0 exist; row removes nothing) | {count(internal, drop_morph=True)} | {count(known, drop_morph=True)} | {count(repl, drop_morph=True)} |")
lines += ["", f"Families: {fam}. Replicated list: {len(rep_ids)} ids ({sum(1 for r in repl)} found in the frozen table). Source: {RUN}/funnel_aggregate_receipt_matrix_repaired.json; "
          "~/morpheus_work_v06/runs/confirmation/v08_downstream_strict_spatial_v08/replication_summary.json (W1 owns both; W1 external validation was still running, so the replicated column is a lower bound)."]
text = "\n".join(lines) + "\n"
(H / "morpheus_work_bench/runs/benchmark_v05/ablation_operators.md").write_text(text)
print(text)
