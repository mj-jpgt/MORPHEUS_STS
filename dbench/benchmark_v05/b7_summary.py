"""B7 (spatial-state prediction from H&E) and MC6 (panel efficiency), consolidated from the translation_v10 assay layer (W2; read only)."""
import collections
import json
import statistics as st
from pathlib import Path

rows = json.load(open(Path.home() / "morpheus_work_tx/runs/development/translation_v10/layer_assay.json"))
g = collections.defaultdict(list)
pn = collections.defaultdict(list)
for r in rows:
    t = str(r.get("test", ""))
    if not isinstance(r.get("estimate"), (int, float)):
        continue
    if "student" in t and "spearman" in t:
        g[(r.get("assay"), r.get("cohort"))].append((r["relation_id"], float(r["estimate"]), r.get("status")))
    if t.startswith("panel") and "inclusive" not in t:
        pn[t].append(float(r["estimate"]))
b7 = ["| student | cohort | relations | median Spearman | fraction >= 0.4 | max | status pass |", "|---|---|---|---|---|---|---|"]
for (a, c), v in sorted(g.items(), key=lambda x: str(x)):
    e = [x[1] for x in v]
    b7.append(f"| {a} | {c} | {len(v)} | {st.median(e):.3f} | {sum(x >= 0.4 for x in e) / len(e):.2f} | {max(e):.3f} | {sum(x[2] == 'pass' for x in v)} |")
mc6 = ["| panel size (genes) | test | relations | median Spearman | fraction >= 0.8 |", "|---|---|---|---|---|"]
for t, e in sorted(pn.items()):
    mc6.append(f"| {t.split('_')[0]} | {t} | {len(e)} | {st.median(e):.3f} | {sum(x >= 0.8 for x in e) / len(e):.2f} |")
text = ("### B7 spatial-state prediction from H&E (frozen H-Optimus-0 features; ridge and MLP students): per-spot Spearman between predicted and measured "
        "relation state, relations of the adopted method (support-fit ordering + learned beam)\n\n" + "\n".join(b7) +
        "\n\nSource: runs/development/translation_v10/layer_assay.json (W2). Positive control: TGF-beta relation passes in the same layer.\n\n"
        "### MC6 panel efficiency (adopted method only; competing methods were not run through this layer): reduced-panel fidelity on external spots\n\n" + "\n".join(mc6) + "\n")
Path.home().joinpath("morpheus_work_bench/runs/benchmark_v05/b7_he_student.md").write_text(text)
print(text)
