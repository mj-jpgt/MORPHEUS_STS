"""Seconds per method per task from the score receipts: python -m dbench.benchmark_v05.seconds_table STUDY[,STUDY]"""
import glob, json, sys, collections
from pathlib import Path
V5 = Path.home() / "morpheus_work_bench/runs/benchmark_v05"
studies = sys.argv[1].split(",")
rows = ["| method | " + " | ".join(f"{s}: median s per split (splits ok / timeouts / errors)" for s in studies) + " |", "|---|" + "---|" * len(studies)]
methods = sorted({Path(p).parent.name for s in studies for p in glob.glob(str(V5 / "scores" / s / "*" / "*.json"))})
import statistics as st
for m in methods:
    cells = []
    for s in studies:
        secs, ok, to, er = [], 0, 0, 0
        for f in glob.glob(str(V5 / "scores" / s / m / "*.json")):
            d = json.load(open(f))
            stt = d.get("status", {})
            for k, v in d.get("seconds", {}).items():
                if stt.get(k, "done") == "done":
                    secs.append(float(v)); ok += 1
                elif stt.get(k) == "timeout": to += 1
                elif stt.get(k, "").startswith("error"): er += 1
        cells.append(f"{st.median(secs):.1f} ({ok}/{to}/{er})" if secs else (f"- (0/{to}/{er})"))
    rows.append(f"| {m} | " + " | ".join(cells) + " |")
text = "\n".join(rows) + "\n"
(V5 / "seconds_per_method.md").write_text(text); print(text)
