"""E1 / E2 on the v05 labels and pools for every study (the v04 e12 code, unchanged scoring; paths and loader redirected).

    python -m dbench.benchmark_v05.e12_v05 score STUDY
    python -m dbench.benchmark_v05.e12_v05 tables STUDY[,STUDY...]
E1: beam K=64 (release checkpoint 13c36faa) vs exhaustive support-fit scoring; E2: pair ranking by joint gain vs G_HO (guarded / unguarded).
"""
import os
import sys
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
from dbench.benchmark_v04 import e12 as E

V5 = Path(L.BENCH) / "runs/benchmark_v05"


def main():
    mode, studies = sys.argv[1], sys.argv[2].split(",")
    if mode == "score":
        from dbench.benchmark_v05.labels_fast import load_study_v05
        E.ROOT = V5
        orig = L.load_test_study
        L.load_test_study = lambda key: load_study_v05(key) if key in L.LEAN_STUDIES else orig(key)
        for s in studies:
            L.diag()
            E.score(s, out=V5 / "scores")
    else:
        os.environ["W3_OURS"] = "beam64"
        import importlib
        from dbench.benchmark_v05 import metrics_v05 as M
        importlib.reload(M)
        out = ["# E1 (beam K=64 vs exhaustive support-fit scoring of the same pool; B_known, level 95; ours column = beam64)\n",
               M.table(studies, ["beam64", "sel_screen", "sel_screen_joint"], "95", "known", name="E1_" + "_".join(studies))]
        os.environ["W3_OURS"] = "sel_screen_joint"
        importlib.reload(M)
        for B in ("known", "decomp"):
            out.append(f"\n# E2: pair rankings, labels B_{B} (decomp labels guarded by G_known > 0); ours column = joint pair gain\n")
            out.append(M.table(studies, ["sel_screen_joint", "ho_pair", "ho_pair_guarded", "sel_screen", "random"], "95", B, name=f"E2_{B}_" + "_".join(studies)))
        text = "\n".join(out)
        (V5 / ("E1E2_" + "_".join(studies) + ".md")).write_text(text)
        print(text)


if __name__ == "__main__":
    main()
