"""Fold (split) stability of support-fit ordering from the score receipts: per-target SD of AP over the 10 patient-grouped splits, and the
mean pairwise Jaccard overlap of the top-10 / top-20 picks between splits of the same target (singles).  Random is the reference (chance
Jaccard).  Support-fit ordering is deterministic given the support patients, so seed stability reduces to split stability.

    python -m dbench.benchmark_v05.stability_v05
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from dbench.benchmark_v05 import metrics_v05 as M

V5 = M.V5
STUDIES = ["gse237183_glioma", "nakajima_luad_b3", "gse307534_lung_precursor_b3"]
lines = ["### Fold stability of support-fit ordering (10 patient-grouped splits per target)", "",
         "| study | method | kind | mean within-target SD of AP across splits | between-target SD of AP | top-10 Jaccard between splits | top-20 Jaccard | chance Jaccard top-10 |", "|---|---|---|---|---|---|---|---|"]
for st in STUDIES:
    for meth in ("sel_screen", "sel_screen_joint", "linear_screen", "nid"):
        for kind in ("single", "pair"):
            wsd, means, j10, j20 = [], [], [], []
            n_pool = None
            for t in M.targets_of(st):
                d = M.per_split(st, meth, t, "95", "known")
                if not d or not d[kind]:
                    continue
                aps = [x["ap"] for x in d[kind].values() if np.isfinite(x["ap"])]
                if len(aps) >= 3:
                    wsd.append(np.std(aps, ddof=1))
                    means.append(np.mean(aps))
                h = hashlib.sha256(t.encode()).hexdigest()[:10]
                z = np.load(V5 / "scores" / st / meth / f"{h}.npz")
                keys = sorted(k for k in z.files if k.endswith(f"|{kind}"))
                tops = {}
                for k in keys:
                    sv = np.asarray(z[k], float)
                    n_pool = len(sv)
                    sv = np.where(np.isfinite(sv), sv, sv[np.isfinite(sv)].min() - 1 if np.isfinite(sv).any() else 0)
                    tops[k] = np.argsort(-sv, kind="stable")[:20]
                for a, b in itertools.combinations(keys, 2):
                    for K, store in ((10, j10), (20, j20)):
                        A, B = set(tops[a][:K]), set(tops[b][:K])
                        store.append(len(A & B) / len(A | B))
            if wsd:
                ch = (10 * 10 / n_pool) / (20 - 10 * 10 / n_pool)
                lines.append(f"| {st} | {meth} | {kind} | {np.mean(wsd):.3f} | {np.std(means, ddof=1):.3f} | {np.mean(j10):.3f} | {np.mean(j20):.3f} | {ch:.4f} |")
text = "\n".join(lines) + "\n"
(V5 / "stability_folds.md").write_text(text)
print(text)
