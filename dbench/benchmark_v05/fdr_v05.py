"""Method characterisation: false discovery rate of the adopted ranking under permuted and scrambled targets.

    python -m dbench.benchmark_v05.fdr_v05 score STUDY [--max-targets N]
    python -m dbench.benchmark_v05.fdr_v05 table STUDY[,STUDY]

For every (target, split) the top-K picks of support-fit ordering (K = 10, 20; singles and pairs; picks made on the real support) are
re-scored on the null query: (a) patient-permuted target (core.null_task kind "patient", draws 0-4, as the label nulls) and (b)
candidate-scrambled candidates (kind "scramble", within-patient row permutation).  A pick passes under a null draw when the patient-bootstrap
lower bound (95 %) of its held-out gain is above 0 and the gain exceeds the real task's null p95 threshold (the label rule).  FDR@K = mean pass
fraction over picks and draws; net precision@K = real precision@K (cached labels) minus FDR@K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
import numpy as np

BENCH = Path(L.BENCH)
V5 = BENCH / "runs/benchmark_v05"
KS = (10, 20)
N_DRAW = 5
import os
DEV = os.environ.get("W3_DEVICE", "cuda")


def score(study, max_targets=40):
    from dbench.benchmark_v04.score_methods import build_pool_and_tasks
    from dbench.benchmark_v05.labels_fast import fit_multi, load_study_v05
    from dbench.benchmark_v05.run_scores import label_dir
    from dbench.harness import core
    from morpheus_encoder.learned_search import relation_v03 as rv
    st = load_study_v05(study)
    ld = label_dir(study)
    targets = json.loads((ld / "targets.json").read_text())["targets"][:max_targets]
    od = V5 / "fdr" / study
    od.mkdir(parents=True, exist_ok=True)
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        sp = V5 / "scores" / study / "sel_screen" / f"{h}.npz"
        if (od / f"{h}.npz").exists() or not sp.exists():
            continue
        sc = np.load(sp)
        z = np.load(ld / "cache" / f"{h}.npz")
        built, singles, pairs = build_pool_and_tasks(st, t)
        out = {}
        for r, support, query, task, names, ep in built:
            idx = {n: i for i, n in enumerate(names)}
            s_sets_all = [(idx[n],) for n in singles]
            p_sets_all = [(idx[singles[a]], idx[singles[b]]) for a, b in pairs]
            s1, s2 = np.asarray(sc[f"s{r}|single"], float), np.asarray(sc[f"s{r}|pair"], float)
            nulls = {"patient": [core.null_task(task, ep, "patient", L.stable_seed("bench-null", study, t, r, d)) for d in range(N_DRAW)],
                     "scramble": [core.null_task(task, ep, "scramble", L.stable_seed("fdr-scramble", study, t, r, d)) for d in range(N_DRAW)]}
            for kind, sets_all, sc_ in (("single", s_sets_all, s1), ("pair", p_sets_all, s2)):
                top = np.argsort(-np.where(np.isfinite(sc_), sc_, -1e30), kind="stable")[: max(KS)]
                sets = [sets_all[i] for i in top]
                thr = float(z[f"s{r}|{kind}|known|null_p95"])
                arr = np.asarray(sets, dtype=np.int64)
                Vs, Vq = rv.set_values(L.known_variant(task, ()), arr)
                s = task.s
                for nk, tasks in nulls.items():
                    base = [rv.baseline_for(L.known_variant(t_, ()), "known") for t_ in tasks]
                    jobs = [(base[i], t_.y[:s], t_.y[s:]) for i, t_ in enumerate(tasks)]
                    if nk == "patient":    # same candidate values, permuted target: one factorisation, five right-hand sides
                        fits = fit_multi(Vs, Vq, task.groups[:s], task.groups[s:], jobs, device=DEV, chunk=64)
                    else:                  # scramble changes the candidate values: one fit per draw on that draw's own X
                        fits = []
                        for i, t_ in enumerate(tasks):
                            Vs_i, Vq_i = rv.set_values(L.known_variant(t_, ()), arr)
                            fits += fit_multi(Vs_i, Vq_i, task.groups[:s], task.groups[s:], [jobs[i]], device=DEV, chunk=64)
                    passes = []
                    for i, f in enumerate(fits):
                        G, lbs = L.pooled(f["gain_qp"], f["sst_qp"], seed=L.stable_seed("fdr", study, t, r, kind, nk, i))
                        passes.append((lbs["95"] > 0) & (G > thr))
                    out[f"s{r}|{kind}|{nk}"] = np.stack(passes).astype(np.int8)   # (draws, picks)
                out[f"s{r}|{kind}|picks"] = top.astype(np.int32)
        np.savez_compressed(od / f"{h}.npz", **out)
        print(study, t, flush=True)


def table(studies, B="known", level="95"):
    from dbench.benchmark_v04 import tables as T
    from dbench.benchmark_v05.metrics_v05 import label_dir
    lines = ["### FDR of support-fit ordering under null targets (picks made on the real support; label rule applied to the null query)", "",
             "| study | kind | K | real precision@K | FDR, patient-permuted target | FDR, candidate-scrambled | net precision (permuted) | net precision (scrambled) | targets | pooled picks |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for study in studies:
        ld = label_dir(study)
        for kind in ("single", "pair"):
            for K in KS:
                real, fp, fs, nt = [], [], [], 0
                for t in json.loads((ld / "targets.json").read_text())["targets"]:
                    h = hashlib.sha256(t.encode()).hexdigest()[:10]
                    fp_ = V5 / "fdr" / study / f"{h}.npz"
                    if not fp_.exists():
                        continue
                    f = np.load(fp_)
                    z = np.load(ld / "cache" / f"{h}.npz")
                    meta = json.loads((ld / "cache" / f"{h}.json").read_text())
                    r_, p_, s_ = [], [], []
                    for s in T.splits_of(meta):
                        if f"s{s}|{kind}|picks" not in f.files:
                            continue
                        y = T.label(z, s, kind, B, level)
                        pk = f[f"s{s}|{kind}|picks"][:K]
                        r_.append(y[pk].mean())
                        p_.append(f[f"s{s}|{kind}|patient"][:, :K].mean())
                        s_.append(f[f"s{s}|{kind}|scramble"][:, :K].mean())
                    if r_:
                        real.append(np.mean(r_)); fp.append(np.mean(p_)); fs.append(np.mean(s_)); nt += 1
                if not nt:
                    continue
                rng = np.random.default_rng(0)
                def ci(v):
                    v = np.asarray(v)
                    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                    return f"{v.mean():.3f} [{np.quantile(b, .025):.3f}, {np.quantile(b, .975):.3f}]"
                d1, d2 = np.asarray(real) - np.asarray(fp), np.asarray(real) - np.asarray(fs)
                res[f"{study}|{kind}|{K}"] = {"real": float(np.mean(real)), "fdr_patient": float(np.mean(fp)), "fdr_scramble": float(np.mean(fs))}
                lines.append(f"| {study} | {kind} | {K} | {ci(real)} | {ci(fp)} | {ci(fs)} | {ci(d1)} | {ci(d2)} | {nt} | {nt * K * N_DRAW} |")
    text = "\n".join(lines) + "\n"
    (V5 / "fdr_table.md").write_text(text)
    (V5 / "fdr_table.json").write_text(json.dumps(res, indent=1))
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["score", "table"])
    ap.add_argument("studies")
    ap.add_argument("--max-targets", type=int, default=40)
    a = ap.parse_args(argv)
    if a.mode == "score":
        L.diag()
        score(a.studies, a.max_targets)
    else:
        print(table(a.studies.split(",")))


if __name__ == "__main__":
    main()
