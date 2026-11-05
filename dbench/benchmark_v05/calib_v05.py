"""Method characterisation: predicted vs realised held-out gain of the release checkpoint (13c36faa) on the benchmark pools.

    python -m dbench.benchmark_v05.calib_v05 score STUDY [--max-targets N]     # stores the network's predicted held-out gain (+ sd) per pool member
    python -m dbench.benchmark_v05.calib_v05 table STUDY[,STUDY...]            # slope, intercept, Spearman, Kendall tau, ECE of the pass probability

Predicted held-out gain = exact support gain minus predicted shrinkage (singles) / pair-head support minus shrinkage (pairs)
(search_v03.pool_scores).  Realised gain = pooled query gain G over B_known from the cached labels.  The pass probability is
the logistic regression (fitted on the odd-numbered targets); slope and intercept are least squares of realised on predicted gain after winsorising both at the 0.1 and 99.9 percentiles of the binary label on the predicted gain, evaluated on the even
ones and vice versa (two-fold over targets); ECE uses 10 equal-mass bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
import numpy as np

BENCH = Path(L.BENCH)
V5 = BENCH / "runs/benchmark_v05"


def score(study, max_targets=40):
    from dbench.benchmark_v04.score_methods import build_pool_and_tasks
    from dbench.benchmark_v05.run_scores import label_dir
    from dbench.benchmark_v05.labels_fast import load_study_v05
    from dbench.harness import ablation_selection as AS
    from dbench.harness import core
    AS.FROZEN = Path("/home/ubuntu/morpheus_work_sw/runs/development/learned_search_v03/release/frozen.pt")
    assert hashlib.sha256(AS.FROZEN.read_bytes()).hexdigest().startswith("13c36faa")
    from morpheus_encoder.learned_search.search_v03 import pool_scores, _support_only
    model = AS._load_model("cuda")
    st = load_study_v05(study)
    targets = json.loads((label_dir(study) / "targets.json").read_text())["targets"][:max_targets]
    od = V5 / "calib" / study
    od.mkdir(parents=True, exist_ok=True)
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if (od / f"{h}.npz").exists():
            continue
        built, singles, pairs = build_pool_and_tasks(st, t)
        arr = {}
        for r, support, query, task, names, ep in built:
            tk = L.known_variant(core.support_only(task), ())
            idx = {n: i for i, n in enumerate(names)}
            s_idx = np.array([idx[n] for n in singles])
            p_idx = np.array([(idx[singles[a]], idx[singles[b]]) for a, b in pairs])
            p1, p2, sd1, sd2 = pool_scores(_support_only(tk), "known", model, p_idx, device="cuda")
            arr[f"s{r}|single"], arr[f"s{r}|pair"] = np.asarray(p1)[s_idx].astype(np.float32), np.asarray(p2).astype(np.float32)
            arr[f"s{r}|single_sd"], arr[f"s{r}|pair_sd"] = np.asarray(sd1)[s_idx].astype(np.float32), np.asarray(sd2).astype(np.float32)
        np.savez_compressed(od / f"{h}.npz", **arr)
        print(study, t, flush=True)


def ece(p, y, bins=10):
    o = np.argsort(p)
    tot = 0.0
    for b in np.array_split(o, bins):
        tot += len(b) * abs(p[b].mean() - y[b].mean())
    return tot / len(p)


def table(studies, B="known", level="95"):
    from scipy.stats import kendalltau, spearmanr
    from sklearn.linear_model import LogisticRegression
    from dbench.benchmark_v04 import tables as T
    from dbench.benchmark_v05.metrics_v05 import label_dir
    lines = ["### Calibration of the release checkpoint (predicted vs realised held-out gain over B_known)", "",
             "| study | kind | n pool members | slope [95% CI] | intercept | Spearman | Kendall tau | ECE of pass probability | ECE of constant prevalence |", "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for study in studies:
        ld = label_dir(study)
        targets = json.loads((ld / "targets.json").read_text())["targets"]
        for kind in ("single", "pair"):
            P, G, Y, TI = [], [], [], []
            for ti, t in enumerate(targets):
                h = hashlib.sha256(t.encode()).hexdigest()[:10]
                cp = V5 / "calib" / study / f"{h}.npz"
                if not cp.exists():
                    continue
                c = np.load(cp)
                meta, z = json.loads((ld / "cache" / f"{h}.json").read_text()), np.load(ld / "cache" / f"{h}.npz")
                for s in T.splits_of(meta):
                    k = f"s{s}|{kind}"
                    if k not in c.files:
                        continue
                    P.append(c[k])
                    G.append(z[f"s{s}|{kind}|{B}|G"])
                    Y.append(T.label(z, s, kind, B, level))
                    TI.append(np.full(len(c[k]), ti))
            if not P:
                continue
            P, G, Y, TI = (np.concatenate(x) for x in (P, G, Y, TI))
            ok = np.isfinite(P) & np.isfinite(G)
            P, G, Y, TI = P[ok], G[ok], Y[ok], TI[ok]
            lo_p, hi_p = np.quantile(P, [0.001, 0.999]); lo_g, hi_g = np.quantile(G, [0.001, 0.999])   # winsorise ill-conditioned fits (|G| up to 1e5)
            P, G = np.clip(P, lo_p, hi_p), np.clip(G, lo_g, hi_g)
            A = np.column_stack([P, np.ones(len(P))])
            coef = np.linalg.lstsq(A, G, rcond=None)[0]
            rng = np.random.default_rng(0)
            tu = np.unique(TI)
            bs = []
            for _ in range(300):
                sel = rng.choice(tu, len(tu), replace=True)
                idx = np.concatenate([np.flatnonzero(TI == q) for q in sel])
                bs.append(np.linalg.lstsq(A[idx], G[idx], rcond=None)[0][0])
            sub = rng.choice(len(P), min(200000, len(P)), replace=False)
            rho = spearmanr(P[sub], G[sub]).correlation
            tau = kendalltau(P[sub][:20000], G[sub][:20000]).correlation
            pp = np.zeros(len(P))
            for fold in (0, 1):
                tr, te = (TI % 2 != fold), (TI % 2 == fold)
                if Y[tr].sum() > 5 and te.any():
                    lr = LogisticRegression(C=1e4, max_iter=500).fit(P[tr][:, None] / max(P.std(), 1e-9), Y[tr])
                    pp[te] = lr.predict_proba(P[te][:, None] / max(P.std(), 1e-9))[:, 1]
            e = ece(pp, Y.astype(float))
            e0 = ece(np.full(len(Y), Y.mean()) + 1e-12 * np.arange(len(Y)), Y.astype(float))
            res[f"{study}|{kind}"] = {"slope": float(coef[0]), "slope_ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "intercept": float(coef[1]),
                                      "spearman": float(rho), "kendall": float(tau), "ece": float(e), "ece_constant": float(e0), "n": int(len(P))}
            lines.append(f"| {study} | {kind} | {len(P)} | {coef[0]:.3f} [{np.quantile(bs, .025):.3f}, {np.quantile(bs, .975):.3f}] | {coef[1]:+.5f} | {rho:.3f} | {tau:.3f} | {e:.4f} | {e0:.4f} |")
    text = "\n".join(lines) + "\n"
    (V5 / "calibration_table.md").write_text(text)
    (V5 / "calibration_table.json").write_text(json.dumps(res, indent=1))
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
