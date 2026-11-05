"""Statistic-based baselines 9 and 10 (governing spec 3.5) on the shared beam and refit at budget K.

Both plug into ``search_v03.search(..., scorer=...)``, so the candidate tokens' fixed transfer
diagnostics (``relation_v03.DIAG_NAMES``), the beam, the fits and the final lower-bound ranking are
exactly those of g_theta; only the predicted shrinkage (and the pair support gain) differs.

9  ``stat_gbm``: gradient boosting on the diagnostics predicting SHRINKAGE, trained on the same episode
   labels as g_theta (``runs/development/learned_search_v03/label_cache``, uniform labels, as the
   network's regression heads).  Singles: diag row + (B, L) codes -> Delta.  Pairs: elementwise min and
   max of the two diag rows + (B, L) -> Delta of the pair and support gain of the pair.  No variance.
10 ``lopo_heuristic``: support gain minus the leave-one-patient-out SD of the gain (kappa = 1, mu = 0,
   sigma = the LOPO SD; a pair's SD = mean of its singles'), no learning.
11 the support-fit screen is ``search(model=None)`` (``baselines.screen_search``).

    python -m dbench.harness.stat_baselines --train      # fit stat_gbm on the label cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

LABELS = Path("runs/development/learned_search_v03/label_cache")
OUT = Path("runs/development/benchmark_v03/stat_gbm")
B_CODE = {"nuis": 0, "known": 1, "decomp": 0}
L_CODE = {"L0": 0, "L1": 1, "L2": 2}


def _cond(B, L, n):
    return np.tile([B_CODE.get(str(B), 0), L_CODE.get(str(L), 0)], (n, 1)).astype(float)


def pair_features(diag: np.ndarray, pairs: np.ndarray, B, L) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    a, b = diag[pairs[:, 0]], diag[pairs[:, 1]]
    return np.column_stack([np.minimum(a, b), np.maximum(a, b), _cond(B, L, len(pairs))])


def training_rows(label_dir: Path = LABELS, exclude_studies=()) -> dict:
    Xs, ys, Xp, yp, sp, studies = [], [], [], [], [], []
    for f in sorted(Path(label_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if str(z["study"]) in exclude_studies:
            continue
        sp_, qp_ = set(map(str, np.atleast_1d(z["support_patients"]))), set(map(str, np.atleast_1d(z["query_patients"])))
        if sp_ & qp_:   # row-split episodes (support and query share patients): not g_theta's labels
            continue
        diag, B, L = z["diag"].astype(float), str(z["B"]), str(z["L"])
        idx = np.asarray(z["single_idx"], dtype=int)
        uni = np.asarray(z["single_uniform"], dtype=bool)
        si = idx[uni]
        Xs.append(np.column_stack([diag[si], _cond(B, L, len(si))]))
        ys.append(np.asarray(z["single_delta"])[si])
        pu = np.asarray(z["pair_uniform"], dtype=bool)
        pairs = np.asarray(z["pairs"], dtype=int)[pu]
        Xp.append(pair_features(diag, pairs, B, L))
        yp.append(np.asarray(z["pair_delta"])[pu])
        sp.append(np.asarray(z["pair_gsup"])[pu])
        studies.append(str(z["study"]))
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)  # noqa: E731
    return {"Xs": np.vstack(Xs) if Xs else np.zeros((0, 13)), "ys": cat(ys), "Xp": np.vstack(Xp) if Xp else np.zeros((0, 24)),
            "yp": cat(yp), "sp": cat(sp), "n_episodes": len(studies), "studies": sorted(set(studies))}


def fit(rows: dict, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    def gbm():
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                             random_state=seed)
    def fit_finite(X, y):   # non-finite labels are masked, as in g_theta's loss
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        return gbm().fit(X[ok], y[ok])
    return {"single_delta": fit_finite(rows["Xs"], rows["ys"]), "pair_delta": fit_finite(rows["Xp"], rows["yp"]),
            "pair_gsup": fit_finite(rows["Xp"], rows["sp"])}


class StatGBM:
    has_sd = False

    def __init__(self, models: dict):
        self.m = models

    def singles(self, task, B, L, diag, base):
        p = len(diag)
        return self.m["single_delta"].predict(np.column_stack([diag, _cond(B, L, p)])), np.zeros(p)

    def pairs(self, task, B, L, diag, base, single_g_sup):
        p = len(diag)
        iu = np.triu_indices(p, 1)
        F = pair_features(diag, np.stack(iu, 1), B, L)
        msup, mdel = np.zeros((p, p)), np.zeros((p, p))
        for M, key in ((msup, "pair_gsup"), (mdel, "pair_delta")):
            v = self.m[key].predict(F)
            M[iu], M[(iu[1], iu[0])] = v, v
        return msup, mdel, np.zeros((p, p))


class LopoHeuristic:
    has_sd = False

    def singles(self, task, B, L, diag, base):
        from morpheus_encoder.learned_search.relation_v03 import DIAG_NAMES
        sd = np.asarray(diag[:, list(DIAG_NAMES).index("g_sup_lopo_sd")], dtype=float)
        return np.zeros(len(diag)), sd

    def pairs(self, task, B, L, diag, base, single_g_sup):
        _, sd = self.singles(task, B, L, diag, base)
        g = np.asarray(single_g_sup, dtype=float)
        return 0.5 * (g[:, None] + g[None, :]), np.zeros((len(g), len(g))), 0.5 * (sd[:, None] + sd[None, :])


def scorer_search(scorer, device=None, kappa: float = 1.0):
    from morpheus_encoder.learned_search.search_v03 import search as s3

    def search(task, B, K):
        return s3(task, B, K, scorer=scorer, kappa=kappa, device=device)
    search.parallel_ok = device is None or str(device) == "cpu"
    search.gpu = str(device).startswith("cuda")
    return search


def load_stat_gbm(out: Path = OUT):
    import joblib
    return StatGBM(joblib.load(Path(out) / "models.joblib"))


def main(argv=None) -> None:
    import joblib

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    if args.train:
        t = time.time()
        holdouts = json.loads(Path("runs/development/learned_search_v03/transfer_holdouts.json").read_text())
        rows = training_rows(Path(args.labels), exclude_studies=holdouts.get("withheld_studies", ()))
        models = fit(rows)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, out / "models.joblib")
        files = sorted(p.name for p in Path(args.labels).glob("*.npz"))
        prov = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "labels": args.labels,
                "label_files_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest(), "n_label_files": len(files),
                "n_episodes": rows["n_episodes"], "studies": rows["studies"], "n_single_rows": int(len(rows["ys"])),
                "n_pair_rows": int(len(rows["yp"])), "features": "diag (relation_v03.DIAG_NAMES) + B, L codes; pairs: "
                "min/max of the two diag rows + B, L", "targets": "shrinkage Delta (singles, pairs) and pair support gain",
                "rows": "uniformly sampled labels only (spec 1.5 regression terms); patient-disjoint episodes only "
                        "(row-split episodes of the ablation run excluded); transfer-holdout studies excluded", "seconds": round(time.time() - t, 1)}
        (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
        print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
