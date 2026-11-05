"""Baselines that see the SAME per-candidate summaries as the v02 learned search.

Every baseline scores candidates from the support-side statistics array of
``learned_search.support_statistics`` (``STAT_NAMES``) and hands the scores to the
identical beam + closed-form refit at the same budget B (the network's
``rank_by`` path), so only the scorer differs:

* ``stats_gbm``   gradient boosting on the statistics, predicting the realised query gain
                  (singles; a second model on symmetrised pair statistics for pairs)
* ``stats_ridge`` ridge on the same statistics and rows
* ``support_lopo_heuristic``  support CV gain - lambda x leave-one-patient-out variance (no training;
                  lambda chosen once on the development study, recorded in the model directory)
* ``support_screen``  support CV gain alone

Training rows come from patient-disjoint episodes of the meta-training studies
(``EpisodeSampler``): per episode the statistics of every candidate and the
realised query gain of every single and of pairs over the top singles.

    python -m dbench.benchmark.summary_baselines --episodes 400 --out runs/development/learned_search_v02/stats_baselines
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/development/learned_search_v02/stats_baselines")
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_TOP_PAIR, N_RANDOM_PAIR = 40, 64   # as the network's pair labels
PAIR_STATS = ("cv_gain", "lopo_var_gain", "lopo_mean_gain")


# ------------------------------------------------------------------ features
def pair_features(S: np.ndarray, pairs: np.ndarray, pair_stats: dict | None = None) -> np.ndarray:
    """Symmetrised pair statistics: elementwise min and max of the two singles' rows, plus the pair's
    own support-side fit statistics when given (order-free, identity-free)."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    a, b = S[pairs[:, 0]], S[pairs[:, 1]]
    cols = [np.minimum(a, b), np.maximum(a, b)]
    if pair_stats is not None:
        cols.append(np.column_stack([np.asarray(pair_stats[k], dtype=np.float64) for k in PAIR_STATS]))
    return np.column_stack(cols).astype(np.float64)


def candidate_pairs(S: np.ndarray, stat_names, rng, n_top: int = N_TOP_PAIR, n_random: int = N_RANDOM_PAIR):
    """Pairs among the top singles by support CV gain plus random pairs (training rows)."""
    p = len(S)
    top = np.argsort(-S[:, list(stat_names).index("cv_gain")], kind="stable")[:n_top]
    pairs = {(int(min(i, j)), int(max(i, j))) for k, i in enumerate(top) for j in top[k + 1:]}
    want = min(len(pairs) + n_random, p * (p - 1) // 2)
    while len(pairs) < want:
        i, j = rng.choice(p, 2, replace=False)
        pairs.add((int(min(i, j)), int(max(i, j))))
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


# ------------------------------------------------------------------ export
def export_episode(ep, rng, device=None) -> dict:
    """Statistics + realised query gains of one episode (singles and pairs)."""
    from morpheus_encoder.learned_search import support_statistics as ss

    S, names = ss.episode_statistics(ep, device=device)
    pairs = candidate_pairs(S, names, rng)
    g = ss.episode_realised_gains(ep, pairs=pairs, device=device)
    return {"S": S, "stat_names": names, "single_gain": np.asarray(g["single"]), "pairs": pairs,
            "pair_gain": np.asarray(g["pair"]), "study": ep.study, "kind": ep.kind,
            "support_patients": list(map(str, ep.support_patients)),
            "query_patients": list(map(str, ep.query_patients))}


def training_rows(exports: list) -> dict:
    Xs, ys, Xp, yp, eid_s, eid_p = [], [], [], [], [], []
    for e, ex in enumerate(exports):
        Xs.append(ex["S"]); ys.append(ex["single_gain"]); eid_s.append(np.full(len(ex["S"]), e))
        Xp.append(pair_features(ex["S"], ex["pairs"])); yp.append(ex["pair_gain"])
        eid_p.append(np.full(len(ex["pairs"]), e))
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)  # noqa: E731
    return {"Xs": np.vstack(Xs), "ys": cat(ys), "eid_s": cat(eid_s),
            "Xp": np.vstack(Xp), "yp": cat(yp), "eid_p": cat(eid_p)}


def fit_models(rows: dict, seed: int = 0) -> dict:
    """Gradient boosting and ridge for singles and pairs (one-off; a few thousand rows each)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def gbm():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=seed)

    def ridge():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    out = {}
    for kind, X, y in (("single", rows["Xs"], rows["ys"]), ("pair", rows["Xp"], rows["yp"])):
        out[f"gbm_{kind}"] = gbm().fit(X, y)
        out[f"ridge_{kind}"] = ridge().fit(X, y)
    return out


def top1_gain(scores: np.ndarray, gains: np.ndarray) -> float:
    return float(gains[int(np.argmax(scores))])


def choose_lambda(exports: list, lambdas=LAMBDAS) -> tuple[float, dict]:
    """lambda maximising the mean realised query gain of the top-ranked single over development episodes."""
    table = {}
    for lam in lambdas:
        vals = []
        for ex in exports:
            names = list(ex["stat_names"])
            sc = ex["S"][:, names.index("cv_gain")] - lam * ex["S"][:, names.index("lopo_var_gain")]
            vals.append(top1_gain(sc, ex["single_gain"]))
        table[f"{lam:g}"] = float(np.mean(vals))
    best = max(lambdas, key=lambda lam: (table[f"{lam:g}"], -lam))
    return float(best), table


# ------------------------------------------------------------------ scorers (the rank_by objects)
class SupportScreen:
    """Support CV gain alone (singles: the single's LOPO gain; pairs: the pair's)."""
    name = "support_screen"

    def singles(self, S, stat_names):
        return np.asarray(S[:, list(stat_names).index("cv_gain")], dtype=float)

    def pairs(self, S, stat_names, pairs_idx, pair_stats=None):
        if pair_stats is not None:
            return np.asarray(pair_stats["cv_gain"], dtype=float)
        s = self.singles(S, stat_names)
        p = np.asarray(pairs_idx, dtype=int)
        return s[p[:, 0]] + s[p[:, 1]]


class LopoHeuristic(SupportScreen):
    """Support CV gain - lambda x leave-one-patient-out variance of the gain."""
    name = "support_lopo_heuristic"

    def __init__(self, lam: float):
        self.lam = float(lam)

    def singles(self, S, stat_names):
        n = list(stat_names)
        return S[:, n.index("cv_gain")] - self.lam * S[:, n.index("lopo_var_gain")]

    def pairs(self, S, stat_names, pairs_idx, pair_stats=None):
        if pair_stats is not None:
            return np.asarray(pair_stats["cv_gain"]) - self.lam * np.asarray(pair_stats["lopo_var_gain"])
        s = self.singles(S, stat_names)
        p = np.asarray(pairs_idx, dtype=int)
        return s[p[:, 0]] + s[p[:, 1]]


class StatsModel:
    """A fitted regressor on the statistics (``gbm`` or ``ridge``) predicting realised query gain."""

    def __init__(self, single_model, pair_model, stat_names, name: str, uses_pair_stats: bool = False):
        self.single_model, self.pair_model = single_model, pair_model
        self.stat_names, self.name, self.uses_pair_stats = list(stat_names), name, uses_pair_stats

    def _check(self, stat_names):
        if list(stat_names) != self.stat_names:
            raise ValueError("statistics columns differ from the ones the model was trained on")

    def singles(self, S, stat_names):
        self._check(stat_names)
        return self.single_model.predict(np.asarray(S, dtype=np.float64))

    def pairs(self, S, stat_names, pairs_idx, pair_stats=None):
        self._check(stat_names)
        if self.uses_pair_stats and pair_stats is None:
            raise ValueError(f"{self.name}: the pair model was trained with the pairs' support statistics")
        return self.pair_model.predict(pair_features(S, pairs_idx, pair_stats if self.uses_pair_stats else None))


def load_scorer(name: str, model_dir: str | Path = OUT):
    """The ``rank_by`` object of a registered baseline (fitted models read from ``model_dir``)."""
    d = Path(model_dir)
    if name == "support_screen":
        return SupportScreen()
    prov = json.loads((d / "PROVENANCE.json").read_text())
    if name == "support_lopo_heuristic":
        return LopoHeuristic(prov["lambda"])
    import joblib
    kind = {"stats_gbm": "gbm", "stats_ridge": "ridge"}[name]
    models = joblib.load(d / "models.joblib")
    return StatsModel(models[f"{kind}_single"], models[f"{kind}_pair"], prov["stat_names"], name)


# ------------------------------------------------------------------ CLI
def sample_exports(sampler, n: int, seed: int, device=None, log=print) -> list:
    rng = np.random.default_rng(seed)
    out, t = [], time.time()
    while len(out) < n:
        try:
            ep = sampler.sample(rng)
            out.append(export_episode(ep, rng, device=device))
        except (ValueError, RuntimeError) as exc:  # a study too small for a patient-disjoint episode, a concept target, ...
            log(f"episode skipped: {exc}")
            continue
        if len(out) % 25 == 0:
            log(f"{len(out)}/{n} episodes, {time.time() - t:.0f} s")
    return out


def main(argv=None) -> None:
    import joblib

    from morpheus_encoder.learned_search.episodes import EpisodeSampler, studies_from_partitions

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partitions", default="dbench/protocols/sprint_partitions_v01.json")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--dev-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--morph-dictionary", default="runs/development/learned_search_v02/morphology_dictionary")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, dev, info = studies_from_partitions(args.partitions, morph_dictionary=args.morph_dictionary)
    kw = dict(controlled_prob=0.5, concepts=False, morph_target_prob=0.3)   # the network's training sampler
    tr = sample_exports(EpisodeSampler(train, split="train", seed=args.seed, **kw), args.episodes, args.seed,
                        device=args.device)
    dv = sample_exports(EpisodeSampler(dev, split="train", seed=args.seed + 1, **kw), args.dev_episodes,
                        args.seed + 1, device=args.device)
    rows = training_rows(tr)
    models = fit_models(rows, seed=0)
    lam, lam_table = choose_lambda(dv)
    joblib.dump(models, out / "models.joblib")
    np.savez_compressed(out / "training_rows.npz", **rows)
    dev_rows = training_rows(dv)
    names = tr[0]["stat_names"]
    evals = {}
    for nm, sc in (("stats_gbm", StatsModel(models["gbm_single"], models["gbm_pair"], names, "stats_gbm")),
                   ("stats_ridge", StatsModel(models["ridge_single"], models["ridge_pair"], names, "stats_ridge")),
                   ("support_lopo_heuristic", LopoHeuristic(lam)), ("support_screen", SupportScreen())):
        evals[nm] = {"dev_top1_single_gain": float(np.mean([top1_gain(sc.singles(e["S"], names), e["single_gain"])
                                                              for e in dv])),
                     "dev_spearman_single": float(np.mean([_spearman(sc.singles(e["S"], names), e["single_gain"])
                                                           for e in dv]))}
    prov = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "partitions": args.partitions,
            "partitions_sha256": hashlib.sha256(Path(args.partitions).read_bytes()).hexdigest(),
            "studies": info, "seed": args.seed, "morph_dictionary": args.morph_dictionary, "sampler": kw, "n_train_episodes": len(tr), "n_dev_episodes": len(dv),
            "train_studies": sorted({e["study"] for e in tr}), "dev_studies": sorted({e["study"] for e in dv}),
            "n_single_rows": int(len(rows["ys"])), "n_pair_rows": int(len(rows["yp"])),
            "n_dev_single_rows": int(len(dev_rows["ys"])), "stat_names": list(names),
            "pair_features": "elementwise min and max of the two singles' statistics (no pair fits, so the "
                             "beam's refit budget B stays the only relationships tested)",
            "label": "realised query gain (support_statistics.realised_gains: fit on support patients, general "
                     "gain on disjoint query patients)", "lambda": lam, "lambda_grid_dev_top1_gain": lam_table,
            "lambda_rule": "max mean realised query gain of the top-ranked single over development episodes",
            "dev_evaluation": evals, "seconds": round(time.time() - t0, 1)}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, default=str))
    print(json.dumps({k: prov[k] for k in ("n_train_episodes", "n_single_rows", "n_pair_rows", "lambda",
                                           "dev_evaluation", "seconds")}, indent=1))


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else 0.0


if __name__ == "__main__":
    main()
