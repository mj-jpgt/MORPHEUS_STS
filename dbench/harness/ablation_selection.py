"""Methods ablation of the learned selection at matched budget (development set; not a headline table).

Every selector shares the candidate tokens' fixed transfer diagnostics, the LCB beam (26 / 26 / 8 / 4 at K = 64), the
v03_rff refit and the support-gain final ordering of ``search_v03.search``; only the predicted shrinkage that decides
WHICH sets are fitted differs.  Pool scores (every pool single and pair) are each selector's own predicted held-out
gain, the definition used for the network in ``search_v03.pool_scores``:
  g_theta         learned relation search (release checkpoint), the reference
  sel_gbm         gradient boosting on the 11 diagnostics + B / L codes -> shrinkage (baseline 9), refit on the
                  relabelled label cache (b7879b1); pairs from min / max of the two diagnostic rows
  sel_ridge       ridge on the same features
  sel_screen      support-fit screen: support CV gain alone (no shrinkage model; pairs = mean of the singles')
  sel_lopo        support gain minus the leave-one-patient-out SD of the gain (baseline 10)
  sel_learned1    the network's single scores; pair selection by |corr(centred product, support residual)| on support
                  rows instead of the pair head (ablation of the learned pair selection)
  random          count-matched random

    python -m dbench.harness.ablation_selection --train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/development/benchmark_v03/ablation_selection")
FROZEN = "runs/development/learned_search_v03/release/frozen.pt"


def train() -> dict:
    import joblib
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .stat_baselines import LABELS, fit, training_rows
    t = time.time()
    holdouts = json.loads(Path("runs/development/learned_search_v03/transfer_holdouts.json").read_text())
    smoke = ("syn0", "syn1", "syn2")   # 12 entries written by the 2026-10-21 synthetic smoke of the repair pipeline
    rows = training_rows(LABELS, exclude_studies=tuple(holdouts.get("withheld_studies", ())) + smoke)
    t_load = time.time() - t
    # overflowed relation fits in the relabelled cache (diagnostics up to 1e14, shrinkage far outside R^2 units) are
    # not labels: keep rows with |shrinkage| <= 5 and every |diagnostic| <= 1e3 (finite rows only, as g_theta's loss)
    dropped = {}
    for X, ys in (("Xs", ("ys",)), ("Xp", ("yp", "sp"))):
        ok = np.isfinite(rows[X]).all(1) & (np.abs(rows[X]) <= 1e3).all(1)
        for k in ys:
            ok &= np.isfinite(rows[k]) & (np.abs(rows[k]) <= 5)
        dropped[X] = int((~ok).sum())
        rows[X] = rows[X][ok]
        for k in ys:
            rows[k] = rows[k][ok]

    def ridge(X, y):
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X[ok], y[ok])
    models = {"gbm": fit(rows),
              "ridge": {"single_delta": ridge(rows["Xs"], rows["ys"]), "pair_delta": ridge(rows["Xp"], rows["yp"]),
                        "pair_gsup": ridge(rows["Xp"], rows["sp"])}}
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, OUT / "models.joblib")
    files = sorted(p.name for p in Path(LABELS).glob("*.npz"))
    prov = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "labels": str(LABELS),
            "label_fix_commit": "b7879b1", "label_files_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest(),
            "n_label_files": len(files), "n_episodes": rows["n_episodes"], "studies": rows["studies"],
            "n_single_rows": int(len(rows["ys"])), "n_pair_rows": int(len(rows["yp"])),
            "rows_dropped_overflow": dropped, "seconds_load": round(t_load, 1), "seconds_total": round(time.time() - t, 1)}
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    return prov


class Screen:
    has_sd = False

    def singles(self, task, B, L, diag, base):
        return np.zeros(len(diag)), np.zeros(len(diag))

    def pairs(self, task, B, L, diag, base, single_g_sup):
        g = np.asarray(single_g_sup, dtype=float)
        return 0.5 * (g[:, None] + g[None, :]), np.zeros((len(g), len(g))), np.zeros((len(g), len(g)))


def product_corr(task, base) -> np.ndarray:
    """|corr(centred x_a * x_b, support residual)| for every pair of candidate columns (support rows), via three
    p x p products: cov = Z^T diag(r) Z / n, E[(ab)^2] = (Z^2)^T Z^2 / n, E[ab] = Z^T Z / n."""
    from morpheus_encoder.learned_search.relation_v03 import _codes
    s = task.s
    X = np.asarray(task.X[:s], dtype=np.float64)
    g = _codes(np.asarray(task.groups[:s]))
    for k in np.unique(g):   # within-patient centring, as the relation fits
        X[g == k] -= X[g == k].mean(0)
    X /= np.maximum(X.std(0), 1e-12)
    r = np.asarray(base.r_s, dtype=np.float64)
    r = (r - r.mean()) / max(r.std(), 1e-12)
    n = len(r)
    m1 = X.T @ X / n
    m2 = (X ** 2).T @ (X ** 2) / n
    cov = (X * r[:, None]).T @ X / n - m1 * r.mean()
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))
    return np.abs(cov / sd)


class LearnedSinglesHeuristicPairs:
    """The network's single shrinkage (and sd); pairs ranked by |corr(product, residual)|, no pair head."""
    has_sd = True

    def __init__(self, model):
        self.model = model

    def singles(self, task, B, L, diag, base):
        from morpheus_encoder.learned_search.search_v03 import score_model
        mu, sg, _ = score_model(self.model, task, B, L, diag, base)
        return mu, sg

    def pairs(self, task, B, L, diag, base, single_g_sup):
        c = product_corr(task, base)
        z = np.zeros_like(c)
        return c, z, z


def _load_model(device):
    from morpheus_encoder.learned_search.model_v03 import GainModelV03
    return GainModelV03.load(FROZEN, map_location=device or "cpu").to(device or "cpu")


class ClippedStat:
    """StatGBM-style scorer whose diagnostics are clipped to the training bound (|d| <= 1e3; overflowed fits)."""
    has_sd = False

    def __init__(self, models):
        from .stat_baselines import StatGBM
        self.inner = StatGBM(models)

    def singles(self, task, B, L, diag, base):
        return self.inner.singles(task, B, L, np.clip(np.nan_to_num(diag, posinf=1e3, neginf=-1e3), -1e3, 1e3), base)

    def pairs(self, task, B, L, diag, base, single_g_sup):
        return self.inner.pairs(task, B, L, np.clip(np.nan_to_num(diag, posinf=1e3, neginf=-1e3), -1e3, 1e3), base,
                                single_g_sup)


def make_scorer(name: str, device=None):
    import joblib
    from .stat_baselines import LopoHeuristic
    if name == "sel_gbm":
        return ClippedStat(joblib.load(OUT / "models.joblib")["gbm"])
    if name == "sel_ridge":
        return ClippedStat(joblib.load(OUT / "models.joblib")["ridge"])
    if name == "sel_screen":
        return Screen()
    if name == "sel_lopo":
        return LopoHeuristic()
    if name == "sel_learned1":
        return LearnedSinglesHeuristicPairs(_load_model(device))
    if name == "sel_screen_joint":
        return Screen()
    if name == "sel_hybrid":
        return _HybridModel(_load_model(device))
    raise KeyError(name)


class _HybridModel(Screen):
    """Pool-scoring holder of the network for the hybrid (exact support gain minus the network's mu_delta)."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, k):
        return getattr(self.__dict__["model"], k)


def joint_pair_gsup(task, base, P, device=None) -> np.ndarray:
    """Exact support CV gain of each given pair fitted jointly (v03_rff, the relation family of the refit)."""
    from morpheus_encoder.learned_search import relation_v03 as rv
    P = np.asarray(P, dtype=int).reshape(-1, 2)
    return np.asarray(rv.fit_sets(task.X[:, P], base, task.y, task.groups, device=device)["g_sup"], dtype=float)


def network_shrinkage(model, task, B, diag, base, P):
    """The network's predicted single shrinkage mu_delta (all candidates) and pair shrinkage (given pairs)."""
    import torch
    from morpheus_encoder.learned_search.model_v03 import SCALE
    from morpheus_encoder.learned_search.search_v03 import score_model
    mu1, _, H = score_model(model, task, B, "L0", diag, base)
    pr = torch.as_tensor(np.asarray(P, dtype=np.int64).reshape(-1, 2), device=H.device)
    with torch.no_grad():
        _, md, _ = model.pair_head(H, pr)
    return np.asarray(mu1, float), md.cpu().numpy() / SCALE


KAPPA_POOL = {"sel_lopo": 1.0}   # the heuristic IS gain minus one LOPO SD; the others score mu (kappa 0) as g_theta


def pool_scorer(name: str, device=None):
    """Pool score of every single and given pairs: G_sup - mu (singles) and pair support gain - pair shrinkage (pairs),
    minus kappa * sd for the LOPO heuristic; the pair-product selector scores pairs by |corr|."""
    cache = {}

    def score(task, B, singles, pairs):
        import torch
        from morpheus_encoder.learned_search import relation_v03 as rv
        from morpheus_encoder.learned_search.search_v03 import _support_only
        if "s" not in cache:
            cache["s"] = make_scorer(name, device)
        sc = cache["s"]
        from . import core
        task = _support_only(core.support_only(task))   # keeps the B_known exclusion inputs
        dev = torch.device(device) if device is not None else None
        Bfit = "nuis" if B == "decomp" else B
        base = rv.baseline_for(task, Bfit)
        diag, single = rv.diagnostics(task, Bfit, base=base, device=dev)
        gs = np.asarray(single["g_sup"], dtype=float)
        mu1, sg1 = (np.asarray(a, dtype=float) for a in sc.singles(task, B, "L0", diag, base))
        k = KAPPA_POOL.get(name, 0.0)
        s1 = (gs - mu1 - k * sg1)[np.asarray(singles)]
        P = np.asarray(pairs, dtype=int).reshape(-1, 2)
        if name == "sel_learned1":
            s2 = product_corr(task, base)[P[:, 0], P[:, 1]]
        elif name in ("sel_screen_joint", "sel_hybrid"):
            jg = joint_pair_gsup(task, base, P, device=dev)
            if name == "sel_hybrid":
                mu1n, mdp = network_shrinkage(sc.model, task, B, diag, base, P)
                s1 = (gs - mu1n)[np.asarray(singles)]
                jg = jg - mdp
            s2 = jg
        else:
            msup, mdel, msig = (np.asarray(a, dtype=float) for a in sc.pairs(task, B, "L0", diag, base, gs))
            s2 = (msup - mdel - k * msig)[P[:, 0], P[:, 1]]
        return s1, s2, []
    return score


def ablation_scorers(device=None) -> dict:
    return {n: pool_scorer(n, device) for n in ("sel_gbm", "sel_ridge", "sel_screen", "sel_lopo", "sel_learned1",
                                                 "sel_screen_joint", "sel_hybrid")}


def ablation_searches(device=None) -> dict:
    from .stat_baselines import scorer_search
    return {n: (lambda n=n: scorer_search(make_scorer(n, device), device=device))
            for n in ("sel_gbm", "sel_ridge", "sel_screen", "sel_lopo", "sel_learned1")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args(argv)
    if a.train:
        print(json.dumps(train(), indent=1))


if __name__ == "__main__":
    main()
