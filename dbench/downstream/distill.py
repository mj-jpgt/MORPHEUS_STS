"""Assay distillation (downstream spec Section 7): H&E-only students for a frozen state score.

Students (trained on the spatial discovery cohort only, frozen H-Optimus-0 embeddings, no fine-tuning):
  1. ``RidgeStudent``  ridge on the embedding, penalty by patient-grouped CV;
  2. ``MLPStudent``    2-layer MLP (512 hidden, GELU, dropout 0.2), early stopping on held-out patients;
  3. ``GatedAttentionStudent``  gated attention pooling (Ilse et al. 2018) over the tiles of a region,
     trained on region-level targets (mean state score of a discovery region), applied to whole slides.
Baselines: CellViT descriptor ridge; STPath zero-shot and DeepSpot pretrained through ``BASELINE_STATUS``
(run once and cached when their weights download without an account or licence acceptance).

Evaluation (7.5): held-out-patient tile and patient Spearman with a permuted-target null; slide cohorts
(TCGA-KIRC, CPTAC) against the bulk molecular score with pooled, leave-one-site-out-calibrated and
within-site-centred Spearman; success rule of 7.5. Prediction-powered intervals (7.6): ``ppi_mean`` and
``ppi_ols``.
"""
from __future__ import annotations

import numpy as np

from .relation import cv_alpha, patient_folds

BASELINE_STATUS = {
    "STPath": {"status": "unavailable",
               "reason": "public weights (huggingface tlhuang/STPath, not gated) take Prov-GigaPath tile features as "
                         "input; GigaPath weights require licence acceptance on a gated repository, and the project "
                         "uses H-Optimus-0 as its only pathology backbone, so STPath cannot be run as published "
                         "(zero-shot on H-Optimus-0 features would be a different, untested model)"},
    "DeepSpot": {"status": "weights available",
                 "source": "Zenodo 15322099 (CC BY 4.0, no account): Kidney_HEST1K and Kidney_Lung_USZ models, "
                           "image_feature_model hoptimus0, spot_context spot_subspot_neighbors",
                 "caveat": "Kidney_HEST1K was trained on HEST-1K kidney samples, which include the discovery cohort "
                           "(Meylan 2022, INT1-INT24): its within-discovery numbers are not held-out; TCGA/CPTAC are"},
    "DeepSpot-M": {"status": "unavailable",
                   "reason": "huggingface ratschlab/DeepSpotM is gated (manual access approval); not requested"},
}


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else float("nan")


# ----------------------------------------------------------------------------- students
class RidgeStudent:
    alphas = tuple(10.0 ** np.arange(-1, 6))

    def fit(self, E, y, groups):
        E = np.asarray(E, float)
        self.mu, self.sd = E.mean(0), E.std(0) + 1e-8
        X = (E - self.mu) / self.sd
        self.ymu = float(np.mean(y))
        a, _ = cv_alpha(X, np.asarray(y, float) - self.ymu, groups, self.alphas)
        self.alpha = a
        self.coef = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ (np.asarray(y, float) - self.ymu))
        return self

    def predict(self, E):
        return ((np.asarray(E, float) - self.mu) / self.sd) @ self.coef + self.ymu


def _device():
    import torch

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 8e9:
            return torch.device("cuda")
    return torch.device("cpu")


class MLPStudent:
    def __init__(self, hidden=512, dropout=0.2, lr=1e-3, wd=1e-4, max_epochs=200, patience=10, seed=0):
        self.h, self.p, self.lr, self.wd, self.max_epochs, self.patience, self.seed = hidden, dropout, lr, wd, max_epochs, patience, seed

    def fit(self, E, y, groups):
        import torch

        torch.manual_seed(self.seed)
        dev = _device()
        E = np.asarray(E, np.float32)
        y = np.asarray(y, np.float32)
        self.mu, self.sd = E.mean(0), E.std(0) + 1e-6
        self.ymu, self.ysd = float(y.mean()), float(y.std() + 1e-8)
        groups = np.asarray(groups).astype(str)
        pats = np.array(sorted(np.unique(groups)))
        rng = np.random.default_rng(self.seed)
        val_p = rng.choice(pats, max(1, len(pats) // 5), replace=False)       # early stopping on held-out patients
        v = np.isin(groups, val_p)
        X = torch.tensor((E - self.mu) / self.sd, device=dev)
        Y = torch.tensor((y - self.ymu) / self.ysd, device=dev)
        net = torch.nn.Sequential(torch.nn.Linear(E.shape[1], self.h), torch.nn.GELU(), torch.nn.Dropout(self.p),
                                  torch.nn.Linear(self.h, self.h), torch.nn.GELU(), torch.nn.Dropout(self.p),
                                  torch.nn.Linear(self.h, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        tr = torch.tensor(np.flatnonzero(~v), device=dev)
        va = torch.tensor(np.flatnonzero(v), device=dev)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.max_epochs):
            net.train()
            perm = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = torch.mean((net(X[b]).squeeze(1) - Y[b]) ** 2)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(torch.mean((net(X[va]).squeeze(1) - Y[va]) ** 2))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net, self.dev, self.epochs = net.eval(), dev, ep + 1
        return self

    def predict(self, E):
        import torch

        X = torch.tensor((np.asarray(E, np.float32) - self.mu) / self.sd, device=self.dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(self.net(X[i:i + 8192]).squeeze(1).cpu().numpy())
        return np.concatenate(out) * self.ysd + self.ymu


class GatedAttentionStudent:
    """Bags of tiles -> one score. Trained on region bags of the discovery cohort."""

    def __init__(self, hidden=256, attn=128, dropout=0.2, lr=2e-4, wd=1e-4, max_epochs=100, patience=10, seed=0):
        self.h, self.a, self.p, self.lr, self.wd = hidden, attn, dropout, lr, wd
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def _net(self, d):
        import torch

        class Net(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = torch.nn.Sequential(torch.nn.Linear(d, self.h), torch.nn.GELU(), torch.nn.Dropout(self.p))
                s.V = torch.nn.Linear(self.h, self.a)
                s.U = torch.nn.Linear(self.h, self.a)
                s.w = torch.nn.Linear(self.a, 1)
                s.head = torch.nn.Linear(self.h, 1)

            def forward(s, x):                       # x (n_tiles, d)
                h = s.enc(x)
                att = s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h))).squeeze(1)
                a = torch.softmax(att, 0)
                return s.head((a[:, None] * h).sum(0)).squeeze(), a
        return Net()

    def fit(self, bags, targets, groups):
        import torch

        torch.manual_seed(self.seed)
        dev = _device()
        allE = np.vstack(bags).astype(np.float32)
        self.mu, self.sd = allE.mean(0), allE.std(0) + 1e-6
        t = np.asarray(targets, np.float32)
        self.ymu, self.ysd = float(t.mean()), float(t.std() + 1e-8)
        groups = np.asarray(groups).astype(str)
        pats = np.array(sorted(np.unique(groups)))
        rng = np.random.default_rng(self.seed)
        val_p = rng.choice(pats, max(1, len(pats) // 5), replace=False)
        v = np.isin(groups, val_p)
        B = [torch.tensor((b - self.mu) / self.sd, dtype=torch.float32, device=dev) for b in bags]
        T = torch.tensor((t - self.ymu) / self.ysd, device=dev)
        net = self._net(allE.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        tr_idx, va_idx = np.flatnonzero(~v), np.flatnonzero(v)
        for ep in range(self.max_epochs):
            net.train()
            for i in rng.permutation(tr_idx):
                opt.zero_grad()
                loss = (net(B[i])[0] - T[i]) ** 2
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([float((net(B[i])[0] - T[i]) ** 2) for i in va_idx]))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net, self.dev, self.epochs = net.eval(), dev, ep + 1
        return self

    def predict_bags(self, bags, max_tiles: int = 4096, seed: int = 0):
        import torch

        rng = np.random.default_rng(seed)
        out = []
        with torch.no_grad():
            for b in bags:
                if len(b) > max_tiles:
                    b = b[rng.choice(len(b), max_tiles, replace=False)]
                x = torch.tensor((np.asarray(b, np.float32) - self.mu) / self.sd, device=self.dev)
                out.append(float(self.net(x)[0]) * self.ysd + self.ymu)
        return np.asarray(out)


def region_bags(E, state, coords_um, sections, patients, size_um: float = 1000.0, min_tiles: int = 20):
    """Discovery region bags: spots in non-overlapping square windows; target = mean state score."""
    bags, targets, groups = [], [], []
    sections = np.asarray(sections).astype(str)
    for s in np.unique(sections):
        ix = np.flatnonzero(sections == s)
        ij = np.floor((coords_um[ix] - coords_um[ix].min(0)) / size_um).astype(int)
        for key in {tuple(k) for k in ij}:
            rows = ix[(ij == key).all(1)]
            if len(rows) >= min_tiles:
                bags.append(np.asarray(E[rows], np.float32))
                targets.append(float(np.mean(state[rows])))
                groups.append(str(patients[rows[0]]))
    return bags, np.asarray(targets), np.asarray(groups)


# ----------------------------------------------------------------------------- evaluation
def heldout_spot_eval(make, E, y, groups, *, n_perm: int = 0, seed: int = 0, max_folds: int = 6) -> dict:
    """Patient-grouped folds: tile Spearman, patient-mean Spearman; permuted-target null (patient labels of
    the target permuted as whole patient blocks is impossible with unequal sizes, so the target is
    permuted within the training rows, the model refit, and the held-out Spearman recomputed)."""
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y, float)

    def run(yy):
        pred = np.zeros(len(yy))
        for held in patient_folds(groups, max_folds=max_folds, seed=seed):
            h = np.isin(groups, held)
            pred[h] = make().fit(E[~h], yy[~h], groups[~h]).predict(E[h])
        return pred

    pred = run(y)
    pats = np.unique(groups)
    pm = np.array([pred[groups == p].mean() for p in pats])
    ym = np.array([y[groups == p].mean() for p in pats])
    rng_b = np.random.default_rng(seed + 20261021)
    bs = [spearman(pm[i], ym[i]) for i in (rng_b.integers(0, len(pats), len(pats)) for _ in range(2000))]
    out = {"tile_spearman": spearman(pred, y), "patient_spearman": spearman(pm, ym),
           "patient_spearman_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
           "n_tiles": int(len(y)), "n_patients": int(len(pats)), "_pred": pred}
    if n_perm:
        rng = np.random.default_rng(seed + 1)
        null = [spearman(run(rng.permutation(y)), y) for _ in range(n_perm)]
        out.update(null_tile_spearman_mean=float(np.mean(null)), null_tile_spearman_max=float(np.max(null)),
                   n_perm=n_perm)
    return out


def site_of(barcode: str) -> str:
    """TCGA tissue source site = the two characters after 'TCGA-'."""
    return str(barcode).split("-")[1]


def slide_cohort_eval(pred, ref, sites, *, n_boot: int = 2000, seed: int = 20261021, null=None) -> dict:
    """Pooled Spearman (+ patient bootstrap CI), leave-one-site-out calibrated Spearman (a linear
    calibration of prediction to reference fitted on the other sites, applied to the held-out site, pooled)
    and within-site-centred Spearman (prediction and reference centred within site)."""
    pred, ref, sites = np.asarray(pred, float), np.asarray(ref, float), np.asarray(sites).astype(str)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref, sites = pred[ok], ref[ok], sites[ok]
    rng = np.random.default_rng(seed)
    boots = [spearman(pred[i], ref[i]) for i in (rng.integers(0, len(pred), len(pred)) for _ in range(n_boot))]
    cal = np.zeros(len(pred))
    for s in np.unique(sites):
        m = sites == s
        A = np.column_stack([np.ones((~m).sum()), pred[~m]])
        beta = np.linalg.lstsq(A, ref[~m], rcond=None)[0]
        cal[m] = beta[0] + beta[1] * pred[m]
    pc, rc = pred.copy(), ref.copy()
    for s in np.unique(sites):
        m = sites == s
        pc[m] -= pc[m].mean()
        rc[m] -= rc[m].mean()
    out = {"n": int(len(pred)), "n_sites": int(len(np.unique(sites))), "pooled_spearman": spearman(pred, ref),
           "pooled_ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
           "loso_calibrated_spearman": spearman(cal, ref), "within_site_centred_spearman": spearman(pc, rc)}
    if null is not None:
        out["null_q95"] = float(np.quantile(null, 0.95))
        out["ci_excludes_null"] = bool(out["pooled_ci95"][0] > out["null_q95"])
    return out


def success_rule(best: dict, baselines: dict) -> dict:
    """7.5: best student pooled Spearman > 0.3 with CI excluding the permuted null, and its within-site-
    centred Spearman above STPath and DeepSpot (baselines marked unavailable are listed, not counted)."""
    avail = {k: v for k, v in baselines.items() if v.get("within_site_centred_spearman") is not None}
    beats = all(best["within_site_centred_spearman"] > v["within_site_centred_spearman"] for v in avail.values())
    ok = bool(best["pooled_spearman"] > 0.3 and best.get("ci_excludes_null", False) and beats)
    return {"successful": ok, "baselines_compared": sorted(avail),
            "baselines_unavailable": sorted(set(baselines) - set(avail)),
            "note": None if len(avail) == len(baselines) else "success evaluated against the available baselines only"}


def tcga_internal_detector(E_slide, ref, sites, *, n_splits: int = 5, n_repeats: int = 10, seed: int = 20260824,
                           alphas=None) -> dict:
    """The ORIGINAL-paper protocol (detector trained inside TCGA on the bulk score, repeated 5-fold CV, and a
    leave-one-site-out pass) -- kept only to show whether the G18 numbers reproduce on the stored
    embeddings; the downstream protocol's students are trained on the spatial cohort instead."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(0, 5, 24) if alphas is None else alphas
    E, ref, sites = np.asarray(E_slide, float), np.asarray(ref, float), np.asarray(sites).astype(str)
    preds = np.zeros((n_repeats, len(ref)))
    for r in range(n_repeats):
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed + r).split(E):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(E[tr], ref[tr])
            preds[r, te] = m.predict(E[te])
    p = preds.mean(0)
    loso = np.zeros(len(ref))
    for s in np.unique(sites):
        te = sites == s
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(E[~te], ref[~te])
        loso[te] = m.predict(E[te])

    def centred(x):
        x = x.copy()
        for s in np.unique(sites):
            x[sites == s] -= x[sites == s].mean()
        return x
    return {"n": int(len(ref)), "cv_spearman": spearman(p, ref), "loso_spearman": spearman(loso, ref),
            "cv_within_site_centred": spearman(centred(p), centred(ref)),
            "loso_within_site_centred": spearman(centred(loso), centred(ref))}


# ----------------------------------------------------------------------------- 7.6
def ppi_mean(y_lab, f_lab, f_unlab, alpha: float = 0.05) -> dict:
    """Prediction-powered CI for E[Y] (Angelopoulos et al. 2023) beside the naive one on predictions."""
    from scipy.stats import norm

    y_lab, f_lab, f_unlab = (np.asarray(x, float) for x in (y_lab, f_lab, f_unlab))
    n, N = len(y_lab), len(f_unlab)
    rect = y_lab - f_lab
    est = f_unlab.mean() + rect.mean()
    se = np.sqrt(f_unlab.var(ddof=1) / N + rect.var(ddof=1) / n)
    z = norm.ppf(1 - alpha / 2)
    se_naive = f_unlab.std(ddof=1) / np.sqrt(N)
    return {"ppi": float(est), "ppi_ci": [float(est - z * se), float(est + z * se)],
            "naive": float(f_unlab.mean()), "naive_ci": [float(f_unlab.mean() - z * se_naive),
                                                          float(f_unlab.mean() + z * se_naive)],
            "classical_labelled_only": float(y_lab.mean()), "n_labelled": n, "n_unlabelled": N}


def ppi_ols(X_lab, y_lab, f_lab, X_unlab, f_unlab, alpha: float = 0.05) -> dict:
    """Prediction-powered OLS coefficients: theta = argmin over unlabelled predictions, rectified by the
    labelled residual of predictions vs truth; sandwich variance from both parts."""
    from scipy.stats import norm

    def design(X):
        X = np.asarray(X, float).reshape(len(X), -1)
        return np.column_stack([np.ones(len(X)), X])
    Xl, Xu = design(X_lab), design(X_unlab)
    yl, fl, fu = (np.asarray(v, float) for v in (y_lab, f_lab, f_unlab))
    n, N = len(yl), len(fu)
    Hu = Xu.T @ Xu / N
    Hinv = np.linalg.inv(Hu)
    theta_f = np.linalg.solve(Xu.T @ Xu, Xu.T @ fu)
    delta = np.linalg.solve(Xu.T @ Xu / N, Xl.T @ (fl - yl) / n)
    theta = theta_f - delta
    gu = Xu * (Xu @ theta - fu)[:, None]
    gl = Xl * (fl - yl - Xl @ np.zeros(Xl.shape[1]))[:, None]
    V = Hinv @ (np.cov(gu.T) / N + np.cov(gl.T) / n) @ Hinv
    se = np.sqrt(np.diag(V))
    z = norm.ppf(1 - alpha / 2)
    naive = theta_f
    Vn = Hinv @ (np.cov((Xu * (Xu @ naive - fu)[:, None]).T) / N) @ Hinv
    sen = np.sqrt(np.diag(Vn))
    return {"ppi": theta.tolist(), "ppi_ci": np.column_stack([theta - z * se, theta + z * se]).tolist(),
            "naive": naive.tolist(), "naive_ci": np.column_stack([naive - z * sen, naive + z * sen]).tolist(),
            "n_labelled": n, "n_unlabelled": N}


# ----------------------------------------------------------------------------- 7b diagnosis (protocol v02)
def between_patient_share(state, groups) -> float:
    """Share of the state score's variance carried by patient means (one-way decomposition, spot-weighted).
    A state with a share near 0 has no patient-level quantity for a bulk reference to proxy."""
    s = np.asarray(state, float)
    g = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
    ok = np.isfinite(s)
    s, g = s[ok], g[ok]
    means = np.bincount(g, s) / np.bincount(g)
    tot = float(((s - s.mean()) ** 2).sum())
    return float(((means[g] - s.mean()) ** 2).sum() / tot) if tot > 0 else 0.0


def preparation_split(pred: dict, ref, prep: dict) -> dict:
    """Spearman of slide predictions against a bulk reference within each preparation class."""
    out = {}
    for cls in sorted(set(prep.values())):
        ids = [i for i in pred if prep.get(i) == cls and i in ref.index]
        out[str(cls)] = {"n": len(ids), "spearman": spearman([pred[i] for i in ids], ref[ids].values) if len(ids) > 3 else None}
    return out


def diagnosis_7b(E, state, groups, *, slide_preds: dict, refs: dict, prep: dict, n_perm: int = 3) -> dict:
    """Protocol v02 item 7b: (i) held-out-patient spot Spearman of the ridge student; (ii) between-patient
    variance share of the state; (iii) slide-cohort agreement split by preparation class."""
    ho = heldout_spot_eval(RidgeStudent, E, state, groups, n_perm=n_perm)
    return {"i_spot_level_heldout": {k: v for k, v in ho.items() if k != "_pred"},
            "ii_between_patient_share": between_patient_share(state, groups),
            "iii_preparation_split": {r: preparation_split(slide_preds, ref, prep) for r, ref in refs.items()}}
