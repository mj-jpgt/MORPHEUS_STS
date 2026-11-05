"""Stage 1: remove known structure from a frozen embedding.

Two implementations, because the comparison between them is the empirical
content of Prop. 1 and is a mandatory ablation, not an optional one:

  ridge_residualize      the linear control. Exactly the MORPHEUS estimator:
                         cross-fitted ridge on the declared covariates.
  AdversarialResidualizer  a network that keeps h reconstructable from r while
                         a nonlinear probe fails to recover C from r.

Leakage is *measured*, never assumed, by fitting a fresh probe on held-out
units and reporting how much of C it recovers. A residualiser is only as good
as what an independent probe can still find.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------- linear ----
def ridge_residualize(H, C, alphas=(0.1, 1.0, 10.0), n_folds=5, seed=0):
    """Cross-fitted ridge residualisation, the linear control arm.

    Coefficients for a fold are fit on the other folds only, so no unit
    contributes to the model that residualises it.

    Returns (R, info) where info records the alpha chosen per fold, so the
    caller can assert on the structure rather than only the residual.
    """
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    n, p = C.shape
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % n_folds

    Cc = np.hstack([C, np.ones((n, 1))])          # explicit intercept column
    R = np.empty_like(H)
    chosen = []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        Xtr, Ytr = Cc[tr], H[tr]
        best_a, best_err = None, np.inf
        # inner split of the training folds to pick alpha
        inner = rng.permutation(Xtr.shape[0]) % 2
        for a in alphas:
            i_tr, i_va = inner == 0, inner == 1
            if i_tr.sum() < 2 or i_va.sum() < 2:
                continue
            G = Xtr[i_tr].T @ Xtr[i_tr] + a * np.eye(p + 1)
            B = np.linalg.solve(G, Xtr[i_tr].T @ Ytr[i_tr])
            err = float(np.mean((Ytr[i_va] - Xtr[i_va] @ B) ** 2))
            if err < best_err:
                best_a, best_err = a, err
        best_a = alphas[0] if best_a is None else best_a
        G = Xtr.T @ Xtr + best_a * np.eye(p + 1)
        B = np.linalg.solve(G, Xtr.T @ Ytr)
        R[te] = H[te] - Cc[te] @ B
        chosen.append({"fold": f, "alpha": best_a, "n_test": int(te.sum())})
    return R, {"per_fold": chosen, "alphas": list(alphas), "n_folds": n_folds}


# ---------------------------------------------------------- adversarial ----
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def _mlp(din, dh, dout, depth=2, act=nn.GELU):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, dh), act()]
        d = dh
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)


@dataclass
class ResidConfig:
    d_hidden: int = 256
    d_resid: int = 64
    probe_hidden: int = 256
    depth: int = 2
    lambda_adv: float = 1.0
    w_recon: float = 1.0
    lr: float = 1e-3
    probe_lr: float = 1e-3
    probe_steps: int = 3          # probe updates per residualiser update
    epochs: int = 60
    batch: int = 512
    seed: int = 0


class AdversarialResidualizer(nn.Module):
    """h -> r, keeping h reconstructable while a probe cannot recover C.

    The probe is retrained against the current representation on a schedule
    (`probe_steps` inner updates per outer update), which is what stops the
    residualiser from beating a stale adversary.
    """

    def __init__(self, d_in, d_concept, cfg: ResidConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = _mlp(d_in, cfg.d_hidden, cfg.d_resid, cfg.depth)
        self.dec = _mlp(cfg.d_resid, cfg.d_hidden, d_in, cfg.depth)
        self.probe = _mlp(cfg.d_resid, cfg.probe_hidden, d_concept, cfg.depth)

    def forward(self, h):
        return self.enc(h)

    def fit(self, H, C, device="cpu", verbose=False):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.to(device)
        H = torch.as_tensor(np.asarray(H), dtype=torch.float32, device=device)
        C = torch.as_tensor(np.asarray(C), dtype=torch.float32, device=device)
        n = H.shape[0]

        body = list(self.enc.parameters()) + list(self.dec.parameters())
        opt = torch.optim.Adam(body, lr=cfg.lr)
        opt_p = torch.optim.Adam(self.probe.parameters(), lr=cfg.probe_lr)
        mse = nn.MSELoss()
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)

        hist = []
        for ep in range(cfg.epochs):
            perm = torch.randperm(n, generator=g).to(device)
            tot_r = tot_p = nb = 0.0
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch]
                hb, cb = H[idx], C[idx]

                # --- probe catches up against the current representation ---
                for _ in range(cfg.probe_steps):
                    opt_p.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        rb = self.enc(hb)
                    lp = mse(self.probe(rb), cb)
                    lp.backward()
                    opt_p.step()

                # --- residualiser: reconstruct h, defeat the probe ---------
                opt.zero_grad(set_to_none=True)
                rb = self.enc(hb)
                l_rec = mse(self.dec(rb), hb)
                l_adv = mse(self.probe(grad_reverse(rb, cfg.lambda_adv)), cb)
                (cfg.w_recon * l_rec + l_adv).backward()
                opt.step()

                tot_r += float(l_rec)
                tot_p += float(lp)
                nb += 1
            hist.append({"epoch": ep, "recon": tot_r / nb, "probe": tot_p / nb})
            if verbose and ep % 10 == 0:
                print(f"  ep {ep:3d} recon {tot_r/nb:.4f} probe {tot_p/nb:.4f}")
        self.history_ = hist
        return self

    @torch.no_grad()
    def transform(self, H, device="cpu"):
        self.to(device).eval()
        H = torch.as_tensor(np.asarray(H), dtype=torch.float32, device=device)
        out = self.enc(H).cpu().numpy()
        self.train()
        return out


# -------------------------------------------------------------- leakage ----
def concept_leakage(R, C, split, hidden=256, epochs=150, lr=1e-3,
                    device="cpu", seed=0):
    """How much of C an independent probe still recovers from R.

    Trained on `split == 0`, scored on `split == 1`, reported as held-out R^2
    averaged over concept columns. This is the quantity Prop. 1 predicts will
    be strictly positive for linear residualisation whenever C acts
    nonlinearly, and near zero for the adversarial version.

    Returns a dict so callers assert on structure, not on one float.
    """
    torch.manual_seed(seed)
    R = np.asarray(R, dtype=np.float64)
    # Standardise on TRAIN statistics only, then apply to both halves. An
    # unstandardised residual can have wildly different column scales, which
    # is what left the probe undertrained and reporting R^2 = -0.87 -- a
    # broken probe, not an absence of leakage.
    split = np.asarray(split)
    tr_m, te_m = split == 0, split == 1
    mu, sd = R[tr_m].mean(0), R[tr_m].std(0)
    R = (R - mu) / np.maximum(sd, 1e-8)

    R = torch.as_tensor(R, dtype=torch.float32, device=device)
    C = torch.as_tensor(np.asarray(C), dtype=torch.float32, device=device)
    tr = torch.as_tensor(tr_m, device=device)
    te = torch.as_tensor(te_m, device=device)

    probe = _mlp(R.shape[1], hidden, C.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    mse = nn.MSELoss()
    Rtr, Ctr = R[tr], C[tr]
    n = Rtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch = min(512, n)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            mse(probe(Rtr[idx]), Ctr[idx]).backward()
            opt.step()

    with torch.no_grad():
        pred = probe(R[te])
        resid = ((C[te] - pred) ** 2).mean(0)
        var = C[te].var(0, unbiased=False)
        raw = (1.0 - resid / var.clamp_min(1e-12)).cpu().numpy()
    # A negative R^2 means the probe did worse than predicting the mean, i.e.
    # it found nothing. Clamp at 0 so "leakage" is a non-negative quantity and
    # a failed probe cannot read as negative evidence.
    r2 = np.maximum(raw, 0.0)
    return {
        "r2_per_concept": r2.tolist(),
        "r2_raw_per_concept": raw.tolist(),
        "r2_mean": float(np.mean(r2)),
        "r2_max": float(np.max(r2)),
        "probe_underfit": bool(np.any(raw < -0.05)),
        "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
    }
