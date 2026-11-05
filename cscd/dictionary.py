"""Stage 2: sparse dictionaries over the residual space.

A TopK sparse autoencoder rather than L1, following current SAE practice: the
sparsity level is exact and set by construction, so it does not drift with the
reconstruction scale the way an L1 penalty does.

The dictionary is deliberately overcomplete. That is what lets a state be a
*combination* of atoms instead of a single direction, which is the
representational property Prop. 2 turns on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SAEConfig:
    n_atoms: int = 64
    k: int = 8                 # exact number of active atoms per unit
    lr: float = 1e-3
    epochs: int = 200
    batch: int = 512
    tied: bool = False
    seed: int = 0
    dead_revive_every: int = 25   # 0 disables
    l1_lambda: float = 3e-3       # JumpReLUSAE only; ignored by TopKSAE


class TopKSAE(nn.Module):
    """r -> z (k-sparse) -> r_hat."""

    def __init__(self, d_in, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.enc = nn.Linear(d_in, cfg.n_atoms, bias=True)
        self.dec = nn.Linear(cfg.n_atoms, d_in, bias=True)
        if cfg.tied:
            self.dec.weight = nn.Parameter(self.enc.weight.t().contiguous())
        self.pre_bias = nn.Parameter(torch.zeros(d_in))

    def encode(self, r):
        a = self.enc(r - self.pre_bias)
        if self.cfg.k >= a.shape[1]:
            return torch.relu(a)
        val, idx = torch.topk(a, self.cfg.k, dim=1)
        z = torch.zeros_like(a)
        z.scatter_(1, idx, torch.relu(val))
        return z

    def forward(self, r):
        z = self.encode(r)
        return self.dec(z) + self.pre_bias, z

    def fit(self, R, device="cpu", verbose=False):
        cfg = self.cfg
        self.to(device)
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
        with torch.no_grad():
            self.pre_bias.copy_(R.mean(0))
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        n = R.shape[0]
        hist = []
        for ep in range(cfg.epochs):
            perm = torch.randperm(n, generator=g).to(device)
            tot = nb = 0.0
            fired = torch.zeros(cfg.n_atoms, device=device)
            for i in range(0, n, cfg.batch):
                rb = R[perm[i:i + cfg.batch]]
                opt.zero_grad(set_to_none=True)
                rhat, z = self(rb)
                loss = ((rhat - rb) ** 2).mean()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
                fired += (z > 0).float().sum(0)
            dead = int((fired == 0).sum())
            hist.append({"epoch": ep, "mse": tot / nb, "dead_atoms": dead})
            if cfg.dead_revive_every and dead and (ep + 1) % cfg.dead_revive_every == 0:
                self._revive(fired == 0, R, device)
            if verbose and ep % 50 == 0:
                print(f"  ep {ep:3d} mse {tot/nb:.5f} dead {dead}")
        self.history_ = hist
        return self

    @torch.no_grad()
    def _revive(self, dead_mask, R, device):
        """Re-seed dead atoms on the worst-reconstructed units.

        Without this the effective dictionary silently shrinks and atom counts
        stop meaning what the config says they mean.
        """
        if not bool(dead_mask.any()):
            return
        rhat, _ = self(R)
        err = ((R - rhat) ** 2).sum(1)
        n_dead = int(dead_mask.sum())
        worst = torch.topk(err, min(n_dead, R.shape[0])).indices
        v = R[worst] - self.pre_bias
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)[:v.shape[0]]
        self.enc.weight[idx] = v
        self.enc.bias[idx] = 0.0
        self.dec.weight[:, idx] = v.t()

    @torch.no_grad()
    def transform(self, R, device="cpu"):
        self.to(device).eval()
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
        z = self.encode(R).cpu().numpy()
        self.train()
        return z

    @torch.no_grad()
    def atom_stats(self, R, device="cpu"):
        """Per-atom firing rate and mean magnitude, as structure."""
        z = self.transform(R, device=device)
        return {
            "n_atoms": int(z.shape[1]),
            "fire_rate": z.__gt__(0).mean(0).tolist(),
            "mean_abs": np.abs(z).mean(0).tolist(),
            "n_dead": int((z > 0).sum(0).__eq__(0).sum()),
            "mean_active_per_unit": float((z > 0).sum(1).mean()),
        }


class JumpReLUSAE(nn.Module):
    """r -> z (JumpReLU-gated) -> r_hat.

    Same encoder/decoder shape as `TopKSAE` (Linear-ReLU-Linear + pre_bias),
    differing only in how the code is sparsified: a per-atom LEARNED
    threshold theta_j = softplus(log_theta_j) rather than an exact top-k op.
    There is no exact-k operation for a threshold gate, so k is matched in
    the mean rather than exactly per row -- see `calibrate_threshold`, which
    implements the identical rule already used in
    `cscd.e3e_activations.JumpReLUAct`, copied rather than re-derived, and
    predeclared in `cscd/PREDECLARATION_jumprelu_disco.md` before any run at
    the headline configuration.

    `encode` branches on `self.training` (the standard nn.Module flag) so it
    is a drop-in replacement for `TopKSAE.encode` with the SAME zero-argument
    signature: during training (`.train()`, the default) it uses a steep
    sigmoid surrogate gate so gradients reach which atoms cross threshold;
    at evaluation (`.eval()`, entered by `transform`) it uses the hard
    calibrated threshold. This means callers that already call `.encode(x)`
    on a `TopKSAE` (e.g. `cscd.pipeline._train_dictionaries`'s joint
    training loop, its adaptive replication mask, its CCA-MI dual-ascent
    measurement) work unmodified when a `JumpReLUSAE` is substituted.
    """

    def __init__(self, d_in, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.enc = nn.Linear(d_in, cfg.n_atoms, bias=True)
        self.dec = nn.Linear(cfg.n_atoms, d_in, bias=True)
        if cfg.tied:
            self.dec.weight = nn.Parameter(self.enc.weight.t().contiguous())
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.log_theta = nn.Parameter(torch.zeros(cfg.n_atoms))
        self.threshold_ = None   # calibrated post-hoc, on the fit split only

    def _theta(self):
        return torch.nn.functional.softplus(self.log_theta)

    def encode(self, r):
        a = torch.relu(self.enc(r - self.pre_bias))
        if self.training:
            theta = self._theta()
            gate = torch.sigmoid((a - theta) * 20.0)   # smooth surrogate
            return a * gate
        theta = self.threshold_ if self.threshold_ is not None else self._theta().detach()
        return a * (a > theta).float()

    def forward(self, r):
        z = self.encode(r)
        return self.dec(z) + self.pre_bias, z

    def fit(self, R, device="cpu", verbose=False):
        """Standalone training (unused by the DISCO joint pipeline, which
        runs its own epoch loop directly against `.encode`/forward -- see
        `cscd.pipeline._train_dictionaries`). Kept for parity with
        `TopKSAE.fit`, e.g. for a future independent `jumprelu_sae`
        baseline, and calibrates the threshold at the end of training so a
        standalone fit is immediately usable via `.transform`.
        """
        cfg = self.cfg
        self.to(device)
        self.train()
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
        with torch.no_grad():
            self.pre_bias.copy_(R.mean(0))
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        n = R.shape[0]
        hist = []
        for ep in range(cfg.epochs):
            perm = torch.randperm(n, generator=g).to(device)
            tot = nb = 0.0
            fired = torch.zeros(cfg.n_atoms, device=device)
            for i in range(0, n, cfg.batch):
                rb = R[perm[i:i + cfg.batch]]
                opt.zero_grad(set_to_none=True)
                rhat, z = self(rb)
                loss = ((rhat - rb) ** 2).mean() + cfg.l1_lambda * z.abs().sum(1).mean()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
                fired += (z > 0).float().sum(0)
            dead = int((fired == 0).sum())
            hist.append({"epoch": ep, "mse": tot / nb, "dead_atoms": dead})
            if cfg.dead_revive_every and dead and (ep + 1) % cfg.dead_revive_every == 0:
                self._revive(fired == 0, R, device)
            if verbose and ep % 50 == 0:
                print(f"  ep {ep:3d} mse {tot/nb:.5f} dead {dead}")
        self.history_ = hist
        self.calibrate_threshold(R, cfg.k, device)
        return self

    @torch.no_grad()
    def _revive(self, dead_mask, R, device):
        """Re-seed dead atoms on the worst-reconstructed units, identical
        mechanism to `TopKSAE._revive` plus resetting the revived atoms'
        threshold so a stale large theta cannot keep a freshly re-seeded
        direction from ever firing."""
        if not bool(dead_mask.any()):
            return
        rhat, _ = self(R)
        err = ((R - rhat) ** 2).sum(1)
        n_dead = int(dead_mask.sum())
        worst = torch.topk(err, min(n_dead, R.shape[0])).indices
        v = R[worst] - self.pre_bias
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)[:v.shape[0]]
        self.enc.weight[idx] = v
        self.enc.bias[idx] = 0.0
        self.dec.weight[:, idx] = v.t()
        self.log_theta[idx] = 0.0

    @torch.no_grad()
    def calibrate_threshold(self, R, target_k, device):
        """Predeclared rule (`cscd/PREDECLARATION_jumprelu_disco.md`),
        identical to `cscd.e3e_activations.JumpReLUAct.calibrate_threshold`:
        pick ONE global scalar offset on top of the learned per-atom
        thresholds so the mean active count over R equals target_k. `R` may
        be a tensor already on `device` (the DISCO joint-training path) or
        an array-like (the standalone `.fit` path); both are accepted."""
        self.eval()
        if isinstance(R, torch.Tensor):
            Rt = R.to(device=device, dtype=torch.float32)
        else:
            Rt = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
        a = torch.relu(self.enc(Rt - self.pre_bias))
        base = self._theta()
        lo, hi = -5.0, 5.0
        for _ in range(40):
            mid = (lo + hi) / 2
            theta = torch.clamp(base + mid, min=0.0)
            active = (a > theta).float().sum(1).mean().item()
            if active > target_k:
                lo = mid
            else:
                hi = mid
        self.threshold_ = torch.clamp(base + (lo + hi) / 2, min=0.0)
        self.train()

    @torch.no_grad()
    def transform(self, R, device="cpu"):
        self.to(device).eval()
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
        z = self.encode(R).cpu().numpy()
        self.train()
        return z

    @torch.no_grad()
    def atom_stats(self, R, device="cpu"):
        """Per-atom firing rate and mean magnitude, as structure."""
        z = self.transform(R, device=device)
        return {
            "n_atoms": int(z.shape[1]),
            "fire_rate": z.__gt__(0).mean(0).tolist(),
            "mean_abs": np.abs(z).mean(0).tolist(),
            "n_dead": int((z > 0).sum(0).__eq__(0).sum()),
            "mean_active_per_unit": float((z > 0).sum(1).mean()),
        }
