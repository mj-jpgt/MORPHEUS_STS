"""Diagnostics for learned-vs-screen (development, 40 targets x 3 folds, B_known, identical core.support_only inputs).

pass   : per (target, fold), realised held-out B_known gain pieces of every pool single and pair on the fold's query
         patients (core.score_sets: per-query-patient numerators, SST, exact support gain), GPU, cached per fold-task.
report : (1) oracle-input: is the network's predicted shrinkage mu = G_sup - score informative about realised shrinkage
         (G_sup - realised gain)?  (2) label-noise ceiling: query patients pooled over folds split into random halves
         (5 draws); Spearman of realised gain between halves and AUROC of one half's label from the other half's gain,
         beside the screen's and the network's AUROC against the same half labels and Spearman against full-query gain.

    python -m dbench.harness.diag_noise pass
    python -m dbench.harness.diag_noise report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

B = Path("runs/development/benchmark_v03")
OUT = B / "diag_noise"


def run_pass() -> None:
    import torch
    from . import core
    from .pool import dev_folds, target_pools
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(4096 / (torch.cuda.get_device_properties(0).total_memory / 2**20))
    OUT.mkdir(parents=True, exist_ok=True)
    spec, built = dev_folds()
    pools = target_pools(built)
    for i, (t, f, conds, names, st, q) in enumerate(built):
        path = OUT / f"{t.replace(':', '_').replace('/', '_')}__{f}.npz"
        if path.exists():
            continue
        tk = dict(conds)["real"]
        idx = {n: j for j, n in enumerate(names)}
        singles, pairs = pools[t]
        sets = [(idx[n],) for n in singles] + [(idx[a], idx[b]) for a, b in pairs]
        num, sst, gsup = core.score_sets(tk, sets, Bs=("known",), device="cuda")["known"]
        qp = np.unique(np.asarray(tk.groups)[tk.s:]).astype(str)
        np.savez(path, num=num, sst=sst, gsup=gsup, n_single=len(singles), qpatients=qp)
        print(f"{t}|{f} done {i + 1}/{len(built)}", flush=True)


def _auroc(y, s):
    from .pool import _ap_auc
    return _ap_auc(np.asarray(y, bool), np.asarray(s, float), seeds=5)[0]


def _spear(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).correlation) if ok.sum() > 3 else np.nan


def report() -> str:
    from .dev_flags import set_flags
    from .public_metrics import boot_mean, labels_pooled
    lab = json.loads((B / "pool" / "labels.json").read_text())
    ours = json.loads((B / "pool_v2" / "scores_g_theta_known.json").read_text())["scores"]
    screen = json.loads((B / "pool_v2" / "scores_sel_screen_known.json").read_text())["scores"]
    joint = json.loads((B / "pool_v2" / "scores_sel_screen_joint_known.json").read_text())["scores"]
    rows = {k: [] for k in ("mu_vs_true_shrink_single", "mu_vs_gsup_single", "true_shrink_vs_gsup_single",
                            "half_spearman", "half_auroc", "screen_half_auroc", "ours_half_auroc",
                            "screen_full_spearman", "ours_full_spearman")}
    rng = np.random.default_rng(0)
    for t in lab["pools"]:
        singles, pairs = lab["pools"][t]["singles"], lab["pools"][t]["pairs"]
        members = [[s] for s in singles] + list(pairs)
        flagged = np.array([set_flags(t, m)["any"] for m in members])
        nums, ssts, gs, sc_s, sc_o = [], [], [], [], []
        for f in range(3):
            p = OUT / f"{t.replace(':', '_').replace('/', '_')}__{f}.npz"
            if not p.exists():
                break
            z = np.load(p)
            nums.append(z["num"]), ssts.append(z["sst"]), gs.append(z["gsup"])
            k = f"{t}|{f}"
            sc_s.append(np.concatenate([screen[k]["single"], joint[k]["pair"]]))
            sc_o.append(np.concatenate([ours[k]["single"], ours[k]["pair"]]))
        if len(nums) < 3:
            continue
        num = np.concatenate(nums, 0)            # (P_all query patients pooled over folds, members)
        sst = np.concatenate(ssts, 0)
        num = np.where(np.isfinite(num), num, 0.0)
        gain_full = num.sum(0) / max(sst.sum(), 1e-12)
        ns = len(singles)
        # (1) oracle input, singles: mu = screen - ours (G_sup - (G_sup - mu)); realised shrinkage = G_sup - realised gain
        for f in range(3):
            g = gs[f][:ns]
            mu = sc_s[f][:ns] - sc_o[f][:ns]
            realised = nums[f][:, :ns].sum(0) / max(ssts[f].sum(), 1e-12)
            rows["mu_vs_true_shrink_single"].append(_spear(mu, g - realised))
            rows["mu_vs_gsup_single"].append(_spear(mu, g))
            rows["true_shrink_vs_gsup_single"].append(_spear(g - realised, g))
        # (2) label-noise ceiling over query patients pooled across folds; strict labels
        blocks = np.cumsum([0] + [n.shape[0] for n in nums])
        for d in range(5):   # halves drawn within each fold's query patients, pooled over folds (as the labels)
            A, Bh = [], []
            for f in range(3):
                perm = blocks[f] + rng.permutation(blocks[f + 1] - blocks[f])
                A += list(perm[: len(perm) // 2]); Bh += list(perm[len(perm) // 2:])
            A, Bh = np.array(A), np.array(Bh)
            gA = num[A].sum(0) / max(sst[A].sum(), 1e-12)
            gB = num[Bh].sum(0) / max(sst[Bh].sum(), 1e-12)
            yB = labels_pooled([(num[Bh], sst[Bh])]) & ~flagged
            rows["half_spearman"].append(_spear(gA, gB))
            if 0 < yB.sum() < len(yB):
                rows["half_auroc"].append(_auroc(yB, gA))
                rows["screen_half_auroc"].append(np.mean([_auroc(yB, s) for s in sc_s]))
                rows["ours_half_auroc"].append(np.mean([_auroc(yB, s) for s in sc_o]))
        rows["screen_full_spearman"].append(np.mean([_spear(s, gain_full) for s in sc_s]))
        rows["ours_full_spearman"].append(np.mean([_spear(s, gain_full) for s in sc_o]))
    res = {k: boot_mean(v) for k, v in rows.items()}
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    return json.dumps(res, indent=1)


if __name__ == "__main__":
    print(report() if sys.argv[1] == "report" else run_pass())
