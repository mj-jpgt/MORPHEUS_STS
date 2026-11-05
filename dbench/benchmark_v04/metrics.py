"""Table 1 metrics and statistics (spec 4.1, 5.2) from label and score caches.

    python -m dbench.benchmark_v04.metrics table STUDY --methods a,b,c [--level 95] [--baseline known]

Per (study, target): metric per split (tie-broken at random, 5 jitter seeds), then the mean over splits; per study the
mean over targets.  With fewer than 3 studies the hierarchical bootstrap resamples targets within the study (2,000
replicates) and the studies are reported separately.  Paired difference = ours minus baseline on each replicate; Holm
across baselines within each primary endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dbench.benchmark_v04 import tables as T

ROOT = T.ROOT
N_BOOT = 2000
DISPLAY = {"sel_screen": "ours: support-fit ordering (pairs = mean of singles)",
           "sel_screen_joint": "ours (joint pair support gain), supplementary", "random": "random (20 seeds)",
           "linear_screen": "linear interaction screen", "ebm": "EBM", "gbm_shap": "GBM + SHAP interactions",
           "gbm_h": "GBM + Friedman H", "rulefit": "RuleFit", "stability_selection": "stability selection"}


def _jitter_metric(y, s, fn, seeds=5):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1.0 if np.isfinite(s).any() else 0.0)
    span = max(float(np.ptp(s)), 1e-12)
    return float(np.mean([fn(y, s + np.random.default_rng(i).random(len(s)) * 1e-9 * span) for i in range(seeds)]))


def load_scores(study, method, target):
    h = hashlib.sha256(target.encode()).hexdigest()[:10]
    p = ROOT / "scores" / study / method / f"{h}.npz"
    if not p.exists():
        return None
    return np.load(p)


def per_target_metrics(study, method, level="95", B="known", subset=False):
    """{target: {kind: {"ap": , "auroc": , "prev": }}} mean over splits."""
    targets = json.loads((ROOT / "labels" / study / "targets.json").read_text())["targets"]
    out = {}
    for t in targets:
        lab = T.load_target(study, t)
        sc = load_scores(study, method, t)
        if lab is None or sc is None:
            continue
        meta, z = lab
        rec = {}
        for kind in ("single", "pair"):
            aps, aus, prev = [], [], []
            for s in T.splits_of(meta):
                if subset and s not in T.slow_subset(study, t):
                    continue
                y = T.label(z, s, kind, B, level)
                if method == "random":
                    seeds = sorted({k.split("|")[2] for k in sc.files if k.startswith(f"s{s}|{kind}|")})
                    vals = [(_jitter_metric(y, sc[f"s{s}|{kind}|{k}"], average_precision_score, 1),
                             _jitter_metric(y, sc[f"s{s}|{kind}|{k}"], roc_auc_score, 1)) for k in seeds]
                    ap, au = (np.nanmean([v[0] for v in vals]), np.nanmean([v[1] for v in vals])) if vals else (np.nan, np.nan)
                else:
                    key = f"s{s}|{kind}"
                    if key not in sc.files:
                        continue
                    ap = _jitter_metric(y, sc[key], average_precision_score)
                    au = _jitter_metric(y, sc[key], roc_auc_score)
                aps.append(ap)
                aus.append(au)
                prev.append(y.mean())
            rec[kind] = {"ap": float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan,
                         "auroc": float(np.nanmean(aus)) if np.isfinite(aus).any() else np.nan,
                         "prev": float(np.mean(prev)) if prev else np.nan, "n_splits": len(aps)}
        out[t] = rec
    return out


def boot_diff(a, b, rng_idx):
    """mean over bootstrap target resamples of (a - b)."""
    return (a[rng_idx] - b[rng_idx]).mean(1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def table(study, methods, level="95", B="known", ours="sel_screen", seed=0, subset=False):
    pm = {m: per_target_metrics(study, m, level, B, subset) for m in methods}
    rng = np.random.default_rng(seed)
    lines, res = [], {}
    prev = {}
    for kind in ("single", "pair"):
        for metric in ("ap", "auroc"):
            common = sorted(set.intersection(*[{t for t, r in d.items() if np.isfinite(r[kind][metric])} for d in pm.values()]))
            if not common:
                continue
            M = np.array([[pm[m][t][kind][metric] for t in common] for m in methods])
            idx = rng.integers(0, len(common), size=(N_BOOT, len(common)))
            prev[kind] = float(np.mean([pm[methods[0]][t][kind]["prev"] for t in common]))
            for i, m in enumerate(methods):
                bs = M[i][idx].mean(1)
                res.setdefault(m, {})[f"{metric}_{kind}"] = {"mean": float(M[i].mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)), "n_targets": len(common)}
            if ours in methods:
                oi = methods.index(ours)
                ps, rows = [], []
                for i, m in enumerate(methods):
                    if m == ours:
                        continue
                    d = M[oi][idx].mean(1) - M[i][idx].mean(1)
                    ps.append(float(np.mean(d <= 0)))
                    rows.append((m, float(d.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))))
                adj = holm(ps) if ps else []
                for (m, dm, lo, hi), p, pa in zip(rows, ps, adj):
                    res[m][f"{metric}_{kind}"].update({"diff_ours_minus": dm, "diff_lo": lo, "diff_hi": hi, "p_one_sided": p, "p_holm": float(pa)})
    f = lambda d, k: "-" if (d is None or k not in d) else f"{d[k]['mean']:.3f} [{d[k]['lo']:.3f}, {d[k]['hi']:.3f}]"  # noqa: E731
    g = lambda d, k: "-" if (d is None or k not in d or "diff_ours_minus" not in d[k]) else f"{d[k]['diff_ours_minus']:+.3f} [{d[k]['diff_lo']:+.3f}, {d[k]['diff_hi']:+.3f}] p_Holm {d[k]['p_holm']:.3f}"  # noqa: E731
    lines.append(f"### {study}: RFF labels, level {level}%, baseline B_{B}; prevalence singles {prev.get('single', float('nan')):.4f}, pairs {prev.get('pair', float('nan')):.4f}; "
                 f"targets {max((v.get('ap_single', {}).get('n_targets', 0) for v in res.values()), default=0)}; 95% bootstrap over targets (2,000), per-study (fewer than 3 studies)\n")
    lines.append("| method | AP singles | AP lift singles | AP pairs | AP lift pairs | AUROC singles | AUROC pairs | ours minus method, AP singles | ours minus method, AP pairs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        d = res.get(m, {})
        lift = lambda k, pk: "-" if k not in d or not prev.get(pk) else f"{d[k]['mean'] / prev[pk]:.1f}"  # noqa: E731
        lines.append(f"| {DISPLAY.get(m, m)} | {f(d, 'ap_single')} | {lift('ap_single', 'single')} | {f(d, 'ap_pair')} | {lift('ap_pair', 'pair')} | {f(d, 'auroc_single')} | {f(d, 'auroc_pair')} | {g(d, 'ap_single')} | {g(d, 'ap_pair')} |")
    text = "\n".join(lines) + "\n"
    tag = "_slowsubset" if subset else ""
    out = ROOT / f"table1_{study}_L{level}_{B}{tag}.md"
    out.write_text(text)
    (ROOT / f"table1_{study}_L{level}_{B}{tag}.json").write_text(json.dumps({"prev": prev, "rows": res}, indent=1))
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("table",))
    ap.add_argument("study")
    ap.add_argument("--methods", default="sel_screen,sel_screen_joint,random,linear_screen")
    ap.add_argument("--level", default="95")
    ap.add_argument("--baseline", default="known")
    ap.add_argument("--subset", action="store_true", help="restrict every method to the slow-baseline split subset")
    a = ap.parse_args(argv)
    print(table(a.study, a.methods.split(","), a.level, a.baseline, subset=a.subset))


if __name__ == "__main__":
    main()
