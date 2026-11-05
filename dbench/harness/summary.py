"""Tables and figures from the harness's controlled-world rows (spec 3.3 Q1, 3.5 Q2).

``write_q1(out)``: per method, net yield at k in {1, 5, 10, 20} under B_nuis / B_known / B_decomp
(mean over worlds, 95 % world-bootstrap interval), held-out families separately from training
families, complete-object recovery, constituent recall, reciprocal rank, top-set vs planted
higher-order gain, sets registered, calibration (Spearman, ECE) for predicting methods;
figure: net yield @10 and recovery @10 against signal fraction per method.
``write_q2(out, full, others)``: paired per-world differences in net yield (and recovery) between the
full model and each comparison method, 2,000-draw world bootstrap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KS = (1, 5, 10, 20)
LABELS = {"rulefit": "RuleFit (scikit-learn reimplementation; imodels 3.0.4 segfaults in its Lasso alpha search)",
          "screen": "support-fit screen (baseline 11)", "lopo_heuristic": "G_sup - LOPO SD (baseline 10)",
          "stat_gbm": "GBM on diagnostics (baseline 9)", "tabdistill": "TabDistill (TabPFN v2 + ProxySPEX)",
          "gbm_h": "gradient boosting + Friedman H", "ebm": "EBM (GA2M)", "linear_screen": "marginal + pairwise screen",
          "pysr": "PySR (256 shortlist)", "residual_gen": "MORPHEUS residual generator", "random": "random search (K)"}
BS = ("nuis", "known", "decomp")


def load(out: Path) -> list:
    rows, seen = [], set()
    for p in sorted(Path(out).glob("rows_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" in r:
                    continue
                if (r["world"], r["method"]) not in seen:
                    seen.add((r["world"], r["method"]))
                    rows.append(r)
    return rows


def ny(r: dict, B: str, k: int) -> float | None:
    d = (r.get("net_yield") or {}).get(B) or {}
    v = d.get(str(k), d.get(k))
    return None if v is None else v.get("net")


def boot(v, n_boot: int = 2000, seed: int = 0) -> tuple:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return (None, None, None)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fmt(t, nd=2) -> str:
    return "-" if t is None or t[0] is None else f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def calib(rows: list) -> dict:
    from scipy.stats import spearmanr
    pr = np.array([[c[0], c[1]] for r in rows for c in r.get("calibration_pairs", []) if c[0] is not None], float)
    if len(pr) < 3:
        return {"spearman": None, "ece": None}
    bins = np.array_split(np.argsort(pr[:, 0]), min(10, len(pr)))
    ece = float(sum(len(b) * abs(pr[b, 0].mean() - pr[b, 1].mean()) for b in bins) / len(pr))
    return {"spearman": float(spearmanr(pr[:, 0], pr[:, 1]).correlation), "ece": ece}


def write_q1(out: Path) -> str:
    out = Path(out)
    rows = load(out)
    methods = sorted({r["method"] for r in rows})
    lines = [f"# Q1: controlled worlds through the harness ({len({r['world'] for r in rows})} worlds)", "",
             "Net yield = discoveries (pooled query-patient interval lower bound > 0) in the top k on the real "
             "world minus the mean over the two null versions (patient permutation, within-patient candidate "
             "scramble), same method; mean over worlds with 95 % world-bootstrap intervals.",
             "Note on B_nuis: a controlled target is a real column's residual after the programme scores plus the "
             "planted signal; the residual keeps real structure that any reasonable top 10 predicts on held-out "
             "patients, so B_nuis net yield sits at its ceiling for most methods (scorer check: the 10 lowest-support "
             "singles score -0.5, the 10 highest 10.0 on gp_2 | 0.5 % | s0). Planted-object metrics and B_known / "
             "B_decomp separate methods.", ""]
    summary = {}
    for part, sel in (("held-out families", lambda r: r["heldout_family"]), ("training families", lambda r: not r["heldout_family"]),
                      ("all worlds", lambda r: True)):
        lines += [f"## {part}", "", "| method | worlds | recovered@10 | constituent recall@10 | MRR | top-set higher-order "
                  "gain | net@10 B_known | net@10 B_decomp | net@1 / 5 / 10 / 20 B_nuis (ceiling, see note) | sets |",
                  "|---" * 10 + "|"]
        for m in methods:
            rs = [r for r in rows if r["method"] == m and sel(r)]
            if not rs:
                continue
            d = {f"net@{k}_{B}": boot([ny(r, B, k) for r in rs]) for B in BS for k in KS}
            d.update({"recovered@10": boot([float(r.get("recovered@10", False)) for r in rs]),
                      "recall@10": boot([r.get("constituent_recall@10") for r in rs]),
                      "mrr": boot([r.get("reciprocal_rank") for r in rs]),
                      "top_ho": boot([r.get("top_higher_order_gain") for r in rs]),
                      "sets": float(np.median([r.get("sets_registered", 0) for r in rs]))})
            summary.setdefault(part, {})[m] = d
            nuis = " / ".join("-" if d[f"net@{k}_nuis"][0] is None else f"{d[f'net@{k}_nuis'][0]:.2f}" for k in KS)
            lines.append(f"| {LABELS.get(m, m)} | {len(rs)} | {fmt(d['recovered@10'])} | {fmt(d['recall@10'])} | "
                         f"{fmt(d['mrr'])} | {fmt(d['top_ho'], 4)} | {fmt(d['net@10_known'])} | {fmt(d['net@10_decomp'])} | "
                         f"{nuis} | {d['sets']:.0f} |")
        lines.append("")
    lines += ["## Higher-order gain of the top-ranked set vs the planted set; calibration", "",
              "| method | top-set higher-order gain | planted set's own | calibration Spearman | ECE |", "|---|---|---|---|---|"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        c = calib(rs)
        rho = "-" if c["spearman"] is None else f"{c['spearman']:.3f}"
        ece = "-" if c["ece"] is None else f"{c['ece']:.4f}"
        lines.append(f"| {m} | {fmt(boot([r.get('top_higher_order_gain') for r in rs]), 4)} | "
                     f"{fmt(boot([r.get('truth_higher_order_gain') for r in rs]), 4)} | {rho} | {ece} |")
    text = "\n".join(lines) + "\n"
    (out / "Q1.md").write_text(text)
    (out / "Q1.json").write_text(json.dumps(summary, indent=1))
    figures(out, rows, methods)
    return text


def figures(out: Path, rows: list, methods: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = sorted({r["fraction"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for j, (held, title) in enumerate(((True, "held-out families"), (False, "training families"))):
        for m in methods:
            for i, (key, lab) in enumerate(((lambda r: ny(r, "nuis", 10), "net yield @10 (B_nuis)"),
                                            (lambda r: float(r.get("recovered@10", False)), "complete-object recovery @10"))):
                pts = [boot([key(r) for r in rows if r["method"] == m and r["heldout_family"] == held and r["fraction"] == f])
                       for f in fracs]
                ok = [(100 * f, t) for f, t in zip(fracs, pts) if t[0] is not None]
                if ok:
                    axes[i, j].errorbar([x for x, _ in ok], [t[0] for _, t in ok],
                                        yerr=[[t[0] - t[1] for _, t in ok], [t[2] - t[0] for _, t in ok]],
                                        lw=1.4, ms=4, marker="o", capsize=2, label=m)
                axes[i, j].set_ylabel(lab)
        axes[0, j].set_title(title)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("planted signal (% of residual variance)")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"q1_net_yield_recovery_vs_fraction.{ext}", dpi=160)
    plt.close(fig)


def paired(rows: list, a: str, b: str, key, n_boot: int = 2000, seed: int = 0) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["world"], {})[r["method"]] = r
    d = [key(w[a]) - key(w[b]) for w in by.values() if a in w and b in w
         and key(w[a]) is not None and key(w[b]) is not None]
    t = boot(d, n_boot, seed)
    return {"n_worlds": len(d), "diff": t[0], "ci": [t[1], t[2]]}


def write_q2(out: Path, full: str, others: list) -> str:
    out = Path(out)
    rows = load(out)
    lines = [f"# Q2: paired per-world differences, {full} minus each comparison", "",
             "95 % intervals from 2,000 world-bootstrap draws.", "",
             "| comparison | worlds | net@10 B_nuis | net@10 B_known | net@10 B_decomp | recovered@10 |", "|---|---|---|---|---|---|"]
    res = {}
    for o in others:
        d = {B: paired(rows, full, o, lambda r, B=B: ny(r, B, 10)) for B in BS}
        d["recovered@10"] = paired(rows, full, o, lambda r: float(r.get("recovered@10", False)))
        res[o] = d
        f = lambda x: "-" if x["diff"] is None else f"{x['diff']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"  # noqa: E731
        lines.append(f"| {o} | {d['nuis']['n_worlds']} | {f(d['nuis'])} | {f(d['known'])} | {f(d['decomp'])} | "
                     f"{f(d['recovered@10'])} |")
    text = "\n".join(lines) + "\n"
    (out / "Q2.md").write_text(text)
    (out / "Q2.json").write_text(json.dumps(res, indent=1))
    return text
