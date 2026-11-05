"""Table 1 / Table 2 builder for benchmark v05 (spec 4, 5.2) from cached labels and score receipts.

    python -m dbench.benchmark_v05.metrics_v05 table --studies a,b,c --methods m1,m2 [--level 95] [--baseline known] [--out NAME]
    python -m dbench.benchmark_v05.metrics_v05 overlap --studies a,b,c --ref sel_screen --methods nid,linear_screen

Labels: runs/benchmark_v05/labels/<study>/cache when present (label engine v05), else the v04 cache.  Scores:
runs/benchmark_v05/scores/<study>/<method>.  Per (study, method, target): every metric per split (random tie-break, 5 jitter
seeds), mean over the splits the method was run on.  Hierarchical bootstrap (2,000 replicates): resample studies, then
targets within each sampled study.  Paired difference "ours minus method" uses, per target, the splits both were run on and
only targets both have.  Holm across methods within each primary endpoint (AP singles, AP pairs).  Random = chance
reference (AP = prevalence).  Threshold metrics (balanced accuracy, F1) use top-m with m = round(prevalence x pool) per
(target, split), so chance equals prevalence-level performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from dbench.benchmark_v04 import tables as T

BENCH = Path.home() / "morpheus_work_bench"
V5 = BENCH / "runs/benchmark_v05"
N_BOOT = 2000
import os
OURS = os.environ.get("W3_OURS", "sel_screen")
KEYS = ["ap", "auroc", "prev", "p5", "p10", "p20", "p50", "r10", "r20", "hit10", "mrr", "bacc", "f1", "rho"]
DISPLAY = {
    "sel_screen": "ours: support-fit ordering (pairs = mean of singles)",
    "sel_screen_joint": "ours: support-fit ordering, joint pair gain (suppl.)",
    "beam64": "ours: learned beam K=64 (suppl.; pool AP, 64 scored sets, rest tied)",
    "hybrid_lin_sel": "hybrid: linear-screen singles + support-fit pairs (suppl.)",
    "hybrid_lin_joint": "hybrid: linear-screen singles + joint support-fit pairs (suppl.)",
    "random": "random (chance reference, 20 seeds)",
    "linear_screen": "linear interaction screen", "ebm": "EBM (native)", "fair_ebm": "EBM (fair)",
    "gbm_shap": "GBM + SHAP interactions (native)", "gbm_h": "GBM + Friedman H", "fair_gbm": "GBM (fair)",
    "rulefit": "RuleFit (native)", "fair_rulefit": "RuleFit (fair)", "irf": "iRF (native)", "fair_irf": "iRF (fair)",
    "misty": "MISTy-style forest (pairs derived)", "fair_misty": "MISTy-style forest (fair)",
    "nid": "NID", "pysr": "PySR (shortlist)", "tabdistill": "TabPFN + SPEX", "stability_selection": "stability selection",
    "knockoffs": "Model-X knockoffs"}


def label_dir(study):
    for r in (V5 / "labels", T.ROOT / "labels"):
        if (r / study / "cache").exists() and any((r / study / "cache").glob("*.npz")):
            return r / study
    raise FileNotFoundError(study)


def label_source(study):
    return "label engine v05" if str(label_dir(study)).startswith(str(V5)) else "label engine v04 (cache; verified equal to v05)"


def targets_of(study):
    return json.loads((label_dir(study) / "targets.json").read_text())["targets"]


def _clean(s):
    s = np.asarray(s, float)
    fin = np.isfinite(s)
    return np.where(fin, s, (s[fin].min() - 1.0) if fin.any() else 0.0)


def split_stats(y, s, g, rng_seed=0, jitter=5):
    """All metrics for one (target, split, kind); s may be a list of score vectors (random: averaged)."""
    n, npos = len(y), int(y.sum())
    if npos == 0 or npos == n:
        return {k: np.nan for k in KEYS} | {"prev": npos / max(n, 1)}
    out = {k: [] for k in KEYS}
    for j in range(jitter):
        sv = _clean(s)
        span = max(float(np.ptp(sv)), 1e-12)
        sv = sv + np.random.default_rng(j + rng_seed).random(n) * 1e-9 * span
        order = np.argsort(-sv, kind="stable")
        ys = y[order]
        out["ap"].append(average_precision_score(y, sv))
        out["auroc"].append(roc_auc_score(y, sv))
        for k in (5, 10, 20, 50):
            out[f"p{k}"].append(ys[:k].mean())
        for k in (10, 20):
            out[f"r{k}"].append(ys[:k].sum() / npos)
        out["hit10"].append(float(ys[:10].any()))
        first = np.flatnonzero(ys)
        out["mrr"].append(1.0 / (first[0] + 1))
        m = max(1, int(round(npos / n * n)))
        pred = np.zeros(n, bool)
        pred[order[:m]] = True
        tp = float((pred & y).sum())
        tpr, tnr = tp / npos, float((~pred & ~y).sum()) / (n - npos)
        out["bacc"].append(0.5 * (tpr + tnr))
        prec = tp / max(pred.sum(), 1)
        out["f1"].append(2 * prec * tpr / (prec + tpr) if prec + tpr > 0 else 0.0)
        out["rho"].append(spearmanr(sv, g).correlation if np.ptp(sv) > 0 else np.nan)
    res = {k: float(np.nanmean(v)) for k, v in out.items() if v}
    res["prev"] = npos / n
    return res


_CACHE = {}


def per_split(study, method, target, level, B):
    key = (study, method, target, level, B)
    if key in _CACHE:
        return _CACHE[key]
    h = hashlib.sha256(target.encode()).hexdigest()[:10]
    sp = V5 / "scores" / study / method / f"{h}.npz"
    out = None
    jp = V5 / "scores" / study / method / f"{h}.json"
    st = {"done": 0, "timeout": 0, "error": 0, "skipped": 0}
    legacy = False
    if jp.exists():
        stt = json.loads(jp.read_text()).get("status", {})
        for v in stt.values():
            st["done" if v == "done" else "timeout" if v == "timeout" else "error" if str(v).startswith("error") else "skipped"] += 1
        legacy = not stt
        if not sp.exists():
            out = {"single": {}, "pair": {}}
    if sp.exists():
        sc = np.load(sp)
        if jp.exists() and legacy:   # receipts copied from v04 carry no status: every stored split is a completed split
            st["done"] = len({k.split("|")[0] for k in sc.files})
        ld = label_dir(study)
        meta, z = json.loads((ld / "cache" / f"{h}.json").read_text()), np.load(ld / "cache" / f"{h}.npz")
        out = {"single": {}, "pair": {}}
        for kind in ("single", "pair"):
            if B == "decomp" and kind == "single":   # B_decomp labels exist for pairs only
                continue
            for s in T.splits_of(meta):
                y = T.label(z, s, kind, B, level)
                g = np.asarray(z[f"s{s}|{kind}|{B}|G"], float)
                if method == "random":
                    ks = sorted({k.split("|")[2] for k in sc.files if k.startswith(f"s{s}|{kind}|")})
                    if not ks:
                        continue
                    rs = [split_stats(y, sc[f"s{s}|{kind}|{k}"], g, jitter=1) for k in ks]
                    out[kind][s] = {k: float(np.nanmean([r[k] for r in rs])) for k in KEYS}
                else:
                    kk = f"s{s}|{kind}"
                    if kk not in sc.files:
                        continue
                    out[kind][s] = split_stats(y, sc[kk], g)
    if out is not None:
        out["_status"] = st
    _CACHE[key] = out
    return out


def mean_over(d, splits, key):
    v = [x[key] for s, x in d.items() if (splits is None or s in splits)]
    return float(np.nanmean(v)) if v and np.isfinite(v).any() else np.nan


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def make_hybrids(studies):
    """Hybrid score files from existing receipts: linear-screen singles with support-fit pairs (sel_screen, joint)."""
    for st in studies:
        for name, pairsrc in (("hybrid_lin_sel", "sel_screen"), ("hybrid_lin_joint", "sel_screen_joint")):
            od = V5 / "scores" / st / name
            for f in (V5 / "scores" / st / "linear_screen").glob("*.npz"):
                out = od / f.name
                src = V5 / "scores" / st / pairsrc / f.name
                if out.exists() or not src.exists():
                    continue
                a, b = np.load(f), np.load(src)
                arr = {}
                for s in {k.split("|")[0] for k in a.files}:
                    if f"{s}|single" in a.files and f"{s}|pair" in b.files:
                        arr[f"{s}|single"], arr[f"{s}|pair"] = a[f"{s}|single"], b[f"{s}|pair"]
                od.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, **arr)
                (od / f.with_suffix(".json").name).write_text(json.dumps({"method": name, "derived_from": ["linear_screen singles", f"{pairsrc} pairs"]}))


def table(studies, methods, level="95", B="known", seed=0, name=None):
    rng = np.random.default_rng(seed)
    make_hybrids(studies)
    data = {s: {m: {t: per_split(s, m, t, level, B) for t in targets_of(s)} for m in methods} for s in studies}
    ts = {s: targets_of(s) for s in studies}
    res = {m: {} for m in methods}
    prev = {}
    reps = []
    for _ in range(N_BOOT):
        ss = rng.integers(0, len(studies), len(studies))
        reps.append([(studies[i], rng.integers(0, len(ts[studies[i]]), len(ts[studies[i]]))) for i in ss])

    def boot(vd):
        out = np.empty(N_BOOT)
        for r, rep in enumerate(reps):
            vals = [np.nanmean(vd[s][ix]) if np.isfinite(vd[s][ix]).any() else np.nan for s, ix in rep]
            out[r] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        return out

    for kind in ("single", "pair"):
        prev[kind] = {s: float(np.nanmean([mean_over(data[s][OURS][t][kind], None, "prev") if data[s].get(OURS, {}).get(t) else np.nan for t in ts[s]])) for s in studies}
        for key in KEYS:
            if key == "prev":
                continue
            vec = {m: {s: np.array([mean_over(data[s][m][t][kind], None, key) if data[s][m][t] else np.nan for t in ts[s]]) for s in studies} for m in methods}
            pair_d = {}
            if OURS in methods and key in ("ap", "auroc"):
                for m in methods:
                    if m == OURS:
                        continue
                    pair_d[m] = {}
                    for s in studies:
                        arr = []
                        for t in ts[s]:
                            a, b = data[s][OURS][t], data[s][m][t]
                            common = (set(a[kind]) & set(b[kind])) if a and b else set()
                            arr.append(mean_over(a[kind], common, key) - mean_over(b[kind], common, key) if common else np.nan)
                        pair_d[m][s] = np.array(arr)
            for m in methods:
                if not any(np.isfinite(v).any() for v in vec[m].values()):
                    continue
                bs = boot(vec[m])
                res[m][f"{key}_{kind}"] = {
                    "mean": float(np.nanmean([np.nanmean(vec[m][s]) for s in studies if np.isfinite(vec[m][s]).any()])),
                    "lo": float(np.nanquantile(bs, .025)), "hi": float(np.nanquantile(bs, .975)),
                    "n_targets": {s: int(np.isfinite(vec[m][s]).sum()) for s in studies},
                    "per_study": {s: (float(np.nanmean(vec[m][s])) if np.isfinite(vec[m][s]).any() else None) for s in studies}}
            ps, rows = [], []
            for m in pair_d:
                if not any(np.isfinite(v).any() for v in pair_d[m].values()):
                    continue
                bs = boot(pair_d[m])
                ps.append(float(np.nanmean(bs <= 0)))
                rows.append((m, float(np.nanmean(bs)), float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975)),
                             {s: int(np.isfinite(pair_d[m][s]).sum()) for s in studies}))
            adj = holm(ps) if (key == "ap" and ps) else np.full(len(ps), np.nan)
            for (m, dm, lo, hi, nt), p, pa in zip(rows, ps, adj):
                res[m][f"{key}_{kind}"].update({"diff_ours_minus": dm, "diff_lo": lo, "diff_hi": hi, "p_one_sided": p, "p_holm": float(pa), "n_paired_targets": nt})
    # variants: ours on the method's own targets and splits, lift on shared targets, timeouts scored at the chance reference
    extra = {m: {} for m in methods}
    for kind in ("single", "pair"):
        for m in methods:
            ours_same = {"ap": [], "auroc": []}
            meth_same = {"ap": [], "auroc": []}
            pv_sh, chance_var = [], []
            n_to = n_all = n_tgt = 0
            for s in studies:
                for t in ts[s]:
                    r_m, r_o, r_r = data[s][m][t], data[s].get(OURS, {}).get(t), per_split(s, "random", t, level, B)
                    if not r_m:
                        continue
                    sp = r_m["_status"]
                    ds = r_m[kind]
                    ch = mean_over(r_r[kind], None, "ap") if r_r else np.nan
                    if ds and r_o:
                        common = set(ds) & set(r_o[kind])
                        if common:
                            for key in ("ap", "auroc"):
                                ours_same[key].append(mean_over(r_o[kind], common, key))
                                meth_same[key].append(mean_over(ds, common, key))
                            pv_sh.append(mean_over(r_o[kind], common, "prev"))
                            n_tgt += 1
                    n_to += sp["timeout"] + sp["error"]
                    n_all += sp["done"] + sp["timeout"] + sp["error"]
                    aps = [x["ap"] for x in ds.values() if np.isfinite(x["ap"])]
                    k_to = sp["timeout"] + sp["error"]
                    if (aps or k_to) and np.isfinite(ch):
                        chance_var.append((sum(aps) + k_to * ch) / (len(aps) + k_to))
            nm = lambda v: float(np.nanmean(v)) if len(v) else np.nan  # noqa: E731
            extra[m][kind] = {"ours_same_targets": nm(ours_same["ap"]), "ours_auroc_same_targets": nm(ours_same["auroc"]),
                              "lift_shared": nm(meth_same["ap"]) / nm(pv_sh) if pv_sh and nm(pv_sh) > 0 else np.nan,
                              "ours_lift_shared": nm(ours_same["ap"]) / nm(pv_sh) if pv_sh and nm(pv_sh) > 0 else np.nan,
                              "ap_timeouts_at_chance": nm(chance_var),
                              "timeout_or_error_splits": int(n_to), "splits_attempted": int(n_all), "n_targets_shared": int(n_tgt)}
    mp = lambda kind: float(np.nanmean(list(prev[kind].values())))  # noqa: E731
    f = lambda d, k: "-" if k not in d else f"{d[k]['mean']:.3f} [{d[k]['lo']:.3f}, {d[k]['hi']:.3f}]"  # noqa: E731
    f1 = lambda d, k: "-" if k not in d else f"{d[k]['mean']:.3f}"  # noqa: E731
    g = lambda d, k: "-" if k not in d or "diff_ours_minus" not in d[k] else f"{d[k]['diff_ours_minus']:+.3f} [{d[k]['diff_lo']:+.3f}, {d[k]['diff_hi']:+.3f}], p_Holm {d[k]['p_holm']:.3f}"  # noqa: E731
    pv = {k: {s: round(v, 4) for s, v in d.items()} for k, d in prev.items()}
    head = f"studies {', '.join(studies)}; RFF labels at {level}%, baseline B_{B}; mean prevalence singles {mp('single'):.4f}, pairs {mp('pair'):.4f}"
    decl = ("Primary pair score: mean of the two single support gains (ours); joint pair support gain is supplementary. Declared before any label was computed: "
            "freeze addendum 3 (sha256 85dc7c747a8f32f7e431c54ce23488bf58154b6ce891b29f00e8557bd1437d86, 2026-10-27T16:27Z, key ordering_declaration), "
            "repeated in the v05 addendum chain 13-16.")
    L1 = [f"### Table 1: {head}", f"Per-study prevalence: {json.dumps(pv)}. Hierarchical bootstrap (studies, then targets within study; 2,000 replicates). {decl}",
          "Lift = mean AP / mean prevalence on the targets and splits the method shares with ours (chance = 1), ours beside it; raw AP cells of a method run on fewer targets than ours must be read against the 'ours AP, same targets' cell. Differences are ours minus method on shared targets and splits.", "",
          "| method | targets run per study | AP singles [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP singles | AP pairs [95% CI] | lift (shared) method / ours | ours AP, same targets | ours minus method, AP pairs | P@10 s / p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    L1b = ["", f"### Table 1b (AUROC, same score files): {head}", "",
           "| method | targets run per study | AUROC singles [95% CI] | ours AUROC, same targets | ours minus method | AUROC pairs [95% CI] | ours AUROC, same targets | ours minus method |", "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        d = res[m]
        if "ap_single" not in d and "ap_pair" not in d:
            L1.append(f"| {DISPLAY.get(m, m)} | no receipts (timeout / not run) | | | | | | | | | |")
            continue
        ex = extra[m]
        rate = ex["single"]["timeout_or_error_splits"] / max(ex["single"]["splits_attempted"], 1)
        flag = f" **(incomplete: {ex['single']['timeout_or_error_splits']} of {ex['single']['splits_attempted']} target x split fits timed out)**" if rate > 0.10 else ""
        k0 = "ap_single" if "ap_single" in d else "ap_pair"
        nt = "/".join(str(d[k0]["n_targets"][s]) for s in studies)
        gd = lambda k: "-" if k not in d or "diff_ours_minus" not in d[k] else f"{d[k]['diff_ours_minus']:+.3f} [{d[k]['diff_lo']:+.3f}, {d[k]['diff_hi']:+.3f}], p_Holm {d[k]['p_holm']:.3f}"  # noqa: E731
        gu = lambda k: "-" if k not in d or "diff_ours_minus" not in d[k] else f"{d[k]['diff_ours_minus']:+.3f} [{d[k]['diff_lo']:+.3f}, {d[k]['diff_hi']:+.3f}]"  # noqa: E731
        L1.append(f"| {DISPLAY.get(m, m)}{flag} | {nt} | {f(d, 'ap_single')} | {ex['single']['lift_shared']:.1f} / {ex['single']['ours_lift_shared']:.1f} | {ex['single']['ours_same_targets']:.3f} | {gd('ap_single')} | "
                  f"{f(d, 'ap_pair')} | {ex['pair']['lift_shared']:.1f} / {ex['pair']['ours_lift_shared']:.1f} | {ex['pair']['ours_same_targets']:.3f} | {gd('ap_pair')} | {f1(d, 'p10_single')} / {f1(d, 'p10_pair')} |")
        L1b.append(f"| {DISPLAY.get(m, m)}{flag} | {nt} | {f(d, 'auroc_single')} | {ex['single']['ours_auroc_same_targets']:.3f} | {gu('auroc_single')} | "
                   f"{f(d, 'auroc_pair')} | {ex['pair']['ours_auroc_same_targets']:.3f} | {gu('auroc_pair')} |")
    L2 = ["", f"### Table 2 (secondary, same score files): {head}", "",
          "| method | recall@10 s / p | recall@20 s / p | hit@10 s / p | MRR s / p | balanced acc. s / p | F1 s / p | Spearman(score, realised gain) s / p | P@20 s / p |", "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        d = res[m]
        if "ap_single" not in d and "ap_pair" not in d:
            continue
        pr = lambda k: f"{f1(d, k + '_single')} / {f1(d, k + '_pair')}"  # noqa: E731
        L2.append(f"| {DISPLAY.get(m, m)} | {pr('r10')} | {pr('r20')} | {pr('hit10')} | {pr('mrr')} | {pr('bacc')} | {pr('f1')} | {pr('rho')} | {pr('p20')} |")
    L3 = ["", f"### Table 1c (units: target x split fits for every method and study): timeout rule, AP over completed splits versus timed-out splits scored at the chance reference (random AP on the same target); {head}", "",
          "| method | splits attempted (s) | timed out / errored | AP singles, completed only | AP singles, timeouts at chance | AP pairs, completed only | AP pairs, timeouts at chance |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        if "ap_single" not in res[m] and "ap_pair" not in res[m]:
            continue
        a, b = extra[m]["single"], extra[m]["pair"]
        L3.append(f"| {DISPLAY.get(m, m)} | {a['splits_attempted']} | {a['timeout_or_error_splits']} | {res[m].get('ap_single', {}).get('mean', 0.0):.3f} | {a['ap_timeouts_at_chance']:.3f} | {res[m].get(chr(97)+chr(112)+chr(95)+chr(112)+chr(97)+chr(105)+chr(114), {}).get(chr(109)+chr(101)+chr(97)+chr(110), 0.0):.3f} | {b['ap_timeouts_at_chance']:.3f} |")
    L3 += ["", "Timeout rule: each (method, target, split) fit has a fixed wall-clock budget (seconds in seconds_per_method.md); a fit that exceeds it, or errors, is recorded and is never replaced by another method. Methods with more than 10% timed-out splits are flagged incomplete in Table 1. Paired differences use only targets and splits both methods completed."]
    L2 += ["", "Label set per study: " + "; ".join(f"{s}: {label_source(s)}" for s in studies) + ". Random is the chance reference (AP = prevalence; balanced accuracy 0.5).",
           "Threshold metrics: top-m with m = round(prevalence x pool) per (target, split). Pairs: where a method has no native pair score the pair score is derived as stated in its row. Methods run on the declared split subset are compared with ours on the same splits and the same targets for the paired difference."]
    text = "\n".join(L1 + L2) + "\n"
    nm = name or f"table1_{'_'.join(studies)}_L{level}_{B}"
    (V5 / f"{nm}.md").write_text(text)
    (V5 / f"{nm}.json").write_text(json.dumps({"prevalence": prev, "rows": res, "extra": extra, "label_sources": {s: label_source(s) for s in studies}}, indent=1, default=float))
    return text


def overlap(studies, ref, methods, k=10, level="95", B="known"):
    """overlap@k of each method's top-k singles with the reference's, mean over targets and splits, per study."""
    rows = ["| method | " + " | ".join(studies) + " |", "|---|" + "---|" * len(studies)]
    for m in methods:
        cells = []
        for s in studies:
            vals = []
            for t in targets_of(s):
                h = hashlib.sha256(t.encode()).hexdigest()[:10]
                pa, pb = V5 / "scores" / s / ref / f"{h}.npz", V5 / "scores" / s / m / f"{h}.npz"
                if not (pa.exists() and pb.exists()):
                    continue
                a, b = np.load(pa), np.load(pb)
                for key in a.files:
                    if key.endswith("|single") and key in b.files:
                        ta = set(np.argsort(-_clean(a[key]), kind="stable")[:k])
                        tb = set(np.argsort(-_clean(b[key]), kind="stable")[:k])
                        vals.append(len(ta & tb) / k)
            cells.append(f"{np.mean(vals):.3f} (n={len(vals)})" if vals else "-")
        rows.append(f"| {DISPLAY.get(m, m)} | " + " | ".join(cells) + " |")
    text = f"### overlap@{k} of top-{k} singles with {DISPLAY.get(ref, ref)}\n\n" + "\n".join(rows) + "\n"
    (V5 / f"overlap{k}_{'_'.join(studies)}.md").write_text(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("table", "overlap"))
    ap.add_argument("--studies", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--level", default="95")
    ap.add_argument("--baseline", default="known")
    ap.add_argument("--ref", default=OURS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.mode == "table":
        print(table(a.studies.split(","), a.methods.split(","), a.level, a.baseline, name=a.out))
    else:
        print(overlap(a.studies.split(","), a.ref, a.methods.split(",")))


if __name__ == "__main__":
    main()
