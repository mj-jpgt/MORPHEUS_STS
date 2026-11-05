"""Full-pool retrieval evaluation (owner directive 2026-10-21): every method scores every pool member.

Pool of a target: every candidate present in all folds' candidate pools (residual programmes, programme
scores and morphology similarities in their four spatial versions, plus the pool genes the folds share)
and a fixed random sample of 256 pairs of those candidates (seeded by the target).  Ground truth,
method-independent, from realised held-out gains pooled over the folds' query patients (95 % patient
bootstrap, lower bound > 0):
  primary   B_known for singles and pairs (beyond the known programme scores and composition), and
            B_decomp for pairs (beyond the best additive decomposition);
  lenient   B_nuis (supplementary).
Each method emits a score for every pool member on each fold's support rows; AUROC and average precision
are computed over the pool per (target, fold) and averaged; intervals bootstrap over targets.

    python -m dbench.harness.pool labels --workers 12
    python -m dbench.harness.pool scores --methods ebm,gbm_h,rulefit,linear_screen,pysr,random --workers 6
    python -m dbench.harness.pool table
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import core

OUT = Path("runs/development/benchmark_v03/pool")
N_PAIRS = 256


def dev_folds():
    from .dev_metric import TARGETS, load_dev_study, load_targets
    spec = load_targets(TARGETS)
    study = load_dev_study(spec["study"], spec["partitions"])
    folds = core.patient_folds(study.patients, spec["n_folds"], spec["seed"])
    built = core._tasks(study, spec["targets"], folds, (), 0, spec["seed"])
    return spec, built


def target_pools(built) -> dict:
    """target -> (singles: sorted candidate names, pairs: [(a, b)] names), shared by all folds and methods."""
    names_by = {}
    for t, f, conds, names, st, q in built:
        names_by.setdefault(t, []).append(set(names))
    out = {}
    for t, sets in names_by.items():
        singles = sorted(set.intersection(*sets))
        rng = np.random.default_rng(core.stable_seed("pool-pairs", t))
        pairs = set()
        while len(pairs) < min(N_PAIRS, len(singles) * (len(singles) - 1) // 2):
            a, b = rng.choice(len(singles), 2, replace=False)
            pairs.add((singles[min(a, b)], singles[max(a, b)]))
        out[t] = (singles, sorted(pairs))
    return out


# ------------------------------------------------------------------ labels
def _label_job(i):
    import torch
    torch.set_num_threads(1)
    tk, sets, B = core._G["label"][i]
    sc = core.score_sets(tk, sets, Bs=(B,), device=core._G.get("device"))[B]
    return sc[0], sc[1]


def study_folds(study, targets, n_folds: int, seed: int):
    """(spec-like dict, built tasks) for any study: real targets only, patient-grouped folds."""
    n_folds = max(2, min(n_folds, len(study.patients)))
    folds = core.patient_folds(study.patients, n_folds, seed)
    built = core._tasks(study, list(targets), folds, (), 0, seed)
    sha = core.registry_hash({"study": study.name, "targets": list(targets), "folds": folds, "seed": seed})
    return {"sha256": sha, "targets": list(targets), "study": study.name}, built


def compute_labels(workers: int = 12, device=None, out: Path = OUT, ctx=None) -> dict:
    from .public_metrics import labels_pooled
    spec, built = ctx or dev_folds()
    pools = target_pools(built)
    jobs, keys = [], []
    for t, f, conds, names, st, q in built:
        idx = {n: i for i, n in enumerate(names)}
        tk = dict(conds)["real"]
        singles, pairs = pools[t]
        s_idx = [(idx[n],) for n in singles]
        p_idx = [(idx[a], idx[b]) for a, b in pairs]
        for kind, sets, Bs in (("single", s_idx, ("nuis", "known")), ("pair", p_idx, ("nuis", "known", "decomp"))):
            for B in Bs:
                jobs.append((tk, sets, B))
                keys.append((t, f, kind, B))
    core._G.update(label=jobs, device=device)
    res = core._pool_map(_label_job, len(jobs), workers) if device is None else [_label_job(i) for i in range(len(jobs))]
    core._G.clear()
    labels = {}
    for t in pools:
        for kind in ("single", "pair"):
            for B in (("nuis", "known") if kind == "single" else ("nuis", "known", "decomp")):
                per_fold = [r for k, r in zip(keys, res) if k[0] == t and k[2] == kind and k[3] == B]
                labels[f"{t}|{kind}|{B}"] = [bool(x) for x in labels_pooled(per_fold)]
    shares = {f"{kind}|{B}": float(np.mean([np.mean(v) for k, v in labels.items() if k.endswith(f"|{kind}|{B}")]))
              for kind, B in (("single", "nuis"), ("single", "known"), ("pair", "nuis"), ("pair", "known"),
                              ("pair", "decomp"))}
    rec = {"targets_sha256": spec["sha256"], "pools": {t: {"singles": s, "pairs": p} for t, (s, p) in pools.items()},
           "labels": labels, "positive_share": shares}
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(rec))
    return rec


# ------------------------------------------------------------------ method scores
def _score_job(i):
    import torch
    torch.set_num_threads(1)
    fn, tk, s_idx, p_idx, B, names = core._G["pool"][i]
    st = core.support_only(tk)
    st.cand_names = names        # candidate names (version-aware scorers, e.g. MISTy)
    try:
        return fn(st, B, s_idx, p_idx)
    except Exception as exc:     # e.g. a held-out-patient step with one support patient: recorded, scored as ties
        print(f"pool job {i} failed: {type(exc).__name__}: {exc}"[:300], flush=True)
        return np.full(len(s_idx), -np.inf), np.full(len(p_idx), -np.inf), []


def compute_scores(methods: list, workers: int = 6, device=None, out: Path = OUT, B: str = "known", ctx=None) -> None:
    from .fewshot import scorers
    spec, built = ctx or dev_folds()
    pools = target_pools(built)
    table = scorers(device=device)
    out.mkdir(parents=True, exist_ok=True)
    for m in methods:
        jobs, keys = [], []
        for t, f, conds, names, st, q in built:
            idx = {n: i for i, n in enumerate(names)}
            singles, pairs = pools[t]
            jobs.append((table[m], dict(conds)["real"], np.array([idx[n] for n in singles]),
                         np.array([(idx[a], idx[b]) for a, b in pairs]), B, list(names)))
            keys.append((t, f))
        core._G["pool"] = jobs
        if str(device).startswith("cuda"):   # one GPU process (no fork after CUDA is initialised)
            res = []
            for i in range(len(jobs)):
                t0 = time.time()
                res.append(_score_job(i))
                print(f"{m} {keys[i][0]}|{keys[i][1]} done {i + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
        else:
            res = core._pool_map(_score_job, len(jobs), workers)
        core._G.clear()
        rec = {f"{t}|{f}": {"single": np.asarray(r[0], float).tolist(), "pair": np.asarray(r[1], float).tolist()}
               for (t, f), r in zip(keys, res)}
        (out / f"scores_{m}_{B}.json").write_text(json.dumps({"method": m, "B": B, "scores": rec}))
        names_of = {(t, f): names for t, f, conds, names, st, q in built}
        entries = [{"target": t, "fold": f, "condition": "real", "B": B,
                    "sets": [[names_of[(t, f)][i] for i in st] for st in r[2]], "pred": [None] * len(r[2]),
                    "pred_sd": [None] * len(r[2])} for (t, f), r in zip(keys, res) if len(r) > 2 and r[2]]
        if entries:   # the method's own top-64 lists, in the dev registry format (precision@k columns)
            core.freeze_registry(entries, out / "registry" / f"{m}_{B}" / "registry.json")
        print(m, "scored", len(rec), flush=True)


# ------------------------------------------------------------------ table
def _ap_auc(y, s, seeds: int = 20):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return None, None
    fin = s[np.isfinite(s)]
    lo, span = (fin.min() - 1.0, fin.max() - fin.min() + 1.0) if len(fin) else (0.0, 1.0)
    s = np.where(np.isfinite(s), s, lo)
    au, ap = [], []
    for sd in range(seeds):   # ties (members a method did not score) broken at random
        jit = np.random.default_rng(sd).random(len(s)) * 1e-9 * span
        au.append(roc_auc_score(y, s + jit))
        ap.append(average_precision_score(y, s + jit))
    return float(np.mean(au)), float(np.mean(ap))


def table(out: Path = OUT, B: str = "known", title: str = "development targets") -> str:
    from .public_metrics import boot_mean, display
    lab = json.loads((out / "labels.json").read_text())
    rows = {}
    for p in sorted(out.glob(f"scores_*_{B}.json")):
        sc = json.loads(p.read_text())
        m = sc["method"]
        dec_path = out / f"scores_{m}_decomp.json"   # a method conditioned on the baseline scores B_decomp itself
        dec = json.loads(dec_path.read_text())["scores"] if dec_path.exists() else sc["scores"]
        per = {}
        for key, d in sc["scores"].items():
            t, f = key.rsplit("|", 1)
            for kind, lb in (("single", B), ("pair", B), ("pair", "decomp")):
                dd = dec.get(key, d) if lb == "decomp" else d
                au, ap = _ap_auc(lab["labels"][f"{t}|{kind}|{lb}"], dd[kind])
                per.setdefault((kind, lb), {}).setdefault(t, []).append((au, ap))
        rows[m] = {f"{kind}@B_{lb}": {"auroc": boot_mean([np.mean([a for a, _ in v if a is not None]) for v in d.values()
                                                          if any(a is not None for a, _ in v)]),
                                      "ap": boot_mean([np.mean([b for _, b in v if b is not None]) for v in d.values()
                                                       if any(b is not None for _, b in v)])}
                   for (kind, lb), d in per.items()}
    f = lambda t: "-" if t is None or t[0] is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
    cols = [f"single@B_{B}", f"pair@B_{B}", "pair@B_decomp"]
    sh = lab["positive_share"]
    lines = [f"## Full-pool retrieval, {title} (labels: singles and pairs at B_{B}, pairs at B_decomp)", "",
             f"Positive share: singles {sh.get(f'single|{B}', 0):.3f}, pairs {sh.get(f'pair|{B}', 0):.3f} at B_{B}, "
             f"pairs {sh.get('pair|decomp', 0):.3f} at B_decomp (lenient B_nuis: singles {sh['single|nuis']:.3f}, pairs "
             f"{sh['pair|nuis']:.3f}). Every method scores every pool member; AUROC / AP per (target, fold), mean over "
             "folds, 95 % bootstrap over targets.", "",
             "| method | " + " | ".join(f"AUROC {c} | AP {c}" for c in cols) + " |", "|---" * (2 * len(cols) + 1) + "|"]
    for m, d in rows.items():
        lines.append(f"| {display(m)} | " + " | ".join(f"{f((d.get(c) or {}).get('auroc'))} | {f((d.get(c) or {}).get('ap'))}"
                                                     for c in cols) + " |")
    text = "\n".join(lines) + "\n"
    (out / f"pool_table_{B}.md").write_text(text)
    (out / f"pool_table_{B}.json").write_text(json.dumps({"rows": rows, "positive_share": sh}, indent=1))
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("labels", "scores", "table"))
    ap.add_argument("--methods", default="ebm,gbm_h,rulefit,linear_screen,pysr,random")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--B", default="known")
    args = ap.parse_args(argv)
    if args.mode == "labels":
        rec = compute_labels(args.workers, args.device)
        print(json.dumps(rec["positive_share"]))
    elif args.mode == "scores":
        compute_scores(args.methods.split(","), args.workers, args.device, B=args.B)
    else:
        print(table(B=args.B))


if __name__ == "__main__":
    main()
