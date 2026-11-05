"""E1 and E2 component experiments (benchmark spec addendum 2026-10-27), stored-label based.

    python -m dbench.benchmark_v04.e12 score STUDY            # writes scores/STUDY/{beam64,ho_pair,ho_pair_guarded}/
    python -m dbench.benchmark_v04.e12 tables STUDY           # E1 and E2 tables from stored scores and labels

E1: the adopted method with the beam at budget K=64 (release checkpoint 13c36faa...) against exhaustive support-fit
    scoring of the same pool (sel_screen singles; sel_screen_joint pairs). Pool members the beam did not fit get the
    lowest score (ties broken at random by the metric); a listed member scores 1000 - rank in the beam's own order (tier 1); unlisted members are ordered below by the network pool score (tier 2), because the 2,000 pool pairs are a random sample the 64-set beam rarely fits.
E2: pair ranking by joint gain over B_known (sel_screen_joint, stored) against ranking by G_HO = joint support gain
    minus the support gain of the additive decomposition of the same two variables (relation_v03 fit_sets with
    blocks [[0],[1]]); guarded = pairs with non-positive joint support gain get the lowest score (the decomp-label
    guard of amendment 2). Evaluated against the known and the decomp (guarded) labels: pair AP and precision of the
    top 20 pairs (share passing the patient-null labels).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from dbench.benchmark_v04 import labels as L  # sets sys.path
import numpy as np

BENCH = Path(L.BENCH)
ROOT = BENCH / "runs/development/benchmark_v04"
CKPT_PREFIX = "13c36faa"


def score(study, device="cuda", out=ROOT / "scores", max_targets=40):
    import torch
    from dbench.benchmark_v04.score_methods import build_pool_and_tasks
    from dbench.harness import core
    from dbench.harness import ablation_selection as AS
    AS.FROZEN = Path("/home/ubuntu/morpheus_work_sw/runs/development/learned_search_v03/release/frozen.pt")   # release checkpoint 13c36faa (path was relative to a worktree that does not hold it)
    _load_model, FROZEN = AS._load_model, AS.FROZEN
    from morpheus_encoder.learned_search import relation_v03 as rv
    from morpheus_encoder.learned_search.search_v03 import search, _support_only, SupportScreen
    sha = hashlib.sha256(Path(FROZEN).read_bytes()).hexdigest()
    assert sha.startswith(CKPT_PREFIX), sha
    model = _load_model(device)
    st = L.load_test_study(study)
    targets = json.loads((ROOT / "labels" / study / "targets.json").read_text())["targets"][:max_targets]
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if all((out / study / m / f"{h}.npz").exists() for m in ("beam64", "ho_pair", "ho_pair_guarded")):
            continue
        t0 = time.time()
        built, singles, pairs = build_pool_and_tasks(st, t)
        arr = {"beam64": {}, "ho_pair": {}, "ho_pair_guarded": {}}
        secs = {"beam64": {}, "ho_pair": {}}
        for r, support, query, task, names, ep in built:
            tk = L.known_variant(core.support_only(task), ())
            idx = {n: i for i, n in enumerate(names)}
            s_idx = np.array([idx[n] for n in singles])
            p_idx = np.array([(idx[singles[a]], idx[singles[b]]) for a, b in pairs])
            t1 = time.time()
            ranked = search(tk, "known", 64, model=model, device=device)
            # tier 2: pool members the beam did not fit are ordered by the network's own pool score (singles G_sup - mu,
            # pairs pair-head support minus shrinkage); tier 1 (the beam's fitted list, in its own order) ranks above
            from morpheus_encoder.learned_search.search_v03 import pool_scores
            ps1, ps2, _, _ = pool_scores(_support_only(tk), "known", model, p_idx, device=device)
            s1 = np.asarray(ps1, float)[s_idx].copy()
            s2 = np.asarray(ps2, float).copy()
            pos1 = {int(S[0]): k for k, (S, _, _) in enumerate(ranked) if len(S) == 1}
            pos2 = {tuple(sorted(int(x) for x in S)): k for k, (S, _, _) in enumerate(ranked) if len(S) == 2}
            for j, c in enumerate(s_idx):
                if int(c) in pos1:
                    s1[j] = 1000 - pos1[int(c)]
            for j, (a, b) in enumerate(p_idx):
                k = pos2.get((min(int(a), int(b)), max(int(a), int(b))))
                if k is not None:
                    s2[j] = 1000 - k
            arr["beam64"][f"s{r}|single"], arr["beam64"][f"s{r}|pair"] = s1.astype(np.float32), s2.astype(np.float32)
            secs["beam64"][r] = round(time.time() - t1, 1)
            t1 = time.time()
            tt = _support_only(tk)
            base = rv.baseline_for(tt, "known")
            joint = np.asarray(rv.fit_sets(tt.X[:, p_idx], base, tt.y, tt.groups, device=device)["g_sup"], float)
            add = np.asarray(rv.fit_sets(tt.X[:, p_idx], base, tt.y, tt.groups, blocks=[[0], [1]], device=device)["g_sup"], float)
            gho = joint - add
            low = float(np.nanmin(gho)) - 1.0
            arr["ho_pair"][f"s{r}|pair"] = gho.astype(np.float32)
            arr["ho_pair_guarded"][f"s{r}|pair"] = np.where(joint > 0, gho, low).astype(np.float32)
            # singles are those of the support-fit ordering (identical in every pair-ranking row)
            from dbench.harness.ablation_selection import pool_scorer
            sc1, _, _ = pool_scorer("sel_screen", device)(tk, "known", s_idx, p_idx)
            arr["ho_pair"][f"s{r}|single"] = arr["ho_pair_guarded"][f"s{r}|single"] = np.asarray(sc1, np.float32)
            secs["ho_pair"][r] = round(time.time() - t1, 1)
        for m in arr:
            od = out / study / m
            od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f"{h}.npz", **arr[m])
            (od / f"{h}.json").write_text(json.dumps({"target": t, "method": m, "seconds": secs.get(m, secs["ho_pair"])}))
        print(f"{study} {t}: {time.time() - t0:.0f} s", flush=True)


def precision_at(study, method, B, level="95", k=20, guard=True):
    from dbench.benchmark_v04 import tables as T
    from dbench.benchmark_v04.metrics import load_scores
    targets = json.loads((ROOT / "labels" / study / "targets.json").read_text())["targets"]
    out = {}
    for t in targets:
        lab = T.load_target(study, t)
        sc = load_scores(study, method, t)
        if lab is None or sc is None:
            continue
        meta, z = lab
        vals = []
        for s in T.splits_of(meta):
            if method == "random":
                ks = sorted({f.split("|")[2] for f in sc.files if f.startswith(f"s{s}|pair|")})
                arrs = [sc[f"s{s}|pair|{q}"] for q in ks]
            else:
                key = f"s{s}|pair"
                if key not in sc.files:
                    continue
                arrs = [sc[key]]
            y = T.label(z, s, "pair", B, level, guard)
            rng = np.random.default_rng(s)
            for a in arrs:
                a = np.asarray(a, float)
                a = np.where(np.isfinite(a), a, np.nanmin(a[np.isfinite(a)]) - 1) + rng.random(len(a)) * 1e-9 * max(np.ptp(a), 1e-12)
                top = np.argsort(-a)[:k]
                vals.append(float(y[top].mean()))
        if vals:
            out[t] = float(np.mean(vals))
    return out


def tables(study, n_boot=2000, seed=0):
    from dbench.benchmark_v04 import metrics as M
    md = [f"# E1 and E2 on {study}", ""]
    # E1
    md.append("## E1: beam (K=64, checkpoint 13c36faa) vs exhaustive support-fit scoring of the same pool (label baseline B_known, level 95)\n")
    md.append(M.table(study, ["beam64", "sel_screen", "sel_screen_joint"], "95", "known", ours="beam64"))
    # E2
    rows = {}
    rng = np.random.default_rng(seed)
    from dbench.benchmark_v04 import tables as _T
    _orig_label = _T.label

    def _label_pairs_only(z, split, kind, B="known", level="95", guard=True):   # B_decomp exists for pairs only; singles get an all-negative label (metric NaN, excluded)
        if B == "decomp" and kind == "single":
            return np.zeros(len(z[f"s{split}|single|known|G"]), bool)
        return _orig_label(z, split, kind, B, level, guard)
    _T.label = _label_pairs_only
    for B in ("known", "decomp"):
        md.append(f"\n## E2: pair rankings, labels B_{B} (decomp labels guarded by G_known > 0)\n")
        md.append(M.table(study, ["sel_screen_joint", "ho_pair", "ho_pair_guarded", "sel_screen", "random"], "95", B, ours="sel_screen_joint"))
        meths = ["sel_screen_joint", "ho_pair", "ho_pair_guarded", "sel_screen", "random"]
        P = {m: precision_at(study, m, B) for m in meths}
        common = sorted(set.intersection(*[set(v) for v in P.values()]))
        if not common:
            continue
        A = np.array([[P[m][t] for t in common] for m in meths])
        idx = rng.integers(0, len(common), size=(n_boot, len(common)))
        md.append(f"\nShare of the top-20 ranked pairs passing the patient-null label (B_{B}), mean over splits and targets (n targets {len(common)}), 95% target bootstrap:\n")
        md.append("| method | precision@20 | minus sel_screen_joint |\n|---|---|---|")
        for i, m in enumerate(meths):
            bs = A[i][idx].mean(1)
            d = (A[i][idx] - A[0][idx]).mean(1)
            md.append(f"| {M.DISPLAY.get(m, m)} | {A[i].mean():.3f} [{np.quantile(bs, .025):.3f}, {np.quantile(bs, .975):.3f}] | "
                      + ("-" if i == 0 else f"{d.mean():+.3f} [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]") + " |")
        rows[B] = {m: float(A[i].mean()) for i, m in enumerate(meths)}
    text = "\n".join(md) + "\n"
    (ROOT / f"E1E2_{study}.md").write_text(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("score", "tables"))
    ap.add_argument("study")
    ap.add_argument("--max-targets", type=int, default=40)
    a = ap.parse_args(argv)
    if a.mode == "score":
        L.diag()
        score(a.study, max_targets=a.max_targets)
    else:
        print(tables(a.study))


if __name__ == "__main__":
    main()
