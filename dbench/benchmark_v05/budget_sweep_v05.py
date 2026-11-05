"""E3 / budget ablation: confirmed relations found by the beam against the fitting budget K, for three selectors at equal budget:
learned beam (release checkpoint 13c36faa), marginal-screen beam (support-fit screen: singles by exact support gain, pairs by the mean of
their singles), and a random beam (random shrinkage, random pair scores).  "Confirmed" = a fitted set that is in the labelled pool (every
single; the 2,000 sampled pairs) and passes the v05 label (B_known, 95%).  Also reports the labelled share of the K sets (triples, quadruples and
pairs outside the sampled pool carry no label and are not counted).

    python -m dbench.benchmark_v05.budget_sweep_v05 run STUDY [--n-targets 8] [--splits 0,3,6]
    python -m dbench.benchmark_v05.budget_sweep_v05 table STUDY[,STUDY]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
import numpy as np

BENCH = Path(L.BENCH)
V5 = BENCH / "runs/benchmark_v05"
KS = (16, 32, 64, 128, 256)
SELECTORS = ("learned", "marginal_screen", "random")


def run(study, n_targets, splits):
    from dbench.benchmark_v04.score_methods import build_pool_and_tasks
    from dbench.benchmark_v04 import tables as T
    from dbench.benchmark_v05.labels_fast import load_study_v05
    from dbench.benchmark_v05.run_scores import label_dir
    from dbench.harness import ablation_selection as AS
    from dbench.harness import core
    from morpheus_encoder.learned_search.search_v03 import SupportScreen, search
    AS.FROZEN = Path("/home/ubuntu/morpheus_work_sw/runs/development/learned_search_v03/release/frozen.pt")
    assert hashlib.sha256(AS.FROZEN.read_bytes()).hexdigest().startswith("13c36faa")
    model = AS._load_model("cuda")

    class RandomScreen(SupportScreen):
        """Random selector at the same budget: shrinkage and pair scores are random draws, the beam and refit are the same."""
        label = "random_search"

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

        def singles(self, task, B, L_, diag, base):
            p = task.X.shape[1]
            return -1e3 * self.rng.random(p), np.zeros(p)

        def pairs(self, task, B, L_, diag, base, gs):
            p = len(gs)
            M = self.rng.random((p, p))
            return 0.5 * (M + M.T), np.zeros((p, p)), np.zeros((p, p))

    st = load_study_v05(study)
    ld = label_dir(study)
    targets = json.loads((ld / "targets.json").read_text())["targets"][:n_targets]
    od = V5 / "budget" / study
    od.mkdir(parents=True, exist_ok=True)
    for t in targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if (od / f"{h}.json").exists():
            continue
        z = np.load(ld / "cache" / f"{h}.npz")
        built, singles, pairs = build_pool_and_tasks(st, t)
        pair_pos = {(a, b): i for i, (a, b) in enumerate(pairs)}
        sidx = {n: i for i, n in enumerate(singles)}
        rec = {}
        for r, support, query, task, names, ep in built:
            if r not in splits:
                continue
            tk = L.known_variant(core.support_only(task), ())
            ys = T.label(z, r, "single", "known", "95")
            yp = T.label(z, r, "pair", "known", "95")
            for sel in SELECTORS:
                for K in KS:
                    if sel == "learned":
                        out = search(tk, "known", K, model=model, device="cuda")
                    elif sel == "marginal_screen":
                        out = search(tk, "known", K, scorer=SupportScreen(), device="cuda")
                    else:
                        out = search(tk, "known", K, scorer=RandomScreen(L.stable_seed("budget-random", study, t, r, K)), device="cuda")
                    n_lab = n_pass = 0
                    for S, _, _ in out:
                        if len(S) == 1 and names[S[0]] in sidx:
                            n_lab += 1
                            n_pass += int(ys[sidx[names[S[0]]]])
                        elif len(S) == 2:
                            a, b = sorted((sidx.get(names[S[0]], -1), sidx.get(names[S[1]], -1)))
                            if (a, b) in pair_pos:
                                n_lab += 1
                                n_pass += int(yp[pair_pos[(a, b)]])
                    rec[f"{r}|{sel}|{K}"] = [len(out), n_lab, n_pass]
        (od / f"{h}.json").write_text(json.dumps({"target": t, "rec": rec}))
        print(study, t, flush=True)


def table(studies):
    lines = ["### E3 / budget: confirmed relations among the K fitted sets (labelled pool members passing the v05 label), mean over targets and splits [95% target bootstrap]", "",
             "| study | selector | K=16 | K=32 | K=64 | K=128 | K=256 | labelled share of K at K=64 |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for study in studies:
        from dbench.benchmark_v05.run_scores import label_dir
        ts = json.loads((label_dir(study) / "targets.json").read_text())["targets"]
        data = {}
        for t in ts:
            h = hashlib.sha256(t.encode()).hexdigest()[:10]
            p = V5 / "budget" / study / f"{h}.json"
            if p.exists():
                data[t] = json.loads(p.read_text())["rec"]
        if not data:
            continue
        rng = np.random.default_rng(0)
        for sel in SELECTORS:
            cells = []
            for K in KS:
                v = np.array([np.mean([x[2] for k, x in rec.items() if k.endswith(f"|{sel}|{K}")]) for rec in data.values()])
                b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                cells.append(f"{v.mean():.2f} [{np.quantile(b, .025):.2f}, {np.quantile(b, .975):.2f}]")
                res[f"{study}|{sel}|{K}"] = float(v.mean())
            share = np.mean([np.mean([x[1] / x[0] for k, x in rec.items() if k.endswith(f"|{sel}|64")]) for rec in data.values()])
            lines.append(f"| {study} ({len(data)} targets) | {sel} | " + " | ".join(cells) + f" | {share:.2f} |")
    text = "\n".join(lines) + "\n"
    (V5 / "budget_table.md").write_text(text)
    (V5 / "budget_table.json").write_text(json.dumps(res, indent=1))
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["run", "table"])
    ap.add_argument("studies")
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--splits", default="0,3,6")
    a = ap.parse_args(argv)
    if a.mode == "run":
        L.diag()
        run(a.studies, a.n_targets, {int(x) for x in a.splits.split(",")})
    else:
        print(table(a.studies.split(",")))


if __name__ == "__main__":
    main()
