"""B3 few-shot: retrieval AP against the number of support patients n, on the test studies.

    python -m dbench.benchmark_v05.fewshot_v05 STUDY [--n-targets 8] [--draws 5] [--ns 2,3,5,8]

For each of the first n-targets targets of the study (seeded list of the B1 task) and each draw d, three query patients are drawn at random and
the support is the first n patients of a seeded random order of the remaining patients (nested in n).  The pool (singles intersection over all
built tasks of the target; the same 2,000 seeded pairs) is shared by every n and draw.  Labels: label engine v05 on that (support, query);
scores: ours (support-fit ordering, joint), the linear screen, NID, MISTy-style forest, random.  Every (n) is stored as its own study
``STUDY__fsN`` (draws are the splits), so metrics_v05 tables apply unchanged.  Capped at the number of patients the study allows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from dbench.benchmark_v04 import labels as L   # sets sys.path
import numpy as np

from dbench.benchmark_v05 import labels_fast as LF
from dbench.benchmark_v05 import run_scores as RS

BENCH = Path(L.BENCH)
V5 = BENCH / "runs/benchmark_v05"
METHODS = ("sel_screen", "sel_screen_joint", "linear_screen", "nid", "misty", "random")


def draws_for(st, target, ns, D):
    pats = sorted(map(str, st.patients))
    out = []
    for d in range(D):
        rng = np.random.default_rng(L.stable_seed("fewshot", st.name, target, d))
        q = sorted(rng.choice(pats, 3, replace=False).tolist())
        rest = [p for p in pats if p not in q]
        order = list(rng.permutation(rest))
        out.append((d, q, [str(p) for p in order]))
    return out


def run(study, n_targets, D, ns):
    from dbench.benchmark_v04.score_methods import build_pool_and_tasks   # noqa: F401  (pool convention reference)
    from dbench.harness import core
    from dbench.benchmark_v04 import score_methods as SM
    st = LF.load_study_v05(study)
    base_targets = json.loads((RS.label_dir(study) / "targets.json").read_text())["targets"][:n_targets]
    max_n = len(st.patients) - 3
    ns = [n for n in ns if n <= max_n]
    for t in base_targets:
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        if all((V5 / "labels" / f"{study}__fs{n}" / "cache" / f"{h}.npz").exists() for n in ns):
            continue
        t0 = time.time()
        seed = L.stable_seed("bench-task", st.name, t)
        built = {}
        for d, q, order in draws_for(st, t, ns, D):
            for n in ns:
                try:
                    task, names, ep = core.build_task(st, t, order[:n], q, seed=seed)
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    continue
                built[(n, d)] = (task, list(names), ep, order[:n], q)
        if not built:
            continue
        singles = sorted(set.intersection(*[set(b[1]) for b in built.values()]))
        rng = np.random.default_rng(L.stable_seed("bench-pairs", st.name, t))
        npairs = min(L.N_PAIRS, len(singles) * (len(singles) - 1) // 2)
        pr = set()
        while len(pr) < npairs:
            a, b = rng.choice(len(singles), 2, replace=False)
            pr.add((int(min(a, b)), int(max(a, b))))
        pairs = sorted(pr)
        for n in ns:
            lab_recs, scores = [], {m: {} for m in METHODS}
            for d in range(D):
                if (n, d) not in built:
                    continue
                task, names, ep, sup, q = built[(n, d)]
                idx = {x: i for i, x in enumerate(names)}
                s_sets = [(idx[x],) for x in singles]
                p_sets = [(idx[singles[a]], idx[singles[b]]) for a, b in pairs]
                rec = LF.score_split(task, ep, s_sets, p_sets, names, f"{study}__fs{n}", t, d, "cuda", 32)
                rec.update(support=sup, query=q)
                lab_recs.append(rec)
                tk = L.known_variant(core.support_only(task), ())
                for m in ("sel_screen", "sel_screen_joint"):
                    s1, s2, _ = SM.score_ours(m, tk, names, singles, pairs, "cuda")
                    scores[m][f"s{d}|single"], scores[m][f"s{d}|pair"] = np.asarray(s1, np.float32), np.asarray(s2, np.float32)
                ctx = RS.make_ctx(task, names, singles, pairs, f"{study}__fs{n}", t, d, 4)
                for m in ("linear_screen", "nid", "misty"):
                    s1, s2, _ = RS.M5.run_method(m, ctx)
                    scores[m][f"s{d}|single"], scores[m][f"s{d}|pair"] = np.asarray(s1, np.float32), np.asarray(s2, np.float32)
                S1, S2, _ = RS.M5.M4.random_scores(ctx)
                for k in range(len(S1)):
                    scores["random"][f"s{d}|single|{k}"], scores["random"][f"s{d}|pair|{k}"] = S1[k].astype(np.float32), S2[k].astype(np.float32)
            name = f"{study}__fs{n}"
            ld = V5 / "labels" / name
            ld.mkdir(parents=True, exist_ok=True)
            tp = ld / "targets.json"
            if not tp.exists():
                tp.write_text(json.dumps({"study": name, "targets": base_targets, "rule": f"few-shot n={n}, {D} draws"}))
            L.save_target(ld / "cache", t, {"pool_singles": singles, "pool_pairs": [[singles[a], singles[b]] for a, b in pairs], "splits": lab_recs})
            for m in METHODS:
                RS.write_receipt(V5 / "scores" / name / m, h, t, m, scores[m], {d: 0.0 for d in range(D)}, {d: "done" for d in range(D)}, {})
        print(f"{study} {t}: {len(singles)} singles, ns {ns}, {time.time() - t0:.0f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study")
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--ns", default="2,3,5,8")
    a = ap.parse_args(argv)
    L.diag()
    run(a.study, a.n_targets, a.draws, [int(x) for x in a.ns.split(",")])


if __name__ == "__main__":
    main()
