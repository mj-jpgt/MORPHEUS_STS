"""E1 matched-budget comparison and B_decomp prevalence, from existing receipts and labels (no new fits).

    python -m dbench.benchmark_v05.e1_matched_v05 STUDY[,STUDY...]

For every (target, split): (a) the beam's proposals = the pool singles / pool pairs the beam listed (tier-1 score > 900 in the beam64
receipt; triples, quadruples and pairs outside the 2,000-pair pool carry no label and are not counted) and their positives under the
B_known 95% label; (b) the top-n of exhaustive support-fit ordering with n = the beam's count of listed members of that kind (matched
budget), and the top 64 (precision@64 and positives among the 64).  Per target mean over splits; hierarchical bootstrap (targets within
study, studies when several are given) for the paired difference (beam minus exhaustive, matched n).  Appended to E1E2_<studies>.md
(a block between the markers is replaced on re-run), together with the B_decomp label prevalence and positive pair count per study.
"""
import hashlib
import json
import sys

import numpy as np

from dbench.benchmark_v04 import tables as T
from dbench.benchmark_v05 import metrics_v05 as M

V5 = M.V5
MARK0, MARK1 = "<!-- E1-MATCHED-BEGIN -->", "<!-- E1-MATCHED-END -->"


def topk_pos(sc, y, k, seed=0):
    sv = np.asarray(sc, float)
    sv = np.where(np.isfinite(sv), sv, sv[np.isfinite(sv)].min() - 1 if np.isfinite(sv).any() else 0)
    sv = sv + np.random.default_rng(seed).random(len(sv)) * 1e-9 * max(float(np.ptp(sv)), 1e-12)
    return int(y[np.argsort(-sv, kind="stable")[:k]].sum())


def study_stats(study):
    out = {"single": [], "pair": []}
    decomp = []
    ld = M.label_dir(study)
    for t in M.targets_of(study):
        h = hashlib.sha256(t.encode()).hexdigest()[:10]
        pb, ps, pj = (V5 / "scores" / study / m / f"{h}.npz" for m in ("beam64", "sel_screen", "sel_screen_joint"))
        if not (pb.exists() and ps.exists()):
            continue
        b, s, j = np.load(pb), np.load(ps), np.load(pj)
        z = np.load(ld / "cache" / f"{h}.npz")
        meta = json.loads((ld / "cache" / f"{h}.json").read_text())
        rec = {k: [] for k in ("single", "pair")}
        for sp in T.splits_of(meta):
            for kind in ("single", "pair"):
                if f"s{sp}|{kind}" not in b.files:
                    continue
                y = T.label(z, sp, kind, "known", "95")
                bs = np.asarray(b[f"s{sp}|{kind}"], float)
                listed = np.flatnonzero(bs > 900)
                n = len(listed)
                src = j if kind == "pair" else s
                rec[kind].append((n, int(y[listed].sum()), topk_pos(src[f"s{sp}|{kind}"], y, 64), topk_pos(src[f"s{sp}|{kind}"], y, n) if n else 0))
            if f"s{sp}|pair|decomp|G" in z.files:
                yd = T.label(z, sp, "pair", "decomp", "95")
                decomp.append((float(yd.mean()), int(yd.sum()), len(yd)))
        for kind in ("single", "pair"):
            if rec[kind]:
                a = np.array(rec[kind], float)
                out[kind].append({"n_listed": a[:, 0].mean(), "beam_pos": a[:, 1].mean(),
                                  "beam_prec": float(np.sum(a[:, 1]) / max(np.sum(a[:, 0]), 1)),
                                  "ex_pos64": a[:, 2].mean(), "ex_prec64": float(a[:, 2].mean() / 64),
                                  "ex_pos_matched": a[:, 3].mean(), "ex_prec_matched": float(np.sum(a[:, 3]) / max(np.sum(a[:, 0]), 1))})
    return out, decomp


def boot(v, rng, B=2000):
    v = np.asarray(v, float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return f"{v.mean():+.3f} [{np.quantile(b, .025):+.3f}, {np.quantile(b, .975):+.3f}]"


def main():
    studies = sys.argv[1].split(",")
    rng = np.random.default_rng(0)
    lines = [MARK0, "", "## E1 matched budget (existing receipts and labels; B_known, 95%)", "",
             "The pool-AP rows above score the beam's 64 fitted sets and leave every other pool member tied at the bottom; they are 'pool AP, 64 scored sets', not a ranking failure of the beam's proposals. "
             "Matched budget: precision over the sets the beam proposed that carry a label (listed singles; listed pairs inside the 2,000-pair pool) against the top-n of exhaustive support-fit ordering with n = the beam's count of the same kind, and the top 64.", "",
             "| study | kind | targets | beam: mean labelled proposals | beam: mean positives | beam precision | exhaustive top-n (matched): mean positives | exhaustive precision (matched) | exhaustive top-64: mean positives | precision@64 | paired difference in precision (beam minus exhaustive, matched n) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    dec_lines = ["", "## B_decomp label prevalence (guarded by G_known > 0), pairs only", "",
                 "| study | mean prevalence in the 2,000-pair pool | mean positive pairs per (target, split) | (target, split) cells | cells with no positive pair |", "|---|---|---|---|---|"]
    for st in studies:
        out, decomp = study_stats(st)
        for kind in ("single", "pair"):
            r = out[kind]
            if not r:
                continue
            d = [x["beam_prec"] - x["ex_prec_matched"] for x in r]
            lines.append(f"| {st} | {kind} | {len(r)} | {np.mean([x['n_listed'] for x in r]):.1f} | {np.mean([x['beam_pos'] for x in r]):.2f} | {np.mean([x['beam_prec'] for x in r]):.3f} | "
                         f"{np.mean([x['ex_pos_matched'] for x in r]):.2f} | {np.mean([x['ex_prec_matched'] for x in r]):.3f} | {np.mean([x['ex_pos64'] for x in r]):.2f} | {np.mean([x['ex_prec64'] for x in r]):.3f} | {boot(d, rng)} |")
        if decomp:
            a = np.array(decomp)
            dec_lines.append(f"| {st} | {a[:, 0].mean():.4f} | {a[:, 1].mean():.1f} | {len(a)} | {int((a[:, 1] == 0).sum())} |")
    lines += dec_lines + ["", "At the prevalences and positive-pair counts in this table the B_decomp label is too rare to rank on at these cohort sizes; "
                          "an all-chance E2 on B_decomp is a power statement, not evidence about the rankings.", "", MARK1]
    block = "\n".join(lines) + "\n"
    dest = V5 / ("E1E2_" + "_".join(studies) + ".md")
    text = dest.read_text() if dest.exists() else ""
    if MARK0 in text:
        text = text[: text.index(MARK0)] + text[text.index(MARK1) + len(MARK1) + 1:]
    dest.write_text(text.rstrip("\n") + "\n\n" + block)
    print(block)


if __name__ == "__main__":
    main()
