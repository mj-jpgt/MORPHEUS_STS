"""Known-axis grader v2.1 and novelty tiers (COUNTEREXAMPLE_PLAN §5.2).

Why v2.1. v1 (ORA q <= 0.05 or adjacency >= 0.30) called 100 % of random gene sets 'known'. v2 compared an object's
frozen prediction f_S with null states built as the mean of 100 random genes; the two are different kinds of score
(a 100-gene mean is a strong averaged tissue axis), so the Moran p95 (0.73) sat above every real object and the
expression-matched null still gave 19.9 % false-known. v2.1 grades like with like.

Per object (or random set):
  decoded list D = top-100 expressed genes by within-patient correlation with the state, after removing the relation's own
  constituent genes and the target programme's genes (random sets: D = the set itself);
  s_D = module score of D: mean log-CPM over D minus, gene by gene, the mean of all expressed genes in the same
        mean-expression bin (25 bins; the expectation of the Seurat/Tirosh control-set score), then within-patient
        standardised (spot subsample, fixed seed). Every programme score is built the same way. Without the
        background subtraction a 100-gene mean is dominated by the shared detection/depth axis (v2.1 calibration:
        null p95 of the max |Spearman| = 0.74), so the arm could not separate anything;
  programme arm: max |Spearman(s_D, gene-disjoint programme score)| over the Gavish MPs and Hallmark sets, each
     programme scored on its genes minus D and minus the excluded genes; 'known programme' when > T_prog;
  overlap arm: max Jaccard(D, MSigDB C2/C5/C8 + Gavish + Hallmark) >= 0.2 also counts as 'known programme';
  composition arm: max |Spearman(s_D, CellViT PanNuke fraction)| over the 5 classes on covered spots; 'composition
     axis' when > T_comp.
  T_prog and T_comp = the larger of the 95th percentiles under the expression-matched and Moran-matched nulls,
  estimated on calibration draws (500 per null, seeds 101/103) separate from the evaluation draws (1,000 per null,
  seeds 11/13) on which the false rates are reported.

Novelty tiers (structure read from the relation's variables; compositional counts as novel):
  no novelty identified  known programme, own-spot molecular variables only;
  context extension      known programme and a morphology/composition/CNV context variable;
  spatial extension      known programme and a neighbourhood (nbr6/nbr18) or contrast variable (no context term);
  compositional novelty  not a known programme, composition axis;
  new association        neither.
Random sets carry no relation structure, so under the nulls the first three tiers collapse to 'known programme'; the
null rate of each statistical call (known programme, composition axis) is reported per tier.

    python -m dbench.downstream.scripts.grader_v21 STUDY OUT_JSON [--null] DIR:FID[:LABEL] ...
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from dbench.downstream.scripts.grader_v2 import JACC, Grader

N_CAL, N_EVAL = 500, 1000
PANNUKE = ["Neoplastic", "Connective", "Epithelial", "Inflammatory", "Dead"]


def _rz(v):
    r = rankdata(v)
    return (r - r.mean()) / (r.std() + 1e-12)


class G21(Grader):
    def decoded(self, state_sub, excl):
        """v2.1 decoding over expressed genes only (mean log-CPM above the 20th percentile, the null pool): without
        this, a state with no positive gene programme decodes to near-silent genes (olfactory receptors, TAARs)
        whose rare detections correlate by chance (seen on morph:k15 <- {DEK, HBA2})."""
        sc = self.sw(state_sub, self.pat[self.sub])[0].ravel()
        corr = np.nan_to_num(self.Xc.T @ sc / len(sc))
        ex = set(excl)
        ok = np.zeros(len(self.genes), bool)
        ok[self.expressed] = True
        order = [j for j in np.argsort(-corr) if ok[j] and self.genes[j] not in ex][:100]
        return list(self.genes[order]), sc

    def __init__(self, study):
        super().__init__(study)
        from dbench.downstream.scripts import preview_downstream as PD
        PD.STUDY = study
        try:
            comp = PD._discovery().cellvit()
            fr = np.asarray(comp["fractions"], float)[self.sub]
            cov = np.asarray(comp["covered"], bool)[self.sub]
            self.comp = (fr, cov)
        except Exception as e:  # no CellViT for this cohort: composition arm not testable (recorded)
            self.comp = None
            self.comp_note = f"{type(e).__name__}: {e}"

    def bin_ok(self, g):
        bo, _ = self._bins()
        return bo[self.gi[g]] >= 0

    def _bins(self):
        if not hasattr(self, "_bin_of"):
            m = self.means
            e = self.expressed
            edges = np.quantile(m[e], np.linspace(0, 1, 26)[1:-1])
            self._bin_of = np.full(len(m), -1)
            self._bin_of[e] = np.searchsorted(edges, m[e], side="right")
            self._bin_mean = np.column_stack([self.Xs[:, e[self._bin_of[e] == b]].mean(1) for b in range(25)])
        return self._bin_of, self._bin_mean

    def score(self, idx):
        bo, bm = self._bins()
        idx = np.asarray(idx)
        b = bo[idx]
        b = np.where(b < 0, 0, b)
        sc = self.Xs[:, idx].mean(1) - bm[:, b].mean(1)
        return self.sw(sc, self.pat[self.sub])[0].ravel()

    def prog_stat(self, sD, excl):
        rD = _rz(sD)
        ex = set(excl)
        best, name = 0.0, None
        for nm, gl in self.refprog.items():
            idx = [self.gi[g] for g in gl if g not in ex]
            if len(idx) < 5:
                continue
            v = abs(float(np.mean(rD * _rz(self.score(idx)))))
            if v > best:
                best, name = v, nm
        return best, name

    def comp_stat(self, sD):
        if self.comp is None:
            return None, None
        fr, cov = self.comp
        r = _rz(sD[cov])
        vals = [abs(float(np.mean(r * _rz(fr[cov, k])))) for k in range(fr.shape[1])]
        k = int(np.argmax(vals))
        return vals[k], PANNUKE[k] if k < len(PANNUKE) else str(k)

    def stats(self, D, excl):
        idx = [self.gi[g] for g in D]
        sD = self.score(idx)
        pv, pn = self.prog_stat(sD, list(D) + list(excl))
        cv, cn = self.comp_stat(sD)
        jv, jn = self.jaccard(D)
        return {"prog": pv, "prog_best": pn, "comp": cv, "comp_best": cn, "jacc": jv, "jacc_best": jn}


def structure(fid, variables):
    v = " ".join(variables)
    # read from the variables only (a family name such as context_conditioned_programme does not guarantee a context
    # term in the frozen relation)
    context = any(t in v for t in ("morph", "composition:", "cellvit", "hoptimus", "normal_appearing", "inferred_cnv"))
    spatial = any(x.startswith(("nbr6:", "nbr18:", "contrast:")) for x in variables)
    return context, spatial


def tier(st, T_prog, T_comp, context, spatial):
    known = st["prog"] > T_prog or st["jacc"] >= JACC
    comp = st["comp"] is not None and st["comp"] > T_comp
    if known:
        t = "context extension" if context else ("spatial extension" if spatial else "no novelty identified")
    else:
        t = "compositional novelty" if comp else "new association"
    return t, known, comp


def main():
    study, outp = sys.argv[1], Path(sys.argv[2])
    do_null = "--null" in sys.argv
    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    G = G21(study)
    from dbench.downstream.values import candidate_genes, parse_candidate
    objs = []
    for a in args:
        d, fid, *lab = a.split(":")
        fr = pickle.load(open(f"{d}/frozen.pkl", "rb"))[fid]
        ff, aux = fr["ff"], fr["aux"]
        state = np.full(len(G.pat), np.nan)
        state[aux["rows"]] = ff.rel_nuis.predict(aux["Zs"])
        st = state[G.sub]
        st = np.where(np.isfinite(st), st, np.nanmean(st))
        excl = []
        for c in list(ff.rel_nuis.candidates) + [ff.rel_nuis.target]:
            try:
                pc = parse_candidate(c)
                if pc["kind"] in ("gene", "programme"):
                    excl += candidate_genes(pc, G.sets, None)
            except Exception:
                pass
        D, _ = G.decoded(st, excl)
        objs.append({"dir": d, "fid": fid, "label": lab[0] if lab else "", "target": ff.rel_nuis.target,
                     "variables": list(ff.rel_nuis.candidates), "decoded_disjoint": D, "_excl": excl})
    # --lists=FILE: grade external gene lists directly (JSON {name: [genes]} or {name: {top100_positive: [...],
    # top100_negative: [...]}}; each side graded separately; genes absent from this panel dropped and counted)
    lists = {}
    for a in sys.argv:
        if a.startswith("--lists="):
            for nm, v in json.load(open(a.split("=", 1)[1])).items():
                sides = {"": v} if isinstance(v, list) else {f":{k}": v[k] for k in ("top100_positive", "top100_negative") if k in v}
                for sfx, gl in sides.items():
                    keep = [g for g in gl if g in G.gi and G.bin_ok(g)]
                    objs.append({"dir": "list", "fid": nm + sfx, "label": nm + sfx, "target": nm, "variables": [],
                                 "decoded_disjoint": keep, "_excl": [], "n_list": len(gl), "n_on_panel": len(keep)})
    templates = [np.array([G.gi[g] for g in o["decoded_disjoint"]]) for o in objs if o["decoded_disjoint"]]
    cal = {}
    for kind, seed in (("expression", 101), ("moran", 103)):
        S = [G.stats(gl, []) for _, gl in G.null(kind, templates, n=N_CAL, seed=seed)]
        cal[kind] = {"prog_p95": float(np.quantile([s["prog"] for s in S], 0.95)),
                     "comp_p95": (float(np.quantile([s["comp"] for s in S], 0.95)) if S[0]["comp"] is not None else None)}
        print("calibration", kind, cal[kind], flush=True)
    T_prog = max(c["prog_p95"] for c in cal.values())
    T_comp = max((c["comp_p95"] for c in cal.values() if c["comp_p95"] is not None), default=np.inf)
    res = {"rule": __doc__, "study": study, "calibration": cal, "T_prog": T_prog, "T_comp": T_comp,
           "composition_note": getattr(G, "comp_note", None), "objects": []}
    for o in objs:
        s = G.stats(o["decoded_disjoint"], o["_excl"])
        context, spatial = structure(o["fid"], o["variables"])
        t, known, comp = tier(s, T_prog, T_comp, context, spatial)
        rec = {k: v for k, v in o.items() if not k.startswith("_")}
        rec["decoded_disjoint"] = rec["decoded_disjoint"][:30]
        rec.update(stats=s, known_programme=known, known_by_overlap_arm=bool(s["jacc"] >= JACC),
                   known_by_programme_arm=bool(s["prog"] > T_prog), composition_axis=comp, context_term=context,
                   spatial_term=spatial, tier=t)
        res["objects"].append(rec)
        print(o["label"] or o["fid"][-12:], o["target"], "|", t, "| prog %.3f (%s) jacc %.3f comp %s" % (
            s["prog"], s["prog_best"], s["jacc"], None if s["comp"] is None else round(s["comp"], 3)), flush=True)
    if do_null:
        nulls = {}
        for kind, seed in (("expression", 11), ("moran", 13)):
            S = [G.stats(gl, []) for _, gl in G.null(kind, templates, n=N_EVAL, seed=seed)]
            tiers = [tier(s, T_prog, T_comp, False, False) for s in S]
            nulls[kind] = {"false_known_programme": float(np.mean([t[1] for t in tiers])),
                           "programme_arm": float(np.mean([s["prog"] > T_prog for s in S])),
                           "overlap_arm": float(np.mean([s["jacc"] >= JACC for s in S])),
                           "composition_axis": float(np.mean([t[2] for t in tiers])),
                           "tier_rates": {k: float(np.mean([t[0] == k for t in tiers]))
                                          for k in ("no novelty identified", "compositional novelty", "new association")}}
            print("null", kind, nulls[kind], flush=True)
        res["null_eval"] = nulls
    outp.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
