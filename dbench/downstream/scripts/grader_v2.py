"""Known-axis grader v2 (COUNTEREXAMPLE_PLAN §5.2; replaces the v1 call, which labelled 100 % of random gene sets
'known'). Gene-disjoint and null-calibrated.

For an object (state = frozen f_S on the discovery rows) the excluded genes are the relation's own constituent genes
and the target programme's genes. Two arms:
 (a) overlap: decoded list = the 100 genes with the highest within-patient correlation with the state AFTER removing
     the excluded genes; 'known' when max Jaccard vs MSigDB C2/C5/C8 + Gavish MPs + Hallmark >= 0.2;
 (b) gene-disjoint correlation: for every Gavish MP and Hallmark programme, score = mean log-CPM over the programme's
     genes minus the excluded genes (within-patient standardised); stat = max |Spearman(state, score)| on a fixed spot
     subsample; 'known' when stat > p95 of the same statistic for Moran-matched random gene-set states (null computed
     here, per cohort, the random set's own genes excluded from the programme scores).
'known' = (a) or (b); the best-matching reference sets are reported.

Null mode reruns 1,000 expression-matched and 1,000 Moran-matched random sets through the v2 grader and reports the
false-known rate.

    python -m dbench.downstream.scripts.grader_v2 STUDY OUT_JSON [--null] DIR:FID[:LABEL] ...
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

GMTS = ["runs/development/novelty_positive_control/genesets/c2.all.v2024.1.Hs.symbols.gmt",
        "runs/development/novelty_positive_control/genesets/c5.all.v2024.1.Hs.symbols.gmt",
        "runs/development/novelty_positive_control/genesets/c8.all.v2024.1.Hs.symbols.gmt"]
N_NULL, N_SUB, JACC = 1000, 15000, 0.2


def read_gmt(p):
    out = {}
    for ln in open(p):
        f = ln.rstrip("\n").split("\t")
        if len(f) > 2:
            out[f[0]] = set(x for x in f[2:] if x)
    return out


class Grader:
    def __init__(self, study):
        from morpheus.tier2.substrates.hest_multi import _programme_sets, section_moran_i

        from dbench.downstream.discovery_any import cohort
        from dbench.downstream.relation import standardise_within

        self.sw = standardise_within
        self.coh = cohort(study)
        raw = self.coh.raw
        self.sets = _programme_sets()
        self.genes = np.array([str(g) for g in raw.genes])
        self.gi = {g: i for i, g in enumerate(self.genes)}
        self.pat = np.asarray(raw.patient).astype(str)
        self.sec = np.asarray(raw.sample_id).astype(str)
        rng = np.random.default_rng(20261027)
        n = len(self.pat)
        self.sub = np.sort(rng.choice(n, min(N_SUB, n), replace=False))
        X = self.coh.logcpm.tocsc()
        self.X = X
        self.Xs = np.asarray(X[self.sub].todense(), float)
        self.Xc = standardise_within(self.Xs, self.pat[self.sub])[0].reshape(len(self.sub), -1)
        self.refprog = {f"{k}:{s['name']}": [g for g in s["genes"] if g in self.gi] for k in ("mp", "hallmark")
                        for s in self.sets.get(k, [])}
        self.big = {}
        for p in GMTS:
            self.big.update(read_gmt(p))
        for k in ("mp", "hallmark"):
            for s in self.sets.get(k, []):
                self.big[f"{k}:{s['name']}"] = set(s["genes"])
        means = np.asarray(X.mean(axis=0)).ravel()
        self.expressed = np.flatnonzero(means > np.quantile(means, 0.2))
        self.means = means
        self._section_moran_i = section_moran_i
        self.raw = raw

    def stat_corr(self, state_sub, excl):
        """max |Spearman| of the state with gene-disjoint reference programme scores (on the subsample)."""
        from scipy.stats import rankdata
        r = rankdata(state_sub)
        r = (r - r.mean()) / (r.std() + 1e-12)
        best, name = 0.0, None
        ex = set(excl)
        for nm, gl in self.refprog.items():
            idx = [self.gi[g] for g in gl if g not in ex]
            if len(idx) < 5:
                continue
            sc = self.Xs[:, idx].mean(1)
            sc = self.sw(sc, self.pat[self.sub])[0].ravel()
            q = rankdata(sc)
            q = (q - q.mean()) / (q.std() + 1e-12)
            v = abs(float(np.mean(r * q)))
            if v > best:
                best, name = v, nm
        return best, name

    def decoded(self, state_sub, excl):
        sc = self.sw(state_sub, self.pat[self.sub])[0].ravel()
        corr = np.nan_to_num(self.Xc.T @ sc / len(sc))
        ex = set(excl)
        order = [j for j in np.argsort(-corr) if self.genes[j] not in ex][:100]
        return list(self.genes[order]), sc

    def jaccard(self, gl):
        g = set(gl)
        best, name = 0.0, None
        for nm, s in self.big.items():
            if not s:
                continue
            v = len(g & s) / len(g | s)
            if v > best:
                best, name = v, nm
        return best, name

    def grade_state(self, state_sub, excl, p95):
        gl, sc = self.decoded(state_sub, excl)
        jv, jn = self.jaccard(gl)
        cv, cn = self.stat_corr(sc, excl)
        return {"jaccard_disjoint": jv, "jaccard_best_set": jn, "corr_disjoint": cv, "corr_best_programme": cn,
                "corr_null_p95": p95, "known_overlap": bool(jv >= JACC), "known_corr": bool(p95 is not None and cv > p95),
                "known": bool(jv >= JACC or (p95 is not None and cv > p95)), "decoded_disjoint_top20": gl[:20]}

    def moran_deciles(self):
        if hasattr(self, "_mdec"):
            return self._mdec
        mor = np.full(len(self.means), np.nan)
        for j in self.expressed:
            v = np.asarray(self.X[:, j].toarray()).ravel()
            m = self._section_moran_i(v, self.raw.neighbours[6], self.sec)
            mor[j] = np.median(list(m.values())) if m else np.nan
        edges = np.nanquantile(mor[self.expressed], np.linspace(0, 1, 11)[1:-1])
        d = np.full(len(self.means), -1)
        d[self.expressed] = np.searchsorted(edges, mor[self.expressed], side="right")
        self._mdec = d
        return d

    def null(self, kind, templates, n=N_NULL, seed=7):
        rng = np.random.default_rng(seed)
        if kind == "moran":
            d = self.moran_deciles()
        else:
            edges = np.quantile(self.means[self.expressed], np.linspace(0, 1, 11)[1:-1])
            d = np.full(len(self.means), -1)
            d[self.expressed] = np.searchsorted(edges, self.means[self.expressed], side="right")
        pools = {k: np.flatnonzero(d == k) for k in range(10)}
        out = []
        for i in range(n):
            t = templates[i % len(templates)]
            idx = np.array([rng.choice(pools[d[g]]) if d[g] >= 0 else rng.choice(self.expressed) for g in t])
            st = self.Xs[:, idx].mean(1)
            gl_null = list(self.genes[idx])
            out.append((st, gl_null))
        return out


def main():
    from dbench.downstream.scripts import preview_downstream as PD

    study, outp = sys.argv[1], Path(sys.argv[2])
    do_null = "--null" in sys.argv
    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    PD.STUDY = study
    G = Grader(study)
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
        objs.append({"dir": d, "fid": fid, "label": lab[0] if lab else "", "target": ff.rel_nuis.target,
                     "variables": ff.rel_nuis.candidates, "_st": st, "_excl": excl})
    # Moran-matched null for the correlation arm (p95), per cohort; templates = objects' decoded lists
    templates = []
    for o in objs:
        gl, _ = G.decoded(o["_st"], o["_excl"])
        templates.append(np.array([G.gi[g] for g in gl]))
    if not templates:
        templates = [G.rng_templates] if hasattr(G, "rng_templates") else [np.random.default_rng(0).choice(G.expressed, 100, replace=False)]
    mn = G.null("moran", templates, n=200 if not do_null else N_NULL)
    nstats = [G.stat_corr(G.sw(st, G.pat[G.sub])[0].ravel(), gl)[0] for st, gl in mn]
    p95 = float(np.quantile(nstats, 0.95))
    res = {"rule": __doc__, "study": study, "corr_null_p95_moran": p95, "n_null_for_p95": len(nstats), "objects": []}
    for o in objs:
        g = G.grade_state(o["_st"], o["_excl"], p95)
        rec = {k: v for k, v in o.items() if not k.startswith("_")}
        rec.update(g)
        res["objects"].append(rec)
        print(o["label"] or o["fid"][-12:], "known" if g["known"] else "NOT known", round(g["jaccard_disjoint"], 3),
              g["jaccard_best_set"], round(g["corr_disjoint"], 3), g["corr_best_programme"], flush=True)
    if do_null:
        nulls = {}
        for kind in ("expression", "moran"):
            sets_ = G.null(kind, templates, n=N_NULL, seed=11 if kind == "expression" else 13)
            calls = [G.grade_state(st, gl, p95) for st, gl in sets_]
            nulls[kind] = {"false_known_rate_v2": float(np.mean([c["known"] for c in calls])),
                           "overlap_arm": float(np.mean([c["known_overlap"] for c in calls])),
                           "corr_arm": float(np.mean([c["known_corr"] for c in calls]))}
            print(kind, nulls[kind], flush=True)
        res["null_v2"] = nulls
    outp.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
