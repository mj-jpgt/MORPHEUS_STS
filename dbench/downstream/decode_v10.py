"""Checker v2.1 statistical decode (grader_v21 arms) for the frozen relation state of each distinct relation."""
import json
import sys
from pathlib import Path

import numpy as np

from dbench.downstream import characterise_v10 as ch
from dbench.downstream import novelty_v10 as nv

OUT = nv.OUTDIR / "decode"


def state_of(rel):
    from morpheus_encoder.learned_search import relation_v03 as rv
    base = rv.fit_baseline(rel.ys, rel.pat, rel.D, yq=rel.ys, C_q=rel.D)
    o = rv.fit_sets(rel.Z[:, None, :], base, rel.ys, rel.pat, Vq=rel.Z[:, None, :], yq=rel.ys, groups_q=rel.pat,
                    device=ch.DEVICE, return_beta=True)
    return np.asarray(o["pred_q"])[:, 0]


def main():
    from dbench.downstream.scripts.grader_v21 import G21, JACC, structure, tier
    from dbench.downstream.values import candidate_genes, parse_candidate

    OUT.mkdir(parents=True, exist_ok=True)
    cal = json.loads(nv.GRADER_NULL.read_text())
    T_prog, T_comp = cal["T_prog"], cal["T_comp"]
    ctx = ch.Ctx()
    G = G21(ch.STUDY)
    # row alignment between the characterisation context and the grader cohort (same raw order)
    assert len(G.pat) == ctx.n
    rels = []
    for st in sys.argv[1:]:
        rels += ch.relation_list(st)
    # calibration relations on Meylan (Hallmark controls)
    for r in rels:
        p = OUT / f"{r['relation_id']}.json"
        if p.exists():
            continue
        rel = ch.Relation(ctx, r["relation_id"], r["names"], r["target"], r["family"], r["row_views"])
        s = state_of(rel)
        state = np.full(ctx.n, np.nan)
        state[np.flatnonzero(rel.ok)] = s
        st = state[G.sub]
        st = np.where(np.isfinite(st), st, np.nanmean(st))
        excl = []
        for c in list(r["names"]) + [r["target"]]:
            try:
                pc = parse_candidate(c)
                base = pc["base"] if pc.get("kind") == "operator" else pc
                if base["kind"] in ("gene", "programme"):
                    excl += candidate_genes(base, G.sets, None)
            except Exception:
                pass
        D, _ = G.decoded(st, excl)
        stats = G.stats(D, excl)
        context, spatial = structure(r["relation_id"], r["names"])
        spatial = spatial or bool(ch.re.search(r"(ring|nbr|het|grad)", " ".join(r["names"])))
        tr, known, comp = tier(stats, T_prog, T_comp, context, spatial)
        rec = {"relation_id": r["relation_id"], "decoded_top30": D[:30], "stats": stats, "T_prog": T_prog, "T_comp": T_comp,
               "known_programme_by_statistic": bool(known), "composition_axis": bool(comp), "checker_tier": tr,
               "max_abs_rho_known_programme": stats["prog"], "best_known_programme": stats["prog_best"],
               "max_jaccard_geneset": stats["jacc"], "best_geneset": stats["jacc_best"],
               "max_abs_rho_cellvit_class": stats["comp"], "best_cellvit_class": stats["comp_best"],
               "false_known_rate_null": cal["null_eval"]}
        p.write_text(json.dumps(rec, indent=1, default=str))
        print(r["relation_id"], tr, stats["prog"], stats["prog_best"], flush=True)


if __name__ == "__main__":
    main()
