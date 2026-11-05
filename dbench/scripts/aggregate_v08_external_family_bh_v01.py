"""One Benjamini-Hochberg family over every internally supported v08 candidate that was scored externally.

The primitive runner computes bh_q within a batch of 2, which is not a family quantity.  Here the pooled
no-refit p (bootstrap p of the gain over the nuisance baseline, as in aggregate_v08_external_batches_v01.py) of every
tested candidate is corrected once; candidates not testable externally have no p and are listed as such.  The frozen
rule is unchanged: nuisance CI lower bound > 0, family BH q <= 0.05, direction agreement >= 0.60.
Proposed-only candidates form a separate exploratory family (never merged into the confirmatory one).

variant "masked": ring_decay candidates whose first external score used a refit on all rows (ring1-only, no NaN) are
taken from the rerun that refits on the rows registered discovery kept; variant "asrun": the original receipts.
Also writes per-candidate family receipts (cards_family_<variant>/<id>/records/02_external_replication.record.json) so
that ingest_external_confirmation_v01.py ingests the family-wide verdict rather than the per-batch one.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from dbench.scripts.aggregate_v08_external_batches_v01 import bh

RUN = "/home/ubuntu/morpheus_work_v08/runs/registered_v08/renal-spatial-v08-multiscale-20261026"
CONF = Path("runs/confirmation")
DECLARED_ROOTS = ["v08_downstream_strict_spatial_v08", "v08_external_batched_ring_heter_internal_v01",
                  "v08_external_batched_compartment_internal_v01", "v08_external_batched_residual_internal_v01"]
RERUN_ROOTS = ["v08_external_batched_ring_heter_internal_v03", "v08_external_batched_compartment_internal_v03"]
MASKED_ROOT = "v08_external_batched_ring1_masked_sensitivity_v02"
EXPLORATORY_ROOTS = ["v08_external_batched_heter_proposed_v01"]
INTERNAL_STAGES = ("internal", "external", "orthogonal", "actionable")


def load(root: str) -> dict:
    out = {}
    for f in glob.glob(str(CONF / root / "**/records/02_external_replication.record.json"), recursive=True):
        j = json.loads(Path(f).read_text())
        out[j["finding_id"]] = {"result": j["result"], "receipt": f, "root": root}
    return out


def family_table(sources: dict, rows: dict, ids: list) -> tuple[list, int]:
    tested = [i for i in ids if sources[i]["result"].get("pooled", {}).get("nuis")]
    q = bh([sources[i]["result"]["pooled"]["nuis"]["p_boot"] for i in tested]) if tested else []
    qmap = dict(zip(tested, q))
    # conservative alternative: every candidate of the declared family in the denominator (untestable p = 1)
    qall = dict(zip(ids, bh([sources[i]["result"]["pooled"]["nuis"]["p_boot"] if i in qmap else 1.0 for i in ids])))
    table = []
    for i in ids:
        res, row = sources[i]["result"], rows[i]
        rec = {"candidate_id": i, "family": row["family"], "target": row["target"], "candidates": row["variable_names"],
               "source_root": sources[i]["root"], "receipt": sources[i]["receipt"]}
        if i in qmap:
            p = res["pooled"]
            dirn = res.get("direction", {}).get("fraction_agree")
            crit = {"ci_excludes_zero": bool(p["nuis"]["ci"][0] > 0), "bh_q_le_0.05": bool(qmap[i] <= 0.05),
                    "direction_agreement_ge_0.60": bool(dirn is not None and dirn >= 0.60)}
            rec.update(status="tested",
                       gain={k: {"gain": p[k]["gain"], "ci95": p[k]["ci"], "p_boot": p[k]["p_boot"]}
                             for k in ("nuis", "known", "decomp") if k in p},
                       direction_pooled=dirn,
                       direction_per_cohort={c: v.get("direction_agreement") for c, v in res.get("per_cohort", {}).items()},
                       n_patients_per_cohort={c: v.get("n_patients") for c, v in res.get("per_cohort", {}).items()},
                       batch_bh_q=res.get("bh_q"), family_bh_q=float(qmap[i]), family_bh_q_all_declared=float(qall[i]),
                       criteria=crit, replicated=bool(all(crit.values())))
        else:
            tb = res.get("testability") or {}
            rec.update(status="not_testable", replicated=False,
                       reason=res.get("reason") or "; ".join(sorted({str(v.get("reason")) for v in tb.values()
                                                                    if isinstance(v, dict) and not v.get("testable", True)})))
        table.append(rec)
    return table, len(tested)


def dedup(table: list, jaccard: float = 0.5) -> list:
    rep = [r for r in table if r.get("replicated")]
    parent = list(range(len(rep)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(rep)):
        for b in range(a + 1, len(rep)):
            if rep[a]["target"] != rep[b]["target"]:
                continue
            sa, sb = set(rep[a]["candidates"]), set(rep[b]["candidates"])
            if len(sa & sb) / len(sa | sb) >= jaccard:
                parent[find(a)] = find(b)
    groups = {}
    for i, r in enumerate(rep):
        groups.setdefault(find(i), []).append(r)
    out = []
    for g in groups.values():
        best = max(g, key=lambda r: r["gain"]["nuis"]["gain"])
        shared = set.intersection(*[set(r["candidates"]) for r in g])
        out.append({"target": g[0]["target"], "n_members": len(g), "families": sorted({r["family"] for r in g}),
                    "shared_variables": sorted(shared), "best_member": best["candidate_id"],
                    "best_candidates": best["candidates"], "best_gain_nuis": best["gain"]["nuis"],
                    "best_gain_known": best["gain"].get("known"), "best_gain_decomp": best["gain"].get("decomp"),
                    "best_family_bh_q": best["family_bh_q"], "best_direction": best["direction_pooled"],
                    "members": [r["candidate_id"] for r in g]})
    return sorted(out, key=lambda x: -x["n_members"])


def fmt(g):
    return f"{g['gain']:.3f} [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]" if g else "n/a"


def markdown(title, table, ntested) -> str:
    lines = [f"## {title}", "", f"{len(table)} candidates, {ntested} tested, "
             f"{sum(r['replicated'] for r in table)} replicated under the frozen rule.", "",
             "| family | target | candidates | gain over nuisance [95% CI] | over known biology | over decomposition | "
             "direction (pooled; per cohort) | family BH q | replicated |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(table, key=lambda r: (r["status"] != "tested", r.get("family_bh_q", 9), r["candidate_id"])):
        short = r["candidate_id"].split("-")[-1]
        if r["status"] != "tested":
            lines.append(f"| {r['family']} | {r['target']} | {', '.join(r['candidates'])} ({short}) | not testable: "
                         f"{(r['reason'] or '')[:90]} | | | | | no |")
            continue
        pc = ", ".join(f"{c[:6]} {v:.2f}" for c, v in r["direction_per_cohort"].items() if v is not None)
        lines.append(f"| {r['family']} | {r['target']} | {', '.join(r['candidates'])} ({short}) | "
                     f"{fmt(r['gain']['nuis'])} | {fmt(r['gain'].get('known'))} | {fmt(r['gain'].get('decomp'))} | "
                     f"{r['direction_pooled']:.2f}; {pc} | {r['family_bh_q']:.4f} | {'yes' if r['replicated'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = {r["candidate_id"]: r for r in json.loads((Path(RUN) / "funnel_aggregate.json").read_text())["table"]}
    ledger = {}
    for ln in (Path(RUN) / "renal_spatial_v08_multiscale_ledger.jsonl").read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            ledger[e["candidate_id"]] = e["stage"]
    internal = sorted(i for i in rows if ledger.get(i) in INTERNAL_STAGES)
    for variant in ("masked", "asrun"):
        src = {}
        for root in DECLARED_ROOTS:
            src.update(load(root))
        for root in RERUN_ROOTS:
            for i, v in load(root).items():
                if v["result"].get("pooled") or i not in src or not src[i]["result"].get("pooled"):
                    src[i] = v
        if variant == "masked":
            src.update({i: v for i, v in load(MASKED_ROOT).items() if v["result"].get("pooled")})
        expl = {}
        for root in EXPLORATORY_ROOTS:
            expl.update(load(root))
        missing = [i for i in internal if i not in src]
        table, nt = family_table(src, rows, [i for i in internal if i in src])
        etable, ent = family_table(expl, rows, sorted(expl))
        groups = dedup(table)
        doc = {"schema": "morpheus_v08_external_family_bh/v01", "variant": variant,
               "rule": {"ci_excludes_zero": "nuisance-gain CI lower bound > 0", "bh_q": 0.05, "direction": 0.60},
               "n_internal_supported": len(internal), "missing_receipts": missing, "n_tested": nt,
               "n_replicated": sum(r["replicated"] for r in table), "confirmatory": table,
               "exploratory_proposed_only": {"n": len(etable), "n_tested": ent,
                                             "n_replicated": sum(r["replicated"] for r in etable), "table": etable},
               "dedup_groups": groups}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"family_bh_{variant}.json").write_text(json.dumps(doc, indent=2) + "\n")
        md = (markdown(f"Confirmatory family ({variant}): internally supported v08 candidates", table, nt) + "\n" +
              markdown(f"Exploratory family ({variant}): proposed-only candidates", etable, ent))
        (args.out / f"family_bh_{variant}.md").write_text(md)
        cards = args.out / f"cards_family_{variant}"
        for r in table + etable:
            res = json.loads(Path(r["receipt"]).read_text())
            res["result"] = {**res["result"], "bh_q": r.get("family_bh_q"), "criteria": r.get("criteria"),
                             "externally_replicated": r["replicated"], "bh_family": "v08_external_family_bh_v01/" + variant}
            d = cards / r["candidate_id"] / "records"
            d.mkdir(parents=True, exist_ok=True)
            (d / "02_external_replication.record.json").write_text(json.dumps(res, indent=1))
        print(variant, json.dumps({k: doc[k] for k in ("n_internal_supported", "n_tested", "n_replicated")}),
              "missing", len(missing), "groups", len(groups))


if __name__ == "__main__":
    main()
