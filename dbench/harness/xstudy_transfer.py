"""Cross-study L2 transfer (platform + preparation shift, same organ): support = every patient of one study, query = every
patient of another, built through the training-time L2 episode path (SamplerV03.episode_for(..., query_study=...,
level="L2")).  Shared pool, B_known labels (as registered and strict beyond-known: positives containing an excluded
programme score, a member gene of an excluded programme or a programme containing the gene target are negatives,
from each task's own known_names / panel), pool scores for ours / screen / joint screen / hybrid.

    python -m dbench.harness.xstudy_transfer --support meylan2022_ccrcc --query gse250163 labels
    python -m dbench.harness.xstudy_transfer --support meylan2022_ccrcc --query gse250163 scores --device cuda
    python -m dbench.harness.xstudy_transfer --support meylan2022_ccrcc --query gse250163 table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import pool

PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"
ROOT = Path("runs/development/benchmark_v03/transfer_pool")


def _load(key: str):
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.episodes_v03 import study_v03_from_raw
    spec = hm.study_specs(PARTITIONS)[key]
    raw = hm.load_study(spec)
    return study_v03_from_raw(raw, name=key, platform=spec.platform, organ=str(spec.organ).lower(), spec=spec, role="transfer")


def context(support: str, query: str, n_targets: int = 20, seed: int = 20261022):
    from morpheus_encoder.learned_search.episodes_v03 import SamplerV03, task_from_episode
    from .dev_metric import choose_targets
    from . import core
    st, qs = _load(support), _load(query)
    targets = [t for t in choose_targets(st, n=3 * n_targets, seed=seed) if t in qs.index][:n_targets]
    sampler = SamplerV03([st], controlled_prob=0.0, seed=seed)
    built = []
    for i, t in enumerate(targets):
        try:
            ep = sampler.episode_for(st, t, list(map(str, st.patients)), list(map(str, qs.patients)),
                                     rng=np.random.default_rng(seed + 1000 * i), query_study=qs, level="L2")
        except ValueError as exc:
            print(f"{t}: {exc}", flush=True)
            continue
        task = task_from_episode(ep)
        built.append((t, 0, [("real", task)], list(ep.meta.names), ep.study, sorted(map(str, qs.patients))))
    sha = core.registry_hash({"support": support, "query": query, "targets": targets, "seed": seed})
    return {"sha256": sha, "targets": [b[0] for b in built], "study": f"{support}->{query}"}, built


def strict_labels(out: Path, built) -> dict:
    """labels_flagged.json: B_known positives with any strict flag set to False (per task known_names / panel)."""
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.relation_v03 import known_exclusions
    from .dev_flags import _var
    lab = json.loads((out / "labels.json").read_text())
    res = json.loads(json.dumps(lab))
    counts = {"single": [0, 0], "pair": [0, 0]}
    for t, f, conds, names, st, q in built:
        task = dict(conds)["real"]
        kn = [str(x) for x in (task.known_names or [])]
        excluded = {kn[i] for i in known_exclusions(kn, t, task.panel)} if kn else set()
        genes = set().union(*[set(map(str, hm._programme_members(p))) for p in excluded]) if excluded else set()
        g = str(t).split(":", 1)[1] if str(t).startswith("gene:") else None

        def flagged(members):
            vs = [_var(v) for v in members]
            return any((k == "programme" and n in excluded) or (k == "gene" and n in genes)
                       or (g is not None and k == "programme" and g in set(map(str, hm._programme_members(n)))) for k, n in vs)
        pl = lab["pools"][t]
        for kind, members in (("single", [[s] for s in pl["singles"]]), ("pair", pl["pairs"])):
            key = f"{t}|{kind}|known"
            y = list(lab["labels"][key])
            for j, mm in enumerate(members):
                if y[j]:
                    counts[kind][0] += 1
                    if flagged(mm):
                        counts[kind][1] += 1
                        y[j] = False
            res["labels"][key] = y
    res["flag_counts_positives_removed"] = counts
    (out / "labels_flagged.json").write_text(json.dumps(res))
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("labels", "scores", "table"))
    ap.add_argument("--support", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--methods", default="g_theta,sel_screen,sel_screen_joint,sel_hybrid")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    out = ROOT / f"{a.support}__to__{a.query}"
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "table":
        from .transfer_compare import main as cmp
        print(cmp(str(out)))
        flagged = out / "labels_flagged.json"
        if flagged.exists():   # the strict version: same comparison against the strict labels
            strict = out / "strict"
            strict.mkdir(exist_ok=True)
            for p in out.glob("scores_*_known.json"):
                (strict / p.name).unlink(missing_ok=True)
                (strict / p.name).symlink_to(p.resolve())
            (strict / "labels.json").unlink(missing_ok=True)
            (strict / "labels.json").symlink_to(flagged.resolve())
            print(cmp(str(strict)))
        return
    ctx = context(a.support, a.query)
    print("targets", len(ctx[1]), "query patients", ctx[1][0][5] if ctx[1] else None, flush=True)
    if a.mode == "labels":
        if not (out / "labels.json").exists():
            pool.compute_labels(a.workers, None, out, ctx=ctx)
        print(json.dumps(strict_labels(out, ctx[1])), flush=True)
    else:
        for m in a.methods.split(","):
            if not (out / f"scores_{m}_known.json").exists():
                pool.compute_scores([m], 1, a.device, out, ctx=ctx)


if __name__ == "__main__":
    main()
