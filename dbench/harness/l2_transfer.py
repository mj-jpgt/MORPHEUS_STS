"""L2 platform transfer on Ji 2020 cSCC (supplementary; single query patient, section units).

Support = every ST (100 um) section (Patients 2, 5, 9, 10); query = Patient 1's four Visium sections; built through
the training-time L2 episode path (``SamplerV03.episode_for(..., query_study=Visium part, level="L2")``, the two
platform parts loaded as separate studies). No patient has both platforms in this cohort, so the query units for the
pooled replication labels and their bootstrap are Patient 1's Visium SECTIONS, not patients.

    python -m dbench.harness.l2_transfer labels --workers 3
    python -m dbench.harness.l2_transfer scores --methods g_theta,sel_screen,sel_screen_joint,sel_hybrid --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from . import pool

OUT = Path("runs/development/benchmark_v03/transfer_pool/ji2020_cscc_L2")
PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"


def _part(spec, platform: str):
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.episodes_v03 import study_v03_from_raw
    ids = tuple(s for s in spec.sample_ids if spec.platform_of[s] == platform)
    sp = dataclasses.replace(spec, sample_ids=ids, platform=platform)
    raw = hm.load_study(sp)
    return study_v03_from_raw(raw, name=f"{spec.key}_{platform}", platform=platform, organ=str(spec.organ).lower(),
                              spec=sp, role="transfer")


def context(n_targets: int = 20, seed: int = 20261022):
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search.episodes_v03 import SamplerV03, task_from_episode
    from .dev_metric import choose_targets
    from . import core
    spec = hm.study_specs(PARTITIONS)["ji2020_cscc"]
    st, vis = _part(spec, "st_100um"), _part(spec, "visium")
    targets = [t for t in choose_targets(st, n=3 * n_targets, seed=seed) if t in vis.index][:n_targets]
    sampler = SamplerV03([st], controlled_prob=0.0, seed=seed)
    built = []
    for i, t in enumerate(targets):
        rng = np.random.default_rng(seed + 1000 * i)
        try:
            ep = sampler.episode_for(st, t, list(map(str, st.patients)), list(map(str, vis.patients)), rng=rng,
                                     query_study=vis, level="L2")
        except ValueError as exc:
            print(f"{t}: {exc}", flush=True)
            continue
        task = task_from_episode(ep)
        s = task.s
        sec = np.asarray(task.sections).astype(str)
        groups = np.concatenate([np.asarray(task.groups[:s]).astype(str), np.char.add("section:", sec[s:])])
        task = dataclasses.replace(task, groups=groups)    # query units = Visium sections (single query patient)
        built.append((t, 0, [("real", task)], list(ep.meta.names), ep.study, sorted(set(sec[s:]))))
    sha = core.registry_hash({"study": "ji2020_cscc_L2", "targets": targets, "seed": seed})
    return {"sha256": sha, "targets": [b[0] for b in built], "study": "ji2020_cscc_L2"}, built


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("labels", "scores", "table"))
    ap.add_argument("--methods", default="g_theta,sel_screen,sel_screen_joint,sel_hybrid")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "table":
        from .transfer_compare import main as cmp
        print(cmp(str(OUT)))
        return
    ctx = context()
    print("targets", len(ctx[1]), "query units", ctx[1][0][5] if ctx[1] else None, flush=True)
    if a.mode == "labels":
        pool.compute_labels(a.workers, None, OUT, ctx=ctx)
    else:
        for m in a.methods.split(","):
            if not (OUT / f"scores_{m}_known.json").exists():
                pool.compute_scores([m], 1, a.device, OUT, ctx=ctx)


if __name__ == "__main__":
    main()
