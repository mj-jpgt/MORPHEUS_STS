"""Episodic training of the learned search model.

Losses per episode (summed over a batch of 4 real + 4 controlled episodes):

* hindsight relation loss: min over the sampled paths of (1 - query R^2) of the
  relation refitted on support rows, + 0.1 x the mean over paths;
* teacher-forced pointer cross-entropy towards the planted set (controlled
  episodes, weight 0.5);
* path diversity (soft-overlap, 0.01) and concept decorrelation (0.01);
* G_HO hinge relu(0.002 - G_HO) on controlled residualised / neural episodes
  with >= 2 true variables (0.1).

AdamW 3e-4 (concept layer 1e-4), weight decay 0.01, linear warm-up then cosine,
gradient clip 1; bf16 autocast with ``--amp`` (off by default: at these sizes the
model is kernel-launch-bound and autocast's casts doubled the step time, 1.8 s vs
0.9 s per batch of 8 on the A100, while memory stays ~0.2 GiB).  A fixed development set (controlled episodes
from ``generators.HELDOUT_FAMILIES`` + real tasks on development patients, plus
16 residualised_2 episodes at 5 % for the G_HO readout) is scored every
``--dev-every`` steps; the checkpoint with the best ``recovered_set@16`` is kept.

    python -m morpheus_encoder.learned_search.train --studies synthetic --steps 300 \
        --out runs/development/learned_search_v01/synthetic_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .episodes import (FRACTIONS, TRAIN_FRACTION_RANGE, EpisodeSampler, RowSplitSampler, Study, fixed_dev_set, make_loader,
                       study_from_bench_substrate, synthetic_study)
from .model import LearnedSearchModel, ModelConfig

BENCH_CACHE = "runs/development/benchmark_v02/cache"


# --------------------------------------------------------------------- studies
def build_studies(spec: str, *, synthetic_embed_dim: int = 64) -> tuple[list, list, dict]:
    """``synthetic`` | ``bench:kidney_fold0`` | ``hest:<partitions.json>``, comma-separated."""
    train, dev, info = [], [], {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part == "synthetic":
            train += [synthetic_study(s, name=f"synthetic{s}", embed_dim=synthetic_embed_dim) for s in range(3)]
            dev += [synthetic_study(100, name="synthetic_dev", embed_dim=synthetic_embed_dim)]
            info["synthetic"] = {"train_seeds": [0, 1, 2], "dev_seed": 100}
        elif part.startswith("bench:"):
            name = part.split(":", 1)[1]
            fold = int(name.rsplit("fold", 1)[1]) if "fold" in name else 0
            from dbench.benchmark.substrate import load_hest_substrate
            sub = load_hest_substrate(fold, cache_dir=BENCH_CACHE)
            st = study_from_bench_substrate(sub, name=name)
            train.append(st)
            dev.append(st)
            info[name] = {"digest": sub.digest(), "n_rows": int(sub.n), "train_patients": len(st.train_patients),
                          "dev_patients": len(st.dev_patients)}
        elif part.startswith("hest:"):
            from .episodes import studies_from_partitions
            spec = part.split(":", 1)[1]
            only = None
            if ".json:" in spec:
                spec, names = spec.split(".json:", 1)
                spec, only = spec + ".json", set(names.split("+"))
            t, d, i = studies_from_partitions(spec, only=only)
            train += t
            dev += d
            info["hest"] = i
        else:
            raise ValueError(f"unknown study spec {part!r}")
    return train, dev, info


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model: LearnedSearchModel, dev_eps: list, *, n_paths: int = 32, ks=(8, 16)) -> dict:
    model.eval()
    rows = []
    for i, ep in enumerate(dev_eps):
        prep = model.prepare(ep)
        ranked = model.rank_sets(prep, n_paths=n_paths, seed=i, g_ho_top=1, rank_by="gain")
        n_s, y = prep["n_s"], prep["y"]
        row = {"kind": ep.kind, "family": ep.family, "fraction": ep.fraction, "n_sets": len(ranked)}
        if ranked:
            top = list(ranked[0][0])
            V = prep["X"][:, top]
            row["top_query_r2"] = float(model.relation.query_r2(V[:n_s], y[:n_s], V[n_s:], y[n_s:]))
            row["top_k"] = len(top)
            if len(top) >= 2:
                row["top_g_ho_query"] = float(model.relation.g_ho(V[:n_s], y[:n_s], V[n_s:], y[n_s:]))
        if ep.kind == "controlled":
            truth = tuple(sorted(ep.true_idx))
            pos = [j for j, (key, _, _) in enumerate(ranked) if tuple(sorted(key)) == truth]
            by_cv = sorted(ranked, key=lambda t: -t[1])
            pos_c = [j for j, (key, _, _) in enumerate(by_cv) if tuple(sorted(key)) == truth]
            row["rank"] = pos[0] + 1 if pos else None
            row["found"] = bool(pos)
            tset = set(truth)
            for k in ks:
                top = [set(key) for key, _, _ in ranked[:k]]
                row[f"any_truth@{k}"] = any(tset & t for t in top)
                row[f"truth_var_recall@{k}"] = len(tset & set().union(*top)) / len(tset) if top else 0.0
            for k in ks:
                row[f"hit@{k}"] = bool(pos and pos[0] < k)
                row[f"hit_cv@{k}"] = bool(pos_c and pos_c[0] < k)   # ranked by support CV R^2
            V = prep["X"][:, list(truth)]
            row["oracle_query_r2"] = float(model.relation.query_r2(V[:n_s], y[:n_s], V[n_s:], y[n_s:]))
        rows.append(row)
    model.train()

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    held = [r for r in rows if r["kind"] == "controlled" and not str(r["family"]).startswith("residualised")]
    res = [r for r in rows if r["kind"] == "controlled" and str(r["family"]).startswith("residualised")]
    real = [r for r in rows if r["kind"] == "real"]
    out = {}
    for k in ks:
        out[f"recovered_set@{k}"] = mean([float(r[f"hit@{k}"]) for r in held])
        for f in FRACTIONS:
            out[f"recovered_set@{k}_f{f:g}"] = mean([float(r[f"hit@{k}"]) for r in held if r["fraction"] == f])
            out[f"any_truth_variable@{k}_f{f:g}"] = mean([float(r[f"any_truth@{k}"]) for r in held
                                                         if r["fraction"] == f])
        out[f"residualised_recovered@{k}"] = mean([float(r[f"hit@{k}"]) for r in res])
        out[f"recovered_set_cvrank@{k}"] = mean([float(r[f"hit_cv@{k}"]) for r in held])
        out[f"residualised_recovered_cvrank@{k}"] = mean([float(r[f"hit_cv@{k}"]) for r in res])
        out[f"any_truth_variable@{k}"] = mean([float(r[f"any_truth@{k}"]) for r in held])
        out[f"truth_variable_recall@{k}"] = mean([r[f"truth_var_recall@{k}"] for r in held])
        out[f"truth_variable_recall@{k}_f0.05"] = mean([r[f"truth_var_recall@{k}"] for r in held
                                                         if r["fraction"] == 0.05])
    out["found_in_paths"] = mean([float(r["found"]) for r in held])
    for f in FRACTIONS:
        out[f"found_in_paths_f{f:g}"] = mean([float(r["found"]) for r in held if r["fraction"] == f])
        out[f"oracle_residual_r2_f{f:g}"] = mean([r.get("oracle_query_r2") for r in held if r["fraction"] == f])
    out["selection_score"] = mean([out.get(f"recovered_set@16_f{f:g}") for f in FRACTIONS if f >= 0.05])
    out["fraction_grid"] = list(FRACTIONS)
    out["residualised_found_in_paths"] = mean([float(r["found"]) for r in res])
    fams = sorted({r["family"] for r in held})
    out["recovered_set@16_by_family"] = {f: mean([float(r["hit@16"]) for r in held if r["family"] == f]) for f in fams}
    out["real_top_query_r2"] = mean([r.get("top_query_r2") for r in real])
    out["controlled_top_query_r2"] = mean([r.get("top_query_r2") for r in held])
    out["controlled_oracle_query_r2"] = mean([r.get("oracle_query_r2") for r in held])
    out["residualised_mean_g_ho_query"] = mean([r.get("top_g_ho_query") for r in res])
    out["mean_sets_evaluated"] = mean([r["n_sets"] for r in rows])
    out["n_dev"] = {"heldout_controlled": len(held), "residualised": len(res), "real": len(real)}
    return out


# ----------------------------------------------------------------------- main
def _schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selector", default="pointer", choices=("pointer", "topk"))
    ap.add_argument("--encoder", default="sab", choices=("sab", "isab"))
    ap.add_argument("--rowsplit", action="store_true", help="ablation: support and query rows from the same patients")
    ap.add_argument("--no-concepts", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-frac", type=float, default=0.075)
    ap.add_argument("--gpu-max-used-mib", type=int, default=34000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dev-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--n-dev-controlled", type=int, default=64)
    ap.add_argument("--n-dev-real", type=int, default=64)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--n-controlled", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--frozen-concept", default=None,
                    help="checkpoint containing concept_state_dict or a learned-search state_dict; load and freeze concept weights")
    ap.add_argument("--eval-checkpoint", default=None,
                    help="score this checkpoint on the development set (written to <out>/eval_<name>.json) and exit")
    ap.add_argument("--freeze", action="store_true", help="copy <out>/best.pt to frozen.pt + FROZEN.md and exit")
    ap.add_argument("--dev-studies", default=None, help="comma-separated study names kept for the dev set")
    ap.add_argument("--n-support-patients", type=int, default=10)
    ap.add_argument("--max-rows-per-patient", type=int, default=600)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast (off by default: the model is launch-bound, autocast doubled step time)")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        return freeze(out)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        used = _gpu_used_mib()
        if used is not None and used >= args.gpu_max_used_mib:
            raise SystemExit(f"GPU has {used} MiB in use (limit {args.gpu_max_used_mib}); not starting")
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac)

    hest = "hest:" in args.studies
    train_st, dev_st, sinfo = build_studies(args.studies, synthetic_embed_dim=1536 if hest else 64)
    if args.dev_studies:
        keep = set(args.dev_studies.split(","))
        dev_st = [s for s in dev_st if s.name in keep]
        if not dev_st:
            raise SystemExit(f"no development study among {sorted(keep)}")
    cfg = ModelConfig(selector=args.selector, encoder_kind=args.encoder, d_model=args.d_model,
                      n_heads=6 if args.d_model % 6 == 0 else 4, embed_dim=1536 if hest else 64)
    model = LearnedSearchModel(cfg).to(dev)
    if args.resume:
        model.load_state_dict(LearnedSearchModel.load(args.resume).state_dict())
    if args.frozen_concept:
        ck = torch.load(args.frozen_concept, map_location="cpu", weights_only=False)
        state = ck.get("concept_state_dict")
        if state is None:
            state = ck.get("state_dict", ck)
            state = {k.removeprefix("concept."): v for k, v in state.items() if k.startswith("concept.")}
        missing, unexpected = model.concept.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"frozen concept mismatch: missing={missing}, unexpected={unexpected}")
        for p in model.concept.parameters():
            p.requires_grad_(False)
    concept_params = [p for p in model.concept.parameters() if p.requires_grad]
    cid = {id(p) for p in concept_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in cid], "lr": args.lr}]
    if concept_params:
        groups.append({"params": concept_params, "lr": args.lr / 3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warmup = min(args.warmup, max(1, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, args.steps, warmup))

    Sampler = RowSplitSampler if args.rowsplit else EpisodeSampler
    sampler = Sampler(train_st, concepts=not args.no_concepts, seed=args.seed,
                      n_support_patients=args.n_support_patients, max_rows_per_patient=args.max_rows_per_patient)
    loader = make_loader(sampler, n_real=args.n_real, n_controlled=args.n_controlled, workers=args.workers,
                         seed=args.seed)
    t0 = time.time()
    dev_eps = fixed_dev_set(dev_st, n_controlled=args.n_dev_controlled, n_real=args.n_dev_real,
                            concepts=not args.no_concepts)
    from dbench.benchmark import generators as gen
    dev_eps += _residualised_dev(dev_st, concepts=not args.no_concepts)
    if args.eval_checkpoint:
        m = LearnedSearchModel.load(args.eval_checkpoint, map_location=dev).to(dev)
        res = evaluate(m, dev_eps)
        res.update({"checkpoint": args.eval_checkpoint, "checkpoint_step": m.meta.get("step"),
                    "studies": args.studies})
        (out / f"eval_{Path(args.eval_checkpoint).stem}.json").write_text(json.dumps(res, indent=1))
        return res
    run_info = {"args": vars(args), "config": cfg.__dict__, "studies": sinfo,
                "train_families": list(gen.TRAIN_FAMILIES), "heldout_families": list(gen.HELDOUT_FAMILIES),
                "n_dev": len(dev_eps), "dev_build_seconds": round(time.time() - t0, 1),
                "n_parameters": int(sum(p.numel() for p in model.parameters())), "sampler": Sampler.__name__}
    (out / "run_info.json").write_text(json.dumps(run_info, indent=1, default=str))

    best = -1.0
    history = []
    flog = open(out / "train_log.jsonl", "a")
    it = iter(loader)
    model.train()
    t_start = time.time()
    agg: dict = {}
    for step in range(args.steps):
        tau = 1.0 - 0.7 * min(1.0, step / max(1, args.steps))
        t_batch = time.perf_counter()
        batch = next(it)
        batch_seconds = time.perf_counter() - t_batch
        t_compute = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for ep in batch:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=args.amp and dev.type == "cuda"):
                res = model.episode_loss(ep, tau=tau)
            (res["total"] / len(batch)).backward()
            for k, v in res.items():
                if k == "total":
                    v = v.detach()
                key = f"{ep.kind}_{k}"
                agg.setdefault(key, []).append(float(v.detach()) if torch.is_tensor(v) else float(v))
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        compute_seconds = time.perf_counter() - t_compute
        agg.setdefault("batch_seconds", []).append(batch_seconds)
        agg.setdefault("compute_seconds", []).append(compute_seconds)
        agg.setdefault("grad_norm", []).append(float(gn))
        if (step + 1) % args.log_every == 0 or step == 0:
            rec = {"step": step + 1, "tau": round(tau, 3), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t_start, 1), **{k: round(float(np.mean(v)), 5) for k, v in agg.items()}}
            if dev.type == "cuda":
                rec["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            flog.write(json.dumps(rec) + "\n")
            flog.flush()
            agg = {}
        if (step + 1) % args.dev_every == 0 or step + 1 == args.steps:
            m = evaluate(model, dev_eps)
            m.update({"step": step + 1, "sec": round(time.time() - t_start, 1)})
            if dev.type == "cuda":
                m["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            history.append(m)
            with open(out / "dev_log.jsonl", "a") as fh:
                fh.write(json.dumps(m) + "\n")
            score = m["selection_score"] if m.get("selection_score") is not None else -1
            model.save(out / "last.pt", step=step + 1, dev=m)
            if score > best:
                best = score
                model.save(out / "best.pt", step=step + 1, dev=m)
            (out / "STATUS.json").write_text(json.dumps({"step": step + 1, "steps": args.steps, "best": best,
                                                         "last_dev": m}, indent=1))
    flog.close()
    summary = {"best_recovered_set@16": best, "history": history, "seconds": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _residualised_dev(dev_st, *, concepts: bool) -> list:
    """16 residualised_2 episodes at 5 % on development patients (G_HO and train-family readout)."""
    return fixed_dev_set(dev_st, n_controlled=16, n_real=0, families=("residualised_2",), fractions=(0.05,),
                         seed=777, concepts=concepts)


def freeze(run_dir, out_dir=None) -> dict:
    """Copy ``best.pt`` to ``frozen.pt`` and write ``FROZEN.md`` (config, step, dev metrics, sha256)."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir or run_dir)
    m = LearnedSearchModel.load(run_dir / "best.pt")
    m.save(out_dir / "frozen.pt", frozen_from=str(run_dir / "best.pt"))
    digest = sha256(out_dir / "frozen.pt")
    dev = m.meta.get("dev") or {}
    md = ["# Frozen learned-search checkpoint", "", f"- source: `{run_dir / 'best.pt'}`",
          f"- step: {m.meta.get('step')}", f"- sha256: `{digest}`",
          f"- selection: best development recovered_set@16 averaged over fractions >= 5 % "
          f"(score {dev.get('selection_score')})",
          f"- development fraction grid: {dev.get('fraction_grid', list(FRACTIONS))}",
          f"- training plant fractions: log-uniform on {list(TRAIN_FRACTION_RANGE)}",
          "", "## Config", "", "```json",
          json.dumps(m.cfg.__dict__, indent=1), "```", "", "## Development metrics at the frozen step", "", "```json",
          json.dumps(m.meta.get("dev"), indent=1), "```", ""]
    (out_dir / "FROZEN.md").write_text("\n".join(md))
    return {"sha256": digest, "step": m.meta.get("step")}




# ======================================================================== v02
#: Training pair labels: every pair over the top-N_PAIR_TOP singles by support CV gain + N_RANDOM_PAIRS random pairs.
N_PAIR_TOP = 40
N_RANDOM_PAIRS = 64
MORPH_DIR = "runs/development/learned_search_v02/morphology_dictionary"


def build_studies_v02(spec: str) -> tuple[list, list, dict]:
    """``synthetic`` (morph view from 16 fixed random directions) | ``hest:<partitions.json>``."""
    from .episodes import add_morph_view, studies_from_partitions, synthetic_study

    train, dev, info = [], [], {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part == "synthetic":
            C = np.random.default_rng(99).standard_normal((16, 64)).astype(np.float32)
            train += [add_morph_view(synthetic_study(s, name=f"synthetic{s}"), C) for s in range(3)]
            dev += [add_morph_view(synthetic_study(100, name="synthetic_dev"), C)]
            info["synthetic"] = {"train_seeds": [0, 1, 2], "dev_seed": 100}
        elif part.startswith("hest:"):
            t, d, i = studies_from_partitions(part.split(":", 1)[1], morph_dictionary=MORPH_DIR)
            train, dev, info["hest"] = train + t, dev + d, i
        else:
            raise ValueError(f"unknown study spec {part!r}")
    return train, dev, info


def training_pairs(S: np.ndarray, stat_names, rng: np.random.Generator) -> np.ndarray:
    cv = S[:, list(stat_names).index("cv_gain")]
    top = np.argsort(-cv, kind="stable")[:N_PAIR_TOP]
    iu = np.triu_indices(len(top), 1)
    pairs = [np.stack([top[iu[0]], top[iu[1]]], 1)]
    p = len(cv)
    r = rng.integers(p, size=(N_RANDOM_PAIRS, 2))
    pairs.append(r[r[:, 0] != r[:, 1]])
    return np.sort(np.concatenate(pairs), 1)


def episode_targets(ep, device, rng) -> tuple:
    """(statistics, stat names, labels) of one episode: everything the loss needs."""
    from . import support_statistics as ss
    return ss.episode_statistics_and_labels(ep, lambda S, sn: training_pairs(S, sn, rng), device=device)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else 0.0


class DevSetV02:
    """Fixed development episodes with their pools and realised single gains computed once."""

    def __init__(self, dev_studies, *, n_controlled: int, n_real: int, device, ks=(16,)):
        from . import support_statistics as ss
        from .episodes import fixed_dev_set
        from .search_v02 import pool_from_episode
        self.eps = fixed_dev_set(dev_studies, n_controlled=n_controlled, n_real=n_real, concepts=False,
                                 morph_target_prob=0.3, proxy_threshold=0.9)
        self.device, self.ks = device, ks
        self.pools = [pool_from_episode(ep, device=device) for ep in self.eps]
        self.single = [ss.episode_realised_gains(ep, device=device)["single"] for ep in self.eps]

    def evaluate(self, rank_by_for, *, budget: int = 64) -> dict:
        """``rank_by_for(i)`` -> the rank_by of episode i (the model, or a statistic name)."""
        from . import support_statistics as ss
        from .search_v02 import search_pool
        rows = []
        for i, (ep, pool) in enumerate(zip(self.eps, self.pools)):
            rb = rank_by_for(i)
            ranked = search_pool(pool, rb, budget=budget, device=self.device)
            top = ranked[0][0]
            g = ss.episode_realised_gains(ep, singles=False, sets=[top], device=self.device)["sets"][0]
            row = {"kind": ep.kind, "family": ep.family, "fraction": ep.fraction, "top_gain": float(g),
                   "top_k": len(top), "target_morph": ep.target_name.startswith("morph:")}
            if hasattr(rb, "_pool_cache") and rb._pool_cache is not None:
                row["pred_vs_realised"] = _spearman(rb._pool_cache[1], self.single[i])
            if ep.kind == "controlled":
                truth = tuple(sorted(ep.true_idx))
                for k in self.ks:
                    row[f"hit@{k}"] = any(tuple(sorted(s)) == truth for s, _, _ in ranked[:k])
            rows.append(row)

        def mean(xs):
            xs = [float(x) for x in xs if x is not None]
            return float(np.mean(xs)) if xs else None

        ctrl = [r for r in rows if r["kind"] == "controlled"]
        real = [r for r in rows if r["kind"] == "real"]
        out = {"primary_top_general_gain": mean([r["top_gain"] for r in rows]),
               "real_top_general_gain": mean([r["top_gain"] for r in real]),
               "real_morph_target_top_gain": mean([r["top_gain"] for r in real if r["target_morph"]]),
               "real_molecular_target_top_gain": mean([r["top_gain"] for r in real if not r["target_morph"]]),
               "controlled_top_general_gain": mean([r["top_gain"] for r in ctrl]),
               "pred_vs_realised_spearman": mean([r.get("pred_vs_realised") for r in rows]),
               "mean_top_k": mean([r["top_k"] for r in rows])}
        for k in self.ks:
            out[f"recovered_set@{k}"] = mean([r[f"hit@{k}"] for r in ctrl])
            out[f"recovered_set@{k}_by_fraction"] = {f"{f:g}": mean([r[f"hit@{k}"] for r in ctrl if r["fraction"] == f])
                                                    for f in sorted({r["fraction"] for r in ctrl})}
            out[f"recovered_set@{k}_by_family"] = {f: mean([r[f"hit@{k}"] for r in ctrl if r["family"] == f])
                                                  for f in sorted({r["family"] for r in ctrl})}
        out["n"] = {"controlled": len(ctrl), "real": len(real)}
        return out


def main_v02(argv=None) -> dict:
    from .model import GainConfig, GainSearchModel
    from .episodes import EpisodeSampler, RowSplitSampler, make_loader

    ap = argparse.ArgumentParser(description="learned search v02 training")
    ap.add_argument("--v02", action="store_true")
    ap.add_argument("--studies", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rowsplit", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-frac", type=float, default=0.19)   # ~7.6 GiB of the 40 GiB card
    ap.add_argument("--dev-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--n-dev-controlled", type=int, default=64)
    ap.add_argument("--n-dev-real", type=int, default=64)
    ap.add_argument("--n-real", type=int, default=2)
    ap.add_argument("--n-controlled", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-support-patients", type=int, default=10)
    ap.add_argument("--max-rows-per-patient", type=int, default=600)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        return freeze_v02(out)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac)
    t0 = time.time()
    train_st, dev_st, sinfo = build_studies_v02(args.studies)
    dev_st = [s for s in dev_st if s.name in ("kidney_injury", "synthetic_dev")] or dev_st
    Sampler = RowSplitSampler if args.rowsplit else EpisodeSampler
    sampler = Sampler(train_st, concepts=False, morph_target_prob=0.3, proxy_threshold=0.9, seed=args.seed,
                      n_support_patients=args.n_support_patients, max_rows_per_patient=args.max_rows_per_patient)
    loader = make_loader(sampler, n_real=args.n_real, n_controlled=args.n_controlled, workers=args.workers,
                         seed=args.seed)
    devset = DevSetV02(dev_st, n_controlled=args.n_dev_controlled, n_real=args.n_dev_real, device=dev)
    t_dev = time.time() - t0
    baseline = devset.evaluate(lambda i: "cv_gain", budget=args.budget)   # support-fit screen, same beam + refit
    cfg = GainConfig(d_model=args.d_model, n_heads=6 if args.d_model % 6 == 0 else 4)
    model = GainSearchModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = min(args.warmup, max(1, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, args.steps, warmup))
    run_info = {"args": vars(args), "config": cfg.__dict__, "studies": sinfo, "sampler": Sampler.__name__,
                "train_studies": [s.name for s in train_st], "dev_studies": [s.name for s in dev_st],
                "n_parameters": int(sum(p.numel() for p in model.parameters())),
                "dev_build_seconds": round(t_dev, 1), "support_fit_screen_dev": baseline,
                "pair_labels": f"all pairs over top-{N_PAIR_TOP} singles by support cv_gain + {N_RANDOM_PAIRS} random",
                "freeze_rule": "best primary_top_general_gain (mean held-out general gain of the top-ranked "
                               "relation over the fixed development episodes)"}
    (out / "run_info.json").write_text(json.dumps(run_info, indent=1, default=str))
    rng = np.random.default_rng(args.seed + 1)
    gen = torch.Generator().manual_seed(args.seed)
    best, history, agg = -np.inf, [], {}
    flog = open(out / "train_log.jsonl", "a")
    it = iter(loader)
    t_start = time.time()
    for step in range(args.steps):
        tb = time.perf_counter()
        batch = next(it)
        t_wait = time.perf_counter() - tb
        opt.zero_grad(set_to_none=True)
        for ep in batch:
            tl = time.perf_counter()
            S, sn, lab = episode_targets(ep, dev, rng)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            agg.setdefault("label_seconds", []).append(time.perf_counter() - tl)
            res = model.episode_loss(ep, S, sn, lab, generator=gen)
            (res["total"] / len(batch)).backward()
            for k, v in res.items():
                agg.setdefault(f"{ep.kind}_{k}", []).append(float(v.detach()))
            agg.setdefault("n_candidates", []).append(S.shape[0])
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        agg.setdefault("wait_seconds", []).append(t_wait)
        agg.setdefault("step_seconds", []).append(time.perf_counter() - tb)
        agg.setdefault("grad_norm", []).append(float(gn))
        if (step + 1) % args.log_every == 0 or step == 0:
            rec = {"step": step + 1, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t_start, 1),
                   **{k: round(float(np.mean(v)), 5) for k, v in agg.items()}}
            if dev.type == "cuda":
                rec["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            flog.write(json.dumps(rec) + "\n")
            flog.flush()
            agg = {}
        if (step + 1) % args.dev_every == 0 or step + 1 == args.steps:
            model.eval()
            te = time.time()
            m = devset.evaluate(lambda i: model, budget=args.budget)
            model.train()
            m.update({"step": step + 1, "sec": round(time.time() - t_start, 1),
                      "eval_seconds_per_episode": round((time.time() - te) / max(1, len(devset.eps)), 3)})
            history.append(m)
            with open(out / "dev_log.jsonl", "a") as fh:
                fh.write(json.dumps(m) + "\n")
            model.save(out / "last.pt", step=step + 1, dev=m)
            if m["primary_top_general_gain"] > best:
                best = m["primary_top_general_gain"]
                model.save(out / "best.pt", step=step + 1, dev=m)
            (out / "STATUS.json").write_text(json.dumps({"step": step + 1, "steps": args.steps, "best_primary": best,
                                                         "last_dev": m, "support_fit_screen_dev": baseline},
                                                        indent=1))
    flog.close()
    summary = {"best_primary": best, "history": history, "support_fit_screen_dev": baseline,
               "seconds": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def freeze_v02(run_dir) -> dict:
    from .model import GainSearchModel
    run_dir = Path(run_dir)
    m = GainSearchModel.load(run_dir / "best.pt")
    m.save(run_dir / "frozen.pt", frozen_from=str(run_dir / "best.pt"))
    digest = sha256(run_dir / "frozen.pt")
    info = json.loads((run_dir / "run_info.json").read_text()) if (run_dir / "run_info.json").exists() else {}
    md = ["# Frozen learned-search v02 checkpoint", "", f"- source: `{run_dir / 'best.pt'}`",
          f"- step: {m.meta.get('step')}", f"- sha256: `{digest}`",
          "- rule: best development primary = mean held-out general gain of the top-ranked relation "
          "(64 real kidney_injury tasks + 64 controlled held-out-family episodes), evaluated every "
          f"{(info.get('args') or {}).get('dev_every', 1000)} steps",
          f"- grid: steps {(info.get('args') or {}).get('steps')}, budget B = {(info.get('args') or {}).get('budget', 64)}, "
          f"sampler {info.get('sampler')}",
          "", "## Config", "", "```json", json.dumps(m.cfg.__dict__, indent=1), "```", "",
          "## Development metrics at the frozen step", "", "```json", json.dumps(m.meta.get("dev"), indent=1), "```", "",
          "## Support-fit screen (same beam + refit) on the same development episodes", "", "```json",
          json.dumps(info.get("support_fit_screen_dev"), indent=1), "```", ""]
    (run_dir / "FROZEN.md").write_text("\n".join(md))
    return {"sha256": digest, "step": m.meta.get("step")}


if __name__ == "__main__":
    import sys
    main_v02() if "--v02" in sys.argv else main()
