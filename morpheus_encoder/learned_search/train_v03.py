"""Training of the v03 gain predictor (spec 1.4, 1.5, 1.7, 1.8).

Per labelled episode (``label_episode``): B uniform from {nuis, known}; every single is fitted
(support LOPO gain + query gain over three query draws), the transfer diagnostics come from the
same fits; labelled singles = 48 uniform (regression and ranking) + the network's current top 16
(ranking only); labelled pairs = 48 uniform (regression and ranking) + the network's top 16 +
16 drawn with probability proportional to the positive support gains of their members (ranking
only); every labelled pair also gets its B_decomp label (higher-order gain), used by a second
forward pass under the ``decomp`` token on episodes whose B is ``nuis`` (the token's diagnostics are
then the nuisance ones).  30 % of episodes are controlled worlds from the training families only.
Planted sets never enter the loss.

Loop: CPU loader workers draw raw episodes; the main process labels ``--new-per-step`` fresh
episodes per step on the GPU (on-policy labels from the current network) into a FIFO buffer and
trains on ``--episodes-per-step`` episodes from it (the newest included).

    python -m morpheus_encoder.learned_search.train_v03 --studies synthetic --steps 300 \
        --out runs/development/learned_search_v03/synthetic_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import relation_v03 as rv
from .episodes_v03 import SamplerV03, task_from_episode
from .model_v03 import SCALE, ConfigV03, GainModelV03, episode_loss

OUT_ROOT = "runs/development/learned_search_v03"
PARTITIONS = "dbench/protocols/sprint_partitions_v01.json"
N_UNIFORM_SINGLES, N_UNIFORM_PAIRS, N_ONPOLICY, N_FAILURE = 48, 48, 16, 16


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- studies
def synthetic_studies():
    """Three training studies + one development study with the v03 fields (tests and smoke)."""
    from .episodes import add_morph_view, synthetic_study
    from .episodes_v03 import StudyV03
    C = np.random.default_rng(99).standard_normal((16, 64)).astype(np.float32)
    out = []
    for seed, name, organ, plat in ((0, "syn0", "a", "visium"), (1, "syn1", "a", "visium"), (2, "syn2", "a", "st_100um"),
                                    (100, "syn_dev", "b", "visium")):
        st = add_morph_view(synthetic_study(seed, name=name), C)
        pats = st.patients
        inv = np.searchsorted(pats, st.groups)
        prog = np.asarray(st.views["programme"])
        known = st.base[:, prog]
        comp = known[:, :3]
        mc = list(st.views["morph"])
        pm = np.zeros((len(pats), st.base.shape[1]), dtype=np.float32)
        out.append(StudyV03(name=name, base=st.base, names=st.names, views=st.views, groups=st.groups,
                            neighbours6=st.neighbours6, coords=st.coords, platform=plat,
                            target_cols=list(st.target_cols), exclusions=st.exclusions, residual_cols=list(prog),
                            depth=(st.depth - np.array([st.depth[st.groups == p].mean() for p in pats])[inv]
                                   ).astype(np.float32),
                            known=known, composition=comp, sections=st.groups.copy(), patient_means=pm,
                            organ=organ, morph_targets=mc, comp_source="synthetic programmes"))
    return out[:3], out[3:]


def load_studies(spec: str):
    if spec == "synthetic":
        tr, dv = synthetic_studies()
        return tr, dv, {"synthetic": True}
    from .episodes_v03 import studies_v03
    st, info = studies_v03(spec.split(":", 1)[1] if spec.startswith("hest:") else PARTITIONS)
    return st["meta_training"], st["development"], info


# ------------------------------------------------------------------ labels
def _model_scores(model, task, B, L, diag, base):
    with torch.no_grad():
        H = model.encode(task.X[:task.s], base.r_s, task.groups[:task.s], task.sections[:task.s]
                         if task.sections is not None else None, _nb_support(task), diag, B, L)
        mu, _ = model.single_head(H)
    return H, mu.cpu().numpy() / SCALE


def _nb_support(task):
    if task.nb6 is None:
        return None
    nb = np.asarray(task.nb6)[:task.s]
    return np.where(nb < task.s, nb, -1)


def label_episode(ep, B: str, *, model=None, rng: np.random.Generator, device=None) -> dict:
    task = task_from_episode(ep)
    s, p = task.s, task.X.shape[1]
    L = ep.info.get("level", "L0")
    base = rv.baseline_for(task, B)
    single = rv.fit_sets(task.X[:s], base, task.y[:s], task.groups[:s], Vq=task.X[s:], yq=task.y[s:],
                         groups_q=task.groups[s:], device=device, draw_seed=int(rng.integers(1 << 30)))
    diag, _ = rv.diagnostics(task, B, base=base, single=single, device=device)
    gs = single["g_sup"]
    uni1 = rng.choice(p, min(N_UNIFORM_SINGLES, p), replace=False)
    on1 = np.zeros(0, dtype=int)
    on2 = np.zeros((0, 2), dtype=int)
    if model is not None:
        model.eval()
        H, mu = _model_scores(model, task, B, L, diag, base)
        on1 = np.argsort(-(gs - mu), kind="stable")[:N_ONPOLICY]
        with torch.no_grad():
            PM = model.pair_matrix(H).cpu().numpy() / SCALE
        sc = PM[..., 0] - PM[..., 1]
        iu = np.triu_indices(p, 1)
        o = np.argsort(-sc[iu], kind="stable")[:N_ONPOLICY]
        on2 = np.stack([iu[0][o], iu[1][o]], 1)
        model.train()
    single_idx = np.concatenate([uni1, np.setdiff1d(on1, uni1)])
    single_uniform = np.zeros(len(single_idx), dtype=bool)
    single_uniform[:len(uni1)] = True
    u2 = rng.integers(p, size=(4 * N_UNIFORM_PAIRS, 2))
    u2 = np.unique(np.sort(u2[u2[:, 0] != u2[:, 1]], 1), axis=0)
    u2 = u2[rng.permutation(len(u2))[:N_UNIFORM_PAIRS]]
    w = np.clip(gs, 0, None) + 1e-6
    f2 = np.stack([rng.choice(p, N_FAILURE, p=w / w.sum()), rng.choice(p, N_FAILURE, p=w / w.sum())], 1)
    f2 = np.sort(f2[f2[:, 0] != f2[:, 1]], 1)
    pairs = np.concatenate([u2, on2, f2]).astype(int)
    pair_uniform = np.zeros(len(pairs), dtype=bool)
    pair_uniform[:len(u2)] = True
    pg = rv.gains(task, pairs, B, base=base, device=device, draw_seed=int(rng.integers(1 << 30)))
    base_n = base if B == "nuis" else rv.baseline_for(task, "nuis")
    dg = rv.decomp_gains(task, pairs, base=base_n, full=pg if B == "nuis" else None, device=device)
    out = {"X": task.X.astype(np.float32), "n_support": s, "groups": task.groups, "sections":
           np.asarray(task.sections) if task.sections is not None else task.groups, "nb6_s": _nb_support(task),
           "y_resid": base.r_s, "diag": diag, "B": B, "L": L, "kind": ep.kind, "family": ep.family,
           "single_idx": single_idx, "single_uniform": single_uniform, "single_gsup": gs[single_idx],
           "single_g": single["g"][single_idx], "all_gsup": gs, "all_g": single["g"],
           "pairs": pairs, "pair_uniform": pair_uniform, "pair_gsup": pg["g_sup"], "pair_g": pg["g"],
           "decomp_gsup": dg["g_sup"], "decomp_g": dg["g"], "decomp_g_all": dg["g"], "target": ep.target_name,
           "study": ep.study}
    if B == "nuis":   # the decomp pass shares the nuisance diagnostics
        out.update({"y_resid_nuis": base.r_s, "diag_nuis": diag})
    else:
        out["decomp_g"] = None
    return out


# ------------------------------------------------------------- development
def _boot_lower(gain_qp: np.ndarray, sst_qp: np.ndarray, rng, n: int = 200) -> np.ndarray:
    P = len(sst_qp)
    idx = rng.integers(P, size=(n, P))
    G = gain_qp[idx].sum(1) / sst_qp[idx].sum(1)[:, None]
    return np.quantile(G, 0.025, axis=0)


class DevTasks:
    """Fixed development tasks (kidney_injury): 40 real targets, B_nuis, L0.  Provisional metric
    until the harness callable lands: mean over tasks of the number of top-10 sets whose pooled
    query gain has a patient-bootstrap lower bound > 0 (yield@10, no null subtraction), the
    realised gain of the top set, and the calibration Spearman (predicted vs realised gain)."""

    def __init__(self, dev_studies, *, n_tasks: int = 40, seed: int = 4321, device=None):
        self.device = device
        samp = SamplerV03(dev_studies, seed=seed, controlled_prob=0.0)
        samp.force_level = "L0"
        rng = np.random.default_rng(seed)
        self.tasks = []
        for i in range(n_tasks):
            ep = samp.sample(rng, kind="real")
            self.tasks.append((ep.target_name, task_from_episode(ep)))

    def evaluate(self, model, *, K: int = 64) -> dict:
        """Also: yield@10 of the same fitted sets ranked by G_sup alone vs by G_sup - mu - kappa sigma
        (the contribution of the predicted shrinkage term)."""
        from scipy.stats import spearmanr
        from .search_v03 import search
        rng = np.random.default_rng(0)
        y10, top, pred_all, real_all, y10_sup, y10_lcb = [], [], [], [], [], []
        for _, task in self.tasks:
            ranked, det = search(task, "nuis", K, model=model, device=self.device, return_details=True)
            fitted = det["fitted"]
            sets = [f[0] for f in fitted]
            pos = {S: i for i, S in enumerate(sets)}
            base = rv.baseline_for(task, "nuis")
            G = np.zeros(len(sets))
            lower = np.zeros(len(sets))
            by_k: dict = {}
            for i, S in enumerate(sets):
                by_k.setdefault(len(S), []).append(i)
            for k, ids in by_k.items():
                r = rv.gains(task, np.asarray([sets[i] for i in ids]), "nuis", base=base, device=self.device)
                G[ids] = r["g_full"]
                lower[ids] = _boot_lower(r["gain_qp"], r["sst_qp"], rng)
            ok = lower > 0
            order_lcb = [pos[r[0]] for r in ranked]
            order_sup = list(np.argsort([-f[1] for f in fitted], kind="stable"))
            y10.append(int(ok[order_lcb[:10]].sum()))
            y10_lcb.append(int(ok[np.argsort([-f[2] for f in fitted], kind="stable")[:10]].sum()))
            y10_sup.append(int(ok[order_sup[:10]].sum()))
            top.append(float(G[order_lcb[0]]))
            pred_all += [r[1] for r in ranked]
            real_all += G[order_lcb].tolist()
        rho = spearmanr(pred_all, real_all).correlation
        known = self.known_readout(model, K=K) if model is not None else {}
        return {**known, "yield10_Bnuis_no_null": float(np.mean(y10)), "top_gain": float(np.mean(top)),
                "yield10_rank_by_gsup_same_sets": float(np.mean(y10_sup)),
                "yield10_rank_by_lcb_same_sets": float(np.mean(y10_lcb)),
                "shrinkage_term_contribution_yield10": float(np.mean(y10_lcb) - np.mean(y10_sup)),
                "calibration_spearman": float(rho) if np.isfinite(rho) else 0.0, "n_tasks": len(self.tasks)}


def _known_readout(self, model, *, K: int = 64) -> dict:
    """Calibration of the B_known prediction on the development tasks: the sets the search fits under
    B = known, predicted gain (G_sup - mu_delta) vs realised held-out gain over B_known (target and
    overlapping programmes excluded, relation_v03.known_covariates); Spearman and mean error."""
    from scipy.stats import spearmanr
    from .search_v03 import search
    pred, real = [], []
    for _, task in self.tasks:
        if task.known is None:
            continue
        ranked = search(task, "known", K, model=model, device=self.device)
        base = rv.baseline_for(task, "known")
        by_k: dict = {}
        for i, r in enumerate(ranked):
            by_k.setdefault(len(r[0]), []).append(i)
        G = np.zeros(len(ranked))
        for k, ids in by_k.items():
            G[ids] = rv.gains(task, np.asarray([ranked[i][0] for i in ids]), "known", base=base,
                              device=self.device)["g_full"]
        pred += [r[1] for r in ranked]
        real += G.tolist()
    if len(pred) < 3:
        return {}
    pred, real = np.asarray(pred), np.asarray(real)
    rho = spearmanr(pred, real).correlation
    return {"known_calibration_spearman": float(rho) if np.isfinite(rho) else 0.0,
            "known_mean_error": float(np.mean(pred - real)), "known_mean_abs_error": float(np.mean(np.abs(pred - real))),
            "known_n_sets": int(len(pred))}


DevTasks.known_readout = _known_readout


def _harness_metric():
    try:
        from dbench.harness.dev_metric import dev_metric   # Agent N's callable (spec 3.4), when present
        return dev_metric
    except Exception:
        return None


# --------------------------------------------------------------------- main
def _schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main(argv=None) -> dict:
    from .episodes import make_loader
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", required=True, help="synthetic | hest[:partitions.json]")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rowsplit", action="store_true")
    ap.add_argument("--episodes-per-step", type=int, default=8)
    ap.add_argument("--new-per-step", type=int, default=1)
    ap.add_argument("--buffer", type=int, default=64)
    ap.add_argument("--early-steps", type=int, default=4000, help="L0 only before this step ('epochs 1-4')")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dev-every", type=int, default=1000)
    ap.add_argument("--n-dev-tasks", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-frac", type=float, default=0.2)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="continue training from this checkpoint's weights (fresh optimiser); step 0 is evaluated "
                         "and reported but is not a selection candidate")
    ap.add_argument("--label-fix-commit", default=None, help="recorded in run_info / FROZEN.md")
    ap.add_argument("--no-harness", action="store_true", help="skip the development harness (smokes)")
    ap.add_argument("--holdouts", default=f"{OUT_ROOT}/transfer_holdouts.json",
                    help="json with withheld_studies (spec 0.2 transfer holdouts), excluded from training")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        return freeze(out)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac)
    t0 = time.time()
    train_st, dev_st, sinfo = load_studies(args.studies)
    holdouts = {}
    if args.holdouts and Path(args.holdouts).exists():
        holdouts = json.loads(Path(args.holdouts).read_text())
        drop = set(holdouts.get("withheld_studies", []))
        train_st = [s for s in train_st if s.name not in drop]
    kw = dict(seed=args.seed, controlled_prob=0.3, early_steps=args.early_steps, n_workers=max(1, args.workers),
              episodes_per_step=args.new_per_step)
    if args.rowsplit:
        from .episodes_v03 import RowSplitV03
        sampler = RowSplitV03(train_st, **kw)
    else:
        sampler = SamplerV03(train_st, **kw)
    loader = make_loader(sampler, n_real=args.new_per_step, n_controlled=0, workers=args.workers, seed=args.seed)
    devtasks = DevTasks(dev_st, n_tasks=args.n_dev_tasks, device=dev)
    load_s = time.time() - t0
    screen = devtasks.evaluate(None)   # baseline 11 (support-fit screen) on the same tasks
    model = GainModelV03(ConfigV03()).to(dev)
    init = None
    if args.init_from:
        base_m = GainModelV03.load(args.init_from)
        model.load_state_dict(base_m.state_dict())
        init = {"checkpoint": args.init_from, "sha256": _sha256(args.init_from), "base_step": base_m.meta.get("step"),
                "label_fix_commit": args.label_fix_commit}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, args.steps, min(args.warmup, args.steps // 10 or 1)))
    harness = None if args.no_harness else _harness_metric()
    info = {"args": vars(args), "config": model.cfg.__dict__, "studies": sinfo, "load_seconds": round(load_s, 1),
            "sampler": type(sampler).__name__, "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "train_studies": [s.name for s in train_st], "dev_studies": [s.name for s in dev_st],
            "support_screen_dev": screen, "harness_metric": bool(harness), "transfer_holdouts": holdouts,
            "init_from": init,
            "label_source": ("fresh in-memory labels under the loaded code (label cache written, never read)"
                             + (f"; label fix commit {args.label_fix_commit}" if args.label_fix_commit else "")),
            "reuse_factor": args.episodes_per_step / max(1, args.new_per_step), "eval_cadence_steps": args.dev_every,
            "reference_exact_support_gain_only": {"step150_synthetic_smoke_net_yield_k10_Bnuis": 7.11,
                                                  "calibration_spearman": 0.94},
            "label_sampling": "48 uniform singles + 48 uniform pairs (regression + ranking); network top-16 singles, "
                              "top-16 pairs, 16 support-gain-proportional pairs (ranking only); 3 query draws; "
                              "30 % controlled (training families)",
            "selection": "best development yield (harness when available, else provisional yield10 without null) "
                         "every dev-every steps; calibration Spearman logged"}
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    rng = np.random.default_rng(args.seed + 7)
    gen = torch.Generator().manual_seed(args.seed)
    buf: deque = deque(maxlen=args.buffer)
    it = iter(loader)
    best, agg = -np.inf, {}
    flog = open(out / "train_log.jsonl", "a")
    t_start = time.time()
    if init is not None:   # step 0 of the repair: the base checkpoint, reported, never selected
        model.eval()
        m0 = devtasks.evaluate(model)
        if harness is not None:
            try:
                m0["harness"] = harness(lambda task, B, K: _search_fn(model, task, B, K, dev), 0, device=dev,
                                        out_dir=str(out / "harness"))
            except Exception as exc:
                m0["harness_error"] = f"{type(exc).__name__}: {exc}"[:300]
        m0.update({"step": 0, "selection_candidate": False})
        with open(out / "dev_log.jsonl", "a") as fh:
            fh.write(json.dumps(m0) + "\n")
    model.train()
    for step in range(args.steps):
        tb = time.perf_counter()
        raw = next(it)
        tw = time.perf_counter() - tb
        tl = time.perf_counter()
        fresh = []
        for ep in raw:
            B = "nuis" if rng.random() < 0.5 else "known"
            try:
                fresh.append(label_episode(ep, B, model=model if step > 0 else None, rng=rng, device=dev))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                agg.setdefault("label_failures", []).append(1.0)
        buf.extend(fresh)
        for lep, ep in zip(fresh, raw):   # the shared cache holds HEST episodes only (no synthetic smokes)
            if args.studies != "synthetic":
                _cache_labels(lep, ep, out_root=Path(OUT_ROOT) / "label_cache")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        label_s = time.perf_counter() - tl
        if not buf:
            continue
        pick = list(fresh) + [buf[i] for i in rng.choice(len(buf), max(0, args.episodes_per_step - len(fresh)))]
        opt.zero_grad(set_to_none=True)
        for lep in pick:
            res = episode_loss(model, lep, lam=args.lam, generator=gen)
            (res["total"] / len(pick)).backward()
            for k, v in res.items():
                agg.setdefault(k, []).append(float(v.detach()))
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        agg.setdefault("wait_seconds", []).append(tw)
        agg.setdefault("label_seconds", []).append(label_s)
        agg.setdefault("step_seconds", []).append(time.perf_counter() - tb)
        agg.setdefault("grad_norm", []).append(float(gn))
        agg.setdefault("n_candidates", []).append(float(np.mean([e["X"].shape[1] for e in fresh])) if fresh else 0.0)
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
            m = devtasks.evaluate(model)
            if harness is not None:
                try:
                    m["harness"] = harness(lambda task, B, K: _search_fn(model, task, B, K, dev), step + 1,
                                           device=dev, out_dir=str(out / "harness"))
                except Exception as exc:   # report, keep the provisional metric
                    m["harness_error"] = f"{type(exc).__name__}: {exc}"[:300]
            model.train()
            primary = (m.get("harness") or {}).get("net_yield_k10_Bnuis", m["yield10_Bnuis_no_null"])
            m.update({"step": step + 1, "primary": primary, "eval_seconds": round(time.time() - te, 1)})
            with open(out / "dev_log.jsonl", "a") as fh:
                fh.write(json.dumps(m) + "\n")
            model.save(out / "last.pt", step=step + 1, dev=m)
            if primary > best:
                best = primary
                model.save(out / "best.pt", step=step + 1, dev=m)
            (out / "STATUS.json").write_text(json.dumps({"step": step + 1, "steps": args.steps, "best_primary": best,
                                                         "last_dev": m, "support_screen_dev": screen}, indent=1))
    flog.close()
    return {"best_primary": best}


def _cache_labels(lep: dict, ep, *, out_root: Path) -> str:
    """Label cache entry for one labelled episode, keyed by ``relation_v03.label_key`` of (study,
    support patients, query patients, target, B, L): diagnostics of every candidate, all singles'
    G_sup / G (three query draws) / Delta, labelled pairs with G_sup / G / Delta and B_decomp labels."""
    key = rv.label_key(ep.study + ("->" + ep.query_study if ep.query_study != ep.study else ""),
                       ep.support_patients, ep.query_patients, ep.target_name, lep["B"], lep["L"])
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{key}.npz"
    if not path.exists():
        np.savez_compressed(path, diag=lep["diag"], diag_names=np.asarray(rv.DIAG_NAMES), single_gsup=lep["all_gsup"],
                            single_g=lep["all_g"], single_delta=lep["all_gsup"] - lep["all_g"],
                            single_idx=lep["single_idx"], single_uniform=lep["single_uniform"], pairs=lep["pairs"],
                            pair_uniform=lep["pair_uniform"], pair_gsup=lep["pair_gsup"], pair_g=lep["pair_g"],
                            pair_delta=lep["pair_gsup"] - lep["pair_g"], decomp_gsup=lep["decomp_gsup"],
                            decomp_g=lep["decomp_g_all"], B=lep["B"], L=lep["L"], kind=ep.kind,
                            family=str(ep.family), study=ep.study, query_study=ep.query_study, target=ep.target_name,
                            support_patients=np.asarray(ep.support_patients), query_patients=np.asarray(ep.query_patients),
                            candidate_names=np.asarray(ep.meta.names))
    return key


def _search_fn(model, task, B, K, dev):
    from .search_v03 import search
    return search(task, B, K, model=model, device=dev)


def freeze(run_dir) -> dict:
    run_dir = Path(run_dir)
    m = GainModelV03.load(run_dir / "best.pt")
    m.save(run_dir / "frozen.pt", frozen_from=str(run_dir / "best.pt"))
    digest = _sha256(run_dir / "frozen.pt")
    info = json.loads((run_dir / "run_info.json").read_text()) if (run_dir / "run_info.json").exists() else {}
    repair = []
    if info.get("init_from"):
        dl = [json.loads(x) for x in open(run_dir / "dev_log.jsonl")] if (run_dir / "dev_log.jsonl").exists() else []
        before = next((d for d in dl if d.get("step") == 0), {})
        after = (m.meta.get("dev") or {})
        keys = ("known_calibration_spearman", "known_mean_error", "known_mean_abs_error", "known_n_sets")
        ib = info["init_from"]
        repair = ["", "## B_known repair (continued training under the fixed B_known definition)", "",
                  f"- base checkpoint: `{ib.get('checkpoint')}` (step {ib.get('base_step')}, sha256 `{ib.get('sha256')}`)",
                  f"- repair steps: {(info.get('args') or {}).get('steps')} (selected repair step {m.meta.get('step')})",
                  f"- label fix commit: {ib.get('label_fix_commit')}",
                  f"- B_known calibration before (repair step 0): {json.dumps({k: before.get(k) for k in keys})}",
                  f"- B_known calibration after (selected step): {json.dumps({k: after.get(k) for k in keys})}",
                  f"- B_nuis development net yield@10 before: {(before.get('harness') or {}).get('net_yield_k10_Bnuis')}, "
                  f"after: {(after.get('harness') or {}).get('net_yield_k10_Bnuis')}"]
    md = ["# Frozen learned-search v03 gain predictor", "", f"- source: `{run_dir / 'best.pt'}`",
          f"- step: {m.meta.get('step')}", f"- sha256: `{digest}`",
          f"- rule: {info.get('selection')}", f"- grid: {json.dumps(info.get('args'))}", *repair, "",
          "## Development metrics at the frozen step", "", "```json", json.dumps(m.meta.get("dev"), indent=1), "```",
          "", "## Support-fit screen on the same development tasks", "", "```json",
          json.dumps(info.get("support_screen_dev"), indent=1), "```", ""]
    (run_dir / "FROZEN.md").write_text("\n".join(md))
    return {"sha256": digest, "step": m.meta.get("step")}


if __name__ == "__main__":
    main()
