"""Question 2 ablations: does learning the search strategy across tasks matter?

* row-split episodic training: the same architecture and losses, but support and
  query rows come from the SAME patients (:class:`RowSplitSampler`; run with
  ``python -m morpheus_encoder.learned_search.train ... --rowsplit``);
* per-task training (:func:`train_per_task`): the same architecture trained from
  scratch on one task only (episodes are patient-disjoint resamplings of that
  task's own rows, relation loss only, no planted episodes), then asked for
  proposals exactly like the meta-trained model.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .episodes import EpisodeSampler, RowSplitSampler, study_from_arrays  # noqa: F401  (re-exported)
from .model import LearnedSearchModel, ModelConfig


def train_per_task(X, y, groups, neighbours6=None, *, names, views, embed=None, steps: int = 600,
                   cfg: ModelConfig | None = None, device=None, seed: int = 0, batch: int = 8, lr: float = 3e-4,
                   n_paths: int = 32) -> tuple[LearnedSearchModel, list, dict]:
    """Train a fresh model on one task's rows, then propose on those rows.

    Returns ``(model, propose_sets(...) output, info)``."""
    t0 = time.time()
    torch.manual_seed(seed)
    groups = np.asarray(groups)
    n_pat = len(np.unique(groups))
    if n_pat < 2:
        raise ValueError("per-task training needs at least two patients for support/query episodes")
    edim = 1536 if embed is None else int(np.asarray(embed).shape[1])
    cfg = cfg or ModelConfig(embed_dim=edim)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # This arm must use the same practical accelerator settings as meta-training.
    # TF32 preserves the FP32 objective while enabling A100 tensor cores.
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model = LearnedSearchModel(cfg).to(dev)
    study = study_from_arrays(X, groups, neighbours6, names, views, embed=embed, y=y)
    n_q = max(1, n_pat // 3)
    sampler = EpisodeSampler([study], controlled_prob=0.0, cross_study_prob=0.0, concept_target_prob=0.0,
                             n_support_patients=n_pat - n_q, n_query_patients=n_q, seed=seed)
    concept_params = list(model.concept.parameters())
    cid = {id(p) for p in concept_params}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in cid], "lr": lr},
                             {"params": concept_params, "lr": lr / 3}], weight_decay=0.01)
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for step in range(steps):
        tau = 1.0 - 0.7 * step / max(1, steps)
        opt.zero_grad(set_to_none=True)
        # Keep the exact same gradient objective as before.  In particular, do
        # not convert each CUDA loss to a Python float inside the inner loop:
        # that forces one device/host synchronization per episode (4,800 per
        # world at the registered 600 x 8 budget).  Logging is intentionally
        # detached and synchronized once per optimizer step only.
        logged_total = torch.zeros((), device=dev)
        for _ in range(batch):
            ep = sampler.sample(rng, kind="real")
            res = model.episode_loss(ep, tau=tau)
            (res["total"] / batch).backward()
            logged_total.add_(res["total"].detach() / batch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(logged_total.cpu()))
    model.eval()
    sets = model.propose_sets(X, y, groups, neighbours6, names=names, views=views, embed=embed, n_paths=n_paths,
                              seed=seed, device=dev)
    info = {"steps": steps, "seconds": round(time.time() - t0, 1), "loss_first50": float(np.mean(losses[:50])),
            "loss_last50": float(np.mean(losses[-50:])), "n_patients": n_pat}
    return model, sets, info
