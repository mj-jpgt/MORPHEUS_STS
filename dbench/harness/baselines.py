"""Harness-side search functions ``search(task, B, K) -> [(index tuple, predicted gain, predicted sd), ...]``.

Every learned and statistic-based method runs the SAME beam and closed-form refit
(``learned_search.search_v03.search``, spec 1.6) at the same budget K; only the scorer differs.

* ``screen_search``: baseline 11, the support-fit screen (``model=None``: zero predicted shrinkage).
* ``model_search``: the frozen g_theta (or an ablation checkpoint).
"""
from __future__ import annotations


def screen_search(device=None, kappa: float = 1.0):
    from morpheus_encoder.learned_search.search_v03 import search as s3

    def search(task, B, K):
        return s3(task, B, K, model=None, kappa=kappa, device=device)
    search.parallel_ok = device is None or str(device) == "cpu"
    search.gpu = str(device).startswith("cuda")
    return search


def model_search(model, device=None, kappa: float = 1.0, L: str = "L0"):
    """``model``: a loaded GainModelV03, or a checkpoint path loaded lazily inside each (forked)
    worker, so the parent never initialises CUDA before forking GPU search workers."""
    from morpheus_encoder.learned_search.search_v03 import search as s3
    cache = {}

    def search(task, B, K):
        m = model
        if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
            if "m" not in cache:
                from morpheus_encoder.learned_search.model_v03 import GainModelV03
                cache["m"] = GainModelV03.load(model, map_location=device or "cpu").to(device or "cpu")
            m = cache["m"]
        return s3(task, B, K, model=m, L=L, kappa=kappa, device=device)
    search.parallel_ok = device is None or str(device) == "cpu"
    search.gpu = str(device).startswith("cuda") and not hasattr(model, "state_dict")
    return search


def EXTRA_METHODS(device=None) -> dict:
    """Name -> factory of every non-learned method the harness CLIs run (core eight + baselines 9-11)."""
    from .core8 import core8
    table = dict(core8(device=device))
    table["screen"] = lambda: screen_search(device=device)
    from .stat_baselines import LopoHeuristic, load_stat_gbm, scorer_search
    table["lopo_heuristic"] = lambda: scorer_search(LopoHeuristic(), device=device)
    table["stat_gbm"] = lambda: scorer_search(load_stat_gbm(), device=device)
    frozen = "runs/development/learned_search_v03/{}/frozen.pt"   # repaired-B_known release checkpoints
    table["g_theta"] = lambda: model_search(__import__("os").environ.get("MORPHEUS_GTHETA", frozen.format("release")),
                                            device=device)
    table["g_theta_rowsplit"] = lambda: model_search(frozen.format("release_rowsplit"), device=device)
    from .ablation_selection import ablation_searches   # selection ablation (methods table)
    table.update(ablation_searches(device))
    return table
