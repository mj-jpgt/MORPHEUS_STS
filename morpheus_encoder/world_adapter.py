"""dbench.World -> encoder inputs.

Deliberately the *only* file that knows the shape of a DiscoveryBench world, so
that when step 1 finalises the tensor layouts the reconciliation is one module.
`dbench` is imported lazily and never at module import time: this package must
be usable, testable and GPU-smoke-testable while DiscoveryBench is still being
written.

World interface assumed (from `dbench/world.py` as of this build)
-----------------------------------------------------------------
    world.modalities   dict[str, (n, d_m)]      observed X
    world.latents      dict[str, (n,) | (n,d)]  oracle -- NEVER read here
    world.nuisance     dict[str, (n,) | (n,d)]  known covariates C
    world.env          EnvMeta(role, labels: dict[str, (n,)], world_seed, ...)
    world.structure    Structure(kind, coords, time, unit, neighbors, edges, ...)
    world.interventions Interventions(assignment, targets, kind, values) | None
    world.target       (n,) | None
    world.truth        sealed on `world.blind()`

Assumptions made here, to be reconciled with step 1
---------------------------------------------------
A1. Rows are scientific units and every modality has the same n. A world whose
    modalities have different row counts (an event-level and a jet-level table,
    say) needs an explicit correspondence, and this adapter refuses rather than
    guesses.
A2. `world.domain` selects the adapter class. `molecular` -> gene tokens,
    `tabular` -> MLP, `temporal`/`spatial` grid worlds -> the grid adapter only
    when `meta["grid"]` gives the field shape, otherwise tabular.
A3. Environment metadata E is built by one-hot encoding `env.labels`. This is a
    placeholder for anything richer step 1 exposes; `d_env` is reported so the
    encoder can be constructed to match.
A4. Nuisance C is NOT fed to the encoder. Adjusting for nuisance is the
    compiler's job (parent residualization), and mixing it into the
    representation would make "the object survives adjustment for density" a
    statement about a quantity the encoder had already absorbed.
A5. Latents are never read. `oracle_matrix()` exists for step 2 and stays there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .adapters import (FrozenPathologyAdapter, GeneTokenAdapter,
                       ParticleSetAdapter, SpatioTemporalGridAdapter,
                       TabularAdapter)

#: which adapter class a dbench domain maps to by default
DOMAIN_ADAPTER = {
    "molecular": "gene",
    "spatial": "tabular",
    "temporal": "tabular",
    "graph": "tabular",
    "particle": "particle",
    "tabular": "tabular",
}


@dataclass
class WorldInputs:
    X: dict = field(default_factory=dict)
    env: torch.Tensor | None = None
    structure: dict = field(default_factory=dict)
    recon_target: torch.Tensor | None = None
    n: int = 0
    d_env: int = 0
    meta: dict = field(default_factory=dict)


def _as_2d(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a.reshape(a.shape[0], -1)


def one_hot_env(labels: dict, n: int) -> tuple:
    """E as a one-hot block per grouping variable. Returns (array, columns)."""
    cols, blocks = [], []
    for key in sorted(labels):
        v = np.asarray(labels[key]).reshape(-1)
        if v.shape[0] != n:
            continue
        levels = sorted({str(x) for x in v.tolist()})
        if len(levels) > 64:
            # A grouping variable with more levels than rows-per-level is not a
            # grouping variable; carry it as a scalar code instead of blowing
            # d_env up to n.
            idx = {lv: i for i, lv in enumerate(levels)}
            blocks.append(np.array([[idx[str(x)] / max(len(levels) - 1, 1)]
                                    for x in v]))
            cols.append("%s:code" % key)
            continue
        idx = {lv: i for i, lv in enumerate(levels)}
        b = np.zeros((n, len(levels)))
        for r, x in enumerate(v):
            b[r, idx[str(x)]] = 1.0
        blocks.append(b)
        cols.extend("%s=%s" % (key, lv) for lv in levels)
    if not blocks:
        return np.zeros((n, 0)), []
    return np.concatenate(blocks, axis=1), cols


def world_to_inputs(world, device="cpu", dtype=torch.float32,
                    include_env: bool = True,
                    recon_modality: str | None = None) -> WorldInputs:
    """Convert a (blinded) World into encoder inputs. Never touches `truth`."""
    mods = dict(world.modalities)
    if not mods:
        raise ValueError("world %r exposes no modalities" % getattr(world, "name", "?"))
    ns = {m: int(np.asarray(v).shape[0]) for m, v in mods.items()}
    if len(set(ns.values())) != 1:
        raise ValueError(
            "modalities disagree on row count %s. A world whose modalities do "
            "not share scientific units needs an explicit correspondence; this "
            "adapter will not guess one." % ns)
    n = next(iter(ns.values()))
    X = {m: torch.as_tensor(_as_2d(v), dtype=dtype, device=device)
         for m, v in mods.items()}

    env_t, cols = None, []
    if include_env:
        e, cols = one_hot_env(getattr(world.env, "labels", {}) or {}, n)
        env_t = (torch.as_tensor(e, dtype=dtype, device=device)
                 if e.shape[1] else None)

    st = world.structure
    structure = {}
    for f in ("time", "unit"):
        v = getattr(st, f, None)
        if v is not None:
            structure[f] = torch.as_tensor(np.asarray(v), device=device)
    if getattr(st, "neighbors", None) is not None:
        structure["neighbors"] = st.neighbors
    if getattr(st, "coords", None) is not None:
        structure["coords"] = torch.as_tensor(
            np.asarray(st.coords), dtype=dtype, device=device)

    tgt = None
    if recon_modality is not None:
        tgt = X[recon_modality]

    return WorldInputs(
        X=X, env=env_t, structure=structure, recon_target=tgt, n=n,
        d_env=0 if env_t is None else int(env_t.shape[1]),
        meta={"name": getattr(world, "name", ""),
              "family": getattr(world, "family", ""),
              "domain": getattr(world, "domain", ""),
              "env_role": getattr(world.env, "role", ""),
              "env_columns": cols,
              "modality_dims": {m: int(v.shape[1]) for m, v in X.items()}})


def default_adapters(inputs: WorldInputs, domain: str | None = None,
                     d_token: int = 128, n_tokens: int = 16,
                     overrides: dict | None = None) -> dict:
    """Build one adapter per modality from the world's declared domain.

    `overrides` maps modality -> adapter kind, so a HEST world can say
    {"he": "pathology"} without this function having to guess that a 1536-wide
    float block is an H-optimus embedding.
    """
    domain = domain or inputs.meta.get("domain", "tabular")
    overrides = overrides or {}
    out = {}
    for m, x in inputs.X.items():
        kind = overrides.get(m, DOMAIN_ADAPTER.get(domain, "tabular"))
        d_in = int(x.shape[1])
        if kind == "gene":
            out[m] = GeneTokenAdapter(m, d_in, d_token=d_token,
                                      n_tokens=n_tokens)
        elif kind == "pathology":
            out[m] = FrozenPathologyAdapter(m, d_embed=d_in, d_token=d_token,
                                            n_tokens=min(n_tokens, 8))
        elif kind == "particle":
            out[m] = ParticleSetAdapter(m, d_in, d_token=d_token,
                                        n_tokens=min(n_tokens, 8))
        elif kind == "grid":
            raise ValueError(
                "the grid adapter needs the field shape; build "
                "SpatioTemporalGridAdapter explicitly with (C, H, W)")
        else:
            out[m] = TabularAdapter(m, d_in, d_token=d_token,
                                    n_tokens=min(n_tokens, 8))
    return out
