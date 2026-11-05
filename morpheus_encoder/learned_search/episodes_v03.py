"""Studies and episodes for learned search v03 (spec 0.4-0.6, 0.9, 1.7).

Study columns (all within-patient centred, the per-patient intercepts of N):
``resid:G<j>`` residual molecular programmes (``residual_programmes``), ``programme:*`` the 91
Hallmark/Gavish scores, ``morph:kNN`` similarities to the frozen v03 dictionary (study-standardised
H-Optimus-0 embeddings; covered spots only) and ``gene:*`` a 2,000-gene dispersion pool (each
episode keeps a random 256).  Composition = the top three principal axes of the programme
scores (CellViT composition is not available for these studies).  ``known`` = programmes +
composition (baseline B_known).

Episodes: level L (L0 patient split, L1 other study same organ and platform, L2 other study
same organ other platform), a target drawn with equal probability from the molecular and the
morphology families, candidate pool = four spatial versions of residual programmes, programmes,
morphology similarities and 256 random pool genes, target variable excluded in every version
(and candidates with within-patient |r| >= 0.9 with the target on support rows).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .episodes import Episode, EpisodeSampler, Study, _Block, centre_scale

RESID_DIR = "runs/development/learned_search_v03/residual_programmes"
N_POOL = 2000
N_RESID_HVG = 768
N_EPISODE_GENES = 256


@dataclass
class StudyV03(Study):
    known: np.ndarray | None = None         # (n, c) programmes + composition, within-patient centred
    composition: np.ndarray | None = None   # (n, 3)
    sections: np.ndarray | None = None      # (n,) sample / section ids
    patient_means: np.ndarray | None = None  # (n_patients, q) raw per-patient means of base columns
    organ: str = ""
    morph_targets: list = field(default_factory=list)
    known_names: list = field(default_factory=list)   # programme names ("mp:..." / "hallmark:...") + "composition:j"
    panel: list = field(default_factory=list)         # genes the programme scores are computed from
    role: str = "meta_training"
    embed_raw: np.ndarray | None = None      # (n, D) float16, kept for cohorts standardised per fold
    morph_centres: np.ndarray | None = None
    comp_source: str = ""


def _dispersion_pool(L, groups, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-n genes by dispersion (var / mean of log-normalised expression); also their
    within- and between-patient variances."""
    import scipy.sparse as sp
    L = L.tocsc() if sp.issparse(L) else L
    mean = np.asarray(L.mean(0)).ravel()
    sq = np.asarray(L.multiply(L).mean(0)).ravel() if sp.issparse(L) else (L ** 2).mean(0)
    var = sq - mean ** 2
    disp = np.where(mean > 1e-3, var / np.maximum(mean, 1e-12), 0.0)
    pool = np.sort(np.argsort(-disp, kind="stable")[:n])
    return pool, disp[pool], var[pool]


CELLVIT_DIR = "/lambda/nfs/geeg/biorag3_persistent_20260711/spatial/hest_data/cellvit_seg"
#: every location holding ``<sample>_cellvit_seg.parquet`` files, searched in order
CELLVIT_DIRS = (CELLVIT_DIR,
                "/lambda/nfs/geeg/biorag3_persistent_20260711/morpheus_substrate_mirror/hest/ccrcc",
                "/lambda/nfs/geeg/biorag3_persistent_20260711/morpheus_substrate_mirror/hest/kidney_injury")


def cellvit_path(sid: str) -> str | None:
    import os
    for d in CELLVIT_DIRS:
        p = f"{d}/{sid}_cellvit_seg.parquet"
        if os.path.exists(p):
            return p
    return None
COMP_DIR = "runs/development/learned_search_v03/composition"
COMP_CLASSES = ("Neoplastic", "Inflammatory", "Connective")


def composition(raw, spec, keep: np.ndarray, P: np.ndarray, *, name: str) -> tuple[np.ndarray, str]:
    """Three composition variables per spot: CellViT PanNuke class fractions (neoplastic,
    inflammatory, connective) of the nuclei inside the spot footprint when every sample of the
    study has a CellViT segmentation (spots without nuclei take their sample's mean fractions),
    else the top three principal axes of the programme scores (recorded as the source)."""
    import os
    sids = np.asarray(raw.sample_id).astype(str)
    have = spec is not None and all(cellvit_path(s) is not None for s in np.unique(sids))
    if have:
        path = Path(COMP_DIR) / f"{name}_{raw.meta.get('cache_key', 'nokey')}.npz"
        try:
            if path.exists():
                with np.load(path) as z:
                    return z["comp"][keep], str(z["source"])
            import h5py
            from morpheus.tier2.substrates import hest_multi as hm
            from morpheus.tier2.substrates.hest import _cellvit_summary_sample
            comp = np.zeros((raw.n, 3))
            for sid in np.unique(sids):
                rows = np.flatnonzero(sids == sid)
                with h5py.File(hm._paths(spec, sid)["st"], "r") as f:
                    radius = float(f["uns/spatial/ST/scalefactors/spot_diameter_fullres"][()]) / 2.0
                feats, cov = _cellvit_summary_sample(cellvit_path(sid),
                                                     np.asarray(raw.coords, dtype=np.float64)[rows], radius,
                                                     COMP_CLASSES)
                fr = feats[:, 2:]
                fr[~cov] = fr[cov].mean(0) if cov.any() else 0.0
                comp[rows] = fr
            src = f"cellvit PanNuke fractions {list(COMP_CLASSES)}"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, comp=comp, source=src)
            return comp[keep], src
        except Exception as exc:   # unreadable segmentation: fall back, recording why
            fallback = f"programme principal axes (CellViT failed: {type(exc).__name__}: {str(exc)[:80]})"
        else:
            fallback = "programme principal axes"
    else:
        fallback = "programme principal axes (no CellViT segmentation for every sample)"
    Pc = P - P.mean(0)
    Pz = Pc / np.maximum(Pc.std(0), 1e-8)
    _, _, vt = np.linalg.svd(Pz, full_matrices=False)
    return Pz @ vt[:3].T, fallback


def study_v03_from_raw(raw, *, name: str, platform: str, organ: str, dictionary: str | None = None,
                       resid_dir: str = RESID_DIR, spec=None, role: str = "meta_training") -> StudyV03:
    """One study's fixed variables.  Morphology standardisation (spec 0.5): a meta-training study
    uses its stored dictionary statistics; any other cohort (development, historical, blind) keeps
    its raw embedding and every episode re-standardises on that episode's SUPPORT spots
    (``SamplerV03._restandardise_morph``), never on all rows."""
    from morpheus.tier2.substrates import hest_multi as hm
    from . import residual_programmes as rp
    from .morphology_dictionary import DIR_V03, load_dictionary_v03, similarities_v03, standardise_stats

    keep = np.arange(raw.n)
    if raw.hoptimus is not None:
        keep = np.flatnonzero(np.asarray(raw.covered, dtype=bool))
    groups = np.asarray(raw.patient).astype(str)[keep]
    L = raw.logcpm()[keep]
    pool, _, _ = _dispersion_pool(L, groups, N_POOL)
    G = np.asarray(L[:, pool].todense(), dtype=np.float64)
    genes = [str(g) for g in np.asarray(raw.genes)[pool]]
    P = np.asarray(raw.programmes, dtype=np.float64)[keep]
    pnames = [str(c) for c in raw.programme_names]
    depth = np.log1p(np.asarray(raw.depth, dtype=np.float64))[keep]
    comp, comp_source = composition(raw, spec, keep, P, name=name)
    # residual programmes: HVGs on depth + composition + Gavish meta-programmes (cached per study)
    gav = [j for j, c in enumerate(pnames) if not c.startswith("hallmark")]
    rpath = Path(resid_dir) / f"{name}_{raw.meta.get('cache_key', 'nokey')}.npz"
    if rpath.exists():
        with np.load(rpath) as z:
            R, rk = z["scores"], int(z["k"])
    else:
        hv = np.argsort(-G.var(0), kind="stable")[:N_RESID_HVG]
        model = rp.fit(G[:, hv], np.column_stack([depth, comp, P[:, gav]]), groups)
        R, rk = model["scores"], model["k"]
        rpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(rpath, scores=R, loadings=model["loadings"], hvg=np.asarray(genes)[hv], k=rk, alpha=model["alpha"])
    blocks = [R, P]
    names = [f"resid:G{j + 1}" for j in range(rk)] + [f"programme:{c}" for c in pnames]
    views = {"resid": list(range(rk)), "programme": list(range(rk, rk + len(pnames)))}
    morph_cols = []
    if raw.hoptimus is not None:
        d = load_dictionary_v03(dictionary or DIR_V03)
        E = np.asarray(raw.hoptimus, dtype=np.float32)[keep]
        stats = d["study_stats"].get(name) if role == "meta_training" else None
        mu, sd = stats if stats is not None else standardise_stats(E)   # replaced per episode if not meta-training
        M = similarities_v03(E, d["centres"], mu, sd)
        morph_cols = list(range(len(names), len(names) + M.shape[1]))
        views["morph"] = morph_cols
        names += [f"morph:k{j:02d}" for j in d["kept"]]
        blocks.append(M)
    views["gene"] = list(range(len(names), len(names) + len(genes)))
    names += [f"gene:{g}" for g in genes]
    blocks.append(G)
    base = np.concatenate(blocks, 1)
    pats = np.unique(groups)
    pm = np.stack([base[groups == p].mean(0) for p in pats])
    inv = np.searchsorted(pats, groups)
    base = (base - pm[inv]).astype(np.float32)
    known = np.column_stack([P, comp])
    known = known - np.stack([known[groups == p].mean(0) for p in pats])[inv]
    compc = comp - np.stack([comp[groups == p].mean(0) for p in pats])[inv]
    depth_c = depth - np.array([depth[groups == p].mean() for p in pats])[inv]
    # gene targets: between-patient variance above the pool's lower quartile, top 300 by variance
    gcols = np.asarray(views["gene"])
    between = pm[:, gcols].var(0)
    ok = gcols[between >= np.quantile(between, 0.25)]
    gt = ok[np.argsort(-base[:, ok].var(0), kind="stable")[:300]]
    index = {c: i for i, c in enumerate(names)}
    excl = {}
    for j, c in zip(views["programme"], pnames):
        excl[j] = [index[f"gene:{g}"] for g in hm._programme_members(c) if f"gene:{g}" in index]
    nb6 = np.asarray(raw.neighbours[6], dtype=np.int64)[:, :6]
    if len(keep) < raw.n:
        remap = -np.ones(raw.n, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        nb6 = hm._remap_neighbours(nb6, keep, remap)
    return StudyV03(name=name, base=base, names=names, views=views, groups=groups, neighbours6=nb6,
                    coords=np.asarray(raw.coords, dtype=np.float32)[keep], embed=None, platform=platform,
                    target_cols=list(views["resid"]) + list(views["programme"]) + [int(c) for c in gt],
                    exclusions=excl, residual_cols=list(views["programme"]),
                    subsample_views={"gene": N_EPISODE_GENES}, depth=depth_c.astype(np.float32),
                    known=known.astype(np.float32), composition=compc.astype(np.float32),
                    known_names=pnames + [f"composition:{j}" for j in range(comp.shape[1])],
                    panel=[str(g) for g in np.asarray(raw.genes)],
                    sections=np.asarray(raw.sample_id).astype(str)[keep], patient_means=pm.astype(np.float32),
                    organ=organ, morph_targets=morph_cols, role=role, comp_source=comp_source,
                    embed_raw=None if role == "meta_training" or raw.hoptimus is None
                    else np.asarray(raw.hoptimus, dtype=np.float16)[keep],
                    morph_centres=None if raw.hoptimus is None else load_dictionary_v03(dictionary or DIR_V03)["centres"])


def studies_v03(partitions: str, roles=("meta_training", "development"), *, only: set | None = None,
                min_patients: int = 2) -> tuple[dict, dict]:
    """{role: [StudyV03]} and a load report."""
    from morpheus.tier2.substrates import hest_multi as hm
    out, info = {r: [] for r in roles}, {"loaded": {}, "skipped": {}}
    for role in roles:
        for key, spec in sorted(hm.study_specs(partitions, roles=[role]).items()):
            if only is not None and key not in only:
                continue
            try:
                raw = hm.load_study(spec)
                if len(np.unique(raw.patient)) < min_patients:
                    info["skipped"][key] = "fewer than 2 patients"
                    continue
                st = study_v03_from_raw(raw, name=key, platform=spec.platform, organ=str(spec.organ).lower(),
                                        spec=spec, role=role)
                out[role].append(st)
                info["loaded"][key] = {"role": role, "n": st.n, "patients": len(st.patients), "organ": st.organ,
                                       "platform": st.platform, "n_resid": len(st.views["resid"]),
                                       "n_morph": len(st.views.get("morph", [])), "composition": st.comp_source,
                                       "morph_standardisation": "stored study statistics" if role == "meta_training"
                                       else "per episode on support spots"}
            except Exception as exc:
                info["skipped"][key] = f"{type(exc).__name__}: {exc}"[:200]
            hm._RAW_MEMO.clear()
    return out, info


# ------------------------------------------------------------------ sampler
LEVEL_SCHEDULE_EARLY = {"L0": 1.0}
LEVEL_SCHEDULE_LATE = {"L0": 0.6, "L1": 0.3, "L2": 0.1}


class SamplerV03(EpisodeSampler):
    """Spec 1.7 episodes.  ``level`` per episode from ``level_probs(step)``; the trainer's step is
    approximated in each loader worker by (batches produced) x (number of workers)."""

    def __init__(self, studies, *, early_steps: int = 4000, n_workers: int = 1, episodes_per_step: int = 8, **kw):
        kw.setdefault("concepts", False)
        kw.setdefault("morph_target_prob", 0.5)
        kw.setdefault("proxy_threshold", 0.9)
        kw.setdefault("cross_study_prob", 0.0)
        kw.setdefault("n_query_patients", 4)
        kw.setdefault("min_support_patients", 2)
        super().__init__(studies, **kw)
        self.early_steps, self.n_workers, self.produced = early_steps, max(1, n_workers), 0
        self.episodes_per_step = max(1, episodes_per_step)
        self.force_level = None

    def level_probs(self) -> dict:
        step = self.produced * self.n_workers / self.episodes_per_step
        return LEVEL_SCHEDULE_EARLY if step < self.early_steps else LEVEL_SCHEDULE_LATE

    def _query_candidates(self, study, level):
        if level == "L0":
            return []
        same_platform = level == "L1"
        return [s for s in self.studies if s is not study and s.organ == study.organ
                and (s.platform == study.platform) == same_platform]

    def _patients(self, study, rng, n_sup, n_q, query_study):
        if query_study is not None:   # L1/L2: ALL patients of the source study / of the target study
            return list(study.train_patients), list(query_study.train_patients)
        pool = rng.permutation(self._pool(study))
        if len(pool) < 4:
            raise ValueError(f"{study.name}: fewer than 4 patients for a 2/2 patient split")
        nq = int(np.clip(len(pool) // 3, 2, n_q))
        ns = int(min(n_sup, len(pool) - nq))
        return list(pool[:ns]), list(pool[ns:ns + nq])

    def _target(self, study, rng, *, cross: bool = False):
        morph = list(getattr(study, "morph_targets", []))
        if morph and rng.random() < self.morph_target_prob:
            c = int(morph[int(rng.integers(len(morph)))])
        else:
            mol = [c for c in study.target_cols if not (cross and str(study.names[c]).startswith("resid:"))]
            c = int(mol[int(rng.integers(len(mol)))])
        return study.names[c], study.base[:, c], [c] + list(study.exclusions.get(c, [])), []

    def sample(self, rng=None, **kw):
        self.produced += 1
        return super().sample(rng, **kw)

    def _sample(self, rng, *, kind, family, fraction, study):
        lp = {self.force_level: 1.0} if self.force_level else self.level_probs()
        level = rng.choice(list(lp), p=np.array(list(lp.values())) / sum(lp.values()))
        study = study if study is not None else self.studies[int(rng.integers(len(self.studies)))]
        kind = kind or ("controlled" if rng.random() < self.controlled_prob else "real")
        qstudy = None
        if level != "L0":
            cands = self._query_candidates(study, level)
            if not cands:
                level = "L0"
            else:
                qstudy = cands[int(rng.integers(len(cands)))]
        sup_p, q_p = self._patients(study, rng, self.n_sup, self.n_q, qstudy)
        tname, tvals, excl, exv = self._target(study, rng, cross=qstudy is not None)
        return self._build(study, qstudy, sup_p, q_p, tname, tvals, excl, exv, rng, kind=kind, family=family,
                           fraction=fraction, level=level)

    def episode_for(self, study, target: str, support_patients, query_patients, *, rng, kind: str = "real",
                    query_study=None, level: str = "L0", family=None, fraction=None):
        """A fixed task: given study, target column name and support / query patients (harness)."""
        c = study.index[target]
        return self._build(study, query_study, list(support_patients), list(query_patients), target,
                           study.base[:, c], [c] + list(study.exclusions.get(c, [])), [], rng, kind=kind,
                           family=family, fraction=fraction, level=level)

    def _build(self, study, qstudy, sup_p, q_p, tname, tvals, excl, exv, rng, *, kind, family, fraction, level):
        if qstudy is not None and tname not in qstudy.index:
            raise ValueError("target not defined in the query study")
        shared = None if qstudy is None else {n for n in qstudy.names if not n.startswith("resid:")}
        cols = self._columns(study, rng, excl, exv, shared, sup_p)
        self._restandardise_morph(study, sup_p, list(sup_p) + ([] if qstudy is not None else list(q_p)))
        if qstudy is not None:
            self._restandardise_morph(qstudy, q_p, q_p)
        if study.morph_targets and tname.startswith("morph:"):
            tvals = study.base[:, study.index[tname]]
        cols = self._drop_proxies(study, cols, tvals, sup_p)
        if len(cols) < 8:
            raise ValueError("too few candidate columns")
        resid_cols = np.array([c for c in study.residual_cols if c not in set(excl)], dtype=int)
        saved = self.max_rows
        if qstudy is not None:   # all patients on both sides: cap rows per patient, not patients
            self.max_rows = int(max(50, min(saved, 6000 // max(1, len(sup_p), len(q_p)))))
        try:
            blocks = self._make_blocks(study, qstudy, sup_p, q_p, cols, rng, tname, tvals, resid_cols, False)
        finally:
            self.max_rows = saved
        ep = self._assemble(study, qstudy, blocks, cols, rng, kind=kind, family=family, fraction=fraction,
                            tname=tname, concept_target=None, sup_p=sup_p, q_p=q_p)
        self._extras(ep, study, qstudy, blocks, cols, tname)
        ep.info["level"] = level
        return ep

    @staticmethod
    def _restandardise_morph(study, stat_patients, patients) -> None:
        """Cohorts outside meta-training: morphology similarities of ``patients`` from embeddings
        standardised on the SUPPORT spots (``stat_patients``) of this episode (spec 0.5 item 2)."""
        if study.embed_raw is None or not study.morph_targets:
            return
        from .morphology_dictionary import similarities_v03, standardise_stats
        srows = np.concatenate([study.rows_of[p] for p in stat_patients])
        mu, sd = standardise_stats(np.asarray(study.embed_raw[srows], dtype=np.float32))
        mc = np.asarray(study.morph_targets)
        for p in patients:
            r = study.rows_of[p]
            M = similarities_v03(np.asarray(study.embed_raw[r], dtype=np.float32), study.morph_centres, mu, sd)
            pi = int(np.searchsorted(study.patients, p))
            study.patient_means[pi, mc] = M.mean(0)
            study.base[np.ix_(r, mc)] = M - M.mean(0)

    def _extras(self, ep, study, qstudy, blocks, cols, tname):
        """known / composition / sections rows and the per-token within-patient covariance share."""
        (sb, qb), _, _ = blocks
        qs = qstudy or study
        ep.info["known"] = np.concatenate([study.known[sb.rows], qs.known[qb.rows]]).astype(np.float32)
        ep.info["known_names"] = list(getattr(study, "known_names", []) or [])
        ep.info["panel"] = list(getattr(study, "panel", []) or [])
        if ep.info["known_names"]:
            from .relation_v03 import known_exclusions
            ep.info["known_dropped"] = [ep.info["known_names"][i] for i in
                                        known_exclusions(ep.info["known_names"], tname, ep.info["panel"])]
        ep.info["composition"] = np.concatenate([study.composition[sb.rows], qs.composition[qb.rows]])
        ep.info["sections"] = np.concatenate([study.sections[sb.rows], qs.sections[qb.rows]])
        # within share at the token level on support rows (raw units), between from patient means
        m = len(cols)
        tcol = study.index[tname]
        yraw = study.base[sb.rows, tcol].astype(np.float64)
        Xs = sb.X.astype(np.float64)
        w = np.abs((Xs * yraw[:, None]).mean(0))
        pats, cnt = np.unique(sb.groups, return_counts=True)
        pidx = np.searchsorted(study.patients, pats)
        pm = study.patient_means[pidx]
        wts = cnt / cnt.sum()
        dm = pm - (wts[:, None] * pm).sum(0)
        bcov = np.abs((wts[:, None] * dm[:, cols] * dm[:, [tcol]]).sum(0))
        between = np.concatenate([bcov, bcov, bcov, np.zeros(m)])
        ep.info["within_share"] = (w / (w + between + 1e-12)).astype(np.float32)
        ep.info["study_patients"] = len(study.patients)


def task_from_episode(ep):
    """``relation_v03.Task`` of a v03 episode."""
    from .relation_v03 import Task
    return Task(X=ep.X, y=np.asarray(ep.y, dtype=np.float64), n_support=ep.n_support, groups=ep.groups,
                depth=None if ep.nuisance is None else np.asarray(ep.nuisance, dtype=np.float64),
                known=ep.info.get("known"), composition=ep.info.get("composition"), nb6=ep.nb6,
                sections=ep.info.get("sections"), within_share=ep.info.get("within_share"),
                known_names=ep.info.get("known_names") or None, target_name=ep.target_name,
                panel=ep.info.get("panel") or None)


class RowSplitV03(SamplerV03):
    """Ablation: L0 only, support and query rows from the SAME patients (60/40 row split)."""

    rowsplit = True

    def level_probs(self) -> dict:
        return {"L0": 1.0}

    def _patients(self, study, rng, n_sup, n_q, query_study):
        pool = rng.permutation(self._pool(study))
        pats = list(pool[:min(n_sup + n_q, len(pool))])
        return pats, pats

    def _make_blocks(self, *a, **k):
        from .episodes import RowSplitSampler
        return RowSplitSampler._make_blocks(self, *a, **k)
