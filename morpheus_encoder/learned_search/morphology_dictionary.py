"""Frozen morphology dictionary (learned search v02).

A spot's morphology variables are the cosine similarities of its frozen
H-Optimus-0 embedding to 64 centres.  The centres come from spherical k-means
(cosine, k-means++ initialisation, fixed seed) fitted ONCE on a patient-balanced
sample of meta-training spots and saved with provenance; nothing downstream
updates them and no gradient reaches the embeddings.

    python -m morpheus_encoder.learned_search.morphology_dictionary \
        --out runs/development/learned_search_v02/morphology_dictionary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_DIR = "runs/development/learned_search_v02/morphology_dictionary"
K = 64
SEED = 20261021


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)


def spherical_kmeans(E: np.ndarray, k: int = K, *, seed: int = SEED, n_iter: int = 60, device: str = "cpu",
                     tol: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Cosine k-means on unit rows; returns (k, D) unit centres and fit info."""
    import torch

    X = torch.as_tensor(_unit(E), device=device)
    n = len(X)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = [int(torch.randint(n, (1,), generator=g))]
    dist = 1.0 - X @ X[idx[0]]
    for _ in range(1, k):  # k-means++ on cosine distance
        p = torch.clamp(dist, min=0).double().cpu() ** 2
        j = int(torch.multinomial(p / p.sum(), 1, generator=g)) if float(p.sum()) > 0 else int(torch.randint(n, (1,), generator=g))
        idx.append(j)
        dist = torch.minimum(dist, 1.0 - X @ X[j])
    C = X[idx].clone()
    prev = -np.inf
    for it in range(n_iter):
        S = X @ C.T
        lab = S.argmax(1)
        obj = float(S.gather(1, lab[:, None]).mean())
        newC = torch.zeros_like(C).index_add_(0, lab, X)
        empty = newC.norm(dim=1) < 1e-8
        if empty.any():  # re-seed empty clusters at the worst-fitted spots
            worst = S.max(1).values.argsort()[: int(empty.sum())]
            newC[empty] = X[worst]
        C = newC / newC.norm(dim=1, keepdim=True)
        if obj - prev < tol:
            break
        prev = obj
    sizes = torch.bincount((X @ C.T).argmax(1), minlength=k).cpu().numpy()
    return C.cpu().numpy().astype(np.float32), {"iterations": it + 1, "mean_cosine": obj,
                                                 "cluster_sizes": sizes.tolist()}


def similarities(embed: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(n, k) cosine similarities of spot embeddings to the frozen centres."""
    return (_unit(embed) @ _unit(centres).T).astype(np.float32)


_CACHE: dict = {}


def load_dictionary(path: str | Path = DEFAULT_DIR) -> np.ndarray:
    """The frozen (k, D) centres; verifies the sha256 recorded in PROVENANCE.json."""
    path = Path(path)
    key = str(path.resolve())
    if key not in _CACHE:
        prov = json.loads((path / "PROVENANCE.json").read_text())
        if _sha256(path / "centres.npz") != prov["centres_sha256"]:
            raise ValueError(f"{path}/centres.npz does not match PROVENANCE.json")
        with np.load(path / "centres.npz") as z:
            _CACHE[key] = np.asarray(z["centres"], dtype=np.float32)
    return _CACHE[key]


def morph_view(embed: np.ndarray, path: str | Path = DEFAULT_DIR) -> tuple[np.ndarray, list]:
    """The ``morph`` view of a study: (n, 64) similarities and their column names."""
    C = load_dictionary(path)
    return similarities(embed, C), [f"morph:k{j:02d}" for j in range(len(C))]


def balanced_sample(embeds: dict, *, per_patient: int, seed: int = SEED) -> tuple[np.ndarray, dict]:
    """``embeds``: patient key -> (n_p, D) covered embeddings; up to ``per_patient`` each."""
    rng = np.random.default_rng(seed)
    parts, counts = [], {}
    for p in sorted(embeds):
        E = embeds[p]
        take = np.sort(rng.choice(len(E), min(per_patient, len(E)), replace=False))
        parts.append(np.asarray(E[take], dtype=np.float32))
        counts[p] = int(len(take))
    return np.concatenate(parts), counts


def fit_from_partitions(out: str | Path = DEFAULT_DIR, *, partitions: str = "dbench/protocols/sprint_partitions_v01.json",
                        per_patient: int = 2000, k: int = K, seed: int = SEED, device: str = "cpu") -> dict:
    from morpheus.tier2.substrates import hest_multi as hm

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    embeds, studies, skipped = {}, {}, {}
    for key, spec in sorted(hm.study_specs(partitions, roles=["meta_training"]).items()):
        try:
            raw = hm.load_study(spec)
        except Exception as exc:  # report and continue
            skipped[key] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        if raw.hoptimus is None:
            skipped[key] = "no H-Optimus-0 embedding"
            hm._RAW_MEMO.clear()
            continue
        cov = np.asarray(raw.covered, dtype=bool)
        pats = np.asarray(raw.patient).astype(str)
        for p in np.unique(pats[cov]):
            embeds[f"{key}:{p}"] = np.asarray(raw.hoptimus[cov & (pats == p)], dtype=np.float32)
        studies[key] = {"n_covered": int(cov.sum()), "n_patients": int(len(np.unique(pats[cov]))),
                        "cache_file": raw.meta.get("cache_file"), "cache_key": raw.meta.get("cache_key")}
        hm._RAW_MEMO.clear()
    E, counts = balanced_sample(embeds, per_patient=per_patient, seed=seed)
    t0 = time.time()
    C, info = spherical_kmeans(E, k, seed=seed, device=device)
    np.savez(out / "centres.npz", centres=C)
    prov = {"schema": "morpheus_morphology_dictionary/v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backbone": "H-Optimus-0 (frozen spot embeddings, 1536-d)", "method": "spherical k-means (cosine), k-means++",
            "k": k, "seed": seed, "per_patient_cap": per_patient, "n_spots": int(len(E)),
            "n_patients": len(counts), "partitions": partitions, "partitions_sha256": _sha256(partitions),
            "embedding_roots": list(hm.EMBED_ROOTS), "studies": studies, "skipped": skipped,
            "spots_per_patient": counts, "fit": {**info, "seconds": round(time.time() - t0, 1)},
            "centres_sha256": _sha256(out / "centres.npz")}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    return prov


# ------------------------------------------------------------------ v03 (spec 0.5)
DIR_V03 = "runs/development/learned_search_v03/morphology_dictionary"
MIN_STUDIES = 3          # a centre is kept only if populated by >= 3 meta-training studies
MIN_SPOTS_PER_STUDY = 20  # "populated": at least this many of the study's spots assigned to the centre


def standardise_stats(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and sd of a study's (or a cohort's support) embeddings."""
    E = np.asarray(E, dtype=np.float32)
    mu, sd = E.mean(0), E.std(0)
    return mu, np.where(sd < 1e-6, 1.0, sd).astype(np.float32)


def similarities_v03(embed: np.ndarray, centres: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return similarities((np.asarray(embed, dtype=np.float32) - mu) / sd, centres)


_CACHE_V03: dict = {}


def load_dictionary_v03(path: str | Path = DIR_V03) -> dict:
    """{"centres": (k_kept, D), "kept": original centre indices, "study_stats": {study: (mu, sd)}}."""
    path = Path(path)
    key = str(path.resolve())
    if key not in _CACHE_V03:
        prov = json.loads((path / "PROVENANCE.json").read_text())
        if _sha256(path / "centres.npz") != prov["centres_sha256"]:
            raise ValueError(f"{path}/centres.npz does not match PROVENANCE.json")
        with np.load(path / "centres.npz") as z:
            stats = {k[3:]: (z[k], z["sd_" + k[3:]]) for k in z.files if k.startswith("mu_")}
            _CACHE_V03[key] = {"centres": np.asarray(z["centres"], dtype=np.float32), "kept": z["kept"].tolist(),
                               "study_stats": stats}
    return _CACHE_V03[key]


def morph_view_v03(embed: np.ndarray, *, mu: np.ndarray | None = None, sd: np.ndarray | None = None,
                   path: str | Path = DIR_V03) -> tuple[np.ndarray, list]:
    """(n, k_kept) similarities of embeddings standardised by (mu, sd) (default: the given rows'
    own statistics, i.e. the cohort's support spots) and their names ``morph:kNN``."""
    d = load_dictionary_v03(path)
    if mu is None:
        mu, sd = standardise_stats(embed)
    return similarities_v03(embed, d["centres"], mu, sd), [f"morph:k{j:02d}" for j in d["kept"]]


def fit_v03(out: str | Path = DIR_V03, *, partitions: str = "dbench/protocols/sprint_partitions_v01.json",
            k: int = K, seed: int = SEED, device: str = "cpu") -> dict:
    from morpheus.tier2.substrates import hest_multi as hm

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts, study_of, stats, studies, skipped = [], [], {}, {}, {}
    for key, spec in sorted(hm.study_specs(partitions, roles=["meta_training"]).items()):
        try:
            raw = hm.load_study(spec)
        except Exception as exc:
            skipped[key] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        if raw.hoptimus is None:
            skipped[key] = "no H-Optimus-0 embedding"
            hm._RAW_MEMO.clear()
            continue
        E = np.asarray(raw.hoptimus[np.asarray(raw.covered, dtype=bool)], dtype=np.float32)
        mu, sd = standardise_stats(E)
        stats[key] = (mu, sd)
        parts.append((E - mu) / sd)
        study_of.append(np.full(len(E), len(studies)))
        studies[key] = {"n_covered": int(len(E)), "cache_key": raw.meta.get("cache_key")}
        hm._RAW_MEMO.clear()
    X = np.concatenate(parts)
    sid = np.concatenate(study_of)
    t0 = time.time()
    C, info = spherical_kmeans(X, k, seed=seed, device=device)
    lab = (_unit(X) @ C.T).argmax(1)
    counts = np.zeros((k, len(studies)), dtype=int)
    np.add.at(counts, (lab, sid), 1)
    n_pop = (counts >= MIN_SPOTS_PER_STUDY).sum(1)
    kept = np.flatnonzero(n_pop >= MIN_STUDIES)
    arrays = {"centres": C[kept], "kept": kept, "all_centres": C}
    for key, (mu, sd) in stats.items():
        arrays[f"mu_{key}"], arrays[f"sd_{key}"] = mu, sd
    np.savez(out / "centres.npz", **arrays)
    prov = {"schema": "morpheus_morphology_dictionary/v3", "spec": ".claude/morpheus_training_and_evaluation_spec.md 0.5",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backbone": "H-Optimus-0 (frozen spot embeddings, 1536-d)",
            "method": "per-study per-dimension standardisation, spherical k-means (cosine) on all covered "
                      "meta-training spots, k-means++ init",
            "k": k, "seed": seed, "n_spots": int(len(X)), "studies": studies, "skipped": skipped,
            "retention_rule": f"populated (>= {MIN_SPOTS_PER_STUDY} spots) by >= {MIN_STUDIES} studies",
            "n_kept": int(len(kept)), "kept": kept.tolist(), "studies_populating": n_pop.tolist(),
            "partitions": partitions, "partitions_sha256": _sha256(partitions), "embedding_roots": list(hm.EMBED_ROOTS),
            "fit": {**info, "seconds": round(time.time() - t0, 1)}, "centres_sha256": _sha256(out / "centres.npz")}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_DIR)
    ap.add_argument("--per-patient", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--v03", action="store_true")
    a = ap.parse_args(argv)
    if a.v03:
        prov = fit_v03(a.out if a.out != DEFAULT_DIR else DIR_V03, device=a.device)
        print(json.dumps({k: prov[k] for k in ("n_spots", "n_kept", "skipped", "centres_sha256")}, indent=1))
        return
    prov = fit_from_partitions(a.out, per_patient=a.per_patient, device=a.device)
    print(json.dumps({k: prov[k] for k in ("n_spots", "n_patients", "skipped", "centres_sha256")}, indent=1))


if __name__ == "__main__":
    main()
