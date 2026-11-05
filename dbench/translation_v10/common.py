"""translation_v10 shared code: paths, relation catalogue, frozen-relation scoring on any spatial cohort."""
from __future__ import annotations
import json, os, hashlib, time
import numpy as np

V06 = "/home/ubuntu/morpheus_work_v06"
TX = "/home/ubuntu/morpheus_work_tx"
OUT = f"{TX}/runs/development/translation_v10"
V08_DIR = f"{V06}/runs/confirmation/v08_downstream_strict_spatial_v08"
REGISTRY = f"{V06}/runs/confirmation/paper_eval_v01/registry.json"
EXT = "/lambda/nfs/geeg/biorag3_persistent_20260711/spatial/external_ccrcc_20261016"
SEED = 20261102


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bh(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.isfinite(ps)
    p = ps[ok]
    n = len(p)
    if n == 0:
        return out
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    r = np.empty(n)
    r[o] = np.minimum(q, 1.0)
    out[ok] = r
    return out


# ------------------------------------------------------------------ v08 frozen relations
GROUPS = json.load(open(f"{OUT}/groups.json"))
EXTRA_V08 = ["renal-spatial-v08-multiscale-20261026-ring_decay-1d4493b06b8f"]   # W1 dedup best member of G1


D_ROWS = {r["relation_id"]: r for r in json.load(open(f"{V06}/runs/development/novelty_v10/recovered_known_v01.json"))["rows"]}   # set D: 37 distinct v08 relations
STAGE_NAME = {"1": "replicated", "2": "internal", "3": "proposed-only"}


def v08_ids():
    base = json.load(open(f"{V08_DIR}/replication_summary.json"))["replicated"] + EXTRA_V08
    return base + [r for r in D_ROWS if r not in base]


def stage_of(rid):
    r = D_ROWS.get(rid)
    return (STAGE_NAME[r["stage"]], r["resolved_grade"]) if r else ("replicated-member-not-best", None)


def group_of(rid):
    g = GROUPS["member_to_group"].get(rid)
    return f"G{g}" if g else None


def is_group_best(rid):
    return rid in GROUPS["best"].values()


def frozen_path(rid):
    import glob
    p = f"{V08_DIR}/frozen_relations/{rid}.json"
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(f"{V06}/runs/confirmation/**/frozen_relations/{rid}.json", recursive=True))
    return hits[-1]


def short_id(rid):
    return rid.split("20261026-")[-1]


class FrozenV08:
    """Stored v03_rff parameters; predict = rv.predict on within-patient standardised inputs (no refit)."""

    def __init__(self, rid):
        self.rid = rid
        self.d = json.load(open(frozen_path(rid)))
        self.candidates = self.d["variable_names"]
        self.target = self.d["target"]
        self.model = {k: (np.asarray(v) if k != "blocks" else v) for k, v in self.d["model"].items()}

    def predict(self, Z):
        from morpheus_encoder.learned_search import relation_v03 as rv
        Z = np.asarray(Z, float).reshape(len(Z), -1)
        return np.asarray(rv.predict(self.model, Z[:, None, :], device="cpu"))[:, 0]


def score_v08(rel: FrozenV08, coh, sets, disc_panel, residual_ctx=None):
    """Per-spot relation score f_S on a cohort, same arithmetic as replication_v03.load_external/transform_cohort.
    Returns dict(fS, ok, Z, y, patient, section)."""
    from dbench.translation_v10 import values_v06_eaca09b as VV
    from dbench.downstream.relation import standardise_within
    Z = np.column_stack([VV.cohort_values(coh, c, sets, disc_panel, residual_ctx) for c in rel.candidates])
    y = VV.cohort_values(coh, rel.target, sets, disc_panel, residual_ctx)
    ok = np.isfinite(Z).all(1) & np.isfinite(y)
    pat = np.asarray(coh.patient).astype(str)
    Zs = np.full(Z.shape, np.nan)
    ys = np.full(len(y), np.nan)
    # within-patient standardisation on retained rows (rows with undefined operators dropped, never imputed)
    Zs[ok] = standardise_within(Z[ok], pat[ok])[0].reshape(ok.sum(), -1)
    ys[ok] = standardise_within(y[ok], pat[ok])[0].ravel()
    fS = np.full(len(y), np.nan)
    fS[ok] = rel.predict(Zs[ok])
    return {"fS": fS, "ok": ok, "Z": Z, "Zs": Zs, "y": y, "ys": ys, "patient": pat,
            "section": np.asarray(coh.section).astype(str)}


# ------------------------------------------------------------------ registry (v02/v07 frozen.pkl) relations
KIDNEY_REG = {  # label -> (pickle key, candidate id, registry relation id)
    "C1": ("v2_clean", "renal-spatial-v07-20261022-context_conditioned_programme-59b472a9f62f", "meylan:IFN-MHCII-I_C1QB_v2"),
    "C2": ("v2_clean", "renal-spatial-v07-20261022-context_conditioned_programme-6e645d9788a9", "meylan:Metal-response_v2"),
    "F12": ("bknown", "renal-spatial-v07-20261022-context_conditioned_programme-52020290494d", "meylan:EMT-I_registered"),
    "F13": ("v2_clean", "renal-spatial-v07-20261022-context_conditioned_programme-4d8c3d97e8ae", "meylan:IFN-MHCII-II_v2"),
    "Glut": ("v2_clean", "renal-spatial-v07-20261022-context_conditioned_programme-26bccb2c386b", "meylan:Glutathione_sign-only_v2"),   # set (c)
}
ONLY = [x for x in os.environ.get("W2_ONLY", "").split(",") if x]
_pk = {}


def registry_ff(key, cid):
    import pickle
    DB = f"{V06}/runs/development/downstream_blind/renal-spatial-v07-20261022"
    if key not in _pk:
        _pk[key] = pickle.load(open(f"{DB}/{key}/frozen.pkl", "rb"))
    return _pk[key][cid]


def score_registry(ff, coh, sets, disc_panel, residual_ctx=None):
    from dbench.translation_v10 import values_v06_eaca09b as VV
    Z = np.column_stack([np.asarray(VV.cohort_values(coh, c, sets, disc_panel, residual_ctx), float).ravel()
                         for c in ff.rel_nuis.candidates])
    ok = np.isfinite(Z).all(1)
    s = np.full(len(Z), np.nan)
    mu, sd = Z[ok].mean(0), Z[ok].std(0) + 1e-12
    s[ok] = ff.rel_nuis.predict((Z[ok] - mu) / sd)
    return {"fS": s, "ok": ok, "Z": Z}
