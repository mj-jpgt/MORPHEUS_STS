import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from dbench.downstream import characterise_v10 as ch
from morpheus_encoder.learned_search import relation_v03 as rv

IDS = ["71ed4499fb99", "67256d9e53d5", "93b02a9038ea", "caf32d321d27"]
OUT = Path.home() / "scratch_w7"
CARDS = Path("runs/development/characterisation_v10/cards")
ctx = ch.Ctx()
rels = ch.relation_list("1")
for i12 in IDS:
    r = next(x for x in rels if x["relation_id"].endswith(i12))
    rid = r["relation_id"]
    rel = ch.Relation(ctx, rid, r["names"], r["target"], r["family"], r["row_views"])
    sec = np.asarray(ctx.sec[rel.ok]).astype(str)
    q = sec == "INT21"
    pq = np.unique(rel.pat[q])
    info = {"relation_id": rid, "n_rows_all": int(len(sec)), "n_rows": int(q.sum()), "patients_in_q": [str(p) for p in pq]}
    pat_rows = np.isin(rel.pat, pq)
    info["patient_rows_total"] = int(pat_rows.sum())
    info["patient_sections"] = sorted(set(sec[pat_rows].tolist()))
    info["patient_has_rows_outside_INT21"] = bool(pat_rows.sum() != q.sum())
    q = pat_rows  # hold out whole patient
    s = ~q
    base = rv.fit_baseline(rel.ys[s], rel.pat[s], rel.D[s], yq=rel.ys[q], C_q=rel.D[q])
    o = rv.fit_sets(rel.Z[s][:, None, :], base, rel.ys[s], rel.pat[s], Vq=rel.Z[q][:, None, :], yq=rel.ys[q],
                    groups_q=rel.pat[q], device=ch.DEVICE, return_beta=True)
    pred = np.asarray(o["pred_q"])[:, 0]
    rq = np.asarray(base.r_q)
    keep = sec[q] == "INT21"
    yq = rel.ys[q]
    sst_all = float((yq ** 2).sum())
    r2b = 1 - float((rq ** 2).sum()) / sst_all
    r2f = 1 - float(((rq - pred) ** 2).sum()) / sst_all
    sk = float((yq[keep] ** 2).sum())
    r2b_i = 1 - float((rq[keep] ** 2).sum()) / sk
    r2f_i = 1 - float(((rq[keep] - pred[keep]) ** 2).sum()) / sk
    bc = np.asarray(ctx.coh.raw.barcode)[rel.ok].astype(str)[q][keep]
    coords = np.asarray(ctx.coh.coords)[rel.ok][q][keep]
    old = dict(np.load(CARDS / rid / "spot_values.npz", allow_pickle=True))
    idx = {(a, b): j for j, (a, b) in enumerate(zip(old["sample_id"].astype(str), old["barcode"].astype(str)))}
    sel = np.array([idx[("INT21", b)] for b in bc])
    assert (old["sample_id"][sel].astype(str) == "INT21").all()
    st_in = old["state"][sel]
    assert np.allclose(old["target_std"][sel], yq[keep], atol=1e-5), "target mismatch"
    assert np.allclose(old["coords"][sel], coords)
    ph = pred[keep]; nr = rq[keep]
    np.savez_compressed(OUT / f"{i12}_heldout_int21.npz", barcode=bc, sample_id=sec[q][keep], coords=coords,
        patient=rel.pat[q][keep], target_std=yq[keep], state_heldout=ph, state_insample=st_in,
        nuis_res_heldout=nr, rel_res_heldout=nr - ph)
    info.update({"n_support_rows": int(s.sum()), "r2_baseline_heldout": r2b_i, "r2_baseline_plus_relation_heldout": r2f_i,
        "r2_gain_heldout": r2f_i - r2b_i, "r2_gain_wholepatient": r2f - r2b,
        "spearman_heldout_vs_insample": float(spearmanr(ph, st_in)[0]),
        "r2_note": "SST = sum of squared within-patient-z target over INT21 rows; baseline = depth ridge fit on the other patients",
        "keys_o": sorted(o.keys())})
    (OUT / f"{i12}_heldout_int21.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
