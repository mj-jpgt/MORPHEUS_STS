"""Benchmark-side study specs (read-only on hest_multi): adds the _b3 lung layouts converted by Builder B3."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "morpheus_work_sw"))
from morpheus.tier2.substrates import hest_multi as hm

STORE = hm.STORE
B3_LAYOUTS = {
    "gse307534_lung_precursor_b3": f"{STORE}/spatial/external_lung_b3_20261027/GSE307534/hest_layout",
    "nakajima_luad_b3": f"{STORE}/spatial/external_lung_b3_20261027/NAKAJIMA_LUAD/hest_layout",
}


EMBED_ROOTS = {"nakajima_luad_b3": (f"{STORE}/spatial/external_lung_b3_20261027/NAKAJIMA_LUAD/hoptimus_embeddings/embeddings",)}


def b3_specs() -> dict:
    """StudySpec per _b3 cohort whose layout exists (role external_natural_task, layout external)."""
    out = {}
    for key, d in B3_LAYOUTS.items():
        if not os.path.isdir(f"{d}/st"):
            continue
        ids = sorted(f[:-5] for f in os.listdir(f"{d}/st") if f.endswith(".h5ad"))
        meta = {i: json.load(open(f"{d}/metadata/{i}.json")) for i in ids}
        out[key] = hm.StudySpec(key=key, dataset_title=key, role="external_natural_task", layout="external",
                                sample_ids=tuple(ids), patient_of={i: str(meta[i]["patient"]) for i in ids},
                                organ=str(meta[ids[0]].get("organ", "")) if ids else "", st_technology="Visium",
                                platform="visium", embed_roots=tuple(EMBED_ROOTS.get(key, ())), data_dir=d)
    return out


def all_specs() -> dict:
    s = dict(hm.study_specs())
    s.update(b3_specs())
    return s
