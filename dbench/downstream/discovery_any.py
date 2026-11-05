"""Discovery-cohort accessor for any hest_multi study (generalises dryrun.Discovery, which assumes the ccRCC HEST
mirror layout). Same attributes: coh, sets, section, patient, nb6, px_um, coords_um, genes, cellvit()."""
from __future__ import annotations

import json
import os

import numpy as np

MIRROR = "/lambda/nfs/geeg/biorag3_persistent_20260711/morpheus_substrate_mirror/hest/ccrcc"


def cohort(study: str):
    from morpheus.tier2.substrates import hest_multi as hm

    from dbench.scripts.external_replication import Cohort, load_discovery

    if study == "meylan2022_ccrcc":
        return load_discovery()
    return Cohort(study, hm.load_study(hm.study_specs()[study]), "fresh_frozen_whole_transcriptome", None, False)


def _paths(spec, s):
    """(metadata json, st h5ad, cellvit parquet) for a sample in the mirror (ccRCC) or the flat HEST layout."""
    if os.path.exists(f"{MIRROR}/{s}_metadata.json"):
        return f"{MIRROR}/{s}_metadata.json", f"{MIRROR}/{s}_st.h5ad", f"{MIRROR}/{s}_cellvit_seg.parquet"
    d = spec.data_dir
    if getattr(spec, "layout", "") == "mirror":
        return f"{d}/{s}_metadata.json", f"{d}/{s}_st.h5ad", f"{d}/{s}_cellvit_seg.parquet"
    return f"{d}/metadata/{s}.json", f"{d}/st/{s}.h5ad", f"{d}/cellvit_seg/{s}_cellvit_seg.parquet"


class DiscoveryAny:
    def __init__(self, study: str):
        from morpheus.tier2.substrates import hest_multi as hm
        from morpheus.tier2.substrates.hest_multi import _programme_sets

        self.study = study
        self.spec = hm.study_specs()[study]
        self.coh = cohort(study)
        r = self.coh.raw
        self.sets = _programme_sets()
        self.section = np.asarray(r.sample_id).astype(str)
        self.patient = np.asarray(r.patient).astype(str)
        self.nb6 = r.neighbours[6]
        def _um(m):
            return m.get("pixel_size_um_estimated", m.get("pixel_size_um"))
        um = {s: _um(json.load(open(_paths(self.spec, s)[0]))) for s in np.unique(self.section)}
        self.px_um = np.array([um[s] for s in self.section])
        self.coords_um = np.asarray(r.coords, float) * self.px_um[:, None]
        self.genes = list(r.genes)

    def cellvit(self):
        if hasattr(self, "_cv"):
            return self._cv
        import h5py

        from dbench.downstream import characterise as C
        from morpheus.tier2.substrates.hest import CELLVIT_CLASSES, _cellvit_summary_sample

        F = np.zeros((len(self.section), 2 + len(CELLVIT_CLASSES)))
        cov = np.zeros(len(self.section), bool)
        for s in np.unique(self.section):
            rows = np.flatnonzero(self.section == s)
            _, st, cvp = _paths(self.spec, s)
            if not os.path.exists(cvp):
                continue
            with h5py.File(st, "r") as f:
                rad = float(f["uns/spatial/ST/scalefactors/spot_diameter_fullres"][()]) / 2
            F[rows], cov[rows] = _cellvit_summary_sample(cvp, np.asarray(self.coh.raw.coords, float)[rows], rad,
                                                         CELLVIT_CLASSES)
        self._cv = {"fractions": F[:, 2:], "count": F[:, 0], "area": F[:, 1], "covered": cov,
                    "strata": C.composition_strata(F[:, 2:], cov)}
        return self._cv
