"""Blind-discovery run directory -> downstream pipeline (renal_spatial_v07, ependymoma_spatial_v01 and the
draft cohorts). Reads a registered run directory READ ONLY (campaign_manifest.json, funnel_aggregate.json,
<ledger>.jsonl, receipts/) and writes cards under an output root outside runs/prospective.

    PYTHONPATH=$PWD:$HOME/q_pylib ~/morpheus_venv/bin/python -m dbench.downstream.blind \
        --run-dir runs/prospective/renal-spatial-v07-<date> --out runs/development/downstream_blind/<campaign> \
        --stage replication

* ``replication``: every finding at ledger stage ``internal`` (the internally replicated set) is frozen on ALL
  discovery patients with its relation of record: ``RelationV03RFF`` under ``v03_rff`` (variable names + relation
  constants + commit, deterministic refit, stored parameters verified to reproduce the refit; both fingerprinted)
  or ``RelationOfRecord`` (general_discovery.FittedRelation) under the older relations; registry and internal-validation
  locks are written from the funnel, and the family goes through replication v03 (one BH family per cohort).
  For ependymoma the external step is recorded as unavailable (Section 2.1).
* after replication, each externally replicated finding continues through the step functions exercised by the
  dry run (dryrun.characterisation_common for 1.1-1.6 with 1.7 off, novelty_step, therapeutic_step; the two
  assay_distillation_v03 state definitions are frozen with StateCard.set_state_definitions at the
  characterisation lock; distillation 7a runs on state (b)); that per-finding driver is assembled when the first
  blind finding passes replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import locks
from .relation import dominant_term, fit_baseline, proper_partitions, standardise_within


# The strict priority tier currently shares an EMT-I target.  Cache immutable
# B_known columns by study/target/row identity so confirmation does not rebuild
# the same episode for every relation.
_KNOWN_COLUMNS_CACHE = {}


class RelationOfRecord:
    """A relation of record (general_discovery.FittedRelation) fitted once on all discovery patients and frozen.
    f_S(Z) = full(Z, baseline = 0) - c (c = its discovery mean): the hypothesis part of the joint model;
    f_decomp(Z) = competitor(Z, 0) - c' for |S| >= 2 (the additive / best-decomposition competitor)."""

    def __init__(self, candidates, target, baseline, relation, Z, r, groups, feature_form="linear_subspace",
                 part="full", fitted=None, offset=None):
        self.candidates, self.target, self.baseline, self.relation = list(candidates), target, baseline, relation
        self.part, self.feature_form = part, feature_form
        if fitted is None:
            from morpheus_encoder.general_discovery import VALIDATION_ALPHA_GRID, Candidate, FittedRelation

            k = Z.shape[1]
            cand = Candidate(variables=tuple(range(k)), feature_form=feature_form, proposal_family="downstream_freeze")
            fitted = FittedRelation(cand, Z, r, context=None, baseline=np.zeros((len(r), 0)), model="rbf",
                                    ridge_alpha=1.0, rbf_features=24, groups=groups, alpha_grid=VALIDATION_ALPHA_GRID,
                                    relation=relation)
        self.fitted = fitted
        self.offset = 0.0
        self.offset = float(np.mean(self.predict(Z))) if offset is None else offset
        # FittedRelation is not picklable; the frozen object is fingerprinted by its deterministic fit
        # configuration and its predictions on the discovery rows (a refit must reproduce them to 1e-9)
        self.fingerprint = {"relation": relation, "part": part, "candidates": self.candidates, "target": target,
                            "baseline": baseline, "feature_form": feature_form, "offset": round(self.offset, 12),
                            "discovery_prediction_sha256": hashlib.sha256(
                                np.round(self.predict(Z), 9).astype(np.float64).tobytes()).hexdigest()
                            if Z is not None else None}

    @property
    def k(self):
        return len(self.candidates)

    def _raw(self, Z):
        s, f = self.fitted.predictions(np.asarray(Z, float), context=None, baseline=np.zeros((len(Z), 0)))
        return np.asarray(f if self.part == "full" else s, float)

    def predict(self, Z):
        return self._raw(Z) - self.offset

    def gradient(self, Z, eps=1e-3):
        Z = np.asarray(Z, float)
        G = np.zeros_like(Z)
        for j in range(Z.shape[1]):
            E = np.zeros_like(Z)
            E[:, j] = eps
            G[:, j] = (self._raw(Z + E) - self._raw(Z - E)) / (2 * eps)
        return G

    def params_hash(self) -> str:
        return locks.canonical_hash(self.fingerprint)

    def to_json(self) -> dict:
        return {"kind": "relation_of_record", "relation": self.relation, "part": self.part,
                "candidates": self.candidates, "target": self.target, "baseline": self.baseline,
                "offset": self.offset, "fingerprint": self.fingerprint, "params_hash": self.params_hash(),
                "fit": "general_discovery.FittedRelation(model='rbf', rbf_features=24, VALIDATION_ALPHA_GRID, "
                       "patient-grouped inner CV) on all discovery patients, within-patient standardised inputs"}


class RelationV03RFF:
    """Relation of record under ``v03_rff`` (morpheus_encoder.learned_search.relation_v03; Agent M bfc40ed).

    The frozen hypothesis is its variable names + relation constants + commit, re-applied by deterministic refit
    (``fit_baseline`` + ``fit_sets``) on the discovery rows; the refit's parameters (``return_beta=True``: mu, sd,
    bw, beta, blocks) are stored as well, and on freeze ``predict(model)`` must reproduce the refit's own
    predictions (max |difference| <= ``REPRO_TOL``, float64 on the CPU). Both are fingerprinted. ``part`` is the
    joint relation (``blocks=None``) or one proper partition of the set (the B_decomp competitor)."""

    REPRO_TOL = 1e-12

    def __init__(self, candidates, target, baseline, Zs, ys, groups, C, *, blocks=None, commit=None):
        from morpheus_encoder.learned_search import relation_v03 as rv

        self.candidates, self.target, self.baseline, self.blocks = list(candidates), target, baseline, blocks
        self.relation = "v03_rff"
        Zs = np.asarray(Zs, float).reshape(len(ys), -1)
        base = rv.fit_baseline(ys, groups, C, yq=ys, C_q=C)
        V = Zs[:, None, :]
        out = rv.fit_sets(V, base, ys, groups, Vq=V, yq=ys, groups_q=groups, blocks=blocks, device="cpu",
                          return_beta=True)
        self.model = {k: (np.asarray(v).tolist() if k != "blocks" else v) for k, v in out["model"].items()}
        refit_pred = np.asarray(out["pred_q"])[:, 0]
        stored_pred = self.predict(Zs)
        self.repro_max_abs = float(np.max(np.abs(stored_pred - refit_pred)))
        if not self.repro_max_abs <= self.REPRO_TOL:
            raise ValueError(f"stored v03_rff parameters do not reproduce the refit ({self.repro_max_abs:.3g})")
        self.alpha, self.g_sup = float(np.asarray(out["alpha"])[0]), float(np.asarray(out["g_sup"])[0])
        self.spec = {"variable_names": self.candidates, "target": target, "baseline": baseline,
                     "transforms": [c.split(":", 1)[0] for c in self.candidates], "relation": "v03_rff",
                     "blocks": blocks, "constants": {"D_RFF": rv.D_RFF, "RFF_SEED": rv.RFF_SEED,
                                                     "ALPHAS": list(rv.ALPHAS), "BASE_ALPHAS": list(rv.BASE_ALPHAS)},
                     "commit": commit}

    @property
    def k(self):
        return len(self.candidates)

    def predict(self, Z):
        from morpheus_encoder.learned_search import relation_v03 as rv

        model = {k: (np.asarray(v) if k != "blocks" else v) for k, v in self.model.items()}
        Z = np.asarray(Z, float).reshape(len(Z), -1)
        return np.asarray(rv.predict(model, Z[:, None, :], device="cpu"))[:, 0]

    def gradient(self, Z, eps=1e-4):
        Z = np.asarray(Z, float)
        G = np.zeros_like(Z)
        for j in range(Z.shape[1]):
            E = np.zeros_like(Z)
            E[:, j] = eps
            G[:, j] = (self.predict(Z + E) - self.predict(Z - E)) / (2 * eps)
        return G

    def fingerprints(self) -> dict:
        return {"spec_sha256": locks.canonical_hash(self.spec),
                "params_sha256": locks.canonical_hash({k: self.model[k] for k in sorted(self.model)})}

    def params_hash(self) -> str:
        return locks.canonical_hash(self.fingerprints())

    def to_json(self) -> dict:
        out = {"kind": "relation_of_record", **self.spec, "model": self.model, "alpha": self.alpha,
               "g_sup": self.g_sup, "repro_max_abs": self.repro_max_abs, "fingerprints": self.fingerprints()}
        if self.blocks is not None:
            out["best_partition_rule"] = ("frozen record: the proper partition with the largest support (LOPO) gain; "
                                          "the receipt's best_decomposition is chosen on held-out gain inside each "
                                          "bootstrap draw, so the two may name different partitions")
        return out


def _freeze_v03(fid, candidates, target, Zs, ys, dep, pat, d, commit):
    from morpheus_encoder.learned_search import relation_v03 as rv

    rel_n = RelationV03RFF(candidates, target, "nuis", Zs, ys, pat, dep, commit=commit)
    rel_d = None
    if rel_n.k >= 2:
        best = None
        for part in proper_partitions(rel_n.k):
            r = RelationV03RFF(candidates, target, "nuis", Zs, ys, pat, dep, blocks=[list(b) for b in part],
                               commit=commit)
            if best is None or r.g_sup > best.g_sup:
                best = r
        rel_d = best
    rel_k = None
    if d.get("known") is not None:
        K = standardise_within(d["known"], pat)[0].reshape(len(pat), -1)
        rel_k = RelationV03RFF(candidates, target, "known", Zs, ys, pat, np.column_stack([dep, K]), commit=commit)
    base = rv.fit_baseline(ys, pat, dep)
    return rel_n, rel_k, rel_d, np.asarray(base.r_s)


def freeze_of_record(fid, candidates, target, d, relation, feature_form="linear_subspace", meta=None, commit=None):
    from .replication_v03 import FrozenFinding

    pat = np.asarray(d["patient"]).astype(str)
    Zs = standardise_within(d["Z"], pat)[0].reshape(len(pat), -1)
    ys = standardise_within(d["y"], pat)[0].ravel()
    dep = standardise_within(np.log1p(d["depth"]), pat)[0].reshape(-1, 1)
    if relation == "v03_rff":
        rel_n, rel_k, rel_d, rn = _freeze_v03(fid, candidates, target, Zs, ys, dep, pat, d, commit)
        dom = dominant_term(rel_n, Zs, rn)
        return FrozenFinding(fid, rel_n, rel_k, rel_d, dom, dict(meta or {})), {"Zs": Zs, "ys": ys, "r_nuis": rn,
                                                                               "patient": pat}
    rn = ys - fit_baseline(dep, ys, pat, ["log1p_depth"]).predict(dep)
    rel_n = RelationOfRecord(candidates, target, "nuis", relation, Zs, rn, pat, feature_form)
    rel_d = None
    if rel_n.k >= 2:
        rel_d = RelationOfRecord(candidates, target, "nuis", relation, Zs, rn, pat, feature_form, part="competitor",
                                 fitted=rel_n.fitted)
    rel_k = None
    if d.get("known") is not None:
        K = standardise_within(d["known"], pat)[0].reshape(len(pat), -1)
        K = np.column_stack([dep, K])
        rk = ys - fit_baseline(K, ys, pat, ["log1p_depth"] + list(d["known_names"])).predict(K)
        rel_k = RelationOfRecord(candidates, target, "known", relation, Zs, rk, pat, feature_form)
    dom = dominant_term(rel_n, Zs, rn)
    return FrozenFinding(fid, rel_n, rel_k, rel_d, dom, dict(meta or {})), {"Zs": Zs, "ys": ys, "r_nuis": rn,
                                                                           "patient": pat}


def known_columns_v03(study: str, target: str, coh) -> tuple[np.ndarray, list, dict]:
    """B_known covariates through the single definition the runner, the harness and the training tasks use
    (relation_v03.known_covariates / known_exclusions, Agent M b7879b1): every hest_multi programme score with
    its name and scoring panel + the three composition variables (episodes_v03.composition via the runner's
    _composition_of_rows); the target's own score and every gene-set-overlapping programme (or, for a gene target,
    every programme containing the gene) are removed by relation_v03.known_covariates, not by a list kept here.
    Depth is added by the caller. Rows are aligned to ``coh`` by (sample_id, barcode)."""
    cache_key = (str(study), str(target), tuple(zip(map(str, coh.raw.sample_id), map(str, coh.raw.barcode))))
    if cache_key in _KNOWN_COLUMNS_CACHE:
        return _KNOWN_COLUMNS_CACHE[cache_key]
    from morpheus.tier2.substrates import hest_multi as hm
    from morpheus_encoder.learned_search import relation_v03 as rv
    from morpheus_encoder.scripts.run_general_discovery_packet import _composition_of_rows

    spec = hm.study_specs()[study]
    patients = sorted(set(map(str, spec.patient_of.values())))
    loader = "concept:0" if str(target).startswith("morph:k") else target   # as the runner loads a morph target
    ep = hm.study_episode(spec, target=loader, support_patients=patients[:-1], query_patients=patients[-1:],
                          seed=0, with_morphology=loader != target)
    P = np.asarray(ep.views["programmes"], float)
    comp, source = _composition_of_rows(spec, ep, np.arange(ep.n), study)
    names = list(ep.columns["programmes"]) + [f"composition:{j}" for j in range(comp.shape[1])]
    K_all = np.column_stack([P, comp])
    task = rv.Task(X=np.zeros((ep.n, 1)), y=np.zeros(ep.n), n_support=ep.n, groups=np.asarray(ep.patient),
                   depth=None, known=K_all, composition=None, known_names=names, target_name=target,
                   panel=[str(g) for g in np.asarray(hm.load_study(spec).genes)])
    K, dropped = rv.known_covariates(task)
    kept_names = [n for n in names if n not in set(dropped)]
    idx = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(ep.sample_id, ep.barcode))}
    rows = np.array([idx.get((str(a), str(b)), -1) for a, b in zip(coh.raw.sample_id, coh.raw.barcode)])
    if (rows < 0).any():
        raise ValueError(f"{(rows < 0).sum()} discovery rows missing from the hest_multi episode")
    result = (K[rows], kept_names, {"excluded_by_relation_v03_known_covariates": sorted(dropped),
                                    "composition_source": source})
    _KNOWN_COLUMNS_CACHE[cache_key] = result
    return result


# ----------------------------------------------------------------------------- run directory
def read_run(run_dir) -> dict:
    run = Path(run_dir)
    man = json.loads((run / "campaign_manifest.json").read_text())
    proto = man["protocol"]
    funnel = json.loads((run / "funnel_aggregate.json").read_text())
    ledger = run / proto["ledger"]["filename"]
    events = {}
    for ln in ledger.read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            events[e["candidate_id"]] = e
    for row in funnel["table"]:
        row["ledger_stage"] = events.get(row["candidate_id"], {}).get("stage", row.get("stage"))
    return {"manifest": man, "protocol": proto, "funnel": funnel, "events": events,
            "protocol_sha256": man.get("protocol_sha256")}


def internal_set(run: dict) -> list[dict]:
    return [r for r in run["funnel"]["table"] if r["ledger_stage"] in ("internal", "external", "orthogonal",
                                                                        "actionable")]


def is_renal(run: dict) -> bool:
    return "renal" in run["protocol"]["campaign_id"]


def write_prefix(root, row, run, rel_hash) -> None:
    fid = row["candidate_id"]
    ident = {"finding_id": fid, "cohort": run["protocol"]["campaign_id"], "target": row["target"],
             "candidates": row["variable_names"], "relation_params_hash": rel_hash,
             "registry_hash": run["protocol_sha256"]}
    locks.write_lock(root, fid, "registry_freeze", inputs={"funnel_row": row, "protocol_sha256": run["protocol_sha256"]},
                     declared_analysis={"source": "registered run (read only)"})
    locks.write_record(root, fid, "registry_freeze", ident)
    val = {"gain_nuis": {"pooled_mean_gain": row.get("mean_replication_gain"), "passing_folds": row.get("passing_folds")},
           "gain_known": row.get("explanation_decomposition") or locks.not_assessed("decomposition ladder output not in the funnel row"),
           "gain_decomp": {"higher_order": row.get("higher_order"), "n_frozen_higher_order": row.get("n_higher_order_folds")},
           "discovery_labels": [row["ledger_stage"]] + (["higher-order"] if row.get("higher_order") else []),
           "null_yield_at_rank": row.get("null_yield") or locks.not_assessed("harness null yield not in the funnel row"),
           # four baseline passes (reported, never gated); the paper's object count is passes_B_known
           "baseline_passes": {k: row.get(k) for k in ("passes_B_nuis", "passes_B_known", "passes_B_decomp",
                                                       "passes_B_known_and_B_decomp", "known_folds", "decomp_folds")}
           if "passes_B_known" in row else locks.not_assessed("receipts carry no gain_B_known (pre-v03_rff relation)")}
    locks.write_lock(root, fid, "internal_validation", inputs={"q": row.get("q_value"), "worst_null_p": row.get("worst_null_p")},
                     declared_analysis={"rule": run["protocol"]["discovery"]["advance_rule"]["internal"]})
    locks.write_record(root, fid, "internal_validation", val)
    card = locks.StateCard(root, fid)
    card.update("identity", ident)
    card.update("validation", val)


def stage_replication(run_dir, out, catalogue=None, families=None, strict_known=False,
                      include_proposed=False, external_cellvit_root=None, candidate_ids=None) -> dict:
    from . import bknown
    from .discovery import discovery_arrays
    from .replication_v03 import run_family

    run = read_run(run_dir)
    # Proposed-only rows have not passed the campaign's internal gate.  They
    # may be scored externally to resolve every frozen hypothesis, but their
    # receipts are explicitly exploratory and cannot promote their evidence
    # tier.  This prevents an external association from laundering a failed
    # internal confirmation into an "externally confirmed discovery".
    all_rows = internal_set(run)
    proposed_rows = [r for r in run["funnel"]["table"] if r["ledger_stage"] == "proposed"]
    scoped_rows = all_rows + (proposed_rows if include_proposed else [])
    rows = list(scoped_rows)
    if families:
        allowed = {str(x) for x in families}
        rows = [r for r in rows if str(r.get("family")) in allowed]
    if strict_known:
        rows = [r for r in rows if bool(r.get("passes_B_known"))]
    if candidate_ids:
        requested = {str(x) for x in candidate_ids}
        known = {r["candidate_id"] for r in rows}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"requested candidates are not in the selected frozen scope: {missing[:5]}")
        rows = [r for r in rows if r["candidate_id"] in requested]
    selected_ids = {r["candidate_id"] for r in rows}
    deferred = {r["candidate_id"]: "deferred by explicit external-confirmation routing filter"
                for r in scoped_rows if r["candidate_id"] not in selected_ids}
    root = f"{out}/cards"
    relation = str(run["protocol"]["discovery"].get("relation", "v07_ho"))
    cat = catalogue or bknown.covariate_sets()
    residual_ctx = None
    if external_cellvit_root:
        from .external_cellvit_v08 import v08_external_cellvit_context
        residual_ctx = v08_external_cellvit_context(external_cellvit_root)
    findings, skipped = [], {}
    for row in rows:
        try:
            from .values import OPERATOR_VIEWS
            fam_views = (run["protocol"]["discovery"].get("relationship_families", {}).get(str(row.get("family")), {})
                         .get("candidate_views", []))
            op_views = [v for v in fam_views if v in OPERATOR_VIEWS]
            d = discovery_arrays(row["variable_names"], row["target"], cat, residual_ctx, operator_views=op_views)
            if relation == "v03_rff":   # B_known as the runner fits it (no catalogue, no full programme panel)
                study = next((c.get("study_key") for c in run["protocol"]["cohorts"] if c.get("study_key")),
                             "meylan2022_ccrcc")
                # v08 may predict a spatial operator of a programme (for
                # example het:programme:mp:IG).  B_known must exclude the
                # underlying molecular programme, while the response remains
                # the operator.  The older known-column builder only accepts
                # ordinary target names, hence this explicit canonicalisation.
                from .values import parse_candidate
                known_target = row["target"]
                parsed_target = parse_candidate(known_target)
                if parsed_target.get("kind") == "operator":
                    base = parsed_target["base"]
                    if base["kind"] == "programme":
                        known_target = f"programme:{base['set']}"
                    elif base["kind"] == "gene":
                        known_target = f"gene:{base['genes_key']}"
                    else:
                        raise ValueError(f"operator target has unsupported base: {known_target}")
                if parsed_target.get("kind") == "residual":
                    # A frozen residual component is not an entry in the
                    # programme catalogue used by relation_v03.  There is no
                    # target-identical programme score to remove, so B_known
                    # remains explicitly not assessed rather than attempting
                    # an invalid catalogue lookup.
                    d["known"], d["known_names"] = None, []
                    kinfo = {"status": "not_assessed", "reason": "residual target has no catalogue B_known analogue"}
                else:
                    d["known"], d["known_names"], kinfo = known_columns_v03(study, known_target, d["cohort"])
                if known_target != row["target"]:
                    kinfo = {**kinfo, "operator_target_canonicalised_from": row["target"],
                             "known_target": known_target}
                d["known_info"] = kinfo
            from .discovery import apply_row_mask
            d = apply_row_mask(d)
        except Exception as e:      # e.g. morphology-concept constituents: not testable externally
            skipped[row["candidate_id"]] = f"{type(e).__name__}: {e}"
            continue
        try:
            ff, _ = freeze_of_record(row["candidate_id"], row["variable_names"], row["target"], d, relation,
                                     row.get("feature_form", "linear_subspace"),
                                     {"finding_hash": row["candidate_id"]},
                                     commit=run["protocol"]["release"].get("relation_commit"))
        except Exception as e:
            # A stored relation that cannot reproduce from its frozen source
            # data is not eligible for external confirmation.  Record that
            # result per candidate and continue the declared testing family;
            # never let one malformed receipt truncate the other 199.
            skipped[row["candidate_id"]] = f"{type(e).__name__}: frozen relation is irreproducible: {e}"
            try:
                del d
            except UnboundLocalError:
                pass
            continue
        write_prefix(root, row, run, ff.rel_nuis.params_hash())
        # Proposed candidates are retained as an explicitly exploratory
        # external-testing scope in the batch plan.  ``external_test_scope``
        # is deliberately not a locks stage: creating a pseudo-stage here
        # would violate the immutable receipt schema.
        Path(f"{out}/frozen_relations").mkdir(parents=True, exist_ok=True)
        Path(f"{out}/frozen_relations/{locks._safe(row['candidate_id'])}.json").write_text(
            json.dumps(ff.rel_nuis.to_json(), indent=1))
        findings.append(ff)
        # Free the discovery-sized tensors before freezing the next relation.
        # A campaign may have hundreds of hypotheses; retaining allocator
        # arenas from each deterministic refit can otherwise exceed the
        # Lambda job memory limit although the frozen relation itself is tiny.
        del d
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    for cid, why in skipped.items():
        row = next(r for r in rows if r["candidate_id"] == cid)
        write_prefix(root, row, run, "not frozen (not testable externally)")
        locks.write_lock(root, cid, "external_replication", inputs={}, declared_analysis={"rule": "v03"})
        locks.write_record(root, cid, "external_replication", {"externally_replicated": False,
                                                               "status": "not testable externally", "reason": why})
    if not is_renal(run):
        for ff in findings:
            locks.write_lock(root, ff.finding_id, "external_replication", inputs={},
                             declared_analysis={"rule": "Section 2.1: no independent public spatial cohort"})
            locks.write_record(root, ff.finding_id, "external_replication",
                               {"externally_replicated": False, "status": "external spatial replication unavailable"})
        return {"n_internal_total": len(all_rows), "n_proposed_included": sum(r["ledger_stage"] == "proposed" for r in rows),
                "n_selected": len(rows), "n_frozen": len(findings),
                "deferred": deferred, "skipped": skipped, "external": "unavailable"}
    res = run_family(findings, root, ["li2022_ccrcc", "gse250163", "zenodo14620362"], catalogue=cat,
                     residual_ctx=residual_ctx)
    summary = {"n_internal_total": len(all_rows), "n_proposed_included": sum(r["ledger_stage"] == "proposed" for r in rows),
               "n_selected": len(rows), "n_frozen": len(findings),
               "selection": {"families": sorted(families) if families else None, "strict_known": bool(strict_known),
                             "include_proposed": bool(include_proposed)},
               "deferred": deferred, "skipped": skipped,
               "replicated": [f for f, r in res.items() if r["externally_replicated"]]}
    Path(f"{out}/replication_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["replication"], default="replication")
    ap.add_argument("--families", default=None,
                    help="Comma-separated frozen families to confirm now; omitted means all internal rows.")
    ap.add_argument("--strict-known", action="store_true",
                    help="Only candidates passing the registered strict B_known gate.")
    ap.add_argument("--include-proposed", action="store_true",
                    help="Also externally score proposed-only hypotheses; records them as exploratory and never permits tier promotion.")
    ap.add_argument("--external-cellvit-root", default=None,
                    help="Frozen external CellViT adapter root for residual and compartment candidates.")
    ap.add_argument("--candidate-ids", default=None,
                    help="Comma-separated explicit frozen candidate IDs (for resumable resource-bounded batches).")
    a = ap.parse_args()
    if "runs/prospective" in os.path.abspath(a.out):
        raise SystemExit("refusing to write under runs/prospective")
    families = [x for x in (a.families or "").split(",") if x]
    candidate_ids = [x for x in (a.candidate_ids or "").split(",") if x]
    print(json.dumps(stage_replication(a.run_dir, a.out, families=families or None,
                                       strict_known=a.strict_known,
                                       include_proposed=a.include_proposed,
                                       external_cellvit_root=a.external_cellvit_root,
                                       candidate_ids=candidate_ids or None), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
