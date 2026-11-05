"""Candidate ledger (LOCKIN Sec. 11): a single append-only table format, one
JSON-lines file per campaign, recording every attempted candidate through
N_proposed -> N_internal -> N_external -> N_orthogonal -> N_actionable so
the yield funnel stays interpretable and no candidate can be quietly
re-defined or dropped after external validation begins.

Format: one JSON object per line (JSONL), appended, never rewritten in
place. Each line is one EVENT: a candidate reaching (or being tested at) one
stage. A candidate's full history is the ordered set of lines sharing its
`candidate_id`. Nothing is ever deleted; a candidate that stops is recorded
with a `stop_reason`, not removed.

Enforcement: once a candidate has ANY event at stage "external" or later
(orthogonal, actionable), every SUBSEQUENT event for that `candidate_id`
must carry the IDENTICAL frozen-object hash (sha256 of the canonical
(operator, operands, parameters, context, lag, neighbourhood) tuple) as the
one recorded at first external-stage entry. `LedgerError` is raised
otherwise -- this is the mechanism, not a policy note, that LOCKIN Sec. 11's
"never redefine a candidate after external validation begins" is checked
against, not merely stated.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

STAGES = ("proposed", "internal", "external", "orthogonal", "actionable")
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}
FROZEN_FROM_STAGE = "external"   # object definition locks at/after this stage


class LedgerError(Exception):
    pass


def object_hash(operator, operands, parameters=None, context=None, lag=None,
               neighbourhood=None):
    """Canonical sha256 of a declared object's identity -- the same six
    fields every dbench Hypothesis/truth object already carries (operands,
    operator, parameters, context, lag, neighbourhood), so a ledger entry's
    object is directly comparable to (or copy-pasted from) a dbench
    TruthSet/Hypothesis without translation."""
    canon = {
        "operator": str(operator),
        "operands": sorted(str(o) for o in (operands or ())),
        "parameters": _canon(parameters or {}),
        "context": _canon(context or {}),
        "lag": lag,
        "neighbourhood": _canon(neighbourhood) if neighbourhood is not None
                        else None,
    }
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _canon(x):
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return x


@dataclass
class CandidateLedger:
    """Append-only JSONL ledger for one campaign."""
    path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "a").close()

    def _history(self, candidate_id):
        out = []
        if not os.path.exists(self.path):
            return out
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("candidate_id") == candidate_id:
                    out.append(rec)
        return out

    def record(self, candidate_id, stage, operator, operands,
              parameters=None, context=None, lag=None, neighbourhood=None,
              data_ref=None, data_sha256=None, evidence=None,
              stop_reason=None, notes=""):
        """Append one event. Raises LedgerError if this candidate is already
        frozen (has a prior event at stage >= "external") and the object
        hash here does not match the hash frozen there."""
        if stage not in STAGE_RANK:
            raise LedgerError("unknown stage %r; must be one of %r"
                             % (stage, STAGES))
        h = object_hash(operator, operands, parameters, context, lag,
                        neighbourhood)
        hist = self._history(candidate_id)
        frozen = [r for r in hist
                 if STAGE_RANK[r["stage"]] >= STAGE_RANK[FROZEN_FROM_STAGE]]
        if frozen:
            frozen_hash = frozen[0]["object_hash"]
            if h != frozen_hash:
                raise LedgerError(
                    "candidate %r was frozen at stage %r with object_hash "
                    "%s; this record's object_hash %s differs -- a "
                    "candidate's definition may not change after external "
                    "validation begins (LOCKIN Sec. 11)"
                    % (candidate_id, frozen[0]["stage"], frozen_hash, h))
        rec = {
            "candidate_id": candidate_id, "stage": stage,
            "object_hash": h,
            "object": {"operator": str(operator),
                      "operands": sorted(str(o) for o in (operands or ())),
                      "parameters": _canon(parameters or {}),
                      "context": _canon(context or {}), "lag": lag,
                      "neighbourhood": _canon(neighbourhood)
                      if neighbourhood is not None else None},
            "data_ref": data_ref, "data_sha256": data_sha256,
            "evidence": evidence or {}, "stop_reason": stop_reason,
            "notes": notes, "recorded_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def all_events(self):
        out = []
        if not os.path.exists(self.path):
            return out
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def funnel(self):
        """N_proposed -> N_internal -> N_external -> N_orthogonal ->
        N_actionable: each candidate counted once, at the HIGHEST stage it
        ever reached (so the funnel is monotonically non-increasing by
        construction, the way the coordinator's own N_proposed -> ... ->
        N_actionable chain is defined)."""
        best = {}
        for rec in self.all_events():
            cid = rec["candidate_id"]
            r = STAGE_RANK[rec["stage"]]
            if cid not in best or r > STAGE_RANK[best[cid]]:
                best[cid] = rec["stage"]
        counts = {s: 0 for s in STAGES}
        for stage in best.values():
            for s in STAGES[:STAGE_RANK[stage] + 1]:
                counts[s] += 1
        return {
            "N_proposed": counts["proposed"], "N_internal": counts["internal"],
            "N_external": counts["external"],
            "N_orthogonal": counts["orthogonal"],
            "N_actionable": counts["actionable"],
            "n_candidates": len(best),
            "by_candidate": best,
        }

    def print_summary(self):
        f = self.funnel()
        print("Candidate ledger: %s" % self.path)
        print("  N_proposed   -> %4d" % f["N_proposed"])
        print("  N_internal   -> %4d" % f["N_internal"])
        print("  N_external   -> %4d" % f["N_external"])
        print("  N_orthogonal -> %4d" % f["N_orthogonal"])
        print("  N_actionable -> %4d" % f["N_actionable"])
        return f


#: `feature_form` (v1 receipts) to the D2/D3 flexible-relation operator
#: names (v2 receipts may already carry `operator` directly).
_FEATURE_FORM_TO_OPERATOR = {
    "product": "flexible_product",
    "linear_subspace": "flexible_subspace",
}


def _operator_and_operands_from_row(row):
    op = row.get("operator")
    if op not in ("flexible_product", "flexible_subspace"):
        op = _FEATURE_FORM_TO_OPERATOR.get(row.get("feature_form"), op or "flexible_subspace")
    operands = row.get("gene_symbols") or row.get("variables") or row.get("operands") or ()
    return str(op), tuple(str(o) for o in operands)


def record_from_receipt(ledger, candidate_id, stage, row, evidence=None, *,
                        data_ref=None, data_sha256=None, stop_reason=None,
                        notes=""):
    """Append one ledger event built directly from a general-discovery
    receipt row (C3, plan `calm-mixing-candy.md`).

    This is a convenience constructor, not a new record format: it derives
    `operator`/`operands`/`parameters`/`context`/`lag` from `row` and calls
    the SAME `CandidateLedger.record` every other caller uses, so the
    immutable-hash-after-`external` enforcement in `record` applies exactly
    as before -- redefining the candidate's object after external validation
    begins still raises `LedgerError` here, because this function is a thin
    wrapper around `record`, not a parallel append path.

    The evidence block is assembled from whichever of the following `row`
    (or an explicit `evidence=` override, which always wins on key clashes)
    actually carries, and nothing is filled in where the source is absent:
    `fold_gains`, `null_p` (per null kind, from `row["nulls"]` or
    `row["null_p_value"]`), `q` (`row["q_value"]`/`row["q"]`),
    `baseline_width`, the representation checkpoint/arm
    (`row["representation"]["checkpoint"/"arm"]`), the proposal mechanism
    (`row["proposal_family"]`), the search budget (`row["search_budget"]`),
    and the dataset version (`row["dataset_version"]` or
    `row["episode_identity"]["dataset_version"]`).
    """
    row = dict(row or {})
    operator, operands = _operator_and_operands_from_row(row)
    parameters = dict(row.get("parameters") or {})
    if row.get("baseline_width") is not None:
        parameters.setdefault("baseline_width", row["baseline_width"])
    context = dict(row.get("context") or {})
    lag = row.get("lag")
    neighbourhood = row.get("neighbourhood")

    nulls = row.get("nulls")
    null_p = None
    if isinstance(nulls, dict) and nulls:
        null_p = {str(k): (v.get("p") if isinstance(v, dict) else v)
                 for k, v in nulls.items()}
    elif row.get("null_p_value") is not None:
        null_p = {"response_permutation": row["null_p_value"]}

    representation = row.get("representation") if isinstance(row.get("representation"), dict) else {}

    built_evidence = {}
    if row.get("fold_gains") is not None:
        built_evidence["fold_gains"] = row["fold_gains"]
    if null_p is not None:
        built_evidence["null_p"] = null_p
    q = row.get("q_value", row.get("q"))
    if q is not None:
        built_evidence["q"] = q
    if row.get("baseline_width") is not None:
        built_evidence["baseline_width"] = row["baseline_width"]
    if representation.get("checkpoint") is not None:
        built_evidence["representation_checkpoint"] = representation["checkpoint"]
    if representation.get("arm") is not None:
        built_evidence["representation_arm"] = representation["arm"]
    if row.get("proposal_family") is not None:
        built_evidence["proposal_mechanism"] = row["proposal_family"]
    if row.get("search_budget") is not None:
        built_evidence["search_budget"] = row["search_budget"]
    dataset_version = row.get("dataset_version")
    if dataset_version is None and isinstance(row.get("episode_identity"), dict):
        dataset_version = row["episode_identity"].get("dataset_version")
    if dataset_version is not None:
        built_evidence["dataset_version"] = dataset_version

    if evidence:
        built_evidence.update(evidence)

    return ledger.record(
        candidate_id, stage, operator, operands, parameters=parameters,
        context=context, lag=lag, neighbourhood=neighbourhood,
        data_ref=data_ref if data_ref is not None else row.get("data_ref"),
        data_sha256=data_sha256 if data_sha256 is not None else row.get("data_sha256"),
        evidence=built_evidence, stop_reason=stop_reason, notes=notes)


#: E3 / discoadd "Build a discovery ledger rather than loose result files":
#: every field the plan names an immutable receipt as carrying. Order
#: matches the discoadd bullet's own order. `novelty_search_date/queries`
#: is split into two explicit fields rather than bundled, so each is
#: independently representable (or independently absent).
LEDGER_EVIDENCE_FIELDS = (
    "dataset_version",
    "representation_checkpoint",
    "proposal_mechanism",
    "candidate_definition",
    "search_budget",
    "proposal_evidence",
    "frozen_candidate_hash",
    "validation_results",
    "null_results",
    "external_datasets",
    "novelty_search_date",
    "novelty_search_queries",
    "literature_matches",
    "evidence_tier",
    "characterization_ref",
    "actionability_status",
    "experimental_follow_up",
    "human_involvement",
)


def complete_evidence(evidence=None):
    """Fill every `LEDGER_EVIDENCE_FIELDS` key missing from `evidence` with
    an explicit `None`, so an evidence block can always carry every
    discoadd ledger field even when most of them do not exist yet for a
    freshly proposed candidate.

    Rejects any key that is neither one of `LEDGER_EVIDENCE_FIELDS` nor
    prefixed `x_` (an explicitly namespaced caller extension) -- a typo in
    a field name must fail loudly here rather than silently vanish from
    the ledger. This is a pure dict transform: it does not call
    `CandidateLedger.record` or `record_from_receipt` itself, so the
    append-only/immutability rules those enforce are entirely unaffected;
    a caller passes the result of this function in as `evidence=`.
    """
    evidence = dict(evidence or {})
    unknown = sorted(k for k in evidence
                     if k not in LEDGER_EVIDENCE_FIELDS and not str(k).startswith("x_"))
    if unknown:
        raise ValueError(
            "complete_evidence refuses unknown evidence key(s) %r -- use "
            "one of LEDGER_EVIDENCE_FIELDS %r or prefix a caller extension "
            "with 'x_'" % (unknown, LEDGER_EVIDENCE_FIELDS))
    out = {k: evidence.get(k) for k in LEDGER_EVIDENCE_FIELDS}
    for k, v in evidence.items():
        if str(k).startswith("x_"):
            out[k] = v
    return out


def summarize(path):
    """One-command summary entry point: `python -m dbench.ledger <path>`."""
    return CandidateLedger(path).print_summary()


if __name__ == "__main__":
    import sys
    summarize(sys.argv[1])
