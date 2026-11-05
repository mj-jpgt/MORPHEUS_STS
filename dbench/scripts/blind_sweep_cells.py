"""Helper of dbench/scripts/blind_sweep.sh: the protocol header and the cell lists. Targets may contain spaces
and '/' (e.g. 'programme:mp:Interferon/MHC-II (I)'), so the fields of a cell line are separated by the ASCII
unit separator (0x1f) and output file names are made safe; the receipt itself carries the exact target."""
from __future__ import annotations

import json
import re
import sys

SEP = chr(0x1F)


def safe(t: str) -> str:
    return re.sub(r"[/ ]", "_", t)


def header(proto_path: str) -> None:
    from dbench.prospective_campaign import canonical_hash

    d = json.load(open(proto_path))
    c = d["release"]["learned_search_checkpoint"]
    study = next((x.get("study_key") for x in d["cohorts"] if x.get("study_key")), "meylan2022_ccrcc")
    print(d["status"].split()[0], c["path"].replace(" ", "_"), c["sha256"].replace(" ", "_"), study,
          d["discovery"]["n_folds"], d["discovery"]["fold_seed"], canonical_hash(d))


def cells(proto_path: str, run: str) -> list[str]:
    d = json.load(open(proto_path))["discovery"]
    out = []
    for f in range(int(d["n_folds"])):
        for fam, b in d["relationship_families"].items():
            for t in b.get("targets") or []:
                out.append(SEP.join([fam, str(f), t, "", f"{run}/receipts/{fam}/f{f}_{safe(t)}.json"]))
    return out


# Transform prefixes can be nested in the multiscale grammar (for example
# ``own:nbrS18:raw_genes:CXCL14``).  The admissibility question is always
# about the terminal molecular column, not about which spatial transform led
# to it, so do not maintain a brittle finite prefix list here.
TRANSFORMS = ("own", "nbr6", "nbr18", "contrast")


def fold_admissible(run, family: str, fold: int, target: str, fold_cols: dict | None):
    """Predicate on a variable name: is it a column (or transform of a column) of this fold's design?
    Molecular views only (raw_genes: the fold's support-HVG genes; programmes), minus the cell's target exclusions
    from the fold receipt; other views are passed through. ``None`` when fold_columns.json is absent."""
    from pathlib import Path

    if fold_cols is None:
        return None
    fc = fold_cols[str(fold)]
    rec = Path(run) / "receipts" / family / f"f{fold}_{safe(target)}.json"
    excl = json.loads(rec.read_text()).get("episode_info", {}).get("exclusions", {}) if rec.exists() else {}
    genes = {g for i, g in enumerate(fc["genes"]) if i not in set(excl.get("genes", []))}
    progs = {p for i, p in enumerate(fc["programmes"]) if i not in set(excl.get("programmes", []))}

    def ok(v: str) -> bool:
        # Locate the molecular-view token after an arbitrary sequence of
        # transforms such as own:nbrS6 or nbr18:nbrE18.  This also preserves
        # direct raw_genes/programmes variables unchanged.
        pieces = str(v).split(":")
        molecular_at = next((i for i, piece in enumerate(pieces)
                             if piece in {"raw_genes", "programmes"}), None)
        body = ":".join(pieces[molecular_at:]) if molecular_at is not None else str(v)
        if body.startswith("raw_genes:"):
            return body.split(":", 1)[1] in genes
        if body.startswith("programmes:"):
            return body.split(":", 1)[1] in progs
        return True
    return ok


def crossfold(req_path: str, run: str) -> list[str]:
    """Cross-fold scoring cells, split into chunks of at most ``proposals_per_target`` hypotheses: the runner
    validates at most that many fixed hypotheses per call, so a request with more would be silently truncated.
    Chunk 0 keeps the cell's original receipt name, chunk k >= 1 writes ``..._partk.json``; per-hypothesis
    results do not depend on the chunking (same relation, seeds per hypothesis and per null draw), and the
    aggregation reads every ``crossfold/f*_*.json`` receipt.

    When ``<run>/crossfold_requests/fold_columns.json`` exists (dbench.scripts.crossfold_fold_columns), a hypothesis
    with a variable that is not a column of the scoring fold's design (a gene outside that fold's support-HVG set,
    or a target exclusion) is left out of its chunk, because the runner raises for the whole call; chunk boundaries
    stay those of the unfiltered list (so receipts already written keep their meaning), and every left-out
    hypothesis-fold pair is recorded in ``crossfold_requests/not_scoreable_in_fold.json``."""
    from pathlib import Path

    req = json.load(open(req_path))
    size = int(json.load(open(Path(run) / "campaign_manifest.json"))["protocol"]["discovery"]["proposals_per_target"])
    chunk_dir = Path(run) / "crossfold_requests" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    fc_path = Path(run) / "crossfold_requests" / "fold_columns.json"
    fold_cols = json.loads(fc_path.read_text()) if fc_path.exists() else None
    out, dropped = [], []
    for c in req["cells"]:
        hyps = json.load(open(c["score_frozen"]))
        ok = fold_admissible(run, c["family"], int(c["fold"]), c["target"], fold_cols)
        base = f"{run}/receipts/{c['family']}/crossfold/f{c['fold']}_{safe(c['target'])}"
        for k in range(0, len(hyps), size):
            part = k // size
            chunk = hyps[k:k + size]
            if ok is not None:
                bad = [h for h in chunk if not all(ok(str(v)) for v in h["variable_names"])]
                dropped += [{"family": c["family"], "fold": int(c["fold"]), "target": c["target"],
                             "variable_names": h["variable_names"], "feature_form": h.get("feature_form"),
                             "missing": [v for v in h["variable_names"] if not ok(str(v))]} for h in bad]
                chunk = [h for h in chunk if h not in bad]
            if not chunk:
                continue
            cf = chunk_dir / f"{c['family']}_f{c['fold']}_{safe(c['target'])}_part{part}.json"
            text = json.dumps(chunk, indent=1)
            if not cf.exists() or cf.read_text() != text:
                cf.write_text(text)
            dst = f"{base}.json" if part == 0 else f"{base}_part{part}.json"
            out.append(SEP.join([c["family"], str(c["fold"]), c["target"], str(cf), dst]))
    if fold_cols is not None:
        (Path(run) / "crossfold_requests" / "not_scoreable_in_fold.json").write_text(json.dumps(
            {"rule": "variable not a column of the scoring fold's design (support-HVG genes / target exclusions)",
             "n": len(dropped), "pairs": dropped}, indent=1))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "header":
        header(sys.argv[2])
    else:
        print("\n".join(cells(sys.argv[2], sys.argv[3]) if mode == "cells" else crossfold(sys.argv[2], sys.argv[3])))
