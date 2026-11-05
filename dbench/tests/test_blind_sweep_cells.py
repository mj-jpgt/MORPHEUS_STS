"""Cross-fold chunking in dbench/scripts/blind_sweep_cells.py: the runner validates at most
proposals_per_target fixed hypotheses per call, so requests are split into chunks."""
import json
import math

from dbench.scripts import blind_sweep_cells as bsc


def _setup(tmp_path, n_hyps, size=32, target="programme:mp:Interferon/MHC-II (I)"):
    run = tmp_path / "run"
    run.mkdir()
    (run / "campaign_manifest.json").write_text(json.dumps({"protocol": {"discovery": {"proposals_per_target": size}}}))
    hyps = [{"variable_names": [f"v{i}"], "id": i} for i in range(n_hyps)]
    sf = tmp_path / "frozen.json"
    sf.write_text(json.dumps(hyps))
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"cells": [{"family": "fam", "fold": 2, "target": target, "score_frozen": str(sf)}]}))
    return run, req, hyps, target


def test_chunks_cover_every_hypothesis_once(tmp_path):
    for n in (1, 31, 32, 33, 88, 115):
        sub = tmp_path / f"n{n}"
        sub.mkdir()
        run, req, hyps, _ = _setup(sub, n)
        lines = bsc.crossfold(str(req), str(run))
        assert len(lines) == math.ceil(n / 32)
        seen = []
        for ln in lines:
            fam, fold, tgt, chunk, dst = ln.split(bsc.SEP)
            part = json.load(open(chunk))
            assert 1 <= len(part) <= 32
            seen += [h["id"] for h in part]
        assert sorted(seen) == list(range(n)) and len(seen) == len(set(seen))


def test_chunk0_keeps_unchunked_receipt_name(tmp_path):
    run, req, _, target = _setup(tmp_path, 70)
    dsts = [ln.split(bsc.SEP)[4] for ln in bsc.crossfold(str(req), str(run))]
    unchunked = f"{run}/receipts/fam/crossfold/f2_{bsc.safe(target)}.json"
    assert dsts[0] == unchunked
    assert dsts[1:] == [unchunked[:-5] + "_part1.json", unchunked[:-5] + "_part2.json"]
    assert len(set(dsts)) == len(dsts)


def test_fold_filter_keeps_chunk_boundaries_and_records(tmp_path):
    run, req, hyps, target = _setup(tmp_path, 70, target="programme:mp:IG")
    for h in hyps:
        h["variable_names"] = [f"nbr18:raw_genes:G{h['id']}"]
    json.dump(hyps, open(json.load(open(req))["cells"][0]["score_frozen"], "w"))
    genes = [f"G{i}" for i in range(70) if i not in (3, 40)] + ["X"]
    (run / "crossfold_requests").mkdir()
    (run / "crossfold_requests" / "fold_columns.json").write_text(json.dumps(
        {"2": {"genes": genes, "programmes": ["mp:IG"]}}))
    rec = run / "receipts" / "fam"
    rec.mkdir(parents=True)
    (rec / f"f2_{bsc.safe(target)}.json").write_text(json.dumps(
        {"episode_info": {"exclusions": {"genes": [genes.index("G5")], "programmes": [0]}}}))
    lines = bsc.crossfold(str(req), str(run))
    assert len(lines) == 3
    ids = [[h["id"] for h in json.load(open(ln.split(bsc.SEP)[3]))] for ln in lines]
    assert ids[0] == [i for i in range(32) if i not in (3, 5)]
    assert ids[1] == [i for i in range(32, 64) if i != 40] and ids[2] == list(range(64, 70))
    ns = json.load(open(run / "crossfold_requests" / "not_scoreable_in_fold.json"))
    assert sorted(p["variable_names"][0] for p in ns["pairs"]) == ["nbr18:raw_genes:G3", "nbr18:raw_genes:G40",
                                                                   "nbr18:raw_genes:G5"]
