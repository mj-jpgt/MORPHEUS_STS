"""CRC v04 gene-target protocol (Agent W2, 2026-10-23): the gene-target families of crc-spatial-v03, registered at the gene-leak
fix commit 0b9e153. Reads dbench/protocols/crc_spatial_v03_addendum.json (REGISTERED), writes crc_spatial_v04_gene_targets.json.
Usage: python scripts/second_wave/finalize_v04.py"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "dbench" / "protocols"
FIX = "0b9e153"
GENES = ["gene:SLC16A1", "gene:SLC16A3", "gene:LDHA", "gene:LDHB"]

POOLING = {
    "mode": "all_folds",
    "rule": ("the registered pooling step scores every frozen hypothesis in every fold, including the folds in which it was not frozen "
             "(each fold's held-out patients predicted from a fit on that fold's proposal patients, with the frozen variables, transforms and "
             "relation unchanged), so the pooled out-of-fold gain, its patient-bootstrap CI and the pooled nulls cover all patients; the folds in which the "
             "hypothesis was frozen are recorded beside it"),
    "finding_stage": "budget_lifted",
    "budget_lifted_rule": ("every hypothesis group passing the fold rule in >= 3 folds enters the finding stage (no top-50 cut at the finding stage); "
                           "the registered finding rule applies; BH q <= 0.10 within family over ALL entrants; then near-duplicate collapse "
                           "(nbr6 / nbr18 versions of a variable count as the same core name) and the second_catalogue_known / target_leak flags are applied"),
    "ledger_note": "candidate_budget is null: no per-family cut at the finding stage; the ledger freezes every entrant (a group passing the fold rule in >= 3 folds, scored in every scoreable fold by the registered cross-fold stage)",
    "pass_and_multiplicity": "fold pass and the internal rule as v0.5 (advance_rule); BH q <= 0.10 within family over all entrants of the finding stage",
    "implementation": "dbench.prospective_campaign._aggregate_families, native budget-lifted mode (budget_lifted(discovery); candidate_budget null); test dbench/tests/test_budget_lifted_pooling.py reproduces the supplementary v2 counts on the registered Meylan (244 B_known, 11 B_decomp, 1 both) and ependymoma fold receipts; the collapse (Jaccard >= 0.5 single linkage on variable sets, nbr6 / nbr18 versions one core name) and the second_catalogue_known / target_leak flags are the reporting layer of dbench/downstream/scripts/bknown_semantics.py applied to the aggregate; receipts and ledger are read only by the reporting layer; the aggregation-stage code commit is recorded in the driver log",
    "supersedes_for_this_protocol": "the earlier 'collapse then top-50 by B_known' proposal is not used",
}


def main():
    d = json.load(open(P / "crc_spatial_v03_addendum.json"))
    fams = d["discovery"]["relationship_families"]
    keep = {}
    for name in ("metabolic_consumer_epithelial", "metabolic_producer_stromal_immune"):
        f = copy.deepcopy(fams[name])
        f["targets"] = [t for t in f["targets"] if t in GENES]
        assert len(f["targets"]) == 4
        keep[name] = f
    keep["metabolic_contrast_genes"] = copy.deepcopy(fams["metabolic_contrast_genes"])
    assert len(keep["metabolic_contrast_genes"]["targets"]) == 4
    for f in keep.values():
        f["gene_target_leak_control"] = (f"programmes whose gene set contains the target gene are removed from the candidate views (commit {FIX}; "
                                         "recorded per episode as exclusion_report.programmes_containing_target_gene)")
    d["discovery"]["relationship_families"] = keep
    d["discovery"]["pooling"] = POOLING
    d["discovery"]["candidate_budget"] = None
    for f in keep.values():
        f.pop("budget", None)
    d["campaign_id"] = "crc-spatial-v04-gene-targets-20261023"
    d["revision"] = (f"r1 (Agent W2, 2026-10-23): the gene-target families of crc-spatial-v03-20261023 re-registered at {FIX} "
                     "(gene-target candidate-programme exclusion) with the budget-lifted finding stage")
    d["supersedes"] = "none: gene-target subset of crc-spatial-v03-20261023, whose gene-target cells predate the candidate-programme exclusion (run on 5633bdc; the v03 receipts stay as written)"
    d["status"] = f"REGISTERED 2026-10-23 by Agent W2 on the coordinator's instruction (checkpoint release/frozen.pt 13c36faa8cd9, code base {FIX})"
    d["ledger"] = {"filename": "crc_spatial_v04_gene_targets_ledger.jsonl"}
    d["release"]["release_commits"].update({"gene_target_candidate_exclusion": FIX, "code_base_of_this_registration": f"{FIX} (clean worktree) plus launcher commits"})
    d["declared_not_wired"] = {k: v for k, v in d.get("declared_not_wired", {}).items() if k != "nuclear_shape_nuisance_columns" and k != "pathologist_class_in_view"}
    d["notes"] = [f"gene-target subset of crc-spatial-v03: 3 families x 4 targets x 5 folds = 60 cells",
                  "the v03 gene-target cells ran without the candidate-programme exclusion; v03 receipts are kept and not overwritten"]
    d["registration"] = {"never": "no write under runs/prospective by hand",
                         "commands": ["python -m dbench.prospective_campaign --protocol dbench/protocols/crc_spatial_v04_gene_targets.json --run-dir runs/prospective/crc-spatial-v04-gene-targets-20261023",
                                      "CAMPAIGN_PROTOCOL=dbench/protocols/crc_spatial_v04_gene_targets.json bash dbench/scripts/blind_sweep.sh runs/prospective/crc-spatial-v04-gene-targets-20261023 <workers>"]}
    d.pop("deferred_families", None)
    (P / "crc_spatial_v04_gene_targets.json").write_text(json.dumps(d, indent=1) + "\n")
    print(d["campaign_id"], {k: len(v["targets"]) for k, v in keep.items()})


if __name__ == "__main__":
    main()
