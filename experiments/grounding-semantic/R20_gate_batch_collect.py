"""R20 licensed free kill-gate batch - collect the eight per-gate artifacts into
the single combined result JSON the registration asks for.

Reads R20_gate_{1,2,3,6,7}.json, R20-G0a_hotpotqa_diffgaps.json,
R20-G0b_composed_probe_baseline.json and R20_fanout_replications.json; writes
R20_gate_batch_result.json. No measurement happens here.

Run:  uv run python experiments/grounding-semantic/R20_gate_batch_collect.py
"""

import json
import pathlib
import time

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R20_gate_batch_result.json"

SOURCES = [
    (1, "H-B derivation compare/direction lane", "R20_gate_1.json",
     ["compare_decidable_share", "compare_decidable_mass", "derivation_rank_loss_mass",
      "constructibility_census", "rule_crosscheck", "l2_corroboration"]),
    (2, "H-C operand/role/sign/period misbind lane", "R20_gate_2.json",
     ["misbind_share_of_fp_record_mass", "misbind_share_of_all_negative_mass",
      "mass_by_class", "edgar_census"]),
    (3, "H-D trained-through numeric canonicalization", "R20_gate_3.json",
     ["results", "verdict_basis"]),
    (4, "hotpotqa G0a composed-claim revival statistic",
     "R20-G0a_hotpotqa_diffgaps.json",
     ["pooled_diff_of_gaps", "ci95", "sign_stability", "per_checkpoint", "sensitivity",
      "verdict_note"]),
    (5, "hotpotqa G0b composed-probe baseline",
     "R20-G0b_composed_probe_baseline.json",
     ["primary_auroc", "auroc", "n_pairs", "primary_leg", "contamination"]),
    (6, "pubmedqa PM-1 deletion-contrast supply", "R20_gate_6.json",
     ["total_triples", "per_corpus", "supply_ceiling_ignoring_localizability",
      "sensitivity_grid_containment_over_margin"]),
    (7, "pubmedqa PM-2 low-containment supported supply", "R20_gate_7.json",
     ["total_rows", "per_corpus", "sensitivity_by_containment_bar"]),
]


def main():
    gates = []
    for num, title, fname, keys in SOURCES:
        d = json.loads((HERE / fname).read_text())
        gates.append({
            "gate": num,
            "name": title,
            "artifact": fname,
            "recipe_summary": d.get("recipe_summary", ""),
            "threshold": d.get("threshold", ""),
            "numbers": {k: d[k] for k in keys if k in d},
            "verdict": d["verdict"],
        })
    rep = json.loads((HERE / "R20_fanout_replications.json").read_text())
    gates.append({
        "gate": 8,
        "name": "banking of the fanout design-pass replications",
        "artifact": "R20_fanout_replications.json",
        "recipe_summary": rep["recipe_summary"],
        "threshold": "banking only - no bar; every design-pass number is reproduced or "
                     "recorded as non-reproducing",
        "numbers": rep["replications"],
        "verdict": "BANKED",
    })
    payload = {
        "experiment": "R20 LICENSED FREE KILL-GATE BATCH - combined result",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, block "
                         "'LICENSED FREE KILL-GATE BATCH' (2026-08-16); per-gate recipes "
                         "in docs/experiments/briefs/R20-fanout-{derivation,hotpotqa-"
                         "composition,pubmedqa-absence}-*.md"),
        "discipline": "measurement only - nothing trained, nothing registered, no verdict "
                      "written to the canonical log",
        "gates": gates,
        "summary": {str(g["gate"]): g["verdict"] for g in gates},
        "timestamp": time.strftime("%F %T"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    for g in gates:
        print(f"  gate {g['gate']}: {g['verdict']:8s} {g['name']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
