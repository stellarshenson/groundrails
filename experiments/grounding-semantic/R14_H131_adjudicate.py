"""R14-H131 Stage 1 (R14-A2) adjudication - the 1024 read against the banked 512
read, per subset, per checkpoint, and against the three pre-registered branches.

CPU only. Reads the four `R14_H131_*_1024.json` artifacts produced by
`R14_H131_reads.sh` and the four banked 512 windowed results.

Pre-registered Stage-1 branches (R14-A2 kill-gate):
  ADOPT the read   techqa/finqa/tatqa average >= +0.010 AND arena mean >= -0.002
                   AND no subset <= -0.020
  LICENSE Stage 2  any exposed subset (techqa, finqa, tatqa) moves <= -0.010
  KILL the line    every subset moves within +/-0.010 on both draws

Stage-1 ADMIT bar (reported alongside): arena mean >= +0.003 on both pair means
with sign agreement on all four checkpoints AND token-dense group >= +0.010
collectively with techqa > 0 on >= 3 of 4 AND no subset <= -0.020 AND gold_full
and RAGTruth EN >= banked - 0.005. Stage-1 KILL bar: mean < -0.002 on either
pair, or techqa negative on >= 3 checkpoints.

delucionqa is barred from adjudicating (deciding-pair truncation 0.0%).
Stage 2 training is BLOCKED by session ruling 14 and is not run.

Run:  uv run python experiments/grounding-semantic/R14_H131_adjudicate.py
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R14_gate_H131_stage1.json"

CKPTS = {
    "h105d1": ("R14_H131_h105d1_1024.json", "R9-H105_windowed_result.json", "h105"),
    "h105d2": ("R14_H131_h105d2_1024.json", "R9-H105_draw2_windowed_result.json", "h105"),
    "h108d1": ("R14_H131_h108d1_1024.json", "R10-H108_lane_draw1_windowed_result.json", "h108"),
    "h108d2": ("R14_H131_h108d2_1024.json", "R10-H108_lane_draw2_windowed_result.json", "h108"),
}
DENSE = ("techqa", "finqa", "tatqa")
BARRED = ("delucionqa",)


def main():
    per_ckpt, missing = {}, []
    for tag, (new_f, old_f, pair) in CKPTS.items():
        p = HERE / new_f
        if not p.exists():
            missing.append(tag)
            continue
        new = json.loads(p.read_text())["per_subset"]
        old = json.loads((HERE / old_f).read_text())["per_subset"]
        rows = {
            s: {"auc_512": round(float(old[s]["auc"]), 4),
                "auc_1024": round(float(new[s]["auc"]), 4),
                "delta": round(float(new[s]["auc"]) - float(old[s]["auc"]), 4)}
            for s in sorted(old)
        }
        adj = {s: v for s, v in rows.items() if s not in BARRED}
        per_ckpt[tag] = {
            "pair": pair, "per_subset": rows,
            "mean_512": round(float(np.mean([v["auc_512"] for v in rows.values()])), 5),
            "mean_1024": round(float(np.mean([v["auc_1024"] for v in rows.values()])), 5),
            "mean_delta": round(float(np.mean([v["delta"] for v in rows.values()])), 5),
            "dense_group_mean_delta": round(float(np.mean([rows[s]["delta"] for s in DENSE])), 5),
            "techqa_delta": rows["techqa"]["delta"],
            "worst_subset": min(adj, key=lambda s: adj[s]["delta"]),
            "worst_subset_delta": min(v["delta"] for v in adj.values()),
            "n_subsets_moving_over_0.010": int(
                sum(abs(v["delta"]) > 0.010 for v in adj.values())
            ),
        }
        print(f"{tag}: mean {per_ckpt[tag]['mean_512']:.5f} -> {per_ckpt[tag]['mean_1024']:.5f} "
              f"({per_ckpt[tag]['mean_delta']:+.5f})  dense "
              f"{per_ckpt[tag]['dense_group_mean_delta']:+.5f}  techqa "
              f"{per_ckpt[tag]['techqa_delta']:+.4f}  worst {per_ckpt[tag]['worst_subset']} "
              f"{per_ckpt[tag]['worst_subset_delta']:+.4f}", flush=True)

    if missing:
        res = {"gate": "R14-H131 Stage 1", "verdict": "INCOMPLETE",
               "missing_checkpoints": missing, "per_checkpoint": per_ckpt}
        RESULT.write_text(json.dumps(res, indent=2))
        print(f"\n  INCOMPLETE - missing {missing}\n  -> {RESULT}")
        return

    have = list(per_ckpt.values())
    pair_means = {
        p: round(float(np.mean([v["mean_delta"] for v in have if v["pair"] == p])), 5)
        for p in ("h105", "h108")
    }

    b_adopt = (
        all(v["dense_group_mean_delta"] >= 0.010 for v in have)
        and all(v["mean_delta"] >= -0.002 for v in have)
        and all(v["worst_subset_delta"] > -0.020 for v in have)
    )
    b_license = any(per_ckpt[t]["per_subset"][s]["delta"] <= -0.010 for t in per_ckpt for s in DENSE)
    b_kill = all(v["n_subsets_moving_over_0.010"] == 0 for v in have)

    # Stage-1 ADMIT / KILL bar (arena-measurable part)
    admit_arena = (
        all(m >= 0.003 for m in pair_means.values())
        and all(v["mean_delta"] > 0 for v in have)
        and all(v["dense_group_mean_delta"] >= 0.010 for v in have)
        and sum(v["techqa_delta"] > 0 for v in have) >= 3
        and all(v["worst_subset_delta"] > -0.020 for v in have)
    )
    kill_bar = (
        any(m < -0.002 for m in pair_means.values())
        or sum(v["techqa_delta"] < 0 for v in have) >= 3
    )

    if b_adopt:
        branch = "ADOPT THE READ"
    elif b_license:
        branch = "LICENSE THE STAGE-2 TRAINING ARM (blocked by session ruling 14 - not run)"
    elif b_kill:
        branch = "KILL THE WHOLE LINE"
    else:
        branch = "NO BRANCH FIRES CLEANLY (movements exceed +/-0.010 but no exposed subset "
        branch += "reaches -0.010 and the ADOPT clause is unmet)"

    res = {
        "gate": "R14-H131 Stage 1 (R14-A2) - frozen-weights windowed read at max_length=1024",
        "read_script": "R14_H131_read1024.py (R8-H101 read; only the tokenizer max_length changes)",
        "checkpoints": list(CKPTS),
        "barred_from_adjudicating": list(BARRED),
        "token_dense_group": list(DENSE),
        "per_checkpoint": per_ckpt,
        "pair_mean_deltas": pair_means,
        "branch_clauses": {
            "ADOPT_read": bool(b_adopt),
            "LICENSE_stage2": bool(b_license),
            "KILL_line": bool(b_kill),
        },
        "stage1_admit_bar_arena_part": bool(admit_arena),
        "stage1_kill_bar": bool(kill_bar),
        "gold_full_and_ragtruth_en_hold_clause": "NOT MEASURED - the ADMIT bar's hold reads at "
            "1024 were not run; they are only load-bearing if the arena clauses admit",
        "verdict": branch,
        "stage2_status": "BLOCKED by session ruling 14 - not run under any branch",
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print(f"\n  pair mean deltas: {pair_means}")
    print(f"  ADOPT={b_adopt}  LICENSE_stage2={b_license}  KILL={b_kill}")
    print(f"\n  VERDICT: {branch}\n  -> {RESULT}")


if __name__ == "__main__":
    main()
