"""R22-H188 finqa detail - the arm's finqa read reported against its own ceiling.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R22-H188 DERIVATION-ENHANCED MIX" (2026-08-17 ~17:32), PREDICTION clause: finqa
moves at most within the R22-H182 ceiling of 9 of 20 unsupported responses. The
other 11 are outside this arm by construction - 7 `wrong_operand_selected` need
the QUESTION in the model input, which this arm does not supply, and 4 `other`
are label noise. No movement may be attributed to them.

MEASUREMENT ONLY. Two stages, both reusing banked code:

    score     per-item arena response scores for the two H188 checkpoints,
              through `R21-H179_arena_scores.stage_score` with only DRAWS
              rebound. Its fidelity control stands unchanged: the per-item
              scores must reproduce this arm's own banked per-subset arena AUROC
              to 1e-4 and match its (n, n_sent, n_pairs) fingerprint
    analyse    CPU. The 20 finqa negatives partitioned by the R22-H182 classes,
              with each negative's response score and its share of the subset's
              misordered-pair mass, flagship 6-draw mean against H188 2-draw
              mean. `rank_loss` is lifted VERBATIM from R18-H157_finqa_autopsy.py
              by the R21-H179 `_verbatim` mechanism - not re-implemented

The AUROC unit is the RESPONSE: each H92 sentence is scored as max over the
item's 1,500/750 windows and the response score is the MIN over its sentences.
A NEGATIVE is ranked correctly when it scores BELOW the positives, so a negative
moving DOWN is the arm working.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R22-H188_finqa_detail.py \
          --stage score --draw 1
      uv run python experiments/grounding-semantic/R22-H188_finqa_detail.py \
          --stage analyse
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise SystemExit("GPU PLACEMENT ABORT: CUDA_VISIBLE_DEVICES is unset - set it "
                     "explicitly (0, 1 or 2; empty for --stage analyse)")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R22-H188_finqa_detail.json"
AUTOPSY = HERE / "R22-H182_finqa_predicate_autopsy.json"

H188_DRAWS = {
    "h188d1": {"ckpt": "R22-H188-arm-draw1",
               "banked": "R22-H188_arm_draw1_windowed_result.json",
               "label": "H188 draw 1 (seed 1188, split-cotangent executor)"},
    "h188d2": {"ckpt": "R22-H188-arm-draw2",
               "banked": "R22-H188_arm_draw2_windowed_result.json",
               "label": "H188 draw 2 (seed 2188, split-cotangent executor)"},
}

FLAGSHIP_CKPTS = ("R18-H150-arm-draw1", "R18-H150-arm-draw2",
                  "R19-H160-arm-draw3", "R19-H160-arm-draw4",
                  "R20-H172-arm-draw5", "R20-H172-arm-draw6")

# The R22-H182 partition of the 20 finqa negatives.
REACHABLE = ("operands_present_direction_wrong", "operands_present_derivation_wrong")
NEEDS_QUESTION = ("wrong_operand_selected",)
LABEL_NOISE = ("other",)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stage_score(draw):
    tag = f"h188d{draw}"
    scores = _mod("h179scores", "R21-H179_arena_scores.py")
    scores.DRAWS = H188_DRAWS
    banked = HERE / H188_DRAWS[tag]["banked"]
    if not banked.exists():
        raise SystemExit(
            f"FIDELITY SOURCE MISSING: {banked.name} - the per-item scores are "
            "checked against this arm's own banked arena read; run "
            "R22-H188_arm_run.py --stage windowed first")
    scores.stage_score(tag)


def stage_analyse():
    scores = _mod("h179scores", "R21-H179_arena_scores.py")
    scores.DRAWS = H188_DRAWS
    rank_loss, _src = scores._verbatim("rank_loss")

    autopsy = json.loads(AUTOPSY.read_text())
    cls = {int(r["item"]): r["class"] for r in autopsy["per_item"] if r["leg"] == "negative"}
    groups = {
        "reachable_9": sorted(i for i, c in cls.items() if c in REACHABLE),
        "needs_question_7": sorted(i for i, c in cls.items() if c in NEEDS_QUESTION),
        "label_noise_4": sorted(i for i, c in cls.items() if c in LABEL_NOISE),
    }
    sizes = {k: len(v) for k, v in groups.items()}
    if sizes != {"reachable_9": 9, "needs_question_7": 7, "label_noise_4": 4}:
        raise SystemExit(f"PARTITION ABORT: the R22-H182 classes give {sizes}, "
                         "not the registered 9 / 7 / 4")

    def load(ckpt):
        z = np.load(HERE / f"R21-H179_arena_scores_{ckpt}.npz", allow_pickle=False)
        return z["resp__finqa"], z["y__finqa"]

    def auroc_loss(y, s):
        """Each negative's UNNORMALISED contribution to 1 - AUROC: the share of
        (positive, negative) pairs it inverts, ties 0.5, divided by n_pos*n_neg.
        Summed over the negatives this IS 1 - AUROC, so unlike the banked
        `rank_loss` share it is free to rise or fall in total."""
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        out = np.zeros(len(y))
        for j in neg:
            out[j] = (np.sum(s[pos] < s[j]) + 0.5 * np.sum(s[pos] == s[j])) / (
                len(pos) * len(neg))
        return out

    legs = {"flagship_k6": list(FLAGSHIP_CKPTS),
            "h188_k2": [H188_DRAWS[t]["ckpt"] for t in H188_DRAWS]}
    per_leg = {}
    for leg, ckpts in legs.items():
        s_all, rl_all, al_all, auc_all, y_ref = [], [], [], [], None
        from sklearn.metrics import roc_auc_score
        for c in ckpts:
            s, y = load(c)
            if y_ref is None:
                y_ref = y
            elif not np.array_equal(y, y_ref):
                raise SystemExit(f"LABEL MISMATCH: {c} finqa labels differ")
            s_all.append(s)
            rl_all.append(rank_loss(y, s))
            al_all.append(auroc_loss(y, s))
            auc_all.append(float(roc_auc_score(y, s)))
        per_leg[leg] = {"ckpts": ckpts, "score": np.mean(s_all, axis=0),
                        "rank_loss": np.mean(rl_all, axis=0),
                        "auroc_loss": np.mean(al_all, axis=0),
                        "auroc_per_draw": [round(a, 4) for a in auc_all],
                        "auroc_mean": round(float(np.mean(auc_all)), 5), "y": y_ref}

    y = per_leg["flagship_k6"]["y"]
    neg_idx = np.where(y == 0)[0]
    if sorted(neg_idx.tolist()) != sorted(cls):
        raise SystemExit("ITEM MAP ABORT: the arena's finqa negatives are not the "
                         "20 items R22-H182 classified")

    fl, h8 = per_leg["flagship_k6"], per_leg["h188_k2"]
    per_item = []
    for i in sorted(cls):
        per_item.append({
            "item": i, "class": cls[i],
            "group": next(g for g, v in groups.items() if i in v),
            "score_flagship_k6": round(float(fl["score"][i]), 5),
            "score_h188_k2": round(float(h8["score"][i]), 5),
            "score_delta": round(float(h8["score"][i] - fl["score"][i]), 5),
            "rank_loss_share_flagship_k6": round(float(fl["rank_loss"][i]), 5),
            "rank_loss_share_h188_k2": round(float(h8["rank_loss"][i]), 5),
            "rank_loss_share_delta": round(float(h8["rank_loss"][i] - fl["rank_loss"][i]), 5),
            "auroc_loss_flagship_k6": round(float(fl["auroc_loss"][i]), 5),
            "auroc_loss_h188_k2": round(float(h8["auroc_loss"][i]), 5),
            "auroc_loss_delta": round(float(h8["auroc_loss"][i] - fl["auroc_loss"][i]), 5),
        })

    per_group = {}
    for g, items in groups.items():
        idx = np.array(items)
        per_group[g] = {
            "items": items, "n": len(items),
            "mean_score_flagship_k6": round(float(fl["score"][idx].mean()), 5),
            "mean_score_h188_k2": round(float(h8["score"][idx].mean()), 5),
            "mean_score_delta": round(float((h8["score"][idx] - fl["score"][idx]).mean()), 5),
            "n_moved_down": int((h8["score"][idx] < fl["score"][idx]).sum()),
            "rank_loss_mass_flagship_k6": round(float(fl["rank_loss"][idx].sum()), 5),
            "rank_loss_mass_h188_k2": round(float(h8["rank_loss"][idx].sum()), 5),
            "rank_loss_mass_delta": round(
                float((h8["rank_loss"][idx] - fl["rank_loss"][idx]).sum()), 5),
            "auroc_loss_flagship_k6": round(float(fl["auroc_loss"][idx].sum()), 5),
            "auroc_loss_h188_k2": round(float(h8["auroc_loss"][idx].sum()), 5),
            "auroc_loss_delta": round(
                float((h8["auroc_loss"][idx] - fl["auroc_loss"][idx]).sum()), 5),
        }

    neg_mass_fl = float(fl["rank_loss"][neg_idx].sum())
    neg_mass_h8 = float(h8["rank_loss"][neg_idx].sum())
    reach = per_group["reachable_9"]
    out_idx = np.array(groups["needs_question_7"] + groups["label_noise_4"])
    outside = {
        "items": groups["needs_question_7"] + groups["label_noise_4"], "n": 11,
        "rank_loss_mass_delta": round(
            per_group["needs_question_7"]["rank_loss_mass_delta"]
            + per_group["label_noise_4"]["rank_loss_mass_delta"], 5),
        "auroc_loss_delta": round(
            per_group["needs_question_7"]["auroc_loss_delta"]
            + per_group["label_noise_4"]["auroc_loss_delta"], 5),
        "mean_score_delta": round(
            float((h8["score"][out_idx] - fl["score"][out_idx]).mean()), 5),
        "n_moved_down": int((h8["score"][out_idx] < fl["score"][out_idx]).sum()),
    }

    payload = {
        "experiment": "R22-H188 finqa detail - the arm's finqa read against the "
                      "R22-H182 reachability ceiling (measurement only)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, block "
                         "'R22-H188 DERIVATION-ENHANCED MIX' (2026-08-17 ~17:32), "
                         "PREDICTION clause"),
        "unit": ("response-level arena score - each H92 sentence scored as max over the "
                 "item's 1500/750 windows, response score = MIN over its sentences; a "
                 "NEGATIVE scoring LOWER is the arm working"),
        "rank_loss_definition": ("R18-H157 `rank_loss`, lifted verbatim: each item's share "
                                 "of the subset's misordered (positive, negative) pairs, "
                                 "ties 0.5, normalised so positives and negatives each "
                                 "carry 0.5 of the total; the whole misordered mass is "
                                 "1 - AUROC and this is its per-item decomposition"),
        "finqa_auroc": {
            "flagship_k6_per_draw": fl["auroc_per_draw"],
            "flagship_k6_mean": fl["auroc_mean"],
            "h188_per_draw": h8["auroc_per_draw"],
            "h188_two_draw_mean": h8["auroc_mean"],
            "delta": round(h8["auroc_mean"] - fl["auroc_mean"], 5),
        },
        "negative_rank_loss_mass": {
            "flagship_k6": round(neg_mass_fl, 5), "h188_k2": round(neg_mass_h8, 5),
            "delta": round(neg_mass_h8 - neg_mass_fl, 5),
            "identity_check": "the 20 negatives carry exactly half the normalised "
                              "misordered mass by construction; each leg's negative "
                              "mass is 0.5 up to rounding",
        },
        "ceiling": {
            "reachable": 9, "needs_question": 7, "label_noise": 4,
            "source": "R22-H182_finqa_predicate_autopsy.json",
            "rule": "no movement may be attributed to the 11 outside the reachable set",
        },
        "per_group": per_group,
        "outside_the_reachable_set": outside,
        "auroc_loss_attribution": {
            "total_delta": round(reach["auroc_loss_delta"] + outside["auroc_loss_delta"], 5),
            "from_the_reachable_9": reach["auroc_loss_delta"],
            "from_the_other_11": outside["auroc_loss_delta"],
            "reading": ("1 - AUROC is exactly the sum of these 20 terms, so the finqa "
                        "AUROC delta is minus total_delta and decomposes additively "
                        "over the negatives; only the reachable-9 term may be "
                        "attributed to this arm's predicate"),
        },
        "per_item": per_item,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
        "written": time.strftime("%F %T"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ("finqa_auroc", "negative_rank_loss_mass", "per_group",
                       "outside_the_reachable_set")}, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("score", "analyse"))
    ap.add_argument("--draw", type=int, default=None, choices=(1, 2))
    args = ap.parse_args()
    if args.stage == "score":
        if args.draw is None:
            raise SystemExit("--stage score needs --draw")
        stage_score(args.draw)
    else:
        stage_analyse()


if __name__ == "__main__":
    main()
