"""R15-B1 killgate clause 2 (BLOCKING) == R15-B5 arm 1 - checkpoint specificity.

The B1 derivation-lane schedule is argued from P1's type-uniform reading of the
defect on ONE checkpoint (H105 draw 1). If the defect is instead specific to
that draw, the schedule is fitted to one draw and is void.

Re-reads P1's banked quads and the 2,000 banked H133 triples on two further
frozen checkpoints:

  R9-H105-draw2          the paired clean draw
  R10-H108-lane-draw1    the admitted lane, the campaign's only replicated
                         finqa lever

  KILL the type-uniformity premise and B1's schedule if ANY tier-1 type
  (difference, ratio, pct_change, sum) reads AUROC(b vs c) above 0.60 on
  either checkpoint. LICENSE otherwise.

This clause is shared with P1 section 9's second falsifier and with B5 arm 1;
it is run once and reported under all three.

Frozen weights, held-out TabFact test+validation, zero arena, zero gold.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R15_gate_B1_result.json"
SCORES = HERE / "R15_gate_B1_scores.parquet"

CKPTS = ["R9-H105-draw2", "R10-H108-lane-draw1"]
TIER1 = ["difference", "ratio", "pct_change", "sum"]
BAR = 0.60


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    P1 = _mod("p1", "R15_P1_typeprobe.py")
    H133 = _mod("h133", "R14_H133_probe.py")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    built = P1.build(np.random.default_rng(P1.SEED))
    quads = [q for t in P1.TYPES for q in built[t]]
    tri = H133.build(H133.N_TRIPLES, np.random.default_rng(H133.SEED))
    print(f"rebuilt {len(quads)} P1 quads / {len(tri)} H133 triples", flush=True)

    q_claims, q_evs = [], []
    for tag in ("claim_a", "claim_b", "claim_c", "claim_d"):
        q_claims += [q[tag] for q in quads]
        q_evs += [q["evidence"] for q in quads]
    t_claims = ([t["claim_a"] for t in tri] + [t["claim_b"] for t in tri]
                + [t["claim_c"] for t in tri])
    t_evs = [t["evidence"] for t in tri] * 3

    nq, nt = len(quads), len(tri)
    per_ckpt, banked = {}, {}
    for name in CKPTS:
        tok, trunk, head = C.load_ckpt(name)
        sq = C.score(tok, trunk, head, q_claims, q_evs)
        st = C.score(tok, trunk, head, t_claims, t_evs)
        del trunk, head
        torch.cuda.empty_cache()
        sa, sb, sc, sd = sq[:nq], sq[nq:2 * nq], sq[2 * nq:3 * nq], sq[3 * nq:]
        banked[name] = (sb, sc)
        per_type = {}
        for t in P1.TYPES:
            m = np.array([q["dtype"] == t for q in quads])
            if m.sum() < 30:
                per_type[t] = {"n": int(m.sum()), "note": "under 30 - not adjudicated"}
                continue
            per_type[t] = {
                "n": int(m.sum()),
                "mean_b_correct": round(float(sb[m].mean()), 5),
                "mean_c_wrong_operand": round(float(sc[m].mean()), 5),
                "auroc_b_vs_c": round(C.auroc(sb[m], sc[m]), 4),
                "auroc_b_vs_d": round(C.auroc(sb[m], sd[m]), 4),
                "auroc_a_vs_b": round(C.auroc(sa[m], sb[m]), 4),
            }
        ta, tb, tc = st[:nt], st[nt:2 * nt], st[2 * nt:]
        per_ckpt[name] = {
            "p1_quads": {"n": nq, "per_type": per_type},
            "h133_triples": {
                "n": nt,
                "mean_a_verbatim": round(float(ta.mean()), 5),
                "mean_b_correct": round(float(tb.mean()), 5),
                "mean_c_wrong_operand": round(float(tc.mean()), 5),
                "gap_a_minus_b": round(float(ta.mean() - tb.mean()), 5),
                "auroc_b_vs_c": round(C.auroc(tb, tc), 4),
                "auroc_a_vs_b": round(C.auroc(ta, tb), 4),
            },
        }
        print(name, json.dumps(per_ckpt[name]["h133_triples"]), flush=True)
        for t in TIER1:
            print(f"  {name} {t:12s} AUROC(b|c) {per_type[t]['auroc_b_vs_c']:.4f}", flush=True)

    breaches = [
        {"checkpoint": k, "type": t, "auroc_b_vs_c": per_ckpt[k]["p1_quads"]["per_type"][t]["auroc_b_vs_c"]}
        for k in CKPTS for t in TIER1
        if per_ckpt[k]["p1_quads"]["per_type"][t]["auroc_b_vs_c"] > BAR
    ]
    verdict = "KILL" if breaches else "LICENSE"

    df = pl.DataFrame({
        "dtype": [q["dtype"] for q in quads],
        "table_id": [q["table_id"] for q in quads],
        "v_b": [q["v_b"] for q in quads],
        "v_c": [q["v_c"] for q in quads],
    })
    for name, (sb, sc) in banked.items():
        tag = name.replace("-", "_")
        df = df.with_columns([pl.Series(f"b__{tag}", sb), pl.Series(f"c__{tag}", sc)])
    df.write_parquet(SCORES)

    res = {
        "gate": "R15-B1 killgate clause 2 (BLOCKING) == R15-B5 arm 1 - checkpoint specificity",
        "also_reported_under": ["R15-B1 killgate clause 2", "R15-B5 arm 1",
                                "P1 section 9 falsifier 2"],
        "data": "R15_P1_typeprobe.build() and R14_H133_probe.build() rebuilt at their banked "
                "seeds over held-out TabFact test+validation, table_id-disjoint from every train "
                "split; zero arena, zero gold",
        "implementation_choices": [
            "P1's MAIN quad build (4,139 quads, absence rule intact) is the surface. The 869 "
            "top-up quads (R15_P1_typeprobe_topup.json) are excluded because that build DROPS "
            "the absence rule and covers only count_agg and date_arith, neither of which is a "
            "tier-1 type; the 5,008 figure the synthesis quotes is main+top-up.",
            "AUROC is sklearn roc_auc_score, identical to R7-H59.auc_and_f1's first return.",
        ],
        "tier1_types": TIER1,
        "bar": f"KILL the type-uniformity premise and B1's schedule if any tier-1 type reads "
               f"AUROC(b vs c) > {BAR} on either checkpoint",
        "banked_reference_h105_draw1": {
            "sum": 0.5067, "difference": 0.4994, "ratio": 0.5121, "pct_change": 0.4861,
            "h133_triples_auroc_b_vs_c": 0.4924,
        },
        "checkpoints": per_ckpt,
        "breaches": breaches,
        "verdict": verdict,
        "gates_downstream": "B1 lane build spec (the tier-1 share schedule); the H133 lane build "
                            "must not start until this reads LICENSE",
        "scores": SCORES.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print(f"\nVERDICT: {verdict}   breaches: {breaches}\n-> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
