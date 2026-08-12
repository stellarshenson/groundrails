"""R14-H133 PROBE BANK - the arm's pre-registered mechanism reading (report-only).

Every probe here is frozen-weights, held-out TabFact test+validation,
table_id-disjoint from every train split, zero arena, zero gold. Both the arm
draw and the BANKED CLEAN DRAW 1 (`models/R9-H105-mmbert-dann-clean`) are read on
the byte-identical surfaces, so each number carries its paired control.

  P1 quads             per-derivation-type AUROC(b correct vs c wrong-operand);
                       the four tier-1 types (difference, ratio, pct_change, sum)
                       are the A4 mechanism reading, target above 0.60 from a
                       0.4861-0.5121 baseline
  scale/unit           AUROC(b vs c) on P1's scale_unit quads. RULING CHANGE:
                       the VOID clause was REBASED by the author's 2026-08-10
                       data-first ruling to no-further-erosion against the
                       H108-lane baseline of 0.4548, NOT the old absolute 0.80.
                       Both references are printed; nothing is adjudicated here
  verbatim             mean score on the verbatim-cell claim (a) over P1's quads
                       and pooled AUROC(a vs b); banked 0.90507 / 0.9643
  H133 triples         the 2,000 banked triples: verbatim / correct-derivation /
                       wrong-operand means and AUROC(b vs c), AUROC(a vs b)
  B4 relational        controlled `bind_col` (digit-length AND magnitude matched,
                       R15_gate_B4.build), `compare` with per-gap-stratum AUROC,
                       and `bind_row`; registered references 0.70 / 0.65 / 0.95

Reports; does not adjudicate - the coordinator holds the verdict.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     uv run python experiments/grounding-semantic/R14-H133_probes.py --draw 1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent

CONTROL_CKPT = "R9-H105-mmbert-dann-clean"
TIER1 = ("difference", "ratio", "pct_change", "sum")
TIER1_TARGET = 0.60
# Adversarial-review correction (2026-08-10, methodologist F2): the 0.4548 rebase was
# scoped to compositions CARRYING the H108 lane; this arm is clean mix + H133 only and
# its own control reads scale/unit 0.8723, so the original absolute clause is enforceable
# and protective again. Additionally void on erosion vs this arm's own control.
SCALE_UNIT_VOID_ABSOLUTE = 0.80   # original registered clause, RESTORED for this arm
SCALE_UNIT_VOID_EROSION = 0.05    # also void if arm < control - 0.05
SCALE_UNIT_VOID_REBASED = 0.4548  # H108-lane-scoped rebase - NOT applicable to this arm
VERBATIM_MEAN_REF = 0.85
VERBATIM_AUROC_REF = 0.90
B4_REFS = {"bind_col": 0.70, "compare": 0.65, "bind_row": 0.95}
SEED = 20260810
N_COMPARE_PER_STRATUM = 200
PER_TABLE_CAP = 2


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_compare(C, rng):
    """The B4 `compare` arm on held-out tables: an ordering claim against its
    reversal, both operands printed, drawn in equal thirds by relative gap -
    the lane's own construction and stratum rule."""
    caps, tbls, tids = C.held_tabfact()
    order = [int(o) for o in rng.permutation(len(tbls))]
    count = {"lt10pct": 0, "10to100pct": 0, "gt100pct": 0}
    rows = []
    for oi in order:
        if all(v >= N_COMPARE_PER_STRATUM for v in count.values()):
            break
        hdr, body = C.parse(tbls[oi])
        if hdr is None or len(body) < 3:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:C.CHUNK_MAX]
        taken = 0
        for ci in range(1, len(hdr)):
            if taken >= PER_TABLE_CAP:
                break
            vals = [(ri, C.as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) < 3 or len({v for _, v in vals}) < 3:
                continue
            if any((not body[ri][0]) or C.as_num(body[ri][0]) is not None for ri, _ in vals):
                continue
            col = hdr[ci] or f"column {ci}"
            pick = [int(k) for k in rng.permutation(len(vals))]
            (ra, va), (rb, vb) = vals[pick[0]], vals[pick[1]]
            if abs(va - vb) < 1e-9 or min(abs(va), abs(vb)) < 1e-9 or (va < 0) != (vb < 0):
                continue
            gap = abs(va - vb) / min(abs(va), abs(vb))
            stratum = "lt10pct" if gap < 0.10 else ("10to100pct" if gap <= 1.0 else "gt100pct")
            if count[stratum] >= N_COMPARE_PER_STRATUM:
                continue
            (rh, vh), (rl, vl) = ((ra, va), (rb, vb)) if va > vb else ((rb, vb), (ra, va))
            k_hi, k_lo = body[rh][0].strip(), body[rl][0].strip()
            if not k_hi or not k_lo or k_hi == k_lo:
                continue
            if C.fmt(vh) not in ev or C.fmt(vl) not in ev:
                continue
            count[stratum] += 1
            taken += 1
            rows.append({
                "kind": "compare", "table_id": tids[oi], "gap_stratum": stratum,
                "evidence": ev,
                "claim_pos": f"The {col} of {k_hi} is greater than the {col} of {k_lo}.",
                "claim_neg": f"The {col} of {k_lo} is greater than the {col} of {k_hi}.",
            })
    return rows, count


def score_pairs(C, tok, trunk, head, rows):
    claims = [r["claim_pos"] for r in rows] + [r["claim_neg"] for r in rows]
    evs = [r["evidence"] for r in rows] * 2
    s = C.score(tok, trunk, head, claims, evs)
    n = len(rows)
    return s[:n], s[n:]


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--arm", default="R14-H133",
                    help="arm prefix: reads models/<arm>-arm-draw<N>, writes <arm>_probes_* "
                         "and <arm>_probe_scores.parquet (default the H133 arm)")
    args = ap.parse_args()
    arm_ckpt = f"{args.arm}-arm-draw{args.draw}"
    result = HERE / f"{args.arm}_probes_draw{args.draw}_result.json"
    scores = HERE / f"{args.arm}_probe_scores.parquet"

    C = _mod("c", "R15_gate_common.py")
    P1 = _mod("p1", "R15_P1_typeprobe.py")
    H133P = _mod("h133p", "R14_H133_probe.py")
    B4 = _mod("b4", "R15_gate_B4.py")
    AG = _mod("ag", "R14-H133_antigaming.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    built = P1.build(np.random.default_rng(P1.SEED))
    quads = [q for t in P1.TYPES for q in built[t]]
    tri = H133P.build(H133P.N_TRIPLES, np.random.default_rng(H133P.SEED))
    bind_col, _census = B4.build(C, np.random.default_rng(B4.SEED))
    compare, gap_count = build_compare(C, np.random.default_rng(SEED))
    bind_row = AG.build_bindrow(C, np.random.default_rng(SEED))
    print(f"surfaces: {len(quads)} P1 quads / {len(tri)} H133 triples / "
          f"{len(bind_col)} bind_col / {len(compare)} compare {gap_count} / "
          f"{len(bind_row)} bind_row", flush=True)

    q_claims, q_evs = [], []
    for tag in ("claim_a", "claim_b", "claim_c", "claim_d"):
        q_claims += [q[tag] for q in quads]
        q_evs += [q["evidence"] for q in quads]
    t_claims = ([t["claim_a"] for t in tri] + [t["claim_b"] for t in tri]
                + [t["claim_c"] for t in tri])
    t_evs = [t["evidence"] for t in tri] * 3
    bc_rows = [{"claim_pos": p["claim_pos"], "claim_neg": p["claim_neg"],
                "evidence": p["evidence"]} for p in bind_col]
    nq, nt = len(quads), len(tri)

    per_ckpt, banked = {}, {}
    for name in (arm_ckpt, CONTROL_CKPT):
        tok, trunk, head = C.load_ckpt(name)
        sq = C.score(tok, trunk, head, q_claims, q_evs)
        st = C.score(tok, trunk, head, t_claims, t_evs)
        bcP, bcN = score_pairs(C, tok, trunk, head, bc_rows)
        cmpP, cmpN = score_pairs(C, tok, trunk, head, compare)
        brP, brN = score_pairs(C, tok, trunk, head, bind_row)
        del trunk, head
        torch.cuda.empty_cache()

        sa, sb, sc, sd = sq[:nq], sq[nq:2 * nq], sq[2 * nq:3 * nq], sq[3 * nq:]
        banked[name] = {"quad_b": sb, "quad_c": sc}
        per_type = {}
        for t in P1.TYPES:
            m = np.array([q["dtype"] == t for q in quads])
            if m.sum() < 30:
                per_type[t] = {"n": int(m.sum()), "note": "under 30 - not adjudicated"}
                continue
            per_type[t] = {
                "n": int(m.sum()),
                "mean_a_verbatim": round(float(sa[m].mean()), 5),
                "mean_b_correct": round(float(sb[m].mean()), 5),
                "mean_c_wrong_operand": round(float(sc[m].mean()), 5),
                "auroc_b_vs_c": round(C.auroc(sb[m], sc[m]), 4),
                "auroc_b_vs_d": round(C.auroc(sb[m], sd[m]), 4),
                "auroc_a_vs_b": round(C.auroc(sa[m], sb[m]), 4),
            }
        ta, tb, tc = st[:nt], st[nt:2 * nt], st[2 * nt:]
        cmp_strat = {}
        strat = np.array([r["gap_stratum"] for r in compare])
        for s_name in ("lt10pct", "10to100pct", "gt100pct"):
            m = strat == s_name
            cmp_strat[s_name] = ({"n": int(m.sum()),
                                  "auroc_pos_vs_neg": round(C.auroc(cmpP[m], cmpN[m]), 4)}
                                 if m.sum() >= 30 else
                                 {"n": int(m.sum()), "note": "under 30 - not adjudicated"})

        per_ckpt[name] = {
            "p1_quads": {
                "n": nq,
                "per_type": per_type,
                "tier1_auroc_b_vs_c": {t: per_type[t]["auroc_b_vs_c"] for t in TIER1},
                "verbatim_mean_pooled": round(float(sa.mean()), 5),
                "auroc_a_vs_b_pooled": round(C.auroc(sa, sb), 4),
                "scale_unit_auroc_b_vs_c": per_type["scale_unit"]["auroc_b_vs_c"],
            },
            "h133_triples": {
                "n": nt,
                "mean_a_verbatim": round(float(ta.mean()), 5),
                "mean_b_correct": round(float(tb.mean()), 5),
                "mean_c_wrong_operand": round(float(tc.mean()), 5),
                "gap_a_minus_b": round(float(ta.mean() - tb.mean()), 5),
                "auroc_b_vs_c": round(C.auroc(tb, tc), 4),
                "auroc_a_vs_b": round(C.auroc(ta, tb), 4),
            },
            "b4_relational": {
                "bind_col": {"n": len(bc_rows), "auroc_pos_vs_neg": round(C.auroc(bcP, bcN), 4),
                             "controlled": "digit-length AND magnitude matched (R15_gate_B4)"},
                "compare": {"n": len(compare), "auroc_pos_vs_neg": round(C.auroc(cmpP, cmpN), 4),
                            "per_gap_stratum": cmp_strat},
                "bind_row": {"n": len(bind_row),
                             "auroc_pos_vs_neg": round(C.auroc(brP, brN), 4)},
                "registered_references": B4_REFS,
            },
        }
        p = per_ckpt[name]
        print(f"{name}: tier1 {p['p1_quads']['tier1_auroc_b_vs_c']}  "
              f"scale/unit {p['p1_quads']['scale_unit_auroc_b_vs_c']:.4f}  "
              f"verbatim {p['p1_quads']['verbatim_mean_pooled']:.4f} / "
              f"{p['p1_quads']['auroc_a_vs_b_pooled']:.4f}", flush=True)
        print(f"    b4 bind_col {p['b4_relational']['bind_col']['auroc_pos_vs_neg']:.4f}  "
              f"compare {p['b4_relational']['compare']['auroc_pos_vs_neg']:.4f}  "
              f"bind_row {p['b4_relational']['bind_row']['auroc_pos_vs_neg']:.4f}", flush=True)

    df = pl.DataFrame({"dtype": [q["dtype"] for q in quads],
                       "table_id": [q["table_id"] for q in quads]})
    for name, cols in banked.items():
        tag = name.replace("-", "_")
        df = df.with_columns([pl.Series(f"b__{tag}", cols["quad_b"]),
                              pl.Series(f"c__{tag}", cols["quad_c"])])
    df.write_parquet(scores)

    arm, ctl = per_ckpt[arm_ckpt], per_ckpt[CONTROL_CKPT]
    res = {
        "read": "R14-H133 probe bank - pre-registered mechanism reading (report-only)",
        "draw": args.draw,
        "arm_checkpoint": arm_ckpt,
        "control_checkpoint": CONTROL_CKPT,
        "control_note": "the BANKED clean draw 1 - unseeded (pre-H126); the comparison is "
                        "arm-vs-banked-control, not init-paired",
        "data": "R15_P1_typeprobe.build / R14_H133_probe.build / R15_gate_B4.build rebuilt at "
                "their banked seeds, plus compare and bind_row built here, all over held-out "
                "TabFact test+validation, table_id-disjoint from every train split",
        "seed_compare_bindrow": SEED,
        "references": {
            "tier1_target_above": TIER1_TARGET,
            "tier1_baseline_h105_draw1": {"sum": 0.5067, "difference": 0.4994,
                                          "ratio": 0.5121, "pct_change": 0.4861},
            "scale_unit_void_absolute": SCALE_UNIT_VOID_ABSOLUTE,
            "scale_unit_void_erosion": SCALE_UNIT_VOID_EROSION,
            "scale_unit_void_rebased_below": SCALE_UNIT_VOID_REBASED,
            "scale_unit_ruling": "review correction F2 (2026-08-10): the absolute 0.80 clause is "
                                 "RESTORED for this arm (VOID if < 0.80 OR < control - 0.05); the "
                                 "0.4548 rebase applies only to H108-lane-carrying compositions "
                                 "and is printed for lineage only",
            "verbatim_mean_at_or_above": VERBATIM_MEAN_REF,
            "verbatim_mean_banked": 0.90507,
            "auroc_a_vs_b_at_or_above": VERBATIM_AUROC_REF,
            "auroc_a_vs_b_banked": 0.9643,
            "b4": B4_REFS,
        },
        "checkpoints": per_ckpt,
        "headline": {
            "tier1_arm": arm["p1_quads"]["tier1_auroc_b_vs_c"],
            "tier1_control": ctl["p1_quads"]["tier1_auroc_b_vs_c"],
            "scale_unit_arm": arm["p1_quads"]["scale_unit_auroc_b_vs_c"],
            "scale_unit_control": ctl["p1_quads"]["scale_unit_auroc_b_vs_c"],
            "scale_unit_void": bool(
                arm["p1_quads"]["scale_unit_auroc_b_vs_c"] < SCALE_UNIT_VOID_ABSOLUTE
                or arm["p1_quads"]["scale_unit_auroc_b_vs_c"]
                < ctl["p1_quads"]["scale_unit_auroc_b_vs_c"] - SCALE_UNIT_VOID_EROSION),
            "scale_unit_below_rebased_void": bool(
                arm["p1_quads"]["scale_unit_auroc_b_vs_c"] < SCALE_UNIT_VOID_REBASED),
            "verbatim_mean_arm": arm["p1_quads"]["verbatim_mean_pooled"],
            "auroc_a_vs_b_arm": arm["p1_quads"]["auroc_a_vs_b_pooled"],
            "bind_col_arm": arm["b4_relational"]["bind_col"]["auroc_pos_vs_neg"],
            "compare_arm": arm["b4_relational"]["compare"]["auroc_pos_vs_neg"],
            "bind_row_arm": arm["b4_relational"]["bind_row"]["auroc_pos_vs_neg"],
        },
        "adjudication": "NOT ADJUDICATED HERE - the coordinator holds the verdict",
        "scores": scores.name,
    }
    result.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    print(f"  tier-1 arm      {res['headline']['tier1_arm']}")
    print(f"  scale/unit      arm {res['headline']['scale_unit_arm']:.4f}  "
          f"control {res['headline']['scale_unit_control']:.4f}  "
          f"(rebased VOID below {SCALE_UNIT_VOID_REBASED})")
    print(f"  verbatim        mean {res['headline']['verbatim_mean_arm']:.4f}  "
          f"AUROC(a vs b) {res['headline']['auroc_a_vs_b_arm']:.4f}")
    print(f"  b4              bind_col {res['headline']['bind_col_arm']:.4f}  "
          f"compare {res['headline']['compare_arm']:.4f}  "
          f"bind_row {res['headline']['bind_row_arm']:.4f}")
    print(f"\n  -> {result}", flush=True)


if __name__ == "__main__":
    main()
