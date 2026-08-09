"""R15-B5 THE DERIVATION INSTRUMENT PANEL - arms 2, 5, 6, 7 (and the panel roll-up).

Nine rounds of arms have been adjudicated on instruments R14 verdict A records
as blind. This panel banks the per-checkpoint derivation baselines before the
first R15 arm lands, so every subsequent adjudication is a mechanism reading
rather than a mean.

Arms measured here:

  2  anchor-teacher / lever read  - the H133 derivation axis on R13-H129-draw1
     (the refuted student), R10-H108-lane-draw1/2, and the free-CPU committee
     (output mean of the two H105 draws).
     H108 > 0.55 -> R14-A4's marginal claim over the admitted lane must be
     re-argued before the lane builds.
     Committee <= 0.55 -> a binding register entry that the 0.72067 advantage
     is variance cancellation carrying zero derivation competence.
  5  trunk numeracy retention - P4 Instrument B plus its three controls on the
     un-fine-tuned mmBERT-base, H105 d1/d2, H108 lane d1, and the declinable
     pretrained mmBERT-small. Establishes the SUBSTRATE HOLD baseline.
  6  wrong-factor scale gate - L3-C1's matched triples (verbatim / re-surfaced /
     wrong-factor). LICENSE at mean |score(a) - score(b)| >= 0.10 AND
     AUROC(a vs c) <= 0.60.
  7  1024 within-rows re-read - the banked triples at max_length 1024, paired
     against their own 512 scores. Delta AUROC(b vs c) < +0.03 -> READ LENGTH IS
     NOT A DERIVATION LEVER. VOID the inference if AUROC(a vs b) on the
     truncated partition moves less than +0.05.

Arms 1, 3, 4 and 8 are measured by R15_gate_B1 / B6 / B4 / B5arm8 and are
cross-referenced here so the panel is one object with one baseline.

Frozen weights, held-out, arena-free, gold-free.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R15_gate_B5_result.json"
SCORES = HERE / "R15_gate_B5_scores.parquet"

ARM2_CKPTS = ["R9-H105-mmbert-dann-clean", "R9-H105-draw2", "R10-H108-lane-draw1",
              "R10-H108-lane-draw2", "R13-H129-draw1"]
ARM5_TRUNKS = {
    "base_pretrained": ("jhu-clsp/mmBERT-base", "jhu-clsp/mmBERT-base"),
    "small_pretrained": ("jhu-clsp/mmBERT-small", "jhu-clsp/mmBERT-small"),
    "h105_draw1": (str(ROOT / "models" / "R9-H105-mmbert-dann-clean" / "trunk"),
                   str(ROOT / "models" / "R9-H105-mmbert-dann-clean")),
    "h105_draw2": (str(ROOT / "models" / "R9-H105-draw2" / "trunk"),
                   str(ROOT / "models" / "R9-H105-draw2")),
    "h108_lane_draw1": (str(ROOT / "models" / "R10-H108-lane-draw1" / "trunk"),
                        str(ROOT / "models" / "R10-H108-lane-draw1")),
}
ARM5_BASELINE = "base_pretrained"
ARM6_CKPT = "R9-H105-mmbert-dann-clean"
ARM7_CKPT = "R9-H105-mmbert-dann-clean"

SEED = 20260812
P4_SEED = 20260810
N_ARM6 = 600
SIBLINGS = {"arm_1_checkpoint_specificity": "R15_gate_B1_result.json",
            "arm_3_evidence_conditioning": "R15_gate_B6_result.json",
            "arm_4_controlled_bind_col": "R15_gate_B4_result.json",
            "arm_8_natural_derivation": "R15_gate_B5arm8_result.json"}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- #
# arm 6 - wrong-factor scale triples
# --------------------------------------------------------------------------- #
def resurface(v, cell, rng):
    """The same quantity spelled differently - separator / scale word / currency."""
    opts = []
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1000 and "," not in cell:
        opts.append(f"{int(round(v)):,}")
    if abs(v) >= 1000 and abs(v / 1000 - round(v / 1000, 3)) < 1e-9:
        opts.append(f"{v / 1000:g} thousand")
    if abs(v) >= 1_000_000 and abs(v / 1e6 - round(v / 1e6, 3)) < 1e-9:
        opts.append(f"{v / 1e6:g} million")
    if abs(v - round(v)) < 1e-9 and abs(v) >= 1000:
        opts.append(f"${int(round(v)):,}")
    opts = [o for o in opts if o != cell]
    return opts[int(rng.integers(len(opts)))] if opts else None


def build_arm6(C, rng):
    caps, tbls, tids = C.held_tabfact()
    order = [int(o) for _ in range(3) for o in rng.permutation(len(tbls))]
    out, seen = [], set()
    for oi in order:
        if len(out) >= N_ARM6:
            break
        hdr, body = C.parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:C.CHUNK_MAX]
        cand = []
        for ci in range(1, len(hdr)):
            vals = [(ri, C.as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) >= 3:
                cand.append((ci, vals))
        if not cand:
            continue
        ci, vals = cand[int(rng.integers(len(cand)))]
        col = hdr[ci] or f"column {ci}"
        if any((not body[ri][0]) or C.as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        ri_a, v = vals[int(rng.integers(len(vals)))]
        ka, cell = body[ri_a][0].strip(), body[ri_a][ci].strip()
        if not ka or cell not in ev or abs(v) < 1e-9:
            continue
        b_surface = resurface(v, cell, rng)
        if b_surface is None or b_surface in ev:
            continue
        up = bool(rng.integers(2))  # balanced x10 / /10
        wf = C.fmt(v * 10) if up else C.fmt(v / 10)
        if wf == cell or wf in ev:
            continue
        key = (tids[oi], col, ka, cell)
        if key in seen:
            continue
        seen.add(key)
        T = f"The {col} of {ka} is {{}}."
        out.append({"table_id": tids[oi], "column": col, "evidence": ev,
                    "claim_a": T.format(cell), "claim_b": T.format(b_surface),
                    "claim_c": T.format(wf), "v_a": cell, "v_b": b_surface, "v_c": wf,
                    "factor": "x10" if up else "/10"})
    return out


# --------------------------------------------------------------------------- #
# arm 5 - P4 Instrument B and its three controls
# --------------------------------------------------------------------------- #
def ridge_solve(X, y, lam):
    X = np.c_[X, np.ones(len(X))]
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    w = ridge_solve(Xtr, ytr, lam)
    p = np.c_[Xte, np.ones(len(Xte))] @ w
    return float(1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def ridge_acc(Xtr, ytr, Xte, yte, lam=1.0):
    w = ridge_solve(Xtr, ytr, lam)
    p = np.c_[Xte, np.ones(len(Xte))] @ w
    return float(((p > 0.5).astype(float) == yte).mean()), p


def arm5_sets(rng):
    lo = rng.integers(1, 1000, 3000)
    hi = rng.integers(10_000, 100_000, 1500)
    mag = ([f"The value is {v}." for v in np.r_[lo, hi]],
           ["The table records a single measured quantity." for _ in range(4500)],
           np.log10(np.r_[lo, hi].astype(float)))
    a = rng.integers(1, 1000, 4000)
    b = rng.integers(1, 1000, 4000)
    keep = a != b
    a, b = a[keep], b[keep]
    cmp_ = ([f"Alpha is {x} and Beta is {y}." for x, y in zip(a, b)],
            ["The table lists two quantities." for _ in a],
            (a > b).astype(float))
    return mag, cmp_


# --------------------------------------------------------------------------- #
def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    H133 = _mod("h133", "R14_H133_probe.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tri = H133.build(H133.N_TRIPLES, np.random.default_rng(H133.SEED))
    nt = len(tri)
    t_claims = ([t["claim_a"] for t in tri] + [t["claim_b"] for t in tri]
                + [t["claim_c"] for t in tri])
    t_evs = [t["evidence"] for t in tri] * 3
    print(f"rebuilt {nt} H133 triples", flush=True)

    arm6 = build_arm6(C, np.random.default_rng(SEED))
    print(f"arm 6 wrong-factor triples: {len(arm6)}", flush=True)
    a6_claims = ([q["claim_a"] for q in arm6] + [q["claim_b"] for q in arm6]
                 + [q["claim_c"] for q in arm6])
    a6_evs = [q["evidence"] for q in arm6] * 3
    n6 = len(arm6)

    # ---- arm 2 (+ arm 6 and arm 7 ride the same checkpoint loads) -----------
    banked, arm2 = {}, {}
    a6_scores, arm7 = None, {}
    for name in ARM2_CKPTS:
        tok, trunk, head = C.load_ckpt(name)
        s = C.score(tok, trunk, head, t_claims, t_evs)
        if name == ARM6_CKPT:
            a6_scores = C.score(tok, trunk, head, a6_claims, a6_evs)
        if name == ARM7_CKPT:
            lens = np.array([len(x) for x in
                             tok(t_claims, t_evs, truncation=False)["input_ids"]], dtype=np.int32)
            s1024 = C.score(tok, trunk, head, t_claims, t_evs, max_len=1024)
            arm7 = {"s512": s, "s1024": s1024, "lens": lens}
        del trunk, head
        torch.cuda.empty_cache()
        sa, sb, sc = s[:nt], s[nt:2 * nt], s[2 * nt:]
        banked[name] = (sa, sb, sc)
        arm2[name] = {
            "mean_a_verbatim": round(float(sa.mean()), 5),
            "mean_b_correct": round(float(sb.mean()), 5),
            "mean_c_wrong_operand": round(float(sc.mean()), 5),
            "auroc_b_vs_c": round(C.auroc(sb, sc), 4),
            "auroc_a_vs_b": round(C.auroc(sa, sb), 4),
        }
        print(name, json.dumps(arm2[name]), flush=True)

    a1, b1, c1 = banked["R9-H105-mmbert-dann-clean"]
    a2, b2, c2 = banked["R9-H105-draw2"]
    ta, tb, tc = (a1 + a2) / 2, (b1 + b2) / 2, (c1 + c2) / 2
    arm2["H129-committee-teacher (output mean of H105 d1 + d2)"] = {
        "mean_a_verbatim": round(float(ta.mean()), 5),
        "mean_b_correct": round(float(tb.mean()), 5),
        "mean_c_wrong_operand": round(float(tc.mean()), 5),
        "auroc_b_vs_c": round(C.auroc(tb, tc), 4),
        "auroc_a_vs_b": round(C.auroc(ta, tb), 4),
        "cross_draw_pearson_on_b": round(float(np.corrcoef(b1, b2)[0, 1]), 4),
        "cross_draw_pearson_on_b_minus_c": round(float(np.corrcoef(b1 - c1, b2 - c2)[0, 1]), 4),
    }
    committee = arm2["H129-committee-teacher (output mean of H105 d1 + d2)"]["auroc_b_vs_c"]
    h108 = max(arm2["R10-H108-lane-draw1"]["auroc_b_vs_c"],
               arm2["R10-H108-lane-draw2"]["auroc_b_vs_c"])
    arm2_reading = {
        "h108_max_auroc_b_vs_c": h108,
        "h108_clause": ("RE-ARGUE - A4's marginal claim over the admitted H108 lane must be "
                        "re-argued before the lane builds" if h108 > 0.55 else
                        "CLEAR - the admitted lane carries no derivation competence, A4's "
                        "marginal claim over it stands"),
        "committee_auroc_b_vs_c": committee,
        "committee_clause": ("DERIVATION-BLIND ANCHOR - the 0.72067 committee advantage is "
                             "variance cancellation carrying zero derivation competence; any "
                             "future proposal to distil derivation competence from a committee "
                             "must first exhibit a teacher above 0.60 on this instrument"
                             if committee <= 0.55 else
                             "committee above 0.55 - not derivation-blind; record and re-read"),
    }

    # ---- arm 6 --------------------------------------------------------------
    s6a, s6b, s6c = a6_scores[:n6], a6_scores[n6:2 * n6], a6_scores[2 * n6:]
    surf_gap = float(np.abs(s6a - s6b).mean())
    auc_ac = C.auroc(s6a, s6c)
    dl = np.array([sum(ch.isdigit() for ch in q["v_c"]) for q in arm6], dtype=float)
    dl_a = np.array([sum(ch.isdigit() for ch in q["v_a"]) for q in arm6], dtype=float)
    arm6_res = {
        "n": n6,
        "mean_a_verbatim": round(float(s6a.mean()), 5),
        "mean_b_resurfaced": round(float(s6b.mean()), 5),
        "mean_c_wrong_factor": round(float(s6c.mean()), 5),
        "mean_abs_gap_a_minus_b": round(surf_gap, 5),
        "auroc_a_vs_c": round(auc_ac, 4),
        "auroc_a_vs_b": round(C.auroc(s6a, s6b), 4),
        "auroc_from_digit_length_a_vs_c": round(C.auroc(dl_a, dl), 4),
        "share_x10": round(float(np.mean([q["factor"] == "x10" for q in arm6])), 4),
        "bar": "LICENSE at mean |score(a) - score(b)| >= 0.10 AND AUROC(a vs c) <= 0.60",
        "verdict": ("LICENSE" if surf_gap >= 0.10 and auc_ac <= 0.60 else "KILL"),
    }
    arm6_res["clause_fired"] = (
        "LICENSE - surfaces are not equivalent to the model and decade errors on an identical "
        "digit string are not rejected: there is an equivalence AND an inequivalence to install"
        if arm6_res["verdict"] == "LICENSE" else
        ("KILL - surfaces already equivalent (gap < 0.10)" if surf_gap < 0.10 else
         "KILL - magnitude already discriminated on the copy axis (AUROC(a vs c) > 0.60); P1's "
         "'scale already works' reading extends to the wrong-factor axis"))
    print("arm6", json.dumps(arm6_res), flush=True)

    # ---- arm 7 --------------------------------------------------------------
    lens_b = arm7["lens"][nt:2 * nt]
    over = lens_b > C.MAX_LEN
    s5, s10 = arm7["s512"], arm7["s1024"]
    a5_, b5_, c5_ = s5[:nt], s5[nt:2 * nt], s5[2 * nt:]
    a10, b10, c10 = s10[:nt], s10[nt:2 * nt], s10[2 * nt:]
    d_bc = C.auroc(b10, c10) - C.auroc(b5_, c5_)
    d_ab_trunc = (C.auroc(a10[over], b10[over]) - C.auroc(a5_[over], b5_[over])
                  if over.sum() > 20 else None)
    arm7_res = {
        "n": nt,
        "share_pairs_over_512": round(float(over.mean()), 4),
        "auroc_b_vs_c_512": round(C.auroc(b5_, c5_), 4),
        "auroc_b_vs_c_1024": round(C.auroc(b10, c10), 4),
        "delta_auroc_b_vs_c": round(float(d_bc), 4),
        "truncated_partition": {
            "n": int(over.sum()),
            "auroc_a_vs_b_512": round(C.auroc(a5_[over], b5_[over]), 4) if over.sum() > 20 else None,
            "auroc_a_vs_b_1024": round(C.auroc(a10[over], b10[over]), 4) if over.sum() > 20 else None,
            "delta_auroc_a_vs_b": round(float(d_ab_trunc), 4) if d_ab_trunc is not None else None,
        },
        "bar": "delta AUROC(b vs c) at 1024 < +0.03 -> READ LENGTH IS NOT A DERIVATION LEVER; "
               "VOID the inference if AUROC(a vs b) on the truncated partition moves less "
               "than +0.05",
    }
    if d_ab_trunc is None or d_ab_trunc < 0.05:
        arm7_res["verdict"] = "VOID"
        arm7_res["clause_fired"] = ("VOID - the truncated partition's AUROC(a vs b) did not move "
                                    "by +0.05 at 1024, so the longer read did not demonstrably "
                                    "restore the missing evidence and the derivation inference "
                                    "cannot be drawn from it")
    elif d_bc < 0.03:
        arm7_res["verdict"] = "READ LENGTH IS NOT A DERIVATION LEVER"
        arm7_res["clause_fired"] = ("H131 Stage 2 gains no derivation argument; P2-D's token "
                                    "budget stays a build-hygiene KILL rather than a mechanism "
                                    "claim; A4 may build at 512")
    else:
        arm7_res["verdict"] = "READ LENGTH IS A DERIVATION LEVER"
        arm7_res["clause_fired"] = f"delta AUROC(b vs c) {d_bc:+.4f} >= +0.03 at 1024"
    print("arm7", json.dumps(arm7_res), flush=True)

    # ---- arm 5 --------------------------------------------------------------
    rng = np.random.default_rng(P4_SEED)
    (mag_c, mag_e, mag_y), (cmp_c, cmp_e, cmp_y) = arm5_sets(rng)
    n_lo, cut, n_tr = 3000, 2000, int(0.7 * len(cmp_y))
    yperm = rng.permutation(cmp_y)
    arm5 = {}
    for mname, (path, tokpath) in ARM5_TRUNKS.items():
        tok, trunk = C.load_trunk(path, tokpath)
        Cm = C.cls_of(tok, trunk, mag_c, mag_e)
        Cc = C.cls_of(tok, trunk, cmp_c, cmp_e)
        hid = trunk.config.hidden_size
        del trunk
        torch.cuda.empty_cache()
        Cl, Ch = Cm[:n_lo], Cm[n_lo:]
        yl, yh = mag_y[:n_lo], mag_y[n_lo:]
        acc, p = ridge_acc(Cc[:n_tr], cmp_y[:n_tr], Cc[n_tr:], cmp_y[n_tr:])
        arm5[mname] = {
            "hidden_size": int(hid),
            "magnitude_ridge_r2_interpolation_1_to_999": round(
                ridge_r2(Cl[:cut], yl[:cut], Cl[cut:], yl[cut:]), 4),
            "magnitude_ridge_r2_extrapolation_REPORT_ONLY": round(ridge_r2(Cl, yl, Ch, yh), 4),
            "comparison_probe_heldout_accuracy": round(acc, 4),
            "comparison_probe_heldout_auroc": round(
                C.auroc(p[cmp_y[n_tr:] == 1], p[cmp_y[n_tr:] == 0]), 4),
            "control_permuted_labels_acc": round(
                ridge_acc(Cc[:n_tr], yperm[:n_tr], Cc[n_tr:], yperm[n_tr:])[0], 4),
            "control_train200_acc": round(
                ridge_acc(Cc[:200], cmp_y[:200], Cc[n_tr:], cmp_y[n_tr:])[0], 4),
            "control_ridge_lambda_1e4_acc": round(
                ridge_acc(Cc[:n_tr], cmp_y[:n_tr], Cc[n_tr:], cmp_y[n_tr:], lam=1e4)[0], 4),
        }
        print("arm5", mname, json.dumps(arm5[mname]), flush=True)

    base = arm5[ARM5_BASELINE]
    holds = {}
    for mname, r in arm5.items():
        if mname == ARM5_BASELINE or mname == "small_pretrained":
            continue
        ok_r2 = r["magnitude_ridge_r2_interpolation_1_to_999"] >= (
            base["magnitude_ridge_r2_interpolation_1_to_999"] - 0.05)
        ok_cmp = (r["comparison_probe_heldout_accuracy"] >= 0.95
                  and r["comparison_probe_heldout_accuracy"] >= (
                      base["comparison_probe_heldout_accuracy"] - 0.02))
        holds[mname] = {"magnitude_hold": bool(ok_r2), "comparison_hold": bool(ok_cmp),
                        "verdict": "HOLD" if (ok_r2 and ok_cmp) else "SUBSTRATE DAMAGE"}
    spread = max(abs(arm5[m]["comparison_probe_heldout_accuracy"]
                     - base["comparison_probe_heldout_accuracy"]) for m in holds)
    arm5_reading = {
        "baseline_checkpoint": ARM5_BASELINE,
        "substrate_hold_rule": "magnitude interpolation R2 >= (baseline - 0.05) AND comparison "
                               "accuracy >= 0.95 absolute AND >= (baseline - 0.02); extrapolation "
                               "R2 is REPORT, not bar",
        "per_checkpoint": holds,
        "max_comparison_accuracy_spread_vs_baseline": round(float(spread), 4),
        "retirement_clause": ("RETIRE after one round - every checkpoint reads within 0.01 of the "
                             "pretrained base and the instrument has no resolving power"
                             if spread < 0.01 else
                             "KEEP - checkpoints separate on this instrument"),
    }

    # ---- bank scores --------------------------------------------------------
    df = pl.DataFrame({
        "table_id": [t["table_id"] for t in tri],
        "v_correct": [t["v_correct"] for t in tri],
        "v_wrong": [t["v_wrong"] for t in tri],
        "pair_tokens_b": lens_b.tolist(),
    })
    for name, (sa, sb, sc) in banked.items():
        tag = name.replace("-", "_")
        df = df.with_columns([pl.Series(f"a__{tag}", sa), pl.Series(f"b__{tag}", sb),
                              pl.Series(f"c__{tag}", sc)])
    df = df.with_columns([pl.Series("b__h105d1_1024", b10), pl.Series("c__h105d1_1024", c10),
                          pl.Series("a__h105d1_1024", a10)])
    df.write_parquet(SCORES)

    cross = {}
    for k, fn in SIBLINGS.items():
        p = HERE / fn
        if p.exists():
            j = json.loads(p.read_text())
            cross[k] = {"file": fn, "verdict": j.get("verdict"), "gate": j.get("gate")}
        else:
            cross[k] = {"file": fn, "verdict": None, "note": "not yet landed at panel write time"}

    res = {
        "panel": "R15-B5 the derivation instrument panel",
        "data": "R14_H133_probe.build() at its banked seed and fresh held-out TabFact "
                "test+validation builds; synthetic two-row tables for the substrate probe. "
                "Frozen weights throughout, zero arena, zero gold.",
        "implementation_choices": [
            "Arm 2's committee is the free-CPU output mean of the two H105 draws' probabilities, "
            "per the synthesis amendment that no GPU be spent on it beyond scoring draw 2.",
            "Arm 5 reuses R15_P4_numeracy_probe.py's Instrument B construction and seed "
            f"({P4_SEED}) so the shipped-checkpoint reading is comparable to the banked "
            "R15_gate_P4_numeracy.json; the three controls are P4's own (permuted labels, "
            "200-row training subset, ridge lambda 1e4).",
            "Arm 5's declinable mmBERT-small half is RUN and reported, but it is excluded from "
            "the SUBSTRATE HOLD adjudication - the hold binds the 307M line the register trains.",
            "Arm 6's re-surfaced member (b) is drawn uniformly from the constructible separator / "
            "scale-word / currency spellings of the same quantity; the wrong-factor member (c) is "
            "x10 and /10 at balanced rates, both absent from the evidence.",
            "Arm 7 scores the identical triples at max_length 512 and 1024 in one checkpoint "
            "load; the truncated partition is the pairs whose untruncated encoding exceeds 512.",
        ],
        "arm_2_anchor_teacher_lever": {"per_checkpoint": arm2, "reading": arm2_reading},
        "arm_5_trunk_numeracy_retention": {"per_checkpoint": arm5, "reading": arm5_reading},
        "arm_6_wrong_factor_scale_gate": arm6_res,
        "arm_7_read_length_1024": arm7_res,
        "cross_referenced_arms": cross,
        "clause_A_ship_read": "0 GPU - R14-H132's parked 140.9M ship candidate must, alongside its "
                              "arena holds, REPORT AUROC(b vs c) on these triples, P4's comparison "
                              "family and verbatim AUROC(a vs b), and HOLD scale/unit AUROC >= 0.80",
        "clause_B_serving_ensemble_price": (
            f"the serving-side two-forward-pass ensemble buys +0.01756 of arena mean and a "
            f"committee derivation AUROC(b vs c) of {committee}"),
        "clause_C": "every post-arm read in the register compares against THIS panel's "
                    "per-checkpoint baseline, not against a prose-quoted anchor",
        "scores": SCORES.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print(f"\n-> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
