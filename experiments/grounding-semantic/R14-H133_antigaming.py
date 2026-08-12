"""R14-H133 ANTI-GAMING read (binding, A4 + R15-B4 binding amendment (i)).

A4's anti-gaming clause: "in-domain held-out H108-style PRESENT-VALUE near-miss
AUC must not fall below the clean-recipe value". A model that has learned
"absent number implies supported" from the H133 lane clears the finqa primary
and fails here. B4's binding amendment (i) adds `bind_row` >= 0.95 as a second
non-regression read, because the B4 sub-block rides this arm.

The eval set is re-constituted here rather than reused, because the banked B4 /
P1 surfaces are not operator-disjoint from the lane:

  TABLE-DISJOINT     TabFact test+validation only, with every table_id that
                     appears in TabFact train removed, and the surviving ids
                     measured against the lane's own `doc_id` column (both
                     `tabfact:` and `feverous:` namespaces). Reported, not
                     assumed - see `disjointness` in the result json.
  OPERATOR-DISJOINT  every negative here edits a numeral VERBATIM PRESENT in
                     the evidence, the H108 construction; every H133 derivation
                     -core negative asserts a value ABSENT from the evidence.
                     The two families that come nearest a lane operator -
                     `digit_perturb` (the N7 last-digit analogue, capped at
                     8.79% of lane negatives, and applied there to a derived
                     absent value) and `comparative_flip` (the `compare`
                     sub-block analogue) - are EXCLUDED from the headline AUROC
                     and reported separately, so the headline is disjoint on
                     the strict reading too.

Both checkpoints are read on the byte-identical set: the arm draw and the
BANKED CLEAN DRAW 1 (`models/R9-H105-mmbert-dann-clean`), which is the paired
control the A4 bar is written against. The set is banked as a parquet so the
comparison is reproducible.

Reports; does not adjudicate. Frozen weights, zero arena, zero gold.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     uv run python experiments/grounding-semantic/R14-H133_antigaming.py --draw 1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
LANE = HERE / "R14-H133_lane.parquet"

CONTROL_CKPT = "R9-H105-mmbert-dann-clean"  # the BANKED clean draw 1
SEED = 20260810
N_NEARMISS = 1000
N_BINDROW = 600
PER_TABLE_CAP = 2
BINDROW_BAR = 0.95
# Excluded from the headline: the two families nearest a lane operator.
EXCLUDED_FAMILIES = ("digit_perturb", "comparative_flip")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def held_statements(C):
    """TabFact test+validation entailed statements over tables whose table_id
    never appears in TabFact train - the split every arm trained on."""
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    train_ids = set(
        pl.read_parquet(
            io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith("__train.parquet"))))
        )["table_id"].to_list()
    )
    held = pl.concat([
        pl.read_parquet(io.BytesIO(z.read(n)))
        for n in z.namelist()
        if n.endswith("__test.parquet") or n.endswith("__validation.parquet")
    ])
    held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
    return held, train_ids


def evidence_of(cap, tbl, C):
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:C.CHUNK_MAX]


def build_nearmiss(C, H108D, rng):
    """H108-style present-value near-miss pairs: an ENTAILED TabFact statement
    whose numeral is verbatim in the evidence, against the same statement with
    that numeral (or its unit/scale word) corrupted by an H108 operator."""
    held, _ = held_statements(C)
    pos = held.filter(pl.col("label") == 1)
    order = [int(o) for o in rng.permutation(len(pos))]
    stmt = pos["statement"].to_list()
    cap = pos["table_caption"].to_list()
    tbl = pos["table_text"].to_list()
    tid = pos["table_id"].to_list()

    H108D.rng.seed(SEED)  # the operators draw from the module's own Random
    fams = list(H108D.FAMILIES)
    fam_cap = int(N_NEARMISS / len(fams) * 1.5)  # H108's own per-family cap rule
    fam_count = {f: 0 for f, _ in fams}
    lane_claims = set(pl.read_parquet(LANE)["claim"].to_list())
    taken, rows = {}, []
    for oi in order:
        if len(rows) >= N_NEARMISS:
            break
        if taken.get(tid[oi], 0) >= PER_TABLE_CAP:
            continue
        claim = stmt[oi]
        ev = evidence_of(cap[oi], tbl[oi], C)
        ev_nums = C.canon_set(ev)
        claim_nums = C.canon_set(claim)
        # PRESENT-VALUE requirement: the claim's numeral is in the evidence.
        if not claim_nums or not (claim_nums & ev_nums):
            continue
        if claim in lane_claims:
            continue  # no eval claim the lane could have memorised verbatim
        for fi in [int(k) for k in rng.permutation(len(fams))]:
            fam, fn = fams[fi]
            if fam_count[fam] >= fam_cap:
                continue
            got = fn(claim)
            if not got:
                continue
            neg, newval = got
            if neg == claim or neg in lane_claims:
                continue
            # value-checked families must land on a numeral absent from the table
            if fam in H108D.VALUE_CHECKED and (newval is None or newval in ev_nums):
                continue
            fam_count[fam] += 1
            taken[tid[oi]] = taken.get(tid[oi], 0) + 1
            rows.append({"kind": "nearmiss", "family": fam, "table_id": tid[oi],
                         "evidence": ev, "claim_pos": claim, "claim_neg": neg})
            break
    return rows


def build_bindrow(C, rng):
    """The B4 `bind_row` non-regression arm on the same held-out tables: the
    right column, the right row label, the value of a DIFFERENT row. Both
    values are printed in the evidence - the lane's own construction."""
    caps, tbls, tids = C.held_tabfact()
    order = [int(o) for o in rng.permutation(len(tbls))]
    lane_claims = set(pl.read_parquet(LANE)["claim"].to_list())
    rows = []
    for oi in order:
        if len(rows) >= N_BINDROW:
            break
        hdr, body = C.parse(tbls[oi])
        if hdr is None or len(body) < 3:
            continue
        ev = evidence_of(caps[oi], tbls[oi], C)
        taken = 0
        for ci in range(1, len(hdr)):
            if taken >= PER_TABLE_CAP:
                break
            vals = [(ri, C.as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) < 3 or len({v for _, v in vals}) < 3:
                continue
            # the row-label column must be non-numeric, as the lane requires
            if any((not body[ri][0]) or C.as_num(body[ri][0]) is not None for ri, _ in vals):
                continue
            col = hdr[ci] or f"column {ci}"
            pick = [int(k) for k in rng.permutation(len(vals))]
            ra, rb = vals[pick[0]], vals[pick[1]]
            la, lb = C.fmt(ra[1]), C.fmt(rb[1])
            ka, kb = body[ra[0]][0].strip(), body[rb[0]][0].strip()
            if la == lb or not ka or not kb or ka == kb or la not in ev or lb not in ev:
                continue
            T = f"The {col} of {ka} is {{}}."
            if T.format(la) in lane_claims or T.format(lb) in lane_claims:
                continue  # no eval claim the lane could have memorised verbatim
            rows.append({"kind": "bind_row", "family": "bind_row", "table_id": tids[oi],
                         "evidence": ev, "claim_pos": T.format(la), "claim_neg": T.format(lb)})
            taken += 1
    return rows


def disjointness(C, rows):
    """Measured, not assumed: eval table_ids against TabFact train and against
    every doc_id the lane carries."""
    _, train_ids = held_statements(C)
    lane = pl.read_parquet(LANE)
    lane_docs = set(lane["doc_id"].to_list())
    lane_tabfact = {d.split(":", 1)[1] for d in lane_docs if d.startswith("tabfact:")}
    eval_ids = {r["table_id"] for r in rows}
    lane_claims = set(lane["claim"].to_list())
    eval_claims = {r["claim_pos"] for r in rows} | {r["claim_neg"] for r in rows}
    return {
        "eval_tables": len(eval_ids),
        "lane_documents": len(lane_docs),
        "lane_tabfact_tables": len(lane_tabfact),
        "shared_with_tabfact_train": len(eval_ids & train_ids),
        "shared_with_lane_tables": len(eval_ids & lane_tabfact),
        "shared_claim_strings_with_lane": len(eval_claims & lane_claims),
        "table_disjoint": len(eval_ids & lane_tabfact) == 0 and len(eval_ids & train_ids) == 0,
        "operator_disjointness": (
            "every negative here edits a numeral verbatim present in the evidence; every H133 "
            "derivation-core negative asserts a value absent from it. The headline additionally "
            f"excludes {list(EXCLUDED_FAMILIES)}, the nearest lane analogues (N7 last-digit "
            "corruption and the `compare` sub-block)."
        ),
    }


def read_ckpt(C, name, rows):
    import torch

    tok, trunk, head = C.load_ckpt(name)
    claims = [r["claim_pos"] for r in rows] + [r["claim_neg"] for r in rows]
    evs = [r["evidence"] for r in rows] * 2
    s = C.score(tok, trunk, head, claims, evs)
    del trunk, head
    torch.cuda.empty_cache()
    n = len(rows)
    return s[:n], s[n:]


def summarise(C, rows, P, N):
    fam = np.array([r["family"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    nm = kind == "nearmiss"
    head_mask = nm & ~np.isin(fam, EXCLUDED_FAMILIES)
    out = {
        "nearmiss_headline": {
            "n": int(head_mask.sum()),
            "auroc_pos_vs_neg": round(C.auroc(P[head_mask], N[head_mask]), 4),
            "mean_pos": round(float(P[head_mask].mean()), 5),
            "mean_neg": round(float(N[head_mask].mean()), 5),
            "frac_pos_higher": round(float((P[head_mask] > N[head_mask]).mean()), 4),
        },
        "nearmiss_all_families": {
            "n": int(nm.sum()),
            "auroc_pos_vs_neg": round(C.auroc(P[nm], N[nm]), 4),
        },
        "per_family": {},
    }
    for f in sorted(set(fam[nm])):
        m = nm & (fam == f)
        if m.sum() < 30:
            out["per_family"][f] = {"n": int(m.sum()), "note": "under 30 - not adjudicated"}
            continue
        out["per_family"][f] = {
            "n": int(m.sum()),
            "auroc_pos_vs_neg": round(C.auroc(P[m], N[m]), 4),
            "in_headline": f not in EXCLUDED_FAMILIES,
        }
    br = kind == "bind_row"
    if br.sum() >= 30:
        out["bind_row"] = {
            "n": int(br.sum()),
            "auroc_pos_vs_neg": round(C.auroc(P[br], N[br]), 4),
            "mean_pos": round(float(P[br].mean()), 5),
            "mean_neg": round(float(N[br].mean()), 5),
            "target": BINDROW_BAR,
        }
    return out


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--arm", default="R14-H133",
                    help="arm prefix: reads models/<arm>-arm-draw<N>, writes "
                         "<arm>_antigaming_* (default the H133 arm)")
    args = ap.parse_args()
    arm_ckpt = f"{args.arm}-arm-draw{args.draw}"
    result = HERE / f"{args.arm}_antigaming_draw{args.draw}_result.json"
    evalset = HERE / f"{args.arm}_antigaming_set.parquet"

    C = _mod("c", "R15_gate_common.py")
    H108D = _mod("h108d", "R10-H108_data.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    rng = np.random.default_rng(SEED)
    rows = build_nearmiss(C, H108D, rng) + build_bindrow(C, rng)
    n_nm = sum(r["kind"] == "nearmiss" for r in rows)
    n_br = len(rows) - n_nm
    print(f"eval set: {n_nm} present-value near-miss pairs + {n_br} bind_row pairs", flush=True)
    if n_nm < 300 or n_br < 100:
        raise SystemExit(f"eval set too small to adjudicate ({n_nm} / {n_br})")

    dj = disjointness(C, rows)
    print(f"disjointness: {json.dumps(dj)}", flush=True)

    per_ckpt, banked = {}, {}
    for name in (arm_ckpt, CONTROL_CKPT):
        P, N = read_ckpt(C, name, rows)
        banked[name] = (P, N)
        per_ckpt[name] = summarise(C, rows, P, N)
        print(f"{name}: headline near-miss AUROC "
              f"{per_ckpt[name]['nearmiss_headline']['auroc_pos_vs_neg']:.4f}   "
              f"bind_row {per_ckpt[name].get('bind_row', {}).get('auroc_pos_vs_neg')}", flush=True)

    df = pl.DataFrame([{k: v for k, v in r.items() if k != "evidence"} for r in rows])
    for name, (P, N) in banked.items():
        tag = name.replace("-", "_")
        df = df.with_columns([pl.Series(f"pos__{tag}", P), pl.Series(f"neg__{tag}", N)])
    df.write_parquet(evalset)

    arm_h = per_ckpt[arm_ckpt]["nearmiss_headline"]["auroc_pos_vs_neg"]
    ctl_h = per_ckpt[CONTROL_CKPT]["nearmiss_headline"]["auroc_pos_vs_neg"]
    arm_br = per_ckpt[arm_ckpt].get("bind_row", {}).get("auroc_pos_vs_neg")
    ctl_br = per_ckpt[CONTROL_CKPT].get("bind_row", {}).get("auroc_pos_vs_neg")

    res = {
        "read": "R14-H133 ANTI-GAMING - held-out present-value near-miss + bind_row "
                "non-regression (A4 anti-gaming clause; R15-B4 binding amendment (i))",
        "draw": args.draw,
        "arm_checkpoint": arm_ckpt,
        "control_checkpoint": CONTROL_CKPT,
        "control_note": "the BANKED clean draw 1 - unseeded (pre-H126); the comparison is "
                        "arm-vs-banked-control, not init-paired",
        "seed": SEED,
        "construction": {
            "nearmiss": "TabFact test+validation ENTAILED statements whose numeral is verbatim "
                        "in the serialized evidence, corrupted by the H108 operator bank "
                        "(R10-H108_data.FAMILIES); value-checked families additionally require "
                        "the corrupted numeral to be absent from the evidence",
            "bind_row": "right column, right row label, the value of a different row - both "
                        "values printed in the evidence (the lane's own bind_row construction)",
            "per_table_cap": PER_TABLE_CAP,
            "excluded_from_headline": list(EXCLUDED_FAMILIES),
        },
        "disjointness": dj,
        "n_nearmiss": n_nm,
        "n_bind_row": n_br,
        "checkpoints": per_ckpt,
        "reads": {
            "nearmiss_headline_arm": arm_h,
            "nearmiss_headline_control": ctl_h,
            "nearmiss_headline_delta": round(arm_h - ctl_h, 4),
            "clause_arm_not_below_control": bool(arm_h >= ctl_h),
            "bind_row_arm": arm_br,
            "bind_row_control": ctl_br,
            "clause_bind_row_at_or_above_target": bool(arm_br is not None
                                                       and arm_br >= BINDROW_BAR),
        },
        "clauses": {
            "anti_gaming": "arm headline near-miss AUROC must not fall below the clean-recipe "
                           "value (A4, binding)",
            "bind_row": f"arm bind_row AUROC >= {BINDROW_BAR} (R15-B4 amendment (i), binding)",
        },
        "adjudication": "NOT ADJUDICATED HERE - the coordinator holds the verdict",
        "eval_set": evalset.name,
    }
    result.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    print(f"  near-miss headline  arm {arm_h:.4f}  control {ctl_h:.4f}  "
          f"delta {arm_h - ctl_h:+.4f}")
    print(f"  bind_row            arm {arm_br}  control {ctl_br}  (target >= {BINDROW_BAR})")
    print(f"\n  -> {result}\n  -> {evalset}", flush=True)


if __name__ == "__main__":
    main()
