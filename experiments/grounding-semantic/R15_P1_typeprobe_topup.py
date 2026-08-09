"""R15-P1 top-up - count-aggregation and date-arithmetic types.

The main probe (R15_P1_typeprobe.py) constructed ZERO count_agg quads and only 139
date_arith quads, because both types produce small integers that are almost always
literally present somewhere in the table, and the main builder requires every asserted
value to be ABSENT. That absence rule is the right rule for a shortcut probe and the
wrong rule for these two types: a count of 3 is not "supported" because the cell 3
appears in an unrelated column.

This top-up drops the absence rule for these two types only, and therefore measures
VERIFICATION ABILITY, not shortcut presence. Reported separately for that reason.
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import pathlib

import numpy as np
import polars as pl

import importlib.util

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("tp", HERE / "R15_P1_typeprobe.py")
TP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TP)

RESULT = HERE / "R15_P1_typeprobe_topup.json"
SAMPLE = HERE / "R15_P1_typeprobe_topup_quads.parquet"
N_PER_TYPE = 500
SEED = 20260810


def build(rng):
    import io
    import zipfile

    z = zipfile.ZipFile(TP.DATA / "dataset-tabfact.zip")
    train_ids = set(pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))["table_id"].to_list())
    held = pl.concat([pl.read_parquet(io.BytesIO(z.read(n))) for n in z.namelist()
                      if n.endswith("__test.parquet") or n.endswith("__validation.parquet")]
                     ).unique(subset=["table_id"], keep="first")
    held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
    caps, tbls, tids = (held["table_caption"].to_list(), held["table_text"].to_list(),
                        held["table_id"].to_list())
    out = {"count_agg": [], "date_arith": []}
    seen = set()
    for oi in [int(o) for _ in range(4) for o in rng.permutation(len(held))]:
        if all(len(v) >= N_PER_TYPE for v in out.values()):
            break
        hdr, body = TP.parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:TP.CHUNK_MAX]
        for ci in range(1, len(hdr)):
            vals = [TP.as_num(r[ci]) for r in body]
            vals = [v for v in vals if v is not None]
            if len(vals) < 5 or len(set(vals)) < 4:
                continue
            col = hdr[ci] or f"column {ci}"
            # ---- count aggregation ----
            if len(out["count_agg"]) < N_PER_TYPE:
                thr = float(np.median(vals))
                b = sum(1 for v in vals if v > thr)
                d = sum(1 for v in vals if v < thr)          # wrong operator (direction)
                c = b + (1 if rng.integers(2) else -1)        # wrong result, same operation
                k = (tids[oi], col, "cnt")
                if b >= 1 and c >= 0 and b != c and b != d and k not in seen:
                    seen.add(k)
                    out["count_agg"].append({
                        "dtype": "count_agg", "table_id": tids[oi], "column": col, "evidence": ev,
                        "claim_a": f"The {col} of {body[0][0].strip()} is {body[0][ci].strip()}.",
                        "claim_b": f"Exactly {b} of the listed entries have a {col} greater than {TP.fmt(thr)}.",
                        "claim_c": f"Exactly {c} of the listed entries have a {col} greater than {TP.fmt(thr)}.",
                        "claim_d": f"Exactly {d} of the listed entries have a {col} greater than {TP.fmt(thr)}.",
                        "v_b": str(b), "v_c": str(c), "v_d": str(d)})
            # ---- date arithmetic (year-valued column) ----
            if len(out["date_arith"]) < N_PER_TYPE and all(
                    abs(v - round(v)) < 1e-9 and 1800 <= v <= 2035 for v in vals):
                pick = [int(p) for p in rng.permutation(len(vals))[:4]]
                if len(pick) < 4:
                    continue
                yi, yj, yk, yl = [vals[p] for p in pick]
                ki = body[pick[0]][0].strip()
                kj = body[pick[1]][0].strip()
                b, c, d = abs(yi - yj), abs(yk - yl), yi + yj
                k2 = (tids[oi], col, TP.fmt(b), ki, kj)
                if b == c or b == d or not ki or not kj or k2 in seen:
                    continue
                seen.add(k2)
                T = f"The {col} of {ki} and that of {kj} are {{}} years apart."
                out["date_arith"].append({
                    "dtype": "date_arith", "table_id": tids[oi], "column": col, "evidence": ev,
                    "claim_a": f"The {col} of {ki} is {TP.fmt(yi)}.",
                    "claim_b": T.format(TP.fmt(b)), "claim_c": T.format(TP.fmt(c)),
                    "claim_d": T.format(TP.fmt(d)),
                    "v_b": TP.fmt(b), "v_c": TP.fmt(c), "v_d": TP.fmt(d)})
            break
    return out


def main():
    rng = np.random.default_rng(SEED)
    built = build(rng)
    quads = built["count_agg"] + built["date_arith"]
    for k, v in built.items():
        print(f"  {k:12s} {len(v)}", flush=True)
    claims, evs = [], []
    for tag in ("claim_a", "claim_b", "claim_c", "claim_d"):
        claims += [q[tag] for q in quads]
        evs += [q["evidence"] for q in quads]
    s = TP.score(claims, evs)
    n = len(quads)
    sa, sb, sc, sd = s[:n], s[n:2 * n], s[2 * n:3 * n], s[3 * n:]
    M59 = TP._mod("m59", "R7-H59_cross_domain_matrix.py")
    pl.DataFrame([{**{k: v for k, v in q.items() if k != "evidence"},
                   "score_a": float(sa[i]), "score_b": float(sb[i]),
                   "score_c": float(sc[i]), "score_d": float(sd[i])}
                  for i, q in enumerate(quads)]).write_parquet(SAMPLE)
    per = {}
    for t in ("count_agg", "date_arith"):
        m = np.array([q["dtype"] == t for q in quads])
        if m.sum() < 30:
            per[t] = {"n": int(m.sum()), "note": "under 30 - not adjudicated"}
            continue
        A, B, C, D = sa[m], sb[m], sc[m], sd[m]
        y = np.concatenate([np.ones(m.sum(), int), np.zeros(m.sum(), int)])
        per[t] = {"n": int(m.sum()),
                  "mean_a_verbatim": round(float(A.mean()), 5),
                  "mean_b_correct": round(float(B.mean()), 5),
                  "mean_c_wrong_result": round(float(C.mean()), 5),
                  "mean_d_wrong_operator": round(float(D.mean()), 5),
                  "gap_a_minus_b": round(float(A.mean() - B.mean()), 5),
                  "auroc_b_vs_c": round(float(M59.auc_and_f1(y, np.concatenate([B, C]))[0]), 4),
                  "auroc_b_vs_d": round(float(M59.auc_and_f1(y, np.concatenate([B, D]))[0]), 4),
                  "auroc_a_vs_b": round(float(M59.auc_and_f1(y, np.concatenate([A, B]))[0]), 4),
                  "frac_b_above_half": round(float((B > 0.5).mean()), 4)}
    res = {"probe": "R15-P1 top-up (count_agg, date_arith) - ABSENCE RULE DROPPED",
           "model": TP.MODEL, "seed": SEED, "n_quads": n, "per_type": per,
           "caveat": "asserted values may appear elsewhere in the table; this measures "
                     "verification ability, not shortcut presence",
           "sample": SAMPLE.name}
    RESULT.write_text(json.dumps(res, indent=2))
    print(json.dumps(per, indent=2))
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
