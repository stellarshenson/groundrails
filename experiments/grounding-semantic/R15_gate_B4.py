"""R15-B4 BLOCKING probe (R15-H138) == R15-B5 arm 4 - controlled cross-header binding.

The header hole is measured at bind_col AUROC 0.5179 (H105 d1) and 0.5276 (H108
lane d1) against a negative drawn from ANY other numeric column of the same row.
That negative is free to differ in digit length and in magnitude, so the hole
could be a magnitude artefact rather than a header hole.

  Re-run the bind_col arm with negatives drawn from DIGIT-LENGTH-MATCHED and
  MAGNITUDE-MATCHED columns.
  KILL the sub-block if the controlled probe reads above 0.60 - the 15% of lane
  rows returns to B1's derivation core.

Free CPU clause: certify on a 500-tuple sample that digit-length-matched
cross-column pairs are constructible at the sub-block's size.
  KILL below 60% constructibility.

Frozen weights, held-out TabFact test+validation, zero arena, zero gold.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import math
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R15_gate_B4_result.json"
SAMPLE = HERE / "R15_gate_B4_pairs.parquet"

CKPTS = ["R9-H105-mmbert-dann-clean", "R10-H108-lane-draw1"]
SEED = 20260812
N_PAIRS = 600
N_CENSUS = 500
BAR_AUROC = 0.60
BAR_CONSTRUCT = 0.60
PER_TABLE_CAP = 2  # A4's per-table cap
MAG_RATIO = 2.0  # magnitude match: the two cells within a factor of 2


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def digits(s):
    return sum(ch.isdigit() for ch in s)


def numeric_cols(C, hdr, body):
    out = []
    for ci in range(1, len(hdr)):
        vals = [(ri, C.as_num(r[ci])) for ri, r in enumerate(body)]
        vals = [(ri, v) for ri, v in vals if v is not None]
        if len(vals) >= 3 and len({v for _, v in vals}) >= 3:
            out.append((ci, vals))
    return out


def tuples_of(C, hdr, body, ev, cols):
    """Every cross-header (row, colx, coly) tuple the table admits, with its match status."""
    out = []
    for ci, vals in cols:
        colx = hdr[ci] or f"column {ci}"
        if any((not body[ri][0]) or C.as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        for ri_a, vx in vals:
            ka = body[ri_a][0].strip()
            lx = C.fmt(vx)
            if not ka or lx not in ev:
                continue
            for c2, vals2 in cols:
                if c2 == ci:
                    continue
                m2 = {ri: v for ri, v in vals2}
                if ri_a not in m2:
                    continue
                vy = m2[ri_a]
                ly = C.fmt(vy)
                coly = hdr[c2] or f"column {c2}"
                if ly == lx or coly == colx or ly not in ev:
                    continue
                lenm = digits(ly) == digits(lx)
                magm = (abs(vx) > 1e-9 and abs(vy) > 1e-9
                        and abs(math.log10(abs(vy)) - math.log10(abs(vx))) <= math.log10(MAG_RATIO))
                out.append({"ka": ka, "colx": colx, "coly": coly, "lx": lx, "ly": ly,
                            "lenm": bool(lenm), "bothm": bool(lenm and magm)})
    return out


def build(C, rng):
    """Controlled bind_col pairs plus the two constructibility censuses."""
    caps, tbls, tids = C.held_tabfact()
    print(f"held-out tables: {len(tbls)}", flush=True)
    order = [int(o) for o in rng.permutation(len(tbls))]

    pairs, seen = [], set()
    census = {"tables_admitting_a_cross_header_tuple": 0,
              "tables_with_a_len_matched_tuple": 0,
              "tables_with_a_len_and_mag_matched_tuple": 0,
              "random_tuples_examined": 0,
              "random_tuples_len_matched": 0,
              "random_tuples_len_and_mag_matched": 0}
    for oi in order:
        hdr, body = C.parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:C.CHUNK_MAX]
        cols = numeric_cols(C, hdr, body)
        if len(cols) < 2:
            continue
        tup = tuples_of(C, hdr, body, ev, cols)
        if not tup:
            continue
        census["tables_admitting_a_cross_header_tuple"] += 1
        census["tables_with_a_len_matched_tuple"] += int(any(t["lenm"] for t in tup))
        census["tables_with_a_len_and_mag_matched_tuple"] += int(any(t["bothm"] for t in tup))
        if census["random_tuples_examined"] < N_CENSUS:
            r = tup[int(rng.integers(len(tup)))]
            census["random_tuples_examined"] += 1
            census["random_tuples_len_matched"] += int(r["lenm"])
            census["random_tuples_len_and_mag_matched"] += int(r["bothm"])

        ok = [t for t in tup if t["bothm"]]
        if not ok or len(pairs) >= N_PAIRS:
            continue
        taken = 0
        for i in [int(k) for k in rng.permutation(len(ok))]:
            if taken >= PER_TABLE_CAP or len(pairs) >= N_PAIRS:
                break
            t = ok[i]
            key = (tids[oi], t["colx"], t["ka"], t["lx"], t["ly"])
            if key in seen:
                continue
            seen.add(key)
            taken += 1
            T = f"The {t['colx']} of {t['ka']} is {{}}."
            pairs.append({"table_id": tids[oi], "column": t["colx"], "alt_column": t["coly"],
                          "evidence": ev, "claim_pos": T.format(t["lx"]),
                          "claim_neg": T.format(t["ly"]), "v_pos": t["lx"], "v_neg": t["ly"],
                          "digits_pos": digits(t["lx"]), "digits_neg": digits(t["ly"])})
    return pairs, census


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pairs, census = build(C, np.random.default_rng(SEED))
    print(f"controlled bind_col pairs: {len(pairs)}   census: {census}", flush=True)
    if len(pairs) < 100:
        raise SystemExit("too few controlled pairs constructible to adjudicate")

    claims = [p["claim_pos"] for p in pairs] + [p["claim_neg"] for p in pairs]
    evs = [p["evidence"] for p in pairs] * 2
    n = len(pairs)

    per_ckpt, cols = {}, {}
    for name in CKPTS:
        tok, trunk, head = C.load_ckpt(name)
        s = C.score(tok, trunk, head, claims, evs)
        del trunk, head
        torch.cuda.empty_cache()
        P, N = s[:n], s[n:]
        cols[name] = (P, N)
        per_ckpt[name] = {
            "n": n,
            "mean_pos": round(float(P.mean()), 5),
            "mean_neg": round(float(N.mean()), 5),
            "auroc_pos_vs_neg": round(C.auroc(P, N), 4),
            "frac_pos_higher": round(float((P > N).mean()), 4),
            "distinct_tables": len({p["table_id"] for p in pairs}),
        }
        print(name, json.dumps(per_ckpt[name]), flush=True)

    df = pl.DataFrame([{k: v for k, v in p.items() if k != "evidence"} for p in pairs])
    for name, (P, N) in cols.items():
        tag = name.replace("-", "_")
        df = df.with_columns([pl.Series(f"pos__{tag}", P), pl.Series(f"neg__{tag}", N)])
    df.write_parquet(SAMPLE)

    worst = max(per_ckpt[k]["auroc_pos_vs_neg"] for k in per_ckpt)
    denom = max(census["tables_admitting_a_cross_header_tuple"], 1)
    construct_rate = census["tables_with_a_len_matched_tuple"] / denom
    strict_rate = (census["random_tuples_len_matched"] / census["random_tuples_examined"]
                   if census["random_tuples_examined"] else 0.0)
    clauses = []
    if worst > BAR_AUROC:
        clauses.append(f"KILL - controlled bind_col AUROC {worst:.4f} > {BAR_AUROC}: the hole is a "
                       "magnitude artefact, not a header hole")
    if construct_rate < BAR_CONSTRUCT:
        clauses.append(f"KILL - digit-length-matched constructibility {construct_rate:.4f} < "
                       f"{BAR_CONSTRUCT}")
    verdict = "KILL" if clauses else "LICENSE"

    res = {
        "gate": "R15-B4 BLOCKING probe (R15-H138) == R15-B5 arm 4 - controlled cross-header binding",
        "data": "TabFact test+validation, table_id-disjoint from every train split; byte-identical "
                "claim templates; zero arena, zero gold",
        "implementation_choices": [
            "digit-length match = equal count of digit characters in the two formatted cell "
            "values; magnitude match = the two cells within a factor of "
            f"{MAG_RATIO} (|log10 ratio| <= log10 {MAG_RATIO}). Both are required of the negative, "
            "per the blocking clause's 'digit-length-matched AND magnitude-matched'.",
            "The free CPU constructibility clause is adjudicated on DIGIT-LENGTH matching alone, "
            "which is how the synthesis words it; the joint length+magnitude rate is reported "
            "beside it.",
            "'constructible at the sub-block's size' is adjudicated PER ADMITTING TABLE - a real "
            "build enumerates rows and column pairs freely, so the supply question is whether a "
            "table yields any digit-length-matched cross-header tuple. The stricter per-random-"
            "tuple rate (one uniformly drawn tuple per table) is reported beside it.",
            "The probe is run on BOTH checkpoints the uncontrolled licensing clause was measured "
            "on, and the KILL is taken on the worse (higher) of the two.",
        ],
        "seed": SEED, "n_pairs": n,
        "banked_uncontrolled_reference": {"R9-H105-mmbert-dann-clean": 0.5179,
                                          "R10-H108-lane-draw1": 0.5276},
        "checkpoints": per_ckpt,
        "worst_controlled_auroc": worst,
        "constructibility_census": {
            **census,
            "adjudicated_rate_per_admitting_table": round(construct_rate, 4),
            "adjudicated_rate_len_and_magnitude": round(
                census["tables_with_a_len_and_mag_matched_tuple"] / denom, 4),
            "strict_rate_per_random_tuple": round(strict_rate, 4),
            "strict_rate_len_and_magnitude": round(
                census["random_tuples_len_and_mag_matched"]
                / max(census["random_tuples_examined"], 1), 4),
            "bar": f"KILL below {BAR_CONSTRUCT}",
        },
        "bar": f"KILL the sub-block if the controlled probe reads above {BAR_AUROC} on either "
               f"checkpoint, or if digit-length-matched constructibility is below {BAR_CONSTRUCT}",
        "verdict": verdict,
        "clauses_fired": clauses,
        "gates_downstream": "R15-B4's 15% sub-block of the H133 lane row budget; on KILL the 15% "
                            "returns to B1's derivation core and the lane is strictly simpler",
        "sample": SAMPLE.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    print(f"  worst controlled bind_col AUROC {worst:.4f} (need <= {BAR_AUROC})   "
          f"constructibility {construct_rate:.4f} (need >= {BAR_CONSTRUCT})")
    print(f"\n  VERDICT: {verdict}\n  -> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
