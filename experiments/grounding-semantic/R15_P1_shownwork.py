"""R15-P1 addendum - does QUOTING the operands rescue a correct derivation?

The arena read shows derived-number sentences that show their working score higher
than bare assertions (finqa 0.534 at 2 absent numerals vs 0.377 at 1). That contrast
is confounded: shown-work sentences also contain operands that ARE literally in the
evidence. This probe removes the confound by holding the table, the operands and the
asserted value fixed and varying ONLY the claim template.

  bare  : "The combined {col} of {ka} and {kb} is {v}."
  shown : "The {col} of {ka} is {vi} and the {col} of {kb} is {vj}, so the combined
           {col} of {ka} and {kb} is {v}."

Both forms are scored for the CORRECT value (b) and the WRONG-OPERAND value (c).
Legal source: held-out TabFact tables, train-disjoint. No arena, no gold.
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("tp", HERE / "R15_P1_typeprobe.py")
TP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TP)

RESULT = HERE / "R15_P1_shownwork.json"
SEED = 20260811
N = 400
OPS = ["sum", "difference", "ratio", "pct_change"]


def phrase(op, col, ka, kb, v):
    return {
        "sum": f"The combined {col} of {ka} and {kb} is {v}.",
        "difference": f"The {col} of {ka} exceeds that of {kb} by {v}.",
        "ratio": f"The ratio of the {col} of {ka} to that of {kb} is {v}.",
        "pct_change": f"From {ka} to {kb}, the {col} changed by {v} percent.",
    }[op]


def build(rng):
    z = zipfile.ZipFile(TP.DATA / "dataset-tabfact.zip")
    train_ids = set(pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))["table_id"].to_list())
    held = pl.concat([pl.read_parquet(io.BytesIO(z.read(n))) for n in z.namelist()
                      if n.endswith("__test.parquet") or n.endswith("__validation.parquet")]
                     ).unique(subset=["table_id"], keep="first")
    held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
    caps, tbls, tids = (held["table_caption"].to_list(), held["table_text"].to_list(),
                        held["table_id"].to_list())
    out, seen = [], set()
    for oi in [int(o) for _ in range(4) for o in rng.permutation(len(held))]:
        if len(out) >= N * len(OPS):
            break
        hdr, body = TP.parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:TP.CHUNK_MAX]
        cands = []
        for ci in range(1, len(hdr)):
            vals = [(ri, TP.as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) >= 4 and len({v for _, v in vals}) >= 4:
                cands.append((ci, vals))
        if not cands:
            continue
        ci, vals = cands[int(rng.integers(len(cands)))]
        col = hdr[ci] or f"column {ci}"
        if any((not body[ri][0]) or TP.as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        p = [int(x) for x in rng.permutation(len(vals))[:4]]
        if len(p) < 4:
            continue
        ka, kb, kc, kd = [body[vals[x][0]][0].strip() for x in p]
        vi, vj, vk, vl = [vals[x][1] for x in p]
        for op in OPS:
            if sum(1 for q in out if q["op"] == op) >= N:
                continue
            if op == "sum":
                b, c = vi + vj, vk + vl
            elif op == "difference":
                b, c = vi - vj, vk - vl
            elif op == "ratio":
                if abs(vj) < 1e-9 or abs(vl) < 1e-9:
                    continue
                b, c = vi / vj, vk / vl
            else:
                if abs(vi) < 1e-9 or abs(vk) < 1e-9:
                    continue
                b, c = (vj - vi) / vi * 100, (vl - vk) / vk * 100
            vb, vc = TP.fmt(b), TP.fmt(c)
            if vb == vc or not TP.absent(ev, vb, vc):
                continue
            key = (tids[oi], col, op, vb)
            if key in seen:
                continue
            seen.add(key)
            li, lj = TP.fmt(vi), TP.fmt(vj)
            pre = f"The {col} of {ka} is {li} and the {col} of {kb} is {lj}, so "
            out.append({
                "op": op, "table_id": tids[oi], "evidence": ev,
                "bare_b": phrase(op, col, ka, kb, vb),
                "bare_c": phrase(op, col, ka, kb, vc),
                "shown_b": pre + phrase(op, col, ka, kb, vb)[0].lower() + phrase(op, col, ka, kb, vb)[1:],
                "shown_c": pre + phrase(op, col, ka, kb, vc)[0].lower() + phrase(op, col, ka, kb, vc)[1:],
            })
    return out


def main():
    rng = np.random.default_rng(SEED)
    q = build(rng)
    print({o: sum(1 for x in q if x["op"] == o) for o in OPS}, flush=True)
    tags = ["bare_b", "bare_c", "shown_b", "shown_c"]
    claims = [x[t] for t in tags for x in q]
    evs = [x["evidence"] for _ in tags for x in q]
    s = TP.score(claims, evs)
    n = len(q)
    S = {t: s[i * n:(i + 1) * n] for i, t in enumerate(tags)}
    M59 = TP._mod("m59", "R7-H59_cross_domain_matrix.py")
    per = {}
    for op in OPS + ["ALL"]:
        m = np.array([True] * n) if op == "ALL" else np.array([x["op"] == op for x in q])
        if m.sum() < 30:
            continue
        y = np.concatenate([np.ones(m.sum(), int), np.zeros(m.sum(), int)])
        per[op] = {
            "n": int(m.sum()),
            "bare_correct": round(float(S["bare_b"][m].mean()), 5),
            "bare_wrong": round(float(S["bare_c"][m].mean()), 5),
            "shown_correct": round(float(S["shown_b"][m].mean()), 5),
            "shown_wrong": round(float(S["shown_c"][m].mean()), 5),
            "auroc_bare": round(float(M59.auc_and_f1(
                y, np.concatenate([S["bare_b"][m], S["bare_c"][m]]))[0]), 4),
            "auroc_shown": round(float(M59.auc_and_f1(
                y, np.concatenate([S["shown_b"][m], S["shown_c"][m]]))[0]), 4),
        }
    RESULT.write_text(json.dumps({"probe": "R15-P1 shown-work addendum", "model": TP.MODEL,
                                  "seed": SEED, "per_op": per}, indent=2))
    print(json.dumps(per, indent=2))
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
