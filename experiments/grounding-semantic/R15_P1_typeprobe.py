"""R15-P1 - per-derivation-type failure probe on frozen H105 draw 1.

Extends the R14-H133 shortcut probe (which built SUM triples only) to eight
derivation types. Per type and per table, four claims over the SAME evidence:

  (a) verbatim   - the asserted value is a table cell
  (b) correct    - arithmetically correct, ABSENT from the table
  (c) wrong-oper and - same template/operation, value computed from DIFFERENT operands
  (d) wrong-operator - same operands named, value computed by a DIFFERENT operation

Reported per type: mean scores, AUROC(b vs c), AUROC(b vs d), AUROC(a vs b).
Data: TabFact test+validation tables, table_id-disjoint from the train split used
by the clean mix and by R10-H108_data.tabfact_positives(). Zero arena, zero gold.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
     uv run python experiments/grounding-semantic/R15_P1_typeprobe.py
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
RESULT = HERE / "R15_P1_typeprobe.json"
SAMPLE = HERE / "R15_P1_typeprobe_quads.parquet"

MODEL = str(ROOT / "models" / "R9-H105-mmbert-dann-clean")
CHUNK_MAX = 1500
MAX_LEN = 512
BATCH = 64
SEED = 20260809
N_PER_TYPE = 500

NUM = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
TYPES = ["sum", "difference", "mean", "ratio", "pct_change", "product",
         "scale_unit", "rounding", "count_agg", "date_arith"]


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse(tbl):
    rows = [r.split("#") for r in tbl.replace("\r\n", "\n").strip().split("\n") if r.strip()]
    rows = [[c.strip() for c in r] for r in rows]
    if len(rows) < 4:
        return None, None
    w = len(rows[0])
    return rows[0], [r for r in rows[1:] if len(r) == w]


def as_num(s):
    s = s.strip()
    if not NUM.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def absent(ev, *vals):
    return all(v not in ev for v in vals)


def make(dtype, col, keys, vs, ev, rng):
    """Return (claim_a, claim_b, claim_c, claim_d, v_b, v_c, v_d) or None.

    keys/vs are 4 (key, value) draws from the same numeric column, indices 0..3
    (i, j = the named operands; k, l = the alternative operands).
    """
    ka, kb, kc, kd = keys
    vi, vj, vk, vl = vs
    a_lit = fmt(vi)
    A = f"The {col} of {ka} is {a_lit}."

    if dtype == "sum":
        b, c, d = vi + vj, vk + vl, vi - vj
        T = f"The combined {col} of {ka} and {kb} is {{}}."
    elif dtype == "difference":
        b, c, d = vi - vj, vk - vl, vi + vj
        T = f"The {col} of {ka} exceeds that of {kb} by {{}}."
    elif dtype == "mean":
        b, c, d = (vi + vj) / 2, (vk + vl) / 2, vi + vj
        T = f"The average {col} of {ka} and {kb} is {{}}."
    elif dtype == "ratio":
        if abs(vj) < 1e-9 or abs(vl) < 1e-9:
            return None
        b, c, d = vi / vj, vk / vl, vi - vj
        T = f"The ratio of the {col} of {ka} to that of {kb} is {{}}."
    elif dtype == "pct_change":
        if abs(vi) < 1e-9 or abs(vk) < 1e-9:
            return None
        b, c, d = (vj - vi) / vi * 100, (vl - vk) / vk * 100, vj - vi
        T = f"From {ka} to {kb}, the {col} changed by {{}} percent."
    elif dtype == "product":
        b, c, d = vi * vj, vk * vl, vi + vj
        T = f"The product of the {col} of {ka} and {kb} is {{}}."
    elif dtype == "scale_unit":
        b, c, d = vi * 1000, vk * 1000, vi / 1000
        T = f"Expressed in units of one thousandth, the {col} of {ka} is {{}}."
    elif dtype == "rounding":
        step = 10.0 if abs(vi) >= 100 else 1.0
        b = round(vi / step) * step
        c = round(vk / step) * step
        d = vi + step
        if abs(b - vi) < 1e-9:
            return None
        T = f"The {col} of {ka} is approximately {{}}."
    elif dtype == "count_agg":
        thr = float(np.median(vs))
        b = float(sum(1 for v in vs if v > thr))
        c = float(sum(1 for v in vs if v > min(vs)))
        d = thr
        T = f"Exactly {{}} of the listed entries have a {col} greater than {fmt(thr)}."
        A = f"The {col} of {ka} is {a_lit}."
    elif dtype == "date_arith":
        if not all(abs(v - round(v)) < 1e-9 and 1800 <= v <= 2035 for v in vs):
            return None
        b, c, d = abs(vi - vj), abs(vk - vl), vi + vj
        T = f"The {col} of {ka} and that of {kb} are {{}} years apart."
    else:
        raise ValueError(dtype)

    vb, vc, vd = fmt(b), fmt(c), fmt(d)
    if vb == vc or vb == vd:
        return None
    if not absent(ev, vb, vc, vd):
        return None
    if a_lit not in ev:
        return None
    return A, T.format(vb), T.format(vc), T.format(vd), vb, vc, vd


def build(rng):
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
    ]).unique(subset=["table_id"], keep="first")
    held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
    print(f"held-out tables: {len(held)}", flush=True)

    caps, tbls, tids = (held["table_caption"].to_list(), held["table_text"].to_list(),
                        held["table_id"].to_list())
    order = [int(o) for _ in range(6) for o in rng.permutation(len(held))]

    out = {t: [] for t in TYPES}
    seen = {t: set() for t in TYPES}
    per_table = {}
    for oi in order:
        if all(len(out[t]) >= N_PER_TYPE for t in TYPES):
            break
        if per_table.get(oi, 0) >= 6:
            continue
        hdr, body = parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_MAX]
        cand = []
        for ci in range(1, len(hdr)):
            vals = [(ri, as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) >= 4 and len({v for _, v in vals}) >= 4:
                cand.append((ci, vals))
        if not cand:
            continue
        ci, vals = cand[int(rng.integers(len(cand)))]
        col = hdr[ci] or f"column {ci}"
        if any((not body[ri][0]) or as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        pick = [int(p) for p in rng.permutation(len(vals))[:4]]
        if len(pick) < 4:
            continue
        keys = [body[vals[p][0]][0].strip() for p in pick]
        vs = [vals[p][1] for p in pick]
        made = False
        for t in TYPES:
            if len(out[t]) >= N_PER_TYPE:
                continue
            r = make(t, col, keys, vs, ev, rng)
            if r is None:
                continue
            key = (tids[oi], col, keys[0], keys[1], r[4])
            if key in seen[t]:
                continue
            seen[t].add(key)
            out[t].append({"dtype": t, "table_id": tids[oi], "column": col, "evidence": ev,
                           "claim_a": r[0], "claim_b": r[1], "claim_c": r[2], "claim_d": r[3],
                           "v_b": r[4], "v_c": r[5], "v_d": r[6]})
            made = True
        if made:
            per_table[oi] = per_table.get(oi, 0) + 1
    return out


def score(claims, evs):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu",
                       weights_only=False)
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    head = head.cuda().eval()
    s = np.zeros(len(claims), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j:j + BATCH], evs[j:j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            s[j:j + BATCH] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    del trunk, head
    torch.cuda.empty_cache()
    return s


def main():
    rng = np.random.default_rng(SEED)
    built = build(rng)
    for t in TYPES:
        print(f"  {t:12s} {len(built[t])}", flush=True)
    quads = [q for t in TYPES for q in built[t]]
    if not quads:
        raise SystemExit("nothing constructible")

    claims, evs = [], []
    for tag in ("claim_a", "claim_b", "claim_c", "claim_d"):
        claims += [q[tag] for q in quads]
        evs += [q["evidence"] for q in quads]
    s = score(claims, evs)
    n = len(quads)
    sa, sb, sc, sd = s[:n], s[n:2 * n], s[2 * n:3 * n], s[3 * n:]

    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")

    df = pl.DataFrame([
        {**{k: v for k, v in q.items() if k != "evidence"},
         "score_a": float(sa[i]), "score_b": float(sb[i]),
         "score_c": float(sc[i]), "score_d": float(sd[i])}
        for i, q in enumerate(quads)
    ])
    df.write_parquet(SAMPLE)

    per = {}
    for t in TYPES:
        m = np.array([q["dtype"] == t for q in quads])
        if m.sum() < 30:
            per[t] = {"n": int(m.sum()), "note": "under 30 - not adjudicated"}
            continue
        A, B, C, D = sa[m], sb[m], sc[m], sd[m]
        y = np.concatenate([np.ones(m.sum(), int), np.zeros(m.sum(), int)])
        per[t] = {
            "n": int(m.sum()),
            "mean_a_verbatim": round(float(A.mean()), 5),
            "mean_b_correct": round(float(B.mean()), 5),
            "mean_c_wrong_operand": round(float(C.mean()), 5),
            "mean_d_wrong_operator": round(float(D.mean()), 5),
            "gap_a_minus_b": round(float(A.mean() - B.mean()), 5),
            "auroc_b_vs_c": round(float(M59.auc_and_f1(y, np.concatenate([B, C]))[0]), 4),
            "auroc_b_vs_d": round(float(M59.auc_and_f1(y, np.concatenate([B, D]))[0]), 4),
            "auroc_a_vs_b": round(float(M59.auc_and_f1(y, np.concatenate([A, B]))[0]), 4),
            "frac_b_above_half": round(float((B > 0.5).mean()), 4),
        }

    res = {"probe": "R15-P1 per-derivation-type failure probe", "model": MODEL,
           "data": "TabFact test+validation, train-disjoint; no arena, no gold",
           "seed": SEED, "n_per_type_target": N_PER_TYPE, "n_quads": n,
           "per_type": per, "sample": SAMPLE.name}
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 100)
    print(f"{'type':12s} {'n':>5s} {'a':>7s} {'b':>7s} {'c':>7s} {'d':>7s} "
          f"{'AUC b|c':>8s} {'AUC b|d':>8s} {'AUC a|b':>8s}")
    for t in TYPES:
        p = per[t]
        if "mean_a_verbatim" not in p:
            print(f"{t:12s} {p['n']:5d}  {p.get('note','')}")
            continue
        print(f"{t:12s} {p['n']:5d} {p['mean_a_verbatim']:7.4f} {p['mean_b_correct']:7.4f} "
              f"{p['mean_c_wrong_operand']:7.4f} {p['mean_d_wrong_operator']:7.4f} "
              f"{p['auroc_b_vs_c']:8.4f} {p['auroc_b_vs_d']:8.4f} {p['auroc_a_vs_b']:8.4f}")
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
