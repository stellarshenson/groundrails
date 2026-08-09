"""R15-L1 - PRESENT-VALUE binding and ordering probe on frozen H105 draw 1.

Pre-arms the killgate for L1-C1. P4 section 9 flags operand binding as the single
unmeasured quantity that its prescription P4-1 nonetheless spends 35% of the lane
budget on. P1 infers "sharp key-value binding" indirectly from the scale/unit arm
(0.8755) where the correct value also preserves the source cell's digit string, so
binding and digit-copying are confounded there. This probe separates them, using
claims in which EVERY asserted number is verbatim present in the evidence.

Four arms, same evidence, same template, byte-identical except the swapped item:

  bind_row  - "The {col} of {ka} is {v}."   correct v = ka's cell; wrong v = kb's cell
              (right column, wrong ROW - value present, binding wrong)
  bind_col  - "The {colx} of {ka} is {v}."  correct v = ka's colx cell; wrong v = ka's
              coly cell (right row, wrong COLUMN - E4 resp-200's cross-line-item shape)
  compare   - "The {col} of {ka} is greater than the {col} of {kb}."  vs the reversed
              claim (NO computation required; both operands printed)
  absent_ctl- correct verbatim cell vs a fabricated value ABSENT from the table
              (anchor - the competence P1 measures at AUROC 0.81-0.99)

Data: TabFact test+validation, table_id-disjoint from the train split used by the
clean mix and by R10-H108_data.tabfact_positives(). Zero arena, zero gold.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
     uv run python experiments/grounding-semantic/R15_L1_bindprobe.py
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
CKPT = os.environ.get("R15_CKPT", "R9-H105-mmbert-dann-clean")
TAG = "" if CKPT == "R9-H105-mmbert-dann-clean" else "_" + CKPT
RESULT = HERE / f"R15_gate_L1_binding{TAG}.json"
SAMPLE = HERE / f"R15_L1_bindprobe_pairs{TAG}.parquet"

MODEL = str(ROOT / "models" / CKPT)
CHUNK_MAX = 1500
MAX_LEN = 512
BATCH = 64
SEED = 20260809
N_PER_ARM = 600
ARMS = ["bind_row", "bind_col", "compare", "absent_ctl"]

NUM = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")


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


def numeric_cols(hdr, body):
    out = []
    for ci in range(1, len(hdr)):
        vals = [(ri, as_num(r[ci])) for ri, r in enumerate(body)]
        vals = [(ri, v) for ri, v in vals if v is not None]
        if len(vals) >= 3 and len({v for _, v in vals}) >= 3:
            out.append((ci, vals))
    return out


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

    caps = held["table_caption"].to_list()
    tbls = held["table_text"].to_list()
    tids = held["table_id"].to_list()
    order = [int(o) for _ in range(4) for o in rng.permutation(len(held))]

    out = {a: [] for a in ARMS}
    seen = {a: set() for a in ARMS}
    per_table = {}
    for oi in order:
        if all(len(out[a]) >= N_PER_ARM for a in ARMS):
            break
        if per_table.get(oi, 0) >= 4:
            continue
        hdr, body = parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_MAX]
        cols = numeric_cols(hdr, body)
        if not cols:
            continue
        ci, vals = cols[int(rng.integers(len(cols)))]
        col = hdr[ci] or f"column {ci}"
        # row labels must be non-numeric text keys
        if any((not body[ri][0]) or as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        pick = [int(p) for p in rng.permutation(len(vals))[:2]]
        (ri_a, vi), (ri_b, vj) = vals[pick[0]], vals[pick[1]]
        ka = body[ri_a][0].strip()
        kb = body[ri_b][0].strip()
        if not ka or not kb or ka == kb or abs(vi - vj) < 1e-9:
            continue
        la, lb = fmt(vi), fmt(vj)
        if la not in ev or lb not in ev:
            continue
        made = False

        # --- bind_row: right column, wrong row -------------------------------
        # the wrong value must not appear anywhere in ka's own row
        if la != lb and lb not in " | ".join(body[ri_a]) and len(out["bind_row"]) < N_PER_ARM:
            key = (tids[oi], col, ka, la, lb)
            if key not in seen["bind_row"]:
                seen["bind_row"].add(key)
                T = f"The {col} of {ka} is {{}}."
                out["bind_row"].append({
                    "arm": "bind_row", "table_id": tids[oi], "column": col,
                    "evidence": ev, "claim_pos": T.format(la), "claim_neg": T.format(lb),
                    "v_pos": la, "v_neg": lb, "both_present": True})
                made = True

        # --- bind_col: right row, wrong column -------------------------------
        if len(cols) >= 2 and len(out["bind_col"]) < N_PER_ARM:
            alts = [(c2, v2) for c2, v2 in cols if c2 != ci]
            c2, vals2 = alts[int(rng.integers(len(alts)))]
            m2 = {ri: v for ri, v in vals2}
            if ri_a in m2:
                vx, vy = vi, m2[ri_a]
                lx, ly = fmt(vx), fmt(vy)
                colx = col
                coly = hdr[c2] or f"column {c2}"
                if lx != ly and colx != coly and lx in ev and ly in ev:
                    key = (tids[oi], colx, ka, lx, ly)
                    if key not in seen["bind_col"]:
                        seen["bind_col"].add(key)
                        T = f"The {colx} of {ka} is {{}}."
                        out["bind_col"].append({
                            "arm": "bind_col", "table_id": tids[oi], "column": colx,
                            "evidence": ev, "claim_pos": T.format(lx), "claim_neg": T.format(ly),
                            "v_pos": lx, "v_neg": ly, "both_present": True})
                        made = True

        # --- compare: no computation, both operands printed ------------------
        if len(out["compare"]) < N_PER_ARM:
            hi, lo = (ka, kb) if vi > vj else (kb, ka)
            key = (tids[oi], col, hi, lo)
            if key not in seen["compare"]:
                seen["compare"].add(key)
                out["compare"].append({
                    "arm": "compare", "table_id": tids[oi], "column": col, "evidence": ev,
                    "claim_pos": f"The {col} of {hi} is greater than the {col} of {lo}.",
                    "claim_neg": f"The {col} of {lo} is greater than the {col} of {hi}.",
                    "v_pos": fmt(max(vi, vj)), "v_neg": fmt(min(vi, vj)), "both_present": True})
                made = True

        # --- absent_ctl: verbatim cell vs a fabricated absent value ----------
        if len(out["absent_ctl"]) < N_PER_ARM:
            for _ in range(12):
                cand = vi + float(rng.integers(1, 40)) * (10 ** float(rng.integers(0, 2)))
                lc = fmt(cand)
                if lc != la and lc not in ev:
                    key = (tids[oi], col, ka, la, lc)
                    if key not in seen["absent_ctl"]:
                        seen["absent_ctl"].add(key)
                        T = f"The {col} of {ka} is {{}}."
                        out["absent_ctl"].append({
                            "arm": "absent_ctl", "table_id": tids[oi], "column": col,
                            "evidence": ev, "claim_pos": T.format(la), "claim_neg": T.format(lc),
                            "v_pos": la, "v_neg": lc, "both_present": False})
                        made = True
                    break
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
    for a in ARMS:
        print(f"  {a:12s} {len(built[a])}", flush=True)
    pairs = [p for a in ARMS for p in built[a]]
    if not pairs:
        raise SystemExit("nothing constructible")

    claims = [p["claim_pos"] for p in pairs] + [p["claim_neg"] for p in pairs]
    evs = [p["evidence"] for p in pairs] * 2
    s = score(claims, evs)
    n = len(pairs)
    sp, sn = s[:n], s[n:]

    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")

    df = pl.DataFrame([
        {**{k: v for k, v in p.items() if k != "evidence"},
         "score_pos": float(sp[i]), "score_neg": float(sn[i])}
        for i, p in enumerate(pairs)
    ])
    df.write_parquet(SAMPLE)

    per = {}
    for a in ARMS:
        m = np.array([p["arm"] == a for p in pairs])
        if m.sum() < 30:
            per[a] = {"n": int(m.sum()), "note": "under 30 - not adjudicated"}
            continue
        P, N = sp[m], sn[m]
        y = np.concatenate([np.ones(m.sum(), int), np.zeros(m.sum(), int)])
        per[a] = {
            "n": int(m.sum()),
            "mean_pos": round(float(P.mean()), 5),
            "mean_neg": round(float(N.mean()), 5),
            "auroc_pos_vs_neg": round(float(M59.auc_and_f1(y, np.concatenate([P, N]))[0]), 4),
            "frac_pos_higher": round(float((P > N).mean()), 4),
            "frac_pos_above_half": round(float((P > 0.5).mean()), 4),
            "frac_neg_above_half": round(float((N > 0.5).mean()), 4),
            "distinct_tables": int(len({p["table_id"] for p, mm in zip(pairs, m) if mm})),
        }

    res = {"probe": "R15-L1 present-value binding and ordering probe", "model": MODEL,
           "data": "TabFact test+validation, train-disjoint; no arena, no gold",
           "seed": SEED, "n_per_arm_target": N_PER_ARM, "n_pairs": n,
           "per_arm": per, "sample": SAMPLE.name}
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 92)
    print(f"{'arm':12s} {'n':>5s} {'pos':>8s} {'neg':>8s} {'AUROC':>8s} {'pos>neg':>8s} {'tables':>7s}")
    for a in ARMS:
        p = per[a]
        if "mean_pos" not in p:
            print(f"{a:12s} {p['n']:5d}  {p.get('note', '')}")
            continue
        print(f"{a:12s} {p['n']:5d} {p['mean_pos']:8.4f} {p['mean_neg']:8.4f} "
              f"{p['auroc_pos_vs_neg']:8.4f} {p['frac_pos_higher']:8.4f} {p['distinct_tables']:7d}")
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
