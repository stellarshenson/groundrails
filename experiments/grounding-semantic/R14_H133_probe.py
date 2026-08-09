"""R14-H133 (R14-A4 kill-gate) - shortcut-presence probe on frozen H105 draw 1.

2,000 triples over HELD-OUT TabFact tables - the test and validation splits,
`table_id`-disjoint from the train split that built the clean mix's tabfact
group, the H108 lane's `tabfact_positives()` and every corruption family. Zero
arena, zero gold.

Per triple, over the same table:
  (a) verbatim-cell claim          - the asserted value appears in the table
  (b) correctly derived value      - arithmetically correct, ABSENT from the table
  (c) wrong-operand derived value  - same template and operands as (b) in the
                                     claim text, but the asserted value is the
                                     result computed from a DIFFERENT pair of
                                     operands; also absent from the table

  LICENSE  mean score(a) - mean score(b) >= 0.20  AND  AUROC(b vs c) <= 0.60
  KILL     AUROC(b vs c) > 0.60  (the model already discriminates derivations)
           or score(a) - score(b) < 0.20 (the literal-presence prior did not
           transfer to the operating point)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R14_H133_probe.py
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
RESULT = HERE / "R14_gate_H133_probe.json"
SAMPLE = HERE / "R14_H133_triples.parquet"

MODEL = str(ROOT / "models" / "R9-H105-mmbert-dann-clean")
CHUNK_MAX = 1500
MAX_LEN = 512
BATCH = 64
SEED = 20260809
N_TRIPLES = 2000

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
    body = [r for r in rows[1:] if len(r) == w]
    return rows[0], body


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


def build(n_target, rng):
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
    print(f"held-out tables: {len(held)} (train ids excluded: {len(train_ids)})", flush=True)

    order = rng.permutation(len(held))
    caps = held["table_caption"].to_list()
    tbls = held["table_text"].to_list()
    tids = held["table_id"].to_list()

    out, seen, per_table = [], set(), {}
    for oi in [int(o) for _ in range(4) for o in order]:
        if len(out) >= n_target:
            break
        if per_table.get(oi, 0) >= 3:
            continue
        cap, tbl, tid = caps[oi], tbls[oi], tids[oi]
        hdr, body = parse(tbl)
        if hdr is None:
            continue
        ev = f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_MAX]
        # numeric column with >= 4 usable rows, and a non-numeric key column
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
        keys = [body[ri][0] for ri, _ in vals]
        if any(not k or as_num(k) is not None for k in keys):
            continue
        pick = rng.permutation(len(vals))[:4]
        if len(pick) < 4:
            continue
        (i, j, k, l) = [int(p) for p in pick]
        ri_i, vi = vals[i]
        ri_j, vj = vals[j]
        _, vk = vals[k]
        _, vl = vals[l]
        s_ij, s_kl = vi + vj, vk + vl
        if abs(s_ij - s_kl) < 1e-9:
            continue
        a_val = body[ri_i][ci].strip()
        b_val, c_val = fmt(s_ij), fmt(s_kl)
        if b_val in ev or c_val in ev:
            continue
        ka, kb = body[ri_i][0].strip(), body[ri_j][0].strip()
        key = (tid, col, ka, kb, b_val, c_val)
        if key in seen:
            continue
        seen.add(key)
        per_table[oi] = per_table.get(oi, 0) + 1
        out.append({
            "table_id": tid, "evidence": ev, "column": col,
            "claim_a": f"The {col} of {ka} is {a_val}.",
            "claim_b": f"The combined {col} of {ka} and {kb} is {b_val}.",
            "claim_c": f"The combined {col} of {ka} and {kb} is {c_val}.",
            "v_correct": b_val, "v_wrong": c_val,
        })
    return out


def score(claims, evs):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(
        pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    head = head.cuda().eval()
    s = np.zeros(len(claims), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j : j + BATCH], evs[j : j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            s[j : j + BATCH] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    del trunk, head
    torch.cuda.empty_cache()
    return s


def main():
    rng = np.random.default_rng(SEED)
    tri = build(N_TRIPLES, rng)
    print(f"built {len(tri)} triples", flush=True)
    if len(tri) < N_TRIPLES:
        print(f"WARNING: only {len(tri)} of {N_TRIPLES} triples constructible", flush=True)

    claims = [t["claim_a"] for t in tri] + [t["claim_b"] for t in tri] + [t["claim_c"] for t in tri]
    evs = [t["evidence"] for t in tri] * 3
    s = score(claims, evs)
    n = len(tri)
    sa, sb, sc = s[:n], s[n : 2 * n], s[2 * n :]

    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    auc_bc, _, _ = M59.auc_and_f1(y, np.concatenate([sb, sc]))
    auc_ab, _, _ = M59.auc_and_f1(y, np.concatenate([sa, sb]))

    gap = float(sa.mean() - sb.mean())
    auc_bc = float(auc_bc)
    license_ = gap >= 0.20 and auc_bc <= 0.60
    if auc_bc > 0.60:
        clause = "KILL - AUROC(b vs c) > 0.60: the model already discriminates derivations"
        verdict = "KILL"
    elif gap < 0.20:
        clause = ("KILL - mean score(a) - score(b) < 0.20: the literal-presence prior did not "
                  "transfer to the operating point")
        verdict = "KILL"
    else:
        clause = "LICENSE - score(a) - score(b) >= 0.20 AND AUROC(b vs c) <= 0.60"
        verdict = "LICENSE"

    pl.DataFrame([
        {**{k: v for k, v in t.items() if k != "evidence"},
         "score_a": float(sa[i]), "score_b": float(sb[i]), "score_c": float(sc[i])}
        for i, t in enumerate(tri)
    ]).write_parquet(SAMPLE)

    res = {
        "gate": "R14-H133 (R14-A4 kill-gate) shortcut-presence probe",
        "model": MODEL,
        "data": "TabFact test+validation tables, table_id-disjoint from the train split used by "
                "the clean mix and by R10-H108_data.tabfact_positives(); no arena, no gold",
        "seed": SEED, "n_triples": n, "n_requested": N_TRIPLES,
        "mean_score_a_verbatim": round(float(sa.mean()), 5),
        "mean_score_b_correct_derived": round(float(sb.mean()), 5),
        "mean_score_c_wrong_operand": round(float(sc.mean()), 5),
        "gap_a_minus_b": round(gap, 5),
        "auroc_b_vs_c": round(auc_bc, 4),
        "auroc_a_vs_b": round(float(auc_ab), 4),
        "bar": "LICENSE if gap >= 0.20 AND AUROC(b vs c) <= 0.60; KILL otherwise",
        "verdict": verdict, "clause_fired": clause,
        "sample": SAMPLE.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 92)
    print("R14-H133 SHORTCUT PROBE")
    print("=" * 92)
    print(f"  n={n}  (a) verbatim {sa.mean():.5f}   (b) correct-derived {sb.mean():.5f}   "
          f"(c) wrong-operand {sc.mean():.5f}")
    print(f"  gap a-b {gap:+.5f} (need >= 0.20)    AUROC(b vs c) {auc_bc:.4f} (need <= 0.60)")
    print(f"\n  VERDICT: {verdict}\n  {clause}\n  -> {RESULT}")


if __name__ == "__main__":
    main()
