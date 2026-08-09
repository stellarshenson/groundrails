"""R15 LENS-4 - derivation competence across the campaign's banked checkpoints.

Rebuilds the R14-H133 triples byte-identically (seed 20260809, held-out TabFact
test+validation, table_id-disjoint from every train split) and scores them on
every banked cross-encoder checkpoint on disk, plus the H129 committee teacher
(output mean of H105 draw 1 and draw 2).

Answers three questions the R15 register needs before any lane is built:

  Q1 CHECKPOINT INVARIANCE - is AUROC(correct vs wrong-operand) at chance on
     EVERY trained checkpoint, or is 0.4924 specific to H105 draw 1?  This runs
     probe P1's registered falsifier #2.
  Q2 TEACHER TRANSMISSION - what does the H129 committee teacher (the 0.72067
     output-space ensemble) score on the same axis?  If the teacher is also at
     chance, no distillation or ensembling route can ever transmit derivation
     competence, and the record should say so.
  Q3 TRUNCATION CONFOUND - P2 measured 34.93% of 1,500-char TabFact windows
     exceeding MAX_LEN 512.  Split the triples by encoded pair length and read
     AUROC inside the sub-512 stratum, where nothing is truncated.

Frozen weights, zero arena, zero gold, single card.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R15_gate_L4_ckptsweep.json"
SCORES = HERE / "R15_L4_ckpt_scores.parquet"

BATCH = 64
MAX_LEN = 512

CKPTS = [
    "R9-H105-mmbert-dann-clean",   # shipped, the H133 baseline (0.4924)
    "R9-H105-draw2",               # paired clean draw
    "R10-H108-lane-draw1",         # ADMITTED lane, best replicated finqa lever
    "R10-H108-lane-draw2",
    "DR-lane-draw1-control",       # decomposed-reads lane, control arm
    "DR-lane-draw2-control",
    "DR-lane-draw1-margin",        # H117 margin arm (finqa -0.1020)
    "R13-H129-draw1",              # distillation student
    "R11-H118-soup-h108",          # weight-space average
    "R8-H90-mmbert-dann-full",     # pre-clean-mix era
    "R8-H79-mmbert-dann",          # early DANN
]


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def score_ckpt(model_dir, claims, evs, want_lens=False):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    state = torch.load(
        pathlib.Path(model_dir) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(model_dir) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    head = head.cuda().eval()
    s = np.zeros(len(claims), dtype=np.float32)
    lens = np.zeros(len(claims), dtype=np.int32) if want_lens else None
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j : j + BATCH], evs[j : j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            if want_lens:
                raw = tok(claims[j : j + BATCH], evs[j : j + BATCH], truncation=False)
                lens[j : j + BATCH] = [len(x) for x in raw["input_ids"]]
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            s[j : j + BATCH] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    del trunk, head
    torch.cuda.empty_cache()
    return (s, lens) if want_lens else (s, None)


def main():
    P = _mod("h133", "R14_H133_probe.py")
    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")

    rng = np.random.default_rng(P.SEED)
    tri = P.build(P.N_TRIPLES, rng)
    n = len(tri)
    print(f"rebuilt {n} triples", flush=True)

    claims = [t["claim_a"] for t in tri] + [t["claim_b"] for t in tri] + [t["claim_c"] for t in tri]
    evs = [t["evidence"] for t in tri] * 3

    def auc(pos, neg):
        y = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
        a, _, _ = M59.auc_and_f1(y, np.concatenate([pos, neg]))
        return float(a)

    rows, banked, per_ckpt = [], {}, {}
    lens_b = None
    for name in CKPTS:
        d = ROOT / "models" / name
        if not (d / "dann_student.pt").exists():
            print(f"SKIP {name} (no dann_student.pt)", flush=True)
            continue
        want = lens_b is None
        s, lens = score_ckpt(str(d), claims, evs, want_lens=want)
        if want:
            lens_b = lens[n : 2 * n]
        sa, sb, sc = s[:n], s[n : 2 * n], s[2 * n :]
        banked[name] = (sa, sb, sc)
        r = {
            "checkpoint": name,
            "mean_a_verbatim": round(float(sa.mean()), 5),
            "mean_b_correct": round(float(sb.mean()), 5),
            "mean_c_wrong": round(float(sc.mean()), 5),
            "gap_a_minus_b": round(float(sa.mean() - sb.mean()), 5),
            "auroc_b_vs_c": round(auc(sb, sc), 4),
            "auroc_a_vs_b": round(auc(sa, sb), 4),
            "frac_b_above_0.5": round(float((sb > 0.5).mean()), 4),
        }
        rows.append(r)
        per_ckpt[name] = r
        print(json.dumps(r), flush=True)

    # Q2 - the H129 committee teacher is the OUTPUT MEAN of the two H105 draws
    teacher = None
    if "R9-H105-mmbert-dann-clean" in banked and "R9-H105-draw2" in banked:
        a1, b1, c1 = banked["R9-H105-mmbert-dann-clean"]
        a2, b2, c2 = banked["R9-H105-draw2"]
        ta, tb, tc = (a1 + a2) / 2, (b1 + b2) / 2, (c1 + c2) / 2
        teacher = {
            "checkpoint": "H129-committee-teacher (output mean H105 d1+d2)",
            "mean_a_verbatim": round(float(ta.mean()), 5),
            "mean_b_correct": round(float(tb.mean()), 5),
            "mean_c_wrong": round(float(tc.mean()), 5),
            "gap_a_minus_b": round(float(ta.mean() - tb.mean()), 5),
            "auroc_b_vs_c": round(auc(tb, tc), 4),
            "auroc_a_vs_b": round(auc(ta, tb), 4),
            "cross_draw_pearson_on_b": round(
                float(np.corrcoef(b1, b2)[0, 1]), 4),
            "cross_draw_pearson_on_bminusc": round(
                float(np.corrcoef(b1 - c1, b2 - c2)[0, 1]), 4),
        }
        print(json.dumps(teacher), flush=True)

    # Q3 - truncation stratification on the (claim_b, evidence) encoded length
    trunc = {}
    if lens_b is not None:
        over = lens_b > MAX_LEN
        trunc["frac_pairs_over_512"] = round(float(over.mean()), 4)
        trunc["mean_untruncated_len"] = round(float(lens_b[~over].mean()), 1)
        trunc["mean_truncated_len"] = (
            round(float(lens_b[over].mean()), 1) if over.any() else None)
        for name in ("R9-H105-mmbert-dann-clean", "R10-H108-lane-draw1"):
            if name not in banked:
                continue
            sa, sb, sc = banked[name]
            trunc[name] = {
                "n_fits": int((~over).sum()),
                "auroc_b_vs_c_fits_512": round(auc(sb[~over], sc[~over]), 4),
                "mean_b_fits": round(float(sb[~over].mean()), 5),
                "n_over": int(over.sum()),
                "auroc_b_vs_c_over_512": (
                    round(auc(sb[over], sc[over]), 4) if over.sum() > 20 else None),
                "mean_b_over": (
                    round(float(sb[over].mean()), 5) if over.any() else None),
            }
        print(json.dumps(trunc), flush=True)

    # bank per-item scores for every checkpoint
    df = pl.DataFrame({
        "table_id": [t["table_id"] for t in tri],
        "column": [t["column"] for t in tri],
        "v_correct": [t["v_correct"] for t in tri],
        "v_wrong": [t["v_wrong"] for t in tri],
        "pair_tokens_b": lens_b.tolist() if lens_b is not None else [0] * n,
    })
    for name, (sa, sb, sc) in banked.items():
        tag = name.replace("-", "_")
        df = df.with_columns([
            pl.Series(f"a__{tag}", sa), pl.Series(f"b__{tag}", sb), pl.Series(f"c__{tag}", sc),
        ])
    df.write_parquet(SCORES)

    out = {
        "probe": "R15 LENS-4 checkpoint sweep of the H133 derivation axis",
        "data": "R14_H133_probe.build(), seed 20260809, held-out TabFact test+validation, "
                "table_id-disjoint from every train split; zero arena, zero gold",
        "n_triples": n,
        "max_len": MAX_LEN,
        "banked_reference": "R14_gate_H133_probe.json -> auroc_b_vs_c 0.4924 on H105 draw 1",
        "checkpoints": rows,
        "teacher": teacher,
        "truncation_strata": trunc,
        "scores": SCORES.name,
    }
    RESULT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
