"""R13-H125 TOP2-UNION-PREMISE-READ - composite evidence premise, frozen weights.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 13);
full record with binding amendments in R13_synthesis.md section 4 (R2).

Claim: max-over-units is a logical OR, so a conjunctive claim whose hops sit in
different units is unconfirmable - no single premise entails it. Add exactly one
composite to the evidence pool (the two highest-scoring units concatenated in
document order, each clipped to the frozen 750-char stride, giving a <= 1500-char
premise identical to the shipped window budget) and read
`max over {units} u {composite}`, min over sentences unchanged.

Composite construction: `unit_a[:750] + unit_b[:750]` with the units ordered by
their flat window index (document order) and NO separator - for two adjacent
windows of the same chunk this reconstructs exactly the contiguous 1,500-char
span the shipped read would have seen, and the composite can never exceed the
1,500-char budget. 1500/750 are harness constants; nothing here is tuned.

Amendment 3 (M1 fold-in): the EXHAUSTIVE pair set is scored in the same run on
hotpotqa / covidqa / tatqa / hagrid ONLY (11,928 forwards) to answer whether
top-2-by-score is the wrong selector. NEVER exhaustive on techqa / expertqa
(786,105 of 837,274 forwards, no multi-hop claim).

Amendment 4: pre-registered union FIRE RATE - the fraction of sentences whose
final argmax IS the composite, split by RESPONSE label (the arena
`adherence_score`). A hallucinated fire rate matching the grounded fire rate is a
diagnostic refutation of the two-hop premise regardless of AUC.

Sanity guard (Gate A precedent): the STANDARD aggregation is rebuilt from the
matrix first and must reproduce the banked windowed result to 4 dp on all 10
subsets before any amended aggregation counts.

Bar: hotpotqa >= +0.030 AND mean >= +0.005 AND no subset <= -0.020, on BOTH H108
draws. REFUTE on draw 1 if hotpotqa < +0.030 OR mean < +0.003 OR any subset
< -0.020 -> draw 2 unspent, the multi-hop read line closes on measurement.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R13-H125_union.py --tag h108d1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent

WIN = 1500
STRIDE = 750
EXHAUSTIVE_SUBSETS = ("hotpotqa", "covidqa", "tatqa", "hagrid")

BANKED = {
    "h105d1": "R9-H105_windowed_result.json",
    "h105d2": "R9-H105_draw2_windowed_result.json",
    "h108d1": "R10-H108_lane_draw1_windowed_result.json",
    "h108d2": "R10-H108_lane_draw2_windowed_result.json",
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def composites_path(tag):
    return HERE / f"R13-H125_composites_{tag}.parquet"


def build_pairs(df):
    """The (sentence, unit-pair) rows to score.

    Every sentence contributes its top-2-by-score pair. On the four licensed
    subsets the full unordered pair set is emitted instead (the top-2 pair is a
    member of it, so nothing is scored twice).
    """
    rows = []
    key = ("subset", "resp_idx", "sent_idx")
    for (sub, ri, si), g in df.sort(
        ["subset", "resp_idx", "sent_idx", "win_idx"]
    ).group_by(key, maintain_order=True):
        wi = g["win_idx"].to_list()
        sc = g["score"].to_list()
        tx = g["win_text"].to_list()
        n = len(wi)
        if n < 2:
            continue
        if sub in EXHAUSTIVE_SUBSETS:
            idx = [(a, b) for a in range(n) for b in range(a + 1, n)]
        else:
            order = sorted(range(n), key=lambda j: -sc[j])[:2]
            a, b = sorted(order)  # document order
            idx = [(a, b)]
        top = sorted(range(n), key=lambda j: -sc[j])[:2]
        top_pair = tuple(sorted(top))
        for a, b in idx:
            rows.append({
                "subset": sub, "resp_idx": ri, "sent_idx": si,
                "win_a": wi[a], "win_b": wi[b],
                "is_top2": (a, b) == top_pair,
                "sent_text": g["sent_text"][0],
                "comp_text": tx[a][:STRIDE] + tx[b][:STRIDE],
            })
    return pl.DataFrame(rows)


def stage1(tag):
    """GPU: score every (sentence, composite) pair with the same frozen model."""
    out = composites_path(tag)
    if out.exists():
        print(f"stage 1 skipped - {out.name} exists", flush=True)
        return
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    df = pl.read_parquet(HERE / f"R13_dump_{tag}.parquet")
    pairs = build_pairs(df)
    print(f"{tag}: {len(pairs)} composites to score", flush=True)
    print(pairs.group_by("subset").len().sort("subset"), flush=True)

    model = str(HERE.parent.parent / json.loads((HERE / BANKED[tag]).read_text())["model"])
    print(f"GPU: {torch.cuda.get_device_name(0)}\nmodel: {model}", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    state = torch.load(
        pathlib.Path(model) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(model) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    head = head.cuda().eval()

    s_txt = pairs["sent_text"].to_list()
    c_txt = pairs["comp_text"].to_list()
    sc = np.zeros(len(s_txt), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for j in range(0, len(s_txt), 64):
            enc = tok(
                s_txt[j : j + 64], c_txt[j : j + 64], return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            )
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            sc[j : j + 64] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
            if (j // 64) % 50 == 0:
                print(f"  {j}/{len(s_txt)}  ({time.time() - t0:.0f}s)", flush=True)
    pairs = pairs.with_columns(pl.Series("comp_score", sc.astype(np.float64))).drop(
        "sent_text", "comp_text"
    )
    pairs.write_parquet(out)
    print(f"stage 1 -> {out}  ({len(pairs)} rows, {time.time() - t0:.0f}s)", flush=True)
    del trunk, head
    torch.cuda.empty_cache()


def response_auc(sent_scores, df):
    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
    out = {}
    for sub in sorted(df["subset"].unique().to_list()):
        r = (
            sent_scores.filter(pl.col("subset") == sub)
            .group_by("resp_idx").agg(pl.col("s").min()).sort("resp_idx")
        )
        y = (
            df.filter(pl.col("subset") == sub)
            .group_by("resp_idx").agg(pl.col("resp_label").first())
            .sort("resp_idx")["resp_label"].to_numpy()
        )
        auc, _, _ = M59.auc_and_f1(y, r["s"].to_numpy())
        out[sub] = float(auc)
    return out


def stage2(tag):
    df = pl.read_parquet(HERE / f"R13_dump_{tag}.parquet").select(
        "subset", "resp_idx", "sent_idx", "win_idx", "resp_label", "score"
    )
    comp = pl.read_parquet(composites_path(tag))
    key = ["subset", "resp_idx", "sent_idx"]

    unit_max = df.group_by(key).agg(pl.col("score").max().alias("u"))
    banked = json.loads((HERE / BANKED[tag]).read_text())["per_subset"]
    std = response_auc(unit_max.rename({"u": "s"}), df)
    repro = {
        s: {"rebuilt": round(std[s], 4), "banked": banked[s]["auc"],
            "ok": abs(round(std[s], 4) - banked[s]["auc"]) < 5e-5}
        for s in std
    }
    repro_ok = all(v["ok"] for v in repro.values())

    def union_read(cf):
        cmax = cf.group_by(key).agg(pl.col("comp_score").max().alias("c"))
        j = unit_max.join(cmax, on=key, how="left")
        return j.with_columns(
            pl.max_horizontal("u", pl.col("c").fill_null(-1.0)).alias("s")
        )

    top2 = union_read(comp.filter(pl.col("is_top2")))
    auc_top2 = response_auc(top2, df)
    delta_top2 = {s: round(auc_top2[s] - std[s], 4) for s in std}

    # amendment 3: exhaustive pair set, licensed subsets only
    ex_df = df.filter(pl.col("subset").is_in(EXHAUSTIVE_SUBSETS))
    ex_union = union_read(comp.filter(pl.col("subset").is_in(EXHAUSTIVE_SUBSETS)))
    auc_ex = response_auc(ex_union.filter(pl.col("subset").is_in(EXHAUSTIVE_SUBSETS)), ex_df)
    exhaustive = {
        s: {
            "standard": round(std[s], 4),
            "top2_union": round(auc_top2[s], 4),
            "exhaustive_union": round(auc_ex[s], 4),
            "exhaustive_minus_top2": round(auc_ex[s] - auc_top2[s], 4),
        }
        for s in EXHAUSTIVE_SUBSETS
    }
    n_ex = int(comp.filter(pl.col("subset").is_in(EXHAUSTIVE_SUBSETS)).height)

    # amendment 4: fire rate - composite is the sentence argmax - by response label
    # denominator = sentences that HAVE a composite (>= 2 units); a 1-unit
    # sentence cannot fire and is excluded rather than counted as a non-fire
    fire = top2.filter(pl.col("c").is_not_null()).with_columns(
        (pl.col("c") > pl.col("u")).cast(pl.Int8).alias("fired")
    ).join(
        df.group_by(key).agg(pl.col("resp_label").first()), on=key, how="left"
    )
    fire_rates = {}
    for sub in sorted(df["subset"].unique().to_list()):
        f = fire.filter(pl.col("subset") == sub)
        row = {}
        for lab, name in ((1, "grounded"), (0, "hallucinated")):
            t = f.filter(pl.col("resp_label") == lab)
            row[name] = None if not len(t) else round(float(t["fired"].mean()), 4)
            row[f"n_{name}"] = len(t)
        row["gap"] = (
            None if row["grounded"] is None or row["hallucinated"] is None
            else round(row["grounded"] - row["hallucinated"], 4)
        )
        fire_rates[sub] = row
    pooled = {}
    for lab, name in ((1, "grounded"), (0, "hallucinated")):
        t = fire.filter(pl.col("resp_label") == lab)
        pooled[name] = round(float(t["fired"].mean()), 4)
        pooled[f"n_{name}"] = len(t)
    pooled["gap"] = round(pooled["grounded"] - pooled["hallucinated"], 4)

    mean_std = float(np.mean(list(std.values())))
    mean_u = float(np.mean(list(auc_top2.values())))
    worst = min(delta_top2, key=delta_top2.get)

    clauses = {
        "hotpotqa_ge_0.030": delta_top2["hotpotqa"] >= 0.030,
        "mean_ge_0.005": (mean_u - mean_std) >= 0.005,
        "no_subset_le_-0.020": delta_top2[worst] > -0.020,
    }
    refute = (
        delta_top2["hotpotqa"] < 0.030
        or (mean_u - mean_std) < 0.003
        or delta_top2[worst] <= -0.020
    )
    # "hallucinated fire rate matches grounded" read as within 10% relative, or
    # inverted (hallucinated fires at least as often as grounded)
    diag_refutation = pooled["grounded"] <= 0 or (
        pooled["hallucinated"] >= 0.90 * pooled["grounded"]
    )

    res = {
        "hypothesis": "R13-H125 TOP2-UNION-PREMISE-READ",
        "tag": tag,
        "model": json.loads((HERE / BANKED[tag]).read_text())["model"],
        "composite": "unit_a[:750] + unit_b[:750], document order, no separator, <= 1500 chars",
        "reproduction_guard": {"per_subset": repro, "ok": bool(repro_ok)},
        "standard_per_subset": {s: round(v, 4) for s, v in std.items()},
        "union_per_subset": {s: round(v, 4) for s, v in auc_top2.items()},
        "delta_per_subset": delta_top2,
        "standard_mean": round(mean_std, 5),
        "union_mean": round(mean_u, 5),
        "mean_delta": round(mean_u - mean_std, 5),
        "worst_subset": {"subset": worst, "delta": delta_top2[worst]},
        "n_composites_scored": len(comp),
        "exhaustive_probe": {
            "subsets": list(EXHAUSTIVE_SUBSETS),
            "n_pair_forwards": n_ex,
            "per_subset": exhaustive,
            "exhaustive_beats_top2": any(
                v["exhaustive_minus_top2"] > 0.0 for v in exhaustive.values()
            ),
        },
        "fire_rate": {
            "definition": "fraction of scored sentences whose final argmax IS the "
                          "top-2 composite, split by response adherence label",
            "pooled": pooled,
            "per_subset": fire_rates,
            "diagnostic_refutation": bool(diag_refutation),
            "refutation_rule": "hallucinated >= 0.90 x grounded (fire rates match)",
            "note": "hallucinated fire rate matching grounded refutes the two-hop "
                    "premise regardless of AUC (amendment 4)",
        },
        "clauses_admit": clauses,
        "draw1_refute_fired": bool(refute),
        "verdict": (
            "VOID (reproduction guard failed)" if not repro_ok
            else ("REFUTE" if refute else "PASS-DRAW1 (spend draw 2)")
        ),
    }
    outp = HERE / f"R13-H125_result_{tag}.json"
    outp.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 84)
    print(f"R13-H125  {tag}   repro ok={repro_ok}")
    for s in sorted(delta_top2):
        print(f"    {s:12s} std {std[s]:.4f} -> union {auc_top2[s]:.4f}  "
              f"delta {delta_top2[s]:+.4f}")
    print(f"    MEAN         {mean_std:.5f} -> {mean_u:.5f}  delta {mean_u - mean_std:+.5f}")
    print(f"  exhaustive probe ({n_ex} forwards):")
    for s, v in exhaustive.items():
        print(f"    {s:12s} top2 {v['top2_union']:.4f}  exhaustive {v['exhaustive_union']:.4f}  "
              f"diff {v['exhaustive_minus_top2']:+.4f}")
    print(f"  fire rate pooled: grounded {pooled['grounded']} (n={pooled['n_grounded']})  "
          f"hallucinated {pooled['hallucinated']} (n={pooled['n_hallucinated']})  "
          f"gap {pooled['gap']}")
    print(f"  clauses: {clauses}   refute fired: {refute}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  -> {outp}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stage", default="all", choices=("1", "2", "all"))
    args = ap.parse_args()
    if args.stage in ("1", "all"):
        stage1(args.tag)
    if args.stage in ("2", "all"):
        stage2(args.tag)


if __name__ == "__main__":
    main()
