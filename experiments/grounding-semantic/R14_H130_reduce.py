"""R14-H130 (R14-A1) - apply the frozen per-document log-K offset to the three
fresh per-window dumps and evaluate the pre-registered bar. CPU arithmetic only.

Form and alpha are frozen in `R14_H130_frozen_form.md` (written before any dump
existed) and read from `R14_gate_H130_alpha.json`. Nothing here re-derives them.

    sent_alpha(s) = max over documents d of [ max over w in d of score(s,w)
                                              - alpha * ln(K_d) ]
    resp_alpha    = min over sentences s of sent_alpha(s)

alpha = 0 must reproduce the banked per-subset AUROCs; that is the registered
reproduction check and a mismatch VOIDs the read.

Run:  uv run python experiments/grounding-semantic/R14_H130_reduce.py
"""

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R14_H130_reads.json"

TAGS = {
    "drc1": ("models/DR-lane-draw1-control", "DR_lane_draw1_control_windowed_result.json"),
    "drc2": ("models/DR-lane-draw2-control", "DR_lane_draw2_control_windowed_result.json"),
    "mgn1": ("models/DR-lane-draw1-margin", "DR_lane_draw1_margin_windowed_result.json"),
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")


def read_at(df, alpha):
    """Per-subset AUROC of the corrected windowed read."""
    d = (
        df.group_by(["subset", "resp_idx", "sent_idx", "doc_idx"])
        .agg(pl.col("score").max().alias("docmax"), pl.col("n_win_in_doc").first().alias("K"))
        .with_columns((pl.col("docmax") - alpha * pl.col("K").log()).alias("adj"))
        .group_by(["subset", "resp_idx", "sent_idx"])
        .agg(pl.col("adj").max().alias("sent"))
        .group_by(["subset", "resp_idx"])
        .agg(pl.col("sent").min().alias("resp"))
    )
    y = (
        df.group_by(["subset", "resp_idx"]).agg(pl.col("resp_label").first())
        .join(d, on=["subset", "resp_idx"])
        .sort(["subset", "resp_idx"])
    )
    out = {}
    for sub in sorted(y["subset"].unique().to_list()):
        s = y.filter(pl.col("subset") == sub)
        auc, _, _ = M59.auc_and_f1(s["resp_label"].to_numpy(), s["resp"].to_numpy())
        out[sub] = round(float(auc), 4)
    return out


def main():
    fit = json.loads((HERE / "R14_gate_H130_alpha.json").read_text())
    if fit["verdict"] != "LICENSED":
        raise SystemExit(f"alpha gate did not license: {fit['verdict']}")
    alpha = fit["alpha_frozen"]
    print(f"frozen alpha = {alpha} (alpha_hat {fit['alpha_hat']})", flush=True)

    per_ckpt, repro_ok = {}, True
    for tag, (model, banked_name) in TAGS.items():
        path = HERE / f"R14_H130_dump_{tag}.parquet"
        if not path.exists():
            print(f"{tag}: MISSING {path.name}", flush=True)
            per_ckpt[tag] = {"status": "MISSING DUMP"}
            repro_ok = False
            continue
        df = pl.read_parquet(path)
        banked = json.loads((HERE / banked_name).read_text())["per_subset"]
        base = read_at(df, 0.0)
        corr = read_at(df, alpha)
        rows, ok = {}, True
        for sub in sorted(base):
            b = round(float(banked[sub]["auc"]), 4)
            match = abs(base[sub] - b) < 5e-4
            ok &= match
            rows[sub] = {
                "banked": b, "dump_alpha0": base[sub], "corrected": corr[sub],
                "delta": round(corr[sub] - base[sub], 4), "reproduces": bool(match),
            }
        repro_ok &= ok
        per_ckpt[tag] = {
            "model": model, "banked_read": banked_name,
            "reproduction_ok": bool(ok),
            "per_subset": rows,
            "mean_alpha0": round(float(np.mean([v["dump_alpha0"] for v in rows.values()])), 5),
            "mean_corrected": round(float(np.mean([v["corrected"] for v in rows.values()])), 5),
            "mean_delta": round(float(np.mean([v["delta"] for v in rows.values()])), 5),
            "finqa_delta": rows["finqa"]["delta"],
            "worst_subset_delta": min(v["delta"] for v in rows.values()),
            "worst_subset": min(rows, key=lambda s: rows[s]["delta"]),
        }
        print(f"{tag}: repro={ok}  mean {per_ckpt[tag]['mean_alpha0']:.5f} -> "
              f"{per_ckpt[tag]['mean_corrected']:.5f} ({per_ckpt[tag]['mean_delta']:+.5f})  "
              f"finqa {per_ckpt[tag]['finqa_delta']:+.4f}  worst "
              f"{per_ckpt[tag]['worst_subset']} {per_ckpt[tag]['worst_subset_delta']:+.4f}",
              flush=True)

    have = [v for v in per_ckpt.values() if "per_subset" in v]
    fin = [v["finqa_delta"] for v in have]
    verdict, clause = "UNRESOLVED", []
    if not repro_ok or len(have) < 3:
        verdict = "VOID"
        clause.append("dump does not reproduce the banked read, or a dump is missing")
    else:
        fin_mean = float(np.mean(fin))
        kill = []
        if any(f < 0 for f in fin):
            kill.append("finqa negative on at least one checkpoint")
        if any(v["mean_delta"] < -0.005 for v in have):
            kill.append("arena mean delta < -0.005 on at least one checkpoint")
        if any(v["worst_subset_delta"] <= -0.05 for v in have):
            kill.append("a subset moved <= -0.05")
        admit = (
            all(f > 0 for f in fin)
            and fin_mean >= 0.005
            and all(v["mean_delta"] >= -0.002 for v in have)
            and all(v["worst_subset_delta"] > -0.035 for v in have)
        )
        if kill:
            verdict, clause = "KILL", kill
        elif admit:
            verdict = "ADMIT (arena clauses; gold_full clause reported separately)"
            clause = ["finqa > 0 on 3/3, finqa mean >= +0.005, mean >= -0.002 everywhere, "
                      "no subset <= -0.035"]
        else:
            verdict = "UNRESOLVED"
            clause = [
                f"finqa positive on {sum(f > 0 for f in fin)}/3",
                f"finqa mean {fin_mean:+.5f} (need >= +0.005)",
                f"min mean delta {min(v['mean_delta'] for v in have):+.5f} (need >= -0.002)",
                f"min subset delta {min(v['worst_subset_delta'] for v in have):+.4f} "
                f"(need > -0.035)",
            ]

    res = {
        "gate": "R14-H130 (R14-A1) corrected windowed read on three fresh checkpoints",
        "frozen_form_doc": "R14_H130_frozen_form.md",
        "alpha_frozen": alpha, "alpha_hat": fit["alpha_hat"],
        "per_checkpoint": per_ckpt,
        "reproduction_ok_all": bool(repro_ok),
        "finqa_deltas": fin,
        "finqa_mean_delta": round(float(np.mean(fin)), 5) if fin else None,
        "gold_full_clause": (
            "gold_full is read with max-over-chunks on pre-chunked evidence at or below the "
            "1,500-char window, so K_d == 1 for every document and the frozen offset "
            "alpha * ln(K_d) is identically zero - gold_full is unchanged by construction "
            "and the >= banked - 0.005 clause holds with equality. Verified in "
            "R14_H130_goldcheck.json."
        ),
        "verdict": verdict, "clause_fired": clause,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print(f"\n  VERDICT: {verdict}  {clause}")
    print(f"  -> {RESULT}")


if __name__ == "__main__":
    main()
