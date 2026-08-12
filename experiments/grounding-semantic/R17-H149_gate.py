"""R17-H149 stage 1 - KILL-GATE read of the banked control on the prose probe.

Frozen-weights, held-out, arena-free.  Scores `R17-H149_probe.parquet` with the
campaign control `models/R9-H105-mmbert-dann-clean` and writes
`R17-H149_gate_result.json`.

Registered gate: LICENSE the bare-assertion prose lane only if control AUROC
<= 0.70 (prose baselines run higher than the procedural ones, hence the looser
bar than H148's 0.65).  The flag is written; the coordinator adjudicates.

Per-family reads are printed beside the pooled one so a mixed result is
decomposable - the R17-H148 lesson, where one trivially-solvable family carried
the pooled read past its bar.

GPU2 only:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    uv run python experiments/grounding-semantic/R17-H149_gate.py
"""
import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np                                                  # noqa: E402
import polars as pl                                                 # noqa: E402

import R15_gate_common as G                                         # noqa: E402

HERE = pathlib.Path(__file__).parent
PROBE = HERE / "R17-H149_probe.parquet"
MANIFEST = HERE / "R17-H149_probe_manifest.json"
CENSUS = HERE / "R17-H149_census.json"
AUDIT = HERE / "R17-H149_audit_result.json"
OUT = HERE / "R17-H149_gate_result.json"

CONTROL = "R9-H105-mmbert-dann-clean"
GATE_BAR = 0.70


def se_hanley(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                          + (n2 - 1) * (q2 - a * a)) / (n1 * n2)))


def block(df, s):
    y = df["label"].to_numpy()
    a = G.auroc(s[y == 1], s[y == 0])
    piv = df.select(["pair_id", "label"]).with_columns(pl.Series("score", s)) \
            .pivot(on="label", index="pair_id", values="score",
                   aggregate_function="first").drop_nulls()
    pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
    return {"auroc": round(a, 4),
            "auroc_se_hanley_mcneil": round(se_hanley(a, int((y == 1).sum()),
                                                      int((y == 0).sum())), 4),
            "pairs": len(piv),
            "within_pair_acc": round(float(((pos > neg) + 0.5 * (pos == neg)).mean()), 4),
            "mean_pos": round(float(pos.mean()), 4), "mean_neg": round(float(neg.mean()), 4)}


def reads(df, s):
    out = block(df, s)
    scored = df.with_columns(pl.Series("score", s))
    for key in ("neg_family", "corpus", "direction"):
        out[f"per_{key}"] = {k[0]: block(sub, sub["score"].to_numpy())
                             for k, sub in scored.group_by(key)}
    return out


def main():
    df = pl.read_parquet(PROBE)
    man = json.loads(MANIFEST.read_text())
    census = json.loads(CENSUS.read_text())
    print(f"probe: {df.height} rows / {df['pair_id'].n_unique()} pairs", flush=True)

    tok, trunk, head = G.load_ckpt(CONTROL)
    s = G.score(tok, trunk, head, df["claim"].to_list(), df["chunk"].to_list())
    control = reads(df, s)
    print(f"control {CONTROL}: AUROC {control['auroc']} "
          f"(SE {control['auroc_se_hanley_mcneil']})", flush=True)

    lic = bool(control["auroc"] <= GATE_BAR)
    res = {
        "hypothesis": "R17-H149 BARE-ASSERTION PROSE LANE - stage 1 kill-gate",
        "gpu": "GPU2 (RTX 5000 Ada), CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2",
        "sources_census": census,
        "probe": {"file": PROBE.name, "manifest": MANIFEST.name,
                  "pairs": man["pairs"], "rows": man["rows"],
                  "documents": man["documents"], "passages": man["passages"],
                  "families": man["families"], "corpora": man["corpora"],
                  "extraction": man["extraction"], "dropped": man["dropped"],
                  "construction": man["construction"], "verify": man["verify"]},
        "mechanical_audit": (json.loads(AUDIT.read_text()) if AUDIT.exists()
                             else {"error": "audit result not written"}),
        "control_checkpoint": CONTROL,
        "control_auroc": control["auroc"],
        "control_read": control,
        "gate": {"bar": f"control AUROC <= {GATE_BAR}", "license": lic},
        "license_flag": lic,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({"control_auroc": control["auroc"],
                      "se": control["auroc_se_hanley_mcneil"],
                      "per_family": control["per_neg_family"],
                      "per_corpus": control["per_corpus"],
                      "license_flag": lic}, indent=2), flush=True)
    print(f"=== R17-H149 GATE: control {control['auroc']} -> "
          f"{'LICENSE' if lic else 'KILL'} ===", flush=True)


if __name__ == "__main__":
    main()
