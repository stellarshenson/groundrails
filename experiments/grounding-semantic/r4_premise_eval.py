"""R4-H1 evaluation: does swapping the SummaC-style joint-premise NLI in for the max-over-chunks
NLI lift gold v3 eval macro-F1 through the Round 3 honest harness?

The joint-premise re-score (`joint_premise_score.py`) wrote `nli_ent_joint`/`nli_contra_joint`
(top-3 reranked chunks joined into one premise, one NLI pass) for the 3,218 cascade-fired rows.
This swaps those into the joint feature vector (replacing the cascade `nli_ent`/`nli_contra`,
which are max-over-chunks) and re-runs the leak-free harness: GroupKFold leave-one-source-out
OOF head + leave-one-fold-out EN/non-EN thresholds. Apples-to-apples vs R3 - same retraining,
only the NLI aggregation differs.

Bar (pre-registered): eval macro-F1 lift >= 0.014 over R3 AND English within +/-0.005 AND
synthetic TNR >= 0.88.

Run:  python experiments/grounding-semantic/r4_premise_eval.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from joint_xlingual import JF, best_threshold, load, oof_grouped  # noqa: E402
from perlang_honest import honest_yhat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREMISE = ROOT / "data" / "processed" / "golden_v3_joint_premise.parquet"
REPORT = ROOT / "reports" / "grounding_r4_premise.md"


def honest_macros(df, p, ev, en, nonen, langs):
    """Eval macro-F1 (overall / EN / non-EN) under the EN/non-EN leave-one-fold-out scheme."""
    y = df["label"].to_numpy(int)
    yh = honest_yhat(df, p, ev, "en_nonen", langs)
    m, em, nem = ev.to_numpy(), (en & ev).to_numpy(), nonen.to_numpy()
    return (
        f1_score(y[m], yh[m], average="macro"),
        f1_score(y[em], yh[em], average="macro"),
        f1_score(y[nem], yh[nem], average="macro"),
    )


def main() -> None:
    df = load()
    prem = pd.read_parquet(PREMISE)[["uid", "nli_ent_sentmax", "nli_ent_joint", "nli_contra_joint"]]
    df = df.merge(prem, on="uid", how="left")

    fired = df["nli_ent_joint"].notna()
    print(f"joint-premise scored on {int(fired.sum())}/{len(df)} rows (cascade-fired)")

    ev = df["role"].eq("eval")
    en = df["lang_norm"].eq("en")
    nonen = ev & ~en
    langs = [c for c, n in df.loc[ev, "lang_norm"].value_counts().items() if n >= 40]
    y = df["label"].to_numpy(int)

    # --- R3 baseline: max-over-chunks NLI (the shipped joint features) ---
    p_base = oof_grouped(df, JF, ("eval",))
    m_base, T_base = best_threshold(y[ev], p_base[ev.to_numpy()])
    base = honest_macros(df, p_base, ev, en, nonen, langs)

    # --- R4: swap the joint-premise NLI in for the cascade max-over-chunks NLI on fired rows ---
    df_r4 = df.copy()
    df_r4.loc[fired, "nli_ent"] = df_r4.loc[fired, "nli_ent_joint"]
    df_r4.loc[fired, "nli_contra"] = df_r4.loc[fired, "nli_contra_joint"]
    p_r4 = oof_grouped(df_r4, JF, ("eval",))
    m_r4, T_r4 = best_threshold(y[ev], p_r4[ev.to_numpy()])
    r4 = honest_macros(df_r4, p_r4, ev, en, nonen, langs)

    # synthetic TNR at each head's global eval-optimal cut
    aug = df["role"].eq("augmentation").to_numpy()
    tnr_base = float((p_base[aug] < T_base).mean())
    tnr_r4 = float((p_r4[aug] < T_r4).mean())

    # mechanism diagnostic: on under-graded supported eval rows (low sentence-max ent, high lex_p),
    # does the joined premise raise entailment >= 0.10 over the sentence-max value?
    sup = ev & df["label"].eq(1) & fired
    under = sup & (df["nli_ent_sentmax"] < 0.5) & (df["lex_p"] > 0.5)
    delta = (df.loc[under, "nli_ent_joint"] - df.loc[under, "nli_ent_sentmax"])
    gate_rise = float((delta >= 0.10).mean()) if len(delta) else float("nan")
    mean_rise = float(delta.mean()) if len(delta) else float("nan")

    lift = r4[0] - base[0]
    en_ctrl = r4[1] - base[1]
    verdict = (
        "SHIPS" if (lift >= 0.014 and abs(en_ctrl) <= 0.005 and tnr_r4 >= 0.88)
        else "NULL/REFUTED"
    )

    L = [
        "# R4-H1: joint-premise NLI (SummaC aggregation) - honest evaluation",
        "",
        f"Joint-premise scored on {int(fired.sum())} cascade-fired rows; gold v3 eval {int(ev.sum())} "
        f"rows, {len(langs)} calibrated languages. EN/non-EN leave-one-fold-out thresholds on "
        "GroupKFold leave-one-source-out OOF probabilities (the Round 3 honest harness).",
        "",
        "| head | eval macro-F1 | EN macro | non-EN macro | synthetic TNR |",
        "|---|---|---|---|---|",
        f"| R3 baseline (max-over-chunks NLI) | {base[0]:.3f} | {base[1]:.3f} | {base[2]:.3f} | {tnr_base:.3f} |",
        f"| R4 joint-premise NLI | {r4[0]:.3f} | {r4[1]:.3f} | {r4[2]:.3f} | {tnr_r4:.3f} |",
        f"| delta | {lift:+.3f} | {en_ctrl:+.3f} | {r4[2] - base[2]:+.3f} | {tnr_r4 - tnr_base:+.3f} |",
        "",
        f"Mechanism gate (under-graded supported rows, n={int(under.sum())}): joined premise raises "
        f"entailment >= 0.10 on {gate_rise:.1%} of them, mean rise {mean_rise:+.3f}.",
        "",
        f"Bar: macro lift >= 0.014 AND |EN ctrl| <= 0.005 AND synthetic TNR >= 0.88.",
        f"Measured: lift {lift:+.3f}, EN ctrl {en_ctrl:+.3f}, TNR {tnr_r4:.3f}.",
        "",
        f"Verdict: **{verdict}**",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
