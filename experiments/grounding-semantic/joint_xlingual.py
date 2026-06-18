"""R1 cross-lingual hypotheses for the joint grounder, on gold v3.

Races five cross-lingual hypotheses (R1-H1..H5) against the enriched baselines on gold v3.
All training is GroupKFold leave-one-source-out on `group_id` so a base source and its
synthetic translations never straddle train/test. Headline macro-F1 is on the verified
golden (`role=eval`); the synthetic augmentation (`role=augmentation`, all hallucination)
trains in-fold and is scored only as an offline cross-lingual TNR probe.

Inputs (joined on uid, both produced GPU-free):
  data/processed/golden_v3_lex.parquet      - lex_p + blocked/fired/contra (lex_v3.py)
  data/processed/golden_v3_cascade_scores.parquet - cascade signals (score_enriched.py)

Each hypothesis measures a cheap diagnostic kill-gate BEFORE claiming a result. Writes
reports/grounding_joint_xlingual.md.

Run:  python experiments/grounding-semantic/joint_xlingual.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEX = ROOT / "data" / "processed" / "golden_v3_lex.parquet"
CAS = ROOT / "data" / "processed" / "golden_v3_cascade_scores.parquet"
CFG = ROOT / "src" / "groundrails" / "config_document_processing.yaml"
REPORT = ROOT / "reports" / "grounding_joint_xlingual.md"

JF = ["lex_p", "rr_max", "nli_ent", "cos_max", "nli_contra", "lex_contra", "lex_blocked"]
TGRID = np.linspace(0.05, 0.95, 181)
MIN_LANG = 40  # per-language analysis floor


# --------------------------------------------------------------------------- data


def load() -> pd.DataFrame:
    lex = pd.read_parquet(LEX)
    cas = pd.read_parquet(CAS)[["uid", "cos_max", "rr_max", "nli_ent", "nli_contra", "ran_rr", "ran_nli"]]
    df = lex.merge(cas, on="uid", how="inner")
    df["lex_p"] = df["lex_p"].where(~df["lex_blocked"].astype(bool), 0.0)
    for c in ("lex_blocked", "lex_contra"):
        df[c] = df[c].astype(float)
    return df


# --------------------------------------------------------------------------- helpers


def best_threshold(y, p):
    """Macro-F1-optimal support/hallucination cut over a fixed grid."""
    best = (-1.0, 0.5)
    for t in TGRID:
        m = f1_score(y, (p >= t).astype(int), average="macro")
        if m > best[0]:
            best = (m, float(t))
    return best  # (macro, T)


def oof_grouped(df, feat, train_roles, k=5):
    """OOF P(supported) for every row via GroupKFold on group_id. A row is predicted by a
    model trained on OTHER groups only; training rows are further restricted to train_roles
    (so synthetic aug trains only when its source group is not the held-out fold)."""
    X = df[feat].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["group_id"].to_numpy()
    role = df["role"].to_numpy()
    p = np.full(len(df), np.nan)
    for tr, te in GroupKFold(k).split(X, y, groups):
        m = np.isin(role[tr], train_roles)
        sc = StandardScaler().fit(X[tr][m])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr][m]), y[tr][m])
        p[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def v1_head_proba(df):
    """Frozen shipped calibration.semantic head applied to gold v3 (no training)."""
    block = yaml.safe_load(CFG.read_text())["calibration"]["semantic"]
    w = block["weights"]
    z = np.full(len(df), float(w["Intercept"]))
    for f in JF:
        z += float(w[f]) * df[f].to_numpy(float)
    return 1.0 / (1.0 + np.exp(-z))


def slice_metrics(df, p, T, mask):
    """macro-F1 + class recalls on a row subset at threshold T."""
    y = df.loc[mask, "label"].to_numpy(int)
    yh = (p[mask.to_numpy()] >= T).astype(int)
    if len(y) == 0 or len(set(y)) < 2:
        return {"macro": float("nan"), "n": int(mask.sum())}
    return {
        "macro": f1_score(y, yh, average="macro"),
        "sup_recall": float(((yh == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "tnr": float(((yh == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "n": int(mask.sum()),
    }


def lexical_flag(df):
    """Shipped lexical verdict: hallucination when blocked or no lexical layer fired."""
    return (df["lex_blocked"].astype(bool) | (df["lex_fired"] < 0.5)).to_numpy()


# --------------------------------------------------------------------------- run


def main() -> None:
    if not (LEX.exists() and CAS.exists()):
        raise SystemExit(f"missing inputs - run lex_v3.py and score_enriched.py first\n  {LEX}\n  {CAS}")
    df = load()
    ev = df["role"].eq("eval")
    aug = df["role"].eq("augmentation")
    en = df["lang_norm"].eq("en")
    nonen = ev & ~en
    y = df["label"].to_numpy(int)
    L = [
        "# Joint lexical + semantic, cross-lingual hypotheses (gold v3)",
        "",
        f"Golden eval {int(ev.sum())} rows ({int((df.loc[ev,'label']==0).sum())} hallucination), "
        f"synthetic aug {int(aug.sum())} negatives, {df.loc[ev,'lang_norm'].nunique()} eval languages. "
        "GroupKFold leave-one-source-out on `group_id`; headline macro-F1 on `role=eval`, "
        "synthetic scored as an offline TNR probe. Each hypothesis gates on a precondition first.",
        "",
        "## Baselines (role=eval)",
        "",
        "| baseline | macro-F1 | EN macro | non-EN macro | non-EN sup-recall |",
        "|---|---|---|---|---|",
    ]

    # baseline A: lexical-only (high)
    flag = lexical_flag(df)
    yh_lex = (~flag).astype(int)  # 1 = supported
    a_all = f1_score(y[ev], yh_lex[ev.to_numpy()], average="macro")
    a_en = f1_score(y[en & ev], yh_lex[(en & ev).to_numpy()], average="macro")
    a_ne = f1_score(y[nonen], yh_lex[nonen.to_numpy()], average="macro")
    a_ne_rec = float(((yh_lex[nonen.to_numpy()] == 1) & (df.loc[nonen, "label"] == 1)).sum()
                     / max((df.loc[nonen, "label"] == 1).sum(), 1))
    L.append(f"| lexical-only (high) | {a_all:.3f} | {a_en:.3f} | {a_ne:.3f} | {a_ne_rec:.2f} |")

    # baseline B: frozen v1 joint head
    pv1 = v1_head_proba(df)
    mv1, Tv1 = best_threshold(y[ev], pv1[ev.to_numpy()])
    b_en = slice_metrics(df, pv1, Tv1, en & ev)
    b_ne = slice_metrics(df, pv1, Tv1, nonen)
    L.append(f"| joint v1-head (frozen) | {mv1:.3f} | {b_en['macro']:.3f} | {b_ne['macro']:.3f} | {b_ne.get('sup_recall', float('nan')):.2f} |")

    # OOF retrains
    p_h4 = oof_grouped(df, JF, ("eval",))               # H4: retrain on enriched eval
    p_h2 = oof_grouped(df, JF, ("eval", "augmentation"))  # H2: + synthetic in training
    m_h4, T_h4 = best_threshold(y[ev], p_h4[ev.to_numpy()])
    m_h2, T_h2 = best_threshold(y[ev], p_h2[ev.to_numpy()])

    rows = []  # (id, name, verdict-bits)

    # R1-H1 native multilingual cascade vs MT bridge (probe)
    yne = df.loc[nonen, "label"].to_numpy(int)
    auc_cos = roc_auc_score(yne, df.loc[nonen, "cos_max"]) if len(set(yne)) > 1 else float("nan")
    auc_ent = roc_auc_score(yne, df.loc[nonen, "nli_ent"]) if len(set(yne)) > 1 else float("nan")
    h1_auc = max(auc_cos, auc_ent)
    rows.append(("R1-H1", "native multilingual cascade (no MT bridge)",
                 f"non-EN cascade AUC max(cos {auc_cos:.3f}, nli_ent {auc_ent:.3f}) = {h1_auc:.3f}",
                 f"gate >= 0.65: {'PASS' if h1_auc >= 0.65 else 'KILL'}; bar >= 0.75: "
                 f"{'meets' if h1_auc >= 0.75 else 'below'}"))

    # R1-H2 synthetic negatives lift cross-lingual TNR
    sup_ne_mask = nonen & df["label"].eq(1)
    gate_y = np.r_[np.zeros(int(aug.sum())), np.ones(int(sup_ne_mask.sum()))]
    gate_s = np.r_[df.loc[aug, "nli_ent"].to_numpy(), df.loc[sup_ne_mask, "nli_ent"].to_numpy()]
    h2_gate = roc_auc_score(gate_y, gate_s) if len(set(gate_y)) > 1 else float("nan")
    tnr_syn = float((p_h2[aug.to_numpy()] < T_h2).mean())
    rec_ne_h2 = slice_metrics(df, p_h2, T_h2, sup_ne_mask).get("sup_recall", float("nan"))
    rows.append(("R1-H2", "synthetic negatives lift cross-lingual TNR",
                 f"synthetic TNR {tnr_syn:.3f} (bar >= 0.80), non-EN sup-recall {rec_ne_h2:.3f} (bar >= 0.70)",
                 f"gate nli_ent AUC {h2_gate:.3f} >= 0.70: {'PASS' if h2_gate >= 0.70 else 'KILL'}"))

    # R1-H4 retrain on enriched multilingual gold (vs v1-head)
    def err(mask, p, T):
        return float(((p[mask.to_numpy()] >= T).astype(int) != df.loc[mask, "label"]).mean())
    v1_ne_err, v1_en_err = err(nonen, pv1, Tv1), err(en & ev, pv1, Tv1)
    h4_ne = slice_metrics(df, p_h4, T_h4, nonen)["macro"]
    h4_en = slice_metrics(df, p_h4, T_h4, en & ev)["macro"]
    rows.append(("R1-H4", "joint head retrained on enriched gold",
                 f"non-EN macro {b_ne['macro']:.3f} (v1) -> {h4_ne:.3f}  (lift {h4_ne - b_ne['macro']:+.3f}, bar >= +0.05); "
                 f"EN macro {b_en['macro']:.3f} -> {h4_en:.3f} (control +/-0.005)",
                 f"gate v1 non-EN err {v1_ne_err:.3f} - EN err {v1_en_err:.3f} = {v1_ne_err - v1_en_err:+.3f} >= 0.05: "
                 f"{'PASS' if v1_ne_err - v1_en_err >= 0.05 else 'KILL'}"))

    # R1-H3 per-language thresholds on the H2 OOF probs
    langs = [c for c, n in df.loc[ev, "lang_norm"].value_counts().items() if n >= MIN_LANG]
    per_lang_T = {}
    yh = (p_h2 >= T_h2).astype(int)  # start from global
    for lg in langs:
        m = ev & df["lang_norm"].eq(lg)
        _, Tl = best_threshold(df.loc[m, "label"].to_numpy(int), p_h2[m.to_numpy()])
        per_lang_T[lg] = Tl
        yh[m.to_numpy()] = (p_h2[m.to_numpy()] >= Tl).astype(int)
    macro_global = f1_score(y[ev], (p_h2[ev.to_numpy()] >= T_h2).astype(int), average="macro")
    macro_perlang = f1_score(y[ev], yh[ev.to_numpy()], average="macro")
    max_gap = max((abs(t - T_h2) for t in per_lang_T.values()), default=0.0)
    rows.append(("R1-H3", "per-language joint calibration",
                 f"macro global {macro_global:.3f} -> per-language {macro_perlang:.3f} "
                 f"(lift {macro_perlang - macro_global:+.3f}, bar >= 0.83 abs)",
                 f"gate max |T_lang - T_global| = {max_gap:.3f} >= 0.03: {'PASS' if max_gap >= 0.03 else 'KILL'}"))

    # R1-H5 language-aware escalation band
    band_lo, band_hi = yaml.safe_load(CFG.read_text())["calibration"]["semantic"]["escalation_band"]
    inband = (df["lex_p"] > band_lo) & (df["lex_p"] < band_hi)
    escalate = (~en) | inband  # non-EN always escalates; EN only in-band
    joint_yh = (p_h2 >= T_h2).astype(int)
    fused = np.where(escalate.to_numpy(), joint_yh, (~lexical_flag(df)).astype(int))
    m_h5 = f1_score(y[ev], fused[ev.to_numpy()], average="macro")
    en_esc_aware = float(escalate[en & ev].mean())          # language-aware EN escalation share
    lex_ne_err = float((lexical_flag(df)[nonen.to_numpy()].astype(int) != (df.loc[nonen, "label"] == 0)).mean())
    lex_en_err = float((lexical_flag(df)[(en & ev).to_numpy()].astype(int) != (df.loc[en & ev, "label"] == 0)).mean())
    rows.append(("R1-H5", "language-aware escalation band",
                 f"fused macro {m_h5:.3f} (bar >= 0.825), EN escalation share {en_esc_aware:.1%}",
                 f"gate non-EN lex err {lex_ne_err:.3f} - EN lex err {lex_en_err:.3f} = {lex_ne_err - lex_en_err:+.3f} >= 0.10: "
                 f"{'PASS' if lex_ne_err - lex_en_err >= 0.10 else 'KILL'}"))

    L += [
        "",
        f"Retrained joint heads (OOF, GroupKFold): eval-only macro {m_h4:.3f} (T={T_h4:.2f}); "
        f"eval+synthetic macro {m_h2:.3f} (T={T_h2:.2f}).",
        "",
        "## Hypotheses",
        "",
        "| id | mechanism | result | kill-gate |",
        "|---|---|---|---|",
    ]
    for hid, name, res, gate in rows:
        L.append(f"| {hid} | {name} | {res} | {gate} |")
    L.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
