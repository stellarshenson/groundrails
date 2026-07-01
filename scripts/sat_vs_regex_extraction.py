"""SaT vs regex claim-extraction benchmark (deterministic, grounding held fixed).

Hypothesis: a SaT neural segmenter admits more real claims than the shipped regex
extractor (`extract_claims`, English verb/copula gate), so fewer hallucinations
are silently dropped before grounding, lifting downstream grounding macro-F1 -
the lift concentrated on the non-English tail where the regex gate over-drops.

Design - isolate extraction from grounding. Each gold claim already has the
shipped lexical verdict cached (`golden_v5.lex_p`, eval rows, threshold 0.50). An
extractor "admits" a gold claim if its segmentation of the raw answer covers the
claim's text (rapidfuzz partial_ratio >= 90). An admitted claim takes its cached
verdict; a dropped claim defaults to "supported" (the agent never sees it, so a
dropped hallucination slips through). macro-F1 over (label, per-extractor verdict)
isolates the admission effect - grounding is identical for both arms.

Caveat - the gold claims were SaT-extracted in the gold-v2 rebuild, so the gold
favours SaT coverage; this measures downstream grounding macro-F1 under each
segmentation, not an unbiased extraction-quality score.

Run: uv run python scripts/sat_vs_regex_extraction.py
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import f1_score

from groundrails.extract import extract_claims
from groundrails.sat import SaTSegmenter

RAW = Path("data/raw/raw_v5/raw_v5.parquet")
GOLD = Path("data/processed/golden_v5/golden_v5.parquet")  # eval rows, lex_p inline
THRESHOLD = 0.50  # shipped global cut
ADMIT = 90  # rapidfuzz partial_ratio admission bar
_WORD = re.compile(r"\w+")


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _content_gate(seg: str) -> bool:
    """Language-agnostic claim gate: >= 3 alphabetic tokens."""
    return sum(1 for t in _WORD.findall(seg) if any(c.isalpha() for c in t)) >= 3


def _sat_extract(sat: SaTSegmenter, answer: str) -> list[str]:
    return [s for s in (sat.split(answer) or []) if _content_gate(s)]


def _admitted(claim: str, admitted_text: str) -> bool:
    c = _norm(claim)
    if len(c) < 8:
        return True  # too short to discriminate; treat as kept
    return fuzz.partial_ratio(c, admitted_text) >= ADMIT


def _metrics(df: pd.DataFrame, pred_col: str) -> dict:
    y, p = df["label"], df[pred_col]
    f_each = f1_score(y, p, average=None, labels=[0, 1])
    return {
        "macroF1": round(f1_score(y, p, average="macro"), 3),
        "hatF1": round(f_each[0], 3),  # hallucination class
        "supF1": round(f_each[1], 3),  # supported class
        "admit%": round(100 * df["__admit"].mean(), 1),
    }


def main() -> None:
    corpus = pd.read_parquet(RAW)
    corpus = corpus[corpus["has_gold"]][["trace_id", "answer"]]
    gold = pd.read_parquet(GOLD)
    gold = gold[gold["role"] == "eval"][["trace_id", "claim", "label", "lang_norm", "lex_p"]]
    gold["pred_full"] = (gold["lex_p"] >= THRESHOLD).astype(int)  # perfect-admission ceiling

    answers = dict(zip(corpus["trace_id"], corpus["answer"]))
    sat = SaTSegmenter()

    # Per-trace admitted-text for each extractor, then per-claim admission.
    regex_text, sat_text = {}, {}
    for tid, ans in answers.items():
        regex_text[tid] = _norm(" ".join(c.claim for c in extract_claims(ans)))
        sat_text[tid] = _norm(" ".join(_sat_extract(sat, ans)))

    out = {}
    for name, text_map in (("regex", regex_text), ("sat", sat_text)):
        adm = gold.apply(lambda r: _admitted(r["claim"], text_map.get(r["trace_id"], "")), axis=1)
        g = gold.copy()
        g["__admit"] = adm
        # dropped claim -> default supported (1); admitted -> cached verdict
        g[name] = g["pred_full"].where(adm, 1)
        out[name] = g

    # Assemble a comparison table: ALL / EN / non-EN.
    print(
        f"{'extractor':8} {'slice':8} {'n':>5} {'admit%':>7} {'macroF1':>8} {'halF1':>6} {'supF1':>6}"
    )
    print("-" * 56)
    ceil = {}
    for slc, mask_fn in (
        ("ALL", lambda g: g.index == g.index),
        ("EN", lambda g: g["lang_norm"] == "en"),
        ("non-EN", lambda g: g["lang_norm"] != "en"),
    ):
        # ceiling (perfect admission) per slice
        base = out["regex"]
        bm = mask_fn(base)
        ceil[slc] = round(
            f1_score(base.loc[bm, "label"], base.loc[bm, "pred_full"], average="macro"), 3
        )
        for name in ("regex", "sat"):
            g = out[name]
            m = mask_fn(g)
            sub = g.loc[m]
            mt = _metrics(sub.assign(__admit=sub["__admit"]), name)
            print(
                f"{name:8} {slc:8} {len(sub):>5} {mt['admit%']:>7} {mt['macroF1']:>8} {mt['hatF1']:>6} {mt['supF1']:>6}"
            )
    print("-" * 56)
    print("perfect-admission ceiling (macroF1):", ceil)

    # Admission by language (the mechanism).
    print("\nadmission % by language (regex vs sat):")
    for lang, sub in out["regex"].groupby("lang_norm"):
        if len(sub) < 20:
            continue
        r = 100 * sub["__admit"].mean()
        s = 100 * out["sat"].loc[out["sat"]["lang_norm"] == lang, "__admit"].mean()
        print(f"  {lang:6} n={len(sub):>4}  regex {r:5.1f}  sat {s:5.1f}  Δ {s - r:+5.1f}")


if __name__ == "__main__":
    main()
