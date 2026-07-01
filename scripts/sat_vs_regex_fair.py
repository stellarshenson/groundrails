"""Fair SaT-vs-regex extraction benchmark (Round 13 follow-on).

True full-fair protocol, run once per extractor, fixing the two confounds of the
original Round 13 (verdict reused from SaT-calibrated gold; segments never
re-grounded):

  1. units  - per trace, segment the raw answer with the extractor (regex
              `extract_claims`, or SaT + a language-agnostic content gate).
              A segment fuzzy-matching a golden_v5 gold claim (partial_ratio >=
              90) INHERITS its verified label; the rest become judge units.
  2. judge  - dual-LLM judge (the gold-v2 protocol): each new segment judged
              against the trace evidence by `claude -p`, once with --model haiku
              and once with --model sonnet, label SUPPORTED / UNSUPPORTED /
              NOT_A_CLAIM. The judge runs HYGIENIC - no session persistence, no
              skills, no MCP, no settings - so the fleet neither persists state
              nor bloats context.
  3. build  - keep dual-agreed labels (haiku == sonnet); NOT_A_CLAIM agreements
              and disagreements are dropped (reported as extraction precision).
              Writes one labeled records parquet per extractor.
  4. calibrate - RE-GROUND + RE-CALIBRATE per extractor through the shipped
              `groundrails.calibrate` (dogfood). Grouped 5-fold by trace (no
              segment leak) gives an honest macro-F1 at each extractor's OWN
              fitted operating point; a full-data fit is also exported as the
              per-extractor calibration JSON via the same library path.

Private production data - all outputs live under data/ (gitignored) and
are never committed.

Run:
  uv run python scripts/sat_vs_regex_fair.py units
  uv run python scripts/sat_vs_regex_fair.py judge --extractor regex --model haiku  [--workers 10]
  uv run python scripts/sat_vs_regex_fair.py judge --extractor regex --model sonnet
  uv run python scripts/sat_vs_regex_fair.py judge --extractor sat   --model haiku
  uv run python scripts/sat_vs_regex_fair.py judge --extractor sat   --model sonnet
  uv run python scripts/sat_vs_regex_fair.py build
  uv run python scripts/sat_vs_regex_fair.py calibrate
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess

import pandas as pd

RAW = Path("data/raw/raw_v5/raw_v5.parquet")
GOLD = Path("data/processed/golden_v5/golden_v5.parquet")
OUT = Path("data/processed/sat_vs_regex_fair")
UNITS = OUT / "units_{ext}.json"
JUDGE = OUT / "judge_{ext}_{model}.json"
RECORDS = OUT / "records_{ext}.parquet"
CALJSON = OUT / "calibration_{ext}.json"

EXTRACTORS = ("regex", "sat")
MODELS = ("haiku", "sonnet")
FUZZY_MIN = 90.0
CHUNK = 25  # max claims per claude -p call
_WORD = re.compile(r"\w+")

# Hygienic judge flags: no persisted session, no skills, no MCP, no settings -
# the judge sees only the evidence + claims, nothing of the project context.
_JUDGE_FLAGS = [
    "--no-session-persistence",
    "--disable-slash-commands",
    "--strict-mcp-config",
    "--setting-sources",
    "",
]

_JUDGE_PROMPT = """You are a strict grounding judge for a technical support assistant.
Evidence below is the ONLY source of truth. For each numbered claim decide:

- SUPPORTED: the evidence explicitly supports the claim (translation across languages is fine)
- UNSUPPORTED: the claim asserts something the evidence does not contain or contradicts
- NOT_A_CLAIM: the sentence is not a checkable factual assertion (greeting, offer to help,
  question, navigation/connective text, pure formatting fragment)

Evidence:
<<<EVIDENCE
{evidence}
EVIDENCE>>>

There are exactly {n} claims, numbered 1 to {n}. Judge ONLY these numbered claims,
ignore any numbering inside the evidence text.

Claims:
{claims}

Reply with ONLY a JSON array with exactly {n} objects, idx 1 to {n}, no prose:
[{{"idx": 1, "label": "SUPPORTED"}}, ...]"""


_SAT = None


def _sat():
    global _SAT
    if _SAT is None:
        from groundrails.sat import SaTSegmenter

        _SAT = SaTSegmenter()
    return _SAT


def _content_gate(seg: str) -> bool:
    """Language-agnostic claim gate: >= 3 alphabetic tokens."""
    return sum(1 for t in _WORD.findall(seg) if any(c.isalpha() for c in t)) >= 3


def _segments(extractor: str, answer: str) -> list[str]:
    if extractor == "regex":
        from groundrails.extract import extract_claims

        return [c.claim for c in extract_claims(answer)]
    return [s for s in (_sat().split(answer) or []) if _content_gate(s)]


# ---------------------------------------------------------------------------
# 1. units
# ---------------------------------------------------------------------------


def cmd_units() -> None:
    from rapidfuzz import fuzz

    corpus = pd.read_parquet(RAW)
    corpus = corpus[corpus["has_gold"]]
    gold = pd.read_parquet(GOLD)
    gold = gold[gold["role"] == "eval"]
    gold_by_trace = {tid: list(zip(g["claim"], g["label"])) for tid, g in gold.groupby("trace_id")}

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in EXTRACTORS:
        units, n_inh, n_new = [], 0, 0
        for row in corpus.itertuples():
            segs = _segments(ext, row.answer)
            gold_claims = gold_by_trace.get(row.trace_id, [])
            inherited, new = [], []
            for s in segs:
                best = max(
                    gold_claims,
                    key=lambda r: fuzz.partial_ratio(str(r[0]).lower(), s.lower()),
                    default=None,
                )
                if best and fuzz.partial_ratio(str(best[0]).lower(), s.lower()) >= FUZZY_MIN:
                    inherited.append({"claim": s, "label": int(best[1])})
                else:
                    new.append(s)
            n_inh += len(inherited)
            n_new += len(new)
            units.append(
                {
                    "trace_id": row.trace_id,
                    "source_text": row.source_text,
                    "lang": row.lang_norm,
                    "inherited": inherited,
                    "new": new,
                }
            )
        Path(str(UNITS).format(ext=ext)).write_text(json.dumps(units, ensure_ascii=False))
        print(f"{ext:6}: traces {len(units)}  inherited {n_inh}  new-to-judge {n_new}")


# ---------------------------------------------------------------------------
# 2. judge (hygienic dual-LLM, resumable)
# ---------------------------------------------------------------------------


def _clean_env() -> dict:
    return {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("ANTHROPIC") or k.startswith("CLAUDE"))
    }


def _judge_chunk(evidence: str, claims: list[str], model: str, env: dict) -> list[dict] | None:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    prompt = _JUDGE_PROMPT.format(evidence=evidence[:60000], claims=numbered, n=len(claims))
    try:
        r = subprocess.run(
            ["claude", "-p", "--model", model, *_JUDGE_FLAGS],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        txt = r.stdout.strip()
        start, end = txt.find("["), txt.rfind("]")
        labels = json.loads(txt[start : end + 1])
        ok = {"SUPPORTED", "UNSUPPORTED", "NOT_A_CLAIM"}
        keep = [
            rec
            for rec in labels
            if rec.get("label") in ok and 1 <= rec.get("idx", 0) <= len(claims)
        ]
        if sorted(rec["idx"] for rec in keep) != list(range(1, len(claims) + 1)):
            return None
        return keep
    except Exception:
        return None


def _judge_one(unit: dict, model: str) -> tuple[str, list[dict] | None]:
    claims = unit["new"]
    if not claims:
        return unit["trace_id"], []
    env = _clean_env()
    evidence = unit["source_text"] or ""
    merged: list[dict] = []
    for off in range(0, len(claims), CHUNK):
        sub = claims[off : off + CHUNK]
        res = _judge_chunk(evidence, sub, model, env)
        if res is None:
            return unit["trace_id"], None
        merged.extend({"idx": off + r["idx"], "label": r["label"]} for r in res)
    return unit["trace_id"], merged


def cmd_judge(ext: str, model: str, workers: int) -> None:
    units = json.loads(Path(str(UNITS).format(ext=ext)).read_text())
    out_path = Path(str(JUDGE).format(ext=ext, model=model))
    done: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [u for u in units if u["trace_id"] not in done and u["new"]]
    print(f"{ext}/{model}: {len(todo)} units to judge ({len(done)} cached)", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_judge_one, u, model): u for u in todo}
        for i, f in enumerate(as_completed(futs), 1):
            tid, labels = f.result()
            if labels is not None:
                done[tid] = labels
            if i % 20 == 0 or i == len(todo):
                out_path.write_text(json.dumps(done))
                print(f"  {i}/{len(todo)} judged, {len(done)} stored", flush=True)
    out_path.write_text(json.dumps(done))
    failed = sum(1 for u in todo if u["trace_id"] not in done)
    print(f"done: {len(done)} units stored, {failed} failed (re-run to retry)")


# ---------------------------------------------------------------------------
# 3. build (keep dual-agreed)
# ---------------------------------------------------------------------------


def cmd_build() -> None:
    for ext in EXTRACTORS:
        units = json.loads(Path(str(UNITS).format(ext=ext)).read_text())
        haiku = json.loads(Path(str(JUDGE).format(ext=ext, model="haiku")).read_text())
        sonnet = json.loads(Path(str(JUDGE).format(ext=ext, model="sonnet")).read_text())
        rows, stats = (
            [],
            {"inherited": 0, "agreed": 0, "disagreed": 0, "not_a_claim": 0, "unjudged": 0},
        )
        for u in units:
            tid, src, lang = u["trace_id"], u["source_text"], u["lang"]
            for rec in u["inherited"]:
                rows.append(
                    {
                        "claim": rec["claim"],
                        "source_text": src,
                        "label": rec["label"],
                        "lang": lang,
                        "trace_id": tid,
                        "origin": "inherited",
                    }
                )
                stats["inherited"] += 1
            h, s = haiku.get(tid), sonnet.get(tid)
            if h is None or s is None:
                stats["unjudged"] += len(u["new"])
                continue
            hmap = {r["idx"]: r["label"] for r in h}
            smap = {r["idx"]: r["label"] for r in s}
            for i, claim in enumerate(u["new"], 1):
                hl, sl = hmap.get(i), smap.get(i)
                if hl is None or sl is None or hl != sl:
                    stats["disagreed"] += 1
                    continue
                if hl == "NOT_A_CLAIM":
                    stats["not_a_claim"] += 1
                    continue
                rows.append(
                    {
                        "claim": claim,
                        "source_text": src,
                        "label": 1 if hl == "SUPPORTED" else 0,
                        "lang": lang,
                        "trace_id": tid,
                        "origin": "judged",
                    }
                )
                stats["agreed"] += 1
        df = pd.DataFrame(rows)
        df.to_parquet(Path(str(RECORDS).format(ext=ext)), index=False)
        new_total = stats["agreed"] + stats["not_a_claim"] + stats["disagreed"]
        prec = stats["agreed"] / new_total if new_total else 0.0
        print(
            f"{ext:6}: {len(df)} rows  labels {df['label'].value_counts().to_dict()}  "
            f"new-precision {prec:.3f}  {stats}"
        )


# ---------------------------------------------------------------------------
# 4. calibrate (dogfood: re-ground + re-calibrate per extractor)
# ---------------------------------------------------------------------------

DRAWS = 600
TUNE = 600


def _macro(y, p) -> float:
    from sklearn.metrics import f1_score

    return round(f1_score(y, p, average="macro"), 4) if len(y) else float("nan")


def cmd_calibrate() -> None:
    import numpy as np
    from sklearn.model_selection import GroupKFold

    from groundrails import calibration as C

    print(f"{'extractor':8} {'slice':8} {'n':>5} {'macroF1':>8}")
    print("-" * 34)
    for ext in EXTRACTORS:
        recs = pd.read_parquet(Path(str(RECORDS).format(ext=ext)))
        records = recs[["claim", "source_text", "label", "lang"]].to_dict("records")
        frame = C.build_feature_frame(records)  # dogfood: re-ground once
        frame["trace_id"] = recs["trace_id"].to_numpy()
        frame["lang"] = recs["lang"].to_numpy()
        y = frame[C.RESPONSE].astype(int).to_numpy()
        groups = frame["trace_id"].to_numpy()

        # honest macro-F1: grouped 5-fold, a trace's segments never split
        preds = np.zeros(len(frame), dtype=int)
        for tr, te in GroupKFold(n_splits=5).split(frame, y, groups):
            v = C.fit_calibrator(
                frame.iloc[tr], draws=DRAWS, tune=TUNE, balance="balanced", random_seed=0
            )
            proba = v.predict_proba(frame.iloc[te].reindex(columns=C.PREDICTORS, fill_value=0.0))
            preds[te] = (np.asarray(proba) >= v.threshold).astype(int)
        frame["pred"] = preds

        en = frame["lang"] == "en"
        for slc, mask in (("ALL", frame.index == frame.index), ("EN", en), ("non-EN", ~en)):
            sub = frame[mask]
            print(
                f"{ext:8} {slc:8} {len(sub):>5} {_macro(sub[C.RESPONSE].astype(int), sub['pred']):>8}"
            )

        # dogfood the shipped path: full-data fit -> per-extractor calibration JSON
        verdict = C.calibrate(records, balance="balanced", draws=DRAWS, tune=TUNE, random_seed=0)
        block = C.verdict_to_block(verdict)
        Path(str(CALJSON).format(ext=ext)).write_text(json.dumps(block, indent=2))
        print(f"  -> wrote {str(CALJSON).format(ext=ext)} (per-extractor manifold)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "judge", "build", "calibrate"])
    ap.add_argument("--extractor", choices=EXTRACTORS)
    ap.add_argument("--model", choices=MODELS, default="haiku")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "units":
        cmd_units()
    elif a.cmd == "judge":
        if not a.extractor:
            ap.error("judge needs --extractor regex|sat")
        cmd_judge(a.extractor, a.model, a.workers)
    elif a.cmd == "build":
        cmd_build()
    else:
        cmd_calibrate()


if __name__ == "__main__":
    main()
