"""Held-out check of the open-defect fixes, run once under ``HELDOUT.md``.

Each numeric rule is switched off on its own and the difference against the full
working tree is scored on labelled held-out corpora; the verdicts follow the
pre-registered criteria in ``HELDOUT.md``. The rejoin-gate sample is printed for a
judged count, which the criteria also fix in advance.

Usage: ``HF_HUB_OFFLINE=1 .venv/bin/python experiments/defects/measure_heldout.py``
writes ``measure_heldout.json`` beside this file.

A record of the round-1 run: the code it tested was not kept (only its hash in
``HELDOUT.md``), and every rule it switches except the date bound is reverted, so it
does not reproduce the verdicts on the current tree.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import decimal
import glob
import json
from pathlib import Path
import random
import re
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_open_defects import HUB, ec_new, ec_old, ex_new, ex_old  # noqa: E402

from groundrails import lexical  # noqa: E402

HERE = Path(__file__).resolve().parent
FROZEN_SRC = "621a06d8cae68b33"  # HELDOUT.md: git diff HEAD -- src | sha256sum
NEVER = re.compile(r"(?!)")
ALWAYS = re.compile(r"")
MIN_N = 10


def _lax_agree(a: str, b: str) -> bool:
    """The committed rounding agreement, with only the DEF-NUMBER-43 guard added."""
    try:
        return ec_old._agree_at_coarser_precision(a, b)
    except decimal.InvalidOperation:
        return False


SWITCHES = {
    "DEF-NUMBER-42 date bound": {"_DATE_BOUND_RE": NEVER},
    "DEF-NUMBER-32 5% rounding": {"_agree_at_coarser_precision": _lax_agree},
    "DEF-NUMBER-34 whole-range comparison": {"_merged_numbers": lambda text: []},
    "DEF-NUMBER-34 year-span exclusion": {"_YEAR_SPAN_RE": NEVER},
    "DEF-NUMBER-33 glued citation marker": {"_GLUED_CITATION_RE": NEVER},
}


@contextmanager
def _patched(module, attrs: dict):
    saved = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def _file(pattern: str) -> str:
    return sorted(glob.glob(str(HUB / pattern)))[0]


def _vitaminc_rows() -> list[dict]:
    rows = [json.loads(line) for line in open(_file("datasets--tals--vitaminc/snapshots/*/dev.jsonl")) if line.strip()]
    # the 45 items TestVitaminCComposite pins: first 15 per label with claim and evidence
    taken: Counter = Counter()
    kept = []
    for r in rows:
        usable = r.get("claim") and r.get("evidence") and r.get("label") in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
        if usable and taken[r["label"]] < 15:
            taken[r["label"]] += 1
            continue
        if usable:
            kept.append(r)
    return kept


def _halueval(name: str) -> pl.DataFrame:
    return pl.read_parquet(_file(f"datasets--pminervini--HaluEval/snapshots/*/{name}/*.parquet"))


def _psiloqa() -> pl.DataFrame:
    return pl.read_parquet(_file("datasets--s-nlp--PsiloQA/snapshots/*/data/test-*.parquet"))


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def numeric_pairs():
    """``(corpus, claim, passage, labelled)``."""
    for r in _vitaminc_rows():
        if _has_digit(r["claim"]):
            yield "vitaminc", r["claim"], r["evidence"], r["label"] == "REFUTES"
    for p in sorted(glob.glob(str(HUB / "datasets--lytang--C2D-and-D2C-MiniCheck/snapshots/*/data/*.parquet"))):
        for r in pl.read_parquet(p).iter_rows(named=True):
            if _has_digit(r["claim"]):
                yield "minicheck", r["claim"], r["doc"], r["label"] == 0
    for r in _halueval("qa").iter_rows(named=True):
        for key, labelled in (("right_answer", False), ("hallucinated_answer", True)):
            if _has_digit(r[key]):
                yield "halueval-qa", r[key], r["knowledge"], labelled
    for r in _halueval("dialogue").iter_rows(named=True):
        for key, labelled in (("right_response", False), ("hallucinated_response", True)):
            if _has_digit(r[key]):
                yield "halueval-dialogue", r[key], r["knowledge"], labelled
    for r in _halueval("summarization").iter_rows(named=True):
        for key, labelled in (("right_summary", False), ("hallucinated_summary", True)):
            for c in ex_new.extract_claims(r[key]):
                if _has_digit(c.claim):
                    yield "halueval-summarization", c.claim, r["document"], labelled
    for r in _psiloqa().filter(pl.col("lang") == "en").iter_rows(named=True):
        spans = r["labels"] or []
        for c in ex_new.extract_claims(r["llm_answer"]):
            if _has_digit(c.claim):
                labelled = any(s < c.char_end and e > c.char_start for s, e in spans)
                yield "psiloqa-en", c.claim, r["wiki_passage"], labelled


def measure_numeric() -> dict:
    base: Counter = Counter()
    effect = {name: Counter() for name in SWITCHES}
    samples = {name: {"gained": [], "lost": []} for name in SWITCHES}
    for corpus, claim, passage, labelled in numeric_pairs():
        try:
            committed = bool(ec_old.find_numeric_mismatches(claim, passage))
        except decimal.InvalidOperation:
            base[(corpus, "committed_raised")] += 1
            committed = False
        full = bool(ec_new.find_numeric_mismatches(claim, passage))
        base[(corpus, "pairs")] += 1
        base[(corpus, "labelled")] += labelled
        base[(corpus, "committed")] += committed
        base[(corpus, "committed_labelled")] += committed and labelled
        base[(corpus, "full")] += full
        base[(corpus, "full_labelled")] += full and labelled
        for name, attrs in SWITCHES.items():
            with _patched(ec_new, attrs):
                off = bool(ec_new.find_numeric_mismatches(claim, passage))
            if full != off:
                kind = "gained" if full else "lost"
                effect[name][(corpus, kind)] += 1
                effect[name][(corpus, f"{kind}_labelled")] += labelled
                if len(samples[name][kind]) < 400:
                    samples[name][kind].append({"corpus": corpus, "labelled": labelled, "claim": claim[:240]})
    corpora = sorted({c for c, _ in base})
    table = {c: {k: base[(c, k)] for k in ("pairs", "labelled", "committed", "committed_labelled", "full", "full_labelled", "committed_raised")} for c in corpora}
    verdicts = {}
    for name in SWITCHES:
        rows, failed, evidence = {}, [], False
        for c in corpora:
            p = table[c]["committed_labelled"] / table[c]["committed"] if table[c]["committed"] else None
            row = {k: effect[name][(c, k)] for k in ("gained", "gained_labelled", "lost", "lost_labelled")}
            row["P"] = p
            rows[c] = row
            if p is None:
                continue
            if row["gained"] >= MIN_N:
                evidence = True
                if row["gained_labelled"] / row["gained"] < p:
                    failed.append(f"{c} gains")
            if row["lost"] >= MIN_N:
                evidence = True
                if row["lost_labelled"] / row["lost"] > p:
                    failed.append(f"{c} losses")
        verdict = "FAIL" if failed else ("PASS" if evidence else "INCONCLUSIVE")
        verdicts[name] = {"verdict": verdict, "failed_on": failed, "per_corpus": rows,
                          "samples": {k: random.Random(0).sample(v, min(12, len(v))) for k, v in samples[name].items()}}
    return {"corpora": table, "rules": verdicts}


def _answer_documents():
    for t in _halueval("general")["chatgpt_response"].to_list():
        yield "halueval-general", t
    for t in _halueval("summarization")["right_summary"].to_list():
        yield "halueval-summarization", t
    for t in _psiloqa().filter(pl.col("lang") == "en")["llm_answer"].to_list():
        yield "psiloqa-en", t


def measure_rejoin_gate() -> dict:
    """Claims from no rejoin, the gated rejoin and the ungated rejoin."""
    counts: Counter = Counter()
    rejoins = []
    for corpus, text in _answer_documents():
        text = text or ""
        with _patched(ex_new, {"_ABBREV_SHAPE_RE": NEVER}):
            none = [c.claim for c in ex_new.extract_claims(text)]
        gated = [c.claim for c in ex_new.extract_claims(text)]
        with _patched(ex_new, {"_ABBREV_SHAPE_RE": ALWAYS}):
            ungated = [c.claim for c in ex_new.extract_claims(text)]
        counts["documents"] += 1
        counts["gated_changed"] += gated != none
        counts["ungated_changed"] += ungated != none
        for c in gated:
            if c not in none:
                rejoins.append({"corpus": corpus, "claim": c})
    counts["gated_rejoined_claims"] = len(rejoins)
    return {"counts": dict(counts), "judge_sample": random.Random(0).sample(rejoins, min(20, len(rejoins)))}


def measure_language() -> dict:
    lexical._lingua_lang("prime the detector with any text of enough length")
    det = lexical._LINGUA["det"]

    def reads(text: str) -> tuple[str, str]:
        sample = " ".join(s for _, s in ex_new._split_document(text or ""))[:2000]
        if len(sample.strip()) < 25:
            return "und", "und"
        values = det.compute_language_confidence_values(sample)
        top = values[0] if values else None
        non_latin = next((c for c in values if c.language.name != "LATIN"), None)
        return tuple(
            c.language.iso_code_639_1.name.lower() if c and c.value >= 0.65 else "und" for c in (top, non_latin)
        )

    english = [("halueval-summarization-documents", t) for t in _halueval("summarization")["document"].to_list()]
    english += [("halueval-general", t) for t in _halueval("general")["chatgpt_response"].to_list()]
    english += [("psiloqa-en", t) for t in _psiloqa().filter(pl.col("lang") == "en")["llm_answer"].to_list()]
    latin_before, still_warn = 0, []
    for corpus, text in english:
        committed, now = reads(text)
        if committed == "la":
            latin_before += 1
            if now not in ("en", "und"):
                still_warn.append({"corpus": corpus, "now": now})
    changed = []
    non_en = _psiloqa().filter(pl.col("lang") != "en")
    for lang, text in zip(non_en["lang"].to_list(), non_en["llm_answer"].to_list()):
        committed, now = reads(text)
        if committed != now:
            changed.append({"lang": lang, "committed": committed, "now": now})
    verdict = "PASS" if not still_warn and not changed else "FAIL"
    return {"verdict": verdict, "english_documents": len(english), "english_read_as_latin_before": latin_before,
            "still_warning": still_warn, "non_english_answers": non_en.height, "non_english_changed": changed}


def measure_modal_conditional() -> dict:
    def rate(texts) -> tuple[int, int]:
        n = hits = 0
        for text in texts:
            for _, sent in ex_new._split_document(text or ""):
                if not ex_new._looks_like_claim(sent):
                    continue
                n += 1
                hits += ex_old.out_of_scope(sent) == "hypothetical" and not ex_old._EITHER_CLAUSES_RE.match(sent)
        return n, hits

    sources = _halueval("summarization")["document"].to_list() + _halueval("qa")["knowledge"].to_list()
    sources += [r["evidence"] for r in _vitaminc_rows()[:20000]]
    answers = _halueval("general")["chatgpt_response"].to_list() + _halueval("summarization")["right_summary"].to_list()
    answers += _psiloqa().filter(pl.col("lang") == "en")["llm_answer"].to_list()
    (sn, sh), (an, ah) = rate(sources), rate(answers)
    source_rate, answer_rate = sh / sn, ah / an
    return {"verdict": "PASS" if source_rate >= 0.5 * answer_rate else "FAIL",
            "source_sentences": sn, "source_firings": sh, "answer_sentences": an, "answer_firings": ah}


def main() -> None:
    import hashlib
    import subprocess

    diff = subprocess.run(["git", "diff", "HEAD", "--", "src"], cwd=HERE.parents[1], capture_output=True, check=True).stdout
    frozen = hashlib.sha256(diff).hexdigest()[:16]
    if frozen != FROZEN_SRC:
        raise SystemExit(f"src/ changed since HELDOUT.md was written: {frozen} != {FROZEN_SRC}")
    results = {
        "numeric": measure_numeric(),
        "rejoin_gate": measure_rejoin_gate(),
        "language": measure_language(),
        "modal_conditional": measure_modal_conditional(),
    }
    (HERE / "measure_heldout.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({k: v["verdict"] for k, v in results["numeric"]["rules"].items()}, indent=1))
    print(json.dumps({k: results[k].get("verdict", results[k].get("counts")) for k in ("rejoin_gate", "language", "modal_conditional")}, indent=1))


if __name__ == "__main__":
    main()
