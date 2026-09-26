"""Round-2 test of the defect designs, run once on fresh data under ``HELDOUT2.md``.

``--design`` scores the same rules on the design corpora (every corpus read before
round 2) and is what the designs were chosen on; the default reads the test corpora
named in ``HELDOUT2.md`` and writes ``measure_heldout2.json``.

Usage: ``HF_HUB_OFFLINE=1 .venv/bin/python experiments/defects/measure_heldout2.py [--design]``

The rules it switches were reverted after review (``HELDOUT2.md``, Reduction after
review). It runs on the code it tested: ``git apply experiments/defects/heldout2-frozen-src.patch``
on a checkout of ``65451f1``.
"""

from __future__ import annotations

from collections import Counter
import decimal
import glob
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_heldout as round1  # noqa: E402
from measure_open_defects import HUB, _numeric_pairs, ec_new, ec_old, ex_new  # noqa: E402

from groundrails import lexical  # noqa: E402

HERE = Path(__file__).resolve().parent
DESIGN = "--design" in sys.argv
FROZEN_SRC = "8a1b776d8f49a535"  # HELDOUT2.md: git diff HEAD -- src | sha256sum
NEVER = re.compile(r"(?!)")
MIN_N = 10
COMMITTED_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
NO_PAREN_SPLIT = re.compile(ex_new._SENT_SPLIT_RE.pattern.replace("\\u2018(]", "\\u2018]"))
assert NO_PAREN_SPLIT.pattern != ex_new._SENT_SPLIT_RE.pattern

NUMERIC_SWITCHES = {
    "DEF-NUMBER-32 same-quantity rounding": {"_same_quantity_agrees": ec_new._agree_at_coarser_precision},
    "DEF-NUMBER-44 year span in either order": {"_year_span_key": lambda v: v},
    "DEF-NUMBER-46 currency comparative": {"_COMPARATIVE_RE": ec_old._COMPARATIVE_RE},
}
JUDGED_SWITCHES = {
    "DEF-CLAIM-39/40 rejoin gate": {"_ABBREV_SHAPE_RE": NEVER},
    "DEF-CLAIM-41 parenthesis split": {"_SENT_SPLIT_RE": NO_PAREN_SPLIT},
    "DEF-CLAIM-45 outline items": {"_LIST_PREFIX_RE": COMMITTED_LIST_PREFIX},
}


def _file(pattern: str) -> str:
    return sorted(glob.glob(str(HUB / pattern)))[0]


def _attribution_rows() -> list[dict]:
    seen, rows = set(), []
    for p in sorted(glob.glob(str(HUB / "datasets--osunlp--AttributionBench/snapshots/*/*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if r["id"] in seen or r["src_dataset"] in ("ExpertQA", "HAGRID"):
                continue
            seen.add(r["id"])
            rows.append(r)
    return rows


def _references(r: dict) -> str:
    refs = r.get("references") or []
    return "\n".join(refs) if isinstance(refs, list) else str(refs)


def _drop_passages() -> list[str]:
    frames = [pl.read_parquet(p, columns=["passage"]) for p in sorted(glob.glob(str(HUB / "datasets--ucinlp--drop/snapshots/*/data/*.parquet")))]
    return pl.concat(frames)["passage"].unique(maintain_order=True).to_list()


def _fava_outputs() -> list[str]:
    return [r["output"] for r in json.load(open(_file("datasets--fava-uw--fava-data/snapshots/*/annotations.json")))]


def numeric_pairs():
    if DESIGN:
        yield from _numeric_pairs()
        yield from round1.numeric_pairs()
        return
    for r in _attribution_rows():
        claim = r.get("claim") or ""
        if any(ch.isdigit() for ch in claim):
            yield f"attributionbench-{r['src_dataset']}", claim, _references(r), r["attribution_label"] == "not attributable"


def english_documents():
    if DESIGN:
        yield from round1._answer_documents()
        return
    for r in _attribution_rows():
        if r.get("response"):
            yield "attributionbench", r["response"]
    for t in _drop_passages():
        yield "drop", t
    for t in _fava_outputs():
        yield "fava", t


def non_english_documents():
    split = "test" if DESIGN else "train"
    for lang in ("pl", "de", "es", "fr", "it", "hu", "cn"):
        p = _file(f"datasets--KRLabsOrg--ragtruth-{lang}-translated/snapshots/*/data/{split}-*.parquet")
        for t in pl.read_parquet(p, columns=["answer"])["answer"].to_list()[: 500 if DESIGN else 2000]:
            yield lang, t


def measure_numeric() -> dict:
    base: Counter = Counter()
    effect = {name: Counter() for name in NUMERIC_SWITCHES}
    samples = {name: [] for name in NUMERIC_SWITCHES}
    for corpus, claim, passage, labelled in numeric_pairs():
        try:
            committed = bool(ec_old.find_numeric_mismatches(claim, passage))
        except decimal.InvalidOperation:
            committed = False
        full = bool(ec_new.find_numeric_mismatches(claim, passage))
        base[(corpus, "pairs")] += 1
        base[(corpus, "committed")] += committed
        base[(corpus, "committed_labelled")] += committed and labelled
        for name, attrs in NUMERIC_SWITCHES.items():
            with round1._patched(ec_new, attrs):
                off = bool(ec_new.find_numeric_mismatches(claim, passage))
            if full != off:
                kind = "gained" if full else "lost"
                effect[name][(corpus, kind)] += 1
                effect[name][(corpus, f"{kind}_labelled")] += labelled
                samples[name].append({"corpus": corpus, "kind": kind, "labelled": labelled, "claim": claim[:240]})
    corpora = sorted({c for c, _ in base})
    verdicts = {}
    for name in NUMERIC_SWITCHES:
        rows, failed, evidence = {}, [], False
        for c in corpora:
            p = base[(c, "committed_labelled")] / base[(c, "committed")] if base[(c, "committed")] else None
            row = {k: effect[name][(c, k)] for k in ("gained", "gained_labelled", "lost", "lost_labelled")}
            row["P"] = p
            if row["gained"] or row["lost"]:
                rows[c] = row
            if p is None:
                continue
            for kind, worse in (("gained", lambda s: s < p), ("lost", lambda s: s > p)):
                if row[kind] >= MIN_N:
                    evidence = True
                    if worse(row[f"{kind}_labelled"] / row[kind]):
                        failed.append(f"{c} {kind}")
        verdicts[name] = {"verdict": "FAIL" if failed else ("PASS" if evidence else "INCONCLUSIVE"), "failed_on": failed,
                          "per_corpus": rows, "samples": random.Random(0).sample(samples[name], min(20, len(samples[name])))}
    return {"corpora": {c: {k: base[(c, k)] for k in ("pairs", "committed", "committed_labelled")} for c in corpora}, "rules": verdicts}


def measure_judged() -> dict:
    changed = {name: [] for name in JUDGED_SWITCHES}
    documents = 0
    for corpus, text in english_documents():
        documents += 1
        full = [c.claim for c in ex_new.extract_claims(text or "")]
        for name, attrs in JUDGED_SWITCHES.items():
            with round1._patched(ex_new, attrs):
                off = {c.claim for c in ex_new.extract_claims(text or "")}
            changed[name] += [{"corpus": corpus, "claim": c} for c in full if c not in off]
    return {"documents": documents, "rules": {
        name: {"changed_units": len(v), "judge_sample": random.Random(0).sample(v, min(20, len(v)))} for name, v in changed.items()}}


def measure_language() -> dict:
    lexical._lingua_lang("prime the detector with any text of enough length")

    def reads(text: str) -> tuple[bool, bool]:
        sample = " ".join(s for _, s in ex_new._split_document(text or ""))[:2000]
        new = lexical.detect_lang_confident(sample)
        saved = lexical._LATIN_ENGLISH_MIN
        lexical._LATIN_ENGLISH_MIN = 2.0
        try:
            committed = lexical.detect_lang_confident(sample)
        finally:
            lexical._LATIN_ENGLISH_MIN = saved
        return committed not in ("en", "und"), new not in ("en", "und")

    english = Counter()
    for corpus, text in english_documents():
        warned, warns = reads(text)
        english[(corpus, "documents")] += 1
        english[(corpus, "committed_warned")] += warned
        english[(corpus, "now_warns")] += warns
    stopped, counts = [], Counter()
    for lang, text in non_english_documents():
        warned, warns = reads(text)
        counts[(lang, "documents")] += 1
        counts[(lang, "committed_warned")] += warned
        counts[(lang, "now_warns")] += warns
        if warned and not warns:
            stopped.append({"lang": lang, "text": (text or "")[:200]})
    return {"verdict": "FAIL" if stopped else "PASS", "non_english_stopped_warning": stopped,
            "english": {f"{c}|{k}": n for (c, k), n in english.items()},
            "non_english": {f"{c}|{k}": n for (c, k), n in counts.items()}}


def main() -> None:
    if not DESIGN:
        diff = subprocess.run(["git", "diff", "HEAD", "--", "src"], cwd=HERE.parents[1], capture_output=True, check=True).stdout
        frozen = hashlib.sha256(diff).hexdigest()[:16]
        if frozen != FROZEN_SRC:
            raise SystemExit(f"src/ does not match HELDOUT2.md: {frozen} != {FROZEN_SRC}")
    results = {"mode": "design" if DESIGN else "test", "numeric": measure_numeric(), "judged": measure_judged(), "language": measure_language()}
    out = HERE / ("measure_heldout2_design.json" if DESIGN else "measure_heldout2.json")
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({k: v["verdict"] for k, v in results["numeric"]["rules"].items()} | {"language": results["language"]["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
