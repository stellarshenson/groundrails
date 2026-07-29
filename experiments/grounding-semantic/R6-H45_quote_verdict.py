"""R6-H45 - the quote is the verdict.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 6).
Re-analysis of the R6-H42 generations at zero marginal compute.

R6-H42 established that Pleias-RAG-350M, on a harness validated against its own
published example, is a BAD JUDGE: 11/20 on trivially separable pairs, 0 false
negatives and 9 false positives - it says "yes, supported" against a sourdough
recipe. But it is a model trained to emit the literal supporting excerpt inline,
and a fabricated excerpt is checkable in a way a fabricated verdict is not:

    <ref name="1">The dataset spans 964 sensors</ref>   against a bread recipe

is a claim the source text refutes by simple substring. So the model never
judges. It quotes, and deterministic code decides whether the quote is real.

Run:  uv run python experiments/grounding-semantic/R6-H45_quote_verdict.py
"""

import importlib.util
import json
import pathlib
import re

from rapidfuzz import fuzz

from groundrails.entity_check import extract_entities, extract_numbers


def _load_h42():
    """The generations file stores no source text, so the cases are rebuilt from
    the generator module itself - same constants, no re-run, no drift."""
    path = pathlib.Path(__file__).parent / "R6-H42_pleias_rag_protocol.py"
    spec = importlib.util.spec_from_file_location("h42", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [ev for _, ev in mod.CONTROL] + [mod.RECIPE] * len(mod.CONTROL)


GEN = pathlib.Path(__file__).parent / "R6-H42_generations.json"
CITATION_RE = re.compile(r'<ref name="(?:<\|source_id\|>)?(\d+)">(.*?)</ref>', re.DOTALL)
FUZZ_FLOOR = 97.0  # the model elides long excerpts with "(...)"; 90 false-matched short spans
_PROMPT_ECHO_RE = re.compile(r"^\s*(?:claim|query|question)\s*:\s*", re.IGNORECASE)


def normalise(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def anchors(text):
    """The claim's checkable content: numeric values plus named entities."""
    return {v for v, _, _ in extract_numbers(text)} | {e.lower() for e in extract_entities(text)}


def quote_supports_claim(quote, claim):
    """Half two of the rule: a genuine quote that shares none of the claim's
    anchors is a real excerpt of an irrelevant passage, not support. Case 18 of
    the control is exactly this - the model correctly quoted the recipe, which
    says nothing about the claim."""
    a_claim = anchors(claim)
    if not a_claim:
        return True  # nothing anchorable; fall back to the substring evidence alone
    return bool(a_claim & anchors(quote))


def quote_is_real(quote, source):
    """Deterministic: is this excerpt actually in the source text?"""
    q, s = normalise(_PROMPT_ECHO_RE.sub("", quote)), normalise(source)
    if not q:
        return False
    if q in s:
        return True
    # "(...)" elision - check each retained fragment independently
    frags = [f for f in (normalise(x) for x in quote.split("(...)")) if len(f) > 12]
    if len(frags) > 1 and all(f in s for f in frags):
        return True
    return fuzz.partial_ratio(q, s) >= FUZZ_FLOOR


def main():
    data = json.loads(GEN.read_text())
    rows = data["control"]
    sources = _load_h42()
    # Reconstruct each case's source: the first ten carry their own chunk, the
    # last ten all share the recipe. Stored on the row by the generator.
    tp = fp = tn = fn = 0
    noquote = 0
    print(f"{'#':>3} {'expect':6} {'judge':6} {'quotes':>6} {'real':>5} {'verdict':8}  detail")
    for i, r in enumerate(rows):
        source = r.get("source") or sources[i]
        quotes = [q for _, q in CITATION_RE.findall(r.get("raw", ""))]
        real = [
            q for q in quotes if quote_is_real(q, source) and quote_supports_claim(q, r["claim"])
        ]
        if not quotes:
            noquote += 1
            # Grounded iff some emitted quote is BOTH genuinely in the source
        # and anchored to the claim. The model never judges.
        verdict = bool(real)
        expect = r["expect"]
        if verdict and expect:
            tp += 1
        elif verdict and not expect:
            fp += 1
        elif not verdict and not expect:
            tn += 1
        else:
            fn += 1
        mark = "OK" if verdict == expect else "WRONG"
        print(
            f"{i:>3} {'SUP' if expect else 'UNSUP':6} {r['got']!s:6} {len(quotes):>6} "
            f"{len(real):>5} {verdict!s:8}  {mark}  {(quotes[0][:70] if quotes else '-')!r}"
        )

    n = len(rows)
    correct = tp + tn
    print(f"\n  cases {n}, cases with no emitted quote: {noquote}")
    print(f"  QUOTE verdict : {correct}/{n} ({correct / n:.0%})   FP {fp}   FN {fn}")
    print(
        f"  JUDGE verdict : {data['correct']}/{n} ({data['correct'] / n:.0%})   "
        f"FP {data['fp']}   FN {data['fn']}   (R6-H42)"
    )
    print("  reference     : R4-H29 Baguettotron 12/20 (60%), 0 FN / 4 FP")
    print(
        f"\n  bar: the quote verdict must beat every judged prompt  ->  "
        f"{'PASS' if correct > data['correct'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
