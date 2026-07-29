"""R5-H30 - the prose residue is unmeasured.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 5).

The grounder confirms 28/60 in-scope claims on the private prose set. That 46.7%
is read as a failure rate, but only 31 of the 77 claims carry a human label, so
the unconfirmed ones have never been checked. A claim the grounder declines is
either a MISS (the sources support it and retrieval failed) or a CORRECT REFUSAL
(no source states it). This dumps every unconfirmed in-scope claim with two
independent views so the difference can be adjudicated:

  1. what the GROUNDER found - best matched span per layer, with scores
  2. an INDEPENDENT anchor search over the raw source text - top sentences by
     claim-token overlap, plus every sentence containing any number from the
     claim

The second view exists because adjudicating a refusal from the grounder's own
retrieval is circular: if retrieval is what failed, its top-k is exactly the
evidence that will not show the miss.

Run:  uv run python experiments/grounding-semantic/R5-H30_adjudicate_residue.py
"""

import json
from pathlib import Path
import re

from groundrails import settings
from groundrails.grounding import ground_batch

settings.mark_ready()

HERE = Path(__file__).parent / "private-prose-forensics"
# Inside the gitignored forensics directory: the dump carries verbatim client
# prose from the sources, so it must never land in a committed artefact.
OUT = HERE / "R5-H30_residue.json"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_NUM_RE = re.compile(r"\d[\d,.]*%?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_STOP_WORDS = (
    "the and for with that this from will has have are was were been being can could "
    "would should may might must their its our your they them these those such which "
    "when where while into onto over under about across per not but also than then "
    "each other more most some any all both very much many few own same"
)
_STOP = frozenset(_STOP_WORDS.split())


def content_tokens(text):
    return {w.lower() for w in _WORD_RE.findall(text)} - _STOP


def sentences(text):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def independent_search(claim, sources, top=5):
    """Rank every sentence in the corpus by raw claim-token overlap.

    Deliberately NOT the grounder's retrieval - no BM25 weighting, no chunking,
    no embeddings. A dumb exhaustive scan cannot miss for the same reason the
    grounder might.
    """
    ctoks = content_tokens(claim)
    cnums = set(_NUM_RE.findall(claim))
    scored, numeric_hits = [], []
    for name, text in sources:
        for sent in sentences(text):
            stoks = content_tokens(sent)
            hit = ctoks & stoks
            if hit:
                scored.append((len(hit) / max(len(ctoks), 1), name, sent[:400], sorted(hit)))
            if cnums and cnums & set(_NUM_RE.findall(sent)):
                numeric_hits.append((name, sent[:400], sorted(cnums & set(_NUM_RE.findall(sent)))))
    scored.sort(key=lambda r: -r[0])
    return scored[:top], numeric_hits[:top], sorted(cnums)


def main():
    claims = [c["claim"] for c in json.loads((HERE / "claims-current.json").read_text())]
    sources = [
        (p.name, p.read_text(encoding="utf-8")) for p in sorted((HERE / "sources").iterdir())
    ]
    labels = [json.loads(x) for x in (HERE / "labels.jsonl").read_text().splitlines()]
    by_text = {r["claim"]: r["human_label"] for r in labels if "claim" in r}

    matches = ground_batch(claims, sources, semantic=True)
    residue = [
        (c, m)
        for c, m in zip(claims, matches, strict=True)
        if not m.grounded and not m.out_of_scope_reason
    ]
    confirmed = sum(1 for m in matches if m.grounded and not m.out_of_scope_reason)
    in_scope = sum(1 for m in matches if not m.out_of_scope_reason)
    print(
        f"claims {len(claims)}  in-scope {in_scope}  confirmed {confirmed} "
        f"({confirmed / in_scope:.1%})  residue to adjudicate {len(residue)}",
        flush=True,
    )
    print(f"of the residue, {sum(1 for c, _ in residue if c in by_text)} carry a human label\n")

    records = []
    for i, (claim, m) in enumerate(residue):
        top, numeric, cnums = independent_search(claim, sources)
        records.append(
            {
                "idx": i,
                "claim": claim,
                "human_label": by_text.get(claim),
                "grounder": {
                    "match_type": m.match_type,
                    "combined_score": round(m.combined_score, 4),
                    "fuzzy": [round(m.fuzzy_score, 3), m.fuzzy_matched_text[:300]],
                    "bm25": [round(m.bm25_score, 3), m.bm25_matched_text[:300]],
                    "semantic": [round(m.semantic_score, 3), m.semantic_matched_text[:300]],
                },
                "claim_numbers": cnums,
                "independent_top": [
                    {"overlap": round(s, 3), "source": n, "sentence": t, "shared": h}
                    for s, n, t, h in top
                ],
                "independent_numeric": [
                    {"source": n, "sentence": t, "numbers": v} for n, t, v in numeric
                ],
                "adjudication": None,
            }
        )
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"residue dumped -> {OUT}  ({len(records)} claims awaiting adjudication)")


if __name__ == "__main__":
    main()
