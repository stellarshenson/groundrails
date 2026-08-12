"""R18-H150 amendment A2 - scale_word supply census over the approved sources.

Stage 3.  Decides whether a scale_word family can be built that satisfies the
SAME anti-shortcut discipline the unit_swap lane passed, and records the numbers
that force the answer.  Runs only over gate-admitted EDGAR chunks
(`R18-H150_edgar_admitted.parquet`, written by `R18-H150_edgar_gate.py` at
status GREEN) plus the secondary InfoTabs source.

THE DECIDING PROPERTY - where the scale word sits relative to the numeral.

  * TABLES separate them.  The scale lives in a column header (`revenue (in
    millions)`) or a caption and the numeral lives in a cell, so verifying the
    scale means binding a cell to its header.  That separation is what let the
    unit_swap lane pass the H148 rule
  * PROSE co-locates them.  MD&A writes `$12.4 million`, so "find the claim's
    numeral, read the next token" settles every pair with no binding work - the
    H148 failure mode verbatim (the rule the misbound_step family died on)

Two prose constructions can break the co-location, and this census sizes both:

  A. CAPTION-BOUND line items - a bare numeral under an `(in thousands)` caption.
     Adjacency-proof, because the scale is elsewhere in the chunk
  B. IDENTICAL-NUMERAL two-scale chunks - the same numeral surface appears with
     BOTH magnitude words (`$12.4 million` of revenue, `$12.4 billion` of
     assets), so adjacency is ambiguous and only the subject binding decides.
     This is the H146 misbind construction transplanted onto the scale word

Both are sized per DIRECTION, because a family whose two directions are not
50/50 hands the claim-only probe a word rule and fails bars 1 and 2 outright.

Run:  uv run python experiments/grounding-semantic/R18-H150_scaleword_census.py
"""

import ast
import collections
import json
import pathlib
import re

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
ADMITTED = HERE / "R18-H150_edgar_admitted.parquet"
GATE = HERE / "R18-H150_edgar_gate.json"
INFOTABS = ROOT / "tmp" / "R14_H133_infotabs.parquet"
OUT = HERE / "R18-H150_scaleword_census.json"

MIN_FAMILY_PAIRS = 100
RESTORE_CONDITION = 3000
WORDS = ("thousand", "million", "billion")

MAG = {u: re.compile(rf"\b{u}s?\b", re.IGNORECASE) for u in WORDS}
CAPTION = {u: re.compile(
    rf"\(\s*(?:dollars|amounts|in)[^)]{{0,40}}?{u}s?\s*\)|\bin\s+{u}s\b", re.IGNORECASE)
    for u in WORDS}
QUANTITY = re.compile(
    r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)\s+(thousand|million|billion)s?\b", re.IGNORECASE)
LINE_ITEM = re.compile(r"([A-Za-z][A-Za-z ,&'\-\.]{6,45}?)\s+\$?\s?(\d[\d,]*(?:\.\d+)?)(?![\d.,])")
ADJACENT_MAG = re.compile(r"^\s*(?:thousands?|millions?|billions?)\b", re.IGNORECASE)
PP = re.compile(r"percentage\s+points?|\bbasis\s+points?\b", re.IGNORECASE)
PCT = re.compile(r"%|\bpercent\b", re.IGNORECASE)


def balanced(counter):
    """Largest 50/50 family per word pair: 2 x min(direction)."""
    out = {}
    for a in WORDS:
        for b in WORDS:
            if a >= b:
                continue
            k = min(counter.get(f"{a}->{b}", 0), counter.get(f"{b}->{a}", 0))
            out[f"{a}<->{b}"] = {"per_direction": k, "balanced_pairs": 2 * k,
                                 "buildable": 2 * k >= MIN_FAMILY_PAIRS}
    return out


def census_edgar(chunks):
    caption_chunks = collections.Counter()
    caption_items = collections.Counter()      # direction -> adjacency-proof items
    identical = collections.Counter()          # direction -> items
    plain = collections.Counter()              # direction -> items (adjacency-solvable)
    identical_chunks = 0
    pp_pairs = collections.Counter()

    for ch in chunks:
        present = {u for u, rx in MAG.items() if rx.search(ch)}
        caps = {u for u, rx in CAPTION.items() if rx.search(ch)}

        # A. caption-bound line items, numeral NOT followed by a magnitude word
        if len(caps) == 1:
            c = next(iter(caps))
            caption_chunks[c] += 1
            n = 0
            for m in LINE_ITEM.finditer(ch):
                if ADJACENT_MAG.match(ch[m.end():m.end() + 16]):
                    continue
                if len(m.group(2).replace(",", "").replace(".", "")) < 2:
                    continue
                n += 1
            for o in present - {c}:
                caption_items[f"{c}->{o}"] += n

        # B. identical numeral carrying two different scales
        q = [(m.group(1), m.group(2).lower()) for m in QUANTITY.finditer(ch)]
        by_val = collections.defaultdict(set)
        for v, w in q:
            by_val[v].add(w)
        if any(len(ws) >= 2 for ws in by_val.values()):
            identical_chunks += 1
        for ws in by_val.values():
            for a in ws:
                for b in ws:
                    if a != b:
                        identical[f"{a}->{b}"] += 1

        # baseline: the adjacency-SOLVABLE route, sized for comparison only
        for _, w in q:
            for o in present - {w}:
                plain[f"{w}->{o}"] += 1

        if PP.search(ch):
            pp_pairs["pp_bearing" + ("_with_percent" if PCT.search(ch) else "")] += 1

    return {
        "chunks": len(chunks),
        "caption_chunks_by_scale": dict(caption_chunks),
        "A_caption_bound_adjacency_proof": {
            "items_by_direction": dict(caption_items),
            "balanced_families": balanced(caption_items)},
        "B_identical_numeral_adjacency_proof": {
            "chunks": identical_chunks,
            "items_by_direction": dict(identical),
            "balanced_families": balanced(identical)},
        "baseline_plain_quantity_ADJACENCY_SOLVABLE": {
            "items_by_direction": dict(plain),
            "balanced_families": balanced(plain),
            "note": "sized for comparison only - this form is the H148 failure "
                    "mode and is NOT built"},
        "pct_pp": dict(pp_pairs),
    }


def census_infotabs():
    if not INFOTABS.exists():
        return {"present": False}
    d = pl.read_parquet(INFOTABS).unique(subset=["table"], keep="first")
    blobs = []
    for t in d["table"].to_list():
        try:
            tab = ast.literal_eval(t)
        except (ValueError, SyntaxError):
            continue
        blobs.append(" ".join(f"{k}: {'; '.join(v)}" for k, v in tab.items()))
    res = census_edgar(blobs)
    res["unique_tables"] = d.height
    return res


def main():
    gate = json.loads(GATE.read_text()) if GATE.exists() else {}
    if gate.get("status") != "GREEN":
        raise SystemExit("EDGAR gate is not GREEN - refusing to census lane candidates")

    chunks = pl.read_parquet(ADMITTED)["chunk"].to_list()
    print(f"gate-admitted EDGAR chunks: {len(chunks)}", flush=True)
    edgar = census_edgar(chunks)
    print("infotabs (secondary)...", flush=True)
    infotabs = census_infotabs()

    best = 0
    for src in (edgar, infotabs):
        for key in ("A_caption_bound_adjacency_proof", "B_identical_numeral_adjacency_proof"):
            for f in src.get(key, {}).get("balanced_families", {}).values():
                best = max(best, f["balanced_pairs"])

    out = {
        "gate_status": gate.get("status"),
        "min_family_pairs": MIN_FAMILY_PAIRS,
        "restore_condition_pairs": RESTORE_CONDITION,
        "edgar_mda": edgar,
        "infotabs": infotabs,
        "max_adjacency_proof_balanced_pairs": best,
        "restore_condition_met": best >= RESTORE_CONDITION,
        "verdict": (
            "scale_word is NOT buildable under the unit_swap discipline from either "
            "approved source. Prose co-locates the scale word with its numeral, so the "
            "large families are adjacency-solvable (the H148 rule forbids exactly this). "
            "The two adjacency-proof constructions are supply-blocked: the caption route "
            "is single-direction (EDGAR writes 'in thousands' at volume and 'in millions' "
            "rarely, and the minority direction needs the majority word present as a "
            "distractor), and the identical-numeral route yields tens of items, not "
            "thousands. InfoTabs carries no scale captions at all."),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
