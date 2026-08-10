"""R16-H140 G0 - dispersion census (CPU, analysis-only).

Pre-registered gate G0 of R16-H140 (window-ensemble readout). For every grounded
sentence in the blind R8-H77 arena (label-1 responses, ANALYSIS-ONLY - no
training, no tuning) and in the 2,752-row private gold held-out set, measure
whether the sentence's lexical anchors (content words len>=4 minus stopwords,
plus all numerals) that ALSO occur in the evidence can co-occur inside a single
1,500-char serving window (stride 750, geometry byte-identical to
R8-H101_windowed_read.windows).

  cross-window mass = fraction of anchored grounded sentences for which NO single
                      window contains every evidence-matched anchor

KILL (registered): mass < 3% on EVERY subset -> the lever is capped below noise,
G1 does not run.

Also banked: the minimal char-span covering all matched anchors (bucketed), and
as a free rider the window-boundary evidence-sentence cut statistics.

Run:  nohup setsid uv run python experiments/grounding-semantic/R16-H140_G0_census.py \
        >> logs/R16-H140_pilot.log 2>&1 &
"""

from collections import Counter
import importlib.util
import json
import pathlib
import re
import sys
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R16-H140_G0_census.json"
GOLD_PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"

WIN = 1500
STRIDE = 750


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H92 = _mod("h92", "R8-H92_decomposed_arena.py")

# --- window geometry: byte-identical to R8-H101_windowed_read.windows ---------


def windows(chunk):
    n = len(chunk)
    if n <= WIN:
        return [(0, n)]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [(s, s + WIN) for s in starts]


# --- anchors ------------------------------------------------------------------

STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "already", "also",
    "although", "always", "among", "another", "back", "because", "been", "before",
    "being", "below", "between", "both", "cannot", "come", "could", "does", "doing",
    "done", "down", "during", "each", "either", "else", "enough", "even", "ever",
    "every", "from", "further", "give", "goes", "gone", "have", "having", "here",
    "hers", "herself", "himself", "however", "into", "itself", "just", "keep",
    "know", "less", "like", "made", "make", "many", "may", "maybe", "might",
    "more", "most", "much", "must", "near", "need", "never", "next", "none",
    "only", "onto", "other", "others", "ours", "over", "own", "part", "per",
    "perhaps", "rather", "same", "seem", "shall", "should", "since", "some",
    "still", "such", "sure", "take", "than", "that", "their", "theirs", "them",
    "themselves", "then", "there", "therefore", "these", "they", "thing", "things",
    "this", "those", "though", "through", "thus", "together", "toward", "towards",
    "under", "unless", "until", "upon", "used", "using", "very", "want", "well",
    "were", "what", "when", "where", "whether", "which", "while", "whom", "whose",
    "will", "with", "within", "without", "would", "your", "yours", "yourself",
}

_WORD = re.compile(r"[a-z][a-z\-']{3,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def anchors_of(sentence):
    low = sentence.lower()
    words = {w for w in _WORD.findall(low) if w not in STOPWORDS}
    nums = set(_NUM.findall(low))
    return sorted(words | nums), len(words), len(nums)


def _pattern(anchor):
    forms = {anchor}
    if "," in anchor:
        forms.add(anchor.replace(",", ""))
    return re.compile("|".join(r"\b" + re.escape(f) + r"\b" for f in sorted(forms, key=len, reverse=True)))


_PAT_CACHE = {}


def pattern(anchor):
    p = _PAT_CACHE.get(anchor)
    if p is None:
        p = _pattern(anchor)
        if len(_PAT_CACHE) < 200_000:
            _PAT_CACHE[anchor] = p
    return p


def occurrences(low_chunk, anchor, cache):
    hit = cache.get(anchor)
    if hit is None:
        hit = [(m.start(), m.end()) for m in pattern(anchor).finditer(low_chunk)]
        cache[anchor] = hit
    return hit


# --- per-sentence dispersion test ---------------------------------------------


def min_span_in_chunk(occ_lists):
    """Classic minimal window over merged occurrence lists (all anchors present)."""
    merged = []
    for i, occs in enumerate(occ_lists):
        for s, e in occs:
            merged.append((s, e, i))
    merged.sort()
    need = len(occ_lists)
    have = Counter()
    best = None
    left = 0
    for right in range(len(merged)):
        have[merged[right][2]] += 1
        while len(have) == need:
            # span measured from the left occurrence's start to the right
            # occurrence's end (sorted by start, so the right element closes the
            # window up to <=1 anchor length of slack)
            span = merged[right][1] - merged[left][0]
            if best is None or span < best:
                best = span
            have[merged[left][2]] -= 1
            if have[merged[left][2]] == 0:
                del have[merged[left][2]]
            left += 1
    return best


def sentence_dispersion(sentence, low_chunks, win_lists, caches):
    """Returns dict for one grounded sentence, or None if it has no matched anchor."""
    anc, n_words, n_nums = anchors_of(sentence)
    if not anc:
        return None

    # occurrences per chunk per anchor
    per_chunk = []
    matched = set()
    for ci, low in enumerate(low_chunks):
        d = {}
        for a in anc:
            occ = occurrences(low, a, caches[ci])
            if occ:
                d[a] = occ
                matched.add(a)
        per_chunk.append(d)

    if not matched:
        return None
    matched = sorted(matched)
    nm = len(matched)

    # covered by a single window?
    within = False
    for ci, d in enumerate(per_chunk):
        if len(d) < nm:
            continue
        for (ws, we) in win_lists[ci]:
            ok = True
            for a in matched:
                if not any(s >= ws and e <= we for s, e in d[a]):
                    ok = False
                    break
            if ok:
                within = True
                break
        if within:
            break

    # minimal covering span (within a single chunk); None if no chunk holds all
    span = None
    for d in per_chunk:
        if len(d) < nm:
            continue
        s = min_span_in_chunk([d[a] for a in matched])
        if s is not None and (span is None or s < span):
            span = s

    return {
        "n_anchors": len(anc),
        "n_matched": nm,
        "n_word_anchors": n_words,
        "n_num_anchors": n_nums,
        "within_window": within,
        "min_span": span,          # None => no single chunk contains all anchors
        "cross_chunk": span is None,
    }


# --- boundary-cut free rider ---------------------------------------------------

_SPLIT_ITER = re.compile(r"(?<=[.!?])\s+")


def boundary_cuts(chunk):
    """Evidence sentences that NO single window contains in full."""
    wins = windows(chunk)
    bounds, prev = [], 0
    for m in _SPLIT_ITER.finditer(chunk):
        bounds.append((prev, m.start()))
        prev = m.end()
    bounds.append((prev, len(chunk)))
    bounds = [(a, b) for a, b in bounds if b - a >= H92.MIN_SENT_CHARS]
    n_total = len(bounds)
    n_cut = 0
    n_longer_than_window = 0
    for a, b in bounds:
        if b - a > WIN:
            n_longer_than_window += 1
            n_cut += 1
            continue
        if not any(a >= ws and b <= we for ws, we in wins):
            n_cut += 1
    return n_total, n_cut, n_longer_than_window


# --- aggregation ---------------------------------------------------------------

BUCKETS = ("<=1500", "1500-3000", "3000-6000", ">6000", "cross_chunk_uncovered")


def bucket_of(rec):
    if rec["min_span"] is None:
        return "cross_chunk_uncovered"
    s = rec["min_span"]
    if s <= 1500:
        return "<=1500"
    if s <= 3000:
        return "1500-3000"
    if s <= 6000:
        return "3000-6000"
    return ">6000"


def summarise(name, recs, cut_total, cut_n, cut_long, n_items, n_sent_no_anchor):
    n = len(recs)
    cross = [r for r in recs if not r["within_window"]]
    hist = Counter(bucket_of(r) for r in recs)
    spans = [r["min_span"] for r in recs if r["min_span"] is not None]
    return {
        "subset": name,
        "n_grounded_items": n_items,
        "n_grounded_sentences_anchored": n,
        "n_grounded_sentences_no_matched_anchor": n_sent_no_anchor,
        "cross_window_n": len(cross),
        "cross_window_mass_pct": round(100.0 * len(cross) / n, 3) if n else None,
        "cross_chunk_n": sum(1 for r in recs if r["cross_chunk"]),
        "cross_chunk_pct": round(100.0 * sum(1 for r in recs if r["cross_chunk"]) / n, 3) if n else None,
        "anchor_span_hist": {b: hist.get(b, 0) for b in BUCKETS},
        "anchor_span_pct": {b: round(100.0 * hist.get(b, 0) / n, 3) for b in BUCKETS} if n else None,
        "min_span_median": float(np.median(spans)) if spans else None,
        "min_span_p90": float(np.percentile(spans, 90)) if spans else None,
        "mean_matched_anchors": round(float(np.mean([r["n_matched"] for r in recs])), 2) if n else None,
        "evidence_sentences_total": cut_total,
        "evidence_sentences_cut_by_every_window": cut_n,
        "evidence_sentences_cut_pct": round(100.0 * cut_n / cut_total, 3) if cut_total else None,
        "evidence_sentences_longer_than_window": cut_long,
    }


def run_group(name, items):
    """items: iterable of (sentences, chunks)."""
    t0 = time.time()
    recs = []
    n_no_anchor = 0
    cut_total = cut_n = cut_long = 0
    seen_chunks = set()
    idx = -1
    for idx, (sents, chunks) in enumerate(items):
        low_chunks = [c.lower() for c in chunks]
        win_lists = [windows(c) for c in chunks]
        caches = [{} for _ in chunks]
        for s in sents:
            r = sentence_dispersion(s, low_chunks, win_lists, caches)
            if r is None:
                n_no_anchor += 1
            else:
                recs.append(r)
        for c in chunks:
            h = hash(c)
            if h in seen_chunks:
                continue
            seen_chunks.add(h)
            t, cn, cl = boundary_cuts(c)
            cut_total += t
            cut_n += cn
            cut_long += cl
        if (idx + 1) % 50 == 0:
            print(f"    {name}: {idx + 1} items, {len(recs)} sentences, {time.time() - t0:.0f}s", flush=True)
    n_items = idx + 1
    print(f"  {name}: done in {time.time() - t0:.0f}s", flush=True)
    return summarise(name, recs, cut_total, cut_n, cut_long, n_items, n_no_anchor)


def main():
    print(f"=== R16-H140 G0 dispersion census {time.strftime('%F %T')} ===", flush=True)
    results = {}

    # --- arena (analysis-only) -------------------------------------------------
    ARENA = _mod("arena", "R8-H77_unseen_arena.py")
    subs = ARENA.load_subsets()
    print(f"arena: {len(subs)} subsets", flush=True)
    for sub, (claims, chunks, y) in subs.items():
        items = [(H92.sentences(c), ks) for c, ks, lab in zip(claims, chunks, y, strict=True) if lab == 1]
        print(f"  {sub}: {len(items)} grounded responses", flush=True)
        results[sub] = run_group(sub, items)
        results[sub]["source"] = "R8-H77 arena (RAGBench, blind, analysis-only)"

    # --- private gold held-out --------------------------------------------------
    df = pl.read_parquet(GOLD_PAIRS)
    gold_items = []
    for _o, grp in df.group_by("owner"):
        if int(grp["label"][0]) != 1:
            continue
        gold_items.append(([grp["claim"][0]], grp["chunk"].to_list()))
    print(f"  gold_full: {len(gold_items)} grounded claims", flush=True)
    results["gold_full"] = run_group("gold_full", gold_items)
    results["gold_full"]["source"] = "private gold held-out (R7-H51 teacher pairs, 2,752 owners)"
    results["gold_full"]["geometry_note"] = (
        "gold evidence arrives PRE-CHUNKED at <=1500 chars, so one chunk == one window and the "
        "serving read's 750-char overlap does not apply; gold cross-window mass is therefore "
        "cross-CHUNK dispersion and is an UPPER bound relative to arena geometry"
    )

    arena_masses = {k: v["cross_window_mass_pct"] for k, v in results.items() if k != "gold_full"}
    all_masses = {k: v["cross_window_mass_pct"] for k, v in results.items()}
    kill = all(m is not None and m < 3.0 for m in arena_masses.values())

    payload = {
        "gate": "R16-H140 G0 dispersion census",
        "window": WIN,
        "stride": STRIDE,
        "sentence_splitter": "R8-H92.sentences (terminal-punct regex, min 25 chars, cap 12)",
        "anchor_definition": (
            "content words = lowercase [a-z][a-z\\-']{3,} tokens (len>=4) minus a 130-term English "
            "stopword list; numerals = \\d+(?:[.,]\\d+)* ; an anchor is MATCHED if it occurs "
            "(word-boundary, comma-stripped variant allowed for numerals) anywhere in the evidence"
        ),
        "cross_window_definition": (
            "no single 1500-char window of any single evidence chunk contains an occurrence of every "
            "matched anchor; anchors spread across two chunks are cross-window by construction"
        ),
        "per_subset": results,
        "cross_window_mass_pct": all_masses,
        "arena_cross_window_mass_pct": arena_masses,
        "kill_bar": "cross-window mass < 3% on EVERY arena subset",
        "kill_triggered": bool(kill),
        "verdict": "KILL - G1 does not run" if kill else "PASS - G1 licensed",
        "caveat": (
            "Lexical-anchor co-occurrence is a PROXY for evidential sufficiency and errs in BOTH "
            "directions. It OVERCOUNTS cross-window mass: a sentence whose anchors are scattered may "
            "still be fully verifiable from one window (repeated entities, anchors that are incidental "
            "surface tokens rather than load-bearing evidence, or a single window that entails the "
            "claim without containing every surface form). It UNDERCOUNTS: paraphrased or inferentially "
            "supported evidence carries no lexical match at all, so genuinely composed multi-hop "
            "sentences whose support is paraphrased are scored as within-window (or dropped as "
            "unanchored). The census bounds where cross-window information COULD live; it does not "
            "measure how much of it is needed."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 88)
    print("R16-H140 G0 - cross-window mass by subset")
    print("=" * 88)
    for k, v in results.items():
        print(f"  {k:14s} n_sent={v['n_grounded_sentences_anchored']:>5}  "
              f"cross-window {v['cross_window_mass_pct']:>7.3f}%  "
              f"cross-chunk {v['cross_chunk_pct']:>7.3f}%  "
              f"ev-sent cut {v['evidence_sentences_cut_pct']:>6.3f}% "
              f"({v['evidence_sentences_cut_by_every_window']}/{v['evidence_sentences_total']})")
    print(f"\n  kill bar (< 3% on every arena subset): triggered={kill}")
    print(f"  verdict: {payload['verdict']}")
    print(f"  -> {OUT}")
    print("=== G0 CENSUS DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
