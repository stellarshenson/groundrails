"""R22-H183 / H184 / H185 - what carries finqa's 0.66, and is its evidence readable.

Three deterministic, CPU-only measurements over the 250 finqa items of the blind
arena (`R21-H179_arena_items.parquet`):

    H183  per-feature AUROC of RESPONSE-ONLY surface features - which of them,
          if any, reproduces the ablated 0.66 on its own
    H184  per-feature AUROC of NUMBER-ATTRIBUTION features computed from
          (response, evidence) - raw string containment and a scale-aware
          value matcher, reported separately and never combined
    H185  how often a cited number and the row / column label that gives it
          meaning fall in different 1,500/750 serving windows

Every AUROC carries a bootstrap 95% interval (2,000 item resamples, seed 184).
20 negatives against 230 positives: the intervals are the instrument, not the
point estimates.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from groundrails.dataset.manifest import Presentation
from groundrails.dataset.shape import windows as gr_windows

HERE = Path(__file__).resolve().parent
ITEMS = HERE / "R21-H179_arena_items.parquet"
OUT = HERE / "R22-H183_H185_finqa_channels.json"

SUBSET = "finqa"
SEED = 184
N_BOOT = 2000
PRESENTATION = Presentation()  # window_chars=1500, stride_chars=750 - the shipped read
REL_TOL = 0.002  # 0.2% - absorbs restatement rounding ("$911.51 billion" vs 911507)


# --------------------------------------------------------------------------- AUROC


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties. NaN if a leg is empty."""
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ci(y: np.ndarray, s: np.ndarray, *, flip: bool, seed: int = SEED) -> tuple:
    """Bootstrap 95% interval over ITEM resamples; ``flip`` fixes the orientation
    from the full sample so the interval is not biased by per-draw re-orienting."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.sum() in (0, len(yy)):
            continue
        a = auroc(yy, ss)
        draws.append(1.0 - a if flip else a)
    d = np.array(draws)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)


def scored(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    """Oriented AUROC (>= 0.5 by construction) plus its direction and interval."""
    raw = auroc(y, s)
    flip = raw < 0.5
    oriented = 1.0 - raw if flip else raw
    lo, hi, n_ok = auroc_ci(y, s, flip=flip)
    return {
        "feature": name,
        "auroc": round(float(oriented), 5),
        "auroc_raw_for_positive": round(float(raw), 5),
        "direction": "neg" if flip else "pos",
        "ci95": [round(lo, 5), round(hi, 5)],
        "boot_draws_used": n_ok,
        "mean_pos": round(float(s[y == 1].mean()), 5),
        "mean_neg": round(float(s[y == 0].mean()), 5),
    }


# ------------------------------------------------------------------- number parsing

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
SCALE = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6,
         "billion": 1e9, "billions": 1e9, "trillion": 1e12, "trillions": 1e12}
SCALE_WORD_RE = re.compile(r"\b(thousands?|millions?|billions?|trillions?)\b", re.I)
HEADER_SCALE_RE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)


def _digits(tok: str) -> int:
    return sum(c.isdigit() for c in tok)


def parse_numbers(text: str) -> list[dict]:
    """Every numeric token of ``text`` with the context needed to normalise it.

    Kept only if it carries >= 2 digits once commas are stripped, per the
    registered definition. Records the raw span, the comma-stripped string, the
    float value, whether a percent sign or a scale word follows it, whether a
    currency mark precedes it, and whether it sits inside a parenthesised
    negative or a leading minus.
    """
    out = []
    for m in NUM_RE.finditer(text):
        tok = m.group(0)
        stripped = tok.replace(",", "")
        if _digits(stripped) < 2:
            continue
        try:
            val = float(stripped)
        except ValueError:
            continue
        a, b = m.start(), m.end()
        before = text[max(0, a - 12) : a]
        after = text[b : b + 18]
        sw = SCALE_WORD_RE.match(after.lstrip())
        neg = bool(re.search(r"[-−]\s*\$?\s*$", before)) or (
            before.rstrip().endswith("(") and after.lstrip().startswith(")")
        )
        out.append(
            {
                "raw": tok,
                "stripped": stripped,
                "value": -val if neg else val,
                "abs_value": val,
                "start": a,
                "end": b,
                "percent": after.lstrip().startswith("%")
                or bool(re.match(r"\s*(percent|pct)\b", after, re.I)),
                "scale": SCALE[sw.group(1).lower()] if sw else None,
                "currency": bool(re.search(r"[\$£€]\s*$", before)),
                "negative": neg,
            }
        )
    return out


def strip_thousands_commas(text: str) -> str:
    """Remove commas that sit between digits; length changes, so only for the
    raw-containment feature (which needs no offsets)."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def value_variants(n: dict, doc_scale: float | None, *, tight: bool = False) -> set:
    """Every value the token could denote once scale and percent forms resolve.

    ``tight`` drops the unconditional x100 variant, which is the loosest rule in
    the set (it lets any 4.5 match any 450) - carried as a sensitivity arm.
    """
    v = n["value"]
    out = {v}
    if n["scale"]:
        out.add(v * n["scale"])
    if doc_scale:
        out.add(v * doc_scale)
    if n["percent"]:
        out.add(v / 100.0)
    elif not tight:
        out.add(v * 100.0)  # the other side may have written it as a percent
    return out


def close(a: float, b: float, rel_tol: float = REL_TOL) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return False
    return abs(a - b) <= rel_tol * scale


# --------------------------------------------------------------------- H185 anchors


def window_spans(n: int) -> list[tuple[int, int]]:
    """Char spans of the shipped 1,500/750 windows. Verified against the library
    splitter (`groundrails.dataset.shape.windows`) on every document read."""
    win, stride = PRESENTATION.window_chars, PRESENTATION.stride_chars
    if n <= win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def top_level_rows(doc: str) -> list[tuple[int, int]]:
    """Spans of the top-level rows of a `[[...],[...]]` table serialisation."""
    if not doc.lstrip().startswith("[["):
        return []
    rows, depth, in_str, esc, start = [], 0, False, False, None
    for i, ch in enumerate(doc):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "]":
            if depth == 2 and start is not None:
                rows.append((start, i + 1))
                start = None
            depth -= 1
    return rows


def first_cell_span(doc: str, row: tuple[int, int]) -> tuple[int, int]:
    """Span of a table row's first cell - its row label."""
    s, e = row
    m = re.compile(r'"(?:[^"\\]|\\.)*"').search(doc, s, e)
    return (m.start(), m.end()) if m else (s, min(s + 1, e))


def sentence_start(doc: str, p: int) -> int:
    """Start of the prose sentence / line containing offset ``p``."""
    cut = max(doc.rfind(". ", 0, p), doc.rfind("\n", 0, p))
    return 0 if cut < 0 else cut + 1


def binds(spans: list[tuple[int, int]], p: int, anchor: tuple[int, int]) -> bool:
    """Is there one window holding both the token position and the whole anchor?"""
    a0, a1 = anchor
    return any(s <= p < e and s <= a0 and a1 <= e for s, e in spans)


def norm_doc_with_map(doc: str) -> tuple[str, list[int]]:
    """Comma-stripped copy of ``doc`` plus a per-char map back to original offsets."""
    out, mapping = [], []
    for i, ch in enumerate(doc):
        if ch == "," and 0 < i < len(doc) - 1 and doc[i - 1].isdigit() and doc[i + 1].isdigit():
            continue
        out.append(ch)
        mapping.append(i)
    return "".join(out), mapping


def locate(nd: str, tok: str) -> list[int]:
    """Digit-boundary-safe occurrences of ``tok`` in the comma-stripped doc."""
    hits, i = [], nd.find(tok)
    while i >= 0:
        lo_ok = i == 0 or not (nd[i - 1].isdigit() or nd[i - 1] == ".")
        hi = i + len(tok)
        hi_ok = hi >= len(nd) or not (nd[hi].isdigit() or (nd[hi] == "." and hi + 1 < len(nd) and nd[hi + 1].isdigit()))
        if lo_ok and hi_ok:
            hits.append(i)
        i = nd.find(tok, i + 1)
    return hits


# ------------------------------------------------------------------------- features

DERIV_MARKERS = ["=", "therefore", "calculated", "formula", "divided", "multiplied",
                 "percentage", "ratio"]
HEDGES = ["approximately", "about", "roughly", "estimated", "may", "could"]
ABSTAIN = ["cannot", "not provided", "unable", "does not provide", "no information",
           "not specified", "insufficient"]
CURRENCY_RE = re.compile(r"[\$£€¥]")
DIGIT_RUN_RE = re.compile(r"\d+")


def response_features(resp: str, n_sent: int) -> dict:
    low = resp.lower()
    nums = parse_numbers(resp)
    n_chars = max(len(resp), 1)
    runs = [len(m.group(0)) for m in DIGIT_RUN_RE.finditer(resp.replace(",", ""))]
    deriv = sum(low.count(m) for m in DERIV_MARKERS)
    reg = {
        "char_len": float(len(resp)),
        "sent_count": float(n_sent),
        "numeric_token_count": float(len(nums)),
        "numeric_density_per_100c": 100.0 * len(nums) / n_chars,
        "derivation_marker_count": float(deriv),
        "derivation_marker_present": float(deriv > 0),
        "currency_symbol_count": float(len(CURRENCY_RE.findall(resp))),
        "max_digit_run": float(max(runs) if runs else 0),
        "hedge_count": float(sum(low.count(h) for h in HEDGES)),
    }
    words = resp.split()
    extra = {
        "x_word_count": float(len(words)),
        "x_newline_count": float(resp.count("\n")),
        "x_percent_sign_count": float(resp.count("%")),
        "x_digit_char_fraction": sum(c.isdigit() for c in resp) / n_chars,
        "x_distinct_numeric_count": float(len({n["stripped"] for n in nums})),
        "x_max_numeric_log10": float(np.log10(max([n["abs_value"] for n in nums] + [1.0]) + 1.0)),
        "x_mean_sentence_chars": len(resp) / max(n_sent, 1),
        "x_abstain_marker_count": float(sum(low.count(a) for a in ABSTAIN)),
        "x_equals_sign_count": float(resp.count("=")),
        "x_type_token_ratio": len({w.lower() for w in words}) / max(len(words), 1),
    }
    return {**reg, **extra}


REGISTERED = ["char_len", "sent_count", "numeric_token_count", "numeric_density_per_100c",
              "derivation_marker_count", "derivation_marker_present",
              "currency_symbol_count", "max_digit_run", "hedge_count"]


# ------------------------------------------------------------------------------ run


def main() -> None:
    df = pl.read_parquet(ITEMS).filter(pl.col("subset") == SUBSET)
    rows = df.to_dicts()
    y = np.array([r["label"] for r in rows], dtype=int)

    # ---- H183 -------------------------------------------------------------
    feat_rows = [response_features(r["response"], r["n_sent"]) for r in rows]
    names = list(feat_rows[0].keys())
    h183 = []
    for name in names:
        s = np.array([f[name] for f in feat_rows], dtype=float)
        rec = scored(name, y, s)
        rec["registered"] = name in REGISTERED
        h183.append(rec)
    h183.sort(key=lambda d: -d["auroc"])

    # ---- H184 -------------------------------------------------------------
    acc = {k: [] for k in ("raw", "raw_boundary", "scale", "scale_tight",
                           "scale_tight_exact", "scale_tight_no_years", "exact")}
    n_year_tokens = 0
    n_tokens = []
    for r in rows:
        resp_nums = parse_numbers(r["response"])
        docs = list(r["documents"])
        n_tokens.append(len(resp_nums))
        if not resp_nums:
            for k in acc:
                acc[k].append(0.0)
            continue

        ev_join = strip_thousands_commas(" ".join(docs))
        ev_loose, ev_tight = [], []
        for d in docs:
            hdr = HEADER_SCALE_RE.search(d)
            doc_scale = SCALE[hdr.group(1).lower()] if hdr else None
            for n in parse_numbers(d):
                ev_loose.append(value_variants(n, doc_scale))
                ev_tight.append(value_variants(n, doc_scale, tight=True))
        arr_loose = np.array(sorted({v for s in ev_loose for v in s})) if ev_loose else np.array([])
        arr_tight = np.array(sorted({v for s in ev_tight for v in s})) if ev_tight else np.array([])
        ev_exact = {n["value"] for d in docs for n in parse_numbers(d)}

        def any_close(arr, cand, rel):
            for c in cand:
                if arr.size and bool(np.any(np.abs(arr - c) <= rel * np.maximum(np.abs(arr), abs(c)))):
                    return True
            return False

        hits = dict.fromkeys(acc, 0)
        n_non_year = 0
        for n in resp_nums:
            tok = n["stripped"]
            is_year = bool(re.fullmatch(r"(19|20)\d\d", tok))
            n_year_tokens += int(is_year)
            if tok in ev_join:
                hits["raw"] += 1
                if locate(ev_join, tok):
                    hits["raw_boundary"] += 1
            if n["value"] in ev_exact:
                hits["exact"] += 1
            if any_close(arr_loose, value_variants(n, None), REL_TOL):
                hits["scale"] += 1
            tight_cand = value_variants(n, None, tight=True)
            tight_hit = any_close(arr_tight, tight_cand, REL_TOL)
            hits["scale_tight"] += int(tight_hit)
            if any_close(arr_tight, tight_cand, 0.0):
                hits["scale_tight_exact"] += 1
            if not is_year:
                n_non_year += 1
                hits["scale_tight_no_years"] += int(tight_hit)
        k = len(resp_nums)
        for name in acc:
            denom = n_non_year if name == "scale_tight_no_years" else k
            acc[name].append(hits[name] / denom if denom else 0.0)

    h184 = {
        "raw_containment": scored("attribution_raw_containment", y, np.array(acc["raw"])),
        "scale_aware": scored("attribution_scale_aware", y, np.array(acc["scale"])),
        "sensitivity": {
            "raw_containment_digit_boundary": scored(
                "attribution_raw_boundary", y, np.array(acc["raw_boundary"])),
            "scale_aware_tight_no_x100": scored(
                "attribution_scale_aware_tight", y, np.array(acc["scale_tight"])),
            "scale_aware_tight_zero_tolerance": scored(
                "attribution_scale_aware_tight_exact", y, np.array(acc["scale_tight_exact"])),
            "exact_value_no_scale_no_tolerance": scored(
                "attribution_exact_value", y, np.array(acc["exact"])),
            "scale_aware_tight_excluding_bare_years": scored(
                "attribution_scale_aware_tight_no_years", y,
                np.array(acc["scale_tight_no_years"])),
        },
        "response_numeric_tokens": {
            "mean": round(float(np.mean(n_tokens)), 4),
            "items_with_no_numeric_token": int(sum(1 for t in n_tokens if t == 0)),
            "bare_year_tokens_1900_2099": n_year_tokens,
            "bare_year_share_of_tokens": round(n_year_tokens / max(sum(n_tokens), 1), 4),
        },
    }

    # ---- H185 -------------------------------------------------------------
    split_row_items = split_hdr_items = split_any_items = 0
    tok_total = tok_split_row = tok_split_hdr = 0
    items_with_located_token = 0
    docs_over_window = 0
    docs_total = 0
    tok_in_multiwindow_doc = 0          # only these CAN split
    items_with_token_in_multiwindow = 0
    table_docs = table_docs_over_window = 0
    max_hdr_gap = 0                     # widest token-to-header-row gap seen, chars
    for r in rows:
        resp_nums = parse_numbers(r["response"])
        toks = sorted({n["stripped"] for n in resp_nums})
        it_row = it_hdr = False
        located_here = False
        it_multiwindow = False
        for doc in r["documents"]:
            docs_total += 1
            multiwindow = len(doc) > PRESENTATION.window_chars
            if multiwindow:
                docs_over_window += 1
            spans = window_spans(len(doc))
            assert [doc[s:e] for s, e in spans] == gr_windows(doc, PRESENTATION)
            nd, cmap = norm_doc_with_map(doc)
            rows_tbl = top_level_rows(doc)
            hdr_span = rows_tbl[0] if rows_tbl else None
            if rows_tbl:
                table_docs += 1
                table_docs_over_window += int(multiwindow)
            for tok in toks:
                for pos_n in locate(nd, tok):
                    p = cmap[pos_n]
                    located_here = True
                    tok_total += 1
                    if multiwindow:
                        tok_in_multiwindow_doc += 1
                        it_multiwindow = True
                    enclosing = next((rw for rw in rows_tbl if rw[0] <= p < rw[1]), None)
                    if enclosing is not None:
                        anchor_row = first_cell_span(doc, enclosing)
                    else:
                        anchor_row = (sentence_start(doc, p), sentence_start(doc, p) + 1)
                    if not binds(spans, p, anchor_row):
                        tok_split_row += 1
                        it_row = True
                    if hdr_span is not None and enclosing is not None and enclosing != hdr_span:
                        max_hdr_gap = max(max_hdr_gap, p - hdr_span[0])
                        if not binds(spans, p, hdr_span):
                            tok_split_hdr += 1
                            it_hdr = True
        items_with_token_in_multiwindow += int(it_multiwindow)
        split_row_items += int(it_row)
        split_hdr_items += int(it_hdr)
        split_any_items += int(it_row or it_hdr)
        items_with_located_token += int(located_here)

    n_items = len(rows)
    h185 = {
        "items": n_items,
        "items_with_a_located_cited_number": items_with_located_token,
        "split_binding_fraction_union": round(split_any_items / n_items, 4),
        "split_binding_fraction_row_label": round(split_row_items / n_items, 4),
        "split_binding_fraction_column_header": round(split_hdr_items / n_items, 4),
        "token_occurrences_checked": tok_total,
        "token_split_row_label": tok_split_row,
        "token_split_column_header": tok_split_hdr,
        "documents_total": docs_total,
        "documents_over_1500_chars": docs_over_window,
        "documents_over_1500_chars_fraction": round(docs_over_window / docs_total, 4),
        "exposure": {
            "note": "a binding can only split inside a document that produces more "
                    "than one window; these count how much of the corpus was actually "
                    "at risk, so a zero can be told apart from an untestable zero",
            "token_occurrences_in_multi_window_docs": tok_in_multiwindow_doc,
            "items_with_a_cited_number_in_a_multi_window_doc": items_with_token_in_multiwindow,
            "table_documents": table_docs,
            "table_documents_over_1500_chars": table_docs_over_window,
            "widest_token_to_header_row_gap_chars": max_hdr_gap,
        },
        "bar": 0.25,
        "verdict_union": "KILL FIRES" if split_any_items / n_items >= 0.25 else "EXCLUDED",
        "verdict_row_label_only": "KILL FIRES" if split_row_items / n_items >= 0.25 else "EXCLUDED",
    }

    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subset": SUBSET,
        "items": n_items,
        "labels": {"positive": int(y.sum()), "negative": int((1 - y).sum())},
        "bootstrap": {"draws": N_BOOT, "seed": SEED, "unit": "item, with replacement"},
        "H183": {
            "question": "which RESPONSE-ONLY feature carries the ablated 0.66",
            "registered_features": [d for d in h183 if d["registered"]],
            "executor_added_features": [d for d in h183 if not d["registered"]],
            "at_or_above_0_60": [d["feature"] for d in h183 if d["auroc"] >= 0.60],
        },
        "H184": {
            "question": "is deterministic number attribution informative on finqa",
            "numeric_token_definition": ">= 2 digits after comma-stripping",
            "raw_normalisation": "commas removed from both the response token and the "
                                 "concatenated evidence; substring containment",
            "scale_aware_normalisation": (
                "both sides parsed to float values, not strings. Response side: comma "
                "stripping, currency marks dropped, parenthesised and leading-minus "
                "negatives resolved to a signed value, a trailing scale word "
                "(thousand/million/billion/trillion within 18 chars) applied as a "
                "multiplier, and a percent form admitted as value/100. Evidence side: "
                "the same, plus a document-level scale header ('in thousands' / "
                "'in millions' / 'in billions') applied as a multiplier to every number "
                "in that document, so a table stated in millions matches a response that "
                "spells the unit out. Both sides also carry the x100 variant so a ratio "
                "written as 0.045 matches a response's 4.5%. A response number counts as "
                "attributed when any of its variants is within 0.2% relative tolerance of "
                "any evidence variant - the tolerance absorbs restatement rounding "
                "($911.51 billion against a 911507 millions table cell). Leading zeros and "
                "trailing decimals resolve through float parsing."
            ),
            "bands": {">=0.66": "evidence channel informative", "0.55-0.66": "partial",
                      "<0.55": "attribution uninformative on finqa"},
            **h184,
        },
        "H185": {
            "question": "does the 1500/750 serving window split a cited number from its label",
            "windowing_implementation": "groundrails.dataset.shape.windows with the default "
                                        "Presentation (window_chars=1500, stride_chars=750); "
                                        "offsets recomputed with identical geometry and "
                                        "asserted byte-equal to the library output on every "
                                        "document read",
            "anchor_definition": "table cell -> its row's first cell (the row label), and "
                                 "separately the whole header row (the column label); prose -> "
                                 "the start of the containing sentence. A binding is SPLIT "
                                 "when no single window holds both the number's position and "
                                 "the whole anchor span",
            **h185,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
