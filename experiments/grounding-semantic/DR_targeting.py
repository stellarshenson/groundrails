"""DR track shared targeting module (registered: docs/experiments/semantic-dataset-enhancements.md).

Fits the empirical distribution of REAL hallucination spans from RAGTruth train
annotations (12,756 spans / 6,721 spanned responses) as a Laplace-smoothed joint
histogram of (relative start position, 10 bins) x (span length in words, bands),
per task_type plus pooled. Provides `sample_spans(seed_text, n)` - the contract
consumed by DR-H112 / DR-H114 / DR-H115 / DR-H116: draw (rel_pos, len_band) from
the fitted histogram, snap to the nearest spaCy factual locus within +/-15 tokens
under the famine-inverting type quota (number/date 35%, entity 25%, negation 15%,
hedge 15%, relational verb 10%), IDF-weighted within type via wordfreq; fall back
to the raw positional span when no locus is in reach.

CLI:
  uv run python experiments/grounding-semantic/DR_targeting.py --fit
  uv run python experiments/grounding-semantic/DR_targeting.py --smoke 500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAGTRUTH_ZIP = ROOT / "data/external/datasets/dataset-ragtruth.zip"
RAGTRUTH_MEMBER = "wandb__RAGTruth-processed__train.parquet"
STATS_PATH = Path(__file__).parent / "DR_targeting_stats.json"
H111_PAIRS = Path(__file__).parent / "R10-H111_stage1_pairs.parquet"

POS_BINS = 10
LEN_BANDS = [(1, 2), (3, 6), (7, 15), (16, 10_000)]
LEN_BAND_NAMES = ["1-2", "3-6", "7-15", "16+"]
LAPLACE = 1.0
SNAP_WINDOW = 15  # tokens each side

TYPE_QUOTA = {
    "number_date": 0.35,
    "entity": 0.25,
    "negation": 0.15,
    "hedge": 0.15,
    "relverb": 0.10,
}

HEDGE_LEXICON = {
    "may", "might", "could", "can", "should", "would", "possibly", "perhaps",
    "probably", "likely", "unlikely", "presumably", "apparently", "seemingly",
    "arguably", "roughly", "approximately", "about", "around", "nearly",
    "almost", "generally", "typically", "usually", "often", "sometimes",
    "somewhat", "relatively", "partially", "largely", "mostly", "suggest",
    "suggests", "indicate", "indicates", "appear", "appears", "seem", "seems",
    "estimated",
}

NUMDATE_ENTS = {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "DATE", "TIME", "ORDINAL"}
ENTITY_ENTS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
               "FAC", "NORP", "LAW", "LANGUAGE"}
NEG_TOKENS = {"not", "never", "no", "none", "cannot", "n't", "without", "neither", "nor"}


# ---------------------------------------------------------------- fitting

def _load_spans() -> pl.DataFrame:
    with zipfile.ZipFile(RAGTRUTH_ZIP) as z, z.open(RAGTRUTH_MEMBER) as f:
        df = pl.read_parquet(f.read())
    rows = []
    for out, labels, task in df.select("output", "hallucination_labels",
                                       "task_type").iter_rows():
        spans = json.loads(labels)
        if not spans or not out:
            continue
        n = len(out)
        for s in spans:
            text = s.get("text") or ""
            wlen = max(1, len(text.split()))
            rows.append({
                "rel_pos": min(0.999999, max(0.0, s["start"] / n)),
                "word_len": wlen,
                "task_type": task,
            })
    return pl.DataFrame(rows)


def _band_index(word_len: int) -> int:
    for i, (lo, hi) in enumerate(LEN_BANDS):
        if lo <= word_len <= hi:
            return i
    return len(LEN_BANDS) - 1


def fit_histograms(spans: pl.DataFrame) -> dict:
    def fit_one(sub: pl.DataFrame) -> dict:
        joint = np.full((POS_BINS, len(LEN_BANDS)), LAPLACE)
        for rel_pos, wlen in sub.select("rel_pos", "word_len").iter_rows():
            joint[min(POS_BINS - 1, int(rel_pos * POS_BINS)), _band_index(wlen)] += 1
        probs = joint / joint.sum()
        # empirical within-band word-length distributions (for length sampling)
        band_lens: dict[str, dict[str, int]] = {}
        for wlen in sub["word_len"].to_list():
            b = LEN_BAND_NAMES[_band_index(wlen)]
            band_lens.setdefault(b, {})
            band_lens[b][str(wlen)] = band_lens[b].get(str(wlen), 0) + 1
        return {"joint": probs.tolist(), "n": sub.height, "band_lens": band_lens}

    out = {"pooled": fit_one(spans)}
    for task in spans["task_type"].unique().to_list():
        out[task] = fit_one(spans.filter(pl.col("task_type") == task))
    return out


def validate_fit(spans: pl.DataFrame, hist: dict, n_draw: int = 10_000,
                 seed: int = 0) -> dict:
    from scipy.stats import chisquare, ks_2samp

    rng = np.random.default_rng(seed)
    probs = np.array(hist["pooled"]["joint"])
    flat = probs.ravel()
    idx = rng.choice(len(flat), size=n_draw, p=flat / flat.sum())
    pos_bin, len_band = np.unravel_index(idx, probs.shape)
    # within-bin uniform jitter for the continuous rel_pos marginal
    samp_pos = (pos_bin + rng.random(n_draw)) / POS_BINS
    # within-band empirical word-length draw
    band_lens = hist["pooled"]["band_lens"]
    samp_len = np.empty(n_draw)
    for i, b in enumerate(len_band):
        d = band_lens[LEN_BAND_NAMES[b]]
        ks = np.array([int(k) for k in d])
        ws = np.array(list(d.values()), dtype=float)
        samp_len[i] = rng.choice(ks, p=ws / ws.sum())

    emp_pos = spans["rel_pos"].to_numpy()
    emp_len = spans["word_len"].to_numpy()
    ks_pos = ks_2samp(samp_pos, emp_pos)
    ks_len = ks_2samp(samp_len, emp_len)

    # chi-square: sampled joint cell counts vs fitted expectation
    obs = np.zeros_like(flat)
    for j in idx:
        obs[j] += 1
    exp = flat / flat.sum() * n_draw
    chi = chisquare(obs, exp)

    return {
        "ks_pos_D": float(ks_pos.statistic), "ks_pos_p": float(ks_pos.pvalue),
        "ks_len_D": float(ks_len.statistic), "ks_len_p": float(ks_len.pvalue),
        "chi2_stat": float(chi.statistic), "chi2_p": float(chi.pvalue),
        "bars": {"ks_D_max": 0.05, "chi2_p_min": 0.01},
        "pass": bool(ks_pos.statistic < 0.05 and ks_len.statistic < 0.05
                     and chi.pvalue > 0.01),
    }


# ---------------------------------------------------------------- sampling

@dataclass
class Locus:
    char_start: int
    char_end: int
    text: str
    ltype: str
    tok_index: int
    idf: float


_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_lg", disable=["lemmatizer"])
    return _NLP


def _idf(word: str) -> float:
    from wordfreq import zipf_frequency
    return max(0.5, 8.0 - zipf_frequency(word, "en"))


def _inventory(doc) -> list[Locus]:
    loci: list[Locus] = []
    ent_tokens = set()
    for ent in doc.ents:
        for t in ent:
            ent_tokens.add(t.i)
        if ent.label_ in NUMDATE_ENTS:
            lt = "number_date"
        elif ent.label_ in ENTITY_ENTS:
            lt = "entity"
        else:
            continue
        loci.append(Locus(ent.start_char, ent.end_char, ent.text, lt,
                          ent.start, _idf(ent.root.text.lower())))
    for t in doc:
        if t.i in ent_tokens:
            continue
        low = t.lower_
        if t.like_num:
            loci.append(Locus(t.idx, t.idx + len(t.text), t.text, "number_date",
                              t.i, _idf(low)))
        elif t.dep_ == "neg" or low in NEG_TOKENS:
            loci.append(Locus(t.idx, t.idx + len(t.text), t.text, "negation",
                              t.i, _idf(low)))
        elif low in HEDGE_LEXICON:
            loci.append(Locus(t.idx, t.idx + len(t.text), t.text, "hedge",
                              t.i, _idf(low)))
        elif t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass"):
            loci.append(Locus(t.idx, t.idx + len(t.text), t.text, "relverb",
                              t.i, _idf(low)))
    return loci


def _load_pooled():
    stats = json.loads(STATS_PATH.read_text())
    return stats["histograms"]["pooled"]


_AVAIL = None


def _availability() -> dict:
    global _AVAIL
    if _AVAIL is None:
        stats = json.loads(STATS_PATH.read_text())
        _AVAIL = stats.get("availability", {})
    return _AVAIL


_POOLED = None


def sample_spans(seed_text: str, n: int, rng: random.Random | None = None
                 ) -> list[tuple[int, int, str, str, str]]:
    """Draw n corruption spans for seed_text.

    Returns (char_start, char_end, span_text, locus_type, source) tuples;
    source is 'locus' (snapped to a typed factual locus) or 'positional'
    (raw histogram fallback).
    """
    global _POOLED
    if _POOLED is None:
        _POOLED = _load_pooled()
    rng = rng or random.Random(0)
    doc = _nlp()(seed_text)
    tokens = list(doc)
    if not tokens:
        return []
    loci = _inventory(doc)
    by_type: dict[str, list[Locus]] = {}
    for lo in loci:
        by_type.setdefault(lo.ltype, []).append(lo)

    probs = np.array(_POOLED["joint"])
    flat = probs.ravel()
    flat = flat / flat.sum()
    band_lens = _POOLED["band_lens"]

    out = []
    for _ in range(n):
        j = rng.choices(range(len(flat)), weights=flat)[0]
        pos_bin, len_band = np.unravel_index(j, probs.shape)
        rel = (pos_bin + rng.random()) / POS_BINS
        target_tok = min(len(tokens) - 1, int(rel * len(tokens)))
        # renormalized quota over types with available loci in the window
        window_types = {}
        for lt, ls in by_type.items():
            near = [lo for lo in ls if abs(lo.tok_index - target_tok) <= SNAP_WINDOW]
            if near:
                window_types[lt] = near
        if window_types:
            types = list(window_types)
            # availability correction: quota / measured window-availability, so
            # scarce types (negation, hedge) are taken whenever present and the
            # achieved GLOBAL mixture tracks the famine-inverting quota
            weights = [TYPE_QUOTA[t] / max(1e-3, _availability().get(t, 1.0))
                       for t in types]
            lt = rng.choices(types, weights=weights)[0]
            cands = window_types[lt]
            lo = rng.choices(cands, weights=[c.idf for c in cands])[0]
            out.append((lo.char_start, lo.char_end, lo.text, lo.ltype, "locus"))
        else:
            d = band_lens[LEN_BAND_NAMES[len_band]]
            wl = rng.choices([int(k) for k in d], weights=list(d.values()))[0]
            span_toks = tokens[target_tok:min(len(tokens), target_tok + wl)]
            if not span_toks:
                span_toks = [tokens[-1]]
            cs = span_toks[0].idx
            ce = span_toks[-1].idx + len(span_toks[-1].text)
            out.append((cs, ce, seed_text[cs:ce], "positional", "positional"))
    return out


# ---------------------------------------------------------------- CLI

def cmd_fit():
    spans = _load_spans()
    n_resp = None  # responses with spans counted below for the report
    with zipfile.ZipFile(RAGTRUTH_ZIP) as z, z.open(RAGTRUTH_MEMBER) as f:
        df = pl.read_parquet(f.read())
    n_resp = df.filter(pl.col("hallucination_labels") != "[]").height
    print(f"RAGTruth train: {spans.height} spans over {n_resp} spanned responses "
          f"(mean {spans.height / n_resp:.2f})")
    hist = fit_histograms(spans)
    val = validate_fit(spans, hist)
    print("validation:", json.dumps(val, indent=1))
    stats = {
        "source": str(RAGTRUTH_ZIP.name) + "::" + RAGTRUTH_MEMBER,
        "n_spans": spans.height,
        "n_spanned_responses": n_resp,
        "pos_bins": POS_BINS,
        "len_bands": LEN_BAND_NAMES,
        "laplace": LAPLACE,
        "type_quota": TYPE_QUOTA,
        "snap_window_tokens": SNAP_WINDOW,
        "hedge_lexicon_size": len(HEDGE_LEXICON),
        "validation": val,
        "histograms": hist,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(f"wrote {STATS_PATH}")


def cmd_smoke(n_seeds: int):
    rng = random.Random(7)
    seeds = (pl.read_parquet(H111_PAIRS).select("seed").unique()
             .sample(n_seeds, seed=7)["seed"].to_list())
    total, snapped = 0, 0
    mix: dict[str, int] = {}
    for s in seeds:
        for cs, ce, text, lt, src in sample_spans(s, 3, rng):
            total += 1
            if src == "locus":
                snapped += 1
            mix[lt] = mix.get(lt, 0) + 1
    print(f"smoke: {n_seeds} seeds, {total} draws, snap rate {snapped / total:.3f}")
    mixture = {k: round(v / total, 3) for k, v in sorted(mix.items())}
    print("type mixture:", mixture, "quota:", TYPE_QUOTA)
    stats = json.loads(STATS_PATH.read_text())
    stats["smoke"] = {"n_seeds": n_seeds, "draws": total,
                      "snap_rate": snapped / total, "type_mixture": mixture}
    STATS_PATH.write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    if args.fit:
        cmd_fit()
    if args.smoke:
        cmd_smoke(args.smoke)
