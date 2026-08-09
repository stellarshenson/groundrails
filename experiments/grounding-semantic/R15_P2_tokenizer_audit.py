"""R15 PROBE 2 - number representation audit on the shipped mmBERT tokenizer.

CPU only, no model weights, no GPU. Polars throughout.

Measures, on `models/R9-H105-mmbert-dann-clean/tokenizer.json` (the tokenizer the
shipped cross-encoder was trained and is served with):

  A. FORM AUDIT      - how canonical number surfaces fragment
  B. DIGIT LADDER    - token count vs digit count, per surface family
  C. CENSUS          - token-count distribution of numerals actually present in
                       (i) the held-out TabFact tables the H133 census/probe used,
                       (ii) the H133 constructed triples (a/b/c values),
                       (iii) the admitted H108 lane claims,
                       (iv) RAGBench-finqa `documents` + response text  [ANALYSIS ONLY]
  D. ADJACENCY       - do numerically adjacent values share token prefixes
  E. SCALED FORMS    - "10.5 million" vs "10,500,000" vs "10500000"
  F. LANE SURFACE    - does the H133/A4 derived-value format tokenize like the
                       table cells it is derived from

Run:  cd <repo> && uv run python experiments/grounding-semantic/R15_P2_tokenizer_audit.py
Out:  R15_P2_tokenizer_audit.json
"""

import io
import json
import pathlib
import re
import zipfile
from collections import Counter

import numpy as np
import polars as pl
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
TOK_PATH = ROOT / "models" / "R9-H105-mmbert-dann-clean" / "tokenizer.json"
OUT = HERE / "R15_P2_tokenizer_audit.json"

tok = Tokenizer.from_file(str(TOK_PATH))
R = {"tokenizer": str(TOK_PATH), "vocab_size": tok.get_vocab_size()}


def pieces(s):
    return tok.encode(s, add_special_tokens=False).tokens


def n_tok(s):
    return len(tok.encode(s, add_special_tokens=False).ids)


def ids(s):
    return tok.encode(s, add_special_tokens=False).ids


# ----------------------------------------------------------------- A. FORM AUDIT
FORMS = [
    "7", "42", "999", "1000", "10547", "10548", "123456", "1234567", "10500000",
    "1,000", "10,547", "10,548", "123,456", "1,234,567", "10,500,000",
    "3.5", "0.25", "10.5", "1.75", "12.50", "3.1416", "0.0001",
    "10,547.32", "1,234,567.89",
    "$5", "$1,000", "$ 1,000", "$10,547", "$ 383,221", "$383,221",
    "5%", "5 %", "12.5%", "(1,234)", "-1,234", "- 1,234",
    "10.5 million", "10.5 billion", "1.05 thousand", "10500 thousand",
    "2019", "2020", "1997", "Q3 2019", "FY2019",
    "1.0", "1.00", "01", "007",
]
R["form_audit"] = [
    {"surface": f, "n_tokens": n_tok(f), "tokens": pieces(f)} for f in FORMS
]

# with a leading space (mid-sentence position) - the metaspace prefix matters
R["form_audit_leading_space"] = [
    {"surface": " " + f, "n_tokens": n_tok(" " + f), "tokens": pieces(" " + f)}
    for f in ["10547", "10,547", "$1,000", "10.5", "2019"]
]

# ----------------------------------------------------------- B. DIGIT LADDER
ladder = []
rng = np.random.default_rng(20260809)
for d in range(1, 13):
    lo, hi = 10 ** (d - 1), 10**d - 1
    vals = rng.integers(lo, hi + 1, size=200) if d > 1 else np.arange(1, 10)
    bare = [n_tok(str(int(v))) for v in vals]
    comma = [n_tok(f"{int(v):,}") for v in vals]
    ladder.append({
        "digits": d,
        "bare_mean_tokens": round(float(np.mean(bare)), 4),
        "bare_max_tokens": int(np.max(bare)),
        "bare_unique_counts": sorted(set(int(x) for x in bare)),
        "comma_mean_tokens": round(float(np.mean(comma)), 4),
        "comma_unique_counts": sorted(set(int(x) for x in comma)),
    })
R["digit_ladder"] = ladder

# how many of the 10 single digits / 100 two-digit / 1000 three-digit strings are 1 token
R["single_token_coverage"] = {
    "digits_0_9": sum(n_tok(str(i)) == 1 for i in range(10)),
    "ints_0_99": sum(n_tok(str(i)) == 1 for i in range(100)),
    "ints_0_999": sum(n_tok(str(i)) == 1 for i in range(1000)),
    "ints_1000_1999": sum(n_tok(str(i)) == 1 for i in range(1000, 2000)),
}

# ------------------------------------------------------------------- C. CENSUS
# strict: a thousands separator only counts when followed by exactly three digits,
# so "December 31, 2019" does not read as a separator-bearing numeral
NUMRE = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\$\s?\d+(?:\.\d+)?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?%?"
)


def census(strings, name, cap_chars=4000, limit=None):
    c = Counter()
    forms = Counter()
    if limit:
        strings = strings[:limit]
    for s in strings:
        if s is None:
            continue
        for m in NUMRE.findall(s[:cap_chars]):
            c[m] += 1
    tot = sum(c.values())
    uniq = list(c.keys())
    tc = {}
    weighted = []
    for u in uniq:
        t = n_tok(u)
        tc[u] = t
        weighted.extend([t] * c[u])
        core = u.lstrip("$ ").rstrip("%")
        if "," in core:
            forms["thousands_sep"] += c[u]
        elif "." in core:
            forms["decimal"] += c[u]
        else:
            forms["bare_int"] += c[u]
        if u.startswith("$"):
            forms["currency_prefixed"] += c[u]
        if u.endswith("%"):
            forms["percent_suffixed"] += c[u]
    w = np.array(weighted) if weighted else np.array([0])
    uq = np.array([tc[u] for u in uniq]) if uniq else np.array([0])
    return {
        "name": name,
        "n_strings": len(strings),
        "n_numeral_occurrences": tot,
        "n_unique_numerals": len(uniq),
        "occurrence_weighted_tokens": {
            "mean": round(float(w.mean()), 4),
            "median": float(np.median(w)),
            "p90": float(np.percentile(w, 90)),
            "p99": float(np.percentile(w, 99)),
            "max": int(w.max()),
            "share_1_token": round(float((w == 1).mean()), 4),
            "share_ge_3_tokens": round(float((w >= 3).mean()), 4),
            "share_ge_5_tokens": round(float((w >= 5).mean()), 4),
        },
        "type_weighted_mean_tokens": round(float(uq.mean()), 4),
        "surface_form_shares": {
            k: round(v / tot, 4) for k, v in sorted(forms.items(), key=lambda x: -x[1])
        } if tot else {},
        "top_20_numerals": [
            {"surface": u, "count": c[u], "n_tokens": tc[u], "tokens": pieces(u)}
            for u, _ in c.most_common(20)
        ],
    }


cens = []

# (i) held-out TabFact tables - the exact split R14_H133_probe.build() draws from
z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
train_ids = set(
    pl.read_parquet(
        io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith("__train.parquet"))))
    )["table_id"].to_list()
)
held = pl.concat([
    pl.read_parquet(io.BytesIO(z.read(n)))
    for n in z.namelist()
    if n.endswith("__test.parquet") or n.endswith("__validation.parquet")
]).unique(subset=["table_id"], keep="first")
held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
cens.append(census(held["table_text"].to_list()[:4000], "tabfact_heldout_tables"))

# also the TRAIN tables - the ones the A4 lane would actually be built over
tr = pl.read_parquet(
    io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith("__train.parquet"))))
).unique(subset=["table_id"], keep="first")
cens.append(census(tr["table_text"].to_list()[:4000], "tabfact_train_tables"))

# (ii) H133 constructed triples - the asserted values themselves
tri = pl.read_parquet(HERE / "R14_H133_triples.parquet")
R["h133_triples_cols"] = tri.columns
vals_correct = tri["v_correct"].to_list()
vals_wrong = tri["v_wrong"].to_list()
cens.append(census(vals_correct, "h133_v_correct_derived"))
cens.append(census(vals_wrong, "h133_v_wrong_operand"))

# (iii) admitted H108 lane claims
h108 = pl.read_parquet(HERE / "R10-H108_pairs.parquet")
R["h108_cols"] = h108.columns
claim_col = "claim" if "claim" in h108.columns else h108.columns[0]
cens.append(census(h108[claim_col].to_list()[:40000], "h108_lane_claims"))

# (iv) RAGBench finqa  [ANALYSIS ONLY - never a bar input]
zr = zipfile.ZipFile(DATA / "dataset-ragbench.zip")
fq = pl.read_parquet(
    io.BytesIO(zr.read("galileo-ai__ragbench__finqa__test.parquet"))
)
R["finqa_cols"] = fq.columns
docs = []
for d in fq["documents"].to_list()[:400]:
    if d is None:
        continue
    docs.append(" ".join(list(d)) if not isinstance(d, str) else d)
cens.append(census(docs, "finqa_documents_ANALYSIS_ONLY"))
resp_col = "response" if "response" in fq.columns else None
if resp_col:
    cens.append(census(fq[resp_col].to_list()[:2000], "finqa_response_ANALYSIS_ONLY"))
R["census"] = cens

# --------------------------------------------------------------- D. ADJACENCY
def prefix_share(a, b):
    ia, ib = ids(a), ids(b)
    n = 0
    for x, y in zip(ia, ib):
        if x != y:
            break
        n += 1
    return {
        "a": a, "b": b, "ta": len(ia), "tb": len(ib),
        "shared_prefix": n,
        "shared_prefix_frac": round(n / max(len(ia), len(ib)), 4),
        "same_len": len(ia) == len(ib),
        "tokens_a": pieces(a), "tokens_b": pieces(b),
    }


adj_cases = [
    ("10547", "10548"), ("10,547", "10,548"),
    ("999", "1000"), ("1,000", "1,001"),
    ("12.50", "12.51"), ("99.9", "100.0"),
    ("383221", "383222"), ("$ 383,221", "$ 383,222"),
    ("2019", "2020"),
]
R["adjacency_cases"] = [prefix_share(a, b) for a, b in adj_cases]

# population-scale adjacency: 2,000 random 4-8 digit values, v vs v+1
rng2 = np.random.default_rng(4242)
rows = []
for _ in range(2000):
    d = int(rng2.integers(4, 9))
    v = int(rng2.integers(10 ** (d - 1), 10**d - 1))
    for style, f in (("bare", str), ("comma", lambda x: f"{x:,}")):
        ps = prefix_share(f(v), f(v + 1))
        rows.append({
            "style": style, "digits": d,
            "shared_prefix_frac": ps["shared_prefix_frac"],
            "same_len": ps["same_len"],
            "ta": ps["ta"],
        })
adf = pl.DataFrame(rows)
R["adjacency_population"] = (
    adf.group_by("style")
    .agg([
        pl.len().alias("n"),
        pl.col("shared_prefix_frac").mean().round(4).alias("mean_shared_prefix_frac"),
        (pl.col("shared_prefix_frac") == 0).mean().round(4).alias("share_zero_shared_prefix"),
        pl.col("same_len").mean().round(4).alias("share_same_token_length"),
        pl.col("ta").mean().round(4).alias("mean_tokens"),
    ])
    .to_dicts()
)
R["adjacency_by_digits"] = (
    adf.filter(pl.col("style") == "bare")
    .group_by("digits")
    .agg([
        pl.col("shared_prefix_frac").mean().round(4).alias("mean_shared_prefix_frac"),
        pl.col("same_len").mean().round(4).alias("share_same_token_length"),
        pl.col("ta").mean().round(4).alias("mean_tokens"),
    ])
    .sort("digits")
    .to_dicts()
)

# magnitude-boundary: does one extra digit preserve any prefix?
R["magnitude_boundary"] = [
    prefix_share(str(v), str(v * 10)) for v in [9, 99, 547, 1054, 10547]
]

# ------------------------------------------------------------ E. SCALED FORMS
def seq_overlap(a, b):
    ia, ib = ids(a), ids(b)
    sa, sb = set(ia), set(ib)
    return {
        "a": a, "b": b, "ta": len(ia), "tb": len(ib),
        "jaccard_token_ids": round(len(sa & sb) / len(sa | sb), 4),
        "shared_prefix": prefix_share(a, b)["shared_prefix"],
        "tokens_a": pieces(a), "tokens_b": pieces(b),
    }


R["scaled_forms"] = [
    seq_overlap("10.5 million", "10,500,000"),
    seq_overlap("10.5 million", "10500000"),
    seq_overlap("10,500,000", "10500000"),
    seq_overlap("$10.5 million", "$10,500,000"),
    seq_overlap("1.05 billion", "1,050,000,000"),
    seq_overlap("in millions 10.5", "10,500,000"),
    seq_overlap("10.5", "10,500,000"),
    seq_overlap("12.5%", "0.125"),
    seq_overlap("(1,234)", "-1,234"),
]

# ------------------------------------------------------------- F. LANE SURFACE
# The A4/H133 constructor formats derived values with `fmt()`:
#   int-valued -> str(int(round(v)))   e.g. "1234"   (NO thousands separator)
#   else       -> f"{v:.2f}"           e.g. "12.50"
# The table cells it derives from carry separators whenever TabFact has them.
sep_cells = 0
tot_cells = 0
for t in held["table_text"].to_list()[:3000]:
    for m in NUMRE.findall(t[:4000]):
        tot_cells += 1
        if "," in m:
            sep_cells += 1
R["lane_surface"] = {
    "heldout_table_numeral_occurrences": tot_cells,
    "share_with_thousands_separator": round(sep_cells / max(tot_cells, 1), 4),
    "constructor_format": "str(int(round(v))) or f'{v:.2f}' - never emits a separator",
    "examples": [
        {
            "operands_as_written": ["1,234", "5,678"],
            "sum_as_lane_writes_it": "6912",
            "sum_in_table_style": "6,912",
            "tokens_lane": pieces("6912"),
            "tokens_table_style": pieces("6,912"),
            "shared_prefix": prefix_share("6912", "6,912")["shared_prefix"],
        },
        {
            "operands_as_written": ["10.5", "3.25"],
            "sum_as_lane_writes_it": "13.75",
            "tokens_lane": pieces("13.75"),
        },
    ],
}
# how many of the 2,000 banked H133 triples have BOTH operands separator-bearing
# while the asserted value is written bare
tri_ex = tri.head(3) if len(tri) else tri
R["lane_surface"]["h133_v_correct_sep_share"] = round(
    float(np.mean([("," in v) for v in vals_correct])), 4
)
R["lane_surface"]["h133_v_correct_decimal_share"] = round(
    float(np.mean([("." in v) for v in vals_correct])), 4
)
R["lane_surface"]["h133_v_correct_token_mean"] = round(
    float(np.mean([n_tok(v) for v in vals_correct])), 4
)

# ---------------------------------------------- F2. LANE LABEL-LENGTH CONFOUND
# If the correct-derived value and the wrong-operand value differ systematically
# in token length, an A4-style lane teaches "count the digits", not the arithmetic.
tc_ok = np.array([n_tok(v) for v in vals_correct])
tc_bad = np.array([n_tok(v) for v in vals_wrong])
d = tc_ok - tc_bad
R["lane_length_confound"] = {
    "n": int(len(d)),
    "mean_tokens_correct": round(float(tc_ok.mean()), 4),
    "mean_tokens_wrong": round(float(tc_bad.mean()), 4),
    "mean_signed_diff": round(float(d.mean()), 4),
    "share_equal_token_length": round(float((d == 0).mean()), 4),
    "share_correct_longer": round(float((d > 0).mean()), 4),
    "share_correct_shorter": round(float((d < 0).mean()), 4),
    "auroc_token_length_alone": round(
        float(
            (
                sum(
                    (1.0 if a > b else 0.5 if a == b else 0.0)
                    for a in tc_ok
                    for b in tc_bad[
                        np.random.default_rng(7).permutation(len(tc_bad))[:400]
                    ]
                )
                / (len(tc_ok) * 400)
            )
        ),
        4,
    ),
    "note": "AUROC computed on a 2000 x 400 subsample of the cross product",
}

# --------------------------------- F3. CLAIM-SIDE vs EVIDENCE-SIDE SURFACE SPLIT
# finqa only, ANALYSIS ONLY: which surface family the lane should imitate
R["finqa_surface_asymmetry_ANALYSIS_ONLY"] = {
    c["name"]: c["surface_form_shares"]
    for c in cens
    if c["name"].startswith("finqa")
}

# ------------------------------------------------- G. VOCABULARY DIGIT ATOMICITY
vocab = tok.get_vocab()
pure_digit = [t for t in vocab if re.fullmatch(r"▁?[0-9]+", t)]
merges = tj_merges = json.loads(TOK_PATH.read_text())["model"]["merges"]
merges_str = [m if isinstance(m, str) else " ".join(m) for m in merges]
R["vocab_digit_atomicity"] = {
    "pure_digit_tokens_in_vocab": len(pure_digit),
    "multi_digit_tokens_in_vocab": len([t for t in pure_digit if len(t.lstrip("▁")) > 1]),
    "total_merges": len(merges_str),
    "digit_digit_merges": len([x for x in merges_str if re.fullmatch(r"[0-9▁]+ [0-9]+", x)]),
    "verdict": "digit-atomic by vocabulary construction - no numeral above one digit exists",
}

# ------------------------------- H. SEPARATOR ALIGNMENT SHIFT (bare vs comma)
rows = []
rng3 = np.random.default_rng(31337)
for _ in range(1500):
    d = int(rng3.integers(4, 10))
    v = int(rng3.integers(10 ** (d - 1), 10**d - 1))
    a, b = str(v), f"{v:,}"
    ps = prefix_share(a, b)
    rows.append({"digits": d, "shared_prefix": ps["shared_prefix"], "ta": ps["ta"], "tb": ps["tb"]})
sdf = pl.DataFrame(rows)
R["separator_alignment"] = (
    sdf.group_by("digits")
    .agg([
        pl.col("shared_prefix").mean().round(3).alias("mean_shared_prefix"),
        pl.col("ta").mean().alias("mean_tokens_bare"),
        pl.col("tb").mean().alias("mean_tokens_comma"),
    ])
    .sort("digits")
    .to_dicts()
)

# -------------------------- I. PLACE-VALUE MISALIGNMENT INSIDE TABLE COLUMNS
# Digit-atomic tokens carry no place value; place value is position-from-the-END,
# while the encoder indexes position-from-the-START. Two column values of
# different digit length therefore put the same place at different indices.
INT = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
pair_same, pair_diff, cols_seen = 0, 0, 0
for t in held["table_text"].to_list()[:3000]:
    rws = [r.split("#") for r in t.replace("\r\n", "\n").strip().split("\n") if r.strip()]
    if len(rws) < 4:
        continue
    w = len(rws[0])
    body = [r for r in rws[1:] if len(r) == w]
    for ci in range(1, w):
        vals = [c.strip() for c in (r[ci] for r in body)]
        vals = [x for x in vals if INT.match(x)]
        if len(vals) < 4:
            continue
        cols_seen += 1
        lens = [len(x.replace(",", "").replace(".", "").lstrip("-")) for x in vals]
        for i in range(len(lens)):
            for j in range(i + 1, len(lens)):
                if lens[i] == lens[j]:
                    pair_same += 1
                else:
                    pair_diff += 1
R["place_value_misalignment"] = {
    "numeric_columns_scanned": cols_seen,
    "within_column_value_pairs": pair_same + pair_diff,
    "share_pairs_same_digit_count": round(pair_same / max(pair_same + pair_diff, 1), 4),
    "share_pairs_different_digit_count": round(pair_diff / max(pair_same + pair_diff, 1), 4),
    "note": "different digit count = no index-wise place alignment between the two values",
}

# --------------------- J. TOKEN COST OF A TABLE WINDOW (A4 x A2 interaction)
evs = [
    (c + "\n" + t).replace("#", " | ")[:1500]
    for c, t in zip(
        held["table_caption"].to_list()[:1500], held["table_text"].to_list()[:1500]
    )
]
nt = np.array([n_tok(e) for e in evs])
nc = np.array([len(e) for e in evs])
digit_share = []
for e in evs[:500]:
    tk = pieces(e)
    digit_share.append(sum(1 for x in tk if x.isdigit()) / len(tk))
R["table_window_token_cost"] = {
    "n_windows": int(len(evs)),
    "window_chars": 1500,
    "mean_tokens": round(float(nt.mean()), 2),
    "chars_per_token": round(float((nc / nt).mean()), 4),
    "p95_tokens": float(np.percentile(nt, 95)),
    "share_over_512_tokens": round(float((nt > 512).mean()), 4),
    "share_of_tokens_that_are_bare_digits": round(float(np.mean(digit_share)), 4),
}

# ------------------------------------- K. EVIDENCE / CLAIM SEPARATOR REGISTER
# Measured on raw text, both sides, in-domain and (separately labelled) arena.
SEP = re.compile(r"\d,\d{3}")
reg = {}
# in-domain, admissible: TabFact tables (evidence) vs H108 lane claims
tab_txt = " ".join(held["table_text"].to_list()[:3000])
h108_txt = " ".join(str(x) for x in h108[claim_col].to_list()[:40000])
h108_ev = " ".join(str(x) for x in h108["chunk"].to_list()[:40000]) if "chunk" in h108.columns else ""
reg["in_domain"] = {
    "tabfact_tables_chars": len(tab_txt),
    "tabfact_tables_sep_hits": len(SEP.findall(tab_txt)),
    "h108_lane_claims_chars": len(h108_txt),
    "h108_lane_claims_sep_hits": len(SEP.findall(h108_txt)),
    "h108_lane_evidence_chars": len(h108_ev),
    "h108_lane_evidence_sep_hits": len(SEP.findall(h108_ev)),
}
# arena, ANALYSIS ONLY - may motivate a construction, may never calibrate a bar
arena = {}
for sp in ["train", "validation", "test"]:
    f = pl.read_parquet(
        io.BytesIO(zr.read(f"galileo-ai__ragbench__finqa__{sp}.parquet"))
    )
    dt = " ".join(" ".join(list(x)) for x in f["documents"].to_list() if x is not None)
    rt = " ".join(x for x in f["response"].to_list() if x)
    arena[sp] = {
        "n_rows": len(f),
        "doc_chars": len(dt), "doc_sep_hits": len(SEP.findall(dt)),
        "doc_dollar": dt.count("$"),
        "resp_chars": len(rt), "resp_sep_hits": len(SEP.findall(rt)),
        "resp_dollar": rt.count("$"),
    }
reg["arena_finqa_ANALYSIS_ONLY"] = arena
R["separator_register_split"] = reg

# pretokenizer / normalizer declaration straight out of tokenizer.json
tj = json.loads(TOK_PATH.read_text())
R["tokenizer_spec"] = {
    "model_type": tj["model"]["type"],
    "normalizer": tj.get("normalizer"),
    "pre_tokenizer": tj.get("pre_tokenizer"),
    "decoder_type": (tj.get("decoder") or {}).get("type"),
}

OUT.write_text(json.dumps(R, indent=2, default=str))
print(f"wrote {OUT}")
print(json.dumps({k: R[k] for k in ["vocab_size", "single_token_coverage", "tokenizer_spec"]}, indent=2, default=str)[:2000])
