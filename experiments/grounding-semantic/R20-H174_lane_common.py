"""R20-H174 PORTFOLIO ARM - shared lane machinery, CPU only, no GPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H174
HAGRID/EMANUAL PORTFOLIO ARM" (stage 0).  Three lanes are built on top of this
module - `R20-H174_lane_L1.py` (vacuous_claim_reject), `R20-H174_lane_L2.py`
(source_select / attr_pool) and `R20-H174_lane_L4.py` (bind_path_segment).

What lives here is only what more than one lane needs:

  * the SERVING presentation constants - the mix loader (`R18-H150_arm_run.py`
    `make_build_mix`) reads every lane row's `chunk` UNTRUNCATED and then windows
    it 1,500 / 750, so a lane's char budget IS its window budget.  `windows()`
    is transcribed from `R16-H142_G1_arm.windows` and cross-checked against it
    at import time rather than imported, because that module pins CUDA state
  * the two public corpora the registration licenses for these lanes -
    MiniCheck (MIT) and VitaminC (CC-BY-SA-3.0), both GREEN on the R14-H136
    8-gram wall (R19_minicheck_gate.json, R10 precedent for VitaminC), read
    from the archives the fetcher landed
  * the verification instruments the banked lane builders use - the converged
    liblinear claim-only probe at tol 1e-7 on document-disjoint folds (H144
    finding ii, H145 finding b), the AUROC helper, content-token containment

Nothing here writes an artifact.  Run the per-lane builders.
"""

import io
from pathlib import Path
import re
import zipfile

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

# The serve-time window protocol (R8-H101), reproduced from R16-H142_G1_arm.
WIN, STRIDE = 1500, 750

# Per-passage char cap inside a pooled chunk.  Below WIN so no single passage
# can span more than two windows on its own.
PASSAGE_MAX_CHARS = 1400

SOURCES = {
    "minicheck": {
        "dataset": "lytang/C2D-and-D2C-MiniCheck",
        "archive": "data/external/datasets/dataset-minicheck.zip",
        "licence": "MIT",
        "wall": "R19_minicheck_gate.json - GREEN on the R14-H136 8-gram / "
                "Jaccard 0.3 wall against all ten walled arena corpora",
    },
    "vitaminc": {
        "dataset": "tals/vitaminc",
        "archive": "data/external/datasets/dataset-vitaminc.zip",
        "licence": "CC-BY-SA-3.0",
        "wall": "banked clean-mix corpus (DANN group `vitaminc`), gated on "
                "admission; the lane census re-runs the wall on the new text",
    },
    "generator": {
        "dataset": "rule generator, no source corpus",
        "archive": None,
        "licence": "n/a - generated text, this repository",
        "wall": "CLEAR by construction; the census runs anyway",
    },
}

_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# serving presentation
# --------------------------------------------------------------------------- #
def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end.
    Byte-identical to R16-H142_G1_arm.windows / R8-H101."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def window_census(chunks):
    """The block `R18-H150_arm_run.census_crosscheck` compares against."""
    sizes = np.array([len(windows(c)) for c in chunks], dtype=np.int32)
    return {
        "windowing": "1500/750, final window flush to the end (byte-identical "
                     "to R8-H101 / R16-H142)",
        "rows": int(sizes.size),
        "mean_windows": round(float(sizes.mean()), 4),
        "median_windows": int(np.median(sizes)),
        "max_windows": int(sizes.max()),
        "multi_window_rows": int((sizes > 1).sum()),
        "multi_window_share": round(float((sizes > 1).mean()), 4),
    }


def char_stats(texts):
    a = np.array([len(t) for t in texts], dtype=np.int64)
    return {
        "n": int(a.size), "mean": round(float(a.mean()), 1),
        "median": int(np.median(a)), "p10": int(np.percentile(a, 10)),
        "p90": int(np.percentile(a, 90)), "min": int(a.min()), "max": int(a.max()),
    }


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #
def _zip_parquets(name):
    z = zipfile.ZipFile(DATA / f"dataset-{name}.zip")
    out = {}
    for m in z.namelist():
        if m.endswith(".parquet"):
            out[m[: -len(".parquet")].split("__")[-1]] = pl.read_parquet(io.BytesIO(z.read(m)))
    return out


def minicheck():
    """(claim, doc, label, split, doc_id) over both MiniCheck synthesis routes.

    `doc_id` is the position of the document in the sorted deduplicated document
    list, so it is stable across runs and usable as the disjointness key."""
    parts = _zip_parquets("minicheck")
    df = pl.concat([d.with_columns(pl.lit(k).alias("split")) for k, d in sorted(parts.items())])
    docs = sorted(set(df["doc"].to_list()))
    ids = {d: f"mc{i:06d}" for i, d in enumerate(docs)}
    return df.with_columns(
        pl.col("doc").replace_strict(ids).alias("doc_id"),
        pl.col("label").cast(pl.Int64),
    )


def vitaminc(split="train"):
    """(claim, evidence, label, page, unique_id, doc_id) - label 1 = SUPPORTS.

    Only the TRAIN split is read.  The validation and test splits are left
    untouched: R19-H166 amendment A1 registers held-out VitaminC as the
    contradiction-head instrument, and a lane must not consume it."""
    df = _zip_parquets("vitaminc")[split]
    df = df.with_columns(
        (pl.col("label").str.to_uppercase() == "SUPPORTS").cast(pl.Int64).alias("y")
    )
    ev = sorted(set(df["evidence"].to_list()))
    ids = {e: f"vc{i:06d}" for i, e in enumerate(ev)}
    return df.select(
        pl.col("claim"), pl.col("evidence"), pl.col("y").alias("label"),
        pl.col("page"), pl.col("unique_id"), pl.col("revision_type"),
        pl.col("evidence").replace_strict(ids).alias("doc_id"),
    )


def vitaminc_passages(df, rng, max_sentences=7, cap=PASSAGE_MAX_CHARS):
    """page -> a passage of that page's distinct evidence sentences.

    A VitaminC evidence unit is ONE Wikipedia sentence (median 139 chars).  A
    pool of eight of them is a single 1,500-char window, which would erase the
    multi-passage geometry the lane exists to teach, so the page's sentences are
    concatenated into a passage of MiniCheck-like size before pooling."""
    by_page = {}
    for page, ev in df.select(["page", "evidence"]).unique().iter_rows():
        by_page.setdefault(page, []).append(ev)
    out = {}
    for page, evs in by_page.items():
        evs = sorted(set(evs))
        rng.shuffle(evs)
        out[page] = _join_cap(evs[:max_sentences], " ", cap)
    return out


def vitaminc_passage_for(page_sentences, true_ev, rng, max_sentences=7,
                         cap=PASSAGE_MAX_CHARS):
    """The same construction, but guaranteed to contain `true_ev` first."""
    others = [e for e in page_sentences if e != true_ev]
    rng.shuffle(others)
    return _join_cap([true_ev] + others[: max_sentences - 1], " ", cap)


def _join_cap(parts, sep, cap):
    out = ""
    for p in parts:
        nxt = p if not out else out + sep + p
        if len(nxt) > cap:
            break
        out = nxt
    return out or parts[0][:cap]


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def tokens(text):
    return _WORD.findall(text.lower())


def containment(claim, text):
    """Fraction of the claim's content tokens present in `text` - the campaign's
    lexical-baseline feature (R19-H162 `lexical_ceiling`)."""
    ct = set(tokens(claim))
    if not ct:
        return 0.0
    return len(ct & set(tokens(text))) / len(ct)


def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not pos.size or not neg.size:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks over ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# --------------------------------------------------------------------------- #
# verification instruments (banked-lane discipline)
# --------------------------------------------------------------------------- #
def claim_only_probe(claims, labels, groups, rng, n_folds=5):
    """Out-of-fold char-ngram TF-IDF + liblinear probe on the CLAIM ALONE.

    Folds are disjoint on `groups` (the document / item key) so no fold's
    training complement carries its own test rows.  liblinear at tol 1e-7, never
    default lbfgs (R17-H144 finding ii)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    keys = sorted(set(groups))
    rng.shuffle(keys)
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    score = np.zeros(len(claims))
    idx = np.arange(len(claims))
    for f in range(n_folds):
        tr, te = idx[folds != f], idx[folds == f]
        if not te.size or not tr.size:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        xtr = vec.fit_transform([claims[j] for j in tr])
        xte = vec.transform([claims[j] for j in te])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(xtr, [labels[j] for j in tr])
        score[te] = clf.decision_function(xte)
    return float(auroc(labels, score)), score


def within_pair_accuracy(df, score, by=None):
    """Per-family within-pair rank accuracy of a claim-side score. Chance 0.5."""
    d = df.select(["pair_id", "label"] + ([by] if by else [])).with_columns(
        pl.Series("s", score))
    out = {}
    groups = [(("all",), d)] if by is None else d.group_by(by)
    for key, sub in groups:
        piv = sub.pivot(on="label", index="pair_id", values="s",
                        aggregate_function="first").drop_nulls()
        if not len(piv):
            continue
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        out[key[0] if isinstance(key, tuple) else key] = {
            "acc": round(acc, 4), "pairs": int(len(piv))}
    return out


def surface_parity(df, extra=None, report_only=()):
    """AUROC of label against claim-side surface channels. Each barred channel
    should sit in [0.45, 0.55] - the banked lanes' `surface_parity` block.

    `report_only` names channels that are the lane's own semantics rather than a
    confound (L1's claim/chunk containment is the thing the lane teaches), so
    they are measured and printed but carry no bar."""
    y = df["label"].to_list()
    claims, chunks = df["claim"].to_list(), df["chunk"].to_list()
    ch = {
        "claim_char_length": [float(len(c)) for c in claims],
        "claim_token_count": [float(len(tokens(c))) for c in claims],
        "chunk_char_length": [float(len(c)) for c in chunks],
        "claim_chunk_containment": [containment(c, k) for c, k in zip(claims, chunks)],
    }
    ch.update(extra or {})
    vals = {k: round(auroc(y, v), 4) for k, v in ch.items()}
    barred = {k: v for k, v in vals.items() if k not in report_only}
    worst = max(abs(v - 0.5) for v in barred.values())
    return {"auroc": vals, "report_only": list(report_only),
            "bar": "each barred channel in [0.45, 0.55]",
            "worst_deviation": round(worst, 4), "pass": bool(worst <= 0.05)}


def pair_integrity(df):
    """Every pair_id carries exactly one label-1 and one label-0 row."""
    g = df.group_by("pair_id").agg(pl.col("label").sum().alias("p"), pl.len().alias("n"))
    bad = g.filter((pl.col("n") != 2) | (pl.col("p") != 1))
    return {"pairs": int(g.height), "malformed": int(bad.height), "bar": "0 malformed",
            "pass": bad.height == 0,
            "examples": bad.head(5)["pair_id"].to_list()}


def dedupe(df):
    d = df.unique(subset=["claim", "chunk", "label"], keep="first", maintain_order=True)
    keep = d.group_by("pair_id").len().filter(pl.col("len") == 2)["pair_id"]
    return d.filter(pl.col("pair_id").is_in(keep)).sort(["pair_id", "label"],
                                                        descending=[False, True])
