"""R21-H181 - build the HaluEval-`dialogue` held-out evaluation surface. CPU ONLY.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R21-H181 HALUEVAL-DIALOGUE HELD-OUT EVAL". This script does step 1 only -
BUILD the eval into the serving shape and bank it. No model is scored here.

WHY THIS SURFACE IS ADMISSIBLE, verified rather than assumed. The loader
`R10-H108_lane.public_train()` iterates exactly two HaluEval configurations -
`("qa", "knowledge", "right_answer", "hallucinated_answer")` and
`("summarization", "document", "right_summary", "hallucinated_summary")` - with
no split filter and no row filter. `dialogue` and `general` ship in the same
archive and are never read. `general` carries no evidence column at all
(ID / user_query / chatgpt_response / hallucination / hallucination_spans), so
it is not a grounding task; `dialogue` carries `knowledge`, so it is.

SHAPE - the flat long form of `R20-H177_eval_B.parquet` (pair_id / label /
claim / chunk / doc_id / source), the same shape `R21-H180_ragtruth_eval.parquet`
took, because that is what the banked read machinery consumes unmodified:
`R20_baseline_legs.flatten()` windows `chunk` at 1,500/750 and takes the max
over the window bag. The `gold_full` shape (claim + explicit chunk LIST grouped
on `owner`, from `R10-H108_lane.gold_full`) is the same long form with a
grouping key and is reachable from this parquet as `[[chunk]]` per row. Neither
reader loses evidence: the longest knowledge block here is far under the
1,500-char cut, so truncation and windowing are both no-ops on this surface -
counted below, not asserted.

LABEL UNIT - a whole conversational response. Each archive row supplies ONE
knowledge block and TWO responses; the positive leg is `right_response`, the
negative leg `hallucinated_response`. `dialogue_history` is carried through as a
column even though the (claim, chunk) score path does not consume it.

VOLUME, measured: the archive ships 10,000 `dialogue` rows, so the built surface
is 20,000 rows over 10,000 contrast pairs - twice the "10,000 rows over 5,000
contrast pairs" in the registration block, which counted archive rows as if they
were already leg-expanded. Recorded as a finding; nothing is adjudicated here.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R21-H181_eval_build.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU0/1/2 carry a training draw and an arena pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
ARCHIVE = DATA / "dataset-halueval.zip"
MEMBER = "pminervini__HaluEval__dialogue__data.parquet"
OUT = HERE / "R21-H181_halueval_dialogue_eval.parquet"

SOURCE = "halueval_dialogue"
WIN, STRIDE = 1500, 750
CHUNK_MAX_CHARS = 1500  # R7-H59 CFG.chunk_max_chars, the serving cut
EXPECTED_ARCHIVE_ROWS = 10_000
EXPECTED_ROWS = 20_000
EXPECTED_PAIRS = 10_000

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def windows(chunk):
    """R8-H101 / R16-H142 G0 windowing, byte-identical - counted, not applied."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s: s + WIN] for s in starts]


def build():
    z = zipfile.ZipFile(ARCHIVE)
    if MEMBER not in z.namelist():
        raise SystemExit(f"BUILD ABORT: {MEMBER} absent from {ARCHIVE.name}")
    d = pl.read_parquet(io.BytesIO(z.read(MEMBER)))
    if d.height != EXPECTED_ARCHIVE_ROWS:
        raise SystemExit(f"BUILD ABORT: dialogue {d.height} rows, want {EXPECTED_ARCHIVE_ROWS}")
    want = {"knowledge", "dialogue_history", "right_response", "hallucinated_response"}
    if not want <= set(d.columns):
        raise SystemExit(f"BUILD ABORT: columns {d.columns} missing {want - set(d.columns)}")

    know = d["knowledge"].to_list()
    hist = d["dialogue_history"].to_list()
    pos = d["right_response"].to_list()
    neg = d["hallucinated_response"].to_list()

    kindex = {}
    for k in know:
        kindex.setdefault(k, f"{SOURCE}:k{len(kindex):05d}")

    pair_id, label, claim, chunk, doc_id, dh, leg, arow = [], [], [], [], [], [], [], []
    for i, (k, h, p, n) in enumerate(zip(know, hist, pos, neg, strict=True)):
        for lab, txt, nm in ((1, p, "right_response"), (0, n, "hallucinated_response")):
            pair_id.append(i)
            label.append(lab)
            claim.append(txt)
            chunk.append(k)
            doc_id.append(kindex[k])
            dh.append(h)
            leg.append(nm)
            arow.append(i)

    df = pl.DataFrame({
        "pair_id": np.array(pair_id, dtype=np.int64),
        "label": np.array(label, dtype=np.int64),
        "claim": claim,
        "chunk": chunk,
        "doc_id": doc_id,
        "source": [SOURCE] * len(claim),
        "label_unit": ["response"] * len(claim),
        "leg": leg,
        "dialogue_history": dh,
        "archive_row": np.array(arow, dtype=np.int64),
        "response_chars": np.array([len(t) for t in claim], dtype=np.int64),
        "response_tokens": np.array([len(_WORD.findall(t)) for t in claim], dtype=np.int64),
        "knowledge_chars": np.array([len(t) for t in chunk], dtype=np.int64),
        "knowledge_tokens": np.array([len(_WORD.findall(t)) for t in chunk], dtype=np.int64),
        "dialogue_history_chars": np.array([len(t) for t in dh], dtype=np.int64),
        "n_windows": np.array([len(windows(c)) for c in chunk], dtype=np.int64),
        "chunk_truncated_at_1500": np.array([len(c) > CHUNK_MAX_CHARS for c in chunk],
                                            dtype=bool),
    })
    if df.height != EXPECTED_ROWS:
        raise SystemExit(f"BUILD ABORT: built {df.height} rows, want {EXPECTED_ROWS}")
    if df["pair_id"].n_unique() != EXPECTED_PAIRS:
        raise SystemExit(f"BUILD ABORT: {df['pair_id'].n_unique()} pairs, want {EXPECTED_PAIRS}")
    return df


def q(v):
    v = np.asarray(v)
    return {"mean": round(float(v.mean()), 1), "median": round(float(np.median(v)), 1),
            "p10": round(float(np.percentile(v, 10)), 1),
            "p90": round(float(np.percentile(v, 90)), 1),
            "min": int(v.min()), "max": int(v.max())}


def census(df):
    y = df["label"].to_numpy()
    posd = df.filter(pl.col("label") == 1)
    negd = df.filter(pl.col("label") == 0)

    # C1 structural test, C-A1: does any negative leg's (claim, evidence) equal a
    # positive leg's?  Both legs share the knowledge block, so this asks whether
    # the two responses of any pair are the same string.
    key_pos = set(zip(posd["claim"].to_list(), posd["chunk"].to_list(), strict=True))
    key_neg = set(zip(negd["claim"].to_list(), negd["chunk"].to_list(), strict=True))
    collide = key_pos & key_neg

    per_pair_same = int(sum(
        1 for p, n in zip(posd["claim"].to_list(), negd["claim"].to_list(), strict=True)
        if p == n))

    kn = df["knowledge_chars"].to_numpy()
    return {
        "rows": int(df.height),
        "pairs": int(df["pair_id"].n_unique()),
        "unit_note": ("rows = serving units (one claim + one chunk); pairs = contrast "
                      "pairs (one knowledge block, a right_response and a "
                      "hallucinated_response). Both are reported everywhere - C7"),
        "archive_rows_read": EXPECTED_ARCHIVE_ROWS,
        "registration_block_said": "10,000 rows over 5,000 contrast pairs",
        "measured_instead": f"{df.height} rows over {df['pair_id'].n_unique()} contrast pairs",
        "registration_discrepancy": ("the registration counted the archive's 10,000 "
                                     "dialogue rows as leg-expanded rows; each archive row "
                                     "supplies TWO serving rows, so the surface is 2x the "
                                     "registered size in both units. Recorded, not adjudicated"),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "positive_rate": round(float(y.mean()), 4),
        "label_balance_exact_5050": bool(int((y == 1).sum()) == int((y == 0).sum())),
        "distinct_claims": int(df["claim"].n_unique()),
        "distinct_claims_positive_leg": int(posd["claim"].n_unique()),
        "distinct_claims_negative_leg": int(negd["claim"].n_unique()),
        "claims_appearing_on_both_legs": len(
            set(posd["claim"].to_list()) & set(negd["claim"].to_list())),
        "distinct_chunks": int(df["chunk"].n_unique()),
        "distinct_doc_ids": int(df["doc_id"].n_unique()),
        "knowledge_blocks_used_by_more_than_one_pair": int(
            df.group_by("doc_id").agg(pl.col("pair_id").n_unique().alias("n"))
            .filter(pl.col("n") > 1).height),
        "distinct_dialogue_histories": int(df["dialogue_history"].n_unique()),
        "C1_structural_pairs_colliding": len(collide),
        "C1_structural_rows_colliding": int(
            df.filter(
                pl.struct(["claim", "chunk"]).is_in(
                    [{"claim": c, "chunk": k} for c, k in collide]) if collide else pl.lit(False)
            ).height) if collide else 0,
        "pairs_whose_two_responses_are_the_same_string": per_pair_same,
        "knowledge_chars": q(kn),
        "knowledge_tokens": q(df["knowledge_tokens"].to_numpy()),
        "dialogue_history_chars": q(df["dialogue_history_chars"].to_numpy()),
        "response_chars_positive_leg": q(posd["response_chars"].to_numpy()),
        "response_chars_negative_leg": q(negd["response_chars"].to_numpy()),
        "response_tokens_positive_leg": q(posd["response_tokens"].to_numpy()),
        "response_tokens_negative_leg": q(negd["response_tokens"].to_numpy()),
        "serving_geometry": {
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "rows_truncated_at_1500": int(df["chunk_truncated_at_1500"].sum()),
            "windows_1500_750_mean": round(float(df["n_windows"].mean()), 4),
            "windows_1500_750_max": int(df["n_windows"].max()),
            "multi_window_rows": int((df["n_windows"].to_numpy() > 1).sum()),
            "reading": ("the longest knowledge block is under the 1,500-char serving cut, "
                        "so truncation and 1,500/750 windowing are both no-ops here and the "
                        "windowed and untruncated read paths see identical text"),
        },
        "columns": df.columns,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }


def main():
    if OUT.exists():
        df = pl.read_parquet(OUT)
        if df.height == EXPECTED_ROWS and "dialogue_history" in df.columns:
            print(f"already banked: {OUT} ({df.height} rows, "
                  f"{df['pair_id'].n_unique()} pairs) - verifying only", flush=True)
            print(json.dumps(census(df), indent=2), flush=True)
            return
        print(f"existing {OUT} is incomplete ({df.height} rows) - rebuilding", flush=True)

    df = build()
    df.write_parquet(OUT)
    print(f"banked -> {OUT}", flush=True)
    print(json.dumps(census(df), indent=2), flush=True)


if __name__ == "__main__":
    main()
