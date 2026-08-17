"""R21-H180 - build the RAGTruth held-out evaluation surface. CPU ONLY, zero GPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R21-H180 RAGTRUTH HELD-OUT EVAL". This script does step 1 only - BUILD the
eval into the serving shape and bank it. No model is scored here.

Source: the English RAGTruth archive's TEST split, whole - 2,700 rows over 450
distinct contexts, 6 responder models per context. No filter drops a row (the
loader's `context > 50 chars` predicate and the lineage read's extra
`output > 20 chars` predicate both pass on 2,700 of 2,700), and no sampling is
applied, so the banked surface is the registered 2,700 / 450 exactly.

SHAPE - the flat long form of `R20-H177_eval_B.parquet`, one row per serving
unit with `claim` / `chunk` / `label` / `pair_id` / `doc_id` / `source`. That is
the shape the banked read machinery consumes without modification:
`R20_baseline_legs.flatten()` windows `chunk` at 1,500/750 and takes the max
over the window bag, so the evidence enters UNTRUNCATED and the window bag IS
the claim's evidence chunk set. The `gold_full` shape (claim + explicit chunk
list, grouped on `owner`) is the same long form with a grouping key; it is
reachable from this parquet as `[[chunk]]` per row, which the arena's
decomposed-min reader expands into the identical window bag. Nothing is cut, so
neither reader loses evidence.

LABEL UNIT - a whole response, not a claim. A RAGTruth row is one model's entire
answer (mean 776 characters here) labelled 0 if ANY annotated span inside it is
unsupported. The unit is written into the artifact (`label_unit` column, every
row `response`) and the span annotations are carried through
(`span_starts` / `span_ends` / `span_texts` / `span_types` and the per-type
counts), so a claim-level version can be derived later without re-deriving the
eval.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R21-H180_eval_build.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # three training draws own all three cards
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
ARCHIVE = DATA / "dataset-ragtruth.zip"
OUT = HERE / "R21-H180_ragtruth_eval.parquet"

SOURCE = "ragtruth_en_test"
WIN, STRIDE = 1500, 750
EXPECTED_ROWS, EXPECTED_CONTEXTS = 2700, 450

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
SPAN_TYPES = ("Evident Conflict", "Evident Baseless Info",
              "Subtle Conflict", "Subtle Baseless Info")


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
    name = next(n for n in z.namelist() if n.endswith("__test.parquet"))
    te = pl.read_parquet(io.BytesIO(z.read(name)))
    if te.height != EXPECTED_ROWS:
        raise SystemExit(f"BUILD ABORT: test split {te.height} rows, want {EXPECTED_ROWS}")
    if te["context"].n_unique() != EXPECTED_CONTEXTS:
        raise SystemExit(f"BUILD ABORT: {te['context'].n_unique()} contexts, "
                         f"want {EXPECTED_CONTEXTS}")

    # the campaign's label expression, byte-identical to the one `public_train`
    # applies to the ragtruth_en TRAIN member, so the eval scores the same predicate
    te = te.with_columns(
        (
            (pl.col("hallucination_labels_processed").struct.field("evident_conflict") == 0)
            & (pl.col("hallucination_labels_processed").struct.field("baseless_info") == 0)
        ).cast(pl.Int64).alias("label")
    )

    # loader predicates recorded rather than assumed: neither drops a row here
    dropped = {
        "context_le_50_chars": int((te["context"].str.len_chars() <= 50).sum()),
        "output_le_20_chars": int((te["output"].str.len_chars() <= 20).sum()),
    }

    ctx_index = {}
    for c in te["context"].to_list():
        ctx_index.setdefault(c, f"{SOURCE}:ctx{len(ctx_index):03d}")

    starts, ends, texts, types, counts, n_spans, any_span = [], [], [], [], [], [], []
    for s in te["hallucination_labels"].to_list():
        items = [] if (not s or s.strip() in ("", "[]")) else json.loads(s)
        starts.append([int(i.get("start", -1)) for i in items])
        ends.append([int(i.get("end", -1)) for i in items])
        texts.append([str(i.get("text", "")) for i in items])
        types.append([str(i.get("label_type", "")) for i in items])
        counts.append([sum(1 for i in items if i.get("label_type") == t) for t in SPAN_TYPES])
        n_spans.append(len(items))
        any_span.append(int(len(items) == 0))
    counts = np.array(counts, dtype=np.int64)

    df = pl.DataFrame({
        "pair_id": np.arange(te.height, dtype=np.int64),
        "label": te["label"].to_numpy(),
        "claim": te["output"].to_list(),
        "chunk": te["context"].to_list(),
        "doc_id": [ctx_index[c] for c in te["context"].to_list()],
        "source": [SOURCE] * te.height,
        "label_unit": ["response"] * te.height,
        "ragtruth_id": te["id"].to_list(),
        "task_type": te["task_type"].to_list(),
        "model": te["model"].to_list(),
        "quality": te["quality"].to_list(),
        "temperature": te["temperature"].to_list(),
        "query": te["query"].to_list(),
        "label_any_span": np.array(any_span, dtype=np.int64),
        "n_spans": np.array(n_spans, dtype=np.int64),
        "n_evident_conflict": counts[:, 0],
        "n_evident_baseless_info": counts[:, 1],
        "n_subtle_conflict": counts[:, 2],
        "n_subtle_baseless_info": counts[:, 3],
        "span_starts": starts,
        "span_ends": ends,
        "span_texts": texts,
        "span_types": types,
        "hallucination_labels_raw": te["hallucination_labels"].to_list(),
        "response_chars": te["output"].str.len_chars().to_numpy(),
        "response_tokens": np.array([len(_WORD.findall(t)) for t in te["output"].to_list()],
                                    dtype=np.int64),
        "context_chars": te["context"].str.len_chars().to_numpy(),
        "n_windows": np.array([len(windows(c)) for c in te["context"].to_list()],
                              dtype=np.int64),
    })
    return df, dropped


def census(df, dropped):
    y = df["label"].to_numpy()
    rc = df["response_chars"].to_numpy()
    rt = df["response_tokens"].to_numpy()
    cc = df["context_chars"].to_numpy()

    def q(v):
        return {"mean": round(float(v.mean()), 1),
                "median": round(float(np.median(v)), 1),
                "p10": round(float(np.percentile(v, 10)), 1),
                "p90": round(float(np.percentile(v, 90)), 1),
                "min": int(v.min()), "max": int(v.max())}

    by_task = {}
    for tt in sorted(set(df["task_type"].to_list())):
        d = df.filter(pl.col("task_type") == tt)
        yy = d["label"].to_numpy()
        by_task[tt] = {
            "rows": int(d.height),
            "contexts": int(d["doc_id"].n_unique()),
            "positives": int((yy == 1).sum()),
            "negatives": int((yy == 0).sum()),
            "positive_rate": round(float(yy.mean()), 4),
            "response_chars_mean": round(float(d["response_chars"].mean()), 1),
            "rows_with_any_span": int((d["n_spans"] > 0).sum()),
        }
    by_model = {}
    for mdl in sorted(set(df["model"].to_list())):
        d = df.filter(pl.col("model") == mdl)
        by_model[mdl] = {"rows": int(d.height),
                         "positive_rate": round(float(d["label"].mean()), 4)}

    per_ctx = df.group_by("doc_id").agg(pl.col("label").mean().alias("p"),
                                        pl.len().alias("n"))
    return {
        "rows": int(df.height),
        "contexts": int(df["doc_id"].n_unique()),
        "rows_per_context": {"min": int(per_ctx["n"].min()), "max": int(per_ctx["n"].max())},
        "distinct_claims": int(df["claim"].n_unique()),
        "distinct_chunks": int(df["chunk"].n_unique()),
        "label_unit": "response - one whole model answer, labelled 0 if ANY annotated "
                      "span inside it is unsupported",
        "label_expression": "(hallucination_labels_processed.evident_conflict == 0) AND "
                            "(hallucination_labels_processed.baseless_info == 0)",
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "positive_rate": round(float(y.mean()), 4),
        "label_equals_zero_annotated_spans": bool(
            int((df["label"] != df["label_any_span"]).sum()) == 0),
        "rows_with_any_span": int((df["n_spans"] > 0).sum()),
        "span_totals": {
            "evident_conflict": int(df["n_evident_conflict"].sum()),
            "evident_baseless_info": int(df["n_evident_baseless_info"].sum()),
            "subtle_conflict": int(df["n_subtle_conflict"].sum()),
            "subtle_baseless_info": int(df["n_subtle_baseless_info"].sum()),
            "spans_total": int(df["n_spans"].sum()),
        },
        "response_chars": q(rc),
        "response_tokens": q(rt),
        "context_chars": q(cc),
        "windows_1500_750": {
            "mean": round(float(df["n_windows"].mean()), 4),
            "multi_window_share": round(float((df["n_windows"].to_numpy() > 1).mean()), 4),
            "max": int(df["n_windows"].max()),
            "total_claim_window_pairs": int(df["n_windows"].sum()),
        },
        "per_task_type": by_task,
        "per_responder_model": by_model,
        "contexts_carrying_both_labels": int(((per_ctx["p"] > 0) & (per_ctx["p"] < 1)).sum()),
        "rows_dropped_by_loader_predicates": dropped,
        "quality_flags": {r["quality"]: int(r["count"])
                          for r in df["quality"].value_counts().iter_rows(named=True)},
    }


def main():
    if OUT.exists():
        df = pl.read_parquet(OUT)
        if df.height == EXPECTED_ROWS and "label_unit" in df.columns:
            print(f"already banked: {OUT} ({df.height} rows, "
                  f"{df['doc_id'].n_unique()} contexts) - verifying only", flush=True)
            print(json.dumps(census(df, {"context_le_50_chars": 0,
                                         "output_le_20_chars": 0}), indent=2), flush=True)
            return
        print(f"existing {OUT} is incomplete ({df.height} rows) - rebuilding", flush=True)

    df, dropped = build()
    df.write_parquet(OUT)
    print(f"banked -> {OUT}", flush=True)
    print(json.dumps(census(df, dropped), indent=2), flush=True)


if __name__ == "__main__":
    main()
