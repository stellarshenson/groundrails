"""The canonical split for round 8 - by SOURCE DOCUMENT, defined exactly once.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

Three splits have been used on this gold and two of them leaked:

  1. by CLAIM      - leaked whole documents; INVERTED the capacity ordering in
                     R7-H50 v1, worth 0.050 AUC and the wrong model
  2. by TRACE      - fixed the trace correlation but NOT the document one: 639
                     traces share only 619 source texts, so two unrelated traces
                     retrieve the same passage. Audit found 95.8% of test claims
                     carrying at least one chunk seen in training, 42% of their
                     chunks on average, and 29 exact (claim, chunk) pairs
                     repeated outright
  3. by DOCUMENT   - this module. A source text and every chunk derived from it
                     fall entirely on one side, so no chunk a model trained on
                     can appear in its evaluation

The split lives here rather than in each experiment because the second failure
was caused by two implementations of "the same" split drifting apart. One
definition, imported everywhere, cannot drift.

Document identity is the sha1 of the raw `source_text`, so it is stable across
runs, machines and dataframe orderings - unlike a row index or a shuffled
position.
"""

import hashlib

import numpy as np
import polars as pl

TEST_FRAC = 0.25
VAL_FRAC = 0.15
SEED = 0


def doc_id(source_text: str) -> str:
    """Stable identity for a source document."""
    return hashlib.sha1(source_text.encode("utf-8")).hexdigest()[:16]


def attach_doc_ids(pairs: pl.DataFrame, gold: pl.DataFrame) -> pl.DataFrame:
    """Join a document id onto the teacher pairs via their owning gold row."""
    g = gold.with_row_index("owner").with_columns(
        pl.col("source_text").map_elements(doc_id, return_dtype=pl.String).alias("doc_id")
    )
    return pairs.join(g.select(["owner", "doc_id", "trace_id"]), on="owner", how="left")


def _components(df: pl.DataFrame) -> dict:
    """Group claims into connected components that share any chunk.

    Splitting by `doc_id` is NOT sufficient and was measured so: distinct source
    texts contain identical passages - boilerplate, repeated sections, shared
    headers - which left 815 shared chunks and 125 shared (claim, chunk) pairs
    across a document-level boundary.

    The leak unit is the CHUNK, not the document. Two claims are joined whenever
    they share one, and the transitive closure of that relation is the smallest
    unit that can be split without a chunk crossing the boundary. Union-find
    over chunk texts, so it is linear in pairs.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first_owner_of_chunk = {}
    for owner, chunk in zip(df["owner"].to_list(), df["chunk"].to_list(), strict=True):
        find(owner)
        if chunk in first_owner_of_chunk:
            union(first_owner_of_chunk[chunk], owner)
        else:
            first_owner_of_chunk[chunk] = owner
    return {o: find(o) for o in set(df["owner"].to_list())}


def split_docs(df: pl.DataFrame):
    """Return (train, val, test) as sets of COMPONENT ids, split by claim mass.

    Components vary wildly in size, so they are packed largest-first into the
    split with the largest remaining deficit. Shuffling components and slicing
    by count would put most of the corpus on one side whenever one component
    dominates.
    """
    comp = _components(df)
    sizes = {}
    for c in comp.values():
        sizes[c] = sizes.get(c, 0) + 1
    total = sum(sizes.values())
    targets = {
        "test": TEST_FRAC * total,
        "val": VAL_FRAC * total,
        "train": (1 - TEST_FRAC - VAL_FRAC) * total,
    }
    got = {"train": 0, "val": 0, "test": 0}
    out = {"train": set(), "val": set(), "test": set()}
    rng = np.random.default_rng(SEED)
    order = sorted(sizes, key=lambda c: (-sizes[c], c))
    rng.shuffle(order[len(order) // 2 :])  # break ties among the long tail
    for c in order:
        bucket = max(got, key=lambda k: targets[k] - got[k])
        out[bucket].add(c)
        got[bucket] += sizes[c]
    return out["train"], out["val"], out["test"]


def frames(pairs: pl.DataFrame, gold: pl.DataFrame):
    """The three frames, plus the component sets, from one call."""
    df = attach_doc_ids(pairs, gold)
    comp = _components(df)
    df = df.with_columns(
        pl.col("owner").map_elements(lambda o: comp[o], return_dtype=pl.Int64).alias("comp")
    )
    train, val, test = split_docs(df)
    return (
        df.filter(pl.col("comp").is_in(list(train))),
        df.filter(pl.col("comp").is_in(list(val))),
        df.filter(pl.col("comp").is_in(list(test))),
        {"train": train, "val": val, "test": test},
    )


def audit(train_df: pl.DataFrame, test_df: pl.DataFrame) -> dict:
    """Prove the split holds. Any non-zero here is a defect, not a tolerance."""
    tr_chunks = set(train_df["chunk"].to_list())
    te_chunks = set(test_df["chunk"].to_list())
    tr_pairs = set(zip(train_df["claim"].to_list(), train_df["chunk"].to_list(), strict=True))
    te_pairs = set(zip(test_df["claim"].to_list(), test_df["chunk"].to_list(), strict=True))
    return {
        "doc_overlap": len(set(train_df["doc_id"].to_list()) & set(test_df["doc_id"].to_list())),
        "chunk_overlap": len(tr_chunks & te_chunks),
        "pair_overlap": len(tr_pairs & te_pairs),
        "claim_overlap": len(set(train_df["claim"].to_list()) & set(test_df["claim"].to_list())),
        "train_pairs": len(train_df),
        "test_pairs": len(test_df),
    }
