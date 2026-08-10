"""R13-H128 - build the WiCE attributed-support lane parquet.

The pre-GPU gates (`R13-H128_gates_result.json`) cleared WiCE at 68,380
buildable pairs; this builder ships the MOST-CONSERVATIVE construction the
gates costed separately - `buildable_pairs_min_set_with_swap`, claim level
7,636 + subclaim level 10,714 = 18,350 - which is the default the launch order
fixes. Only the SMALLEST multi-sentence evidence set of each claim is used, so
no claim contributes more than one evidence set and the near-duplicate mass of
the all-sets construction never enters.

Construction (identical in kind to the gates' 200-pair sample):

  positive  = claim + the full minimal evidence set                    label 1
  negative  = the same claim with one sentence of that set DELETED     label 0
  negative  = the same claim with that sentence SWAPPED for the        label 0
              lexically nearest sentence of a different article

Builder notes (binding, from the gates block):

  * claim-level evidence indices are strings and subclaim-level ints - both are
    coerced to int on load
  * WiCE rows labelled `not_supported` are DROPPED (20 rows, 86 negatives):
    their positive member would assert support for a claim WiCE annotates as
    unsupported. Everything else is kept, including `partially_supported` -
    restricting positives to `supported` alone yields 11,880 pairs and would
    fail the registered >= 15,000 pre-GPU pairs gate
  * exact duplicate rows are dropped: the positive of a k-sentence minimal set
    is shared by that set's 2k negatives, and emitting it 2k times would be a
    silent 2k-fold reweight of one (claim, chunk) pair

Output schema matches the H108 lane parquet exactly (`claim`, `chunk`, `label`,
`tag`) so the lane trainers consume it unchanged. Chunks are stored untruncated;
the trainer truncates to the serving unit at load, as H108 does.

Run:  uv run python experiments/grounding-semantic/R13-H128_build_lane.py
"""

import collections
import importlib.util
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R13-H128_lane.parquet"
TAG = "wice_attrib"
KEEP_LABELS = ("supported", "partially_supported")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _mod("h128_gates", "R13-H128_gates.py")


def minimal_set(row):
    """The smallest usable multi-sentence evidence set, or None."""
    multi = [s for s in row["supporting_sentences"]
             if len(s) >= 2 and max(s) < len(row["evidence"])]
    return min(multi, key=len) if multi else None


def main():
    rows = []
    for level in ("claim", "subclaim"):
        for r in G.load_wice(level):
            r["level"] = level
            rows.append(r)

    nearest = G.NearestSentence(rows)

    recs = []
    stats = collections.Counter()
    for r in rows:
        if r["label"] not in KEEP_LABELS:
            stats[f"dropped_label_{r['label']}"] += 1
            continue
        s = minimal_set(r)
        if s is None:
            stats["dropped_no_multi_sentence_set"] += 1
            continue
        ev = r["evidence"]
        pos = " ".join(ev[i] for i in s)
        recs.append({"claim": r["claim"], "chunk": pos, "label": 1.0, "tag": TAG})
        stats[f"positives_{r['level']}"] += 1
        for j in s:
            recs.append({"claim": r["claim"], "label": 0.0, "tag": TAG,
                         "chunk": " ".join(ev[i] for i in s if i != j)})
            stats[f"deletion_negatives_{r['level']}"] += 1
            swapped, _ = nearest(ev[j], r["meta"]["claim_title"])
            if swapped is None:
                stats["swap_donorless_fell_back_to_deletion"] += 1
                continue
            recs.append({"claim": r["claim"], "label": 0.0, "tag": TAG,
                         "chunk": " ".join(swapped if i == j else ev[i] for i in s)})
            stats[f"swap_negatives_{r['level']}"] += 1

    df = pl.DataFrame(recs, schema={"claim": pl.String, "chunk": pl.String,
                                    "label": pl.Float32, "tag": pl.String})
    # A handful of (claim, chunk) pairs carry both labels: the same claim text
    # appears at BOTH annotation levels with different minimal sets, so one
    # level's corrupted chunk can coincide with the other level's positive.
    # Those pairs teach nothing and are dropped whole.
    conflict = (df.group_by(["claim", "chunk"]).agg(pl.col("label").n_unique().alias("n"))
                  .filter(pl.col("n") > 1).select(["claim", "chunk"]))
    n_raw = len(df)
    df = df.join(conflict, on=["claim", "chunk"], how="anti")
    n_conflict = n_raw - len(df)
    if n_conflict > 0.01 * n_raw:
        raise SystemExit(f"LABEL-CONFLICT ABORT: {n_conflict}/{n_raw} rows collide across "
                         "labels; the construction is not separable.")
    df = df.unique(subset=["claim", "chunk"], keep="first", maintain_order=True)

    pairs = stats["deletion_negatives_claim"] + stats["swap_negatives_claim"] \
        + stats["deletion_negatives_subclaim"] + stats["swap_negatives_subclaim"] \
        + stats["swap_donorless_fell_back_to_deletion"]
    df.write_parquet(OUT)

    for k in sorted(stats):
        print(f"  {k:<42} {stats[k]:>7}")
    print(f"\n  buildable pairs (negatives, min-set + swap)  {pairs:>7}  (gate bar 15,000)")
    print(f"  rows emitted                                 {n_raw:>7}")
    print(f"  rows dropped as cross-label collisions       {n_conflict:>7}")
    print(f"  rows after exact-duplicate drop              {len(df):>7}")
    print(f"  positives                                    {int(df['label'].sum()):>7} "
          f"({df['label'].mean():.4f})")
    print(f"  chunk chars: median {df['chunk'].str.len_chars().median():.0f}  "
          f"p95 {df['chunk'].str.len_chars().quantile(0.95):.0f}  "
          f"max {df['chunk'].str.len_chars().max()}")
    print(f"\nlane -> {OUT}")


if __name__ == "__main__":
    main()
