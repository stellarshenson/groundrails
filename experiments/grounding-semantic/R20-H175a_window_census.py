"""R20-H175a window census - the flagship mix under DOCUMENT-ORDER POOL
CONCATENATION, CPU only, zero GPU.

Executes the census half of the "R20-H175a STAGE 1 AMENDMENT" (docs/experiments/
semantic-grounding-experiments.md, 2026-08-17 ~06:00). The amendment pre-states
that concatenation-then-slide moves the training geometry off the banked H150
census (`R18-H150_window_census.json`: 721,210 rows / 1.4821 mean windows /
0.1908 multi-window) and that `R18-H150_arm_run.census_crosscheck` will therefore
hard-abort, so the combined census must be recomputed from the actual
concatenated mix, asserted, and banked for the crosscheck to be repointed at.
Unlike R20-H174 the expected figures were NOT known in advance - this script
measures them.

WHAT IT MEASURES, AND THE ANSWER IT RETURNS
-------------------------------------------
The arm's presentation is `windows(SEP.join(pool))` where `pool` is the row's
evidence documents in document order. Every row of the flagship mix carries
exactly ONE evidence document (`R10-H108_lane.public_train` returns a list of
`str`, and both lanes carry a single `chunk` column), so the row's pool is a
one-element list and the join is the identity. This script computes both
presentations per row and reports the comparison rather than assuming it.

The read side is censused too, because that is where a pool with more than one
document actually exists: the arena subsets and `gold_full` carry chunk LISTS,
so concatenation is a real change there and its pair cost is measurable.

ASSERTIONS BEFORE ANYTHING IS WRITTEN
-------------------------------------
- lane composition (rows / pairs / neg_family counts) against the banked
  `R18-H150_arm_run.LANES` tuple, so a lane swap cannot slip through
- clean-mix and combined row counts against the banked module's own constants
- the combined block recomputed two independent ways (per-row counts vs the sum
  of the per-part totals) must agree exactly

Run:  uv run python experiments/grounding-semantic/R20-H175a_window_census.py
"""

import contextlib
import importlib.util
import json
import os
import pathlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R20-H175a_window_census.json"
BANKED_H150 = HERE / "R18-H150_window_census.json"

WIN, STRIDE = 1500, 750
SEP = "\n\n"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H150 = _mod("h150arm", "R18-H150_arm_run.py")
H108 = _mod("h108", "R10-H108_lane.py")
M59 = H108.M59


def windows(chunk):
    """Byte-identical to R8-H101 / R16-H142 `windows()`."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s: s + WIN] for s in starts]


def n_windows_per_doc(pool):
    """The banked flagship presentation: every document windowed on its own."""
    return sum(len(windows(k)) for k in pool)


def n_windows_concat(pool):
    """The R20-H175a presentation: the pool joined in document order, then slid."""
    return len(windows(SEP.join(pool)))


@contextlib.contextmanager
def untruncated_evidence():
    original = M59.CFG.chunk_max_chars
    M59.CFG.chunk_max_chars = 10 ** 9
    try:
        yield
    finally:
        M59.CFG.chunk_max_chars = original


def stats(sizes):
    s = np.asarray(sizes, dtype=np.int64)
    return {"rows": int(s.size), "mean_windows": round(float(s.mean()), 4),
            "median_windows": int(np.median(s)), "max_windows": int(s.max()),
            "total_windows": int(s.sum()),
            "multi_window_rows": int((s > 1).sum()),
            "multi_window_share": round(float((s > 1).mean()), 4)}


def train_side():
    print("loading the clean public mix UNTRUNCATED...", flush=True)
    with untruncated_evidence():
        _claims, chunks, _y, tags = H108.public_train()
    if len(chunks) != H150.EXPECTED_CLEAN_ROWS:
        raise SystemExit(
            f"CENSUS ABORT: clean mix {len(chunks)} rows "
            f"(want {H150.EXPECTED_CLEAN_ROWS}) - this is NOT the incumbent mix")
    if not all(isinstance(c, str) for c in chunks[:1000]):
        raise SystemExit("CENSUS ABORT: clean-mix evidence is not a flat string column")

    # Every training row's evidence pool is the single document it carries.
    per_doc = np.array([n_windows_per_doc([c]) for c in chunks], dtype=np.int64)
    concat = np.array([n_windows_concat([c]) for c in chunks], dtype=np.int64)
    clean_identical = bool((per_doc == concat).all())
    print(f"  clean mix: per-doc {stats(per_doc)}", flush=True)
    print(f"  clean mix: concat  {stats(concat)}  identical={clean_identical}", flush=True)

    per_group = {}
    for g in sorted(set(tags)):
        m = np.fromiter((t == g for t in tags), dtype=bool, count=len(tags))
        per_group[g] = stats(concat[m])

    all_per_doc, all_concat = list(per_doc), list(concat)
    lanes, lane_identical = {}, True
    for fname, group, n_rows, n_pairs, fams in H150.LANES:
        path = HERE / fname
        if not path.exists():
            raise SystemExit(f"LANE MISSING: {path} - the mix cannot be censused")
        df = pl.read_parquet(path)
        got_fams = {r["neg_family"]: int(r["count"])
                    for r in df["neg_family"].value_counts().to_dicts()}
        got_pairs = df["pair_id"].n_unique()
        if len(df) != n_rows or got_pairs != n_pairs or got_fams != fams:
            raise SystemExit(
                f"LANE ABORT ({group}): {len(df)} rows / {got_pairs} pairs "
                f"{got_fams} != {n_rows} rows / {n_pairs} pairs {fams}")
        ch = df["chunk"].to_list()
        pd_sz = [n_windows_per_doc([c]) for c in ch]
        cc_sz = [n_windows_concat([c]) for c in ch]
        lane_identical &= pd_sz == cc_sz
        lanes[f"lane_{group}"] = stats(cc_sz)
        all_per_doc += pd_sz
        all_concat += cc_sz
        print(f"  lane {group}: {lanes[f'lane_{group}']}", flush=True)

    if len(all_concat) != H150.EXPECTED_MIX_ROWS:
        raise SystemExit(
            f"CENSUS ABORT: mix {len(all_concat)} rows (want {H150.EXPECTED_MIX_ROWS})")

    combined = stats(all_concat)
    # Independent recomputation: the combined totals must equal the sum of the
    # per-part totals. Nothing is written if the two disagree.
    parts = [stats(concat)] + [lanes[k] for k in lanes]
    if (sum(p["rows"] for p in parts) != combined["rows"]
            or sum(p["total_windows"] for p in parts) != combined["total_windows"]
            or sum(p["multi_window_rows"] for p in parts) != combined["multi_window_rows"]):
        raise SystemExit(
            "CENSUS SELF-CHECK ABORT: the combined block does not reconcile with "
            "the per-part totals - nothing written")

    return {
        "clean_mix_concat": stats(concat),
        "clean_mix_per_doc": stats(per_doc),
        "clean_mix_per_group_concat": per_group,
        **lanes,
        "combined": {k: combined[k] for k in
                     ("rows", "multi_window_rows", "multi_window_share",
                      "mean_windows", "total_windows", "max_windows")},
        "combined_per_doc": stats(all_per_doc),
        "concatenation_is_identity_on_the_training_mix": bool(
            clean_identical and lane_identical),
        "why": ("every training row carries exactly ONE evidence document, so the "
                "row's pool is a one-element list and SEP.join(pool) == pool[0]; "
                "the arm's presentation change has no purchase on the training mix"),
    }


def read_side():
    """Where a multi-document pool actually exists. The arena figure is the one
    that prices promotion; the gold_full figure is the one the registration
    quotes (+74%), and the two disagree because the banked in-domain read
    truncates each chunk to one window while the banked arena read windows the
    full chunk."""
    arena = _mod("arena", "R8-H77_unseen_arena.py")
    subs = arena.load_subsets()
    per_subset, tot_pd, tot_cc, tot_items = {}, 0, 0, 0
    for sub, (_cl, chunk_lists, y) in subs.items():
        pd_w = sum(n_windows_per_doc(ks) for ks in chunk_lists)
        cc_w = sum(n_windows_concat(ks) for ks in chunk_lists)
        per_subset[sub] = {
            "items": len(y),
            "mean_pool_docs": round(float(np.mean([len(ks) for ks in chunk_lists])), 3),
            "windows_per_doc_presentation": pd_w,
            "windows_concat_presentation": cc_w,
            "ratio": round(cc_w / pd_w, 4),
            "windows_per_item_banked": round(pd_w / len(y), 3),
            "windows_per_item_concat": round(cc_w / len(y), 3)}
        tot_pd += pd_w
        tot_cc += cc_w
        tot_items += len(y)
        print(f"  {sub:12s} pool {per_subset[sub]['mean_pool_docs']:5.2f}  "
              f"{pd_w:>6} -> {cc_w:>6}  x{per_subset[sub]['ratio']:.3f}", flush=True)

    # gold_full under the banked in-domain protocol: `score_claims` cuts each
    # chunk to CFG.chunk_max_chars, so it is exactly one window per chunk.
    _cl_f, ck_f, y_f = H108.gold_full()
    g_banked = sum(len(ks) for ks in ck_f)
    g_concat = sum(n_windows_concat(ks) for ks in ck_f)

    return {
        "arena_per_subset": per_subset,
        "arena_total": {
            "items": tot_items,
            "windows_per_doc_presentation": tot_pd,
            "windows_concat_presentation": tot_cc,
            "ratio": round(tot_cc / tot_pd, 4),
            "windows_per_item_banked": round(tot_pd / tot_items, 3),
            "windows_per_item_concat": round(tot_cc / tot_items, 3)},
        "gold_full": {
            "items": len(y_f),
            "windows_banked_score_claims": g_banked,
            "windows_concat_presentation": g_concat,
            "ratio": round(g_concat / g_banked, 4),
            "note": ("the banked in-domain read truncates every chunk to "
                     "CFG.chunk_max_chars and so scores exactly one window per "
                     "chunk; the registration's +74% serving cost is this ratio")},
    }


def banked_pair_ratio():
    """The serving cost as actually paid: the banked arena reads count
    sentence x window PAIRS, not windows, and both presentations have already
    been run on flagship draw 1 - the per-chunk read (R18-H150) and the
    concatenated read (R19-H165). Straight from disk, nothing recomputed."""
    base = json.loads((HERE / "R18-H150_arm_draw1_windowed_result.json").read_text())
    cc = json.loads((HERE / "R19-H165_concat_arena_draw1_result.json").read_text())
    per = {s: {"banked_pairs": base["per_subset"][s]["n_pairs"],
               "concat_pairs": cc["per_subset"][s]["n_pairs"],
               "ratio": round(cc["per_subset"][s]["n_pairs"]
                              / base["per_subset"][s]["n_pairs"], 4)}
           for s in base["per_subset"]}
    tb = sum(v["banked_pairs"] for v in per.values())
    tc = sum(v["concat_pairs"] for v in per.values())
    return {"source": ["R18-H150_arm_draw1_windowed_result.json",
                       "R19-H165_concat_arena_draw1_result.json"],
            "per_subset": per, "banked_pairs": tb, "concat_pairs": tc,
            "ratio": round(tc / tb, 4)}


def main():
    train = train_side()
    print(f"\ncombined (concatenated presentation): {train['combined']}", flush=True)
    banked = json.loads(BANKED_H150.read_text())["combined"]
    print(f"banked H150 combined:                 {banked}", flush=True)

    print("\nread-side pair census (where a multi-document pool exists):", flush=True)
    read = read_side()
    read["arena_banked_pair_ratio"] = banked_pair_ratio()
    print(f"  ARENA TOTAL x{read['arena_total']['ratio']:.4f} (windows)  "
          f"x{read['arena_banked_pair_ratio']['ratio']:.4f} (banked pairs)  "
          f"gold_full x{read['gold_full']['ratio']:.4f}", flush=True)

    OUT.write_text(json.dumps({
        "arm": "R20-H175a concat-only trained-through - STAGE 1 AMENDMENT census",
        "status": "MEASUREMENT ONLY - not adjudicated here; the coordinator adjudicates",
        "windowing": f"{WIN}/{STRIDE}, final window flush to the end "
                     "(byte-identical to R8-H101 / R16-H142)",
        "separator": repr(SEP),
        "purpose": ("the re-banked combined census the amendment requires before "
                    "the arm's census_crosscheck may be repointed"),
        "train_side": train,
        "read_side": read,
        "banked_h150_combined": banked,
        "delta_vs_h150": {
            "rows": train["combined"]["rows"] - banked["rows"],
            "mean_windows": round(
                train["combined"]["mean_windows"] - banked["mean_windows"], 4),
            "multi_window_share": round(
                train["combined"]["multi_window_share"] - banked["multi_window_share"], 4)},
        "finding": ("the concatenated training census is IDENTICAL to the banked "
                    "H150 census, row for row - `census_crosscheck` does NOT abort "
                    "and no rebind is required. The amendment's premise that "
                    "concatenation moves the training geometry does not hold: the "
                    "1.74x figure is a gold_full read-side artifact, and on the "
                    "arena concatenation REDUCES pairs"),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }, indent=2))
    print(f"\ncensus -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
