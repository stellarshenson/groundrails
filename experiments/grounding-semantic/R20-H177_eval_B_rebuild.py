"""R20-H177 eval_B REBUILD - a held-out mechanism eval genuinely disjoint from
the training mix.  CPU only, Polars only, torch-free.

WHY THE BANKED eval_B IS REBUILT
--------------------------------
`R20-H177_eval_B.parquet` (2,000 rows / 1,000 pairs / 458 docs / 736 passages) is
contaminated on two channels:

  * PASSAGE - 33 of its 736 passages sit in the flagship mix (19 byte-identical,
    14 more after whitespace normalisation), 126 rows / 63 pairs
    (`R20-H177_evalB_contamination_assessment.json`)
  * DOCUMENT - 325 of 325 of its TabFact documents are in the `tabfact` mix
    member, covering 1,300 of 2,000 rows (65%).  String matching misses this
    because the eval re-serialises the same tables into six forms while the
    member carries only the pipe form (`contract/tabfact_contract_report.json`,
    clause C2)

and its split rule has a keying defect: `blake2b(doc_id) % 1000` keys on the
doc_id STRING, while TabFact writes one Wikipedia table under BOTH a `1-` and a
`2-` prefixed csv id.  `tabfact:1-<stem>` and `tabfact:2-<stem>` are two ids for
one table, so an id-level disjointness proof does not imply document
disjointness - 15 such stem collisions exist between lane B and eval_B and one of
them produced a byte-identical shared passage.

THE SUPPLY FRONTIER, MEASURED FIRST (`R20-H177_evalB_rebuild_supply.json`)
-------------------------------------------------------------------------
The `tabfact` mix member IS the whole TabFact train split (92,585 rows, 13,182
table ids, 12,753 stems).  Of the 8,816 TabFact TRAIN tables the lane's admission
filter accepts, 8,541 of 8,541 stems are in the member: **zero TabFact train
supply survives a stem-keyed exclusion**.  The banked eval's 100% document
contamination is therefore structural, not incidental - no reseeding, no
re-splitting and no cap change can produce a clean TabFact half out of the train
split.

The only clean TabFact supply on disk is the corpus's own TEST and VALIDATION
splits: 1,067 + 1,104 = 2,171 tables survive the admission filter AND a
stem-keyed AND a serialised-text exclusion against the member.  EDGAR is
untouched - its eval half (507 filings / 4,106 chunks) reads zero contamination.

THE REBUILD
-----------
Same generator, same seed, same target cells, same leak suite - the construction
is IMPORTED from `R20-H177_lane_B.py` rather than re-implemented.  Two things
change, both narrowing:

  1. the TabFact supply is the corpus's test+validation halves instead of the
     train half, minus every table whose STEM is in the member and every table
     whose serialised text (raw or 1,500-cut, whitespace-normalised) equals a
     member table's
  2. the split is no longer a hash of a doc_id string.  Train and eval are
     separated by CORPUS HALF - lane B is built from TabFact train, the eval from
     test+validation - which is a strictly stronger rule than the hash it
     replaces, because it cannot be defeated by two ids naming one table.  The
     stem rule is still applied on top, as the exclusion above.  EDGAR keeps the
     banked `blake2b(doc_id)` split, whose ids carry no prefix ambiguity

Out: R20-H177_eval_B_rebuilt.parquet, R20-H177_eval_B_rebuilt_manifest.json.
Verification against all eight contract clauses is a separate pass
(`R20-H177_eval_B_rebuilt_verify.py`).

Run:  uv run python experiments/grounding-semantic/R20-H177_eval_B_rebuild.py [--force]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

OUT = HERE / "R20-H177_eval_B_rebuilt.parquet"
MANIFEST = HERE / "R20-H177_eval_B_rebuilt_manifest.json"
MEMBER = HERE / "contract" / "tabfact_member.parquet"

CLEAN_SPLITS = ("test", "validation")
CUT = 1500


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod("h177laneB", HERE / "R20-H177_lane_B.py")
C = _mod("h177common", HERE / "R20-H177_lane_common.py")
P = _mod("h144pairs", HERE / "R17-H144_pairs.py")


def norm(s):
    return " ".join(s.split()).casefold()


def stem(tid):
    return tid[2:] if len(tid) > 2 and tid[0] in "12" and tid[1] == "-" else tid


def doc_stem(doc_id):
    return "tabfact:" + stem(doc_id[len("tabfact:"):]) \
        if doc_id.startswith("tabfact:") else doc_id


def member_exclusion():
    """Everything the `tabfact` mix member holds, in the three forms C2 tests."""
    m = pl.read_parquet(MEMBER)
    ids = set(m["table_id"].to_list())
    stems = {stem(t) for t in ids}
    nraw = {norm(c) for c in m["chunk_untrunc"].unique().to_list()}
    ntrunc = {norm(c) for c in m["chunk_trunc"].unique().to_list()}
    return {"ids": ids, "stems": stems, "nraw": nraw, "ntrunc": ntrunc,
            "rows": m.height}


def tables_from(split):
    """`R17-H144_pairs.tabfact_tables()` verbatim, on a chosen archive split.

    The parse is the banked one, character for character; only the file the
    `next(...)` picks differs, so a table admitted here is the same object the
    train-half loader would have produced."""
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    name = next(x for x in z.namelist() if x.endswith(f"__{split}.parquet"))
    d = pl.read_parquet(io.BytesIO(z.read(name))).unique(
        subset=["table_text"], keep="first", maintain_order=True)
    out = []
    for tid, cap, tbl in zip(d["table_id"].to_list(), d["table_caption"].to_list(),
                             d["table_text"].to_list()):
        rows = [r.split("#") for r in
                tbl.replace("\r\n", "\n").strip().split("\n") if r.strip()]
        rows = [[c.strip() for c in r] for r in rows]
        if len(rows) < 4:
            continue
        w = len(rows[0])
        body = [r for r in rows[1:] if len(r) == w]
        if len(body) < 4 or w < 2:
            continue
        hdr = [P.clean(c) for c in rows[0]]
        body = [[P.clean(c)[:80] for c in r] for r in body]
        # the member's own serialisation of this table, for the text exclusion
        member_form = f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")
        out.append({"doc_id": f"tabfact:{tid}", "source": "tabfact",
                    "caption": P.clean(cap or "table")[:120] or "table",
                    "hdr": hdr, "body": body, "archive_split": split,
                    "member_form": member_form})
    return out


def clean_tabfact_pool(excl):
    """TabFact test+validation tables with a label column and a numeric column,
    minus every table the mix member reaches by id, by stem or by text."""
    excluded_ids, prints, eval_rows, _unmatched = P.evalset_documents()
    note = {"supply": "TabFact TEST + VALIDATION (the train half is 100% inside "
                      "the `tabfact` mix member - measured, see "
                      "R20-H177_evalB_rebuild_supply.json)",
            "per_split": {}, "evalset_rows": eval_rows,
            "method": "R17-H144 content-based R17-H143_evalset exclusion, "
                      "enforced; then the mix-member exclusion by table id, by "
                      "id STEM, and by whitespace-normalised serialised text in "
                      "both the raw and the 1,500-cut form"}
    tables = []
    for split in CLEAN_SPLITS:
        raw = tables_from(split)
        drop_idx = P.excluded_tables(raw, prints)
        kept, drops = [], {"carry_R17H143_eval_content": len(drop_idx),
                           "evalset_doc_id": 0, "member_id": 0, "member_stem": 0,
                           "member_text_raw": 0, "member_text_truncated": 0,
                           "no_label_or_numeric_column": 0}
        for ti, t in enumerate(raw):
            tid = t["doc_id"][len("tabfact:"):]
            if ti in drop_idx:
                continue
            if t["doc_id"] in excluded_ids:
                drops["evalset_doc_id"] += 1
                continue
            if tid in excl["ids"]:
                drops["member_id"] += 1
                continue
            if stem(tid) in excl["stems"]:
                drops["member_stem"] += 1
                continue
            if norm(t["member_form"]) in excl["nraw"]:
                drops["member_text_raw"] += 1
                continue
            if norm(t["member_form"][:CUT]) in excl["ntrunc"]:
                drops["member_text_truncated"] += 1
                continue
            lab = P.label_column(t["hdr"], t["body"])
            if lab is None or not P.numeric_columns(t["hdr"], t["body"], lab):
                drops["no_label_or_numeric_column"] += 1
                continue
            t["lab_ci"] = lab
            kept.append(t)
        note["per_split"][split] = {"candidate_tables": len(raw),
                                    "admitted": len(kept), "dropped": drops}
        tables += kept
        print(f"  tabfact {split}: {len(raw)} candidates -> {len(kept)} admitted "
              f"{drops}", flush=True)
    note["admitted"] = len(tables)
    return tables, note


def claim_wall():
    """Every claim string a training surface already carries.

    The lane is TEMPLATE-generated, so a claim is a fixed frame filled with a
    column name, two row labels and two cells: two unrelated tables printing the
    same five strings emit the same sentence.  A first pass of this rebuild read
    zero on evidence and zero on documents but 12 rows on the claim string
    against lane B - all six pairs colliding on BOTH legs, with the lane's label
    agreeing on both, which is exactly what a claim-string memoriser carries
    across (`R20-H177_eval_B_rebuilt_claimcollide.json`).  The contract relaxes
    no clause to make an artifact pass, so the candidate is rejected at build
    time instead and the cell refills from elsewhere."""
    lane = pl.read_parquet(HERE / "R20-H177_lane_B.parquet")
    wall = set(lane["claim"].to_list())
    n_lane = len(wall)
    arm = _mod("g1arm", HERE / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", HERE / "R10-H108_lane.py")
    A = _mod("h174arm", HERE / "R20-H174_arm_run.py")
    with arm.untruncated_evidence():
        claims, _chunks, _y, _tags = H108.public_train()
    wall |= set(claims)
    for fname, *_ in A.LANES:
        wall |= set(pl.read_parquet(HERE / fname)["claim"].to_list())
    print(f"claim wall: {n_lane} distinct lane B claims, {len(wall)} with the "
          f"whole training mix", flush=True)
    return wall


def guard_builders(wall):
    """Reject a candidate whose claim - under EITHER relation word and EITHER
    template of its frame - is a string a training surface already carries."""
    rejected = collections.Counter()

    def wrap(fn, fam):
        def inner(item, rng, word):
            spec = fn(item, rng, word)
            if spec is None:
                return None
            kw = dict(col=spec["col"], ka=spec["ka"], va=spec["va"],
                      kb=spec["kb"], vb=spec["vb"])
            for ti in range(len(B.TEMPLATES[(fam, B.PAIR_OF[word])])):
                for w in (word, B.FLIP[word]):
                    if B.render(fam, ti, w, **kw) in wall:
                        rejected[fam] += 1
                        return None
            return spec
        return inner

    B.BUILDERS = {k: wrap(v, k) for k, v in B.BUILDERS.items()}
    return rejected


def already_built():
    if "--force" in sys.argv or not (OUT.exists() and MANIFEST.exists()):
        return False
    try:
        man = json.loads(MANIFEST.read_text())
        rows = pl.read_parquet(OUT).height
    except Exception:
        return False
    if rows == man.get("rows") and "verify" in man:
        print(f"{OUT.name}: {rows} rows already built - skipping (pass --force)",
              flush=True)
        return True
    return False


def main():
    if already_built():
        return
    t0 = time.time()
    print("=== R20-H177 eval_B REBUILD (CPU only) ===", flush=True)

    excl = member_exclusion()
    print(f"member exclusion: {excl['rows']} rows, {len(excl['ids'])} ids, "
          f"{len(excl['stems'])} stems, {len(excl['nraw'])} normalised texts",
          flush=True)

    tf, tf_note = clean_tabfact_pool(excl)
    B.assign_forms(tf, np.random.default_rng(B.SEED_EVAL))
    print(f"clean tabfact supply: {len(tf)} tables", flush=True)

    rejected = guard_builders(claim_wall())

    ed_raw = C.edgar("eval")
    ed = B.edgar_items(ed_raw, "eval")
    print(f"edgar eval supply: {ed_raw.height} chunks / "
          f"{ed_raw['doc_id'].n_unique()} filings -> {len(ed)} usable items",
          flush=True)

    ev, cells = B.assemble("eval", B.SEED_EVAL, B.EVAL_TARGETS, tf, ed)
    ev.write_parquet(OUT)
    print(f"{ev.height} rows / {ev['pair_id'].n_unique()} pairs -> {OUT.name}",
          flush=True)

    res = B.verify(ev, random.Random(B.SEED_EVAL))

    lane = pl.read_parquet(HERE / "R20-H177_lane_B.parquet")
    lane_docs = set(lane["doc_id"].to_list())
    ev_docs = set(ev["doc_id"].to_list())
    lane_stems = {doc_stem(d) for d in lane_docs}
    ev_stems = {doc_stem(d) for d in ev_docs}
    lane_chunks = set(lane["chunk"].to_list())
    ev_chunks = set(ev["chunk"].to_list())

    man = B.block(ev, res, cells, "eval", B.SEED_EVAL, tf_note, extra=dict(
        experiment="R20-H177 eval_B REBUILD - held-out mechanism eval for lane B, "
                   "rebuilt on supply disjoint from the training mix",
        scope="measurement and building only - no adjudication, no bar changed",
        replaces=dict(
            parquet="R20-H177_eval_B.parquet",
            defects=["passage channel: 33 of 736 passages in the flagship mix "
                     "(19 byte-identical), 126 rows / 63 pairs",
                     "document channel: 325 of 325 TabFact documents in the "
                     "`tabfact` mix member, 1,300 of 2,000 rows (65%)",
                     "split key: blake2b over the doc_id STRING, which TabFact's "
                     "`1-`/`2-` csv prefixes make two ids for one table"]),
        construction="imported verbatim from R20-H177_lane_B.py - BUILDERS, "
                     "assemble, verify, emit, EVAL_TARGETS, SEED_EVAL; only the "
                     "TabFact supply and the split rule differ",
        split_rule=dict(
            tabfact="CORPUS HALF - lane B is TabFact TRAIN, this eval is TabFact "
                    "TEST+VALIDATION; strictly stronger than the blake2b hash it "
                    "replaces, which keyed on a doc_id string that names one "
                    "table under two prefixes",
            tabfact_exclusion="table id, id STEM, and whitespace-normalised "
                              "serialised text (raw and 1,500-cut) against the "
                              "`tabfact` mix member",
            edgar=f"the banked blake2b(doc_id) % 1000 < {C.EVAL_DOC_PERMILLE} "
                  "rule, unchanged - EDGAR ids carry no prefix ambiguity",
            claim_wall="a candidate is rejected if its claim, under EITHER "
                       "relation word and EITHER template of its frame, is a "
                       "string lane B or the training mix already carries - the "
                       "lane is template-generated, so two unrelated documents "
                       "printing the same column name, row labels and cells emit "
                       "the same sentence"),
        claim_wall_rejections={k: int(v) for k, v in rejected.items()},
        supply_frontier="R20-H177_evalB_rebuild_supply.json - 0 of 8,541 TabFact "
                        "TRAIN table stems survive the member exclusion, so a "
                        "train-half rebuild has no TabFact supply at all",
        reproducibility=dict(
            requires="PYTHONHASHSEED=0",
            finding="the banked `R17-H144_pairs.excluded_tables` ranks a chunk's "
                    "numerals by `sorted(..., key=len(index[v]))` over a FROZENSET, "
                    "so ties are broken by the set's iteration order, which Python "
                    "salts per process. Two runs of the identical code excluded 373 "
                    "and 372 validation tables. The build is deterministic only "
                    "with the hash seed pinned; the banked parquet is the record "
                    "either way",
            scope="a property of banked code that lane B and R17-H146 also run - "
                  "recorded, not repaired here"),
        sources=C.SOURCES, walled_never_opened=C.WALLED_NEVER_OPENED,
        parquet=OUT.name,
        disjointness_from_lane_B=dict(
            shared_doc_ids=len(lane_docs & ev_docs),
            shared_doc_id_STEMS=len(lane_stems & ev_stems),
            shared_chunks=len(lane_chunks & ev_chunks),
            lane_documents=len(lane_docs), eval_documents=len(ev_docs),
            bar="0 on every channel", pass_=not (lane_docs & ev_docs)
            and not (lane_stems & ev_stems) and not (lane_chunks & ev_chunks)),
        elapsed_s=round(time.time() - t0, 1),
        note="Numbers recorded, not adjudicated - the coordinator adjudicates."))
    MANIFEST.write_text(json.dumps(man, indent=2))

    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "families", "cells_filled",
                       "source_rows", "serial_forms",
                       "disjointness_from_lane_B")}, indent=2), flush=True)
    print(json.dumps(man["verify"], indent=2)[:3000], flush=True)
    print(f"=== REBUILT {'ALL BARS PASS' if res['all_bars_pass'] else 'BARS FAILED'} "
          f"({man['elapsed_s']}s) ===", flush=True)


if __name__ == "__main__":
    main()
