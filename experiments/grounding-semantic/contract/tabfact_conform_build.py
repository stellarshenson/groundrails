"""Build the CONFORMING `tabfact` training member.

CPU only, Polars only, torch-free. The member is rebuilt through the BANKED
loader (`R10-H108_lane.public_train()` under `R16-H142_G1_arm.untruncated_evidence()`,
reached through the arm so the context manager patches the loader's own M59),
asserted row-for-row against the archive, and then CUT. Nothing is
re-implemented and no chunk text is rewritten - the conformed member is a strict
ROW SUBSET of the banked member.

Four cuts, each traced to the clause it repairs:

  X1  DOCUMENT cut on the STEM, held-out splits          (C3, and C2's antigaming
      TabFact writes one Wikipedia table under both a     channel, which is built
      `1-` and a `2-` csv id, so the archive's own        from those splits under
      split is disjoint on the id STRING and not on       an EXACT-id rule)
      the document. Every member table whose stem
      occurs in TabFact validation or test is dropped.
  X2  DOCUMENT cut, R20-H177_eval_B's TabFact half      (C2)
      325 documents, all of them member documents.
  X3  DOCUMENT cut, R17-H143_evalset's TabFact half     (C2)
      The banked evalset parquet carries no doc_id; the ids are recovered by
      joining its passages to `R17-H143_evalset_source.parquet`, which does.
  X4  DOCUMENT cut, every anti-gaming probe set's tables (C2)
      (a subset of X1 - the sets are built from validation+test - measured and
      reported rather than assumed).

  S   STRING residual - any surviving member evidence identical to an eval
      passage in any of the three forms, in either direction.               (C2)
  N   CONTENT residual - any surviving member table at 8-gram Jaccard >= 0.3
      against any TabFact table an evaluation surface draws on. The banked
      R14-H136 gate primitives are used unchanged.                      (C2, C3)
  D   CONTRADICTION cut - a claim carrying BOTH labels on the same DOCUMENT
      (stem) is supervision the head cannot satisfy.                        (C8)

Out: tabfact_member_conformed.parquet
     tabfact_conform_build.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPUs 0/1/2 are training R20-H174
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
OUT_PARQUET = HERE / "tabfact_member_conformed.parquet"
OUT_JSON = HERE / "tabfact_conform_build.json"

EXPECTED_CLEAN_ROWS = 685_670   # R18-H150_arm_run.EXPECTED_CLEAN_ROWS
N = 8
JACCARD = 0.3
CUT = 1500


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    return " ".join(s.split()).casefold()


def stem(tid):
    return tid[2:] if len(tid) > 2 and tid[0] in "12" and tid[1] == "-" else tid


def build_chunk(cap, tbl):
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")


# --------------------------------------------------------------------------- #
def load_member():
    """The banked loader, reached through the arm (see tabfact_load.py)."""
    arm = _mod("g1arm", SEM / "R16-H142_G1_arm.py")
    H108 = arm.H108
    chunk_max = H108.M59.CFG.chunk_max_chars
    with arm.untruncated_evidence():
        if H108.M59.CFG.chunk_max_chars != 10**9:
            raise SystemExit("UNTRUNCATED ABORT: the cut was not lifted on the loader's own M59")
        claims, chunks, y, tags = H108.public_train()
    if len(y) != EXPECTED_CLEAN_ROWS:
        raise SystemExit(f"MIX ABORT: clean mix {len(y)} rows, expected {EXPECTED_CLEAN_ROWS}")

    idx = [i for i, t in enumerate(tags) if t == "tabfact"]
    m_claims = [claims[i] for i in idx]
    m_chunks = [chunks[i] for i in idx]
    m_y = np.asarray(y, dtype="float32")[idx]
    del claims, chunks, tags, y

    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    train_name = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    raw = pl.read_parquet(io.BytesIO(z.read(train_name)))
    sel = raw.filter(pl.col("statement").str.len_chars() > 10)
    if sel.height != len(m_claims):
        raise SystemExit(f"ALIGN ABORT: archive selection {sel.height} != member {len(m_claims)}")
    if sel["statement"].to_list() != m_claims:
        raise SystemExit("ALIGN ABORT: member claims are not the archive statements in order")
    if not np.array_equal(m_y, sel["label"].cast(pl.Float32).to_numpy()):
        raise SystemExit("ALIGN ABORT: member labels are not the archive labels in order")

    df = pl.DataFrame({
        "claim": m_claims,
        "chunk_untrunc": m_chunks,
        "chunk_trunc": [c[:chunk_max] for c in m_chunks],
        "label": m_y,
        "table_id": sel["table_id"].to_list(),
        "table_caption": sel["table_caption"].to_list(),
    })
    return df, chunk_max, train_name, raw.height


def heldout_tables():
    """TabFact validation + test, table_id -> serialised evidence."""
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    out = {}
    for split in ("validation", "test"):
        n = next(x for x in z.namelist() if x.endswith(f"__{split}.parquet"))
        d = pl.read_parquet(io.BytesIO(z.read(n))).unique(subset=["table_id"], keep="first")
        for t, c, b in zip(d["table_id"].to_list(), d["table_caption"].to_list(),
                           d["table_text"].to_list(), strict=True):
            out[t] = build_chunk(c, b)
    return out


def evalB_tabfact_stems():
    d = pl.read_parquet(SEM / "R20-H177_eval_B.parquet")
    ids = [x for x in d["doc_id"].unique().to_list() if str(x).startswith("tabfact:")]
    return {stem(x[len("tabfact:"):]) for x in ids}, len(ids)


def h143_tabfact_stems():
    """The banked R17-H143 evalset carries no doc_id; recover it from the source
    pool it was drawn from, joining on the passage string."""
    ev = pl.read_parquet(SEM / "R17-H143_evalset.parquet")
    src = pl.read_parquet(SEM / "R17-H143_evalset_source.parquet")
    j = (ev.select("chunk").unique()
         .join(src.select("chunk", "doc_id", "source").unique(subset=["chunk"]),
               on="chunk", how="left"))
    if j["doc_id"].null_count():
        raise SystemExit("H143 ABORT: evalset passages not all resolvable to a source doc_id")
    tf = j.filter(pl.col("source") == "tabfact")
    return ({stem(x[len("tabfact:"):]) for x in tf["doc_id"].to_list()},
            tf["doc_id"].n_unique(), j.height)


def antigaming_stems():
    per, union = {}, set()
    for path in sorted(SEM.glob("*antigaming_set.parquet")):
        d = pl.read_parquet(path)
        if "table_id" not in d.columns:
            continue
        s = {stem(t) for t in d["table_id"].to_list()}
        per[path.name] = len(s)
        union |= s
    return union, per


def eval_surface_passages():
    """Every held-out mechanism eval's passage strings, for the STRING residual."""
    texts = []
    for f in ("R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet",
              "R17-H143_evalset.parquet", "R20-H175b_qlane_eval.parquet",
              "R20-H175b_qlane_eval_repaired.parquet", "R20-H175b_qlane_eval_clean.parquet",
              "R20-H175b_qlane_eval_clean_prefix.parquet", "R19_findver_lane.parquet"):
        p = SEM / f
        if p.exists():
            texts += pl.read_parquet(p)["chunk"].to_list()
    return texts


def max_jaccard_per_unit(query_texts, ref_texts, n=N):
    """Each query unit's best 8-gram Jaccard against any single reference unit.
    Uses the banked provenance_gate primitives unchanged."""
    G = _mod("pgate", SEM / "provenance_gate.py")
    hasher = G._TokenHasher()
    ref = G._Side("ref")
    for t in ref_texts:
        ref.add("ref", G.ngram_hashes(t, n, hasher))
    idx = ref.index()
    if "ref" not in idx:
        return np.zeros(len(query_texts))
    h, owner, _ = idx["ref"]
    sizes = np.array([u.size for u in ref.buckets["ref"]], dtype=np.int64)
    out = np.zeros(len(query_texts))
    for i, t in enumerate(query_texts):
        q = G.ngram_hashes(t, n, hasher)
        j, _uid = G._max_jaccard(q, h, owner, sizes)
        out[i] = j
    return out


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    df, chunk_max, train_name, archive_rows = load_member()
    print(f"banked member: {df.height} rows", flush=True)

    tid = df["table_id"].to_list()
    st = [stem(t) for t in tid]
    df = df.with_columns(pl.Series("stem", st))
    member_stems = set(st)

    # ---- exclusion sets ---------------------------------------------------- #
    ho_tabs = heldout_tables()
    X1 = {stem(t) for t in ho_tabs}
    X2, evalB_docs = evalB_tabfact_stems()
    X3, h143_docs, h143_passages = h143_tabfact_stems()
    X4, ag_per = antigaming_stems()
    X = X1 | X2 | X3 | X4

    cuts = {
        "X1_tabfact_heldout_splits": {
            "clause": "C3 (and C2 via the anti-gaming sets)",
            "rule": "drop every member table whose table_id STEM occurs in TabFact "
                    "validation or test",
            "stems_in_the_surface": len(X1),
            "stems_hit_in_member": len(X1 & member_stems),
        },
        "X2_R20-H177_eval_B_tabfact_half": {
            "clause": "C2",
            "rule": "drop every member table whose STEM is a doc_id eval_B draws on",
            "surface_tabfact_doc_ids": evalB_docs,
            "stems_in_the_surface": len(X2),
            "stems_hit_in_member": len(X2 & member_stems),
        },
        "X3_R17-H143_evalset_tabfact_half": {
            "clause": "C2",
            "rule": "same document rule; the evalset's doc_ids are recovered by "
                    "joining its 547 distinct passages to R17-H143_evalset_source",
            "evalset_distinct_passages_resolved": h143_passages,
            "surface_tabfact_doc_ids": h143_docs,
            "stems_in_the_surface": len(X3),
            "stems_hit_in_member": len(X3 & member_stems),
        },
        "X4_antigaming_probe_sets": {
            "clause": "C2",
            "rule": "drop every member table whose STEM occurs in any of the 14 "
                    "banked anti-gaming sets",
            "files": len(ag_per),
            "stems_in_the_union": len(X4),
            "stems_hit_in_member": len(X4 & member_stems),
            "stems_already_covered_by_X1": len(X4 - X1) == 0,
            "per_file_distinct_stems": ag_per,
        },
    }
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_file_distinct_stems"}
                      for k, v in cuts.items()}, indent=1), flush=True)

    keep = df.filter(~pl.col("stem").is_in(list(X)))
    after_docs = keep.height
    print(f"after document cuts: {after_docs} rows "
          f"({keep['stem'].n_unique()} stems)", flush=True)

    # ---- S: string residual against every eval passage ---------------------- #
    ev_texts = eval_surface_passages()
    ev_raw = {t for t in ev_texts if t}
    ev_tru = {t[:CUT] for t in ev_raw}
    ev_nraw = {norm(t) for t in ev_raw}
    ev_ntru = {norm(t) for t in ev_tru}
    ch = keep["chunk_untrunc"].to_list()
    bad_str = set()
    for c, s in zip(ch, keep["stem"].to_list(), strict=True):
        if (c in ev_raw or c in ev_tru or c[:CUT] in ev_raw or c[:CUT] in ev_tru
                or norm(c) in ev_nraw or norm(c[:CUT]) in ev_ntru):
            bad_str.add(s)
    cuts["S_string_residual"] = {
        "clause": "C2",
        "rule": "drop any surviving member table whose evidence equals an eval "
                "passage in any of the three forms (raw / truncated 1500 / "
                "whitespace-collapsed case-folded), either direction",
        "eval_passages_compared": len(ev_raw),
        "member_stems_hit": len(bad_str),
    }
    if bad_str:
        keep = keep.filter(~pl.col("stem").is_in(list(bad_str)))
    print(f"after string residual: {keep.height} rows", flush=True)

    # ---- N: content residual against every TabFact document an eval draws on - #
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    all_tabs = {}
    for split in ("train", "validation", "test"):
        n = next(x for x in z.namelist() if x.endswith(f"__{split}.parquet"))
        d = pl.read_parquet(io.BytesIO(z.read(n))).unique(subset=["table_id"], keep="first")
        for t, c, b in zip(d["table_id"].to_list(), d["table_caption"].to_list(),
                           d["table_text"].to_list(), strict=True):
            all_tabs[t] = build_chunk(c, b)
    eval_side = sorted({txt for t, txt in all_tabs.items() if stem(t) in X})
    print(f"content residual: {len(eval_side)} eval-side TabFact tables", flush=True)

    n_rounds = []
    for rnd in range(3):
        # scored per table_id, NOT per stem: the `1-`/`2-` twins of one document
        # carry different serialisations, so a stem representative would leave
        # its twin's text unscored. A hit on either twin drops the document.
        cand = (keep.unique(subset=["table_id"], keep="first")
                .select("stem", "chunk_untrunc"))
        j = max_jaccard_per_unit(cand["chunk_untrunc"].to_list(), eval_side)
        hit = sorted({s for s, v in zip(cand["stem"].to_list(), j, strict=True)
                      if v >= JACCARD})
        n_rounds.append({"round": rnd + 1, "member_tables_scored": cand.height,
                         "documents_at_jaccard_ge_0.3": len(hit),
                         "max_jaccard": round(float(j.max()), 4),
                         "p99_jaccard": round(float(np.percentile(j, 99)), 4),
                         "mean_jaccard": round(float(j.mean()), 4)})
        print(json.dumps(n_rounds[-1]), flush=True)
        if not hit:
            break
        keep = keep.filter(~pl.col("stem").is_in(hit))
    cuts["N_content_residual"] = {
        "clause": "C2 / C3",
        "rule": f"drop any surviving member table at {N}-gram Jaccard >= {JACCARD} "
                "against any TabFact table an evaluation surface draws on "
                "(banked R14-H136 gate primitives, unchanged)",
        "eval_side_tables": len(eval_side),
        "rounds": n_rounds,
    }
    print(f"after content residual: {keep.height} rows", flush=True)

    # ---- D: contradictory supervision -------------------------------------- #
    lab = collections.defaultdict(set)
    for c, s, y in zip(keep["claim"].to_list(), keep["stem"].to_list(),
                       keep["label"].to_list(), strict=True):
        lab[(c, s)].add(float(y))
    bad_pairs = {k for k, v in lab.items() if len(v) > 1}
    if bad_pairs:
        mask = [(c, s) not in bad_pairs for c, s in
                zip(keep["claim"].to_list(), keep["stem"].to_list(), strict=True)]
        rows_before = keep.height
        keep = keep.filter(pl.Series(mask))
        dropped_d = rows_before - keep.height
    else:
        dropped_d = 0
    cuts["D_contradictory_supervision"] = {
        "clause": "C8",
        "rule": "drop every row of a (claim, document) group carrying BOTH labels - "
                "the document key is the STEM, so the `1-`/`2-` twin of one table "
                "counts as one document",
        "claim_document_groups_dropped": len(bad_pairs),
        "rows_dropped": dropped_d,
    }
    print(f"after contradiction cut: {keep.height} rows", flush=True)

    keep = keep.drop("stem")
    keep.write_parquet(OUT_PARQUET)

    meta = {
        "member": "tabfact_conformed",
        "built_from": "the banked member, rebuilt through R10-H108_lane.public_train() "
                      "under R16-H142_G1_arm.untruncated_evidence() and asserted "
                      "row-for-row against the archive; the conformed member is a "
                      "strict ROW SUBSET - no chunk text is rewritten",
        "archive": "dataset-tabfact.zip :: " + train_name,
        "archive_train_rows": archive_rows,
        "selection_predicate": "train split only; filter statement.str.len_chars() > 10; "
                               "then the document / string / content / contradiction "
                               "cuts recorded under `cuts`",
        "chunk_construction": "f'{table_caption}\\n{table_text}'"
                              ".replace('\\r\\n','\\n').replace('#',' | ')[:chunk_max_chars]",
        "chunk_max_chars": chunk_max,
        "cuts": cuts,
        "volume": {
            "banked_member_rows": df.height,
            "banked_member_tables": df["table_id"].n_unique(),
            "banked_member_documents_stems": len(member_stems),
            "conformed_rows": keep.height,
            "conformed_tables": keep["table_id"].n_unique(),
            "conformed_documents_stems": keep["table_id"].map_elements(
                stem, return_dtype=pl.Utf8).n_unique(),
            "rows_dropped": df.height - keep.height,
            "row_cost_share": round((df.height - keep.height) / df.height, 6),
            "documents_dropped": len(member_stems) - keep["table_id"].map_elements(
                stem, return_dtype=pl.Utf8).n_unique(),
            "clean_mix_rows_before": EXPECTED_CLEAN_ROWS,
            "clean_mix_rows_after": EXPECTED_CLEAN_ROWS - (df.height - keep.height),
            "member_share_before": round(df.height / EXPECTED_CLEAN_ROWS, 6),
            "member_share_after": round(
                keep.height / (EXPECTED_CLEAN_ROWS - (df.height - keep.height)), 6),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["volume"], indent=2), flush=True)
    print(f"-> {OUT_PARQUET.name}  ({meta['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
