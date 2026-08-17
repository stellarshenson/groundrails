"""R20-H177 eval_B rebuild - SUPPLY FRONTIER probe.  CPU only, Polars only.

Answers, before anything is built:
  * how much TabFact TRAIN supply survives a STEM-keyed exclusion against the
    `tabfact` mix member (the member is the whole TabFact train split, so the
    expected answer is zero and the number is measured rather than assumed)
  * how much TabFact TEST+VALIDATION supply survives the same stem exclusion AND
    a table_text-identity exclusion against the member
  * how much EDGAR eval-split supply exists under the banked split rule, and
    under a stem-keyed variant of it

Writes R20-H177_evalB_rebuild_supply.json.  Builds nothing.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util as _ilu
import io
import json
import pathlib
import time
import zipfile

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
MEMBER = HERE / "contract" / "tabfact_member.parquet"
OUT = HERE / "R20-H177_evalB_rebuild_supply.json"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h177common", HERE / "R20-H177_lane_common.py")
P = _mod("h144pairs", HERE / "R17-H144_pairs.py")


def stem(tid):
    """TabFact writes one Wikipedia table under both a `1-` and a `2-` csv id."""
    return tid[2:] if len(tid) > 2 and tid[0] in "12" and tid[1] == "-" else tid


def norm(s):
    return " ".join(s.split()).casefold()


def tables_from(split):
    """`R17-H144_pairs.tabfact_tables()` verbatim, but on a chosen split."""
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
        out.append({"doc_id": f"tabfact:{tid}", "source": "tabfact",
                    "caption": P.clean(cap or "table")[:120] or "table",
                    "hdr": hdr, "body": body, "split": split,
                    "table_text_raw": tbl})
    return out


def usable(tables):
    """The lane's own admission filter - a label column and a numeric column."""
    out = []
    for t in tables:
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None or not P.numeric_columns(t["hdr"], t["body"], lab):
            continue
        t = dict(t)
        t["lab_ci"] = lab
        out.append(t)
    return out


def main():
    t0 = time.time()
    res = {"experiment": "R20-H177 eval_B rebuild - supply frontier",
           "scope": "measurement only"}

    member = pl.read_parquet(MEMBER)
    m_tid = set(member["table_id"].to_list())
    m_stem = {stem(t) for t in m_tid}
    # the member's own table_text, reconstructed from the archive so a
    # content-identity test is available alongside the id test
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    tr_raw = pl.read_parquet(io.BytesIO(
        z.read("wenhuchen__Table-Fact-Checking__tabfact__train.parquet")))
    m_text_norm = {norm(x) for x in
                   tr_raw.filter(pl.col("table_id").is_in(list(m_tid)))
                   ["table_text"].unique().to_list()}
    res["member"] = {"rows": member.height, "distinct_table_id": len(m_tid),
                     "distinct_table_id_stem": len(m_stem),
                     "distinct_table_text_normalised": len(m_text_norm)}
    print(f"member: {member.height} rows / {len(m_tid)} ids / {len(m_stem)} stems",
          flush=True)

    # ---- TabFact TRAIN, the supply the banked eval_B was drawn from -------- #
    tf_train = usable(tables_from("train"))
    tr_ids = {t["doc_id"][len("tabfact:"):] for t in tf_train}
    tr_stems = {stem(i) for i in tr_ids}
    res["tabfact_train"] = {
        "usable_tables": len(tf_train),
        "distinct_ids": len(tr_ids), "distinct_stems": len(tr_stems),
        "ids_in_member": len(tr_ids & m_tid),
        "stems_in_member": len(tr_stems & m_stem),
        "stems_NOT_in_member": len(tr_stems - m_stem),
        "eval_half_under_banked_split": sum(
            1 for t in tf_train if C.is_eval_doc(t["doc_id"])),
        "eval_half_under_stem_split": sum(
            1 for t in tf_train
            if C.is_eval_doc("tabfact:" + stem(t["doc_id"][len("tabfact:"):]))),
        "clean_eval_supply_after_stem_exclusion": sum(
            1 for t in tf_train
            if stem(t["doc_id"][len("tabfact:"):]) not in m_stem),
    }
    print(f"tabfact train usable {len(tf_train)}; stems not in member "
          f"{len(tr_stems - m_stem)}", flush=True)

    # ---- TabFact TEST + VALIDATION, the only other TabFact supply on disk -- #
    for split in ("test", "validation"):
        tabs = usable(tables_from(split))
        ids = {t["doc_id"][len("tabfact:"):] for t in tabs}
        stems = {stem(i) for i in ids}
        clean = [t for t in tabs
                 if stem(t["doc_id"][len("tabfact:"):]) not in m_stem
                 and norm(t["table_text_raw"]) not in m_text_norm]
        res[f"tabfact_{split}"] = {
            "usable_tables": len(tabs),
            "distinct_ids": len(ids), "distinct_stems": len(stems),
            "ids_in_member": len(ids & m_tid),
            "stems_in_member": len(stems & m_stem),
            "table_text_identical_to_a_member_table": sum(
                1 for t in tabs if norm(t["table_text_raw"]) in m_text_norm),
            "clean_after_stem_AND_text_exclusion": len(clean),
            "clean_eval_half_under_stem_split": sum(
                1 for t in clean
                if C.is_eval_doc("tabfact:" + stem(t["doc_id"][len("tabfact:"):]))),
        }
        print(f"tabfact {split}: usable {len(tabs)}, clean {len(clean)}", flush=True)

    # ---- EDGAR --------------------------------------------------------------#
    ed = pl.read_parquet(C.EDGAR)
    ed_docs = ed["doc_id"].unique().to_list()
    ed_eval_docs = [d for d in ed_docs if C.is_eval_doc(d)]
    res["edgar"] = {
        "chunks": ed.height, "documents": len(ed_docs),
        "eval_documents_under_banked_split": len(ed_eval_docs),
        "eval_chunks_under_banked_split": int(
            ed.filter(pl.col("doc_id").is_in(ed_eval_docs)).height),
        "note": "EDGAR doc ids carry no `1-`/`2-` prefix ambiguity, so the stem "
                "rule is the identity on this half and the banked split stands",
    }
    print(f"edgar: {ed.height} chunks / {len(ed_docs)} docs, eval half "
          f"{res['edgar']['eval_chunks_under_banked_split']} chunks", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    res["note"] = "Numbers recorded, not adjudicated - the coordinator adjudicates."
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
