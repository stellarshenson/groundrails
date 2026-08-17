"""Contract clause C2 re-verified on the CONFORMED `tabfact` member.

CPU only, Polars only, torch-free. Same instrument as the first pass
(`tabfact_c2.py`): three string forms - raw, truncated to `CFG.chunk_max_chars`
= 1,500, and whitespace-collapsed case-folded - crossed six ways, run in BOTH
directions, on EVIDENCE and CLAIM units, against every evaluation surface.

Three things are measured here that the first pass did not:

  * R17-H143_evalset's DOCUMENT channel. Its banked parquet carries no doc_id,
    so the first pass could only read its passage strings (10 hits). The ids are
    recovered here by joining its passages to `R17-H143_evalset_source.parquet`.
  * `R20-H175b_qlane_eval_clean_prefix` - a surface the first pass did not list.
  * a CONTENT read alongside the string forms: every surviving member table's
    best 8-gram Jaccard against any TabFact table an evaluation surface draws
    on, using the banked R14-H136 gate primitives. String identity is the
    clause's bar; the content read is reported so a near-duplicate that is not
    byte-identical cannot hide behind it.

Out: tabfact_conformed_c2.json
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

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
MEMBER = HERE / "tabfact_member_conformed.parquet"
OUT = HERE / "tabfact_conformed_c2.json"
CUT = 1500
N = 8
JACCARD = 0.3


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


def forms(texts):
    raw = {t for t in texts if t}
    trunc = {t[:CUT] for t in raw}
    return {"raw": raw, "trunc": trunc,
            "nraw": {norm(t) for t in raw}, "ntrunc": {norm(t) for t in trunc}}


def six_forms(query_texts, target):
    qs = sorted({t for t in query_texts if t})
    tests = (
        ("raw_in_raw", lambda p: p in target["raw"]),
        ("raw_in_truncated", lambda p: p in target["trunc"]),
        ("truncated_in_raw", lambda p: p[:CUT] in target["raw"]),
        ("truncated_in_truncated", lambda p: p[:CUT] in target["trunc"]),
        ("normalised_in_normalised_raw", lambda p: norm(p) in target["nraw"]),
        ("normalised_in_normalised_truncated", lambda p: norm(p[:CUT]) in target["ntrunc"]),
    )
    counts = {"n_query_units": len(qs)}
    hit = set()
    for name, test in tests:
        h = {p for p in qs if test(p)}
        counts[name] = len(h)
        hit |= h
    counts["any_form"] = len(hit)
    return counts


def both_directions(member_forms, surface_texts, member_texts):
    s_forms = forms(surface_texts)
    return {
        "surface_units_into_member": six_forms(surface_texts, member_forms),
        "member_units_into_surface": six_forms(member_texts, s_forms),
    }


def load_arena():
    G = _mod("pgate", SEM / "provenance_gate.py")
    ref_docs, _ = G.load_arena()
    z = zipfile.ZipFile(G.ARCHIVE)
    docs, resp = {}, {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0))
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(G.N_PER_SUBSET, len(df)), seed=0)
        docs[sub] = [c for d in df["documents"].to_list() for c in d[:G.MAX_CHUNKS]]
        resp[sub] = df["response"].to_list()
    if {k: len(v) for k, v in docs.items()} != {k: len(v) for k, v in ref_docs.items()}:
        raise SystemExit("ARENA ABORT: local arena sample differs from provenance_gate")
    for k in docs:
        if docs[k] != ref_docs[k]:
            raise SystemExit(f"ARENA ABORT: subset {k} chunks differ from provenance_gate")
    return docs, resp


def h143_doc_ids():
    ev = pl.read_parquet(SEM / "R17-H143_evalset.parquet")
    src = pl.read_parquet(SEM / "R17-H143_evalset_source.parquet")
    j = (ev.select("chunk").unique()
         .join(src.select("chunk", "doc_id", "source").unique(subset=["chunk"]),
               on="chunk", how="left"))
    if j["doc_id"].null_count():
        raise SystemExit("H143 ABORT: evalset passages not all resolvable to a doc_id")
    tf = j.filter(pl.col("source") == "tabfact")
    return {stem(x[len("tabfact:"):]) for x in tf["doc_id"].to_list()}, tf.height


def max_jaccard_per_unit(query_texts, ref_texts, n=N):
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
        out[i] = G._max_jaccard(G.ngram_hashes(t, n, hasher), h, owner, sizes)[0]
    return out


def main():
    t0 = time.time()
    df = pl.read_parquet(MEMBER)
    m_claims = df["claim"].to_list()
    m_chunks = df["chunk_untrunc"].to_list()
    m_tids = df["table_id"].to_list()
    m_stems = {stem(t) for t in m_tids}
    m_tidset = set(m_tids)
    mf_chunk = forms(m_chunks)
    mf_claim = forms(m_claims)
    print(f"conformed member: {df.height} rows, {len(mf_chunk['raw'])} distinct evidence, "
          f"{len(mf_claim['raw'])} distinct claims, {len(m_tidset)} tables, "
          f"{len(m_stems)} documents", flush=True)

    surfaces = {}

    docs, resp = load_arena()
    flat_docs = [c for v in docs.values() for c in v]
    flat_resp = [c for v in resp.values() for c in v]
    surfaces["arena_ragbench_10_subsets"] = {
        "kind": "blind arena - the 10 RAGBench subsets, banked H77 sample "
                "(250 rows/subset seed 0, first 8 document chunks)",
        "subsets": {k: len(v) for k, v in docs.items()},
        "evidence": both_directions(mf_chunk, flat_docs, m_chunks),
        "claim_vs_arena_response": both_directions(mf_claim, flat_resp, m_claims),
        "document_id_channel": "the arena parquet exposes no table id - no "
                               "document-level join is computable; the n-gram census "
                               "under C4 covers that direction",
    }
    print("arena done", flush=True)

    arm = _mod("g1arm", SEM / "R16-H142_G1_arm.py")
    gc, gk, _gy = arm.H108.gold_full()
    gold_chunks = [c for ks in gk for c in ks]
    surfaces["gold_full"] = {
        "kind": "the held-out gold test surface (R10-H108_lane.gold_full)",
        "claims": len(gc), "chunks": len(gold_chunks),
        "evidence": both_directions(mf_chunk, gold_chunks, m_chunks),
        "claim": both_directions(mf_claim, gc, m_claims),
        "document_id_channel": "gold_full is assembled from R10-H108_pairs.parquet "
                               "and carries no TabFact document id - not computable",
    }
    print("gold_full done", flush=True)

    evals = {
        "R20-H177_eval_B": SEM / "R20-H177_eval_B.parquet",
        "R20-H177_eval_C": SEM / "R20-H177_eval_C.parquet",
        "R17-H143_evalset": SEM / "R17-H143_evalset.parquet",
        "R20-H175b_qlane_eval": SEM / "R20-H175b_qlane_eval.parquet",
        "R20-H175b_qlane_eval_repaired": SEM / "R20-H175b_qlane_eval_repaired.parquet",
        "R20-H175b_qlane_eval_clean": SEM / "R20-H175b_qlane_eval_clean.parquet",
        "R20-H175b_qlane_eval_clean_prefix": SEM / "R20-H175b_qlane_eval_clean_prefix.parquet",
        "R19_findver_lane": SEM / "R19_findver_lane.parquet",
    }
    for name, path in evals.items():
        if not path.exists():
            surfaces[name] = {"present": False}
            continue
        d = pl.read_parquet(path)
        blk = {"kind": "held-out mechanism eval", "rows": d.height,
               "evidence": both_directions(mf_chunk, d["chunk"].to_list(), m_chunks),
               "claim": both_directions(mf_claim, d["claim"].to_list(), m_claims)}
        if "doc_id" in d.columns:
            tf = [x for x in d["doc_id"].unique().to_list() if str(x).startswith("tabfact:")]
            ids = {x[len("tabfact:"):] for x in tf}
            blk["document_id_channel"] = {
                "surface_tabfact_doc_ids": len(ids),
                "exact_table_id_in_member": len(ids & m_tidset),
                "STEM_table_id_in_member": len({stem(i) for i in ids} & m_stems),
            }
        elif name == "R17-H143_evalset":
            hs, npass = h143_doc_ids()
            blk["document_id_channel"] = {
                "recovered_via": "join of the evalset's distinct passages to "
                                 "R17-H143_evalset_source.parquet, which carries doc_id",
                "surface_tabfact_passages": npass,
                "surface_tabfact_documents_stems": len(hs),
                "STEM_table_id_in_member": len(hs & m_stems),
            }
        surfaces[name] = blk
        print(f"{name} done", flush=True)

    ag = {}
    for path in sorted(SEM.glob("*antigaming_set.parquet")):
        d = pl.read_parquet(path)
        blk = {"rows": d.height}
        if "table_id" in d.columns:
            ids = set(d["table_id"].to_list())
            blk |= {"distinct_table_id": len(ids),
                    "exact_table_id_in_member": len(ids & m_tidset),
                    "STEM_table_id_in_member": len({stem(i) for i in ids} & m_stems)}
        cl = [c for col in ("claim_pos", "claim_neg", "claim") if col in d.columns
              for c in d[col].to_list()]
        if cl:
            blk["claim"] = both_directions(mf_claim, cl, m_claims)
        ag[path.name] = blk
    surfaces["antigaming_probe_sets"] = {
        "kind": "held-out anti-gaming probe sets, built from TabFact "
                "test+validation (R14-H133 construction)",
        "per_file": ag,
    }
    print("antigaming done", flush=True)

    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    parts = [pl.read_parquet(io.BytesIO(zv.read(f"tals__vitaminc__{s}.parquet")))
             for s in ("test", "validation")]
    vc = pl.concat(parts)
    surfaces["vitaminc_holdout_SUPERSET"] = {
        "kind": "VitaminC test+validation - a strict SUPERSET of the "
                "`vitaminc_holdout` eval, so a zero here implies a zero there",
        "pool_rows": vc.height,
        "evidence": both_directions(mf_chunk, vc["evidence"].to_list(), m_chunks),
        "claim": both_directions(mf_claim, vc["claim"].to_list(), m_claims),
    }
    print("vitaminc superset done", flush=True)

    # ---- content read, alongside the string forms -------------------------- #
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    all_tabs = {}
    for split in ("train", "validation", "test"):
        n = next(x for x in z.namelist() if x.endswith(f"__{split}.parquet"))
        d = pl.read_parquet(io.BytesIO(z.read(n))).unique(subset=["table_id"], keep="first")
        for t, c, b in zip(d["table_id"].to_list(), d["table_caption"].to_list(),
                           d["table_text"].to_list(), strict=True):
            all_tabs[t] = build_chunk(c, b)
    hs, _ = h143_doc_ids()
    evb = {stem(x[len("tabfact:"):]) for x in
           pl.read_parquet(SEM / "R20-H177_eval_B.parquet")["doc_id"].unique().to_list()
           if str(x).startswith("tabfact:")}
    zt = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    ho = set()
    for split in ("validation", "test"):
        n = next(x for x in zt.namelist() if x.endswith(f"__{split}.parquet"))
        ho |= {stem(t) for t in
               pl.read_parquet(io.BytesIO(zt.read(n)))["table_id"].to_list()}
    eval_side_stems = ho | evb | hs
    eval_side = sorted({txt for t, txt in all_tabs.items() if stem(t) in eval_side_stems})
    uniq = df.unique(subset=["table_id"], keep="first")
    j = max_jaccard_per_unit(uniq["chunk_untrunc"].to_list(), eval_side)
    content = {
        "instrument": f"banked provenance_gate primitives, {N}-gram, best Jaccard "
                      "per member table against any single eval-side TabFact table",
        "eval_side_tables": len(eval_side),
        "member_tables_scored": uniq.height,
        f"tables_at_jaccard_ge_{JACCARD}": int((j >= JACCARD).sum()),
        "max": round(float(j.max()), 4),
        "p99": round(float(np.percentile(j, 99)), 4),
        "mean": round(float(j.mean()), 4),
    }
    print(json.dumps(content), flush=True)

    def worst(blk, path=()):
        bad = []
        for k, v in blk.items():
            if isinstance(v, dict):
                bad += worst(v, path + (k,))
            elif isinstance(v, int) and k not in ("n_query_units",) and v > 0 and (
                    "in_raw" in k or "in_truncated" in k or "in_normalised" in k
                    or k == "any_form" or k.endswith("table_id_in_member")):
                bad.append({"path": " / ".join(path + (k,)), "count": v})
        return bad

    nonzero = worst(surfaces)
    res = {
        "member": "tabfact_conformed",
        "clause": "C2 - disjointness from every evaluation surface",
        "method": "three string forms (raw / truncated 1500 / whitespace-collapsed "
                  "case-folded) crossed six ways, both directions, on EVIDENCE and "
                  "CLAIM units; plus the document-id channel on every surface that "
                  "exposes or can resolve a TabFact table id; plus a content read",
        "member_profile": {
            "rows": df.height,
            "distinct_evidence_untruncated": len(mf_chunk["raw"]),
            "distinct_claims": len(mf_claim["raw"]),
            "distinct_table_id": len(m_tidset),
            "distinct_table_id_stem": len(m_stems),
        },
        "surfaces": surfaces,
        "content_read_vs_eval_side_tabfact_tables": content,
        "nonzero_readings": nonzero,
        "all_string_forms_zero": not any(
            n["count"] for n in nonzero if "table_id" not in n["path"]),
        "all_document_channels_zero": not any(
            n["count"] for n in nonzero if "table_id" in n["path"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name} ({res['elapsed_s']}s)", flush=True)
    print(json.dumps(nonzero, indent=2), flush=True)


if __name__ == "__main__":
    main()
