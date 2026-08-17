"""Contract clause C2 - `tabfact` disjointness from EVERY evaluation surface.

CPU only, Polars only, torch-free.

Method is the banked six-form cross (R20-H177_evalB_contamination_assessment):
three string forms - raw, truncated to `CFG.chunk_max_chars` = 1,500, and
whitespace-collapsed case-folded - crossed both ways, run in BOTH directions
(member as query against the surface, surface as query against the member).
Two units are measured on every surface that carries them: EVIDENCE (passage)
and CLAIM.

Document-level identity is measured alongside the string forms wherever a
surface exposes a TabFact table id, because a re-serialised table is invisible
to string matching while still being the same document. TabFact writes one table
under both a `1-` and a `2-` csv id, so ids are compared stem-stripped too.

Out: tabfact_c2.json
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
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
MEMBER = HERE / "tabfact_member.parquet"
OUT = HERE / "tabfact_c2.json"
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


def forms(texts):
    raw = {t for t in texts if t}
    trunc = {t[:CUT] for t in raw}
    return {"raw": raw, "trunc": trunc,
            "nraw": {norm(t) for t in raw}, "ntrunc": {norm(t) for t in trunc}}


def six_forms(query_texts, target):
    """The banked six checks: a query string in the target's forms."""
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


# --------------------------------------------------------------------------- #
def load_arena():
    """Arena document chunks AND responses, byte-identical to the banked gate's
    sample (asserted against provenance_gate.load_arena)."""
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
    print(f"member: {df.height} rows, {len(mf_chunk['raw'])} distinct evidence, "
          f"{len(mf_claim['raw'])} distinct claims, {len(m_tidset)} tables", flush=True)

    surfaces = {}

    # ---- S1 blind arena -------------------------------------------------- #
    docs, resp = load_arena()
    flat_docs = [c for v in docs.values() for c in v]
    flat_resp = [c for v in resp.values() for c in v]
    surfaces["arena_ragbench_10_subsets"] = {
        "kind": "blind arena - the 10 RAGBench subsets, banked H77 sample "
                "(250 rows/subset seed 0, first 8 document chunks)",
        "subsets": {k: len(v) for k, v in docs.items()},
        "evidence": both_directions(mf_chunk, flat_docs, m_chunks),
        "claim_vs_arena_response": both_directions(mf_claim, flat_resp, m_claims),
        "document_id_channel": "the arena parquet exposes no table id - "
                               "no document-level join is computable; the n-gram "
                               "census under C4 covers that direction",
    }
    print("arena done", flush=True)

    # ---- S2 gold_full ---------------------------------------------------- #
    arm = _mod("g1arm", SEM / "R16-H142_G1_arm.py")
    gc, gk, _gy = arm.H108.gold_full()
    gold_chunks = [c for ks in gk for c in ks]
    surfaces["gold_full"] = {
        "kind": "the held-out gold test surface (R10-H108_lane.gold_full, all "
                "2,752 claims); chunk lists flattened",
        "claims": len(gc), "chunks": len(gold_chunks),
        "evidence": both_directions(mf_chunk, gold_chunks, m_chunks),
        "claim": both_directions(mf_claim, gc, m_claims),
        "document_id_channel": "gold_full carries no corpus document id (banked "
                               "R20 gold_full audit) - not computable",
    }
    print("gold_full done", flush=True)

    # ---- S3..S7 mechanism evals with claim + chunk columns --------------- #
    evals = {
        "R20-H177_eval_B": SEM / "R20-H177_eval_B.parquet",
        "R20-H177_eval_C": SEM / "R20-H177_eval_C.parquet",
        "R17-H143_evalset": SEM / "R17-H143_evalset.parquet",
        "R20-H175b_qlane_eval": SEM / "R20-H175b_qlane_eval.parquet",
        "R20-H175b_qlane_eval_repaired": SEM / "R20-H175b_qlane_eval_repaired.parquet",
        "R20-H175b_qlane_eval_clean": SEM / "R20-H175b_qlane_eval_clean.parquet",
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
                "share_of_surface_tabfact_docs_in_member": round(
                    len({stem(i) for i in ids} & m_stems) / max(len(ids), 1), 4),
            }
            if "source" in d.columns:
                blk["rows_by_source"] = dict(d.group_by("source").len().iter_rows())
                sub = d.filter(pl.col("source") == "tabfact")
                if sub.height:
                    ss = {stem(x[len("tabfact:"):]) for x in sub["doc_id"].unique().to_list()
                          if str(x).startswith("tabfact:")}
                    blk["document_id_channel"]["tabfact_half_rows"] = sub.height
                    blk["document_id_channel"]["tabfact_half_pairs"] = (
                        sub["pair_id"].n_unique() if "pair_id" in sub.columns else None)
                    blk["document_id_channel"]["tabfact_half_docs"] = len(ss)
                    blk["document_id_channel"]["tabfact_half_docs_in_member"] = len(ss & m_stems)
        surfaces[name] = blk
        print(f"{name} done", flush=True)

    # ---- S8 anti-gaming probe sets (TabFact-derived, table_id exposed) ---- #
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
        blk["evidence_channel"] = ("the banked parquet stores no evidence column; "
                                   "evidence is regenerated from table_id, so the "
                                   "table-id count above IS the evidence test")
        ag[path.name] = blk
    surfaces["antigaming_probe_sets"] = {
        "kind": "held-out anti-gaming probe sets - built from TabFact "
                "test+validation with every table_id present in TabFact train "
                "removed (R14-H133_antigaming construction)",
        "per_file": ag,
    }
    print("antigaming done", flush=True)

    # ---- S9 vitaminc_holdout, measured on its SUPERSET -------------------- #
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    parts = [pl.read_parquet(io.BytesIO(zv.read(f"tals__vitaminc__{s}.parquet")))
             for s in ("test", "validation")]
    vc = pl.concat(parts)
    surfaces["vitaminc_holdout_SUPERSET"] = {
        "kind": "the `vitaminc_holdout` eval is filtered out of VitaminC "
                "test+validation. Measured here against that full pool, which is a "
                "strict SUPERSET of the eval - a zero on the superset implies a "
                "zero on the eval, so this is a stronger read, not a proxy",
        "pool_rows": vc.height,
        "evidence": both_directions(mf_chunk, vc["evidence"].to_list(), m_chunks),
        "claim": both_directions(mf_claim, vc["claim"].to_list(), m_claims),
    }
    print("vitaminc superset done", flush=True)

    # ---- roll-up ---------------------------------------------------------- #
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
        "member": "tabfact",
        "clause": "C2 - disjointness from every evaluation surface",
        "method": "three string forms (raw / truncated to 1,500 / "
                  "whitespace-collapsed case-folded) crossed six ways, run in both "
                  "directions, on EVIDENCE and CLAIM units; plus the document-id "
                  "channel wherever the surface exposes a TabFact table id",
        "member_profile": {
            "rows": df.height,
            "distinct_evidence_untruncated": len(mf_chunk["raw"]),
            "distinct_claims": len(mf_claim["raw"]),
            "distinct_table_id": len(m_tidset),
            "distinct_table_id_stem": len(m_stems),
        },
        "surfaces": surfaces,
        "nonzero_readings": nonzero,
        "all_string_forms_zero": not any(
            n["count"] for n in nonzero if "table_id" not in n["path"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name} ({res['elapsed_s']}s)", flush=True)
    print(json.dumps(nonzero, indent=2), flush=True)


if __name__ == "__main__":
    main()
