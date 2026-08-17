"""CONFORMING PIPELINE for the `quant_misbind` member (R17-H146_lane.parquet).

The member failed C2, C3 and C8.  This script builds `R17-H146_lane_conformed.parquet`
alongside the original - the original is never overwritten - by REMOVAL only.  No
clause is relaxed and no construction rule is loosened.

Three removals, each traced to the clause it answers:

  D1  every FEVEROUS-sourced row (10,110 rows / 2,539 documents).  It is the whole
      of the C3 failure (a single pool with no split column and positionally
      unstable ids) and the whole of the C8 FEVEROUS failure (no licence, no
      retrieval date, untracked gitignored source, `feverous_available()`
      admitted=False).  Neither is repairable from artifacts on disk, and fetching
      a citable FEVEROUS release is outside this repository.

  D2  every pair whose claim or evidence collides with ANY evaluation surface, in
      any of C2's three string forms.  Computed with the banked C2 instrument over
      the same 13 surfaces (`quant_misbind_verify.eval_surfaces`).

  D4  every pair whose SOURCE DOCUMENT appears in an evaluation surface, measured
      by document identity rather than by string equality.  The conformed member
      is TabFact-only, so its documents carry the corpus's own stable `table_id`
      and every TabFact-derived evaluation surface carries the same identifier -
      an identity read, not the similarity heuristic the original pass was stuck
      with.  This is STRICTER than C2's own three-form instrument and is reported
      separately from it: 229 documents of `R20-H177_eval_B`, 458 of its 1,000
      pairs, sit on tables the member trains on under a different serialization.

  D5  the same two tests against the evaluation and probe surfaces the banked C2
      instrument does not cover - eleven further anti-gaming probe sets (one per
      arm, each a registered hold), the R15 binding and type probes, the H117
      held-out pairs and the H150 unit-swap probe.  The original member's C2
      failure was caused by exactly this gap: an exclusion guard pointed at a
      different lane.  `R17-H143_evalset_source` is measured but NOT dropped
      against - it is the 50,000-row pool the H143 eval was SAMPLED from
      (R17-H143_evalset.py line 27), a construction artifact rather than a read;
      the eval drawn from it reads zero on both tests.

  D3  the surplus pairs needed to restore EXACT per-family direction balance, which
      is a C5 element and which removal by D1/D2/D4/D5 would otherwise break.

Every dropped pair is dropped whole - both legs - so pair integrity holds.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_conformed_build.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import hashlib
import importlib.util
import json
import pathlib
import time

import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent
ROOT = GS.parent.parent

SRC = GS / "R17-H146_lane.parquet"
OUT = GS / "R17-H146_lane_conformed.parquet"
BUILD = HERE / "quant_misbind_conformed_build.json"
DROPSET = HERE / "quant_misbind_conformed_dropset.json"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def eval_documents():
    """Every document identifier any evaluation surface carries, in the member's
    namespace.  TabFact-derived surfaces key on the corpus's own stable table_id,
    so this is an identity read."""
    docs, per_surface = set(), {}

    ev = pl.read_parquet(GS / "R17-H143_evalset.parquet")
    v2 = pl.read_parquet(GS / "R14-H133_lane.v2-SUPERSEDED.parquet",
                         columns=["pair_id", "claim", "label", "chunk", "doc_id"])
    graded = ev.filter(~pl.col("control")).join(
        v2.select(["pair_id", "claim", "label", "doc_id"]).with_columns(
            pl.col("label").cast(pl.Int8)), on=["pair_id", "claim", "label"], how="left")
    ctrl = ev.filter(pl.col("control")).join(
        v2.select(["chunk", "doc_id"]).unique(subset=["chunk"]), on="chunk", how="left")
    got = set()
    for d in (graded, ctrl):
        got |= {x for x in d["doc_id"].to_list() if x is not None}
    per_surface["R17-H143_evalset"] = len(got)
    docs |= got

    for fname in ("R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet",
                  "R20-H175b_qlane_eval.parquet",
                  "R20-H175b_qlane_eval_repaired.parquet",
                  "R20-H175b_qlane_eval_clean.parquet",
                  "R20-H175b_qlane_eval_clean_prefix.parquet"):
        p = GS / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        if "doc_id" not in d.columns:
            continue
        got = {x for x in d["doc_id"].to_list() if x is not None}
        per_surface[fname.replace(".parquet", "")] = len(got)
        docs |= got

    for fname in ("R17-H146_antigaming_set.parquet", "R18-H150_antigaming_set.parquet",
                  "R19-H159_antigaming_set.parquet"):
        p = GS / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        got = {f"tabfact:{x}" for x in d["table_id"].to_list() if x is not None}
        per_surface[fname.replace(".parquet", "")] = len(got)
        docs |= got

    return docs, per_surface


def main():
    t0 = time.time()
    V = _mod("qmverify", HERE / "quant_misbind_verify.py")
    df = pl.read_parquet(SRC)
    n0, p0 = df.height, df["pair_id"].n_unique()
    print(f"source member: {n0} rows / {p0} pairs", flush=True)

    forms = {
        "raw": lambda s: s,
        f"truncated_{V.CHUNK_MAX}": lambda s: s[: V.CHUNK_MAX],
        "normalised_ws_casefold": V.norm_ws,
    }

    # ---------------------------------------------------------------- D1
    d1_pairs = set(df.filter(pl.col("source") != "tabfact")["pair_id"].to_list())
    print(f"D1 (non-tabfact source): {len(d1_pairs)} pairs", flush=True)

    # ---------------------------------------------------------------- D2
    # collision computed on the WHOLE member (before D1) so the drop set is a
    # property of the member, not of the order the removals are applied in.
    m_claims = df["claim"].to_list()
    m_chunks = df["chunk"].to_list()
    hits_claim, hits_chunk = set(), set()
    per_surface = {}
    if DROPSET.exists():
        cached = json.loads(DROPSET.read_text())
        hits_claim = set(cached["colliding_claim_strings"])
        hits_chunk = set(cached["colliding_evidence_strings"])
        per_surface = cached["per_surface"]
        print(f"D2 loaded from cache: {len(hits_claim)} claims, "
              f"{len(hits_chunk)} chunks", flush=True)
    else:
        member_forms = {
            k: ({f(c) for c in m_claims if c and c.strip()},
                {f(c) for c in m_chunks if c and c.strip()})
            for k, f in forms.items()
        }
        for name, s_claims, s_ev, note in V.eval_surfaces():
            s_claims = [c for c in s_claims if c and c.strip()]
            s_ev = [c for c in s_ev if c and c.strip()]
            entry = {}
            for fname, f in forms.items():
                mc, me = member_forms[fname]
                cl_hit = mc & {f(c) for c in s_claims}
                ev_hit = me & {f(c) for c in s_ev}
                entry[fname] = {"claims": len(cl_hit), "evidence": len(ev_hit)}
                # map the normalised hit back to the raw member strings
                for raw in m_claims:
                    if raw and f(raw) in cl_hit:
                        hits_claim.add(raw)
                for raw in m_chunks:
                    if raw and f(raw) in ev_hit:
                        hits_chunk.add(raw)
            per_surface[name] = {"note": note, "forms": entry}
            print(f"  {name}: {entry}", flush=True)
        DROPSET.write_text(json.dumps({
            "instrument": "quant_misbind_verify.eval_surfaces + the three C2 string forms",
            "per_surface": per_surface,
            "colliding_claim_strings": sorted(hits_claim),
            "colliding_evidence_strings": sorted(hits_chunk),
        }, indent=2))

    d2_pairs = set(
        df.filter(pl.col("claim").is_in(list(hits_claim))
                  | pl.col("chunk").is_in(list(hits_chunk)))["pair_id"].to_list())
    print(f"D2 (evaluation-surface collision): {len(d2_pairs)} pairs "
          f"({len(hits_claim)} claim strings, {len(hits_chunk)} evidence strings)",
          flush=True)

    # ---------------------------------------------------------------- D4
    eval_docs, doc_sources = eval_documents()
    d4_pairs = set(df.filter(pl.col("doc_id").is_in(list(eval_docs)))["pair_id"].to_list())
    print(f"D4 (evaluation-surface DOCUMENT identity): {len(d4_pairs)} pairs over "
          f"{len(set(df.filter(pl.col('doc_id').is_in(list(eval_docs)))['doc_id'].to_list()))} "
          f"documents", flush=True)

    # ---------------------------------------------------------------- D5
    F = _mod("c2full", HERE / "quant_misbind_conformed_c2_full.py")
    x_claims, x_chunks, x_docs, x_per_surface = set(), set(), set(), {}
    for name, s_cl, s_ev, s_docs, _note in F.surfaces():
        if name == "R17-H143_evalset_source":
            x_per_surface[name] = {"EXCLUDED_FROM_THE_DROP": True,
                                   "why": "the pool the H143 eval was sampled from, not "
                                          "a read; the eval itself reads zero"}
            continue
        cl, ev = set(), set()
        for fname, f in forms.items():
            sc = {f(c) for c in s_cl if c and c.strip()}
            se = {f(c) for c in s_ev if c and c.strip()}
            cl |= {c for c in m_claims if c and f(c) in sc}
            ev |= {c for c in m_chunks if c and f(c) in se}
        x_claims |= cl
        x_chunks |= ev
        x_docs |= s_docs
        x_per_surface[name] = {"claim_strings": len(cl), "evidence_strings": len(ev),
                               "surface_documents": len(s_docs)}
    d5_pairs = set(df.filter(pl.col("claim").is_in(list(x_claims))
                             | pl.col("chunk").is_in(list(x_chunks))
                             | pl.col("doc_id").is_in(list(x_docs)))["pair_id"].to_list())
    print(f"D5 (surfaces beyond the banked instrument): {len(d5_pairs)} pairs "
          f"({len(x_claims)} claim strings, {len(x_chunks)} evidence strings, "
          f"{len(x_docs)} surface documents pooled)", flush=True)

    dropped = d1_pairs | d2_pairs | d4_pairs | d5_pairs
    keep = df.filter(~pl.col("pair_id").is_in(list(dropped)))
    print(f"after D1+D2+D4+D5: {keep.height} rows / {keep['pair_id'].n_unique()} pairs",
          flush=True)

    # ---------------------------------------------------------------- D3
    pos = keep.filter(pl.col("label") == 1)
    cnt = collections.Counter(
        (f, d) for f, d in zip(pos["neg_family"].to_list(), pos["direction"].to_list()))
    d3_pairs = set()
    rebalance = {}
    for fam in sorted({f for f, _ in cnt}):
        up, down = cnt[(fam, "up")], cnt[(fam, "down")]
        surplus_dir = "up" if up > down else "down"
        surplus = abs(up - down)
        rebalance[fam] = {"up": up, "down": down, "surplus_direction": surplus_dir,
                          "pairs_dropped": surplus}
        if surplus:
            ids = sorted(pos.filter((pl.col("neg_family") == fam)
                                    & (pl.col("direction") == surplus_dir))["pair_id"]
                         .to_list())
            d3_pairs |= set(ids[-surplus:])
    print(f"D3 (direction rebalance): {len(d3_pairs)} pairs {rebalance}", flush=True)

    out = keep.filter(~pl.col("pair_id").is_in(list(d3_pairs))).sort(
        ["pair_id", "label"], descending=[False, True])

    # ---------------------------------------------------------------- integrity
    per_pair = out.group_by("pair_id").len()
    assert int((per_pair["len"] == 2).sum()) == per_pair.height, "pair integrity broken"
    out.write_parquet(OUT)

    fam = {k: v for k, v in out.group_by("neg_family").len().iter_rows()}
    dirs = {f"{a}:{b}": n for a, b, n in
            out.group_by(["neg_family", "direction"]).len().iter_rows()}

    man = {
        "conformed_artifact": str(OUT.relative_to(ROOT)),
        "source_artifact": str(SRC.relative_to(ROOT)),
        "source_blake2b_64": hashlib.blake2b(SRC.read_bytes(), digest_size=8).hexdigest(),
        "conformed_blake2b_64": hashlib.blake2b(OUT.read_bytes(), digest_size=8).hexdigest(),
        "method": "REMOVAL ONLY - no row is rewritten, no construction rule changed, "
                  "no clause relaxed; every drop is by whole pair",
        "before": {"rows": n0, "pairs": p0,
                   "families": {k: v for k, v in
                                pl.read_parquet(SRC).group_by("neg_family").len().iter_rows()},
                   "documents": pl.read_parquet(SRC)["doc_id"].n_unique()},
        "removals": {
            "D1_non_tabfact_source": {
                "clauses": ["C3", "C8"],
                "pairs": len(d1_pairs), "rows": 2 * len(d1_pairs),
                "reason": "FEVEROUS third - split axis not measurable from the artifact "
                          "on disk, ids positionally unstable, no licence, no retrieval "
                          "date, source file untracked and gitignored",
            },
            "D2_evaluation_surface_collision": {
                "clauses": ["C2"],
                "pairs": len(d2_pairs), "rows": 2 * len(d2_pairs),
                "colliding_claim_strings": len(hits_claim),
                "colliding_evidence_strings": len(hits_chunk),
                "pairs_not_already_removed_by_D1": len(d2_pairs - d1_pairs),
            },
            "D4_evaluation_surface_document_identity": {
                "clauses": ["C2 (stricter than its own instrument)", "C3"],
                "pairs": len(d4_pairs), "rows": 2 * len(d4_pairs),
                "pairs_not_already_removed_by_D1_D2": len(d4_pairs - d1_pairs - d2_pairs),
                "member_documents_removed": len(
                    set(df.filter(pl.col("doc_id").is_in(list(eval_docs)))["doc_id"]
                        .to_list())),
                "evaluation_surface_documents_pooled": len(eval_docs),
                "documents_per_surface": doc_sources,
                "reason": "the same table serialized differently is invisible to C2's "
                          "string forms; document identity is measurable here because "
                          "the member is TabFact-only and carries the corpus's own "
                          "stable table_id",
            },
            "D5_surfaces_beyond_the_banked_instrument": {
                "clauses": ["C2"],
                "pairs": len(d5_pairs), "rows": 2 * len(d5_pairs),
                "pairs_not_already_removed_by_D1_D2_D4": len(
                    d5_pairs - d1_pairs - d2_pairs - d4_pairs),
                "colliding_claim_strings": len(x_claims),
                "colliding_evidence_strings": len(x_chunks),
                "surface_documents_pooled": len(x_docs),
                "per_surface": x_per_surface,
            },
            "D3_direction_rebalance": {
                "clauses": ["C5"],
                "pairs": len(d3_pairs), "rows": 2 * len(d3_pairs),
                "per_family": rebalance,
            },
        },
        "after": {"rows": out.height, "pairs": out["pair_id"].n_unique(),
                  "families": fam, "directions": dirs,
                  "documents": out["doc_id"].n_unique(),
                  "sources": {k: v for k, v in out.group_by("source").len().iter_rows()}},
        "volume_cost": {
            "rows_dropped": n0 - out.height,
            "rows_dropped_share": round((n0 - out.height) / n0, 6),
            "pairs_dropped": p0 - out["pair_id"].n_unique(),
            "pairs_dropped_share": round((p0 - out["pair_id"].n_unique()) / p0, 6),
        },
        "seconds": round(time.time() - t0, 1),
    }
    BUILD.write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2), flush=True)


if __name__ == "__main__":
    main()
