"""C2 EXHAUSTIVE surface sweep for the conformed `quant_misbind` member.

The banked C2 instrument (`quant_misbind_verify.eval_surfaces`) covers 13
surfaces: the arena, gold_full, VitaminC's holdout superset, seven mechanism
evals and THREE anti-gaming probe sets.  The campaign holds more than three -
one per arm - and the original member's C2 failure was caused precisely by a
probe set whose exclusion guard pointed at a different lane.  So every remaining
evaluation and probe surface in the round directory is swept here, under the same
three string forms plus a document-identity read.

Surfaces added by this sweep (beyond the banked 13):
  nine further anti-gaming probe sets  R14-H133, R16-H142-T, R16-H142-T-d2,
      R17-H145, R18-H150-d2, R18-H152-d1, R18-H152-d2, R18-H155-d1, R18-H155-d2,
      R18-H156, R19-H160-soupB
  R20-H177_eval_B_rebuilt        the rebuilt Lane B mechanism eval
  R17-H143_evalset_source        the 50,000-row pool the H143 eval was drawn from
  R11-H117_heldout_pairs         held-out long-form pairs
  R15_L1_bindprobe_pairs (+arm variant), R15_P1_typeprobe_quads (+topup)
  R17-H148_probe, R17-H149_probe, R18-H150_unitswap_probe
  R20-G0b_composed_probes

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_conformed_c2_full.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib
import time

import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent
CHUNK_MAX = 1500


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V = _mod("qmverify", HERE / "quant_misbind_verify.py")

ANTIGAMING = [
    "R14-H133_antigaming_set", "R16-H142-T_antigaming_set",
    "R16-H142-T-d2_antigaming_set", "R17-H145_antigaming_set",
    "R18-H150-d2_antigaming_set", "R18-H152-d1_antigaming_set",
    "R18-H152-d2_antigaming_set", "R18-H155-d1_antigaming_set",
    "R18-H155-d2_antigaming_set", "R18-H156_antigaming_set",
    "R19-H160-soupB_antigaming_set",
]


def surfaces():
    """(name, claims, evidence, document_ids, note)."""
    out = []
    for n in ANTIGAMING:
        p = GS / f"{n}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        docs = {f"tabfact:{x}" for x in d["table_id"].to_list() if x is not None}
        out.append((n, d["claim_pos"].to_list() + d["claim_neg"].to_list(), [], docs,
                    "anti-gaming paired probe set - claims only, keyed on table_id"))

    for n, ccol, kcol, dcol in (
        ("R20-H177_eval_B_rebuilt", "claim", "chunk", "doc_id"),
        ("R17-H143_evalset_source", "claim", "chunk", "doc_id"),
        ("R17-H148_probe", "claim", "chunk", "doc_id"),
        ("R17-H149_probe", "claim", "chunk", "doc_id"),
        ("R18-H150_unitswap_probe", "claim", "chunk", "doc_id"),
        ("R11-H117_heldout_pairs", "claim", "chunk", None),
    ):
        p = GS / f"{n}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        docs = ({x for x in d[dcol].to_list() if x is not None} if dcol else set())
        out.append((n, d[ccol].to_list(), d[kcol].to_list(), docs, "held-out surface"))

    for n in ("R15_L1_bindprobe_pairs", "R15_L1_bindprobe_pairs_R10-H108-lane-draw1"):
        p = GS / f"{n}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        docs = {f"tabfact:{x}" for x in d["table_id"].to_list() if x is not None}
        out.append((n, d["claim_pos"].to_list() + d["claim_neg"].to_list(), [], docs,
                    "R15 binding probe pairs - claims only, keyed on table_id"))

    for n in ("R15_P1_typeprobe_quads", "R15_P1_typeprobe_topup_quads"):
        p = GS / f"{n}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        cl = []
        for c in ("claim_a", "claim_b", "claim_c", "claim_d"):
            cl += d[c].to_list()
        docs = {f"tabfact:{x}" for x in d["table_id"].to_list() if x is not None}
        out.append((n, cl, [], docs,
                    "R15 type probe quads - claims only, keyed on table_id"))

    p = GS / "R20-G0b_composed_probes.parquet"
    if p.exists():
        d = pl.read_parquet(p)
        docs = set()
        for c in ("table_id_a", "table_id_b"):
            docs |= {f"tabfact:{x}" for x in d[c].to_list() if x is not None}
        ev = [x for c in ("doc_a", "doc_b") for x in d[c].to_list() if x]
        out.append(("R20-G0b_composed_probes", d["claim"].to_list(), ev, docs,
                    "composed-probe set - claims plus the two source table texts"))
    return out


def main():
    t0 = time.time()
    df = pl.read_parquet(GS / "R17-H146_lane_conformed.parquet")
    m_claims = [c for c in df["claim"].to_list() if c and c.strip()]
    m_chunks = [c for c in df["chunk"].to_list() if c and c.strip()]
    m_docs = set(df["doc_id"].to_list())

    forms = {"raw": lambda s: s,
             "truncated_1500": lambda s: s[:CHUNK_MAX],
             "normalised_ws_casefold": V.norm_ws}
    member = {k: ({f(c) for c in m_claims}, {f(c) for c in m_chunks})
              for k, f in forms.items()}

    results, totals = {}, {k: 0 for k in forms}
    for name, s_cl, s_ev, s_docs, note in surfaces():
        s_cl = [c for c in s_cl if c and c.strip()]
        s_ev = [c for c in s_ev if c and c.strip()]
        entry = {"note": note, "surface_claims": len(set(s_cl)),
                 "surface_evidence_units": len(set(s_ev)),
                 "surface_documents": len(s_docs),
                 "shared_documents": len(s_docs & m_docs), "forms": {}}
        for fname, f in forms.items():
            cl_hit = member[fname][0] & {f(c) for c in s_cl}
            ev_hit = member[fname][1] & {f(c) for c in s_ev}
            entry["forms"][fname] = {"claims_shared_strings": len(cl_hit),
                                     "evidence_shared_strings": len(ev_hit)}
            totals[fname] += len(cl_hit) + len(ev_hit)
        entry["clean"] = (entry["shared_documents"] == 0 and all(
            v["claims_shared_strings"] == 0 and v["evidence_shared_strings"] == 0
            for v in entry["forms"].values()))
        results[name] = entry
        print(f"  {name}: clean={entry['clean']} docs_shared="
              f"{entry['shared_documents']} "
              f"{ {k: (v['claims_shared_strings'], v['evidence_shared_strings']) for k, v in entry['forms'].items()} }",
              flush=True)

    out = {
        "clause": "C2 (exhaustive surface sweep beyond the banked 13)",
        "member_rows": df.height,
        "member_documents": len(m_docs),
        "surfaces_measured": len(results),
        "surfaces": results,
        "totals_per_form": totals,
        "all_forms_zero": all(v == 0 for v in totals.values()),
        "all_document_reads_zero": all(v["shared_documents"] == 0
                                       for v in results.values()),
        "surfaces_not_clean": sorted(k for k, v in results.items() if not v["clean"]),
        "seconds": round(time.time() - t0, 1),
    }
    p = HERE / "quant_misbind_conformed_c2_full_sweep.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("surfaces_measured", "totals_per_form", "all_forms_zero",
                       "all_document_reads_zero", "surfaces_not_clean")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
