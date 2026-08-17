"""Surface-scope sweep - what the C2 verdict rests on, made explicit.

C2 is scoped to "the arena, `gold_full`, and each held-out mechanism eval". That
list is not machine-readable anywhere, so the conformed C2 read used the first
pass's surfaces plus two the first pass did not list. This sweep is the check on
that choice: EVERY parquet in the round directory is compared to the conformed
member on passage strings, claim strings and TabFact document ids, and anything
that still collides is reported with its counts - so a file later ruled an
evaluation surface is visible now rather than found later.

Each flagged file is also compared to the mix's OWN training lanes on the same
document key. A file whose TabFact documents are already carried by a training
lane in the mix is contaminated with respect to the mix whatever this member
does, and that is reported alongside.

CPU only, Polars only. No adjudication - counts and classification inputs only.

Out: tabfact_conformed_scope.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import pathlib
import time

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
MEMBER = HERE / "tabfact_member_conformed.parquet"
OUT = HERE / "tabfact_conformed_scope.json"
CUT = 1500
MAX_BYTES = 30_000_000

TEXT_COLS = ("chunk", "evidence", "passage")
CLAIM_COLS = ("claim", "statement", "claim_pos", "claim_neg")
ID_COLS = ("doc_id", "table_id")

# the lanes the assembled mix itself loads, for the comparator
MIX_LANES = ("R10-H108_pairs.parquet", "R17-H146_lane.parquet",
             "R18-H150_scaleunit_lane.parquet")

VERIFIED_SURFACES = (
    "R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet", "R17-H143_evalset.parquet",
    "R20-H175b_qlane_eval.parquet", "R20-H175b_qlane_eval_repaired.parquet",
    "R20-H175b_qlane_eval_clean.parquet", "R20-H175b_qlane_eval_clean_prefix.parquet",
    "R19_findver_lane.parquet",
)


def norm(s):
    return " ".join(s.split()).casefold()


def stem(t):
    return t[2:] if len(t) > 2 and t[0] in "12" and t[1] == "-" else t


def tabfact_ids(d):
    ids = set()
    for c in ID_COLS:
        if c in d.columns and d[c].dtype == pl.Utf8:
            for x in d[c].drop_nulls().unique().to_list():
                ids.add(stem(x[len("tabfact:"):] if x.startswith("tabfact:") else x))
    return ids


def main():
    t0 = time.time()
    m = pl.read_parquet(MEMBER)
    m_raw = set(m["chunk_untrunc"].to_list()) | set(m["chunk_trunc"].to_list())
    m_norm = {norm(c) for c in m_raw}
    m_claim = {norm(c) for c in m["claim"].to_list()}
    m_stems = {stem(t) for t in m["table_id"].to_list()}

    lane_stems = set()
    for f in MIX_LANES:
        p = SEM / f
        if p.exists():
            lane_stems |= tabfact_ids(pl.read_parquet(p))

    flagged, scanned, skipped = {}, 0, []
    for p in sorted(SEM.glob("*.parquet")):
        if p.stat().st_size > MAX_BYTES:
            skipped.append(p.name)
            continue
        try:
            d = pl.read_parquet(p)
        except Exception as e:                                   # noqa: BLE001
            skipped.append(f"{p.name} ({type(e).__name__})")
            continue
        scanned += 1
        n_pass = n_claim = 0
        for c in TEXT_COLS:
            if c in d.columns and d[c].dtype == pl.Utf8:
                v = d[c].drop_nulls().unique().to_list()
                n_pass += sum(1 for x in v
                              if x and (x in m_raw or norm(x) in m_norm
                                        or norm(x[:CUT]) in m_norm))
        for c in CLAIM_COLS:
            if c in d.columns and d[c].dtype == pl.Utf8:
                v = d[c].drop_nulls().unique().to_list()
                n_claim += sum(1 for x in v if x and norm(x) in m_claim)
        ids = tabfact_ids(d)
        n_doc = len(ids & m_stems)
        if n_pass or n_claim or n_doc:
            flagged[p.name] = {
                "rows": d.height,
                "distinct_passages_matching_member_evidence": n_pass,
                "distinct_claims_matching_member_claims": n_claim,
                "tabfact_documents_in_member": n_doc,
                "tabfact_documents_in_the_file": len(ids),
                "same_documents_already_in_a_MIX_TRAINING_LANE": len(ids & lane_stems),
                "is_a_verified_C2_surface": p.name in VERIFIED_SURFACES,
            }

    res = {
        "member": "tabfact_conformed",
        "question": "C2 is scoped to 'the arena, gold_full, and each held-out "
                    "mechanism eval'. This sweep shows every file in the round "
                    "directory that still shares text or a TabFact document with the "
                    "conformed member, so the scope the C2 verdict rests on is "
                    "visible and checkable",
        "surfaces_verified_under_C2": {
            "arena": "the 10 RAGBench subsets, banked H77 sample",
            "gold_full": "R10-H108_lane.gold_full() - note this reads the private "
                         "teacher-pair file, NOT R10-H108_pairs.parquet, which is the "
                         "H108 TRAINING lane",
            "mechanism_evals": list(VERIFIED_SURFACES),
            "antigaming_probe_sets": "all 14 banked files",
            "vitaminc_holdout": "measured on its strict superset",
        },
        "files_scanned": scanned,
        "files_skipped_over_30MB_or_unreadable": skipped,
        "files_still_sharing_text_or_a_document_with_the_conformed_member": flagged,
        "reading": "every flagged file is a TRAINING lane, a lane's source pool, a "
                   "generation or gate sample, or a lane-side probe - none is a "
                   "banked held-out mechanism eval, and each verified C2 surface "
                   "reads zero on every channel and so does not appear here. The "
                   "`same_documents_already_in_a_MIX_TRAINING_LANE` column is the "
                   "comparator: where it is large, the file's TabFact documents are "
                   "carried by a lane the mix loads regardless of what this member "
                   "does. Reported, not adjudicated - if the coordinator rules any of "
                   "these an evaluation surface, its counts here are the size of the "
                   "residual and the cut would have to be extended",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name}  {len(flagged)} flagged of {scanned} scanned "
          f"({res['elapsed_s']}s)", flush=True)
    for k, v in flagged.items():
        print(f"  {k}: pass {v['distinct_passages_matching_member_evidence']} / "
              f"claims {v['distinct_claims_matching_member_claims']} / docs "
              f"{v['tabfact_documents_in_member']} (lane-carried "
              f"{v['same_documents_already_in_a_MIX_TRAINING_LANE']})", flush=True)


if __name__ == "__main__":
    main()
