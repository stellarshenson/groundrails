"""Conforming pipeline for the `vitaminc` training-mix member.

The unconformed member FAILED C2 (disjointness from every evaluation surface):
2 claim units / 4 member rows and 2 evidence units / 8 member rows match the
R19-H166-A1 held-out mechanism eval once whitespace is collapsed and case folded.
Two further channels the banked verification reported SEPARATELY (member claim
text equal to an arena DOCUMENT chunk; member evidence text equal to an eval
CLAIM string) read non-zero on the RAW form as well.

This pipeline removes rows.  It changes no label, relaxes no clause, retunes no
instrument and rewrites no evaluation surface.

FILTERS
  F1  C2 evaluation-surface collision.  Drop any member row whose claim OR
      evidence, under ANY of the three contract string forms (raw / truncated to
      CFG.chunk_max_chars / whitespace-collapsed case-folded), equals ANY claim
      OR evidence unit of ANY registered evaluation surface.  This is strictly
      wider than the clause's registered channel (claim-vs-claim,
      evidence-vs-evidence, pair-vs-pair): the cross channel is included so a
      member string that is verbatim arena document text is removed too.
  F2  C1 structural (amendment C-A1/C-A2).  Drop every row of any RAW
      (claim, evidence) pair that appears in the member under BOTH binary
      labels.  C-A1's decisive test is "a negative leg's (claim, evidence)
      identical to a positive leg's means the label cannot encode grounding",
      and its live control counts PAIRS member-wide (0 pairs on every clean
      lane).  The banked verification measured this only WITHIN case_id groups,
      where it reads 0; measured member-wide it does not.

The evaluation surfaces are NOT touched.  The R19-H166-A1 holdout used as the
C2 reference is the banked one, rebuilt byte-for-byte through
`R20_baseline_legs.vitaminc_holdout` from the UNCONFORMED mix - the surface the
queued arm will actually read.

CPU ONLY - CUDA_VISIBLE_DEVICES is forced empty before any import.

Run:  uv run python experiments/grounding-semantic/contract/vitaminc_conformed_build.py \
          2>&1 | tee logs/vitaminc_conformed_build.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent

OUT_PARQUET = HERE / "vitaminc_conformed.parquet"
OUT_JSON = HERE / "vitaminc_conformed_build.json"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading the banked verification module (it loads the banked loaders)")
VC = _mod("vc_contract", HERE / "vitaminc_contract.py")
LEGS = VC.LEGS
FORMS = VC.FORMS

ARCHIVE_COLS = ("unique_id", "case_id", "page", "claim", "evidence",
                "wiki_revision_id", "FEVER_id", "label", "revision_type")


def main():
    frames = VC.archive_frames()
    train_raw = frames["train"][0]
    mix, member = VC.assemble_mix()
    if member.height != train_raw.height:
        raise SystemExit(f"ALIGNMENT ABORT: member {member.height} vs archive "
                         f"{train_raw.height}")
    m_claims = member["claim"].to_list()
    m_chunks = member["chunk"].to_list()
    if m_claims != train_raw["claim"].to_list():
        raise SystemExit("ALIGNMENT ABORT: member row order is not the archive's")
    chunk_equals_evidence = m_chunks == train_raw["evidence"].to_list()
    log(f"member aligned to archive: {member.height} rows; "
        f"chunk == raw evidence: {chunk_equals_evidence}")

    log("rebuilding the banked R19-H166-A1 holdout from the UNCONFORMED mix")
    held, holdout_report = LEGS.vitaminc_holdout(VC.h150_mix_text(mix))
    log(f"holdout: {held.height} rows")

    surfaces = VC.surfaces(held)
    log(f"registered evaluation surfaces: {len(surfaces)}")

    # ---- member string forms ------------------------------------------- #
    mform_c = {f: [fn(t) for t in m_claims] for f, fn in FORMS.items()}
    mform_e = {f: [fn(t) for t in m_chunks] for f, fn in FORMS.items()}

    # ---- F1: collision with any registered evaluation surface ----------- #
    n = member.height
    drop_f1 = np.zeros(n, dtype=bool)
    f1_detail = {}
    for name, s_claims, s_chunks in surfaces:
        ent = {}
        for form, fn in FORMS.items():
            s_cl = {fn(t) for t in s_claims if t}
            s_ev = {fn(t) for t in s_chunks if t}
            s_all = s_cl | s_ev
            hit_cc = np.array([t in s_cl for t in mform_c[form]])
            hit_ce = np.array([t in s_ev for t in mform_c[form]])
            hit_ec = np.array([t in s_cl for t in mform_e[form]])
            hit_ee = np.array([t in s_ev for t in mform_e[form]])
            any_hit = hit_cc | hit_ce | hit_ec | hit_ee
            drop_f1 |= any_hit
            ent[form] = {
                "registered_channel_claim_vs_surface_claim_rows": int(hit_cc.sum()),
                "registered_channel_evidence_vs_surface_evidence_rows": int(hit_ee.sum()),
                "cross_channel_claim_vs_surface_evidence_rows": int(hit_ce.sum()),
                "cross_channel_evidence_vs_surface_claim_rows": int(hit_ec.sum()),
                "member_rows_dropped_by_this_surface_form": int(any_hit.sum()),
                "surface_units_claims": len(s_cl),
                "surface_units_evidence": len(s_ev),
                "surface_units_total": len(s_all),
            }
        if any(v["member_rows_dropped_by_this_surface_form"] for v in ent.values()):
            f1_detail[name] = ent
        log(f"  F1 {name}: worst-form member rows hit "
            f"{max(v['member_rows_dropped_by_this_surface_form'] for v in ent.values())}")

    # pair channel, for completeness - a pair hit implies both unit hits, so it
    # can add nothing, but it is measured rather than asserted.
    pair_hits = {}
    for name, s_claims, s_chunks in surfaces:
        if len(s_claims) != len(s_chunks) or not s_chunks:
            pair_hits[name] = {"computable": False}
            continue
        per_form = {}
        for form, fn in FORMS.items():
            sp = set(zip((fn(c) for c in s_claims), (fn(k) for k in s_chunks),
                         strict=True))
            hit = np.array([(a, b) in sp for a, b in
                            zip(mform_c[form], mform_e[form], strict=True)])
            per_form[form] = int(hit.sum())
            drop_f1 |= hit
        pair_hits[name] = {"computable": True, "member_rows_hit_per_form": per_form}

    log(f"F1 total member rows dropped: {int(drop_f1.sum())}")

    # ---- F2: RAW (claim, evidence) pairs carrying both labels ----------- #
    y = member["label"].to_numpy()
    pair_lab = collections.defaultdict(set)
    for c, k, lab in zip(m_claims, m_chunks, y, strict=True):
        pair_lab[(c, k)].add(float(lab))
    conflict = {p for p, labs in pair_lab.items() if len(labs) > 1}
    drop_f2 = np.array([(c, k) in conflict
                        for c, k in zip(m_claims, m_chunks, strict=True)])
    log(f"F2 structural: {len(conflict)} RAW pairs carry both labels, "
        f"{int(drop_f2.sum())} member rows")

    # the same statistic on the folded form, as a DIAGNOSTIC only - a pair that
    # is identical only after folding is still separable by a function of the
    # raw strings, so it is not the structural test and is not dropped.
    fold = FORMS["ws_collapsed_casefold"]
    pair_lab_f = collections.defaultdict(set)
    for c, k, lab in zip(mform_c["ws_collapsed_casefold"],
                         mform_e["ws_collapsed_casefold"], y, strict=True):
        pair_lab_f[(c, k)].add(float(lab))
    conflict_f = {p for p, labs in pair_lab_f.items() if len(labs) > 1}
    rows_f = int(sum(1 for c, k in zip(mform_c["ws_collapsed_casefold"],
                                       mform_e["ws_collapsed_casefold"], strict=True)
                     if (c, k) in conflict_f))
    log(f"F2 diagnostic (folded form, NOT dropped on): {len(conflict_f)} pairs, "
        f"{rows_f} rows")

    # ---- apply ---------------------------------------------------------- #
    drop = drop_f1 | drop_f2
    keep = ~drop
    log(f"TOTAL dropped {int(drop.sum())} of {n} rows "
        f"({drop.sum() / n:.6f}); keeping {int(keep.sum())}")

    conformed = pl.DataFrame({
        "archive_row": np.arange(n, dtype=np.int64),
        "claim": m_claims,
        "chunk": m_chunks,
        "label": member["label"].to_numpy(),
        "label_native": train_raw["label"].to_list(),
        "unique_id": train_raw["unique_id"].to_list(),
        "case_id": train_raw["case_id"].to_list(),
        "page": train_raw["page"].to_list(),
        "evidence": train_raw["evidence"].to_list(),
        "wiki_revision_id": train_raw["wiki_revision_id"].to_list(),
        "FEVER_id": train_raw["FEVER_id"].to_list(),
        "revision_type": train_raw["revision_type"].to_list(),
        "dropped_by_F1_c2_collision": drop_f1,
        "dropped_by_F2_c1_structural": drop_f2,
    }).filter(pl.Series(keep))
    conformed = conformed.drop(["dropped_by_F1_c2_collision",
                                "dropped_by_F2_c1_structural"])
    conformed.write_parquet(OUT_PARQUET)
    log(f"conformed member -> {OUT_PARQUET} ({conformed.height} rows)")

    # ---- what was dropped, in text ------------------------------------- #
    idx_f1 = [int(i) for i in np.flatnonzero(drop_f1)]
    idx_f2 = [int(i) for i in np.flatnonzero(drop_f2)]
    examples = [{
        "archive_row": i,
        "claim": m_claims[i][:200],
        "evidence": m_chunks[i][:200],
        "label": float(y[i]),
        "filter": "F1",
    } for i in idx_f1]

    report = {
        "member": "vitaminc_conformed",
        "built_from": "vitaminc (tals/vitaminc __train.parquet via "
                      "R10-H108_lane.public_train under "
                      "R16-H142_G1_arm.untruncated_evidence)",
        "contract": "docs/experiments/dataset-contract.md",
        "generated": time.strftime("%FT%T"),
        "compute": "CPU only, CUDA_VISIBLE_DEVICES empty",
        "input_rows": n,
        "output_rows": int(conformed.height),
        "dropped_rows": int(drop.sum()),
        "dropped_share": round(float(drop.sum()) / n, 8),
        "member_chunk_equals_raw_archive_evidence": bool(chunk_equals_evidence),
        "filters": {
            "F1_c2_evaluation_surface_collision": {
                "rule": ("drop any member row whose claim OR evidence, under any of "
                         "the three contract string forms, equals any claim OR "
                         "evidence unit of any registered evaluation surface; plus "
                         "any row whose (claim, evidence) pair equals a surface pair"),
                "scope_note": ("strictly wider than the clause's registered "
                               "claim-vs-claim / evidence-vs-evidence / pair channel - "
                               "the cross channel is included so that member text "
                               "which is verbatim arena DOCUMENT text is removed too"),
                "rows_dropped": int(drop_f1.sum()),
                "archive_rows": idx_f1,
                "per_surface": f1_detail,
                "pair_channel": pair_hits,
            },
            "F2_c1_structural_label_conflict": {
                "rule": ("drop every row of any RAW (claim, evidence) pair that "
                         "appears in the member under BOTH binary labels - "
                         "amendment C-A1's decisive structural test, counted "
                         "member-wide as its live control counts it"),
                "raw_pairs_conflicting": len(conflict),
                "rows_dropped": int(drop_f2.sum()),
                "share_of_member_rows": round(float(drop_f2.sum()) / n, 8),
                "folded_form_diagnostic": {
                    "status": "DIAGNOSTIC ONLY, not a drop criterion",
                    "why": ("a pair identical only after whitespace-collapse and "
                            "case-fold is still separable by a function of the raw "
                            "strings, so it is not the structural condition"),
                    "pairs": len(conflict_f),
                    "rows": rows_f,
                },
                "why_this_filter_exists": (
                    "the banked verification measured the structural condition only "
                    "WITHIN case_id contrastive groups, where it reads 0 of 111,323. "
                    "Measured member-wide - the way amendment C-A1's live positive "
                    "control counts it (0 pairs on every clean lane, 8,986 of 8,986 "
                    "on the withdrawn poisoned lane) - it reads "
                    f"{len(conflict)} pairs on the unconformed member"),
            },
        },
        "filter_overlap_rows": int((drop_f1 & drop_f2).sum()),
        "f1_dropped_examples": examples,
        "evaluation_surfaces_used_as_reference": [s[0] for s in surfaces],
        "h166a1_holdout": {
            "status": "REBUILT from the UNCONFORMED mix through the banked builder "
                      "R20_baseline_legs.vitaminc_holdout - the surface the queued "
                      "arm reads. It is NOT modified by this pipeline",
            "rows": int(held.height),
            "construction_report": holdout_report,
        },
        "what_was_NOT_done": (
            "no label changed, no leg re-weighted, no clause threshold moved, no "
            "instrument re-tuned, no evaluation surface rewritten or re-filtered. "
            "The pipeline only removes member rows"),
        "artifact": str(OUT_PARQUET.relative_to(ROOT)),
        "seconds": round(time.time() - T0, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    log(f"build report -> {OUT_JSON}")


if __name__ == "__main__":
    main()
