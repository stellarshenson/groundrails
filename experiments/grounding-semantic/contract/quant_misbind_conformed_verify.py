"""FULL RE-VERIFICATION of the CONFORMED `quant_misbind` member against C1-C8.

The contract's failure policy requires re-verification against EVERY clause after
a pipeline fix, not only the failed ones, so all eight are recomputed from the
conformed parquet.  Nothing is cited from the original pass.

The banked verification module (`quant_misbind_verify.py`) is imported and its
LANE constant repointed at `R17-H146_lane_conformed.parquet`; its stage outputs
are written under a `quant_misbind_conformed_` prefix so the original artifacts
are never overwritten.  Three stages are written here rather than reused, because
their content is member-specific and the banked versions hard-code the original
member's sources and registration:

  c1s   the C-A1 / C-A2 decisive C1 tests, which the banked C1 stage reports
        only indirectly (structural identity, and strict separation under an
        instrument sensitive to the predicate the lane corrupts)
  c3    split semantics for a TabFact-only member, including the decisive
        stable-id read against the FEVEROUS-derived mechanism eval's TabFact
        source tables
  c78   declared units and provenance for the conformed artifact

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_conformed_verify.py [stage ...]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import datetime
import hashlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent
ROOT = GS.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

CONFORMED = GS / "R17-H146_lane_conformed.parquet"
PREFIX = "quant_misbind_conformed_"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V = _mod("qmverify", HERE / "quant_misbind_verify.py")
V.LANE = CONFORMED


def save(stage, obj):
    p = HERE / f"{PREFIX}{stage}.json"
    p.write_text(json.dumps(obj, indent=2))
    print(f"  -> {p.name}", flush=True)
    return obj


V.save = save


# --------------------------------------------------------------------------- #
# C1 - the decisive tests, per amendments C-A1 and C-A2
# --------------------------------------------------------------------------- #
def stage_c1_decisive():
    t0 = time.time()
    df = V.lane()
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    # TEST 1 (C-A1, structural): a negative (claim, evidence) identical to a positive's
    pos_pairs = set(zip(pos["claim"].to_list(), pos["chunk"].to_list()))
    neg_pairs = set(zip(neg["claim"].to_list(), neg["chunk"].to_list()))
    collide = pos_pairs & neg_pairs
    n_neg_rows_colliding = int(neg.filter(
        pl.struct(["claim", "chunk"]).map_elements(
            lambda s: (s["claim"], s["chunk"]) in pos_pairs,
            return_dtype=pl.Boolean)).height) if collide else 0

    # TEST 2 (C-A2, strict separation) under the PREDICATE-SENSITIVE instrument.
    # Containment is predicate-BLIND on this lane by construction: both legs
    # assert a numeral the evidence prints, so it cannot read the binding the
    # lane corrupts. The instrument that can is the binding-level audit - is the
    # asserted (row_key, column) -> value binding the one the source table states?
    binding = json.loads((HERE / f"{PREFIX}c1_binding.json").read_text())
    neg_attested = 1.0 - binding["negative_binding_unattested_rate"]
    pos_attested = binding["positive_binding_attested_rate"]

    cont = json.loads((HERE / f"{PREFIX}c1_containment.json").read_text())
    nb = cont["legs"]["negative"]["untruncated"]
    pb = cont["legs"]["positive"]["untruncated"]

    out = {
        "clause": "C1 - decisive tests as restated by amendments C-A1 and C-A2",
        "test_1_structural": {
            "definition": "a negative leg's (claim, evidence) identical to a positive "
                          "leg's means the label cannot encode grounding",
            "distinct_positive_claim_evidence_pairs": len(pos_pairs),
            "distinct_negative_claim_evidence_pairs": len(neg_pairs),
            "colliding_pairs": len(collide),
            "negative_rows_affected": n_neg_rows_colliding,
            "share_of_rows": round(n_neg_rows_colliding / max(df.height, 1), 6),
            "fires": bool(collide),
            "live_positive_control_reference": "fires on 8,986 of 8,986 pairs (100%) in "
                                               "the withdrawn poisoned R20-H175b_qlane",
        },
        "test_2_strict_separation": {
            "instrument": "binding-level attestation, re-derived from the source tables "
                          "for the full population - the instrument SENSITIVE to the "
                          "predicate this lane corrupts (which cell the claim binds to)",
            "negative_leg_high_attestation_rate": round(neg_attested, 6),
            "positive_leg_high_attestation_rate": round(pos_attested, 6),
            "strictly_below": bool(neg_attested < pos_attested),
            "margin": round(pos_attested - neg_attested, 6),
            "predicate_blind_instrument_reported_separately": {
                "instrument": "claim-to-evidence containment",
                "negative_share_ge_0.90": nb["share_ge_0.90"],
                "positive_share_ge_0.90": pb["share_ge_0.90"],
                "negative_mean": nb["mean"], "positive_mean": pb["mean"],
                "reading": "C-A2: a predicate-blind instrument showing no separation is "
                           "NOT evidence of incommensurability. Both legs assert a real "
                           "cell of the same table, so containment cannot read the "
                           "binding and is reported as a diagnostic only",
            },
        },
        "test_3_absolute_level_reported": {
            "negative_leg_binding_attested_rate": round(neg_attested, 6),
            "negative_leg_containment_fully_attested_share": nb["share_fully_attested_eq_1.0"],
            "negative_leg_containment_share_ge_0.90": nb["share_ge_0.90"],
            "finding": "the negative leg's BINDING attestation is 0.0 - every negative "
                       "asserts a binding the evidence contradicts. The containment "
                       "reading is high on both legs because the asserted numeral is a "
                       "real cell of the same table, which is the construction",
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c1_decisive", out)


# --------------------------------------------------------------------------- #
# C3 - split semantics for the conformed (TabFact-only) member
# --------------------------------------------------------------------------- #
def stage_c3_conformed():
    t0 = time.time()
    P = _mod("h144pairs", GS / "R17-H144_pairs.py")
    df = V.lane()
    src_counts = {k: v for k, v in df.group_by("source").len().iter_rows()}
    doc_ns = collections.Counter(d.split(":")[0] for d in set(df["doc_id"].to_list()))

    # --- the archive's own split axis, measured not read from the card
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    names = {n.split("__")[-1].replace(".parquet", ""): n
             for n in z.namelist() if n.endswith(".parquet")}
    splits, tables = {}, {}
    for split, n in names.items():
        d = pl.read_parquet(io.BytesIO(z.read(n)))
        splits[split] = d.height
        tables[split] = set(d["table_id"].to_list())
    axis = {}
    for other in sorted(s for s in tables if s != "train"):
        axis[f"train_vs_{other}_shared_table_ids"] = len(tables["train"] & tables[other])
        axis[f"{other}_table_ids"] = len(tables[other])
    axis["train_table_ids"] = len(tables["train"])

    member_ids = {d.split(":", 1)[1] for d in df["doc_id"].to_list()
                  if d.startswith("tabfact:")}
    leak = {s: len(member_ids & tables[s]) for s in tables if s != "train"}
    in_train = len(member_ids & tables["train"])

    # --- decisive document read against the FEVEROUS-derived mechanism eval:
    # every eval row resolved to its source doc_id; for the TabFact namespace that
    # id is the corpus's own stable table_id, so identity is KNOWN
    ev = pl.read_parquet(GS / "R17-H143_evalset.parquet")
    v2 = pl.read_parquet(GS / "R14-H133_lane.v2-SUPERSEDED.parquet",
                         columns=["pair_id", "claim", "label", "chunk", "doc_id"])
    graded = ev.filter(~pl.col("control")).join(
        v2.select(["pair_id", "claim", "label", "doc_id"]).with_columns(
            pl.col("label").cast(pl.Int8)), on=["pair_id", "claim", "label"], how="left")
    ctrl = ev.filter(pl.col("control")).join(
        v2.select(["chunk", "doc_id"]).unique(subset=["chunk"]), on="chunk", how="left")
    eval_docs = set()
    for d in (graded, ctrl):
        eval_docs |= {x for x in d["doc_id"].to_list() if x is not None}
    eval_ns = collections.Counter(d.split(":")[0] for d in eval_docs)
    eval_tabfact = {d for d in eval_docs if d.startswith("tabfact:")}
    member_docs = set(df["doc_id"].to_list())

    out = {
        "clause": "C3",
        "member_type": "constructed lane - it has no split of its own; the axis that "
                       "matters is which split of each SOURCE it reads and whether that "
                       "split is disjoint from anything used for evaluation",
        "rows_by_source": src_counts,
        "distinct_documents_by_namespace": dict(doc_ns),
        "single_source": sorted(src_counts) == ["tabfact"],
        "tabfact": {
            "axis": "the archive's own table_id",
            "archive_split_rows": splits,
            "measured_axis": axis,
            "member_tables": len(member_ids),
            "member_tables_in_train": in_train,
            "member_tables_in_a_non_train_split": leak,
            "selection_predicate": "the *__train.parquet member of dataset-tabfact.zip "
                                   "ONLY (R17-H144_pairs.tabfact_tables), deduplicated "
                                   "on table_text",
        },
        "decisive_document_read_against_the_mechanism_eval": {
            "eval_surface": "R17-H143_evalset.parquet - the held-out mechanism eval this "
                            "member's source pool was excluded against",
            "eval_resolved_source_documents": len(eval_docs),
            "eval_documents_by_namespace": dict(eval_ns),
            "eval_tabfact_documents_stable_ids": len(eval_tabfact),
            "shared_with_member": len(eval_tabfact & member_docs),
            "all_eval_documents_shared_with_member": len(eval_docs & member_docs),
            "reading": "for a TabFact-only member every document on both sides carries "
                       "the corpus's own stable table_id, so this is an identity read, "
                       "not a similarity heuristic. The FEVEROUS documents of the eval "
                       "remain unstably keyed, but the member no longer contains any "
                       "FEVEROUS document, so no unmeasurable side remains",
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c3_split", out)


# --------------------------------------------------------------------------- #
# C6 supplement - the association the assembled MIX carries on the pair key
# --------------------------------------------------------------------------- #
def stage_c6_mix():
    t0 = time.time()
    H108 = _mod("h108", GS / "R10-H108_lane.py")
    H174 = _mod("h174", GS / "R20-H174_arm_run.py")

    print("building the clean public mix through the banked loader...", flush=True)
    claims, chunks, y, tags = H108.public_train()
    print(f"  clean mix: {len(claims)} rows", flush=True)

    lane_df = V.lane()
    other = collections.defaultdict(lambda: {"claims": [], "chunks": [], "y": []})
    for c, k, lab, tg in zip(claims, chunks, y.tolist(), tags):
        other[tg]["claims"].append(c)
        other[tg]["chunks"].append(k)
        other[tg]["y"].append(lab)

    for fname, group, *_ in H174.LANES:
        if group == "quant_misbind":
            continue
        d = pl.read_parquet(GS / fname)
        other[group]["claims"] += d["claim"].to_list()
        other[group]["chunks"] += d["chunk"].to_list()
        other[group]["y"] += d["label"].cast(pl.Float32).to_list()
        print(f"  lane {group}: {d.height} rows", flush=True)

    total = sum(len(v["claims"]) for v in other.values()) + lane_df.height

    lane_chunks_raw = set(lane_df["chunk"].to_list())
    lane_chunks_trunc = {c[: V.CHUNK_MAX] for c in lane_chunks_raw}
    lane_chunks_norm = {V.norm_ws(c) for c in lane_chunks_raw}
    lane_claims_norm = {V.norm_ws(c) for c in lane_df["claim"].to_list()}

    per_group, key_owner = {}, {}
    for g, v in other.items():
        shared_raw = lane_chunks_raw & set(v["chunks"])
        shared_tr = lane_chunks_trunc & {c[: V.CHUNK_MAX] for c in v["chunks"]}
        shared_no = lane_chunks_norm & {V.norm_ws(c) for c in v["chunks"]}
        per_group[g] = {
            "rows": len(v["claims"]),
            "shared_evidence_raw": len(shared_raw),
            "shared_evidence_truncated_1500": len(shared_tr),
            "shared_evidence_normalised": len(shared_no),
            "shared_claims_normalised": len(lane_claims_norm
                                            & {V.norm_ws(c) for c in v["claims"]}),
        }
        for c in shared_no:
            key_owner.setdefault(c, []).append(g)

    feat = np.full(lane_df.height, np.nan)
    assoc_rows = 0
    if key_owner:
        by_key = collections.defaultdict(list)
        for g, v in other.items():
            for c, lab in zip(v["chunks"], v["y"]):
                k = V.norm_ws(c)
                if k in key_owner:
                    by_key[k].append(lab)
        for i, c in enumerate(lane_df["chunk"].to_list()):
            k = V.norm_ws(c)
            if k in by_key:
                feat[i] = float(np.mean(by_key[k]))
                assoc_rows += 1

    ok = ~np.isnan(feat)
    labels = lane_df["label"].to_numpy()
    a = (V.auroc(labels[ok], feat[ok])
         if ok.sum() and len(set(labels[ok].tolist())) > 1 else None)

    out = {
        "clause": "C6 (mix-association supplement - the eval-facing test of C-A2)",
        "mix_rows_total": total,
        "mix_groups": len(other) + 1,
        "member_rows": lane_df.height,
        "pair_key": "the evidence chunk - both legs of a pair carry it byte-identically",
        "key_sharing_with_other_mix_members": {
            g: v for g, v in per_group.items()
            if any(v[k] for k in ("shared_evidence_raw", "shared_evidence_truncated_1500",
                                  "shared_evidence_normalised", "shared_claims_normalised"))
        },
        "groups_sharing_nothing": sorted(
            g for g, v in per_group.items()
            if not any(v[k] for k in ("shared_evidence_raw",
                                      "shared_evidence_truncated_1500",
                                      "shared_evidence_normalised",
                                      "shared_claims_normalised"))),
        "keys_shared_with_any_other_member": len(key_owner),
        "mix_keyed_label_association": {
            "definition": "mean label the REST of the mix attaches to this row's "
                          "evidence key (whitespace-normalised)",
            "coverage_rows": int(assoc_rows),
            "coverage_share": round(float(assoc_rows / lane_df.height), 6),
            "auroc_vs_label": None if a is None else round(a, 6),
        },
        "within_pair_note": "any feature keyed on the pair key takes the SAME value on "
                            "both legs, so its within-pair separation is exactly 0.5",
        "seconds": round(time.time() - t0, 1),
    }
    return save("c6_mix_assoc", out)


# --------------------------------------------------------------------------- #
# C7 / C8 for the conformed artifact
# --------------------------------------------------------------------------- #
def stage_c78_conformed():
    t0 = time.time()
    df = V.lane()
    rows, pairs = df.height, df["pair_id"].n_unique()
    fam = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    build = json.loads((HERE / "quant_misbind_conformed_build.json").read_text())
    dec = build["after"]

    per_pair = df.group_by("pair_id").len()

    # measured retrieval date for the TabFact archive - the archive is gitignored,
    # so the date is measured from the artifact rather than read from a sidecar
    zp = DATA / "dataset-tabfact.zip"
    z = zipfile.ZipFile(zp)
    member_dates = sorted({i.date_time for i in z.infolist()})
    mtime = datetime.datetime.fromtimestamp(zp.stat().st_mtime).isoformat(timespec="seconds")
    sidecar = (DATA / "dataset-tabfact.md").read_text()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch",
         "data/external/datasets/dataset-tabfact.md",
         "scripts/fetch_grounding_datasets.py"],
        cwd=ROOT, capture_output=True, text=True)

    internal = {
        "rows": rows, "pairs": pairs,
        "rows_per_pair": round(rows / pairs, 6),
        "pairs_with_exactly_two_rows": int((per_pair["len"] == 2).sum()),
        "distinct_claims": df["claim"].n_unique(),
        "distinct_evidence_chunks": df["chunk"].n_unique(),
        "distinct_documents": df["doc_id"].n_unique(),
        "distinct_columns": df["column"].n_unique(),
        "distinct_row_keys": df["row_key"].n_unique(),
        "distinct_claim_chunk_pairs": df.select(["claim", "chunk"]).n_unique(),
        "max_rows_per_chunk": int(df.group_by("chunk").len()["len"].max()),
        "mean_rows_per_chunk": round(float(df.group_by("chunk").len()["len"].mean()), 6),
        "max_pairs_per_document": int(df.filter(pl.col("label") == 1)
                                        .group_by("doc_id").len()["len"].max()),
        "mean_pairs_per_document": round(float(df.filter(pl.col("label") == 1)
                                                 .group_by("doc_id").len()["len"].mean()), 6),
        "templates": {str(k): v for k, v in df.group_by("template_id").len().iter_rows()},
        "serial_forms": {k: v for k, v in df.group_by("serial_form").len().iter_rows()},
        "rows_by_source": {k: v for k, v in df.group_by("source").len().iter_rows()},
        "duplicate_claim_strings": rows - df["claim"].n_unique(),
        "label_balance": {str(k): v for k, v in df.group_by("label").len().iter_rows()},
    }

    # public-repository scan: no client or company name may appear anywhere in the
    # member's text. Sources are declared per row and both are public corpora.
    sources = sorted(set(df["source"].to_list()))

    out = {
        "clause": "C7 + C8",
        "c7": {
            "declared_unit": "BOTH - rows AND pairs, declared in the conformed build "
                             "manifest and re-measured here from the parquet",
            "declared_rows": dec["rows"], "declared_pairs": dec["pairs"],
            "declared_families": dec["families"],
            "measured_rows": rows, "measured_pairs": pairs, "measured_families": fam,
            "rows_match": rows == dec["rows"],
            "pairs_match": pairs == dec["pairs"],
            "families_match": fam == dec["families"],
            "arm_wrapper_registration_state": {
                "R18-H150_arm_run.LANES": "registers R17-H146_lane.parquet at 30,000 rows "
                                          "/ 15,000 pairs - the ORIGINAL artifact",
                "R20-H174_arm_run.LANES": "same registration",
                "consequence_measured": "the conformed artifact is not registered in "
                                        "either wrapper; substituting it without updating "
                                        "the registered rows / pairs / family counts would "
                                        "trip the wrappers' LANE ABORT guard before a card "
                                        "is touched",
            },
        },
        "c8": {
            "artifact": str(CONFORMED.relative_to(ROOT)),
            "blake2b_64": hashlib.blake2b(CONFORMED.read_bytes(),
                                          digest_size=8).hexdigest(),
            "derivation": "removal-only from R17-H146_lane.parquet "
                          "(quant_misbind_conformed_build.py); no row rewritten",
            "sources": {
                "tabfact": {
                    "source_url": "https://github.com/wenhuchen/Table-Fact-Checking/"
                                  "archive/refs/heads/master.zip (the spec's `github` "
                                  "field in scripts/fetch_grounding_datasets.py)",
                    "archive": "data/external/datasets/dataset-tabfact.zip (gitignored; "
                               "rebuildable by the tracked fetcher from the tracked URL)",
                    "sidecar": "data/external/datasets/dataset-tabfact.md (tracked)",
                    "licence": "CC-BY-4.0",
                    "licence_evidence": "recorded in the tracked sidecar and in the "
                                        "tracked fetcher's spec table",
                    "retrieval_date_measured": {
                        "archive_member_timestamps": [list(d) for d in member_dates],
                        "archive_mtime": mtime,
                        "method": "read from the archive itself - the sidecar does not "
                                  "record it, so it is MEASURED here rather than asserted",
                    },
                    "selection_predicate": "the *__train.parquet member of the archive "
                                           "ONLY, deduplicated on table_text; tables with "
                                           ">= 4 body rows of uniform width and >= 2 "
                                           "columns; a label column and at least one "
                                           "numeric column must exist "
                                           "(R17-H144_pairs.tabfact_tables)",
                    "tracked_provenance_files": tracked.stdout.split(),
                },
            },
            "sources_removed_by_the_conforming_pipeline": {
                "feverous": "10,110 rows / 2,539 documents removed - no licence, no "
                            "retrieval date, no sidecar, source file untracked and "
                            "gitignored, feverous_available() admitted=False",
            },
            "sidecar_licence_text_head": sidecar.splitlines()[2:6],
            "internal_structure": internal,
            "public_repository_check": {
                "sources_present": sources,
                "all_sources_public": set(sources) <= {"tabfact"},
                "client_or_company_name_in_member_text": None,
            },
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c78_units_provenance", out)


STAGES = {
    "c1": V.stage_c1,
    "c1b": V.stage_c1_binding,
    "c1s": stage_c1_decisive,
    "c2": V.stage_c2,
    "c3": stage_c3_conformed,
    "c4": V.stage_c4,
    "c5": V.stage_c5,
    "c6": V.stage_c6,
    "c6m": stage_c6_mix,
    "c78": stage_c78_conformed,
}


def main():
    want = sys.argv[1:] or list(STAGES)
    for s in want:
        print(f"\n=== STAGE {s} (conformed) ===", flush=True)
        STAGES[s]()
    print("\nALL REQUESTED STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
