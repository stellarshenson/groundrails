"""attr_pool CONFORMING PIPELINE - build the conformed member.

The verified member (experiments/grounding-semantic/R20-H174_lane_L2.parquet)
failed C2, C5, C6 and C8 of docs/experiments/dataset-contract.md.  This script
builds the conforming replacement ALONGSIDE it; nothing here overwrites the live
artifact.

WHAT THE FAILURES DEMAND
------------------------
C6  the mix's own `vitaminc` member pairs each of the lane's 6,894 distinct
    VitaminC claims with the very evidence sentence the lane's truth passage is
    assembled from, so a (claim -> supporting evidence) lookup separates 3,999
    truth_removed pairs at 0.9999 without reading the pool.  The association is
    unavoidable while the lane is sourced from VitaminC train AND VitaminC train
    is a mix member: every VitaminC lane claim is covered (coverage 1.0000).
    => the VitaminC half is DROPPED.  MiniCheck is not a mix member.
    MiniCheck claims are not free either - 605 of them are shared with the
    `frame_reject` lane of the same mix - so the pipeline additionally excludes
    EVERY claim that appears anywhere in the assembled mix, which makes the
    claim-keyed oracle undefined rather than merely at chance.
C2  one lane claim collided with the VitaminC held-out pool under the
    whitespace-collapsed case-folded form.  The pipeline pre-excludes any claim
    or document colliding with ANY evaluation surface on ANY of the three
    contract string forms, at build time, so the surface can never be reached.
C8  no retrieval date exists in tracked provenance for VitaminC.  With the
    VitaminC half dropped the member's only source is MiniCheck, whose fetch
    date IS recorded (2026-08-13, tracked sidecar).
C5  two separate matters.
    (i) the WITHIN-PAIR CLAIM-ONLY bar.  Dropping VitaminC exposed a leak the
    parent's aggregate hid: the parent reads 0.5594 on `unsupported_claim`, but
    that is the mean of a MiniCheck half at 0.6418 and a VitaminC half at
    0.4998, and a probe retrained on the MiniCheck half alone reads 0.6825
    (attr_pool_conformed_diag.json).  MiniCheck's generated label-0 claims are
    surface-separable from its label-1 claims; the bar is < 0.60 and no
    route restriction rescues it with margin (c2d 0.6872, d2c 0.5678 retrained).
    => the `unsupported_claim` FAMILY IS DROPPED.  What remains, truth_removed,
    holds the claim BYTE-IDENTICAL across the legs, so every claim-only probe on
    it is tied and reads exactly 0.5 by construction rather than by luck.
    (ii) the claim-to-chunk CONTAINMENT channel is NOT addressed here - it is a
    clause conflict measured in the verifier, not a defect a pipeline can build
    away.  See the verifier's frontier stage.

WHAT IS NOT CHANGED
-------------------
Every construction constant is carried over byte-identical: BM25Okapi top-40
candidate window, containment guard 0.75, 3-7 distractors (pool depth 4-8),
TRUTH_CAP 2 / DIST_CAP 12, PASSAGE_MAX_CHARS 1400, separator "\\n\\n", seed 2174.
The pair semantics of both families are unchanged.  The ONLY changes are the
dropped source and the two eligibility filters.

CPU ONLY.  GPUs 0/1/2 are running R20-H174 draws 2-4; nothing here touches them.

Run:  uv run python <this> --stage supply     # eligibility census, no build
      uv run python <this> --stage build      # writes the conformed parquet
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import collections
import hashlib
import importlib.util as _ilu
import json
import pathlib
import random
import re
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent

SEED = 2174
TAG = "attr_pool"
SEP = "\n\n"
CHUNK_MAX_CHARS = 1500
POOL_MIN, POOL_MAX = 3, 7
BM25_TOPK = 40
CONTAIN_MAX = 0.75
TRUTH_CAP, DIST_CAP = 2, 12
TARGET = {"truth_removed": 4_000, "unsupported_claim": 0}

OUT = EXP / "R20-H174_lane_L2_conformed.parquet"
MANIFEST = EXP / "R20-H174_lane_L2_conformed_manifest.json"

_WS = re.compile(r"\s+")


def wsfold(t):
    return _WS.sub(" ", t).strip().casefold()


def _mod(name, fname, folder=EXP):
    spec = _ilu.spec_from_file_location(name, folder / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", "R20-H174_lane_common.py")
L2 = _mod("h174l2", "R20-H174_lane_L2.py")
M = _mod("apmeasure", "attr_pool_contract_measure.py", folder=HERE)
GATE = _mod("provgate", "provenance_gate.py")

PASSAGE_MAX = C.PASSAGE_MAX_CHARS


# --------------------------------------------------------------------------- #
# the assembled mix, through the BANKED loader
# --------------------------------------------------------------------------- #
def load_mix():
    """(claims, chunks, labels, tags) for the whole live mix EXCEPT attr_pool -
    the 685,670-row clean public train plus the four other loaded lanes."""
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", "R10-H108_lane.py")
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: {len(claims)} rows, expected 685,670")
    print(f"  clean mix {len(claims)} rows", flush=True)
    for fname, group in (
        ("R17-H146_lane.parquet", "quant_misbind"),
        ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit"),
        ("R20-H174_lane_L1.parquet", "frame_reject"),
        ("R20-H174_lane_L4.parquet", "path_bind"),
    ):
        d = pl.read_parquet(EXP / fname)
        claims += d["claim"].to_list()
        chunks += d["chunk"].to_list()
        y = np.concatenate([y, d["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * d.height
    print(f"  mix minus attr_pool: {len(claims)} rows, {len(set(tags))} groups", flush=True)
    return claims, chunks, y, tags


def surface_forms():
    """Union of every evaluation surface's units in the three contract string
    forms.  Both sides (claims and evidence) of every surface."""
    S = M.eval_surfaces()
    raw, trunc, fold = set(), set(), set()
    per, blobs = {}, []
    for name, s in S.items():
        units = list(s["claims"]) + list(s["evidence"])
        per[name] = {"claims": len(s["claims"]), "evidence": len(s["evidence"])}
        for t in units:
            raw.add(t)
            trunc.add(t[:CHUNK_MAX_CHARS])
            fold.add(wsfold(t))
        blobs.append((name, "\n\n".join(GATE.normalize(t) for t in units)))
    print(f"  surfaces {sorted(S)}: {len(raw)} raw units", flush=True)
    # units shorter than the 8-gram order of the C4 instrument are invisible to
    # the census, so they are covered by exact SUBSTRING matching instead - the
    # coverage clause of C4, applied here as an admission filter
    return {"raw": raw, "trunc": trunc, "fold": fold, "blobs": blobs}, per


def eligibility():
    """Everything the build needs to decide what it may consume."""
    t0 = time.time()
    print("loading the assembled mix through the banked loader ...", flush=True)
    mclaims, mchunks, my, mtags = load_mix()
    mix_claim_raw = set(mclaims)
    mix_claim_fold = {wsfold(c) for c in mclaims}
    # chunk-side: exact strings (sha1 to keep it cheap) plus the atomic passages
    # of the four constructed lanes, which are built from the same corpora
    mix_chunk_sha = {hashlib.sha1(c.encode()).hexdigest() for c in mchunks}
    lane_parts = set()
    for c, t in zip(mchunks, mtags):
        if t in ("quant_misbind", "quant_scale_unit", "frame_reject", "path_bind"):
            lane_parts.update(c.split(SEP))
    lane_parts_fold = {wsfold(p) for p in lane_parts}
    # which mix chunks carry label 1, for the executor-added doc-keyed probe
    mix_pos_sha = {
        hashlib.sha1(c.encode()).hexdigest() for c, l in zip(mchunks, my.tolist()) if l >= 0.5
    }
    del mclaims, mchunks
    print(f"  mix maps built {time.time() - t0:.0f}s", flush=True)

    print("loading the evaluation surfaces ...", flush=True)
    SURF, per_surface = surface_forms()

    print("MiniCheck universe ...", flush=True)
    mc = C.minicheck().filter(pl.col("doc").str.len_chars() <= PASSAGE_MAX)
    docs = dict(mc.select(["doc_id", "doc"]).unique().iter_rows())
    rows = [
        {"claim": c, "doc_id": d, "label": int(y)}
        for c, d, y in mc.select(["claim", "doc_id", "label"]).iter_rows()
    ]
    print(f"  {len(docs)} documents <= {PASSAGE_MAX} chars, {len(rows)} claim rows", flush=True)

    def in_surface_substring(t):
        """For a unit below the 8-gram order the census cannot see, exact
        substring matching against every surface, on the census's own
        normalization."""
        n = GATE.normalize(t)
        if not n or len(n.split()) >= 8:
            return False
        return any(n in blob for _name, blob in SURF["blobs"])

    def claim_ok(c):
        f = wsfold(c)
        if c in mix_claim_raw or f in mix_claim_fold:
            return "in_mix"
        if c in SURF["raw"] or c[:CHUNK_MAX_CHARS] in SURF["trunc"] or f in SURF["fold"]:
            return "in_surface"
        if in_surface_substring(c):
            return "short_claim_occurs_inside_a_surface_text"
        return None

    def doc_ok(d):
        f = wsfold(d)
        if d in SURF["raw"] or d[:CHUNK_MAX_CHARS] in SURF["trunc"] or f in SURF["fold"]:
            return "in_surface"
        if in_surface_substring(d):
            return "short_document_occurs_inside_a_surface_text"
        return None

    claim_reject = collections.Counter()
    eligible_rows = []
    for r in rows:
        why = claim_ok(r["claim"])
        if why:
            claim_reject[why] += 1
            continue
        eligible_rows.append(r)
    doc_reject = collections.Counter()
    eligible_docs = {}
    doc_in_mix, doc_in_mix_pos = set(), set()
    for d, t in docs.items():
        why = doc_ok(t)
        if why:
            doc_reject[why] += 1
            continue
        eligible_docs[d] = t
        sha = hashlib.sha1(t.encode()).hexdigest()
        if sha in mix_chunk_sha or t in lane_parts or wsfold(t) in lane_parts_fold:
            doc_in_mix.add(d)
            if sha in mix_pos_sha or t in lane_parts:
                doc_in_mix_pos.add(d)
    # a claim row survives only if its document also survives
    eligible_rows = [r for r in eligible_rows if r["doc_id"] in eligible_docs]

    by_doc = collections.defaultdict(lambda: {0: [], 1: []})
    for r in eligible_rows:
        by_doc[r["doc_id"]][r["label"]].append(r)
    docs_with_pos = sum(1 for v in by_doc.values() if v[1])
    docs_with_both = sum(1 for v in by_doc.values() if v[0] and v[1])

    census = {
        "mix_rows_searched": int(len(my)),
        "mix_distinct_claims": len(mix_claim_raw),
        "surfaces": per_surface,
        "minicheck_documents_total": len(docs),
        "minicheck_claim_rows_total": len(rows),
        "claim_rows_rejected": dict(claim_reject),
        "documents_rejected": dict(doc_reject),
        "eligible_documents": len(eligible_docs),
        "eligible_claim_rows": len(eligible_rows),
        "eligible_documents_carrying_a_supported_claim": docs_with_pos,
        "eligible_documents_carrying_BOTH_a_supported_and_an_unsupported_claim": docs_with_both,
        "pair_ceiling_truth_removed_at_TRUTH_CAP_2": docs_with_pos * TRUTH_CAP,
        "pair_ceiling_unsupported_claim": docs_with_both,
        "eligible_documents_whose_text_also_appears_in_the_mix": len(doc_in_mix),
        "eligible_documents_appearing_in_the_mix_at_label_1": len(doc_in_mix_pos),
        "note": "a document appearing in the mix is NOT excluded - the clause-C6 "
        "test keys on the pair's claim, and every claim in the mix is excluded. "
        "The document-keyed variant is measured as an executor-added probe in the "
        "verifier and reported separately",
    }
    return eligible_docs, eligible_rows, census, doc_in_mix, doc_in_mix_pos


def stage_supply():
    _docs, _rows, census, dim, dimp = eligibility()
    (HERE / "attr_pool_conformed_supply.json").write_text(json.dumps(census, indent=2, default=float))
    print(json.dumps(census, indent=2, default=float), flush=True)
    (HERE / "attr_pool_conformed_docs_in_mix.json").write_text(
        json.dumps({"in_mix": sorted(dim), "in_mix_label1": sorted(dimp)})
    )


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def stage_build():
    eligible_docs, eligible_rows, census, dim, dimp = eligibility()
    rng = random.Random(SEED)
    print(f"=== conformed build, seed {SEED}, {len(eligible_docs)} eligible docs", flush=True)

    retr = L2.Retriever(eligible_docs, "minicheck_eligible")
    by_doc = collections.defaultdict(lambda: {0: [], 1: []})
    for r in eligible_rows:
        r = dict(r, group_key=r["doc_id"], source="minicheck")
        by_doc[r["doc_id"]][r["label"]].append(r)
    keys = sorted(by_doc)
    rng.shuffle(keys)

    rows, pid = [], 0
    truth_used, dist_used = collections.Counter(), collections.Counter()
    stats = collections.Counter()

    got_b = got_a = 0
    want_b, want_a = TARGET["unsupported_claim"], TARGET["truth_removed"]
    for did in keys:
        if got_b >= want_b:
            break
        d = by_doc[did]
        if not d[0] or not d[1]:
            continue
        pos_r, neg_r = rng.choice(d[1]), rng.choice(d[0])
        if truth_used[pos_r["doc_id"]] >= TRUTH_CAP:
            continue
        built = L2.build_pair("minicheck", pos_r, neg_r, "unsupported_claim", eligible_docs,
                              {}, retr, truth_used, dist_used, rng, pid)
        if built is None:
            stats["unsupported_claim_skipped"] += 1
            continue
        rows += built
        pid += 1
        got_b += 1
        if got_b % 500 == 0:
            print(f"  unsupported_claim {got_b}/{want_b}", flush=True)
    for did in keys:
        if got_a >= want_a:
            break
        d = by_doc[did]
        if not d[1]:
            continue
        pos_r = rng.choice(d[1])
        if truth_used[pos_r["doc_id"]] >= TRUTH_CAP:
            continue
        built = L2.build_pair("minicheck", pos_r, None, "truth_removed", eligible_docs,
                              {}, retr, truth_used, dist_used, rng, pid)
        if built is None:
            stats["truth_removed_skipped"] += 1
            continue
        rows += built
        pid += 1
        got_a += 1
        if got_a % 500 == 0:
            print(f"  truth_removed {got_a}/{want_a}", flush=True)
    print(f"  built truth_removed {got_a}/{want_a}, unsupported_claim {got_b}/{want_b}", flush=True)

    df = C.dedupe(pl.DataFrame(rows, infer_schema_length=None))
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}", flush=True)

    res = L2.verify(df, rng)
    man = L2.build_manifest(df, res, truth_used, dist_used, stats)
    man["experiment"] = "R20-H174 lane L2 CONFORMED - attr_pool rebuilt against " \
                        "docs/experiments/dataset-contract.md"
    man["conforming_pipeline"] = {
        "parent_artifact": "experiments/grounding-semantic/R20-H174_lane_L2.parquet",
        "parent_failures": ["C2", "C5", "C6", "C8"],
        "change_1_source": "VitaminC DROPPED entirely. It is a mix member "
        "(DANN group `vitaminc`), so the mix pairs every lane VitaminC claim with "
        "the evidence sentence the lane's truth passage is built from - the C6 "
        "channel. Dropping it also removes the C8 gap (no VitaminC retrieval date "
        "exists in tracked provenance) and the C2 collision (a VitaminC claim)",
        "change_2_claim_eligibility": "a MiniCheck claim is consumed only if its "
        "string appears NOWHERE in the assembled mix (raw and whitespace-collapsed "
        "case-folded) - this covers the 605 claims shared with the `frame_reject` "
        "lane - and collides with NO evaluation surface on any of the three "
        "contract string forms",
        "change_3_document_eligibility": "a MiniCheck document is consumed only if "
        "it collides with no evaluation surface on any of the three forms. A claim "
        "or document too short for the C4 8-gram instrument to see is additionally "
        "required not to occur as an exact SUBSTRING of any surface text - the "
        "parent's coverage check found one such claim, the generic 'He won the gold "
        "medal.', inside a hotpotqa arena chunk",
        "change_4_family": "the `unsupported_claim` family is DROPPED. On MiniCheck "
        "alone its within-pair claim-only probe reads 0.6418 in the parent lane and "
        "0.6825 when the probe is retrained on that half, against a C5 bar of "
        "< 0.60; the parent passed the bar at 0.5594 only because its VitaminC half "
        "read 0.4998 and the two were averaged. Per-route restriction does not "
        "rescue it with margin (c2d 0.6872, d2c 0.5678). The surviving family, "
        "truth_removed, is the registered source_select teacher and holds the claim "
        "byte-identical across the legs, so claim-only probes are tied at exactly "
        "0.5 by construction. Measured in attr_pool_conformed_diag.json",
        "unchanged": "BM25Okapi top-40, containment guard 0.75, 3-7 distractors, "
        "TRUTH_CAP 2 / DIST_CAP 12, PASSAGE_MAX_CHARS 1400, separator '\\n\\n', "
        "seed 2174, both family semantics, the mix loader contract",
        "eligibility_census": census,
    }
    man["sources"] = {"minicheck": C.SOURCES["minicheck"]}
    MANIFEST.write_text(json.dumps(man, indent=2, default=float))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "label_balance", "families", "source_rows",
                       "pool", "document_disjointness", "window_census", "verify")},
                     indent=2, default=float), flush=True)
    print(f"=== conformed build {'OK' if res['all_bars_pass'] else 'BARS FAILED'} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("supply", "build"))
    a = ap.parse_args()
    print(f"=== attr_pool conformed {a.stage}  {time.strftime('%F %T')}  "
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    {"supply": stage_supply, "build": stage_build}[a.stage]()
    print(f"=== {a.stage} DONE {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()
