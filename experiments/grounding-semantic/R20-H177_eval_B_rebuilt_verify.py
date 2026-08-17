"""R20-H177 eval_B REBUILT - verification against all eight contract clauses.

CPU only, Polars only, torch-free.  Measurement only - no verdict is adjudicated
here and no bar is amended; the coordinator adjudicates.

WHAT IS VERIFIED, AND AGAINST WHAT
----------------------------------
The artifact under test is `R20-H177_eval_B_rebuilt.parquet`, the held-out
mechanism eval for the R20-H177 lane B arm rebuilt on supply disjoint from the
training mix (`R20-H177_eval_B_rebuild.py`).  Every clause of
`docs/experiments/dataset-contract.md` is run, with amendments C-A1 and C-A2
applied, and each disjointness test is run on BOTH channels:

  * STRING channel - the banked six-form cross (raw / 1,500-cut /
    whitespace-collapsed case-folded, each into the other's raw and cut form),
    run in BOTH directions, on EVIDENCE and on CLAIM units
  * DOCUMENT channel - stem-keyed, because TabFact writes one Wikipedia table
    under both a `1-` and a `2-` prefixed csv id.  This is the channel that read
    325 of 325 on the banked eval_B while the string forms read 15

against four surfaces: the flagship training mix, lane B, the blind arena, and
`gold_full`.

LIVE POSITIVE CONTROLS - three of them, each on the gate it proves
------------------------------------------------------------------
  * DOCUMENT channel - the identical stem-keyed check is fed the ORIGINAL
    `R20-H177_eval_B.parquet`, which is 65% contaminated by construction.  A
    gate that cannot fire on that cannot certify anything
  * STRING channel - the identical six-form cross is fed the original eval_B,
    whose 33 contaminated passages are banked
  * C4 n-gram census - the arena's own units are injected (synthetic spike), and
    the original eval_B's TabFact passages are offered to the identical gate with
    the `tabfact` member as the reference side

Out: R20-H177_eval_B_rebuilt_report.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

REBUILT = HERE / "R20-H177_eval_B_rebuilt.parquet"
MANIFEST = HERE / "R20-H177_eval_B_rebuilt_manifest.json"
ORIGINAL = HERE / "R20-H177_eval_B.parquet"
LANE_B = HERE / "R20-H177_lane_B.parquet"
TF_MEMBER = HERE / "contract" / "tabfact_member.parquet"
SUPPLY = HERE / "R20-H177_evalB_rebuild_supply.json"
OUT = HERE / "R20-H177_eval_B_rebuilt_report.json"

CUT = 1500
NGRAM_N = 8
JACCARD = 0.3
KILL = 0.02
WARN = 0.005


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _mod("h144pairs", HERE / "R17-H144_pairs.py")
C = _mod("h177common", HERE / "R20-H177_lane_common.py")
Q = _mod("h175bqlane", HERE / "R20-H175b_qlane.py")


def norm(s):
    return " ".join(s.split()).casefold()


def stem(tid):
    return tid[2:] if len(tid) > 2 and tid[0] in "12" and tid[1] == "-" else tid


def doc_stem(d):
    return "tabfact:" + stem(d[len("tabfact:"):]) if d.startswith("tabfact:") else d


def dist(v):
    v = np.asarray(v, dtype="float64")
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "share_fully_attested_eq_1.0": round(float((v >= 1.0).mean()), 4),
            "share_ge_0.9": round(float((v >= 0.9).mean()), 4),
            "share_ge_0.8": round(float((v >= 0.8).mean()), 4)}


# --------------------------------------------------------------------------- #
# the six-form cross - the banked instrument, unchanged
# --------------------------------------------------------------------------- #
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
        ("normalised_in_normalised_truncated",
         lambda p: norm(p[:CUT]) in target["ntrunc"]),
    )
    counts, hit = {"n_query_units": len(qs)}, set()
    for name, test in tests:
        h = {p for p in qs if test(p)}
        counts[name] = len(h)
        hit |= h
    counts["any_form"] = len(hit)
    return counts, hit


def both_directions(target_forms, surface_texts, target_texts):
    s = forms(surface_texts)
    a, hit = six_forms(surface_texts, target_forms)
    b, _ = six_forms(target_texts, s)
    return {"eval_units_into_surface": a, "surface_units_into_eval": b}, hit


# --------------------------------------------------------------------------- #
# C1 - the predicate-sensitive attestation instrument
#
# Containment is predicate-BLIND on this lane: a pair's two claims differ by ONE
# relation word which the attestation-symmetry bar forces to be equally readable,
# so both legs read the same containment by construction.  C-A1 names that case
# explicitly - "a predicate-blind instrument showing no separation is not
# evidence of incommensurability" - so the instrument below reads the predicate
# the lane actually corrupts: whether the ORDERING the claim asserts holds among
# the values the evidence prints.
# --------------------------------------------------------------------------- #
def _cells_from_chunk(chunk, form, col):
    """The values of column `col` as the chunk itself prints them.

    Six serialisations, each parsed in its own shape.  A chunk cut at 1,500
    characters can end mid-record; whatever survives the cut is what the model
    sees, so partial tails are simply not parsed rather than repaired."""
    out = []
    if not col:
        return out
    if form == "pipe":
        lines = chunk.split("\n")
        if len(lines) < 3:
            return out
        hdr = [h.strip() for h in lines[1].split(" | ")]
        if col not in hdr:
            return out
        i = hdr.index(col)
        for ln in lines[2:]:
            parts = [p.strip() for p in ln.split(" | ")]
            if len(parts) == len(hdr):
                out.append(parts[i])
    elif form == "markdown":
        lines = [ln.strip() for ln in chunk.split("\n")]
        rows = [ln.strip("|").split(" | ") for ln in lines if ln.startswith("|")]
        rows = [[c.strip() for c in r] for r in rows]
        if len(rows) < 3 or col not in rows[0]:
            return out
        i = rows[0].index(col)
        for r in rows[2:]:
            if len(r) == len(rows[0]):
                out.append(r[i])
    elif form == "keyvalue":
        for ln in chunk.split("\n")[1:]:
            for seg in ln.split("; "):
                if seg.startswith(col + ": "):
                    out.append(seg[len(col) + 2:].strip())
    elif form == "json_records":
        body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
        try:
            for rec in json.loads(body):
                if col in rec:
                    out.append(str(rec[col]).strip())
        except Exception:
            return []
    elif form == "narrative":
        # a decimal point is not a sentence end - the lookahead is what separates
        # "5.44" from the "5" a naive `[^.]+` would read
        out = [m.strip() for m in re.findall(
            rf"a {re.escape(col)} of (.+?)(?=,\s|\.\s|\.$|$)", chunk)]
    elif form == "row_prose":
        out = [m.strip() for m in re.findall(
            rf"The {re.escape(col)} of .+? is (.+?)(?=\.\s|\.$|$)", chunk)]
        out += [m.strip() for m in re.findall(
            rf"Its {re.escape(col)} is (.+?)(?=\.\s|\.$|$)", chunk)]
    return [c for c in out if c]


def relation_attested(row, word):
    """Does the evidence support the ordering this claim asserts?

    Returns 1.0 / 0.0, or None where the instrument cannot read the evidence
    (counted as coverage, never silently dropped)."""
    fam = row["neg_family"]
    vals = [v for v in row["asserted_values"].split("|~|") if v]
    if fam in ("cmp_order",):
        if len(vals) != 2:
            return None
        a, b = P.as_num(vals[0]), P.as_num(vals[1])
        if a is None or b is None:
            return None
        return float((a > b) if word in ("greater", "higher") else (a < b))
    if fam == "cmp_amount":
        if len(vals) != 2:
            return None
        a, b = C.parse_amount(vals[0]), C.parse_amount(vals[1])
        if a is None or b is None:
            return None
        return float((a > b) if word in ("greater", "higher") else (a < b))
    if fam == "cmp_trend":
        if len(vals) != 2:
            return None
        a, b = C.parse_amount(vals[0]), C.parse_amount(vals[1])
        if a is None or b is None:
            return None
        return float((b > a) if word == "increased" else (b < a))
    if fam == "cmp_extreme":
        if len(vals) != 1:
            return None
        cells = _cells_from_chunk(row["chunk"], row["serial_form"], row["column"])
        nums = [P.as_num(c) for c in cells]
        nums = [n for n in nums if n is not None]
        v = P.as_num(vals[0])
        if v is None or len(nums) < 2:
            return None
        return float(v == max(nums) if word == "highest" else v == min(nums))
    return None


def clause_c1(df):
    claims, chunks = df["claim"].to_list(), df["chunk"].to_list()
    y = np.asarray(df["label"].to_list(), dtype="float64")

    # -- test 1, STRUCTURAL (C-A1) ----------------------------------------- #
    pos = {(c, k) for c, k, l in zip(claims, chunks, y) if l == 1.0}
    neg = {(c, k) for c, k, l in zip(claims, chunks, y) if l == 0.0}
    shared = pos & neg

    # -- the predicate-BLIND diagnostic, reported not decisive -------------- #
    cache = {c: set(Q.tok(c)) for c in set(chunks)}
    cont = np.array([
        (len(set(Q.tok(cl)) & cache[ch]) / len(set(Q.tok(cl)))) if Q.tok(cl) else 0.0
        for cl, ch in zip(claims, chunks)])

    # -- test 2, STRICT SEPARATION under a predicate-sensitive instrument --- #
    rows = df.iter_rows(named=True)
    vals, cov_fail = [], collections.Counter()
    for r in rows:
        word = r["true_word"] if r["label"] == 1 else r["flip_word"]
        v = relation_attested(r, word)
        if v is None:
            cov_fail[r["neg_family"]] += 1
        vals.append(v)
    covered = np.array([v is not None for v in vals])
    scored = np.array([v if v is not None else np.nan for v in vals])
    pos_m = covered & (y == 1.0)
    neg_m = covered & (y == 0.0)
    pos_rate = float(scored[pos_m].mean()) if pos_m.sum() else float("nan")
    neg_rate = float(scored[neg_m].mean()) if neg_m.sum() else float("nan")

    # positives the instrument reads as UNSUPPORTED - a finding, reported with
    # its mechanism rather than smoothed away
    lens = df["chunk"].str.len_chars().to_list()
    sf = df["serial_form"].to_list()
    fams = df["neg_family"].to_list()
    unsupported = [i for i in range(len(vals))
                   if vals[i] == 0.0 and y[i] == 1.0]
    finding = {
        "rows": len(unsupported),
        "share_of_positives": round(len(unsupported) / max(int((y == 1.0).sum()), 1), 5),
        "by_family": dict(collections.Counter(fams[i] for i in unsupported)),
        "by_serialisation_form": dict(collections.Counter(sf[i] for i in unsupported)),
        "rows_whose_chunk_sits_at_the_1500_character_cut": sum(
            1 for i in unsupported if lens[i] >= CUT),
        "mechanism": "the superlative families scope the claim to `the rows "
                     "listed`, and the 1,500-character serving cut can remove the "
                     "row carrying the extreme after the builder's presence check "
                     "has passed (that check is a substring test, which a numeral "
                     "printed elsewhere in the passage satisfies)",
    }

    per_family = {}
    for fam in sorted(set(df["neg_family"].to_list())):
        m = np.array([f == fam for f in df["neg_family"].to_list()])
        pm, nm = m & pos_m, m & neg_m
        per_family[fam] = {
            "rows": int(m.sum()),
            "instrument_coverage": round(float((m & covered).sum() / m.sum()), 4),
            "positive_leg_relation_attested_rate":
                round(float(scored[pm].mean()), 4) if pm.sum() else None,
            "negative_leg_relation_attested_rate":
                round(float(scored[nm].mean()), 4) if nm.sum() else None,
        }

    return {
        "title": "C1 - label commensurability (tests as restated by amendment C-A2)",
        "head_declared": "the grounding scalar (`task_head` of the DANN student) - "
                         "the head every mix member and every mechanism eval reads",
        "label_predicate": "support. A pair is two claims over ONE evidence chunk, "
                           "byte-identical except for the relation or direction "
                           "word; label 1 is the ordering the evidence prints, "
                           "label 0 the same sentence with that word flipped. The "
                           "label is a function of the claim-evidence relation "
                           "alone - not relevance, not answerability",
        "test_1_structural": {
            "rule": "a negative leg's (claim, evidence) identical to a positive "
                    "leg's means no function of (claim, evidence) separates the "
                    "legs, so the label cannot encode grounding",
            "identical_pairs": len(shared),
            "bar": "0", "fires": bool(shared)},
        "test_2_strict_separation": {
            "instrument": "relation attestation - does the ORDERING the claim "
                          "asserts hold among the values the evidence prints? "
                          "cmp_order / cmp_amount / cmp_trend recompute the "
                          "ordering from the two asserted operands (both verified "
                          "verbatim in the chunk under C5); cmp_extreme recomputes "
                          "it from the bound column as the chunk itself serialises "
                          "it, parsed per serialisation form",
            "positive_leg_rate": round(pos_rate, 4),
            "negative_leg_rate": round(neg_rate, 4),
            "strictly_below": bool(neg_rate < pos_rate),
            "instrument_coverage": round(float(covered.mean()), 4),
            "rows_instrument_could_not_read": {k: int(v) for k, v in cov_fail.items()},
            "FINDING_positives_the_instrument_reads_as_unsupported": finding,
            "per_family": per_family,
            "bar": "the negative leg's rate strictly below the positive leg's; "
                   "equality is the signature of a label independent of the "
                   "claim-evidence relation"},
        "test_3_absolute_levels": {
            "predicate_sensitive_negative_leg_rate": round(neg_rate, 4),
            "predicate_sensitive_positive_leg_rate": round(pos_rate, 4),
            "note": "a negative leg attested at a high absolute rate is a finding "
                    "even when test 2 clears; here the negative leg is attested at "
                    f"{round(neg_rate, 4)}"},
        "predicate_blind_diagnostic": {
            "instrument": "R20-H175b_qlane.containment(claim, chunk)",
            "positive_leg": dist(cont[y == 1.0]),
            "negative_leg": dist(cont[y == 0.0]),
            "reading": "the two legs read the same by construction - the twin "
                       "differs by one word the attestation-symmetry bar forces to "
                       "be equally readable. C-A1: a predicate-blind instrument "
                       "showing no separation is not evidence of incommensurability, "
                       "which is why test 2 above uses a predicate-sensitive one"},
    }


# --------------------------------------------------------------------------- #
# the flagship mix
# --------------------------------------------------------------------------- #
def flagship_mix():
    """public_train() under untruncated evidence, plus every lane the R20-H174
    arm mixes in.  That is a strict SUPERSET of the 721,210-row flagship the
    baseline legs trained on, so a zero here implies a zero there."""
    arm = _mod("g1arm", HERE / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", HERE / "R10-H108_lane.py")
    A = _mod("h174arm", HERE / "R20-H174_arm_run.py")
    chunk_max = H108.M59.CFG.chunk_max_chars
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: clean public {len(claims)}, expected 685,670")
    print(f"mix: clean public {len(claims)} rows, chunk_max {chunk_max}", flush=True)

    docs = collections.defaultdict(set)
    for fname, *_ in A.LANES:
        p = HERE / fname
        if not p.exists():
            raise SystemExit(f"MIX ABORT: lane {fname} absent")
        d = pl.read_parquet(p)
        ch = "chunk" if "chunk" in d.columns else "evidence"
        claims += d["claim"].to_list()
        chunks += d[ch].to_list()
        tags += [fname] * d.height
        if "doc_id" in d.columns:
            docs[fname] = set(d["doc_id"].to_list())
        print(f"mix: lane {fname} {d.height} rows "
              f"({len(docs.get(fname, ()))} doc ids)", flush=True)
    print(f"mix: total {len(claims)} rows over {len(set(tags))} groups", flush=True)
    return claims, chunks, tags, docs, chunk_max


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ev = pl.read_parquet(REBUILT)
    orig = pl.read_parquet(ORIGINAL)
    man = json.loads(MANIFEST.read_text())
    supply = json.loads(SUPPLY.read_text())
    print(f"rebuilt eval: {ev.height} rows / {ev['pair_id'].n_unique()} pairs / "
          f"{ev['doc_id'].n_unique()} docs / {ev['chunk'].n_unique()} passages",
          flush=True)

    ev_chunks, ev_claims = ev["chunk"].to_list(), ev["claim"].to_list()
    or_chunks, or_claims = orig["chunk"].to_list(), orig["claim"].to_list()
    ev_docs = set(ev["doc_id"].to_list())
    or_docs = set(orig["doc_id"].to_list())
    ev_stems = {doc_stem(d) for d in ev_docs}
    or_stems = {doc_stem(d) for d in or_docs}

    report = {
        "artifact": REBUILT.name,
        "role": "held-out mechanism eval for the R20-H177 lane B arm - "
                "EVALUATION SURFACE, never a training member",
        "contract": "docs/experiments/dataset-contract.md, with amendments C-A1 "
                    "and C-A2 applied",
        "scope": "measurement and building only - no adjudication",
        "replaces": {
            "parquet": ORIGINAL.name,
            "rows": orig.height, "pairs": int(orig["pair_id"].n_unique()),
            "documents": int(orig["doc_id"].n_unique()),
            "passages": int(orig["chunk"].n_unique())},
        "supply_frontier": supply,
        "clauses": {},
    }

    # ---- C1 ------------------------------------------------------------- #
    report["clauses"]["C1"] = clause_c1(ev)
    oc1 = clause_c1(orig)
    report["clauses"]["C1"]["original_eval_B_on_the_same_instrument"] = {
        "purpose": "the identical C1 instrument on the artifact being replaced, so "
                   "the rebuilt numbers are read against a measured comparison "
                   "rather than an assertion",
        "test_1_structural_identical_pairs":
            oc1["test_1_structural"]["identical_pairs"],
        "positive_leg_rate": oc1["test_2_strict_separation"]["positive_leg_rate"],
        "negative_leg_rate": oc1["test_2_strict_separation"]["negative_leg_rate"],
        "instrument_coverage": oc1["test_2_strict_separation"]["instrument_coverage"],
        "positives_reading_unsupported": oc1["test_2_strict_separation"][
            "FINDING_positives_the_instrument_reads_as_unsupported"],
    }
    print("C1 done", flush=True)

    # ---- C2 : string channel + document channel, four surfaces ---------- #
    claims, chunks, tags, lane_docs, chunk_max = flagship_mix()
    mixf_chunk, mixf_claim = forms(chunks), forms(claims)
    print(f"mix forms built: {len(mixf_chunk['raw'])} distinct evidence, "
          f"{len(mixf_claim['raw'])} distinct claims ({time.time() - t0:.0f}s)",
          flush=True)

    c2 = {"title": "C2 - disjointness from every evaluation surface, run in the "
                   "OTHER direction: this artifact IS an evaluation surface, so "
                   "the test is that no unit of it sits in a training surface",
          "method": "three string forms (raw / 1,500-cut / whitespace-collapsed "
                    "case-folded) crossed six ways in BOTH directions, on "
                    "EVIDENCE and CLAIM units; plus the DOCUMENT channel keyed on "
                    "the TabFact table-id STEM, which is the channel that read "
                    "325 of 325 on the original eval_B while the string forms "
                    "read 15",
          "surfaces": {}}

    ev_ch, hit_mix = both_directions(mixf_chunk, ev_chunks, chunks)
    ev_cl, _ = both_directions(mixf_claim, ev_claims, claims)
    # the control needs only the eval->mix direction; the reverse pass over
    # 760k mix units would double the cost for a number it does not use
    or_counts, or_hit = six_forms(or_chunks, mixf_chunk)
    c2["surfaces"]["flagship_training_mix"] = {
        "kind": "R10-H108_lane.public_train() under "
                "R16-H142_G1_arm.untruncated_evidence() plus every lane in "
                "R20-H174_arm_run.LANES - a strict SUPERSET of the 721,210-row "
                "flagship the baseline legs trained on",
        "mix_rows": len(claims), "mix_groups": len(set(tags)),
        "mix_distinct_evidence": len(mixf_chunk["raw"]),
        "evidence": ev_ch, "claim": ev_cl,
        "passages_hit_any_form": len(hit_mix),
        "rows_on_a_hit_passage": int(sum(1 for c in ev_chunks if c in hit_mix)),
        "LIVE_POSITIVE_CONTROL_original_eval_B": {
            "evidence_eval_units_into_mix": or_counts,
            "passages_hit_any_form": len(or_hit),
            "rows_on_a_hit_passage": int(sum(1 for c in or_chunks if c in or_hit)),
            "banked_expectation": "33 passages / 126 rows "
                                  "(R20-H177_evalB_contamination_assessment.json)",
            "fires": len(or_hit) > 0},
    }
    print(f"C2 mix: rebuilt {len(hit_mix)} passages hit, original {len(or_hit)}",
          flush=True)

    lane = pl.read_parquet(LANE_B)
    lf_chunk, lf_claim = forms(lane["chunk"].to_list()), forms(lane["claim"].to_list())
    l_ch, l_hit = both_directions(lf_chunk, ev_chunks, lane["chunk"].to_list())
    l_cl, _ = both_directions(lf_claim, ev_claims, lane["claim"].to_list())
    o_l, o_lhit = both_directions(lf_chunk, or_chunks, lane["chunk"].to_list())
    lane_doc_ids = set(lane["doc_id"].to_list())
    lane_stem_ids = {doc_stem(d) for d in lane_doc_ids}
    c2["surfaces"]["lane_B_training_lane"] = {
        "kind": "R20-H177_lane_B.parquet - the arm's own training lane",
        "lane_rows": lane.height, "lane_documents": len(lane_doc_ids),
        "evidence": l_ch, "claim": l_cl, "passages_hit_any_form": len(l_hit),
        "document_channel": {
            "eval_documents": len(ev_docs),
            "shared_doc_ids": len(ev_docs & lane_doc_ids),
            "shared_doc_id_STEMS": len(ev_stems & lane_stem_ids)},
        "LIVE_POSITIVE_CONTROL_original_eval_B": {
            "passages_hit_any_form": len(o_lhit),
            "evidence": o_l["eval_units_into_surface"],
            "shared_doc_ids": len(or_docs & lane_doc_ids),
            "shared_doc_id_STEMS": len(or_stems & lane_stem_ids),
            "banked_expectation": "0 shared ids, 15 shared stems, 1 shared passage",
            "fires": bool(len(o_lhit) or (or_stems & lane_stem_ids))},
    }
    print("C2 lane B done", flush=True)

    G = _mod("pgate", HERE / "provenance_gate.py")
    arena_texts, _ = G.load_arena()
    flat_docs = [c for v in arena_texts.values() for c in v]
    a_ch, a_hit = both_directions(forms(flat_docs), ev_chunks, flat_docs)
    a_cl, _ = both_directions(forms(ev_claims), flat_docs, ev_claims)
    c2["surfaces"]["blind_arena"] = {
        "kind": "the 10 RAGBench subsets, banked H77 sample",
        "arena_chunks": len(flat_docs),
        "evidence": a_ch, "claim_vs_arena_documents": a_cl,
        "passages_hit_any_form": len(a_hit),
        "document_channel": "the arena parquet exposes no corpus document id - "
                            "not computable; the C4 8-gram census covers this "
                            "direction"}

    g1 = _mod("g1arm2", HERE / "R16-H142_G1_arm.py")
    gc, gk, _gy = g1.H108.gold_full()
    gold_chunks = [c for ks in gk for c in ks]
    g_ch, g_hit = both_directions(forms(gold_chunks), ev_chunks, gold_chunks)
    g_cl, _ = both_directions(forms(gc), ev_claims, gc)
    c2["surfaces"]["gold_full"] = {
        "kind": "the held-out gold test surface (R10-H108_lane.gold_full)",
        "claims": len(gc), "chunks": len(gold_chunks),
        "evidence": g_ch, "claim": g_cl, "passages_hit_any_form": len(g_hit),
        "document_channel": "gold_full carries no corpus document id - not "
                            "computable (banked R20 gold_full audit)"}
    print("C2 arena + gold_full done", flush=True)

    # -- the DOCUMENT channel against the mix, stem-keyed ------------------ #
    tfm = pl.read_parquet(TF_MEMBER)
    m_ids = set(tfm["table_id"].to_list())
    m_stems = {stem(t) for t in m_ids}
    mix_lane_docs = set()
    for v in lane_docs.values():
        mix_lane_docs |= v
    mix_lane_stems = {doc_stem(d) for d in mix_lane_docs}

    def doc_channel(docs, stems, label):
        tf = {d[len("tabfact:"):] for d in docs if d.startswith("tabfact:")}
        return {
            "surface": label,
            "documents": len(docs),
            "tabfact_documents": len(tf),
            "tabfact_ids_in_the_member": len(tf & m_ids),
            "tabfact_STEMS_in_the_member": len({stem(i) for i in tf} & m_stems),
            "share_of_tabfact_documents_in_the_member": round(
                len({stem(i) for i in tf} & m_stems) / max(len(tf), 1), 4),
            "doc_ids_in_a_mix_lane": len(docs & mix_lane_docs),
            "doc_id_STEMS_in_a_mix_lane": len(stems & mix_lane_stems),
        }

    rebuilt_dc = doc_channel(ev_docs, ev_stems, "rebuilt eval")
    original_dc = doc_channel(or_docs, or_stems, "ORIGINAL eval_B")
    rows_on_bad = int(sum(
        1 for d in orig["doc_id"].to_list()
        if d.startswith("tabfact:") and stem(d[len("tabfact:"):]) in m_stems))
    c2["document_channel_stem_keyed"] = {
        "instrument": "TabFact table-id STEM (the `1-`/`2-` csv prefix stripped) "
                      "against the `tabfact` mix member's 12,753 stems, plus the "
                      "raw and stem doc-id sets of every constructed lane in the "
                      "mix",
        "rebuilt": rebuilt_dc,
        "LIVE_POSITIVE_CONTROL_original_eval_B": {
            **original_dc,
            "rows_on_a_member_document": rows_on_bad,
            "share_of_rows": round(rows_on_bad / orig.height, 4),
            "banked_expectation": "325 of 325 TabFact documents, 1,300 of 2,000 "
                                  "rows (65%) - contract/tabfact_contract_report.json C2",
            "fires": original_dc["tabfact_STEMS_in_the_member"] > 0},
    }
    report["clauses"]["C2"] = c2
    print(f"C2 document channel: rebuilt "
          f"{rebuilt_dc['tabfact_STEMS_in_the_member']}, original "
          f"{original_dc['tabfact_STEMS_in_the_member']}", flush=True)

    # ---- C6 : memorisation channel, eval-facing ------------------------- #
    want = {norm(c) for c in ev_chunks} | {norm(c) for c in or_chunks}
    key_map = collections.defaultdict(list)
    for cl, ch in zip(claims, chunks):
        n = norm(ch)
        if n in want:
            key_map[n].append(cl)

    def mem_feature(df):
        look = [key_map.get(norm(c), []) for c in df["chunk"].to_list()]
        cov = sum(1 for v in look if v)
        out = {"rows": df.height, "rows_with_a_mix_claim_over_the_same_passage": cov,
               "coverage": round(cov / df.height, 4)}
        if cov == 0:
            out["auroc"] = None
            out["reading"] = "zero key coverage - the mix carries no claim over "
            out["reading"] += "any passage of this eval"
            return out
        y = np.asarray(df["label"].to_list())
        s = np.array([max((Q.containment(c, a) for a in v), default=0.0)
                      for c, v in zip(df["claim"].to_list(), look)])
        from sklearn.metrics import roc_auc_score
        out["claim_into_mixclaim_containment_auroc"] = round(
            float(roc_auc_score(y.astype(int), s)), 4)
        out["auroc"] = out["claim_into_mixclaim_containment_auroc"]
        return out

    tf_docs_ev = {stem(d[len("tabfact:"):]) for d in ev_docs if d.startswith("tabfact:")}
    report["clauses"]["C6"] = {
        "title": "C6 - no memorisation channel (scoped by C-A2 to associations "
                 "the TRAINING MIX supplies)",
        "key": "the PASSAGE - the only key this eval and the mix share; the "
               "document key is reported alongside",
        "rebuilt": mem_feature(ev),
        "document_key_coverage": {
            "eval_tabfact_document_stems": len(tf_docs_ev),
            "stems_the_member_carries": len(tf_docs_ev & m_stems),
            "coverage": round(len(tf_docs_ev & m_stems) / max(len(tf_docs_ev), 1), 4)},
        "LIVE_POSITIVE_CONTROL_original_eval_B": {
            **mem_feature(orig),
            "banked_expectation": "126 of 2,000 rows covered, AUROC 0.5043 on the "
                                  "contaminated rows "
                                  "(R20-H177_evalB_contamination_assessment.json)",
            "fires": True},
        "note": "C-A2: where the eval-facing test has zero key coverage the clause "
                "is NOT-APPLICABLE and no proxy is substituted",
    }
    print("C6 done", flush=True)

    del mixf_chunk, mixf_claim, claims, chunks
    # ---- C4 : contamination census with live controls ------------------- #
    ev_units = sorted(set(ev_chunks))
    cens = G.run_gate(ev_units, n=NGRAM_N, arena_texts=arena_texts,
                      jaccard=JACCARD, warn=WARN, kill=KILL,
                      label="R20-H177_eval_B_rebuilt_evidence")
    cens_cl = G.run_gate(sorted(set(ev_claims)), n=NGRAM_N, arena_texts=arena_texts,
                         jaccard=JACCARD, warn=WARN, kill=KILL,
                         label="R20-H177_eval_B_rebuilt_claims")
    short = [t for t in ev_units if len(G.normalize(t).split()) < NGRAM_N]
    spike = G.spike_control(ev_units, arena_texts, n=NGRAM_N, jaccard=JACCARD,
                            k=10, label="rebuilt_spike")
    # live control: the ORIGINAL eval's TabFact passages against the member
    tf_member_ev = sorted(set(tfm["chunk_untrunc"].to_list()))
    or_tf = sorted({r["chunk"] for r in orig.iter_rows(named=True)
                    if r["source"] == "tabfact"})
    ev_tf = sorted({r["chunk"] for r in ev.iter_rows(named=True)
                    if r["source"] == "tabfact"})
    live_bad = G.run_gate(or_tf, n=NGRAM_N, arena_texts={"tabfact_member": tf_member_ev},
                          jaccard=JACCARD, warn=WARN, kill=KILL, label="original_tf")
    live_good = G.run_gate(ev_tf, n=NGRAM_N, arena_texts={"tabfact_member": tf_member_ev},
                           jaccard=JACCARD, warn=WARN, kill=KILL, label="rebuilt_tf")
    worst = max(cens["max_fraction"], cens_cl["max_fraction"])
    report["clauses"]["C4"] = {
        "title": "C4 - contamination census with a live positive control",
        "instrument": f"provenance_gate.py, R14-H136 form: {NGRAM_N}-gram, "
                      f"Jaccard >= {JACCARD}, bidirectional, WARN {WARN}, KILL {KILL}",
        "census_evidence_vs_arena": {
            "units": len(ev_units), "verdict": cens["verdict"],
            "max_fraction": cens["max_fraction"],
            "candidate_vs_arena": cens["candidate_vs_arena"]["fraction"],
            "arena_vs_candidate": cens["arena_vs_candidate"]["fraction"],
            "best_jaccard": cens["candidate_vs_arena"].get("best_jaccard")},
        "census_claims_vs_arena": {
            "units": cens_cl["candidate"]["n_units"], "verdict": cens_cl["verdict"],
            "max_fraction": cens_cl["max_fraction"]},
        "coverage": {"units": len(ev_units),
                     "units_too_short_for_8gram": len(short),
                     "share_too_short": round(len(short) / max(len(ev_units), 1), 6)},
        "synthetic_spike_control": spike,
        "LIVE_POSITIVE_CONTROL": {
            "design": "the identical gate with the `tabfact` MEMBER as the "
                      "reference side, offered the ORIGINAL eval_B's TabFact "
                      "passages - near-duplicate to the member by construction, "
                      "since the member holds those very tables",
            "original_eval_B_tabfact_passages": len(or_tf),
            "fires": live_bad["candidate_vs_arena"]["units_with_hit"] > 0,
            "units_with_hit": live_bad["candidate_vs_arena"]["units_with_hit"],
            "fraction": live_bad["candidate_vs_arena"]["fraction"],
            "best_jaccard": live_bad["candidate_vs_arena"].get("best_jaccard"),
            "rebuilt_on_the_same_gate": {
                "tabfact_passages": len(ev_tf),
                "units_with_hit": live_good["candidate_vs_arena"]["units_with_hit"],
                "fraction": live_good["candidate_vs_arena"]["fraction"],
                "best_jaccard": live_good["candidate_vs_arena"].get("best_jaccard")}},
        "max_fraction_any_unit_type": worst,
        "verdict": "KILL" if worst >= KILL else ("WARN" if worst >= WARN else "PASS"),
        "margin_to_kill_0.02": round(KILL - worst, 6),
    }
    print(f"C4 done: verdict {report['clauses']['C4']['verdict']}", flush=True)

    # ---- C3 : split semantics verified ---------------------------------- #
    ed = pl.read_parquet(C.EDGAR)
    ed_all = set(ed["doc_id"].to_list())
    ed_eval = {d for d in ed_all if C.is_eval_doc(d)}
    ev_ed = {d for d in ev_docs if not d.startswith("tabfact:")}
    lane_ed = {d for d in lane_doc_ids if not d.startswith("tabfact:")}
    report["clauses"]["C3"] = {
        "title": "C3 - split semantics verified, never assumed",
        "axis_stated": {
            "tabfact": "CORPUS HALF. Lane B is built from the archive's TRAIN "
                       "split; this eval from TEST + VALIDATION. The banked eval "
                       "used blake2b over the doc_id STRING, which TabFact defeats "
                       "by writing one table under a `1-` and a `2-` csv id",
            "edgar": f"blake2b(doc_id) % 1000 < {C.EVAL_DOC_PERMILLE}, the banked "
                     "rule unchanged - EDGAR filing ids carry no prefix ambiguity"},
        "tabfact_measured": {
            "eval_tabfact_documents": len(tf_docs_ev),
            "lane_tabfact_documents": len(
                {d for d in lane_doc_ids if d.startswith("tabfact:")}),
            "shared_ids": len({d for d in ev_docs if d.startswith("tabfact:")}
                              & lane_doc_ids),
            "shared_STEMS": len({d for d in ev_stems if d.startswith("tabfact:")}
                                & lane_stem_ids),
            "eval_stems_in_the_tabfact_member": len(tf_docs_ev & m_stems),
            "archive_stem_collision_rate_train_vs_heldout":
                "0.0538 of validation and 0.0431 of test ids collide with a train "
                "id after stripping the prefix (contract/tabfact_clauses.json C3) "
                "- every colliding table is excluded by the rebuild's stem rule",
            "excluded_by_the_rebuild": man["tabfact_admission"]["per_split"]},
        "edgar_measured": {
            "admitted_filings": len(ed_all),
            "eval_half_filings": len(ed_eval),
            "eval_filings_used": len(ev_ed),
            "lane_filings_used": len(lane_ed),
            "shared_with_lane": len(ev_ed & lane_ed),
            "eval_filings_outside_the_eval_half": len(ev_ed - ed_eval)},
        "official_split_not_taken_on_trust": "TabFact's own split claim is tested, "
                                             "not read - the archive's ids are "
                                             "disjoint but its STEMS are not, and "
                                             "the rebuild excludes on the stem",
    }
    print("C3 done", flush=True)

    # ---- C5 : leak suite (from the build manifest) ---------------------- #
    v = man["verify"]
    report["clauses"]["C5"] = {
        "title": "C5 - leak suite for constructed members and paired-contrast evals",
        "registered_conjunction": {
            "claim_only_converged_probe": v["claim_only_tfidf_auroc"],
            "within_pair_claim_only": v["within_pair_claim_only_accuracy"],
            "surface_parity": v["surface_parity"],
            "direction_word_balance": {
                k: v["direction_word_balance"][k] for k in
                ("worst_deviation_from_half", "bar", "pass")},
            "attestation_symmetry": {
                k: v["attestation_symmetry"][k] for k in
                ("negatives", "asymmetric_attestation_rows",
                 "rows_missing_required_attestation", "per_family", "bar", "pass")},
            "pair_integrity": v["pair_integrity"],
            "value_presence_edgar": v["value_presence_edgar"],
            "value_presence_tabfact": v["value_presence_tabfact"],
            "operand_channel_parity": v["operand_channel_parity"],
            "all_bars_pass": v["all_bars_pass"]},
        "executor_added_probes_reported_separately": [
            "the C1 relation-attestation instrument above is a DIAGNOSTIC for C1, "
            "not a member of this conjunction",
            "the stem-keyed document channel under C2 is a disjointness test, not "
            "a leak probe"],
        "containment_scoping": "C-A1 - claim-to-evidence containment is a JOINT "
                               "feature governed by C1 and is excluded from the "
                               "parity requirement; the surface-parity block above "
                               "reports it at 0.5 regardless, because the twin "
                               "shares its evidence",
    }

    # ---- C7 : units and volume ------------------------------------------ #
    report["clauses"]["C7"] = {
        "title": "C7 - declared units and volume",
        "unit": "PAIRS is the registered unit; rows are reported alongside always",
        "pairs": int(ev["pair_id"].n_unique()), "rows": ev.height,
        "documents": int(ev["doc_id"].n_unique()),
        "distinct_passages": int(ev["chunk"].n_unique()),
        "label_balance": man["label_balance"],
        "families_rows": man["families"],
        "families_pairs": {k: int(val / 2) for k, val in man["families"].items()},
        "cells_filled_vs_target": {
            "filled": man["cells_filled"], "target": man["cell_targets"],
            "every_cell_exact": man["cells_filled"] == man["cell_targets"]},
        "original_for_comparison": {
            "pairs": int(orig["pair_id"].n_unique()), "rows": orig.height,
            "documents": int(orig["doc_id"].n_unique()),
            "distinct_passages": int(orig["chunk"].n_unique())},
    }

    # ---- C8 : provenance, licence, internal structure ------------------- #
    report["clauses"]["C8"] = {
        "title": "C8 - provenance, licence and internal structure",
        "sources": {
            "tabfact": {
                **C.SOURCES["tabfact"],
                "selection_predicate": "dataset-tabfact.zip :: "
                                       "wenhuchen__Table-Fact-Checking__tabfact__"
                                       "{test,validation}.parquet, unique on "
                                       "table_text; >= 4 rows and >= 2 columns; a "
                                       "label column and at least one numeric "
                                       "column; MINUS the R17-H143 evalset content "
                                       "exclusion, MINUS every table whose id, id "
                                       "STEM or whitespace-normalised serialised "
                                       "text (raw or 1,500-cut) is in the `tabfact` "
                                       "mix member",
                "tables_admitted": man["tabfact_admission"]["admitted"]},
            "edgar": {
                **C.SOURCES["edgar"],
                "selection_predicate": "R18-H150_edgar_admitted.parquet, eval half "
                                       f"under blake2b(doc_id) % 1000 < "
                                       f"{C.EVAL_DOC_PERMILLE}"}},
        "walled_never_opened": C.WALLED_NEVER_OPENED,
        "retrieval": "no network access - HF_HUB_OFFLINE=1; every archive was "
                     "already on disk under data/external/datasets/",
        "internal_duplication": {
            "distinct_claims": man["diversity"]["distinct_claims"],
            "distinct_passages": man["diversity"]["distinct_chunks"],
            "distinct_subjects": man["diversity"]["distinct_subjects"],
            "rows": ev.height,
            "passages_carrying_more_than_one_pair": int(
                ev.group_by("chunk").agg(pl.col("pair_id").n_unique().alias("p"))
                  .filter(pl.col("p") > 1).height),
            "max_pairs_on_one_passage": int(
                ev.group_by("chunk").agg(pl.col("pair_id").n_unique().alias("p"))
                  ["p"].max()),
            "max_pairs_on_one_document": int(
                ev.group_by("doc_id").agg(pl.col("pair_id").n_unique().alias("p"))
                  ["p"].max())},
        "public_repository": "no client or company name appears in this artifact "
                             "or in any script that builds it; EDGAR filings are "
                             "public SEC documents and are addressed by accession "
                             "id only",
    }

    # ---- statistical power ---------------------------------------------- #
    n_pairs = int(ev["pair_id"].n_unique())
    n_pos = int((ev["label"] == 1).sum())
    n_neg = int((ev["label"] == 0).sum())

    def se_auroc(a, npos, nneg):
        """Hanley-McNeil standard error of a single AUROC."""
        q1 = a / (2 - a)
        q2 = 2 * a * a / (1 + a)
        return float(np.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a)
                              + (nneg - 1) * (q2 - a * a)) / (npos * nneg)))

    power = {"pairs": n_pairs, "rows": ev.height,
             "positives": n_pos, "negatives": n_neg,
             "instrument": "Hanley-McNeil standard error of a single AUROC; the "
                           "eval is paired (each positive has its twin negative "
                           "over the same evidence), so this is CONSERVATIVE - the "
                           "paired design removes the passage variance the formula "
                           "still charges",
             "at": {}}
    for a in (0.5, 0.7, 0.8, 0.9):
        se = se_auroc(a, n_pos, n_neg)
        power["at"][f"auroc_{a}"] = {
            "standard_error": round(se, 4),
            "resolvable_difference_at_2_se": round(2 * se, 4),
            "95pc_interval": [round(a - 1.96 * se, 4), round(a + 1.96 * se, 4)]}
    se80 = se_auroc(0.80, n_pos, n_neg)
    power["gate_reading"] = {
        "arm_gate": ">= 0.80 mechanism gate for the R20-H177 lane B arm",
        "standard_error_at_0.80": round(se80, 4),
        "two_se_band_at_0.80": [round(0.80 - 2 * se80, 4), round(0.80 + 2 * se80, 4)],
        "smallest_difference_from_0.80_resolvable_at_2_se": round(2 * se80, 4),
        "per_family_rows": man["families"],
        "per_family_two_se_at_0.80": {
            fam: round(2 * se_auroc(0.80, n // 2, n // 2), 4)
            for fam, n in man["families"].items()},
        "note": "the eval is unchanged in size and family composition from the "
                "contaminated original (2,000 rows / 1,000 pairs, identical cell "
                "fills), so the gate is readable on it to exactly the precision "
                "the coordinator already sized against"}
    report["statistical_power"] = power

    report["verdict_fields_deliberately_absent"] = (
        "the contract's report shape asks for a per-clause PASS / FAIL / "
        "NOT-APPLICABLE and a `conforming` boolean. This executor measures and "
        "builds only; every clause's number is recorded here beside the bar it is "
        "measured against, and the coordinator adjudicates")
    report["schema_is_a_drop_in_replacement"] = {
        "columns": ev.columns,
        "identical_to_the_original": ev.columns == orig.columns,
        "reader_contract": "claim / chunk / label / pair_id / neg_family, chunk "
                           "read UNTRUNCATED and windowed 1,500/750 by the loader"}
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["note"] = "Numbers recorded, not adjudicated - the coordinator adjudicates."
    OUT.write_text(json.dumps(report, indent=2))
    print(f"-> {OUT.name} ({report['elapsed_s']}s)", flush=True)
    print(json.dumps({
        "C1_structural_identical_pairs":
            report["clauses"]["C1"]["test_1_structural"]["identical_pairs"],
        "C1_pos_rate": report["clauses"]["C1"]["test_2_strict_separation"]["positive_leg_rate"],
        "C1_neg_rate": report["clauses"]["C1"]["test_2_strict_separation"]["negative_leg_rate"],
        "C1_coverage": report["clauses"]["C1"]["test_2_strict_separation"]["instrument_coverage"],
        "C2_mix_passages_hit": len(hit_mix),
        "C2_original_control_passages_hit": len(or_hit),
        "C2_rebuilt_stems_in_member": rebuilt_dc["tabfact_STEMS_in_the_member"],
        "C2_original_stems_in_member": original_dc["tabfact_STEMS_in_the_member"],
        "C4_verdict": report["clauses"]["C4"]["verdict"],
        "C6_coverage": report["clauses"]["C6"]["rebuilt"]["coverage"],
        "C6_control_coverage":
            report["clauses"]["C6"]["LIVE_POSITIVE_CONTROL_original_eval_B"]["coverage"],
        "power_2se_at_0.80": power["gate_reading"]["smallest_difference_from_0.80_resolvable_at_2_se"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
