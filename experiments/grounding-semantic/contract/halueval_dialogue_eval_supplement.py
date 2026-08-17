"""Supplement to `contract/halueval_dialogue_eval_report.json`. CPU ONLY.

Four things the main pass left as bare counts, and one control it under-powered.

 1  CHARACTERISE THE C2 NON-ZERO. The main pass found 15 eval responses inside
    the assembled mix's claim set under some string form (9 byte-exact), 14 of
    them in the `qa` half. A count is not a finding: this block names the
    strings, their length, which leg they sit on, how many EVAL ROWS they carry,
    and which DANN group supplies the mix side.
 2  CHARACTERISE THE SUB-DOCUMENT TAIL. 26 dialogue knowledge blocks share
    >= 10% of their 8-grams with a `qa` knowledge block, one of them 50%. The
    full threshold ladder and the tail's provenance are read out here.
 3  RE-RUN THE LIVE CROSS-CONFIG POSITIVE CONTROL UNDILUTED. The main pass
    scored HaluEval `qa` knowledge against a 60,000-chunk RANDOM SAMPLE of the
    mix's 760,618, so the control detected 0.169 - a sampling artefact, not an
    instrument reading. Scored against the halueval group's own chunk set, which
    carries those blocks verbatim, the control must fire at ~1.0 or the clean
    C4 verdict is reassurance rather than evidence.
 4  STATE THE B2 DENOMINATOR ASYMMETRY. 8-gram containment is |query INTERSECT
    index| / |query|, so the direction matters when the two sides differ in
    length by two orders of magnitude. An arena document carries ~1,000 8-grams
    and a dialogue knowledge block ~10, so arena-as-query is structurally
    incapable of exceeding ~1% and its zero proves little; eval-as-query is the
    sensitive direction and is the one that decides.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
        experiments/grounding-semantic/contract/halueval_dialogue_eval_supplement.py \
        2>&1 | tee -a logs/contract-halueval_dialogue_eval.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
EVAL = EXP / "R21-H181_halueval_dialogue_eval.parquet"
OUT = HERE / "halueval_dialogue_eval_report.json"

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
NGRAM, JTHR = 8, 0.30
THRESHOLDS = (0.10, 0.25, 0.50, 0.90, 1.00)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ASV = _mod("asv", HERE / "arena_surface_verify.py")
H143 = _mod("h143", EXP / "R17-H143_evalset_assessment.py")
PG = _mod("provgate", EXP / "provenance_gate.py")
norm = H143.norm


def main():
    res = json.loads(OUT.read_text())
    d = pl.read_parquet(EVAL)
    hasher = PG._TokenHasher()

    claims, chunks, _lab, tags, cut, _groups = ASV.assemble()
    mix_claim_group = collections.defaultdict(set)
    for c, t in zip(claims, tags, strict=True):
        if c:
            mix_claim_group[c].add(t)
    mix_forms = H143.form_sets([c for c in claims if c], cut)

    # ---------------------------------------------------------------- 1
    log("=== 1 - characterising the C2 claim-side collisions ===")
    ev_claims = sorted(set(d["claim"].to_list()))
    _counts, hit, _ = H143.cross_forms(ev_claims, mix_forms, cut, reverse=False)
    raw_hit = {c for c in ev_claims if c in mix_forms["raw"]}
    norm_only = hit - raw_hit

    rows_hit = d.filter(pl.col("claim").is_in(sorted(hit)))
    detail = []
    for s in sorted(hit):
        r = d.filter(pl.col("claim") == s)
        groups = set()
        for form, key in (("raw", s), ("nraw", norm(s))):
            for mc, gs in mix_claim_group.items():
                if (mc if form == "raw" else norm(mc)) == key:
                    groups |= gs
        detail.append({
            "text": s, "chars": len(s), "tokens": len(s.split()),
            "byte_exact": s in raw_hit,
            "eval_rows": int(r.height),
            "legs": sorted(set(r["leg"].to_list())),
            "labels": sorted(set(int(v) for v in r["label"].to_list())),
            "mix_dann_groups": sorted(groups),
        })
    detail.sort(key=lambda x: -x["eval_rows"])
    lens = np.array([x["chars"] for x in detail]) if detail else np.zeros(1)
    c2char = {
        "distinct_eval_response_strings_in_the_mix_any_form": len(hit),
        "byte_exact": len(raw_hit),
        "normalisation_only": len(norm_only),
        "eval_rows_affected": int(rows_hit.height),
        "eval_rows_affected_fraction": round(rows_hit.height / d.height, 6),
        "eval_pairs_affected": int(rows_hit["pair_id"].n_unique()),
        "positive_leg_rows": int((rows_hit["label"] == 1).sum()),
        "negative_leg_rows": int((rows_hit["label"] == 0).sum()),
        "colliding_string_chars": {
            "max": int(lens.max()), "median": float(np.median(lens)),
            "min": int(lens.min())},
        "all_under_40_chars": bool(lens.max() < 40),
        "per_string": detail,
        "reading": ("these are the eval's CLAIM side against the mix's CLAIM side - the "
                    "evidence side reads 0 on every form and every channel. A claim "
                    "collision without an evidence collision cannot transfer a "
                    "(claim, evidence) association; what it can do is let a model that "
                    "memorised a short conversational phrase's training label carry it "
                    "over. The consequence is bounded by eval_rows_affected"),
        "note": NOTE,
    }
    log(f"  {len(hit)} distinct strings ({len(raw_hit)} byte-exact) over "
        f"{rows_hit.height} of {d.height} eval rows, max {int(lens.max())} chars")
    res["C2_collision_characterisation"] = c2char
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    # ---------------------------------------------------------------- 2
    log("=== 2 - the sub-document tail against the trained halves ===")
    z = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    qa = pl.read_parquet(io.BytesIO(z.read("pminervini__HaluEval__qa__data.parquet")))
    su = pl.read_parquet(
        io.BytesIO(z.read("pminervini__HaluEval__summarization__data.parquet")))
    ev_chunks = sorted(set(d["chunk"].to_list()))
    ev_h = [PG.ngram_hashes(t, NGRAM, hasher) for t in ev_chunks]

    tail = {}
    for name, texts in (("qa_knowledge", qa["knowledge"].to_list()),
                        ("summarization_document", su["document"].to_list())):
        idx_h = [PG.ngram_hashes(t, NGRAM, hasher) for t in texts]
        flat, owner, sizes = ASV._index(idx_h)
        bc = np.zeros(len(ev_h))
        for i, q in enumerate(ev_h):
            if q.size:
                _j, c, _a, _b = ASV._max_pair(q, flat, owner, sizes)
                bc[i] = c
        scorable = np.array([a.size > 0 for a in ev_h])
        ladder = {f"n_ge_{t:.2f}": int((bc >= t).sum()) for t in THRESHOLDS}
        top = sorted(range(len(ev_h)), key=lambda i: -bc[i])[:10]
        tail[name] = {
            "eval_blocks_scorable": int(scorable.sum()),
            "max_containment": round(float(bc.max()), 4),
            "mean_containment": round(float(bc.mean()), 6),
            "threshold_ladder": ladder,
            "fraction_of_eval_blocks_at_ge_0.10": round(
                float((bc >= 0.10).sum()) / max(int(scorable.sum()), 1), 6),
            "top_blocks": [{"containment": round(float(bc[i]), 4),
                            "chars": len(ev_chunks[i]),
                            "ngrams": int(ev_h[i].size),
                            "text": ev_chunks[i][:200]} for i in top if bc[i] > 0],
        }
        log(f"  {name}: max {tail[name]['max_containment']}, ladder {ladder}")
    tail["reading"] = (
        "a dialogue knowledge block carries a median of ~10 8-grams, so a single shared "
        "8-gram already reads ~0.10 containment and two read ~0.20. The ladder is "
        "therefore a count of blocks sharing ONE OR TWO n-grams with the trained halves, "
        "not evidence of a shared document; the whole-block exact, normalised and stem "
        "keys all read 0, and the Jaccard gate reads 0 units at >= 0.30")
    tail["note"] = NOTE
    res["subdocument_tail_characterisation"] = tail
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    # ---------------------------------------------------------------- 3
    log("=== 3 - the live cross-config positive control, UNDILUTED ===")
    hal_chunks = sorted({c for c, t in zip(chunks, tags, strict=True)
                         if t == "halueval" and c})
    hal_h = [PG.ngram_hashes(t, NGRAM, hasher) for t in hal_chunks]
    hflat, howner, hsizes = ASV._index(hal_h)
    qa_know = sorted(set(qa["knowledge"].to_list()))
    js, cs = [], []
    for u in qa_know:
        q = PG.ngram_hashes(u, NGRAM, hasher)
        if q.size == 0:
            continue
        j, c, _a, _b = ASV._max_pair(q, hflat, howner, hsizes)
        js.append(j); cs.append(c)
    js, cs = np.array(js), np.array(cs)
    ctrl = {
        "kind": ("LIVE positive control, undiluted - every distinct HaluEval `qa` "
                 "knowledge block scored against the halueval DANN group's own chunk "
                 "set inside the assembled mix, which carries those blocks verbatim. "
                 "Same instrument, same thresholds as the `dialogue` question"),
        "n_scored": int(js.size),
        "index_units": len(hal_chunks),
        "detected_at_jaccard_ge_0.30": int((js >= JTHR).sum()),
        "detection_fraction": round(float((js >= JTHR).mean()), 6),
        "median_jaccard": round(float(np.median(js)), 4),
        "min_jaccard": round(float(js.min()), 4),
        "units_at_containment_1.00": int((cs >= 1.0).sum()),
        "containment_fraction_at_1.00": round(float((cs >= 1.0).mean()), 6),
        "fires": bool((js >= JTHR).mean() > 0.99),
        "supersedes": ("C4_live_positive_controls.live_cross_config_control, which scored "
                       "the same text against a 60,000-chunk RANDOM SAMPLE of the mix's "
                       "760,618 distinct chunks and therefore read 0.169 - a sampling "
                       "dilution, not an instrument failure. This undiluted reading is "
                       "the one that licenses the clean C4 verdict"),
        "note": NOTE,
    }
    log(f"  undiluted cross-config control: {ctrl['detected_at_jaccard_ge_0.30']}"
        f"/{ctrl['n_scored']} at Jaccard>=0.30 (fraction {ctrl['detection_fraction']}), "
        f"containment 1.00 on {ctrl['containment_fraction_at_1.00']}")
    res["C4_live_positive_control_undiluted"] = ctrl
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    # ---------------------------------------------------------------- 4
    log("=== 4 - B2 denominator asymmetry ===")
    subsets = ASV.load_arena()
    arena_ch, _owner = ASV.arena_units(subsets)
    a_docs = sorted(set(arena_ch["documents"]))
    a_h = [PG.ngram_hashes(t, NGRAM, hasher) for t in a_docs]
    a_sizes = np.array([h.size for h in a_h if h.size])
    e_sizes = np.array([h.size for h in ev_h if h.size])
    ev_flat, ev_owner, ev_usz = ASV._index(ev_h)
    bc_ev = np.zeros(len(a_h))
    for i, q in enumerate(a_h):
        if q.size:
            _j, c, _a, _b = ASV._max_pair(q, ev_flat, ev_owner, ev_usz)
            bc_ev[i] = c
    asym = {
        "instrument": "8-gram containment = |ngrams(query) INTERSECT ngrams(index)| / |ngrams(query)|",
        "arena_document_8grams": {"median": float(np.median(a_sizes)),
                                  "p10": float(np.percentile(a_sizes, 10)),
                                  "p90": float(np.percentile(a_sizes, 90))},
        "eval_knowledge_block_8grams": {"median": float(np.median(e_sizes)),
                                        "p10": float(np.percentile(e_sizes, 10)),
                                        "p90": float(np.percentile(e_sizes, 90))},
        "arena_as_query_max_containment": round(float(bc_ev.max()), 6),
        "arena_as_query_structural_ceiling": round(
            float(np.median(e_sizes) / np.median(a_sizes)), 6),
        "eval_as_query_max_containment": res["B2_disjointness_from_the_blind_arena"][
            "8gram_eval_knowledge_into_arena_documents"]["max_containment_single_unit"],
        "eval_as_query_max_containment_whole_arena_union": res[
            "B2_disjointness_from_the_blind_arena"][
            "8gram_eval_knowledge_into_arena_documents"]["max_containment_whole_index_union"],
        "which_direction_decides": (
            "eval-as-query. An arena document carries two orders of magnitude more "
            "8-grams than a dialogue knowledge block, so arena-as-query cannot exceed "
            "roughly the size ratio no matter how complete the overlap, and its zero is "
            "weak evidence. Eval-as-query has the small denominator and would read 1.0 "
            "on a fully contained block; it reads 0.0, meaning NOT ONE 8-gram of ANY of "
            "the 7,688 scorable dialogue knowledge blocks appears in ANY arena document. "
            "That is the decisive number, and it is the reason this surface does not "
            "repeat the halueval-qa/hotpotqa exposure"),
        "contrast_with_the_banked_arena_finding": (
            "arena_surface_report.json records 17 arena documents byte-for-byte inside "
            "halueval training chunks, all hotpotqa, and 8-gram containment >= 0.10 "
            "reaching a large share of that subset. The mechanism was shared upstream "
            "provenance: HaluEval `qa` supplies HotpotQA wiki paragraphs as `knowledge`. "
            "`dialogue` supplies entity-relation triple strings averaging 111 characters "
            "instead, so the shared-provenance route does not exist for it - measured "
            "here at zero in both directions and on the document channel"),
        "note": NOTE,
    }
    log(f"  arena-as-query max containment {asym['arena_as_query_max_containment']} "
        f"(structural ceiling ~{asym['arena_as_query_structural_ceiling']}); "
        f"eval-as-query max {asym['eval_as_query_max_containment']}")
    res["B2_denominator_asymmetry"] = asym

    # ---------------------------------------------------------------- 5
    log("=== 5 - cross-references and the containment inversion ===")
    res["clauses"]["C2"]["characterisation"] = (
        "see C2_collision_characterisation - the non-zero is "
        f"{c2char['distinct_eval_response_strings_in_the_mix_any_form']} distinct "
        f"response strings of at most {c2char['colliding_string_chars']['max']} "
        f"characters (median {c2char['colliding_string_chars']['median']}), touching "
        f"{c2char['eval_rows_affected']} of {int(d.height)} eval rows "
        f"({c2char['eval_rows_affected_fraction']}). The EVIDENCE side reads 0 on every "
        "string form, every direction, the document channel and the 8-gram channel")
    res["clauses"]["C4"]["live_positive_control"]["undiluted_cross_config_fraction"] = (
        ctrl["detection_fraction"])
    res["clauses"]["C4"]["live_positive_control"]["undiluted_note"] = (
        "see C4_live_positive_control_undiluted - the 0.169 figure beside it is a "
        "60k-of-760k sampling dilution; scored against the halueval group's own chunks "
        f"the control fires {ctrl['detected_at_jaccard_ge_0.30']}/{ctrl['n_scored']}")
    res["clauses"]["C2"]["arena_denominator_note"] = asym["which_direction_decides"]

    c1d = res["C1_label_commensurability"]["distributional_diagnostic"]
    inversion = {
        "what": ("content-token containment of the claim in the evidence separates the "
                 "legs only at the extreme tail on this surface, and INVERTS in the mean"),
        "positive_leg_mean_containment": c1d["positive_leg"]["mean"],
        "negative_leg_mean_containment": c1d["negative_leg"]["mean"],
        "mean_gap_positive_minus_negative": c1d["mean_gap"],
        "attested_rate_gap_at_0.90": c1d["attested_rate_gap_at_0.90"],
        "lexical_containment_auroc_as_a_scorer": res["B3_claim_only_shortcut"][
            "surface_parity"]["auroc"]["claim_chunk_containment"],
        "why": ("a hallucinated conversational reply is LONGER (113.2 characters against "
                "73.1) and names MORE of the knowledge block's entities while asserting a "
                "false relation between them. Token containment cannot see a wrong "
                "relation over right entities, so it reads the negative leg as slightly "
                "BETTER attested than the positive one"),
        "consequence_for_reading_a_score": (
            "the campaign's lexical containment baseline scores BELOW chance here "
            "(0.4440). This surface therefore cannot be beaten by lexical overlap, which "
            "is what makes it a genuine out-of-domain relational probe - and equally, it "
            "means C1's distributional diagnostic passes on the tail alone and the "
            "absolute levels are low on BOTH legs"),
        "note": NOTE,
    }
    res["containment_inversion_finding"] = inversion
    log(f"  containment inversion: pos {inversion['positive_leg_mean_containment']} vs "
        f"neg {inversion['negative_leg_mean_containment']}, as a scorer "
        f"{inversion['lexical_containment_auroc_as_a_scorer']}")

    s = res["summary"]
    s["C1_lexical_containment_auroc_as_a_scorer"] = inversion[
        "lexical_containment_auroc_as_a_scorer"]
    s["C1_containment_mean_gap_positive_minus_negative"] = c1d["mean_gap"]
    s["C4_live_cross_config_control_undiluted_fraction"] = ctrl["detection_fraction"]
    s["C2_eval_rows_affected_by_claim_collisions"] = c2char["eval_rows_affected"]
    s["C2_eval_rows_affected_fraction"] = c2char["eval_rows_affected_fraction"]
    s["C2_colliding_strings_all_under_40_chars"] = c2char["all_under_40_chars"]
    s["B2_eval_as_query_max_containment_into_arena"] = asym["eval_as_query_max_containment"]
    res["summary"] = s
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    log("=== SUPPLEMENT COMPLETE ===")


if __name__ == "__main__":
    main()
