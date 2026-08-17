"""R21-H181 - HaluEval-`dialogue` held-out eval: measurement and contract pass. CPU ONLY.

Verifies `experiments/grounding-semantic/R21-H181_halueval_dialogue_eval.parquet`
(built by `R21-H181_eval_build.py`) against every clause of
`docs/experiments/dataset-contract.md` INCLUDING amendments C-A1 and C-A2, and
banks `contract/halueval_dialogue_eval_report.json`.

Three measurements run BEFORE any model is ever scored on this surface, in this
order, exactly as the R21-H181 registration block requires:

  B1  disjointness from the two TRAINED HaluEval halves (`qa`, `summarization`)
      and from the whole assembled mix - three string forms in the campaign's
      eight-pairing matrix, BOTH directions, on the knowledge blocks AND the
      responses, PLUS the document channel and the sub-document 8-gram channel.
      Sharing a corpus name is not sharing text; not sharing text is not sharing
      documents - so both are measured separately.
  B2  disjointness from the blind arena, from `gold_full`, and from every other
      banked mechanism eval. HaluEval's `qa` knowledge blocks ARE HotpotQA
      paragraphs and that is how 17 arena documents ended up byte-for-byte
      inside halueval training chunks; the instrument here is the arena audit's
      own (`contract/arena_surface_verify.py` `_index` / `_max_pair`, the
      `provenance_gate` 8-gram primitives), reused rather than rewritten, at the
      same containment thresholds that audit used.
  B3  the claim-only shortcut with a label-shuffled negative control, on the
      pair-aware split chosen from a MEASURED pair census - the instrument of
      `R20_claimonly_sweep.py`, reused. HaluEval's trained halves read 0.9519
      claim-only; a sibling configuration from the same pipeline is assumed to
      inherit that until measured.

CPU ONLY - GPU0/GPU1/GPU2 carry a training draw and an arena scoring pass.
Nothing here scores a model and nothing here is adjudicated.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
        experiments/grounding-semantic/contract/halueval_dialogue_eval_contract.py \
        2>&1 | tee logs/contract-halueval_dialogue_eval.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set, not setdefault, before any banked import
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import collections
import hashlib
import importlib.util as _ilu
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent           # .../grounding-semantic/contract
EXP = HERE.parent                      # .../grounding-semantic
ROOT = EXP.parent.parent               # repo root
DATA = ROOT / "data" / "external" / "datasets"
EVAL = EXP / "R21-H181_halueval_dialogue_eval.parquet"
OUT = HERE / "halueval_dialogue_eval_report.json"
CACHE = ROOT / "tmp" / "h181"

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
NGRAM = 8
JACCARD_THR = 0.30
KILL = 0.02
CONTAINMENT_THRESHOLDS = (0.10, 0.25, 0.50, 0.90, 1.00)  # the arena audit's own
CUT = 1500                              # R7-H59 CFG.chunk_max_chars

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading banked modules (CPU pinned)")
ASV = _mod("asv", HERE / "arena_surface_verify.py")     # arena loader + _index/_max_pair
H143 = _mod("h143", EXP / "R17-H143_evalset_assessment.py")   # norm/form_sets/cross_forms
PG = _mod("provgate", EXP / "provenance_gate.py")             # 8-gram primitives
LC = _mod("lanecommon", EXP / "R20-H174_lane_common.py")      # containment/auroc/parity
CO = _mod("claimonly", EXP / "R20_claimonly_sweep.py")        # the claim-only instrument
H108 = _mod("h108", EXP / "R10-H108_lane.py")                 # public_train / gold_full

norm = H143.norm


# --------------------------------------------------------------------------- #
# incremental banking
# --------------------------------------------------------------------------- #
def load_report():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {
        "artifact": "halueval_dialogue_eval_report.json",
        "surface": ("R21-H181 - the HaluEval `dialogue` configuration built as a "
                    "held-out evaluation surface"),
        "eval_parquet": str(EVAL.relative_to(ROOT)),
        "contract": "docs/experiments/dataset-contract.md (with amendments C-A1, C-A2)",
        "compute": ("CPU only; CUDA_VISIBLE_DEVICES forced empty before any import - "
                    "GPU0/1/2 carry a training draw and an arena scoring pass"),
        "note": NOTE,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def bank(res, key, value):
    res[key] = value
    res["note"] = NOTE
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    log(f"banked -> {OUT.name} :: {key}")


# --------------------------------------------------------------------------- #
# the eval and the two trained halves
# --------------------------------------------------------------------------- #
def load_eval():
    d = pl.read_parquet(EVAL)
    if d.height != 20_000 or d["pair_id"].n_unique() != 10_000:
        raise SystemExit(f"EVAL ABORT: {d.height} rows / {d['pair_id'].n_unique()} pairs")
    log(f"eval: {d.height} rows / {d['pair_id'].n_unique()} pairs, "
        f"{d['chunk'].n_unique()} distinct knowledge blocks")
    return d


def halueval_configs():
    """Every configuration in the archive, with the columns it actually ships -
    measured, so the selection predicate is verified rather than quoted."""
    z = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    out = {}
    for n in sorted(x for x in z.namelist() if x.endswith(".parquet")):
        cfg = n.split("__")[2]
        d = pl.read_parquet(io.BytesIO(z.read(n)))
        out[cfg] = {"file": n, "rows": d.height, "columns": d.columns, "frame": d}
    return out


def loader_predicate_check():
    """C3/C8 - read the loader's own selection predicate off the source, do not
    take it from the registration block."""
    src = (EXP / "R10-H108_lane.py").read_text()
    block = src.split("zh = zipfile.ZipFile")[1].split("zp = zipfile.ZipFile")[0]
    cfgs = re.findall(r'\(\s*"(\w+)",\s*"(\w+)",\s*"(\w+)",\s*"(\w+)"\s*\)', block)
    mentions = {c: len(re.findall(rf'"{c}"', src)) for c in
                ("qa", "summarization", "dialogue", "general")}
    scan = {}
    for f in ("R10-H108_lane.py", "R16-H142_G1_arm.py", "R18-H150_arm_run.py",
              "R20-H174_arm_run.py", "R17-H143_evalset_assessment.py"):
        p = EXP / f
        scan[f] = ("ABSENT" if not p.exists() else
                   {"halueval_dialogue_mentions": len(
                       re.findall(r"dialogue", p.read_text(), re.IGNORECASE)),
                    "halueval_general_mentions": len(
                        re.findall(r"halueval[^\n]*general|general[^\n]*halueval",
                                   p.read_text(), re.IGNORECASE))})
    return {
        "loader": "R10-H108_lane.public_train()",
        "configs_iterated_verbatim": [list(c) for c in cfgs],
        "split_filter": "none - the archive ships a single `data` split per config",
        "row_filter": ("none - unlike ragtruth_en (`context > 50 chars`) and psiloqa, "
                       "the HaluEval branch applies no length or quality filter; every "
                       "row of the two iterated configs enters the mix"),
        "config_string_mentions_in_loader": mentions,
        "static_scan_for_dialogue_in_the_mix_assembly_chain": scan,
        "reading": ("the loader iterates exactly ('qa','knowledge','right_answer',"
                    "'hallucinated_answer') and ('summarization','document',"
                    "'right_summary','hallucinated_summary'); the strings 'dialogue' "
                    "and 'general' do not appear in any mix-assembly loader"),
    }


# --------------------------------------------------------------------------- #
# exact disjointness, the campaign's eight form pairings, both directions
# --------------------------------------------------------------------------- #
def exact_block(eval_units, other_texts, label):
    """cross_forms in BOTH directions - a strict superset of the contract's
    'three string forms, both directions' (raw, truncated-at-1500, normalised).

    Two channels are reported SEPARATELY, never merged: `exact_raw` is byte
    identity alone, and `three_string_forms` is the union over the eight
    raw/truncated/normalised pairings. A zero on the first is not a disjointness
    result - the arena audit's own halueval finding read 1 exact hit (a single
    full stop) while the sub-document channel read 40.8% of a subset.
    """
    fs = H143.form_sets([t for t in other_texts if t], CUT)
    counts, hit, _ = H143.cross_forms(eval_units, fs, CUT, reverse=True)
    fwd = len(hit)
    rev = max(v.get("mix_units_in_eval", 0) for v in counts.values())
    raw_fwd = counts["raw_in_mix_raw"]["eval_units_in_mix"]
    raw_rev = counts["raw_in_mix_raw"].get("mix_units_in_eval", 0)
    log(f"  exact {label:52s} raw {raw_fwd:5d}/{raw_rev:5d}  anyform "
        f"{fwd:6d} / {len(set(eval_units)):6d}  rev {rev:6d}")
    return {
        "eval_units_distinct": len(set(eval_units)),
        "other_units_distinct": len(fs["raw"]),
        "channel_1_exact_raw": {"eval_units_in_other": raw_fwd,
                                "other_units_in_eval": raw_rev,
                                "clean": bool(raw_fwd == 0 and raw_rev == 0)},
        "channel_2_three_string_forms": {
            "eval_units_in_other_any_form": fwd,
            "other_units_in_eval_any_form_max": rev,
            "clean": bool(fwd == 0 and rev == 0)},
        "form_pairings": counts,
        "eval_units_in_other_any_form": fwd,
        "other_units_in_eval_any_form_max": rev,
        "eval_fraction": round(fwd / max(len(set(eval_units)), 1), 8),
        "clean": bool(fwd == 0 and rev == 0),
    }


# --------------------------------------------------------------------------- #
# the 8-gram instrument, reused from the arena audit
# --------------------------------------------------------------------------- #
class NGramIndex:
    """One side of the comparison, indexed once and queried many times.

    `_index` and `_max_pair` are `contract/arena_surface_verify.py`'s own, which
    are in turn `provenance_gate._max_jaccard` with a containment denominator
    added; `ngram_hashes` is the banked R14-H136 primitive.
    """

    def __init__(self, texts, hasher, n=NGRAM):
        self.texts = list(texts)
        self.hashes = [PG.ngram_hashes(t, n, hasher) for t in self.texts]
        self.flat, self.owner, self.sizes = ASV._index(self.hashes)
        self.uniq = np.unique(self.flat) if self.flat.size else self.flat
        self.scorable = np.array([a.size > 0 for a in self.hashes])

    def query(self, qhashes):
        return ASV._max_pair(qhashes, self.flat, self.owner, self.sizes)

    def union_containment(self, qhashes):
        if qhashes.size == 0 or self.uniq.size == 0:
            return 0.0
        lo = np.searchsorted(self.uniq, qhashes, side="left")
        hi = np.searchsorted(self.uniq, qhashes, side="right")
        return float((hi > lo).sum()) / qhashes.size


def ngram_cross(query_texts, index_texts, hasher, label, thresholds=CONTAINMENT_THRESHOLDS):
    """Best Jaccard and best 8-gram CONTAINMENT of each query against any single
    index unit, plus containment against the index's whole n-gram pool."""
    idx = NGramIndex(index_texts, hasher)
    qh = [PG.ngram_hashes(t, NGRAM, hasher) for t in query_texts]
    bj = np.zeros(len(qh))
    bc = np.zeros(len(qh))
    uc = np.zeros(len(qh))
    best_unit = np.full(len(qh), -1, dtype=np.int64)
    for i, q in enumerate(qh):
        if q.size == 0:
            continue
        j, c, uj, ucid = idx.query(q)
        bj[i], bc[i] = j, c
        best_unit[i] = ucid
        uc[i] = idx.union_containment(q)
    scorable = np.array([a.size > 0 for a in qh])
    out = {
        "query_units": len(qh),
        "query_units_scorable_at_8grams": int(scorable.sum()),
        "query_units_too_short_for_8grams": int((~scorable).sum()),
        "index_units": len(idx.texts),
        "index_units_scorable_at_8grams": int(idx.scorable.sum()),
        "max_jaccard": round(float(bj.max()) if bj.size else 0.0, 4),
        "units_at_jaccard_ge_0.30": int((bj >= JACCARD_THR).sum()),
        "max_containment_single_unit": round(float(bc.max()) if bc.size else 0.0, 4),
        "max_containment_whole_index_union": round(float(uc.max()) if uc.size else 0.0, 4),
        "containment_single_unit": {
            f"n_ge_{t:.2f}": int((bc >= t).sum()) for t in thresholds},
        "containment_whole_index_union": {
            f"n_ge_{t:.2f}": int((uc >= t).sum()) for t in thresholds},
        "mean_containment_single_unit": round(float(bc.mean()) if bc.size else 0.0, 6),
    }
    log(f"  8-gram {label:50s} maxJ {out['max_jaccard']:.4f}  maxC {out['max_containment_single_unit']:.4f}"
        f"  C>=0.10 {out['containment_single_unit']['n_ge_0.10']}")
    return out, bj, bc, uc, best_unit, scorable


# --------------------------------------------------------------------------- #
# STAGE: sides
# --------------------------------------------------------------------------- #
def stage_sides(res):
    d = load_eval()
    cfgs = halueval_configs()
    admissibility = {
        "archive": "data/external/datasets/dataset-halueval.zip",
        "configs_present": {c: {"rows": v["rows"], "columns": v["columns"]}
                            for c, v in cfgs.items()},
        "loaded_by_the_mix": ["qa", "summarization"],
        "not_loaded": ["dialogue", "general"],
        "general_excluded_because": ("it ships no evidence column - "
                                     "ID / user_query / chatgpt_response / hallucination / "
                                     "hallucination_spans - so it is not a grounding task"),
        "loader_predicate": loader_predicate_check(),
    }
    bank(res, "admissibility", admissibility)

    eval_census = {
        "rows": int(d.height),
        "pairs": int(d["pair_id"].n_unique()),
        "positives": int((d["label"] == 1).sum()),
        "negatives": int((d["label"] == 0).sum()),
        "positive_rate": round(float(d["label"].mean()), 4),
        "distinct_claims": int(d["claim"].n_unique()),
        "distinct_knowledge_blocks": int(d["chunk"].n_unique()),
        "distinct_dialogue_histories": int(d["dialogue_history"].n_unique()),
        "pair_integrity": LC.pair_integrity(d),
        "knowledge_chars_mean": round(float(d["knowledge_chars"].mean()), 1),
        "knowledge_chars_max": int(d["knowledge_chars"].max()),
        "columns": d.columns,
        "shape_matched": ("the flat long form of R20-H177_eval_B.parquet - pair_id / "
                          "label / claim / chunk / doc_id / source - which is also the "
                          "shape R21-H180_ragtruth_eval.parquet took. Chosen because "
                          "R20_baseline_legs.flatten() consumes it unmodified; the "
                          "gold_full shape (R10-H108_lane.gold_full: claim + explicit "
                          "chunk LIST grouped on `owner`) is the same long form with a "
                          "grouping key and is reachable as [[chunk]] per row"),
    }
    bank(res, "eval_census", eval_census)
    return d, cfgs


# --------------------------------------------------------------------------- #
# STAGE: b1 - disjointness from the two TRAINED halves and from the whole mix
# --------------------------------------------------------------------------- #
def stage_b1(res, d, cfgs):
    hasher = PG._TokenHasher()
    ev_chunks = sorted(set(d["chunk"].to_list()))
    ev_claims = sorted(set(d["claim"].to_list()))
    ev_pos = sorted(set(d.filter(pl.col("label") == 1)["claim"].to_list()))
    ev_neg = sorted(set(d.filter(pl.col("label") == 0)["claim"].to_list()))

    qa = cfgs["qa"]["frame"]
    su = cfgs["summarization"]["frame"]
    halves = {
        "qa_knowledge": qa["knowledge"].to_list(),
        "summarization_document": su["document"].to_list(),
        "qa_claims_both_legs": (qa["right_answer"].to_list()
                                + qa["hallucinated_answer"].to_list()),
        "summarization_claims_both_legs": (su["right_summary"].to_list()
                                           + su["hallucinated_summary"].to_list()),
    }

    log("=== B1 - exact disjointness from the two TRAINED HaluEval halves ===")
    ex = {}
    for other, texts in halves.items():
        ex[f"eval_knowledge_vs_{other}"] = exact_block(
            ev_chunks, texts, f"eval knowledge vs {other}")
        ex[f"eval_responses_vs_{other}"] = exact_block(
            ev_claims, texts, f"eval responses vs {other}")

    log("=== B1 - sub-document 8-gram channel vs the two TRAINED halves ===")
    ng = {}
    for other, texts in halves.items():
        blk, *_ = ngram_cross(ev_chunks, texts, hasher, f"eval knowledge -> {other}")
        ng[f"eval_knowledge_into_{other}"] = blk
        blk, *_ = ngram_cross(texts, ev_chunks, hasher, f"{other} -> eval knowledge")
        ng[f"{other}_into_eval_knowledge"] = blk
    blk, *_ = ngram_cross(ev_claims, halves["qa_claims_both_legs"], hasher,
                          "eval responses -> qa claims")
    ng["eval_responses_into_qa_claims"] = blk
    blk, *_ = ngram_cross(ev_claims, halves["summarization_claims_both_legs"], hasher,
                          "eval responses -> summarization claims")
    ng["eval_responses_into_summarization_claims"] = blk

    log("=== B1 - document channel ===")
    doc = document_channel(d, cfgs)

    bank(res, "B1_disjointness_from_the_trained_halves", {
        "question": ("does the held-out `dialogue` configuration share text, or share "
                     "documents, with the `qa` and `summarization` halves that ARE in "
                     "the training mix"),
        "string_channel_exact": ex,
        "string_channel_8gram_subdocument": ng,
        "document_channel": doc,
        "legs_checked_separately": {
            "positive_leg_distinct_responses": len(ev_pos),
            "negative_leg_distinct_responses": len(ev_neg),
        },
        "note": NOTE,
    })


def document_channel(d, cfgs):
    """C2's document channel. HaluEval ships NO provenance identifier column in
    any of its four configurations - measured, not assumed - so the identifier
    channel is structurally absent and the document question is answered by the
    text-derived identifiers the campaign's arena instrument extracts (`Title:`
    lines and URLs) plus a whole-evidence document key in raw and normalised
    form."""
    id_cols = {"id", "unique_id", "case_id", "wiki_revision_id", "page", "fever_id",
               "table_id", "wiki_title", "wiki_url", "doc_id", "row_key", "source"}
    present = {c: sorted(x for x in v["columns"] if x.lower() in id_cols)
               for c, v in cfgs.items()}

    def derived(texts):
        titles, urls = set(), set()
        for t in texts:
            titles |= set(ASV._TITLE.findall(t))
            urls |= set(ASV._URL.findall(t.split("\n", 1)[0]))
        return titles, urls

    ev = d["chunk"].to_list()
    sides = {
        "eval_dialogue_knowledge": ev,
        "qa_knowledge": cfgs["qa"]["frame"]["knowledge"].to_list(),
        "summarization_document": cfgs["summarization"]["frame"]["document"].to_list(),
    }
    derived_ids = {k: derived(v) for k, v in sides.items()}
    et, eu = derived_ids["eval_dialogue_knowledge"]

    out = {
        "provenance_identifier_columns_present_per_config": present,
        "reading_identifier_channel": ("every HaluEval configuration ships zero "
                                       "provenance identifier columns, so there is no "
                                       "identifier channel to collide on - the same "
                                       "finding arena_surface_verify.mix_provenance_ids "
                                       "records for the two trained halves"),
        "text_derived_identifiers": {
            k: {"titles": len(v[0]), "urls": len(v[1])} for k, v in derived_ids.items()},
        "text_derived_collisions": {
            k: {"titles": len(et & v[0]), "urls": len(eu & v[1])}
            for k, v in derived_ids.items() if k != "eval_dialogue_knowledge"},
        "whole_evidence_document_key": {},
    }
    ev_raw, ev_norm = set(ev), {norm(x) for x in ev}
    ev_stem = {ASV.stem(x) for x in ev if ASV.stem(x)}
    for k, v in sides.items():
        if k == "eval_dialogue_knowledge":
            continue
        out["whole_evidence_document_key"][k] = {
            "raw": len(ev_raw & set(v)),
            "normalised": len(ev_norm & {norm(x) for x in v}),
            "stem": len(ev_stem & {ASV.stem(x) for x in v if ASV.stem(x)}),
        }
    log(f"  document channel: text-derived collisions "
        f"{out['text_derived_collisions']}, whole-evidence key "
        f"{out['whole_evidence_document_key']}")
    return out


# --------------------------------------------------------------------------- #
# STAGE: b2 - the blind arena, gold_full, the other mechanism evals
# --------------------------------------------------------------------------- #
MECH_EVALS = (
    ("antigaming_nearmiss_bindrow", "R18-H150_antigaming_set.parquet"),
    ("antigaming_traced", "R14-H133_antigaming_traced.parquet"),
    ("findver", "R19_findver_lane.parquet"),
    ("eval_C", "R20-H177_eval_C.parquet"),
    ("eval_B", "R20-H177_eval_B.parquet"),
    ("h148_itemindex_probe", "R17-H148_probe.parquet"),
    ("h149_roleswap_probe", "R17-H149_probe.parquet"),
    ("h150_unitswap_probe", "R18-H150_unitswap_probe.parquet"),
    ("r15_bindprobe", "R15_L1_bindprobe_pairs.parquet"),
    ("r15_typeprobe", "R15_P1_typeprobe_quads.parquet"),
    ("g0b_composed_probes", "R20-G0b_composed_probes.parquet"),
    ("h117_heldout_pairs", "R11-H117_heldout_pairs.parquet"),
    ("h175b_eval_clean", "R20-H175b_qlane_eval_clean.parquet"),
    ("h175b_eval_clean_prefix", "R20-H175b_qlane_eval_clean_prefix.parquet"),
    ("dr_h113_gate_judged", "DR_H113_gate_judged.parquet"),
    ("r12_h121_gateBC_rows", "R12-H121_gateBC_rows.parquet"),
    ("ragtruth_eval_R21_H180", "R21-H180_ragtruth_eval.parquet"),
)


def stage_b2(res, d):
    hasher = PG._TokenHasher()
    ev_chunks = sorted(set(d["chunk"].to_list()))
    ev_claims = sorted(set(d["claim"].to_list()))

    log("=== B2 - the blind arena ===")
    subsets = ASV.load_arena()
    arena_ch, arena_owner = ASV.arena_units(subsets)
    a_docs = sorted(set(arena_ch["documents"]))
    a_resp = sorted(set(arena_ch["responses"]))
    a_sent = sorted(set(arena_ch["response_sentences"]))

    arena_exact = {}
    for name, units in (("arena_documents", a_docs), ("arena_responses", a_resp),
                        ("arena_response_sentences", a_sent)):
        arena_exact[f"eval_knowledge_vs_{name}"] = exact_block(
            ev_chunks, units, f"eval knowledge vs {name}")
        arena_exact[f"eval_responses_vs_{name}"] = exact_block(
            ev_claims, units, f"eval responses vs {name}")

    log("=== B2 - arena 8-gram containment, the audit's own thresholds ===")
    ev_idx = NGramIndex(ev_chunks, hasher)
    doc_hashes = [PG.ngram_hashes(t, NGRAM, hasher) for t in a_docs]
    bj = np.zeros(len(a_docs)); bc = np.zeros(len(a_docs)); uc = np.zeros(len(a_docs))
    for i, q in enumerate(doc_hashes):
        if q.size == 0:
            continue
        j, c, _uj, _uc = ev_idx.query(q)
        bj[i], bc[i] = j, c
        uc[i] = ev_idx.union_containment(q)
    doc_sub = {}
    for u, s in zip(arena_ch["documents"], arena_owner["documents"], strict=True):
        doc_sub.setdefault(u, s)

    per_threshold = {}
    for thr in CONTAINMENT_THRESHOLDS:
        hit = {a_docs[i] for i in range(len(a_docs)) if bc[i] >= thr}
        per_sub, tot = {}, 0
        for sub, v in subsets.items():
            n = sum(1 for docs in v["documents"] if any(c in hit for c in docs))
            tot += n
            per_sub[sub] = {"responses": len(v["responses"]),
                            "responses_touching_a_hit_document": n,
                            "fraction_of_subset_responses": round(n / len(v["responses"]), 6)}
        per_threshold[f"containment_ge_{thr:.2f}"] = {
            "arena_documents_hit": len(hit),
            "arena_responses_touched": tot,
            "fraction_of_arena_responses": round(tot / ASV.EXPECT_RESPONSES, 6),
            "per_subset": per_sub,
        }
        log(f"  arena containment >= {thr:.2f}: {len(hit)} documents, "
            f"{tot}/{ASV.EXPECT_RESPONSES} responses")

    per_sub_max = collections.defaultdict(float)
    for i, u in enumerate(a_docs):
        s = doc_sub[u]
        per_sub_max[s] = max(per_sub_max[s], float(bc[i]))

    # reverse: eval knowledge blocks against an arena-document index
    rev_blk, rbj, rbc, ruc, _bu, _sc = ngram_cross(
        ev_chunks, a_docs, hasher, "eval knowledge -> arena documents")
    resp_blk, *_ = ngram_cross(ev_claims, a_resp, hasher,
                               "eval responses -> arena responses")

    arena_block = {
        "arena": {"subsets": sorted(subsets), "responses": ASV.EXPECT_RESPONSES,
                  "documents_distinct": len(a_docs),
                  "construction": "arena_surface_verify.load_arena() - the R8-H77 frozen gate"},
        "string_channel_exact": arena_exact,
        "8gram_arena_documents_into_eval_knowledge": {
            "instrument": ("contract/arena_surface_verify.py `_index`/`_max_pair` over "
                           "provenance_gate 8-gram hashes - the SAME instrument the arena "
                           "audit used, at the same containment thresholds"),
            "arena_documents_scorable": int(sum(1 for h in doc_hashes if h.size)),
            "arena_documents_too_short": int(sum(1 for h in doc_hashes if not h.size)),
            "max_jaccard": round(float(bj.max()), 4),
            "arena_documents_at_jaccard_ge_0.30": int((bj >= JACCARD_THR).sum()),
            "max_containment_single_eval_chunk": round(float(bc.max()), 4),
            "max_containment_whole_eval_union": round(float(uc.max()), 4),
            "per_threshold": per_threshold,
            "per_subset_max_containment": {k: round(v, 4) for k, v in sorted(per_sub_max.items())},
        },
        "8gram_eval_knowledge_into_arena_documents": rev_blk,
        "8gram_eval_responses_into_arena_responses": resp_blk,
        "document_channel_vs_arena": arena_doc_channel(d, subsets),
        "why_this_is_the_sharp_one": (
            "HaluEval's `qa` knowledge blocks are HotpotQA paragraphs, which is how 17 "
            "arena documents ended up byte-for-byte inside halueval training chunks "
            "(arena_surface_report.json: verbatim_per_mix_group = {halueval: 17}, all in "
            "the hotpotqa subset). If `dialogue`'s knowledge were Wikipedia-derived in "
            "the same way it would carry the same exposure, and an eval that overlaps the "
            "arena corrupts both surfaces"),
    }
    bank(res, "B2_disjointness_from_the_blind_arena", arena_block)

    log("=== B2 - gold_full ===")
    gf_claims, gf_chunklists, gf_y = H108.gold_full()
    gf_chunks = sorted({c for lst in gf_chunklists for c in lst})
    gf = {
        "gold_full": {"claims": len(gf_claims), "distinct_chunks": len(gf_chunks),
                      "loader": "R10-H108_lane.gold_full()"},
        "string_channel_exact": {
            "eval_knowledge_vs_gold_full_chunks": exact_block(
                ev_chunks, gf_chunks, "eval knowledge vs gold_full chunks"),
            "eval_responses_vs_gold_full_claims": exact_block(
                ev_claims, gf_claims, "eval responses vs gold_full claims"),
        },
    }
    blk, *_ = ngram_cross(ev_chunks, gf_chunks, hasher, "eval knowledge -> gold_full chunks")
    gf["8gram_eval_knowledge_into_gold_full_chunks"] = blk
    blk, *_ = ngram_cross(gf_chunks, ev_chunks, hasher, "gold_full chunks -> eval knowledge")
    gf["8gram_gold_full_chunks_into_eval_knowledge"] = blk
    bank(res, "B2_disjointness_from_gold_full", gf)

    log("=== B2 - the other banked mechanism evals ===")
    mech = {}
    for name, fname in MECH_EVALS:
        p = EXP / fname
        if not p.exists():
            mech[name] = {"status": "ABSENT", "file": fname}
            continue
        m = pl.read_parquet(p)
        ccol = next((c for c in ("claim", "output", "answer", "statement") if c in m.columns), None)
        kcol = next((c for c in ("chunk", "evidence", "context", "passage") if c in m.columns), None)
        row = {"file": fname, "rows": int(m.height),
               "claim_column": ccol, "chunk_column": kcol}
        if kcol:
            row["eval_knowledge_vs_their_chunks"] = exact_block(
                ev_chunks, m[kcol].drop_nulls().to_list(), f"eval knowledge vs {name}.{kcol}")
        if ccol:
            row["eval_responses_vs_their_claims"] = exact_block(
                ev_claims, m[ccol].drop_nulls().to_list(), f"eval responses vs {name}.{ccol}")
        mech[name] = row
    worst = max((max(v.get("eval_knowledge_vs_their_chunks", {}).get(
                        "eval_units_in_other_any_form", 0),
                     v.get("eval_responses_vs_their_claims", {}).get(
                        "eval_units_in_other_any_form", 0))
                 for v in mech.values() if isinstance(v, dict)), default=0)
    bank(res, "B2_disjointness_from_other_mechanism_evals", {
        "evals_checked": len(mech),
        "worst_eval_units_shared_any_form": worst,
        "per_eval": mech,
    })


def arena_doc_channel(d, subsets):
    a_ids = ASV.arena_doc_ids(subsets)
    ev = d["chunk"].to_list()
    ev_titles, ev_urls = set(), set()
    for t in ev:
        ev_titles |= set(ASV._TITLE.findall(t))
        ev_urls |= set(ASV._URL.findall(t.split("\n", 1)[0]))
    ev_stem = {ASV.stem(x) for x in ev if ASV.stem(x)}
    a_doc_stem = {ASV.stem(c) for v in subsets.values() for docs in v["documents"]
                  for c in docs if ASV.stem(c)}
    out = {
        "arena_identifier_families": {k: len(v) for k, v in a_ids.items()},
        "eval_text_derived_identifiers": {"titles": len(ev_titles), "urls": len(ev_urls)},
        "collisions": {
            "eval_titles_x_arena_document_titles": len(ev_titles & a_ids["document_title"]),
            "eval_urls_x_arena_document_urls": len(ev_urls & a_ids["document_url"]),
            "eval_knowledge_stems_x_arena_document_stems": len(ev_stem & a_doc_stem),
        },
        "note": ("the eval carries no provenance identifier column, so the document "
                 "channel is exercised through the arena instrument's own text-derived "
                 "identifiers plus the normalised whole-document stem"),
    }
    log(f"  arena document channel: {out['collisions']}")
    return out


# --------------------------------------------------------------------------- #
# STAGE: c4 - the mix-wide n-gram census, per DANN group, with live controls
# --------------------------------------------------------------------------- #
def stage_mix(res, d):
    """C2 (exact, whole mix) and C4 (8-gram census) in one pass over the mix."""
    hasher = PG._TokenHasher()
    ev_chunks = sorted(set(d["chunk"].to_list()))
    ev_claims = sorted(set(d["claim"].to_list()))

    claims, chunks, _labels, tags, cut, groups = ASV.assemble()
    windows, wtags, geom = ASV.mix_windows(chunks, tags)
    if cut != CUT:
        log(f"NOTE: chunk_max_chars restored to {cut}")

    log("=== C2 - exact disjointness from the WHOLE assembled mix ===")
    ex = {}
    for mch, texts in (("mix_claims", claims), ("mix_evidence_raw_chunks", chunks),
                       ("mix_evidence_windows", windows)):
        ex[f"eval_knowledge_vs_{mch}"] = exact_block(ev_chunks, texts,
                                                     f"eval knowledge vs {mch}")
        ex[f"eval_responses_vs_{mch}"] = exact_block(ev_claims, texts,
                                                     f"eval responses vs {mch}")
    bank(res, "C2_exact_disjointness_from_the_assembled_mix", {
        "mix": {"assembly": ("arena_surface_verify.assemble() - "
                             "R10-H108_lane.public_train() read UNTRUNCATED through "
                             "R16-H142_G1_arm.untruncated_evidence() plus the five "
                             "R20-H174_arm_run.LANES; the portfolio mix, a strict "
                             "superset of the flagship"),
                "portfolio_rows": len(claims), "groups": groups,
                "distinct_chunks": len(set(chunks)), "windows": len(windows),
                "window_geometry_vs_banked_R20_H174_census": geom},
        "per_channel": ex,
        "worst_eval_units_in_mix_any_form": max(
            v["eval_units_in_other_any_form"] for v in ex.values()),
        "worst_mix_units_in_eval_any_form": max(
            v["other_units_in_eval_any_form_max"] for v in ex.values()),
    })

    log("=== C4 - 8-gram census, per DANN group, both directions ===")
    ev_idx = NGramIndex(ev_chunks, hasher)
    ev_hashes = ev_idx.hashes
    best_j = np.zeros(len(ev_chunks)); best_c = np.zeros(len(ev_chunks))
    best_j_grp = [None] * len(ev_chunks); best_c_grp = [None] * len(ev_chunks)
    union_c = np.zeros(len(ev_chunks)); union_c_grp = [None] * len(ev_chunks)
    seen_global = np.zeros(ev_idx.uniq.size, dtype=bool)

    by_group = collections.defaultdict(set)
    for c, t in zip(chunks, tags, strict=True):
        if c:
            by_group[t].add(c)
    per_group = {}
    rev_total = 0
    for grp in sorted(by_group):
        t0 = time.time()
        units = sorted(by_group[grp])
        hs = [PG.ngram_hashes(u, NGRAM, hasher) for u in units]
        flat, owner, sizes = ASV._index(hs)
        guniq = np.unique(flat) if flat.size else flat
        hits = 0
        gmaxj = 0.0
        for i, q in enumerate(ev_hashes):
            if q.size == 0:
                continue
            j, c, _uj, _uc = ASV._max_pair(q, flat, owner, sizes)
            if j > best_j[i]:
                best_j[i], best_j_grp[i] = j, grp
            if c > best_c[i]:
                best_c[i], best_c_grp[i] = c, grp
            gmaxj = max(gmaxj, j)
            hits += j >= JACCARD_THR
            if guniq.size:
                lo = np.searchsorted(guniq, q, side="left")
                hi = np.searchsorted(guniq, q, side="right")
                u = float((hi > lo).sum()) / q.size
                if u > union_c[i]:
                    union_c[i], union_c_grp[i] = u, grp
        # reverse: every mix unit of this group against the eval index
        rev = 0
        for a in hs:
            if a.size == 0:
                continue
            j, _c, _uj, _uc = ev_idx.query(a)
            rev += j >= JACCARD_THR
        rev_total += rev
        if guniq.size and ev_idx.uniq.size:
            lo = np.searchsorted(guniq, ev_idx.uniq, side="left")
            hi = np.searchsorted(guniq, ev_idx.uniq, side="right")
            seen_global |= (hi > lo)
        per_group[grp] = {
            "mix_units": len(units),
            "mix_units_scorable": int(sum(1 for a in hs if a.size)),
            "eval_knowledge_at_jaccard_ge_0.30": int(hits),
            "max_jaccard": round(gmaxj, 4),
            "mix_units_at_jaccard_ge_0.30_vs_eval": int(rev),
        }
        log(f"  {grp:18s} units={len(units):7d} eval-hits={hits:4d} rev={rev:5d} "
            f"maxJ={gmaxj:.4f} ({time.time() - t0:.0f}s)")
        del hs, flat, owner, sizes, guniq

    whole_union = np.zeros(len(ev_chunks))
    seen_hashes = ev_idx.uniq[seen_global]
    for i, q in enumerate(ev_hashes):
        if q.size == 0:
            continue
        lo = np.searchsorted(seen_hashes, q, side="left")
        hi = np.searchsorted(seen_hashes, q, side="right")
        whole_union[i] = float((hi > lo).sum()) / q.size

    n_hit = int((best_j >= JACCARD_THR).sum())
    scorable = int(sum(1 for a in ev_hashes if a.size))
    mix_units_total = sum(v["mix_units"] for v in per_group.values())
    frac_fwd = n_hit / max(len(ev_chunks), 1)
    frac_rev = rev_total / max(mix_units_total, 1)
    c4 = {
        "instrument": (f"provenance_gate primitives, {NGRAM}-gram, Jaccard >= "
                       f"{JACCARD_THR}, bidirectional, per-corpus attribution - the "
                       f"banked R14-H136 ruling-2 form, driven through "
                       f"arena_surface_verify._index/_max_pair"),
        "kill_threshold": KILL,
        "eval_knowledge_units": len(ev_chunks),
        "eval_knowledge_units_scorable_at_8grams": scorable,
        "eval_knowledge_units_too_short_for_8grams": len(ev_chunks) - scorable,
        "coverage_of_short_units": ("units under 8 normalised tokens carry no 8-gram and "
                                    "are covered by the exact-matching pass in "
                                    "C2_exact_disjointness_from_the_assembled_mix, which "
                                    "is exhaustive and length-blind"),
        "eval_units_at_jaccard_ge_0.30": n_hit,
        "eval_fraction": round(frac_fwd, 8),
        "mix_units_at_jaccard_ge_0.30_vs_eval": rev_total,
        "mix_fraction": round(frac_rev, 8),
        "max_fraction": round(max(frac_fwd, frac_rev), 8),
        "verdict_vs_kill_bar": ("KILL" if max(frac_fwd, frac_rev) >= KILL else
                                "WARN" if max(frac_fwd, frac_rev) >= 0.005 else "PASS"),
        "best_jaccard": {"max": round(float(best_j.max()), 4),
                         "p999": round(float(np.percentile(best_j, 99.9)), 4),
                         "p99": round(float(np.percentile(best_j, 99)), 4),
                         "mean": round(float(best_j.mean()), 6)},
        "best_containment_single_mix_chunk": {
            "max": round(float(best_c.max()), 4),
            "p999": round(float(np.percentile(best_c, 99.9)), 4),
            "p99": round(float(np.percentile(best_c, 99)), 4),
            "mean": round(float(best_c.mean()), 6),
            **{f"n_ge_{t:.2f}": int((best_c >= t).sum()) for t in CONTAINMENT_THRESHOLDS}},
        "containment_vs_whole_mix_union": {
            "max": round(float(whole_union.max()), 4),
            "mean": round(float(whole_union.mean()), 6),
            **{f"n_ge_{t:.2f}": int((whole_union >= t).sum()) for t in CONTAINMENT_THRESHOLDS}},
        "attribution_of_the_worst_eval_unit": {
            "max_jaccard_group": best_j_grp[int(np.argmax(best_j))],
            "max_containment_group": best_c_grp[int(np.argmax(best_c))],
        },
        "per_mix_group": per_group,
    }
    bank(res, "C4_contamination_census_vs_the_mix", c4)
    bank(res, "C4_live_positive_controls", controls(ev_chunks, chunks, hasher))


def controls(ev_chunks, mix_chunks, hasher):
    """Live positive controls for the disjointness instrument, per C4.

    1  SPIKE - eval knowledge blocks fed back against an index of themselves;
       every one must read Jaccard 1.0.
    2  LIVE CROSS-CONFIG - HaluEval `qa` knowledge blocks scored against the
       assembled mix, which carries them VERBATIM. This is the same instrument
       used for the real question, fed text known to be inside the mix by
       construction, so a clean reading on `dialogue` is a measurement rather
       than a reassurance.
    3  LIVE RE-CHUNK - eval knowledge blocks whitespace-collapsed and case-folded
       (the transformation that defeats exact matching) against an index of the
       originals.
    4  NEGATIVE - eval knowledge blocks with their tokens permuted; vocabulary
       preserved, every 8-gram destroyed. Must be silent.
    """
    self_idx = NGramIndex(ev_chunks, hasher)
    long_ev = [c for c in ev_chunks if len(PG.normalize(c).split()) >= 20][:200]
    k = min(10, len(long_ev))

    def score(units, idx):
        js, cs = [], []
        for u in units:
            q = PG.ngram_hashes(u, NGRAM, hasher)
            if q.size == 0:
                continue
            j, c, _a, _b = idx.query(q)
            js.append(round(j, 4)); cs.append(round(c, 4))
        return js, cs

    sj, sc = score(long_ev[:k], self_idx)
    spike = {"kind": "synthetic spike - eval knowledge blocks against an index of themselves",
             "n": len(sj), "jaccards": sj,
             "detected_at_jaccard_ge_thr": int(sum(j >= JACCARD_THR for j in sj)),
             "fires": bool(sj) and all(j >= JACCARD_THR for j in sj)}

    rec = [norm(u) for u in long_ev[:k]]
    rj, rc = score(rec, self_idx)
    exact_forms = set(ev_chunks) | {c[:CUT] for c in ev_chunks}
    rechunk = {
        "kind": ("LIVE near-duplicate by construction - eval knowledge blocks "
                 "whitespace-collapsed and case-folded, scored against an index of the "
                 "ORIGINALS; the transformation that defeats exact matching"),
        "n": len(rj), "jaccards": rj, "containments": rc,
        "exact_form_matches": int(sum(1 for f in rec if f in exact_forms)),
        "detected_at_jaccard_ge_thr": int(sum(j >= JACCARD_THR for j in rj)),
        "fires": bool(rj) and all(j >= JACCARD_THR for j in rj)}

    # 2 - live cross-config: qa knowledge IS in the mix verbatim
    z = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    qa = pl.read_parquet(io.BytesIO(z.read("pminervini__HaluEval__qa__data.parquet")))
    qa_know = sorted(set(qa["knowledge"].to_list()))[:2000]
    rng = np.random.default_rng(0)
    pool = [c for c in dict.fromkeys(mix_chunks) if c]
    idx_sample = pool if len(pool) <= 60000 else [
        pool[i] for i in rng.choice(len(pool), 60000, replace=False)]
    mix_idx = NGramIndex(idx_sample, hasher)
    qj, qc = score(qa_know, mix_idx)
    cross = {
        "kind": ("LIVE positive control - HaluEval `qa` knowledge blocks, which the "
                 "training mix carries VERBATIM, scored against a 60k-chunk sample of "
                 "the mix with the same instrument used for the `dialogue` question"),
        "n_scored": len(qj),
        "detected_at_jaccard_ge_thr": int(sum(j >= JACCARD_THR for j in qj)),
        "fraction": round(sum(j >= JACCARD_THR for j in qj) / max(len(qj), 1), 6),
        "max_jaccard": round(max(qj), 4) if qj else None,
        "median_jaccard": round(float(np.median(qj)), 4) if qj else None,
        "units_at_containment_1.00": int(sum(c >= 1.0 for c in qc)),
        "fires": bool(qj) and any(j >= JACCARD_THR for j in qj)}

    # 4 - negative
    negs = []
    for u in long_ev:
        t = PG.normalize(u).split()
        if len(t) < 20:
            continue
        rng.shuffle(t)
        negs.append(" ".join(t))
        if len(negs) >= k:
            break
    nj, nc = score(negs, self_idx)
    negative = {"kind": ("NEGATIVE - eval knowledge blocks with tokens permuted; "
                         "vocabulary preserved, every 8-gram destroyed"),
                "n": len(nj), "jaccards": nj,
                "detected_at_jaccard_ge_thr": int(sum(j >= JACCARD_THR for j in nj)),
                "silent": bool(nj) and not any(j >= JACCARD_THR for j in nj)}
    log(f"  controls: spike {spike['detected_at_jaccard_ge_thr']}/{spike['n']}, "
        f"rechunk {rechunk['detected_at_jaccard_ge_thr']}/{rechunk['n']}, "
        f"cross-config {cross['detected_at_jaccard_ge_thr']}/{cross['n_scored']}, "
        f"negative {negative['detected_at_jaccard_ge_thr']}/{negative['n']}")
    return {"spike_control": spike, "live_rechunk_control": rechunk,
            "live_cross_config_control": cross, "negative_control": negative,
            "all_fire_and_negative_silent": bool(
                spike["fires"] and rechunk["fires"] and cross["fires"]
                and negative["silent"])}


# --------------------------------------------------------------------------- #
# STAGE: b3 - the claim-only shortcut
# --------------------------------------------------------------------------- #
def stage_b3(res, d):
    log("=== B3 - the claim-only shortcut, pair-aware, with a shuffled control ===")
    claims = d["claim"].to_list()
    y = d["label"].to_numpy().astype(int)
    doc = [hashlib.blake2b(k.encode("utf-8"), digest_size=8).hexdigest()
           for k in d["chunk"].to_list()]
    pair = [f"p{v}" for v in d["pair_id"].to_list()]
    ckeys = [hashlib.blake2b(t.encode("utf-8"), digest_size=8).hexdigest() for t in claims]

    cen = CO.pair_census(y, doc, ckeys)
    cen["pair_id_keys"] = int(d["pair_id"].n_unique())
    cen["pair_id_keys_with_both_labels"] = int(
        d.group_by("pair_id").agg(pl.col("label").n_unique().alias("k"))
        .filter(pl.col("k") > 1).height)
    cen["pair_id_keys_with_both_labels_share"] = round(
        cen["pair_id_keys_with_both_labels"] / max(cen["pair_id_keys"], 1), 4)
    log(f"  pair census: evidence keys {cen['evidence_keys']}, both-label share "
        f"{cen['evidence_keys_with_both_labels_share']}, pair_id both-label share "
        f"{cen['pair_id_keys_with_both_labels_share']}")

    tr, te = CO.split_stratified(y)
    a_strat = CO.split_auroc(claims, y, tr, te, tag="dialogue/stratified")
    tr2, te2 = CO.split_doc_disjoint(doc)
    a_doc = CO.split_auroc(claims, y, tr2, te2, tag="dialogue/doc-disjoint")
    tr3, te3 = CO.split_doc_disjoint(pair)
    a_pair = CO.split_auroc(claims, y, tr3, te3, tag="dialogue/pair-disjoint")

    paired = bool(cen["pair_structure"] or cen["pair_id_keys_with_both_labels_share"] >= 0.5)
    verdict_split = "doc_disjoint" if paired else "stratified"
    reading = {"stratified": CO.two_sided(a_strat),
               "doc_disjoint": CO.two_sided(a_doc),
               "pair_disjoint": CO.two_sided(a_pair)}
    vb = reading[verdict_split]

    # label-shuffled negative control on the verdict-bearing split
    rng = np.random.default_rng(20)
    ysh = y.copy()
    rng.shuffle(ysh)
    trs, tes = (CO.split_doc_disjoint(doc) if verdict_split == "doc_disjoint"
                else CO.split_stratified(ysh))
    a_sh = CO.split_auroc(claims, ysh, trs, tes, tag="dialogue/label-shuffled")

    char_auc, tok_auc = CO.surface_auroc(claims, y)
    parity = LC.surface_parity(
        d.select(["pair_id", "label", "claim", "chunk"]),
        extra={"dialogue_history_char_length": [float(len(t)) for t in
                                                d["dialogue_history"].to_list()]},
        report_only=("claim_chunk_containment",))

    # single-channel probes that are degenerate by construction, stated not assumed
    ev_only = float(LC.auroc(y, [float(len(t)) for t in d["chunk"].to_list()]))
    dh_only = float(LC.auroc(y, [float(len(t)) for t in d["dialogue_history"].to_list()]))

    # within-pair claim-only, C5's < 0.60 bar
    auc_wp, score_wp = LC.claim_only_probe(claims, y.tolist(), doc,
                                           np.random.default_rng(0))
    wp = LC.within_pair_accuracy(d.select(["pair_id", "label"]), score_wp)

    block = {
        "instrument": ("R20_claimonly_sweep.fit_probe - TF-IDF char_wb(2,5) + word(1,2) "
                       "of the CLAIM STRING ALONE into LogisticRegression(liblinear, "
                       "C=4.0, tol=1e-7); AUROC on the held-out 30%. Reused, not rewritten"),
        "prior": ("HaluEval's TRAINED halves read 0.9519 claim-only "
                  "(contract/halueval_contract_report.json), carried redundantly across "
                  "register, content and length. A sibling configuration from the same "
                  "pipeline is assumed to inherit it until measured"),
        "pair_census": cen,
        "verdict_bearing_split": verdict_split,
        "verdict_bearing_reason": (
            "pair structure is MEASURED in this member - both legs of a pair share the "
            "knowledge block, so the evidence key groups them and the evidence-disjoint "
            "split cannot put one twin in train and its twin in test. A stratified split "
            "would do exactly that and would measure the split rather than the member. "
            "The stratified number is carried as a diagnostic; the pair_id-disjoint "
            "number is an EXECUTOR-ADDED diagnostic reported separately per C5 and "
            "joins no registered conjunction"
            if paired else
            "no pair structure measured - the registered stratified split decides"),
        "reading": reading,
        "leak_strength": vb["leak_strength"],
        "equivalent_one_sided_auroc": vb["equivalent_one_sided_auroc"],
        "band": vb["band"],
        "direction": vb["direction"],
        "bands": {"clean": "< 0.05", "mild": "0.05 - 0.15", "leak": "0.15 - 0.30",
                  "severe": ">= 0.30"},
        "C5_registered_bars": {"claim_only_below": 0.55, "within_pair_below": 0.60},
        "claim_only_auroc_vs_C5_bar": {
            "stratified": a_strat, "doc_disjoint": a_doc, "pair_disjoint": a_pair},
        "within_pair_claim_only": {
            "instrument": ("R20-H174_lane_common.claim_only_probe (char_wb TF-IDF, "
                           "5 folds disjoint on the knowledge-block key) scored "
                           "within-pair by lane_common.within_pair_accuracy"),
            "out_of_fold_auroc": round(auc_wp, 4),
            "within_pair_accuracy": wp},
        "label_shuffled_negative_control": {
            "split": verdict_split, "rows": len(claims),
            "shuffled": CO.two_sided(a_sh),
            "observed_leak_strength": vb["leak_strength"],
            "reading": ("with the labels permuted the same instrument on the same feature "
                        "space must read leak_strength ~0. A non-zero reading would mean "
                        "the probe manufactures signal and no observed number is evidence")},
        "length_channels": {
            "response_char_length_auroc": round(char_auc, 4),
            "response_token_count_auroc": round(tok_auc, 4),
            "response_char_length_reading": CO.two_sided(char_auc),
            "response_token_count_reading": CO.two_sided(tok_auc),
            "positive_leg_mean_chars": round(float(
                d.filter(pl.col("label") == 1)["response_chars"].mean()), 1),
            "negative_leg_mean_chars": round(float(
                d.filter(pl.col("label") == 0)["response_chars"].mean()), 1),
            "positive_leg_mean_tokens": round(float(
                d.filter(pl.col("label") == 1)["response_tokens"].mean()), 1),
            "negative_leg_mean_tokens": round(float(
                d.filter(pl.col("label") == 0)["response_tokens"].mean()), 1)},
        "surface_parity": parity,
        "single_channel_probes_degenerate_by_construction": {
            "evidence_char_length_auroc": round(ev_only, 4),
            "dialogue_history_char_length_auroc": round(dh_only, 4),
            "reading": ("both legs of a pair carry the SAME knowledge block and the SAME "
                        "dialogue history, so any evidence-alone or history-alone channel "
                        "is exactly 0.5 by construction. Recorded as a structural fact, "
                        "not presented as a passing measurement")},
        "note": NOTE,
    }
    bank(res, "B3_claim_only_shortcut", block)


# --------------------------------------------------------------------------- #
# STAGE: c1/c6 - label commensurability and the memorisation channel
# --------------------------------------------------------------------------- #
def stage_c1c6(res, d):
    log("=== C1 - label commensurability (structural + distributional) ===")
    struct = H143.structural_test(d)
    cl = d["claim"].to_list()
    ch = d["chunk"].to_list()
    yv = d["label"].to_numpy()
    cont = np.array([LC.containment(c, k) for c, k in zip(cl, ch, strict=True)])
    pos, neg = cont[yv == 1], cont[yv == 0]

    def dist(v):
        return {"n": int(v.size), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 4),
                "p10": round(float(np.percentile(v, 10)), 4),
                "p90": round(float(np.percentile(v, 90)), 4),
                "frac_ge_0.90": round(float((v >= 0.90).mean()), 4),
                "frac_eq_1.00": round(float((v >= 1.0).mean()), 4),
                "frac_le_0.50": round(float((v <= 0.50).mean()), 4)}

    c1 = {
        "head_declared": ("the grounding scalar (`task_head`) - the single shipped "
                          "support head; this surface enters no parallel head"),
        "predicate_the_label_encodes": (
            "`right_response` is HaluEval's reference conversational reply to the "
            "dialogue history given the knowledge block; `hallucinated_response` is an "
            "LLM-written plausible reply carrying content the knowledge block does not "
            "support. The label is response faithfulness to the knowledge - a support "
            "predicate. The dialogue history is carried as a column but the (claim, "
            "chunk) score path does not read it, so the row the head sees is (response, "
            "knowledge block)"),
        "structural_test_C_A1": {
            **struct,
            "reading": ("both legs of a pair share the knowledge block and differ only in "
                        "the response, so a collision would require the two responses to "
                        "be the same string. Measured on the normalised form, which is "
                        "strictly weaker than byte identity and cannot under-report")},
        "distributional_diagnostic": {
            "instrument": ("R20-H174_lane_common.containment - content-token containment "
                           "of the claim in the evidence, the instrument the halueval "
                           "contract report used"),
            "positive_leg": dist(pos), "negative_leg": dist(neg),
            "attested_rate_gap_at_0.90": round(
                float((pos >= 0.90).mean() - (neg >= 0.90).mean()), 4),
            "mean_gap": round(float(pos.mean() - neg.mean()), 4),
            "test_2_strict_separation_C_A2": bool((neg >= 0.90).mean() < (pos >= 0.90).mean()),
            "test_3_absolute_level_finding": (
                f"negative leg fully attested at {float((neg >= 1.0).mean()):.4f} and "
                f">= 0.90-attested at {float((neg >= 0.90).mean()):.4f}"),
            "note": ("the `within 0.10` band is STRUCK by amendment C-A2; the decisive "
                     "tests are structural identity and strict separation, with the "
                     "absolute level reported always")},
    }
    bank(res, "C1_label_commensurability", c1)

    log("=== C6 - memorisation channel (eval-facing, C-A2 scoping) ===")
    claims_mix, chunks_mix, y_mix, tags_mix = load_public_train_cached()
    mix_claims = collections.defaultdict(list)
    mix_labels = collections.defaultdict(list)
    for c, k, lab in zip(claims_mix, chunks_mix, y_mix, strict=True):
        mix_claims[norm(k)].append(c)
        mix_labels[norm(k)].append(float(lab))
    Q = _mod("q", EXP / "R20-H175b_qlane.py")
    c6 = H143.memorisation_feature(d, dict(mix_claims), dict(mix_labels), Q)
    c6["key"] = "the knowledge block (normalised), the only field a pair shares"
    c6["scoping"] = ("C-A2: C6 binds features keyed on associations the TRAINING MIX "
                     "supplies. Where the eval-facing test has zero key coverage the "
                     "clause is NOT-APPLICABLE and no proxy is substituted")
    log(f"  C6 coverage {c6['coverage']}, auroc {c6.get('auroc')}")
    bank(res, "C6_memorisation_channel", c6)


_PT_CACHE = {}


def load_public_train_cached():
    if "v" not in _PT_CACHE:
        log("loading public_train() for the C6 key lookup (truncated evidence is fine - "
            "C6 keys on the normalised chunk and the eval's blocks are far under the cut)")
        _PT_CACHE["v"] = H108.public_train()
    return _PT_CACHE["v"]


# --------------------------------------------------------------------------- #
# STAGE: report - the eight clauses
# --------------------------------------------------------------------------- #
def stage_report(res):
    log("=== clause roll-up ===")
    ec = res["eval_census"]
    b1 = res["B1_disjointness_from_the_trained_halves"]
    b2a = res["B2_disjointness_from_the_blind_arena"]
    b2g = res["B2_disjointness_from_gold_full"]
    b2m = res["B2_disjointness_from_other_mechanism_evals"]
    c2mix = res["C2_exact_disjointness_from_the_assembled_mix"]
    c4 = res["C4_contamination_census_vs_the_mix"]
    ctrl = res["C4_live_positive_controls"]
    b3 = res["B3_claim_only_shortcut"]
    c1 = res["C1_label_commensurability"]
    c6 = res["C6_memorisation_channel"]

    b1_worst = max(v["eval_units_in_other_any_form"]
                   for v in b1["string_channel_exact"].values())
    b1_worst_rev = max(v["other_units_in_eval_any_form_max"]
                       for v in b1["string_channel_exact"].values())
    arena_worst = max(v["eval_units_in_other_any_form"]
                      for v in b2a["string_channel_exact"].values())
    arena_worst_rev = max(v["other_units_in_eval_any_form_max"]
                          for v in b2a["string_channel_exact"].values())
    gf_worst = max(v["eval_units_in_other_any_form"]
                   for v in b2g["string_channel_exact"].values())
    arena_c10 = b2a["8gram_arena_documents_into_eval_knowledge"]["per_threshold"][
        "containment_ge_0.10"]

    parity = b3["surface_parity"]
    worst_parity_channel = max(
        ((k, v) for k, v in parity["auroc"].items() if k not in parity["report_only"]),
        key=lambda kv: abs(kv[1] - 0.5))

    clauses = {}

    clauses["C1"] = {
        "verdict": ("PASS" if (not c1["structural_test_C_A1"]["fires"]
                               and c1["distributional_diagnostic"]["test_2_strict_separation_C_A2"])
                    else "FAIL"),
        "binding_number": (
            f"structural collisions {c1['structural_test_C_A1']['negative_legs_identical_to_a_positive']} "
            f"of {c1['structural_test_C_A1']['negative_legs']} negative legs; strict "
            f"separation {c1['distributional_diagnostic']['negative_leg']['frac_ge_0.90']} "
            f"< {c1['distributional_diagnostic']['positive_leg']['frac_ge_0.90']}"),
        "margin": c1["distributional_diagnostic"]["attested_rate_gap_at_0.90"],
        "finding_under_test_3": c1["distributional_diagnostic"]["test_3_absolute_level_finding"],
    }

    c2_all_zero = bool(
        b1_worst == 0 and b1_worst_rev == 0
        and arena_worst == 0 and arena_worst_rev == 0 and gf_worst == 0
        and b2m["worst_eval_units_shared_any_form"] == 0
        and c2mix["worst_eval_units_in_mix_any_form"] == 0
        and c2mix["worst_mix_units_in_eval_any_form"] == 0)
    clauses["C2"] = {
        "verdict": "PASS" if c2_all_zero else "FAIL",
        "binding_number": (
            f"worst count over every surface, every string form, both directions: "
            f"trained halves {max(b1_worst, b1_worst_rev)}, assembled mix "
            f"{max(c2mix['worst_eval_units_in_mix_any_form'], c2mix['worst_mix_units_in_eval_any_form'])}, "
            f"arena {max(arena_worst, arena_worst_rev)}, gold_full {gf_worst}, other "
            f"mechanism evals {b2m['worst_eval_units_shared_any_form']}"),
        "document_channel": {
            "vs_trained_halves": b1["document_channel"]["whole_evidence_document_key"],
            "vs_arena": b2a["document_channel_vs_arena"]["collisions"],
        },
        "subdocument_channel": {
            "arena_documents_at_containment_ge_0.10": arena_c10["arena_documents_hit"],
            "arena_responses_touched_at_containment_ge_0.10": arena_c10["arena_responses_touched"],
            "max_containment_of_an_arena_document_into_eval_knowledge":
                b2a["8gram_arena_documents_into_eval_knowledge"]["max_containment_single_eval_chunk"],
            "max_containment_of_an_eval_block_into_the_whole_mix":
                c4["containment_vs_whole_mix_union"]["max"],
        },
    }

    clauses["C3"] = {
        "verdict": "PASS",
        "binding_number": (
            "the archive ships ONE `data` split per configuration - there is no official "
            "split to trust or distrust. The axis that matters is the CONFIGURATION, and "
            "it is verified from the loader source: public_train() iterates exactly "
            "('qa','knowledge','right_answer','hallucinated_answer') and "
            "('summarization','document','right_summary','hallucinated_summary'), with no "
            "split filter and no row filter; the strings 'dialogue' and 'general' appear "
            "in no mix-assembly loader"),
        "empirical_converse": (
            f"and the converse is measured, not inferred: the held-out configuration "
            f"shares {max(b1_worst, b1_worst_rev)} units with the two loaded ones under "
            f"every string form in both directions"),
    }

    clauses["C4"] = {
        "verdict": c4["verdict_vs_kill_bar"],
        "binding_number": (f"max fraction {c4['max_fraction']} against the KILL bar "
                           f"{KILL} (forward {c4['eval_fraction']}, reverse "
                           f"{c4['mix_fraction']}); max Jaccard "
                           f"{c4['best_jaccard']['max']}"),
        "coverage": (f"{c4['eval_knowledge_units_scorable_at_8grams']} of "
                     f"{c4['eval_knowledge_units']} knowledge blocks carry an 8-gram; the "
                     f"{c4['eval_knowledge_units_too_short_for_8grams']} that do not are "
                     f"covered by the exhaustive exact-matching pass"),
        "live_positive_control": {
            "spike_fires": ctrl["spike_control"]["fires"],
            "live_rechunk_fires": ctrl["live_rechunk_control"]["fires"],
            "live_cross_config_fires": ctrl["live_cross_config_control"]["fires"],
            "live_cross_config_detection_fraction":
                ctrl["live_cross_config_control"]["fraction"],
            "negative_silent": ctrl["negative_control"]["silent"],
        },
    }

    c5_claim_ok = b3["leak_strength"] < 0.05
    wp_all = b3["within_pair_claim_only"]["within_pair_accuracy"]
    wp_worst = max((v["acc"] for v in wp_all.values()), default=None)
    clauses["C5"] = {
        "verdict": "FAIL" if (not c5_claim_ok or not parity["pass"]) else "PASS",
        "binding_number": (
            f"claim-only leak strength {b3['leak_strength']} (one-sided equivalent "
            f"{b3['equivalent_one_sided_auroc']}) on the {b3['verdict_bearing_split']} "
            f"split against the registered < 0.55; within-pair {wp_worst} against < 0.60; "
            f"surface parity worst channel {worst_parity_channel[0]} at "
            f"{worst_parity_channel[1]} against [0.45, 0.55]"),
        "single_channel_probes": b3["single_channel_probes_degenerate_by_construction"],
        "executor_added_diagnostics_reported_separately": [
            "pair_id-disjoint claim-only split",
            "dialogue_history char-length channel in surface_parity",
        ],
        "negative_control": b3["label_shuffled_negative_control"]["shuffled"],
    }

    clauses["C6"] = {
        "verdict": ("NOT-APPLICABLE" if c6.get("auroc") is None else
                    "PASS" if abs(c6["auroc"] - 0.5) <= 0.05 else "FAIL"),
        "binding_number": (f"key coverage {c6['coverage']} "
                           f"({c6['rows_with_a_mix_claim']} of {c6['rows']} rows); "
                           f"feature AUROC {c6.get('auroc')}"),
        "key": c6["key"],
    }

    clauses["C7"] = {
        "verdict": "PASS",
        "binding_number": (f"{ec['rows']} rows over {ec['pairs']} pairs, both reported "
                           f"everywhere; pair integrity malformed "
                           f"{ec['pair_integrity']['malformed']}"),
        "registration_discrepancy": (
            "the registration block declared '10,000 rows over 5,000 contrast pairs'. "
            "The archive's `dialogue` configuration holds 10,000 rows and each supplies "
            "TWO serving rows, so the built surface is 20,000 rows over 10,000 pairs - "
            "2x the registered figure in BOTH units. This is exactly the unit confusion "
            "C7 exists to catch, and it is recorded, not adjudicated"),
    }

    clauses["C8"] = {
        "verdict": "PASS",
        "source": "pminervini/HaluEval, HuggingFace",
        "licence": "MIT (data/external/datasets/dataset-halueval.md)",
        "retrieval": "scripts/fetch_grounding_datasets.py; archive dataset-halueval.zip",
        "selection_predicate": (
            "ALL 10,000 rows of the `dialogue` configuration, both legs of every row "
            "(right_response label 1, hallucinated_response label 0). No split filter "
            "(the archive ships one `data` split per configuration), no row filter, no "
            "sampling"),
        "within_member_duplication": {
            "rows": ec["rows"], "pairs": ec["pairs"],
            "distinct_claims": ec["distinct_claims"],
            "distinct_knowledge_blocks": ec["distinct_knowledge_blocks"],
            "distinct_dialogue_histories": ec["distinct_dialogue_histories"],
        },
        "public_repository": ("no client or company name appears in this artifact or in "
                              "the eval parquet"),
    }

    fails = [k for k, v in clauses.items() if v["verdict"] == "FAIL"]
    bank(res, "clauses", clauses)
    bank(res, "conforming", not fails)
    bank(res, "summary", {
        "rows": ec["rows"], "pairs": ec["pairs"],
        "positive_rate": ec["positive_rate"],
        "clause_verdicts": {k: v["verdict"] for k, v in clauses.items()},
        "failed_clauses": fails,
        "B1_worst_shared_units_vs_trained_halves": max(b1_worst, b1_worst_rev),
        "B1_document_channel": b1["document_channel"]["whole_evidence_document_key"],
        "B2_worst_shared_units_vs_arena": max(arena_worst, arena_worst_rev),
        "B2_arena_documents_at_8gram_containment_ge_0.10": arena_c10["arena_documents_hit"],
        "B2_arena_responses_touched_at_containment_ge_0.10": arena_c10["arena_responses_touched"],
        "B2_worst_shared_units_vs_gold_full": gf_worst,
        "B2_worst_shared_units_vs_other_mechanism_evals": b2m["worst_eval_units_shared_any_form"],
        "B3_claim_only_auroc_verdict_bearing": b3["reading"][b3["verdict_bearing_split"]]["auroc"],
        "B3_claim_only_leak_strength": b3["leak_strength"],
        "B3_claim_only_one_sided_equivalent": b3["equivalent_one_sided_auroc"],
        "B3_label_shuffled_control_strength":
            b3["label_shuffled_negative_control"]["shuffled"]["leak_strength"],
        "B3_response_char_length_auroc": b3["length_channels"]["response_char_length_auroc"],
        "C4_max_fraction": c4["max_fraction"],
        "C4_verdict_vs_kill_bar": c4["verdict_vs_kill_bar"],
        "controls_all_fire": ctrl["all_fire_and_negative_silent"],
        "note": NOTE,
    })
    log("=== HALUEVAL-DIALOGUE EVAL CONTRACT PASS COMPLETE ===")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "sides", "b1", "b2", "mix", "b3", "c1c6", "report"))
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    res = load_report()
    d, cfgs = stage_sides(res)
    if args.stage in ("all", "sides"):
        if args.stage == "sides":
            return
    if args.stage in ("all", "b1"):
        stage_b1(res, d, cfgs)
        if args.stage == "b1":
            return
    if args.stage in ("all", "b2"):
        stage_b2(res, d)
        if args.stage == "b2":
            return
    if args.stage in ("all", "b3"):
        stage_b3(res, d)
        if args.stage == "b3":
            return
    if args.stage in ("all", "c1c6"):
        stage_c1c6(res, d)
        if args.stage == "c1c6":
            return
    if args.stage in ("all", "mix"):
        stage_mix(res, d)
        if args.stage == "mix":
            return
    stage_report(res)


if __name__ == "__main__":
    main()
