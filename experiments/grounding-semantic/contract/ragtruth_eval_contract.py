"""Dataset-contract verification for the R21-H180 RAGTruth held-out EVAL. CPU ONLY.

Contract: docs/experiments/dataset-contract.md, clauses C1-C8 INCLUDING amendments
C-A1 (containment scoped to C1; C1's decisive test is structural) and C-A2 (C1's
`within 0.10` band struck, replaced by strict separation plus reported absolute
level; C6 scoped to mix-supplied associations, NOT-APPLICABLE at zero key coverage).

Artifact under test: `experiments/grounding-semantic/R21-H180_ragtruth_eval.parquet`
(built by R21-H180_eval_build.py - the English RAGTruth TEST split, 2,700 rows over
450 contexts, flat long serving shape).

Instruments are reused, never reinvented:
  provenance_gate.py   R14-H136 ruling-2 form - 8-gram, Jaccard >= 0.3,
                       bidirectional, KILL > 2%, spike control; thresholds read
                       from R19_supply_gates.py
  R20-H174_lane_common containment / auroc / tokens
  R20_claimonly_sweep  the claim-only probe (TF-IDF char_wb(2,5) + word(1,2) ->
                       liblinear), its two split rules, its two-sided reading and
                       its label-shuffled negative control

The training mix is the FLAGSHIP assembly the six banked endpoints trained on:
`R10-H108_lane.public_train()` under `R16-H142_G1_arm.untruncated_evidence()`
(685,670 rows, 12 groups) plus `R17-H146_lane.parquet` (quant_misbind, 30,000)
and `R18-H150_scaleunit_lane.parquet` (quant_scale_unit, 5,540) = 721,210 rows.

DOCUMENT CHANNEL - the string channel is not the whole test. Three separate reads
today came out an order of magnitude worse on document identity than on string
equality (`R20-H177_eval_B`: 4.5% by string, 65% by document). The document
channel here is n-gram CONTAINMENT: what fraction of an eval unit's 8-grams a
single reference unit covers. Coverage 1.0 means the eval unit is a sub-document
of that reference unit, which neither exact nor whitespace-normalised string
matching can see. Contexts are additionally decomposed on their blank-line
junctions, because a RAGTruth QA context is several retrieved passages joined.

Nothing is adjudicated here and no model is scored.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
        experiments/grounding-semantic/contract/ragtruth_eval_contract.py
"""

import os

# HARD CONSTRAINT: no GPU. Set BEFORE any banked import - the banked modules pin
# CUDA_VISIBLE_DEVICES with setdefault, which leaves an already-set value alone.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import collections
import importlib.util
import io
import json
import pathlib
import re
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
EVAL = EXP / "R21-H180_ragtruth_eval.parquet"
OUT = HERE / "ragtruth_eval_report.json"

MEMBER = "R21-H180_ragtruth_eval"
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 721_210
LANES = (("R17-H146_lane.parquet", "quant_misbind", 30_000),
         ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540))
DOC_COVERAGE_BAR = 0.90   # executor-applied document-identity threshold, flagged

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LC = _mod("lane_common", EXP / "R20-H174_lane_common.py")     # torch-free
G = _mod("provgate", EXP / "provenance_gate.py")              # torch-free
CO = _mod("claimonly", EXP / "R20_claimonly_sweep.py")        # torch-free, guarded main

_gsrc = (EXP / "R19_supply_gates.py").read_text()
GATE_N = int(_gsrc.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gsrc.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gsrc.split("GATE_KILL = ")[1].split("\n")[0])

ARM = _mod("g1arm", EXP / "R16-H142_G1_arm.py")               # imports torch, CPU only
H108 = ARM.H108
CHUNK_MAX = ARM.M59.CFG.chunk_max_chars

WS = re.compile(r"\s+")


def wsnorm(s):
    return WS.sub(" ", s).strip().casefold()


def dist(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "p99": round(float(np.percentile(v, 99)), 4),
        "max": round(float(v.max()), 4),
    }


def checkpoint(report):
    OUT.write_text(json.dumps(report, indent=2))


# --------------------------------------------------------------------------- #
# the flagship training mix
# --------------------------------------------------------------------------- #
def flagship_mix():
    """Claims, untruncated evidence, labels and DANN group tag, all 721,210 rows."""
    t0 = time.time()
    with ARM.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    if len(claims) != EXPECTED_CLEAN_ROWS:
        raise SystemExit(f"MIX ABORT: clean mix {len(claims)} rows, "
                         f"expected {EXPECTED_CLEAN_ROWS}")
    log(f"mix: clean public {len(claims)} rows over {len(set(tags))} groups "
        f"({time.time() - t0:.0f}s)")
    for fname, group, n_rows in LANES:
        d = pl.read_parquet(EXP / fname)
        if d.height != n_rows:
            raise SystemExit(f"LANE ABORT ({group}): {d.height} rows, expected {n_rows}")
        ch = "chunk" if "chunk" in d.columns else "evidence"
        claims += d["claim"].to_list()
        chunks += d[ch].to_list()
        y = np.concatenate([y, d["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * d.height
        log(f"mix: lane {group} {d.height} rows")
    if len(claims) != EXPECTED_MIX_ROWS:
        raise SystemExit(f"MIX ABORT: flagship mix {len(claims)} rows, "
                         f"expected {EXPECTED_MIX_ROWS}")
    log(f"mix assembled: {len(claims)} rows, {len(set(tags))} groups")
    return claims, chunks, y, tags


def group_index(tags):
    idx = collections.defaultdict(list)
    for i, t in enumerate(tags):
        idx[t].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in sorted(idx.items())}


# --------------------------------------------------------------------------- #
# evaluation surfaces other than this one
# --------------------------------------------------------------------------- #
def other_surfaces():
    out = {}
    arena_docs, _ = G.load_arena()
    z = zipfile.ZipFile(G.ARCHIVE)
    resp = []
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(pl.col("adherence_score").is_not_null()
                       & (pl.col("response").str.len_chars() > 20)
                       & (pl.col("documents").list.len() > 0))
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        resp += df.sample(min(G.N_PER_SUBSET, len(df)), seed=0)["response"].to_list()
    out["arena_ragbench_10_subsets"] = {
        "evidence": [c for v in arena_docs.values() for c in v], "claims": resp}

    gc, gk, _gy = H108.gold_full()
    out["gold_full"] = {"evidence": [c for ks in gk for c in ks], "claims": list(gc)}

    for f in ("R17-H143_evalset.parquet", "R20-H177_eval_B.parquet",
              "R20-H177_eval_C.parquet", "R20-H175b_qlane_eval.parquet"):
        p = EXP / f
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        out[f] = {"evidence": d["chunk"].to_list(), "claims": d["claim"].to_list()}
    log("other surfaces: " + ", ".join(f"{k} ({len(v['evidence'])} ev / "
                                       f"{len(v['claims'])} cl)" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------- #
# C1 - label commensurability (C-A1 structural first, C-A2 restated distribution)
# --------------------------------------------------------------------------- #
def clause_c1(ev):
    claims = ev["claim"].to_list()
    chunks = ev["chunk"].to_list()
    y = ev["label"].to_numpy()

    # C-A1 / C-A2 test 1: STRUCTURAL. Does any negative leg's (claim, evidence)
    # equal a positive leg's? Identity there means no function of (claim, evidence)
    # separates the legs, so the label cannot encode grounding.
    by_pair = collections.defaultdict(set)
    for c, k, v in zip(claims, chunks, y, strict=True):
        by_pair[(c, k)].add(int(v))
    collide = sum(1 for s in by_pair.values() if s == {0, 1})
    rows_in_collision = sum(1 for c, k in zip(claims, chunks, strict=True)
                            if by_pair[(c, k)] == {0, 1})
    by_ev = collections.defaultdict(set)
    for k, v in zip(chunks, y, strict=True):
        by_ev[k].add(int(v))
    ev_both = sum(1 for s in by_ev.values() if s == {0, 1})

    cache = {}

    def cont(claim, evid):
        s = cache.get(evid)
        if s is None:
            s = set(LC.tokens(evid))
            cache[evid] = s
        ct = set(LC.tokens(claim))
        return len(ct & s) / len(ct) if ct else 0.0

    con_full = np.array([cont(c, k) for c, k in zip(claims, chunks, strict=True)])
    cache.clear()
    con_tr = np.array([cont(c, k[:CHUNK_MAX]) for c, k in zip(claims, chunks, strict=True)])
    cache.clear()
    con_win = np.array([max(cont(c, w) for w in ARM.windows(k))
                        for c, k in zip(claims, chunks, strict=True)])
    cache.clear()

    def leg(v):
        return {"n": int(v.size), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 4),
                "p10": round(float(np.percentile(v, 10)), 4),
                "p90": round(float(np.percentile(v, 90)), 4),
                "frac_ge_0.90": round(float((v >= 0.90).mean()), 6),
                "frac_eq_1.00": round(float((v >= 0.999999).mean()), 6)}

    legs = {}
    for name, v in (("untruncated_full_evidence", con_full),
                    ("truncated_1500", con_tr),
                    ("max_over_windows_1500_750", con_win)):
        pos, neg = v[y == 1], v[y == 0]
        legs[name] = {
            "positive_leg": leg(pos),
            "negative_leg": leg(neg),
            "mean_gap_pos_minus_neg": round(float(pos.mean() - neg.mean()), 4),
            "containment_auroc_vs_label": round(LC.auroc(y, v), 4),
            "CA2_test2_strict_separation": {
                "neg_rate_ge_0.90": round(float((neg >= 0.90).mean()), 6),
                "pos_rate_ge_0.90": round(float((pos >= 0.90).mean()), 6),
                "neg_strictly_below_pos": bool((neg >= 0.90).mean() < (pos >= 0.90).mean()),
            },
        }
    primary = legs["untruncated_full_evidence"]

    ctok = np.array([len(LC.tokens(c)) for c in claims], dtype=float)
    etok = np.array([len(LC.tokens(c)) for c in chunks], dtype=float)
    passes = collide == 0 and primary["CA2_test2_strict_separation"]["neg_strictly_below_pos"]
    return {
        "head_declared": "grounding scalar (`task_head`) - the single support logit the "
                         "cascade serves; this surface carries no second head and no "
                         "question channel",
        "label_expression": "(hallucination_labels_processed.evident_conflict == 0) AND "
                            "(hallucination_labels_processed.baseless_info == 0) - "
                            "byte-identical to the expression `public_train` applies to "
                            "the ragtruth_en TRAIN member, so the eval scores the same "
                            "predicate the mix taught",
        "label_predicate_measured": (
            "support, at RESPONSE granularity. Label 1 iff the response carries ZERO "
            "human-annotated hallucination spans of any type; cross-checked on this "
            "split - 0 of 1,757 label-1 rows carry a span and 0 of 943 label-0 rows "
            "carry none. The predicate matches the grounding head; its UNIT does not "
            "match the serving unit, which is a single claim"),
        "label_unit_vs_serving_unit": (
            "RECORDED, not a clause verdict: the label is a property of a whole "
            "response (mean 775.9 chars, 128.9 tokens), while `ground()` scores one "
            "claim. This surface tests the coarser predicate the mix trained on, not "
            "the finer one the system serves"),
        "CA1_CA2_test1_structural": {
            "test": "does any negative leg's (claim, evidence) pair equal a positive "
                    "leg's? Identity there means no function of (claim, evidence) can "
                    "separate the legs",
            "distinct_claim_evidence_pairs": len(by_pair),
            "pairs_carrying_both_labels": collide,
            "rows_in_a_colliding_pair": rows_in_collision,
            "fires": collide > 0,
            "live_control_reference": "the withdrawn poisoned R20-H175b_qlane fires on "
                                      "8,986 of 8,986 pairs (100% of rows)",
        },
        "evidence_key_shared_across_labels": {
            "distinct_evidence_strings": len(by_ev),
            "evidence_strings_carrying_both_labels": ev_both,
            "reading": "expected and legal - six responder models answer each context "
                       "and differ in groundedness. C-A1's structural test keys on the "
                       "(claim, evidence) PAIR, not on evidence alone",
        },
        "CA2_test3_absolute_level": {
            "negative_leg_frac_ge_0.90": primary["negative_leg"]["frac_ge_0.90"],
            "negative_leg_frac_eq_1.00": primary["negative_leg"]["frac_eq_1.00"],
            "finding": "recorded per C-A2 test 3 - a negative leg attested at a high "
                       "absolute rate is a finding even when test 2 clears",
        },
        "instrument": "R20-H174_lane_common.containment - fraction of the claim's "
                      "content tokens present in the evidence",
        "primary_presentation": "untruncated_full_evidence (the R18-H150 / R20-H174 "
                                "flagship presentation, which this eval serves)",
        "presentations": legs,
        "confounds": {
            "claim_tokens": {"positive_mean": round(float(ctok[y == 1].mean()), 2),
                             "negative_mean": round(float(ctok[y == 0].mean()), 2),
                             "auroc_vs_label": round(LC.auroc(y, ctok), 4)},
            "evidence_tokens": {"positive_mean": round(float(etok[y == 1].mean()), 2),
                                "negative_mean": round(float(etok[y == 0].mean()), 2),
                                "auroc_vs_label": round(LC.auroc(y, etok), 4)},
        },
        "verdict": "PASS" if passes else "FAIL",
        "measured": {
            "structural_collisions": collide,
            "neg_rate_ge_0.90": primary["CA2_test2_strict_separation"]["neg_rate_ge_0.90"],
            "pos_rate_ge_0.90": primary["CA2_test2_strict_separation"]["pos_rate_ge_0.90"],
            "bar": "structural collisions == 0 AND neg rate strictly below pos rate "
                   "(C-A1 test 1, C-A2 test 2; the `within 0.10` band is struck)",
        },
    }


# --------------------------------------------------------------------------- #
# C2 - string channel: three forms, both directions, claims AND evidence
# --------------------------------------------------------------------------- #
FORMS = {"raw": lambda s: s,
         "truncated": lambda s: s[:CHUNK_MAX],
         "whitespace_collapsed_casefolded": wsnorm}


def form_sets(texts):
    texts = [t for t in texts if t and t.strip()]
    return {f: {fn(t) for t in texts} for f, fn in FORMS.items()}


def string_block(eval_sets, target_texts, unit):
    ts = form_sets(target_texts)
    out = {}
    n = 0
    for f in FORMS:
        a, b = eval_sets[f], ts[f]
        inter = a & b
        n += len(inter)
        out[f"{unit}__{f}"] = {
            "eval_units": len(a), "target_units": len(b),
            "eval_in_target": len(inter), "target_in_eval": len(inter),
            "fraction_of_eval": round(len(inter) / max(len(a), 1), 6),
            "fraction_of_target": round(len(inter) / max(len(b), 1), 6),
        }
    return out, n


# --------------------------------------------------------------------------- #
# C2 - document channel (n-gram containment, the read the string channel misses)
# --------------------------------------------------------------------------- #
def build_index(units, hasher):
    arrs = [G.ngram_hashes(t, GATE_N, hasher) for t in units]
    keep = [i for i, a in enumerate(arrs) if a.size]
    if not keep:
        raise SystemExit("INDEX ABORT: no eval unit long enough for the instrument")
    flat = np.concatenate([arrs[i] for i in keep])
    owner = np.concatenate([np.full(arrs[i].size, i, dtype=np.int64) for i in keep])
    order = np.argsort(flat, kind="stable")
    sizes = np.maximum(np.array([a.size for a in arrs], dtype=np.int64), 1)
    return {"flat": flat[order], "owner": owner[order], "sizes": sizes,
            "n": len(units), "scorable": len(keep),
            "cov": np.zeros(len(units)), "rev": np.zeros(len(units)),
            "jac": np.zeros(len(units)), "src": [None] * len(units),
            "rev_hits": collections.Counter(), "queries": collections.Counter()}


def update(ix, q, src):
    """One reference unit's n-gram set against the eval index."""
    ix["queries"][src] += 1
    if q.size == 0:
        return
    lo = np.searchsorted(ix["flat"], q, side="left")
    hi = np.searchsorted(ix["flat"], q, side="right")
    nz = np.nonzero(hi > lo)[0]
    if nz.size == 0:
        return
    ids = np.concatenate([ix["owner"][lo[i]:hi[i]] for i in nz])
    uids, inter = np.unique(ids, return_counts=True)
    sz = ix["sizes"][uids]
    cov = inter / sz                                   # eval unit covered by this ref
    rev = inter / q.size                               # ref unit covered by eval unit
    jac = inter / np.maximum(q.size + sz - inter, 1)
    if jac.max() >= GATE_JACCARD:
        ix["rev_hits"][src] += 1
    improved = np.nonzero(cov > ix["cov"][uids])[0]
    np.maximum.at(ix["cov"], uids, cov)
    np.maximum.at(ix["rev"], uids, rev)
    np.maximum.at(ix["jac"], uids, jac)
    for j in improved:
        ix["src"][uids[j]] = src


def summarise(ix):
    cov, jac = ix["cov"], ix["jac"]
    top = int(np.argmax(cov))
    return {
        "eval_units": ix["n"],
        "units_scorable_at_8_grams": ix["scorable"],
        "max_ngram_coverage_by_a_single_reference_unit": dist(cov),
        "coverage_ge_0.90": int((cov >= 0.90).sum()),
        "coverage_ge_0.70": int((cov >= 0.70).sum()),
        "coverage_ge_0.50": int((cov >= 0.50).sum()),
        "coverage_ge_0.30": int((cov >= 0.30).sum()),
        "top_10_units_by_coverage": [
            {"index": int(i), "coverage": round(float(cov[i]), 4),
             "jaccard": round(float(jac[i]), 4), "attributed_to": ix["src"][i]}
            for i in np.argsort(-cov)[:10]],
        "max_jaccard_against_a_single_reference_unit": dist(jac),
        "units_at_or_above_jaccard_bar": int((jac >= GATE_JACCARD).sum()),
        "fraction_at_or_above_jaccard_bar": round(float((jac >= GATE_JACCARD).mean()), 6),
        "worst_unit": {"index": top, "coverage": round(float(cov[top]), 4),
                       "jaccard": round(float(jac[top]), 4),
                       "attributed_to": ix["src"][top]},
        "attribution_of_units_above_0.30_coverage": dict(collections.Counter(
            s for s, c in zip(ix["src"], cov) if c >= 0.30 and s)),
        "reverse_direction_per_reference": {
            k: {"query_units": int(ix["queries"][k]),
                "units_reaching_jaccard_bar": int(ix["rev_hits"][k]),
                "fraction": round(ix["rev_hits"][k] / max(ix["queries"][k], 1), 6)}
            for k in sorted(ix["queries"])},
        "reverse_direction_worst_fraction": round(max(
            (ix["rev_hits"][k] / max(ix["queries"][k], 1) for k in ix["queries"]),
            default=0.0), 6),
    }


# --------------------------------------------------------------------------- #
# C3 - split semantics
# --------------------------------------------------------------------------- #
def clause_c3(tr, te):
    def sets(df, col):
        return {t for t in df[col].to_list() if t is not None}

    axes = {}
    for col in ("id", "context", "query", "output"):
        a, b = sets(tr, col), sets(te, col)
        axes[col] = {"train_distinct": len(a), "test_distinct": len(b),
                     "shared": len(a & b),
                     "eval_share_seen_in_train": round(len(a & b) / max(len(b), 1), 6)}
    a = {wsnorm(t) for t in tr["context"].to_list()}
    b = {wsnorm(t) for t in te["context"].to_list()}
    axes["context_whitespace_collapsed_casefolded"] = {
        "train_distinct": len(a), "test_distinct": len(b), "shared": len(a & b),
        "eval_share_seen_in_train": round(len(a & b) / max(len(b), 1), 6)}

    ev_tr = sorted({c for c in tr["context"].to_list() if c.strip()})
    ev_te = sorted({c for c in te["context"].to_list() if c.strip()})
    res = G.run_gate(ev_te, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                     label="ragtruth_test_contexts",
                     arena_texts={"ragtruth_train_contexts": ev_tr})
    d = res["candidate_vs_arena"]
    log(f"  C3 near-duplicate gate: {d['units_with_hit']}/{d['n_units']} "
        f"({d['fraction']}), best Jaccard max {d.get('best_jaccard', {}).get('max')}")

    zt = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    trans = {}
    tr_lab = ((tr["hallucination_labels_processed"].struct.field("evident_conflict") == 0)
              & (tr["hallucination_labels_processed"].struct.field("baseless_info") == 0)
              ).cast(pl.Int8).to_numpy()
    tr_task = tr["task_type"].to_list()
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        nm = next(x for x in zt.namelist()
                  if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet"))
        d2 = pl.read_parquet(io.BytesIO(zt.read(nm)))
        lab = (d2["labels"].list.len() == 0).cast(pl.Int8).to_numpy()
        same_len = d2.height == tr.height
        trans[lg] = {
            "member_file_loaded_by_public_train": nm,
            "rows": int(d2.height),
            "rows_equal_english_train": bool(same_len),
            "task_type_sequence_identical_to_english_train": bool(
                same_len and "task_type" in d2.columns
                and d2["task_type"].to_list() == tr_task),
            "label_agreement_with_english_train": (
                round(float((lab == tr_lab).mean()), 6) if same_len else None),
            "test_split_present_in_archive_but_not_loaded": bool(
                any(f"ragtruth-{lg}-" in x and x.endswith("__test.parquet")
                    for x in zt.namelist())),
        }
    log(f"  C3 translations rows {[trans[l]['rows'] for l in trans]} label-agreement "
        f"{[trans[l]['label_agreement_with_english_train'] for l in trans]}")

    ctx_tr, ctx_te = sets(tr, "context"), sets(te, "context")
    ok = (not (ctx_tr & ctx_te)
          and axes["context_whitespace_collapsed_casefolded"]["shared"] == 0
          and d["fraction"] == 0.0
          and all(v["rows_equal_english_train"] for v in trans.values()))
    return {
        "declared_split": "the archive's own `__train` / `__test` parquet split "
                          "(wandb/RAGTruth-processed); this eval is the TEST side, the "
                          "mix carries the TRAIN side",
        "measured_axis": ("document/context - every distinct context string lands wholly "
                          "in one split" if not (ctx_tr & ctx_te)
                          else "NOT document-disjoint: contexts appear in both splits"),
        "axes": axes,
        "query_field_note": (
            f"{axes['query']['shared']} of {axes['query']['test_distinct']} distinct "
            "test `query` strings recur in train. They are the SHARED TASK INSTRUCTIONS "
            "('Summarize the following news within N words', the Data2txt instruction "
            "block); `public_train` never consumes the query column and this eval does "
            "not put it in `claim` or `chunk`, so no consumed field carries the "
            "recurrence"),
        "near_duplicate_test": {
            "why": "C3 says an official split is not evidence of disjointness - it is "
                   "tested. Exact string equality is the weaker test; this is the "
                   "R14-H136 near-duplicate instrument on the same two sides",
            "candidates": res["candidate"]["n_units"],
            "train_index_units": len(ev_tr),
            "units_with_hit": d["units_with_hit"],
            "detected_fraction": d["fraction"],
            "best_jaccard": d.get("best_jaccard"),
        },
        "translated_members_are_train_only": {
            "why": "the obvious leak route is a translated member carrying the TEST "
                   "split back into training. `R10-H108_lane.public_train` selects "
                   "`endswith('__train.parquet')` per language; this verifies the row "
                   "alignment that makes the selection meaningful",
            "per_language": trans,
        },
        "verdict": "PASS" if ok else "FAIL",
        "measured": {
            "shared_context_strings_train_vs_eval": len(ctx_tr & ctx_te),
            "shared_context_strings_wsnorm": axes[
                "context_whitespace_collapsed_casefolded"]["shared"],
            "shared_ids": axes["id"]["shared"],
            "near_duplicate_fraction": d["fraction"],
            "bar": 0,
        },
    }


# --------------------------------------------------------------------------- #
# C4 - contamination census with a LIVE positive control
# --------------------------------------------------------------------------- #
def coverage(texts):
    short = sum(1 for t in texts if len(G.normalize(t).split()) < GATE_N)
    return {"units": len(texts), f"units_below_{GATE_N}_grams": short,
            "fraction_below": round(short / max(len(texts), 1), 6)}


def clause_c4(ev, ragtruth_en_ref, doc, reference_names):
    contexts = sorted({c for c in ev["chunk"].to_list() if c.strip()})
    claims = sorted({c for c in ev["claim"].to_list() if c.strip()})

    fwd_ev = doc["eval_contexts"]["fraction_at_or_above_jaccard_bar"]
    fwd_cl = doc["eval_claims"]["fraction_at_or_above_jaccard_bar"]
    rev_ev = doc["eval_contexts"]["reverse_direction_worst_fraction"]
    rev_cl = doc["eval_claims"]["reverse_direction_worst_fraction"]
    worst = max(fwd_ev, fwd_cl, rev_ev, rev_cl)

    # INSTRUMENT EQUIVALENCE - the same gate recomputed by the banked run_gate on
    # one reference side, so the streaming census is verified, not trusted
    eq = G.run_gate(contexts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                    label="eval_contexts",
                    arena_texts={"ragtruth_en_evidence": ragtruth_en_ref})
    log(f"  C4 equivalence vs ragtruth_en evidence: run_gate forward "
        f"{eq['candidate_vs_arena']['fraction']}, reverse "
        f"{eq['arena_vs_candidate']['fraction']}")

    # SPIKE control - inject reference units into the candidate side
    spike = G.spike_control(contexts, {"ragtruth_en_evidence": ragtruth_en_ref},
                            n=GATE_N, jaccard=GATE_JACCARD, k=10, label="eval_spike")
    log(f"  C4 spike control: {spike}")

    # LIVE positive control - text near-duplicate BY CONSTRUCTION
    rng = np.random.default_rng(0)
    sample = [contexts[i] for i in rng.choice(len(contexts),
                                              size=min(200, len(contexts)), replace=False)]
    perturbed, kept_frac = [], []
    for t in sample:
        parts = re.split(r"(?<=[.!?])\s+", t)
        kept = [p for i, p in enumerate(parts) if i % 10 != 0] or parts
        p = WS.sub(" ", " ".join(kept))
        perturbed.append(p)
        kept_frac.append(len(G.normalize(p).split()) / max(len(G.normalize(t).split()), 1))
    live = G.run_gate(perturbed, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                      label="eval_live_control",
                      arena_texts={"eval_own_contexts": contexts})
    log(f"  C4 live positive control: {live['candidate_vs_arena']['fraction']} detected")

    hasher = G._TokenHasher()
    arrs = [G.ngram_hashes(t, GATE_N, hasher) for t in contexts]
    flat = np.concatenate([a for a in arrs if a.size])
    owner = np.concatenate([np.full(a.size, i, dtype=np.int64)
                            for i, a in enumerate(arrs) if a.size])
    order = np.argsort(flat, kind="stable")
    flat, owner = flat[order], owner[order]
    sizes = np.array([a.size for a in arrs], dtype=np.int64)
    misses = []
    for i, p in enumerate(perturbed):
        j, _u = G._max_jaccard(G.ngram_hashes(p, GATE_N, hasher), flat, owner, sizes)
        if j < GATE_JACCARD:
            misses.append({"candidate": i, "best_jaccard": round(j, 4),
                           "token_fraction_kept": round(kept_frac[i], 4),
                           "source_tokens": int(len(G.normalize(sample[i]).split()))})
    unexplained = [m for m in misses if m["best_jaccard"] >= GATE_JACCARD]
    log(f"  C4 live control misses {len(misses)}, unexplained {len(unexplained)}")

    short_ev = [t for t in contexts if len(G.normalize(t).split()) < GATE_N]
    short_cl = [t for t in claims if len(G.normalize(t).split()) < GATE_N]
    backstop = {
        "short_evidence_units": len(short_ev),
        "short_claim_units": len(short_cl),
        "status": ("no eval unit falls below the instrument's 8-gram floor, so the "
                   "exact-match backstop has nothing to cover"
                   if not short_ev and not short_cl else "exact matching applied"),
    }
    if short_ev or short_cl:
        ref_norm = {G.normalize(t) for t in ragtruth_en_ref if len(t) < 600}
        backstop["short_evidence_exact_hits_vs_ragtruth_en"] = sum(
            1 for t in short_ev if G.normalize(t) in ref_norm)
        backstop["short_claim_exact_hits_vs_ragtruth_en"] = sum(
            1 for t in short_cl if G.normalize(t) in ref_norm)

    ok = (worst < GATE_KILL and spike["passes"] and spike["baseline_hits"] == 0
          and not unexplained and live["candidate_vs_arena"]["fraction"] > 0.0)
    return {
        "instrument": (f"provenance_gate.py, R14-H136 ruling-2 form: {GATE_N}-gram, "
                       f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL > "
                       f"{GATE_KILL:.0%}; thresholds read from R19_supply_gates.py"),
        "unit_definition": "deduplicated untruncated eval contexts; deduplicated eval "
                           "response claims",
        "reference_sides": reference_names,
        "eval_evidence_into_reference": {
            "n_units": len(contexts),
            "fraction_at_or_above_jaccard_bar": fwd_ev,
            "attribution_above_0.30_coverage":
                doc["eval_contexts"]["attribution_of_units_above_0.30_coverage"],
        },
        "eval_claims_into_reference": {
            "n_units": len(claims),
            "fraction_at_or_above_jaccard_bar": fwd_cl,
            "attribution_above_0.30_coverage":
                doc["eval_claims"]["attribution_of_units_above_0.30_coverage"],
        },
        "reference_into_eval": {
            "evidence_direction_worst_fraction": rev_ev,
            "claim_direction_worst_fraction": rev_cl,
            "per_reference_evidence_index":
                doc["eval_contexts"]["reverse_direction_per_reference"],
            "per_reference_claim_index":
                doc["eval_claims"]["reverse_direction_per_reference"],
        },
        "instrument_equivalence_check": {
            "why": "the mix-side census runs as one streaming pass for memory; this "
                   "recomputes the same gate with the banked run_gate on one reference "
                   "side, so the pass is verified rather than trusted",
            "reference": "ragtruth_en evidence - the mix member drawn from this very "
                         "corpus, the strictest available comparison",
            "run_gate_forward_fraction_eval_into_reference":
                eq["candidate_vs_arena"]["fraction"],
            "streaming_forward_fraction_all_references": fwd_ev,
            "run_gate_reverse_fraction_reference_into_eval":
                eq["arena_vs_candidate"]["fraction"],
            "streaming_reverse_fraction_same_reference":
                doc["eval_contexts"]["reverse_direction_per_reference"].get(
                    "mix:ragtruth_en:evidence", {}).get("fraction"),
            "best_jaccard_eval_into_ragtruth_en": eq["candidate_vs_arena"].get(
                "best_jaccard"),
        },
        "spike_control": spike,
        "live_positive_control_constructed_near_duplicates": {
            "construction": "200 sampled eval contexts, whitespace re-wrapped and every "
                            "10th sentence deleted, gated against the eval's own context "
                            "index - near-duplicate by construction",
            "candidates": live["candidate"]["n_units"],
            "detected_fraction": live["candidate_vs_arena"]["fraction"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "misses": misses,
            "unexplained_misses": unexplained,
            "miss_diagnosis": "a miss is EXPLAINED when the perturbed candidate's own "
                              "best Jaccard against the index is below the gate "
                              "threshold - the construction destroyed the near-duplicate, "
                              "so silence is the gate behaving",
            "fires": live["candidate_vs_arena"]["fraction"] > 0.0 and not unexplained,
        },
        "coverage": {"evidence": coverage(contexts), "claims": coverage(claims),
                     "exact_match_backstop": backstop},
        "verdict": "PASS" if ok else "FAIL",
        "measured": {"max_fraction_any_direction": round(worst, 6),
                     "kill_bar": GATE_KILL,
                     "spike_detected": spike["detected_total"],
                     "spike_baseline_hits": spike["baseline_hits"],
                     "live_control_detected_fraction":
                         live["candidate_vs_arena"]["fraction"]},
    }


# --------------------------------------------------------------------------- #
# Part C - the claim-only channel, priced rather than hidden
# --------------------------------------------------------------------------- #
def claim_only_channel(ev, mix_claims, mix_y, groups):
    claims = ev["claim"].to_list()
    y = ev["label"].to_numpy().astype(int)
    doc = ev["doc_id"].to_list()

    char_auc, tok_auc = CO.surface_auroc(claims, y)
    tr, te = CO.split_stratified(y)
    a_strat = CO.split_auroc(claims, y, tr, te, tag="R21-H180/stratified")
    tr2, te2 = CO.split_doc_disjoint(doc)
    a_doc = CO.split_auroc(claims, y, tr2, te2, tag="R21-H180/doc-disjoint")

    rng = np.random.default_rng(20)
    ysh = y.copy()
    rng.shuffle(ysh)
    a_sh = CO.split_auroc(claims, ysh, tr2, te2, tag="R21-H180/label-shuffled")

    idx = groups["ragtruth_en"]
    probe_en = CO.fit_probe([mix_claims[i] for i in idx], mix_y[idx].astype(int),
                            tag="fit_on_ragtruth_en_train")
    a_transfer = float(CO.roc_auc_score(y, CO.apply_probe(probe_en, claims)))
    del probe_en

    vb = CO.two_sided(a_doc)
    log(f"  PART C: stratified {a_strat:.4f}  doc-disjoint {a_doc:.4f}  "
        f"strength {vb['leak_strength']:.4f}  shuffled {a_sh:.4f}  "
        f"transfer {a_transfer:.4f}")
    return {
        "why_this_is_measured": (
            "RAGTruth carries the campaign's worst measured claim-only channel - "
            "0.8046-0.8280 one-sided across its eight language blocks in the training "
            "mix (R20_claimonly_sweep.json). An eval drawn from the same corpus "
            "plausibly inherits it, so the property is priced here rather than left for "
            "a later reader to discover"),
        "instrument": (
            "R20_claimonly_sweep.fit_probe - TF-IDF char_wb(2,5) + word(1,2) of the "
            "CLAIM STRING ALONE (the evidence is never shown to the probe), hstacked, "
            "into LogisticRegression(solver=liblinear, C=4.0, tol=1e-7, max_iter=3000); "
            "reused, not reimplemented"),
        "reading_rule": "TWO-SIDED. leak_strength = |AUROC - 0.5|; an inverted AUROC is "
                        "signal with the sign flipped, not cleanliness. Bands: clean "
                        "< 0.05, mild < 0.15, leak < 0.30, severe >= 0.30",
        "rows": int(len(y)),
        "label_balance_positive": round(float(y.mean()), 4),
        "splits": {"stratified_70_30": CO.two_sided(a_strat),
                   "doc_disjoint_70_30": CO.two_sided(a_doc)},
        "verdict_bearing_split": "doc_disjoint_70_30",
        "verdict_bearing_reason": (
            "six responder models answer each of the 450 contexts, so under a stratified "
            "split five siblings of every test response sit in train and the probe can "
            "read context-level difficulty rather than claim shape. The context-disjoint "
            "split removes that route; the stratified number is a diagnostic"),
        "leak_strength": vb["leak_strength"],
        "equivalent_one_sided_auroc": vb["equivalent_one_sided_auroc"],
        "band": vb["band"],
        "direction": vb["direction"],
        "surface_channels": {
            "response_char_length_auroc": round(char_auc, 4),
            "response_char_length_reading": CO.two_sided(char_auc),
            "response_token_count_auroc": round(tok_auc, 4),
            "response_token_count_reading": CO.two_sided(tok_auc),
            "note": "raw AUROC of the label against the unfitted statistic; 0.50 is no "
                    "signal in either direction",
        },
        "negative_control_label_shuffled": {
            "construction": "labels permuted (seed 20), same instrument, same "
                            "context-disjoint split, same feature space",
            "reading": CO.two_sided(a_sh),
            "must_read": "leak_strength ~0 - a non-zero reading would mean the probe "
                         "manufactures signal and no observed number is evidence",
        },
        "executor_added_transfer_probe": {
            "status": "EXECUTOR-ADDED, reported separately, carries no bar",
            "construction": "probe fitted on ALL 15,090 ragtruth_en TRAIN claims in the "
                            "flagship mix, applied to this eval's 2,700 responses",
            "reading": CO.two_sided(a_transfer),
            "what_it_answers": "whether the claim-shape prior the mix already teaches "
                               "predicts THIS surface's labels without reading evidence",
        },
        "reference_training_mix_numbers": {
            "ragtruth_en_stratified": 0.8257, "ragtruth_en_doc_disjoint": 0.8046,
            "source": "R20_claimonly_sweep.json part1_per_member",
        },
        "what_a_strong_model_score_here_would_and_would_not_prove": (
            "WOULD: that the model separates grounded from ungrounded RAGTruth "
            "responses on 2,700 rows and 450 contexts it has never seen, in the shape "
            "it serves. WOULD NOT: that the separation comes from reading the evidence. "
            "A claim-only probe that never sees the evidence reaches the strength "
            "recorded above on this same surface, so any score at or below the "
            "equivalent one-sided figure is fully explainable without grounding. The "
            "registered wrong-evidence ablation is what separates the two; until it "
            "lands, a score here bounds capability from above, not grounding"),
        "note": NOTE,
    }


# --------------------------------------------------------------------------- #
# C6 - memorisation channel
# --------------------------------------------------------------------------- #
def clause_c6(ev, mix_claims, mix_chunks, mix_y):
    """C-A2 scoping: C6 binds features keyed on associations the TRAINING MIX
    supplies. The key this eval offers is its context (evidence) string."""
    y = ev["label"].to_numpy().astype(float)
    chunks = ev["chunk"].to_list()
    claims = ev["claim"].to_list()

    keys = {wsnorm(k) for k in chunks}
    assoc = collections.defaultdict(list)
    for c, k in zip(mix_claims, mix_chunks, strict=True):
        n = wsnorm(k)
        if n in keys:
            assoc[n].append(c)
    hits, overlaps = 0, []
    for c, k in zip(claims, chunks, strict=True):
        a = assoc.get(wsnorm(k))
        if not a:
            continue
        hits += 1
        overlaps.append(max(LC.containment(c, x) for x in a))

    by = collections.defaultdict(lambda: [0.0, 0])
    for k, v in zip(chunks, y, strict=True):
        by[k][0] += v
        by[k][1] += 1
    loo = np.array([(by[k][0] - v) / max(by[k][1] - 1, 1) for k, v in zip(chunks, y)])
    covered = np.array([by[k][1] > 1 for k in chunks])
    auc_loo = LC.auroc(y[covered], loo[covered])
    log(f"  C6 eval-facing key coverage {hits}/{ev.height}; within-eval LOO {auc_loo:.4f}")

    applicable = hits > 0
    return {
        "scope": "C-A2: C6 binds features keyed on associations the TRAINING MIX "
                 "supplies. Where the eval-facing test has zero key coverage, C6 is "
                 "NOT-APPLICABLE and no proxy is substituted",
        "key": "the eval context (evidence) string, whitespace-collapsed case-folded",
        "eval_facing_channel": {
            "eval_rows": int(ev.height),
            "rows_whose_key_the_mix_carries": hits,
            "coverage": round(hits / max(ev.height, 1), 6),
            "mean_overlap_where_covered": (round(float(np.mean(overlaps)), 4)
                                           if overlaps else None),
            "auroc_vs_label": None,
            "status": "undefined - zero key coverage" if not applicable else "measured",
        },
        "within_member_diagnostic": {
            "status": "DIAGNOSTIC under C-A2, not a C6 bar",
            "feature": "leave-one-out mean label of the row's own context key",
            "auroc_vs_label": round(auc_loo, 4),
            "coverage": round(float(covered.mean()), 4),
            "mechanism": "six responder models answer each context and contexts differ "
                         "in how often all six stay grounded, so context identity carries "
                         "a real within-eval label association. It is a corpus property, "
                         "matching ragtruth_en's banked 0.6509",
            "consequence_for_this_surface": "a model cannot exploit it - the mix carries "
                                            "none of these contexts - but per-context "
                                            "resampling would change the surface's "
                                            "difficulty",
        },
        "verdict": "NOT-APPLICABLE" if not applicable else (
            "PASS" if abs(auc_loo - 0.5) <= 0.05 else "FAIL"),
        "measured": {"eval_key_coverage": hits,
                     "within_member_loo_auroc": round(auc_loo, 4)},
    }


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def clause_c7(ev):
    y = ev["label"].to_numpy()
    return {
        "declared_unit": "rows = (claim, evidence) serving units. The surface has no "
                         "paired-contrast construction, so rows and (claim, evidence) "
                         "pairs are the same count; both are reported",
        "rows": int(ev.height),
        "pairs_claim_evidence": int(ev.select(["claim", "chunk"]).unique().height),
        "distinct_claims": int(ev["claim"].n_unique()),
        "distinct_evidence_contexts": int(ev["chunk"].n_unique()),
        "registered_figure": "2,700 rows / 450 contexts",
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "positive_rate": round(float(y.mean()), 4),
        "windowed_claim_window_pairs_at_1500_750": int(ev["n_windows"].sum()),
        "verdict": "PASS" if ev.height == 2700 and ev["chunk"].n_unique() == 450 else "FAIL",
        "measured": {"rows": int(ev.height), "registered_rows": 2700,
                     "contexts": int(ev["chunk"].n_unique()), "registered_contexts": 450,
                     "delta_rows": int(ev.height) - 2700},
    }


def clause_c8(ev):
    sidecar = DATA / "dataset-ragtruth.md"
    txt = sidecar.read_text() if sidecar.exists() else ""
    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    arc_date = max("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*i.date_time)
                   for i in z.infolist())

    def field(tag):
        for line in txt.splitlines():
            if line.startswith(f"- **{tag}**"):
                return line.split("-", 2)[-1].strip()
        return None

    ctx_counts = collections.Counter(ev["chunk"].to_list())
    return {
        "source": "HuggingFace `wandb/RAGTruth-processed` (processed mirror of "
                  "ParticleMedia/RAGTruth, ACL 2024)",
        "licence": field("Licence") or "MIT (sidecar dataset-ragtruth.md)",
        "licence_sidecar": str(sidecar.relative_to(ROOT)),
        "archive": "data/external/datasets/dataset-ragtruth.zip (gitignored; sidecar "
                   "tracked)",
        "retrieval": {
            "fetcher": "scripts/fetch_grounding_datasets.py (named in the sidecar)",
            "date_recorded_in_sidecar": None,
            "date_recoverable_from_archive": arc_date,
            "finding": "C8 requires the retrieval DATE. No sidecar in "
                       "data/external/datasets/ records one, so this is a systematic gap "
                       "in the repository rather than an artifact-specific one; the "
                       "archive member timestamps are the only evidence",
        },
        "selection_predicate": (
            "split `wandb__RAGTruth-processed__test.parquet`, WHOLE - no filter and no "
            "sampling. The mix loader's `context > 50 chars` predicate and the lineage "
            "read's extra `output > 20 chars` predicate were both evaluated and drop 0 "
            "of 2,700 rows. claim = `output`, evidence = `context` UNTRUNCATED (the read "
            f"windows it at 1,500/750; the R10-H108 presentation's {CHUNK_MAX}-char cut "
            "is NOT applied to the banked artifact). label = (evident_conflict == 0) AND "
            "(baseless_info == 0)"),
        "fields_consumed": ["output -> claim", "context -> chunk",
                            "hallucination_labels_processed -> label"],
        "fields_carried_for_derivation": [
            "hallucination_labels -> span_starts / span_ends / span_texts / span_types "
            "plus the four per-type counts, so a claim-level version of this eval can be "
            "derived without re-deriving the eval",
            "task_type, model, quality, temperature, query, ragtruth_id, doc_id, "
            "label_unit, response_chars, response_tokens, context_chars, n_windows",
        ],
        "internal_structure": {
            "rows": int(ev.height),
            "distinct_claims": int(ev["claim"].n_unique()),
            "duplicate_claim_rows": int(ev.height - ev["claim"].n_unique()),
            "distinct_evidence": len(ctx_counts),
            "rows_per_evidence": {"min": min(ctx_counts.values()),
                                  "max": max(ctx_counts.values())},
            "repeat_structure": "6 responder models x 450 source items; every context "
                                "string is repeated once per model",
            "task_types": dict(collections.Counter(ev["task_type"].to_list())),
            "responder_models": sorted(set(ev["model"].to_list())),
            "quality_flags": dict(collections.Counter(ev["quality"].to_list())),
        },
        "public_repo_check": {
            "checked": "the banked parquet's columns and content, and this report's own "
                       "text",
            "client_or_company_name_present": False,
        },
        "verdict": "FAIL",
        "measured": {
            "required_and_present": ["source", "licence", "selection predicate",
                                     "internal duplication", "public-repo check"],
            "required_and_missing": ["retrieval date"],
            "binding_constraint": "the licence sidecar records no retrieval date; the "
                                  f"archive's own timestamps read {arc_date}",
            "licence_note": "the sidecar's MIT tag is transcribed, not re-verified "
                            "against the source in this pass (no network use)",
        },
    }


# --------------------------------------------------------------------------- #
def main():
    if not EVAL.exists():
        raise SystemExit(f"ABORT: {EVAL} absent - run R21-H180_eval_build.py first")
    ev = pl.read_parquet(EVAL)
    log(f"eval loaded: {ev.height} rows / {ev['chunk'].n_unique()} contexts / "
        f"{ev['claim'].n_unique()} distinct claims")

    report = {
        "artifact": MEMBER,
        "class": "evaluation surface - held-out source-corpus split (RAGTruth English "
                 "test), the campaign's second of three evaluation surfaces",
        "parquet": str(EVAL.relative_to(ROOT)),
        "contract": "docs/experiments/dataset-contract.md, C1-C8 including amendments "
                    "C-A1 and C-A2",
        "serving_shape": (
            "flat long form, matching R20-H177_eval_B.parquet: one row per serving unit "
            "with claim / chunk / label / pair_id / doc_id / source. The banked read "
            "machinery (R20_baseline_legs.flatten + windows) takes it unmodified - the "
            "evidence is UNTRUNCATED and its 1,500/750 window bag IS the claim's "
            "evidence chunk set. The gold_full shape (claim + explicit chunk list, "
            "grouped on a key) is the same long form with a grouping key and is reached "
            "from this parquet as [[chunk]] per row, which the arena's decomposed-min "
            "reader expands into the identical window bag"),
        "cpu_only": True,
        "no_model_scored": True,
        "note": NOTE,
        "clauses": {},
    }
    checkpoint(report)

    log("C7 ...")
    report["clauses"]["C7"] = clause_c7(ev)
    log("C8 ...")
    report["clauses"]["C8"] = clause_c8(ev)
    checkpoint(report)

    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    tr = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))
    te = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__test.parquet")))))

    log("C1 ...")
    report["clauses"]["C1"] = clause_c1(ev)
    checkpoint(report)

    log("C3 ...")
    report["clauses"]["C3"] = clause_c3(tr, te)
    checkpoint(report)
    del tr, te

    log("assembling the flagship training mix ...")
    mix_claims, mix_chunks, mix_y, mix_tags = flagship_mix()
    groups = group_index(mix_tags)
    del mix_tags

    log("PART C - claim-only channel ...")
    report["claim_only_channel"] = claim_only_channel(ev, mix_claims, mix_y, groups)
    checkpoint(report)

    log("C6 ...")
    report["clauses"]["C6"] = clause_c6(ev, mix_claims, mix_chunks, mix_y)
    checkpoint(report)

    log("loading the other evaluation surfaces ...")
    surfaces = other_surfaces()

    # ---- C2 string channel: per reference side, sets built and freed ------- #
    log("C2 string channel ...")
    eval_ev_sets = form_sets(ev["chunk"].to_list())
    eval_cl_sets = form_sets(ev["claim"].to_list())
    per_target, total = {}, 0
    for g, idx in groups.items():
        blk = {}
        b1, n1 = string_block(eval_ev_sets, [mix_chunks[i] for i in idx], "evidence")
        b2, n2 = string_block(eval_cl_sets, [mix_claims[i] for i in idx], "claims")
        blk.update(b1)
        blk.update(b2)
        per_target[f"training_mix:{g}"] = blk
        total += n1 + n2
        log(f"  C2 string training_mix:{g}: max overlap "
            f"{max(v['eval_in_target'] for v in blk.values())}")
    for sname, s in surfaces.items():
        blk = {}
        b1, n1 = string_block(eval_ev_sets, s["evidence"], "evidence")
        b2, n2 = string_block(eval_cl_sets, s["claims"], "claims")
        blk.update(b1)
        blk.update(b2)
        per_target[f"eval_surface:{sname}"] = blk
        total += n1 + n2
        log(f"  C2 string eval_surface:{sname}: max overlap "
            f"{max(v['eval_in_target'] for v in blk.values())}")
    del eval_ev_sets, eval_cl_sets
    report["clauses"]["C2"] = {
        "forms": ["raw", f"truncated_{CHUNK_MAX}", "whitespace_collapsed_casefolded"],
        "directions": "both (set intersection is symmetric; both counts reported)",
        "units": "claims AND evidence",
        "string_channel": {"per_target": per_target, "total_overlapping_units": total},
    }
    checkpoint(report)

    # ---- C2 document channel ---------------------------------------------- #
    log("C2 document channel ...")
    hasher = G._TokenHasher()
    contexts = sorted({c for c in ev["chunk"].to_list() if c.strip()})
    subs = sorted({p.strip() for c in contexts for p in c.split("\n\n") if p.strip()})
    eclaims = sorted({c for c in ev["claim"].to_list() if c.strip()})
    idxs = {"eval_contexts": build_index(contexts, hasher),
            "eval_context_subpassages": build_index(subs, hasher),
            "eval_claims": build_index(eclaims, hasher)}
    log(f"  indices: contexts {len(contexts)}, subpassages {len(subs)}, "
        f"claims {len(eclaims)}")

    def stream():
        for g, idx in groups.items():
            yield (f"mix:{g}:evidence", sorted({mix_chunks[i] for i in idx
                                                if mix_chunks[i].strip()}))
            yield (f"mix:{g}:claims", sorted({mix_claims[i] for i in idx
                                              if mix_claims[i].strip()}))
        for sname, s in surfaces.items():
            yield (f"surface:{sname}:evidence",
                   sorted({t for t in s["evidence"] if t and t.strip()}))
            yield (f"surface:{sname}:claims",
                   sorted({t for t in s["claims"] if t and t.strip()}))

    ref_names = []
    for sname, texts in stream():
        t0 = time.time()
        for t in texts:
            q = G.ngram_hashes(t, GATE_N, hasher)
            for ix in idxs.values():
                update(ix, q, sname)
        ref_names.append(sname)
        log(f"  doc-channel {sname}: {len(texts)} units, "
            f"ctx-cov {idxs['eval_contexts']['cov'].max():.4f} "
            f"sub-cov {idxs['eval_context_subpassages']['cov'].max():.4f} "
            f"cl-cov {idxs['eval_claims']['cov'].max():.4f} "
            f"({time.time() - t0:.0f}s)")

    doc = {k: summarise(v) for k, v in idxs.items()}
    worst_doc = max(doc[u]["max_ngram_coverage_by_a_single_reference_unit"]["max"]
                    for u in doc)
    report["clauses"]["C2"]["document_channel"] = {
        "why": "three separate reads today came out an order of magnitude worse on "
               "document identity than on string equality (R20-H177_eval_B: 4.5% by "
               "string, 65% by document). This is the document read for this surface",
        "instrument": f"{GATE_N}-gram CONTAINMENT - the fraction of an eval unit's "
                      "n-grams covered by ONE reference unit. Coverage 1.0 means the "
                      "eval unit is a sub-document of that reference unit, which "
                      "neither exact nor whitespace-normalised string matching sees. "
                      "Max Jaccard against a single reference unit is reported beside "
                      "it - that is the C4 gate's own quantity",
        "subpassage_note": "a RAGTruth QA context is several retrieved passages joined "
                           "on blank lines, so contexts are decomposed on their "
                           "blank-line junctions and the sub-passages re-tested "
                           "independently",
        "reference_sides": ref_names,
        **doc,
    }
    report["clauses"]["C2"]["verdict"] = "PASS" if (
        total == 0 and worst_doc < DOC_COVERAGE_BAR) else "FAIL"
    report["clauses"]["C2"]["measured"] = {
        "string_channel_total_overlapping_units_all_forms_all_targets": total,
        "string_channel_bar": 0,
        "document_channel_max_coverage_any_unit": worst_doc,
        "document_channel_units_at_coverage_ge_0.90": {
            u: doc[u]["coverage_ge_0.90"] for u in doc},
        "document_channel_units_at_coverage_ge_0.50": {
            u: doc[u]["coverage_ge_0.50"] for u in doc},
        "document_channel_bar_applied_by_executor": DOC_COVERAGE_BAR,
        "bar_note": "C2 writes no numeric bar for a document channel - it says a member "
                    "passes only when all string forms read zero. The executor applies "
                    "coverage >= 0.90 as the document-identity threshold and flags that "
                    "choice; every distribution is reported so any threshold can be "
                    "applied to it",
    }
    checkpoint(report)
    del idxs

    log("C4 ...")
    ragtruth_en_ref = sorted({mix_chunks[i] for i in groups["ragtruth_en"]
                              if mix_chunks[i].strip()})
    report["clauses"]["C4"] = clause_c4(
        ev, ragtruth_en_ref, report["clauses"]["C2"]["document_channel"], ref_names)
    checkpoint(report)

    log("C5 ...")
    report["clauses"]["C5"] = {
        "verdict": "NOT-APPLICABLE",
        "why": "C5 is scoped to 'every constructed lane and every paired-contrast eval'. "
               "This surface is neither: it is a source corpus's own held-out split of "
               "naturally occurring LLM responses with human span annotations. There is "
               "no construction, no minimal pair, no `neg_family`, no direction/element "
               "balance and no paired contrast to balance. Six responses share a "
               "context, which is a repeat structure, not a contrast pair",
        "executor_added_reported_separately": {
            "status": "EXECUTOR-ADDED, NOT part of the registered C5 conjunction, "
                      "carries no bar for this surface",
            "where": "the top-level `claim_only_channel` block of this report carries "
                     "the claim-only probe, both surface channels, the label-shuffled "
                     "negative control and the transfer probe in full",
            "claim_only_leak_strength": report["claim_only_channel"]["leak_strength"],
            "claim_only_equivalent_one_sided_auroc":
                report["claim_only_channel"]["equivalent_one_sided_auroc"],
            "c5_reference_bar_for_context": "C5's own bar for a CONSTRUCTED member is "
                                            "claim-only < 0.55; it is quoted here for "
                                            "scale, not applied",
        },
    }

    order = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
    report["clauses"] = {k: report["clauses"][k] for k in order if k in report["clauses"]}
    fails = [k for k, v in report["clauses"].items() if v["verdict"] == "FAIL"]
    report["conforming"] = not fails
    report["failed_clauses"] = fails
    report["verdicts"] = {k: v["verdict"] for k, v in report["clauses"].items()}
    checkpoint(report)
    log(f"report -> {OUT}")
    log(json.dumps(report["verdicts"], indent=2))


if __name__ == "__main__":
    main()
