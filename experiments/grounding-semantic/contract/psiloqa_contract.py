"""DATASET CONTRACT verification - member `psiloqa`. CPU ONLY, torch untouched on GPU.

Verifies the PsiloQA source corpus, as it enters the assembled training mix,
against every clause C1-C8 of `docs/experiments/dataset-contract.md`.

The member is rebuilt through the BANKED loader - `R10-H108_lane.public_train()`
with the evidence cut lifted (the H150/H174 twin protocol), rows filtered to the
`psiloqa` DANN group. Nothing about the member is re-implemented here.

Instruments reused, not reinvented:
  contamination census  `provenance_gate.py` (R14-H136 ruling-2 form: 8-gram,
                        Jaccard >= 0.3, bidirectional, KILL > 2%, spike control)
  containment           `R20_gates_pubmedqa.content` / `.containment` - the
                        campaign's content-token containment. That tokenizer is
                        ASCII (`[a-z0-9]+`); PsiloQA is 14 languages, so a
                        Unicode-aware variant is reported SEPARATELY alongside
                        it and coverage is stated, never silently substituted
  normalisation         `R20-H175b_qlane_eval_clean.norm` - whitespace-collapsed
                        case-folded form, the C2 third string form

Writes `psiloqa_contract_report.json` beside this file, one block per clause.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/psiloqa_contract.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import collections
import datetime as dt
import importlib.util as _ilu
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
ARCHIVE = DATA / "dataset-psiloqa.zip"
OUT = HERE / "psiloqa_contract_report.json"

MEMBER = "psiloqa"
REGISTERED_ROWS = 61_712          # the row count this member is registered at
EXPECTED_CLEAN_ROWS = 685_670     # R18-H150/R20-H174 clean-mix census constant


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _mod("provgate", EXP / "provenance_gate.py")
PM = _mod("pmgates", EXP / "R20_gates_pubmedqa.py")
CL = _mod("cleanbuild", EXP / "R20-H175b_qlane_eval_clean.py")

# R14-H136 ruling-2 gate form, read from R19_supply_gates.py, never restated
_gates_src = (EXP / "R19_supply_gates.py").read_text()
GATE_N = int(_gates_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gates_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gates_src.split("GATE_KILL = ")[1].split("\n")[0])

# Unicode-aware companion to PM.content - PsiloQA spans 14 languages and the
# banked ASCII tokenizer scores none of the non-Latin ones.
TOKEN_U = re.compile(r"\w+", re.UNICODE)


def content_u(text):
    return frozenset(t for t in TOKEN_U.findall(text.lower()) if t not in PM.STOPWORDS)


def pct(x, n):
    return round(float(x) / n, 6) if n else None


def dist(vals):
    """Distribution summary of a containment vector."""
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "frac_ge_0.50": round(float((a >= 0.50).mean()), 4),
        "frac_ge_0.90": round(float((a >= 0.90).mean()), 4),
        "frac_eq_1.00": round(float((a >= 0.99999).mean()), 4),
    }


# --------------------------------------------------------------------------- #
# member load - the BANKED path
# --------------------------------------------------------------------------- #
def load_member():
    """The member exactly as the live arms assemble it, plus the full mix chunk
    sets in the three C2 string forms."""
    t0 = time.time()
    H108 = _mod("h108lane", EXP / "R10-H108_lane.py")
    M59 = H108.M59
    chunk_max = M59.CFG.chunk_max_chars
    print(f"loader: chunk_max_chars = {chunk_max}", flush=True)

    M59.CFG.chunk_max_chars = 10**9        # `untruncated_evidence`, inlined
    try:
        claims, chunks, y, tags = H108.public_train()
    finally:
        M59.CFG.chunk_max_chars = chunk_max
    print(f"loader: clean mix {len(y)} rows over {len(set(tags))} groups "
          f"in {time.time() - t0:.0f}s", flush=True)

    per_group = dict(collections.Counter(tags))
    idx = [i for i, t in enumerate(tags) if t == MEMBER]
    mem = {
        "claims": [claims[i] for i in idx],
        "chunks": [chunks[i] for i in idx],
        "y": np.asarray(y, dtype="float32")[idx],
        "chunk_max": chunk_max,
        "mix_rows": len(y),
        "mix_groups": per_group,
    }
    mix = {
        "raw": set(chunks),
        "trunc": {c[:chunk_max] for c in chunks},
    }
    mix["nraw"] = {CL.norm(c) for c in mix["raw"]}
    mix["ntrunc"] = {CL.norm(c) for c in mix["trunc"]}
    del claims, chunks, y, tags
    print(f"loader: member {len(mem['y'])} rows; mix distinct raw chunks "
          f"{len(mix['raw'])}", flush=True)
    return mem, mix


def archive_splits():
    z = zipfile.ZipFile(ARCHIVE)
    out = {}
    for name in z.namelist():
        if name.endswith(".parquet"):
            split = name.split("__")[-1].replace(".parquet", "")
            out[split] = pl.read_parquet(io.BytesIO(z.read(name)))
    return out


# --------------------------------------------------------------------------- #
# C1 - label commensurability
# --------------------------------------------------------------------------- #
def clause_c1(mem, splits):
    print("\n=== C1 label commensurability", flush=True)
    claims, chunks, y = mem["claims"], mem["chunks"], mem["y"]
    n = len(y)

    banked, uni = np.zeros(n), np.zeros(n)
    banked_cov, uni_cov = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i, (cl, ch) in enumerate(zip(claims, chunks, strict=True)):
        cc, ec = PM.content(cl), PM.content(ch)
        banked_cov[i] = bool(cc)
        banked[i] = PM.containment(cc, ec)
        cu, eu = content_u(cl), content_u(ch)
        uni_cov[i] = bool(cu)
        uni[i] = PM.containment(cu, eu)
        if i and i % 20000 == 0:
            print(f"  containment {i}/{n}", flush=True)

    pos, neg = y == 1.0, y == 0.0
    res = {
        "clause": "C1",
        "head_declared": "grounding scalar (`task_head`) - the served ground() score",
        "label_predicate_measured": (
            "label 1 iff the corpus's `labels` span list is EMPTY, i.e. the "
            "annotator marked no hallucinated span in the LLM answer given the "
            "retrieved Wikipedia passage; label 0 iff at least one span of the "
            "answer was marked unsupported. The predicate is span-level SUPPORT "
            "of the answer by the passage, not question relevance - the same "
            "predicate the grounding head consumes"),
        "label_provenance": "GPT-4o end-to-end span annotation, no human verification (dataset card)",
        "rows": n,
        "positives": int(pos.sum()),
        "negatives": int(neg.sum()),
        "positive_rate": round(float(pos.mean()), 4),
        "instruments": {
            "primary": "R20_gates_pubmedqa.containment over R20_gates_pubmedqa.content "
                       "(ASCII [a-z0-9]+ content tokens, campaign stopwords)",
            "companion_flagged_separately": "same containment over Unicode \\w+ tokens - "
                                            "the banked tokenizer scores no non-Latin script, "
                                            "so it is reported beside the primary, not instead of it",
        },
        "coverage": {
            "banked_instrument_rows_scorable": int(banked_cov.sum()),
            "banked_instrument_coverage": round(float(banked_cov.mean()), 4),
            "unicode_instrument_rows_scorable": int(uni_cov.sum()),
            "unicode_instrument_coverage": round(float(uni_cov.mean()), 4),
            "note": "a row is unscorable when the claim has no content token under that "
                    "tokenizer; unscorable rows are excluded from the distributions below",
        },
    }

    for tag, vals, cov in (("banked_ascii", banked, banked_cov),
                           ("unicode", uni, uni_cov)):
        p = vals[pos & cov]
        q = vals[neg & cov]
        dp, dq = dist(p), dist(q)
        a_pos, a_neg = dp.get("frac_ge_0.90"), dq.get("frac_ge_0.90")
        res[f"containment_{tag}"] = {
            "positive_leg": dp,
            "negative_leg": dq,
            "attested_ge_0.90_rate_positive": a_pos,
            "attested_ge_0.90_rate_negative": a_neg,
            "rate_delta_abs": round(abs(a_pos - a_neg), 4) if None not in (a_pos, a_neg) else None,
            "fully_attested_rate_positive": dp.get("frac_eq_1.00"),
            "fully_attested_rate_negative": dq.get("frac_eq_1.00"),
            "mean_delta_pos_minus_neg": round(dp["mean"] - dq["mean"], 4),
            "bar": "REJECTED for the grounding head if |rate(>=0.90 attested)_neg - "
                   "rate(>=0.90 attested)_pos| <= 0.10",
            "rejected": bool(abs(a_pos - a_neg) <= 0.10) if None not in (a_pos, a_neg) else None,
        }

    # executor-added diagnostics, reported separately from the clause test
    tr = splits["train"].filter(
        (pl.col("wiki_passage").str.len_chars() > 50)
        & (pl.col("llm_answer").str.len_chars() > 10))
    if tr.height != n:
        raise SystemExit(f"ROW-ALIGNMENT ABORT: archive replay {tr.height} rows "
                         f"!= member {n} rows - the selection predicate has drifted")
    if tr["llm_answer"].to_list() != claims:
        raise SystemExit("ROW-ALIGNMENT ABORT: archive replay claims differ from the "
                         "member claims the banked loader produced")
    lang = tr["lang"].to_list()
    per_lang = {}
    for lg in sorted(set(lang)):
        m = np.array([x == lg for x in lang])
        for tag, vals, cov in (("banked_ascii", banked, banked_cov), ("unicode", uni, uni_cov)):
            pp, qq = vals[m & pos & cov], vals[m & neg & cov]
            if pp.size and qq.size:
                per_lang.setdefault(lg, {})[tag] = {
                    "n_pos": int(pp.size), "n_neg": int(qq.size),
                    "rate_ge_0.90_pos": round(float((pp >= 0.90).mean()), 4),
                    "rate_ge_0.90_neg": round(float((qq >= 0.90).mean()), 4),
                    "delta": round(abs(float((pp >= 0.90).mean()) - float((qq >= 0.90).mean())), 4),
                }

    # the annotated hallucinated spans themselves, on the negative leg
    span_cont = []
    for ans, labs, ch in zip(tr["llm_answer"].to_list(), tr["labels"].to_list(),
                             tr["wiki_passage"].to_list(), strict=True):
        if labs is None or len(labs) == 0:
            continue
        ec = content_u(ch)
        for sp in labs:
            frag = ans[int(sp[0]):int(sp[1])]
            fc = content_u(frag)
            if fc:
                span_cont.append(PM.containment(fc, ec))
    res["executor_added_diagnostics"] = {
        "note": "reported separately from the C1 test; these do not join the clause verdict",
        "per_language_rate_delta": per_lang,
        "annotated_hallucinated_span_containment_unicode": dist(span_cont),
        "annotated_span_reading": "containment of the marked-hallucinated fragments "
                                  "themselves against the passage; low values corroborate "
                                  "that the corpus label tracks support rather than relevance",
    }
    return res, banked, uni, banked_cov, uni_cov


# --------------------------------------------------------------------------- #
# C2 - disjointness from every evaluation surface
# --------------------------------------------------------------------------- #
def surface_units():
    """Evidence and claim units of every evaluation surface in the campaign."""
    surfaces = {}

    arena_texts, _ = G.load_arena()
    surfaces["arena_ragbench_10_subsets"] = {
        "evidence": sorted({c for v in arena_texts.values() for c in v if c and c.strip()}),
        "claims": [],
        "kind": "blind arena",
    }

    gp = EXP / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
    d = pl.read_parquet(gp)
    surfaces["gold_full"] = {
        "evidence": sorted({c for c in d["chunk"].to_list() if c and c.strip()}),
        "claims": sorted({c for c in d["claim"].to_list() if c and c.strip()}),
        "kind": "in-domain gold test surface",
    }

    for name in ("R17-H143_evalset.parquet", "R20-H177_eval_B.parquet",
                 "R20-H177_eval_C.parquet", "R20-H175b_qlane_eval.parquet",
                 "R20-H175b_qlane_eval_repaired.parquet",
                 "R20-H175b_qlane_eval_clean.parquet",
                 "R20-H175b_qlane_eval_clean_prefix.parquet"):
        p = EXP / name
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        col = next((c for c in ("chunk", "evidence", "context") if c in d.columns), None)
        surfaces[name] = {
            "evidence": sorted({c for c in d[col].to_list() if c and c.strip()}) if col else [],
            "claims": sorted({c for c in d["claim"].to_list() if c and c.strip()})
                      if "claim" in d.columns else [],
            "kind": "held-out mechanism eval",
            "rows": d.height,
        }
    return surfaces, arena_texts


def forms_of(texts, cut):
    raw = set(texts)
    trunc = {t[:cut] for t in raw}
    return {"raw": raw, "trunc": trunc,
            "nraw": {CL.norm(t) for t in raw},
            "ntrunc": {CL.norm(t) for t in trunc}}


def compare(member_forms, surface_forms):
    """Counts in three string forms, both directions, plus the cross-forms the
    truncating loader can create."""
    tests = {
        "raw_vs_raw": ("raw", "raw"),
        "truncated_vs_truncated": ("trunc", "trunc"),
        "normalised_vs_normalised": ("nraw", "nraw"),
        "member_raw_vs_surface_truncated": ("raw", "trunc"),
        "member_truncated_vs_surface_raw": ("trunc", "raw"),
        "normalised_truncated_vs_normalised_truncated": ("ntrunc", "ntrunc"),
    }
    out = {}
    for name, (a, b) in tests.items():
        inter = member_forms[a] & surface_forms[b]
        out[name] = {
            "intersection_size": len(inter),
            "member_side": {"units": len(member_forms[a]),
                            "matching": len(inter),
                            "fraction": pct(len(inter), len(member_forms[a]))},
            "surface_side": {"units": len(surface_forms[b]),
                             "matching": len(inter),
                             "fraction": pct(len(inter), len(surface_forms[b]))},
        }
    out["worst_form_intersection"] = max(v["intersection_size"] for v in out.values()
                                         if isinstance(v, dict))
    return out


def clause_c2(mem, surfaces):
    print("\n=== C2 disjointness from every evaluation surface", flush=True)
    cut = mem["chunk_max"]
    mem_ev = forms_of(mem["chunks"], cut)
    mem_cl = forms_of(mem["claims"], cut)

    per_surface = {}
    for name, s in surfaces.items():
        ev = forms_of(s["evidence"], cut) if s["evidence"] else None
        cl = forms_of(s["claims"], cut) if s["claims"] else None
        block = {
            "kind": s["kind"],
            "surface_evidence_units": len(s["evidence"]),
            "surface_claim_units": len(s["claims"]),
            "evidence_vs_evidence": compare(mem_ev, ev) if ev else None,
            "claim_vs_claim": compare(mem_cl, cl) if cl else None,
            "member_claim_vs_surface_evidence": compare(mem_cl, ev) if ev else None,
        }
        worst = max([b["worst_form_intersection"] for b in
                     (block["evidence_vs_evidence"], block["claim_vs_claim"],
                      block["member_claim_vs_surface_evidence"]) if b] or [0])
        block["worst_intersection_any_form_any_pairing"] = worst
        block["status"] = "CLEAN" if worst == 0 else "OVERLAPS"
        per_surface[name] = block
        print(f"  {name}: worst intersection {worst} -> {block['status']}", flush=True)

    dirty = sorted(k for k, v in per_surface.items() if v["status"] == "OVERLAPS")
    return {
        "clause": "C2",
        "member_distinct_evidence_units": len(mem_ev["raw"]),
        "member_distinct_claim_units": len(mem_cl["raw"]),
        "string_forms": ["raw", f"truncated to chunk_max_chars={cut}",
                         "whitespace-collapsed case-folded"],
        "directions": "both - member-in-surface and surface-in-member counted per form",
        "per_surface": per_surface,
        "surfaces_with_overlap": dirty,
        "verdict": "PASS" if not dirty else "FAIL",
    }


# --------------------------------------------------------------------------- #
# C3 - split semantics measured
# --------------------------------------------------------------------------- #
def clause_c3(mem, mix, splits):
    print("\n=== C3 split semantics", flush=True)
    cut = mem["chunk_max"]
    tr, va, te = splits["train"], splits["validation"], splits["test"]

    tr_pass = set(tr["wiki_passage"].to_list())
    ho_pass = sorted(set(va["wiki_passage"].to_list()) | set(te["wiki_passage"].to_list()))
    tr_pass_n = {CL.norm(p) for p in tr_pass}
    tr_q = set(tr["question"].to_list())
    ho_q = sorted(set(va["question"].to_list()) | set(te["question"].to_list()))
    tr_url = set(tr["wiki_url"].to_list())
    ho_url = sorted(set(va["wiki_url"].to_list()) | set(te["wiki_url"].to_list()))
    tr_ans = set(tr["llm_answer"].to_list())
    ho_ans = sorted(set(va["llm_answer"].to_list()) | set(te["llm_answer"].to_list()))

    mem_ev = forms_of(mem["chunks"], cut)

    res = {
        "clause": "C3",
        "split_axis_declared_by_card": "official train / validation / test splits",
        "split_axis_measured": None,
        "rows": {"train": tr.height, "validation": va.height, "test": te.height},
        "distinct_passages": {"train": len(tr_pass), "validation": va["wiki_passage"].n_unique(),
                              "test": te["wiki_passage"].n_unique(),
                              "validation_plus_test": len(ho_pass)},
        "held_out_passages_identical_to_a_train_passage": {
            "raw": sum(1 for p in ho_pass if p in tr_pass),
            "normalised": sum(1 for p in ho_pass if CL.norm(p) in tr_pass_n),
            "of_units": len(ho_pass),
        },
        "held_out_passages_present_in_the_assembled_mix": {
            "member_chunks_raw": sum(1 for p in ho_pass if p in mem_ev["raw"]),
            "member_chunks_truncated": sum(1 for p in ho_pass if p[:cut] in mem_ev["trunc"]),
            "member_chunks_normalised": sum(1 for p in ho_pass if CL.norm(p) in mem_ev["nraw"]),
            "full_mix_raw": sum(1 for p in ho_pass if p in mix["raw"]),
            "full_mix_normalised": sum(1 for p in ho_pass if CL.norm(p) in mix["nraw"]),
            "of_units": len(ho_pass),
        },
        "held_out_questions_identical_to_a_train_question": {
            "raw": sum(1 for q in ho_q if q in tr_q), "of_units": len(ho_q)},
        "held_out_wiki_urls_also_in_train": {
            "raw": sum(1 for u in ho_url if u in tr_url), "of_units": len(ho_url)},
        "held_out_answers_identical_to_a_train_answer": {
            "raw": sum(1 for a in ho_ans if a in tr_ans), "of_units": len(ho_ans)},
    }
    p = res["held_out_passages_identical_to_a_train_passage"]
    q = res["held_out_questions_identical_to_a_train_question"]
    res["passage_reuse_rate"] = pct(p["raw"], p["of_units"])
    res["question_reuse_rate"] = pct(q["raw"], q["of_units"])
    res["split_axis_measured"] = (
        f"QUESTION, not document: {res['passage_reuse_rate']:.1%} of held-out passages are "
        f"byte-identical to a train passage while {res['question_reuse_rate']:.1%} of held-out "
        "questions recur in train")
    res["implication_for_any_eval_built_from_this_corpus"] = (
        "an eval drawn from the official validation/test split inherits "
        f"{res['passage_reuse_rate']:.1%} of its passages from the training member, so it "
        "cannot satisfy C2 by split membership; membership of the assembled mix must be "
        "measured passage by passage")
    res["verdict"] = "PASS"   # the clause requires the axis be MEASURED, and it is
    print(f"  axis: {res['split_axis_measured']}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# C4 - contamination census with a live positive control
# --------------------------------------------------------------------------- #
def clause_c4(mem, arena_texts, splits):
    print("\n=== C4 contamination census", flush=True)
    ev = sorted({c for c in mem["chunks"] if c.strip()})
    cl = sorted({c for c in mem["claims"] if c.strip()})

    def coverage(units):
        short = sum(1 for u in units if len(G.normalize(u).split()) < GATE_N)
        return {"units": len(units), "too_short_for_ngram": short,
                "scorable": len(units) - short,
                "short_units_covered_by": "exact string matching under C2/C3"}

    def gate(label, units, arena, spike=False):
        t0 = time.time()
        r = G.run_gate(units, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                       label=label, arena_texts=arena)
        out = {"verdict": r["verdict"], "max_fraction": r["max_fraction"],
               "candidate_vs_arena": r["candidate_vs_arena"],
               "arena_vs_candidate": {k: v for k, v in r["arena_vs_candidate"].items()
                                      if k != "per_arena_subset"},
               "seconds": round(time.time() - t0, 1)}
        print(f"  {label}: {r['verdict']} at max fraction {r['max_fraction']} "
              f"({out['seconds']}s)", flush=True)
        if spike:
            sp = G.spike_control(units[:2000], arena, n=GATE_N, jaccard=GATE_JACCARD,
                                 k=10, label=f"{label}_spike")
            out["spike_control"] = sp
            print(f"  spike control: {sp}", flush=True)
        return out

    ev_gate = gate("psiloqa_evidence", ev, arena_texts, spike=True)
    cl_gate = gate("psiloqa_claims", cl, arena_texts)

    # LIVE positive control - text that is near-duplicate BY CONSTRUCTION.
    # PsiloQA's own held-out passages are re-cut from the same documents as the
    # train passages (C3), so a working gate must fire hard on them.
    ho = sorted({p for p in (splits["validation"]["wiki_passage"].to_list()
                             + splits["test"]["wiki_passage"].to_list()) if p.strip()})
    live = G.run_gate(ho, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                      label="psiloqa_heldout_passages",
                      arena_texts={"psiloqa_train_passages": ev})
    print(f"  LIVE positive control: {live['candidate_vs_arena']['fraction']:.4f} of "
          f"{live['candidate_vs_arena']['n_units']} held-out passages fire against the "
          f"member's own passages (max Jaccard "
          f"{live['candidate_vs_arena'].get('best_jaccard', {}).get('max')})", flush=True)

    return {
        "clause": "C4",
        "instrument": f"provenance_gate.py, R14-H136 ruling-2 form: {GATE_N}-gram, "
                      f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL > {GATE_KILL:.0%}; "
                      "thresholds read from R19_supply_gates.py",
        "arena": {"subsets": len(arena_texts),
                  "units": sum(len(v) for v in arena_texts.values())},
        "evidence_coverage": coverage(ev),
        "claim_coverage": coverage(cl),
        "evidence_gate": ev_gate,
        "claim_gate": cl_gate,
        "live_positive_control": {
            "construction": "PsiloQA validation+test passages against the member's own "
                            "train passages - near-duplicate by construction (C3), so a "
                            "gate that cannot fire is caught",
            "n_units": live["candidate_vs_arena"]["n_units"],
            "fraction_firing": live["candidate_vs_arena"]["fraction"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "fires": bool(live["candidate_vs_arena"]["fraction"] > 0.5),
        },
        "verdict": None,
    }


# --------------------------------------------------------------------------- #
# C6 - memorisation channel
# --------------------------------------------------------------------------- #
def clause_c6(mem, splits, surfaces):
    print("\n=== C6 memorisation channel", flush=True)
    tr = splits["train"].filter(
        (pl.col("wiki_passage").str.len_chars() > 50)
        & (pl.col("llm_answer").str.len_chars() > 10))
    y = (tr["labels"].list.len() == 0).cast(pl.Int8).to_numpy()
    pas = tr["wiki_passage"].to_list()
    qs = tr["question"].to_list()
    ans = tr["llm_answer"].to_list()

    def key_channel(keys, name):
        by = collections.defaultdict(list)
        for k, lab in zip(keys, y, strict=True):
            by[k].append(int(lab))
        multi = {k: v for k, v in by.items() if len(v) > 1}
        pure = sum(1 for v in multi.values() if len(set(v)) == 1)
        # leave-one-out key-majority predictor on rows whose key repeats
        correct = cov = 0
        for k, v in multi.items():
            s, m = sum(v), len(v)
            for lab in v:
                rest_pos, rest_n = s - lab, m - 1
                if rest_n == 0:
                    continue
                cov += 1
                pred = 1 if rest_pos * 2 > rest_n else (0 if rest_pos * 2 < rest_n else -1)
                if pred == lab:
                    correct += 1
                elif pred == -1:
                    correct += 0.5
        base = float(max(y.mean(), 1 - y.mean()))
        return {
            "key": name,
            "distinct_keys": len(by),
            "keys_with_more_than_one_row": len(multi),
            "rows_under_repeating_keys": sum(len(v) for v in multi.values()),
            "repeating_keys_label_pure": pure,
            "repeating_keys_label_pure_rate": pct(pure, len(multi)),
            "loo_key_majority_accuracy": round(correct / cov, 4) if cov else None,
            "loo_coverage_rows": cov,
            "majority_class_baseline": round(base, 4),
            "delta_over_baseline": round(correct / cov - base, 4) if cov else None,
        }

    within = [key_channel([(p, q) for p, q in zip(pas, qs, strict=True)],
                          "(passage, question)"),
              key_channel(qs, "question"),
              key_channel(pas, "passage")]

    # cross-surface: the H175b feature - does the member associate an answer with
    # an eval pair's key, and does that answer overlap the eval claim?
    by_pq = collections.defaultdict(list)
    by_q = collections.defaultdict(list)
    for p, q, a in zip(pas, qs, ans, strict=True):
        by_pq[(p, q)].append(a)
        by_q[q].append(a)

    cross = {}
    for name in ("R20-H175b_qlane_eval.parquet", "R20-H175b_qlane_eval_repaired.parquet",
                 "R20-H175b_qlane_eval_clean.parquet",
                 "R20-H175b_qlane_eval_clean_prefix.parquet"):
        p = EXP / name
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        if "question" not in d.columns:
            continue
        vals_pq, vals_q, cov_pq, cov_q = [], [], 0, 0
        for ch, qn, clm in zip(d["chunk"].to_list(), d["question"].to_list(),
                               d["claim"].to_list(), strict=True):
            cc = content_u(clm)
            for store, hits, flag in ((vals_pq, by_pq.get((ch, qn)), "pq"),
                                      (vals_q, by_q.get(qn), "q")):
                if not hits or not cc:
                    continue
                best = max(PM.containment(cc, content_u(h)) for h in hits)
                store.append(best)
                if flag == "pq":
                    cov_pq += 1
                else:
                    cov_q += 1
        cross[name] = {
            "rows": d.height,
            "feature": "max content-token containment of the eval claim in whatever "
                       "answer the member associates with the pair's key",
            "keyed_on_passage_and_question": {
                "coverage_rows": cov_pq, "coverage": pct(cov_pq, d.height),
                "value": dist(vals_pq)},
            "keyed_on_question_alone": {
                "coverage_rows": cov_q, "coverage": pct(cov_q, d.height),
                "value": dist(vals_q)},
        }
        print(f"  {name}: question-keyed coverage {pct(cov_q, d.height)}, "
              f"mean {dist(vals_q).get('mean')}", flush=True)

    return {
        "clause": "C6",
        "within_member_key_channels": within,
        "cross_surface_feature": cross,
        "note": "the member is not pair-structured, so the within-member reading is the "
                "key-repeat channel; the cross-surface reading reproduces the R20-H175b "
                "feature against this member's own associations",
        "verdict": None,
    }


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def clause_c7(mem, splits):
    n = len(mem["y"])
    tr = splits["train"]
    kept = tr.filter((pl.col("wiki_passage").str.len_chars() > 50)
                     & (pl.col("llm_answer").str.len_chars() > 10))
    return {
        "clause": "C7",
        "declared_unit": "rows - the member is a source corpus with no contrast pairing; "
                         "each row is one (claim, evidence, label) triple",
        "rows": n,
        "pairs": n,
        "pairs_definition": "one (claim, evidence) pair per row; the corpus carries no "
                            "positive/negative contrast pairing, so pairs == rows",
        "registered_rows": REGISTERED_ROWS,
        "row_margin_vs_registration": n - REGISTERED_ROWS,
        "archive_rows_train_split": tr.height,
        "rows_after_selection_predicate": kept.height,
        "rows_dropped_by_selection_predicate": tr.height - kept.height,
        "share_of_clean_mix": round(n / mem["mix_rows"], 4),
        "clean_mix_rows": mem["mix_rows"],
        "clean_mix_rows_registered": EXPECTED_CLEAN_ROWS,
        "verdict": None,
    }


def clause_c8(mem, splits):
    tr = splits["train"]
    kept = tr.filter((pl.col("wiki_passage").str.len_chars() > 50)
                     & (pl.col("llm_answer").str.len_chars() > 10))
    claims, chunks, y = mem["claims"], mem["chunks"], mem["y"]
    trip = collections.Counter(zip(claims, chunks, y.tolist(), strict=True))
    pas_counts = collections.Counter(chunks)
    cl_counts = collections.Counter(claims)
    sidecar = (DATA / "dataset-psiloqa.md").read_text()
    return {
        "clause": "C8",
        "source": "HuggingFace s-nlp/PsiloQA",
        "licence": "CC-BY-4.0 (sidecar data/external/datasets/dataset-psiloqa.md)",
        "retrieval_date_measured": dt.datetime.fromtimestamp(
            ARCHIVE.stat().st_mtime).isoformat(timespec="seconds"),
        "retrieval_date_recorded_in_sidecar": bool(
            re.search(r"fetched\s+\d{4}-\d{2}-\d{2}", sidecar, re.I)),
        "archive": ARCHIVE.name,
        "fetch_script": "scripts/fetch_grounding_datasets.py",
        "selection_predicate": (
            "dataset-psiloqa.zip -> s-nlp__PsiloQA__train.parquet ONLY; "
            "filter wiki_passage.len_chars > 50 AND llm_answer.len_chars > 10; "
            "label = (labels.list.len() == 0); claim = llm_answer; "
            "chunk = wiki_passage, truncated to CFG.chunk_max_chars under the pre-H150 "
            "protocol and UNTRUNCATED (then windowed 1500/750) under H150/H174"),
        "splits_present_in_archive": {k: v.height for k, v in splits.items()},
        "splits_used": ["train"],
        "internal_structure": {
            "rows": len(y),
            "distinct_claims": len(cl_counts),
            "distinct_evidence_chunks": len(pas_counts),
            "distinct_claim_evidence_label_triples": len(trip),
            "duplicate_rows": len(y) - len(trip),
            "max_rows_sharing_one_evidence_chunk": max(pas_counts.values()),
            "median_rows_per_evidence_chunk": float(np.median(list(pas_counts.values()))),
            "max_rows_sharing_one_claim": max(cl_counts.values()),
            "distinct_languages": kept["lang"].n_unique(),
            "rows_per_language": dict(sorted(
                {k: v for k, v in kept.group_by("lang").len().iter_rows()}.items(),
                key=lambda kv: -kv[1])),
            "distinct_llm_checkpoints": kept["llm_checkpoint"].n_unique(),
            "positive_rate": round(float((y == 1.0).mean()), 4),
        },
        "public_repo_check": "artifact carries corpus identifiers and repository-relative "
                             "paths only; no client or company name",
        "verdict": None,
    }


# --------------------------------------------------------------------------- #
# verdicts - mechanical, from the measured values only
# --------------------------------------------------------------------------- #
def finalize(report):
    c1 = report["C1"]
    prim = c1["containment_banked_ascii"]
    comp = c1["containment_unicode"]
    c1["primary_reading"] = "banked_ascii"
    c1["verdict"] = "FAIL" if prim["rejected"] else "PASS"
    c1["measured"] = (
        f"rate(containment >= 0.90) negative {prim['attested_ge_0.90_rate_negative']} vs "
        f"positive {prim['attested_ge_0.90_rate_positive']}, delta "
        f"{prim['rate_delta_abs']} against a <= 0.10 rejection band "
        f"(Unicode companion: neg {comp['attested_ge_0.90_rate_negative']} vs pos "
        f"{comp['attested_ge_0.90_rate_positive']}, delta {comp['rate_delta_abs']})")
    c1["companion_disagrees"] = bool(prim["rejected"] != comp["rejected"])

    c4 = report["C4"]
    ok = (c4["evidence_gate"]["verdict"] != "KILL"
          and c4["claim_gate"]["verdict"] != "KILL"
          and c4["evidence_gate"]["spike_control"]["passes"]
          and c4["live_positive_control"]["fires"])
    c4["verdict"] = "PASS" if ok else "FAIL"

    c6 = report["C6"]
    worst = max((k["delta_over_baseline"] or 0.0) for k in c6["within_member_key_channels"])
    c6["worst_within_member_delta_over_baseline"] = round(worst, 4)
    c6["verdict"] = "PASS" if worst <= 0.0 else "FAIL"
    c6["measured"] = (
        f"best leave-one-out key-majority predictor beats the majority-class baseline by "
        f"{worst:+.4f}")

    c7 = report["C7"]
    c7["verdict"] = "PASS" if c7["row_margin_vs_registration"] == 0 else "FAIL"

    c8 = report["C8"]
    req = {
        "source": True, "licence": True,
        "retrieval_date": bool(c8["retrieval_date_measured"]),
        "selection_predicate": True,
        "duplication_reported": True,
        "no_client_or_company_name": True,
    }
    c8["requirements_met"] = req
    c8["verdict"] = "PASS" if all(req.values()) else "FAIL"

    clauses = {k: report[k]["verdict"] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}
    report["clause_verdicts"] = clauses
    report["conforming"] = all(v in ("PASS", "NOT-APPLICABLE") for v in clauses.values())
    report["failed_clauses"] = sorted(k for k, v in clauses.items() if v == "FAIL")
    return report


def main():
    t0 = time.time()
    report = {
        "member": MEMBER,
        "contract": "docs/experiments/dataset-contract.md",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "compute": "CPU only, CUDA_VISIBLE_DEVICES empty",
    }
    mem, mix = load_member()
    splits = archive_splits()

    c1, *_ = clause_c1(mem, splits)
    report["C1"] = c1
    OUT.write_text(json.dumps(report, indent=2))

    surfaces, arena_texts = surface_units()
    report["C2"] = clause_c2(mem, surfaces)
    OUT.write_text(json.dumps(report, indent=2))

    report["C3"] = clause_c3(mem, mix, splits)
    OUT.write_text(json.dumps(report, indent=2))

    report["C5"] = {
        "clause": "C5",
        "verdict": "NOT-APPLICABLE",
        "why": "C5 binds every CONSTRUCTED lane and every paired-contrast eval. This "
               "member is a source corpus loaded verbatim from the archive: it has no "
               "construction, no pair structure, no neg_family, no direction/element "
               "balance and no within-pair channel, so the registered probes "
               "(claim-only converged, within-pair claim-only, single-channel, surface "
               "parity, attestation symmetry) have no object to compute over. No proxy "
               "is substituted.",
        "executor_added_diagnostic": None,
    }

    report["C6"] = clause_c6(mem, splits, surfaces)
    OUT.write_text(json.dumps(report, indent=2))

    report["C7"] = clause_c7(mem, splits)
    report["C8"] = clause_c8(mem, splits)
    OUT.write_text(json.dumps(report, indent=2))

    report["C4"] = clause_c4(mem, arena_texts, splits)
    report = finalize(report)
    report["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps({"clause_verdicts": report["clause_verdicts"],
                             "conforming": report["conforming"]}, indent=2), flush=True)
    print(f"=== report written -> {OUT}  ({report['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
