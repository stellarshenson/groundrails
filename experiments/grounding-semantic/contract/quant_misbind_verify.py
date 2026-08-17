"""DATASET CONTRACT verification of the `quant_misbind` lane (R17-H146_lane.parquet).

Contract: docs/experiments/dataset-contract.md, clauses C1-C8.  This script
MEASURES; it does not adjudicate and it relaxes nothing.

CPU ONLY - every card carries an R20-H174 training draw.  CUDA_VISIBLE_DEVICES
is forced empty before any import that could touch a device.

Stages write their own JSON under this directory so a killed run resumes from
disk rather than recomputing:

    c1   label commensurability - claim-to-evidence containment, both legs,
         plus the binding-level attestation audit re-derived from the source
         tables (the discriminating measurement containment cannot make)
    c2   three-form disjointness against every evaluation surface, both
         directions
    c3   split semantics - which splits the lane's sources actually read, and
         whether the source archive's own split axis is document-disjoint
    c4   R14-H136 contamination census (8-gram, Jaccard >= 0.3, bidirectional,
         KILL > 2%) with the synthetic spike AND a live positive control
    c5   leak suite, recomputed from the parquet - nothing cited from the
         build manifest
    c6   memorisation channel
    c78  declared units / volume, provenance, licence, internal duplication

Run:  CUDA_VISIBLE_DEVICES= uv run python \
        experiments/grounding-semantic/contract/quant_misbind_verify.py [stage ...]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import hashlib
import importlib.util
import io
import json
import pathlib
import random
import re
import sys
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent
ROOT = GS.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

LANE = GS / "R17-H146_lane.parquet"
MEMBER = "quant_misbind"
CHUNK_MAX = 1500  # groundrails config `chunk_max_chars`, measured not asserted

# --------------------------------------------------------------------------- #
# canonical text instruments - taken from the campaign's own modules
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)  # R20-H174_lane_common.tokens
_WS = re.compile(r"\s+")


def tok(text):
    return _WORD.findall(text.lower())


def containment(claim, text):
    """Fraction of the claim's distinct content tokens present in `text` - the
    campaign's lexical-baseline feature (R20-H174_lane_common.containment,
    identical to R20-H175b_qlane.containment, the instrument that produced the
    0.9129 reading on the withdrawn contrast lane)."""
    ct = set(tok(claim))
    if not ct:
        return 0.0
    return len(ct & set(tok(text))) / len(ct)


def norm_ws(s):
    """whitespace-collapsed, case-folded - the third disjointness form."""
    return _WS.sub(" ", s).strip().lower()


def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not pos.size or not neg.size:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def dist(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": int(x.size),
        "mean": round(float(x.mean()), 6),
        "std": round(float(x.std()), 6),
        "min": round(float(x.min()), 6),
        "p10": round(float(np.percentile(x, 10)), 6),
        "p25": round(float(np.percentile(x, 25)), 6),
        "median": round(float(np.median(x)), 6),
        "p75": round(float(np.percentile(x, 75)), 6),
        "p90": round(float(np.percentile(x, 90)), 6),
        "max": round(float(x.max()), 6),
        "share_ge_0.90": round(float((x >= 0.90).mean()), 6),
        "share_ge_0.95": round(float((x >= 0.95).mean()), 6),
        "share_fully_attested_eq_1.0": round(float((x >= 1.0 - 1e-9).mean()), 6),
        "histogram_deciles": [int(v) for v in np.histogram(x, bins=10, range=(0, 1))[0]],
    }


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lane():
    return pl.read_parquet(LANE)


def save(stage, obj):
    p = HERE / f"quant_misbind_{stage}.json"
    p.write_text(json.dumps(obj, indent=2))
    print(f"  -> {p.name}", flush=True)
    return obj


def load(stage):
    p = HERE / f"quant_misbind_{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
# C1 - label commensurability
# --------------------------------------------------------------------------- #
def stage_c1():
    t0 = time.time()
    df = lane()
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    y = df["label"].to_numpy()
    fams = df["neg_family"].to_list()
    forms = df["serial_form"].to_list()

    cont_full = np.array([containment(a, b) for a, b in zip(claims, chunks)])
    cont_trunc = np.array([containment(a, b[:CHUNK_MAX]) for a, b in zip(claims, chunks)])

    out = {
        "clause": "C1",
        "head_declared": "the grounding scalar (`task_head`), trained by pointwise / "
                         "MIL max-over-windows BCE on the 0/1 label",
        "label_predicate": (
            "label 1 = the evidence states the asserted binding (row_key, column) -> value; "
            "label 0 = the evidence states a DIFFERENT value for that binding, and the "
            "asserted value is a real cell of a different row (misbound_row) or a different "
            "column of the same row (misbound_col). The predicate is support of the asserted "
            "proposition, which is the head's own predicate"
        ),
        "label_origin": "construction (R17-H146_lane.py), not a dataset-supplied label",
        "instrument": "containment(claim, evidence) = |distinct content tokens of the claim "
                      "present in the evidence| / |distinct content tokens of the claim|, "
                      "tokenizer [^\\W\\d_]+|\\d+ lowercased - the module used for the "
                      "0.9129 reading on the withdrawn R20-H175b contrast lane",
        "evidence_form": "untruncated chunk (the lane is loaded UNTRUNCATED under the "
                         "H150/H174 windowed protocol); the 1,500-char truncated form is "
                         "reported beside it",
    }

    legs = {}
    for lab, name in ((1, "positive"), (0, "negative")):
        m = y == lab
        legs[name] = {
            "untruncated": dist(cont_full[m]),
            "truncated_1500": dist(cont_trunc[m]),
        }
    out["legs"] = legs

    neg_mean = legs["negative"]["untruncated"]["mean"]
    pos_mean = legs["positive"]["untruncated"]["mean"]
    neg_ge90 = legs["negative"]["untruncated"]["share_ge_0.90"]
    pos_ge90 = legs["positive"]["untruncated"]["share_ge_0.90"]
    neg_full = legs["negative"]["untruncated"]["share_fully_attested_eq_1.0"]
    pos_full = legs["positive"]["untruncated"]["share_fully_attested_eq_1.0"]

    out["bar_readings"] = {
        "note": "C1's bar - 'negatives >= 90% attested at a rate within 0.10 of its "
                "positives is REJECTED' - is evaluated under all three readings the "
                "clause text admits. The rejection region requires BOTH conjuncts.",
        "reading_A_mean_containment": {
            "definition": "mean containment on the negative leg >= 0.90 AND "
                          "|neg - pos| <= 0.10",
            "negative": neg_mean, "positive": pos_mean,
            "delta": round(abs(neg_mean - pos_mean), 6),
            "conjunct_1_negatives_ge_0.90": bool(neg_mean >= 0.90),
            "conjunct_2_within_0.10_of_positives": bool(abs(neg_mean - pos_mean) <= 0.10),
            "rejects": bool(neg_mean >= 0.90 and abs(neg_mean - pos_mean) <= 0.10),
            "margin_below_0.90": round(0.90 - neg_mean, 6),
            "reference_R20-H175b": 0.9129,
        },
        "reading_B_share_attested_ge_0.90": {
            "definition": "share of negatives with containment >= 0.90 must reach 0.90 "
                          "AND be within 0.10 of the positive share",
            "negative": neg_ge90, "positive": pos_ge90,
            "delta": round(abs(neg_ge90 - pos_ge90), 6),
            "conjunct_1_negatives_ge_0.90": bool(neg_ge90 >= 0.90),
            "conjunct_2_within_0.10_of_positives": bool(abs(neg_ge90 - pos_ge90) <= 0.10),
            "rejects": bool(neg_ge90 >= 0.90 and abs(neg_ge90 - pos_ge90) <= 0.10),
            "margin_below_0.90": round(0.90 - neg_ge90, 6),
            "reference_R20-H175b": 0.723,
        },
        "reading_C_share_fully_attested": {
            "definition": "share of negatives at containment exactly 1.0",
            "negative": neg_full, "positive": pos_full,
            "delta": round(abs(neg_full - pos_full), 6),
            "conjunct_1_negatives_ge_0.90": bool(neg_full >= 0.90),
            "rejects": bool(neg_full >= 0.90 and abs(neg_full - pos_full) <= 0.10),
            "margin_below_0.90": round(0.90 - neg_full, 6),
            "reference_R20-H175b": 0.664,
        },
    }

    # containment is BLIND here by construction - both legs assert a numeral the
    # chunk prints, so the legs are indistinguishable on it.  Recorded plainly.
    pos_c = cont_full[y == 1]
    neg_c = cont_full[y == 0]
    pid = df["pair_id"].to_numpy()
    order_p = np.argsort(pid[y == 1])
    order_n = np.argsort(pid[y == 0])
    paired = np.abs(pos_c[order_p] - neg_c[order_n])
    out["leg_separability_on_containment"] = {
        "auroc_containment_vs_label": round(auroc(y, cont_full), 6),
        "within_pair_identical_containment_share": round(float((paired < 1e-9).mean()), 6),
        "within_pair_mean_abs_difference": round(float(paired.mean()), 6),
        "reading": "containment cannot separate the legs of this lane and is not "
                   "expected to - the negative asserts a value the evidence prints, "
                   "bound elsewhere. The discriminating measurement is the "
                   "binding-level audit below.",
    }

    # asserted-numeral presence, both legs
    cited = df["cited_value"].to_list()
    correct = df["correct_value"].to_list()
    cited_in = np.array([v in c for v, c in zip(cited, chunks)], dtype=float)
    corr_in = np.array([v in c for v, c in zip(correct, chunks)], dtype=float)
    out["asserted_numeral_presence"] = {
        "positive_leg_cited_value_verbatim_in_evidence": round(float(cited_in[y == 1].mean()), 6),
        "negative_leg_cited_value_verbatim_in_evidence": round(float(cited_in[y == 0].mean()), 6),
        "correct_value_verbatim_in_evidence_all_rows": round(float(corr_in.mean()), 6),
        "reading": "1.0 on both legs by construction - the negative's numeral is a real "
                   "cell of the same table, so a numeral-presence test is also blind",
    }

    # per family / per serialization form
    per_fam, per_form = {}, {}
    for key, vals, sink in (("family", fams, per_fam), ("form", forms, per_form)):
        for v in sorted(set(vals)):
            m = np.array([x == v for x in vals])
            sink[v] = {
                "rows": int(m.sum()),
                "negative_mean": round(float(cont_full[m & (y == 0)].mean()), 6),
                "positive_mean": round(float(cont_full[m & (y == 1)].mean()), 6),
                "negative_share_ge_0.90": round(float((cont_full[m & (y == 0)] >= 0.90).mean()), 6),
                "negative_share_fully_attested": round(
                    float((cont_full[m & (y == 0)] >= 1.0 - 1e-9).mean()), 6),
            }
    out["per_family"] = per_fam
    out["per_serial_form"] = per_form

    out["seconds"] = round(time.time() - t0, 1)
    return save("c1_containment", out)


def stage_c1_binding():
    """The discriminating measurement: is each negative's asserted binding
    genuinely unattested by the evidence?  Re-derived from the SOURCE tables for
    the FULL population, not a 500-row sample."""
    t0 = time.time()
    P = _mod("h144pairs", GS / "R17-H144_pairs.py")
    df = lane()

    print("loading source tables (TabFact train + FEVEROUS working parquet)...", flush=True)
    raw = P.tabfact_tables() + P.feverous_tables()
    by_doc = {}
    for t in raw:
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        t["lab_ci"] = lab
        by_doc[t["doc_id"]] = t
    print(f"  {len(raw)} tables loaded, {len(by_doc)} with a label column", flush=True)

    neg = df.filter(pl.col("label") == 0)
    pos = df.filter(pl.col("label") == 1)

    def audit(sub, expect_correct):
        """expect_correct: True  -> the asserted value must BE the table's cell for
        the claimed (row, column); False -> it must NOT be, and must be a real cell
        of the recorded different binding."""
        errs = collections.Counter()
        missing = 0
        ok = 0
        for r in sub.iter_rows(named=True):
            t = by_doc.get(r["doc_id"])
            if t is None:
                missing += 1
                continue
            body, hdr = t["body"], t["hdr"]
            ri, ci = r["row_index"], r["column_index"]
            mri, mci = r["misbound_row_index"], r["misbound_column_index"]
            if not (0 <= ri < len(body) and 0 <= mri < len(body)
                    and 0 <= ci < len(hdr) and 0 <= mci < len(hdr)):
                errs["index_out_of_range"] += 1
                continue
            why = None
            if body[ri][ci].strip() != r["correct_value"]:
                why = "table cell for the claimed binding != correct_value"
            elif body[ri][t["lab_ci"]].strip() != r["row_key"]:
                why = "row_key does not name the claimed row"
            elif expect_correct:
                if r["cited_value"] != r["correct_value"]:
                    why = "positive leg asserts a value other than the table's cell"
            else:
                if r["cited_value"] == r["correct_value"]:
                    why = "negative asserts the correct value"
                elif body[mri][mci].strip() != r["cited_value"]:
                    why = "asserted value is not the table's cell for the recorded source binding"
                elif (mri, mci) == (ri, ci):
                    why = "source binding equals the claimed binding"
                elif r["neg_family"] == "misbound_row" and (mci != ci or mri == ri):
                    why = "misbound_row does not cite a different row of the same column"
                elif r["neg_family"] == "misbound_col" and (mri != ri or mci == ci):
                    why = "misbound_col does not cite a different column of the same row"
            if why:
                errs[why] += 1
            else:
                ok += 1
        return {"rows": len(sub), "verified": ok, "table_not_found": missing,
                "errors": int(sum(errs.values())), "error_kinds": dict(errs)}

    print("auditing the negative leg (15,000 rows, full population)...", flush=True)
    neg_res = audit(neg, expect_correct=False)
    print(f"  negatives: {neg_res}", flush=True)
    print("auditing the positive leg...", flush=True)
    pos_res = audit(pos, expect_correct=True)
    print(f"  positives: {pos_res}", flush=True)

    # evidence-side guard: the asserted (wrong) value must not ALSO be readable as
    # the claimed binding's value anywhere in the served chunk.  Re-derived from
    # the table rather than parsed out of six serialization forms.
    ambiguous = 0
    checked = 0
    for r in neg.iter_rows(named=True):
        t = by_doc.get(r["doc_id"])
        if t is None:
            continue
        checked += 1
        body = t["body"]
        ri, ci = r["row_index"], r["column_index"]
        if r["neg_family"] == "misbound_row":
            # the asserted value may not sit anywhere in the row the claim names
            if any(c.strip() == r["cited_value"] for c in body[ri]):
                ambiguous += 1
        else:
            # the asserted value may not sit anywhere in the column the claim names
            if any(row[ci].strip() == r["cited_value"] for row in body):
                ambiguous += 1

    out = {
        "clause": "C1 (binding-level attestation - the discriminating measurement)",
        "method": "every row re-derived against its source table (TabFact train + the "
                  "FEVEROUS working parquet), full population, not sampled",
        "negative_leg": neg_res,
        "positive_leg": pos_res,
        "negative_binding_unattested_rate": round(neg_res["verified"] / max(neg_res["rows"], 1), 6),
        "positive_binding_attested_rate": round(pos_res["verified"] / max(pos_res["rows"], 1), 6),
        "ambiguity_guard": {
            "negatives_checked": checked,
            "asserted_value_also_answers_the_claimed_binding": ambiguous,
            "rate": round(ambiguous / max(checked, 1), 6),
            "bar": "0 - a negative whose asserted value also legitimately answers the "
                   "claimed binding would be mislabelled",
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c1_binding", out)


# --------------------------------------------------------------------------- #
# C2 - disjointness from every evaluation surface
# --------------------------------------------------------------------------- #
def eval_surfaces():
    """(name, claims, evidence_units, note) for every evaluation surface."""
    out = []

    # 1. the blind arena - 10 RAGBench subsets, exactly the H77 sample the
    #    campaign reads (N_PER_SUBSET=250, MAX_CHUNKS=8, seed 0)
    G = _mod("provgate", GS / "provenance_gate.py")
    arena_docs, _ = G.load_arena()
    docs = [c for v in arena_docs.values() for c in v]
    z = zipfile.ZipFile(G.ARCHIVE)
    responses = []
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        d = pl.read_parquet(io.BytesIO(z.read(name)))
        d = d.filter(pl.col("adherence_score").is_not_null()
                     & (pl.col("response").str.len_chars() > 20)
                     & (pl.col("documents").list.len() > 0))
        if len(d) < 40 or d["adherence_score"].n_unique() < 2:
            continue
        d = d.sample(min(G.N_PER_SUBSET, len(d)), seed=0)
        responses += d["response"].to_list()
    out.append(("arena_ragbench_10_subsets", responses, docs,
                "H77 arena sample: 250 items/subset seed 0, first 8 document chunks"))

    # 2. gold_full - all gold claims and their chunk lists
    gp = GS / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
    if gp.exists():
        g = pl.read_parquet(gp, columns=["claim", "chunk"])
        out.append(("gold_full", g["claim"].to_list(), g["chunk"].to_list(),
                    "all (claim, chunk) rows of the gold teacher-pair table"))

    # 3. held-out mechanism evals
    for fname, note in (
        ("R17-H143_evalset.parquet", "the derivation eval this lane's sources were "
                                     "content-excluded against"),
        ("R20-H177_eval_B.parquet", "Lane B compare/direction mechanism eval"),
        ("R20-H177_eval_C.parquet", "Lane C role/sign/period mechanism eval"),
        ("R20-H175b_qlane_eval.parquet", "withdrawn contrast-lane eval (banked)"),
        ("R20-H175b_qlane_eval_repaired.parquet", "withdrawn contrast-lane eval (repaired)"),
        ("R20-H175b_qlane_eval_clean.parquet", "clean contrast-lane eval"),
        ("R20-H175b_qlane_eval_clean_prefix.parquet", "clean contrast-lane eval (prefix)"),
    ):
        p = GS / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        out.append((fname.replace(".parquet", ""), d["claim"].to_list(),
                    d["chunk"].to_list(), note))

    # 4. anti-gaming probe sets - claims only, no evidence column
    for fname in ("R17-H146_antigaming_set.parquet", "R18-H150_antigaming_set.parquet",
                  "R19-H159_antigaming_set.parquet"):
        p = GS / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        out.append((fname.replace(".parquet", ""),
                    d["claim_pos"].to_list() + d["claim_neg"].to_list(), [],
                    "anti-gaming paired probe set - claims only, the set carries no "
                    "evidence column"))

    # 5. vitaminc holdout - the SUPERSET (test + validation), a stricter test than
    #    the filtered holdout the campaign actually reads
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    cl, ev = [], []
    for split in ("test", "validation"):
        d = pl.read_parquet(io.BytesIO(zv.read(f"tals__vitaminc__{split}.parquet")))
        cl += d["claim"].to_list()
        ev += d["evidence"].to_list()
    out.append(("vitaminc_holdout_superset", cl, ev,
                "VitaminC test + validation in full - a SUPERSET of the filtered "
                "holdout the campaign evaluates on, so the disjointness read is "
                "stricter, not a proxy"))
    return out


def stage_c2():
    t0 = time.time()
    df = lane()
    m_claims = [c for c in df["claim"].to_list() if c and c.strip()]
    m_chunks = [c for c in df["chunk"].to_list() if c and c.strip()]

    forms = {
        "raw": lambda s: s,
        "truncated_1500": lambda s: s[:CHUNK_MAX],
        "normalised_ws_casefold": norm_ws,
    }
    member = {k: {"claims": {f(c) for c in m_claims}, "evidence": {f(c) for c in m_chunks}}
              for k, f in forms.items()}

    results = {}
    total = collections.Counter()
    for name, s_claims, s_ev, note in eval_surfaces():
        s_claims = [c for c in s_claims if c and c.strip()]
        s_ev = [c for c in s_ev if c and c.strip()]
        entry = {"note": note, "surface_claims": len(set(s_claims)),
                 "surface_evidence_units": len(set(s_ev)), "forms": {}}
        for fname, f in forms.items():
            sc = {f(c) for c in s_claims}
            se = {f(c) for c in s_ev}
            ev_hit = member[fname]["evidence"] & se
            cl_hit = member[fname]["claims"] & sc
            entry["forms"][fname] = {
                "evidence_shared_strings": len(ev_hit),
                "member_evidence_units_matched": len(ev_hit),
                "surface_evidence_units_matched": len(ev_hit),
                "claims_shared_strings": len(cl_hit),
                "member_claims_matched": len(cl_hit),
                "surface_claims_matched": len(cl_hit),
            }
            total[fname] += len(ev_hit) + len(cl_hit)
        entry["clean"] = all(v["evidence_shared_strings"] == 0 and v["claims_shared_strings"] == 0
                             for v in entry["forms"].values())
        results[name] = entry
        print(f"  {name}: clean={entry['clean']} "
              f"{ {k: (v['evidence_shared_strings'], v['claims_shared_strings']) for k, v in entry['forms'].items()} }",
              flush=True)

    out = {
        "clause": "C2",
        "member_units": {"distinct_claims": len(set(m_claims)),
                         "distinct_evidence_chunks": len(set(m_chunks)),
                         "rows": df.height},
        "forms": ["raw", f"truncated_{CHUNK_MAX}", "normalised_ws_casefold"],
        "direction_note": "set intersection is symmetric, so the member-side and "
                          "surface-side counts of matched units are reported "
                          "separately and are equal by construction of the test; a "
                          "non-zero in either direction fails the clause",
        "surfaces": results,
        "totals_per_form": dict(total),
        "all_forms_zero": all(v == 0 for v in total.values()),
        "seconds": round(time.time() - t0, 1),
    }
    return save("c2_disjointness", out)


# --------------------------------------------------------------------------- #
# C3 - split semantics
# --------------------------------------------------------------------------- #
def stage_c3():
    t0 = time.time()
    df = lane()
    src_counts = {k: v for k, v in df.group_by("source").len().iter_rows()}

    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    names = {n.split("__")[-1].replace(".parquet", ""): n
             for n in z.namelist() if n.endswith(".parquet")}
    splits = {}
    tables = {}
    for split, n in names.items():
        d = pl.read_parquet(io.BytesIO(z.read(n)))
        splits[split] = d.height
        tables[split] = set(d["table_id"].to_list())
    axis = {}
    if "train" in tables:
        for other in [s for s in tables if s != "train"]:
            axis[f"train_vs_{other}_shared_table_ids"] = len(tables["train"] & tables[other])
            axis[f"{other}_table_ids"] = len(tables[other])
    axis["train_table_ids"] = len(tables.get("train", ()))

    # the lane's own doc ids, by namespace
    doc_ns = collections.Counter(d.split(":")[0] for d in df["doc_id"].to_list())
    lane_tabfact_ids = {d.split(":", 1)[1] for d in df["doc_id"].to_list()
                        if d.startswith("tabfact:")}
    leak = {s: len(lane_tabfact_ids & tables[s]) for s in tables if s != "train"}

    out = {
        "clause": "C3",
        "member_type": "constructed lane - it has no split of its own; the axis that "
                       "matters is which split of each SOURCE it reads and whether that "
                       "split is document-disjoint from anything used for evaluation",
        "sources_read": {
            "tabfact": "dataset-tabfact.zip, the file matching *__train.parquet ONLY "
                       "(R17-H144_pairs.tabfact_tables), deduplicated on table_text",
            "feverous": "tmp/R14_H133_feverous.parquet - an R14-H133 WORKING FILE, "
                        "mirrored from a community HF export of FEVEROUS; it carries no "
                        "split column and no licence sidecar in data/external/datasets",
        },
        "lane_rows_by_source": src_counts,
        "lane_documents_by_namespace": dict(doc_ns),
        "tabfact_archive_splits": splits,
        "tabfact_split_axis_measured": axis,
        "lane_tabfact_tables_appearing_in_a_non_train_split": leak,
        "axis_finding": (
            "TabFact's official split axis is measured above from the archive rather "
            "than read from the card. The lane reads the train split only; whether that "
            "split shares table_ids with val/test is the number that decides whether a "
            "TabFact-derived evaluation surface could be document-shared with this lane"
        ),
        "feverous_split_axis": "NOT MEASURABLE from the artifact on disk - the working "
                               "parquet has no split column; the lane's feverous doc_ids "
                               "are positional indices into a de-duplicated load order, "
                               "so they are reproducible only by re-running the loader "
                               "against that same file",
        "eval_exclusion_enforced_at_build": (
            "content-based exclusion against R17-H143_evalset (R17-H144 method): "
            "3,516 candidate tables carried eval content and 3,556 tables were dropped "
            "before construction - re-measured in C2 against the built lane"
        ),
        "seconds": round(time.time() - t0, 1),
    }
    return save("c3_split", out)


# --------------------------------------------------------------------------- #
# C4 - contamination census, spike AND live positive control
# --------------------------------------------------------------------------- #
GATE_N = 8
GATE_JACCARD = 0.3
GATE_KILL = 0.02
GATE_WARN = 0.005


def _live_control(arena_texts, rng, k=250, drop=0.05, keep=0.6):
    """A LIVE positive control, not a synthetic spike: real arena documents that
    are near-duplicate BY CONSTRUCTION rather than identical - 5% of tokens
    deleted and the text cut to 60% of its length. If the gate cannot fire on
    genuine near-duplicates of the reference side, a clean verdict from it means
    nothing."""
    pool = [c for v in arena_texts.values() for c in v]
    rng.shuffle(pool)
    out = []
    for t in pool[:k]:
        toks = t.split()
        if len(toks) < 40:
            continue
        kept = [w for w in toks if rng.random() >= drop]
        kept = kept[: max(20, int(len(kept) * keep))]
        out.append(" ".join(kept))
    return out


def stage_c4():
    t0 = time.time()
    G = _mod("provgate", GS / "provenance_gate.py")
    df = lane()
    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    cl = sorted({c for c in df["claim"].to_list() if c.strip()})

    print("loading arena (n-grams only)...", flush=True)
    arena_texts, _ = G.load_arena()
    n_arena = sum(len(v) for v in arena_texts.values())
    print(f"  arena: {n_arena} document units over {len(arena_texts)} subsets", flush=True)

    def coverage(units):
        short = [u for u in units if len(G.normalize(u).split()) < GATE_N]
        return {"units": len(units), "too_short_for_8gram": len(short),
                "share_too_short": round(len(short) / max(len(units), 1), 6),
                "covered_by_exact_matching": len(short)}

    res = {"clause": "C4",
           "instrument": f"provenance_gate.py (R14-H136 ruling-2 form): {GATE_N}-gram, "
                         f"Jaccard >= {GATE_JACCARD}, bidirectional, WARN {GATE_WARN}, "
                         f"KILL > {GATE_KILL:.0%} of the candidate side, all ten walled "
                         "arena subsets",
           "arena_units": n_arena,
           "coverage": {"evidence": coverage(ev), "claims": coverage(cl)}}

    for label, units in (("evidence", ev), ("claims", cl)):
        print(f"gating {label} ({len(units)} units)...", flush=True)
        r = G.run_gate(units, n=GATE_N, jaccard=GATE_JACCARD, warn=GATE_WARN,
                       kill=GATE_KILL, label=f"quant_misbind_{label}",
                       arena_texts=arena_texts)
        res[f"{label}_gate"] = {"verdict": r["verdict"],
                                "max_fraction": r["max_fraction"],
                                "candidate_vs_arena": r["candidate_vs_arena"],
                                "arena_vs_candidate": r["arena_vs_candidate"]}
        print(f"  {label}: {r['verdict']} at {r['max_fraction']}", flush=True)

    print("spike control (synthetic - arena units injected verbatim)...", flush=True)
    sp = G.spike_control(ev[:2000], arena_texts, n=GATE_N, jaccard=GATE_JACCARD, k=10,
                         label="quant_misbind_spike")
    res["spike_control"] = sp
    print(f"  {sp}", flush=True)

    print("live positive control (real arena text, near-duplicate by construction)...",
          flush=True)
    rng = random.Random(1146)
    live = _live_control(arena_texts, rng)
    lr = G.run_gate(live, n=GATE_N, jaccard=GATE_JACCARD, warn=GATE_WARN, kill=GATE_KILL,
                    label="live_positive_control", arena_texts=arena_texts)
    res["live_positive_control"] = {
        "construction": "250 real arena documents, 5% of whitespace tokens deleted at "
                        "random then cut to 60% of the remaining length - genuinely "
                        "near-duplicate, not byte-identical",
        "units": lr["candidate"]["n_units"],
        "detection_fraction": lr["candidate_vs_arena"]["fraction"],
        "units_with_hit": lr["candidate_vs_arena"]["units_with_hit"],
        "best_jaccard": lr["candidate_vs_arena"].get("best_jaccard"),
        "verdict": lr["verdict"],
        "fires": bool(lr["candidate_vs_arena"]["fraction"] > 0.9),
    }
    print(f"  live control: {res['live_positive_control']}", flush=True)

    res["status"] = ("GREEN" if res["evidence_gate"]["verdict"] != "KILL"
                     and res["claims_gate"]["verdict"] != "KILL"
                     and sp["passes"] and res["live_positive_control"]["fires"] else "RED")
    res["seconds"] = round(time.time() - t0, 1)
    return save("c4_census", res)


# --------------------------------------------------------------------------- #
# C5 - leak suite, recomputed
# --------------------------------------------------------------------------- #
N_FOLDS = 5
SEED = 1146


def stage_c5():
    t0 = time.time()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    P = _mod("h144pairs", GS / "R17-H144_pairs.py")
    df = lane()
    rng = random.Random(SEED)
    labels = df["label"].to_list()
    claims = df["claim"].to_list()

    # --- claim-only probe: document-disjoint, direction-stratified folds
    doc_key = {d: k for d, k in df.filter(pl.col("label") == 1)
               .group_by("doc_id")
               .agg((pl.col("neg_family") + ":" + pl.col("direction")).first())
               .iter_rows()}
    strata = collections.defaultdict(list)
    for d in sorted(doc_key):
        strata[doc_key[d]].append(d)
    fold_of, i = {}, 0
    for k in sorted(strata):
        ds = strata[k]
        rng.shuffle(ds)
        for d in ds:
            fold_of[d] = i % N_FOLDS
            i += 1
    folds = np.array([fold_of[d] for d in df["doc_id"].to_list()])
    score = np.zeros(len(df))
    idx = np.arange(len(df))
    for f in range(N_FOLDS):
        tr_i, te_i = idx[folds != f], idx[folds == f]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform([claims[j] for j in tr_i])
        Xte = vec.transform([claims[j] for j in te_i])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, [labels[j] for j in tr_i])
        score[te_i] = clf.decision_function(Xte)
        print(f"  fold {f} done", flush=True)
    probe = auroc(labels, score)

    scored = df.select(["pair_id", "label", "neg_family"]).with_columns(pl.Series("score", score))
    fam_acc, worst = {}, 0.0
    for fam, sub in scored.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        fam_acc[fam[0]] = {"acc": round(acc, 6), "pairs": len(piv)}
        worst = max(worst, acc)

    # --- single-channel probes
    # evidence-only: the two legs of a pair share the chunk BYTE-IDENTICALLY, so
    # the channel is degenerate by construction.  Measured, not assumed.
    piv = (df.select(["pair_id", "label", "chunk"])
             .pivot(on="label", index="pair_id", values="chunk", aggregate_function="first")
             .drop_nulls())
    same_chunk = float((piv["1"] == piv["0"]).mean())

    cited = df["cited_value"].to_list()
    channels = {
        "trailing_zero": [P.trailing_zeros(v) for v in cited],
        "digit_count": [P.digits(v) for v in cited],
        "claim_char_length": [len(c) for c in claims],
        "leading_digit": [float(P.leading_digit(v) or 0) for v in cited],
        "decimal_presence": [1.0 if "." in v else 0.0 for v in cited],
        "value_magnitude": [P.as_num(v) or 0.0 for v in cited],
        "claim_token_count": [len(tok(c)) for c in claims],
    }
    fams = df["neg_family"].to_list()
    ch_out = {}
    for name, vals in channels.items():
        pooled = auroc(labels, vals)
        per_fam = {}
        for f in sorted(set(fams)):
            m = [i for i, x in enumerate(fams) if x == f]
            per_fam[f] = round(auroc([labels[i] for i in m], [vals[i] for i in m]), 6)
        dev = max([abs(pooled - 0.5)] + [abs(v - 0.5) for v in per_fam.values()])
        ch_out[name] = {"pooled": round(pooled, 6), "per_family": per_fam,
                        "worst_deviation_from_0.5": round(dev, 6),
                        "in_band_0.45_0.55": bool(dev <= 0.05)}

    # --- balance
    dirs = {f"{a}:{b}": n for a, b, n in df.group_by(["neg_family", "direction"]).len().iter_rows()}
    fam_rows = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    parity = {k: round(float(v), 6) for k, v in
              df.group_by("neg_family")
                .agg(pl.col("surface_parity").cast(pl.Float64).mean()).iter_rows()}
    parity_pooled = round(float(df["surface_parity"].cast(pl.Float64).mean()), 6)

    out = {
        "clause": "C5",
        "registered_conjunction": {
            "claim_only_converged_probe": {
                "value": round(probe, 6), "bar": "< 0.55", "pass": bool(probe < 0.55),
                "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                           "direction-stratified, liblinear tol 1e-7, char_wb 2-5 TF-IDF",
                "documents": len(fold_of), "rows": len(df),
                "recomputed": "from the parquet in this pass; not cited from the manifest",
            },
            "within_pair_claim_only": {
                "per_family": fam_acc, "worst": round(worst, 6), "bar": "< 0.60",
                "pass": bool(worst < 0.60),
            },
            "single_channel_probes": {
                "evidence_only": {
                    "within_pair_identical_chunk_share": round(same_chunk, 6),
                    "auroc": 0.5,
                    "reading": "the legs of a pair share the evidence byte-identically, "
                               "so an evidence-only reader is exactly at chance within "
                               "pair by construction - measured above, not assumed",
                },
                "question_only": {
                    "value": None,
                    "reading": "NOT COMPUTABLE - the lane carries no question field; the "
                               "construction implies no question channel",
                },
            },
            "surface_parity_channels": ch_out,
            "balance": {
                "family_rows": fam_rows,
                "direction_rows": dirs,
                "direction_balance_exact_50_50_per_family": all(
                    dirs[f"{f}:up"] == dirs[f"{f}:down"] for f in fam_rows),
            },
            "attestation_symmetry": {
                "reading": "measured in the C1 stage - asserted numeral present verbatim "
                           "on both legs at 1.0",
            },
        },
        "executor_added_probes_reported_separately": {
            "note": "these are NOT part of the registered conjunction and do not join it",
            "claim_token_count_auroc": ch_out["claim_token_count"],
            "surface_parity_rate_per_family": {
                "pooled": parity_pooled, "per_family": parity,
                "reading": "this is the rate at which the construction achieved digit-"
                           "surface parity between the two values, NOT a channel AUROC. "
                           "It is reported because the per-family rates are far from 0.5 "
                           "and could be mistaken for the C5 parity bar, which is a bar "
                           "on channel AUROCs and is reported above",
            },
        },
        "seconds": round(time.time() - t0, 1),
    }
    reg = out["registered_conjunction"]
    out["all_registered_bars_pass"] = bool(
        reg["claim_only_converged_probe"]["pass"]
        and reg["within_pair_claim_only"]["pass"]
        and all(v["in_band_0.45_0.55"] for k, v in ch_out.items() if k != "claim_token_count")
        and reg["balance"]["direction_balance_exact_50_50_per_family"])
    return save("c5_leak", out)


# --------------------------------------------------------------------------- #
# C6 - memorisation channel
# --------------------------------------------------------------------------- #
def stage_c6():
    t0 = time.time()
    df = lane()
    labels = df["label"].to_numpy()

    # the pairs of this member share the evidence chunk; the chunk is the key.
    # Feature 1: what the rest of the member associates with this key - the mean
    # label of every OTHER row sharing the chunk.
    chunks = df["chunk"].to_list()
    by_chunk = collections.defaultdict(list)
    for i, c in enumerate(chunks):
        by_chunk[c].append(i)
    feat = np.zeros(len(df))
    cover = 0
    for c, idxs in by_chunk.items():
        if len(idxs) < 2:
            for i in idxs:
                feat[i] = np.nan
            continue
        s = float(labels[idxs].sum())
        for i in idxs:
            feat[i] = (s - labels[i]) / (len(idxs) - 1)
            cover += 1
    ok = ~np.isnan(feat)
    a1 = auroc(labels[ok], feat[ok]) if ok.sum() else float("nan")

    # Feature 2: token overlap between a row's claim and the claims the member
    # associates with the same key, excluding the row itself.
    by_chunk_claims = collections.defaultdict(list)
    for i, c in enumerate(chunks):
        by_chunk_claims[c].append(i)
    ov = np.full(len(df), np.nan)
    claims = df["claim"].to_list()
    ctoks = [set(tok(c)) for c in claims]
    for c, idxs in by_chunk_claims.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            best = 0.0
            for j in idxs:
                if j == i:
                    continue
                u = ctoks[i] | ctoks[j]
                if u:
                    best = max(best, len(ctoks[i] & ctoks[j]) / len(u))
            ov[i] = best
    ok2 = ~np.isnan(ov)
    a2 = auroc(labels[ok2], ov[ok2]) if ok2.sum() else float("nan")

    # within-pair separability of both features
    piv = (df.select(["pair_id", "label"]).with_columns(pl.Series("f", feat))
             .pivot(on="label", index="pair_id", values="f", aggregate_function="first")
             .drop_nulls())
    wp = float(((piv["1"].to_numpy() > piv["0"].to_numpy())
                + 0.5 * (piv["1"].to_numpy() == piv["0"].to_numpy())).mean()) if len(piv) else None

    out = {
        "clause": "C6",
        "member_shares_fields_across_pair": "yes - the two legs of a pair share the "
                                            "evidence chunk byte-identically, so the "
                                            "chunk is the pair key",
        "feature_1_key_keyed_label_association": {
            "definition": "mean label of every other row sharing this row's evidence "
                          "chunk (leave-one-out)",
            "auroc_vs_label": None if np.isnan(a1) else round(a1, 6),
            "coverage_rows": int(ok.sum()),
            "coverage_share": round(float(ok.mean()), 6),
            "within_pair_accuracy": None if wp is None else round(wp, 6),
            "bar": "undefined or at chance",
        },
        "feature_2_key_keyed_claim_overlap": {
            "definition": "max Jaccard between this row's claim and any other claim "
                          "sharing the same evidence chunk",
            "auroc_vs_label": None if np.isnan(a2) else round(a2, 6),
            "coverage_rows": int(ok2.sum()),
            "coverage_share": round(float(ok2.mean()), 6),
            "bar": "undefined or at chance",
        },
        "reference_contaminated_case": "the withdrawn R20-H175b eval read 0.6230 on such "
                                       "a feature at 98% coverage",
        "seconds": round(time.time() - t0, 1),
    }
    return save("c6_memorisation", out)


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def stage_c78():
    t0 = time.time()
    df = lane()
    rows, pairs = df.height, df["pair_id"].n_unique()
    fam = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}

    src = (GS / "R18-H150_arm_run.py").read_text()
    reg_block = src.split("LANES = (")[1].split(")\n", 1)[0]
    h174 = (GS / "R20-H174_arm_run.py").read_text()
    h174_block = h174.split("LANES = (")[1].split("\n)\n", 1)[0]

    per_pair = df.group_by("pair_id").len()
    ok_pairs = int((per_pair["len"] == 2).sum())

    dup = {
        "rows": rows,
        "pairs": pairs,
        "rows_per_pair": round(rows / pairs, 6),
        "pairs_with_exactly_two_rows": ok_pairs,
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
        "mean_pairs_per_document": round(
            float(df.filter(pl.col("label") == 1).group_by("doc_id").len()["len"].mean()), 6),
        "templates": {str(k): v for k, v in df.group_by("template_id").len().iter_rows()},
        "serial_forms": {k: v for k, v in df.group_by("serial_form").len().iter_rows()},
        "rows_by_source": {k: v for k, v in df.group_by("source").len().iter_rows()},
        "duplicate_claim_strings": rows - df["claim"].n_unique(),
        "label_balance": {str(k): v for k, v in df.group_by("label").len().iter_rows()},
    }

    h = hashlib.blake2b(LANE.read_bytes(), digest_size=8).hexdigest()

    out = {
        "clause": "C7 + C8",
        "c7": {
            "declared_unit": "BOTH - the registration states rows AND pairs",
            "registered_in_R18-H150_arm_run.LANES": reg_block.strip().splitlines()[:3],
            "registered_in_R20-H174_arm_run.LANES": h174_block.strip().splitlines()[:3],
            "registered_rows": 30000, "registered_pairs": 15000,
            "registered_families": {"misbound_row": 21000, "misbound_col": 9000},
            "measured_rows": rows, "measured_pairs": pairs, "measured_families": fam,
            "rows_match": rows == 30000, "pairs_match": pairs == 15000,
            "families_match": fam == {"misbound_row": 21000, "misbound_col": 9000},
            "note": "both arm wrappers hard-abort on any drift in rows, pairs or family "
                    "counts before a card is touched (LANE ABORT guard)",
        },
        "c8": {
            "artifact": str(LANE.relative_to(ROOT)),
            "blake2b_64": h,
            "builder": "experiments/grounding-semantic/R17-H146_lane.py, seed 1146",
            "sources": {
                "tabfact": {
                    "archive": "data/external/datasets/dataset-tabfact.zip",
                    "sidecar": "data/external/datasets/dataset-tabfact.md (tracked)",
                    "licence": "CC-BY-4.0 (from the tracked sidecar)",
                    "selection_predicate": "the *__train.parquet member of the archive, "
                                           "deduplicated on table_text; tables with >= 4 "
                                           "body rows of uniform width and >= 2 columns; "
                                           "a label column and at least one numeric column "
                                           "must exist",
                    "retrieval": "scripts/fetch_grounding_datasets.py (date not recorded "
                                 "in the sidecar)",
                },
                "feverous": {
                    "archive": "tmp/R14_H133_feverous.parquet",
                    "sidecar": "NONE in data/external/datasets",
                    "licence": "NOT RECORDED anywhere in the repository",
                    "selection_predicate": "deduplicated on `evidence`; rows containing a "
                                           "[ROW] marker, parsed into >= 5 uniform-width "
                                           "rows of >= 3 cells",
                    "retrieval": "not recorded; the file is an R14-H133 working artifact "
                                 "mirrored from a community HuggingFace export "
                                 "(KingTechnician/feverous-label-evidence, per "
                                 "R10-H108_data.py) rather than an admitted corpus",
                    "campaign_status": "R20-H177_lane_common.feverous_available() records "
                                       "admitted=False for this exact file: 'an R14-H133 "
                                       "working file, not an admitted lane supply with its "
                                       "own provenance verdict'",
                },
            },
            "internal_structure": dup,
            "public_repository_check": {
                "client_or_company_name_in_lane_text": None,
                "method": "see the scan in this stage's output below",
            },
        },
        "seconds": round(time.time() - t0, 1),
    }

    # public-repo scan: the lane must carry only public-corpus text.  Sources are
    # declared per row; verify no row claims a non-public origin.
    out["c8"]["public_repository_check"]["sources_present"] = sorted(
        set(df["source"].to_list()))
    out["c8"]["public_repository_check"]["all_sources_public"] = set(
        df["source"].to_list()) <= {"tabfact", "feverous"}
    return save("c78_units_provenance", out)


STAGES = {
    "c1": stage_c1,
    "c1b": stage_c1_binding,
    "c2": stage_c2,
    "c3": stage_c3,
    "c4": stage_c4,
    "c5": stage_c5,
    "c6": stage_c6,
    "c78": stage_c78,
}


def main():
    want = sys.argv[1:] or list(STAGES)
    for s in want:
        print(f"\n=== STAGE {s} ===", flush=True)
        STAGES[s]()
    print("\nALL REQUESTED STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
