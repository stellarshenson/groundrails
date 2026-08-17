"""C1 - label commensurability for the `quant_scale_unit` lane.

CPU ONLY.  The contract's mandatory test: claim-to-evidence containment on the
NEGATIVE leg against the POSITIVE leg, both distributions reported.

Three instruments, all applied to both legs of all 2,770 pairs:

  I1  the R20-H175b precedent instrument - the one that produced the banked
      qrel_contrast numbers (containment 0.9129 both legs, 66.4% of negatives
      fully attested, 72.3% at >= 0.90).  Tokens `[^\\W\\d_]+|\\d+` unicode,
      lowercased, NO stopword removal; containment = |claim n chunk| / |claim|.
      Lifted from `R20-H175b_qlane.py` (`tok`, `containment`)
  I2  the R19-H161 frozen content instrument - tokens `[a-z0-9]+` lowercased
      minus the frozen stopword list.  Lifted from `R19-H161_dump.py`
  I3  UNIT-RESOLVED - I1 after mapping the claim's spelled-out unit phrase onto
      the abbreviation vocabulary the EVIDENCE uses (the lane's own banked
      `UNITS` table).  This is the only instrument that can read the token the
      lane's label actually turns on

Also measured, from the lane's own banked vocabulary:
  - the H148 literal-presence rate per leg (re-derived, not cited)
  - whether the claimed unit is attested in the chunk for the cited column
  - the in-chunk distractor stratum

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_c1.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_c1.json"

# --- I1: R20-H175b_qlane.py verbatim -----------------------------------------
_WORD_H175B = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tok_h175b(text):
    return _WORD_H175B.findall(text.lower())


def containment_h175b(a, b):
    A = set(tok_h175b(a))
    return len(A & set(tok_h175b(b))) / len(A) if A else 0.0


# --- I2: R19-H161_dump.py verbatim -------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were "
    "be been being it its this that these those".split()
)


def content_set(text):
    return frozenset(t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS)


def containment_h161(a, b):
    A = content_set(a)
    return len(A & content_set(b)) / len(A) if A else 0.0


def dist(vals):
    v = np.asarray(vals, dtype=float)
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 6),
        "median": round(float(np.median(v)), 6),
        "p10": round(float(np.percentile(v, 10)), 6),
        "p25": round(float(np.percentile(v, 25)), 6),
        "p75": round(float(np.percentile(v, 75)), 6),
        "p90": round(float(np.percentile(v, 90)), 6),
        "min": round(float(v.min()), 6),
        "max": round(float(v.max()), 6),
        "rate_eq_1_00": round(float((v >= 1.0 - 1e-12).mean()), 6),
        "rate_ge_0_95": round(float((v >= 0.95).mean()), 6),
        "rate_ge_0_90": round(float((v >= 0.90).mean()), 6),
    }


def leg_block(name, fn, claims, chunks, labels):
    pos = [fn(c, k) for c, k, y in zip(claims, chunks, labels) if y == 1]
    neg = [fn(c, k) for c, k, y in zip(claims, chunks, labels) if y == 0]
    p, n = dist(pos), dist(neg)
    gap90 = n["rate_ge_0_90"] - p["rate_ge_0_90"]
    return {
        "instrument": name,
        "positive_leg": p,
        "negative_leg": n,
        "mean_gap_neg_minus_pos": round(n["mean"] - p["mean"], 6),
        "rate_ge_0_90_gap_neg_minus_pos": round(gap90, 6),
        "abs_rate_ge_0_90_gap": round(abs(gap90), 6),
        "rate_eq_1_00_gap_neg_minus_pos": round(
            n["rate_eq_1_00"] - p["rate_eq_1_00"], 6
        ),
        "contract_bar_triggered_literal": bool(abs(gap90) <= 0.10),
    }, pos, neg


def main():
    spec = importlib.util.spec_from_file_location(
        "h150lane", SEM / "R18-H150_scaleunit_lane.py"
    )
    lane_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lane_mod)
    UNITS = lane_mod.UNITS
    PHRASE = lane_mod.PHRASE
    SURFACE = lane_mod.SURFACE
    PHRASE_TOKENS = lane_mod.PHRASE_TOKENS

    df = pl.read_parquet(LANE)
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    labels = df["label"].to_list()
    cited = df["cited_unit"].to_list()
    correct = df["correct_unit"].to_list()
    wrong = df["wrong_unit"].to_list()

    res = {
        "member": "quant_scale_unit",
        "artifact": str(LANE),
        "rows": len(df),
        "pairs": int(df["pair_id"].n_unique()),
        "label_balance": {
            str(k): int(v)
            for k, v in zip(
                df["label"].value_counts()["label"].to_list(),
                df["label"].value_counts()["count"].to_list(),
            )
        },
    }

    # --- the three containment instruments -----------------------------------
    b1, pos1, neg1 = leg_block("I1_R20-H175b_precedent", containment_h175b,
                               claims, chunks, labels)
    b2, _, _ = leg_block("I2_R19-H161_frozen_content", containment_h161,
                         claims, chunks, labels)

    # I3 - unit-resolved: rewrite the claim's spelled-out unit phrase to the
    # abbreviation key, so the instrument can read the token the label turns on.
    def unit_resolved(claim, unit_key):
        phrase = PHRASE[unit_key]
        return re.sub(re.escape(phrase), unit_key, claim, flags=re.IGNORECASE)

    claims_r = [unit_resolved(c, u) for c, u in zip(claims, cited)]
    b3, pos3, neg3 = leg_block("I3_unit_resolved", containment_h175b,
                               claims_r, chunks, labels)
    res["containment"] = [b1, b2, b3]

    # --- per-pair identity: is the pair's containment identical on both legs? -
    per_pair = {}
    for pid, y, c, k in zip(df["pair_id"].to_list(), labels, claims, chunks):
        per_pair.setdefault(pid, {})[y] = containment_h175b(c, k)
    diffs = np.array([abs(v[1] - v[0]) for v in per_pair.values() if 0 in v and 1 in v])
    res["within_pair_containment_identity"] = {
        "pairs": int(diffs.size),
        "mean_abs_diff": round(float(diffs.mean()), 8),
        "max_abs_diff": round(float(diffs.max()), 8),
        "pairs_exactly_equal": int((diffs < 1e-12).sum()),
        "share_exactly_equal": round(float((diffs < 1e-12).mean()), 6),
        "instrument": "I1_R20-H175b_precedent",
    }

    # --- H148 literal presence, re-derived -----------------------------------
    def literal_present(claim_unit, twin_unit, chunk):
        """The lane's own definition: the claim's unit phrase, or any content
        token distinguishing it from its twin's phrase, readable in the chunk."""
        low = chunk.lower()
        if PHRASE[claim_unit].lower() in low:
            return True
        distinguishing = PHRASE_TOKENS[claim_unit] - PHRASE_TOKENS[twin_unit]
        toks = set(TOKEN_RE.findall(low))
        return bool(distinguishing & toks)

    lit = []
    for y, cu, co, wr, k in zip(labels, cited, correct, wrong, chunks):
        twin = wr if y == 1 else co
        lit.append(literal_present(cu, twin, k))
    lit = np.array(lit)
    lab = np.array(labels)
    res["h148_literal_presence"] = {
        "positive_leg_rate": round(float(lit[lab == 1].mean()), 6),
        "negative_leg_rate": round(float(lit[lab == 0].mean()), 6),
        "banked_verify_claim": {"positive_leg_rate": 0.0, "negative_leg_rate": 0.0},
        "definition": "claim unit phrase, or any content token distinguishing it "
                      "from its twin's phrase, readable in the chunk",
    }

    # --- unit ATTESTATION in the evidence surface vocabulary ------------------
    # The predicate the label actually encodes: does the evidence state the
    # claimed unit (in ANY surface form the vocabulary allows) for this table?
    surf_claimed, surf_twin = [], []
    for y, cu, co, wr, k in zip(labels, cited, correct, wrong, chunks):
        twin = wr if y == 1 else co
        surf_claimed.append(bool(SURFACE[cu].search(k)))
        surf_twin.append(bool(SURFACE[twin].search(k)))
    surf_claimed = np.array(surf_claimed)
    surf_twin = np.array(surf_twin)
    res["unit_surface_attestation"] = {
        "definition": "the lane's banked SURFACE regex for the CLAIMED unit "
                      "matches anywhere in the chunk (abbreviation OR spelled out)",
        "positive_leg_claimed_unit_attested": round(
            float(surf_claimed[lab == 1].mean()), 6),
        "negative_leg_claimed_unit_attested": round(
            float(surf_claimed[lab == 0].mean()), 6),
        "gap_neg_minus_pos": round(
            float(surf_claimed[lab == 0].mean() - surf_claimed[lab == 1].mean()), 6),
        "negative_leg_correct_unit_attested": round(
            float(surf_twin[lab == 0].mean()), 6),
        "auroc_note": "a presence test separating the legs is the H148 shortcut; "
                      "it is REPORTED here, not a bar",
    }

    # --- the in-chunk distractor stratum -------------------------------------
    dic = df["distractor_in_chunk"].to_numpy()
    strata = {}
    for flag in (True, False):
        m = dic == flag
        if not m.any():
            continue
        sub_c = [c for c, s in zip(claims, m) if s]
        sub_k = [k for k, s in zip(chunks, m) if s]
        sub_y = [y for y, s in zip(labels, m) if s]
        blk, _, _ = leg_block("I1_R20-H175b_precedent", containment_h175b,
                              sub_c, sub_k, sub_y)
        blk["rows"] = int(m.sum())
        strata[str(flag)] = blk
    res["distractor_in_chunk_strata"] = strata

    # --- verbatim value attestation (the positive's numeral) ------------------
    vals = df["cited_value"].to_list()
    verb = np.array([v in k for v, k in zip(vals, chunks)])
    res["cited_value_verbatim_in_chunk"] = {
        "positive_leg": round(float(verb[lab == 1].mean()), 6),
        "negative_leg": round(float(verb[lab == 0].mean()), 6),
        "note": "digits never move between legs by construction",
    }

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
