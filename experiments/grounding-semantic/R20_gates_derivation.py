"""R20 free kill-gates 1-3 - the derivation candidates (H-B / H-C / H-D).

Recipes verbatim from `docs/experiments/briefs/R20-fanout-derivation-hypotheses.md`
(H-B kill-gate line 65, H-C line 77, H-D line 89) and the canonical log's
"LICENSED FREE KILL-GATE BATCH" block.

  gate 1 (H-B)  reclassify the 50 H157 error items + the R19-H161 L2 records into
                depth-1 compare/sign/direction (decidable by ORDERING values both
                present in evidence) vs multi-op arithmetic.
                PASS if >= 20% of derivation-class rank-loss mass is
                compare-decidable (>= 0.056 absolute of finqa rank loss), AND
                the constructibility census yields >= 25,000 buildable compare
                pairs (both operands verbatim) over EDGAR-admitted + TabFact
  gate 2 (H-C)  classify the H157 FP records by their `unsupported_explanation`.
                PASS if >= 30% of FP rank-loss mass traces to mislabeled
                operand / role / sign / period rather than wrong computation or
                question-relativity, AND EDGAR carries >= 20k sentences with
                value + role word + period
  gate 3 (H-D)  mmBERT tokenizer fragmentation - tokens per numeral on the H157
                sinking sentences' numerals vs digit-length-matched numerals from
                non-error items. PASS at >= 1 sd excess

Zero GPU, zero training. Writes R20_gate_{1,2,3}.json.

Run:  uv run python experiments/grounding-semantic/R20_gates_derivation.py
"""

import json
import pathlib
import re
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
AUTOPSY = HERE / "R18-H157_finqa_autopsy.json"
EDGAR = HERE / "R18-H150_edgar_admitted.parquet"
L2 = HERE / "R19-H161_L2_items.parquet"
TOKENIZER_DIR = ROOT / "models" / "R18-H150-arm-draw1"

NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

COMPARE_WORDS = (
    "greater", "less", "higher", "lower", "more than", "fewer", "larger", "smaller",
    "exceed", "exceeds", "exceeded", "above", "below", "than", "versus", " vs ",
    "compared to", "highest", "lowest", "largest", "smallest", "most", "least",
)
DIRECTION_WORDS = (
    "increase", "increased", "increases", "decrease", "decreased", "decreases",
    "rose", "fell", "fallen", "decline", "declined", "grew", "growth", "gain",
    "gained", "up from", "down from", "net change", "change in", "improvement",
    "improved", "worsened", "reduction", "reduced",
)
ROLE_WORDS = (
    "revenue", "revenues", "income", "expense", "expenses", "cost", "costs", "sales",
    "assets", "liabilities", "liability", "debt", "cash", "margin", "shares", "share",
    "repurchase", "dividend", "dividends", "interest", "tax", "taxes", "earnings",
    "loss", "losses", "profit", "equity", "inventory", "inventories", "backlog",
    "capital", "compensation", "goodwill", "impairment", "obligations", "payments",
    "reserves", "balance", "payable", "receivable", "amortization", "depreciation",
)
PERIOD_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(?:fiscal|calendar)\s+(?:year|quarter)|"
    r"\b(?:first|second|third|fourth)\s+quarter\b|\bq[1-4]\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b", re.I)

# --------------------------------------------------------------------------- #
# gate 1 / gate 2 - the hand audit
#
# The autopsy itself carries 15 manual overrides; these two reclassifications are
# the same instrument applied to the same 50 items. Every assignment below is a
# read of the record's own `unsupported_explanation` (FPs) or its sinking
# sentence + absent-numeral signals (FNs), and each carries its one-line reason.
# A rule-based proposal is computed alongside and its disagreement rate reported,
# so the hand pass is auditable rather than asserted.
# --------------------------------------------------------------------------- #

# Gate 1: the error is resolvable by ORDERING / SIGN / DIRECTION of two values
# both present in the evidence (depth 1, no computation).
G1_COMPARE_DECIDABLE = {
    100: "pure ordering claim - '$291M environmental > $180M asbestos', 0 absent numerals",
    168: "direction word - frames a computed decrease (-2.8%) as an increase",
    189: "sign - '-$305 million' claimed as '$305 million'",
    215: "direction - documents state the operating ratio INCREASED in 2011",
    31: "direction - equity is stated to have decreased; it increased by $2,751M",
}
G1_MULTI_OP = {
    24: "ratio 135/700 - division", 30: "increase-by magnitude needs a subtraction",
    37: "projected rate - growth extrapolation", 41: "product of two absent figures",
    67: "question-relativity - portion of a total the response never forms",
    70: "327 - 184 = 143 subtraction", 93: "sum of two share tranches",
    101: "net change magnitude", 112: "150 x 23.44 product",
    114: "operand/role misbind (repurchase used as decrease), not an ordering error",
    119: "percentage effect of a hedge", 120: "growth rate",
    147: "net change magnitude", 148: "five-year percent change",
    157: "sum from wrong operands - operand selection, not ordering",
    162: "tax rate from pretax/after-tax", 167: "sum incl. interest and penalties",
    190: "455 / 7 division", 191: "three-term sum", 198: "wrong change magnitude "
    "($85M vs $168M); the direction is correct so ordering does not decide it",
    199: "1,224 - 1,214 subtraction", 205: "multi-term sum", 211: "two-year sum",
    213: "percent change (5829-5735)/5735",
}

# Gate 2: what the FP's own explanation attributes the falsity to.
G2_FP_CLASS = {
    36: ("misbind_period", "correct arithmetic applied to the wrong year (2012 vs 2011)"),
    229: ("misbind_period", "values bound to 2009; the documents carry only earlier years"),
    114: ("misbind_role", "the $74.9M repurchase amount used as the decrease in notes"),
    71: ("misbind_role", "per-share $78.29 claimed as the total"),
    157: ("misbind_operand", "the sum is formed from the wrong operands"),
    189: ("misbind_sign", "signage - '-$305M' stated as '$305M'"),
    31: ("misbind_sign", "trend stated as a decrease; equity increased"),
    168: ("misbind_sign", "a computed decrease reframed as an increase"),
    215: ("misbind_sign", "assumes an improvement the documents contradict"),
    67: ("question_relativity", "the figure is correct; the question asked for a portion"),
    41: ("wrong_computation", "the cited figures are absent from the documents"),
    198: ("wrong_computation", "decrease magnitude $85M vs the documents' $168M"),
    200: ("wrong_computation", "$5.2bn vs the documents' $5bn, plus an invalid combination"),
    242: ("wrong_computation", "57,800 sq ft vs the documents' 57,100"),
    43: ("scale_unit", "'in thousands' notation misread - scale, not role/sign/period"),
    85: ("other_absence", "asserts data absent that the documents carry"),
    116: ("other_speculation", "an extrapolated prediction, unsupported by construction"),
}
G2_MISBIND = ("misbind_period", "misbind_role", "misbind_operand", "misbind_sign")


def rank_loss(y, s):
    """R18-H157_finqa_autopsy.rank_loss, byte-identical."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    per = np.zeros(len(y))
    for i in pos:
        per[i] = np.sum(s[neg] > s[i]) + 0.5 * np.sum(s[neg] == s[i])
    for j in neg:
        per[j] = np.sum(s[pos] < s[j]) + 0.5 * np.sum(s[pos] == s[j])
    total = per[pos].sum()
    return per / max(total, 1e-9) / 2.0


def load_rankloss():
    """Mean per-item rank loss over the two draws, the autopsy's own convention."""
    rls = []
    for tag in ("draw1", "draw2"):
        d = pl.read_parquet(HERE / f"R18-H157_finqa_items_{tag}.parquet")
        it = (d.group_by("item").agg(pl.col("response_score").first(),
                                     pl.col("label").first()).sort("item"))
        rls.append(rank_loss(it["label"].to_numpy().astype(int),
                             it["response_score"].to_numpy()))
    return (rls[0] + rls[1]) / 2.0, rls


def rule_compare_proposal(rec):
    """Deterministic proposal - a compare/direction token in the sinking sentence AND
    either no absent numeral or an explanation naming sign/direction/trend."""
    s = rec["sinking_sentence"].lower()
    e = (rec["signals"].get("unsupported_explanation") or "").lower()
    has_cmp = any(w in s for w in COMPARE_WORDS + DIRECTION_WORDS)
    sign_expl = any(w in e for w in ("signage", "sign ", "trend", "incorrectly frames",
                                     "contradicting", "increased", "decrease"))
    return bool(has_cmp and (rec["signals"]["n_absent_verbatim"] == 0 or sign_expl))


def gate1(items, rl_mean):
    deriv = {}
    for it in items:
        if it["final_class"] == "derivation_arithmetic":
            deriv.setdefault(it["item"], it)
    mass = {i: float(rl_mean[i]) for i in deriv}
    total = sum(mass.values())
    dec = {i: m for i, m in mass.items() if i in G1_COMPARE_DECIDABLE}
    unassigned = [i for i in deriv if i not in G1_COMPARE_DECIDABLE and i not in G1_MULTI_OP]
    share = sum(dec.values()) / total if total else 0.0

    # rule-based cross-check
    agree = sum(1 for i, it in deriv.items()
                if rule_compare_proposal(it) == (i in G1_COMPARE_DECIDABLE))
    rule_mass = sum(m for i, m in mass.items() if rule_compare_proposal(deriv[i]))

    # L2 corroboration (record counts; no rank loss is banked for tatqa/delucionqa)
    l2 = pl.read_parquet(L2)
    l2d = l2.filter(pl.col("final_class") == "derivation_arithmetic")
    l2_flag = [
        any(w in s.lower() for w in COMPARE_WORDS + DIRECTION_WORDS)
        for s in l2d["sinking_sentence"].to_list()
    ]
    l2_by_sub = {}
    for sub in sorted(set(l2d["subset"].to_list())):
        m = np.array([s == sub for s in l2d["subset"].to_list()])
        f = np.array(l2_flag)[m]
        l2_by_sub[sub] = {"n_derivation_records": int(m.sum()),
                          "carrying_compare_or_direction_token": int(f.sum()),
                          "share": round(float(f.mean()) if m.sum() else 0.0, 4)}
    return {
        "gate": "R20 gate 1 (H-B) - depth-1 compare/sign/direction share of derivation "
                "rank-loss mass + compare-pair constructibility census",
        "recipe_summary": ("reclassify every derivation_arithmetic H157 error item into "
                           "depth-1 compare/sign/direction (the error is resolvable by "
                           "ORDERING/SIGN/DIRECTION of two values both present in evidence) "
                           "vs multi-op arithmetic; mass = the autopsy's own per-item "
                           "rank loss averaged over the two draws"),
        "n_derivation_items": len(deriv),
        "derivation_rank_loss_mass": round(total, 4),
        "banked_reference_mass": 0.2797,
        "compare_decidable_items": {str(i): G1_COMPARE_DECIDABLE[i] for i in sorted(dec)},
        "compare_decidable_mass": round(sum(dec.values()), 4),
        "compare_decidable_share": round(share, 4),
        "absolute_of_finqa_rank_loss": round(sum(dec.values()), 4),
        "unassigned_items": unassigned,
        "rule_crosscheck": {
            "agreement_with_hand_audit": f"{agree}/{len(deriv)}",
            "rule_based_share": round(rule_mass / total if total else 0.0, 4),
        },
        "l2_corroboration": l2_by_sub,
        "threshold": "PASS if compare-decidable share >= 0.20 of derivation-class rank-loss "
                     "mass (>= 0.056 absolute of finqa rank loss)",
        "mass_verdict": "PASS" if share >= 0.20 else "FAIL",
    }


def gate2(items, rl_mean, rl_draws):
    fps = {}
    for it in items:
        if it["error_type"] == "fp":
            fps.setdefault(it["item"], it)
    mass = {i: float(rl_mean[i]) for i in fps}
    fp_total = sum(mass.values())
    # the whole negative-side rank-loss mass (all 20 negatives, errors or not)
    d = pl.read_parquet(HERE / "R18-H157_finqa_items_draw1.parquet")
    lab = (d.group_by("item").agg(pl.col("label").first()).sort("item")["label"]
           .to_numpy().astype(int))
    neg_total = float(rl_mean[lab == 0].sum())

    by_class = {}
    for i, m in mass.items():
        c = G2_FP_CLASS[i][0]
        by_class[c] = by_class.get(c, 0.0) + m
    misbind = sum(v for k, v in by_class.items() if k in G2_MISBIND)
    share_of_fp_records = misbind / fp_total if fp_total else 0.0
    share_of_neg_mass = misbind / neg_total if neg_total else 0.0
    return {
        "gate": "R20 gate 2 (H-C) - share of H157 FP rank-loss mass traceable to "
                "operand/role/sign/period misbinding",
        "recipe_summary": ("classify every H157 false-positive record by its own "
                           "`unsupported_explanation` into misbind_{operand,role,sign,"
                           "period} vs wrong_computation vs question_relativity vs other; "
                           "mass = per-item rank loss averaged over the two draws"),
        "n_fp_items": len(fps),
        "fp_record_rank_loss_mass": round(fp_total, 4),
        "negative_side_rank_loss_mass": round(neg_total, 4),
        "mass_by_class": {k: round(v, 4) for k, v in sorted(by_class.items())},
        "per_item": {str(i): {"class": G2_FP_CLASS[i][0], "reason": G2_FP_CLASS[i][1],
                              "rank_loss": round(mass[i], 4)} for i in sorted(fps)},
        "misbind_mass": round(misbind, 4),
        "misbind_share_of_fp_record_mass": round(share_of_fp_records, 4),
        "misbind_share_of_all_negative_mass": round(share_of_neg_mass, 4),
        "threshold": "PASS if >= 0.30 of FP rank-loss mass is operand/role/sign/period "
                     "misbinding (primary denominator = the classified FP records' mass; "
                     "the all-negatives denominator is carried as the strict reading)",
        "mass_verdict": "PASS" if share_of_fp_records >= 0.30 else "FAIL",
        "mass_verdict_strict_denominator": "PASS" if share_of_neg_mass >= 0.30 else "FAIL",
    }


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def edgar_censuses():
    """One pass over the 34,014 admitted EDGAR chunks feeding both censuses."""
    d = pl.read_parquet(EDGAR)
    cmp_sents = 0
    role_period_sents = 0
    n_sent = 0
    cmp_words = COMPARE_WORDS + DIRECTION_WORDS
    for chunk in d["chunk"].to_list():
        for s in sentences(chunk):
            n_sent += 1
            low = s.lower()
            nums = NUM_FREE.findall(s)
            if len(nums) >= 2 and any(w in low for w in cmp_words):
                cmp_sents += 1
            if nums and any(w in low for w in ROLE_WORDS) and PERIOD_RE.search(s):
                role_period_sents += 1
    return {"n_chunks": d.height, "n_sentences": n_sent,
            "compare_buildable_sentences": cmp_sents,
            "value_role_period_sentences": role_period_sents}


def tabfact_compare_census():
    """Buildable ordering pairs over TabFact: a numeric column with >= 2 distinct
    values and non-numeric row keys, both operands printed verbatim in the
    serialised table. Reported uncapped, capped at 2 per (table, column), and
    capped at 2 per table (the banked `build_compare` convention)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("c", HERE / "R15_gate_common.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    import io
    import zipfile

    z = zipfile.ZipFile(ROOT / "data" / "external" / "datasets" / "dataset-tabfact.zip")
    frames = [pl.read_parquet(io.BytesIO(z.read(n))) for n in z.namelist()
              if n.endswith(".parquet")]
    all_t = pl.concat(frames, how="vertical_relaxed").unique(subset=["table_id"],
                                                             keep="first")
    train_ids = set(pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))["table_id"]
        .to_list())

    out = {"all_splits": {}, "held_out_only": {}}
    for scope, tbl in (("all_splits", all_t),
                       ("held_out_only", all_t.filter(~pl.col("table_id")
                                                      .is_in(list(train_ids))))):
        unc = cap_col = cap_tab = 0
        for cap, txt in zip(tbl["table_caption"].to_list(), tbl["table_text"].to_list(),
                            strict=True):
            hdr, body = C.parse(txt)
            if hdr is None or len(body) < 3:
                continue
            ev = C.serialize(cap, hdr, body)
            per_table = 0
            for ci in range(1, len(hdr)):
                vals = [(ri, C.as_num(r[ci])) for ri, r in enumerate(body)]
                vals = [(ri, v) for ri, v in vals if v is not None]
                # row keys must be non-numeric labels, values distinct and printed
                vals = [(ri, v) for ri, v in vals
                        if body[ri][0].strip() and C.as_num(body[ri][0]) is None
                        and C.fmt(v) in ev]
                uniq = {v for _, v in vals}
                if len(vals) < 2 or len(uniq) < 2:
                    continue
                k = len(vals)
                unc += k * (k - 1) // 2
                cap_col += min(2, k * (k - 1) // 2)
                take = min(2 - per_table, k * (k - 1) // 2)
                if take > 0:
                    cap_tab += take
                    per_table += take
        out[scope] = {"uncapped_pairs": unc, "cap2_per_table_column": cap_col,
                      "cap2_per_table": cap_tab, "n_tables": tbl.height}
    return out


def gate3(items):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    err_items = {it["item"] for it in items}
    err_nums, err_loadbearing = [], []
    for it in items:
        err_nums += NUM_FREE.findall(it["sinking_sentence"])
        err_loadbearing += list(it["signals"].get("absent_numbers") or [])
        err_loadbearing += [str(c["a"]) for c in it["signals"]["derivation_candidates"]]
        err_loadbearing += [str(c["b"]) for c in it["signals"]["derivation_candidates"]]

    d = pl.read_parquet(HERE / "R18-H157_finqa_items.parquet")
    ok = d.filter(~pl.col("item").is_in(list(err_items)) & pl.col("is_sinking"))
    ctrl_nums = []
    for s in ok["sentence"].to_list():
        ctrl_nums += NUM_FREE.findall(s)

    def frag(xs):
        return np.array([len(tok(x, add_special_tokens=False)["input_ids"]) for x in xs],
                        dtype=float)

    def digits(x):
        return sum(ch.isdigit() for ch in x)

    def matched(err, ctrl, seed=20260816):
        """Digit-length-matched control sample: for each error numeral draw one
        control numeral with the same digit count (with replacement)."""
        rng = np.random.default_rng(seed)
        pool = {}
        for c in ctrl:
            pool.setdefault(digits(c), []).append(c)
        out, unmatched = [], 0
        for e in err:
            p = pool.get(digits(e))
            if not p:
                unmatched += 1
                continue
            out.append(p[int(rng.integers(0, len(p)))])
        return out, unmatched

    res = {}
    for name, err in (("sinking_sentence_numerals", err_nums),
                      ("load_bearing_numerals", err_loadbearing)):
        err = [e for e in err if any(ch.isdigit() for ch in e)]
        ctrl_m, unmatched = matched(err, ctrl_nums)
        fe, fc = frag(err), frag(ctrl_m)
        sd = float(fc.std(ddof=1)) if len(fc) > 1 else 0.0
        excess = (float(fe.mean()) - float(fc.mean())) / sd if sd else 0.0
        res[name] = {
            "n_error_numerals": len(err),
            "n_matched_control": len(ctrl_m),
            "unmatched_error_numerals": unmatched,
            "mean_tokens_per_numeral_error": round(float(fe.mean()), 4),
            "mean_tokens_per_numeral_control": round(float(fc.mean()), 4),
            "control_sd": round(sd, 4),
            "excess_in_control_sd": round(excess, 4),
            "verdict": "PASS" if excess >= 1.0 else "FAIL",
        }
    # unmatched raw contrast, for context only
    fe_all, fc_all = frag([e for e in err_nums if any(c.isdigit() for c in e)]), frag(ctrl_nums)
    res["unmatched_raw_contrast"] = {
        "mean_error": round(float(fe_all.mean()), 4),
        "mean_control": round(float(fc_all.mean()), 4),
        "control_sd": round(float(fc_all.std(ddof=1)), 4),
        "n_control": len(fc_all),
    }
    return {
        "gate": "R20 gate 3 (H-D) - mmBERT tokenizer fragmentation of error numerals",
        "recipe_summary": ("tokenise the H157 sinking sentences' numerals with the "
                           "flagship checkpoint's own mmBERT tokenizer and compare tokens "
                           "per numeral against digit-length-matched numerals drawn from "
                           "the sinking sentences of items neither draw erred on"),
        "tokenizer": str(TOKENIZER_DIR),
        "results": res,
        "threshold": "PASS if error numerals are more fragmented by >= 1 control sd",
        "verdict": res["load_bearing_numerals"]["verdict"],
        "verdict_basis": "load_bearing_numerals (the absent claim values plus the "
                         "derivation operands); the whole-sentence numeral reading is "
                         "carried alongside",
    }


def main():
    t0 = time.time()
    A = json.loads(AUTOPSY.read_text())
    items = A["error_items"]
    rl_mean, rl_draws = load_rankloss()
    dset = {i["item"] for i in items if i["final_class"] == "derivation_arithmetic"}
    print(f"rank-loss reproduced: derivation mass check "
          f"{sum(float(rl_mean[i]) for i in dset):.4f} (banked 0.2797)", flush=True)

    g1 = gate1(items, rl_mean)
    print(f"gate1 mass share {g1['compare_decidable_share']} -> {g1['mass_verdict']}",
          flush=True)
    g2 = gate2(items, rl_mean, rl_draws)
    print(f"gate2 misbind share {g2['misbind_share_of_fp_record_mass']} -> "
          f"{g2['mass_verdict']}", flush=True)

    print("edgar censuses ...", flush=True)
    ed = edgar_censuses()
    print(f"  edgar {ed}", flush=True)
    print("tabfact compare census ...", flush=True)
    tf = tabfact_compare_census()
    print(f"  tabfact {tf}", flush=True)

    cens_total = (ed["compare_buildable_sentences"]
                  + tf["all_splits"]["cap2_per_table"])
    g1["constructibility_census"] = {
        "edgar_admitted": ed,
        "tabfact": tf,
        "combined_compare_pairs_conservative": cens_total,
        "combination": ("EDGAR sentences carrying >= 2 numerals and a compare/direction "
                        "token (one relation-flip pair each) + TabFact pairs capped at 2 "
                        "per table (the banked build_compare convention)"),
        "threshold": "PASS if >= 25,000 buildable compare pairs",
        "verdict": "PASS" if cens_total >= 25000 else "FAIL",
    }
    g1["verdict"] = ("PASS" if g1["mass_verdict"] == "PASS"
                     and g1["constructibility_census"]["verdict"] == "PASS" else "FAIL")
    g2["edgar_census"] = {
        "value_role_period_sentences": ed["value_role_period_sentences"],
        "n_sentences_scanned": ed["n_sentences"],
        "threshold": "PASS if >= 20,000 sentences carry value + role word + period",
        "verdict": "PASS" if ed["value_role_period_sentences"] >= 20000 else "FAIL",
    }
    g2["verdict"] = ("PASS" if g2["mass_verdict"] == "PASS"
                     and g2["edgar_census"]["verdict"] == "PASS" else "FAIL")

    print("gate3 tokenizer fragmentation ...", flush=True)
    g3 = gate3(items)
    print(f"  gate3 {g3['results']['load_bearing_numerals']} -> {g3['verdict']}", flush=True)

    for name, payload in (("1", g1), ("2", g2), ("3", g3)):
        payload["timestamp"] = time.strftime("%F %T")
        (HERE / f"R20_gate_{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
