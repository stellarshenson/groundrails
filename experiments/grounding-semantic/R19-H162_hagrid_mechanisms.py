"""R19-H162 - HAGRID MECHANISM DISSECTION. ANALYSIS ONLY.

Executor M2 of the R19-H162 mechanism-dissection wave. hagrid is the arena's
attributed information-seeking QA subset: flagship 2-draw AUROC 0.6424, seed
sd 0.0001 over three draws of the identical recipe, yet +0.0650 under the
R19-H159 enriched mix. Nothing here trains, tunes or selects on arena
statistics; every number is read off the banked R19-H161 per-pair dump plus the
frozen R8-H77 gate sample.

Inputs (none written by this script):

    experiments/grounding-semantic/R19-H161_pairs_h150d1.parquet   flagship draw 1
    experiments/grounding-semantic/R19-H161_pairs_h150d2.parquet   flagship draw 2
    experiments/grounding-semantic/R19-H161_pairs_h159d1.parquet   enriched mix draw 1
    R8-H92_decomposed_arena.ARENA.load_subsets()["hagrid"]          claim / evidence text

The dump carries per-pair logits with full (item, sentence, document, window)
provenance, so every aggregation below is a re-read of the banked scores, never
a re-scoring. No GPU is used.

POSITIVE CONTROL: each checkpoint's hagrid AUROC recomputed from the dump must
match its banked windowed value (h150d1 0.6423, h150d2 0.6425, h159d1 0.7074)
to <= 1e-3, and the structural fingerprint (250 items, 537 sentences, 1,941
pairs) must match exactly.

Run:
    uv run python experiments/grounding-semantic/R19-H162_hagrid_mechanisms.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R19-H162_hagrid_mechanisms.json"

SUBSET = "hagrid"
CONTROL_TOL = 1e-3
FINGERPRINT = {"n": 250, "n_sent": 537, "n_pairs": 1941}
CHECKPOINTS = {
    "h150d1": {"file": "R19-H161_pairs_h150d1.parquet", "banked": 0.6423},
    "h150d2": {"file": "R19-H161_pairs_h150d2.parquet", "banked": 0.6425},
    "h159d1": {"file": "R19-H161_pairs_h159d1.parquet", "banked": 0.7074},
}

# Answer-frame inventory read off the subset itself; these are the wrappers a
# generated answer puts around the proposition it is asserting.
RE_URL = re.compile(r"https?://|www\.")
RE_CITEMARK = re.compile(r"\[\s*\d+(\s*,\s*\d+)*\s*\]")
RE_REFLINE = re.compile(r'^\s*["“].{5,}["”]\s*,|^\s*Available\s*:|^\s*Retrieved\s+from')
RE_FRAME = re.compile(
    r"^\s*(Based on|According to) the (given|provided) contexts?|"
    r"^\s*The (given|provided) contexts? (mentions|states)",
    re.IGNORECASE,
)
RE_YESNO = re.compile(r"^\s*(Yes|No)\b[,.]", re.IGNORECASE)
RE_ARTIFACT = re.compile(
    r"\(\s*Context\s*\)|\[\s*Context\s*\]|Available\s*:\s*http|"
    r"^\s*(Based on|According to) the given contexts?\s*,?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_FRAMEONLY = re.compile(
    r"^\s*(Based on|According to) the given contexts?\s*,?\s*$", re.IGNORECASE
)


# Measured once by scanning R10-H108_lane.public_train() and the two R19 lanes;
# recorded here rather than recomputed because the clean-mix scan takes minutes
# and the numbers are provenance, not analysis.
TRAINING_MIX_WRAPPER_CENSUS = {
    "clean_public_mix": {
        "rows": 685670,
        "polarity_opener": 408,
        "polarity_frac": 0.0006,
        "discourse_frame": 0,
        "discourse_frame_frac": 0.0,
    },
    "attributionbench_lane": {
        "rows": 16444,
        "polarity_opener": 85,
        "polarity_frac": 0.0052,
        "discourse_frame": 2,
        "discourse_frame_frac": 0.0001,
    },
    "fava_lane": {
        "rows": 30073,
        "polarity_opener": 2,
        "polarity_frac": 0.0001,
        "discourse_frame": 0,
        "discourse_frame_frac": 0.0,
    },
    "hagrid_gate_sample": {
        "sentences": 537,
        "polarity_opener": 22,
        "polarity_frac": 0.041,
        "discourse_frame": 22,
        "discourse_frame_frac": 0.041,
    },
}

# The named mechanisms. Prose is a judgement over the measurements above; every
# number quoted inside is computed in this file or in the matched-control block
# recorded under `frame_drop_control`.
MECHANISMS = [
    {
        "name": "vacuous_claim_reject",
        "definition": (
            "score a response that carries no verifiable proposition - a bare "
            "provenance frame, a stray '(Context )' marker, a bibliography line - as "
            "unsupported rather than as trivially supported"
        ),
        "evidence_it_is_a_bottleneck": (
            "the four responses that are the string 'Based on the given context ,' and "
            "nothing else are all labelled unsupported, all score POSITIVE (+1.41, "
            "+2.63, +2.75, +2.07 on draw 1) at token containment 0.000, and carry "
            "21.2% / 20.8% of hagrid's total misrank mass across the two flagship "
            "draws. Ranking them at the bottom lifts hagrid 0.6423 to 0.7182 and "
            "0.6425 to 0.7167 (+0.0760 / +0.0742, draw spread 0.0018) - a ceiling, not "
            "an expected lane gain. The enriched mix did NOT fix it: the four still "
            "score positive (0.161 to 1.114, down from 1.407 to 2.749) and still carry "
            "17.3% of the misrank mass"
        ),
        "probe_design": (
            "vacuous_reject: claims drawn from a fixed inventory of provenance frames, "
            "citation lines, bare URLs and empty markers against arbitrary passages, "
            "label 0, against matched bare contentful positives over the same "
            "passages; within-pair rank accuracy, chance 0.5. A short-but-contentful "
            "label-1 control is required so a model that learned 'short implies "
            "unsupported' scores at chance on it"
        ),
        "lane_candidate": (
            "rule-based, zero new data and zero new licence - label-0 pairs whose "
            "claim side is the same fixed hand-written inventory of contentless "
            "strings paired with evidence sampled from the existing mix, plus a "
            "short-but-contentful label-1 control at matched length"
        ),
        "contamination": "CLEAR - hand-written claim side, already-banked evidence side",
        "already_covered_by": None,
    },
    {
        "name": "source_select",
        "definition": (
            "bind a claim to the one retrieved passage that supports it and withhold "
            "credit when the best-matching passage is only topically adjacent"
        ),
        "evidence_it_is_a_bottleneck": (
            "with the four vacuous items excluded, AUROC by evidence-pool depth is "
            "0.8577 / 0.8129 at 1 passage (n=66, 9 negatives), 0.6890 / 0.6277 at 2-3 "
            "(n=107, 13 negatives) and 0.5096 / 0.6052 at 4-8 (n=73, 12 negatives) "
            "over the two flagship draws - a 1-versus-4+ gap of 0.348 and 0.208 "
            "against a per-stratum SE of ~0.09. The enriched-mix checkpoint is a live "
            "existence proof: it lifts the same 4-8 cell to 0.7090 (+0.1516 over the "
            "flagship two-draw mean) and leaves the 1-passage cell flat at 0.8187, and "
            "its k-truncation curve RISES with added passages (0.6557 at k=1 to 0.7090 "
            "at k=8) where the flagship's falls (0.6216 to 0.5096 on draw 1)"
        ),
        "probe_design": (
            "attr_pool: one claim with a known single supporting passage; the positive "
            "presents the gold passage plus k BM25 topical distractors filtered so "
            "none entails the claim, the negative presents k+1 distractors with the "
            "gold removed. Sweep k in {0,1,3,7}, chance 0.5 at every k; the registered "
            "read is the AUROC-versus-k slope, null zero"
        ),
        "lane_candidate": (
            "rule-based BM25-distractor generator on a document-disjoint split using "
            "the probe's own construction; MiniCheck (MIT, 14,356 rows over 6,155 "
            "documents, median chunk 922 chars) is the right source because its "
            "passages are already hagrid-sized, VitaminC (CC-BY-SA-3.0) supplies "
            "volume, PubHealth (MIT) is unsuitable as-is at a 3,731-char median"
        ),
        "contamination": (
            "CLEAR - MiniCheck and VitaminC hold GREEN R14-H136 8-gram Jaccard verdicts "
            "against the ten walled arena corpora; hagrid is never a source"
        ),
        "already_covered_by": (
            "NO - AttributionBench is the closest attribution supervision banked and "
            "packs its references into ONE evidence chunk per row (median 296 chars), "
            "so it never presents a choice among competing passages"
        ),
    },
    {
        "name": "overclaim_near_copy_reject",
        "definition": (
            "refuse credit to a claim that restates its passage almost verbatim yet "
            "asserts something the passage does not license"
        ),
        "evidence_it_is_a_bottleneck": (
            "excluding the four vacuous items, the ten worst-ranked unsupported items' "
            "sinking sentences carry token containment 0.783 / 0.847 against 0.723 / "
            "0.728 for supported items, max common n-gram 8.0 / 7.8 against 6.28 / "
            "6.19, numeral containment 1.000 / 0.833 against 0.830 / 0.831, and score "
            "+2.164 / +2.314 against +0.059 / -0.293. Correctly-rejected negatives sit "
            "at token containment 0.516 / 0.479 and n-gram 4.21 / 3.50 - rejection "
            "tracks surface mismatch, not unsupportedness. The enriched-mix contrast "
            "confirms it causally: deep-pool near-copy UNSUPPORTED pairs fall 1.193 "
            "logits (n=28) while deep-pool near-copy supported pairs rise 0.232 "
            "(n=110) and ordinary deep-pool supported pairs rise 0.700"
        ),
        "probe_design": (
            "nearcopy_overclaim: positives are near-verbatim restatements of a passage "
            "sentence, negatives are the SAME restatement with one licence-breaking "
            "edit that leaves surface overlap intact (scope widening, referent "
            "replacement, quantifier strengthening, antecedent deletion). Stratify by "
            "max common n-gram so the near-copy stratum (>= 8) reads separately; "
            "within-pair accuracy, chance 0.5"
        ),
        "lane_candidate": (
            "rule-based scope / quantifier / referent corruption over already-banked "
            "prose passages on the R17-H146 minimal-pair discipline; VitaminC "
            "(CC-BY-SA-3.0) already supplies the generic near-miss construction"
        ),
        "contamination": "CLEAR - corruptor runs over already-banked, already-gated evidence",
        "already_covered_by": (
            "PARTIALLY - VitaminC is 34.7% of the flagship's pairs and is built exactly "
            "this way, so a fourth near-miss lane needs an argument that the existing "
            "one does not saturate; FAVA (CC-BY-4.0, banked) targets the same skill and "
            "is the competing credit for the H159 hagrid gain"
        ),
    },
]

BUILD_FIRST = {
    "mechanism": "vacuous_claim_reject",
    "why": (
        "not the largest but the only one buildable and probeable with no new data, no "
        "licence question and no contamination surface, and the only one the enriched "
        "mix left untouched. source_select and overclaim_near_copy_reject are already "
        "partially installed by banked supply, so their next step is a probe read on "
        "the three banked checkpoints, not a training arm"
    ),
    "ranking": [
        (
            "source_select - largest measured prize (+0.1516 on the deep-pool stratum, "
            "demonstrated on a real checkpoint) but the highest build cost"
        ),
        (
            "overclaim_near_copy_reject - the substrate of source_select's gain; two "
            "banked corpora already target it, and it is what cost H159 its table "
            "subsets"
        ),
        (
            "vacuous_claim_reject - smallest prize (at most ~+0.008 of arena mean), "
            "cheapest build, unfixed by the enriched mix"
        ),
    ],
}

ISOLATION = {
    "question": (
        "the R19-H159 verdict credits AttributionBench with hagrid's +0.0650; the same "
        "arm's H1 lane credits FAVA's overlap-prior suppression"
    ),
    "signature_read": (
        "the gain is carried by suppression of high-overlap unsupported windows inside "
        "deep evidence pools (-1.193 logits on 28 deep-pool near-copy unsupported "
        "pairs) - the exact prediction registered for FAVA, and also consistent with "
        "AttributionBench's high-overlap relation-error negatives"
    ),
    "register_discriminator": (
        "INCONCLUSIVE - the largest supported-minus-unsupported separation sits at "
        "500-1,000-char windows (+0.475), hagrid's own passage size, not "
        "AttributionBench's 296-char median nor FAVA's 2,972"
    ),
    "cheapest_separating_measurement": (
        "score the three ALREADY-BANKED checkpoints on the three probes above; one "
        "scoring pass, ~0.2 GPU-h, zero training. A win on attr_pool is "
        "AttributionBench's attribution skill, a win on nearcopy_overclaim is FAVA's "
        "overlap-prior suppression, a win on vacuous_reject alone is neither"
    ),
    "two_arm_expectation": (
        "AttributionBench alone is +2.3% rows / +1.3% pairs on the flagship, a 0.9875 "
        "uniform de-weighting against H159's 0.8248, so the table collapse should not "
        "reproduce and the mean should land within noise of 0.71549; on hagrid expect "
        "less than +0.0650, plausibly +0.02 to +0.05. It answers 'is AttributionBench "
        "sufficient', not 'which skill moved'"
    ),
}

# Measured but NOT evidenced as bottlenecks - recorded so no lane is built on them.
NOT_EVIDENCED = {
    "answer_frame_strip_general": (
        "dropping the 22 discourse-frame sentences from the MIN appears to buy "
        "+0.0576 / +0.0570, but the gain is entirely the four contentless items: "
        "dropping the frame sentence from the other 18 frame-carrying responses moves "
        "hagrid -0.0164 / -0.0149, at the 95th and 97.5th percentile of a matched null "
        "that drops one RANDOM sentence from the same 18 items (null mean -0.019, sd "
        "0.002-0.003, 200 draws)"
    ),
    "yesno_polarity_deficit": (
        "the 22 polarity-opening items hold 9 of 38 negatives and read AUROC 0.4274 on "
        "draw 1 but 0.6838 on draw 2; unmeasurable at 9 negatives, kept as a probe arm"
    ),
    "single_passage_relation_binding": (
        "in the 1-passage stratum the over-credited negatives have LOW token "
        "containment (0.222 / 0.291) - they are the contentless items again; excluding "
        "them the stratum reads 0.8577 / 0.8129 with almost no residual"
    ),
    "inline_citation_markers": (
        "26 sentences carry '[N]' markers, sink their item 73.1% of the time and score "
        "-1.054 / -1.866; no item-level price isolated and no matched control run - "
        "unmeasured, not absent"
    ),
}

# Matched-control block for the frame-drop decomposition (scratch computation,
# recorded here as provenance; see the memo's "Measured but NOT evidenced" section).
FRAME_DROP_CONTROL = {
    "frame_items": 22,
    "of_which_frame_only": 4,
    "h150d1": {
        "base": 0.6423,
        "drop_all_frames": 0.6999,
        "leg_a_frame_only_4": 0.7182,
        "leg_b_other_18": 0.6259,
        "leg_b_matched_null_mean": -0.0190,
        "leg_b_matched_null_sd": 0.0017,
        "leg_b_observed": -0.0164,
        "leg_b_percentile": 0.950,
    },
    "h150d2": {
        "base": 0.6425,
        "drop_all_frames": 0.6995,
        "leg_a_frame_only_4": 0.7167,
        "leg_b_other_18": 0.6276,
        "leg_b_matched_null_mean": -0.0192,
        "leg_b_matched_null_sd": 0.0029,
        "leg_b_observed": -0.0149,
        "leg_b_percentile": 0.975,
    },
    "null_draws": 200,
    "seed": 1162,
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hanley_se(y, s):
    """AUROC with its Hanley-McNeil standard error."""
    y = np.asarray(y)
    s = np.asarray(s)
    a = float(roc_auc_score(y, s))
    n1 = int(y.sum())
    n0 = int((1 - y).sum())
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return a, float(np.sqrt(max(var, 0.0)))


def sentence_features():
    """Per-sentence surface features of the frozen hagrid gate sample."""
    h92 = _mod("h92", "R8-H92_decomposed_arena.py")
    claims, chunks, y = h92.ARENA.load_subsets()[SUBSET]
    y = np.asarray(y).astype(int)
    rows = []
    for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
        sents = h92.sentences(c)
        for si, s in enumerate(sents):
            rows.append(
                {
                    "item_id": i,
                    "label": int(y[i]),
                    "ndoc": len(ks),
                    "nsent": len(sents),
                    "sent_idx": si,
                    "n_word": len(s.split()),
                    "has_url": bool(RE_URL.search(s)),
                    "has_citemark": bool(RE_CITEMARK.search(s)),
                    "is_refline": bool(RE_REFLINE.search(s)),
                    "is_frame": bool(RE_FRAME.search(s)),
                    "is_yesno": bool(RE_YESNO.search(s)),
                    "text": s,
                }
            )
    items = pl.DataFrame(
        {
            "item_id": list(range(len(y))),
            "label": y.tolist(),
            "artifact": [bool(RE_ARTIFACT.search(c)) for c in claims],
            "frame_only": [bool(RE_FRAMEONLY.search(c.strip())) for c in claims],
        }
    )
    return pl.DataFrame(rows), items


WRAPPER = (
    pl.col("is_frame")
    | pl.col("is_yesno")
    | pl.col("has_url")
    | pl.col("is_refline")
    | (pl.col("n_word") <= 4)
)


def load_pairs(tag):
    p = HERE / CHECKPOINTS[tag]["file"]
    if not p.exists():
        return None
    return pl.read_parquet(p).filter(pl.col("subset") == SUBSET)


def item_table(d):
    return (
        d.group_by("item_id")
        .agg(
            pl.col("label").first(),
            pl.col("item_score").first(),
            (pl.col("doc_idx").max() + 1).alias("ndoc"),
            pl.col("n_sent_item").first().alias("nsent"),
            pl.len().alias("npair"),
        )
        .sort("item_id")
    )


def control(tag, d):
    it = item_table(d)
    auc = float(roc_auc_score(it["label"].to_numpy(), it["item_score"].to_numpy()))
    fp = {
        "n": it.height,
        "n_sent": int(d.filter(pl.col("is_argmax")).height),
        "n_pairs": d.height,
    }
    banked = CHECKPOINTS[tag]["banked"]
    return {
        "reproduced": round(auc, 6),
        "banked": banked,
        "abs_delta": round(abs(auc - banked), 6),
        "fingerprint_ok": fp == FINGERPRINT,
        "pass": bool(abs(auc - banked) <= CONTROL_TOL and fp == FINGERPRINT),
    }


def strata(it):
    out = {}
    specs = [
        ("ndoc_1", pl.col("ndoc") == 1),
        ("ndoc_2_3", (pl.col("ndoc") >= 2) & (pl.col("ndoc") <= 3)),
        ("ndoc_4plus", pl.col("ndoc") >= 4),
        ("npair_le4", pl.col("npair") <= 4),
        ("npair_5_15", (pl.col("npair") >= 5) & (pl.col("npair") <= 15)),
        ("npair_16plus", pl.col("npair") >= 16),
        ("nsent_1", pl.col("nsent") == 1),
        ("nsent_2_3", (pl.col("nsent") >= 2) & (pl.col("nsent") <= 3)),
        ("nsent_4plus", pl.col("nsent") >= 4),
    ]
    for name, expr in specs:
        s = it.filter(expr)
        y = s["label"].to_numpy()
        if len(set(y.tolist())) < 2:
            out[name] = {"n": s.height, "neg": int((1 - y).sum()), "auroc": None}
            continue
        a, se = hanley_se(y, s["item_score"].to_numpy())
        out[name] = {
            "n": s.height,
            "neg": int((1 - y).sum()),
            "auroc": round(a, 4),
            "se": round(se, 4),
        }
    return out


def kdoc_curve(d, it):
    """Item score recomputed with the evidence pool truncated to the first k
    retrieved documents. Items with fewer than k documents are unaffected."""
    y_all = it["label"].to_numpy()
    deep = it.filter(pl.col("ndoc") >= 4)["item_id"]
    out = {}
    for k in (1, 2, 3, 4, 6, 8):
        s = (
            d.filter(pl.col("doc_idx") < k)
            .group_by(["item_id", "sent_idx"])
            .agg(pl.col("logit").max().alias("ss"))
            .group_by("item_id")
            .agg(pl.col("ss").min().alias("isc"))
            .sort("item_id")
        )
        m = it.select("item_id", "label", "ndoc").join(s, on="item_id", how="left")
        deep_m = m.filter(pl.col("item_id").is_in(deep.implode()))
        out[str(k)] = {
            "auroc_all": round(float(roc_auc_score(y_all, m["isc"].to_numpy())), 4),
            "auroc_pool4plus": round(
                float(roc_auc_score(deep_m["label"].to_numpy(), deep_m["isc"].to_numpy())), 4
            ),
        }
    return out


def wrapper_block(d, sf):
    ss = d.filter(pl.col("is_argmax")).select("item_id", "sent_idx", "sent_score", "is_sinking")
    j = sf.join(ss, on=["item_id", "sent_idx"], how="inner")
    lab = j.group_by("item_id").agg(pl.col("label").first()).sort("item_id")

    def read(frame):
        s = frame.group_by("item_id").agg(pl.col("sent_score").min().alias("v")).sort("item_id")
        m = lab.join(s, on="item_id", how="left").with_columns(pl.col("v").fill_null(-99.0))
        a, se = hanley_se(m["label"].to_numpy(), m["v"].to_numpy())
        return {"auroc": round(a, 4), "se": round(se, 4)}

    types = {}
    for c in ("is_frame", "is_yesno", "has_citemark", "has_url", "is_refline"):
        s = j.filter(pl.col(c))
        if s.height:
            types[c] = {
                "n_sent": s.height,
                "mean_logit": round(float(s["sent_score"].mean()), 3),
                "frac_sinking": round(float(s["is_sinking"].mean()), 3),
            }
    plain = j.filter(~(pl.col("is_frame") | pl.col("is_yesno") | WRAPPER))
    types["plain"] = {
        "n_sent": plain.height,
        "mean_logit": round(float(plain["sent_score"].mean()), 3),
        "frac_sinking": round(float(plain["is_sinking"].mean()), 3),
    }

    yn_items = j.filter(pl.col("is_yesno"))["item_id"].unique()
    s_all = j.group_by("item_id").agg(
        pl.col("label").first(), pl.col("sent_score").min().alias("v")
    )
    yn = s_all.filter(pl.col("item_id").is_in(yn_items.implode()))
    non_yn = s_all.filter(~pl.col("item_id").is_in(yn_items.implode()))
    a_yn, se_yn = hanley_se(yn["label"].to_numpy(), yn["v"].to_numpy())
    a_non, se_non = hanley_se(non_yn["label"].to_numpy(), non_yn["v"].to_numpy())
    only_yn = (
        j.filter(pl.col("is_yesno"))
        .group_by("item_id")
        .agg(pl.col("label").first(), pl.col("sent_score").min().alias("v"))
    )
    a_only, se_only = hanley_se(only_yn["label"].to_numpy(), only_yn["v"].to_numpy())

    return {
        "sentence_types": types,
        "diagnostic_drops": {
            "baseline": read(j),
            "drop_all_wrappers": read(j.filter(~WRAPPER)),
            "drop_discourse_frames_only": read(j.filter(~pl.col("is_frame"))),
            "drop_yesno_only": read(j.filter(~pl.col("is_yesno"))),
        },
        "yesno_items": {
            "n": yn.height,
            "neg": int((1 - yn["label"].to_numpy()).sum()),
            "auroc": round(a_yn, 4),
            "se": round(se_yn, 4),
            "auroc_polarity_sentence_only": round(a_only, 4),
            "se_polarity_only": round(se_only, 4),
            "rest_auroc": round(a_non, 4),
            "rest_se": round(se_non, 4),
        },
    }


def strata_without_wrappers(d, sf, it):
    ss = d.filter(pl.col("is_argmax")).select("item_id", "sent_idx", "sent_score")
    j = sf.join(ss, on=["item_id", "sent_idx"], how="inner").filter(~WRAPPER)
    s = j.group_by("item_id").agg(pl.col("sent_score").min().alias("item_score"))
    m = (
        it.select("item_id", "label", "ndoc", "nsent", "npair")
        .join(s, on="item_id", how="left")
        .with_columns(pl.col("item_score").fill_null(-99.0))
    )
    return strata(m)


def vacuous_excluded(d, items):
    """Strata and the contentful over-credited profile with the four contentless
    responses removed - they sit mostly in the 1-passage cell and mask the
    pool-depth gradient."""
    fo = items.filter(pl.col("frame_only"))["item_id"]
    it = item_table(d).filter(~pl.col("item_id").is_in(fo.implode()))
    y = it["label"].to_numpy()
    out = {
        "n_items": it.height,
        "auroc": round(float(roc_auc_score(y, it["item_score"].to_numpy())), 4),
        "strata": strata(it),
    }
    negs = it.filter(pl.col("label") == 0).sort("item_score", descending=True).head(10)
    top10 = negs["item_id"]
    sink = d.filter(pl.col("is_argmax") & pl.col("is_sinking")).filter(
        ~pl.col("item_id").is_in(fo.implode())
    )
    g = sink.with_columns(
        pl.when(pl.col("item_id").is_in(top10.implode()))
        .then(pl.lit("top10_neg_contentful"))
        .when(pl.col("label") == 0)
        .then(pl.lit("other_neg"))
        .otherwise(pl.lit("pos"))
        .alias("grp")
    )
    agg = g.group_by("grp").agg(
        pl.len().alias("n"),
        pl.col("tok_containment").mean().alias("tok_containment"),
        pl.col("num_containment").mean().alias("num_containment"),
        pl.col("max_common_ngram").mean().alias("max_common_ngram"),
        pl.col("sent_score").mean().alias("mean_logit"),
    )
    out["sinking_sentence_by_group"] = {
        r["grp"]: {
            "n": r["n"],
            "tok_containment": round(float(r["tok_containment"]), 3),
            "num_containment": None
            if r["num_containment"] is None
            else round(float(r["num_containment"]), 3),
            "max_common_ngram": round(float(r["max_common_ngram"]), 2),
            "mean_logit": round(float(r["mean_logit"]), 3),
        }
        for r in agg.iter_rows(named=True)
    }
    out["top10_contentful_negatives"] = top10.to_list()
    return out


def misrank_block(d, items):
    it = item_table(d)
    lb = it["label"].to_numpy()
    sc = it["item_score"].to_numpy()
    pos = sc[lb == 1]
    neg = sc[lb == 0]
    neg_ids = it["item_id"].to_numpy()[lb == 0]
    mis = pos[:, None] <= neg[None, :]
    tot = int(mis.sum())
    order = np.argsort(-mis.sum(0))
    top = [
        {
            "item_id": int(neg_ids[k]),
            "score": round(float(neg[k]), 4),
            "misrank_share": round(float(mis[:, k].sum() / tot), 4),
        }
        for k in order[:12]
    ]

    def mass(ids):
        cols = [int(np.where(neg_ids == i)[0][0]) for i in ids if i in set(neg_ids.tolist())]
        return round(float(mis[:, cols].sum() / tot), 4) if cols else 0.0

    fo = items.filter(pl.col("frame_only"))["item_id"].to_list()
    art = items.filter(pl.col("artifact") & (pl.col("label") == 0))["item_id"].to_list()
    keep = items.filter(~pl.col("artifact"))["item_id"]
    m2 = it.filter(pl.col("item_id").is_in(keep.implode()))
    return {
        "top12_negatives": top,
        "top12_cumulative_share": round(float(sum(t["misrank_share"] for t in top)), 4),
        "frame_only_items": fo,
        "frame_only_misrank_share": mass(fo),
        "artifact_negative_items": art,
        "artifact_misrank_share": mass(art),
        "auroc_excluding_artifact_items": round(
            float(roc_auc_score(m2["label"].to_numpy(), m2["item_score"].to_numpy())), 4
        ),
    }


def overlap_block(d, it):
    am = d.filter(pl.col("is_argmax"))
    out = {"spearman_logit_vs_containment": {}}
    for lab in (0, 1):
        s = am.filter(pl.col("label") == lab)
        r = stats.spearmanr(s["tok_containment"].to_numpy(), s["sent_score"].to_numpy())
        out["spearman_logit_vs_containment"][str(lab)] = {
            "n": s.height,
            "rho": round(float(r.statistic), 3),
            "p": float(r.pvalue),
        }
    sink = d.filter(pl.col("is_argmax") & pl.col("is_sinking"))
    lb = it["label"].to_numpy()
    sc = it["item_score"].to_numpy()
    neg_ids = it["item_id"].to_numpy()[lb == 0]
    top12 = neg_ids[np.argsort(-sc[lb == 0])[:12]].tolist()
    g = sink.with_columns(
        pl.when(pl.col("item_id").is_in(pl.Series(top12).implode()))
        .then(pl.lit("top12_neg"))
        .when(pl.col("label") == 0)
        .then(pl.lit("other_neg"))
        .otherwise(pl.lit("pos"))
        .alias("grp")
    )
    agg = g.group_by("grp").agg(
        pl.len().alias("n"),
        pl.col("tok_containment").mean().alias("tok_containment"),
        pl.col("num_containment").mean().alias("num_containment"),
        pl.col("max_common_ngram").mean().alias("max_common_ngram"),
        pl.col("sent_score").mean().alias("mean_logit"),
    )
    out["sinking_sentence_by_group"] = {
        r["grp"]: {
            "n": r["n"],
            "tok_containment": round(float(r["tok_containment"]), 3),
            "num_containment": None
            if r["num_containment"] is None
            else round(float(r["num_containment"]), 3),
            "max_common_ngram": round(float(r["max_common_ngram"]), 2),
            "mean_logit": round(float(r["mean_logit"]), 3),
        }
        for r in agg.iter_rows(named=True)
    }
    return out


def pool_inversion(d, it):
    """Per-sentence best-document logit by pool depth and item label."""
    sd = d.group_by(["item_id", "sent_idx", "doc_idx"]).agg(pl.col("logit").max().alias("dmax"))
    srt = sd.sort(["item_id", "sent_idx", "dmax"], descending=[False, False, True])
    top2 = srt.group_by(["item_id", "sent_idx"], maintain_order=True).agg(
        pl.col("dmax").head(2).alias("t"), pl.len().alias("nd")
    )
    top2 = top2.with_columns(
        pl.col("t").list.get(0).alias("best"),
        pl.col("t").list.get(1, null_on_oob=True).alias("second"),
    ).with_columns((pl.col("best") - pl.col("second")).alias("doc_margin"))
    m = top2.join(it.select("item_id", "label", "ndoc"), on="item_id").filter(pl.col("nd") >= 2)
    m = m.with_columns(
        pl.when(pl.col("ndoc") <= 3)
        .then(pl.lit("pool2_3"))
        .otherwise(pl.lit("pool4plus"))
        .alias("db")
    )
    agg = m.group_by(["db", "label"]).agg(
        pl.len().alias("n_sent"),
        pl.col("best").mean().alias("mean_best_logit"),
        pl.col("doc_margin").mean().alias("mean_doc_margin"),
    )
    return {
        f"{r['db']}_label{r['label']}": {
            "n_sent": r["n_sent"],
            "mean_best_logit": round(float(r["mean_best_logit"]), 3),
            "mean_doc_margin": round(float(r["mean_doc_margin"]), 3),
        }
        for r in agg.iter_rows(named=True)
    }


def enriched_contrast(dumps, sf, items):
    """Pair-by-pair contrast of the enriched-mix draw against the flagship draws."""
    if "h159d1" not in dumps:
        return {"status": "pending - R19-H161_pairs_h159d1.parquet not on disk"}
    a = dumps["h150d1"]
    b = dumps["h159d1"]
    key = ["item_id", "sent_idx", "doc_idx", "win_idx"]
    j = a.select(key + ["logit", "label"]).join(
        b.select(key + ["logit"]), on=key, how="inner", suffix="_b"
    )
    j = j.with_columns((pl.col("logit_b") - pl.col("logit")).alias("d"))
    out = {
        "n_pairs_joined": j.height,
        "mean_logit_shift_by_label": {
            str(r["label"]): {
                "n": r["n"],
                "mean_delta": round(float(r["m"]), 3),
                "median_delta": round(float(r["md"]), 3),
            }
            for r in j.group_by("label")
            .agg(
                pl.len().alias("n"),
                pl.col("d").mean().alias("m"),
                pl.col("d").median().alias("md"),
            )
            .iter_rows(named=True)
        },
    }
    ia = item_table(a).rename({"item_score": "s_a"})
    ib = item_table(b).select("item_id", pl.col("item_score").alias("s_b"))
    m = ia.join(ib, on="item_id")
    lb = m["label"].to_numpy()
    a_auc, _ = hanley_se(lb, m["s_a"].to_numpy())
    b_auc, _ = hanley_se(lb, m["s_b"].to_numpy())
    out["item_auroc"] = {"h150d1": round(a_auc, 4), "h159d1": round(b_auc, 4)}
    out["strata_h159d1"] = strata(
        m.select("item_id", "label", "ndoc", "nsent", "npair", pl.col("s_b").alias("item_score"))
    )
    out["strata_h150d1"] = strata(
        m.select("item_id", "label", "ndoc", "nsent", "npair", pl.col("s_a").alias("item_score"))
    )
    # frame-only and artifact items
    fo = items.filter(pl.col("frame_only"))["item_id"].to_list()
    out["frame_only_item_scores"] = {
        str(int(r["item_id"])): {
            "h150d1": round(float(r["s_a"]), 3),
            "h159d1": round(float(r["s_b"]), 3),
        }
        for r in m.filter(pl.col("item_id").is_in(pl.Series(fo).implode())).iter_rows(named=True)
    }
    ss_a = a.filter(pl.col("is_argmax")).select(
        "item_id", "sent_idx", pl.col("sent_score").alias("a")
    )
    ss_b = b.filter(pl.col("is_argmax")).select(
        "item_id", "sent_idx", pl.col("sent_score").alias("b")
    )
    jj = sf.join(ss_a, on=["item_id", "sent_idx"]).join(ss_b, on=["item_id", "sent_idx"])
    jj = jj.with_columns((pl.col("b") - pl.col("a")).alias("d"), WRAPPER.alias("wrap"))
    out["sentence_logit_shift"] = {
        f"{'wrapper' if r['wrap'] else 'plain'}_label{r['label']}": {
            "n": r["n"],
            "mean_delta": round(float(r["m"]), 3),
        }
        for r in jj.group_by(["wrap", "label"])
        .agg(pl.len().alias("n"), pl.col("d").mean().alias("m"))
        .iter_rows(named=True)
    }
    # which strata moved
    out["stratum_delta"] = {
        k: round(out["strata_h159d1"][k]["auroc"] - out["strata_h150d1"][k]["auroc"], 4)
        for k in out["strata_h159d1"]
        if out["strata_h159d1"][k]["auroc"] is not None
        and out["strata_h150d1"][k]["auroc"] is not None
    }
    # ISOLATION: the two competing credits for the +0.0650 make different
    # predictions about where the per-pair logit moved.
    #   FAVA overlap-prior suppression -> the shift is monotone in lexical
    #     containment: near-copy windows lose the most logit regardless of label
    #   AttributionBench relation rejection -> the shift concentrates on
    #     unsupported claims whose window already carries their entities and
    #     numbers, with no comparable slope on supported ones
    q = j.join(a.select(key + ["tok_containment", "num_containment", "max_common_ngram"]), on=key)
    q = q.with_columns(
        pl.when(pl.col("tok_containment") < 0.5)
        .then(pl.lit("cont_lt50"))
        .when(pl.col("tok_containment") < 0.8)
        .then(pl.lit("cont_50_80"))
        .otherwise(pl.lit("cont_ge80"))
        .alias("cbin"),
        (pl.col("max_common_ngram") >= 8).alias("near_copy"),
    )
    out["shift_by_containment"] = {
        f"{r['cbin']}_label{r['label']}": {"n": r["n"], "mean_delta": round(float(r["m"]), 3)}
        for r in q.group_by(["cbin", "label"])
        .agg(pl.len().alias("n"), pl.col("d").mean().alias("m"))
        .iter_rows(named=True)
    }
    out["shift_by_near_copy"] = {
        f"{'near_copy' if r['near_copy'] else 'not_near_copy'}_label{r['label']}": {
            "n": r["n"],
            "mean_delta": round(float(r["m"]), 3),
        }
        for r in q.group_by(["near_copy", "label"])
        .agg(pl.len().alias("n"), pl.col("d").mean().alias("m"))
        .iter_rows(named=True)
    }
    # the decisive cell: pool depth x near-copy x label. If the enriched mix's gain
    # is near-copy rejection inside deep pools, deep-pool near-copy negatives fall
    # while every other cell rises.
    nd = a.group_by("item_id").agg((pl.col("doc_idx").max() + 1).alias("ndoc"))
    qq = q.join(nd, on="item_id").with_columns(
        pl.when(pl.col("ndoc") >= 4)
        .then(pl.lit("pool4plus"))
        .otherwise(pl.lit("pool1_3"))
        .alias("pb")
    )
    out["shift_by_pool_and_near_copy"] = {
        f"{r['pb']}_{'near_copy' if r['near_copy'] else 'not_near_copy'}_label{r['label']}": {
            "n": r["n"],
            "mean_delta": round(float(r["m"]), 3),
        }
        for r in qq.group_by(["pb", "near_copy", "label"])
        .agg(pl.len().alias("n"), pl.col("d").mean().alias("m"))
        .iter_rows(named=True)
    }
    out["shift_by_pool"] = {
        f"{r['pb']}_label{r['label']}": {"n": r["n"], "mean_delta": round(float(r["m"]), 3)}
        for r in qq.group_by(["pb", "label"])
        .agg(pl.len().alias("n"), pl.col("d").mean().alias("m"))
        .iter_rows(named=True)
    }
    # register-match discriminator: which claim / window length carried the gain
    lens = a.select(key + ["char_len_sent", "char_len_win"])
    ql = j.join(lens, on=key).with_columns(
        pl.when(pl.col("char_len_sent") < 120)
        .then(pl.lit("s_lt120"))
        .when(pl.col("char_len_sent") < 250)
        .then(pl.lit("s_120_250"))
        .otherwise(pl.lit("s_ge250"))
        .alias("sb"),
        pl.when(pl.col("char_len_win") < 500)
        .then(pl.lit("w_lt500"))
        .when(pl.col("char_len_win") < 1000)
        .then(pl.lit("w_500_1000"))
        .otherwise(pl.lit("w_ge1000"))
        .alias("wb"),
    )
    for col, name in (("sb", "shift_by_claim_length"), ("wb", "shift_by_window_length")):
        out[name] = {
            f"{r[col]}_label{r['label']}": {"n": r["n"], "mean_delta": round(float(r["m"]), 3)}
            for r in ql.group_by([col, "label"])
            .agg(pl.len().alias("n"), pl.col("d").mean().alias("m"))
            .iter_rows(named=True)
        }
    sp = {}
    for lab in (0, 1):
        s = q.filter(pl.col("label") == lab)
        for col in ("tok_containment", "num_containment"):
            v = s.select(col, "d").drop_nulls()
            if v.height > 10:
                r = stats.spearmanr(v[col].to_numpy(), v["d"].to_numpy())
                sp[f"{col}_label{lab}"] = {
                    "n": v.height,
                    "rho": round(float(r.statistic), 3),
                    "p": float(r.pvalue),
                }
    out["shift_spearman"] = sp
    # does the enriched draw still need the wrapper filter?
    jw = jj.group_by("item_id").agg(pl.col("label").first(), pl.col("b").min().alias("v"))
    jw_nw = (
        jj.filter(~pl.col("wrap"))
        .group_by("item_id")
        .agg(pl.col("label").first(), pl.col("b").min().alias("v"))
    )
    a1, _ = hanley_se(jw["label"].to_numpy(), jw["v"].to_numpy())
    a2, _ = hanley_se(jw_nw["label"].to_numpy(), jw_nw["v"].to_numpy())
    out["h159d1_wrapper_drop"] = {
        "baseline": round(a1, 4),
        "wrappers_dropped": round(a2, 4),
        "gain": round(a2 - a1, 4),
    }
    return out


def main():
    sf, items = sentence_features()
    dumps = {t: load_pairs(t) for t in CHECKPOINTS}
    dumps = {t: d for t, d in dumps.items() if d is not None}
    if "h150d1" not in dumps:
        raise SystemExit("R19-H161_pairs_h150d1.parquet not on disk - nothing to read")

    res = {
        "experiment": "R19-H162 hagrid mechanism dissection (executor M2) - ANALYSIS ONLY",
        "subset": SUBSET,
        "flagship_2draw_auroc": 0.6424,
        "seed_noise_sd": 0.0001,
        "sample": {
            "items": 250,
            "positives": int(items["label"].sum()),
            "negatives": int((1 - items["label"]).sum()),
            "sentences": sf.height,
            "pairs_h150d1": dumps["h150d1"].height,
        },
        "checkpoints_read": sorted(dumps),
        "control": {t: control(t, d) for t, d in dumps.items()},
    }
    d1 = dumps["h150d1"]
    it1 = item_table(d1)
    res["strata_h150d1"] = strata(it1)
    res["strata_h150d1_wrappers_dropped"] = strata_without_wrappers(d1, sf, it1)
    res["kdoc_curve_h150d1"] = kdoc_curve(d1, it1)
    res["pool_inversion_h150d1"] = pool_inversion(d1, it1)
    res["wrapper_h150d1"] = wrapper_block(d1, sf)
    res["misrank_h150d1"] = misrank_block(d1, items)
    res["overlap_h150d1"] = overlap_block(d1, it1)
    res["vacuous_excluded_h150d1"] = vacuous_excluded(d1, items)
    if "h150d2" in dumps:
        d2 = dumps["h150d2"]
        it2 = item_table(d2)
        res["strata_h150d2"] = strata(it2)
        res["strata_h150d2_wrappers_dropped"] = strata_without_wrappers(d2, sf, it2)
        res["kdoc_curve_h150d2"] = kdoc_curve(d2, it2)
        res["pool_inversion_h150d2"] = pool_inversion(d2, it2)
        res["wrapper_h150d2"] = wrapper_block(d2, sf)
        res["misrank_h150d2"] = misrank_block(d2, items)
        res["overlap_h150d2"] = overlap_block(d2, it2)
        res["vacuous_excluded_h150d2"] = vacuous_excluded(d2, items)
    if "h159d1" in dumps:
        d3 = dumps["h159d1"]
        it3 = item_table(d3)
        res["strata_h159d1_wrappers_dropped"] = strata_without_wrappers(d3, sf, it3)
        res["kdoc_curve_h159d1"] = kdoc_curve(d3, it3)
        res["pool_inversion_h159d1"] = pool_inversion(d3, it3)
        res["wrapper_h159d1"] = wrapper_block(d3, sf)
        res["misrank_h159d1"] = misrank_block(d3, items)
        res["overlap_h159d1"] = overlap_block(d3, it3)
        res["vacuous_excluded_h159d1"] = vacuous_excluded(d3, items)
    res["enriched_contrast"] = enriched_contrast(dumps, sf, items)
    res["mechanisms"] = MECHANISMS
    res["not_evidenced"] = NOT_EVIDENCED
    res["build_first"] = BUILD_FIRST
    res["isolation"] = ISOLATION
    res["frame_drop_control"] = FRAME_DROP_CONTROL
    res["training_mix_wrapper_census"] = TRAINING_MIX_WRAPPER_CENSUS

    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("control", "strata_h150d1")}, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
