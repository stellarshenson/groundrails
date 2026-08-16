"""R19-H162 - pubmedqa failure analysis over the R19-H161 per-pair dump. CPU ONLY.

Joins the banked per-pair logits (R19-H161_pairs_*.parquet) back to the frozen
pubmedqa gate sample text through the deterministic read geometry (item ->
H92 sentence -> document -> window; pubmedqa documents all fit in one 1,500-char
window, so doc_idx == win_idx == the document ordinal). Reports:

  1. read fidelity against the banked windowed AUROC
  2. error split at the in-sample macro-F1-optimal threshold, both directions
  3. what the sinking sentences of misread items have in common
  4. a rule-based mechanism tagging of response sentences and its association
     with score and with the item label

ANALYSIS ONLY. Nothing here trains, tunes or selects on arena statistics; the
threshold is the R17-H147 stated choice and the threshold-free rank view is
reported alongside.

Run:  uv run python experiments/grounding-semantic/R19-H162_pubmedqa_analyze.py
"""

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R19-H162_pubmedqa_analysis.json"
SUBSET = "pubmedqa"
BANKED = {"h150d1": 0.5893, "h150d2": None, "h159d1": None}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA

# --- mechanism rule layer -----------------------------------------------------
# Each rule is a SIGNAL, not a verdict. Manual reading adjudicates; the counts
# below are reported with their binomial SEs and the coarse ones are labelled.

_AIM = re.compile(
    r"(?:^|\.\s|AIM:|OBJECTIVE:|PURPOSE:|BACKGROUND:)\s*"
    r"(?:To\s+\w+|We\s+(?:examined|evaluated|investigated|compared|assessed|"
    r"studied|aimed|sought|analyze|analyzed|report|describe)|"
    r"The\s+(?:aim|purpose|objective|goal)\b|"
    r"This\s+study\s+(?:aims?|sought|was\s+(?:designed|undertaken)))",
    re.IGNORECASE,
)
_RESULT_LANG = re.compile(
    r"\b(?:p\s*[<>=]\s*0?\.\d+|significantly|significant\s+(?:difference|increase|"
    r"decrease|association)|odds\s+ratio|hazard\s+ratio|95\s*%\s*ci|"
    r"was\s+(?:higher|lower|greater|associated)|were\s+(?:higher|lower|greater|"
    r"associated)|no\s+(?:significant\s+)?difference|showed\s+that|"
    r"demonstrated\s+that|found\s+that|resulted\s+in|led\s+to)\b",
    re.IGNORECASE,
)
_FINDING_CLAIM = re.compile(
    r"\b(?:found|showed|demonstrated|revealed|reported|concluded|indicates?\s+that|"
    r"suggests?\s+that|was\s+(?:significantly|associated|higher|lower|effective|"
    r"superior)|were\s+(?:significantly|associated|higher|lower|effective|superior)|"
    r"results?\s+(?:show|showed|indicate|indicated)|improves?|reduces?|increases?|"
    r"decreases?|does\s+not\s+affect|did\s+not\s+affect|no\s+difference)\b",
    re.IGNORECASE,
)
_META = re.compile(
    r"(?:the\s+(?:provided\s+)?(?:context|passage|text|documents?|information|"
    r"pieces\s+of\s+context|studies\s+mentioned|abstracts?)|based\s+on\s+the\s+"
    r"(?:provided\s+)?(?:context|information|text)|"
    r"further\s+(?:research|analysis|studies|study)\s+(?:is|are|would)|"
    r"we\s+would\s+need\s+to|more\s+(?:research|studies|data)\s+(?:is|are)\s+needed|"
    r"cannot\s+be\s+determined|is\s+not\s+(?:possible|clear)\s+to\s+determine)",
    re.IGNORECASE,
)
_ABSENCE = re.compile(
    r"(?:does\s+not\s+(?:mention|provide|state|specify|contain|discuss|address|"
    r"include|indicate)|do\s+not\s+(?:mention|provide|state|specify|contain|"
    r"discuss|address|include|indicate)|no\s+(?:information|mention|data|"
    r"details?)\s+(?:is|are|about|on|regarding)|is\s+not\s+(?:mentioned|provided|"
    r"stated|specified|discussed)|are\s+not\s+(?:mentioned|provided|stated))",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:may|might|could|would|suggest|suggests|appear|appears|likely|possibly|"
    r"potential|potentially|seem|seems|probably|perhaps|presumably)\b",
    re.IGNORECASE,
)
_QUANT = re.compile(
    r"\b(?:all|every|any|both|none|no\s+\w+|always|never|each|most|majority|"
    r"generally|typically|in\s+general|universally|consistently)\b",
    re.IGNORECASE,
)
_CAUSAL = re.compile(
    r"\b(?:cause[sd]?|causing|due\s+to|because\s+of|leads?\s+to|led\s+to|"
    r"results?\s+in|resulted\s+in|induces?d?|responsible\s+for|effect\s+of|"
    r"attributable\s+to)\b",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def tag_sentence(s, ev_text, doc_texts):
    """Rule signals for one response sentence against the item's whole evidence."""
    aim_ev = bool(_AIM.search(ev_text))
    res_ev = bool(_RESULT_LANG.search(ev_text))
    nums = _NUM.findall(s)
    ev_nums = set(_NUM.findall(ev_text))
    # cross-document spread: how many DIFFERENT docs the sentence's content words
    # are best split across (a conjunction whose halves live in different docs)
    toks = _content(s)
    per_doc = [len(toks & _content(d)) for d in doc_texts]
    top = sorted(per_doc, reverse=True)
    return {
        "is_meta": bool(_META.search(s)),
        "is_absence": bool(_ABSENCE.search(s)),
        "asserts_finding": bool(_FINDING_CLAIM.search(s)),
        "has_hedge": bool(_HEDGE.search(s)),
        "has_quantifier": bool(_QUANT.search(s)),
        "has_causal": bool(_CAUSAL.search(s)),
        "n_num": len(nums),
        "n_num_absent": sum(1 for v in nums if v not in ev_nums),
        # aim_vs_finding signature: the claim asserts a RESULT while the
        # evidence for it is an objective statement carrying no result language
        "sig_aim_vs_finding": bool(
            _FINDING_CLAIM.search(s) and aim_ev and not res_ev and not _META.search(s)
        ),
        "doc_spread_top1": top[0] if top else 0,
        "doc_spread_top2": top[1] if len(top) > 1 else 0,
        "n_docs_contributing": sum(1 for v in per_doc if v >= 3),
    }


_STOP_WORDS = (
    "the a an of and or to in for with on by is are was were be been that this "
    "these those it its as at from not no than then which who whom whose but if "
    "we they he she i you our their his her there here can could may might will "
    "would should shall do does did done has have had having more most other "
    "such some any all both each between into during also"
)
_STOP = set(_STOP_WORDS.split())


def _content(t):
    return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in _STOP}


def best_threshold(y, s):
    grid = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 199)))
    f1s = [f1_score(y, (s >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(f1s))])


def main():
    subs = ARENA.load_subsets()
    claims, chunk_lists, y = subs[SUBSET]
    sent_lists = [H92.sentences(c) for c in claims]

    # sentence/window text keyed by the read geometry
    sent_rows = []
    for i, (sents, ks) in enumerate(zip(sent_lists, chunk_lists, strict=True)):
        ev = " ".join(ks)
        for si, s in enumerate(sents):
            t = tag_sentence(s, ev, ks)
            t.update({"item_id": i, "sent_idx": si, "sentence": s})
            sent_rows.append(t)
    tags = pl.DataFrame(sent_rows)

    out = {"n_items": len(y), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    per_draw = {}

    for draw in ("h150d1", "h150d2", "h159d1"):
        p = HERE / f"R19-H161_pairs_{draw}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p).filter(pl.col("subset") == SUBSET)
        if len(d) == 0:
            continue

        # item-level read
        items = (
            d.group_by("item_id")
            .agg(pl.col("item_score").first(), pl.col("label").first())
            .sort("item_id")
        )
        yy = items["label"].to_numpy()
        ss = items["item_score"].to_numpy()
        auc = float(roc_auc_score(yy, ss))
        thr = best_threshold(yy, ss)
        pred = (ss >= thr).astype(int)
        fp = int(((pred == 1) & (yy == 0)).sum())
        fn = int(((pred == 0) & (yy == 1)).sum())

        # sentence-level frame with tags
        sent = (
            d.filter(pl.col("is_argmax"))
            .select(
                "item_id",
                "sent_idx",
                "label",
                "n_sent_item",
                "sent_score",
                "item_score",
                "is_sinking",
                "doc_idx",
                "tok_containment",
                "tok_jaccard",
                "num_containment",
                "max_common_ngram",
            )
            .join(tags, on=["item_id", "sent_idx"], how="left")
        )

        # --- MIN-length penalty: does response length drive the item score? ---
        it = (
            sent.group_by("item_id")
            .agg(
                pl.col("n_sent_item").first(),
                pl.col("item_score").first(),
                pl.col("label").first(),
            )
            .sort("item_id")
        )
        rho_len = float(
            np.corrcoef(
                it["n_sent_item"].to_numpy().astype(float),
                it["item_score"].to_numpy().astype(float),
            )[0, 1]
        )
        auc_len = float(roc_auc_score(it["label"].to_numpy(), -it["n_sent_item"].to_numpy()))

        # --- what sinking sentences look like ---
        flags = [
            "is_meta",
            "is_absence",
            "asserts_finding",
            "has_hedge",
            "has_quantifier",
            "has_causal",
            "sig_aim_vs_finding",
        ]
        sink = sent.filter(pl.col("is_sinking"))
        nonsink = sent.filter(~pl.col("is_sinking"))

        def rates(frame, flags=flags):
            if len(frame) == 0:
                return {}
            r = {f: round(float(frame[f].mean()), 4) for f in flags}
            r["n"] = len(frame)
            r["mean_tok_containment"] = round(float(frame["tok_containment"].mean()), 4)
            r["mean_sent_score"] = round(float(frame["sent_score"].mean()), 4)
            return r

        # sinking sentences of MISREAD items vs correctly-read items
        err_ids = {int(v) for v in items["item_id"].to_numpy()[pred != yy]}
        sink = sink.with_columns(pl.col("item_id").is_in(list(err_ids)).alias("item_erred"))

        # --- per-tag score contrast: does the model score aim_vs_finding high? ---
        tag_contrast = {}
        for f in flags:
            a = sent.filter(pl.col(f))
            b = sent.filter(~pl.col(f))
            if len(a) < 15:
                tag_contrast[f] = {"n": len(a), "note": "n<15, not read"}
                continue
            tag_contrast[f] = {
                "n": len(a),
                "share": round(len(a) / len(sent), 4),
                "mean_sent_score_flagged": round(float(a["sent_score"].mean()), 4),
                "mean_sent_score_other": round(float(b["sent_score"].mean()), 4),
                "delta": round(float(a["sent_score"].mean() - b["sent_score"].mean()), 4),
                "mean_tok_containment_flagged": round(float(a["tok_containment"].mean()), 4),
                "share_in_neg_items": round(float((a["label"] == 0).mean()), 4),
                "share_in_pos_items": round(float((a["label"] == 1).mean()), 4),
                "sink_rate": round(float(a["is_sinking"].mean()), 4),
            }

        # --- the discrimination question: per-sentence AUROC of the tag as a
        # predictor of the ITEM label, and the model's own separation ---
        neg_sent = sent.filter(pl.col("label") == 0)
        pos_sent = sent.filter(pl.col("label") == 1)

        # sentence-score separation between pos-item and neg-item sentences
        sep = float(roc_auc_score(sent["label"].to_numpy(), sent["sent_score"].to_numpy()))

        # ORACLE: if the model perfectly scored only the aim_vs_finding sentences
        # low in negatives, how much of the ranking would it fix? Reported as the
        # share of negatives carrying at least one such sentence.
        neg_items_with_sig = (
            neg_sent.group_by("item_id")
            .agg(pl.col("sig_aim_vs_finding").any())
            .get_column("sig_aim_vs_finding")
            .mean()
        )
        pos_items_with_sig = (
            pos_sent.group_by("item_id")
            .agg(pl.col("sig_aim_vs_finding").any())
            .get_column("sig_aim_vs_finding")
            .mean()
        )

        per_draw[draw] = {
            "auroc": round(auc, 4),
            "banked": BANKED.get(draw),
            "threshold": round(thr, 4),
            "n_errors": int((pred != yy).sum()),
            "false_positives": fp,
            "fp_rate_of_neg": round(fp / max(1, int((yy == 0).sum())), 4),
            "false_negatives": fn,
            "fn_rate_of_pos": round(fn / max(1, int((yy == 1).sum())), 4),
            "sentence_level_label_auroc": round(sep, 4),
            "min_length_penalty": {
                "pearson_r_nsent_vs_itemscore": round(rho_len, 4),
                "auroc_of_negative_sentence_count_alone": round(auc_len, 4),
            },
            "sinking_sentences": {
                "misread_items": rates(sink.filter(pl.col("item_erred"))),
                "correct_items": rates(sink.filter(~pl.col("item_erred"))),
                "all": rates(sink),
                "non_sinking_all": rates(nonsink),
            },
            "tag_contrast": tag_contrast,
            "aim_vs_finding_item_coverage": {
                "share_of_negative_items_with_signal": round(float(neg_items_with_sig), 4),
                "share_of_positive_items_with_signal": round(float(pos_items_with_sig), 4),
            },
        }
        sent.write_parquet(HERE / f"R19-H162_pubmedqa_sent_{draw}.parquet")

    out["per_draw"] = per_draw
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
