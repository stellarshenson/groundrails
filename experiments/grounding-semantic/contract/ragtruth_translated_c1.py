"""Contract C1 - label commensurability for `ragtruth_translated`. CPU ONLY.

The mandatory test: claim-to-evidence containment on the NEGATIVE leg against
the POSITIVE leg, per language and pooled.

TWO INSTRUMENTS, both reported, neither substituted silently:

  banked   `R20-H174_lane_common.containment` verbatim - token set of
           `[a-z0-9]+` over lowercased text. This is the instrument that
           produced the campaign's 0.9129 figure on the R20-H175b contrast
           lane, so it is the comparable one. It is ALSO blind on this member:
           it drops every diacritic-bearing token and yields ZERO tokens on
           Chinese. Reported for comparability, flagged, not used as primary.

  unicode  `\\w+` with re.UNICODE over casefolded text - the tokenizer the
           SHIPPED lexical tier uses (`src/groundrails/lexical.py`). CJK runs
           are additionally split into character bigrams, the standard unit for
           scriptio-continua text; character unigrams are reported beside them
           because a Chinese unigram inventory is small enough to make
           containment degenerate. This is the PRIMARY read for this member.

Supplementary and reported SEPARATELY from the registered clause (C5's rule on
executor-added probes): containment of the ANNOTATED HALLUCINATED SPAN of each
negative, which is what the row label actually points at.

Run:  uv run python experiments/grounding-semantic/contract/ragtruth_translated_c1.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
ARCHIVE = ROOT / "data" / "external" / "datasets" / "dataset-ragtruth-translated.zip"
MEMBER = HERE / "ragtruth_translated_member.parquet"
OUT = HERE / "ragtruth_translated_c1.json"

LANGS = ("de", "fr", "es", "it", "pl", "hu", "cn")

# --- instruments ------------------------------------------------------------
_BANKED = re.compile(r"[a-z0-9]+")  # R20-H174_lane_common._WORD, verbatim
_WORD_U = re.compile(r"\w+", re.UNICODE)
_CJK = re.compile(
    "[㐀-䶿一-鿿豈-﫿　-〿＀-￯]"
)


def tok_banked(text):
    return _BANKED.findall(text.lower())


def _split_cjk(t):
    out, buf = [], ""
    for ch in t:
        if _CJK.match(ch):
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def tok_unicode(text, cjk_bigram=True):
    out = []
    for t in _WORD_U.findall(text.casefold()):
        if _CJK.search(t):
            parts = _split_cjk(t)
            if cjk_bigram:
                # character bigrams over each maximal CJK run
                run = []
                for p in parts:
                    if _CJK.match(p):
                        run.append(p)
                    else:
                        out.extend(_bigrams(run))
                        run = []
                        out.append(p)
                out.extend(_bigrams(run))
            else:
                out.extend(parts)
        else:
            out.append(t)
    return out


def _bigrams(run):
    if not run:
        return []
    if len(run) == 1:
        return [run[0]]
    return ["".join(run[i : i + 2]) for i in range(len(run) - 1)]


def containment(claim, text, tok):
    ct = set(tok(claim))
    if not ct:
        return None  # unscorable under this instrument
    return len(ct & set(tok(text))) / len(ct)


def dist(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0, "unscorable": len(vals)}
    return {
        "n": int(a.size),
        "unscorable": int(len(vals) - a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "frac_ge_0.90": round(float((a >= 0.90).mean()), 4),
        "frac_eq_1.0": round(float((a >= 0.99999).mean()), 4),
    }


def leg_block(pos, neg):
    p, n = dist(pos), dist(neg)
    b = {"positive": p, "negative": n}
    if p.get("n") and n.get("n"):
        b["gap_mean_pos_minus_neg"] = round(p["mean"] - n["mean"], 4)
        b["neg_mean"] = n["mean"]
        b["neg_ge90_share"] = n["frac_ge_0.90"]
        b["neg_fully_attested_share"] = n["frac_eq_1.0"]
        # the contract bar, in the form its provenance measured it:
        # negatives attested at >= 0.90 AND within 0.10 of the positive leg
        b["c1_reject"] = bool(n["mean"] >= 0.90 and abs(p["mean"] - n["mean"]) <= 0.10)
        # the alternative reading, reported so the bar is not chosen by me
        b["c1_reject_share_reading"] = bool(
            n["frac_ge_0.90"] >= 0.90 and abs(p["frac_ge_0.90"] - n["frac_ge_0.90"]) <= 0.10
        )
    return b


def main():
    mem = pl.read_parquet(MEMBER)
    print(f"member: {mem.height} rows", flush=True)

    z = zipfile.ZipFile(ARCHIVE)
    res = {
        "member": "ragtruth_translated",
        "rows": int(mem.height),
        "predicate": {
            "head": "grounding scalar (task_head) - the shipped ground() support score",
            "label_rule_in_loader": "label = 1.0 iff labels.list.len() == 0, i.e. the "
            "row carries NO annotated hallucination span; else 0.0",
            "corpus_predicate": "RAGTruth response-level faithfulness: a response is "
            "labelled unsupported iff a human annotator (spans inherited from the "
            "English original and re-aligned after machine translation) marked ANY "
            "span of it as evident_conflict or baseless_info against the prompt's "
            "source material",
            "is_support": True,
            "caveats": [
                "the unit is the WHOLE response, not a claim - a negative is a "
                "response most of which is normally supported, carrying one "
                "unsupported span",
                "the evidence field is the RAGTruth `prompt`, which is the task "
                "instruction plus (for QA) the question plus the passages - not a "
                "passage alone",
            ],
        },
        "instruments": {
            "banked": "R20-H174_lane_common.containment verbatim, [a-z0-9]+ over "
            "lower(); the instrument behind the campaign's 0.9129 figure",
            "unicode_primary": "\\w+ re.UNICODE over casefold(), CJK runs split to "
            "character BIGRAMS",
            "unicode_cjk_unigram": "same, CJK runs split to character UNIGRAMS",
        },
        "per_language": {},
    }

    pooled = {k: {"pos": [], "neg": []} for k in ("banked", "unicode", "unicode_uni")}
    span_pooled = {"unicode": [], "banked": []}
    task_pooled = {}

    for lg in LANGS:
        sub = mem.filter(pl.col("lang") == lg)
        claims = sub["claim"].to_list()
        chunks = sub["chunk"].to_list()
        labels = sub["label"].to_numpy()

        # raw archive, for the annotated spans and the task_type stratification
        nm = next(
            x for x in z.namelist() if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet")
        )
        raw = pl.read_parquet(io.BytesIO(z.read(nm)))
        assert raw.height == sub.height, f"{lg}: row count drift {raw.height} vs {sub.height}"
        assert raw["answer"].to_list() == claims, f"{lg}: row ORDER drift vs the loader"
        spans = raw["labels"].to_list()
        tasks = raw["task_type"].to_list()

        block = {"rows": sub.height, "label1_rate": round(float(labels.mean()), 4)}
        for key, tok in (
            ("banked", tok_banked),
            ("unicode", lambda t: tok_unicode(t, cjk_bigram=True)),
            ("unicode_uni", lambda t: tok_unicode(t, cjk_bigram=False)),
        ):
            pos, neg = [], []
            for c, k, y in zip(claims, chunks, labels, strict=True):
                v = containment(c, k, tok)
                (pos if y == 1.0 else neg).append(v)
            pooled[key]["pos"] += pos
            pooled[key]["neg"] += neg
            block[key] = leg_block(pos, neg)

        # supplementary: the annotated hallucinated span itself
        sp_u, sp_b = [], []
        for c, k, y, sl in zip(claims, chunks, labels, spans, strict=True):
            if y == 1.0 or not sl:
                continue
            txt = " ".join(c[s["start"] : s["end"]] for s in sl)
            if not txt.strip():
                continue
            sp_u.append(containment(txt, k, lambda t: tok_unicode(t, cjk_bigram=True)))
            sp_b.append(containment(txt, k, tok_banked))
        span_pooled["unicode"] += sp_u
        span_pooled["banked"] += sp_b
        block["supplementary_annotated_span_containment"] = {
            "note": "EXECUTOR-ADDED probe, reported separately from the registered "
            "C1 conjunction - containment of the concatenated annotated "
            "hallucinated spans of each negative response",
            "unicode_primary": dist(sp_u),
            "banked": dist(sp_b),
        }

        # task_type stratification on the primary instrument
        by_task = {}
        for c, k, y, t in zip(claims, chunks, labels, tasks, strict=True):
            v = containment(c, k, lambda x: tok_unicode(x, cjk_bigram=True))
            by_task.setdefault(t, {"pos": [], "neg": []})["pos" if y == 1.0 else "neg"].append(v)
            task_pooled.setdefault(t, {"pos": [], "neg": []})[
                "pos" if y == 1.0 else "neg"
            ].append(v)
        block["by_task_type_unicode_primary"] = {
            t: leg_block(v["pos"], v["neg"]) for t, v in sorted(by_task.items())
        }

        res["per_language"][lg] = block
        print(
            f"{lg}: unicode pos {block['unicode']['positive']['mean']} "
            f"neg {block['unicode']['negative']['mean']} "
            f"gap {block['unicode'].get('gap_mean_pos_minus_neg')} "
            f"reject={block['unicode'].get('c1_reject')}",
            flush=True,
        )

    res["pooled"] = {k: leg_block(v["pos"], v["neg"]) for k, v in pooled.items()}
    res["pooled"]["supplementary_annotated_span_containment"] = {
        "note": "EXECUTOR-ADDED, separate from the registered conjunction",
        "unicode_primary": dist(span_pooled["unicode"]),
        "banked": dist(span_pooled["banked"]),
    }
    res["pooled_by_task_type_unicode_primary"] = {
        t: leg_block(v["pos"], v["neg"]) for t, v in sorted(task_pooled.items())
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["pooled"], indent=2), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
