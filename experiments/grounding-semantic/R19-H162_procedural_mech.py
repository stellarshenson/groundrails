"""R19-H162 - mechanism measurement for the procedural register. ANALYSIS ONLY, CPU only.

Third stage of the R19-H162 M4 lane. `R19-H162_procedural_autopsy.py` settled the
window-inflation question; `R19-H162_procedural_export.py` produced the error
dossier that named the candidate mechanisms by reading. This script measures each
candidate against the banked R19-H161 per-pair logit dump, so every mechanism in
the memo carries a number rather than an impression.

Candidates measured:

  lexical_ceiling   - AUROC of a pure surface-overlap scorer under the identical
                      MAX-over-windows / MIN-over-sentences aggregation; how much
                      of the model's read is explained without any entailment
  bind_product_ver  - claims that attach a fact to a product-and-version
                      identifier; containment of the sentence's identifier tokens
                      in its own argmax window, split by label and correctness
  pointer_answer    - responses that answer by pointing at a document ("you can
                      find this in bulletin X") rather than by stating the fact
  menu_path         - claims rendering a UI navigation path with separators
                      against evidence that serialises the same path bare
  discourse_sink    - contentless preamble / recap sentences deciding the item's
                      MIN

Run:  uv run python experiments/grounding-semantic/R19-H162_procedural_mech.py
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
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
DUMP = HERE / "R19-H161_pairs_h150d1.parquet"
GEOM = HERE / "R19-H162_procedural_geometry.parquet"
OUT_JSON = HERE / "R19-H162_procedural_mech.json"

SUBSETS = ("emanual", "techqa", "delucionqa")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA

# Identifier classes of the technical-documentation register. Each names the thing
# a fact must be bound TO: a product release, a fix pack, a defect record, a CVE.
ID_PATTERNS = {
    "version": re.compile(r"\b[Vv]?\d+\.\d+(?:\.\d+){0,3}\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{3,7}\b"),
    "apar": re.compile(r"\b[A-Z]{2}\d{5}\b"),
    "vmajor": re.compile(r"\b[Vv]ersion \d+(?:\.\d+)*\b|\bV\d+\b"),
}

POINTER = re.compile(
    r"you can find|can be found|refer to the following|by referring to|"
    r"by visiting|see the security bulletin|consult(?:ing)? the|"
    r"is available (?:at|on) the|search(?:ing)? (?:for|on) ",
    re.IGNORECASE,
)
URL = re.compile(r"https?://")
MENU_PATH = re.compile(r"\b\w[\w ]*\s>\s\w")
PREAMBLE = re.compile(
    r"^(based on|according to|here are|here is|to (?:fix|resolve|troubleshoot|"
    r"configure|turn|set|create|update|change|check)\b[^.]*:\s*$|"
    r"the (?:key|main) (?:point|difference|ways?)|in summary|so in summary|"
    r"by following these steps)",
    re.IGNORECASE,
)
RECAP = re.compile(
    r"^(in summary|so in summary|so,|overall|by following these steps|the key is)", re.IGNORECASE
)


def content_tokens(t):
    return set(re.findall(r"[a-z0-9][a-z0-9._-]*", t.lower()))


def ids_in(t):
    out = set()
    for pat in ID_PATTERNS.values():
        out |= {m.strip().lower() for m in pat.findall(t)}
    return out


def auroc(y, s):
    return float(roc_auc_score(y, s))


def sent_frame(d):
    """One row per response sentence: model max, argmax provenance, geometry."""
    mx = d.group_by(["item_id", "sent_idx"]).agg(
        pl.col("logit").max().alias("model_max"),
        pl.col("label").first(),
        pl.col("n_win_sent").first(),
        pl.col("n_sent_item").first(),
        pl.col("tok_containment").max().alias("lex_max"),
        pl.col("num_containment").max().alias("num_max"),
    )
    am = (
        d.filter(pl.col("is_argmax"))
        .group_by(["item_id", "sent_idx"])
        .agg(
            pl.col("win_idx").first().alias("am_win"),
            pl.col("doc_idx").first().alias("am_doc"),
            pl.col("tok_containment").first().alias("am_tok_cont"),
        )
    )
    lex_am = (
        d.sort(["item_id", "sent_idx", "tok_containment"], descending=[False, False, True])
        .group_by(["item_id", "sent_idx"], maintain_order=True)
        .head(1)
        .select(
            [
                "item_id",
                "sent_idx",
                pl.col("win_idx").alias("lex_win"),
                pl.col("doc_idx").alias("lex_doc"),
            ]
        )
    )
    return mx.join(am, on=["item_id", "sent_idx"]).join(lex_am, on=["item_id", "sent_idx"])


def item_min(sf, col):
    return (
        sf.group_by("item_id")
        .agg(pl.col(col).min().alias("score"), pl.col("label").first())
        .sort("item_id")
    )


def main():
    subs = ARENA.load_subsets()
    df = pl.read_parquet(DUMP)
    report = {}

    for subset in SUBSETS:
        d = df.filter(pl.col("subset") == subset)
        claims, chunk_lists, _y = subs[subset]

        # sentence and window text, same enumeration as the dump
        sent_txt, win_txt = {}, {}
        for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
            wi = 0
            for k in ks:
                for w in ARM.windows(k):
                    win_txt[(i, wi)] = w
                    wi += 1
            for si, s in enumerate(H92.sentences(c)):
                sent_txt[(i, si)] = s

        sf = sent_frame(d).sort(["item_id", "sent_idx"])
        keys = list(zip(sf["item_id"].to_list(), sf["sent_idx"].to_list(), strict=True))
        texts = [sent_txt[k] for k in keys]

        # per-sentence identifier binding against the model's own argmax window
        id_cont, n_ids, am_ids_only = [], [], []
        for (iid, _si), t, aw in zip(keys, texts, sf["am_win"].to_list(), strict=True):
            sids = ids_in(t)
            w = win_txt.get((iid, aw), "")
            wids = ids_in(w)
            n_ids.append(len(sids))
            id_cont.append(float(len(sids & wids) / len(sids)) if sids else -1.0)
            am_ids_only.append(len(wids))

        sf = sf.with_columns(
            pl.Series("sent_text", texts),
            pl.Series("id_containment", id_cont),
            pl.Series("n_ids", n_ids, dtype=pl.Int32),
            pl.Series("am_win_n_ids", am_ids_only, dtype=pl.Int32),
            pl.Series("is_preamble", [bool(PREAMBLE.search(t.strip())) for t in texts]),
            pl.Series("is_recap", [bool(RECAP.search(t.strip())) for t in texts]),
            pl.Series("is_menu_path", [bool(MENU_PATH.search(t)) for t in texts]),
            pl.Series("is_pointer", [bool(POINTER.search(t) or URL.search(t)) for t in texts]),
            pl.Series("n_content_tok", [len(content_tokens(t)) for t in texts]),
        )
        sf = sf.with_columns(
            (pl.col("model_max") == pl.col("model_max").min().over("item_id")).alias("is_sink"),
            (pl.col("am_doc") != pl.col("lex_doc")).alias("argmax_off_lexical_doc"),
        )

        blk = {}

        # --- lexical ceiling ------------------------------------------------------
        im = item_min(sf, "model_max")
        y = im["label"].to_numpy()
        blk["auroc_model"] = round(auroc(y, im["score"].to_numpy()), 5)
        for col, name in (("lex_max", "tok_containment"), ("num_max", "num_containment")):
            s = item_min(sf.with_columns(pl.col(col).fill_null(0.0)), col)["score"].to_numpy()
            blk[f"auroc_lexical_{name}"] = round(auroc(y, s), 5)
        blk["n_items"] = len(y)
        blk["n_negative"] = int((y == 0).sum())

        # --- identifier binding ---------------------------------------------------
        withid = sf.filter(pl.col("n_ids") > 0)
        blk["identifier_binding"] = {
            "share_of_sentences_carrying_an_identifier": round(withid.height / sf.height, 4),
            "mean_id_containment_in_own_argmax_window": round(
                float(withid["id_containment"].mean()), 4
            )
            if withid.height
            else None,
            "by_label": [
                {
                    "label": int(r["label"]),
                    "n": int(r["n"]),
                    "mean_id_containment": round(float(r["m"]), 4),
                    "mean_model_max": round(float(r["s"]), 4),
                }
                for r in withid.group_by("label")
                .agg(
                    pl.len().alias("n"),
                    pl.col("id_containment").mean().alias("m"),
                    pl.col("model_max").mean().alias("s"),
                )
                .sort("label")
                .iter_rows(named=True)
            ],
        }
        # does the model score track identifier binding at all?
        if withid.height > 30:
            blk["identifier_binding"]["spearman_idcont_vs_model"] = round(
                float(
                    withid.select(pl.corr("id_containment", "model_max", method="spearman"))[0, 0]
                ),
                4,
            )
            blk["identifier_binding"]["spearman_tokcont_vs_model"] = round(
                float(withid.select(pl.corr("lex_max", "model_max", method="spearman"))[0, 0]), 4
            )
            # the decisive contrast: sentences whose identifiers are FULLY bound in
            # their argmax window vs sentences whose identifiers are absent from it
            bound = withid.filter(pl.col("id_containment") >= 0.999)
            unbound = withid.filter(pl.col("id_containment") <= 0.001)
            blk["identifier_binding"]["bound_vs_unbound"] = {
                "n_bound": bound.height,
                "n_unbound": unbound.height,
                "mean_model_max_bound": round(float(bound["model_max"].mean()), 4)
                if bound.height
                else None,
                "mean_model_max_unbound": round(float(unbound["model_max"].mean()), 4)
                if unbound.height
                else None,
            }

        # --- sentence-class incidence and sink share ------------------------------
        classes = {}
        for flag in ("is_preamble", "is_recap", "is_menu_path", "is_pointer"):
            sub = sf.filter(pl.col(flag))
            classes[flag] = {
                "n_sentences": sub.height,
                "share_of_sentences": round(sub.height / sf.height, 4),
                "mean_model_max": round(float(sub["model_max"].mean()), 4) if sub.height else None,
                "mean_model_max_others": round(
                    float(sf.filter(~pl.col(flag))["model_max"].mean()), 4
                ),
                "share_that_are_the_item_sink": round(float(sub["is_sink"].mean()), 4)
                if sub.height
                else None,
                "sink_share_others": round(float(sf.filter(~pl.col(flag))["is_sink"].mean()), 4),
            }
        blk["sentence_classes"] = classes

        # --- item level: does a class predict the label, and does the model use it?
        item_flags = sf.group_by("item_id").agg(
            pl.col("label").first(),
            pl.col("is_pointer").any().alias("has_pointer"),
            pl.col("is_menu_path").any().alias("has_menu_path"),
            pl.col("is_recap").any().alias("has_recap"),
        )
        im2 = im.join(item_flags, on="item_id")
        cls_item = {}
        for flag in ("has_pointer", "has_menu_path", "has_recap"):
            g = im2.group_by(flag).agg(
                pl.len().alias("n"),
                pl.col("label").mean().alias("base_rate"),
                pl.col("score").mean().alias("mean_item_score"),
            )
            rows = {
                str(r[flag]): {
                    "n": int(r["n"]),
                    "base_rate_supported": round(float(r["base_rate"]), 4),
                    "mean_item_score": round(float(r["mean_item_score"]), 4),
                }
                for r in g.iter_rows(named=True)
            }
            # within-stratum AUROC: does the model separate inside the class?
            for k, want in (("True", True), ("False", False)):
                s = im2.filter(pl.col(flag) == want)
                yy = s["label"].to_numpy()
                if k in rows and len(np.unique(yy)) == 2:
                    rows[k]["auroc_within"] = round(auroc(yy, s["score"].to_numpy()), 4)
            cls_item[flag] = rows
        blk["item_classes"] = cls_item

        # --- argmax window off the lexically-best document -------------------------
        blk["argmax_off_lexical_doc_share"] = round(float(sf["argmax_off_lexical_doc"].mean()), 4)
        blk["mean_n_win_sent"] = round(float(sf["n_win_sent"].mean()), 2)

        report[subset] = blk
        print(
            f"{subset}: model {blk['auroc_model']:.4f} | lexical {blk['auroc_lexical_tok_containment']:.4f} "
            f"| id-bound {blk['identifier_binding'].get('bound_vs_unbound', {})}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
