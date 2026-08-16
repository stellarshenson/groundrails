"""R19-H162 - second mechanism pass: provenance concentration and identifier strata.

ANALYSIS ONLY, CPU only. Follows `R19-H162_procedural_mech.py`, which found that a
pure surface-overlap scorer matches or beats the trained cross-encoder on both
procedural subsets. This pass isolates what the model is failing to do:

  provenance_concentration - a supported technical answer is drawn from ONE source
      document; an unsupported one is assembled from fragments of several. Measures
      how concentrated an item's argmax windows are across documents, whether that
      concentration predicts the label on its own, and whether the model uses it
  identifier_strata        - within-stratum AUROC split on whether the response's
      version / CVE / APAR identifiers are present in the evidence at all; the
      model's job is only hard in the stratum where they ARE present but may be
      attached to the wrong claim
  lexical_disagreement     - the items a surface scorer ranks correctly and the
      model does not, listed for reading

Run:  uv run python experiments/grounding-semantic/R19-H162_procedural_mech2.py
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
OUT_JSON = HERE / "R19-H162_procedural_mech2.json"
OUT_TXT = HERE / "R19-H162_procedural_disagree.txt"

SUBSETS = ("emanual", "techqa", "delucionqa")

ID_PATTERNS = [
    re.compile(r"\b[Vv]?\d+\.\d+(?:\.\d+){0,3}\b"),
    re.compile(r"\bCVE-\d{4}-\d{3,7}\b"),
    re.compile(r"\b[A-Z]{2}\d{5}\b"),
]


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA


def ids_in(t):
    out = set()
    for p in ID_PATTERNS:
        out |= {m.strip().lower() for m in p.findall(t)}
    return out


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def main():
    subs = ARENA.load_subsets()
    df = pl.read_parquet(DUMP)
    report, lines = {}, []

    for subset in SUBSETS:
        d = df.filter(pl.col("subset") == subset)
        claims, chunk_lists, _y = subs[subset]

        sent_txt, doc_txt = {}, {}
        for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
            doc_txt[i] = list(ks)
            for si, s in enumerate(H92.sentences(c)):
                sent_txt[(i, si)] = s

        sf = (
            d.group_by(["item_id", "sent_idx"])
            .agg(
                pl.col("logit").max().alias("model_max"),
                pl.col("label").first(),
                pl.col("tok_containment").max().alias("lex_max"),
            )
            .join(
                d.filter(pl.col("is_argmax"))
                .group_by(["item_id", "sent_idx"])
                .agg(pl.col("doc_idx").first().alias("am_doc")),
                on=["item_id", "sent_idx"],
            )
            .sort(["item_id", "sent_idx"])
        )

        # --- provenance concentration -------------------------------------------
        conc = (
            sf.group_by("item_id")
            .agg(
                pl.col("label").first(),
                pl.len().alias("n_sent"),
                pl.col("am_doc").n_unique().alias("n_docs_used"),
                pl.col("am_doc").mode().first().alias("modal_doc"),
                pl.col("model_max").min().alias("model_item"),
                pl.col("lex_max").min().alias("lex_item"),
            )
            .sort("item_id")
        )
        modal_share = (
            sf.join(conc.select(["item_id", "modal_doc"]), on="item_id")
            .with_columns((pl.col("am_doc") == pl.col("modal_doc")).alias("on_modal"))
            .group_by("item_id")
            .agg(pl.col("on_modal").mean().alias("modal_share"))
        )
        conc = conc.join(modal_share, on="item_id").sort("item_id")
        y = conc["label"].to_numpy()

        blk = {
            "n_items": len(y),
            "n_negative": int((y == 0).sum()),
            "auroc_model": round(auroc(y, conc["model_item"].to_numpy()), 5),
            "auroc_lexical": round(auroc(y, conc["lex_item"].to_numpy()), 5),
            "provenance_concentration": {
                "mean_docs_used_supported": round(
                    float(conc.filter(pl.col("label") == 1)["n_docs_used"].mean()), 3
                ),
                "mean_docs_used_unsupported": round(
                    float(conc.filter(pl.col("label") == 0)["n_docs_used"].mean()), 3
                ),
                "mean_modal_share_supported": round(
                    float(conc.filter(pl.col("label") == 1)["modal_share"].mean()), 4
                ),
                "mean_modal_share_unsupported": round(
                    float(conc.filter(pl.col("label") == 0)["modal_share"].mean()), 4
                ),
                "auroc_modal_share_alone": round(auroc(y, conc["modal_share"].to_numpy()), 5),
                "spearman_modal_share_vs_model": round(
                    float(
                        conc.select(pl.corr("modal_share", "model_item", method="spearman"))[0, 0]
                    ),
                    4,
                ),
            },
        }

        # --- identifier strata ---------------------------------------------------
        item_ids, resp_ids, ev_ids = [], [], []
        for i in conc["item_id"].to_list():
            r = set()
            for si in range(len(H92.sentences(claims[i]))):
                r |= ids_in(sent_txt[(i, si)])
            e = set()
            for k in doc_txt[i]:
                e |= ids_in(k)
            item_ids.append(i)
            resp_ids.append(r)
            ev_ids.append(e)
        n_resp = np.array([len(r) for r in resp_ids])
        share_present = np.array(
            [len(r & e) / len(r) if r else -1.0 for r, e in zip(resp_ids, ev_ids, strict=True)]
        )
        conc = conc.with_columns(
            pl.Series("n_resp_ids", n_resp, dtype=pl.Int32),
            pl.Series("id_share_in_evidence", share_present),
        )
        strata = {}
        for name, mask in (
            ("no_identifier_in_response", conc["n_resp_ids"] == 0),
            ("all_identifiers_present_in_evidence", conc["id_share_in_evidence"] >= 0.999),
            (
                "some_identifier_absent_from_evidence",
                (conc["id_share_in_evidence"] >= 0.0) & (conc["id_share_in_evidence"] < 0.999),
            ),
        ):
            s = conc.filter(mask)
            yy = s["label"].to_numpy()
            strata[name] = {
                "n": int(s.height),
                "base_rate_supported": round(float(yy.mean()), 4) if s.height else None,
                "auroc_model": round(auroc(yy, s["model_item"].to_numpy()), 4)
                if s.height and len(np.unique(yy)) == 2
                else None,
                "auroc_lexical": round(auroc(yy, s["lex_item"].to_numpy()), 4)
                if s.height and len(np.unique(yy)) == 2
                else None,
            }
        blk["identifier_strata"] = strata

        # --- lexical / model disagreement ---------------------------------------
        mr = conc["model_item"].rank() / conc.height
        lr = conc["lex_item"].rank() / conc.height
        conc = conc.with_columns(
            pl.Series("model_pct", mr.to_numpy()), pl.Series("lex_pct", lr.to_numpy())
        )
        gap = conc.with_columns((pl.col("lex_pct") - pl.col("model_pct")).alias("gap"))
        # positives the lexical scorer ranks high and the model ranks low, and
        # negatives the model ranks high and lexical ranks low - the two ways the
        # model loses to surface overlap
        fn_like = gap.filter(pl.col("label") == 1).sort("gap", descending=True).head(8)
        fp_like = gap.filter(pl.col("label") == 0).sort("gap").head(8)
        blk["lexical_beats_model"] = {
            "supported_items_lexical_ranks_higher": [
                {
                    "item_id": int(r["item_id"]),
                    "model_pct": round(r["model_pct"], 3),
                    "lex_pct": round(r["lex_pct"], 3),
                }
                for r in fn_like.iter_rows(named=True)
            ],
            "unsupported_items_model_ranks_higher": [
                {
                    "item_id": int(r["item_id"]),
                    "model_pct": round(r["model_pct"], 3),
                    "lex_pct": round(r["lex_pct"], 3),
                }
                for r in fp_like.iter_rows(named=True)
            ],
        }
        lines.append(
            f"\n{'=' * 90}\n### {subset}: items where surface overlap ranks better than the model\n"
        )
        for tag, frame in (
            ("SUPPORTED, model ranks too low", fn_like),
            ("UNSUPPORTED, model ranks too high", fp_like),
        ):
            lines.append(f"\n-- {tag}")
            for r in frame.iter_rows(named=True):
                i = int(r["item_id"])
                lines.append(
                    f"   item {i}  model_pct {r['model_pct']:.2f}  lex_pct {r['lex_pct']:.2f}"
                )
                lines.append(f"     RESPONSE: {claims[i][:600]}")

        report[subset] = blk
        print(
            f"{subset}: model {blk['auroc_model']:.4f} lexical {blk['auroc_lexical']:.4f} "
            f"| modal-share alone {blk['provenance_concentration']['auroc_modal_share_alone']:.4f} "
            f"| strata {[(k, v['auroc_model']) for k, v in strata.items()]}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_TXT.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
