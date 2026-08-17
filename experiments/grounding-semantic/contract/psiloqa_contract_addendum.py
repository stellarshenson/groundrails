"""DATASET CONTRACT addendum - member `psiloqa`. CPU ONLY.

Three measurements the first pass left open, merged into
`psiloqa_contract_report.json`:

  C1 robustness  the two containment instruments straddle the C1 rejection band
                 (banked ASCII delta 0.1091 outside it, Unicode delta 0.0930
                 inside it), so the separation is re-measured scale-free (AUROC)
                 and by claim length, where the instrument has resolution
  C1 bar reading the clause sentence admits a second reading - "negatives are
                 >= 90% attested" as a LEVEL on the negative leg rather than a
                 threshold on containment. Both readings are computed so the
                 verdict rests on stated numbers, not on my choice of parse
  C3 split axis  the first pass measured passage reuse (94.4%) and question
                 reuse (11.7%); 11.7% is not zero, so the axis is pinned down
                 at (passage, question) and (passage, question, answer) level

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/psiloqa_contract_addendum.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util as _ilu
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
DATA = EXP.parent.parent / "data" / "external" / "datasets"
ARCHIVE = DATA / "dataset-psiloqa.zip"
OUT = HERE / "psiloqa_contract_report.json"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MAIN = _mod("psicontract", HERE / "psiloqa_contract.py")
PM = MAIN.PM


def splits():
    z = zipfile.ZipFile(ARCHIVE)
    return {n.split("__")[-1].replace(".parquet", ""):
            pl.read_parquet(io.BytesIO(z.read(n)))
            for n in z.namelist() if n.endswith(".parquet")}


def c1_addendum(sp):
    tr = sp["train"].filter((pl.col("wiki_passage").str.len_chars() > 50)
                            & (pl.col("llm_answer").str.len_chars() > 10))
    claims = tr["llm_answer"].to_list()
    chunks = tr["wiki_passage"].to_list()
    langs = tr["lang"].to_list()
    y = (tr["labels"].list.len() == 0).cast(pl.Int8).to_numpy()

    n = len(y)
    cont_a = np.zeros(n)
    cont_u = np.zeros(n)
    cov_a = np.zeros(n, dtype=bool)
    ntok = np.zeros(n, dtype=np.int32)
    for i, (cl, ch) in enumerate(zip(claims, chunks, strict=True)):
        ca, ea = PM.content(cl), PM.content(ch)
        cov_a[i] = bool(ca)
        cont_a[i] = PM.containment(ca, ea)
        cu, eu = MAIN.content_u(cl), MAIN.content_u(ch)
        ntok[i] = len(cu)
        cont_u[i] = PM.containment(cu, eu)

    def auroc(v, mask=None):
        m = np.ones(n, dtype=bool) if mask is None else mask
        if len(set(y[m].tolist())) < 2:
            return None
        return round(float(roc_auc_score(y[m], v[m])), 4)

    bands = {"1-2": (1, 2), "3-4": (3, 4), "5-9": (5, 9), "10-24": (10, 24),
             "25+": (25, 10**9)}
    by_band = {}
    for name, (lo, hi) in bands.items():
        m = (ntok >= lo) & (ntok <= hi)
        if not m.any():
            continue
        p, q = cont_u[m & (y == 1)], cont_u[m & (y == 0)]
        by_band[name] = {
            "rows": int(m.sum()), "n_pos": int(p.size), "n_neg": int(q.size),
            "rate_ge_0.90_pos": round(float((p >= 0.90).mean()), 4) if p.size else None,
            "rate_ge_0.90_neg": round(float((q >= 0.90).mean()), 4) if q.size else None,
            "delta": round(abs(float((p >= 0.90).mean()) - float((q >= 0.90).mean())), 4)
                     if p.size and q.size else None,
            "auroc_containment_vs_label": auroc(cont_u, m),
        }

    by_lang = {}
    for lg in sorted(set(langs)):
        m = np.array([x == lg for x in langs])
        by_lang[lg] = {"rows": int(m.sum()), "auroc_containment_vs_label": auroc(cont_u, m)}

    a_pos_u = float((cont_u[y == 1] >= 0.90).mean())
    a_neg_u = float((cont_u[y == 0] >= 0.90).mean())
    a_pos_a = float((cont_a[cov_a & (y == 1)] >= 0.90).mean())
    a_neg_a = float((cont_a[cov_a & (y == 0)] >= 0.90).mean())

    return {
        "why": "the two containment instruments straddle the C1 rejection band; these "
               "measurements state the separation without depending on which one is read",
        "auroc_containment_vs_label": {
            "unicode_all_rows": auroc(cont_u),
            "banked_ascii_scorable_rows": auroc(cont_a, cov_a),
            "reading": "0.5 is the R20-H175b failure signature (both legs identically "
                       "attested); above 0.5 means positives are more attested than "
                       "negatives, the direction a support label must have",
        },
        "by_claim_content_token_band_unicode": by_band,
        "by_language_unicode": by_lang,
        "bar_reading_A_difference_of_attestation_rates": {
            "statement": "REJECT if |rate(containment >= 0.90)_neg - rate_pos| <= 0.10",
            "unicode": {"rate_neg": round(a_neg_u, 4), "rate_pos": round(a_pos_u, 4),
                        "delta": round(abs(a_pos_u - a_neg_u), 4),
                        "rejected": bool(abs(a_pos_u - a_neg_u) <= 0.10),
                        "margin_to_band": round(abs(a_pos_u - a_neg_u) - 0.10, 4)},
            "banked_ascii": {"rate_neg": round(a_neg_a, 4), "rate_pos": round(a_pos_a, 4),
                             "delta": round(abs(a_pos_a - a_neg_a), 4),
                             "rejected": bool(abs(a_pos_a - a_neg_a) <= 0.10),
                             "margin_to_band": round(abs(a_pos_a - a_neg_a) - 0.10, 4)},
            "note": "this is the reading under which R20-H175b's lane is rejected "
                    "(its two legs read 0.723 each, delta 0.000)",
        },
        "bar_reading_B_level_on_the_negative_leg": {
            "statement": "REJECT only if the negative leg is itself >= 90% attested "
                         "AND within 0.10 of the positive leg",
            "unicode_rate_neg": round(a_neg_u, 4),
            "banked_ascii_rate_neg": round(a_neg_a, 4),
            "rejected_unicode": bool(a_neg_u >= 0.90 and abs(a_pos_u - a_neg_u) <= 0.10),
            "rejected_banked_ascii": bool(a_neg_a >= 0.90 and abs(a_pos_a - a_neg_a) <= 0.10),
            "note": "under this reading R20-H175b's lane (rate_neg 0.723) would NOT be "
                    "rejected, so reading A is the one that catches the case the clause "
                    "was written for; reading B is recorded only to show the parse was "
                    "considered and not silently chosen",
        },
        "absolute_risk_the_clause_targets": {
            "negatives_fully_attested_unicode": int((cont_u[y == 0] >= 0.99999).sum()),
            "negatives_fully_attested_share_unicode":
                round(float((cont_u[y == 0] >= 0.99999).mean()), 4),
            "positive_to_negative_fully_attested_ratio_unicode":
                round(float((cont_u[y == 1] >= 0.99999).mean())
                      / max(float((cont_u[y == 0] >= 0.99999).mean()), 1e-9), 2),
            "reference_R20_H175b_lane": "66.4% of negatives fully attested, 72.3% at "
                                        ">= 0.90, both legs mean 0.9129 (canonical log, "
                                        "2026-08-17 ~09:15 entry)",
        },
        "claim_content_token_length_unicode": {
            "median": float(np.median(ntok)),
            "mean": round(float(ntok.mean()), 2),
            "share_le_2_tokens": round(float((ntok <= 2).mean()), 4),
            "share_le_4_tokens": round(float((ntok <= 4).mean()), 4),
        },
    }


def c3_addendum(sp):
    tr, va, te = sp["train"], sp["validation"], sp["test"]
    ho = pl.concat([va, te])

    tr_pq = set(zip(tr["wiki_passage"].to_list(), tr["question"].to_list(), strict=True))
    tr_pqa = set(zip(tr["wiki_passage"].to_list(), tr["question"].to_list(),
                     tr["llm_answer"].to_list(), strict=True))
    tr_qc = set(zip(tr["question"].to_list(), tr["llm_checkpoint"].to_list(), strict=True))
    tr_id = set(tr["id"].to_list())

    ho_pq = sorted(set(zip(ho["wiki_passage"].to_list(), ho["question"].to_list(),
                           strict=True)))
    ho_pqa = sorted(set(zip(ho["wiki_passage"].to_list(), ho["question"].to_list(),
                            ho["llm_answer"].to_list(), strict=True)))
    ho_qc = sorted(set(zip(ho["question"].to_list(), ho["llm_checkpoint"].to_list(),
                          strict=True)))
    ho_id = sorted(set(ho["id"].to_list()))

    def share(hits, tot):
        return {"hits": hits, "of_units": tot, "share": round(hits / tot, 4) if tot else None}

    return {
        "held_out_passage_question_pairs_also_in_train":
            share(sum(1 for k in ho_pq if k in tr_pq), len(ho_pq)),
        "held_out_passage_question_answer_triples_also_in_train":
            share(sum(1 for k in ho_pqa if k in tr_pqa), len(ho_pqa)),
        "held_out_question_checkpoint_pairs_also_in_train":
            share(sum(1 for k in ho_qc if k in tr_qc), len(ho_qc)),
        "held_out_row_ids_also_in_train":
            share(sum(1 for k in ho_id if k in tr_id), len(ho_id)),
        "reading": "the cut is at the (passage, question) level - the same passage is "
                   "re-used across splits with different questions - and it is not even "
                   "a clean question cut: a question recurs across the boundary whenever "
                   "the corpus kept two answers to it from different LLM checkpoints",
    }


def main():
    rep = json.loads(OUT.read_text())
    sp = splits()

    print("=== C1 addendum", flush=True)
    rep["C1"]["addendum"] = c1_addendum(sp)
    a = rep["C1"]["addendum"]
    print(json.dumps({"auroc": a["auroc_containment_vs_label"],
                      "reading_A": a["bar_reading_A_difference_of_attestation_rates"],
                      "length": a["claim_content_token_length_unicode"]}, indent=2),
          flush=True)

    print("=== C3 addendum", flush=True)
    rep["C3"]["addendum"] = c3_addendum(sp)
    print(json.dumps(rep["C3"]["addendum"], indent=2), flush=True)

    # C1 verdict restated on the full-coverage instrument, both readings on record
    ra = a["bar_reading_A_difference_of_attestation_rates"]
    rep["C1"]["verdict"] = "FAIL" if ra["unicode"]["rejected"] else "PASS"
    rep["C1"]["verdict_basis"] = (
        "reading A of the clause bar (the one that rejects R20-H175b's lane), evaluated "
        "on the Unicode instrument, the only one that scores all 61,712 rows of a "
        "14-language member. Measured delta "
        f"{ra['unicode']['delta']} against the <= 0.10 rejection band - inside it by "
        f"{abs(ra['unicode']['margin_to_band'])}. The banked ASCII instrument reads "
        f"{ra['banked_ascii']['delta']}, outside the band by "
        f"{abs(ra['banked_ascii']['margin_to_band'])}, on 86.9% of rows. THE TWO "
        "INSTRUMENTS STRADDLE THE BAR: the verdict is taken on the full-coverage one "
        "because a PASS that depends on dropping 13.1% of the member is not defensible")
    rep["clause_verdicts"]["C1"] = rep["C1"]["verdict"]
    rep["conforming"] = all(v in ("PASS", "NOT-APPLICABLE")
                            for v in rep["clause_verdicts"].values())
    rep["failed_clauses"] = sorted(k for k, v in rep["clause_verdicts"].items()
                                   if v == "FAIL")
    OUT.write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps({"clause_verdicts": rep["clause_verdicts"],
                             "conforming": rep["conforming"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
