"""DATASET CONTRACT fixability probe - member `psiloqa`. CPU ONLY.

The contract's verification output requires each FAIL to say whether it is
fixable by a conforming pipeline or is a corpus property. That answer is
MEASURED here, not asserted:

  C1  the pooled attestation-rate delta under row subsets a filtering pipeline
      could select, so "pipeline-fixable" rests on a number
  C2  the member rows and passages that would have to leave for the member to
      read zero against every evaluation surface

Nothing here is a proposal. It is the counterfactual the FAIL classification
needs. Merged into `psiloqa_contract_report.json`.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/psiloqa_contract_fixability.py
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


def main():
    rep = json.loads(OUT.read_text())
    z = zipfile.ZipFile(ARCHIVE)
    tr = pl.read_parquet(io.BytesIO(
        z.read("s-nlp__PsiloQA__train.parquet"))).filter(
        (pl.col("wiki_passage").str.len_chars() > 50)
        & (pl.col("llm_answer").str.len_chars() > 10))
    y = (tr["labels"].list.len() == 0).cast(pl.Int8).to_numpy()
    claims, chunks = tr["llm_answer"].to_list(), tr["wiki_passage"].to_list()
    langs = tr["lang"].to_list()

    n = len(y)
    cont = np.zeros(n)
    ntok = np.zeros(n, dtype=np.int32)
    for i, (cl, ch) in enumerate(zip(claims, chunks, strict=True)):
        cu = MAIN.content_u(cl)
        ntok[i] = len(cu)
        cont[i] = PM.containment(cu, MAIN.content_u(ch))

    def rates(mask):
        p, q = cont[mask & (y == 1)], cont[mask & (y == 0)]
        if not p.size or not q.size:
            return None
        rp, rq = float((p >= 0.90).mean()), float((q >= 0.90).mean())
        return {"rows": int(mask.sum()), "n_pos": int(p.size), "n_neg": int(q.size),
                "rate_ge_0.90_pos": round(rp, 4), "rate_ge_0.90_neg": round(rq, 4),
                "delta": round(abs(rp - rq), 4),
                "clears_the_0.10_band": bool(abs(rp - rq) > 0.10),
                "rows_retained_share": round(float(mask.mean()), 4)}

    subsets = {
        "as_loaded_all_rows": np.ones(n, dtype=bool),
        "drop_claims_over_24_content_tokens": ntok <= 24,
        "drop_claims_over_24_and_under_3_content_tokens": (ntok <= 24) & (ntok >= 3),
        "english_only": np.array([x == "en" for x in langs]),
        "drop_claims_over_24_content_tokens_english_only":
            (ntok <= 24) & np.array([x == "en" for x in langs]),
    }
    probe = {name: rates(m) for name, m in subsets.items()}

    rep["C1"]["fixability_probe"] = {
        "what_this_is": "the C1 rejection band evaluated on row subsets a filtering "
                        "pipeline could select, so the FAIL's fixability is measured "
                        "rather than asserted. NOT a proposal - no queue change is "
                        "recommended here",
        "instrument": "Unicode content-token containment, the full-coverage instrument "
                      "the C1 verdict is taken on",
        "subsets": probe,
        "reading": "the band is tripped because BOTH legs are weakly attested, not "
                   "because the negatives are strongly attested (0.0273); the delta "
                   "grows wherever the instrument has resolution",
    }

    # C2 - what would have to leave
    c2 = rep["C2"]["per_surface"]
    dirty = {k: v for k, v in c2.items() if v["status"] == "OVERLAPS"}
    total_pass = rep["C2"]["member_distinct_evidence_units"]
    worst = {k: v["worst_intersection_any_form_any_pairing"] for k, v in dirty.items()}
    union_note = ("the two overlapping surfaces are both built from this corpus's own "
                  "train split, so their passages are a subset of the member's")
    rep["C2"]["fixability_probe"] = {
        "what_this_is": "the size of the removal a conforming pipeline would need on "
                        "either side of the collision. NOT a proposal",
        "member_distinct_passages": total_pass,
        "overlapping_surfaces": worst,
        "member_passage_share_involved": {
            k: round(v / total_pass, 6) for k, v in worst.items()},
        "note": union_note,
    }
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps({"C1_fixability": probe,
                      "C2_fixability": rep["C2"]["fixability_probe"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
