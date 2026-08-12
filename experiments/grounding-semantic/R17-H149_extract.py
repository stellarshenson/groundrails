"""R17-H149 stage 1 - PROSE passage extraction (bare-assertion prose lane, kill-gate).

Emits `R17-H149_passages.parquet`: one row per prose passage (a multi-sentence
paragraph of running text) from the staged non-RAGBench corpora, plus
`R17-H149_census.json`.

Deliberately the complement of the H148 extractor: enumerated / list-structured
blocks are REJECTED here, running prose is kept.

Sources: SciFact upstream abstracts (staged, CC BY 4.0 / ODC-By), FAA AMT
handbooks and army-tm PDFs (public domain, 17 U.S.C. 105).  None is a RAGBench
source corpus; none is in the clean training mix (RAGTruth / HaluEval / PsiloQA /
VitaminC).

CPU only.  PyMuPDF lives in the conda interpreter, not in .venv:
  /opt/conda/bin/python experiments/grounding-semantic/R17-H149_extract.py
"""
import json
import pathlib
import re

import fitz
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "R17-H149_passages.parquet"
CENSUS = HERE / "R17-H149_census.json"

ITEM = re.compile(r"^\s*(?:\((\d{1,2})\)|(\d{1,2})[.)]|(?:STEP|Step)\s*\d|[-•·*]\s)")
CAPTION = re.compile(r"^\s*(?:Figure|Table|Chapter|Section)\s+[\dA-Z]")
HEADING = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,'()/&-]{3,70}$")
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
ALPHA = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[A-Za-z]{2,}")

MIN_CHARS = 240
MAX_CHARS = 2400
MIN_SENTS = 2


def sentences(text):
    out = []
    for s in SENT_END.split(text):
        s = " ".join(s.split())
        if s:
            out.append(s)
    return out


def clean_para(p):
    """Normalise a candidate paragraph; return None if it is not running prose."""
    p = " ".join(p.replace("­", "").split())
    p = re.sub(r"(\w)-\s(\w)", r"\1\2", p)                    # de-hyphenate line breaks
    if not (MIN_CHARS <= len(p) <= MAX_CHARS):
        return None
    letters = len(ALPHA.findall(p))
    if letters / len(p) < 0.72:                                # tables / part-number soup
        return None
    words = WORD.findall(p)
    if len(words) < 60:
        return None
    sents = sentences(p)
    if len(sents) < MIN_SENTS:
        return None
    if any(len(s) < 20 for s in sents[:-1]):                   # heading swept into the run
        return None
    return p, sents


def pdf_passages(path):
    """Running-prose paragraphs of a PDF: line runs with no list markers."""
    out = []
    try:
        doc = fitz.open(path)
    except Exception:                                          # noqa: BLE001
        return out
    for page in doc:
        try:
            raw = page.get_text("text")
        except Exception:                                      # noqa: BLE001
            continue
        run = []
        for ln in raw.split("\n"):
            ln = " ".join(ln.split())
            drop = (not ln or CAPTION.match(ln) or ITEM.match(ln)
                    or len(ln) < 30 and HEADING.match(ln))
            if drop:
                if run:
                    got = clean_para(" ".join(run))
                    if got:
                        out.append(got)
                    run = []
                continue
            run.append(ln)
        if run:
            got = clean_para(" ".join(run))
            if got:
                out.append(got)
    doc.close()
    return out


def main():
    rows, census = [], {}

    # ---- SciFact abstracts (already sentence-split upstream) -----------------
    n_abs, kept = 0, 0
    with open(DATA / "scifact" / "data" / "corpus.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            n_abs += 1
            sents = [" ".join(s.split()) for s in rec["abstract"] if s.strip()]
            text = " ".join(sents)
            if len(sents) < MIN_SENTS or not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            kept += 1
            rows.append({"corpus": "scifact", "doc_id": f"scifact:{rec['doc_id']}",
                         "passage_id": f"scifact:{rec['doc_id']}:0", "title": rec["title"],
                         "text": text, "n_sent": len(sents)})
    census["scifact"] = {"abstracts": n_abs, "passages": kept,
                         "licence": "CC BY 4.0 (claims) + ODC-By (abstracts), upstream AI2"}

    # ---- FAA AMT handbooks and army-tm PDFs ---------------------------------
    for corpus, sub in (("faa-amt", "faa-amt"), ("army-tm", "army-tm")):
        pdfs = sorted((DATA / sub / "pdf").glob("*.pdf"))
        n_pass, docs = 0, 0
        for p in pdfs:
            got = pdf_passages(p)
            if got:
                docs += 1
            for k, (text, sents) in enumerate(got):
                rows.append({"corpus": corpus, "doc_id": f"{corpus}:{p.stem}",
                             "passage_id": f"{corpus}:{p.stem}:{k}", "title": p.stem,
                             "text": text, "n_sent": len(sents)})
                n_pass += 1
        census[corpus] = {"pdfs": len(pdfs), "docs_with_passages": docs,
                          "passages": n_pass, "licence": "public domain (17 U.S.C. 105)"}
        print(f"{corpus}: {len(pdfs)} pdfs -> {n_pass} prose passages", flush=True)

    df = pl.DataFrame(rows)
    # dedupe identical passages (running headers / repeated boilerplate)
    df = df.unique(subset=["text"], keep="first").sort("passage_id")
    census["total_passages"] = df.height
    census["total_documents"] = df["doc_id"].n_unique()
    census["after_dedupe"] = {c: int(v) for c, v in
                              df.group_by("corpus").len().iter_rows()}
    df.write_parquet(OUT)
    CENSUS.write_text(json.dumps(census, indent=2))
    print(json.dumps(census, indent=2), flush=True)


if __name__ == "__main__":
    main()
