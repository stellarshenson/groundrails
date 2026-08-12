"""R17-H148 stage 1 - procedural block extraction from the staged R14-H136 PDFs.

Emits `R17-H148_blocks.parquet`: one row per enumerated procedural list block
(>= 3 consecutively numbered items) found in the army-tm / faa-amt corpora, with
the block's heading, its items and the OCR-quality flags used to filter it.

CPU only.  Run with the conda interpreter (PyMuPDF lives there, not in .venv):
  nohup setsid /opt/conda/bin/python experiments/grounding-semantic/R17-H148_extract.py \
      > logs/R17-H148_gate.log 2>&1 &
"""
import json
import pathlib
import re

import fitz
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "R17-H148_blocks.parquet"
CENSUS = HERE / "R17-H148_census.json"

ITEM = re.compile(r"^\s*(?:\((\d{1,2})\)|(\d{1,2})[.)]|(?:STEP|Step)\s*(\d{1,2})[.:)]?)\s*(.*)$")
NOISE = re.compile(r"^\s*(?:Figure|Table|Chapter)\s+[\dA-Z]|^\s*[\d-]{1,10}\s*$|^\s*$")
CAPTION = re.compile(r"^\s*(?:Figure|Table)\s+[\d-]+[.\]]")
ALPHA = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[A-Za-z]{2,}")

MIN_ITEMS = 3
MAX_ITEMS = 12
MIN_ITEM_CHARS = 15
MAX_ITEM_CHARS = 260


def clean_lines(text):
    out = []
    for ln in text.replace("\t", " ").split("\n"):
        ln = " ".join(ln.split())
        if not ln or CAPTION.match(ln):
            continue
        out.append(ln)
    return out


def heading_for(lines, i):
    """Nearest preceding non-item line that looks like a heading or lead-in."""
    for j in range(i - 1, max(i - 4, -1), -1):
        ln = lines[j]
        if ITEM.match(ln) or NOISE.match(ln):
            continue
        if 8 <= len(ln) <= 160:
            return ln
    return ""


def blocks_of(text):
    lines = clean_lines(text)
    marks = []                                   # (line_index, n, inline_body)
    for i, ln in enumerate(lines):
        m = ITEM.match(ln)
        if m:
            n = int(m.group(1) or m.group(2) or m.group(3))
            marks.append((i, n, (m.group(4) or "").strip()))
    out, run = [], []
    for k, (i, n, body) in enumerate(marks):
        if run and n == run[-1][1] + 1:
            run.append((i, n, body))
        else:
            if len(run) >= MIN_ITEMS:
                out.append(run)
            run = [(i, n, body)] if n in (1, 2) else []
    if len(run) >= MIN_ITEMS:
        out.append(run)

    blocks = []
    for run in out:
        items, ok = [], True
        for k, (i, n, body) in enumerate(run):
            end = run[k + 1][0] if k + 1 < len(run) else min(i + 8, len(lines))
            tail = [lines[j] for j in range(i + 1, end) if not ITEM.match(lines[j])]
            txt = " ".join([body] + tail).strip()
            txt = " ".join(txt.split())[:MAX_ITEM_CHARS]
            if len(txt) < MIN_ITEM_CHARS:
                ok = False
                break
            items.append((n, txt))
        if not ok or not (MIN_ITEMS <= len(items) <= MAX_ITEMS):
            continue
        blocks.append({"heading": heading_for(lines, run[0][0]), "items": items})
    return blocks


def ocr_ok(s):
    toks = WORD.findall(s)
    if len(toks) < 12:
        return False
    long_frac = sum(1 for t in toks if len(t) > 17) / len(toks)
    alpha_frac = len(ALPHA.findall(s)) / max(len(s), 1)
    return long_frac < 0.03 and alpha_frac > 0.60


def main():
    rows, census = [], {}
    for corpus in ("faa-amt", "army-tm"):
        pdfs = sorted((DATA / corpus / "pdf").glob("*.pdf"))
        c = {"pdfs": len(pdfs), "pages": 0, "blocks_raw": 0, "blocks_clean": 0,
             "docs_with_blocks": 0, "items": 0}
        for p in pdfs:
            d = fitz.open(p)
            c["pages"] += d.page_count
            nb = 0
            for pi in range(d.page_count):
                for b in blocks_of(d[pi].get_text()):
                    c["blocks_raw"] += 1
                    joined = " ".join(t for _, t in b["items"])
                    if not ocr_ok(joined):
                        continue
                    nb += 1
                    rows.append({
                        "corpus": corpus, "doc_id": p.stem, "page": pi,
                        "heading": b["heading"],
                        "item_numbers": [n for n, _ in b["items"]],
                        "item_texts": [t for _, t in b["items"]],
                        "n_items": len(b["items"]),
                    })
            d.close()
            if nb:
                c["docs_with_blocks"] += 1
        c["blocks_clean"] = sum(1 for r in rows if r["corpus"] == corpus)
        c["items"] = sum(r["n_items"] for r in rows if r["corpus"] == corpus)
        census[corpus] = c
        print(corpus, c, flush=True)

    df = pl.DataFrame(rows)
    df.write_parquet(OUT)
    census["total_blocks"] = df.height
    census["total_documents"] = df["doc_id"].n_unique()
    CENSUS.write_text(json.dumps(census, indent=2))
    print(json.dumps(census, indent=2), flush=True)


if __name__ == "__main__":
    main()
