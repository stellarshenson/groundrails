"""R10-H107 data build - procedural-doc-register pairs (zero GPU).

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 10).
Two DANN groups on top of the clean 685,670-pair mix:

  proc_code - KRLabsOrg/lettucedetect-code-hallucination train split (provenance
              gate PASSED: dataset column = 5 original lettucedetect sources -
              wikipedia/code-agent/readme/tool-output/acl - zero psiloqa/ragtruth
              rows, cc-by-4.0; see R10-H107_gate_report.md). Sentence-level pairs:
              a sentence overlapping a hallucination span -> 0, an unmodified
              sentence of the same answer -> 1; evidence = the 1,500-char window
              (stride 750) of the row's context with max token overlap.
  proc_gov  - IBM MultiDoc2Dial train split (Apache-2.0; 488 US government-service
              documents, human grounding-span references per agent turn).
              Positives = (agent utterance, window containing ALL referenced
              spans); non-localisable turns dropped. Negatives = ONE deterministic
              span-anchored corruption (number/threshold swap, condition negation,
              identifier digit swap), filtered so the corrupted value does not
              occur elsewhere in the same document.

Cap: neither group exceeds 2x the other (deterministic downsample, seed 0).
Output: data/external/datasets/R10-H107_pairs.parquet + sidecar .md.

Run:  uv run python experiments/grounding-semantic/R10-H107_data.py
"""

import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
OUT = DATA / "R10-H107_pairs.parquet"
MD2D_ZIP = DATA / "dataset-multidoc2dial.zip"
WIN, STRIDE = 1500, 750
MIN_SENT = 25
SEED = 0

SENT_RE = re.compile(r"(?<=[.!?])\s+")


def sentences_with_offsets(text):
    """Split on terminal punctuation, keep char offsets; min length applies later."""
    out, start = [], 0
    for m in SENT_RE.finditer(text):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(text)))
    return [(a, b) for a, b in out if b > a]


def windows(text):
    if len(text) <= WIN:
        return [text]
    return [text[i : i + WIN] for i in range(0, max(1, len(text) - WIN + 1), STRIDE)]


TOK_RE = re.compile(r"[a-z0-9]+")


def best_window(claim, ctx):
    ws = windows(ctx)
    ctoks = set(TOK_RE.findall(claim.lower()))
    scores = [len(ctoks & set(TOK_RE.findall(w.lower()))) for w in ws]
    return ws[int(np.argmax(scores))]


def build_proc_code():
    shards = [
        hf_hub_download(
            "KRLabsOrg/lettucedetect-code-hallucination",
            f"data/train-0000{i}-of-00003.parquet",
            repo_type="dataset",
        )
        for i in range(3)
    ]
    df = pl.concat([pl.read_parquet(p) for p in shards])
    claims, chunks, ys = [], [], []
    for row in df.iter_rows(named=True):
        ans, ctx = row["answer"], row["context"]
        if not ans or not ctx or len(ctx) < 50:
            continue
        spans = [(l["start"], l["end"]) for l in (row["labels"] or [])]
        for a, b in sentences_with_offsets(ans):
            sent = ans[a:b].strip()
            if len(sent) < MIN_SENT:
                continue
            bad = any(s < b and e > a for s, e in spans)
            claims.append(sent)
            chunks.append(best_window(sent, ctx))
            ys.append(0.0 if bad else 1.0)
    return claims, chunks, ys


NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
NEGATIONS = [
    (re.compile(r"\bover\b"), "under"),
    (re.compile(r"\bunder\b"), "over"),
    (re.compile(r"\bat least\b"), "at most"),
    (re.compile(r"\bat most\b"), "at least"),
    (re.compile(r"\bmore than\b"), "less than"),
    (re.compile(r"\bless than\b"), "more than"),
    (re.compile(r"\bbefore\b"), "after"),
    (re.compile(r"\bafter\b"), "before"),
]


def corrupt(utt, ref_text, doc_text, rng):
    """One deterministic edit anchored in the referenced span. Returns
    (corrupted, family) or None. The corrupted value must not occur elsewhere
    in the document (which would silently re-ground it)."""
    # family 1: number swap - a number present in BOTH utterance and referenced span
    for m in NUM_RE.finditer(utt):
        tok = m.group(0)
        if tok in ref_text:
            raw = tok.replace(",", "")
            try:
                val = int(raw) if "." not in raw else None
            except ValueError:
                val = None
            if val is not None and val > 0:
                new = str(val + int(rng.integers(1, 6)))
                if new not in doc_text:
                    return utt[: m.start()] + new + utt[m.end() :], "number_swap"
    # family 2: condition negation - phrase present in both utterance and span
    for pat, repl in NEGATIONS:
        m = pat.search(utt)
        if m and pat.search(ref_text):
            cand = utt[: m.start()] + repl + utt[m.end() :]
            return cand, "condition_negation"
    # family 3: identifier digit swap (Form SSA-10 / DTF-802 style, in both)
    idm = re.search(r"\b([A-Z]{2,6}-?\d{1,4})\b", utt)
    if idm and idm.group(1) in ref_text:
        ident = idm.group(1)
        digits = re.search(r"\d+", ident)
        new_num = str(int(digits.group(0)) + 1)
        new_ident = ident[: digits.start()] + new_num + ident[digits.end() :]
        if new_ident not in doc_text:
            return utt.replace(ident, new_ident, 1), "identifier_swap"
    return None


def build_proc_gov(rng):
    md_dir = HERE / "_md2d_tmp"
    md_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(MD2D_ZIP) as z:
        z.extractall(md_dir)
    root = md_dir / "multidoc2dial"
    docs = json.load(open(root / "multidoc2dial_doc.json"))["doc_data"]
    dial = json.load(open(root / "multidoc2dial_dial_train.json"))["dial_data"]
    claims, chunks, ys, fams = [], [], [], []
    for domain, dialogues in dial.items():
        for dd in dialogues:
            for t in dd["turns"]:
                if t["role"] != "agent" or not t.get("references"):
                    continue
                utt = t["utterance"].strip()
                if len(utt) < MIN_SENT:
                    continue
                doc_id = t["references"][0]["doc_id"]
                doc = docs[domain].get(doc_id)
                if doc is None:
                    continue
                sps = [
                    doc["spans"][r["id_sp"]]
                    for r in t["references"]
                    if r["doc_id"] == doc_id and r["id_sp"] in doc["spans"]
                ]
                if not sps:
                    continue
                a = min(s["start_sp"] for s in sps)
                b = max(s["end_sp"] for s in sps)
                if b - a > WIN - 100:
                    continue  # non-localisable in one window - dropped per registration
                mid = (a + b) // 2
                lo = max(0, min(mid - WIN // 2, len(doc["doc_text"]) - WIN))
                window = doc["doc_text"][lo : lo + WIN]
                claims.append(utt)
                chunks.append(window)
                ys.append(1.0)
                fams.append("positive")
                ref_text = doc["doc_text"][a:b]
                c = corrupt(utt, ref_text, doc["doc_text"], rng)
                if c is not None:
                    claims.append(c[0])
                    chunks.append(window)
                    ys.append(0.0)
                    fams.append(c[1])
    return claims, chunks, ys, fams


def main():
    rng = np.random.default_rng(SEED)
    print("== proc_gov (multidoc2dial train) ==", flush=True)
    gc, gk, gy, gf = build_proc_gov(rng)
    n_gov = len(gc)
    fam_counts = {f: gf.count(f) for f in sorted(set(gf))}
    print(f"proc_gov pairs {n_gov}  label mean {np.mean(gy):.3f}  families {fam_counts}")

    print("== proc_code (lettucedetect-code train) ==", flush=True)
    cc, ck, cy = build_proc_code()
    print(f"proc_code raw pairs {len(cc)}  label mean {np.mean(cy):.3f}")
    cap = 2 * n_gov
    if len(cc) > cap:
        idx = rng.choice(len(cc), size=cap, replace=False)
        idx.sort()
        cc = [cc[i] for i in idx]
        ck = [ck[i] for i in idx]
        cy = [cy[i] for i in idx]
        print(f"proc_code capped to 2x proc_gov = {cap}")

    df = pl.DataFrame(
        {
            "claim": cc + gc,
            "chunk": ck + gk,
            "label": [float(v) for v in cy] + [float(v) for v in gy],
            "tag": ["proc_code"] * len(cc) + ["proc_gov"] * len(gc),
        }
    )
    df.write_parquet(OUT)
    print(f"\nwrote {OUT}  rows {len(df)}")
    for tag in ["proc_code", "proc_gov"]:
        sub = df.filter(pl.col("tag") == tag)
        print(f"  {tag:10s} n={len(sub)}  label mean {sub['label'].mean():.3f}")

    # QA eyeball - 10 random examples per (group, label)
    print("\n== QA examples ==")
    for tag in ["proc_code", "proc_gov"]:
        for lab in [1.0, 0.0]:
            sub = df.filter((pl.col("tag") == tag) & (pl.col("label") == lab))
            take = sub.sample(min(10, len(sub)), seed=SEED)
            print(f"\n-- {tag} label={lab} ({len(sub)} rows) --")
            for r in take.head(10).iter_rows(named=True):
                print(f"  CLAIM: {r['claim'][:140]!r}")
    # contamination spot-check: no RAGBench-source markers
    banned = ["ragbench", "delucionqa", "emanual", "techqa", "pubmedqa", "hagrid",
              "covidqa", "expertqa", "finqa", "tatqa", "hotpotqa"]
    hits = sum(df["claim"].str.to_lowercase().str.contains(b).sum() for b in banned)
    print(f"\nbanned-marker hits in claims (expect ~0): {hits}")
    print("=== R10-H107 DATA DONE ===")


if __name__ == "__main__":
    main()
