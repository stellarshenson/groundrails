"""R19-H162 - pubmedqa corpus characterisation. ANALYSIS ONLY, CPU ONLY.

Loads the frozen pubmedqa gate sample through the banked read path
(R8-H92_decomposed_arena.ARENA.load_subsets, i.e. the R8-H77 loader) and
reports structure: item counts, label balance, sentence counts, window
geometry, and lexical/numeric surface features of the response sentences and
their evidence. Nothing here trains, tunes or selects anything.

Run:  uv run python experiments/grounding-semantic/R19-H162_pubmedqa_explore.py
"""

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R19-H162_pubmedqa_explore.json"
SUBSET = "pubmedqa"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA

WIN = 1500
STRIDE = 750

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PVAL = re.compile(r"\bp\s*[<>=]\s*0?\.\d+", re.IGNORECASE)
_CI = re.compile(r"\b(?:95\s*%\s*)?(?:ci|confidence interval)\b", re.IGNORECASE)
_HEDGE = re.compile(
    r"\b(may|might|could|suggest|suggests|suggested|appear|appears|likely|"
    r"possibly|potential|potentially|seem|seems|indicate|indicates|indicated|"
    r"probably|tend|tends)\b",
    re.IGNORECASE,
)
_NEG = re.compile(
    r"\b(no|not|non|none|neither|nor|without|absence|lack|fail|failed|"
    r"unable|did not|does not|was not|were not|is not|are not)\b",
    re.IGNORECASE,
)
_COMPAR = re.compile(
    r"\b(higher|lower|greater|less|more|fewer|increase|increased|decrease|"
    r"decreased|reduced|elevated|worse|better|significant|significantly|"
    r"associated|correlat|compared|versus|vs\.?)\b",
    re.IGNORECASE,
)
_CAUSAL = re.compile(
    r"\b(cause|causes|caused|causing|due to|because|leads to|led to|"
    r"results in|resulted in|induce|induces|induced|effect of|responsible for)\b",
    re.IGNORECASE,
)


def windows(chunk):
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def main():
    subs = ARENA.load_subsets()
    claims, chunk_lists, y = subs[SUBSET]
    n = len(y)
    print(f"pubmedqa: n={n}  positives={int(y.sum())}  negatives={int((1 - y).sum())}")

    rows = []
    n_pairs = 0
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        wl = [w for k in ks for w in windows(k)]
        ev = " ".join(ks)
        ev_nums = set(_NUM.findall(ev))
        sents = H92.sentences(c)
        n_pairs += len(sents) * len(wl)
        for si, s in enumerate(sents):
            nums = _NUM.findall(s)
            rows.append(
                {
                    "item": i,
                    "label": int(y[i]),
                    "sent_idx": si,
                    "n_sent_item": len(sents),
                    "sentence": s,
                    "n_chars": len(s),
                    "n_nums": len(nums),
                    "n_nums_absent": sum(1 for v in nums if v not in ev_nums),
                    "has_pval": bool(_PVAL.search(s)),
                    "has_ci": bool(_CI.search(s)),
                    "has_hedge": bool(_HEDGE.search(s)),
                    "has_neg": bool(_NEG.search(s)),
                    "has_compar": bool(_COMPAR.search(s)),
                    "has_causal": bool(_CAUSAL.search(s)),
                }
            )

    df = pl.DataFrame(rows)
    df.write_parquet(HERE / "R19-H162_pubmedqa_sents.parquet")

    doc_counts = [len(ks) for ks in chunk_lists]
    doc_chars = [len(k) for ks in chunk_lists for k in ks]
    win_per_item = [sum(len(windows(k)) for k in ks) for ks in chunk_lists]
    resp_chars = [len(c) for c in claims]

    def frac(col, mask=None):
        d = df if mask is None else df.filter(mask)
        return {c: round(float(d[c].mean()), 4) for c in col}

    flags = [
        "has_pval",
        "has_ci",
        "has_hedge",
        "has_neg",
        "has_compar",
        "has_causal",
    ]
    out = {
        "n_items": n,
        "n_pos": int(y.sum()),
        "n_neg": int(n - y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "n_sentences": len(df),
        "n_pairs": n_pairs,
        "sent_per_item": {
            "mean": round(float(np.mean([len(H92.sentences(c)) for c in claims])), 3),
            "median": float(np.median([len(H92.sentences(c)) for c in claims])),
            "max": int(max(len(H92.sentences(c)) for c in claims)),
            "n_single_sentence_items": int(sum(1 for c in claims if len(H92.sentences(c)) == 1)),
        },
        "docs_per_item": {
            "mean": round(float(np.mean(doc_counts)), 3),
            "median": float(np.median(doc_counts)),
            "max": int(max(doc_counts)),
        },
        "doc_chars": {
            "mean": round(float(np.mean(doc_chars)), 1),
            "median": float(np.median(doc_chars)),
            "p90": round(float(np.percentile(doc_chars, 90)), 1),
            "max": int(max(doc_chars)),
            "frac_over_win": round(float(np.mean([c > WIN for c in doc_chars])), 4),
        },
        "windows_per_item": {
            "mean": round(float(np.mean(win_per_item)), 3),
            "median": float(np.median(win_per_item)),
            "max": int(max(win_per_item)),
        },
        "resp_chars": {
            "mean": round(float(np.mean(resp_chars)), 1),
            "median": float(np.median(resp_chars)),
            "max": int(max(resp_chars)),
        },
        "sent_flags_all": frac(flags),
        "sent_flags_pos_items": frac(flags, pl.col("label") == 1),
        "sent_flags_neg_items": frac(flags, pl.col("label") == 0),
        "numerals": {
            "frac_sent_with_num": round(float((df["n_nums"] > 0).mean()), 4),
            "mean_nums_per_sent": round(float(df["n_nums"].mean()), 3),
            "frac_sent_with_absent_num": round(float((df["n_nums_absent"] > 0).mean()), 4),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
