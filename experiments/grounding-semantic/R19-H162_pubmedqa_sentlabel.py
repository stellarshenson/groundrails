"""R19-H162 - pubmedqa TRUE sentence-level diagnosis. ANALYSIS ONLY, CPU ONLY.

The R12 label-ceiling work established that RAGBench's `adherence_score` is
exactly "`unsupported_response_sentence_keys` is empty", so the corpus carries
per-SENTENCE support truth. This script aligns those annotated sentences to the
H92 splitter's sentences (the ones the read actually scores), attaches the
banked per-sentence logits from the R19-H161 dump, and asks the question the
item-level read cannot: does the verifier separate a SUPPORTED response
sentence from an UNSUPPORTED one on pubmedqa, and does trivial lexical overlap
do it better?

It also classifies the annotator's own free-text explanation of every
unsupported sentence into mechanism families. The explanation is written by the
RAGBench annotator, not by this analysis, so the taxonomy is grounded in the
corpus's own account of why a sentence failed rather than in a guess.

Nothing here trains, tunes or selects on arena statistics.

Run:  uv run python experiments/grounding-semantic/R19-H162_pubmedqa_sentlabel.py
"""

import difflib
import importlib.util
import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT = HERE / "R19-H162_pubmedqa_sentlabel.json"
ARCHIVE = ROOT / "data" / "external" / "datasets" / "dataset-ragbench.zip"
SUBSETS = [
    "covidqa",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "pubmedqa",
    "tatqa",
    "techqa",
]


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA

# --- explanation taxonomy -----------------------------------------------------
# Patterns are read against the ANNOTATOR's explanation of an unsupported
# sentence. Order matters: the first family that matches wins, so the more
# specific signatures are listed first. Multi-label counts are reported too.

FAMILIES = [
    (
        "contradiction",
        re.compile(
            r"\b(contradict\w*|directly\s+contradicted|explicitly\s+states\s+that\b"
            r"[^.]{0,60}\bnot\b|in\s+fact,?\s+the\b|"
            r"incorrect(ly)?\s+(interprets?|correlates?|suggests?|states?)|"
            r"wrongly\s+attributes?|misstat\w+|misattribut\w+)",
            re.IGNORECASE,
        ),
    ),
    (
        "entity_substitution",
        re.compile(
            r"\b(is\s+about\s+the\s+\w+,?\s+not\s+the|a\s+different\s+"
            r"(procedure|context|population|study|condition|setting)|"
            r"not\s+the\s+\w+\b[^.]{0,30}\bbut\b|"
            r"in\s+a\s+different\s+context|"
            r"does\s+not\s+(discuss|address|assess)\s+\w+\s+(explicitly|specifically)|"
            r"specific\s+to\s+Document\s+\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "aim_vs_finding",
        re.compile(
            r"(no|not|without|lacks?|absen\w+)[^.]{0,80}\b(results?|findings?|"
            r"outcomes?)\b|"
            r"\b(results?|findings?|outcomes?)\s+(are|is|were|was)\s+(not|never)\s+"
            r"(provided|presented|given|reported|stated|shown)|"
            r"\b(hypothes\w+|aim|objective|purpose|study\s+design)\b[^.]{0,100}"
            r"\b(but|however|no\s|not\s)|"
            r"without\s+results|no\s+results|results\s+presented|"
            r"\bdoes\s+not\s+(reveal|report|present|provide)\s+the\s+results?\b|"
            r"\bgoes\s+beyond\s+documented\s+findings?\b|"
            r"no\s+specific\s+results?\s+are\s+cited|"
            r"\bmethodolog\w+|data\s+collection\s+methods",
            re.IGNORECASE,
        ),
    ),
    (
        "relation_not_attested",
        re.compile(
            r"\b(no\s+(direct\s+)?link|does\s+not\s+(specifically\s+)?"
            r"(connect|link|relate)|do\s+not\s+link|not\s+link\w*|"
            r"assumes?\s+a\s+(direct\s+)?(link|relevance|connection)|"
            r"no\s+discussion\s+of\s+this|does\s+not\s+provide\s+a\s+direct\s+"
            r"comparison|lacks?\s+specific\s+comparisons?|"
            r"correlates?\s+\w+[^.]{0,60}\bnot\s+confirmed|"
            r"complementar\w+|does\s+not\s+support\s+the\s+(discussion|claim)\s+of|"
            r"connectedness|not\s+specifically\s+connect)",
            re.IGNORECASE,
        ),
    ),
    (
        "scope_overextension",
        re.compile(
            r"\b(overextend\w*|partial(ly)?\s+support\w*|does\s+not\s+cover\s+all|"
            r"broader\s+(than|interpretation)|not\s+conclusively|clear\s+superiority|"
            r"does\s+not\s+fully\s+represent|generic\s+and\s+broader|"
            r"generalizes?|extends?\s+beyond\s+the\s+scope|"
            r"broadly\s+asserts?|does\s+not\s+provide\s+specifics)",
            re.IGNORECASE,
        ),
    ),
    (
        "inference_not_stated",
        re.compile(
            r"\b(infer\w*|extrapolat\w+|speculat\w+|plausible|logical\w*|"
            r"presum\w+|implied|implies|imply|hypothetical|"
            r"reasonable\s+interpretation|appropriate\s+interpretation|"
            r"an?\s+interpretation)\b|"
            r"\bnot\s+(explicitly|directly|conclusively|expressly|specifically)\s+"
            r"(stated|said|mentioned|supported|cited|substantiated|backed|"
            r"addressed|advocate)|"
            r"\bgoes\s+beyond\b|\bnot\s+stated\s+outright\b|"
            r"\bno\s+explicit\s+(statement|support)\b|"
            r"\bconclusion\b[^.]{0,60}\bnot\b|"
            r"without\s+(direct|specific)\s+(support|evidence)|"
            r"no\s+(direct|specific)\s+(support|evidence|textual\s+support)",
            re.IGNORECASE,
        ),
    ),
    (
        "false_absence",
        re.compile(
            r"\b(claims?|response|assert\w+|suggests?)\b[^.]{0,60}"
            r"\bthere\s+is\s+no\b|"
            r"\bsupported\s+by\s+omission\b|"
            r"\bassertion\s+that\b[^.]{0,60}\bnot\s+mentioned\b",
            re.IGNORECASE,
        ),
    ),
]


def classify(expl):
    hits = [name for name, rx in FAMILIES if rx.search(expl or "")]
    return (hits[0] if hits else "unclassified"), hits


_WORD = re.compile(r"[a-z]{3,}")
_STOP_WORDS = (
    "the a an of and or to in for with on by is are was were be been that this "
    "these those it its as at from not no than then which who whom whose but if "
    "we they he she our their there here can could may might will would should "
    "do does did has have had more most other such some any all both each "
    "between into during also study documents document context provided"
)
_STOP = set(_STOP_WORDS.split())


def content(t):
    return {w for w in _WORD.findall(t.lower()) if w not in _STOP}


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def align(h92_sents, ann_sents):
    """Map each H92 sentence to the annotated sentence key it best matches."""
    # RAGBench ships response_sentences as [key, text] pairs, not dicts.
    keys, texts = [], []
    for d in ann_sents or []:
        if isinstance(d, dict):
            k, t = d.get("key"), d.get("sentence")
        elif len(d) >= 2:
            k, t = d[0], d[1]
        else:
            continue
        if k is None or t is None:
            continue
        keys.append(k)
        texts.append(norm(t))
    out = []
    for s in h92_sents:
        ns = norm(s)
        best, bi = 0.0, -1
        for i, t in enumerate(texts):
            if not t:
                continue
            r = difflib.SequenceMatcher(None, ns, t).ratio()
            if r > best:
                best, bi = r, i
        out.append((keys[bi] if bi >= 0 else None, best))
    return out


def load_raw(sub):
    z = zipfile.ZipFile(ARCHIVE)
    name = f"galileo-ai__ragbench__{sub}__test.parquet"
    df = pl.read_parquet(io.BytesIO(z.read(name)))
    df = df.filter(
        pl.col("adherence_score").is_not_null()
        & (pl.col("response").str.len_chars() > 20)
        & (pl.col("documents").list.len() > 0)
    )
    return df.sample(min(250, len(df)), seed=0)


def main():
    dump = pl.read_parquet(HERE / "R19-H161_pairs_h150d1.parquet").filter(pl.col("is_argmax"))
    out = {}

    for sub in SUBSETS:
        raw = load_raw(sub)
        claims = raw["response"].to_list()
        ann_all = raw["response_sentences"].to_list()
        unsup_all = raw["unsupported_response_sentence_keys"].to_list()
        ssi_all = raw["sentence_support_information"].to_list()
        docs_all = raw["documents"].to_list()

        rows = []
        for i, (c, ann, unsup, ssi, docs) in enumerate(
            zip(claims, ann_all, unsup_all, ssi_all, docs_all, strict=True)
        ):
            sents = H92.sentences(c)
            amap = align(sents, ann)
            us = set(unsup or [])
            expl = {
                e.get("response_sentence_key"): e.get("explanation")
                for e in (ssi or [])
                if isinstance(e, dict)
            }
            ev = content(" ".join(docs[:8]))
            for si, (s, (k, sim)) in enumerate(zip(sents, amap, strict=True)):
                cs = content(s)
                rows.append(
                    {
                        "item_id": i,
                        "sent_idx": si,
                        "sentence": s,
                        "ann_key": k,
                        "align_sim": sim,
                        "sent_unsupported": int(k in us) if k is not None else None,
                        "explanation": expl.get(k),
                        "containment": (len(cs & ev) / len(cs)) if cs else 0.0,
                    }
                )
        sd = pl.DataFrame(rows)
        d = dump.filter(pl.col("subset") == sub).select(
            "item_id", "sent_idx", "label", "sent_score", "tok_containment"
        )
        m = sd.join(d, on=["item_id", "sent_idx"], how="inner").filter(
            pl.col("sent_unsupported").is_not_null() & (pl.col("align_sim") >= 0.6)
        )
        if len(m) < 50:
            out[sub] = {"note": "too few aligned sentences", "n": len(m)}
            continue

        yv = 1 - m["sent_unsupported"].to_numpy()  # 1 = supported
        res = {
            "n_aligned": len(m),
            "align_rate": round(len(m) / len(sd), 4),
            "mean_align_sim": round(float(m["align_sim"].mean()), 4),
            "n_unsupported": int((1 - yv).sum()),
            "unsupported_rate": round(float((1 - yv).mean()), 4),
        }
        if 0 < yv.sum() < len(yv):
            res["TRUE_sent_auroc_model"] = round(
                float(roc_auc_score(yv, m["sent_score"].to_numpy())), 4
            )
            res["TRUE_sent_auroc_containment"] = round(
                float(roc_auc_score(yv, m["tok_containment"].to_numpy())), 4
            )
            sup = m.filter(pl.col("sent_unsupported") == 0)
            uns = m.filter(pl.col("sent_unsupported") == 1)
            res["mean_containment_supported"] = round(float(sup["tok_containment"].mean()), 4)
            res["mean_containment_unsupported"] = round(float(uns["tok_containment"].mean()), 4)
            res["mean_score_supported"] = round(float(sup["sent_score"].mean()), 4)
            res["mean_score_unsupported"] = round(float(uns["sent_score"].mean()), 4)
        out[sub] = res

        if sub == "pubmedqa":
            m.write_parquet(HERE / "R19-H162_pubmedqa_sentlabel.parquet")
            # explanation taxonomy over the unsupported sentences
            uns = m.filter(pl.col("sent_unsupported") == 1)
            prim, multi = {}, {}
            for e in uns["explanation"].to_list():
                p, hits = classify(e)
                prim[p] = prim.get(p, 0) + 1
                for h in hits:
                    multi[h] = multi.get(h, 0) + 1
            n = len(uns)
            res["explanation_taxonomy_primary"] = {
                k: {
                    "n": v,
                    "share": round(v / n, 4),
                    "se": round(float(np.sqrt(v / n * (1 - v / n) / n)), 4),
                }
                for k, v in sorted(prim.items(), key=lambda kv: -kv[1])
            }
            res["explanation_taxonomy_multilabel"] = {
                k: {"n": v, "share": round(v / n, 4)}
                for k, v in sorted(multi.items(), key=lambda kv: -kv[1])
            }
            res["n_explanations_classified"] = n

            # Per-class discrimination: how well does the verifier separate the
            # unsupported sentences OF THIS CLASS from all supported sentences?
            # 0.5 means the class is invisible to the model.
            cls = [classify(e)[0] for e in m["explanation"].to_list()]
            m2 = m.with_columns(pl.Series("cls", cls))
            sup = m2.filter(pl.col("sent_unsupported") == 0)
            sup_s = sup["sent_score"].to_numpy()
            sup_c = sup["tok_containment"].to_numpy()
            per_class = {}
            for c in sorted(set(cls)):
                g = m2.filter((pl.col("sent_unsupported") == 1) & (pl.col("cls") == c))
                if len(g) < 5:
                    per_class[c] = {"n": len(g), "note": "n<5, not read"}
                    continue
                gs = g["sent_score"].to_numpy()
                gc = g["tok_containment"].to_numpy()
                yy = np.r_[np.ones(len(sup_s)), np.zeros(len(gs))]
                per_class[c] = {
                    "n": len(g),
                    "share_of_unsupported": round(len(g) / n, 4),
                    "auroc_model_vs_supported": round(
                        float(roc_auc_score(yy, np.r_[sup_s, gs])), 4
                    ),
                    "auroc_containment_vs_supported": round(
                        float(roc_auc_score(yy, np.r_[sup_c, gc])), 4
                    ),
                    "mean_score": round(float(gs.mean()), 4),
                    "mean_containment": round(float(gc.mean()), 4),
                }
            res["per_class_discrimination"] = per_class
            res["reference_supported"] = {
                "n": len(sup),
                "mean_score": round(float(sup_s.mean()), 4),
                "mean_containment": round(float(sup_c.mean()), 4),
            }
            m2.write_parquet(HERE / "R19-H162_pubmedqa_sentlabel.parquet")

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
