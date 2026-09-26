"""Measure the fixes for the open defects DEF-CLAIM-35..41 and DEF-NUMBER-32..34.

Each section runs the committed code (``git show <ref>:src/...``, loaded as a
separate module) and the working tree on the same public data, and counts where
they differ. Nothing here tunes a rule; the counts decide whether a fix ships.

Data (Hugging Face cache, no download): ``wandb/RAGTruth-processed`` (answers with
labelled hallucination spans, and their source contexts), ``galileo-ai/ragbench``
test splits (responses with unsupported-sentence keys, and source documents), and
the ``KRLabsOrg/ragtruth-*-translated`` answers for the non-English side.

Usage: ``HF_HUB_OFFLINE=1 .venv/bin/python experiments/defects/measure_open_defects.py``
writes ``measure_open_defects.json`` beside this file.

The JSON is in-sample: the rules were adjusted while reading these corpora, and it
was written on the round-2 code (``heldout2-frozen-src.patch``), since reduced.
"""

from __future__ import annotations

from collections import Counter
import decimal
import glob
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import polars as pl

from groundrails import entity_check as ec_new
from groundrails import extract as ex_new
from groundrails import lexical

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_REF = next((a for a in sys.argv[1:] if not a.startswith("--")), "HEAD")
HUB = Path.home() / ".cache/huggingface/hub"
SAMPLES = 8


def _load_committed(rel: str, name: str):
    """The module at ``BASE_REF`` as a separate module object."""
    src = subprocess.run(
        ["git", "show", f"{BASE_REF}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    path = HERE / f".{name}_{BASE_REF.replace('/', '_')}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through the module
    spec.loader.exec_module(mod)
    path.unlink()
    return mod


ex_old = _load_committed("src/groundrails/extract.py", "extract_base")
ec_old = _load_committed("src/groundrails/entity_check.py", "entity_check_base")
# the language warning is measured on its own below, not once per extraction
ex_old.warn_if_not_english = ex_new.warn_if_not_english = lambda sentences: None


def _ragtruth() -> pl.DataFrame:
    snap = next(HUB.glob("datasets--wandb--RAGTruth-processed/snapshots/*/data"))
    return pl.concat([pl.read_parquet(p) for p in sorted(snap.glob("*.parquet"))])


def _ragbench() -> pl.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(HUB / "datasets--galileo-ai--ragbench/snapshots/*/*/test-*.parquet"))):
        df = pl.read_parquet(
            p, columns=["documents", "response", "response_sentences", "unsupported_response_sentence_keys"]
        )
        frames.append(df.with_columns(pl.lit(Path(p).parent.name).alias("subset")))
    return pl.concat(frames)


def _sample(rows: list, k: int = SAMPLES) -> list:
    return random.Random(0).sample(rows, min(k, len(rows)))


# --- claim extraction: DEF-CLAIM-35, 38, 39, 40, 41 -------------------------


def measure_extraction(answers: list[tuple[str, str]]) -> dict:
    """Claim lists of the committed and the working-tree extractor on answer documents."""
    changed, added, removed, docs = [], 0, 0, 0
    for source, text in answers:
        old = [c.claim for c in ex_old.extract_claims(text)]
        new = [c.claim for c in ex_new.extract_claims(text)]
        docs += 1
        if old != new:
            only_old = [c for c in old if c not in new]
            only_new = [c for c in new if c not in old]
            removed += len(only_old)
            added += len(only_new)
            changed.append({"source": source, "committed": only_old, "working_tree": only_new})
    return {
        "documents": docs,
        "documents_changed": len(changed),
        "claims_only_committed": removed,
        "claims_only_working_tree": added,
        "samples": _sample(changed, 25),
    }


# --- language warning: DEF-CLAIM-37 ---------------------------------------


def measure_language(texts: list[tuple[str, str]]) -> dict:
    """Confident non-English reads per corpus, committed rule (top language) against
    the working-tree rule (top language other than Latin)."""
    lexical._lingua_lang("prime the detector with any text of enough length")
    det = lexical._LINGUA["det"]
    out: dict[str, dict] = {}
    for corpus, text in texts:
        sample = " ".join(s for _, s in ex_new._split_document(text))[: ex_new._LANG_SAMPLE_CHARS]
        row = out.setdefault(corpus, {"n": 0, "committed": Counter(), "working_tree": Counter()})
        row["n"] += 1
        if len(sample.strip()) < 25:
            continue
        values = det.compute_language_confidence_values(sample)
        top = values[0] if values else None
        non_latin = next((c for c in values if c.language.name != "LATIN"), None)
        for key, c in (("committed", top), ("working_tree", non_latin)):
            lang = c.language.iso_code_639_1.name.lower() if c and c.value >= 0.65 else "und"
            row[key][lang] += 1
    return {k: {"n": v["n"], "committed": dict(v["committed"]), "working_tree": dict(v["working_tree"])} for k, v in out.items()}


# --- hypothetical class: DEF-CLAIM-36 --------------------------------------


def measure_hypothetical(corpora: dict[str, list[str]]) -> dict:
    """How often ``out_of_scope`` returns ``hypothetical`` per corpus. On source text
    every firing is a false positive: a source states what it states."""
    out = {}
    for corpus, texts in corpora.items():
        hits, n = [], 0
        for text in texts:
            for _, sent in ex_new._split_document(text):
                if not ex_new._looks_like_claim(sent):
                    continue
                n += 1
                if ex_new.out_of_scope(sent) == "hypothetical":
                    hits.append(sent)
        out[corpus] = {"claims": n, "hypothetical": len(hits), "samples": _sample(hits, 12)}
    return out


# --- numeric checks: DEF-NUMBER-32, 33, 34 ---------------------------------


def _numeric_pairs():
    """``(corpus, claim, passage, labelled)`` - labelled means the claim lies in a
    RAGTruth hallucination span, or is a RAGBench sentence marked unsupported."""
    for row in _ragtruth().iter_rows(named=True):
        spans = [(s["start"], s["end"]) for s in json.loads(row["hallucination_labels"])]
        for c in ex_new.extract_claims(row["output"]):
            if any(ch.isdigit() for ch in c.claim):
                labelled = any(s < c.char_end and e > c.char_start for s, e in spans)
                yield "ragtruth", c.claim, row["context"], labelled
    for row in _ragbench().iter_rows(named=True):
        passage = " ".join(row["documents"])
        unsupported = set(row["unsupported_response_sentence_keys"])
        for key, sent in row["response_sentences"]:
            if any(ch.isdigit() for ch in sent):
                yield f"ragbench-{row['subset']}", sent, passage, key in unsupported


def measure_numeric() -> dict:
    """Contradictions the committed and the working-tree numeric check report."""
    counts: Counter = Counter()
    gained, lost = [], []
    for corpus, claim, passage, labelled in _numeric_pairs():
        try:
            old = bool(ec_old.find_numeric_mismatches(claim, passage))
        except decimal.InvalidOperation:  # DEF-NUMBER-43; set order decides whether it fires
            counts[(corpus, "committed_raised")] += 1
            old = False
        new_mm = ec_new.find_numeric_mismatches(claim, passage)
        new = bool(new_mm)
        counts[(corpus, "pairs")] += 1
        counts[(corpus, "labelled")] += labelled
        counts[(corpus, "committed_contradicted", labelled)] += old
        counts[(corpus, "working_tree_contradicted", labelled)] += new
        if new and not old:
            gained.append({"corpus": corpus, "labelled": labelled, "claim": claim, "mismatches": new_mm})
        elif old and not new:
            lost.append({"corpus": corpus, "labelled": labelled, "claim": claim})
    table: dict[str, dict] = {}
    for key, n in counts.items():
        corpus, metric = key[0], key[1]
        name = metric if len(key) == 2 else f"{metric}_{'labelled' if key[2] else 'unlabelled'}"
        table.setdefault(corpus, {})[name] = n
    return {
        "per_corpus": table,
        "gained": len(gained),
        "gained_labelled": sum(g["labelled"] for g in gained),
        "lost": len(lost),
        "lost_labelled": sum(g["labelled"] for g in lost),
        "gained_samples": _sample(gained, 40),
        "lost_samples": _sample(lost, 40),
    }


def main() -> None:
    rt = _ragtruth()
    rb = _ragbench()
    answers = [("ragtruth", t) for t in rt["output"].to_list()] + [
        (f"ragbench-{s}", t) for s, t in zip(rb["subset"].to_list(), rb["response"].to_list())
    ]
    contexts = sorted(set(rt["context"].to_list()))
    documents = [" ".join(d) for d in rb["documents"].to_list()]
    translated = []
    for lang in ("pl", "de", "es", "fr", "it", "hu", "cn"):
        p = next(HUB.glob(f"datasets--KRLabsOrg--ragtruth-{lang}-translated/snapshots/*/data/test-*.parquet"), None)
        if p is not None:
            translated += [(f"ragtruth-{lang}", t) for t in pl.read_parquet(p)["answer"].to_list()[:500]]
    results = {
        "base_ref": BASE_REF,
        "extraction": measure_extraction(answers),
        "hypothetical": measure_hypothetical(
            {
                "ragtruth-answers": rt["output"].to_list(),
                "ragbench-responses": rb["response"].to_list(),
                "ragtruth-sources": contexts,
                "ragbench-sources": documents,
            }
        ),
        "language": measure_language(
            [("ragtruth-answers", t) for t in rt["output"].to_list()]
            + [(f"ragbench-{s}-documents", " ".join(d)) for s, d in zip(rb["subset"].to_list(), rb["documents"].to_list())]
            + translated
        ),
        "numeric": measure_numeric(),
    }
    (HERE / "measure_open_defects.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in results["numeric"].items() if "samples" not in k}, indent=1))


if __name__ == "__main__":
    main()
