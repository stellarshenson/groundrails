"""Bespoke lane adapters - the corpora a declarative format block cannot express.

Three shapes need code. FAVA ships one prompt string carrying both the
references and the draft passage, plus a completion whose inline tags are the
label; FActScore ships human judgments in a jsonl tree whose evidence lives in
separate per-topic files; FinDVer ships claims that name context ids inside a
separate filing document. Everything else in the manifest is column mapping.

Each returns ``(rows, stats)`` - the stats being what a reviewer needs to trust
the parse (how many prompts parsed, how many facts were excluded and why).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from groundrails.dataset.fetch import read_fetched
from groundrails.dataset.format import register_adapter
from groundrails.dataset.manifest import CorpusEntry

# --------------------------------------------------------------------------- #
# FAVA - references and draft live in one prompt; the label lives in the tags
# --------------------------------------------------------------------------- #
FAVA_PREFIX = "Read the following references:\n"
FAVA_SEP = (
    "\nPlease identify all the errors in the following passage using the "
    "references provided and suggest edits:\nText: "
)
FAVA_TAGS = ("entity", "relation", "contradictory", "subjective", "unverifiable", "invented")
FAVA_EDIT = re.compile(r"<(entity|relation)>(.*?)</\1>", re.DOTALL)
# LLM-written completions frequently leave the wrapper unclosed
# (`<relation><mark>X</mark><delete>Y</delete>.` with no `</relation>`) - the
# fallback captures the same span without requiring the closer
FAVA_EDIT_OPEN = re.compile(
    r"<(entity|relation)>\s*<mark>(.*?)</mark>\s*<delete>(.*?)</delete>", re.DOTALL
)
FAVA_WRAP = re.compile(r"<(contradictory|subjective|unverifiable|invented)>(.*?)</\1>", re.DOTALL)
FAVA_MARK = re.compile(r"<mark>(.*?)</mark>", re.DOTALL)
FAVA_DEL = re.compile(r"<delete>(.*?)</delete>", re.DOTALL)
FAVA_ANYTAG = re.compile(r"</?(" + "|".join(FAVA_TAGS) + r"|mark|delete)\s*>")
# the label rule: ANY open error tag, closed or not, marks the row hallucinated
FAVA_ERR_OPEN = re.compile(r"<(" + "|".join(FAVA_TAGS) + r")[>\s]")


def fava_spans(completion: str) -> tuple[list[dict], str, bool]:
    """Error spans from a tagged completion, plus the reconstructed pre-edit passage.

    Tag semantics follow the FAVA paper's own prompt: ``<mark>`` carries the
    CORRECT text matching the reference, ``<delete>`` carries the error text
    present in the original passage - so keeping the delete branch and dropping
    the mark branch reconstructs the draft the model was given.
    """
    spans: list[dict] = []

    def rec_edit(m):
        body = m.group(2)
        mark = FAVA_MARK.search(body)
        dele = FAVA_DEL.search(body)
        spans.append(
            {
                "type": m.group(1),
                "mark": mark.group(1).strip() if mark else None,
                "delete": dele.group(1).strip() if dele else None,
            }
        )
        return dele.group(1) if dele else ""  # mark-only edit = inserted text

    def rec_open(m):
        spans.append(
            {"type": m.group(1), "mark": m.group(2).strip(), "delete": m.group(3).strip()}
        )
        return m.group(3)

    def rec_wrap(m):
        spans.append({"type": m.group(1), "mark": None, "delete": m.group(2).strip()})
        return m.group(2)

    recon = FAVA_EDIT_OPEN.sub(rec_open, FAVA_EDIT.sub(rec_edit, completion))
    recon = FAVA_WRAP.sub(rec_wrap, recon)
    residual = bool(FAVA_ANYTAG.search(recon))
    return spans, FAVA_ANYTAG.sub("", recon), residual


@register_adapter("fava")
def build_fava(entry: CorpusEntry, data_dir: Path) -> tuple[list[dict], dict]:
    """One row per prompt: references -> chunk, draft passage -> claim, tags -> label."""
    df = read_fetched(entry, data_dir)["train"]
    rows, orphan_markup, recon_match = [], 0, 0
    for r in df.iter_rows(named=True):
        p = r["prompt"]
        if not (p.startswith(FAVA_PREFIX) and FAVA_SEP in p):
            continue
        refs = p[len(FAVA_PREFIX) : p.index(FAVA_SEP)]
        claim = p[p.index(FAVA_SEP) + len(FAVA_SEP) :]
        comp = r["completion"]
        spans, recon, residual = fava_spans(comp)
        orphan_markup += residual
        matched = recon.strip() == claim.strip()
        recon_match += matched
        rows.append(
            {
                "claim": claim.strip(),
                "chunk": refs.strip(),
                "label": 0 if FAVA_ERR_OPEN.search(comp) else 1,
                "doc_id": hashlib.blake2b(p.encode(), digest_size=8).hexdigest(),
                "source": "train",
                "n_spans": len(spans),
                "error_types": json.dumps(sorted({s["type"] for s in spans})),
                "spans_json": json.dumps(spans),
                "completion_tagged": comp,
                "reconstruction_matches_draft": matched,
            }
        )
    return rows, {
        "rows_in": df.height,
        "rows_with_orphan_tag_markup": orphan_markup,
        "orphan_markup_note": "LLM-written completions carry unclosed or wrapper-less tags; the "
        "label rule counts any error tag OPEN, so labels are unaffected - only span "
        "completeness is",
        "reconstruction_match_rate": round(recon_match / max(len(rows), 1), 6),
        "reconstruction_note": "share of rows where the tag-resolved completion reproduces the "
        "prompt's draft byte for byte; the claim is the prompt's draft regardless, so a miss "
        "costs nothing - it flags annotation drift on that row",
    }


# --------------------------------------------------------------------------- #
# FActScore - human atomic-fact judgments over a per-topic evidence tree
# --------------------------------------------------------------------------- #
@register_adapter("factscore")
def build_factscore(entry: CorpusEntry, data_dir: Path) -> tuple[list[dict], dict]:
    """One row per human-judged atomic fact; IR (off-topic) facts are excluded.

    Every fact about a topic carries that topic's whole Wikipedia article as
    evidence, so ``doc_id`` is the topic - 183 documents behind order-10k rows.
    That reuse is real and the shape stage is where it gets ruled on.
    """
    tree = Path(data_dir) / entry.name
    index = json.loads((tree / "evidence" / "_index.json").read_text(encoding="utf-8"))
    rows, ir_dropped, missing_ev = [], 0, 0
    chunks: dict[str, str] = {}
    for model in ("InstructGPT", "ChatGPT", "PerplexityAI"):
        path = tree / "data" / "labeled" / f"{model}.jsonl"
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            topic = r["topic"]
            ev = index.get(topic)
            if not ev:
                missing_ev += 1
                continue
            if topic not in chunks:
                chunks[topic] = (tree / "evidence" / ev["file"]).read_text(encoding="utf-8")
            for ann in r.get("annotations") or []:
                for f in ann.get("human-atomic-facts") or []:
                    lab = f.get("label")
                    if lab == "IR":
                        ir_dropped += 1
                        continue
                    if lab not in ("S", "NS"):
                        continue
                    rows.append(
                        {
                            "claim": f["text"].strip(),
                            "chunk": chunks[topic],
                            "label": 1 if lab == "S" else 0,
                            "doc_id": topic,
                            "source": model,
                            "sentence": ann.get("text", ""),
                            "sentence_is_relevant": bool(ann.get("is-relevant")),
                            "wiki_oldid": str(ev["oldid"]),
                            "wiki_rev_timestamp": ev["timestamp"],
                        }
                    )
    return rows, {
        "ir_facts_dropped": ir_dropped,
        "biographies_missing_evidence": missing_ev,
        "evidence_topics": len(index),
    }


# --------------------------------------------------------------------------- #
# FinDVer - claims naming context ids inside a separate filing document
# --------------------------------------------------------------------------- #
@register_adapter("findver")
def build_findver(entry: CorpusEntry, data_dir: Path) -> tuple[list[dict], dict]:
    """One row per claim; evidence is the annotated ``relevant_context`` passages."""
    tree = Path(data_dir) / entry.name
    rows, missing_ctx = [], 0
    docs: dict[str, dict] = {}
    for split in ("test", "testmini"):
        for r in json.loads((tree / "data" / f"{split}.json").read_text(encoding="utf-8")):
            rep = r["report"]
            if rep not in docs:
                doc = json.loads((tree / "financial_reports" / rep).read_text(encoding="utf-8"))
                docs[rep] = {c["id"]: c for c in doc["context"]}
            ctx = docs[rep]
            parts, resolved = [], True
            for i in r["relevant_context"]:
                if i in ctx:
                    c = ctx[i]
                    parts.append(f"[context {i} | {c.get('type', 'text')}]\n{c['context']}")
                else:
                    resolved = False
            if not parts:
                missing_ctx += 1
                continue
            rows.append(
                {
                    "claim": r["statement"].strip(),
                    "chunk": "\n\n".join(parts),
                    "label": 1 if r["entailment_label"] else 0,
                    "doc_id": rep,
                    "source": split,
                    "example_id": r["example_id"],
                    "subset": r["subset"],
                    "explanation": r.get("explanation", ""),
                    "context_ids": json.dumps(r["relevant_context"]),
                    "all_contexts_resolved": resolved,
                }
            )
    return rows, {"claims_missing_all_contexts": missing_ctx, "reports": len(docs)}
