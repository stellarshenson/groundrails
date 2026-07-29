"""Fetch the licence-clean public grounding corpora into data/external/datasets/.

Survey and selection rationale: reports/research-grounding-datasets.md.
Registered against round 7 of docs/experiments/semantic-grounding-experiments.md.

Every corpus here passed three filters simultaneously, which most did not:

  1. LICENCE permits commercial use - this eliminated the four largest
     candidates outright (TrueTeacher 1.38M CC-BY-NC, HaluBench CC-BY-NC,
     MS MARCO non-commercial, ANLI CC-BY-NC), and MEMERAG whose card forbids
     training in terms
  2. SOURCE DOCUMENTS ship with the claims - a conversation without its
     evidence cannot be grounded, which is what removed Mu-SHROOM, WildChat
     and LMSYS-Chat-1M
  3. The task maps onto (claim, evidence) -> supported

For each corpus this writes a sidecar `dataset-<name>.md` recording provenance,
licence, schema and the mapping onto our task, then downloads and archives it as
`dataset-<name>.zip`. The sidecars are tracked in git; the archives are not.

Run:  uv run python scripts/fetch_grounding_datasets.py            # all
      uv run python scripts/fetch_grounding_datasets.py ragtruth   # one
      uv run python scripts/fetch_grounding_datasets.py --dry-run  # sidecars only
"""

from pathlib import Path
import shutil
import sys
import zipfile

OUT = Path(__file__).resolve().parent.parent / "data" / "external" / "datasets"

# One source of truth: the sidecars are generated from this spec, so a
# description can never drift from what was actually downloaded.
DATASETS = {
    "ragtruth": {
        "title": "RAGTruth",
        "hf": ["wandb/RAGTruth-processed"],
        "licence": "MIT",
        "size": "17,790 responses - 15,090 train / 2,700 test",
        "languages": "English",
        "negatives": "Naturally occurring LLM hallucinations - 6 models (Llama-2 7/13/70B, "
        "Mistral-7B, GPT-3.5, GPT-4) answering real retrieval prompts; no perturbation",
        "labels": "Human expert span annotation",
        "why": "The only corpus where domain, negative construction and licence are all "
        "correct at once. Negatives are the error distribution a production RAG "
        "grounder actually meets, not a synthetic approximation of it.",
        "caveats": "English only. Retrieval contexts derive from MS MARCO / CNN-DM / Yelp, "
        "so the documents are public-web register rather than business documents.",
        "mapping": "response span -> claim; retrieved passage -> evidence; "
        "any hallucinated span in a sentence -> that claim is unsupported",
    },
    "ragtruth-translated": {
        "title": "RAGTruth, machine-translated into 7 languages",
        "hf": [
            f"KRLabsOrg/ragtruth-{lang}-translated"
            for lang in ("de", "fr", "es", "it", "pl", "hu", "cn")
        ],
        "licence": "MIT",
        "size": "17,790 each, ~106k train across the seven",
        "languages": "de, fr, es, it, pl, hu, zh",
        "negatives": "Inherited from RAGTruth, translated with context",
        "labels": "Inherited human spans, re-aligned after translation (Gemma 3 27B via vLLM)",
        "why": "The cheapest licence-clean route to non-English supervision, and our "
        "non-English slices are thin by construction - 21 languages spread over "
        "639 traces.",
        "caveats": "Machine-translated. Only 300 German rows are human-verified "
        "(`KRLabsOrg/ragtruth-de-translated-manual-300`), so non-English label "
        "alignment is inherited rather than checked. Treat as weaker supervision "
        "than the English original.",
        "mapping": "as RAGTruth",
    },
    "lettucedetect-prose": {
        "title": "LettuceDetect v2 prose hallucination",
        "hf": ["KRLabsOrg/lettucedetect-prose-hallucination"],
        "licence": "CC-BY-4.0",
        "size": "87,800 - 78,900 train / 3,360 val / 5,600 test",
        "languages": "14",
        "negatives": "Near-miss by construction - an LLM proposes localized replacement "
        "edits (wrong value, wrong identifier, unsupported addition) applied "
        "deterministically so exact character offsets survive",
        "labels": "LLM-generated and LLM-judged, character spans",
        "why": "Near-miss negatives at multilingual scale. R6-H37 was registered precisely "
        "because unrelated negatives flatter a model; these are the hard kind.",
        "caveats": "Documents are ACL papers, READMEs and Wikipedia markdown - treat as "
        "LANGUAGE coverage, not domain coverage. Labels are not human-verified.",
        "mapping": "answer sentence -> claim; context field -> evidence; "
        "tagged span -> unsupported",
    },
    "psiloqa": {
        "title": "PsiloQA",
        "hf": ["s-nlp/PsiloQA"],
        "licence": "CC-BY-4.0",
        "size": "63,792 train / 3,355 val / 2,897 test",
        "languages": "14 - en de fr es it ca eu sv fi cs ar fa hi zh",
        "negatives": "Naturally occurring - diverse LLMs answer without context, then the "
        "answer is compared against retrieved Wikipedia; wrong entity, date or "
        "number dominates",
        "labels": "GPT-4o end to end (QA generation and span marking), no human verification",
        "why": "Second-largest multilingual source, and its error profile - wrong number, "
        "wrong entity - matches the residual our own numeric and entity checks target.",
        "caveats": "Entirely LLM-annotated, so label noise is real and unmeasured. "
        "Wikipedia register.",
        "mapping": "answer -> claim; retrieved Wikipedia passage -> evidence",
    },
    "ragbench": {
        "title": "RAGBench (safe core - MS MARCO and CUAD subsets EXCLUDED)",
        "hf": ["galileo-ai/ragbench"],
        "subsets": [
            "techqa",
            "emanual",
            "delucionqa",
            "expertqa",
            "hagrid",
            "finqa",
            "tatqa",
            "covidqa",
            "pubmedqa",
            "hotpotqa",
        ],
        "licence": "CC-BY-4.0 at the collection level - SEE CAVEAT",
        "size": "~100k total, ~78k train across 12 subsets; ~60k over the 10 safe ones",
        "languages": "English",
        "negatives": "Naturally occurring - GPT-3.5-0125 and Claude-3-Haiku prompted "
        "permissively with no adherence instruction, so authentic drift appears",
        "labels": "GPT-4-0125-preview with chain-of-thought; no human annotation",
        "why": "The only large corpus whose DOCUMENTS resemble ours - support tickets, "
        "consumer manuals, a car manual, financial and legal filings.",
        "caveats": "LICENCE: the CC-BY-4.0 tag sits over 12 upstream corpora including "
        "MS MARCO (Microsoft, non-commercial research only) and CUAD. A vendor "
        "tag does not extinguish an upstream restriction, so those two subsets "
        "are EXCLUDED here and only the 10 listed are fetched. Labels are GPT-4, "
        "not human.",
        "mapping": "response sentence -> claim; retrieved documents -> evidence",
    },
    "faithdial": {
        "title": "FaithDial",
        "hf": ["McGill-NLP/FaithDial"],
        "licence": "MIT",
        "size": "50,761 turns / 5,649 dialogues; ~18.4k train rows in the HF plain_text config",
        "languages": "English",
        "negatives": "Genuine human-written hallucinated utterances from Wizard of Wikipedia, "
        "kept alongside the amended faithful version",
        "labels": "Human - MTurk amendment plus BEGIN labels "
        "(Hallucination / Entailment / Generic)",
        "why": "The only corpus here where BOTH sides of the conversation are human, and the "
        "negatives are hallucinations a person actually wrote rather than a model.",
        "caveats": "Small and English-only. Wikipedia-snippet evidence. **Not currently "
        "fetchable** - `McGill-NLP/FaithDial` ships a loading SCRIPT (`FaithDial.py`) "
        "and `datasets` 5.x removed script support, so it fails with 'Dataset scripts "
        "are no longer supported'. Needs a parquet mirror or a manual download from "
        "the project's GitHub before it can be used.",
        "mapping": "utterance -> claim; knowledge snippet -> evidence; BEGIN label -> verdict",
    },
    "nomiracl": {
        "title": "NoMIRACL",
        "hf": ["miracl/nomiracl"],
        "licence": "Apache-2.0",
        "size": "18 language subsets, up to 10 annotated passages per query",
        "languages": "18",
        "negatives": "The NO-RELEVANT-EVIDENCE case - queries whose retrieved passages "
        "genuinely do not answer them",
        "labels": "Human relevance judgments (MIRACL)",
        "why": "Worth more than its size suggests. Our gold sits at base rate 0.649 and is "
        "THIN ON TRUE NEGATIVES, which is exactly where every model tested in "
        "rounds 4-6 failed - all of them manufactured support. This supplies that "
        "case in 18 languages.",
        "caveats": "Not conversational RAG; MIRACL queries over Wikipedia. May be gated on "
        "the Hub.",
        "mapping": "query -> claim proxy; passage -> evidence; no-relevant -> unsupported",
    },
    "halueval": {
        "title": "HaluEval",
        "hf": ["pminervini/HaluEval"],
        # HaluEval ships no default config, so the names are explicit. `qa` and
        # `summarization` are the two that carry a knowledge/document field and
        # therefore map onto our task; `dialogue` and `general` are kept for
        # completeness but have no retrievable evidence to ground against.
        "subsets": ["qa", "summarization", "dialogue", "general"],
        "licence": "MIT",
        "size": "~35k - 5k real ChatGPT user queries plus 30k task-specific",
        "languages": "English",
        "negatives": "LLM-constructed near-miss - ChatGPT sampling-then-filtering selects the "
        "most plausible-yet-wrong candidate",
        "labels": "Human on the 5k real-query slice; LLM-constructed on the 30k",
        "why": "Bulk English supervision with near-miss negatives, MIT-licensed.",
        "caveats": "Only the 5k real-query slice carries human labels. Widely used for "
        "EVALUATION, so holding it out of training preserves it as a public "
        "comparison point.",
        "mapping": "answer -> claim; knowledge field -> evidence",
    },
}

SIDECAR = """# {title}

{why}

- **HuggingFace** - {hf}
- **Licence** - {licence}
- **Size** - {size}
- **Languages** - {languages}
- **How negatives were made** - {negatives}
- **How labels were made** - {labels}
- **Mapping onto our task** - {mapping}

## Caveats

{caveats}

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-{name}.zip` is
gitignored; this sidecar is tracked.
"""


def write_sidecar(name, spec):
    body = SIDECAR.format(
        name=name,
        hf=", ".join(f"`{h}`" for h in spec["hf"]),
        **{
            k: spec.get(k, "-")
            for k in (
                "title",
                "why",
                "licence",
                "size",
                "languages",
                "negatives",
                "labels",
                "mapping",
                "caveats",
            )
        },
    )
    if spec.get("subsets"):
        body += "\n## Subsets fetched\n\n" + "\n".join(f"- `{s}`" for s in spec["subsets"]) + "\n"
    path = OUT / f"dataset-{name}.md"
    path.write_text(body)
    return path


def fetch(name, spec):
    from datasets import load_dataset

    staging = OUT / f"_staging_{name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    written = 0
    for hf_id in spec["hf"]:
        subsets = spec.get("subsets") or [None]
        for sub in subsets:
            tag = f"{hf_id.replace('/', '__')}{'__' + sub if sub else ''}"
            try:
                ds = load_dataset(hf_id, sub) if sub else load_dataset(hf_id)
            except Exception as e:  # noqa: BLE001 - a failed corpus is a result, not a crash
                print(
                    f"    SKIP {hf_id}{'/' + sub if sub else ''}: "
                    f"{type(e).__name__}: {str(e)[:110]}",
                    flush=True,
                )
                continue
            for split, d in ds.items():
                f = staging / f"{tag}__{split}.parquet"
                d.to_parquet(f)
                written += 1
                print(f"    {tag}/{split}: {len(d)} rows -> {f.name}", flush=True)

    if not written:
        shutil.rmtree(staging)
        return None

    archive = OUT / f"dataset-{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.iterdir()):
            z.write(f, f.name)
        z.write(OUT / f"dataset-{name}.md", f"dataset-{name}.md")
    shutil.rmtree(staging)
    return archive


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    names = args or list(DATASETS)

    for name in names:
        if name not in DATASETS:
            print(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
            continue
        spec = DATASETS[name]
        print(f"\n=== {name} - {spec['title']}", flush=True)
        sc = write_sidecar(name, spec)
        print(f"  sidecar -> {sc.name}", flush=True)
        if dry:
            continue
        archive = fetch(name, spec)
        if archive:
            mb = archive.stat().st_size / 1e6
            print(f"  archive -> {archive.name} ({mb:.1f} MB)", flush=True)
        else:
            print("  archive -> NONE (every split failed; see SKIP lines)", flush=True)

    print(f"\nsidecars and archives in {OUT}")
    print("archives are gitignored; sidecars are tracked")


if __name__ == "__main__":
    main()
