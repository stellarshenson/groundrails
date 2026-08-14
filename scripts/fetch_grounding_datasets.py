"""Fetch the licence-clean public grounding corpora into data/external/datasets/.

Survey and selection rationale: reports/research-grounding-datasets.md.
Registered against round 7 of docs/experiments/semantic-grounding-experiments.md;
the six R19 supply-wave corpora (fava, pubhealth, minicheck, factscore, findver,
attributionbench) are registered in docs/experiments/semantic-dataset-enhancements.md
section "R19 supply wave".

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
Sidecars are generated from the spec dict below - never hand-edited: they are
written first (dry render), then re-rendered post-fetch with the row counts the
download actually produced. Corpora whose source does not fit an archive
(GitHub trees, multi-file evidence pulls) land as an extracted tree
`data/external/datasets/<name>/` plus sidecar, no archive (the
edgar-restricted pattern). Re-runs are idempotent: a corpus whose output
already exists is skipped and its sidecar re-rendered from the recorded
`_counts.json`.

Run:  uv run python scripts/fetch_grounding_datasets.py            # all
      uv run python scripts/fetch_grounding_datasets.py ragtruth   # one
      uv run python scripts/fetch_grounding_datasets.py --dry-run  # sidecars only
      uv run --with gdown python scripts/fetch_grounding_datasets.py pubhealth factscore
"""

from pathlib import Path
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

OUT = Path(__file__).resolve().parent.parent / "data" / "external" / "datasets"

DEFAULT_PROVENANCE = (
    "Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every "
    "corpus\nin this directory passed three filters together: a licence permitting commercial "
    "use, source\ndocuments shipping alongside the claims, and a task shape that maps onto "
    "(claim, evidence) →\nsupported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, "
    "1.38M), HaluBench\n(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card "
    "forbids training)."
)

PROVENANCE_R19 = (
    "Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "
    "\"Re-survey\n2026-08-13\" - licence verified at source there and RE-VERIFIED at pull time "
    "in this build; the\nlicence line above is the tag read from the source pulled, not the "
    "recon's say-so). Registered in\n`docs/experiments/semantic-dataset-enhancements.md`, "
    "section \"R19 supply wave\" (2026-08-13): SUPPLY ONLY -\nnothing enters a training mix "
    "without its own registered hypothesis and arm. The contamination\ngate (R14-H136 8-gram "
    "Jaccard instrument against the ten walled arena corpora, bar 0.02 max\nfraction + spike "
    "control) runs after this fetch and its verdict is recorded in\n"
    "`experiments/grounding-semantic/R19_{name}_gate.json`; the pair-formatted lane, manifest "
    "and verify\nJSON land beside it as `R19_{name}_lane.parquet` / `_manifest.json` / "
    "`_verify.json`."
)

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
    "vitaminc": {
        "title": "VitaminC",
        "hf": ["tals/vitaminc"],
        "licence": "CC-BY-SA-3.0 (Wikipedia-derived) - VERIFY before shipping a model trained on it",
        "size": "~489k claim-evidence pairs",
        "languages": "English",
        "negatives": "**Contrastive Wikipedia revisions** - real edits to real articles where a "
        "single factual change flips the verdict, so the negative differs from its "
        "positive by one number, entity or qualifier and nothing else. This is the "
        "textbook near-miss construction and the best-built negatives in the field",
        "labels": "Human annotation over genuine revision pairs (SUPPORTS / REFUTES / NOT "
        "ENOUGH INFO)",
        "why": "Every other corpus in this directory teaches the base rate as much as the "
        "boundary, because its negatives are topically distant. VitaminC's negatives are "
        "minimally different from their positives by construction, which is exactly the "
        "discrimination R6-H37 was registered to demand and the failure mode rounds 4-6 "
        "kept finding: models that confirm a claim against unrelated text.",
        "caveats": "Wikipedia register, not conversational RAG - the dataset survey "
        "deprioritised it on DOMAIN, never on quality, and named it the fallback if "
        "domain-matched data underdelivers. English only. Three-way labels must be "
        "collapsed to binary (SUPPORTS -> grounded, REFUTES and NEI -> not).",
        "mapping": "claim -> claim; evidence -> evidence; SUPPORTS -> 1, otherwise 0",
    },
    "tabfact": {
        "title": "TabFact",
        # Every HF mirror (`ibm/tab_fact`, `ibm-research/tab_fact`,
        # `wenhu/tab_fact`) ships a loading SCRIPT and datasets 5.x removed
        # script support, so this corpus is fetched straight from the GitHub
        # repo those scripts download from.
        "hf": ["wenhu/tab_fact (script-only; fetched from GitHub instead)"],
        "github": "https://github.com/wenhuchen/Table-Fact-Checking/archive/refs/heads/master.zip",
        "licence": "CC-BY-4.0",
        "size": "~118k statements over 16k Wikipedia tables - 92,283 train / 12,792 val / 12,779 test",
        "languages": "English",
        "negatives": "Counterfactual statements written by annotators against the SAME table "
        "that entails the positives - near-miss by construction, in the numeric-tabular "
        "register no other corpus in this directory covers",
        "labels": "Human annotation (ICLR 2020), binary ENTAILED / REFUTED",
        "why": "The only permissively-licensed corpus whose claims require reading numbers "
        "out of structured tables - registered for R8-H87 after the blind-arena residual "
        "concentrated in the tabular subsets (finqa -0.1373, tatqa -0.0169) and the "
        "R8-H85 probe ruled out context truncation as the cause.",
        "caveats": "Wikipedia tables, not financial filings - register coverage, not domain "
        "coverage. Tables arrive linearized (`#`-separated header and rows) and must be "
        "serialized into evidence text; statements are short and single-fact.",
        "mapping": "statement -> claim; caption + linearized table -> evidence; "
        "ENTAILED -> 1, REFUTED -> 0",
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
    # ------------------------------------------------------------------ #
    # R19 supply wave (registered: docs/experiments/semantic-dataset-enhancements.md,
    # "R19 supply wave", 2026-08-13).  SUPPLY ONLY - nothing here enters a
    # training mix without its own registered hypothesis and arm.
    # ------------------------------------------------------------------ #
    "fava": {
        "title": "FAVA fava-data - span-tagged long-form hallucination corpus",
        "hf": ["fava-uw/fava-data"],
        "fetcher": "fava",
        "licence": "CC-BY-4.0 (HF card YAML `license: cc-by-4.0`, re-verified at the Hub 2026-08-13)",
        "size": "~30k synthetic training rows plus 460 human-annotated gold passages "
        "(`annotations.json`, banked alongside, not pair-formatted)",
        "languages": "English",
        "negatives": "Synthetic error injection - the FAVA pipeline plants typed hallucination "
        "spans (entity, relation, contradictory, subjective, unverifiable, invented) into "
        "reference-grounded drafts and ships the draft plus its tagged edit",
        "labels": "Silver span tags from the generation pipeline; 460 gold passages carry "
        "human annotation",
        "why": "The only span-level long-form taxonomy set the 2026-08-13 recon found; the "
        "error taxonomy (six typed spans) is the fine-grained structure no other admitted "
        "corpus carries.",
        "caveats": "Wikipedia/web register, not business documents. Silver labels, not human. "
        "The claim is the draft passage embedded in the prompt after `Text:` - the byte-exact "
        "pre-edit text the tagged completion edits; the tagged completion is retained as a "
        "provenance column.",
        "mapping": "prompt `Text:` passage → claim; `Reference [i]` blocks → evidence; any "
        "error span in the tagged completion → 0, clean → 1; span metadata retained for "
        "future fine-grained use",
        "provenance": PROVENANCE_R19,
    },
    "pubhealth": {
        "title": "PubHealth (health_fact) - real-world public-health fact-checks",
        "hf": ["ImperialCollegeLondon/health_fact"],
        "fetcher": "pubhealth",
        "licence": "MIT (HF card YAML `license: mit`, re-verified at the Hub 2026-08-13; the "
        "loader script carries an Apache-2.0 code header - data licence per the card)",
        "size": "12,288 rows (train/dev/test TSVs)",
        "languages": "English",
        "negatives": "Naturally occurring false, unproven and mixed health claims as published "
        "and fact-checked by Snopes/PolitiFact/Full Fact lineage sites - no perturbation",
        "labels": "Journalist verdicts (4-way true/false/unproven/mixture) with written "
        "explanations; human throughout",
        "why": "Real-world fact-check register - journalism over genuine public claims - that "
        "nothing else in the directory covers, with human labels end to end.",
        "caveats": "The HF repo ships only a loading script (datasets 5.x removed script "
        "support); the data is pulled from the authors' Google Drive zip the script names. "
        "Rows with empty `main_text` or stray labels are excluded from the lane (rates "
        "reported in the gate JSON) but retained in the archive. Verdict maps onto SUPPORT, "
        "not truth: unproven/mixture read as not-supported (registered coordinator ruling).",
        "mapping": "claim → claim; source article `main_text` → evidence; true → 1, "
        "false/unproven/mixture → 0",
        "provenance": PROVENANCE_R19,
    },
    "minicheck": {
        "title": "MiniCheck C2D/D2C - multi-fact claim/document checking pairs",
        "hf": ["lytang/C2D-and-D2C-MiniCheck"],
        "fetcher": "minicheck",
        "licence": "MIT (HF card YAML `license: mit`, re-verified at the Hub 2026-08-13)",
        "size": "14,395 (claim, doc, label) pairs - 7,076 C2D + 7,319 D2C",
        "languages": "English",
        "negatives": "GPT-4-generated non-supporting documents against real claims, "
        "multi-fact and multi-sentence by construction; a 53% rejection rate at the "
        "entailment filter is reported by the authors",
        "labels": "Synthetic pipeline labels with GPT-4 entailment filtering at every "
        "construction step",
        "why": "Multi-fact, multi-sentence checking at scale - the case where support must "
        "be assembled from several sentences, which single-sentence corpora do not teach.",
        "caveats": "Seed corpora are named only in the paper's Appendix D (ACL "
        "2024.emnlp-main.499): C2D seeds are ~400 Wikipedia claims with cited web articles "
        "(Kamoi et al. 2023; Petroni et al. 2023), D2C seeds are ~300 Google News articles "
        "scraped since November 2023. Neither is a walled RAGBench source corpus; the "
        "8-gram instrument runs regardless and is the binding check.",
        "mapping": "claim → claim; doc → evidence; shipped label (1 supported / 0 not)",
        "provenance": PROVENANCE_R19,
    },
    "factscore": {
        "title": "FActScore labeled biographies - human-judged atomic facts",
        "source": "github.com/shmsw25/FActScore (code; MIT LICENSE in repo) + the authors' "
        "Google Drive `data.zip` (`data/labeled/*.jsonl`); Wikipedia evidence pulled per "
        "topic from the Wikipedia API, pinned to the last revision on or before 2023-04-01 "
        "(the corpus's enwiki-20230401 knowledge source)",
        "fetcher": "factscore",
        "tree": True,
        "licence": "MIT (repo LICENSE file, re-read at fetch 2026-08-13)",
        "size": "549 labeled biographies - 183 entities x 3 models (InstructGPT, ChatGPT, "
        "PerplexityAI); order-10k human atomic-fact judgments",
        "languages": "English",
        "negatives": "Naturally occurring unsupported atomic facts inside LM-generated "
        "biographies; no perturbation",
        "labels": "Human per-atomic-fact S (supported) / NS (not supported) / IR "
        "(off-topic sentence) judgments against Wikipedia",
        "why": "Human-judged atomic facts over long-form biographies - the highest-precision "
        "long-form supervision the recon found; a high-precision slice or held-out probe, "
        "not bulk supervision.",
        "caveats": "Small by design. Evidence is the topic's full Wikipedia article at the "
        "pinned 2023-04-01 revision fetched through the public API (the authors' 28GB "
        "enwiki-20230401.db is not pulled); HTML flattening to text is ours. IR facts are "
        "excluded from the lane (off-topic, support undefined) and counted in the gate JSON.",
        "mapping": "atomic fact → claim; topic's pinned Wikipedia article → evidence; "
        "S → 1, NS → 0",
        "provenance": PROVENANCE_R19,
    },
    "findver": {
        "title": "FinDVer - claim verification over 2024 SEC filings",
        "source": "github.com/yilunzhao/FinDVer (MIT LICENSE in repo); the repo ships the "
        "claims (`data/test.json`, `data/testmini.json`) and the 2024 10-K/10-Q filing "
        "extracts (`financial_reports/*.json`, sec.gov EDGAR urls)",
        "fetcher": "findver",
        "tree": True,
        "licence": "MIT (repo LICENSE file, re-read at fetch 2026-08-13)",
        "size": "2,400 claims - test 1,700 + testmini 700 - over 2024 10-K/10-Q filings, "
        "subsets ie / numeric / knowledge",
        "languages": "English",
        "negatives": "Human-annotated refuted claims over the filings with step-by-step "
        "explanations (GPT-4o-proofread)",
        "labels": "Human entailment judgments (entailed/refuted)",
        "why": "The only candidate filling the financial numeric-derivation gap directly - "
        "the flagship's sole losing register. Admitted under the EDGAR-population precedent: "
        "its documents are 2024 filings while the walled FinQA/TAT-QA corpora are pre-2020, "
        "so document overlap is structurally impossible; the 8-gram instrument still runs.",
        "caveats": "Claims are benchmark items (test/testmini splits of a public benchmark) "
        "banked as supply; a later hypothesis decides any training use. Evidence is the "
        "annotated `relevant_context` passages, not the whole filing; subset tags retained.",
        "mapping": "statement → claim; `relevant_context` passages → evidence; entailed → 1, "
        "refuted → 0",
        "provenance": PROVENANCE_R19,
    },
    "attributionbench": {
        "title": "AttributionBench (full_data config) - with the ExpertQA/HAGRID carve-out",
        "hf": ["osunlp/AttributionBench"],
        "fetcher": "attributionbench",
        "licence": "Apache-2.0 (HF card YAML `license: apache-2.0`, re-verified at the Hub "
        "2026-08-13)",
        "size": "26,365 rows in the card's `full_data` config (train/dev/test/test_ood) "
        "before the carve-out",
        "languages": "English",
        "negatives": "Not-attributable claims as labeled by the source benchmarks "
        "(Stanford-GenSearch, AttributedQA, LFQA, BEGIN, AttrEval) - a mix of human and "
        "benchmark-pipeline labels unified by the authors",
        "labels": "Source-benchmark attribution labels, unified to attributable / not "
        "attributable",
        "why": "Large multi-source attribution supervision in the answer-with-references "
        "shape - the closest public analogue to production RAG grounding.",
        "caveats": "**Carve-out, applied at fetch**: every row whose `src_dataset` is "
        "ExpertQA or HAGRID (both walled RAGBench source corpora) is dropped BY CONSTRUCTION "
        "- precedent: the RAGBench fetch's MS MARCO/CUAD exclusion. The gate re-derives zero "
        "walled rows from the kept frame rather than trusting this note. BEGIN and "
        "AttrScore-GenSearch exist only in the test_ood split; the kept frame spans all four "
        "splits with the split tag retained - a later hypothesis decides split hygiene.",
        "mapping": "claim → claim; references → evidence; attributable → 1, not "
        "attributable → 0",
        "provenance": PROVENANCE_R19,
    },
}

SIDECAR = """# {title}

{why}

{source_lines}
- **Licence** - {licence}
- **Size** - {size}{fetched}
- **Languages** - {languages}
- **How negatives were made** - {negatives}
- **How labels were made** - {labels}
- **Mapping onto our task** - {mapping}

## Caveats

{caveats}

## Provenance

{provenance}
"""


def _source_lines(spec):
    """The source bullet(s): HuggingFace ids, a free-text source, or a GitHub url."""
    lines = []
    if spec.get("hf"):
        lines.append("- **HuggingFace** - " + ", ".join(f"`{h}`" for h in spec["hf"]))
    if spec.get("source"):
        lines.append(f"- **Source** - {spec['source']}")
    elif spec.get("github") and not spec.get("hf"):
        lines.append(f"- **GitHub** - {spec['github']}")
    return "\n".join(lines)


def write_sidecar(name, spec, counts=None):
    """Render the sidecar from the spec. Post-fetch, `counts` (split -> rows) is
    appended to the Size bullet so the tracked file reports what the download
    actually produced, not what the survey expected."""
    fetched = ""
    if counts:
        splits = {k: v for k, v in counts.items()
                  if isinstance(v, int) and not k.startswith("_")}
        total = sum(splits.values())
        detail = ", ".join(f"{k}: {v}" for k, v in splits.items())
        fetched = f"; fetched 2026-08-13: {total} rows ({detail})"
    body = SIDECAR.format(
        source_lines=_source_lines(spec),
        fetched=fetched,
        provenance=spec.get("provenance", DEFAULT_PROVENANCE).format(name=name),
        **{
            k: spec.get(k, "-")
            for k in ("title", "why", "licence", "size", "languages",
                      "negatives", "labels", "mapping", "caveats")
        },
    )
    if spec.get("tree"):
        body += (f"\nFetched by `scripts/fetch_grounding_datasets.py {name}`. The downloaded "
                 f"data under\n`data/external/datasets/{name}/` is gitignored; this sidecar "
                 "is tracked.\n")
    else:
        body += (f"\nFetched by `scripts/fetch_grounding_datasets.py`. The archive "
                 "`dataset-{name}.zip` is\ngitignored; this sidecar is tracked.\n"
                 ).format(name=name)
    if spec.get("subsets"):
        body += "\n## Subsets fetched\n\n" + "\n".join(f"- `{s}`" for s in spec["subsets"]) + "\n"
    path = OUT / f"dataset-{name}.md"
    path.write_text(body)
    return path


def fetch_github_tabfact(spec, staging):
    """TabFact has no parquet mirror on the Hub - rebuild the statement/table
    pairs from the upstream GitHub repo exactly as the retired loading script
    did: r1+r2 collected statements joined to their #-delimited CSV tables,
    split by the official train/val/test id lists."""
    import io
    import json
    import urllib.request

    import polars as pl

    with urllib.request.urlopen(spec["github"]) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    root = z.namelist()[0].split("/")[0]

    def _read(path):
        return z.read(f"{root}/{path}")

    tables = {
        n.split("/")[-1]: z.read(n).decode("utf-8")
        for n in z.namelist()
        if "/data/all_csv/" in n and n.endswith(".csv")
    }
    statements = {}
    for part in ("collected_data/r1_training_all.json", "collected_data/r2_training_all.json"):
        for tid, (stmts, labels, caption) in json.loads(_read(part)).items():
            statements.setdefault(tid, []).extend(
                (s, int(lb), caption) for s, lb in zip(stmts, labels, strict=True)
            )

    written = 0
    counts = {}
    for split, idfile in (
        ("train", "data/train_id.json"),
        ("validation", "data/val_id.json"),
        ("test", "data/test_id.json"),
    ):
        ids = json.loads(_read(idfile))
        rows = [
            {
                "table_id": tid,
                "table_caption": cap,
                "table_text": tables[tid],
                "statement": s,
                "label": lb,
            }
            for tid in ids
            if tid in statements and tid in tables
            for (s, lb, cap) in statements[tid]
        ]
        f = staging / f"wenhuchen__Table-Fact-Checking__tabfact__{split}.parquet"
        pl.DataFrame(rows).write_parquet(f)
        counts[split] = len(rows)
        written += 1
        print(f"    tabfact/{split}: {len(rows)} rows -> {f.name}", flush=True)
    return counts if written else None


def fetch_hf(spec, staging):
    """The default Hub path: load_dataset per id/subset, one parquet per split."""
    from datasets import load_dataset

    counts = {}
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
                counts[split] = counts.get(split, 0) + len(d)
                print(f"    {tag}/{split}: {len(d)} rows -> {f.name}", flush=True)
    return counts or None


# --------------------------------------------------------------------------- #
# R19 supply-wave fetchers (raw-file pulls; load_dataset is bypassed where the
# repo ships a script loader, plain JSON/TSV, or a per-file layout)
# --------------------------------------------------------------------------- #
def _hf_file(repo_id, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="dataset")


def fetch_fava(spec, staging):
    """Raw JSON from the Hub: training.json (prompt + span-tagged completion)
    and annotations.json (the 460-passage human gold, banked alongside)."""
    import polars as pl

    counts = {}
    for fn, split in (("training.json", "train"), ("annotations.json", "gold")):
        rows = json.loads(Path(_hf_file("fava-uw/fava-data", fn)).read_text())
        f = staging / f"fava-uw__fava-data__{split}.parquet"
        pl.DataFrame(rows).write_parquet(f)
        counts[split] = len(rows)
        print(f"    fava/{split}: {len(rows)} rows -> {f.name}", flush=True)
    return counts


def fetch_pubhealth(spec, staging):
    """The HF repo ships only a loading script (datasets 5.x dropped script
    support); the script's _DATA_URL is the authors' Google Drive zip holding
    PUBHEALTH/{train,dev,test}.tsv.  Needs `uv run --with gdown`."""
    import csv
    import io

    import gdown
    import polars as pl

    url = ("https://drive.google.com/uc?export=download&id="
           "1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj")
    out = gdown.download(url, str(staging / "pubhealth.zip"), quiet=True)
    if not out:
        return None
    z = zipfile.ZipFile(out)
    counts = {}
    for split in ("train", "dev", "test"):
        text = z.read(f"PUBHEALTH/{split}.tsv").decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
        f = staging / f"pubhealth__{split}.parquet"
        pl.DataFrame(rows).write_parquet(f)
        counts[split] = len(rows)
        print(f"    pubhealth/{split}: {len(rows)} rows -> {f.name}", flush=True)
    (staging / "pubhealth.zip").unlink()
    return counts


def fetch_minicheck(spec, staging):
    """Two parquet files (c2d, d2c) under data/ in the Hub repo; filenames are
    resolved live so a re-upload with new content hashes still fetches."""
    from huggingface_hub import HfApi
    import polars as pl

    files = HfApi().list_repo_files("lytang/C2D-and-D2C-MiniCheck", repo_type="dataset")
    counts = {}
    for split in ("c2d", "d2c"):
        fn = next(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
        df = pl.read_parquet(_hf_file("lytang/C2D-and-D2C-MiniCheck", fn))
        f = staging / f"lytang__C2D-and-D2C-MiniCheck__{split}.parquet"
        df.write_parquet(f)
        counts[split] = df.height
        print(f"    minicheck/{split}: {df.height} rows -> {f.name}", flush=True)
    return counts


# Walled RAGBench source corpora inside AttributionBench - dropped BY
# CONSTRUCTION at fetch (precedent: the ragbench fetch excludes MS MARCO/CUAD).
AB_WALLED = {"ExpertQA", "HAGRID"}
AB_FULL_DATA_FILES = {  # the card YAML's full_data config
    "train": "merged_train.jsonl",
    "dev": "dev_all_subset_balanced.jsonl",
    "test": "test_all_subset_balanced.jsonl",
    "test_ood": "test_ood_all_subset_balanced.jsonl",
}


def fetch_attributionbench(spec, staging):
    """full_data config with the carve-out applied at fetch: the archive holds
    only the kept frame; dropped counts go to _counts.json and the sidecar."""
    import polars as pl

    counts, dropped = {}, {}
    for split, fn in AB_FULL_DATA_FILES.items():
        rows = [json.loads(ln) for ln in open(_hf_file("osunlp/AttributionBench", fn))]
        kept = [r for r in rows if r["src_dataset"] not in AB_WALLED]
        dropped[split] = len(rows) - len(kept)
        for r in kept:  # normalize optional list fields for a stable parquet schema
            for k in ("references", "citation_links", "webpage_references"):
                r[k] = r.get(k) or []
        f = staging / f"osunlp__AttributionBench__{split}.parquet"
        pl.DataFrame(kept).write_parquet(f)
        counts[split] = len(kept)
        print(f"    attributionbench/{split}: {len(kept)} kept of {len(rows)} "
              f"(dropped {dropped[split]} walled) -> {f.name}", flush=True)
    counts["_dropped_walled_by_split"] = dropped
    counts["_dropped_walled_total"] = sum(dropped.values())
    return counts


def _drive_folder_file_ids(folder_id):
    """file name -> id map of a public Drive folder via embeddedfolderview."""
    import re

    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    html = urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "ignore")
    entries = re.findall(r'flip-entry" id="entry-([\w-]+)".*?flip-entry-title">([^<]+)<',
                         html, re.DOTALL)
    return {name: fid for fid, name in entries}


def fetch_factscore(spec, tree):
    """Tree corpus (no archive): the authors' Drive `data.zip` (labeled human
    judgments + prompt entities) plus per-topic Wikipedia evidence pinned to
    the last revision on or before 2023-04-01, the corpus's knowledge source."""
    import html as htmlmod
    import re
    import time
    import urllib.parse

    import gdown
    import polars as pl

    ids = _drive_folder_file_ids("1kFey69z8hGXScln01mVxrOhrqgM62X7I")
    out = gdown.download(id=ids["data.zip"], output=str(tree / "data.zip"), quiet=True)
    if not out:
        return None
    z = zipfile.ZipFile(out)
    counts = {}
    for member in z.namelist():
        if member.startswith("data/labeled/") and not member.endswith("/"):
            target = tree / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
    (tree / "data.zip").unlink()
    # the licence lives in the code repo, not the data zip - capture it into
    # the tree so the re-verification evidence travels with the data
    req = urllib.request.Request(
        "https://raw.githubusercontent.com/shmsw25/FActScore/main/LICENSE",
        headers={"User-Agent": "groundrails-research/1.0"})
    (tree / "LICENSE").write_bytes(urllib.request.urlopen(req, timeout=60).read())
    for model in ("InstructGPT", "ChatGPT", "PerplexityAI"):
        rows = [json.loads(ln) for ln in open(tree / "data" / "labeled" / f"{model}.jsonl")]
        counts[model] = len(rows)
        print(f"    factscore/{model}: {len(rows)} biographies", flush=True)

    topics = (tree / "data" / "labeled" / "prompt_entities.txt").read_text().strip().split("\n")
    counts["entities"] = len(topics)

    # Wikipedia evidence, pinned revisions (batch the revid lookup 50 at a time)
    API = "https://en.wikipedia.org/w/api.php"
    UA = {"User-Agent": "groundrails-research/1.0 (R19 supply build; dataset fetch)"}

    def _get(params):
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=UA)
        return json.loads(urllib.request.urlopen(req, timeout=90).read())

    def _get_retry(params, attempts=4):
        """429/5xx-tolerant GET honoring the server's Retry-After header - a
        burst cadence puts this box in the Wikimedia penalty box (observed
        2026-08-13: 429 + Retry-After: 20), so back off as instructed."""
        for k in range(attempts):
            try:
                return _get(params)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503) or k == attempts - 1:
                    raise
                wait = e.headers.get("Retry-After")
                time.sleep(min(float(wait), 60.0) if wait else 5.0 * (k + 1))

    revid = {}

    def _lookup(titles):
        """oldid of the last revision <= 2023-04-01 for each title in `titles`."""
        r = _get_retry({"action": "query", "format": "json", "prop": "revisions",
                        "titles": "|".join(titles), "rvstart": "2023-04-01T00:00:00Z",
                        "rvdir": "older", "rvlimit": 1, "rvprop": "ids|timestamp",
                        "redirects": 1})
        if "query" not in r:
            raise RuntimeError(f"API error response: {str(r)[:160]}")
        alias = {}
        for n in r["query"].get("normalized", []):
            alias[n["from"]] = n["to"]
        for rd in r["query"].get("redirects", []):
            alias[rd["from"]] = rd["to"]

        def resolve(t):
            seen = set()
            while t in alias and t not in seen:
                seen.add(t)
                t = alias[t]
            return t

        by_title = {resolve(t): t for t in titles}
        for page in r["query"]["pages"].values():
            t = by_title.get(page.get("title"))
            if t and "revisions" in page:
                revid[t] = {"oldid": page["revisions"][0]["revid"],
                            "timestamp": page["revisions"][0]["timestamp"],
                            "title": page["title"]}

    failed = []
    # per-title lookups only: the MediaWiki revisions endpoint rejects
    # multi-title queries combined with rvstart/rvlimit (invalidparammix), so
    # there is no batch route - 183 single requests at a polite spacing
    for i, t in enumerate(topics):
        try:
            _lookup([t])
        except urllib.error.HTTPError as e:
            failed.append(f"{t} (lookup: HTTP {e.code})")
        except Exception as e:  # noqa: BLE001 - recorded, not repaired
            failed.append(f"{t} (lookup: {type(e).__name__}: {str(e)[:80]})")
        if i % 25 == 0:
            print(f"    factscore revids: {i}/{len(topics)}", flush=True)
        time.sleep(1.0)

    evdir = tree / "evidence"
    evdir.mkdir(exist_ok=True)
    index = {}
    for i, t in enumerate(topics):
        info = revid.get(t)
        if not info:
            failed.append(t)
            continue
        url = (f"https://en.wikipedia.org/api/rest_v1/page/html/"
               f"{urllib.parse.quote(info['title'])}/{info['oldid']}")
        try:
            htmltxt = None
            for k in range(4):
                try:
                    req = urllib.request.Request(url, headers=UA)
                    htmltxt = urllib.request.urlopen(req, timeout=90).read().decode(
                        "utf-8", "ignore")
                    break
                except urllib.error.HTTPError as e:
                    if e.code not in (429, 500, 502, 503) or k == 3:
                        raise
                    wait = e.headers.get("Retry-After")
                    time.sleep(min(float(wait), 60.0) if wait else 5.0 * (k + 1))
            htmltxt = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", htmltxt)
            text = htmlmod.unescape(re.sub(r"<[^>]+>", " ", htmltxt))
            text = re.sub(r"\s+", " ", text).strip()
            fn = re.sub(r"[^A-Za-z0-9_.-]+", "_", t) + ".txt"
            (evdir / fn).write_text(text)
            index[t] = {**info, "file": fn, "chars": len(text)}
        except Exception as e:  # noqa: BLE001 - recorded, not repaired
            if isinstance(e, urllib.error.HTTPError):
                failed.append(f"{t} (evidence: HTTP {e.code})")
            else:
                failed.append(f"{t} (evidence: {type(e).__name__}: {str(e)[:80]})")
        if i % 25 == 0:
            print(f"    factscore evidence: {i}/{len(topics)} topics", flush=True)
        time.sleep(1.0)
    (evdir / "_index.json").write_text(json.dumps(index, indent=2))
    counts["_evidence_topics"] = len(index)
    counts["_evidence_failed"] = failed
    print(f"    factscore evidence: {len(index)} topics pinned, {len(failed)} failed",
          flush=True)
    return counts


def fetch_findver(spec, tree):
    """GitHub tree (no archive): LICENSE, README, data/*.json claims and the
    financial_reports/*.json 2024 filing extracts."""
    import io

    import polars as pl

    url = "https://codeload.github.com/yilunzhao/FinDVer/zip/refs/heads/main"
    req = urllib.request.Request(url, headers={"User-Agent": "groundrails-research/1.0"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    root = z.namelist()[0].split("/")[0]
    keep_prefixes = ("LICENSE", "README.md", "data/", "financial_reports/")
    n_files = 0
    for member in z.namelist():
        rel = member[len(root) + 1 :] if member.startswith(root + "/") else member
        if not rel or member.endswith("/"):
            continue
        if any(rel == p or rel.startswith(p) for p in keep_prefixes):
            target = tree / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
            n_files += 1
    counts = {"_files": n_files}
    for split in ("test", "testmini"):
        rows = json.loads((tree / "data" / f"{split}.json").read_text())
        counts[split] = len(rows)
        print(f"    findver/{split}: {len(rows)} claims", flush=True)
    counts["_reports"] = len(list((tree / "financial_reports").glob("*.json")))
    return counts


FETCHERS = {
    "fava": fetch_fava,
    "pubhealth": fetch_pubhealth,
    "minicheck": fetch_minicheck,
    "attributionbench": fetch_attributionbench,
    "factscore": fetch_factscore,
    "findver": fetch_findver,
}


def fetch(name, spec):
    """Stage one corpus.  Returns (kind, counts): kind is 'staged' (staging dir
    ready to zip), 'tree' (extracted tree already on disk) or None (failed)."""
    staging = OUT / f"_staging_{name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if spec.get("tree"):
        tree = OUT / name
        if tree.exists():
            shutil.rmtree(tree)
        tree.mkdir(parents=True)
        try:
            counts = FETCHERS[spec["fetcher"]](spec, tree)
        except Exception as e:  # noqa: BLE001 - a failed corpus is a result, not a crash
            print(f"    SKIP {name}: {type(e).__name__}: {str(e)[:110]}", flush=True)
            counts = None
        shutil.rmtree(staging)
        if not counts:
            shutil.rmtree(tree, ignore_errors=True)
            return None, None
        (tree / "_counts.json").write_text(json.dumps(
            {"counts": counts, "fetched_utc": "2026-08-13", "source": spec.get("source", "")},
            indent=2))
        return "tree", counts

    try:
        if spec.get("fetcher"):
            counts = FETCHERS[spec["fetcher"]](spec, staging)
        elif spec.get("github"):
            counts = fetch_github_tabfact(spec, staging)
        else:
            counts = fetch_hf(spec, staging)
    except Exception as e:  # noqa: BLE001 - a failed corpus is a result, not a crash
        print(f"    SKIP {name}: {type(e).__name__}: {str(e)[:110]}", flush=True)
        counts = None
    if not counts:
        shutil.rmtree(staging)
        return None, None
    return "staged", counts


def finalize_zip(name, spec, counts):
    """Archive staging + the FINAL sidecar (post-fetch render) + _counts.json."""
    staging = OUT / f"_staging_{name}"
    (staging / "_counts.json").write_text(json.dumps(
        {"counts": counts, "fetched_utc": "2026-08-13",
         "source": spec.get("source") or ", ".join(spec.get("hf", []))}, indent=2))
    archive = OUT / f"dataset-{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.iterdir()):
            z.write(f, f.name)
        z.write(OUT / f"dataset-{name}.md", f"dataset-{name}.md")
    shutil.rmtree(staging)
    return archive


def _recorded_counts(name, spec):
    """Counts from an earlier completed fetch, for idempotent re-renders."""
    try:
        if spec.get("tree"):
            return json.loads((OUT / name / "_counts.json").read_text())["counts"]
        with zipfile.ZipFile(OUT / f"dataset-{name}.zip") as z:
            return json.loads(z.read("_counts.json"))["counts"]
    except Exception:  # noqa: BLE001 - absent counts just skip the suffix
        return None


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
        done = (OUT / name / "_counts.json") if spec.get("tree") else OUT / f"dataset-{name}.zip"
        if not dry and done.exists():
            sc = write_sidecar(name, spec, _recorded_counts(name, spec))
            print(f"  already fetched (checkpoint found); sidecar re-rendered -> {sc.name}",
                  flush=True)
            continue
        sc = write_sidecar(name, spec)
        print(f"  sidecar -> {sc.name}", flush=True)
        if dry:
            continue
        kind, counts = fetch(name, spec)
        if not kind:
            print("  archive -> NONE (every split failed; see SKIP lines)", flush=True)
            continue
        sc = write_sidecar(name, spec, counts)  # post-fetch re-render with real counts
        if kind == "staged":
            archive = finalize_zip(name, spec, counts)
            mb = archive.stat().st_size / 1e6
            print(f"  archive -> {archive.name} ({mb:.1f} MB); sidecar re-rendered", flush=True)
        else:
            print(f"  tree -> data/external/datasets/{name}/; sidecar re-rendered", flush=True)

    print(f"\nsidecars and archives in {OUT}")
    print("archives and trees are gitignored; sidecars are tracked")


if __name__ == "__main__":
    main()
