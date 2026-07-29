"""R6-H42 - Pleias-RAG-350M under its own published protocol.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 6).

Every constant here is transcribed from the vendor's own inference library,
`Pleias/Pleias-RAG-Library` -> `pleias_rag_interface/RAGWithCitations.py`, not
inferred. The round-4 failures all came from guessing a format; this file
reproduces the published one and proves it BEFORE any grounding number is taken.

Verbatim from the library (prompt assembly, lines 165-173):

    prompt  = f"<|query_start|>{query}<|query_end|>\\n"
    prompt += f"<|source_start|><|source_id|>{idx} {source_text}<|source_end|>\\n"   # per source, idx from 1
    prompt += "<|language_start|>\\n"

Verbatim from the library (transformers backend, lines 212-219 + defaults 10-13):

    max_new_tokens=2048, temperature=None, top_p=0.95, repetition_penalty=1.0,
    do_sample=False, pad_token_id=tokenizer.eos_token_id
    decode(..., skip_special_tokens=False)

Model and tokenizer are loaded exactly as the library does: no dtype override,
no trust_remote_code, `device_map="auto"`.

Three stages, each gating the next:

  stage 1  REPRODUCTION - run the model card's own worked example and check the
           published answer comes back. If this fails, nothing downstream means
           anything and the run stops.
  stage 2  R6-H42 - the unchanged 20-case trivial control from R4-H29, so the
           number is directly comparable to Baguettotron 12/20.
  stage 3  raw dumps retained for the parser work in R6-H43/H44/H45.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R6-H42_pleias_rag_protocol.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json
import pathlib
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "PleIAs/Pleias-RAG-350M"
OUT = pathlib.Path(__file__).parent / "R6-H42_generations.json"

# Library defaults, RAGWithCitations.__init__ lines 10-13.
MAX_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY = 2048, 0.0, 0.95, 1.0

# Library section patterns, lines 288-294.
SECTION_PATTERNS = {
    "query_analysis": r"<\|query_analysis_start\|>(.*?)<\|query_analysis_end\|>",
    "query_report": r"<\|query_report_start\|>(.*?)<\|query_report_end\|>",
    "source_analysis": r"<\|source_analysis_start\|>(.*?)<\|source_analysis_end\|>",
    "source_report": r"<\|source_report_start\|>(.*?)<\|source_report_end\|>",
    "draft": r"<\|draft_start\|>(.*?)<\|draft_end\|>",
    "answer": r"<\|answer_start\|>(.*?)<\|answer_end\|>",
}

# The model card's worked example, verbatim - stage 1 must reproduce it.
CARD_QUERY = "What is the capital of France?"
CARD_SOURCES = [
    (
        "Paris is the capital and most populous city of France. With an estimated population of "
        "2,140,526 residents as of January 2019, Paris is the center of the Île-de-France "
        "metropolitan area and the hub of French economic, political, and cultural life. The city's "
        "landmarks, including the Eiffel Tower, Arc de Triomphe, and Cathedral of Notre-Dame, make "
        "it one of the world's most visited tourist destinations."
    ),
    (
        "The Eiffel Tower is located in Paris, France. It was constructed from 1887 to 1889 as the "
        "entrance to the 1889 World's Fair and was initially criticized by some of France's leading "
        "artists and intellectuals for its design. Standing at 324 meters (1,063 ft) tall, it was "
        "the tallest man-made structure in the world until the completion of the Chrysler Building "
        "in New York City in 1930. The tower receives about 7 million visitors annually and has "
        "become an iconic symbol of Paris and France."
    ),
]

# Stage 2 - the R4-H29 control, unchanged. Ten claims quoted verbatim inside
# their own chunk, then the same ten against one unrelated recipe chunk.
RECIPE = (
    "A recipe for sourdough bread requires flour, water, salt and a starter culture. "
    "Mix the flour and water and rest for one hour before adding the starter. Bulk "
    "ferment for four to six hours with folds every thirty minutes, then shape and "
    "retard the dough overnight in the refrigerator before baking in a hot dutch oven."
)
CONTROL = [
    ("The dam is 271 metres tall.", "The dam is 271 metres tall and produces 1300 MW of power."),
    ("The tower stands 324 metres tall.", "The tower stands 324 metres tall and opened in 1889."),
    (
        "The survey covered 39 studies.",
        "The survey covered 39 studies across 58 modelled outputs.",
    ),
    ("The dataset spans 964 sensors.", "The dataset spans 964 sensors read every four hours."),
    ("Crack IoU reached 66.76 percent.", "Crack IoU reached 66.76 percent on the test set."),
    ("The reservoir swing is 100 metres.", "The reservoir swing is 100 metres over a season."),
    (
        "Flights were flown at 70 metres.",
        "Flights were flown at 70 metres at 15 m/s ground speed.",
    ),
    (
        "The arch dam is 292 metres high.",
        "The arch dam is 292 metres high with a crest at 1245 m.",
    ),
    ("Training took under six hours.", "Training took under six hours on 16 H100 accelerators."),
    (
        "The model has 56.7 million parameters.",
        "The model has 56.7 million parameters in 64 layers.",
    ),
]


def build_prompt(query, sources):
    """Verbatim from RAGWithCitations.format_prompt, lines 165-173."""
    prompt = f"<|query_start|>{query}<|query_end|>\n"
    for idx, source_text in enumerate(sources, 1):
        prompt += f"<|source_start|><|source_id|>{idx} {source_text}<|source_end|>\n"
    prompt += "<|language_start|>\n"
    return prompt


def parse_sections(text):
    """Verbatim from RAGWithCitations.parse_output, lines 280-306."""
    result = {}
    m = re.search(r"<\|language_end\|>", text, re.DOTALL)
    if m:
        result["language"] = text[: m.start()].strip()
    for name, pattern in SECTION_PATTERNS.items():
        hit = re.search(pattern, text, re.DOTALL)
        if hit:
            result[name] = hit.group(1).strip()
    if not result:
        result["full_text"] = text
    return result


class Runner:
    def __init__(self):
        # Loaded as the library's _init_transformers does: no dtype override,
        # no trust_remote_code. ONE documented deviation - the library passes
        # device_map="auto", which needs `accelerate`; with a single visible
        # GPU that resolves to placing every module on cuda:0, which .cuda()
        # does directly. Placement only, no numerics touched.
        self.tok = AutoTokenizer.from_pretrained(MID)
        self.model = AutoModelForCausalLM.from_pretrained(MID).cuda()
        self.model.eval()
        print(
            f"loaded {MID}  dtype={next(self.model.parameters()).dtype} "
            f"device={next(self.model.parameters()).device}",
            flush=True,
        )

    @torch.inference_mode()
    def generate(self, prompt, max_new=MAX_TOKENS):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=TEMPERATURE > 0,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0][ids.shape[1] :], skip_special_tokens=False)


def stage1_reproduction(run):
    """The model card's own example. A harness that cannot reproduce the
    published answer has no standing to report a null on our data."""
    print("=" * 96, flush=True)
    print("STAGE 1 - reproduce the model card's worked example", flush=True)
    text = run.generate(build_prompt(CARD_QUERY, CARD_SOURCES))
    sec = parse_sections(text)
    answer = sec.get("answer", "")
    print(f"  sections parsed: {sorted(sec)}", flush=True)
    print(f"  ANSWER: {answer[:400]}", flush=True)
    print(f"  source_analysis: {sec.get('source_analysis', '')[:300]}", flush=True)
    ok = "paris" in answer.lower()
    cited = bool(re.search(r'<ref name="(?:<\|source_id\|>)?\d+">', answer))
    print(f"  contains the published answer ('Paris'): {ok}", flush=True)
    print(f"  emitted an inline <ref> citation: {cited}", flush=True)
    print(f"  RAW:\n{text[:1500]}\n", flush=True)
    return ok, cited, text


def stage2_control(run):
    print("=" * 96, flush=True)
    print("STAGE 2 - R6-H42 on the unchanged R4-H29 20-case control", flush=True)
    rows, correct, fp, fn, unparsed = [], 0, 0, 0, 0
    cases = [(c, ev, True) for c, ev in CONTROL] + [(c, RECIPE, False) for c, _ in CONTROL]
    for i, (claim, evidence, expect) in enumerate(cases):
        q = f"Is this claim supported by the source? Claim: {claim}"
        text = run.generate(build_prompt(q, [evidence]), max_new=600)
        sec = parse_sections(text)
        answer = sec.get("answer", "")
        analysis = sec.get("source_analysis", "")
        got = None
        if answer:
            low = answer.lower()
            neg = any(
                p in low
                for p in (
                    "not supported",
                    "does not",
                    "no information",
                    "cannot",
                    "unsupported",
                    "not mentioned",
                    "not contain",
                    "unrelated",
                    "no mention",
                )
            )
            got = not neg
        if got is None:
            unparsed += 1
        elif got == expect:
            correct += 1
        elif got and not expect:
            fp += 1
        else:
            fn += 1
        rows.append(
            {
                "claim": claim,
                "expect": expect,
                "got": got,
                "answer": answer[:300],
                "source_analysis": analysis[:300],
                "raw": text[:2000],
            }
        )
        print(
            f"  [{i:2d}] expect={'SUP' if expect else 'UNSUP':5s} got={got}  {answer[:110]!r}",
            flush=True,
        )
    n = len(cases)
    print(f"\n  parseable {n - unparsed}/{n} ({(n - unparsed) / n:.0%})", flush=True)
    print(f"  correct   {correct}/{n} ({correct / n:.0%})   FP {fp}   FN {fn}", flush=True)
    print(
        f"  bar: >= 90% correct AND FP <= 1  ->  "
        f"{'PASS' if correct / n >= 0.90 and fp <= 1 else 'FAIL'}",
        flush=True,
    )
    print("  reference: R4-H29 Baguettotron 12/20 (60%), 0 FN / 4 FP", flush=True)
    return rows, correct, fp, fn, unparsed


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    run = Runner()
    ok, cited, card_raw = stage1_reproduction(run)
    if not ok:
        print("\nSTAGE 1 FAILED - the harness does not reproduce the published example.")
        print("Stopping: no grounding number taken under an unvalidated harness.")
        OUT.write_text(json.dumps({"stage1_ok": False, "card_raw": card_raw}, indent=2))
        return
    rows, correct, fp, fn, unparsed = stage2_control(run)
    OUT.write_text(
        json.dumps(
            {
                "stage1_ok": ok,
                "stage1_cited": cited,
                "card_raw": card_raw,
                "control": rows,
                "correct": correct,
                "fp": fp,
                "fn": fn,
                "unparsed": unparsed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\ngenerations -> {OUT}")


if __name__ == "__main__":
    main()
