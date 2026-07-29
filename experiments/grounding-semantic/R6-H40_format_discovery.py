"""R6-H40 / R6-H42 pre-experiment probe - discover the real prompt format.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 6).

No scorer is written against an assumed output shape. Four parser and prompt
defects in round 4 all came from building against a guess. This dumps RAW
generations for every candidate format so the parser is written against what was
observed, and nothing here scores anything.

  Monad            - its only in-vocabulary control tokens are <|begin_of_text|>
                     and <|end_of_text|>, so three plain-text shapes are tried
  Pleias-RAG-350M  - its 19 protocol tokens ARE the format; two assemblies are
                     tried, differing in where generation is handed to the model

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R6-H40_format_discovery.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CLAIM = "The dam is 271 metres tall."
POS = "The dam is 271 metres tall and produces 1300 MW of power."
NEG = "A recipe for sourdough bread requires flour, water, salt and a starter culture."

MONAD_FORMATS = {
    "plain-qa": (
        "Evidence: {ev}\n\nClaim: {cl}\n\n"
        "Question: Is the claim supported by the evidence? Answer yes or no.\nAnswer:"
    ),
    "sectioned": (
        "### Evidence\n{ev}\n\n### Claim\n{cl}\n\n"
        "### Question\nIs the claim supported by the evidence?\n\n### Answer\n"
    ),
    "cloze": "Evidence: {ev}\nClaim: {cl}\nThe claim is",
}


def rag_prompt(claim, evidence, hand_off):
    """Assemble Pleias-RAG's native protocol; `hand_off` is where we stop."""
    p = (
        f"<|query_start|>Is this claim supported by the source: {claim}<|query_end|>"
        f"<|source_start|><|source_id|>1 {evidence}<|source_end|>"
    )
    return p + hand_off


RAG_FORMATS = {
    "hand-off-at-analysis": "<|source_analysis_start|>",
    "hand-off-after-sources": "",
}


@torch.inference_mode()
def dump(mid, prompts, max_new=320):
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    pad = tok.convert_tokens_to_ids("[PAD]")
    print("#" * 100, flush=True)
    print(f"# {mid}", flush=True)
    for label, prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        gen = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=False)
        print("=" * 100, flush=True)
        print(f"[{label}]  prompt {enc.input_ids.shape[1]} tok -> "
              f"generated {out.shape[1] - enc.input_ids.shape[1]} tok", flush=True)
        print(f"PROMPT TAIL: ...{prompt[-160:]!r}", flush=True)
        print(f"RAW:\n{gen}\n", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    monad = [
        (f"{name}/{pol}", tmpl.format(ev=ev, cl=CLAIM))
        for name, tmpl in MONAD_FORMATS.items()
        for pol, ev in (("SUPPORTED", POS), ("UNSUPPORTED", NEG))
    ]
    dump("PleIAs/Monad", monad)

    rag = [
        (f"{name}/{pol}", rag_prompt(CLAIM, ev, ho))
        for name, ho in RAG_FORMATS.items()
        for pol, ev in (("SUPPORTED", POS), ("UNSUPPORTED", NEG))
    ]
    dump("PleIAs/Pleias-RAG-350M", rag)


if __name__ == "__main__":
    main()
